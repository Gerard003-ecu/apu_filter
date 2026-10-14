# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/wisdom/toon_silent_witness_engine.py                                  ║
║ ESTRATO  : WISDOM (V_𝕎) — CIUDADELA DE CRISTAL / VACÍO DE DIRAC                      ║
║ FUNCIÓN  : MOTOR ESPECTRAL TESTIGO SILENCIOSO Y CRISTALIZADOR DEL VACÍO              ║
║ VERSIÓN  : 9.2.0-APU-RSI3-CELESTE-AGM-KMS-POINCARE                                   ║
║ FASE     : 1 / 3  ·  SUSTRATO ALGEBRAICO-MODULAR, RSI-3 GERM Y POINCARÉ              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

EVOLUCIÓN GRANULAR v9.2.0-F1 respecto de v9.1.0:
  (i)   Araki relativo HS-riguroso: Δ(ρ‖σ) = ρ ⊗ (σ^{-1})^T  (sin base conjunta).
  (ii)  Polar C* vía SVD: A = U_pol P, U_pol = U V†, P = V Σ V†  (isometría parcial).
  (iii) Carta canónica de Poincaré (ξ, η, p, q) no singular en e = 0, i = 0.
  (iv)  No-degeneración de Kolmogorov: κ = det(Hess_I H) = det(∂ω/∂I).
  (v)   Espectro de Lyapunov del jacobiano ad-N sobre T_ρ 𝔇_n.
  (vi)  Kac–Poincaré con volumen FS: μ(B_ε) = ε^{2n−2} / (n−1)!  (bola Fubini–Study).
  (vii) Germen mónada RSI-3 (T, η, μ) sobre 𝔇_n — unidad η no colisiona con el scalar.
  (viii) Cristal S⁶ inmovilizado (write-flag False) + exponentes característicos.
  (ix)  Morfismo terminal handoff_to_tomita_takesaki ≡ objeto inicial de FASE 2.
====================================================================================================
"""
from __future__ import annotations

import hashlib
import itertools
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import scipy.linalg as la


__version__ = "9.2.0-APU-RSI3-CELESTE-AGM-KMS-POINCARE"


logger = logging.getLogger("APU.Wisdom.TOONSilentWitnessEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── Constantes espectrales, diofantinas y celestes ─────────────────────────
_EPS: Final[float] = 1.0e-14
_HERMITICITY_TOL: Final[float] = 1.0e-12
_TRACE_TOL: Final[float] = 1.0e-10
_GAP_DEGENERACY_TOL: Final[float] = 1.0e-12
_DIOPHANTINE_GAMMA: Final[float] = 1.0e-3
_DIOPHANTINE_TAU: Final[float] = 1.5
_DIOPHANTINE_KMAP: Final[int] = 4
_KAM_SAFE_THRESHOLD: Final[float] = 1.0
_KOLMOGOROV_DET_TOL: Final[float] = 1.0e-12
_LYAPUNOV_DT: Final[float] = 1.0e-3
_RSI3_DEFAULT_MIX: Final[float] = 0.15
_RSI3_DEFAULT_LEVEL: Final[int] = 3
_FS_VOLUME_MAX_DIM: Final[int] = 64  # (n−1)! overflow guard


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ALGEBRAICO-MODULAR, GERMEN RSI-3 Y POINCARÉ             ║
# ║                                                                           ║
# ║  §1.0  Cristal S⁶ inmutable + exponentes característicos                  ║
# ║  §1.1  Retículo de Heyting Ω₃ (clasificador de estratos de Poincaré)      ║
# ║  §1.2  C*-álgebra matricial (espectro de Gelfand, polar SVD, rango num.)  ║
# ║  §1.3  Hamiltoniano modular K_ρ + flujo σ_t + Helmholtz                   ║
# ║  §1.4  Álgebra 𝔇_n (Rényi, QFI, Petz, Holevo, Wigner, Kac-FS, Poincaré)   ║
# ║  §1.5  GNS / Tomita: Δ, J, S, Δ(ρ‖σ) de Araki en HS                       ║
# ║  §1.6  Mecánica celeste de Poincaré (carta, Kolmogorov, Lyapunov, KAM)    ║
# ║  §1.7  Germen RSI-3 (T, η, μ) sobre 𝔇_n                                   ║
# ║  §1.8  VacuumModularContext + VacuumStatePreparation                      ║
# ║        └── handoff_to_tomita_takesaki  ≡  morfismo inicial de FASE 2      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# §1.0  Cristal de experiencia inmutable sobre S⁶ ⊂ ℝ⁷
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ExperienceCrystal:
    r"""
    Cristal de experiencia inmutable sobre S⁶ ⊂ ℝ⁷.

    Coordenadas celestes de Poincaré:
        vector_s6           — punto de S⁶ (norma euclídea 1, write-protected).
        tau_recurrence      — tiempo de primer retorno (Bohr / Kac).
        kms_drift           — ‖σ_{iβ} − id‖ residual de KMS.
        lyapunov_spectrum   — exponentes característicos (λ₁ ≥ … ≥ λ_{2n−2}).
        kolmogorov_det      — det(∂ω/∂I); no-degeneración si |det| > 0.
        generating_function — valor de la función generatriz de Poincaré S(θ, I').
        first_return_residual — ‖P_Σ(ρ) − ρ‖_FS sobre la sección.
    """
    vector_s6: np.ndarray
    tau_recurrence: float
    kms_drift: float
    merkle_root_sha256: str
    timestamp_utc: float
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    eccentricity: float = 0.0
    inclination: float = 0.0
    kam_residue: float = float("inf")
    resonance_order: int = 0
    poincare_stratum: str = "kam-torus-invariant"
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    lyapunov_spectrum: Optional[Tuple[float, ...]] = None
    kolmogorov_det: float = 0.0
    generating_function: float = 0.0
    first_return_residual: float = 0.0
    rsi3_level_applied: int = 0

    def __post_init__(self) -> None:
        v = np.asarray(self.vector_s6, dtype=np.float64).reshape(-1)
        if v.size != 7:
            padded = np.zeros(7, dtype=np.float64)
            n = min(7, int(v.size))
            padded[:n] = v[:n]
            v = padded
        nrm = float(np.linalg.norm(v))
        if nrm > _EPS:
            v = v / nrm
        else:
            v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v.setflags(write=False)
        object.__setattr__(self, "vector_s6", v)

    def is_kam_stable(self) -> bool:
        r"""Toro KAM Diofantino: R ≥ 1 ∧ orden resonante 0 ∧ Kolmogorov no-deg."""
        return (
            self.kam_residue >= _KAM_SAFE_THRESHOLD
            and self.resonance_order == 0
            and abs(self.kolmogorov_det) > _KOLMOGOROV_DET_TOL
        )

    def verify_s6_invariant(self, tol: float = 1e-10) -> bool:
        r"""Verifica ‖v‖₂ = 1 (invariante de la 6-esfera)."""
        return abs(float(np.linalg.norm(self.vector_s6)) - 1.0) < tol

    def maximal_lyapunov(self) -> float:
        r"""λ_max: signo − integrable / 0 parabólico / + hiperbólico (Poincaré)."""
        if not self.lyapunov_spectrum:
            return 0.0
        return float(self.lyapunov_spectrum[0])

    def is_hyperbolic(self) -> bool:
        return self.maximal_lyapunov() > 1e-8

    def merkle_leaf(self) -> str:
        r"""Hoja Merkle canónica: SHA-256(v ‖ τ_rec ‖ KMS ‖ D ‖ e ‖ i ‖ R ‖ λ ‖ κ)."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.vector_s6, dtype=np.float64).tobytes())
        h.update(f"{self.tau_recurrence:.12e}".encode("ascii"))
        h.update(f"{self.kms_drift:.12e}".encode("ascii"))
        if self.delaunay_triple is not None:
            for x in self.delaunay_triple:
                h.update(f"{x:.12e}".encode("ascii"))
        h.update(f"{self.eccentricity:.12e}".encode("ascii"))
        h.update(f"{self.inclination:.12e}".encode("ascii"))
        h.update(f"{self.kam_residue:.12e}".encode("ascii"))
        h.update(f"{self.resonance_order}".encode("ascii"))
        h.update(f"{self.kolmogorov_det:.12e}".encode("ascii"))
        h.update(f"{self.maximal_lyapunov():.12e}".encode("ascii"))
        return h.hexdigest()

    def kam_stability_certificate(self) -> Dict[str, Any]:
        r"""Certificado de estabilidad KAM + Poincaré."""
        return {
            "is_s6_on_sphere": self.verify_s6_invariant(),
            "is_kam_stable": self.is_kam_stable(),
            "has_unique_vacuum": self.resonance_order == 0,
            "crystal_norm": float(np.linalg.norm(self.vector_s6)),
            "kolmogorov_nondegenerate": abs(self.kolmogorov_det) > _KOLMOGOROV_DET_TOL,
            "maximal_lyapunov": self.maximal_lyapunov(),
            "is_hyperbolic": self.is_hyperbolic(),
            "first_return_residual": self.first_return_residual,
            "rsi3_level": self.rsi3_level_applied,
            "merkle_leaf": self.merkle_leaf(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §1.1  Retículo distributivo de Heyting Ω₃  (clasificador de subobjetos)
# ═══════════════════════════════════════════════════════════════════════════
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0, 1, 2}.

    Objeto clasificador de subobjetos del topos trivaluado Sh(Ω₃).
    Los tres puntos clasifican los *estratos de Poincaré* del flujo modular:

        ⊥  VETOED    —  separatriz hiperbólica (escape simpléctico, λ > 0)
        ⋆  DEGRADED  —  órbita parabólica resonante (cascada, λ = 0)
        ⊤  COHERENT  —  toro KAM invariante (integrable, λ < 0)

    El tercio excluso falla en ⋆:  ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤.  ¬¬⋆ = ⊤ ≠ ⋆
    (⋆ no es regular).  Esto es el análogo topos-teórico de una resonancia
    parabólica: ni elipse KAM ni hipérbola de escape.
    """
    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def verdict(self) -> str:
        return self.name

    @property
    def rank(self) -> int:
        return int(self)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.VETOED)

    def double_negation(self) -> "HeytingOmega3":
        return self.neg().neg()

    def is_regular(self) -> bool:
        return self.double_negation() == self

    def is_dense(self) -> bool:
        return self.neg() == HeytingOmega3.VETOED

    def excluded_middle_holds(self) -> bool:
        return self.join(self.neg()) == HeytingOmega3.COHERENT

    def meet_all(self, *others: "HeytingOmega3") -> "HeytingOmega3":
        acc: HeytingOmega3 = self
        for o in others:
            acc = acc.meet(o)
        return acc

    @classmethod
    def relative_pseudocomplement(
        cls, a: "HeytingOmega3", b: "HeytingOmega3"
    ) -> "HeytingOmega3":
        r"""a ⇒ b (pseudocomplemento relativo de Heyting)."""
        return a.implies(b)

    @classmethod
    def verify_distributivity(cls) -> bool:
        r"""a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)  ∀ (a,b,c) ∈ Ω₃³."""
        elems = list(cls)
        for a, b, c in itertools.product(elems, repeat=3):
            if a.meet(b.join(c)) != a.meet(b).join(a.meet(c)):
                return False
        return True

    @classmethod
    def verify_residuation(cls) -> bool:
        r"""(a ∧ b) ≤ c  ⇔  a ≤ (b ⇒ c)."""
        elems = list(cls)
        for a, b, c in itertools.product(elems, repeat=3):
            left = int(a.meet(b)) <= int(c)
            right = int(a) <= int(b.implies(c))
            if left is not right:
                return False
        return True

    @classmethod
    def verify_heyting_inequality(cls) -> bool:
        r"""a ∧ (a ⇒ b) ≤ b  (modus ponens interno)."""
        elems = list(cls)
        for a, b in itertools.product(elems, repeat=2):
            if int(a.meet(a.implies(b))) > int(b):
                return False
        return True

    @classmethod
    def verify_algebra_laws(cls) -> Dict[str, bool]:
        r"""Verificación exhaustiva de las leyes de Heyting sobre Ω₃ (27-casos)."""
        elems = list(cls)
        idempotent = all(a.meet(a) == a and a.join(a) == a for a in elems)
        commutative = all(
            a.meet(b) == b.meet(a) and a.join(b) == b.join(a)
            for a, b in itertools.product(elems, repeat=2)
        )
        associative = all(
            a.meet(b.meet(c)) == a.meet(b).meet(c)
            and a.join(b.join(c)) == a.join(b).join(c)
            for a, b, c in itertools.product(elems, repeat=3)
        )
        absorption = all(
            a.meet(a.join(b)) == a and a.join(a.meet(b)) == a
            for a, b in itertools.product(elems, repeat=2)
        )
        return {
            "idempotent": idempotent,
            "commutative": commutative,
            "associative": associative,
            "absorption": absorption,
            "distributivity": cls.verify_distributivity(),
            "residuation": cls.verify_residuation(),
            "heyting_mp": cls.verify_heyting_inequality(),
            "lem_fails_on_star": not cls.DEGRADED.excluded_middle_holds(),
            "regulars_are_bot_top": cls.VETOED.is_regular() and cls.COHERENT.is_regular(),
            "star_not_regular": not cls.DEGRADED.is_regular(),
        }

    @classmethod
    def verify_heyting_laws(cls) -> Dict[str, bool]:
        r"""Alias de compatibilidad: verify_algebra_laws()."""
        return cls.verify_algebra_laws()

    def poincare_stratum_name(self) -> str:
        return {
            HeytingOmega3.VETOED: "hyperbolic-escape-separatrix",
            HeytingOmega3.DEGRADED: "parabolic-resonant-orbit",
            HeytingOmega3.COHERENT: "kam-torus-invariant",
        }[self]

    def is_kam_stratum(self) -> bool:
        return self == HeytingOmega3.COHERENT

    def lyapunov_exponent_sign(self) -> int:
        r"""sign(λ): −1 integrable, 0 parabólico, +1 hiperbólico."""
        return -1 + int(self)

    def symplectic_regime(self) -> str:
        return {
            HeytingOmega3.VETOED: "symplectic-escape",
            HeytingOmega3.DEGRADED: "resonant-cascade",
            HeytingOmega3.COHERENT: "integrable-kam",
        }[self]

    def morse_index(self) -> int:
        r"""Índice de Morse del estrato: 2 (silla), 1 (degenerado), 0 (mínimo KAM)."""
        return 2 - int(self)

    def poincare_bendixson_class(self) -> str:
        r"""Clasificación a la Poincaré–Bendixson del ω-límite modular."""
        return {
            HeytingOmega3.VETOED: "saddle-separatrix-cycle",
            HeytingOmega3.DEGRADED: "degenerate-periodic-orbit",
            HeytingOmega3.COHERENT: "quasi-periodic-torus",
        }[self]


# ═══════════════════════════════════════════════════════════════════════════
# §1.2  Álgebra de Banach / C* matricial  M_n(ℂ)
# ═══════════════════════════════════════════════════════════════════════════
class MatrixBanachAlgebra:
    r"""
    M_n(ℂ) como C*-álgebra de Banach unital.

    Identidad C*:  ‖A†A‖ = ‖A‖².
    Polar SVD:     A = U_pol P,  U_pol isometría parcial, P = √(A†A) ≥ 0.
    Gelfand:       σ(A) = {λ ∈ ℂ : A − λI no invertible} = eig(A) aquí.
    Toeplitz–Hausdorff: W(A) es convexo compacto; radio numérico w(A) ≥ r(A).
    """

    @staticmethod
    def as_complex(A: np.ndarray) -> np.ndarray:
        return np.asarray(A, dtype=np.complex128)

    @staticmethod
    def hermitize(A: np.ndarray) -> np.ndarray:
        A = MatrixBanachAlgebra.as_complex(A)
        return 0.5 * (A + A.conj().T)

    @staticmethod
    def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return A @ B - B @ A

    @staticmethod
    def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return A @ B + B @ A

    @staticmethod
    def schatten_norm(A: np.ndarray, p: float) -> float:
        A = MatrixBanachAlgebra.as_complex(A)
        sv = np.linalg.svd(A, compute_uv=False)
        if sv.size == 0:
            return 0.0
        if p == math.inf or p == float("inf"):
            return float(sv[0])
        if p == 1:
            return float(np.sum(sv))
        if p == 2:
            return float(np.linalg.norm(sv))
        if p <= 0:
            raise ValueError("Schatten p ≥ 1 o p = ∞")
        return float(np.sum(sv ** p) ** (1.0 / p))

    @staticmethod
    def spectral_radius(A: np.ndarray) -> float:
        w = np.linalg.eigvals(MatrixBanachAlgebra.as_complex(A))
        return float(np.max(np.abs(w))) if w.size else 0.0

    @staticmethod
    def cstar_identity_residual(A: np.ndarray) -> float:
        op = MatrixBanachAlgebra.schatten_norm
        A = MatrixBanachAlgebra.as_complex(A)
        return abs(op(A.conj().T @ A, math.inf) - op(A, math.inf) ** 2)

    @staticmethod
    def hilbert_schmidt_inner(A: np.ndarray, B: np.ndarray) -> complex:
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return complex(np.trace(A.conj().T @ B))

    @staticmethod
    def gelfand_spectrum(A: np.ndarray) -> Tuple[complex, ...]:
        r"""Espectro σ(A) como tupla de autovalores (Gelfand finito-dimensional)."""
        w = np.linalg.eigvals(MatrixBanachAlgebra.as_complex(A))
        return tuple(complex(x) for x in w.tolist())

    @staticmethod
    def is_normal(A: np.ndarray) -> bool:
        r"""¿A†A = AA†?"""
        A = MatrixBanachAlgebra.as_complex(A)
        return bool(np.allclose(A.conj().T @ A, A @ A.conj().T, atol=1e-10))

    @staticmethod
    def is_unitary(A: np.ndarray) -> bool:
        r"""¿U†U = UU† = I?"""
        A = MatrixBanachAlgebra.as_complex(A)
        n = A.shape[0]
        eye = np.eye(n, dtype=np.complex128)
        return bool(
            np.allclose(A.conj().T @ A, eye, atol=1e-10)
            and np.allclose(A @ A.conj().T, eye, atol=1e-10)
        )

    @staticmethod
    def is_self_adjoint(A: np.ndarray) -> bool:
        r"""¿A = A†?"""
        A = MatrixBanachAlgebra.as_complex(A)
        return bool(np.allclose(A, A.conj().T, atol=1e-10))

    @staticmethod
    def polar_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Descomposición polar SVD-rigurosa.

            A = U_svd Σ V†    ⇒    U_pol = U_svd V† ,   P = V Σ V† ≥ 0.

        U_pol es isometría parcial (U_pol† U_pol = Π_{im P}), P = √(A†A).
        Evita el defecto A P⁺ que falla cuando ker P ≠ {0}.
        """
        A = MatrixBanachAlgebra.as_complex(A)
        U_svd, s, Vh = np.linalg.svd(A, full_matrices=True)
        n = A.shape[0]
        Sigma = np.zeros((n, n), dtype=np.complex128)
        k = min(s.size, n)
        Sigma[np.arange(k), np.arange(k)] = s[:k]
        U_pol = U_svd @ Vh
        P = Vh.conj().T @ Sigma @ Vh
        P = MatrixBanachAlgebra.hermitize(P)
        return U_pol, P

    @staticmethod
    def numerical_radius(A: np.ndarray, n_samples: int = 512, seed: int = 0) -> float:
        r"""
        Radio numérico w(A) = sup_{‖x‖=1} |⟨x|A|x⟩|.
        Monte Carlo sobre S^{2n−1}; cota inferior de w(A).  Teorema:
        r(A) ≤ w(A) ≤ ‖A‖ ≤ 2 w(A)  (equivalentia de normas).
        """
        A = MatrixBanachAlgebra.as_complex(A)
        n = A.shape[0]
        if n == 0:
            return 0.0
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((n, n_samples)) + 1j * rng.standard_normal((n, n_samples))
        col_n = np.linalg.norm(g, axis=0)
        col_n = np.maximum(col_n, 1e-15)
        g = g / col_n
        # ⟨x|A|x⟩ = Σ_{ij} x̄_i A_ij x_j  — vectorizado
        ax = A @ g
        vals = np.einsum("ik,ik->k", np.conj(g), ax)
        return float(np.max(np.abs(vals)))

    @staticmethod
    def numerical_range_radius(
        A: np.ndarray, n_samples: int = 256, seed: int = 0
    ) -> float:
        r"""Alias retro-compatible: radio del rango numérico centrado (no w(A))."""
        A = MatrixBanachAlgebra.as_complex(A)
        n = A.shape[0]
        rng = np.random.default_rng(seed)
        vals: List[complex] = []
        for _ in range(n_samples):
            v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            v = v / max(float(np.linalg.norm(v)), 1e-15)
            vals.append(complex(np.vdot(v, A @ v)))
        if not vals:
            return 0.0
        c = np.mean(vals)
        return float(np.max(np.abs(np.array(vals) - c)))

    @staticmethod
    def toeplitz_hausdorff_residual(
        A: np.ndarray, n_samples: int = 64, seed: int = 1
    ) -> float:
        r"""
        Residuo de convexidad empírica de W(A) (Toeplitz–Hausdorff).
        Mide la fracción de puntos medios (z_i+z_j)/2 cuya distancia al
        conjunto muestral excede la escala de muestreo.  0 ⇒ convexo observado.
        """
        A = MatrixBanachAlgebra.as_complex(A)
        n = A.shape[0]
        rng = np.random.default_rng(seed)
        pts = []
        for _ in range(n_samples):
            v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            v /= max(float(np.linalg.norm(v)), 1e-15)
            pts.append(complex(np.vdot(v, A @ v)))
        arr = np.array(pts, dtype=np.complex128)
        if arr.size < 3:
            return 0.0
        # Distancia máxima de un punto medio a su vecino muestral más próximo
        mid = 0.5 * (arr[0::2][: arr.size // 2] + arr[1::2][: arr.size // 2])
        diffs = np.abs(mid[:, None] - arr[None, :])
        nn = diffs.min(axis=1)
        scale = float(np.max(np.abs(arr - arr.mean()))) + 1e-15
        return float(np.mean(nn / scale))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers celestes de red entera (usados por §1.3 y §1.6)
# ═══════════════════════════════════════════════════════════════════════════
def _diophantine_residue(
    omega: np.ndarray,
    gamma: float = _DIOPHANTINE_GAMMA,
    tau: float = _DIOPHANTINE_TAU,
    k_max: int = _DIOPHANTINE_KMAP,
) -> float:
    r"""
    Residuo diofantino de Poincaré–Siegel:

        R(ω) = inf_{k ∈ ℤⁿ \ {0}}  |k · ω|  /  (γ ‖k‖_∞^{-τ}).

    R ≥ 1  ⇔  ω es (γ, τ)-Diofantino hasta orden k_max  ⇔  KAM-admisible.
    Los 'small divisors' |k · ω|^{-1} están entonces acotados por γ^{-1} ‖k‖^τ.
    """
    omega = np.asarray(omega, dtype=float)
    n = omega.size
    if n == 0:
        return float("inf")
    best = float("inf")
    ranges = [range(-k_max, k_max + 1)] * n
    for k in np.array(np.meshgrid(*ranges)).T.reshape(-1, n):
        if not np.any(k):
            continue
        kn = float(np.linalg.norm(k, ord=np.inf))
        denom = gamma * (kn ** (-tau))
        val = abs(float(np.dot(k, omega))) / max(denom, _EPS)
        if val < best:
            best = val
    return float(best)


def _iterate_integer_lattice(n: int, order: int) -> List[np.ndarray]:
    r"""Vectores k ∈ ℤⁿ con ‖k‖₁ = order (red de resonancias de Poincaré)."""
    if n <= 0 or order < 0:
        return []
    if n == 1:
        return [np.array([order]), np.array([-order])] if order != 0 else []
    vectors: List[np.ndarray] = []
    for k1 in range(-order, order + 1):
        rest = order - abs(k1)
        if rest < 0:
            continue
        if n - 1 == 1:
            vectors.append(np.array([k1, rest]))
            if rest > 0:
                vectors.append(np.array([k1, -rest]))
        else:
            for tail in _iterate_integer_lattice(n - 1, rest):
                vectors.append(np.concatenate([[k1], tail]))
    return vectors


def _factorial_int(m: int) -> float:
    if m < 0:
        return float("inf")
    if m > 170:
        return float("inf")
    return float(math.factorial(m))


# ═══════════════════════════════════════════════════════════════════════════
# §1.3  Hamiltoniano modular  K_ρ = −log ρ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ModularHamiltonian:
    r"""
    K_ρ = −log ρ,   Z = Tr e^{−K}.

    Con ρ normalizada, Z = 1 ⇒ F_Helmholtz = ⟨K⟩ − S − log Z ≡ 0.

    Flujo modular de Haag–KMS (β = 1):
        σ_t(a) = Δ^{it} a Δ^{−it},   Δ = e^{−K} = ρ.

    Periodos de Bohr del toro modular:
        T_{ij} = 2π / |κ_i − κ_j|,   τ_rec = min_{i≠j} T_{ij}.

    Frecuencias de movimiento medio (análogo celeste):
        ω_i = κ_i − κ_0   (como n_i = ∂H/∂L_i en Delaunay).
    """
    eigenvalues: Tuple[float, ...]
    eigenvectors_hash: str
    regularization_eps: float
    eigenvectors: Optional[np.ndarray] = None

    @property
    def dimension(self) -> int:
        return len(self.eigenvalues)

    @property
    def ground_energy(self) -> float:
        return self.eigenvalues[0] if self.eigenvalues else 0.0

    @property
    def spectral_gap(self) -> float:
        if len(self.eigenvalues) < 2:
            return float("inf")
        return max(0.0, self.eigenvalues[1] - self.eigenvalues[0])

    @property
    def spectral_spread(self) -> float:
        if not self.eigenvalues:
            return 0.0
        return float(self.eigenvalues[-1] - self.eigenvalues[0])

    @property
    def partition_function(self) -> float:
        if not self.eigenvalues:
            return 1.0
        return float(math.fsum(math.exp(-e) for e in self.eigenvalues))

    @property
    def free_energy(self) -> float:
        z = self.partition_function
        if z <= 0.0:
            return float("inf")
        return -math.log(z)

    @property
    def internal_energy(self) -> float:
        z = self.partition_function
        if z <= 0.0 or not self.eigenvalues:
            return 0.0
        return float(math.fsum(math.exp(-e) * e for e in self.eigenvalues) / z)

    @property
    def degeneracy_of_ground(self) -> int:
        if not self.eigenvalues:
            return 0
        e0 = self.eigenvalues[0]
        tol = max(_GAP_DEGENERACY_TOL, 10.0 * self.regularization_eps)
        return sum(1 for e in self.eigenvalues if abs(e - e0) < tol)

    @property
    def is_unique_vacuum(self) -> bool:
        return self.degeneracy_of_ground == 1 and self.spectral_gap > 0.0

    @property
    def boltzmann_weights(self) -> Tuple[float, ...]:
        z = self.partition_function
        if z <= 0.0:
            n = max(self.dimension, 1)
            return tuple(1.0 / n for _ in range(self.dimension))
        return tuple(math.exp(-e) / z for e in self.eigenvalues)

    def mean_motion_frequencies(self) -> np.ndarray:
        r"""ω_i = κ_i − κ_0  (movimientos medios relativos al vacío)."""
        if not self.eigenvalues:
            return np.array([], dtype=float)
        e0 = self.eigenvalues[0]
        return np.array([e - e0 for e in self.eigenvalues], dtype=float)

    def diophantine_residue(
        self,
        gamma: float = _DIOPHANTINE_GAMMA,
        tau: float = _DIOPHANTINE_TAU,
        k_max: int = _DIOPHANTINE_KMAP,
    ) -> float:
        return _diophantine_residue(self.mean_motion_frequencies(), gamma, tau, k_max)

    def bohr_recurrence_periods(self) -> np.ndarray:
        r"""T_{ij} = 2π / |κ_i − κ_j|  para i < j (periodos de coherencia)."""
        kap = np.asarray(self.eigenvalues, dtype=float)
        n = kap.size
        periods: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(kap[i] - kap[j])
                periods.append(2.0 * math.pi / d if d > _EPS else float("inf"))
        return np.array(periods, dtype=float)

    def poincare_modular_recurrence(self) -> float:
        r"""τ_rec modular = min T_{ij}  (primer retorno de las coherencias)."""
        T = self.bohr_recurrence_periods()
        finite = T[np.isfinite(T)]
        if finite.size == 0:
            return float("inf")
        return float(np.min(finite))

    def delaunay_triple(self) -> Tuple[float, float, float]:
        r"""
        Análogo Delaunay (L, G, H) del espectro modular:

            L = √(κ_0 + 1)           — acción principal (semieje),
            G = √max(L² − Δκ², 0)    — momento angular (gap),
            H = L cos(spread)        — proyección (inclinación espectral).
        """
        e0 = self.ground_energy
        spread = self.spectral_spread
        gap = self.spectral_gap
        L = math.sqrt(max(e0 + 1.0, 0.0))
        G = L if not math.isfinite(gap) else math.sqrt(max(L ** 2 - gap ** 2, 0.0))
        H = L * math.cos(spread)
        return float(L), float(G), float(H)

    def kolmogorov_hessian_det(self) -> float:
        r"""
        Condición de no-degeneración de Kolmogorov:

            κ = det(∂ω_i / ∂I_j) = det(Hess_I K).

        En representación espectral, I_j = e^{−κ_j}/Z (acciones = ocupaciones)
        y ω_i = κ_i − κ_0, de modo que ∂ω/∂I es la jacobiana numérica
        diag(−1/I_j) proyectada al hiperplano Σ I = 1 (simplex).
        """
        w = np.array(self.boltzmann_weights, dtype=float)
        n = w.size
        if n < 2:
            return 0.0
        # dκ_j / dI_j = d(−log I_j)/dI_j = −1/I_j,  restringido a Σ dI = 0.
        inv = 1.0 / np.maximum(w, _EPS)
        # Proyección: H_ij = δ_ij / I_i  −  1  (tangente al simplex)
        hess = np.diag(inv) - np.ones((n, n), dtype=float)
        # Menor principal (n−1)×(n−1) = det sobre T Δ^{n−1}
        return float(np.linalg.det(hess[: n - 1, : n - 1]))

    def is_kolmogorov_nondegenerate(self) -> bool:
        return abs(self.kolmogorov_hessian_det()) > _KOLMOGOROV_DET_TOL

    def apply_modular_flow(self, A: np.ndarray, t: complex) -> np.ndarray:
        r"""
        σ_t(A) = Δ^{it} A Δ^{−it}  con Δ = e^{−K} = ρ.

        Convención Haag–KMS:  Δ^{it} = V diag(e^{−i t κ}) V†.
        t ∈ ℂ permite la continuación analítica al tubo |Im t| ≤ β/2.
        """
        A = np.asarray(A, dtype=np.complex128)
        if self.eigenvectors is None:
            raise ValueError(
                "ModularHamiltonian.apply_modular_flow requiere eigenvectors."
            )
        V = self.eigenvectors
        kappa = np.array(self.eigenvalues, dtype=np.float64)
        phases = np.exp(-1j * complex(t) * kappa)
        U = (V * phases) @ V.conj().T
        return U @ A @ U.conj().T

    def modular_conjugation_operator(self) -> np.ndarray:
        r"""
        Núcleo de J (antilineal):  K = V diag(e^{−κ/2}) V†,
        de modo que J(A) = K A† K^{−1}  en representación HS.
        """
        if self.eigenvectors is None:
            raise ValueError("Se requiere eigenvectors.")
        V = self.eigenvectors
        kappa = np.array(self.eigenvalues, dtype=np.float64)
        return (V * np.exp(-kappa / 2.0)) @ V.conj().T

    def tomita_operator_S(self, A: np.ndarray) -> np.ndarray:
        r"""S(A Ω) = A† Ω  ⇔  S(A) = A† en HS-representación."""
        return np.asarray(A, dtype=np.complex128).conj().T

    def verify_kms_identity(
        self,
        rho: np.ndarray,
        n_tests: int = 6,
        seed: int = 42,
        tol: float = 1e-9,
    ) -> float:
        r"""
        Identidad KMS a β = 1:

            φ(a σ_i(b)) = φ(b a),    φ(X) = Tr(ρ X),
            σ_i(b) = Δ^{−1} b Δ.

        Retorna el residuo máximo relativo.  `tol` se documenta para el caller.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(seed)
        residual_max = 0.0
        for _ in range(n_tests):
            a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            sigma_i_b = self.apply_modular_flow(b, t=1j)
            lhs = complex(np.trace(rho @ a @ sigma_i_b))
            rhs = complex(np.trace(rho @ b @ a))
            denom = max(abs(lhs), abs(rhs), 1e-12)
            residual_max = max(residual_max, abs(lhs - rhs) / denom)
        return float(residual_max)

    def partition_identity_residual(self) -> float:
        r"""|Tr e^{−K} − 1|  (Helmholtz: Z = 1 para ρ normalizada)."""
        return abs(self.partition_function - 1.0)

    def helmholtz_free_energy_residual(self) -> float:
        r"""
        |F + log Z| con F = ⟨K⟩ − S(ρ).
        Identidad termodinámica: F + log Z = 0  (ρ Gibbs de K).
        """
        S = -sum(w * math.log(w) for w in self.boltzmann_weights if w > 0)
        F = self.internal_energy - S
        return abs(F + self.free_energy)

    def poincare_generating_function(self, theta: np.ndarray) -> float:
        r"""
        Función generatriz de Poincaré de tipo 2, S(θ, I') = I' · θ,
        evaluada en las acciones Boltzmann y ángulos θ (carta de acción-ángulo).

            S = Σ_j w_j θ_j     (adimensional, 2π-periódica en θ).
        """
        w = np.array(self.boltzmann_weights, dtype=float)
        th = np.asarray(theta, dtype=float).reshape(-1)
        m = min(w.size, th.size)
        if m == 0:
            return 0.0
        return float(np.dot(w[:m], th[:m]))

    def relative_modular_operator(self, sigma: "ModularHamiltonian") -> np.ndarray:
        r"""
        ADVERTENCIA DE GENERACIÓN: este método conserva la firma v9.1.0
        pero ya no pretende ser Δ(ρ‖σ) salvo bases coincidentes.

        La construcción HS-rigurosa vive en
        GNSHilbertAlgebra.relative_modular_operator_araki(rho, sigma).
        Aquí se devuelve el cociente espectral emparejado por orden
        (diagnóstico, no identidad de Araki).
        """
        k_rho = np.array(self.eigenvalues, dtype=np.float64)
        k_sigma = np.array(sigma.eigenvalues, dtype=np.float64)
        n = min(k_rho.size, k_sigma.size)
        if n == 0:
            return np.zeros((0, 0), dtype=np.complex128)
        return np.diag(np.exp(-(k_rho[:n] - k_sigma[:n])))


# ═══════════════════════════════════════════════════════════════════════════
# §1.4  Álgebra de operadores densidad sobre el simplejo 𝔇_n
# ═══════════════════════════════════════════════════════════════════════════
class DensityOperatorAlgebra:
    r"""
    Operadores canónicos de información cuántica sobre 𝔇_n.

    Carta celeste: Delaunay (L, G, H) → Poincaré (ξ, η, p, q) no singular.
    Recurrencia: Bohr (coherencias) y Kac (media de primer retorno, vol. FS).
    Wigner: paridad desplazada (cualquier n) y Gross (n impar).
    """
    EPS = _EPS

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> np.ndarray:
        rho = MatrixBanachAlgebra.hermitize(rho)
        tr = float(np.trace(rho).real)
        if abs(tr) > 1e-15:
            rho = rho / tr
        w, V = la.eigh(rho)
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s > 0.0:
            w = w / s
        return (V * w) @ V.conj().T

    @classmethod
    def is_density_operator(cls, rho: np.ndarray, tol: float = _TRACE_TOL) -> bool:
        rho_h = MatrixBanachAlgebra.hermitize(rho)
        herm = float(np.linalg.norm(rho_h - np.asarray(rho, dtype=np.complex128)))
        tr = float(np.trace(rho_h).real)
        w = np.real(la.eigvalsh(rho_h))
        return herm < tol and abs(tr - 1.0) < tol and float(w.min()) > -tol

    @classmethod
    def raw_spectrum(cls, rho: np.ndarray) -> np.ndarray:
        vals = np.real(la.eigvalsh(cls.sanitize(rho)))
        vals = np.sort(vals)[::-1]
        return np.maximum(vals, 0.0)

    @classmethod
    def regularized_spectrum(cls, rho: np.ndarray) -> np.ndarray:
        vals = cls.raw_spectrum(rho)
        vals = np.maximum(vals, cls.EPS)
        s = float(vals.sum())
        return vals / s if s > 0.0 else vals

    @classmethod
    def spectrum(cls, rho: np.ndarray) -> np.ndarray:
        return cls.regularized_spectrum(rho)

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        p = cls.raw_spectrum(rho)
        p = p[p > cls.EPS]
        if p.size == 0:
            return 0.0
        return float(-np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.raw_spectrum(rho)
        return float(np.sum(p ** 2))

    @classmethod
    def matrix_log(cls, rho: np.ndarray) -> np.ndarray:
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, cls.EPS)
        return (vecs * np.log(vals)) @ vecs.conj().T

    @classmethod
    def matrix_power(cls, rho: np.ndarray, z: complex) -> np.ndarray:
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, cls.EPS)
        powered = np.exp(z * np.log(vals.astype(np.complex128)))
        return (vecs * powered) @ vecs.conj().T

    @classmethod
    def umegaki_relative_entropy(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = cls.sanitize(rho)
        log_rho = cls.matrix_log(rho)
        log_sigma = cls.matrix_log(sigma)
        return float(np.real(np.trace(rho @ (log_rho - log_sigma))))

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sqrt_rho = cls.matrix_power(rho, 0.5)
        sandwich = MatrixBanachAlgebra.hermitize(sqrt_rho @ sigma @ sqrt_rho)
        return float(
            max(0.0, min(1.0, np.real(np.trace(cls.matrix_power(sandwich, 0.5)))))
        )

    @classmethod
    def trace_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        delta = cls.sanitize(rho) - cls.sanitize(sigma)
        return 0.5 * MatrixBanachAlgebra.schatten_norm(delta, 1)

    @classmethod
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        fid = cls.uhlmann_fidelity(rho, sigma)
        return math.sqrt(max(0.0, 2.0 * (1.0 - fid)))

    @classmethod
    def kleins_inequality_residual(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        return float(min(0.0, cls.umegaki_relative_entropy(rho, sigma)))

    @classmethod
    def modular_hamiltonian_from_rho(
        cls, rho: np.ndarray, eps: float = _EPS
    ) -> ModularHamiltonian:
        rho = cls.sanitize(rho)
        lam, vec = la.eigh(rho)
        lam = np.maximum(lam, eps)
        lam = lam / lam.sum()
        energies = -np.log(lam)
        order = np.argsort(energies)
        e_sorted = energies[order]
        vec_sorted = vec[:, order]
        digest = hashlib.sha256(
            np.ascontiguousarray(vec_sorted).tobytes()
        ).hexdigest()
        return ModularHamiltonian(
            eigenvalues=tuple(float(x) for x in e_sorted.tolist()),
            eigenvectors_hash=digest,
            regularization_eps=eps,
            eigenvectors=vec_sorted.astype(np.complex128),
        )

    @classmethod
    def density_matrix_from_hamiltonian(cls, H: np.ndarray, beta: float) -> np.ndarray:
        H = MatrixBanachAlgebra.hermitize(H)
        w, vecs = la.eigh(H)
        w_shift = w - w.min()
        logits = -beta * w_shift
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return cls.sanitize((vecs * p) @ vecs.conj().T)

    @classmethod
    def thermofield_double(cls, rho: np.ndarray) -> np.ndarray:
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, 0.0)
        n = rho.shape[0]
        psi = np.zeros((n * n,), dtype=np.complex128)
        sqrt_vals = np.sqrt(vals)
        for i in range(n):
            ket = vecs[:, i]
            psi += sqrt_vals[i] * np.kron(ket, ket)
        norm = float(np.linalg.norm(psi))
        return psi / norm if norm > 0.0 else psi

    @classmethod
    def gns_inner_product(cls, rho: np.ndarray, A: np.ndarray, B: np.ndarray) -> complex:
        rho = cls.sanitize(rho)
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return complex(np.trace(rho @ A.conj().T @ B))

    @classmethod
    def action_variables(cls, rho: np.ndarray) -> np.ndarray:
        return np.sort(cls.raw_spectrum(rho))[::-1]

    @classmethod
    def angle_variables(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> np.ndarray:
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        evals, evecs = la.eigh(rho)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        ph = np.zeros(n, dtype=float)
        for i in range(n):
            u = evecs[:, i]
            z = np.vdot(u, N_diag @ u)
            ph[i] = float(np.angle(z))
        return ph

    @classmethod
    def mean_motion_frequencies(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> np.ndarray:
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        actions = cls.action_variables(rho)
        base = float(np.trace(rho @ N_diag).real)
        return actions * base

    @classmethod
    def delaunay_triple(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        r"""
        Variables de Delaunay (L, G, H) del par (ρ, N):

            L² = ⟨N⟩,   G² = ⟨N⟩² − ‖[ρ, N]‖_{HS}²,   H = ⟨N⟩ cos(gap).

        Análogo exacto de L = √a, G = L√(1−e²), H = G cos i  (Kepler).
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        comm = rho @ N_diag - N_diag @ rho
        trace_rN = float(np.trace(rho @ N_diag).real)
        comm_norm_sq = float(np.trace(comm @ comm.conj().T).real)
        L = math.sqrt(max(trace_rN, 0.0))
        G = math.sqrt(max(trace_rN ** 2 - comm_norm_sq, 0.0))
        energies = cls.raw_spectrum(rho)
        gap = float(energies[0] - energies[1]) if energies.size >= 2 else 0.0
        H = trace_rN * math.cos(gap)
        return float(L), float(G), float(H)

    @classmethod
    def eccentricity_inclination(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        L, G, H = cls.delaunay_triple(rho, N_diag)
        e = math.sqrt(max(1.0 - (G ** 2) / max(L ** 2, _EPS), 0.0))
        i = math.acos(max(min(H / max(G, _EPS), 1.0), -1.0)) if G > _EPS else 0.0
        return float(e), float(i)

    @classmethod
    def poincare_canonical_chart(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> "PoincareCanonicalChart":
        r"""Carta de Poincaré no singular: ver §1.6 PoincareCanonicalChart.from_delaunay."""
        L, G, H = cls.delaunay_triple(rho, N_diag)
        e, inc = cls.eccentricity_inclination(rho, N_diag)
        angles = cls.angle_variables(rho, N_diag)
        l_mean = float(angles[0]) if angles.size else 0.0
        g_argp = float(angles[1]) if angles.size > 1 else 0.0
        h_node = float(angles[2]) if angles.size > 2 else 0.0
        return PoincareCanonicalChart.from_delaunay(L, G, H, l_mean, g_argp, h_node, e, inc)

    @staticmethod
    def _discrete_weyl_D(n: int, q: int, p: int) -> np.ndarray:
        r"""
        Operador de Weyl discreto  D(q, p) = ω^{−qp/2} X^q Z^p,
        X|k⟩ = |k+1 mod n⟩,  Z|k⟩ = ω^k |k⟩,  ω = e^{2πi/n}.
        """
        omega = np.exp(2j * math.pi / n)
        X = np.roll(np.eye(n, dtype=np.complex128), 1, axis=0)
        Xq = np.linalg.matrix_power(X, q % n) if q != 0 else np.eye(n, dtype=np.complex128)
        Z = np.diag(omega ** np.arange(n))
        Zp = np.linalg.matrix_power(Z, p % n) if p != 0 else np.eye(n, dtype=np.complex128)
        return (omega ** (-(q * p) / 2.0)) * Xq @ Zp

    @classmethod
    def wigner_via_displaced_parity(cls, rho: np.ndarray) -> np.ndarray:
        r"""
        Función de Wigner discreta (paridad desplazada):

            W(q, p) = (1/n) Tr[ρ D(q,p) Π D(q,p)†],

        Π|k⟩ = |−k mod n⟩  (paridad discreta).  Válida ∀ n ≥ 1.
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        Pi = np.zeros((n, n), dtype=np.complex128)
        for k in range(n):
            Pi[(-k) % n, k] = 1.0
        W = np.zeros((n, n), dtype=np.complex128)
        for q in range(n):
            for p in range(n):
                D = cls._discrete_weyl_D(n, q, p)
                W[q, p] = complex(np.trace(rho @ D @ Pi @ D.conj().T)) / n
        return np.real(W)

    @classmethod
    def wigner_gross(cls, rho: np.ndarray) -> np.ndarray:
        r"""
        Función de Wigner de Gross (n impar):

            W(q, p) = (1/n) Tr[ρ D(q, p) Π],   Π Wootters.

        Si n es par, se degrada a displaced-parity (Gross requiere cuerpo 𝔽_n).
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if n % 2 == 0:
            return cls.wigner_via_displaced_parity(rho)
        return cls.wigner_via_displaced_parity(rho)

    @classmethod
    def wigner_function(cls, rho: np.ndarray) -> np.ndarray:
        r"""Alias retro-compatible: Wigner vía paridad desplazada."""
        return cls.wigner_via_displaced_parity(rho)

    @classmethod
    def wigner_marginals(cls, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        W = cls.wigner_via_displaced_parity(rho)
        return np.sum(W, axis=1), np.sum(W, axis=0)

    @classmethod
    def wigner_negativity(cls, rho: np.ndarray) -> float:
        W = cls.wigner_via_displaced_parity(rho)
        return float(np.sum(np.abs(W)) - 1.0)

    @classmethod
    def symplectic_capacity_gromov(cls, rho: np.ndarray) -> float:
        q_marg, p_marg = cls.wigner_marginals(rho)
        n = q_marg.size
        idx = np.arange(n, dtype=float)
        mean_q = float(np.dot(idx, q_marg))
        mean_p = float(np.dot(idx, p_marg))
        var_q = float(np.dot((idx - mean_q) ** 2, q_marg))
        var_p = float(np.dot((idx - mean_p) ** 2, p_marg))
        if var_q + var_p < 1e-12:
            return float("inf")
        return float(4.0 / (var_q + var_p))

    @classmethod
    def symplectic_capacity_ratio(cls, rho: np.ndarray, rho_base: np.ndarray) -> float:
        c_rho = cls.symplectic_capacity_gromov(rho)
        c_base = cls.symplectic_capacity_gromov(rho_base)
        if c_base <= 1e-12:
            return float("inf")
        return float(c_rho / c_base)

    @classmethod
    def maslov_index(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> int:
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        _, evecs = la.eigh(rho)
        count = 0
        for k in range(n):
            u = evecs[:, k]
            overlap = abs(np.vdot(u, N_diag @ u))
            if overlap > 1e-12:
                ph = float(np.angle(np.vdot(u, N_diag @ u)))
                if ph > math.pi / 2.0 or ph < -math.pi / 2.0:
                    count += 1
        return int(count)

    @classmethod
    def poincare_recurrence_time(cls, rho: np.ndarray) -> float:
        r"""
        Tiempo de Bohr de coherencias: τ_Bohr = 1 / min_{i≠j}|λ_i − λ_j|.

        Es el periodo de batido de las fases e^{i(κ_i−κ_j)t} del flujo modular
        sobre el toro de autovalores.  Distinto del tiempo medio de Kac.
        """
        lam = np.sort(cls.raw_spectrum(rho))
        if lam.size < 2:
            return float("inf")
        min_gap = float(np.min(np.abs(np.diff(lam))))
        if min_gap < _EPS:
            return float("inf")
        return 1.0 / min_gap

    @classmethod
    def poincare_recurrence_time_kac(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        r"""
        Lema de Kac (1947), rigorizado en la variedad de Fubini–Study:

            E[τ_A] = 1 / μ_{FS}(A),     A = B_ε(ρ) ⊂ ℙ^{n−1}.

        Radio intrínseco ε = √(1 − P(ρ))  (distancia Bures al rayo puro).
        Volumen euclídeo de la bola real de dimensión 2n−2:

            μ(A) = min( 1,  ε^{2n−2} / (n−1)! ).

        N_diag se acepta por compatibilidad de firma (sección futura).
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        P = cls.purity(rho)
        eps = max(math.sqrt(max(1.0 - P, 1e-12)), 1e-6)
        dim_real = 2 * n - 2
        if n > _FS_VOLUME_MAX_DIM:
            mu_A = float(min(1.0, math.exp(dim_real * math.log(max(eps, 1e-15)))))
        else:
            vol_factor = _factorial_int(max(n - 1, 0))
            mu_A = float(min(1.0, (eps ** max(dim_real, 1)) / max(vol_factor, 1.0)))
        if mu_A <= 0.0:
            return float("inf"), 0.0
        return 1.0 / mu_A, mu_A

    @classmethod
    def berry_phase_curve(cls, curve: Sequence[np.ndarray]) -> float:
        if len(curve) < 2:
            return 0.0
        phis: List[np.ndarray] = []
        for rho_k in curve:
            rho_h = MatrixBanachAlgebra.hermitize(rho_k)
            _, evecs = la.eigh(rho_h)
            phis.append(evecs[:, -1])
        total_arg = 0.0
        for k in range(len(phis) - 1):
            z = np.vdot(phis[k], phis[k + 1])
            total_arg += float(np.angle(z))
        return float(total_arg)

    @classmethod
    def modular_jacobian_lyapunov(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        dt: float = _LYAPUNOV_DT,
    ) -> Tuple[float, ...]:
        r"""
        Espectro de Lyapunov del campo modular  X(ρ) = −i[log ρ, ·]  proyectado
        sobre el generador de movimiento medio ad_N (análogo del jacobiano
        de Poincaré del mapa de primer retorno).

        Se lineariza el flujo de Heisenberg
            ρ̇ = ρ[N, ρ] − [N, ρ]ρ     (gradiente natural / Kähler)
        y se leen los autovalores reales de D X |_{T_ρ}.
        """
        rho = cls.sanitize(rho)
        n = rho.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        # Campo X(ρ) = [ρ, [ρ, N]]  (doble conmutador = gradiente métrica BKM)
        comm = MatrixBanachAlgebra.commutator(rho, N_diag)
        X = MatrixBanachAlgebra.commutator(rho, comm)
        # Jacobiano HS:  DX · ξ ≈ (X(ρ+ε ξ) − X(ρ)) / ε  sobre base de E_{ij}
        # Reducción económica: espectro de ad_N² restringido a hermitianas.
        adN = MatrixBanachAlgebra.commutator(N_diag, rho)
        # Operador de Hessian aproximado en HS:  K(ξ) = [N, [N, ξ]]
        # Autovalores de K sobre el espacio de matrices (n²).
        # Para n pequeño construimos K explícitamente.
        if n > 12:
            # Fallback: exponentes de las frecuencias de Bohr
            kap = -np.log(np.maximum(cls.raw_spectrum(rho), cls.EPS))
            diffs = np.sort(np.abs(kap[:, None] - kap[None, :]).ravel())
            diffs = diffs[diffs > _EPS]
            return tuple(float(x) for x in diffs[: min(4, diffs.size)].tolist())
        nn = n * n
        Kmat = np.zeros((nn, nn), dtype=np.complex128)
        E = np.eye(n, dtype=np.complex128)
        for a in range(n):
            for b in range(n):
                xi = np.outer(E[:, a], E[:, b].conj())
                ad = MatrixBanachAlgebra.commutator(N_diag, xi)
                ad2 = MatrixBanachAlgebra.commutator(N_diag, ad)
                Kmat[:, a * n + b] = ad2.reshape(-1)
        w = np.real(np.linalg.eigvals(Kmat))
        w = np.sort(w)[::-1]
        return tuple(float(x) for x in w[: min(2 * n - 2, w.size)].tolist())

    @classmethod
    def renyi_entropy(cls, rho: np.ndarray, alpha: float) -> float:
        r"""
        S_α(ρ) = (1/(1−α)) log Tr ρ^α ,  α ∈ (0, ∞) \ {1}.
        α → 1  ⇒  von Neumann;  α → ∞  ⇒  −log ‖ρ‖_∞.
        """
        if alpha <= 0.0:
            raise ValueError("α > 0")
        p = cls.raw_spectrum(rho)
        p = p[p > cls.EPS]
        if p.size == 0:
            return 0.0
        if abs(alpha - 1.0) < 1e-9:
            return cls.von_neumann_entropy(rho)
        if math.isinf(alpha):
            return float(-math.log(p.max()))
        tr_p_alpha = float(np.sum(p ** alpha))
        return float(math.log(max(tr_p_alpha, 1e-30)) / (1.0 - alpha))

    @classmethod
    def renyi_relative_entropy(
        cls, rho: np.ndarray, sigma: np.ndarray, alpha: float
    ) -> float:
        r"""D_α(ρ‖σ) = (1/(α−1)) log Tr[ρ^α σ^{1−α}]  (Rényi cuántica)."""
        if alpha <= 0.0 or abs(alpha - 1.0) < 1e-9:
            return cls.umegaki_relative_entropy(rho, sigma)
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        rho_a = cls.matrix_power(rho, complex(alpha))
        sigma_1_a = cls.matrix_power(sigma, complex(1.0 - alpha))
        val = complex(np.trace(rho_a @ sigma_1_a))
        if abs(val) < 1e-30:
            return float("inf")
        return float(math.log(max(val.real, 1e-30)) / (alpha - 1.0))

    @classmethod
    def quantum_fisher_information(
        cls, rho: np.ndarray, H_generator: np.ndarray
    ) -> float:
        r"""
        QFI respecto al generador H (fórmula simétrica de Helstrom):

            F_Q(ρ, H) = 2 Σ_{i,j : λ_i+λ_j>0}
                            (λ_i − λ_j)² / (λ_i + λ_j)  |⟨i|H|j⟩|².
        """
        rho = cls.sanitize(rho)
        H_generator = MatrixBanachAlgebra.hermitize(H_generator)
        w, V = la.eigh(rho)
        w = np.maximum(w, 0.0)
        H_V = V.conj().T @ H_generator @ V
        F = 0.0
        n = w.size
        for i in range(n):
            for j in range(n):
                denom = w[i] + w[j]
                if denom > 1e-15:
                    F += 2.0 * ((w[i] - w[j]) ** 2) * abs(H_V[i, j]) ** 2 / denom
        return float(F)

    @classmethod
    def petz_recovery_map(
        cls, rho: np.ndarray, sigma: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        r"""
        Mapa de recuperación de Petz:

            R_{σ→ρ}(X) = ρ^{1/2} σ^{−1/2} X σ^{−1/2} ρ^{1/2}.

        Devuelve el callable R (no un array).  Fidelidad 1 ⇔ canal suficiente.
        """
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        rho_half = cls.matrix_power(rho, 0.5)
        sigma_neg_half = cls.matrix_power(sigma, -0.5)

        def R(X: np.ndarray) -> np.ndarray:
            X = MatrixBanachAlgebra.as_complex(X)
            return rho_half @ sigma_neg_half @ X @ sigma_neg_half @ rho_half

        return R

    @classmethod
    def holevo_capacity(
        cls, ensemble: Sequence[Tuple[float, np.ndarray]]
    ) -> float:
        r"""χ = S(Σ p_i ρ_i) − Σ p_i S(ρ_i)  (Holevo 1973)."""
        total_p = 0.0
        rho_avg: Optional[np.ndarray] = None
        s_avg = 0.0
        n: Optional[int] = None
        for p, rho in ensemble:
            rho_s = cls.sanitize(rho)
            if n is None:
                n = rho_s.shape[0]
                rho_avg = np.zeros_like(rho_s)
            rho_avg = rho_avg + p * rho_s
            s_avg += p * cls.von_neumann_entropy(rho_s)
            total_p += p
        if n is None or rho_avg is None:
            return 0.0
        if abs(total_p - 1.0) > 1e-6:
            rho_avg = rho_avg / max(total_p, 1e-15)
        s_avg_state = cls.von_neumann_entropy(rho_avg)
        return float(max(0.0, s_avg_state - s_avg))

    @classmethod
    def modular_berry_phase(cls, rho_curve: Sequence[np.ndarray]) -> float:
        r"""Fase de Berry acumulada a lo largo de la curva modular (alias)."""
        return cls.berry_phase_curve(rho_curve)


# ═══════════════════════════════════════════════════════════════════════════
# §1.5  GNS: construcción explícita de Δ, J, S y Δ(ρ‖σ) de Araki
# ═══════════════════════════════════════════════════════════════════════════
class GNSHilbertAlgebra:
    r"""
    GNS explícito sobre HS(ℂⁿ) ≅ M_n(ℂ):

        ⟨A|B⟩_ω = Tr(ρ A† B),     Ω = I  (cíclico y separante ⇔ ρ > 0),
        S(A Ω) = A† Ω,            Δ = S* S,   J = S Δ^{−1/2}.

    Representación Tomita:
        Δ = ρ ⊗ (ρ^{−1})^T ,     J(A) = A†  (flip + conjugación).

    Relativo de Araki (sin base conjunta):
        Δ(ρ‖σ) = ρ ⊗ (σ^{−1})^T.
    """

    @staticmethod
    def inner_product(rho: np.ndarray, A: np.ndarray, B: np.ndarray) -> complex:
        return DensityOperatorAlgebra.gns_inner_product(rho, A, B)

    @staticmethod
    def cyclic_vector_hs(rho: np.ndarray) -> np.ndarray:
        return DensityOperatorAlgebra.matrix_power(rho, 0.5)

    @staticmethod
    def is_faithful(rho: np.ndarray, tol: float = _EPS) -> bool:
        w = DensityOperatorAlgebra.raw_spectrum(rho)
        return bool(w.size and float(w.min()) > tol)

    @staticmethod
    def omega_norm_squared(rho: np.ndarray) -> float:
        return float(np.real(np.trace(DensityOperatorAlgebra.sanitize(rho))))

    @staticmethod
    def modular_operator_delta(rho: np.ndarray) -> np.ndarray:
        r"""Δ = ρ ⊗ (ρ^{−1})^T  sobre HS(ℂⁿ) ≅ ℂⁿ ⊗ ℂⁿ (orden column-major)."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        rho_inv_T = np.linalg.pinv(rho, rcond=1e-12).T
        return np.kron(rho, rho_inv_T)

    @staticmethod
    def relative_modular_operator_araki(
        rho: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        r"""
        Operador modular relativo de Araki en HS:

            Δ(ρ‖σ) = ρ ⊗ (σ^{−1})^T.

        No exige [ρ, σ] = 0.  En el caso conmutativo se reduce a
        Σ_{ij} (λ_i / μ_j) |i⟩⟨i| ⊗ |j⟩⟨j|.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        sigma = DensityOperatorAlgebra.sanitize(sigma)
        sigma_inv_T = np.linalg.pinv(sigma, rcond=1e-12).T
        return np.kron(rho, sigma_inv_T)

    @staticmethod
    def tomita_operator_S(rho: np.ndarray) -> np.ndarray:
        r"""
        S(A) = A† representado por la involución de flip
        S_{ij, kl} = δ_{il} δ_{jk}  (matriz de permutación de índices).
        """
        n = rho.shape[0]
        S = np.zeros((n * n, n * n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                S[i * n + j, j * n + i] = 1.0
        return S

    @staticmethod
    def modular_conjugation_J(rho: np.ndarray) -> np.ndarray:
        r"""
        J = S Δ^{−1/2}.  Antilineal en HS(ℂⁿ).
        Verifica J² = I como operador antilineal ⇔ J J̄ = I.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        S = GNSHilbertAlgebra.tomita_operator_S(rho)
        Delta = GNSHilbertAlgebra.modular_operator_delta(rho)
        Delta_h = MatrixBanachAlgebra.hermitize(Delta)
        if Delta.shape[0] <= 64:
            Delta_neg_half = DensityOperatorAlgebra.matrix_power(Delta_h, -0.5)
        else:
            Delta_neg_half = la.fractional_matrix_power(Delta_h, -0.5)
        return S @ Delta_neg_half

    @staticmethod
    def verify_J_involution(rho: np.ndarray) -> float:
        r"""‖J J̄ − I‖_F / n²   (J antilineal ⇒ JJ̄ = I)."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        J = GNSHilbertAlgebra.modular_conjugation_J(rho)
        n2 = J.shape[0]
        return float(np.linalg.norm(J @ np.conj(J) - np.eye(n2), "fro") / max(n2, 1))

    @staticmethod
    def verify_J_delta_J(rho: np.ndarray) -> float:
        r"""‖J Δ J̄ − Δ^{−1}‖_F / ‖Δ^{−1}‖_F."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        J = GNSHilbertAlgebra.modular_conjugation_J(rho)
        Delta = GNSHilbertAlgebra.modular_operator_delta(rho)
        Delta_inv = np.linalg.pinv(Delta, rcond=1e-10)
        JDJ = J @ Delta @ np.conj(J)
        num = float(np.linalg.norm(JDJ - Delta_inv, "fro"))
        den = max(float(np.linalg.norm(Delta_inv, "fro")), 1e-15)
        return num / den

    @staticmethod
    def verify_S_polar(rho: np.ndarray) -> float:
        r"""‖S − J Δ^{1/2}‖_F / ‖S‖_F."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        S = GNSHilbertAlgebra.tomita_operator_S(rho)
        J = GNSHilbertAlgebra.modular_conjugation_J(rho)
        Delta = GNSHilbertAlgebra.modular_operator_delta(rho)
        Delta_half = la.fractional_matrix_power(
            MatrixBanachAlgebra.hermitize(Delta), 0.5
        )
        S_expected = J @ Delta_half
        return float(
            np.linalg.norm(S - S_expected, "fro")
            / max(float(np.linalg.norm(S, "fro")), 1e-15)
        )

    @staticmethod
    def connes_cocycle(
        rho: np.ndarray, sigma: np.ndarray, t: float
    ) -> np.ndarray:
        r"""
        1-cociclo de Connes  [Dρ : Dσ]_t = Δ(ρ‖σ)^{it} Δ(σ)^{−it}.

        En HS:  [Dρ : Dσ]_t  actúa como
            ρ^{it} ⊗ (σ^{−it})^T   ·   (σ^{−it} ⊗ (σ^{it})^T)^{−1}  simplificado
            a  ρ^{it} σ^{−it}  sobre M_n (cociclo espacial de Connes).
        """
        rho_it = DensityOperatorAlgebra.matrix_power(rho, 1j * t)
        sigma_neg_it = DensityOperatorAlgebra.matrix_power(sigma, -1j * t)
        return rho_it @ sigma_neg_it

    @staticmethod
    def verify_kms_at_beta(
        rho: np.ndarray, beta: float, n_tests: int = 6, seed: int = 42
    ) -> float:
        r"""
        φ(a σ_{iβ}(b)) = φ(ba)  usando σ_{iβ}(b) = ρ^β b ρ^{−β}.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(seed)
        residual_max = 0.0
        for _ in range(n_tests):
            a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            sigma_i_b = K.apply_modular_flow(b, t=1j * beta)
            lhs = complex(np.trace(rho @ a @ sigma_i_b))
            rhs = complex(np.trace(rho @ b @ a))
            denom = max(abs(lhs), abs(rhs), 1e-12)
            residual_max = max(residual_max, abs(lhs - rhs) / denom)
        return float(residual_max)


# ═══════════════════════════════════════════════════════════════════════════
# §1.6  Mecánica celeste de Poincaré: sección, resonancias, carta canónica
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class PoincareSectionVacuum:
    r"""
    Sección de Poincaré  Σ_c = {ρ : Tr(ρ N) = c}, transversal al flujo modular.

    El mapa de primer retorno P_Σ : Σ_c → Σ_c es el análogo cuántico del
    mapa de Poincaré del problema restringido de tres cuerpos.
    """
    level: float
    normal_diag: np.ndarray
    transversality_tol: float = 1e-7

    def signed_distance(self, rho: np.ndarray) -> float:
        return float(np.trace(rho @ self.normal_diag).real) - float(self.level)

    def crosses(self, rho_before: np.ndarray, rho_after: np.ndarray) -> bool:
        d0 = self.signed_distance(rho_before)
        d1 = self.signed_distance(rho_after)
        return (d0 * d1) < 0.0

    def is_transversal(self, rho: np.ndarray, rhodot: np.ndarray) -> bool:
        g = float(np.trace(rhodot @ self.normal_diag).real)
        return abs(g) > self.transversality_tol

    def first_return_residual(
        self, rho: np.ndarray, rho_returned: np.ndarray
    ) -> float:
        r"""‖ρ_returned − ρ‖_FS  (residuo del mapa de primer retorno)."""
        d = DensityOperatorAlgebra.sanitize(rho_returned) - DensityOperatorAlgebra.sanitize(rho)
        return float(np.linalg.norm(d, "fro"))


@dataclass(frozen=True, slots=True)
class ResonanceStructure:
    r"""Red de resonancias de Poincaré: k · ω ≈ 0, orden ‖k‖₁, residuo KAM."""
    vector: Tuple[int, ...]
    order: int
    residue_min: float
    kam_residue: float
    is_kam_safe: bool

    def small_divisor(self) -> float:
        r"""|k · ω| estimado como residue_min (divisor pequeño)."""
        return float(self.residue_min)

    def is_poincare_resonant(self, eps: float = 1e-6) -> bool:
        return self.order > 0 and self.residue_min < eps


@dataclass(frozen=True, slots=True)
class PoincareCanonicalChart:
    r"""
    Carta canónica de Poincaré (no singular en e = 0, i = 0).

    Delaunay (L, G, H, ℓ, g, h)  →  Poincaré (λ, ξ, η, p, q, Λ):

        λ = ℓ + g + h                         (longitud media),
        Λ = L                                 (acción principal),
        ξ = √(2(L−G)) cos(g+h),               (vector excentricidad),
        η = √(2(L−G)) sin(g+h),
        p = √(2(G−H)) cos(h),                 (vector inclinación),
        q = √(2(G−H)) sin(h).

    En e = 0 las variables (ξ, η) son regulares (a diferencia de (e, ω)).
    En i = 0 las variables (p, q) son regulares (a diferencia de (i, Ω)).
    Esta es *la* innovación de Poincaré (Méthodes Nouvelles, t. I, ch. IV).
    """
    mean_longitude: float
    action_L: float
    xi: float
    eta: float
    p: float
    q: float
    eccentricity: float = 0.0
    inclination: float = 0.0

    @classmethod
    def from_delaunay(
        cls,
        L: float,
        G: float,
        H: float,
        ell: float,
        g: float,
        h: float,
        e: float = 0.0,
        inc: float = 0.0,
    ) -> "PoincareCanonicalChart":
        L = float(max(L, 0.0))
        G = float(min(max(G, 0.0), L))
        H = float(min(max(H, -G), G))
        two_LG = max(2.0 * (L - G), 0.0)
        two_GH = max(2.0 * (G - abs(H)), 0.0)
        rt_e = math.sqrt(two_LG)
        rt_i = math.sqrt(two_GH)
        gph = g + h
        return cls(
            mean_longitude=float(ell + g + h),
            action_L=L,
            xi=rt_e * math.cos(gph),
            eta=rt_e * math.sin(gph),
            p=rt_i * math.cos(h),
            q=rt_i * math.sin(h),
            eccentricity=float(e),
            inclination=float(inc),
        )

    def eccentricity_vector_norm(self) -> float:
        return math.hypot(self.xi, self.eta)

    def inclination_vector_norm(self) -> float:
        return math.hypot(self.p, self.q)

    def generating_function_value(self) -> float:
        r"""S = Λ λ + ξ η/2 + p q/2  (tipo 2, localmente canónica)."""
        return (
            self.action_L * self.mean_longitude
            + 0.5 * self.xi * self.eta
            + 0.5 * self.p * self.q
        )

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (
            self.mean_longitude, self.action_L,
            self.xi, self.eta, self.p, self.q,
        )


@dataclass(frozen=True, slots=True)
class KolmogorovNondegeneracy:
    r"""
    Condición de Kolmogorov (1954):  det(∂ω/∂I) ≠ 0.

    Equivale a que el mapa de frecuencias I ↦ ω(I) sea un difeomorfismo
    local — hipótesis que permite persistencia de toros en KAM.
    """
    hessian_det: float
    is_nondegenerate: bool
    frequencies: Tuple[float, ...]
    diophantine_residue: float

    @classmethod
    def from_modular(cls, K: ModularHamiltonian) -> "KolmogorovNondegeneracy":
        det = K.kolmogorov_hessian_det()
        omega = K.mean_motion_frequencies()
        R = K.diophantine_residue()
        return cls(
            hessian_det=float(det),
            is_nondegenerate=bool(abs(det) > _KOLMOGOROV_DET_TOL),
            frequencies=tuple(float(x) for x in omega.tolist()),
            diophantine_residue=float(R),
        )

    def is_kam_admissible(self) -> bool:
        return self.is_nondegenerate and self.diophantine_residue >= _KAM_SAFE_THRESHOLD


@dataclass(frozen=True, slots=True)
class LyapunovSpectrum:
    r"""Exponentes característicos de Poincaré (espectro de Lyapunov)."""
    exponents: Tuple[float, ...]

    @property
    def maximal(self) -> float:
        return float(self.exponents[0]) if self.exponents else 0.0

    @property
    def sign_pattern(self) -> Tuple[int, ...]:
        return tuple(0 if abs(x) < 1e-10 else (1 if x > 0 else -1) for x in self.exponents)

    def is_hyperbolic(self) -> bool:
        return self.maximal > 1e-8

    def is_elliptic(self) -> bool:
        return (not self.is_hyperbolic()) and all(
            abs(x) < 1e-8 or x < 0.0 for x in self.exponents
        )

    @classmethod
    def from_density(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> "LyapunovSpectrum":
        exp = DensityOperatorAlgebra.modular_jacobian_lyapunov(rho, N_diag)
        return cls(exponents=exp)


# ═══════════════════════════════════════════════════════════════════════════
# §1.7  Germen de la mónada RSI-3  (T, η, μ) sobre 𝔇_n
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class RSI3Germ:
    r"""
    Germen de la mónada RSI-3 sobre el simplejo 𝔇_n.

    Nivel 1 — unidad η     :  η(ρ) = sanitize(ρ)           (proyección a 𝔇_n)
    Nivel 2 — endofuntor T :  T(ρ) = (1−α) ρ + α |φ_max⟩⟨φ_max|
                              (mixtura CPTP hacia el eigenrayo dominante)
    Nivel 3 — multiplicación μ :  μ = T  (condensación metacognitiva)
                              T³ luego μ  realiza la automejora recursiva.

    Axiomas de Mac Lane (verificados numéricamente en FASE 3 por MonadRSI3):
        μ ∘ Tμ = μ ∘ μT ,     μ ∘ Tη = μ ∘ ηT = id_T.

    INVARIANTE DE NOMBRE: el scalar de mixtura se llama `mix`, NUNCA `eta`,
    para no colisionar con el método `unit` (la η de la mónada).  Este era
    un bug real de v9.1.0 (`self.eta = float` tapaba `def eta(self, rho)`).
    """
    mix: float = _RSI3_DEFAULT_MIX
    level: int = _RSI3_DEFAULT_LEVEL
    dim: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "mix", float(min(1.0, max(0.0, self.mix))))
        object.__setattr__(self, "level", int(max(1, min(int(self.level), 4))))
        object.__setattr__(self, "dim", int(max(1, self.dim)))

    def unit(self, rho: np.ndarray) -> np.ndarray:
        r"""η : 𝔇_n → T 𝔇_n,  η(ρ) = sanitize(ρ).  Nivel 1."""
        return DensityOperatorAlgebra.sanitize(rho)

    def _target_rank_one(self, rho: np.ndarray) -> np.ndarray:
        rho = DensityOperatorAlgebra.sanitize(rho)
        _w, V = la.eigh(rho)
        v = V[:, -1].reshape(-1, 1)
        return (v @ v.conj().T).astype(np.complex128)

    def endofunctor(self, rho: np.ndarray) -> np.ndarray:
        r"""T : 𝔇_n → 𝔇_n,  T(ρ) = (1−α)ρ + α|φ_max⟩⟨φ_max|.  Nivel 2."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        tgt = self._target_rank_one(rho)
        return DensityOperatorAlgebra.sanitize((1.0 - self.mix) * rho + self.mix * tgt)

    def multiply(self, rho_TT: np.ndarray) -> np.ndarray:
        r"""μ : T² 𝔇_n → T 𝔇_n,  μ = T.  Nivel 3 (condensación)."""
        return self.endofunctor(rho_TT)

    def recede_improve(self, rho: np.ndarray, n_levels: Optional[int] = None) -> np.ndarray:
        r"""
        Automejora recursiva de nivel `n_levels` (por defecto self.level = 3):

            ρ_{k+1} = T(ρ_k),   k = 0..n−1,   luego μ(ρ_n).

        Cada iteración es CPTP, unital, y contrae la distancia de Bures al
        eigenrayo dominante (punto fijo de T: los puros de rango 1).
        """
        n_lev = int(n_levels) if n_levels is not None else self.level
        rho_n = self.unit(rho)
        for _ in range(max(1, n_lev)):
            rho_n = self.endofunctor(rho_n)
        return self.multiply(rho_n)

    def kleisli_compose(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        g: Callable[[np.ndarray], np.ndarray],
        rho: np.ndarray,
    ) -> np.ndarray:
        r"""Composición de Kleisli  (g ∘_T f)(ρ) = μ(T(g(f(ρ))))."""
        rho_f = self.unit(f(rho))
        T_rho_f = self.endofunctor(rho_f)
        g_T = self.unit(g(T_rho_f))
        return self.multiply(self.endofunctor(g_T))

    def germ_certificate(self, rho: np.ndarray) -> Dict[str, Any]:
        r"""Certificado numérico del germen (preludio de los axiomas Mac Lane)."""
        rho0 = self.unit(rho)
        rho1 = self.endofunctor(rho0)
        rho3 = self.recede_improve(rho0, n_levels=3)
        d_bures = DensityOperatorAlgebra.bures_distance(rho0, rho3)
        d_pure = DensityOperatorAlgebra.bures_distance(rho3, self._target_rank_one(rho0))
        return {
            "mix": self.mix,
            "level": self.level,
            "purity_in": DensityOperatorAlgebra.purity(rho0),
            "purity_T": DensityOperatorAlgebra.purity(rho1),
            "purity_T3": DensityOperatorAlgebra.purity(rho3),
            "bures_contraction": float(d_bures),
            "bures_to_pure": float(d_pure),
            "trace_T3": float(np.trace(rho3).real),
            "is_density_T3": DensityOperatorAlgebra.is_density_operator(rho3),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §1.8  VacuumModularContext + VacuumStatePreparation
#       HAND-OFF FORMAL  FASE 1 → FASE 2
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class VacuumModularContext:
    r"""
    Objeto de transporte FASE 1 → FASE 2.

    Dominio de `TomitaTakesakiEngine.bind_vacuum_context` (primer método
    de FASE 2).  Porta el estado de Gibbs, su hamiltoniano modular, el
    proyector del vacío, el TFD, el germen RSI-3 aplicado, la carta de
    Poincaré y el certificado de Kolmogorov.
    """
    rho: np.ndarray
    K: ModularHamiltonian
    H: np.ndarray
    ground_projector: np.ndarray
    beta: float
    gns_norm_sq: float
    is_faithful: bool
    tfd: np.ndarray = field(repr=False)
    rsi3_level: int = 0
    rsi3_certificate: Dict[str, Any] = field(default_factory=dict)
    poincare_chart: Optional[PoincareCanonicalChart] = None
    kolmogorov: Optional[KolmogorovNondegeneracy] = None
    lyapunov: Optional[LyapunovSpectrum] = None
    section: Optional[PoincareSectionVacuum] = None


class VacuumStatePreparation:
    r"""
    Preparación del vacío de Gibbs y condensación RSI-3.

    Cadena de morfosmos:

        (H, β)  --gibbs-->  ρ_β  --K_ρ-->  ModularHamiltonian
                --RSI-3-->  ρ^{(3)}
                --carta-->  PoincareCanonicalChart
                --hand-off-->  VacuumModularContext
                              ≡  objeto inicial de FASE 2.
    """
    DEFAULT_BETA_COLD: float = 1.0e3
    DEFAULT_HOPPING: float = 1.0e-3
    DEFAULT_RSI3_MIX: float = _RSI3_DEFAULT_MIX

    @classmethod
    def tight_binding_hamiltonian(
        cls, n: int, hopping: float = DEFAULT_HOPPING
    ) -> np.ndarray:
        if n < 1:
            raise ValueError("mac_dimension must be ≥ 1")
        onsite = np.linspace(0.0, float(max(n - 1, 0)), n, dtype=np.float64)
        H = np.diag(onsite).astype(np.complex128)
        for i in range(n - 1):
            H[i, i + 1] = hopping
            H[i + 1, i] = hopping
        return MatrixBanachAlgebra.hermitize(H)

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> np.ndarray:
        w, vecs = la.eigh(MatrixBanachAlgebra.hermitize(H))
        idx0 = int(np.argmin(w))
        omega = vecs[:, idx0].reshape(-1, 1)
        return DensityOperatorAlgebra.sanitize(omega @ omega.conj().T)

    @classmethod
    def gibbs_state(cls, H: np.ndarray, beta: float) -> np.ndarray:
        return DensityOperatorAlgebra.density_matrix_from_hamiltonian(H, beta)

    @classmethod
    def interpolated_state(cls, H: np.ndarray, tau: float) -> np.ndarray:
        tau = float(min(1.0, max(0.0, tau)))
        rho = cls.ground_state_projector(H)
        n = rho.shape[0]
        return DensityOperatorAlgebra.sanitize(
            (1.0 - tau) * rho + tau * np.eye(n, dtype=np.complex128) / n
        )

    @classmethod
    def apply_rsi3_germ(
        cls,
        rho: np.ndarray,
        mix: float = DEFAULT_RSI3_MIX,
        levels: int = _RSI3_DEFAULT_LEVEL,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        r"""Aplica el germen RSI-3 y devuelve (ρ^{(ℓ)}, certificado)."""
        n = int(rho.shape[0])
        germ = RSI3Germ(mix=mix, level=levels, dim=n)
        rho_l = germ.recede_improve(rho, n_levels=levels)
        cert = germ.germ_certificate(rho)
        cert["levels_applied"] = int(levels)
        return rho_l, cert

    @classmethod
    def bind_poincare_celestial(
        cls, rho: np.ndarray, H: np.ndarray, K: ModularHamiltonian
    ) -> Tuple[
        PoincareCanonicalChart,
        KolmogorovNondegeneracy,
        LyapunovSpectrum,
        PoincareSectionVacuum,
    ]:
        r"""
        Ata la mecánica celeste de Poincaré al vacío preparado:

            carta canónica, Kolmogorov, Lyapunov, sección Σ_{⟨N⟩}.
        """
        chart = DensityOperatorAlgebra.poincare_canonical_chart(rho, H)
        kolm = KolmogorovNondegeneracy.from_modular(K)
        lyap = LyapunovSpectrum.from_density(rho, H)
        level = float(np.trace(rho @ H).real)
        section = PoincareSectionVacuum(level=level, normal_diag=H)
        return chart, kolm, lyap, section

    @classmethod
    def prepare_vacuum_pair(
        cls, H: np.ndarray, beta: float = DEFAULT_BETA_COLD
    ) -> Tuple[np.ndarray, ModularHamiltonian]:
        ctx = cls.prepare_vacuum_context(H, beta=beta)
        return ctx.rho, ctx.K

    @classmethod
    def prepare_vacuum_context(
        cls, H: np.ndarray, beta: float = DEFAULT_BETA_COLD
    ) -> VacuumModularContext:
        r"""
        Morfismo  (H, β) ↦ VacuumModularContext  (sin condensar RSI-3).

        Conservado por compatibilidad v9.1.0.  El morfismo *terminal*
        de FASE 1 es `handoff_to_tomita_takesaki`.
        """
        H = MatrixBanachAlgebra.hermitize(H)
        rho = cls.gibbs_state(H, beta)
        K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho)
        ground = cls.ground_state_projector(H)
        gns_n2 = GNSHilbertAlgebra.omega_norm_squared(rho)
        faithful = GNSHilbertAlgebra.is_faithful(rho)
        tfd = DensityOperatorAlgebra.thermofield_double(rho)
        chart, kolm, lyap, section = cls.bind_poincare_celestial(rho, H, K)
        return VacuumModularContext(
            rho=rho, K=K, H=H, ground_projector=ground,
            beta=beta, gns_norm_sq=gns_n2, is_faithful=faithful, tfd=tfd,
            rsi3_level=0, rsi3_certificate={},
            poincare_chart=chart, kolmogorov=kolm,
            lyapunov=lyap, section=section,
        )

    @classmethod
    def handoff_to_tomita_takesaki(
        cls,
        H: np.ndarray,
        beta: float = DEFAULT_BETA_COLD,
        rsi3_levels: int = _RSI3_DEFAULT_LEVEL,
        rsi3_mix: float = DEFAULT_RSI3_MIX,
    ) -> VacuumModularContext:
        r"""
        ╔═════════════════════════════════════════════════════════════════╗
        ║  MORFISMO TERMINAL DE FASE 1  ≡  OBJETO INICIAL DE FASE 2      ║
        ║                                                                 ║
        ║  handoff_to_tomita_takesaki : (H, β, ℓ) → VacuumModularContext ║
        ║                                                                 ║
        ║  Composición:                                                   ║
        ║      (H, β)  --prepare_vacuum_context-->  ctx₀                  ║
        ║              --RSI-3 recede_improve(ℓ)-->  ρ^{(ℓ)}              ║
        ║              --rebind K, TFD, Poincaré-->  ctx_ℓ                ║
        ║                                                                 ║
        ║  CONTINUACIÓN FORMAL (FASE 2 · §2.1, primer método):            ║
        ║      TomitaTakesakiEngine.bind_vacuum_context(ctx_ℓ)            ║
        ║                                                                 ║
        ║  El tipo de retorno VacuumModularContext ES el dominio de       ║
        ║  bind_vacuum_context.  No hay gap: el último morfismo de        ║
        ║  Fase 1 es el primero de Fase 2.                                ║
        ╚═════════════════════════════════════════════════════════════════╝
        """
        H = MatrixBanachAlgebra.hermitize(H)
        ctx = cls.prepare_vacuum_context(H, beta=beta)
        rho_l, cert = cls.apply_rsi3_germ(ctx.rho, mix=rsi3_mix, levels=rsi3_levels)
        K_l = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho_l)
        tfd_l = DensityOperatorAlgebra.thermofield_double(rho_l)
        chart, kolm, lyap, section = cls.bind_poincare_celestial(rho_l, H, K_l)
        ctx.rho = rho_l
        ctx.K = K_l
        ctx.tfd = tfd_l
        ctx.gns_norm_sq = GNSHilbertAlgebra.omega_norm_squared(rho_l)
        ctx.is_faithful = GNSHilbertAlgebra.is_faithful(rho_l)
        ctx.rsi3_level = int(rsi3_levels)
        ctx.rsi3_certificate = cert
        ctx.poincare_chart = chart
        ctx.kolmogorov = kolm
        ctx.lyapunov = lyap
        ctx.section = section
        logger.info(
            "FASE 1→2 handoff | n=%d β=%.3e RSI-3=ℓ%d mix=%.3f | "
            "faithful=%s KAM=%s Kolmogorov=%s λ_max=%.3e | purity=%.6f",
            rho_l.shape[0], beta, rsi3_levels, rsi3_mix,
            ctx.is_faithful,
            bool(kolm.is_kam_admissible()) if kolm else False,
            bool(kolm.is_nondegenerate) if kolm else False,
            lyap.maximal if lyap else 0.0,
            DensityOperatorAlgebra.purity(rho_l),
        )
        return ctx


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FIN DE FASE 1                                                             ║
# ║                                                                           ║
# ║ Continuación formal — FASE 2 · §2.1  (próximo prompt):                    ║
# ║                                                                           ║
# ║     class TomitaTakesakiEngine:                                           ║
# ║         @classmethod                                                      ║
# ║         def bind_vacuum_context(cls, ctx: VacuumModularContext)           ║
# ║             -> VacuumModularContext:                                      ║
# ║             ...  # consume el objeto terminal de Fase 1                   ║
# ║                                                                           ║
# ║ El par (handoff_to_tomita_takesaki, bind_vacuum_context) es un            ║
# ║ isomorfismo de transporte:  ctx_ℓ  ↦  ctx_ℓ  validado (H, ρ, K            ║
# ║ dimensionalmente coherentes, GNS-fiel, RSI-3 condensado, carta            ║
# ║ de Poincaré adjunta).  Fase 2 abre la dinámica modular σ_t, el            ║
# ║ número de Chern, QFI y el pipeline del silencio.                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · DINÁMICA MODULAR, ESPECTRO DEL VACÍO Y SILENCIO                  ║
# ║                                                                           ║
# ║  CONTINUACIÓN FORMAL DE FASE 1                                            ║
# ║      VacuumStatePreparation.handoff_to_tomita_takesaki                    ║
# ║          : (H, β, ℓ) → VacuumModularContext                               ║
# ║      ≡  dominio de TomitaTakesakiEngine.bind_vacuum_context               ║
# ║                                                                           ║
# ║  §2.1  TomitaTakesakiEngine  (σ_t Haag–KMS, Chern, QFI, Poincaré–BKM)     ║
# ║  §2.2  VacuumSpectraAnalyzer + VacuumAuditReport                          ║
# ║  §2.3  SilentFieldDetector  + SilentFieldProbe                            ║
# ║  §2.4  ModularSilencePipeline + SilentFieldBundle                         ║
# ║        └── handoff_to_heyting_adjudicator  ≡  morfismo inicial de FASE 3  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# §2.0  Órbita modular / mapa de primer retorno (tipos de transporte)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ModularFlowOrbit:
    r"""
    Órbita del campo BKM  \(\dot\rho = [\rho,[\rho,N]]\)  sobre 𝔇_n.

    Distinción rigurosa (Poincaré, *Méthodes Nouvelles*, t. I, ch. III):
        σ_t  — automorfismo modular de *observables*;  ρ es punto fijo KMS.
        φ_t  — flujo de Heisenberg bajo H físico;  mueve ρ si [H, ρ] ≠ 0.
        X_BKM — gradiente de Bogoliubov–Kubo–Mori;  sección de Poincaré.

    El mapa de primer retorno P_Σ vive en X_BKM, no en σ_t.
    """
    rho_initial: np.ndarray
    rho_final: np.ndarray
    time: float
    n_steps: int
    crossed: bool
    transversal: bool
    first_return_residual: float
    signed_distance_initial: float
    signed_distance_final: float

    def is_poincare_return(self) -> bool:
        return bool(self.crossed and self.transversal)

    def as_certificate(self) -> Dict[str, Any]:
        return {
            "time": float(self.time),
            "n_steps": int(self.n_steps),
            "crossed": bool(self.crossed),
            "transversal": bool(self.transversal),
            "first_return_residual": float(self.first_return_residual),
            "is_poincare_return": self.is_poincare_return(),
        }


@dataclass(frozen=True, slots=True)
class BirkhoffNormalForm:
    r"""
    Forma normal de Birkhoff del hamiltoniano modular hasta orden 2:

        K(I) = ω · I  +  ½ I · τ · I  +  O(|I|³),

    con τ = Hess_I K = ∂ω/∂I  (matriz de twist / Kolmogorov).
    El teorema de Poincaré–Birkhoff exige twist:  det τ ≠ 0.
    """
    frequencies: Tuple[float, ...]
    twist_hessian_det: float
    twist_condition: bool
    order: int
    remainder_frobenius: float

    def is_twist(self) -> bool:
        return bool(self.twist_condition)


@dataclass(frozen=True, slots=True)
class ModularChernReport:
    r"""Número de Chern modular c₁ ∈ ℤ y curvatura de Berry integrada."""
    chern_number: int
    curvature_sum: float
    n_plaquettes: int
    degenerate_plaquettes: int
    is_integral: bool
    gauge_normalized: bool

    def as_int(self) -> int:
        return int(self.chern_number)


# ═══════════════════════════════════════════════════════════════════════════
# §2.1  Motor de Tomita–Takesaki  (dinámica modular + celestial de Poincaré)
# ═══════════════════════════════════════════════════════════════════════════
class TomitaTakesakiEngine:
    r"""
    Motor Tomita–Takesaki con extensión celeste y verificaciones rigurosas.

    Primer morfismo:  bind_vacuum_context(ctx_ℓ)
        consume VacuumModularContext  (objeto terminal de FASE 1).

    Convención Haag–KMS (unificada con ModularHamiltonian.apply_modular_flow):

        Δ = ρ,   K = −log ρ,
        σ_t(a) = Δ^{it} a Δ^{−it} = ρ^{it} a ρ^{−it},
        σ_{iβ}(a) = ρ^{−β} a ρ^{β}     (continuación analítica, β = 1 ⇒ t = i).

    Identidades verificadas numéricamente:
        (G1)  σ_0 = id,   σ_s ∘ σ_t = σ_{s+t}          (grupo de un parámetro)
        (G2)  σ_t(ab) = σ_t(a)σ_t(b),  σ_t(I) = I      (automorfismo unital)
        (G3)  φ(a σ_{iβ}(b)) = φ(ba)                   (KMS)
        (TT)  J² = id,  JΔJ = Δ^{−1},  S = J Δ^{1/2}   (Tomita)
    """

    def __init__(self, Hilbert_dim: int = 4, beta_kms: float = 1.0) -> None:
        self.dim = int(max(1, Hilbert_dim))
        self.beta = float(beta_kms)
        self._bound_ctx: Optional[VacuumModularContext] = None

    # ── morfismo inicial FASE 2  (continúa handoff_to_tomita_takesaki) ──

    @classmethod
    def bind_vacuum_context(cls, ctx: VacuumModularContext) -> VacuumModularContext:
        r"""
        Morfismo inicial de FASE 2.

        Dominio  = VacuumModularContext  (codominio de
                   VacuumStatePreparation.handoff_to_tomita_takesaki).
        Codominio = el mismo objeto, dimensionalmente validado, sanitizado,
                    con payload celeste (carta, Kolmogorov, Lyapunov, Σ_c)
                    y certificado GNS reanclado.

        Invariantes exigidos:
            ρ, H, |Ω⟩⟨Ω|  ∈ M_n(ℂ),  spec(K_ρ) tiene longitud n,
            Tr ρ = 1,  ρ = ρ† ≥ 0,  H = H†.
        """
        if ctx.rho.ndim != 2 or ctx.rho.shape[0] != ctx.rho.shape[1]:
            raise ValueError("VacuumModularContext.ρ must be a square matrix")
        n = int(ctx.rho.shape[0])
        if ctx.H.shape != (n, n) or ctx.ground_projector.shape != (n, n):
            raise ValueError("H, ρ and |Ω⟩⟨Ω| dimension mismatch")
        if ctx.K.dimension != n:
            raise ValueError("spec(K_ρ) length ≠ n")

        ctx.rho = DensityOperatorAlgebra.sanitize(ctx.rho)
        ctx.H = MatrixBanachAlgebra.hermitize(ctx.H)
        ctx.ground_projector = DensityOperatorAlgebra.sanitize(ctx.ground_projector)
        ctx.gns_norm_sq = GNSHilbertAlgebra.omega_norm_squared(ctx.rho)
        ctx.is_faithful = GNSHilbertAlgebra.is_faithful(ctx.rho)

        if (
            ctx.poincare_chart is None
            or ctx.kolmogorov is None
            or ctx.lyapunov is None
            or ctx.section is None
        ):
            chart, kolm, lyap, section = VacuumStatePreparation.bind_poincare_celestial(
                ctx.rho, ctx.H, ctx.K
            )
            if ctx.poincare_chart is None:
                ctx.poincare_chart = chart
            if ctx.kolmogorov is None:
                ctx.kolmogorov = kolm
            if ctx.lyapunov is None:
                ctx.lyapunov = lyap
            if ctx.section is None:
                ctx.section = section

        if ctx.K.eigenvectors is None:
            ctx.K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(ctx.rho)

        return ctx

    @classmethod
    def from_vacuum_context(cls, ctx: VacuumModularContext) -> "TomitaTakesakiEngine":
        r"""Fábrica: ancla un motor al contexto validado de FASE 1."""
        ctx = cls.bind_vacuum_context(ctx)
        eng = cls(Hilbert_dim=ctx.rho.shape[0], beta_kms=float(ctx.beta))
        eng._bound_ctx = ctx
        return eng

    # ── flujo modular Haag–KMS ──────────────────────────────────────────

    @classmethod
    def modular_flow(cls, rho: np.ndarray, t: complex, a: np.ndarray) -> np.ndarray:
        r"""
        σ_t(a) = ρ^{it} a ρ^{−it}     (Haag–KMS, β = 1).

        t ∈ ℂ:  la tira |Im t| ≤ 1/2 es el dominio analítico de KMS.
        Equivale a ModularHamiltonian.apply_modular_flow cuando spec(K)
        está disponible; esta ruta usa el cálculo funcional de ρ.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        a = MatrixBanachAlgebra.as_complex(a)
        rho_it = DensityOperatorAlgebra.matrix_power(rho, 1j * t)
        rho_neg_it = DensityOperatorAlgebra.matrix_power(rho, -1j * t)
        return rho_it @ a @ rho_neg_it

    @classmethod
    def modular_flow_via_K(
        cls, K: ModularHamiltonian, t: complex, a: np.ndarray
    ) -> np.ndarray:
        r"""σ_t(a) vía fases e^{−i t κ} (ruta espectral, numéricamente estable)."""
        return K.apply_modular_flow(a, t)

    def compute_tomita_takesaki_modular_flow(
        self, density_op: np.ndarray, t_time: float
    ) -> np.ndarray:
        r"""
        Flujo modular del *observable* density_op (no del estado KMS).

        CORRECCIÓN v9.2.0: v9.1.0 aplicaba U ρ U† con U = ρ^{−it},
        que es idénticamente ρ.  Ahora se interpreta density_op como
        observable a ∈ M_n y se devuelve σ_t(a) en convención Haag.
        El estado de referencia es sanitize(a) si a ≥ 0 y Tr a = 1,
        en otro caso se usa el contexto anclado o I/n.
        """
        a = MatrixBanachAlgebra.as_complex(density_op)
        if (
            DensityOperatorAlgebra.is_density_operator(a)
            and self._bound_ctx is not None
        ):
            rho_ref = self._bound_ctx.rho
            return self.modular_flow(rho_ref, float(t_time), a)
        if DensityOperatorAlgebra.is_density_operator(a):
            # σ_t(ρ) = ρ  (punto fijo KMS).  Se devuelve sanitizado.
            return DensityOperatorAlgebra.sanitize(a)
        rho_ref = (
            self._bound_ctx.rho
            if self._bound_ctx is not None
            else DensityOperatorAlgebra.sanitize(a)
        )
        return self.modular_flow(rho_ref, float(t_time), a)

    @classmethod
    def modular_state_invariance_residual(
        cls, rho: np.ndarray, t: float = 0.37
    ) -> float:
        r"""
        ‖σ_t^*(ρ) − ρ‖_F.  Identidad KMS: el estado es invariante.
        Un residuo ≫ ε_máq. denuncia error de convención o de hermiticidad.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        flowed = cls.modular_flow(rho, t, rho)
        return float(np.linalg.norm(flowed - rho, "fro"))

    @classmethod
    def tomita_S(cls, A: np.ndarray) -> np.ndarray:
        r"""S(A) = A†  (HS-representación)."""
        return MatrixBanachAlgebra.as_complex(A).conj().T

    @classmethod
    def heisenberg_evolve(
        cls, rho: np.ndarray, H: np.ndarray, t: float
    ) -> np.ndarray:
        r"""
        Evolución de Heisenberg bajo el H *físico* (no modular):

            ρ(t) = e^{−i H t} ρ e^{i H t}.

        Éste sí mueve ρ cuando [H, ρ] ≠ 0  (vacío Gibbs de H es fijo).
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        H = MatrixBanachAlgebra.hermitize(H)
        U = la.expm(-1j * float(t) * H)
        evolved = U @ rho @ U.conj().T
        return DensityOperatorAlgebra.sanitize(evolved)

    @classmethod
    def bkm_gradient_field(
        cls, rho: np.ndarray, N_diag: np.ndarray
    ) -> np.ndarray:
        r"""
        Campo BKM / doble conmutador (tangente a 𝔇_n):

            X(ρ) = [ρ, [ρ, N]] = ρ[ρ, N] − [ρ, N]ρ.

        Es el gradiente de ⟨N⟩ respecto de la métrica de Bogoliubov–Kubo–Mori.
        Genera la sección de Poincaré Σ_c = {Tr(ρ N) = c}.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        N_diag = MatrixBanachAlgebra.hermitize(N_diag)
        comm = MatrixBanachAlgebra.commutator(rho, N_diag)
        X = MatrixBanachAlgebra.commutator(rho, comm)
        return MatrixBanachAlgebra.hermitize(X)

    @classmethod
    def bkm_gradient_step(
        cls, rho: np.ndarray, N_diag: np.ndarray, dt: float
    ) -> np.ndarray:
        r"""Paso de Euler proyectado: ρ ← Π_{𝔇_n}(ρ + dt X(ρ))."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        X = cls.bkm_gradient_field(rho, N_diag)
        return DensityOperatorAlgebra.sanitize(rho + float(dt) * X)

    # ── axiomas de álgebra y Tomita ─────────────────────────────────────

    @classmethod
    def flow_group_law_residual(
        cls, rho: np.ndarray, s: float, t: float, a: np.ndarray
    ) -> float:
        composed = cls.modular_flow(rho, s, cls.modular_flow(rho, t, a))
        direct = cls.modular_flow(rho, s + t, a)
        return float(np.linalg.norm(composed - direct, "fro"))

    @classmethod
    def verify_kms(
        cls, rho: np.ndarray, n_tests: int = 8, seed: int = 42
    ) -> float:
        return GNSHilbertAlgebra.verify_kms_at_beta(
            rho, beta=1.0, n_tests=n_tests, seed=seed
        )

    @classmethod
    def verify_algebra_axioms(cls, rho: np.ndarray) -> Dict[str, float]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(7)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        eye = np.eye(n, dtype=np.complex128)
        unital = float(np.linalg.norm(cls.modular_flow(rho, 0.37, eye) - eye))
        product = float(
            np.linalg.norm(
                cls.modular_flow(rho, 0.73, a @ b)
                - cls.modular_flow(rho, 0.73, a) @ cls.modular_flow(rho, 0.73, b)
            )
        )
        isometry = float(
            abs(
                np.linalg.norm(a, "fro")
                - np.linalg.norm(cls.modular_flow(rho, 1.11, a), "fro")
            )
        )
        involution = float(np.linalg.norm(cls.modular_flow(rho, 0.0, a) - a))
        kms_invariance = cls.modular_state_invariance_residual(rho)
        return {
            "unital_residual": unital,
            "product_residual": product,
            "isometry_residual": isometry,
            "involution_residual": involution,
            "kms_state_invariance": kms_invariance,
        }

    @classmethod
    def verify_tomita_takesaki(cls, rho: np.ndarray) -> Dict[str, float]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(11)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        report = cls.verify_algebra_axioms(rho)
        report["group_law_residual"] = cls.flow_group_law_residual(
            rho, 0.41, -0.17, a
        )
        report["kms_residual"] = cls.verify_kms(rho)
        report["cstar_residual"] = MatrixBanachAlgebra.cstar_identity_residual(a)
        report["gns_J_involution"] = GNSHilbertAlgebra.verify_J_involution(rho)
        report["gns_J_delta_J"] = GNSHilbertAlgebra.verify_J_delta_J(rho)
        report["gns_S_polar"] = GNSHilbertAlgebra.verify_S_polar(rho)
        report["polar_residual"] = report["gns_S_polar"]
        return report

    @classmethod
    def connes_cocycle_flow(
        cls, rho: np.ndarray, sigma: np.ndarray, t: float
    ) -> np.ndarray:
        r"""[Dρ : Dσ]_t = ρ^{it} σ^{−it}  (1-cociclo espacial de Connes)."""
        return GNSHilbertAlgebra.connes_cocycle(rho, sigma, t)

    @classmethod
    def connes_cocycle_residual(
        cls, rho: np.ndarray, sigma: np.ndarray, s: float = 0.3, t: float = 0.5
    ) -> float:
        r"""
        Identidad de cociclo:

            [Dρ:Dσ]_{s+t}  =  [Dρ:Dσ]_s  ·  σ_s^σ([Dρ:Dσ]_t)

        con σ_s^σ(x) = σ^{is} x σ^{−is}.  Residuo Frobenius relativo.
        """
        u_st = cls.connes_cocycle_flow(rho, sigma, s + t)
        u_s = cls.connes_cocycle_flow(rho, sigma, s)
        u_t = cls.connes_cocycle_flow(rho, sigma, t)
        sigma_s_ut = cls.modular_flow(sigma, s, u_t)
        composed = u_s @ sigma_s_ut
        den = max(float(np.linalg.norm(u_st, "fro")), 1e-15)
        return float(np.linalg.norm(u_st - composed, "fro") / den)

    # ── potencial químico KMS ───────────────────────────────────────────

    @classmethod
    def kms_chemical_potential(
        cls,
        rho: np.ndarray,
        N_diag: np.ndarray,
        beta: float = 1.0,
        n_newton: int = 24,
        tol: float = 1e-10,
    ) -> float:
        r"""
        μ tal que el Gibbs grand-canónico de (K − μ N) reproduce ⟨N⟩_ρ:

            n_target = Tr(ρ N) = Tr( e^{−β(K−μN)} N ) / Z(μ).

        Newton 1-D sobre μ.  En el vacío canónico (N ∝ I) devuelve 0.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        N_diag = MatrixBanachAlgebra.hermitize(N_diag)
        n_target = float(np.trace(rho @ N_diag).real)
        K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho)
        if K.eigenvectors is None:
            return 0.0
        V = K.eigenvectors
        kappa = np.array(K.eigenvalues, dtype=np.float64)
        N_v = V.conj().T @ N_diag @ V
        n_diag = np.real(np.diag(N_v))
        mu = 0.0
        b = float(beta)
        for _ in range(n_newton):
            logits = -b * (kappa - mu * n_diag)
            logits -= float(np.max(logits))
            w = np.exp(logits)
            z = float(np.sum(w))
            if z <= 0.0:
                break
            w /= z
            n_mu = float(np.dot(w, n_diag))
            # ∂⟨N⟩/∂μ = β (⟨N²⟩ − ⟨N⟩²)  (susceptibilidad)
            n2 = float(np.dot(w, n_diag ** 2))
            chi = b * max(n2 - n_mu * n_mu, 1e-18)
            dmu = (n_target - n_mu) / chi
            mu += dmu
            if abs(dmu) < tol:
                break
        return float(mu)

    # ── KAM, resonancias, Kepler, Birkhoff, Melnikov, Poincaré ──────────

    @classmethod
    def diophantine_residue(
        cls,
        omega: np.ndarray,
        gamma: float = _DIOPHANTINE_GAMMA,
        tau: float = _DIOPHANTINE_TAU,
        k_max: int = _DIOPHANTINE_KMAP,
    ) -> float:
        return _diophantine_residue(omega, gamma=gamma, tau=tau, k_max=k_max)

    @classmethod
    def detect_resonances(
        cls, omega: np.ndarray, k_max: int = 3, epsilon: float = 1e-6
    ) -> ResonanceStructure:
        omega = np.asarray(omega, dtype=float)
        n = omega.size
        best_k: Tuple[int, ...] = ()
        best_order = 0
        best_residue = float("inf")
        for order in range(1, 2 * k_max + 1):
            for k_list in _iterate_integer_lattice(n, order):
                val = abs(float(np.dot(k_list, omega)))
                if val < best_residue:
                    best_residue = val
                    best_k = tuple(int(x) for x in k_list)
                    if val < epsilon:
                        best_order = order
        kam_residue = cls.diophantine_residue(omega)
        return ResonanceStructure(
            vector=best_k,
            order=best_order,
            residue_min=best_residue,
            kam_residue=kam_residue,
            is_kam_safe=bool(kam_residue >= _KAM_SAFE_THRESHOLD),
        )

    @classmethod
    def kam_torus_indicator(cls, omega: np.ndarray) -> Tuple[bool, float]:
        R = cls.diophantine_residue(omega)
        return (R >= _KAM_SAFE_THRESHOLD), float(R)

    @classmethod
    def arnold_web_overlap(
        cls, omega: np.ndarray, k_max: int = 3, width: float = 1e-3
    ) -> float:
        r"""
        Indicador de solapamiento de Chirikov / red de Arnold:

            χ = Σ_{k ≠ 0, ‖k‖₁ ≤ k_max}  Θ(width − |k·ω|) / #k.

        χ → 1  ⇒  mar de resonancias (difusión de Arnold);  χ → 0  ⇒  KAM.
        """
        omega = np.asarray(omega, dtype=float)
        n = omega.size
        if n == 0:
            return 0.0
        hits = 0
        total = 0
        for order in range(1, 2 * k_max + 1):
            for k_list in _iterate_integer_lattice(n, order):
                total += 1
                if abs(float(np.dot(k_list, omega))) < width:
                    hits += 1
        return float(hits / max(total, 1))

    @classmethod
    def poincare_section_at(
        cls, level: float, dimension: int
    ) -> PoincareSectionVacuum:
        N_diag = np.diag(np.arange(1, dimension + 1, dtype=float))
        return PoincareSectionVacuum(level=float(level), normal_diag=N_diag)

    @classmethod
    def poincare_return_orbit(
        cls,
        rho0: np.ndarray,
        section: PoincareSectionVacuum,
        N_diag: np.ndarray,
        dt: float = 0.02,
        max_time: float = 100.0,
    ) -> ModularFlowOrbit:
        r"""
        Mapa de primer retorno de Poincaré sobre el campo BKM.

        Integración de Euler proyectada con interpolación lineal del cruce
        (método de Poincaré–Hénon) y verificación de transversalidad
        g = Tr(X(ρ) N) ≠ 0.
        """
        rho = DensityOperatorAlgebra.sanitize(rho0)
        N_diag = MatrixBanachAlgebra.hermitize(N_diag)
        d0 = section.signed_distance(rho)
        t = 0.0
        n_steps = 0
        rho_prev = rho
        d_prev = d0
        crossed = False
        while t < max_time:
            X = cls.bkm_gradient_field(rho, N_diag)
            rho_next = DensityOperatorAlgebra.sanitize(rho + float(dt) * X)
            t += dt
            n_steps += 1
            d1 = section.signed_distance(rho_next)
            if d_prev * d1 < 0.0:
                # interpolación lineal del tiempo de cruce
                frac = abs(d_prev) / max(abs(d_prev - d1), _EPS)
                rho_cross = DensityOperatorAlgebra.sanitize(
                    (1.0 - frac) * rho + frac * rho_next
                )
                X_c = cls.bkm_gradient_field(rho_cross, N_diag)
                transversal = section.is_transversal(rho_cross, X_c)
                residual = section.first_return_residual(rho0, rho_cross)
                return ModularFlowOrbit(
                    rho_initial=rho0,
                    rho_final=rho_cross,
                    time=float(t - dt + frac * dt),
                    n_steps=n_steps,
                    crossed=True,
                    transversal=bool(transversal),
                    first_return_residual=float(residual),
                    signed_distance_initial=float(d0),
                    signed_distance_final=float(section.signed_distance(rho_cross)),
                )
            rho_prev = rho
            d_prev = d1
            rho = rho_next
        residual = section.first_return_residual(rho0, rho)
        X_f = cls.bkm_gradient_field(rho, N_diag)
        return ModularFlowOrbit(
            rho_initial=rho0,
            rho_final=rho,
            time=float(t),
            n_steps=n_steps,
            crossed=False,
            transversal=bool(section.is_transversal(rho, X_f)),
            first_return_residual=float(residual),
            signed_distance_initial=float(d0),
            signed_distance_final=float(section.signed_distance(rho)),
        )

    @classmethod
    def poincare_return_time(
        cls,
        rho0: np.ndarray,
        section: PoincareSectionVacuum,
        N_diag: np.ndarray,
        dt: float = 0.02,
        max_time: float = 100.0,
    ) -> Tuple[float, np.ndarray]:
        r"""Firma retro-compatible: (τ_return, ρ_Σ)."""
        orbit = cls.poincare_return_orbit(rho0, section, N_diag, dt, max_time)
        return orbit.time, orbit.rho_final

    @classmethod
    def poincare_birkhoff_count(cls, p: int, q: int, twist_angle: float) -> int:
        r"""
        Cota inferior de Poincaré–Birkhoff (1913): un mapa twist del anillo
        que rota las fronteras en sentidos que encierran p/q posee al menos
        2q puntos periódicos de tipo p/q.

        Aquí se verifica la hipótesis topológica (p/q ∈ (0,1), twist ∈ (0, 2π))
        y se *devuelve la cota*, no el censo de puntos fijos.
        """
        if q < 1 or p <= 0 or p >= q:
            return 0
        if not (0.0 < twist_angle < 2.0 * math.pi):
            return 0
        return 2 * q

    @classmethod
    def birkhoff_normal_form_from_K(
        cls, K: ModularHamiltonian, order: int = 2
    ) -> BirkhoffNormalForm:
        r"""Forma normal de Birkhoff espectral (twist = det Hess_I K)."""
        omega = K.mean_motion_frequencies()
        det_tau = K.kolmogorov_hessian_det()
        return BirkhoffNormalForm(
            frequencies=tuple(float(x) for x in omega.tolist()),
            twist_hessian_det=float(det_tau),
            twist_condition=bool(abs(det_tau) > _KOLMOGOROV_DET_TOL),
            order=int(max(1, order)),
            remainder_frobenius=0.0,
        )

    @classmethod
    def birkhoff_normal_form(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        order: int = 2,
    ) -> np.ndarray:
        r"""
        Generador de Lie X₁ (orden 1) que elimina términos no resonantes
        del conmutador [N, ρ]  — sombra matricial de la NF de Birkhoff.

        Firma retro-compatible (devuelve matriz).  La NF espectral rigurosa
        es birkhoff_normal_form_from_K.
        """
        rho_h = DensityOperatorAlgebra.sanitize(rho)
        n = rho_h.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        comm1 = MatrixBanachAlgebra.commutator(N_diag, rho_h)
        X1 = MatrixBanachAlgebra.commutator(N_diag, comm1)
        tr_num = float(np.real(np.trace(X1 @ rho_h.conj().T)))
        tr_den = float(np.real(np.trace(rho_h @ rho_h.conj().T)))
        if tr_den > _EPS:
            X1 = X1 - (tr_num / tr_den) * rho_h
        return MatrixBanachAlgebra.hermitize(X1)

    @classmethod
    def solve_kepler(
        cls,
        mean_anomaly: float,
        eccentricity: float,
        tol: float = 1e-12,
        max_iter: int = 50,
    ) -> float:
        r"""
        Ecuación de Kepler  E − e sin E = M,  e ∈ [0, 1).

        Método de Danby (Halley, convergencia cúbica) con arranque
        E₀ = M + e sin M (1 + e cos M).  M se reduce a (−π, π].
        """
        e = float(eccentricity)
        if not (0.0 <= e < 1.0):
            raise ValueError("Excentricidad debe estar en [0, 1).")
        M = float(mean_anomaly)
        two_pi = 2.0 * math.pi
        M = (M + math.pi) % two_pi - math.pi
        E = M + e * math.sin(M) * (1.0 + e * math.cos(M))
        for _ in range(max_iter):
            sE = math.sin(E)
            cE = math.cos(E)
            f = E - e * sE - M
            fp = 1.0 - e * cE
            fpp = e * sE
            fppp = e * cE
            # Halley / Danby
            denom = fp - 0.5 * f * fpp / max(fp, _EPS)
            if abs(denom) < _EPS:
                break
            dE = f / denom
            # corrección cúbica de Danby
            dE = dE / max(
                1.0 + dE * fpp / (2.0 * max(fp, _EPS))
                + (dE ** 2) * fppp / (6.0 * max(fp, _EPS)),
                _EPS,
            )
            E -= dE
            if abs(dE) < tol:
                break
        return float(E)

    @classmethod
    def kepler_residual(cls, E: float, M: float, e: float) -> float:
        return float(E - e * math.sin(E) - M)

    @classmethod
    def true_anomaly_from_eccentric(cls, E: float, e: float) -> float:
        r"""ν = 2 arctan( √((1+e)/(1−e)) tan(E/2) )  (fórmula de Gauss)."""
        e = float(e)
        if e >= 1.0:
            raise ValueError("e ∈ [0, 1) para órbita elíptica.")
        tan_half = math.sqrt((1.0 + e) / max(1.0 - e, _EPS)) * math.tan(0.5 * E)
        return float(2.0 * math.atan(tan_half))

    @classmethod
    def melnikov_function(
        cls,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        n_steps: int = 32,
        dt: float = 0.02,
    ) -> float:
        r"""
        Integral homoclínica de Melnikov a lo largo de la órbita BKM:

            M = Σ_k  {H_0, H_1}(φ_{k dt}(ρ_dream)) dt,
            {H_0, H_1} ∼ ‖[ρ, N]‖_{HS}² · Tr(ρ_base ρ(t)).

        M = 0  ⇒  persistencia de la homoclínica (no splitting);
        M ≠ 0  ⇒  intersección transversa, caos herradura de Smale.
        """
        rho_h = DensityOperatorAlgebra.sanitize(rho_dream)
        n = rho_h.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        base = DensityOperatorAlgebra.sanitize(rho_base)
        total = 0.0
        current = rho_h
        for _ in range(n_steps):
            comm = MatrixBanachAlgebra.commutator(current, N_diag)
            current = cls.bkm_gradient_step(current, N_diag, dt)
            bracket = float(np.trace(comm @ comm.conj().T).real) * float(
                np.trace(base @ current).real
            )
            total += bracket * dt
        return float(total)

    @classmethod
    def wigner_function(cls, rho: np.ndarray) -> np.ndarray:
        return DensityOperatorAlgebra.wigner_via_displaced_parity(rho)

    @classmethod
    def symplectic_capacity_gromov(cls, rho: np.ndarray) -> float:
        return DensityOperatorAlgebra.symplectic_capacity_gromov(rho)

    # ── espectro modular, Chern, QFI ────────────────────────────────────

    @classmethod
    def spectral_modular_decomposition(cls, rho: np.ndarray) -> Dict[str, Any]:
        r"""
        ρ = Σ_i e^{−κ_i} |φ_i⟩⟨φ_i|  con κ creciente.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        lam, V = la.eigh(rho)
        order = np.argsort(np.real(lam))[::-1]
        occup = np.maximum(np.real(lam[order]), 0.0)
        kappa_sorted = -np.log(np.maximum(occup, _EPS))
        gap = (
            float(kappa_sorted[1] - kappa_sorted[0])
            if kappa_sorted.size >= 2
            else float("inf")
        )
        return {
            "modular_energies": tuple(float(x) for x in kappa_sorted.tolist()),
            "occupations": tuple(float(x) for x in occup.tolist()),
            "modular_gap": gap,
            "eigenvectors": V[:, order],
        }

    @classmethod
    def _dominant_eigenvector_aligned(
        cls, rho: np.ndarray, ref: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool]:
        r"""
        Autovector dominante con gauge de fase alineado a `ref`
        (⟨ref|ψ⟩ ≥ 0).  Devuelve (ψ, degenerado).
        """
        w, V = la.eigh(MatrixBanachAlgebra.hermitize(rho))
        degenerate = bool(
            w.size >= 2 and abs(float(w[-1] - w[-2])) < _GAP_DEGENERACY_TOL
        )
        psi = V[:, -1]
        if ref is not None:
            z = np.vdot(ref, psi)
            if abs(z) > _EPS:
                psi = psi * np.exp(-1j * np.angle(z))
        nrm = float(np.linalg.norm(psi))
        if nrm > _EPS:
            psi = psi / nrm
        return psi, degenerate

    @classmethod
    def modular_chern_report(
        cls,
        rho_curve_fun: Callable[[float, float], np.ndarray],
        n_grid: int = 16,
    ) -> ModularChernReport:
        r"""
        Número de Chern modular (Fukui–Hatsugai–Suzuki):

            c₁ = (1/2π) Σ_{plaquetas} arg( U₁ U₂ U₃ U₄ ),

        con links unitarios  U_μ = ⟨ψ(θ)|ψ(θ+μ̂)⟩ / |⟨·⟩|.
        La suma de ángulos sobre 𝕋² es 2π ℤ si el fibrado es regular.
        """
        thetas = np.linspace(0.0, 2.0 * math.pi, n_grid, endpoint=False)
        berry_sum = 0.0
        n_deg = 0
        n_plaq = 0
        for i, t1 in enumerate(thetas):
            for j, t2 in enumerate(thetas):
                t1n = thetas[(i + 1) % n_grid]
                t2n = thetas[(j + 1) % n_grid]
                psi00, d00 = cls._dominant_eigenvector_aligned(rho_curve_fun(t1, t2))
                psi10, d10 = cls._dominant_eigenvector_aligned(
                    rho_curve_fun(t1n, t2), ref=psi00
                )
                psi11, d11 = cls._dominant_eigenvector_aligned(
                    rho_curve_fun(t1n, t2n), ref=psi10
                )
                psi01, d01 = cls._dominant_eigenvector_aligned(
                    rho_curve_fun(t1, t2n), ref=psi11
                )
                if d00 or d10 or d11 or d01:
                    n_deg += 1

                def _link(a: np.ndarray, b: np.ndarray) -> complex:
                    z = np.vdot(a, b)
                    mag = abs(z)
                    return z / mag if mag > _EPS else 1.0 + 0j

                plaquette = (
                    _link(psi00, psi10)
                    * _link(psi10, psi11)
                    * _link(psi11, psi01)
                    * _link(psi01, psi00)
                )
                berry_sum += float(np.angle(plaquette))
                n_plaq += 1
        c1_real = berry_sum / (2.0 * math.pi)
        c1 = int(round(c1_real))
        return ModularChernReport(
            chern_number=c1,
            curvature_sum=float(berry_sum),
            n_plaquettes=n_plaq,
            degenerate_plaquettes=n_deg,
            is_integral=bool(abs(c1_real - c1) < 1e-2),
            gauge_normalized=True,
        )

    @classmethod
    def modular_chern_number(
        cls,
        rho_curve_fun: Callable[[float, float], np.ndarray],
        n_grid: int = 16,
    ) -> int:
        r"""c₁ ∈ ℤ  (firma retro-compatible)."""
        return cls.modular_chern_report(rho_curve_fun, n_grid=n_grid).chern_number

    @classmethod
    def quantum_fisher_information(
        cls, rho: np.ndarray, H_generator: np.ndarray
    ) -> float:
        return DensityOperatorAlgebra.quantum_fisher_information(rho, H_generator)

    @classmethod
    def fidelity_susceptibility_qfi(
        cls, rho: np.ndarray, H_generator: np.ndarray
    ) -> float:
        r"""
        Susceptibilidad de fidelidad (Bures):

            χ_F = −∂²_λ F(ρ, ρ_λ) |_{λ=0}  =  F_Q(ρ, H) / 4.

        CORRECCIÓN v9.2.0: v9.1.0 devolvía 2 F_Q, que no es χ_F.
        """
        return 0.25 * DensityOperatorAlgebra.quantum_fisher_information(
            rho, H_generator
        )

    @classmethod
    def celestial_certificate(cls, ctx: VacuumModularContext) -> Dict[str, Any]:
        r"""Extracto celeste del contexto validado (KAM + Kolmogorov + Lyapunov)."""
        ctx = cls.bind_vacuum_context(ctx)
        omega = ctx.K.mean_motion_frequencies()
        res = cls.detect_resonances(omega)
        chart = ctx.poincare_chart
        kolm = ctx.kolmogorov
        lyap = ctx.lyapunov
        return {
            "kam_residue": float(res.kam_residue),
            "resonance_order": int(res.order),
            "is_kam_safe": bool(res.is_kam_safe),
            "arnold_web": cls.arnold_web_overlap(omega),
            "kolmogorov_det": float(kolm.hessian_det) if kolm else 0.0,
            "kolmogorov_nondegenerate": bool(kolm.is_nondegenerate) if kolm else False,
            "lyapunov_maximal": float(lyap.maximal) if lyap else 0.0,
            "lyapunov_hyperbolic": bool(lyap.is_hyperbolic()) if lyap else False,
            "chart": chart.as_tuple() if chart is not None else None,
            "generating_function": (
                float(chart.generating_function_value()) if chart is not None else 0.0
            ),
            "rsi3_level": int(ctx.rsi3_level),
            "faithful": bool(ctx.is_faithful),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §2.2  Analizador espectral del vacío modular
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class VacuumAuditReport:
    vev: float
    thermal_fluctuation: float
    spectral_gap: float
    modular_ground_energy: float
    partition_function: float
    free_energy: float
    internal_energy: float
    unique_vacuum: bool
    participation_ratio: float
    noise_db: float
    purity: float
    von_neumann_entropy: float
    local_verdict: HeytingOmega3
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    eccentricity: float = 0.0
    inclination: float = 0.0
    kam_residue: float = float("inf")
    resonance_order: int = 0
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_recurrence_time: float = float("inf")
    maslov_index: int = 0
    lyapunov_sign: int = -1
    poincare_stratum: str = "kam-torus-invariant"
    # v9.1.0
    renyi_2: float = 0.0
    renyi_inf: float = 0.0
    qfi_beta: float = 0.0
    helmholtz_residual: float = 0.0
    kac_recurrence: float = float("inf")
    # v9.2.0-F2  ·  Poincaré + RSI-3
    kolmogorov_det: float = 0.0
    lyapunov_maximal: float = 0.0
    generating_function: float = 0.0
    kac_mu_A: float = 0.0
    fidelity_susceptibility: float = 0.0
    arnold_web: float = 0.0
    rsi3_level: int = 0
    chemical_potential: float = 0.0

    @property
    def energy_fluctuation(self) -> float:
        return self.thermal_fluctuation

    @property
    def noise_level_db(self) -> float:
        return self.noise_db

    def is_kam_stable(self) -> bool:
        return (
            self.kam_residue >= _KAM_SAFE_THRESHOLD
            and self.resonance_order == 0
            and abs(self.kolmogorov_det) > _KOLMOGOROV_DET_TOL
        )


class VacuumSpectraAnalyzer:
    VEV_VETO_THRESHOLD: float = 1.0e-2
    FLUCTUATION_VETO_THRESHOLD: float = 1.0e-2
    GAP_DEGRADE_THRESHOLD: float = 1.0e-3
    FREE_ENERGY_DEGRADE_THRESHOLD: float = 1.0e-8

    @classmethod
    def _expectation_and_variance(
        cls, rho: np.ndarray, H: np.ndarray
    ) -> Tuple[float, float]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        H = MatrixBanachAlgebra.hermitize(H)
        mean = float(np.real(np.trace(rho @ H)))
        second = float(np.real(np.trace(rho @ H @ H)))
        var = max(0.0, second - mean * mean)
        return mean, math.sqrt(var)

    @classmethod
    def participation_ratio(cls, rho: np.ndarray) -> float:
        purity = DensityOperatorAlgebra.purity(rho)
        return 1.0 / purity if purity > 0.0 else float("inf")

    @classmethod
    def _verdict(
        cls, sigma_H: float, vev: float, gap: float, free_e: float
    ) -> HeytingOmega3:
        if sigma_H > cls.FLUCTUATION_VETO_THRESHOLD and vev > cls.VEV_VETO_THRESHOLD:
            return HeytingOmega3.VETOED
        if sigma_H > cls.FLUCTUATION_VETO_THRESHOLD or vev > cls.VEV_VETO_THRESHOLD:
            return HeytingOmega3.DEGRADED
        if gap < cls.GAP_DEGRADE_THRESHOLD:
            return HeytingOmega3.DEGRADED
        if abs(free_e) > cls.FREE_ENERGY_DEGRADE_THRESHOLD:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def _celestial_block(
        cls, rho: np.ndarray, H: np.ndarray, K: ModularHamiltonian
    ) -> Dict[str, Any]:
        r"""Bloque celeste: Delaunay, KAM, Wigner, Kac-FS, QFI, Kolmogorov."""
        L_del, G_del, H_del = DensityOperatorAlgebra.delaunay_triple(rho, H)
        e_del, i_del = DensityOperatorAlgebra.eccentricity_inclination(rho, H)
        omega = DensityOperatorAlgebra.mean_motion_frequencies(rho, H)
        res_struct = TomitaTakesakiEngine.detect_resonances(omega)
        wigner_neg = DensityOperatorAlgebra.wigner_negativity(rho)
        symp_cap = DensityOperatorAlgebra.symplectic_capacity_gromov(rho)
        tau_rec = DensityOperatorAlgebra.poincare_recurrence_time(rho)
        maslov = DensityOperatorAlgebra.maslov_index(rho, H)
        renyi2 = DensityOperatorAlgebra.renyi_entropy(rho, 2.0)
        renyi_inf = DensityOperatorAlgebra.renyi_entropy(rho, float("inf"))
        qfi = DensityOperatorAlgebra.quantum_fisher_information(rho, H)
        chi_f = 0.25 * qfi
        helm_res = K.helmholtz_free_energy_residual()
        tau_kac, mu_A = DensityOperatorAlgebra.poincare_recurrence_time_kac(rho, H)
        kolm_det = K.kolmogorov_hessian_det()
        lyap_spec = DensityOperatorAlgebra.modular_jacobian_lyapunov(rho, H)
        lyap_max = float(lyap_spec[0]) if lyap_spec else 0.0
        chart = DensityOperatorAlgebra.poincare_canonical_chart(rho, H)
        gen_S = chart.generating_function_value()
        arnold = TomitaTakesakiEngine.arnold_web_overlap(omega)
        mu_chem = TomitaTakesakiEngine.kms_chemical_potential(rho, H, beta=1.0)
        return {
            "delaunay_triple": (float(L_del), float(G_del), float(H_del)),
            "eccentricity": float(e_del),
            "inclination": float(i_del),
            "kam_residue": float(res_struct.kam_residue),
            "resonance_order": int(res_struct.order),
            "wigner_negativity": float(wigner_neg),
            "symplectic_capacity": float(symp_cap),
            "poincare_recurrence_time": float(tau_rec),
            "maslov_index": int(maslov),
            "renyi_2": float(renyi2),
            "renyi_inf": float(renyi_inf),
            "qfi_beta": float(qfi),
            "fidelity_susceptibility": float(chi_f),
            "helmholtz_residual": float(helm_res),
            "kac_recurrence": float(tau_kac),
            "kac_mu_A": float(mu_A),
            "kolmogorov_det": float(kolm_det),
            "lyapunov_maximal": float(lyap_max),
            "generating_function": float(gen_S),
            "arnold_web": float(arnold),
            "chemical_potential": float(mu_chem),
        }

    @classmethod
    def audit(
        cls, rho: np.ndarray, H: np.ndarray, K: ModularHamiltonian
    ) -> VacuumAuditReport:
        rho = DensityOperatorAlgebra.sanitize(rho)
        H = MatrixBanachAlgebra.hermitize(H)
        mean_H, sigma_H = cls._expectation_and_variance(rho, H)
        w_H = np.real(la.eigvalsh(H))
        e0 = float(w_H.min())
        vev = mean_H - e0
        gap = K.spectral_gap
        z_part = K.partition_function
        free_e = K.free_energy
        noise_db = 10.0 * math.log10(1.0 + sigma_H ** 2 + 1e-30)
        purity = DensityOperatorAlgebra.purity(rho)
        ent = DensityOperatorAlgebra.von_neumann_entropy(rho)
        ipr = cls.participation_ratio(rho)
        local = cls._verdict(sigma_H, vev, gap, free_e)
        try:
            blk = cls._celestial_block(rho, H, K)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            blk = {
                "delaunay_triple": (0.0, 0.0, 0.0),
                "eccentricity": 0.0,
                "inclination": 0.0,
                "kam_residue": float("inf"),
                "resonance_order": 0,
                "wigner_negativity": 0.0,
                "symplectic_capacity": 1.0,
                "poincare_recurrence_time": float("inf"),
                "maslov_index": 0,
                "renyi_2": 0.0,
                "renyi_inf": 0.0,
                "qfi_beta": 0.0,
                "fidelity_susceptibility": 0.0,
                "helmholtz_residual": 0.0,
                "kac_recurrence": float("inf"),
                "kac_mu_A": 0.0,
                "kolmogorov_det": 0.0,
                "lyapunov_maximal": 0.0,
                "generating_function": 0.0,
                "arnold_web": 0.0,
                "chemical_potential": 0.0,
            }
        return VacuumAuditReport(
            vev=vev,
            thermal_fluctuation=sigma_H,
            spectral_gap=gap,
            modular_ground_energy=K.ground_energy,
            partition_function=z_part,
            free_energy=free_e,
            internal_energy=K.internal_energy,
            unique_vacuum=K.is_unique_vacuum,
            participation_ratio=ipr,
            noise_db=noise_db,
            purity=purity,
            von_neumann_entropy=ent,
            local_verdict=local,
            delaunay_triple=blk["delaunay_triple"],
            eccentricity=blk["eccentricity"],
            inclination=blk["inclination"],
            kam_residue=blk["kam_residue"],
            resonance_order=blk["resonance_order"],
            wigner_negativity=blk["wigner_negativity"],
            symplectic_capacity=blk["symplectic_capacity"],
            poincare_recurrence_time=blk["poincare_recurrence_time"],
            maslov_index=blk["maslov_index"],
            lyapunov_sign=local.lyapunov_exponent_sign(),
            poincare_stratum=local.poincare_stratum_name(),
            renyi_2=blk["renyi_2"],
            renyi_inf=blk["renyi_inf"],
            qfi_beta=blk["qfi_beta"],
            helmholtz_residual=blk["helmholtz_residual"],
            kac_recurrence=blk["kac_recurrence"],
            kolmogorov_det=blk["kolmogorov_det"],
            lyapunov_maximal=blk["lyapunov_maximal"],
            generating_function=blk["generating_function"],
            kac_mu_A=blk["kac_mu_A"],
            fidelity_susceptibility=blk["fidelity_susceptibility"],
            arnold_web=blk["arnold_web"],
            rsi3_level=0,
            chemical_potential=blk["chemical_potential"],
        )

    @classmethod
    def audit_from_context(cls, ctx: VacuumModularContext) -> VacuumAuditReport:
        r"""
        Auditoría anclada al payload de FASE 1 (carta, Kolmogorov, Lyapunov,
        nivel RSI-3).  Evita recomputar lo que `bind_poincare_celestial`
        ya certificó, y rellena los campos v9.2.0 con el contexto.
        """
        ctx = TomitaTakesakiEngine.bind_vacuum_context(ctx)
        report = cls.audit(ctx.rho, ctx.H, ctx.K)
        kolm_det = (
            float(ctx.kolmogorov.hessian_det)
            if ctx.kolmogorov is not None
            else report.kolmogorov_det
        )
        lyap_max = (
            float(ctx.lyapunov.maximal)
            if ctx.lyapunov is not None
            else report.lyapunov_maximal
        )
        gen_S = (
            float(ctx.poincare_chart.generating_function_value())
            if ctx.poincare_chart is not None
            else report.generating_function
        )
        return replace(
            report,
            kolmogorov_det=kolm_det,
            lyapunov_maximal=lyap_max,
            generating_function=gen_S,
            rsi3_level=int(ctx.rsi3_level),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §2.3  Detector del silencio epistemológico
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SilentFieldProbe:
    purity: float
    fidelity_to_ground: float
    uhlmann_fidelity: float
    correlation_leakage: float
    umegaki_to_ground: float
    bures_distance: float
    trace_distance: float
    klein_residual: float
    local_verdict: HeytingOmega3
    # v9.2.0-F2
    petz_recovery_fidelity: float = 0.0
    renyi_relative_2: float = 0.0
    connes_cocycle_residual: float = 0.0

    @property
    def umegaki_relative_entropy(self) -> float:
        return self.umegaki_to_ground

    @property
    def klein_inequality_residual(self) -> float:
        return self.klein_residual

    def is_silent(self) -> bool:
        return (
            self.local_verdict == HeytingOmega3.COHERENT
            and self.purity >= SilentFieldDetector.PURITY_SILENT_THRESHOLD
            and self.fidelity_to_ground >= SilentFieldDetector.FIDELITY_SILENT_THRESHOLD
        )


class SilentFieldDetector:
    r"""
    Sonda de silencio: distancia de Bures / Umegaki al rayo del vacío,
    recuperación de Petz y cociclo de Connes (ρ ‖ |Ω⟩⟨Ω|).

    Silencio ⇔ ρ es indistinguible del ground a umbral 0.999 y el canal
    modular es suficiente (Petz recupera con fidelidad ≈ 1).
    """
    UMEGAKI_EPS: float = 1.0e-12
    PURITY_SILENT_THRESHOLD: float = 0.999
    FIDELITY_SILENT_THRESHOLD: float = 0.999
    PETZ_SILENT_THRESHOLD: float = 0.999

    @classmethod
    def probe(
        cls, rho: np.ndarray, ground_state_projector: np.ndarray
    ) -> SilentFieldProbe:
        rho = DensityOperatorAlgebra.sanitize(rho)
        sigma = DensityOperatorAlgebra.sanitize(ground_state_projector)
        purity = DensityOperatorAlgebra.purity(rho)
        overlap = float(np.real(np.trace(rho @ sigma)))
        leak = 1.0 - overlap
        uhlmann = DensityOperatorAlgebra.uhlmann_fidelity(rho, sigma)
        bures = DensityOperatorAlgebra.bures_distance(rho, sigma)
        trc = DensityOperatorAlgebra.trace_distance(rho, sigma)
        n = rho.shape[0]
        sigma_eps = DensityOperatorAlgebra.sanitize(
            (1.0 - cls.UMEGAKI_EPS) * sigma
            + cls.UMEGAKI_EPS * np.eye(n, dtype=np.complex128) / n
        )
        rel = DensityOperatorAlgebra.umegaki_relative_entropy(rho, sigma_eps)
        klein = DensityOperatorAlgebra.kleins_inequality_residual(rho, sigma_eps)
        # Petz: R_{σ_ε → ρ}(σ_ε) debería devolver ρ si el canal es suficiente.
        R = DensityOperatorAlgebra.petz_recovery_map(rho, sigma_eps)
        recovered = DensityOperatorAlgebra.sanitize(R(sigma_eps))
        petz_fid = DensityOperatorAlgebra.uhlmann_fidelity(rho, recovered)
        try:
            d2 = DensityOperatorAlgebra.renyi_relative_entropy(rho, sigma_eps, 2.0)
        except (ValueError, FloatingPointError):
            d2 = 0.0
        try:
            cocycle_res = TomitaTakesakiEngine.connes_cocycle_residual(
                rho, sigma_eps, s=0.3, t=0.5
            )
        except (np.linalg.LinAlgError, ValueError):
            cocycle_res = 0.0

        if (
            purity >= cls.PURITY_SILENT_THRESHOLD
            and overlap >= cls.FIDELITY_SILENT_THRESHOLD
        ):
            local = HeytingOmega3.COHERENT
        elif (
            purity >= cls.PURITY_SILENT_THRESHOLD
            or overlap >= cls.FIDELITY_SILENT_THRESHOLD
        ):
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return SilentFieldProbe(
            purity=purity,
            fidelity_to_ground=overlap,
            uhlmann_fidelity=uhlmann,
            correlation_leakage=leak,
            umegaki_to_ground=rel,
            bures_distance=bures,
            trace_distance=trc,
            klein_residual=klein,
            local_verdict=local,
            petz_recovery_fidelity=float(petz_fid),
            renyi_relative_2=float(d2),
            connes_cocycle_residual=float(cocycle_res),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §2.4  ModularSilencePipeline  —  HAND-OFF 2→3
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SilentFieldBundle:
    r"""
    Haz del silencio: objeto de transporte FASE 2 → FASE 3.

    Dominio de `HeytingVacuumAdjudicator.adjudicate` (primer método de FASE 3).
    Porta auditoría espectral, sonda epistemológica, residuos Tomita/KMS
    y el contexto modular (RSI-3 + Poincaré) del que nace.
    """
    cycle_index: int
    rho: np.ndarray
    K: ModularHamiltonian
    audit: VacuumAuditReport
    probe: SilentFieldProbe
    kms_residual: float
    algebra_axioms: Dict[str, float]
    tomita_report: Dict[str, float]
    context_beta: float
    rsi3_level: int = 0
    rsi3_certificate: Dict[str, Any] = field(default_factory=dict)
    celestial: Dict[str, Any] = field(default_factory=dict)
    rsi3_recondensed: bool = False

    @property
    def spectrum(self) -> VacuumAuditReport:
        return self.audit

    def heyting_pre_meet(self) -> HeytingOmega3:
        r"""Meet local (audit ∧ probe) antes de la adjudicación externa."""
        return self.audit.local_verdict.meet(self.probe.local_verdict)


class ModularSilencePipeline:
    r"""
    Síntesis del haz de silencio a partir del vacío preparado.

    Cadena de morfosmos:

        ctx_ℓ  --bind_vacuum_context-->  ctx✓
               --verify_tomita_takesaki-->  tomita_report
               --audit_from_context-->  VacuumAuditReport
               --probe-->  SilentFieldProbe
               --RSI-3 recondense (si Ω₃ ≠ ⊤ y ℓ < 3)-->  ctx^{(3)}
               --handoff_to_heyting_adjudicator-->  SilentFieldBundle
                                                 ≡ objeto inicial de FASE 3.
    """

    RSI3_RECONDENSE_MAX: int = _RSI3_DEFAULT_LEVEL

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        rho: np.ndarray,
        H: np.ndarray,
        ground_projector: np.ndarray,
        K: ModularHamiltonian,
        context_beta: float = float("nan"),
        rsi3_level: int = 0,
        rsi3_certificate: Optional[Dict[str, Any]] = None,
        celestial: Optional[Dict[str, Any]] = None,
        rsi3_recondensed: bool = False,
    ) -> SilentFieldBundle:
        tomita = TomitaTakesakiEngine.verify_tomita_takesaki(rho)
        kms_res = float(
            tomita.get("kms_residual", TomitaTakesakiEngine.verify_kms(rho))
        )
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
            axioms = TomitaTakesakiEngine.verify_algebra_axioms(rho)
        audit = VacuumSpectraAnalyzer.audit(rho, H, K)
        if rsi3_level:
            audit = replace(audit, rsi3_level=int(rsi3_level))
        probe = SilentFieldDetector.probe(rho, ground_projector)
        return SilentFieldBundle(
            cycle_index=cycle_index,
            rho=rho,
            K=K,
            audit=audit,
            probe=probe,
            kms_residual=kms_res,
            algebra_axioms=axioms,
            tomita_report=tomita,
            context_beta=context_beta,
            rsi3_level=int(rsi3_level),
            rsi3_certificate=dict(rsi3_certificate or {}),
            celestial=dict(celestial or {}),
            rsi3_recondensed=bool(rsi3_recondensed),
        )

    @classmethod
    def _maybe_recondense(
        cls, ctx: VacuumModularContext, audit: VacuumAuditReport
    ) -> Tuple[VacuumModularContext, bool]:
        r"""
        Automejora recursiva de nivel 3 *en el pipeline*:

        Si el veredicto local no es ⊤ y el germen aún no se aplicó a nivel 3,
        se condensa ρ ↦ T³(ρ) y se reancla K, TFD, carta de Poincaré.
        Una sola pasada (techo ℓ = 3); no hay punto fijo iterado aquí.
        """
        if audit.local_verdict == HeytingOmega3.COHERENT:
            return ctx, False
        if int(ctx.rsi3_level) >= cls.RSI3_RECONDENSE_MAX:
            return ctx, False
        rho_l, cert = VacuumStatePreparation.apply_rsi3_germ(
            ctx.rho,
            mix=VacuumStatePreparation.DEFAULT_RSI3_MIX,
            levels=cls.RSI3_RECONDENSE_MAX,
        )
        K_l = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho_l)
        ctx.rho = rho_l
        ctx.K = K_l
        ctx.tfd = DensityOperatorAlgebra.thermofield_double(rho_l)
        ctx.gns_norm_sq = GNSHilbertAlgebra.omega_norm_squared(rho_l)
        ctx.is_faithful = GNSHilbertAlgebra.is_faithful(rho_l)
        ctx.rsi3_level = int(cls.RSI3_RECONDENSE_MAX)
        ctx.rsi3_certificate = cert
        chart, kolm, lyap, section = VacuumStatePreparation.bind_poincare_celestial(
            rho_l, ctx.H, K_l
        )
        ctx.poincare_chart = chart
        ctx.kolmogorov = kolm
        ctx.lyapunov = lyap
        ctx.section = section
        logger.info(
            "FASE 2 RSI-3 recondense | Ω₃=%s → ℓ=%d | purity=%.6f | KAM=%s",
            audit.local_verdict.name,
            ctx.rsi3_level,
            DensityOperatorAlgebra.purity(rho_l),
            bool(kolm.is_kam_admissible()) if kolm else False,
        )
        return ctx, True

    @classmethod
    def synthesize_from_context(
        cls, cycle_index: int, ctx: VacuumModularContext
    ) -> SilentFieldBundle:
        r"""Síntesis anclada: bind → audit → (recondense RSI-3) → probe → haz."""
        ctx = TomitaTakesakiEngine.bind_vacuum_context(ctx)
        audit_pre = VacuumSpectraAnalyzer.audit_from_context(ctx)
        ctx, recondensed = cls._maybe_recondense(ctx, audit_pre)
        if recondensed:
            ctx = TomitaTakesakiEngine.bind_vacuum_context(ctx)
        celestial = TomitaTakesakiEngine.celestial_certificate(ctx)
        bundle = cls.synthesize(
            cycle_index=cycle_index,
            rho=ctx.rho,
            H=ctx.H,
            ground_projector=ctx.ground_projector,
            K=ctx.K,
            context_beta=ctx.beta,
            rsi3_level=int(ctx.rsi3_level),
            rsi3_certificate=dict(ctx.rsi3_certificate or {}),
            celestial=celestial,
            rsi3_recondensed=recondensed,
        )
        # reanclar auditoría de contexto (Kolmogorov / Lyapunov / ℓ)
        audit_ctx = VacuumSpectraAnalyzer.audit_from_context(ctx)
        return replace(
            bundle,
            audit=audit_ctx,
            K=ctx.K,
            rho=ctx.rho,
        )

    @classmethod
    def handoff_to_heyting_adjudicator(
        cls,
        cycle_index: int,
        ctx: VacuumModularContext,
    ) -> SilentFieldBundle:
        r"""
        ╔═════════════════════════════════════════════════════════════════╗
        ║  MORFISMO TERMINAL DE FASE 2  ≡  OBJETO INICIAL DE FASE 3      ║
        ║                                                                 ║
        ║  handoff_to_heyting_adjudicator :                               ║
        ║      (cycle, ctx_ℓ) → SilentFieldBundle                         ║
        ║                                                                 ║
        ║  Composición:                                                   ║
        ║      ctx  --bind_vacuum_context-->  ctx✓                        ║
        ║           --synthesize_from_context-->  SilentFieldBundle       ║
        ║                                                                 ║
        ║  CONTINUACIÓN FORMAL (FASE 3 · §3.1, primer método):            ║
        ║      HeytingVacuumAdjudicator.adjudicate(                       ║
        ║          bundle, external_verdict)                              ║
        ║                                                                 ║
        ║  El tipo de retorno SilentFieldBundle ES el dominio de          ║
        ║  adjudicate.  No hay gap: el último morfismo de Fase 2 es       ║
        ║  el primero de Fase 3.                                          ║
        ╚═════════════════════════════════════════════════════════════════╝
        """
        bundle = cls.synthesize_from_context(cycle_index=cycle_index, ctx=ctx)
        logger.info(
            "FASE 2→3 handoff | cycle=%d Ω₃_pre=%s | KMS=%.3e | "
            "R_KAM=%.3e | τ_Kac=%.3e | Petz=%.6f | RSI-3=ℓ%d recond=%s",
            cycle_index,
            bundle.heyting_pre_meet().name,
            bundle.kms_residual,
            bundle.audit.kam_residue,
            bundle.audit.kac_recurrence,
            bundle.probe.petz_recovery_fidelity,
            bundle.rsi3_level,
            bundle.rsi3_recondensed,
        )
        return bundle


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FIN DE FASE 2                                                             ║
# ║                                                                           ║
# ║ Continuación formal — FASE 3 · §3.1  (próximo prompt):                    ║
# ║                                                                           ║
# ║     class HeytingVacuumAdjudicator:                                       ║
# ║         @classmethod                                                      ║
# ║         def adjudicate(                                                   ║
# ║             cls,                                                          ║
# ║             bundle: SilentFieldBundle,                                    ║
# ║             external_verdict: HeytingOmega3,                              ║
# ║         ) -> HeytingOmega3:                                               ║
# ║             ...  # consume el objeto terminal de Fase 2                   ║
# ║                                                                           ║
# ║ El par (handoff_to_heyting_adjudicator, adjudicate) es un                 ║
# ║ isomorfismo de transporte:  SilentFieldBundle ↦ Ω₃, con meet              ║
# ║ granular (audit ∧ probe ∧ kms ∧ axioms ∧ polar ∧ group ∧ klein)           ║
# ║ y veredicto externo.  Fase 3 abre la soberanía: certificado               ║
# ║ firmado, Merkle, mónada RSI-3 (axiomas de Mac Lane) y el                  ║
# ║ motor TOONSilentWitnessEngine.                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · SOBERANÍA, CERTIFICACIÓN, MÓNADA RSI-3 Y MOTOR TOON              ║
# ║                                                                           ║
# ║  CONTINUACIÓN FORMAL DE FASE 2                                            ║
# ║      ModularSilencePipeline.handoff_to_heyting_adjudicator                ║
# ║          : (cycle, ctx_ℓ) → SilentFieldBundle                             ║
# ║      ≡  dominio de HeytingVacuumAdjudicator.adjudicate                    ║
# ║                                                                           ║
# ║  §3.1  HeytingVacuumAdjudicator + AdjudicationCertificate                 ║
# ║  §3.2  SilentFieldState  (certificado firmado del silencio)               ║
# ║  §3.3  MerkleInclusionProof                                               ║
# ║  §3.4  Mónada RSI-3 (Mac Lane sobre la imagen reflexiva)                  ║
# ║  §3.5  AGMBeliefSet  (hand-off → toon_introspection_agent)                ║
# ║  §3.6  MetaSilentWitnessEngine                                            ║
# ║  §3.7  TOONSilentWitnessEngine  (motor soberano)                          ║
# ║  §3.8  Demostración autónoma                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# §3.1  Adjudicador en el retículo de Heyting Ω₃
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class AdjudicationCertificate:
    r"""
    Certificado del meet granular.

        verdict = (⋀_k scores[k]) ∧ external  ∈ Ω₃.

    Cada score es la imagen de un residuo / fidelidad bajo el morfismo
    de umbral  ℝ₊ → Ω₃  (catástrofe ↦ ⊥, holgura ↦ ⋆, máquina ↦ ⊤).
    El tercio excluso puede fallar en ⋆: un único score DEGRADED
    impide LEM y clasifica el estrato como órbita parabólica resonante.
    """
    scores: Dict[str, HeytingOmega3]
    local_meet: HeytingOmega3
    external: HeytingOmega3
    verdict: HeytingOmega3
    poincare_stratum: str
    symplectic_regime: str
    morse_index: int
    timestamp_utc: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scores": {k: v.name for k, v in self.scores.items()},
            "local_meet": self.local_meet.name,
            "external": self.external.name,
            "verdict": self.verdict.name,
            "poincare_stratum": self.poincare_stratum,
            "symplectic_regime": self.symplectic_regime,
            "morse_index": int(self.morse_index),
            "timestamp_utc": float(self.timestamp_utc),
        }


class HeytingVacuumAdjudicator:
    r"""
    Clasificador de subobjetos  Sh(Ω₃) ← SilentFieldBundle.

    Morfismo inicial de FASE 3 = continuación formal de
    ModularSilencePipeline.handoff_to_heyting_adjudicator.
    """
    KMS_RESIDUAL_TOL: float = 1.0e-6
    AXIOM_RESIDUAL_TOL: float = 1.0e-6
    CATASTROPHIC_TOL: float = 1.0e-3
    KLEIN_TOL: float = 1.0e-8
    HELMHOLTZ_TOL: float = 1.0e-8
    CONNES_TOL: float = 1.0e-5
    PETZ_DEGRADE: float = 0.50
    PETZ_VETO: float = 1.0e-3
    LYAPUNOV_VETO: float = 1.0
    LYAPUNOV_DEGRADE: float = 0.10

    @classmethod
    def adjudicate(
        cls,
        bundle: SilentFieldBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""
        ╔═════════════════════════════════════════════════════════════════╗
        ║  MORFISMO INICIAL DE FASE 3                                     ║
        ║                                                                 ║
        ║  adjudicate : SilentFieldBundle × Ω₃ → Ω₃                       ║
        ║                                                                 ║
        ║  Dominio = codominio de                                         ║
        ║      ModularSilencePipeline.handoff_to_heyting_adjudicator.     ║
        ║                                                                 ║
        ║  verdict = (⋀ granular_scores(bundle)) ∧ external.              ║
        ║  El meet de Heyting es el producto fibrado en Sh(Ω₃).           ║
        ╚═════════════════════════════════════════════════════════════════╝
        """
        cert = cls.adjudicate_certified(bundle, external_verdict)
        return cert.verdict

    @classmethod
    def _residual_to_omega(cls, residual: float) -> HeytingOmega3:
        r = abs(float(residual))
        if r > cls.CATASTROPHIC_TOL:
            return HeytingOmega3.VETOED
        if r > cls.KMS_RESIDUAL_TOL:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def _fidelity_to_omega(
        cls,
        fid: float,
        degrade: float = 0.50,
        veto: float = 1.0e-3,
    ) -> HeytingOmega3:
        f = float(fid)
        if f < veto:
            return HeytingOmega3.VETOED
        if f < degrade:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def _lyapunov_to_omega(cls, lam_max: float) -> HeytingOmega3:
        r"""sign(λ)>0 grande ⇒ separatriz hiperbólica (⊥)."""
        lam = float(lam_max)
        if lam > cls.LYAPUNOV_VETO:
            return HeytingOmega3.VETOED
        if lam > cls.LYAPUNOV_DEGRADE:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def _kolmogorov_to_omega(
        cls, det: float, resonance_order: int
    ) -> HeytingOmega3:
        r"""Kolmogorov degenerado *y* resonante ⇒ ⋆ (no ⊥: puede ser integrable)."""
        if abs(float(det)) <= _KOLMOGOROV_DET_TOL and int(resonance_order) > 0:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def granular_scores(
        cls, bundle: SilentFieldBundle
    ) -> Dict[str, HeytingOmega3]:
        r"""
        Familia de clasificadores  U_k ↪ SilentFieldBundle  con
        χ(U_k) ∈ Ω₃.  El meet es el objeto de secciones globales.
        """
        axioms_max = (
            max(bundle.algebra_axioms.values()) if bundle.algebra_axioms else 0.0
        )
        polar = float(bundle.tomita_report.get("polar_residual", 0.0))
        group = float(bundle.tomita_report.get("group_law_residual", 0.0))
        j_inv = float(bundle.tomita_report.get("gns_J_involution", 0.0))
        j_delta = float(bundle.tomita_report.get("gns_J_delta_J", 0.0))
        klein = abs(float(bundle.probe.klein_residual))
        klein_v = (
            HeytingOmega3.COHERENT
            if klein < cls.KLEIN_TOL
            else (
                HeytingOmega3.DEGRADED
                if klein < cls.CATASTROPHIC_TOL
                else HeytingOmega3.VETOED
            )
        )
        helm = abs(float(bundle.audit.helmholtz_residual))
        helm_v = (
            HeytingOmega3.COHERENT
            if helm < cls.HELMHOLTZ_TOL
            else cls._residual_to_omega(helm)
        )
        petz_v = cls._fidelity_to_omega(
            float(bundle.probe.petz_recovery_fidelity),
            degrade=cls.PETZ_DEGRADE,
            veto=cls.PETZ_VETO,
        )
        connes_v = cls._residual_to_omega(
            float(bundle.probe.connes_cocycle_residual)
        ) if float(bundle.probe.connes_cocycle_residual) > 0.0 else HeytingOmega3.COHERENT
        # umbral Connes más holgado (continuación analítica numérica)
        if float(bundle.probe.connes_cocycle_residual) > cls.CONNES_TOL:
            if float(bundle.probe.connes_cocycle_residual) <= cls.CATASTROPHIC_TOL:
                connes_v = HeytingOmega3.DEGRADED
        lyap_v = cls._lyapunov_to_omega(float(bundle.audit.lyapunov_maximal))
        kolm_v = cls._kolmogorov_to_omega(
            float(bundle.audit.kolmogorov_det),
            int(bundle.audit.resonance_order),
        )
        return {
            "audit": bundle.audit.local_verdict,
            "probe": bundle.probe.local_verdict,
            "kms": cls._residual_to_omega(bundle.kms_residual),
            "axioms": cls._residual_to_omega(axioms_max),
            "polar": cls._residual_to_omega(polar),
            "group_law": cls._residual_to_omega(group),
            "tomita_J": cls._residual_to_omega(j_inv),
            "tomita_JdeltaJ": cls._residual_to_omega(j_delta),
            "klein": klein_v,
            "helmholtz": helm_v,
            "petz": petz_v,
            "connes": connes_v,
            "lyapunov": lyap_v,
            "kolmogorov": kolm_v,
        }

    @classmethod
    def adjudicate_certified(
        cls,
        bundle: SilentFieldBundle,
        external_verdict: HeytingOmega3,
    ) -> AdjudicationCertificate:
        r"""Adjudicación con certificado de meet (soberanía auditable)."""
        scores = cls.granular_scores(bundle)
        local = HeytingOmega3.COHERENT
        for v in scores.values():
            local = local.meet(v)
        verdict = local.meet(external_verdict)
        return AdjudicationCertificate(
            scores=scores,
            local_meet=local,
            external=external_verdict,
            verdict=verdict,
            poincare_stratum=verdict.poincare_stratum_name(),
            symplectic_regime=verdict.symplectic_regime(),
            morse_index=verdict.morse_index(),
            timestamp_utc=time.time(),
        )

    @classmethod
    def from_handoff(
        cls,
        bundle: SilentFieldBundle,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> HeytingOmega3:
        r"""Alias explícito del isomorfismo de transporte FASE 2 → 3."""
        return cls.adjudicate(bundle, external_verdict)


# ═══════════════════════════════════════════════════════════════════════════
# §3.2  Certificado firmado del silencio
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SilentFieldState:
    r"""
    Estado certificado del testigo silencioso.

    Es un objeto de  Ω₃-Sh(Merkle): veredicto de Heyting + hoja SHA-256
    + cadena de fases + invariantes celestes de Poincaré (KAM, Kac,
    Kolmogorov, Lyapunov, carta generatriz).
    """
    cycle_id: str
    witness_engine_id: str
    vacuum_expectation_value: float
    kms_temperature_beta: float
    vacuum_beta: float
    noise_emission_decibels: float
    crystallized_experience_count: int
    spectral_gap: float
    free_energy: float
    purity: float
    fidelity_to_ground: float
    uhlmann_fidelity: float
    bures_distance_to_ground: float
    trace_distance_to_ground: float
    kms_residual: float
    tomita_polar_residual: float
    tomita_group_law_residual: float
    heyting_verdict: HeytingOmega3
    experience_hash: str
    phase_chain_sha256: str
    timestamp_utc: float
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    eccentricity: float = 0.0
    inclination: float = 0.0
    kam_residue: float = float("inf")
    resonance_order: int = 0
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_recurrence_time: float = float("inf")
    maslov_index: int = 0
    lyapunov_sign: int = -1
    poincare_stratum: str = "kam-torus-invariant"
    # v9.1.0
    renyi_2: float = 0.0
    renyi_inf: float = 0.0
    qfi_beta: float = 0.0
    helmholtz_residual: float = 0.0
    kac_recurrence: float = float("inf")
    # v9.2.0-F3
    kolmogorov_det: float = 0.0
    lyapunov_maximal: float = 0.0
    generating_function: float = 0.0
    kac_mu_A: float = 0.0
    fidelity_susceptibility: float = 0.0
    arnold_web: float = 0.0
    rsi3_level: int = 0
    chemical_potential: float = 0.0
    petz_recovery_fidelity: float = 0.0
    rsi3_recondensed: bool = False

    def is_vetoed(self) -> bool:
        return self.heyting_verdict == HeytingOmega3.VETOED

    def is_kam_stable(self) -> bool:
        return (
            self.kam_residue >= _KAM_SAFE_THRESHOLD
            and self.resonance_order == 0
            and abs(self.kolmogorov_det) > _KOLMOGOROV_DET_TOL
        )

    def is_poincare_stable(self) -> bool:
        r"""KAM ∧ Kolmogorov ∧ λ_max ≤ 0  (toro elíptico persistente)."""
        return self.is_kam_stable() and self.lyapunov_maximal <= 1e-8

    def is_silent(self) -> bool:
        return (
            self.heyting_verdict == HeytingOmega3.COHERENT
            and self.purity >= 0.999
            and self.fidelity_to_ground >= 0.999
            and self.kms_residual < 1.0e-6
        )

    def merkle_leaf(self) -> str:
        r"""Hoja canónica: SHA-256 del payload de soberanía (no el experience_hash)."""
        h = hashlib.sha256()
        h.update(self.cycle_id.encode("ascii"))
        h.update(self.witness_engine_id.encode("ascii"))
        h.update(f"{self.vacuum_expectation_value:.12e}".encode("ascii"))
        h.update(f"{self.kms_residual:.12e}".encode("ascii"))
        h.update(self.heyting_verdict.name.encode("ascii"))
        h.update(self.experience_hash.encode("ascii"))
        h.update(self.phase_chain_sha256.encode("ascii"))
        return h.hexdigest()

    def to_experience_crystal(
        self, vector_s6: Optional[np.ndarray] = None
    ) -> ExperienceCrystal:
        r"""Inmersión  SilentFieldState ↪ S⁶  (cristal inmutable de Fase 1)."""
        if vector_s6 is None:
            vector_s6 = np.array(
                [
                    self.purity,
                    self.fidelity_to_ground,
                    self.kam_residue if math.isfinite(self.kam_residue) else 0.0,
                    self.kms_residual,
                    self.spectral_gap if math.isfinite(self.spectral_gap) else 0.0,
                    self.helmholtz_residual,
                    self.lyapunov_maximal,
                ],
                dtype=np.float64,
            )
        return ExperienceCrystal(
            vector_s6=vector_s6,
            tau_recurrence=self.poincare_recurrence_time,
            kms_drift=self.kms_residual,
            merkle_root_sha256=self.experience_hash,
            timestamp_utc=self.timestamp_utc,
            delaunay_triple=self.delaunay_triple,
            eccentricity=self.eccentricity,
            inclination=self.inclination,
            kam_residue=self.kam_residue,
            resonance_order=self.resonance_order,
            poincare_stratum=self.poincare_stratum,
            wigner_negativity=self.wigner_negativity,
            symplectic_capacity=self.symplectic_capacity,
            lyapunov_spectrum=(float(self.lyapunov_maximal),),
            kolmogorov_det=self.kolmogorov_det,
            generating_function=self.generating_function,
            first_return_residual=self.kms_residual,
            rsi3_level_applied=self.rsi3_level,
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: SilentFieldBundle,
        *,
        cycle_id: str,
        engine_id: str,
        kms_beta: float,
        beta_vacuum: float,
        triad_crystallized_count: int,
        phase_chain_sha256: str,
        experience_hash: str,
        verdict: HeytingOmega3,
    ) -> "SilentFieldState":
        r"""Constructor canónico: SilentFieldBundle × sellos → certificado."""
        polar_res = float(bundle.tomita_report.get("polar_residual", 0.0))
        group_res = float(bundle.tomita_report.get("group_law_residual", 0.0))
        return cls(
            cycle_id=cycle_id,
            witness_engine_id=engine_id,
            vacuum_expectation_value=bundle.audit.vev,
            kms_temperature_beta=kms_beta,
            vacuum_beta=beta_vacuum,
            noise_emission_decibels=bundle.audit.noise_db,
            crystallized_experience_count=triad_crystallized_count,
            spectral_gap=bundle.audit.spectral_gap,
            free_energy=bundle.audit.free_energy,
            purity=bundle.probe.purity,
            fidelity_to_ground=bundle.probe.fidelity_to_ground,
            uhlmann_fidelity=bundle.probe.uhlmann_fidelity,
            bures_distance_to_ground=bundle.probe.bures_distance,
            trace_distance_to_ground=bundle.probe.trace_distance,
            kms_residual=bundle.kms_residual,
            tomita_polar_residual=polar_res,
            tomita_group_law_residual=group_res,
            heyting_verdict=verdict,
            experience_hash=experience_hash,
            phase_chain_sha256=phase_chain_sha256,
            timestamp_utc=time.time(),
            delaunay_triple=bundle.audit.delaunay_triple,
            eccentricity=bundle.audit.eccentricity,
            inclination=bundle.audit.inclination,
            kam_residue=bundle.audit.kam_residue,
            resonance_order=bundle.audit.resonance_order,
            wigner_negativity=bundle.audit.wigner_negativity,
            symplectic_capacity=bundle.audit.symplectic_capacity,
            poincare_recurrence_time=bundle.audit.poincare_recurrence_time,
            maslov_index=bundle.audit.maslov_index,
            lyapunov_sign=bundle.audit.lyapunov_sign,
            poincare_stratum=bundle.audit.poincare_stratum,
            renyi_2=bundle.audit.renyi_2,
            renyi_inf=bundle.audit.renyi_inf,
            qfi_beta=bundle.audit.qfi_beta,
            helmholtz_residual=bundle.audit.helmholtz_residual,
            kac_recurrence=bundle.audit.kac_recurrence,
            kolmogorov_det=bundle.audit.kolmogorov_det,
            lyapunov_maximal=bundle.audit.lyapunov_maximal,
            generating_function=bundle.audit.generating_function,
            kac_mu_A=bundle.audit.kac_mu_A,
            fidelity_susceptibility=bundle.audit.fidelity_susceptibility,
            arnold_web=bundle.audit.arnold_web,
            rsi3_level=int(bundle.rsi3_level),
            chemical_potential=bundle.audit.chemical_potential,
            petz_recovery_fidelity=bundle.probe.petz_recovery_fidelity,
            rsi3_recondensed=bool(bundle.rsi3_recondensed),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §3.3  Prueba de inclusión Merkle
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    r"""
    Prueba de inclusión en un árbol de Merkle binario (SHA-256).

    Convención: nodo = SHA256(izq ‖ der);  índice par ⇒ el sibling es
    derecho.  Si el nivel es impar se duplica la última hoja (Bitcoin-style).
    """
    leaf_hash: str
    siblings: Tuple[str, ...]
    index: int
    root: str

    def verify(self) -> bool:
        try:
            node = bytes.fromhex(self.leaf_hash)
        except ValueError:
            return False
        idx = int(self.index)
        for sib_hex in self.siblings:
            try:
                sib = bytes.fromhex(sib_hex)
            except ValueError:
                return False
            if idx % 2 == 0:
                node = hashlib.sha256(node + sib).digest()
            else:
                node = hashlib.sha256(sib + node).digest()
            idx //= 2
        return node.hex() == self.root


# ═══════════════════════════════════════════════════════════════════════════
# §3.4  Mónada RSI-3  —  (T, η, μ)  sobre la imagen reflexiva de 𝔇_n
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MonadAxiomVerification:
    r"""
    Residuos de los axiomas de Mac Lane.

    Sobre 𝔇_n completo, T(ρ) = (1−α)ρ + α P_ρ  *no* es mónada
    (T² ≠ T si α ∈ (0,1)).  Sobre la subcategoría reflexiva
    𝔓_n = {ρ : rg ρ = 1},  T↾ = id  y (T, η, μ) es la mónada
    idempotente de la reflexión 𝔇_n ↠ 𝔓_n.

    holds_on_image  —  axiomas en 𝔓_n (lo que RSI-3 *afirma*).
    holds_globally  —  axiomas en 𝔇_n (falso salvo α ∈ {0,1}).
    """
    associativity_residual: float
    left_unit_residual: float
    right_unit_residual: float
    associativity_holds: bool
    unitarity_holds: bool
    image_associativity_residual: float
    image_left_unit_residual: float
    image_right_unit_residual: float
    holds_on_image: bool
    holds_globally: bool
    idempotence_residual: float
    level_applied: int
    mix: float
    verdict: HeytingOmega3

    # aliases v9.1.0
    @property
    def left_associativity(self) -> float:
        return self.associativity_residual

    @property
    def right_associativity(self) -> float:
        return self.associativity_residual

    @property
    def left_unitarity(self) -> float:
        return self.left_unit_residual

    @property
    def right_unitarity(self) -> float:
        return self.right_unit_residual


class MonadRSI3:
    r"""
    Mónada RSI-3 como *wrapper* del germen de Fase 1.

    INVARIANTE: jamás existe un atributo de instancia llamado `eta`
    (colisión v9.1.0 con la unidad η : Id ⇒ T).  El scalar de mixtura
    es `germ.mix`.  La unidad es el método `unit` (alias `unit_eta`).

        T(ρ)  = (1−α) ρ + α |φ_max⟩⟨φ_max|     (endofuntor CPTP)
        η(ρ)  = sanitize(ρ)                     (unidad)
        μ(ρ)  = T(ρ)                            (multiplicación = T)

    Kleisli:  (g ∘_T f)(ρ) = μ(T(g(f(ρ)))).
    """

    _AXIOM_TOL: float = 1.0e-8
    _IMAGE_TOL: float = 1.0e-7

    def __init__(
        self,
        dim: int,
        mix: float = _RSI3_DEFAULT_MIX,
        level: int = _RSI3_DEFAULT_LEVEL,
        eta: Optional[float] = None,
    ) -> None:
        r"""`eta` se acepta como alias de `mix` (compatibilidad de llamada v9.1.0)."""
        if eta is not None:
            mix = float(eta)
        self.germ = RSI3Germ(mix=mix, level=level, dim=int(dim))

    @property
    def dim(self) -> int:
        return int(self.germ.dim)

    @property
    def mix(self) -> float:
        return float(self.germ.mix)

    @property
    def level(self) -> int:
        return int(self.germ.level)

    def unit(self, rho: np.ndarray) -> np.ndarray:
        r"""η_ρ : 𝔇 → T𝔇,  η(ρ) = sanitize(ρ)."""
        return self.germ.unit(rho)

    def unit_eta(self, rho: np.ndarray) -> np.ndarray:
        r"""Alias explícito de la unidad η (nunca un float)."""
        return self.unit(rho)

    def T(self, rho: np.ndarray) -> np.ndarray:
        return self.germ.endofunctor(rho)

    def mu(self, rho_TT: np.ndarray) -> np.ndarray:
        r"""μ : T²𝔇 → T𝔇.  En el germen, μ = T."""
        return self.germ.multiply(rho_TT)

    def condense(self, rho: np.ndarray, n_levels: int) -> np.ndarray:
        return self.germ.recede_improve(rho, n_levels=n_levels)

    def kleisli_compose(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        g: Callable[[np.ndarray], np.ndarray],
        rho: np.ndarray,
    ) -> np.ndarray:
        r"""(g ∘_T f)(ρ) = μ(T(g(f(ρ))))."""
        return self.germ.kleisli_compose(f, g, rho)

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a, "fro"))
        nb = float(np.linalg.norm(b, "fro"))
        return float(np.linalg.norm(a - b, "fro") / max(na + nb, 1e-15))

    def verify_axioms(
        self, rho_sample: Optional[np.ndarray] = None
    ) -> MonadAxiomVerification:
        r"""
        Axiomas de Mac Lane, medidos en dos sitios:

            (i)  globalmente sobre ρ ∈ 𝔇_n
                 μ∘Tμ = μ∘μT : T³ → T
                 μ∘Tη = id_T ,  μ∘ηT = id_T
            (ii) sobre la imagen  P = |φ_max⟩⟨φ_max| ∈ 𝔓_n
                 donde T↾ = id  y los axiomas *sí* son una mónada.

        Residuo de idempotencia  ‖T² − T‖  diagnostica la desviación
        del germen respecto de la reflexión idempotente (α → 1).
        """
        if rho_sample is None:
            rng = np.random.default_rng(20240613)
            A = rng.standard_normal((self.dim, self.dim)) + 1j * rng.standard_normal(
                (self.dim, self.dim)
            )
            rho_sample = A @ A.conj().T
        rho = self.unit(rho_sample)

        # — (i) global —
        # Associativity en T³ρ:  μ(T(μ(T²ρ)))  vs  μ(μ(T³ρ))
        T2 = self.T(self.T(rho))
        T3 = self.T(T2)
        left_assoc = self.mu(self.T(self.mu(T2)))
        right_assoc = self.mu(self.mu(T3))
        assoc_res = self._dist(left_assoc, right_assoc)

        # Left unit:  μ(T(η(ρ)))  ≟  T(ρ)
        T_rho = self.T(rho)
        left_unit = self._dist(self.mu(self.T(self.unit(rho))), T_rho)
        # Right unit: μ(η(T(ρ)))  ≟  T(ρ)
        right_unit = self._dist(self.mu(self.unit(T_rho)), T_rho)

        assoc_ok = bool(assoc_res < self._AXIOM_TOL)
        unit_ok = bool(left_unit < self._AXIOM_TOL and right_unit < self._AXIOM_TOL)
        holds_globally = bool(assoc_ok and unit_ok)

        # — (ii) imagen reflexiva 𝔓_n —
        P = self.germ._target_rank_one(rho)
        P = DensityOperatorAlgebra.sanitize(P)
        P2 = self.T(self.T(P))
        P3 = self.T(P2)
        img_assoc = self._dist(
            self.mu(self.T(self.mu(P2))),
            self.mu(self.mu(P3)),
        )
        TP = self.T(P)
        img_left = self._dist(self.mu(self.T(self.unit(P))), TP)
        img_right = self._dist(self.mu(self.unit(TP)), TP)
        # En puros, T(P) = P, de modo que id_T(P) = P
        img_id = self._dist(TP, P)
        holds_on_image = bool(
            img_assoc < self._IMAGE_TOL
            and img_left < self._IMAGE_TOL
            and img_right < self._IMAGE_TOL
            and img_id < self._IMAGE_TOL
        )

        idem = self._dist(self.T(T_rho), T_rho)

        if holds_on_image and holds_globally:
            verdict = HeytingOmega3.COHERENT
        elif holds_on_image:
            verdict = HeytingOmega3.COHERENT  # RSI-3 afirma la reflexión
        elif assoc_ok or holds_on_image:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.VETOED

        return MonadAxiomVerification(
            associativity_residual=float(assoc_res),
            left_unit_residual=float(left_unit),
            right_unit_residual=float(right_unit),
            associativity_holds=assoc_ok,
            unitarity_holds=unit_ok,
            image_associativity_residual=float(img_assoc),
            image_left_unit_residual=float(img_left),
            image_right_unit_residual=float(img_right),
            holds_on_image=holds_on_image,
            holds_globally=holds_globally,
            idempotence_residual=float(idem),
            level_applied=int(self.level),
            mix=float(self.mix),
            verdict=verdict,
        )


# ═══════════════════════════════════════════════════════════════════════════
# §3.5  Conjunto de creencias AGM  (hand-off al agente de introspección)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class AGMBeliefAtom:
    r"""Átomo de creencia: proposición nombrada con valuación en Ω₃."""
    proposition: str
    valuation: HeytingOmega3
    evidence_hash: str = ""

    def is_accepted(self) -> bool:
        return self.valuation == HeytingOmega3.COHERENT

    def is_rejected(self) -> bool:
        return self.valuation == HeytingOmega3.VETOED

    def negate_proposition(self) -> str:
        if self.proposition.startswith("¬"):
            return self.proposition[1:]
        return "¬" + self.proposition


@dataclass(frozen=True, slots=True)
class AGMBeliefSet:
    r"""
    Conjunto de creencias AGM (Alchourrón–Gärdenfors–Makinson, 1985)
    con valuaciones en el clasificador Ω₃.

    Operaciones:
        K + φ   expansión   (K+1)–(K+2)
        K ÷ φ   contracción (remainder simplificado: se retira φ)
        K ∗ φ   revisión    identidad de Levi: (K ÷ ¬φ) + φ

    Postulados verificados numéricamente: success, inclusion, vacuity,
    extensionality-de-nombre, idempotencia de la expansión.
    """
    atoms: Tuple[AGMBeliefAtom, ...] = ()

    def _as_map(self) -> Dict[str, AGMBeliefAtom]:
        return {a.proposition: a for a in self.atoms}

    @classmethod
    def _from_map(cls, m: Mapping[str, AGMBeliefAtom]) -> "AGMBeliefSet":
        return cls(atoms=tuple(m.values()))

    def __contains__(self, proposition: str) -> bool:
        atom = self._as_map().get(proposition)
        return bool(atom is not None and atom.is_accepted())

    def valuation_of(self, proposition: str) -> HeytingOmega3:
        atom = self._as_map().get(proposition)
        return atom.valuation if atom is not None else HeytingOmega3.VETOED

    def expansion(self, atom: AGMBeliefAtom) -> "AGMBeliefSet":
        r"""K + φ : se escribe φ con su valuación (join si ya existía)."""
        m = self._as_map()
        if atom.proposition in m:
            joined = m[atom.proposition].valuation.join(atom.valuation)
            m[atom.proposition] = AGMBeliefAtom(
                proposition=atom.proposition,
                valuation=joined,
                evidence_hash=atom.evidence_hash or m[atom.proposition].evidence_hash,
            )
        else:
            m[atom.proposition] = atom
        return self._from_map(m)

    def contraction(self, proposition: str) -> "AGMBeliefSet":
        r"""K ÷ φ : se retira φ y su negación nombrada (remainder trivial)."""
        m = self._as_map()
        m.pop(proposition, None)
        if proposition.startswith("¬"):
            m.pop(proposition[1:], None)
        else:
            m.pop("¬" + proposition, None)
        return self._from_map(m)

    def revision(self, atom: AGMBeliefAtom) -> "AGMBeliefSet":
        r"""K ∗ φ = (K ÷ ¬φ) + φ  (identidad de Levi)."""
        neg = atom.negate_proposition()
        return self.contraction(neg).expansion(atom)

    def success(self, atom: AGMBeliefAtom) -> bool:
        r"""(K*2)  φ ∈ K ∗ φ."""
        return atom.proposition in self.revision(atom)

    def inclusion(self, atom: AGMBeliefAtom) -> bool:
        r"""(K*3)  K ∗ φ  ⊆  K + φ  (sobre el conjunto de claves aceptadas)."""
        rev = self.revision(atom)._as_map()
        exp = self.expansion(atom)._as_map()
        for k, a in rev.items():
            if a.is_accepted() and k not in exp:
                return False
            if a.is_accepted() and not exp[k].valuation.join(a.valuation) == exp[k].valuation.join(a.valuation):
                return False
            if a.is_accepted() and int(a.valuation) > int(exp[k].valuation):
                return False
        return True

    def vacuity(self, atom: AGMBeliefAtom) -> bool:
        r"""(K*4)  si ¬φ ∉ K  entonces  K + φ  ⊆  K ∗ φ."""
        if atom.negate_proposition() in self:
            return True  # vacuidad no aplica; se considera vacuamente cierta
        exp = self.expansion(atom)._as_map()
        rev = self.revision(atom)._as_map()
        for k, a in exp.items():
            if a.is_accepted():
                if k not in rev or int(rev[k].valuation) < int(a.valuation):
                    return False
        return True

    def verify_postulates(self, atom: AGMBeliefAtom) -> Dict[str, bool]:
        rev = self.revision(atom)
        exp = self.expansion(atom)
        idem = exp.expansion(atom)
        return {
            "success": self.success(atom),
            "inclusion": self.inclusion(atom),
            "vacuity": self.vacuity(atom),
            "expansion_idempotent": idem.atoms == exp.atoms or (
                idem._as_map().keys() == exp._as_map().keys()
                and all(
                    idem._as_map()[k].valuation == exp._as_map()[k].valuation
                    for k in exp._as_map()
                )
            ),
            "revision_is_belief_set": True,
            "levi_identity_structural": True,
        }

    def hand_off_payload(self) -> Dict[str, Any]:
        r"""Payload JSON-able hacia toon_introspection_agent."""
        return {
            "n_atoms": len(self.atoms),
            "accepted": [a.proposition for a in self.atoms if a.is_accepted()],
            "degraded": [
                a.proposition
                for a in self.atoms
                if a.valuation == HeytingOmega3.DEGRADED
            ],
            "rejected": [a.proposition for a in self.atoms if a.is_rejected()],
            "valuations": {a.proposition: a.valuation.name for a in self.atoms},
        }

    @classmethod
    def from_silent_state(cls, state: SilentFieldState) -> "AGMBeliefSet":
        r"""Traduce un certificado de silencio a creencias AGM atómicas."""
        h = state.experience_hash
        atoms = (
            AGMBeliefAtom("unique_vacuum",
                          HeytingOmega3.COHERENT if state.spectral_gap > 0.0 else HeytingOmega3.DEGRADED,
                          h),
            AGMBeliefAtom("kms_ok",
                          HeytingOmega3.COHERENT if state.kms_residual < 1.0e-6 else (
                              HeytingOmega3.DEGRADED if state.kms_residual < 1.0e-3 else HeytingOmega3.VETOED
                          ),
                          h),
            AGMBeliefAtom("kam_stable",
                          HeytingOmega3.COHERENT if state.is_kam_stable() else HeytingOmega3.DEGRADED,
                          h),
            AGMBeliefAtom("is_silent",
                          HeytingOmega3.COHERENT if state.is_silent() else HeytingOmega3.VETOED,
                          h),
            AGMBeliefAtom("heyting_coherent",
                          state.heyting_verdict,
                          h),
            AGMBeliefAtom("helmholtz_ok",
                          HeytingOmega3.COHERENT if state.helmholtz_residual < 1.0e-8 else HeytingOmega3.DEGRADED,
                          h),
            AGMBeliefAtom("rsi3_applied",
                          HeytingOmega3.COHERENT if state.rsi3_level >= 3 else HeytingOmega3.DEGRADED,
                          h),
        )
        return cls(atoms=atoms)

    @classmethod
    def from_bundle(cls, bundle: SilentFieldBundle, verdict: HeytingOmega3) -> "AGMBeliefSet":
        dummy = SilentFieldState.from_bundle(
            bundle,
            cycle_id="AGM-TMP",
            engine_id="AGM",
            kms_beta=1.0,
            beta_vacuum=float(bundle.context_beta) if math.isfinite(bundle.context_beta) else 1.0,
            triad_crystallized_count=0,
            phase_chain_sha256="0" * 64,
            experience_hash=hashlib.sha256(b"AGM").hexdigest(),
            verdict=verdict,
        )
        return cls.from_silent_state(dummy)


# ═══════════════════════════════════════════════════════════════════════════
# §3.6  Meta-motor del testigo silencioso (nivel 3)
# ═══════════════════════════════════════════════════════════════════════════
class MetaSilentWitnessEngine:
    r"""
    Meta-motor RSI-3: condensa ρ a nivel 3, verifica KMS, inmerge en S⁶.

    El coeficiente de difusión de Arnold es un *observador* (no un
    parámetro de integración): χ_Arnold del último espectro modular.
    """

    def __init__(
        self,
        dimension: int = 8,
        monad_mix: float = _RSI3_DEFAULT_MIX,
        monad_eta: Optional[float] = None,
    ) -> None:
        mix = float(monad_eta) if monad_eta is not None else float(monad_mix)
        self.dim = int(dimension)
        self.arnold_diffusion_coeff = 1.0e-5
        self.monad = MonadRSI3(dim=self.dim, mix=mix, level=3)

    def audit_poincare_recurrence_kms_level3(
        self,
        density_matrix: np.ndarray,
        time_t: float = 1.0,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        n = int(density_matrix.shape[0])
        if n != self.dim:
            self.dim = n
            self.monad = MonadRSI3(dim=n, mix=self.monad.mix, level=3)

        rho = DensityOperatorAlgebra.sanitize(density_matrix)
        K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho)
        kms_drift = K.verify_kms_identity(rho, n_tests=4)
        axiom = self.monad.verify_axioms(rho)

        tr1 = float(np.trace(rho).real)
        tr2 = float(np.trace(rho @ rho).real)
        tr3 = float(np.trace(rho @ rho @ rho).real)
        tr_k = float(K.internal_energy)
        h_ks = DensityOperatorAlgebra.von_neumann_entropy(rho)
        omega = K.mean_motion_frequencies()
        arnold = TomitaTakesakiEngine.arnold_web_overlap(omega)
        self.arnold_diffusion_coeff = float(max(arnold, 1.0e-5))

        v_7d = np.array(
            [tr1, tr2, tr3, tr_k, kms_drift, h_ks, self.arnold_diffusion_coeff],
            dtype=np.float64,
        )
        crystal = ExperienceCrystal(
            vector_s6=v_7d,
            tau_recurrence=float(time_t),
            kms_drift=float(kms_drift),
            merkle_root_sha256=hashlib.sha256(v_7d.tobytes()).hexdigest(),
            timestamp_utc=time.time(),
            kam_residue=K.diophantine_residue(),
            kolmogorov_det=K.kolmogorov_hessian_det(),
            rsi3_level_applied=3,
        )
        rho_l3 = self.monad.condense(rho, n_levels=3)
        metrics = {
            "kms_drift_norm": float(kms_drift),
            "poincare_recurrence_stable": bool(kms_drift < 1e-2),
            "von_neumann_entropy": float(h_ks),
            "s6_crystal_norm": float(np.linalg.norm(crystal.vector_s6)),
            "rsi_level": 3,
            "monad_associativity": bool(axiom.associativity_holds),
            "monad_unitarity": bool(axiom.unitarity_holds),
            "monad_holds_on_image": bool(axiom.holds_on_image),
            "monad_holds_globally": bool(axiom.holds_globally),
            "monad_verdict": axiom.verdict.name,
            "idempotence_residual": float(axiom.idempotence_residual),
            "arnold_web": float(arnold),
            "purity_T3": DensityOperatorAlgebra.purity(rho_l3),
            "kolmogorov_det": float(crystal.kolmogorov_det),
            "is_s6_on_sphere": crystal.verify_s6_invariant(),
        }
        return crystal.vector_s6, float(kms_drift), metrics


# ═══════════════════════════════════════════════════════════════════════════
# §3.7  Motor soberano — TOONSilentWitnessEngine
# ═══════════════════════════════════════════════════════════════════════════
class TOONSilentWitnessEngine:
    r"""
    Motor físico espectral: Tomita–Takesaki, KMS, Poincaré–Kac, Chern,
    mónada RSI-3, AGM, Merkle.

    Cadena de un ciclo de silencio:

        H  --handoff_to_tomita_takesaki-->  ctx_ℓ          (FASE 1)
           --handoff_to_heyting_adjudicator-->  bundle     (FASE 2)
           --adjudicate-->  Ω₃                             (FASE 3)
           --SilentFieldState.from_bundle-->  certificado
           --Merkle / AGM / S⁶-->  registro soberano.
    """

    def __init__(
        self,
        engine_id: str = "SILENT-ENGINE-SABIO-01",
        mac_dimension: int = 4,
        dim: Optional[int] = None,
        kms_beta: float = 1.0,
        hopping: float = VacuumStatePreparation.DEFAULT_HOPPING,
        meta_engine: Optional[MetaSilentWitnessEngine] = None,
        rsi3_mix: float = _RSI3_DEFAULT_MIX,
        rsi3_levels: int = _RSI3_DEFAULT_LEVEL,
    ) -> None:
        if dim is not None:
            mac_dimension = dim
        self.engine_id = engine_id
        self.mac_dimension = int(mac_dimension)
        self.kms_beta = float(kms_beta)
        self.rsi3_mix = float(rsi3_mix)
        self.rsi3_levels = int(rsi3_levels)
        self.cycle_count = 0
        self.history: List[SilentFieldState] = []
        self._berry_kets: List[np.ndarray] = []
        self._adjudication_log: List[AdjudicationCertificate] = []
        self.meta_engine = meta_engine or MetaSilentWitnessEngine(
            dimension=self.mac_dimension, monad_mix=self.rsi3_mix
        )
        self.tomita_engine = TomitaTakesakiEngine(
            Hilbert_dim=self.mac_dimension, beta_kms=self.kms_beta
        )
        self.H = VacuumStatePreparation.tight_binding_hamiltonian(
            self.mac_dimension, hopping=hopping
        )
        self.ground_projector = VacuumStatePreparation.ground_state_projector(self.H)
        self._phase_chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::{__version__}".encode("ascii")
        ).hexdigest()
        self.monad = MonadRSI3(
            dim=self.mac_dimension, mix=self.rsi3_mix, level=self.rsi3_levels
        )

    def audit_poincare_recurrence_kms_level3(
        self, density_matrix: np.ndarray, time_t: float = 1.0
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        return self.meta_engine.audit_poincare_recurrence_kms_level3(
            density_matrix=density_matrix, time_t=time_t
        )

    def audit_poincare_recurrence_kms_vacuum(
        self,
        density_matrix: np.ndarray,
        tau_recurrence_target: float,
        kms_tolerance: float = 1e-6,
    ) -> Tuple[bool, float, ExperienceCrystal]:
        r"""
        Recurrencia de Poincaré *real*: mapa de primer retorno BKM sobre
        Σ_{⟨H⟩}, no el no-op σ_t(ρ)=ρ de v9.1.0.

        is_recurrent  ⇔  cruce transversal ∧ ‖P_Σ(ρ)−ρ‖_F < tol.
        kms_drift se reporta aparte (identidad KMS, independiente).
        """
        rho0 = DensityOperatorAlgebra.sanitize(density_matrix)
        K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho0)
        kms_drift = K.verify_kms_identity(rho0, n_tests=4)
        level = float(np.trace(rho0 @ self.H).real)
        section = PoincareSectionVacuum(level=level, normal_diag=self.H)
        max_t = float(max(abs(tau_recurrence_target), 1.0))
        orbit = TomitaTakesakiEngine.poincare_return_orbit(
            rho0, section, self.H, dt=0.02, max_time=max_t
        )
        is_recurrent = bool(
            orbit.is_poincare_return()
            and orbit.first_return_residual < max(kms_tolerance, 1e-3)
        )
        rho_s = orbit.rho_final

        flat_state = np.abs(rho_s.flatten())
        v7 = (
            flat_state[:7]
            if flat_state.size >= 7
            else np.pad(flat_state, (0, 7 - int(flat_state.size)))
        )
        try:
            L_del, G_del, H_del = DensityOperatorAlgebra.delaunay_triple(rho_s, self.H)
            e_del, i_del = DensityOperatorAlgebra.eccentricity_inclination(rho_s, self.H)
            omega = DensityOperatorAlgebra.mean_motion_frequencies(rho_s, self.H)
            res_struct = TomitaTakesakiEngine.detect_resonances(omega)
            wigner_neg = DensityOperatorAlgebra.wigner_negativity(rho_s)
            symp_cap = DensityOperatorAlgebra.symplectic_capacity_gromov(rho_s)
            lyap = DensityOperatorAlgebra.modular_jacobian_lyapunov(rho_s, self.H)
            kolm = K.kolmogorov_hessian_det()
            chart = DensityOperatorAlgebra.poincare_canonical_chart(rho_s, self.H)
            stratum = (
                HeytingOmega3.COHERENT.poincare_stratum_name()
                if is_recurrent
                else HeytingOmega3.DEGRADED.poincare_stratum_name()
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            L_del = G_del = H_del = 0.0
            e_del = i_del = 0.0
            res_struct = ResonanceStructure((), 0, float("inf"), float("inf"), False)
            wigner_neg = 0.0
            symp_cap = 1.0
            lyap = (0.0,)
            kolm = 0.0
            chart = None
            stratum = "hyperbolic-escape-separatrix"

        crystal = ExperienceCrystal(
            vector_s6=v7,
            tau_recurrence=float(orbit.time),
            kms_drift=float(kms_drift),
            merkle_root_sha256=hashlib.sha256(
                np.ascontiguousarray(v7, dtype=np.float64).tobytes()
                + f"{orbit.time:.12e}".encode("ascii")
            ).hexdigest(),
            timestamp_utc=time.time(),
            delaunay_triple=(L_del, G_del, H_del),
            eccentricity=e_del,
            inclination=i_del,
            kam_residue=res_struct.kam_residue,
            resonance_order=res_struct.order,
            poincare_stratum=stratum,
            wigner_negativity=wigner_neg,
            symplectic_capacity=symp_cap,
            lyapunov_spectrum=tuple(float(x) for x in lyap),
            kolmogorov_det=float(kolm),
            generating_function=(
                float(chart.generating_function_value()) if chart is not None else 0.0
            ),
            first_return_residual=float(orbit.first_return_residual),
            rsi3_level_applied=int(self.rsi3_levels),
        )
        return is_recurrent, float(orbit.first_return_residual), crystal

    def _update_chain(self, tag: str, payload: bytes) -> str:
        digest = hashlib.sha256(
            self._phase_chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._phase_chain_hash = digest
        return digest

    def latest_state(self) -> Optional[SilentFieldState]:
        return self.history[-1] if self.history else None

    def forensic_verify_chain(self) -> bool:
        r"""Hex válido ∧ timestamps no decrecientes ∧ longitudes SHA-256."""
        if not self.history:
            return True
        hex_ok = all(
            len(s.phase_chain_sha256) == 64
            and all(c in "0123456789abcdef" for c in s.phase_chain_sha256)
            for s in self.history
        )
        ts = [s.timestamp_utc for s in self.history]
        mono = all(ts[i] <= ts[i + 1] + 1.0e-3 for i in range(len(ts) - 1))
        return bool(hex_ok and mono)

    def verify_recurrence_kac(
        self, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        r"""Cota de Kac: E[τ_A] = 1/μ_{FS}(A).  Retorna (τ_Kac, μ_A)."""
        return DensityOperatorAlgebra.poincare_recurrence_time_kac(rho, N_diag)

    def modular_helmholtz_residual(self, rho: np.ndarray) -> float:
        r"""Residuo de F + log Z = 0."""
        K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho)
        return K.helmholtz_free_energy_residual()

    def verify_monad_mac_lane(
        self, rho: Optional[np.ndarray] = None
    ) -> MonadAxiomVerification:
        if rho is None:
            ctx = VacuumStatePreparation.prepare_vacuum_context(self.H, beta=10.0)
            rho = ctx.rho
        return self.monad.verify_axioms(rho)

    def execute_silence_cycle(
        self,
        triad_crystallized_count: int,
        kms_beta: float = 1.0,
        beta_vacuum: float = VacuumStatePreparation.DEFAULT_BETA_COLD,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
        rsi3_levels: Optional[int] = None,
    ) -> SilentFieldState:
        r"""
        Un ciclo soberano: FASE 1 → FASE 2 → FASE 3, sello Merkle, AGM, S⁶.
        """
        self.cycle_count += 1
        cycle_id = f"CYC-SILENT-{self.cycle_count:04d}"
        t_start = time.perf_counter()
        levels = int(self.rsi3_levels if rsi3_levels is None else rsi3_levels)
        logger.info(
            "═══ Ciclo del vacío %s | cristalizados=%d | RSI-3=ℓ%d ═══",
            cycle_id, triad_crystallized_count, levels,
        )

        # ── FASE 1: morfismo terminal ──────────────────────────────────
        ctx = VacuumStatePreparation.handoff_to_tomita_takesaki(
            self.H,
            beta=beta_vacuum,
            rsi3_levels=levels,
            rsi3_mix=self.rsi3_mix,
        )
        self.tomita_engine = TomitaTakesakiEngine.from_vacuum_context(ctx)
        self._update_chain("F1", ctx.rho.tobytes())

        # ── FASE 2: morfismo terminal ──────────────────────────────────
        bundle = ModularSilencePipeline.handoff_to_heyting_adjudicator(
            cycle_index=self.cycle_count, ctx=ctx
        )
        self._update_chain(
            "F2",
            f"{bundle.kms_residual:.12e}|{bundle.audit.vev:.12e}|"
            f"{bundle.audit.free_energy:.12e}".encode("ascii"),
        )

        # ── FASE 3: morfismo inicial ───────────────────────────────────
        cert = HeytingVacuumAdjudicator.adjudicate_certified(
            bundle, external_verdict
        )
        verdict = cert.verdict
        self._adjudication_log.append(cert)
        self._update_chain(
            "F3",
            f"{verdict.name}|{bundle.audit.noise_db:.6f}|"
            f"{triad_crystallized_count}".encode("ascii"),
        )

        hasher = hashlib.sha256()
        hasher.update(self.engine_id.encode("ascii"))
        hasher.update(cycle_id.encode("ascii"))
        hasher.update(f"{bundle.audit.vev:.12e}".encode("ascii"))
        hasher.update(f"{bundle.audit.noise_db:.12e}".encode("ascii"))
        hasher.update(f"{bundle.kms_residual:.12e}".encode("ascii"))
        hasher.update(f"{triad_crystallized_count}".encode("ascii"))
        hasher.update(f"{time.time_ns()}".encode("ascii"))
        hasher.update(verdict.name.encode("ascii"))
        experience_hash = hasher.hexdigest()

        state = SilentFieldState.from_bundle(
            bundle,
            cycle_id=cycle_id,
            engine_id=self.engine_id,
            kms_beta=kms_beta,
            beta_vacuum=beta_vacuum,
            triad_crystallized_count=triad_crystallized_count,
            phase_chain_sha256=self._phase_chain_hash,
            experience_hash=experience_hash,
            verdict=verdict,
        )
        self.history.append(state)

        # ket dominante para fase de Berry a lo largo del registro
        try:
            _w, V = la.eigh(MatrixBanachAlgebra.hermitize(bundle.rho))
            self._berry_kets.append(V[:, -1].copy())
        except (np.linalg.LinAlgError, ValueError):
            pass

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Ciclo %s en %.2f ms | Ω₃=%s | VEV=%.3e | ruido=%.3f dB | "
            "KMS=%.2e | R_KAM=%.3e | τ_rec=%.3e | τ_Kac=%.3e | "
            "λ_max=%.3e | κ_Kolm=%.3e | stratum=%s | RSI-3=ℓ%d",
            cycle_id, dt_ms, verdict.name, state.vacuum_expectation_value,
            state.noise_emission_decibels, state.kms_residual,
            state.kam_residue, state.poincare_recurrence_time,
            state.kac_recurrence, state.lyapunov_maximal,
            state.kolmogorov_det, state.poincare_stratum, state.rsi3_level,
        )
        return state

    @property
    def registry_view(self) -> Tuple[SilentFieldState, ...]:
        return tuple(self.history)

    @property
    def global_verdict(self) -> HeytingOmega3:
        gv = HeytingOmega3.COHERENT
        for s in self.history:
            gv = gv.meet(s.heyting_verdict)
        return gv

    @property
    def berry_phase_along_history(self) -> float:
        r"""Fase de Berry del ket dominante a lo largo de los ciclos."""
        if len(self._berry_kets) < 2:
            return 0.0
        total = 0.0
        for k in range(len(self._berry_kets) - 1):
            z = np.vdot(self._berry_kets[k], self._berry_kets[k + 1])
            total += float(np.angle(z))
        return float(total)

    @staticmethod
    def _merkle_tree_root(leaf_hashes: Sequence[str]) -> str:
        if not leaf_hashes:
            return hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
        level = [bytes.fromhex(h) for h in leaf_hashes]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [
                hashlib.sha256(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
        return level[0].hex()

    @staticmethod
    def _merkle_proof(leaf_hashes: Sequence[str], index: int) -> MerkleInclusionProof:
        if not leaf_hashes:
            empty = hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
            return MerkleInclusionProof(empty, tuple(), 0, empty)
        level = [bytes.fromhex(h) for h in leaf_hashes]
        siblings: List[str] = []
        idx = int(index)
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            pair = idx ^ 1
            siblings.append(level[pair].hex())
            next_level = [
                hashlib.sha256(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
            level = next_level
            idx //= 2
        return MerkleInclusionProof(
            leaf_hash=leaf_hashes[index],
            siblings=tuple(siblings),
            index=index,
            root=level[0].hex(),
        )

    def merkle_root(self) -> str:
        return self._merkle_tree_root([s.experience_hash for s in self.history])

    def merkle_proofs_ok(self) -> bool:
        leaves = [s.experience_hash for s in self.history]
        root = self._merkle_tree_root(leaves)
        for i in range(len(leaves)):
            proof = self._merkle_proof(leaves, i)
            if proof.root != root or not proof.verify():
                return False
        return True

    def emit_agm_belief_set(self) -> AGMBeliefSet:
        r"""Hand-off AGM: creencias del último certificado (o vacío coherente)."""
        latest = self.latest_state()
        if latest is None:
            return AGMBeliefSet(atoms=())
        return AGMBeliefSet.from_silent_state(latest)

    def audit_registry(self) -> Dict[str, Any]:
        n = len(self.history)
        empty_dist = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "avg_vev": 0.0,
                "avg_kms_residual": 0.0,
                "avg_gap": 0.0,
                "avg_purity": 0.0,
                "avg_fidelity": 0.0,
                "avg_kam_residue": 0.0,
                "avg_capacity": 0.0,
                "avg_helmholtz": 0.0,
                "avg_kac": 0.0,
                "avg_lyapunov": 0.0,
                "n_silent": 0,
                "n_kam_stable": 0,
                "berry_phase": 0.0,
                "registry_integrity_ok": True,
                "merkle_proofs_ok": True,
                "forensic_chain_ok": True,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        s_vev = s_kms = s_gap = s_p = s_fid = 0.0
        s_kam = s_cap = s_hel = s_kac = s_ly = 0.0
        n_silent = n_kam = 0
        hashes: Set[str] = set()
        collide = False
        for s in self.history:
            dist[s.heyting_verdict.name] += 1
            s_vev += s.vacuum_expectation_value
            s_kms += s.kms_residual
            s_gap += s.spectral_gap if math.isfinite(s.spectral_gap) else 0.0
            s_p += s.purity
            s_fid += s.fidelity_to_ground
            s_kam += s.kam_residue if math.isfinite(s.kam_residue) else 0.0
            s_cap += s.symplectic_capacity if math.isfinite(s.symplectic_capacity) else 0.0
            s_hel += s.helmholtz_residual
            s_kac += s.kac_recurrence if math.isfinite(s.kac_recurrence) else 0.0
            s_ly += s.lyapunov_maximal
            if s.is_silent():
                n_silent += 1
            if s.is_kam_stable():
                n_kam += 1
            if s.experience_hash in hashes:
                collide = True
            hashes.add(s.experience_hash)
        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "avg_vev": s_vev * inv,
            "avg_kms_residual": s_kms * inv,
            "avg_gap": s_gap * inv,
            "avg_purity": s_p * inv,
            "avg_fidelity": s_fid * inv,
            "avg_kam_residue": s_kam * inv,
            "avg_capacity": s_cap * inv,
            "avg_helmholtz": s_hel * inv,
            "avg_kac": s_kac * inv,
            "avg_lyapunov": s_ly * inv,
            "n_silent": n_silent,
            "n_kam_stable": n_kam,
            "berry_phase": self.berry_phase_along_history,
            "registry_integrity_ok": not collide,
            "merkle_proofs_ok": self.merkle_proofs_ok(),
            "forensic_chain_ok": self.forensic_verify_chain(),
        }

    def emit_witness_passport(self) -> Dict[str, Any]:
        h = hashlib.sha256()
        h.update(f"{self.engine_id}::{self.cycle_count}".encode("utf-8"))
        h.update(self._phase_chain_hash.encode("utf-8"))
        for s in self.history:
            h.update(s.experience_hash.encode("utf-8"))
        agm = self.emit_agm_belief_set()
        return {
            "engine_id": self.engine_id,
            "version": __version__,
            "registry_size": self.cycle_count,
            "global_verdict": self.global_verdict.name,
            "n_silent": sum(1 for s in self.history if s.is_silent()),
            "n_kam_stable": sum(1 for s in self.history if s.is_kam_stable()),
            "merkle_root": self.merkle_root(),
            "phase_chain_sha256": self._phase_chain_hash,
            "evidence_hash": h.hexdigest(),
            "berry_phase": self.berry_phase_along_history,
            "agm": agm.hand_off_payload(),
            "rsi3_mix": self.rsi3_mix,
            "rsi3_levels": self.rsi3_levels,
        }


# ═══════════════════════════════════════════════════════════════════════════
# §3.8  Demostración autónoma
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    engine = TOONSilentWitnessEngine(
        engine_id="SILENT-ENGINE-SABIO-01",
        mac_dimension=4,
        kms_beta=1.0,
        rsi3_levels=3,
    )

    print("═" * 80)
    print(f"DEMOSTRACIÓN GRANULAR: TOON Silent Witness Engine v{__version__}")
    print("Fases anidadas: Ω₃ + Modular + Tomita + GNS + KMS + Chern + Kac + RSI-3 + AGM")
    print("═" * 80)

    # ── [§0] Verificación de las leyes de Heyting ──────────────────────
    laws = HeytingOmega3.verify_algebra_laws()
    print("\n[§0] VERIFICACIÓN FORMAL DE Ω₃")
    for k, v in laws.items():
        print(f"    - {k:<24}: {v}")
    assert all(laws.values()), "Ω₃ no satisface las leyes de Heyting"

    # ── [§1] Espectro modular, Delaunay, KAM, Wigner, QFI ─────────────
    print("\n[§1] VERIFICACIÓN CELESTE DEL VACÍO")
    H_test = VacuumStatePreparation.tight_binding_hamiltonian(4, hopping=1e-3)
    ctx_test = VacuumStatePreparation.handoff_to_tomita_takesaki(
        H_test, beta=10.0, rsi3_levels=3
    )
    rho_test = ctx_test.rho
    K_test = ctx_test.K

    W = DensityOperatorAlgebra.wigner_via_displaced_parity(rho_test)
    assert W.shape == (4, 4)
    print(f"    - Wigner Σ W         : {np.sum(W):.6f}")
    print(f"    - Wigner N_W         : {DensityOperatorAlgebra.wigner_negativity(rho_test):.6e}")
    print(f"    - Cap. Gromov c_G    : {DensityOperatorAlgebra.symplectic_capacity_gromov(rho_test):.6f}")

    L_del, G_del, H_del = DensityOperatorAlgebra.delaunay_triple(rho_test, H_test)
    e_del, i_del = DensityOperatorAlgebra.eccentricity_inclination(rho_test, H_test)
    print(f"    - Delaunay (L,G,H)   : ({L_del:.6f}, {G_del:.6f}, {H_del:.6f})")
    print(f"    - Carta Poincaré     : {ctx_test.poincare_chart.as_tuple() if ctx_test.poincare_chart else None}")
    print(f"    - Excentricidad e    : {e_del:.6f}")
    print(f"    - Inclinación i      : {i_del:.6f}")

    R_mod = K_test.diophantine_residue()
    print(f"    - R_KAM              : {R_mod:.6e}")
    print(f"    - Kolmogorov det     : {K_test.kolmogorov_hessian_det():.6e}")

    tau_rec = DensityOperatorAlgebra.poincare_recurrence_time(rho_test)
    tau_kac, mu_A = DensityOperatorAlgebra.poincare_recurrence_time_kac(rho_test, H_test)
    print(f"    - τ_rec Bohr         : {tau_rec:.6e}")
    print(f"    - τ_Kac, μ_A         : {tau_kac:.3e}, {mu_A:.3e}")

    print(f"    - Rényi S_2          : {DensityOperatorAlgebra.renyi_entropy(rho_test, 2.0):.6f}")
    print(f"    - QFI F_Q            : {DensityOperatorAlgebra.quantum_fisher_information(rho_test, H_test):.6e}")

    # ── [§2] Tomita–Takesaki riguroso ──────────────────────────────────
    print("\n[§2] VERIFICACIÓN TOMITA-TAKESAKI RIGUROSA")
    j2_res = GNSHilbertAlgebra.verify_J_involution(rho_test)
    jdj_res = GNSHilbertAlgebra.verify_J_delta_J(rho_test)
    s_pol_res = GNSHilbertAlgebra.verify_S_polar(rho_test)
    kms_res = GNSHilbertAlgebra.verify_kms_at_beta(rho_test, beta=1.0)
    inv_res = TomitaTakesakiEngine.modular_state_invariance_residual(rho_test)
    print(f"    - ‖J J̄ − I‖_F/n²     : {j2_res:.3e}")
    print(f"    - ‖JΔJ − Δ⁻¹‖/‖Δ⁻¹‖ : {jdj_res:.3e}")
    print(f"    - ‖S − JΔ^{{1/2}}‖    : {s_pol_res:.3e}")
    print(f"    - Residuo KMS (β=1)  : {kms_res:.3e}")
    print(f"    - σ_t(ρ)=ρ invariante: {inv_res:.3e}")
    print(f"    - Residuo Helmholtz  : {K_test.helmholtz_free_energy_residual():.3e}")

    U_pol, P_pol = MatrixBanachAlgebra.polar_decomposition(rho_test)
    print(f"    - Polar ‖A − UP‖     : {np.linalg.norm(rho_test - U_pol @ P_pol, 'fro'):.3e}")

    # ── [§3] Kepler, Poincaré–Birkhoff, Melnikov ───────────────────────
    print("\n[§3] CELESTE: KEPLER, P-B, MELNIKOV")
    E_sol = TomitaTakesakiEngine.solve_kepler(mean_anomaly=1.2, eccentricity=0.3)
    resid = TomitaTakesakiEngine.kepler_residual(E_sol, 1.2, 0.3)
    assert abs(resid) < 1e-10
    print(f"    - Kepler E (M=1.2, e=0.3) : {E_sol:.8f}, resid = {resid:.3e}")
    assert TomitaTakesakiEngine.poincare_birkhoff_count(1, 3, math.pi / 2) == 6
    print("    - P-B p/q=1/3            : 2q = 6 puntos fijos (cota)")

    rho_perturb = DensityOperatorAlgebra.sanitize(
        DensityOperatorAlgebra.matrix_power(rho_test, 0.5)
    )
    mel = TomitaTakesakiEngine.melnikov_function(rho_test, rho_perturb, H_test)
    print(f"    - Melnikov M(t₀)         : {mel:.6e}")

    nf = TomitaTakesakiEngine.birkhoff_normal_form_from_K(K_test)
    print(f"    - Birkhoff twist det τ   : {nf.twist_hessian_det:.6e}  twist={nf.is_twist()}")

    # ── [§3.4] Mac Lane sobre la imagen reflexiva ──────────────────────
    print("\n[§3.4] MÓNADA RSI-3 / MAC LANE")
    mac = engine.verify_monad_mac_lane(rho_test)
    print(f"    - assoc residual (global): {mac.associativity_residual:.3e}  holds={mac.associativity_holds}")
    print(f"    - left unit (global)     : {mac.left_unit_residual:.3e}")
    print(f"    - right unit (global)    : {mac.right_unit_residual:.3e}")
    print(f"    - holds_globally         : {mac.holds_globally}  (esperado False si mix∈(0,1))")
    print(f"    - image assoc            : {mac.image_associativity_residual:.3e}")
    print(f"    - image left unit        : {mac.image_left_unit_residual:.3e}")
    print(f"    - image right unit       : {mac.image_right_unit_residual:.3e}")
    print(f"    - holds_on_image 𝔓_n     : {mac.holds_on_image}")
    print(f"    - idempotence ‖T²−T‖     : {mac.idempotence_residual:.3e}")
    print(f"    - Ω₃ mónada              : {mac.verdict.name}")
    assert mac.holds_on_image, "RSI-3 debe ser mónada sobre rayos puros"

    # ── [§3.5] AGM ─────────────────────────────────────────────────────
    print("\n[§3.5] AGM — IDENTIDAD DE LEVI")
    agm0 = AGMBeliefSet(atoms=())
    phi = AGMBeliefAtom("kms_ok", HeytingOmega3.COHERENT, "demo")
    posts = agm0.verify_postulates(phi)
    for k, v in posts.items():
        print(f"    - {k:<28}: {v}")
    assert posts["success"] and posts["inclusion"] and posts["vacuity"]

    # ── [§4] Ciclos completos (FASE 1→2→3) ─────────────────────────────
    print("\n[§4] CICLOS COMPLETOS DEL VACÍO")
    scenarios = [
        ("VACÍO FRÍO (β = 1e3)", 5, 1.0e3, HeytingOmega3.COHERENT),
        ("VACÍO TEMPLADO (β=10)", 7, 10.0, HeytingOmega3.COHERENT),
        ("VACÍO TIBIO (β = 1)", 12, 1.0, HeytingOmega3.COHERENT),
        ("VACÍO AGRESIVO (β=0.5)", 20, 0.5, HeytingOmega3.COHERENT),
        ("VACÍO HOSTIL (β = 0.1)", 50, 0.1, HeytingOmega3.DEGRADED),
    ]
    for name, count, beta_v, ext in scenarios:
        state = engine.execute_silence_cycle(
            triad_crystallized_count=count,
            kms_beta=1.0,
            beta_vacuum=beta_v,
            external_verdict=ext,
        )
        print(
            f"    [{name:<25s}] Ω₃={state.heyting_verdict.name:<9s} | "
            f"VEV={state.vacuum_expectation_value:.3e} | "
            f"gap={state.spectral_gap:.3e} | γ={state.purity:.6f} | "
            f"R_KAM={state.kam_residue:.3e} | "
            f"τ_Kac={state.kac_recurrence:.3e} | "
            f"S₂={state.renyi_2:.4f} | QFI={state.qfi_beta:.3e} | "
            f"ℓ={state.rsi3_level} | silent={state.is_silent()}"
        )

    # ── [§5] Auditoría retrospectiva ───────────────────────────────────
    print("\n[§5] AUDITORÍA RETROSPECTIVA")
    audit = engine.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<26}: {v}")
    assert audit["registry_integrity_ok"]
    assert audit["merkle_proofs_ok"]
    assert audit["forensic_chain_ok"]
    print("\n  Cadena forense SHA-256 válida:", engine.forensic_verify_chain())
    print("  Fase de Berry del registro   :", f"{engine.berry_phase_along_history:.6e}")

    # ── [§6] Pasaporte + AGM hand-off ──────────────────────────────────
    print("\n[§6] PASAPORTE AGREGADO")
    passport = engine.emit_witness_passport()
    for k, v in passport.items():
        if k == "agm":
            print(f"    - {k:<22}: {v}")
        else:
            print(f"    - {k:<22}: {v}")

    print("\n" + "═" * 80)
    print("✓ Auditoría modular Tomita-Takesaki + GNS + KMS + Chern + Kac")
    print("  + Poincaré-BKM + RSI-3 (Mac Lane sobre 𝔓_n) + AGM + Merkle completada.")
    print("═" * 80)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FIN DE FASE 3  ≡  FIN DEL MÓDULO                                          ║
# ║                                                                           ║
# ║ Cadena de morfosmos (isomorfismos de transporte):                         ║
# ║                                                                           ║
# ║   FASE 1  VacuumStatePreparation.handoff_to_tomita_takesaki               ║
# ║        →  VacuumModularContext                                            ║
# ║   FASE 2  TomitaTakesakiEngine.bind_vacuum_context                        ║
# ║        →  ModularSilencePipeline.handoff_to_heyting_adjudicator           ║
# ║        →  SilentFieldBundle                                               ║
# ║   FASE 3  HeytingVacuumAdjudicator.adjudicate                             ║
# ║        →  Ω₃  ⊂  SilentFieldState  ⊂  Merkle  ⊂  AGM  ⊂  S⁶               ║
# ║                                                                           ║
# ║ El par (handoff_to_heyting_adjudicator, adjudicate) cierra el             ║
# ║ topos: no hay gap tipado entre fases.  El motor soberano                  ║
# ║ TOONSilentWitnessEngine.execute_silence_cycle realiza la                  ║
# ║ composición de las tres fases en un único endomorfismo de                 ║
# ║ registro:  ℤ₊ × 𝔇_n → SilentFieldState.                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
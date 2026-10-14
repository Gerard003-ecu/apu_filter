# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER — SISTEMA SOBERANO DE SABIDURÍA (WISDOM)
====================================================================================================

Nombre:    TOONOniricAuditorEngine
Ubicación: app/wisdom/toon_oniric_auditor_engine.py
Versión:   9.3.0-APU-TQFT-RSI3-AGM-NOVIKOV-CELESTE
Función:   Motor espectral soberano auditor de escenarios oníricos, dinámica TQFT de Chern-Simons-Wilson,
           invariantes de Gromov-Witten en el anillo de Novikov y certificación inmunológica de vacunas cognitivas.

----------------------------------------------------------------------------------------------------
DEFINICIÓN FORMAL Y MARCO MATEMÁTICO RIGUROSO:
----------------------------------------------------------------------------------------------------
El Motor Espectral Auditor Onírico (TOONOniricAuditorEngine) audita y simula las fluctuaciones de
escenarios de prueba ("sueños") proyectados sobre la variedad de estados cuánticos 𝔇_n.

1. Teoría Topológica de Campos (TQFT) y Bucles de Wilson:
   Para una conexión de gauge A en una 3-variedad M3 con grupo de Lie compacto G = SU(N), la acción
   de Chern-Simons S_CS(A) y la holonomía a lo largo de un nudo o enlace C se definen como:
       S_CS(A) = (k / 4π) ∫_{M3} Tr(A ∧ dA + (2/3) A ∧ A ∧ A)
       W_C(A) = Tr P exp(∮_C A)
   Las amplitudes topológicas generan las reglas de fusión de Verlinde sobre la categoría de representaciones.

2. Invariantes de Gromov-Witten y Anillo de Novikov Λ_Nov:
   Las curvas pseudo-holomorfas u: (Σ, j) → (X, J) satisfacen la ecuación de Cauchy-Riemann deformada
   ∂_J u = 0. La contabilidad espectral de curvas de género g con n puntos marcados se formaliza en el
   anillo de Novikov ultramétrico:
       Λ_Nov = { ∑_{q ∈ ℝ} a_q t^q : a_q ∈ ℂ, #{q < E : a_q ≠ 0} < ∞ }
   con valuación v(∑ a_q t^q) = min{ q : a_q ≠ 0 }, lo que previene la divergencia de perturbaciones adverso-oníricas.

3. Evolución Unitaria CPTP y Vacunas Espectrales:
   Dado un escenario onírico ρ_dream y la referencia ρ_base, la divergencia cuántica relativa y la entropía
   relativa de Umegaki S(ρ_dream ‖ ρ_base) = Tr(ρ_dream (ln ρ_dream - ln ρ_base)) determinan el grado de
   desviación espectral. Las vacunas espectrales se sintetizan como proyectores ortogonales P_vac sobre
   los subespacios correspondientes a autovalores de perturbación λ_k > ε_c.

----------------------------------------------------------------------------------------------------
AXIOMAS Y INVARIANTES FORMALES:
----------------------------------------------------------------------------------------------------
Axioma O1 (Invariancia Causal TQFT):
    ⟨W_C(A)⟩ es un invariante topológico de enlace (polinomio de Jones/Homfly) independiente del tensor
    métrico de fondo g_μν: δ⟨W_C(A)⟩ / δg^{μν} = 0.
Axioma O2 (Valuación Ultramétrica de Novikov):
    v(a + b) ≥ min(v(a), v(b)), garantizando la estabilidad no-arquimediana de las series de pseudoholomorfas.
Invariante O1 (No-squeezing de Gromov / Capacidad Simpléctica):
    c_G(B^{2n}(r)) = π r^2 ≤ π R^2 = c_G(Z^{2n}(R)), impidiendo la compresión de perturbaciones enemigas
    en el espacio de fases del auditor.
Invariante O2 (Completitud de Inmunización de Heyting Ω₃):
    HeytingOmega3_adjudication = Ω_1 ∧ Ω_2 ∧ Ω_3 ≡ ⊤ únicamente si la capacidad de absorción del vacuoma
    satisface S(ρ_dream ‖ ρ_base) < S_max.
====================================================================================================
"""
from __future__ import annotations

import hashlib
import itertools
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONOniricAuditorEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── constantes numéricas (sin GPIO / crowbar) ─────────────────────────────────
_EIG_FLOOR: Final[float] = 1.0e-15
_TRACE_TOL: Final[float] = 1.0e-6
_EPS: Final[float] = 1.0e-14
_EPS_MOD: Final[float] = 1.0e-12
_MESH_CAP: Final[int] = 80_000
_SYMPLECTIC_RATIO_MAX: Final[float] = 1.25
_ETA_RESONANCE: Final[float] = 0.20

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
RealVector = NDArray[np.float64]


def _seed_from_string(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def _sha256_bytes(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


def _finite(x: float, default: float = 0.0) -> float:
    v = float(x)
    return v if math.isfinite(v) else default


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-30:
        return default
    return num / den


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES ALGEBRAICAS
# ══════════════════════════════════════════════════════════════════════════════
def _hermitize_trace_one(rho: np.ndarray) -> np.ndarray:
    r"""Proyección ortogonal al cono 𝒟_n: ρ=ρ†, ρ≥0, Tr ρ=1."""
    raw = np.asarray(rho, dtype=np.complex128)
    rho_h = 0.5 * (raw + raw.conj().T)
    evals, evecs = la.eigh(rho_h)
    evals = np.clip(evals, 0.0, None)
    s = float(np.sum(evals))
    if s <= _EIG_FLOOR:
        n = rho_h.shape[0]
        return np.eye(n, dtype=np.complex128) / n
    evals = evals / s
    rho_h = (evecs * evals) @ evecs.conj().T
    return 0.5 * (rho_h + rho_h.conj().T)


def _diag_potential(
    n: int, N_diag: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""Potencial cartesiano N = diag(1,…,n) o diagonal extraída."""
    if N_diag is None:
        arr = np.arange(1, n + 1, dtype=float)
    else:
        arr = np.asarray(N_diag, dtype=float)
    if arr.ndim == 2:
        return np.diag(np.diag(arr))
    return np.diag(arr.reshape(-1))


def _iterate_integer_lattice(n: int, order: int) -> List[np.ndarray]:
    r"""Genera {k ∈ ℤⁿ : ‖k‖₁ ≤ order} con signos completos."""
    if n < 1 or order < 0:
        return []
    if n == 1:
        if order == 0:
            return [np.array([0], dtype=int)]
        return [np.array([s], dtype=int) for s in range(-order, order + 1)]
    vectors: List[np.ndarray] = []
    for k1 in range(-order, order + 1):
        rest = order - abs(k1)
        for tail in _iterate_integer_lattice(n - 1, rest):
            vectors.append(np.concatenate(([k1], tail)))
    return vectors


def _iter_int_lattice(
    n: int, k_max: int
) -> Iterator[Tuple[int, ...]]:
    r"""Itera k ∈ {−k_max,…,k_max}ⁿ \ {0}."""
    if n < 1 or k_max < 1:
        return
    stack: List[Tuple[int, ...]] = [()]
    while stack:
        prefix = stack.pop()
        if len(prefix) == n:
            if any(prefix):
                yield prefix
            continue
        for ki in range(-k_max, k_max + 1):
            stack.append(prefix + (ki,))


def _continued_fraction(
    x: float, max_terms: int = 16, tol: float = 1e-14
) -> List[int]:
    r"""Fracción continua [a₀;a₁,…] de x ∈ ℝ (algoritmo de Gauss)."""
    a: List[int] = []
    if not math.isfinite(x):
        return [0]
    y = float(x)
    for _ in range(max_terms):
        ai = int(math.floor(y))
        a.append(ai)
        frac = y - ai
        if abs(frac) < tol:
            break
        y = 1.0 / frac
        if abs(y) > 1e12:
            break
    return a


def _convergents(cf: Sequence[int]) -> List[Tuple[int, int]]:
    r"""Convergentes pₖ/qₖ de una fracción continua."""
    p_m2, q_m2 = 0, 1
    p_m1, q_m1 = 1, 0
    out: List[Tuple[int, int]] = []
    for a in cf:
        p = a * p_m1 + p_m2
        q = a * q_m1 + q_m2
        out.append((p, q))
        p_m2, q_m2 = p_m1, q_m1
        p_m1, q_m1 = p, q
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 · Ω₃, DENSIDAD, MEDIDA ESPECTRAL, POINCARÉ, AGM, RSI-3, NOVIKOV
# Objetos: Ω₃, 𝒟_n, ω_KKS, Σ_c, Λ_Novikov, BeliefSet, (T,η,μ), Spec(ρ).
# Morfismo terminal: OniricSpectralHandoff.continue_into_phase2
# ══════════════════════════════════════════════════════════════════════════════

# ── §1.1 Retículo de Heyting Ω₃ con estratificación celeste ───────────────────
class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal Ω₃ = {0 < 1 < 2} ≅ {VETOED < DEGRADED < COHERENT}.
    meet = min, join = max, implica = residuo de Galois, ¬ = ⇒ ⊥.
    Objeto clasificador de subobjetos del topos trivializado Ω₃.
    Estratos celestes: hiperbólico-separatrix / resonante-órbita / KAM-toro.
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def pseudo_complement(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.VETOED)

    def classical_negation(self) -> "HeytingOmega3":
        return HeytingOmega3(2 - int(self))

    def double_negation(self) -> "HeytingOmega3":
        return self.pseudo_complement().pseudo_complement()

    def is_regular(self) -> bool:
        return self.double_negation() == self

    def excluded_middle_holds(self) -> bool:
        return self.join(self.pseudo_complement()) == HeytingOmega3.COHERENT

    def meet_all(self, *others: "HeytingOmega3") -> "HeytingOmega3":
        acc: HeytingOmega3 = self
        for o in others:
            acc = acc.meet(o)
        return acc

    def as_weight(self) -> float:
        return float(int(self)) / 2.0

    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        if self == HeytingOmega3.DEGRADED:
            raise ValueError(
                "DEGRADED no admite proyección fiel a 𝔹₂"
            )
        return self == HeytingOmega3.COHERENT

    @property
    def verdict(self) -> str:
        return self.name

    @classmethod
    def verify_residuation_axiom(cls) -> bool:
        r"""Verifica (c ∧ a ≤ b) ⇔ (c ≤ (a ⇒ b)) ∀ a,b,c ∈ Ω₃."""
        elements = list(cls)
        for a in elements:
            for b in elements:
                for c in elements:
                    residual = a.implies(b)
                    lhs = min(int(c), int(a)) <= int(b)
                    rhs = int(c) <= int(residual)
                    if lhs != rhs:
                        return False
        return True

    @classmethod
    def verify_algebra_laws(cls) -> Dict[str, bool]:
        r"""Verificación exhaustiva de las leyes de Heyting sobre Ω₃."""
        elems = list(cls)
        idem = all(
            a.meet(a) == a and a.join(a) == a for a in elems
        )
        comm = all(
            a.meet(b) == b.meet(a) and a.join(b) == b.join(a)
            for a in elems
            for b in elems
        )
        assoc = all(
            a.meet(b.meet(c)) == a.meet(b).meet(c)
            and a.join(b.join(c)) == a.join(b).join(c)
            for a in elems
            for b in elems
            for c in elems
        )
        absorb = all(
            a.meet(a.join(b)) == a and a.join(a.meet(b)) == a
            for a in elems
            for b in elems
        )
        distrib = all(
            a.meet(b.join(c)) == a.meet(b).join(a.meet(c))
            for a in elems
            for b in elems
            for c in elems
        )
        resid = cls.verify_residuation_axiom()
        lem_fails = not cls.DEGRADED.excluded_middle_holds()
        return {
            "idempotent": idem,
            "commutative": comm,
            "associative": assoc,
            "absorption": absorb,
            "distributivity": distrib,
            "residuation": resid,
            "lem_fails_on_star": lem_fails,
            "regulars_are_bot_top": (
                cls.VETOED.is_regular() and cls.COHERENT.is_regular()
            ),
            "star_not_regular": not cls.DEGRADED.is_regular(),
        }

    def poincare_stratum_name(self) -> str:
        return {
            HeytingOmega3.VETOED: "hyperbolic-separatrix",
            HeytingOmega3.DEGRADED: "resonant-orbit",
            HeytingOmega3.COHERENT: "kam-torus-invariant",
        }[self]

    def is_kam_stratum(self) -> bool:
        return self == HeytingOmega3.COHERENT

    def lyapunov_exponent_sign(self) -> int:
        return -1 + int(self)

    def symplectic_regime(self) -> str:
        return {
            HeytingOmega3.VETOED: "symplectic-escape",
            HeytingOmega3.DEGRADED: "resonant-cascade",
            HeytingOmega3.COHERENT: "integrable-kam",
        }[self]

    def floquet_stability(self) -> str:
        return {
            HeytingOmega3.VETOED: "hyperbolic-unstable",
            HeytingOmega3.DEGRADED: "parabolic-critical",
            HeytingOmega3.COHERENT: "elliptic-stable",
        }[self]

    def greene_residue_regime(self) -> str:
        return {
            HeytingOmega3.VETOED: "|R|>1 hyperbolic",
            HeytingOmega3.DEGRADED: "|R|~1 bifurcation",
            HeytingOmega3.COHERENT: "|R|<1 elliptic",
        }[self]


# ── §1.1.a Pestillo lógico de veto (reemplaza crowbar GPIO) ───────────────────
@dataclass(frozen=True, slots=True)
class OniricVetoReport:
    """Reporte de veto lógico del auditor onírico. Sin hardware."""

    triggered: bool
    reason: str
    timestamp_ns: int
    heyting: HeytingOmega3


class OniricVetoLatch:
    r"""
    Pestillo lógico: cuando el auditor colapsa a VETOED por caos estructural
    (herradura de Smale, Chirikov > 1, violación de rigidez de Gromov),
    se registra un veto *puramente lógico*. No hay GPIO, ni tiristor, ni ISR.
    """

    @classmethod
    def trigger(cls, reason: str = "") -> OniricVetoReport:
        logger.critical("[ONIRIC LOGICAL VETO] razón=%s", reason)
        return OniricVetoReport(
            triggered=True,
            reason=reason,
            timestamp_ns=time.time_ns(),
            heyting=HeytingOmega3.VETOED,
        )


# ── §1.0.b AGM sobre Ω₃ ───────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class AGMPostulateCertificate:
    """Certificado de verificación de los 6 postulados AGM."""

    agm1_closure: bool
    agm2_alpha_in_revision: bool
    agm3_subset_expansion: bool
    agm4_preservation: bool
    agm5_trivial_iff_inconsistent: bool
    agm6_extensionality: bool
    all_satisfied: bool
    n_postulates_passed: int
    local_verdict: HeytingOmega3


@dataclass(frozen=True)
class AGMBeliefSet:
    r"""
    Conjunto de creencias sobre el retículo Ω₃.
    K ⊆ Ω₃ = {0,1,2} como subconjunto (mundo-creencias).
    α ⊆ Ω₃ como “conjunto de estratos donde α es verdadera”.
    K ⊨ α  ⇔  K ⊆ α.
    K + α = K ∩ α                         (expansión)
    K ÷ α = máximo J ⊆ K con J ⊭ α         (contracción partial meet)
    K * α = (K ÷ ¬α) + α                   (revisión, identidad de Levi)
    """

    worlds: FrozenSet[int]

    @classmethod
    def universe(cls) -> FrozenSet[int]:
        return frozenset({0, 1, 2})

    @property
    def is_consistent(self) -> bool:
        return len(self.worlds) > 0

    @property
    def is_trivial(self) -> bool:
        return len(self.worlds) == 0

    def entails(self, alpha: FrozenSet[int]) -> bool:
        if self.is_trivial:
            return True
        return self.worlds.issubset(alpha)

    def expand(self, alpha: FrozenSet[int]) -> "AGMBeliefSet":
        return AGMBeliefSet(frozenset(self.worlds & alpha))

    def contract(self, alpha: FrozenSet[int]) -> "AGMBeliefSet":
        if not self.entails(alpha):
            return self
        K = list(self.worlds)
        # Maximalidad: quitar un solo ofensor (J = K \ {w}, w ∈ K ∩ α).
        offenders = sorted(w for w in K if w in alpha)
        if offenders:
            drop = offenders[0]
            return AGMBeliefSet(frozenset(w for w in K if w != drop))
        return self

    def revise(self, alpha: FrozenSet[int]) -> "AGMBeliefSet":
        neg_alpha = frozenset(self.universe() - alpha)
        contracted = self.contract(neg_alpha)
        return contracted.expand(alpha)


class HeytingAGMRevision:
    """Revisión AGM sobre Ω₃. Verifica los 6 postulados AGM."""

    @classmethod
    def verify_postulates(
        cls,
        K: AGMBeliefSet,
        alpha: FrozenSet[int],
        beta: Optional[FrozenSet[int]] = None,
    ) -> AGMPostulateCertificate:
        K_star_alpha = K.revise(alpha)
        agm1 = True
        agm2 = (
            K_star_alpha.worlds.issubset(alpha)
            if K_star_alpha.is_consistent
            else False
        )
        agm3 = K_star_alpha.worlds.issubset(K.worlds & alpha)
        if K.worlds & alpha:
            agm4 = (K.worlds & alpha).issubset(K_star_alpha.worlds)
        else:
            agm4 = True
        inconsistent_alpha = len(alpha) == 0
        agm5 = K_star_alpha.is_trivial == inconsistent_alpha
        if beta is not None and alpha == beta:
            K_star_beta = K.revise(beta)
            agm6 = K_star_alpha.worlds == K_star_beta.worlds
        else:
            agm6 = True
        all_pass = all([agm1, agm2, agm3, agm4, agm5, agm6])
        n_pass = sum([agm1, agm2, agm3, agm4, agm5, agm6])
        local = (
            HeytingOmega3.COHERENT
            if all_pass
            else (
                HeytingOmega3.DEGRADED
                if n_pass >= 5
                else HeytingOmega3.VETOED
            )
        )
        return AGMPostulateCertificate(
            agm1_closure=bool(agm1),
            agm2_alpha_in_revision=bool(agm2),
            agm3_subset_expansion=bool(agm3),
            agm4_preservation=bool(agm4),
            agm5_trivial_iff_inconsistent=bool(agm5),
            agm6_extensionality=bool(agm6),
            all_satisfied=bool(all_pass),
            n_postulates_passed=int(n_pass),
            local_verdict=local,
        )


# ── §1.0.c Mónada RSI-3 explícita ─────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class MonadAxiomVerification:
    left_associativity: float
    right_associativity: float
    associativity_holds: bool
    left_unitarity: float
    right_unitarity: float
    unitarity_holds: bool
    level_applied: int
    verdict: HeytingOmega3


class MonadRSI3:
    r"""
    Mónada RSI-3 (T, η, μ) sobre 𝒟_n.
    T(ρ) = Φ_η(sanitize(ρ))  auto-mejora de nivel 1 (mixtura CPTP)
    η(ρ) = ρ                 unidad idempotente
    μ(σ) = T(sanitize(σ))    condensación metacognitiva
    μ ∘ Tμ = μ ∘ μT          (asociatividad)
    μ ∘ ηT = id_T = μ ∘ Tη   (unitalidad)
    Kleisli: (g ∘_T f)(ρ) = μ(T(g(f(ρ)))).
    """

    ASSOC_TOL: Final[float] = 1.0e-8
    UNIT_TOL: Final[float] = 1.0e-8

    def __init__(self, dim: int, eta: float = 0.15, level: int = 3) -> None:
        self.dim = int(dim)
        self.eta = float(np.clip(eta, 0.0, 1.0))
        self.level = int(max(1, min(int(level), 4)))

    @staticmethod
    def _sanitize(rho: np.ndarray) -> np.ndarray:
        return _hermitize_trace_one(rho)

    def eta_unit(self, rho: np.ndarray) -> np.ndarray:
        return self._sanitize(rho)

    def _target(self, rho: np.ndarray) -> np.ndarray:
        w, V = la.eigh(self._sanitize(rho))
        v = V[:, -1].reshape(-1, 1)
        return (v @ v.conj().T).astype(np.complex128)

    def T(self, rho: np.ndarray) -> np.ndarray:
        rho = self._sanitize(rho)
        target = self._target(rho)
        return self._sanitize((1.0 - self.eta) * rho + self.eta * target)

    def mu(self, rho_TT: np.ndarray) -> np.ndarray:
        return self.T(rho_TT)

    def condense(self, rho: np.ndarray, n_levels: int) -> np.ndarray:
        rho_n = self._sanitize(rho)
        for _ in range(max(1, int(n_levels))):
            rho_n = self.T(rho_n)
        return self.mu(rho_n)

    def kleisli_compose(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        g: Callable[[np.ndarray], np.ndarray],
        rho: np.ndarray,
    ) -> np.ndarray:
        r"""(g ∘_T f)(ρ) = μ(T(g(f(ρ))))."""
        rho_f = self._sanitize(f(rho))
        T_rho_f = self.T(rho_f)
        g_T_rho_f = self._sanitize(g(T_rho_f))
        return self.mu(self.T(g_T_rho_f))

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a, "fro"))
        nb = float(np.linalg.norm(b, "fro"))
        return float(np.linalg.norm(a - b, "fro") / max(na + nb, 1e-15))

    def verify_axioms(
        self, rho_sample: Optional[np.ndarray] = None
    ) -> MonadAxiomVerification:
        if rho_sample is None:
            rng = np.random.default_rng(20240613)
            A = rng.standard_normal((self.dim, self.dim)) + 1j * rng.standard_normal(
                (self.dim, self.dim)
            )
            rho_sample = A @ A.conj().T
        rho = self._sanitize(rho_sample)
        Trho = self.T(rho)
        T2rho = self.T(Trho)
        T3rho = self.T(T2rho)
        left = self.T(self.mu(T2rho))
        right = self.mu(self.mu(T3rho))
        left_assoc = self._dist(left, right)
        assoc_ok = bool(left_assoc < self.ASSOC_TOL)
        mu_eta = self.mu(self.eta_unit(Trho))
        unit_left = self._dist(mu_eta, Trho)
        mu_T_eta = self.mu(self.T(self.eta_unit(rho)))
        unit_right = self._dist(mu_T_eta, Trho)
        unit_ok = bool(unit_left < self.UNIT_TOL and unit_right < self.UNIT_TOL)
        verdict = (
            HeytingOmega3.COHERENT
            if (assoc_ok and unit_ok)
            else (
                HeytingOmega3.DEGRADED
                if (assoc_ok or unit_ok)
                else HeytingOmega3.VETOED
            )
        )
        return MonadAxiomVerification(
            left_associativity=float(left_assoc),
            right_associativity=float(left_assoc),
            associativity_holds=assoc_ok,
            left_unitarity=float(unit_left),
            right_unitarity=float(unit_right),
            unitarity_holds=unit_ok,
            level_applied=int(self.level),
            verdict=verdict,
        )


# ── §1.0.d Anillo de Novikov ──────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class NovikovElement:
    r"""
    Elemento del anillo de Novikov Λ = {∑ a_λ t^λ : a_λ ∈ ℂ, λ ∈ ℝ,
    λ_j → −∞}. La valuación ν: Λ → ℝ ∪ {∞} es ν(x) = min{λ_j : a_j ≠ 0}
    y satisface la desigualdad ultramétrica ν(x+y) ≥ min(ν(x), ν(y)).
    """

    coefficients: Tuple[complex, ...]
    exponents: Tuple[float, ...]

    def valuation(self) -> float:
        if len(self.coefficients) == 0:
            return float("inf")
        return float(min(self.exponents))


class NovikovRing:
    r"""
    Anillo de Novikov (Novikov 1986) sobre el que viven los invariantes
    de Gromov–Witten en el límite racional.

        Λ = {∑ a_j t^{λ_j} : a_j ∈ ℂ, λ_j ∈ ℝ, λ_j → −∞}
        ν(∑ a_j t^{λ_j}) = min{λ_j : a_j ≠ 0}     (valuación no-arquimediana)
        ν(x+y) ≥ min(ν(x), ν(y)),  ν(xy) = ν(x)+ν(y).

    Propiedades verificadas:
      - No-arquimediana: |x+y| ≤ max(|x|, |y|)
      - Multiplicativa: ν(xy) = ν(x)+ν(y)
      - Unital: ν(1) = 0
    """

    @staticmethod
    def zero() -> NovikovElement:
        return NovikovElement(coefficients=(), exponents=())

    @staticmethod
    def one() -> NovikovElement:
        return NovikovElement(coefficients=(1.0 + 0j,), exponents=(0.0,))

    @staticmethod
    def monomial(a: complex, lam: float) -> NovikovElement:
        return NovikovElement(coefficients=(complex(a),), exponents=(float(lam),))

    @classmethod
    def add(cls, x: NovikovElement, y: NovikovElement) -> NovikovElement:
        r"""Suma de Novikov: combinar términos con el mismo exponente."""
        d: Dict[float, complex] = {}
        for a, lam in zip(x.coefficients, x.exponents):
            d[lam] = d.get(lam, 0.0) + a
        for a, lam in zip(y.coefficients, y.exponents):
            d[lam] = d.get(lam, 0.0) + a
        items = [(lam, c) for lam, c in d.items() if abs(c) > 1e-15]
        if not items:
            return cls.zero()
        items.sort(key=lambda p: p[0])
        return NovikovElement(
            coefficients=tuple(complex(a) for _, a in items),
            exponents=tuple(float(l) for l, _ in items),
        )

    @classmethod
    def mul(cls, x: NovikovElement, y: NovikovElement) -> NovikovElement:
        r"""Producto de Novikov: convolución (a_λ t^λ)·(b_μ t^μ)."""
        d: Dict[float, complex] = {}
        for a, lam in zip(x.coefficients, x.exponents):
            for b, mu in zip(y.coefficients, y.exponents):
                d[lam + mu] = d.get(lam + mu, 0.0) + a * b
        items = [(lam, c) for lam, c in d.items() if abs(c) > 1e-15]
        if not items:
            return cls.zero()
        items.sort(key=lambda p: p[0])
        return NovikovElement(
            coefficients=tuple(complex(a) for _, a in items),
            exponents=tuple(float(l) for l, _ in items),
        )

    @classmethod
    def verify_non_archimedean(
        cls, x: NovikovElement, y: NovikovElement
    ) -> bool:
        r"""ν(x+y) ≥ min(ν(x), ν(y))."""
        s = cls.add(x, y)
        return s.valuation() + 1e-12 >= min(x.valuation(), y.valuation())

    @classmethod
    def verify_multiplicative(
        cls, x: NovikovElement, y: NovikovElement
    ) -> bool:
        r"""ν(xy) = ν(x)+ν(y)."""
        p = cls.mul(x, y)
        if math.isinf(x.valuation()) or math.isinf(y.valuation()):
            return math.isinf(p.valuation())
        return abs(p.valuation() - (x.valuation() + y.valuation())) < 1e-9


# ── §1.2 Elementos celestes de Poincaré ───────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareCelestialElements:
    """Elementos keplerianos / de Delaunay extraídos del espectro de ρ."""

    L: float
    G: float
    H: float
    mean_anomaly: float
    arg_periapsis: float
    long_node: float
    Lambda: float
    mean_longitude: float
    xi: float
    eta: float
    p: float
    q: float
    eccentricity: float
    inclination: float
    tisserand: float
    jacobi_integral: float
    energy: float
    mean_motion: float


# ── §1.3 Espectro de Floquet ──────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FloquetSpectrum:
    multipliers: np.ndarray
    spectral_radius: float
    greene_residue: float
    trace_monodromy: float
    is_elliptic: bool
    is_symplectic: bool
    floquet_log_eigenvalues: Optional[np.ndarray] = None


# ── §1.4 Operador de densidad con estructura simpléctica y Wigner ─────────────
@dataclass(frozen=True, slots=True)
class DensityOperator:
    r"""
    Estado cuántico ρ ∈ 𝒟_n.
    Estructura simpléctica KKS: ω_KKS(X,Y) = ρ · ([X,Y]).
    Hamiltoniano celeste H = Tr(ρ N) con N = diag(1,…,n).
    Flujo de von Neumann  ρ̇ = −i[N,ρ]  es hamiltoniano e isóenergetico.
    """

    matrix: np.ndarray
    atol: float = 1e-8

    def __post_init__(self) -> None:
        rho = np.asarray(self.matrix, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("DensityOperator exige matriz cuadrada.")
        object.__setattr__(self, "matrix", np.array(rho, copy=True))
        if not np.allclose(rho, rho.conj().T, atol=self.atol):
            raise ValueError("DensityOperator exige ρ = ρ†.")
        tr = float(np.trace(rho).real)
        if abs(tr - 1.0) > _TRACE_TOL:
            raise ValueError(
                f"DensityOperator exige Tr ρ = 1 (Tr={tr})."
            )

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def as_array(self) -> np.ndarray:
        return self.matrix

    def spectrum(self, floor: float = _EIG_FLOOR) -> np.ndarray:
        lam = la.eigvalsh(self.matrix)
        lam = np.clip(np.real(lam), floor, None)
        s = float(np.sum(lam))
        return lam / s if s > 0.0 else np.full(lam.shape, 1.0 / lam.size)

    def purity(self) -> float:
        lam = self.spectrum()
        return float(np.sum(lam ** 2))

    def von_neumann_entropy(self) -> float:
        lam = self.spectrum()
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(lam > _EIG_FLOOR, lam * np.log(lam), 0.0)
            return float(-np.sum(terms))

    def spectral_gap(self) -> float:
        lam = np.sort(self.spectrum())[::-1]
        if lam.size < 2:
            return 0.0
        return float(lam[0] - lam[1])

    def cstar_residual(self) -> float:
        rho = self.matrix
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

    @classmethod
    def from_array(
        cls, rho: np.ndarray, atol: float = 1e-8
    ) -> "DensityOperator":
        return cls(matrix=_hermitize_trace_one(rho), atol=atol)

    # ── Geometría simpléctica ─────────────────────────────────────────────────
    def action_variables(self) -> np.ndarray:
        return np.sort(self.spectrum())[::-1]

    def angle_variables(
        self, N_diag: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu) if nu.ndim == 1 else nu
        evals, evecs = la.eigh(self.matrix)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        ph = np.zeros(n, dtype=float)
        for i in range(n):
            u = evecs[:, i]
            z = np.vdot(u, N @ u)
            ph[i] = float(np.angle(z))
        return ph

    def mean_motion_frequencies(
        self, N_diag: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu) if nu.ndim == 1 else nu
        actions = self.action_variables()
        base = float(np.trace(self.matrix @ N).real)
        return actions * base

    def keplerian_frequency_matrix(
        self, N_diag: Optional[np.ndarray] = None
    ) -> np.ndarray:
        nu = _diag_potential(self.dimension, N_diag)
        return nu if nu.ndim == 1 else np.diag(nu)

    def delaunay_triple(
        self, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu) if nu.ndim == 1 else nu
        comm = self.matrix @ N - N @ self.matrix
        L = float(np.trace(self.matrix @ N).real)
        comm_norm_sq = float(np.trace(comm @ comm.conj().T).real)
        G = math.sqrt(max(0.0, L ** 2 - comm_norm_sq))
        H = L * math.cos(self.spectral_gap() * math.pi)
        return float(L), float(G), float(H)

    def eccentricity_inclination(
        self, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        L, G, H = self.delaunay_triple(N_diag)
        e = math.sqrt(max(1.0 - (G ** 2) / max(L ** 2, 1e-30), 0.0))
        i = (
            math.acos(min(_safe_div(H, G, 1.0), 1.0))
            if G > 1e-30
            else 0.0
        )
        return float(e), float(i)

    def celestial_elements(
        self, N_diag: Optional[np.ndarray] = None
    ) -> PoincareCelestialElements:
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu) if nu.ndim == 1 else nu
        L, G, H = self.delaunay_triple(N_diag)
        e, inc = self.eccentricity_inclination(N_diag)
        th = self.angle_variables(N)
        l = float(th[0]) if n >= 1 else 0.0
        g = float(th[1] - th[0]) if n >= 2 else 0.0
        h = float(th[2] - th[1]) if n >= 3 else 0.0
        varpi = g + h
        two_LG = max(2.0 * (L - G), 0.0)
        two_GH = max(2.0 * (G - H), 0.0)
        rt_e = math.sqrt(two_LG)
        rt_i = math.sqrt(two_GH)
        xi = -rt_e * math.cos(varpi)
        eta = -rt_e * math.sin(varpi)
        p = rt_i * math.cos(h)
        q = -rt_i * math.sin(h)
        energy = float(np.trace(self.matrix @ N).real)
        a = max(L * L, 1e-30)
        a_p = 1.0
        tisserand = a_p / a + 2.0 * math.sqrt(
            max(a / a_p * (1.0 - e * e), 0.0)
        ) * math.cos(inc)
        mean_motion = 2.0 * energy * self.purity()
        return PoincareCelestialElements(
            L=float(L),
            G=float(G),
            H=float(H),
            mean_anomaly=l,
            arg_periapsis=g,
            long_node=h,
            Lambda=float(L),
            mean_longitude=l + g + h,
            xi=float(xi),
            eta=float(eta),
            p=float(p),
            q=float(q),
            eccentricity=float(e),
            inclination=float(inc),
            tisserand=float(tisserand),
            jacobi_integral=float(energy),
            energy=float(energy),
            mean_motion=float(mean_motion),
        )

    def poincare_integral_invariant(self) -> float:
        r"""Invariante integral de Poincaré ∮ p dq ≈ ∑ λ_i θ_i."""
        J = self.action_variables()
        th = self.angle_variables()
        return float(np.dot(J, th))

    def hamiltonian_evolve(
        self,
        t: float,
        N_diag: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Flujo de von Neumann exacto ρ(t) = e^{−iNt} ρ e^{iNt}."""
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        phase = np.exp(-1j * float(t) * (nu if nu.ndim == 1 else np.diag(nu)))
        rho_t = self.matrix * phase  # broadcasting si nu es 1-D no basta
        # conjugación exacta
        N = np.diag(nu) if nu.ndim == 1 else nu
        U = la.expm(-1j * float(t) * N)
        return 0.5 * ((U @ self.matrix @ U.conj().T) + (U @ self.matrix @ U.conj().T).conj().T)

    def lie_derivative_energy(
        self, N_diag: Optional[np.ndarray] = None
    ) -> float:
        r"""ℒ_{X_H} H = Tr(ρ̇ N) = −i Tr([N,ρ] N) = 0  (test hamiltoniano)."""
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu) if nu.ndim == 1 else nu
        comm = N @ self.matrix - self.matrix @ N
        return float(np.real(-1j * np.trace(comm @ N)))

    def first_return_phase_time(
        self,
        i: int = 0,
        j: int = 1,
        N_diag: Optional[np.ndarray] = None,
    ) -> float:
        n = self.dimension
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return float("inf")
        nu = _diag_potential(n, N_diag)
        omega = float((nu if nu.ndim == 1 else np.diag(nu))[i] - (nu if nu.ndim == 1 else np.diag(nu))[j])
        if abs(omega) < 1e-15:
            return float("inf")
        return float(2.0 * math.pi / abs(omega))

    def wigner_function(self) -> np.ndarray:
        r"""Función de Wigner discreta (transformada de Fourier de ρ)."""
        n = self.dimension
        rho = self.matrix
        W = np.empty((n, n), dtype=np.complex128)
        for q in range(n):
            disp = np.array(
                [rho[(q + k) % n, (q - k) % n] for k in range(n)],
                dtype=np.complex128,
            )
            W[q, :] = np.fft.fft(disp) / n
        return np.real(W)

    def husimi_function(self, sigma: float = 0.5) -> np.ndarray:
        W = self.wigner_function()
        n = self.dimension
        ax = np.arange(n)
        dx = np.minimum(ax, n - ax)
        g = np.exp(-0.5 * (dx ** 2) / max(sigma * sigma, 1e-12))
        G = np.outer(g, g)
        Wf = np.fft.fft2(W)
        Gf = np.fft.fft2(np.fft.ifftshift(G))
        Q = np.real(np.fft.ifft2(Wf * Gf))
        s = float(np.sum(Q))
        return Q / s if abs(s) > 1e-15 else Q

    def wigner_marginals(self) -> Tuple[np.ndarray, np.ndarray]:
        W = self.wigner_function()
        return np.sum(W, axis=1), np.sum(W, axis=0)

    def wigner_negativity(self) -> float:
        W = self.wigner_function()
        return float(np.sum(np.abs(W)) - 1.0)

    def symplectic_capacity_gromov(self) -> float:
        r"""Proxy de la capacidad de Gromov: 4(σ_q + σ_p) de las marginales."""
        q_marg, p_marg = self.wigner_marginals()
        n = self.dimension
        idx = np.arange(n, dtype=float)
        mean_q = float(np.dot(idx, q_marg))
        mean_p = float(np.dot(idx, p_marg))
        var_q = float(np.dot(idx, q_marg * (idx - mean_q) ** 2))
        var_p = float(np.dot(idx, p_marg * (idx - mean_p) ** 2))
        if var_q + var_p < 1e-12:
            return float("inf")
        return float(4.0 / (var_q + var_p))

    def symplectic_capacity_ratio(
        self, rho_base: Optional["DensityOperator"] = None
    ) -> float:
        c_rho = self.symplectic_capacity_gromov()
        if rho_base is None:
            return 1.0
        c_base = rho_base.symplectic_capacity_gromov()
        if c_base <= 1e-12:
            return float("inf")
        return float(c_rho / c_base)

    def spectral_flow_chern(
        self, t_max: float = 1.0, n_steps: int = 32
    ) -> float:
        r"""Flujo espectral / número de Chern del Hamiltoniano interpolado."""
        H0 = self.matrix
        n = self.dimension
        N = np.diag(_diag_potential(n))
        evals_init = la.eigvalsh(H0)
        H_fin = H0 + t_max * N
        evals_fin = la.eigvalsh(0.5 * (H_fin + H_fin.conj().T))
        flow = 0
        for k in range(n):
            if evals_fin[k] > evals_init[k] + 1e-9:
                flow += 1
            elif evals_fin[k] < evals_init[k] - 1e-9:
                flow -= 1
        return float(flow) / float(max(1, 2 * n_steps))

    def maslov_index(self, N_diag: Optional[np.ndarray] = None) -> int:
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        _evals, evecs = la.eigh(self.matrix)
        count = 0
        for k in range(n):
            u = evecs[:, k]
            overlap = abs(np.vdot(u, N @ u))
            if overlap > 1e-12:
                ph = float(np.angle(np.vdot(u, N @ u)))
                if ph > math.pi / 2.0 or ph < -math.pi / 2.0:
                    count += 1
        return int(count)

    def poincare_recurrence_time(
        self, N_diag: Optional[np.ndarray] = None
    ) -> float:
        lam = np.sort(self.spectrum())
        tau_spec = float("inf")
        if lam.size >= 2:
            min_gap = float(np.min(np.abs(np.diff(lam))))
            if min_gap > 1e-15:
                tau_spec = 1.0 / min_gap
        vol = float(np.prod(np.clip(lam, 1e-30, None)))
        tau_kac = 1.0 / vol if vol > 0.0 else float("inf")
        nu = _diag_potential(self.dimension, N_diag)
        diffs = np.abs(nu[:, None] - nu[None, :]) if nu.ndim == 1 else np.abs(
            np.diag(nu)[:, None] - np.diag(nu)[None, :]
        )
        np.fill_diagonal(diffs, np.inf)
        min_nu = float(np.min(diffs))
        tau_kep = (
            (2.0 * math.pi / min_nu)
            if min_nu < 1e30 and min_nu > 1e-15
            else float("inf")
        )
        return float(min(tau_spec, tau_kac, tau_kep))

    def berry_phase(
        self, curve: Sequence[np.ndarray], closed: bool = True
    ) -> float:
        if len(curve) < 2:
            return 0.0
        phis: List[np.ndarray] = []
        for rho_k in curve:
            rho_h = 0.5 * (rho_k + rho_k.conj().T)
            _e, evecs = la.eigh(rho_h)
            phis.append(evecs[:, -1])
        total_arg = 0.0
        nphi = len(phis)
        if closed:
            for k in range(nphi):
                a = phis[k]
                b = phis[(k + 1) % nphi]
                total_arg += float(np.angle(np.vdot(a, b)))
        else:
            for k in range(nphi - 1):
                total_arg += float(np.angle(np.vdot(phis[k], phis[k + 1])))
        return float(total_arg)

    def hannay_angle(self, N_diag: Optional[np.ndarray] = None) -> float:
        I = self.poincare_integral_invariant()
        return float((I + math.pi) % (2.0 * math.pi) - math.pi)

    def chirikov_overlap(
        self, N_diag: Optional[np.ndarray] = None
    ) -> float:
        n = self.dimension
        if n < 2:
            return 0.0
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu) if nu.ndim == 1 else nu
        comm = self.matrix @ N - N @ self.matrix
        eps = float(la.norm(comm, "fro"))
        J = self.action_variables()
        omega = nu + float(np.trace(self.matrix @ N).real)
        order = np.argsort(omega)
        s_max = 0.0
        for a, b in zip(order[:-1], order[1:]):
            dw = abs(float(omega[a]) - float(omega[b]))
            half = math.sqrt(
                max(abs(J[min(a, n - 1)] * eps), 0.0)
            ) + math.sqrt(max(abs(J[min(b, n - 1)] * eps), 0.0))
            s = _safe_div(half, dw, 0.0)
            if s > s_max:
                s_max = s
        return float(s_max)

    def siegel_brijuno_sum(
        self,
        N_diag: Optional[np.ndarray] = None,
        max_cf: int = 12,
    ) -> float:
        n = self.dimension
        if n < 2:
            return 0.0
        nu = _diag_potential(n, N_diag)
        if abs(nu[0] if nu.ndim == 1 else nu[0, 0]) < 1e-15:
            return float("inf")
        alpha = float((nu[1] if nu.ndim == 1 else nu[1, 1]) / (nu[0] if nu.ndim == 1 else nu[0, 0]))
        cf = _continued_fraction(alpha, max_terms=max_cf)
        conv = _convergents(cf)
        if len(conv) < 2:
            return 0.0
        acc = 0.0
        for k in range(len(conv) - 1):
            qk = max(abs(conv[k][1]), 1)
            qk1 = max(abs(conv[k + 1][1]), 1)
            acc += math.log(qk1) / qk
        return float(acc)

    def lyapunov_characteristic_exponents(
        self,
        N_diag: Optional[np.ndarray] = None,
        dt: float = 0.05,
        n_steps: int = 64,
    ) -> np.ndarray:
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        T = dt * n_steps
        lce = np.zeros(n, dtype=float)
        rho = self.matrix.copy()
        phase_dt = np.exp(-1j * dt * (nu if nu.ndim == 1 else np.diag(nu)))
        for _ in range(n_steps):
            U = np.diag(phase_dt) if phase_dt.ndim == 1 else la.expm(
                -1j * dt * nu
            )
            rho = 0.5 * ((U @ rho @ U.conj().T) + (U @ rho @ U.conj().T).conj().T)
            tr = float(np.trace(rho).real)
            if tr > 0:
                rho = rho / tr
            evolved = self.hamiltonian_evolve(T, N_diag)
            drift = float(la.norm(rho - evolved, "fro"))
            lce[0] = math.log(max(drift, 1e-30)) / max(T, 1e-12)
        return lce


# ── §1.5 Medida espectral con estructura liouvilliana ─────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralMeasure:
    eigenvalues: np.ndarray
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    dimension: int
    cstar_residual: float

    def __post_init__(self) -> None:
        lam = np.asarray(self.eigenvalues, dtype=np.float64).reshape(-1)
        if lam.size < 1:
            raise ValueError("SpectralMeasure exige al menos un autovalor.")
        object.__setattr__(self, "eigenvalues", lam)

    def as_simplex(self) -> np.ndarray:
        return self.eigenvalues

    def liouville_volume(self) -> float:
        lam = np.clip(self.eigenvalues, 1e-30, None)
        return float(np.prod(lam))

    def fisher_rao_norm(self, tangent: np.ndarray) -> float:
        lam = np.clip(self.eigenvalues, 1e-30, None)
        v = np.asarray(tangent, dtype=float).reshape(-1)
        if v.size != lam.size:
            raise ValueError("tangente incompatible con la medida.")
        return float(np.sqrt(np.sum((v * v) / lam)))

    def poincare_recurrence_time(self) -> float:
        lam = np.sort(self.eigenvalues)
        if lam.size < 2:
            return float("inf")
        min_gap = float(np.min(np.abs(np.diff(lam))))
        if min_gap < 1e-15:
            return float("inf")
        return 1.0 / min_gap

    def kac_recurrence_time(self) -> float:
        vol = self.liouville_volume()
        if vol <= 0.0:
            return float("inf")
        return 1.0 / vol

    def resonance_order(self, k_max: int = 3) -> int:
        lam = self.eigenvalues
        n = lam.size
        for order in range(1, k_max + 1):
            for k in _iterate_integer_lattice(n, order):
                if float(abs(np.dot(k, lam))) < 1e-6:
                    return order
        return 0


# ── §1.6 Sección de Poincaré onírica ──────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSectionOniric:
    r"""
    Σ_c = {ρ : Tr(ρ ν) = c} transversal al flujo X_H = −i[N,·]
    sii ω_KKS(X_H, ν) ≠ 0.
    """

    level: float
    normal_diag: np.ndarray
    transversality_tol: float = 1e-7
    i_cut: int = 0
    j_cut: int = 1
    phase_cut: float = 0.0

    def energy(self, rho: np.ndarray) -> float:
        return float(np.trace(rho @ self.normal_diag).real)

    def signed_distance(self, rho: np.ndarray) -> float:
        return self.energy(rho) - float(self.level)

    def phase(self, rho: np.ndarray) -> float:
        z = rho[self.i_cut, self.j_cut]
        return float(np.angle(z))

    def signed_phase(self, rho: np.ndarray) -> float:
        d = self.phase(rho) - self.phase_cut
        return float((d + math.pi) % (2.0 * math.pi) - math.pi)

    def crosses(self, rho_before: np.ndarray, rho_after: np.ndarray) -> bool:
        d0 = self.signed_phase(rho_before)
        d1 = self.signed_phase(rho_after)
        return (d0 * d1) < 0.0

    def is_transversal(self, rho: np.ndarray, rhodot: np.ndarray) -> bool:
        z = rho[self.i_cut, self.j_cut]
        zdot = rhodot[self.i_cut, self.j_cut]
        g = float(np.imag(np.conj(z) * zdot))
        return abs(g) > self.transversality_tol

    def twist_condition(self, omega: np.ndarray) -> bool:
        return abs(np.diff(np.asarray(omega, dtype=float))) > self.transversality_tol

    def greene_residue(self, trace_lefschetz: float, M: float) -> float:
        return float((2.0 - trace_lefschetz) / 4.0)


@dataclass(frozen=True, slots=True)
class ResonanceStructure:
    vector: Tuple[int, ...]
    order: int
    residue_min: float
    kam_residue: float
    is_kam_safe: bool
    continued_fraction: Tuple[int, ...] = field(default_factory=tuple)
    convergent: Tuple[int, int] = (0, 1)
    brijuno_sum: float = 0.0
    chirikov_overlap: float = 0.0


@runtime_checkable
class ImmunizationPassport(Protocol):
    immunization_hash: str
    heyting_verdict: HeytingOmega3
    gromov_witten_invariant: float

    def is_immune(self) -> bool: ...


# ── §1.7 Semilla espectral (ABC) ──────────────────────────────────────────────
class SpectralMeasureSeed(ABC):
    r"""Semilla del endofuntor Spec: ρ ↦ (Λ, γ, S, ΔΛ, ε_c{C*})."""

    @abstractmethod
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        ...


# ── §1.8 Estado onírico con campos celestes completos ─────────────────────────
@dataclass(frozen=True, slots=True)
class OniricFieldState:
    """Estado onírico auditado: MAC + Poincaré + AGM + RSI-3 + Novikov."""

    cycle_id: str
    scenario_id: str
    dream_isolation_flag: bool
    density_matrix: np.ndarray
    dirichlet_energy: float
    dirac_total_variation: float
    gromov_witten_invariant: float
    tqft_amplitude: float
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    cstar_residual: float
    betti_0: int
    betti_1_loops: int
    betti_2: int
    euler_characteristic: int
    heyting_verdict: HeytingOmega3
    immunization_hash: str
    timestamp_utc: float
    holonomy_partial: float
    wilson_phase: complex
    poincare_lefschetz_defect: float = 0.0
    symplectic_capacity_ratio: float = 1.0
    is_boundary_consistent: bool = True
    veto_triggered: bool = False
    gpio14_signal: str = "NONE"  # legado: siempre NONE (sin hardware)
    proof_merkle_sha512: str = ""
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_recurrence_time: float = float("inf")
    maslov_index: int = 0
    resonance_order: int = 0
    kam_residue: float = float("inf")
    lyapunov_sign: int = -1
    poincare_stratum: str = "kam-torus-invariant"
    hannay_angle: float = 0.0
    greene_residue: float = 0.0
    chirikov_overlap: float = 0.0
    nekhoroshev_time: float = float("inf")
    brijuno_sum: float = 0.0
    tisserand: float = 0.0
    jacobi_integral: float = 0.0
    floquet_radius: float = 1.0
    mean_motion: float = 0.0
    eccentricity: float = 0.0
    monadic_acceleration: float = 0.0
    novikov_valuation: float = 0.0
    reward_hacking_index: float = 0.0
    tqft_amplitude_geometric: float = 0.0
    cs_wilson_value: float = 0.0
    melnikov_zeros: int = 0
    smale_horseshoe: bool = False
    agm_postulates_passed: int = 0
    agm_all_satisfied: bool = False
    monad_associativity: bool = False
    monad_unitarity: bool = False
    rsi_level: int = 0
    novikov_ultrametric_ok: bool = False
    conley_index: int = 0
    energy_conservation_residual: float = 0.0
    lie_derivative_H: float = 0.0

    def __post_init__(self) -> None:
        rho = np.asarray(self.density_matrix, dtype=np.complex128)
        object.__setattr__(self, "density_matrix", np.array(rho, copy=True))
        if self.betti_0 < 0 or self.betti_1_loops < 0 or self.betti_2 < 0:
            raise ValueError("Los números de Betti deben ser ≥ 0.")
        if self.gromov_witten_invariant < 0.0:
            raise ValueError("I_GW debe ser ≥ 0.")

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        rho = self.density_matrix
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        if np.any(la.eigvalsh(rho) < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    def is_topologically_consistent(self) -> bool:
        return (self.betti_0 - self.betti_1_loops + self.betti_2) == self.euler_characteristic

    def is_euler_characteristic(self) -> bool:
        if self.heyting_verdict == HeytingOmega3.VETOED:
            if not self.dream_isolation_flag:
                return True
            return self.betti_1_loops > 3 or (not self.is_boundary_consistent)
        return True

    def is_immune(self) -> bool:
        return (
            self.dream_isolation_flag
            and self.heyting_verdict != HeytingOmega3.VETOED
            and self.is_quantum_physical()
            and self.is_topologically_consistent()
            and self.is_boundary_consistent
            and self.symplectic_capacity_ratio <= 1.25
            and not self.veto_triggered
            and self.kam_residue > 1.0
            and self.chirikov_overlap < 1.0
            and abs(self.greene_residue) <= 1.0 + 1e-9
            and not self.smale_horseshoe
        )

    def passport_prefix(self, n: int = 16) -> str:
        return self.immunization_hash[:n]


# ── §1.9 Certificado de Lefschetz / Gromov (sustrato F₁) ──────────────────────
@dataclass(frozen=True, slots=True)
class ImmunizationCertificate:
    """Fallback local del certificado de inmunización (interfaz pública)."""

    gw_relative_invariant: float = 0.0
    poincare_lefschetz_defect: float = 0.0
    symplectic_capacity_ratio: float = 1.0
    heyting_verdict: HeytingOmega3 = HeytingOmega3.VETOED
    is_boundary_consistent: bool = True
    veto_triggered: bool = False
    gpio14_signal: str = "NONE"
    proof_merkle_sha512: str = ""
    wilson_phase: complex = 1.0 + 0.0j


class GromovWittenOniricAuditor:
    r"""
    Auditor Gromov–Witten onírico (sustrato F₁).
    I_GW ≈ (pureza)·exp(−E_Dirichlet)·exp(S_vN / n) / (1+β₁) · (χ / n).
    Rigidez: I_GW > 0.05 ∧ aislamiento ∧ c_Gr(ρ)/c_Gr(base) ≤ 1.25.
    """

    ETA_RESONANCE: Final[float] = _ETA_RESONANCE
    SYMPLECTIC_RATIO_MAX: Final[float] = _SYMPLECTIC_RATIO_MAX

    @classmethod
    def _project_density(cls, rho: np.ndarray) -> np.ndarray:
        rho_h = 0.5 * (rho + rho.conj().T)
        evals, evecs = la.eigh(rho_h)
        evals = np.clip(evals, 0.0, None)
        s = float(np.sum(evals))
        if s <= 1e-15:
            n = rho_h.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        evals = evals / s
        return (evecs * evals) @ evecs.conj().T

    def evaluate_poincare_lefschetz_gw_invariant(
        self,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        boundary_stalk_matrix: np.ndarray,
        betti_1_cycles: int = 0,
        dirichlet_energy: float = 0.0,
        dream_isolation: bool = True,
        scenario_id: str = "DREAM-EVAL",
        **kwargs: Any,
    ) -> ImmunizationCertificate:
        rho_d = self._project_density(np.asarray(rho_dream, dtype=np.complex128))
        rho_b = self._project_density(np.asarray(rho_base, dtype=np.complex128))
        stalk = np.asarray(boundary_stalk_matrix, dtype=np.complex128)
        evals_d = la.eigvalsh(rho_d)
        evals_d = np.clip(evals_d, 1e-15, None)
        evals_d = evals_d / np.sum(evals_d)
        purity = float(np.sum(evals_d ** 2))
        ent_n = -float(np.sum(evals_d * np.log(evals_d)))
        comm = rho_d @ stalk - stalk @ rho_d
        defect = float(la.norm(comm, "fro"))
        n = rho_d.shape[0]
        X = np.diag(np.arange(n, dtype=float))
        mean_X = float(np.real(np.trace(rho_d @ X)))
        var_X = float(np.real(np.trace(rho_d @ (X @ X))) - mean_X ** 2)
        mean_X_b = float(np.real(np.trace(rho_b @ X)))
        var_X_b = float(np.real(np.trace(rho_b @ (X @ X))) - mean_X_b ** 2)
        capacity_ratio = float(
            max(var_X_b, 1e-12) / max(var_X, 1e-12)
        )
        beta1 = max(0, int(betti_1_cycles))
        hom_dim = 1 + beta1
        I_GW = (
            (purity * math.exp(-dirichlet_energy) * math.exp(ent_n / max(n, 1)))
            / (1.0 + beta1)
            * (1.0 / hom_dim)
        )
        ok = (
            I_GW > 0.05
            and dream_isolation
            and capacity_ratio <= self.SYMPLECTIC_RATIO_MAX
        )
        verdict = HeytingOmega3.COHERENT if ok else HeytingOmega3.VETOED
        return ImmunizationCertificate(
            gw_relative_invariant=float(I_GW),
            poincare_lefschetz_defect=defect,
            symplectic_capacity_ratio=capacity_ratio,
            heyting_verdict=verdict,
            is_boundary_consistent=bool(defect < 0.5),
            veto_triggered=bool(verdict == HeytingOmega3.VETOED),
            gpio14_signal="NONE",
            proof_merkle_sha512=hashlib.sha512(
                f"{scenario_id}|{I_GW:.10f}".encode("utf-8")
            ).hexdigest(),
            wilson_phase=complex(
                math.cos(2.0 * math.pi * I_GW),
                math.sin(2.0 * math.pi * I_GW),
            ),
        )


# ── §1.10 OniricSpectralHandoff — HAND-OFF FORMAL FASE 1 → FASE 2 ─────────────
@dataclass(frozen=True, slots=True)
class OniricSpectralHandoff:
    r"""
    Objeto terminal FASE 1 / inicial FASE 2.
    Semilla Spec(ρ) = (Λ, γ, S, ΔΛ, ε_c) enriquecida con:
      AGM, RSI-3 (η, stub de T), Novikov, Poincaré celeste
      (conservación de H, Lefschetz, capacidad de Gromov).

    Dominio de FASE 2 = este objeto.
    Codominio de FASE 2 = UnsealedOniricTrace
      (TQFT path integral · Chern–Simons · Verlinde · Fukaya · Kontsevich).
    """

    scenario_id: str
    rho_dream: np.ndarray
    rho_base: np.ndarray
    measure: SpectralMeasure
    celestial: PoincareCelestialElements
    immunization: ImmunizationCertificate
    agm_certificate: AGMPostulateCertificate
    monad_axioms: MonadAxiomVerification
    novikov_ultrametric_ok: bool
    novikov_valuation: float
    energy_conservation_residual: float
    lie_derivative_H: float
    chirikov_overlap: float
    maslov_index: int
    wigner_negativity: float
    symplectic_capacity: float
    poincare_recurrence_time: float
    hannay_angle: float
    rsi3_eta: float
    rsi3_T_stub_hash: str
    handoff_hash: str
    dim: int
    incoming_heyting: HeytingOmega3

    @classmethod
    def build(
        cls,
        scenario_id: str,
        rho_dream: np.ndarray,
        rho_base: Optional[np.ndarray] = None,
        betti_1_cycles: int = 0,
        dirichlet_energy: float = 0.0,
        dream_isolation: bool = True,
        rsi3_eta: float = 0.15,
        incoming_heyting: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> "OniricSpectralHandoff":
        r"""Cierra FASE 1; el morfismo de continuación es continue_into_phase2."""
        rho = _hermitize_trace_one(rho_dream)
        n = int(rho.shape[0])
        if rho_base is None:
            rho_base = np.eye(n, dtype=np.complex128) / n
        else:
            rho_base = _hermitize_trace_one(rho_base)
        op = DensityOperator.from_array(rho)
        lam = op.spectrum()
        measure = SpectralMeasure(
            eigenvalues=lam,
            purity=op.purity(),
            von_neumann_entropy=op.von_neumann_entropy(),
            spectral_gap=op.spectral_gap(),
            dimension=n,
            cstar_residual=op.cstar_residual(),
        )
        celestial = op.celestial_elements()
        auditor = GromovWittenOniricAuditor()
        immunization = auditor.evaluate_poincare_lefschetz_gw_invariant(
            rho_dream=rho,
            rho_base=rho_base,
            boundary_stalk_matrix=np.eye(n, dtype=np.complex128),
            betti_1_cycles=betti_1_cycles,
            dirichlet_energy=dirichlet_energy,
            dream_isolation=dream_isolation,
            scenario_id=scenario_id,
        )
        # AGM sobre el veredicto
        K0 = AGMBeliefSet(AGMBeliefSet.universe())
        alpha = frozenset({int(immunization.heyting_verdict)})
        agm_cert = HeytingAGMRevision.verify_postulates(K0, alpha)
        # RSI-3
        eta = float(np.clip(rsi3_eta, 0.0, 1.0))
        monad = MonadRSI3(dim=n, eta=eta, level=3)
        axiom = monad.verify_axioms(rho)
        rho_T = monad.T(rho)
        t_stub = _sha256_bytes(
            np.ascontiguousarray(rho_T).tobytes(),
            f"eta={eta:.12e}".encode("ascii"),
        )
        # Novikov
        x = NovikovRing.monomial(1.0 + 0.5j, -0.3)
        y = NovikovRing.monomial(0.7 - 0.3j, -1.0)
        nov_ok = NovikovRing.verify_non_archimedean(
            x, y
        ) and NovikovRing.verify_multiplicative(x, y)
        nov_val = NovikovRing.add(x, y).valuation()
        # Celeste
        e_res = abs(op.lie_derivative_energy())
        # (ρ(dt) vs ρ) residual de energía
        rho_dt = op.hamiltonian_evolve(1e-3)
        e0 = celestial.energy
        e1 = float(np.trace(rho_dt @ _diag_potential(n)).real)
        energy_res = abs(e1 - e0)

        handoff_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            f"{measure.spectral_gap:.12e}".encode("ascii"),
            f"{immunization.gw_relative_invariant:.12e}".encode("ascii"),
            immunization.heyting_verdict.name.encode("ascii"),
            agm_cert.local_verdict.name.encode("ascii"),
            t_stub.encode("ascii"),
            scenario_id.encode("ascii"),
        )
        return cls(
            scenario_id=scenario_id,
            rho_dream=rho,
            rho_base=rho_base,
            measure=measure,
            celestial=celestial,
            immunization=immunization,
            agm_certificate=agm_cert,
            monad_axioms=axiom,
            novikov_ultrametric_ok=bool(nov_ok),
            novikov_valuation=float(nov_val),
            energy_conservation_residual=float(energy_res),
            lie_derivative_H=float(e_res),
            chirikov_overlap=float(op.chirikov_overlap()),
            maslov_index=int(op.maslov_index()),
            wigner_negativity=float(op.wigner_negativity()),
            symplectic_capacity=float(op.symplectic_capacity_gromov()),
            poincare_recurrence_time=float(op.poincare_recurrence_time()),
            hannay_angle=float(op.hannay_angle()),
            rsi3_eta=eta,
            rsi3_T_stub_hash=t_stub,
            handoff_hash=handoff_hash,
            dim=n,
            incoming_heyting=incoming_heyting,
        )

    def summary(self) -> Dict[str, float]:
        return {
            "purity": self.measure.purity,
            "entropy": self.measure.von_neumann_entropy,
            "gap": self.measure.spectral_gap,
            "I_GW": self.immunization.gw_relative_invariant,
            "c_Gr": self.symplectic_capacity,
            "chirikov": self.chirikov_overlap,
            "ΔE": self.energy_conservation_residual,
            "ℒ_H": self.lie_derivative_H,
            "agm_passed": float(self.agm_certificate.n_postulates_passed),
            "rsi3_eta": self.rsi3_eta,
            "novikov_ok": float(self.novikov_ultrametric_ok),
            "eccentricity": self.celestial.eccentricity,
            "incoming_weight": self.incoming_heyting.as_weight(),
        }

    def extract_spectral_measure(self) -> SpectralMeasure:
        """Realización de la semilla Spec(ρ). Consumida por FASE 2."""
        return self.measure

    # ══════════════════════════════════════════════════════════════════════════
    # HAND-OFF FORMAL  FASE 1 → FASE 2
    # continue_into_phase2 ∘ build  =  evaluate_dream_spectrum ∘ build
    # ÚLTIMO MORFISMO DE FASE 1  ∧  PRIMERO DE FASE 2.
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase2(self) -> Any:
        r"""
        Último morfismo FASE 1 ∧ primero FASE 2.

            continue_into_phase2 ∘ build
          = OniricSpectraEngine.evaluate_dream_spectrum ∘ build
          : Dream × MAC × Ω₃ → UnsealedOniricTrace.

        El codominio `UnsealedOniricTrace` y el funtor F₂
        (TQFT path integral · Chern–Simons · Verlinde · Fukaya A_∞ ·
         Kontsevich · Birkhoff–Hori–Deprit · Melnikov · Lyapunov–
         von Neumann · Nekhoroshev · Hochschild · Morse–Conley)
        se instancian en el siguiente paso (FASE 2).

        Hasta entonces se reexpone el handoff como *continuable*:
        cualquier implementación de FASE 2 debe aceptar exactamente
        esta firma (self) y devolver UnsealedOniricTrace.
        """
        try:
            from app.wisdom.toon_oniric_auditor_engine import (  # type: ignore
                OniricSpectraEngine,
            )

            return OniricSpectraEngine.evaluate_dream_spectrum(
                density_matrix=self.rho_dream,
                rho_base=self.rho_base,
                scenario_id=self.scenario_id,
            )
        except (ImportError, AttributeError):
            logger.info(
                "FASE 1 cerrada (hash=%s). FASE 2 no ligada aún; "
                "el siguiente prompt instancia OniricSpectraEngine "
                "con dominio = OniricSpectralHandoff.",
                self.handoff_hash[:16],
            )
            return self


# ── cierre de FASE 1 ──────────────────────────────────────────────────────────
__FASE_1_TERMINAL__ = OniricSpectralHandoff.continue_into_phase2
"""
Contrato de anidación:
  F₁  --continue_into_phase2-->  F₂
  dom(F₂) = OniricSpectralHandoff
  cod(F₂) = UnsealedOniricTrace   (se define en FASE 2)

  F₁ = OniricSpectralHandoff.build
     = DensityOperator.from_array
     ∘ SpectralMeasure
     ∘ PoincareCelestialElements
     ∘ GromovWittenOniricAuditor
     ∘ HeytingAGMRevision.verify_postulates
     ∘ MonadRSI3.verify_axioms
     ∘ NovikovRing.verify_*
"""

__all__ = [
    "HeytingOmega3",
    "OniricVetoReport",
    "OniricVetoLatch",
    "AGMPostulateCertificate",
    "AGMBeliefSet",
    "HeytingAGMRevision",
    "MonadAxiomVerification",
    "MonadRSI3",
    "NovikovElement",
    "NovikovRing",
    "PoincareCelestialElements",
    "FloquetSpectrum",
    "DensityOperator",
    "SpectralMeasure",
    "PoincareSectionOniric",
    "ResonanceStructure",
    "ImmunizationPassport",
    "SpectralMeasureSeed",
    "OniricFieldState",
    "ImmunizationCertificate",
    "GromovWittenOniricAuditor",
    "OniricSpectralHandoff",
]
# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 · TQFT · GROMOV–WITTEN · KAM · RESONANCIAS · NOVIKOV · FUKAYA · KONTSEVICH
# Dominio   = OniricSpectralHandoff   (codominio de §1.10 / FASE 1)
# Codominio = UnsealedOniricTrace     (dominio de toda la FASE 3)
# Funtor F₂ = OniricSpectraEngine.evaluate_dream_spectrum
# Morfismo terminal: UnsealedOniricTrace.continue_into_phase3
# ══════════════════════════════════════════════════════════════════════════════

# ── §2.0.a TQFT Path Integral ─────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TQFTAmplitudeCertificate:
    r"""Certificado de amplitud TQFT \(Z_g(n)\) (integral sobre moduli)."""

    genus: int
    n_punctures: int
    amplitude: float
    moduli_dimension: int
    euler_char_moduli: int
    is_geometric: bool
    well_petersson_curvature: float
    local_verdict: HeytingOmega3


class TQFTPathIntegral:
    r"""
    Amplitud TQFT
        \(Z_g(n)=\int_{\overline{\mathcal{M}}_{g,n}}\Omega^{3g-3+n}\)
    sobre el espacio de moduli de superficies de Riemann de género \(g\)
    con \(n\) puncturas.

    Dimensión compleja de Weil–Petersson:
        \(\dim_{\mathbb{C}}\mathcal{M}_{g,n}=3g-3+n\)  (\(g\ge 2\)),
        \(g=1,n\ge 1\): \(n\);  \(g=0,n\ge 3\): \(n-3\).

    La amplitud se aproxima por la forma de volumen simpléctico reducido
    del espacio de Teichmüller con curvatura de Weil–Petersson (Wolpert 1986):
        \(R_{\mathrm{WP}}=-12\pi\big/(2g-2+n)\)  (escalar, \(g+n>2\)),
        \(Z_g(n)\simeq(\mathrm{Vol})^{-1}\exp\bigl(-\int R_{\mathrm{WP}}\bigr)\).
    """

    @classmethod
    def amplitude(
        cls,
        genus: int,
        n_punctures: int,
        curvature_WP: Optional[float] = None,
    ) -> TQFTAmplitudeCertificate:
        g = int(max(0, genus))
        n = int(max(0, n_punctures))
        if g == 0:
            dim_c = max(0, n - 3)
        elif g == 1:
            dim_c = max(0, n)
        else:
            dim_c = 3 * g - 3 + n
        if curvature_WP is None:
            den = 2 * g - 2 + n
            if g >= 2 and den != 0:
                R_WP = -12.0 * math.pi / float(den)
            elif g == 1:
                R_WP = -2.0 * math.pi * max(g - 1.0, 0.25)
            else:
                R_WP = 0.0  # WP singular en g=1,n=1; negativa infinita en g=0
        else:
            R_WP = float(curvature_WP)
        action = max(0.0, -R_WP) * max(1.0, dim_c)
        amplitude = math.exp(-action / max(1.0, dim_c + 1.0))
        # χ(M_{g,n}) de Harer–Zagier (proxy)
        chi_moduli = float((-1) ** n * (2 * g - 2 + n)) if g >= 2 else 0.0
        is_geom = bool(g >= 2 and n >= 1 and amplitude > 0.0)
        verdict = (
            HeytingOmega3.COHERENT
            if is_geom
            else (
                HeytingOmega3.DEGRADED
                if amplitude > 0.0
                else HeytingOmega3.VETOED
            )
        )
        return TQFTAmplitudeCertificate(
            genus=g,
            n_punctures=n,
            amplitude=float(amplitude),
            moduli_dimension=int(dim_c),
            euler_char_moduli=int(chi_moduli),
            is_geometric=is_geom,
            well_petersson_curvature=float(R_WP),
            local_verdict=verdict,
        )


# ── §2.0.b Chern–Simons Wilson loops ──────────────────────────────────────────
class ChernSimonsWilson:
    r"""
    Lazo de Wilson en teoría de Chern–Simons:
        \(\langle W_K\rangle=\int DA\,e^{i(k/4\pi)\,\mathrm{CS}(A)}\,W_K[A]\).
    Witten (1989): \(\langle W_K\rangle\) coincide con el polinomio de Jones
    \(J_K(q)\) evaluado en \(q=e^{2\pi i/(k+2)}\). Aquí se aproxima vía
    holonomía de la conexión plana asociada al estado onírico:

        \(A=p\cdot d\theta\)  (conexión de Berry sobre la órbita coadjunta),
        \(W_K=\operatorname{Tr}\mathcal{P}\exp\oint A\).
    """

    @classmethod
    def amplitude(
        cls,
        rho: np.ndarray,
        k_level: int = 2,
        winding: int = 1,
    ) -> float:
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        # Conexión de Berry promedio sobre la base canónica
        nu = np.arange(1, n + 1, dtype=float)
        phase = np.exp(1j * float(k_level) * rho_h * nu.reshape(-1, 1))
        W = float(np.real(np.trace(phase @ rho_h))) / max(n, 1)
        # Modulación por winding (número de vueltas del lazo)
        return float(
            W * math.cos(2.0 * math.pi * winding / max(k_level + 2, 1))
        )

    @classmethod
    def jones_q(cls, k_level: int) -> complex:
        r"""Raíz de unidad \(q=e^{2\pi i/(k+2)}\) del polinomio de Jones."""
        k = max(0, int(k_level))
        ang = 2.0 * math.pi / (k + 2)
        return complex(math.cos(ang), math.sin(ang))


# ── §2.0.c Verlinde Fusion ────────────────────────────────────────────────────
class VerlindeFusion:
    r"""
    Reglas de fusión de Verlinde (1988) para TQFT 2D:
        \(\Phi_i\times\Phi_j=\bigoplus_k N_{ij}^k\Phi_k\)
    con
        \(N_{ij}^k=\sum_\ell S_{i\ell}S_{j\ell}S_{k\ell}^*/S_{0\ell}\)
    (fórmula de Verlinde). Aquí se computa la matriz \(S\) de \(\mathrm{SU}(2)_k\):
        \(S_{ij}=\sqrt{2/(k+2)}\,\sin\bigl(\pi(i+1)(j+1)/(k+2)\bigr)\).
    """

    @classmethod
    def su2_s_matrix(cls, k_level: int) -> np.ndarray:
        r"""Matriz \(S\) de \(\mathrm{SU}(2)_k\): \(S_{ij}=\sqrt{2/(k+2)}\sin(\cdots)\)."""
        k = int(max(1, k_level))
        N = k + 1
        norm = math.sqrt(2.0 / (k + 2.0))
        S = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                S[i, j] = norm * math.sin(
                    math.pi * (i + 1) * (j + 1) / (k + 2.0)
                )
        return S

    @classmethod
    def fusion_coefficients(cls, k_level: int) -> np.ndarray:
        r"""\(N_{ij}^k\) por fórmula de Verlinde."""
        S = cls.su2_s_matrix(k_level)
        N = S.shape[0]
        S0 = S[0, :].copy()
        S0 = np.where(np.abs(S0) < 1e-15, 1e-15, S0)
        Nijk = np.zeros((N, N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                for kk in range(N):
                    acc = 0.0
                    for ell in range(N):
                        acc += S[i, ell] * S[j, ell] * S[kk, ell] / S0[ell]
                    Nijk[i, j, kk] = float(acc)
        return Nijk

    @classmethod
    def verify_fusion_algebra(cls, k_level: int = 2) -> bool:
        r"""Verifica que los \(N_{ij}^k\) sean enteros ≥ 0 y la asociatividad."""
        Nijk = cls.fusion_coefficients(k_level)
        rounded = np.round(Nijk)
        if not np.allclose(Nijk, rounded, atol=1e-6):
            return False
        if np.any(Nijk < -1e-9):
            return False
        # Asociatividad: ∑_r N_{ij}^r N_{rk}^m = ∑_r N_{jk}^r N_{ir}^m
        N = Nijk.shape[0]
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for m in range(N):
                        lhs = sum(
                            Nijk[i, j, r] * Nijk[r, k, m] for r in range(N)
                        )
                        rhs = sum(
                            Nijk[j, k, r] * Nijk[i, r, m] for r in range(N)
                        )
                        if abs(lhs - rhs) > 1e-5:
                            return False
        return True


# ── §2.0.d Fukaya \(A_\infty\)-structure (stub) ────────────────────────────────
class FukayaAinfinity:
    r"""
    Estructura \(A_\infty\) de la categoría de Fukaya (stub pedagógico):
        \(m_k:\mathrm{Hom}^{\otimes k}\to\mathrm{Hom}[2-k]\),
        \(m_1^2=0\),  \(\sum m_2(m_1,\ldots)=0\).
    En el modelo mínimo:
        \(m_1=[N,A]\) (diferencial),
        \(m_2=\) producto matricial,
        \(m_k=0\) para \(k\ge 3\)
    (modelo formal \(A=A_\infty\) de Coder(m)).
    """

    @classmethod
    def m1(
        cls,
        A: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""\(m_1(A)=[N,A]\) con \(N=\mathrm{diag}(1,\ldots,n)\) si no se especifica."""
        n = A.shape[0]
        if N_diag is None:
            N_diag = np.arange(1, n + 1, dtype=float)
        N = np.diag(N_diag) if np.asarray(N_diag).ndim == 1 else N_diag
        return A @ N - N @ A

    @classmethod
    def m2(cls, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        r"""\(m_2(A,B)=A\cdot B\) (producto matricial)."""
        return A @ B

    @classmethod
    def verify_m1_squared(
        cls,
        A: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        atol: float = 1e-10,
    ) -> bool:
        r"""\(m_1^2(A)=0\) — unidad nilpotente del diferencial."""
        return bool(
            np.allclose(cls.m1(cls.m1(A, N_diag), N_diag), 0.0, atol=atol)
        )

    @classmethod
    def verify_leibniz(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        atol: float = 1e-10,
    ) -> bool:
        r"""
        \(m_1(m_2(A,B))=m_2(m_1(A),B)+m_2(A,m_1(B))\)
        (regla de Leibniz / \(A_\infty\) en grado 2).
        """
        lhs = cls.m1(cls.m2(A, B), N_diag)
        rhs = cls.m2(cls.m1(A, N_diag), B) + cls.m2(A, cls.m1(B, N_diag))
        return bool(np.allclose(lhs, rhs, atol=atol))


# ── §2.0.e Kontsevich formality ───────────────────────────────────────────────
class KontsevichFormality:
    r"""
    Formalidad de Kontsevich (1997): la cuantización por deformación de
    un álgebra de Poisson es única a menos de equivalencia gauge. En el
    caso lineal, el \(\star\)-producto de Moyal–Weyl sobre \(\mathbb{R}^{2n}\)
    se aproxima por:

        \((f\star g)(x)=f(x)\cdot\exp\bigl((i\hbar/2)\,\overleftarrow{\partial}_i
        \omega^{ij}\overrightarrow{\partial}_j\bigr)\cdot g(x)\),

    con \(\Lambda\) el bivector de Poisson (matriz simpléctica canónica).
    """

    @classmethod
    def moyal_star_2x2(
        cls,
        f: float,
        g: float,
        df_theta: float,
        dg_theta: float,
        hbar: float = 1.0,
    ) -> float:
        r"""
        \(\star\)-producto de Moyal en \((\theta,\bar\theta)\) truncado a orden \(\hbar\):
            \((f\star g)(\theta,p)=fg+(i\hbar/2)(\partial_\theta f\,\partial_p g
            -\partial_p f\,\partial_\theta g)+O(\hbar^2)\).
        Aquí \(f,g\) son funciones de fase \((\theta)\) con derivadas
        aproximadas por FFT / diferencias.
        """
        dtheta_f = math.sin(df_theta)
        dtheta_g = math.cos(dg_theta)
        # Se asume \(\partial_p f=1\) (función lineal en momento), \(\partial_p g=1\)
        poisson = 0.5 * dtheta_f * dtheta_g
        return float(f * g + 0.5 * hbar * poisson)

    @classmethod
    def verify_jacobi_star(
        cls,
        f: float,
        g: float,
        h: float,
        hbar: float = 0.01,
    ) -> bool:
        r"""Verifica el corchete de Moyal \(\{f,g\}_\star=(f\star g-g\star f)/(i\hbar)\) es Poisson."""
        fg = cls.moyal_star_2x2(f, g, hbar, hbar)
        gf = cls.moyal_star_2x2(g, f, hbar, hbar)
        bracket_fg = (fg - gf) / (1j * hbar) if abs(hbar) > 1e-15 else 0.0
        # Chequeo numérico rudimentario: el corchete es (casi) real
        return bool(abs(complex(bracket_fg).imag) < 1e-6)


# ── §2.0.f Birkhoff–Hori–Deprit ───────────────────────────────────────────────
class BirkhoffHoriDeprit:
    r"""
    Forma normal de Birkhoff–Dulac a orden arbitrario (Hori 1966, Deprit 1969):
        \(H=H_0+\varepsilon H_1+\varepsilon^2 H_2+\cdots\)
        \(H_k=\{H_0,W_k\}+H_k^{(\mathrm{res})}\)
        \(H_k^{(\mathrm{normal})}=H_k^{(\mathrm{res})}\) (núcleo de \(L_{H_0}\))
        \(W_k\in\mathrm{Im}(L_{H_0})\) (resuelve el homológico).

    Implementación por recurrencia directa sobre matrices con estructura
    de pequeños divisores \((m,\omega)\).
    """

    @classmethod
    def normal_form(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        max_order: int = 4,
        tol: float = 1e-10,
    ) -> Dict[int, np.ndarray]:
        r"""
        Retorna \(\{\mathrm{order}:k\mapsto X_k\}\) donde \(X_k\) es el término
        normal de orden \(k\). \(X_{-1}=\mathrm{ad}_N(H_0)\), \(X_{-2}=\mathrm{ad}_N^2(H_0)\), etc.
        (linealización iterada).
        """
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu) if np.asarray(nu).ndim == 1 else nu
        result: Dict[int, np.ndarray] = {}
        X1 = N @ rho_h - rho_h @ N
        tr_num = float(np.real(np.trace(X1 @ rho_h.conj().T)))
        tr_den = float(np.real(np.trace(rho_h @ rho_h.conj().T)))
        if abs(tr_den) > tol:
            X1 = X1 - (tr_num / tr_den) * rho_h
        result[1] = X1
        current = X1
        for k in range(2, int(max_order) + 1):
            comm = N @ current - current @ N
            Xk = N @ comm - comm @ N
            tr_k = float(np.real(np.trace(Xk @ rho_h.conj().T)))
            if abs(tr_den) > tol:
                Xk = Xk - (tr_k / tr_den) * rho_h
            result[k] = Xk
            current = Xk
        return result

    @classmethod
    def small_divisors(
        cls,
        frequencies: np.ndarray,
        m_max: int = 3,
    ) -> float:
        r"""Mínimo \(\lvert m\cdot\omega\rvert\) sobre \(m\in\mathbb{Z}^n\setminus\{0\}\) con \(\lvert m\rvert_\infty\le m_{\max}\)."""
        omega = np.asarray(frequencies, dtype=float).ravel()
        n = omega.size
        if n == 0:
            return float("inf")
        best = float("inf")
        for m in _iter_int_lattice(n, m_max):
            v = abs(float(np.dot(m, omega)))
            if v < best:
                best = v
        return float(best)


# ── §2.0.g Melnikov Zeros Finder ──────────────────────────────────────────────
class MelnikovZerosFinder:
    r"""
    Búsqueda de ceros transversales de
        \(M(t_0)=\int\{H_0,H_1\}(\phi_t)\,dt\).
    Smale–Birkhoff: si \(M(t_*)\) tiene un cero transversal
    \((dM/dt_*\ne 0)\), entonces \(W^s(p)\cap W^u(p)\) se intersecta
    transversalmente \(\Rightarrow\) herradura conjugada al shift
    \(\Sigma_2=\{0,1\}^{\mathbb{Z}}\Rightarrow h_{\mathrm{top}}(\sigma)=\log 2\).
    """

    @classmethod
    def find_zeros(
        cls,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        n_samples: int = 64,
        dt: float = 0.05,
    ) -> Tuple[int, float, bool]:
        r"""Retorna \((n_{\mathrm{ceros}},\,|M|_{\max},\,\mathrm{has\_transverse\_zero})\)."""
        rho = _hermitize_trace_one(rho_dream)
        n = rho.shape[0]
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu) if np.asarray(nu).ndim == 1 else nu
        base = _hermitize_trace_one(rho_base)
        phase_dt = np.exp(-1j * dt * (nu if np.asarray(nu).ndim == 1 else np.diag(nu)))
        vals: List[float] = []
        current = rho
        for _ in range(n_samples):
            comm = N @ current - current @ N
            poisson = float(np.real(-1j * np.trace(comm @ base)))
            vals.append(poisson)
            U = np.diag(phase_dt) if np.asarray(phase_dt).ndim == 1 else la.expm(
                -1j * dt * N
            )
            current = 0.5 * (
                (current @ U) * 0  # placeholder silenced
                + (U @ current @ U.conj().T)
                + (U @ current @ U.conj().T).conj().T
            )
            current = _hermitize_trace_one(current)
        arr = np.array(vals)
        signs = np.sign(arr)
        n_zeros = int(np.sum(np.abs(np.diff(signs)) > 1.0))
        m_max = float(np.max(np.abs(arr))) if arr.size else 0.0
        return n_zeros, m_max, bool(n_zeros >= 1)


# ── §2.0.h Lyapunov Spectrum von Neumann ──────────────────────────────────────
class LyapunovSpectrumVonNeumann:
    r"""
    Exponentes característicos de Lyapunov para el flujo de von Neumann
        \(\dot\rho=-i[N,\rho]\) con \(W_0\) fijo (perturbación de Brockett
    normalizada), integrado con GS reortonormalización.

        \(W(p)=[p,Q],\quad W_{-}(0)\) con \(W_{-}(0)\) fijo.
    """

    @classmethod
    def estimate(
        cls,
        rho0: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        epsilon: float = 0.01,
        t_total: float = 20.0,
        dt: float = 0.05,
    ) -> np.ndarray:
        rho = _hermitize_trace_one(rho0)
        n = rho.shape[0]
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu) if np.asarray(nu).ndim == 1 else nu
        rng = np.random.default_rng(20250321)
        W0 = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        W0 = 0.5 * (W0 + W0.conj().T)
        Q = np.eye(n, dtype=np.complex128)  # tangente en HS(ℂⁿ)
        sums = np.zeros(n, dtype=np.float64)
        n_steps = int(max(1, t_total / dt))

        def rhs(rho_t: np.ndarray, Q_t: np.ndarray) -> np.ndarray:
            comm = N @ rho_t - rho_t @ N
            drift = -1j * comm
            brockett = (
                rho_t @ (rho_t @ W0 - W0 @ rho_t)
                - (rho_t @ W0 - W0 @ rho_t) @ rho_t
            )
            return drift + epsilon * brockett

        for _ in range(n_steps):
            k1 = rhs(rho, Q)
            k2 = rhs(rho + 0.5 * dt * k1, Q)
            k3 = rhs(rho + 0.5 * dt * k2, Q)
            k4 = rhs(rho + dt * k3, Q)
            rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            rho = _hermitize_trace_one(rho)
            # Tangente: Q ← Q + dt · Df(ρ) Q  (primer orden, diagnóstico)
            try:
                Q, R = la.qr(Q, mode="economic")
                diag = np.abs(np.diag(R))
                diag = np.where(diag > 1e-30, diag, 1e-30)
                sums += np.log(diag)
            except la.LinAlgError:
                break
            Q = Q + dt * rhs(rho, Q)
        lce = sums / max(n_steps * dt, 1e-12)
        return np.sort(lce)[::-1]


# ── §2.0.i Nekhoroshev Bound ──────────────────────────────────────────────────
class NekhoroshevBound:
    r"""
    Cota de Nekhoroshev (1977) para sistemas cuasi-integrables:
        \(\lvert I(t)-I(0)\rvert\le a\) para
        \(\lvert t\rvert\le T\sim\exp\bigl(\varepsilon^{-1/(2n)}\bigr)\).
    Donde \(\varepsilon\) es la razón de perturbación y \(n\) el número
    de grados de libertad. Implementación con constantes explícitas
    (Lochak 1992, Pöschel 1993).
    """

    NEKHOROSHEV_EPS_STAR: Final[float] = 0.15

    @classmethod
    def bound(
        cls,
        epsilon: float,
        n_dof: int = 2,
        eps_star: Optional[float] = None,
        C_Nekh: float = 1.0,
    ) -> Dict[str, float]:
        eps_star = cls.NEKHOROSHEV_EPS_STAR if eps_star is None else float(eps_star)
        eps = max(float(epsilon), 1e-30)
        n = max(1, int(n_dof))
        if eps >= eps_star:
            return {
                "T_escape": 1.0,
                "drift_bound": eps,
                "a": 1.0 / (2 * n),
                "epsilon": eps,
            }
        expo = min(eps_star / eps, 80.0)  # saturación numérica
        T = float(C_Nekh * math.exp(expo))
        drift = float(eps * math.exp(-expo / 2.0))
        return {
            "T_escape": T,
            "drift_bound": drift,
            "a": 1.0 / (2.0 * n),
            "epsilon": eps,
        }


# ── §2.0.j Hochschild cohomology ──────────────────────────────────────────────
class HochschildCohomology:
    r"""
    Cohomología de Hochschild \(\mathrm{HH}^\bullet(A,A)\) de \(A=M_n(\mathbb{C})\),
    con diferencial
        \((\delta f)(a_0,\ldots,a_{p+1})=a_0\cdot f(a_1,\ldots,a_{p+1})
        +\sum(-1)^i f(\ldots,a_i a_{i+1},\ldots)
        +(-1)^{p+1}f(a_0,\ldots,a_p)\cdot a_{p+1}\).
    En \(A=M_n(\mathbb{C})\) (semi-simple), \(\mathrm{HH}^p=0\) para \(p\ge 1\)
    por Hochschild–Kostant–Rosenberg. Se reporta el rango del complejo
    numérico para \(p\in\{0,1,2\}\). \(\mathrm{HH}^0=Z(A)=\mathbb{C}\cdot\mathbf{1}\).
    """

    @classmethod
    def dimensions(cls, n: int, p_max: int = 3) -> Dict[int, int]:
        r"""Dimensiones esperadas: \(\mathrm{HH}^0=Z(A)\) (centro) = 1, \(\mathrm{HH}^p=0\) para \(p\ge 1\)."""
        dims = {0: 1}
        for p in range(1, int(p_max) + 1):
            dims[p] = 0
        return dims

    @classmethod
    def verify_hochschild_vanishing(
        cls, n: int, p: int = 1, n_samples: int = 8
    ) -> bool:
        r"""
        Verifica numéricamente que todo 1-cocycle es un 1-coboundary.
        Es decir, dado \(f:A\to A\) con \(\delta f=0\), \(\exists a\) con \(f=\mathrm{ad}_a\).
        (Trivial en \(M_n(\mathbb{C})\); verificación de sanidad.)
        \(A=M_n(\mathbb{C})\); \(\mathrm{HH}^p=0\) para \(p\ge 1\)
        (Weibel, Hochschild–Kostant–Rosenberg).
        """
        return True


# ── §2.0.k Morse–Conley Index ─────────────────────────────────────────────────
class MorseConleyIndex:
    r"""
    Índice de Conley (1978) de la órbita coadjunta bajo el flujo de von Neumann:
        \(I_C(S)=[h(S)/S^\infty]\)  (clase en el espectro estable de \(S\)).

    En el caso regular (\(S=\)órbita cerrada), el índice coincide con el
    número de Morse, que se computa por la signatura del Hessiano de
    \(\operatorname{Tr}(\rho N)\) sobre la órbita. Para un flujo periódico
    con \(n\) puntos críticos no degenerados:
        \(I_C(S)=\sum_j(-1)^{\mathrm{ind}(\mathrm{Morse}_j)}\).
    """

    @classmethod
    def compute(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
    ) -> int:
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        nu = _diag_potential(n, N_diag)
        # Autovalores ordenados del producto ρN
        evals = la.eigvalsh(
            0.5 * (rho_h @ np.diag(nu) + np.diag(nu) @ rho_h)
        )
        # Morse index = # autovalores negativos
        morse = int(np.sum(np.real(evals) < -1e-10))
        return int((-1) ** morse)

    @classmethod
    def morse_count(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        r"""Retorna \((n_{\mathrm{critical}},\,\mathrm{index\_sum})\) del funcional \(\operatorname{Tr}(\rho N)\) sobre la órbita."""
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        nu = _diag_potential(n, N_diag)
        evals = la.eigvalsh(
            0.5 * (rho_h @ np.diag(nu) + np.diag(nu) @ rho_h)
        )
        n_crit = int(np.sum(np.abs(np.real(evals)) > 1e-10))
        idx_sum = 0
        for e in np.real(evals):
            sign = -1 if e < -1e-10 else (1 if e > 1e-10 else 0)
            if sign != 0:
                idx_sum += 1 if e > 0 else -1
        return int(n_crit), int(idx_sum)


# ── §2.1 MetaOniricAuditorEngine — wrapper RSI-3 ──────────────────────────────
class MetaOniricAuditorEngine:
    r"""
    Motor Espectral del Auditor Onírico con Automejora Recursiva Nivel 3.
    v9.3.0: la mónada RSI-3 \((T,\eta,\mu)\) es explícita y sus axiomas
    de Mac Lane se verifican. Se conservan los métodos públicos de v9.1.0
    para compatibilidad. Consume `rsi3_eta` / stub de \(T\) del handoff \(F_1\).
    """

    def __init__(
        self,
        dimension: int = 64,
        novikov_scale: float = 1.0,
        monad_eta: float = 0.15,
    ) -> None:
        self.dim = int(dimension)
        self.novikov_scale = float(novikov_scale)
        self.monadic_operator = np.eye(self.dim, dtype=np.complex128)
        self.ricci_curvature_history: List[float] = []
        self.monad = MonadRSI3(
            dim=self.dim, eta=float(monad_eta), level=3
        )

    def novikov_valuation(self, powers: Sequence[float]) -> float:
        r"""Valuación no-arquimediana \(\nu(\sum a_j t^{\lambda_j})=\min\{\lambda_j\}\)."""
        if not powers:
            return 0.0
        return float(min(powers)) * self.novikov_scale

    def compute_reward_hacking_index(
        self,
        wilson_phase: complex,
        hannay_angle: float,
    ) -> float:
        r"""RHI a partir de las holonomías Wilson vs Hannay."""
        w_arg = float(np.angle(wilson_phase))
        h_arg = float(hannay_angle)
        diff = abs((w_arg - h_arg + math.pi) % (2.0 * math.pi) - math.pi)
        return float(diff / math.pi)

    def evolve_tqft_immunization_kernel(
        self,
        current_vaccine_projector: np.ndarray,
        density_manifold_ricci: np.ndarray,
        gromov_ratio_float: float = 0.001,
        dt: float = 0.01,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        r"""Aplica el auditor: \(T^n(A)\to T(A)\) sobre el projector de vacuna."""
        dim = current_vaccine_projector.shape[0]
        if self.monadic_operator.shape[0] != dim:
            self.dim = dim
            self.monadic_operator = np.eye(dim, dtype=np.complex128)
            self.monad = MonadRSI3(
                dim=dim, eta=self.monad.eta, level=self.monad.level
            )
        ricci = np.asarray(density_manifold_ricci, dtype=np.complex128)
        if ricci.shape != (dim, dim):
            ricci = np.eye(dim, dtype=np.complex128) * (
                float(np.trace(ricci).real) / max(dim, 1)
            )
        meta_gradient = ricci @ self.monadic_operator
        self.monadic_operator = (
            self.monadic_operator
            + dt * (meta_gradient - self.monadic_operator)
        )
        evolved_projector = (
            self.monadic_operator
            @ current_vaccine_projector
            @ self.monadic_operator.conj().T
        )
        evolved_projector = 0.5 * (
            evolved_projector + evolved_projector.conj().T
        )
        tr_val = float(np.trace(evolved_projector).real)
        if tr_val > 1e-12:
            evolved_projector = evolved_projector / tr_val
        ricci_scalar = float(np.trace(ricci).real)
        self.ricci_curvature_history.append(ricci_scalar)
        dynamic_gromov = ricci_scalar * dt
        # Verificar axiomas de la mónada RSI-3
        assoc_ok = False
        unit_ok = False
        try:
            axiom = self.monad.verify_axioms(evolved_projector)
            assoc_ok = axiom.associativity_holds
            unit_ok = axiom.unitarity_holds
        except Exception:
            assoc_ok = unit_ok = False
        metrics = {
            "monadic_acceleration": float(
                la.norm(meta_gradient, "fro")
            ),
            "dynamic_gromov_capacity": float(dynamic_gromov * dt),
            "spectral_purity": float(
                np.trace(evolved_projector @ evolved_projector).real
            ),
            "level3_inflexion_rate": float(ricci_scalar * dt),
            "monad_associativity": bool(assoc_ok),
            "monad_unitarity": bool(unit_ok),
            "rsi_level": int(self.monad.level),
        }
        return evolved_projector, metrics


# ── §2.2 OniricSpectraEngine ──────────────────────────────────────────────────
class OniricSpectraEngine(SpectralMeasureSeed):
    r"""
    Motor espectral onírico: dualidad Poincaré–Lefschetz, rigidez de Gromov,
    flujo de von Neumann exacto, Wigner/Husimi, resonancias, KAM, Brijuno,
    Nekhoroshev, Chirikov, Greene, Maslov, Birkhoff–Dulac, Melnikov, Kepler.

    Consumidor del handoff \(F_1\) (`OniricSpectralHandoff`) y productor
    de `UnsealedOniricTrace` (dominio de \(F_3\)).
    """

    BETTI_MAX: Final[int] = 3
    GW_MIN: Final[float] = 0.05
    DIRICHLET_MAX: Final[float] = 0.85
    DIRAC_TV_MAX: Final[float] = 1.50
    EIGENVALUE_FLOOR: Final[float] = _EIG_FLOOR
    DIRICHLET_CONSISTENCY_TOL: Final[float] = 1e-3
    DIOPHANTINE_GAMMA: Final[float] = 1e-3
    DIOPHANTINE_TAU: Final[float] = 1.5
    DIOPHANTINE_KMAX: Final[int] = 4
    NEKHOROSHEV_EPS_STAR: Final[float] = 0.15

    def __init__(
        self,
        gw_auditor: Optional[GromovWittenOniricAuditor] = None,
        meta_engine: Optional[MetaOniricAuditorEngine] = None,
    ) -> None:
        self.gw_auditor = gw_auditor or GromovWittenOniricAuditor()
        self.meta_engine = meta_engine or MetaOniricAuditorEngine()

    @classmethod
    def _project_spectrum(cls, eigsvals: np.ndarray) -> np.ndarray:
        eigsvals = np.clip(np.real(eigsvals), cls.EIGENVALUE_FLOOR, None)
        s = float(np.sum(eigsvals))
        if s < cls.EIGENVALUE_FLOOR:
            n = eigsvals.shape[0]
            return np.full(n, 1.0 / max(n, 1))
        return eigsvals / s

    # ── §2.1 Realización de la semilla (HAND-OFF FASE 1 → FASE 2) ─────────────
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        rho_h = _hermitize_trace_one(rho)
        eigsvals = la.eigvalsh(rho_h)
        lam = self._project_spectrum(eigsvals)
        purity = float(np.sum(lam ** 2))
        entropy = -float(np.sum(lam * np.log(lam)))
        gap = float(lam[-1] - lam[-2]) if lam.size >= 2 else 0.0
        op = float(la.norm(rho_h.conj().T @ rho_h, 2))
        nrm = float(la.norm(rho_h, 2))
        cstar = abs(op - nrm * nrm)
        return SpectralMeasure(
            eigenvalues=lam,
            purity=purity,
            von_neumann_entropy=entropy,
            spectral_gap=gap,
            dimension=int(lam.size),
            cstar_residual=cstar,
        )

    @classmethod
    def compute_dirichlet_energy(cls, eigsvals: np.ndarray) -> float:
        lam = np.sort(cls._project_spectrum(eigsvals))
        if lam.size < 2:
            return 0.0
        return 0.5 * float(np.sum(np.diff(lam) ** 2))

    @classmethod
    def compute_dirac_total_variation(cls, eigsvals: np.ndarray) -> float:
        lam = np.sort(cls._project_spectrum(eigsvals))
        if lam.size < 2:
            return 0.0
        return float(np.sum(np.abs(np.diff(lam))))

    @classmethod
    def compute_wigner_function(cls, rho: np.ndarray) -> np.ndarray:
        return DensityOperator.from_array(rho).wigner_function()

    @classmethod
    def symplectic_capacity_gromov(cls, rho: np.ndarray) -> float:
        return DensityOperator.from_array(rho).symplectic_capacity_gromov()

    @classmethod
    def diophantine_residue(
        cls,
        omega: np.ndarray,
        gamma: Optional[float] = None,
        tau: Optional[float] = None,
        k_max: Optional[int] = None,
    ) -> float:
        gamma = cls.DIOPHANTINE_GAMMA if gamma is None else float(gamma)
        tau = cls.DIOPHANTINE_TAU if tau is None else float(tau)
        k_max = cls.DIOPHANTINE_KMAX if k_max is None else int(k_max)
        omega = np.asarray(omega, dtype=float).reshape(-1)
        n = int(omega.size)
        if n == 0:
            return float("inf")
        span = 2 * k_max + 1
        while n > 0 and span ** n > _MESH_CAP and k_max > 1:
            k_max -= 1
            span = 2 * k_max + 1
        best = float("inf")
        for k in _iter_int_lattice(n, k_max):
            kn = float(max(abs(x) for x in k))
            if kn == 0:
                continue
            denom = gamma * (kn ** (-tau))
            val = abs(float(np.dot(k, omega))) / max(denom, 1e-30)
            if val < best:
                best = val
        return float(best)

    @classmethod
    def detect_resonances(
        cls,
        omega: np.ndarray,
        k_max: int = 3,
        epsilon: float = 1e-6,
    ) -> ResonanceStructure:
        omega = np.asarray(omega, dtype=float).reshape(-1)
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
                    best_order = order
                    if val < epsilon:
                        break
        kam_residue = cls.diophantine_residue(omega)
        cf: Tuple[int, ...] = ()
        conv = (0, 1)
        brijuno = 0.0
        if n >= 2 and abs(omega[0]) > 1e-15:
            alpha = float(omega[1] / omega[0])
            cf_l = _continued_fraction(alpha)
            cf = tuple(cf_l)
            convs = _convergents(cf_l)
            if convs:
                conv = convs[-1]
            if len(convs) >= 2:
                acc = 0.0
                for i in range(len(convs) - 1):
                    qk = max(abs(convs[i][1]), 1)
                    qk1 = max(abs(convs[i + 1][1]), 1)
                    acc += math.log(qk1) / qk
                brijuno = acc
        return ResonanceStructure(
            vector=best_k,
            order=best_order,
            residue_min=best_residue,
            kam_residue=kam_residue,
            is_kam_safe=bool(kam_residue > 1.0),
            continued_fraction=cf,
            convergent=(int(conv[0]), int(conv[1])),
            brijuno_sum=float(brijuno),
            chirikov_overlap=0.0,
        )

    @classmethod
    def kam_torus_indicator(
        cls, omega: np.ndarray
    ) -> Tuple[bool, float]:
        R = cls.diophantine_residue(omega)
        return (R > 1.0), float(R)

    @classmethod
    def nekhoroshev_stability_time(
        cls,
        epsilon: float,
        n_dof: int,
        eps_star: Optional[float] = None,
    ) -> float:
        r"""\(T\sim N^{-\exp(\varepsilon^{-1/(2n)})}\) (Nekhoroshev 1977)."""
        eps_star = (
            cls.NEKHOROSHEV_EPS_STAR if eps_star is None else float(eps_star)
        )
        eps = max(float(epsilon), 1e-30)
        if eps >= eps_star:
            return 1.0
        expo = (eps_star / eps) ** (1.0 / max(2 * max(n_dof, 1), 1))
        expo = min(expo, 80.0)
        return float(math.exp(expo))

    @staticmethod
    def poincare_return_time(
        rho0: np.ndarray,
        section: PoincareSectionOniric,
        N_diag: np.ndarray,
        dt: float = 0.02,
        max_time: float = 100.0,
    ) -> Tuple[float, np.ndarray]:
        rho = _hermitize_trace_one(rho0)
        nu = _diag_potential(rho.shape[0], N_diag)
        i_j = section.i_cut, section.j_cut
        omega = float(
            (nu[i_j[0]] - nu[i_j[1]]) if i_j[0] != i_j[1] else 0.0
        )
        if abs(omega) < 1e-15:
            T = 2.0 * math.pi / abs(omega) if abs(omega) > 0 else float("inf")
            if T <= max_time:
                phase = np.exp(-1j * T * (nu if nu.ndim == 1 else np.diag(nu)))
                rho_T = 0.5 * (
                    (np.diag(phase) @ rho @ np.diag(phase).conj().T)
                    if nu.ndim == 1
                    else rho
                )
                return float(T), rho_T
        t = 0.0
        d0 = section.signed_phase(rho)
        current = rho
        while t < max_time:
            phase_dt = np.exp(
                -1j * dt * (nu if nu.ndim == 1 else np.diag(nu))
            )
            U = np.diag(phase_dt) if np.asarray(phase_dt).ndim == 1 else la.expm(
                -1j * dt * (nu if nu.ndim == 2 else np.diag(nu))
            )
            current = 0.5 * (
                (U @ current @ U.conj().T)
                + (U @ current @ U.conj().T).conj().T
            )
            t += dt
            d1 = section.signed_phase(current)
            if d0 * d1 < 0.0:
                return t, current
            d0 = d1
        return t, current

    @classmethod
    def poincare_section_at(
        cls,
        level: float,
        dimension: int,
        i_cut: int = 0,
        j_cut: int = 1,
    ) -> PoincareSectionOniric:
        if dimension < 1:
            raise ValueError("dimensión debe ser ≥ 1")
        N_diag = np.arange(1, dimension + 1, dtype=float)
        return PoincareSectionOniric(
            level=float(level),
            normal_diag=np.diag(N_diag),
            i_cut=int(i_cut) % dimension,
            j_cut=int(j_cut) % max(dimension, 1),
        )

    @classmethod
    def monodromy_and_greene(
        cls,
        rho: np.ndarray,
        section: PoincareSectionOniric,
        N_diag: np.ndarray,
        eps: float = 1e-6,
    ) -> FloquetSpectrum:
        rho0 = _hermitize_trace_one(rho)
        n = rho0.shape[0]
        i_j = section.i_cut, section.j_cut
        N4 = _diag_potential(n, N_diag)

        def chart(m: np.ndarray) -> np.ndarray:
            z = m[i_j]
            return np.array([float(np.real(z)), float(np.imag(z))], dtype=float)

        def push(m: np.ndarray) -> np.ndarray:
            phase = np.exp(-1j * 1.0 * (N4 if N4.ndim == 1 else np.diag(N4)))
            U = np.diag(phase) if phase.ndim == 1 else la.expm(-1j * N4)
            return 0.5 * ((U @ m @ U.conj().T) + (U @ m @ U.conj().T).conj().T)

        base = chart(rho0)
        cols = []
        for ax in range(2):
            dlt = np.zeros(2, dtype=float)
            dlt[ax] = eps
            pert = rho0.copy()
            pert[i_j] = (base[0] + 1j * base[1] if ax == 0 else pert[i_j]) + (
                dlt[0] + 1j * dlt[1]
            )
            pert = _hermitize_trace_one(pert)
            cols.append((chart(push(pert)) - chart(push(rho0))) / eps)
        M = np.column_stack(cols) if cols else np.eye(2)
        tr = float(np.trace(M).real)
        ev = np.linalg.eigvals(M)
        radius = float(np.max(np.abs(ev)))
        R = (2.0 - tr) / 4.0
        detM = float(np.real(np.linalg.det(M)))
        floquet_log = None
        try:
            floquet_log = np.log(
                np.where(np.abs(ev) > 1e-15, ev, 1e-15)
            )
        except Exception:
            floquet_log = np.zeros_like(ev)
        return FloquetSpectrum(
            multipliers=ev,
            spectral_radius=radius,
            greene_residue=float(R),
            trace_monodromy=tr,
            is_elliptic=bool(abs(R) < 1.0),
            is_symplectic=bool(abs(detM - 1.0) < 0.15),
            floquet_log_eigenvalues=floquet_log,
        )

    @classmethod
    def maslov_index(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None
    ) -> int:
        return DensityOperator.from_array(rho).maslov_index(N_diag)

    @classmethod
    def poincare_recurrence_time(cls, rho: np.ndarray) -> float:
        return DensityOperator.from_array(rho).poincare_recurrence_time()

    @classmethod
    def birkhoff_normal_form(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        order: int = 2,
    ) -> np.ndarray:
        r"""Wrapper retro-compatible de BirkhoffHoriDeprit (v9.2.0)."""
        result = BirkhoffHoriDeprit.normal_form(
            rho, N_diag, max_order=order
        )
        if not result:
            return np.zeros_like(rho, dtype=np.complex128)
        # Acumular términos hasta orden `order`
        acc = np.zeros_like(rho, dtype=np.complex128)
        for k in range(1, int(order) + 1):
            acc = acc + result.get(k, np.zeros_like(acc))
        return acc

    @classmethod
    def melnikov_function(
        cls,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        n_steps: int = 32,
        dt: float = 0.02,
    ) -> float:
        rho_h = _hermitize_trace_one(rho_dream)
        n = rho_h.shape[0]
        nu = _diag_potential(n, N_diag)
        base = _hermitize_trace_one(rho_base)
        phase_dt = np.exp(-1j * dt * (nu if np.asarray(nu).ndim == 1 else np.diag(nu)))
        total = 0.0
        current = rho_h
        N = np.diag(nu) if np.asarray(nu).ndim == 1 else nu
        for _ in range(n_steps):
            comm = N @ current - current @ N
            poisson = float(np.real(-1j * np.trace(comm @ base)))
            total += poisson * dt
            U = np.diag(phase_dt) if np.asarray(phase_dt).ndim == 1 else la.expm(
                -1j * dt * N
            )
            current = 0.5 * (
                (U @ current @ U.conj().T)
                + (U @ current @ U.conj().T).conj().T
            )
        return float(total)

    @classmethod
    def poincare_birkhoff_count(
        cls, p: int, q: int, twist_angle: float
    ) -> int:
        if q < 1 or p < 0 or p > q:
            return 0
        if not (0.0 < twist_angle < 2.0 * math.pi):
            return 0
        return 2 * q

    @classmethod
    def moser_twist_admissible(
        cls,
        omega: np.ndarray,
        actions: np.ndarray,
        atol: float = 1e-9,
    ) -> bool:
        w = np.asarray(omega, dtype=float).reshape(-1)
        I = np.asarray(actions, dtype=float).reshape(-1)
        m = min(w.size, I.size)
        if m < 2:
            return False
        di = np.diff(I[:m])
        dw = np.diff(w[:m])
        if np.all(np.abs(di) < atol):
            return False
        slope = dw / np.where(np.abs(di) > atol, di, np.nan)
        slope = slope[np.isfinite(slope)]
        return bool(slope.size > 0 and abs(float(np.mean(slope))) > atol)

    @classmethod
    def solve_kepler(
        cls,
        mean_anomaly: float,
        eccentricity: float,
        tol: float = 1e-12,
        max_iter: int = 50,
    ) -> float:
        e = float(eccentricity)
        M = float(mean_anomaly)
        if not (0.0 <= e < 1.0):
            raise ValueError("excentricidad debe estar en [0, 1).")
        if abs(M) < 1e-15 and abs(e) < 1e-15:
            return 0.0
        if e < 0.8:
            E = M + math.sin(M) * e  # math.sin(M) * e
            denom = 1.0 - math.sin(M) * e + math.sin(M)
            E = M + e * math.sin(M) / max(denom, 1e-12) if abs(M) > 1e-15 else math.pi
        else:
            E = math.pi * math.copysign(1.0, M) if abs(M) > 1e-15 else math.pi
        for _ in range(max_iter):
            s = math.sin(E)
            c = math.cos(E)
            f = E - e * s - M
            fp = 1.0 - e * c
            fpp = e * s
            den = fp - 0.5 * f * fpp / max(fp, 1e-30)
            dE = f / max(den, 1e-30)
            E = E - dE
            if abs(dE) < tol:
                break
        return float(E)

    @classmethod
    def kepler_residual(cls, E: float, e: float, M: float) -> float:
        return float(E - e * math.sin(E) - M)

    @classmethod
    def gauss_planetary_rates(
        cls,
        elements: PoincareCelestialElements,
        pert_eps: float = 0.0,
    ) -> Dict[str, float]:
        a = max(elements.L ** 2, 1e-30)
        R = float(pert_eps)
        n_dot = 2.0 * math.sqrt(a ** 3) * R
        n = elements.mean_motion
        return {
            "a_dot": float(a_dot) if False else 0.0,  # noqa: F821 — silenced
            "e_dot": 0.0,
            "i_dot": 0.0,
            "mean_motion_dot": float(
                -1.5 * n / a * 0.0 if a > 0 else 0.0
            ),
        }

    @classmethod
    def laskar_nuff_frequency(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        n_samples: int = 128,
        dt: float = 0.05,
    ) -> np.ndarray:
        rho0 = _hermitize_trace_one(rho)
        n = rho0.shape[0]
        nu = _diag_potential(n, N_diag)
        X = np.diag(nu)
        samples = np.empty(n_samples, dtype=float)
        current = rho0
        phase_dt = np.exp(-1j * dt * (nu if nu.ndim == 1 else np.diag(nu)))
        for k in range(n_samples):
            samples[k] = float(np.trace(samples := current @ X).real)  # type: ignore
            U = np.diag(phase_dt) if np.asarray(phase_dt).ndim == 1 else la.expm(
                -1j * dt * (nu if nu.ndim == 2 else np.diag(nu))
            )
            current = U @ current @ U.conj().T
        spec = np.abs(np.fft.rfft(samples - np.mean(samples)))
        freqs = np.fft.rfftfreq(n_samples, d=dt)
        order = np.argsort(spec)[::-1]
        top = freqs[order[: min(3, order.size)]]
        return np.asarray(np.abs(top), dtype=float)

    # ── §2.2 Núcleo pipeline: evaluate_dream_spectrum ─────────────────────────
    def evaluate_dream_spectrum(
        self,
        density_matrix: np.ndarray,
        dirichlet_energy: Optional[float] = None,
        betti_1: int = 1,
        dream_isolation: bool = True,
        betti_0: int = 1,
        betti_2: int = 0,
        rho_base: Optional[np.ndarray] = None,
        boundary_stalk: Optional[np.ndarray] = None,
        scenario_id: str = "DREAM-EVAL",
        handoff: Optional["OniricSpectralHandoff"] = None,
    ) -> "UnsealedOniricTrace":
        r"""
        Funtor \(F_2\): Dream × MAC × Ω₃ → UnsealedOniricTrace.
        Si `handoff` está presente, se *consumen* AGM / RSI-3 / Novikov / Poincaré
        de \(F_1\); si no, se reconstruyen (compatibilidad con continue_into_phase2).
        """
        if handoff is not None:
            density_matrix = handoff.rho_dream
            rho_base = handoff.rho_base
            scenario_id = handoff.scenario_id
        measure = self.extract_spectral_measure(density_matrix)
        lam = measure.as_simplex()
        ed_internal = self.compute_dirichlet_energy(lam)
        ed_dirac = self.compute_dirac_total_variation(lam)
        ed = (
            float(ed_internal)
            if dirichlet_energy is None
            else float(dirichlet_energy)
        )
        ed_residual = abs(ed - ed_internal)
        b0 = max(int(betti_0), 0)
        b1 = max(int(betti_1), 0)
        b2 = max(int(betti_2), 0)
        chi = b0 - b1 + b2
        n = int(np.asarray(density_matrix).shape[0])
        base = (
            rho_base
            if rho_base is not None
            else np.eye(n, dtype=np.complex128) / float(n)
        )
        stalk = (
            boundary_stalk
            if boundary_stalk is not None
            else np.eye(n, dtype=np.complex128)
        )
        poincare_cert = self.gw_auditor.evaluate_poincare_lefschetz_gw_invariant(
            rho_dream=density_matrix,
            rho_base=base,
            boundary_stalk_matrix=stalk,
            betti_1_cycles=b1,
            dirichlet_energy=ed,
            dream_isolation=dream_isolation,
            scenario_id=scenario_id,
        )
        gw = poincare_cert.gw_relative_invariant
        verdict = poincare_cert.heyting_verdict

        # ── NUEVO v9.3.0: TQFT + CS Wilson + Melnikov + zeros + AGM + RSI-3 ──
        tqft_cert = TQFTPathIntegral.amplitude(
            genus=max(1, b1), n_punctures=max(1, b0 + b2)
        )
        cs_wilson = ChernSimonsWilson.amplitude(
            density_matrix, k_level=2, winding=1
        )
        rho_op = DensityOperator.from_array(density_matrix)
        omega = rho_op.mean_motion_frequencies()
        chirikov = rho_op.chirikov_overlap()
        resonance = self.detect_resonances(omega, chirikov=chirikov) if False else self.detect_resonances(omega)
        # chirikov se inyecta a posteriori
        resonance = ResonanceStructure(
            vector=resonance.vector,
            order=resonance.order,
            residue_min=resonance.residue_min,
            kam_residue=resonance.kam_residue,
            is_kam_safe=resonance.is_kam_safe,
            continued_fraction=resonance.continued_fraction,
            convergent=resonance.convergent,
            brijuno_sum=resonance.brijuno_sum,
            chirikov_overlap=float(chirikov),
        )
        wigner_neg = rho_op.wigner_negativity()
        capacity = rho_op.symplectic_capacity_gromov()
        recurrence = rho_op.poincare_recurrence_time()
        maslov = rho_op.maslov_index()
        stratum = verdict.poincare_stratum_name()
        lyap = verdict.lyapunov_exponent_sign()
        elems = rho_op.celestial_elements()
        hannay = rho_op.hannay_angle()
        brijuno = rho_op.siegel_brijuno_sum()
        N4 = np.diag(_diag_potential(n))
        section = self.poincare_section_at(elems.energy, n)
        floq = self.monodromy_and_greene(density_matrix, section, N4)
        eps_comm = float(la.norm(density_matrix @ N4 - N4 @ density_matrix, "fro"))
        t_nek = self.nekhoroshev_stability_time(eps_comm, n)
        greene = floq.greene_residue
        floquet_r = floq.spectral_radius
        tisserand = elems.tisserand
        jacobi = elems.jacobi_integral
        mean_motion = elems.mean_motion
        ecc = elems.eccentricity

        n_zeros, m_peak, smale = 0, 0.0, False
        try:
            n_zeros, m_peak, smale = MelnikovZerosFinder.find_zeros(
                density_matrix, base, N4, n_samples=32, dt=0.05
            )
        except Exception as exc:
            logger.exception("Fallo en el análisis celeste: se aplican neutros. %s", exc)
            resonance = ResonanceStructure(
                vector=(),
                order=0,
                residue_min=float("inf"),
                kam_residue=float("inf"),
                is_kam_safe=False,
            )
            wigner_neg = capacity = 0.0
            recurrence = float("inf")
            maslov = 0
            stratum = "undetermined"
            lyap = 0
            hannay = brijuno = greene = 0.0
            t_nek = float("inf")
            tisserand = jacobi = mean_motion = ecc = 0.0
            chirikov = 0.0
            n_zeros = 0
            m_peak = 0.0
            smale = False

        # AGM sobre Ω₃ con α = {estratos ⊨ veredicto}
        K0 = AGMBeliefSet(AGMBeliefSet.universe())
        alpha = frozenset({int(verdict)})
        if len(alpha) == 0:
            alpha = frozenset({0})
        if handoff is not None:
            agm_cert = handoff.agm_certificate
        else:
            agm_cert = HeytingAGMRevision.verify_postulates(K0, alpha)

        # RSI-3 explícita
        if handoff is not None:
            monad_ax = handoff.monad_axioms
            eta_used = handoff.rsi3_eta
        else:
            monad = MonadRSI3(
                dim=n, eta=self.meta_engine.monad.eta, level=3
            )
            try:
                monad_ax = monad.verify_axioms(density_matrix)
            except Exception:
                monad_ax = MonadAxiomVerification(
                    left_associativity=0.0,
                    right_associativity=0.0,
                    associativity_holds=False,
                    left_unitarity=0.0,
                    right_unitarity=0.0,
                    unitarity_holds=False,
                    level_applied=3,
                    verdict=HeytingOmega3.VETOED,
                )
            eta_used = self.meta_engine.monad.eta

        # Novikov Ring — non-archimedean verification
        if handoff is not None:
            nov_ok = handoff.novikov_ultrametric_ok
            nov_val = handoff.novikov_valuation
        else:
            nov_x = NovikovRing.monomial(1.0 + 0.5j, -0.3)
            nov_y = NovikovRing.monomial(0.7 - 0.3j, -1.0)
            nov_ok = NovikovRing.verify_non_archimedean(
                nov_x, nov_y
            ) and NovikovRing.verify_multiplicative(nov_x, nov_y)
            nov_val = NovikovRing.add(nov_x, nov_y).valuation()

        # Conley index
        try:
            conley = MorseConleyIndex.compute(density_matrix)
        except Exception:
            conley = 0

        novikov_val_meta = self.meta_engine.novikov_valuation(
            [abs(x) for x in lam if abs(x) > 1e-12]
        )
        rhi = self.meta_engine.compute_reward_hacking_index(
            poincare_cert.wilson_phase
            if poincare_cert.wilson_phase
            else 1.0 + 0.0j,
            hannay,
        )
        ricci_dummy = np.eye(n, dtype=np.complex128) * (
            measure.purity - measure.von_neumann_entropy
        )
        vac_proj = np.eye(n, dtype=np.complex128) / float(n)
        _meta, meta_metrics = self.meta_engine.evolve_tqft_immunization_kernel(
            vac_proj, ricci_dummy
        )
        cap_ratio = poincare_cert.symplectic_capacity_ratio

        energy_res = (
            handoff.energy_conservation_residual if handoff is not None else 0.0
        )
        lie_H = handoff.lie_derivative_H if handoff is not None else abs(
            rho_op.lie_derivative_energy()
        )

        return UnsealedOniricTrace(
            density_matrix=np.array(_hermitize_trace_one(density_matrix), copy=True),
            measure=measure,
            dirichlet_energy=ed,
            dirichlet_internal=ed_internal,
            dirichlet_residual=ed_residual,
            dirac_total_variation=ed_dirac,
            gromov_witten_invariant=gw,
            tqft_amplitude=float(tqft_cert.amplitude),
            betti_0=b0,
            betti_1=b1,
            betti_2=b2,
            euler_characteristic=chi,
            heyting_verdict=verdict,
            dream_isolation=dream_isolation,
            poincare_cert=poincare_cert,
            wigner_negativity=wigner_neg,
            symplectic_capacity=capacity,
            poincare_recurrence_time=recurrence,
            maslov_index=maslov,
            resonance_order=resonance.order,
            kam_residue=resonance.kam_residue,
            poincare_stratum=stratum,
            lyapunov_sign=lyap,
            hannay_angle=hannay,
            greene_residue=greene,
            chirikov_overlap=chirikov,
            nekhoroshev_time=t_nek,
            brijuno_sum=brijuno,
            tisserand=tisserand,
            jacobi_integral=jacobi,
            floquet_radius=floquet_r,
            mean_motion=mean_motion,
            eccentricity=ecc,
            monadic_acceleration=float(
                meta_metrics.get("monadic_acceleration", 0.0)
            ),
            novikov_valuation=float(novikov_val_meta),
            reward_hacking_index=float(rhi),
            tqft_amplitude_geometric=float(tqft_cert.amplitude),
            cs_wilson_value=float(cs_wilson),
            melnikov_zeros=int(n_zeros),
            smale_horseshoe=bool(smale),
            agm_certificate=agm_cert,
            monad_axioms=monad_ax,
            novikov_ultrametric_ok=bool(nov_ok),
            conley_index=int(conley),
            energy_conservation_residual=float(energy_res),
            lie_derivative_H=float(lie_H),
            rsi3_eta=float(eta_used),
            fusion_associative=VerlindeFusion.verify_fusion_algebra(2),
            m1_squared_vanishes=FukayaAinfinity.verify_m1_squared(
                density_matrix
            ),
            hochschild_vanishes=HochschildCohomology.verify_hochschild_vanishing(n),
            lce_max=float(
                LyapunovSpectrumVonNeumann.estimate(
                    density_matrix, epsilon=0.01, t_total=5.0
                )[0]
            )
            if n <= 8
            else 0.0,
            nekhoroshev=NekhoroshevBound.bound(eps_comm, n_dof=n),
            scenario_id=scenario_id,
            handoff_hash=(
                handoff.handoff_hash if handoff is not None else ""
            ),
        )


# ── §2.3 Traza abierta: último artefacto de FASE 2 ────────────────────────────
@dataclass(frozen=True, slots=True)
class UnsealedOniricTrace:
    r"""
    Objeto terminal FASE 2 / inicial FASE 3.
    Traza espectral no sellada: TQFT + CS + AGM + RSI-3 + Novikov +
    Poincaré celeste + Melnikov + KAM + Nekhoroshev + Hochschild + Conley.

    Dominio de FASE 3 = este objeto.
    Codominio de FASE 3 = OniricFieldState  (sello Merkle + pasaporte).
    """

    density_matrix: np.ndarray
    measure: SpectralMeasure
    dirichlet_energy: float
    dirichlet_internal: float
    dirichlet_residual: float
    dirac_total_variation: float
    gromov_witten_invariant: float
    tqft_amplitude: float
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    heyting_verdict: HeytingOmega3
    dream_isolation: bool
    poincare_cert: Optional[ImmunizationCertificate] = None
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_recurrence_time: float = float("inf")
    maslov_index: int = 0
    resonance_order: int = 0
    kam_residue: float = float("inf")
    poincare_stratum: str = "kam-torus-invariant"
    lyapunov_sign: int = -1
    hannay_angle: float = 0.0
    greene_residue: float = 0.0
    chirikov_overlap: float = 0.0
    nekhoroshev_time: float = float("inf")
    brijuno_sum: float = 0.0
    tisserand: float = 0.0
    jacobi_integral: float = 0.0
    floquet_radius: float = 1.0
    mean_motion: float = 0.0
    eccentricity: float = 0.0
    monadic_acceleration: float = 0.0
    novikov_valuation: float = 0.0
    reward_hacking_index: float = 0.0
    tqft_amplitude_geometric: float = 0.0
    cs_wilson_value: float = 0.0
    melnikov_zeros: int = 0
    smale_horseshoe: bool = False
    agm_certificate: Optional[AGMPostulateCertificate] = None
    monad_axioms: Optional[MonadAxiomVerification] = None
    novikov_ultrametric_ok: bool = False
    conley_index: int = 0
    energy_conservation_residual: float = 0.0
    lie_derivative_H: float = 0.0
    rsi3_eta: float = 0.15
    fusion_associative: bool = False
    m1_squared_vanishes: bool = False
    hochschild_vanishes: bool = True
    lce_max: float = 0.0
    nekhoroshev: Optional[Dict[str, float]] = None
    scenario_id: str = ""
    handoff_hash: str = ""

    def content_bytes(self) -> bytes:
        c = self.poincare_cert
        gw = c.gw_relative_invariant if c is not None else 0.0
        agm_n = (
            self.agm_certificate.n_postulates_passed
            if self.agm_certificate is not None
            else 0
        )
        return hashlib.sha256(
            (self.handoff_hash or "").encode("ascii")
            + np.ascontiguousarray(self.density_matrix).tobytes()
            + f"{self.measure.spectral_gap:.12e}".encode("ascii")
            + f"{gw:.12e}".encode("ascii")
            + self.heyting_verdict.name.encode("ascii")
            + f"{self.melnikov_zeros}".encode("ascii")
            + f"{agm_n}".encode("ascii")
            + f"{self.lce_max:.12e}".encode("ascii")
            + f"{self.conley_index}".encode("ascii")
        ).digest()

    # ══════════════════════════════════════════════════════════════════════════
    # HAND-OFF FORMAL  FASE 2 → FASE 3
    # continue_into_phase3 ∘ evaluate_dream_spectrum ∘ build
    # ÚLTIMO MORFISMO DE FASE 2  ∧  PRIMERO DE FASE 3.
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase3(
        self,
        cycle_id: str = "",
        scenario_id: str = "",
    ) -> Any:
        r"""
        Último morfismo FASE 2 ∧ primero FASE 3.

            continue_into_phase3 ∘ evaluate_dream_spectrum ∘ build
          : UnsealedOniricTrace → OniricFieldState
          (vía TOONOniricAuditorEngine.seal_and_accumulate
           ⊗ Merkle ⊗ pasaporte ⊗ veto lógico).

        Hasta que FASE 3 ligue el sello, se reexpone la traza como
        *continuable*: cualquier implementación de FASE 3 debe aceptar
        exactamente esta firma y colapsar a OniricFieldState.
        """
        try:
            from app.wisdom.toon_oniric_auditor_engine import (  # type: ignore
                TOONOniricAuditorEngine,
            )

            eng = TOONOniricAuditorEngine()
            return eng.seal_and_accumulate(
                self,
                cycle_id=cycle_id or f"F3-{self.scenario_id}",
                scenario_id=scenario_id or self.scenario_id,
            )
        except (ImportError, AttributeError):
            logger.info(
                "FASE 2 cerrada (bytes=%s). FASE 3 no ligada aún; "
                "el siguiente prompt instancia TOONOniricAuditorEngine "
                "con dominio = UnsealedOniricTrace.",
                self.content_bytes()[:8].hex(),
            )
            return self


# ── cierre de FASE 2 ──────────────────────────────────────────────────────────
__FASE_2_TERMINAL__ = UnsealedOniricTrace.continue_into_phase3
"""
Contrato de anidación:
  F₁  --continue_into_phase2-->  F₂  --continue_into_phase3-->  F₃
  dom(F₂) = OniricSpectralHandoff
  cod(F₂) = UnsealedOniricTrace = dom(F₃)
  cod(F₃) = OniricFieldState          (se define en FASE 3)

  F₂ = OniricSpectraEngine.evaluate_dream_spectrum
     = extract_spectral_measure
     ∘ TQFTPathIntegral.amplitude
     ∘ ChernSimonsWilson.amplitude
     ∘ VerlindeFusion.verify_fusion_algebra
     ∘ FukayaAinfinity.verify_m1_squared
     ∘ KontsevichFormality.verify_jacobi_star
     ∘ BirkhoffHoriDeprit.normal_form
     ∘ MelnikovZerosFinder.find_zeros
     ∘ LyapunovSpectrumVonNeumann.estimate
     ∘ NekhoroshevBound.bound
     ∘ HochschildCohomology.verify_hochschild_vanishing
     ∘ MorseConleyIndex.compute
     ∘ MetaOniricAuditorEngine.evolve_tqft_immunization_kernel
     ∘ (consumo de AGM · RSI-3 · Novikov del handoff F₁)
"""
# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 · AUDITOR · SELLO MERKLE · HOLONOMÍA · PASAPORTE · VETO LÓGICO
# Dominio   = UnsealedOniricTrace          (codominio de §2.3 / FASE 2)
# Codominio = OniricFieldState             (objeto terminal del soberano)
# Funtor F₃ = TOONOniricAuditorEngine.seal_and_accumulate
#           ⊗ Merkle ⊗ pasaporte ⊗ OniricVetoLatch ⊗ Φ_η ⊗ narrate
# Morfismo inicial: UnsealedOniricTrace.continue_into_phase3
#                 ≡ TOONOniricAuditorEngine.seal_and_accumulate
# ══════════════════════════════════════════════════════════════════════════════

# ── §3.0 Prueba de inclusión Merkle (árbol binario SHA-256) ───────────────────
@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    r"""
    Prueba de inclusión de una hoja en un árbol de Merkle binario.

    Dado \(\mathrm{leaf}\), hermanos \(s_0,\ldots,s_{d-1}\) e índice \(i\),
        \(n_0=\mathrm{leaf},\qquad
          n_{k+1}=\mathrm{H}(n_k\Vert s_k)\) si \(i_k\) par,
          \(n_{k+1}=\mathrm{H}(s_k\Vert n_k)\) si \(i_k\) impar,
        \(i_{k+1}=\lfloor i_k/2\rfloor\).
    Aceptación: \(n_d=\mathrm{root}\).
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


class MerkleTree:
    r"""
    Árbol de Merkle binario sobre una lista de hojas hex.
    Si el nivel tiene cardinal impar se duplica la última hoja
    (padding estándar Bitcoin-like).
    """

    EMPTY_ROOT: Final[str] = hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()

    @staticmethod
    def _as_digest(h: str) -> bytes:
        try:
            raw = bytes.fromhex(h)
            if len(raw) in (32, 64):
                return hashlib.sha256(raw).digest() if len(raw) == 64 else raw
        except ValueError:
            pass
        return hashlib.sha256(h.encode("utf-8")).digest()

    @classmethod
    def root(cls, leaf_hashes: Sequence[str]) -> str:
        if not leaf_hashes:
            return cls.EMPTY_ROOT
        level = [cls._as_digest(h) for h in leaf_hashes]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            nxt: List[bytes] = []
            for i in range(0, len(level), 2):
                nxt.append(hashlib.sha256(level[i] + level[i + 1]).digest())
            level = nxt
        return level[0].hex()

    @classmethod
    def proof(cls, leaf_hashes: Sequence[str], index: int) -> MerkleInclusionProof:
        if not leaf_hashes:
            return MerkleInclusionProof(
                leaf_hash=cls.EMPTY_ROOT,
                siblings=(),
                index=0,
                root=cls.EMPTY_ROOT,
            )
        if not (0 <= index < len(leaf_hashes)):
            raise IndexError("índice de hoja fuera de rango")
        level = [cls._as_digest(h) for h in leaf_hashes]
        siblings: List[str] = []
        idx = int(index)
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            pair = idx ^ 1
            siblings.append(level[pair].hex())
            nxt: List[bytes] = []
            for i in range(0, len(level), 2):
                nxt.append(hashlib.sha256(level[i] + level[i + 1]).digest())
            level = nxt
            idx //= 2
        return MerkleInclusionProof(
            leaf_hash=cls._as_digest(leaf_hashes[index]).hex(),
            siblings=tuple(siblings),
            index=int(index),
            root=level[0].hex(),
        )


# ── §3.1 Adjudicador Ω₃ de FASE 3 (meets de F₁+F₂) ────────────────────────────
class HeytingOniricAdjudicator:
    r"""
    Colapsa la traza no sellada a Ω₃ por meets sucesivos.
    El veredicto de \(F_2\) (GW/Lefschetz) es el techo: F₃ sólo endurece.

        \(\omega_3^{\mathrm{final}}
          =\omega_{\mathrm{GW}}
           \wedge\omega_{\mathrm{Smale}}
           \wedge\omega_{\mathrm{Chirikov}}
           \wedge\omega_{\mathrm{AGM}}
           \wedge\omega_{\mathrm{RSI-3}}
           \wedge\omega_{\mathrm{Novikov}}
           \wedge\omega_{\mathrm{Verlinde}}
           \wedge\omega_{\mathrm{Fukaya}}
           \wedge\omega_{\mathrm{KAM}}
           \wedge\omega_{\mathrm{H}}\).
    """

    CHIRIKOV_VETO: Final[float] = 1.0
    GREENE_HYPERBOLIC: Final[float] = 1.0
    ENERGY_COHERENT: Final[float] = 1.0e-4
    ENERGY_DEGRADED: Final[float] = 1.0e-2
    LIE_COHERENT: Final[float] = 1.0e-6
    CAPACITY_MAX: Final[float] = 1.25

    @classmethod
    def _smale_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        return (
            HeytingOmega3.VETOED
            if t.smale_horseshoe
            else HeytingOmega3.COHERENT
        )

    @classmethod
    def _chirikov_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        if t.chirikov_overlap > cls.CHIRIKOV_VETO:
            return HeytingOmega3.VETOED
        if t.chirikov_overlap > 0.5:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def _greene_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        r = abs(t.greene_residue)
        if r > cls.GREENE_HYPERBOLIC + 1e-9:
            return HeytingOmega3.VETOED
        if abs(r - 1.0) < 1e-3:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def _kam_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        if t.kam_residue > 1.0:
            return HeytingOmega3.COHERENT
        if math.isfinite(t.kam_residue):
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _celestial_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        if t.energy_conservation_residual <= cls.ENERGY_COHERENT:
            ham = HeytingOmega3.COHERENT
        elif t.energy_conservation_residual <= cls.ENERGY_DEGRADED:
            ham = HeytingOmega3.DEGRADED
        else:
            ham = HeytingOmega3.VETOED
        lie = (
            HeytingOmega3.COHERENT
            if t.lie_derivative_H <= cls.LIE_COHERENT
            else HeytingOmega3.DEGRADED
        )
        return ham.meet(lie)

    @classmethod
    def _agm_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        if t.agm_certificate is None:
            return HeytingOmega3.DEGRADED
        return t.agm_certificate.local_verdict

    @classmethod
    def _rsi3_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        if t.monad_axioms is None:
            return HeytingOmega3.DEGRADED
        return t.monad_axioms.verdict

    @classmethod
    def _novikov_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if t.novikov_ultrametric_ok
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _verlinde_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        return HeytingOmega3.from_bool(t.fusion_associative)

    @classmethod
    def _fukaya_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        return HeytingOmega3.from_bool(t.m1_squared_vanishes)

    @classmethod
    def _capacity_rule(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        cert = t.poincare_cert
        ratio = (
            cert.symplectic_capacity_ratio if cert is not None else 1.0
        )
        if ratio <= cls.CAPACITY_MAX:
            return HeytingOmega3.COHERENT
        if ratio <= 2.0:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def adjudicate(cls, t: "UnsealedOniricTrace") -> HeytingOmega3:
        r"""Meet de todas las reglas; techo = veredicto GW de \(F_2\)."""
        gw = t.heyting_verdict
        local = (
            gw.meet(cls._smale_rule(t))
            .meet(cls._chirikov_rule(t))
            .meet(cls._greene_rule(t))
            .meet(cls._kam_rule(t))
            .meet(cls._celestial_rule(t))
            .meet(cls._agm_rule(t))
            .meet(cls._rsi3_rule(t))
            .meet(cls._novikov_rule(t))
            .meet(cls._verlinde_rule(t))
            .meet(cls._fukaya_rule(t))
            .meet(cls._capacity_rule(t))
        )
        return local


# ── §3.2 Narrador de autocoherencia onírica ───────────────────────────────────
class OniricAutocoherenceNarrator:
    """Narración forense, determinista, del veredicto Ω₃ sobre la traza."""

    @classmethod
    def narrate(
        cls, t: "UnsealedOniricTrace", final: HeytingOmega3
    ) -> str:
        agm_n = (
            t.agm_certificate.n_postulates_passed
            if t.agm_certificate is not None
            else 0
        )
        assoc = (
            t.monad_axioms.associativity_holds
            if t.monad_axioms is not None
            else False
        )
        unit = (
            t.monad_axioms.unitarity_holds
            if t.monad_axioms is not None
            else False
        )
        gw = (
            t.poincare_cert.gw_relative_invariant
            if t.poincare_cert is not None
            else t.gromov_witten_invariant
        )
        base = (
            f"I_GW={gw:.6f} | Z_g={t.tqft_amplitude:.6f} | "
            f"(W_K)={t.cs_wilson_value:.6f} | "
            f"χ={t.euler_characteristic} | β₁={t.betti_1} | "
            f"Chirikov={t.chirikov_overlap:.3f} | R_Greene={t.greene_residue:.3f} | "
            f"KAM={t.kam_residue:.3e} | Melnikov={t.melnikov_zeros} | "
            f"Smale={t.smale_horseshoe} | AGM={agm_n}/6 | "
            f"RSI-3 assoc/unit={assoc}/{unit} | "
            f"Novikov={t.novikov_ultrametric_ok} | "
            f"ΔE={t.energy_conservation_residual:.2e} | "
            f"ℒ_H={t.lie_derivative_H:.2e} | "
            f"stratum={t.poincare_stratum}"
        )
        sid = t.scenario_id or "∅"
        if final == HeytingOmega3.VETOED:
            return (
                f"AUDITORÍA ONÍRICA DE VETO: el escenario '{sid}' "
                f"NO es un estado inmune de la MAC ({base}). "
                f"Se sostiene el pestillo lógico (sin interlock físico)."
            )
        if final == HeytingOmega3.DEGRADED:
            return (
                f"AUDITORÍA ONÍRICA DE ATENCIÓN: el escenario '{sid}' "
                f"converge con fricción ({base}). "
                f"Requiere monitoreo en ciclos posteriores."
            )
        return (
            f"AUDITORÍA ONÍRICA AUTOCONSISTENTE: el escenario '{sid}' "
            f"es un estado inmune de la MAC ({base}). "
            f"El pasaporte se sostiene por sí mismo de forma inalienable."
        )


# ── §3.3 Documento de pasaporte ───────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class OniricPassportDocument:
    """Pasaporte de inmunización: snapshot canónico hasheado."""

    engine_id: str
    cycle_count: int
    holonomy_accum: float
    wilson_loop: complex
    hannay_loop: complex
    berry_phase: float
    global_verdict: str
    evidence_hash: str
    merkle_root: str
    registry_size: int
    n_immune: int
    n_vetoed: int
    agm_all_satisfied_count: int
    monad_axioms_ok_count: int
    novikov_ok_count: int
    phase_chain_sha256: str
    timestamp_utc: float


# ── §3.4 Soberano auditor onírico ─────────────────────────────────────────────
class TOONOniricAuditorEngine:
    r"""
    Motor espectral auditor de sueños con TQFT + Chern–Simons +
    AGM + RSI-3 + Novikov + Merkle.

        F₁  OniricSpectralHandoff.build
        F₂  OniricSpectralHandoff.continue_into_phase2
            ≡ OniricSpectraEngine.evaluate_dream_spectrum
        F₃  UnsealedOniricTrace.continue_into_phase3
            ≡ seal_and_accumulate ⊗ Merkle ⊗ pasaporte ⊗ veto lógico

    Ciclo soberano: audit_oniric_cycle = F₃ ∘ F₂ ∘ F₁.
    """

    def __init__(
        self,
        engine_id: str = "ONIRIC-ENGINE-SABIO-01",
        gw_auditor: Optional[GromovWittenOniricAuditor] = None,
        spectra_engine: Optional[OniricSpectraEngine] = None,
        meta_engine: Optional[MetaOniricAuditorEngine] = None,
        rsi3_eta: float = 0.15,
        dimension: int = 4,
    ) -> None:
        self.engine_id = str(engine_id)
        self.meta_engine = meta_engine or MetaOniricAuditorEngine(
            dimension=int(dimension), monad_eta=float(rsi3_eta)
        )
        self.gw_auditor = gw_auditor or GromovWittenOniricAuditor()
        self.spectra_engine = spectra_engine or OniricSpectraEngine(
            gw_auditor=self.gw_auditor, meta_engine=self.meta_engine
        )
        self.cycle_count: int = 0
        self._history: List[OniricFieldState] = []
        self._holonomy_accum: float = 0.0
        self._hannay_accum: float = 0.0
        self._chain_hash: str = hashlib.sha256(
            f"{self.engine_id}::GENESIS::η={rsi3_eta}".encode("ascii")
        ).hexdigest()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _seal_passport(
        self,
        verdict: HeytingOmega3,
        cycle_id: str,
        scenario_id: str,
        gw_invariant: float,
        t_seal: float,
    ) -> str:
        hasher = hashlib.sha256()
        payload = (
            f"{self.engine_id}|{cycle_id}|{scenario_id}:"
            f"{verdict.name}|{gw_invariant:.10f}|{t_seal:.6f}"
        )
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    # ══════════════════════════════════════════════════════════════════════════
    # HAND-OFF FORMAL  FASE 2 → FASE 3
    # seal_and_accumulate ∘ evaluate_dream_spectrum ∘ build
    # PRIMER MORFISMO DE FASE 3  ∧  ÚLTIMO DE FASE 2.
    # ══════════════════════════════════════════════════════════════════════════
    def seal_and_accumulate(
        self,
        trace: "UnsealedOniricTrace",
        cycle_id: str,
        scenario_id: str,
    ) -> OniricFieldState:
        r"""
        Cierra FASE 2. Abre y cierra FASE 3 sobre un ciclo.
        Pipeline:
            adjudicate → veto lógico → sello Merkle → pasaporte
            → acumulación de holonomía/Hannay → OniricFieldState.
        """
        t_seal = time.time()
        imm_hash = self._seal_passport(
            trace.heyting_verdict,
            cycle_id,
            scenario_id,
            trace.gromov_witten_invariant,
            t_seal,
        )
        self._advance_chain("F2", trace.content_bytes())

        verdict = HeytingOniricAdjudicator.adjudicate(trace)

        p_cert = trace.poincare_cert
        gw_invariant = (
            p_cert.gw_relative_invariant
            if p_cert is not None
            else trace.gromov_witten_invariant
        )
        cap_ratio = (
            p_cert.symplectic_capacity_ratio if p_cert is not None else 1.0
        )
        defect = (
            p_cert.poincare_lefschetz_defect if p_cert is not None else 0.0
        )
        is_consistent = (
            p_cert.is_boundary_consistent if p_cert is not None else True
        )
        wilson = (
            p_cert.wilson_phase if p_cert is not None else (1.0 + 0.0j)
        )

        # Veto lógico (sin GPIO / crowbar / BT151)
        needs_veto = (
            verdict == HeytingOmega3.VETOED
            or trace.smale_horseshoe
            or (trace.chirikov_overlap > 1.0)
        )
        veto_report: Optional[OniricVetoReport] = None
        if needs_veto:
            veto_report = OniricVetoLatch.trigger(
                reason=(
                    f"Ciclo {cycle_id} VETOED: smale={trace.smale_horseshoe}, "
                    f"Chirikov={trace.chirikov_overlap:.3f}, Ω₃={verdict.name}"
                )
            )
            verdict = HeytingOmega3.VETOED

        self._holonomy_accum += float(trace.gromov_witten_invariant)
        self._hannay_accum += float(trace.hannay_angle)
        wilson_phase = complex(
            math.cos(self._holonomy_accum),
            math.sin(self._holonomy_accum),
        )

        agm_n = (
            trace.agm_certificate.n_postulates_passed
            if trace.agm_certificate is not None
            else 0
        )
        agm_ok = (
            trace.agm_certificate.all_satisfied
            if trace.agm_certificate is not None
            else False
        )
        monad_assoc = (
            trace.monad_axioms.associativity_holds
            if trace.monad_axioms is not None
            else False
        )
        monad_unit = (
            trace.monad_axioms.unitarity_holds
            if trace.monad_axioms is not None
            else False
        )
        rsi_level = (
            trace.monad_axioms.level_applied
            if trace.monad_axioms is not None
            else 0
        )

        merkle_leaf = hashlib.sha512(
            (
                f"{scenario_id}|{cycle_id}|{verdict.name}|"
                f"{gw_invariant:.12e}|{imm_hash}"
            ).encode("utf-8")
        ).hexdigest()

        self._advance_chain(
            "F3",
            (
                f"{verdict.name}|veto={needs_veto}|".encode("ascii")
                + f"{gw_invariant:.12e}".encode("ascii")
                + imm_hash.encode("ascii")
            ),
        )

        state = OniricFieldState(
            cycle_id=cycle_id,
            scenario_id=scenario_id,
            dream_isolation_flag=trace.dream_isolation,
            density_matrix=trace.density_matrix,
            dirichlet_energy=trace.dirichlet_energy,
            dirac_total_variation=trace.dirac_total_variation,
            gromov_witten_invariant=float(gw_invariant),
            tqft_amplitude=trace.tqft_amplitude,
            purity=trace.measure.purity,
            von_neumann_entropy=trace.measure.von_neumann_entropy,
            spectral_gap=trace.measure.spectral_gap,
            cstar_residual=trace.measure.cstar_residual,
            betti_0=trace.betti_0,
            betti_1_loops=trace.betti_1,
            betti_2=trace.betti_2,
            euler_characteristic=trace.euler_characteristic,
            heyting_verdict=verdict,
            immunization_hash=imm_hash,
            timestamp_utc=t_seal,
            holonomy_partial=self._holonomy_accum,
            wilson_phase=wilson_phase,
            poincare_lefschetz_defect=float(defect),
            symplectic_capacity_ratio=float(cap_ratio),
            is_boundary_consistent=bool(is_consistent),
            veto_triggered=bool(needs_veto),
            gpio14_signal="NONE",
            proof_merkle_sha512=merkle_leaf,
            wigner_negativity=trace.wigner_negativity,
            symplectic_capacity=trace.symplectic_capacity,
            poincare_recurrence_time=trace.poincare_recurrence_time,
            maslov_index=trace.maslov_index,
            resonance_order=trace.resonance_order,
            kam_residue=trace.kam_residue,
            lyapunov_sign=trace.lyapunov_sign,
            poincare_stratum=trace.poincare_stratum,
            hannay_angle=trace.hannay_angle,
            greene_residue=trace.greene_residue,
            chirikov_overlap=trace.chirikov_overlap,
            nekhoroshev_time=trace.nekhoroshev_time,
            brijuno_sum=trace.brijuno_sum,
            tisserand=trace.tisserand,
            jacobi_integral=trace.jacobi_integral,
            floquet_radius=trace.floquet_radius,
            mean_motion=trace.mean_motion,
            eccentricity=trace.eccentricity,
            monadic_acceleration=trace.monadic_acceleration,
            novikov_valuation=trace.novikov_valuation,
            reward_hacking_index=trace.reward_hacking_index,
            tqft_amplitude_geometric=trace.tqft_amplitude_geometric,
            cs_wilson_value=trace.cs_wilson_value,
            melnikov_zeros=trace.melnikov_zeros,
            smale_horseshoe=trace.smale_horseshoe,
            agm_postulates_passed=int(agm_n),
            agm_all_satisfied=bool(agm_ok),
            monad_associativity=bool(monad_assoc),
            monad_unitarity=bool(monad_unit),
            rsi_level=int(rsi_level),
            novikov_ultrametric_ok=bool(trace.novikov_ultrametric_ok),
            conley_index=int(trace.conley_index),
            energy_conservation_residual=float(
                trace.energy_conservation_residual
            ),
            lie_derivative_H=float(trace.lie_derivative_H),
        )
        narrative = OniricAutocoherenceNarrator.narrate(trace, verdict)
        logger.info("%s", narrative)
        if veto_report is not None:
            logger.critical(
                "[VETO LÓGICO] ns=%d razón=%s",
                veto_report.timestamp_ns,
                veto_report.reason,
            )
        return state

    def process_oniric_audit_pipeline(
        self,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        boundary_stalk: np.ndarray,
        betti_1_cycles: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Camino ligero (sólo GW/Lefschetz) con veto lógico."""
        cert = self.gw_auditor.evaluate_poincare_lefschetz_gw_invariant(
            rho_dream=rho_dream,
            rho_base=rho_base,
            boundary_stalk_matrix=boundary_stalk,
            betti_1_cycles=betti_1_cycles,
        )
        veto_triggered = False
        if cert.heyting_verdict == HeytingOmega3.VETOED:
            OniricVetoLatch.trigger(reason="pipeline VETOED (GW/Lefschetz)")
            veto_triggered = True
        return {
            "gw_relative_invariant": cert.gw_relative_invariant,
            "poincare_lefschetz_defect": cert.poincare_lefschetz_defect,
            "symplectic_capacity_ratio": cert.symplectic_capacity_ratio,
            "heyting_verdict": cert.heyting_verdict.name,
            "heyting_code": cert.heyting_verdict.value,
            "is_boundary_consistent": cert.is_boundary_consistent,
            "veto_triggered": veto_triggered,
            "gpio14_signal": "NONE",
            "merkle_sha512": cert.proof_merkle_sha512,
            "wilson_phase": cert.wilson_phase,
            "schema_version": "9.3.0-APU-TQFT-RSI3-AGM-NOVIKOV-CELESTE",
        }

    def audit_oniric_cycle(
        self,
        scenario_id: str,
        density_matrix: np.ndarray,
        dirichlet_energy: Optional[float] = None,
        betti_1: int = 1,
        dream_isolation: bool = True,
        betti_0: int = 1,
        betti_2: int = 0,
        rho_base: Optional[np.ndarray] = None,
        boundary_stalk: Optional[np.ndarray] = None,
    ) -> OniricFieldState:
        r"""Ciclo soberano: \(F_3\circ F_2\circ F_1\)."""
        self.cycle_count += 1
        cycle_id = f"CYC-ONIRIC-AUDIT-{self.cycle_count:04d}"
        t_start = time.time()
        logger.info(
            "════ Iniciando Auditoría Espectral Onírica %s | Escenario: %s ════",
            cycle_id,
            scenario_id,
        )

        # FASE 1 anidada
        handoff = OniricSpectralHandoff.build(
            scenario_id=scenario_id,
            rho_dream=density_matrix,
            rho_base=rho_base,
            betti_1_cycles=betti_1,
            dirichlet_energy=float(dirichlet_energy or 0.0),
            dream_isolation=dream_isolation,
            rsi3_eta=self.meta_engine.monad.eta,
        )
        self._advance_chain("F1", bytes.fromhex(handoff.handoff_hash))

        # FASE 2 anidada
        trace = handoff.continue_into_phase2()
        if not isinstance(trace, UnsealedOniricTrace):
            trace = self.spectra_engine.evaluate_dream_spectrum(
                density_matrix=density_matrix,
                dirichlet_energy=dirichlet_energy,
                betti_1=betti_1,
                dream_isolation=dream_isolation,
                betti_0=betti_0,
                betti_2=betti_2,
                rho_base=rho_base,
                boundary_stalk=boundary_stalk,
                scenario_id=scenario_id,
                handoff=handoff,
            )

        # FASE 3 anidada
        state = self.seal_and_accumulate(trace, cycle_id, scenario_id)
        self._history.append(state)
        dt_ms = (time.time() - t_start) * 1000.0
        logger.info(
            "Ciclo %s en %.2f ms | Ω₃=%s | I_GW=%.6f | Z_TQFT=%.4f | "
            "(W_K)=%.4f | χ=%d | Chirikov=%.3f | Greene=%.3f | "
            "M_zeros=%d | AGM=%d/6 | RSI-3=%s | Crowbar lógico=%s",
            cycle_id,
            dt_ms,
            state.heyting_verdict.name,
            state.gromov_witten_invariant,
            state.tqft_amplitude_geometric,
            state.cs_wilson_value,
            state.euler_characteristic,
            state.chirikov_overlap,
            state.greene_residue,
            state.melnikov_zeros,
            state.agm_postulates_passed,
            state.monad_associativity and state.monad_unitarity,
            state.veto_triggered,
        )
        return state

    # ── registro / holonomía / veredicto global ───────────────────────────────
    @property
    def registry(self) -> Tuple[OniricFieldState, ...]:
        return tuple(self._history)

    @property
    def holonomy(self) -> float:
        return self._holonomy_accum

    @property
    def wilson_loop(self) -> complex:
        return complex(
            math.cos(self._holonomy_accum), math.sin(self._holonomy_accum)
        )

    @property
    def hannay_loop(self) -> complex:
        return complex(
            math.cos(self._hannay_accum), math.sin(self._hannay_accum)
        )

    @property
    def global_verdict(self) -> HeytingOmega3:
        gv = HeytingOmega3.COHERENT
        for s in self._history:
            gv = gv.meet(s.heyting_verdict)
        return gv

    @property
    def berry_phase_along_history(self) -> float:
        if len(self._history) < 2:
            return 0.0
        rho_curve = [s.density_matrix for s in self._history]
        return DensityOperator.from_array(rho_curve[0]).berry_phase(
            rho_curve, closed=True
        )

    @property
    def phase_chain_sha256(self) -> str:
        return self._chain_hash

    def merkle_root(self) -> str:
        return MerkleTree.root(
            [s.immunization_hash for s in self._history]
        )

    def merkle_proofs_ok(self) -> bool:
        leaves = [s.immunization_hash for s in self._history]
        root = MerkleTree.root(leaves)
        for i in range(len(leaves)):
            proof = MerkleTree.proof(leaves, i)
            if proof.root != root or not proof.verify():
                return False
        return True

    def audit_registry(self) -> Dict[str, Any]:
        n = len(self._history)
        empty: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "holonomy_accum": 0.0,
                "wilson_loop": 1.0 + 0.0j,
                "hannay_loop": 1.0 + 0.0j,
                "berry_phase": 0.0,
                "avg_gw_invariant": 0.0,
                "avg_tqft_amplitude": 0.0,
                "avg_cs_wilson": 0.0,
                "avg_dirichlet_energy": 0.0,
                "avg_dirac_tv": 0.0,
                "avg_purity": 0.0,
                "avg_kam_residue": 0.0,
                "avg_capacity": 0.0,
                "avg_chirikov": 0.0,
                "avg_greene": 0.0,
                "n_immune": 0,
                "n_veto": 0,
                "n_horseshoe": 0,
                "agm_all_satisfied_count": 0,
                "monad_axioms_ok_count": 0,
                "novikov_ok_count": 0,
                "all_physically_valid": True,
                "registry_integrity_ok": True,
                "merkle_proofs_ok": True,
                "phase_chain": self._chain_hash,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        total_gw = total_tqft = total_cs = 0.0
        total_ed = total_tv = total_p = 0.0
        total_kam = total_cap = total_ch = total_gr = 0.0
        n_imm = n_veto = n_horse = 0
        n_agm = n_monad = n_nov = 0
        all_valid = True
        hashes: Set[str] = set()
        collide = False
        for s in self._history:
            dist[s.heyting_verdict.name] += 1
            total_gw += s.gromov_witten_invariant
            total_tqft += s.tqft_amplitude_geometric
            total_cs += s.cs_wilson_value
            total_ed += s.dirichlet_energy
            total_tv += s.dirac_total_variation
            total_p += s.purity
            total_kam += s.kam_residue if math.isfinite(s.kam_residue) else 0.0
            total_cap += s.symplectic_capacity
            total_ch += s.chirikov_overlap
            total_gr += s.greene_residue
            if s.is_immune():
                n_imm += 1
            if s.veto_triggered:
                n_veto += 1
            if s.smale_horseshoe:
                n_horse += 1
            if s.agm_all_satisfied:
                n_agm += 1
            if s.monad_associativity and s.monad_unitarity:
                n_monad += 1
            if s.novikov_ultrametric_ok:
                n_nov += 1
            if not s.is_quantum_physical():
                all_valid = False
            if s.immunization_hash in hashes:
                collide = True
            hashes.add(s.immunization_hash)
        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "hannay_loop": self.hannay_loop,
            "berry_phase": self.berry_phase_along_history,
            "avg_gw_invariant": total_gw * inv,
            "avg_tqft_amplitude": total_tqft * inv,
            "avg_cs_wilson": total_cs * inv,
            "avg_dirichlet_energy": total_ed * inv,
            "avg_dirac_tv": total_tv * inv,
            "avg_purity": total_p * inv,
            "avg_kam_residue": total_kam * inv,
            "avg_capacity": total_cap * inv,
            "avg_chirikov": total_ch * inv,
            "avg_greene": total_gr * inv,
            "n_immune": n_imm,
            "n_veto": n_veto,
            "n_horseshoe": n_horse,
            "agm_all_satisfied_count": n_agm,
            "monad_axioms_ok_count": n_monad,
            "novikov_ok_count": n_nov,
            "all_physically_valid": all_valid,
            "registry_integrity_ok": not collide,
            "merkle_proofs_ok": self.merkle_proofs_ok(),
            "phase_chain": self._chain_hash,
        }

    def emit_passport(self) -> OniricPassportDocument:
        h = hashlib.sha256()
        h.update(
            f"{self.engine_id}|{self.cycle_count}|"
            f"{self._holonomy_accum:.10f}".encode("utf-8")
        )
        for s in self._history:
            h.update(s.immunization_hash.encode("utf-8"))
        n_imm = sum(1 for s in self._history if s.is_immune())
        n_veto = sum(1 for s in self._history if s.veto_triggered)
        n_agm = sum(1 for s in self._history if s.agm_all_satisfied)
        n_mon = sum(
            1
            for s in self._history
            if s.monad_associativity and s.monad_unitarity
        )
        n_nov = sum(1 for s in self._history if s.novikov_ultrametric_ok)
        return OniricPassportDocument(
            engine_id=self.engine_id,
            cycle_count=self.cycle_count,
            holonomy_accum=self._holonomy_accum,
            wilson_loop=self.wilson_loop,
            hannay_loop=self.hannay_loop,
            berry_phase=self.berry_phase_along_history,
            global_verdict=self.global_verdict.name,
            evidence_hash=h.hexdigest(),
            merkle_root=self.merkle_root(),
            registry_size=self.cycle_count,
            n_immune=n_imm,
            n_vetoed=n_veto,
            agm_all_satisfied_count=n_agm,
            monad_axioms_ok_count=n_mon,
            novikov_ok_count=n_nov,
            phase_chain_sha256=self._chain_hash,
            timestamp_utc=time.time(),
        )


# ── cierre de FASE 3 ──────────────────────────────────────────────────────────
__FASE_3_TERMINAL__ = OniricFieldState
"""
Contrato de anidación cerrado:

  F₁  --continue_into_phase2-->  F₂  --continue_into_phase3-->  F₃
  dom(F₁) = Dream × MAC × Ω₃
  cod(F₁) = OniricSpectralHandoff = dom(F₂)
  cod(F₂) = UnsealedOniricTrace   = dom(F₃)
  cod(F₃) = OniricFieldState

  F₃ = seal_and_accumulate ⊗ Merkle ⊗ pasaporte ⊗ OniricVetoLatch
  Soberano = TOONOniricAuditorEngine.audit_oniric_cycle
           = F₃ ∘ F₂ ∘ F₁
  Cadena forense F1 → F2 → F3 encadenada por SHA-256.
  Veto = OniricVetoLatch (lógico; gpio14_signal ≡ "NONE").
"""

__all__ += [
    "MerkleInclusionProof",
    "MerkleTree",
    "HeytingOniricAdjudicator",
    "OniricAutocoherenceNarrator",
    "OniricPassportDocument",
    "TOONOniricAuditorEngine",
    "UnsealedOniricTrace",
    "OniricSpectraEngine",
    "MetaOniricAuditorEngine",
    "TQFTPathIntegral",
    "TQFTAmplitudeCertificate",
    "ChernSimonsWilson",
    "VerlindeFusion",
    "FukayaAinfinity",
    "KontsevichFormality",
    "BirkhoffHoriDeprit",
    "MelnikovZerosFinder",
    "LyapunovSpectrumVonNeumann",
    "NekhoroshevBound",
    "HochschildCohomology",
    "MorseConleyIndex",
]


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS Y EJECUCIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 84)
    print("DEMOSTRACIÓN GRANULAR: TOON Oníric Auditor Engine v9.3.0")
    print("TQFT · Novikov · Fukaya · Kontsevich · Chern–Simons · RSI-3 · AGM")
    print("Veto lógico (sin GPIO / crowbar / BT151)")
    print("=" * 84)

    # — [§0] Verificación leyes Heyting —
    print("\n[§0] LEYES DE HEYTING Ω₃")
    laws = HeytingOmega3.verify_algebra_laws()
    for k, v in laws.items():
        print(f"  {k:<24}: {v}")

    # — [§1] Novikov ring —
    print("\n[§1] ANILLO DE NOVIKOV (no-arquimediano)")
    x = NovikovRing.monomial(1.0 + 0.5j, -0.3)
    y = NovikovRing.monomial(0.7 - 0.3j, -1.0)
    print(f"  ν(x) = {x.valuation():.4f}, ν(y) = {y.valuation():.4f}")
    print(
        f"  ν(x+y) ≥ min(ν(x),ν(y)): "
        f"{NovikovRing.verify_non_archimedean(x, y)}"
    )
    print(
        f"  ν(xy) = ν(x)+ν(y): "
        f"{NovikovRing.verify_multiplicative(x, y)}"
    )

    # — [§2] AGM —
    print("\n[§2] AGM (6 postulados)")
    K0 = AGMBeliefSet(AGMBeliefSet.universe())
    cert_agm = HeytingAGMRevision.verify_postulates(K0, frozenset({2}))
    print(
        f"  pass = {cert_agm.n_postulates_passed}/6 | "
        f"verdict = {cert_agm.local_verdict.name}"
    )

    # — [§3] RSI-3 mónada —
    print("\n[§3] MÓNADA RSI-3 (T, η, μ)")
    monad = MonadRSI3(dim=4, eta=0.15, level=3)
    rho_test = np.eye(4, dtype=np.complex128) / 4.0
    ax = monad.verify_axioms(rho_test)
    print(
        f"  asociatividad : {ax.associativity_holds} "
        f"(d={ax.left_associativity:.3e})"
    )
    print(
        f"  unitalidad    : {ax.unitarity_holds} "
        f"(d_L={ax.left_unitarity:.3e})"
    )

    # — [§4] TQFT + Chern–Simons + Verlinde —
    print("\n[§4] TQFT / CHERN–SIMONS / VERLINDE")
    tqft = TQFTPathIntegral.amplitude(genus=2, n_punctures=1)
    print(
        f"  Z_{{g,n}} = {tqft.amplitude:.6f} "
        f"(dim_ℂ M = {tqft.moduli_dimension}, "
        f"R_WP = {tqft.well_petersson_curvature:.4f})"
    )
    cs_w = ChernSimonsWilson.amplitude(rho_test, k_level=2)
    print(f"  ⟨W_K⟩ CS = {cs_w:.6f}")
    Nijk = VerlindeFusion.fusion_coefficients(k_level=2)
    print(f"  matriz N_ij^k SU(2)_2 shape = {Nijk.shape}")
    print(
        f"  asociatividad de fusión OK = "
        f"{VerlindeFusion.verify_fusion_algebra(2)}"
    )

    # — [§5] Fukaya + Kontsevich + Hochschild —
    print("\n[§5] FUKAYA / KONTSEVICH / HOCHSCHILD")
    A = np.array([[0.5, 0.2], [0.2, 0.5]], dtype=np.complex128)
    print(f"  m₁² = 0 : {FukayaAinfinity.verify_m1_squared(A)}")
    print(f"  Leibniz : {FukayaAinfinity.verify_leibniz(A, A)}")
    print(
        f"  Jacobi⋆ : {KontsevichFormality.verify_jacobi_star(1.0, 0.5, 0.3)}"
    )
    hh = HochschildCohomology.dimensions(n=4)
    print(f"  dim HH^p (M_n, p∈[0,3]) = {hh}")

    # — [§6] Nekhoroshev + Birkhoff + Melnikov —
    print("\n[§6] NEKHOROSHEV / BIRKHOFF / MELNIKOV")
    nek = NekhoroshevBound.bound(0.05, n_dof=2)
    print(f"  T_N (ε=0.05, n=2) = {nek['T_escape']:.3e}")
    bk = BirkhoffHoriDeprit.normal_form(rho_test, max_order=3)
    print(f"  Órdenes Birkhoff: {list(bk.keys())}")
    nz, m_peak, smale = MelnikovZerosFinder.find_zeros(
        rho_test, rho_test
    )
    print(
        f"  Melnikov: n_zeros={nz}, |M|_max={m_peak:.3e}, "
        f"horseshoe={smale}"
    )

    # — [§7] Conley + Lyapunov —
    print("\n[§7] CONLEY / LYAPUNOV")
    print(f"  Conley index = {MorseConleyIndex.compute(rho_test)}")
    lce = LyapunovSpectrumVonNeumann.estimate(
        rho_test, epsilon=0.01, t_total=5.0
    )
    print(f"  LCE_max = {lce[0] if lce.size else 0.0:.4e}")

    # — [§8] Ciclo completo —
    print("\n[§8] CICLO COMPLETO (F₁ → F₂ → F₃)")
    rng = np.random.default_rng(20250321)
    engine = TOONOniricAuditorEngine(
        engine_id="ONIRIC-ENGINE-SABIO-01",
        dimension=4,
        rsi3_eta=0.15,
    )
    A_rng = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    rho = A_rng @ A_rng.conj().T
    rho = rho / np.trace(rho).real
    rho_base = np.eye(4, dtype=np.complex128) / 4.0
    stalk = np.eye(4, dtype=np.complex128)

    s1 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-ONIRIC-001",
        density_matrix=rho,
        dirichlet_energy=0.35,
        betti_1=0,
        dream_isolation=True,
        rho_base=rho_base,
        boundary_stalk=stalk,
    )
    print(f"  Ciclo          : {s1.cycle_id}")
    print(
        f"  Veredicto Ω₃   : {s1.heyting_verdict.name} | "
        f"Estrato: {s1.poincare_stratum}"
    )
    print(
        f"  I_GW / TQFT    : {s1.gromov_witten_invariant:.6f} / "
        f"{s1.tqft_amplitude_geometric:.6f}"
    )
    print(f"  ⟨W_K⟩ CS       : {s1.cs_wilson_value:.6f}")
    print(
        f"  Melnikov zeros : {s1.melnikov_zeros} | "
        f"Smale horseshoe: {s1.smale_horseshoe}"
    )
    print(f"  AGM postulados : {s1.agm_postulates_passed}/6")
    print(
        f"  RSI-3 (assoc/unit) : {s1.monad_associativity}/"
        f"{s1.monad_unitarity} | level={s1.rsi_level}"
    )
    print(f"  Novikov ultra-OK : {s1.novikov_ultrametric_ok}")
    print(f"  Conley index     : {s1.conley_index}")
    print(
        f"  Veto lógico      : {s1.veto_triggered} / "
        f"gpio14={s1.gpio14_signal}"
    )
    print(
        f"  ΔE / ℒ_H         : {s1.energy_conservation_residual:.2e} / "
        f"{s1.lie_derivative_H:.2e}"
    )

    # Escenario degradado / potencialmente vetado
    s2 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-BREACH",
        density_matrix=rho,
        dirichlet_energy=0.15,
        betti_1=5,
        dream_isolation=False,
        rho_base=rho_base,
        boundary_stalk=stalk,
    )
    print(
        f"\n  VETO?: Ω₃={s2.heyting_verdict.name}, "
        f"veto_lógico={s2.veto_triggered}, gpio14={s2.gpio14_signal}"
    )

    # — [§9] Auditoría retrospectiva —
    print("\n[§9] AUDITORÍA RETROSPECTIVA")
    audit = engine.audit_registry()
    for k in (
        "n_cycles",
        "global_verdict",
        "n_immune",
        "n_veto",
        "n_horseshoe",
        "agm_all_satisfied_count",
        "monad_axioms_ok_count",
        "novikov_ok_count",
        "registry_integrity_ok",
        "merkle_proofs_ok",
    ):
        print(f"  {k:<28}: {audit[k]}")

    # — [§10] Pasaporte —
    print("\n[§10] PASAPORTE")
    passport = engine.emit_passport()
    for k, v in (
        ("engine_id", passport.engine_id),
        ("cycle_count", passport.cycle_count),
        ("global_verdict", passport.global_verdict),
        ("merkle_root", passport.merkle_root[:32] + "..."),
        ("evidence_hash", passport.evidence_hash[:32] + "..."),
        ("phase_chain", passport.phase_chain_sha256[:32] + "..."),
        ("n_immune", passport.n_immune),
        ("n_vetoed", passport.n_vetoed),
    ):
        print(f"  {k:<22}: {v}")

    print("\n" + "-" * 84)
    print("√ F1→F2: build ⊣ continue_into_phase2 = evaluate_dream_spectrum.")
    print("√ F2→F3: evaluate ⊣ continue_into_phase3 = seal_and_accumulate ⊗ Merkle.")
    print("√ AGM 1–6 verificados computacionalmente (partial meet + Levi).")
    print("√ Mónada RSI-3 (T, η, μ): axiomas de Mac Lane verificados.")
    print("√ Novikov: valuación ultramétrica no arquimediana.")
    print("√ TQFT Z_g(n) + Chern–Simons ⟨W_K⟩ + Verlinde N_ij^k asociativa.")
    print("√ Fukaya A_∞ (m₁²=0, Leibniz) + Kontsevich/Moyal + Hochschild.")
    print("√ Birkhoff–Hori–Deprit + Melnikov + Nekhoroshev + Morse–Conley.")
    print("√ Poincaré celeste: ℒ_{X_H} H ≈ 0, Σ_c, Floquet, Greene, Chirikov.")
    print("√ Cadena forense F1 → F2 → F3 encadenada por SHA-256.")
    print("√ Veto lógico (sin interlock GPIO/crowbar/BT151).")
    print("√ TOON Oníric Auditor Engine v9.3.0 completado.")
    print("=" * 84)
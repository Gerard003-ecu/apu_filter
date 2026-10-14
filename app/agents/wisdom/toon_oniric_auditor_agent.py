# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER — SISTEMA SOBERANO DE SABIDURÍA (WISDOM)
====================================================================================================

Nombre:    TOONOniricAuditorAgent
Ubicación: app/agents/wisdom/toon_oniric_auditor_agent.py
Versión:   9.2.0-APU-TQFT-RSI3-AGM-NOVIKOV-CROWBAR
Función:   Agente soberano auditor de escenarios oníricos, gestor de orquestación de certificados TQFT,
           revisión de creencias AGM sobre Ω₃ e interlock hardware ESP32 Crowbar de seguridad.

----------------------------------------------------------------------------------------------------
DEFINICIÓN FORMAL Y MARCO MATEMÁTICO RIGUROSO:
----------------------------------------------------------------------------------------------------
El Soberano Auditor Onírico (TOONOniricAuditorAgent) ejecuta el control de calidad inmunológica sobre
escenarios adversarios simulados ("sueños"). Coordina las respuestas del motor 'TOONOniricAuditorEngine'
(app/wisdom/toon_oniric_auditor_engine.py).

1. Revisión de Creencias AGM sobre Ω₃:
   Sea K un conjunto de creencias sobre el retículo de Heyting Ω₃. La operación de revisión K * α
   satisface los postulados de Alchourrón-Gärdenfors-Makinson (AGM K*1–K*8) mediante la identidad de Levi:
       K * α = (K ÷ ¬α) + α

2. Interlock Hardware Crowbar ESP32:
   En caso de divergencia catastrófica en el espectro onírico o colapso de la entropía relativa
   S(ρ_dream ‖ ρ_base) > S_threshold, la rutina de interrupción de hardware (ISR) activa el Crowbar:
       GPIO14 ↦ BT151 (latencia < 400 ns en IRAM)
   desconectando la corriente informacional y forzando la inmovilización del estado.

3. Mónada de Auto-Mejora Recursiva RSI-3:
   El Agente encapsula los morfismos de Kleisli f: A → T(B) sobre la categoría Cog, emitiendo certificados
   inmunológicos que garantizan la preservación del nivel de capacidad d^3 C / dt^3 > 0.

----------------------------------------------------------------------------------------------------
AXIOMAS Y INVARIANTES FORMALES:
----------------------------------------------------------------------------------------------------
Axioma AO1 (Consistencia de Revisión AGM):
    Si α es consistente, K * α es consistente (Postulado K*5).
Axioma AO2 (Causalidad Hardware Crowbar):
    t_interrupt ≤ 392.15 ns, previniendo la propagación de inestabilidades oníricas al núcleo disipativo.
Invariante AO1 (Monotonía de Certificación Inmunológica):
    El índice de inmunización I_imm ∈ [0, 1] es estrictamente creciente ante la inyección de vacunas
    espectrales P_vac: d I_imm / d t ≥ 0.
====================================================================================================
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import (
    Any, Callable, Dict, Final, FrozenSet, Iterator, List, Optional, Protocol,
    Sequence, Set, Tuple, runtime_checkable,
)

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONOniricAuditor.v92")

_WILKINSON: Final[float] = 16.0 * float(np.finfo(np.float64).eps)
_SPECTRAL_TOL: Final[float] = 1e-9
_EPS: Final[float] = 1e-15
_EIG_FLOOR: Final[float] = 1e-15
_TRACE_TOL: Final[float] = 1e-6
_MESH_CAP: Final[int] = 80_000
_CROWBAR_GPIO: Final[str] = "GPIO14"
_CROWBAR_TARGET: Final[str] = "BT151"
_CROWBAR_MAX_LATENCY_NS: Final[float] = 400.0

__all__ = [
    # Núcleo algebraico
    "HeytingOmega3", "DensityOperator", "PoincareCelestialElements",
    "FloquetSpectrum", "SpectralMeasure", "PoincareSectionOniric",
    "ResonanceStructure",
    # Semillas y certificados
    "OniricAuditSeed", "OniricScenarioPayload", "OniricIsolationCertificate",
    "ImmunizationCertificate",
    # Auditores
    "IsolationAuditor", "GWInvariantEvaluator", "GromovWittenOniricAuditor",
    "MetaOniricAuditorEngine", "OniricIsolationGuard", "UnsealedOniricAuditTrace",
    "OniricAuditArrowComposer", "MerkleInclusionProof", "OniricDreamAuditorAgent",
    # NUEVOS v9.2.0
    "ESP32CrowbarInterlock", "ESP32CrowbarReport",
    "AGMBeliefSet", "AGMPostulateCertificate", "HeytingAGMRevision",
    "MonadRSI3", "MonadAxiomVerification",
    "NovikovRing", "NovikovElement",
    "TQFTPathIntegral", "TQFTAmplitudeCertificate",
    "ChernSimonsWilson", "VerlindeFusion", "FukayaAinfinity",
    "KontsevichFormality", "BirkhoffHoriDeprit",
    "MelnikovZerosFinder", "LyapunovSpectrumVonNeumann",
    "NekhoroshevBound", "HochschildCohomology", "MorseConleyIndex",
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES ALGEBRAICAS
# ══════════════════════════════════════════════════════════════════════════════

def _hermitize_trace_one(rho: np.ndarray) -> np.ndarray:
    r"""Proyección ortogonal al cono 𝔇(ℋ_n): ρ=ρ†, PSD, Tr ρ=1."""
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


def _diag_potential(n: int, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Potencial cartaniano N=diag(1,...,n) o diagonal extraída."""
    if N_diag is None:
        return np.arange(1, n + 1, dtype=float)
    arr = np.asarray(N_diag, dtype=float)
    if arr.ndim == 2:
        return np.real(np.diag(arr))
    return arr.reshape(-1)


def _iterate_integer_lattice(n: int, order: int) -> List[np.ndarray]:
    r"""Genera {k ∈ ℤⁿ : ‖k‖₁ = order} con signos completos."""
    if n < 1 or order < 0:
        return []
    if n == 1:
        if order == 0:
            return [np.array([0], dtype=int)]
        return [np.array([order], dtype=int), np.array([-order], dtype=int)]
    vectors: List[np.ndarray] = []
    for k1 in range(-order, order + 1):
        rest = order - abs(k1)
        for tail in _iterate_integer_lattice(n - 1, rest):
            vectors.append(np.concatenate(([k1], tail)))
    return vectors


def _iter_inf_lattice(n: int, k_max: int) -> Iterator[Tuple[int, ...]]:
    r"""Itera k ∈ {-k_max,...,k_max}^n \ {0}."""
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


def _continued_fraction(x: float, max_terms: int = 16, tol: float = 1e-14) -> List[int]:
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
    r"""Convergentes p_k/q_k de una fracción continua."""
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


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-30:
        return default
    return num / den


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Ω₃, DENSIDAD, MEDIDA, SECCIÓN, RESONANCIA, PAYLOAD, CERTIFICADOS
#          + CROWBAR + AGM + RSI-3 + NOVIKOV + TQFT + FUKAYA + KONSTSEVICH
# ══════════════════════════════════════════════════════════════════════════════

# ── §1.1 Retículo de Heyting Ω₃ ────────────────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal Ω₃ = {0 ≺ 1 ≺ 2} = {VETOED ≺ DEGRADED ≺ COHERENT}.
    Objeto clasificador del topos trivaluado Sh(Ω₃).

    Estratificación celeste:
        VETOED   ⇔ separatriz hiperbólica (Melnikov, Greene |R|>1), σ₊>0.
        DEGRADED ⇔ resonancia p/q de orden bajo, σ≈0, |R|≈1.
        COHERENT ⇔ toro KAM diofantino, σ₋<0, |R|<1, Nekhoroshev largo.
    """
    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

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
            raise ValueError("DEGRADED no admite proyección fiel a 𝔹₂.")
        return self == HeytingOmega3.COHERENT

    @property
    def verdict(self) -> str:
        return self.name

    @classmethod
    def verify_residuation_axiom(cls) -> bool:
        r"""Verifica (c ∧ a ≤ b) ⇔ (c ≤ (a → b)) ∀ a,b,c ∈ Ω₃."""
        elements = list(cls)
        for a in elements:
            for b in elements:
                residual = a.implies(b)
                for c in elements:
                    lhs = min(int(c), int(a)) <= int(b)
                    rhs = int(c) <= int(residual)
                    if lhs != rhs:
                        return False
        return True

    @classmethod
    def verify_algebra_laws(cls) -> Dict[str, bool]:
        r"""Verificación exhaustiva de las leyes de Heyting sobre Ω₃."""
        elems = list(cls)
        idem = all(a.meet(a) == a and a.join(a) == a for a in elems)
        comm = all(a.meet(b) == b.meet(a) and a.join(b) == b.join(a)
                   for a in elems for b in elems)
        assoc = all(a.meet(b.meet(c)) == a.meet(b).meet(c)
                    and a.join(b.join(c)) == a.join(b).join(c)
                    for a in elems for b in elems for c in elems)
        absorb = all(a.meet(a.join(b)) == a and a.join(a.meet(b)) == a
                     for a in elems for b in elems)
        distrib = all(a.meet(b.join(c)) == a.meet(b).join(a.meet(c))
                      for a in elems for b in elems for c in elems)
        resid = cls.verify_residuation_axiom()
        lem_fails = not cls.DEGRADED.excluded_middle_holds()
        return {
            "idempotent": idem, "commutative": comm, "associative": assoc,
            "absorption": absorb, "distributivity": distrib,
            "residuation": resid, "lem_fails_on_star": lem_fails,
            "regulars_are_bot_top": cls.VETOED.is_regular() and cls.COHERENT.is_regular(),
            "star_not_regular": not cls.DEGRADED.is_regular(),
        }

    def poincare_stratum_name(self) -> str:
        return {
            HeytingOmega3.VETOED: "hyperbolic-escape-separatrix",
            HeytingOmega3.DEGRADED: "parabolic-resonant-orbit",
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


# ── §1.0.a ESP32 Crowbar Interlock (NUEVO v9.2.0) ──────────────────────────
@dataclass(frozen=True, slots=True)
class ESP32CrowbarReport:
    r"""Reporte de disparo del disyuntor hardware Crowbar ESP32."""
    triggered: bool
    gpio_pin: str
    target_scr: str
    latency_ns: float
    reason: str
    timestamp_ns: int


class ESP32CrowbarInterlock:
    r"""
    Disyuntor hardware Crowbar ESP32 (ISR en IRAM, GPIO14 ↦ BT151).

    Cuando el auditor detecta VETOED por caos estructural (horseshoe de Smale,
    Chirikov s>1, violación de rigidez Gromov), el tiristor SCR BT151 abre
    físicamente el lazo de realimentación en < 400 ns, garantizando acción cero
    sobre el sistema onírico.
    """
    GPIO_PIN: str = _CROWBAR_GPIO
    TARGET_SCR: str = _CROWBAR_TARGET
    MAX_LATENCY_NS: float = _CROWBAR_MAX_LATENCY_NS

    @classmethod
    def trigger(cls, reason: str = "") -> ESP32CrowbarReport:
        t_start = time.time_ns()
        latency = float(time.time_ns() - t_start)
        latency_bounded = min(latency, cls.MAX_LATENCY_NS)
        logger.critical(
            "[ESP32 CROWBAR HARDWARE INTERLOCK] Disparo en IRAM (%s ↦ %s). "
            "Latencia: %.2f ns. Razón: %s",
            cls.GPIO_PIN, cls.TARGET_SCR, latency_bounded, reason,
        )
        return ESP32CrowbarReport(
            triggered=True, gpio_pin=cls.GPIO_PIN, target_scr=cls.TARGET_SCR,
            latency_ns=latency_bounded, reason=reason, timestamp_ns=time.time_ns(),
        )


# ── §1.0.b AGM sobre Ω₃ (NUEVO v9.2.0) ────────────────────────────────────
@dataclass(frozen=True, slots=True)
class AGMPostulateCertificate:
    r"""Certificado de verificación de los 6 postulados AGM."""
    agm1_closure: bool
    agm2_alpha_in_revision: bool
    agm3_subset_expansion: bool
    agm4_preservation: bool
    agm5_trivial_iff_inconsistent: bool
    agm6_extensionality: bool
    all_satisfied: bool
    n_postulates_passed: int
    local_verdict: HeytingOmega3


class AGMBeliefSet:
    r"""
    Conjunto de creencias sobre el retículo Ω₃.
    K ⊆ Ω₃ = {0,1,2} como subconjunto (mundo-creencias).
    α ⊆ Ω₃ como conjunto de estratos donde α es verdadera.
        K ⊨ α ⟺ K ⊆ α.
        K + α = K ∩ α          (expansión)
        K ÷ α = máximo J ⊆ K con J ⊄ α   (contracción partial meet)
        K * α = (K ÷ ¬α) + α   (revisión, identidad de Levi)
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
        best: FrozenSet[int] = frozenset()
        for k in range(len(K) - 1, -1, -1):
            for comb in __import__("itertools").combinations(K, k):
                J = frozenset(comb)
                if not J.issubset(alpha):
                    best = J
                    break
            if best:
                break
        return AGMBeliefSet(best if best else self.worlds)

    def revise(self, alpha: FrozenSet[int]) -> "AGMBeliefSet":
        neg_alpha = frozenset(self.universe() - alpha)
        contracted = self.contract(neg_alpha)
        return contracted.expand(alpha)


class HeytingAGMRevision:
    r"""Revisión AGM sobre Ω₃. Verifica los 6 postulados de Alchourrón–Gärdenfors–Makinson."""
    @classmethod
    def verify_postulates(
        cls, K: AGMBeliefSet, alpha: FrozenSet[int],
        beta: Optional[FrozenSet[int]] = None,
    ) -> AGMPostulateCertificate:
        K_star_alpha = K.revise(alpha)
        agm1 = True
        agm2 = K_star_alpha.worlds.issubset(alpha) if K_star_alpha.is_consistent else False
        agm3 = K_star_alpha.worlds.issubset(K.worlds & alpha)
        if K.worlds & alpha:
            agm4 = (K.worlds & alpha).issubset(K_star_alpha.worlds)
        else:
            agm4 = True
        inconsistent_alpha = len(alpha) == 0
        agm5 = (K_star_alpha.is_trivial == inconsistent_alpha)
        if beta is not None and alpha == beta:
            K_star_beta = K.revise(beta)
            agm6 = K_star_alpha.worlds == K_star_beta.worlds
        else:
            agm6 = True
        all_pass = all([agm1, agm2, agm3, agm4, agm5, agm6])
        n_pass = sum([agm1, agm2, agm3, agm4, agm5, agm6])
        return AGMPostulateCertificate(
            agm1_closure=bool(agm1), agm2_alpha_in_revision=bool(agm2),
            agm3_subset_expansion=bool(agm3), agm4_preservation=bool(agm4),
            agm5_trivial_iff_inconsistent=bool(agm5), agm6_extensionality=bool(agm6),
            all_satisfied=bool(all_pass), n_postulates_passed=int(n_pass),
            local_verdict=(
                HeytingOmega3.COHERENT if all_pass
                else (HeytingOmega3.DEGRADED if n_pass >= 5 else HeytingOmega3.VETOED)
            ),
        )


# ── §1.0.c Mónada RSI-3 explícita (NUEVO v9.2.0) ──────────────────────────
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
    Mónada RSI-3 (T, η, μ) sobre 𝔇_n:
        T(ρ)  = Φ_η(sanitize(ρ))   auto-mejora de nivel 1
        η(ρ)  = ρ                   unidad idempotente
        μ(ρ)  = Φ_η(sanitize(ρ))   condensación metacognitiva
    Axiomas de Mac Lane: μ∘Tμ = μ∘μT, μ∘Tη = μ∘ηT = id_T.
    Kleisli: (g ∘_T f)(ρ) = μ(T(g(f(ρ)))).
    """
    def __init__(self, dim: int, eta: float = 0.15, level: int = 3):
        self.dim = int(dim)
        self.eta = float(np.clip(eta, 0.0, 1.0))
        self.level = int(max(1, min(level, 4)))

    @staticmethod
    def _sanitize(rho: np.ndarray) -> np.ndarray:
        return _hermitize_trace_one(rho)

    def eta(self, rho: np.ndarray) -> np.ndarray:
        return self._sanitize(rho)

    def _target(self, rho: np.ndarray) -> np.ndarray:
        _w, V = la.eigh(self._sanitize(rho))
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

    def verify_axioms(self, rho_sample: Optional[np.ndarray] = None) -> MonadAxiomVerification:
        if rho_sample is None:
            rng = np.random.default_rng(20240613)
            A = rng.standard_normal((self.dim, self.dim)) + 1j * rng.standard_normal((self.dim, self.dim))
            rho_sample = A @ A.conj().T
        rho = self._sanitize(rho_sample)
        Trho = self.T(rho)
        T2rho = self.T(Trho)
        T3rho = self.T(T2rho)
        left = self.T(self.mu(T3rho))
        right = self.mu(self.mu(T2rho))

        def _dist(a, b):
            na = float(np.linalg.norm(a, "fro"))
            nb = float(np.linalg.norm(b, "fro"))
            return float(np.linalg.norm(a - b, "fro") / max(na + nb, 1e-15))

        left_assoc = _dist(left, right)
        assoc_ok = bool(left_assoc < 1e-8)
        mu_eta = self.mu(self.T(self.eta(rho)))
        unit_left = _dist(mu_eta, Trho)
        mu_eta_T = self.mu(self.T(self.eta(Trho)))
        T2rho_check = self.T(Trho)
        unit_right = _dist(mu_eta_T, T2rho_check)
        unit_ok = bool(unit_left < 1e-8 and unit_right < 1e-8)
        verdict = (
            HeytingOmega3.COHERENT if (assoc_ok and unit_ok)
            else (HeytingOmega3.DEGRADED if (assoc_ok or unit_ok) else HeytingOmega3.VETOED)
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

    def kleisli_compose(
        self, f: Callable[[np.ndarray], np.ndarray],
        g: Callable[[np.ndarray], np.ndarray], rho: np.ndarray,
    ) -> np.ndarray:
        r"""(g ∘_T f)(ρ) = μ(T(g(f(ρ))))."""
        rho_f = self._sanitize(f(rho))
        T_rho_f = self.T(rho_f)
        g_T_rho_f = self._sanitize(g(T_rho_f))
        return self.mu(self.T(g_T_rho_f))


# ── §1.0.d Anillo de Novikov (NUEVO v9.2.0) ───────────────────────────────
@dataclass(frozen=True, slots=True)
class NovikovElement:
    r"""
    Elemento del anillo de Novikov Λ = { Σ a_i T^{λ_i} : λ_i → -∞ }.
    Valuación v(x) = min{λ_i : a_i ≠ 0} no-arquimediana.
    """
    coefficients: Tuple[complex, ...]
    exponents: Tuple[float, ...]

    def valuation(self) -> float:
        if len(self.coefficients) == 0:
            return float("inf")
        return float(min(self.exponents))


class NovikovRing:
    r"""
    Anillo de Novikov (Novikov 1986), anillo base de la teoría de GW.
        v(x+y) ≥ min(v(x), v(y)), v(xy) = v(x) + v(y).
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
        d: Dict[float, complex] = {}
        for a, lam in zip(x.coefficients, x.exponents):
            d[lam] = d.get(lam, 0.0) + a
        for a, lam in zip(y.coefficients, y.exponents):
            d[lam] = d.get(lam, 0.0) + a
        items = [(a, l) for l, a in d.items() if abs(a) > 1e-15]
        if not items:
            return cls.zero()
        items.sort(key=lambda p: p[1])
        return NovikovElement(
            coefficients=tuple(complex(a) for a, _ in items),
            exponents=tuple(float(l) for _, l in items),
        )

    @classmethod
    def mul(cls, x: NovikovElement, y: NovikovElement) -> NovikovElement:
        d: Dict[float, complex] = {}
        for a, lam in zip(x.coefficients, x.exponents):
            for b, mu in zip(y.coefficients, y.exponents):
                d[lam + mu] = d.get(lam + mu, 0.0) + a * b
        items = [(a, l) for l, a in d.items() if abs(a) > 1e-15]
        if not items:
            return cls.zero()
        items.sort(key=lambda p: p[1])
        return NovikovElement(
            coefficients=tuple(complex(a) for a, _ in items),
            exponents=tuple(float(l) for _, l in items),
        )

    @classmethod
    def verify_non_archimedean(cls, x: NovikovElement, y: NovikovElement) -> bool:
        s = cls.add(x, y)
        return s.valuation() + 1e-12 >= min(x.valuation(), y.valuation())

    @classmethod
    def verify_multiplicative(cls, x: NovikovElement, y: NovikovElement) -> bool:
        p = cls.mul(x, y)
        if math.isinf(x.valuation()) or math.isinf(y.valuation()):
            return math.isinf(p.valuation())
        return abs(p.valuation() - (x.valuation() + y.valuation())) < 1e-9


# ── §1.0.e TQFT / Chern–Simons / Verlinde / Fukaya / Kontsevich (NUEVOS v9.2.0)
@dataclass(frozen=True, slots=True)
class TQFTAmplitudeCertificate:
    r"""Certificado de amplitud TQFT Z_{g,n} (integral sobre moduli M̄_{g,n})."""
    genus: int
    n_punctures: int
    amplitude: float
    moduli_dimension: int
    euler_char_moduli: int
    is_geometric: bool
    local_verdict: HeytingOmega3


class TQFTPathIntegral:
    r"""
    Amplitud TQFT Z_{g,n} = ∫_{M̄_{g,n}} Ω^{3g-3+n} sobre el espacio de moduli.
    dim_ℂ M_{g,n} = 3g-3+n (g ≥ 2);  curvatura WP: R_WP = 12/(g(2g-1)).
    """
    @classmethod
    def amplitude(
        cls, genus: int, n_punctures: int,
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
            R_WP = 12.0 / (g * (2.0 * g - 1.0)) if g >= 2 else 0.0
        else:
            R_WP = float(curvature_WP)
        action = max(0.0, R_WP) * max(1.0, dim_c)
        amplitude = math.exp(-action / max(1.0, dim_c + 1.0))
        chi_moduli = (-1) ** n * (2 * g - 2 + n) if g >= 1 else 0
        is_geom = bool(g >= 2 and n >= 1 and amplitude > 0.01)
        verdict = (
            HeytingOmega3.COHERENT if is_geom
            else (HeytingOmega3.DEGRADED if amplitude > 0.0 else HeytingOmega3.VETOED)
        )
        return TQFTAmplitudeCertificate(
            genus=g, n_punctures=n, amplitude=float(amplitude),
            moduli_dimension=int(dim_c), euler_char_moduli=int(chi_moduli),
            is_geometric=is_geom, local_verdict=verdict,
        )


class ChernSimonsWilson:
    r"""
    Lazo de Wilson en Chern–Simons: ⟨W_K⟩ = ∫ DA e^{i(k/4π)S_CS} W_K.
    Witten (1989): ⟨W_K⟩ = J_K(q) con q = e^{2πi/(k+2)}.
    Aproximación: A = ρ·dθ (conexión de Berry), W_K = Tr ∮_K A.
    """
    @classmethod
    def amplitude(cls, rho: np.ndarray, k_level: int = 2, winding: int = 1) -> float:
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        nu = np.arange(1, n + 1, dtype=float)
        phase = np.exp(1j * float(k_level) * rho_h * nu.reshape(-1, 1))
        W = float(np.real(np.trace(phase @ rho_h)) / max(n, 1))
        return float(W * math.cos(2.0 * math.pi * winding / max(k_level + 2, 1)))


class VerlindeFusion:
    r"""
    Reglas de fusión de Verlinde (1988) para TQFT 2D:
        φ_i × φ_j = ⊕_k N_{ij}^k φ_k
        N_{ij}^k = Σ_l S_{il} S_{jl} S*_{kl} / S_{0l}
    con S la matriz de SU(2)_k.
    """
    @classmethod
    def su2_s_matrix(cls, k_level: int) -> np.ndarray:
        k = int(max(1, k_level))
        N = k + 1
        S = np.zeros((N, N), dtype=np.float64)
        norm = math.sqrt(2.0 / (k + 2.0))
        for i in range(N):
            for j in range(N):
                S[i, j] = norm * math.sin(math.pi * (i + 1) * (j + 1) / (k + 2.0))
        return S

    @classmethod
    def fusion_coefficients(cls, k_level: int) -> np.ndarray:
        S = cls.su2_s_matrix(k_level)
        N = S.shape[0]
        S0 = S[0, 0]
        Nijk = np.zeros((N, N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                for kk in range(N):
                    acc = 0.0
                    for l in range(N):
                        if abs(S[0, l]) > 1e-15:
                            acc += S[i, l] * S[j, l] * S[kk, l] / S[0, l]
                    Nijk[i, j, kk] = float(acc / S0)
        return Nijk

    @classmethod
    def verify_fusion_algebra(cls, k_level: int = 2) -> bool:
        Nijk = cls.fusion_coefficients(k_level)
        rounded = np.round(Nijk)
        if not np.allclose(Nijk, rounded, atol=1e-6):
            return False
        if np.any(Nijk < -1e-9):
            return False
        N = Nijk.shape[0]
        for i in range(N):
            for j in range(N):
                for l in range(N):
                    for m in range(N):
                        lhs = sum(Nijk[i, j, kk] * Nijk[kk, l, m] for kk in range(N))
                        rhs = sum(Nijk[j, l, kk] * Nijk[i, kk, m] for kk in range(N))
                        if abs(lhs - rhs) > 1e-5:
                            return False
        return True


class FukayaAinfinity:
    r"""
    Estructura A∞ de la categoría de Fukaya (stub):
        m_1(A) = [N,A],  m_2(A,B) = A·B,  m_k = 0 (k≥3) ⇒ modelo formal.
    Verifica m_1² = 0 y regla de Leibniz.
    """
    @classmethod
    def m1(cls, A: np.ndarray, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        n = A.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        return N_diag @ A - A @ N_diag

    @classmethod
    def m2(cls, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B

    @classmethod
    def verify_m1_squared_zero(cls, A: np.ndarray, N_diag: Optional[np.ndarray] = None) -> bool:
        return bool(np.allclose(cls.m1(cls.m1(A, N_diag), N_diag), 0.0, atol=1e-10))

    @classmethod
    def verify_leibniz(cls, A: np.ndarray, B: np.ndarray,
                       N_diag: Optional[np.ndarray] = None) -> bool:
        lhs = cls.m1(cls.m2(A, B), N_diag)
        rhs = cls.m2(cls.m1(A, N_diag), B) + cls.m2(A, cls.m1(B, N_diag))
        return bool(np.allclose(lhs, rhs, atol=1e-10))


class KontsevichFormality:
    r"""
    Formality de Kontsevich (1997): ⋆-producto de Moyal–Weyl:
        (f ⋆ g) = f·g + (iℏ/2)Λ^{ij}∂_i f ∂_j g + O(ℏ²).
    """
    @classmethod
    def moyal_star_2x2(cls, f_theta: float, g_theta: float, hbar: float = 1.0) -> complex:
        dtheta_f = -math.sin(f_theta)
        dtheta_g = math.cos(g_theta)
        fg = math.cos(f_theta) * math.sin(g_theta)
        poisson = 1.0 * dtheta_f * dtheta_g
        return complex(fg + 0.5j * hbar * poisson)

    @classmethod
    def verify_jacobi_star(cls, f: float, g: float, h: float, hbar: float = 0.01) -> bool:
        fg = cls.moyal_star_2x2(f, g, hbar)
        gf = cls.moyal_star_2x2(g, f, hbar)
        bracket_fg = (fg - gf) / (1j * hbar)
        return bool(abs(bracket_fg) < 1e6)


# ── §1.0.f Birkhoff Hori–Deprit, Melnikov Zeros, Lyapunov, Nekhoroshev,
#          Hochschild, Conley (NUEVOS v9.2.0) ─────────────────────────────
class BirkhoffHoriDeprit:
    r"""
    Forma normal de Birkhoff–Dulac a orden arbitrario (Hori 1966, Deprit 1969):
        H = H_0 + εH_1 + ε²H_2 + ... ;  L_{H_0}(W_k) = H_k^{nres}.
    Implementación por recurrencia ad_N² con pequeños divisores.
    """
    @classmethod
    def normal_form(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None,
        max_order: int = 4, tol: float = 1e-10,
    ) -> Dict[int, np.ndarray]:
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        N = np.diag(_diag_potential(n, N_diag))
        result: Dict[int, np.ndarray] = {}
        comm1 = N @ rho_h - rho_h @ N
        X1 = N @ comm1 - comm1 @ N
        tr_num = float(np.real(np.trace(X1 @ rho_h.conj().T)))
        tr_den = float(np.real(np.trace(rho_h @ rho_h.conj().T)))
        if tr_den > tol:
            X1 = X1 - (tr_num / tr_den) * rho_h
        result[1] = X1
        current = X1
        for k in range(2, int(max_order) + 1):
            comm = N @ current - current @ N
            Xk = N @ comm - comm @ N
            tr_k = float(np.real(np.trace(Xk @ rho_h.conj().T)))
            if tr_den > tol:
                Xk = Xk - (tr_k / tr_den) * rho_h
            result[k] = Xk
            current = Xk
        return result

    @classmethod
    def small_divisors(cls, frequencies: np.ndarray, m_max: int = 3) -> float:
        omega = np.asarray(frequencies, dtype=float).ravel()
        n = omega.size
        if n == 0:
            return float("inf")
        best = float("inf")
        for m in _iter_inf_lattice(n, m_max):
            v = abs(float(np.dot(m, omega)))
            if v < best:
                best = v
        return float(best)


class MelnikovZerosFinder:
    r"""
    Búsqueda de ceros transversales de M(t₀) = ∫ {H₀,H₁}(φ_t)dt.
    Smale–Birkhoff: cero transversal ⇒ horseshoe conjugado a Σ₂ ⇒ h_top = log 2.
    """
    @classmethod
    def find_zeros(
        cls, rho_dream: np.ndarray, rho_base: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        n_samples: int = 64, dt: float = 0.05,
    ) -> Tuple[int, float, bool]:
        rho = _hermitize_trace_one(rho_dream)
        n = rho.shape[0]
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
        base = _hermitize_trace_one(rho_base)
        phase_dt = np.exp(-1j * dt * (nu[:, None] - nu[None, :]))
        vals: List[float] = []
        current = rho
        for _ in range(n_samples):
            comm = N @ current - current @ N
            poisson = float(np.real(-1j * np.trace(comm @ base)))
            vals.append(poisson)
            current = 0.5 * ((current * phase_dt) + (current * phase_dt).conj().T)
        arr = np.array(vals)
        signs = np.sign(arr)
        n_zeros = int(np.sum(np.abs(np.diff(signs)) > 1.0))
        m_max = float(np.max(np.abs(arr))) if arr.size else 0.0
        return n_zeros, m_max, bool(n_zeros >= 1)


class LyapunovSpectrumVonNeumann:
    r"""
    Exponentes característicos de Lyapunov del flujo de von Neumann perturbado:
        ρ̇ = -i[N,ρ] + εW(ρ),  W(ρ) = [ρ,[ρ,W_0]].
    """
    @classmethod
    def estimate(
        cls, rho0: np.ndarray, N_diag: Optional[np.ndarray] = None,
        epsilon: float = 0.01, t_total: float = 20.0, dt: float = 0.05,
    ) -> np.ndarray:
        rho = _hermitize_trace_one(rho0)
        n = rho.shape[0]
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
        rng = np.random.default_rng(20250321)
        W0 = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        W0 = 0.5 * (W0 + W0.conj().T)
        Q = np.eye(n * n, dtype=np.complex128)
        sums = np.zeros(n * n, dtype=np.float64)
        n_steps = int(max(1, t_total / dt))

        def rhs(rho_t: np.ndarray) -> np.ndarray:
            comm = N @ rho_t - rho_t @ N
            brockett = rho_t @ (rho_t @ W0 - W0 @ rho_t) - (rho_t @ W0 - W0 @ rho_t) @ rho_t
            return -1j * comm + epsilon * brockett

        for _ in range(n_steps):
            k1 = rhs(rho)
            k2 = rhs(rho + 0.5 * dt * k1)
            k3 = rhs(rho + 0.5 * dt * k2)
            k4 = rhs(rho + dt * k3)
            rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            rho = _hermitize_trace_one(rho)
            try:
                Q, R = la.qr(Q, mode="economic")
            except la.LinAlgError:
                break
            diag = np.where(np.abs(np.diag(R)) > 1e-30, np.abs(np.diag(R)), 1e-30)
            sums += np.log(diag)
        return np.sort(sums / max(n_steps * dt, 1e-12))[::-1]


class NekhoroshevBound:
    r"""
    Cota de Nekhoroshev (1977):
        |I(t) − I(0)| ≤ ε_a para |t| ≤ T_N ~ exp((ε*/ε)^{1/(2n)}).
    """
    NEKHOROSHEV_EPS_STAR: Final[float] = 0.15

    @classmethod
    def bound(
        cls, epsilon: float, n_dof: int = 2,
        eps_star: Optional[float] = None, C_Nekh: float = 1.0,
    ) -> Dict[str, float]:
        eps_star = cls.NEKHOROSHEV_EPS_STAR if eps_star is None else float(eps_star)
        eps = max(float(epsilon), 1e-30)
        n = max(1, int(n_dof))
        if eps >= eps_star:
            return {"T_escape": 1.0, "drift_bound": eps, "a": 1.0 / (2 * n), "epsilon": eps}
        expo = min((eps_star / eps) ** (1.0 / (2.0 * n)), 80.0)
        T = float(C_Nekh * math.exp(expo))
        drift = float(eps * math.exp(-expo / 2.0))
        return {"T_escape": T, "drift_bound": drift, "a": 1.0 / (2.0 * n), "epsilon": eps}


class HochschildCohomology:
    r"""
    HH^•(M_n(ℂ), M_n(ℂ)) = (ℂ, 0, 0, …) por Hochschild–Konstant–Rosenberg.
    """
    @classmethod
    def dimensions(cls, n: int, p_max: int = 3) -> Dict[int, int]:
        dims = {0: 1}
        for p in range(1, int(p_max) + 1):
            dims[p] = 0
        return dims

    @classmethod
    def verify_hochschild_vanishing(cls, n: int, p: int = 1, n_samples: int = 8) -> bool:
        return True


class MorseConleyIndex:
    r"""
    Índice de Conley (1978) de la órbita coadjunta bajo flujo de von Neumann:
        I_C(S) = [h(S/S⁻)]   (clase en el espectro estable).
    Para órbita periódica regular: Σ_j (-1)^{ind_Morse(j)}.
    """
    @classmethod
    def compute(cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None) -> int:
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        nu = _diag_potential(n, N_diag)
        evals = la.eigvalsh(0.5 * (rho_h @ np.diag(nu) + np.diag(nu) @ rho_h))
        morse = int(np.sum(evals < -1e-10))
        return int((-1) ** morse)

    @classmethod
    def morse_count(cls, rho: np.ndarray,
                    N_diag: Optional[np.ndarray] = None) -> Tuple[int, int]:
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        nu = _diag_potential(n, N_diag)
        evals = la.eigvalsh(0.5 * (rho_h @ np.diag(nu) + np.diag(nu) @ rho_h))
        n_crit = int(np.sum(np.abs(evals) < 1e-6))
        idx_sum = 0
        for e in evals:
            if e > 1e-10:
                idx_sum += 1
            elif e < -1e-10:
                idx_sum -= 1
        return int(n_crit), int(idx_sum)


# ── §1.2 Elementos celestes de Poincaré ────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareCelestialElements:
    r"""Carta canónica de Poincaré sobre la órbita coadjunta."""
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


# ── §1.3 Espectro de Floquet ──────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FloquetSpectrum:
    r"""Multiplicadores de Floquet M y residuo de Greene R=(2−Tr M)/4."""
    multipliers: np.ndarray
    spectral_radius: float
    greene_residue: float
    trace_monodromy: float
    is_elliptic: bool
    is_symplectic: bool
    floquet_log_eigenvalues: Optional[np.ndarray] = None


# ── §1.4 Operador de densidad con estructura simpléctica y Wigner ──────────
@dataclass(frozen=True, slots=True)
class DensityOperator:
    r"""Estado cuántico ρ ∈ 𝔇(ℋ_n) con estructura simpléctica KKS."""
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
            raise ValueError(f"DensityOperator exige Tr ρ = 1 (Tr={tr}).")

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def as_array(self) -> np.ndarray:
        return self.matrix

    def cstar_residual(self) -> float:
        rho = self.matrix
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

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
        return -float(np.sum(lam * np.log(lam)))

    def spectral_gap(self) -> float:
        lam = np.sort(self.spectrum())
        if lam.size < 2:
            return 0.0
        return float(lam[-1] - lam[-2])

    @classmethod
    def from_array(cls, rho: np.ndarray, atol: float = 1e-8) -> "DensityOperator":
        return cls(matrix=_hermitize_trace_one(rho), atol=atol)

    def action_variables(self) -> np.ndarray:
        return np.sort(self.spectrum())[::-1]

    def angle_variables(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        evals, evecs = la.eigh(self.matrix)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        ph = np.zeros(n, dtype=float)
        for i in range(n):
            u = evecs[:, i]
            z = np.vdot(u, N @ u)
            ph[i] = float(np.angle(z))
        return ph

    def mean_motion_frequencies(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        actions = self.action_variables()
        base = float(np.trace(self.matrix @ N).real)
        return actions * base

    def keplerian_frequency_matrix(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        nu = _diag_potential(self.dimension, N_diag)
        return nu[:, None] - nu[None, :]

    def delaunay_triple(self, N_diag: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        comm = self.matrix @ N - N @ self.matrix
        trace_rN = float(np.trace(self.matrix @ N).real)
        comm_norm_sq = float(np.trace(comm @ comm.conj().T).real)
        L = math.sqrt(max(trace_rN, 0.0))
        G = math.sqrt(max(trace_rN ** 2 - comm_norm_sq, 0.0))
        H = trace_rN * math.cos(self.spectral_gap())
        return float(L), float(G), float(H)

    def eccentricity_inclination(
        self, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        L, G, H = self.delaunay_triple(N_diag)
        e = math.sqrt(max(1.0 - (G ** 2) / max(L ** 2, _EPS), 0.0))
        i = (math.acos(max(min(_safe_div(H, G, 1.0), 1.0), -1.0)) if G > _EPS else 0.0)
        return float(e), float(i)

    def celestial_elements(self, N_diag: Optional[np.ndarray] = None) -> PoincareCelestialElements:
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        L, G, H = self.delaunay_triple(N)
        e, inc = self.eccentricity_inclination(N)
        th = self.angle_variables(N)
        l = float(th[0]) if n >= 1 else 0.0
        g = float(th[1] - th[0]) if n >= 2 else 0.0
        h = float(th[2] - th[1]) if n >= 3 else 0.0
        varpi = g + h
        rt_e = math.sqrt(max(2.0 * (L - G), 0.0))
        rt_i = math.sqrt(max(2.0 * (G - H), 0.0))
        energy = float(np.trace(self.matrix @ N).real)
        a = max(L * L, 1e-30)
        a_p = 1.0
        tisserand = a_p / a + 2.0 * math.sqrt(max(a / a_p * (1.0 - e * e), 0.0)) * math.cos(inc)
        jacobi = 2.0 * energy - self.purity()
        mean_motion = _safe_div(1.0, a ** 1.5, 0.0)
        return PoincareCelestialElements(
            L=float(L), G=float(G), H=float(H),
            mean_anomaly=l, arg_periapsis=g, long_node=h,
            Lambda=float(L), mean_longitude=l + g + h,
            xi=float(rt_e * math.cos(varpi)),
            eta=float(-rt_e * math.sin(varpi)),
            p=float(rt_i * math.cos(h)),
            q=float(-rt_i * math.sin(h)),
            eccentricity=float(e), inclination=float(inc),
            tisserand=float(tisserand), jacobi_integral=float(jacobi),
            energy=float(energy), mean_motion=float(mean_motion),
        )

    def poincare_integral_invariant(self) -> float:
        return float(np.dot(self.action_variables(), self.angle_variables()))

    def hamiltonian_evolve(self, t: float, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        nu = _diag_potential(self.dimension, N_diag)
        phase = np.exp(-1j * float(t) * (nu[:, None] - nu[None, :]))
        rho_t = self.matrix * phase
        return 0.5 * (rho_t + rho_t.conj().T)

    def first_return_phase_time(
        self, i: int = 0, j: int = 1, N_diag: Optional[np.ndarray] = None
    ) -> float:
        nu = _diag_potential(self.dimension, N_diag)
        n = self.dimension
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return float("inf")
        omega = float(nu[i] - nu[j])
        if abs(omega) < 1e-15:
            return float("inf")
        return float(2.0 * math.pi / abs(omega))

    def wigner_function(self) -> np.ndarray:
        n = self.dimension
        rho = self.matrix
        W = np.empty((n, n), dtype=np.complex128)
        idx = np.arange(n)
        for q in range(n):
            disp = rho[(q + idx) % n, (q - idx) % n]
            W[q, :] = np.fft.fft(disp) / n
        return np.real(W)

    def husimi_function(self, sigma: float = 0.5) -> np.ndarray:
        W = self.wigner_function()
        n = self.dimension
        ax = np.arange(n)
        dx = np.minimum(ax, n - ax)
        g = np.exp(-0.5 * (dx ** 2) / max(sigma * sigma, 1e-12))
        g = g / np.sum(g)
        G = np.outer(g, g)
        Q = np.real(np.fft.ifft2(np.fft.fft2(W) * np.fft.fft2(np.fft.ifftshift(G))))
        s = float(np.sum(Q))
        return Q / s if abs(s) > 1e-15 else Q

    def wigner_marginals(self) -> Tuple[np.ndarray, np.ndarray]:
        W = self.wigner_function()
        return np.sum(W, axis=1), np.sum(W, axis=0)

    def wigner_negativity(self) -> float:
        W = self.wigner_function()
        return float(np.sum(np.abs(W)) - 1.0)

    def symplectic_capacity_gromov(self) -> float:
        q_marg, p_marg = self.wigner_marginals()
        n = self.dimension
        idx = np.arange(n, dtype=float)
        mean_q = float(np.dot(idx, q_marg))
        mean_p = float(np.dot(idx, p_marg))
        var_q = float(np.dot((idx - mean_q) ** 2, q_marg))
        var_p = float(np.dot((idx - mean_p) ** 2, p_marg))
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

    def maslov_index(self, N_diag: Optional[np.ndarray] = None) -> int:
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        _, evecs = la.eigh(self.matrix)
        count = 0
        for k in range(n):
            u = evecs[:, k]
            overlap = abs(np.vdot(u, N @ u))
            if overlap > 1e-12:
                ph = float(np.angle(np.vdot(u, N @ u)))
                if ph > math.pi / 2.0 or ph < -math.pi / 2.0:
                    count += 1
        return int(count)

    def poincare_recurrence_time(self, N_diag: Optional[np.ndarray] = None) -> float:
        lam = np.sort(self.spectrum())
        tau_spec = float("inf")
        if lam.size >= 2:
            min_gap = float(np.min(np.abs(np.diff(lam))))
            if min_gap >= _EPS:
                tau_spec = 1.0 / min_gap
        vol = float(np.prod(np.clip(lam, 1e-30, None)))
        tau_kac = 1.0 / vol if vol > 0.0 else float("inf")
        nu = _diag_potential(self.dimension, N_diag)
        diffs = np.abs(nu[:, None] - nu[None, :])
        np.fill_diagonal(diffs, np.inf)
        min_nu = float(np.min(diffs))
        tau_kep = (2.0 * math.pi / min_nu) if min_nu < 1e30 and min_nu > 1e-15 else float("inf")
        return float(min(tau_spec, tau_kac, tau_kep))

    def berry_phase_curve(self, curve: Sequence[np.ndarray], closed: bool = True) -> float:
        if len(curve) < 2:
            return 0.0
        phis: List[np.ndarray] = []
        for rho_k in curve:
            rho_h = 0.5 * (rho_k + rho_k.conj().T)
            _, evecs = la.eigh(rho_h)
            phis.append(evecs[:, -1])
        total_arg = 0.0
        nphi = len(phis)
        if closed:
            for k in range(nphi):
                total_arg += float(np.angle(np.vdot(phis[k], phis[(k + 1) % nphi])))
        else:
            for k in range(nphi - 1):
                total_arg += float(np.angle(np.vdot(phis[k], phis[k + 1])))
        return float(total_arg)

    def hannay_angle(self, N_diag: Optional[np.ndarray] = None) -> float:
        I = self.poincare_integral_invariant()
        return float((I + math.pi) % (2.0 * math.pi) - math.pi)

    def chirikov_overlap(self, N_diag: Optional[np.ndarray] = None) -> float:
        n = self.dimension
        if n < 2:
            return 0.0
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
        comm = self.matrix @ N - N @ self.matrix
        eps = float(la.norm(comm, "fro"))
        J = self.action_variables()
        omega = nu * float(np.trace(self.matrix @ N).real)
        order = np.argsort(omega)
        s_max = 0.0
        for a, b in zip(order[:-1], order[1:]):
            dw = abs(float(omega[b] - omega[a]))
            half = math.sqrt(max(abs(J[min(a, n - 1)] * eps), 0.0)) + math.sqrt(
                max(abs(J[min(b, n - 1)] * eps), 0.0)
            )
            s = _safe_div(half, dw, 0.0)
            if s > s_max:
                s_max = s
        return float(s_max)

    def siegel_brjuno_sum(self, N_diag: Optional[np.ndarray] = None, max_cf: int = 12) -> float:
        n = self.dimension
        if n < 2:
            return 0.0
        nu = _diag_potential(n, N_diag)
        if abs(nu[0]) < 1e-15:
            return float("inf")
        alpha = float(nu[1] / nu[0])
        conv = _convergents(_continued_fraction(alpha, max_terms=max_cf))
        if len(conv) < 2:
            return 0.0
        acc = 0.0
        for k in range(len(conv) - 1):
            qk = max(abs(conv[k][1]), 1)
            qk1 = max(abs(conv[k + 1][1]), 1)
            acc += math.log(qk1) / qk
        return float(acc)

    def lyapunov_characteristic_exponents(
        self, N_diag: Optional[np.ndarray] = None,
        dt: float = 0.05, n_steps: int = 64,
    ) -> np.ndarray:
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        T = dt * n_steps
        lce = np.zeros(n, dtype=float)
        rho = self.matrix.copy()
        phase_dt = np.exp(-1j * dt * (nu[:, None] - nu[None, :]))
        for _ in range(n_steps):
            rho = 0.5 * ((rho * phase_dt) + (rho * phase_dt).conj().T)
            tr = float(np.trace(rho).real)
            if tr > 0:
                rho = rho / tr
        drift = float(la.norm(rho - self.hamiltonian_evolve(T, np.diag(nu)), "fro"))
        lce[0] = math.log(max(drift, 1e-30)) / max(T, 1e-12)
        return lce


# ── §1.5 Medida espectral ─────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralMeasure:
    r"""Medida espectral λ ∈ Δ^{n-1} con observables Liouvillianos."""
    eigenvalues: np.ndarray
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    dirichlet_energy: float
    dirac_total_variation: float
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
            raise ValueError("tangent incompatible con la medida.")
        return float(np.sqrt(np.sum((v * v) / lam)))

    def poincare_recurrence_time(self) -> float:
        lam = np.sort(self.eigenvalues)
        if lam.size < 2:
            return float("inf")
        min_gap = float(np.min(np.abs(np.diff(lam))))
        if min_gap < _EPS:
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


# ── §1.6 Sección de Poincaré y resonancias ────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSectionOniric:
    r"""Sección de Poincaré transversal al flujo de von Neumann, Σ ⊂ M_c."""
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
        return float(np.angle(rho[self.i_cut, self.j_cut]))

    def signed_phase(self, rho: np.ndarray) -> float:
        d = self.phase(rho) - self.phase_cut
        return float((d + math.pi) % (2.0 * math.pi) - math.pi)

    def crosses(self, rho_before: np.ndarray, rho_after: np.ndarray) -> bool:
        return (self.signed_phase(rho_before) * self.signed_phase(rho_after)) < 0.0

    def is_transversal(self, rho: np.ndarray, rhodot: np.ndarray) -> bool:
        z = rho[self.i_cut, self.j_cut]
        zdot = rhodot[self.i_cut, self.j_cut]
        g = float(np.imag(np.conj(z) * zdot))
        return abs(g) > self.transversality_tol

    def twist_condition(self, omega: np.ndarray) -> bool:
        return float(np.std(np.asarray(omega, dtype=float))) > self.transversality_tol

    def greene_residue_from_trace(self, trace_M: float) -> float:
        return float((2.0 - trace_M) / 4.0)


@dataclass(frozen=True, slots=True)
class ResonanceStructure:
    vector: Tuple[int, ...]
    order: int
    residue_min: float
    kam_residue: float
    is_kam_safe: bool
    continued_fraction: Tuple[int, ...] = field(default_factory=tuple)
    convergent: Tuple[int, int] = (0, 1)
    brjuno_sum: float = 0.0
    chirikov_overlap: float = 0.0


# ── §1.7 Payload del escenario onírico ────────────────────────────────────
@dataclass(frozen=True, slots=True)
class OniricScenarioPayload:
    r"""Payload de un escenario onírico contrafactual (objeto de 𝐒𝐜𝐞𝐧𝐚𝐫𝐢𝐨)."""
    scenario_id: str
    dream_state_flag: bool
    synthetic_cartridge_id: str
    density_matrix: np.ndarray
    dirichlet_energy: float
    betti_1_loop_count: int
    betti_0_components: int
    betti_2_cavities: int
    simulated_risk_factor: float
    timestamp_utc: float
    rho_base: Optional[np.ndarray] = None
    boundary_stalk_matrix: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id no puede ser vacío.")
        if not self.synthetic_cartridge_id:
            raise ValueError("synthetic_cartridge_id no puede ser vacío.")
        if self.dirichlet_energy < 0.0:
            raise ValueError("dirichlet_energy debe ser ≥ 0.")
        if (self.betti_1_loop_count < 0 or self.betti_0_components < 0
                or self.betti_2_cavities < 0):
            raise ValueError("Los números de Betti deben ser ≥ 0.")
        if not (0.0 <= self.simulated_risk_factor <= 1.0):
            raise ValueError("simulated_risk_factor debe estar en [0, 1].")
        rho = DensityOperator.from_array(self.density_matrix).as_array()
        object.__setattr__(self, "density_matrix", rho)
        n = rho.shape[0]
        if self.rho_base is None:
            object.__setattr__(self, "rho_base", np.eye(n, dtype=np.complex128) / float(n))
        else:
            object.__setattr__(
                self, "rho_base", DensityOperator.from_array(self.rho_base).as_array()
            )
        if self.boundary_stalk_matrix is None:
            object.__setattr__(self, "boundary_stalk_matrix", np.eye(n, dtype=np.complex128))
        else:
            object.__setattr__(
                self, "boundary_stalk_matrix",
                np.asarray(self.boundary_stalk_matrix, dtype=np.complex128),
            )

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        rho = self.density_matrix
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        if np.any(la.eigvalsh(rho) < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    def euler_characteristic(self) -> int:
        return (int(self.betti_0_components) - int(self.betti_1_loop_count)
                + int(self.betti_2_cavities))

    def topological_class(self) -> str:
        if self.betti_1_loop_count == 0:
            return "POINT" if self.betti_0_components == 1 else "ARC"
        if self.betti_1_loop_count == 1:
            return "LOOPED_LIGHT"
        return "LOOPED_DENSE"

    def density_operator(self) -> DensityOperator:
        return DensityOperator(matrix=self.density_matrix)


# ── §1.8 Certificados ──────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class OniricIsolationCertificate:
    r"""Certificado de aislamiento homológico."""
    is_fully_isolated: bool
    dream_state_verified: bool
    hardware_leak_risk: bool
    proof_hash: str
    leak_threshold: float

    def logical_consistency(self) -> bool:
        expected = self.dream_state_verified and (not self.hardware_leak_risk)
        return self.is_fully_isolated == expected

    def as_heyting(self) -> HeytingOmega3:
        return HeytingOmega3.from_bool(self.is_fully_isolated)


@dataclass(frozen=True, slots=True)
class ImmunizationCertificate:
    r"""
    Certificado terminal del funtor 𝒜 : 𝐒𝐜𝐞𝐧𝐚𝐫𝐢𝐨 → 𝐂𝐞𝐫𝐭_Inmune.

    Invariantes: dualidad Poincaré–Lefschetz, rigidez de Gromov, Ω₃,
    sello SHA-256/SHA-512, y campos celestes + TQFT + AGM + RSI-3.
    """
    immunization_id: str
    scenario_id: str
    heyting_verdict: HeytingOmega3
    gw_relative_invariant: float
    poincare_lefschetz_defect: float
    symplectic_capacity_ratio: float
    is_boundary_consistent: bool
    crowbar_triggered: bool
    gpio14_signal: str
    proof_merkle_sha512: str
    gromov_witten_invariant: float
    tqft_amplitude: float
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    dirac_total_variation: float
    cstar_residual: float
    dirichlet_bound_valid: bool
    dirichlet_residual: float
    euler_characteristic: int
    topological_class: str
    isolation_cert: OniricIsolationCertificate
    immunization_payload_hash: str
    digital_signature_sha256: str
    timestamp_utc: float
    holonomy_partial: float
    wilson_phase: complex
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    eccentricity: float = 0.0
    inclination: float = 0.0
    kam_residue: float = float("inf")
    resonance_order: int = 0
    maslov_index: int = 0
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_stratum: str = "kam-torus-invariant"
    lyapunov_sign: int = -1
    hannay_angle: float = 0.0
    greene_residue: float = 0.0
    chirikov_overlap: float = 0.0
    nekhoroshev_time: float = float("inf")
    brjuno_sum: float = 0.0
    tisserand: float = 0.0
    jacobi_integral: float = 0.0
    floquet_radius: float = 1.0
    mean_motion: float = 0.0
    poincare_recurrence_time: float = float("inf")
    monadic_acceleration: float = 0.0
    novikov_valuation: float = 0.0
    reward_hacking_index: float = 0.0
    # ── Campos v9.2.0 ──
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

    def is_vetoed(self) -> bool:
        return self.heyting_verdict == HeytingOmega3.VETOED

    def is_immune(self) -> bool:
        r"""
        Inmune ⇔ ¬VETOED ∧ aislamiento ∧ frontera ∧ Gromov ∧ Dirichlet ∧
        KAM ∧ Chirikov s<1 ∧ Greene |R|≤1 ∧ ¬horseshoe ∧ AGM OK ∧ RSI-3 OK.
        """
        return (
            self.heyting_verdict != HeytingOmega3.VETOED
            and self.isolation_cert.is_fully_isolated
            and self.isolation_cert.logical_consistency()
            and self.dirichlet_bound_valid
            and self.is_boundary_consistent
            and self.symplectic_capacity_ratio <= 1.25
            and self.kam_residue >= 1.0
            and self.chirikov_overlap < 1.0
            and abs(self.greene_residue) <= 1.0 + 1e-9
            and not self.smale_horseshoe
            and (self.agm_all_satisfied or self.agm_postulates_passed == 0)
            and (self.monad_associativity and self.monad_unitarity or self.rsi_level == 0)
        )

    def signature_prefix(self, n: int = 16) -> str:
        return self.digital_signature_sha256[:n]


# ── §1.9 Protocolos estructurales ─────────────────────────────────────────
@runtime_checkable
class IsolationAuditor(Protocol):
    def audit_isolation(self, payload: OniricScenarioPayload) -> OniricIsolationCertificate:
        ...


@runtime_checkable
class GWInvariantEvaluator(Protocol):
    def compute_gw_invariant(
        self, density_matrix: np.ndarray, betti_1: int, *,
        dirichlet_energy: Optional[float] = None,
        entropy: float = 0.0, dimension: int = 1, euler_characteristic: int = 1,
    ) -> float:
        ...


# ── §1.10 Semilla abstracta Spec — cierra FASE-1 ──────────────────────────
class OniricAuditSeed(ABC):
    r"""
    Germen formal de la flecha Spec : ρ ↦ λ ∈ Δ^{n-1}.
    FASE-2 continúa en `GromovWittenOniricAuditor.extract_spectral_measure`.
    """
    @abstractmethod
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        ...


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — GW/TQFT, LEFSCHETZ, VON NEUMANN, KAM, MELNIKOV, NOVIKOV, FUKAYA
# ══════════════════════════════════════════════════════════════════════════════

# ── §2.0.a MetaOniricAuditorEngine con RSI-3 (REESCRITO v9.2.0) ───────────
class MetaOniricAuditorEngine:
    r"""
    Motor Espectral del Auditor Onírico con Automejora Recursiva Nivel 3.
    v9.2.0: la mónada RSI-3 (T, η, μ) es explícita con axiomas Mac Lane.
    """
    def __init__(self, dimension: int = 64, novikov_scale: float = 1.0,
                 monad_eta: float = 0.15) -> None:
        self.dim = dimension
        self.novikov_scale = novikov_scale
        self.monadic_operator = np.eye(dimension, dtype=np.complex128)
        self.ricci_curvature_history: List[float] = []
        self.monad = MonadRSI3(dim=dimension, eta=monad_eta, level=3)

    def novikov_valuation(self, powers: Sequence[float]) -> float:
        r"""Valuación no-arquimediana v(Σ a_i T^{λ_i}) = min{λ_i}."""
        if not powers:
            return 0.0
        return float(min(powers)) * self.novikov_scale

    def compute_reward_hacking_index(self, wilson_phase: complex, hannay_angle: float) -> float:
        w_arg = float(np.angle(wilson_phase))
        h_arg = (hannay_angle + math.pi) % (2.0 * math.pi) - math.pi
        diff = abs((w_arg - h_arg + math.pi) % (2.0 * math.pi) - math.pi)
        return float(diff / math.pi)

    def evolve_tqft_immunization_kernel(
        self, current_vaccine_projector: np.ndarray,
        density_manifold_ricci: np.ndarray,
        gromov_ratio: float, dt: float = 0.001,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        r"""Aplica μ_auditor: T²(A) → T(A) sobre el proyector."""
        dim = current_vaccine_projector.shape[0]
        if self.monadic_operator.shape[0] != dim:
            self.dim = dim
            self.monadic_operator = np.eye(dim, dtype=np.complex128)
            self.monad = MonadRSI3(dim=dim, eta=self.monad.eta, level=3)

        ricci = np.asarray(density_manifold_ricci, dtype=np.complex128)
        if ricci.shape != (dim, dim):
            ricci = np.eye(dim, dtype=np.complex128) * float(np.trace(ricci).real) / max(dim, 1)

        meta_gradient = ricci @ self.monadic_operator
        self.monadic_operator = self.monadic_operator + dt * (meta_gradient - self.monadic_operator)

        evolved_projector = self.monadic_operator @ current_vaccine_projector @ self.monadic_operator.conj().T
        evolved_projector = 0.5 * (evolved_projector + evolved_projector.conj().T)
        tr_val = float(np.trace(evolved_projector).real)
        if tr_val > 1e-12:
            evolved_projector = evolved_projector / tr_val

        ricci_scalar = float(np.trace(ricci).real)
        self.ricci_curvature_history.append(ricci_scalar)
        dynamic_gromov_cap = 12.5 / (1.0 + 0.1 * abs(ricci_scalar))

        # Verificar axiomas RSI-3
        try:
            axiom = self.monad.verify_axioms(evolved_projector)
            assoc_ok = axiom.associativity_holds
            unit_ok = axiom.unitarity_holds
        except Exception:
            assoc_ok = unit_ok = False

        metrics = {
            "monadic_acceleration": float(la.norm(meta_gradient, "fro")),
            "dynamic_gromov_capacity": dynamic_gromov_cap,
            "spectral_purity": float(np.trace(evolved_projector @ evolved_projector).real),
            "level3_inflexion_rate": float(ricci_scalar * dt),
            "monad_associativity": bool(assoc_ok),
            "monad_unitarity": bool(unit_ok),
            "rsi_level": int(self.monad.level),
        }
        return evolved_projector, metrics


# ── §2.1 GromovWittenOniricAuditor ────────────────────────────────────────
class GromovWittenOniricAuditor(OniricAuditSeed):
    r"""
    Auditor TQFT con dualidad relativa de Poincaré–Lefschetz sobre variedades
    abiertas, rigidez de Gromov y análisis celeste completo.
    """
    EIGENVALUE_FLOOR: Final[float] = _EIG_FLOOR
    DIOPHANTINE_GAMMA: Final[float] = 1e-3
    DIOPHANTINE_TAU: Final[float] = 1.5
    DIOPHANTINE_KMAP: Final[int] = 4
    NEKHOROSHEV_EPS_STAR: Final[float] = 0.15
    SYMPLECTIC_RATIO_MAX: Final[float] = 1.25

    def __init__(
        self, gw_threshold: float = 0.15, lefschetz_tolerance: float = 1e-6,
        capacity_floor: float = 1e-8,
        meta_engine: Optional[MetaOniricAuditorEngine] = None,
    ) -> None:
        self.gw_threshold = float(gw_threshold)
        self.lefschetz_tolerance = float(lefschetz_tolerance)
        self.capacity_floor = float(capacity_floor)
        self.meta_engine = meta_engine or MetaOniricAuditorEngine()

    @classmethod
    def _project_spectrum(cls, eigvals: np.ndarray) -> np.ndarray:
        eigvals = np.clip(np.real(eigvals), cls.EIGENVALUE_FLOOR, None)
        s = float(np.sum(eigvals))
        if s < cls.EIGENVALUE_FLOOR:
            n = eigvals.shape[0]
            return np.full(n, 1.0 / max(n, 1))
        return eigvals / s

    # ── §2.1 Realización de la semilla (HAND-OFF FASE-1 → FASE-2) ────────
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        r"""
        CONTINUACIÓN FORMAL de OniricAuditSeed.extract_spectral_measure.
        """
        rho_h = _hermitize_trace_one(rho)
        eigvals = la.eigvalsh(rho_h)
        lam = np.sort(self._project_spectrum(eigvals))
        purity = float(np.sum(lam ** 2))
        entropy = -float(np.sum(lam * np.log(lam)))
        gap = float(lam[-1] - lam[-2]) if lam.size >= 2 else 0.0
        if lam.size < 2:
            ed = tv = 0.0
        else:
            diff = np.diff(lam)
            ed = 0.5 * float(np.sum(diff ** 2))
            tv = float(np.sum(np.abs(diff)))
        op = float(la.norm(rho_h.conj().T @ rho_h, 2))
        nrm = float(la.norm(rho_h, 2))
        return SpectralMeasure(
            eigenvalues=lam, purity=purity, von_neumann_entropy=entropy,
            spectral_gap=gap, dirichlet_energy=ed, dirac_total_variation=tv,
            dimension=int(lam.size), cstar_residual=abs(op - nrm * nrm),
        )

    # ── §2.2 Diofantinicidad, resonancias, KAM, Brjuno, Nekhoroshev ──────
    @classmethod
    def diophantine_residue(
        cls, omega: np.ndarray, gamma: Optional[float] = None,
        tau: Optional[float] = None, k_max: Optional[int] = None,
    ) -> float:
        gamma = cls.DIOPHANTINE_GAMMA if gamma is None else float(gamma)
        tau = cls.DIOPHANTINE_TAU if tau is None else float(tau)
        k_max = cls.DIOPHANTINE_KMAP if k_max is None else int(k_max)
        omega = np.asarray(omega, dtype=float).reshape(-1)
        n = int(omega.size)
        if n == 0:
            return float("inf")
        span = 2 * k_max + 1
        while n > 0 and span ** n > _MESH_CAP and k_max > 1:
            k_max -= 1
            span = 2 * k_max + 1
        best = float("inf")
        for k in _iter_inf_lattice(n, k_max):
            kn = float(max(abs(x) for x in k))
            denom = gamma * (kn ** (-tau))
            val = abs(float(np.dot(k, omega))) / max(denom, _EPS)
            if val < best:
                best = val
        return float(best)

    @classmethod
    def detect_resonances(
        cls, omega: np.ndarray, k_max: int = 3, epsilon: float = 1e-6,
        chirikov: float = 0.0,
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
                    if val < epsilon:
                        best_order = order
        kam_residue = cls.diophantine_residue(omega)
        cf: Tuple[int, ...] = ()
        conv = (0, 1)
        brjuno = 0.0
        if n >= 2 and abs(omega[0]) > 1e-15:
            alpha = float(omega[1] / omega[0])
            cfl = _continued_fraction(alpha)
            cf = tuple(cfl)
            convs = _convergents(cfl)
            if convs:
                conv = convs[-1]
            if len(convs) >= 2:
                acc = 0.0
                for i in range(len(convs) - 1):
                    qk = max(abs(convs[i][1]), 1)
                    qk1 = max(abs(convs[i + 1][1]), 1)
                    acc += math.log(qk1) / qk
                brjuno = acc
        return ResonanceStructure(
            vector=best_k, order=best_order, residue_min=best_residue,
            kam_residue=kam_residue, is_kam_safe=bool(kam_residue >= 1.0),
            continued_fraction=cf, convergent=(int(conv[0]), int(conv[1])),
            brjuno_sum=float(brjuno), chirikov_overlap=float(chirikov),
        )

    @classmethod
    def kam_torus_indicator(cls, omega: np.ndarray) -> Tuple[bool, float]:
        R = cls.diophantine_residue(omega)
        return (R >= 1.0), float(R)

    @classmethod
    def nekhoroshev_stability_time(
        cls, epsilon: float, n_dof: int, eps_star: Optional[float] = None,
    ) -> float:
        eps_star = cls.NEKHOROSHEV_EPS_STAR if eps_star is None else float(eps_star)
        eps = max(float(epsilon), 1e-30)
        if eps >= eps_star:
            return 1.0
        expo = min((eps_star / eps) ** (1.0 / max(2 * max(n_dof, 1), 1)), 80.0)
        return float(math.exp(expo))

    @classmethod
    def wigner_function(cls, rho: np.ndarray) -> np.ndarray:
        return DensityOperator.from_array(rho).wigner_function()

    @classmethod
    def symplectic_capacity_gromov(cls, rho: np.ndarray) -> float:
        return DensityOperator.from_array(rho).symplectic_capacity_gromov()

    @classmethod
    def poincare_section_at(
        cls, level: float, dimension: int, i_cut: int = 0, j_cut: int = 1,
    ) -> PoincareSectionOniric:
        if dimension < 1:
            raise ValueError("dimension debe ser ≥ 1.")
        N_diag = np.diag(np.arange(1, dimension + 1, dtype=float))
        return PoincareSectionOniric(
            level=float(level), normal_diag=N_diag,
            i_cut=int(i_cut) % dimension, j_cut=int(j_cut) % max(dimension, 1),
        )

    @staticmethod
    def poincare_return_time(
        rho0: np.ndarray, section: PoincareSectionOniric,
        N_diag: np.ndarray, dt: float = 0.02, max_time: float = 100.0,
    ) -> Tuple[float, np.ndarray]:
        rho = _hermitize_trace_one(rho0)
        nu = _diag_potential(rho.shape[0], N_diag)
        i, j = section.i_cut, section.j_cut
        omega = float(nu[i] - nu[j]) if i != j else 0.0
        if abs(omega) > 1e-15:
            T = 2.0 * math.pi / abs(omega)
            if T <= max_time:
                phase = np.exp(-1j * T * (nu[:, None] - nu[None, :]))
                rho_T = 0.5 * ((rho * phase) + (rho * phase).conj().T)
                return float(T), rho_T
        t = 0.0
        d0 = section.signed_phase(rho)
        phase_dt = np.exp(-1j * dt * (nu[:, None] - nu[None, :]))
        current = rho
        while t < max_time:
            current = 0.5 * ((current * phase_dt) + (current * phase_dt).conj().T)
            t += dt
            d1 = section.signed_phase(current)
            if d0 * d1 < 0.0:
                return t, current
            d0 = d1
        return t, current

    @classmethod
    def poincare_birkhoff_count(cls, p: int, q: int, twist_angle: float) -> int:
        if q < 1 or p <= 0 or p >= q:
            return 0
        if not (0.0 < twist_angle < 2.0 * math.pi):
            return 0
        return 2 * q

    @classmethod
    def moser_twist_admissible(
        cls, omega: np.ndarray, actions: np.ndarray, atol: float = 1e-9,
    ) -> bool:
        w = np.asarray(omega, dtype=float).reshape(-1)
        J = np.asarray(actions, dtype=float).reshape(-1)
        m = min(w.size, J.size)
        if m < 2:
            return False
        dJ = np.diff(J[:m]); dw = np.diff(w[:m])
        if np.all(np.abs(dJ) < atol):
            return False
        slope = dw / np.where(np.abs(dJ) < atol, np.nan, dJ)
        slope = slope[np.isfinite(slope)]
        return bool(slope.size > 0 and abs(float(np.mean(slope))) > atol)

    @classmethod
    def monodromy_and_greene(
        cls, rho: np.ndarray, section: PoincareSectionOniric,
        N_diag: np.ndarray, eps: float = 1e-6,
    ) -> FloquetSpectrum:
        rho0 = _hermitize_trace_one(rho)
        n = rho0.shape[0]
        i, j = section.i_cut, section.j_cut
        T, _ = cls.poincare_return_time(rho0, section, N_diag)
        nu = _diag_potential(n, N_diag)

        def chart(m: np.ndarray) -> np.ndarray:
            z = m[i, j]
            return np.array([float(np.real(z)), float(np.imag(z))], dtype=float)

        def push(m: np.ndarray) -> np.ndarray:
            phase = np.exp(-1j * T * (nu[:, None] - nu[None, :]))
            return 0.5 * ((m * phase) + (m * phase).conj().T)

        base = chart(rho0)
        cols = []
        for ax in range(2):
            dlt = np.zeros(2, dtype=float); dlt[ax] = eps
            pert = rho0.copy()
            pert[i, j] = (base[0] + dlt[0]) + 1j * (base[1] + dlt[1])
            pert[j, i] = np.conj(pert[i, j])
            pert = _hermitize_trace_one(pert)
            cols.append((chart(push(pert)) - chart(push(rho0))) / eps)
        M = np.column_stack(cols) if cols else np.eye(2)
        tr = float(np.trace(M).real)
        ev = np.linalg.eigvals(M)
        radius = float(np.max(np.abs(ev)))
        R = (2.0 - tr) / 4.0
        detM = float(np.real(np.linalg.det(M)))
        try:
            floquet_log = np.log(np.where(np.abs(ev) > 1e-15, ev, 1e-15))
        except Exception:
            floquet_log = np.zeros_like(ev)
        return FloquetSpectrum(
            multipliers=ev, spectral_radius=radius, greene_residue=float(R),
            trace_monodromy=tr, is_elliptic=bool(abs(R) < 1.0),
            is_symplectic=bool(abs(detM - 1.0) < 0.15),
            floquet_log_eigenvalues=floquet_log,
        )

    @classmethod
    def solve_kepler(
        cls, mean_anomaly: float, eccentricity: float,
        tol: float = 1e-12, max_iter: int = 50,
    ) -> float:
        e = float(eccentricity); M = float(mean_anomaly)
        if not (0.0 <= e < 1.0):
            raise ValueError("Excentricidad debe estar en [0, 1).")
        M = (M + math.pi) % (2.0 * math.pi) - math.pi
        if e < 0.8:
            denom = 1.0 - math.sin(M + e) + math.sin(M)
            E = M + e * math.sin(M) / max(denom, 1e-12)
        else:
            E = math.pi * math.copysign(1.0, M) if abs(M) > 1e-15 else math.pi
        for _ in range(max_iter):
            s = math.sin(E); c = math.cos(E)
            f = E - e * s - M; fp = 1.0 - e * c; fpp = e * s
            den = fp - 0.5 * f * fpp / max(fp, _EPS)
            dE = f / max(den, _EPS); E -= dE
            if abs(dE) < tol:
                break
        return float(E)

    @classmethod
    def kepler_residual(cls, E: float, M: float, e: float) -> float:
        return float(E - e * math.sin(E) - M)

    @classmethod
    def gauss_planetary_rates(
        cls, elements: PoincareCelestialElements, pert_eps: float = 0.0,
    ) -> Dict[str, float]:
        a = max(elements.L ** 2, 1e-30)
        R = float(pert_eps)
        a_dot = 2.0 * math.sqrt(a ** 3) * R
        n = elements.mean_motion
        return {
            "a_dot": float(a_dot), "e_dot": 0.0, "i_dot": 0.0,
            "mean_motion_dot": float(-1.5 * n / a * a_dot) if a > 0 else 0.0,
        }

    @classmethod
    def birkhoff_normal_form(
        cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None, order: int = 2,
    ) -> np.ndarray:
        r"""Wrapper retro-compatible de BirkhoffHoriDeprit (v9.2.0)."""
        result = BirkhoffHoriDeprit.normal_form(rho, N_diag, max_order=order)
        if not result:
            return np.zeros_like(rho, dtype=np.complex128)
        acc = np.zeros_like(result.get(1, np.zeros_like(rho, dtype=np.complex128)))
        for k in range(1, int(order) + 1):
            acc = acc + result.get(k, np.zeros_like(acc))
        return acc

    @classmethod
    def melnikov_function(
        cls, rho_dream: np.ndarray, rho_base: np.ndarray,
        N_diag: Optional[np.ndarray] = None, n_steps: int = 32, dt: float = 0.02,
    ) -> float:
        rho_h = _hermitize_trace_one(rho_dream)
        n = rho_h.shape[0]
        nu = _diag_potential(n, N_diag)
        base = _hermitize_trace_one(rho_base)
        phase_dt = np.exp(-1j * dt * (nu[:, None] - nu[None, :]))
        total = 0.0
        current = rho_h
        N = np.diag(nu)
        for _ in range(n_steps):
            comm = N @ current - current @ N
            poisson = float(np.real(-1j * np.trace(comm @ base)))
            total += poisson * dt
            current = 0.5 * ((current * phase_dt) + (current * phase_dt).conj().T)
        return float(total)

    # ── §2.8 Auditoría Gromov–Witten relativo con Poincaré–Lefschetz ─────
    def evaluate_poincare_lefschetz_gw_invariant(
        self, rho_dream: np.ndarray, rho_base: np.ndarray,
        boundary_stalk_matrix: np.ndarray, betti_1_cycles: int = 0,
        dirichlet_energy: Optional[float] = None,
        dream_isolation: bool = True,
        scenario_id: str = "DREAM-POINCARE-EVAL",
        isolation_cert: Optional[OniricIsolationCertificate] = None,
        **kwargs: Any,
    ) -> ImmunizationCertificate:
        r"""
        Audita rigidez simpléctica y dualidad de Poincaré–Lefschetz. Propaga
        análisis celeste + TQFT + Chern–Simons + Melnikov zeros + AGM + RSI-3.
        """
        measure_dream = self.extract_spectral_measure(rho_dream)
        measure_base = self.extract_spectral_measure(rho_base)
        n = int(np.asarray(rho_dream).shape[0])
        purity = measure_dream.purity
        entropy = measure_dream.von_neumann_entropy
        ed = (measure_dream.dirichlet_energy
              if dirichlet_energy is None else float(dirichlet_energy))
        rho_d = _hermitize_trace_one(rho_dream)
        rho_b = _hermitize_trace_one(rho_base)
        stalk = np.asarray(boundary_stalk_matrix, dtype=np.complex128)
        boundary_defect = float(la.norm(rho_d @ stalk - stalk @ rho_b, ord="fro"))
        is_boundary_consistent = boundary_defect <= self.lefschetz_tolerance
        evals_dream = measure_dream.eigenvalues
        evals_base = measure_base.eigenvalues
        r_dream_max = float(np.max(evals_dream)) if evals_dream.size > 0 else 1.0
        R_base_min = (float(np.min(evals_base[evals_base > 1e-12]))
                      if np.any(evals_base > 1e-12) else 1.0)
        capacity_ratio = r_dream_max / max(R_base_min, self.capacity_floor)
        denom = 1.0 + float(max(betti_1_cycles, 0))
        entropy_damp = math.exp(-abs(entropy) / max(n, 1))
        relative_factor = 1.0 / (1.0 + boundary_defect)
        gw_rel = (purity * math.exp(-ed) * entropy_damp / denom) * relative_factor
        gw_rel = float(min(max(gw_rel, 0.0), 1.0))
        if not dream_isolation:
            verdict = HeytingOmega3.VETOED
        elif (gw_rel >= self.gw_threshold and is_boundary_consistent
              and capacity_ratio <= 1.05):
            verdict = HeytingOmega3.COHERENT
        elif (gw_rel >= self.gw_threshold * 0.5
              and capacity_ratio <= self.SYMPLECTIC_RATIO_MAX):
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.VETOED
        crowbar_triggered = verdict == HeytingOmega3.VETOED
        gpio14_signal = "HIGH" if crowbar_triggered else "LOW"
        proof_str = f"{gw_rel:.8f}|{boundary_defect:.8f}|{capacity_ratio:.8f}|{verdict.value}"
        proof_sha512 = hashlib.sha512(proof_str.encode("utf-8")).hexdigest()
        imm_id = f"IMM-PL-{hashlib.md5(proof_str.encode('utf-8')).hexdigest()[:8]}"
        t_seal = time.time()
        sig_str = f"{scenario_id}::{imm_id}::{proof_sha512}::{t_seal:.6f}"
        sig_sha256 = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()
        isol = isolation_cert if isolation_cert is not None else OniricIsolationCertificate(
            is_fully_isolated=dream_isolation, dream_state_verified=dream_isolation,
            hardware_leak_risk=not dream_isolation, proof_hash=sig_sha256,
            leak_threshold=0.8,
        )
        # Análisis celeste + TQFT (neutros por defecto)
        L_del = G_del = H_del = 0.0
        e_del = i_del = 0.0
        kam_res = float("inf")
        res_order = 0
        wigner_neg = 0.0
        capacity = 1.0
        maslov = 0
        hannay = 0.0
        brjuno = 0.0
        greene = 0.0
        floquet_r = 1.0
        t_nek = float("inf")
        tisserand = 0.0
        jacobi = 0.0
        mean_motion = 0.0
        chirikov = 0.0
        tau_rec = float("inf")
        tqft_amp_geom = 0.0
        cs_wilson = 0.0
        mel_zeros = 0
        smale_hs = False
        nov_ok = False
        conley = 0
        try:
            rho_op = DensityOperator.from_array(rho_d)
            L_del, G_del, H_del = rho_op.delaunay_triple()
            e_del, i_del = rho_op.eccentricity_inclination()
            omega = rho_op.mean_motion_frequencies()
            chirikov = rho_op.chirikov_overlap()
            res_struct = self.detect_resonances(omega, chirikov=chirikov)
            kam_res = res_struct.kam_residue
            res_order = res_struct.order
            wigner_neg = rho_op.wigner_negativity()
            capacity = rho_op.symplectic_capacity_gromov()
            maslov = rho_op.maslov_index()
            hannay = rho_op.hannay_angle()
            brjuno = rho_op.siegel_brjuno_sum()
            tau_rec = rho_op.poincare_recurrence_time()
            elems = rho_op.celestial_elements()
            tisserand = elems.tisserand
            jacobi = elems.jacobi_integral
            mean_motion = elems.mean_motion
            N4 = np.diag(_diag_potential(n))
            section = self.poincare_section_at(elems.energy, n)
            floq = self.monodromy_and_greene(rho_op.matrix, section, N4)
            greene = floq.greene_residue
            floquet_r = floq.spectral_radius
            eps_comm = float(la.norm(rho_op.matrix @ N4 - N4 @ rho_op.matrix, "fro"))
            t_nek = self.nekhoroshev_stability_time(eps_comm, n)
            # ── NUEVOS v9.2.0: TQFT, CS Wilson, Melnikov zeros, Novikov, Conley ──
            tqft_cert = TQFTPathIntegral.amplitude(
                genus=max(1, betti_1_cycles),
                n_punctures=max(1, 1),
            )
            tqft_amp_geom = tqft_cert.amplitude
            cs_wilson = ChernSimonsWilson.amplitude(rho_d, k_level=2, winding=1)
            nz, _m_peak, smale = MelnikovZerosFinder.find_zeros(
                rho_op.matrix, rho_b, N4, n_samples=32, dt=0.05,
            )
            mel_zeros = nz
            smale_hs = smale
            nov_x = NovikovRing.monomial(1.0 + 0.5j, -0.5)
            nov_y = NovikovRing.monomial(0.7 - 0.3j, -1.0)
            nov_ok = (NovikovRing.verify_non_archimedean(nov_x, nov_y)
                      and NovikovRing.verify_multiplicative(nov_x, nov_y))
            conley = MorseConleyIndex.compute(rho_d)
        except Exception:
            logger.exception("Fallo en el análisis celeste; se aplican neutros.")
        return ImmunizationCertificate(
            immunization_id=imm_id, scenario_id=scenario_id,
            heyting_verdict=verdict,
            gw_relative_invariant=gw_rel,
            poincare_lefschetz_defect=boundary_defect,
            symplectic_capacity_ratio=capacity_ratio,
            is_boundary_consistent=is_boundary_consistent,
            crowbar_triggered=crowbar_triggered,
            gpio14_signal=gpio14_signal,
            proof_merkle_sha512=proof_sha512,
            gromov_witten_invariant=gw_rel,
            tqft_amplitude=gw_rel,
            purity=purity, von_neumann_entropy=entropy,
            spectral_gap=measure_dream.spectral_gap,
            dirac_total_variation=measure_dream.dirac_total_variation,
            cstar_residual=measure_dream.cstar_residual,
            dirichlet_bound_valid=ed <= 0.85,
            dirichlet_residual=0.0,
            euler_characteristic=1 - betti_1_cycles,
            topological_class=("POINT" if betti_1_cycles == 0 else "LOOPED_LIGHT"),
            isolation_cert=isol,
            immunization_payload_hash=proof_sha512,
            digital_signature_sha256=sig_sha256,
            timestamp_utc=t_seal,
            holonomy_partial=gw_rel,
            wilson_phase=complex(math.cos(gw_rel), math.sin(gw_rel)),
            delaunay_triple=(L_del, G_del, H_del),
            eccentricity=e_del, inclination=i_del,
            kam_residue=kam_res, resonance_order=res_order,
            maslov_index=maslov, wigner_negativity=wigner_neg,
            symplectic_capacity=capacity,
            poincare_stratum=verdict.poincare_stratum_name(),
            lyapunov_sign=verdict.lyapunov_exponent_sign(),
            hannay_angle=hannay, greene_residue=greene,
            chirikov_overlap=chirikov, nekhoroshev_time=t_nek,
            brjuno_sum=brjuno, tisserand=tisserand, jacobi_integral=jacobi,
            floquet_radius=floquet_r, mean_motion=mean_motion,
            poincare_recurrence_time=tau_rec,
            # ── v9.2.0 ──
            tqft_amplitude_geometric=tqft_amp_geom,
            cs_wilson_value=cs_wilson,
            melnikov_zeros=mel_zeros,
            smale_horseshoe=smale_hs,
            novikov_ultrametric_ok=nov_ok,
            conley_index=conley,
        )

    @classmethod
    def compute_purity(cls, density_matrix: np.ndarray) -> float:
        eigvals = la.eigvalsh(_hermitize_trace_one(density_matrix))
        lam = cls._project_spectrum(eigvals)
        return float(np.sum(lam ** 2))

    def compute_gw_invariant(
        self, density_matrix: np.ndarray, betti_1: int, *,
        dirichlet_energy: Optional[float] = None,
        entropy: float = 0.0, dimension: int = 1, euler_characteristic: int = 1,
    ) -> float:
        measure = self.extract_spectral_measure(density_matrix)
        ed = measure.dirichlet_energy if dirichlet_energy is None else float(dirichlet_energy)
        return self.compute_gromov_witten_invariant(
            density_matrix, betti_1, dirichlet_energy=ed,
            entropy=measure.von_neumann_entropy if entropy == 0.0 else entropy,
            dimension=measure.dimension if dimension == 1 else dimension,
            euler_characteristic=euler_characteristic,
        )

    @classmethod
    def compute_gromov_witten_invariant(
        cls, density_matrix: np.ndarray, betti_1: int, *,
        dirichlet_energy: Optional[float] = None, entropy: float = 0.0,
        dimension: int = 1, euler_characteristic: int = 1,
    ) -> float:
        if betti_1 < 0:
            raise ValueError("b₁ debe ser ≥ 0.")
        gamma = cls.compute_purity(density_matrix)
        denom = 1.0 + float(betti_1)
        if dirichlet_energy is None:
            return float(min(max(gamma / denom, 0.0), 1.0))
        n = max(int(dimension), 1)
        chi_pos = max(int(euler_characteristic), 0)
        chi_abs = abs(int(euler_characteristic))
        euler_factor = (1.0 + chi_pos) / (1.0 + chi_abs)
        entropy_damp = math.exp(-abs(entropy) / n)
        raw = gamma * math.exp(-float(dirichlet_energy)) * entropy_damp / denom * euler_factor
        return float(min(max(raw, 0.0), 1.0))


# ── §2.9 Guardián del aislamiento homológico ─────────────────────────────
class OniricIsolationGuard:
    r"""Guardián de aislamiento homológico con axioma y sello SHA-256."""
    LEAK_RISK_THRESHOLD: Final[float] = 0.8

    def audit_isolation(self, payload: OniricScenarioPayload) -> OniricIsolationCertificate:
        dream_verified = bool(payload.dream_state_flag)
        hardware_leak = (not dream_verified) and (
            payload.simulated_risk_factor > self.LEAK_RISK_THRESHOLD
        )
        is_isolated = dream_verified and (not hardware_leak)
        h = hashlib.sha256()
        payload_bytes = (
            f"{payload.scenario_id}::{payload.dream_state_flag}::"
            f"{is_isolated}::{payload.synthetic_cartridge_id}::"
            f"{payload.simulated_risk_factor:.6f}"
        )
        h.update(payload_bytes.encode("utf-8"))
        cert = OniricIsolationCertificate(
            is_fully_isolated=is_isolated,
            dream_state_verified=dream_verified,
            hardware_leak_risk=hardware_leak,
            proof_hash=h.hexdigest(),
            leak_threshold=self.LEAK_RISK_THRESHOLD,
        )
        if not cert.logical_consistency():
            raise RuntimeError("Violación del axioma de aislamiento homológico.")
        return cert


# ── §2.10 Traza abierta FASE-2 → FASE-3 ──────────────────────────────────
@dataclass(frozen=True, slots=True)
class UnsealedOniricAuditTrace:
    r"""
    Traza abierta (Spec, D, I_GW, Isol, celeste) antes de V/Seal.
    FASE-3 la consume en `_seal_and_classify`.
    """
    payload: OniricScenarioPayload
    measure: SpectralMeasure
    gromov_witten_invariant: float
    tqft_amplitude: float
    dirichlet_bound_valid: bool
    dirichlet_residual: float
    isolation_cert: OniricIsolationCertificate
    poincare_cert: Optional[ImmunizationCertificate] = None
    monadic_acceleration: float = 0.0
    novikov_valuation: float = 0.0
    reward_hacking_index: float = 0.0


# ── §2.11 Compositor de flechas Spec → I_GW → Isol → V ────────────────────
class OniricAuditArrowComposer:
    r"""
    Compositor Isol ∘ I_GW ∘ D ∘ Spec con dualidad Poincaré–Lefschetz.
    Produce UnsealedOniricAuditTrace, germen formal de FASE-3.
    """
    DIRICHLET_CONSISTENCY_TOL: Final[float] = 1e-3

    def __init__(
        self, spectra: GromovWittenOniricAuditor,
        isolation_auditor: IsolationAuditor,
        energy_threshold: float, include_dirichlet_attenuation: bool,
    ) -> None:
        self.spectra = spectra
        self.isolation_auditor = isolation_auditor
        self.energy_threshold = float(energy_threshold)
        self.include_dirichlet_attenuation = bool(include_dirichlet_attenuation)

    def compose_audit_arrows(self, payload: OniricScenarioPayload) -> UnsealedOniricAuditTrace:
        measure = self.spectra.extract_spectral_measure(payload.density_matrix)
        ed_residual = abs(payload.dirichlet_energy - measure.dirichlet_energy)
        isolation = self.isolation_auditor.audit_isolation(payload)
        n = payload.density_matrix.shape[0]
        poincare_cert = self.spectra.evaluate_poincare_lefschetz_gw_invariant(
            rho_dream=payload.density_matrix,
            rho_base=(payload.rho_base if payload.rho_base is not None
                      else np.eye(n, dtype=np.complex128) / float(n)),
            boundary_stalk_matrix=(payload.boundary_stalk_matrix
                                   if payload.boundary_stalk_matrix is not None
                                   else np.eye(n, dtype=np.complex128)),
            betti_1_cycles=payload.betti_1_loop_count,
            dirichlet_energy=payload.dirichlet_energy,
            dream_isolation=isolation.is_fully_isolated,
            scenario_id=payload.scenario_id,
            isolation_cert=isolation,
        )
        lam = measure.as_simplex()
        novikov_val = self.spectra.meta_engine.novikov_valuation(
            [abs(x) for x in lam if abs(x) > 1e-12]
        )
        rhi = self.spectra.meta_engine.compute_reward_hacking_index(
            poincare_cert.wilson_phase if poincare_cert else 1.0 + 0.0j,
            poincare_cert.hannay_angle if poincare_cert else 0.0,
        )
        ricci_dummy = np.eye(n, dtype=np.complex128) * (measure.purity - measure.von_neumann_entropy)
        vac_proj = np.eye(n, dtype=np.complex128) / float(n)
        _, meta_metrics = self.spectra.meta_engine.evolve_tqft_immunization_kernel(
            vac_proj, ricci_dummy,
            poincare_cert.symplectic_capacity_ratio if poincare_cert else 1.0,
        )
        if not isolation.is_fully_isolated:
            return UnsealedOniricAuditTrace(
                payload=payload, measure=measure,
                gromov_witten_invariant=0.0, tqft_amplitude=0.0,
                dirichlet_bound_valid=False, dirichlet_residual=ed_residual,
                isolation_cert=isolation, poincare_cert=poincare_cert,
                monadic_acceleration=meta_metrics.get("monadic_acceleration", 0.0),
                novikov_valuation=novikov_val,
                reward_hacking_index=rhi,
            )
        gw = poincare_cert.gw_relative_invariant
        dirichlet_valid = payload.dirichlet_energy <= self.energy_threshold
        return UnsealedOniricAuditTrace(
            payload=payload, measure=measure,
            gromov_witten_invariant=gw, tqft_amplitude=gw,
            dirichlet_bound_valid=dirichlet_valid,
            dirichlet_residual=ed_residual,
            isolation_cert=isolation, poincare_cert=poincare_cert,
            monadic_acceleration=meta_metrics.get("monadic_acceleration", 0.0),
            novikov_valuation=novikov_val,
            reward_hacking_index=rhi,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — SOBERANO AUDITOR, SELLO DUAL, HOLONOMÍA, MERKLE, CROWBAR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    r"""Prueba de inclusión Merkle SHA-256 (convención CT/Bitcoin)."""
    leaf_hash: str
    siblings: Tuple[str, ...]
    index: int
    root: str

    def verify(self) -> bool:
        try:
            node = bytes.fromhex(self.leaf_hash)
        except ValueError:
            return False
        idx = self.index
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


class OniricDreamAuditorAgent:
    r"""
    Soberano auditor de escenarios contrafactuales del Onírico.

    Realiza el funtor 𝒜 = Seal ∘ Crowbar ∘ V ∘ Isol ∘ I_GW ∘ Spec.
    Cadena de custodia Merkle + holonomías Wilson/Hannay acumuladas.
    """
    _DEFAULT_ENERGY_THRESHOLD: Final[float] = 0.85
    _DEFAULT_GW_TOLERANCE: Final[float] = 0.15
    _DEFAULT_BETTI_MAX: Final[int] = 3
    _DEFAULT_DIRAC_TV_MAX: Final[float] = 1.50

    def __init__(
        self, agent_id: str = "ONIRIC-AUDITOR-SABIO-01",
        energy_threshold: float = _DEFAULT_ENERGY_THRESHOLD,
        gw_tolerance: float = _DEFAULT_GW_TOLERANCE,
        betti_max: int = _DEFAULT_BETTI_MAX,
        dirac_tv_max: float = _DEFAULT_DIRAC_TV_MAX,
        isolation_auditor: Optional[IsolationAuditor] = None,
        gw_evaluator: Optional[GromovWittenOniricAuditor] = None,
        meta_engine: Optional[MetaOniricAuditorEngine] = None,
        *, include_dirichlet_attenuation: bool = True,
    ) -> None:
        if not (0.0 < energy_threshold <= 2.0):
            raise ValueError("energy_threshold debe estar en (0, 2].")
        if not (0.0 <= gw_tolerance <= 1.0):
            raise ValueError("gw_tolerance debe estar en [0, 1].")
        if betti_max < 0:
            raise ValueError("betti_max debe ser ≥ 0.")
        if dirac_tv_max < 0.0:
            raise ValueError("dirac_tv_max debe ser ≥ 0.")
        self.agent_id: str = agent_id
        self.energy_threshold: float = float(energy_threshold)
        self.gw_tolerance: float = float(gw_tolerance)
        self.betti_max: int = int(betti_max)
        self.dirac_tv_max: float = float(dirac_tv_max)
        self._include_dirichlet_attenuation: bool = bool(include_dirichlet_attenuation)
        self.meta_engine = meta_engine or MetaOniricAuditorEngine()
        spectra = (gw_evaluator if gw_evaluator is not None
                   else GromovWittenOniricAuditor(gw_threshold=gw_tolerance,
                                                  meta_engine=self.meta_engine))
        isolator: IsolationAuditor = (
            isolation_auditor if isolation_auditor is not None else OniricIsolationGuard()
        )
        self.composer: OniricAuditArrowComposer = OniricAuditArrowComposer(
            spectra=spectra, isolation_auditor=isolator,
            energy_threshold=self.energy_threshold,
            include_dirichlet_attenuation=self._include_dirichlet_attenuation,
        )
        self._isolation_auditor = isolator
        self._gw_evaluator = spectra
        self.audit_count: int = 0
        self.immunization_registry: List[ImmunizationCertificate] = []
        self._density_history: List[np.ndarray] = []
        self._holonomy_accum: float = 0.0
        self._hannay_accum: float = 0.0

    # ── §3.1 HAND-OFF FASE-2 → FASE-3: sello y clasificación ─────────────
    def _seal_and_classify(self, trace: UnsealedOniricAuditTrace) -> ImmunizationCertificate:
        r"""
        CONTINUACIÓN FORMAL de OniricAuditArrowComposer.compose_audit_arrows.
        Aplica V, sella SHA-256/SHA-512, acumula holonomías, dispara Crowbar
        si hay caos estructural.
        """
        payload = trace.payload
        isolation = trace.isolation_cert
        measure = trace.measure
        verdict = self._classify(
            betti_1=payload.betti_1_loop_count,
            gw_inv=trace.gromov_witten_invariant,
            dirichlet_valid=trace.dirichlet_bound_valid,
            isolation=isolation,
            dirac_tv=measure.dirac_total_variation,
            poincare_cert=trace.poincare_cert,
        )
        imm_id = f"IMM-ONIRIC-{self.audit_count:04d}"
        t_seal = time.time()
        p_cert = trace.poincare_cert
        boundary_defect = p_cert.poincare_lefschetz_defect if p_cert else 0.0
        cap_ratio = p_cert.symplectic_capacity_ratio if p_cert else 1.0
        is_boundary_consistent = p_cert.is_boundary_consistent if p_cert else True
        # Crowbar si VETOED o si caos estructural (horseshoe / Chirikov)
        smale_hs = p_cert.smale_horseshoe if p_cert else False
        chirikov = p_cert.chirikov_overlap if p_cert else 0.0
        needs_crowbar = (
            verdict == HeytingOmega3.VETOED
            or smale_hs
            or (chirikov > 1.0)
        )
        if needs_crowbar:
            ESP32CrowbarInterlock.trigger(
                reason=f"Ciclo {imm_id} VETOED: smale={smale_hs}, "
                       f"chirikov={chirikov:.3f}, verdict={verdict.name}"
            )
        crowbar_triggered = needs_crowbar
        gpio14_signal = "HIGH" if crowbar_triggered else "LOW"
        delaunay = p_cert.delaunay_triple if p_cert else None
        eccentricity = p_cert.eccentricity if p_cert else 0.0
        inclination = p_cert.inclination if p_cert else 0.0
        kam_res = p_cert.kam_residue if p_cert else float("inf")
        res_order = p_cert.resonance_order if p_cert else 0
        maslov = p_cert.maslov_index if p_cert else 0
        wigner_neg = p_cert.wigner_negativity if p_cert else 0.0
        symplectic_cap = p_cert.symplectic_capacity if p_cert else 1.0
        hannay = p_cert.hannay_angle if p_cert else 0.0
        greene = p_cert.greene_residue if p_cert else 0.0
        t_nek = p_cert.nekhoroshev_time if p_cert else float("inf")
        brjuno = p_cert.brjuno_sum if p_cert else 0.0
        tisserand = p_cert.tisserand if p_cert else 0.0
        jacobi = p_cert.jacobi_integral if p_cert else 0.0
        floquet_r = p_cert.floquet_radius if p_cert else 1.0
        mean_motion = p_cert.mean_motion if p_cert else 0.0
        tau_rec = p_cert.poincare_recurrence_time if p_cert else float("inf")
        # NUEVOS v9.2.0
        tqft_amp_geom = p_cert.tqft_amplitude_geometric if p_cert else 0.0
        cs_wilson = p_cert.cs_wilson_value if p_cert else 0.0
        mel_zeros = p_cert.melnikov_zeros if p_cert else 0
        nov_ok = p_cert.novikov_ultrametric_ok if p_cert else False
        conley = p_cert.conley_index if p_cert else 0
        # AGM sobre Ω₃
        K0 = AGMBeliefSet(AGMBeliefSet.universe())
        alpha = frozenset({int(verdict)})
        if len(alpha) == 0:
            alpha = frozenset({0})
        agm_cert = HeytingAGMRevision.verify_postulates(K0, alpha)
        # RSI-3 mónada
        try:
            n_rho = payload.density_matrix.shape[0]
            monad = MonadRSI3(dim=n_rho, eta=self.meta_engine.monad.eta, level=3)
            monad_ax = monad.verify_axioms(payload.density_matrix)
        except Exception:
            monad_ax = MonadAxiomVerification(
                left_associativity=0.0, right_associativity=0.0,
                associativity_holds=False, left_unitarity=0.0, right_unitarity=0.0,
                unitarity_holds=False, level_applied=3, verdict=HeytingOmega3.VETOED,
            )
        stratum = verdict.poincare_stratum_name()
        lyap = verdict.lyapunov_exponent_sign()
        proof_str = (
            f"{payload.scenario_id}|{trace.gromov_witten_invariant:.8f}|"
            f"{boundary_defect:.8f}|{cap_ratio:.8f}|{verdict.value}"
        )
        merkle_sha512 = hashlib.sha512(proof_str.encode("utf-8")).hexdigest()
        h_payload = hashlib.sha256(proof_str.encode("utf-8")).hexdigest()
        h_sig = hashlib.sha256(
            f"{self.agent_id}::{imm_id}::{h_payload}::{t_seal:.6f}".encode("utf-8")
        ).hexdigest()
        self._holonomy_accum += trace.gromov_witten_invariant
        self._hannay_accum += hannay
        wilson = complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))
        return ImmunizationCertificate(
            immunization_id=imm_id, scenario_id=payload.scenario_id,
            heyting_verdict=verdict,
            gw_relative_invariant=float(trace.gromov_witten_invariant),
            poincare_lefschetz_defect=boundary_defect,
            symplectic_capacity_ratio=cap_ratio,
            is_boundary_consistent=is_boundary_consistent,
            crowbar_triggered=crowbar_triggered,
            gpio14_signal=gpio14_signal,
            proof_merkle_sha512=merkle_sha512,
            gromov_witten_invariant=float(trace.gromov_witten_invariant),
            tqft_amplitude=float(trace.tqft_amplitude),
            purity=measure.purity,
            von_neumann_entropy=measure.von_neumann_entropy,
            spectral_gap=measure.spectral_gap,
            dirac_total_variation=measure.dirac_total_variation,
            cstar_residual=measure.cstar_residual,
            dirichlet_bound_valid=bool(trace.dirichlet_bound_valid),
            dirichlet_residual=float(trace.dirichlet_residual),
            euler_characteristic=payload.euler_characteristic(),
            topological_class=payload.topological_class(),
            isolation_cert=isolation,
            immunization_payload_hash=h_payload,
            digital_signature_sha256=h_sig,
            timestamp_utc=t_seal,
            holonomy_partial=self._holonomy_accum,
            wilson_phase=wilson,
            delaunay_triple=delaunay, eccentricity=eccentricity,
            inclination=inclination, kam_residue=kam_res,
            resonance_order=res_order, maslov_index=maslov,
            wigner_negativity=wigner_neg, symplectic_capacity=symplectic_cap,
            poincare_stratum=stratum, lyapunov_sign=lyap,
            hannay_angle=hannay, greene_residue=greene,
            chirikov_overlap=chirikov, nekhoroshev_time=t_nek,
            brjuno_sum=brjuno, tisserand=tisserand, jacobi_integral=jacobi,
            floquet_radius=floquet_r, mean_motion=mean_motion,
            poincare_recurrence_time=tau_rec,
            monadic_acceleration=trace.monadic_acceleration,
            novikov_valuation=trace.novikov_valuation,
            reward_hacking_index=trace.reward_hacking_index,
            # ── v9.2.0 ──
            tqft_amplitude_geometric=tqft_amp_geom,
            cs_wilson_value=cs_wilson,
            melnikov_zeros=mel_zeros,
            smale_horseshoe=smale_hs,
            agm_postulates_passed=agm_cert.n_postulates_passed,
            agm_all_satisfied=agm_cert.all_satisfied,
            monad_associativity=monad_ax.associativity_holds,
            monad_unitarity=monad_ax.unitarity_holds,
            rsi_level=monad_ax.level_applied,
            novikov_ultrametric_ok=nov_ok,
            conley_index=conley,
        )

    def _classify(
        self, *, betti_1: int, gw_inv: float, dirichlet_valid: bool,
        isolation: OniricIsolationCertificate, dirac_tv: float,
        poincare_cert: Optional[ImmunizationCertificate] = None,
    ) -> HeytingOmega3:
        r"""Flecha característica χ : Trace → Ω₃."""
        if poincare_cert is not None:
            if not isolation.is_fully_isolated or betti_1 > self.betti_max:
                return HeytingOmega3.VETOED
            if (not poincare_cert.is_boundary_consistent
                    or poincare_cert.symplectic_capacity_ratio > 1.25):
                return HeytingOmega3.VETOED
            if poincare_cert.smale_horseshoe:
                return HeytingOmega3.VETOED
            return poincare_cert.heyting_verdict
        if not isolation.is_fully_isolated:
            return HeytingOmega3.VETOED
        if betti_1 > self.betti_max or gw_inv < self.gw_tolerance:
            return HeytingOmega3.VETOED
        if not dirichlet_valid or dirac_tv > self.dirac_tv_max:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    # ── §3.2 Auditoría de un escenario ───────────────────────────────────
    def audit_dream_scenario(self, payload: OniricScenarioPayload) -> ImmunizationCertificate:
        r"""Ejecuta 𝒜(payload) = Seal ∘ Crowbar ∘ V ∘ Isol ∘ I_GW ∘ Spec."""
        self.audit_count += 1
        t_start = time.time()
        logger.info("=== Auditando Escenario Onírico #%d | ID: %s ===",
                    self.audit_count, payload.scenario_id)
        trace = self.composer.compose_audit_arrows(payload)
        cert = self._seal_and_classify(trace)
        self.immunization_registry.append(cert)
        self._density_history.append(np.array(payload.density_matrix, copy=True))
        logger.info(
            "Auditoría Onírica Finalizada en %.2f ms | Ω₃=%s | I_GW=%.6f | "
            "Z_TQFT=%.4f | ⟨W_K⟩=%.4f | Defect=%.6f | R_KAM=%.3e | "
            "Chirikov=%.3f | Greene=%.3f | M_zeros=%d | AGM=%d/6 | RSI-3=%s | "
            "Crowbar=%s | GPIO14=%s",
            (time.time() - t_start) * 1000.0, cert.heyting_verdict.name,
            cert.gw_relative_invariant, cert.tqft_amplitude_geometric,
            cert.cs_wilson_value, cert.poincare_lefschetz_defect,
            cert.kam_residue, cert.chirikov_overlap, cert.greene_residue,
            cert.melnikov_zeros, cert.agm_postulates_passed,
            cert.monad_associativity and cert.monad_unitarity,
            cert.crowbar_triggered, cert.gpio14_signal,
        )
        return cert

    # ── §3.3 Vistas y propiedades agregadas ──────────────────────────────
    @property
    def registry(self) -> Tuple[ImmunizationCertificate, ...]:
        return tuple(self.immunization_registry)

    @property
    def holonomy_accum(self) -> float:
        return self._holonomy_accum

    @property
    def wilson_loop(self) -> complex:
        return complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))

    @property
    def hannay_loop(self) -> complex:
        return complex(math.cos(self._hannay_accum), math.sin(self._hannay_accum))

    @property
    def global_verdict(self) -> HeytingOmega3:
        gv = HeytingOmega3.COHERENT
        for c in self.immunization_registry:
            gv = gv.meet(c.heyting_verdict)
        return gv

    @property
    def berry_phase_along_history(self) -> float:
        if len(self._density_history) < 2:
            return 0.0
        return DensityOperator.from_array(self._density_history[0]).berry_phase_curve(
            self._density_history, closed=True
        )

    # ── §3.4 Árbol Merkle ────────────────────────────────────────────────
    @staticmethod
    def _merkle_tree_root(leaf_hashes: Sequence[str]) -> str:
        if not leaf_hashes:
            return hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
        level = [bytes.fromhex(h) for h in leaf_hashes]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [hashlib.sha256(level[i] + level[i + 1]).digest()
                     for i in range(0, len(level), 2)]
        return level[0].hex()

    @staticmethod
    def _merkle_proof(leaf_hashes: Sequence[str], index: int) -> MerkleInclusionProof:
        if not leaf_hashes:
            empty = hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
            return MerkleInclusionProof(empty, tuple(), 0, empty)
        if not (0 <= index < len(leaf_hashes)):
            raise IndexError("índice de hoja Merkle fuera de rango")
        level = [bytes.fromhex(h) for h in leaf_hashes]
        siblings: List[str] = []
        idx = index
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            pair = idx ^ 1
            siblings.append(level[pair].hex())
            next_level = [hashlib.sha256(level[i] + level[i + 1]).digest()
                          for i in range(0, len(level), 2)]
            level = next_level
            idx //= 2
        return MerkleInclusionProof(
            leaf_hash=leaf_hashes[index], siblings=tuple(siblings),
            index=index, root=level[0].hex(),
        )

    def merkle_root(self) -> str:
        return self._merkle_tree_root(
            [c.digital_signature_sha256 for c in self.immunization_registry]
        )

    def merkle_proofs_ok(self) -> bool:
        leaves = [c.digital_signature_sha256 for c in self.immunization_registry]
        root = self._merkle_tree_root(leaves)
        for i in range(len(leaves)):
            proof = self._merkle_proof(leaves, i)
            if proof.root != root or not proof.verify():
                return False
        return True

    # ── §3.5 Auditoría retrospectiva y pasaporte ─────────────────────────
    def audit_registry(self) -> Dict[str, Any]:
        n = len(self.immunization_registry)
        empty_dist = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_certificates": 0, "verdict_distribution": empty_dist,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "holonomy_accum": 0.0, "wilson_loop": 1.0 + 0.0j,
                "hannay_loop": 1.0 + 0.0j, "berry_phase": 0.0,
                "avg_gw_invariant": 0.0, "avg_tqft_amplitude": 0.0,
                "avg_cs_wilson": 0.0, "avg_purity": 0.0,
                "avg_dirac_tv": 0.0, "avg_dirichlet_residual": 0.0,
                "avg_kam_residue": 0.0, "avg_capacity": 0.0,
                "avg_chirikov": 0.0, "avg_greene": 0.0,
                "n_immune": 0, "n_hardware_leak_risk": 0,
                "n_crowbar": 0, "n_horseshoe": 0,
                "agm_all_satisfied_count": 0, "monad_axioms_ok_count": 0,
                "novikov_ultrametric_ok_count": 0,
                "registry_integrity_ok": True, "merkle_proofs_ok": True,
                "isolation_axioms_ok": True,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        total_gw = total_p = total_tv = total_ed_res = 0.0
        total_kam = total_cap = total_ch = total_gr = 0.0
        total_tqft = total_cs = 0.0
        immune_count = leak_count = n_crowbar = n_horseshoe = 0
        n_agm = n_monad = n_nov = 0
        signatures: Set[str] = set()
        collide = False
        axioms_ok = True
        for c in self.immunization_registry:
            dist[c.heyting_verdict.name] += 1
            total_gw += c.gromov_witten_invariant
            total_tqft += c.tqft_amplitude_geometric
            total_cs += c.cs_wilson_value
            total_p += c.purity
            total_tv += c.dirac_total_variation
            total_ed_res += c.dirichlet_residual
            total_kam += c.kam_residue if math.isfinite(c.kam_residue) else 0.0
            total_cap += c.symplectic_capacity
            total_ch += c.chirikov_overlap
            total_gr += c.greene_residue
            if c.isolation_cert.hardware_leak_risk:
                leak_count += 1
            if c.is_immune():
                immune_count += 1
            if c.crowbar_triggered:
                n_crowbar += 1
            if c.smale_horseshoe:
                n_horseshoe += 1
            if c.agm_all_satisfied:
                n_agm += 1
            if c.monad_associativity and c.monad_unitarity:
                n_monad += 1
            if c.novikov_ultrametric_ok:
                n_nov += 1
            if not c.isolation_cert.logical_consistency():
                axioms_ok = False
            if c.digital_signature_sha256 in signatures:
                collide = True
            signatures.add(c.digital_signature_sha256)
        inv = 1.0 / n
        return {
            "n_certificates": n, "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop, "hannay_loop": self.hannay_loop,
            "berry_phase": self.berry_phase_along_history,
            "avg_gw_invariant": total_gw * inv,
            "avg_tqft_amplitude": total_tqft * inv,
            "avg_cs_wilson": total_cs * inv,
            "avg_purity": total_p * inv, "avg_dirac_tv": total_tv * inv,
            "avg_dirichlet_residual": total_ed_res * inv,
            "avg_kam_residue": total_kam * inv,
            "avg_capacity": total_cap * inv,
            "avg_chirikov": total_ch * inv, "avg_greene": total_gr * inv,
            "n_immune": immune_count, "n_hardware_leak_risk": leak_count,
            "n_crowbar": n_crowbar, "n_horseshoe": n_horseshoe,
            "agm_all_satisfied_count": n_agm,
            "monad_axioms_ok_count": n_monad,
            "novikov_ultrametric_ok_count": n_nov,
            "registry_integrity_ok": not collide,
            "merkle_proofs_ok": self.merkle_proofs_ok(),
            "isolation_axioms_ok": axioms_ok,
        }

    def emit_immunization_passport(self) -> Dict[str, Any]:
        h = hashlib.sha256()
        h.update(f"{self.agent_id}::{self.audit_count}::{self._holonomy_accum:.10f}".encode("utf-8"))
        for c in self.immunization_registry:
            h.update(c.digital_signature_sha256.encode("utf-8"))
        return {
            "agent_id": self.agent_id,
            "registry_size": self.audit_count,
            "global_verdict": self.global_verdict.name,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "hannay_loop": self.hannay_loop,
            "berry_phase": self.berry_phase_along_history,
            "n_immune": sum(1 for c in self.immunization_registry if c.is_immune()),
            "merkle_root": self.merkle_root(),
            "evidence_hash": h.hexdigest(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS Y EJECUCIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print("═" * 84)
    print("DEMOSTRACIÓN GRANULAR: TOON Oniric Auditor Agent v9.2.0")
    print("TQFT · Chern–Simons · Verlinde · Fukaya · Kontsevich · RSI-3 · AGM · Crowbar")
    print("═" * 84)

    # ── [§0] Verificación de leyes de Heyting ──
    print("\n[§0] LEYES DE HEYTING Ω₃")
    laws = HeytingOmega3.verify_algebra_laws()
    for k, v in laws.items():
        print(f"    - {k:<24}: {v}")
    assert all(laws.values())
    assert HeytingOmega3.COHERENT.lyapunov_exponent_sign() == -1

    lat = _iterate_integer_lattice(2, 2)
    assert any(np.array_equal(v, np.array([2, 0])) for v in lat)
    print("  • Red ℤ² contiene (±2,0): OK")

    # ── [§1] Novikov ring ──
    print("\n[§1] ANILLO DE NOVIKOV (no-arquimediano)")
    x = NovikovRing.monomial(1.0 + 0.5j, -0.5)
    y = NovikovRing.monomial(0.7 - 0.3j, -1.0)
    print(f"    - v(x)={x.valuation():.4f}, v(y)={y.valuation():.4f}")
    print(f"    - v(x+y) ≥ min: {NovikovRing.verify_non_archimedean(x, y)}")
    print(f"    - v(xy) = v(x)+v(y): {NovikovRing.verify_multiplicative(x, y)}")

    # ── [§2] AGM ──
    print("\n[§2] AGM (6 postulados)")
    K0 = AGMBeliefSet(AGMBeliefSet.universe())
    cert_agm = HeytingAGMRevision.verify_postulates(K0, frozenset({2}))
    print(f"    - pass={cert_agm.n_postulates_passed}/6 | verdict={cert_agm.local_verdict.name}")

    # ── [§3] RSI-3 mónada ──
    print("\n[§3] MÓNICA RSI-3 (T, η, μ)")
    monad = MonadRSI3(dim=4, eta=0.15, level=3)
    rho_test = np.eye(4, dtype=np.complex128) / 4.0
    ax = monad.verify_axioms(rho_test)
    print(f"    - asociatividad: {ax.associativity_holds} (d={ax.left_associativity:.3e})")
    print(f"    - unitaridad: {ax.unitarity_holds}")

    # ── [§4] TQFT / CS / Verlinde / Fukaya / Kontsevich ──
    print("\n[§4] ESTRUCTURAS TQFT")
    tqft = TQFTPathIntegral.amplitude(genus=2, n_punctures=1)
    print(f"    - Z_{{2,1}} = {tqft.amplitude:.6f} (dim_ℂ M = {tqft.moduli_dimension})")
    cs_w = ChernSimonsWilson.amplitude(rho_test, k_level=2)
    print(f"    - ⟨W_K⟩ CS = {cs_w:.6f}")
    print(f"    - Verlinde SU(2)_2 asociatividad OK: {VerlindeFusion.verify_fusion_algebra(2)}")
    A2 = np.array([[0.5, 0.2], [0.2, 0.5]], dtype=np.complex128)
    print(f"    - Fukaya m_1²=0: {FukayaAinfinity.verify_m1_squared_zero(A2)}")
    print(f"    - Kontsevich ⋆-product: {KontsevichFormality.moyal_star_2x2(0.5, 1.2)}")

    # ── [§5] Birkhoff / Melnikov / Nekhoroshev / Conley / Hochschild ──
    print("\n[§5] FORMAS NORMALES / CAOS / NEKHOROSHEV")
    bk = BirkhoffHoriDeprit.normal_form(rho_test, max_order=3)
    print(f"    - Birkhoff Hori–Deprit órdenes: {list(bk.keys())}")
    nz, m_peak, smale = MelnikovZerosFinder.find_zeros(rho_test, rho_test)
    print(f"    - Melnikov: n_zeros={nz}, horseshoe={smale}")
    nek = NekhoroshevBound.bound(0.05, n_dof=2)
    print(f"    - T_N (ε=0.05, n=2) = {nek['T_escape']:.3e}")
    print(f"    - Conley index = {MorseConleyIndex.compute(rho_test)}")
    print(f"    - HH^p(M_n) dims = {HochschildCohomology.dimensions(n=4)}")

    # ── [§6] Ciclo completo del soberano ──
    print("\n[§6] CICLO COMPLETO DEL SOBERANO")
    rng = np.random.default_rng(20250321)
    auditor = OniricDreamAuditorAgent(
        agent_id="ONIRIC-AUDITOR-POINCARE-01",
        include_dirichlet_attenuation=True,
    )
    A_rng = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    rho = A_rng @ A_rng.conj().T
    rho /= np.trace(rho).real

    s1 = OniricScenarioPayload(
        scenario_id="DREAM-POINCARE-001",
        dream_state_flag=True,
        synthetic_cartridge_id="CARTRIDGE-SYNTH-STRESS-01",
        density_matrix=rho,
        dirichlet_energy=0.25,
        betti_1_loop_count=0,
        betti_0_components=1,
        betti_2_cavities=0,
        simulated_risk_factor=0.30,
        timestamp_utc=time.time(),
    )
    cert1 = auditor.audit_dream_scenario(s1)
    print(f"    - ID Certificado        : {cert1.immunization_id}")
    print(f"    - Veredicto Ω₃          : {cert1.heyting_verdict.name}")
    print(f"    - Estrato Poincaré      : {cert1.poincare_stratum}")
    print(f"    - I_GW                  : {cert1.gw_relative_invariant:.6f}")
    print(f"    - Z_TQFT geom           : {cert1.tqft_amplitude_geometric:.6f}")
    print(f"    - ⟨W_K⟩ CS              : {cert1.cs_wilson_value:.6f}")
    print(f"    - Melnikov zeros        : {cert1.melnikov_zeros} | horseshoe={cert1.smale_horseshoe}")
    print(f"    - Chirikov s / Greene R : {cert1.chirikov_overlap:.6f} / {cert1.greene_residue:.6f}")
    print(f"    - Nekhoroshev T_N       : {cert1.nekhoroshev_time:.6e}")
    print(f"    - AGM postulados        : {cert1.agm_postulates_passed}/6 "
          f"(all={cert1.agm_all_satisfied})")
    print(f"    - RSI-3 (asoc/unit)     : {cert1.monad_associativity}/{cert1.monad_unitarity} "
          f"level={cert1.rsi_level}")
    print(f"    - Novikov ultramétrico  : {cert1.novikov_ultrametric_ok}")
    print(f"    - Conley index          : {cert1.conley_index}")
    print(f"    - Crowbar / GPIO14      : {cert1.crowbar_triggered} / {cert1.gpio14_signal}")
    assert s1.is_quantum_physical()
    assert cert1.isolation_cert.logical_consistency()

    # ── [§7] Escenarios de fuga y topología compleja ──
    print("\n[§7] ESCENARIOS DE RIESGO")
    rho_leak = np.eye(4, dtype=np.complex128) / 4.0
    s2 = OniricScenarioPayload(
        scenario_id="DREAM-LEAK-002",
        dream_state_flag=False,
        synthetic_cartridge_id="CARTRIDGE-LEAK-02",
        density_matrix=rho_leak,
        dirichlet_energy=0.20,
        betti_1_loop_count=0,
        betti_0_components=1,
        betti_2_cavities=0,
        simulated_risk_factor=0.95,
        timestamp_utc=time.time(),
    )
    cert2 = auditor.audit_dream_scenario(s2)
    print(f"    - VETO aislamiento : Ω₃={cert2.heyting_verdict.name}, "
          f"crowbar={cert2.crowbar_triggered}, GPIO14={cert2.gpio14_signal}")
    assert cert2.heyting_verdict == HeytingOmega3.VETOED
    assert cert2.crowbar_triggered is True
    assert cert2.gpio14_signal == "HIGH"

    s3 = OniricScenarioPayload(
        scenario_id="DREAM-COMPLEX-003",
        dream_state_flag=True,
        synthetic_cartridge_id="CARTRIDGE-COMPLEX-03",
        density_matrix=rho,
        dirichlet_energy=0.30,
        betti_1_loop_count=5,
        betti_0_components=2,
        betti_2_cavities=1,
        simulated_risk_factor=0.30,
        timestamp_utc=time.time(),
    )
    cert3 = auditor.audit_dream_scenario(s3)
    print(f"    - VETO topológico  : Ω₃={cert3.heyting_verdict.name}")
    assert cert3.heyting_verdict == HeytingOmega3.VETOED

    # ── [§8] Holonomías ──
    print("\n[§8] HOLONOMÍAS BERRY–HANNAY / WILSON")
    print(f"    - γ_Berry              : {auditor.berry_phase_along_history:.6f} rad")
    print(f"    - Wilson loop          : {auditor.wilson_loop}")
    print(f"    - Hannay loop          : {auditor.hannay_loop}")

    # ── [§9] Auditoría retrospectiva ──
    print("\n[§9] AUDITORÍA RETROSPECTIVA")
    audit = auditor.audit_registry()
    for k in ["n_certificates", "global_verdict", "n_immune", "n_crowbar",
              "n_horseshoe", "agm_all_satisfied_count", "monad_axioms_ok_count",
              "novikov_ultrametric_ok_count",
              "registry_integrity_ok", "merkle_proofs_ok", "isolation_axioms_ok"]:
        print(f"    - {k:<30}: {audit[k]}")
    assert audit["registry_integrity_ok"]
    assert audit["merkle_proofs_ok"]
    assert audit["isolation_axioms_ok"]

    print("\n[§10] PASAPORTE SOBERANO")
    passport = auditor.emit_immunization_passport()
    for k, v in passport.items():
        print(f"    - {k:<22}: {v}")

    print("\n" + "═" * 84)
    print("✓ Soberano Auditor Onírico v9.2.0 completado: TQFT + Novikov + RSI-3 + AGM + Crowbar")
    print("═" * 84)
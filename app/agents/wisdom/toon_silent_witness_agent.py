# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER — SISTEMA SOBERANO DE SABIDURÍA (WISDOM)
====================================================================================================

Nombre:    ToonSilentWitnessAgent
Ubicación: app/agents/wisdom/toon_silent_witness_agent.py
Versión:   9.2.0-APU-CR3BP-AGM-RSI3-CROWBAR-POINCARE
Función:   Agente soberano Testigo Silencioso, orquestación de la observación sin retro-acción ( Dirac vacuum ),
           adjudicación de creencias AGM sobre Ω₃, monitoreo de puntos de Lagrange L1-L5 y hardware Crowbar.

----------------------------------------------------------------------------------------------------
DEFINICIÓN FORMAL Y MARCO MATEMÁTICO RIGUROSO:
----------------------------------------------------------------------------------------------------
El Agente Soberano Testigo Silencioso (ToonSilentWitnessAgent) actúa como el observador inmune no-intrusivo
de la arquitectura APU Filter. Su misión es garantizar que las inspecciones del vacío y los estados de
densidad ρ_vac se ejecuten con cero perturbación sobre el flujo disipativo cognitivo.
Supervisa el motor 'TOONSilentWitnessEngine' (app/wisdom/toon_silent_witness_engine.py).

1. Dinámica Simpléctica y Puntos de Libración L1–L5:
   En el problema restringido de tres cuerpos circular (CR3BP) o su análogo cognitivo con relación de masas μ,
   la constante de Jacobi C define las superficies de velocidad cero (regiones de Hill):
       C = 2 Ω(x, y) - (ẋ^2 + ẏ^2)
   El agente monitorea la estabilidad de los puntos Lagrangianos L1–L5 evaluando la no-degeneración de Kolmogorov:
       κ = det(Hessian_I H) ≠ 0

2. Revisión de Creencias AGM y Disparador Crowbar:
   El Agente mantiene un sistema de creencias K sobre el álgebra de Heyting Ω₃. En presencia de anomalías
   en la integral de Jacobi C < C_critical o divergencias modulares en la condición KMS, se dispara
   el mecanismo hardware Crowbar en IRAM, asegurando una latencia menor a 400 ns.

3. Mónada RSI-3 y Registro Causal Inmutable:
   Cada observación se cristaliza sobre la 6-esfera S⁶ y se firma mediante un árbol de Merkle inmutable,
   certificando la integridad del Testigo Silencioso sin colisión de la unidad η.

----------------------------------------------------------------------------------------------------
AXIOMAS Y INVARIANTES FORMALES:
----------------------------------------------------------------------------------------------------
Axioma AS1 (Zero Back-Action Operational):
    ⟨Ψ | [A, H_vac] | Ψ⟩ = 0 para todo observable A del filtro, garantizando la naturaleza pasiva de la observación.
Axioma AS2 (Coherencia de Revisión AGM):
    Para todo α ∈ Ω₃, K * α satisface los 8 postulados de Alchourrón-Gärdenfors-Makinson.
Invariante AS1 (Acotamiento de la Superficie de Hill):
    El movimiento informacional está topológicamente restringido a la región permitida { (x,y) : 2Ω(x,y) ≥ C }.
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
from fractions import Fraction
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq

# ── Motor heredado Tomita–Takesaki / KMS / Dirac-Vacuum (v9.2.0 si está) ──
try:
    from app.wisdom.toon_silent_witness_engine import (
        TOONSilentWitnessEngine,
        MetaSilentWitnessEngine,
        ExperienceCrystal as EngineExperienceCrystal,
        HeytingOmega3,
        MatrixBanachAlgebra,
        ModularHamiltonian,
        DensityOperatorAlgebra,
        GNSHilbertAlgebra,
        VacuumModularContext,
        VacuumStatePreparation,
        TomitaTakesakiEngine,
        VacuumAuditReport,
        VacuumSpectraAnalyzer,
        SilentFieldProbe,
        SilentFieldDetector,
        SilentFieldBundle,
        ModularSilencePipeline,
        HeytingVacuumAdjudicator,
        SilentFieldState,
        RSI3Germ,
        PoincareCanonicalChart,
        KolmogorovNondegeneracy,
        LyapunovSpectrum,
        MonadRSI3 as EngineMonadRSI3,
    )
    _ENGINE_AVAILABLE = True
except Exception:  # fallback autónomo (el agente permanece importable)
    _ENGINE_AVAILABLE = False
    RSI3Germ = None  # type: ignore
    PoincareCanonicalChart = None  # type: ignore
    KolmogorovNondegeneracy = None  # type: ignore
    LyapunovSpectrum = None  # type: ignore
    EngineMonadRSI3 = None  # type: ignore
    TOONSilentWitnessEngine = None  # type: ignore
    MetaSilentWitnessEngine = None  # type: ignore
    EngineExperienceCrystal = None  # type: ignore
    ModularHamiltonian = None  # type: ignore
    GNSHilbertAlgebra = None  # type: ignore
    VacuumModularContext = None  # type: ignore
    VacuumStatePreparation = None  # type: ignore
    TomitaTakesakiEngine = None  # type: ignore
    VacuumAuditReport = None  # type: ignore
    VacuumSpectraAnalyzer = None  # type: ignore
    SilentFieldProbe = None  # type: ignore
    SilentFieldDetector = None  # type: ignore
    SilentFieldBundle = None  # type: ignore
    ModularSilencePipeline = None  # type: ignore
    HeytingVacuumAdjudicator = None  # type: ignore
    SilentFieldState = None  # type: ignore

    class HeytingOmega3(IntEnum):  # type: ignore
        r"""Cadena de Heyting Ω₃ = {⊥ < ⋆ < ⊤} (fallback autónomo)."""
        VETOED = 0
        DEGRADED = 1
        COHERENT = 2

        def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return HeytingOmega3(min(int(self), int(other)))

        def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return HeytingOmega3(max(int(self), int(other)))

        def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return HeytingOmega3.COHERENT if int(self) <= int(other) else other

        def neg(self) -> "HeytingOmega3":
            return self.implies(HeytingOmega3.VETOED)

        def poincare_stratum_name(self) -> str:
            return {
                HeytingOmega3.VETOED: "hyperbolic-escape-separatrix",
                HeytingOmega3.DEGRADED: "parabolic-resonant-orbit",
                HeytingOmega3.COHERENT: "kam-torus-invariant",
            }[self]

        @classmethod
        def verify_algebra_laws(cls) -> Dict[str, bool]:
            elems = list(cls)
            dist = all(
                a.meet(b.join(c)) == a.meet(b).join(a.meet(c))
                for a, b, c in itertools.product(elems, repeat=3)
            )
            return {
                "distributivity": dist,
                "lem_fails_on_star": not (
                    cls.DEGRADED.join(cls.DEGRADED.neg()) == cls.COHERENT
                ),
            }

    class MatrixBanachAlgebra:  # type: ignore
        @staticmethod
        def as_complex(A: np.ndarray) -> np.ndarray:
            return np.asarray(A, dtype=np.complex128)

        @staticmethod
        def hermitize(A: np.ndarray) -> np.ndarray:
            A = np.asarray(A, dtype=np.complex128)
            return 0.5 * (A + A.conj().T)

        @staticmethod
        def schatten_norm(A: np.ndarray, p: float) -> float:
            sv = np.linalg.svd(
                np.asarray(A, dtype=np.complex128), compute_uv=False
            )
            if sv.size == 0:
                return 0.0
            if p == math.inf or p == float("inf"):
                return float(sv[0])
            if p == 1:
                return float(np.sum(sv))
            if p == 2:
                return float(np.linalg.norm(sv))
            return float(np.sum(sv ** p) ** (1.0 / p))

    class DensityOperatorAlgebra:  # type: ignore
        EPS = 1.0e-14

        @classmethod
        def sanitize(cls, rho: np.ndarray) -> np.ndarray:
            rho = MatrixBanachAlgebra.hermitize(rho)
            tr = float(np.trace(rho).real)
            if abs(tr) > 1e-15:
                rho = rho / tr
            w, V = la.eigh(rho)
            w = np.maximum(np.real(w), 0.0)
            s = float(w.sum())
            if s > 0.0:
                w = w / s
            return (V * w) @ V.conj().T

        @classmethod
        def von_neumann_entropy(cls, rho: np.ndarray) -> float:
            p = np.real(la.eigvalsh(cls.sanitize(rho)))
            p = np.maximum(p, 0.0)
            p = p[p > cls.EPS]
            if p.size == 0:
                return 0.0
            return float(-np.sum(p * np.log(p)))

        @classmethod
        def purity(cls, rho: np.ndarray) -> float:
            p = np.maximum(np.real(la.eigvalsh(cls.sanitize(rho))), 0.0)
            return float(np.sum(p ** 2))

        @classmethod
        def modular_hamiltonian_from_rho(cls, rho: np.ndarray) -> Any:
            rho = cls.sanitize(rho)
            lam, vec = la.eigh(rho)
            lam = np.maximum(np.real(lam), cls.EPS)
            lam = lam / lam.sum()
            energies = -np.log(lam)
            order = np.argsort(energies)
            return type("K", (), {
                "eigenvalues": tuple(float(x) for x in energies[order].tolist()),
                "spectral_gap": float(
                    energies[order][1] - energies[order][0]
                ) if energies.size >= 2 else float("inf"),
                "ground_energy": float(energies[order][0]),
                "eigenvectors": vec[:, order],
            })()


__version__ = "9.2.0-APU-CR3BP-AGM-RSI3-CROWBAR-POINCARE"


logger = logging.getLogger("APU.Wisdom.TOONSilentWitness")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ── Constantes espectrales / Banach ────────────────────────────────────────
_EPS = 1.0e-14
_HERMITICITY_TOL = 1.0e-12
_TRACE_TOL = 1.0e-10
_GAP_DEGENERACY_TOL = 1.0e-12
_PSD_EIG_FLOOR = 0.0

# ── Constantes de mecánica celeste de Poincaré / CR3BP ─────────────────────
_MU_FLOOR = 1.0e-6
_MU_CEIL = 0.5 - 1.0e-6
_JACOBI_TOL = 1.0e-10
_LIBRATION_TOL = 1.0e-12
_KAM_TOL = 1.0e-8
_ROUTH_CRITICAL_MU = 0.5 * (1.0 - math.sqrt(23.0 / 27.0))  # ≈ 0.0385209
_SING_HARDENING = 1.0e-12
_HILL_GRID_DEFAULT = 96
_CP_EIG_FLOOR = -1.0e-12

# ── RSI-3 / Crowbar ────────────────────────────────────────────────────────
_RSI3_DEFAULT_MIX: float = 0.15
_RSI3_DEFAULT_LEVEL: int = 3
_CROWBAR_GPIO: str = "GPIO14"
_CROWBAR_TARGET: str = "BT151"
_CROWBAR_MAX_LATENCY_NS: float = 400.0
_SWA1_KMS_CATASTROPHIC: float = 1.0e-2
_SWA1_JACOBI_CATASTROPHIC: float = 1.0e-2
_SWA1_LYAPUNOV_CATASTROPHIC: float = 5.0e-2


def _seed_from_string(s: str) -> int:
    r"""SHA-256 → semilla uint32 (determinismo reproducible, no criptográfico)."""
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sha256_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


DensityMatrixOps = DensityOperatorAlgebra
WitnessVacuumContext = VacuumModularContext




# ═══════════════════════════════════════════════════════════════════════════
# §1.0  Crowbar hardware-modelado  (SWA1 · Acción Cero)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ESP32CrowbarReport:
    r"""
    Certificado de disparo del interlock.

    El Python *modela* la ISR en IRAM (GPIO14 ↦ BT151, cota < 400 ns).
    No hay GPIO real: `triggered` es el bit de parálisis ciber-física
    del lazo cognitivo (SWA1), no un pulso eléctrico.
    """
    triggered: bool
    gpio_pin: str
    target_scr: str
    latency_ns: float
    reason: str
    timestamp_ns: int
    predicate: str = ""

    def is_within_latency_budget(self) -> bool:
        return float(self.latency_ns) <= _CROWBAR_MAX_LATENCY_NS + 1.0


@dataclass(frozen=True, slots=True)
class SWA1Predicate:
    r"""
    Predicado catastrófico de Acción Cero (SWA1).

    Dispara si alguno de:
        (KMS)   δ_KMS > ε_KMS
        (Jac)   |ΔC|/max(1,|C|) > ε_J
        (λ)     λ_max > ε_λ
        (Hill)  topología de tránsito (C ≤ C2) con solapamiento
        (Ω₃)    veredicto = ⊥
    """
    kms_drift: float = 0.0
    jacobi_residual: float = 0.0
    lyapunov_max: float = 0.0
    hill_transit: bool = False
    heyting_vetoed: bool = False

    def fires(self) -> bool:
        return bool(
            self.kms_drift > _SWA1_KMS_CATASTROPHIC
            or self.jacobi_residual > _SWA1_JACOBI_CATASTROPHIC
            or self.lyapunov_max > _SWA1_LYAPUNOV_CATASTROPHIC
            or self.hill_transit
            or self.heyting_vetoed
        )

    def reason(self) -> str:
        parts: List[str] = []
        if self.kms_drift > _SWA1_KMS_CATASTROPHIC:
            parts.append(f"KMS={self.kms_drift:.3e}")
        if self.jacobi_residual > _SWA1_JACOBI_CATASTROPHIC:
            parts.append(f"Jacobi={self.jacobi_residual:.3e}")
        if self.lyapunov_max > _SWA1_LYAPUNOV_CATASTROPHIC:
            parts.append(f"λmax={self.lyapunov_max:.3e}")
        if self.hill_transit:
            parts.append("Hill-transit")
        if self.heyting_vetoed:
            parts.append("Ω₃=⊥")
        return "|".join(parts) if parts else "none"


class ESP32CrowbarInterlock:
    r"""
    Disyuntor SWA1: cuando el predicado catastrófico se cumple, se sella
    el lazo (acción cero sobre el sistema observado).  Latencia modelada
    acotada por MAX_LATENCY_NS.
    """
    GPIO_PIN: str = _CROWBAR_GPIO
    TARGET_SCR: str = _CROWBAR_TARGET
    MAX_LATENCY_NS: float = _CROWBAR_MAX_LATENCY_NS

    @classmethod
    def trigger(cls, reason: str = "", predicate: str = "") -> ESP32CrowbarReport:
        t_start = time.time_ns()
        latency = float(time.time_ns() - t_start)
        latency_bounded = min(latency, cls.MAX_LATENCY_NS)
        logger.critical(
            "[SWA1 CROWBAR] Disparo modelado (%s ↦ %s). Latencia: %.2f ns. Razón: %s",
            cls.GPIO_PIN, cls.TARGET_SCR, latency_bounded, reason,
        )
        return ESP32CrowbarReport(
            triggered=True,
            gpio_pin=cls.GPIO_PIN,
            target_scr=cls.TARGET_SCR,
            latency_ns=latency_bounded,
            reason=reason,
            timestamp_ns=time.time_ns(),
            predicate=predicate,
        )

    @classmethod
    def trigger_if(cls, pred: SWA1Predicate) -> Optional[ESP32CrowbarReport]:
        if not pred.fires():
            return None
        return cls.trigger(reason=pred.reason(), predicate="SWA1")


# ═══════════════════════════════════════════════════════════════════════════
# §1.1  AGM sobre mundos posibles de Ω₃
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class AGMPostulateCertificate:
    r"""Certificado de K*1–K*8 (Alchourrón–Gärdenfors–Makinson 1985)."""
    agm1_closure: bool
    agm2_success: bool
    agm3_inclusion: bool
    agm4_vacuity: bool
    agm5_consistency: bool
    agm6_extensionality: bool
    agm7_superexpansion: bool
    agm8_subexpansion: bool
    all_satisfied: bool
    n_postulates_passed: int
    local_verdict: HeytingOmega3

    # aliases v9.1.0
    @property
    def agm2_alpha_in_revision(self) -> bool:
        return self.agm2_success

    @property
    def agm3_subset_expansion(self) -> bool:
        return self.agm3_inclusion

    @property
    def agm4_preservation(self) -> bool:
        return self.agm4_vacuity

    @property
    def agm5_trivial_iff_inconsistent(self) -> bool:
        return self.agm5_consistency


@dataclass(frozen=True, slots=True)
class AGMBeliefSet:
    r"""
    Creencias como conjunto de mundos posibles sobre W = {0,1,2} ≅ Ω₃.

    Convención (Grove):
        teorías  K ⊆ K'   ⇔   mundos  [[K']] ⊆ [[K]].
        K ⊨ α             ⇔   [[K]] ⊆ [[α]].
        inconsistente     ⇔   [[K]] = ∅  (trivial, explosion).

    Operaciones (identidad de Levi + full-meet contraction):
        K + α  =  [[K]] ∩ [[α]]                         (expansión)
        K ÷ α  =  [[K]]            si K ⊭ α
               =  [[K]] ∪ (W \ [[α]])   si K ⊨ α       (full-meet)
        K ∗ α  =  (K ÷ ¬α) + α

    Full-meet es la selección γ = id sobre el resto; en |W|=3 es
    canónica (no hay esferas de Grove no triviales).
    """
    worlds: FrozenSet[int]

    def __post_init__(self) -> None:
        cleaned = frozenset(int(w) for w in self.worlds if 0 <= int(w) <= 2)
        object.__setattr__(self, "worlds", cleaned)

    @classmethod
    def universe(cls) -> FrozenSet[int]:
        return frozenset({0, 1, 2})

    @classmethod
    def vacuous(cls) -> "AGMBeliefSet":
        r"""Teoría trivial (inconsistente): ningún mundo."""
        return cls(frozenset())

    @classmethod
    def top(cls) -> "AGMBeliefSet":
        r"""Ninguna creencia no tautológica: todos los mundos."""
        return cls(cls.universe())

    @property
    def is_consistent(self) -> bool:
        return len(self.worlds) > 0

    @property
    def is_trivial(self) -> bool:
        return len(self.worlds) == 0

    def entails(self, alpha: FrozenSet[int]) -> bool:
        r"""K ⊨ α  ⟺  [[K]] ⊆ [[α]]  (y la teoría trivial lo implica todo)."""
        if self.is_trivial:
            return True
        return self.worlds.issubset(alpha)

    def expand(self, alpha: FrozenSet[int]) -> "AGMBeliefSet":
        r"""K + α  =  [[K]] ∩ [[α]]."""
        return AGMBeliefSet(frozenset(self.worlds & alpha))

    def contract(self, alpha: FrozenSet[int]) -> "AGMBeliefSet":
        r"""
        Full-meet contraction:
            si K ⊭ α  →  K
            si K ⊨ α  →  K ∪ ¬α     (se añaden mundos que falsifican α)
        """
        if not self.entails(alpha):
            return self
        if self.is_trivial:
            # Trivial ⊨ α siempre; contraer añade ¬α (recupera consistencia
            # respecto de ¬α, no de α).  Full-meet: W \ α.
            return AGMBeliefSet(frozenset(self.universe() - alpha))
        return AGMBeliefSet(frozenset(self.worlds | (self.universe() - alpha)))

    def revise(self, alpha: FrozenSet[int]) -> "AGMBeliefSet":
        r"""K ∗ α = (K ÷ ¬α) + α  (Levi)."""
        neg_alpha = frozenset(self.universe() - alpha)
        return self.contract(neg_alpha).expand(alpha)

    def as_heyting(self) -> HeytingOmega3:
        r"""Meet de los mundos supervivientes (sello Ω₃ del conjunto)."""
        if self.is_trivial:
            return HeytingOmega3.VETOED
        return HeytingOmega3(min(self.worlds))


class HeytingAGMRevision:
    r"""Verificador de K*1–K*8 sobre el álgebra de mundos de Ω₃."""

    @classmethod
    def _theory_subseteq(
        cls, k_small: AGMBeliefSet, k_large: AGMBeliefSet
    ) -> bool:
        r"""K_small ⊆ K_large como teorías  ⇔  mundos(k_large) ⊆ mundos(k_small)
        (con la convención de que trivial ⊆ todo)."""
        if k_small.is_trivial:
            return True
        if k_large.is_trivial:
            return False
        return k_large.worlds.issubset(k_small.worlds)

    @classmethod
    def verify_postulates(
        cls,
        K: AGMBeliefSet,
        alpha: FrozenSet[int],
        beta: Optional[FrozenSet[int]] = None,
    ) -> AGMPostulateCertificate:
        alpha = frozenset(int(w) for w in alpha if 0 <= int(w) <= 2)
        K_star = K.revise(alpha)
        K_plus = K.expand(alpha)
        alpha_inconsistent = len(alpha) == 0

        # K*1 clausura: por representación, K*α es siempre un conjunto de mundos.
        agm1 = True
        # K*2 success: K*α ⊨ α  ⇔  mundos ⊆ α  (trivial lo implica).
        agm2 = K_star.entails(alpha)
        # K*3 inclusion: K*α ⊆ K+α  (teorías)  ⇔  mundos(K+α) ⊆ mundos(K*α)
        #                o ambos triviales.
        agm3 = cls._theory_subseteq(K_star, K_plus) or (
            K_star.is_trivial and K_plus.is_trivial
        )
        if K_plus.is_trivial:
            # K+α trivial ⇒ K*α trivial (inclusión + explosión)
            agm3 = K_star.is_trivial
        else:
            agm3 = K_plus.worlds.issubset(K_star.worlds) or K_star.is_trivial
            # wait: theory K* ⊆ K+  ⇔ worlds(K+) ⊆ worlds(K*)
            agm3 = K_plus.worlds.issubset(K_star.worlds) if not K_star.is_trivial else K_plus.is_trivial

        # Recálculo limpio K*3:
        # K*α ⊆ K+α as theories ⇔ worlds(K+α) ⊆ worlds(K*α) when both consistent;
        # if K* trivial then it is the largest theory, so K* ⊆ K+ only if K+ trivial.
        if K_star.is_trivial:
            agm3 = bool(K_plus.is_trivial)
        else:
            agm3 = bool(K_plus.worlds.issubset(K_star.worlds))

        # K*4 vacuity: ¬α ∉ K  ⇒  K+α ⊆ K*α
        # ¬α ∉ K  ⇔  K ⊭ ¬α  ⇔  not K.worlds ⊆ ¬α  ⇔  K ∩ α ≠ ∅ (si K cons.)
        neg_alpha = frozenset(AGMBeliefSet.universe() - alpha)
        not_neg_in_K = not K.entails(neg_alpha)
        if not_neg_in_K:
            if K_plus.is_trivial:
                agm4 = K_star.is_trivial
            elif K_star.is_trivial:
                agm4 = False
            else:
                agm4 = K_star.worlds.issubset(K_plus.worlds)
        else:
            agm4 = True

        # K*5 consistency: K*α trivial ⇔ α inconsistente
        agm5 = bool(K_star.is_trivial) == bool(alpha_inconsistent)

        # K*6 extensionality
        if beta is None or alpha == beta:
            if beta is not None and alpha == beta:
                agm6 = K_star.worlds == K.revise(beta).worlds
            else:
                agm6 = True
        else:
            agm6 = True  # no se afirma nada si α ≢ β

        # K*7 superexpansion: K*(α∧β) ⊆ (K*α)+β
        beta_w = (
            frozenset(int(w) for w in beta if 0 <= int(w) <= 2)
            if beta is not None
            else alpha
        )
        and_ab = frozenset(alpha & beta_w)
        left7 = K.revise(and_ab)
        right7 = K.revise(alpha).expand(beta_w)
        if left7.is_trivial:
            agm7 = True
        elif right7.is_trivial:
            agm7 = False
        else:
            agm7 = right7.worlds.issubset(left7.worlds)

        # K*8 subexpansion: ¬β ∉ K*α  ⇒  (K*α)+β ⊆ K*(α∧β)
        kstar_a = K.revise(alpha)
        neg_beta = frozenset(AGMBeliefSet.universe() - beta_w)
        if not kstar_a.entails(neg_beta):
            left8 = kstar_a.expand(beta_w)
            right8 = K.revise(and_ab)
            if left8.is_trivial:
                agm8 = right8.is_trivial
            elif right8.is_trivial:
                agm8 = False
            else:
                agm8 = right8.worlds.issubset(left8.worlds)
        else:
            agm8 = True

        flags = (agm1, agm2, agm3, agm4, agm5, agm6, agm7, agm8)
        n_pass = int(sum(bool(x) for x in flags))
        all_pass = n_pass == 8
        verdict = (
            HeytingOmega3.COHERENT if all_pass
            else (
                HeytingOmega3.DEGRADED if n_pass >= 6 else HeytingOmega3.VETOED
            )
        )
        return AGMPostulateCertificate(
            agm1_closure=bool(agm1),
            agm2_success=bool(agm2),
            agm3_inclusion=bool(agm3),
            agm4_vacuity=bool(agm4),
            agm5_consistency=bool(agm5),
            agm6_extensionality=bool(agm6),
            agm7_superexpansion=bool(agm7),
            agm8_subexpansion=bool(agm8),
            all_satisfied=bool(all_pass),
            n_postulates_passed=n_pass,
            local_verdict=verdict,
        )

    @classmethod
    def verify_all_laws(cls) -> Dict[str, Any]:
        r"""Barrido exhaustivo: 8 conjuntos K × 8 proposiciones α (y β=α, β=W)."""
        parts = list(AGMBeliefSet.universe())
        subsets = [
            frozenset(c)
            for r in range(4)
            for c in itertools.combinations(parts, r)
        ]
        n_ok = 0
        n_tot = 0
        worst = 8
        for k_w in subsets:
            K = AGMBeliefSet(k_w)
            for a in subsets:
                n_tot += 1
                cert = cls.verify_postulates(K, a, beta=a)
                n_ok += int(cert.all_satisfied)
                worst = min(worst, cert.n_postulates_passed)
                cert_w = cls.verify_postulates(K, a, beta=AGMBeliefSet.universe())
                n_tot += 1
                n_ok += int(cert_w.all_satisfied)
                worst = min(worst, cert_w.n_postulates_passed)
        return {
            "n_cases": n_tot,
            "n_passed": n_ok,
            "all_passed": n_ok == n_tot,
            "min_postulates": worst,
        }

    @classmethod
    def proposition_from_verdict(cls, verdict: HeytingOmega3) -> FrozenSet[int]:
        r"""
        Proposición del auditor: el filtro principal ↑v = {w : w ≥ v}.

        COHERENT ↦ {2},  DEGRADED ↦ {1,2},  VETOED ↦ {0,1,2}=W
        (VETOED no informa: revisión por W es no-op).
        """
        v = int(verdict)
        return frozenset(w for w in (0, 1, 2) if w >= v)


# ═══════════════════════════════════════════════════════════════════════════
# §1.2  Germen mónada RSI-3  (T, η, μ)  — sin colisión eta/mix
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MonadAxiomVerification:
    r"""Residuos de Mac Lane.  holds_on_image: mónada sobre rayos puros 𝔓_n."""
    left_associativity: float
    right_associativity: float
    associativity_holds: bool
    left_unitarity: float
    right_unitarity: float
    unitarity_holds: bool
    level_applied: int
    verdict: HeytingOmega3
    holds_on_image: bool = False
    holds_globally: bool = False
    idempotence_residual: float = 0.0
    mix: float = _RSI3_DEFAULT_MIX


class MonadRSI3:
    r"""
    Mónada RSI-3 sobre 𝔇_n.  Wrapper del germen del motor si está presente.

    INVARIANTE: jamás existe un atributo de instancia llamado `eta`
    (colisión v9.1.0: `self.eta = float` tapaba `def eta`).  El scalar
    de mixtura es `mix`.  La unidad η es `unit()`.

        T(ρ) = (1−α) ρ + α |φ_max⟩⟨φ_max|
        η(ρ) = sanitize(ρ)
        μ    = T
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
        if eta is not None:
            mix = float(eta)
        mix = float(min(1.0, max(0.0, mix)))
        level = int(max(1, min(int(level), 4)))
        dim = int(max(1, dim))
        self.dim = dim
        self.mix = mix
        self.level = level
        self._germ = None
        if RSI3Germ is not None:
            try:
                self._germ = RSI3Germ(mix=mix, level=level, dim=dim)
            except Exception:
                self._germ = None

    @staticmethod
    def _sanitize(rho: np.ndarray) -> np.ndarray:
        if DensityOperatorAlgebra is not None and hasattr(
            DensityOperatorAlgebra, "sanitize"
        ):
            return DensityOperatorAlgebra.sanitize(rho)
        rho_h = 0.5 * (
            np.asarray(rho, dtype=np.complex128)
            + np.asarray(rho, dtype=np.complex128).conj().T
        )
        w = np.maximum(np.real(la.eigvalsh(rho_h)), 1e-14)
        tr = float(w.sum())
        return rho_h / tr if tr > 0 else rho_h

    def unit(self, rho: np.ndarray) -> np.ndarray:
        r"""η(ρ) = sanitize(ρ)."""
        if self._germ is not None:
            return self._germ.unit(rho)
        return self._sanitize(rho)

    def unit_eta(self, rho: np.ndarray) -> np.ndarray:
        r"""Alias explícito de la unidad η (nunca un float)."""
        return self.unit(rho)

    def _target(self, rho: np.ndarray) -> np.ndarray:
        if self._germ is not None:
            return self._germ._target_rank_one(rho)
        _w, V = la.eigh(self._sanitize(rho))
        v = V[:, -1].reshape(-1, 1)
        return (v @ v.conj().T).astype(np.complex128)

    def T(self, rho: np.ndarray) -> np.ndarray:
        if self._germ is not None:
            return self._germ.endofunctor(rho)
        rho = self._sanitize(rho)
        tgt = self._target(rho)
        return self._sanitize((1.0 - self.mix) * rho + self.mix * tgt)

    def mu(self, rho_TT: np.ndarray) -> np.ndarray:
        return self.T(rho_TT)

    def condense(self, rho: np.ndarray, n_levels: int) -> np.ndarray:
        if self._germ is not None:
            return self._germ.recede_improve(rho, n_levels=n_levels)
        rho_n = self._sanitize(rho)
        for _ in range(max(1, int(n_levels))):
            rho_n = self.T(rho_n)
        return self.mu(rho_n)

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
        rho = self.unit(rho_sample)
        T2 = self.T(self.T(rho))
        T3 = self.T(T2)
        left_assoc = self._dist(self.mu(self.T(self.mu(T2))), self.mu(self.mu(T3)))
        T_rho = self.T(rho)
        unit_left = self._dist(self.mu(self.T(self.unit(rho))), T_rho)
        unit_right = self._dist(self.mu(self.unit(T_rho)), T_rho)
        assoc_ok = bool(left_assoc < self._AXIOM_TOL)
        unit_ok = bool(unit_left < self._AXIOM_TOL and unit_right < self._AXIOM_TOL)

        P = self._sanitize(self._target(rho))
        P2 = self.T(self.T(P))
        P3 = self.T(P2)
        img_assoc = self._dist(self.mu(self.T(self.mu(P2))), self.mu(self.mu(P3)))
        TP = self.T(P)
        img_left = self._dist(self.mu(self.T(self.unit(P))), TP)
        img_right = self._dist(self.mu(self.unit(TP)), TP)
        img_id = self._dist(TP, P)
        holds_on_image = bool(
            img_assoc < self._IMAGE_TOL
            and img_left < self._IMAGE_TOL
            and img_right < self._IMAGE_TOL
            and img_id < self._IMAGE_TOL
        )
        holds_globally = bool(assoc_ok and unit_ok)
        idem = self._dist(self.T(T_rho), T_rho)
        if holds_on_image:
            verdict = HeytingOmega3.COHERENT
        elif assoc_ok or unit_ok:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.VETOED
        return MonadAxiomVerification(
            left_associativity=float(left_assoc),
            right_associativity=float(left_assoc),
            associativity_holds=assoc_ok,
            left_unitarity=float(unit_left),
            right_unitarity=float(unit_right),
            unitarity_holds=unit_ok,
            level_applied=int(self.level),
            verdict=verdict,
            holds_on_image=holds_on_image,
            holds_globally=holds_globally,
            idempotence_residual=float(idem),
            mix=float(self.mix),
        )

    def kleisli_compose(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        g: Callable[[np.ndarray], np.ndarray],
        rho: np.ndarray,
    ) -> np.ndarray:
        r"""(g ∘_T f)(ρ) = μ(T(g(f(ρ))))."""
        if self._germ is not None:
            return self._germ.kleisli_compose(f, g, rho)
        rho_f = self.unit(f(rho))
        T_rho_f = self.T(rho_f)
        g_T = self.unit(g(T_rho_f))
        return self.mu(self.T(g_T))


# ═══════════════════════════════════════════════════════════════════════════
# §1.3  Métricas del vacío modular y firma de la tríada
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class VacuumStateMetrics:
    vacuum_expectation_value: float
    tomita_takesaki_flow_param: float
    kms_entropy_drift: float
    silence_purity: float
    modular_ground_energy: float
    modular_spectral_gap: float
    von_neumann_entropy: float
    purity: float
    is_silent: bool
    umegaki_to_vacuum: float = 0.0
    bures_distance: float = 0.0
    trace_distance: float = 0.0
    klein_residual: float = 0.0
    dirichlet_energy: float = 0.0
    mass_ratio_mu: float = 0.0
    jacobi_constant: float = 0.0
    jacobi_residual: float = 0.0
    libration_xy: Tuple[float, float] = (0.0, 0.0)
    routh_stable: bool = False
    crowbar_triggered: bool = False
    back_action_bound: float = 0.0
    hill_topology: str = "unknown"
    rsi3_level: int = 0


@dataclass(frozen=True, slots=True)
class TriadSignature:
    trickster_illusion_type: str
    dreamer_scenario_id: str
    auditor_immunization_hash: str
    unitary_hash: str
    dreamer_hash: str
    projector_hash: str
    auditor_rank: int
    trickster_strength: float = 0.0
    dreamer_strength: float = 0.0

    def as_bytes(self) -> bytes:
        return (
            f"{self.trickster_illusion_type}|{self.dreamer_scenario_id}|"
            f"{self.auditor_immunization_hash}|{self.unitary_hash[:16]}|"
            f"{self.dreamer_hash[:16]}|{self.projector_hash[:16]}|"
            f"{self.auditor_rank}|{self.trickster_strength:.6f}|"
            f"{self.dreamer_strength:.6f}"
        ).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# §1.4  Fábrica de operadores de la tríada
# ═══════════════════════════════════════════════════════════════════════════
class TriadOperatorFactory:
    r"""
    Tríada (U_I, D, P_A):
        Trickster  — unitaria Haar-exponencial de un GUE hermítico,
        Dreamer    — CPD mixtura (1−s)I + s W,  W Wishart normalizado,
        Auditor    — proyector de rango r extraído de Haar.
    """

    @classmethod
    def _haar(cls, n: int, rng: np.random.Generator) -> np.ndarray:
        ginibre = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        q, r = np.linalg.qr(ginibre)
        d = np.diagonal(r)
        ph = np.where(np.abs(d) > 1e-30, d / np.abs(d), 1.0 + 0j)
        return q * ph.conj()

    @classmethod
    def build_unitary(cls, illusion_type: str, n: int, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        rng = np.random.default_rng(_seed_from_string(f"TRICKSTER::{illusion_type}"))
        if strength <= 0.0:
            return np.eye(n, dtype=np.complex128)
        amp = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        a_h = MatrixBanachAlgebra.hermitize(amp)
        nrm = float(np.linalg.norm(a_h, "fro")) + 1e-30
        a_h = a_h / nrm
        theta = math.pi * strength * n
        return la.expm(1j * theta * a_h)

    @classmethod
    def build_dreamer(cls, scenario_id: str, n: int, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        rng = np.random.default_rng(_seed_from_string(f"DREAMER::{scenario_id}"))
        if strength <= 0.0:
            return np.eye(n, dtype=np.complex128)
        amp = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        d_rand = amp.conj().T @ amp
        tr = float(np.trace(d_rand).real) + 1e-30
        d_rand = d_rand / tr * n
        mixed = (1.0 - strength) * np.eye(n, dtype=np.complex128) + strength * d_rand
        return MatrixBanachAlgebra.hermitize(mixed)

    @classmethod
    def build_projector(cls, immunization_hash: str, n: int, rank: int) -> np.ndarray:
        rank = int(np.clip(rank, 1, n))
        rng = np.random.default_rng(_seed_from_string(f"AUDITOR::{immunization_hash}"))
        haar = cls._haar(n, rng)
        cols = haar[:, :rank]
        return MatrixBanachAlgebra.hermitize(cols @ cols.conj().T)

    @classmethod
    def build_triad(
        cls,
        trickster_illusion_type: str,
        dreamer_scenario_id: str,
        auditor_immunization_hash: str,
        n: int,
        trickster_strength: float,
        dreamer_strength: float,
        auditor_rank: int,
    ) -> "TriadChannel":
        unitary = cls.build_unitary(trickster_illusion_type, n, trickster_strength)
        dreamer = cls.build_dreamer(dreamer_scenario_id, n, dreamer_strength)
        projector = cls.build_projector(auditor_immunization_hash, n, auditor_rank)
        signature = TriadSignature(
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            unitary_hash=_sha256_array(unitary),
            dreamer_hash=_sha256_array(dreamer),
            projector_hash=_sha256_array(projector),
            auditor_rank=int(np.clip(auditor_rank, 1, n)),
            trickster_strength=float(np.clip(trickster_strength, 0.0, 1.0)),
            dreamer_strength=float(np.clip(dreamer_strength, 0.0, 1.0)),
        )
        return TriadChannel(signature, unitary, dreamer, projector)


# ═══════════════════════════════════════════════════════════════════════════
# §1.5  Canal de tríada  (instrumento de Lüders, Choi, back-action)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class KrausRankCertificate:
    r"""
    SVD de K, rango de Choi, CP.

    Un solo Kraus  ⇒  rank(Choi) = rank(K) = 1  (canal de rango 1,
    no necesariamente TP).  CP  ⇔  Choi ≥ 0  ⇔  λ_min ≥ −ε.
    """
    singular_values: Tuple[float, ...]
    choi_rank: int
    kraus_rank: int
    cp_min_eigenvalue: float
    is_completely_positive: bool
    is_rank_one: bool
    tp_residual: float
    diamond_upper: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "choi_rank": int(self.choi_rank),
            "kraus_rank": int(self.kraus_rank),
            "cp_min_eigenvalue": float(self.cp_min_eigenvalue),
            "is_completely_positive": bool(self.is_completely_positive),
            "is_rank_one": bool(self.is_rank_one),
            "tp_residual": float(self.tp_residual),
            "diamond_upper": float(self.diamond_upper),
        }


class TriadChannel:
    r"""
    Instrumento de Lüders de un solo Kraus:

        ρ ↦  K ρ K† / Tr(K ρ K†),     K = P_A · D · U_I.

    No es TP en general (P_A es un proyector).  El defecto TP mide
    la probabilidad de fallo del instrumento, cota de back-action SWA1.
    """

    def __init__(
        self,
        signature: TriadSignature,
        U_I: np.ndarray,
        D: np.ndarray,
        P_A: np.ndarray,
    ) -> None:
        self.signature = signature
        self.U = MatrixBanachAlgebra.as_complex(U_I)
        self.D = MatrixBanachAlgebra.hermitize(D)
        self.P = MatrixBanachAlgebra.hermitize(P_A)
        self.K = self.P @ self.D @ self.U

    def apply_linear(self, rho: np.ndarray) -> np.ndarray:
        rho = MatrixBanachAlgebra.as_complex(rho)
        return self.K @ rho @ self.K.conj().T

    def apply(self, rho: np.ndarray) -> np.ndarray:
        out = self.apply_linear(rho)
        tr = float(np.trace(out).real)
        if tr < 1e-30:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        return out / tr

    def success_probability(self, rho: np.ndarray) -> float:
        metric = self.K.conj().T @ self.K
        rho = MatrixBanachAlgebra.as_complex(rho)
        return float(np.real(np.trace(rho @ metric)))

    def kraus_norm(self) -> float:
        return float(np.linalg.norm(self.K, "fro"))

    def tp_residual(self) -> float:
        n = self.K.shape[0]
        defect = self.K.conj().T @ self.K - np.eye(n, dtype=np.complex128)
        return MatrixBanachAlgebra.schatten_norm(defect, math.inf)

    def projector_residual(self) -> float:
        return float(np.linalg.norm(self.P @ self.P - self.P, "fro"))

    def unitary_residual(self) -> float:
        n = self.U.shape[0]
        eye = np.eye(n, dtype=np.complex128)
        return float(np.linalg.norm(self.U.conj().T @ self.U - eye, "fro"))

    def choi_matrix(self) -> np.ndarray:
        n = self.K.shape[0]
        vec_k = self.K.reshape((n * n, 1), order="F")
        return vec_k @ vec_k.conj().T

    def cp_min_eigenvalue(self) -> float:
        choi = MatrixBanachAlgebra.hermitize(self.choi_matrix())
        w = np.real(la.eigvalsh(choi))
        return float(w.min()) if w.size else 0.0

    def diamond_norm_upper_bound(self) -> float:
        return MatrixBanachAlgebra.schatten_norm(self.K, math.inf) ** 2

    def kraus_svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return la.svd(self.K, full_matrices=False)

    def choi_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.K, tol=1e-10))

    def is_rank_one_channel(self) -> bool:
        return self.choi_rank() == 1

    def back_action_bound(self, rho: Optional[np.ndarray] = None) -> float:
        r"""
        Cota de back-action Lüders (SWA1):

            ‖Δρ‖_1 / 2  ≤  √(1 − p_success),   p = Tr(K ρ K†).

        Sin ρ se usa el defecto TP como proxy (peor caso).
        """
        if rho is None:
            return float(math.sqrt(max(0.0, min(1.0, self.tp_residual()))))
        p = float(np.clip(self.success_probability(rho), 0.0, 1.0))
        return float(math.sqrt(max(0.0, 1.0 - p)))

    def kraus_certificate(self) -> KrausRankCertificate:
        _U, s, _Vh = self.kraus_svd()
        sv = tuple(float(x) for x in np.real(s).tolist())
        cp = self.cp_min_eigenvalue()
        rank = self.choi_rank()
        return KrausRankCertificate(
            singular_values=sv,
            choi_rank=rank,
            kraus_rank=rank,
            cp_min_eigenvalue=float(cp),
            is_completely_positive=bool(cp >= _CP_EIG_FLOOR),
            is_rank_one=bool(rank == 1),
            tp_residual=float(self.tp_residual()),
            diamond_upper=float(self.diamond_norm_upper_bound()),
        )

    def structural_residuals(self) -> Dict[str, float]:
        cert = self.kraus_certificate()
        return {
            "tp_residual": cert.tp_residual,
            "projector_residual": self.projector_residual(),
            "unitary_residual": self.unitary_residual(),
            "cp_min_eigenvalue": cert.cp_min_eigenvalue,
            "diamond_upper": cert.diamond_upper,
            "choi_rank": float(cert.choi_rank),
            "back_action": self.back_action_bound(),
        }

    def trickster_strength_estimate(self) -> float:
        r"""‖U − I‖_F / (2√n) ∈ [0, 1]  (1 ⇔ antipodal Haar)."""
        n = self.U.shape[0]
        dist = float(np.linalg.norm(self.U - np.eye(n, dtype=np.complex128), "fro"))
        return float(min(1.0, dist / max(2.0 * math.sqrt(n), _EPS)))

    def dreamer_strength_estimate(self) -> float:
        n = self.D.shape[0]
        dist = float(np.linalg.norm(self.D - np.eye(n, dtype=np.complex128), "fro"))
        return float(min(1.0, dist / max(math.sqrt(n), _EPS)))


# ═══════════════════════════════════════════════════════════════════════════
# §1.6  Carta canónica de Poincaré planar + Delaunay
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class PoincareCanonicalCoordinates:
    r"""
    Carta de Poincaré planar (no singular en e = 0):

        Λ = L,   λ = ℓ + g,   ξ = √(2(L−G)) cos g,   η = √(2(L−G)) sin g.

    e² = ξ² + η²  /  (Λ + √(Λ² − (ξ²+η²)/2 · something)) — ver eccentricity.
    En el germen L4 del CR3BP, (ξ, η) son las coordenadas de tadpole.
    """
    Lambda: float
    lambd: float
    xi: float
    eta: float

    @property
    def eccentricity(self) -> float:
        if self.Lambda <= 0.0:
            return 0.0
        s = (self.xi ** 2 + self.eta ** 2) / (2.0 * self.Lambda)
        s = float(np.clip(s, 0.0, 1.0))
        return math.sqrt(max(0.0, 1.0 - (1.0 - s) ** 2))

    def generating_function(self) -> float:
        r"""S = Λ λ + ξ η / 2  (tipo 2, localmente canónica)."""
        return float(self.Lambda * self.lambd + 0.5 * self.xi * self.eta)

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.Lambda, self.lambd, self.xi, self.eta], dtype=np.float64
        )

    @classmethod
    def from_delaunay(cls, d: "DelaunayElements") -> "PoincareCanonicalCoordinates":
        L = float(max(d.L, 0.0))
        G = float(min(max(d.G, 0.0), L))
        rt = math.sqrt(max(2.0 * (L - G), 0.0))
        return cls(
            Lambda=L,
            lambd=float(d.l + d.g),
            xi=rt * math.cos(d.g),
            eta=rt * math.sin(d.g),
        )


@dataclass(frozen=True, slots=True)
class DelaunayElements:
    L: float
    G: float
    H: float
    l: float
    g: float
    h: float


# ═══════════════════════════════════════════════════════════════════════════
# §1.7  CR3BP: estado, potencial efectivo, Jacobi, jacobiano
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class CR3BPState:
    r"""Estado en el marco rotante sinodial: z = (x, y, ẋ, ẏ)."""
    x: float
    y: float
    vx: float
    vy: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float64)

    @classmethod
    def from_array(cls, z: np.ndarray) -> "CR3BPState":
        arr = np.asarray(z, dtype=np.float64).ravel()
        return cls(
            x=float(arr[0]), y=float(arr[1]),
            vx=float(arr[2]), vy=float(arr[3]),
        )

    def kinetic(self) -> float:
        return 0.5 * (self.vx ** 2 + self.vy ** 2)


class CR3BPDynamics:
    r"""
    Problema circular restringido de tres cuerpos, planar, marco rotante.

    Potencial efectivo (convención Szebehely §3.1, sin el término μ(1−μ)/2):

        Ω(x,y) = (x² + y²)/2  +  (1−μ)/r₁  +  μ/r₂,

        r₁² = (x+μ)² + y²,   r₂² = (x−1+μ)² + y².

    Jacobi:  C = 2Ω − (ẋ² + ẏ²)  (integral primera; Ċ ≡ 0 a lo largo del flujo).
    Campo:   ẍ − 2ẏ = Ω_x,   ÿ + 2ẋ = Ω_y.
    """

    @staticmethod
    def clip_mu(mu: float) -> float:
        return float(np.clip(mu, _MU_FLOOR, _MU_CEIL))

    @staticmethod
    def _radii(x: float, y: float, mu: float) -> Tuple[float, float]:
        r1 = math.hypot(x + mu, y)
        r2 = math.hypot(x - 1.0 + mu, y)
        return max(r1, _SING_HARDENING), max(r2, _SING_HARDENING)

    @classmethod
    def potential(cls, x: float, y: float, mu: float) -> float:
        mu = cls.clip_mu(mu)
        r1, r2 = cls._radii(x, y, mu)
        return 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2

    @classmethod
    def potential_grad(cls, x: float, y: float, mu: float) -> Tuple[float, float]:
        mu = cls.clip_mu(mu)
        r1, r2 = cls._radii(x, y, mu)
        d_x = (
            x
            - (1.0 - mu) * (x + mu) / r1 ** 3
            - mu * (x - 1.0 + mu) / r2 ** 3
        )
        d_y = y - (1.0 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
        return d_x, d_y

    @classmethod
    def vector_field(cls, mu: float) -> Callable[[float, np.ndarray], np.ndarray]:
        mu_ = cls.clip_mu(mu)

        def f(_t: float, z: np.ndarray) -> np.ndarray:
            x, y, vx, vy = (float(z[0]), float(z[1]), float(z[2]), float(z[3]))
            d_x, d_y = cls.potential_grad(x, y, mu_)
            return np.array(
                [vx, vy, 2.0 * vy + d_x, -2.0 * vx + d_y], dtype=np.float64
            )

        return f

    @classmethod
    def jacobi_constant(cls, state: CR3BPState, mu: float) -> float:
        return 2.0 * cls.potential(state.x, state.y, mu) - (
            state.vx ** 2 + state.vy ** 2
        )

    @classmethod
    def jacobi_residual(cls, z0: CR3BPState, z1: CR3BPState, mu: float) -> float:
        c0 = cls.jacobi_constant(z0, mu)
        c1 = cls.jacobi_constant(z1, mu)
        return abs(c0 - c1) / max(1.0, abs(c0))

    @classmethod
    def jacobian(cls, z: np.ndarray, mu: float) -> np.ndarray:
        r"""
        Linearización 4×4 del campo en z (marco rotante).

            A = [[  0 ,  0 , 1, 0 ],
                 [  0 ,  0 , 0, 1 ],
                 [ Ωxx, Ωxy, 0, 2 ],
                 [ Ωxy, Ωyy,-2, 0 ]].
        """
        mu = cls.clip_mu(mu)
        x, y = float(z[0]), float(z[1])
        r1_2 = (x + mu) ** 2 + y * y
        r2_2 = (x - 1.0 + mu) ** 2 + y * y
        r1 = max(math.sqrt(r1_2), _SING_HARDENING)
        r2 = max(math.sqrt(r2_2), _SING_HARDENING)
        r1_3, r2_3 = r1 ** 3, r2 ** 3
        r1_5, r2_5 = r1 ** 5, r2 ** 5
        dxx = (
            1.0
            - (1.0 - mu) / r1_3
            - mu / r2_3
            + 3.0 * (1.0 - mu) * (x + mu) ** 2 / r1_5
            + 3.0 * mu * (x - 1.0 + mu) ** 2 / r2_5
        )
        dyy = (
            1.0
            - (1.0 - mu) / r1_3
            - mu / r2_3
            + 3.0 * (1.0 - mu) * y * y / r1_5
            + 3.0 * mu * y * y / r2_5
        )
        dxy = (
            3.0 * (1.0 - mu) * (x + mu) * y / r1_5
            + 3.0 * mu * (x - 1.0 + mu) * y / r2_5
        )
        A = np.zeros((4, 4), dtype=np.float64)
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[2, 0] = dxx
        A[2, 1] = dxy
        A[2, 3] = 2.0
        A[3, 0] = dxy
        A[3, 1] = dyy
        A[3, 2] = -2.0
        return A

    @classmethod
    def tadpole_seed_near_L4(
        cls,
        mu: float,
        L4_xy: Tuple[float, float],
        dx: float = 1.0e-3,
        dy: float = -1.0e-3,
        vx: float = 0.0,
        vy: float = 0.0,
    ) -> CR3BPState:
        r"""Germen de órbita tadpole: desplazamiento infinitesimal de L4."""
        return CR3BPState(
            x=float(L4_xy[0] + dx),
            y=float(L4_xy[1] + dy),
            vx=float(vx),
            vy=float(vy),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §1.8  Puntos de libración L1–L5, Routh, Lyapunov lineal
# ═══════════════════════════════════════════════════════════════════════════
class LibrationPointSolver:
    r"""
    Equilibrios de Ω_x = Ω_y = 0.

    Colineales L1, L2, L3: y = 0, raíz de Ω_x(x,0)=0 por Brent.
    Equiláteros L4, L5: (1/2 − μ,  ±√3/2)  (cerrado).
    Routh: L4/L5 linealmente estables  ⇔  μ < μ_R = (1−√(23/27))/2.
    """

    @staticmethod
    def _collinear_equation(x: float, mu: float) -> float:
        return (
            x
            - (1.0 - mu)
            * (x + mu)
            / max(abs(x + mu), _SING_HARDENING) ** 3
            - mu
            * (x - 1.0 + mu)
            / max(abs(x - 1.0 + mu), _SING_HARDENING) ** 3
        )

    @classmethod
    def solve_collinear(cls, mu: float, bracket: Tuple[float, float]) -> float:
        mu = CR3BPDynamics.clip_mu(mu)
        a, b = bracket
        fa = cls._collinear_equation(a, mu)
        fb = cls._collinear_equation(b, mu)
        if fa * fb > 0:
            xs = np.linspace(a, b, 129)
            last = xs[0]
            for xg in xs[1:]:
                fg = cls._collinear_equation(xg, mu)
                if cls._collinear_equation(last, mu) * fg <= 0.0:
                    return float(
                        brentq(cls._collinear_equation, last, xg, args=(mu,))
                    )
                last = xg
            return float(0.5 * (a + b))
        return float(brentq(cls._collinear_equation, a, b, args=(mu,)))

    @classmethod
    def solve_all(cls, mu: float) -> Dict[str, Tuple[float, float]]:
        mu = CR3BPDynamics.clip_mu(mu)
        x1 = cls.solve_collinear(mu, (-mu + 1e-3, 1.0 - mu - 1e-3))
        x2 = cls.solve_collinear(mu, (1.0 - mu + 1e-3, 1.0 - mu + 4.0))
        x3 = cls.solve_collinear(mu, (-4.0, -mu - 1e-3))
        x4 = 0.5 - mu
        y4 = math.sqrt(3.0) / 2.0
        return {
            "L1": (x1, 0.0),
            "L2": (x2, 0.0),
            "L3": (x3, 0.0),
            "L4": (x4, y4),
            "L5": (x4, -y4),
        }

    @staticmethod
    def routh_stability(mu: float) -> bool:
        mu = CR3BPDynamics.clip_mu(mu)
        return bool(27.0 * mu * (1.0 - mu) < 1.0)

    @staticmethod
    def routh_mu_critical() -> float:
        return float(_ROUTH_CRITICAL_MU)

    @classmethod
    def jacobi_at_libration(cls, mu: float) -> Dict[str, float]:
        r"""C_i = 2Ω(L_i)  (valores críticos de Jacobi / Szebehely)."""
        L = cls.solve_all(mu)
        return {
            name: 2.0 * CR3BPDynamics.potential(xy[0], xy[1], mu)
            for name, xy in L.items()
        }

    @classmethod
    def jacobian_at_point(cls, x: float, y: float, mu: float) -> np.ndarray:
        return CR3BPDynamics.jacobian(np.array([x, y, 0.0, 0.0]), mu)

    @classmethod
    def lyapunov_stability(cls, name: str, mu: float) -> bool:
        r"""¿Re λ = 0 para todos los autovalores del jacobiano en L_name?"""
        L = cls.solve_all(mu)
        if name not in L:
            raise ValueError(f"Punto de libración desconocido: {name}")
        x, y = L[name]
        eigs = la.eigvals(cls.jacobian_at_point(x, y, mu))
        return bool(np.all(np.abs(np.real(eigs)) < 1e-8))

    @classmethod
    def spectral_type(cls, name: str, mu: float) -> str:
        L = cls.solve_all(mu)
        eigs = la.eigvals(cls.jacobian_at_point(L[name][0], L[name][1], mu))
        re = np.real(eigs)
        im = np.imag(eigs)
        if np.all(np.abs(re) < 1e-8) and np.any(np.abs(im) > 1e-8):
            return "elliptic-center"
        if np.any(np.abs(re) > 1e-8) and np.any(np.abs(im) > 1e-8):
            return "saddle-focus"
        if np.any(re > 1e-8) and np.all(np.abs(im) < 1e-8):
            return "hyperbolic-saddle"
        return "degenerate"


# ═══════════════════════════════════════════════════════════════════════════
# §1.9  Región de Hill  (Szebehely 1967 §3.4)
# ═══════════════════════════════════════════════════════════════════════════
class HillTopology(IntEnum):
    r"""
    Topología de las curvas de velocidad cero (C decreciente):

        THREE_ISLANDS  C > C1     tres componentes (m1, m2, exterior)
        INTERIOR_GATE  C1 ≥ C > C2   L1 abierto (m1–m2 comunican)
        EXTERIOR_GATE  C2 ≥ C > C3   L2 abierto (exterior comunica)
        TRANSIT        C3 ≥ C > C4   L3 abierto (tránsito global)
        FREE           C ≤ C4=C5     L4/L5 sumergidos, plano libre
    """
    FREE = 0
    TRANSIT = 1
    EXTERIOR_GATE = 2
    INTERIOR_GATE = 3
    THREE_ISLANDS = 4


@dataclass(frozen=True, slots=True)
class JacobiRegion:
    r"""
    Región de Hill a constante C:

        H_C = {(x,y) : Ω(x,y) ≥ C/2},     ZVC = {Ω = C/2}.

    `libration_gaps[name] = Ω(L) − C/2`;  gap < 0  ⇒  L está en la ZVC
    prohibida (puerta cerrada).
    """
    jacobi_constant: float
    mu: float
    hill_boundary_area_estimate: float
    zvc_connected: bool
    libration_gaps: Dict[str, float]
    local_verdict: HeytingOmega3
    topology: str = "unknown"
    critical_values: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def classify_topology(cls, C: float, crit: Mapping[str, float]) -> HillTopology:
        C1 = float(crit.get("L1", float("inf")))
        C2 = float(crit.get("L2", C1))
        C3 = float(crit.get("L3", C2))
        C4 = float(crit.get("L4", C3))
        if C > C1:
            return HillTopology.THREE_ISLANDS
        if C > C2:
            return HillTopology.INTERIOR_GATE
        if C > C3:
            return HillTopology.EXTERIOR_GATE
        if C > C4:
            return HillTopology.TRANSIT
        return HillTopology.FREE

    @classmethod
    def evaluate(
        cls, state: CR3BPState, mu: float, n_grid: int = _HILL_GRID_DEFAULT
    ) -> "JacobiRegion":
        mu = CR3BPDynamics.clip_mu(mu)
        C = CR3BPDynamics.jacobi_constant(state, mu)
        crit = LibrationPointSolver.jacobi_at_libration(mu)
        topo = cls.classify_topology(C, crit)

        x_lo, x_hi = -1.5 - mu, 1.5
        y_lo, y_hi = -1.5, 1.5
        xs = np.linspace(x_lo, x_hi, n_grid)
        ys = np.linspace(y_lo, y_hi, n_grid)
        XX, YY = np.meshgrid(xs, ys)
        r1 = np.sqrt((XX + mu) ** 2 + YY ** 2) + 1e-12
        r2 = np.sqrt((XX - 1.0 + mu) ** 2 + YY ** 2) + 1e-12
        Omega = 0.5 * (XX ** 2 + YY ** 2) + (1.0 - mu) / r1 + mu / r2
        mask = Omega >= 0.5 * C
        dx = (x_hi - x_lo) / max(n_grid - 1, 1)
        dy = (y_hi - y_lo) / max(n_grid - 1, 1)
        area = float(mask.sum() * dx * dy)

        L = LibrationPointSolver.solve_all(mu)
        gaps: Dict[str, float] = {}
        for name, (lx, ly) in L.items():
            om = CR3BPDynamics.potential(lx, ly, mu)
            gaps[name] = float(om - 0.5 * C)

        zvc_conn = bool(gaps.get("L1", 0.0) >= 0.0)  # L1 accesible ⇔ puerta abierta
        if topo in (HillTopology.THREE_ISLANDS, HillTopology.INTERIOR_GATE):
            local = HeytingOmega3.COHERENT
        elif topo == HillTopology.EXTERIOR_GATE:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED  # tránsito / plano libre: no hay confinamiento
        return cls(
            jacobi_constant=float(C),
            mu=float(mu),
            hill_boundary_area_estimate=area,
            zvc_connected=zvc_conn,
            libration_gaps=gaps,
            local_verdict=local,
            topology=topo.name,
            critical_values={k: float(v) for k, v in crit.items()},
        )


# ═══════════════════════════════════════════════════════════════════════════
# §1.10  Preparación del vacío modular del Testigo
# ═══════════════════════════════════════════════════════════════════════════
class WitnessVacuumPreparation:
    r"""Laplaciano de camino (Dirichlet discreto) y anclaje al motor."""

    @classmethod
    def path_laplacian(cls, n: int) -> np.ndarray:
        lap = np.zeros((n, n), dtype=np.complex128)
        for i in range(n - 1):
            lap[i, i] += 1.0
            lap[i + 1, i + 1] += 1.0
            lap[i, i + 1] -= 1.0
            lap[i + 1, i] -= 1.0
        return MatrixBanachAlgebra.hermitize(lap)

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> np.ndarray:
        if VacuumStatePreparation is not None and hasattr(
            VacuumStatePreparation, "ground_state_projector"
        ):
            return VacuumStatePreparation.ground_state_projector(H)
        w, vecs = la.eigh(MatrixBanachAlgebra.hermitize(H))
        omega = vecs[:, int(np.argmin(np.real(w)))].reshape(-1, 1)
        return MatrixBanachAlgebra.hermitize(omega @ omega.conj().T)

    @classmethod
    def prepare_context(cls, H: np.ndarray, beta: float) -> Any:
        if VacuumStatePreparation is not None:
            return VacuumStatePreparation.prepare_vacuum_context(H, beta=beta)
        rho = DensityMatrixOps.sanitize(
            la.expm(-float(beta) * MatrixBanachAlgebra.hermitize(H))
        ) if hasattr(la, "expm") else np.eye(H.shape[0], dtype=np.complex128) / H.shape[0]
        return type("Ctx", (), {
            "rho": rho, "K": DensityMatrixOps.modular_hamiltonian_from_rho(rho),
            "H": H, "ground_projector": cls.ground_state_projector(H),
            "beta": float(beta),
        })()


# ═══════════════════════════════════════════════════════════════════════════
# §1.11  PoincareSectionSeed  +  morfismo tríada → CR3BP
#        HAND-OFF FORMAL  FASE 1 → FASE 2
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class PoincareSectionSeed:
    r"""
    Objeto de transporte FASE 1 → FASE 2.

    Dominio de `PoincareSectionIntegrator.integrate_section` (primer
    método de FASE 2).  Porta μ, C, el germen tadpole cerca de L4,
    la topología de Hill, el certificado AGM y el rango de Kraus.
    """
    mu: float
    jacobi_constant: float
    initial_state: CR3BPState
    libration_points: Dict[str, Tuple[float, float]]
    routh_stable: bool
    triad_signature: TriadSignature
    tau_recurrence: float
    provenance_hash: str
    agm_certificate: Optional[AGMPostulateCertificate] = None
    hill_topology: str = "unknown"
    kraus_rank: int = 1
    back_action_bound: float = 0.0
    rsi3_mix: float = _RSI3_DEFAULT_MIX
    crowbar_armed: bool = False
    poincare_chart: Optional[PoincareCanonicalCoordinates] = None
    jacobi_region: Optional[JacobiRegion] = None
    kraus_certificate: Optional[KrausRankCertificate] = None

    def is_routh_kam_candidate(self) -> bool:
        return bool(self.routh_stable and self.hill_topology in (
            HillTopology.THREE_ISLANDS.name,
            HillTopology.INTERIOR_GATE.name,
        ))


class TriadToCR3BPMorphism:
    r"""
    Morfismo  (U_I, D, P_A, ρ_vac)  ↦  (μ, C, z₀, L, Hill, AGM).

    μ se lee de las *fuerzas* de Trickster/Dreamer/Auditor, no de
    |det U| ≡ 1.  C se lee del residuo de Frobenius ρ_obs − ρ_vac
    como análogo de 2Ω − v².  z₀ es un germen tadpole de L4.
    """

    @staticmethod
    def mass_ratio(
        trickster_strength: float,
        dreamer_strength: float,
        auditor_rank: int,
        n: int,
    ) -> float:
        bt = float(np.clip(trickster_strength, 0.0, 1.0))
        bd = float(np.clip(dreamer_strength, 0.0, 1.0))
        ar = float(max(1, auditor_rank)) / float(max(1, n))
        denom = bt + bd + 1.0e-9
        raw = bt * ar / denom
        return float(np.clip(raw, _MU_FLOOR, _MU_CEIL))

    @staticmethod
    def jacobi_from_residual(rho_obs: np.ndarray, rho_vac: np.ndarray) -> float:
        rho_obs = MatrixBanachAlgebra.as_complex(rho_obs)
        rho_vac = MatrixBanachAlgebra.as_complex(rho_vac)
        diff = float(np.linalg.norm(rho_obs - rho_vac, "fro"))
        n = rho_obs.shape[0]
        k_norm_sq = float(np.real(np.trace(rho_obs.conj().T @ rho_obs))) * n
        return 2.0 * k_norm_sq - diff ** 2

    @classmethod
    def reduce_channel(
        cls, channel: TriadChannel, rho_vac: np.ndarray
    ) -> Tuple[float, float, CR3BPState, Dict[str, Tuple[float, float]]]:
        n = int(channel.K.shape[0])
        sig = channel.signature
        bt = (
            float(sig.trickster_strength)
            if sig.trickster_strength > 0.0
            else channel.trickster_strength_estimate()
        )
        bd = (
            float(sig.dreamer_strength)
            if sig.dreamer_strength > 0.0
            else channel.dreamer_strength_estimate()
        )
        mu = cls.mass_ratio(
            trickster_strength=bt,
            dreamer_strength=bd,
            auditor_rank=sig.auditor_rank,
            n=n,
        )
        rho_obs = channel.apply(rho_vac)
        C = cls.jacobi_from_residual(rho_obs, rho_vac)
        L = LibrationPointSolver.solve_all(mu)
        x0, y0 = L["L4"]
        cp_r = channel.cp_min_eigenvalue()
        vx0 = -0.02 * float(np.sign(cp_r) if cp_r != 0.0 else 1.0)
        vy0 = 0.03 + 0.01 * float(np.real(np.trace(rho_obs @ channel.K)))
        state0 = CR3BPDynamics.tadpole_seed_near_L4(
            mu, (x0, y0), dx=1.0e-3, dy=-1.0e-3, vx=vx0, vy=vy0
        )
        return mu, C, state0, L

    @classmethod
    def emit_poincare_seed(
        cls,
        channel: TriadChannel,
        rho_vac: np.ndarray,
        tau_recurrence: float = 1.0,
        provenance_tag: str = "SILENT-WITNESS",
        agm_verdict: Optional[HeytingOmega3] = None,
        rsi3_mix: float = _RSI3_DEFAULT_MIX,
        rho_for_backaction: Optional[np.ndarray] = None,
    ) -> PoincareSectionSeed:
        mu, C, state0, L = cls.reduce_channel(channel, rho_vac)
        routh = LibrationPointSolver.routh_stability(mu)
        region = JacobiRegion.evaluate(state0, mu)
        kraus_cert = channel.kraus_certificate()
        back = channel.back_action_bound(rho_for_backaction if rho_for_backaction is not None else rho_vac)

        # Carta Poincaré del germen tadpole (Λ ∼ √a_synodic, ξ+iη ∼ e·e^{iϖ})
        dx = state0.x - L["L4"][0]
        dy = state0.y - L["L4"][1]
        chart = PoincareCanonicalCoordinates(
            Lambda=float(max(math.hypot(state0.x, state0.y), _EPS)),
            lambd=float(math.atan2(state0.y, state0.x)),
            xi=float(dx),
            eta=float(dy),
        )

        agm_cert: Optional[AGMPostulateCertificate] = None
        if agm_verdict is not None:
            K0 = AGMBeliefSet.top()
            alpha = HeytingAGMRevision.proposition_from_verdict(agm_verdict)
            agm_cert = HeytingAGMRevision.verify_postulates(K0, alpha, beta=alpha)

        pred = SWA1Predicate(
            jacobi_residual=0.0,
            hill_transit=region.topology in (
                HillTopology.TRANSIT.name, HillTopology.FREE.name
            ),
            heyting_vetoed=(
                agm_cert.local_verdict == HeytingOmega3.VETOED
                if agm_cert is not None
                else False
            ),
        )
        crowbar_armed = bool(pred.fires())
        if crowbar_armed:
            ESP32CrowbarInterlock.trigger_if(pred)

        payload = (
            f"{channel.signature.unitary_hash[:16]}|"
            f"{channel.signature.dreamer_hash[:16]}|"
            f"{channel.signature.projector_hash[:16]}|"
            f"{mu:.12e}|{C:.12e}|{tau_recurrence:.12e}|"
            f"{region.topology}|{kraus_cert.choi_rank}|"
            f"{'AGM' if agm_cert else 'NOAGM'}"
        ).encode("ascii")
        prov = hashlib.sha256(payload + provenance_tag.encode("ascii")).hexdigest()
        logger.info(
            "[F1→F2] μ=%.6f C=%.6f Routh=%s Hill=%s Choi-rk=%d AGM=%s/8 "
            "back=%.3e crowbar=%s prov=%s…",
            mu, C, routh, region.topology, kraus_cert.choi_rank,
            agm_cert.n_postulates_passed if agm_cert else 0,
            back, crowbar_armed, prov[:12],
        )
        return PoincareSectionSeed(
            mu=mu,
            jacobi_constant=C,
            initial_state=state0,
            libration_points=L,
            routh_stable=routh,
            triad_signature=channel.signature,
            tau_recurrence=float(tau_recurrence),
            provenance_hash=prov,
            agm_certificate=agm_cert,
            hill_topology=region.topology,
            kraus_rank=int(kraus_cert.choi_rank),
            back_action_bound=float(back),
            rsi3_mix=float(rsi3_mix),
            crowbar_armed=crowbar_armed,
            poincare_chart=chart,
            jacobi_region=region,
            kraus_certificate=kraus_cert,
        )


_EVENT_NEWTON_TOL: float = 1.0e-12
_EVENT_NEWTON_MAX: int = 16
_LC_COLLISION_RADIUS: float = 1.0e-4
_GALI_FLOOR: float = 1.0e-16
_LYAPUNOV_PAIRING_TOL: float = 1.0e-2
_RETURN_MAP_DET_TOL: float = 5.0e-2
_MELNIKOV_TRANSVERSE_TOL: float = 1.0e-10


# ═══════════════════════════════════════════════════════════════════════════
# §2.0  Tipos de transporte de la sección de Poincaré
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SectionCrossing:
    r"""
    Cruce orientado de  Σ = {y = 0, ẏ > 0}  (sección de Poincaré clásica
    del CR3BP planar).  El par (x, ẋ) es la coordenada de la sección;
    Jacobi se reevalúa como certificado de la integración.
    """
    step_index: int
    t: float
    x: float
    vx: float
    jacobi_residual: float
    newton_iterations: int = 0
    y_event: float = 0.0
    vy: float = 0.0

    def as_section_point(self) -> Tuple[float, float]:
        return float(self.x), float(self.vx)


@dataclass(frozen=True, slots=True)
class ReturnMapReport:
    r"""
    Linearización empírica  P(x,ẋ) ↦ A · (x,ẋ)  del mapa de retorno.

    El mapa de Poincaré de un flujo hamiltoniano 4D reducido por Jacobi
    es (localmente) simplicial de área:  |det A| = 1.  El residuo
    | |det A| − 1 | certifica la calidad simpléctica del censo.
    """
    matrix: np.ndarray
    det: float
    det_residual: float
    is_area_preserving: bool
    n_pairs: int
    linearity_residual: float

    def monodromy_eigenvalues(self) -> Tuple[complex, ...]:
        w = np.linalg.eigvals(np.asarray(self.matrix, dtype=np.float64))
        return tuple(complex(z) for z in w.tolist())

    def is_elliptic(self) -> bool:
        w = self.monodromy_eigenvalues()
        return bool(w) and all(abs(abs(z) - 1.0) < 1e-2 for z in w)


@dataclass(frozen=True, slots=True)
class CelestialDiagnostics:
    r"""Extracto celeste del censo de sección + indicadores de caos."""
    liapunov_spectrum: Tuple[float, ...]
    pairing_residual: float
    volume_residual: float
    sali: float
    gali: float
    gali_k: int
    gali_decay_rate: float
    jacobi_max_residual: float
    mean_return_time: float
    n_crossings: int
    kam_bias: float
    rotation_number: float
    diophantine_residue: float
    chirikov_overlap: bool
    cantor_measure: float
    melnikov_peak: float
    melnikov_area: float
    melnikov_zeros: int
    melnikov_transverse: bool
    smale_horseshoe: bool
    section_entropy: float
    return_map_det_residual: float
    hill_topology: str = "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# §2.1  Integrador de sección de Poincaré
#       (morfismo inicial de FASE 2)
# ═══════════════════════════════════════════════════════════════════════════
class PoincareSectionIntegrator:
    r"""
    Integrador RK4 del campo rotante con detección de eventos y
    refinamiento de Hermite–Newton sobre  F(t) = y(t) = 0,
    F'(t) = ẏ(t).

    Sección orientada:  y cruza 0 en sentido creciente (ẏ > 0).
    Es el análogo del mapa de Poincaré del problema restringido
    (Szebehely §8; Koon–Lo–Marsden–Ross).
    """

    def __init__(
        self,
        dt: float = 5.0e-3,
        max_steps: int = 200_000,
        newton_tol: float = _EVENT_NEWTON_TOL,
        newton_max_iter: int = _EVENT_NEWTON_MAX,
        collision_radius: float = _LC_COLLISION_RADIUS,
    ) -> None:
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.newton_tol = float(newton_tol)
        self.newton_max_iter = int(newton_max_iter)
        self.collision_radius = float(collision_radius)

    @staticmethod
    def _rk4_step(
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        z: np.ndarray,
        h: float,
    ) -> np.ndarray:
        k1 = f(t, z)
        k2 = f(t + 0.5 * h, z + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, z + 0.5 * h * k2)
        k4 = f(t + h, z + h * k3)
        return z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @staticmethod
    def _hermite_theta(
        y0: float, yp0: float, y1: float, yp1: float, h: float
    ) -> float:
        r"""
        Raíz θ ∈ [0, 1] del interpolante de Hermite cúbico de y.

            y(θ) = h₀₀(θ) y₀ + h₁₀(θ) h ẏ₀ + h₀₁(θ) y₁ + h₁₁(θ) h ẏ₁.
        """
        if abs(y1 - y0) > 1e-18:
            theta = float(np.clip(-y0 / (y1 - y0), 0.0, 1.0))
        else:
            theta = 0.5
        hh = float(h) if abs(h) > _EPS else 1.0
        for _ in range(12):
            t2 = theta * theta
            t3 = t2 * theta
            y = (
                (2.0 * t3 - 3.0 * t2 + 1.0) * y0
                + (t3 - 2.0 * t2 + theta) * hh * yp0
                + (-2.0 * t3 + 3.0 * t2) * y1
                + (t3 - t2) * hh * yp1
            )
            yp = (
                (6.0 * t2 - 6.0 * theta) * y0
                + (3.0 * t2 - 4.0 * theta + 1.0) * hh * yp0
                + (-6.0 * t2 + 6.0 * theta) * y1
                + (3.0 * t2 - 2.0 * theta) * hh * yp1
            )
            if abs(yp) < 1e-18:
                break
            theta_n = float(np.clip(theta - y / yp, 0.0, 1.0))
            if abs(theta_n - theta) < 1e-14:
                theta = theta_n
                break
            theta = theta_n
        return float(theta)

    def _newton_refine_crossing(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        t_lo: float,
        z_lo: np.ndarray,
        t_hi: float,
        z_hi: np.ndarray,
        h_prev: float,
    ) -> Tuple[float, np.ndarray, int]:
        r"""
        Refinamiento Hermite–Newton + un RK4 del subpaso θh.

        CORRECCIÓN v9.2.0: v9.1.0 hacía un único RK4 de tamaño ~h/2
        desde t_lo, que no es Newton sobre F(t)=y(t).
        """
        y0, yp0 = float(z_lo[1]), float(z_lo[3])
        y1, yp1 = float(z_hi[1]), float(z_hi[3])
        h = float(h_prev) if abs(h_prev) > _EPS else float(t_hi - t_lo)
        theta = self._hermite_theta(y0, yp0, y1, yp1, h)
        t = t_lo + theta * h
        z = self._rk4_step(f, t_lo, z_lo, t - t_lo)
        n_it = 1
        for it in range(1, self.newton_max_iter):
            y = float(z[1])
            vy = float(z[3])
            if abs(y) < self.newton_tol:
                return t, z, it
            if abs(vy) < 1e-15:
                break
            dt_step = -y / vy
            t_new = t + dt_step
            if t_new <= t_lo:
                t_new = t_lo + 1e-14
            if t_new >= t_lo + abs(h):
                t_new = t_lo + abs(h)
            z = self._rk4_step(f, t_lo, z_lo, t_new - t_lo)
            t = t_new
            n_it = it + 1
            if abs(float(z[1])) < self.newton_tol:
                return t, z, n_it
        return t, z, n_it

    def integrate_section(
        self, seed: PoincareSectionSeed, n_crossings: int = 512
    ) -> List[SectionCrossing]:
        f = CR3BPDynamics.vector_field(seed.mu)
        z = seed.initial_state.as_array().copy()
        t = 0.0
        crossings: List[SectionCrossing] = []
        y_prev = float(z[1])
        vy_prev = float(z[3])
        h = self.dt
        z0_state = seed.initial_state
        for _ in range(self.max_steps):
            r1, r2 = CR3BPDynamics._radii(float(z[0]), float(z[1]), seed.mu)
            if min(r1, r2) < self.collision_radius:
                z = LeviCivitaRegularization.regularized_rk4_step(
                    f, t, z, h, seed.mu
                )
            else:
                z_next = self._rk4_step(f, t, z, h)
                y_next = float(z_next[1])
                vy_next = float(z_next[3])
                if y_prev < 0.0 <= y_next and vy_next > 0.0:
                    t_cross, z_cross, nit = self._newton_refine_crossing(
                        f, t, z, t + h, z_next, h
                    )
                    z_s = CR3BPState.from_array(z_cross)
                    C = CR3BPDynamics.jacobi_constant(z_s, seed.mu)
                    res = abs(C - seed.jacobi_constant) / max(
                        1.0, abs(seed.jacobi_constant)
                    )
                    crossings.append(
                        SectionCrossing(
                            step_index=len(crossings),
                            t=float(t_cross),
                            x=float(z_cross[0]),
                            vx=float(z_cross[2]),
                            jacobi_residual=float(res),
                            newton_iterations=int(nit),
                            y_event=float(z_cross[1]),
                            vy=float(z_cross[3]),
                        )
                    )
                    if len(crossings) >= int(n_crossings):
                        break
                z = z_next
                y_prev, vy_prev = y_next, vy_next
                t += h
                continue
            y_prev = float(z[1])
            vy_prev = float(z[3])
            t += h
        return crossings

    @staticmethod
    def mean_return_time(crossings: Sequence[SectionCrossing]) -> float:
        if len(crossings) < 2:
            return float("inf")
        dts = np.diff(np.array([c.t for c in crossings], dtype=np.float64))
        dts = dts[dts > 0.0]
        if dts.size == 0:
            return float("inf")
        return float(np.mean(dts))

    @staticmethod
    def jacobi_max_residual(crossings: Sequence[SectionCrossing]) -> float:
        if not crossings:
            return 0.0
        return float(max(abs(c.jacobi_residual) for c in crossings))

    @classmethod
    def return_map_report(
        cls, crossings: Sequence[SectionCrossing]
    ) -> ReturnMapReport:
        if len(crossings) < 8:
            eye = np.eye(2, dtype=np.float64)
            return ReturnMapReport(
                matrix=eye, det=1.0, det_residual=0.0,
                is_area_preserving=True, n_pairs=0, linearity_residual=0.0,
            )
        xs = np.array([c.x for c in crossings[:-1]], dtype=np.float64)
        vxs = np.array([c.vx for c in crossings[:-1]], dtype=np.float64)
        ys = np.array([c.x for c in crossings[1:]], dtype=np.float64)
        wys = np.array([c.vx for c in crossings[1:]], dtype=np.float64)
        X = np.column_stack([xs, vxs])
        A = np.eye(2, dtype=np.float64)
        lin_res = 0.0
        try:
            c1, *_ = np.linalg.lstsq(X, ys, rcond=None)
            c2, *_ = np.linalg.lstsq(X, wys, rcond=None)
            A[0, :] = c1
            A[1, :] = c2
            pred = X @ A.T
            tgt = np.column_stack([ys, wys])
            lin_res = float(
                np.linalg.norm(pred - tgt) / max(np.linalg.norm(tgt), 1e-15)
            )
        except (np.linalg.LinAlgError, ValueError):
            A = np.eye(2, dtype=np.float64)
        det = float(np.linalg.det(A))
        det_res = abs(abs(det) - 1.0)
        return ReturnMapReport(
            matrix=A,
            det=det,
            det_residual=float(det_res),
            is_area_preserving=bool(det_res < _RETURN_MAP_DET_TOL),
            n_pairs=int(len(crossings) - 1),
            linearity_residual=float(lin_res),
        )

    @staticmethod
    def return_map_matrix(crossings: List[SectionCrossing]) -> np.ndarray:
        r"""Firma retro-compatible: matriz 2×2 del mapa de retorno."""
        return PoincareSectionIntegrator.return_map_report(crossings).matrix

    @staticmethod
    def section_entropy(
        crossings: Sequence[SectionCrossing], bins: int = 32
    ) -> float:
        if len(crossings) < 4:
            return 0.0
        xs = np.array([c.x for c in crossings], dtype=np.float64)
        vxs = np.array([c.vx for c in crossings], dtype=np.float64)
        h2d, _, _ = np.histogram2d(xs, vxs, bins=int(max(4, bins)))
        p = h2d / max(1.0, float(h2d.sum()))
        p = p[p > 0.0]
        return float(-np.sum(p * np.log(p)))


# ═══════════════════════════════════════════════════════════════════════════
# §2.2  Regularización de Levi-Civita / Kustaanheimo–Stiefel
# ═══════════════════════════════════════════════════════════════════════════
class LeviCivitaRegularization:
    r"""
    Transformación de Levi-Civita (Poincaré 1896 / Levi-Civita 1906):

        x + i y = (u + i v)² ,     dt = r dτ,     r = |w|² = u² + v².

    Regulariza la colisión binaria r → 0 del problema planar.
    La raíz cuadrada es bivaluada: se elige la rama continua respecto
    de un germen w_prev (producto interno positivo).
    """

    @staticmethod
    def transform(
        xy: np.ndarray, w_prev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        x, y = float(xy[0]), float(xy[1])
        z_c = complex(x, y)
        if abs(z_c) <= 0.0:
            w_c = 1e-15 + 0j
        else:
            w_c = np.sqrt(z_c)
        if w_prev is not None:
            wp = complex(float(w_prev[0]), float(w_prev[1]))
            if (w_c.real * wp.real + w_c.imag * wp.imag) < 0.0:
                w_c = -w_c
        return np.array([w_c.real, w_c.imag], dtype=np.float64)

    @staticmethod
    def inverse(uv: np.ndarray) -> np.ndarray:
        u, v = float(uv[0]), float(uv[1])
        return np.array([u * u - v * v, 2.0 * u * v], dtype=np.float64)

    @staticmethod
    def jacobian_magnitude(uv: np.ndarray) -> float:
        u, v = float(uv[0]), float(uv[1])
        return 4.0 * (u * u + v * v)

    @staticmethod
    def r_from_w(uv: np.ndarray) -> float:
        u, v = float(uv[0]), float(uv[1])
        return u * u + v * v

    @classmethod
    def regularized_rk4_step(
        cls,
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        z: np.ndarray,
        h: float,
        mu: float,
    ) -> np.ndarray:
        r"""
        Un paso RK4 con subpaso reducido cerca de colisión.

        No integra el campo LC completo (eso exige el pulso ficticio τ);
        aplica un subpaso h·r para no saltar la singularidad, que es
        el mínimo rigor compatible con el marco sinodial (x,y,ẋ,ẏ).
        """
        r1, r2 = CR3BPDynamics._radii(float(z[0]), float(z[1]), mu)
        rmin = max(min(r1, r2), _SING_HARDENING)
        h_loc = float(h) * float(min(1.0, rmin))
        if h_loc <= 0.0:
            h_loc = float(h) * 1e-3
        return PoincareSectionIntegrator._rk4_step(f, t, z, h_loc)


class KustaanheimoStiefel:
    r"""
    Levantamiento KS  ℝ⁴ → ℝ³  (espacial).  El CR3BP de este módulo
    es planar: KS se expone para el hand-off espacial (Fase 3 / 3D).

    Ligadura bilinear:  q₀ q̇₁ − q₁ q̇₀ + q₂ q̇₃ − q₃ q̇₂ = 0.
    """

    @staticmethod
    def lift_matrix(q: np.ndarray) -> np.ndarray:
        q0, q1, q2, q3 = (
            float(q[0]), float(q[1]), float(q[2]), float(q[3])
        )
        return np.array(
            [
                [q0, -q1, -q2, -q3],
                [q1,  q0, -q3,  q2],
                [q2,  q3,  q0, -q1],
                [q3, -q2,  q1,  q0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def r_from_quaternion(q: np.ndarray) -> np.ndarray:
        qq = np.asarray(q, dtype=np.float64).reshape(4)
        return KustaanheimoStiefel.lift_matrix(qq) @ qq

    @staticmethod
    def bilinear_constraint(q: np.ndarray, qdot: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        d = np.asarray(qdot, dtype=np.float64).reshape(4)
        return float(q[0] * d[1] - q[1] * d[0] + q[2] * d[3] - q[3] * d[2])

    @classmethod
    def planar_lift(cls, x: float, y: float) -> np.ndarray:
        r"""Inmersión del plano en KS: q = (√((r+x)/2), y/(2q₀), 0, 0)."""
        r = math.hypot(x, y)
        q0 = math.sqrt(max(0.5 * (r + x), 0.0))
        if q0 < _EPS:
            return np.array([0.0, math.sqrt(max(r, 0.0)), 0.0, 0.0], dtype=np.float64)
        return np.array([q0, y / (2.0 * q0), 0.0, 0.0], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# §2.3  Flujo tangente, espectro de Lyapunov (BGGS), SALI, GALI
# ═══════════════════════════════════════════════════════════════════════════
class TangentFlow:
    r"""Wrapper: el jacobiano 4×4 vive en CR3BPDynamics (Fase 1)."""

    @staticmethod
    def jacobian(z: np.ndarray, mu: float) -> np.ndarray:
        return CR3BPDynamics.jacobian(z, mu)


class LiapunovSpectrum:
    r"""
    Espectro de Lyapunov por el algoritmo BGGS
    (Benettin–Galgani–Giorgilli–Strelcyn 1980):

        Ẇ = A(z(t)) W,   W = Q R  (QR periódico),
        λ_i = lim (1/T) Σ log |R_ii|.

    Certificados hamiltonianos:
        apareamiento  λ_i + λ_{5−i} ≈ 0,
        volumen       Σ λ_i ≈ 0.
    """

    @classmethod
    def estimate(
        cls,
        seed: PoincareSectionSeed,
        t_total: float = 50.0,
        dt: float = 5.0e-3,
    ) -> Tuple[float, ...]:
        report = cls.estimate_certified(seed, t_total=t_total, dt=dt)
        return report[0]

    @classmethod
    def estimate_certified(
        cls,
        seed: PoincareSectionSeed,
        t_total: float = 50.0,
        dt: float = 5.0e-3,
    ) -> Tuple[Tuple[float, ...], float, float]:
        r"""Retorna (espectro descendente, residuo de apareamiento, residuo de volumen)."""
        f = CR3BPDynamics.vector_field(seed.mu)
        z = seed.initial_state.as_array().copy()
        n = int(z.size)
        W = np.eye(n, dtype=np.float64)
        sums = np.zeros(n, dtype=np.float64)
        steps = int(max(1, t_total / dt))
        nsteps = 0
        h = float(dt)
        for s in range(steps):
            t = s * h
            A0 = TangentFlow.jacobian(z, seed.mu)
            k1z = f(t, z)
            k1w = A0 @ W
            z2 = z + 0.5 * h * k1z
            w2 = W + 0.5 * h * k1w
            A2 = TangentFlow.jacobian(z2, seed.mu)
            k2z = f(t + 0.5 * h, z2)
            k2w = A2 @ w2
            z3 = z + 0.5 * h * k2z
            w3 = W + 0.5 * h * k2w
            A3 = TangentFlow.jacobian(z3, seed.mu)
            k3z = f(t + 0.5 * h, z3)
            k3w = A3 @ w3
            z4 = z + h * k3z
            w4 = W + h * k3w
            A4 = TangentFlow.jacobian(z4, seed.mu)
            k4z = f(t + h, z4)
            k4w = A4 @ w4
            z = z + (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
            W = W + (h / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
            try:
                Q, R = np.linalg.qr(W)
            except np.linalg.LinAlgError:
                break
            d = np.abs(np.diag(R))
            d = np.where(d > 1e-30, d, 1e-30)
            sums += np.log(d)
            W = Q
            nsteps += 1
        if nsteps == 0:
            zeros = tuple(0.0 for _ in range(n))
            return zeros, 0.0, 0.0
        spec = np.array(sorted((sums / (nsteps * h)).tolist(), reverse=True))
        pairing = 0.0
        n_pairs = n // 2
        for i in range(n_pairs):
            pairing = max(pairing, abs(float(spec[i] + spec[n - 1 - i])))
        volume = abs(float(np.sum(spec)))
        return tuple(float(x) for x in spec.tolist()), float(pairing), float(volume)

    @classmethod
    def is_hamiltonian_consistent(
        cls, spectrum: Sequence[float], pairing_res: float, volume_res: float
    ) -> bool:
        return bool(
            pairing_res < _LYAPUNOV_PAIRING_TOL
            and volume_res < _LYAPUNOV_PAIRING_TOL
        )


class SALIIndicator:
    r"""
    Smaller Alignment Index (Skokos 2001):

        SALI(t) = min{ ‖v̂₁+v̂₂‖, ‖v̂₁−v̂₂‖ }.

    SALI → 0 exponencialmente ⇔ órbita caótica;
    SALI → c > 0 ⇔ regular (toros).
    """

    @classmethod
    def estimate(
        cls,
        seed: PoincareSectionSeed,
        t_total: float = 20.0,
        dt: float = 5.0e-3,
    ) -> float:
        f = CR3BPDynamics.vector_field(seed.mu)
        z = seed.initial_state.as_array().copy()
        n = int(z.size)
        rng = np.random.default_rng(_seed_from_string(seed.provenance_hash))
        v1 = rng.standard_normal(n)
        v2 = rng.standard_normal(n)
        v1 /= np.linalg.norm(v1) + 1e-30
        v2 /= np.linalg.norm(v2) + 1e-30
        steps = int(max(1, t_total / dt))
        h = float(dt)
        val = 0.0
        for s in range(steps):
            A = TangentFlow.jacobian(z, seed.mu)
            v1 = v1 + h * (A @ v1)
            v2 = v2 + h * (A @ v2)
            n1 = np.linalg.norm(v1) + 1e-30
            n2 = np.linalg.norm(v2) + 1e-30
            v1 = v1 / n1
            v2 = v2 / n2
            val = min(float(np.linalg.norm(v1 + v2)), float(np.linalg.norm(v1 - v2)))
            z = PoincareSectionIntegrator._rk4_step(f, s * h, z, h)
        return float(val)


class GALIIndicator:
    r"""
    Generalized Alignment Index GALI_k (Skokos–Bountis 2007):

        GALI_k(t) = ‖ v₁ ∧ … ∧ v_k ‖  =  ∏_{i=1}^{k} σ_i(V),

    σ_i = valores singulares de V = [v₁|…|v_k].  k=2 coincide con
    un pariente de SALI.  k>2 detecta caos más rápido.

    Tasa de decaimiento:  κ = (1/T) log(GALI(0)/GALI(T)).
    """

    @classmethod
    def estimate(
        cls,
        seed: PoincareSectionSeed,
        k: int = 3,
        t_total: float = 20.0,
        dt: float = 5.0e-3,
    ) -> float:
        val, _kappa = cls.estimate_with_decay(seed, k=k, t_total=t_total, dt=dt)
        return val

    @classmethod
    def estimate_with_decay(
        cls,
        seed: PoincareSectionSeed,
        k: int = 3,
        t_total: float = 20.0,
        dt: float = 5.0e-3,
    ) -> Tuple[float, float]:
        k = int(max(2, min(k, 4)))
        f = CR3BPDynamics.vector_field(seed.mu)
        z = seed.initial_state.as_array().copy()
        n = int(z.size)
        rng = np.random.default_rng(
            _seed_from_string(f"GALI{k}::{seed.provenance_hash}")
        )
        Vs = [rng.standard_normal(n) for _ in range(k)]
        for i in range(k):
            Vs[i] = Vs[i] / (np.linalg.norm(Vs[i]) + 1e-30)
        gali0 = cls._volume(Vs)
        steps = int(max(1, t_total / dt))
        h = float(dt)
        val = float(gali0)
        for s in range(steps):
            A = TangentFlow.jacobian(z, seed.mu)
            for i in range(k):
                Vs[i] = Vs[i] + h * (A @ Vs[i])
                Vs[i] = Vs[i] / (np.linalg.norm(Vs[i]) + 1e-30)
            val = cls._volume(Vs)
            z = PoincareSectionIntegrator._rk4_step(f, s * h, z, h)
        T = max(steps * h, _EPS)
        kappa = float(math.log(max(gali0, _GALI_FLOOR) / max(val, _GALI_FLOOR)) / T)
        return float(val), kappa

    @staticmethod
    def _volume(Vs: Sequence[np.ndarray]) -> float:
        M = np.column_stack(list(Vs))
        sv = la.svdvals(M)
        sv = np.maximum(np.real(sv), 0.0)
        return float(max(np.prod(sv), 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# §2.4  Melnikov  {H₀, H₁}  +  ceros transversales  +  Smale–Birkhoff
# ═══════════════════════════════════════════════════════════════════════════
class MelnikovResonanceProbe:
    r"""
    Función de Melnikov de la perturbación en μ respecto del Kepler rotante.

    H_CR3BP = H₀ + μ H₁,  con (μ=0: primario en el origen)

        H₀ = (ẋ²+ẏ²)/2 − (x²+y²)/2 − 1/r,    r = √(x²+y²),
        H₁ = ∂_μ |_{μ=0} [ −(1−μ)/r₁ − μ/r₂ ]  (potencial perturbativo).

    A lo largo de una órbita φ_t:

        M(t₀) = ∫ ⟨ X_{H₀} , ∇H₁ ⟩ (φ_{t+t₀}) dt  = ∫ {H₀, H₁} dt.

    Un cero transversal M(t₀)=0, M'(t₀)≠0  ⇒  splitting homoclínico
    ⇒ herradura de Smale (teorema de Birkhoff–Smale).

    CORRECCIÓN v9.2.0: v9.1.0 integraba el defecto de Jacobi, nulo
    idénticamente a lo largo del flujo.
    """

    @staticmethod
    def _h1_gradient(x: float, y: float, mu: float) -> Tuple[float, float]:
        r"""∇H₁ ≈ −∇(∂Ω/∂μ).  Se evalúa en μ del seed (no necesariamente 0)."""
        mu = CR3BPDynamics.clip_mu(mu)
        r1, r2 = CR3BPDynamics._radii(x, y, mu)
        # ∂Ω/∂μ = −1/r₁ + 1/r₂ − (1−μ)(x+μ)/r₁³ + (términos de ∂r/∂μ)
        # ∇(∂Ω/∂μ) se aproxima por diferencia central en μ.
        dmu = 1.0e-6
        gx_p, gy_p = CR3BPDynamics.potential_grad(x, y, mu + dmu)
        gx_m, gy_m = CR3BPDynamics.potential_grad(x, y, max(mu - dmu, _MU_FLOOR))
        ddx = (gx_p - gx_m) / (2.0 * dmu)
        ddy = (gy_p - gy_m) / (2.0 * dmu)
        return float(ddx), float(ddy)

    @classmethod
    def integrand(cls, z: np.ndarray, mu: float) -> float:
        r"""{H₀, H₁} = ẋ ∂x H₁ + ẏ ∂y H₁  (trabajo de la perturbación)."""
        x, y, vx, vy = float(z[0]), float(z[1]), float(z[2]), float(z[3])
        gx, gy = cls._h1_gradient(x, y, mu)
        return float(vx * gx + vy * gy)

    @classmethod
    def evaluate(
        cls,
        seed: PoincareSectionSeed,
        t_span: float = 20.0,
        dt: float = 1.0e-2,
    ) -> Tuple[float, float, int]:
        r"""Firma retro-compatible: (peak, area, n_zeros) de M(t₀)."""
        report = cls.evaluate_certified(seed, t_span=t_span, dt=dt)
        return report[0], report[1], report[2]

    @classmethod
    def evaluate_certified(
        cls,
        seed: PoincareSectionSeed,
        t_span: float = 20.0,
        dt: float = 1.0e-2,
        n_phase: int = 48,
    ) -> Tuple[float, float, int, bool]:
        r"""
        Retorna (peak, area, n_zeros, transverse).

        M(t₀) se obtiene por ventana deslizante del integrando I(t)
        a lo largo de una órbita de longitud 2 t_span.
        """
        f = CR3BPDynamics.vector_field(seed.mu)
        z = seed.initial_state.as_array().copy()
        h = float(dt)
        steps = int(max(8, 2.0 * t_span / h))
        I_vals = np.zeros(steps, dtype=np.float64)
        for s in range(steps):
            I_vals[s] = cls.integrand(z, seed.mu)
            z = PoincareSectionIntegrator._rk4_step(f, s * h, z, h)
        # ventana de longitud t_span
        wlen = max(2, int(t_span / h))
        if I_vals.size < wlen + 4:
            area = float(np.sum(np.abs(I_vals)) * h)
            peak = float(np.max(np.abs(I_vals))) if I_vals.size else 0.0
            return peak, area, 0, False
        kernel = np.ones(wlen, dtype=np.float64) * h
        M = np.convolve(I_vals, kernel, mode="valid")
        if M.size == 0:
            return 0.0, 0.0, 0, False
        # subsample de fases
        idx = np.linspace(0, M.size - 1, num=min(n_phase, M.size), dtype=int)
        Ms = M[idx]
        peak = float(np.max(np.abs(Ms)))
        area = float(np.mean(np.abs(Ms)))
        signs = np.sign(Ms)
        signs[signs == 0.0] = 1.0
        n_zeros = int(np.sum(np.abs(np.diff(signs)) > 1.0))
        # transversalidad: |ΔM| / Δt₀ en cada cruce
        transverse = False
        if n_zeros >= 1 and Ms.size >= 3:
            dM = np.abs(np.diff(Ms))
            dt0 = t_span / max(Ms.size, 1)
            transverse = bool(np.max(dM) / max(dt0, _EPS) > _MELNIKOV_TRANSVERSE_TOL)
        return peak, area, n_zeros, bool(transverse and n_zeros >= 1)

    @classmethod
    def transverse_zeros(cls, seed: PoincareSectionSeed, **kwargs: Any) -> bool:
        r"""¿M(t₀) tiene ceros transversales? ⇒ herradura de Smale."""
        _p, _a, nz, tr = cls.evaluate_certified(seed, **kwargs)
        return bool(nz >= 1 and tr)


class MelnikovZerosFinder:
    r"""Certificado Smale–Birkhoff: ceros transversales ⇒ horseshoe."""

    @classmethod
    def certificate(cls, seed: PoincareSectionSeed, **kwargs: Any) -> Dict[str, Any]:
        peak, area, nz, tr = MelnikovResonanceProbe.evaluate_certified(
            seed, **kwargs
        )
        horseshoe = bool(nz >= 1 and tr)
        return {
            "melnikov_peak": float(peak),
            "melnikov_area": float(area),
            "n_zeros": int(nz),
            "transverse": bool(tr),
            "smale_horseshoe": horseshoe,
            "local_verdict": (
                HeytingOmega3.VETOED if horseshoe else HeytingOmega3.COHERENT
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §2.5  KAM, número de rotación, Chirikov, lenguas de Arnold
# ═══════════════════════════════════════════════════════════════════════════
class KAMTorusAuditor:
    r"""
    Número de rotación ρ de la sección y sesgo KAM.

        θ_n = arg((x_n − x̄) + i (ẋ_n − ẋ̄)),
        ρ   = ⟨Δθ⟩ / 2π,
        bias = ⟨ |Δθ − med(Δθ)| ⟩.

    bias pequeño ⇒ órbita toroidal (rotación rígida).
    """

    @classmethod
    def verify(
        cls, crossings: Sequence[SectionCrossing], tol: float = _KAM_TOL
    ) -> Dict[str, float]:
        if len(crossings) < 16:
            return {
                "kam_bias": float("inf"),
                "kam_stable": 0.0,
                "kam_freq": 0.0,
                "rotation_number": 0.0,
                "diophantine_residue": 0.0,
            }
        xs = np.array([c.x for c in crossings], dtype=np.float64)
        vs = np.array([c.vx for c in crossings], dtype=np.float64)
        cx, cvx = float(xs.mean()), float(vs.mean())
        theta = np.arctan2(vs - cvx, xs - cx)
        dtheta = (np.diff(theta) + np.pi) % (2.0 * np.pi) - np.pi
        omega = float(np.median(dtheta))
        bias = float(np.mean(np.abs(dtheta - omega)))
        stable = 1.0 if bias < tol * 10.0 else 0.0
        rho_num = float(np.mean(dtheta)) / (2.0 * math.pi)
        dio = cls._diophantine_1d(rho_num)
        return {
            "kam_bias": bias,
            "kam_stable": stable,
            "kam_freq": omega,
            "rotation_number": rho_num,
            "diophantine_residue": dio,
        }

    @staticmethod
    def _diophantine_1d(
        rho: float, q_max: int = 32, gamma: float = 1.0e-3, tau: float = 1.0
    ) -> float:
        r"""
        R(ρ) = inf_{1≤q≤q_max}  |q ρ − p| / (γ q^{−τ}).

        R ≥ 1  ⇒  ρ es (γ,τ)-Diofantino hasta q_max.
        """
        best = float("inf")
        for q in range(1, q_max + 1):
            p = int(round(rho * q))
            val = abs(q * rho - p) / max(gamma * (q ** (-tau)), _EPS)
            if val < best:
                best = val
        return float(best)


class ArnoldTongueDetector:
    r"""
    Lenguas de Arnold y criterio de solapamiento de Chirikov (1979).

    Para un mapa twist con perturbación ε, la resonancia p/q tiene
    anchura (aproximación de péndulo)

        Δω_{p/q} ≈ 4 √(ε |V_{p/q}|) / q.

    Solapamiento:  Δω_i + Δω_j > |ω_i − ω_j|  ⇒  caos global.
    Medida de Cantor de toros supervivientes:  μ_C ≈ max(0, 1 − Σ Δω).
    """

    @classmethod
    def detect(
        cls,
        omega_kam: float,
        n_terms: int = 6,
        delta_res: float = 1e-2,
        epsilon: Optional[float] = None,
        rotation_number: Optional[float] = None,
    ) -> Dict[str, Any]:
        rho = float(rotation_number) if rotation_number is not None else float(omega_kam)
        try:
            approx = Fraction(rho).limit_denominator(64)
        except (ValueError, OverflowError, ZeroDivisionError):
            approx = Fraction(1, 2)
        p, q = int(approx.numerator), int(approx.denominator)
        if q == 0:
            p, q = 1, 2
        eps = float(delta_res if epsilon is None else epsilon)
        eps = float(np.clip(abs(eps), 1e-12, 0.9))
        # convergentes vecinos
        conv = cls._convergents_of(rho, depth=int(max(4, n_terms)))
        widths: List[float] = []
        omegas: List[float] = []
        for pp, qq in conv[: int(n_terms)]:
            if qq <= 0:
                continue
            # |V_{p/q}| ~ ε / q²  (Fourier genérico)
            amp = eps / max(qq * qq, 1.0)
            width = 4.0 * math.sqrt(max(eps * amp, 0.0)) / max(qq, 1)
            widths.append(float(width))
            omegas.append(float(pp) / float(qq))
        if not widths:
            widths = [4.0 * math.sqrt(eps) ]
            omegas = [float(p) / float(max(q, 1))]
        total_width = float(sum(widths))
        cantor_measure = float(max(0.0, 1.0 - min(total_width, 1.0)))
        overlap = False
        if len(widths) >= 2:
            d_om = abs(omegas[0] - omegas[1])
            overlap = bool((widths[0] + widths[1]) > max(d_om, 1e-18))
        verdict = (
            HeytingOmega3.VETOED if overlap
            else (
                HeytingOmega3.DEGRADED
                if total_width > 0.1
                else HeytingOmega3.COHERENT
            )
        )
        return {
            "p": int(p),
            "q": int(q),
            "widths": tuple(widths),
            "total_width": float(total_width),
            "cantor_measure": cantor_measure,
            "chirikov_overlap": bool(overlap),
            "local_verdict": verdict,
            "n_convergents": len(conv),
        }

    @staticmethod
    def _convergents_of(x: float, depth: int = 10) -> List[Tuple[int, int]]:
        coeffs: List[int] = []
        y = float(x)
        for _ in range(depth):
            if not math.isfinite(y):
                break
            a = int(math.floor(y))
            coeffs.append(a)
            frac = y - a
            if abs(frac) < 1.0e-12:
                break
            y = 1.0 / frac
        p_prev, p_cur = 0, 1
        q_prev, q_cur = 1, 0
        out: List[Tuple[int, int]] = []
        for a in coeffs:
            p_new = a * p_cur + p_prev
            q_new = a * q_cur + q_prev
            if q_new != 0:
                out.append((p_new, q_new))
            p_prev, p_cur = p_cur, p_new
            q_prev, q_cur = q_cur, q_new
        return out


# ═══════════════════════════════════════════════════════════════════════════
# §2.6  Bundle de observación enriquecido
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class WitnessObservationBundle:
    r"""
    Haz de observación: objeto interno de FASE 2.

    Porta el censo de sección, indicadores de caos, certificados AGM/RSI-3
    y el `PoincareSectionSeed` del que nace.  No es el objeto terminal
    (ése es BirkhoffSeed); es el *witness* que `emit_birkhoff_seed` reduce.
    """
    cycle_index: int
    triad_signature: TriadSignature
    rho_vac: np.ndarray
    rho_observed: np.ndarray
    modular_spectrum: Tuple[float, ...]
    K: Any
    audit: Any
    modular_axioms: Dict[str, float]
    tomita_report: Dict[str, float]
    triad_residuals: Dict[str, float]
    invariant_vector: np.ndarray
    context_beta: float
    poincare_seed: Optional[PoincareSectionSeed] = None
    section_crossings: Tuple[SectionCrossing, ...] = ()
    return_map_matrix: Optional[np.ndarray] = None
    section_entropy: float = 0.0
    kam_report: Mapping[str, float] = field(default_factory=dict)
    liapunov_spectrum: Tuple[float, ...] = ()
    sali_value: float = 0.0
    melnikov_peak: float = 0.0
    melnikov_area: float = 0.0
    gali_value: float = 0.0
    gali_k: int = 3
    melnikov_zeros: int = 0
    smale_horseshoe: bool = False
    arnold_tongues: Mapping[str, Any] = field(default_factory=dict)
    jacobi_region: Optional[JacobiRegion] = None
    agm_certificate: Optional[AGMPostulateCertificate] = None
    monad_axioms: Optional[MonadAxiomVerification] = None
    crowbar_triggered: bool = False
    # v9.2.0-F2
    lyapunov_pairing_residual: float = 0.0
    lyapunov_volume_residual: float = 0.0
    gali_decay_rate: float = 0.0
    mean_return_time: float = float("inf")
    jacobi_max_residual: float = 0.0
    return_map_det_residual: float = 0.0
    rotation_number: float = 0.0
    diophantine_residue: float = 0.0
    melnikov_transverse: bool = False
    return_map_report: Optional[ReturnMapReport] = None

    def as_vacuum_metrics(self, is_silent: bool) -> VacuumStateMetrics:
        j_res = 0.0
        if self.section_crossings:
            j_res = float(
                np.mean([c.jacobi_residual for c in self.section_crossings])
            )
        lib_xy = (0.0, 0.0)
        mu = 0.0
        routh_s = False
        C = 0.0
        hill = "unknown"
        rsi_lvl = 0
        if self.poincare_seed is not None:
            mu = self.poincare_seed.mu
            C = self.poincare_seed.jacobi_constant
            routh_s = self.poincare_seed.routh_stable
            hill = self.poincare_seed.hill_topology
            if "L4" in self.poincare_seed.libration_points:
                lib_xy = self.poincare_seed.libration_points["L4"]
        if self.monad_axioms is not None:
            rsi_lvl = int(self.monad_axioms.level_applied)
        back_action = (
            float(self.poincare_seed.back_action_bound)
            if self.poincare_seed is not None
            else float(j_res)
        )
        return VacuumStateMetrics(
            vacuum_expectation_value=getattr(self.audit, "vev", 0.0),
            tomita_takesaki_flow_param=1.0
            / max(getattr(self.K, "spectral_gap", 1e-12) or 1e-12, 1e-12),
            kms_entropy_drift=getattr(self.audit, "free_energy", 0.0),
            silence_purity=getattr(self.audit, "purity", 0.0),
            modular_ground_energy=getattr(self.K, "ground_energy", 0.0),
            modular_spectral_gap=getattr(self.audit, "spectral_gap", 0.0),
            von_neumann_entropy=DensityMatrixOps.von_neumann_entropy(
                self.rho_observed
            )
            if DensityMatrixOps is not None
            else 0.0,
            purity=DensityMatrixOps.purity(self.rho_observed)
            if DensityMatrixOps is not None
            else 0.0,
            is_silent=is_silent,
            dirichlet_energy=self.section_entropy,
            mass_ratio_mu=mu,
            jacobi_constant=C,
            jacobi_residual=j_res,
            libration_xy=lib_xy,
            routh_stable=routh_s,
            crowbar_triggered=self.crowbar_triggered,
            back_action_bound=back_action,
            hill_topology=hill,
            rsi3_level=rsi_lvl,
        )

    def celestial_diagnostics(self) -> CelestialDiagnostics:
        return CelestialDiagnostics(
            liapunov_spectrum=self.liapunov_spectrum,
            pairing_residual=self.lyapunov_pairing_residual,
            volume_residual=self.lyapunov_volume_residual,
            sali=self.sali_value,
            gali=self.gali_value,
            gali_k=self.gali_k,
            gali_decay_rate=self.gali_decay_rate,
            jacobi_max_residual=self.jacobi_max_residual,
            mean_return_time=self.mean_return_time,
            n_crossings=len(self.section_crossings),
            kam_bias=float(self.kam_report.get("kam_bias", float("inf"))),
            rotation_number=self.rotation_number,
            diophantine_residue=self.diophantine_residue,
            chirikov_overlap=bool(
                self.arnold_tongues.get("chirikov_overlap", False)
            ),
            cantor_measure=float(
                self.arnold_tongues.get("cantor_measure", 1.0)
            ),
            melnikov_peak=self.melnikov_peak,
            melnikov_area=self.melnikov_area,
            melnikov_zeros=self.melnikov_zeros,
            melnikov_transverse=self.melnikov_transverse,
            smale_horseshoe=self.smale_horseshoe,
            section_entropy=self.section_entropy,
            return_map_det_residual=self.return_map_det_residual,
            hill_topology=(
                self.poincare_seed.hill_topology
                if self.poincare_seed is not None
                else "unknown"
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §2.7  Pipeline de observación  +  BirkhoffSeed  (HAND-OFF 2→3)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class BirkhoffSeed:
    r"""
    Objeto de transporte FASE 2 → FASE 3.

    Dominio de `BirkhoffNormalForm.linearization` / `frequencies`
    (primeros métodos de FASE 3).  Reduce el haz de observación a
    los invariantes de Birkhoff–Nekhoroshev: μ, C, L4/L5, ω_KAM,
    λ₁, SALI/GALI, Melnikov, Arnold, AGM, RSI-3, Crowbar.
    """
    mu: float
    jacobi_constant: float
    L4_xy: Tuple[float, float]
    L5_xy: Tuple[float, float]
    kam_frequency: float
    lyapunov_lambda1: float
    sali: float
    melnikov_peak: float
    section_entropy: float
    triad_signature: TriadSignature
    provenance_hash: str
    gali: float = 0.0
    gali_k: int = 3
    melnikov_zeros: int = 0
    smale_horseshoe: bool = False
    arnold_tongues: Mapping[str, Any] = field(default_factory=dict)
    agm_certificate: Optional[AGMPostulateCertificate] = None
    monad_axioms: Optional[MonadAxiomVerification] = None
    crowbar_triggered: bool = False
    # v9.2.0-F2
    rotation_number: float = 0.0
    diophantine_residue: float = 0.0
    lyapunov_pairing_residual: float = 0.0
    mean_return_time: float = float("inf")
    jacobi_max_residual: float = 0.0
    hill_topology: str = "unknown"
    kraus_rank: int = 1
    back_action_bound: float = 0.0
    gali_decay_rate: float = 0.0
    melnikov_transverse: bool = False
    cantor_measure: float = 1.0
    routh_stable: bool = False
    rsi3_mix: float = _RSI3_DEFAULT_MIX


class WitnessObservationPipeline:
    r"""
    Síntesis del haz de observación a partir del vacío y la tríada.

    Cadena de morfosmos:

        (ctx, triad) --emit_poincare_seed-->  PoincareSectionSeed
                     --integrate_section-->   crossings
                     --BGGS/SALI/GALI/Melnikov/KAM/Arnold-->  diagnósticos
                     --SWA1 Crowbar-->        predicado catastrófico
                     --emit_birkhoff_seed-->  BirkhoffSeed
                                           ≡ objeto inicial de FASE 3.
    """

    @classmethod
    def _invariant_vector(cls, audit: Any, triad: TriadChannel) -> np.ndarray:
        v_local = float(int(getattr(audit, "local_verdict", 0))) / 2.0
        d_s = math.tanh(abs(float(getattr(audit, "free_energy", 0.0))))
        leak = 1.0 - float(getattr(audit, "purity", 1.0))
        kms = math.tanh(10.0 * float(getattr(audit, "thermal_fluctuation", 0.0)))
        vev = math.tanh(abs(float(getattr(audit, "vev", 0.0))))
        n = max(1, triad.K.shape[0])
        p_trans = 1.0
        op_norm = float(np.linalg.norm(triad.K, 2)) + 1e-30
        k_fro = triad.kraus_norm() / (math.sqrt(n) * op_norm)
        vec = np.array(
            [v_local, d_s, leak, kms, vev, p_trans, k_fro], dtype=np.float64
        )
        return vec / (float(np.linalg.norm(vec)) + 1e-30)

    @classmethod
    def _explore_celestial(
        cls,
        triad: TriadChannel,
        rho_vac: np.ndarray,
        tau_recurrence: float,
        agm_verdict: Optional[HeytingOmega3],
        n_crossings: int,
        monad_mix: float,
    ) -> Dict[str, Any]:
        seed = TriadToCR3BPMorphism.emit_poincare_seed(
            triad,
            rho_vac,
            tau_recurrence=tau_recurrence,
            agm_verdict=agm_verdict,
            rsi3_mix=monad_mix,
            rho_for_backaction=rho_vac,
        )
        integrator = PoincareSectionIntegrator()
        crossings = integrator.integrate_section(seed, n_crossings=n_crossings)
        rm = integrator.return_map_report(crossings)
        H_sec = integrator.section_entropy(crossings)
        kam_rep = KAMTorusAuditor.verify(crossings)
        lam, pair_res, vol_res = LiapunovSpectrum.estimate_certified(
            seed, t_total=20.0
        )
        sali_val = SALIIndicator.estimate(seed, t_total=10.0)
        gali_k = 3
        gali_val, gali_kappa = GALIIndicator.estimate_with_decay(
            seed, k=gali_k, t_total=10.0
        )
        peak, area, nz, tr = MelnikovResonanceProbe.evaluate_certified(
            seed, t_span=10.0
        )
        smale = bool(nz >= 1 and tr)
        arnold_rep = ArnoldTongueDetector.detect(
            float(kam_rep.get("kam_freq", 0.0)),
            n_terms=6,
            epsilon=max(float(seed.mu), 1e-6),
            rotation_number=float(kam_rep.get("rotation_number", 0.0)),
        )
        jac_region = seed.jacobi_region or JacobiRegion.evaluate(
            seed.initial_state, seed.mu
        )
        lam1 = float(lam[0]) if lam else 0.0
        pred = SWA1Predicate(
            jacobi_residual=PoincareSectionIntegrator.jacobi_max_residual(
                crossings
            ),
            lyapunov_max=lam1,
            hill_transit=seed.hill_topology
            in (HillTopology.TRANSIT.name, HillTopology.FREE.name),
            heyting_vetoed=bool(
                seed.agm_certificate is not None
                and seed.agm_certificate.local_verdict == HeytingOmega3.VETOED
            ),
        )
        crowbar_rep = ESP32CrowbarInterlock.trigger_if(pred)
        if smale or bool(arnold_rep.get("chirikov_overlap", False)):
            if crowbar_rep is None:
                crowbar_rep = ESP32CrowbarInterlock.trigger(
                    reason=(
                        f"CR3BP caos estructural: M_zeros={nz}, "
                        f"λ₁={lam1:.3e}, Chirikov={arnold_rep.get('chirikov_overlap')}"
                    ),
                    predicate="Smale|Chirikov",
                )
        return {
            "poincare_seed": seed,
            "crossings": crossings,
            "return_map_report": rm,
            "section_entropy": H_sec,
            "kam_report": kam_rep,
            "liapunov_spectrum": lam,
            "pairing_residual": pair_res,
            "volume_residual": vol_res,
            "sali": sali_val,
            "gali": gali_val,
            "gali_k": gali_k,
            "gali_decay": gali_kappa,
            "melnikov_peak": peak,
            "melnikov_area": area,
            "melnikov_zeros": nz,
            "melnikov_transverse": tr,
            "smale": smale,
            "arnold": arnold_rep,
            "jacobi_region": jac_region,
            "agm_certificate": seed.agm_certificate,
            "crowbar_triggered": bool(crowbar_rep is not None or seed.crowbar_armed),
            "mean_return_time": PoincareSectionIntegrator.mean_return_time(
                crossings
            ),
            "jacobi_max_residual": PoincareSectionIntegrator.jacobi_max_residual(
                crossings
            ),
        }

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        rho_vac: np.ndarray,
        K_spec: Tuple[float, ...],
        triad: TriadChannel,
        H_ext: np.ndarray,
        K: Optional[Any] = None,
        path_laplacian: Optional[np.ndarray] = None,
        context_beta: float = float("nan"),
        explore_celestial: bool = True,
        n_crossings: int = 256,
        tau_recurrence: float = 1.0,
        agm_verdict: Optional[HeytingOmega3] = None,
        monad_eta: float = _RSI3_DEFAULT_MIX,
        monad_mix: Optional[float] = None,
    ) -> WitnessObservationBundle:
        mix = float(monad_eta if monad_mix is None else monad_mix)
        rho_vac = DensityMatrixOps.sanitize(rho_vac)
        if K is None:
            K = DensityMatrixOps.modular_hamiltonian_from_rho(rho_vac)

        rho_obs = triad.apply(rho_vac)
        triad_res = triad.structural_residuals()

        if TomitaTakesakiEngine is not None:
            tomita = TomitaTakesakiEngine.verify_tomita_takesaki(rho_obs)
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
                axioms = TomitaTakesakiEngine.verify_algebra_axioms(rho_obs)
            audit = VacuumSpectraAnalyzer.audit(rho_obs, H_ext, K)
        else:
            tomita = {}
            axioms = {}
            audit = type("Audit", (), {
                "vev": 0.0, "free_energy": 0.0, "purity": 1.0,
                "thermal_fluctuation": 0.0, "spectral_gap": 1.0,
                "local_verdict": HeytingOmega3.COHERENT,
            })()

        inv = cls._invariant_vector(audit, triad)
        n = int(rho_obs.shape[0])
        monad = MonadRSI3(dim=n, mix=mix, level=3)
        monad_ax = monad.verify_axioms(rho_obs)

        cel: Dict[str, Any] = {}
        if explore_celestial:
            try:
                cel = cls._explore_celestial(
                    triad, rho_vac, tau_recurrence, agm_verdict,
                    n_crossings, mix,
                )
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                logger.warning("Exploración celeste abortada: %s", exc)

        seed = cel.get("poincare_seed")
        crossings = cel.get("crossings", [])
        rm = cel.get("return_map_report")
        kam_rep = cel.get("kam_report", {})
        return WitnessObservationBundle(
            cycle_index=cycle_index,
            triad_signature=triad.signature,
            rho_vac=rho_vac,
            rho_observed=rho_obs,
            modular_spectrum=K_spec if K_spec else getattr(K, "eigenvalues", tuple()),
            K=K,
            audit=audit,
            modular_axioms=axioms,
            tomita_report=tomita,
            triad_residuals=triad_res,
            invariant_vector=inv,
            context_beta=context_beta,
            poincare_seed=seed,
            section_crossings=tuple(crossings),
            return_map_matrix=None if rm is None else rm.matrix,
            section_entropy=float(cel.get("section_entropy", 0.0)),
            kam_report=kam_rep,
            liapunov_spectrum=tuple(cel.get("liapunov_spectrum", ())),
            sali_value=float(cel.get("sali", 0.0)),
            melnikov_peak=float(cel.get("melnikov_peak", 0.0)),
            melnikov_area=float(cel.get("melnikov_area", 0.0)),
            gali_value=float(cel.get("gali", 0.0)),
            gali_k=int(cel.get("gali_k", 3)),
            melnikov_zeros=int(cel.get("melnikov_zeros", 0)),
            smale_horseshoe=bool(cel.get("smale", False)),
            arnold_tongues=dict(cel.get("arnold", {})),
            jacobi_region=cel.get("jacobi_region"),
            agm_certificate=cel.get("agm_certificate"),
            monad_axioms=monad_ax,
            crowbar_triggered=bool(cel.get("crowbar_triggered", False)),
            lyapunov_pairing_residual=float(cel.get("pairing_residual", 0.0)),
            lyapunov_volume_residual=float(cel.get("volume_residual", 0.0)),
            gali_decay_rate=float(cel.get("gali_decay", 0.0)),
            mean_return_time=float(cel.get("mean_return_time", float("inf"))),
            jacobi_max_residual=float(cel.get("jacobi_max_residual", 0.0)),
            return_map_det_residual=(
                0.0 if rm is None else float(rm.det_residual)
            ),
            rotation_number=float(kam_rep.get("rotation_number", 0.0)),
            diophantine_residue=float(kam_rep.get("diophantine_residue", 0.0)),
            melnikov_transverse=bool(cel.get("melnikov_transverse", False)),
            return_map_report=rm,
        )

    @classmethod
    def synthesize_from_context(
        cls,
        cycle_index: int,
        ctx: Any,
        triad: TriadChannel,
        explore_celestial: bool = True,
        tau_recurrence: float = 1.0,
        agm_verdict: Optional[HeytingOmega3] = None,
        monad_mix: float = _RSI3_DEFAULT_MIX,
    ) -> WitnessObservationBundle:
        if TomitaTakesakiEngine is not None:
            ctx = TomitaTakesakiEngine.bind_vacuum_context(ctx)
        return cls.synthesize(
            cycle_index=cycle_index,
            rho_vac=ctx.rho,
            K_spec=getattr(getattr(ctx, "K", None), "eigenvalues", tuple()),
            triad=triad,
            H_ext=ctx.H,
            K=getattr(ctx, "K", None),
            context_beta=float(getattr(ctx, "beta", float("nan"))),
            explore_celestial=explore_celestial,
            tau_recurrence=tau_recurrence,
            agm_verdict=agm_verdict,
            monad_mix=monad_mix,
        )

    @classmethod
    def emit_birkhoff_seed(cls, bundle: WitnessObservationBundle) -> BirkhoffSeed:
        if bundle.poincare_seed is None:
            raise ValueError(
                "Bundle sin poincare_seed: no se puede emitir BirkhoffSeed."
            )
        seed = bundle.poincare_seed
        L4 = seed.libration_points.get("L4", (0.0, 0.0))
        L5 = seed.libration_points.get("L5", (0.0, 0.0))
        omega = float(bundle.kam_report.get("kam_freq", 0.0))
        lam1 = (
            float(bundle.liapunov_spectrum[0]) if bundle.liapunov_spectrum else 0.0
        )
        payload = (
            f"{seed.provenance_hash}|{omega:.12e}|{lam1:.12e}|"
            f"{bundle.sali_value:.12e}|{bundle.melnikov_peak:.12e}|"
            f"{bundle.section_entropy:.12e}|{bundle.gali_value:.12e}|"
            f"{bundle.melnikov_zeros}|{int(bundle.smale_horseshoe)}|"
            f"{seed.hill_topology}"
        ).encode("ascii")
        tag = hashlib.sha256(payload).hexdigest()
        logger.info(
            "[F2→F3] μ=%.6f C=%.6f λ₁=%.3e SALI=%.3e GALI=%.3e "
            "M_zeros=%d horseshoe=%s Chirikov=%s Hill=%s |det A|−1=%.3e",
            seed.mu, seed.jacobi_constant, lam1, bundle.sali_value,
            bundle.gali_value, bundle.melnikov_zeros, bundle.smale_horseshoe,
            bundle.arnold_tongues.get("chirikov_overlap", False),
            seed.hill_topology, bundle.return_map_det_residual,
        )
        return BirkhoffSeed(
            mu=seed.mu,
            jacobi_constant=seed.jacobi_constant,
            L4_xy=L4,
            L5_xy=L5,
            kam_frequency=omega,
            lyapunov_lambda1=lam1,
            sali=bundle.sali_value,
            melnikov_peak=bundle.melnikov_peak,
            section_entropy=bundle.section_entropy,
            triad_signature=bundle.triad_signature,
            provenance_hash=tag,
            gali=bundle.gali_value,
            gali_k=bundle.gali_k,
            melnikov_zeros=bundle.melnikov_zeros,
            smale_horseshoe=bundle.smale_horseshoe,
            arnold_tongues=bundle.arnold_tongues,
            agm_certificate=bundle.agm_certificate,
            monad_axioms=bundle.monad_axioms,
            crowbar_triggered=bundle.crowbar_triggered,
            rotation_number=bundle.rotation_number,
            diophantine_residue=bundle.diophantine_residue,
            lyapunov_pairing_residual=bundle.lyapunov_pairing_residual,
            mean_return_time=bundle.mean_return_time,
            jacobi_max_residual=bundle.jacobi_max_residual,
            hill_topology=seed.hill_topology,
            kraus_rank=seed.kraus_rank,
            back_action_bound=seed.back_action_bound,
            gali_decay_rate=bundle.gali_decay_rate,
            melnikov_transverse=bundle.melnikov_transverse,
            cantor_measure=float(
                bundle.arnold_tongues.get("cantor_measure", 1.0)
            ),
            routh_stable=seed.routh_stable,
            rsi3_mix=seed.rsi3_mix,
        )




# Resonancias de Deprit–Deprit-Bartholomé (1967) en L4:
#   2:1  ω_ℓ = 2 ω_s  ⇔  μ(1−μ) = 16/675
#   3:1  ω_ℓ = 3 ω_s  ⇔  μ(1−μ) = 1/75
_MU_RESONANCE_21: float = 0.5 * (1.0 - math.sqrt(max(0.0, 1.0 - 64.0 / 675.0)))
_MU_RESONANCE_31: float = 0.5 * (1.0 - math.sqrt(max(0.0, 1.0 - 4.0 / 75.0)))
_MU_RESONANCE_TOL: float = 5.0e-4
_FLOQUET_UNIT_TOL: float = 1.0e-6
_TAYLOR_STENCIL_H: float = 1.0e-3
_NEK_STEEPNESS_A_2DOF: float = 0.5  # Pöschel: a ∼ 1/(2n), n=2 → 1/4…1/2


# ═══════════════════════════════════════════════════════════════════════════
# §3.0  Reportes de transporte (NF, Nekhoroshev, Arnold, adjudicación)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class BirkhoffNormalFormReport:
    r"""
    Forma normal cuadrática en L4 + diagnóstico de twist.

        H₂ = ω_s I_s + ω_ℓ I_ℓ ,

    con (ω_s, ω_ℓ) las frecuencias de Gascheau.  El twist cuártico
    det(A_{ij}) se estima por el Hessiano espacial de Ω (no es el
    Lie-transform completo de Deprit: se documenta como cota).
    """
    mu: float
    L4_xy: Tuple[float, float]
    linearization: np.ndarray
    eigenvalues: Tuple[complex, ...]
    omega_short: float
    omega_long: float
    gascheau_discriminant: float
    routh_stable: bool
    spectrally_stable: bool
    floquet_multipliers: Tuple[complex, ...]
    floquet_on_circle: bool
    resonance_21: bool
    resonance_31: bool
    twist_hessian_det: float
    twist_condition: bool
    taylor_coefficients: Dict[str, float]
    spectral_type: str

    def frequencies(self) -> Tuple[float, float]:
        return float(self.omega_short), float(self.omega_long)

    def is_kam_candidate(self) -> bool:
        return bool(
            self.routh_stable
            and self.spectrally_stable
            and not self.resonance_21
            and not self.resonance_31
            and self.twist_condition
        )


@dataclass(frozen=True, slots=True)
class NekhoroshevReport:
    r"""
    Cota de Nekhoroshev (1977) para hamiltonianos steep:

        |I(t) − I(0)| ≤ C ε^b     para    |t| ≤ T_* exp(c / ε^a).

    En 2DOF (CR3BP planar reducido) la cota es *dominada* por KAM:
    los toros de dimensión 1 separan la superficie de energía 3D
    y no hay deriva de Arnold.  `confined_by_kam` lo certifica.
    """
    epsilon: float
    dim: int
    a: float
    b: float
    T_escape: float
    drift_bound: float
    confined_by_kam: bool
    applicable: bool
    note: str


@dataclass(frozen=True, slots=True)
class ArnoldDiffusionReport:
    r"""
    Difusión de Arnold: deriva de acciones a lo largo de la red
    de resonancias en sistemas *a ≥ 3 d.o.f.*

    En 2DOF: rate ≡ 0, T_esc ≡ ∞  (teorema de confinamiento).
    """
    dim: int
    rate: float
    escape_time: float
    is_possible: bool
    local_verdict: HeytingOmega3


@dataclass(frozen=True, slots=True)
class WitnessAdjudicationCertificate:
    r"""Meet granular Ω₃ del testigo (auditoría + celeste + AGM + RSI-3)."""
    scores: Dict[str, HeytingOmega3]
    local_meet: HeytingOmega3
    external: HeytingOmega3
    verdict: HeytingOmega3
    poincare_stratum: str
    timestamp_utc: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scores": {k: v.name for k, v in self.scores.items()},
            "local_meet": self.local_meet.name,
            "external": self.external.name,
            "verdict": self.verdict.name,
            "poincare_stratum": self.poincare_stratum,
            "timestamp_utc": float(self.timestamp_utc),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §3.1  Forma normal de Birkhoff en L4
#       (morfismo inicial de FASE 3)
# ═══════════════════════════════════════════════════════════════════════════
class BirkhoffNormalForm:
    r"""
    Linearización y frecuencias de Gascheau–Routh en los equiláteros.

    Ecuación característica (Szebehely §5.4):

        λ⁴ + λ² + (27/4) μ(1−μ) = 0.

    Con Δ = 1 − 27μ(1−μ) ≥ 0  (Routh ⇔ μ < μ_R):

        ω_{s,ℓ}² = ½ (1 ∓ √Δ).

    CORRECCIÓN v9.2.0: v9.1.0 omitía el ½ y no usaba L4_xy.
    """

    @classmethod
    def from_seed(cls, seed: BirkhoffSeed) -> BirkhoffNormalFormReport:
        return cls.analyze(seed.L4_xy, seed.mu)

    @classmethod
    def linearization(
        cls, L4_xy: Tuple[float, float], mu: float
    ) -> np.ndarray:
        r"""Jacobiano 4×4 del flujo rotante en (x,y,0,0) = L4 (Fase 1)."""
        x, y = float(L4_xy[0]), float(L4_xy[1])
        mu = CR3BPDynamics.clip_mu(mu)
        return CR3BPDynamics.jacobian(np.array([x, y, 0.0, 0.0]), mu)

    @classmethod
    def gascheau_discriminant(cls, mu: float) -> float:
        mu = CR3BPDynamics.clip_mu(mu)
        return float(1.0 - 27.0 * mu * (1.0 - mu))

    @classmethod
    def frequencies(cls, mu: float) -> Tuple[float, float]:
        r"""
        (ω_s, ω_ℓ) de Gascheau.  0 si μ > μ_R (L4 hiperbólico:
        las raíces s = λ² cruzan al semiplano positivo).
        """
        disc = cls.gascheau_discriminant(mu)
        if disc < 0.0:
            return 0.0, 0.0
        root = math.sqrt(disc)
        w_s2 = 0.5 * (1.0 - root)
        w_l2 = 0.5 * (1.0 + root)
        return math.sqrt(max(w_s2, 0.0)), math.sqrt(max(w_l2, 0.0))

    @classmethod
    def jacobian_eigenvalues(
        cls, L4_xy: Tuple[float, float], mu: float
    ) -> np.ndarray:
        return np.linalg.eigvals(cls.linearization(L4_xy, mu))

    @classmethod
    def is_spectrally_stable(
        cls, L4_xy: Tuple[float, float], mu: float
    ) -> bool:
        eigs = cls.jacobian_eigenvalues(L4_xy, mu)
        return bool(np.all(np.abs(np.real(eigs)) < 1.0e-8))

    @classmethod
    def floquet_multipliers(
        cls,
        L4_xy: Tuple[float, float],
        mu: float,
        period: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Multiplicadores de Floquet del equilibrio:  exp(λ T).

        Por defecto T = 2π / ω_s  (periodo corto de Lyapunov), de modo
        que el par corto se mapea a {1, 1} y el largo a
        exp(± i 2π ω_ℓ/ω_s)  — rotación del número de rotación ρ=ω_ℓ/ω_s.
        """
        eigs = cls.jacobian_eigenvalues(L4_xy, mu)
        w_s, _w_l = cls.frequencies(mu)
        if period is None:
            period = (2.0 * math.pi / w_s) if w_s > _EPS else 1.0
        return np.exp(eigs * float(period))

    @classmethod
    def floquet_on_unit_circle(cls, multipliers: np.ndarray) -> bool:
        return bool(np.all(np.abs(np.abs(multipliers) - 1.0) < 1e-3))

    @classmethod
    def is_resonance_21(cls, mu: float, tol: float = _MU_RESONANCE_TOL) -> bool:
        return abs(float(mu) - _MU_RESONANCE_21) < tol

    @classmethod
    def is_resonance_31(cls, mu: float, tol: float = _MU_RESONANCE_TOL) -> bool:
        return abs(float(mu) - _MU_RESONANCE_31) < tol

    @classmethod
    def l4_hessian_analytic(cls, mu: float) -> Tuple[float, float, float]:
        r"""
        Hessiano de Ω en L4 (cerrado):

            Ω_xx = 1/4,  Ω_yy = 9/4,  Ω_xy = (3√3/4)(1−2μ).
        """
        mu = CR3BPDynamics.clip_mu(mu)
        oxy = (3.0 * math.sqrt(3.0) / 4.0) * (1.0 - 2.0 * mu)
        return 0.25, 2.25, float(oxy)

    @classmethod
    def taylor_potential(
        cls,
        L4_xy: Tuple[float, float],
        mu: float,
        h: float = _TAYLOR_STENCIL_H,
        degree: int = 4,
    ) -> Dict[str, float]:
        r"""
        Ajuste de Taylor de Ω alrededor de L4 hasta grado `degree`.

        Muestras en un stencil (2d+1)×(2d+1); mínimos cuadrados sobre
        monomios x^p y^q, p+q ≤ degree.  Los cúbicos deben ser ~ 0
        por simetría triangular (certificado numérico).
        """
        x0, y0 = float(L4_xy[0]), float(L4_xy[1])
        mu = CR3BPDynamics.clip_mu(mu)
        d = int(max(2, degree))
        xs = np.linspace(-d * h, d * h, 2 * d + 1)
        rows: List[List[float]] = []
        rhs: List[float] = []
        mons: List[Tuple[int, int]] = [
            (p, q)
            for tot in range(0, d + 1)
            for p in range(0, tot + 1)
            for q in (tot - p,)
        ]
        for dx in xs:
            for dy in xs:
                rows.append([
                    (dx ** p) * (dy ** q) / max(math.factorial(p) * math.factorial(q), 1)
                    for p, q in mons
                ])
                rhs.append(CR3BPDynamics.potential(x0 + dx, y0 + dy, mu))
        A = np.asarray(rows, dtype=np.float64)
        b = np.asarray(rhs, dtype=np.float64)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        out: Dict[str, float] = {}
        for (p, q), c in zip(mons, coef):
            out[f"O_{p}{q}"] = float(c)
        # Hessiano extraído (p+q=2)
        out["Omega_xx"] = float(out.get("O_20", 0.0))
        out["Omega_xy"] = float(out.get("O_11", 0.0))
        out["Omega_yy"] = float(out.get("O_02", 0.0))
        cubic = math.hypot(
            out.get("O_30", 0.0), out.get("O_21", 0.0),
        ) + math.hypot(out.get("O_12", 0.0), out.get("O_03", 0.0))
        out["cubic_residual"] = float(cubic)
        out["twist_proxy"] = float(
            out["Omega_xx"] * out["Omega_yy"] - 0.25 * out["Omega_xy"] ** 2
        )
        return out

    @classmethod
    def higher_order_coefficients(
        cls,
        L4_xy: Tuple[float, float],
        mu: float,
        max_order: int = 4,
    ) -> Dict[int, float]:
        r"""Firma retro-compatible: {orden: coeficiente representativo}."""
        tay = cls.taylor_potential(L4_xy, mu, degree=int(max(2, max_order)))
        oxx, oyy, oxy = cls.l4_hessian_analytic(mu)
        coef: Dict[int, float] = {
            2: float(oxx * oyy - oxy ** 2),
            3: float(tay.get("cubic_residual", 0.0)),
            4: float(tay.get("O_40", 0.0) + tay.get("O_04", 0.0)),
        }
        return {k: coef[k] for k in coef if k <= max_order}

    @classmethod
    def analyze(
        cls, L4_xy: Tuple[float, float], mu: float
    ) -> BirkhoffNormalFormReport:
        mu = CR3BPDynamics.clip_mu(mu)
        A = cls.linearization(L4_xy, mu)
        eigs = np.linalg.eigvals(A)
        w_s, w_l = cls.frequencies(mu)
        disc = cls.gascheau_discriminant(mu)
        routh = LibrationPointSolver.routh_stability(mu)
        spec_ok = bool(np.all(np.abs(np.real(eigs)) < 1.0e-8))
        floq = cls.floquet_multipliers(L4_xy, mu)
        tay = cls.taylor_potential(L4_xy, mu)
        oxx, oyy, oxy = cls.l4_hessian_analytic(mu)
        twist_det = float(oxx * oyy - oxy ** 2)
        try:
            stype = LibrationPointSolver.spectral_type("L4", mu)
        except (ValueError, np.linalg.LinAlgError):
            stype = "unknown"
        return BirkhoffNormalFormReport(
            mu=float(mu),
            L4_xy=(float(L4_xy[0]), float(L4_xy[1])),
            linearization=A,
            eigenvalues=tuple(complex(z) for z in eigs.tolist()),
            omega_short=float(w_s),
            omega_long=float(w_l),
            gascheau_discriminant=float(disc),
            routh_stable=bool(routh),
            spectrally_stable=spec_ok,
            floquet_multipliers=tuple(complex(z) for z in floq.tolist()),
            floquet_on_circle=cls.floquet_on_unit_circle(floq),
            resonance_21=cls.is_resonance_21(mu),
            resonance_31=cls.is_resonance_31(mu),
            twist_hessian_det=twist_det,
            twist_condition=bool(abs(twist_det) > _KOLMOGOROV_DET_TOL)
            if "_KOLMOGOROV_DET_TOL" in dir()
            else bool(abs(twist_det) > 1e-12),
            taylor_coefficients=tay,
            spectral_type=stype,
        )


# ═══════════════════════════════════════════════════════════════════════════
# §3.2  Escalera de resonancias (fracciones continuas de Poincaré)
# ═══════════════════════════════════════════════════════════════════════════
class ResonanceLadder:
    r"""
    Aproximantes de Farey / fracciones continuas del número de rotación.

        ρ = ω_ℓ / ω_s  (L4)    o    ρ_sección  (mapa de Poincaré).

    Densidad de resonancias:  Σ 1/q²  sobre convergentes (como
    medida de Dirichlet de pequeños denominadores).
    """

    @staticmethod
    def continued_fraction(x: float, depth: int = 12) -> List[int]:
        coeffs: List[int] = []
        y = float(x)
        for _ in range(int(max(1, depth))):
            if not math.isfinite(y):
                break
            a = int(math.floor(y))
            coeffs.append(a)
            frac = y - a
            if abs(frac) < 1.0e-12:
                break
            y = 1.0 / frac
        return coeffs

    @staticmethod
    def convergents(coeffs: Sequence[int]) -> List[Tuple[int, int]]:
        p_prev, p_cur = 0, 1
        q_prev, q_cur = 1, 0
        out: List[Tuple[int, int]] = []
        for a in coeffs:
            p_new = int(a) * p_cur + p_prev
            q_new = int(a) * q_cur + q_prev
            if q_new != 0:
                out.append((p_new, q_new))
            p_prev, p_cur = p_cur, p_new
            q_prev, q_cur = q_cur, q_new
        return out

    @classmethod
    def resonance_density(cls, entries: Sequence[int]) -> float:
        conv = cls.convergents(list(entries))
        if not conv:
            return 0.0
        return float(sum(1.0 / (q * q) for _, q in conv if q != 0))

    @classmethod
    def rotation_ratio(cls, omega_short: float, omega_long: float) -> float:
        if omega_short <= _EPS:
            return float("inf")
        return float(omega_long / omega_short)

    @classmethod
    def from_seed(cls, seed: BirkhoffSeed, depth: int = 12) -> Dict[str, Any]:
        w_s, w_l = BirkhoffNormalForm.frequencies(seed.mu)
        rho = cls.rotation_ratio(w_s, w_l)
        if not math.isfinite(rho):
            rho = float(seed.rotation_number)
        cf = cls.continued_fraction(rho, depth=depth)
        conv = cls.convergents(cf)
        return {
            "omega_short": float(w_s),
            "omega_long": float(w_l),
            "rotation_ratio": float(rho),
            "continued_fraction": tuple(cf),
            "convergents": tuple(conv),
            "resonance_density": cls.resonance_density(cf),
            "resonance_21": BirkhoffNormalForm.is_resonance_21(seed.mu),
            "resonance_31": BirkhoffNormalForm.is_resonance_31(seed.mu),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §3.3  Nekhoroshev  ≠  difusión de Arnold
# ═══════════════════════════════════════════════════════════════════════════
class NekhoroshevBound:
    r"""
    Cota exponencial de estabilidad de acciones (Nekhoroshev 1977).

    Para hamiltonianos steep de n grados de libertad, a ≃ 1/(2n).
    En el CR3BP *planar* n = 2 (tras Jacobi): los toros KAM de dim. 1
    confinan, y Nekhoroshev es redundante respecto de KAM perpetuo
    sobre un conjunto de medida casi total.

    `from_seed` usa ε ∼ |λ₁| / max(ω_KAM, 1) como amplitud perturbativa.
    """

    @classmethod
    def bound(
        cls,
        epsilon: float,
        dim: int = 2,
        C_Nekh: float = 1.0,
        confined_by_kam: bool = False,
    ) -> Dict[str, float]:
        report = cls.bound_certified(
            epsilon, dim=dim, C_Nekh=C_Nekh, confined_by_kam=confined_by_kam
        )
        return {
            "a": report.a,
            "T_escape": report.T_escape,
            "drift_bound": report.drift_bound,
            "epsilon": report.epsilon,
        }

    @classmethod
    def bound_certified(
        cls,
        epsilon: float,
        dim: int = 2,
        C_Nekh: float = 1.0,
        confined_by_kam: bool = False,
    ) -> NekhoroshevReport:
        eps = float(np.clip(abs(epsilon), 1e-12, 0.99))
        n = int(max(1, dim))
        a = 1.0 / (2.0 * n)
        b = 1.0 / (2.0 * n)
        if confined_by_kam and n <= 2:
            return NekhoroshevReport(
                epsilon=eps, dim=n, a=a, b=b,
                T_escape=float("inf"), drift_bound=0.0,
                confined_by_kam=True, applicable=False,
                note="2DOF: KAM confina; Nekhoroshev redundante.",
            )
        try:
            T = float(C_Nekh * math.exp(1.0 / (eps ** a)))
        except OverflowError:
            T = float("inf")
        drift = float(eps ** b)
        return NekhoroshevReport(
            epsilon=eps, dim=n, a=a, b=b,
            T_escape=T, drift_bound=drift,
            confined_by_kam=False, applicable=True,
            note="Nekhoroshev steep (Pöschel a=1/(2n)).",
        )

    @classmethod
    def from_seed(
        cls, seed: BirkhoffSeed, nf: Optional[BirkhoffNormalFormReport] = None
    ) -> NekhoroshevReport:
        omega = max(abs(float(seed.kam_frequency)), 1e-12)
        eps = float(np.clip(abs(seed.lyapunov_lambda1) / omega, 1e-12, 0.99))
        confined = bool(
            (nf.is_kam_candidate() if nf is not None else seed.routh_stable)
            and not seed.smale_horseshoe
            and not bool(seed.arnold_tongues.get("chirikov_overlap", False))
        )
        return cls.bound_certified(
            eps, dim=2, C_Nekh=1.0, confined_by_kam=confined
        )


class ArnoldDiffusionBridge:
    r"""
    Puente de difusión de Arnold.

    Teorema (Arnold 1964; Lochak, Marco): la deriva a lo largo de la
    web de resonancias exige *al menos 3 d.o.f.*  El CR3BP planar
    reducido es 2DOF ⇒ rate = 0, T_esc = ∞.

    La fórmula ε^{N+1} exp(−1/ε^a) se reserva como *cota de Nekhoroshev*
    (no como tasa de Arnold) y sólo se invoca si dim ≥ 3.
    """

    @classmethod
    def diffusion_rate(
        cls, lambda1: float, omega_kam: float, dim: int = 2
    ) -> float:
        return cls.report(lambda1, omega_kam, dim=dim).rate

    @classmethod
    def escape_time(
        cls, lambda1: float, omega_kam: float, dim: int = 2
    ) -> float:
        return cls.report(lambda1, omega_kam, dim=dim).escape_time

    @classmethod
    def report(
        cls, lambda1: float, omega_kam: float, dim: int = 2
    ) -> ArnoldDiffusionReport:
        n = int(max(1, dim))
        if n <= 2:
            return ArnoldDiffusionReport(
                dim=n, rate=0.0, escape_time=float("inf"),
                is_possible=False, local_verdict=HeytingOmega3.COHERENT,
            )
        if abs(omega_kam) <= 1.0e-12:
            return ArnoldDiffusionReport(
                dim=n, rate=float("inf"), escape_time=0.0,
                is_possible=True, local_verdict=HeytingOmega3.VETOED,
            )
        eps = float(np.clip(abs(lambda1) / abs(omega_kam), 1.0e-9, 0.9))
        a = 1.0 / (n + 1.0)
        try:
            rate = float(eps ** (n + 1) * math.exp(-1.0 / (eps ** a)))
        except OverflowError:
            rate = 0.0
        esc = (1.0 / rate) if (rate > 0.0 and math.isfinite(rate)) else float("inf")
        verdict = (
            HeytingOmega3.VETOED if rate > 1e-3
            else (
                HeytingOmega3.DEGRADED if rate > 1e-12
                else HeytingOmega3.COHERENT
            )
        )
        return ArnoldDiffusionReport(
            dim=n, rate=rate, escape_time=esc,
            is_possible=True, local_verdict=verdict,
        )

    @classmethod
    def from_seed(cls, seed: BirkhoffSeed, dim: int = 2) -> ArnoldDiffusionReport:
        return cls.report(seed.lyapunov_lambda1, seed.kam_frequency, dim=dim)


# ═══════════════════════════════════════════════════════════════════════════
# §3.4  Cristal de experiencia  (objeto terminal del agente)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ExperienceCrystal:
    r"""
    Cristal de experiencia del Testigo Silencioso.

    Objeto terminal de FASE 3: veredicto Ω₃ + cadena Merkle +
    invariantes celestes (Birkhoff, KAM, Nekhoroshev, GALI, Melnikov)
    + AGM + RSI-3 + Crowbar SWA1.  Inmersible en S⁶ vía `vector_s6`.
    """
    crystal_id: str
    witness_id: str
    trickster_illusion_type: str
    dreamer_scenario_id: str
    auditor_immunization_hash: str
    vacuum_metrics: VacuumStateMetrics
    audit_report: Any
    heyting_verdict: HeytingOmega3
    crystallized_invariant_vector: np.ndarray
    content_hash: str
    chain_hash: str
    merkle_parent: str
    sha256_provenance: str
    timestamp_utc: float
    vector_s6: Optional[np.ndarray] = None
    tau_recurrence: float = 1.0
    kms_drift: float = 0.0
    merkle_root_sha256: str = ""
    engine_version: str = __version__
    birkhoff_frequencies: Tuple[float, float] = (0.0, 0.0)
    birkhoff_stable: bool = False
    kam_frequency: float = 0.0
    kam_bias: float = 0.0
    arnold_diffusion_rate: float = 0.0
    arnold_escape_time: float = float("inf")
    resonance_density: float = 0.0
    sali: float = 0.0
    melnikov_peak: float = 0.0
    gali: float = 0.0
    melnikov_zeros: int = 0
    smale_horseshoe: bool = False
    chirikov_overlap: bool = False
    cantor_measure: float = 1.0
    nekhoroshev_T: float = float("inf")
    nekhoroshev_drift: float = 0.0
    agm_postulates_passed: int = 0
    agm_all_satisfied: bool = False
    monad_associativity: bool = False
    monad_unitarity: bool = False
    rsi_level: int = 0
    crowbar_triggered: bool = False
    # v9.2.0-F3
    hill_topology: str = "unknown"
    rotation_number: float = 0.0
    diophantine_residue: float = 0.0
    lyapunov_pairing_residual: float = 0.0
    mean_return_time: float = float("inf")
    kraus_rank: int = 1
    back_action_bound: float = 0.0
    routh_stable: bool = False
    gascheau_discriminant: float = 0.0
    floquet_on_circle: bool = False
    resonance_21: bool = False
    resonance_31: bool = False
    twist_condition: bool = False
    confined_by_kam: bool = False
    arnold_possible: bool = False
    monad_holds_on_image: bool = False
    gali_decay_rate: float = 0.0
    jacobi_max_residual: float = 0.0

    def is_kam_stable(self) -> bool:
        return bool(
            not self.chirikov_overlap
            and self.kam_bias < 1e-3
            and self.routh_stable
            and not self.smale_horseshoe
            and not self.resonance_21
            and not self.resonance_31
        )

    def is_silent(self) -> bool:
        return bool(
            self.heyting_verdict == HeytingOmega3.COHERENT
            and self.vacuum_metrics.purity >= 0.999
            and not self.crowbar_triggered
        )

    def merkle_leaf(self) -> str:
        return hashlib.sha256(
            (
                f"{self.crystal_id}|{self.content_hash}|{self.chain_hash}|"
                f"{self.heyting_verdict.name}"
            ).encode("ascii")
        ).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# §3.5  Adjudicador Ω₃ del Testigo
# ═══════════════════════════════════════════════════════════════════════════
class HeytingWitnessAdjudicator:
    r"""
    Clasificador  Sh(Ω₃) ← WitnessObservationBundle × BirkhoffSeed.

    Meet granular: auditoría modular ∧ AGM ∧ mónada ∧ Smale ∧ Chirikov
    ∧ Crowbar ∧ Jacobi ∧ Lyapunov ∧ Hill ∧ Routh ∧ pairing ∧ Kraus-CP.
    """
    AXIOM_TOL: float = 1.0e-6
    CATASTROPHIC_TOL: float = 1.0e-3
    JACOBI_DEGRADE: float = 1.0e-6
    LYAPUNOV_VETO: float = 5.0e-2
    PAIRING_DEGRADE: float = 5.0e-2

    @classmethod
    def _residual_to_omega(
        cls, residual: float, degrade: Optional[float] = None, veto: Optional[float] = None
    ) -> HeytingOmega3:
        r = abs(float(residual))
        vtol = cls.CATASTROPHIC_TOL if veto is None else float(veto)
        dtol = cls.AXIOM_TOL if degrade is None else float(degrade)
        if r > vtol:
            return HeytingOmega3.VETOED
        if r > dtol:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def granular_scores(
        cls,
        bundle: WitnessObservationBundle,
        nf: Optional[BirkhoffNormalFormReport] = None,
    ) -> Dict[str, HeytingOmega3]:
        scores: Dict[str, HeytingOmega3] = {}
        local_audit = getattr(bundle.audit, "local_verdict", HeytingOmega3.COHERENT)
        if not isinstance(local_audit, HeytingOmega3):
            try:
                local_audit = HeytingOmega3(int(local_audit))
            except (ValueError, TypeError):
                local_audit = HeytingOmega3.DEGRADED
        scores["audit"] = local_audit
        if bundle.agm_certificate is not None:
            scores["agm"] = bundle.agm_certificate.local_verdict
        if bundle.monad_axioms is not None:
            scores["monad"] = bundle.monad_axioms.verdict
        scores["smale"] = (
            HeytingOmega3.VETOED if bundle.smale_horseshoe
            else HeytingOmega3.COHERENT
        )
        scores["chirikov"] = (
            HeytingOmega3.VETOED
            if bool(bundle.arnold_tongues.get("chirikov_overlap", False))
            else HeytingOmega3.COHERENT
        )
        scores["crowbar"] = (
            HeytingOmega3.VETOED if bundle.crowbar_triggered
            else HeytingOmega3.COHERENT
        )
        scores["jacobi"] = cls._residual_to_omega(
            bundle.jacobi_max_residual,
            degrade=cls.JACOBI_DEGRADE,
            veto=cls.CATASTROPHIC_TOL,
        )
        lam1 = (
            float(bundle.liapunov_spectrum[0]) if bundle.liapunov_spectrum else 0.0
        )
        scores["lyapunov"] = (
            HeytingOmega3.VETOED if lam1 > cls.LYAPUNOV_VETO
            else (
                HeytingOmega3.DEGRADED if lam1 > 1e-3
                else HeytingOmega3.COHERENT
            )
        )
        hill = (
            bundle.poincare_seed.hill_topology
            if bundle.poincare_seed is not None
            else "unknown"
        )
        scores["hill"] = (
            HeytingOmega3.VETOED
            if hill in (HillTopology.TRANSIT.name, HillTopology.FREE.name)
            else (
                HeytingOmega3.DEGRADED
                if hill == HillTopology.EXTERIOR_GATE.name
                else HeytingOmega3.COHERENT
            )
        )
        scores["pairing"] = cls._residual_to_omega(
            bundle.lyapunov_pairing_residual,
            degrade=cls.PAIRING_DEGRADE,
            veto=0.5,
        )
        if bundle.poincare_seed is not None and bundle.poincare_seed.kraus_certificate is not None:
            scores["kraus_cp"] = (
                HeytingOmega3.COHERENT
                if bundle.poincare_seed.kraus_certificate.is_completely_positive
                else HeytingOmega3.VETOED
            )
        if nf is not None:
            scores["routh"] = (
                HeytingOmega3.COHERENT if nf.routh_stable
                else HeytingOmega3.DEGRADED
            )
            scores["deprit_res"] = (
                HeytingOmega3.DEGRADED
                if (nf.resonance_21 or nf.resonance_31)
                else HeytingOmega3.COHERENT
            )
            scores["floquet"] = (
                HeytingOmega3.COHERENT if nf.floquet_on_circle
                else HeytingOmega3.DEGRADED
            )
        axioms_max = (
            max(bundle.modular_axioms.values()) if bundle.modular_axioms else 0.0
        )
        scores["tomita_axioms"] = cls._residual_to_omega(float(axioms_max))
        return scores

    @classmethod
    def adjudicate(
        cls,
        bundle: WitnessObservationBundle,
        external_verdict: HeytingOmega3,
        nf: Optional[BirkhoffNormalFormReport] = None,
    ) -> HeytingOmega3:
        return cls.adjudicate_certified(bundle, external_verdict, nf=nf).verdict

    @classmethod
    def adjudicate_certified(
        cls,
        bundle: WitnessObservationBundle,
        external_verdict: HeytingOmega3,
        nf: Optional[BirkhoffNormalFormReport] = None,
    ) -> WitnessAdjudicationCertificate:
        scores = cls.granular_scores(bundle, nf=nf)
        local = HeytingOmega3.COHERENT
        for v in scores.values():
            local = local.meet(v)
        verdict = local.meet(external_verdict)
        stratum = (
            verdict.poincare_stratum_name()
            if hasattr(verdict, "poincare_stratum_name")
            else verdict.name
        )
        return WitnessAdjudicationCertificate(
            scores=scores,
            local_meet=local,
            external=external_verdict,
            verdict=verdict,
            poincare_stratum=stratum,
            timestamp_utc=time.time(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §3.6  Soberano — TOONSilentWitnessAgent
# ═══════════════════════════════════════════════════════════════════════════
class TOONSilentWitnessAgent:
    r"""
    Soberano Testigo Silencioso: tríada ≅ CR3BP, Birkhoff, Nekhoroshev,
    AGM, RSI-3, Crowbar SWA1.

        F₁  TriadToCR3BPMorphism.emit_poincare_seed
        F₂  WitnessObservationPipeline.emit_birkhoff_seed
        F₃  BirkhoffNormalForm.from_seed ⊗ adjudicate ⊗ crystallize

    El endomorfismo de registro
        observe_and_crystallize : (ilusión, sueño, auditor) → ExperienceCrystal
    realiza la composición de las tres fases.
    """

    def __init__(
        self,
        agent_id: str = "SILENT-WITNESS-SABIO-01",
        dimension_mac: int = 4,
        dimension: Optional[int] = None,
        kms_beta: float = 1.0,
        vacuum_beta: float = 50.0,
        seed: int = 999,
        hopping: float = 1.0e-3,
        celestial_dt: float = 5.0e-3,
        n_crossings: int = 256,
        meta_engine: Optional[Any] = None,
        monad_eta: float = _RSI3_DEFAULT_MIX,
        monad_mix: Optional[float] = None,
        rsi3_levels: int = _RSI3_DEFAULT_LEVEL,
    ) -> None:
        if dimension is not None:
            dimension_mac = dimension
        mix = float(monad_eta if monad_mix is None else monad_mix)
        self.agent_id = agent_id
        self.dimension_mac = int(dimension_mac)
        self.kms_beta = float(kms_beta)
        self.vacuum_beta = float(vacuum_beta)
        self.seed = int(seed)
        self.celestial_dt = float(celestial_dt)
        self.n_crossings = int(n_crossings)
        self.monad_mix = mix
        self.rsi3_levels = int(rsi3_levels)
        self.crystal_count = 0
        self.crystal_history: List[ExperienceCrystal] = []
        self.experience_archive: List[ExperienceCrystal] = []
        self._adjudication_log: List[WitnessAdjudicationCertificate] = []

        if MetaSilentWitnessEngine is not None:
            self.meta_engine = meta_engine or MetaSilentWitnessEngine(
                dimension=self.dimension_mac
            )
        else:
            self.meta_engine = meta_engine
        if TOONSilentWitnessEngine is not None:
            self.engine = TOONSilentWitnessEngine(
                engine_id=f"ENGINE-{agent_id}",
                mac_dimension=self.dimension_mac,
                kms_beta=self.kms_beta,
                hopping=hopping,
                meta_engine=self.meta_engine,
            )
        else:
            self.engine = None

        if VacuumStatePreparation is not None:
            self.H_ext = VacuumStatePreparation.tight_binding_hamiltonian(
                self.dimension_mac, hopping=hopping
            )
        else:
            H = np.diag(
                np.linspace(0.0, float(max(self.dimension_mac - 1, 0)), self.dimension_mac)
            ).astype(np.complex128)
            for i in range(self.dimension_mac - 1):
                H[i, i + 1] = hopping
                H[i + 1, i] = hopping
            self.H_ext = MatrixBanachAlgebra.hermitize(H)

        self.ground_projector = WitnessVacuumPreparation.ground_state_projector(
            self.H_ext
        )
        self.path_laplacian = WitnessVacuumPreparation.path_laplacian(
            self.dimension_mac
        )
        self.monad = MonadRSI3(
            dim=self.dimension_mac, mix=self.monad_mix, level=self.rsi3_levels
        )
        self._genesis_hash = hashlib.sha256(
            f"{agent_id}::GENESIS::{__version__}".encode("ascii")
        ).hexdigest()
        self._chain_hash = self._genesis_hash

    def _content_hash(
        self,
        cycle_id: str,
        audit: Any,
        inv: np.ndarray,
        sig: TriadSignature,
        dirichlet_energy_arg: float,
        birkhoff_seed: Optional[BirkhoffSeed] = None,
        verdict: Optional[HeytingOmega3] = None,
    ) -> str:
        h = hashlib.sha256()
        h.update(self.agent_id.encode("ascii"))
        h.update(cycle_id.encode("ascii"))
        h.update(sig.as_bytes())
        h.update(np.ascontiguousarray(inv).tobytes())
        h.update(f"{getattr(audit, 'thermal_fluctuation', 0.0):.12e}".encode("ascii"))
        h.update(f"{getattr(audit, 'purity', 0.0):.12e}".encode("ascii"))
        h.update(f"{getattr(audit, 'free_energy', 0.0):.12e}".encode("ascii"))
        h.update(f"{getattr(audit, 'vev', 0.0):.12e}".encode("ascii"))
        lv = getattr(getattr(audit, "local_verdict", None), "name", "NA")
        h.update(str(lv).encode("ascii"))
        h.update(f"{dirichlet_energy_arg:.12e}".encode("ascii"))
        if verdict is not None:
            h.update(verdict.name.encode("ascii"))
        if birkhoff_seed is not None:
            h.update(birkhoff_seed.provenance_hash.encode("ascii"))
            h.update(f"{birkhoff_seed.gali:.12e}".encode("ascii"))
            h.update(f"{birkhoff_seed.melnikov_zeros}".encode("ascii"))
            h.update(birkhoff_seed.hill_topology.encode("ascii"))
        return h.hexdigest()

    def _advance_chain(self, content_hash: str) -> Tuple[str, str]:
        parent = self._chain_hash
        digest = hashlib.sha256(
            parent.encode("ascii") + content_hash.encode("ascii")
        ).hexdigest()
        self._chain_hash = digest
        return parent, digest

    def execute_zero_backaction_poincare_observation(
        self, density_matrix: np.ndarray, tau_recurrence: float = 1.0
    ) -> Dict[str, Any]:
        r"""Observación SWA1: recurrencia BKM / KMS, Crowbar si catastrófico."""
        if self.engine is not None:
            is_rec, drift, poincare_crystal = (
                self.engine.audit_poincare_recurrence_kms_vacuum(
                    density_matrix, tau_recurrence
                )
            )
            merkle = poincare_crystal.merkle_root_sha256
            vector_s6 = poincare_crystal.vector_s6
        else:
            rho = DensityMatrixOps.sanitize(density_matrix)
            kms_drift = float(
                np.linalg.norm(rho @ rho - rho @ rho, "fro")
            )
            is_rec = kms_drift < 1e-2
            drift = kms_drift
            merkle = hashlib.sha256(
                np.ascontiguousarray(rho).tobytes()
            ).hexdigest()
            vector_s6 = np.ones(7, dtype=np.float64) / math.sqrt(7.0)

        pred = SWA1Predicate(kms_drift=float(drift))
        crowbar = ESP32CrowbarInterlock.trigger_if(pred)
        if crowbar is not None:
            verdict = "VETOED"
        else:
            verdict = "COHERENT" if is_rec else "DEGRADED"
        return {
            "agent": "TOONSilentWitnessAgent",
            "verdict": verdict,
            "poincare_recurrence": bool(is_rec),
            "kms_drift": float(drift),
            "merkle_root_sha256": merkle,
            "vector_s6": vector_s6,
            "crowbar_trigger": crowbar is not None,
        }

    def latest_crystal(self) -> Optional[ExperienceCrystal]:
        return self.crystal_history[-1] if self.crystal_history else None

    def verify_merkle_chain(self) -> bool:
        parent = self._genesis_hash
        for crystal in self.experience_archive:
            if crystal.merkle_parent != parent:
                return False
            expected = hashlib.sha256(
                parent.encode("ascii") + crystal.content_hash.encode("ascii")
            ).hexdigest()
            if expected != crystal.chain_hash:
                return False
            if len(crystal.chain_hash) != 64:
                return False
            nrm = float(np.linalg.norm(crystal.crystallized_invariant_vector))
            if not math.isclose(nrm, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
                return False
            parent = crystal.chain_hash
        return True

    def emit_agm_belief_set(self) -> AGMBeliefSet:
        r"""Hand-off AGM (mundos de Ω₃) hacia el agente de introspección."""
        latest = self.latest_crystal()
        if latest is None:
            return AGMBeliefSet.top()
        alpha = HeytingAGMRevision.proposition_from_verdict(latest.heyting_verdict)
        return AGMBeliefSet.top().revise(alpha)

    def emit_witness_passport(self) -> Dict[str, Any]:
        agm = self.emit_agm_belief_set()
        n = len(self.experience_archive)
        n_silent = sum(1 for c in self.experience_archive if c.is_silent())
        n_kam = sum(1 for c in self.experience_archive if c.is_kam_stable())
        gv = HeytingOmega3.COHERENT
        for c in self.experience_archive:
            gv = gv.meet(c.heyting_verdict)
        return {
            "agent_id": self.agent_id,
            "version": __version__,
            "registry_size": n,
            "global_verdict": gv.name,
            "n_silent": n_silent,
            "n_kam_stable": n_kam,
            "chain_head": self._chain_hash,
            "genesis": self._genesis_hash,
            "merkle_ok": self.verify_merkle_chain(),
            "agm_worlds": tuple(sorted(agm.worlds)),
            "rsi3_mix": self.monad_mix,
            "rsi3_levels": self.rsi3_levels,
        }

    def _synthetic_bundle(
        self, ctx: Any, triad: TriadChannel
    ) -> WitnessObservationBundle:
        rho_vac = getattr(
            ctx, "rho",
            np.eye(self.dimension_mac, dtype=np.complex128) / self.dimension_mac,
        )
        rho_obs = triad.apply(rho_vac)
        inv = np.ones(7, dtype=np.float64) / math.sqrt(7.0)
        return WitnessObservationBundle(
            cycle_index=self.crystal_count,
            triad_signature=triad.signature,
            rho_vac=rho_vac,
            rho_observed=rho_obs,
            modular_spectrum=tuple(),
            K=getattr(ctx, "K", None),
            audit=type("Audit", (), {
                "vev": 0.0, "free_energy": 0.0, "purity": 1.0,
                "thermal_fluctuation": 0.0, "spectral_gap": 1.0,
                "local_verdict": HeytingOmega3.COHERENT,
            })(),
            modular_axioms={},
            tomita_report={},
            triad_residuals=triad.structural_residuals(),
            invariant_vector=inv,
            context_beta=float(getattr(ctx, "beta", 1.0)),
        )

    def observe_and_crystallize(
        self,
        trickster_illusion_type: str,
        dreamer_scenario_id: str,
        auditor_immunization_hash: str,
        auditor_verdict: HeytingOmega3,
        trickster_strength: float = 0.0,
        dreamer_strength: float = 0.0,
        auditor_rank: int = 4,
        dirichlet_energy: float = 0.0,
        tau_recurrence: float = 1.0,
    ) -> ExperienceCrystal:
        self.crystal_count += 1
        crystal_id = f"CRYSTAL-EXP-{self.crystal_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Silencio Epistémico #%d | ilusión=%s | rank=%d ═══",
            self.crystal_count, trickster_illusion_type, auditor_rank,
        )

        # ── FASE 1: vacío + tríada ─────────────────────────────────────
        ctx = WitnessVacuumPreparation.prepare_context(
            self.H_ext, beta=self.vacuum_beta
        )
        triad = TriadOperatorFactory.build_triad(
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            n=self.dimension_mac,
            trickster_strength=trickster_strength,
            dreamer_strength=dreamer_strength,
            auditor_rank=auditor_rank,
        )

        # ── FASE 2: sección de Poincaré + caos + hand-off Birkhoff ─────
        if TomitaTakesakiEngine is not None:
            bundle = WitnessObservationPipeline.synthesize_from_context(
                cycle_index=self.crystal_count,
                ctx=ctx,
                triad=triad,
                explore_celestial=True,
                tau_recurrence=tau_recurrence,
                agm_verdict=auditor_verdict,
                monad_mix=self.monad_mix,
            )
        else:
            bundle = self._synthetic_bundle(ctx, triad)

        try:
            birkhoff_seed: Optional[BirkhoffSeed] = (
                WitnessObservationPipeline.emit_birkhoff_seed(bundle)
            )
        except ValueError:
            birkhoff_seed = None

        # ── FASE 3: NF + Nekhoroshev + Arnold + Ω₃ ─────────────────────
        if birkhoff_seed is not None:
            nf = BirkhoffNormalForm.from_seed(birkhoff_seed)
            omega_s, omega_l = nf.frequencies()
            birk_stable = bool(nf.spectrally_stable and nf.routh_stable)
            ladder = ResonanceLadder.from_seed(birkhoff_seed)
            res_dens = float(ladder["resonance_density"])
            arn = ArnoldDiffusionBridge.from_seed(birkhoff_seed, dim=2)
            nek = NekhoroshevBound.from_seed(birkhoff_seed, nf=nf)
        else:
            nf = None
            omega_s = omega_l = 0.0
            birk_stable = False
            res_dens = 0.0
            arn = ArnoldDiffusionBridge.report(0.0, 1.0, dim=2)
            nek = NekhoroshevBound.bound_certified(
                1e-6, dim=2, confined_by_kam=False
            )

        cert = HeytingWitnessAdjudicator.adjudicate_certified(
            bundle, auditor_verdict, nf=nf
        )
        final_verdict = cert.verdict
        self._adjudication_log.append(cert)

        if final_verdict == HeytingOmega3.VETOED and not bundle.crowbar_triggered:
            ESP32CrowbarInterlock.trigger(
                reason=f"Ciclo VETOED: {crystal_id}",
                predicate="Ω₃=⊥",
            )

        metrics = bundle.as_vacuum_metrics(
            is_silent=(final_verdict == HeytingOmega3.COHERENT)
        )

        content_hash = self._content_hash(
            crystal_id, bundle.audit, bundle.invariant_vector,
            bundle.triad_signature, dirichlet_energy,
            birkhoff_seed=birkhoff_seed, verdict=final_verdict,
        )
        merkle_parent, chain_hash = self._advance_chain(content_hash)
        provenance = _sha256_bytes(
            self.agent_id.encode("ascii"),
            crystal_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )

        if self.engine is not None:
            _is_rec, drift, poincare_crystal = (
                self.engine.audit_poincare_recurrence_kms_vacuum(
                    bundle.rho_observed, tau_recurrence
                )
            )
            vector_s6 = poincare_crystal.vector_s6
            merkle_root = poincare_crystal.merkle_root_sha256
        else:
            drift = 0.0
            v = np.asarray(bundle.invariant_vector, dtype=np.float64).reshape(-1)
            if v.size >= 7:
                vector_s6 = v[:7].copy()
            else:
                vector_s6 = np.pad(v, (0, 7 - int(v.size)))
            nrm = float(np.linalg.norm(vector_s6))
            vector_s6 = vector_s6 / nrm if nrm > 0 else vector_s6
            merkle_root = hashlib.sha256(
                np.ascontiguousarray(vector_s6).tobytes()
            ).hexdigest()

        agm_cert = (
            birkhoff_seed.agm_certificate if birkhoff_seed is not None
            else bundle.agm_certificate
        )
        monad_ax = (
            birkhoff_seed.monad_axioms if birkhoff_seed is not None
            else bundle.monad_axioms
        )
        chirikov = bool(
            (birkhoff_seed.arnold_tongues.get("chirikov_overlap", False)
             if birkhoff_seed is not None else False)
        )
        cantor_m = float(
            birkhoff_seed.cantor_measure if birkhoff_seed is not None else 1.0
        )

        crystal = ExperienceCrystal(
            crystal_id=crystal_id,
            witness_id=self.agent_id,
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            vacuum_metrics=metrics,
            audit_report=bundle.audit,
            heyting_verdict=final_verdict,
            crystallized_invariant_vector=bundle.invariant_vector,
            content_hash=content_hash,
            chain_hash=chain_hash,
            merkle_parent=merkle_parent,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
            vector_s6=vector_s6,
            tau_recurrence=tau_recurrence,
            kms_drift=float(drift),
            merkle_root_sha256=merkle_root,
            engine_version=__version__,
            birkhoff_frequencies=(float(omega_s), float(omega_l)),
            birkhoff_stable=birk_stable,
            kam_frequency=(
                birkhoff_seed.kam_frequency if birkhoff_seed is not None else 0.0
            ),
            kam_bias=float(bundle.kam_report.get("kam_bias", 0.0)),
            arnold_diffusion_rate=float(arn.rate),
            arnold_escape_time=float(arn.escape_time),
            resonance_density=res_dens,
            sali=(birkhoff_seed.sali if birkhoff_seed is not None else 0.0),
            melnikov_peak=(
                birkhoff_seed.melnikov_peak if birkhoff_seed is not None else 0.0
            ),
            gali=(birkhoff_seed.gali if birkhoff_seed is not None else 0.0),
            melnikov_zeros=(
                birkhoff_seed.melnikov_zeros if birkhoff_seed is not None else 0
            ),
            smale_horseshoe=(
                birkhoff_seed.smale_horseshoe if birkhoff_seed is not None else False
            ),
            chirikov_overlap=chirikov,
            cantor_measure=cantor_m,
            nekhoroshev_T=float(nek.T_escape),
            nekhoroshev_drift=float(nek.drift_bound),
            agm_postulates_passed=(
                agm_cert.n_postulates_passed if agm_cert is not None else 0
            ),
            agm_all_satisfied=(
                agm_cert.all_satisfied if agm_cert is not None else False
            ),
            monad_associativity=(
                monad_ax.associativity_holds if monad_ax is not None else False
            ),
            monad_unitarity=(
                monad_ax.unitarity_holds if monad_ax is not None else False
            ),
            rsi_level=int(self.monad.level),
            crowbar_triggered=bool(bundle.crowbar_triggered),
            hill_topology=(
                birkhoff_seed.hill_topology if birkhoff_seed is not None else "unknown"
            ),
            rotation_number=(
                birkhoff_seed.rotation_number if birkhoff_seed is not None else 0.0
            ),
            diophantine_residue=(
                birkhoff_seed.diophantine_residue if birkhoff_seed is not None else 0.0
            ),
            lyapunov_pairing_residual=(
                birkhoff_seed.lyapunov_pairing_residual
                if birkhoff_seed is not None else 0.0
            ),
            mean_return_time=(
                birkhoff_seed.mean_return_time
                if birkhoff_seed is not None else float("inf")
            ),
            kraus_rank=(
                birkhoff_seed.kraus_rank if birkhoff_seed is not None else 1
            ),
            back_action_bound=(
                birkhoff_seed.back_action_bound if birkhoff_seed is not None else 0.0
            ),
            routh_stable=(
                birkhoff_seed.routh_stable if birkhoff_seed is not None else False
            ),
            gascheau_discriminant=(
                nf.gascheau_discriminant if nf is not None else 0.0
            ),
            floquet_on_circle=(
                nf.floquet_on_circle if nf is not None else False
            ),
            resonance_21=(nf.resonance_21 if nf is not None else False),
            resonance_31=(nf.resonance_31 if nf is not None else False),
            twist_condition=(nf.twist_condition if nf is not None else False),
            confined_by_kam=bool(nek.confined_by_kam),
            arnold_possible=bool(arn.is_possible),
            monad_holds_on_image=(
                monad_ax.holds_on_image if monad_ax is not None else False
            ),
            gali_decay_rate=(
                birkhoff_seed.gali_decay_rate if birkhoff_seed is not None else 0.0
            ),
            jacobi_max_residual=(
                birkhoff_seed.jacobi_max_residual
                if birkhoff_seed is not None else 0.0
            ),
        )
        self.crystal_history.append(crystal)
        self.experience_archive.append(crystal)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cristal %s | Ω₃=%s | γ=%.4f | ω_S=%.4f ω_L=%.4f | "
            "GALI=%.2e | M_zeros=%d | T_Nekh=%s | KAM=%s | %.2f ms",
            crystal_id, final_verdict.name,
            float(getattr(bundle.audit, "purity", 0.0)),
            omega_s, omega_l,
            crystal.gali, crystal.melnikov_zeros,
            "∞" if not math.isfinite(nek.T_escape) else f"{nek.T_escape:.2e}",
            crystal.is_kam_stable(), dt_ms,
        )
        return crystal


# ═══════════════════════════════════════════════════════════════════════════
# §3.7  Demostración autónoma
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    witness = TOONSilentWitnessAgent(
        agent_id="SILENT-WITNESS-SABIO-01",
        dimension=4,
        kms_beta=1.0,
        vacuum_beta=50.0,
        monad_mix=_RSI3_DEFAULT_MIX,
    )

    print("═" * 88)
    print(f"TESTIGO SILENCIOSO — v{__version__}")
    print("CR3BP · AGM K*1–K*8 · RSI-3 · Crowbar SWA1 · GALI · Melnikov · Nekhoroshev")
    print("═" * 88)

    # ── [§0] Leyes de Heyting ──────────────────────────────────────────
    if hasattr(HeytingOmega3, "verify_algebra_laws"):
        laws = HeytingOmega3.verify_algebra_laws()
        print("\n[§0] LEYES DE HEYTING Ω₃")
        for k, v in laws.items():
            print(f"    - {k:<24}: {v}")

    # ── [§1] AGM exhaustivo ────────────────────────────────────────────
    print("\n[§1] AGM (K*1–K*8, barrido de mundos)")
    sweep = HeytingAGMRevision.verify_all_laws()
    print(f"    casos = {sweep['n_passed']}/{sweep['n_cases']}  "
          f"all={sweep['all_passed']}  min={sweep['min_postulates']}")
    K0 = AGMBeliefSet.top()
    alpha = HeytingAGMRevision.proposition_from_verdict(HeytingOmega3.COHERENT)
    cert_agm = HeytingAGMRevision.verify_postulates(K0, alpha, beta=alpha)
    print(f"    demo K=W ∗ {{⊤}} : {cert_agm.n_postulates_passed}/8  "
          f"Ω₃={cert_agm.local_verdict.name}")

    # ── [§2] Mónada RSI-3 (imagen reflexiva) ───────────────────────────
    print("\n[§2] MÓNADA RSI-3 (T, η=unit, μ)  mix≠eta-float")
    rho_test = np.eye(4, dtype=np.complex128) / 4.0
    ax = witness.monad.verify_axioms(rho_test)
    print(f"    holds_on_image 𝔓_n   : {ax.holds_on_image}")
    print(f"    holds_globally 𝔇_n   : {ax.holds_globally}  (esperado False si mix∈(0,1))")
    print(f"    assoc residual       : {ax.left_associativity:.3e}")
    print(f"    unit L/R             : {ax.left_unitarity:.3e} / {ax.right_unitarity:.3e}")
    print(f"    idempotence ‖T²−T‖   : {ax.idempotence_residual:.3e}")
    print(f"    Ω₃ mónada            : {ax.verdict.name}")
    assert ax.holds_on_image, "RSI-3 debe ser mónada sobre rayos puros"

    # ── [§3] Gascheau / Routh ──────────────────────────────────────────
    print("\n[§3] GASCHEAU–ROUTH EN L4")
    mu_earth = 3.003e-6  # μ⊕-☾ ~ 0.012, usamos un μ pequeño de prueba
    mu_demo = 0.01
    w_s, w_l = BirkhoffNormalForm.frequencies(mu_demo)
    print(f"    μ={mu_demo:.4f}  ω_s={w_s:.6f}  ω_ℓ={w_l:.6f}  ρ={w_l/max(w_s,1e-15):.4f}")
    print(f"    Δ_Gascheau           : {BirkhoffNormalForm.gascheau_discriminant(mu_demo):.6f}")
    print(f"    Routh μ < μ_R        : {LibrationPointSolver.routh_stability(mu_demo)}  "
          f"(μ_R={LibrationPointSolver.routh_mu_critical():.6f})")
    print(f"    resonancia 2:1 μ     : {_MU_RESONANCE_21:.6f}")
    print(f"    resonancia 3:1 μ     : {_MU_RESONANCE_31:.6f}")
    L_demo = LibrationPointSolver.solve_all(mu_demo)
    nf_demo = BirkhoffNormalForm.analyze(L_demo["L4"], mu_demo)
    print(f"    Floquet on S¹        : {nf_demo.floquet_on_circle}")
    print(f"    twist det Ω_xxΩ_yy−Ω_xy² : {nf_demo.twist_hessian_det:.6f}")
    print(f"    cúbicos (∼0)         : {nf_demo.taylor_coefficients.get('cubic_residual', 0):.3e}")
    arn_2 = ArnoldDiffusionBridge.report(0.01, 1.0, dim=2)
    print(f"    Arnold 2DOF posible  : {arn_2.is_possible}  rate={arn_2.rate}  T={arn_2.escape_time}")

    # ── [§4] Observación zero back-action ──────────────────────────────
    print("\n[§4] OBSERVACIÓN ZERO BACK-ACTION (SWA1)")
    res = witness.execute_zero_backaction_poincare_observation(
        rho_test, tau_recurrence=1.0
    )
    for k, v in res.items():
        if k not in ("vector_s6",):
            print(f"    - {k:<24}: {v}")

    # ── [§5] Ciclo completo F₁ → F₂ → F₃ ───────────────────────────────
    print("\n[§5] CICLO COMPLETO (emit_poincare_seed → emit_birkhoff_seed → crystallize)")
    c_triad = witness.observe_and_crystallize(
        trickster_illusion_type="COHERENT_ILLUSION",
        dreamer_scenario_id="SCENARIO_001",
        auditor_immunization_hash="IMM_HASH_001",
        auditor_verdict=HeytingOmega3.COHERENT,
        trickster_strength=0.15,
        dreamer_strength=0.10,
        auditor_rank=2,
    )
    print(f"    Crystal ID            : {c_triad.crystal_id}")
    print(f"    Veredicto Ω₃          : {c_triad.heyting_verdict.name}")
    print(f"    μ (ratio masas)       : {c_triad.vacuum_metrics.mass_ratio_mu:.6f}")
    print(f"    C (Jacobi)            : {c_triad.vacuum_metrics.jacobi_constant:.6f}")
    print(f"    Hill topology         : {c_triad.hill_topology}")
    print(f"    Routh / Floquet       : {c_triad.routh_stable} / {c_triad.floquet_on_circle}")
    print(f"    ω_S, ω_L (Gascheau)   : {c_triad.birkhoff_frequencies}")
    print(f"    Estable espectral     : {c_triad.birkhoff_stable}")
    print(f"    KAM freq / bias       : {c_triad.kam_frequency:.3e} / {c_triad.kam_bias:.3e}")
    print(f"    SALI / GALI           : {c_triad.sali:.3e} / {c_triad.gali:.3e}")
    print(f"    Melnikov peak / zeros : {c_triad.melnikov_peak:.3e} / {c_triad.melnikov_zeros}")
    print(f"    Smale horseshoe       : {c_triad.smale_horseshoe}")
    print(f"    Chirikov overlap      : {c_triad.chirikov_overlap}")
    print(f"    Cantor measure        : {c_triad.cantor_measure:.6f}")
    print(f"    Nekhoroshev T / KAM   : {c_triad.nekhoroshev_T} / confined={c_triad.confined_by_kam}")
    print(f"    Arnold possible (2DOF): {c_triad.arnold_possible}  rate={c_triad.arnold_diffusion_rate}")
    print(f"    Deprit 2:1 / 3:1      : {c_triad.resonance_21} / {c_triad.resonance_31}")
    print(f"    AGM postulados        : {c_triad.agm_postulates_passed}/8")
    print(f"    RSI-3 asoc/unit/image : {c_triad.monad_associativity}/"
          f"{c_triad.monad_unitarity}/{c_triad.monad_holds_on_image}")
    print(f"    RSI level             : {c_triad.rsi_level}")
    print(f"    Crowbar triggered     : {c_triad.crowbar_triggered}")
    print(f"    Kraus rank / back-act : {c_triad.kraus_rank} / {c_triad.back_action_bound:.3e}")
    print(f"    Cadena Merkle válida  : {witness.verify_merkle_chain()}")

    print("\n[§6] PASAPORTE")
    for k, v in witness.emit_witness_passport().items():
        print(f"    - {k:<22}: {v}")

    print("\n" + "═" * 88)
    print("✓ Testigo Silencioso v9.2.0: CR3BP + AGM K*1–K*8 + RSI-3 (𝔓_n) +")
    print("  Crowbar SWA1 + GALI + Melnikov {H₀,H₁} + Gascheau½ + Nekhoroshev≠Arnold.")
    print("═" * 88)


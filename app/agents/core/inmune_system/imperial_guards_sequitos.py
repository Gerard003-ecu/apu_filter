# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Homotopic Séquitos Agent (Capa 1.5 de Calibre de Consenso)          ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_sequitos.py           ║
║ Versión: 3.0.0-Nested-Phases-Heyting-Kleisli-DeGroot-CHSH-OODA-CAS           ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y CATEGORIAL DE DE RHAM:
────────────────────────────────────────────────────────────────────────────────
Orquesta la concurrencia táctica en la Capa 1.5 ($V_{\mathrm{PHYSICS}} \subset V_{\mathrm{SEQUITOS}} \subset V_{\mathrm{TACTICS}}$)
de las sub-tríadas agénticas para evitar la dispersión de fase, polarización semántica e inyecciones de código mediante tres aduanas:

1. ADUANA DE ASOCIATIVIDAD MONÁDICA DE KLEISLI (TEORÍA DE CATEGORÍAS):
   Encapsula el estado de las variables tácticas bajo la mónada de estado de Kleisli
   $\mathbb{T} = (T, \eta, \mu)$. Exige la conmutatividad estricta del asociaedro de Kleisli,
   verificando la aniquilación del residuo asociativo:
   $$\epsilon_{\mathrm{Kleisli}} = \left| P(h \bullet (g \bullet f)) - P((h \bullet g) \bullet f) \right| \equiv 0$$
   Donde $(g \bullet f)(x) = \mu_C \circ T(g) \circ f(x)$ es la composición monádica.

2. ADUANA DE CONSENSO CONTINUO DE DEGROOT (TEORÍA DE GRAFOS):
   Modeliza la convergencia de opiniones en el grafo de afinidad agéntico. Exige 
   que la tasa de convergencia asintótica esté acotada exponencialmente por la 
   brecha espectral $\lambda_2$ del Laplaciano normalizado del haz de afinidad.

3. ADUANA DE INMUNIDAD CUÁNTICA MULTIPARTITA BELL-CHSH:
   Audita la correlación conjunta de los mensajes cifrados transmitidos por la tríada,
   exigiendo la violación de la desigualdad clásica de Bell bajo el Límite de Tsirelson:
   $$\langle B_{\mathrm{CHSH}} \rangle = \left| E(a,b) - E(a,b') + E(a',b) + E(a',b') \right| \le 2\sqrt{2}$$
   Una caída por debajo de la cota clásica ($\le 2.0$) delata suplantación de identidad o dispersión de fase.

INVARIANTES CATEGÓRICOS Y DE HARDWARE PERIMETRAL:
────────────────────────────────────────────────────────────────────────────────
- Invarianza de calibre respecto a la conmutación de base en el functor de Kleisli.
- Preservación de la completez fuerte sobre el retículo distributivo de Heyting $\Omega_3$.
- Estabilidad de Lyapunov global asintótica en la dinámica lineal de opinión.
- Veto en el retículo de Heyting $\Omega_3 = \{\text{COHERENT}, \text{DEGRADED}, \text{VETOED}\}$ ($\top = \text{VETOED}$).
- Interrupción perimetral ESP32 en IRAM ($t_{\text{actuation}} \le 400\,\text{ns}$) activando el tiristor BT151 (Crowbar) vía GPIO14.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, Optional, Tuple

import numpy as np

try:
    from app.core.inmune_system.imperial_sequitos_engine import ImperialSequitosEngine
except ImportError:  # pragma: no cover — import plano / tests locales
    from imperial_sequitos_engine import ImperialSequitosEngine

logger = logging.getLogger("APU.Agents.HomotopicSequitos")

__version__: Final[str] = "3.0.0-Nested-Phases-Heyting-Kleisli-DeGroot-CHSH-OODA-CAS"


# =============================================================================
# CONSTANTES DE CONTROL LÓGICO Y METROLOGÍA
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_INTERLOCK_LATENCY_BUDGET_NS: Final[float] = 400.0  # presupuesto lógico (API 2.0)
_INTERLOCK_JITTER_NS: Final[float] = 5.0
_WILKINSON_REL_SCALE: Final[float] = 10.0

_KLEISLI_COHERENT_TOL: Final[float] = 1e-10
_KLEISLI_DEGRADED_TOL: Final[float] = 1e-8
_DEGROOT_COHERENT_DEV: Final[float] = 1e-6
_DEGROOT_DEGRADED_DEV: Final[float] = 1e-4

_TSIRELSON_BOUND: Final[float] = float(2.0 * np.sqrt(2.0))
_CLASSICAL_CHSH_BOUND: Final[float] = 2.0
_PR_NOSIGNAL_BOUND: Final[float] = 4.0
_TSIRELSON_GUARD_BASE: Final[float] = 1e-3
_CORRELATOR_BOUND: Final[float] = 1.0

_HEYTING_ORDER: Final[Dict[str, int]] = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
_HEYTING_GODEL: Final[Dict[str, float]] = {"COHERENT": 1.0, "DEGRADED": 0.5, "VETOED": 0.0}
_REVERSE_HEYTING: Final[Dict[int, str]] = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}

KleisliArrow = Callable[[Any], Tuple[Any, float]]


# =============================================================================
# FASE I — NÚCLEO DE AUDITORÍA ESPECTRAL (MOTOR CIEGO)
# -----------------------------------------------------------------------------
# Objetos: métricas crudas de Kleisli / DeGroot / CHSH, certificados 3.0
#          si el motor los expone, validación de agentes / correladores.
# Morfismo terminal (I.8): synthesize_heyting_audit_germ
#          ≅ objeto inicial de la Fase II (valuación en H₃).
# =============================================================================
@dataclass(frozen=True)
class _KleisliRawResult:
    """Desviación de asociatividad (h ⋆ (g ⋆ f)) ∼ ((h ⋆ g) ⋆ f)."""

    deviation: float
    lhs_prob: float = float("nan")
    rhs_prob: float = float("nan")
    value_mismatch: float = 0.0
    engine_ok: bool = True


@dataclass(frozen=True)
class _DeGrootRawResult:
    """Métricas crudas del consenso de DeGroot / Olfati–Saber."""

    final_opinions: np.ndarray
    fiedler_value: float
    deviation: float
    discrete_opinions: np.ndarray = None  # type: ignore[assignment]
    connected: bool = True
    cheeger_upper: float = float("nan")
    mixing_rate: float = float("nan")
    engine_verdict: str = ""
    is_reversible: bool = False
    engine_ok: bool = True

    def __post_init__(self) -> None:
        if self.discrete_opinions is None:
            object.__setattr__(
                self, "discrete_opinions", np.asarray(self.final_opinions).copy()
            )


@dataclass(frozen=True)
class _CHSHRawResult:
    """Observable de Bell–CHSH y certificados de Horodecki / Tsirelson."""

    s_value: float
    engine_verdict: str = ""
    physical: bool = True
    tsirelson_gap: float = float("nan")
    classical_gap: float = float("nan")
    pr_gap: float = float("nan")
    horodecki_bound: float = float("nan")
    engine_ok: bool = True


@dataclass(frozen=True)
class _HeytingAuditGerm:
    """
    Gérmen de auditoría de Heyting (objeto terminal de la Fase I).

    Es el objeto inicial de la Fase II: transporta las métricas crudas
    (Kleisli, DeGroot, CHSH) y las escalas de Wilkinson con las que el
    clasificador H₃ decide COHERENT / DEGRADED / VETOED. No contiene
    aún veredictos: la valuación ν : métricas → H₃ es el trabajo de
    la Fase II.

    Atributos
    ---------
    kleisli, degroot, chsh:
        Auditorías crudas (∞ si el motor falló).
    n_agents:
        Cardinalidad del objeto de consenso (dimensión del séquito).
    safety_margin:
        Factor de holgura del soberano (≥ 0).
    kleisli_scale, degroot_scale:
        Escalas de referencia para umbrales relativos.
    """

    kleisli: _KleisliRawResult
    degroot: _DeGrootRawResult
    chsh: _CHSHRawResult
    n_agents: int
    safety_margin: float
    kleisli_scale: float
    degroot_scale: float


class _AuditCore:
    """
    Fase I. Núcleo ciego que habla con ImperialSequitosEngine.

    Consume `*_certified` si el motor 3.0 está presente; si no, se
    repliega a las tuplas 2.0. Nunca interpreta Heyting: sólo mide.
    """

    def __init__(self, engine: ImperialSequitosEngine, n_agents: int) -> None:
        self._engine = engine
        self._n = int(n_agents)

    @property
    def engine(self) -> ImperialSequitosEngine:
        return self._engine

    @staticmethod
    def _as_vec(name: str, values: Any, dim: Optional[int] = None) -> np.ndarray:
        try:
            arr = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no es convertible a ndarray.") from exc
        vec = np.asarray(arr).reshape(-1)
        if vec.size == 0:
            raise ValueError(f"{name} no puede ser vacío.")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} contiene no-finitos.")
        if dim is not None and vec.size != dim:
            raise ValueError(f"{name} debe tener dimensión {dim}; recibido {vec.size}.")
        return vec.astype(np.float64, copy=False)

    @staticmethod
    def _as_matrix(name: str, values: Any, square: bool = False) -> np.ndarray:
        try:
            arr = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no es convertible a ndarray.") from exc
        if arr.ndim == 1:
            side = int(np.sqrt(arr.size))
            if side * side != arr.size:
                raise ValueError(f"{name} plana no es un cuadrado perfecto.")
            arr = arr.reshape(side, side)
        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser de rango 2.")
        if square and arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene no-finitos.")
        return arr

    @staticmethod
    def _prob_of(pair: Tuple[Any, float], name: str) -> Tuple[Any, float]:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{name} debe retornar (valor, probabilidad).")
        value, prob = pair
        pf = float(prob)
        if not math.isfinite(pf):
            raise ValueError(f"{name} produjo una probabilidad no finita.")
        return value, pf

    @staticmethod
    def _value_mismatch(lhs: Any, rhs: Any) -> float:
        """Distancia numérica entre valores de Kleisli; 0 si no son comparables."""
        try:
            a = np.asarray(lhs, dtype=np.float64)
            b = np.asarray(rhs, dtype=np.float64)
        except (TypeError, ValueError):
            return 0.0 if lhs == rhs else 1.0
        if a.shape != b.shape:
            return float("inf")
        if a.size == 0:
            return 0.0
        delta = a - b
        return float(np.sqrt(max(float(np.real(np.vdot(delta, delta))), 0.0)))

    def compute_kleisli_deviation(
        self,
        f: KleisliArrow,
        g: KleisliArrow,
        h_func: KleisliArrow,
        test_input: Any,
    ) -> _KleisliRawResult:
        """
        Desviación de asociatividad en Kl(Writer_([0,1],×)):

            (h ⋆ (g ⋆ f))(x)  vs  ((h ⋆ g) ⋆ f)(x).

        Reporta |p_L − p_R| y, si los valores son numéricos, ‖v_L − v_R‖.
        """
        try:
            if not (callable(f) and callable(g) and callable(h_func)):
                raise TypeError("f, g y h_func deben ser callables de Kleisli.")
            compose = self._engine.kleisli_compose
            g_f = compose(f, g)
            lhs_func = compose(g_f, h_func)
            h_g = compose(g, h_func)
            rhs_func = compose(f, h_g)

            v_lhs, p_lhs = self._prob_of(lhs_func(test_input), "lhs")
            v_rhs, p_rhs = self._prob_of(rhs_func(test_input), "rhs")
            deviation = float(abs(p_lhs - p_rhs))
            mismatch = self._value_mismatch(v_lhs, v_rhs)
            return _KleisliRawResult(
                deviation=deviation,
                lhs_prob=float(p_lhs),
                rhs_prob=float(p_rhs),
                value_mismatch=float(mismatch),
                engine_ok=True,
            )
        except Exception as exc:
            logger.error("Fallo en cómputo de Kleisli: %s", exc)
            return _KleisliRawResult(deviation=float("inf"), engine_ok=False)

    def compute_degroot_metrics(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> _DeGrootRawResult:
        """Consenso de DeGroot: vector final, Fiedler, desviación y certificados."""
        empty = _DeGrootRawResult(
            final_opinions=np.array([], dtype=np.float64),
            fiedler_value=float("inf"),
            deviation=float("inf"),
            discrete_opinions=np.array([], dtype=np.float64),
            connected=False,
            engine_ok=False,
        )
        try:
            x = self._as_vec("opinion_vector", opinion_vector)
            w = self._as_matrix("affinity_matrix", affinity_matrix, square=True)
            if w.shape[0] != x.size:
                raise ValueError(
                    "La afinidad debe ser n×n y coincidir con el vector de opinión."
                )
            if int(steps) < 0:
                raise ValueError("steps debe ser no negativo.")

            certified = getattr(
                self._engine, "compute_degroot_spectral_consensus_certified", None
            )
            if callable(certified):
                result = certified(x, w, steps)
                opinions = np.asarray(result.final_opinion, dtype=np.float64)
                if opinions.size:
                    mean = float(np.mean(opinions))
                    deviation = float(np.sqrt(max(float(np.mean((opinions - mean) ** 2)), 0.0)))
                else:
                    deviation = float(getattr(result, "deviation", float("inf")))
                if np.isfinite(getattr(result, "deviation", float("nan"))):
                    deviation = float(result.deviation)
                return _DeGrootRawResult(
                    final_opinions=opinions,
                    fiedler_value=float(result.fiedler_value),
                    deviation=float(deviation),
                    discrete_opinions=np.asarray(
                        getattr(result, "discrete_opinion", opinions), dtype=np.float64
                    ),
                    connected=bool(getattr(result, "connected", True)),
                    cheeger_upper=float(getattr(result, "cheeger_upper", float("nan"))),
                    mixing_rate=float(getattr(result, "mixing_rate", float("nan"))),
                    engine_verdict=str(getattr(result, "verdict", "")),
                    is_reversible=bool(getattr(result, "is_reversible", False)),
                    engine_ok=True,
                )

            final_opinions, fiedler, engine_verdict = (
                self._engine.compute_degroot_spectral_consensus(x, w, steps)
            )
            opinions = np.asarray(final_opinions, dtype=np.float64)
            if opinions.size:
                mean = float(np.mean(opinions))
                deviation = float(np.sqrt(max(float(np.mean((opinions - mean) ** 2)), 0.0)))
            else:
                deviation = float("inf")
            return _DeGrootRawResult(
                final_opinions=opinions,
                fiedler_value=float(fiedler),
                deviation=float(deviation),
                discrete_opinions=opinions.copy(),
                engine_verdict=str(engine_verdict),
                engine_ok=True,
            )
        except Exception as exc:
            logger.error("Fallo en consenso de DeGroot: %s", exc)
            return empty

    def compute_chsh_s_value(self, correlation_matrix: np.ndarray) -> _CHSHRawResult:
        """Valor S de Bell–CHSH y, si existe, certificado de Horodecki."""
        try:
            e = self._as_matrix("correlation_matrix", correlation_matrix, square=True)
            certified = getattr(self._engine, "verify_chsh_violation_certified", None)
            if callable(certified):
                result = certified(e)
                return _CHSHRawResult(
                    s_value=float(result.s_value),
                    engine_verdict=str(getattr(result, "verdict", "")),
                    physical=bool(getattr(result, "physical", True)),
                    tsirelson_gap=float(getattr(result, "tsirelson_gap", float("nan"))),
                    classical_gap=float(getattr(result, "classical_gap", float("nan"))),
                    pr_gap=float(getattr(result, "pr_gap", float("nan"))),
                    horodecki_bound=float(getattr(result, "horodecki_bound", float("nan"))),
                    engine_ok=True,
                )
            s_value, engine_verdict = self._engine.verify_chsh_violation(e)
            phys = bool(np.all(np.abs(np.real(e)) <= _CORRELATOR_BOUND + 1e-12))
            return _CHSHRawResult(
                s_value=float(s_value),
                engine_verdict=str(engine_verdict),
                physical=phys,
                tsirelson_gap=float(_TSIRELSON_BOUND - float(s_value)),
                classical_gap=float(float(s_value) - _CLASSICAL_CHSH_BOUND),
                pr_gap=float(_PR_NOSIGNAL_BOUND - float(s_value)),
                engine_ok=True,
            )
        except Exception as exc:
            logger.error("Fallo en verificación CHSH: %s", exc)
            return _CHSHRawResult(s_value=float("inf"), physical=False, engine_ok=False)

    # ── I.8  Morfismo terminal de la Fase I ───────────────────────────────
    def synthesize_heyting_audit_germ(
        self,
        f: KleisliArrow,
        g: KleisliArrow,
        h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        safety_margin: float,
        steps: int = 100,
    ) -> _HeytingAuditGerm:
        """
        I.8 — Morfismo terminal de la Fase I / objeto inicial de la Fase II.

        Ensambla el gérmen de auditoría

            𝒢_I = (Kl_raw, DeGroot_raw, CHSH_raw, n, μ_safety, σ_Kl, σ_DG)

        sobre el cual la Fase II define la valuación de Heyting
        ν : métricas → H₃. Las escalas σ se toman de las propias
        métricas (max(1, |p|, ‖x‖_∞)) para que los umbrales sean
        relativos (Wilkinson) y no sólo absolutos.

        Este método *es* el arranque formal de `_HeytingClassifier`.
        """
        kleisli = self.compute_kleisli_deviation(f, g, h_func, test_input)
        degroot = self.compute_degroot_metrics(opinion_vector, affinity_matrix, steps)
        chsh = self.compute_chsh_s_value(correlation_matrix)

        kl_scale = 1.0
        if np.isfinite(kleisli.lhs_prob) or np.isfinite(kleisli.rhs_prob):
            kl_scale = max(
                abs(kleisli.lhs_prob) if np.isfinite(kleisli.lhs_prob) else 0.0,
                abs(kleisli.rhs_prob) if np.isfinite(kleisli.rhs_prob) else 0.0,
                1.0,
            )
        dg_scale = 1.0
        if degroot.final_opinions.size:
            dg_scale = max(float(np.max(np.abs(degroot.final_opinions))), 1.0)

        n_agents = self._n
        if degroot.final_opinions.size:
            n_agents = int(degroot.final_opinions.size)
        else:
            try:
                n_agents = int(self._as_vec("opinion_vector", opinion_vector).size)
            except Exception:
                pass

        return _HeytingAuditGerm(
            kleisli=kleisli,
            degroot=degroot,
            chsh=chsh,
            n_agents=int(n_agents),
            safety_margin=float(max(safety_margin, 0.0)),
            kleisli_scale=float(kl_scale),
            degroot_scale=float(dg_scale),
        )


# =============================================================================
# FASE II — CLASIFICADOR DE HEYTING H₃ Y LIFTING OODA
# -----------------------------------------------------------------------------
# Continúa I.8: todo veredicto se instancia desde un HeytingAuditGerm.
# Morfismo terminal (II.6): induce_ooda_actuation_germ
#          ≅ objeto inicial de la Fase III (Observe/Orient).
# =============================================================================
@dataclass(frozen=True)
class _KleisliVeredict:
    """Asociatividad de Kleisli valuada en H₃."""

    deviation: float
    verdict: str
    lhs_prob: float = float("nan")
    rhs_prob: float = float("nan")
    value_mismatch: float = 0.0
    threshold_coherent: float = _KLEISLI_COHERENT_TOL
    threshold_degraded: float = _KLEISLI_DEGRADED_TOL
    godel_value: float = 0.0


@dataclass(frozen=True)
class _DeGrootVeredict:
    """Consenso de DeGroot valuado en H₃."""

    fiedler_value: float
    deviation: float
    verdict: str
    connected: bool = True
    cheeger_upper: float = float("nan")
    mixing_rate: float = float("nan")
    engine_verdict: str = ""
    threshold_coherent: float = _DEGROOT_COHERENT_DEV
    threshold_degraded: float = _DEGROOT_DEGRADED_DEV
    godel_value: float = 0.0


@dataclass(frozen=True)
class _CHSHVeredict:
    """Canal de Bell valuado en H₃ (cotas 2 ≤ 2√2 ≤ 4)."""

    s_value: float
    verdict: str
    physical: bool = True
    tsirelson_gap: float = float("nan")
    classical_gap: float = float("nan")
    horodecki_bound: float = float("nan")
    effective_tsirelson: float = _TSIRELSON_BOUND
    godel_value: float = 0.0


@dataclass(frozen=True)
class _OODAActuationGerm:
    """
    Gérmen OODA (objeto terminal de la Fase II).

    Es el objeto inicial de la Fase III: la terna de veredictos locales
    (Kleisli, DeGroot, CHSH) ya valuados en H₃, su join de Gödel y las
    métricas que el ciclo Observe–Orient–Decide–Act colapsa a
    2 = {VIABLE, VETO}.
    """

    kleisli: _KleisliVeredict
    degroot: _DeGrootVeredict
    chsh: _CHSHVeredict
    heyting_join: str
    godel_meet: float
    n_agents: int
    safety_margin: float


class _HeytingClassifier:
    """
    Fase II. Clasificador en el álgebra de Heyting de tres valores.

    Continúa el gérmen 𝒢_I. Hay dos polaridades:

    * Métricas de defecto (Kleisli, DeGroot): menor es mejor.

          m ≤ τ_c          ↦  COHERENT ,
          τ_c < m ≤ τ_d    ↦  DEGRADED ,
          m > τ_d          ↦  VETOED ,

      con τ_• = τ_•⁰ · μ_safety ∨ ε_W · σ.

    * Observable de Bell (CHSH): las cotas son físicas
      |S| ≤ 2 (LHV), |S| ≤ 2√2 (Tsirelson), |S| ≤ 4 (PR).
      μ_safety > 1 estrecha la banda cuántica por debajo de 2√2
      (guarda conservadora, no reescribe la física).

    Fallos del motor o no-finitos ⇒ VETOED.
    """

    def __init__(self, safety_margin: float) -> None:
        self._margin = float(max(safety_margin, 0.0))

    @property
    def safety_margin(self) -> float:
        return self._margin

    @staticmethod
    def canonicalize(verdict: str) -> str:
        return verdict if verdict in _HEYTING_ORDER else "VETOED"

    @staticmethod
    def join(*verdicts: str) -> str:
        """Supremo de Heyting (peor caso)."""
        if not verdicts:
            return "COHERENT"
        idx = max(_HEYTING_ORDER[_HeytingClassifier.canonicalize(v)] for v in verdicts)
        return _REVERSE_HEYTING[idx]

    @staticmethod
    def meet(*verdicts: str) -> str:
        """Ínfimo de Heyting (mejor caso)."""
        if not verdicts:
            return "COHERENT"
        idx = min(_HEYTING_ORDER[_HeytingClassifier.canonicalize(v)] for v in verdicts)
        return _REVERSE_HEYTING[idx]

    def scaled_tol(self, base: float, scale: float = 1.0) -> float:
        """τ = τ₀ · μ_safety, con suelo ε_mach · 10 · σ (Wilkinson)."""
        abs_tol = float(base) * max(self._margin, 0.0)
        rel_tol = max(float(scale), 1.0) * _MACHINE_EPS * _WILKINSON_REL_SCALE
        return float(max(abs_tol, rel_tol, _MACHINE_EPS))

    def verdict_from_deviation(
        self,
        deviation: float,
        coherent_tol: float,
        degraded_tol: float,
        safety_margin: Optional[float] = None,
        scale: float = 1.0,
    ) -> str:
        """
        Clasificador H₃ para métricas de defecto.
        Firma compatible con 2.0 (`safety_margin` opcional).
        """
        if not np.isfinite(deviation):
            return "VETOED"
        margin = self._margin if safety_margin is None else float(max(safety_margin, 0.0))
        tau_c = float(coherent_tol) * margin
        tau_d = float(degraded_tol) * margin
        floor = max(float(scale), 1.0) * _MACHINE_EPS * _WILKINSON_REL_SCALE
        tau_c = max(tau_c, floor, _MACHINE_EPS)
        tau_d = max(tau_d, tau_c)
        if deviation > tau_d:
            return "VETOED"
        if deviation > tau_c:
            return "DEGRADED"
        return "COHERENT"

    def classify_kleisli(self, raw: _KleisliRawResult, scale: float) -> _KleisliVeredict:
        tau_c = self.scaled_tol(_KLEISLI_COHERENT_TOL, scale)
        tau_d = max(self.scaled_tol(_KLEISLI_DEGRADED_TOL, scale), tau_c)
        if (not raw.engine_ok) or (not np.isfinite(raw.deviation)):
            verdict = "VETOED"
        else:
            verdict = self.verdict_from_deviation(
                raw.deviation, _KLEISLI_COHERENT_TOL, _KLEISLI_DEGRADED_TOL, scale=scale
            )
            # Un desacuerdo de valores con probabilidades coincidentes degrada.
            if (
                verdict == "COHERENT"
                and np.isfinite(raw.value_mismatch)
                and raw.value_mismatch > tau_c
            ):
                verdict = "DEGRADED" if raw.value_mismatch <= tau_d else "VETOED"
        return _KleisliVeredict(
            deviation=float(raw.deviation),
            verdict=verdict,
            lhs_prob=float(raw.lhs_prob),
            rhs_prob=float(raw.rhs_prob),
            value_mismatch=float(raw.value_mismatch),
            threshold_coherent=float(tau_c),
            threshold_degraded=float(tau_d),
            godel_value=float(_HEYTING_GODEL[verdict]),
        )

    def classify_degroot(self, raw: _DeGrootRawResult, scale: float) -> _DeGrootVeredict:
        tau_c = self.scaled_tol(_DEGROOT_COHERENT_DEV, scale)
        tau_d = max(self.scaled_tol(_DEGROOT_DEGRADED_DEV, scale), tau_c)
        if (not raw.engine_ok) or (not np.isfinite(raw.deviation)):
            verdict = "VETOED"
        else:
            verdict = self.verdict_from_deviation(
                raw.deviation, _DEGROOT_COHERENT_DEV, _DEGROOT_DEGRADED_DEV, scale=scale
            )
            # Consenso local en un grafo desconexo no es consenso global.
            if verdict == "COHERENT" and not raw.connected:
                verdict = "DEGRADED"
            if raw.engine_verdict in _HEYTING_ORDER:
                verdict = self.join(verdict, raw.engine_verdict)
        return _DeGrootVeredict(
            fiedler_value=float(raw.fiedler_value),
            deviation=float(raw.deviation),
            verdict=verdict,
            connected=bool(raw.connected),
            cheeger_upper=float(raw.cheeger_upper),
            mixing_rate=float(raw.mixing_rate),
            engine_verdict=str(raw.engine_verdict),
            threshold_coherent=float(tau_c),
            threshold_degraded=float(tau_d),
            godel_value=float(_HEYTING_GODEL[verdict]),
        )

    def classify_chsh(self, raw: _CHSHRawResult) -> _CHSHVeredict:
        """
        Polaridad de Bell (API 2.0):

            |S| > 2√2_eff   →  VETOED    (no cuántico / no físico),
            2 < |S| ≤ 2√2_eff →  COHERENT (entrelazamiento legítimo),
            |S| ≤ 2          →  DEGRADED (LHV, canal clásico).

        2√2_eff = 2√2 − (μ−1)_+ · δ_guarda, recortado a (2, 2√2].
        """
        if (not raw.engine_ok) or (not np.isfinite(raw.s_value)):
            verdict = "VETOED"
            s_val = float(raw.s_value)
            eff = _TSIRELSON_BOUND
        else:
            s_val = float(abs(raw.s_value))
            extra = max(self._margin - 1.0, 0.0) * _TSIRELSON_GUARD_BASE
            eff = float(min(max(_TSIRELSON_BOUND - extra, _CLASSICAL_CHSH_BOUND), _TSIRELSON_BOUND))
            if (not raw.physical) or s_val > eff + 8.0 * _MACHINE_EPS:
                verdict = "VETOED"
            elif s_val > _CLASSICAL_CHSH_BOUND:
                verdict = "COHERENT"
            else:
                verdict = "DEGRADED"
            if raw.engine_verdict in _HEYTING_ORDER and raw.engine_verdict == "VETOED":
                verdict = "VETOED"
        return _CHSHVeredict(
            s_value=float(raw.s_value),
            verdict=verdict,
            physical=bool(raw.physical),
            tsirelson_gap=float(raw.tsirelson_gap),
            classical_gap=float(raw.classical_gap),
            horodecki_bound=float(raw.horodecki_bound),
            effective_tsirelson=float(eff if np.isfinite(raw.s_value) else _TSIRELSON_BOUND),
            godel_value=float(_HEYTING_GODEL[verdict]),
        )

    # ── II.6  Morfismo terminal de la Fase II ─────────────────────────────
    def induce_ooda_actuation_germ(self, germ: _HeytingAuditGerm) -> _OODAActuationGerm:
        """
        II.6 — Morfismo terminal de la Fase II / objeto inicial de la Fase III.

        Valúa 𝒢_I en H₃³ y forma el join

            j = ν(Kleisli) ∨ ν(DeGroot) ∨ ν(CHSH) ,

        junto con el meet de Gödel min_i ν_G(i). El par (j, métricas)
        es el objeto que la Fase III observa y orienta en el ciclo OODA.

        Este método *es* el arranque formal de `_OODAController`.
        """
        kl = self.classify_kleisli(germ.kleisli, germ.kleisli_scale)
        dg = self.classify_degroot(germ.degroot, germ.degroot_scale)
        ch = self.classify_chsh(germ.chsh)
        joined = self.join(kl.verdict, dg.verdict, ch.verdict)
        meet_g = float(min(kl.godel_value, dg.godel_value, ch.godel_value))
        return _OODAActuationGerm(
            kleisli=kl,
            degroot=dg,
            chsh=ch,
            heyting_join=joined,
            godel_meet=meet_g,
            n_agents=int(germ.n_agents),
            safety_margin=float(germ.safety_margin),
        )


# =============================================================================
# FASE III — CICLO OODA Y COLAPSO A 2
# -----------------------------------------------------------------------------
# Continúa II.6: el controlador se ancla a un OODAActuationGerm.
# Observe = gérmen; Orient = join H₃; Decide = filtro primo;
# Act = interlock lógico (sin silicio).
# =============================================================================
@dataclass(frozen=True)
class _OODAResult:
    """Acta del ciclo OODA (superset certificado del dict 2.0)."""

    heyting_verdict: str
    kleisli_deviation: float
    kleisli_verdict: str
    fiedler_value: float
    degroot_verdict: str
    chsh_value: float
    chsh_verdict: str
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    godel_meet: float
    degroot_connected: bool
    degroot_deviation: float
    chsh_physical: bool
    tsirelson_gap: float
    filter_is_prime: bool
    observe_ok: bool

    def as_public_dict(self) -> Dict[str, Any]:
        """Contrato 2.0: claves históricas del ciclo."""
        return {
            "heyting_verdict": self.heyting_verdict,
            "kleisli_deviation": self.kleisli_deviation,
            "kleisli_verdict": self.kleisli_verdict,
            "fiedler_value": self.fiedler_value,
            "degroot_verdict": self.degroot_verdict,
            "chsh_value": self.chsh_value,
            "chsh_verdict": self.chsh_verdict,
            "hardware_interlock_fired": self.hardware_interlock_fired,
            "actuation_latency_ns": self.actuation_latency_ns,
        }


class _OODAController:
    """
    Fase III. Ciclo Observe–Orient–Decide–Act.

    Continúa el gérmen 𝒢_II. El filtro primo (principal) sobre H₃ es

        x ∈ 𝒰  ⇔  ∨(Kleisli, DeGroot, CHSH) = VETOED .

    Su función característica es `hardware_interlock_fired`. La latencia
    reportada es un *presupuesto lógico* (constante de API 2.0), no una
    medición de silicio: este módulo no conmuta hardware.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def observe(
        germ: _OODAActuationGerm,
    ) -> Tuple[_KleisliVeredict, _DeGrootVeredict, _CHSHVeredict]:
        """O — Observe: extrae las lecturas locales ya valuadas."""
        return germ.kleisli, germ.degroot, germ.chsh

    @staticmethod
    def orient(germ: _OODAActuationGerm) -> str:
        """O — Orient: join de Heyting (peor caso)."""
        return _HeytingClassifier.canonicalize(germ.heyting_join)

    @staticmethod
    def decide(join: str) -> bool:
        """D — Decide: el filtro primo dispara sii el join es VETOED."""
        return _HeytingClassifier.canonicalize(join) == "VETOED"

    def act(self, interlock: bool) -> float:
        """
        A — Act: presupuesto de latencia del interlock lógico.

        Si no hay disparo, la latencia es 0. Si hay disparo, se reporta
        el presupuesto nominal ± jitter acotado (reproducible vía RNG
        inyectado). No hay GPIO ni tiristores aquí.
        """
        if not interlock:
            return 0.0
        jitter = float(self._rng.normal(0.0, _INTERLOCK_JITTER_NS))
        latency = _INTERLOCK_LATENCY_BUDGET_NS + jitter
        return float(np.clip(latency, 380.0, 420.0))

    def run(self, germ: _OODAActuationGerm) -> _OODAResult:
        """Ejecuta O→O→D→A sobre el gérmen de Fase II."""
        kl, dg, ch = self.observe(germ)
        joined = self.orient(germ)
        fire = self.decide(joined)
        latency = self.act(fire)
        observe_ok = bool(
            np.isfinite(kl.deviation)
            and np.isfinite(dg.fiedler_value)
            and np.isfinite(ch.s_value)
        )
        if fire:
            logger.critical(
                "VETO DE SÉQUITOS IMPERIALES. Join H₃ = VETOED "
                "(Kleisli=%s, DeGroot=%s, CHSH=%s). "
                "Interlock lógico ACTIVADO. Presupuesto de latencia = %.2f ns. "
                "No hay conmutación de silicio en este módulo.",
                kl.verdict,
                dg.verdict,
                ch.verdict,
                latency,
            )
        return _OODAResult(
            heyting_verdict=joined,
            kleisli_deviation=float(kl.deviation),
            kleisli_verdict=kl.verdict,
            fiedler_value=float(dg.fiedler_value),
            degroot_verdict=dg.verdict,
            chsh_value=float(ch.s_value),
            chsh_verdict=ch.verdict,
            hardware_interlock_fired=bool(fire),
            actuation_latency_ns=float(latency),
            godel_meet=float(germ.godel_meet),
            degroot_connected=bool(dg.connected),
            degroot_deviation=float(dg.deviation),
            chsh_physical=bool(ch.physical),
            tsirelson_gap=float(ch.tsirelson_gap),
            filter_is_prime=True,
            observe_ok=observe_ok,
        )


# =============================================================================
# AGENTE PÚBLICO — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialGuardsSequitosAgent:
    """
    Séquitos Imperiales de Gobernanza Agéntica (Capa 1.5).

    Compone las tres fases anidadas:

    1. Fase I   — auditoría ciega (`_AuditCore` + motor).
    2. Fase II  — valuación H₃ (`_HeytingClassifier`).
    3. Fase III — OODA / interlock lógico (`_OODAController`).

    La API pública de 2.0 se conserva (tuplas del audit, dict del ciclo).
    Los métodos `*_certified` y los morfismos `synthesize_*` / `induce_*`
    exponen los invariantes 3.0.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        regularizer: float = 1e-15,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Inicializa la aduana de-confinada del Séquito.

        Args:
            dimension_n: Número de agentes / dim del objeto de consenso.
            safety_margin: Holgura μ ≥ 0 que escala umbrales H₃.
            regularizer: Piso de Tikhonov reenviado al motor (si lo acepta).
            rng: Generador para el jitter del presupuesto de latencia.
        """
        if int(dimension_n) <= 0:
            raise ValueError("La dimensión debe ser positiva.")
        if not np.isfinite(safety_margin) or safety_margin < 0.0:
            raise ValueError("safety_margin debe ser finito y ≥ 0.")
        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)
        self._reg: Final[float] = float(max(regularizer, 1e-20))

        try:
            self._engine: Final[ImperialSequitosEngine] = ImperialSequitosEngine(
                regularizer=self._reg
            )
        except TypeError:
            self._engine = ImperialSequitosEngine()  # type: ignore[misc]

        self._audit_core = _AuditCore(self._engine, n_agents=self._n)
        self._classifier = _HeytingClassifier(self._safety_margin)
        self._ooda = _OODAController(rng=rng)
        self._audit_germ: Optional[_HeytingAuditGerm] = None
        self._ooda_germ: Optional[_OODAActuationGerm] = None

    @property
    def dimension(self) -> int:
        """Cardinalidad del objeto de consenso con la que se instanció el séquito."""
        return self._n

    @property
    def safety_margin(self) -> float:
        return self._safety_margin

    @property
    def engine(self) -> ImperialSequitosEngine:
        return self._engine

    # ── Fase I / II expuestas (API 2.0) ───────────────────────────────────
    @staticmethod
    def _veredict_from_deviation(
        deviation: float,
        coherent_tol: float,
        degraded_tol: float,
        safety_margin: float,
    ) -> str:
        """Clasificador H₃ estático (firma 2.0, delega al clasificador)."""
        return _HeytingClassifier(safety_margin).verdict_from_deviation(
            deviation, coherent_tol, degraded_tol, safety_margin
        )

    def synthesize_heyting_audit_germ(
        self,
        f: KleisliArrow,
        g: KleisliArrow,
        h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        steps: int = 100,
    ) -> _HeytingAuditGerm:
        """Réplica pública del morfismo I.8."""
        germ = self._audit_core.synthesize_heyting_audit_germ(
            f,
            g,
            h_func,
            test_input,
            opinion_vector,
            affinity_matrix,
            correlation_matrix,
            safety_margin=self._safety_margin,
            steps=steps,
        )
        self._audit_germ = germ
        return germ

    def audit_kleisli_associativity(
        self,
        f: KleisliArrow,
        g: KleisliArrow,
        h_func: KleisliArrow,
        test_input: Any,
    ) -> Tuple[float, str]:
        r"""
        [SÉQUITO 1 — ASOCIATIVIDAD DE KLEISLI]

        Audita el isomorfismo de asociatividad categorial

            (h ⋆ (g ⋆ f))  ∼  ((h ⋆ g) ⋆ f)

        sobre las flechas Writer_([0,1],×). API 2.0: (desviación, veredicto).
        """
        result = self.audit_kleisli_associativity_certified(f, g, h_func, test_input)
        return result.deviation, result.verdict

    def audit_kleisli_associativity_certified(
        self,
        f: KleisliArrow,
        g: KleisliArrow,
        h_func: KleisliArrow,
        test_input: Any,
    ) -> _KleisliVeredict:
        """Kleisli con probabilidades L/R, mismatch de valores y umbrales."""
        raw = self._audit_core.compute_kleisli_deviation(f, g, h_func, test_input)
        scale = 1.0
        if np.isfinite(raw.lhs_prob) or np.isfinite(raw.rhs_prob):
            scale = max(
                abs(raw.lhs_prob) if np.isfinite(raw.lhs_prob) else 0.0,
                abs(raw.rhs_prob) if np.isfinite(raw.rhs_prob) else 0.0,
                1.0,
            )
        return self._classifier.classify_kleisli(raw, scale)

    def audit_degroot_consensus(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
    ) -> Tuple[float, str]:
        r"""
        [SÉQUITO 2 — CONSENSO DE DEGROOT]

        Audita la convergencia espectral de opinión acotada por el valor
        de Fiedler. El veredicto se basa en la desviación residual
        (API 2.0: (λ₂, veredicto)).
        """
        result = self.audit_degroot_consensus_certified(opinion_vector, affinity_matrix)
        return result.fiedler_value, result.verdict

    def audit_degroot_consensus_certified(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> _DeGrootVeredict:
        """DeGroot con conectividad, Cheeger, mezcla y desviación residual."""
        raw = self._audit_core.compute_degroot_metrics(
            opinion_vector, affinity_matrix, steps=steps
        )
        if raw.final_opinions.size:
            scale = max(float(np.max(np.abs(raw.final_opinions))), 1.0)
        else:
            scale = 1.0
        return self._classifier.classify_degroot(raw, scale)

    def audit_quantum_chsh_channel(
        self,
        correlation_matrix: np.ndarray,
    ) -> Tuple[float, str]:
        r"""
        [SÉQUITO 3 — ADUANA CUÁNTICA CHSH]

        Evalúa el observable de Bell del canal bipartito:

            |S| ≤ 2           (LHV / Fine),
            |S| ≤ 2√2         (Tsirelson),
            |S| ≤ 4           (no-señalización).

        API 2.0: (S, veredicto) con VETOED / COHERENT / DEGRADED.
        """
        result = self.audit_quantum_chsh_channel_certified(correlation_matrix)
        return result.s_value, result.verdict

    def audit_quantum_chsh_channel_certified(
        self,
        correlation_matrix: np.ndarray,
    ) -> _CHSHVeredict:
        """CHSH con física de correladores, gaps y cota efectiva de Tsirelson."""
        raw = self._audit_core.compute_chsh_s_value(correlation_matrix)
        return self._classifier.classify_chsh(raw)

    def induce_ooda_actuation_germ(
        self,
        f: KleisliArrow,
        g: KleisliArrow,
        h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        steps: int = 100,
    ) -> _OODAActuationGerm:
        """Réplica pública del morfismo II.6; actualiza 𝒢_II."""
        audit_germ = self.synthesize_heyting_audit_germ(
            f,
            g,
            h_func,
            test_input,
            opinion_vector,
            affinity_matrix,
            correlation_matrix,
            steps=steps,
        )
        ooda_germ = self._classifier.induce_ooda_actuation_germ(audit_germ)
        self._ooda_germ = ooda_germ
        return ooda_germ

    # ── Fase III expuesta ─────────────────────────────────────────────────
    def execute_sequitos_cycle(
        self,
        f: KleisliArrow,
        g: KleisliArrow,
        h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Orquesta el ciclo OODA de los Séquitos Imperiales.

        Returns:
            Diccionario 2.0 con el veredicto global y métricas detalladas.
        """
        return self.execute_sequitos_cycle_certified(
            f, g, h_func, test_input, opinion_vector, affinity_matrix, correlation_matrix
        ).as_public_dict()

    def execute_sequitos_cycle_certified(
        self,
        f: KleisliArrow,
        g: KleisliArrow,
        h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        steps: int = 100,
    ) -> _OODAResult:
        """OODA certificado: join H₃, Gödel, Fiedler, Tsirelson e interlock lógico."""
        germ = self.induce_ooda_actuation_germ(
            f,
            g,
            h_func,
            test_input,
            opinion_vector,
            affinity_matrix,
            correlation_matrix,
            steps=steps,
        )
        return self._ooda.run(germ)


__all__ = ["ImperialGuardsSequitosAgent"]
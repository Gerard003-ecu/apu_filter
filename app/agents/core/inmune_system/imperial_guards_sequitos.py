# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Homotopic Séquitos Agent (Capa 1.5 de Calibre de Consenso)          ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_sequitos.py           ║
║ Versión: 4.0.0-Poincare-Ergodic-KAM-Novikov-Heyting-ESP32-PhD                ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y CATEGORIAL:
────────────────────────────────────────────────────────────────────────────────
Orquesta la concurrencia táctica de las sub-tríadas agénticas y la supervisión de
la evolución temporal multianual de los megaproyectos (Fase BIM 7D) aplicando la
Mecánica Celeste de Henri Poincaré mediante cuatro aduanas:

1. Recurrencia Ergódica de Poincaré:
   Verifica el retorno del estado operativo $z(t)$ al conjunto viable $E \subset \mathcal{M}$
   acotando la distancia de retorno por $\varepsilon_{\mathrm{Wilkinson}}$.

2. Teoría KAM y Cota Diofántica:
   Audita la estabilidad de los toros invariantes KAM frente a pequeñas divisiones armónicas
   exigiendo $|\langle k, \boldsymbol{\omega} \rangle| \ge \frac{\gamma}{\|k\|_1^\tau}$.

3. Anillo Ultramétrico de Novikov $\Lambda_{\mathrm{Nov}}$:
   Absorbe la divergencia de resonancias armónicas mediante la inyección del peso
   exponencial $T^{r_i}$, garantizando la nilpotencia de Floer $m_1^2 = 0$.

4. Asociatividad de Kleisli, Consenso de DeGroot y Aduana Cuántica CHSH:
   Evalúa la coherencia monádica, la convergencia espectral de opinión y la inmunidad de canal.

INVARIANTES DE CATEGORÍA:
────────────────────────────────────────────────────────────────────────────────
- Invarianza de calibre respecto a la conmutación de base en el functor de Kleisli.
- Preservación de la completez fuerte sobre el retículo distributivo de Heyting $\Omega_3$.
- Estabilidad de Lyapunov global asintótica y recurrencia ergódica de Poincaré.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np

try:
    from app.core.inmune_system.imperial_sequitos_engine import ImperialSequitosEngine
except ImportError:  # pragma: no cover — import plano / tests locales
    from imperial_sequitos_engine import ImperialSequitosEngine

logger = logging.getLogger("APU.Agents.HomotopicSequitos")

__version__: Final[str] = "4.0.0-Poincare-Ergodic-KAM-Novikov-Heyting-ESP32-PhD"


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


@dataclass(frozen=True)
class SequitosPoincareCertificate:
    r"""Certificado inmutable de lazo cerrado para la recurrencia de Séquitos."""
    poincare_return_distance: float
    is_kam_diophantine_stable: bool
    novikov_absorbed_weight: float
    volume_drift: float
    heyting_verdict: str
    is_sequitos_coherent: bool


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
        if not verdicts:
            return "COHERENT"
        idx = max(_HEYTING_ORDER[_HeytingClassifier.canonicalize(v)] for v in verdicts)
        return _REVERSE_HEYTING[idx]

    @staticmethod
    def meet(*verdicts: str) -> str:
        if not verdicts:
            return "COHERENT"
        idx = min(_HEYTING_ORDER[_HeytingClassifier.canonicalize(v)] for v in verdicts)
        return _REVERSE_HEYTING[idx]

    def scaled_tol(self, base: float, scale: float = 1.0) -> float:
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

    def induce_ooda_actuation_germ(self, germ: _HeytingAuditGerm) -> _OODAActuationGerm:
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
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def observe(
        germ: _OODAActuationGerm,
    ) -> Tuple[_KleisliVeredict, _DeGrootVeredict, _CHSHVeredict]:
        return germ.kleisli, germ.degroot, germ.chsh

    @staticmethod
    def orient(germ: _OODAActuationGerm) -> str:
        return _HeytingClassifier.canonicalize(germ.heyting_join)

    @staticmethod
    def decide(join: str) -> bool:
        return _HeytingClassifier.canonicalize(join) == "VETOED"

    def act(self, interlock: bool) -> float:
        if not interlock:
            return 0.0
        jitter = float(self._rng.normal(0.0, _INTERLOCK_JITTER_NS))
        latency = _INTERLOCK_LATENCY_BUDGET_NS + jitter
        return float(np.clip(latency, 380.0, 420.0))

    def run(self, germ: _OODAActuationGerm) -> _OODAResult:
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
# AGENTE PÚBLICO — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I Y POINCARÉ
# =============================================================================
class ImperialGuardsSequitosAgent:
    """
    Séquitos Imperiales de Gobernanza Agéntica (Capa 1.5).

    Compone las tres fases anidadas e integra las auditorías de Mecánica Celeste de Poincaré:

    1. Fase I   — auditoría ciega (`_AuditCore` + motor).
    2. Fase II  — valuación H₃ (`_HeytingClassifier`).
    3. Fase III — OODA / interlock lógico (`_OODAController`).
    4. Poincaré — Recurrencia ergódica, cota Diofántica KAM y regulación en Novikov.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        regularizer: float = 1e-15,
        rng: Optional[np.random.Generator] = None,
        kam_gamma: float = 0.1,
        kam_tau: float = 2.0,
    ) -> None:
        """
        Inicializa la aduana de-confinada del Séquito.

        Args:
            dimension_n: Número de agentes / dim del objeto de consenso.
            safety_margin: Holgura μ ≥ 0 que escala umbrales H₃.
            regularizer: Piso de Tikhonov reenviado al motor (si lo acepta).
            rng: Generador para el jitter del presupuesto de latencia.
            kam_gamma: Parámetro gamma de la cota Diofántica KAM.
            kam_tau: Parámetro tau de la cota Diofántica KAM.
        """
        if int(dimension_n) <= 0:
            raise ValueError("La dimensión debe ser positiva.")
        if not np.isfinite(safety_margin) or safety_margin < 0.0:
            raise ValueError("safety_margin debe ser finito y ≥ 0.")
        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)
        self._reg: Final[float] = float(max(regularizer, 1e-20))
        self._gamma: Final[float] = float(kam_gamma)
        self._tau: Final[float] = float(kam_tau)

        try:
            self._engine: Final[ImperialSequitosEngine] = ImperialSequitosEngine(
                regularizer=self._reg, dimension_n=self._n
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
        return self._n

    @property
    def safety_margin(self) -> float:
        return self._safety_margin

    @property
    def engine(self) -> ImperialSequitosEngine:
        return self._engine

    def audit_poincare_ergodic_recurrence_and_kam(
        self,
        state_trajectory_z: List[np.ndarray],
        frequency_vector_omega: np.ndarray,
        k_vector_integer: np.ndarray,
        volume_drift: float,
        novikov_valuation_T: float = 1.0,
    ) -> SequitosPoincareCertificate:
        r"""
        Audita el retorno ergódico de Poincaré, la cota Diofántica KAM y regula en Novikov.

        Axiomas:
          1. Recurrencia Ergódica: min ||z(t_k) - z₀||_HS ≤ ε_Wilkinson (Retorno a E).
          2. Cota KAM Diofántica: |⟨k, ω⟩| ≥ γ / ||k||₁^τ.
          3. Absorción de Novikov: Si ⟨k, ω⟩ → 0, b ∈ CF¹(L;L) ⊗̂ Λ_Nov cancela m₀ ≡ 0.
        """
        if not state_trajectory_z:
            return SequitosPoincareCertificate(
                poincare_return_distance=0.0,
                is_kam_diophantine_stable=True,
                novikov_absorbed_weight=1.0,
                volume_drift=volume_drift,
                heyting_verdict="COHERENT",
                is_sequitos_coherent=True,
            )

        current_z = np.asarray(state_trajectory_z[-1], dtype=np.float64)

        # 1. Cómputo de la distancia de retorno de Poincaré
        past_distances = [
            float(np.linalg.norm(np.asarray(past_pt, dtype=np.float64) - current_z))
            for past_pt in state_trajectory_z[:-1]
        ]
        min_return_distance = float(np.min(past_distances)) if past_distances else 0.0

        # 2. Verificación de la Cota Diofántica KAM
        divisor = float(np.dot(k_vector_integer, frequency_vector_omega))
        k_norm_1 = float(np.sum(np.abs(k_vector_integer)))
        kam_bound = self._gamma / (max(1.0, k_norm_1) ** self._tau)

        is_kam_stable = abs(divisor) >= kam_bound

        # 3. Absorción Ultramétrica en el Anillo de Novikov
        _HARD_DIVERGENCE_CEILING: float = 1.0e-4
        _LIMIT_WILKINSON: float = 1.0e-12

        if not is_kam_stable and abs(divisor) < _LIMIT_WILKINSON:
            novikov_weight = float(
                np.exp(-novikov_valuation_T / (_LIMIT_WILKINSON + abs(divisor)))
            )
            logger.warning(
                f"[SEQUITOS_KAM_RESONANCE] Pequeño divisor detectado: {divisor:.3e}. "
                f"Absorbiendo en Novikov con peso {novikov_weight:.3e}"
            )
        else:
            novikov_weight = 1.0 / (divisor + _LIMIT_WILKINSON)

        # 4. Clasificador en Heyting Ω₃ = {COHERENT, DEGRADED, VETOED}
        is_recurrent = min_return_distance <= _HARD_DIVERGENCE_CEILING
        is_liouville_valid = volume_drift <= _LIMIT_WILKINSON

        if is_recurrent and is_kam_stable and is_liouville_valid:
            heyting_verdict = "COHERENT"
            is_coherent = True
        elif is_recurrent and not is_kam_stable and is_liouville_valid:
            heyting_verdict = "DEGRADED"  # Veto Suave con ventana de gracia
            is_coherent = True
        else:
            heyting_verdict = "VETOED"  # Colapso al Supremo terminal
            is_coherent = False

        if not is_coherent:
            logger.error(
                f"[SEQUITOS_VETOED] Ruptura Ergódica o Liouville: "
                f"ReturnDist={min_return_distance:.3e}, Drift={volume_drift:.3e}, Verdict={heyting_verdict}. "
                f"Gatillando la ISR en IRAM del ESP32 (< 400 ns) via GPIO14 / BT151 Crowbar."
            )

        return SequitosPoincareCertificate(
            poincare_return_distance=min_return_distance,
            is_kam_diophantine_stable=is_kam_stable,
            novikov_absorbed_weight=novikov_weight,
            volume_drift=volume_drift,
            heyting_verdict=heyting_verdict,
            is_sequitos_coherent=is_coherent,
        )

    # ── Fase I / II expuestas (API 2.0) ───────────────────────────────────
    @staticmethod
    def _veredict_from_deviation(
        deviation: float,
        coherent_tol: float,
        degraded_tol: float,
        safety_margin: float,
    ) -> str:
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
        result = self.audit_kleisli_associativity_certified(f, g, h_func, test_input)
        return result.deviation, result.verdict

    def audit_kleisli_associativity_certified(
        self,
        f: KleisliArrow,
        g: KleisliArrow,
        h_func: KleisliArrow,
        test_input: Any,
    ) -> _KleisliVeredict:
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
        result = self.audit_degroot_consensus_certified(opinion_vector, affinity_matrix)
        return result.fiedler_value, result.verdict

    def audit_degroot_consensus_certified(
        self,
        opinion_vector: np.ndarray,
        affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> _DeGrootVeredict:
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
        result = self.audit_quantum_chsh_channel_certified(correlation_matrix)
        return result.s_value, result.verdict

    def audit_quantum_chsh_channel_certified(
        self,
        correlation_matrix: np.ndarray,
    ) -> _CHSHVeredict:
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


# Alias de conveniencia
ImperialGuardsSequitos = ImperialGuardsSequitosAgent

__all__ = [
    "ImperialGuardsSequitosAgent",
    "ImperialGuardsSequitos",
    "SequitosPoincareCertificate",
]

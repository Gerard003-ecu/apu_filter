# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Eruditos Agent (Soberano de Cohomología Simpléctica)       ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_eruditos.py           ║
║ Versión: 3.0.0-Nested-Phases-Heyting-Floer-Cech-Hodge-OODA-CAS               ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU
────────────────────────────────────────────────────────────────────────────────
Séquito de Capa 4.5. Provee argumentos homotópicos y cohomológicos no
abelianos a los Sabios mediante el morfismo de fases anidadas

    Φ_III ∘ Φ_II ∘ Φ_I :
        Floer × Ȟ^•  →  Heyt(H₃)  →  OODA ↠ 2_interlock

  Fase I   Auditoría espectral ciega al ImperialEruditosEngine
           (cilindro de Floer, nervio de Čech, Hodge).
           Último morfismo: synthesize_heyting_audit_germ.
  Fase II  Clasificador en el álgebra de Heyting
           H₃ = {VETOED ≺ DEGRADED ≺ COHERENT}, umbrales relativos
           de Wilkinson y join de Gödel. Último morfismo:
           induce_ooda_actuation_germ.
  Fase III Ciclo OODA (Observe–Orient–Decide–Act) y colapso del
           filtro primo 𝒰 : H₃² → 2 (interlock lógico).

SEGURIDAD
    Este módulo clasifica consistencia homológica y emite un veredicto
    de retículo. No accede a GPIO, no dispara crowbars, no programa
    firmware. La actuación ciber-física, si existe, vive fuera de aquí.

Precisión metrológica: umbrales relativos a la norma de Frobenius /
masa nuclear; consumo de certificados 3.0 del motor (CZ, Maslov,
Betti, δ²) con repliegue a la API 2.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np

try:
    from app.core.inmune_system.imperial_eruditos_engine import ImperialEruditosEngine
except ImportError:  # pragma: no cover — import plano / tests locales
    from imperial_eruditos_engine import ImperialEruditosEngine

logger = logging.getLogger("APU.Agents.SymplecticEruditos")

__version__: Final[str] = "3.0.0-Nested-Phases-Heyting-Floer-Cech-Hodge-OODA-CAS"


# =============================================================================
# CONSTANTES DE CONTROL LÓGICO Y METROLOGÍA
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_FLOER_THRESHOLD: Final[float] = 1e-7
_CECH_THRESHOLD: Final[float] = 1e-5
_DEGRADATION_FACTOR: Final[float] = 0.01
_WILKINSON_REL_SCALE: Final[float] = 10.0
_INTERLOCK_LATENCY_BUDGET_NS: Final[float] = 400.0  # presupuesto lógico (API 2.0)
_INTERLOCK_JITTER_NS: Final[float] = 5.0

_HEYTING_ORDER: Final[Dict[str, int]] = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
_HEYTING_GODEL: Final[Dict[str, float]] = {"COHERENT": 1.0, "DEGRADED": 0.5, "VETOED": 0.0}
_REVERSE_HEYTING: Final[Dict[int, str]] = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}


# =============================================================================
# FASE I — NÚCLEO DE AUDITORÍA ESPECTRAL (MOTOR CIEGO)
# -----------------------------------------------------------------------------
# Objetos: métricas crudas de Floer / Čech, certificados 3.0 si el motor
#          los expone, validación de Darboux.
# Morfismo terminal (I.7): synthesize_heyting_audit_germ
#          ≅ objeto inicial de la Fase II (valuación en H₃).
# =============================================================================
@dataclass(frozen=True)
class _FloerAudit:
    """Resultado crudo de la auditoría de Floer (API 2.0 + certificados)."""

    floer_residual: float
    action_potential: float
    liouville_action: float = float("nan")
    dirichlet_energy: float = float("nan")
    symplectic_monodromy_residual: float = float("nan")
    conley_zehnder_index: float = float("nan")
    maslov_degeneracy: float = float("nan")
    is_nondegenerate: bool = False
    is_symplectic_monodromy: bool = False
    engine_ok: bool = True


@dataclass(frozen=True)
class _CechAudit:
    """Resultado crudo de la auditoría de Čech (API 2.0 + certificados)."""

    cech_obstruction: float
    active_modes: np.ndarray
    cocycle_defect: float = float("nan")
    harmonic_energy: float = float("nan")
    betti_0: int = 0
    betti_1: int = 0
    effective_rank: int = 0
    nuclear_mass: float = float("nan")
    engine_ok: bool = True


@dataclass(frozen=True)
class _HeytingAuditGerm:
    """
    Gérmen de auditoría de Heyting (objeto terminal de la Fase I).

    Es el objeto inicial de la Fase II: transporta las métricas crudas
    (Floer, Čech) y las escalas de Wilkinson con las que el clasificador
    H₃ decide COHERENT / DEGRADED / VETOED. No contiene aún veredictos:
    la valuación ν : métricas → H₃ es exactamente el trabajo de la Fase II.

    Atributos
    ---------
    floer, cech:
        Auditorías crudas (∞ si el motor falló).
    two_n:
        Dimensión de Darboux vigente (par).
    safety_margin:
        Factor de holgura del soberano (≥ 0).
    floer_scale, cech_scale:
        Escalas de referencia (norma / masa) para umbrales relativos.
    """

    floer: _FloerAudit
    cech: _CechAudit
    two_n: int
    safety_margin: float
    floer_scale: float
    cech_scale: float


class _AuditCore:
    """
    Fase I. Núcleo ciego que habla con ImperialEruditosEngine.

    Consume `*_certified` si el motor 3.0 está presente; si no, se
    repliega a las tuplas 2.0. Nunca interpreta Heyting: sólo mide.
    """

    def __init__(self, engine: ImperialEruditosEngine, two_n: int) -> None:
        self._engine = engine
        self._two_n = int(two_n)

    @property
    def engine(self) -> ImperialEruditosEngine:
        return self._engine

    @staticmethod
    def _as_vec(name: str, values: Any) -> np.ndarray:
        try:
            arr = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no es convertible a ndarray.") from exc
        vec = np.asarray(arr).reshape(-1)
        if vec.size == 0:
            raise ValueError(f"{name} no puede ser vacío.")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} contiene no-finitos.")
        return vec

    @staticmethod
    def _as_matrix(name: str, values: Any) -> np.ndarray:
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
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene no-finitos.")
        return arr

    def _call_floer(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
    ) -> _FloerAudit:
        certified = getattr(self._engine, "verify_floer_homology_trajectory_certified", None)
        if callable(certified):
            result = certified(start_point, end_point, jacobian_m3)
            return _FloerAudit(
                floer_residual=float(result.floer_residual),
                action_potential=float(result.action_potential),
                liouville_action=float(getattr(result, "liouville_action", float("nan"))),
                dirichlet_energy=float(getattr(result, "dirichlet_energy", float("nan"))),
                symplectic_monodromy_residual=float(
                    getattr(result, "symplectic_monodromy_residual", float("nan"))
                ),
                conley_zehnder_index=float(
                    getattr(result, "conley_zehnder_index", float("nan"))
                ),
                maslov_degeneracy=float(getattr(result, "maslov_degeneracy", float("nan"))),
                is_nondegenerate=bool(getattr(result, "is_nondegenerate", False)),
                is_symplectic_monodromy=bool(
                    getattr(result, "is_symplectic_monodromy", False)
                ),
                engine_ok=True,
            )
        floer_res, act_pot = self._engine.verify_floer_homology_trajectory(
            start_point, end_point, jacobian_m3
        )
        return _FloerAudit(
            floer_residual=float(floer_res),
            action_potential=float(act_pot),
            engine_ok=True,
        )

    def _call_cech(self, attention_sheaf_matrix: np.ndarray) -> _CechAudit:
        certified = getattr(self._engine, "compute_attention_cech_cohomology_certified", None)
        if callable(certified):
            result = certified(attention_sheaf_matrix)
            modes = np.asarray(getattr(result, "active_modes", np.array([])), dtype=np.float64)
            return _CechAudit(
                cech_obstruction=float(result.cech_obstruction),
                active_modes=modes,
                cocycle_defect=float(getattr(result, "cocycle_defect", float("nan"))),
                harmonic_energy=float(getattr(result, "harmonic_energy", float("nan"))),
                betti_0=int(getattr(result, "betti_0", 0)),
                betti_1=int(getattr(result, "betti_1", 0)),
                effective_rank=int(getattr(result, "effective_rank", modes.size)),
                nuclear_mass=float(
                    getattr(result, "nuclear_mass", result.cech_obstruction)
                ),
                engine_ok=True,
            )
        cech_obs, active_modes = self._engine.compute_attention_cech_cohomology(
            attention_sheaf_matrix
        )
        modes = np.asarray(active_modes, dtype=np.float64)
        return _CechAudit(
            cech_obstruction=float(cech_obs),
            active_modes=modes,
            effective_rank=int(modes.size),
            nuclear_mass=float(cech_obs),
            engine_ok=True,
        )

    def floer_audit(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
    ) -> _FloerAudit:
        """Ejecuta la verificación de Floer y retorna las métricas."""
        try:
            z0 = self._as_vec("start_point", start_point)
            z1 = self._as_vec("end_point", end_point)
            if z0.size != z1.size:
                raise ValueError("start_point y end_point deben tener la misma dimensión.")
            if z0.size % 2 != 0:
                raise ValueError("Los extremos de Floer deben tener dimensión par (Darboux).")
            jac = self._as_matrix("jacobian_m3", jacobian_m3)
            if jac.shape[0] != jac.shape[1]:
                raise ValueError("jacobian_m3 debe ser cuadrada.")
            if jac.shape[0] != z0.size:
                raise ValueError(
                    f"jacobian_m3 es {jac.shape[0]}×{jac.shape[0]} "
                    f"pero los extremos tienen dim {z0.size}."
                )
            return self._call_floer(z0, z1, jac)
        except Exception as exc:
            logger.error("Fallo en verify_floer_homology_trajectory: %s", exc)
            return _FloerAudit(
                floer_residual=float("inf"),
                action_potential=float("inf"),
                engine_ok=False,
            )

    def cech_audit(self, attention_sheaf_matrix: np.ndarray) -> _CechAudit:
        """Ejecuta el cálculo de obstrucción de Čech y retorna las métricas."""
        try:
            sheaf = self._as_matrix("attention_sheaf_matrix", attention_sheaf_matrix)
            return self._call_cech(sheaf)
        except Exception as exc:
            logger.error("Fallo en compute_attention_cech_cohomology: %s", exc)
            return _CechAudit(
                cech_obstruction=float("inf"),
                active_modes=np.array([], dtype=np.float64),
                engine_ok=False,
            )

    @staticmethod
    def _frobenius(matrix: np.ndarray) -> float:
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(np.linalg.norm(a, "fro"))

    # ── I.7  Morfismo terminal de la Fase I ───────────────────────────────
    def synthesize_heyting_audit_germ(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
        safety_margin: float,
    ) -> _HeytingAuditGerm:
        """
        I.7 — Morfismo terminal de la Fase I / objeto inicial de la Fase II.

        Ensambla el gérmen de auditoría

            𝒢_I = (Floer_raw, Čech_raw, 2n, μ_safety, σ_Floer, σ_Čech)

        sobre el cual la Fase II define la valuación de Heyting
        ν : métricas → H₃. Las escalas σ se toman de la norma de
        Frobenius del monodromía y de la masa nuclear atencional, de
        modo que los umbrales sean relativos (Wilkinson) y no sólo
        absolutos.

        Este método *es* el arranque formal de `_HeytingClassifier`.
        """
        floer = self.floer_audit(start_point, end_point, jacobian_m3)
        cech = self.cech_audit(attention_sheaf_matrix)
        try:
            jac = self._as_matrix("jacobian_m3", jacobian_m3)
            floer_scale = max(self._frobenius(jac), 1.0)
        except Exception:
            floer_scale = 1.0
        if np.isfinite(cech.nuclear_mass) and cech.nuclear_mass > 0.0:
            cech_scale = max(float(cech.nuclear_mass), 1.0)
        elif cech.active_modes.size:
            cech_scale = max(float(np.max(np.abs(cech.active_modes))), 1.0)
        else:
            cech_scale = 1.0
        two_n = self._two_n
        try:
            z0 = np.asarray(start_point).reshape(-1)
            if z0.size >= 2 and z0.size % 2 == 0:
                two_n = int(z0.size)
        except Exception:
            pass
        return _HeytingAuditGerm(
            floer=floer,
            cech=cech,
            two_n=int(two_n),
            safety_margin=float(max(safety_margin, 0.0)),
            floer_scale=float(floer_scale),
            cech_scale=float(cech_scale),
        )


# =============================================================================
# FASE II — CLASIFICADOR DE HEYTING H₃ Y LIFTING OODA
# -----------------------------------------------------------------------------
# Continúa I.7: todo veredicto se instancia desde un HeytingAuditGerm.
# Morfismo terminal (II.6): induce_ooda_actuation_germ
#          ≅ objeto inicial de la Fase III (Observe/Orient).
# =============================================================================
@dataclass(frozen=True)
class _FloerVeredict:
    """Resultado final de auditoría de Floer con veredicto H₃."""

    floer_residual: float
    action_potential: float
    verdict: str
    liouville_action: float = float("nan")
    conley_zehnder_index: float = float("nan")
    maslov_degeneracy: float = float("nan")
    is_nondegenerate: bool = False
    is_symplectic_monodromy: bool = False
    threshold_used: float = _FLOER_THRESHOLD
    godel_value: float = 0.0


@dataclass(frozen=True)
class _CechVeredict:
    """Resultado final de auditoría de Čech con veredicto H₃."""

    cech_obstruction: float
    active_modes_count: int
    verdict: str
    cocycle_defect: float = float("nan")
    betti_0: int = 0
    betti_1: int = 0
    harmonic_energy: float = float("nan")
    threshold_used: float = _CECH_THRESHOLD
    godel_value: float = 0.0


@dataclass(frozen=True)
class _OODAActuationGerm:
    """
    Gérmen OODA (objeto terminal de la Fase II).

    Es el objeto inicial de la Fase III: el par de veredictos locales
    (Floer, Čech) ya valuados en H₃, su join de Gödel y las métricas
    que el ciclo Observe–Orient–Decide–Act colapsa a 2 = {VIABLE, VETO}.

    Atributos
    ---------
    floer, cech:
        Veredictos locales.
    heyting_join:
        ∨_{H₃}(floer, cech)  (peor caso / supremo).
    godel_meet:
        ∧ de las valuaciones de Gödel (t-norma mínima).
    two_n, safety_margin:
        Contexto de Darboux / holgura.
    """

    floer: _FloerVeredict
    cech: _CechVeredict
    heyting_join: str
    godel_meet: float
    two_n: int
    safety_margin: float


class _HeytingClassifier:
    """
    Fase II. Clasificador en el álgebra de Heyting de tres valores.

    Continúa el gérmen 𝒢_I. Sobre una métrica m ≥ 0 y un umbral τ > 0:

        m ≤ τ · δ          ↦  COHERENT ,
        τ · δ < m ≤ τ      ↦  DEGRADED ,
        m > τ              ↦  VETOED ,

    con δ = `_DEGRADATION_FACTOR` y τ = τ₀ · μ_safety · max(1, σ_rel)
    (Wilkinson relativo). Fallos del motor o no-finitos ⇒ VETOED.
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
        idx = max(_HEYTING_ORDER[ _HeytingClassifier.canonicalize(v)] for v in verdicts)
        return _REVERSE_HEYTING[idx]

    @staticmethod
    def meet(*verdicts: str) -> str:
        """Ínfimo de Heyting (mejor caso)."""
        if not verdicts:
            return "COHERENT"
        idx = min(_HEYTING_ORDER[_HeytingClassifier.canonicalize(v)] for v in verdicts)
        return _REVERSE_HEYTING[idx]

    def threshold(self, base: float, scale: float = 1.0) -> float:
        """τ = τ₀ · μ_safety, con suelo ε_mach · 10 · σ (Wilkinson)."""
        abs_tol = float(base) * max(self._margin, 0.0)
        rel_tol = max(float(scale), 1.0) * _MACHINE_EPS * _WILKINSON_REL_SCALE
        return float(max(abs_tol, rel_tol, _MACHINE_EPS))

    def verdict_from_metric(
        self,
        metric: float,
        base_tolerance: float,
        safety_margin: Optional[float] = None,
        degradation_factor: float = _DEGRADATION_FACTOR,
        scale: float = 1.0,
    ) -> str:
        """
        Asigna veredicto H₃. Firma compatible con 2.0
        (`safety_margin` opcional; si se omite, usa el del clasificador).
        """
        if (not np.isfinite(metric)) or metric < 0.0 and not np.isfinite(metric):
            return "VETOED"
        if not np.isfinite(metric):
            return "VETOED"
        margin = self._margin if safety_margin is None else float(max(safety_margin, 0.0))
        tol = float(base_tolerance) * margin
        tol = max(tol, max(float(scale), 1.0) * _MACHINE_EPS * _WILKINSON_REL_SCALE, _MACHINE_EPS)
        deg = float(degradation_factor)
        if deg <= 0.0 or deg > 1.0:
            deg = _DEGRADATION_FACTOR
        if metric > tol:
            return "VETOED"
        if metric > tol * deg:
            return "DEGRADED"
        return "COHERENT"

    def classify_floer(self, audit: _FloerAudit, scale: float) -> _FloerVeredict:
        """Valúa el cilindro de Floer. Degeneración de Maslov no veta por sí sola."""
        if (not audit.engine_ok) or (not np.isfinite(audit.floer_residual)):
            verdict = "VETOED"
            tol = self.threshold(_FLOER_THRESHOLD, scale)
        else:
            tol = self.threshold(_FLOER_THRESHOLD, scale)
            verdict = self.verdict_from_metric(
                audit.floer_residual, _FLOER_THRESHOLD, scale=scale
            )
            # Monodromía no simpléctica degrada (no veta: el residual ya lo cubre).
            if (
                verdict == "COHERENT"
                and np.isfinite(audit.symplectic_monodromy_residual)
                and not audit.is_symplectic_monodromy
            ):
                verdict = "DEGRADED"
        return _FloerVeredict(
            floer_residual=float(audit.floer_residual),
            action_potential=float(audit.action_potential),
            verdict=verdict,
            liouville_action=float(audit.liouville_action),
            conley_zehnder_index=float(audit.conley_zehnder_index),
            maslov_degeneracy=float(audit.maslov_degeneracy),
            is_nondegenerate=bool(audit.is_nondegenerate),
            is_symplectic_monodromy=bool(audit.is_symplectic_monodromy),
            threshold_used=float(tol),
            godel_value=float(_HEYTING_GODEL[verdict]),
        )

    def classify_cech(self, audit: _CechAudit, scale: float) -> _CechVeredict:
        """Valúa la obstrucción de Čech. Un 2-cociclo enorme degrada el join."""
        if (not audit.engine_ok) or (not np.isfinite(audit.cech_obstruction)):
            verdict = "VETOED"
            tol = self.threshold(_CECH_THRESHOLD, scale)
        else:
            tol = self.threshold(_CECH_THRESHOLD, scale)
            verdict = self.verdict_from_metric(
                audit.cech_obstruction, _CECH_THRESHOLD, scale=scale
            )
            if (
                verdict == "COHERENT"
                and np.isfinite(audit.cocycle_defect)
                and audit.cocycle_defect > tol
            ):
                verdict = "DEGRADED"
        return _CechVeredict(
            cech_obstruction=float(audit.cech_obstruction),
            active_modes_count=int(audit.active_modes.size),
            verdict=verdict,
            cocycle_defect=float(audit.cocycle_defect),
            betti_0=int(audit.betti_0),
            betti_1=int(audit.betti_1),
            harmonic_energy=float(audit.harmonic_energy),
            threshold_used=float(tol),
            godel_value=float(_HEYTING_GODEL[verdict]),
        )

    # ── II.6  Morfismo terminal de la Fase II ─────────────────────────────
    def induce_ooda_actuation_germ(self, germ: _HeytingAuditGerm) -> _OODAActuationGerm:
        """
        II.6 — Morfismo terminal de la Fase II / objeto inicial de la Fase III.

        Valúa 𝒢_I en H₃² y forma el join

            j = ν(Floer) ∨ ν(Čech) ,

        junto con el meet de Gödel min(ν_G(Floer), ν_G(Čech)). El par
        (j, métricas) es el objeto que la Fase III observa y orienta
        en el ciclo OODA.

        Este método *es* el arranque formal de `_OODAController`.
        """
        floer_v = self.classify_floer(germ.floer, germ.floer_scale)
        cech_v = self.classify_cech(germ.cech, germ.cech_scale)
        joined = self.join(floer_v.verdict, cech_v.verdict)
        meet_g = float(min(floer_v.godel_value, cech_v.godel_value))
        return _OODAActuationGerm(
            floer=floer_v,
            cech=cech_v,
            heyting_join=joined,
            godel_meet=meet_g,
            two_n=int(germ.two_n),
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
    floer_residual: float
    action_potential: float
    floer_verdict: str
    cech_obstruction: float
    cech_active_modes: int
    cech_verdict: str
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    godel_meet: float
    conley_zehnder_index: float
    cech_betti_0: int
    cech_betti_1: int
    filter_is_prime: bool
    observe_ok: bool

    def as_public_dict(self) -> Dict[str, Any]:
        """Contrato 2.0: claves históricas del ciclo."""
        return {
            "heyting_verdict": self.heyting_verdict,
            "floer_residual": self.floer_residual,
            "action_potential": self.action_potential,
            "floer_verdict": self.floer_verdict,
            "cech_obstruction": self.cech_obstruction,
            "cech_active_modes": self.cech_active_modes,
            "cech_verdict": self.cech_verdict,
            "hardware_interlock_fired": self.hardware_interlock_fired,
            "actuation_latency_ns": self.actuation_latency_ns,
        }


class _OODAController:
    """
    Fase III. Ciclo Observe–Orient–Decide–Act.

    Continúa el gérmen 𝒢_II. El filtro primo (principal) sobre H₃ es

        x ∈ 𝒰  ⇔  ∨(Floer, Čech) = VETOED .

    Su función característica es `hardware_interlock_fired`. La latencia
    reportada es un *presupuesto lógico* (constante de API 2.0), no una
    medición de silicio: este módulo no conmuta hardware.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def observe(germ: _OODAActuationGerm) -> Tuple[_FloerVeredict, _CechVeredict]:
        """O — Observe: extrae las lecturas locales ya valuadas."""
        return germ.floer, germ.cech

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
        inyectado). No hay GPIO, ni IRAM, ni tiristores aquí.
        """
        if not interlock:
            return 0.0
        jitter = float(self._rng.normal(0.0, _INTERLOCK_JITTER_NS))
        latency = _INTERLOCK_LATENCY_BUDGET_NS + jitter
        return float(np.clip(latency, 380.0, 420.0))

    def run(self, germ: _OODAActuationGerm) -> _OODAResult:
        """Ejecuta O→O→D→A sobre el gérmen de Fase II."""
        floer, cech = self.observe(germ)
        joined = self.orient(germ)
        fire = self.decide(joined)
        latency = self.act(fire)
        observe_ok = bool(
            np.isfinite(floer.floer_residual) and np.isfinite(cech.cech_obstruction)
        )
        if fire:
            logger.critical(
                "VETO ATÓMICO DE ERUDITOS COHOMOLÓGICOS. "
                "Join H₃ = VETOED (Floer=%s, Čech=%s). "
                "Interlock lógico ACTIVADO. Presupuesto de latencia = %.2f ns. "
                "No hay conmutación de silicio en este módulo.",
                floer.verdict,
                cech.verdict,
                latency,
            )
        return _OODAResult(
            heyting_verdict=joined,
            floer_residual=float(floer.floer_residual),
            action_potential=float(floer.action_potential),
            floer_verdict=floer.verdict,
            cech_obstruction=float(cech.cech_obstruction),
            cech_active_modes=int(cech.active_modes_count),
            cech_verdict=cech.verdict,
            hardware_interlock_fired=bool(fire),
            actuation_latency_ns=float(latency),
            godel_meet=float(germ.godel_meet),
            conley_zehnder_index=float(floer.conley_zehnder_index),
            cech_betti_0=int(cech.betti_0),
            cech_betti_1=int(cech.betti_1),
            filter_is_prime=True,
            observe_ok=observe_ok,
        )


# =============================================================================
# AGENTE PÚBLICO — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialGuardsEruditosAgent:
    """
    Soberano agéntico de Cohomología Simpléctica y Atencional (Capa 4.5).

    Compone las tres fases anidadas:

    1. Fase I   — auditoría ciega (`_AuditCore` + motor).
    2. Fase II  — valuación H₃ (`_HeytingClassifier`).
    3. Fase III — OODA / interlock lógico (`_OODAController`).

    La API pública de 2.0 se conserva (`_FloerVeredict`, `_CechVeredict`,
    dict del ciclo). Los métodos `*_certified` y los morfismos
    `synthesize_*` / `induce_*` exponen los invariantes 3.0.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        regularizer: float = 1e-15,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Inicializa las aduanas de control cohomológico.

        Args:
            dimension_n: Dimensión del espacio de fases T*Q (debe ser par).
            safety_margin: Holgura μ ≥ 0 que escala los umbrales H₃.
            regularizer: Piso de Tikhonov reenviado al motor (si lo acepta).
            rng: Generador para el jitter del presupuesto de latencia.
        """
        if int(dimension_n) <= 0 or int(dimension_n) % 2 != 0:
            raise ValueError(
                f"La dimensión del espacio simpléctico de fase n={dimension_n} debe ser par."
            )
        if not np.isfinite(safety_margin) or safety_margin < 0.0:
            raise ValueError("safety_margin debe ser finito y ≥ 0.")
        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)
        self._reg: Final[float] = float(max(regularizer, 1e-20))

        try:
            self._engine: Final[ImperialEruditosEngine] = ImperialEruditosEngine(
                regularizer=self._reg
            )
        except TypeError:
            self._engine = ImperialEruditosEngine()  # type: ignore[misc]

        # Fase I → objeto inicial de Fase II (se rellena en cada ciclo).
        self._audit_core = _AuditCore(self._engine, two_n=self._n)
        self._classifier = _HeytingClassifier(self._safety_margin)
        self._ooda = _OODAController(rng=rng)
        self._audit_germ: Optional[_HeytingAuditGerm] = None
        self._ooda_germ: Optional[_OODAActuationGerm] = None

    @property
    def dimension(self) -> int:
        """Dimensión de Darboux (2n) con la que se instanció el soberano."""
        return self._n

    @property
    def safety_margin(self) -> float:
        return self._safety_margin

    @property
    def engine(self) -> ImperialEruditosEngine:
        return self._engine

    # ── Fase I / II expuestas (API 2.0) ───────────────────────────────────
    @staticmethod
    def _veredict_from_metric(
        metric: float,
        base_tolerance: float,
        safety_margin: float,
        degradation_factor: float = _DEGRADATION_FACTOR,
    ) -> str:
        """Clasificador H₃ estático (firma 2.0, delega al clasificador)."""
        return _HeytingClassifier(safety_margin).verdict_from_metric(
            metric, base_tolerance, safety_margin, degradation_factor
        )

    def synthesize_heyting_audit_germ(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
    ) -> _HeytingAuditGerm:
        """Réplica pública del morfismo I.7."""
        germ = self._audit_core.synthesize_heyting_audit_germ(
            start_point,
            end_point,
            jacobian_m3,
            attention_sheaf_matrix,
            safety_margin=self._safety_margin,
        )
        self._audit_germ = germ
        return germ

    def audit_floer_homology_trajectory(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
    ) -> _FloerVeredict:
        r"""
        [ERUDITO 1 — AUDIT DE FLOER]

        Audita la estabilidad de las trayectorias en el complejo de
        cadenas de Floer. Exige que el residuo del cilindro no perturbe
        la nilpotencia:

            residuo_{Floer} \le τ_{Floer}(μ, σ_M₃)
        """
        if np.asarray(start_point).ndim != 1 or np.asarray(end_point).ndim != 1:
            logger.error("Los puntos deben ser vectores unidimensionales.")
            return _FloerVeredict(
                floer_residual=float("inf"),
                action_potential=float("inf"),
                verdict="VETOED",
                godel_value=0.0,
            )
        if np.asarray(start_point).shape != np.asarray(end_point).shape:
            logger.error("Los puntos deben tener la misma dimensión.")
            return _FloerVeredict(
                floer_residual=float("inf"),
                action_potential=float("inf"),
                verdict="VETOED",
                godel_value=0.0,
            )
        audit = self._audit_core.floer_audit(start_point, end_point, jacobian_m3)
        try:
            scale = max(float(np.linalg.norm(np.asarray(jacobian_m3), "fro")), 1.0)
        except Exception:
            scale = 1.0
        return self._classifier.classify_floer(audit, scale)

    def audit_attention_cech_cohomology(
        self,
        attention_sheaf_matrix: np.ndarray,
    ) -> _CechVeredict:
        r"""
        [ERUDITO 2 — AUDIT DE ČECH]

        Audita la obstrucción de Čech sobre las mallas de tokens.
        Exige nulidad homológica numérica

            Ȟ¹(𝔘; ℱ_att) ≈ 0   (masa nuclear ≤ τ_Čech).
        """
        audit = self._audit_core.cech_audit(attention_sheaf_matrix)
        if np.isfinite(audit.nuclear_mass) and audit.nuclear_mass > 0.0:
            scale = max(float(audit.nuclear_mass), 1.0)
        else:
            scale = 1.0
        return self._classifier.classify_cech(audit, scale)

    def induce_ooda_actuation_germ(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
    ) -> _OODAActuationGerm:
        """Réplica pública del morfismo II.6; actualiza 𝒢_II."""
        audit_germ = self.synthesize_heyting_audit_germ(
            start_point, end_point, jacobian_m3, attention_sheaf_matrix
        )
        ooda_germ = self._classifier.induce_ooda_actuation_germ(audit_germ)
        self._ooda_germ = ooda_germ
        return ooda_germ

    # ── Fase III expuesta ─────────────────────────────────────────────────
    def execute_eruditos_cycle(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Ejecuta el ciclo OODA de los Eruditos de Cohomología.

        Returns:
            Diccionario 2.0 con el veredicto global y métricas detalladas.
        """
        return self.execute_eruditos_cycle_certified(
            start_point, end_point, jacobian_m3, attention_sheaf_matrix
        ).as_public_dict()

    def execute_eruditos_cycle_certified(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
    ) -> _OODAResult:
        """OODA certificado: join H₃, Gödel, CZ, Betti e interlock lógico."""
        germ = self.induce_ooda_actuation_germ(
            start_point, end_point, jacobian_m3, attention_sheaf_matrix
        )
        return self._ooda.run(germ)


__all__ = ["ImperialGuardsEruditosAgent"]
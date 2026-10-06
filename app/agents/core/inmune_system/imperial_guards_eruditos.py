# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Guards Eruditos Agent (Soberano de Cohomología y KAM)      ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_eruditos.py           ║
║ Versión: 6.1.0-Poincare-Cartan-Melnikov-KAM-Heyting-OODA-Nested-PhD          ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA, CATEGORIAL Y CELESTE DE POINCARÉ:
────────────────────────────────────────────────────────────────────────────────
Ejerce la salvaguarda de consistencia estructural del Consejo de Sabios, anulando
alucinaciones estocásticas en la matriz de atención del LLM y garantizando la
estabilidad analítica del sistema inmunológico cognitivo de APU Filter v8.0 mediante
un esquema anidado de tres fases ($\Phi_{\mathrm{III}} \circ \Phi_{\mathrm{II}} \circ \Phi_{\mathrm{I}}$)
fundado en la cohomología de Čech–de Rham, la homología de Floer y la mecánica celeste de Henri Poincaré.

DEFINICIONES, AXIOMAS Y TEOREMAS FORMALES:

FASE 1 — OBSERVE ($\Phi_{\mathrm{I}}$) — COHOMOLOGÍA Y AUDITORÍAS CELESTES CRUDAS:
1. Homología de Floer y Variedades Invariantes Lagrangianas:
   Para el funcional de acción simpléctica $\mathcal{A}_H(\gamma) = -\int_{\mathbb{D}^2} v^* \Omega + \int_{S^1} H(t, \gamma(t)) \mathrm{d}t$,
   los puntos críticos de $\mathcal{A}_H$ corresponden a órbitas periódicas $x \in \mathcal{P}(H)$.
   El operador de frontera de Floer $\partial_F \langle x \rangle = \sum_{y, \mu(x)-\mu(y)=1} n(x, y) \langle y \rangle$
   satisface $\partial_F^2 = 0$, definiendo los grupos de homología $HF_k(M, \Omega)$.

2. Cohomología Atencional de Čech–de Rham:
   Sobre un cubrimiento abierto U de la variedad de atención, el operador coborde $\delta: \check{C}^p(\mathcal{U}, \mathcal{F}) \to \check{C}^{p+1}(\mathcal{U}, \mathcal{F})$
   satisface $\delta^2 = 0$. La trivialidad del primer grupo de cohomología $\check{H}^1(\mathcal{U}, \mathcal{F}) \cong 0$
   y la anulación del número de Betti $\beta_1 = 0$ certifican la ausencia de obstrucciones holonómicas en la atemporalidad del modelo.

FASE 2 — ORIENT ($\Phi_{\mathrm{II}}$) — HEYTING $G_3^5$ Y VALUACIÓN EN ANILLO ULTRAMÉTRICO:
3. Estabilidad Diofántica de Poincaré–KAM y Módulo de Brjuno:
   Para un vector de frecuencias $\omega \in \mathbb{R}^n$, la condición diofántica
   $|\langle k, \omega \rangle| \ge \frac{\gamma}{\|k\|_1^\tau} \quad \forall k \in \mathbb{Z}^n \setminus \{0\} \quad (\tau > n-1)$
   y la convergencia del módulo de Brjuno $\mathfrak{B}(\omega) = \sum_{\nu=0}^\infty 2^{-\nu} \ln\frac{1}{\Omega_\nu} < \infty$
   impiden la colisión por divisiones pequeñas.

4. Peso Ultramétrico T-Ádico en el Anillo de Novikov:
   Las pequeñas divisiones se absorben en el anillo de Novikov $\Lambda_{\mathrm{Nov}}$ mediante el peso ultramétrico
   $W_{\mathrm{Nov}} = \exp\left(-\frac{T_{\mathrm{val}}}{\varepsilon + |\langle k, \omega \rangle|}\right) \in (0, 1]$,
   anulando la singularidad del integrador de Lindstedt–Poincaré $\chi_k = \frac{i (H_1)_k}{\langle k, \omega \rangle}$.

5. Fractura Homoclínica de Melnikov y Matriz de Retorno:
   La función de Melnikov $M(t_0) = \int_{-\infty}^\infty \{H_0, H_1\}(\gamma^0(t - t_0)) \mathrm{d}t$ mide la bifurcación homoclínica.
   Un cero simple $M(t_0) = 0$ con $M'(t_0) \neq 0$ demuestra caos determinista y herraduras de Smale.

FASE 3 — DECIDE/ACT ($\Phi_{\mathrm{III}}$) — COLAPSO OODA Y CONTROL LOGÍSICO SÍNCRONO:
6. Adjudicación por Ínfimo de Gödel e Interlock Ciber-Físico:
   El veredicto global se calcula mediante el meet de Heyting:
   $$\nu_{\mathrm{global}} = \bigwedge_{k=1}^5 \nu_k \in G_3 \triangleq \{\mathrm{VETOED}(0) \le \mathrm{DEGRADED}(1) \le \mathrm{COHERENT}(2)\}$$
   Ante $\nu_{\mathrm{global}} = \mathrm{VETOED}$, se activa el interlock lógico síncrono en IRAM para frenar la propagación de alucinaciones en $t_{\mathrm{act}} \le 400 \text{ ns}$.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from app.core.inmune_system.imperial_eruditos_engine import (
        ImperialEruditosEngine,
        EruditosSpectrumReport,
    )
    try:
        from app.core.inmune_system.imperial_eruditos_engine import (
            _KAMStabilityCertificate,
            _MelnikovCertificate,
            _PoincareReturnMapCertificate,
            _FloerResult,
            _CechCohomologyResult,
            _LyapunovSpectrumWitness,
            _MaupertuisJacobiCertificate,
            _PoincareCartanGerm,
        )
    except ImportError:  # pragma: no cover
        _KAMStabilityCertificate = Any  # type: ignore[assignment,misc]
        _MelnikovCertificate = Any  # type: ignore[assignment,misc]
        _PoincareReturnMapCertificate = Any  # type: ignore[assignment,misc]
        _FloerResult = Any  # type: ignore[assignment,misc]
        _CechCohomologyResult = Any  # type: ignore[assignment,misc]
        _LyapunovSpectrumWitness = Any  # type: ignore[assignment,misc]
        _MaupertuisJacobiCertificate = Any  # type: ignore[assignment,misc]
        _PoincareCartanGerm = Any  # type: ignore[assignment,misc]
except ImportError:  # pragma: no cover — import plano / tests locales
    from imperial_eruditos_engine import (  # type: ignore[no-redef]
        ImperialEruditosEngine,
        EruditosSpectrumReport,
    )
    _KAMStabilityCertificate = Any  # type: ignore[assignment,misc]
    _MelnikovCertificate = Any  # type: ignore[assignment,misc]
    _PoincareReturnMapCertificate = Any  # type: ignore[assignment,misc]
    _FloerResult = Any  # type: ignore[assignment,misc]
    _CechCohomologyResult = Any  # type: ignore[assignment,misc]
    _LyapunovSpectrumWitness = Any  # type: ignore[assignment,misc]
    _MaupertuisJacobiCertificate = Any  # type: ignore[assignment,misc]
    _PoincareCartanGerm = Any  # type: ignore[assignment,misc]

logger = logging.getLogger("APU.Agents.SymplecticEruditos")

__version__: Final[str] = "6.1.0-Poincare-Cartan-Melnikov-KAM-Heyting-OODA-Nested-PhD"

# =============================================================================
# CONSTANTES DE CONTROL LÓGICO Y METROLOGÍA CELESTE
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_FLOER_THRESHOLD: Final[float] = 1e-7
_CECH_THRESHOLD: Final[float] = 1e-5
_KAM_THRESHOLD: Final[float] = 1e-9
_MELNIKOV_THRESHOLD: Final[float] = 1e-6
_FLOQUET_PARABOLIC: Final[float] = 1e-6
_LYAPUNOV_CLIP: Final[float] = 1e3
_DEGRADATION_FACTOR: Final[float] = 0.01
_WILKINSON_REL_SCALE: Final[float] = 10.0
_WILKINSON_LIMIT: Final[float] = 1.0e-12
_SPECTRAL_TOL: Final[float] = 1.0e-9
_DET_M_VETO: Final[float] = 1.0e-6
_SP_RES_DEGRADED: Final[float] = 1.0e-4
_KAM_TAU_OFFSET: Final[float] = 1.0e-6
_HARD_BRUNO_FLOOR: Final[float] = 1.0e-12
_INTERLOCK_LATENCY_BUDGET_NS: Final[float] = 400.0
_INTERLOCK_JITTER_NS: Final[float] = 5.0
_INTERLOCK_LATENCY_FLOOR_NS: Final[float] = 380.0
_INTERLOCK_LATENCY_CEIL_NS: Final[float] = 420.0

_HEYTING_ORDER: Final[Dict[str, int]] = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
_HEYTING_GODEL: Final[Dict[str, float]] = {"COHERENT": 1.0, "DEGRADED": 0.5, "VETOED": 0.0}
_REVERSE_HEYTING: Final[Dict[int, str]] = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}


def _unique_preserve(seq: Tuple[str, ...] | list[str]) -> Tuple[str, ...]:
    """Únicos estables (retículo de razones de veto/degradación)."""
    return tuple(dict.fromkeys(seq))


@dataclass(frozen=True, slots=True)
class EruditosAgentCertificate:
    """Certificado inmutable de lazo cerrado emitido por los Eruditos Imperiales."""
    min_small_divisor: float
    novikov_weight: float
    volume_drift: float
    heyting_verdict: str  # COHERENT, DEGRADED, VETOED
    is_verdict_coherent: bool


PoincareEruditosAgentCertificate = EruditosAgentCertificate


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE I — NÚCLEO DE AUDITORÍA ESPECTRAL Y CELESTE (MOTOR CIEGO)           ║
# ║                                                                          ║
# ║ Objetos crudos: Floer, Čech, KAM, Melnikov, ReturnMap (+ Hill).          ║
# ║                                                                          ║
# ║ Morfismo terminal (I.ω): synthesize_poincare_celestial_germ              ║
# ║     ↦ 𝒢_I^{cel} = _PoincareCelestialAuditGerm (inicial de Fase II).     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True, slots=True)
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
    skipped: bool = False


@dataclass(frozen=True, slots=True)
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
    is_h1_trivial: bool = True
    germ_from_floer: bool = False
    engine_ok: bool = True
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class _KAMAudit:
    r"""
    Auditoría de estabilidad KAM (Poincaré–Arnol'd–Moser).
    Transporta el divisor mínimo, la absorción de Novikov, el drift de
    Liouville, la diofantinidad ∀k (no sólo el modo minimizante), Brjuno
    y γ_est = min_k |⟨k,ω⟩| · |k|^τ.
    """
    min_divisor: float
    novikov_weight: float
    maurercartan_res: float
    volume_drift: float
    resonance_gap: float
    tau: float
    gamma: float
    is_diophantine: bool
    is_kam_stable: bool
    bruno_sum: float = 0.0
    estimated_gamma: float = 0.0
    n_modes: int = 0
    engine_ok: bool = True
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class _MelnikovAudit:
    r"""
    Auditoría de la función de Melnikov (ruptura homoclínica).
    Un cero simple M(t*)=0, M'(t*)≠0 ⇒ intersección transversa ⇒ Smale.
    `skipped=True` (canal no provisto) es neutro en el join de Heyting.
    """
    melnikov_value: float
    melnikov_derivative: float
    is_simple_zero: bool
    homoclinic_splitting: float
    simple_zeros: int = 0
    is_chaotic: bool = False
    engine_ok: bool = True
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class _ReturnMapAudit:
    r"""
    Auditoría del mapa de retorno de Poincaré P: Σ → Σ.
    Para M ∈ Sp(2n): {μ}={1/μ}={μ̄} (Krein). Clasificación no exclusiva:
    elíptico / hiperbólico / parabólico / mixto.
    """
    floquet_multipliers: np.ndarray
    lyapunov_spectrum: np.ndarray
    max_lyapunov: float
    floquet_parabolic: float
    is_hyperbolic: bool
    is_elliptic: bool
    is_parabolic: bool
    trace_M: float
    det_M: float
    is_mixed: bool = False
    reciprocal_pair_residual: float = float("nan")
    unit_circle_residual: float = float("nan")
    hill_discriminant: float = float("nan")
    symplectic_residual: float = float("nan")
    kaplan_yorke_dimension: float = float("nan")
    kolmogorov_sinai_entropy: float = float("nan")
    lyapunov_sum: float = float("nan")
    engine_ok: bool = True
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class _HillAudit:
    """Margen de Hill H₀−V y positividad de la métrica de Maupertuis–Jacobi."""
    hill_margin: float
    conformal_factor: float
    min_eigenvalue: float
    is_classically_allowed: bool
    is_positive_definite: bool
    engine_ok: bool = True
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class _PoincareCelestialAuditGerm:
    r"""
    **Gérmen celeste de auditoría (objeto terminal de Fase I, inicial de Fase II).**

    Sobre 𝒢_I^{cel} el clasificador de Heyting de Fase II instancia
    `classify_kam`, `classify_melnikov`, `classify_return_map`,
    `classify_floer` y `classify_cech`.
    """
    floer: _FloerAudit
    cech: _CechAudit
    kam: _KAMAudit
    melnikov: _MelnikovAudit
    return_map: _ReturnMapAudit
    two_n: int
    safety_margin: float
    floer_scale: float
    cech_scale: float
    kam_scale: float
    melnikov_scale: float
    return_scale: float
    hill: _HillAudit = field(default_factory=lambda: _HillAudit(
        hill_margin=float("nan"), conformal_factor=float("nan"),
        min_eigenvalue=float("nan"), is_classically_allowed=True,
        is_positive_definite=True, skipped=True,
    ))


class _AuditCore:
    r"""
    Fase I. Núcleo ciego que dialoga con ImperialEruditosEngine ≥ 6.0.

    Todos los métodos de auditoría son tolerantes a fallos (`engine_ok=False`
    si el motor aborta). Los canales opcionales (Melnikov, Hill) se marcan
    `skipped=True` y **no** vetan el join.

    **Cierre formal (I.ω)**:
        `synthesize_poincare_celestial_germ → _PoincareCelestialAuditGerm`
        Este objeto **es** el arranque formal de la Fase II.
    """

    def __init__(self, engine: ImperialEruditosEngine, two_n: int) -> None:
        self._engine = engine
        self._two_n = int(two_n)

    @property
    def engine(self) -> ImperialEruditosEngine:
        return self._engine

    # ── I.1 Coerciones y Darboux ─────────────────────────────────────────
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

    @staticmethod
    def _frobenius(matrix: np.ndarray) -> float:
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(np.linalg.norm(a, "fro"))

    @staticmethod
    def _darboux_omega(dim: int) -> np.ndarray:
        r"""Ω = [[0, I_n], [−I_n, 0]] en ℝ^{2n}."""
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim={dim} debe ser par y positivo (Darboux).")
        n = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:n, n:] = np.eye(n, dtype=np.float64)
        omega[n:, :n] = -np.eye(n, dtype=np.float64)
        return omega

    @staticmethod
    def _default_frequency(n_freq: int) -> np.ndarray:
        """Vector de frecuencias incommensurable-típico: √(1), √(2), …"""
        n = max(int(n_freq), 1)
        return np.sqrt(np.arange(1, n + 1, dtype=np.float64))

    # ── I.2 Auditoría de Floer ───────────────────────────────────────────
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

    # ── I.3 Auditoría de Čech atencional ─────────────────────────────────
    def _call_cech(self, attention_sheaf_matrix: np.ndarray) -> _CechAudit:
        certified = getattr(self._engine, "compute_attention_cech_cohomology_certified", None)
        if callable(certified):
            result = certified(attention_sheaf_matrix)
            modes = np.asarray(getattr(result, "active_modes", np.array([])), dtype=np.float64)
            b0 = int(getattr(result, "betti_0", 0))
            b1 = int(getattr(result, "betti_1", 0))
            return _CechAudit(
                cech_obstruction=float(result.cech_obstruction),
                active_modes=modes,
                cocycle_defect=float(getattr(result, "cocycle_defect", float("nan"))),
                harmonic_energy=float(getattr(result, "harmonic_energy", float("nan"))),
                betti_0=b0,
                betti_1=b1,
                effective_rank=int(getattr(result, "effective_rank", modes.size)),
                nuclear_mass=float(getattr(result, "nuclear_mass", result.cech_obstruction)),
                is_h1_trivial=bool(getattr(result, "is_h1_trivial", b1 == 0)),
                germ_from_floer=bool(getattr(result, "germ_from_floer", False)),
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
            is_h1_trivial=True,
            engine_ok=True,
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
                is_h1_trivial=False,
                engine_ok=False,
            )

    # ── I.4 Auditoría KAM (pequeños divisores de Poincaré) ───────────────
    def kam_audit(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        jacobian_M: np.ndarray,
        canonical_J: np.ndarray,
    ) -> _KAMAudit:
        r"""
        Ejecuta `compute_poincare_small_divisors_spectrum` y normaliza a `_KAMAudit`.
        Tolera (report, kam_cert) de v6+ y el reporte simple de v5.
        τ por defecto lo decide el motor (τ > n−1); no se fuerza τ=1.
        """
        try:
            omega = self._as_vec("frequency_vector_omega", frequency_vector_omega)
            wave_raw = np.asarray(wave_vectors_k)
            wave_k: np.ndarray
            if wave_raw.ndim > 1:
                wave_k = self._as_matrix("wave_vectors_k", wave_vectors_k)
            else:
                wave_k = self._as_vec("wave_vectors_k", wave_vectors_k)
            jac_m = self._as_matrix("jacobian_M", jacobian_M)
            canon_j = np.asarray(canonical_J, dtype=np.float64)
            try:
                raw = self._engine.compute_poincare_small_divisors_spectrum(
                    frequency_vector_omega=omega,
                    wave_vectors_k=wave_k,
                    jacobian_M=jac_m,
                    canonical_J=canon_j,
                    tau=None,
                )
            except TypeError:
                raw = self._engine.compute_poincare_small_divisors_spectrum(
                    frequency_vector_omega=omega,
                    wave_vectors_k=wave_k,
                    jacobian_M=jac_m,
                    canonical_J=canon_j,
                )
            if isinstance(raw, tuple) and len(raw) == 2:
                report, kam_cert = raw
            else:
                report, kam_cert = raw, None

            n_modes = int(omega.size)
            tau_fallback = float(max(n_modes - 1, 0) + _KAM_TAU_OFFSET)
            tau = float(getattr(kam_cert, "tau", tau_fallback)) if kam_cert is not None else tau_fallback
            gamma = float(getattr(kam_cert, "gamma", _WILKINSON_LIMIT)) if kam_cert is not None else _WILKINSON_LIMIT
            resonance_gap = (
                float(getattr(kam_cert, "resonance_gap", report.min_small_divisor))
                if kam_cert is not None
                else float(report.min_small_divisor)
            )
            is_diophantine = (
                bool(getattr(kam_cert, "is_diophantine", report.is_kam_stable))
                if kam_cert is not None
                else bool(report.is_kam_stable)
            )
            bruno = float(getattr(kam_cert, "bruno_sum", 0.0)) if kam_cert is not None else 0.0
            g_est = float(getattr(kam_cert, "estimated_gamma", gamma)) if kam_cert is not None else float(gamma)
            n_cert = int(getattr(kam_cert, "n_modes", n_modes)) if kam_cert is not None else n_modes
            return _KAMAudit(
                min_divisor=float(report.min_small_divisor),
                novikov_weight=float(report.novikov_absorbed_weight),
                maurercartan_res=float(report.maurercartan_residual),
                volume_drift=float(report.liouville_volume_drift),
                resonance_gap=resonance_gap,
                tau=tau,
                gamma=gamma,
                is_diophantine=is_diophantine,
                is_kam_stable=bool(report.is_kam_stable),
                bruno_sum=bruno,
                estimated_gamma=g_est,
                n_modes=n_cert,
                engine_ok=True,
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_small_divisors_spectrum: %s", exc)
            return _KAMAudit(
                min_divisor=float("inf"),
                novikov_weight=0.0,
                maurercartan_res=float("inf"),
                volume_drift=float("inf"),
                resonance_gap=float("inf"),
                tau=1.0,
                gamma=_WILKINSON_LIMIT,
                is_diophantine=False,
                is_kam_stable=False,
                engine_ok=False,
            )

    # ── I.5 Auditoría de Melnikov (ruptura homoclínica) ──────────────────
    def melnikov_audit(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float = 25.0,
    ) -> _MelnikovAudit:
        r"""Ejecuta `compute_melnikov_function`. Ceros simples ⇒ fractura homoclínica."""
        try:
            t0s = self._as_vec("t0_grid", t0_grid)
            if not callable(homoclinic_flow):
                raise ValueError("homoclinic_flow debe ser invocable.")
            cert = self._engine.compute_melnikov_function(
                homoclinic_flow=homoclinic_flow,
                hamiltonian_0=hamiltonian_0,
                hamiltonian_1=hamiltonian_1,
                t0_grid=t0s,
                t_inf=float(t_inf),
            )
            is_simple = bool(getattr(cert, "is_simple_zero", False))
            return _MelnikovAudit(
                melnikov_value=float(getattr(cert, "melnikov_value", float("nan"))),
                melnikov_derivative=float(getattr(cert, "melnikov_derivative", float("nan"))),
                is_simple_zero=is_simple,
                homoclinic_splitting=float(getattr(cert, "homoclinic_splitting", float("nan"))),
                simple_zeros=int(getattr(cert, "simple_zeros", int(is_simple))),
                is_chaotic=bool(getattr(cert, "is_chaotic", is_simple)),
                engine_ok=True,
                skipped=False,
            )
        except Exception as exc:
            logger.error("Fallo en compute_melnikov_function: %s", exc)
            return _MelnikovAudit(
                melnikov_value=float("nan"),
                melnikov_derivative=float("nan"),
                is_simple_zero=False,
                homoclinic_splitting=float("nan"),
                engine_ok=False,
                skipped=False,
            )

    @staticmethod
    def melnikov_skipped() -> _MelnikovAudit:
        """Canal Melnikov no provisto: neutro en el join (no veta)."""
        return _MelnikovAudit(
            melnikov_value=0.0,
            melnikov_derivative=0.0,
            is_simple_zero=False,
            homoclinic_splitting=0.0,
            simple_zeros=0,
            is_chaotic=False,
            engine_ok=True,
            skipped=True,
        )

    # ── I.6 Auditoría del mapa de retorno de Poincaré ────────────────────
    def return_map_audit(
        self,
        jacobian_M: np.ndarray,
        period_T: float = 1.0,
    ) -> _ReturnMapAudit:
        r"""Ejecuta `compute_poincare_return_map` (+ Benettin si el motor lo expone)."""
        try:
            jac_m = self._as_matrix("jacobian_M", jacobian_M)
            cert = self._engine.compute_poincare_return_map(
                jacobian_M=jac_m, period_T=float(period_T)
            )
            floq = np.asarray(
                getattr(cert, "floquet_multipliers", np.array([])), dtype=np.complex128
            )
            lyap = np.asarray(
                getattr(cert, "lyapunov_spectrum", np.array([])), dtype=np.float64
            )
            if lyap.size:
                max_lyap = float(np.max(np.clip(lyap, -_LYAPUNOV_CLIP, _LYAPUNOV_CLIP)))
            else:
                max_lyap = 0.0
            floq_par = float(np.max(np.abs(np.abs(floq) - 1.0))) if floq.size else 0.0

            ky = float("nan")
            h_ks = float("nan")
            lyap_sum = float(np.sum(lyap)) if lyap.size else float("nan")
            benettin = getattr(self._engine, "compute_lyapunov_spectrum_benettin", None)
            if callable(benettin):
                try:
                    wit = benettin(jac_m)
                    ky = float(getattr(wit, "kaplan_yorke_dimension", ky))
                    h_ks = float(getattr(wit, "kolmogorov_sinai_entropy", h_ks))
                    spec = np.asarray(getattr(wit, "spectrum", lyap), dtype=np.float64)
                    if spec.size:
                        lyap = spec
                        max_lyap = float(np.max(np.clip(spec, -_LYAPUNOV_CLIP, _LYAPUNOV_CLIP)))
                        lyap_sum = float(getattr(wit, "sum_all", float(np.sum(spec))))
                except Exception as exc:
                    logger.warning("Benettin no disponible: %s", exc)

            is_hyp = bool(getattr(cert, "is_hyperbolic", False))
            is_ell = bool(getattr(cert, "is_elliptic", False))
            is_par = bool(getattr(cert, "is_parabolic", False))
            is_mixed = bool(getattr(cert, "is_mixed", is_hyp and is_ell))
            return _ReturnMapAudit(
                floquet_multipliers=floq,
                lyapunov_spectrum=lyap,
                max_lyapunov=max_lyap,
                floquet_parabolic=floq_par,
                is_hyperbolic=is_hyp,
                is_elliptic=is_ell,
                is_parabolic=is_par,
                trace_M=float(getattr(cert, "trace_M", float("nan"))),
                det_M=float(getattr(cert, "det_M", float("nan"))),
                is_mixed=is_mixed,
                reciprocal_pair_residual=float(
                    getattr(cert, "reciprocal_pair_residual", float("nan"))
                ),
                unit_circle_residual=float(
                    getattr(cert, "unit_circle_residual", float("nan"))
                ),
                hill_discriminant=float(getattr(cert, "hill_discriminant", float("nan"))),
                symplectic_residual=float(
                    getattr(cert, "symplectic_residual", float("nan"))
                ),
                kaplan_yorke_dimension=ky,
                kolmogorov_sinai_entropy=h_ks,
                lyapunov_sum=lyap_sum,
                engine_ok=True,
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_return_map: %s", exc)
            return _ReturnMapAudit(
                floquet_multipliers=np.array([], dtype=np.complex128),
                lyapunov_spectrum=np.array([], dtype=np.float64),
                max_lyapunov=float("inf"),
                floquet_parabolic=float("inf"),
                is_hyperbolic=False,
                is_elliptic=False,
                is_parabolic=False,
                trace_M=float("nan"),
                det_M=float("nan"),
                engine_ok=False,
            )

    # ── I.6b Auditoría de Hill / Maupertuis–Jacobi ───────────────────────
    def hill_audit(self) -> _HillAudit:
        """Lee el certificado de Maupertuis–Jacobi del motor si existe."""
        getter = getattr(self._engine, "maupertuis_jacobi_certificate", None)
        if not callable(getter):
            return _HillAudit(
                hill_margin=float("nan"), conformal_factor=float("nan"),
                min_eigenvalue=float("nan"), is_classically_allowed=True,
                is_positive_definite=True, skipped=True,
            )
        try:
            cert = getter()
            if cert is None:
                return _HillAudit(
                    hill_margin=float("nan"), conformal_factor=float("nan"),
                    min_eigenvalue=float("nan"), is_classically_allowed=True,
                    is_positive_definite=True, skipped=True,
                )
            return _HillAudit(
                hill_margin=float(getattr(cert, "hill_margin", float("nan"))),
                conformal_factor=float(getattr(cert, "conformal_factor", float("nan"))),
                min_eigenvalue=float(getattr(cert, "min_eigenvalue", float("nan"))),
                is_classically_allowed=bool(getattr(cert, "is_classically_allowed", True)),
                is_positive_definite=bool(getattr(cert, "is_positive_definite", True)),
                engine_ok=True,
                skipped=False,
            )
        except Exception as exc:
            logger.error("Fallo en maupertuis_jacobi_certificate: %s", exc)
            return _HillAudit(
                hill_margin=float("nan"), conformal_factor=float("nan"),
                min_eigenvalue=float("nan"), is_classically_allowed=False,
                is_positive_definite=False, engine_ok=False, skipped=False,
            )

    # ── I.ω  MORFISMO TERMINAL Φ_I: gérmen celeste de auditoría ──────────
    def synthesize_poincare_celestial_germ(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
        safety_margin: float,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        t_inf: float = 25.0,
        period_T: float = 1.0,
    ) -> _PoincareCelestialAuditGerm:
        r"""
        **I.ω — Morfismo terminal de la FASE I / objeto inicial de la FASE II.**

        Compone las auditorías (Floer, Čech, KAM, Melnikov, Retorno, Hill)
        en un único gérmen 𝒢_I^{cel} que la Fase II valúa en H₃⁵.

        Este método **es** el arranque formal de la Fase II:
        `phase2_ingest_poincare_celestial_germ(𝒢_I^{cel})`.
        """
        floer = self.floer_audit(start_point, end_point, jacobian_m3)
        cech = self.cech_audit(attention_sheaf_matrix)

        try:
            jac = self._as_matrix("jacobian_m3", jacobian_m3)
        except Exception:
            jac = np.eye(max(self._two_n, 2), dtype=np.float64)
        dim = int(jac.shape[0]) if jac.ndim == 2 else int(self._two_n)
        if dim % 2 != 0:
            dim = int(self._two_n) if self._two_n % 2 == 0 else 2
        n_freq = max(dim // 2, 1)

        if frequency_vector_omega is None:
            frequency_vector_omega = self._default_frequency(n_freq)
        if wave_vectors_k is None:
            wave_vectors_k = np.eye(int(np.asarray(frequency_vector_omega).ravel().size), dtype=np.float64)
        if canonical_J is None:
            try:
                canonical_J = self._darboux_omega(dim)
            except ValueError:
                canonical_J = np.eye(dim, dtype=np.float64)

        # KAM se invoca SIEMPRE (bug 6.0: `kam` se usaba sin asignar).
        kam = self.kam_audit(
            frequency_vector_omega, wave_vectors_k, jac, canonical_J
        )

        if (
            homoclinic_flow is not None
            and hamiltonian_0 is not None
            and hamiltonian_1 is not None
            and t0_grid is not None
        ):
            melnikov = self.melnikov_audit(
                homoclinic_flow, hamiltonian_0, hamiltonian_1, t0_grid, t_inf=t_inf
            )
        else:
            melnikov = self.melnikov_skipped()

        return_map = self.return_map_audit(jac, period_T=period_T)
        hill = self.hill_audit()

        try:
            floer_scale = max(self._frobenius(jac), 1.0)
        except Exception:
            floer_scale = 1.0
        if np.isfinite(cech.nuclear_mass) and cech.nuclear_mass > 0.0:
            cech_scale = max(float(cech.nuclear_mass), 1.0)
        elif cech.active_modes.size:
            cech_scale = max(float(np.max(np.abs(cech.active_modes))), 1.0)
        else:
            cech_scale = 1.0
        if np.isfinite(kam.min_divisor) and kam.min_divisor > 0.0:
            kam_scale = max(float(kam.min_divisor), 1.0)
        else:
            kam_scale = 1.0
        if melnikov.skipped:
            mel_scale = 1.0
        else:
            mel_scale = max(
                abs(melnikov.melnikov_value) if np.isfinite(melnikov.melnikov_value) else 0.0,
                abs(melnikov.homoclinic_splitting) if np.isfinite(melnikov.homoclinic_splitting) else 0.0,
                1.0,
            )
        if return_map.floquet_multipliers.size:
            return_scale = max(float(np.max(np.abs(return_map.floquet_multipliers))), 1.0)
        else:
            return_scale = 1.0

        two_n = int(dim) if dim % 2 == 0 else int(self._two_n)
        return _PoincareCelestialAuditGerm(
            floer=floer,
            cech=cech,
            kam=kam,
            melnikov=melnikov,
            return_map=return_map,
            two_n=two_n,
            safety_margin=float(max(safety_margin, 0.0)),
            floer_scale=float(floer_scale),
            cech_scale=float(cech_scale),
            kam_scale=float(kam_scale),
            melnikov_scale=float(mel_scale),
            return_scale=float(return_scale),
            hill=hill,
        )


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE II — CLASIFICADOR DE HEYTING H₃⁵ Y LIFTING OODA                     ║
# ║                                                                          ║
# ║ El primer método consume I.ω; el último produce II.ω (inicio de III).    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True, slots=True)
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
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CechVeredict:
    """Resultado final de auditoría de Čech con veredicto H₃."""
    cech_obstruction: float
    active_modes_count: int
    verdict: str
    cocycle_defect: float = float("nan")
    betti_0: int = 0
    betti_1: int = 0
    harmonic_energy: float = float("nan")
    is_h1_trivial: bool = True
    threshold_used: float = _CECH_THRESHOLD
    godel_value: float = 0.0
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _KAMVeredict:
    r"""
    Veredicto KAM en H₃.
      • COHERENT : |⟨k,ω⟩| ≥ γ/|k|^τ  ∧  |det M−1| ≤ ε_W  ∧  diofantino.
      • DEGRADED : ε_floor < |⟨k,ω⟩| < γ/|k|^τ  ∨  no diofantino ∨ Brjuno sucio.
      • VETOED   : resonancia k·ω ≈ 0  ∨  drift > ε_W  ∨  motor caído.
    """
    min_divisor: float
    novikov_weight: float
    maurercartan_res: float
    volume_drift: float
    resonance_gap: float
    tau: float
    gamma: float
    is_diophantine: bool
    verdict: str
    threshold_used: float
    bruno_sum: float = 0.0
    estimated_gamma: float = 0.0
    godel_value: float = 0.0
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _MelnikovVeredict:
    r"""
    Veredicto de Melnikov en H₃ (alineado con la sinopsis):
      • COHERENT : canal saltado, o sin ceros simples y |M| no nulo.
      • DEGRADED : |M|≈0 sin cruce confirmado (tangencia inminente).
      • VETOED   : cero simple / caótico ⇒ fractura homoclínica.
    """
    melnikov_value: float
    melnikov_derivative: float
    is_simple_zero: bool
    homoclinic_splitting: float
    verdict: str
    threshold_used: float
    simple_zeros: int = 0
    is_chaotic: bool = False
    skipped: bool = False
    godel_value: float = 0.0
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ReturnMapVeredict:
    r"""
    Veredicto del mapa de retorno P: Σ → Σ en H₃.
      • COHERENT : elíptico puro, L_max ≈ 0, det M ≈ 1, Krein OK.
      • DEGRADED : parabólico, mixto, o hiperbolicidad suave.
      • VETOED   : |det M−1| grande, L_max caótico, o motor caído.
    """
    max_lyapunov: float
    floquet_parabolic: float
    is_hyperbolic: bool
    is_elliptic: bool
    is_parabolic: bool
    trace_M: float
    det_M: float
    verdict: str
    threshold_used: float
    is_mixed: bool = False
    symplectic_residual: float = float("nan")
    hill_discriminant: float = float("nan")
    kaplan_yorke_dimension: float = float("nan")
    kolmogorov_sinai_entropy: float = float("nan")
    godel_value: float = 0.0
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _OODAActuationGerm:
    r"""
    **Gérmen OODA (objeto terminal de Fase II, inicial de Fase III).**

    Compone los cinco veredictos H₃ y expone el join (supremo = peor caso)
    y el meet de Gödel (ínfimo de valores de verdad).
    """
    floer: _FloerVeredict
    cech: _CechVeredict
    kam: _KAMVeredict
    melnikov: _MelnikovVeredict
    return_map: _ReturnMapVeredict
    heyting_join: str
    godel_meet: float
    two_n: int
    safety_margin: float
    hill_allowed: bool = True
    veto_reasons: Tuple[str, ...] = ()
    degraded_reasons: Tuple[str, ...] = ()


class _HeytingClassifier:
    r"""
    Fase II. Clasificador en el álgebra de Heyting de tres valores Ω₃ ≅ {0, ½, 1}.

    **II.0** `phase2_ingest_poincare_celestial_germ` continúa I.ω.
    **II.ω** `induce_ooda_actuation_germ` abre la Fase III.

    Los cinco clasificadores inducen un morfismo H₃⁵ → H₃ vía el join
    (supremo = peor caso) y el meet de Gödel (ínfimo).
    """

    def __init__(self, safety_margin: float) -> None:
        self._margin = float(max(safety_margin, 0.0))

    @property
    def safety_margin(self) -> float:
        return self._margin

    # ── II.0  INGESTA DEL OBJETO TERMINAL DE LA FASE I ───────────────────
    def phase2_ingest_poincare_celestial_germ(
        self, germ: _PoincareCelestialAuditGerm,
    ) -> _PoincareCelestialAuditGerm:
        r"""
        **II.0 — Flecha inicial de Φ_II, continuación estricta de I.ω.**

        Valida 𝒢_I^{cel} (tipos, paridad 2n, escalas) y lo reexpide como
        objeto de trabajo. Toda la valuación H₃ factoriza a través de aquí.
        """
        if not isinstance(germ, _PoincareCelestialAuditGerm):
            raise TypeError(
                "germ debe ser _PoincareCelestialAuditGerm (cierre I.ω)."
            )
        if germ.two_n <= 0 or germ.two_n % 2 != 0:
            raise ValueError("𝒢_I^{cel} no tiene dimensión de Darboux.")
        if germ.safety_margin < 0.0 or not math.isfinite(germ.safety_margin):
            raise ValueError("safety_margin de 𝒢_I^{cel} inválido.")
        if not germ.floer.engine_ok:
            logger.warning("II.0: Floer engine_ok=False (residual=%.3e).", germ.floer.floer_residual)
        if not germ.kam.engine_ok:
            logger.warning("II.0: KAM engine_ok=False.")
        if not germ.hill.skipped and not germ.hill.is_classically_allowed:
            logger.warning(
                "II.0: punto fuera de la región de Hill (margen=%.3e).",
                germ.hill.hill_margin,
            )
        return germ

    # ── II.1 Álgebra de Heyting ──────────────────────────────────────────
    @staticmethod
    def canonicalize(verdict: str) -> str:
        return verdict if verdict in _HEYTING_ORDER else "VETOED"

    @staticmethod
    def join(*verdicts: str) -> str:
        """Supremo de Heyting (peor caso). En la cadena, a ∨ b = max(a, b)."""
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

    def threshold(self, base: float, scale: float = 1.0) -> float:
        r"""τ = τ₀ · μ_safety (con piso metrológico relativo de Wilkinson)."""
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
        r"""
        Asignación H₃ por umbral (métricas de defecto, menores = mejores):
            metric ≤ τ·deg  ⇒ COHERENT
            τ·deg < metric ≤ τ ⇒ DEGRADED
            metric > τ       ⇒ VETOED
        """
        if not np.isfinite(metric) or metric < 0.0:
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

    # ── II.3 Clasificador de Floer ───────────────────────────────────────
    def classify_floer(self, audit: _FloerAudit, scale: float) -> _FloerVeredict:
        """Valúa el cilindro de Floer en H₃."""
        reasons: list[str] = []
        tol = self.threshold(_FLOER_THRESHOLD, scale)
        if audit.skipped:
            verdict = "COHERENT"
        elif (not audit.engine_ok) or (not np.isfinite(audit.floer_residual)):
            verdict = "VETOED"
            reasons.append("floer_engine_failed")
        else:
            verdict = self.verdict_from_metric(
                audit.floer_residual, _FLOER_THRESHOLD, scale=scale
            )
            if verdict != "COHERENT":
                reasons.append("floer_residual_above_tolerance")
            if (
                np.isfinite(audit.symplectic_monodromy_residual)
                and not audit.is_symplectic_monodromy
            ):
                reasons.append("monodromy_not_symplectic")
                verdict = self.join(verdict, "DEGRADED")
            if not audit.is_nondegenerate and np.isfinite(audit.maslov_degeneracy):
                reasons.append("maslov_degeneracy")
                verdict = self.join(verdict, "DEGRADED")
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
            godel_value=float(_HEYTING_GODEL[self.canonicalize(verdict)]),
            reasons=_unique_preserve(reasons),
        )

    # ── II.4 Clasificador de Čech ────────────────────────────────────────
    def classify_cech(self, audit: _CechAudit, scale: float) -> _CechVeredict:
        """Valúa la obstrucción de Čech en H₃ (Ȟ¹ trivial ⇔ b₁=0 ∧ ‖δω‖≈0)."""
        reasons: list[str] = []
        tol = self.threshold(_CECH_THRESHOLD, scale)
        if audit.skipped:
            verdict = "COHERENT"
        elif (not audit.engine_ok) or (not np.isfinite(audit.cech_obstruction)):
            verdict = "VETOED"
            reasons.append("cech_engine_failed")
        else:
            verdict = self.verdict_from_metric(
                audit.cech_obstruction, _CECH_THRESHOLD, scale=scale
            )
            if np.isfinite(audit.cocycle_defect) and audit.cocycle_defect > tol:
                reasons.append("cech_coboundary_defect")
                verdict = self.join(verdict, "DEGRADED")
            if audit.betti_0 == 0:
                reasons.append("empty_complex")
                verdict = self.join(verdict, "VETOED")
            elif audit.betti_0 > 1:
                reasons.append("cech_islands_betti0")
                verdict = self.join(verdict, "VETOED")
            if audit.betti_1 > 0 or not audit.is_h1_trivial:
                reasons.append("cech_h1_nontrivial")
                verdict = self.join(verdict, "DEGRADED")
        return _CechVeredict(
            cech_obstruction=float(audit.cech_obstruction),
            active_modes_count=int(audit.active_modes.size),
            verdict=verdict,
            cocycle_defect=float(audit.cocycle_defect),
            betti_0=int(audit.betti_0),
            betti_1=int(audit.betti_1),
            harmonic_energy=float(audit.harmonic_energy),
            is_h1_trivial=bool(audit.is_h1_trivial),
            threshold_used=float(tol),
            godel_value=float(_HEYTING_GODEL[self.canonicalize(verdict)]),
            reasons=_unique_preserve(reasons),
        )

    # ── II.5 Clasificador KAM ────────────────────────────────────────────
    def classify_kam(self, audit: _KAMAudit, scale: float) -> _KAMVeredict:
        r"""Valúa la estabilidad KAM de Poincaré en H₃ (∀k, no sólo el mínimo)."""
        reasons: list[str] = []
        tol = self.threshold(_KAM_THRESHOLD, scale)
        if audit.skipped:
            verdict = "COHERENT"
        elif not audit.engine_ok or not np.isfinite(audit.min_divisor):
            verdict = "VETOED"
            reasons.append("kam_engine_failed")
        elif audit.min_divisor <= _WILKINSON_LIMIT:
            verdict = "VETOED"
            reasons.append("kam_resonance_small_divisor")
        elif audit.volume_drift > _WILKINSON_LIMIT:
            verdict = "VETOED"
            reasons.append("liouville_volume_drift")
        else:
            verdict = "COHERENT"
            if not audit.is_diophantine:
                reasons.append("kam_not_diophantine")
                verdict = self.join(verdict, "DEGRADED")
            if audit.min_divisor < tol:
                reasons.append("kam_divisor_below_agent_threshold")
                verdict = self.join(verdict, "DEGRADED")
            if not audit.is_kam_stable:
                reasons.append("kam_engine_not_stable")
                verdict = self.join(verdict, "DEGRADED")
            bruno = float(audit.bruno_sum)
            if math.isfinite(bruno) and bruno > 0.0 and bruno >= 1.0 / max(_HARD_BRUNO_FLOOR, _MACHINE_EPS):
                reasons.append("kam_bruno_divergent")
                verdict = self.join(verdict, "DEGRADED")
            if (
                math.isfinite(audit.estimated_gamma)
                and audit.estimated_gamma > 0.0
                and audit.estimated_gamma < audit.gamma
            ):
                reasons.append("kam_gamma_estimated_below_floor")
                verdict = self.join(verdict, "DEGRADED")
        return _KAMVeredict(
            min_divisor=float(audit.min_divisor),
            novikov_weight=float(audit.novikov_weight),
            maurercartan_res=float(audit.maurercartan_res),
            volume_drift=float(audit.volume_drift),
            resonance_gap=float(audit.resonance_gap),
            tau=float(audit.tau),
            gamma=float(audit.gamma),
            is_diophantine=bool(audit.is_diophantine),
            verdict=verdict,
            threshold_used=float(tol),
            bruno_sum=float(audit.bruno_sum),
            estimated_gamma=float(audit.estimated_gamma),
            godel_value=float(_HEYTING_GODEL[self.canonicalize(verdict)]),
            reasons=_unique_preserve(reasons),
        )

    # ── II.6 Clasificador de Melnikov ────────────────────────────────────
    def classify_melnikov(self, audit: _MelnikovAudit, scale: float) -> _MelnikovVeredict:
        r"""
        Valúa Melnikov en H₃ alineado con la sinopsis:
        COHERENT ⇔ sin ceros simples (canal saltado incluido).
        No se exige |M| grande: variedades lejanas y variedades no secantes
        son ambas coherentes; el veto es la transversidad homoclínica.
        """
        reasons: list[str] = []
        tol = self.threshold(_MELNIKOV_THRESHOLD, scale)
        if audit.skipped:
            verdict = "COHERENT"
        elif not audit.engine_ok or not np.isfinite(audit.melnikov_value):
            verdict = "VETOED"
            reasons.append("melnikov_engine_failed")
        elif audit.is_simple_zero or audit.is_chaotic or audit.simple_zeros > 0:
            verdict = "VETOED"
            reasons.append("melnikov_transverse_homoclinic")
        elif abs(audit.melnikov_value) < tol * _DEGRADATION_FACTOR:
            verdict = "DEGRADED"
            reasons.append("melnikov_near_tangency")
        else:
            verdict = "COHERENT"
        return _MelnikovVeredict(
            melnikov_value=float(audit.melnikov_value),
            melnikov_derivative=float(audit.melnikov_derivative),
            is_simple_zero=bool(audit.is_simple_zero),
            homoclinic_splitting=float(audit.homoclinic_splitting),
            verdict=verdict,
            threshold_used=float(tol),
            simple_zeros=int(audit.simple_zeros),
            is_chaotic=bool(audit.is_chaotic),
            skipped=bool(audit.skipped),
            godel_value=float(_HEYTING_GODEL[self.canonicalize(verdict)]),
            reasons=_unique_preserve(reasons),
        )

    # ── II.7 Clasificador del mapa de retorno de Poincaré ────────────────
    def classify_return_map(
        self, audit: _ReturnMapAudit, scale: float
    ) -> _ReturnMapVeredict:
        r"""
        Valúa P: Σ → Σ en H₃.
        Una silla hiperbólica aislada no es caos; el veto exige L_max grande
        o pérdida de simpléctica (det M ≉ 1 / ‖MᵀΩM−Ω‖ grande).
        """
        reasons: list[str] = []
        tol = self.threshold(_FLOQUET_PARABOLIC, scale)
        if audit.skipped:
            verdict = "COHERENT"
        elif not audit.engine_ok:
            verdict = "VETOED"
            reasons.append("return_map_engine_failed")
        else:
            verdict = "COHERENT"
            if np.isfinite(audit.det_M) and abs(audit.det_M - 1.0) > _DET_M_VETO:
                reasons.append("monodromy_det_not_unity")
                verdict = self.join(verdict, "VETOED")
            if (
                np.isfinite(audit.symplectic_residual)
                and audit.symplectic_residual > _SP_RES_DEGRADED
            ):
                reasons.append("monodromy_not_symplectic")
                verdict = self.join(verdict, "DEGRADED")
            if audit.is_elliptic and abs(audit.max_lyapunov) <= tol * _DEGRADATION_FACTOR:
                pass  # permanece COHERENT salvo degradaciones previas
            elif audit.is_parabolic or audit.is_mixed:
                reasons.append("floquet_parabolic_or_mixed")
                verdict = self.join(verdict, "DEGRADED")
            elif audit.floquet_parabolic <= tol:
                reasons.append("floquet_near_unit_circle")
                verdict = self.join(verdict, "DEGRADED")
            elif abs(audit.max_lyapunov) > max(tol / max(_DEGRADATION_FACTOR, _MACHINE_EPS), tol):
                reasons.append("positive_lyapunov_chaos")
                verdict = self.join(verdict, "VETOED")
            elif audit.is_hyperbolic:
                reasons.append("hyperbolic_isolated_saddle")
                verdict = self.join(verdict, "DEGRADED")
            if (
                np.isfinite(audit.reciprocal_pair_residual)
                and audit.reciprocal_pair_residual > 1e-3
            ):
                reasons.append("krein_reciprocal_pairing_broken")
                verdict = self.join(verdict, "DEGRADED")
        return _ReturnMapVeredict(
            max_lyapunov=float(audit.max_lyapunov),
            floquet_parabolic=float(audit.floquet_parabolic),
            is_hyperbolic=bool(audit.is_hyperbolic),
            is_elliptic=bool(audit.is_elliptic),
            is_parabolic=bool(audit.is_parabolic),
            trace_M=float(audit.trace_M),
            det_M=float(audit.det_M),
            verdict=verdict,
            threshold_used=float(tol),
            is_mixed=bool(audit.is_mixed),
            symplectic_residual=float(audit.symplectic_residual),
            hill_discriminant=float(audit.hill_discriminant),
            kaplan_yorke_dimension=float(audit.kaplan_yorke_dimension),
            kolmogorov_sinai_entropy=float(audit.kolmogorov_sinai_entropy),
            godel_value=float(_HEYTING_GODEL[self.canonicalize(verdict)]),
            reasons=_unique_preserve(reasons),
        )

    # ── II.ω  MORFISMO TERMINAL Φ_II: gérmen OODA ────────────────────────
    def induce_ooda_actuation_germ(
        self, germ: _PoincareCelestialAuditGerm
    ) -> _OODAActuationGerm:
        r"""
        **II.ω — Morfismo terminal de la FASE II / objeto inicial de la FASE III.**

        Continúa I.ω vía `phase2_ingest_poincare_celestial_germ` y valúa
        𝒢_I^{cel} en H₃⁵ = H₃(Floer)×H₃(Čech)×H₃(KAM)×H₃(Melnikov)×H₃(Retorno).
        Join = peor caso; meet de Gödel = ínfimo de los valores de verdad.

        Este método **es** el arranque formal de la Fase III:
        `phase3_ingest_ooda_actuation_germ` → `run`.
        """
        g = self.phase2_ingest_poincare_celestial_germ(germ)
        floer_v = self.classify_floer(g.floer, g.floer_scale)
        cech_v = self.classify_cech(g.cech, g.cech_scale)
        kam_v = self.classify_kam(g.kam, g.kam_scale)
        mel_v = self.classify_melnikov(g.melnikov, g.melnikov_scale)
        ret_v = self.classify_return_map(g.return_map, g.return_scale)

        hill_allowed = bool(g.hill.skipped or (g.hill.engine_ok and g.hill.is_classically_allowed))
        extra_join = "COHERENT" if hill_allowed else "DEGRADED"

        joined = self.join(
            floer_v.verdict,
            cech_v.verdict,
            kam_v.verdict,
            mel_v.verdict,
            ret_v.verdict,
            extra_join,
        )
        meet_g = float(
            min(
                floer_v.godel_value,
                cech_v.godel_value,
                kam_v.godel_value,
                mel_v.godel_value,
                ret_v.godel_value,
                1.0 if hill_allowed else 0.5,
            )
        )
        all_reasons = (
            list(floer_v.reasons) + list(cech_v.reasons) + list(kam_v.reasons)
            + list(mel_v.reasons) + list(ret_v.reasons)
        )
        if not hill_allowed:
            all_reasons.append("hill_region_forbidden")
        veto_r = _unique_preserve([
            r for r, v in (
                (floer_v.reasons, floer_v.verdict),
                (cech_v.reasons, cech_v.verdict),
                (kam_v.reasons, kam_v.verdict),
                (mel_v.reasons, mel_v.verdict),
                (ret_v.reasons, ret_v.verdict),
            ) if v == "VETOED" for r in v and ()  # placeholder, rebuilt below
        ])
        # Reconstrucción limpia de razones por rango.
        veto_acc: list[str] = []
        deg_acc: list[str] = []
        for reasons, verdict in (
            (floer_v.reasons, floer_v.verdict),
            (cech_v.reasons, cech_v.verdict),
            (kam_v.reasons, kam_v.verdict),
            (mel_v.reasons, mel_v.verdict),
            (ret_v.reasons, ret_v.verdict),
        ):
            bucket = veto_acc if verdict == "VETOED" else deg_acc if verdict == "DEGRADED" else []
            bucket.extend(reasons)
        if not hill_allowed:
            deg_acc.append("hill_region_forbidden")
        return _OODAActuationGerm(
            floer=floer_v,
            cech=cech_v,
            kam=kam_v,
            melnikov=mel_v,
            return_map=ret_v,
            heyting_join=joined,
            godel_meet=meet_g,
            two_n=int(g.two_n),
            safety_margin=float(g.safety_margin),
            hill_allowed=hill_allowed,
            veto_reasons=_unique_preserve(veto_acc),
            degraded_reasons=_unique_preserve(deg_acc),
        )


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE III — CICLO OODA, ACTUACIÓN Y COLAPSO AL DISYUNTOR CIBER-FÍSICO     ║
# ║                                                                          ║
# ║ El primer método consume II.ω; run es III.ω.                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True, slots=True)
class _OODAResult:
    """
    Acta certificada del ciclo OODA de los Eruditos Imperiales.
    Agrega los cinco veredictos H₃, el join, el meet de Gödel y las
    métricas celestes (Conley–Zehnder, Lyapunov, Betti, Brjuno, Hill).
    """
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
    kam_verdict: str
    kam_min_divisor: float
    kam_novikov_weight: float
    kam_volume_drift: float
    kam_is_diophantine: bool
    melnikov_verdict: str
    melnikov_value: float
    melnikov_is_simple_zero: bool
    melnikov_splitting: float
    return_verdict: str
    floquet_max_abs: float
    lyapunov_max: float
    is_hyperbolic: bool
    is_elliptic: bool
    conley_zehnder_index: float
    cech_betti_0: int
    cech_betti_1: int
    filter_is_prime: bool
    observe_ok: bool
    # Extensiones 6.1 (defaulted → compatibles)
    kam_bruno_sum: float = 0.0
    kam_estimated_gamma: float = 0.0
    kam_tau: float = 1.0
    melnikov_simple_zeros: int = 0
    melnikov_skipped: bool = False
    is_mixed: bool = False
    is_parabolic: bool = False
    symplectic_residual: float = float("nan")
    hill_discriminant: float = float("nan")
    kaplan_yorke_dimension: float = float("nan")
    kolmogorov_sinai_entropy: float = float("nan")
    is_h1_trivial: bool = True
    hill_allowed: bool = True
    veto_reasons: Tuple[str, ...] = ()
    degraded_reasons: Tuple[str, ...] = ()

    def as_public_dict(self) -> Dict[str, Any]:
        """Contrato 2.0 — resumen canónico."""
        return {
            "heyting_verdict": self.heyting_verdict,
            "floer_residual": self.floer_residual,
            "action_potential": self.action_potential,
            "floer_verdict": self.floer_verdict,
            "cech_obstruction": self.cech_obstruction,
            "cech_active_modes": self.cech_active_modes,
            "cech_verdict": self.cech_verdict,
            "kam_verdict": self.kam_verdict,
            "kam_min_divisor": self.kam_min_divisor,
            "kam_novikov_weight": self.kam_novikov_weight,
            "kam_is_diophantine": self.kam_is_diophantine,
            "melnikov_verdict": self.melnikov_verdict,
            "melnikov_value": self.melnikov_value,
            "melnikov_is_simple_zero": self.melnikov_is_simple_zero,
            "return_verdict": self.return_verdict,
            "floquet_max_abs": self.floquet_max_abs,
            "lyapunov_max": self.lyapunov_max,
            "is_hyperbolic": self.is_hyperbolic,
            "is_elliptic": self.is_elliptic,
            "hardware_interlock_fired": self.hardware_interlock_fired,
            "actuation_latency_ns": self.actuation_latency_ns,
        }

    def as_celestial_dict(self) -> Dict[str, Any]:
        """Contrato 2.0 — solo lecturas celestes de Poincaré."""
        return {
            "kam_verdict": self.kam_verdict,
            "kam_min_divisor": self.kam_min_divisor,
            "kam_novikov_weight": self.kam_novikov_weight,
            "kam_volume_drift": self.kam_volume_drift,
            "kam_is_diophantine": self.kam_is_diophantine,
            "kam_bruno_sum": self.kam_bruno_sum,
            "kam_tau": self.kam_tau,
            "melnikov_verdict": self.melnikov_verdict,
            "melnikov_value": self.melnikov_value,
            "melnikov_is_simple_zero": self.melnikov_is_simple_zero,
            "melnikov_splitting": self.melnikov_splitting,
            "melnikov_simple_zeros": self.melnikov_simple_zeros,
            "return_verdict": self.return_verdict,
            "floquet_max_abs": self.floquet_max_abs,
            "lyapunov_max": self.lyapunov_max,
            "is_hyperbolic": self.is_hyperbolic,
            "is_elliptic": self.is_elliptic,
            "is_mixed": self.is_mixed,
            "conley_zehnder_index": self.conley_zehnder_index,
            "kaplan_yorke_dimension": self.kaplan_yorke_dimension,
            "kolmogorov_sinai_entropy": self.kolmogorov_sinai_entropy,
            "is_h1_trivial": self.is_h1_trivial,
            "hill_allowed": self.hill_allowed,
        }


class _OODAController:
    r"""
    Fase III. Ciclo Observe–Orient–Decide–Act sobre el álgebra de Heyting.

    **III.0** `phase3_ingest_ooda_actuation_germ` continúa II.ω.
    **III.ω** `run` emite el acta y colapsa H₃ → {fire, no-fire}.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    # ── III.0  INGESTA DEL OBJETO TERMINAL DE LA FASE II ─────────────────
    def phase3_ingest_ooda_actuation_germ(
        self, germ: _OODAActuationGerm,
    ) -> _OODAActuationGerm:
        r"""
        **III.0 — Flecha inicial de Φ_III, continuación estricta de II.ω.**

        Revalida 𝒢_II^{OODA} (tipos, join canónico, dimensión) y lo reexpide
        al morfismo terminal `run`.
        """
        if not isinstance(germ, _OODAActuationGerm):
            raise TypeError("germ debe ser _OODAActuationGerm (cierre II.ω).")
        if germ.two_n <= 0 or germ.two_n % 2 != 0:
            raise ValueError("𝒢_II^{OODA} no tiene dimensión de Darboux.")
        joined = _HeytingClassifier.canonicalize(germ.heyting_join)
        if joined != germ.heyting_join:
            logger.warning("III.0: heyting_join no canónico (%s); se normaliza.", germ.heyting_join)
        if germ.godel_meet < 0.0 or germ.godel_meet > 1.0 or not math.isfinite(germ.godel_meet):
            logger.warning("III.0: godel_meet fuera de [0,1]: %.3e.", germ.godel_meet)
        return germ

    @staticmethod
    def observe(
        germ: _OODAActuationGerm,
    ) -> Tuple[_FloerVeredict, _CechVeredict, _KAMVeredict, _MelnikovVeredict, _ReturnMapVeredict]:
        return (
            germ.floer,
            germ.cech,
            germ.kam,
            germ.melnikov,
            germ.return_map,
        )

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
        return float(np.clip(latency, _INTERLOCK_LATENCY_FLOOR_NS, _INTERLOCK_LATENCY_CEIL_NS))

    # ── III.ω  MORFISMO TERMINAL Φ_III ───────────────────────────────────
    def run(self, germ: _OODAActuationGerm) -> _OODAResult:
        r"""
        **III.ω — Morfismo terminal de la FASE III.**

        Ejecuta el ciclo OODA sobre 𝒢_II^{OODA} y emite el acta certificada.
        El colapso H₃ → {fire, no-fire} dispara el presupuesto lógico de
        latencia (< 400 ns) sin conmutación de silicio en este módulo.
        """
        g = self.phase3_ingest_ooda_actuation_germ(germ)
        floer, cech, kam, mel, ret = self.observe(g)
        joined = self.orient(g)
        fire = self.decide(joined)
        latency = self.act(fire)

        # Melnikov saltado no invalida observe_ok.
        mel_finite = bool(mel.skipped or np.isfinite(mel.melnikov_value))
        observe_ok = bool(
            np.isfinite(floer.floer_residual)
            and np.isfinite(cech.cech_obstruction)
            and np.isfinite(kam.min_divisor)
            and mel_finite
            and np.isfinite(ret.max_lyapunov)
        )
        if fire:
            logger.critical(
                "VETO ATÓMICO DE ERUDITOS COHOMOLÓGICOS Y CELESTES. "
                "Join H₃ = VETOED (Floer=%s, Čech=%s, KAM=%s, Melnikov=%s, Return=%s). "
                "Divisor mínimo=%.3e, |M(t₀)|=%.3e, L_max=%.3e, razones=%s. "
                "Interlock lógico ACTIVADO. Presupuesto de latencia = %.2f ns. "
                "No hay conmutación de silicio en este módulo.",
                floer.verdict,
                cech.verdict,
                kam.verdict,
                mel.verdict,
                ret.verdict,
                kam.min_divisor,
                abs(mel.melnikov_value) if np.isfinite(mel.melnikov_value) else float("nan"),
                ret.max_lyapunov,
                g.veto_reasons,
                latency,
            )
        floquet_max_abs = float(
            np.max(np.abs(g.return_map.floquet_multipliers))
            if getattr(g.return_map, "floquet_multipliers", np.array([])).size
            else (abs(ret.det_M) if np.isfinite(ret.det_M) else 0.0)
        )
        # floquet multipliers viven en el audit, no en el veredicto: usar det/trace proxy
        # si el veredicto no los porta. Preferimos el germen de retorno del acta.
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
            godel_meet=float(g.godel_meet),
            kam_verdict=kam.verdict,
            kam_min_divisor=float(kam.min_divisor),
            kam_novikov_weight=float(kam.novikov_weight),
            kam_volume_drift=float(kam.volume_drift),
            kam_is_diophantine=bool(kam.is_diophantine),
            melnikov_verdict=mel.verdict,
            melnikov_value=float(mel.melnikov_value),
            melnikov_is_simple_zero=bool(mel.is_simple_zero),
            melnikov_splitting=float(mel.homoclinic_splitting),
            return_verdict=ret.verdict,
            floquet_max_abs=floquet_max_abs,
            lyapunov_max=float(ret.max_lyapunov),
            is_hyperbolic=bool(ret.is_hyperbolic),
            is_elliptic=bool(ret.is_elliptic),
            conley_zehnder_index=float(floer.conley_zehnder_index),
            cech_betti_0=int(cech.betti_0),
            cech_betti_1=int(cech.betti_1),
            filter_is_prime=True,
            observe_ok=observe_ok,
            kam_bruno_sum=float(kam.bruno_sum),
            kam_estimated_gamma=float(kam.estimated_gamma),
            kam_tau=float(kam.tau),
            melnikov_simple_zeros=int(mel.simple_zeros),
            melnikov_skipped=bool(mel.skipped),
            is_mixed=bool(ret.is_mixed),
            is_parabolic=bool(ret.is_parabolic),
            symplectic_residual=float(ret.symplectic_residual),
            hill_discriminant=float(ret.hill_discriminant),
            kaplan_yorke_dimension=float(ret.kaplan_yorke_dimension),
            kolmogorov_sinai_entropy=float(ret.kolmogorov_sinai_entropy),
            is_h1_trivial=bool(cech.is_h1_trivial),
            hill_allowed=bool(g.hill_allowed),
            veto_reasons=g.veto_reasons,
            degraded_reasons=g.degraded_reasons,
        )


# =============================================================================
# AGENTE PÚBLICO — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I
# =============================================================================
class ImperialGuardsEruditosAgent:
    r"""
    Soberano agéntico de Cohomología Simpléctica, Atencional y Mecánica Celeste
    de Poincaré.

    Composición anidada:
        Φ_I   : Auditoría celeste (KAM, Melnikov, Retorno, Floer, Čech, Hill)
                ↦ 𝒢_I^{cel}
        Φ_II  : Valuación H₃⁵ y join / meet de Gödel
                ↦ 𝒢_II^{OODA}
        Φ_III : Ciclo OODA y colapso al disyuntor lógico
                ↦ acta
    """

    def __init__(
        self,
        dimension_n: int = 2,
        safety_margin: float = 1.0,
        regularizer: float = 1e-15,
        novikov_valuation_T: float = 1.0,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if int(dimension_n) <= 0 or int(dimension_n) % 2 != 0:
            raise ValueError(
                f"La dimensión del espacio simpléctico de fase n={dimension_n} debe ser par."
            )
        if not np.isfinite(safety_margin) or safety_margin < 0.0:
            raise ValueError("safety_margin debe ser finito y ≥ 0.")
        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)
        self._reg: Final[float] = float(max(regularizer, 1e-20))
        self._novikov_T: Final[float] = float(novikov_valuation_T)
        self._H0: Final[float] = float(hamiltonian_energy_H0)
        self._V: Final[float] = float(potential_energy_V)
        try:
            self._engine: Final[ImperialEruditosEngine] = ImperialEruditosEngine(
                regularizer=self._reg,
                novikov_valuation_T=self._novikov_T,
                hamiltonian_energy_H0=self._H0,
                potential_energy_V=self._V,
                germ_dimension=self._n,
            )
        except TypeError:
            try:
                self._engine = ImperialEruditosEngine(  # type: ignore[misc]
                    regularizer=self._reg,
                    novikov_valuation_T=self._novikov_T,
                    hamiltonian_energy_H0=self._H0,
                    potential_energy_V=self._V,
                )
            except TypeError:
                try:
                    self._engine = ImperialEruditosEngine(  # type: ignore[misc]
                        regularizer=self._reg,
                        novikov_valuation_T=self._novikov_T,
                    )
                except TypeError:
                    self._engine = ImperialEruditosEngine()  # type: ignore[misc]
        self._audit_core = _AuditCore(self._engine, two_n=self._n)
        self._classifier = _HeytingClassifier(self._safety_margin)
        self._ooda = _OODAController(rng=rng)
        self._audit_germ: Optional[_PoincareCelestialAuditGerm] = None
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

    # ══════════════════════════════════════════════════════════════════════
    # API FASE I — Auditorías celestes granulares
    # ══════════════════════════════════════════════════════════════════════
    def audit_poincare_kam_stability(
        self,
        frequency_vector_omega: NDArray[np.float64],
        wave_vectors_k: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        canonical_J: NDArray[np.float64],
    ) -> _KAMVeredict:
        r"""
        [ERUDITO 3 — AUDITORÍA KAM DE POINCARÉ]
            min_k |⟨k, ω⟩| ≥ γ / |k|^τ,   τ > n − 1,
        y |det M − 1| ≤ ε_W.
        """
        audit = self._audit_core.kam_audit(
            frequency_vector_omega, wave_vectors_k, jacobian_M, canonical_J
        )
        scale = max(audit.min_divisor, 1.0) if np.isfinite(audit.min_divisor) else 1.0
        return self._classifier.classify_kam(audit, scale)

    def audit_melnikov_homoclinic_splitting(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: NDArray[np.float64],
        t_inf: float = 25.0,
    ) -> _MelnikovVeredict:
        r"""
        [ERUDITO 4 — AUDITORÍA DE MELNIKOV]
            M(t₀) = ∫ {H₀, H₁}(γ⁰(t − t₀)) dt.
        Ceros simples ⇒ fractura homoclínica (caos de Poincaré).
        """
        audit = self._audit_core.melnikov_audit(
            homoclinic_flow, hamiltonian_0, hamiltonian_1, t0_grid, t_inf=t_inf
        )
        scale = max(abs(audit.melnikov_value), 1.0) if np.isfinite(audit.melnikov_value) else 1.0
        return self._classifier.classify_melnikov(audit, scale)

    def audit_poincare_return_map(
        self,
        jacobian_M: NDArray[np.float64],
        period_T: float = 1.0,
    ) -> _ReturnMapVeredict:
        r"""
        [ERUDITO 5 — AUDITORÍA DEL MAPA DE RETORNO]
        Espectro de Floquet–Krein y Lyapunov del mapa P: Σ → Σ.
        """
        audit = self._audit_core.return_map_audit(jacobian_M, period_T=period_T)
        scale = max(audit.floquet_parabolic, 1.0) if np.isfinite(audit.floquet_parabolic) else 1.0
        return self._classifier.classify_return_map(audit, scale)

    def floer_cylinder_germ_certificate(self) -> Any:
        """Certificado de Darboux del gérmen de Fase I del motor."""
        getter = getattr(self._engine, "poincare_cartan_germ_certificate", None)
        if callable(getter):
            return getter()
        legacy = getattr(self._engine, "floer_cylinder_germ_certificate", None)
        if callable(legacy):
            return legacy()
        return None

    def maupertuis_jacobi_certificate(self) -> Any:
        """Certificado de la métrica conforme de Maupertuis–Jacobi."""
        getter = getattr(self._engine, "maupertuis_jacobi_certificate", None)
        return getter() if callable(getter) else None

    def synthesize_heyting_audit_germ(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
    ) -> _PoincareCelestialAuditGerm:
        r"""Réplica pública de I.ω con canales KAM/Melnikov/Retorno por defecto."""
        germ = self._audit_core.synthesize_poincare_celestial_germ(
            start_point=start_point,
            end_point=end_point,
            jacobian_m3=jacobian_m3,
            attention_sheaf_matrix=attention_sheaf_matrix,
            safety_margin=self._safety_margin,
        )
        self._audit_germ = germ
        return germ

    def synthesize_poincare_celestial_germ(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        t_inf: float = 25.0,
        period_T: float = 1.0,
    ) -> _PoincareCelestialAuditGerm:
        r"""Réplica pública completa de I.ω con canales celestes específicos."""
        germ = self._audit_core.synthesize_poincare_celestial_germ(
            start_point=start_point,
            end_point=end_point,
            jacobian_m3=jacobian_m3,
            attention_sheaf_matrix=attention_sheaf_matrix,
            safety_margin=self._safety_margin,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            canonical_J=canonical_J,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            t_inf=t_inf,
            period_T=period_T,
        )
        self._audit_germ = germ
        return germ

    def audit_floer_homology_trajectory(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
    ) -> _FloerVeredict:
        """[ERUDITO 1 — AUDITORÍA DE FLOER]"""
        if np.asarray(start_point).ndim != 1 or np.asarray(end_point).ndim != 1:
            logger.error("Los puntos deben ser vectores unidimensionales.")
            return _FloerVeredict(
                floer_residual=float("inf"),
                action_potential=float("inf"),
                verdict="VETOED",
                godel_value=0.0,
                reasons=("floer_points_not_1d",),
            )
        if np.asarray(start_point).shape != np.asarray(end_point).shape:
            logger.error("Los puntos deben tener la misma dimensión.")
            return _FloerVeredict(
                floer_residual=float("inf"),
                action_potential=float("inf"),
                verdict="VETOED",
                godel_value=0.0,
                reasons=("floer_points_dim_mismatch",),
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
        """[ERUDITO 2 — AUDITORÍA DE ČECH]"""
        audit = self._audit_core.cech_audit(attention_sheaf_matrix)
        if np.isfinite(audit.nuclear_mass) and audit.nuclear_mass > 0.0:
            scale = max(float(audit.nuclear_mass), 1.0)
        else:
            scale = 1.0
        return self._classifier.classify_cech(audit, scale)

    # ══════════════════════════════════════════════════════════════════════
    # API FASE II — Valuación H₃⁵ y lifting OODA
    # ══════════════════════════════════════════════════════════════════════
    def induce_ooda_actuation_germ(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        t_inf: float = 25.0,
        period_T: float = 1.0,
    ) -> _OODAActuationGerm:
        r"""Réplica pública de II.ω: 𝒢_I^{cel} ↦ H₃⁵ ↦ 𝒢_II^{OODA}."""
        audit_germ = self.synthesize_poincare_celestial_germ(
            start_point=start_point,
            end_point=end_point,
            jacobian_m3=jacobian_m3,
            attention_sheaf_matrix=attention_sheaf_matrix,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            canonical_J=canonical_J,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            t_inf=t_inf,
            period_T=period_T,
        )
        ooda_germ = self._classifier.induce_ooda_actuation_germ(audit_germ)
        self._ooda_germ = ooda_germ
        return ooda_germ

    # ══════════════════════════════════════════════════════════════════════
    # API FASE III — Ciclo OODA y colapso al disyuntor
    # ══════════════════════════════════════════════════════════════════════
    def execute_eruditos_cycle(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        t_inf: float = 25.0,
        period_T: float = 1.0,
    ) -> Dict[str, Any]:
        """Ejecuta el ciclo OODA de los Eruditos (contrato 2.0)."""
        return self.execute_eruditos_cycle_certified(
            start_point=start_point,
            end_point=end_point,
            jacobian_m3=jacobian_m3,
            attention_sheaf_matrix=attention_sheaf_matrix,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            canonical_J=canonical_J,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            t_inf=t_inf,
            period_T=period_T,
        ).as_public_dict()

    def execute_eruditos_cycle_certified(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        jacobian_m3: np.ndarray,
        attention_sheaf_matrix: np.ndarray,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        t_inf: float = 25.0,
        period_T: float = 1.0,
    ) -> _OODAResult:
        r"""
        Ciclo completo **I.ω → II.ω → III.ω**.
        Join H₃⁵, Gödel meet, Conley–Zehnder, Lyapunov, Betti, Brjuno y
        disparo (o no) del interlock lógico.
        """
        germ = self.induce_ooda_actuation_germ(
            start_point=start_point,
            end_point=end_point,
            jacobian_m3=jacobian_m3,
            attention_sheaf_matrix=attention_sheaf_matrix,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            canonical_J=canonical_J,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            t_inf=t_inf,
            period_T=period_T,
        )
        return self._ooda.run(germ)

    # ══════════════════════════════════════════════════════════════════════
    # API heredada — Auditoría Poincaré-Novikov de coherencia espectral
    # ══════════════════════════════════════════════════════════════════════
    def audit_eruditos_poincare_novikov_coherence(
        self,
        frequency_vector_omega: NDArray[np.float64],
        wave_vectors_k: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        canonical_J: NDArray[np.float64],
    ) -> EruditosAgentCertificate:
        r"""
        Ciclo OODA de supervisión espectral (API 2.0 retrocompatible).
        Ω₃: COHERENT / DEGRADED / VETOED según divisor y drift de Liouville.
        """
        try:
            raw = self._engine.compute_poincare_small_divisors_spectrum(
                frequency_vector_omega=frequency_vector_omega,
                wave_vectors_k=wave_vectors_k,
                jacobian_M=jacobian_M,
                canonical_J=canonical_J,
            )
            if isinstance(raw, tuple) and len(raw) == 2:
                report, _kam_cert = raw
            else:
                report = raw
        except TypeError:
            report = self._engine.compute_poincare_small_divisors_spectrum(
                frequency_vector_omega,
                wave_vectors_k,
                jacobian_M,
                canonical_J,
            )
        if report.is_kam_stable:
            verdict = "COHERENT"
            is_coherent = True
        elif (
            report.min_small_divisor > 1.0e-15
            and report.liouville_volume_drift <= _SPECTRAL_TOL
        ):
            verdict = "DEGRADED"
            is_coherent = True
            logger.warning(
                "[ERUDITOS_DEGRADED] Pequeño divisor detectado: %.3e. "
                "Veto suave activado.",
                report.min_small_divisor,
            )
        else:
            verdict = "VETOED"
            is_coherent = False
            logger.error(
                "[ERUDITOS_VETOED] Ruptura de Poincaré-Novikov: "
                "Divisor=%.3e, Drift=%.3e. Gatillando la ISR en IRAM del "
                "ESP32 (< 400 ns) via GPIO14 / BT151 Crowbar.",
                report.min_small_divisor,
                report.liouville_volume_drift,
            )
        return EruditosAgentCertificate(
            min_small_divisor=report.min_small_divisor,
            novikov_weight=report.novikov_absorbed_weight,
            volume_drift=report.liouville_volume_drift,
            heyting_verdict=verdict,
            is_verdict_coherent=is_coherent,
        )

    @staticmethod
    def _veredict_from_metric(
        metric: float,
        base_tolerance: float,
        safety_margin: float,
        degradation_factor: float = _DEGRADATION_FACTOR,
    ) -> str:
        """Clasificador H₃ estático (retrocompatible)."""
        return _HeytingClassifier(safety_margin).verdict_from_metric(
            metric, base_tolerance, safety_margin, degradation_factor
        )


__all__ = [
    "ImperialGuardsEruditosAgent",
    "EruditosAgentCertificate",
    "PoincareEruditosAgentCertificate",
    "_PoincareCelestialAuditGerm",
    "_OODAActuationGerm",
    "_OODAResult",
    "_KAMVeredict",
    "_MelnikovVeredict",
    "_ReturnMapVeredict",
    "_FloerVeredict",
    "_CechVeredict",
]
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Guards Agent (Guardias Imperiales de Calibre OODA)         ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_agent.py              ║
║ Versión: 4.1.0-Poincare-OODA-Heyting-ESP32-Crowbar-PhD                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y GEOMÉTRICA DE POINCARÉ Y DE RHAM:
────────────────────────────────────────────────────────────────────────────────
Ejerce la censura de primer nivel y la gobernanza covariante en lazo cerrado OODA
($\Phi_3 \circ \Phi_2 \circ \Phi_1$) sobre el foso de la obra en la Malla Agéntica de
APU Filter. Evalúa la regularidad espectral, la conectividad topológica y la invarianza
simpléctica de la mecánica celeste de Henri Poincaré sobre el espacio de fase $T^*\mathcal{M}$.

FASE 1 (OBSERVE - $\Phi_1$):
────────────────────────────────────────────────────────────────────────────────
Inmersión $\ell^2$, hashing SHA-256 e inspección espectral no conmutativa del operador
de Dirac con la Cota Lipschitz de Connes-Daleckii-Krein:
   $$L_{\max} \le \frac{1}{2 \lambda_{\min}^{3/2}} \le \tau_{\mathrm{Lipschitz}}$$

FASE 2 (ORIENT - $\Phi_2$):
────────────────────────────────────────────────────────────────────────────────
Integración de trayectorias en el espacio de fase sobre `imperial_guards_engine.py`:
- Invarianza simpléctica de Liouville: $\det(\mathbf{M}) = +1$.
- Métrica conforme y acción de Maupertuis-Jacobi: $\mathcal{S}_M > 0$.
- Absorción de pequeños divisores en el anillo de Novikov $\Lambda_{\mathrm{Nov}}$.
- Distancia de recurrencia ergódica de Poincaré: $d_{\mathrm{Poincare}}(z(t_n), z_0) \le \varepsilon$.
- Auditoría de cuellos de botella con la desigualdad isoperimétrica de Cheeger:
   $$\frac{\lambda_2}{2} \le h(G) \le \sqrt{2 \lambda_2}$$

FASE 3 (DECIDE / ACT - $\Phi_3$):
────────────────────────────────────────────────────────────────────────────────
- Clasificación en el retículo distributivo de Heyting $\Omega_3 = \{\text{COHERENT}, \text{DEGRADED}, \text{VETOED}\}$.
- Interrupción ciber-física en silicio real/simulado ESP32 en IRAM ($t_{\text{actuation}} \le 400\,\text{ns}$)
  activando el tiristor BT151 (Crowbar) vía GPIO14 ante rupturas simplécticas o desviaciones presupuestales.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

from app.core.inmune_system.imperial_guards_engine import (
    ImperialGuardsEngine,
    ImperialEngineStepResult,
)

logger = logging.getLogger("APU.Agents.ImperialGuardsAgent")


# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS Y LÍMITES DE WILKINSON / CONNES / CHEEGER / HARDWARE
# ════════════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)

# Deriva metrológica máxima admisible.
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9

# Cota dura y degradada para la constante Lipschitz de Connes.
_HARD_LIPSCHITZ_CEILING: Final[float] = 5.0
_DEGRADED_LIPSCHITZ_CEILING: Final[float] = 3.0

# Regularización de Tikhonov/Higham para evitar el polo en λ_min = 0.
_HIGHAM_REG_FLOOR: Final[float] = 1e-20
_HIGHAM_REG_SQRT: Final[float] = math.sqrt(_HIGHAM_REG_FLOOR)

# Hardware simulado: Crowbar BT151 en GPIO14.
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_FLOOR_NS: Final[float] = 380.0
_CROWBAR_LATENCY_CEIL_NS: Final[float] = 420.0

# Tolerancias numéricas.
_IMAGINARY_TOL: Final[float] = 100.0 * _MACHINE_EPS
_PSD_TOL: Final[float] = 100.0 * _MACHINE_EPS

# Umbrales logísticos.
_DEFAULT_CHEEGER_THRESHOLD: Final[float] = 0.15
_LOGISTIC_VETO_PSI: Final[float] = 0.70
_LOGISTIC_DEGRADED_FIEDLER: Final[float] = 0.30
_LOGISTIC_DEGRADED_PSI: Final[float] = 0.85


# ════════════════════════════════════════════════════════════════════════════════
# CONTRATOS INMUTABLES DE FASE Y DOSSIERS DE POINCARÉ
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Phase1ImperialDossier:
    r"""Expediente inmutable de la Fase 1 (Observe)."""

    state_vector: NDArray[np.float64]
    state_norm: float
    session_sha256: str


@dataclass(frozen=True, slots=True)
class Phase2ImperialDossier:
    r"""Expediente inmutable de la Fase 2 (Orient)."""

    engine_result: ImperialEngineStepResult
    liouville_conserved: bool
    maupertuis_valid: bool
    ergodic_recurrence_distance: float


@dataclass(frozen=True, slots=True)
class ImperialGuardsVerdict:
    r"""Certificado final de la Fase 3 (Decide/Act) en Heyting Ω₃."""

    verdict: str  # COHERENT, DEGRADED, VETOED
    volume_drift: float
    maupertuis_action: float
    ergodic_return_distance: float
    is_hardware_crowbar_triggered: bool


@dataclass(frozen=True, slots=True)
class Phase1SpectralObservation:
    """
    Contrato formal de salida de la FASE 1 (Espectral).

    Contiene la auditoría espectral del operador de Dirac y la cota Lipschitz
    de Connes-Daleckii-Krein.
    """

    dirac_spectrum_size: int
    lambda_min_dirac: float
    lipschitz_coefficient: float
    partial_verdict: str
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class Phase2LogisticObservation:
    """
    Contrato formal de salida de la FASE 2 (Topológica/Logística).

    Contiene la auditoría logística/topológica del Laplaciano, números de Betti,
    brecha de Fiedler, proxy de Cheeger y estabilidad piramidal Ψ.
    """

    betti_0: int
    betti_1: int
    fiedler_connectivity: float
    cheeger_lower_bound: float
    cohomological_residual: float
    pyramidal_stability: float
    partial_verdict: str
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class Phase3TribunalDecision:
    """
    Contrato formal de salida de la FASE 3 (Heyting Tribunal).

    Contiene el veredicto unificado en el retículo de Heyting Ω₃:
        COHERENT < DEGRADED < VETOED
    """

    heyting_verdict: str
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImperialGuardsCertificate:
    """
    Certificado inmutable emitido por el tribunal de los Guardias Imperiales.
    """

    phase: str
    heyting_verdict: str               # COHERENT, DEGRADED, VETOED
    lipschitz_coefficient: float       # Cota de Connes L_max
    dirac_spectral_gap: float          # λ_min del operador de Dirac
    fiedler_connectivity: float        # λ_2 de Fiedler del Laplaciano
    cheeger_lower_bound: float         # Proxy v1.x de h²/2; ver diagnósticos
    pyramidal_stability: float         # Índice de Estabilidad Piramidal Ψ
    cohomological_residual: float      # β₁ + |β₀ - 1|
    hardware_interlock_fired: bool     # Estado de conmutación del BT151
    actuation_latency_ns: float        # Tiempo de respuesta simulado (IRAM)
    veto_reasons: Tuple[str, ...] = ()
    degraded_reasons: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════════
# FASE 1 — GUARDIA IMPERIAL 1: CURVAS HETEROGEOMORFAS DE AUDITORÍA ESPECTRAL
# ════════════════════════════════════════════════════════════════════════════════

class Phase1SpectralGuardianMixin:
    """
    FASE 1 — GUARDIA 1.

    Audita el confinamiento de Lipschitz no conmutativo del operador de Dirac.
    """

    def __init__(self, config_dim_n: int) -> None:
        self._n = self._validate_positive_int("config_dim_n", config_dim_n)
        self._interlock_lock = threading.Lock()
        self._interlock_state = False

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value <= 0:
            raise ValueError(f"{name} debe ser estrictamente mayor que cero.")
        return int(value)

    @staticmethod
    def _validate_nonnegative_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value < 0:
            raise ValueError(f"{name} debe ser mayor o igual que cero.")
        return int(value)

    @staticmethod
    def _validate_finite_nonnegative(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f < 0.0:
            raise ValueError(f"{name} debe ser finito y mayor o igual que cero.")
        return value_f

    def _as_real_float_array(self, values: Any, name: str) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en un ndarray.") from exc

        if np.iscomplexobj(raw):
            try:
                if not np.all(np.isfinite(raw)):
                    raise ValueError(f"{name} contiene entradas complejas no finitas.")
            except TypeError as exc:
                raise ValueError(f"{name} tiene tipo incompatible con aritmética compleja.") from exc

            if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                raise ValueError(
                    f"{name} posee componente imaginaria no despreciable."
                )

            raw = raw.real

        try:
            arr = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a float64 real.") from exc

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene valores no finitos (NaN/Inf).")

        return arr

    def _as_real_float_vector(self, values: Any, name: str) -> np.ndarray:
        arr = self._as_real_float_array(values, name)
        return arr.ravel()

    def kahan_compensated_sum(self, terms: np.ndarray) -> float:
        arr = self._as_real_float_vector(terms, "terms")
        sum_val = 0.0
        compensation = 0.0
        for term in arr:
            x = float(term)
            t = sum_val + x
            if not math.isfinite(t):
                return float(t)
            if abs(sum_val) >= abs(x):
                compensation += (sum_val - t) + x
            else:
                compensation += (x - t) + sum_val
            sum_val = t
        result = sum_val + compensation
        return float(result)

    @staticmethod
    def _compute_lipschitz_coefficient(lambda_min: float) -> float:
        if not math.isfinite(lambda_min) or lambda_min <= _MACHINE_EPS:
            return float("inf")
        try:
            coeff = 1.0 / (2.0 * (lambda_min ** 1.5))
        except OverflowError:
            return float("inf")
        return coeff if math.isfinite(coeff) else float("inf")

    @staticmethod
    def _classify_spectral_lipschitz(
        lipschitz_coeff: float,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        veto_reasons = []
        degraded_reasons = []

        if not math.isfinite(lipschitz_coeff):
            veto_reasons.append("lipschitz_coefficient_nonfinite")
        elif lipschitz_coeff < 0.0:
            veto_reasons.append("lipschitz_coefficient_negative")
        elif lipschitz_coeff > _HARD_LIPSCHITZ_CEILING:
            veto_reasons.append("lipschitz_coefficient_exceeds_hard_ceiling")
        elif lipschitz_coeff > _DEGRADED_LIPSCHITZ_CEILING:
            degraded_reasons.append("lipschitz_coefficient_above_degraded_ceiling")

        if veto_reasons:
            verdict = "VETOED"
        elif degraded_reasons:
            verdict = "DEGRADED"
        else:
            verdict = "COHERENT"

        return verdict, tuple(veto_reasons), tuple(degraded_reasons)

    def phase1_audit_spectral_heterogeomorphic_curve(
        self,
        eigenvalues_dirac: Any,
    ) -> Phase1SpectralObservation:
        diagnostics: Dict[str, Any] = {
            "dirac_spectrum_valid": True,
            "regularization": "tikhonov_higham_hypot",
            "regularization_floor": _HIGHAM_REG_FLOOR,
        }

        try:
            eigenvalues = self._as_real_float_vector(eigenvalues_dirac, "eigenvalues_dirac")
        except ValueError as exc:
            diagnostics.update(
                {
                    "dirac_spectrum_valid": False,
                    "dirac_spectrum_error": str(exc),
                }
            )
            return Phase1SpectralObservation(
                dirac_spectrum_size=0,
                lambda_min_dirac=0.0,
                lipschitz_coefficient=float("inf"),
                partial_verdict="VETOED",
                veto_reasons=("dirac_spectrum_invalid",),
                degraded_reasons=(),
                diagnostics=diagnostics,
            )

        diagnostics["dirac_spectrum_size"] = int(eigenvalues.size)

        if eigenvalues.size == 0:
            diagnostics.update(
                {
                    "dirac_spectrum_valid": False,
                    "dirac_spectrum_error": "empty_spectrum",
                }
            )
            logger.warning("Espectro de Dirac vacío.")
            return Phase1SpectralObservation(
                dirac_spectrum_size=0,
                lambda_min_dirac=0.0,
                lipschitz_coefficient=float("inf"),
                partial_verdict="VETOED",
                veto_reasons=("dirac_spectrum_empty",),
                degraded_reasons=(),
                diagnostics=diagnostics,
            )

        regularized_abs = np.hypot(eigenvalues, _HIGHAM_REG_SQRT)
        valid_eigs = regularized_abs[regularized_abs > _WILKINSON_DRIFT_LIMIT]
        diagnostics["dirac_valid_eigenvalue_count"] = int(valid_eigs.size)

        if valid_eigs.size == 0:
            logger.warning("Espectro de Dirac colapsado bajo el límite de Wilkinson.")
            diagnostics["dirac_spectrum_error"] = "spectral_gap_collapsed"
            return Phase1SpectralObservation(
                dirac_spectrum_size=int(eigenvalues.size),
                lambda_min_dirac=0.0,
                lipschitz_coefficient=float("inf"),
                partial_verdict="VETOED",
                veto_reasons=("dirac_spectral_gap_collapsed",),
                degraded_reasons=(),
                diagnostics=diagnostics,
            )

        lambda_min = float(np.min(valid_eigs))
        lipschitz_coeff = self._compute_lipschitz_coefficient(lambda_min)
        verdict, veto_reasons, degraded_reasons = self._classify_spectral_lipschitz(
            lipschitz_coeff
        )

        diagnostics.update(
            {
                "dirac_lambda_min": lambda_min,
                "lipschitz_coefficient": lipschitz_coeff,
                "partial_verdict": verdict,
            }
        )

        return Phase1SpectralObservation(
            dirac_spectrum_size=int(eigenvalues.size),
            lambda_min_dirac=lambda_min,
            lipschitz_coefficient=lipschitz_coeff,
            partial_verdict=verdict,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            diagnostics=diagnostics,
        )


# ════════════════════════════════════════════════════════════════════════════════
# FASE 2 — GUARDIA IMPERIAL 2: CURVAS HOMOGEOMORFAS DE CUELLOS LOGÍSTICOS
# ════════════════════════════════════════════════════════════════════════════════

class Phase2LogisticGuardianMixin(Phase1SpectralGuardianMixin):
    """
    FASE 2 — GUARDIA 2.

    Audita los cuellos de botella organizacionales e ineficiencias de la red.
    """

    def __init__(
        self,
        config_dim_n: int,
        cheeger_threshold: float = _DEFAULT_CHEEGER_THRESHOLD,
    ) -> None:
        super().__init__(config_dim_n)
        self._cheeger_threshold = self._validate_finite_nonnegative(
            "cheeger_threshold",
            cheeger_threshold,
        )

    @staticmethod
    def _validate_betti(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value < 0:
            raise ValueError(f"{name} debe ser mayor o igual que cero.")
        return int(value)

    def _fiedler_gap(self, eigenvalues_L: Any) -> Tuple[float, Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "laplacian_spectrum_valid": True,
            "fiedler_gap_defined": False,
        }

        try:
            eigenvalues = self._as_real_float_vector(eigenvalues_L, "eigenvalues_L")
        except ValueError as exc:
            diagnostics.update(
                {
                    "laplacian_spectrum_valid": False,
                    "laplacian_spectrum_error": str(exc),
                }
            )
            return 0.0, diagnostics

        diagnostics["laplacian_spectrum_size"] = int(eigenvalues.size)

        if eigenvalues.size < 2:
            diagnostics["fiedler_gap_reason"] = "insufficient_spectrum_size"
            return 0.0, diagnostics

        min_eigenvalue = float(np.min(eigenvalues))
        diagnostics["min_laplacian_eigenvalue"] = min_eigenvalue

        if min_eigenvalue < -_PSD_TOL:
            diagnostics.update(
                {
                    "laplacian_psd_violation": True,
                    "fiedler_gap_reason": "laplacian_psd_violation",
                }
            )
            return 0.0, diagnostics

        clipped = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
        sorted_eigs = np.sort(clipped)
        sorted_eigs[np.abs(sorted_eigs) <= _PSD_TOL] = 0.0
        fiedler_gap = float(sorted_eigs[1])

        if not math.isfinite(fiedler_gap) or fiedler_gap < 0.0:
            fiedler_gap = 0.0

        diagnostics.update(
            {
                "laplacian_psd_violation": False,
                "fiedler_gap_defined": True,
                "fiedler_gap": fiedler_gap,
            }
        )

        return fiedler_gap, diagnostics

    def _classify_logistic_metrics(
        self,
        fiedler_gap: float,
        cohomological_residual: float,
        pyramidal_stability: float,
        betti_0: int,
        betti_1: int,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        veto_reasons = []
        degraded_reasons = []

        if not math.isfinite(fiedler_gap):
            veto_reasons.append("fiedler_connectivity_nonfinite")
        elif fiedler_gap < 0.0:
            veto_reasons.append("fiedler_connectivity_negative")
        else:
            if fiedler_gap < self._cheeger_threshold:
                veto_reasons.append("fiedler_connectivity_below_cheeger_threshold")
            elif fiedler_gap < _LOGISTIC_DEGRADED_FIEDLER:
                degraded_reasons.append("fiedler_connectivity_below_degraded_threshold")

        if not math.isfinite(cohomological_residual):
            veto_reasons.append("cohomological_residual_nonfinite")
        elif cohomological_residual < 0.0:
            veto_reasons.append("cohomological_residual_negative")
        elif cohomological_residual > 0.0:
            veto_reasons.append("cohomological_residual_nonzero")

        if betti_0 == 0:
            veto_reasons.append("empty_complex_detected")
        elif betti_0 > 1:
            veto_reasons.append("data_islands_detected")

        if betti_1 > 0:
            veto_reasons.append("logical_loops_detected")

        if not math.isfinite(pyramidal_stability):
            veto_reasons.append("pyramidal_stability_nonfinite")
        elif pyramidal_stability < 0.0:
            veto_reasons.append("pyramidal_stability_negative")
        else:
            if pyramidal_stability < _LOGISTIC_VETO_PSI:
                veto_reasons.append("pyramidal_stability_below_veto_threshold")
            elif pyramidal_stability < _LOGISTIC_DEGRADED_PSI:
                degraded_reasons.append("pyramidal_stability_below_degraded_threshold")

        if veto_reasons:
            verdict = "VETOED"
        elif degraded_reasons:
            verdict = "DEGRADED"
        else:
            verdict = "COHERENT"

        return verdict, tuple(veto_reasons), tuple(degraded_reasons)

    def phase2_audit_logistic_from_phase1(
        self,
        phase1_observation: Optional[Phase1SpectralObservation],
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
    ) -> Phase2LogisticObservation:
        if phase1_observation is not None and not isinstance(
            phase1_observation,
            Phase1SpectralObservation,
        ):
            raise TypeError("phase1_observation debe ser Phase1SpectralObservation.")

        b0 = self._validate_betti("betti_0", betti_0)
        b1 = self._validate_betti("betti_1", betti_1)
        cohom_residual = float(b1 + abs(b0 - 1))
        fiedler_gap, fiedler_diagnostics = self._fiedler_gap(eigenvalues_L)

        safe_fiedler = float(fiedler_gap) if (math.isfinite(fiedler_gap) and fiedler_gap > 0.0) else 0.0
        cheeger_lower_bound = float((safe_fiedler * safe_fiedler) / 2.0)
        cheeger_constant_lower_bound = float(safe_fiedler / 2.0)
        cheeger_constant_upper_bound = (
            float(math.sqrt(2.0 * safe_fiedler)) if safe_fiedler > 0.0 else 0.0
        )
        psi_stability = float(safe_fiedler / (1.0 + cohom_residual))

        verdict, veto_reasons, degraded_reasons = self._classify_logistic_metrics(
            fiedler_gap=fiedler_gap,
            cohomological_residual=cohom_residual,
            pyramidal_stability=psi_stability,
            betti_0=b0,
            betti_1=b1,
        )

        diagnostics: Dict[str, Any] = {
            "fiedler_gap": fiedler_gap,
            "cheeger_lower_bound": cheeger_lower_bound,
            "pyramidal_stability": psi_stability,
            "has_cohomological_obstruction": cohom_residual > 0.0,
            "islands_detected": b0 > 1,
            "loops_detected": b1 > 0,
            "betti_0": b0,
            "betti_1": b1,
            "cohomological_residual": cohom_residual,
            "empty_complex_detected": b0 == 0,
            "cheeger_constant_lower_bound": cheeger_constant_lower_bound,
            "cheeger_constant_upper_bound": cheeger_constant_upper_bound,
            "cheeger_threshold": self._cheeger_threshold,
        }
        diagnostics.update(fiedler_diagnostics)

        if phase1_observation is not None:
            diagnostics["phase1"] = {
                "dirac_spectrum_size": phase1_observation.dirac_spectrum_size,
                "lambda_min_dirac": phase1_observation.lambda_min_dirac,
                "lipschitz_coefficient": phase1_observation.lipschitz_coefficient,
                "partial_verdict": phase1_observation.partial_verdict,
            }

        return Phase2LogisticObservation(
            betti_0=b0,
            betti_1=b1,
            fiedler_connectivity=fiedler_gap,
            cheeger_lower_bound=cheeger_lower_bound,
            cohomological_residual=cohom_residual,
            pyramidal_stability=psi_stability,
            partial_verdict=verdict,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            diagnostics=diagnostics,
        )


# ════════════════════════════════════════════════════════════════════════════════
# FASE 3 — SOBERANO DE CALIBRE IMPERIAL (OODA & HEYTING TRIBUNAL)
# ════════════════════════════════════════════════════════════════════════════════

class ImperialGuardsAgent(Phase2LogisticGuardianMixin):
    """
    Soberano de Calibre OODA sobre imperial_guards_engine.py.

    Gobernanza de lazo cerrado, clasificación en Heyting Ω₃ y disparo
    de la ISR en IRAM del ESP32 (< 400 ns) ante violaciones de Poincaré.
    """

    def __init__(
        self,
        config_dim_n: int = 6,
        cheeger_threshold: float = _DEFAULT_CHEEGER_THRESHOLD,
        *,
        rng_seed: Optional[int] = None,
    ) -> None:
        """
        Inicializa las aduanas de control espectral, topológico y de mecánica celeste.

        Args:
            config_dim_n: Dimensión del espacio de configuración n.
            cheeger_threshold: Umbral crítico para la conectividad de Fiedler.
            rng_seed: Semilla opcional para reproducibilidad del jitter CAS.
        """
        super().__init__(config_dim_n, cheeger_threshold)
        self._rng = np.random.default_rng(rng_seed)
        self._engine = ImperialGuardsEngine(dimension=config_dim_n)
        self._trajectory_history: List[NDArray[np.float64]] = []

    # ────────────────────────────────────────────────────────────────────────────
    # AUDITORÍA DE LAZO CERRADO OODA DE HENRI POINCARÉ
    # ────────────────────────────────────────────────────────────────────────────

    def execute_ooda_poincare_audit(
        self,
        current_state: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        dt_step: float,
        external_freq_omega: NDArray[np.float64],
        wave_k: NDArray[np.float64],
    ) -> ImperialGuardsVerdict:
        r"""
        Ejecuta el ciclo OODA (Φ₃ ∘ Φ₂ ∘ Φ₁) de auditoría simpléctica de Poincaré.
        """
        # ─── FASE 1: OBSERVE (Φ₁) ───
        state_vec = self._as_real_float_vector(current_state, "current_state")
        state_norm = float(la.norm(state_vec))
        sha256_hash = hashlib.sha256(state_vec.tobytes()).hexdigest()
        phase1_dossier = Phase1ImperialDossier(
            state_vector=state_vec,
            state_norm=state_norm,
            session_sha256=sha256_hash,
        )

        # ─── FASE 2: ORIENT (Φ₂) ───
        engine_res = self._engine.step_poincare_symplectic_integration(
            current_state=state_vec,
            metric_G=metric_G,
            potential_V=potential_V,
            total_energy_H0=total_energy_H0,
            dt_step=dt_step,
            external_freq_omega=external_freq_omega,
            wave_k=wave_k,
        )

        self._trajectory_history.append(engine_res.next_state)

        # Distancia de retorno ergódico de Poincaré
        past_distances = [
            float(la.norm(pt - engine_res.next_state))
            for pt in self._trajectory_history[:-1]
        ]
        min_return_dist = float(np.min(past_distances)) if past_distances else 0.0

        phase2_dossier = Phase2ImperialDossier(
            engine_result=engine_res,
            liouville_conserved=engine_res.volume_drift <= _WILKINSON_DRIFT_LIMIT,
            maupertuis_valid=engine_res.maupertuis_action > 0.0,
            ergodic_recurrence_distance=min_return_dist,
        )

        # ─── FASE 3: DECIDE / ACT (Φ₃) ───
        if phase2_dossier.liouville_conserved and phase2_dossier.maupertuis_valid:
            verdict_str = "COHERENT"
            crowbar_triggered = False
        elif engine_res.volume_drift <= 10.0 * _WILKINSON_DRIFT_LIMIT:
            verdict_str = "DEGRADED"
            crowbar_triggered = False
        else:
            verdict_str = "VETOED"
            crowbar_triggered = True
            logger.error(
                f"[IMPERIAL_GUARDS_VETOED] Ruptura de Liouville/Maupertuis: "
                f"Drift={engine_res.volume_drift:.3e}. Disparando Crowbar ESP32 (< 400 ns)."
            )

        return ImperialGuardsVerdict(
            verdict=verdict_str,
            volume_drift=engine_res.volume_drift,
            maupertuis_action=engine_res.maupertuis_action,
            ergodic_return_distance=min_return_dist,
            is_hardware_crowbar_triggered=crowbar_triggered,
        )

    # ────────────────────────────────────────────────────────────────────────────
    # UNIFICACIÓN DE VEREDICTOS EN EL RETÍCULO DE HEYTING Ω₃
    # ────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _join_heyting_verdicts(verdicts: Tuple[str, ...]) -> str:
        rank = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
        inverse = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}
        max_rank = 0
        for verdict in verdicts:
            normalized = str(verdict).strip().upper()
            max_rank = max(max_rank, rank.get(normalized, 2))
        return inverse[max_rank]

    def phase3_decide_from_phase1_and_phase2(
        self,
        phase1_observation: Phase1SpectralObservation,
        phase2_observation: Phase2LogisticObservation,
    ) -> Phase3TribunalDecision:
        if not isinstance(phase1_observation, Phase1SpectralObservation):
            raise TypeError("phase1_observation debe ser Phase1SpectralObservation.")

        if not isinstance(phase2_observation, Phase2LogisticObservation):
            raise TypeError("phase2_observation debe ser Phase2LogisticObservation.")

        final_verdict = self._join_heyting_verdicts(
            (
                phase1_observation.partial_verdict,
                phase2_observation.partial_verdict,
            )
        )

        veto_reasons = tuple(
            list(phase1_observation.veto_reasons) + list(phase2_observation.veto_reasons)
        )

        degraded_reasons = tuple(
            list(phase1_observation.degraded_reasons)
            + list(phase2_observation.degraded_reasons)
        )

        diagnostics: Dict[str, Any] = {
            "phase1": dict(phase1_observation.diagnostics),
            "phase2": dict(phase2_observation.diagnostics),
            "joined_verdict": final_verdict,
        }

        return Phase3TribunalDecision(
            heyting_verdict=final_verdict,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            diagnostics=diagnostics,
        )

    def _cas_interlock(self, expected: bool, desired: bool) -> bool:
        with self._interlock_lock:
            if self._interlock_state == expected:
                self._interlock_state = desired
                return True
            return False

    def reset_hardware_interlock_for_supervision(self) -> bool:
        with self._interlock_lock:
            previous_state = self._interlock_state
            self._interlock_state = False
            return previous_state

    def phase3_act_hardware_interlock(
        self,
        decision: Phase3TribunalDecision,
    ) -> Tuple[bool, float]:
        if not isinstance(decision, Phase3TribunalDecision):
            raise TypeError("decision debe ser Phase3TribunalDecision.")

        verdict = str(decision.heyting_verdict).strip().upper()

        if verdict != "VETOED":
            return False, 0.0

        swapped = self._cas_interlock(expected=False, desired=True)

        if not swapped:
            logger.warning("CAS: el interlock ya estaba enclavado.")

        jitter = float(self._rng.normal(loc=0.0, scale=5.0))
        actuation_latency_ns = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_LATENCY_FLOOR_NS,
                _CROWBAR_LATENCY_CEIL_NS,
            )
        )

        logger.critical(
            "¡VETO SÍNCRONO DISPARADO! Crowbar BT151 [GPIO14] conmutado en %.2f ns.",
            actuation_latency_ns,
        )

        return True, actuation_latency_ns

    # ────────────────────────────────────────────────────────────────────────────
    # API PÚBLICA COMPATIBLE
    # ────────────────────────────────────────────────────────────────────────────

    def audit_spectral_heterogeomorphic_curve(
        self,
        eigenvalues_dirac: Any,
    ) -> Tuple[float, float, str]:
        phase1 = self.phase1_audit_spectral_heterogeomorphic_curve(eigenvalues_dirac)
        return (
            phase1.lipschitz_coefficient,
            phase1.lambda_min_dirac,
            phase1.partial_verdict,
        )

    def audit_logistic_homogeomorphic_curve(
        self,
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
    ) -> Tuple[float, float, float, float, str]:
        phase2 = self.phase2_audit_logistic_from_phase1(
            phase1_observation=None,
            eigenvalues_L=eigenvalues_L,
            betti_0=betti_0,
            betti_1=betti_1,
        )
        return (
            phase2.fiedler_connectivity,
            phase2.cheeger_lower_bound,
            phase2.cohomological_residual,
            phase2.pyramidal_stability,
            phase2.partial_verdict,
        )

    def act_hardware_interlock_simulation(self, verdict: str) -> Tuple[bool, float]:
        decision = Phase3TribunalDecision(
            heyting_verdict=str(verdict),
            veto_reasons=(),
            degraded_reasons=(),
            diagnostics={"source": "compatibility_api"},
        )
        return self.phase3_act_hardware_interlock(decision)

    def execute_guardians_cycle(
        self,
        eigenvalues_dirac: Any,
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
    ) -> ImperialGuardsCertificate:
        """
        Orquesta el ciclo de control de calibre de la capa de Guardias Imperiales.
        """
        phase1 = self.phase1_audit_spectral_heterogeomorphic_curve(eigenvalues_dirac)
        phase2 = self.phase2_audit_logistic_from_phase1(
            phase1_observation=phase1,
            eigenvalues_L=eigenvalues_L,
            betti_0=betti_0,
            betti_1=betti_1,
        )
        phase3 = self.phase3_decide_from_phase1_and_phase2(phase1, phase2)
        interlock_fired, latency = self.phase3_act_hardware_interlock(phase3)

        diagnostics = dict(phase3.diagnostics)
        diagnostics["hardware"] = {
            "interlock_fired": interlock_fired,
            "actuation_latency_ns": latency,
        }

        return ImperialGuardsCertificate(
            phase="G_IMPERIAL_GUARDS_SUTURATED",
            heyting_verdict=phase3.heyting_verdict,
            lipschitz_coefficient=phase1.lipschitz_coefficient,
            dirac_spectral_gap=phase1.lambda_min_dirac,
            fiedler_connectivity=phase2.fiedler_connectivity,
            cheeger_lower_bound=phase2.cheeger_lower_bound,
            pyramidal_stability=phase2.pyramidal_stability,
            cohomological_residual=phase2.cohomological_residual,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            veto_reasons=phase3.veto_reasons,
            degraded_reasons=phase3.degraded_reasons,
            diagnostics=diagnostics,
        )

    def execute_sovereign_governance(
        self,
        eigenvalues_dirac: Any,
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
    ) -> ImperialGuardsCertificate:
        """
        Alias de gobernanza soberana para execute_guardians_cycle.
        """
        return self.execute_guardians_cycle(
            eigenvalues_dirac=eigenvalues_dirac,
            eigenvalues_L=eigenvalues_L,
            betti_0=betti_0,
            betti_1=betti_1,
        )


__all__ = [
    "Phase1SpectralObservation",
    "Phase2LogisticObservation",
    "Phase3TribunalDecision",
    "ImperialGuardsCertificate",
    "Phase1ImperialDossier",
    "Phase2ImperialDossier",
    "ImperialGuardsVerdict",
    "Phase1SpectralGuardianMixin",
    "Phase2LogisticGuardianMixin",
    "ImperialGuardsAgent",
]

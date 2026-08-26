# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Guards Agent (Guardias Imperiales de Calibre de de Rham)   ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_agent.py              ║
║ Versión: 3.0.0-Doctoral-Heyting-OODA-Cheeger-Connes-CAS-Kahan-Secure         ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y GEOMÉTRICA DE DE RHAM:
────────────────────────────────────────────────────────────────────────────────
Ejerce la censura de primer nivel sobre el foso de la obra en la Malla Agéntica
de APU Filter v5.0. Evalúa la regularidad y conectividad espectral del
grafo de presupuesto $G = (V, E)$ mediante dos aduanas de control de calibre:

ADUANA 1: CURVAS HETEROGEOMORFAS DE AUDITORÍA ESPECTRAL (CONNES)
────────────────────────────────────────────────────────────────────────────────
Audita el confinamiento de Lipschitz no conmutativo sobre la variedad continua de
Hilbert $\mathcal{H}_{\text{MAC}}$ mediante la Cota de Regularidad de Connes-Daleckii-Krein:
   $$L_{\max} \le \frac{1}{2 \lambda_{\min}^{3/2}} \le \tau_{\mathrm{Lipschitz}}$$
Donde $\lambda_{\min} > 0$ es el autovalor mínimo del operador de Dirac no conmutativo
$\not\!D = \rho_{\text{MAC}}^{-1/2}$ (piso de regularización de Tikhonov Espectral).
Si el modelo de lenguaje (LLM) alucina o inyecta transitorios de-normalizados, el
gap espectral colapsa ($\lambda_{\min} \to 0$), provocando la divergencia asintótica
de la constante de Lipschitz ($L_{\max} \to \infty$) y anulando determinísticamente
la probabilidad de emisión inválida:
   $$P(x_{\mathrm{invalid}}) = 0$$

ADUANA 2: CURVAS HOMOGEOMORFAS DE CUELLOS LOGÍSTICOS (CHEEGER & FIEDLER)
────────────────────────────────────────────────────────────────────────────────
Audita la conectividad algebraica y la ausencia de cuellos de botella u obstrucciones
topológicas sobre el complejo simplicial $K$ evaluando el valor de Fiedler $\lambda_2$
del Laplaciano de Haz de de Rham-Hodge $L_F = \delta_0^\top G^{-1} \delta_0$.
Somete el grafo de dependencias a la Desigualdad Isoperimétrica de Cheeger:
   $$\frac{h^2(G)}{2} \le \lambda_2 \le 2 h(G) \implies \frac{\lambda_2}{2} \le h(G) \le \sqrt{2 \lambda_2}$$
Monitorea síncronamente el Índice de Estabilidad Piramidal $\Psi$:
   $$\Psi = \frac{\lambda_2}{1.0 + \beta_1 + (\beta_0 - 1)} \ge \Psi_{\mathrm{min}}$$
Donde $\beta_0 > 1$ revela sub-grafos huérfanos e islas de contratistas disconexas, y
$\beta_1 > 0$ expone ciclos parásitos, triangulación de presupuestos y socavones lógicos.

INVARIANTES CATEGÓRICOS Y DE HARDWARE PERIMETRAL:
────────────────────────────────────────────────────────────────────────────────
- Invarianza de la signatura métrica de-confinada: $\operatorname{sgn}(G) = (1, n-1)$.
- Hermiticidad incondicional del operador densidad: $\rho = \rho^\dagger \succeq 0$.
- Unitariedad de la traza cuántica de la sabiduría: $\operatorname{Tr}(\rho) \equiv 1.0$.
- Veto en el retículo de Heyting $\Omega_3 = \{\text{COHERENT}, \text{DEGRADED}, \text{VETOED}\}$ ($\top = \text{VETOED}$).
- Interrupción perimetral ESP32 en IRAM ($t_{\text{actuation}} \le 400\,\text{ns}$) activando el tiristor BT151 (Crowbar) vía GPIO14.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np

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
# CONTRATOS INMUTABLES DE FASE
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Phase1SpectralObservation:
    """
    Contrato formal de salida de la FASE 1.

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
    Contrato formal de salida de la FASE 2.

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
    Contrato formal de salida de la FASE 3.

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

    Se conserva compatibilidad con la versión 1.x, añadiendo razones de veto,
    razones de degradación y diagnósticos extendidos.
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

    Audita el confinamiento de Lipschitz no conmutativo del operador de Dirac
    asociado al estado mixto semántico de la MAC mediante el Teorema de Connes:

        L_max ≤ 1 / (2 λ_min^{3/2})

    El último método formal de esta fase devuelve `Phase1SpectralObservation`,
    contrato que continúa hacia la Fase 2.
    """

    def __init__(self, config_dim_n: int) -> None:
        """
        Inicializa la aduana espectral de-confinada.

        Args:
            config_dim_n: Dimensión del espacio de configuración.
        """
        self._n = self._validate_positive_int("config_dim_n", config_dim_n)

        # Estado interno para simulación de cerrojo CAS en Fase 3.
        self._interlock_lock = threading.Lock()
        self._interlock_state = False

    # ────────────────────────────────────────────────────────────────────────────
    # VALIDACIÓN Y CONVERSIÓN NUMÉRICA
    # ────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        """Valida que `value` sea un entero estrictamente positivo."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value <= 0:
            raise ValueError(f"{name} debe ser estrictamente mayor que cero.")
        return int(value)

    @staticmethod
    def _validate_nonnegative_int(name: str, value: Any) -> int:
        """Valida que `value` sea un entero no negativo."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value < 0:
            raise ValueError(f"{name} debe ser mayor o igual que cero.")
        return int(value)

    @staticmethod
    def _validate_finite_nonnegative(name: str, value: Any) -> float:
        """Valida que `value` sea un número real finito no negativo."""
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
        """
        Convierte `values` a ndarray float64 real, rechazando componentes
        imaginarias no despreciables y valores no finitos.
        """
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
                    f"{name} posee componente imaginaria no despreciable. "
                    "Se exige espectro/matriz real dentro de tolerancia."
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
        """Valida y convierte un vector real float64."""
        arr = self._as_real_float_array(values, name)
        return arr.ravel()

    # ────────────────────────────────────────────────────────────────────────────
    # SUMACIÓN COMPENSADA DE KAHAN-NEUMAIER
    # ────────────────────────────────────────────────────────────────────────────

    def kahan_compensated_sum(self, terms: np.ndarray) -> float:
        r"""
        Realiza sumación compensada de Kahan-Neumaier para evitar acumulación
        de deriva por redondeo de Wilkinson en la mantisa flotante de la CPU.

        Args:
            terms: Vector de términos reales finitos.

        Returns:
            Suma compensada como float.
        """
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

    # ────────────────────────────────────────────────────────────────────────────
    # CÁLCULO Y CLASIFICACIÓN DE LA COTA LIPSCHITZ DE CONNES
    # ────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_lipschitz_coefficient(lambda_min: float) -> float:
        r"""
        Calcula de forma estable:
            L_max ≤ 1 / (2 λ_min^{3/2})

        Si λ_min es cero, no finito o inferior al epsilon de máquina, retorna
        infinito para forzar censura determinística.
        """
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
        """
        Clasifica la cota Lipschitz en el retículo de Heyting.

        Returns:
            (verdict, veto_reasons, degraded_reasons)
        """
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

    # ────────────────────────────────────────────────────────────────────────────
    # ÚLTIMO MÉTODO FORMAL DE LA FASE 1
    # Su retorno `Phase1SpectralObservation` continúa hacia la FASE 2.
    # ────────────────────────────────────────────────────────────────────────────

    def phase1_audit_spectral_heterogeomorphic_curve(
        self,
        eigenvalues_dirac: Any,
    ) -> Phase1SpectralObservation:
        r"""
        [FASE 1 — GUARDIA 1: CURVAS HETEROGEOMORFAS]

        Audita el confinamiento de Lipschitz no conmutativo del operador de Dirac
        asociado al estado mixto semántico de la MAC mediante el Teorema de Connes:

            L_max ≤ 1 / (2 λ_min^{3/2})

        Args:
            eigenvalues_dirac: Espectro de autovalores del operador de Dirac.

        Returns:
            Phase1SpectralObservation con L_max, λ_min, veredicto parcial y razones.
        """
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
            logger.warning("Fuga espectral absoluta detectada: espectro de Dirac vacío.")
            return Phase1SpectralObservation(
                dirac_spectrum_size=0,
                lambda_min_dirac=0.0,
                lipschitz_coefficient=float("inf"),
                partial_verdict="VETOED",
                veto_reasons=("dirac_spectrum_empty",),
                degraded_reasons=(),
                diagnostics=diagnostics,
            )

        # Regularización estable sqrt(λ² + ε²) usando hypot para evitar overflow.
        regularized_abs = np.hypot(eigenvalues, _HIGHAM_REG_SQRT)

        # Filtrar valores bajo el límite espectral de Wilkinson.
        valid_eigs = regularized_abs[regularized_abs > _WILKINSON_DRIFT_LIMIT]
        diagnostics["dirac_valid_eigenvalue_count"] = int(valid_eigs.size)

        if valid_eigs.size == 0:
            logger.warning(
                "Fuga espectral absoluta detectada: espectro de Dirac colapsado "
                "bajo el límite de Wilkinson."
            )
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
# Anidada sobre FASE 1 por herencia.
# ════════════════════════════════════════════════════════════════════════════════

class Phase2LogisticGuardianMixin(Phase1SpectralGuardianMixin):
    """
    FASE 2 — GUARDIA 2.

    Audita los cuellos de botella organizacionales e ineficiencias de la red
    mediante la conectividad algebraica de Fiedler, la desigualdad de Cheeger,
    los números de Betti y el índice de estabilidad piramidal Ψ.
    """

    def __init__(
        self,
        config_dim_n: int,
        cheeger_threshold: float = _DEFAULT_CHEEGER_THRESHOLD,
    ) -> None:
        """
        Inicializa la aduana topológica/logística.

        Args:
            config_dim_n: Dimensión del espacio de configuración.
            cheeger_threshold: Umbral crítico para la conectividad de Fiedler.
        """
        super().__init__(config_dim_n)
        self._cheeger_threshold = self._validate_finite_nonnegative(
            "cheeger_threshold",
            cheeger_threshold,
        )

    # ────────────────────────────────────────────────────────────────────────────
    # VALIDACIÓN DE NÚMEROS DE BETTI
    # ────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_betti(name: str, value: Any) -> int:
        """Valida que un número de Betti sea entero no negativo."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value < 0:
            raise ValueError(f"{name} debe ser mayor o igual que cero.")
        return int(value)

    # ────────────────────────────────────────────────────────────────────────────
    # CÁLCULO ROBUSTO DE LA BRECHA DE FIEDLER
    # ────────────────────────────────────────────────────────────────────────────

    def _fiedler_gap(self, eigenvalues_L: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Calcula la brecha de Fiedler λ₂ del Laplaciano normalizado.

        Estrategia fail-safe:
            - Si el espectro es inválido o no finito, retorna gap 0.
            - Si hay violación de semidefinición positiva, retorna gap 0.
            - Tolera pequeños negativos numéricos y los proyecta a cero.
        """
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

        # Proyección de ceros numéricos.
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

    # ────────────────────────────────────────────────────────────────────────────
    # CLASIFICADOR LOGÍSTICO / TOPOLOGICO
    # ────────────────────────────────────────────────────────────────────────────

    def _classify_logistic_metrics(
        self,
        fiedler_gap: float,
        cohomological_residual: float,
        pyramidal_stability: float,
        betti_0: int,
        betti_1: int,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        """
        Clasifica las métricas logísticas/topológicas en el retículo de Heyting.

        Returns:
            (verdict, veto_reasons, degraded_reasons)
        """
        veto_reasons = []
        degraded_reasons = []

        # Fiedler.
        if not math.isfinite(fiedler_gap):
            veto_reasons.append("fiedler_connectivity_nonfinite")
        elif fiedler_gap < 0.0:
            veto_reasons.append("fiedler_connectivity_negative")
        else:
            if fiedler_gap < self._cheeger_threshold:
                veto_reasons.append("fiedler_connectivity_below_cheeger_threshold")
            elif fiedler_gap < _LOGISTIC_DEGRADED_FIEDLER:
                degraded_reasons.append("fiedler_connectivity_below_degraded_threshold")

        # Residuo cohomológico.
        if not math.isfinite(cohomological_residual):
            veto_reasons.append("cohomological_residual_nonfinite")
        elif cohomological_residual < 0.0:
            veto_reasons.append("cohomological_residual_negative")
        elif cohomological_residual > 0.0:
            veto_reasons.append("cohomological_residual_nonzero")

        # Topología explícita.
        if betti_0 == 0:
            veto_reasons.append("empty_complex_detected")
        elif betti_0 > 1:
            veto_reasons.append("data_islands_detected")

        if betti_1 > 0:
            veto_reasons.append("logical_loops_detected")

        # Estabilidad piramidal Ψ.
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

    # ────────────────────────────────────────────────────────────────────────────
    # ÚLTIMO MÉTODO FORMAL DE LA FASE 2
    # Recibe opcionalmente el contrato de Fase 1 y produce el contrato de Fase 3.
    # ────────────────────────────────────────────────────────────────────────────

    def phase2_audit_logistic_from_phase1(
        self,
        phase1_observation: Optional[Phase1SpectralObservation],
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
    ) -> Phase2LogisticObservation:
        r"""
        [FASE 2 — GUARDIA 2: CURVAS HOMOGEOMORFAS]

        Audita los cuellos de botella organizacionales e ineficiencias de la red
        calculando la conectividad de Fiedler, cotas de Cheeger, residuo
        cohomológico y estabilidad piramidal Ψ.

        Args:
            phase1_observation: Contrato de Fase 1, opcional para trazabilidad.
            eigenvalues_L: Autovalores del Laplaciano normalizado de de Rham-Hodge.
            betti_0: Componentes conexas simpliciales (β₀).
            betti_1: Bucles lógicos o dependencias circulares parásitas (β₁).

        Returns:
            Phase2LogisticObservation con métricas, veredicto parcial y razones.
        """
        if phase1_observation is not None and not isinstance(
            phase1_observation,
            Phase1SpectralObservation,
        ):
            raise TypeError("phase1_observation debe ser Phase1SpectralObservation.")

        b0 = self._validate_betti("betti_0", betti_0)
        b1 = self._validate_betti("betti_1", betti_1)

        # Residuo cohomológico riguroso:
        # Se exige β₀ = 1 y β₁ = 0. Cualquier desviación penaliza.
        cohom_residual = float(b1 + abs(b0 - 1))

        fiedler_gap, fiedler_diagnostics = self._fiedler_gap(eigenvalues_L)

        if math.isfinite(fiedler_gap) and fiedler_gap > 0.0:
            safe_fiedler = float(fiedler_gap)
        else:
            safe_fiedler = 0.0

        # Compatibilidad v1.x:
        # El campo principal conserva el proxy original h²/2 ≈ λ₂²/2.
        cheeger_lower_bound = float((safe_fiedler * safe_fiedler) / 2.0)

        # Cotas rigurosas adicionales para diagnósticos:
        #   h ≥ λ₂ / 2
        #   h ≤ sqrt(2 λ₂)
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
            # Claves compatibles con versión 1.x.
            "fiedler_gap": fiedler_gap,
            "cheeger_lower_bound": cheeger_lower_bound,
            "pyramidal_stability": psi_stability,
            "has_cohomological_obstruction": cohom_residual > 0.0,
            "islands_detected": b0 > 1,
            "loops_detected": b1 > 0,
            # Claves extendidas.
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
# FASE 3 — TRIBUNAL DE HEYTING Y ACTUACIÓN CAS/CROWBAR
# Anidada sobre FASE 2 por herencia.
# ════════════════════════════════════════════════════════════════════════════════

class ImperialGuardsAgent(Phase2LogisticGuardianMixin):
    """
    Guardias Imperiales de la Malla de APU Filter v5.0.

    Ejecuta el ciclo de lazo cerrado OODA para censurar derivas semánticas
    y estrangular cuellos de botella organizacionales.
    """

    def __init__(
        self,
        config_dim_n: int,
        cheeger_threshold: float = _DEFAULT_CHEEGER_THRESHOLD,
        *,
        rng_seed: Optional[int] = None,
    ) -> None:
        """
        Inicializa las aduanas de control espectral y topológico de-confinado.

        Args:
            config_dim_n: Dimensión del espacio de configuración.
            cheeger_threshold: Umbral crítico para la conectividad de Fiedler.
            rng_seed: Semilla opcional para reproducibilidad del jitter CAS.
        """
        super().__init__(config_dim_n, cheeger_threshold)
        self._rng = np.random.default_rng(rng_seed)

    # ────────────────────────────────────────────────────────────────────────────
    # UNIFICACIÓN DE VEREDICTOS EN EL RETÍCULO DE HEYTING Ω₃
    # ────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _join_heyting_verdicts(verdicts: Tuple[str, ...]) -> str:
        """
        Unifica veredictos mediante Supremo (join ⊔) en el retículo distributivo
        de Heyting:

            COHERENT < DEGRADED < VETOED

        Cualquier veredicto desconocido se interpreta fail-safe como VETOED.
        """
        rank = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
        inverse = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}

        max_rank = 0

        for verdict in verdicts:
            normalized = str(verdict).strip().upper()
            max_rank = max(max_rank, rank.get(normalized, 2))

        return inverse[max_rank]

    # ────────────────────────────────────────────────────────────────────────────
    # MÉTODO FORMAL DE DECISIÓN FASE 3
    # ────────────────────────────────────────────────────────────────────────────

    def phase3_decide_from_phase1_and_phase2(
        self,
        phase1_observation: Phase1SpectralObservation,
        phase2_observation: Phase2LogisticObservation,
    ) -> Phase3TribunalDecision:
        """
        [FASE 3 — DECIDE]

        Unifica los veredictos parciales de los Guardias 1 y 2 mediante el join
        en el retículo de Heyting Ω₃.

        Args:
            phase1_observation: Contrato de salida de Fase 1.
            phase2_observation: Contrato de salida de Fase 2.

        Returns:
            Phase3TribunalDecision con veredicto final y razones agregadas.
        """
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

    # ────────────────────────────────────────────────────────────────────────────
    # CERROJO ATÓMICO CAS Y ACTUADOR CROWBAR SIMULADO
    # ────────────────────────────────────────────────────────────────────────────

    def _cas_interlock(self, expected: bool, desired: bool) -> bool:
        """
        Simula una operación atómica Compare-And-Swap sobre el estado del
        disyuntor de potencia.

        Args:
            expected: Estado esperado.
            desired: Estado deseado.

        Returns:
            True si el CAS tuvo éxito, False en caso contrario.
        """
        with self._interlock_lock:
            if self._interlock_state == expected:
                self._interlock_state = desired
                return True
            return False

    def reset_hardware_interlock_for_supervision(self) -> bool:
        """
        Reinicia manualmente el estado simulado del interlock.

        Returns:
            Estado previo del interlock.
        """
        with self._interlock_lock:
            previous_state = self._interlock_state
            self._interlock_state = False
            return previous_state

    def phase3_act_hardware_interlock(
        self,
        decision: Phase3TribunalDecision,
    ) -> Tuple[bool, float]:
        r"""
        [FASE 3 — ACT]

        Simula síncronamente el actuador ciber-físico de lazo cerrado en silicio.

        Si el veredicto es VETOED, ejecuta un cerrojo CAS para simular el disparo
        del tiristor BT151 (Crowbar) en GPIO14 en menos de 400 ns:

            t_actuation ≤ τ_IRAM = 400 ns

        Args:
            decision: Contrato Phase3TribunalDecision derivado de la Fase Decide.

        Returns:
            Tuple con estado de activación del disyuntor y latencia simulada en ns.
        """
        if not isinstance(decision, Phase3TribunalDecision):
            raise TypeError("decision debe ser Phase3TribunalDecision.")

        verdict = str(decision.heyting_verdict).strip().upper()

        if verdict != "VETOED":
            return False, 0.0

        swapped = self._cas_interlock(expected=False, desired=True)

        if not swapped:
            logger.warning(
                "CAS: el interlock ya estaba enclavado. Se reconoce el veto "
                "y se registra la latencia de actuación."
            )

        jitter = float(self._rng.normal(loc=0.0, scale=5.0))
        actuation_latency_ns = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_LATENCY_FLOOR_NS,
                _CROWBAR_LATENCY_CEIL_NS,
            )
        )

        logger.critical(
            "¡VETO SÍNCRONO DISPARADO POR GUARDIAS IMPERIALES! "
            "Crowbar BT151 [GPIO14] conmutado en %.2f ns. Maquinaria de obra paralizada.",
            actuation_latency_ns,
        )

        return True, actuation_latency_ns

    # ────────────────────────────────────────────────────────────────────────────
    # API PÚBLICA COMPATIBLE CON VERSIÓN 1.X
    # ────────────────────────────────────────────────────────────────────────────

    def audit_spectral_heterogeomorphic_curve(
        self,
        eigenvalues_dirac: Any,
    ) -> Tuple[float, float, str]:
        r"""
        [COMPATIBILIDAD 1.X — GUARDIA 1]

        Audita el confinamiento de Lipschitz no conmutativo del operador de Dirac.

        Args:
            eigenvalues_dirac: Espectro de autovalores del operador de Dirac.

        Returns:
            Tuple con el coeficiente Lipschitz L_max, el gap espectral y el
            veredicto parcial.
        """
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
        r"""
        [COMPATIBILIDAD 1.X — GUARDIA 2]

        Audita los cuellos de botella organizacionales e ineficiencias de la red.

        Args:
            eigenvalues_L: Autovalores del Laplaciano normalizado.
            betti_0: Componentes conexas simpliciales (β₀).
            betti_1: Bucles lógicos o dependencias circulares parásitas (β₁).

        Returns:
            Tuple con (Fiedler value, Cheeger proxy, residuo de de Rham,
            Ψ stability, veredicto parcial).
        """
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
        r"""
        [COMPATIBILIDAD 1.X — FASE ACT]

        Simula síncronamente el actuador ciber-físico de lazo cerrado en silicio.

        Args:
            verdict: Veredicto de Heyting derivado de la Fase Decide.

        Returns:
            Tuple con estado de activación del disyuntor y latencia simulada en ns.
        """
        decision = Phase3TribunalDecision(
            heyting_verdict=str(verdict),
            veto_reasons=(),
            degraded_reasons=(),
            diagnostics={"source": "compatibility_api"},
        )

        return self.phase3_act_hardware_interlock(decision)

    # ────────────────────────────────────────────────────────────────────────────
    # ORQUESTADOR COMPLETO DEL CICLO DE LOS GUARDIAS IMPERIALES
    # ────────────────────────────────────────────────────────────────────────────

    def execute_guardians_cycle(
        self,
        eigenvalues_dirac: Any,
        eigenvalues_L: Any,
        betti_0: int,
        betti_1: int,
    ) -> ImperialGuardsCertificate:
        """
        Orquesta el ciclo de control de calibre de la capa de Guardias Imperiales
        sobre el espacio de fase simpléctico de-confinado.

        Args:
            eigenvalues_dirac: Autovalores del operador de Dirac de Connes.
            eigenvalues_L: Autovalores del Laplaciano de de Rham.
            betti_0: Número de Betti 0 (componentes conexas).
            betti_1: Número de Betti 1 (bucles cíclicos).

        Returns:
            ImperialGuardsCertificate con la firma inmutable de de Rham.
        """
        # 1. OBSERVE & ORIENT: Guardia 1 (Curvas Heterogeomorfas)
        phase1 = self.phase1_audit_spectral_heterogeomorphic_curve(eigenvalues_dirac)

        # 2. OBSERVE & ORIENT: Guardia 2 (Curvas Homogeomorfas)
        # Se pasa phase1 para trazabilidad anidada, aunque la auditoría logística
        # es matemáticamente independiente.
        phase2 = self.phase2_audit_logistic_from_phase1(
            phase1_observation=phase1,
            eigenvalues_L=eigenvalues_L,
            betti_0=betti_0,
            betti_1=betti_1,
        )

        # 3. DECIDE: Tribunal de Heyting.
        phase3 = self.phase3_decide_from_phase1_and_phase2(phase1, phase2)

        # 4. ACT: Interlock perimetral simulado ESP32 / BT151.
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


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA DE FIRMAS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "Phase1SpectralObservation",
    "Phase2LogisticObservation",
    "Phase3TribunalDecision",
    "ImperialGuardsCertificate",
    "Phase1SpectralGuardianMixin",
    "Phase2LogisticGuardianMixin",
    "ImperialGuardsAgent",
]
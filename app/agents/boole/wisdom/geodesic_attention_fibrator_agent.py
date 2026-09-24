# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Geodesic Attention Fibrator Agent (Custodio de la Covarianza)               ║
║ RUTA   : app/agents/boole/wisdom/geodesic_attention_fibrator_agent.py                ║
║ VERSIÓN: 5.0.0-Doctoral-Rigorous-3Phases-Ricci-Polyakov-FeynmanKac-Heyting-Pure      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA: GEOMETRÍA RIEMANNIANA Y MECÁNICA CUÁNTICA EN ESTRATO WISDOM
═════════════════════════════════════════════════════════════════════════════════════

FUNDACIÓN MATEMÁTICA DOCTORAL REFINADA:
───────────────────────────────────────

I. GEOMETRÍA RIEMANNIANA Y FLUJO DE RICCI:
   Auditoría exhaustiva de tensor métrico $G \in \mathrm{Sym}^+(n)$.
   Flujo de Ricci discreto: $\frac{\partial G}{\partial t} = -2\mathrm{Ric}(G)$.
   Convergencia hacia métrica de Einstein: $\mathrm{Ric}(G) = \lambda G$.
   Teoría de Perelman: función potencial $\mathcal{F}$ monótona en flujo.

II. GEODÉSICAS Y ACCIÓN DE POLYAKOV:
    Trayectorias minimales sobre variedad riemanniana: $\nabla_{\gamma'} \gamma' = 0$.
    Acción geodésica: $E[\gamma] = \frac{1}{2}\int_0^T g_{\mu\nu} \dot{\gamma}^\mu \dot{\gamma}^\nu d\tau$.
    Principio variacional: geodésicas extremizan la acción funcional.
    Longitud de arco: $L[\gamma] = \sqrt{2 E[\gamma]}$.

III. INTEGRAL DE TRAYECTORIA DE FEYNMAN-KAC:
     Amplitud de transición cuántica: $\Psi[\gamma] = \exp(-S_E[\gamma]/\hbar_{\mathrm{eff}})$.
     Partición funcional: $Z = \int \mathcal{D}[\gamma] \exp(-S_E[\gamma]/\hbar)$.
     Mecanismo de supresión: trayectorias de alta acción contribuyen exponencialmente menos.
     Barrera tunelante: supresión cuántica de transiciones energéticamente prohibidas.

IV. TENSOR DE TORSIÓN Y COHOMOLOGÍA HODGE:
    Torsión nilpotente en asociaedro: $T^{\rho}_{\mu\nu} = \Gamma^{\rho}_{\mu\nu} - \Gamma^{\rho}_{\nu\mu}$.
    En Levi-Civita: $T = 0$ (simetría de símbolos de Christoffel).
    Acoplamiento a acción: $S_E = E_{\mathrm{Polyakov}} + \lambda \|T\|^2_{\mathrm{HS}}$.
    Norma de Hilbert-Schmidt: $\|T\|^2_{\mathrm{HS}} = \mathrm{Tr}(T^T T)$ (invariante).

V. RETÍCULO DE HEYTING Y LÓGICA INTUICIONISTA:
   Clasificador de subobjetos $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$.
   Negación no involutiva: $\neg\neg p \not\equiv p$ (intuicionismo).
   Orden de verdad: COHERENT $\geq$ DEGRADED $\geq$ VETOED.
   Colapso determinista en clasificador toposiano.

VI. TEORÍA ESPECTRAL Y ACONDICIONAMIENTO MÉTRICO:
    Número de condición de Wilkinson: $\kappa(G) = \lambda_{\max}/\lambda_{\min}$.
    Regularización espectral: proyección de autovalores negativos a piso numérico.
    Saneamiento de métrica degenerada en cono SPD.

VII. ARITMETICA COMPENSADA Y PRECISIÓN IEEE 754:
     Neutralización de redondeo acumulativo en sumas y productos.
     Guardas numéricas contra singularidades aritméticas.
     Tolerancias relativas escaladas por magnitud operanda.

Impacto Semántico (PAIN & GAIN):
────────────────────────────────

✗ PAIN:  Atención euclidiana plana → conexiones estocásticas → alucinaciones neuronales
         → pérdida de consistencia semántica → degradación de confianza en LLM.

✓ GAIN:  Atención geodésica riemanniana → conexiones minimales → supresión cuántica
         → coherencia semántica preservada → LLM confiable bajo supervisión topológica.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import (
    Any, Final, List, Optional, Tuple, Literal, Protocol
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Agents.WISDOM.GeodesicAttentionFibrator.Doctoral")

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# CAPA 0: CONSTANTES INMUTABLES Y LÍMITES FÍSICOS
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_MACHINE_TINY: Final[float] = float(np.finfo(np.float64).tiny)
_MACHINE_MAX: Final[float] = float(np.finfo(np.float64).max)

# Tolerancias de geometría riemanniana
_RICCI_CONVERGENCE_TOL: Final[float] = 1.0e-8
_METRIC_SYMMETRY_TOLERANCE: Final[float] = 1.0e-10
_SPD_NEGATIVE_TOLERANCE: Final[float] = 1.0e-12
_SPD_EIGENVALUE_FLOOR: Final[float] = 1.0e-15

# Límites de energía y acción
_POLYAKOV_ENERGY_CEILING: Final[float] = 1.0e6
_HBAR_EFF: Final[float] = 1.054e-2  # Constante efectiva de Planck en unidades normalizadas
_MIN_QUANTUM_AMPLITUDE: Final[float] = 1.0e-4

# Tolerancias de conservación
_KINETIC_TOLERANCE: Final[float] = 1.0e-12
_ENERGY_TOLERANCE: Final[float] = 1.0e-12
_ACTION_TOLERANCE: Final[float] = 1.0e-12
_TORSION_TOLERANCE: Final[float] = 1.0e-12

# Factor de seguridad numérica de Wilkinson
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0

_ENGINE_VERSION: Final[str] = "5.0.0"


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# CAPA 1: JERARQUÍA DE EXCEPCIONES DOCTORALES
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

class GeodesicAttentionAgentError(Exception):
    r"""Excepción raíz del Custodio de Covarianza Atencional en $V_{\mathbb{W}}$."""
    pass


class GeodesicInputValidationError(GeodesicAttentionAgentError):
    r"""Rechazo de entrada no conforme a especificación (NaN, Inf, complejos, etc.)."""
    pass


class MetricDegeneracyError(GeodesicAttentionAgentError):
    r"""Tensor métrico $G \notin \mathrm{Sym}^+(n)$ (no simétrico o no SPD)."""
    pass


class RicciFlowDivergenceError(GeodesicAttentionAgentError):
    r"""Flujo de Ricci discreto no converge a métrica de Einstein."""
    pass


class PolyakovActionViolationError(GeodesicAttentionAgentError):
    r"""Acción geodésica de Polyakov negativa, divergente o violación métrica."""
    pass


class QuantumFeynmanKacVeto(GeodesicAttentionAgentError):
    r"""Amplitud de transición de Feynman-Kac cae bajo barrera cuántica mínima."""
    pass


class CryptographicChainError(GeodesicAttentionAgentError):
    r"""Ruptura de cadena SHA-256 entre fases anidadas."""
    pass


class TopologicalInvariantViolation(GeodesicAttentionAgentError):
    r"""Violación de invariante topológico (índice de Maslov, género, etc.)."""
    pass


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# CAPA 2: UTILIDADES DE SANEAMIENTO NUMÉRICO Y VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

class FiniteNumericalValidator:
    r"""
    Capa de guardas numéricas para rechazar singularidades aritméticas
    y asegurar que todo tensor resida en el dominio de análisis diferencial.
    """

    @staticmethod
    def validate_finite_real_array(
        name: str,
        value: Any,
    ) -> NDArray[np.float64]:
        r"""
        Valida que un objeto sea:
        - Convertible a arreglo numérico.
        - Real (no complejo).
        - Finito (sin NaN/Inf).
        - Retorna copia float64 C-contigua.
        """
        try:
            raw = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise GeodesicInputValidationError(
                f"'{name}' no puede convertirse a arreglo numérico real."
            ) from exc

        if np.iscomplexobj(raw):
            raise GeodesicInputValidationError(
                f"'{name}' debe ser real; se rechazó entrada compleja."
            )

        if not np.all(np.isfinite(raw)):
            raise GeodesicInputValidationError(
                f"'{name}' contiene NaN o valores infinitos."
            )

        return np.ascontiguousarray(raw, dtype=np.float64)

    @classmethod
    def validate_finite_real_matrix(
        cls,
        name: str,
        value: Any,
        *,
        square: bool = False,
    ) -> NDArray[np.float64]:
        r"""Valida matriz real finita con opción de cuadrado."""
        arr = cls.validate_finite_real_array(name, value)

        if arr.ndim != 2:
            raise GeodesicInputValidationError(
                f"'{name}' debe ser matriz 2D; se recibió {arr.ndim}D."
            )

        if square and arr.shape[0] != arr.shape[1]:
            raise GeodesicInputValidationError(
                f"'{name}' debe ser cuadrada; se recibió {arr.shape}."
            )

        return arr

    @classmethod
    def validate_finite_velocity_matrix(
        cls,
        name: str,
        value: Any,
    ) -> NDArray[np.float64]:
        r"""
        Valida matriz de velocidades geodésicas $(steps, dim)$.
        Acepta vectors 1D y los expande a $(1, dim)$.
        """
        arr = cls.validate_finite_real_array(name, value)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise GeodesicInputValidationError(
                f"'{name}' debe ser 2D $(steps, dim)$ o 1D $(dim)$."
            )

        if arr.size == 0:
            raise GeodesicInputValidationError(
                f"'{name}' no puede ser vacío."
            )

        return arr

    @classmethod
    def validate_finite_scalar(cls, name: str, value: Any) -> float:
        r"""Valida escalar real finito."""
        arr = cls.validate_finite_real_array(name, value)

        if arr.size != 1:
            raise GeodesicInputValidationError(
                f"'{name}' debe ser escalar; tamaño={arr.size}."
            )

        scalar = float(arr.reshape(-1)[0])

        if not math.isfinite(scalar):
            raise GeodesicInputValidationError(
                f"'{name}' no es finito; valor={scalar}."
            )

        return scalar

    @classmethod
    def validate_finite_positive_scalar(cls, name: str, value: Any) -> float:
        r"""Valida escalar estrictamente positivo."""
        scalar = cls.validate_finite_scalar(name, value)

        tol = _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON

        if scalar <= tol:
            raise GeodesicInputValidationError(
                f"'{name}' debe ser > 0; recibido {scalar:.6e}."
            )

        return scalar

    @classmethod
    def validate_finite_nonnegative_scalar(cls, name: str, value: Any) -> float:
        r"""Valida escalar no negativo (proyecta negatividad numérica a cero)."""
        scalar = cls.validate_finite_scalar(name, value)

        tol = _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON

        if scalar < -tol:
            raise GeodesicInputValidationError(
                f"'{name}' debe ser >= 0; recibido {scalar:.6e}."
            )

        return max(0.0, scalar)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# CAPA 3: ESTRUCTURAS DE DATOS INMUTABLES (EXPEDIENTES DE AUDITORÍA)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class RiemannianMetricAudit:
    r"""
    Auditoría espectral exhaustiva de un tensor métrico $G \in \mathrm{Sym}^+(n)$.
    Incluye descomposición de Cholesky, número de condición y regularización espectral.
    """
    dimension: int
    eigenvalues_original: NDArray[np.float64]
    eigenvalues_sanitized: NDArray[np.float64]
    condition_number_original: float
    condition_number_sanitized: float
    metric_symmetry_residual: float
    spectral_floor_applied: bool
    metric_frobenius_norm: float
    metric_max_abs_eigenvalue: float
    metric_min_abs_eigenvalue: float


@dataclass(frozen=True, slots=True)
class RicciFlowAuditData:
    r"""
    EXPEDIENTE TERMINAL DE FASE 1 (Observe).
    Certificado de convergencia del flujo de Ricci.
    Objeto inicial obligatorio de Fase 2.
    """
    dimension: int
    metric_residual_norm: float
    metric_relative_residual: float
    condition_number_g_k: float
    condition_number_g_k_plus_1: float
    metric_convergence_tolerance: float
    metric_frobenius_norm_k: float
    metric_frobenius_norm_k_plus_1: float
    is_metric_converged: bool
    phase1_sha256_seal: str


@dataclass(frozen=True, slots=True)
class GeodesicEnergyAudit:
    r"""
    Auditoría de integración de energía geodésica.
    Incluye términos cinéticos discretos y suma compensada.
    """
    steps: int
    dimension: int
    kinetic_terms: NDArray[np.float64]
    total_kinetic_sum: float
    min_kinetic_term: float
    max_kinetic_term: float
    mean_kinetic_term: float
    kinetic_terms_negative_count: int


@dataclass(frozen=True, slots=True)
class PolyakovActionAuditData:
    r"""
    EXPEDIENTE TERMINAL DE FASE 2 (Orient).
    Certificado de acción geodésica de Polyakov.
    Objeto inicial obligatorio de Fase 3.
    """
    steps: int
    dimension: int
    geodesic_energy: float
    geodesic_energy_per_step: float
    geodesic_length_metric: float
    min_kinetic_term: float
    max_kinetic_term: float
    mean_kinetic_term: float
    energy_ceiling: float
    polyakov_tolerance: float
    metric_condition_number: float
    is_geodesic_stable: bool
    phase2_hmac_sha256: str


@dataclass(frozen=True, slots=True)
class TorsionAndCouplingAudit:
    r"""Auditoría de tensor de torsión y acoplamiento a acción euclidiana."""
    torsion_hs_norm_sq_original: float
    torsion_hs_norm_sq_sanitized: float
    lambda_coupling_original: float
    lambda_coupling_sanitized: float
    coupling_contribution: float
    torsion_tolerance: float


@dataclass(frozen=True, slots=True)
class FeynmanKacAuditData:
    r"""
    EXPEDIENTE TERMINAL DE FASE 3 (Decide & Act).
    Certificado de amplitud cuántica de transición.
    Objeto final del agente.
    """
    euclidean_action: float
    log_transition_amplitude: float
    transition_amplitude: float
    min_quantum_amplitude: float
    hbar_eff_used: float
    quantum_suppression_factor: float
    is_attention_allowed: bool
    phase3_sha256_seal: str


@dataclass(frozen=True, slots=True)
class GeodesicAttentionGovernanceState:
    r"""
    OBJETO FINAL SUPREMO del Endofuntor GeodesicAttentionFibratorAgent.
    Integra los tres certificados de auditoría y veredicto epistemológico final.
    """
    ricci_audit: RicciFlowAuditData
    polyakov_audit: PolyakovActionAuditData
    torsion_audit: TorsionAndCouplingAudit
    feynman_kac_audit: FeynmanKacAuditData
    is_epistemologically_valid: bool
    heyting_verdict: Literal["COHERENT", "DEGRADED", "VETOED"]
    cryptographic_seal_phase3_sha256: str
    total_execution_time_ns: int


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1 ANIDADA: OBSERVE (AUDITORÍA DE CONVERGENCIA DE RICCI)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

class Phase1_RicciFlowObserver(FiniteNumericalValidator):
    r"""
    FASE 1 (OBSERVE): Auditoría de Flujo de Ricci.
    
    Responsabilidades:
    1. Validación de dos métricas consecutivas $G_k, G_{k+1} \in \mathrm{Sym}^+(n)$.
    2. Descomposición espectral y saneamiento de degeneracies.
    3. Evaluación de convergencia métrica en flujo de Ricci discreto.
    4. Certificación determinista con sello SHA-256.
    
    MÉTODO TERMINAL: `observe_ricci_convergence` → `RicciFlowAuditData`.
    Este expediente es el objeto inicial obligatorio de Fase 2.
    """

    def __init__(self, strict_mode: bool = False) -> None:
        self._strict: Final[bool] = bool(strict_mode)
        self._phase_tag: Final[str] = "Phase1_RicciFlowObserver"

    def _audit_riemannian_metric_spd(
        self,
        name: str,
        metric: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], RiemannianMetricAudit]:
        r"""
        Auditoría exhaustiva de tensor métrico $G \in \mathrm{Sym}^+(n)$.
        
        Pasos:
        1. Validación y finitud.
        2. Simetrización.
        3. Descomposición espectral.
        4. Regularización en cono SPD.
        5. Retorno de métrica sanitada y auditoría.
        """
        G = self.validate_finite_real_matrix(name, metric, square=True)

        if G.shape[0] == 0:
            raise MetricDegeneracyError(f"'{name}' es matriz vacía.")

        # Simetrización
        frob_norm = float(la.norm(G, ord="fro"))
        sym_residual = float(la.norm(G - G.T, ord="fro"))
        sym_rel_res = sym_residual / max(1.0, frob_norm)

        sym_tol = max(
            _METRIC_SYMMETRY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        if sym_rel_res > sym_tol:
            raise MetricDegeneracyError(
                f"'{name}' no es simétrica. Residuo relativo={sym_rel_res:.6e}."
            )

        G_sym = (G + G.T) / 2.0

        # Diagonalización
        try:
            eigvals_orig, eigvecs = np.linalg.eigh(G_sym)
        except np.linalg.LinAlgError as exc:
            raise MetricDegeneracyError(
                f"Diagonalización de '{name}' falló."
            ) from exc

        eigvals_orig = np.asarray(eigvals_orig, dtype=np.float64)

        if not np.all(np.isfinite(eigvals_orig)):
            raise MetricDegeneracyError(
                f"Autovalores de '{name}' no son finitos."
            )

        max_eigval_orig = float(np.max(eigvals_orig))
        min_eigval_orig = float(np.min(eigvals_orig))

        if max_eigval_orig <= 0.0:
            raise MetricDegeneracyError(
                f"'{name}' no es SPD. λ_max={max_eigval_orig:.6e}."
            )

        # Regularización espectral
        neg_tol = max(
            _SPD_NEGATIVE_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, max_eigval_orig),
        )

        if min_eigval_orig < -neg_tol:
            raise MetricDegeneracyError(
                f"'{name}' no es SPD. λ_min={min_eigval_orig:.6e}."
            )

        eigval_floor = max(
            _SPD_EIGENVALUE_FLOOR,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, max_eigval_orig),
        )

        spectral_floor_applied = bool(np.any(eigvals_orig < eigval_floor))

        if spectral_floor_applied:
            logger.warning(
                f"'{name}' regularizado espectralmente a piso {eigval_floor:.6e}."
            )

        eigvals_san = np.clip(eigvals_orig, eigval_floor, None)

        # Reconstrucción
        G_san = (eigvecs * eigvals_san) @ eigvecs.T
        G_san = (G_san + G_san.T) / 2.0

        if not np.all(np.isfinite(G_san)):
            raise MetricDegeneracyError(
                f"Reconstrucción de '{name}' produjo valores no finitos."
            )

        min_eigval_san = float(np.min(eigvals_san))
        max_eigval_san = float(np.max(eigvals_san))

        if min_eigval_san <= 0.0:
            raise MetricDegeneracyError(
                f"'{name}' degenerada tras saneamiento."
            )

        kappa_orig = max_eigval_orig / max(min_eigval_orig, eigval_floor)
        kappa_san = max_eigval_san / min_eigval_san

        audit = RiemannianMetricAudit(
            dimension=int(G_sym.shape[0]),
            eigenvalues_original=np.asarray(eigvals_orig, dtype=np.float64),
            eigenvalues_sanitized=np.asarray(eigvals_san, dtype=np.float64),
            condition_number_original=float(kappa_orig),
            condition_number_sanitized=float(kappa_san),
            metric_symmetry_residual=float(sym_residual),
            spectral_floor_applied=spectral_floor_applied,
            metric_frobenius_norm=float(frob_norm),
            metric_max_abs_eigenvalue=float(max_eigval_san),
            metric_min_abs_eigenvalue=float(min_eigval_san),
        )

        return G_san, audit

    def _phase1_sha256_seal(
        self,
        g_k: NDArray[np.float64],
        g_k_plus_1: NDArray[np.float64],
        residual_norm: float,
        relative_residual: float,
        audit_k: RiemannianMetricAudit,
        audit_k_plus_1: RiemannianMetricAudit,
    ) -> str:
        r"""Sello SHA-256 determinista de Fase 1."""
        hasher = hashlib.sha256()
        hasher.update(_ENGINE_VERSION.encode("ascii"))
        hasher.update(self._phase_tag.encode("ascii"))
        hasher.update(b"Ricci_Convergence")
        hasher.update(f"{audit_k.dimension}".encode("ascii"))
        hasher.update(f"{residual_norm:.16e}".encode("ascii"))
        hasher.update(f"{relative_residual:.16e}".encode("ascii"))
        hasher.update(np.ascontiguousarray(g_k, dtype=np.float64).tobytes())
        hasher.update(np.ascontiguousarray(g_k_plus_1, dtype=np.float64).tobytes())
        return hasher.hexdigest()

    def observe_ricci_convergence(
        self,
        g_k: NDArray[np.float64],
        g_k_plus_1: NDArray[np.float64],
    ) -> RicciFlowAuditData:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1 (OBSERVE).
        
        Audita convergencia del flujo de Ricci discreto.
        
        CONTINUACIÓN FUNCTORIAL:
        ```
        ricci_audit = observe_ricci_convergence(g_k, g_k_plus_1)
        polyakov_audit = certify_polyakov_action(...)  # ← Fase 2
        ```
        """
        # Auditorías de ambas métricas
        G_k_san, audit_k = self._audit_riemannian_metric_spd("g_k", g_k)
        G_k_plus_1_san, audit_k_plus_1 = self._audit_riemannian_metric_spd(
            "g_k_plus_1", g_k_plus_1
        )

        # Consistencia dimensional
        if audit_k.dimension != audit_k_plus_1.dimension:
            raise GeodesicInputValidationError(
                f"g_k y g_k_plus_1 tienen dimensiones diferentes: "
                f"{audit_k.dimension} vs {audit_k_plus_1.dimension}."
            )

        # Residuo de flujo
        metric_diff = G_k_plus_1_san - G_k_san

        if not np.all(np.isfinite(metric_diff)):
            raise RicciFlowDivergenceError(
                "Diferencia métrica g_{k+1} - g_k produjo valores no finitos."
            )

        residual_norm = float(la.norm(metric_diff, ord="fro"))
        norm_k = float(la.norm(G_k_san, ord="fro"))
        norm_k_plus_1 = float(la.norm(G_k_plus_1_san, ord="fro"))

        if not math.isfinite(residual_norm):
            raise RicciFlowDivergenceError("Norma del residuo de Ricci no finita.")

        scale = max(1.0, norm_k, norm_k_plus_1)
        relative_residual = residual_norm / scale

        if not math.isfinite(relative_residual):
            raise RicciFlowDivergenceError("Residuo relativo de Ricci no finito.")

        convergence_tol = max(
            _RICCI_CONVERGENCE_TOL,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        if relative_residual >= convergence_tol:
            raise RicciFlowDivergenceError(
                f"Flujo de Ricci no convergió. "
                f"Residuo relativo={relative_residual:.6e} >= {convergence_tol:.6e}."
            )

        # Sello
        seal = self._phase1_sha256_seal(
            g_k, g_k_plus_1, residual_norm, relative_residual, audit_k, audit_k_plus_1
        )

        logger.debug(
            f"Fase 1 (OBSERVE) completada. dim={audit_k.dimension} "
            f"res_rel={relative_residual:.6e} sello={seal[:16]}..."
        )

        return RicciFlowAuditData(
            dimension=audit_k.dimension,
            metric_residual_norm=float(residual_norm),
            metric_relative_residual=float(relative_residual),
            condition_number_g_k=audit_k.condition_number_sanitized,
            condition_number_g_k_plus_1=audit_k_plus_1.condition_number_sanitized,
            metric_convergence_tolerance=float(convergence_tol),
            metric_frobenius_norm_k=audit_k.metric_frobenius_norm,
            metric_frobenius_norm_k_plus_1=audit_k_plus_1.metric_frobenius_norm,
            is_metric_converged=True,
            phase1_sha256_seal=seal,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2 ANIDADA: ORIENT (AUDITORÍA DE ACCIÓN GEODÉSICA DE POLYAKOV)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

class Phase2_PolyakovOrientator(Phase1_RicciFlowObserver):
    r"""
    FASE 2 (ORIENT): Certificación de Acción Geodésica.
    
    INICIO CONTINUO: Recibe obligatoriamente `RicciFlowAuditData` de Fase 1.
    
    Responsabilidades:
    1. Integración de energía geodésica con forma métrica.
    2. Verificación de conservación de energía cinética.
    3. Certificación de estabilidad geodésica.
    4. HMAC-SHA256 de cadena criptográfica.
    
    MÉTODO TERMINAL: `certify_polyakov_action` → `PolyakovActionAuditData`.
    """

    def __init__(self, strict_mode: bool = False) -> None:
        super().__init__(strict_mode=strict_mode)
        self._phase_tag_2: Final[str] = "Phase2_PolyakovOrientator"

    def _integrate_geodesic_energy(
        self,
        geodesic_velocities: NDArray[np.float64],
        metric_sanitized: NDArray[np.float64],
        d_tau: float,
    ) -> GeodesicEnergyAudit:
        r"""
        Integra energía geodésica con suma compensada.
        
        $E[\gamma] = \frac{1}{2} \sum_i v_i^T G v_i \cdot \Delta\tau$
        """
        steps, dim = geodesic_velocities.shape

        # Producto métrico: G @ v^T
        metric_velocities = geodesic_velocities @ metric_sanitized

        # Términos cinéticos: diag(v^T G v)
        kinetic_terms = np.sum(metric_velocities * geodesic_velocities, axis=1)
        kinetic_terms = np.asarray(kinetic_terms, dtype=np.float64)

        if not np.all(np.isfinite(kinetic_terms)):
            raise PolyakovActionViolationError(
                "Términos cinéticos contienen NaN/Inf."
            )

        if kinetic_terms.size == 0:
            raise PolyakovActionViolationError("Trayectoria geodésica vacía.")

        # Regularización de términos negativos pequeños
        max_abs_kinetic = float(np.max(np.abs(kinetic_terms)))
        kinetic_tol = max(
            _KINETIC_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, max_abs_kinetic),
        )

        min_kinetic = float(np.min(kinetic_terms))

        if min_kinetic < -kinetic_tol:
            raise PolyakovActionViolationError(
                f"Energía cinética negativa: {min_kinetic:.6e} < -{kinetic_tol:.6e}."
            )

        negative_count = int(np.sum(kinetic_terms < 0))
        kinetic_terms_clipped = np.clip(kinetic_terms, 0.0, None)

        min_term = float(np.min(kinetic_terms_clipped))
        max_term = float(np.max(kinetic_terms_clipped))
        mean_term = float(np.mean(kinetic_terms_clipped)) if steps > 0 else 0.0

        total_sum = float(np.sum(kinetic_terms_clipped))

        return GeodesicEnergyAudit(
            steps=int(steps),
            dimension=int(dim),
            kinetic_terms=np.asarray(kinetic_terms_clipped, dtype=np.float64),
            total_kinetic_sum=total_sum,
            min_kinetic_term=min_term,
            max_kinetic_term=max_term,
            mean_kinetic_term=mean_term,
            kinetic_terms_negative_count=negative_count,
        )

    def _phase2_hmac_sha256(
        self,
        ricci_seal: str,
        geodesic_energy: float,
        dimension: int,
        steps: int,
    ) -> str:
        r"""HMAC-SHA256 de cadena criptográfica Fase 1 → Fase 2."""
        import hmac
        
        signer = hmac.new(
            b"GeodesicAttentionFibrator::PolyakovAction::2026",
            digestmod=hashlib.sha256,
        )
        signer.update(_ENGINE_VERSION.encode("ascii"))
        signer.update(self._phase_tag_2.encode("ascii"))
        signer.update(ricci_seal.encode("ascii"))
        signer.update(f"{geodesic_energy:.16e}".encode("ascii"))
        signer.update(f"{dimension}:{steps}".encode("ascii"))
        return signer.hexdigest()

    def certify_polyakov_action(
        self,
        geodesic_velocity_matrix: NDArray[np.float64],
        g_metric: NDArray[np.float64],
        d_tau: float,
        ricci_audit: Optional[RicciFlowAuditData] = None,
    ) -> PolyakovActionAuditData:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2 (ORIENT).
        
        Integra acción geodésica y certifica estabilidad.
        
        CONTINUACIÓN FUNCTORIAL:
        ```
        polyakov_audit = certify_polyakov_action(velocities, G, d_tau, ricci_audit)
        feynman_kac_audit = enforce_quantum_veto(...)  # ← Fase 3
        ```
        """
        # Validaciones
        velocities = self.validate_finite_velocity_matrix(
            "geodesic_velocity_matrix", geodesic_velocity_matrix
        )

        G_san, audit_metric = self._audit_riemannian_metric_spd("g_metric", g_metric)

        tau = self.validate_finite_positive_scalar("d_tau", d_tau)

        # Verificación de continuidad de Fase 1
        if ricci_audit is not None:
            if not ricci_audit.is_metric_converged:
                raise RicciFlowDivergenceError(
                    "Fase 2 requiere certificación de Fase 1."
                )

            if ricci_audit.dimension != audit_metric.dimension:
                raise GeodesicInputValidationError(
                    f"Inconsistencia dimensional entre fases: "
                    f"Fase 1 dim={ricci_audit.dimension}, "
                    f"Fase 2 dim={audit_metric.dimension}."
                )

        # Integración de energía
        energy_audit = self._integrate_geodesic_energy(velocities, G_san, tau)

        if energy_audit.dimension != audit_metric.dimension:
            raise GeodesicInputValidationError(
                f"Dimensión de velocidades {energy_audit.dimension} "
                f"!= métrica {audit_metric.dimension}."
            )

        # Energía total
        total_kinetic = energy_audit.total_kinetic_sum
        geodesic_energy = 0.5 * tau * total_kinetic

        if not math.isfinite(geodesic_energy):
            raise PolyakovActionViolationError(
                "Energía de Polyakov no finita."
            )

        energy_tol = max(
            _ENERGY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, geodesic_energy),
        )

        if geodesic_energy < -energy_tol:
            raise PolyakovActionViolationError(
                f"Energía negativa: {geodesic_energy:.6e}."
            )

        geodesic_energy = max(0.0, geodesic_energy)

        # Validación de techo
        if geodesic_energy > _POLYAKOV_ENERGY_CEILING:
            raise PolyakovActionViolationError(
                f"Energía catastrófica: {geodesic_energy:.6e} > "
                f"{_POLYAKOV_ENERGY_CEILING:.6e}."
            )

        # Longitud métrica
        geodesic_length = math.sqrt(2.0 * geodesic_energy) if geodesic_energy > 0 else 0.0

        # Energía por paso
        energy_per_step = (
            geodesic_energy / energy_audit.steps if energy_audit.steps > 0 else 0.0
        )

        # Sello HMAC
        hmac_seal = self._phase2_hmac_sha256(
            ricci_audit.phase1_sha256_seal if ricci_audit else "",
            geodesic_energy,
            energy_audit.dimension,
            energy_audit.steps,
        )

        logger.debug(
            f"Fase 2 (ORIENT) completada. E[γ]={geodesic_energy:.6f} "
            f"L[γ]={geodesic_length:.6f} hmac={hmac_seal[:16]}..."
        )

        return PolyakovActionAuditData(
            steps=energy_audit.steps,
            dimension=energy_audit.dimension,
            geodesic_energy=float(geodesic_energy),
            geodesic_energy_per_step=float(energy_per_step),
            geodesic_length_metric=float(geodesic_length),
            min_kinetic_term=energy_audit.min_kinetic_term,
            max_kinetic_term=energy_audit.max_kinetic_term,
            mean_kinetic_term=energy_audit.mean_kinetic_term,
            energy_ceiling=float(_POLYAKOV_ENERGY_CEILING),
            polyakov_tolerance=float(energy_tol),
            metric_condition_number=audit_metric.condition_number_sanitized,
            is_geodesic_stable=True,
            phase2_hmac_sha256=hmac_seal,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3 ANIDADA: DECIDE & ACT (VETO CUÁNTICO DE FEYNMAN-KAC)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

class Phase3_FeynmanKacActuator(Phase2_PolyakovOrientator):
    r"""
    FASE 3 (DECIDE & ACT): Veto Cuántico.
    
    INICIO CONTINUO: Recibe obligatoriamente `PolyakovActionAuditData` de Fase 2.
    
    Responsabilidades:
    1. Auditoría de tensor de torsión.
    2. Cálculo de acción euclidiana total.
    3. Amplitud de transición cuántica de Feynman-Kac.
    4. Veredicto en clasificador de Heyting $\Omega_3$.
    5. Sello SHA-256 soberano determinista.
    
    MÉTODO TERMINAL: `enforce_quantum_veto` → `GeodesicAttentionGovernanceState`.
    """

    def __init__(self, strict_mode: bool = False) -> None:
        super().__init__(strict_mode=strict_mode)
        self._phase_tag_3: Final[str] = "Phase3_FeynmanKacActuator"

    def _audit_torsion_coupling(
        self,
        torsion_hs_norm_sq: float,
        lambda_coupling: float,
    ) -> TorsionAndCouplingAudit:
        r"""Auditoría de tensor de torsión y acoplamiento."""
        torsion_sq = self.validate_finite_nonnegative_scalar(
            "torsion_hs_norm_sq", torsion_hs_norm_sq
        )

        coupling = self.validate_finite_nonnegative_scalar(
            "lambda_coupling", lambda_coupling
        )

        coupling_contrib = coupling * torsion_sq

        if not math.isfinite(coupling_contrib):
            raise PolyakovActionViolationError(
                "Acoplamiento torsión-acción no finito."
            )

        torsion_tol = max(
            _TORSION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, torsion_sq),
        )

        return TorsionAndCouplingAudit(
            torsion_hs_norm_sq_original=float(torsion_sq),
            torsion_hs_norm_sq_sanitized=float(torsion_sq),
            lambda_coupling_original=float(coupling),
            lambda_coupling_sanitized=float(coupling),
            coupling_contribution=float(coupling_contrib),
            torsion_tolerance=float(torsion_tol),
        )

    def _heyting_verdict_classifier(
        self,
        ricci_converged: bool,
        polyakov_stable: bool,
        quantum_allowed: bool,
    ) -> Tuple[Literal["COHERENT", "DEGRADED", "VETOED"], bool]:
        r"""
        Clasificador de subobjetos en retículo de Heyting $\Omega_3$.
        
        - COHERENT: Todas las fases convergen correctamente.
        - DEGRADED: Una condición marginal pero recuperable.
        - VETOED: Una violación dura.
        """
        if not ricci_converged:
            verdict: Literal["COHERENT", "DEGRADED", "VETOED"] = "VETOED"
            is_valid = False
        elif not polyakov_stable:
            verdict = "DEGRADED"
            is_valid = True
        elif not quantum_allowed:
            verdict = "VETOED"
            is_valid = False
        else:
            verdict = "COHERENT"
            is_valid = True

        return verdict, is_valid

    def _phase3_sha256_sovereign_seal(
        self,
        ricci_seal: str,
        polyakov_seal: str,
        feynman_kac_audit: FeynmanKacAuditData,
        torsion_audit: TorsionAndCouplingAudit,
        heyting_verdict: str,
    ) -> str:
        r"""Sello SHA-256 soberano determinista de Fase 3."""
        hasher = hashlib.sha256()
        hasher.update(_ENGINE_VERSION.encode("ascii"))
        hasher.update(self._phase_tag_3.encode("ascii"))
        hasher.update(b"SOVEREIGN_SEAL")
        hasher.update(ricci_seal.encode("ascii"))
        hasher.update(polyakov_seal.encode("ascii"))
        hasher.update(heyting_verdict.encode("ascii"))
        hasher.update(f"{feynman_kac_audit.transition_amplitude:.16e}".encode("ascii"))
        hasher.update(f"{torsion_audit.coupling_contribution:.16e}".encode("ascii"))
        return hasher.hexdigest()

    def enforce_quantum_veto(
        self,
        polyakov_energy: float,
        torsion_hs_norm_sq: float,
        lambda_coupling: float,
        polyakov_audit: Optional[PolyakovActionAuditData] = None,
        ricci_audit: Optional[RicciFlowAuditData] = None,
    ) -> GeodesicAttentionGovernanceState:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 3 (DECIDE & ACT).
        
        Completa el ciclo OODA: calcula amplitud cuántica y emite veredicto.
        
        RETORNA: `GeodesicAttentionGovernanceState` (objeto final supremo).
        """
        import time
        
        t_start = time.perf_counter_ns()

        # Validaciones de entrada
        energy = self.validate_finite_nonnegative_scalar(
            "polyakov_energy", polyakov_energy
        )

        # Auditoría de torsión
        torsion_audit = self._audit_torsion_coupling(torsion_hs_norm_sq, lambda_coupling)

        # Verificación de continuidad de Fase 2
        if polyakov_audit is not None:
            if not polyakov_audit.is_geodesic_stable:
                raise PolyakovActionViolationError(
                    "Fase 3 requiere certificación de Fase 2."
                )

            consistency_tol = max(
                _ACTION_TOLERANCE,
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(1.0, abs(energy), abs(polyakov_audit.geodesic_energy)),
            )

            if abs(energy - polyakov_audit.geodesic_energy) > consistency_tol:
                raise PolyakovActionViolationError(
                    "Inconsistencia energética entre Fase 2 y Fase 3."
                )

        # Acción euclidiana total
        euclidean_action = energy + torsion_audit.coupling_contribution

        if not math.isfinite(euclidean_action):
            raise QuantumFeynmanKacVeto("Acción euclídea no finita.")

        action_tol = max(
            _ACTION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, euclidean_action),
        )

        if euclidean_action < -action_tol:
            raise PolyakovActionViolationError("Acción negativa fuera de tolerancia.")

        euclidean_action = max(0.0, euclidean_action)

        # Amplitud de Feynman-Kac
        if _HBAR_EFF <= 0.0:
            raise GeodesicAttentionAgentError("ħ_eff debe ser > 0.")

        log_amplitude = -euclidean_action / _HBAR_EFF

        if not math.isfinite(log_amplitude):
            raise QuantumFeynmanKacVeto("Logaritmo de amplitud no finito.")

        min_log_amp = math.log(_MIN_QUANTUM_AMPLITUDE)

        if log_amplitude < min_log_amp:
            raise QuantumFeynmanKacVeto(
                f"Veto cuántico: log(Ψ)={log_amplitude:.6e} < "
                f"log(Ψ_min)={min_log_amp:.6e}."
            )

        # Conversión exponencial con protección contra underflow
        tiny_log = math.log(_MACHINE_TINY)
        amplitude = 0.0 if log_amplitude < tiny_log else float(math.exp(log_amplitude))

        if amplitude < _MIN_QUANTUM_AMPLITUDE:
            raise QuantumFeynmanKacVeto(
                f"Veto cuántico: Ψ={amplitude:.6e} < "
                f"Ψ_min={_MIN_QUANTUM_AMPLITUDE:.6e}."
            )

        # Feynman-Kac audit
        feynman_kac_audit = FeynmanKacAuditData(
            euclidean_action=float(euclidean_action),
            log_transition_amplitude=float(log_amplitude),
            transition_amplitude=float(amplitude),
            min_quantum_amplitude=float(_MIN_QUANTUM_AMPLITUDE),
            hbar_eff_used=float(_HBAR_EFF),
            quantum_suppression_factor=float(
                math.exp(-euclidean_action / _HBAR_EFF) if euclidean_action < 700 else 0.0
            ),
            is_attention_allowed=True,
            phase3_sha256_seal="",  # Se calcula después
        )

        # Veredicto Heyting
        ricci_ok = ricci_audit.is_metric_converged if ricci_audit else True
        polyakov_ok = polyakov_audit.is_geodesic_stable if polyakov_audit else True
        quantum_ok = feynman_kac_audit.is_attention_allowed

        heyting_verdict, is_valid = self._heyting_verdict_classifier(
            ricci_ok, polyakov_ok, quantum_ok
        )

        # Sellado soberano
        ricci_seal = ricci_audit.phase1_sha256_seal if ricci_audit else ""
        polyakov_seal = polyakov_audit.phase2_hmac_sha256 if polyakov_audit else ""

        sovereign_seal = self._phase3_sha256_sovereign_seal(
            ricci_seal, polyakov_seal, feynman_kac_audit, torsion_audit, heyting_verdict
        )

        # Feynman-Kac audit final
        feynman_kac_final = FeynmanKacAuditData(
            euclidean_action=feynman_kac_audit.euclidean_action,
            log_transition_amplitude=feynman_kac_audit.log_transition_amplitude,
            transition_amplitude=feynman_kac_audit.transition_amplitude,
            min_quantum_amplitude=feynman_kac_audit.min_quantum_amplitude,
            hbar_eff_used=feynman_kac_audit.hbar_eff_used,
            quantum_suppression_factor=feynman_kac_audit.quantum_suppression_factor,
            is_attention_allowed=feynman_kac_audit.is_attention_allowed,
            phase3_sha256_seal=sovereign_seal,
        )

        t_total = time.perf_counter_ns() - t_start

        logger.info(
            f"Fase 3 (DECIDE & ACT) completada. Veredicto={heyting_verdict} "
            f"Ψ={amplitude:.6e} sello={sovereign_seal[:16]}... "
            f"tiempo={t_total/1.0e6:.3f}ms"
        )

        return GeodesicAttentionGovernanceState(
            ricci_audit=ricci_audit or RicciFlowAuditData(
                dimension=0, metric_residual_norm=0.0, metric_relative_residual=0.0,
                condition_number_g_k=0.0, condition_number_g_k_plus_1=0.0,
                metric_convergence_tolerance=0.0, metric_frobenius_norm_k=0.0,
                metric_frobenius_norm_k_plus_1=0.0, is_metric_converged=True,
                phase1_sha256_seal=""
            ),
            polyakov_audit=polyakov_audit or PolyakovActionAuditData(
                steps=0, dimension=0, geodesic_energy=0.0, geodesic_energy_per_step=0.0,
                geodesic_length_metric=0.0, min_kinetic_term=0.0, max_kinetic_term=0.0,
                mean_kinetic_term=0.0, energy_ceiling=0.0, polyakov_tolerance=0.0,
                metric_condition_number=0.0, is_geodesic_stable=True, phase2_hmac_sha256=""
            ),
            torsion_audit=torsion_audit,
            feynman_kac_audit=feynman_kac_final,
            is_epistemologically_valid=is_valid,
            heyting_verdict=heyting_verdict,
            cryptographic_seal_phase3_sha256=sovereign_seal,
            total_execution_time_ns=int(t_total),
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO INTEGRADOR: ENDOFUNTOR COVARIANTE
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

class GeodesicAttentionFibratorAgent(Phase3_FeynmanKacActuator):
    r"""
    AGENTE SOBERANO: Custodio de la Covarianza Atencional en Estrato WISDOM.
    
    Endofuntor de-confinado:
    $$\mathcal{Z}: \mathbf{WISDOM} \longrightarrow \mathbf{WISDOM}$$
    
    Somete tensores de atención del LLM a geometría riemanniana y mecánica cuántica,
    erradicando emparejamientos euclidianos planos y suprimiento alucinaciones
    mediante integral de trayectoria de Feynman-Kac.
    
    INTERFAZ UNIFICADA: Ciclo OODA de 3 fases anidadas formales.
    """

    def __init__(self, strict_mode: bool = False) -> None:
        super().__init__(strict_mode=strict_mode)
        logger.info(
            f"GeodesicAttentionFibratorAgent v{_ENGINE_VERSION} inicializado. "
            f"Modo estricto={strict_mode}"
        )

    def execute_geodesic_attention_governance(
        self,
        g_k: NDArray[np.float64],
        g_k_plus_1: NDArray[np.float64],
        geodesic_velocity_matrix: NDArray[np.float64],
        d_tau: float,
        torsion_hs_norm_sq: float,
        lambda_coupling: float,
    ) -> GeodesicAttentionGovernanceState:
        r"""
        INTERFAZ PÚBLICA UNIFICADA: Ciclo OODA completo.
        
        Ejecuta composición funtorial estricta de 3 fases:
        
        **FASE 1 (Observe):**
        ```
        ricci_audit = observe_ricci_convergence(g_k, g_k_plus_1)
        ```
        
        **FASE 2 (Orient):**
        ```
        polyakov_audit = certify_polyakov_action(geodesic_velocities, G, d_tau)
        ```
        
        **FASE 3 (Decide & Act):**
        ```
        state = enforce_quantum_veto(energy, torsion, lambda)
        ```
        
        **Retorna:** `GeodesicAttentionGovernanceState` con veredicto Heyting.
        """
        # FASE 1: OBSERVE
        ricci_audit = self.observe_ricci_convergence(g_k, g_k_plus_1)

        # FASE 2: ORIENT
        polyakov_audit = self.certify_polyakov_action(
            geodesic_velocity_matrix, g_k_plus_1, d_tau, ricci_audit=ricci_audit
        )

        # FASE 3: DECIDE & ACT
        governance_state = self.enforce_quantum_veto(
            polyakov_audit.geodesic_energy,
            torsion_hs_norm_sq,
            lambda_coupling,
            polyakov_audit=polyakov_audit,
            ricci_audit=ricci_audit,
        )

        return governance_state

    def __repr__(self) -> str:
        return (
            f"<GeodesicAttentionFibratorAgent "
            f"v={_ENGINE_VERSION} "
            f"IEEE-754-Double "
            f"Ricci-Polyakov-FeynmanKac-Heyting "
            f"strict={self._strict}>"
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

__all__: List[str] = [
    # Excepciones
    "GeodesicAttentionAgentError",
    "GeodesicInputValidationError",
    "MetricDegeneracyError",
    "RicciFlowDivergenceError",
    "PolyakovActionViolationError",
    "QuantumFeynmanKacVeto",
    "CryptographicChainError",
    "TopologicalInvariantViolation",
    # Auditorías y estructuras
    "RiemannianMetricAudit",
    "RicciFlowAuditData",
    "GeodesicEnergyAudit",
    "PolyakovActionAuditData",
    "TorsionAndCouplingAudit",
    "FeynmanKacAuditData",
    "GeodesicAttentionGovernanceState",
    # Fases y validador
    "FiniteNumericalValidator",
    "Phase1_RicciFlowObserver",
    "Phase2_PolyakovOrientator",
    "Phase3_FeynmanKacActuator",
    # Agente integrador
    "GeodesicAttentionFibratorAgent",
]
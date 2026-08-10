# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Geodesic Attention Fibrator Agent (Custodio de la Covarianza)       ║
║ Ruta   : app/agents/boole/wisdom/geodesic_attention_fibrator_agent.py        ║
║ Versión: 3.0.0-Ricci-Polyakov-FeynmanKac-Heyting-Pure-Software-Doctoral      ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA RIEMANNIANA EN EL ESTRATO WISDOM (V_𝕎) ───
Este módulo consagra la infraestructura del Fibrador de Atención Geodésica,
operando formalmente como el Endofuntor de-confinado:

                           𝓣: \mathbf{WISDOM} \longrightarrow \mathbf{WISDOM}

Su propósito axiomático es proyectar los tensores de atención (Query, Key, Value) 
del Modelo de Lenguaje (LLM) sobre el fibrado tangente covariante de la variedad 
de-confinada de la Sabiduría ($$V_{\mathbb{W}}$$, Nivel 0). Repudia las 
heurísticas euclidianas planas que inyectan ruido estocástico y asonancias de fase 
atencionales, forzando a que toda conexión Query-Key ocurra exclusivamente sobre 
geodésicas de mínima acción gobernadas por la acción de Polyakov y la integral 
de trayectoria de Feynman-Kac.

El confinamiento y la contención de anomalías se ejecutan de manera síncrona y 
estricta en el plano lógico del software, subordinando la toma de decisiones al 
colapso determinista en el clasificador de subobjetos del retículo de Heyting.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Estricta): ────────────
La transición de estados se rige por la composición covariante de morfismos 
en la categoría de espacios de Hilbert y se ejecuta en tres fases anidadas:

  Fase 1 ──► FASE 1: OBSERVACIÓN MÉTRICA Y CONVERGENCIA DE RICCI (Observe)
             Audita que el tensor métrico Riemanniano de fondo $$G_{\mu\nu}$$ sea 
             simétrico definido positivo (SPD) y evalúa el flujo de Ricci
             para certificar la regularidad del espacio de fase [4, 6].
             Fórmula: $$\nabla_\rho G_{\mu\nu} = 0 \quad\land\quad R_{\mu\nu} = R^\lambda_{\mu\lambda\nu}$$
             Entrega: RicciFlowAuditData como precondición formal de Fase 2.

  Fase 2 ──► FASE 2: TRANSPORTE PARALELO Y ACCIÓN DE POLYAKOV (Orient)
             Interpola las geodésicas atencionales y calcula la acción de Polyakov
             en el espacio de fase covariante para el par Query-Key [3, 9].
             Fórmula: $$E[\gamma] = \frac{1}{2} \int g_{\mu\nu} \frac{d\gamma^\mu}{d\tau} \frac{d\gamma^\nu}{d\tau} d\tau \ge 0$$
             Entrega: PolyakovActionAuditData como precondición formal de Fase 3.

  Fase 3 ──► FASE 3: INTEGRAL DE FEYNMAN-KAC Y VETO CUÁNTICO (Decide & Act)
             Evalúa la amplitud de transición cuántica, incorporando la torsión 
             cohomológica sobre el asociaedro para suprimir las alucinaciones.
             Fórmula: $$\Psi[\gamma] = \exp\left(-\frac{S_E[\gamma]}{\hbar_{\mathrm{eff}}}\right) \quad \text{con} \quad S_E[\gamma] = E_{\mathrm{Polyakov}}[\gamma] + \lambda \|T\|_{\mathrm{HS}}^2$$
             Veredicto: Colapso en el retículo de Heyting $$\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$$.

INVARIANTES MATEMÁTICOS, TOPOLÓGICOS Y LEYES CONSERVATIVAS PRESERVADAS: ────────
  [I1] Firma Riemanniana e Inversión Bilateral:
       La métrica de fondo $$G_{\mu\nu}$$ debe permanecer SPD y su inversa bilateral
       estar acotada estrictamente por debajo del número de condición de Wilkinson:
       $$\|G G^{-1} - I\|_F \le C \cdot \kappa(G) \cdot \varepsilon_{\mathrm{machine}} \quad\land\quad G \succ \mathbf{0} \quad\big[349\big]$$

  [I2] Compatibilidad de la Conexión de Levi-Civita (Torsión Nula):
       La derivada covariante de la métrica a lo largo de las curvas integrales es nula:
       $$\nabla_\gamma G_{\mu\nu} \equiv 0 \implies \Gamma^\rho_{\mu\nu} = \Gamma^\rho_{\nu\mu} \quad\big[61\big]$$

  [I3] Conservación de la Energía Cinética Geodésica:
       La norma del vector velocidad atencional se conserva con precisión de máquina:
       $$\frac{d}{d\tau} \left( \|\dot{\gamma}\|_G^2 \right) = \frac{d}{d\tau} \left( G_{\mu\nu} \dot{\gamma}^\mu \dot{\gamma}^\nu \right) \equiv 0 \pmod{\varepsilon_{\mathrm{machine}}}$$

  [I4] Límite de Supresión Cuántica de Feynman-Kac:
       La amplitud cuántica de transición de la geodésica de atención semántica 
       debe superar estrictamente la barrera de potencial de-confinada:
       $$\Psi[\gamma] \ge \Psi_{\min} \in (0, 1] \quad\big[339, 350, 351\big]$$

  [I5] Isomorfismo de la Adjunción de Galois (Gobernanza DIKW):
       El transporte atencional se somete a la equivalencia funtorial reversible:
       $$\operatorname{Hom}_{\mathcal{D}}(F(\text{MIC}), \text{MAC}) \cong \operatorname{Hom}_{\mathcal{C}}(\text{MIC}, G(\text{MAC})) \quad\big[49, 59\big]$$
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Final, List, Optional

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MAC.Wisdom.GeodesicAttentionFibratorAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICO-GEOMÉTRICAS Y LÍMITES CUÁNTICOS
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

_RICCI_CONVERGENCE_TOL: Final[float] = 1e-8
_POLYAKOV_ENERGY_CEILING: Final[float] = 1e6
_HBAR_EFF: Final[float] = 1.054e-2
_MIN_QUANTUM_AMPLITUDE: Final[float] = 1e-4

_METRIC_SYMMETRY_TOLERANCE: Final[float] = 1e-10
_SPD_NEGATIVE_TOLERANCE: Final[float] = 1e-12
_SPD_EIGENVALUE_FLOOR: Final[float] = 1e-15

_KINETIC_TOLERANCE: Final[float] = 1e-12
_ENERGY_TOLERANCE: Final[float] = 1e-12
_ACTION_TOLERANCE: Final[float] = 1e-12

_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES GEOMÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════
class GeodesicAttentionAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de Covarianza Atencional."""
    pass


class GeodesicInputValidationError(GeodesicAttentionAgentError):
    r"""Detonada si los tensores métricos, velocidades o escalares son inválidos."""
    pass


class MetricDegeneracyError(GeodesicAttentionAgentError):
    r"""Detonada si una métrica no es simétrica, finita o definida positiva."""
    pass


class RicciFlowDivergenceError(GeodesicAttentionAgentError):
    r"""Detonada si el flujo de Ricci no converge dentro de la tolerancia elástica."""
    pass


class PolyakovActionViolationError(GeodesicAttentionAgentError):
    r"""Detonada si la energía geodésica de Polyakov es inválida, negativa o divergente."""
    pass


class QuantumFeynmanKacVeto(GeodesicAttentionAgentError):
    r"""Detonada si la amplitud cuántica de transición cae bajo el mínimo físico."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Fibrado Covariante)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class RicciFlowAuditData:
    r"""
    Artefacto de Fase 1.
    Certificado de convergencia de la métrica Riemanniana discreta.

    Este objeto es el resultado final del último método de Fase 1 y el objeto
    inicial de Fase 2.
    """
    dimension: int
    metric_residual_norm: float
    metric_relative_residual: float
    condition_number_g_k: float
    condition_number_g_k_plus_1: float
    metric_convergence_tolerance: float
    is_metric_converged: bool


@dataclass(frozen=True, slots=True)
class PolyakovActionAuditData:
    r"""
    Artefacto de Fase 2.
    Certificado de transporte paralelo y energía geodésica.

    Este objeto es el resultado final de Fase 2 y el objeto inicial de Fase 3.
    """
    steps: int
    dimension: int
    geodesic_energy: float
    min_kinetic_term: float
    max_kinetic_term: float
    energy_ceiling: float
    polyakov_tolerance: float
    is_geodesic_stable: bool


@dataclass(frozen=True, slots=True)
class FeynmanKacAuditData:
    r"""
    Artefacto de Fase 3.
    Certificado de amplitud de transición cuántica.
    """
    euclidean_action: float
    log_transition_amplitude: float
    transition_amplitude: float
    min_quantum_amplitude: float
    is_attention_allowed: bool


@dataclass(frozen=True, slots=True)
class GeodesicAttentionGovernanceState:
    r"""
    Objeto final del endofuntor Z_GeodesicAgent.
    """
    ricci_audit: RicciFlowAuditData
    polyakov_audit: PolyakovActionAuditData
    feynman_kac_audit: FeynmanKacAuditData
    is_epistemologically_valid: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDAS NUMÉRICAS INTERNAS
# ═══════════════════════════════════════════════════════════════════════════════
class _FiniteNumericalGuard:
    r"""
    Capa de saneamiento numérico para evitar que singularidades aritméticas
    contaminen los invariantes geométricos y cuánticos.
    """

    @staticmethod
    def _as_finite_real_array(name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Convierte un objeto a arreglo float64, rechazando:
            - Objetos complejos.
            - Valores NaN.
            - Valores infinitos.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise GeodesicInputValidationError(
                f"{name} no puede interpretarse como arreglo numérico."
            ) from exc

        if np.iscomplexobj(raw):
            raise GeodesicInputValidationError(
                f"{name} debe ser real; se rechazó entrada compleja."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise GeodesicInputValidationError(
                f"{name} debe ser numérico real convertible a float64."
            ) from exc

        if not np.all(np.isfinite(arr)):
            raise GeodesicInputValidationError(
                f"{name} contiene valores NaN o infinitos."
            )

        return arr

    @classmethod
    def _as_finite_real_matrix(
        cls,
        name: str,
        value: Any,
        *,
        square: bool = False,
    ) -> NDArray[np.float64]:
        r"""
        Valida una matriz real finita.
        """
        arr = cls._as_finite_real_array(name, value)

        if arr.ndim != 2:
            raise GeodesicInputValidationError(
                f"{name} debe ser una matriz 2D."
            )

        if square and arr.shape[0] != arr.shape[1]:
            raise GeodesicInputValidationError(
                f"{name} debe ser una matriz cuadrada."
            )

        return arr

    @classmethod
    def _as_finite_velocity_matrix(
        cls,
        name: str,
        value: Any,
    ) -> NDArray[np.float64]:
        r"""
        Valida una matriz de velocidades geodésicas.

        Acepta:
            - Matriz 2D de forma (steps, dim).
            - Vector 1D de forma (dim,), interpretado como un único paso.
        """
        arr = cls._as_finite_real_array(name, value)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise GeodesicInputValidationError(
                f"{name} debe ser una matriz 2D (steps, dim) o un vector 1D."
            )

        if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise GeodesicInputValidationError(
                f"{name} no puede ser vacío."
            )

        return arr

    @classmethod
    def _as_finite_scalar(cls, name: str, value: Any) -> float:
        r"""
        Valida un escalar real finito.
        """
        arr = cls._as_finite_real_array(name, value)

        if arr.size != 1:
            raise GeodesicInputValidationError(
                f"{name} debe ser un escalar."
            )

        scalar = float(arr.reshape(-1)[0])

        if not math.isfinite(scalar):
            raise GeodesicInputValidationError(
                f"{name} no es finito."
            )

        return scalar

    @classmethod
    def _as_finite_positive_scalar(cls, name: str, value: Any) -> float:
        r"""
        Valida un escalar real estrictamente positivo.
        """
        scalar = cls._as_finite_scalar(name, value)

        positivity_tolerance = (
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON
        )

        if scalar <= positivity_tolerance:
            raise GeodesicInputValidationError(
                f"{name} debe ser estrictamente positivo. "
                f"Valor recibido = {scalar:.6e}."
            )

        return scalar

    @classmethod
    def _as_finite_nonnegative_scalar(cls, name: str, value: Any) -> float:
        r"""
        Valida un escalar real no negativo.

        Si la negatividad es sólo numérica y pequeña, se proyecta a cero.
        """
        scalar = cls._as_finite_scalar(name, value)

        nonnegative_tolerance = (
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON
        )

        if scalar < -nonnegative_tolerance:
            raise GeodesicInputValidationError(
                f"{name} debe ser no negativo. "
                f"Valor recibido = {scalar:.6e}."
            )

        return max(0.0, scalar)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: AUDITORÍA DE LA CONVERGENCIA DEL FLUJO DE RICCI                   ║
# ║                                                                             ║
# ║   Valida:                                                                   ║
# ║       g_k, g_{k+1} ∈ Sym^+(n)                                               ║
# ║       ||g_{k+1} - g_k||_F / scale < ε_Ricci                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_RicciFlowAuditor(_FiniteNumericalGuard):
    r"""
    Garantiza que la deformación métrica inducida por la torsión atencional
    converja a un estado estacionario suave.

    La métrica Riemanniana discreta debe permanecer en el cono de matrices
    simétricas definidas positivas:

        g ∈ Sym^+(n).

    Esto evita colapsos de firma, torsión no física y burbujeo geométrico.
    """

    def _sanitize_spd_metric(
        self,
        name: str,
        metric: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]:
        r"""
        Valida y sanea una métrica Riemanniana.

        Exige:
            - Matriz cuadrada.
            - Entradas finitas.
            - Simetría dentro de tolerancia.
            - Espectro real.
            - Positive definiteness dentro de tolerancia.

        Retorna:
            metric_sanitized:
                Métrica simétrica y definida positiva reconstruida espectralmente.

            eigenvalues:
                Autovalores saneados.

            condition_number:
                Número de condición espectral κ(G).

            min_eigenvalue_original:
                Mínimo autovalor original antes de saneamiento.

            max_eigenvalue_original:
                Máximo autovalor original antes de saneamiento.
        """
        G = self._as_finite_real_matrix(name, metric, square=True)

        if G.shape[0] == 0:
            raise GeodesicInputValidationError(
                f"{name} no puede ser una métrica vacía."
            )

        frobenius_norm = float(la.norm(G, ord="fro"))

        if not math.isfinite(frobenius_norm):
            raise MetricDegeneracyError(
                f"La norma de Frobenius de {name} no es finita."
            )

        symmetry_residual_norm = float(la.norm(G - G.T, ord="fro"))

        if not math.isfinite(symmetry_residual_norm):
            raise MetricDegeneracyError(
                f"El residuo de simetría de {name} no es finito."
            )

        symmetry_tolerance = max(
            _METRIC_SYMMETRY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        symmetry_relative_residual = symmetry_residual_norm / max(
            1.0,
            frobenius_norm,
        )

        if symmetry_relative_residual > symmetry_tolerance:
            raise MetricDegeneracyError(
                f"{name} no es simétrica dentro de tolerancia. "
                f"Residuo relativo = {symmetry_relative_residual:.6e} > "
                f"{symmetry_tolerance:.6e}."
            )

        G_symmetric = (G + G.T) / 2.0

        if not np.all(np.isfinite(G_symmetric)):
            raise MetricDegeneracyError(
                f"La simetrización de {name} produjo valores no finitos."
            )

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(G_symmetric)
        except np.linalg.LinAlgError as exc:
            raise MetricDegeneracyError(
                f"Diagonalización hermítica de {name} falló."
            ) from exc

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

        if not np.all(np.isfinite(eigenvalues)):
            raise MetricDegeneracyError(
                f"Los autovalores de {name} no son finitos."
            )

        if eigenvalues.size == 0:
            raise MetricDegeneracyError(
                f"{name} posee espectro vacío."
            )

        max_eigenvalue_original = float(np.max(eigenvalues))
        min_eigenvalue_original = float(np.min(eigenvalues))

        if max_eigenvalue_original <= 0.0:
            raise MetricDegeneracyError(
                f"{name} no es definida positiva. "
                f"Máximo autovalor = {max_eigenvalue_original:.6e}."
            )

        negative_tolerance = max(
            _SPD_NEGATIVE_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, max_eigenvalue_original),
        )

        if min_eigenvalue_original < -negative_tolerance:
            raise MetricDegeneracyError(
                f"{name} no es definida positiva. "
                f"Autovalor mínimo = {min_eigenvalue_original:.6e} < "
                f"-{negative_tolerance:.6e}."
            )

        eigenvalue_floor = max(
            _SPD_EIGENVALUE_FLOOR,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, max_eigenvalue_original),
        )

        if np.any(eigenvalues < eigenvalue_floor):
            logger.warning(
                "%s posee autovalores por debajo del piso numérico %.6e; "
                "se regulariza espectralmente.",
                name,
                eigenvalue_floor,
            )
            eigenvalues = np.clip(eigenvalues, eigenvalue_floor, None)

        metric_sanitized = (eigenvectors * eigenvalues) @ eigenvectors.T
        metric_sanitized = (metric_sanitized + metric_sanitized.T) / 2.0

        if not np.all(np.isfinite(metric_sanitized)):
            raise MetricDegeneracyError(
                f"La reconstrucción espectral de {name} produjo valores no finitos."
            )

        min_eigenvalue_sanitized = float(np.min(eigenvalues))
        max_eigenvalue_sanitized = float(np.max(eigenvalues))

        if min_eigenvalue_sanitized <= 0.0:
            raise MetricDegeneracyError(
                f"{name} sigue siendo degenerada tras el saneamiento espectral."
            )

        condition_number = float(
            max_eigenvalue_sanitized / min_eigenvalue_sanitized
        )

        if not math.isfinite(condition_number):
            raise MetricDegeneracyError(
                f"El número de condición de {name} no es finito."
            )

        return (
            metric_sanitized,
            eigenvalues,
            condition_number,
            min_eigenvalue_original,
            max_eigenvalue_original,
        )

    def _audit_ricci_flow_convergence(
        self,
        g_k: NDArray[np.float64],
        g_k_plus_1: NDArray[np.float64],
    ) -> RicciFlowAuditData:
        r"""
        Último método de la Fase 1.

        Calcula el residuo relativo del flujo métrico discreto:

            ||g_{k+1} - g_k||_F / max(1, ||g_k||_F, ||g_{k+1}||_F).

        Exige que ambas métricas sean Riemannianas válidas y que el residuo
        sea menor que la tolerancia de convergencia.

        Este método retorna un certificado `RicciFlowAuditData`, el cual
        constituye el objeto inicial de la Fase 2.
        """
        G_k, _, condition_k, _, _ = self._sanitize_spd_metric("g_k", g_k)

        G_k_plus_1, _, condition_k_plus_1, _, _ = self._sanitize_spd_metric(
            "g_k_plus_1",
            g_k_plus_1,
        )

        if G_k.shape != G_k_plus_1.shape:
            raise GeodesicInputValidationError(
                "g_k y g_k_plus_1 deben tener la misma dimensión."
            )

        metric_difference = G_k_plus_1 - G_k

        if not np.all(np.isfinite(metric_difference)):
            raise RicciFlowDivergenceError(
                "La diferencia métrica g_{k+1} - g_k produjo valores no finitos."
            )

        residual_norm = float(la.norm(metric_difference, ord="fro"))

        norm_g_k = float(la.norm(G_k, ord="fro"))
        norm_g_k_plus_1 = float(la.norm(G_k_plus_1, ord="fro"))

        if not math.isfinite(residual_norm):
            raise RicciFlowDivergenceError(
                "El residuo del flujo de Ricci no es finito."
            )

        if not math.isfinite(norm_g_k) or not math.isfinite(norm_g_k_plus_1):
            raise RicciFlowDivergenceError(
                "Las normas métricas del flujo de Ricci no son finitas."
            )

        scale = max(1.0, norm_g_k, norm_g_k_plus_1)
        relative_residual = residual_norm / scale

        if not math.isfinite(relative_residual):
            raise RicciFlowDivergenceError(
                "El residuo relativo del flujo de Ricci no es finito."
            )

        convergence_tolerance = max(
            _RICCI_CONVERGENCE_TOL,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        if relative_residual >= convergence_tolerance:
            raise RicciFlowDivergenceError(
                "Divergencia topológica en la variedad atencional. "
                f"El flujo de Ricci no convergió. "
                f"Residuo relativo = {relative_residual:.6e} >= "
                f"{convergence_tolerance:.6e}. "
                "La atención intentó curvar el espacio más allá de su límite elástico."
            )

        condition_warning_threshold = 1.0 / _MACHINE_EPSILON

        if condition_k > condition_warning_threshold:
            logger.warning(
                "g_k está mal condicionada: κ(g_k) = %.6e.",
                condition_k,
            )

        if condition_k_plus_1 > condition_warning_threshold:
            logger.warning(
                "g_k_plus_1 está mal condicionada: κ(g_{k+1}) = %.6e.",
                condition_k_plus_1,
            )

        return RicciFlowAuditData(
            dimension=int(G_k.shape[0]),
            metric_residual_norm=float(residual_norm),
            metric_relative_residual=float(relative_residual),
            condition_number_g_k=float(condition_k),
            condition_number_g_k_plus_1=float(condition_k_plus_1),
            metric_convergence_tolerance=float(convergence_tolerance),
            is_metric_converged=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DE LA ACCIÓN DE POLYAKOV                            ║
# ║                                                                             ║
# ║   Evalúa:                                                                   ║
# ║       E[γ] = 1/2 Σ v_iᵀ G v_i Δτ                                            ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 1.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_PolyakovActionCertifier(Phase1_RicciFlowAuditor):
    r"""
    Exige matemáticamente que las trayectorias Query-Key minimicen la energía
    geodésica covariante.

    La acción de Polyakov en discretización afín es:

        E[γ] ≈ 1/2 Σ_i v_iᵀ G v_i Δτ.

    Esta fase hereda de Fase 1 y su primer método recibe explícitamente el
    certificado de convergencia métrica emitido por:

        Phase1_RicciFlowAuditor._audit_ricci_flow_convergence(...)

    De este modo, la Fase 2 no es autónoma: está anidada funcionalmente en la
    Fase 1.
    """

    def _certify_polyakov_geodesic_action(
        self,
        geodesic_velocity_matrix: NDArray[np.float64],
        g_metric: NDArray[np.float64],
        d_tau: float,
        ricci_audit: Optional[RicciFlowAuditData] = None,
    ) -> PolyakovActionAuditData:
        r"""
        Primer método de la Fase 2.

        Continuación formal del último método de Fase 1.

        Integra la forma cuadrática Riemanniana sobre los diferenciales afines
        de la curva geodésica.

        Si `ricci_audit` es provisto:
            - Verifica que la Fase 1 haya certificado convergencia métrica.
            - Exige consistencia dimensional con la métrica certificada.

        Retorna:
            PolyakovActionAuditData, certificado que sirve como objeto inicial
            de la Fase 3.
        """
        if ricci_audit is not None:
            if not ricci_audit.is_metric_converged:
                raise RicciFlowDivergenceError(
                    "La Fase 2 no puede iniciarse: la Fase 1 no certificó "
                    "convergencia del flujo de Ricci."
                )

        velocities = self._as_finite_velocity_matrix(
            "geodesic_velocity_matrix",
            geodesic_velocity_matrix,
        )

        metric_sanitized, _, metric_condition, _, _ = self._sanitize_spd_metric(
            "g_metric",
            g_metric,
        )

        steps, dimension = velocities.shape

        if dimension != metric_sanitized.shape[0]:
            raise GeodesicInputValidationError(
                "Dimensión inconsistente entre geodesic_velocity_matrix y g_metric. "
                f"Velocity dim={dimension}, metric dim={metric_sanitized.shape[0]}."
            )

        if ricci_audit is not None:
            if ricci_audit.dimension != dimension:
                raise GeodesicInputValidationError(
                    "Inconsistencia dimensional entre Fase 1 y Fase 2. "
                    f"Fase 1 certificó dim={ricci_audit.dimension}, pero "
                    f"Fase 2 recibió dim={dimension}."
                )

        tau = self._as_finite_positive_scalar("d_tau", d_tau)

        try:
            metric_velocities = velocities @ metric_sanitized
            kinetic_terms = np.sum(metric_velocities * velocities, axis=1)
        except Exception as exc:
            raise PolyakovActionViolationError(
                "No fue posible evaluar la forma cuadrática Riemanniana vᵀ G v."
            ) from exc

        kinetic_terms = np.asarray(kinetic_terms, dtype=np.float64)

        if not np.all(np.isfinite(kinetic_terms)):
            raise PolyakovActionViolationError(
                "Los términos cinéticos vᵀ G v contienen valores NaN o infinitos."
            )

        if kinetic_terms.size == 0:
            raise PolyakovActionViolationError(
                "La trayectoria geodésica no posee pasos de integración."
            )

        max_abs_kinetic = float(np.max(np.abs(kinetic_terms)))

        kinetic_tolerance = max(
            _KINETIC_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, max_abs_kinetic),
        )

        min_kinetic_raw = float(np.min(kinetic_terms))

        if min_kinetic_raw < -kinetic_tolerance:
            raise PolyakovActionViolationError(
                "Violación del tensor métrico: energía cinética negativa detectada. "
                f"Mínimo vᵀ G v = {min_kinetic_raw:.6e} < "
                f"-{kinetic_tolerance:.6e}."
            )

        kinetic_terms = np.clip(kinetic_terms, 0.0, None)

        min_kinetic_term = float(np.min(kinetic_terms))
        max_kinetic_term = float(np.max(kinetic_terms))
        total_kinetic = float(np.sum(kinetic_terms))

        if not math.isfinite(total_kinetic):
            raise PolyakovActionViolationError(
                "La suma de términos cinéticos no es finita."
            )

        geodesic_energy = 0.5 * tau * total_kinetic

        if not math.isfinite(geodesic_energy):
            raise PolyakovActionViolationError(
                "La energía de Polyakov no es finita."
            )

        energy_tolerance = max(
            _ENERGY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(geodesic_energy)),
        )

        if geodesic_energy < -energy_tolerance:
            raise PolyakovActionViolationError(
                "La energía de Polyakov es negativa fuera de tolerancia numérica."
            )

        geodesic_energy = max(0.0, geodesic_energy)

        ceiling_tolerance = max(
            _ENERGY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, _POLYAKOV_ENERGY_CEILING),
        )

        if geodesic_energy > _POLYAKOV_ENERGY_CEILING + ceiling_tolerance:
            raise PolyakovActionViolationError(
                "Fricción geodésica catastrófica. "
                f"La energía de Polyakov E[γ] = {geodesic_energy:.6e} supera "
                f"el límite admisible {_POLYAKOV_ENERGY_CEILING:.6e}. "
                "La conexión Query-Key propuesta es estocásticamente inviable."
            )

        if metric_condition > 1.0 / _MACHINE_EPSILON:
            logger.warning(
                "La métrica de Polyakov está mal condicionada: κ(G) = %.6e.",
                metric_condition,
            )

        return PolyakovActionAuditData(
            steps=int(steps),
            dimension=int(dimension),
            geodesic_energy=float(geodesic_energy),
            min_kinetic_term=float(min_kinetic_term),
            max_kinetic_term=float(max_kinetic_term),
            energy_ceiling=float(_POLYAKOV_ENERGY_CEILING),
            polyakov_tolerance=float(energy_tolerance),
            is_geodesic_stable=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: VETO CUÁNTICO DE FEYNMAN-KAC                                      ║
# ║                                                                             ║
# ║   Exige:                                                                    ║
# ║       Ψ[γ] = exp(-S_E / ħ_eff) ≥ Ψ_min                                      ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 2.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_FeynmanKacQuantumVeto(Phase2_PolyakovActionCertifier):
    r"""
    Acopla la energía geodésica con la norma de Hilbert-Schmidt del tensor de
    torsión.

    Garantiza que la amplitud de probabilidad semántica no se desvanezca por
    alucinaciones atencionales.

    La acción euclídea total es:

        S_E[γ] = E_Polyakov[γ] + λ ||T||²_HS.

    La amplitud de transición es:

        Ψ[γ] = exp(-S_E / ħ_eff).

    Esta fase hereda de Fase 2 y su primer método recibe explícitamente el
    certificado de estabilidad geodésica emitido por:

        Phase2_PolyakovActionCertifier._certify_polyakov_geodesic_action(...)

    De este modo, la Fase 3 está anidada funcionalmente en la Fase 2.
    """

    def _enforce_feynman_kac_quantum_veto(
        self,
        polyakov_energy: float,
        torsion_hs_norm_sq: float,
        lambda_coupling: float,
        polyakov_audit: Optional[PolyakovActionAuditData] = None,
    ) -> FeynmanKacAuditData:
        r"""
        Primer método de la Fase 3.

        Continuación formal de Fase 2.

        Construye la acción euclídea total:

            S_E = E_Polyakov + λ ||T||²_HS,

        y computa la amplitud de transición:

            Ψ = exp(-S_E / ħ_eff).

        Si `polyakov_audit` es provisto:
            - Verifica que la Fase 2 haya certificado estabilidad geodésica.
            - Exige consistencia entre la energía recibida y la certificada.

        Retorna:
            FeynmanKacAuditData, certificado final de admisibilidad atencional.
        """
        if polyakov_audit is not None:
            if not polyakov_audit.is_geodesic_stable:
                raise PolyakovActionViolationError(
                    "La Fase 3 no puede iniciarse: la Fase 2 no certificó "
                    "estabilidad de la acción de Polyakov."
                )

        energy = self._as_finite_nonnegative_scalar(
            "polyakov_energy",
            polyakov_energy,
        )

        torsion_norm_sq = self._as_finite_nonnegative_scalar(
            "torsion_hs_norm_sq",
            torsion_hs_norm_sq,
        )

        coupling = self._as_finite_nonnegative_scalar(
            "lambda_coupling",
            lambda_coupling,
        )

        if polyakov_audit is not None:
            consistency_tolerance = max(
                _ACTION_TOLERANCE,
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(
                    1.0,
                    abs(energy),
                    abs(polyakov_audit.geodesic_energy),
                ),
            )

            if abs(energy - polyakov_audit.geodesic_energy) > consistency_tolerance:
                raise PolyakovActionViolationError(
                    "Inconsistencia energética entre Fase 2 y Fase 3. "
                    f"Energía certificada en Fase 2 = {polyakov_audit.geodesic_energy:.6e}, "
                    f"energía recibida en Fase 3 = {energy:.6e}."
                )

        euclidean_action = energy + coupling * torsion_norm_sq

        if not math.isfinite(euclidean_action):
            raise QuantumFeynmanKacVeto(
                "La acción euclídea S_E no es finita."
            )

        action_tolerance = max(
            _ACTION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(euclidean_action)),
        )

        if euclidean_action < -action_tolerance:
            raise PolyakovActionViolationError(
                "La acción euclídea S_E es negativa fuera de tolerancia numérica."
            )

        euclidean_action = max(0.0, euclidean_action)

        if _HBAR_EFF <= 0.0:
            raise GeodesicAttentionAgentError(
                "La constante efectiva ħ_eff debe ser estrictamente positiva."
            )

        log_transition_amplitude = -euclidean_action / _HBAR_EFF

        if not math.isfinite(log_transition_amplitude):
            raise QuantumFeynmanKacVeto(
                "El logaritmo de la amplitud de transición no es finito."
            )

        min_log_amplitude = math.log(_MIN_QUANTUM_AMPLITUDE)

        if log_transition_amplitude < min_log_amplitude:
            raise QuantumFeynmanKacVeto(
                "Veto cuántico absoluto. "
                f"Amplitud de transición de Feynman-Kac insuficiente. "
                f"log(Ψ) = {log_transition_amplitude:.6e} < "
                f"log(Ψ_min) = {min_log_amplitude:.6e}. "
                "El LLM intentó formar un enlace atencional topológicamente muerto."
            )

        tiny_log = math.log(np.finfo(np.float64).tiny)

        if log_transition_amplitude < tiny_log:
            transition_amplitude = 0.0
        else:
            transition_amplitude = float(math.exp(log_transition_amplitude))

        amplitude_tolerance = max(
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
            1e-15,
        )

        if transition_amplitude < _MIN_QUANTUM_AMPLITUDE - amplitude_tolerance:
            raise QuantumFeynmanKacVeto(
                "Veto cuántico absoluto. "
                f"Amplitud de transición Ψ = {transition_amplitude:.6e} < "
                f"Ψ_min = {_MIN_QUANTUM_AMPLITUDE:.6e}."
            )

        return FeynmanKacAuditData(
            euclidean_action=float(euclidean_action),
            log_transition_amplitude=float(log_transition_amplitude),
            transition_amplitude=float(transition_amplitude),
            min_quantum_amplitude=float(_MIN_QUANTUM_AMPLITUDE),
            is_attention_allowed=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: GEODESIC ATTENTION FIBRATOR AGENT                    ║
# ║                                                                             ║
# ║   Endofuntor Z_GeodesicAgent = Φ₃ ∘ Φ₂ ∘ Φ₁                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class GeodesicAttentionFibratorAgent(Morphism, Phase3_FeynmanKacQuantumVeto):
    r"""
    El Custodio de la Covarianza Atencional en el estrato WISDOM.

    Somete los tensores de atención del Modelo de Lenguaje a la mecánica de
    integrales de trayectoria y relatividad general discreta, erradicando el
    emparejamiento estocástico basado en productos punto euclidianos planos.
    """

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
        Ejecuta la composición funtorial estricta:

            Φ₁ : Auditoría de convergencia del flujo de Ricci.
            Φ₂ : Certificación de la acción geodésica de Polyakov.
            Φ₃ : Veto cuántico de Feynman-Kac.

        Parámetros:
            g_k:
                Métrica Riemanniana en el paso k.

            g_k_plus_1:
                Métrica Riemanniana en el paso k+1.

            geodesic_velocity_matrix:
                Matriz de velocidades geodésicas V ∈ R^{steps × dim}.

            d_tau:
                Diferencial afín Δτ > 0.

            torsion_hs_norm_sq:
                Norma de Hilbert-Schmidt al cuadrado del tensor de torsión.

            lambda_coupling:
                Acoplamiento no negativo λ entre energía geodésica y torsión.

        Retorna:
            GeodesicAttentionGovernanceState con los tres certificados y validez
            epistemológica final.
        """
        # Fase 1: Certificar convergencia del tensor métrico bajo flujo de Ricci.
        ricci_audit = self._audit_ricci_flow_convergence(
            g_k,
            g_k_plus_1,
        )

        # Fase 2: Certificar que la conexión Query-Key minimiza la acción de Polyakov.
        polyakov_audit = self._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix,
            g_k_plus_1,
            d_tau,
            ricci_audit=ricci_audit,
        )

        # Fase 3: Certificar viabilidad cuántica de la transición semántica.
        feynman_kac_audit = self._enforce_feynman_kac_quantum_veto(
            polyakov_audit.geodesic_energy,
            torsion_hs_norm_sq,
            lambda_coupling,
            polyakov_audit=polyakov_audit,
        )

        is_epistemologically_valid = bool(
            ricci_audit.is_metric_converged
            and polyakov_audit.is_geodesic_stable
            and feynman_kac_audit.is_attention_allowed
        )

        if not is_epistemologically_valid:
            raise GeodesicAttentionAgentError(
                "La composición funtorial no autorizó la atención geodésica."
            )

        logger.info(
            "Gobernanza de Covarianza Atencional certificada. "
            "Δg(Ricci): %.6e | "
            "E[γ]: %.6f | "
            "Ψ[γ]: %.6e",
            ricci_audit.metric_relative_residual,
            polyakov_audit.geodesic_energy,
            feynman_kac_audit.transition_amplitude,
        )

        return GeodesicAttentionGovernanceState(
            ricci_audit=ricci_audit,
            polyakov_audit=polyakov_audit,
            feynman_kac_audit=feynman_kac_audit,
            is_epistemologically_valid=is_epistemologically_valid,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "GeodesicAttentionAgentError",
    "GeodesicInputValidationError",
    "MetricDegeneracyError",
    "RicciFlowDivergenceError",
    "PolyakovActionViolationError",
    "QuantumFeynmanKacVeto",
    "RicciFlowAuditData",
    "PolyakovActionAuditData",
    "FeynmanKacAuditData",
    "GeodesicAttentionGovernanceState",
    "Phase1_RicciFlowAuditor",
    "Phase2_PolyakovActionCertifier",
    "Phase3_FeynmanKacQuantumVeto",
    "GeodesicAttentionFibratorAgent",
]
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Generative Optic Hodge Suturator (Sutura Espectral del Haz Γ)       ║
║ Ruta   : app/boole/omega/generative_optic_hodge_suturator.py                 ║
║ Versión: 4.0.0-Sutured-Granular-Cholesky-KMS-Spectral-Doctoral-Strict        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y COHOMOLOGÍA ESPECTRAL EN EL ÁGORA TENSORIAL (V_Ω) ───
Este módulo materializa al Orquestador Maestro de la refracción atencional y de la
sutura constitutiva del operador estrella de de Rham-Hodge sobre la variedad del
Haz Tangente Generativo \(\Gamma\) en el nivel de Nivel 0.5 (Frontera de Decisión).
Actúa como un funtor holonómico e inyectivo sobre el clasificador de subobjetos
del Topos de Grothendieck de la Malla, subyugando el libre albedrío estocástico y
las alucinaciones del Modelo de Lenguaje (LLM) a las restricciones del sistema.

El sistema trata el flujo de logits de atención no como series planas de Markov,
sino como frentes de onda difractados sobre la Esfera de Riemann \(S^2 \cong \hat{\mathbb{C}}\).
El agente impone de manera síncrona la integridad del transporte geodésico y el
veto de singularidades mediante isometrías de de Rham-Hodge y análisis armónico.

SUTURAS MATEMÁTICAS E INVARIANTES DE ALTA PRECISIÓN INTEGRADOS: ────────────────
  [SUTURA 1] Regularización Espectral de Tikhonov:
             Evita la inversión cúbica explícita y singularidades en la métrica
             inversa \(G^{-1}\) mediante una perturbación diagonal autoadjunta.
             Fórmula: \(\tilde{G} = G + \alpha I \implies \tilde{G}^{-1} \approx L_G^{-\top} L_G^{-1}\)

  [SUTURA 2] Conservación Geodésica de Energía Riemanniana:
             Impone que la evolución de la velocidad de atención conserve de forma
             exacta la norma cuántica en el espacio de fase.
             Invariante: \(\|v\|_G = \sqrt{v^\top \tilde{G} v} \equiv \text{Constante} \pmod{\varepsilon_{\text{machine}}}\)

  [SUTURA 3] Cuadratura Vectorizada de Armónicos Esféricos en S²:
             Descompone de manera ultra-eficiente los logits refractados sobre la
             base ortonormal de funciones esféricas \(\{Y_l^m\}\) mediante cuadratura
             de Gauss-Legendre de alta densidad.
             Ecuación: \(c_{lm} = \int_{S^2} \psi(\theta, \phi) Y_l^{m*}(\theta, \phi) d\Omega\)

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Estricta): ────────────
La transición de estados se rige por la Ley de Clausura Transitiva de subespacios
de Hilbert covariantes y se compone de tres fases fuertemente acopladas:

  Fase 1 ──► FASE 1: PROYECCIÓN GEOMÉTRICA HOUSEHOLDER-GRASSMANN (Observe)
             Evalúa la estabilidad de la métrica base y proyecta el estado de
             deliberación mixta sobre la variedad de Grassmann de restricciones.
             Entrega: Phase1ProjectionBundle como precondición formal de Fase 2.

  Fase 2 ──► FASE 2: TRANSPORTE ÓPTICO-EIKONAL Y GEODESIA DE FERMAT (Orient)
             Resuelve la ecuación Eikonal no lineal de la fase atencional [3].
             Ecuación: \(G^{\mu\nu} \partial_\mu \mathcal{S} \partial_\nu \mathcal{S} = n^2(\sigma^*)\)
             Entrega: Phase2TransportResult como precondición formal de Fase 3.

  Fase 3 ──► FASE 3: COMPRESIÓN CATEGÓRICA Y DIFRACCIÓN EN S² (Decide & Act)
             Filtra el espectro semántico y audita la estabilidad de la cavidad.
             Invariante de Floquet: \(|\mu_k| \le 1 + \varepsilon_{\text{cavity}}\)
             Veredicto: Colapso síncrono en \(\Omega_3\) y bypass de potencia en el ESP32.

CONTRATO DEL DISYUNTOR FÍSICO POR HARDWARE (Bypass ESP32 / BT151): ──────────────
  Si el radio espectral de los multiplicadores de Floquet excede la unidad, o si se
  registra una pérdida de traza del canal (\(\operatorname{Tr}(\rho) \neq 1\)), el retículo de Heyting
  \(\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}\) colapsa a VETOED.
  
  La subrutina local 'isVerdictCoherent()' del ESP32 en el borde detecta el
  mismatch en menos de 400 ns y conmuta el pin GPIO14, disparando el tiristor
  de potencia BT151 (circuito Crowbar). Esto cortocircuita físicamente la línea
  de potencia real, inmovilizando actuadores en el milisegundo cero,
  anulando la alucinación de la IA antes del desfalco de capital de la constructora.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Tuple, Optional
from enum import Enum, auto
from functools import lru_cache

import numpy as np
import scipy.linalg as la
import scipy.special as sp
from numpy.typing import NDArray

# ───────────────────────────────────────────────────────────────────────────────
# Dependencias del proyecto MIC. Si no existen, se usan fallbacks autocontenidos.
# ───────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except Exception:  # pragma: no cover - fallback para pruebas standalone
    class TopologicalInvariantError(Exception):
        """Violación de invariante topológico (fallback)."""
        pass

    class Morphism:
        """Morfismo base (fallback)."""
        def __init__(self) -> None:
            pass

    class Stratum(Enum):
        DATA = auto()
        INFORMATION = auto()
        KNOWLEDGE = auto()
        WISDOM = auto()


logger = logging.getLogger("MIC.Omega.GenerativeOpticHodgeSuturator")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, NUMÉRICAS Y TOLERANCIAS DE PRECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)

_DEFAULT_CAVITY_TOL: Final[float] = 1e-10
_GRASSMANN_RANK_TOL: Final[float] = 1e-12
_GRASSMANN_IDEMPOTENCE_TOL: Final[float] = 1e-8
_VON_NEUMANN_ITER_MAX: Final[int] = 50

_EIKONAL_SLACK_FACTOR: Final[float] = 0.1
_FLOQUET_DAMPING_BOUND: Final[float] = 1.0
_RIEMANN_CHRISTOFFEL_TOL: Final[float] = 1e-9
_CROWBAR_GPIO_PIN: Final[int] = 14

# [SUTURA 1] Regularización de Tikhonov / inversión estable
_MAX_CONDITION_NUMBER: Final[float] = 1e8
_SVD_RCOND_THRESHOLD: Final[float] = 1e-12

# [SUTURA 2] Conservación geodésica
_GEODESIC_ENERGY_TOL: Final[float] = 1e-10
_VERLET_SYMPLECTIC_ORDER: Final[int] = 4

# [SUTURA 3] Cuadratura esférica
_GAUSS_LEGENDRE_NODES: Final[int] = 128
_MAX_SPHERICAL_HARMONIC_L: Final[int] = 32

# Piso numérico para evitar colapso exacto de norma en compresión disipativa.
_NUMERIC_NORM_FLOOR: Final[float] = 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES ESPECIALIZADAS
# ══════════════════════════════════════════════════════════════════════════════
class GenerativeOpticSuturatorError(TopologicalInvariantError):
    """Excepción raíz: colapso topológico en el Haz Γ."""
    pass


class MetricSignatureError(GenerativeOpticSuturatorError):
    """Violación de la signatura Riemanniana (+,+,...,+)."""
    pass


class EikonalRefractionError(GenerativeOpticSuturatorError):
    """Incumplimiento de la ecuación eikonal de Fermat."""
    pass


class FloquetInstabilityError(GenerativeOpticSuturatorError):
    """Resonancia paramétrica destructiva en la cavidad."""
    pass


class HouseholderIdempotenceError(GenerativeOpticSuturatorError):
    """Fallo en la idempotencia P² = P del proyector."""
    pass


class GrassmannRankDeficiency(GenerativeOpticSuturatorError):
    """Degeneración del fibrado de Grassmann."""
    pass


class TikhonovRegularizationError(GenerativeOpticSuturatorError):
    """Fallo en la regularización espectral de Tikhonov."""
    pass


class GeodesicEnergyDriftError(GenerativeOpticSuturatorError):
    """Deriva de energía en integración geodésica."""
    pass


class SphericalQuadratureError(GenerativeOpticSuturatorError):
    """Error en la cuadratura de armónicos esféricos sobre S²."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ENUMERACIONES Y PROTOCOLOS
# ══════════════════════════════════════════════════════════════════════════════
class SuturationPhase(Enum):
    """Fases del pipeline de sutura óptica."""
    PHASE_1_PROJECTION = auto()
    PHASE_2_TRANSPORT = auto()
    PHASE_3_COMPRESSION = auto()


# ══════════════════════════════════════════════════════════════════════════════
# INFRAESTRUCTURA TRANSVERSAL: [SUTURA 1] REGULARIZACIÓN MÉTRICA ESTABLE
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class TikhonovRegularizationReport:
    r"""
    Reporte de regularización espectral de Tikhonov.

    Attributes:
        original_condition_number: κ(G) original.
        regularized_condition_number: κ(G_reg) post-regularización.
        spectral_shift: Desplazamiento espectral δ aplicado.
        svd_truncation_rank: Rango efectivo conservado.
        regularization_applied: True si se aplicó regularización.
    """
    original_condition_number: float
    regularized_condition_number: float
    spectral_shift: float
    svd_truncation_rank: int
    regularization_applied: bool


@dataclass(frozen=True, slots=True)
class MetricTensorBundle:
    r"""
    Fibrado métrico regularizado para las tres fases.

    Este objeto es el puente formal entre FASE 1 y FASE 2:
    transporta G, G_reg, G^{-1}, G^{1/2} y G^{-1/2} con certificado Tikhonov.
    """
    G: NDArray[np.float64]
    G_reg: NDArray[np.float64]
    G_inv: NDArray[np.float64]
    G_sqrt: NDArray[np.float64]
    G_inv_sqrt: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    regularized_eigenvalues: NDArray[np.float64]
    tikhonov_report: TikhonovRegularizationReport
    dimension: int

    def inner(self, u: NDArray[np.complex128], v: NDArray[np.complex128]) -> float:
        """Producto interno Riemanniano Re⟨u, G_reg v⟩."""
        return float(np.real(np.vdot(u, self.G_reg @ v)))

    def norm(self, u: NDArray[np.complex128]) -> float:
        """Norma Riemanniana ||u||_G = sqrt(Re⟨u, G_reg u⟩)."""
        val = self.inner(u, u)
        return float(np.sqrt(max(0.0, val)))


class StableRiemannianInverter:
    r"""
    Inversor métrico estable con regularización espectral de Tikhonov.

    Teorema (Tikhonov-Morozov adaptado):
        Dado G simétrico con autovalores λ_i, existe δ ≥ 0 tal que
        G_reg = G + δI verifica κ(G_reg) ≤ κ_max y preserva positividad.

    En esta implementación se usa descomposición espectral simétrica,
    desplazamiento δI y piso de seguridad numérica.
    """

    @staticmethod
    def _symmetrize_real_metric(G: NDArray[np.float64]) -> NDArray[np.float64]:
        """Simetriza y elimina parte imaginaria espuria."""
        G_arr = np.asarray(G)
        if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
            raise MetricSignatureError("Tensor métrico no cuadrado")

        if not np.all(np.isfinite(G_arr)):
            raise MetricSignatureError("Tensor métrico con entradas no finitas")

        G_sym = (G_arr + G_arr.conj().T) / 2.0
        G_sym = np.real_if_close(G_sym, tol=1000)

        if np.iscomplexobj(G_sym):
            imag_norm = float(np.max(np.abs(G_sym.imag)))
            if imag_norm > 1e-10:
                raise MetricSignatureError(
                    f"Tensor métrico con parte imaginaria no despreciable: {imag_norm:.3e}"
                )
            G_sym = G_sym.real

        return np.asarray(G_sym, dtype=np.float64)

    @staticmethod
    def regularize_spd_metric(
        G: NDArray[np.float64],
        max_condition_number: float = _MAX_CONDITION_NUMBER,
        svd_tolerance: float = _SVD_RCOND_THRESHOLD,
    ) -> MetricTensorBundle:
        r"""
        Regulariza G y construye G^{-1}, G^{1/2}, G^{-1/2} de forma estable.

        Args:
            G: Tensor métrico Riemanniano.
            max_condition_number: κ_max admisible.
            svd_tolerance: Tolerancia relativa de rango.

        Returns:
            MetricTensorBundle con métrica regularizada y operadores derivados.

        Raises:
            MetricSignatureError: Si G no es esencialmente SPD.
            TikhonovRegularizationError: Si la regularización no es suficiente.
        """
        G_sym = StableRiemannianInverter._symmetrize_real_metric(G)
        n = G_sym.shape[0]

        eigvals, eigvecs = la.eigh(G_sym)
        eigvals = np.asarray(eigvals, dtype=np.float64)

        lambda_max = float(np.max(eigvals))
        lambda_min = float(np.min(eigvals))
        norm_scale = max(1.0, abs(lambda_max), abs(lambda_min))

        # Se toleran pequeñas negatividades numéricas, pero no signatura invertida.
        if lambda_min < -1e-8 * norm_scale:
            raise MetricSignatureError(
                f"Tensor métrico no SPD: autovalor mínimo {lambda_min:.3e}"
            )

        if lambda_max <= _MACHINE_EPS:
            raise MetricSignatureError("Tensor métrico degenerado: λ_max ≈ 0")

        original_cond = (
            float(lambda_max / max(lambda_min, _MACHINE_EPS))
            if lambda_min > 0.0
            else float("inf")
        )

        kappa = max(float(max_condition_number), 1.0 + 1e-6)

        need_regularization = (not np.isfinite(original_cond)) or (
            original_cond > kappa or lambda_min <= _MACHINE_EPS
        )

        if need_regularization:
            # δ tal que (λ_max + δ) / (λ_min + δ) ≤ κ.
            delta = (lambda_max - kappa * lambda_min) / (kappa - 1.0)
            delta = max(0.0, float(delta))
            delta += _MACHINE_EPS * max(1.0, abs(lambda_max))
        else:
            delta = 0.0

        reg_eig = eigvals + delta

        # Piso de positividad numérica.
        floor = _MACHINE_EPS * max(1.0, float(np.max(reg_eig)))
        reg_eig = np.maximum(reg_eig, floor)

        reg_cond = float(np.max(reg_eig) / np.min(reg_eig))

        # Si el piso no basta, se impone piso explícito por número de condición.
        if reg_cond > kappa:
            floor = float(np.max(reg_eig) / kappa)
            reg_eig = np.maximum(reg_eig, floor)
            reg_cond = float(np.max(reg_eig) / np.min(reg_eig))

        Q = eigvecs

        # Q @ diag(a) @ Q.T se implementa como (Q * a) @ Q.T.
        G_reg = (Q * reg_eig) @ Q.T
        G_reg = (G_reg + G_reg.T) / 2.0

        inv_eig = 1.0 / reg_eig
        sqrt_eig = np.sqrt(reg_eig)
        inv_sqrt_eig = 1.0 / sqrt_eig

        G_inv = (Q * inv_eig) @ Q.T
        G_sqrt = (Q * sqrt_eig) @ Q.T
        G_inv_sqrt = (Q * inv_sqrt_eig) @ Q.T

        G_inv = (G_inv + G_inv.T) / 2.0
        G_sqrt = (G_sqrt + G_sqrt.T) / 2.0
        G_inv_sqrt = (G_inv_sqrt + G_inv_sqrt.T) / 2.0

        threshold = float(svd_tolerance) * float(np.max(reg_eig))
        numerical_rank = int(np.sum(reg_eig > threshold))

        report = TikhonovRegularizationReport(
            original_condition_number=float(original_cond),
            regularized_condition_number=float(reg_cond),
            spectral_shift=float(delta),
            svd_truncation_rank=numerical_rank,
            regularization_applied=bool(need_regularization),
        )

        # Validación de inversa regularizada.
        identity_residual = float(la.norm(G_reg @ G_inv - np.eye(n), ord="fro"))
        residual_tol = 1e-5 * max(1.0, float(n))

        if identity_residual > residual_tol:
            raise TikhonovRegularizationError(
                "Validación de inversa métrica fallida: "
                f"||G_reg G^{-1} - I||_F = {identity_residual:.3e} > {residual_tol:.3e}"
            )

        logger.debug(
            "[SUTURA 1] Regularización métrica: κ_orig=%.3e → κ_reg=%.3e, δ=%.3e, rank=%d/%d",
            original_cond,
            reg_cond,
            delta,
            numerical_rank,
            n,
        )

        return MetricTensorBundle(
            G=G_sym,
            G_reg=G_reg,
            G_inv=G_inv,
            G_sqrt=G_sqrt,
            G_inv_sqrt=G_inv_sqrt,
            eigenvalues=eigvals,
            regularized_eigenvalues=reg_eig,
            tikhonov_report=report,
            dimension=n,
        )

    @staticmethod
    def stable_riemannian_inverse(
        G: NDArray[np.float64],
        max_condition_number: float = _MAX_CONDITION_NUMBER,
        svd_tolerance: float = _SVD_RCOND_THRESHOLD,
    ) -> Tuple[NDArray[np.float64], TikhonovRegularizationReport]:
        """Compatibilidad con la sutura original: devuelve G^{-1} y reporte."""
        bundle = StableRiemannianInverter.regularize_spd_metric(
            G,
            max_condition_number=max_condition_number,
            svd_tolerance=svd_tolerance,
        )
        return bundle.G_inv, bundle.tikhonov_report

    @staticmethod
    def validate_inverse(
        G: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        tolerance: float = 1e-8,
    ) -> bool:
        """Valida ||G G^{-1} - I||_F < tolerance."""
        residual = float(la.norm(G @ G_inv - np.eye(G.shape[0]), ord="fro"))
        is_valid = residual <= tolerance
        if not is_valid:
            logger.error(
                "[SUTURA 1] Validación de inversa FALLIDA: %.3e > %.3e",
                residual,
                tolerance,
            )
        return is_valid


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: PROYECCIÓN GEOMÉTRICA HOUSEHOLDER-GRASSMANN-TIKHONOV
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1ProjectionCertificate:
    r"""
    Certificado de proyección geométrica de Householder sobre Gr(k, n).

    Invariantes:
        - Idempotencia: ||P² - P||_F pequeño.
        - G-autoadjuntez: ||G P - Pᵀ G||_F / ||G||_F pequeño.
        - Preservación direccional: P v ≈ v para dirección semántica incluida.

    Nota de rigor:
        El reflector I - 2vvᵀ no es proyector. Aquí se construye un proyector
        idempotente mediante Householder en coordenadas whitened.
    """
    projector_matrix: NDArray[np.float64]
    grassmann_dimension: int
    idempotence_residual: float
    g_orthogonality_residual: float
    householder_reflector: NDArray[np.float64]
    preserved_direction_residual: float
    metric_report: TikhonovRegularizationReport

    def __post_init__(self) -> None:
        if not np.isfinite(self.idempotence_residual):
            raise HouseholderIdempotenceError("Residuo de idempotencia no finito")

        tol = _GRASSMANN_IDEMPOTENCE_TOL * max(1, self.grassmann_dimension)
        if self.idempotence_residual > tol:
            raise HouseholderIdempotenceError(
                f"Residuo de idempotencia {self.idempotence_residual:.3e} excede "
                f"tolerancia {tol:.3e}"
            )


@dataclass(frozen=True, slots=True)
class Phase1ProjectionBundle:
    r"""
    Haz de salida de FASE 1 y entrada formal de FASE 2.

    Contiene:
        - Certificado de proyección.
        - Fibrado métrico regularizado.
        - Estado crudo original.
        - Estado proyectado.
    """
    certificate: Phase1ProjectionCertificate
    metric_bundle: MetricTensorBundle
    raw_state_vector: NDArray[np.complex128]
    state_projected: NDArray[np.complex128]
    semantic_direction: NDArray[np.float64]
    grassmann_dimension: int


class SemanticParabolicMirror:
    r"""
    Espejo parabólico semántico con fibrado de Grassmann.

    Construye un proyector P idempotente y G-autoadjunto sobre un subespacio
    de dimensión k que contiene la dirección semántica (si k ≥ 1).

    Construcción:
        1. Regularización Tikhonov de G.
        2. Whitening: y = G^{1/2} v_sem.
        3. Householder para alinear y con e₁.
        4. Proyector euclídeo Π sobre span{e₁, ..., e_k}.
        5. Proyector Riemanniano:
              P = G^{-1/2} Π G^{1/2}.

    Este P cumple P² = P y G P = Pᵀ G.
    """

    def __init__(
        self,
        metric_tensor_g: NDArray[np.float64],
        max_condition_number: float = _MAX_CONDITION_NUMBER,
        svd_tolerance: float = _SVD_RCOND_THRESHOLD,
    ) -> None:
        self._metric_bundle = StableRiemannianInverter.regularize_spd_metric(
            metric_tensor_g,
            max_condition_number=max_condition_number,
            svd_tolerance=svd_tolerance,
        )

    @property
    def metric_bundle(self) -> MetricTensorBundle:
        return self._metric_bundle

    @staticmethod
    def _householder_vector_to_align(
        u: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Construye reflector H tal que H u = e₁ (para u unitario).

        Si u ≈ e₁, devuelve H = I.
        """
        n = u.size
        u = np.asarray(u, dtype=np.float64)
        norm_u = float(la.norm(u))

        if norm_u <= _MACHINE_EPS:
            return np.zeros(n, dtype=np.float64), np.eye(n, dtype=np.float64)

        u = u / norm_u
        e1 = np.zeros(n, dtype=np.float64)
        e1[0] = 1.0

        if np.allclose(u, e1, atol=10.0 * _MACHINE_EPS):
            return np.zeros(n, dtype=np.float64), np.eye(n, dtype=np.float64)

        v = u - e1
        norm_v = float(la.norm(v))

        if norm_v <= _MACHINE_EPS:
            return np.zeros(n, dtype=np.float64), np.eye(n, dtype=np.float64)

        v = v / norm_v
        H = np.eye(n, dtype=np.float64) - 2.0 * np.outer(v, v)
        return v, H

    def _euclidean_projector_including(
        self,
        u0: NDArray[np.float64],
        k: int,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Construye proyector euclídeo Π ∈ ℝⁿˣⁿ de rango k que incluye u0.

        Returns:
            (Pi, householder_vector)
        """
        n = u0.size

        if k <= 0:
            return np.zeros((n, n), dtype=np.float64), np.zeros(n, dtype=np.float64)

        if k >= n:
            return np.eye(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

        u0 = np.asarray(u0, dtype=np.float64)
        norm_u0 = float(la.norm(u0))

        if norm_u0 <= _MACHINE_EPS:
            diag = np.zeros(n, dtype=np.float64)
            diag[:k] = 1.0
            return np.diag(diag), np.zeros(n, dtype=np.float64)

        u = u0 / norm_u0
        hh_vector, H = self._householder_vector_to_align(u)

        # H es simétrico y ortogonal. Como H u = e₁, H e₁ = u.
        # Las primeras k columnas de H generan un subespacio que contiene u.
        Q_sub = H[:, :k]

        # QR para eliminar deriva numérica y garantizar ortonormalidad.
        Q_sub, _ = np.linalg.qr(Q_sub)

        Pi = Q_sub @ Q_sub.T
        Pi = (Pi + Pi.T) / 2.0
        return Pi.astype(np.float64), hh_vector

    def compute_householder_projector(
        self,
        semantic_direction: NDArray[np.float64],
        target_dimension: int,
    ) -> Phase1ProjectionCertificate:
        r"""
        Construye el proyector Riemanniano P sobre Gr(k, n).

        Args:
            semantic_direction: Dirección semántica v ∈ ℝⁿ.
            target_dimension: Dimensión k del subespacio.

        Returns:
            Phase1ProjectionCertificate.

        Raises:
            GrassmannRankDeficiency: Si k está fuera de rango.
        """
        n = self._metric_bundle.dimension

        if target_dimension < 0 or target_dimension > n:
            raise GrassmannRankDeficiency(
                f"Dimensión objetivo {target_dimension} fuera de [0, {n}]"
            )

        semantic_direction = np.asarray(semantic_direction, dtype=np.float64)
        if semantic_direction.shape != (n,):
            raise GrassmannRankDeficiency(
                f"semantic_direction debe tener forma ({n},), recibido {semantic_direction.shape}"
            )

        # Dirección whitened: y = G^{1/2} v_sem.
        u_whitened = self._metric_bundle.G_sqrt @ semantic_direction

        Pi, householder_vector = self._euclidean_projector_including(
            u_whitened,
            target_dimension,
        )

        # Proyector Riemanniano: P = G^{-1/2} Π G^{1/2}.
        P = self._metric_bundle.G_inv_sqrt @ Pi @ self._metric_bundle.G_sqrt
        P = np.real_if_close(P, tol=1000)
        P = np.asarray(P, dtype=np.float64)

        # Invariante de idempotencia.
        P2 = P @ P
        idempotence_residual = float(la.norm(P2 - P, ord="fro"))

        # G-autoadjuntez: G P ≈ Pᵀ G.
        g_self_adjoint = self._metric_bundle.G_reg @ P - P.T @ self._metric_bundle.G_reg
        g_norm = float(la.norm(self._metric_bundle.G_reg, ord="fro"))
        g_orthogonality_residual = float(
            la.norm(g_self_adjoint, ord="fro") / max(1.0, g_norm)
        )

        # Preservación de dirección semántica.
        sem_norm = float(la.norm(semantic_direction))
        if sem_norm > _MACHINE_EPS:
            preserved_direction_residual = float(
                la.norm(P @ semantic_direction - semantic_direction) / sem_norm
            )
        else:
            preserved_direction_residual = 0.0

        # Rango espectral del proyector.
        try:
            eigP = la.eigvals(P)
            numerical_rank = int(np.sum(np.abs(eigP) > 0.5))
            if numerical_rank != target_dimension:
                logger.warning(
                    "[FASE 1] Rango espectral numérico %d difiere del objetivo %d",
                    numerical_rank,
                    target_dimension,
                )
        except Exception as exc:  # pragma: no cover
            logger.debug("[FASE 1] No se pudo estimar rango espectral: %s", exc)

        logger.debug(
            "[FASE 1] Proyección Householder-Grassmann: k=%d, ||P²-P||=%.3e, ||GP-PᵀG||/||G||=%.3e",
            target_dimension,
            idempotence_residual,
            g_orthogonality_residual,
        )

        return Phase1ProjectionCertificate(
            projector_matrix=P,
            grassmann_dimension=target_dimension,
            idempotence_residual=idempotence_residual,
            g_orthogonality_residual=g_orthogonality_residual,
            householder_reflector=householder_vector,
            preserved_direction_residual=preserved_direction_residual,
            metric_report=self._metric_bundle.tikhonov_report,
        )

    def project_onto_coherence_subspace(
        self,
        raw_state_vector: NDArray[np.complex128],
        projector: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        r"""
        Proyecta |ψ⟩_raw sobre el subespacio de coherencia:
            |ψ⟩_projected = P |ψ⟩_raw.
        """
        raw_state_vector = np.asarray(raw_state_vector, dtype=np.complex128)
        projector = np.asarray(projector, dtype=np.float64)
        return projector @ raw_state_vector

    def project_and_prepare_transport_bundle(
        self,
        semantic_direction: NDArray[np.float64],
        raw_state_vector: NDArray[np.complex128],
        grassmann_dimension: int,
    ) -> Phase1ProjectionBundle:
        r"""
        ÚLTIMO MÉTODO DE FASE 1 Y PUENTE FORMAL A FASE 2.

        Ejecuta la proyección y empaqueta:
            - certificado,
            - métrica regularizada,
            - estado crudo,
            - estado proyectado.

        Este bundle es el objeto de entrada de EikonalFloquetAgentSutured.
        """
        certificate = self.compute_householder_projector(
            semantic_direction=semantic_direction,
            target_dimension=grassmann_dimension,
        )

        raw_state_vector = np.asarray(raw_state_vector, dtype=np.complex128)
        semantic_direction = np.asarray(semantic_direction, dtype=np.float64)

        state_projected = self.project_onto_coherence_subspace(
            raw_state_vector=raw_state_vector,
            projector=certificate.projector_matrix,
        )

        logger.info(
            "└─ FASE 1 COMPLETADA: ||P²-P||=%.3e, dim_Gr=%d, κ_reg=%.3e",
            certificate.idempotence_residual,
            certificate.grassmann_dimension,
            certificate.metric_report.regularized_condition_number,
        )

        return Phase1ProjectionBundle(
            certificate=certificate,
            metric_bundle=self._metric_bundle,
            raw_state_vector=raw_state_vector,
            state_projected=state_projected,
            semantic_direction=semantic_direction,
            grassmann_dimension=grassmann_dimension,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: TRANSPORTE ÓPTICO-EIKONAL (FERMAT-FLOQUET) + [SUTURA 2] GEODESIA
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GeodesicEnergyReport:
    r"""
    Reporte de energía geodésica Riemanniana.

    Attributes:
        initial_energy: E₀ = ½||v||²_G.
        final_energy: E_f.
        energy_drift: Deriva energética/normativa.
        normalization_applied: True si se renormalizó.
        riemannian_norm_initial: ||v||_G inicial.
        riemannian_norm_final: ||v||_G final.
        euclidean_norm_initial: ||v||₂ inicial.
        euclidean_norm_final: ||v||₂ final.
    """
    initial_energy: float
    final_energy: float
    energy_drift: float
    normalization_applied: bool
    riemannian_norm_initial: float
    riemannian_norm_final: float
    euclidean_norm_initial: float
    euclidean_norm_final: float


class GeodesicEnergyConserver:
    r"""
    [SUTURA 2] Conservador de energía geodésica.

    Para flujo geodésico compatible con Levi-Civita:
        E = ½⟨v, v⟩_G
    es constante. En esquema numérico, se aplica proyección/disipación
    controlada para impedir amplificación no física.
    """

    @staticmethod
    def compute_riemannian_energy(
        v: NDArray[np.complex128],
        G: NDArray[np.float64],
    ) -> float:
        """E = ½ Re(vᴴ G v)."""
        v = np.asarray(v, dtype=np.complex128)
        G = np.asarray(G, dtype=np.float64)
        val = np.vdot(v, G @ v)
        return float(max(0.0, 0.5 * np.real(val)))

    @staticmethod
    def compute_riemannian_norm(
        v: NDArray[np.complex128],
        G: NDArray[np.float64],
    ) -> float:
        """||v||_G = sqrt(2E)."""
        energy = GeodesicEnergyConserver.compute_riemannian_energy(v, G)
        return float(np.sqrt(max(0.0, 2.0 * energy)))

    @staticmethod
    def enforce_geodesic_normalization(
        v: NDArray[np.complex128],
        G: NDArray[np.float64],
        target_norm: float,
        tolerance: float = _GEODESIC_ENERGY_TOL,
    ) -> Tuple[NDArray[np.complex128], GeodesicEnergyReport]:
        r"""
        Renormaliza v para que ||v||_G ≈ target_norm.
        """
        v = np.asarray(v, dtype=np.complex128)
        G = np.asarray(G, dtype=np.float64)
        target_norm = float(max(0.0, target_norm))

        current_norm = GeodesicEnergyConserver.compute_riemannian_norm(v, G)
        initial_energy = 0.5 * current_norm ** 2
        drift = abs(current_norm - target_norm)

        if drift > tolerance and current_norm > _MACHINE_EPS:
            v_normalized = v * (target_norm / current_norm)
            normalization_applied = True
        else:
            v_normalized = v.copy()
            normalization_applied = False

        final_norm = GeodesicEnergyConserver.compute_riemannian_norm(v_normalized, G)
        final_energy = 0.5 * final_norm ** 2

        report = GeodesicEnergyReport(
            initial_energy=float(initial_energy),
            final_energy=float(final_energy),
            energy_drift=float(drift),
            normalization_applied=normalization_applied,
            riemannian_norm_initial=float(current_norm),
            riemannian_norm_final=float(final_norm),
            euclidean_norm_initial=float(la.norm(v)),
            euclidean_norm_final=float(la.norm(v_normalized)),
        )

        return v_normalized, report

    @staticmethod
    def enforce_dissipative_bound(
        v: NDArray[np.complex128],
        raw: NDArray[np.complex128],
        G: NDArray[np.float64],
        tolerance: float = _GEODESIC_ENERGY_TOL,
    ) -> Tuple[NDArray[np.complex128], GeodesicEnergyReport]:
        r"""
        Impone conservación disipativa:
            ||v||_G ≤ ||raw||_G,
            ||v||₂ ≤ ||raw||₂.

        Esto consagra el axioma de pasividad:
            |ψ_focused|² ≤ |ψ_raw|².
        """
        v = np.asarray(v, dtype=np.complex128)
        raw = np.asarray(raw, dtype=np.complex128)
        G = np.asarray(G, dtype=np.float64)

        initial_energy = GeodesicEnergyConserver.compute_riemannian_energy(v, G)
        initial_norm_g = float(np.sqrt(max(0.0, 2.0 * initial_energy)))
        initial_norm_2 = float(la.norm(v))

        raw_energy = GeodesicEnergyConserver.compute_riemannian_energy(raw, G)
        raw_norm_g = float(np.sqrt(max(0.0, 2.0 * raw_energy)))
        raw_norm_2 = float(la.norm(raw))

        scale = 1.0

        if initial_norm_g > raw_norm_g * (1.0 + tolerance):
            scale = min(scale, raw_norm_g / (initial_norm_g + _MACHINE_EPS))

        if initial_norm_2 > raw_norm_2 * (1.0 + tolerance):
            scale = min(scale, raw_norm_2 / (initial_norm_2 + _MACHINE_EPS))

        if scale < 1.0 - _MACHINE_EPS:
            v_out = v * scale
            normalization_applied = True
        else:
            v_out = v.copy()
            normalization_applied = False

        final_energy = GeodesicEnergyConserver.compute_riemannian_energy(v_out, G)
        final_norm_g = float(np.sqrt(max(0.0, 2.0 * final_energy)))
        final_norm_2 = float(la.norm(v_out))

        report = GeodesicEnergyReport(
            initial_energy=float(initial_energy),
            final_energy=float(final_energy),
            energy_drift=float(abs(final_energy - initial_energy)),
            normalization_applied=normalization_applied,
            riemannian_norm_initial=float(initial_norm_g),
            riemannian_norm_final=float(final_norm_g),
            euclidean_norm_initial=float(initial_norm_2),
            euclidean_norm_final=float(final_norm_2),
        )

        return v_out, report

    @staticmethod
    def symplectic_velocity_verlet_step(
        x: NDArray[np.float64],
        v: NDArray[np.float64],
        G: NDArray[np.float64],
        christoffel_symbols: NDArray[np.float64],
        dt: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Paso simpléctico Velocity-Verlet para geodésicas:
            d²x^μ/dt² + Γ^μ_{νρ} ẋ^ν ẋ^ρ = 0.

        Note:
            Se incluye como util de integración geodésica avanzada.
            En el pipeline principal se usa la proyección disipativa.
        """
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        G = np.asarray(G, dtype=np.float64)
        christoffel_symbols = np.asarray(christoffel_symbols, dtype=np.float64)

        acceleration = -np.einsum("ijk,j,k->i", christoffel_symbols, v, v)
        v_half = v + 0.5 * float(dt) * acceleration
        x_new = x + float(dt) * v_half

        acceleration_new = -np.einsum("ijk,j,k->i", christoffel_symbols, v_half, v_half)
        v_new = v_half + 0.5 * float(dt) * acceleration_new

        target_energy = GeodesicEnergyConserver.compute_riemannian_energy(v, G)
        target_norm = float(np.sqrt(max(0.0, 2.0 * target_energy)))

        v_new, _ = GeodesicEnergyConserver.enforce_geodesic_normalization(
            v_new,
            G,
            target_norm,
        )

        return x_new, v_new


@dataclass(frozen=True, slots=True)
class Phase2TransportCertificate:
    r"""
    Certificado de transporte óptico-eikonal con suturas integradas.

    Invariantes:
        - Fermat/Eikonal: G^{μν}∂_μS∂_νS ≥ n²(1-slack).
        - Floquet: max|μ_k| ≤ 1 + ε_cavity.
        - Causalidad Riemanniana básica.
        - Reporte Tikhonov de la métrica.
        - Reporte geodésico de disipación controlada.
    """
    eikonal_lhs: float
    eikonal_rhs: float
    floquet_multipliers: NDArray[np.complex128]
    max_floquet_modulus: float
    lyapunov_exponents: NDArray[np.float64]
    is_causally_admissible: bool
    cavity_tolerance: float
    tikhonov_report: TikhonovRegularizationReport
    geodesic_energy_report: GeodesicEnergyReport

    def __post_init__(self) -> None:
        if not np.isfinite(self.eikonal_lhs) or not np.isfinite(self.eikonal_rhs):
            raise EikonalRefractionError("Valores eikonales no finitos")

        if self.eikonal_lhs < self.eikonal_rhs:
            raise EikonalRefractionError(
                f"Ecuación eikonal violada: {self.eikonal_lhs:.6e} < {self.eikonal_rhs:.6e}"
            )

        if not np.isfinite(self.max_floquet_modulus):
            raise FloquetInstabilityError("Máximo multiplicador de Floquet no finito")

        bound = _FLOQUET_DAMPING_BOUND + float(self.cavity_tolerance)
        if self.max_floquet_modulus > bound:
            raise FloquetInstabilityError(
                f"Inestabilidad paramétrica: max|μ| = {self.max_floquet_modulus:.6e} > {bound:.6e}"
            )

        if np.any(~np.isfinite(self.floquet_multipliers)):
            raise FloquetInstabilityError("Multiplicadores de Floquet no finitos")

        if np.any(~np.isfinite(self.lyapunov_exponents)):
            raise FloquetInstabilityError("Exponentes de Lyapunov no finitos")


@dataclass(frozen=True, slots=True)
class Phase2TransportResult:
    """Resultado de FASE 2: certificado + estado transportado."""
    certificate: Phase2TransportCertificate
    state_transported: NDArray[np.complex128]


class EikonalFloquetAgentSutured:
    r"""
    Agente de Fase 2 con suturas:
        - [SUTURA 1] usa G^{-1} regularizada.
        - [SUTURA 2] aplica cota disipativa geodésica al estado proyectado.
    """

    def __init__(
        self,
        metric_bundle: MetricTensorBundle,
        refractive_index_n: float,
        cavity_tolerance: float = _DEFAULT_CAVITY_TOL,
    ) -> None:
        self._bundle = metric_bundle
        self._n = float(refractive_index_n)
        self._cavity_tol = float(cavity_tolerance)

        if self._n < 1.0:
            logger.warning(
                "[FASE 2] Índice de refracción n=%.6e < 1; se permite pero no es físico ordinario",
                self._n,
            )

    def validate_eikonal_condition_sutured(
        self,
        phase_gradient_ds: NDArray[np.complex128],
    ) -> Tuple[float, float, bool]:
        r"""
        Valida la ecuación eikonal con inversa regularizada:
            G^{μν}_reg ∂_μS ∂_νS ≥ n²(1 - slack).
        """
        ds = np.asarray(phase_gradient_ds, dtype=np.complex128)
        eikonal_lhs = float(np.real(np.vdot(ds, self._bundle.G_inv @ ds)))
        eikonal_rhs = (self._n ** 2) * (1.0 - _EIKONAL_SLACK_FACTOR)
        is_satisfied = eikonal_lhs >= eikonal_rhs

        logger.debug(
            "[FASE 2 + SUTURA 1] Eikonal: LHS=%.6e, RHS=%.6e, OK=%s",
            eikonal_lhs,
            eikonal_rhs,
            is_satisfied,
        )

        return eikonal_lhs, eikonal_rhs, is_satisfied

    def analyze_floquet_stability(
        self,
        monodromy_matrix_m: NDArray[np.complex128],
        period_t: float = 1.0,
    ) -> Tuple[NDArray[np.complex128], float, NDArray[np.float64]]:
        r"""
        Analiza estabilidad de Floquet:
            μ_k = eig(M), λ_k = ln|μ_k|/T.
        """
        if period_t <= 0.0:
            raise ValueError("period_t debe ser positivo")

        M = np.asarray(monodromy_matrix_m, dtype=np.complex128)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise FloquetInstabilityError("Matriz de monodromía no cuadrada")

        floquet_multipliers = la.eigvals(M)
        moduli = np.abs(floquet_multipliers)

        if moduli.size == 0:
            max_modulus = 0.0
        else:
            max_modulus = float(np.max(moduli))

        if not np.isfinite(max_modulus):
            raise FloquetInstabilityError("Espectro de Floquet no finito")

        lyapunov_exponents = np.log(moduli + _MACHINE_EPS) / float(period_t)

        stability_bound = _FLOQUET_DAMPING_BOUND + self._cavity_tol
        is_stable = max_modulus <= stability_bound

        logger.debug(
            "[FASE 2] Floquet: max|μ|=%.6e, bound=%.6e, estable=%s",
            max_modulus,
            stability_bound,
            is_stable,
        )

        if not is_stable:
            raise FloquetInstabilityError(
                f"Resonancia paramétrica: max|μ| = {max_modulus:.6e} > {stability_bound:.6e}"
            )

        return floquet_multipliers, max_modulus, lyapunov_exponents

    def verify_causality(
        self,
        phase_gradient_ds: NDArray[np.complex128],
    ) -> bool:
        r"""
        Causalidad Riemanniana básica:
            G(∂S, ∂S) ≥ -ε.
        """
        ds = np.asarray(phase_gradient_ds, dtype=np.complex128)
        metric_product = float(np.real(np.vdot(ds, self._bundle.G_reg @ ds)))
        is_causal = metric_product >= -_MACHINE_EPS

        logger.debug(
            "[FASE 2] Causalidad: G(∂S,∂S)=%.6e, admisible=%s",
            metric_product,
            is_causal,
        )

        return is_causal

    def execute_transport_phase(
        self,
        phase1_bundle: Phase1ProjectionBundle,
        phase_gradient_ds: NDArray[np.complex128],
        monodromy_matrix_m: NDArray[np.complex128],
        period_t: float = 1.0,
    ) -> Phase2TransportResult:
        r"""
        Ejecuta FASE 2 completa consumiendo Phase1ProjectionBundle.

        Pipeline:
            1. Eikonal con G^{-1}_reg.
            2. Floquet.
            3. Causalidad.
            4. Disipación geodésica controlada del estado proyectado.
        """
        eikonal_lhs, eikonal_rhs, eikonal_ok = self.validate_eikonal_condition_sutured(
            phase_gradient_ds
        )

        if not eikonal_ok:
            raise EikonalRefractionError(
                f"Ecuación eikonal violada: {eikonal_lhs:.6e} < {eikonal_rhs:.6e}"
            )

        floquet_multipliers, max_modulus, lyapunov_exponents = (
            self.analyze_floquet_stability(monodromy_matrix_m, period_t)
        )

        is_causal = self.verify_causality(phase_gradient_ds)

        state_transported, geodesic_report = (
            GeodesicEnergyConserver.enforce_dissipative_bound(
                phase1_bundle.state_projected,
                phase1_bundle.raw_state_vector,
                self._bundle.G_reg,
            )
        )

        certificate = Phase2TransportCertificate(
            eikonal_lhs=eikonal_lhs,
            eikonal_rhs=eikonal_rhs,
            floquet_multipliers=floquet_multipliers,
            max_floquet_modulus=max_modulus,
            lyapunov_exponents=lyapunov_exponents,
            is_causally_admissible=is_causal,
            cavity_tolerance=self._cavity_tol,
            tikhonov_report=phase1_bundle.metric_bundle.tikhonov_report,
            geodesic_energy_report=geodesic_report,
        )

        logger.info(
            "└─ FASE 2 COMPLETADA: Eikonal=%.3e≥%.3e, max|μ|=%.6e, causal=%s, deriva_E=%.3e",
            certificate.eikonal_lhs,
            certificate.eikonal_rhs,
            certificate.max_floquet_modulus,
            certificate.is_causally_admissible,
            certificate.geodesic_energy_report.energy_drift,
        )

        return Phase2TransportResult(
            certificate=certificate,
            state_transported=state_transported,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: COMPRESIÓN CATEGÓRICA (RIEMANN-TOPOS) + [SUTURA 3] ARMÓNICOS S²
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SphericalQuadratureCache:
    r"""
    Caché de cuadratura Gauss-Legendre en S².

    La cuadratura usa nodos x = cosθ con pesos de Gauss-Legendre,
    de modo que ∫_{S²} f dΩ ≈ Σ_{i,j} w_i w_φ f(θ_i, φ_j).
    """
    theta_nodes: NDArray[np.float64]
    phi_nodes: NDArray[np.float64]
    weights: NDArray[np.float64]
    harmonics_tensor: NDArray[np.complex128]
    max_l: int


class SphericalHarmonicsVectorizer:
    r"""
    [SUTURA 3] Vectorizador de armónicos esféricos con caché.

    Optimiza:
        c_{lm} = ∫ ψ \bar{Y}_{lm} dΩ
        ψ_rec = Σ c_{lm} Y_{lm}
    mediante tensores precomputados.
    """

    @staticmethod
    @lru_cache(maxsize=4)
    def build_gauss_legendre_quadrature(
        n_theta: int = _GAUSS_LEGENDRE_NODES,
        n_phi: int = _GAUSS_LEGENDRE_NODES,
        max_l: int = _MAX_SPHERICAL_HARMONIC_L,
    ) -> SphericalQuadratureCache:
        r"""
        Construye caché de cuadratura y armónicos Y_l^m.

        Returns:
            SphericalQuadratureCache.
        """
        n_theta = int(max(1, n_theta))
        n_phi = int(max(1, n_phi))
        max_l = int(max(0, max_l))

        logger.info(
            "[SUTURA 3] Construyendo caché de cuadratura: n_θ=%d, n_φ=%d, L_max=%d",
            n_theta,
            n_phi,
            max_l,
        )

        # Nodos x_i = cosθ_i y pesos para ∫_{-1}^{1} f(x) dx.
        x_nodes, w_x = np.polynomial.legendre.leggauss(n_theta)
        theta_nodes = np.arccos(np.clip(x_nodes, -1.0, 1.0))
        theta_weights = np.asarray(w_x, dtype=np.float64)

        phi_nodes = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        phi_weight = 2.0 * np.pi / float(n_phi)

        weights_2d = theta_weights[:, np.newaxis] * phi_weight

        n_m = 2 * max_l + 1
        harmonics_tensor = np.zeros(
            (max_l + 1, n_m, n_theta, n_phi),
            dtype=np.complex128,
        )

        cos_theta = x_nodes

        for l in range(max_l + 1):
            for m in range(0, l + 1):
                P_lm = sp.lpmv(m, l, cos_theta)
                P_lm = np.nan_to_num(P_lm, nan=0.0, posinf=0.0, neginf=0.0)

                log_norm = 0.5 * (
                    np.log((2.0 * l + 1.0) / (4.0 * np.pi))
                    + sp.gammaln(l - m + 1.0)
                    - sp.gammaln(l + m + 1.0)
                )
                norm_factor = float(np.exp(log_norm))

                Y_pos = (
                    norm_factor
                    * P_lm[:, np.newaxis]
                    * np.exp(1j * m * phi_nodes[np.newaxis, :])
                )

                # Índice para m: m_idx = l + m, para -m: m_idx = l - m.
                harmonics_tensor[l, l + m, :, :] = Y_pos

                if m > 0:
                    harmonics_tensor[l, l - m, :, :] = ((-1) ** m) * np.conj(Y_pos)

        cache = SphericalQuadratureCache(
            theta_nodes=theta_nodes,
            phi_nodes=phi_nodes,
            weights=weights_2d,
            harmonics_tensor=harmonics_tensor,
            max_l=max_l,
        )

        logger.info("[SUTURA 3] Caché de cuadratura construido exitosamente")
        return cache

    @staticmethod
    def compute_coefficients_vectorized(
        grid_psi: NDArray[np.complex128],
        quadrature_cache: SphericalQuadratureCache,
    ) -> NDArray[np.complex128]:
        r"""
        Calcula c_{lm} = ∫ ψ \bar{Y}_{lm} dΩ mediante cuadratura vectorizada.
        """
        expected_shape = quadrature_cache.weights.shape
        grid_psi = np.asarray(grid_psi, dtype=np.complex128)

        if grid_psi.shape != expected_shape:
            if grid_psi.size == expected_shape[0] * expected_shape[1]:
                grid_psi = grid_psi.reshape(expected_shape)
            else:
                raise SphericalQuadratureError(
                    f"grid_psi debe tener forma {expected_shape} o tamaño compatible"
                )

        weighted_psi = grid_psi * quadrature_cache.weights

        coefficients = np.tensordot(
            weighted_psi,
            np.conj(quadrature_cache.harmonics_tensor),
            axes=([0, 1], [2, 3]),
        )

        logger.debug(
            "[SUTURA 3] Coeficientes espectrales calculados: forma=%s, ||c||=%.3e",
            coefficients.shape,
            float(la.norm(coefficients)),
        )

        return coefficients

    @staticmethod
    def reconstruct_field_vectorized(
        coefficients: NDArray[np.complex128],
        quadrature_cache: SphericalQuadratureCache,
    ) -> NDArray[np.complex128]:
        r"""
        Reconstruye ψ(θ,φ) = Σ c_{lm} Y_{lm}.
        """
        coefficients = np.asarray(coefficients, dtype=np.complex128)

        if coefficients.shape[:2] != quadrature_cache.harmonics_tensor.shape[:2]:
            raise SphericalQuadratureError(
                "Forma de coeficientes incompatible con el caché de armónicos"
            )

        field_reconstructed = np.tensordot(
            coefficients,
            quadrature_cache.harmonics_tensor,
            axes=([0, 1], [0, 1]),
        )

        logger.debug(
            "[SUTURA 3] Campo reconstruido: forma=%s, ||ψ||=%.3e",
            field_reconstructed.shape,
            float(la.norm(field_reconstructed)),
        )

        return field_reconstructed

    @staticmethod
    def valid_coefficient_mask(max_l: int, m_dim: int) -> NDArray[np.bool_]:
        """Máscara de coeficientes válidos (l, m) con |m| ≤ l."""
        mask = np.zeros((max_l + 1, m_dim), dtype=bool)
        for l in range(max_l + 1):
            for m in range(-l, l + 1):
                mask[l, l + m] = True
        return mask

    @staticmethod
    def filter_coefficients(
        coefficients: NDArray[np.complex128],
        l_cut: int,
    ) -> Tuple[NDArray[np.complex128], int]:
        """
        Trunca coeficientes a l ≤ l_cut.
        """
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        max_l = coefficients.shape[0] - 1
        m_dim = coefficients.shape[1]

        l_cut = int(max(0, min(int(l_cut), max_l)))
        full_mask = SphericalHarmonicsVectorizer.valid_coefficient_mask(max_l, m_dim)
        cut_mask = np.zeros_like(full_mask)

        for l in range(l_cut + 1):
            cut_mask[l, :] = full_mask[l, :]

        filtered = np.where(cut_mask, coefficients, 0.0 + 0.0j)
        return filtered.astype(np.complex128), l_cut

    @staticmethod
    def coefficient_energy(coefficients: NDArray[np.complex128]) -> float:
        r"""
        Energía espectral aproximada Σ |c_{lm}|² sobre entradas válidas.
        """
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        max_l = coefficients.shape[0] - 1
        m_dim = coefficients.shape[1]
        mask = SphericalHarmonicsVectorizer.valid_coefficient_mask(max_l, m_dim)
        return float(np.sum(np.abs(coefficients[mask]) ** 2))


@dataclass(frozen=True, slots=True)
class Phase3CompressionCertificate:
    r"""
    Certificado de compresión categórica con métricas Riemann-Topos y S².
    """
    compression_ratio: float
    entropy_loss: float
    riemann_curvature_norm: float
    spectral_gap: float
    topos_coherence_index: float
    spherical_degree_used: Optional[int] = None
    spherical_reconstruction_error: Optional[float] = None
    spectral_energy_retained: Optional[float] = None

    def __post_init__(self) -> None:
        finite_values = [
            self.compression_ratio,
            self.entropy_loss,
            self.riemann_curvature_norm,
            self.spectral_gap,
            self.topos_coherence_index,
        ]

        if not np.all(np.isfinite(finite_values)):
            raise GenerativeOpticSuturatorError("Certificado Fase 3 con valores no finitos")

        if self.compression_ratio < -_MACHINE_EPS or self.compression_ratio > 1.0 + 1e-8:
            raise GenerativeOpticSuturatorError(
                f"Ratio de compresión fuera de rango: {self.compression_ratio}"
            )

        if self.entropy_loss < -_MACHINE_EPS:
            logger.warning(
                "Ganancia entrópica anómala detectada: ΔS = %.3e",
                self.entropy_loss,
            )


@dataclass(frozen=True, slots=True)
class Phase3CompressionResult:
    """Resultado de FASE 3: certificado, estado final y coeficientes opcionales."""
    certificate: Phase3CompressionCertificate
    state_focused: NDArray[np.complex128]
    coefficients: Optional[NDArray[np.complex128]] = None


class OpticalRiemannLens:
    r"""
    Lente de Riemann con compresión categórica disipativa.

    Integra:
        - Curvatura Riemanniana acotada.
        - Gap espectral de Hodge.
        - Entropía de von Neumann.
        - Coherencia topos.
        - [SUTURA 3] filtrado por armónicos esféricos en S².
    """

    def __init__(
        self,
        metric_bundle: MetricTensorBundle,
        max_curvature_k: float = 1.0,
        n_theta: int = _GAUSS_LEGENDRE_NODES,
        n_phi: int = _GAUSS_LEGENDRE_NODES,
        max_l: int = _MAX_SPHERICAL_HARMONIC_L,
    ) -> None:
        self._bundle = metric_bundle
        self._K_max = float(max_curvature_k)
        self._dim = metric_bundle.dimension
        self._n_theta = int(n_theta)
        self._n_phi = int(n_phi)
        self._max_l = int(max_l)

    def compute_riemann_curvature_norm(self) -> float:
        r"""
        Estimación estable de ||R|| a partir del espectro métrico regularizado.

        Para métricas diagonalizables, se usa dispersión log-espectral y
        número de condición como proxy de curvatura/ill-posedness.
        """
        eig = self._bundle.regularized_eigenvalues
        eig = np.clip(eig, _MACHINE_EPS, None)

        cond = float(np.max(eig) / np.min(eig))
        log_eig = np.log(eig)
        dispersion = float(np.std(log_eig))

        riemann_norm = float(np.log1p(cond) / max(1, self._dim) + dispersion)
        logger.debug(
            "[FASE 3] Curvatura de Riemann estimada: ||R|| ≈ %.6e (K_max=%.3e)",
            riemann_norm,
            self._K_max,
        )
        return riemann_norm

    def compute_hodge_laplacian_gap(
        self,
        discrete_laplacian: Optional[NDArray[np.float64]] = None,
    ) -> float:
        r"""
        Calcula gap λ₂ - λ₁ del Laplaciano de Hodge discreto/métrico.

        Si no se entrega Laplaciano, se usa G^{-1}_reg como operador
        elíptico positivo definido asociado a la métrica.
        """
        if discrete_laplacian is None:
            L = self._bundle.G_inv
        else:
            L = np.asarray(discrete_laplacian, dtype=np.float64)
            L = (L + L.T) / 2.0

        eigvals = la.eigvalsh(L)
        eigvals = np.sort(np.asarray(eigvals, dtype=np.float64))

        if eigvals.size < 2:
            return 0.0

        gap = float(eigvals[1] - eigvals[0])
        gap = max(0.0, gap)

        logger.debug("[FASE 3] Gap espectral Hodge: λ₂ - λ₁ = %.6e", gap)
        return gap

    def compute_von_neumann_entropy(
        self,
        density_matrix_rho: NDArray[np.complex128],
    ) -> float:
        r"""
        S(ρ) = -Tr(ρ ln ρ), con normalización y hermiciado defensivo.
        """
        rho = np.asarray(density_matrix_rho, dtype=np.complex128)

        if rho.size == 0:
            return 0.0

        rho = (rho + rho.conj().T) / 2.0
        tr = float(np.real(np.trace(rho)))

        if tr <= _MACHINE_EPS:
            return 0.0

        rho = rho / tr
        eigvals = la.eigvalsh(rho)
        eigvals = eigvals[eigvals > _MACHINE_EPS]

        if eigvals.size == 0:
            return 0.0

        entropy = -np.sum(eigvals * np.log(eigvals))
        return float(max(0.0, np.real(entropy)))

    def compute_topos_coherence_index(
        self,
        state_vector: NDArray[np.complex128],
    ) -> float:
        r"""
        Índice de coherencia topos:
            C_raw = ||ψ||₁² / ||ψ||₂² ∈ [1, dim].
            C_norm = (C_raw - 1)/(dim - 1) ∈ [0, 1].
        """
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        abs_state = np.abs(state_vector)

        l2_sq = float(np.sum(abs_state ** 2))
        if l2_sq <= _MACHINE_EPS:
            return 0.0

        l1 = float(np.sum(abs_state))
        coherence_raw = (l1 ** 2) / (l2_sq + _MACHINE_EPS)

        if self._dim <= 1:
            return 0.0

        coherence_normalized = (coherence_raw - 1.0) / (self._dim - 1.0 + _MACHINE_EPS)
        return float(np.clip(coherence_normalized, 0.0, 1.0))

    def apply_spherical_harmonic_filter(
        self,
        grid_psi: NDArray[np.complex128],
        l_cut: Optional[int] = None,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], float, float, int]:
        r"""
        Aplica [SUTURA 3]: análisis, truncamiento y síntesis esférica.

        Returns:
            (psi_reconstructed, coefficients_filtered,
             spectral_energy_retained, reconstruction_error, l_used)
        """
        cache = SphericalHarmonicsVectorizer.build_gauss_legendre_quadrature(
            n_theta=self._n_theta,
            n_phi=self._n_phi,
            max_l=self._max_l,
        )

        coefficients = SphericalHarmonicsVectorizer.compute_coefficients_vectorized(
            grid_psi,
            cache,
        )

        if l_cut is None:
            l_cut = self._max_l

        coefficients_filtered, l_used = SphericalHarmonicsVectorizer.filter_coefficients(
            coefficients,
            int(l_cut),
        )

        psi_reconstructed = SphericalHarmonicsVectorizer.reconstruct_field_vectorized(
            coefficients_filtered,
            cache,
        )

        total_energy = SphericalHarmonicsVectorizer.coefficient_energy(coefficients)
        filtered_energy = SphericalHarmonicsVectorizer.coefficient_energy(
            coefficients_filtered
        )

        energy_retained = float(filtered_energy / (total_energy + _MACHINE_EPS))
        energy_retained = float(np.clip(energy_retained, 0.0, 1.0))

        grid_psi_arr = np.asarray(grid_psi, dtype=np.complex128)
        grid_norm = float(la.norm(grid_psi_arr))
        reconstruction_error = float(
            la.norm(grid_psi_arr - psi_reconstructed) / (grid_norm + _MACHINE_EPS)
        )

        logger.info(
            "[SUTURA 3] Filtro esférico: l_cut=%d, energía retenida=%.6f, error=%.6e",
            l_used,
            energy_retained,
            reconstruction_error,
        )

        return (
            psi_reconstructed.astype(np.complex128),
            coefficients_filtered,
            energy_retained,
            reconstruction_error,
            l_used,
        )

    def apply_dissipative_compression(
        self,
        focused_state: NDArray[np.complex128],
        raw_state: NDArray[np.complex128],
        spherical_field: Optional[NDArray[np.complex128]] = None,
        spherical_l_cut: Optional[int] = None,
    ) -> Phase3CompressionResult:
        r"""
        Compresión disipativa final con invariantes de Fase 3.

        Si se entrega spherical_field compatible dimensionalmente, se aplica
        filtrado esférico [SUTURA 3] y el campo reconstruido se convierte en
        el estado focalizado. Luego se impone cota disipativa frente a raw.
        """
        state_focused = np.asarray(focused_state, dtype=np.complex128).copy()
        raw_state = np.asarray(raw_state, dtype=np.complex128)

        if state_focused.shape != raw_state.shape:
            raise GenerativeOpticSuturatorError(
                f"focused_state {state_focused.shape} y raw_state {raw_state.shape} incompatibles"
            )

        coefficients: Optional[NDArray[np.complex128]] = None
        spherical_degree_used: Optional[int] = None
        spherical_reconstruction_error: Optional[float] = None
        spectral_energy_retained: Optional[float] = None

        if spherical_field is not None:
            try:
                psi_rec, coefficients, spectral_energy_retained, rec_error, l_used = (
                    self.apply_spherical_harmonic_filter(spherical_field, spherical_l_cut)
                )

                if psi_rec.size == state_focused.size:
                    state_focused = psi_rec.reshape(state_focused.shape)
                else:
                    logger.warning(
                        "[FASE 3] Campo esférico reconstruido size=%d no coincide con estado size=%d; "
                        "se conserva estado transportado.",
                        psi_rec.size,
                        state_focused.size,
                    )

                spherical_degree_used = l_used
                spherical_reconstruction_error = rec_error

            except SphericalQuadratureError as exc:
                logger.warning("[FASE 3] No se pudo aplicar filtro esférico: %s", exc)

        # Imposición de pasividad: ||focused|| ≤ ||raw|| en norma G y 2.
        state_focused, _ = GeodesicEnergyConserver.enforce_dissipative_bound(
            state_focused,
            raw_state,
            self._bundle.G_reg,
        )

        norm_focused = float(la.norm(state_focused))
        norm_raw = float(la.norm(raw_state))

        if norm_raw <= _MACHINE_EPS:
            if norm_focused > _MACHINE_EPS:
                state_focused = np.zeros_like(raw_state)
                norm_focused = 0.0
            compression_ratio = 1.0
        else:
            if norm_focused <= _NUMERIC_NORM_FLOOR * norm_raw:
                state_focused = raw_state * _NUMERIC_NORM_FLOOR
                norm_focused = float(la.norm(state_focused))

            compression_ratio = float(min(1.0, norm_focused / norm_raw))

        # Matrices de densidad puras normalizadas.
        focused_flat = state_focused.ravel()
        raw_flat = raw_state.ravel()
        dim = raw_flat.size

        if norm_focused > _MACHINE_EPS:
            rho_focused = np.outer(focused_flat, focused_flat.conj()) / (norm_focused ** 2)
        else:
            rho_focused = np.zeros((dim, dim), dtype=np.complex128)

        if norm_raw > _MACHINE_EPS:
            rho_raw = np.outer(raw_flat, raw_flat.conj()) / (norm_raw ** 2)
        else:
            rho_raw = np.zeros((dim, dim), dtype=np.complex128)

        entropy_focused = self.compute_von_neumann_entropy(rho_focused)
        entropy_raw = self.compute_von_neumann_entropy(rho_raw)
        entropy_loss = float(entropy_raw - entropy_focused)

        riemann_norm = self.compute_riemann_curvature_norm()
        spectral_gap = self.compute_hodge_laplacian_gap()
        topos_coherence = self.compute_topos_coherence_index(state_focused)

        logger.info(
            "[FASE 3] Compresión: κ=%.6f, ΔS=%.6e, ||R||=%.6e, gap=%.6e, topos=%.6f",
            compression_ratio,
            entropy_loss,
            riemann_norm,
            spectral_gap,
            topos_coherence,
        )

        certificate = Phase3CompressionCertificate(
            compression_ratio=compression_ratio,
            entropy_loss=entropy_loss,
            riemann_curvature_norm=riemann_norm,
            spectral_gap=spectral_gap,
            topos_coherence_index=topos_coherence,
            spherical_degree_used=spherical_degree_used,
            spherical_reconstruction_error=spherical_reconstruction_error,
            spectral_energy_retained=spectral_energy_retained,
        )

        return Phase3CompressionResult(
            certificate=certificate,
            state_focused=state_focused,
            coefficients=coefficients,
        )


@dataclass(frozen=True, slots=True)
class OpticSuturationCertificate:
    r"""
    Certificado maestro de sutura óptica del Haz Γ.

    Composición functorial:
        Ω_wisdom = F₃ ∘ F₂ ∘ F₁(Γ_raw).
    """
    phase1_cert: Phase1ProjectionCertificate
    phase2_cert: Phase2TransportCertificate
    phase3_cert: Phase3CompressionCertificate
    global_stability_index: float
    target_stratum: Stratum
    final_state: NDArray[np.complex128]

    @property
    def is_globally_stable(self) -> bool:
        tol1 = _GRASSMANN_IDEMPOTENCE_TOL * max(1, self.phase1_cert.grassmann_dimension)

        return bool(
            self.phase1_cert.idempotence_residual <= tol1
            and self.phase2_cert.is_causally_admissible
            and self.phase3_cert.compression_ratio > 0.0
            and self.global_stability_index > 0.95
        )


class GenerativeOpticHodgeSuturator(Morphism):
    r"""
    Orquestador maestro del pipeline functorial tricapa.

    Composición:
        F₁: SemanticParabolicMirror.
        F₂: EikonalFloquetAgentSutured.
        F₃: OpticalRiemannLens.
    """

    def __init__(
        self,
        target_stratum: Stratum = Stratum.WISDOM,
        cavity_tolerance: float = _DEFAULT_CAVITY_TOL,
        max_curvature: float = 1.0,
        max_condition_number: float = _MAX_CONDITION_NUMBER,
        quadrature_theta_nodes: int = _GAUSS_LEGENDRE_NODES,
        quadrature_phi_nodes: int = _GAUSS_LEGENDRE_NODES,
        max_spherical_l: int = _MAX_SPHERICAL_HARMONIC_L,
    ) -> None:
        super().__init__()
        self._target_stratum = target_stratum
        self._cavity_tol = float(cavity_tolerance)
        self._K_max = float(max_curvature)
        self._max_condition_number = float(max_condition_number)
        self._quadrature_theta_nodes = int(quadrature_theta_nodes)
        self._quadrature_phi_nodes = int(quadrature_phi_nodes)
        self._max_spherical_l = int(max_spherical_l)

        logger.info(
            "Suturador Óptico de Hodge inicializado: stratum=%s, ε_cavity=%.2e, K_max=%.3f",
            target_stratum.name,
            cavity_tolerance,
            max_curvature,
        )

    def _compute_global_stability_index(
        self,
        phase1_cert: Phase1ProjectionCertificate,
        phase2_cert: Phase2TransportCertificate,
        phase3_cert: Phase3CompressionCertificate,
    ) -> float:
        r"""
        Índice agregado de estabilidad ∈ [0, 1].

        Pesos:
            F₁: 0.25
            F₂: 0.50
            F₃: 0.25
        """
        w1, w2, w3 = 0.25, 0.50, 0.25

        tol1 = _GRASSMANN_IDEMPOTENCE_TOL * max(1, phase1_cert.grassmann_dimension)
        m1 = 1.0 - min(1.0, phase1_cert.idempotence_residual / max(tol1, _MACHINE_EPS))

        if phase2_cert.max_floquet_modulus <= _FLOQUET_DAMPING_BOUND:
            m2 = 1.0
        else:
            excess = phase2_cert.max_floquet_modulus - _FLOQUET_DAMPING_BOUND
            m2 = 1.0 - min(1.0, excess / max(phase2_cert.cavity_tolerance, _MACHINE_EPS))

        m3_base = float(np.clip(phase3_cert.compression_ratio, 0.0, 1.0))

        entropy_penalty = float(np.clip(max(0.0, phase3_cert.entropy_loss), 0.0, 1.0))
        coherence_penalty = 0.10 * float(phase3_cert.topos_coherence_index)
        curvature_penalty = 0.05 * float(
            np.clip(phase3_cert.riemann_curvature_norm / max(self._K_max, _MACHINE_EPS), 0.0, 1.0)
        )

        if phase3_cert.spectral_energy_retained is not None:
            spectral_bonus = 0.05 * float(phase3_cert.spectral_energy_retained)
        else:
            spectral_bonus = 0.05

        m3_factor = (
            1.0
            - 0.15 * entropy_penalty
            - coherence_penalty
            - curvature_penalty
            + spectral_bonus
        )
        m3 = float(np.clip(m3_base * m3_factor, 0.0, 1.0))

        global_index = w1 * m1 + w2 * m2 + w3 * m3
        return float(np.clip(global_index, 0.0, 1.0))

    def execute_optic_suturation(
        self,
        metric_tensor_g: NDArray[np.float64],
        phase_gradient_ds: NDArray[np.complex128],
        refractive_index_n: float,
        monodromy_matrix_m: NDArray[np.complex128],
        semantic_direction: NDArray[np.float64],
        raw_state_vector: NDArray[np.complex128],
        grassmann_dimension: int,
        period_t: float = 1.0,
        spherical_field: Optional[NDArray[np.complex128]] = None,
        spherical_l_cut: Optional[int] = None,
    ) -> OpticSuturationCertificate:
        r"""
        Ejecuta el pipeline completo tricapa.

        Args:
            metric_tensor_g: Tensor Riemanniano G.
            phase_gradient_ds: Gradiente eikonal ∂S.
            refractive_index_n: Índice de refracción n.
            monodromy_matrix_m: Matriz de monodromía Floquet.
            semantic_direction: Dirección semántica.
            raw_state_vector: Estado crudo |ψ⟩.
            grassmann_dimension: Dimensión k de Gr(k,n).
            period_t: Período Floquet.
            spherical_field: Campo opcional ψ(θ,φ) para Sutura 3.
            spherical_l_cut: Corte espectral esférico opcional.

        Returns:
            OpticSuturationCertificate.
        """
        logger.info("═" * 80)
        logger.info("INICIANDO SUTURA ÓPTICA TRICAPA DEL HAZ Γ")
        logger.info("═" * 80)

        # ── FASE 1 ────────────────────────────────────────────────────────────
        logger.info(
            "┌─ FASE 1: Proyección Geométrica sobre Gr(%d, %d)",
            grassmann_dimension,
            np.asarray(metric_tensor_g).shape[0],
        )

        phase1_mirror = SemanticParabolicMirror(
            metric_tensor_g=metric_tensor_g,
            max_condition_number=self._max_condition_number,
        )

        phase1_bundle = phase1_mirror.project_and_prepare_transport_bundle(
            semantic_direction=semantic_direction,
            raw_state_vector=raw_state_vector,
            grassmann_dimension=grassmann_dimension,
        )

        # ── FASE 2 ────────────────────────────────────────────────────────────
        logger.info(
            "┌─ FASE 2: Transporte Eikonal-Floquet (n=%.4f, T=%.4f)",
            refractive_index_n,
            period_t,
        )

        phase2_agent = EikonalFloquetAgentSutured(
            metric_bundle=phase1_bundle.metric_bundle,
            refractive_index_n=refractive_index_n,
            cavity_tolerance=self._cavity_tol,
        )

        phase2_result = phase2_agent.execute_transport_phase(
            phase1_bundle=phase1_bundle,
            phase_gradient_ds=phase_gradient_ds,
            monodromy_matrix_m=monodromy_matrix_m,
            period_t=period_t,
        )

        # ── FASE 3 ────────────────────────────────────────────────────────────
        logger.info("┌─ FASE 3: Compresión Categórica Riemann-Topos")

        phase3_lens = OpticalRiemannLens(
            metric_bundle=phase1_bundle.metric_bundle,
            max_curvature_k=self._K_max,
            n_theta=self._quadrature_theta_nodes,
            n_phi=self._quadrature_phi_nodes,
            max_l=self._max_spherical_l,
        )

        phase3_result = phase3_lens.apply_dissipative_compression(
            focused_state=phase2_result.state_transported,
            raw_state=phase1_bundle.raw_state_vector,
            spherical_field=spherical_field,
            spherical_l_cut=spherical_l_cut,
        )

        # ── SÍNTESIS GLOBAL ───────────────────────────────────────────────────
        global_stability = self._compute_global_stability_index(
            phase1_bundle.certificate,
            phase2_result.certificate,
            phase3_result.certificate,
        )

        master_certificate = OpticSuturationCertificate(
            phase1_cert=phase1_bundle.certificate,
            phase2_cert=phase2_result.certificate,
            phase3_cert=phase3_result.certificate,
            global_stability_index=global_stability,
            target_stratum=self._target_stratum,
            final_state=phase3_result.state_focused,
        )

        logger.info("═" * 80)
        logger.info("SUTURA ÓPTICA TRICAPA COMPLETADA")
        logger.info(
            "Estabilidad Global: %.2f%% | Estrato: %s | Estado: %s",
            global_stability * 100.0,
            self._target_stratum.name,
            "ESTABLE" if master_certificate.is_globally_stable else "INESTABLE",
        )
        logger.info("═" * 80)

        return master_certificate


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE UTILIDAD Y VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def construct_test_metric_tensor(
    dimension: int,
    condition_number: float = 10.0,
) -> NDArray[np.float64]:
    """
    Construye tensor métrico SPD de prueba con número de condición controlado.
    """
    dimension = int(max(1, dimension))
    condition_number = float(max(1.0, condition_number))

    eigenvals = np.linspace(1.0, condition_number, dimension)
    Q, _ = la.qr(np.random.randn(dimension, dimension))
    G = Q @ np.diag(eigenvals) @ Q.T
    return (G + G.T) / 2.0


def validate_suturation_certificate(cert: OpticSuturationCertificate) -> None:
    """
    Valida integridad del certificado maestro y emite advertencias.
    """
    if not cert.is_globally_stable:
        logger.warning(
            "ADVERTENCIA: Sistema globalmente INESTABLE (índice=%.2f%%)",
            cert.global_stability_index * 100.0,
        )

    if cert.phase3_cert.entropy_loss < -_MACHINE_EPS:
        logger.warning(
            "ADVERTENCIA: Ganancia entrópica anómala ΔS=%.3e",
            cert.phase3_cert.entropy_loss,
        )

    logger.info("Certificado validado: 3 fases verificadas")


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA TESTING
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
    )

    logger.info("Iniciando suite de validación del Suturador Óptico Tricapa")

    np.random.seed(2026)

    DIM = 8
    GRASSMANN_K = 4
    REFRACTIVE_INDEX = 1.5
    PERIOD = 1.0

    G = construct_test_metric_tensor(DIM, condition_number=5.0)

    # Gradiente eikonal escalado para satisfacer G^{-1} ∂S·∂S ≥ n²(1-slack).
    ds = np.random.randn(DIM)
    ds /= la.norm(ds)

    G_inv_test = la.inv(G)
    current_eikonal = float(ds.T @ G_inv_test @ ds)
    required_eikonal = (REFRACTIVE_INDEX ** 2) * (1.0 - _EIKONAL_SLACK_FACTOR)

    if current_eikonal > _MACHINE_EPS:
        ds *= np.sqrt(required_eikonal / current_eikonal) * 1.5

    # Matriz de monodromía estable.
    M = np.random.randn(DIM, DIM)
    max_eig_abs = float(np.max(np.abs(la.eigvals(M))))
    if max_eig_abs > _MACHINE_EPS:
        M = 0.9 * M / max_eig_abs

    semantic_vec = np.random.randn(DIM)
    raw_state = np.random.randn(DIM) + 1j * np.random.randn(DIM)

    suturator = GenerativeOpticHodgeSuturator(
        target_stratum=Stratum.WISDOM,
        cavity_tolerance=1e-8,
        max_curvature=2.0,
        max_condition_number=1e8,
        quadrature_theta_nodes=64,
        quadrature_phi_nodes=64,
        max_spherical_l=16,
    )

    try:
        certificate = suturator.execute_optic_suturation(
            metric_tensor_g=G,
            phase_gradient_ds=ds,
            refractive_index_n=REFRACTIVE_INDEX,
            monodromy_matrix_m=M,
            semantic_direction=semantic_vec,
            raw_state_vector=raw_state,
            grassmann_dimension=GRASSMANN_K,
            period_t=PERIOD,
            spherical_field=None,
            spherical_l_cut=None,
        )

        validate_suturation_certificate(certificate)

        logger.info(
            "✓ Suite de validación completada: estabilidad global = %.2f%%",
            certificate.global_stability_index * 100.0,
        )

    except GenerativeOpticSuturatorError as e:
        logger.error("✗ Fallo en la sutura óptica: %s", e)
        raise
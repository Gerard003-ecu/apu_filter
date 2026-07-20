# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Semantic Translator Agent (Endofuntor de Difeomorfismo Semántico)   ║
║ Ruta   : app/agents/wisdom/semantic_translator_agent.py                      ║
║ Versión: 4.0.0-Riemannian-Galois-Gibbs-Doctoral-3Phases                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DIFERENCIAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna al `semantic_translator.py` en el Estrato Ω (WISDOM).
Subordina la probabilidad estocástica del Modelo de Lenguaje (LLM) al determinismo
del espacio físico, despojándolo de libre albedrío decisional.

ARQUITECTURA DE 3 FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Certificación Métrico-Ordenada:
         Φ₁(G, x, y, V) = (d_M(x,y), ⨆V)
         donde:
             d_M(x,y) = sqrt((x-y)^T G^{-1} (x-y))
             ⨆V = supremo en el retículo de severidad.

Fase 2 → Adjunción de Galois:
         Φ₂(Φ₁(...)) = Hom_D(F(X), Y) ≅ Hom_C(X, G(Y))
         Se audita la reversibilidad X ≅ G(F(X)) y se veta la entropía retórica.

Fase 3 → Modulación Termodinámica:
         Φ₃(Φ₂(Φ₁(...))) = P(v) = exp(-E(v)/(k_B T_gov)) / Z
         con T_gov derivada de la Traza de Dixmier y estabilidad basal Ψ.

NOTA DE COMPOSICIÓN:
────────────────────
El último método de la Fase 1 retorna un `Phase1CertificationBridge`.
Ese objeto es la continuación formal de la Fase 1 y el argumento inicial
explícito del primer método de la Fase 2.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Callable, List, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (Stubs de aislamiento)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos $\mathcal{E}_{MIC}$."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MAC.Wisdom.SemanticTranslatorAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. TIPOS Y CONSTANTES FÍSICO-MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════════════════════
VectorF64 = NDArray[np.float64]
MatrixF64 = NDArray[np.float64]

_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)

# Condicionamiento espectral máximo admisible para el tensor métrico.
_KAPPA_MAX: float = 1e8

# Radio máximo de la bola geodésica admisible en distancia de Mahalanobis.
_MAHALANOBIS_TOLERANCE: float = 5.0

# Tolerancias para la reversibilidad de la Adjunción de Galois.
_GALOIS_ABSOLUTE_TOLERANCE: float = 1e-4
_GALOIS_RELATIVE_TOLERANCE: float = 1e-4
_GALOIS_ISOMORPHISM_TOLERANCE: float = _GALOIS_ABSOLUTE_TOLERANCE

# Constante de Boltzmann normalizada en el ciber-espacio de gobierno.
_BOLTZMANN_K_CYBER: float = 1.0

# Límite práctico para argumentos exponenciales seguros.
_MAX_EXP_ARGUMENT: float = 745.0

# Piso térmico numérico para evitar división por cero.
_THERMAL_TEMPERATURE_FLOOR: float = _MACHINE_EPSILON


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES DEL DIFEOMORFISMO
# ═══════════════════════════════════════════════════════════════════════════════
class SemanticTranslatorAgentError(TopologicalInvariantError):
    """Excepción raíz del Endofuntor de Traducción Semántica."""
    pass


class RiemannianMetricDegeneracyError(SemanticTranslatorAgentError):
    r"""Detonada si $\kappa(G^{-1}) \gg 1$ o si el tensor pierde positividad definida."""
    pass


class LatticeCollapseViolation(SemanticTranslatorAgentError):
    r"""Detonada si el LLM intenta subvertir el supremo $\bigsqcup v_i$."""
    pass


class SemanticDriftVetoError(SemanticTranslatorAgentError):
    r"""Detonada si la Adjunción de Galois falla: $X \not\cong G(F(X))$."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES Y RETÍCULOS (DTOs del Topos)
# ═══════════════════════════════════════════════════════════════════════════════
@unique
class VerdictLevel(IntEnum):
    r"""
    Retículo Distributivo Acotado (Bounded Distributive Lattice).

    Orden de severidad:
        VIABLE      = ⊥
        CONDICIONAL
        PRECAUCION
        RECHAZAR    = ⊤
    """
    VIABLE = 0
    CONDICIONAL = 1
    PRECAUCION = 2
    RECHAZAR = 3


@dataclass(frozen=True, slots=True, eq=False)
class RiemannianRetrievalData:
    r"""
    Artefacto de Fase 1.
    Certificado de distancia de Mahalanobis y regularización métrica.
    """
    mahalanobis_distance: float
    is_geodesically_valid: bool
    regularized_metric: MatrixF64
    condition_number: float
    tikhonov_shift: float


@dataclass(frozen=True, slots=True, eq=False)
class LatticeCollapseState:
    r"""
    Artefacto de Fase 1.
    Colapso determinista del veredicto en el retículo de severidad.
    """
    supremum_verdict: VerdictLevel
    is_worst_case_enforced: bool
    verdict_cardinality: int


@dataclass(frozen=True, slots=True, eq=False)
class GaloisAdjunctionAudit:
    r"""
    Artefacto de Fase 2.
    Auditoría de isomorfismo de la traducción semántica.
    """
    topological_residual: float
    topological_relative_residual: float
    is_isomorphism_preserved: bool


@dataclass(frozen=True, slots=True, eq=False)
class ThermodynamicGovernanceState:
    r"""
    Artefacto de Fase 3.
    Estado térmico que congela los grados de libertad estocásticos del LLM.
    """
    governance_temperature: float
    inverse_temperature_beta: float
    gibbs_distribution: VectorF64
    partition_function: float


@dataclass(frozen=True, slots=True, eq=False)
class Phase1CertificationBridge:
    r"""
    Puente funtorial Φ₁ → Φ₂.

    Este objeto es emitido por el último método de la Fase 1 y constituye
    la entrada formal del primer método de la Fase 2.
    """
    riemannian_audit: RiemannianRetrievalData
    lattice_collapse: LatticeCollapseState
    topological_coords: VectorF64
    narrative_vector_y: VectorF64


@dataclass(frozen=True, slots=True, eq=False)
class Phase2GaloisAuditBridge:
    r"""
    Puente funtorial Φ₂ → Φ₃.
    """
    phase1_bridge: Phase1CertificationBridge
    galois_adjunction: GaloisAdjunctionAudit


@dataclass(frozen=True, slots=True, eq=False)
class DiplomaticTranslationState:
    r"""
    Objeto final del endofuntor $\mathcal{Z}_{Translator}$.
    """
    riemannian_audit: RiemannianRetrievalData
    lattice_collapse: LatticeCollapseState
    galois_adjunction: GaloisAdjunctionAudit
    thermodynamic_state: ThermodynamicGovernanceState
    approved_narrative_vector: VectorF64


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: CERTIFICACIÓN MÉTRICA RIEMANNIANA Y COLAPSO RETICULAR             ║
# ║                                                                             ║
# ║   Φ₁(G, x, y, V) = (d_M(x,y), ⨆V)                                           ║
# ║                                                                             ║
# ║   1. Valida el tensor métrico G como SPD regularizado.                       ║
# ║   2. Computa la distancia de Mahalanobis en el espacio tangente.             ║
# ║   3. Impone el supremo algebraico de severidad en el retículo.              ║
# ║   4. Emite el puente formal hacia la Fase 2.                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_RiemannianLatticeCertifier:
    r"""
    Fase 1 del endofuntor.

    Responsabilidades:
    - Sanidad numérica de vectores y matrices.
    - Regularización Tikhonov espectral de la métrica Riemanniana.
    - Certificación de la distancia de Mahalanobis.
    - Colapso determinista del retículo de veredictos.
    - Emisión del puente funtorial hacia la Fase 2.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Validación numérica elemental
    # ─────────────────────────────────────────────────────────────────────────
    def _as_finite_float(self, name: str, value: float) -> float:
        """
        Convierte un valor a float64 y exige que sea finito.
        """
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SemanticTranslatorAgentError(
                f"{name} no puede convertirse a un escalar float64."
            ) from exc

        if arr.ndim != 0:
            raise SemanticTranslatorAgentError(
                f"{name} debe ser un escalar, no un arreglo de dimensión {arr.ndim}."
            )

        scalar = float(arr)
        if not math.isfinite(scalar):
            raise SemanticTranslatorAgentError(
                f"{name} debe ser finito. Se recibió {scalar!r}."
            )
        return scalar

    def _as_finite_vector(self, name: str, value: VectorF64) -> VectorF64:
        """
        Valida que el objeto sea un vector 1-D no vacío y con componentes finitas.
        """
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SemanticTranslatorAgentError(
                f"{name} no puede convertirse a un vector float64."
            ) from exc

        if arr.ndim != 1:
            raise SemanticTranslatorAgentError(
                f"{name} debe ser un vector 1-D. Dimensión recibida: {arr.ndim}."
            )

        if arr.size == 0:
            raise SemanticTranslatorAgentError(
                f"{name} no puede ser el vector vacío."
            )

        if not np.all(np.isfinite(arr)):
            raise SemanticTranslatorAgentError(
                f"{name} contiene componentes NaN o infinitas."
            )

        return arr

    def _as_finite_square_matrix(self, name: str, value: MatrixF64) -> MatrixF64:
        """
        Valida que el objeto sea una matriz cuadrada no vacía y finita.
        """
        try:
            mat = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SemanticTranslatorAgentError(
                f"{name} no puede convertirse a una matriz float64."
            ) from exc

        if mat.ndim != 2:
            raise SemanticTranslatorAgentError(
                f"{name} debe ser una matriz 2-D. Dimensión recibida: {mat.ndim}."
            )

        if mat.size == 0 or mat.shape[0] != mat.shape[1]:
            raise SemanticTranslatorAgentError(
                f"{name} debe ser una matriz cuadrada no vacía. "
                f"Shape recibido: {getattr(mat, 'shape', None)}."
            )

        if not np.all(np.isfinite(mat)):
            raise SemanticTranslatorAgentError(
                f"{name} contiene entradas NaN o infinitas."
            )

        return mat

    # ─────────────────────────────────────────────────────────────────────────
    # Normas numéricamente seguras
    # ─────────────────────────────────────────────────────────────────────────
    def _safe_l2_norm(self, vector: VectorF64) -> float:
        """
        Calcula ||v||₂ con reescalado para evitar overflow/underflow.
        """
        if vector.size == 0:
            return 0.0

        scale = float(np.max(np.abs(vector)))
        if scale == 0.0:
            return 0.0

        if not math.isfinite(scale):
            return math.inf

        scaled = vector / scale
        ss = float(np.dot(scaled, scaled))

        if not math.isfinite(ss):
            return math.inf

        norm = scale * math.sqrt(ss)
        return float(norm) if math.isfinite(norm) else math.inf

    def _safe_fro_norm(self, matrix: MatrixF64) -> float:
        """
        Calcula ||M||_F con reescalado para evitar overflow/underflow.
        """
        if matrix.size == 0:
            return 0.0

        scale = float(np.max(np.abs(matrix)))
        if scale == 0.0:
            return 0.0

        if not math.isfinite(scale):
            return math.inf

        scaled = matrix / scale
        ss = float(np.sum(scaled * scaled))

        if not math.isfinite(ss):
            return math.inf

        norm = scale * math.sqrt(ss)
        return float(norm) if math.isfinite(norm) else math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # Regularización espectral del tensor métrico Riemanniano
    # ─────────────────────────────────────────────────────────────────────────
    def _regularize_spd_metric(
        self,
        G_metric: MatrixF64
    ) -> Tuple[MatrixF64, float, float]:
        r"""
        Regulariza $G$ para garantizar que sea simétrico y definido positivo,
        con número de condición acotado por $\kappa_{\max}$.

        Estrategia:
        1. Simetrización: $G \leftarrow \frac{1}{2}(G + G^T)$.
        2. Análisis espectral con `eigvalsh`.
        3. Si $\lambda_{\min} \le \delta$ o $\kappa > \kappa_{\max}$,
           se aplica desplazamiento de Tikhonov:
              $G_{\text{eff}} = G + \tau I$.
        4. Se verifica positividad definida y condicionamiento final.

        Returns
        -------
        Tuple[MatrixF64, float, float]
            (G_eff, kappa_eff, tau)

        Raises
        ------
        RiemannianMetricDegeneracyError
            Si no puede garantizarse una métrica SPD bien condicionada.
        """
        G = self._as_finite_square_matrix("G_metric", G_metric)
        n = G.shape[0]

        # Simetrización explícita del tensor métrico.
        G_sym = 0.5 * (G + G.T)

        # Diagnóstico de asimetría original.
        fro_sym = self._safe_fro_norm(G_sym)
        fro_asym = self._safe_fro_norm(G - G_sym)

        if math.isfinite(fro_sym) and math.isfinite(fro_asym):
            if fro_asym > 1e-8 * max(1.0, fro_sym):
                logger.warning(
                    "Tensor métrico con asimetría relevante "
                    f"(||G-G^T||_F={fro_asym:.3e}). Se impone simetrización."
                )
        elif not math.isfinite(fro_asym):
            logger.warning(
                "No fue posible certificar finiteza de la asimetría métrica. "
                "Se procede bajo simetrización forzada."
            )

        # Espectro de la parte simétrica.
        try:
            eigvals = np.linalg.eigvalsh(G_sym)
        except la.LinAlgError as exc:
            raise RiemannianMetricDegeneracyError(
                "Fallo en la descomposición espectral de la métrica simétrica."
            ) from exc

        if not np.all(np.isfinite(eigvals)):
            raise RiemannianMetricDegeneracyError(
                "El espectro de la métrica contiene valores no finitos."
            )

        lambda_min = float(np.min(eigvals))
        lambda_max = float(np.max(eigvals))

        scale = max(1.0, abs(lambda_min), abs(lambda_max))
        delta = _MACHINE_EPSILON * scale

        # Condiciones que exigen regularización.
        needs_positive = lambda_min <= delta

        safe_min_for_cond = max(lambda_min, _MACHINE_EPSILON)
        needs_condition = (
            lambda_max > 0.0
            and (
                lambda_min <= 0.0
                or (lambda_max / safe_min_for_cond) > _KAPPA_MAX
            )
        )

        tau = 0.0
        G_eff = G_sym

        if needs_positive or needs_condition:
            tau_candidates: List[float] = [delta]

            # Garantizar positividad definida.
            if needs_positive:
                tau_candidates.append(delta - lambda_min)

            # Garantizar condicionamiento máximo.
            if _KAPPA_MAX > 1.0:
                numerator = lambda_max - _KAPPA_MAX * lambda_min
                if math.isfinite(numerator):
                    tau_candidates.append(numerator / (_KAPPA_MAX - 1.0))

            valid_tau_candidates = [
                c for c in tau_candidates
                if math.isfinite(c) and c >= 0.0
            ]

            if not valid_tau_candidates:
                raise RiemannianMetricDegeneracyError(
                    "No existe un desplazamiento de Tikhonov finito y no negativo "
                    "capaz de regularizar la métrica."
                )

            tau = max(valid_tau_candidates)

            if not math.isfinite(tau):
                raise RiemannianMetricDegeneracyError(
                    f"Desplazamiento de Tikhonov no finito: tau={tau!r}."
                )

            G_eff = G_sym + tau * np.eye(n, dtype=np.float64)

            if not np.all(np.isfinite(G_eff)):
                raise RiemannianMetricDegeneracyError(
                    "La métrica regularizada contiene entradas no finitas."
                )

            try:
                eigvals_eff = np.linalg.eigvalsh(G_eff)
            except la.LinAlgError as exc:
                raise RiemannianMetricDegeneracyError(
                    "Fallo espectral tras la regularización de Tikhonov."
                ) from exc

            if not np.all(np.isfinite(eigvals_eff)):
                raise RiemannianMetricDegeneracyError(
                    "El espectro regularizado contiene valores no finitos."
                )

            lambda_min_eff = float(np.min(eigvals_eff))
            lambda_max_eff = float(np.max(eigvals_eff))

            if lambda_min_eff <= 0.0:
                raise RiemannianMetricDegeneracyError(
                    f"Métrica regularizada no definida positiva: "
                    f"lambda_min_eff={lambda_min_eff:.6e}."
                )

            kappa_eff = float(lambda_max_eff / lambda_min_eff)

            if not math.isfinite(kappa_eff):
                raise RiemannianMetricDegeneracyError(
                    "Número de condición no finito tras regularización."
                )

            # Margen numérico mínimo para evitar falsos positivos por redondeo.
            if kappa_eff > _KAPPA_MAX * (1.0 + 10.0 * _MACHINE_EPSILON):
                raise RiemannianMetricDegeneracyError(
                    f"Condicionamiento espectral inadmisible tras Tikhonov: "
                    f"kappa={kappa_eff:.6e} > kappa_max={_KAPPA_MAX:.6e}."
                )

            logger.warning(
                "Métrica regularizada con Tikhonov. "
                f"tau={tau:.6e}, kappa_eff={kappa_eff:.6e}."
            )

            return G_eff, kappa_eff, tau

        # Rama sin regularización explícita.
        if lambda_min <= 0.0:
            raise RiemannianMetricDegeneracyError(
                f"Métrica no definida positiva: lambda_min={lambda_min:.6e}."
            )

        kappa_eff = float(lambda_max / lambda_min) if lambda_min > 0.0 else math.inf

        if not math.isfinite(kappa_eff):
            raise RiemannianMetricDegeneracyError(
                "Número de condición no finito en la métrica original."
            )

        if kappa_eff > _KAPPA_MAX:
            raise RiemannianMetricDegeneracyError(
                f"Condicionamiento espectral inadmisible: "
                f"kappa={kappa_eff:.6e} > kappa_max={_KAPPA_MAX:.6e}."
            )

        return G_eff, kappa_eff, tau

    # ─────────────────────────────────────────────────────────────────────────
    # Solución lineal sobre matriz SPD
    # ─────────────────────────────────────────────────────────────────────────
    def _solve_spd_linear_system(
        self,
        A: MatrixF64,
        b: VectorF64
    ) -> VectorF64:
        r"""
        Resuelve $A z = b$ asumiendo $A$ SPD.

        Se intenta factorización de Cholesky; si falla, se usa solver SPD.
        """
        try:
            cho = la.cho_factor(A, lower=True, check_finite=False)
            z = la.cho_solve(cho, b, check_finite=False)
            return np.asarray(z, dtype=np.float64)
        except la.LinAlgError:
            try:
                z = la.solve(A, b, assume_a="pos", check_finite=False)
                return np.asarray(z, dtype=np.float64)
            except (la.LinAlgError, ValueError, TypeError) as exc:
                raise RiemannianMetricDegeneracyError(
                    "El tensor métrico regularizado no admite solución SPD estable."
                ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # Certificación Riemanniana de la recuperación semántica
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_riemannian_retrieval(
        self,
        G_metric: MatrixF64,
        finding_vector: VectorF64,
        narrative_vector: VectorF64
    ) -> RiemannianRetrievalData:
        r"""
        Computa y certifica:
            $d_M(x, y) = \sqrt{(x - y)^T G^{-1} (x - y)}$.

        Raises
        ------
        SemanticTranslatorAgentError
            Si la distancia excede el radio de inyectividad admisible.
        RiemannianMetricDegeneracyError
            Si la métrica es degenerada o numéricamente inviable.
        """
        G = self._as_finite_square_matrix("G_metric", G_metric)
        x = self._as_finite_vector("finding_vector", finding_vector)
        y = self._as_finite_vector("narrative_vector", narrative_vector)

        if x.shape != y.shape:
            raise SemanticTranslatorAgentError(
                f"Vectores incompatibles: finding_vector={x.shape}, "
                f"narrative_vector={y.shape}."
            )

        if G.shape[0] != x.size:
            raise SemanticTranslatorAgentError(
                f"Dimensión métrica incompatible: G={G.shape}, vector_dim={x.size}."
            )

        G_eff, kappa_eff, tau = self._regularize_spd_metric(G)

        diff = x - y
        diff_norm = self._safe_l2_norm(diff)

        if not math.isfinite(diff_norm):
            raise SemanticTranslatorAgentError(
                "La diferencia vectorial x-y tiene norma no finita."
            )

        z = self._solve_spd_linear_system(G_eff, diff)

        d_m_sq = float(np.dot(diff, z))

        if not math.isfinite(d_m_sq):
            raise RiemannianMetricDegeneracyError(
                "Distancia de Mahalanobis al cuadrado no finita."
            )

        # Salvaguarda numérica ante pequeñas negatividades por redondeo.
        if d_m_sq < 0.0:
            negative_floor = -100.0 * _MACHINE_EPSILON * max(1.0, diff_norm * diff_norm)
            if d_m_sq >= negative_floor:
                d_m_sq = 0.0
            else:
                raise RiemannianMetricDegeneracyError(
                    "Forma cuadrática de Mahalanobis negativa más allá del margen numérico."
                )

        d_m = float(math.sqrt(d_m_sq))

        if d_m > _MAHALANOBIS_TOLERANCE:
            raise SemanticTranslatorAgentError(
                "Recuperación semántica ortogonal a la geodésica de negocio. "
                f"Distancia de Mahalanobis ({d_m:.6f}) excede el radio de "
                f"inyectividad ({_MAHALANOBIS_TOLERANCE:.6f})."
            )

        return RiemannianRetrievalData(
            mahalanobis_distance=d_m,
            is_geodesically_valid=True,
            regularized_metric=G_eff,
            condition_number=kappa_eff,
            tikhonov_shift=tau
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Colapso supremo en el retículo distributivo acotado
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_lattice_supremum_collapse(
        self,
        verdicts: List[VerdictLevel]
    ) -> LatticeCollapseState:
        r"""
        Computa el supremo $\bigsqcup V$ en el retículo de severidad.

        Axioma crítico:
            Si $\top \in V$, entonces $\bigsqcup V = \top$.
        """
        if verdicts is None:
            raise LatticeCollapseViolation(
                "El conjunto de veredictos entrantes es None."
            )

        try:
            raw_verdicts = list(verdicts)
        except TypeError as exc:
            raise LatticeCollapseViolation(
                "El conjunto de veredictos no es iterable."
            ) from exc

        if not raw_verdicts:
            raise LatticeCollapseViolation(
                "El conjunto de veredictos entrantes es el conjunto vacío ∅."
            )

        clean_verdicts: List[VerdictLevel] = []

        for idx, verdict in enumerate(raw_verdicts):
            if isinstance(verdict, VerdictLevel):
                clean_verdicts.append(verdict)
                continue

            try:
                clean_verdicts.append(VerdictLevel(int(verdict)))
                continue
            except (TypeError, ValueError):
                pass

            try:
                clean_verdicts.append(VerdictLevel[str(verdict)])
                continue
            except KeyError as exc:
                raise LatticeCollapseViolation(
                    f"Veredicto inválido en índice {idx}: {verdict!r}."
                ) from exc

        supremum = max(clean_verdicts, key=lambda v: v.value)

        if VerdictLevel.RECHAZAR in clean_verdicts and supremum != VerdictLevel.RECHAZAR:
            raise LatticeCollapseViolation(
                "Fallo algebraico: ⊥ ⊔ ⊤ ≠ ⊤. El supremo no fue RECHAZAR."
            )

        return LatticeCollapseState(
            supremum_verdict=supremum,
            is_worst_case_enforced=True,
            verdict_cardinality=len(clean_verdicts)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ÚLTIMO MÉTODO DE FASE 1
    # Puente formal hacia FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _complete_phase1_geometric_order_certification(
        self,
        G_metric: MatrixF64,
        topological_coords: VectorF64,
        narrative_vector_y: VectorF64,
        input_verdicts: List[VerdictLevel]
    ) -> Phase1CertificationBridge:
        r"""
        Último método de la Fase 1.

        Ejecuta:
            1. Certificación Riemanniana.
            2. Colapso reticular.
            3. Emisión del puente funtorial hacia la Fase 2.

        Este retorno es la continuación formal de la Fase 1 y el argumento
        inicial obligatorio del primer método de la Fase 2.
        """
        x = self._as_finite_vector("topological_coords", topological_coords)
        y = self._as_finite_vector("narrative_vector_y", narrative_vector_y)
        G = self._as_finite_square_matrix("G_metric", G_metric)

        riemannian_audit = self._certify_riemannian_retrieval(
            G_metric=G,
            finding_vector=x,
            narrative_vector=y
        )

        lattice_collapse = self._enforce_lattice_supremum_collapse(input_verdicts)

        return Phase1CertificationBridge(
            riemannian_audit=riemannian_audit,
            lattice_collapse=lattice_collapse,
            topological_coords=x,
            narrative_vector_y=y
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: AUDITORÍA DE LA ADJUNCIÓN DE GALOIS                               ║
# ║                                                                             ║
# ║   Φ₂(Φ₁(...)) = Hom_D(F(X), Y) ≅ Hom_C(X, G(Y))                             ║
# ║                                                                             ║
# ║   1. Consume el puente emitido por la Fase 1.                               ║
# ║   2. Aplica el funtor olvidadizo G sobre la narrativa Y.                    ║
# ║   3. Compara X contra G(Y).                                                 ║
# ║   4. Veta la deriva semántica si el residuo topológico excede tolerancia.   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_GaloisSemanticAuditor(Phase1_RiemannianLatticeCertifier):
    r"""
    Fase 2 del endofuntor.

    Responsabilidades:
    - Recibir el puente formal de Fase 1.
    - Auditar la reversibilidad del difeomorfismo semántico.
    - Garantizar que la narrativa no inyecte entropía retórica.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMER MÉTODO DE FASE 2
    # Inicio formal a partir del puente de Fase 1
    # ─────────────────────────────────────────────────────────────────────────
    def _begin_phase2_from_phase1_bridge(
        self,
        phase1_bridge: Phase1CertificationBridge,
        G_forgetful_functor: Callable[[VectorF64], VectorF64]
    ) -> Phase2GaloisAuditBridge:
        r"""
        Primer método de la Fase 2.

        Consume el `Phase1CertificationBridge` emitido por el último método
        de la Fase 1 y ejecuta la auditoría de Adjunción de Galois.
        """
        if not isinstance(phase1_bridge, Phase1CertificationBridge):
            raise SemanticTranslatorAgentError(
                "La Fase 2 requiere un Phase1CertificationBridge emitido por la Fase 1."
            )

        galois_audit = self._audit_galois_adjunction_reversibility(
            original_topological_coords=phase1_bridge.topological_coords,
            narrative_vector_y=phase1_bridge.narrative_vector_y,
            G_forgetful_functor=G_forgetful_functor
        )

        return Phase2GaloisAuditBridge(
            phase1_bridge=phase1_bridge,
            galois_adjunction=galois_audit
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Auditoría de reversibilidad de Galois
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_galois_adjunction_reversibility(
        self,
        original_topological_coords: VectorF64,
        narrative_vector_y: VectorF64,
        G_forgetful_functor: Callable[[VectorF64], VectorF64]
    ) -> GaloisAdjunctionAudit:
        r"""
        Aplica el funtor inverso/olvidadizo $G$ sobre la narrativa $Y$ y
        compara con la coordenada original $X$.

        Condición de isomorfismo:
            $\|X - G(Y)\|_2 \leq \varepsilon_{\text{abs}}
              + \varepsilon_{\text{rel}} \|X\|_2$.

        Raises
        ------
        SemanticDriftVetoError
            Si el residuo topológico excede la tolerancia admisible.
        """
        if not callable(G_forgetful_functor):
            raise SemanticDriftVetoError(
                "G_forgetful_functor debe ser un morfismo callable."
            )

        x = self._as_finite_vector(
            "original_topological_coords",
            original_topological_coords
        )
        y = self._as_finite_vector(
            "narrative_vector_y",
            narrative_vector_y
        )

        try:
            # Se entrega una copia para evitar mutaciones laterales del functor.
            recovered_coords = G_forgetful_functor(y.copy())
        except Exception as exc:
            raise SemanticDriftVetoError(
                "El funtor olvidadizo falló al extraer coordenadas desde la narrativa."
            ) from exc

        x_hat = self._as_finite_vector(
            "G_forgetful_functor(narrative_vector_y)",
            recovered_coords
        )

        if x_hat.shape != x.shape:
            raise SemanticDriftVetoError(
                "El funtor olvidadizo devolvió coordenadas de dimensión incompatible. "
                f"X={x.shape}, G(Y)={x_hat.shape}."
            )

        residual_vector = x - x_hat

        residual = self._safe_l2_norm(residual_vector)
        x_norm = self._safe_l2_norm(x)

        if not math.isfinite(residual):
            raise SemanticDriftVetoError(
                "Residuo topológico no finito entre X y G(Y)."
            )

        if not math.isfinite(x_norm):
            raise SemanticDriftVetoError(
                "Norma de las coordenadas topológicas originales no finita."
            )

        relative_residual = residual / max(1.0, x_norm)
        combined_tolerance = (
            _GALOIS_ABSOLUTE_TOLERANCE
            + _GALOIS_RELATIVE_TOLERANCE * x_norm
        )

        if residual > combined_tolerance:
            raise SemanticDriftVetoError(
                "Fallo en la Adjunción de Galois. El LLM inyectó entropía retórica. "
                f"Residuo topológico ({residual:.6e}) > tolerancia combinada "
                f"({combined_tolerance:.6e})."
            )

        return GaloisAdjunctionAudit(
            topological_residual=residual,
            topological_relative_residual=relative_residual,
            is_isomorphism_preserved=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: MODULACIÓN TERMODINÁMICA Y CRISTALIZACIÓN DIPLOMÁTICA             ║
# ║                                                                             ║
# ║   Φ₃(Φ₂(Φ₁(...))) = P(v) = exp(-E(v)/(k_B T_gov)) / Z                       ║
# ║                                                                             ║
# ║   1. Consume el puente de Fase 2.                                           ║
# ║   2. Calcula T_gov a partir de Dixmier y estabilidad Ψ.                     ║
# ║   3. Construye la distribución de Gibbs con estabilidad log-sum-exp.        ║
# ║   4. Ensambla el objeto final DiplomaticTranslationState.                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_ThermodynamicCrystallizer(Phase2_GaloisSemanticAuditor):
    r"""
    Fase 3 del endofuntor.

    Responsabilidades:
    - Congelar grados de libertad estocásticos del LLM.
    - Calcular temperatura de gobierno y distribución de Gibbs.
    - Ensamblar el estado diplomático final.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMER MÉTODO DE FASE 3
    # Inicio formal a partir del puente de Fase 2
    # ─────────────────────────────────────────────────────────────────────────
    def _begin_phase3_from_phase2_bridge(
        self,
        phase2_bridge: Phase2GaloisAuditBridge,
        dixmier_trace_inverse: float,
        stability_psi: float,
        energy_levels_for_gibbs: VectorF64
    ) -> ThermodynamicGovernanceState:
        r"""
        Primer método de la Fase 3.

        Consume el puente de Fase 2 y ejecuta la modulación termodinámica.
        """
        if not isinstance(phase2_bridge, Phase2GaloisAuditBridge):
            raise SemanticTranslatorAgentError(
                "La Fase 3 requiere un Phase2GaloisAuditBridge emitido por la Fase 2."
            )

        return self._modulate_governance_temperature(
            dixmier_trace_inverse=dixmier_trace_inverse,
            stability_psi=stability_psi,
            energy_levels=energy_levels_for_gibbs
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Modulación termodinámica de Gibbs
    # ─────────────────────────────────────────────────────────────────────────
    def _modulate_governance_temperature(
        self,
        dixmier_trace_inverse: float,
        stability_psi: float,
        energy_levels: VectorF64
    ) -> ThermodynamicGovernanceState:
        r"""
        Calcula:
            $T_{gov} = \max(\tau_D^{-1} \Psi, \epsilon)$

        y la distribución de Gibbs:
            $P(v) = \frac{\exp(-E(v)/(k_B T_{gov}))}{\sum_j \exp(-E(v_j)/(k_B T_{gov}))}$.

        Implementación numérica:
        - Desplazamiento energético por $E_{\min}$ para estabilidad.
        - Clip de exponentes para evitar overflow/underflow destructivo.
        - Normalización explícita.
        """
        trace_inv = self._as_finite_float(
            "dixmier_trace_inverse",
            dixmier_trace_inverse
        )
        psi = self._as_finite_float(
            "stability_psi",
            stability_psi
        )
        energies = self._as_finite_vector(
            "energy_levels_for_gibbs",
            energy_levels
        )

        if trace_inv < 0.0:
            raise SemanticTranslatorAgentError(
                "dixmier_trace_inverse debe ser no negativa."
            )

        if psi < 0.0:
            raise SemanticTranslatorAgentError(
                "stability_psi debe ser no negativa."
            )

        try:
            t_raw = float(trace_inv * psi)
        except OverflowError as exc:
            raise SemanticTranslatorAgentError(
                "El producto dixmier_trace_inverse * stability_psi desborda float64."
            ) from exc

        if not math.isfinite(t_raw):
            raise SemanticTranslatorAgentError(
                "La temperatura bruta de gobierno no es finita."
            )

        t_gov = max(t_raw, _THERMAL_TEMPERATURE_FLOOR)

        if t_raw < _THERMAL_TEMPERATURE_FLOOR:
            logger.warning(
                "Temperatura de gobierno cercana a cero. "
                f"t_raw={t_raw:.6e}; se impone piso térmico {_THERMAL_TEMPERATURE_FLOOR:.6e}."
            )

        beta = 1.0 / (_BOLTZMANN_K_CYBER * t_gov)

        if not math.isfinite(beta):
            raise SemanticTranslatorAgentError(
                "Beta termodinámica no finita."
            )

        e_min = float(np.min(energies))

        if not math.isfinite(e_min):
            raise SemanticTranslatorAgentError(
                "Energía mínima no finita en los niveles energéticos."
            )

        # Estabilización tipo log-sum-exp:
        # exp(-(E - E_min)/T) evita exponentes positivos masivos.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled_energy = (energies - e_min) * beta
            scaled_energy = np.clip(scaled_energy, 0.0, _MAX_EXP_ARGUMENT)
            boltzmann_weights = np.exp(-scaled_energy)

        if not np.all(np.isfinite(boltzmann_weights)):
            raise SemanticTranslatorAgentError(
                "Pesos de Boltzmann no finitos tras estabilización exponencial."
            )

        partition_function = float(np.sum(boltzmann_weights))

        if not math.isfinite(partition_function) or partition_function <= 0.0:
            raise SemanticTranslatorAgentError(
                "Función de partición de Gibbs no finita o no positiva."
            )

        gibbs_distribution = boltzmann_weights / partition_function

        # Renormalización final para mitigar deriva de redondeo.
        gibbs_sum = float(np.sum(gibbs_distribution))

        if not math.isfinite(gibbs_sum) or gibbs_sum <= 0.0:
            raise SemanticTranslatorAgentError(
                "Distribución de Gibbs no normalizable."
            )

        gibbs_distribution = gibbs_distribution / gibbs_sum

        return ThermodynamicGovernanceState(
            governance_temperature=t_gov,
            inverse_temperature_beta=beta,
            gibbs_distribution=np.asarray(gibbs_distribution, dtype=np.float64),
            partition_function=partition_function
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Ensamblaje del objeto diplomático final
    # ─────────────────────────────────────────────────────────────────────────
    def _assemble_diplomatic_translation(
        self,
        phase2_bridge: Phase2GaloisAuditBridge,
        thermodynamic_state: ThermodynamicGovernanceState
    ) -> DiplomaticTranslationState:
        r"""
        Ensambla el estado final del endofuntor:
            $\mathcal{Z}_{Translator} = \Phi_3 \circ \Phi_2 \circ \Phi_1$.
        """
        if not isinstance(phase2_bridge, Phase2GaloisAuditBridge):
            raise SemanticTranslatorAgentError(
                "El ensamblaje final requiere un Phase2GaloisAuditBridge."
            )

        if not phase2_bridge.phase1_bridge.riemannian_audit.is_geodesically_valid:
            raise SemanticTranslatorAgentError(
                "No se puede aprobar una narrativa con auditoría Riemanniana inválida."
            )

        if not phase2_bridge.galois_adjunction.is_isomorphism_preserved:
            raise SemanticDriftVetoError(
                "No se puede aprobar una narrativa con isomorfismo de Galois violado."
            )

        return DiplomaticTranslationState(
            riemannian_audit=phase2_bridge.phase1_bridge.riemannian_audit,
            lattice_collapse=phase2_bridge.phase1_bridge.lattice_collapse,
            galois_adjunction=phase2_bridge.galois_adjunction,
            thermodynamic_state=thermodynamic_state,
            approved_narrative_vector=phase2_bridge.phase1_bridge.narrative_vector_y
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: SEMANTIC TRANSLATOR AGENT                            ║
# ║                                                                             ║
# ║   Endofuntor:                                                               ║
# ║       Z_Translator = Φ₃ ∘ Φ₂ ∘ Φ₁                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class SemanticTranslatorAgent(Morphism, Phase3_ThermodynamicCrystallizer):
    r"""
    El Endofuntor de Difeomorfismo Semántico.

    Garantiza categóricamente que la IA opere únicamente como cristalizador
    de las geodésicas físicas, erradicando la retórica estocástica del ecosistema.
    """

    def execute_diplomatic_translation(
        self,
        G_metric: MatrixF64,
        topological_coords: VectorF64,
        narrative_vector_y: VectorF64,
        input_verdicts: List[VerdictLevel],
        G_forgetful_functor: Callable[[VectorF64], VectorF64],
        dixmier_trace_inverse: float,
        stability_psi: float,
        energy_levels_for_gibbs: VectorF64
    ) -> DiplomaticTranslationState:
        r"""
        Ejecuta la composición funtorial estricta en 3 fases anidadas.

        Flujo:
        ----
        1. Fase 1:
           `_complete_phase1_geometric_order_certification`
           → `Phase1CertificationBridge`

        2. Fase 2:
           `_begin_phase2_from_phase1_bridge`
           → `Phase2GaloisAuditBridge`

        3. Fase 3:
           `_begin_phase3_from_phase2_bridge`
           → `ThermodynamicGovernanceState`

        4. Ensamblaje final:
           `_assemble_diplomatic_translation`
           → `DiplomaticTranslationState`
        """
        # ── Fase 1: Certificación métrica y colapso reticular ────────────────
        phase1_bridge = self._complete_phase1_geometric_order_certification(
            G_metric=G_metric,
            topological_coords=topological_coords,
            narrative_vector_y=narrative_vector_y,
            input_verdicts=input_verdicts
        )

        # ── Fase 2: Adjunción de Galois y reversibilidad semántica ───────────
        phase2_bridge = self._begin_phase2_from_phase1_bridge(
            phase1_bridge=phase1_bridge,
            G_forgetful_functor=G_forgetful_functor
        )

        # ── Fase 3: Modulación termodinámica y congelamiento estocástico ─────
        thermodynamic_state = self._begin_phase3_from_phase2_bridge(
            phase2_bridge=phase2_bridge,
            dixmier_trace_inverse=dixmier_trace_inverse,
            stability_psi=stability_psi,
            energy_levels_for_gibbs=energy_levels_for_gibbs
        )

        # ── Ensamblaje del objeto final ──────────────────────────────────────
        final_state = self._assemble_diplomatic_translation(
            phase2_bridge=phase2_bridge,
            thermodynamic_state=thermodynamic_state
        )

        logger.info(
            "Difeomorfismo Semántico ejecutado con éxito. "
            f"Supremo: {final_state.lattice_collapse.supremum_verdict.name} | "
            f"d_M: {final_state.riemannian_audit.mahalanobis_distance:.6f} | "
            f"T_gov: {final_state.thermodynamic_state.governance_temperature:.6f} | "
            f"Residuo Galois: {final_state.galois_adjunction.topological_residual:.6e}"
        )

        return final_state


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "SemanticTranslatorAgentError",
    "RiemannianMetricDegeneracyError",
    "LatticeCollapseViolation",
    "SemanticDriftVetoError",
    "VerdictLevel",
    "RiemannianRetrievalData",
    "LatticeCollapseState",
    "GaloisAdjunctionAudit",
    "ThermodynamicGovernanceState",
    "Phase1CertificationBridge",
    "Phase2GaloisAuditBridge",
    "DiplomaticTranslationState",
    "Phase1_RiemannianLatticeCertifier",
    "Phase2_GaloisSemanticAuditor",
    "Phase3_ThermodynamicCrystallizer",
    "SemanticTranslatorAgent",
]
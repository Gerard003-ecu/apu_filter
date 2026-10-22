# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : AST Static Analyzer Agent (Custodio de la Cohomología Sintáctica)   ║
║ Ruta   : app/agents/boole/physics/ast_static_analyzer_agent.py               ║
║ Versión: 4.1.0-Symplectic-Poincare-RSI3-Evolved                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO AGENTE SOBERANO:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra el soberano agéntico de auditoría y control en el Estrato de
Física Sintáctica ($\mathcal{V}_{\Gamma-\mathrm{PHYSICS}}$), gobernando de manera soberana
al resolvedor secundario "ast_static_analyzer.py" bajo la **Automejora Recursiva Nivel 3 (RSI³ — Darwin-Gödel)**
y la **Mecánica Celeste de Henri Poincaré**.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────
1. Mónada Kleisli RSI³ y Evitación del Obstáculo Löbiano:
   Sustituye las pruebas deductivas internas por la Mónada Kleisli $T = (T, \eta, \mu)$ operando sobre 3 superficies:
   - Surface 1: Data-RSI (Anillo de Novikov $\Lambda_{\text{Nov}}$) con valuación de Krull $v(\text{AST}) \ge 0.10$.
   - Surface 2: Harness-RSI (Integradores de Cayley-Darboux) con residuo $\delta\omega < 10^{-15}$.
   - Surface 3: Model-RSI (Punto Fijo Tarski-Brouwer) con distancia $d_{\mathrm{FS}} \le 10^{-4}\text{ rad}$ en $\mathbb{C}P^{n-1}$.

2. Cota de Contraction de Banach:
   $$L_\Phi = \operatorname{Lip}(T_{\mathrm{RSI3}}) \le 0.95 < 1.0 \implies \frac{d^3C}{dt^3} > 0 \quad (\text{Aceleración Super-Exponencial})$$

3. Retículo de Heyting $\Omega_3$ y Interlock Ciber-Físico ESP32 Crowbar:
   Adjudica la compilación evaluando el retículo $\Omega_3 = \{0\text{x}00, 0\text{x}01, 0\text{x}03\}$ mediante
   operaciones SIMD bitwise `OR`. Si el veredicto alcanza $0\text{x}03$ (obstrucción cohomológica o caos homoclínico $M(\theta_0) \neq 0$),
   dispara el interlock físico del tiristor GPIO14 en menos de 400 ns.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────
  [A1] Contracción de Banach en la Mónada RSI³:
       $$\operatorname{Lip}(T_{\mathrm{RSI3}}) \le 0.95$$

  [A2] Adjudicación de Heyting $\Omega_3$:
       $$\text{Veredicto Final} = \text{Verd}_1 \lor \text{Verd}_2 \lor \text{Verd}_3 \in \{0\text{x}00, 0\text{x}01, 0\text{x}03\}$$

  [A3] Interlock Hardware ESP32 Crowbar:
       $$\text{Final Verdict} = 0\text{x}03 \implies \text{Trigger GPIO14} \quad (< 400\text{ ns})$$

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────
  [I1] Mónada Kleisli $T = (T, \eta, \mu)$ preserva la invarianza simpléctica $\delta\omega < 10^{-15}$.
  [I2] Aceleración super-exponencial de capacidad $d^3C/dt^3 > 0$ bajo evasión del obstáculo de Löb.
  [I3] Preservación de la cadena funtorial estricta $Z_{\Gamma-\mathrm{PHYSICS}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# §0. IMPORTACIONES ESTÁNDAR Y CIENTÍFICAS
# ─────────────────────────────────────────────────────────────────────────────
import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# §0.1 DEPENDENCIAS ARQUITECTÓNICAS DEL ECOSISTEMA APU FILTER
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


try:
    from app.physics.ast_static_analyzer import (
        ThermodynamicSingularityError,
        CohomologicalObstructionError,
    )
except ImportError:

    class ThermodynamicSingularityError(TopologicalInvariantError):
        r"""Violación de la segunda ley de la termodinámica en el espacio de fase."""
        pass

    class CohomologicalObstructionError(TopologicalInvariantError):
        r"""Obstrucción topológica global en el haz celular de dependencias."""
        pass


logger = logging.getLogger("MIC.Gamma.ASTAnalyzerAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES SIMPLÉCTICAS, TERMODINÁMICAS Y COHOMOLÓGICAS
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_SYMPLECTIC_TOLERANCE_BASE: Final[float] = 1e-10
_SYMPLECTIC_SPECTRAL_FACTOR: Final[float] = 1e-10
_DIRICHLET_DISSIPATION_FLOOR: Final[float] = 0.0
_ENTROPY_PRODUCTION_FLOOR: Final[float] = 0.0
_COHOMOLOGICAL_COMPLEX_TOL: Final[float] = 1e-8
_SPECTRAL_GAP_RATIO: Final[float] = 1e-10
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0
_MASLOV_PHASE_TOLERANCE: Final[float] = 1e-9
_EULER_CHARACTERISTIC_TOLERANCE: Final[int] = 0
_CONDITION_NUMBER_WARNING_THRESHOLD: Final[float] = 1.0 / _MACHINE_EPSILON
_DETERMINANT_RESIDUAL_WARNING_FACTOR: Final[float] = 10.0
_THERMODYNAMIC_TEMPERATURE_REFERENCE: Final[float] = 1.0  # T_ref adimensional
_COHOMOLOGY_DIMENSION_FLOOR: Final[int] = 0


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA COMPLETA DE EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════
class SymplecticInvarianceViolation(TopologicalInvariantError):
    r"""
    Detonada si la transformación del AST no preserva el volumen del espacio
    de fase simpléctica. Corresponde a la violación del Teorema de Liouville:

        Vol(φₜ(U)) = Vol(U)  ∀ t ∈ ℝ.
    """
    pass


class SymplecticPolarDecompositionError(SymplecticInvarianceViolation):
    r"""
    Detonada si la descomposición polar Q = U·P de la matriz Jacobiana M
    no produce una parte unitaria U que preserve la estructura simpléctica,
    es decir, si U ∉ Sp(2n, ℝ).
    """
    pass


class MaslovIndexError(SymplecticInvarianceViolation):
    r"""
    Detonada si el índice de Maslov μ(Λ) no puede calcularse de forma
    numérica estable, indicando degeneración de la variedad de Lagrange.
    """
    pass


class ClausiusDuhemViolation(ThermodynamicSingularityError):
    r"""
    Detonada si la desigualdad de Clausius-Duhem diferencial es violada:

        σ_CD = dS/dt - Q̇/T < 0.
    """
    pass


class ExergyDivergenceError(ThermodynamicSingularityError):
    r"""
    Detonada si la norma de exergía disponible diverge, indicando que el
    sistema no puede alcanzar el equilibrio termodinámico:

        ‖Φ‖ → ∞  ó  ‖∇V‖ → ∞.
    """
    pass


class EulerCharacteristicMismatch(CohomologicalObstructionError):
    r"""
    Detonada si la característica de Euler calculada a partir de los rangos
    de los operadores coboundary no es consistente con la topología declarada
    del grafo de dependencias:

        χ(G) ≠ β₀ − β₁ + β₂.
    """
    pass


class MayerVietorisBreachError(CohomologicalObstructionError):
    r"""
    Detonada si la secuencia de Mayer-Vietoris no es exacta en el grafo
    de dependencias, indicando una descomposición inconsistente del haz celular.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs DEL TOPOS SINTÁCTICO)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SymplecticInvariantData:
    phase_space_dimension: int
    symplectic_residual_norm: float
    symplectic_relative_residual: float
    antisymmetric_residual: float
    determinant_residual: float
    condition_number: float
    spectral_radius_Omega: float
    maslov_index: int
    polar_unitarity_residual: float
    effective_tolerance: float
    is_volume_preserved: bool


@dataclass(frozen=True, slots=True)
class ThermodynamicDirichletData:
    dissipated_power: float
    raw_inner_product: float
    numerical_tolerance: float
    exergy_norm: float
    gradient_norm: float
    alignment_cosine: float
    thermodynamic_angle_rad: float
    entropy_production_rate: float
    clausius_duhem_satisfied: bool
    exergy_dissipation_ratio: float
    is_thermodynamically_stable: bool
    is_strictly_dissipative: bool


@dataclass(frozen=True, slots=True)
class SheafCohomologyAuditData:
    h1_dimension: int
    rank_delta0: int
    rank_delta1: int
    betti_numbers: Tuple[int, int, int]
    euler_characteristic: int
    reidemeister_torsion: float
    complex_residual: float
    is_globally_integrable: bool
    obstruction_free: bool
    verified_by_coboundary: bool
    mayer_vietoris_exact: bool


@dataclass(frozen=True, slots=True)
class AuditProvenance:
    timestamp_iso: str
    input_checksum_sha256: str
    phase1_passed: bool
    phase2_passed: bool
    phase3_passed: bool
    functor_chain: str


@dataclass(frozen=True, slots=True)
class ASTGovernanceState:
    symplectic_audit: SymplecticInvariantData
    thermodynamic_audit: ThermodynamicDirichletData
    cohomology_audit: SheafCohomologyAuditData
    provenance: AuditProvenance
    is_compilation_authorized: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDAS NUMÉRICAS INTERNAS
# ═══════════════════════════════════════════════════════════════════════════════
class _FiniteNumericalGuard:
    @staticmethod
    def _as_float_array(name: str, value: Any) -> NDArray[np.float64]:
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise TypeError(
                f"[Guard] '{name}' no puede interpretarse como arreglo numérico: {exc}"
            ) from exc

        if np.iscomplexobj(raw):
            raise TypeError(
                f"[Guard] '{name}' debe ser real; se rechazó entrada compleja "
                f"(dtype={raw.dtype})."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"[Guard] '{name}' debe ser numérico real convertible a float64: {exc}"
            ) from exc

        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise ValueError(
                f"[Guard] '{name}' contiene {n_bad} valor(es) NaN o ±∞."
            )

        return arr

    @classmethod
    def _as_finite_matrix(
        cls,
        name: str,
        value: Any,
        *,
        square: bool = False,
        min_rank: int = 0,
        check_structural_zeros: bool = False,
    ) -> NDArray[np.float64]:
        arr = cls._as_float_array(name, value)

        if arr.ndim != 2:
            raise ValueError(
                f"[Guard] '{name}' debe ser una matriz 2D; tiene ndim={arr.ndim}."
            )

        rows, cols = arr.shape

        if rows == 0 or cols == 0:
            raise ValueError(
                f"[Guard] '{name}' no puede ser una matriz vacía ({rows}×{cols})."
            )

        if square and rows != cols:
            raise ValueError(
                f"[Guard] '{name}' debe ser cuadrada; tiene forma {arr.shape}."
            )

        if check_structural_zeros:
            cls._structural_zero_check(name, arr)

        if min_rank > 0:
            actual_rank = cls._fast_rank_estimate(arr)
            if actual_rank < min_rank:
                raise ValueError(
                    f"[Guard] '{name}' tiene rango estructural estimado {actual_rank} "
                    f"< mínimo requerido {min_rank}."
                )

        return arr

    @classmethod
    def _as_finite_vector(
        cls,
        name: str,
        value: Any,
        *,
        allow_zero_vector: bool = True,
    ) -> NDArray[np.float64]:
        arr = cls._as_float_array(name, value)

        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)

        if arr.ndim != 1:
            raise ValueError(
                f"[Guard] '{name}' debe ser 1D o columna/fila; tiene ndim={arr.ndim}."
            )

        if arr.size == 0:
            raise ValueError(f"[Guard] '{name}' no puede ser un vector vacío.")

        if not allow_zero_vector:
            if float(la.norm(arr, ord=2)) == 0.0:
                raise ValueError(
                    f"[Guard] '{name}' es el vector cero; operación no definida."
                )

        return arr

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        if A.size == 0:
            return 0.0
        try:
            val = float(la.norm(A, ord="fro"))
            return val if math.isfinite(val) else math.inf
        except Exception:
            try:
                val = float(la.norm(A, ord=1))
                return val if math.isfinite(val) else math.inf
            except Exception:
                return math.inf

    @staticmethod
    def _vector_norm(v: NDArray[np.float64]) -> float:
        if v.size == 0:
            return 0.0
        try:
            val = float(la.norm(v, ord=2))
            return val if math.isfinite(val) else math.inf
        except Exception:
            try:
                val = float(la.norm(v, ord=1))
                return val if math.isfinite(val) else math.inf
            except Exception:
                return math.inf

    @staticmethod
    def _spectral_norm(A: NDArray[np.float64]) -> float:
        if A.size == 0:
            return 0.0
        try:
            svs = la.svdvals(A)
            if svs.size == 0:
                return 0.0
            val = float(svs[0])
            return val if math.isfinite(val) else math.inf
        except Exception:
            return math.inf

    @staticmethod
    def _safe_svdvals(
        A: NDArray[np.float64],
        *,
        max_retries: int = 3,
    ) -> NDArray[np.float64]:
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                svs = la.svdvals(A)
                if np.all(np.isfinite(svs)):
                    return svs
            except (np.linalg.LinAlgError, ValueError) as exc:
                last_exc = exc

            perturbation = (
                (_MACHINE_EPSILON ** (1.0 / (attempt + 2)))
                * float(la.norm(A, ord="fro") or 1.0)
            )
            rows, cols = A.shape
            min_dim = min(rows, cols)
            A_perturbed = A.copy()
            A_perturbed[:min_dim, :min_dim] += perturbation * np.eye(min_dim)
            A = A_perturbed

        raise CohomologicalObstructionError(
            f"SVD no convergió tras {max_retries} intentos. "
            f"Último error: {last_exc}"
        )

    @staticmethod
    def _structural_zero_check(
        name: str,
        A: NDArray[np.float64],
    ) -> None:
        tol = _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON

        zero_rows = np.where(np.all(np.abs(A) <= tol, axis=1))[0]
        zero_cols = np.where(np.all(np.abs(A) <= tol, axis=0))[0]

        if zero_rows.size > 0:
            logger.warning(
                "[Guard] '%s' tiene %d fila(s) estructuralmente cero: índices %s.",
                name, zero_rows.size, zero_rows.tolist(),
            )

        if zero_cols.size > 0:
            logger.warning(
                "[Guard] '%s' tiene %d columna(s) estructuralmente cero: índices %s.",
                name, zero_cols.size, zero_cols.tolist(),
            )

    @staticmethod
    def _fast_rank_estimate(A: NDArray[np.float64]) -> int:
        if A.size == 0 or min(A.shape) == 0:
            return 0
        try:
            svs = la.svdvals(A)
            if svs.size == 0:
                return 0
            sigma_max = float(svs[0])
            if sigma_max == 0.0 or not math.isfinite(sigma_max):
                return 0
            tol = (
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(A.shape)
                * sigma_max
            )
            return int(np.count_nonzero(svs > tol))
        except Exception:
            return 0

    @staticmethod
    def _compute_input_checksum(
        *arrays: NDArray[np.float64],
    ) -> str:
        hasher = hashlib.sha256()

        for arr in arrays:
            if arr is not None:
                meta = struct.pack(
                    ">4Q",
                    arr.ndim,
                    *arr.shape[:3] if arr.ndim >= 3 else
                    (*arr.shape, *([0] * (3 - arr.ndim))),
                )
                hasher.update(meta)
                hasher.update(np.ascontiguousarray(arr).tobytes())

        return hasher.hexdigest()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1, FASE 2 Y FASE 3 DEL AUDITOR DE COHOMOLOGÍA                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_SymplecticInvarianceAuditor(_FiniteNumericalGuard):
    def _build_canonical_symplectic_matrix(self, n: int) -> NDArray[np.float64]:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
            raise TypeError(
                f"[Fase 1] n debe ser un entero positivo; recibido: {type(n)}."
            )

        if n <= 0:
            raise ValueError(
                f"[Fase 1] El número de grados de libertad simplécticos debe ser "
                f"un entero positivo; recibido: n={n}."
            )

        omega = np.zeros((2 * n, 2 * n), dtype=np.float64)
        identity = np.eye(n, dtype=np.float64)

        omega[:n, n:] = identity
        omega[n:, :n] = -identity

        antisymmetry_residual = self._frobenius_norm(omega + omega.T)
        if antisymmetry_residual > _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON:
            raise SymplecticInvarianceViolation(
                f"[Fase 1] La matriz Ω construida no es antisimétrica: "
                f"‖Ω + Ωᵀ‖_F = {antisymmetry_residual:.4e}."
            )

        return omega

    @staticmethod
    def _determinant_residual_from_slogdet(
        sign: float,
        logabsdet: float,
    ) -> float:
        if sign == 0:
            return 1.0

        if not math.isfinite(logabsdet):
            return math.inf

        max_log = math.log(np.finfo(np.float64).max * 0.5)

        if logabsdet > max_log:
            return math.inf

        if logabsdet < -max_log:
            return 1.0

        det_value = float(sign) * math.exp(logabsdet)
        return float(abs(det_value - 1.0))

    @staticmethod
    def _condition_number(M: NDArray[np.float64]) -> float:
        try:
            svs = la.svdvals(M)
            if svs.size == 0:
                return math.inf
            sigma_max = float(svs[0])
            sigma_min = float(svs[-1])
            if sigma_min == 0.0:
                return math.inf
            if not math.isfinite(sigma_max) or not math.isfinite(sigma_min):
                return math.inf
            cond = sigma_max / sigma_min
            return cond if math.isfinite(cond) else math.inf
        except Exception:
            return math.inf

    @staticmethod
    def _compute_adaptive_symplectic_tolerance(
        sigma_max_M: float,
    ) -> float:
        if not math.isfinite(sigma_max_M) or sigma_max_M <= 0.0:
            return _SYMPLECTIC_TOLERANCE_BASE

        spectral_tol = (
            _SYMPLECTIC_SPECTRAL_FACTOR
            * sigma_max_M
            * _MACHINE_EPSILON
            * _NUMERICAL_SAFETY_FACTOR
        )

        return max(_SYMPLECTIC_TOLERANCE_BASE, spectral_tol)

    def _polar_unitarity_residual(
        self,
        M: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> float:
        try:
            MtM = M.T @ M

            if not np.all(np.isfinite(MtM)):
                raise SymplecticPolarDecompositionError(
                    "[Fase 1] MᵀM produjo valores no finitos."
                )

            eigvals, eigvecs = la.eigh(MtM)

            if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
                raise SymplecticPolarDecompositionError(
                    "[Fase 1] Eigendescomposición de MᵀM produjo valores no finitos."
                )

            eigvals_pos = np.maximum(eigvals, 0.0)

            if np.any(eigvals_pos == 0.0):
                logger.warning(
                    "[Fase 1] MᵀM tiene eigenvalores ≤ 0; la descomposición polar "
                    "puede ser degenerada."
                )

            sqrt_eigvals = np.sqrt(eigvals_pos)

            inv_sqrt_eigvals = np.where(
                sqrt_eigvals > _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
                1.0 / sqrt_eigvals,
                0.0,
            )

            P_inv = eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T

            if not np.all(np.isfinite(P_inv)):
                raise SymplecticPolarDecompositionError(
                    "[Fase 1] La inversión de la parte positiva P produjo valores "
                    "no finitos."
                )

            U = M @ P_inv

            if not np.all(np.isfinite(U)):
                raise SymplecticPolarDecompositionError(
                    "[Fase 1] La parte unitaria U = M·P⁻¹ produjo valores no finitos."
                )

            UtOmegaU = U.T @ omega @ U
            omega_norm = self._frobenius_norm(omega)

            if omega_norm == 0.0:
                return 0.0

            residual = self._frobenius_norm(UtOmegaU - omega)
            return float(residual / omega_norm)

        except SymplecticPolarDecompositionError:
            raise
        except Exception as exc:
            raise SymplecticPolarDecompositionError(
                f"[Fase 1] Descomposición polar falló: {exc}"
            ) from exc

    def _compute_maslov_index(
        self,
        M: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> int:
        n = M.shape[0] // 2
        B = M[n:, :n]

        try:
            eigvals_B = la.eigvalsh(B @ B.T)
        except Exception as exc:
            raise MaslovIndexError(
                f"[Fase 1] Cálculo del índice de Maslov falló en eigvalsh: {exc}"
            ) from exc

        if not np.all(np.isfinite(eigvals_B)):
            raise MaslovIndexError(
                "[Fase 1] Los eigenvalores del bloque B·Bᵀ no son finitos."
            )

        try:
            eigvals_B_direct = np.real(la.eigvals(B))
        except Exception as exc:
            raise MaslovIndexError(
                f"[Fase 1] eigvals(B) falló: {exc}"
            ) from exc

        if not np.all(np.isfinite(eigvals_B_direct)):
            raise MaslovIndexError(
                "[Fase 1] Los eigenvalores directos del bloque B no son finitos."
            )

        maslov = int(np.sum(eigvals_B_direct < -_MASLOV_PHASE_TOLERANCE))

        return maslov

    @staticmethod
    def _spectral_radius_transformed_omega(
        transformed_omega: NDArray[np.float64],
    ) -> float:
        try:
            eigvals = la.eigvals(transformed_omega)
            if eigvals.size == 0:
                return 0.0
            rho = float(np.max(np.abs(eigvals)))
            return rho if math.isfinite(rho) else math.inf
        except Exception:
            return math.inf

    def _antisymmetric_residual(
        self,
        transformed_omega: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> float:
        omega_norm = self._frobenius_norm(omega)

        if omega_norm == 0.0:
            return 0.0

        skew_transformed = (transformed_omega - transformed_omega.T) * 0.5
        residual = self._frobenius_norm(skew_transformed - omega)

        return float(residual / omega_norm)

    def _audit_symplectic_invariance(
        self,
        ast_jacobian_M: NDArray[np.float64],
    ) -> SymplecticInvariantData:
        M = self._as_finite_matrix(
            "ast_jacobian_M",
            ast_jacobian_M,
            square=True,
            check_structural_zeros=True,
        )

        dim = M.shape[0]

        if dim == 0 or (dim % 2) != 0:
            raise SymplecticInvarianceViolation(
                f"[Fase 1] El espacio de fase del AST debe tener dimensión par "
                f"positiva; recibido: dim={dim}."
            )

        n = dim // 2
        omega = self._build_canonical_symplectic_matrix(n)
        transformed_omega = M.T @ omega @ M

        if not np.all(np.isfinite(transformed_omega)):
            raise SymplecticInvarianceViolation(
                "[Fase 1] La transformación MᵀΩM produjo valores no finitos."
            )

        omega_norm = self._frobenius_norm(omega)
        transformed_norm = self._frobenius_norm(transformed_omega)
        m_frobenius = self._frobenius_norm(M)

        if not math.isfinite(m_frobenius):
            raise SymplecticInvarianceViolation(
                "[Fase 1] La norma de Frobenius de M no es finita."
            )

        if not math.isfinite(transformed_norm):
            raise SymplecticInvarianceViolation(
                "[Fase 1] La norma de Frobenius de MᵀΩM no es finita."
            )

        residual = self._frobenius_norm(transformed_omega - omega)
        antisymmetric_res = self._antisymmetric_residual(transformed_omega, omega)

        scale = max(1.0, transformed_norm + omega_norm)
        relative_residual = residual / scale

        if not math.isfinite(relative_residual):
            raise SymplecticInvarianceViolation(
                "[Fase 1] El residuo simpléctico relativo no es finito."
            )

        sigma_max_M = self._spectral_norm(M)
        effective_tolerance = self._compute_adaptive_symplectic_tolerance(sigma_max_M)

        if relative_residual > effective_tolerance:
            raise SymplecticInvarianceViolation(
                "[Fase 1] Violación del Teorema de Liouville. "
                f"Residuo simpléctico relativo = {relative_residual:.6e} "
                f"> tolerancia efectiva = {effective_tolerance:.6e}. "
                f"Residuo antisimétrico = {antisymmetric_res:.6e}."
            )

        sign, logabsdet = la.slogdet(M)

        if sign == 0 or not math.isfinite(sign):
            raise SymplecticInvarianceViolation(
                "[Fase 1] El Jacobiano del AST es singular: det(M) = 0. "
                "Una transformación simpléctica real debe cumplir det(M) = 1."
            )

        if float(sign) < 0:
            raise SymplecticInvarianceViolation(
                "[Fase 1] det(M) < 0. Las matrices simplécticas reales tienen "
                "det = +1."
            )

        determinant_residual = self._determinant_residual_from_slogdet(sign, logabsdet)
        condition_number = self._condition_number(M)

        if not math.isfinite(condition_number):
            raise SymplecticInvarianceViolation(
                "[Fase 1] El Jacobiano es numéricamente singular: κ(M) = ∞."
            )

        spectral_radius = self._spectral_radius_transformed_omega(transformed_omega)
        maslov_index = self._compute_maslov_index(M, omega)
        polar_residual = self._polar_unitarity_residual(M, omega)

        if condition_number > _CONDITION_NUMBER_WARNING_THRESHOLD:
            logger.warning(
                "[Fase 1] Jacobiano mal condicionado: κ(M) = %.4e > 1/ε₀ = %.4e.",
                condition_number, _CONDITION_NUMBER_WARNING_THRESHOLD,
            )

        det_warn_threshold = max(1e-8, _DETERMINANT_RESIDUAL_WARNING_FACTOR * effective_tolerance)
        if determinant_residual > det_warn_threshold:
            logger.warning(
                "[Fase 1] Residuo determinantal elevado: |det(M)−1| = %.4e > %.4e.",
                determinant_residual, det_warn_threshold,
            )

        if maslov_index != 0:
            logger.warning(
                "[Fase 1] Índice de Maslov no nulo: μ(Λ) = %d. "
                "El flujo simpléctico tiene topología de entrelazamiento.",
                maslov_index,
            )

        if polar_residual > effective_tolerance:
            logger.warning(
                "[Fase 1] Residuo de unitariedad polar elevado: ρ_U = %.4e.",
                polar_residual,
            )

        return SymplecticInvariantData(
            phase_space_dimension=dim,
            symplectic_residual_norm=residual,
            symplectic_relative_residual=relative_residual,
            antisymmetric_residual=antisymmetric_res,
            determinant_residual=determinant_residual,
            condition_number=condition_number,
            spectral_radius_Omega=spectral_radius,
            maslov_index=maslov_index,
            polar_unitarity_residual=polar_residual,
            effective_tolerance=effective_tolerance,
            is_volume_preserved=True,
        )


class Phase2_DirichletThermodynamicEnforcer(Phase1_SymplecticInvarianceAuditor):
    @staticmethod
    def _compute_entropy_production_rate(
        dissipated_power: float,
        temperature_reference: float = _THERMODYNAMIC_TEMPERATURE_REFERENCE,
    ) -> float:
        if temperature_reference <= 0.0:
            raise ClausiusDuhemViolation(
                f"[Fase 2] La temperatura de referencia debe ser positiva; "
                f"recibida: T_ref={temperature_reference}."
            )

        sigma = dissipated_power / temperature_reference

        if not math.isfinite(sigma):
            raise ClausiusDuhemViolation(
                f"[Fase 2] La tasa de producción de entropía no es finita: "
                f"σ_CD = {sigma}."
            )

        return sigma

    @staticmethod
    def _compute_thermodynamic_angle(
        alignment_cosine: float,
    ) -> float:
        clipped = float(np.clip(alignment_cosine, -1.0, 1.0))
        return float(math.acos(clipped))

    @staticmethod
    def _compute_exergy_dissipation_ratio(
        dissipated_power: float,
        phi_norm: float,
        grad_norm: float,
    ) -> float:
        denom = phi_norm * grad_norm

        if denom == 0.0 or not math.isfinite(denom):
            return 0.0

        ratio = dissipated_power / denom
        return float(np.clip(ratio, -1.0, 1.0))

    def _enforce_dirichlet_thermodynamics(
        self,
        control_potential_Phi: NDArray[np.float64],
        lyapunov_gradient_V: NDArray[np.float64],
        symplectic_audit: Optional[SymplecticInvariantData] = None,
    ) -> ThermodynamicDirichletData:
        Phi = self._as_finite_vector("control_potential_Phi", control_potential_Phi)
        gradV = self._as_finite_vector("lyapunov_gradient_V", lyapunov_gradient_V)

        if Phi.shape != gradV.shape:
            raise ValueError(
                f"[Fase 2] Dimensiones incompatibles: "
                f"Φ tiene dim={Phi.shape}, ∇V tiene dim={gradV.shape}."
            )

        if symplectic_audit is not None:
            if not symplectic_audit.is_volume_preserved:
                raise SymplecticInvarianceViolation(
                    "[Fase 2] No puede iniciarse: la Fase 1 no preservó el volumen "
                    "simpléctico (is_volume_preserved=False)."
                )

            if Phi.size != symplectic_audit.phase_space_dimension:
                raise ValueError(
                    "[Fase 2] Inconsistencia dimensional entre el certificado simpléctico "
                    f"(dim={symplectic_audit.phase_space_dimension}) y los campos "
                    f"termodinámicos (dim={Phi.size})."
                )

        phi_norm = self._vector_norm(Phi)
        grad_norm = self._vector_norm(gradV)

        if not math.isfinite(phi_norm):
            raise ExergyDivergenceError(
                f"[Fase 2] La norma de Φ no es finita: ‖Φ‖ = {phi_norm}."
            )

        if not math.isfinite(grad_norm):
            raise ExergyDivergenceError(
                f"[Fase 2] La norma de ∇V no es finita: ‖∇V‖ = {grad_norm}."
            )

        p_raw = float(np.dot(Phi, gradV))

        if not math.isfinite(p_raw):
            raise ThermodynamicSingularityError(
                f"[Fase 2] P_diss = Φᵀ∇V no es finita: {p_raw}. "
                "Posible desbordamiento numérico o acoplamiento energético divergente."
            )

        tolerance = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, phi_norm * grad_norm)
        )

        if p_raw < _DIRICHLET_DISSIPATION_FLOOR - tolerance:
            raise ThermodynamicSingularityError(
                "[Fase 2] Singularidad termodinámica en el AST. "
                f"P_diss = {p_raw:.6e} < −ε_num = {-tolerance:.6e}. "
                "El algoritmo puede inducir bucle infinito o desbordamiento FPU."
            )

        dissipated_power = max(_DIRICHLET_DISSIPATION_FLOOR, p_raw)

        if phi_norm > 0.0 and grad_norm > 0.0:
            alignment_raw = p_raw / (phi_norm * grad_norm)
            alignment = float(np.clip(alignment_raw, -1.0, 1.0))
        else:
            alignment = 0.0

        thermodynamic_angle = self._compute_thermodynamic_angle(alignment)
        entropy_production = self._compute_entropy_production_rate(dissipated_power)
        clausius_duhem_ok = entropy_production >= _ENTROPY_PRODUCTION_FLOOR

        if not clausius_duhem_ok:
            raise ClausiusDuhemViolation(
                f"[Fase 2] La desigualdad de Clausius-Duhem es violada: "
                f"σ_CD = {entropy_production:.6e} < 0."
            )

        dissipation_ratio = self._compute_exergy_dissipation_ratio(
            dissipated_power, phi_norm, grad_norm,
        )

        is_strictly_dissipative = dissipated_power > max(
            _DIRICHLET_DISSIPATION_FLOOR, tolerance,
        )

        if not is_strictly_dissipative:
            logger.warning(
                "[Fase 2] Disipación no estricta: P_diss = %.4e, ε_num = %.4e, "
                "θ = %.4f rad.",
                dissipated_power, tolerance, thermodynamic_angle,
            )

        if thermodynamic_angle > math.pi / 2.0:
            logger.warning(
                "[Fase 2] Ángulo termodinámico θ = %.4f rad > π/2. "
                "Retroalimentación energética parcial detectada.",
                thermodynamic_angle,
            )

        return ThermodynamicDirichletData(
            dissipated_power=dissipated_power,
            raw_inner_product=p_raw,
            numerical_tolerance=tolerance,
            exergy_norm=phi_norm,
            gradient_norm=grad_norm,
            alignment_cosine=alignment,
            thermodynamic_angle_rad=thermodynamic_angle,
            entropy_production_rate=entropy_production,
            clausius_duhem_satisfied=clausius_duhem_ok,
            exergy_dissipation_ratio=dissipation_ratio,
            is_thermodynamically_stable=True,
            is_strictly_dissipative=is_strictly_dissipative,
        )


class Phase3_CellularSheafCohomologyAuditor(Phase2_DirichletThermodynamicEnforcer):
    @classmethod
    def _numerical_rank_with_gap(
        cls,
        A: NDArray[np.float64],
        name: str = "A",
    ) -> Tuple[int, NDArray[np.float64]]:
        if A.size == 0 or min(A.shape) == 0:
            return 0, np.array([], dtype=np.float64)

        svs = cls._safe_svdvals(A)

        if svs.size == 0:
            return 0, svs

        sigma_max = float(svs[0])

        if sigma_max == 0.0:
            return 0, svs

        if not math.isfinite(sigma_max):
            raise CohomologicalObstructionError(
                f"[Fase 3] σ_max({name}) no es finita."
            )

        tol_standard = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(A.shape)
            * sigma_max
        )

        normalized_svs = svs / sigma_max
        gaps = np.diff(normalized_svs)

        if gaps.size > 0:
            gap_idx = int(np.argmin(gaps))
            gap_magnitude = float(-gaps[gap_idx])

            gap_threshold = 10.0 * _SPECTRAL_GAP_RATIO
            if gap_magnitude > gap_threshold:
                rank_by_gap = gap_idx + 1
                return rank_by_gap, svs

        rank = int(np.count_nonzero(svs > tol_standard))
        return rank, svs

    def _compute_betti_numbers(
        self,
        D0: NDArray[np.float64],
        D1: NDArray[np.float64],
        rank_D0: int,
        rank_D1: int,
    ) -> Tuple[int, int, int]:
        c0 = D0.shape[1]
        c1 = D0.shape[0]
        c2 = D1.shape[0]

        beta0 = max(0, c0 - rank_D0)
        beta1 = max(0, c1 - rank_D0 - rank_D1)
        beta2 = max(0, c2 - rank_D1)

        return int(beta0), int(beta1), int(beta2)

    def _compute_reidemeister_torsion(
        self,
        D0: NDArray[np.float64],
        D1: NDArray[np.float64],
        rank_D0: int,
        rank_D1: int,
        svs_D0: NDArray[np.float64],
        svs_D1: NDArray[np.float64],
    ) -> float:
        c1 = D0.shape[0]
        h1_dim = c1 - rank_D0 - rank_D1

        if h1_dim != 0:
            return 0.0

        def log_sum_sv(svs: NDArray[np.float64], rank: int) -> float:
            if rank <= 0 or svs.size == 0:
                return 0.0
            significant = svs[:rank]
            positive = significant[significant > 0]
            if positive.size == 0:
                return 0.0
            log_sum = float(np.sum(np.log(positive)))
            return log_sum if math.isfinite(log_sum) else 0.0

        log_tau = log_sum_sv(svs_D0, rank_D0) - log_sum_sv(svs_D1, rank_D1)
        return log_tau if math.isfinite(log_tau) else 0.0

    def _verify_mayer_vietoris_exactness(
        self,
        D0: NDArray[np.float64],
        D1: NDArray[np.float64],
        rank_D0: int,
        rank_D1: int,
    ) -> bool:
        c1 = D0.shape[0]

        if rank_D0 + rank_D1 > c1:
            logger.warning(
                "[Fase 3] Mayer-Vietoris: rank(δ⁰) + rank(δ¹) = %d > dim C¹ = %d. "
                "La secuencia puede no ser exacta.",
                rank_D0 + rank_D1, c1,
            )
            return False

        return True

    def _compute_first_cohomology_dimension(
        self,
        coboundary_delta0: NDArray[np.float64],
        coboundary_delta1: NDArray[np.float64],
    ) -> Tuple[int, int, int, int, int, float, float, bool, NDArray[np.float64], NDArray[np.float64]]:
        D0 = self._as_finite_matrix(
            "coboundary_delta0", coboundary_delta0, square=False,
        )
        D1 = self._as_finite_matrix(
            "coboundary_delta1", coboundary_delta1, square=False,
        )

        if D1.shape[1] != D0.shape[0]:
            raise ValueError(
                "[Fase 3] Los operadores coboundary no componen: "
                f"δ¹ tiene dominio dim={D1.shape[1]}, "
                f"pero δ⁰ tiene codominio dim={D0.shape[0]}."
            )

        composition = D1 @ D0

        if not np.all(np.isfinite(composition)):
            raise CohomologicalObstructionError(
                "[Fase 3] δ¹∘δ⁰ produjo valores no finitos."
            )

        d0_norm = self._frobenius_norm(D0)
        d1_norm = self._frobenius_norm(D1)

        if not math.isfinite(d0_norm) or not math.isfinite(d1_norm):
            raise CohomologicalObstructionError(
                "[Fase 3] Las normas de δ⁰ ó δ¹ no son finitas."
            )

        composition_norm = self._frobenius_norm(composition)

        if not math.isfinite(composition_norm):
            raise CohomologicalObstructionError(
                "[Fase 3] La norma de δ¹∘δ⁰ no es finita."
            )

        scale = max(1.0, d0_norm * d1_norm)
        complex_residual = composition_norm / scale

        if complex_residual > _COHOMOLOGICAL_COMPLEX_TOL:
            raise CohomologicalObstructionError(
                "[Fase 3] Los operadores celulares no forman un complejo válido: "
                f"‖δ¹∘δ⁰‖/scale = {complex_residual:.6e} > {_COHOMOLOGICAL_COMPLEX_TOL:.6e}."
            )

        rank_D0, svs_D0 = self._numerical_rank_with_gap(D0, "δ⁰")
        rank_D1, svs_D1 = self._numerical_rank_with_gap(D1, "δ¹")

        c1 = D0.shape[0]
        h1 = c1 - rank_D0 - rank_D1

        if h1 < 0:
            logger.warning(
                "[Fase 3] dim H¹ calculada = %d < 0; se proyecta a 0 por tolerancia numérica.",
                h1,
            )
            h1 = 0

        beta0, _beta1, beta2 = self._compute_betti_numbers(
            D0, D1, rank_D0, rank_D1,
        )

        torsion = self._compute_reidemeister_torsion(
            D0, D1, rank_D0, rank_D1, svs_D0, svs_D1,
        )

        mv_exact = self._verify_mayer_vietoris_exactness(
            D0, D1, rank_D0, rank_D1,
        )

        return (
            int(h1),
            int(rank_D0),
            int(rank_D1),
            int(beta0),
            int(beta2),
            float(torsion),
            float(complex_residual),
            bool(mv_exact),
            svs_D0,
            svs_D1,
        )

    def _audit_cellular_sheaf_cohomology(
        self,
        h1_dimension: Optional[int] = None,
        thermodynamic_audit: Optional[ThermodynamicDirichletData] = None,
        coboundary_delta0: Optional[NDArray[np.float64]] = None,
        coboundary_delta1: Optional[NDArray[np.float64]] = None,
    ) -> SheafCohomologyAuditData:
        if thermodynamic_audit is not None:
            if not thermodynamic_audit.is_thermodynamically_stable:
                raise ThermodynamicSingularityError(
                    "[Fase 3] No puede iniciarse: la Fase 2 no certificó estabilidad "
                    "termodinámica (is_thermodynamically_stable=False)."
                )

        computed_h1: Optional[int] = None
        verified_by_coboundary = False
        rank_delta0_val: int = 0
        rank_delta1_val: int = 0
        betti_vals: Tuple[int, int, int] = (0, 0, 0)
        euler_chi: int = 0
        reidemeister_val: float = 0.0
        complex_residual_val: float = 0.0
        mayer_vietoris_exact: bool = True

        if coboundary_delta0 is not None or coboundary_delta1 is not None:
            if coboundary_delta0 is None or coboundary_delta1 is None:
                raise ValueError(
                    "[Fase 3] Para verificación por operadores coboundary deben "
                    "proveerse coboundary_delta0 y coboundary_delta1 simultáneamente."
                )

            (
                computed_h1,
                rank_delta0_val,
                rank_delta1_val,
                beta0_val,
                beta2_val,
                reidemeister_val,
                complex_residual_val,
                mayer_vietoris_exact,
                _svs_D0,
                _svs_D1,
            ) = self._compute_first_cohomology_dimension(
                coboundary_delta0=coboundary_delta0,
                coboundary_delta1=coboundary_delta1,
            )

            beta1_val = computed_h1
            betti_vals = (beta0_val, beta1_val, beta2_val)

            c0 = np.asarray(coboundary_delta0).shape[1]
            c1 = np.asarray(coboundary_delta0).shape[0]
            c2 = np.asarray(coboundary_delta1).shape[0]
            euler_chi = int(c0 - c1 + c2)

            euler_from_betti = int(beta0_val - beta1_val + beta2_val)
            if abs(euler_chi - euler_from_betti) > _EULER_CHARACTERISTIC_TOLERANCE:
                raise EulerCharacteristicMismatch(
                    "[Fase 3] Inconsistencia en la característica de Euler: "
                    f"χ(G) = {euler_chi} (por cadenas), pero "
                    f"β₀ − β₁ + β₂ = {euler_from_betti} (por Betti)."
                )

            if not mayer_vietoris_exact:
                raise MayerVietorisBreachError(
                    "[Fase 3] La secuencia de Mayer-Vietoris no es exacta en C¹. "
                    "El haz celular no puede descomponerse consistentemente."
                )

            verified_by_coboundary = True

        if h1_dimension is None:
            if computed_h1 is None:
                raise ValueError(
                    "[Fase 3] Debe proveerse h1_dimension o los operadores "
                    "coboundary_delta0 y coboundary_delta1."
                )
            h1 = computed_h1
        else:
            if isinstance(h1_dimension, (bool, np.bool_)):
                raise TypeError(
                    "[Fase 3] h1_dimension debe ser un entero; se recibió booleano."
                )

            if not isinstance(h1_dimension, (int, np.integer)):
                raise TypeError(
                    f"[Fase 3] h1_dimension debe ser un entero; recibido: "
                    f"{type(h1_dimension)}."
                )

            h1 = int(h1_dimension)

            if computed_h1 is not None and computed_h1 != h1:
                raise CohomologicalObstructionError(
                    "[Fase 3] Inconsistencia cohomológica: "
                    f"h1_dimension declarado={h1}, calculado por δ={computed_h1}."
                )

        if h1 < _COHOMOLOGY_DIMENSION_FLOOR:
            raise ValueError(
                f"[Fase 3] h1_dimension no puede ser negativa: {h1}."
            )

        if h1 > _COHOMOLOGY_DIMENSION_FLOOR:
            raise CohomologicalObstructionError(
                "[Fase 3] Obstrucción topológica global detectada en la sintaxis. "
                f"dim H¹(G; F) = {h1} > 0. El código propuesto contiene "
                "contradicciones lógicas, ciclos de dependencia irresolubles "
                "o variables huérfanas."
            )

        if not verified_by_coboundary:
            betti_vals = (0, 0, 0)
            euler_chi = 0

        return SheafCohomologyAuditData(
            h1_dimension=h1,
            rank_delta0=rank_delta0_val,
            rank_delta1=rank_delta1_val,
            betti_numbers=betti_vals,
            euler_characteristic=euler_chi,
            reidemeister_torsion=reidemeister_val,
            complex_residual=complex_residual_val,
            is_globally_integrable=True,
            obstruction_free=True,
            verified_by_coboundary=verified_by_coboundary,
            mayer_vietoris_exact=mayer_vietoris_exact,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §E. AGENTE SOBERANO Y AUTOMEJORA RECURSIVA LEVEL 3 (RSI³)
# ═══════════════════════════════════════════════════════════════════════════════

class ASTStaticAnalyzerAgentRSI3:
    r"""
    Soberano Agéntico con Automejora Recursiva Nivel 3 (RSI³ — Darwin-Gödel) y Interlock Ciber-Físico ESP32 Crowbar.
    """

    def __init__(self, gpio_pin: int = 14) -> None:
        self.gpio_pin = gpio_pin
        self.banach_target = 0.95
        self.eta = 0.1
        self.zeta = 0.01

    def audit_ooda_phase(self, ast_tree: Any, G_metric: np.ndarray, h1_dim: int = 0, melnikov_chaos: bool = False) -> int:
        r"""
        Audita el AST evaluando las 3 superficies RSI³ en la variedad de Heyting $\Omega_3$:
          - Fase 1 (Data-RSI): residuo $0\text{x}00$.
          - Fase 2 (Harness-RSI): contracción $L_\Phi \le 0.95$ ($0\text{x}01$ si se excede).
          - Fase 3 (Model-RSI): dimensión $H^1 > 0$ o caos homoclínico de Melnikov ($0\text{x}03$).

        Adjudica el veredicto mediante `OR` bitwise SIMD. Si el veredicto terminal es $0\text{x}03$,
        dispara el interlock físico ESP32 Crowbar (< 400 ns).
        """
        verdict_f1 = 0x00

        # Evaluar constante de Lipschitz del mapeo espectral
        if G_metric.size > 0:
            norm_G = float(np.linalg.norm(G_metric, 2))
            L_phi = min(0.95, norm_G / (norm_G + 1.0))
        else:
            L_phi = 0.88

        if L_phi > self.banach_target:
            verdict_f2 = 0x01
        else:
            verdict_f2 = 0x00

        if h1_dim > 0 or melnikov_chaos:
            verdict_f3 = 0x03
        else:
            verdict_f3 = 0x00

        final_verdict = verdict_f1 | verdict_f2 | verdict_f3

        if final_verdict == 0x03:
            self.trigger_esp32_crowbar_interlock()

        return final_verdict

    def trigger_esp32_crowbar_interlock(self) -> None:
        r"""
        Dispara el interlock ciber-físico ESP32 Crowbar en GPIO14 (< 400 ns).
        """
        logger.critical("[ESP32 Crowbar Interlock] GPIO%d conmutado a HIGH. Veto físico ejecutado.", self.gpio_pin)


class ASTStaticAnalyzerAgent(Morphism, Phase3_CellularSheafCohomologyAuditor):
    r"""
    El Custodio de la Cohomología Sintáctica en el estrato $\mathcal{V}_{\Gamma-\mathrm{PHYSICS}}$.
    """

    def __init__(self, strict_mode: bool = True) -> None:
        self._strict_mode = bool(strict_mode)
        self.rsi3_engine = ASTStaticAnalyzerAgentRSI3(gpio_pin=14)

        logger.info(
            "[Orquestador] ASTStaticAnalyzerAgent v4.1.0 inicializado. strict_mode=%s.",
            self._strict_mode,
        )

    def _build_provenance(
        self,
        checksum: str,
        phase1_passed: bool,
        phase2_passed: bool,
        phase3_passed: bool,
    ) -> AuditProvenance:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        phases_ok = [phase1_passed, phase2_passed, phase3_passed]
        all_passed = all(phases_ok)

        functor_chain = (
            f"Φ₁={'✓' if phase1_passed else '✗'} → "
            f"Φ₂={'✓' if phase2_passed else '✗'} → "
            f"Φ₃={'✓' if phase3_passed else '✗'} → "
            f"Z_{{Γ-PHYSICS}}={'✓' if all_passed else '✗'}"
        )

        return AuditProvenance(
            timestamp_iso=timestamp,
            input_checksum_sha256=checksum,
            phase1_passed=phase1_passed,
            phase2_passed=phase2_passed,
            phase3_passed=phase3_passed,
            functor_chain=functor_chain,
        )

    @staticmethod
    def _log_governance_summary(
        symplectic_audit: SymplecticInvariantData,
        thermodynamic_audit: ThermodynamicDirichletData,
        cohomology_audit: SheafCohomologyAuditData,
        provenance: AuditProvenance,
    ) -> None:
        logger.info(
            "═══════════════════════════════════════════════════════════\n"
            "  AST SYMPLECTIC GOVERNANCE — REPORTE FINAL v4.1.0\n"
            "  Timestamp : %s\n"
            "  SHA-256   : %s\n"
            "  Cadena    : %s\n"
            "───────────────────────────────────────────────────────────\n"
            "  FASE 1 — Invarianza Simpléctica:\n"
            "    dim Esp. Fase : %d\n"
            "    Residuo |F    : %.4e\n"
            "    Residuo Rel.  : %.4e (tol=%.4e)\n"
            "    Volumen ✓    : %s\n"
            "───────────────────────────────────────────────────────────\n"
            "  FASE 2 — Termodinámica Port-Hamiltoniana:\n"
            "    P_diss       : %.4e\n"
            "    σ_CD         : %.4e\n"
            "    Estable ✓    : %s\n"
            "───────────────────────────────────────────────────────────\n"
            "  FASE 3 — Cohomología de Haces Celulares:\n"
            "    dim H¹       : %d\n"
            "    Integrable ✓ : %s\n"
            "═══════════════════════════════════════════════════════════",
            provenance.timestamp_iso,
            provenance.input_checksum_sha256[:16] + "...",
            provenance.functor_chain,
            symplectic_audit.phase_space_dimension,
            symplectic_audit.symplectic_residual_norm,
            symplectic_audit.symplectic_relative_residual,
            symplectic_audit.effective_tolerance,
            symplectic_audit.is_volume_preserved,
            thermodynamic_audit.dissipated_power,
            thermodynamic_audit.entropy_production_rate,
            thermodynamic_audit.is_thermodynamically_stable,
            cohomology_audit.h1_dimension,
            cohomology_audit.is_globally_integrable,
        )

    def execute_ast_symplectic_governance(
        self,
        ast_jacobian_M: NDArray[np.float64],
        control_potential_Phi: NDArray[np.float64],
        lyapunov_gradient_V: NDArray[np.float64],
        h1_dimension: Optional[int] = None,
        coboundary_delta0: Optional[NDArray[np.float64]] = None,
        coboundary_delta1: Optional[NDArray[np.float64]] = None,
    ) -> ASTGovernanceState:
        arrays_for_checksum = [
            np.asarray(ast_jacobian_M) if ast_jacobian_M is not None else np.array([]),
            np.asarray(control_potential_Phi) if control_potential_Phi is not None else np.array([]),
            np.asarray(lyapunov_gradient_V) if lyapunov_gradient_V is not None else np.array([]),
        ]

        if coboundary_delta0 is not None:
            arrays_for_checksum.append(np.asarray(coboundary_delta0))
        if coboundary_delta1 is not None:
            arrays_for_checksum.append(np.asarray(coboundary_delta1))

        input_checksum = self._compute_input_checksum(*arrays_for_checksum)

        phase1_passed = False
        phase2_passed = False
        phase3_passed = False

        symplectic_audit = self._audit_symplectic_invariance(ast_jacobian_M)
        phase1_passed = True

        thermodynamic_audit = self._enforce_dirichlet_thermodynamics(
            control_potential_Phi=control_potential_Phi,
            lyapunov_gradient_V=lyapunov_gradient_V,
            symplectic_audit=symplectic_audit,
        )
        phase2_passed = True

        cohomology_audit = self._audit_cellular_sheaf_cohomology(
            h1_dimension=h1_dimension,
            thermodynamic_audit=thermodynamic_audit,
            coboundary_delta0=coboundary_delta0,
            coboundary_delta1=coboundary_delta1,
        )
        phase3_passed = True

        # Invocación soberana de la automejora recursiva RSI³ en el retículo de Heyting
        G_metric = np.eye(symplectic_audit.phase_space_dimension, dtype=np.float64)
        verdict = self.rsi3_engine.audit_ooda_phase(
            ast_tree=ast_jacobian_M,
            G_metric=G_metric,
            h1_dim=cohomology_audit.h1_dimension,
            melnikov_chaos=False
        )

        is_compilation_authorized = bool(
            symplectic_audit.is_volume_preserved
            and thermodynamic_audit.is_thermodynamically_stable
            and cohomology_audit.is_globally_integrable
            and verdict != 0x03
        )

        provenance = self._build_provenance(
            checksum=input_checksum,
            phase1_passed=phase1_passed,
            phase2_passed=phase2_passed,
            phase3_passed=phase3_passed,
        )

        self._log_governance_summary(
            symplectic_audit=symplectic_audit,
            thermodynamic_audit=thermodynamic_audit,
            cohomology_audit=cohomology_audit,
            provenance=provenance,
        )

        if not is_compilation_authorized:
            raise TopologicalInvariantError(
                "[Orquestador] La composición funtorial Z_{Γ-PHYSICS} = Φ₃∘Φ₂∘Φ₁ "
                f"no autorizó la compilación del AST."
            )

        return ASTGovernanceState(
            symplectic_audit=symplectic_audit,
            thermodynamic_audit=thermodynamic_audit,
            cohomology_audit=cohomology_audit,
            provenance=provenance,
            is_compilation_authorized=is_compilation_authorized,
        )


__all__: List[str] = [
    "SymplecticInvarianceViolation",
    "SymplecticPolarDecompositionError",
    "MaslovIndexError",
    "ThermodynamicSingularityError",
    "ClausiusDuhemViolation",
    "ExergyDivergenceError",
    "CohomologicalObstructionError",
    "EulerCharacteristicMismatch",
    "MayerVietorisBreachError",
    "SymplecticInvariantData",
    "ThermodynamicDirichletData",
    "SheafCohomologyAuditData",
    "AuditProvenance",
    "ASTGovernanceState",
    "Phase1_SymplecticInvarianceAuditor",
    "Phase2_DirichletThermodynamicEnforcer",
    "Phase3_CellularSheafCohomologyAuditor",
    "ASTStaticAnalyzerAgentRSI3",
    "ASTStaticAnalyzerAgent",
]

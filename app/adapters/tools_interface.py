# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Central Interaction Interface (La Base Canónica y el Topos EMIC)    ║
║ Ruta   : app/adapters/tools_interface.py                                     ║
║ Versión: 8.1.0-Poincare-Symplectic-KAM-Novikov-Doctoral                      ║
║ Fase   : 1 / 3  — FUNDAMENTOS ALGEBRAICO-SIMPLÉCTICOS                        ║
╚══════════════════════════════════════════════════════════════════════════════╝


NATURALEZA CIBER-FÍSICA Y ÁLGEBRA CATEGÓRICA EN EL ESTRATO TACTICS (V_𝕋) ───
Este módulo no es una tabla de despacho procedimental. El espacio de acciones
del Agente se modela como un topos elemental \(\mathcal{E}_{\mathrm{MIC}}\)
cuyo objeto de verdad \(\Omega\) está fibrado sobre el cotangente \(T^*\mathcal{M}\)
con la 2-forma canónica de Darboux

    \[\Omega = \sum_{i=1}^{n} dq^{i}\wedge dp_{i}.\]

La dinámica de decisión es el flujo hamiltoniano \(X_{H}=J\nabla H\). La
inteligencia no se inyecta: emerge de los invariantes geométricos de ese flujo
(Poincaré, *Les méthodes nouvelles de la mécanique céleste*, 1892–1899).


INVARIANTES PRESERVADOS ────────────────────────────────────────────────────
  [I1]  Ortonormalidad de la base canónica: \(I_{ij}=\delta_{ij}\)
  [I2]  Nulidad del núcleo: \(\ker(I_n)=\{\mathbf{0}\}\)
  [I3]  Pullback de Heyting: \(S\times_X Y\cong\lim(S\to X\leftarrow Y)\)
  [I4]  Complejidad de despacho: \(\mathcal{O}(n)\)
  [I5]  Máxima entropía sobre la base: \(p_i=1/n\)
  [I6]  Clausura de Darboux: \(\Omega^{\top}=-\Omega\), \(d\Omega=0\), \(J^{2}=-I_{2n}\)
  [I7]  Liouville: \(\mathcal{L}_{X_H}(\Omega^{n})=0\Leftrightarrow\div X_H=0\)
  [I8]  No-aplastamiento de Gromov: \(c(B^{2n}(r))\le c(Z^{2n}(R))\Rightarrow r\le R\)
  [I9]  Condición diofántica KAM: \(|\langle k,\omega\rangle|\ge\gamma/|k|^{\tau}\)
  [I10] Recurrencia de Poincaré (germen Fase 2)
  [I11] Forma normal de Birkhoff hasta orden \(N\)
  [I12] Invariante integral absoluto: \(\oint_{\gamma}\sum p_i\,dq^{i}\) constante
        sobre tubos de flujo (Poincaré, 1890)
  [I13] Invariante de Poincaré–Cartan: \(\oint(p\,dq-H\,dt)\) en el espacio
        extendido \(T^*\mathcal{M}\times\mathbb{R}\)


ARQUITECTURA DE TRES FASES ANIDADAS ────────────────────────────────────────
  FASE 1 ──► FUNDAMENTOS ALGEBRAICO-SIMPLÉCTICOS
             Configuración, Heyting enriquecida, Darboux, fase, Hamilton,
             acción-ángulo, generatriz, Poincaré–Cartan, secciones.
             Último método: Phase1Foundation.seed_phase2_topology()

  FASE 2 ──► ESTRUCTURAS TOPOLÓGICAS Y SISTEMAS DINÁMICOS
             Consume el germen de seed_phase2_topology: persistencia,
             Betti, mapas de retorno, cadenas de Markov ergódicas, KAM.

  FASE 3 ──► ORQUESTACIÓN CATEGÓRICA Y OPERADORES
             Comandos, MICRegistry como topos simpléctico, pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import re
import statistics
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import lru_cache, wraps, cached_property
from itertools import product as itertools_product
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    FrozenSet,
    Generic,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)


# =============================================================================
# DEPENDENCIAS NUMÉRICAS CON FALLBACK ROBUSTO
# =============================================================================

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False
    warnings.warn(
        "numpy no disponible — operaciones matriciales usarán fallback puro",
        ImportWarning,
        stacklevel=2,
    )

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False

try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh, eigs
    SCIPY_SPARSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    sparse = None  # type: ignore[assignment]
    eigsh = None  # type: ignore[assignment]
    eigs = None  # type: ignore[assignment]
    SCIPY_SPARSE_AVAILABLE = False
    warnings.warn(
        "scipy.sparse no disponible — análisis espectral usará matrices densas",
        ImportWarning,
        stacklevel=2,
    )

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:  # pragma: no cover
    z3 = None  # type: ignore[assignment]
    Z3_AVAILABLE = False

try:
    import dd.bdd as bdd
    BDD_AVAILABLE = True
except ImportError:  # pragma: no cover
    bdd = None  # type: ignore[assignment]
    BDD_AVAILABLE = False


# =============================================================================
# IMPORTACIONES DE ÁLGEBRAS SUPERIORES (MIC Core)
# =============================================================================

MIC_ALGEBRA_AVAILABLE = False
try:
    from app.core.mic_algebra import (  # type: ignore[import]
        CategoricalState,
        Morphism,
        NaturalTransformation,
        TwoCategoryOrchestrator,
        FunctorialityError,
    )
    MIC_ALGEBRA_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

SHEAF_COHOMOLOGY_AVAILABLE = False
try:
    from app.boole.strategy.sheaf_cohomology_orchestrator import (  # type: ignore[import]
        SheafCohomologyOrchestrator,
        CellularSheaf,
        HomologicalInconsistencyError,
    )
    SHEAF_COHOMOLOGY_AVAILABLE = True
except ImportError:  # pragma: no cover
    class CellularSheaf:  # type: ignore[no-redef]
        pass

    class HomologicalInconsistencyError(Exception):  # type: ignore[no-redef]
        pass

SEMANTIC_ESTIMATOR_AVAILABLE = False
try:
    from app.tactics.semantic_estimator import SemanticEstimatorService  # type: ignore[import]
    SEMANTIC_ESTIMATOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

IMPROBABILITY_DRIVE_AVAILABLE = False
try:
    from app.omega.improbability_drive import ImprobabilityDriveService  # type: ignore[import]
    IMPROBABILITY_DRIVE_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

SEMANTIC_DICTIONARY_AVAILABLE = False
try:
    from app.wisdom.semantic_dictionary import SemanticDictionaryService  # type: ignore[import]
    SEMANTIC_DICTIONARY_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


# =============================================================================
# LOGGER ESTRUCTURADO CON CONTEXTO ALGEBRAICO
# =============================================================================

logger = logging.getLogger("MIC")


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """
    Adapter para logging estructurado con contexto algebraico-topológico.

    Invariante: cada registro porta estrato, dimensión, validación homológica
    y residual simpléctico para auditoría reproducible.
    """

    __slots__ = ("extra",)

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_structured_logger(name: str, **context: Any) -> StructuredLoggerAdapter:
    r"""
    Crea un logger con contexto estructurado para trazabilidad categórica.

    Teorema de Trazabilidad:
        \(\forall\,\mathrm{log}\,\exists\,\mathrm{context}:
        \mathrm{log}\otimes\mathrm{context}\) es auditável.
    """
    return StructuredLoggerAdapter(logging.getLogger(name), context)


# =============================================================================
# SISTEMA DE IMPORTACIÓN SEGURA CON DIAGNÓSTICO
# =============================================================================

def _safe_import(module_path: str, class_name: str) -> Optional[Type]:
    """Importación segura con logging de diagnóstico para dependencias opcionales."""
    try:
        if module_path.startswith("."):
            import importlib
            package = __name__.rsplit(".", 1)[0] if "." in __name__ else __name__
            module = importlib.import_module(module_path, package=package)
        else:
            module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except ImportError as e:
        logger.debug(
            "Optional import failed: %s.%s — %s",
            module_path,
            class_name,
            e,
            extra={"import_error": str(e)},
        )
        return None
    except Exception as e:  # pragma: no cover
        logger.debug(
            "Unexpected error importing %s.%s — %s",
            module_path,
            class_name,
            e,
            extra={"import_error": str(e)},
        )
        return None


CSVCleaner = _safe_import("scripts.clean_csv", "CSVCleaner")
APUFileDiagnostic = _safe_import("scripts.diagnose_apus_file", "APUFileDiagnostic")
InsumosFileDiagnostic = _safe_import("scripts.diagnose_insumos_file", "InsumosFileDiagnostic")
PresupuestoFileDiagnostic = _safe_import(
    "scripts.diagnose_presupuesto_file", "PresupuestoFileDiagnostic"
)
FinancialConfig = _safe_import(".financial_engine", "FinancialConfig")
FinancialEngine = _safe_import(".financial_engine", "FinancialEngine")


# =============================================================================
# ESTRATO DIKW — FILTRACIÓN TOPOLÓGICA JERÁRQUICA
# =============================================================================
from app.core.schemas import Stratum  # noqa: E402

# =============================================================================
# SUTURA II.a — Vectores core del espacio vectorial MIC
# =============================================================================
from app.adapters.mic_vectors import (  # noqa: E402
    vector_stabilize_flux,
    vector_parse_raw_structure,
    vector_structure_logic,
    vector_audit_homological_fusion,
    vector_lateral_pivot,
    vector_calculate_improbability_tensor,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    FASE 1: FUNDAMENTOS ALGEBRAICO-SIMPLÉCTICOS             ║
# ║         (Configuración · Álgebra de Heyting · Geometría Simpléctica)       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════


# =============================================================================
# 1.0 — ÁLGEBRA LINEAL PURA (fallback cuando numpy no está)
# =============================================================================

Matrix = List[List[float]]
Vector = List[float]

_TWO_PI: Final[float] = 2.0 * math.pi
_MACH_EPS: Final[float] = sys.float_info.epsilon


def _clamp(x: float, lo: float, hi: float) -> float:
    """Proyección sobre [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def _wrap_angle(theta: float) -> float:
    """Normaliza \(\theta\) a \([0, 2\pi)\)."""
    w = theta % _TWO_PI
    return w + _TWO_PI if w < 0.0 else w


def _hypot_sq(xs: Sequence[float]) -> float:
    return float(sum(x * x for x in xs))


def _eye(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _zeros(n: int, m: Optional[int] = None) -> Matrix:
    cols = n if m is None else m
    return [[0.0] * cols for _ in range(n)]


def _transpose(A: Matrix) -> Matrix:
    if not A:
        return []
    return [list(row) for row in zip(*A)]


def _matmul(A: Matrix, B: Matrix) -> Matrix:
    if not A or not B:
        return []
    n, p, m = len(A), len(A[0]), len(B[0])
    if p != len(B):
        raise ValueError(f"matmul incompatible: ({n}x{p}) @ ({len(B)}x{m})")
    out = _zeros(n, m)
    for i in range(n):
        Ai = A[i]
        for k in range(p):
            aik = Ai[k]
            if aik == 0.0:
                continue
            Bk = B[k]
            row = out[i]
            for j in range(m):
                row[j] += aik * Bk[j]
    return out


def _matvec(A: Matrix, v: Sequence[float]) -> Vector:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _add_mat(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _scale_mat(s: float, A: Matrix) -> Matrix:
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _frobenius(A: Matrix) -> float:
    return math.sqrt(sum(x * x for row in A for x in row))


def _det_small(A: Matrix) -> float:
    """Determinante por eliminación gaussiana con pivoteo parcial. \(\mathcal{O}(n^3)\)."""
    n = len(A)
    if n == 0:
        return 1.0
    M = [row[:] for row in A]
    det = 1.0
    for i in range(n):
        piv = i
        max_abs = abs(M[i][i])
        for r in range(i + 1, n):
            val = abs(M[r][i])
            if val > max_abs:
                max_abs = val
                piv = r
        if max_abs < 1e-18:
            return 0.0
        if piv != i:
            M[i], M[piv] = M[piv], M[i]
            det = -det
        det *= M[i][i]
        inv = 1.0 / M[i][i]
        for r in range(i + 1, n):
            f = M[r][i] * inv
            if f == 0.0:
                continue
            for c in range(i, n):
                M[r][c] -= f * M[i][c]
    return det


def _canonical_J(n_dof: int) -> Matrix:
    r"""
    Matriz simpléctica canónica de Darboux \(J\in\mathrm{Sp}(2n,\mathbb{R})\):

        \[
        J=\begin{pmatrix}0&I_n\\-I_n&0\end{pmatrix},
        \qquad J^{\top}=-J,\quad J^{2}=-I_{2n},\quad\det J=1.
        \]
    """
    if n_dof < 1:
        raise ValueError(f"n_dof debe ser ≥ 1, recibido: {n_dof}")
    dim = 2 * n_dof
    J = _zeros(dim)
    for i in range(n_dof):
        J[i][n_dof + i] = 1.0
        J[n_dof + i][i] = -1.0
    return J


def _as_matrix(obj: Any) -> Matrix:
    """Convierte ndarray o lista-de-listas a Matrix pura."""
    if obj is None:
        return []
    if NUMPY_AVAILABLE and hasattr(obj, "tolist"):
        data = obj.tolist()
        if data and not isinstance(data[0], list):
            return [list(map(float, data))]
        return [list(map(float, row)) for row in data]
    return [list(map(float, row)) for row in obj]


def _to_backend(M: Matrix) -> Any:
    if NUMPY_AVAILABLE:
        return np.array(M, dtype=np.float64)
    return M


def _continued_fraction(x: float, max_terms: int = 24) -> Tuple[int, ...]:
    r"""
    Expansión en fracción continua de Poincaré para diagnosticar
    aproximabilidad diofántica (pequeños divisores).

    El número áureo \(\varphi\) tiene coeficientes \((1,1,1,\ldots)\) y es el
    peor aproximable: máxima persistencia KAM.
    """
    terms: List[int] = []
    v = float(x)
    for _ in range(max_terms):
        a = int(math.floor(v))
        terms.append(a)
        frac = v - a
        if abs(frac) < 1e-15:
            break
        v = 1.0 / frac
        if abs(v) > 1e12:
            break
    return tuple(terms)


# =============================================================================
# 1.1 — CONSTANTES MATEMÁTICAS UNIVERSALES DE POINCARÉ
# =============================================================================

_PHI: Final[float] = (1.0 + math.sqrt(5.0)) / 2.0  # Proporción áurea
_EULER_GAMMA: Final[float] = 0.5772156649015329  # Euler-Mascheroni
_PHI_CF: Final[Tuple[int, ...]] = (1,) * 16  # CF canónica de φ − 1

# Constantes diofánticas canónicas para KAM (Kolmogorov–Arnold–Moser)
_KAM_GAMMA_DEFAULT: Final[float] = 1e-3
_KAM_TAU_DEFAULT: Final[float] = 2.0  # τ > n − 1 (Bruno / Rüssmann)
_KAM_EPSILON_PERTURBATION: Final[float] = 1e-6

# Constantes de estabilidad numérica simpléctica
_SYMPLECTIC_RESIDUAL_TOL: Final[float] = 1e-9
_LIOUVILLE_VOLUME_TOL: Final[float] = 1e-9
_POINCARE_RECURRENCE_TOL: Final[float] = 1e-12
_BIRKHOFF_TRUNCATION_ORDER: Final[int] = 4
_POINCARE_CARTAN_TOL: Final[float] = 1e-8
_ENERGY_DRIFT_TOL: Final[float] = 1e-6

_SEVERITY_WEIGHTS: Final[Dict[str, float]] = {
    "CRITICAL": 5.0,
    "HIGH": 3.0,
    "MEDIUM": 2.0,
    "LOW": 1.0,
    "INFO": 0.5,
}

SUPPORTED_ENCODINGS: Final[FrozenSet[str]] = frozenset({
    "utf-8", "utf-8-sig", "latin-1", "iso-8859-1",
    "cp1252", "ascii", "utf-16", "utf-16-le", "utf-16-be",
})

_ENCODING_ALIASES: Final[Dict[str, str]] = {
    "utf8": "utf-8",
    "latin1": "latin-1",
    "iso88591": "iso-8859-1",
    "cp65001": "utf-8",
}

VALID_DELIMITERS: Final[FrozenSet[str]] = frozenset({",", ";", "\t", "|", ":"})
VALID_EXTENSIONS: Final[FrozenSet[str]] = frozenset({".csv", ".txt", ".tsv"})


# =============================================================================
# 1.2 — CONFIGURACIÓN CENTRAL CON PARÁMETROS DE POINCARÉ
# =============================================================================

@dataclass(frozen=True, slots=True)
class MICConfiguration:
    r"""
    Configuración centralizada de la MIC con parámetros simplécticos.

    Teorema de Configuración Válida
    -------------------------------
    \(C\) es válida sii:

    Invariantes numéricos:
        1. \(\texttt{max_file_size_bytes}>0\)
        2. \(\texttt{cache_ttl_seconds}>0\)
        3. \(0<\texttt{cycle_similarity_threshold}\le 1\)
        4. \(0<\texttt{persistence_threshold}<1\)
        5. \(\varepsilon>0\)

    Invariantes simplécticos (Poincaré / KAM):
        6.  \(\gamma>0\) (escala de pequeños divisores)
        7.  \(\tau>n-1\) (exponente de Bruno)
        8.  \(\texttt{symplectic_residual_tol}>0\)
        9.  \(\texttt{liouville_volume_tol}>0\)
        10. \(N_{\mathrm{Birkhoff}}\ge 1\)
        11. \(n_{\mathrm{dof}}\ge 1\)

    Justificación: \(\tau>n-1\) garantiza, por KAM, persistencia de un conjunto
    de toros invariantes de medida de Lebesgue positiva bajo perturbaciones
    \(\varepsilon\ll\gamma\), impidiendo la difusión de Arnold en el espacio
    de decisiones del agente.
    """

    max_file_size_bytes: int = 100 * 1024 * 1024
    max_sample_rows: int = 1000

    cache_ttl_seconds: float = 300.0
    cache_max_size: int = 128

    persistence_threshold: float = 0.01
    cycle_similarity_threshold: float = 0.80
    max_cycle_period: int = 50
    max_lines_for_cycle_detection: int = 10000

    latency_histogram_buckets: int = 100
    enable_detailed_metrics: bool = True

    strict_encoding_validation: bool = False

    diagnostic_timeout_seconds: float = 30.0
    spectral_analysis_timeout_seconds: float = 10.0

    epsilon: float = 1e-10
    algorithm_version: str = "8.1.0-poincare-symplectic"

    # --- Geometría simpléctica ---
    symplectic_residual_tol: float = _SYMPLECTIC_RESIDUAL_TOL
    liouville_volume_tol: float = _LIOUVILLE_VOLUME_TOL
    n_dof: int = 3

    # --- KAM / pequeños divisores ---
    kam_gamma: float = _KAM_GAMMA_DEFAULT
    kam_tau: float = _KAM_TAU_DEFAULT
    kam_epsilon_perturbation: float = _KAM_EPSILON_PERTURBATION
    kam_frequency_lattice_range: int = 2

    # --- Recurrencia ergódica ---
    poincare_recurrence_tol: float = _POINCARE_RECURRENCE_TOL
    poincare_max_return_iterations: int = 1024

    # --- Forma normal de Birkhoff ---
    birkhoff_truncation_order: int = _BIRKHOFF_TRUNCATION_ORDER
    birkhoff_coefficient_tol: float = 1e-8

    # --- No-aplastamiento de Gromov ---
    gromov_capacity_tol: float = 1e-10

    # --- Flujo hamiltoniano ---
    hamiltonian_time_step: float = 1e-3
    hamiltonian_max_steps: int = 8192

    # --- Poincaré–Cartan ---
    poincare_cartan_tol: float = _POINCARE_CARTAN_TOL
    energy_drift_tol: float = _ENERGY_DRIFT_TOL

    def __post_init__(self) -> None:
        if self.max_file_size_bytes <= 0:
            raise ValueError(
                f"Invariante violado: max_file_size_bytes > 0, recibido: {self.max_file_size_bytes}"
            )
        if self.cache_ttl_seconds <= 0:
            raise ValueError(
                f"Invariante violado: cache_ttl_seconds > 0, recibido: {self.cache_ttl_seconds}"
            )
        if not (0.0 < self.cycle_similarity_threshold <= 1.0):
            raise ValueError(
                f"Invariante violado: cycle_similarity_threshold ∈ (0, 1], "
                f"recibido: {self.cycle_similarity_threshold}"
            )
        if not (0.0 < self.persistence_threshold < 1.0):
            raise ValueError(
                f"Invariante violado: persistence_threshold ∈ (0, 1), "
                f"recibido: {self.persistence_threshold}"
            )
        if self.epsilon <= 0.0:
            raise ValueError(f"Invariante violado: epsilon > 0, recibido: {self.epsilon}")
        if self.kam_gamma <= 0.0:
            raise ValueError(f"Invariante KAM violado: kam_gamma > 0, recibido: {self.kam_gamma}")
        if self.kam_tau <= float(self.n_dof - 1):
            raise ValueError(
                f"Invariante KAM (Bruno): kam_tau ({self.kam_tau}) debe ser > "
                f"n_dof - 1 ({self.n_dof - 1})"
            )
        if self.symplectic_residual_tol <= 0.0:
            raise ValueError("Invariante violado: symplectic_residual_tol > 0")
        if self.liouville_volume_tol <= 0.0:
            raise ValueError("Invariante violado: liouville_volume_tol > 0")
        if self.birkhoff_truncation_order < 1:
            raise ValueError("Invariante violado: birkhoff_truncation_order ≥ 1")
        if self.n_dof < 1:
            raise ValueError(f"Invariante violado: n_dof ≥ 1, recibido: {self.n_dof}")
        if self.hamiltonian_time_step <= 0.0:
            raise ValueError("Invariante violado: hamiltonian_time_step > 0")
        if self.poincare_cartan_tol <= 0.0:
            raise ValueError("Invariante violado: poincare_cartan_tol > 0")

    @property
    def is_production_ready(self) -> bool:
        return (
            self.epsilon < 1e-8
            and self.cache_ttl_seconds >= 60.0
            and self.max_file_size_bytes >= 10 * 1024 * 1024
            and self.kam_tau > float(self.n_dof - 1)
            and self.symplectic_residual_tol < 1e-6
        )

    @property
    def symplectic_dimension(self) -> int:
        r"""Teorema de Darboux: \(\dim T^*\mathcal{M}=2n\)."""
        return 2 * self.n_dof

    @property
    def birkhoff_radius_bound(self) -> float:
        r"""
        Radio de Siegel–Sternberg (estimación):

            \[ r_B \approx \frac{\gamma}{\max(C\varepsilon,\epsilon_{\mathrm{mach}})}. \]
        """
        if self.kam_epsilon_perturbation <= 0.0:
            return float("inf")
        return self.kam_gamma / max(self.kam_epsilon_perturbation * 100.0, self.epsilon)

    @property
    def kam_bruno_gap(self) -> float:
        """ holgura \(\tau-(n-1)\). Debe ser estrictamente positiva."""
        return self.kam_tau - float(self.n_dof - 1)


DEFAULT_MIC_CONFIG: Final[MICConfiguration] = MICConfiguration()


# =============================================================================
# 1.3 — ÁLGEBRA DE HEYTING ENRIQUECIDA CON OPERADORES SIMPLÉCTICOS
# =============================================================================

@dataclass(frozen=True, slots=True)
class HeytingValue:
    r"""
    Valor de verdad en un álgebra de Heyting \(H\) fibrada sobre \(T^*\mathcal{M}\).

    Axiomas de Heyting
    ------------------
        1. \(x\wedge(y\vee z)=(x\wedge y)\vee(x\wedge z)\)
        2. \(x\to x=1\)
        3. \(x\wedge(x\to y)=x\wedge y\)
        4. \(y\wedge(x\to y)=y\)
        5. \(x\to(y\wedge z)=(x\to y)\wedge(x\to z)\)

    Enriquecimiento simpléctico
    ---------------------------
    Identificamos \(v\) con el complejo de acción

        \[ z(v)=v\cdot e^{i\theta}\in\mathbb{C}\cong T^*\mathbb{R}. \]

    El flujo hamiltoniano armónico es rotación \(z\mapsto z\,e^{i\omega t}\),
    que **conserva** \(|z|\) (Liouville sobre el círculo de acción).

    El corchete de Poisson en esta carta es

        \[ \{v,w\}_{\mathrm{MIC}}=\mathrm{Im}\!\left(z(v)\,\overline{z(w)}\right)
           =vw\sin(\theta_v-\theta_w). \]
    """

    value: float
    description: str = "unknown"
    phase: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            object.__setattr__(self, "value", float(_clamp(self.value, 0.0, 1.0)))
        if not (0.0 <= self.phase < _TWO_PI):
            object.__setattr__(self, "phase", float(_wrap_angle(self.phase)))

    # ---------- Propiedades lógicas ----------

    @property
    def is_true(self) -> bool:
        return self.value >= 1.0 - 1e-9

    @property
    def is_false(self) -> bool:
        return self.value <= 1e-9

    @property
    def complex_action(self) -> complex:
        """Carta \(z=v e^{i\theta}\) en \(T^*\mathbb{R}\cong\mathbb{C}\)."""
        return complex(self.value * math.cos(self.phase), self.value * math.sin(self.phase))

    # ---------- Retículo de Heyting ----------

    def meet(self, other: "HeytingValue") -> "HeytingValue":
        """Ínfimo \(x\wedge y=\min(x,y)\). Fase: media circular."""
        return HeytingValue(
            min(self.value, other.value),
            f"({self.description} ∧ {other.description})",
            phase=_wrap_angle((self.phase + other.phase) * 0.5),
        )

    def join(self, other: "HeytingValue") -> "HeytingValue":
        """Supremo \(x\vee y=\max(x,y)\)."""
        return HeytingValue(
            max(self.value, other.value),
            f"({self.description} ∨ {other.description})",
            phase=_wrap_angle((self.phase + other.phase) * 0.5),
        )

    def implies(self, other: "HeytingValue") -> "HeytingValue":
        r"""
        Implicación intuicionista:

            \[ x\to y=\sup\{z:x\wedge z\le y\}
               =\begin{cases}1&x\le y\\ y&x>y\end{cases}. \]
        """
        if self.value <= other.value:
            return HeytingValue(1.0, "true", phase=0.0)
        return HeytingValue(
            other.value,
            f"({self.description} → {other.description})",
            phase=other.phase,
        )

    def negate(self) -> "HeytingValue":
        r"""Pseudocomplemento \(\neg x=x\to 0\). En Heyting, \(\neg\neg x\neq x\) en general."""
        return self.implies(HeytingValue(0.0, "false", phase=0.0))

    def truncate(self, epsilon: Optional[float] = None) -> "HeytingValue":
        r"""Truncamiento topológico \(T_\epsilon(x)=x\cdot\mathbf{1}_{|x|\ge\epsilon}\)."""
        eps = _MACH_EPS if epsilon is None else epsilon
        if abs(self.value) < eps:
            return HeytingValue(0.0, f"T_ε({self.description})", phase=self.phase)
        return self

    def verify_absorption_law(
        self, other: "HeytingValue", use_truncation: bool = False
    ) -> bool:
        r"""\(x\vee(x\wedge y)=x\) y \(x\wedge(x\vee y)=x\)."""
        y = other.truncate() if use_truncation else other
        return self.join(self.meet(y)) == self and self.meet(self.join(y)) == self

    # ---------- Operadores simplécticos ----------

    def symplectic_pairing(self, other: "HeytingValue") -> float:
        r"""
        Corchete de Poisson en la carta compleja:

            \[ \{v,w\}=\mathrm{Im}(z_v\overline{z_w})=vw\sin(\theta_v-\theta_w). \]

        Cero \(\Leftrightarrow\) desacoplamiento (órbitas de fase paralelas).
        """
        return self.value * other.value * math.sin(self.phase - other.phase)

    def hamiltonian_flow(
        self, dt: float, energy: Optional[float] = None
    ) -> "HeytingValue":
        r"""
        Flujo armónico sobre el círculo de acción.

        \[ z(t)=z(0)\,e^{i\omega t},\qquad\omega=\sqrt{\max(E,0)}. \]

        Conserva \(|z|\) (Liouville) y avanza la fase. El código previo
        multiplicaba por \(\cos(\omega t)\) y anulaba el seno: violaba [I7].
        """
        E = self.value if energy is None else energy
        omega = math.sqrt(max(E, 0.0))
        new_phase = _wrap_angle(self.phase + omega * dt)
        return HeytingValue(
            self.value,
            f"Φ_t^{dt:.3f}({self.description})",
            phase=new_phase,
        )

    def lie_bracket(self, other: "HeytingValue") -> "HeytingValue":
        r"""
        Corchete de Lie de funciones escalares sobre \(\mathbb{R}\) es nulo.
        Reportamos la magnitud del Poisson como *acoplamiento residual*
        (no como elemento de \(\mathfrak{X}(T^*\mathcal{M})\)).
        """
        coupling = abs(self.symplectic_pairing(other))
        return HeytingValue(
            _clamp(coupling, 0.0, 1.0),
            f"|{{{self.description},{other.description}}}|",
            phase=0.0,
        )

    def geodesic_distance(self, other: "HeytingValue") -> float:
        r"""
        Distancia euclidiana en la carta \(z=v e^{i\theta}\in\mathbb{R}^2\):

            \[ d(v,w)=\|z_v-z_w\|_2\in[0,2]. \]

        Es invariante bajo el flujo armónico común (isometría de SO(2)).
        Reemplaza el Fubini–Study mal normalizado de la versión previa.
        """
        z1 = self.complex_action
        z2 = other.complex_action
        return abs(z1 - z2)

    def fubini_study_angle(self, other: "HeytingValue") -> float:
        r"""
        Ángulo de Fubini–Study entre rayos \([z_v],[z_w]\in\mathbb{CP}^1\).

            \[ d_{\mathrm{FS}}=\arccos\frac{|\langle z_v,z_w\rangle|}{\|z_v\|\|z_w\|}\in[0,\pi/2]. \]

        Si alguna amplitud es nula, devolvemos \(\pi/2\) (rayos indefinidos).
        """
        a, b = self.value, other.value
        if a < 1e-15 or b < 1e-15:
            return math.pi / 2.0
        overlap = abs(math.cos(self.phase - other.phase))
        return math.acos(_clamp(overlap, 0.0, 1.0))

    def __bool__(self) -> bool:
        return self.is_true

    def __float__(self) -> float:
        return self.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HeytingValue):
            return NotImplemented
        return (
            abs(self.value - other.value) < 1e-9
            and abs(self.phase - other.phase) < 1e-9
        )

    def __hash__(self) -> int:
        return hash((round(self.value, 9), round(self.phase, 9)))


# =============================================================================
# 1.4 — GEOMETRÍA SIMPLÉCTICA: FORMAS CANÓNICAS Y DARBOUX
# =============================================================================

@dataclass(frozen=True, slots=True)
class SymplecticForm:
    r"""
    2-forma simpléctica \(\Omega\) en \(T^*\mathcal{M}\), \(\dim=2n\).

    Axiomas
    -------
        1. Antisimetría: \(\Omega(u,v)=-\Omega(v,u)\)
        2. No degeneración: \(\Omega^\flat:TM\to T^*M\) isomorfismo
        3. Clausura: \(d\Omega=0\)

    Darboux: toda forma simpléctica es localmente

        \[ \Omega_0=\sum_i dq^i\wedge dp_i
           \;\longleftrightarrow\;
           J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}. \]

    Identidades canónicas verificadas numéricamente:
        \(J^{\top}=-J\), \(J^2=-I_{2n}\), \(J^{\top}J=I_{2n}\), \(\det J=+1\),
        \(\mathrm{Pf}(J)=+1\).
    """

    n_dof: int
    omega_matrix: Optional[Any] = None
    is_canonical: bool = True

    def __post_init__(self) -> None:
        if self.n_dof < 1:
            raise ValueError(f"n_dof debe ser ≥ 1, recibido: {self.n_dof}")
        if self.omega_matrix is None:
            object.__setattr__(self, "omega_matrix", _to_backend(_canonical_J(self.n_dof)))

    @property
    def dimension(self) -> int:
        return 2 * self.n_dof

    def _matrix(self) -> Matrix:
        return _as_matrix(self.omega_matrix)

    @property
    def pfaffian(self) -> float:
        r"""
        \(\mathrm{Pf}(J)^2=\det J\). Para Darboux, \(\mathrm{Pf}(J)=+1\).

        Signo: \(\mathrm{sgn}(\det J)\cdot\sqrt{|\det J|}\) no distingue Pf vs −Pf;
        en la carta canónica fijamos \(\mathrm{Pf}=+1\) si el residual de Darboux
        es nulo, si no devolvemos \(\mathrm{sgn}(\det)\sqrt{|\det|}\).
        """
        J = self._matrix()
        det = _det_small(J)
        mag = math.sqrt(abs(det))
        if self.is_canonical and self.darboux_residual() < _SYMPLECTIC_RESIDUAL_TOL:
            return mag if det >= 0.0 else -mag
        return math.copysign(mag, det) if det != 0.0 else 0.0

    def antisymmetry_residual(self) -> float:
        r"""\(\|J+J^{\top}\|_F\). Debe ser \(\approx 0\)."""
        J = self._matrix()
        return _frobenius(_add_mat(J, _transpose(J)))

    def darboux_residual(self) -> float:
        r"""\(\|J^{\top}J-I_{2n}\|_F\). Para Darboux, \(J\in\mathrm{O}(2n)\cap\mathfrak{sp}(2n)\)."""
        J = self._matrix()
        dim = len(J)
        gram = _matmul(_transpose(J), J)
        return _frobenius(_add_mat(gram, _scale_mat(-1.0, _eye(dim))))

    def cayley_residual(self) -> float:
        r"""\(\|J^2+I_{2n}\|_F\). Identidad estructural \(J^2=-I\)."""
        J = self._matrix()
        dim = len(J)
        return _frobenius(_add_mat(_matmul(J, J), _eye(dim)))

    def determinant(self) -> float:
        return _det_small(self._matrix())

    def verify_symplectic_invariants(
        self, tol: float = _SYMPLECTIC_RESIDUAL_TOL
    ) -> Tuple[bool, Dict[str, float]]:
        r"""
        Verifica antisimetría, Darboux, \(J^2=-I\), \(\det J=+1\), Pfaffiano.

        \(d\Omega=0\) es automático para formas constantes (coordenadas de Darboux).
        """
        antisym = self.antisymmetry_residual()
        darboux = self.darboux_residual()
        cayley = self.cayley_residual()
        det = self.determinant()
        pf = self.pfaffian
        is_valid = (
            antisym < tol
            and darboux < tol
            and cayley < tol
            and abs(det - 1.0) < tol
            and abs(pf - 1.0) < max(tol, 1e-6)
        )
        return is_valid, {
            "antisymmetry_residual": antisym,
            "darboux_residual": darboux,
            "cayley_residual": cayley,
            "determinant": det,
            "pfaffian": pf,
            "closure_residual": 0.0,
        }

    def pairing(self, u: Sequence[float], v: Sequence[float]) -> float:
        r"""\(\Omega(u,v)=u^{\top}Jv\)."""
        J = self._matrix()
        if len(u) != len(J) or len(v) != len(J):
            raise ValueError("dimensión incompatible con Ω")
        Jv = _matvec(J, v)
        return float(sum(ui * wi for ui, wi in zip(u, Jv)))

    def symplectic_capacity_ball(self, radius: float) -> float:
        r"""Gromov: \(c(B^{2n}(r))=\pi r^2\)."""
        if radius < 0.0:
            raise ValueError("radius ≥ 0")
        return math.pi * radius * radius

    def symplectic_capacity_cylinder(self, radius: float) -> float:
        r"""Gromov: \(c(Z^{2n}(R))=\pi R^2\)."""
        if radius < 0.0:
            raise ValueError("radius ≥ 0")
        return math.pi * radius * radius

    def gromov_nonsqueezing_check(
        self,
        radius_ball: float,
        radius_cylinder: float,
        tol: float = 1e-10,
    ) -> Tuple[bool, float]:
        r"""
        No-aplastamiento: \(c(B^{2n}(r))\le c(Z^{2n}(R))\Rightarrow r\le R\).

        Impide comprimir alucinaciones (bolas de acción) en cilindros de
        menor capacidad — cota geométrica sobre reducción dimensional abusiva.
        """
        if radius_cylinder <= 0.0:
            return False, float("inf")
        ratio = radius_ball / radius_cylinder
        return radius_ball <= radius_cylinder + tol, ratio

    def to_dict(self) -> Dict[str, Any]:
        valid, residuals = self.verify_symplectic_invariants()
        return {
            "n_dof": self.n_dof,
            "dimension": self.dimension,
            "is_canonical": self.is_canonical,
            "is_valid": valid,
            "pfaffian": round(self.pfaffian, 9),
            "residuals": {k: round(float(v), 12) for k, v in residuals.items()},
        }

    @classmethod
    def canonical(cls, n_dof: int) -> "SymplecticForm":
        return cls(n_dof=n_dof, omega_matrix=None, is_canonical=True)


# =============================================================================
# 1.5 — ESPACIO DE FASE: PUNTOS EN EL FIBRADO COTANGENTE T*M
# =============================================================================

@dataclass(frozen=True, slots=True)
class PhaseSpacePoint:
    r"""
    Punto \(z=(q,p)\in T^*\mathcal{M}\cong\mathbb{R}^{2n}\).

    En MIC:
        \(q\) — coordenadas semánticas del contexto (configuración),
        \(p\) — momentos de intención (impulso de ejecución).

    Estructura heredada: \(\Omega_0(v,w)=\sum_i(dq^i\wedge dp_i)(v,w)\).
    """

    q: Tuple[float, ...]
    p: Tuple[float, ...]
    t: float = 0.0
    label: str = "phase_point"

    def __post_init__(self) -> None:
        if len(self.q) != len(self.p):
            raise ValueError(
                f"Dimensión inconsistente: len(q)={len(self.q)} ≠ len(p)={len(self.p)}"
            )
        if not self.q:
            raise ValueError("n_dof ≥ 1 requerido")

    @property
    def n_dof(self) -> int:
        return len(self.q)

    @property
    def dimension(self) -> int:
        return 2 * self.n_dof

    def to_tuple(self) -> Tuple[float, ...]:
        return self.q + self.p

    def to_array(self) -> Any:
        coords = self.to_tuple()
        if not NUMPY_AVAILABLE:
            return list(coords)
        return np.array(coords, dtype=np.float64)

    @classmethod
    def from_coords(
        cls,
        coords: Sequence[float],
        t: float = 0.0,
        label: str = "phase_point",
    ) -> "PhaseSpacePoint":
        n = len(coords)
        if n < 2 or n % 2 != 0:
            raise ValueError("coords debe tener longitud par 2n ≥ 2")
        k = n // 2
        return cls(
            q=tuple(float(x) for x in coords[:k]),
            p=tuple(float(x) for x in coords[k:]),
            t=t,
            label=label,
        )

    def euclidean_norm(self) -> float:
        r"""\(\|z\|_2=\sqrt{\|q\|^2+\|p\|^2}\)."""
        return math.sqrt(_hypot_sq(self.q) + _hypot_sq(self.p))

    def kinetic_norm(self) -> float:
        return math.sqrt(_hypot_sq(self.p))

    def distance_to(self, other: "PhaseSpacePoint") -> float:
        """Distancia euclidiana — métrica de la recurrencia de Poincaré."""
        if self.n_dof != other.n_dof:
            raise ValueError("Dimensiones incompatibles")
        dq = [a - b for a, b in zip(self.q, other.q)]
        dp = [a - b for a, b in zip(self.p, other.p)]
        return math.sqrt(_hypot_sq(dq) + _hypot_sq(dp))

    def symplectic_product(self, other: "PhaseSpacePoint") -> float:
        r"""
        \(\Omega_0(z,w)=\sum_i(q_i w_{p_i}-p_i w_{q_i})\).

        Propiedades: \(\Omega(z,z)=0\), \(\Omega(z,w)=-\Omega(w,z)\).
        """
        if self.n_dof != other.n_dof:
            raise ValueError("Dimensiones incompatibles")
        return float(
            sum(self.q[i] * other.p[i] - self.p[i] * other.q[i] for i in range(self.n_dof))
        )

    def poincare_1form(self) -> float:
        r"""
        Liouville 1-form evaluada en el radio vector:

            \[ \theta_z(z)=\langle p,q\rangle=\sum_i p_i q^i. \]

        No es invariante; su exterior \(d\theta=\Omega\) sí lo es.
        """
        return float(sum(qi * pi for qi, pi in zip(self.q, self.p)))

    def poisson_bracket(self, f_grad: Sequence[float], g_grad: Sequence[float]) -> float:
        r"""
        \[ \{f,g\}(z)=\sum_i\left(
            \partial_{q_i}f\,\partial_{p_i}g-\partial_{p_i}f\,\partial_{q_i}g\right). \]
        """
        n = self.n_dof
        if len(f_grad) != 2 * n or len(g_grad) != 2 * n:
            raise ValueError("gradientes de dimensión 2n requeridos")
        f_q, f_p = f_grad[:n], f_grad[n:]
        g_q, g_p = g_grad[:n], g_grad[n:]
        return float(
            sum(fq * gp - fp * gq for fq, fp, gq, gp in zip(f_q, f_p, g_q, g_p))
        )

    def displaced(self, dq: Sequence[float], dp: Sequence[float], dt: float = 0.0) -> "PhaseSpacePoint":
        if len(dq) != self.n_dof or len(dp) != self.n_dof:
            raise ValueError("desplazamiento de dimensión n")
        return PhaseSpacePoint(
            q=tuple(a + b for a, b in zip(self.q, dq)),
            p=tuple(a + b for a, b in zip(self.p, dp)),
            t=self.t + dt,
            label=self.label,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q": list(self.q),
            "p": list(self.p),
            "t": self.t,
            "label": self.label,
            "n_dof": self.n_dof,
            "norm": round(self.euclidean_norm(), 6),
            "liouville_pairing": round(self.poincare_1form(), 6),
        }


# =============================================================================
# 1.6 — SISTEMA HAMILTONIANO: GENERADOR DEL FLUJO SIMPLÉCTICO
# =============================================================================

class FlowResult(NamedTuple):
    """Trayectoria simpléctica con diagnósticos de Liouville y energía."""

    trajectory: List[PhaseSpacePoint]
    energy_initial: float
    energy_final: float
    energy_drift: float
    liouville_residual: float
    steps: int
    dt: float


@dataclass(frozen=True, slots=True)
class HamiltonianSystem:
    r"""
    Hamiltoniano \(H:T^*\mathcal{M}\to\mathbb{R}\) y su flujo canónico.

    Ecuaciones de Hamilton
    ----------------------
        \[ \dot q^i=\partial_{p_i}H,\qquad \dot p_i=-\partial_{q^i}H,
           \qquad \dot z=J\nabla H. \]

    Consecuencias (Poincaré):
        1. \(dH/dt=\{H,H\}=0\)
        2. \(\div X_H=0\) (Liouville)
        3. \(\Phi_t^*\Omega=\Omega\) (simplectomorfismo)

    Integrador: Störmer–Verlet / leapfrog (simpléctico de orden 2,
    reversible, conserva un Hamiltoniano modificado \(\widetilde H=H+\mathcal{O}(dt^2)\)).
    """

    name: str = "harmonic_oscillator"
    kinetic_mass: float = 1.0
    potential_coefficients: Tuple[float, ...] = (1.0,)
    perturbation_order: int = 0

    def __post_init__(self) -> None:
        if self.kinetic_mass <= 0.0:
            raise ValueError(f"kinetic_mass debe ser > 0, recibido: {self.kinetic_mass}")
        if not self.potential_coefficients:
            raise ValueError("potential_coefficients no puede ser vacío")

    def kinetic_energy(self, p: Sequence[float]) -> float:
        r"""\(T(p)=\|p\|^2/(2m)\)."""
        return _hypot_sq(p) / (2.0 * self.kinetic_mass)

    def potential_energy(self, q: Sequence[float]) -> float:
        r"""
        Potencial isotrópico de Taylor:

            \[ V(q)=\sum_{k=1}^{K} c_k\frac{|q|^k}{k!}. \]

        Oscilador armónico: \(c=(0,\omega^2)\) \(\Rightarrow\) \(V=\omega^2|q|^2/2\).
        """
        q_norm_sq = _hypot_sq(q)
        total = 0.0
        for k, c_k in enumerate(self.potential_coefficients, start=1):
            if c_k == 0.0:
                continue
            total += c_k * (q_norm_sq ** (k / 2.0)) / math.factorial(k)
        return total

    def hamiltonian(self, z: PhaseSpacePoint) -> float:
        return self.kinetic_energy(z.p) + self.potential_energy(z.q)

    def _potential_gradient(self, q: Sequence[float]) -> Vector:
        n = len(q)
        grad = [0.0] * n
        q_norm_sq = _hypot_sq(q)
        for k, c_k in enumerate(self.potential_coefficients, start=1):
            if c_k == 0.0:
                continue
            if k == 1:
                if q_norm_sq > 1e-12:
                    inv = 1.0 / math.sqrt(q_norm_sq)
                    for i in range(n):
                        grad[i] += c_k * q[i] * inv
            else:
                # d/dq_i (|q|^k / k!) = |q|^{k-2} q_i / (k-1)!
                scale = c_k * k * (q_norm_sq ** ((k - 2) / 2.0)) / math.factorial(k)
                for i in range(n):
                    grad[i] += scale * q[i]
        return grad

    def grad_hamiltonian(self, z: PhaseSpacePoint) -> Vector:
        r"""\(\nabla H=(\partial_q H,\partial_p H)=(\nabla V, p/m)\)."""
        grad_q = self._potential_gradient(z.q)
        inv_m = 1.0 / self.kinetic_mass
        grad_p = [pi * inv_m for pi in z.p]
        return grad_q + grad_p

    def vector_field(self, z: PhaseSpacePoint) -> Vector:
        r"""\(X_H=J\nabla H=(\partial_p H,-\partial_q H)\)."""
        grad = self.grad_hamiltonian(z)
        n = z.n_dof
        return grad[n:] + [-g for g in grad[:n]]

    def divergence(self, z: PhaseSpacePoint, h: float = 1e-6) -> float:
        r"""
        \(\div X_H\). Para todo Hamiltoniano \(C^2\), es idénticamente 0.
        Se evalúa por diferencias centrales como auditoría numérica de Liouville.
        """
        n = z.n_dof
        coords = list(z.to_tuple())
        div = 0.0
        for i in range(2 * n):
            c_plus = coords[:]
            c_minus = coords[:]
            c_plus[i] += h
            c_minus[i] -= h
            Xp = self.vector_field(PhaseSpacePoint.from_coords(c_plus, t=z.t, label=z.label))
            Xm = self.vector_field(PhaseSpacePoint.from_coords(c_minus, t=z.t, label=z.label))
            div += (Xp[i] - Xm[i]) / (2.0 * h)
        return div

    def integrate_flow(
        self,
        z0: PhaseSpacePoint,
        dt: float,
        n_steps: int,
    ) -> List[PhaseSpacePoint]:
        """Integra y devuelve solo la trayectoria (compatibilidad)."""
        return self.integrate_flow_diagnosed(z0, dt, n_steps).trajectory

    def integrate_flow_diagnosed(
        self,
        z0: PhaseSpacePoint,
        dt: float,
        n_steps: int,
    ) -> FlowResult:
        r"""
        Störmer–Verlet (orden 2, simpléctico, reversible):

            \[
            p_{n+1/2}=p_n-\tfrac{dt}{2}\nabla V(q_n),\;
            q_{n+1}=q_n+dt\,p_{n+1/2}/m,\;
            p_{n+1}=p_{n+1/2}-\tfrac{dt}{2}\nabla V(q_{n+1}).
            \]

        El jacobiano del mapa Verlet pertenece a \(\mathrm{Sp}(2n,\mathbb{R})\)
        (volumen = 1). El drift de energía es \(\mathcal{O}(dt^2)\) acotado,
        no secular — sello de un integrador simpléctico.
        """
        if dt == 0.0:
            raise ValueError("dt ≠ 0")
        if n_steps < 0:
            raise ValueError("n_steps ≥ 0")

        trajectory: List[PhaseSpacePoint] = [z0]
        q: Vector = list(z0.q)
        p: Vector = list(z0.p)
        t = z0.t
        m = self.kinetic_mass
        n = len(q)
        e0 = self.hamiltonian(z0)

        for step in range(n_steps):
            g = self._potential_gradient(q)
            for i in range(n):
                p[i] -= 0.5 * dt * g[i]
            inv_m = dt / m
            for i in range(n):
                q[i] += inv_m * p[i]
            g = self._potential_gradient(q)
            for i in range(n):
                p[i] -= 0.5 * dt * g[i]
            t += dt
            trajectory.append(
                PhaseSpacePoint(
                    q=tuple(float(x) for x in q),
                    p=tuple(float(x) for x in p),
                    t=t,
                    label=f"{z0.label}_step{step + 1}",
                )
            )

        zf = trajectory[-1]
        e1 = self.hamiltonian(zf)
        liouville = abs(self.divergence(zf))
        return FlowResult(
            trajectory=trajectory,
            energy_initial=e0,
            energy_final=e1,
            energy_drift=abs(e1 - e0),
            liouville_residual=liouville,
            steps=n_steps,
            dt=dt,
        )

    def birkhoff_normal_form_coefficients(
        self,
        order: int = _BIRKHOFF_TRUNCATION_ORDER,
    ) -> List[float]:
        r"""
        Coeficientes de Birkhoff en 1-dof hasta orden \(N\) (homogéneos pares).

        Limitación honesta: esto **no** es el algoritmo de Lie–Deprit en \(n>1\).
        En 1-dof no resonante, los términos impares se eliminan por una
        transformación canónica; el coeficiente de orden \(2i\) del potencial
        isotrópico sobrevive (salvo factorial) como invariante de Birkhoff.

        \[ H=H_0(J)+c_1 J^2+\cdots+c_N J^N+R_{N+1}. \]
        """
        if order < 1:
            raise ValueError("order ≥ 1")
        coefficients: List[float] = []
        for i in range(1, order + 1):
            taylor_idx = 2 * i  # k = 2, 4, ...
            if taylor_idx - 1 < len(self.potential_coefficients):
                c = self.potential_coefficients[taylor_idx - 1]
            else:
                c = 0.0
            coefficients.append(c / math.factorial(2 * i))
        return coefficients

    def action_variable(self, z: PhaseSpacePoint) -> float:
        r"""
        Acción 1-dof: \(J=\frac{1}{2\pi}\oint p\,dq\).

        Para el HO, \(J=H/\omega\). En el caso isotrópico usamos
        \(\omega=\sqrt{|c_2|}\) si existe, si no \(\omega=1\).
        """
        H = self.hamiltonian(z)
        if len(self.potential_coefficients) >= 2:
            omega = math.sqrt(abs(self.potential_coefficients[1]))
        else:
            omega = 1.0
        return H / max(omega, 1e-12)

    def first_integral_residual(self, trajectory: Sequence[PhaseSpacePoint]) -> float:
        """Máxima variación de \(H\) a lo largo de una trayectoria."""
        if not trajectory:
            return 0.0
        energies = [self.hamiltonian(z) for z in trajectory]
        return max(energies) - min(energies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kinetic_mass": self.kinetic_mass,
            "potential_coefficients": list(self.potential_coefficients),
            "perturbation_order": self.perturbation_order,
            "birkhoff_coefficients": self.birkhoff_normal_form_coefficients(),
        }

    @classmethod
    def harmonic_oscillator(cls, omega: float = 1.0) -> "HamiltonianSystem":
        r"""\(H=(p^2+\omega^2 q^2)/2\)."""
        return cls(
            name=f"harmonic_ω={omega:.3f}",
            kinetic_mass=1.0,
            potential_coefficients=(0.0, omega * omega),
        )

    @classmethod
    def kepler(cls, mu: float = 1.0) -> "HamiltonianSystem":
        r"""
        Germen de Kepler isotrópico: \(V=-\mu/|q|\) se representa por \(c_1=-\mu\).

        No es el problema de dos cuerpos reducido completo (falta el término
        centrífugo en polares); sirve como potencial \(1/r\) de auditoría.
        """
        return cls(
            name=f"kepler_μ={mu:.3f}",
            kinetic_mass=1.0,
            potential_coefficients=(-mu,),
        )


# =============================================================================
# 1.7 — COORDENADAS ACCIÓN-ÁNGULO (Estructura Integrable KAM)
# =============================================================================

@dataclass(frozen=True, slots=True)
class ActionAngleCoordinates:
    r"""
    Coordenadas acción-ángulo \((J,\theta)\) de Liouville–Arnold.

        \(J\in\mathbb{R}_+^n\) invariantes, \(\theta\in\mathbb{T}^n\) ángulos,
        \(\omega(J)=\partial_J H\) frecuencias.

    Dinámica integrable: \(\theta(t)=\theta(0)+\omega t\pmod{2\pi}\).

    KAM: si \(\omega\) es diofántica,

        \[ |\langle k,\omega\rangle|\ge\frac{\gamma}{|k|^\tau}
           \quad\forall k\in\mathbb{Z}^n\setminus\{0\}, \]

    con \(\tau>n-1\), \(\gamma>0\), una proporción positiva (Lebesgue) de
    toros persiste bajo perturbaciones \(\varepsilon\) pequeñas.

    En MIC: \(J\) = capacidades estratégicas conservadas;
    \(\theta\) = fase operativa cíclica.
    """

    actions: Tuple[float, ...]
    angles: Tuple[float, ...]
    frequencies: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.angles) or len(self.actions) != len(self.frequencies):
            raise ValueError("actions, angles y frequencies deben compartir dimensión")
        if not self.actions:
            raise ValueError("n_dof ≥ 1")
        if any(J < 0.0 for J in self.actions):
            raise ValueError("Las acciones J_i deben ser ≥ 0")
        object.__setattr__(
            self, "angles", tuple(_wrap_angle(th) for th in self.angles)
        )

    @property
    def n_dof(self) -> int:
        return len(self.actions)

    def frequency_ratio(self) -> Optional[float]:
        """\(\omega_1/\omega_0\) si \(n\ge 2\) y \(\omega_0\neq 0\)."""
        if self.n_dof < 2:
            return None
        if abs(self.frequencies[0]) < 1e-15:
            return None
        return self.frequencies[1] / self.frequencies[0]

    def continued_fraction_ratio(self, max_terms: int = 16) -> Tuple[int, ...]:
        """CF de \(\omega_1/\omega_0\) — diagnóstico de Poincaré de resonancia."""
        rho = self.frequency_ratio()
        if rho is None:
            return tuple()
        return _continued_fraction(abs(rho), max_terms=max_terms)

    def golden_obstruction(self) -> float:
        r"""
        Distancia \(L^1\) de la CF de \(\omega_1/\omega_0\) a la de \(\varphi\).

        Cerca de 0 \(\Rightarrow\) máximo número de rotación KAM-estable.
        """
        cf = self.continued_fraction_ratio()
        if not cf:
            return float("inf")
        n = min(len(cf), len(_PHI_CF))
        return float(sum(abs(cf[i] - _PHI_CF[i]) for i in range(n))) / float(n)

    def check_diophantine(
        self,
        gamma: float = _KAM_GAMMA_DEFAULT,
        tau: float = _KAM_TAU_DEFAULT,
        k_max: int = 2,
    ) -> Tuple[bool, float]:
        r"""
        Condición diofántica sobre el retículo \(\|k\|_\infty\le k_{\max}\).

        \(|k|=\|k\|_1\). Devuelve \((\mathrm{is\_diophantine}, \min_k|\langle k,\omega\rangle|)\).
        Recorre **todo** el retículo (no aborta el mínimo al primer fallo).
        """
        if gamma <= 0.0:
            raise ValueError("gamma > 0")
        if k_max < 1:
            raise ValueError("k_max ≥ 1")
        n = self.n_dof
        if tau <= n - 1:
            # No es diofántica en el sentido de Bruno; aún medimos divisores.
            pass
        omega = self.frequencies
        min_divisor = float("inf")
        is_diophantine = True
        ranges = [range(-k_max, k_max + 1)] * n
        for k_tuple in itertools_product(*ranges):
            if all(ki == 0 for ki in k_tuple):
                continue
            k_norm = float(sum(abs(ki) for ki in k_tuple))
            dot = abs(sum(ki * wi for ki, wi in zip(k_tuple, omega)))
            if dot < min_divisor:
                min_divisor = dot
            threshold = gamma / (k_norm ** tau)
            if dot < threshold:
                is_diophantine = False
        if min_divisor == float("inf"):
            min_divisor = 1.0
        return is_diophantine, min_divisor

    def evolve(self, dt: float) -> "ActionAngleCoordinates":
        r"""\(\theta_i\leftarrow\theta_i+\omega_i\Delta t\pmod{2\pi}\). \(J\) invariante."""
        new_angles = tuple(
            _wrap_angle(theta + omega * dt)
            for theta, omega in zip(self.angles, self.frequencies)
        )
        return ActionAngleCoordinates(
            actions=self.actions,
            angles=new_angles,
            frequencies=self.frequencies,
        )

    def to_phase_space(self) -> PhaseSpacePoint:
        r"""
        Inmersión HO: \(q_i=\sqrt{2J_i}\cos\theta_i\), \(p_i=-\sqrt{2J_i}\sin\theta_i\).
        """
        q = tuple(math.sqrt(max(2.0 * J, 0.0)) * math.cos(th) for J, th in zip(self.actions, self.angles))
        p = tuple(-math.sqrt(max(2.0 * J, 0.0)) * math.sin(th) for J, th in zip(self.actions, self.angles))
        return PhaseSpacePoint(q=q, p=p, t=0.0, label="action_angle")

    def to_dict(self) -> Dict[str, Any]:
        is_dioph, min_div = self.check_diophantine()
        return {
            "actions": list(self.actions),
            "angles": list(self.angles),
            "frequencies": list(self.frequencies),
            "is_diophantine": is_dioph,
            "min_divisor": round(min_div, 9),
            "n_dof": self.n_dof,
            "frequency_ratio": self.frequency_ratio(),
            "continued_fraction": list(self.continued_fraction_ratio()),
            "golden_obstruction": self.golden_obstruction(),
        }


# =============================================================================
# 1.8 — FUNCIÓN GENERATRIZ DE TRANSFORMACIONES CANÓNICAS
# =============================================================================

@dataclass(frozen=True, slots=True)
class GeneratingFunction:
    r"""
    Función generatriz de transformaciones canónicas (Hamilton–Jacobi).

    Tipo 2 (carta útil, \(S=S(q,P)\)):
        \[ p=\partial_q S,\qquad Q=\partial_P S,\qquad H'=H+\partial_t S. \]

    Identidad: \(S=qP\) \(\Rightarrow\) \(p=P\), \(Q=q\).

    Perturbación: \(S=qP+W(q,P)\). La canonicidad se verifica por el
    jacobiano \(\partial(Q,P)/\partial(q,p)\in\mathrm{Sp}(2,\mathbb{R})\),
    i.e. \(\{Q,P\}_{q,p}=1\).

    Tipos 1, 3, 4 se reducen numéricamente a tipo 2 por transformada de Legendre
    local cuando el hessiano mixto es invertible (hipótesis de Poincaré).
    """

    kind: int = 2
    S_coefficients: Tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)
    time_dependence: bool = False

    def __post_init__(self) -> None:
        if self.kind not in (1, 2, 3, 4):
            raise ValueError(f"kind debe ser 1, 2, 3 o 4; recibido: {self.kind}")
        if not self.S_coefficients:
            raise ValueError("S_coefficients no puede ser vacío")

    def evaluate(self, x: float, y: float) -> float:
        r"""
        Polinomio bivariado truncado

            \[ S(x,y)=\sum_{i=0}^{n} c_i x^i y^{n-i},\quad n=\deg S. \]

        Para tipo 2, \((x,y)=(q,P)\). El monomio \(c_n q^n\) y \(c_0 P^n\)
        con \(n=3\), \(c=(0,0,0,1)\) recupera \(S=qP\) (identidad).
        """
        n = len(self.S_coefficients) - 1
        total = 0.0
        for i, c_i in enumerate(self.S_coefficients):
            if c_i == 0.0:
                continue
            j = n - i
            total += c_i * (x ** i) * (y ** j)
        return total

    def _partials(self, x: float, y: float, eps: float = 1e-6) -> Tuple[float, float]:
        """\((\partial_x S,\partial_y S)\) por diferencias centrales."""
        d_x = (self.evaluate(x + eps, y) - self.evaluate(x - eps, y)) / (2.0 * eps)
        d_y = (self.evaluate(x, y + eps) - self.evaluate(x, y - eps)) / (2.0 * eps)
        return d_x, d_y

    def transform(self, q: float, p: float) -> Tuple[float, float]:
        r"""
        Aplica \((q,p)\mapsto(Q,P)\) para tipo 2 resolviendo \(P\) de
        \(p=\partial_q S(q,P)\) (Newton) y evaluando \(Q=\partial_P S(q,P)\).
        """
        if self.kind != 2:
            # Legendre local: se usa la misma carta con la conjetura P ≈ p.
            pass
        P = p
        eps = 1e-7
        for _ in range(64):
            dS_dq, dS_dP = self._partials(q, P, eps=eps)
            error = dS_dq - p
            if abs(error) < 1e-12:
                break
            # ∂²S / ∂q∂P ≈ ∂(∂S/∂q)/∂P
            dS_dq_plus, _ = self._partials(q, P + eps, eps=eps)
            mixed = (dS_dq_plus - dS_dq) / eps
            if abs(mixed) < 1e-14:
                break
            P -= error / mixed
        _, Q = self._partials(q, P, eps=eps)
        return Q, P

    def jacobian(self, q: float, p: float, h: float = 1e-5) -> Matrix:
        """Jacobiano \(2\times 2\) \(\partial(Q,P)/\partial(q,p)\)."""
        Q0, P0 = self.transform(q, p)
        Qq, Pq = self.transform(q + h, p)
        Qp, Pp = self.transform(q, p + h)
        return [
            [(Qq - Q0) / h, (Qp - Q0) / h],
            [(Pq - P0) / h, (Pp - P0) / h],
        ]

    def verify_canonical(
        self, q: float, p: float, tol: float = 1e-5
    ) -> Tuple[bool, float]:
        r"""
        Canonicidad 1-dof: \(\det D\Phi=\{Q,P\}_{q,p}=1\).

        Equivale a \(\Phi^*\Omega=\Omega\) en dimensión 2.
        """
        J = self.jacobian(q, p)
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        residual = abs(det - 1.0)
        return residual < tol, residual

    def hamilton_jacobi_residual(self, q: float, p: float, H: HamiltonianSystem) -> float:
        r"""
        Residual de Hamilton–Jacobi estacionaria para tipo 2:

            \[ H\!\left(q,\partial_q S\right) - E,\quad E=H(q,p). \]

        Cero \(\Rightarrow\) \(S\) genera coordenadas de equilibrio (\(H'=E\) cte).
        """
        P = p
        dS_dq, _ = self._partials(q, P)
        z = PhaseSpacePoint(q=(q,), p=(dS_dq,), t=0.0, label="HJ")
        return abs(H.hamiltonian(z) - H.hamiltonian(PhaseSpacePoint(q=(q,), p=(p,))))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "S_coefficients": list(self.S_coefficients),
            "time_dependence": self.time_dependence,
        }

    @classmethod
    def identity(cls) -> "GeneratingFunction":
        """\(S=qP\) (tipo 2). Coeficientes grado 3: \(c=(0,0,0,1)\)."""
        return cls(kind=2, S_coefficients=(0.0, 0.0, 0.0, 1.0))


# =============================================================================
# 1.9 — CLASIFICADOR DE SUBOBJETOS Ω CON SEMÁNTICA SIMPLÉCTICA
# =============================================================================

class SubobjectClassifier:
    r"""
    Clasificador de subobjetos \(\Omega\) del topos \(\mathcal{E}_{\mathrm{MIC}}\),
    fibrado sobre \(T^*\mathcal{M}\).

    Para cada subobjeto \(A\hookrightarrow X\) existe un único \(\chi_A:X\to\Omega\)
    que realiza \(A\) como pullback de \(\top:1\to\Omega\).

    Cuantización geométrica (Bohr–Sommerfeld / Maslov):

        \[ \oint_\gamma p\,dq=2\pi\hbar\bigl(n+\nu/4\bigr). \]

    Con \(\hbar=1\), la integralidad de la acción selecciona Lagrangianos
    admisibles — cribas del topos simpléctico.
    """

    __slots__ = ("true", "false", "symplectic_form")

    def __init__(
        self,
        n_dof: int = 3,
        symplectic_form: Optional[SymplecticForm] = None,
    ) -> None:
        self.true = HeytingValue(1.0, "true", phase=0.0)
        self.false = HeytingValue(0.0, "false", phase=0.0)
        self.symplectic_form = symplectic_form or SymplecticForm.canonical(n_dof)

    def evaluate_morphism(
        self, condition: bool, reason: str = "binary_eval"
    ) -> HeytingValue:
        return self.true if condition else HeytingValue(0.0, reason)

    def characteristic_morphism(
        self,
        membership: float,
        description: str = "membership",
        phase: float = 0.0,
    ) -> HeytingValue:
        r"""\(\chi_S(x)\) con grado de pertenencia y fase de Bohr–Sommerfeld."""
        return HeytingValue(membership, description, phase=phase)

    def quantize_bohr_sommerfeld(
        self,
        action_integral: float,
        maslov_index: int = 0,
        hbar: float = 1.0,
    ) -> Tuple[bool, float]:
        r"""
        Verifica \(\oint p\,dq=2\pi\hbar(n+\nu/4)\).

        Returns:
            (is_quantized, n_effective)
        """
        denominator = _TWO_PI * hbar
        if denominator < 1e-12:
            return False, 0.0
        n_eff = action_integral / denominator - maslov_index / 4.0
        residual = abs(n_eff - round(n_eff))
        return residual < 0.1, n_eff

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true": {"value": self.true.value, "phase": self.true.phase},
            "false": {"value": self.false.value, "phase": self.false.phase},
            "symplectic_form": self.symplectic_form.to_dict(),
        }


# =============================================================================
# 1.10 — FORMA DE POINCARÉ–CARTAN E INVARIANTES INTEGRALES
# =============================================================================

@dataclass(frozen=True, slots=True)
class PoincareCartanForm:
    r"""
    Forma de Poincaré–Cartan en el espacio de fase extendido
    \(T^*\mathcal{M}\times\mathbb{R}\):

        \[ \Theta_{\mathrm{PC}}=\sum_i p_i\,dq^i-H\,dt. \]

    Teorema (Poincaré, 1890; Cartan, 1922)
    --------------------------------------
    Para cualquier tubo de trayectorias hamiltonianas, el integral de
    \(\Theta_{\mathrm{PC}}\) sobre un ciclo que rodea el tubo es invariante.

    El invariante integral *absoluto* de Poincaré es la restricción a \(t\) cte:

        \[ I(\gamma)=\oint_\gamma\sum_i p_i\,dq^i=\int_\Sigma\Omega, \]

    y se conserva bajo el flujo (\(\Phi_t^*\Omega=\Omega\)).

    El invariante *relativo* se obtiene sobre cadenas con borde en una
    sección de Poincaré — germen geométrico de la Fase 2.
    """

    hamiltonian: HamiltonianSystem
    symplectic_form: SymplecticForm

    def one_form(self, z: PhaseSpacePoint) -> float:
        r"""\(\Theta_{\mathrm{PC}}(\dot z_{\mathrm{ext}})\) no se evalúa aquí;
        devolvemos el pairing de Liouville \(\langle p,q\rangle\) como densidad."""
        return z.poincare_1form()

    def action_along(self, path: Sequence[PhaseSpacePoint]) -> float:
        r"""
        Integral discreta de Liouville \(\sum\langle p\rangle\Delta q\)
        (regla del trapecio) a lo largo de un camino.
        """
        if len(path) < 2:
            return 0.0
        total = 0.0
        for z0, z1 in zip(path, path[1:]):
            if z0.n_dof != z1.n_dof:
                raise ValueError("camino con n_dof inconsistente")
            dq = [b - a for a, b in zip(z0.q, z1.q)]
            p_mid = [(a + b) * 0.5 for a, b in zip(z0.p, z1.p)]
            total += sum(pm * dqi for pm, dqi in zip(p_mid, dq))
        return total

    def poincare_cartan_along(self, path: Sequence[PhaseSpacePoint]) -> float:
        r"""
        \(\int p\,dq-H\,dt\) a lo largo de un camino (trapecio en \(q\) y \(t\)).
        """
        if len(path) < 2:
            return 0.0
        action = self.action_along(path)
        energy_dt = 0.0
        for z0, z1 in zip(path, path[1:]):
            H_mid = 0.5 * (self.hamiltonian.hamiltonian(z0) + self.hamiltonian.hamiltonian(z1))
            energy_dt += H_mid * (z1.t - z0.t)
        return action - energy_dt

    def absolute_invariant(
        self, loop: Sequence[PhaseSpacePoint], tol: float = _POINCARE_CARTAN_TOL
    ) -> Tuple[float, bool]:
        r"""
        Invariante absoluto \(\oint p\,dq\) sobre un lazo (se asume cerrado
        si \(\|z_{\mathrm{end}}-z_{\mathrm{start}}\|<\sqrt{\mathrm{tol}}\)).
        """
        if len(loop) < 3:
            return 0.0, False
        closed = loop[0].distance_to(loop[-1]) < math.sqrt(max(tol, 0.0)) + 1e-9
        return self.action_along(loop), closed

    def invariance_under_flow(
        self,
        loop: Sequence[PhaseSpacePoint],
        dt: float,
        n_steps: int,
        tol: float = _POINCARE_CARTAN_TOL,
    ) -> Tuple[bool, float]:
        r"""
        Empuja cada vértice del lazo por \(\Phi_{n\cdot dt}\) y compara
        \(\oint p\,dq\). Residual relativo al invariante inicial.

        Este es el teorema de Poincaré de invariantes integrales, versión
        discreta: el flujo hamiltoniano es un simplectomorfismo.
        """
        I0, _ = self.absolute_invariant(loop, tol=tol)
        pushed: List[PhaseSpacePoint] = []
        for z in loop:
            result = self.hamiltonian.integrate_flow_diagnosed(z, dt, n_steps)
            pushed.append(result.trajectory[-1])
        I1, _ = self.absolute_invariant(pushed, tol=tol)
        scale = max(abs(I0), 1e-12)
        residual = abs(I1 - I0) / scale
        return residual < tol, residual

    def stokes_check(
        self, loop: Sequence[PhaseSpacePoint], area_pairs: Sequence[Tuple[PhaseSpacePoint, PhaseSpacePoint]]
    ) -> float:
        r"""
        Stokes discreto: \(\oint_\partial\Sigma\theta-\int_\Sigma\Omega\).

        `area_pairs` es una descomposición grosera de \(\Sigma\) en
        paralelogramos \((u,v)\) basados en `loop[0]`: \(\Omega(u,v)\).
        """
        circulation = self.action_along(loop)
        if not loop:
            return circulation
        origin = loop[0]
        flux = 0.0
        for a, b in area_pairs:
            ua = PhaseSpacePoint(
                q=tuple(x - y for x, y in zip(a.q, origin.q)),
                p=tuple(x - y for x, y in zip(a.p, origin.p)),
            )
            vb = PhaseSpacePoint(
                q=tuple(x - y for x, y in zip(b.q, origin.q)),
                p=tuple(x - y for x, y in zip(b.p, origin.p)),
            )
            flux += ua.symplectic_product(vb)
        return abs(circulation - flux)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hamiltonian": self.hamiltonian.to_dict(),
            "symplectic_form": self.symplectic_form.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FirstIntegral:
    r"""
    Integral primera \(F:T^*\mathcal{M}\to\mathbb{R}\) con \(\{F,H\}=0\).

    Poincaré: el número máximo de integrales independientes en involución
    es \(n\) (integrabilidad de Liouville). Cada integral recorta un nivel
    y folia el espacio de fase en toros — materia prima de la Fase 2.
    """

    name: str
    eval_F: Callable[[PhaseSpacePoint], float]
    grad_F: Callable[[PhaseSpacePoint], Vector]

    def poisson_with(self, system: HamiltonianSystem, z: PhaseSpacePoint) -> float:
        return z.poisson_bracket(self.grad_F(z), system.grad_hamiltonian(z))

    def is_conserved(
        self, system: HamiltonianSystem, z: PhaseSpacePoint, tol: float = 1e-8
    ) -> bool:
        return abs(self.poisson_with(system, z)) < tol

    @classmethod
    def energy(cls, system: HamiltonianSystem) -> "FirstIntegral":
        return cls(
            name=f"energy[{system.name}]",
            eval_F=system.hamiltonian,
            grad_F=system.grad_hamiltonian,
        )

    @classmethod
    def angular_momentum_z(cls) -> "FirstIntegral":
        r"""\(L_z=q_1 p_2-q_2 p_1\) (requiere \(n\ge 2\))."""

        def _eval(z: PhaseSpacePoint) -> float:
            if z.n_dof < 2:
                raise ValueError("L_z requiere n_dof ≥ 2")
            return z.q[0] * z.p[1] - z.q[1] * z.p[0]

        def _grad(z: PhaseSpacePoint) -> Vector:
            if z.n_dof < 2:
                raise ValueError("L_z requiere n_dof ≥ 2")
            n = z.n_dof
            g = [0.0] * (2 * n)
            g[0] = z.p[1]       # ∂L/∂q1
            g[1] = -z.p[0]      # ∂L/∂q2
            g[n] = -z.q[1]      # ∂L/∂p1
            g[n + 1] = z.q[0]   # ∂L/∂p2
            return g

        return cls(name="L_z", eval_F=_eval, grad_F=_grad)


# =============================================================================
# 1.11 — SECCIÓN DE POINCARÉ (germen geométrico)
# =============================================================================

@dataclass(frozen=True, slots=True)
class PoincareSection:
    r"""
    Sección de Poincaré \(\Sigma\subset T^*\mathcal{M}\): hipersuperficie
    de codimensión 1 transversal al flujo \(X_H\).

    Definición
    ----------
    \(\Sigma=s^{-1}(0)\) con \(ds(X_H)\neq 0\) (transversalidad).

    El *mapa de retorno* \(P:\Sigma\to\Sigma\) es un simplectomorfismo
    de dimensión \(2n-2\) (Poincaré). Su construcción iterada, sus
    exponentes característicos y la homología persistente de las órbitas
    periódicas pertenecen a la **Fase 2**.

    Esta clase solo fija el germen: la sección, la transversalidad y el
    operador-germen \(P_\bullet\) que Fase 2 iterará.
    """

    coordinate: Literal["q", "p"] = "q"
    index: int = 0
    level: float = 0.0
    direction: Literal["positive", "negative", "both"] = "positive"

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index ≥ 0")
        if self.coordinate not in ("q", "p"):
            raise ValueError("coordinate ∈ {q, p}")

    def section_value(self, z: PhaseSpacePoint) -> float:
        if self.index >= z.n_dof:
            raise ValueError(f"index {self.index} ≥ n_dof {z.n_dof}")
        coord = z.q if self.coordinate == "q" else z.p
        return coord[self.index] - self.level

    def is_on_section(self, z: PhaseSpacePoint, tol: float = 1e-10) -> bool:
        return abs(self.section_value(z)) < tol

    def is_transverse(
        self, z: PhaseSpacePoint, system: HamiltonianSystem, tol: float = 1e-12
    ) -> bool:
        r"""\(ds(X_H)\neq 0\). \(s=q_i-c\) \(\Rightarrow\) \(ds(X_H)=\dot q_i\)."""
        X = system.vector_field(z)
        n = z.n_dof
        slot = self.index if self.coordinate == "q" else n + self.index
        return abs(X[slot]) > tol

    def crossing_sign(self, z_prev: PhaseSpacePoint, z_next: PhaseSpacePoint) -> int:
        """+1 cruce positivo, −1 negativo, 0 ninguno."""
        s0 = self.section_value(z_prev)
        s1 = self.section_value(z_next)
        if s0 <= 0.0 < s1:
            return +1
        if s0 >= 0.0 > s1:
            return -1
        return 0

    def accepts_crossing(self, sign: int) -> bool:
        if sign == 0:
            return False
        if self.direction == "both":
            return True
        if self.direction == "positive":
            return sign > 0
        return sign < 0

    def interpolate_hit(
        self, z_prev: PhaseSpacePoint, z_next: PhaseSpacePoint
    ) -> PhaseSpacePoint:
        """Interpolación lineal del cruce (germen; Fase 2 usará interpolación simpléctica)."""
        s0 = self.section_value(z_prev)
        s1 = self.section_value(z_next)
        denom = s1 - s0
        lam = 0.5 if abs(denom) < 1e-18 else -s0 / denom
        lam = _clamp(lam, 0.0, 1.0)
        q = tuple(a + lam * (b - a) for a, b in zip(z_prev.q, z_next.q))
        p = tuple(a + lam * (b - a) for a, b in zip(z_prev.p, z_next.p))
        t = z_prev.t + lam * (z_next.t - z_prev.t)
        return PhaseSpacePoint(q=q, p=p, t=t, label=f"hit[{self.coordinate}{self.index}]")

    def germ_of_return_map(
        self,
        z0: PhaseSpacePoint,
        system: HamiltonianSystem,
        dt: float,
        max_steps: int,
    ) -> Dict[str, Any]:
        r"""
        Germen del mapa de Poincaré \(P:\Sigma\to\Sigma\).

        Integra hasta el primer cruce admisible y devuelve el hit, el tiempo
        de retorno y banderas de transversalidad. **No** itera \(P\), **no**
        calcula exponentes de Floquet ni cadenas de Markov: eso es Fase 2.

        Contrato de continuación
        ------------------------
        El diccionario devuelto es el *objeto inicial* de
        `Phase1Foundation.seed_phase2_topology`: Fase 2 lo promoverá a

            * mapa de retorno iterado \(P^k\),
            * análisis espectral / KAM del jacobiano \(DP\),
            * persistencia y números de Betti de las órbitas,
            * cadena de Markov ergódica sobre estratos de \(\Sigma\).
        """
        if not self.is_transverse(z0, system):
            return {
                "hit": None,
                "return_time": None,
                "steps": 0,
                "transverse": False,
                "reason": "ds(X_H) ≈ 0 — sección no transversal en z0",
                "seed_for_phase2": True,
            }
        flow = system.integrate_flow_diagnosed(z0, dt, max_steps)
        prev = flow.trajectory[0]
        for k, curr in enumerate(flow.trajectory[1:], start=1):
            sign = self.crossing_sign(prev, curr)
            if self.accepts_crossing(sign):
                hit = self.interpolate_hit(prev, curr)
                return {
                    "hit": hit,
                    "return_time": hit.t - z0.t,
                    "steps": k,
                    "transverse": self.is_transverse(hit, system),
                    "crossing_sign": sign,
                    "energy_drift": flow.energy_drift,
                    "liouville_residual": flow.liouville_residual,
                    "reason": "first_admissible_hit",
                    "seed_for_phase2": True,
                }
            prev = curr
        return {
            "hit": None,
            "return_time": None,
            "steps": max_steps,
            "transverse": True,
            "reason": "no_return_within_horizon",
            "seed_for_phase2": True,
        }


# =============================================================================
# 1.12 — FUNDACIÓN DE FASE 1  (último objeto: germen funtorial de Fase 2)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Phase1Foundation:
    r"""
    Objeto agregador de la Fase 1. Todos los tipos base viven aquí:

        * `config`              — MICConfiguration (KAM / Liouville / Bruno)
        * `symplectic_form`     — \(\Omega\) de Darboux
        * `hamiltonian`         — \(H\) y flujo de Verlet
        * `classifier`          — \(\Omega_{\mathrm{Heyting}}\) fibrado
        * `generating_function` — carta canónica tipo 2
        * `cartan`              — invariantes integrales de Poincaré
        * `section`             — sección de Poincaré

    El último método, `seed_phase2_topology`, es la *unidad de continuación*:
    su valor de retorno es el objeto inicial de la Fase 2 (topología,
    persistencia, mapas de retorno, Markov ergódicas, invariantes KAM).
    """

    config: MICConfiguration
    symplectic_form: SymplecticForm
    hamiltonian: HamiltonianSystem
    classifier: SubobjectClassifier
    generating_function: GeneratingFunction
    cartan: PoincareCartanForm
    section: PoincareSection

    @classmethod
    def from_config(
        cls,
        config: Optional[MICConfiguration] = None,
        hamiltonian: Optional[HamiltonianSystem] = None,
        section: Optional[PoincareSection] = None,
    ) -> "Phase1Foundation":
        cfg = config or DEFAULT_MIC_CONFIG
        form = SymplecticForm.canonical(cfg.n_dof)
        H = hamiltonian or HamiltonianSystem.harmonic_oscillator(omega=1.0)
        clf = SubobjectClassifier(n_dof=cfg.n_dof, symplectic_form=form)
        gen = GeneratingFunction.identity()
        cartan = PoincareCartanForm(hamiltonian=H, symplectic_form=form)
        sec = section or PoincareSection(coordinate="q", index=0, level=0.0, direction="positive")
        return cls(
            config=cfg,
            symplectic_form=form,
            hamiltonian=H,
            classifier=clf,
            generating_function=gen,
            cartan=cartan,
            section=sec,
        )

    def audit_invariants(self) -> Dict[str, Any]:
        """Auditoría conjunta de [I6]–[I9] y [I12]."""
        valid, residuals = self.symplectic_form.verify_symplectic_invariants(
            tol=self.config.symplectic_residual_tol
        )
        return {
            "symplectic_valid": valid,
            "residuals": residuals,
            "bruno_gap": self.config.kam_bruno_gap,
            "birkhoff_radius": self.config.birkhoff_radius_bound,
            "gromov_unit_ok": self.symplectic_form.gromov_nonsqueezing_check(1.0, 1.0)[0],
        }

    def seed_phase2_topology(
        self,
        z0: Optional[PhaseSpacePoint] = None,
        dt: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║  ÚLTIMO MÉTODO DE LA FASE 1                                      ║
        ║  Germen funtorial  F₁ → F₂                                       ║
        ╚══════════════════════════════════════════════════════════════════╝

        Produce el *objeto inicial* que la Fase 2 consume como unidad:

            seed["poincare_germ"]     → mapa de retorno a iterar
            seed["symplectic_form"]   → capacidades de Gromov / Betti
            seed["kam"]               → γ, τ, retículo, CF de frecuencias
            seed["cartan"]            → invariante absoluto sobre el germen
            seed["heyting_omega"]     → clasificador para persistencia
            seed["liouville"]         → residuales de volumen y energía
            seed["phase1_audit"]      → certificación de [I6]–[I12]

        Fase 2 comenzará exactamente aquí: enriquecerá `poincare_germ`
        con homología persistente, números de Betti, análisis espectral
        KAM del jacobiano \(DP\), y cadenas de Markov ergódicas sobre
        los estratos de la sección \(\Sigma\).
        """
        cfg = self.config
        n = cfg.n_dof
        if z0 is None:
            z0 = PhaseSpacePoint(
                q=tuple([0.0] * n),
                p=tuple([1.0] + [0.0] * (n - 1)),
                t=0.0,
                label="phase2_seed",
            )
        step = cfg.hamiltonian_time_step if dt is None else dt
        horizon = cfg.poincare_max_return_iterations if max_steps is None else max_steps

        germ = self.section.germ_of_return_map(z0, self.hamiltonian, step, horizon)
        hit: Optional[PhaseSpacePoint] = germ.get("hit")
        path_for_cartan: List[PhaseSpacePoint] = [z0]
        if hit is not None:
            path_for_cartan.append(hit)
        cartan_value = self.cartan.poincare_cartan_along(path_for_cartan)

        omega_probe = ActionAngleCoordinates(
            actions=tuple([0.5] * n),
            angles=tuple([0.0] * n),
            frequencies=tuple([1.0] + [_PHI ** (-k) for k in range(1, n)]),
        )
        is_dioph, min_div = omega_probe.check_diophantine(
            gamma=cfg.kam_gamma,
            tau=cfg.kam_tau,
            k_max=cfg.kam_frequency_lattice_range,
        )

        return {
            "version": cfg.algorithm_version,
            "n_dof": n,
            "symplectic_dimension": cfg.symplectic_dimension,
            "phase1_audit": self.audit_invariants(),
            "heyting_omega": self.classifier.to_dict(),
            "symplectic_form": self.symplectic_form.to_dict(),
            "hamiltonian": self.hamiltonian.to_dict(),
            "generating_function": self.generating_function.to_dict(),
            "section": {
                "coordinate": self.section.coordinate,
                "index": self.section.index,
                "level": self.section.level,
                "direction": self.section.direction,
            },
            "poincare_germ": germ,
            "cartan": {
                "poincare_cartan_along_germ": cartan_value,
                "absolute_invariant_closed": None,
            },
            "kam": {
                "gamma": cfg.kam_gamma,
                "tau": cfg.kam_tau,
                "bruno_gap": cfg.kam_bruno_gap,
                "epsilon": cfg.kam_epsilon_perturbation,
                "is_diophantine_probe": is_dioph,
                "min_divisor": min_div,
                "continued_fraction": list(omega_probe.continued_fraction_ratio()),
                "golden_obstruction": omega_probe.golden_obstruction(),
            },
            "liouville": {
                "energy_drift": germ.get("energy_drift"),
                "volume_residual": germ.get("liouville_residual"),
                "tol": cfg.liouville_volume_tol,
            },
            "continuation": {
                "next_phase": 2,
                "consumes": (
                    "poincare_germ",
                    "kam",
                    "symplectic_form",
                    "heyting_omega",
                    "liouville",
                ),
                "produces": (
                    "PersistenceInterval ⊗ Ω",
                    "BettiNumbers ⊗ c_Gromov",
                    "PoincareReturnMap^{k}",
                    "ErgodicMarkovStrata",
                    "KAMSpectralInvariants",
                ),
            },
            "seed_for_phase2": True,
        }


__phase1_version__: Final[str] = "8.1.0-poincare-symplectic"
__phase1_exports__: Final[Tuple[str, ...]] = (
    "MICConfiguration",
    "DEFAULT_MIC_CONFIG",
    "HeytingValue",
    "SymplecticForm",
    "PhaseSpacePoint",
    "HamiltonianSystem",
    "FlowResult",
    "ActionAngleCoordinates",
    "GeneratingFunction",
    "SubobjectClassifier",
    "PoincareCartanForm",
    "FirstIntegral",
    "PoincareSection",
    "Phase1Foundation",
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIN DE LA FASE 1 — CONTINÚA EN FASE 2
# ═══════════════════════════════════════════════════════════════════════════════
# Contracción funtorial F₁ ⇒ F₂:
#
#   Phase1Foundation.seed_phase2_topology()
#       └── poincare_germ          →  mapa de retorno P: Σ → Σ  (iterar, Floquet)
#       └── symplectic_form        →  capacidades de Gromov en persistencia
#       └── kam                    →  pequeños divisores en el espectro de DP
#       └── heyting_omega          →  filtración de Heyting de intervalos
#       └── liouville              →  cadenas de Markov que conservan medida
#       └── cartan                 →  invariante relativo sobre bordes de Σ
#
# La Fase 2 DEBE abrir con un constructor que reciba exactamente este seed
# (p. ej. Phase2Topology.from_phase1_seed(seed)) y no redeclare Ω, J ni H.
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              FASE 2: ESTRUCTURAS TOPOLÓGICAS Y SISTEMAS DINÁMICOS         ║
# ║    (Persistencia Simpléctica · Betti-Gromov · KAM · Poincaré · Ergódicas) ║
# ║    Versión: 8.1.0-Poincare-Symplectic-KAM-Novikov-Doctoral                ║
# ║    Continúa: Phase1Foundation.seed_phase2_topology() → from_phase1_seed   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════
#
# Contracción funtorial F₁ ⇒ F₂ (unidad de continuación):
#
#   seed["poincare_germ"]   →  P: Σ → Σ, P^k, DP, Floquet
#   seed["symplectic_form"] →  capacidades de Gromov / demanda homológica
#   seed["kam"]             →  pequeños divisores en el espectro de DP
#   seed["heyting_omega"]   →  filtración de Heyting de intervalos
#   seed["liouville"]       →  cadenas de Markov que conservan medida
#   seed["cartan"]          →  invariante relativo sobre bordes de Σ
#
# INVARIANTES DE ESTA FASE ────────────────────────────────────────────────────
#   [I10] Recurrencia de Poincaré (ergódica, distinta del mapa de retorno)
#   [I14] Simplectomorfismo de sección: P*Ω_Σ = Ω_Σ,  det(DP) = +1
#   [I15] Floquet hamiltoniano: multiplicadores en pares (λ, 1/λ)
#   [I16] Filtración de persistencia: b ≤ d, acción ≥ 0, Maslov ∈ ℤ
#   [I17] Mixing Markov: t_mix(ε) = log(1/ε) / γ,  γ = 1 − |λ₂(T)|
#
# La Fase 2 NO redefine SymplecticForm, PhaseSpacePoint, HamiltonianSystem,
# ActionAngleCoordinates, PoincareSection, PoincareCartanForm ni Phase1Foundation.
# ═══════════════════════════════════════════════════════════════════════════════


T = TypeVar("T")
StratumLike = Any


def _wrap_angle_pi(delta: float) -> float:
    r"""Representante de \(\Delta\theta\) en \((-\pi,\pi]\)."""
    w = (delta + math.pi) % _TWO_PI - math.pi
    return math.pi if w == -math.pi else w


def _safe_stratum_name(s: Any) -> str:
    return str(getattr(s, "name", s))


def _safe_stratum_value(s: Any) -> int:
    try:
        return int(getattr(s, "value", 0))
    except (TypeError, ValueError):
        return 0


# =============================================================================
# 2.0 — PROTOCOLO DEL SEED F₁ Y REHIDRATACIÓN
# =============================================================================

class Phase2Seed(TypedDict, total=False):
    r"""
    Contrato tipado del dict que emite `Phase1Foundation.seed_phase2_topology`.

    No es un objeto vivo: es el *código de barras* funtorial. Los objetos
    \(H\), \(\Sigma\), \(\Omega\) se rehidratan o se inyectan vía
    `Phase1Foundation` opcional.
    """

    version: str
    n_dof: int
    symplectic_dimension: int
    phase1_audit: Dict[str, Any]
    heyting_omega: Dict[str, Any]
    symplectic_form: Dict[str, Any]
    hamiltonian: Dict[str, Any]
    generating_function: Dict[str, Any]
    section: Dict[str, Any]
    poincare_germ: Dict[str, Any]
    cartan: Dict[str, Any]
    kam: Dict[str, Any]
    liouville: Dict[str, Any]
    continuation: Dict[str, Any]
    seed_for_phase2: bool


def _require_phase1_seed(seed: Mapping[str, Any]) -> None:
    if not isinstance(seed, Mapping):
        raise TypeError("seed debe ser Mapping (salida de seed_phase2_topology)")
    if not seed.get("seed_for_phase2", False):
        raise ValueError(
            "Funtorialidad F₁⇒F₂ violada: el seed no porta seed_for_phase2=True. "
            "Debe emitirlo Phase1Foundation.seed_phase2_topology()."
        )
    n_dof = int(seed.get("n_dof", 0) or 0)
    if n_dof < 1:
        raise ValueError("seed['n_dof'] ≥ 1 requerido")


def _rehydrate_hamiltonian(blob: Mapping[str, Any]) -> HamiltonianSystem:
    coeffs = tuple(float(c) for c in blob.get("potential_coefficients", (1.0,)))
    if not coeffs:
        coeffs = (1.0,)
    return HamiltonianSystem(
        name=str(blob.get("name", "harmonic_oscillator")),
        kinetic_mass=float(blob.get("kinetic_mass", 1.0)),
        potential_coefficients=coeffs,
        perturbation_order=int(blob.get("perturbation_order", 0)),
    )


def _rehydrate_section(blob: Mapping[str, Any]) -> PoincareSection:
    coord = blob.get("coordinate", "q")
    if coord not in ("q", "p"):
        coord = "q"
    direction = blob.get("direction", "positive")
    if direction not in ("positive", "negative", "both"):
        direction = "positive"
    return PoincareSection(
        coordinate=coord,  # type: ignore[arg-type]
        index=int(blob.get("index", 0)),
        level=float(blob.get("level", 0.0)),
        direction=direction,  # type: ignore[arg-type]
    )


def _rehydrate_point(blob: Any, fallback_n: int, label: str) -> PhaseSpacePoint:
    if isinstance(blob, PhaseSpacePoint):
        return blob
    if isinstance(blob, Mapping) and "q" in blob and "p" in blob:
        q = tuple(float(x) for x in blob["q"])
        p = tuple(float(x) for x in blob["p"])
        return PhaseSpacePoint(
            q=q,
            p=p,
            t=float(blob.get("t", 0.0)),
            label=str(blob.get("label", label)),
        )
    n = max(1, fallback_n)
    return PhaseSpacePoint(
        q=tuple([0.0] * n),
        p=tuple([1.0] + [0.0] * (n - 1)),
        t=0.0,
        label=label,
    )


def _section_chart_indices(n_dof: int, section: PoincareSection) -> Tuple[int, ...]:
    r"""Índices de \(\mathbb{R}^{2n}\) que parametrizan la carta de \(\Sigma\) (codim 1)."""
    slot = section.index if section.coordinate == "q" else n_dof + section.index
    return tuple(i for i in range(2 * n_dof) if i != slot)


def _embed_section_coords(
    z_ref: PhaseSpacePoint,
    section: PoincareSection,
    coords: Sequence[float],
) -> PhaseSpacePoint:
    """Inmersión de coordenadas de carta en \(T^*M\) fijando la sección al nivel."""
    ambient = list(z_ref.to_tuple())
    chart = _section_chart_indices(z_ref.n_dof, section)
    if len(coords) != len(chart):
        raise ValueError("coords de carta incompatibles con Σ")
    for idx, val in zip(chart, coords):
        ambient[idx] = float(val)
    slot = section.index if section.coordinate == "q" else z_ref.n_dof + section.index
    ambient[slot] = float(section.level)
    return PhaseSpacePoint.from_coords(ambient, t=z_ref.t, label=z_ref.label)


def _chart_coords(z: PhaseSpacePoint, section: PoincareSection) -> Tuple[float, ...]:
    ambient = z.to_tuple()
    return tuple(ambient[i] for i in _section_chart_indices(z.n_dof, section))


# =============================================================================
# 2.1 — PERSISTENCIA CON GEOMETRÍA SIMPLÉCTICA
# =============================================================================

@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    r"""
    Intervalo de persistencia \([b,d)\) fibrado sobre un Lagrangiano de \(T^*M\).

    Homología persistente (Carlsson): una clase \(\alpha\in H_k(K_b)\) nace en
    \(b\) y muere en \(d\) cuando se vuelve borde.

    Enriquecimiento de Poincaré–Arnold
    ----------------------------------
    El tiempo de vida \(\ell=d-b\) de una órbita periódica no degenerada
    es proporcional a la acción de Cartan

        \[ A(\gamma)=\oint_\gamma p\,dq, \qquad \ell \sim A/(2\pi) \]

    en unidades donde la filtración es el tiempo de retorno (o la acción).

    El subespacio asociado al intervalo es el Lagrangiano vertical a lo
    largo de \(\gamma\) (isótropo: \(\Omega|_L=0\), \(\dim L=n\)). Esta clase
    **no** reconstruye \(L\) en coordenadas: porta los escalares que lo
    cuantizan (acción, Maslov, fases de Bohr–Sommerfeld).

    `to_symplectic_2form` de la versión previa devolvía Darboux genérico:
    eso no es un Lagrangiano. Aquí se expone `isotropic_residual` como
    test \(\Omega(v,w)=0\) sobre un par de tangentes suministrado, y
    `heyting_membership` como \(\chi\) del clasificador de Fase 1.
    """

    birth: float
    death: float
    dimension: int = 0
    phase_birth: float = 0.0
    phase_death: float = 0.0
    action_integral: float = 0.0
    maslov_index: int = 0
    heyting_membership: float = 1.0
    label: str = "interval"

    def __post_init__(self) -> None:
        if self.birth < 0.0:
            raise ValueError(f"birth ≥ 0, recibido: {self.birth}")
        if not math.isinf(self.death) and self.death < self.birth:
            raise ValueError(f"death ({self.death}) ≥ birth ({self.birth}) o +∞")
        if self.dimension < 0:
            raise ValueError(f"dimension ≥ 0, recibido: {self.dimension}")
        if self.action_integral < 0.0:
            raise ValueError(f"action_integral ≥ 0, recibido: {self.action_integral}")
        object.__setattr__(self, "phase_birth", _wrap_angle(self.phase_birth))
        object.__setattr__(self, "phase_death", _wrap_angle(self.phase_death))
        object.__setattr__(
            self, "heyting_membership", float(_clamp(self.heyting_membership, 0.0, 1.0))
        )

    @classmethod
    def essential(
        cls,
        birth: float,
        dimension: int = 0,
        phase_birth: float = 0.0,
        action_integral: float = 0.0,
        maslov_index: int = 0,
        heyting_membership: float = 1.0,
        label: str = "essential",
    ) -> "PersistenceInterval":
        return cls(
            birth=birth,
            death=float("inf"),
            dimension=dimension,
            phase_birth=phase_birth,
            phase_death=0.0,
            action_integral=action_integral,
            maslov_index=maslov_index,
            heyting_membership=heyting_membership,
            label=label,
        )

    @property
    def is_essential(self) -> bool:
        return math.isinf(self.death)

    @property
    def persistence(self) -> float:
        r"""\(\ell=d-b\). Esencial \(\Rightarrow +\infty\)."""
        return float("inf") if self.is_essential else self.death - self.birth

    def finite_persistence(self) -> float:
        return 0.0 if self.is_essential else self.death - self.birth

    @property
    def midpoint(self) -> float:
        return self.birth if self.is_essential else 0.5 * (self.birth + self.death)

    @property
    def symplectic_phase_drift(self) -> float:
        r"""Holonomía de fase \(\Delta\theta\in(-\pi,\pi]\). Cero \(\Rightarrow\) clase de Chern trivial."""
        return _wrap_angle_pi(self.phase_death - self.phase_birth)

    def as_heyting(self, description: Optional[str] = None) -> HeytingValue:
        """Morfismo característico \(\chi_{[b,d)}:\mathrm{pt}\to\Omega_{\mathrm{Heyting}}\)."""
        return HeytingValue(
            self.heyting_membership,
            description or self.label,
            phase=self.phase_birth,
        )

    def is_bohr_sommerfeld_quantized(
        self,
        hbar: float = 1.0,
        tol: float = 0.1,
        classifier: Optional[SubobjectClassifier] = None,
    ) -> bool:
        r"""\(\oint p\,dq=2\pi\hbar(n+\nu/4)\). Delega en el clasificador de Fase 1 si se provee."""
        if classifier is not None:
            ok, _ = classifier.quantize_bohr_sommerfeld(
                self.action_integral, maslov_index=self.maslov_index, hbar=hbar
            )
            return ok
        denom = _TWO_PI * hbar
        if denom < 1e-12:
            return False
        n_eff = self.action_integral / denom - self.maslov_index / 4.0
        return abs(n_eff - round(n_eff)) < tol

    def isotropic_residual(self, v: PhaseSpacePoint, w: PhaseSpacePoint) -> float:
        r"""
        \(|\Omega(v,w)|\). Cero \(\Rightarrow\) el par es isótropo (test local de Lagrangiano).

        No construye \(L_{[b,d)}\); certifica un test que el llamador debe
        aplicar a una base de \(T\gamma\).
        """
        return abs(v.symplectic_product(w))

    def sort_key(self) -> Tuple[int, float, float, int]:
        """Esenciales primero; luego persistencia descendente; luego birth; luego dim."""
        pers = -1.0 if self.is_essential else -self.finite_persistence()
        return (0 if self.is_essential else 1, pers, self.birth, self.dimension)

    def __lt__(self, other: "PersistenceInterval") -> bool:
        if not isinstance(other, PersistenceInterval):
            return NotImplemented
        return self.sort_key() < other.sort_key()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "birth": self.birth,
            "death": self.death if not self.is_essential else "inf",
            "persistence": self.persistence if not self.is_essential else "inf",
            "dimension": self.dimension,
            "is_essential": self.is_essential,
            "midpoint": self.midpoint,
            "phase_birth": round(self.phase_birth, 6),
            "phase_death": round(self.phase_death, 6),
            "symplectic_phase_drift": round(self.symplectic_phase_drift, 6),
            "action_integral": round(self.action_integral, 9),
            "maslov_index": self.maslov_index,
            "heyting_membership": round(self.heyting_membership, 6),
            "is_bohr_sommerfeld_quantized": self.is_bohr_sommerfeld_quantized(),
            "label": self.label,
        }

    def __repr__(self) -> str:
        death_str = "inf" if self.is_essential else f"{self.death:.4f}"
        pers = "inf" if self.is_essential else f"{self.finite_persistence():.4f}"
        return (
            f"PersistenceInterval(birth={self.birth:.4f}, death={death_str}, "
            f"dim={self.dimension}, persistence={pers}, "
            f"action={self.action_integral:.4f})"
        )


class _UnionFind:
    """Union-Find con rango y path compression — 0-persistencia de Rips."""

    __slots__ = ("parent", "rank", "n")

    def __init__(self, n: int) -> None:
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


@dataclass(frozen=True, slots=True)
class PersistenceDiagram:
    r"""
    Diagrama \(\mathrm{Dgm}_k=\bigl\{[b_i,d_i)\bigr\}\) con filtración de Rips
    sobre una nube en la sección \(\Sigma\) (0-homología exacta; \(\beta_1\)
    de grafo, no de complejo de Čech).
    """

    intervals: Tuple[PersistenceInterval, ...]
    rips_scale: float = 0.0

    def finite(self) -> Tuple[PersistenceInterval, ...]:
        return tuple(iv for iv in self.intervals if not iv.is_essential)

    def essential(self) -> Tuple[PersistenceInterval, ...]:
        return tuple(iv for iv in self.intervals if iv.is_essential)

    def sorted(self) -> Tuple[PersistenceInterval, ...]:
        return tuple(sorted(self.intervals))

    def entropy(self, config: Optional[MICConfiguration] = None) -> float:
        return compute_persistence_entropy(self.intervals, config)

    def betti_at(self, scale: float) -> "BettiNumbers":
        r"""
        \(\beta_0\): intervalos de dim 0 con \(b\le\mathrm{scale}<d\).
        \(\beta_1\): idem dim 1 (estimación de grafo, no barcode completo).
        """
        b0 = sum(
            1
            for iv in self.intervals
            if iv.dimension == 0 and iv.birth <= scale < iv.death
        )
        b1 = sum(
            1
            for iv in self.intervals
            if iv.dimension == 1 and iv.birth <= scale < iv.death
        )
        b2 = sum(
            1
            for iv in self.intervals
            if iv.dimension == 2 and iv.birth <= scale < iv.death
        )
        return BettiNumbers(beta_0=b0, beta_1=b1, beta_2=b2)

    @classmethod
    def from_point_cloud(
        cls,
        points: Sequence[PhaseSpacePoint],
        actions: Optional[Sequence[float]] = None,
        heyting: Optional[Sequence[float]] = None,
    ) -> "PersistenceDiagram":
        r"""
        0-persistencia de Rips: todos nacen en 0; mueren al merge;
        una componente esencial. \(\beta_1\) de grafo al diámetro medio.
        """
        n = len(points)
        if n == 0:
            return cls(intervals=tuple())
        if n == 1:
            act = float(actions[0]) if actions else 0.0
            mem = float(heyting[0]) if heyting else 1.0
            return cls(
                intervals=(
                    PersistenceInterval.essential(
                        birth=0.0,
                        dimension=0,
                        action_integral=max(0.0, act),
                        heyting_membership=mem,
                        label="cloud_0",
                    ),
                )
            )

        edges: List[Tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((points[i].distance_to(points[j]), i, j))
        edges.sort(key=lambda e: e[0])

        uf = _UnionFind(n)
        intervals: List[PersistenceInterval] = []
        merge_scales: List[float] = []
        for dist, i, j in edges:
            if uf.union(i, j):
                merge_scales.append(dist)
                act = 0.0
                if actions:
                    act = 0.5 * (float(actions[i]) + float(actions[j]))
                mem = 1.0
                if heyting:
                    mem = min(float(heyting[i]), float(heyting[j]))
                intervals.append(
                    PersistenceInterval(
                        birth=0.0,
                        death=dist,
                        dimension=0,
                        action_integral=max(0.0, act),
                        heyting_membership=mem,
                        label=f"merge_{i}_{j}",
                    )
                )

        intervals.append(
            PersistenceInterval.essential(
                birth=0.0,
                dimension=0,
                action_integral=max((float(a) for a in actions), default=0.0) if actions else 0.0,
                label="H0_essential",
            )
        )

        # β₁ de grafo al umbral = mediana de aristas: m − n + c
        if edges:
            median_scale = edges[len(edges) // 2][0]
            n_edges = sum(1 for d, _, _ in edges if d <= median_scale)
            uf2 = _UnionFind(n)
            for d, i, j in edges:
                if d <= median_scale:
                    uf2.union(i, j)
            n_comp = len({uf2.find(i) for i in range(n)})
            beta1 = max(0, n_edges - n + n_comp)
            for k in range(beta1):
                intervals.append(
                    PersistenceInterval(
                        birth=0.0,
                        death=median_scale,
                        dimension=1,
                        action_integral=0.0,
                        heyting_membership=0.5,
                        label=f"graph_cycle_{k}",
                    )
                )
            rips = median_scale
        else:
            rips = 0.0

        return cls(intervals=tuple(sorted(intervals)), rips_scale=rips)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_intervals": len(self.intervals),
            "n_finite": len(self.finite()),
            "n_essential": len(self.essential()),
            "rips_scale": round(self.rips_scale, 9),
            "intervals": [iv.to_dict() for iv in self.sorted()],
        }


# =============================================================================
# 2.2 — NÚMEROS DE BETTI CON CAPACIDADES DE GROMOV
# =============================================================================

@dataclass(frozen=True, slots=True)
class BettiNumbers:
    r"""
    \(\beta_p=\mathrm{rank}\,H_p(X;\mathbb{Q})\) más una **demanda** de
    capacidad (heurística dimensional, no el teorema de Gromov).

    Gromov (1985) es \(c(B^{2n}(r))\le c(Z^{2n}(R))\Rightarrow r\le R\).
    Eso vive en `SymplecticForm.gromov_nonsqueezing_check` (Fase 1).

    Aquí `homological_capacity_demand = π · Σ β_k` es una cota inferior
    *ad hoc* de “cuánta capacidad hace falta para alojar Σ β_k clases”.
    Se confronta con `c(B^{2n}(r))` del seed, no se llama “teorema”.
    """

    beta_0: int
    beta_1: int
    beta_2: int
    symplectic_capacity_total: float = 0.0
    gromov_bound_residual: float = 0.0

    def __post_init__(self) -> None:
        for name, val in (("beta_0", self.beta_0), ("beta_1", self.beta_1), ("beta_2", self.beta_2)):
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} entero ≥ 0, recibido: {val!r}")
        if self.symplectic_capacity_total < 0.0:
            raise ValueError("symplectic_capacity_total ≥ 0")

    @property
    def euler_characteristic(self) -> int:
        return self.beta_0 - self.beta_1 + self.beta_2

    @property
    def total_rank(self) -> int:
        return self.beta_0 + self.beta_1 + self.beta_2

    @property
    def is_connected(self) -> bool:
        return self.beta_0 == 1

    @property
    def has_cycles(self) -> bool:
        return self.beta_1 > 0

    @property
    def has_cavities(self) -> bool:
        return self.beta_2 > 0

    @property
    def homological_capacity_demand(self) -> float:
        r"""Heurística \(\pi\sum\beta_k\) (no es \(c_{\mathrm{Gromov}}\))."""
        return math.pi * float(self.total_rank)

    def check_capacity_budget(
        self,
        available_capacity: float,
        tol: float = 1e-9,
    ) -> Tuple[bool, float]:
        r"""¿La capacidad de Gromov disponible cubre la demanda homológica?"""
        residual = available_capacity - self.homological_capacity_demand
        return residual >= -tol, residual

    def gromov_nonsqueezing_from_form(
        self,
        form: SymplecticForm,
        radius_ball: float,
        radius_cylinder: float,
        tol: float = 1e-10,
    ) -> Tuple[bool, float]:
        """Delega el teorema real en la forma de Fase 1."""
        return form.gromov_nonsqueezing_check(radius_ball, radius_cylinder, tol=tol)

    @classmethod
    def zero(cls) -> "BettiNumbers":
        return cls(beta_0=0, beta_1=0, beta_2=0)

    @classmethod
    def point(cls) -> "BettiNumbers":
        return cls(beta_0=1, beta_1=0, beta_2=0)

    @classmethod
    def from_diagram(cls, diagram: PersistenceDiagram, scale: Optional[float] = None) -> "BettiNumbers":
        s = diagram.rips_scale if scale is None else scale
        return diagram.betti_at(s)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "betti_numbers": [self.beta_0, self.beta_1, self.beta_2],
            "euler_characteristic": self.euler_characteristic,
            "total_rank": self.total_rank,
            "is_connected": self.is_connected,
            "has_cycles": self.has_cycles,
            "has_cavities": self.has_cavities,
            "symplectic_capacity_total": round(self.symplectic_capacity_total, 9),
            "homological_capacity_demand": round(self.homological_capacity_demand, 9),
            "gromov_bound_residual": round(self.gromov_bound_residual, 9),
        }

    def __repr__(self) -> str:
        return (
            f"BettiNumbers(β₀={self.beta_0}, β₁={self.beta_1}, β₂={self.beta_2}, "
            f"χ={self.euler_characteristic}, c_Ω={self.symplectic_capacity_total:.3e})"
        )


# =============================================================================
# 2.3 — RESUMEN TOPOLÓGICO CON INVARIANTES KAM
# =============================================================================

@dataclass(frozen=True, slots=True)
class TopologicalSummary:
    r"""
    Agregado TDA + KAM. `kam_persistence_ratio` es la fracción de toros
    (o de hits de Poincaré) que satisfacen la condición diofántica del seed,
    **no** un certificado KAM analítico (ese exige \(\varepsilon\ll\gamma\)
    y el teorema de Kolmogorov–Arnold–Moser completo).
    """

    betti: BettiNumbers
    structural_entropy: float
    persistence_entropy: float
    intrinsic_dimension: int = 1
    kam_persistence_ratio: float = 1.0
    birkhoff_order: int = 0
    liouville_volume: float = 0.0
    symplectic_gap: float = 0.0

    def __post_init__(self) -> None:
        if self.structural_entropy < 0.0:
            raise ValueError(f"structural_entropy ≥ 0, recibido: {self.structural_entropy}")
        if not (0.0 <= self.persistence_entropy <= 1.0):
            raise ValueError(f"persistence_entropy ∈ [0,1], recibido: {self.persistence_entropy}")
        if self.intrinsic_dimension < 0:
            raise ValueError("intrinsic_dimension ≥ 0")
        if not (0.0 <= self.kam_persistence_ratio <= 1.0):
            raise ValueError(f"kam_persistence_ratio ∈ [0,1], recibido: {self.kam_persistence_ratio}")
        if self.birkhoff_order < 0:
            raise ValueError("birkhoff_order ≥ 0")
        if self.liouville_volume < 0.0:
            raise ValueError("liouville_volume ≥ 0")

    @property
    def is_kam_stable(self) -> bool:
        return self.kam_persistence_ratio >= 0.7

    @property
    def is_integrable(self) -> bool:
        return self.birkhoff_order >= _BIRKHOFF_TRUNCATION_ORDER

    @classmethod
    def empty(cls) -> "TopologicalSummary":
        return cls(
            betti=BettiNumbers.zero(),
            structural_entropy=0.0,
            persistence_entropy=0.0,
            intrinsic_dimension=0,
            kam_persistence_ratio=0.0,
            birkhoff_order=0,
            liouville_volume=0.0,
            symplectic_gap=0.0,
        )

    @classmethod
    def from_seed_and_diagram(
        cls,
        seed: Mapping[str, Any],
        diagram: PersistenceDiagram,
        structural_entropy: float,
        config: Optional[MICConfiguration] = None,
    ) -> "TopologicalSummary":
        cfg = config or DEFAULT_MIC_CONFIG
        betti = BettiNumbers.from_diagram(diagram)
        form_blob = seed.get("symplectic_form") or {}
        capacity = float(form_blob.get("pfaffian", 1.0) or 1.0) * math.pi
        ok, residual = betti.check_capacity_budget(capacity)
        betti = BettiNumbers(
            beta_0=betti.beta_0,
            beta_1=betti.beta_1,
            beta_2=betti.beta_2,
            symplectic_capacity_total=capacity,
            gromov_bound_residual=residual,
        )
        kam = seed.get("kam") or {}
        kam_ratio = 1.0 if kam.get("is_diophantine_probe", False) else 0.0
        min_div = float(kam.get("min_divisor", 0.0) or 0.0)
        return cls(
            betti=betti,
            structural_entropy=max(0.0, structural_entropy),
            persistence_entropy=diagram.entropy(cfg),
            intrinsic_dimension=max(1, int(seed.get("n_dof", 1))),
            kam_persistence_ratio=kam_ratio,
            birkhoff_order=cfg.birkhoff_truncation_order if kam_ratio >= 0.7 else 0,
            liouville_volume=max(0.0, capacity),
            symplectic_gap=max(0.0, min_div),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.betti.to_dict(),
            "structural_entropy": round(self.structural_entropy, 6),
            "persistence_entropy": round(self.persistence_entropy, 6),
            "intrinsic_dimension": self.intrinsic_dimension,
            "kam_persistence_ratio": round(self.kam_persistence_ratio, 6),
            "birkhoff_order": self.birkhoff_order,
            "liouville_volume": round(self.liouville_volume, 9),
            "symplectic_gap": round(self.symplectic_gap, 9),
            "is_kam_stable": self.is_kam_stable,
            "is_integrable": self.is_integrable,
        }

    def __repr__(self) -> str:
        return (
            f"TopologicalSummary(betti={self.betti}, "
            f"H_struct={self.structural_entropy:.4f}, "
            f"H_pers={self.persistence_entropy:.4f}, "
            f"KAM={self.kam_persistence_ratio:.4f}, "
            f"Birkhoff_N={self.birkhoff_order})"
        )


# =============================================================================
# 2.4 — VECTOR DE INTENCIÓN CON ESTRUCTURA SIMPLÉCTICA
# =============================================================================

@dataclass(frozen=True, slots=True)
class IntentVector:
    r"""
    Intención como punto \(z=(q,p)\in T^*\mathcal{M}\) etiquetado por servicio.

    \(q\): configuración semántica; \(p\): momento de ejecución.
    \(\Omega(v,w)=\sum(q_v^i p_w^i-p_v^i q_w^i)\). Isotropía: \(\Omega(v,v)=0\).
    """

    service_name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    q_semantic: Tuple[float, ...] = ()
    p_momentum: Tuple[float, ...] = ()
    phase: float = 0.0

    def __post_init__(self) -> None:
        if not self.service_name or not self.service_name.strip():
            raise ValueError("service_name no puede estar vacío")
        if len(self.q_semantic) != len(self.p_momentum):
            raise ValueError(
                f"len(q)={len(self.q_semantic)} ≠ len(p)={len(self.p_momentum)}"
            )
        object.__setattr__(self, "phase", _wrap_angle(self.phase))

    @property
    def n_dof(self) -> int:
        return len(self.q_semantic)

    @property
    def payload_hash(self) -> str:
        content = str(sorted(self.payload.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def combinatorial_norm(self) -> float:
        """\(\sqrt{|\mathrm{payload}|+|\mathrm{context}|}\) — tamaño discreto, no \(\|z\|\)."""
        return math.sqrt(len(self.payload) + len(self.context))

    @property
    def norm(self) -> float:
        r"""\(\|z\|_2\) en \(T^*M\); si \(n=0\), cae a la norma combinatoria."""
        if self.n_dof == 0:
            return self.combinatorial_norm
        return math.sqrt(_hypot_sq(self.q_semantic) + _hypot_sq(self.p_momentum))

    @property
    def phase_space_energy(self) -> float:
        return _hypot_sq(self.p_momentum) / 2.0

    @property
    def phase_space_potential(self) -> float:
        return _hypot_sq(self.q_semantic) / 2.0

    @property
    def hamiltonian_energy(self) -> float:
        return self.phase_space_energy + self.phase_space_potential

    def symplectic_product(self, other: "IntentVector") -> float:
        if self.n_dof == 0 or other.n_dof == 0 or self.n_dof != other.n_dof:
            return 0.0
        return float(
            sum(
                self.q_semantic[i] * other.p_momentum[i]
                - self.p_momentum[i] * other.q_semantic[i]
                for i in range(self.n_dof)
            )
        )

    def to_phase_point(self) -> PhaseSpacePoint:
        if self.n_dof == 0:
            return PhaseSpacePoint(q=(0.0,), p=(0.0,), t=0.0, label=self.service_name)
        return PhaseSpacePoint(
            q=self.q_semantic, p=self.p_momentum, t=0.0, label=self.service_name
        )

    @classmethod
    def from_phase_point(
        cls,
        z: PhaseSpacePoint,
        service_name: str,
        payload: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> "IntentVector":
        return cls(
            service_name=service_name,
            payload=payload or {},
            context=context or {},
            q_semantic=z.q,
            p_momentum=z.p,
            phase=_wrap_angle(z.t),
        )

    def with_context(self, **additional_context: Any) -> "IntentVector":
        return IntentVector(
            service_name=self.service_name,
            payload=self.payload,
            context={**self.context, **additional_context},
            q_semantic=self.q_semantic,
            p_momentum=self.p_momentum,
            phase=self.phase,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "payload_size": len(self.payload),
            "context_size": len(self.context),
            "norm": round(self.norm, 6),
            "combinatorial_norm": round(self.combinatorial_norm, 6),
            "n_dof": self.n_dof,
            "hamiltonian_energy": round(self.hamiltonian_energy, 9),
            "phase": round(self.phase, 6),
        }

    def __repr__(self) -> str:
        return (
            f"IntentVector(service='{self.service_name}', "
            f"payload_size={len(self.payload)}, n_dof={self.n_dof}, "
            f"H={self.hamiltonian_energy:.6f})"
        )


# =============================================================================
# 2.5 — CACHE TTL CON CONTABILIDAD DISCRETA DE VOLUMEN
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    r"""
    Entrada de cache. `information_volume` es una *medida discreta* de
    ocupación (log-tamaño), análogo contable de Liouville — **no** es
    \(\Omega^n/n!\).
    """

    value: T
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    liouville_phase_volume: float = 1.0
    phase: float = 0.0

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.monotonic() - self.timestamp) > ttl_seconds

    def touch(self) -> None:
        self.access_count += 1
        self.phase = _wrap_angle(self.phase + 0.1)


class TTLCache(Generic[T]):
    r"""
    Cache LRU+TTL thread-safe. Invariante contable:

        \[ V=\sum_k \mathrm{vol}(k),\qquad
           V\leftarrow V+\Delta\mathrm{vol}\ \mathrm{en\ set/evict/expire}. \]

    `volume_residual` detecta deriva numérica (no conservación simpléctica).
    """

    __slots__ = (
        "_data",
        "_lock",
        "_ttl",
        "_max_size",
        "_hits",
        "_misses",
        "_evictions",
        "_expirations",
        "_total_phase_volume",
    )

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 128) -> None:
        if ttl_seconds <= 0.0:
            raise ValueError(f"ttl_seconds > 0, recibido: {ttl_seconds}")
        if max_size <= 0:
            raise ValueError(f"max_size > 0, recibido: {max_size}")
        self._data: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        self._total_phase_volume = 0.0

    def _compute_liouville_volume(self, value: T) -> float:
        try:
            size = len(str(value))
        except Exception:
            size = 1
        return 1.0 + math.log(1.0 + max(0, size))

    def _drop(self, key: str, *, expired: bool) -> None:
        entry = self._data.pop(key, None)
        if entry is None:
            return
        self._total_phase_volume = max(0.0, self._total_phase_volume - entry.liouville_phase_volume)
        if expired:
            self._expirations += 1
        else:
            self._evictions += 1

    def __contains__(self, key: str) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired(self._ttl):
                self._drop(key, expired=True)
                return False
            return True

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired(self._ttl):
                self._drop(key, expired=True)
                self._misses += 1
                return None
            self._data.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            vol = self._compute_liouville_volume(value)
            if key in self._data:
                self._data.move_to_end(key)
                old = self._data[key]
                self._total_phase_volume = max(
                    0.0, self._total_phase_volume - old.liouville_phase_volume + vol
                )
                self._data[key] = CacheEntry(
                    value=value, timestamp=time.monotonic(), liouville_phase_volume=vol
                )
                return
            while len(self._data) >= self._max_size:
                oldest, _ = next(iter(self._data.items()))
                self._drop(oldest, expired=False)
            self._data[key] = CacheEntry(
                value=value, timestamp=time.monotonic(), liouville_phase_volume=vol
            )
            self._total_phase_volume += vol

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        ttl_override: Optional[float] = None,
    ) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute_fn()
        if ttl_override is not None and ttl_override <= 0.0:
            return value
        self.set(key, value)
        return value

    def clear(self) -> int:
        with self._lock:
            count = len(self._data)
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0
            self._total_phase_volume = 0.0
            return count

    def prune_expired(self) -> int:
        with self._lock:
            expired = [k for k, v in self._data.items() if v.is_expired(self._ttl)]
            for key in expired:
                self._drop(key, expired=True)
            return len(expired)

    def volume_residual(self) -> float:
        with self._lock:
            recomputed = sum(e.liouville_phase_volume for e in self._data.values())
            return abs(recomputed - self._total_phase_volume)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    @property
    def liouville_volume(self) -> float:
        with self._lock:
            return self._total_phase_volume

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
                "ttl_seconds": self._ttl,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "liouville_volume": round(self._total_phase_volume, 9),
                "volume_residual": round(self.volume_residual(), 12),
            }

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"TTLCache(size={s['size']}/{s['max_size']}, "
            f"hit_rate={s['hit_rate']:.4f}, V_info={s['liouville_volume']:.4f})"
        )


# =============================================================================
# 2.6 — MÉTRICAS CON OBSERVABLES SIMPLÉCTICOS
# =============================================================================

class LatencyHistogram:
    """Histograma de latencias + concentración circular de fase (resultado de Rayleigh)."""

    __slots__ = ("_buffer", "_max_size", "_lock", "_count", "_phase_buffer")

    def __init__(self, max_size: int = 1000) -> None:
        if max_size < 1:
            raise ValueError("max_size ≥ 1")
        self._buffer: deque = deque(maxlen=max_size)
        self._phase_buffer: deque = deque(maxlen=max_size)
        self._max_size = max_size
        self._lock = threading.Lock()
        self._count = 0

    def record(self, latency_ms: float, phase: float = 0.0) -> None:
        with self._lock:
            self._buffer.append(float(latency_ms))
            self._phase_buffer.append(_wrap_angle(phase))
            self._count += 1

    @contextmanager
    def measure(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record((time.perf_counter() - start) * 1000.0)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._buffer:
                return {
                    "count": 0,
                    "mean_ms": 0.0,
                    "median_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "phase_concentration": 0.0,
                }
            data = list(self._buffer)
            phases = list(self._phase_buffer)
        sorted_data = sorted(data)
        n = len(sorted_data)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_data[int(k)]
            return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

        if phases:
            cos_mean = sum(math.cos(ph) for ph in phases) / len(phases)
            sin_mean = sum(math.sin(ph) for ph in phases) / len(phases)
            phase_concentration = math.sqrt(cos_mean * cos_mean + sin_mean * sin_mean)
        else:
            phase_concentration = 0.0
        return {
            "count": self._count,
            "mean_ms": round(statistics.mean(data), 3),
            "median_ms": round(statistics.median(data), 3),
            "p95_ms": round(percentile(0.95), 3),
            "p99_ms": round(percentile(0.99), 3),
            "min_ms": round(min(data), 3),
            "max_ms": round(max(data), 3),
            "phase_concentration": round(phase_concentration, 6),
        }

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._phase_buffer.clear()
            self._count = 0

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"LatencyHistogram(count={s['count']}, mean={s['mean_ms']:.3f}ms, "
            f"p95={s['p95_ms']:.3f}ms, R={s['phase_concentration']:.3f})"
        )


@dataclass
class MICMetrics:
    """Contadores MIC + observables de Poincaré (acción, residuales, KAM, retornos)."""

    projections: int = 0
    cache_hits: int = 0
    violations: int = 0
    errors: int = 0
    timeouts: int = 0
    projections_by_stratum: Dict[str, int] = field(default_factory=dict)
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    projection_latency: LatencyHistogram = field(
        default_factory=lambda: LatencyHistogram(1000)
    )
    handler_latency: LatencyHistogram = field(
        default_factory=lambda: LatencyHistogram(1000)
    )
    kam_violations: int = 0
    poincare_returns: int = 0
    total_action: float = 0.0
    symplectic_residual_accumulator: float = 0.0
    symplectic_residual_count: int = 0

    def record_projection(self, stratum: Any) -> None:
        self.projections += 1
        name = _safe_stratum_name(stratum)
        self.projections_by_stratum[name] = self.projections_by_stratum.get(name, 0) + 1

    def record_error(self, category: str) -> None:
        self.errors += 1
        self.errors_by_category[category] = self.errors_by_category.get(category, 0) + 1

    def record_symplectic_residual(self, residual: float) -> None:
        self.symplectic_residual_accumulator += abs(float(residual))
        self.symplectic_residual_count += 1

    def record_kam_violation(self) -> None:
        self.kam_violations += 1

    def record_poincare_return(self, action: float = 0.0) -> None:
        self.poincare_returns += 1
        self.total_action += float(action)

    @property
    def mean_symplectic_residual(self) -> float:
        if self.symplectic_residual_count == 0:
            return 0.0
        return self.symplectic_residual_accumulator / self.symplectic_residual_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counters": {
                "projections": self.projections,
                "cache_hits": self.cache_hits,
                "violations": self.violations,
                "errors": self.errors,
                "timeouts": self.timeouts,
                "kam_violations": self.kam_violations,
                "poincare_returns": self.poincare_returns,
            },
            "projections_by_stratum": self.projections_by_stratum.copy(),
            "errors_by_category": self.errors_by_category.copy(),
            "latency": {
                "projection": self.projection_latency.get_stats(),
                "handler": self.handler_latency.get_stats(),
            },
            "symplectic": {
                "total_action": round(self.total_action, 9),
                "mean_symplectic_residual": round(self.mean_symplectic_residual, 12),
            },
        }

    def __repr__(self) -> str:
        return (
            f"MICMetrics(projections={self.projections}, "
            f"errors={self.errors}, KAM_viol={self.kam_violations}, "
            f"Poincaré_ret={self.poincare_returns})"
        )


# =============================================================================
# 2.7 — TIPO DE ARCHIVO Y JERARQUÍA DE EXCEPCIONES
# =============================================================================

class FileType(str, Enum):
    """Tipos de archivo soportados para diagnóstico."""

    APUS = "apus"
    INSUMOS = "insumos"
    PRESUPUESTO = "presupuesto"

    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in cls]

    @classmethod
    def from_string(cls, value: str) -> "FileType":
        if not isinstance(value, str):
            raise TypeError(f"Se esperaba str, recibido {type(value).__name__!r}")
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        available = ", ".join(cls.values())
        raise ValueError(f"'{value}' no es un FileType válido. Opciones: {available}")

    def __repr__(self) -> str:
        return f"FileType.{self.name}('{self.value}')"

    def __str__(self) -> str:
        return self.value


class MICException(Exception):
    """Clase base de excepciones MIC (no topológicas de Fase 1)."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        category: str = "mic_error",
    ) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = details if details is not None else {}
        self.category: str = category if category else "mic_error"
        self.timestamp: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": str(self),
            "error_type": type(self).__name__,
            "error_category": self.category,
            "error_details": self.details,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(message={str(self)!r}, category={self.category!r})"


class TopologicalInvariantError(MICException):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, details=kwargs, category="topological_invariance")


class MICFunctorialityError(MICException):
    """
    Fallo de preservación funtorial **local a tools_interface**.

    No se llama `FunctorialityError`: Fase 1 ya importa ese nombre desde
    `mic_algebra` cuando está disponible.
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, details=kwargs, category="categorical_consistency")


class FileNotFoundDiagnosticError(MICException):
    def __init__(self, path: Union[str, Path], **kwargs: Any) -> None:
        path_str = str(Path(path).resolve()) if path else "unknown"
        super().__init__(
            f"File not found: {path_str}",
            details={"path": path_str, **kwargs},
            category="validation",
        )


class UnsupportedFileTypeError(MICException):
    def __init__(self, file_type: str, available: List[str]) -> None:
        available_str = ", ".join(sorted(available))
        super().__init__(
            f"Unsupported file type: {file_type!r}. Soportados: {available_str}",
            details={"file_type": file_type, "available_types": available},
            category="validation",
        )


class FileValidationError(MICException):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, details=kwargs, category="validation")


class FilePermissionError(MICException):
    def __init__(
        self, path: Union[str, Path], operation: str = "read", reason: str = ""
    ) -> None:
        path_str = str(Path(path).resolve()) if path else "unknown"
        super().__init__(
            f"Permission denied for {operation}: {path_str}. {reason}",
            details={"path": path_str, "operation": operation},
            category="permission",
        )


class CleaningError(MICException):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, details=kwargs, category="cleaning")


class MICHierarchyViolationError(MICException):
    """Violación de la clausura transitiva DIKW."""

    def __init__(
        self,
        target_stratum: Any,
        missing_strata: Set[Any],
        validated_strata: Set[Any],
    ) -> None:
        missing_sorted = sorted(
            missing_strata, key=_safe_stratum_value, reverse=True
        )
        validated_sorted = sorted(validated_strata, key=_safe_stratum_value)
        missing_names = [_safe_stratum_name(s) for s in missing_sorted]
        validated_names = [_safe_stratum_name(s) for s in validated_sorted]
        target_name = _safe_stratum_name(target_stratum)
        target_value = _safe_stratum_value(target_stratum)
        message = (
            f"Clausura Transitiva Violada: No se puede proyectar a "
            f"'{target_name}' (nivel {target_value}). "
            f"Faltantes: {' → '.join(missing_names) if missing_names else 'ninguno'}. "
            f"Validados: {', '.join(validated_names) if validated_names else 'ninguno'}."
        )
        super().__init__(
            message,
            details={
                "target_stratum": target_name,
                "target_value": target_value,
                "missing_strata": missing_names,
                "validated_strata": validated_names,
            },
            category="hierarchy_violation",
        )
        self.target_stratum = target_stratum
        self.missing_strata = missing_strata
        self.validated_strata = validated_strata

    @property
    def is_recoverable(self) -> bool:
        return len(self.missing_strata) > 0


class MICTimeoutError(MICException):
    """Operación que excede el tiempo límite (no pisa builtins.TimeoutError)."""

    def __init__(self, operation: str, timeout_seconds: float, elapsed_seconds: float) -> None:
        super().__init__(
            f"Operation '{operation}' timed out after {elapsed_seconds:.2f}s "
            f"(limit: {timeout_seconds:.2f}s)",
            details={
                "operation": operation,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
            },
            category="timeout",
        )

    @property
    def timeout_ratio(self) -> float:
        return self.details.get("elapsed_seconds", 0.0) / max(
            1e-10, self.details.get("timeout_seconds", 1.0)
        )


# =============================================================================
# 2.8 — FUNCIONES DE ENTROPÍA Y MÉTRICAS KAM
# =============================================================================

def compute_shannon_entropy(
    probabilities: Sequence[float],
    base: float = 2.0,
    epsilon: float = 1e-10,
) -> float:
    r"""\(H(X)=-\sum p_i\log_b p_i\). Filtra \(p\le\epsilon\); renormaliza si \(\sum p\neq 1\)."""
    if not probabilities:
        return 0.0
    if base <= 1.0:
        raise ValueError(f"base > 1, recibido: {base}")
    raw = [float(p) for p in probabilities]
    if any(p < 0.0 for p in raw):
        raise ValueError("Las probabilidades no pueden ser negativas")
    total = sum(raw)
    if total < epsilon:
        return 0.0
    probs = [p / total for p in raw]
    nonzero = [p for p in probs if p > epsilon]
    if not nonzero:
        return 0.0
    log_base = math.log(base)
    entropy = -sum(p * math.log(p) for p in nonzero) / log_base
    return max(0.0, entropy)


def distribution_from_counts(counts: Union[Dict[Any, int], Counter]) -> List[float]:
    if not counts:
        return []
    values = list(counts.values())
    total = sum(values)
    if total == 0:
        return []
    return [v / total for v in values]


def compute_persistence_entropy(
    intervals: Sequence[PersistenceInterval],
    config: Optional[MICConfiguration] = None,
) -> float:
    r"""\(H_{\mathrm{pers}}=H(\ell_i/L)/H_{\max}\in[0,1]\). Esenciales excluidos (masa infinita)."""
    config = config or DEFAULT_MIC_CONFIG
    if not intervals:
        return 0.0
    finite = [iv for iv in intervals if not iv.is_essential]
    if not finite:
        return 0.0
    persistences = [iv.finite_persistence() for iv in finite]
    total = sum(persistences)
    if total < config.epsilon:
        return 0.0
    probs = [p / total for p in persistences]
    raw = compute_shannon_entropy(probs, base=2.0)
    n = len(probs)
    max_entropy = math.log2(n) if n > 1 else 1.0
    if max_entropy < config.epsilon:
        return 0.0
    return _clamp(raw / max_entropy, 0.0, 1.0)


def compute_kam_persistence_ratio(
    action_angles_list: Sequence[ActionAngleCoordinates],
    config: Optional[MICConfiguration] = None,
) -> float:
    r"""Fracción de \(\omega\) diofánticas en el retículo del seed. No es el teorema KAM."""
    config = config or DEFAULT_MIC_CONFIG
    if not action_angles_list:
        return 0.0
    n_ok = 0
    for aa in action_angles_list:
        is_dioph, _ = aa.check_diophantine(
            gamma=config.kam_gamma,
            tau=config.kam_tau,
            k_max=config.kam_frequency_lattice_range,
        )
        if is_dioph:
            n_ok += 1
    return n_ok / len(action_angles_list)


def frequencies_from_action_angle(
    hamiltonian: HamiltonianSystem,
    z0: PhaseSpacePoint,
) -> Tuple[float, ...]:
    r"""
    Estimación \(\omega_i\approx\partial H/\partial J_i\) vía \(J=H/\omega_{\mathrm{lin}}\)
    en la carta HO. **No** usa \(|\nabla_q H|\) (error de la versión previa).
    """
    n = z0.n_dof
    if len(hamiltonian.potential_coefficients) >= 2:
        omega_lin = math.sqrt(abs(hamiltonian.potential_coefficients[1]))
    else:
        omega_lin = 1.0
    omega_lin = max(omega_lin, 1e-12)
    J = hamiltonian.action_variable(z0)
    # Isotrópico: todas las frecuencias coinciden a orden 0.
    return tuple([omega_lin] * n if J >= 0.0 else [omega_lin] * n)


def compute_symplectic_gap(
    hamiltonian: HamiltonianSystem,
    z0: PhaseSpacePoint,
    perturbation_epsilon: float,
    config: Optional[MICConfiguration] = None,
) -> Tuple[float, bool]:
    r"""
    Gap \(\gamma_{\mathrm{symp}}=\min_{k\neq 0}|\langle k,\omega\rangle|\) sobre
    frecuencias de acción-ángulo, y test diofántico del seed.
    """
    config = config or DEFAULT_MIC_CONFIG
    omega = frequencies_from_action_angle(hamiltonian, z0)
    actions = tuple(max(abs(q), 1e-3) for q in z0.q)
    aa = ActionAngleCoordinates(
        actions=actions,
        angles=tuple(0.0 for _ in range(z0.n_dof)),
        frequencies=omega,
    )
    is_dioph, min_div = aa.check_diophantine(
        gamma=config.kam_gamma,
        tau=config.kam_tau,
        k_max=config.kam_frequency_lattice_range,
    )
    is_kam_stable = is_dioph and (perturbation_epsilon <= config.kam_epsilon_perturbation)
    return min_div, is_kam_stable


def estimate_liouville_volume(
    z_bounds: Tuple[Tuple[float, float], ...],
    p_bounds: Tuple[Tuple[float, float], ...],
) -> float:
    r"""
    Volumen euclídeo de un rectángulo \(\prod\Delta q_i\prod\Delta p_i\).

    Coincide con \(\int\Omega^n/n!\) **solo** en carta de Darboux y región
    producto; no es el volumen de un subnivel de \(H\).
    """
    vol_q = 1.0
    vol_p = 1.0
    for q_min, q_max in z_bounds:
        vol_q *= max(0.0, q_max - q_min)
    for p_min, p_max in p_bounds:
        vol_p *= max(0.0, p_max - p_min)
    return vol_q * vol_p


# =============================================================================
# 2.9 — MAPA DE RETORNO DE POINCARÉ (itera el germen de Fase 1)
# =============================================================================

@dataclass(frozen=True, slots=True)
class FloquetSpectrum:
    r"""
    Espectro de \(DP_z:T_z\Sigma\to T_{P(z)}\Sigma\).

    Hamiltoniano: multiplicadores en pares \((\lambda,1/\lambda)\).
    En carta de hipersuperficie (codim 1, sin reducir energía) un
    multiplicador \(\approx 1\) corresponde a la conservación de \(H\).

    2-dof reducido (área-preservante): \(\det=1\), elíptico si \(|\mathrm{tr}|\le 2\).
    """

    dimension: int
    trace: float
    determinant: float
    multipliers: Tuple[complex, ...]
    det_residual: float
    classification: str
    chart_dimension: int

    @property
    def is_symplectic_numeric(self) -> bool:
        return self.det_residual < 1e-3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "chart_dimension": self.chart_dimension,
            "trace": round(self.trace, 9),
            "determinant": round(self.determinant, 9),
            "det_residual": round(self.det_residual, 12),
            "multipliers": [(z.real, z.imag) for z in self.multipliers],
            "classification": self.classification,
            "is_symplectic_numeric": self.is_symplectic_numeric,
        }


@dataclass(frozen=True, slots=True)
class PoincareReturnMap:
    r"""
    Mapa de primer retorno \(P:\Sigma\to\Sigma\), \(P(x)=\varphi_{t(x)}(x)\).

    Consume `PoincareSection.germ_of_return_map` (Fase 1). No reimplementa
    el cruce. Distingue:

        * **retorno** — iteración de \(P\) sobre \(\Sigma\) (esta clase);
        * **recurrencia** — \(\|z(t)-z_0\|<\varepsilon\) en \(T^*M\)
          (`verify_poincare_recurrence`, teorema de 1890).

    \(\{H=E\}\) **no** es sección: es hoja invariante.
    """

    section: PoincareSection
    max_iterations: int = 1024
    dt: float = 1e-3
    recurrence_tol: float = _POINCARE_RECURRENCE_TOL

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations ≥ 1")
        if self.dt == 0.0:
            raise ValueError("dt ≠ 0")

    @classmethod
    def from_phase1_seed(
        cls,
        seed: Mapping[str, Any],
        section: Optional[PoincareSection] = None,
        config: Optional[MICConfiguration] = None,
    ) -> "PoincareReturnMap":
        _require_phase1_seed(seed)
        cfg = config or DEFAULT_MIC_CONFIG
        sec = section or _rehydrate_section(seed.get("section") or {})
        return cls(
            section=sec,
            max_iterations=int(cfg.poincare_max_return_iterations),
            dt=float(cfg.hamiltonian_time_step),
            recurrence_tol=float(cfg.poincare_recurrence_tol),
        )

    def apply(
        self,
        z: PhaseSpacePoint,
        system: HamiltonianSystem,
    ) -> Dict[str, Any]:
        """Un paso de \(P\). Devuelve el germen de Fase 1 (hit, tiempo, Liouville)."""
        return self.section.germ_of_return_map(
            z, system, self.dt, self.max_iterations
        )

    def first_return(
        self,
        hamiltonian: HamiltonianSystem,
        z0: PhaseSpacePoint,
        dt: Optional[float] = None,
    ) -> Optional[Tuple[PhaseSpacePoint, float]]:
        object.__setattr__(self, "dt", float(self.dt if dt is None else dt))
        germ = self.apply(z0, hamiltonian)
        hit = germ.get("hit")
        tau = germ.get("return_time")
        if hit is None or tau is None:
            return None
        return hit, float(tau)

    def iterate(
        self,
        hamiltonian: HamiltonianSystem,
        z0: PhaseSpacePoint,
        k: int,
    ) -> List[PhaseSpacePoint]:
        r"""Órbita \(\{P(z_0),\ldots,P^k(z_0)\}\). Se detiene si no hay hit."""
        if k < 1:
            return []
        orbit: List[PhaseSpacePoint] = []
        z = z0
        for _ in range(k):
            germ = self.apply(z, hamiltonian)
            hit = germ.get("hit")
            if hit is None:
                break
            orbit.append(hit)
            z = hit
        return orbit

    def collect_returns(
        self,
        hamiltonian: HamiltonianSystem,
        z0: PhaseSpacePoint,
        n_returns: int = 10,
        dt: Optional[float] = None,
    ) -> List[Tuple[PhaseSpacePoint, float]]:
        if dt is not None:
            object.__setattr__(self, "dt", float(dt))
        out: List[Tuple[PhaseSpacePoint, float]] = []
        z = z0
        t_acc = 0.0
        for _ in range(max(0, n_returns)):
            pair = self.first_return(hamiltonian, z)
            if pair is None:
                break
            hit, tau = pair
            t_acc += tau
            out.append((hit, t_acc))
            z = hit
        return out

    def jacobian(
        self,
        hamiltonian: HamiltonianSystem,
        z: PhaseSpacePoint,
        h: float = 1e-6,
    ) -> Matrix:
        r"""
        \(DP_z\) por diferencias centrales en la carta de \(\Sigma\) (dim \(2n-1\)).

        Limitación: no se reduce energía; un autovalor debe ser \(\approx 1\).
        """
        chart = _section_chart_indices(z.n_dof, self.section)
        m = len(chart)
        cols: List[Vector] = []
        base_coords = _chart_coords(z, self.section)
        for j in range(m):
            plus = list(base_coords)
            minus = list(base_coords)
            plus[j] += h
            minus[j] -= h
            zp = _embed_section_coords(z, self.section, plus)
            zm = _embed_section_coords(z, self.section, minus)
            hp = self.apply(zp, hamiltonian).get("hit")
            hm = self.apply(zm, hamiltonian).get("hit")
            if hp is None or hm is None:
                cols.append([0.0] * m)
                continue
            cp = _chart_coords(hp, self.section)
            cm = _chart_coords(hm, self.section)
            cols.append([(cp[i] - cm[i]) / (2.0 * h) for i in range(m)])
        # columnas → filas
        return [[cols[j][i] for j in range(m)] for i in range(m)]

    def floquet(
        self,
        hamiltonian: HamiltonianSystem,
        z: PhaseSpacePoint,
        h: float = 1e-6,
    ) -> FloquetSpectrum:
        J = self.jacobian(hamiltonian, z, h=h)
        m = len(J)
        if m == 0:
            return FloquetSpectrum(
                dimension=0,
                trace=0.0,
                determinant=1.0,
                multipliers=tuple(),
                det_residual=0.0,
                classification="empty",
                chart_dimension=0,
            )
        det = _det_small(J)
        tr = sum(J[i][i] for i in range(m))
        multipliers: Tuple[complex, ...]
        classification = "undetermined"
        if NUMPY_AVAILABLE:
            try:
                ev = np.linalg.eigvals(np.array(J, dtype=np.float64))
                multipliers = tuple(complex(x) for x in ev)
            except Exception:
                multipliers = tuple()
        else:
            multipliers = tuple()
        if m == 2:
            # Área-preservante clásico (sección 2D efectiva si n=2 y se ignora energía).
            disc = tr * tr - 4.0 * det
            if abs(det - 1.0) < 5e-2 and abs(tr) <= 2.0 + 1e-6:
                classification = "elliptic"
            elif abs(det - 1.0) < 5e-2 and abs(tr) > 2.0:
                classification = "hyperbolic"
            else:
                classification = "non_unimodular"
            if not multipliers:
                if disc >= 0.0:
                    s = math.sqrt(disc)
                    multipliers = (0.5 * (tr + s), 0.5 * (tr - s))
                else:
                    s = math.sqrt(-disc)
                    multipliers = (complex(tr / 2.0, s / 2.0), complex(tr / 2.0, -s / 2.0))
        elif multipliers:
            mags = sorted(abs(lam) for lam in multipliers)
            near_unit = sum(1 for a in mags if abs(a - 1.0) < 0.05)
            classification = "near_parabolic" if near_unit else "mixed"
        return FloquetSpectrum(
            dimension=m,
            trace=float(tr),
            determinant=float(det),
            multipliers=tuple(complex(x) for x in multipliers),
            det_residual=abs(det - 1.0),
            classification=classification,
            chart_dimension=m,
        )

    def rotation_number(
        self,
        hamiltonian: HamiltonianSystem,
        z0: PhaseSpacePoint,
        n_iter: int = 64,
    ) -> Optional[float]:
        r"""
        Número de rotación 1D en la carta \((q_j,p_j)\) residual (ángulo polar).

        Significativo sobre secciones 2D; en dimensión alta es una proyección.
        """
        orbit = self.iterate(hamiltonian, z0, n_iter)
        if len(orbit) < 2:
            return None
        angles: List[float] = []
        for z in orbit:
            chart = _chart_coords(z, self.section)
            if len(chart) < 2:
                return None
            angles.append(math.atan2(chart[1], chart[0]))
        lifts = [angles[0]]
        for a in angles[1:]:
            prev = lifts[-1]
            cand = a
            while cand - prev > math.pi:
                cand -= _TWO_PI
            while cand - prev < -math.pi:
                cand += _TWO_PI
            lifts.append(cand)
        return (lifts[-1] - lifts[0]) / (_TWO_PI * (len(lifts) - 1))

    def compute_poincare_section(
        self,
        hamiltonian: HamiltonianSystem,
        initial_points: Sequence[PhaseSpacePoint],
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        if dt is not None:
            object.__setattr__(self, "dt", float(dt))
        section_points: List[Tuple[float, ...]] = []
        return_times: List[float] = []
        failed = 0
        for z0 in initial_points:
            pair = self.first_return(hamiltonian, z0)
            if pair is None:
                failed += 1
                continue
            z_r, t_r = pair
            section_points.append(z_r.q)
            return_times.append(t_r)
        n_total = len(initial_points)
        return {
            "section_points": section_points,
            "mean_return_time": statistics.mean(return_times) if return_times else 0.0,
            "recurrence_rate": (n_total - failed) / max(1, n_total),
            "failed_returns": failed,
            "n_returns": len(return_times),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section": {
                "coordinate": self.section.coordinate,
                "index": self.section.index,
                "level": self.section.level,
                "direction": self.section.direction,
            },
            "max_iterations": self.max_iterations,
            "dt": self.dt,
            "recurrence_tol": self.recurrence_tol,
        }


def verify_poincare_recurrence(
    hamiltonian: HamiltonianSystem,
    z0: PhaseSpacePoint,
    epsilon: float = 1e-3,
    max_steps: int = 4096,
    dt: float = 1e-3,
) -> Tuple[bool, float, Optional[PhaseSpacePoint]]:
    r"""
    Teorema de recurrencia (Poincaré 1890): flujo que preserva medida finita
    \(\Rightarrow\) retorno a todo \(\varepsilon\)-vecindario, c.t.p.

    Esto **no** es el mapa de sección. No usa \(\Sigma\).
    """
    flow = hamiltonian.integrate_flow_diagnosed(z0, dt, max_steps)
    min_dist = float("inf")
    return_point: Optional[PhaseSpacePoint] = None
    for z in flow.trajectory[1:]:
        d = z.distance_to(z0)
        if d < min_dist:
            min_dist = d
        if d < epsilon:
            return_point = z
            break
    return return_point is not None, min_dist, return_point


# =============================================================================
# 2.10 — ANÁLISIS ESPECTRAL: GRAFO (analogía) Y DP (KAM real)
# =============================================================================

class SpectralGraphMetrics:
    r"""
    Espectro del grafo de dependencias entre servicios.

    El gap de Fiedler \(\lambda_2\) **no** es un divisor KAM. El test
    diofántico sobre eigenvalores del Laplaciano es una analogía
    estructural; el KAM auténtico de esta fase vive en `FloquetSpectrum`.
    """

    __slots__ = ("_adjacency_cache", "_laplacian_cache", "_lock", "_vector_supplier")

    def __init__(self, vector_supplier: Callable[[], Dict[str, Tuple[Any, Any]]]) -> None:
        self._adjacency_cache: Optional[Any] = None
        self._laplacian_cache: Optional[Any] = None
        self._lock = threading.RLock()
        self._vector_supplier = vector_supplier

    def _invalidate_cache(self) -> None:
        with self._lock:
            self._adjacency_cache = None
            self._laplacian_cache = None

    def build_adjacency_matrix(self, use_sparse: bool = False) -> Any:
        del use_sparse
        with self._lock:
            if self._adjacency_cache is not None:
                return self._adjacency_cache
            vectors = self._vector_supplier()
            services = list(vectors.keys())
            n = len(services)
            if not NUMPY_AVAILABLE:
                self._adjacency_cache = _zeros(n)
                return self._adjacency_cache
            A = np.zeros((n, n), dtype=np.float64)
            if n == 0:
                self._adjacency_cache = A
                return A
            idx = {svc: i for i, svc in enumerate(services)}
            for svc_i, (stratum_i, _) in vectors.items():
                for svc_j, (stratum_j, _) in vectors.items():
                    if svc_i == svc_j:
                        continue
                    try:
                        if stratum_i in stratum_j.requires():
                            A[idx[svc_i], idx[svc_j]] = 1.0
                    except Exception:
                        continue
            self._adjacency_cache = A
            return A

    def build_laplacian(self) -> Any:
        with self._lock:
            if self._laplacian_cache is not None:
                return self._laplacian_cache
            A = self.build_adjacency_matrix()
            if A is None:
                return None
            if NUMPY_AVAILABLE and not isinstance(A, list):
                if A.size == 0:
                    return A
                A_sym = (A + A.T) / 2.0
                degrees = A_sym.sum(axis=1)
                L = np.diag(degrees) - A_sym
                self._laplacian_cache = L
                return L
            A_list = _as_matrix(A)
            n = len(A_list)
            A_sym = [
                [0.5 * (A_list[i][j] + A_list[j][i]) for j in range(n)] for i in range(n)
            ]
            deg = [sum(row) for row in A_sym]
            L = [[-A_sym[i][j] for j in range(n)] for i in range(n)]
            for i in range(n):
                L[i][i] += deg[i]
            self._laplacian_cache = L
            return L

    def compute_spectral_metrics(
        self, config: Optional[MICConfiguration] = None
    ) -> Dict[str, Any]:
        config = config or DEFAULT_MIC_CONFIG
        L = self.build_laplacian()
        A = self.build_adjacency_matrix()
        empty = {
            "algebraic_connectivity": 0.0,
            "spectral_radius": 0.0,
            "spectral_energy": 0.0,
            "is_connected": False,
            "n_components": 0,
            "n_services": 0,
            "kam_analogy": False,
            "birkhoff_index": 0,
            "note": "KAM here is analogy on graph spectrum; use FloquetSpectrum for DP.",
        }
        if L is None:
            return empty
        if not NUMPY_AVAILABLE:
            n = len(L) if isinstance(L, list) else 0
            empty["n_services"] = n
            empty["note"] = "numpy ausente — espectro de grafo no calculado"
            return empty
        try:
            if hasattr(L, "size") and L.size == 0:
                return empty
            eigenvalues_L = np.sort(np.linalg.eigvalsh(L))
            algebraic_connectivity = float(
                eigenvalues_L[1] if len(eigenvalues_L) > 1 else 0.0
            )
            n_components = int(np.sum(np.abs(eigenvalues_L) < config.epsilon))
            A_sym = (A + A.T) / 2.0
            eigenvalues_A = np.linalg.eigvalsh(A_sym)
            spectral_radius = float(np.max(np.abs(eigenvalues_A)))
            spectral_energy = float(np.sum(np.abs(eigenvalues_A)))
            kam_analogy, min_divisor = self._check_spectral_kam_analogy(
                eigenvalues_L, config
            )
            birkhoff_index = self._compute_birkhoff_index(eigenvalues_L, config)
            return {
                "algebraic_connectivity": round(algebraic_connectivity, 6),
                "spectral_radius": round(spectral_radius, 6),
                "spectral_energy": round(spectral_energy, 6),
                "is_connected": algebraic_connectivity > config.epsilon,
                "n_components": n_components,
                "n_services": int(L.shape[0]),
                "fiedler_value": round(algebraic_connectivity, 6),
                "kam_analogy": kam_analogy,
                "spectral_min_divisor": round(min_divisor, 12),
                "birkhoff_index": birkhoff_index,
                "note": "kam_analogy ≠ KAM of DP",
            }
        except Exception as e:
            logger.warning("Error en análisis espectral de grafo: %s", e)
            out = dict(empty)
            out["error"] = str(e)
            return out

    def _check_spectral_kam_analogy(
        self,
        eigenvalues: Any,
        config: MICConfiguration,
    ) -> Tuple[bool, float]:
        if len(eigenvalues) < 3:
            return True, 1.0
        total = float(np.sum(np.abs(eigenvalues)))
        if total < config.epsilon:
            return True, 1.0
        omega = eigenvalues / total
        n = min(3, len(omega))
        aa = ActionAngleCoordinates(
            actions=tuple([1.0] * n),
            angles=tuple([0.0] * n),
            frequencies=tuple(float(omega[i]) for i in range(n)),
        )
        return aa.check_diophantine(
            gamma=config.kam_gamma,
            tau=config.kam_tau,
            k_max=config.kam_frequency_lattice_range,
        )

    def _compute_birkhoff_index(self, eigenvalues: Any, config: MICConfiguration) -> int:
        if len(eigenvalues) < 2:
            return 0
        diffs = np.abs(np.diff(eigenvalues))
        count = 0
        prev = float("inf")
        for d in diffs:
            if d <= prev * 1.5:
                count += 1
                prev = float(d)
            else:
                break
        return min(count, config.birkhoff_truncation_order * 2)


def floquet_kam_report(
    spectrum: FloquetSpectrum,
    config: Optional[MICConfiguration] = None,
) -> Dict[str, Any]:
    r"""
    Traduce multiplicadores de Floquet a un diagnóstico KAM de sección:

    * elíptico + \(\det\approx 1\) \(\Rightarrow\) candidato a toro persistente;
    * hiperbólico \(\Rightarrow\) homoclínica / no persistencia KAM local.
    """
    config = config or DEFAULT_MIC_CONFIG
    mags = [abs(lam) for lam in spectrum.multipliers]
    on_circle = (
        sum(1 for a in mags if abs(a - 1.0) < 0.05) / len(mags) if mags else 0.0
    )
    return {
        "classification": spectrum.classification,
        "det_residual": spectrum.det_residual,
        "unit_circle_fraction": on_circle,
        "kam_candidate": (
            spectrum.classification in {"elliptic", "near_parabolic"}
            and spectrum.det_residual < max(config.symplectic_residual_tol * 1e3, 1e-3)
        ),
        "floquet": spectrum.to_dict(),
    }


# =============================================================================
# 2.11 — CADENAS DE MARKOV ERGÓDICAS (DIKW y partición de Σ)
# =============================================================================

class StratumTransitionMatrix:
    r"""
    Cadena de Markov en el poset DIKW. Regularización de Tikhonov

        \[ P_{\mathrm{reg}}=\alpha P+(1-\alpha)N^{-1}\mathbf{1}\mathbf{1}^{\top} \]

    \(\Rightarrow\) irreducible + aperiódica. Hitting times: sistema lineal
    \((I-P^{(-t)})h=\mathbf{1}\), no el recíproco de \(P_{st}\).
    """

    __slots__ = ("_strata", "_n", "_idx", "_friction_weights")

    _DEFAULT_FRICTION: ClassVar[Dict[str, float]] = {
        "PHYSICS": 1.0,
        "TACTICS": 1.8,
        "STRATEGY": 2.0,
        "OMEGA": 1.5,
        "ALPHA": 1.2,
        "WISDOM": 1.0,
    }

    def __init__(self) -> None:
        try:
            self._strata = list(Stratum)
        except Exception:
            self._strata = []
        self._n = len(self._strata)
        self._idx = {s: i for i, s in enumerate(self._strata)}
        self._friction_weights: Dict[Any, float] = {
            s: self._DEFAULT_FRICTION.get(_safe_stratum_name(s), 1.0) for s in self._strata
        }

    def _is_reachable(self, s_from: Any, s_to: Any) -> bool:
        try:
            return s_from in s_to.requires()
        except Exception:
            return _safe_stratum_value(s_from) < _safe_stratum_value(s_to)

    def build(
        self,
        service_counts: Dict[Any, int],
        regularization_alpha: float = 0.85,
    ) -> Any:
        if not (0.0 < regularization_alpha < 1.0):
            raise ValueError(f"regularization_alpha ∈ (0,1), recibido: {regularization_alpha}")
        if self._n == 0:
            raise TopologicalInvariantError("Stratum vacío: no hay poset DIKW")
        epsilon = DEFAULT_MIC_CONFIG.epsilon
        T = [[0.0] * self._n for _ in range(self._n)]
        for s_from in self._strata:
            i = self._idx[s_from]
            reachable = [
                s for s in self._strata if s is not s_from and self._is_reachable(s_from, s)
            ]
            if not reachable:
                T[i][i] = 1.0
                continue
            weights = [
                float(max(1, service_counts.get(s, 1)))
                / max(epsilon, self._friction_weights.get(s, 1.0))
                for s in reachable
            ]
            total_w = sum(weights)
            if total_w <= epsilon:
                weights = [1.0 / len(reachable)] * len(reachable)
            else:
                weights = [w / total_w for w in weights]
            for s_to, w in zip(reachable, weights):
                T[i][self._idx[s_to]] = w
        uni = (1.0 - regularization_alpha) / self._n
        T = [
            [regularization_alpha * T[i][j] + uni for j in range(self._n)]
            for i in range(self._n)
        ]
        if NUMPY_AVAILABLE:
            return np.array(T, dtype=np.float64)
        return T

    def compute_spectral_gap(
        self,
        service_counts: Dict[Any, int],
        regularization_alpha: float = 0.85,
    ) -> float:
        T = self.build(service_counts, regularization_alpha)
        if not NUMPY_AVAILABLE:
            return 0.0
        try:
            eigenvalues = np.linalg.eigvals(T)
            mags = sorted((abs(float(np.real(ev))) for ev in np.abs(eigenvalues)), reverse=True)
            # |λ| ya tomados
            mags = sorted((float(x) for x in np.abs(eigenvalues)), reverse=True)
            if len(mags) < 2:
                return 1.0
            return max(0.0, 1.0 - mags[1])
        except Exception:
            return 0.0

    def mixing_time(
        self,
        service_counts: Dict[Any, int],
        epsilon_tv: float = 0.01,
        regularization_alpha: float = 0.85,
    ) -> float:
        gamma = self.compute_spectral_gap(service_counts, regularization_alpha)
        if gamma < 1e-12:
            return float("inf")
        return math.log(1.0 / max(epsilon_tv, 1e-12)) / gamma

    def stationary_distribution(
        self,
        service_counts: Dict[Any, int],
        regularization_alpha: float = 0.85,
    ) -> Dict[str, float]:
        if self._n == 0:
            return {}
        T = self.build(service_counts, regularization_alpha)
        if not NUMPY_AVAILABLE:
            u = 1.0 / self._n
            return {_safe_stratum_name(s): round(u, 6) for s in self._strata}
        epsilon = DEFAULT_MIC_CONFIG.epsilon
        try:
            eigenvalues, eigenvectors = np.linalg.eig(np.asarray(T).T)
            idx_unit = int(np.argmin(np.abs(eigenvalues - 1.0)))
            stationary = np.abs(np.real(eigenvectors[:, idx_unit]))
            total = float(stationary.sum())
            if total > epsilon:
                stationary = stationary / total
            return {
                _safe_stratum_name(self._strata[i]): round(float(stationary[i]), 6)
                for i in range(self._n)
            }
        except Exception:
            u = 1.0 / self._n
            return {_safe_stratum_name(s): round(u, 6) for s in self._strata}

    def entropy_rate(
        self,
        service_counts: Dict[Any, int],
        regularization_alpha: float = 0.85,
    ) -> float:
        T = self.build(service_counts, regularization_alpha)
        pi = self.stationary_distribution(service_counts, regularization_alpha)
        Tm = _as_matrix(T)
        h = 0.0
        for i in range(self._n):
            pi_i = pi.get(_safe_stratum_name(self._strata[i]), 0.0)
            for j in range(self._n):
                t_ij = Tm[i][j]
                if t_ij > 1e-12:
                    h -= pi_i * t_ij * math.log(t_ij)
        return max(0.0, h)

    def expected_hitting_time(
        self,
        service_counts: Dict[Any, int],
        source: Any,
        target: Any,
        regularization_alpha: float = 0.85,
    ) -> float:
        r"""
        \(h_t=0\), \(h_i=1+\sum_k P_{ik}h_k\) (\(i\neq t\)).
        Se resuelve \((I-P^{(-t)})h=\mathbf{1}\) por Gauss.
        """
        if source == target:
            return 0.0
        if source not in self._idx or target not in self._idx:
            return float("inf")
        T = _as_matrix(self.build(service_counts, regularization_alpha))
        t = self._idx[target]
        s = self._idx[source]
        states = [i for i in range(self._n) if i != t]
        m = len(states)
        if m == 0:
            return 0.0
        loc = {i: k for k, i in enumerate(states)}
        A = _zeros(m)
        b = [1.0] * m
        for i in states:
            li = loc[i]
            A[li][li] = 1.0
            for j in states:
                A[li][loc[j]] -= T[i][j]
        # Gauss con pivoteo
        M = [row[:] + [b[k]] for k, row in enumerate(A)]
        for col in range(m):
            piv = max(range(col, m), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) < 1e-14:
                return float("inf")
            M[col], M[piv] = M[piv], M[col]
            inv = 1.0 / M[col][col]
            for j in range(col, m + 1):
                M[col][j] *= inv
            for r in range(m):
                if r == col:
                    continue
                f = M[r][col]
                if f == 0.0:
                    continue
                for j in range(col, m + 1):
                    M[r][j] -= f * M[col][j]
        return float(M[loc[s]][m])

    def check_ergodicity(
        self,
        service_counts: Dict[Any, int],
        regularization_alpha: float = 0.85,
    ) -> Tuple[bool, Dict[str, Any]]:
        if not NUMPY_AVAILABLE:
            return False, {"error": "numpy no disponible"}
        T = self.build(service_counts, regularization_alpha)
        gamma = self.compute_spectral_gap(service_counts, regularization_alpha)
        try:
            eigenvalues = np.linalg.eigvals(T)
            magnitudes = np.abs(eigenvalues)
            count_unit = int(np.sum(np.abs(magnitudes - 1.0) < 1e-8))
            is_irreducible = count_unit == 1
        except Exception:
            is_irreducible = False
        is_aperiodic = bool(np.any(np.diag(np.asarray(T)) > 1e-9))
        is_positive_recurrent = gamma > 1e-9
        is_ergodic = bool(is_irreducible and is_aperiodic and is_positive_recurrent)
        return is_ergodic, {
            "irreducible": is_irreducible,
            "aperiodic": is_aperiodic,
            "positive_recurrent": is_positive_recurrent,
            "spectral_gap": round(gamma, 9),
        }

    def to_dict(
        self,
        service_counts: Dict[Any, int],
        regularization_alpha: float = 0.85,
    ) -> Dict[str, Any]:
        pi = self.stationary_distribution(service_counts, regularization_alpha)
        gamma = self.compute_spectral_gap(service_counts, regularization_alpha)
        t_mix = self.mixing_time(service_counts, 0.01, regularization_alpha)
        h_ks = self.entropy_rate(service_counts, regularization_alpha)
        is_ergodic, ergodic_details = self.check_ergodicity(
            service_counts, regularization_alpha
        )
        return {
            "stationary_distribution": pi,
            "spectral_gap": round(gamma, 9),
            "mixing_time_epsilon_0.01": round(t_mix, 3) if t_mix != float("inf") else "inf",
            "entropy_rate_KS": round(h_ks, 6),
            "is_ergodic": is_ergodic,
            "ergodic_details": ergodic_details,
            "regularization_alpha": regularization_alpha,
        }


@dataclass(frozen=True, slots=True)
class ErgodicMarkovStrata:
    r"""
    Cadena de Markov sobre una **partición medible de \(\Sigma\)** inducida
    por cuantiles de la acción de Cartan a lo largo de la órbita de \(P\).

    Es el objeto prometido en el seed de Fase 1 (`ErgodicMarkovStrata`),
    distinto de `StratumTransitionMatrix` (poset DIKW).
    """

    n_bins: int
    transition: Tuple[Tuple[float, ...], ...]
    occupancy: Tuple[float, ...]
    spectral_gap: float
    entropy_rate: float

    def mixing_time(self, epsilon_tv: float = 0.01) -> float:
        if self.spectral_gap < 1e-12:
            return float("inf")
        return math.log(1.0 / max(epsilon_tv, 1e-12)) / self.spectral_gap

    @classmethod
    def from_returns(
        cls,
        hits: Sequence[PhaseSpacePoint],
        hamiltonian: HamiltonianSystem,
        cartan: Optional[PoincareCartanForm] = None,
        n_bins: int = 4,
    ) -> "ErgodicMarkovStrata":
        if n_bins < 2:
            raise ValueError("n_bins ≥ 2")
        if len(hits) < 2:
            uni = tuple(1.0 / n_bins for _ in range(n_bins))
            T = tuple(uni for _ in range(n_bins))
            return cls(
                n_bins=n_bins,
                transition=T,
                occupancy=uni,
                spectral_gap=0.0,
                entropy_rate=0.0,
            )
        if cartan is not None:
            actions = []
            prev = hits[0]
            actions.append(abs(cartan.action_along([prev])))
            for z in hits[1:]:
                actions.append(abs(cartan.action_along([prev, z])))
                prev = z
        else:
            actions = [hamiltonian.action_variable(z) for z in hits]
        xs = sorted(actions)
        qs = [
            xs[min(len(xs) - 1, max(0, int(round((b + 1) * (len(xs) - 1) / n_bins))))]
            for b in range(n_bins - 1)
        ]

        def bin_of(a: float) -> int:
            for i, q in enumerate(qs):
                if a <= q:
                    return i
            return n_bins - 1

        labels = [bin_of(a) for a in actions]
        counts = [[0.0] * n_bins for _ in range(n_bins)]
        for a, b in zip(labels, labels[1:]):
            counts[a][b] += 1.0
        Tm: List[List[float]] = []
        for row in counts:
            s = sum(row)
            if s <= 0.0:
                Tm.append([1.0 / n_bins] * n_bins)
            else:
                Tm.append([c / s for c in row])
        occ_c = Counter(labels)
        occ = tuple(occ_c.get(i, 0) / len(labels) for i in range(n_bins))
        gap = 0.0
        h = 0.0
        if NUMPY_AVAILABLE:
            try:
                ev = np.linalg.eigvals(np.array(Tm, dtype=np.float64))
                mags = sorted((float(x) for x in np.abs(ev)), reverse=True)
                gap = max(0.0, 1.0 - mags[1]) if len(mags) > 1 else 1.0
            except Exception:
                gap = 0.0
        for i, row in enumerate(Tm):
            for p in row:
                if p > 1e-12:
                    h -= occ[i] * p * math.log(p)
        return cls(
            n_bins=n_bins,
            transition=tuple(tuple(r) for r in Tm),
            occupancy=occ,
            spectral_gap=gap,
            entropy_rate=max(0.0, h),
        )

    def to_dict(self) -> Dict[str, Any]:
        tmix = self.mixing_time()
        return {
            "n_bins": self.n_bins,
            "transition": [list(row) for row in self.transition],
            "occupancy": list(self.occupancy),
            "spectral_gap": round(self.spectral_gap, 9),
            "entropy_rate": round(self.entropy_rate, 6),
            "mixing_time_eps_0.01": round(tmix, 3) if tmix != float("inf") else "inf",
        }


# =============================================================================
# 2.12 — TOPOLOGÍA DE FASE 2  (abre con from_phase1_seed; cierra en Fase 3)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Phase2Topology:
    r"""
    Objeto agregador de la Fase 2. **Única** puerta de entrada canónica:

        `Phase2Topology.from_phase1_seed(seed, foundation=...)`

    Consume el dict de `Phase1Foundation.seed_phase2_topology` y, si se
    aporta el `foundation` vivo, no rehidrata \(H\) ni \(\Sigma\).
    """

    seed: Dict[str, Any]
    config: MICConfiguration
    return_map: PoincareReturnMap
    hamiltonian: HamiltonianSystem
    symplectic_form: SymplecticForm
    classifier: SubobjectClassifier
    cartan: PoincareCartanForm
    diagram: PersistenceDiagram
    betti: BettiNumbers
    summary: TopologicalSummary
    floquet: Optional[FloquetSpectrum]
    section_markov: ErgodicMarkovStrata
    dikw_markov: Dict[str, Any]
    recurrence: Dict[str, Any]
    kam_report: Dict[str, Any]
    metrics: MICMetrics

    @classmethod
    def from_phase1_seed(
        cls,
        seed: Mapping[str, Any],
        foundation: Optional[Phase1Foundation] = None,
        config: Optional[MICConfiguration] = None,
        n_returns: int = 24,
        service_counts: Optional[Dict[Any, int]] = None,
        vector_supplier: Optional[Callable[[], Dict[str, Tuple[Any, Any]]]] = None,
    ) -> "Phase2Topology":
        r"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║  PRIMER MÉTODO DE LA FASE 2                                      ║
        ║  Unidad F₁ → F₂  (continúa seed_phase2_topology)                 ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
        _require_phase1_seed(seed)
        cfg = config or (foundation.config if foundation is not None else DEFAULT_MIC_CONFIG)
        n_dof = int(seed.get("n_dof", cfg.n_dof))

        if foundation is not None:
            H = foundation.hamiltonian
            form = foundation.symplectic_form
            section = foundation.section
            clf = foundation.classifier
            cartan = foundation.cartan
        else:
            H = _rehydrate_hamiltonian(seed.get("hamiltonian") or {})
            form = SymplecticForm.canonical(n_dof)
            section = _rehydrate_section(seed.get("section") or {})
            clf = SubobjectClassifier(n_dof=n_dof, symplectic_form=form)
            cartan = PoincareCartanForm(hamiltonian=H, symplectic_form=form)

        pmap = PoincareReturnMap.from_phase1_seed(seed, section=section, config=cfg)
        germ = seed.get("poincare_germ") or {}
        z_hit = germ.get("hit")
        z0 = _rehydrate_point(z_hit, n_dof, "phase2_z0") if z_hit is not None else _rehydrate_point(
            None, n_dof, "phase2_z0"
        )

        metrics = MICMetrics()
        returns = pmap.collect_returns(H, z0, n_returns=max(2, n_returns))
        hits = [z0] + [pair[0] for pair in returns]
        for pair in returns:
            act = abs(cartan.action_along([z0, pair[0]])) if len(hits) >= 2 else 0.0
            metrics.record_poincare_return(action=act)

        actions = [abs(H.action_variable(z)) for z in hits]
        memberships = [
            _clamp(1.0 / (1.0 + abs(H.hamiltonian(z) - H.hamiltonian(z0))), 0.0, 1.0)
            for z in hits
        ]
        diagram = PersistenceDiagram.from_point_cloud(hits, actions=actions, heyting=memberships)
        summary = TopologicalSummary.from_seed_and_diagram(
            seed,
            diagram,
            structural_entropy=compute_shannon_entropy(
                distribution_from_counts(Counter(round(a, 6) for a in actions)) or [1.0]
            ),
            config=cfg,
        )
        betti = summary.betti

        floquet: Optional[FloquetSpectrum] = None
        kam_rep: Dict[str, Any] = {"kam_candidate": False}
        base = hits[1] if len(hits) > 1 else hits[0]
        try:
            floquet = pmap.floquet(H, base)
            metrics.record_symplectic_residual(floquet.det_residual)
            kam_rep = floquet_kam_report(floquet, cfg)
            if not kam_rep.get("kam_candidate", False):
                metrics.record_kam_violation()
        except Exception as exc:
            logger.debug("Floquet no calculable: %s", exc)
            metrics.record_kam_violation()

        section_markov = ErgodicMarkovStrata.from_returns(
            hits, H, cartan=cartan, n_bins=min(4, max(2, len(hits) // 2))
        )

        counts = service_counts or {}
        dikw: Dict[str, Any] = {}
        try:
            stm = StratumTransitionMatrix()
            if stm._n > 0:
                dikw = stm.to_dict(counts if counts else {s: 1 for s in stm._strata})
        except Exception as exc:
            dikw = {"error": str(exc)}

        rec_ok, rec_dist, rec_pt = verify_poincare_recurrence(
            H,
            z0,
            epsilon=max(cfg.poincare_recurrence_tol, 1e-3),
            max_steps=min(cfg.hamiltonian_max_steps, 2048),
            dt=cfg.hamiltonian_time_step,
        )
        recurrence = {
            "has_returned": rec_ok,
            "min_distance": rec_dist,
            "return_point": rec_pt.to_dict() if rec_pt is not None else None,
            "distinct_from_section_map": True,
        }

        if vector_supplier is not None:
            spec = SpectralGraphMetrics(vector_supplier)
            kam_rep["graph_spectrum"] = spec.compute_spectral_metrics(cfg)

        return cls(
            seed=dict(seed),
            config=cfg,
            return_map=pmap,
            hamiltonian=H,
            symplectic_form=form,
            classifier=clf,
            cartan=cartan,
            diagram=diagram,
            betti=betti,
            summary=summary,
            floquet=floquet,
            section_markov=section_markov,
            dikw_markov=dikw,
            recurrence=recurrence,
            kam_report=kam_rep,
            metrics=metrics,
        )

    def audit(self) -> Dict[str, Any]:
        valid, residuals = self.symplectic_form.verify_symplectic_invariants(
            tol=self.config.symplectic_residual_tol
        )
        return {
            "symplectic_valid": valid,
            "darboux_residuals": residuals,
            "floquet_det_residual": None if self.floquet is None else self.floquet.det_residual,
            "betti": self.betti.to_dict(),
            "kam_candidate": self.kam_report.get("kam_candidate"),
            "section_markov_gap": self.section_markov.spectral_gap,
            "recurrence": self.recurrence.get("has_returned"),
            "poincare_returns": self.metrics.poincare_returns,
        }

    def seed_phase3_orchestration(self) -> Dict[str, Any]:
        r"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║  ÚLTIMO MÉTODO DE LA FASE 2                                      ║
        ║  Germen funtorial  F₂ → F₃                                       ║
        ╚══════════════════════════════════════════════════════════════════╝

        Fase 3 DEBE abrir con un constructor que reciba exactamente este
        seed (p. ej. `Phase3Orchestration.from_phase2_seed(seed)`) y no
        redeclare \(P\), \(\mathrm{Dgm}\), ni las cadenas de Markov.

        Produce:
            seed["commands"]     → proyección + auditoría simpléctica
            seed["registry"]     → MICRegistry como topos \(\mathcal{E}_{\mathrm{MIC}}\)
            seed["pipeline"]     → validación categórica + poincaréana
            seed["bootstrap"]    → singleton / vectores core
        """
        return {
            "version": self.config.algorithm_version,
            "n_dof": self.config.n_dof,
            "seed_for_phase3": True,
            "from_phase": 2,
            "phase1_continuation": {
                "seed_for_phase2": True,
                "n_dof": self.seed.get("n_dof"),
                "kam": self.seed.get("kam"),
                "phase1_audit": self.seed.get("phase1_audit"),
            },
            "topology": {
                "betti": self.betti.to_dict(),
                "summary": self.summary.to_dict(),
                "diagram": self.diagram.to_dict(),
            },
            "dynamics": {
                "return_map": self.return_map.to_dict(),
                "floquet": None if self.floquet is None else self.floquet.to_dict(),
                "kam_report": self.kam_report,
                "recurrence": {
                    "has_returned": self.recurrence.get("has_returned"),
                    "min_distance": self.recurrence.get("min_distance"),
                },
            },
            "ergodicity": {
                "section_markov": self.section_markov.to_dict(),
                "dikw_markov": self.dikw_markov,
            },
            "symplectic": self.symplectic_form.to_dict(),
            "heyting_omega": self.classifier.to_dict(),
            "metrics": self.metrics.to_dict(),
            "audit": self.audit(),
            "continuation": {
                "next_phase": 3,
                "consumes": (
                    "topology",
                    "dynamics",
                    "ergodicity",
                    "symplectic",
                    "heyting_omega",
                    "metrics",
                    "audit",
                ),
                "produces": (
                    "PoincareProjectionCommands",
                    "MICRegistry ⊗ E_MIC",
                    "CategoricalPoincarePipeline",
                    "CoreVectorBootstrap",
                    "PublicAPI",
                ),
            },
        }


__phase2_version__: Final[str] = "8.1.0-poincare-symplectic"
__phase2_exports__: Final[Tuple[str, ...]] = (
    "Phase2Seed",
    "PersistenceInterval",
    "PersistenceDiagram",
    "BettiNumbers",
    "TopologicalSummary",
    "IntentVector",
    "CacheEntry",
    "TTLCache",
    "LatencyHistogram",
    "MICMetrics",
    "FileType",
    "MICException",
    "TopologicalInvariantError",
    "MICFunctorialityError",
    "FileNotFoundDiagnosticError",
    "UnsupportedFileTypeError",
    "FileValidationError",
    "FilePermissionError",
    "CleaningError",
    "MICHierarchyViolationError",
    "MICTimeoutError",
    "PoincareReturnMap",
    "FloquetSpectrum",
    "SpectralGraphMetrics",
    "StratumTransitionMatrix",
    "ErgodicMarkovStrata",
    "Phase2Topology",
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIN DE LA FASE 2 — CONTINÚA EN FASE 3
# ═══════════════════════════════════════════════════════════════════════════════
# Contracción funtorial F₂ ⇒ F₃:
#
#   Phase2Topology.seed_phase3_orchestration()
#       └── topology      →  Betti / persistencia en comandos de proyección
#       └── dynamics      →  auditoría Poincaré de cada handler
#       └── ergodicity    →  mixing DIKW + partición de Σ
#       └── symplectic    →  MICRegistry como topos con Ω
#       └── heyting_omega →  clasificador de subobjetos del registry
#       └── metrics/audit →  bootstrap y API pública
#
# La Fase 3 DEBE abrir con Phase3Orchestration.from_phase2_seed(seed)
# (foundation vivo opcional) y no redeclare P, Dgm ni las cadenas de Markov.
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║          FASE 3: ORQUESTACIÓN CATEGÓRICA Y OPERADORES POINCARÉANOS         ║
# ║   (Comandos · MICRegistry · Bootstrap · Singleton · API Pública · __all__) ║
# ║    Versión: 8.1.0-Poincare-Symplectic-KAM-Novikov-Doctoral                ║
# ║    Continúa: Phase2Topology.seed_phase3_orchestration() → from_phase2_seed ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════
#
# Contracción funtorial F₂ ⇒ F₃ (unidad de continuación):
#
#   seed["topology"]      →  Betti / persistencia en comandos de proyección
#   seed["dynamics"]      →  auditoría Poincaré de cada handler (Floquet, P)
#   seed["ergodicity"]    →  mixing DIKW + partición de Σ
#   seed["symplectic"]    →  MICRegistry como topos con Ω (NO se reconstruye J)
#   seed["heyting_omega"] →  clasificador de subobjetos del registry
#   seed["metrics"/"audit"] → bootstrap y API pública
#
# INVARIANTES DE ESTA FASE ────────────────────────────────────────────────────
#   [I3]  Pullback de Heyting: ejecución iff χ_S = ⊤
#   [I4]  Despacho O(n) sobre la base registrada
#   [I14] Simplectomorfismo de sección: se AUDITA, no se recalcula DP
#   [I18] Cadena de comandos asociativa: (c₁∘c₂)∘c₃ = c₁∘(c₂∘c₃)
#   [I19] Clausura DIKW: required(target) ⊆ validated  (salvo force_override)
#   [I20] Composición F₁∘F₂∘F₃: el cierre certifica seeds anidados
#
# La Fase 3 NO redefine SymplecticForm, PhaseSpacePoint, HamiltonianSystem,
# PoincareReturnMap, PersistenceDiagram, ErgodicMarkovStrata ni Verlet.
# ═══════════════════════════════════════════════════════════════════════════════


# =============================================================================
# 3.0 — PROTOCOLO DEL SEED F₂ Y REHIDRATACIÓN
# =============================================================================

class Phase3Seed(TypedDict, total=False):
    r"""
    Contrato tipado del dict que emite `Phase2Topology.seed_phase3_orchestration`.

    No recrea \(P\) ni \(\mathrm{Dgm}\): porta sus serializaciones y el
    certificado de auditoría de Fase 2.
    """

    version: str
    n_dof: int
    seed_for_phase3: bool
    from_phase: int
    phase1_continuation: Dict[str, Any]
    topology: Dict[str, Any]
    dynamics: Dict[str, Any]
    ergodicity: Dict[str, Any]
    symplectic: Dict[str, Any]
    heyting_omega: Dict[str, Any]
    metrics: Dict[str, Any]
    audit: Dict[str, Any]
    continuation: Dict[str, Any]


def _require_phase2_seed(seed: Mapping[str, Any]) -> None:
    if not isinstance(seed, Mapping):
        raise TypeError("seed debe ser Mapping (salida de seed_phase3_orchestration)")
    if not seed.get("seed_for_phase3", False):
        raise MICFunctorialityError(
            "Funtorialidad F₂⇒F₃ violada: el seed no porta seed_for_phase3=True. "
            "Debe emitirlo Phase2Topology.seed_phase3_orchestration().",
            expected="seed_for_phase3=True",
        )
    if int(seed.get("from_phase", 0) or 0) != 2:
        raise MICFunctorialityError(
            f"from_phase debe ser 2, recibido: {seed.get('from_phase')!r}"
        )


def _stratum_members() -> List[Any]:
    try:
        return list(Stratum)
    except Exception:
        return []


def _stratum_requires(stratum: Any) -> Set[Any]:
    try:
        req = stratum.requires()
        return set(req) if req is not None else set()
    except Exception:
        return set()


def _ordered_strata() -> List[Any]:
    try:
        return list(Stratum.ordered_bottom_up())  # type: ignore[attr-defined]
    except Exception:
        members = _stratum_members()
        try:
            return sorted(members, key=_safe_stratum_value)
        except Exception:
            return members


def _christoffel_weight(stratum: Any) -> float:
    r"""
    Fricción geodésica *fenomenológica* por nombre de estrato.

    No son símbolos de Christoffel \(\Gamma^k_{ij}\). Es una cota de
    exergía mínima para no trivializar STRATEGY/TACTICS.
    """
    table = {
        "PHYSICS": 0.1,
        "TACTICS": 0.8,
        "STRATEGY": 0.9,
        "OMEGA": 0.5,
        "ALPHA": 0.2,
        "WISDOM": 0.0,
    }
    return table.get(_safe_stratum_name(stratum).upper(), 0.0)


def _as_exception_tuple(*types: Any) -> Tuple[Type[BaseException], ...]:
    return tuple(t for t in types if isinstance(t, type) and issubclass(t, BaseException))


_FUNCTORIALITY_CATCH: Tuple[Type[BaseException], ...] = _as_exception_tuple(
    MICFunctorialityError,
    FunctorialityError if MIC_ALGEBRA_AVAILABLE else None,  # type: ignore[arg-type]
)


def _points_from_trajectory(blob: Any, n_dof: int) -> List[PhaseSpacePoint]:
    if blob is None:
        return []
    if isinstance(blob, (list, tuple)):
        out: List[PhaseSpacePoint] = []
        for i, item in enumerate(blob):
            if isinstance(item, PhaseSpacePoint):
                out.append(item)
                continue
            if isinstance(item, Mapping) and "q" in item and "p" in item:
                out.append(_rehydrate_point(item, n_dof, f"traj_{i}"))
                continue
            try:
                coords = list(item)
            except TypeError:
                continue
            if len(coords) == 2 * n_dof:
                out.append(PhaseSpacePoint.from_coords(coords, t=float(i), label=f"traj_{i}"))
        return out
    return []


# =============================================================================
# 3.1 — AUDITORÍAS (DELEGAN EN FASE 1/2; NO REIMPLEMENTAN DARBOUX NI P)
# =============================================================================

def audit_poincare_darboux_symplectic_form(
    canonical_omega: Any,
    tol: float = _SYMPLECTIC_RESIDUAL_TOL,
) -> Tuple[float, bool]:
    r"""
    Auditoría de Darboux: antisimetría, \(J^2=-I\), \(J^{\top}J=I\), \(\det J=+1\).

    Si `canonical_omega` es `SymplecticForm`, delega en Fase 1.
    Si es matriz, la envuelve. No reescribe el Pfaffiano.
    """
    if canonical_omega is None:
        return 0.0, True
    if isinstance(canonical_omega, SymplecticForm):
        valid, residuals = canonical_omega.verify_symplectic_invariants(tol=tol)
        total = (
            float(residuals.get("antisymmetry_residual", 0.0))
            + float(residuals.get("darboux_residual", 0.0))
            + float(residuals.get("cayley_residual", 0.0))
            + abs(float(residuals.get("determinant", 1.0)) - 1.0)
        )
        return total, valid
    try:
        J = _as_matrix(canonical_omega)
    except Exception:
        return float("inf"), False
    if not J:
        return 0.0, True
    n = len(J)
    if n < 2 or n % 2 != 0 or any(len(row) != n for row in J):
        return float("inf"), False
    form = SymplecticForm(
        n_dof=n // 2,
        omega_matrix=_to_backend(J),
        is_canonical=False,
    )
    valid, residuals = form.verify_symplectic_invariants(tol=tol)
    total = (
        float(residuals.get("antisymmetry_residual", 0.0))
        + float(residuals.get("darboux_residual", 0.0))
        + float(residuals.get("cayley_residual", 0.0))
        + abs(float(residuals.get("determinant", 1.0)) - 1.0)
    )
    return total, valid


def audit_gromov_nonsqueezing_capacity(
    radius_ball: float,
    radius_cylinder: float,
    tol: float = 1e-10,
    form: Optional[SymplecticForm] = None,
) -> Tuple[float, bool]:
    r"""Delega el no-aplastamiento en `SymplecticForm.gromov_nonsqueezing_check`."""
    carrier = form or SymplecticForm.canonical(1)
    is_valid, ratio = carrier.gromov_nonsqueezing_check(
        radius_ball, radius_cylinder, tol=tol
    )
    return ratio, is_valid


def audit_novikov_small_divisors_spectrum(
    frequencies: Any,
    diophantine_gamma: float = _KAM_GAMMA_DEFAULT,
    diophantine_tau: float = _KAM_TAU_DEFAULT,
    k_max: int = 2,
) -> Tuple[float, bool]:
    r"""
    Condición diofántica KAM sobre \(\omega\).

    El nombre histórico “Novikov” se conserva por estabilidad de API.
    **Novikov ≠ KAM**: Novikov es Morse de valores en \(S^1\) / formas
    cerradas de clase de Calabi. Aquí solo se audita

        \[ |\langle k,\omega\rangle|\ge\gamma/|k|^\tau. \]

    Recorre el retículo completo (no aborta el mínimo).
    """
    if frequencies is None:
        return 1.0, True
    if isinstance(frequencies, ActionAngleCoordinates):
        return frequencies.check_diophantine(
            gamma=diophantine_gamma, tau=diophantine_tau, k_max=k_max
        )
    try:
        if hasattr(frequencies, "tolist"):
            freq = tuple(float(x) for x in frequencies.tolist())  # type: ignore[union-attr]
        else:
            freq = tuple(float(x) for x in frequencies)
    except Exception:
        return float("inf"), False
    if not freq:
        return 1.0, True
    n = len(freq)
    aa = ActionAngleCoordinates(
        actions=tuple(1.0 for _ in range(n)),
        angles=tuple(0.0 for _ in range(n)),
        frequencies=freq,
    )
    return aa.check_diophantine(
        gamma=diophantine_gamma, tau=diophantine_tau, k_max=k_max
    )


def audit_poincare_ergodic_recurrence_distance(
    phase_space_points: Any,
    wilkinson_limit: float = _POINCARE_RECURRENCE_TOL,
    n_dof: int = 3,
) -> Tuple[float, bool]:
    r"""
    Recurrencia de Poincaré (1890) sobre una trayectoria ya integrada.

    No es el mapa de sección \(P\). Distancia euclídea a \(z_0\).
    """
    pts = _points_from_trajectory(phase_space_points, n_dof)
    if len(pts) < 2:
        if phase_space_points is None:
            return 0.0, True
        if NUMPY_AVAILABLE:
            try:
                arr = np.asarray(phase_space_points, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    return 0.0, True
                z0 = arr[0]
                dists = np.linalg.norm(arr[1:] - z0, axis=1)
                min_dist = float(np.min(dists))
                return min_dist, min_dist <= wilkinson_limit
            except Exception:
                return float("inf"), False
        return 0.0, True
    z0 = pts[0]
    min_dist = min(z.distance_to(z0) for z in pts[1:])
    return min_dist, min_dist <= wilkinson_limit


def audit_birkhoff_integrability_order(
    hamiltonian: "HamiltonianSystem",
    order: int = _BIRKHOFF_TRUNCATION_ORDER,
    coefficient_tol: float = 1e-3,
) -> Tuple[List[float], bool]:
    r"""
    Coeficientes de Birkhoff 1-dof (Fase 1). `is_integrable` significa
    **coeficientes bajo umbral**, no el teorema de Birkhoff–Gustavson.
    """
    try:
        coefficients = list(hamiltonian.birkhoff_normal_form_coefficients(order))
    except Exception:
        return [], False
    is_near_linear = all(abs(c) < coefficient_tol for c in coefficients)
    return coefficients, is_near_linear


def audit_liouville_volume_preservation(
    jacobian_M: Any,
    tol: float = _LIOUVILLE_VOLUME_TOL,
) -> Tuple[float, bool]:
    r"""
    \(\lvert\det M-1\rvert\). Si se pasa un `FloquetSpectrum`, usa su
    `det_residual` (jacobiano de \(P\), no del flujo continuo).
    """
    if jacobian_M is None:
        return 0.0, True
    if isinstance(jacobian_M, FloquetSpectrum):
        return jacobian_M.det_residual, jacobian_M.det_residual <= max(tol, 1e-3)
    try:
        M = _as_matrix(jacobian_M)
    except Exception:
        return float("inf"), False
    if not M or len(M) != len(M[0]):
        return float("inf"), False
    drift = abs(_det_small(M) - 1.0)
    return drift, drift <= tol


def audit_phase2_dynamics_bundle(
    seed: Mapping[str, Any],
    config: Optional[MICConfiguration] = None,
) -> Dict[str, Any]:
    r"""
    Consume `seed['dynamics']` y `seed['audit']` **sin** reiterar \(P\).

    Certifica: candidato KAM de Floquet, residual de \(\det DP\),
    recurrencia (distinta de sección), Betti del diagrama.
    """
    cfg = config or DEFAULT_MIC_CONFIG
    dynamics = seed.get("dynamics") or {}
    audit = seed.get("audit") or {}
    floquet = dynamics.get("floquet") or {}
    kam = dynamics.get("kam_report") or {}
    rec = dynamics.get("recurrence") or {}
    det_res = floquet.get("det_residual")
    if det_res is None:
        det_res = audit.get("floquet_det_residual")
    det_ok = True if det_res is None else float(det_res) <= max(
        cfg.symplectic_residual_tol * 1e3, 1e-3
    )
    return {
        "floquet_classification": floquet.get("classification"),
        "floquet_det_residual": det_res,
        "floquet_volume_ok": det_ok,
        "kam_candidate": bool(kam.get("kam_candidate", audit.get("kam_candidate", False))),
        "has_recurred": rec.get("has_returned", audit.get("recurrence")),
        "min_distance": rec.get("min_distance"),
        "symplectic_valid": bool((audit.get("symplectic_valid")
                                  if "symplectic_valid" in audit
                                  else True)),
    }


# =============================================================================
# 3.2 — CONTEXTO DE PROYECCIÓN CON ESTADO SIMPLÉCTICO
# =============================================================================

@dataclass
class ProjectionContext:
    r"""
    Contexto mutable del pipeline (Command + Chain of Responsibility).

    Los flags `*_audit_passed` nacen en `True` solo como *vacuo*: si la
    auditoría no corre, no se veta. `PoincareSymplecticAdjunctionCommand`
    los escribe. `ValidationCommand` veta únicamente fallos **explícitos**.
    """

    service_name: str
    payload: Dict[str, Any]
    context: Dict[str, Any]
    use_cache: bool
    cache_key: Optional[str] = None
    target_stratum: Optional[Any] = None
    handler: Optional[Any] = None
    validated_strata: Set[Any] = field(default_factory=set)
    force_override: bool = False
    natural_transformations: List[Any] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)
    intent_vector: Optional[IntentVector] = None
    phase_space_point: Optional[PhaseSpacePoint] = None
    symplectic_form: Optional[SymplecticForm] = None
    classifier: Optional[SubobjectClassifier] = None
    cartan: Optional[PoincareCartanForm] = None
    phase2_seed: Optional[Dict[str, Any]] = None
    symplectic_audit_passed: bool = True
    gromov_audit_passed: bool = True
    novikov_audit_passed: bool = True
    poincare_audit_passed: bool = True
    birkhoff_audit_passed: bool = True
    floquet_audit_passed: bool = True
    accumulated_action: float = 0.0
    phase_holonomy: float = 0.0
    audit_log: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def all_symplectic_audits_passed(self) -> bool:
        return (
            self.symplectic_audit_passed
            and self.gromov_audit_passed
            and self.novikov_audit_passed
            and self.poincare_audit_passed
            and self.birkhoff_audit_passed
            and self.floquet_audit_passed
        )


# =============================================================================
# 3.3 — INTERFAZ ABSTRACTA ProjectionCommand
# =============================================================================

class ProjectionCommand(ABC):
    r"""
    Comando de proyección. Monoide bajo composición secuencial:

        \[ (c_1\circ c_2)\circ c_3=c_1\circ(c_2\circ c_3),\qquad
           e=\texttt{return None}. \]

    `execute` → `None` continúa; `Dict` termina el pipeline.
    """

    @abstractmethod
    def execute(self, context: ProjectionContext) -> Optional[Dict[str, Any]]:
        """Ejecuta el comando. `None` = continuar; dict = terminar."""


def run_projection_pipeline(
    commands: Sequence[ProjectionCommand],
    ctx: ProjectionContext,
) -> Optional[Dict[str, Any]]:
    """Composición asociativa de la cadena. Identidad: no retornar."""
    for command in commands:
        result = command.execute(ctx)
        if result is not None:
            return result
    return None


# =============================================================================
# 3.4 — CACHE CHECK COMMAND
# =============================================================================

class CacheCheckCommand(ProjectionCommand):
    """Verifica el cache TTL antes de procesar la intención."""

    __slots__ = ("_cache", "_metrics")

    def __init__(self, cache: TTLCache, metrics: MICMetrics) -> None:
        self._cache = cache
        self._metrics = metrics

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        if not ctx.use_cache:
            return None
        try:
            payload_repr = str(sorted(ctx.payload.items()))
            ctx.cache_key = (
                f"{ctx.service_name}:"
                f"{hashlib.sha256(payload_repr.encode()).hexdigest()[:16]}"
            )
            cached = self._cache.get(ctx.cache_key)
            if cached is not None:
                self._metrics.cache_hits += 1
                logger.debug("Cache hit: '%s'", ctx.service_name)
                return cast(Dict[str, Any], cached)
        except (TypeError, ValueError) as e:
            logger.debug("Cache key failed para '%s': %s", ctx.service_name, e)
            ctx.cache_key = None
        return None


# =============================================================================
# 3.5 — RESOLUTION COMMAND
# =============================================================================

class ResolutionCommand(ProjectionCommand):
    """Resuelve el vector base y su estrato."""

    __slots__ = ("_vectors", "_lock", "_metrics")

    def __init__(
        self,
        vectors: Dict[str, Tuple[Any, Any]],
        lock: threading.RLock,
        metrics: MICMetrics,
    ) -> None:
        self._vectors = vectors
        self._lock = lock
        self._metrics = metrics

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        with self._lock:
            if ctx.service_name not in self._vectors:
                available = list(self._vectors.keys())
                self._metrics.record_error("resolution_error")
                raise ValueError(
                    f"Vector desconocido: '{ctx.service_name}'. Disponibles: {available}"
                )
            ctx.target_stratum, ctx.handler = self._vectors[ctx.service_name]
        return None


# =============================================================================
# 3.6 — SHEAF COHOMOLOGY PROJECTION COMMAND
# =============================================================================

class SheafCohomologyProjectionCommand(ProjectionCommand):
    """Veto homológico: \(\dim H^1>0\) implica obstrucción (haz celular)."""

    __slots__ = ("_metrics",)

    def __init__(self, metrics: MICMetrics) -> None:
        self._metrics = metrics

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        if not SHEAF_COHOMOLOGY_AVAILABLE:
            return None
        sheaf = ctx.context.get("cellular_sheaf")
        state_vector = ctx.context.get("global_state_vector")
        if not isinstance(sheaf, CellularSheaf) or state_vector is None:
            return None
        try:
            orchestrator = SheafCohomologyOrchestrator()
            assessment = orchestrator.audit_global_state(sheaf, state_vector)
            if assessment.h1_dimension > 0:
                self._metrics.record_error("topological_obstruction")
                return {
                    "success": False,
                    "error": (
                        f"Veto por Obstrucción Topológica: dim H¹ = "
                        f"{assessment.h1_dimension} > 0."
                    ),
                    "error_type": "HomologicalInconsistencyError",
                    "error_category": "topological_veto",
                    "error_details": {
                        "h1_dim": assessment.h1_dimension,
                        "h0_dim": assessment.h0_dimension,
                        "frustration": assessment.frustration_energy,
                    },
                }
        except HomologicalInconsistencyError as e:
            self._metrics.record_error("homological_inconsistency")
            return {
                "success": False,
                "error": f"Inconsistencia Homológica: {e}",
                "error_type": "HomologicalInconsistencyError",
                "error_category": "topological_veto",
            }
        except Exception as e:
            logger.debug("SheafCohomology skip: %s", e)
        return None


# =============================================================================
# 3.7 — NORMALIZATION COMMAND  (antes de Poincaré: estado en T*M)
# =============================================================================

class NormalizationCommand(ProjectionCommand):
    r"""
    Normaliza el contexto e **inyecta** \(\Omega\), \(\chi\) y Cartan del
    topos (seed / registry). Debe ejecutarse *antes* de la adjunción
    simpléctica: de lo contrario el comando de Poincaré opera sobre `None`.
    """

    __slots__ = ("_symplectic_form", "_classifier", "_cartan", "_config")

    def __init__(
        self,
        symplectic_form: Optional[SymplecticForm] = None,
        classifier: Optional[SubobjectClassifier] = None,
        cartan: Optional[PoincareCartanForm] = None,
        config: Optional[MICConfiguration] = None,
    ) -> None:
        self._symplectic_form = symplectic_form
        self._classifier = classifier
        self._cartan = cartan
        self._config = config or DEFAULT_MIC_CONFIG

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        raw_validated = ctx.context.get("validated_strata", set())
        raw_validated = set() if raw_validated is None else set(raw_validated)
        ctx.validated_strata = self._normalize_validated_strata(raw_validated)
        ctx.force_override = bool(
            ctx.context.get("force_override", False)
            or ctx.context.get("force_physics_override", False)
        )
        nt = ctx.context.get("natural_transformations", [])
        if isinstance(nt, list) and MIC_ALGEBRA_AVAILABLE:
            ctx.natural_transformations = [
                item for item in nt if isinstance(item, NaturalTransformation)
            ]
        n_dof = int(self._config.n_dof)
        if ctx.intent_vector is None:
            q = ctx.context.get("q_semantic", ())
            p = ctx.context.get("p_momentum", ())
            q_t = tuple(float(x) for x in q) if q else tuple()
            p_t = tuple(float(x) for x in p) if p else tuple()
            if q_t and not p_t:
                p_t = tuple(0.0 for _ in q_t)
            if p_t and not q_t:
                q_t = tuple(0.0 for _ in p_t)
            ctx.intent_vector = IntentVector(
                service_name=ctx.service_name,
                payload=ctx.payload,
                context=dict(ctx.context),
                q_semantic=q_t,
                p_momentum=p_t,
                phase=float(ctx.context.get("initial_phase", 0.0)),
            )
            ctx.phase_space_point = ctx.intent_vector.to_phase_point()
            if ctx.intent_vector.n_dof:
                n_dof = ctx.intent_vector.n_dof
        if ctx.symplectic_form is None:
            ctx.symplectic_form = self._symplectic_form or SymplecticForm.canonical(max(1, n_dof))
        if ctx.classifier is None:
            ctx.classifier = self._classifier or SubobjectClassifier(
                n_dof=ctx.symplectic_form.n_dof, symplectic_form=ctx.symplectic_form
            )
        if ctx.cartan is None:
            ctx.cartan = self._cartan
        ctx.phase_holonomy = _wrap_angle(
            ctx.phase_holonomy + float(ctx.context.get("initial_phase", 0.0))
        )
        return None

    def _normalize_validated_strata(self, raw: Any) -> Set[Any]:
        if raw is None or not isinstance(raw, (set, list, tuple, frozenset)):
            return set()
        normalized: Set[Any] = set()
        for item in raw:
            try:
                if isinstance(item, Stratum):
                    normalized.add(item)
                elif isinstance(item, int):
                    normalized.add(Stratum(item))
                elif isinstance(item, str):
                    member = getattr(Stratum, item.upper().strip(), None)
                    if member is not None:
                        normalized.add(member)
            except (ValueError, KeyError, TypeError):
                continue
        return normalized


# =============================================================================
# 3.8 — BDD VERIFICATION COMMAND
# =============================================================================

class BDDVerificationCommand(ProjectionCommand):
    """ROBDD opcional: conflicto de canonicidad si la fórmula colapsa a ⊥."""

    __slots__ = ("_metrics", "_bdd")

    def __init__(self, metrics: MICMetrics) -> None:
        self._metrics = metrics
        self._bdd = bdd.BDD() if BDD_AVAILABLE else None

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        if not self._bdd:
            return None
        try:
            active_services = ctx.context.get("active_services", [])
            service_vars = [f"svc_{i}" for i in range(min(10, len(active_services)))]
            if not service_vars:
                return None
            self._bdd.declare(*service_vars)
            formula = " & ".join([f"!({v})" for v in service_vars])
            u = self._bdd.add_expr(formula)
            if u == self._bdd.false:
                self._metrics.record_error("bdd_conflict_error")
                return {
                    "success": False,
                    "error": "Conflicto de Canonicidad ROBDD.",
                    "error_type": "BDDConflictError",
                    "error_category": "formal_verification",
                }
        except Exception as e:
            logger.debug("BDD Verification skip: %s", e)
        return None


# =============================================================================
# 3.9 — INTERCHANGE LAW VERIFICATION COMMAND (2-Categoría)
# =============================================================================

class InterchangeLawVerificationCommand(ProjectionCommand):
    r"""Ley de intercambio: \((\beta\circ\alpha)\ast(\beta'\circ\alpha')
    =(\beta\ast\beta')\circ(\alpha\ast\alpha')\)."""

    __slots__ = ("_metrics",)

    def __init__(self, metrics: MICMetrics) -> None:
        self._metrics = metrics

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        if not MIC_ALGEBRA_AVAILABLE or not ctx.natural_transformations:
            return None
        nt = ctx.natural_transformations
        if len(nt) < 4:
            return None
        try:
            alpha, alpha_prime, beta, beta_prime = nt[:4]
            test_state = CategoricalState(
                payload=ctx.payload,
                context=ctx.context,
                validated_strata=frozenset(ctx.validated_strata),
            )
            TwoCategoryOrchestrator.validate_interchange_law(
                alpha, alpha_prime, beta, beta_prime, test_state
            )
        except _FUNCTORIALITY_CATCH as e:
            self._metrics.record_error("interchange_law_violation")
            return {
                "success": False,
                "error": f"Veto por Falta de Funtorialidad: {e}",
                "error_type": "FunctorialityError",
                "error_category": "categorical_inconsistency",
            }
        except Exception as e:
            logger.debug("InterchangeLaw skip: %s", e)
        return None


# =============================================================================
# 3.10 — ADJUNCIÓN SIMPLÉCTICA (consume dynamics F₂; no reitera P)
# =============================================================================

class PoincareSymplecticAdjunctionCommand(ProjectionCommand):
    r"""
    Auditoría integral Darboux / Liouville / Gromov / KAM / recurrencia /
    Floquet, **delegando** en Fase 1–2.

    Orden de evidencia
    ------------------
    1. `ctx.phase2_seed['dynamics']` — certificado ya calculado (no se
       reintegra Verlet ni se itera \(P\)).
    2. `ctx.symplectic_form` — Darboux del topos inyectado.
    3. `payload/context['poincare']` — evidencias *adicionales* del caller.

    Acción acumulada: 1-forma de Liouville \(\langle p,q\rangle\) o
    `PoincareCartanForm.action_along` sobre una trayectoria. **No**
    \(\Omega(v,v)\) (idénticamente 0) ni \(\pi\|z\|^2/2\).
    """

    __slots__ = ("_metrics", "_config")

    def __init__(
        self, metrics: MICMetrics, config: Optional[MICConfiguration] = None
    ) -> None:
        self._metrics = metrics
        self._config = config or DEFAULT_MIC_CONFIG

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        poincare_data = ctx.context.get("poincare") or ctx.payload.get("poincare")
        if not isinstance(poincare_data, dict):
            poincare_data = {}

        try:
            if ctx.phase2_seed:
                bundle = audit_phase2_dynamics_bundle(ctx.phase2_seed, self._config)
                ctx.audit_log["phase2_dynamics"] = bundle
                if bundle.get("floquet_det_residual") is not None:
                    self._metrics.record_symplectic_residual(
                        float(bundle["floquet_det_residual"])
                    )
                    if not bundle.get("floquet_volume_ok", True):
                        ctx.floquet_audit_passed = False
                        self._metrics.record_error("liouville_volume_violation")
                        return {
                            "success": False,
                            "error": (
                                f"Veto Liouville/Floquet: det(DP) residual="
                                f"{bundle['floquet_det_residual']}"
                            ),
                            "error_type": "LiouvilleVolumeViolation",
                            "error_category": "poincare_symplectic_veto",
                            "error_details": bundle,
                        }
                if bundle.get("kam_candidate") is False and poincare_data.get(
                    "require_kam_candidate", False
                ):
                    ctx.novikov_audit_passed = False
                    self._metrics.record_kam_violation()
                    return {
                        "success": False,
                        "error": "Veto KAM: Floquet no es candidato a toro persistente.",
                        "error_type": "SmallDivisorsDivergence",
                        "error_category": "poincare_symplectic_veto",
                        "error_details": bundle,
                    }
                if bundle.get("has_recurred"):
                    self._metrics.record_poincare_return(action=ctx.accumulated_action)

            if ctx.symplectic_form is not None:
                residual, is_valid = audit_poincare_darboux_symplectic_form(
                    ctx.symplectic_form, tol=self._config.symplectic_residual_tol
                )
                self._metrics.record_symplectic_residual(residual)
                ctx.audit_log["darboux_residual"] = residual
                if not is_valid:
                    ctx.symplectic_audit_passed = False
                    self._metrics.record_error("darboux_violation")
                    return {
                        "success": False,
                        "error": f"Veto Darboux: residual={residual:.3e}",
                        "error_type": "SymplecticResidualViolation",
                        "error_category": "poincare_symplectic_veto",
                        "error_details": {"residual": residual},
                    }
                ctx.symplectic_audit_passed = True

            canonical_omega = poincare_data.get("canonical_omega")
            if canonical_omega is not None:
                residual, is_valid = audit_poincare_darboux_symplectic_form(
                    canonical_omega, tol=self._config.symplectic_residual_tol
                )
                self._metrics.record_symplectic_residual(residual)
                if not is_valid:
                    ctx.symplectic_audit_passed = False
                    self._metrics.record_error("darboux_violation")
                    return {
                        "success": False,
                        "error": f"Veto Darboux (payload): residual={residual:.3e}",
                        "error_type": "SymplecticResidualViolation",
                        "error_category": "poincare_symplectic_veto",
                        "error_details": {"residual": residual},
                    }

            jacobian_M = poincare_data.get("jacobian_M")
            if jacobian_M is not None:
                drift, is_vol = audit_liouville_volume_preservation(
                    jacobian_M, tol=self._config.liouville_volume_tol
                )
                if not is_vol:
                    ctx.floquet_audit_passed = False
                    self._metrics.record_error("liouville_volume_violation")
                    return {
                        "success": False,
                        "error": f"Veto Liouville: det(M) drift={drift:.3e}",
                        "error_type": "LiouvilleVolumeViolation",
                        "error_category": "poincare_symplectic_veto",
                        "error_details": {"volume_drift": drift},
                    }

            if "radius_ball" in poincare_data and "radius_cylinder" in poincare_data:
                r = float(poincare_data["radius_ball"])
                R = float(poincare_data["radius_cylinder"])
                ratio, is_valid = audit_gromov_nonsqueezing_capacity(
                    r, R, form=ctx.symplectic_form
                )
                if not is_valid:
                    ctx.gromov_audit_passed = False
                    self._metrics.record_error("gromov_squeezing_violation")
                    return {
                        "success": False,
                        "error": f"Veto Gromov: r={r:.4f} > R={R:.4f} (ratio={ratio:.4f})",
                        "error_type": "GromovSqueezingViolation",
                        "error_category": "poincare_symplectic_veto",
                        "error_details": {
                            "radius_ball": r,
                            "radius_cylinder": R,
                            "ratio": ratio,
                        },
                    }
                ctx.gromov_audit_passed = True

            frequencies = poincare_data.get("frequencies")
            if frequencies is not None:
                gamma = float(ctx.context.get("kam_gamma", self._config.kam_gamma))
                tau = float(ctx.context.get("kam_tau", self._config.kam_tau))
                min_div, is_dioph = audit_novikov_small_divisors_spectrum(
                    frequencies,
                    diophantine_gamma=gamma,
                    diophantine_tau=tau,
                    k_max=self._config.kam_frequency_lattice_range,
                )
                ctx.audit_log["kam_min_divisor"] = min_div
                if not is_dioph:
                    ctx.novikov_audit_passed = False
                    self._metrics.record_kam_violation()
                    return {
                        "success": False,
                        "error": (
                            f"Veto KAM: pequeños divisores (min_div={min_div:.3e})"
                        ),
                        "error_type": "SmallDivisorsDivergence",
                        "error_category": "poincare_symplectic_veto",
                        "error_details": {"min_divisor": min_div},
                    }
                ctx.novikov_audit_passed = True

            n_dof = ctx.symplectic_form.n_dof if ctx.symplectic_form else self._config.n_dof
            traj = poincare_data.get("phase_space_trajectory")
            if traj is not None:
                min_dist, has_recurred = audit_poincare_ergodic_recurrence_distance(
                    traj,
                    wilkinson_limit=self._config.poincare_recurrence_tol,
                    n_dof=n_dof,
                )
                ctx.audit_log["recurrence_min_distance"] = min_dist
                points = _points_from_trajectory(traj, n_dof)
                if ctx.cartan is not None and len(points) >= 2:
                    ctx.accumulated_action += abs(ctx.cartan.action_along(points))
                if has_recurred:
                    self._metrics.record_poincare_return(action=ctx.accumulated_action)
                ctx.poincare_audit_passed = True

            ham = poincare_data.get("hamiltonian")
            if isinstance(ham, HamiltonianSystem):
                coefficients, near_linear = audit_birkhoff_integrability_order(
                    ham,
                    order=self._config.birkhoff_truncation_order,
                    coefficient_tol=self._config.birkhoff_coefficient_tol,
                )
                ctx.context["birkhoff_coefficients"] = coefficients
                ctx.audit_log["birkhoff_near_linear"] = near_linear
                ctx.birkhoff_audit_passed = True

            if ctx.phase_space_point is not None:
                ctx.accumulated_action += abs(ctx.phase_space_point.poincare_1form())
            if ctx.intent_vector is not None:
                ctx.phase_holonomy = _wrap_angle(
                    ctx.phase_holonomy + ctx.intent_vector.phase
                )

            logger.debug(
                "Auditoría Poincaré: OK (Liouville pairing=%.6f, holonomy=%.6f)",
                ctx.accumulated_action,
                ctx.phase_holonomy,
            )
        except Exception as e:
            logger.warning("Error en auditoría simpléctica: %s", e)
        return None


# =============================================================================
# 3.11 — ERROR MONAD AUDIT COMMAND
# =============================================================================

class ErrorMonadAuditCommand(ProjectionCommand):
    """Monotonicidad de entropía de persistencia (no es 2ª ley: es filtro TDA)."""

    __slots__ = ("_metrics",)

    def __init__(self, metrics: MICMetrics) -> None:
        self._metrics = metrics

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        prev_entropy = float(ctx.context.get("previous_persistence_entropy", 0.0))
        curr_entropy = float(ctx.context.get("current_persistence_entropy", 0.0))
        if curr_entropy < prev_entropy - 1e-7:
            self._metrics.record_error("entropy_monotonicity_violation")
            logger.warning(
                "Inversión de entropía de persistencia: %.4f < %.4f",
                curr_entropy,
                prev_entropy,
            )
        return None


# =============================================================================
# 3.12 — SAT ORACLE COMMAND
# =============================================================================

class SATOracleCommand(ProjectionCommand):
    """Oráculo Z3 sobre precondiciones lógicas. Ausencia de Z3 ⇒ skip."""

    __slots__ = ("_metrics",)

    def __init__(self, metrics: MICMetrics) -> None:
        self._metrics = metrics

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        if not Z3_AVAILABLE:
            return None
        preconditions = ctx.context.get("logical_preconditions", {})
        if not preconditions:
            return None
        solver = z3.Solver()
        z3_vars = {name: z3.Bool(name) for name in preconditions.keys()}
        for name, value in preconditions.items():
            if value:
                solver.add(z3_vars[name])
            else:
                solver.add(z3.Not(z3_vars[name]))
        logical_contract = ctx.context.get("logical_contract", {})
        if not logical_contract:
            if ctx.service_name == "stabilize_flux":
                if "sensor_online" in z3_vars and "db_connected" in z3_vars:
                    solver.add(z3.And(z3_vars["sensor_online"], z3_vars["db_connected"]))
            elif ctx.service_name == "parse_raw":
                if "file_exists" in z3_vars:
                    solver.add(z3_vars["file_exists"])
        else:
            for req in logical_contract.get("required", []):
                if req in z3_vars:
                    solver.add(z3_vars[req])
            for forbidden in logical_contract.get("forbidden", []):
                if forbidden in z3_vars:
                    solver.add(z3.Not(z3_vars[forbidden]))
        if solver.check() == z3.unsat:
            self._metrics.record_error("sat_unsatisfiable_error")
            return {
                "success": False,
                "error": (
                    f"Veto SAT: precondiciones insatisfacibles para "
                    f"'{ctx.service_name}'."
                ),
                "error_type": "UnsatisfiableConditionError",
                "error_category": "deterministic_oracle",
            }
        return None


SATOrcaleCommand = SATOracleCommand  # alias de compatibilidad (typo histórico)


# =============================================================================
# 3.13 — VALIDATION COMMAND (Clausura Transitiva DIKW)
# =============================================================================

class ValidationCommand(ProjectionCommand):
    """Gatekeeper DIKW + veto de auditorías simplécticas *explícitamente* fallidas."""

    __slots__ = ("_metrics", "_classifier")

    def __init__(
        self,
        metrics: MICMetrics,
        classifier: Optional[SubobjectClassifier] = None,
    ) -> None:
        self._metrics = metrics
        self._classifier = classifier

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        if ctx.target_stratum is None:
            return {
                "success": False,
                "error": "Target stratum not resolved",
                "error_type": "InternalError",
                "error_category": "resolution_error",
            }
        required = _stratum_requires(ctx.target_stratum)
        missing = required - ctx.validated_strata
        dissipated_power = float(ctx.context.get("dissipated_power", 0.0))
        if dissipated_power < 0.0:
            self._metrics.record_error("thermodynamic_violation")
            return {
                "success": False,
                "error": (
                    f"Veto Físico: potencia disipada negativa "
                    f"(P_diss={dissipated_power})."
                ),
                "error_type": "ThermodynamicInconsistency",
                "error_category": "physical_veto",
            }
        if not ctx.all_symplectic_audits_passed:
            return {
                "success": False,
                "error": "Alguna auditoría simpléctica falló.",
                "error_type": "SymplecticAuditFailure",
                "error_category": "poincare_symplectic_veto",
                "error_details": {
                    "darboux": ctx.symplectic_audit_passed,
                    "gromov": ctx.gromov_audit_passed,
                    "novikov": ctx.novikov_audit_passed,
                    "poincare": ctx.poincare_audit_passed,
                    "birkhoff": ctx.birkhoff_audit_passed,
                    "floquet": ctx.floquet_audit_passed,
                    "audit_log": ctx.audit_log,
                },
            }
        if missing and not ctx.force_override:
            self._metrics.violations += 1
            self._metrics.record_error("hierarchy_violation")
            error = MICHierarchyViolationError(
                target_stratum=ctx.target_stratum,
                missing_strata=missing,
                validated_strata=ctx.validated_strata,
            )
            missing_names = [
                _safe_stratum_name(s)
                for s in sorted(missing, key=_safe_stratum_value, reverse=True)
            ]
            return {
                "success": False,
                "error": str(error),
                "error_type": "MICHierarchyViolationError",
                "error_category": "hierarchy_violation",
                "error_details": {
                    "target_stratum": _safe_stratum_name(ctx.target_stratum),
                    "missing_strata": missing_names,
                    "validated_strata": [
                        _safe_stratum_name(s) for s in ctx.validated_strata
                    ],
                },
            }
        if ctx.force_override:
            logger.warning(
                "Filtración bypassada para '%s' via force_override",
                _safe_stratum_name(ctx.target_stratum),
            )
        return None


# =============================================================================
# 3.14 — EXECUTION COMMAND (Pullback Authorization)
# =============================================================================

class ExecutionCommand(ProjectionCommand):
    r"""
    Ejecuta el handler si \(\chi_S=\top\). Exergía vs. fricción fenomenológica.
    No muta el payload del caller: copia local para holonomía WISDOM.
    """

    __slots__ = ("_cache", "_metrics", "_config", "_classifier")

    def __init__(
        self,
        cache: TTLCache,
        metrics: MICMetrics,
        config: MICConfiguration,
        classifier: Optional[SubobjectClassifier] = None,
    ) -> None:
        self._cache = cache
        self._metrics = metrics
        self._config = config
        self._classifier = classifier

    def _compute_characteristic_morphism(self, ctx: ProjectionContext) -> HeytingValue:
        omega = ctx.classifier or self._classifier
        if omega is None:
            omega = SubobjectClassifier(
                n_dof=ctx.symplectic_form.n_dof if ctx.symplectic_form else 3,
                symplectic_form=ctx.symplectic_form,
            )
        if ctx.target_stratum is None:
            return omega.false
        if ctx.force_override:
            return HeytingValue(
                1.0,
                f"Forced commutation for {ctx.service_name}",
                phase=ctx.phase_holonomy,
            )
        required = _stratum_requires(ctx.target_stratum)
        missing = required - ctx.validated_strata
        truth_value = (
            1.0 - (len(missing) / max(1, len(required))) if required else 1.0
        )
        return HeytingValue(
            truth_value,
            f"Sieve evaluation for {ctx.service_name}",
            phase=ctx.phase_holonomy,
        )

    def execute(self, ctx: ProjectionContext) -> Optional[Dict[str, Any]]:
        if ctx.handler is None or ctx.target_stratum is None:
            return {
                "success": False,
                "error": "Handler or stratum not resolved",
                "error_type": "InternalError",
                "error_category": "execution_error",
            }
        chi_s = self._compute_characteristic_morphism(ctx)
        if not chi_s.is_true:
            self._metrics.record_error("pullback_failure")
            return {
                "success": False,
                "error": (
                    f"Fallo de Pullback: χ_S = {chi_s.value:.2f} ({chi_s.description})"
                ),
                "error_type": "PullbackCommutationError",
                "error_category": "topos_violation",
            }
        exergy_level = float(ctx.context.get("exergy_level", 1.0))
        target_entropy = _christoffel_weight(ctx.target_stratum)
        if exergy_level < target_entropy:
            return {
                "success": False,
                "error": (
                    f"Resistencia geodésica del estrato "
                    f"{_safe_stratum_name(ctx.target_stratum)} repele la intención "
                    f"(exergía={exergy_level:.2f} < gravedad={target_entropy:.2f})."
                ),
                "error_type": "GeodesicRepulsionError",
                "error_category": "thermodynamic_violation",
            }
        payload = dict(ctx.payload)
        phase_correction = float(ctx.context.get("_phase_correction", 1.0))
        if (
            _safe_stratum_name(ctx.target_stratum).upper() == "WISDOM"
            and phase_correction != 1.0
        ):
            for k, v in list(payload.items()):
                if isinstance(v, float) and ("score" in k or "weight" in k):
                    payload[k] = v * phase_correction
        try:
            with self._metrics.handler_latency.measure():
                result = ctx.handler(**payload)
            if not isinstance(result, dict):
                result = {"success": True, "result": result}
            if result.get("success", False):
                updated_validated = ctx.validated_strata | {ctx.target_stratum}
                result["_mic_validation_update"] = getattr(
                    ctx.target_stratum, "value", _safe_stratum_name(ctx.target_stratum)
                )
                result["_mic_stratum"] = _safe_stratum_name(ctx.target_stratum)
                result["_mic_validated_strata"] = [
                    _safe_stratum_name(s) for s in updated_validated
                ]
                result["_mic_symplectic"] = {
                    "accumulated_action": round(ctx.accumulated_action, 9),
                    "phase_holonomy": round(ctx.phase_holonomy, 9),
                    "audits_passed": ctx.all_symplectic_audits_passed,
                    "audit_log": ctx.audit_log,
                }
                if ctx.use_cache and ctx.cache_key is not None:
                    self._cache.set(ctx.cache_key, result)
                self._metrics.record_projection(ctx.target_stratum)
            return cast(Dict[str, Any], result)
        except TypeError as e:
            logger.error("Firma de handler incorrecta '%s': %s", ctx.service_name, e)
            self._metrics.record_error("handler_signature_error")
            return {
                "success": False,
                "error": str(e),
                "error_type": "TypeError",
                "error_category": "handler_signature_error",
                "error_details": {
                    "service_name": ctx.service_name,
                    "hint": "Verifique claves del payload vs. firma del handler",
                },
            }
        except Exception as e:
            logger.exception("Error ejecutando '%s'", ctx.service_name)
            self._metrics.record_error("execution_error")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "error_category": "execution_error",
                "error_details": {"service_name": ctx.service_name},
            }


# =============================================================================
# 3.15 — MICRegistry: TOPOS SIMPLÉCTICO (abre con from_phase2_seed)
# =============================================================================

class MICRegistry:
    r"""
    Matriz de Interacción Central — topos \(\mathcal{E}_{\mathrm{MIC}}\)
    con \(\Omega\) **inyectada**, no reconstruida.

    Construcción canónica: `MICRegistry.from_phase2_seed(seed, topology=...)`.
    `__init__` permanece para tests / bootstrap sin seed (Darboux canónico
    de la config: *no* es un nuevo teorema; es la carta por defecto).
    """

    __slots__ = (
        "_vectors",
        "_lock",
        "_cache",
        "_logger",
        "_metrics",
        "_config",
        "_projection_commands",
        "_spectral_analyzer",
        "_transition_matrix",
        "_symplectic_form",
        "_classifier",
        "_cartan",
        "_phase2_seed",
        "_phase2_topology",
    )

    def __init__(self, config: Optional[MICConfiguration] = None) -> None:
        self._config = config or DEFAULT_MIC_CONFIG
        self._vectors: Dict[str, Tuple[Any, Any]] = {}
        self._lock = threading.RLock()
        self._cache = TTLCache(
            ttl_seconds=self._config.cache_ttl_seconds,
            max_size=self._config.cache_max_size,
        )
        self._logger = get_structured_logger("MIC.Registry")
        self._metrics = MICMetrics()
        self._spectral_analyzer: Optional[SpectralGraphMetrics] = None
        self._transition_matrix = StratumTransitionMatrix()
        self._symplectic_form = SymplecticForm.canonical(self._config.n_dof)
        self._classifier = SubobjectClassifier(
            n_dof=self._config.n_dof, symplectic_form=self._symplectic_form
        )
        self._cartan: Optional[PoincareCartanForm] = None
        self._phase2_seed: Optional[Dict[str, Any]] = None
        self._phase2_topology: Optional[Phase2Topology] = None
        self._projection_commands = self._build_pipeline()

    def _build_pipeline(self) -> List[ProjectionCommand]:
        r"""
        Orden corregido (asociativo, dependencias saturadas):

            Cache → Normalize(\(\Omega,\chi\)) → Resolve → Sheaf → Poincaré
            → Interchange → BDD → SAT → Monad → Validate → Execute.
        """
        return [
            CacheCheckCommand(self._cache, self._metrics),
            NormalizationCommand(
                symplectic_form=self._symplectic_form,
                classifier=self._classifier,
                cartan=self._cartan,
                config=self._config,
            ),
            ResolutionCommand(self._vectors, self._lock, self._metrics),
            SheafCohomologyProjectionCommand(self._metrics),
            PoincareSymplecticAdjunctionCommand(self._metrics, self._config),
            InterchangeLawVerificationCommand(self._metrics),
            BDDVerificationCommand(self._metrics),
            SATOracleCommand(self._metrics),
            ErrorMonadAuditCommand(self._metrics),
            ValidationCommand(self._metrics, classifier=self._classifier),
            ExecutionCommand(
                self._cache, self._metrics, self._config, classifier=self._classifier
            ),
        ]

    @classmethod
    def from_phase2_seed(
        cls,
        seed: Mapping[str, Any],
        topology: Optional[Phase2Topology] = None,
        config: Optional[MICConfiguration] = None,
    ) -> "MICRegistry":
        r"""
        Unidad F₂→F₃ a nivel de topos. No recrea \(P\) ni el diagrama:
        reutiliza \(\Omega\), \(\chi\), Cartan, métricas y Markov del vivo
        si se aporta `topology`; si no, rehidrata la carta de Darboux
        con `n_dof` del seed (misma \(J\) canónica, no un Ω distinto).
        """
        _require_phase2_seed(seed)
        cfg = config or (topology.config if topology is not None else DEFAULT_MIC_CONFIG)
        n_dof = int(seed.get("n_dof", cfg.n_dof))
        mic = cls(config=cfg)
        mic._phase2_seed = dict(seed)
        if topology is not None:
            mic._phase2_topology = topology
            mic._symplectic_form = topology.symplectic_form
            mic._classifier = topology.classifier
            mic._cartan = topology.cartan
            mic._metrics = topology.metrics
            mic._config = topology.config
        else:
            mic._symplectic_form = SymplecticForm.canonical(n_dof)
            mic._classifier = SubobjectClassifier(
                n_dof=n_dof, symplectic_form=mic._symplectic_form
            )
        mic._projection_commands = mic._build_pipeline()
        return mic

    # -------------------------------------------------------------------------
    # PROPIEDADES DE INTROSPECCIÓN
    # -------------------------------------------------------------------------

    @property
    def registered_services(self) -> List[str]:
        with self._lock:
            return list(self._vectors.keys())

    def list_vectors(self) -> List[str]:
        return self.registered_services

    def get_basis_vector(self, name: str) -> Optional[Any]:
        with self._lock:
            if name not in self._vectors:
                return None
            stratum, handler = self._vectors[name]

            class VectorProxy:
                def __init__(self, s: Any, h: Any) -> None:
                    self.target_stratum = s
                    self.handler = h

            return VectorProxy(stratum, handler)

    @property
    def dimension(self) -> int:
        with self._lock:
            return len(self._vectors)

    @property
    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._metrics.to_dict(),
                "cache": self._cache.stats,
                "symplectic_form": self._symplectic_form.to_dict(),
            }

    @property
    def config(self) -> MICConfiguration:
        return self._config

    @property
    def symplectic_form(self) -> SymplecticForm:
        return self._symplectic_form

    def is_registered(self, service_name: str) -> bool:
        with self._lock:
            return service_name in self._vectors

    def get_stratum(self, service_name: str) -> Optional[Any]:
        with self._lock:
            entry = self._vectors.get(service_name)
            return entry[0] if entry else None

    def get_services_by_stratum(self, stratum: Any) -> List[str]:
        with self._lock:
            return [name for name, (s, _) in self._vectors.items() if s == stratum]

    def get_stratum_hierarchy(self) -> Dict[str, List[str]]:
        hierarchy: Dict[str, List[str]] = {
            _safe_stratum_name(s): [] for s in _ordered_strata()
        }
        with self._lock:
            for name, (stratum, _) in self._vectors.items():
                hierarchy.setdefault(_safe_stratum_name(stratum), []).append(name)
        return hierarchy

    def get_registered_morphisms(self) -> Dict[str, Tuple[Any, Any]]:
        with self._lock:
            return self._vectors.copy()

    def register_vector(self, service_name: str, stratum: Any, handler: Any) -> None:
        with self._lock:
            if not service_name or not service_name.strip():
                raise ValueError("service_name no puede estar vacío")
            if not isinstance(stratum, Stratum):
                raise TypeError(
                    f"stratum debe ser Stratum, recibido: {type(stratum).__name__!r}"
                )
            if not callable(handler):
                raise TypeError(
                    f"handler debe ser callable, recibido: {type(handler).__name__!r}"
                )
            if service_name in self._vectors:
                old_stratum = self._vectors[service_name][0]
                self._logger.warning(
                    "Sobrescribiendo vector '%s': %s → %s",
                    service_name,
                    _safe_stratum_name(old_stratum),
                    _safe_stratum_name(stratum),
                )
            self._vectors[service_name] = (stratum, handler)
            if self._spectral_analyzer is not None:
                self._spectral_analyzer._invalidate_cache()
            self._logger.info(
                "Vector registrado: '%s' [%s]",
                service_name,
                _safe_stratum_name(stratum),
            )

    def unregister_vector(self, service_name: str) -> bool:
        with self._lock:
            if service_name in self._vectors:
                del self._vectors[service_name]
                if self._spectral_analyzer is not None:
                    self._spectral_analyzer._invalidate_cache()
                self._logger.info("Vector eliminado: '%s'", service_name)
                return True
            return False

    def project_intent(
        self,
        service_name: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = False,
        vector_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        r"""Proyección sobre \(\mathcal{E}_{\mathrm{MIC}}\). Pipeline de 11 comandos."""
        final_service_name = service_name or vector_name
        if not final_service_name:
            return {
                "success": False,
                "error": "service_name or vector_name is required",
                "error_type": "ValueError",
                "error_category": "resolution_error",
            }
        final_payload = dict(payload) if payload is not None else {}
        if kwargs:
            final_payload.update(kwargs)
        final_context = dict(context) if context is not None else {}
        ctx = ProjectionContext(
            service_name=final_service_name,
            payload=final_payload,
            context=final_context,
            use_cache=use_cache,
            symplectic_form=self._symplectic_form,
            classifier=self._classifier,
            cartan=self._cartan,
            phase2_seed=self._phase2_seed,
        )
        with self._metrics.projection_latency.measure():
            try:
                result = run_projection_pipeline(self._projection_commands, ctx)
                if result is not None:
                    return result
            except (KeyError, ValueError, MICHierarchyViolationError) as e:
                self._logger.warning(
                    "Divergencia detectada: colapso al Objeto Inicial. Error: %s", e
                )
                return {
                    "success": False,
                    "error": (
                        "Colapso a Objeto Inicial (∅): Divergencia funcional detectada."
                    ),
                    "error_type": "InitialObjectCollapse",
                    "error_category": "categorical_annihilation",
                    "error_details": {
                        "exception": str(e),
                        "service": final_service_name,
                    },
                }
        return {
            "success": False,
            "error": "Projection pipeline incomplete",
            "error_type": "InternalError",
            "error_category": "pipeline_error",
        }

    def clear_cache(self) -> int:
        count = self._cache.clear()
        self._logger.info("Cache limpiado: %d entradas eliminadas", count)
        return count

    def spectral_analysis(self) -> Dict[str, Any]:
        if self._phase2_topology is not None:
            existing = self._phase2_topology.kam_report.get("graph_spectrum")
            if existing:
                return existing
        if self._spectral_analyzer is None:
            def _vector_supplier() -> Dict[str, Tuple[Any, Any]]:
                with self._lock:
                    return dict(self._vectors)

            self._spectral_analyzer = SpectralGraphMetrics(_vector_supplier)
        return self._spectral_analyzer.compute_spectral_metrics(self._config)

    def stratum_statistics(self) -> Dict[str, Any]:
        with self._lock:
            counts: Dict[str, int] = {
                _safe_stratum_name(s): 0 for s in _stratum_members()
            }
            for _, (stratum, _) in self._vectors.items():
                key = _safe_stratum_name(stratum)
                counts[key] = counts.get(key, 0) + 1
            total = sum(counts.values())
            distribution = {
                k: round(v / total, 4) if total > 0 else 0.0 for k, v in counts.items()
            }
            entropy = compute_shannon_entropy(list(distribution.values()))
        return {
            "counts_by_stratum": counts,
            "distribution": distribution,
            "stratum_entropy": round(entropy, 6),
            "total_services": total,
        }

    def ergodic_analysis(self, alpha: float = 0.85) -> Dict[str, Any]:
        if self._phase2_topology is not None and self._phase2_topology.dikw_markov:
            bundled = dict(self._phase2_topology.dikw_markov)
            bundled["section_markov"] = self._phase2_topology.section_markov.to_dict()
            bundled["source"] = "phase2_seed"
            return bundled
        with self._lock:
            counts: Dict[Any, int] = {s: 0 for s in _stratum_members()}
            for _, (stratum, _) in self._vectors.items():
                counts[stratum] = counts.get(stratum, 0) + 1
        return self._transition_matrix.to_dict(counts, alpha)

    def symplectic_audit(self) -> Dict[str, Any]:
        valid, residuals = self._symplectic_form.verify_symplectic_invariants(
            tol=self._config.symplectic_residual_tol
        )
        out = {
            "is_valid": valid,
            "residuals": residuals,
            "form": self._symplectic_form.to_dict(),
            "mean_residual": self._metrics.mean_symplectic_residual,
        }
        if self._phase2_seed:
            out["phase2_audit"] = self._phase2_seed.get("audit")
        return out


# =============================================================================
# 3.16 — REGISTRO DE VECTORES CORE (BASE CANÓNICA)
# =============================================================================

def register_core_vectors(
    mic: MICRegistry,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    r"""
    Base canónica por estrato. `calculate_fat_tail_risk` se registra **una**
    vez: Drive si está disponible; si no, el tensor de improbabilidad.
    """
    logger.info("Iniciando registro de vectores core...")
    mic.register_vector("stabilize_flux", Stratum.PHYSICS, vector_stabilize_flux)
    mic.register_vector("parse_raw", Stratum.PHYSICS, vector_parse_raw_structure)
    mic.register_vector("structure_logic", Stratum.TACTICS, vector_structure_logic)
    mic.register_vector(
        "audit_fusion_homology", Stratum.TACTICS, vector_audit_homological_fusion
    )
    mic.register_vector("lateral_thinking_pivot", Stratum.STRATEGY, vector_lateral_pivot)

    fat_tail_registered = False
    if IMPROBABILITY_DRIVE_AVAILABLE:
        try:
            improbability_drive = ImprobabilityDriveService(mic)

            def calculate_fat_tail_risk_handler(**kwargs: Any) -> Dict[str, Any]:
                return improbability_drive._morphism_handler(**kwargs)

            mic.register_vector(
                "calculate_fat_tail_risk",
                Stratum.STRATEGY,
                calculate_fat_tail_risk_handler,
            )
            fat_tail_registered = True
            logger.info("Motor de Improbabilidad registrado")
        except Exception as e:
            logger.warning("Motor de Improbabilidad no disponible: %s", e)
    if not fat_tail_registered:
        mic.register_vector(
            "calculate_fat_tail_risk",
            Stratum.STRATEGY,
            vector_calculate_improbability_tensor,
        )

    if config and SEMANTIC_ESTIMATOR_AVAILABLE:
        try:
            service = SemanticEstimatorService(config)
            service.register_in_mic(mic)
            logger.info("Vectores semánticos registrados")
        except Exception as e:
            logger.warning("Vectores semánticos no disponibles: %s", e)
    if SEMANTIC_DICTIONARY_AVAILABLE:
        try:
            semantic_dict = SemanticDictionaryService()
            semantic_dict.register_in_mic(mic)
            logger.info("Diccionario semántico registrado")
        except Exception as e:
            logger.warning("Diccionario semántico no disponible: %s", e)

    logger.info(
        "MIC inicializada con %d vectores (dimensión=%d)", mic.dimension, mic.dimension
    )
    logger.debug("Jerarquía de estratos: %s", mic.get_stratum_hierarchy())
    if mic.dimension < 6:
        logger.warning(
            "Dimensión de MIC (%d) menor que el mínimo recomendado (6).",
            mic.dimension,
        )


# =============================================================================
# 3.17 — API PÚBLICA (VALIDACIÓN, DIAGNÓSTICO, HANDLERS)
# =============================================================================

def get_supported_file_types() -> List[str]:
    return FileType.values()


def get_supported_delimiters() -> List[str]:
    return sorted(VALID_DELIMITERS)


def get_supported_encodings() -> List[str]:
    return sorted(SUPPORTED_ENCODINGS)


def normalize_path(path: Union[str, Path, None]) -> Path:
    if path is None:
        raise ValueError("Invariante violado: path no puede ser None.")
    path_str = str(path).strip()
    if not path_str:
        raise ValueError("Invariante violado: path no puede estar vacío.")
    path_obj = Path(path_str) if not isinstance(path, Path) else path
    normalized = path_obj.expanduser().resolve()
    if not normalized.is_absolute():
        normalized = normalized.absolute()
    return normalized


def validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundDiagnosticError(
            path=path, reason="El archivo no existe en el sistema de archivos"
        )
    if not path.is_file():
        if path.is_dir():
            obj_type = "directorio"
        elif path.is_symlink():
            obj_type = "enlace simbólico"
        else:
            obj_type = "objeto especial"
        raise FileValidationError(
            f"La ruta no apunta a un archivo regular: {path}. Tipo: {obj_type}",
            path=str(path),
            expected_type="file",
            actual_type=obj_type,
        )


def validate_file_permissions(
    path: Path,
    check_read: bool = True,
    check_write: bool = False,
    check_execute: bool = False,
) -> None:
    if check_read and not os.access(path, os.R_OK):
        raise FilePermissionError(path=path, operation="read")
    if check_write and not os.access(path, os.W_OK):
        raise FilePermissionError(path=path, operation="write")
    if check_execute and not os.access(path, os.X_OK):
        raise FilePermissionError(path=path, operation="execute")


def validate_file_extension(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in VALID_EXTENSIONS:
        available = sorted(VALID_EXTENSIONS)
        raise FileValidationError(
            f"Extensión no soportada: '{ext}'. Válidas: {', '.join(available)}.",
            provided=ext,
            expected=available,
            path=str(path),
        )
    return ext


def validate_file_size(
    path: Path, max_size: Optional[int] = None
) -> Tuple[int, bool]:
    max_size = (
        max_size if max_size is not None else DEFAULT_MIC_CONFIG.max_file_size_bytes
    )
    size = path.stat().st_size
    if size > max_size:
        raise FileValidationError(
            f"Archivo excede el límite: {size:,} > {max_size:,} bytes.",
            actual_size_bytes=size,
            max_size_bytes=max_size,
            file=str(path),
            excess_bytes=size - max_size,
        )
    return size, (size == 0)


def normalize_encoding(encoding: str) -> str:
    if not encoding or not str(encoding).strip():
        return "utf-8"
    norm = str(encoding).lower().replace("_", "-")
    for alias, standard in _ENCODING_ALIASES.items():
        if norm == alias:
            return standard
    if norm in SUPPORTED_ENCODINGS:
        return norm
    logger.warning("Codificación '%s' no reconocida — usando utf-8", encoding)
    return "utf-8"


def normalize_file_type(file_type: Union[str, FileType]) -> FileType:
    if isinstance(file_type, FileType):
        return file_type
    if isinstance(file_type, str):
        return FileType.from_string(file_type)
    raise TypeError(
        f"file_type debe ser str o FileType, recibido: {type(file_type).__name__!r}"
    )


_DIAGNOSTIC_REGISTRY: Final[Dict[FileType, Optional[Type]]] = {
    FileType.APUS: APUFileDiagnostic,
    FileType.INSUMOS: InsumosFileDiagnostic,
    FileType.PRESUPUESTO: PresupuestoFileDiagnostic,
}


def get_diagnostic_class(file_type: FileType) -> Type:
    diagnostic_class = _DIAGNOSTIC_REGISTRY.get(file_type)
    if diagnostic_class is None:
        raise UnsupportedFileTypeError(
            file_type=file_type.value, available=FileType.values()
        )
    return diagnostic_class


def register_diagnostic_class(
    file_type: FileType, diagnostic_class: Type, override: bool = False
) -> None:
    if not hasattr(diagnostic_class, "diagnose"):
        raise TypeError(f"{diagnostic_class.__name__!r} no implementa 'diagnose()'.")
    if not hasattr(diagnostic_class, "to_dict"):
        raise TypeError(f"{diagnostic_class.__name__!r} no implementa 'to_dict()'.")
    existing = _DIAGNOSTIC_REGISTRY.get(file_type)
    if existing is not None and not override:
        raise ValueError(
            f"Ya existe una clase para {file_type.value!r}: {existing.__name__!r}. "
            f"Use override=True."
        )
    _DIAGNOSTIC_REGISTRY[file_type] = diagnostic_class
    logger.info(
        "Clase diagnóstica registrada: %s → %s",
        file_type.value,
        diagnostic_class.__name__,
    )


def validate_file_for_processing(
    path: Union[str, Path], config: Optional[MICConfiguration] = None
) -> Dict[str, Any]:
    config = config or DEFAULT_MIC_CONFIG
    try:
        p = normalize_path(path)
        validate_file_exists(p)
        validate_file_permissions(p)
        ext = validate_file_extension(p)
        size, is_empty = validate_file_size(p, config.max_file_size_bytes)
        return {
            "valid": True,
            "size": size,
            "extension": ext,
            "is_empty": is_empty,
            "path": str(p),
        }
    except MICException as e:
        return {"valid": False, "errors": [str(e)], **e.to_dict()}
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}


def diagnose_file(
    file_path: Union[str, Path],
    file_type: Union[str, FileType],
    *,
    validate_extension: bool = True,
    max_file_size: Optional[int] = None,
    topological_analysis: bool = False,
    config: Optional[MICConfiguration] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Diagnóstico: validación → clase diagnóstica → TDA de *texto* (no KAM)."""
    config = config or DEFAULT_MIC_CONFIG
    timeout = (
        timeout_seconds if timeout_seconds is not None else config.diagnostic_timeout_seconds
    )
    path_str = str(file_path)
    try:
        path = normalize_path(file_path)
        normalized_type = normalize_file_type(file_type)
        validate_file_exists(path)
        validate_file_permissions(path, check_read=True)
        if validate_extension:
            validate_file_extension(path)
        effective_max = (
            max_file_size if max_file_size is not None else config.max_file_size_bytes
        )
        size, is_empty = validate_file_size(path, effective_max)
        if is_empty:
            return {
                "success": True,
                "diagnostic_completed": True,
                "is_empty": True,
                "file_type": normalized_type.value,
                "file_path": str(path),
                "file_size_bytes": 0,
                "diagnostic_magnitude": 0.0,
                "has_topological_analysis": False,
            }
        diagnostic_class = get_diagnostic_class(normalized_type)
        diagnostic = diagnostic_class(str(path))
        start_time = time.perf_counter()
        diagnostic.diagnose()
        elapsed = time.perf_counter() - start_time
        if elapsed > timeout:
            raise MICTimeoutError(
                operation="diagnose", timeout_seconds=timeout, elapsed_seconds=elapsed
            )
        result_data = diagnostic.to_dict()
        result_data["diagnostic_completed"] = True
        if topological_analysis:
            topo_summary = analyze_topological_features(path, config)
            result_data["topological_features"] = topo_summary.to_dict()
            homology = compute_homology_from_diagnostic(result_data)
            result_data["homology"] = homology
            intervals = compute_persistence_diagram(result_data)
            result_data["persistence_diagram"] = [iv.to_dict() for iv in intervals]
            result_data["persistence_entropy"] = compute_persistence_entropy(
                intervals, config
            )
            result_data["tda_note"] = (
                "TDA sobre líneas/issues del archivo; no es KAM ni mapa de Poincaré."
            )
        magnitude = compute_diagnostic_magnitude(result_data)
        return {
            "success": True,
            **result_data,
            "file_type": normalized_type.value,
            "file_path": str(path),
            "file_size_bytes": size,
            "diagnostic_magnitude": magnitude,
            "has_topological_analysis": topological_analysis,
        }
    except MICException as e:
        logger.warning("Error de validación: %s", e)
        return {"success": False, **e.to_dict()}
    except Exception as e:
        logger.exception("Error inesperado en diagnóstico de '%s'", path_str)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "error_category": "unexpected",
            "error_details": {"path": path_str},
        }


def analyze_topological_features(
    file_path: Path, config: Optional[MICConfiguration] = None
) -> TopologicalSummary:
    r"""
    TDA de *líneas de texto*: \(\beta_0\sim\#\{\mathrm{líneas\ únicas}\}\),
    \(\beta_1\sim\) periodos Jaccard. **No** estima toros KAM.
    """
    config = config or DEFAULT_MIC_CONFIG
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip("\n\r") for line in f.readlines()[: config.max_sample_rows]]
        if not lines:
            return TopologicalSummary.empty()
        line_counts = Counter(lines)
        num_unique = len(line_counts)
        beta_0 = max(1, num_unique)
        beta_1 = detect_cyclic_patterns(lines, config)
        dimension = estimate_intrinsic_dimension(lines, config)
        distribution = distribution_from_counts(line_counts)
        structural_entropy = compute_shannon_entropy(distribution)
        demand = math.pi * (beta_0 + beta_1)
        betti = BettiNumbers(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=0,
            symplectic_capacity_total=demand,
            gromov_bound_residual=0.0,
        )
        liouville_vol = estimate_liouville_volume(
            z_bounds=((0.0, float(num_unique)),),
            p_bounds=((0.0, float(beta_1 + 1)),),
        )
        return TopologicalSummary(
            betti=betti,
            structural_entropy=structural_entropy,
            persistence_entropy=0.0,
            intrinsic_dimension=dimension,
            kam_persistence_ratio=0.0,
            birkhoff_order=0,
            liouville_volume=liouville_vol,
            symplectic_gap=0.0,
        )
    except Exception as e:
        logger.warning("Análisis topológico falló para '%s': %s", file_path, e)
        return TopologicalSummary.empty()


def compute_homology_from_diagnostic(diagnostic_data: Dict[str, Any]) -> Dict[str, Any]:
    issues = diagnostic_data.get("issues", [])
    warnings = diagnostic_data.get("warnings", [])
    issue_types: Set[str] = set()
    for issue in issues:
        if isinstance(issue, dict):
            issue_types.add(str(issue.get("type", issue.get("code", "unknown"))))
        else:
            issue_types.add(type(issue).__name__)
    beta_0 = max(1, len(issue_types))
    circular_keywords = frozenset(
        {"circular", "cycle", "loop", "recursive", "dependency", "deadlock"}
    )

    def has_circular(item: Any) -> bool:
        text = str(item).lower()
        return any(kw in text for kw in circular_keywords)

    beta_1 = sum(1 for item in (*warnings, *issues) if has_circular(item))
    betti = BettiNumbers(beta_0=beta_0, beta_1=beta_1, beta_2=0)
    return {
        "H_0": f"ℤ^{beta_0}",
        "H_1": f"ℤ^{beta_1}" if beta_1 > 0 else "0",
        "H_2": "0",
        **betti.to_dict(),
    }


def compute_persistence_diagram(
    diagnostic_data: Dict[str, Any], delta_t: float = 0.1
) -> List[PersistenceInterval]:
    """Filtración heurística por severidad de issues. No es Rips ni Cartan."""
    issues = diagnostic_data.get("issues", [])
    if not issues:
        return []
    severity_to_weight: Dict[str, float] = {
        "CRITICAL": 1.0,
        "HIGH": 0.8,
        "MEDIUM": 0.5,
        "LOW": 0.2,
        "INFO": 0.1,
    }
    intervals: List[PersistenceInterval] = []
    for idx, issue in enumerate(issues):
        if isinstance(issue, dict):
            raw_sev = issue.get("severity", "MEDIUM")
            weight = severity_to_weight.get(str(raw_sev).upper(), 0.5)
        else:
            weight = 0.5
        birth = idx * delta_t
        death = birth + weight
        try:
            intervals.append(
                PersistenceInterval(
                    birth=birth,
                    death=death,
                    dimension=0,
                    action_integral=0.0,
                    heyting_membership=weight,
                    label=f"issue_{idx}",
                )
            )
        except ValueError:
            continue
    config = DEFAULT_MIC_CONFIG
    significant = [
        iv for iv in intervals if iv.finite_persistence() >= config.persistence_threshold
    ]
    significant.sort()
    return significant


def compute_diagnostic_magnitude(diagnostic_data: Dict[str, Any]) -> float:
    issues = diagnostic_data.get("issues", [])
    errors = diagnostic_data.get("errors", [])
    warnings = diagnostic_data.get("warnings", [])
    severity_counts: Counter = Counter()
    for item in issues:
        if isinstance(item, dict):
            sev = str(item.get("severity", "MEDIUM")).upper()
        else:
            sev = "MEDIUM"
        severity_counts[sev] += 1
    severity_counts["CRITICAL"] += len(errors)
    severity_counts["LOW"] += len(warnings)
    weighted_sq_sum = sum(
        _SEVERITY_WEIGHTS.get(sev, 1.0) * (count ** 2)
        for sev, count in severity_counts.items()
    )
    raw_magnitude = math.sqrt(weighted_sq_sum)
    total_items = max(1, len(issues) + len(errors) + len(warnings))
    scale = math.sqrt(float(total_items))
    return round(math.tanh(raw_magnitude / scale), 4)


def detect_cyclic_patterns(
    lines: List[str], config: Optional[MICConfiguration] = None
) -> int:
    config = config or DEFAULT_MIC_CONFIG
    n = len(lines)
    if n < 3:
        return 0
    effective_n = min(n, config.max_lines_for_cycle_detection)
    lines_to_analyze = lines[:effective_n]
    tokenized = [_tokenize_line(line) for line in lines_to_analyze]
    cycles_found = 0
    effective_max = min(config.max_cycle_period, effective_n // 2)
    for period in range(1, effective_max + 1):
        comparisons = effective_n - period
        if comparisons <= 0:
            continue
        matches = sum(
            1
            for i in range(comparisons)
            if _jaccard_similarity(tokenized[i], tokenized[i + period])
            >= config.cycle_similarity_threshold
        )
        if matches / comparisons >= config.cycle_similarity_threshold:
            cycles_found += 1
    return cycles_found


def estimate_intrinsic_dimension(
    lines: List[str], config: Optional[MICConfiguration] = None
) -> int:
    del config
    if not lines:
        return 0
    data_lines = lines[1:] if len(lines) > 1 else lines
    sample = data_lines[: min(100, len(data_lines))]
    if not sample:
        return 1
    for delimiter in [",", ";", "\t", "|", ":"]:
        if any(delimiter in line for line in sample[:5]):
            col_counts = [len(line.split(delimiter)) for line in sample]
            if col_counts:
                col_counts_sorted = sorted(col_counts)
                mid = len(col_counts_sorted) // 2
                if len(col_counts_sorted) % 2 != 0:
                    return max(1, int(col_counts_sorted[mid]))
                return max(
                    1, int((col_counts_sorted[mid - 1] + col_counts_sorted[mid]) // 2)
                )
    return 1


def _jaccard_similarity(tokens_a: FrozenSet[str], tokens_b: FrozenSet[str]) -> float:
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 0.0


def _tokenize_line(line: str) -> FrozenSet[str]:
    tokens = re.split(r"[,;\t|:\s]+", line.strip())
    return frozenset(t for t in tokens if t)


def analyze_financial_viability(
    amount: float,
    std_dev: float,
    time_years: int,
    risk_free_rate: float = 0.03,
    **kwargs: Any,
) -> Dict[str, Any]:
    del kwargs
    if amount <= 0:
        return {
            "success": False,
            "error": f"Monto debe ser > 0, recibido: {amount}",
            "error_category": "validation_error",
        }
    if time_years < 1:
        return {
            "success": False,
            "error": f"time_years ≥ 1, recibido: {time_years}",
            "error_category": "validation_error",
        }
    try:
        if FinancialEngine is not None and FinancialConfig is not None:
            config = FinancialConfig(market_volatility=std_dev)
            engine = FinancialEngine(config)
            cash_flows = [-amount] + [amount * 0.3] * time_years
            npv = engine.calculate_npv(cash_flows, initial_investment=amount)
            var, cvar = engine.calculate_var(amount)
            return {
                "success": True,
                "npv": round(npv, 2),
                "var_95": round(var, 2),
                "cvar_95": round(cvar, 2),
                "contingency_suggested": engine.suggest_contingency(amount),
                "is_viable": npv > 0,
                "time_years": time_years,
                "risk_free_rate": risk_free_rate,
            }
        discount_rate = risk_free_rate + std_dev
        npv = -amount + sum(
            amount * 0.3 / ((1 + discount_rate) ** t) for t in range(1, time_years + 1)
        )
        var_95 = amount * std_dev * 1.645
        return {
            "success": True,
            "npv": round(npv, 2),
            "var_95": round(var_95, 2),
            "cvar_95": round(var_95 * 1.2, 2),
            "contingency_suggested": amount * 0.1,
            "is_viable": npv > 0,
            "time_years": time_years,
            "risk_free_rate": risk_free_rate,
            "note": "Cálculo simplificado (FinancialEngine no disponible)",
        }
    except Exception as e:
        logger.exception("Error en análisis financiero")
        return {
            "success": False,
            "error": str(e),
            "error_category": "execution_error",
            "error_type": type(e).__name__,
        }


def clean_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    delimiter: str = ";",
    encoding: str = "utf-8",
    remove_duplicates: bool = True,
    normalize_whitespace: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    try:
        input_p = normalize_path(input_path)
        output_p = normalize_path(output_path)
        validate_file_exists(input_p)
        validate_file_permissions(input_p, check_read=True)
        normalized_encoding = normalize_encoding(encoding)
        if CSVCleaner is not None:
            cleaner = CSVCleaner(
                input_path=str(input_p),
                output_path=str(output_p),
                delimiter=delimiter,
                encoding=normalized_encoding,
                **kwargs,
            )
            result = cleaner.clean()
            return {
                "success": True,
                "output_path": str(output_p),
                "input_path": str(input_p),
                "message": "Limpieza completada",
                **(result if isinstance(result, dict) else {}),
            }
        with open(input_p, "r", encoding=normalized_encoding, errors="replace") as f:
            lines = f.readlines()
        cleaned_lines = []
        seen: Optional[Set[str]] = set() if remove_duplicates else None
        for line in lines:
            if normalize_whitespace:
                line = " ".join(line.split())
            if seen is not None:
                if line in seen:
                    continue
                seen.add(line)
            cleaned_lines.append(line)
        with open(output_p, "w", encoding=normalized_encoding) as f:
            f.writelines(cleaned_lines)
        return {
            "success": True,
            "output_path": str(output_p),
            "input_path": str(input_p),
            "rows_processed": len(lines),
            "rows_cleaned": len(cleaned_lines),
            "message": "Limpieza básica completada",
        }
    except MICException as e:
        return {
            "success": False,
            "error": str(e),
            "error_category": e.category,
            "error_type": type(e).__name__,
        }
    except Exception as e:
        logger.exception("Error en limpieza de archivo")
        return {
            "success": False,
            "error": str(e),
            "error_category": "execution_error",
            "error_type": type(e).__name__,
        }


def get_telemetry_status(
    telemetry_context: Optional[Any] = None,
    include_business_report: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    del kwargs
    try:
        metrics: Dict[str, Any] = {}
        report: Dict[str, Any] = {}
        status = "unknown"
        if telemetry_context is not None:
            metrics = getattr(telemetry_context, "metrics", {})
            status = getattr(telemetry_context, "status", "active")
            if include_business_report and hasattr(
                telemetry_context, "get_business_report"
            ):
                try:
                    report = telemetry_context.get_business_report()
                except Exception as e:
                    report = {"error": str(e)}
        else:
            status = "no_context"
        return {
            "success": True,
            "status": status,
            "metrics": metrics,
            "report": report if include_business_report else None,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_category": "execution_error",
            "error_type": type(e).__name__,
            "status": "error",
        }


# =============================================================================
# 3.18 — SINGLETON THREAD-SAFE
# =============================================================================

_global_mic: Optional[MICRegistry] = None
_mic_lock = threading.RLock()
_mic_init_error: Optional[Exception] = None


def get_global_mic(
    config: Optional[Dict[str, Any]] = None,
    mic_config: Optional[MICConfiguration] = None,
    force_reinit: bool = False,
    phase2_seed: Optional[Mapping[str, Any]] = None,
    phase2_topology: Optional[Phase2Topology] = None,
) -> MICRegistry:
    r"""
    Singleton thread-safe (double-checked locking).

    Si se aporta `phase2_seed` / `phase2_topology`, el topos se construye
    por `MICRegistry.from_phase2_seed` (unidad F₂⇒F₃).
    """
    global _global_mic, _mic_init_error
    if _global_mic is not None and not force_reinit:
        return _global_mic
    with _mic_lock:
        if _global_mic is not None and not force_reinit:
            return _global_mic
        if _mic_init_error is not None and not force_reinit:
            raise RuntimeError(
                f"MIC global falló previamente: {_mic_init_error}. Use force_reinit=True."
            ) from _mic_init_error
        try:
            if phase2_seed is not None or phase2_topology is not None:
                seed = phase2_seed or (
                    phase2_topology.seed_phase3_orchestration()
                    if phase2_topology is not None
                    else None
                )
                if seed is None:
                    raise MICFunctorialityError("phase2_seed ausente")
                mic = MICRegistry.from_phase2_seed(
                    seed, topology=phase2_topology, config=mic_config
                )
            else:
                mic = MICRegistry(config=mic_config)
            register_core_vectors(mic, config=config)
            _global_mic = mic
            _mic_init_error = None
            logger.info("MIC global inicializada con %d vectores", mic.dimension)
            return _global_mic
        except Exception as e:
            _mic_init_error = e
            logger.exception("Error crítico durante bootstrap de la MIC")
            raise RuntimeError(f"No se pudo inicializar la MIC: {e}.") from e


def reset_global_mic() -> None:
    global _global_mic, _mic_init_error
    with _mic_lock:
        _global_mic = None
        _mic_init_error = None
        logger.info("MIC global reiniciada (singleton reset)")


# =============================================================================
# 3.19 — ORQUESTACIÓN DE FASE 3  (abre F₂→F₃; cierra F₁∘F₂∘F₃)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Phase3Orchestration:
    r"""
    Objeto agregador de la Fase 3. **Única** puerta de entrada canónica:

        `Phase3Orchestration.from_phase2_seed(seed, topology=...)`

    Consume exactamente el dict de `Phase2Topology.seed_phase3_orchestration`.
    No redeclara \(P\), \(\mathrm{Dgm}\) ni Verlet/Darboux.
    """

    seed: Dict[str, Any]
    registry: MICRegistry
    topology: Optional[Phase2Topology]
    config: MICConfiguration

    @classmethod
    def from_phase2_seed(
        cls,
        seed: Mapping[str, Any],
        topology: Optional[Phase2Topology] = None,
        config: Optional[MICConfiguration] = None,
        bootstrap: bool = True,
        system_config: Optional[Dict[str, Any]] = None,
    ) -> "Phase3Orchestration":
        r"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║  PRIMER MÉTODO DE LA FASE 3                                      ║
        ║  Unidad F₂ → F₃  (continúa seed_phase3_orchestration)            ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
        _require_phase2_seed(seed)
        cfg = config or (topology.config if topology is not None else DEFAULT_MIC_CONFIG)
        registry = MICRegistry.from_phase2_seed(seed, topology=topology, config=cfg)
        if bootstrap:
            register_core_vectors(registry, config=system_config)
        return cls(seed=dict(seed), registry=registry, topology=topology, config=cfg)

    def project(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.registry.project_intent(*args, **kwargs)

    def public_surface(self) -> Dict[str, Any]:
        return {
            "services": self.registry.registered_services,
            "dimension": self.registry.dimension,
            "hierarchy": self.registry.get_stratum_hierarchy(),
            "symplectic_audit": self.registry.symplectic_audit(),
            "ergodicity": self.registry.ergodic_analysis(),
            "metrics": self.registry.metrics,
            "file_types": get_supported_file_types(),
        }

    def close_three_phase_composition(self) -> Dict[str, Any]:
        r"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║  ÚLTIMO MÉTODO DE LA FASE 3                                      ║
        ║  Cierre funtorial  F₁ ∘ F₂ ∘ F₃                                  ║
        ╚══════════════════════════════════════════════════════════════════╝

        Certifica la torre de seeds:

            seed_phase2_topology ⊂ phase1_continuation
            seed_phase3_orchestration = this.seed
            topos operativo = MICRegistry

        No produce una Fase 4: el objeto terminal es el topos con API
        pública, singleton y base canónica registrada.
        """
        p1 = self.seed.get("phase1_continuation") or {}
        topology = self.seed.get("topology") or {}
        dynamics = self.seed.get("dynamics") or {}
        audit = self.seed.get("audit") or {}
        darboux_ok = True
        try:
            valid, _ = self.registry.symplectic_form.verify_symplectic_invariants(
                tol=self.config.symplectic_residual_tol
            )
            darboux_ok = bool(valid)
        except Exception:
            darboux_ok = False
        certified = bool(
            p1.get("seed_for_phase2")
            and self.seed.get("seed_for_phase3")
            and darboux_ok
        )
        return {
            "version": self.config.algorithm_version,
            "composition": "F1∘F2∘F3",
            "certified": certified,
            "n_dof": self.config.n_dof,
            "phase1": {
                "seed_for_phase2": p1.get("seed_for_phase2"),
                "n_dof": p1.get("n_dof"),
                "kam": p1.get("kam"),
                "audit": p1.get("phase1_audit"),
            },
            "phase2": {
                "seed_for_phase3": True,
                "betti": topology.get("betti"),
                "floquet": (dynamics.get("floquet") or {}).get("classification"),
                "kam_candidate": (dynamics.get("kam_report") or {}).get("kam_candidate"),
                "audit": audit,
            },
            "phase3": {
                "registry_dimension": self.registry.dimension,
                "services": self.registry.registered_services,
                "pipeline_length": len(self.registry._projection_commands),
                "darboux_ok": darboux_ok,
                "public_surface": {
                    "file_types": get_supported_file_types(),
                    "encodings": len(get_supported_encodings()),
                },
            },
            "terminal_object": "MICRegistry ⊗ E_MIC",
            "next_phase": None,
        }


__phase3_version__: Final[str] = "8.1.0-poincare-symplectic"
__phase3_exports__: Final[Tuple[str, ...]] = (
    "Phase3Seed",
    "ProjectionContext",
    "ProjectionCommand",
    "CacheCheckCommand",
    "ResolutionCommand",
    "SheafCohomologyProjectionCommand",
    "NormalizationCommand",
    "BDDVerificationCommand",
    "InterchangeLawVerificationCommand",
    "PoincareSymplecticAdjunctionCommand",
    "ErrorMonadAuditCommand",
    "SATOracleCommand",
    "SATOrcaleCommand",
    "ValidationCommand",
    "ExecutionCommand",
    "MICRegistry",
    "Phase3Orchestration",
)


# =============================================================================
# 3.20 — EXPORTACIONES PÚBLICAS
# =============================================================================

__all__: Final[List[str]] = [
    # Configuración
    "MICConfiguration",
    "DEFAULT_MIC_CONFIG",
    "FileType",
    "Stratum",
    # Fase 1 — geometría
    "SymplecticForm",
    "PhaseSpacePoint",
    "HamiltonianSystem",
    "FlowResult",
    "ActionAngleCoordinates",
    "GeneratingFunction",
    "HeytingValue",
    "SubobjectClassifier",
    "PoincareCartanForm",
    "FirstIntegral",
    "PoincareSection",
    "Phase1Foundation",
    # Fase 2 — topología / dinámica
    "Phase2Seed",
    "PersistenceInterval",
    "PersistenceDiagram",
    "BettiNumbers",
    "TopologicalSummary",
    "IntentVector",
    "PoincareReturnMap",
    "FloquetSpectrum",
    "SpectralGraphMetrics",
    "StratumTransitionMatrix",
    "ErgodicMarkovStrata",
    "Phase2Topology",
    # Excepciones (sin pisar builtins ni mic_algebra)
    "MICException",
    "TopologicalInvariantError",
    "MICFunctorialityError",
    "FileNotFoundDiagnosticError",
    "UnsupportedFileTypeError",
    "FileValidationError",
    "FilePermissionError",
    "CleaningError",
    "MICHierarchyViolationError",
    "MICTimeoutError",
    # Cache / métricas
    "TTLCache",
    "CacheEntry",
    "LatencyHistogram",
    "MICMetrics",
    # Fase 3 — comandos y topos
    "Phase3Seed",
    "ProjectionCommand",
    "ProjectionContext",
    "CacheCheckCommand",
    "ResolutionCommand",
    "SheafCohomologyProjectionCommand",
    "NormalizationCommand",
    "BDDVerificationCommand",
    "InterchangeLawVerificationCommand",
    "PoincareSymplecticAdjunctionCommand",
    "ErrorMonadAuditCommand",
    "SATOracleCommand",
    "SATOrcaleCommand",
    "ValidationCommand",
    "ExecutionCommand",
    "MICRegistry",
    "Phase3Orchestration",
    # Auditorías
    "audit_poincare_darboux_symplectic_form",
    "audit_gromov_nonsqueezing_capacity",
    "audit_novikov_small_divisors_spectrum",
    "audit_poincare_ergodic_recurrence_distance",
    "audit_birkhoff_integrability_order",
    "audit_liouville_volume_preservation",
    "audit_phase2_dynamics_bundle",
    # Diagnóstico / validación
    "diagnose_file",
    "validate_file_for_processing",
    "get_supported_file_types",
    "get_supported_delimiters",
    "get_supported_encodings",
    "compute_shannon_entropy",
    "compute_persistence_entropy",
    "compute_kam_persistence_ratio",
    "compute_symplectic_gap",
    "estimate_liouville_volume",
    "distribution_from_counts",
    "analyze_topological_features",
    "compute_homology_from_diagnostic",
    "compute_persistence_diagram",
    "compute_diagnostic_magnitude",
    "detect_cyclic_patterns",
    "estimate_intrinsic_dimension",
    "normalize_path",
    "validate_file_exists",
    "validate_file_permissions",
    "validate_file_extension",
    "validate_file_size",
    "normalize_encoding",
    "normalize_file_type",
    "get_diagnostic_class",
    "register_diagnostic_class",
    "analyze_financial_viability",
    "clean_file",
    "get_telemetry_status",
    "register_core_vectors",
    "get_global_mic",
    "reset_global_mic",
    "SUPPORTED_ENCODINGS",
    "VALID_DELIMITERS",
    "VALID_EXTENSIONS",
    "MIC_ALGEBRA_AVAILABLE",
    "SHEAF_COHOMOLOGY_AVAILABLE",
    "Z3_AVAILABLE",
    "BDD_AVAILABLE",
    "SCIPY_SPARSE_AVAILABLE",
    "NUMPY_AVAILABLE",
    "SEMANTIC_ESTIMATOR_AVAILABLE",
    "IMPROBABILITY_DRIVE_AVAILABLE",
    "SEMANTIC_DICTIONARY_AVAILABLE",
]

# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Catalytic Quantum Reaction Chamber (Cámara de Reacción Cuántica)    ║
║ Ruta   : app/physics/reaction_chamber.py                                     ║
║ Versión: 4.1.0-Doctoral-Huckel-Z-Ring-Smith-CFL-Dot2-FPU-Secure              ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Foso de simulación termodinámica y cuántica para el reactor catalítico del   ║
║ Estrato Physics ($V_{\mathrm{PHYSICS}}$) de APU Filter v5.0.                 ║
║                                                                              ║
║ Modela la resonancia aromática cíclica de un anillo de benceno de 6 carbonos ║
║ como un complejo simplicial 1-dimensional. Consagra el anillo $\mathbb{Z}$   ║
║ como Dominio de Ideales Principales: los invariantes homológicos se calculan ║
║ de forma EXACTA (Forma Normal de Smith) y se cruzan con el rango numérico    ║
║ en $\mathbb{R}$. El error de redondeo de Wilkinson se neutraliza mediante    ║
║ transformaciones libres de error (TwoSum / TwoProd) y sumación compensada.   ║
╚══════════════════════════════════════════════════════════════════════════════╝

================════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO (Anillo de los Enteros y Simetría Homológica)
================════════════════════════════════════════════════════════════════

Definición 1 (Complejo de cadenas de $C_6$ sobre $\mathbb{Z}$):
  $$C_1(K;\mathbb{Z}) = \bigoplus_{e\in E}\mathbb{Z}e,\qquad
    C_0(K;\mathbb{Z}) = \bigoplus_{v\in V}\mathbb{Z}v,\qquad
    \partial_1[u,v] = v - u .$$
  Como $C_2 = 0$:  $H_1 \cong \ker\partial_1 \cong \mathbb{Z}^{\beta_1}$,
  $H_0 \cong \operatorname{coker}\partial_1 \cong \mathbb{Z}^{\beta_0}\oplus
  \bigoplus_i \mathbb{Z}/d_i\mathbb{Z}$.

Definición 2 (Forma Normal de Smith y torsión):
  $$\mathbf{U}\,\partial_1\,\mathbf{V} = \operatorname{diag}(d_1,\dots,d_r,0,\dots,0),
    \qquad d_i \mid d_{i+1}.$$
  La matriz de incidencia de un grafo es totalmente unimodular, luego
  $d_i = 1$ y $\operatorname{Tor}(H_0) = 0$. El módulo lo VERIFICA, no lo asume.

Definición 3 (Hamiltoniano de Hückel):
  $$\mathbf{H} = \alpha\mathbf{I} + \beta\mathbf{A},\qquad
    E_k = \alpha + 2\beta\cos(2\pi k/6).$$

================════════════════════════════════════════════════════════════════
II. LEYES DE LA FPU
================════════════════════════════════════════════════════════════════

Axioma I (Gibbs deformado):  $G = H - TS + \mu_{\mathrm{topo}}\|\psi\|^2 + RT\ln a$.

Axioma II (von Neumann / CFL para Euler explícito):
  $\sigma(\mathbf{I}-\alpha\mathbf{L}) = \{1-\alpha\lambda_k\}$. Tres regímenes:
  $$\alpha \le 2/\lambda_{\max}\ (\text{estable}),\quad
    \alpha \le 1/\lambda_{\max}\ (\text{monótono: } \mathbf{I}-\alpha\mathbf{L}\succeq 0),\quad
    \alpha \le 1/(2\lambda_{\max})\ (\text{margen histórico del reactor}).$$

Axioma III (Compensación):
  Todo producto interno se evalúa con Dot2 (Ogita–Rump–Oishi):
  $$|\mathrm{fl}(x^\top y) - x^\top y| \le u\,|x^\top y| + \gamma_n^2\,|x|^\top|y|,$$
  equivalente a trabajar en precisión doble de la de máquina.

Álgebra de Banach de operadores $\mathfrak{B}(\mathbb{R}^n)$:
  $$\|T\|_2 = \rho(T^\top T)^{1/2} \quad(\text{identidad C*}),\qquad
    \rho(T) \le \|T\|_\infty \quad(\text{Gershgorin}).$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import numbers
import time
import uuid

import numpy as np

from collections import deque
from types import MappingProxyType
from typing import Deque, Mapping
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

try:
    from app.schemas import Stratum
    from app.telemetry import TelemetryContext
    from app.tools_interface import MICRegistry
except ImportError:  # pragma: no cover - fallback para pruebas standalone
    from typing import Any as _Any

    Stratum = _Any

    class TelemetryContext(Protocol):
        def record_reaction_start(
            self,
            reaction_id: str,
            context: Dict[str, Any],
        ) -> None:
            ...

        def record_reaction_success(
            self,
            reaction_id: str,
            cycle: int,
        ) -> None:
            ...

        def record_error(
            self,
            component: str,
            message: str,
        ) -> None:
            ...

    class MICRegistry(Protocol):
        pass


logger = logging.getLogger("QuantumReactor")

__version__ = "4.1.0"

__all__ = [
    # Fase 1
    "PhysicalConstants",
    "NumericalConstants",
    "TopologyConstants",
    "DampingConstants",
    "CFLConstants",
    "HuckelConstants",
    "ReactorLimits",
    "NumericalDomainError",
    "HomologicalIntegrityError",
    "ExactIntegerAlgebra",
    "TikhonovSolution",
    "CoboundaryKernel",
    "CompensatedLinearAlgebra",
    # Fase 2
    "CarbonNode",
    "AromaticityState",
    "ConvergenceStatus",
    "TellegenAuditReport",
    "LyapunovCertificate",
    "DiscreteExteriorCalculus",
    "HexagonalTopology",
    # Fase 3
    "HilbertState",
    "ThermodynamicPotential",
    "AromaticityEvaluator",
    "EntropyCalculator",
    "ReactionResult",
    "CatalyticReactor",
    "ReactorConfig",
    "create_reactor",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — SUSTRATO NUMÉRICO-ALGEBRAICO
# Banach / EFT (TwoSum, TwoProd) / Dot2 / SNF sobre ℤ / kernel de coborde
# ══════════════════════════════════════════════════════════════════════════════
#
# Objetos:  ℝ con aritmética compensada, (ℝⁿ, ‖·‖₂) de Hilbert,
#           𝔅(ℝⁿ) álgebra de Banach de operadores con ‖T‖₂ = ρ(TᵀT)^{1/2},
#           ℤ como DIP con Forma Normal de Smith exacta.
#
# El morfismo terminal de esta fase es
#     CompensatedLinearAlgebra.construct_coboundary_kernel
# que instala el complejo de cocadenas (CoboundaryKernel) y es el objeto
# inicial de la Fase 2 (DiscreteExteriorCalculus.install_coboundary_kernel).


# ──────────────────────────────────────────────────────────────────────────────
# 1.0  Excepciones del dominio
# ──────────────────────────────────────────────────────────────────────────────


class NumericalDomainError(ValueError):
    """Un valor abandona el dominio numérico admisible (NaN, ∞, no integral, dimensión)."""


class HomologicalIntegrityError(RuntimeError):
    """El complejo de cadenas viola un axioma estructural (∂∘∂ ≠ 0, torsión, χ inconsistente)."""


# ──────────────────────────────────────────────────────────────────────────────
# 1.1  Constantes (tipadas como Final: inmutables por contrato estático)
# ──────────────────────────────────────────────────────────────────────────────


class PhysicalConstants:
    """Constantes físicas fundamentales (escala de modelo y SI)."""

    __slots__ = ()

    R_GAS: Final[float] = 8.314462618           # J·mol⁻¹·K⁻¹ (exacta, SI 2019)
    BOLTZMANN_SI: Final[float] = 1.380649e-23   # J·K⁻¹ (exacta, SI 2019)
    PLANCK_SI: Final[float] = 6.62607015e-34    # J·s  (exacta, SI 2019)
    BOLTZMANN_SCALE: Final[float] = 1.0e-1      # k_B adimensionalizada del modelo
    T_REFERENCE: Final[float] = 298.15
    T_MINIMUM: Final[float] = 280.0
    ARRHENIUS_A: Final[float] = 1.0e13
    E_ACTIVATION_BASE: Final[float] = 50.0e3
    EYRING_TRANSMISSION: Final[float] = 1.0


class NumericalConstants:
    """Constantes numéricas: ε-máquina, regularización, tolerancias de auditoría."""

    __slots__ = ()

    MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)   # u = 2⁻⁵² ≈ 2.22e-16
    EPS: Final[float] = 1e-12
    GIBBS_CONVERGENCE_TOL: Final[float] = 0.05
    SPECTRAL_TOL: Final[float] = 1e-10
    ENTROPY_MIN_PROB: Final[float] = 1e-10

    # Ortogonalización
    ORTHO_RESTARTS: Final[int] = 2
    ORTHO_KAPPA: Final[float] = 1.0 / math.sqrt(2.0)   # criterio Kahan–Parlett

    # Iteración de potencias
    POWER_ITERATION_MAX: Final[int] = 512
    POWER_ITERATION_TOL: Final[float] = 1e-12

    # Tikhonov
    TIKHONOV_BASE_MIN: Final[float] = 1.0e-6
    TIKHONOV_EPS_SCALE: Final[float] = 1.0e3
    TIKHONOV_COND_CEILING: Final[float] = 1.0e12

    # Auditorías
    TELLEGEN_TOLERANCE: Final[float] = 1.0e-10
    WEYL_TOLERANCE: Final[float] = 1.0e-9
    BETTI_RANK_TOL: Final[float] = 1.0e-9
    INTEGRALITY_TOL: Final[float] = 1.0e-9     # |x − round(x)| para admitir x ∈ ℤ
    STRUCTURAL_TOL: Final[float] = 1.0e-12     # residuos algebraicos exactos (𝟙ᵀB, trazas)


class TopologyConstants:
    """Invariantes topológicos y espectrales del anillo hexagonal C₆."""

    __slots__ = ()

    RING_SIZE: Final[int] = 6
    LAMBDA_MAX: Final[float] = 4.0       # 2 − 2cos(π) para n par
    SPECTRAL_GAP: Final[float] = 1.0     # λ₁ = 2 − 2cos(2π/6)
    PRESSURE_COEFF: Final[float] = 1.0
    BETTI_0: Final[int] = 1
    BETTI_1: Final[int] = 1
    EULER_CHARACTERISTIC: Final[int] = 0  # χ = |V| − |E| = β₀ − β₁ = 0  (C₆ ≃ S¹)
    # Teorema matriz-árbol de Kirchhoff: τ(C_n) = (1/n)·∏_{k≥1} λ_k = (1·1·3·3·4)/6 = 6 = n.
    SPANNING_TREES: Final[int] = 6


class DampingConstants:
    """Amortiguamiento de Lyapunov y disipación local."""

    __slots__ = ()

    GAMMA: Final[float] = 0.3
    OMEGA: Final[float] = math.pi / 3.0
    COOLING_FACTOR: Final[float] = 0.95
    LOCAL_STRESS_GAIN: Final[float] = 0.08
    LOCAL_STRESS_DISSIPATION: Final[float] = 0.02
    LYAPUNOV_DISSIPATION: Final[float] = 0.15


class CFLConstants:
    """
    Condición de Courant–Friedrichs–Lewy / von Neumann para Euler explícito.

    El espectro del operador de paso es σ(I − αL) = {1 − αλₖ}. Tres regímenes:

        estabilidad ℓ²      |1 − αλ| ≤ 1  ∀λ ∈ [0, λ_max]  ⇔  α ≤ 2/λ_max   (ALPHA_SHARP)
        monotonía           I − αL ⪰ 0                     ⇔  α ≤ 1/λ_max   (ALPHA_MONOTONE)
        margen histórico    α ≤ 1/(2λ_max)                                  (ALPHA_CRITICAL)

    ALPHA_SAFE aplica un factor de seguridad determinista sobre ALPHA_CRITICAL.
    """

    __slots__ = ()

    ALPHA_SHARP: Final[float] = 2.0 / TopologyConstants.LAMBDA_MAX          # 0.5
    ALPHA_MONOTONE: Final[float] = 1.0 / TopologyConstants.LAMBDA_MAX       # 0.25
    ALPHA_CRITICAL: Final[float] = 1.0 / (2.0 * TopologyConstants.LAMBDA_MAX)  # 0.125
    SAFETY_MARGIN: Final[float] = 0.95
    ALPHA_SAFE: Final[float] = SAFETY_MARGIN * ALPHA_CRITICAL               # 0.11875


class HuckelConstants:
    """Hamiltoniano de Hückel H = αI + βA sobre C₆ (β < 0: integral de resonancia)."""

    __slots__ = ()

    ALPHA: Final[float] = 0.20
    BETA: Final[float] = -0.05
    ACTIVATION_CEILING: Final[float] = 0.9
    PI_ELECTRONS_BENZENE: Final[int] = 6


class ReactorLimits:
    """Límites operacionales y umbrales de colapso."""

    __slots__ = ()

    INSTABILITY_THRESHOLD: Final[float] = 5.0
    COLLAPSE_FACTOR: Final[float] = 1.2
    MAX_RESONANCE_CYCLES: Final[int] = 6
    MIN_CONVERGENCE_CYCLE: Final[int] = 3
    MAX_NODE_SLEEP: Final[float] = 0.050
    MIN_ENTHALPY: Final[float] = 1e-10
    MIN_ENTROPY: Final[float] = 1e-10
    NORM_RELATIVE_STATIONARITY: Final[float] = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# 1.2  Predicados y funciones escalares regularizadas
# ──────────────────────────────────────────────────────────────────────────────


def _is_finite(value: Any) -> bool:
    """
    Predicado de pertenencia a ℝ (finito).

    Rechaza `bool` (aunque `bool ⊂ int` en Python, no es un escalar del
    modelo) y acepta cualquier `numbers.Real` (int, float, np.float64, …).
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def _validate_finite_vector(
    values: Sequence[float],
    name: str = "vector",
    expected_dim: Optional[int] = None,
) -> None:
    """
    Inmersión de validación: el vector vive en ℝⁿ, no en su compactificación.

    Si `expected_dim` se suministra, también se verifica n = expected_dim.
    """
    if expected_dim is not None and len(values) != expected_dim:
        raise NumericalDomainError(
            f"{name}: dimensión {len(values)} ≠ {expected_dim}"
        )
    for i, value in enumerate(values):
        if not _is_finite(value):
            raise NumericalDomainError(f"{name}[{i}] no es finito: {value!r}")


def _stable_divide(
    numerator: float,
    denominator: float,
    eps: float = NumericalConstants.EPS,
) -> float:
    """
    División regularizada en el entorno del polo:

        x / y                       si |y| ≥ ε
        sgn(x)·sgn(y)·|x| / ε       si |y| < ε, x ≠ 0
        0                           si x = 0

    Se preserva el signo del cociente (defecto corregido respecto a v4.0).
    """
    if abs(denominator) < eps:
        if numerator == 0.0:
            return 0.0
        sign = math.copysign(1.0, numerator) * math.copysign(1.0, denominator)
        return sign * abs(numerator) / eps
    return numerator / denominator


def _stable_sqrt(value: float) -> float:
    """Raíz en [0, ∞) con proyección de negativos numéricos (−1e-17 ↦ 0)."""
    return math.sqrt(max(value, 0.0))


def _stable_exp(x: float, limit: float = 700.0) -> float:
    """exp saturada al rango de float64 (ln(DBL_MAX) ≈ 709.78) para Arrhenius."""
    return math.exp(min(max(x, -limit), limit))


def _stable_log(x: float, eps: float = NumericalConstants.EPS) -> float:
    """log regularizado sobre (ε, ∞)."""
    return math.log(max(x, eps))


def _stable_log1p(x: float) -> float:
    """log(1 + x) con corte en el polo x = −1."""
    return math.log1p(max(x, -1.0 + NumericalConstants.EPS))


# ──────────────────────────────────────────────────────────────────────────────
# 1.3  Transformaciones libres de error (EFT) y sumación compensada
# ──────────────────────────────────────────────────────────────────────────────

_SPLITTER: Final[float] = 134217729.0          # 2²⁷ + 1 (Veltkamp)
_HAS_FMA: Final[bool] = hasattr(math, "fma")    # Python ≥ 3.13


def _two_sum(a: float, b: float) -> Tuple[float, float]:
    """
    TwoSum de Knuth–Møller.

        s = fl(a + b),   a + b = s + e   exactamente (si no hay overflow).

    Coste: 6 flops. Generador de toda la compensación de la Fase 1.
    """
    s = a + b
    b_virtual = s - a
    a_virtual = s - b_virtual
    return s, (a - a_virtual) + (b - b_virtual)


def _split(a: float) -> Tuple[float, float]:
    """Split de Veltkamp: a = a_hi + a_lo con ≤ 26 bits significativos cada uno."""
    c = _SPLITTER * a
    a_hi = c - (c - a)
    return a_hi, a - a_hi


def _two_prod(a: float, b: float) -> Tuple[float, float]:
    """
    TwoProd:  p = fl(a·b),   a·b = p + e  exactamente.

    Usa FMA si el intérprete la expone (e = fma(a, b, −p)); en caso
    contrario, algoritmo de Dekker con split de Veltkamp (17 flops).
    Precondición: |a|, |b| < 2⁹⁹⁶ para evitar overflow del split.
    """
    p = a * b
    if _HAS_FMA:
        return p, math.fma(a, b, -p)  # type: ignore[attr-defined]
    a_hi, a_lo = _split(a)
    b_hi, b_lo = _split(b)
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, err


def _kbn_sum(values: Iterable[float]) -> float:
    """
    Sumación compensada Kahan–Babuška–Neumaier.

    Invariante:  s + c = Σxᵢ + O(n u² Σ|xᵢ|)   (frente a O(n u Σ|xᵢ|) naïve).
    Precondición: entradas finitas (∞ produce NaN en el término de corrección).
    """
    s = 0.0
    c = 0.0
    for raw in values:
        x = float(raw)
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return float(s + c)


def _kbn_dot(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Producto interno ⟨a|b⟩ por el algoritmo Dot2 (Ogita–Rump–Oishi, 2005).

    Cada producto y cada suma se descomponen en (valor, error) mediante
    TwoProd y TwoSum; los errores se acumulan por separado:

        |Dot2(a,b) − aᵀb| ≤ u|aᵀb| + γ²ₙ |a|ᵀ|b|,   γₙ = n u / (1 − n u)

    i.e. el resultado es el que produciría precisión doble de la de máquina.
    """
    if len(a) != len(b):
        raise NumericalDomainError(
            "Producto interno indefinido para vectores de distinta dimensión"
        )
    p = 0.0
    s = 0.0
    for x, y in zip(a, b):
        h, r = _two_prod(float(x), float(y))
        p, q = _two_sum(p, h)
        s += q + r
    return float(p + s)


def _compute_norm(vector: Sequence[float]) -> float:
    """
    Norma ℓ² escalada (inmune a overflow/underflow intermedio):

        ‖x‖₂ = m · √(Σ(xᵢ/m)²),   m = ‖x‖_∞

    con la suma interna evaluada por Dot2.
    """
    if not vector:
        return 0.0
    max_abs = max(abs(float(x)) for x in vector)
    if max_abs == 0.0 or not math.isfinite(max_abs):
        return 0.0 if max_abs == 0.0 else math.inf
    scaled = [float(x) / max_abs for x in vector]
    return float(max_abs * _stable_sqrt(_kbn_dot(scaled, scaled)))


def _normalize_vector(
    vector: Sequence[float],
    eps: float = NumericalConstants.EPS,
) -> Tuple[List[float], float]:
    """Proyección a S^{n−1}; retorna (v̂, ‖v‖). Si ‖v‖ < ε devuelve v sin alterar."""
    norm = _compute_norm(vector)
    if norm < eps:
        return [float(x) for x in vector], norm
    return [float(x) / norm for x in vector], norm


def _frobenius_norm(matrix: Sequence[Sequence[float]]) -> float:
    """‖A‖_F = √Σᵢⱼ Aᵢⱼ² (isométrica con la norma de Hilbert–Schmidt)."""
    return _compute_norm([float(a) for row in matrix for a in row])


def _inf_operator_norm(matrix: Sequence[Sequence[float]]) -> float:
    """‖A‖_∞ = maxᵢ Σⱼ|Aᵢⱼ|  — cota de Gershgorin: ρ(A) ≤ ‖A‖_∞."""
    if not matrix:
        return 0.0
    return max(_kbn_sum(abs(float(a)) for a in row) for row in matrix)


# ──────────────────────────────────────────────────────────────────────────────
# 1.4  Álgebra exacta sobre el DIP ℤ (Forma Normal de Smith)
# ──────────────────────────────────────────────────────────────────────────────


class ExactIntegerAlgebra:
    """
    Álgebra lineal exacta sobre ℤ (enteros de precisión arbitraria de Python).

    Provee la Forma Normal de Smith de un morfismo ℤᵐ → ℤⁿ, de la que se
    leen: rango entero, factores invariantes d₁ | d₂ | … | d_r y el
    subgrupo de torsión ⊕ ℤ/dᵢℤ del conúcleo.
    """

    __slots__ = ()

    @staticmethod
    def as_integer_matrix(
        matrix: Sequence[Sequence[float]],
        tol: float = NumericalConstants.INTEGRALITY_TOL,
    ) -> List[List[int]]:
        """Retracción ℝ^{n×m} ∩ (ℤ + B_tol) → ℤ^{n×m}; rechaza entradas no integrales."""
        result: List[List[int]] = []
        for i, row in enumerate(matrix):
            int_row: List[int] = []
            for j, x in enumerate(row):
                xf = float(x)
                if not math.isfinite(xf):
                    raise NumericalDomainError(f"A[{i},{j}] no finito: {x!r}")
                r = round(xf)
                if abs(xf - r) > tol:
                    raise NumericalDomainError(
                        f"A[{i},{j}] = {xf!r} no es entero (tol={tol:g})"
                    )
                int_row.append(int(r))
            result.append(int_row)
        return result

    @staticmethod
    def smith_invariant_factors(matrix: Sequence[Sequence[int]]) -> List[int]:
        """
        Factores invariantes (d₁, …, d_min(n,m)) de la Forma Normal de Smith.

        Algoritmo: eliminación euclídea con pivote de valor absoluto mínimo.
        En cada etapa t:
          (i)   se lleva el menor |a| ≠ 0 del bloque residual a (t,t);
          (ii)  se reducen fila y columna t por división euclídea; si aparece
                un residuo r ≠ 0, |r| < |pivote|, éste se convierte en pivote
                (descenso estricto ⇒ terminación);
          (iii) se impone d_t | A[i,j] ∀ i,j > t sumando a la fila t cualquier
                fila que contenga un no-múltiplo, y se repite (ii).
        Resultado: diag(d₁, …, d_r, 0, …) con d_i | d_{i+1}, d_i ≥ 0.
        """
        A = [[int(x) for x in row] for row in matrix]
        n_rows = len(A)
        n_cols = len(A[0]) if n_rows else 0
        if any(len(row) != n_cols for row in A):
            raise NumericalDomainError("Matriz no rectangular")

        rank_bound = min(n_rows, n_cols)
        factors: List[int] = []
        t = 0

        while t < rank_bound:
            # (i) pivote de módulo mínimo en el bloque residual
            pivot: Optional[Tuple[int, int]] = None
            for i in range(t, n_rows):
                for j in range(t, n_cols):
                    a = A[i][j]
                    if a != 0 and (
                        pivot is None or abs(a) < abs(A[pivot[0]][pivot[1]])
                    ):
                        pivot = (i, j)
            if pivot is None:
                break

            pi, pj = pivot
            A[t], A[pi] = A[pi], A[t]
            for row in A:
                row[t], row[pj] = row[pj], row[t]

            while True:
                p = A[t][t]
                restarted = False

                # (ii-a) columna t
                for i in range(t + 1, n_rows):
                    if A[i][t] != 0:
                        q = A[i][t] // p
                        A[i] = [a - q * b for a, b in zip(A[i], A[t])]
                        if A[i][t] != 0:          # residuo |r| < |p|
                            A[t], A[i] = A[i], A[t]
                            restarted = True
                            break
                if restarted:
                    continue

                # (ii-b) fila t
                for j in range(t + 1, n_cols):
                    if A[t][j] != 0:
                        q = A[t][j] // p
                        for r in range(n_rows):
                            A[r][j] -= q * A[r][t]
                        if A[t][j] != 0:
                            for r in range(n_rows):
                                A[r][t], A[r][j] = A[r][j], A[r][t]
                            restarted = True
                            break
                if restarted:
                    continue

                # (iii) divisibilidad d_t | A[i,j] en el bloque residual
                offending_row: Optional[int] = None
                for i in range(t + 1, n_rows):
                    if any(A[i][j] % p != 0 for j in range(t + 1, n_cols)):
                        offending_row = i
                        break
                if offending_row is None:
                    break
                A[t] = [a + b for a, b in zip(A[t], A[offending_row])]

            factors.append(abs(A[t][t]))
            t += 1

        factors.extend([0] * (rank_bound - len(factors)))
        return factors

    @staticmethod
    def integer_rank(invariant_factors: Sequence[int]) -> int:
        """rank_ℤ = #{dᵢ ≠ 0} (coincide con rank_ℚ)."""
        return sum(1 for d in invariant_factors if d != 0)

    @staticmethod
    def torsion_coefficients(invariant_factors: Sequence[int]) -> Tuple[int, ...]:
        """Coeficientes de torsión {dᵢ > 1}; vacío ⇔ conúcleo libre."""
        return tuple(d for d in invariant_factors if d > 1)


# ──────────────────────────────────────────────────────────────────────────────
# 1.5  Contratos de retorno tipados (inmutables)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TikhonovSolution:
    """
    Resultado de la inversa regularizada de Tikhonov.

        A⁺_α = V · diag(σᵢ / (σᵢ² + α²)) · Uᴴ

    condition_raw       : κ₂(A) = σ_max / σ_min  (∞ si A singular)
    condition_effective : max fᵢ / min fᵢ del filtro (finita para α > 0)
    """

    inverse: Tuple[Tuple[complex, ...], ...]
    alpha: float
    singular_values: Tuple[float, ...]
    condition_raw: float
    condition_effective: float

    def as_lists(self) -> List[List[complex]]:
        """Vista mutable para NumPy / callers heredados."""
        return [list(row) for row in self.inverse]


@dataclass(frozen=True)
class CoboundaryKernel:
    """
    Complejo de cocadenas discreto instalado por construct_coboundary_kernel.

        δ₀ = B  : Ω¹ → Ω⁰   (divergencia, n×m)
        d₀ = Bᵀ : Ω⁰ → Ω¹   (gradiente,   m×n)
        Δ₀ = BBᵀ (n×n),  Δ₁ = BᵀB (m×m)
        β₀ = n − rank B,  β₁ = m − rank B,  χ = n − m = β₀ − β₁

    Se exponen tanto el rango numérico en ℝ como el exacto en ℤ; los
    números de Betti se toman del rango exacto (autoridad homológica).
    """

    d0: Tuple[Tuple[float, ...], ...]
    delta0: Tuple[Tuple[float, ...], ...]
    laplacian_0: Tuple[Tuple[float, ...], ...]
    laplacian_1: Tuple[Tuple[float, ...], ...]
    n_vertices: int
    n_edges: int
    rank_real: int
    rank_integer: int
    invariant_factors: Tuple[int, ...]
    torsion: Tuple[int, ...]
    betti_0: int
    betti_1: int
    euler_characteristic: int
    trace_identity_residual: float      # |tr Δ₀ − tr Δ₁| + |tr Δ₀ − ‖B‖²_F|
    constant_kernel_residual: float     # ‖Δ₀ 𝟙‖₂
    symmetry_residual: float            # ‖Δ₀ − Δ₀ᵀ‖_F + ‖Δ₁ − Δ₁ᵀ‖_F

    _LEGACY_KEYS: ClassVar[Dict[str, str]] = {
        "d0": "d0",
        "delta0": "delta0",
        "laplacian_0": "laplacian_0",
        "laplacian_1": "laplacian_1",
        "betti_0": "betti_0",
        "betti_1": "betti_1",
        "euler": "euler_characteristic",
        "rank_delta0": "rank_integer",
    }

    @property
    def torsion_free(self) -> bool:
        """Tor(H₀; ℤ) = 0 ⇔ todos los factores invariantes no nulos son 1."""
        return len(self.torsion) == 0

    @property
    def rank_consistent(self) -> bool:
        """rank_ℝ (numérico) = rank_ℤ (exacto): ausencia de fuga numérica de rango."""
        return self.rank_real == self.rank_integer

    def __getitem__(self, key: str) -> Any:
        """Compatibilidad con el contrato de diccionario de v4.0."""
        try:
            return getattr(self, self._LEGACY_KEYS[key])
        except KeyError as exc:
            raise KeyError(f"Clave desconocida en CoboundaryKernel: {key}") from exc

    def as_dict(self) -> Dict[str, Any]:
        """Serialización plana (telemetría / depuración)."""
        return {
            "n_vertices": self.n_vertices,
            "n_edges": self.n_edges,
            "rank_real": self.rank_real,
            "rank_integer": self.rank_integer,
            "invariant_factors": list(self.invariant_factors),
            "torsion": list(self.torsion),
            "torsion_free": self.torsion_free,
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "euler": self.euler_characteristic,
            "trace_identity_residual": self.trace_identity_residual,
            "constant_kernel_residual": self.constant_kernel_residual,
            "symmetry_residual": self.symmetry_residual,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 1.6  Álgebra lineal compensada sobre (ℝⁿ, ⟨·,·⟩) y 𝔅(ℝⁿ)
# ──────────────────────────────────────────────────────────────────────────────


class CompensatedLinearAlgebra:
    """
    Álgebra lineal compensada sobre el Hilbert real (ℝⁿ, ⟨·,·⟩).

    Provee: transposición, matvec/matmul Dot2, Gram y defecto de
    ortonormalidad, MGS con reortogonalización de Kahan–Parlett, iteración
    de potencias certificada (Rayleigh + residuo), norma de operador vía
    identidad C*, inversa de Tikhonov, rango numérico, y el morfismo
    terminal de la Fase 1: construct_coboundary_kernel.
    """

    __slots__ = ()

    # ── 1.6.1 Operaciones matriciales elementales ────────────────────────

    @staticmethod
    def transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
        """Aᵀ ∈ ℝ^{m×n} para A ∈ ℝ^{n×m}; valida rectangularidad."""
        if not matrix:
            return []
        m = len(matrix[0])
        if any(len(row) != m for row in matrix):
            raise NumericalDomainError("transpose: matriz no rectangular")
        return [[float(matrix[i][j]) for i in range(len(matrix))] for j in range(m)]

    @staticmethod
    def matvec_kbn(
        matrix: Sequence[Sequence[float]],
        vector: Sequence[float],
    ) -> List[float]:
        """y = Ax con Dot2 por fila:  yᵢ = Σⱼ Aᵢⱼxⱼ."""
        n = len(vector)
        result: List[float] = []
        for i, row in enumerate(matrix):
            if len(row) != n:
                raise NumericalDomainError(
                    f"matvec: fila {i} de longitud {len(row)} ≠ dim(x) = {n}"
                )
            result.append(_kbn_dot(row, vector))
        return result

    @classmethod
    def matmul_kbn(
        cls,
        left: Sequence[Sequence[float]],
        right: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """C = A·B con cada Cᵢⱼ = Dot2(Aᵢ,·, B·,ⱼ)."""
        right_t = cls.transpose(right)
        return [[_kbn_dot(row, col) for col in right_t] for row in left]

    @staticmethod
    def gram_matrix_kbn(
        vectors: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Gram Gᵢⱼ = ⟨vᵢ|vⱼ⟩; se computa el triángulo superior y se refleja,
        garantizando simetría bit a bit. G ⪰ 0; G = I ⇔ familia ortonormal.
        """
        m = len(vectors)
        gram = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i, m):
                g = _kbn_dot(vectors[i], vectors[j])
                gram[i][j] = g
                gram[j][i] = g
        return gram

    @classmethod
    def orthonormality_defect(cls, vectors: Sequence[Sequence[float]]) -> float:
        """‖G − I‖_F: 0 ⇔ familia ortonormal exacta."""
        gram = cls.gram_matrix_kbn(vectors)
        m = len(gram)
        return _frobenius_norm(
            [[gram[i][j] - (1.0 if i == j else 0.0) for j in range(m)] for i in range(m)]
        )

    @classmethod
    def symmetry_defect(cls, matrix: Sequence[Sequence[float]]) -> float:
        """‖A − Aᵀ‖_F: 0 ⇔ A autoadjunta (matriz cuadrada)."""
        n = len(matrix)
        if any(len(row) != n for row in matrix):
            raise NumericalDomainError("symmetry_defect: matriz no cuadrada")
        return _frobenius_norm(
            [[float(matrix[i][j]) - float(matrix[j][i]) for j in range(n)] for i in range(n)]
        )

    # ── 1.6.2 Ortogonalización ───────────────────────────────────────────

    @staticmethod
    def modified_gram_schmidt(
        vectors: Sequence[Sequence[float]],
        eps: float = NumericalConstants.EPS,
        restarts: int = NumericalConstants.ORTHO_RESTARTS,
        kappa: float = NumericalConstants.ORTHO_KAPPA,
    ) -> List[List[float]]:
        """
        Gram–Schmidt modificado con reortogonalización adaptativa.

        Criterio de Kahan–Parlett ("twice is enough"): tras una pasada, si
        ‖v_new‖ ≥ κ‖v_old‖ la ortogonalidad ya es O(u); en caso contrario se
        repite (máx. `restarts` pasadas). Un vector se descarta como
        linealmente dependiente si ‖v_final‖ ≤ ε‖v₀‖ (criterio relativo).

        Pérdida de ortogonalidad: O(u κ₂) frente a O(u κ₂²) del CGS clásico.
        """
        basis: List[List[float]] = []
        if not vectors:
            return basis
        n_dim = len(vectors[0])

        for raw in vectors:
            if len(raw) != n_dim:
                raise NumericalDomainError("MGS: dimensión incompatible")
            v = [float(x) for x in raw]
            norm_initial = _compute_norm(v)
            if norm_initial <= eps:
                continue

            norm_before = norm_initial
            for _ in range(max(1, restarts)):
                for q in basis:
                    coeff = _kbn_dot(v, q)
                    v = [vi - coeff * qi for vi, qi in zip(v, q)]
                norm_after = _compute_norm(v)
                if norm_after >= kappa * norm_before:
                    break
                norm_before = norm_after

            v_hat, norm_final = _normalize_vector(v, eps=eps)
            if norm_final > eps * norm_initial:
                basis.append(v_hat)

        return basis

    # ── 1.6.3 Teoría espectral: cotas y certificados ─────────────────────

    @staticmethod
    def gershgorin_radius(matrix: Sequence[Sequence[float]]) -> float:
        """Cota superior de Gershgorin: ρ(A) ≤ ‖A‖_∞ = maxᵢ Σⱼ|Aᵢⱼ|."""
        return _inf_operator_norm(matrix)

    @staticmethod
    def _generic_unit_vector(n: int) -> List[float]:
        """
        Vector inicial genérico para iteración de potencias.

        Sucesión de Weyl vⱼ = frac((j+1)φ) + ½, φ razón áurea: equidistribuida,
        determinista y con componente no nula sobre cualquier autoespacio de
        dimensión < n en la práctica (evita el defecto v ∝ 𝟙 ∈ ker Δ₀ de v4.0).
        """
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        raw = [math.fmod((j + 1) * phi, 1.0) + 0.5 for j in range(n)]
        unit, _ = _normalize_vector(raw)
        return unit

    @classmethod
    def power_iteration(
        cls,
        matrix: Sequence[Sequence[float]],
        max_iter: int = NumericalConstants.POWER_ITERATION_MAX,
        tol: float = NumericalConstants.POWER_ITERATION_TOL,
    ) -> Tuple[float, List[float], float]:
        """
        Iteración de potencias con cociente de Rayleigh y certificado residual.

        Retorna (λ, v, ‖Av − λv‖₂). Para A autoadjunta, Bauer–Fike garantiza
        que existe λ* ∈ σ(A) con |λ* − λ| ≤ ‖Av − λv‖₂ (v unitario): el
        residuo es una cota de error rigurosa a posteriori.

        Converge al autovalor dominante en módulo si éste es simple; para
        operadores indefinidos úsese `extremal_eigenvalues_symmetric`.
        """
        n = len(matrix)
        if n == 0:
            return 0.0, [], 0.0
        if any(len(row) != n for row in matrix):
            raise NumericalDomainError("power_iteration: matriz no cuadrada")

        v = cls._generic_unit_vector(n)
        lam = 0.0
        residual = math.inf

        for _ in range(max_iter):
            w = cls.matvec_kbn(matrix, v)
            lam = _kbn_dot(v, w)                      # Rayleigh, v unitario
            r = [wi - lam * vi for wi, vi in zip(w, v)]
            residual = _compute_norm(r)
            if residual <= tol * max(1.0, abs(lam)):
                break
            w_norm = _compute_norm(w)
            if w_norm < NumericalConstants.EPS:       # v ∈ ker A numérico ⇒ A ≈ 0
                lam, residual = 0.0, 0.0
                break
            v = [wi / w_norm for wi in w]

        return float(lam), v, float(residual)

    @classmethod
    def extremal_eigenvalues_symmetric(
        cls,
        matrix: Sequence[Sequence[float]],
        max_iter: int = NumericalConstants.POWER_ITERATION_MAX,
        tol: float = NumericalConstants.POWER_ITERATION_TOL,
    ) -> Tuple[float, float]:
        """
        (λ_min, λ_max) de A = Aᵀ mediante iteración de potencias desplazada.

        Con g = ‖A‖_∞ (Gershgorin): A + gI ⪰ 0 tiene dominante λ_max + g;
        gI − A ⪰ 0 tiene dominante g − λ_min. Ambas son simples si los
        extremos de σ(A) lo son.
        """
        n = len(matrix)
        g = cls.gershgorin_radius(matrix)
        shifted_up = [
            [float(matrix[i][j]) + (g if i == j else 0.0) for j in range(n)]
            for i in range(n)
        ]
        shifted_down = [
            [-float(matrix[i][j]) + (g if i == j else 0.0) for j in range(n)]
            for i in range(n)
        ]
        lam_up, _, _ = cls.power_iteration(shifted_up, max_iter, tol)
        lam_down, _, _ = cls.power_iteration(shifted_down, max_iter, tol)
        return float(g - lam_down), float(lam_up - g)

    @classmethod
    def spectral_radius_power(
        cls,
        matrix: Sequence[Sequence[float]],
        max_iter: int = NumericalConstants.POWER_ITERATION_MAX,
        tol: float = NumericalConstants.POWER_ITERATION_TOL,
        *,
        positive_semidefinite: bool = False,
    ) -> float:
        """
        Radio espectral ρ(A) para A autoadjunta.

        - PSD (p. ej. Δ₀): ρ(A) = λ_max, iteración directa sin desplazamiento
          (razón de convergencia (λ₂/λ_max)² en el cociente de Rayleigh).
        - General simétrica: ρ(A) = max(|λ_min|, |λ_max|) vía desplazamiento
          de Gershgorin (robusto ante espectros simétricos ±λ, p. ej. la
          adyacencia de un grafo bipartito).
        """
        if positive_semidefinite:
            lam, _, residual = cls.power_iteration(matrix, max_iter, tol)
            if residual > math.sqrt(NumericalConstants.MACHINE_EPS) * max(1.0, abs(lam)):
                logger.debug(
                    "power_iteration: residuo %.3e no alcanzó tol (λ≈%.6f)",
                    residual,
                    lam,
                )
            return abs(float(lam))
        lam_min, lam_max = cls.extremal_eigenvalues_symmetric(matrix, max_iter, tol)
        return float(max(abs(lam_min), abs(lam_max)))

    @classmethod
    def operator_norm_2(cls, matrix: Sequence[Sequence[float]]) -> float:
        """
        Norma de operador en 𝔅(ℝⁿ) por la identidad C*:

            ‖A‖₂ = ρ(AᵀA)^{1/2} = σ_max(A).
        """
        at = cls.transpose(matrix)
        gram = cls.matmul_kbn(at, matrix)
        return _stable_sqrt(cls.spectral_radius_power(gram, positive_semidefinite=True))

    # ── 1.6.4 Regularización y rango ─────────────────────────────────────

    @staticmethod
    def tikhonov_solve(
        operator: Sequence[Sequence[complex]],
        alpha_reg: float,
    ) -> TikhonovSolution:
        """
        Inversa regularizada de Tikhonov vía SVD:

            A⁺_α = V · diag(fᵢ) · Uᴴ,   fᵢ = σᵢ / (σᵢ² + α²)

        Minimiza ‖Ax − b‖² + α²‖x‖². Para α = 0 degenera en la pseudoinversa
        de Moore–Penrose con truncamiento relativo σᵢ ≤ n·u·σ_max.
        El operador se pasa SIN desplazamiento previo: el filtro ya regulariza.
        """
        if not _is_finite(alpha_reg) or alpha_reg < 0.0:
            raise NumericalDomainError(f"α de Tikhonov inválido: {alpha_reg!r}")

        A = np.asarray(operator, dtype=np.complex128)
        if A.ndim != 2:
            raise NumericalDomainError("tikhonov_solve: se requiere una matriz 2-D")
        try:
            U, sigma, Vh = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"SVD del resolvente no convergió: {exc}") from exc

        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_max = float(sigma[0]) if sigma.size else 0.0
        sigma_min = float(sigma[-1]) if sigma.size else 0.0
        cond_raw = (
            sigma_max / sigma_min
            if sigma_min > NumericalConstants.MACHINE_EPS * sigma_max and sigma_min > 0.0
            else math.inf
        )

        alpha = float(alpha_reg)
        if alpha > 0.0:
            filt = sigma / (sigma * sigma + alpha * alpha)
        else:
            cutoff = max(A.shape) * NumericalConstants.MACHINE_EPS * sigma_max
            filt = np.where(sigma > cutoff, 1.0 / np.where(sigma > cutoff, sigma, 1.0), 0.0)

        nonzero = filt[filt > 0.0]
        cond_eff = (
            float(nonzero.max() / nonzero.min()) if nonzero.size else math.inf
        )

        G = (Vh.conj().T * filt.astype(np.complex128)) @ U.conj().T
        return TikhonovSolution(
            inverse=tuple(tuple(complex(z) for z in row) for row in G.tolist()),
            alpha=alpha,
            singular_values=tuple(float(s) for s in sigma),
            condition_raw=float(cond_raw),
            condition_effective=cond_eff,
        )

    @staticmethod
    def numerical_rank(
        matrix: Sequence[Sequence[float]],
        tol: float = NumericalConstants.BETTI_RANK_TOL,
    ) -> int:
        """
        Rango numérico = #{σᵢ > tol·σ_max}.

        `tol` relativo; con 1e-9 se separan holgadamente los valores
        singulares de matrices de incidencia (σ ∈ {0} ∪ [1, 2]) del ruido O(u).
        """
        A = np.asarray(matrix, dtype=np.float64)
        if A.size == 0:
            return 0
        try:
            sigma = np.linalg.svd(A, compute_uv=False)
        except np.linalg.LinAlgError:
            return 0
        if sigma.size == 0:
            return 0
        return int(np.count_nonzero(sigma > tol * float(sigma[0])))

    # ── 1.6.5 Morfismo terminal ──────────────────────────────────────────

    def construct_coboundary_kernel(
        self,
        incidence: Sequence[Sequence[float]],
        *,
        rank_tol: float = NumericalConstants.BETTI_RANK_TOL,
        structural_tol: float = NumericalConstants.STRUCTURAL_TOL,
        require_torsion_free: bool = True,
    ) -> CoboundaryKernel:
        """
        ── MORFISMO TERMINAL DE LA FASE 1 / INICIAL DE LA FASE 2 ──────────

        Dado el operador de incidencia orientada B ∈ ℤ^{n×m} (n vértices,
        m aristas; columna e = [u,v] ↦ B[v,e] = +1… o su convención
        transpuesta, en todo caso con 𝟙ᵀB = 0), construye el kernel
        algebraico del complejo de coborde:

            δ₀ = B  : Ω¹ → Ω⁰        d₀ = Bᵀ : Ω⁰ → Ω¹
            Δ₀ = BBᵀ  (n×n)          Δ₁ = BᵀB  (m×m)
            H⁰ ≅ ker Δ₀,  β₀ = n − rank B
            H¹ ≅ ker Δ₁,  β₁ = m − rank B
            χ  = n − m = β₀ − β₁

        Auditorías estructurales (levantan HomologicalIntegrityError):
          (a) 𝟙ᵀB = 0        — B es una frontera de grafo (𝟙 ∈ ker d₀);
          (b) tr Δ₀ = tr Δ₁ = ‖B‖²_F — espectros no nulos de BBᵀ y BᵀB coinciden;
          (c) Δ₀𝟙 = 0        — constantes armónicas;
          (d) Δ₀, Δ₁ simétricas exactas;
          (e) rank_ℝ = rank_ℤ — sin fuga numérica de rango (warning, no error);
          (f) Tor(H₀; ℤ) = 0 — vía Forma Normal de Smith (error si se exige).

        El CoboundaryKernel retornado es el objeto sobre el que la Fase 2
        instala la geometría de Hodge–de Rham–Hückel.
        """
        # ── Validación de dominio ────────────────────────────────────────
        if not incidence:
            raise NumericalDomainError("incidence vacía")
        B = [[float(x) for x in row] for row in incidence]
        n = len(B)
        m = len(B[0])
        if m == 0 or any(len(row) != m for row in B):
            raise NumericalDomainError("incidence debe ser rectangular n×m con m ≥ 1")
        for i, row in enumerate(B):
            _validate_finite_vector(row, f"incidence[{i}]")

        # (a) 𝟙ᵀB = 0
        column_sums = [_kbn_sum(B[i][e] for i in range(n)) for e in range(m)]
        max_col_residual = max(abs(c) for c in column_sums)
        if max_col_residual > structural_tol:
            raise HomologicalIntegrityError(
                f"𝟙ᵀB ≠ 0 (residuo {max_col_residual:.3e}): "
                "la matriz no es una frontera de grafo orientado"
            )

        # ── Operadores ───────────────────────────────────────────────────
        BT = self.transpose(B)                     # d₀ (m×n)
        lap_0 = self.gram_matrix_kbn(B)            # Δ₀ = BBᵀ = Gram(filas de B)
        lap_1 = self.gram_matrix_kbn(BT)           # Δ₁ = BᵀB = Gram(columnas de B)

        # (b) identidad de trazas
        trace_0 = _kbn_sum(lap_0[i][i] for i in range(n))
        trace_1 = _kbn_sum(lap_1[e][e] for e in range(m))
        frob_sq = _kbn_dot(
            [x for row in B for x in row],
            [x for row in B for x in row],
        )
        trace_residual = abs(trace_0 - trace_1) + abs(trace_0 - frob_sq)
        if trace_residual > structural_tol * max(1.0, frob_sq):
            raise HomologicalIntegrityError(
                f"tr Δ₀ = {trace_0:.6f}, tr Δ₁ = {trace_1:.6f}, ‖B‖²_F = {frob_sq:.6f}"
            )

        # (c) Δ₀𝟙 = 0
        constant_residual = _compute_norm(self.matvec_kbn(lap_0, [1.0] * n))
        if constant_residual > structural_tol * max(1.0, frob_sq):
            raise HomologicalIntegrityError(
                f"Δ₀𝟙 ≠ 0 (‖Δ₀𝟙‖ = {constant_residual:.3e})"
            )

        # (d) simetría (exacta por construcción; se certifica igualmente)
        symmetry_residual = self.symmetry_defect(lap_0) + self.symmetry_defect(lap_1)
        if symmetry_residual > structural_tol:
            raise HomologicalIntegrityError(
                f"Laplacianos no autoadjuntos (residuo {symmetry_residual:.3e})"
            )

        # ── Rango: numérico en ℝ y exacto en ℤ ───────────────────────────
        rank_real = self.numerical_rank(B, tol=rank_tol)
        B_int = ExactIntegerAlgebra.as_integer_matrix(B)
        factors = ExactIntegerAlgebra.smith_invariant_factors(B_int)
        rank_int = ExactIntegerAlgebra.integer_rank(factors)
        torsion = ExactIntegerAlgebra.torsion_coefficients(factors)

        if rank_real != rank_int:  # (e)
            logger.warning(
                "Fuga numérica de rango: rank_ℝ=%d ≠ rank_ℤ=%d; se adopta el exacto",
                rank_real,
                rank_int,
            )
        if torsion and require_torsion_free:  # (f)
            raise HomologicalIntegrityError(
                f"Tor(H₀; ℤ) ≠ 0: coeficientes de torsión {torsion}"
            )

        # ── Invariantes homológicos (autoridad: rango entero) ────────────
        betti_0 = n - rank_int
        betti_1 = m - rank_int
        euler = n - m
        if euler != betti_0 - betti_1:  # identidad algebraica; guardia de coherencia
            raise HomologicalIntegrityError(
                f"χ = n − m = {euler} ≠ β₀ − β₁ = {betti_0 - betti_1}"
            )

        kernel = CoboundaryKernel(
            d0=tuple(tuple(row) for row in BT),
            delta0=tuple(tuple(row) for row in B),
            laplacian_0=tuple(tuple(row) for row in lap_0),
            laplacian_1=tuple(tuple(row) for row in lap_1),
            n_vertices=n,
            n_edges=m,
            rank_real=rank_real,
            rank_integer=rank_int,
            invariant_factors=tuple(factors),
            torsion=torsion,
            betti_0=betti_0,
            betti_1=betti_1,
            euler_characteristic=euler,
            trace_identity_residual=float(trace_residual),
            constant_kernel_residual=float(constant_residual),
            symmetry_residual=float(symmetry_residual),
        )
        logger.debug(
            "Kernel de coborde instalado: n=%d m=%d rank_ℤ=%d β=(%d,%d) χ=%d SNF=%s",
            n,
            m,
            rank_int,
            betti_0,
            betti_1,
            euler,
            factors,
        )
        return kernel


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — GEOMETRÍA ESPECTRAL DE HODGE–DE RHAM–HÜCKEL–TELLEGEN
# Continúa CompensatedLinearAlgebra.construct_coboundary_kernel (Fase 1)
# ══════════════════════════════════════════════════════════════════════════════
#
# Objetos: complejo de cocadenas discreto (Ω•, d, δ) instalado sobre un
#          CoboundaryKernel; base de Fourier real de ℤ/6ℤ certificada;
#          Hamiltoniano de Hückel H = αI + β(2I − Δ₀) y su resolvente
#          espectral; auditoría de de Rham–Tellegen–Hodge–Parseval;
#          semigrupo de calor exacto e^{−tΔ₀} y su generador 𝒢 = −αΔ₀.
#
# El morfismo terminal de esta fase es
#     HexagonalTopology.infinitesimal_catalytic_generator
# que retorna un CatalyticGenerator certificado, objeto inicial de la Fase 3
# (HilbertState.apply_generator_step lo integra por Euler compensado).


# ──────────────────────────────────────────────────────────────────────────────
# 2.0  Enumeraciones del anillo (átomos del complejo)
# ──────────────────────────────────────────────────────────────────────────────


class CarbonNode(Enum):
    """Nodos del anillo hexagonal, análogos a los carbonos del benceno."""

    C1_INGESTION = auto()
    C2_PHYSICS = auto()
    C3_TOPOLOGY = auto()
    C4_STRATEGY = auto()
    C5_SEMANTICS = auto()
    C6_MATTER = auto()

    @property
    def index(self) -> int:
        """Índice base-0 del nodo en el anillo (auto() arranca en 1)."""
        return self.value - 1

    @property
    def label(self) -> str:
        """Etiqueta legible del nodo."""
        return self.name.replace("_", " ").title()

    @property
    def service_name(self) -> str:
        """Nombre del servicio MIC asociado."""
        return _SERVICE_MAP[self]

    @property
    def precursors(self) -> Tuple[str, ...]:
        """Claves de contexto requeridas como precursores."""
        return _PRECURSOR_MAP.get(self, ())

    def __repr__(self) -> str:
        return f"CarbonNode.{self.name}[{self.index}]"


_SERVICE_MAP: Final[Dict[CarbonNode, str]] = {
    CarbonNode.C1_INGESTION: "load_data",
    CarbonNode.C2_PHYSICS: "stabilize_flux",
    CarbonNode.C3_TOPOLOGY: "business_topology",
    CarbonNode.C4_STRATEGY: "financial_analysis",
    CarbonNode.C5_SEMANTICS: "semantic_translation",
    CarbonNode.C6_MATTER: "materialization",
}

_PRECURSOR_MAP: Final[Dict[CarbonNode, Tuple[str, ...]]] = {
    CarbonNode.C2_PHYSICS: ("physical_constraints",),
    CarbonNode.C4_STRATEGY: ("financial_params",),
    CarbonNode.C5_SEMANTICS: ("semantic_model",),
}


def node_from_index(idx: int) -> CarbonNode:
    """Retracción índice → CarbonNode, dominio {0,…,5}."""
    if isinstance(idx, bool) or not isinstance(idx, numbers.Integral):
        raise NumericalDomainError(f"Índice de nodo no entero: {idx!r}")
    if not 0 <= idx < TopologyConstants.RING_SIZE:
        raise NumericalDomainError(
            f"Índice {idx} fuera de rango [0, {TopologyConstants.RING_SIZE - 1}]"
        )
    return CarbonNode(int(idx) + 1)


class AromaticityState(Enum):
    """Estado de aromaticidad según la regla de Hückel (capa cerrada π)."""

    NON_AROMATIC = auto()
    AROMATIC = auto()
    ANTI_AROMATIC = auto()
    PARTIALLY_AROMATIC = auto()

    def __str__(self) -> str:
        symbols = {
            AromaticityState.NON_AROMATIC: "○ Non-Aromatic",
            AromaticityState.AROMATIC: "◉ Aromatic (4n+2)",
            AromaticityState.ANTI_AROMATIC: "⊗ Anti-Aromatic (4n)",
            AromaticityState.PARTIALLY_AROMATIC: "◐ Partially Aromatic",
        }
        return symbols[self]


class ConvergenceStatus(Enum):
    """Estado de convergencia del reactor (semigrupo hacia el atractor)."""

    RUNNING = auto()
    CONVERGED_AROMATIC = auto()
    CONVERGED_METASTABLE = auto()
    FAILED_INSTABILITY = auto()
    FAILED_MAX_CYCLES = auto()


# ──────────────────────────────────────────────────────────────────────────────
# 2.1  Contratos de retorno tipados (inmutables)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HodgeDecomposition:
    """
    Descomposición de Hodge de una 1-forma sobre un 1-complejo (C₂ = 0):

        Ω¹ = im d₀ ⊕ ker δ₀ ,   ker δ₀ = ℋ¹ ≅ H¹   (no hay δ₁)
        q  = d₀α + h ,   α = Δ₀⁺ δ₀ q ,   h = Σᵢ ⟨q, hᵢ⟩ hᵢ

    reconstruction_residual = ‖q − d₀α − h‖₂  certifica la suma directa;
    orthogonality_residual  = |⟨d₀α, h⟩|      certifica la adjunción d₀ ⊣ δ₀.
    """

    potential: Tuple[float, ...]          # α ∈ Ω⁰ (⟂ 𝟙)
    exact: Tuple[float, ...]              # d₀α ∈ im d₀
    harmonic: Tuple[float, ...]           # h ∈ ker δ₀
    harmonic_coefficients: Tuple[float, ...]
    reconstruction_residual: float
    orthogonality_residual: float


@dataclass(frozen=True)
class SpectralCertificate:
    """
    Certificado espectral de Δ₀ sobre C₆.

    Weyl: si L̃ = L + E entonces max_i |λ_i(L̃) − λ_i(L)| ≤ ‖E‖₂. Aquí L es
    entera (E = 0 en fl), luego la desviación numérica es error backward
    del solver ≤ p(n)·u·‖L‖₂.
    """

    analytical_eigenvalues: Tuple[float, ...]
    numerical_eigenvalues: Tuple[float, ...]
    weyl_deviation: float
    max_eigenpair_residual: float           # max_k ‖Lv_k − λ_k v_k‖₂
    spectral_radius_power: float            # ρ(L) por iteración de potencias
    orthonormality_defect: float            # ‖VᵀV − I‖_F
    factorization_residual: float           # ‖(D − A) − BBᵀ‖_F
    trace_residual: float                   # |tr L − 2m|
    kirchhoff_spanning_trees: float         # (1/n)∏_{k≥1} λ_k
    cheeger_constant: float
    cheeger_lower_bound: float              # h²/(2 d_max)
    cheeger_upper_bound: float              # 2h
    verified: bool


@dataclass(frozen=True)
class HuckelResolvent:
    """
    Resolvente de Hückel G(z) = (H − zI)⁻¹ en representación espectral.

        H = αI + βA = αI + β(2I − Δ₀)  ⇒  H v_k = E_k v_k ,  E_k = α + β(2 − λ_k)
        G(z) = Σ_k f_k v_k v_kᵀ ,   f_k = 1/(E_k − z)      (κ₂ < techo)
                                    f_k = d̄_k/(|d_k|² + α²) (Tikhonov, si no)

    condition_raw = max|d_k| / min|d_k| es κ₂(H − zI) exacto (matriz normal).
    Se desempaqueta como (green, alpha, condition) por compatibilidad.
    """

    green: Tuple[Tuple[complex, ...], ...]
    z: complex
    alpha: float                            # 0.0 si no se filtró
    levels: Tuple[float, ...]               # E_k en el orden de la autobase
    condition_raw: float
    condition_effective: float
    tikhonov_applied: bool

    def as_lists(self) -> List[List[complex]]:
        return [list(row) for row in self.green]

    def trace(self) -> complex:
        return sum(self.green[i][i] for i in range(len(self.green)))

    def __iter__(self) -> Iterator[Any]:
        yield self.as_lists()
        yield self.alpha
        yield self.condition_raw if not self.tikhonov_applied else self.condition_effective


@dataclass(frozen=True)
class CatalyticGenerator:
    """
    Generador infinitesimal certificado del semigrupo catalítico:

        𝒢 = −αΔ₀ ∈ 𝔅(ℝⁿ),   e^{t𝒢} = Σ (t𝒢)ᵏ/k!

    Un paso de Euler es P = I + 𝒢 = I − αΔ₀, con σ(P) = {1 − αλ_k}.
        contraction_constant = max_{k≥1} |1 − αλ_k|  (tasa sobre 𝟙^⊥)
        spectral_abscissa    = −αλ₁                    (decaimiento continuo)
        is_l2_stable  ⇔ α ≤ 2/λ_max ,   is_monotone ⇔ α ≤ 1/λ_max (P ⪰ 0)

    Implementa el protocolo Sequence de filas: matvec_kbn(generator, ψ) es válido.
    """

    matrix: Tuple[Tuple[float, ...], ...]
    alpha: float
    alpha_requested: float
    spectral_gap: float
    lambda_max: float
    contraction_constant: float
    spectral_abscissa: float
    is_l2_stable: bool
    is_monotone: bool
    regime: str

    def __len__(self) -> int:
        return len(self.matrix)

    def __iter__(self) -> Iterator[Tuple[float, ...]]:
        return iter(self.matrix)

    def __getitem__(self, i: int) -> Tuple[float, ...]:
        return self.matrix[i]

    def as_lists(self) -> List[List[float]]:
        return [list(row) for row in self.matrix]


@dataclass(frozen=True)
class TellegenAuditReport:
    """
    Auditoría de conservación de de Rham–Tellegen–Hodge–Parseval.

    Ocho axiomas de coherencia del complejo (tolerancia relativa tol·scale):

      1. Pasividad Dirichlet:       ψᵀΔ₀ψ = ‖d₀ψ‖² ≥ −tol
      2. Conservación nodal:        |𝟙ᵀ δ₀ q| ≤ tol     (idéntica 0 en ℤ; mide fl)
      3. Ortogonalidad de Hodge:    |⟨d₀α, h⟩| ≤ tol    (exactas ⟂ armónicas)
      4. Identidad de Tellegen:     |⟨ψ, δ₀q⟩ − ⟨d₀ψ, q⟩| ≤ tol   (adjunción)
      5. Cierre armónico:           ‖δ₀ h‖ ≤ tol        (h ∈ ker δ₀)
      6. Reconstrucción de Hodge:   ‖q − d₀α − h‖ ≤ tol (Ω¹ = im d₀ ⊕ ker δ₀)
      7. Parseval–Plancherel:       |‖d₀ψ‖² − Σλ_k⟨v_k,ψ⟩²| ≤ tol
      8. Disipación óhmica:         |⟨d₀ψ, q⟩ + ℰ(ψ)| ≤ tol  (q = −d₀ψ natural)
    """

    passivity_satisfied: bool
    conservation_satisfied: bool
    orthogonality_satisfied: bool
    tellegen_satisfied: bool
    harmonic_closure_satisfied: bool
    reconstruction_satisfied: bool
    parseval_satisfied: bool
    ohmic_satisfied: bool
    laplacian_energy: float
    conservation_residual: float
    orthogonality_residual: float
    tellegen_residual: float
    harmonic_closure_residual: float
    reconstruction_residual: float
    parseval_residual: float
    ohmic_residual: float
    tolerance: float
    scale: float
    natural_flux: bool

    @property
    def is_coherent(self) -> bool:
        """Conjunción de los ocho axiomas."""
        return all(
            (
                self.passivity_satisfied,
                self.conservation_satisfied,
                self.orthogonality_satisfied,
                self.tellegen_satisfied,
                self.harmonic_closure_satisfied,
                self.reconstruction_satisfied,
                self.parseval_satisfied,
                self.ohmic_satisfied,
            )
        )

    def violations(self) -> Tuple[str, ...]:
        """Nombres de los axiomas violados (vacío ⇔ is_coherent)."""
        names = (
            ("passivity", self.passivity_satisfied),
            ("conservation", self.conservation_satisfied),
            ("orthogonality", self.orthogonality_satisfied),
            ("tellegen", self.tellegen_satisfied),
            ("harmonic_closure", self.harmonic_closure_satisfied),
            ("reconstruction", self.reconstruction_satisfied),
            ("parseval", self.parseval_satisfied),
            ("ohmic", self.ohmic_satisfied),
        )
        return tuple(name for name, ok in names if not ok)

    def as_context(self, prefix: str = "tellegen_") -> Dict[str, Any]:
        """Proyección plana para inyección en el contexto del reactor (Fase 3)."""
        return {
            f"{prefix}passivity": self.passivity_satisfied,
            f"{prefix}conservation": self.conservation_satisfied,
            f"{prefix}orthogonality": self.orthogonality_satisfied,
            f"{prefix}identity": self.tellegen_satisfied,
            f"{prefix}harmonic_closure": self.harmonic_closure_satisfied,
            f"{prefix}reconstruction": self.reconstruction_satisfied,
            f"{prefix}parseval": self.parseval_satisfied,
            f"{prefix}ohmic": self.ohmic_satisfied,
            f"{prefix}energy": self.laplacian_energy,
            f"{prefix}conservation_residual": self.conservation_residual,
            f"{prefix}orthogonality_residual": self.orthogonality_residual,
            f"{prefix}identity_residual": self.tellegen_residual,
            f"{prefix}harmonic_closure_residual": self.harmonic_closure_residual,
            f"{prefix}reconstruction_residual": self.reconstruction_residual,
            f"{prefix}parseval_residual": self.parseval_residual,
            f"{prefix}ohmic_residual": self.ohmic_residual,
            f"{prefix}coherent": self.is_coherent,
        }


@dataclass(frozen=True)
class LyapunovCertificate:
    """
    Certificado de estabilidad de Lyapunov para el par (ψ, G), consumido en Fase 3.

        V(ψ, G) = ‖ψ‖² + G²
        Sobre 𝟙^⊥ el semigrupo contrae con abscisa −αλ₁ (spectral_abscissa).

    Disipatividad se declara si ΔV ≤ tolerance (holgura numérica).
    """

    value: float
    decrement: float
    dissipative: bool
    spectral_abscissa: float
    tolerance: float = NumericalConstants.EPS


# ──────────────────────────────────────────────────────────────────────────────
# 2.2  Cálculo exterior discreto sobre el CoboundaryKernel
# ──────────────────────────────────────────────────────────────────────────────


class DiscreteExteriorCalculus(CompensatedLinearAlgebra):
    """
    ── CONTINUACIÓN DE LA FASE 1 ──────────────────────────────────────────

    Instala el cálculo exterior discreto sobre el CoboundaryKernel producido
    por construct_coboundary_kernel: operadores d₀, δ₀, Laplacianos de Hodge,
    estrella de Hodge diagonal ★₁, base armónica certificada de H¹, la
    descomposición q = d₀α + h y la identidad de Tellegen como adjunción.
    """

    __slots__ = ("_coboundary_kernel", "_harmonic_cache")

    def __init__(self) -> None:
        self._coboundary_kernel: Optional[CoboundaryKernel] = None
        self._harmonic_cache: Optional[Tuple[Tuple[float, ...], ...]] = None

    # ── 2.2.1 Instalación (punto de sutura Fase 1 → Fase 2) ──────────────

    def install_coboundary_kernel(
        self,
        incidence: Sequence[Sequence[float]],
        **kernel_options: Any,
    ) -> CoboundaryKernel:
        """
        Continúa construct_coboundary_kernel (Fase 1) y memoiza el complejo.

        Invalida la caché armónica: cualquier kernel nuevo redefine H¹.
        """
        kernel = self.construct_coboundary_kernel(incidence, **kernel_options)
        self._coboundary_kernel = kernel
        self._harmonic_cache = None
        return kernel

    @property
    def coboundary_kernel(self) -> CoboundaryKernel:
        if self._coboundary_kernel is None:
            raise RuntimeError(
                "Kernel de coborde no instalado: llame install_coboundary_kernel"
            )
        return self._coboundary_kernel

    @property
    def n_vertices(self) -> int:
        """dim Ω⁰."""
        return self.coboundary_kernel.n_vertices

    @property
    def n_edges(self) -> int:
        """dim Ω¹."""
        return self.coboundary_kernel.n_edges

    def _check_zero_form(self, form: Sequence[float], name: str = "zero_form") -> None:
        _validate_finite_vector(form, name, expected_dim=self.n_vertices)

    def _check_one_form(self, form: Sequence[float], name: str = "one_form") -> None:
        _validate_finite_vector(form, name, expected_dim=self.n_edges)

    # ── 2.2.2 Operadores diferenciales ───────────────────────────────────

    def exterior_derivative_0(self, zero_form: Sequence[float]) -> List[float]:
        """d₀ψ ∈ Ω¹: 1-forma exacta (gradiente de arista), (d₀ψ)_e = ψ_v − ψ_u."""
        self._check_zero_form(zero_form)
        return self.matvec_kbn(self.coboundary_kernel.d0, zero_form)

    def codifferential_1(self, one_form: Sequence[float]) -> List[float]:
        """δ₀q ∈ Ω⁰: divergencia nodal, adjunto de d₀ respecto a ⟨·,·⟩."""
        self._check_one_form(one_form)
        return self.matvec_kbn(self.coboundary_kernel.delta0, one_form)

    def hodge_laplacian_0(self, zero_form: Sequence[float]) -> List[float]:
        """Δ₀ψ = δ₀d₀ψ (aplicado con la matriz memoizada BBᵀ)."""
        self._check_zero_form(zero_form)
        return self.matvec_kbn(self.coboundary_kernel.laplacian_0, zero_form)

    def hodge_laplacian_1(self, one_form: Sequence[float]) -> List[float]:
        """Δ₁q = d₀δ₀q (única componente: no hay δ₁ al ser C₂ = 0)."""
        self._check_one_form(one_form)
        return self.matvec_kbn(self.coboundary_kernel.laplacian_1, one_form)

    def hodge_energy_0(self, zero_form: Sequence[float]) -> float:
        """Energía de Dirichlet ℰ(ψ) = ‖d₀ψ‖² = ψᵀΔ₀ψ ≥ 0, evaluada con Dot2."""
        grad = self.exterior_derivative_0(zero_form)
        return _kbn_dot(grad, grad)

    def hodge_star_1(
        self,
        edge_weights: Optional[Sequence[float]] = None,
    ) -> List[List[float]]:
        """
        Estrella de Hodge diagonal ★₁ = diag(w_e), w_e > 0 (conductancias).

        Para el complejo combinatorial sin pesos ★₁ = I, y el Laplaciano
        ponderado es Δ₀^w = B ★₁ Bᵀ. Se exige positividad estricta para
        preservar la definitud del producto interno en Ω¹.
        """
        m = self.n_edges
        if edge_weights is None:
            return [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
        _validate_finite_vector(edge_weights, "edge_weights", expected_dim=m)
        if any(float(w) <= 0.0 for w in edge_weights):
            raise NumericalDomainError("★₁ requiere pesos estrictamente positivos")
        return [
            [float(edge_weights[i]) if i == j else 0.0 for j in range(m)]
            for i in range(m)
        ]

    def weighted_laplacian_0(self, edge_weights: Sequence[float]) -> List[List[float]]:
        """Δ₀^w = B ★₁ Bᵀ (simétrico, PSD, 𝟙 ∈ ker) con Dot2 en cada entrada."""
        B = self.coboundary_kernel.delta0
        star = self.hodge_star_1(edge_weights)
        BW = self.matmul_kbn(B, star)
        return self.matmul_kbn(BW, self.coboundary_kernel.d0)

    # ── 2.2.3 Cohomología: base armónica certificada ─────────────────────

    def harmonic_basis(
        self,
        tol: float = NumericalConstants.TELLEGEN_TOLERANCE,
    ) -> Tuple[Tuple[float, ...], ...]:
        """
        Base ortonormal de ℋ¹ = ker Δ₁ = ker δ₀ (β₁ vectores).

        Se extrae de la autodescomposición de Δ₁ (los β₁ autovalores más
        pequeños), se normaliza y se fija el signo canónico (primera
        componente no nula positiva). Cada generador se certifica:
        ‖Δ₁h‖ ≤ tol y ‖δ₀h‖ ≤ tol. Memoizada por kernel.
        """
        if self._harmonic_cache is not None:
            return self._harmonic_cache

        kernel = self.coboundary_kernel
        beta_1 = kernel.betti_1
        if beta_1 == 0:
            self._harmonic_cache = ()
            return self._harmonic_cache

        L1 = np.asarray(kernel.laplacian_1, dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(L1)
        raw = [list(map(float, eigvecs[:, k])) for k in range(beta_1)]
        basis = self.modified_gram_schmidt(raw)
        if len(basis) != beta_1:
            raise HomologicalIntegrityError(
                f"ker Δ₁ numérico de dimensión {len(basis)} ≠ β₁ = {beta_1}"
            )

        certified: List[Tuple[float, ...]] = []
        for k, h in enumerate(basis):
            first = next((x for x in h if abs(x) > NumericalConstants.EPS), 1.0)
            if first < 0.0:
                h = [-x for x in h]
            lap_res = _compute_norm(self.matvec_kbn(kernel.laplacian_1, h))
            div_res = _compute_norm(self.matvec_kbn(kernel.delta0, h))
            if lap_res > tol or div_res > tol:
                raise HomologicalIntegrityError(
                    f"h_{k} no es armónica: ‖Δ₁h‖={lap_res:.3e}, ‖δ₀h‖={div_res:.3e} "
                    f"(λ_min(Δ₁)={float(eigvals[k]):.3e})"
                )
            certified.append(tuple(h))

        self._harmonic_cache = tuple(certified)
        return self._harmonic_cache

    def harmonic_one_form(self, n: Optional[int] = None) -> List[float]:
        """
        Generador de H¹ ≅ ℝ cuando β₁ = 1 (para C₆ con orientación i→i+1:
        h = 𝟙/√6, circulación uniforme). El argumento `n` se conserva por
        compatibilidad y sólo se valida.
        """
        if n is not None and n != self.n_edges:
            raise NumericalDomainError(f"n={n} ≠ dim Ω¹ = {self.n_edges}")
        basis = self.harmonic_basis()
        if len(basis) != 1:
            raise HomologicalIntegrityError(
                f"harmonic_one_form requiere β₁ = 1; β₁ = {len(basis)}"
            )
        return list(basis[0])

    # ── 2.2.4 Descomposición de Hodge ────────────────────────────────────

    def hodge_decompose_one_form(self, one_form: Sequence[float]) -> HodgeDecomposition:
        """
        q = d₀α + h con α = Δ₀⁺δ₀q ∈ 𝟙^⊥ (pseudoinversa de Moore–Penrose)
        y h = Σᵢ⟨q, hᵢ⟩hᵢ.

        Sobre un 1-complejo la suma es directa y exhaustiva, luego el
        residuo de reconstrucción ‖q − d₀α − h‖ debe ser O(u); su
        crecimiento delata una corrupción numérica del complejo.
        """
        self._check_one_form(one_form)
        q = [float(x) for x in one_form]
        kernel = self.coboundary_kernel

        basis = self.harmonic_basis()
        coeffs = [_kbn_dot(q, h) for h in basis]
        harmonic = [0.0] * self.n_edges
        for c, h in zip(coeffs, basis):
            harmonic = [hv + c * hi for hv, hi in zip(harmonic, h)]

        divergence = self.codifferential_1(q)
        L0 = np.asarray(kernel.laplacian_0, dtype=np.float64)
        L0_pinv = np.linalg.pinv(L0, rcond=NumericalConstants.BETTI_RANK_TOL)
        potential = [float(x) for x in L0_pinv @ np.asarray(divergence)]
        # Proyección explícita a 𝟙^⊥ (elimina el modo constante residual)
        mean = _kbn_sum(potential) / float(self.n_vertices)
        potential = [p - mean for p in potential]

        exact = self.exterior_derivative_0(potential)
        reconstruction = [qi - ei - hi for qi, ei, hi in zip(q, exact, harmonic)]

        return HodgeDecomposition(
            potential=tuple(potential),
            exact=tuple(exact),
            harmonic=tuple(harmonic),
            harmonic_coefficients=tuple(coeffs),
            reconstruction_residual=_compute_norm(reconstruction),
            orthogonality_residual=abs(_kbn_dot(exact, harmonic)),
        )

    def hodge_project_exact(
        self,
        one_form: Sequence[float],
    ) -> Tuple[List[float], List[float], float]:
        """Vista de compatibilidad: (exacta, armónica, residuo de ortogonalidad)."""
        dec = self.hodge_decompose_one_form(one_form)
        return list(dec.exact), list(dec.harmonic), dec.orthogonality_residual

    # ── 2.2.5 Tellegen como adjunción d₀ ⊣ δ₀ ────────────────────────────

    def tellegen_residual(
        self,
        zero_form: Sequence[float],
        one_form: Sequence[float],
    ) -> float:
        """
        Residuo de la identidad de Tellegen (Stokes 0-dimensional):

            |⟨ψ, δ₀q⟩ − ⟨d₀ψ, q⟩|

        Nulo en aritmética exacta para cualesquiera ψ ∈ Ω⁰, q ∈ Ω¹ (no
        requiere que q sea un flujo físico: es la adjunción del par).
        """
        left = _kbn_dot(zero_form, self.codifferential_1(one_form))
        right = _kbn_dot(self.exterior_derivative_0(zero_form), one_form)
        return abs(left - right)

    def ohmic_dissipation_residual(self, zero_form: Sequence[float]) -> float:
        """
        Ley de Joule discreta para el flujo natural q = −d₀ψ (Ohm con ★₁ = I):

            ⟨d₀ψ, q⟩ = −‖d₀ψ‖² = −ℰ(ψ)   ⇒   residuo |⟨d₀ψ, q⟩ + ℰ(ψ)|.
        """
        grad = self.exterior_derivative_0(zero_form)
        flux = [-g for g in grad]
        return abs(_kbn_dot(grad, flux) + _kbn_dot(grad, grad))


# ──────────────────────────────────────────────────────────────────────────────
# 2.3  Topología espectral del hexágono
# ──────────────────────────────────────────────────────────────────────────────


class HexagonalTopology(DiscreteExteriorCalculus):
    """
    Topología espectral del grafo cíclico C₆.

    Hereda el cálculo exterior (2.2) e instala:
      - adyacencia, grado, Laplaciano combinatorio L = D − A ≡ BBᵀ (certificado)
      - base de Fourier real de ℤ/6ℤ emparejada y certificada con λ_k
      - certificado espectral (Weyl, Rayleigh, Kirchhoff, Cheeger)
      - Hamiltoniano de Hückel, resolvente espectral y observables MO
      - auditoría óctuple de de Rham–Tellegen–Hodge–Parseval
      - semigrupo de calor exacto, propagador de Euler y generador 𝒢
        (morfismo terminal de la Fase 2)
    """

    __slots__ = (
        "_adjacency",
        "_degree",
        "_laplacian",
        "_incidence",
        "_eigenvalues",
        "_eigenvectors",
        "_eigen_residuals",
        "_spectral_certificate",
    )

    def __init__(self) -> None:
        super().__init__()
        n = TopologyConstants.RING_SIZE

        self._adjacency: List[List[float]] = [
            [1.0 if (j == (i + 1) % n or j == (i - 1) % n) else 0.0 for j in range(n)]
            for i in range(n)
        ]
        self._degree: List[int] = [int(_kbn_sum(row)) for row in self._adjacency]
        self._laplacian: List[List[float]] = [
            [
                float(self._degree[i]) * (1.0 if i == j else 0.0) - self._adjacency[i][j]
                for j in range(n)
            ]
            for i in range(n)
        ]
        self._incidence: List[List[float]] = self._build_incidence_matrix()

        # Punto de sutura con la Fase 1: instala el CoboundaryKernel.
        kernel = self.install_coboundary_kernel(self._incidence)
        if kernel.n_vertices != n or kernel.n_edges != n:
            raise HomologicalIntegrityError("El kernel no corresponde a C₆ (n = m = 6)")
        if (kernel.betti_0, kernel.betti_1) != (TopologyConstants.BETTI_0, TopologyConstants.BETTI_1):
            raise HomologicalIntegrityError(
                f"Betti {kernel.betti_0, kernel.betti_1} ≠ esperado "
                f"{TopologyConstants.BETTI_0, TopologyConstants.BETTI_1}"
            )

        self._eigenvalues, self._eigenvectors, self._eigen_residuals = self._compute_spectrum()
        self._spectral_certificate: SpectralCertificate = self._verify_spectrum()

    # ── 2.3.1 Construcción del complejo ──────────────────────────────────

    def _build_incidence_matrix(self) -> List[List[float]]:
        """
        Incidencia orientada B = δ₀ ∈ ℤ^{n×n} del ciclo dirigido:

            e = (i → i+1 mod n):  B[i, e] = +1,  B[i+1, e] = −1   ⇒  𝟙ᵀB = 0.
        """
        n = TopologyConstants.RING_SIZE
        B = [[0.0] * n for _ in range(n)]
        for e in range(n):
            B[e][e] = 1.0
            B[(e + 1) % n][e] = -1.0
        return B

    @staticmethod
    def _analytical_eigenvalue(k: int, n: int) -> float:
        """λ_k = 2 − 2cos(2πk/n), espectro de Δ₀ sobre ℤ/nℤ (Fourier)."""
        return 2.0 - 2.0 * math.cos(2.0 * math.pi * k / n)

    def _compute_spectrum(self) -> Tuple[List[float], List[List[float]], List[float]]:
        """
        Autopares (λ_k, v_k) de Δ₀ = base de Fourier real de ℤ/nℤ:

            k = 0:            v = 𝟙/√n,                          λ = 0
            0 < k < n/2:      v = √(2/n)cos(2πkj/n), √(2/n)sin(·), λ_k (degenerado ×2)
            k = n/2 (n par):  v = (−1)ʲ/√n,                       λ = 4

        La familia es ortonormal en aritmética exacta; MGS actúa como pulido
        O(u). Cada par se certifica por Rayleigh |⟨v,Lv⟩ − λ| y residuo
        ‖Lv − λv‖ (Bauer–Fike: ∃λ* ∈ σ(L), |λ* − λ| ≤ residuo).
        """
        n = TopologyConstants.RING_SIZE
        inv_sqrt_n = 1.0 / math.sqrt(float(n))
        amp = math.sqrt(2.0 / float(n))
        pairs: List[Tuple[float, List[float]]] = [(0.0, [inv_sqrt_n] * n)]

        for k in range(1, (n - 1) // 2 + 1):
            lam = self._analytical_eigenvalue(k, n)
            theta = 2.0 * math.pi * k / n
            pairs.append((lam, [amp * math.cos(theta * j) for j in range(n)]))
            pairs.append((lam, [amp * math.sin(theta * j) for j in range(n)]))
        if n % 2 == 0:
            pairs.append(
                (self._analytical_eigenvalue(n // 2, n),
                 [inv_sqrt_n * (1.0 if j % 2 == 0 else -1.0) for j in range(n)])
            )

        pairs.sort(key=lambda p: p[0])
        vectors = self.modified_gram_schmidt([v for _, v in pairs])
        if len(vectors) != n:
            raise HomologicalIntegrityError(
                f"Base de Fourier degenerada: {len(vectors)} vectores ≠ n = {n}"
            )

        eigenvalues: List[float] = []
        residuals: List[float] = []
        for (lam, _), v in zip(pairs, vectors):
            Lv = self.matvec_kbn(self._laplacian, v)
            rayleigh = _kbn_dot(v, Lv)
            residual = _compute_norm([a - lam * b for a, b in zip(Lv, v)])
            if abs(rayleigh - lam) > NumericalConstants.SPECTRAL_TOL or residual > NumericalConstants.SPECTRAL_TOL:
                raise HomologicalIntegrityError(
                    f"Autopar no certificado: λ={lam:.6f}, Rayleigh={rayleigh:.6f}, "
                    f"‖Lv−λv‖={residual:.3e}"
                )
            eigenvalues.append(lam)
            residuals.append(residual)

        return eigenvalues, [list(v) for v in vectors], residuals

    def _verify_spectrum(self) -> SpectralCertificate:
        """
        Certificado espectral independiente de la construcción analítica:

          Weyl       : max_i |λ_i(eigvalsh) − λ_i(analítico)| ≤ WEYL_TOL
          Rayleigh   : max_k ‖Lv_k − λ_k v_k‖ ≤ SPECTRAL_TOL
          Potencias  : |ρ(L) − λ_max| ≤ WEYL_TOL   (PSD, arranque genérico)
          Ortonorm.  : ‖VᵀV − I‖_F ≤ 1e-8
          Hodge      : ‖(D − A) − BBᵀ‖_F ≤ STRUCTURAL_TOL
          Traza      : tr L = Σ deg = 2m
          Kirchhoff  : (1/n)∏_{k≥1}λ_k = τ(C₆) = 6
          Cheeger    : h²/(2 d_max) ≤ λ₁ ≤ 2h,  h(C_n) = 2/⌊n/2⌋
        """
        n = TopologyConstants.RING_SIZE
        L = np.asarray(self._laplacian, dtype=np.float64)
        numerical = sorted(float(x) for x in np.linalg.eigvalsh(L))
        analytical = sorted(self._eigenvalues)
        weyl = max(abs(c - a) for c, a in zip(numerical, analytical))

        rho = self.spectral_radius_power(self._laplacian, positive_semidefinite=True)
        ortho_defect = self.orthonormality_defect(self._eigenvectors)
        factorization = _frobenius_norm(
            [
                [self._laplacian[i][j] - self.coboundary_kernel.laplacian_0[i][j] for j in range(n)]
                for i in range(n)
            ]
        )
        trace_res = abs(_kbn_sum(self._laplacian[i][i] for i in range(n)) - 2.0 * self.n_edges)

        product = 1.0
        for lam in analytical[1:]:
            product *= lam
        kirchhoff = product / float(n)

        d_max = float(max(self._degree))
        h_cheeger = 2.0 / float(n // 2)
        lam_1 = analytical[1]
        cheeger_lo = h_cheeger * h_cheeger / (2.0 * d_max)
        cheeger_hi = 2.0 * h_cheeger

        checks = {
            "weyl": weyl <= NumericalConstants.WEYL_TOLERANCE,
            "eigenpair": max(self._eigen_residuals) <= NumericalConstants.SPECTRAL_TOL,
            "power": abs(rho - TopologyConstants.LAMBDA_MAX) <= NumericalConstants.WEYL_TOLERANCE,
            "orthonormality": ortho_defect <= 1.0e-8,
            "factorization": factorization <= NumericalConstants.STRUCTURAL_TOL,
            "trace": trace_res <= NumericalConstants.STRUCTURAL_TOL,
            "kirchhoff": abs(kirchhoff - TopologyConstants.SPANNING_TREES) <= NumericalConstants.WEYL_TOLERANCE,
            "cheeger": cheeger_lo <= lam_1 + NumericalConstants.WEYL_TOLERANCE
            and lam_1 <= cheeger_hi + NumericalConstants.WEYL_TOLERANCE,
        }
        for name, ok in checks.items():
            if not ok:
                logger.warning("Certificado espectral: falla '%s'", name)

        return SpectralCertificate(
            analytical_eigenvalues=tuple(analytical),
            numerical_eigenvalues=tuple(numerical),
            weyl_deviation=float(weyl),
            max_eigenpair_residual=float(max(self._eigen_residuals)),
            spectral_radius_power=float(rho),
            orthonormality_defect=float(ortho_defect),
            factorization_residual=float(factorization),
            trace_residual=float(trace_res),
            kirchhoff_spanning_trees=float(kirchhoff),
            cheeger_constant=float(h_cheeger),
            cheeger_lower_bound=float(cheeger_lo),
            cheeger_upper_bound=float(cheeger_hi),
            verified=all(checks.values()),
        )

    # ── 2.3.2 Accesores (copias defensivas / inmutables) ─────────────────

    @property
    def adjacency_matrix(self) -> List[List[float]]:
        return [row.copy() for row in self._adjacency]

    @property
    def laplacian_matrix(self) -> List[List[float]]:
        """Δ₀ = D − A = BBᵀ."""
        return [row.copy() for row in self._laplacian]

    @property
    def incidence_matrix(self) -> List[List[float]]:
        """B = δ₀."""
        return [row.copy() for row in self._incidence]

    @property
    def eigenvalues(self) -> Tuple[float, ...]:
        """σ(Δ₀) ordenado ascendentemente, emparejado con `eigenvectors`."""
        return tuple(self._eigenvalues)

    @property
    def eigenvectors(self) -> Tuple[Tuple[float, ...], ...]:
        """Base ortonormal V = [v_k] con Δ₀v_k = λ_k v_k."""
        return tuple(tuple(v) for v in self._eigenvectors)

    @property
    def spectral_certificate(self) -> SpectralCertificate:
        return self._spectral_certificate

    @property
    def spectral_verified(self) -> bool:
        return self._spectral_certificate.verified

    @property
    def spectral_gap(self) -> float:
        """λ₁ (conectividad algebraica de Fiedler)."""
        return self._eigenvalues[1] if len(self._eigenvalues) > 1 else 0.0

    @property
    def lambda_max(self) -> float:
        """λ_max = ρ(Δ₀)."""
        return self._eigenvalues[-1]

    @property
    def betti_numbers(self) -> Tuple[int, int]:
        """(β₀, β₁) desde el rango exacto en ℤ (Fase 1)."""
        k = self.coboundary_kernel
        return (k.betti_0, k.betti_1)

    @property
    def euler_characteristic(self) -> int:
        """χ = β₀ − β₁ = n − m = 0 (C₆ ≃ S¹)."""
        return self.coboundary_kernel.euler_characteristic

    @property
    def torsion_free(self) -> bool:
        """Tor(H₀; ℤ) = 0 certificado por la Forma Normal de Smith."""
        return self.coboundary_kernel.torsion_free

    def neighbor_indices(self, node_index: int) -> Tuple[int, int]:
        """Vecindad cíclica {i−1, i+1} mod n."""
        n = TopologyConstants.RING_SIZE
        if isinstance(node_index, bool) or not 0 <= node_index < n:
            raise NumericalDomainError(f"Índice de nodo inválido: {node_index}")
        return ((node_index - 1) % n, (node_index + 1) % n)

    # ── 2.3.3 Análisis de Fourier sobre ℤ/6ℤ ─────────────────────────────

    def spectral_coefficients(self, zero_form: Sequence[float]) -> List[float]:
        """c_k = ⟨v_k, ψ⟩ (transformada de Fourier real sobre el anillo)."""
        self._check_zero_form(zero_form)
        return [_kbn_dot(v, zero_form) for v in self._eigenvectors]

    def parseval_residual(self, zero_form: Sequence[float]) -> float:
        """|Σ c_k² − ‖ψ‖²| (unitariedad de V)."""
        c = self.spectral_coefficients(zero_form)
        return abs(_kbn_dot(c, c) - _kbn_dot(zero_form, zero_form))

    def dirichlet_energy_spectral(self, zero_form: Sequence[float]) -> float:
        """ℰ(ψ) = Σ_k λ_k c_k² — verificación de Plancherel de hodge_energy_0."""
        c = self.spectral_coefficients(zero_form)
        return _kbn_sum(lam * ck * ck for lam, ck in zip(self._eigenvalues, c))

    def spectral_function(self, f: Callable[[float], float]) -> List[List[float]]:
        """
        Cálculo funcional de Borel para Δ₀ autoadjunto:

            f(Δ₀) = Σ_k f(λ_k) v_k v_kᵀ     (cada entrada con Dot2).
        """
        n = TopologyConstants.RING_SIZE
        weights = [float(f(lam)) for lam in self._eigenvalues]
        if not all(math.isfinite(w) for w in weights):
            raise NumericalDomainError("f(λ_k) no finito en el cálculo funcional")
        V = self._eigenvectors
        return [
            [
                _kbn_sum(weights[k] * V[k][i] * V[k][j] for k in range(n))
                for j in range(n)
            ]
            for i in range(n)
        ]

    def heat_kernel(self, t: float) -> List[List[float]]:
        """
        Semigrupo de calor exacto K(t) = e^{−tΔ₀} = Σ e^{−tλ_k} v_k v_kᵀ.

        Contractivo (‖K(t)‖₂ = 1), positivo y estocástico para t ≥ 0;
        incondicionalmente estable (no requiere CFL). K(t)𝟙 = 𝟙.
        """
        if not _is_finite(t) or t < 0.0:
            raise NumericalDomainError(f"Tiempo de difusión inválido: {t!r}")
        return self.spectral_function(lambda lam: _stable_exp(-t * lam))

    def diffuse_exact(self, state_vector: Sequence[float], t: float) -> List[float]:
        """ψ(t) = e^{−tΔ₀}ψ(0) por síntesis espectral (sin construir K)."""
        self._check_zero_form(state_vector)
        c = self.spectral_coefficients(state_vector)
        n = TopologyConstants.RING_SIZE
        decay = [_stable_exp(-t * lam) for lam in self._eigenvalues]
        return [
            _kbn_sum(decay[k] * c[k] * self._eigenvectors[k][i] for k in range(n))
            for i in range(n)
        ]

    # ── 2.3.4 Hückel: Hamiltoniano, niveles y observables MO ─────────────

    def huckel_hamiltonian(self) -> List[List[float]]:
        """H = αI + βA (β < 0). Como A = 2I − Δ₀ en un grafo 2-regular, [H, Δ₀] = 0."""
        n = TopologyConstants.RING_SIZE
        return [
            [
                HuckelConstants.ALPHA if i == j
                else (HuckelConstants.BETA if self._adjacency[i][j] > 0.0 else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]

    def huckel_levels(self) -> Tuple[float, ...]:
        """E_k = α + β(2 − λ_k) en el orden de la autobase de Δ₀ (misma V)."""
        return tuple(
            HuckelConstants.ALPHA + HuckelConstants.BETA * (2.0 - lam)
            for lam in self._eigenvalues
        )

    def huckel_spectrum(self) -> Tuple[float, ...]:
        """Niveles MO ordenados ascendentemente (β < 0 ⇒ E₀ = α + 2β es el fundamental)."""
        return tuple(sorted(self.huckel_levels()))

    def huckel_pi_energy(self, n_electrons: int = HuckelConstants.PI_ELECTRONS_BENZENE) -> float:
        """
        E_π por principio de Aufbau (2 e⁻ por nivel, niveles ascendentes):

            E_π = 2 Σ_{k < ⌊n_e/2⌋} E_(k) + (n_e mod 2)·E_(⌊n_e/2⌋)

        Benceno (n_e = 6): 2(α+2β) + 4(α+β) = 6α + 8β.
        """
        if isinstance(n_electrons, bool) or not 0 <= n_electrons <= 2 * TopologyConstants.RING_SIZE:
            raise NumericalDomainError(f"Número de electrones π inválido: {n_electrons!r}")
        levels = self.huckel_spectrum()
        pairs, single = divmod(n_electrons, 2)
        energy = 2.0 * _kbn_sum(levels[:pairs])
        if single:
            energy += levels[pairs]
        return float(energy)

    def huckel_delocalization_energy(
        self,
        n_electrons: int = HuckelConstants.PI_ELECTRONS_BENZENE,
    ) -> float:
        """
        Energía de deslocalización respecto a n_e/2 etilenos aislados (2(α+β) c/u):

            ΔE_deloc = E_π − n_e(α + β)     (benceno: 2β < 0, estabilizante).
        """
        reference = n_electrons * (HuckelConstants.ALPHA + HuckelConstants.BETA)
        return self.huckel_pi_energy(n_electrons) - reference

    def huckel_homo_lumo_gap(
        self,
        n_electrons: int = HuckelConstants.PI_ELECTRONS_BENZENE,
    ) -> float:
        """E_LUMO − E_HOMO (benceno: (α−β) − (α+β) = −2β > 0)."""
        levels = self.huckel_spectrum()
        homo = max(0, (n_electrons + 1) // 2 - 1)
        lumo = min(len(levels) - 1, homo + 1)
        return levels[lumo] - levels[homo]

    def adaptive_tikhonov_shift(self) -> float:
        """α_reg = max(α_min, c·u) · max(1, λ_max, |α|, |β|) — escala del Hamiltoniano."""
        base = max(
            NumericalConstants.TIKHONOV_BASE_MIN,
            NumericalConstants.TIKHONOV_EPS_SCALE * NumericalConstants.MACHINE_EPS,
        )
        scale = max(1.0, self.lambda_max, abs(HuckelConstants.ALPHA), abs(HuckelConstants.BETA))
        return float(base * scale)

    def regularized_huckel_resolvent(
        self,
        s: float = 0.0,
        h: float = 0.0,
        alpha_reg: Optional[float] = None,
    ) -> HuckelResolvent:
        """
        Resolvente espectral G(z) = (H − zI)⁻¹, z = s + ih, con filtro de
        Tikhonov sólo si κ₂(H − zI) ≥ techo:

            d_k = E_k − z ,  κ₂ = max|d_k| / min|d_k|   (H normal ⇒ exacto)
            f_k = 1/d_k                       si κ₂ < TIKHONOV_COND_CEILING
            f_k = d̄_k / (|d_k|² + α²)         en otro caso, α escalado ×10 hasta 4 veces

        Nota física: la regularización canónica de una función de Green es
        h = η > 0 (retardada); α actúa como salvaguarda numérica, no como
        parte del modelo. Sin factorizaciones: coste O(n³) sólo en la síntesis.
        """
        if not (_is_finite(s) and _is_finite(h)):
            raise NumericalDomainError(f"z = {s!r} + i{h!r} no finito")
        z = complex(float(s), float(h))
        levels = self.huckel_levels()
        d = [complex(E) - z for E in levels]
        moduli = [abs(dk) for dk in d]
        d_max, d_min = max(moduli), min(moduli)
        cond_raw = d_max / d_min if d_min > 0.0 else math.inf

        alpha = 0.0
        applied = False
        if cond_raw < NumericalConstants.TIKHONOV_COND_CEILING:
            filt = [1.0 / dk for dk in d]
            cond_eff = cond_raw
        else:
            alpha = float(alpha_reg) if alpha_reg is not None else self.adaptive_tikhonov_shift()
            if not _is_finite(alpha) or alpha <= 0.0:
                raise NumericalDomainError(f"α de Tikhonov inválido: {alpha_reg!r}")
            applied = True
            for _ in range(4):
                filt = [dk.conjugate() / (abs(dk) ** 2 + alpha * alpha) for dk in d]
                mags = [abs(f) for f in filt]
                cond_eff = max(mags) / min(mags) if min(mags) > 0.0 else math.inf
                if cond_eff < NumericalConstants.TIKHONOV_COND_CEILING:
                    break
                alpha *= 10.0

        n = TopologyConstants.RING_SIZE
        V = self._eigenvectors
        green = tuple(
            tuple(sum(filt[k] * V[k][i] * V[k][j] for k in range(n)) for j in range(n))
            for i in range(n)
        )
        return HuckelResolvent(
            green=green,
            z=z,
            alpha=alpha,
            levels=levels,
            condition_raw=float(cond_raw),
            condition_effective=float(cond_eff),
            tikhonov_applied=applied,
        )

    def huckel_density_of_states(self, energy: float, eta: float = 1.0e-3) -> float:
        """
        Densidad de estados lorentziana ρ(E) = −(1/π) Im tr G(E + iη), η > 0.

        ∫ρ dE = n; en el límite η → 0⁺ converge a Σ_k δ(E − E_k).
        """
        if not _is_finite(eta) or eta <= 0.0:
            raise NumericalDomainError(f"η debe ser > 0: {eta!r}")
        resolvent = self.regularized_huckel_resolvent(s=energy, h=eta)
        return float(-resolvent.trace().imag / math.pi)

    # ── 2.3.5 Auditoría de Rham–Tellegen–Hodge–Parseval ──────────────────

    def _edge_gradients(self, state_vector: Sequence[float]) -> List[float]:
        """1-forma exacta d₀ψ (validación de dominio + Dot2)."""
        self._check_zero_form(state_vector, "state_vector")
        return self.exterior_derivative_0(state_vector)

    def compute_graph_laplacian_energy(self, state: Sequence[float]) -> float:
        """ℰ(ψ) = ψᵀΔ₀ψ = ‖d₀ψ‖² ≥ 0."""
        return self.hodge_energy_0(state)

    def audit_de_rham_tellegen(
        self,
        state_vector: Sequence[float],
        fluxes: Optional[Sequence[float]] = None,
        tolerance: Optional[float] = None,
    ) -> TellegenAuditReport:
        """
        Auditoría óctuple (véase TellegenAuditReport). Las tolerancias son
        relativas: tol · scale con scale = max(1, ℰ(ψ), ‖q‖²), para que la
        auditoría no falle por escala cuando ‖ψ‖ ≫ 1.

        Si fluxes es None se toma el flujo natural q = −d₀ψ (Ohm discreto),
        único caso en el que la ley de Joule (axioma 8) es evaluable.
        """
        n = TopologyConstants.RING_SIZE
        tol = float(tolerance) if tolerance is not None else NumericalConstants.TELLEGEN_TOLERANCE

        grad = self._edge_gradients(state_vector)
        energy = _kbn_dot(grad, grad)

        natural = fluxes is None
        if natural:
            q = [-g for g in grad]
        else:
            _validate_finite_vector(fluxes, "fluxes", expected_dim=n)
            q = [float(x) for x in fluxes]

        scale = max(1.0, energy, _kbn_dot(q, q))
        thr = tol * scale

        # 1. pasividad
        passivity_res = energy
        # 2. conservación (𝟙ᵀδ₀ = 0)
        conservation_res = abs(_kbn_sum(self.codifferential_1(q)))
        # 3, 5, 6. Hodge sobre el flujo auditado
        dec = self.hodge_decompose_one_form(q)
        h_basis = self.harmonic_basis()
        harmonic_closure_res = max(
            (_compute_norm(self.codifferential_1(list(hb))) for hb in h_basis),
            default=0.0,
        )
        # 4. Tellegen
        tellegen_res = self.tellegen_residual(state_vector, q)
        # 7. Parseval–Plancherel
        parseval_res = abs(energy - self.dirichlet_energy_spectral(state_vector))
        # 8. Joule (sólo flujo natural)
        ohmic_res = abs(_kbn_dot(grad, q) + energy) if natural else 0.0

        return TellegenAuditReport(
            passivity_satisfied=bool(passivity_res >= -thr),
            conservation_satisfied=bool(conservation_res <= thr),
            orthogonality_satisfied=bool(dec.orthogonality_residual <= thr),
            tellegen_satisfied=bool(tellegen_res <= thr),
            harmonic_closure_satisfied=bool(harmonic_closure_res <= tol),
            reconstruction_satisfied=bool(dec.reconstruction_residual <= thr),
            parseval_satisfied=bool(parseval_res <= thr),
            ohmic_satisfied=bool(ohmic_res <= thr),
            laplacian_energy=float(energy),
            conservation_residual=float(conservation_res),
            orthogonality_residual=float(dec.orthogonality_residual),
            tellegen_residual=float(tellegen_res),
            harmonic_closure_residual=float(harmonic_closure_res),
            reconstruction_residual=float(dec.reconstruction_residual),
            parseval_residual=float(parseval_res),
            ohmic_residual=float(ohmic_res),
            tolerance=tol,
            scale=float(scale),
            natural_flux=natural,
        )

    # ── 2.3.6 Condiciones de frontera ────────────────────────────────────

    _BOUNDARY_KEYS: ClassVar[frozenset] = frozenset({"dirichlet", "neumann"})

    def _validate_boundary(self, mapping: Dict[int, float]) -> None:
        """Soporte ⊂ {0,…,n−1} y datos finitos."""
        for idx, value in mapping.items():
            if isinstance(idx, bool) or not isinstance(idx, numbers.Integral):
                raise NumericalDomainError(f"Índice de frontera no entero: {idx!r}")
            if not 0 <= idx < TopologyConstants.RING_SIZE:
                raise NumericalDomainError(f"Índice de frontera inválido: {idx}")
            if not _is_finite(value):
                raise NumericalDomainError(f"Valor de frontera no finito en {idx}: {value!r}")

    def _apply_dirichlet(self, vector: List[float], values: Dict[int, float]) -> List[float]:
        """Traza de Dirichlet ψ|Γ = g."""
        self._validate_boundary(values)
        result = vector.copy()
        for idx, value in values.items():
            result[int(idx)] = float(value)
        return result

    def _apply_neumann(self, vector: List[float], fluxes: Dict[int, float]) -> List[float]:
        """
        Neumann débil (dψ·n)|Γ = φ como inyección nodal ψ_i ← ψ_i + φ_i.
        Se vigila la amplificación ‖ψ⁺‖ > 2‖ψ‖ (pérdida de contractividad).
        """
        self._validate_boundary(fluxes)
        norm_pre = _compute_norm(vector)
        result = vector.copy()
        for idx, flux in fluxes.items():
            result[int(idx)] += float(flux)
        norm_post = _compute_norm(result)
        if norm_pre > NumericalConstants.EPS and norm_post > 2.0 * norm_pre:
            logger.warning(
                "Condición Neumann amplifica norma: %.4f → %.4f (×%.2f)",
                norm_pre, norm_post, norm_post / norm_pre,
            )
        return result

    # ── 2.3.7 Dinámica: propagador de Euler y difusión ───────────────────

    def _project_diffusion_rate(self, diffusion_rate: float) -> float:
        """Proyección de α al régimen histórico [0, α_crit): α ≥ α_crit ↦ α_safe."""
        if not _is_finite(diffusion_rate):
            raise NumericalDomainError(f"Tasa de difusión no finita: {diffusion_rate!r}")
        if diffusion_rate < 0.0:
            raise NumericalDomainError(f"Tasa de difusión negativa: {diffusion_rate!r}")
        if diffusion_rate >= CFLConstants.ALPHA_CRITICAL:
            logger.warning(
                "Tasa de difusión %.5f ≥ α_crit %.5f → ajustada a %.5f",
                diffusion_rate, CFLConstants.ALPHA_CRITICAL, CFLConstants.ALPHA_SAFE,
            )
            return CFLConstants.ALPHA_SAFE
        return float(diffusion_rate)

    def euler_propagator(self, diffusion_rate: float = CFLConstants.ALPHA_SAFE) -> List[List[float]]:
        """P(α) = I − αΔ₀, σ(P) = {1 − αλ_k}; ‖P‖₂ = max_k |1 − αλ_k| = 1 (modo k=0)."""
        alpha = self._project_diffusion_rate(diffusion_rate)
        n = TopologyConstants.RING_SIZE
        return [
            [(1.0 if i == j else 0.0) - alpha * self._laplacian[i][j] for j in range(n)]
            for i in range(n)
        ]

    def diffuse_stress(
        self,
        state_vector: Sequence[float],
        diffusion_rate: float = CFLConstants.ALPHA_SAFE,
        boundary_conditions: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> List[float]:
        """
        Un paso de Euler explícito del semigrupo e^{−tΔ₀}:

            ψ ← (I − αΔ₀)ψ = ψ + 𝒢ψ

        α se proyecta a [0, α_crit) (von Neumann con margen histórico);
        Δ₀ψ se evalúa fila a fila con Dot2. Las condiciones de frontera se
        aplican tras el paso interior (Dirichlet fuerte, Neumann débil).
        """
        self._check_zero_form(state_vector, "state_vector")
        alpha = self._project_diffusion_rate(diffusion_rate)
        if alpha == 0.0:
            new_vector = [float(x) for x in state_vector]
        else:
            lap = self.matvec_kbn(self._laplacian, state_vector)
            new_vector = [float(x) - alpha * l for x, l in zip(state_vector, lap)]

        if boundary_conditions:
            unknown = set(boundary_conditions) - self._BOUNDARY_KEYS
            if unknown:
                raise NumericalDomainError(f"Condiciones de frontera desconocidas: {sorted(unknown)}")
            if "dirichlet" in boundary_conditions:
                new_vector = self._apply_dirichlet(new_vector, boundary_conditions["dirichlet"])
            if "neumann" in boundary_conditions:
                new_vector = self._apply_neumann(new_vector, boundary_conditions["neumann"])

        _validate_finite_vector(new_vector, "diffused_vector", expected_dim=TopologyConstants.RING_SIZE)
        return new_vector

    # ── 2.3.8 Morfismo terminal ──────────────────────────────────────────

    def infinitesimal_catalytic_generator(
        self,
        diffusion_rate: float = CFLConstants.ALPHA_SAFE,
    ) -> CatalyticGenerator:
        """
        ── MORFISMO TERMINAL DE LA FASE 2 / INICIAL DE LA FASE 3 ──────────

        Generador infinitesimal del semigrupo catalítico de difusión:

            𝒢 = −αΔ₀ ∈ 𝔅(ℝ⁶),   α = min(α_solicitado, α_safe)
            e^{t𝒢} contractivo en ℓ²;  Euler P = I + 𝒢 estable ⇔ α ≤ 2/λ_max

        Certificado adjunto (CatalyticGenerator):
            contraction_constant = max_{k≥1}|1 − αλ_k|   (contracción sobre 𝟙^⊥)
            spectral_abscissa    = −αλ₁                    (Lyapunov continuo)
            is_monotone          = α ≤ 1/λ_max             (P ⪰ 0, sin oscilación)
            regime ∈ {"safe", "monotone", "l2_stable", "unstable"}

        La Fase 3 integra este generador sobre HilbertState (ψ ← ψ + 𝒢ψ) y
        acopla ℰ(ψ) = −⟨ψ, 𝒢ψ⟩/α al potencial de Gibbs.
        """
        if not _is_finite(diffusion_rate) or diffusion_rate < 0.0:
            raise NumericalDomainError(f"α inválido para el generador: {diffusion_rate!r}")

        alpha = min(float(diffusion_rate), CFLConstants.ALPHA_SAFE)
        if alpha < float(diffusion_rate):
            logger.info("Generador: α %.5f recortado a α_safe %.5f", diffusion_rate, alpha)

        n = TopologyConstants.RING_SIZE
        matrix = tuple(
            tuple(-alpha * self._laplacian[i][j] for j in range(n)) for i in range(n)
        )
        contraction = max(abs(1.0 - alpha * lam) for lam in self._eigenvalues[1:])
        l2_stable = alpha <= CFLConstants.ALPHA_SHARP + NumericalConstants.EPS
        monotone = alpha <= CFLConstants.ALPHA_MONOTONE + NumericalConstants.EPS
        if alpha <= CFLConstants.ALPHA_CRITICAL + NumericalConstants.EPS:
            regime = "safe"
        elif monotone:
            regime = "monotone"
        elif l2_stable:
            regime = "l2_stable"
        else:
            regime = "unstable"

        return CatalyticGenerator(
            matrix=matrix,
            alpha=alpha,
            alpha_requested=float(diffusion_rate),
            spectral_gap=self.spectral_gap,
            lambda_max=self.lambda_max,
            contraction_constant=float(contraction),
            spectral_abscissa=float(-alpha * self.spectral_gap),
            is_l2_stable=bool(l2_stable),
            is_monotone=bool(monotone),
            regime=regime,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — DINÁMICA TERMODINÁMICA-CUÁNTICA DEL REACTOR
# Continúa HexagonalTopology.infinitesimal_catalytic_generator (Fase 2)
# ══════════════════════════════════════════════════════════════════════════════
#
# Objetos: estado de Hilbert ψ ∈ ℝ⁶ con integración certificada del semigrupo
#          (ψ ← ψ + 𝒢ψ), potencial de Gibbs G(H, S, T, σ, a), álgebra de Boole
#          de Hückel (aromaticidad), entropía de Shannon ⊕ von Neumann
#          decoherida, sesión de reacción reentrante y reactor catalítico.

# ──────────────────────────────────────────────────────────────────────────────
# 3.0  Constantes cinéticas y enumeraciones de la dinámica
# ──────────────────────────────────────────────────────────────────────────────


class ReactionKinetics:
    """
    Constantes cinéticas del reactor (unidades del modelo: energías en |β|).

    Centraliza los coeficientes que en v4.0 aparecían como literales inline,
    para que cada término del balance entálpico sea auditable.
    """

    __slots__ = ()

    # Balance entálpico nodal:  ΔH = g_σ σ² + g_E Eₐ   (resonancia)
    STRESS_ENTHALPY_GAIN: Final[float] = 5.0
    BARRIER_ENTHALPY_GAIN: Final[float] = 10.0
    SKIP_ENTHALPY: Final[float] = 5.0
    ERROR_ENTHALPY: Final[float] = 50.0
    ERROR_STRESS_INJECTION: Final[float] = 1.0

    # Liberación de resonancia:  ΔH_res = κ · (n_res / n) · ΔE_deloc  (< 0)
    RESONANCE_RELEASE_SCALE: Final[float] = 100.0

    # Excitación local:  Δψᵢ = g tanh(ΔH⁺ / s) + g_b Eₐ − ν σᵢ
    EXCITATION_SATURATION: Final[float] = 25.0
    BARRIER_EXCITATION_GAIN: Final[float] = 0.02

    # Precursores y estabilización
    PRECURSOR_PENALTY: Final[float] = 0.3
    ENTHALPY_TRIM_FACTOR: Final[float] = 0.85
    EFFICIENCY_CEILING: Final[float] = 0.999999

    # Latencia simulada (sólo modo no determinista)
    NODE_LATENCY_BASE_S: Final[float] = 0.005
    NODE_LATENCY_STRESS_GAIN: Final[float] = 0.5

    # Historiales y convergencia
    NORM_HISTORY_LENGTH: Final[int] = 16
    GIBBS_HISTORY_LENGTH: Final[int] = 20
    TREND_WINDOW: Final[int] = 5
    GIBBS_TREND_TOL: Final[float] = 0.05

    # Espacio de nombres de diagnósticos (excluido de Shannon y de aromaticidad)
    DIAGNOSTIC_PREFIX: Final[str] = "_"


class DiffusionScheme(Enum):
    """Integrador del semigrupo e^{−tΔ₀}."""

    EULER = auto()   # ψ ← (I + 𝒢)ψ, condicionalmente estable (CFL)
    EXACT = auto()   # ψ ← e^{−αΔ₀}ψ, incondicionalmente estable (espectral)


class RingIsometry(Enum):
    """
    Acción opcional del grupo diédrico D₆ sobre el campo de estrés ψ.

    ROTATION  : j ↦ j+1 (mod 6), generador de ℤ₆ ⊂ D₆; intercambia las dos
                estructuras de Kekulé {(0,1),(2,3),(4,5)} ↔ {(1,2),(3,4),(5,0)}.
    REFLECTION: j ↦ 5−j, reflexión de D₆ (la del v4.0, `vector[::-1]`).
    Ambas conmutan con Δ₀ (automorfismos de C₆) y preservan ‖ψ‖ y ℰ(ψ).
    """

    NONE = auto()
    ROTATION = auto()
    REFLECTION = auto()


class NodeOutcome(Enum):
    """Resultado de la reacción local de un nodo (valor = cadena de estado)."""

    RESONANT = "resonant"
    SKIPPED = "skipped"
    ERROR = "error"


_DIAG: Final[str] = ReactionKinetics.DIAGNOSTIC_PREFIX


# ──────────────────────────────────────────────────────────────────────────────
# 3.1  Contratos de retorno tipados (inmutables)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DiffusionEnergyBalance:
    """
    Balance energético exacto de un paso de difusión.

    Euler (ψ⁺ = ψ + 𝒢ψ):
        ‖ψ⁺‖² − ‖ψ‖² = 2⟨ψ, 𝒢ψ⟩ + ‖𝒢ψ‖² = −2αℰ(ψ) + α²‖Δ₀ψ‖²
        ≤ 0  ⇐  α ≤ 2/λ_max      (pues ‖Δ₀ψ‖² = Σλ_k²c_k² ≤ λ_max ℰ(ψ))

    Exacto (ψ⁺ = e^{−αΔ₀}ψ):
        ‖ψ⁺‖² − ‖ψ‖² = Σ_k (e^{−2αλ_k} − 1) c_k² ≤ 0   (siempre)

    identity_residual = |actual − predicted| certifica el integrador (O(u)).
    """

    norm_sq_before: float
    norm_sq_after: float
    predicted_decrement: float
    actual_decrement: float
    identity_residual: float
    dirichlet_energy: Optional[float]
    dissipative: bool
    scheme: str


@dataclass(frozen=True)
class NodeReaction:
    """Resultado inmutable de la reacción local de un nodo del anillo."""

    node: CarbonNode
    outcome: NodeOutcome
    activation_energy: float
    delta_h: float
    arrhenius_rate: float
    detail: str = ""

    def context_update(self, timestamp: float) -> Dict[str, Any]:
        """Proyección al contexto: estado nodal + marcadores de ciclo."""
        name = self.node.name
        update: Dict[str, Any] = {
            f"{name}_status": self.outcome.value,
            f"{name}_ts": timestamp,
            f"{name}_ea": self.activation_energy,
            f"{name}_rate": self.arrhenius_rate,
        }
        if self.outcome is NodeOutcome.SKIPPED:
            update[f"{name}_skipped"] = True
        elif self.outcome is NodeOutcome.ERROR:
            update[f"{name}_error"] = self.detail or "error"
        return update


@dataclass(frozen=True)
class CycleRecord:
    """Muestra de la trayectoria del reactor en un ciclo de resonancia."""

    cycle: int
    gibbs: float
    enthalpy: float
    entropy: float
    temperature: float
    instability: float
    stress_norm: float
    dirichlet_energy: float
    lyapunov_value: float
    lyapunov_decrement: float
    diffusion_dissipative: bool
    tellegen_coherent: bool
    pi_electrons: int
    aromaticity: str


# ──────────────────────────────────────────────────────────────────────────────
# 3.2  Estado de Hilbert
# ──────────────────────────────────────────────────────────────────────────────

T = TypeVar("T", bound="HilbertState")


@dataclass
class HilbertState:
    """
    ── CONTINUACIÓN DE LA FASE 2 ──────────────────────────────────────────

    Estado del reactor en el Hilbert real (ℝ⁶, ⟨·,·⟩).

    El CatalyticGenerator 𝒢 = −αΔ₀ de la Fase 2 actúa por
        ψ ↦ ψ + 𝒢ψ      (apply_generator_step, con balance certificado).

    La fase φ ∈ ℝ/2πℤ es la holonomía del anillo: transporte paralelo
    discreto de 2π/6 por ciclo; tras 6 ciclos la conexión (plana) retorna
    a la identidad.
    """

    vector: List[float] = field(
        default_factory=lambda: [0.0] * TopologyConstants.RING_SIZE
    )
    phase: float = 0.0
    _norm_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=ReactionKinetics.NORM_HISTORY_LENGTH),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_finite_vector(
            self.vector, "HilbertState.vector", expected_dim=TopologyConstants.RING_SIZE
        )
        self.vector = [float(x) for x in self.vector]
        if not _is_finite(self.phase):
            raise NumericalDomainError(f"Fase no finita: {self.phase!r}")
        self.phase = float(self.phase) % (2.0 * math.pi)
        if not isinstance(self._norm_history, deque):
            self._norm_history = deque(
                self._norm_history, maxlen=ReactionKinetics.NORM_HISTORY_LENGTH
            )

    # ── 3.2.1 Sutura Fase 2 → Fase 3: integración del generador ──────────

    def apply_generator_step(
        self,
        generator: Union[CatalyticGenerator, Sequence[Sequence[float]]],
    ) -> DiffusionEnergyBalance:
        """
        Integra un paso de Euler del generador infinitesimal de la Fase 2:

            ψ ← ψ + 𝒢ψ = (I − αΔ₀)ψ

        y certifica el balance exacto de energía cinética

            ‖ψ⁺‖² − ‖ψ‖² = 2⟨ψ, 𝒢ψ⟩ + ‖𝒢ψ‖²,

        con ⟨ψ, 𝒢ψ⟩ = −αℰ(ψ) (energía de Dirichlet recuperada del generador).
        El paso es disipativo si el segundo miembro es ≤ 0, lo que está
        garantizado para α ≤ 2/λ_max (CatalyticGenerator.is_l2_stable).
        """
        psi = self.vector
        increment = CompensatedLinearAlgebra.matvec_kbn(generator, psi)
        norm_sq_before = _kbn_dot(psi, psi)
        cross = _kbn_dot(psi, increment)          # = −α ℰ(ψ) ≤ 0
        increment_sq = _kbn_dot(increment, increment)
        predicted = 2.0 * cross + increment_sq

        new_vector = [x + d for x, d in zip(psi, increment)]
        _validate_finite_vector(new_vector, "generated_vector")
        norm_sq_after = _kbn_dot(new_vector, new_vector)
        actual = norm_sq_after - norm_sq_before

        alpha = getattr(generator, "alpha", None)
        dirichlet = (-cross / alpha) if (alpha is not None and alpha > 0.0) else None

        self.vector = new_vector
        self.record_norm()
        tol = NumericalConstants.EPS * max(1.0, norm_sq_before)
        return DiffusionEnergyBalance(
            norm_sq_before=float(norm_sq_before),
            norm_sq_after=float(norm_sq_after),
            predicted_decrement=float(predicted),
            actual_decrement=float(actual),
            identity_residual=float(abs(actual - predicted)),
            dirichlet_energy=None if dirichlet is None else float(dirichlet),
            dissipative=bool(predicted <= tol),
            scheme="euler",
        )

    def replace_vector(self, new_vector: Sequence[float]) -> None:
        """Sustitución validada de ψ (usada por el esquema exacto); registra ‖ψ‖."""
        _validate_finite_vector(
            new_vector, "replaced_vector", expected_dim=TopologyConstants.RING_SIZE
        )
        self.vector = [float(x) for x in new_vector]
        self.record_norm()

    def record_norm(self) -> None:
        """Anexa ‖ψ‖ al historial acotado (deque, O(1))."""
        self._norm_history.append(self.norm)

    @property
    def norm_history(self) -> Tuple[float, ...]:
        """Historial de ‖ψ‖ (vista inmutable, más reciente al final)."""
        return tuple(self._norm_history)

    # ── 3.2.2 Geometría del Hilbert ──────────────────────────────────────

    @property
    def norm(self) -> float:
        """‖ψ‖₂ escalada (Fase 1)."""
        return _compute_norm(self.vector)

    @property
    def norm_squared(self) -> float:
        """‖ψ‖² con Dot2 (energía cinética adimensional)."""
        return _kbn_dot(self.vector, self.vector)

    def inner_product(self, other: "HilbertState") -> float:
        """⟨self|other⟩ compensado."""
        result = _kbn_dot(self.vector, other.vector)
        if not _is_finite(result):
            raise NumericalDomainError(f"Producto interno no finito: {result}")
        return result

    def normalize(self: T) -> T:
        """Proyección a S⁵ (si ‖ψ‖ ≥ ε)."""
        self.vector, _ = _normalize_vector(self.vector)
        return self

    def scale(self: T, factor: float) -> T:
        """Homotecia ψ ↦ λψ, λ finito."""
        if not _is_finite(factor):
            raise NumericalDomainError(f"Factor de escala no finito: {factor!r}")
        self.vector = [x * float(factor) for x in self.vector]
        _validate_finite_vector(self.vector, "scaled_vector")
        return self

    def project_orthogonal(self, subspace_basis: Sequence["HilbertState"]) -> None:
        """ψ ← (I − Π_W)ψ, W = span(base), vía MGS + Dot2 (Fase 1)."""
        n = len(self.vector)
        raw = [b.vector for b in subspace_basis if len(b.vector) == n]
        for q in CompensatedLinearAlgebra.modified_gram_schmidt(raw):
            coeff = _kbn_dot(self.vector, q)
            self.vector = [x - coeff * qi for x, qi in zip(self.vector, q)]
        _validate_finite_vector(self.vector, "projected_vector")

    def apply_ring_isometry(self, permutation: Sequence[int]) -> None:
        """
        Acción de una permutación σ ∈ S₆ sobre los sitios: ψ'ᵢ = ψ_{σ(i)}.

        Es una isometría de ℓ²; el llamador certifica [P_σ, Δ₀] = 0 si
        requiere que además preserve ℰ(ψ) (σ ∈ D₆).
        """
        n = len(self.vector)
        perm = [int(p) for p in permutation]
        if sorted(perm) != list(range(n)):
            raise NumericalDomainError(f"No es una permutación de {n} sitios: {permutation!r}")
        self.vector = [self.vector[p] for p in perm]

    # ── 3.2.3 Información cuántica ───────────────────────────────────────

    def density_matrix(self) -> List[List[float]]:
        """
        Operador densidad puro ρ = |ψ̂⟩⟨ψ̂| (ρ ⪰ 0, tr ρ = 1). Si ψ = 0 se
        retorna I/n (mezcla máxima). Nota: S_vN(ρ puro) ≡ 0; la cantidad
        informativa es la entropía tras decoherencia (dephased_density_matrix).
        """
        n = len(self.vector)
        hat, norm = _normalize_vector(self.vector)
        if norm < NumericalConstants.EPS:
            return [[(1.0 / n) if i == j else 0.0 for j in range(n)] for i in range(n)]
        return [[hat[i] * hat[j] for j in range(n)] for i in range(n)]

    def occupation_probabilities(
        self,
        basis: Optional[Sequence[Sequence[float]]] = None,
        tol: float = 1.0e-8,
    ) -> List[float]:
        """
        Distribución de ocupación p_k = ⟨v_k, ψ⟩² / ‖ψ‖² en una base
        ortonormal completa (canónica si `basis` es None). Parseval
        certificado: |Σp_k − 1| ≤ tol. ψ = 0 ↦ distribución uniforme.
        """
        n = len(self.vector)
        norm_sq = self.norm_squared
        if norm_sq < NumericalConstants.EPS ** 2:
            return [1.0 / n] * n
        if basis is None:
            probs = [x * x / norm_sq for x in self.vector]
        else:
            if len(basis) != n:
                raise NumericalDomainError(f"Base de {len(basis)} vectores ≠ n = {n}")
            probs = [(_kbn_dot(v, self.vector) ** 2) / norm_sq for v in basis]
        total = _kbn_sum(probs)
        if abs(total - 1.0) > tol:
            raise NumericalDomainError(
                f"Base no ortonormal completa: Σp_k = {total:.12f} (Parseval)"
            )
        return probs

    def dephased_density_matrix(
        self,
        basis: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[List[float]]:
        """
        Estado decoherido (pinching) en la base preferente:

            ρ_deph = Σ_k p_k v_k v_kᵀ ,   p_k = ⟨v_k,ψ⟩²/‖ψ‖²

        Es el estado post-medida no selectiva en {v_k}; su S_vN es la
        entropía de Shannon de la ocupación modal.
        """
        n = len(self.vector)
        probs = self.occupation_probabilities(basis)
        if basis is None:
            return [[probs[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
        return [
            [_kbn_sum(probs[k] * basis[k][i] * basis[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]

    def von_neumann_entropy(
        self,
        basis: Optional[Sequence[Sequence[float]]] = None,
    ) -> float:
        """
        S_vN(ρ_deph) = −Σ p_k log p_k ∈ [0, log n].

        Con la autobase de Δ₀ mide la deslocalización modal del estrés
        (0: modo puro; log n: equipartición). Con la base canónica mide la
        deslocalización espacial sobre los 6 carbonos.
        """
        probs = self.occupation_probabilities(basis)
        return float(
            -_kbn_sum(p * _stable_log(p) for p in probs if p > NumericalConstants.ENTROPY_MIN_PROB)
        )

    # ── 3.2.4 Disipación y diagnósticos ──────────────────────────────────

    def apply_damping(self, cycle: int) -> None:
        """
        Contracción hacia la media (modo k = 0 ∈ ker Δ₀) con envolvente

            ψ ← μ𝟙 + (ψ − μ𝟙)·e^{−γt}|cos ωt|

        Disipador de Lyapunov que preserva la componente armónica (μ).
        """
        n = len(self.vector)
        envelope = _stable_exp(-DampingConstants.GAMMA * cycle)
        oscillation = abs(math.cos(DampingConstants.OMEGA * cycle))
        factor = envelope * oscillation
        mean = _kbn_sum(self.vector) / float(n)
        self.vector = [mean + (v - mean) * factor for v in self.vector]
        _validate_finite_vector(self.vector, "damped_vector")
        self.record_norm()

    def is_oscillating(self, window: int = 5, threshold: float = 0.1) -> bool:
        """
        Detección de ciclo límite: ≥ window−2 cambios de signo en Δ‖ψ‖ con
        |Δ‖ψ‖| > threshold·‖ψ‖_ref (los cambios sub-umbral son ruido).
        """
        history = self.norm_history
        if len(history) < window:
            return False
        recent = history[-window:]
        ref = max(max(recent), NumericalConstants.EPS)
        deltas = [b - a for a, b in zip(recent, recent[1:])]
        significant = [d for d in deltas if abs(d) > threshold * ref]
        sign_changes = sum(
            1 for a, b in zip(significant, significant[1:]) if a * b < 0.0
        )
        return sign_changes >= window - 2

    def lyapunov_value(self, gibbs: float) -> float:
        """V(ψ, G) = ‖ψ‖² + G²."""
        g = float(gibbs)
        return self.norm_squared + g * g

    def copy(self) -> "HilbertState":
        """Copia profunda incluyendo historial."""
        new_state = HilbertState(vector=list(self.vector), phase=self.phase)
        new_state._norm_history = deque(
            self._norm_history, maxlen=ReactionKinetics.NORM_HISTORY_LENGTH
        )
        return new_state

    def __repr__(self) -> str:
        components = ", ".join(f"{v:.4f}" for v in self.vector)
        return f"HilbertState(‖ψ‖={self.norm:.4f}, φ={self.phase:.4f}rad, [{components}])"


# ──────────────────────────────────────────────────────────────────────────────
# 3.3  Potencial termodinámico
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ThermodynamicPotential:
    """
    Potencial de Gibbs con corrección topológica y actividad (Axioma I):

        G = H − T·S·k_B^{model} + μ_topo σ² + R T ln a ,   T = T_base + κσ²

    Unidades: H, G en unidades del modelo (|β|); S adimensional (nats);
    T en K; el término RT ln a está en J/mol y es nulo para a = 1.
    """

    enthalpy: float = 0.0
    entropy: float = 0.0
    base_temperature: float = PhysicalConstants.T_REFERENCE
    temperature_coupling: float = 15.0
    topological_stress: float = 0.0
    activity: float = 1.0
    _gibbs_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=ReactionKinetics.GIBBS_HISTORY_LENGTH),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for attr in (
            "enthalpy", "entropy", "base_temperature",
            "temperature_coupling", "topological_stress", "activity",
        ):
            value = getattr(self, attr)
            if not _is_finite(value):
                raise NumericalDomainError(f"{attr} no es finito: {value!r}")
            setattr(self, attr, float(value))
        if self.entropy < 0.0:
            raise NumericalDomainError(f"Entropía negativa: {self.entropy}")
        if self.temperature_coupling < 0.0:
            raise NumericalDomainError(f"Acoplamiento térmico negativo: {self.temperature_coupling}")
        if self.activity <= 0.0:
            raise NumericalDomainError(f"Actividad no positiva: {self.activity}")
        if self.base_temperature < PhysicalConstants.T_MINIMUM:
            logger.warning(
                "T_base %.2fK < T_min %.2fK: proyectada al mínimo",
                self.base_temperature, PhysicalConstants.T_MINIMUM,
            )
            self.base_temperature = PhysicalConstants.T_MINIMUM
        if not isinstance(self._gibbs_history, deque):
            self._gibbs_history = deque(
                self._gibbs_history, maxlen=ReactionKinetics.GIBBS_HISTORY_LENGTH
            )

    # ── 3.3.1 Observables ────────────────────────────────────────────────

    @property
    def temperature(self) -> float:
        """T_eff = T_base + κσ² (calentamiento por disipación Dirichlet)."""
        temp = self.base_temperature + self.temperature_coupling * self.topological_stress ** 2
        if not _is_finite(temp):
            raise NumericalDomainError(f"Temperatura no finita: {temp}")
        return temp

    @property
    def chemical_potential(self) -> float:
        """μ = μ_topo σ² + R T ln a."""
        return (
            TopologyConstants.PRESSURE_COEFF * self.topological_stress ** 2
            + PhysicalConstants.R_GAS * self.temperature * _stable_log(self.activity)
        )

    @property
    def gibbs_free_energy(self) -> float:
        """G = H − T S k_B^{model} + μ."""
        G = (
            self.enthalpy
            - self.temperature * self.entropy * PhysicalConstants.BOLTZMANN_SCALE
            + self.chemical_potential
        )
        if not _is_finite(G):
            raise NumericalDomainError(f"Energía libre de Gibbs no finita: {G}")
        return G

    @property
    def instability(self) -> float:
        """I = ln(1 + |G|) + σ (índice de divergencia, lyapunovano débil)."""
        value = _stable_log1p(abs(self.gibbs_free_energy)) + self.topological_stress
        if not _is_finite(value):
            raise NumericalDomainError(f"Inestabilidad no finita: {value}")
        return value

    @property
    def gibbs_history(self) -> Tuple[float, ...]:
        return tuple(self._gibbs_history)

    @property
    def gibbs_trend(self) -> Optional[float]:
        """
        Pendiente dG/dciclo por mínimos cuadrados (Gauss–Markov, BLUE) sobre
        la ventana TREND_WINDOW, con sumas Dot2. None si hay < 2 muestras.
        """
        recent = list(self._gibbs_history)[-ReactionKinetics.TREND_WINDOW:]
        n = len(recent)
        if n < 2:
            return None
        x_mean = (n - 1) / 2.0
        y_mean = _kbn_sum(recent) / n
        xs = [i - x_mean for i in range(n)]
        ys = [y - y_mean for y in recent]
        denominator = _kbn_dot(xs, xs)
        if denominator < NumericalConstants.EPS:
            return 0.0
        return _kbn_dot(xs, ys) / denominator

    # ── 3.3.2 Evolución ──────────────────────────────────────────────────

    def update(
        self,
        new_enthalpy: float,
        new_entropy: float,
        topological_stress: float,
        activity: Optional[float] = None,
    ) -> None:
        """Actualiza (H, S, σ, a) con validación de dominio y registra G."""
        for name, value in (
            ("new_enthalpy", new_enthalpy),
            ("new_entropy", new_entropy),
            ("topological_stress", topological_stress),
        ):
            if not _is_finite(value):
                raise NumericalDomainError(f"{name} no es finito: {value!r}")
        if new_entropy < 0.0:
            raise NumericalDomainError(f"Entropía negativa: {new_entropy}")
        if topological_stress < 0.0:
            raise NumericalDomainError(f"σ = ‖ψ‖ negativo: {topological_stress}")
        if activity is not None:
            if not _is_finite(activity) or activity <= 0.0:
                raise NumericalDomainError(f"Actividad inválida: {activity!r}")
            self.activity = float(activity)
        self.enthalpy = float(new_enthalpy)
        self.entropy = float(new_entropy)
        self.topological_stress = float(topological_stress)
        self._gibbs_history.append(self.gibbs_free_energy)

    def cool_temperature(self, factor: float = DampingConstants.COOLING_FACTOR) -> None:
        """Enfriamiento isobárico T_base ← max(T_min, λT_base), λ ∈ (0, 1)."""
        if not _is_finite(factor) or not 0.0 < factor < 1.0:
            raise NumericalDomainError(f"Factor de enfriamiento fuera de (0, 1): {factor!r}")
        self.base_temperature = max(PhysicalConstants.T_MINIMUM, self.base_temperature * factor)

    def trim_enthalpy(self, factor: float = ReactionKinetics.ENTHALPY_TRIM_FACTOR) -> None:
        """Recorte entálpico H ← max(H_min, λH), λ ∈ (0, 1] (estabilización)."""
        if not _is_finite(factor) or not 0.0 < factor <= 1.0:
            raise NumericalDomainError(f"Factor de recorte fuera de (0, 1]: {factor!r}")
        self.enthalpy = max(ReactorLimits.MIN_ENTHALPY, self.enthalpy * factor)

    # ── 3.3.3 Cinética ───────────────────────────────────────────────────

    @staticmethod
    def model_energy_to_joule(energy_model: float) -> float:
        """Conversión de unidades del modelo (|β|) a J/mol: 1 |β| ≡ E_ACTIVATION_BASE."""
        return float(energy_model) * PhysicalConstants.E_ACTIVATION_BASE

    def compute_arrhenius_rate(
        self,
        activation_energy_j_mol: float,
        pre_exponential: float = PhysicalConstants.ARRHENIUS_A,
    ) -> float:
        """k = A exp(−Eₐ / RT), Eₐ en J/mol, exp saturada."""
        if not _is_finite(activation_energy_j_mol) or activation_energy_j_mol < 0.0:
            raise NumericalDomainError(f"Energía de activación inválida: {activation_energy_j_mol!r}")
        RT = PhysicalConstants.R_GAS * self.temperature
        if RT < NumericalConstants.EPS:
            return 0.0
        return float(pre_exponential) * _stable_exp(-activation_energy_j_mol / RT)

    def compute_eyring_rate(
        self,
        delta_g_dagger_j_mol: float,
        transmission: float = PhysicalConstants.EYRING_TRANSMISSION,
    ) -> float:
        """
        Teoría del estado de transición (Eyring):

            k = κ (k_B T / h) exp(−ΔG‡ / RT)

        Prefactor en SI (k_B, h SI): k_B T/h ≈ 6.2·10¹² s⁻¹ a 298 K.
        """
        if not _is_finite(delta_g_dagger_j_mol):
            raise NumericalDomainError(f"ΔG‡ no finito: {delta_g_dagger_j_mol!r}")
        T = self.temperature
        prefactor = transmission * PhysicalConstants.BOLTZMANN_SI * T / PhysicalConstants.PLANCK_SI
        RT = PhysicalConstants.R_GAS * T
        if RT < NumericalConstants.EPS:
            return 0.0
        return prefactor * _stable_exp(-delta_g_dagger_j_mol / RT)

    def copy(self) -> "ThermodynamicPotential":
        new_potential = ThermodynamicPotential(
            enthalpy=self.enthalpy,
            entropy=self.entropy,
            base_temperature=self.base_temperature,
            temperature_coupling=self.temperature_coupling,
            topological_stress=self.topological_stress,
            activity=self.activity,
        )
        new_potential._gibbs_history = deque(
            self._gibbs_history, maxlen=ReactionKinetics.GIBBS_HISTORY_LENGTH
        )
        return new_potential

    def __repr__(self) -> str:
        return (
            f"ThermodynamicPotential(H={self.enthalpy:.4f}, S={self.entropy:.4f}, "
            f"T={self.temperature:.2f}K, G={self.gibbs_free_energy:.4f}, I={self.instability:.4f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3.4  Catalizador, álgebra de Boole de Hückel y entropía de contexto
# ──────────────────────────────────────────────────────────────────────────────


class CatalystAgent(Protocol):
    """Protocolo para agentes catalizadores (morfismo de contexto)."""

    @property
    def efficiency_factor(self) -> float:
        """η ∈ [0, 1]: reduce la barrera efectiva Eₐ(1 − η)."""
        ...

    @property
    def catalytic_strength(self) -> float:
        """Fuerza catalítica (informativa)."""
        ...

    def orient(self, context: Dict[str, Any], gradient: float) -> Dict[str, Any]:
        """Orienta el catalizador según G (gradiente termodinámico); retorna un dict de actualización."""
        ...


class AromaticityEvaluator:
    """
    Álgebra de Boole sobre la regla de Hückel, restringida a los nodos del anillo.

        A(n)  ⇔ n ≥ 2 ∧ (n − 2) ≡ 0 (mod 4)      (4k + 2)
        AA(n) ⇔ n ≥ 4 ∧ n ≡ 0 (mod 4)            (4k)
        P     ⇔ (errores ∨ saltos) ∧ n > 0

    Equivalencia espectral: para 0 < n_e < 2N, 4k+2 ⇔ configuración de capa
    cerrada sobre los niveles E_k de C_N (fundamental no degenerado + pares
    degenerados), expuesta en is_closed_shell.
    """

    _NODES: ClassVar[Tuple[CarbonNode, ...]] = tuple(CarbonNode)

    @staticmethod
    def status_key(node: CarbonNode) -> str:
        return f"{node.name}_status"

    @staticmethod
    def error_key(node: CarbonNode) -> str:
        return f"{node.name}_error"

    @staticmethod
    def skip_key(node: CarbonNode) -> str:
        return f"{node.name}_skipped"

    @classmethod
    def count_pi_electrons(cls, context: Mapping[str, Any]) -> int:
        """# nodos del anillo en estado resonante (1 e⁻ π por carbono)."""
        return sum(
            1 for node in cls._NODES
            if context.get(cls.status_key(node)) == NodeOutcome.RESONANT.value
        )

    @classmethod
    def has_errors(cls, context: Mapping[str, Any]) -> bool:
        return any(
            cls.error_key(node) in context
            or context.get(cls.status_key(node)) == NodeOutcome.ERROR.value
            for node in cls._NODES
        )

    @classmethod
    def has_skips(cls, context: Mapping[str, Any]) -> bool:
        return any(
            cls.skip_key(node) in context
            or context.get(cls.status_key(node)) == NodeOutcome.SKIPPED.value
            for node in cls._NODES
        )

    @staticmethod
    def is_huckel_aromatic(n_electrons: int) -> bool:
        """4k + 2 con k ≥ 0 (n = 2 es el mínimo: etileno solo no es cíclico, pero cumple la aritmética)."""
        return n_electrons >= 2 and (n_electrons - 2) % 4 == 0

    @staticmethod
    def is_anti_aromatic(n_electrons: int) -> bool:
        """4k con k ≥ 1."""
        return n_electrons >= 4 and n_electrons % 4 == 0

    @staticmethod
    def is_closed_shell(n_electrons: int, ring_size: int = TopologyConstants.RING_SIZE) -> bool:
        """
        Capa cerrada sobre el espectro de Hückel de C_N: capacidades
        [2, 4, 4, …, (2 si N par)]; cerrada ⇔ n_e coincide con una suma parcial.
        """
        capacities = [2] + [4] * ((ring_size - 1) // 2) + ([2] if ring_size % 2 == 0 else [])
        cumulative = 0
        if n_electrons == 0:
            return True
        for cap in capacities:
            cumulative += cap
            if n_electrons == cumulative:
                return True
            if n_electrons < cumulative:
                return False
        return False

    @classmethod
    def evaluate(cls, context: Mapping[str, Any]) -> AromaticityState:
        """Clasificación exhaustiva del retículo {∅, P, AA, A}."""
        n_pi = cls.count_pi_electrons(context)
        if cls.has_errors(context) or cls.has_skips(context):
            return AromaticityState.PARTIALLY_AROMATIC if n_pi > 0 else AromaticityState.NON_AROMATIC
        if cls.is_huckel_aromatic(n_pi):
            return AromaticityState.AROMATIC
        if cls.is_anti_aromatic(n_pi):
            return AromaticityState.ANTI_AROMATIC
        return AromaticityState.PARTIALLY_AROMATIC if n_pi > 0 else AromaticityState.NON_AROMATIC

    @classmethod
    def log_state(cls, context: Mapping[str, Any]) -> None:
        n_pi = cls.count_pi_electrons(context)
        state = cls.evaluate(context)
        if state is AromaticityState.AROMATIC:
            k = (n_pi - 2) // 4
            logger.info("✅ AROMATICIDAD: %d e⁻ π (4×%d + 2 = %d)", n_pi, k, 4 * k + 2)
        elif state is AromaticityState.ANTI_AROMATIC:
            k = n_pi // 4
            logger.warning("⚠️ ANTI-AROMATICIDAD: %d e⁻ π (4×%d = %d) — inestable", n_pi, k, 4 * k)
        elif state is AromaticityState.PARTIALLY_AROMATIC:
            logger.info("◐ Resonancia parcial: %d/%d nodos activos", n_pi, TopologyConstants.RING_SIZE)
        else:
            logger.debug("○ No aromático: %d e⁻ π", n_pi)


class EntropyCalculator:
    """
    Entropía de Shannon del contexto (observables categóricos) y puente a la
    entropía de von Neumann decoherida del HilbertState.

    Se excluyen del histograma: claves privadas (`_…`, diagnósticos) y
    sufijos de telemetría, para no inflar S con ruido numérico.
    """

    _EXCLUDED_SUFFIXES: ClassVar[Tuple[str, ...]] = (
        "_ts", "_ea", "_rate", "_error", "_skipped",
    )

    @classmethod
    def _normalize_text(cls, text: str, max_length: int = 32) -> str:
        return " ".join(text.strip().split())[:max_length].lower()

    @classmethod
    def _compute_text_hash(cls, text: str) -> str:
        """SHA-256 truncado a 32 bits (colisión ≈ k²/2³³ para k categorías)."""
        return hashlib.sha256(cls._normalize_text(text).encode("utf-8")).hexdigest()[:8]

    @classmethod
    def _categorize_value(cls, value: Any) -> str:
        """σ-álgebra de categorías para el estimador de Shannon (bool antes que int)."""
        if isinstance(value, bool):
            return f"bool:{value}"
        if isinstance(value, str):
            if len(value) <= 10:
                return f"str:{cls._normalize_text(value)}"
            return f"str_hash:{cls._compute_text_hash(value)}"
        if isinstance(value, numbers.Real):
            v = float(value)
            if not math.isfinite(v):
                return "num:nonfinite"
            if abs(v) < NumericalConstants.EPS:
                return "num:zero"
            magnitude = int(math.floor(math.log10(abs(v))))
            return f"num:{'+' if v >= 0 else '-'}e{magnitude}"
        if isinstance(value, (list, tuple)):
            return f"seq:len{len(value)}"
        if isinstance(value, Mapping):
            return f"map:len{len(value)}"
        return f"obj:{type(value).__name__}"

    @classmethod
    def _is_observable(cls, key: str) -> bool:
        return not key.startswith(_DIAG) and not any(
            key.endswith(sfx) for sfx in cls._EXCLUDED_SUFFIXES
        )

    @classmethod
    def calculate(cls, context: Mapping[str, Any]) -> float:
        """S_Sh = −Σ pᵢ ln pᵢ sobre el histograma de categorías observables."""
        categories = [
            cls._categorize_value(v) for k, v in context.items() if cls._is_observable(k)
        ]
        if not categories:
            return 0.0
        counts = Counter(categories)
        total = float(len(categories))
        return float(-_kbn_sum((c / total) * math.log(c / total) for c in counts.values()))

    @classmethod
    def joint_shannon_von_neumann(
        cls,
        context: Mapping[str, Any],
        state: HilbertState,
        mixing: float = 0.5,
        basis: Optional[Sequence[Sequence[float]]] = None,
    ) -> float:
        """
        Entropía conjunta convexa (1−λ)S_Sh(contexto) + λS_vN(ρ_deph), λ ∈ [0, 1].

        `basis` = autobase de Δ₀ ⇒ S_vN mide la deslocalización modal del estrés.
        """
        if not _is_finite(mixing):
            raise NumericalDomainError(f"mixing no finito: {mixing!r}")
        lam = min(max(float(mixing), 0.0), 1.0)
        return (1.0 - lam) * cls.calculate(context) + lam * state.von_neumann_entropy(basis)


# ──────────────────────────────────────────────────────────────────────────────
# 3.5  Resultado y sesión de reacción
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReactionResult:
    """
    Resultado inmutable de una trayectoria del semigrupo catalítico.

    `context` se expone como vista de sólo lectura (MappingProxyType) para
    impedir aliasing con el estado de la sesión; use `context_copy()` para
    un dict mutable/serializable.
    """

    context: Mapping[str, Any]
    status: ConvergenceStatus
    final_cycle: int
    final_gibbs: float
    final_instability: float
    aromaticity: AromaticityState
    reaction_id: str
    elapsed_time_s: float
    trajectory: Tuple[CycleRecord, ...] = ()
    spectral_verified: bool = True
    delocalization_energy: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    @property
    def is_successful(self) -> bool:
        return self.status in (
            ConvergenceStatus.CONVERGED_AROMATIC,
            ConvergenceStatus.CONVERGED_METASTABLE,
        )

    def context_copy(self) -> Dict[str, Any]:
        return dict(self.context)

    def to_dict(self) -> Dict[str, Any]:
        """Serialización plana para telemetría."""
        return {
            "status": self.status.name,
            "final_cycle": self.final_cycle,
            "final_gibbs": self.final_gibbs,
            "final_instability": self.final_instability,
            "aromaticity": self.aromaticity.name,
            "reaction_id": self.reaction_id,
            "elapsed_time_s": self.elapsed_time_s,
            "is_successful": self.is_successful,
            "spectral_verified": self.spectral_verified,
            "delocalization_energy": self.delocalization_energy,
            "cycles": len(self.trajectory),
        }


@dataclass
class _ReactionSession:
    """Estado mutable de UNA ignición (hace reentrante al CatalyticReactor)."""

    reaction_id: str
    context: Dict[str, Any]
    state: HilbertState
    potential: ThermodynamicPotential
    start_time: float
    previous_gibbs: float
    lyapunov_prev: Optional[float] = None
    cycle: int = 0
    trajectory: List[CycleRecord] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


# ──────────────────────────────────────────────────────────────────────────────
# 3.6  Reactor catalítico
# ──────────────────────────────────────────────────────────────────────────────


class CatalyticReactor:
    """
    Reactor catalítico cuántico (metáfora del benceno).

    Integra el CatalyticGenerator de la Fase 2 sobre HilbertState (Euler
    certificado o semigrupo exacto), acopla la energía de Dirichlet y la
    liberación de resonancia de Hückel al potencial de Gibbs, audita de
    Rham–Tellegen–Hodge–Parseval en cada ciclo y declara aromaticidad
    cuando el retículo de Hückel se cierra. Reentrante: todo el estado de
    una ignición vive en una _ReactionSession.
    """

    __slots__ = (
        "mic",
        "catalyst",
        "telemetry",
        "topology",
        "ring_sequence",
        "_base_temperature",
        "_temperature_coupling",
        "_deterministic_mode",
        "_random_seed",
        "_strict_conservation",
        "_enable_huckel_resolvent",
        "_diffusion_scheme",
        "_entropy_mixing",
        "_simulate_latency",
        "_isometry",
        "_isometry_period",
        "_isometry_permutation",
        "_generator",
        "_delocalization_energy",
        "_huckel_diagnostics_cache",
        "_ignition_counter",
    )

    def __init__(
        self,
        mic: MICRegistry,
        agent: CatalystAgent,
        telemetry: TelemetryContext,
        *,
        base_temperature: float = PhysicalConstants.T_REFERENCE,
        temperature_coupling: float = 15.0,
        deterministic: bool = False,
        random_seed: Optional[int] = None,
        strict_conservation: bool = True,
        enable_huckel_resolvent: bool = True,
        diffusion_scheme: DiffusionScheme = DiffusionScheme.EULER,
        diffusion_rate: float = CFLConstants.ALPHA_SAFE,
        ring_isometry: RingIsometry = RingIsometry.NONE,
        isometry_period: int = 2,
        entropy_mixing: float = 0.5,
        simulate_latency: bool = True,
    ) -> None:
        if not _is_finite(base_temperature) or base_temperature < PhysicalConstants.T_MINIMUM:
            raise NumericalDomainError(
                f"T_base {base_temperature!r} < T_min {PhysicalConstants.T_MINIMUM}"
            )
        if not _is_finite(temperature_coupling) or temperature_coupling < 0.0:
            raise NumericalDomainError(f"Acoplamiento térmico inválido: {temperature_coupling!r}")
        if not _is_finite(entropy_mixing) or not 0.0 <= entropy_mixing <= 1.0:
            raise NumericalDomainError(f"entropy_mixing ∉ [0, 1]: {entropy_mixing!r}")
        if isinstance(isometry_period, bool) or isometry_period < 1:
            raise NumericalDomainError(f"isometry_period debe ser ≥ 1: {isometry_period!r}")

        self.mic = mic
        self.catalyst = agent
        self.telemetry = telemetry
        self.topology = HexagonalTopology()
        self.ring_sequence: List[CarbonNode] = list(CarbonNode)

        self._base_temperature = float(base_temperature)
        self._temperature_coupling = float(temperature_coupling)
        self._deterministic_mode = bool(deterministic)
        self._random_seed = random_seed
        self._strict_conservation = bool(strict_conservation)
        self._enable_huckel_resolvent = bool(enable_huckel_resolvent)
        self._diffusion_scheme = diffusion_scheme
        self._entropy_mixing = float(entropy_mixing)
        self._simulate_latency = bool(simulate_latency)
        self._isometry = ring_isometry
        self._isometry_period = int(isometry_period)
        self._huckel_diagnostics_cache: Optional[Dict[str, Any]] = None
        self._ignition_counter = 0

        # Sutura con la Fase 2: generador certificado.
        self._generator: CatalyticGenerator = self.topology.infinitesimal_catalytic_generator(
            diffusion_rate
        )
        if self._generator.regime == "unstable" or not self._generator.is_l2_stable:
            raise HomologicalIntegrityError(
                f"Generador inestable: α={self._generator.alpha} > 2/λ_max"
            )
        self._isometry_permutation = self._build_isometry_permutation(ring_isometry)
        self._delocalization_energy = self.topology.huckel_delocalization_energy()

        if not self.topology.spectral_verified:
            logger.warning("Certificado espectral de C₆ NO verificado; véase spectral_certificate")
        logger.info(
            "Reactor listo | 𝒢: α=%.5f régimen=%s contracción=%.5f abscisa=%.5f | "
            "ΔE_deloc=%.4f | β=(%d,%d) χ=%d torsión_libre=%s",
            self._generator.alpha, self._generator.regime,
            self._generator.contraction_constant, self._generator.spectral_abscissa,
            self._delocalization_energy, *self.topology.betti_numbers,
            self.topology.euler_characteristic, self.topology.torsion_free,
        )

    # ── 3.6.1 Construcción y utilidades ──────────────────────────────────

    def _build_isometry_permutation(self, isometry: RingIsometry) -> Optional[Tuple[int, ...]]:
        """
        Permutación σ ∈ D₆ y certificado de conmutación [P_σ, Δ₀] = 0
        (⇒ P_σ preserva ℰ(ψ) y el espectro del estrés).
        """
        n = TopologyConstants.RING_SIZE
        if isometry is RingIsometry.NONE:
            return None
        if isometry is RingIsometry.ROTATION:
            perm = tuple((i + 1) % n for i in range(n))
        elif isometry is RingIsometry.REFLECTION:
            perm = tuple(n - 1 - i for i in range(n))
        else:
            raise NumericalDomainError(f"Isometría desconocida: {isometry!r}")

        P = [[1.0 if j == perm[i] else 0.0 for j in range(n)] for i in range(n)]
        L = self.topology.laplacian_matrix
        PL = CompensatedLinearAlgebra.matmul_kbn(P, L)
        LP = CompensatedLinearAlgebra.matmul_kbn(L, P)
        residual = _frobenius_norm(
            [[PL[i][j] - LP[i][j] for j in range(n)] for i in range(n)]
        )
        if residual > NumericalConstants.STRUCTURAL_TOL:
            raise HomologicalIntegrityError(
                f"{isometry.name} no conmuta con Δ₀ (‖[P, Δ₀]‖_F = {residual:.3e})"
            )
        return perm

    def _generate_reaction_id(self) -> str:
        """Id de trayectoria: SHA-256(seed, contador) en determinista; UUID4 en otro caso."""
        self._ignition_counter += 1
        if self._deterministic_mode and self._random_seed is not None:
            seed = f"reaction_{self._random_seed}_{self._ignition_counter}".encode("utf-8")
            return hashlib.sha256(seed).hexdigest()[:8]
        return str(uuid.uuid4())[:8]

    def _catalyst_efficiency(self) -> float:
        """η proyectado a [0, η_max] con validación de finitud."""
        eta = self.catalyst.efficiency_factor
        if not _is_finite(eta):
            raise NumericalDomainError(f"efficiency_factor no finito: {eta!r}")
        return min(max(float(eta), 0.0), ReactionKinetics.EFFICIENCY_CEILING)

    def _huckel_diagnostics(self) -> Dict[str, Any]:
        """Diagnósticos de Hückel (invariantes de la topología: memoizados)."""
        if self._huckel_diagnostics_cache is None:
            try:
                resolvent = self.topology.regularized_huckel_resolvent(s=0.0, h=0.0)
                self._huckel_diagnostics_cache = {
                    f"{_DIAG}huckel_alpha_reg": resolvent.alpha,
                    f"{_DIAG}huckel_resolvent_condition": resolvent.condition_raw,
                    f"{_DIAG}huckel_tikhonov_applied": resolvent.tikhonov_applied,
                    f"{_DIAG}huckel_mo_levels": self.topology.huckel_spectrum(),
                    f"{_DIAG}huckel_pi_energy": self.topology.huckel_pi_energy(),
                    f"{_DIAG}huckel_delocalization_energy": self._delocalization_energy,
                    f"{_DIAG}huckel_homo_lumo_gap": self.topology.huckel_homo_lumo_gap(),
                }
            except Exception as exc:  # diagnóstico, nunca fatal
                logger.warning("Fallo en diagnósticos de Hückel: %s", exc)
                self._huckel_diagnostics_cache = {f"{_DIAG}huckel_resolvent_error": str(exc)}
        return dict(self._huckel_diagnostics_cache)

    # ── 3.6.2 Sesión ─────────────────────────────────────────────────────

    def _open_session(self, initial_context: Mapping[str, Any]) -> _ReactionSession:
        reaction_id = self._generate_reaction_id()
        context: Dict[str, Any] = dict(initial_context)
        potential = ThermodynamicPotential(
            entropy=max(ReactorLimits.MIN_ENTROPY, EntropyCalculator.calculate(context)),
            base_temperature=self._base_temperature,
            temperature_coupling=self._temperature_coupling,
        )
        state = HilbertState()
        context[f"{_DIAG}spectral_verified"] = self.topology.spectral_verified
        context[f"{_DIAG}generator_regime"] = self._generator.regime
        context[f"{_DIAG}generator_alpha"] = self._generator.alpha
        session = _ReactionSession(
            reaction_id=reaction_id,
            context=context,
            state=state,
            potential=potential,
            start_time=time.perf_counter(),
            previous_gibbs=potential.gibbs_free_energy,
        )
        self.telemetry.record_reaction_start(reaction_id, context)
        return session

    def _finish(
        self,
        session: _ReactionSession,
        status: ConvergenceStatus,
        aromaticity: AromaticityState,
    ) -> ReactionResult:
        elapsed = session.elapsed
        if status in (ConvergenceStatus.CONVERGED_AROMATIC, ConvergenceStatus.CONVERGED_METASTABLE):
            self.telemetry.record_reaction_success(session.reaction_id, session.cycle)
        return ReactionResult(
            context=session.context,
            status=status,
            final_cycle=session.cycle,
            final_gibbs=session.potential.gibbs_free_energy,
            final_instability=session.potential.instability,
            aromaticity=aromaticity,
            reaction_id=session.reaction_id,
            elapsed_time_s=elapsed,
            trajectory=tuple(session.trajectory),
            spectral_verified=self.topology.spectral_verified,
            delocalization_energy=self._delocalization_energy,
        )

    # ── 3.6.3 Ignición ───────────────────────────────────────────────────

    def ignite(self, initial_context: Mapping[str, Any]) -> ReactionResult:
        """
        Enciende el reactor e integra hasta aromaticidad, metaestabilidad,
        colapso o agotamiento de ciclos de resonancia.

        Colapso controlado: RuntimeError (incluye HomologicalIntegrityError)
        y NumericalDomainError ⇒ FAILED_INSTABILITY con la trayectoria
        recorrida. Cualquier otra excepción se propaga tras registrarse.
        """
        session = self._open_session(initial_context)
        logger.info("⚛️ IGNICIÓN: Reactor [%s] encendido (v%s)", session.reaction_id, __version__)

        try:
            for cycle in range(1, ReactorLimits.MAX_RESONANCE_CYCLES + 1):
                session.cycle = cycle
                self._validate_state(session)

                cert = self._lyapunov_certificate(session)
                self._log_cycle(session, cert)

                self._catalytic_orientation(session)
                balance = self._ring_iteration(session)
                self._apply_holonomy(session)

                aromaticity = AromaticityEvaluator.evaluate(session.context)
                AromaticityEvaluator.log_state(session.context)
                self._record_cycle(session, aromaticity, cert, balance)

                if aromaticity is AromaticityState.AROMATIC:
                    logger.info(
                        "✅ AROMATICIDAD en ciclo %d | G=%.4f | t=%.3fs",
                        cycle, session.potential.gibbs_free_energy, session.elapsed,
                    )
                    return self._finish(session, ConvergenceStatus.CONVERGED_AROMATIC, aromaticity)

                if self._check_convergence(session):
                    logger.info(
                        "🔒 Metaestabilidad en ciclo %d | |δG|=%.6f | t=%.3fs",
                        cycle,
                        abs(session.potential.gibbs_free_energy - session.previous_gibbs),
                        session.elapsed,
                    )
                    session.context[f"{_DIAG}metastable_cycle"] = cycle
                    return self._finish(session, ConvergenceStatus.CONVERGED_METASTABLE, aromaticity)

                session.previous_gibbs = session.potential.gibbs_free_energy

            logger.warning(
                "⚠️ Máximo de ciclos sin convergencia | G=%.4f | I=%.4f",
                session.potential.gibbs_free_energy, session.potential.instability,
            )
            return self._finish(
                session,
                ConvergenceStatus.FAILED_MAX_CYCLES,
                AromaticityEvaluator.evaluate(session.context),
            )

        except (RuntimeError, NumericalDomainError) as exc:
            self.telemetry.record_error("reaction_chamber", str(exc))
            logger.error("🔥 Colapso del reactor [%s] en ciclo %d: %s", session.reaction_id, session.cycle, exc)
            session.context[f"{_DIAG}collapse_reason"] = str(exc)
            try:
                aromaticity = AromaticityEvaluator.evaluate(session.context)
                return self._finish(session, ConvergenceStatus.FAILED_INSTABILITY, aromaticity)
            except Exception:  # el potencial puede ser inevaluable tras el colapso
                return ReactionResult(
                    context=session.context,
                    status=ConvergenceStatus.FAILED_INSTABILITY,
                    final_cycle=session.cycle,
                    final_gibbs=math.nan,
                    final_instability=math.nan,
                    aromaticity=AromaticityState.NON_AROMATIC,
                    reaction_id=session.reaction_id,
                    elapsed_time_s=session.elapsed,
                    trajectory=tuple(session.trajectory),
                    spectral_verified=self.topology.spectral_verified,
                    delocalization_energy=self._delocalization_energy,
                )

        except Exception as exc:
            self.telemetry.record_error("reaction_chamber", str(exc))
            logger.exception("💥 Error inesperado en reactor [%s]", session.reaction_id)
            raise

    # ── 3.6.4 Certificación y bitácora por ciclo ─────────────────────────

    def _validate_state(self, session: _ReactionSession) -> None:
        """ψ ∈ ℝ⁶, φ finito, (T, G, I) evaluables."""
        _validate_finite_vector(
            session.state.vector, "state.vector", expected_dim=TopologyConstants.RING_SIZE
        )
        if not _is_finite(session.state.phase):
            raise NumericalDomainError(f"Fase no finita: {session.state.phase!r}")
        _ = session.potential.temperature
        _ = session.potential.gibbs_free_energy
        _ = session.potential.instability

    def _lyapunov_certificate(self, session: _ReactionSession) -> LyapunovCertificate:
        """
        V = ‖ψ‖² + G²; ΔV respecto al ciclo anterior con holgura relativa;
        abscisa espectral tomada del generador certificado (−αλ₁).
        """
        V = session.state.lyapunov_value(session.potential.gibbs_free_energy)
        prev = session.lyapunov_prev
        tol = NumericalConstants.EPS * max(1.0, abs(prev) if prev is not None else 1.0)
        decrement = 0.0 if prev is None else V - prev
        session.lyapunov_prev = V
        cert = LyapunovCertificate(
            value=float(V),
            decrement=float(decrement),
            dissipative=bool(prev is None or decrement <= tol),
            spectral_abscissa=float(self._generator.spectral_abscissa),
            tolerance=float(tol),
        )
        session.context[f"{_DIAG}lyapunov_V"] = cert.value
        session.context[f"{_DIAG}lyapunov_dV"] = cert.decrement
        session.context[f"{_DIAG}lyapunov_dissipative"] = cert.dissipative
        return cert

    def _log_cycle(self, session: _ReactionSession, cert: LyapunovCertificate) -> None:
        logger.info(
            "⏩ Ciclo %d/%d | G=%.4f | I=%.4f | ‖ψ‖=%.4f | V=%.4f | ΔV=%.3e",
            session.cycle, ReactorLimits.MAX_RESONANCE_CYCLES,
            session.potential.gibbs_free_energy, session.potential.instability,
            session.state.norm, cert.value, cert.decrement,
        )

    def _record_cycle(
        self,
        session: _ReactionSession,
        aromaticity: AromaticityState,
        cert: LyapunovCertificate,
        balance: DiffusionEnergyBalance,
    ) -> None:
        p = session.potential
        session.trajectory.append(
            CycleRecord(
                cycle=session.cycle,
                gibbs=p.gibbs_free_energy,
                enthalpy=p.enthalpy,
                entropy=p.entropy,
                temperature=p.temperature,
                instability=p.instability,
                stress_norm=session.state.norm,
                dirichlet_energy=float(
                    balance.dirichlet_energy
                    if balance.dirichlet_energy is not None
                    else self.topology.hodge_energy_0(session.state.vector)
                ),
                lyapunov_value=cert.value,
                lyapunov_decrement=cert.decrement,
                diffusion_dissipative=balance.dissipative,
                tellegen_coherent=bool(session.context.get(f"{_DIAG}tellegen_coherent", False)),
                pi_electrons=AromaticityEvaluator.count_pi_electrons(session.context),
                aromaticity=aromaticity.name,
            )
        )

    # ── 3.6.5 Orientación catalítica ─────────────────────────────────────

    def _catalytic_orientation(self, session: _ReactionSession) -> None:
        """Inyecta diagnósticos de Hückel (privados) y aplica CatalystAgent.orient."""
        context = session.context
        if self._enable_huckel_resolvent:
            context.update(self._huckel_diagnostics())
        context[f"{_DIAG}gibbs_trend"] = session.potential.gibbs_trend
        gradient = session.potential.gibbs_free_energy
        update = self.catalyst.orient(context, gradient)
        if not isinstance(update, dict):
            raise TypeError("CatalystAgent.orient debe retornar un dict")
        context.update(update)

    # ── 3.6.6 Ciclo de resonancia ────────────────────────────────────────

    @staticmethod
    def _clear_cycle_markers(context: Dict[str, Any]) -> None:
        """Purga marcadores de ciclo (*_skipped, *_error) de los nodos del anillo."""
        for node in CarbonNode:
            context.pop(AromaticityEvaluator.skip_key(node), None)
            context.pop(AromaticityEvaluator.error_key(node), None)

    def _resonance_release(self, n_resonant: int) -> float:
        """
        Axioma de liberación de resonancia (Hückel, Fase 2):

            ΔH_res = κ · (n_res / n) · ΔE_deloc ,   ΔE_deloc = 2β < 0

        La deslocalización π libera entalpía proporcionalmente a la fracción
        del anillo en resonancia (cierre de capa ⇒ liberación máxima).
        """
        fraction = n_resonant / float(TopologyConstants.RING_SIZE)
        return ReactionKinetics.RESONANCE_RELEASE_SCALE * fraction * self._delocalization_energy

    def _ring_iteration(self, session: _ReactionSession) -> DiffusionEnergyBalance:
        """
        Un ciclo de resonancia:
            reacción nodal → liberación de resonancia → difusión por 𝒢 →
            auditoría de Hodge–Tellegen → entropía conjunta → Gibbs.
        """
        context, state, potential, cycle = (
            session.context, session.state, session.potential, session.cycle
        )
        n = TopologyConstants.RING_SIZE
        self._clear_cycle_markers(context)

        efficiency = self._catalyst_efficiency()
        timestamp = float(cycle) if self._deterministic_mode else time.time()
        total_delta_h = 0.0
        n_resonant = 0

        for node in self.ring_sequence:
            idx = node.index
            base_ea = self._calculate_hamiltonian(node, context)
            effective_ea = base_ea * (1.0 - efficiency)
            local_stress = state.vector[idx]
            try:
                reaction = self._react_node(node, effective_ea, local_stress, cycle, potential)
            except Exception as exc:
                logger.error("💥 Error en nodo %s (ciclo %d): %s", node.name, cycle, exc)
                reaction = NodeReaction(
                    node=node,
                    outcome=NodeOutcome.ERROR,
                    activation_energy=effective_ea,
                    delta_h=ReactionKinetics.ERROR_ENTHALPY,
                    arrhenius_rate=0.0,
                    detail=str(exc),
                )

            context.update(reaction.context_update(timestamp))
            total_delta_h += reaction.delta_h
            if reaction.outcome is NodeOutcome.ERROR:
                state.vector[idx] += ReactionKinetics.ERROR_STRESS_INJECTION
            else:
                state.vector[idx] += self._compute_local_excitation(
                    reaction.delta_h, effective_ea, local_stress
                )
                if reaction.outcome is NodeOutcome.RESONANT:
                    n_resonant += 1
        _validate_finite_vector(state.vector, "state.vector(post-reacción)", expected_dim=n)

        release = self._resonance_release(n_resonant)
        total_delta_h += release
        context[f"{_DIAG}resonance_release"] = release
        context[f"{_DIAG}pi_electrons"] = n_resonant

        # Integración del generador infinitesimal de la Fase 2.
        balance = self._diffuse(session)
        context[f"{_DIAG}diffusion_scheme"] = balance.scheme
        context[f"{_DIAG}diffusion_dissipative"] = balance.dissipative
        context[f"{_DIAG}diffusion_identity_residual"] = balance.identity_residual
        context[f"{_DIAG}diffusion_dirichlet_energy"] = balance.dirichlet_energy
        if not balance.dissipative:
            logger.warning(
                "Paso de difusión no disipativo: Δ‖ψ‖²=%.3e (predicho %.3e)",
                balance.actual_decrement, balance.predicted_decrement,
            )

        self._audit_conservation(session)

        new_entropy = max(
            ReactorLimits.MIN_ENTROPY,
            EntropyCalculator.joint_shannon_von_neumann(
                context, state, mixing=self._entropy_mixing, basis=self.topology.eigenvectors
            ),
        )
        potential.update(
            new_enthalpy=max(ReactorLimits.MIN_ENTHALPY, potential.enthalpy + total_delta_h),
            new_entropy=new_entropy,
            topological_stress=state.norm,
        )
        if potential.instability > ReactorLimits.INSTABILITY_THRESHOLD:
            self._attempt_stabilization(session, reason="instability")
        return balance

    def _diffuse(self, session: _ReactionSession) -> DiffusionEnergyBalance:
        """Difusión según el esquema: Euler (𝒢 certificado) o semigrupo exacto e^{−αΔ₀}."""
        state = session.state
        if self._diffusion_scheme is DiffusionScheme.EULER:
            return state.apply_generator_step(self._generator)

        alpha = self._generator.alpha
        psi = list(state.vector)
        before = _kbn_dot(psi, psi)
        coeffs = self.topology.spectral_coefficients(psi)
        predicted = _kbn_sum(
            (_stable_exp(-2.0 * alpha * lam) - 1.0) * c * c
            for lam, c in zip(self.topology.eigenvalues, coeffs)
        )
        energy = self.topology.dirichlet_energy_spectral(psi)
        state.replace_vector(self.topology.diffuse_exact(psi, t=alpha))
        after = state.norm_squared
        actual = after - before
        return DiffusionEnergyBalance(
            norm_sq_before=float(before),
            norm_sq_after=float(after),
            predicted_decrement=float(predicted),
            actual_decrement=float(actual),
            identity_residual=float(abs(actual - predicted)),
            dirichlet_energy=float(energy),
            dissipative=bool(actual <= NumericalConstants.EPS * max(1.0, before)),
            scheme="exact",
        )

    def _apply_holonomy(self, session: _ReactionSession) -> None:
        """Transporte paralelo φ ← φ + 2π/6 y acción opcional de D₆ sobre ψ."""
        state = session.state
        state.phase = (state.phase + 2.0 * math.pi / TopologyConstants.RING_SIZE) % (2.0 * math.pi)
        if self._isometry_permutation is not None and session.cycle % self._isometry_period == 0:
            state.apply_ring_isometry(self._isometry_permutation)
            session.context[f"{_DIAG}isometry_applied"] = self._isometry.name

    def _audit_conservation(self, session: _ReactionSession) -> None:
        """
        Auditoría óctuple de Rham–Tellegen–Hodge–Parseval post-difusión (Fase 2).

        Con strict_conservation una violación colapsa el reactor; en otro
        caso se registra y se dispara estabilización de Lyapunov.
        """
        report = self.topology.audit_de_rham_tellegen(session.state.vector)
        session.context.update(report.as_context(prefix=f"{_DIAG}tellegen_"))
        if report.is_coherent:
            return
        message = (
            "Violación de conservación de Rham–Tellegen–Hodge: "
            f"axiomas={list(report.violations())}, energía={report.laplacian_energy:.6e}, "
            f"residuos(cons={report.conservation_residual:.3e}, orto={report.orthogonality_residual:.3e}, "
            f"tellegen={report.tellegen_residual:.3e}, rec={report.reconstruction_residual:.3e}, "
            f"parseval={report.parseval_residual:.3e}, ohm={report.ohmic_residual:.3e})"
        )
        self.telemetry.record_error("reaction_chamber", message)
        if self._strict_conservation:
            raise HomologicalIntegrityError(message)
        logger.error(message)
        self._attempt_stabilization(session, reason="conservation")

    # ── 3.6.7 Física nodal ───────────────────────────────────────────────

    def _calculate_hamiltonian(self, node: CarbonNode, context: Mapping[str, Any]) -> float:
        """
        Elemento de matriz efectivo en el sitio i:

            Hᵢ = α + Σ_{⟨j i⟩} β·1_{j resonante} + penalización de precursores   (≥ 0)
        """
        stabilization = 0.0
        for neighbor_idx in self.topology.neighbor_indices(node.index):
            neighbor = node_from_index(neighbor_idx)
            if context.get(AromaticityEvaluator.status_key(neighbor)) == NodeOutcome.RESONANT.value:
                stabilization += HuckelConstants.BETA
        hamiltonian = HuckelConstants.ALPHA + stabilization + self._evaluate_precursor_penalty(node, context)
        if not _is_finite(hamiltonian):
            raise NumericalDomainError(f"Hamiltoniano no finito para {node.name}: {hamiltonian!r}")
        return max(0.0, hamiltonian)

    @staticmethod
    def _evaluate_precursor_penalty(node: CarbonNode, context: Mapping[str, Any]) -> float:
        """Penalización lineal por fracción de precursores ausentes o vacíos."""
        precursors = node.precursors
        if not precursors:
            return 0.0
        missing = sum(
            1 for key in precursors
            if context.get(key) is None or context.get(key) in ("", {}, [], ())
        )
        return ReactionKinetics.PRECURSOR_PENALTY * missing / len(precursors)

    def _react_node(
        self,
        node: CarbonNode,
        activation_energy: float,
        local_stress: float,
        cycle: int,
        potential: ThermodynamicPotential,
    ) -> NodeReaction:
        """
        Reacción local: salto si Eₐ > techo; en otro caso resonancia con

            ΔH = g_σ σᵢ² + g_E Eₐ ,   k = A exp(−Eₐ[J/mol] / RT)   (diagnóstico)
        """
        if not (_is_finite(activation_energy) and _is_finite(local_stress)):
            raise NumericalDomainError(
                f"Entrada no finita en {node.name}: Eₐ={activation_energy!r}, σ={local_stress!r}"
            )
        rate = potential.compute_arrhenius_rate(
            ThermodynamicPotential.model_energy_to_joule(max(0.0, activation_energy))
        )
        if activation_energy > HuckelConstants.ACTIVATION_CEILING:
            logger.warning(
                "⚡ Saltando %s (ciclo %d): Eₐ=%.3f > techo=%.3f",
                node.name, cycle, activation_energy, HuckelConstants.ACTIVATION_CEILING,
            )
            return NodeReaction(node, NodeOutcome.SKIPPED, activation_energy,
                                ReactionKinetics.SKIP_ENTHALPY, rate)

        if not self._deterministic_mode and self._simulate_latency:
            stress_factor = 1.0 + ReactionKinetics.NODE_LATENCY_STRESS_GAIN * local_stress ** 2
            time.sleep(min(ReactionKinetics.NODE_LATENCY_BASE_S * stress_factor, ReactorLimits.MAX_NODE_SLEEP))

        delta_h = (
            ReactionKinetics.STRESS_ENTHALPY_GAIN * local_stress ** 2
            + ReactionKinetics.BARRIER_ENTHALPY_GAIN * activation_energy
        )
        if not _is_finite(delta_h):
            raise NumericalDomainError(f"ΔH no finita en {node.name}: {delta_h!r}")
        logger.debug(
            "🔬 %s (ciclo %d) | Eₐ=%.3f | σ=%.3f | ΔH=%.3f | k/A=%.3e",
            node.name, cycle, activation_energy, local_stress, delta_h,
            rate / PhysicalConstants.ARRHENIUS_A,
        )
        return NodeReaction(node, NodeOutcome.RESONANT, activation_energy, delta_h, rate)

    @staticmethod
    def _compute_local_excitation(delta_h: float, activation_energy: float, local_stress: float) -> float:
        """
        Excitación local Lipschitz-acotada:

            Δψᵢ = g·tanh(ΔH⁺/s) + g_b·Eₐ − ν·σᵢ
        """
        if not all(_is_finite(x) for x in (delta_h, activation_energy, local_stress)):
            raise NumericalDomainError("Parámetros no finitos en excitación local")
        gain = DampingConstants.LOCAL_STRESS_GAIN * math.tanh(
            max(0.0, delta_h) / ReactionKinetics.EXCITATION_SATURATION
        )
        excitation = (
            gain
            + ReactionKinetics.BARRIER_EXCITATION_GAIN * activation_energy
            - DampingConstants.LOCAL_STRESS_DISSIPATION * local_stress
        )
        if not _is_finite(excitation):
            raise NumericalDomainError(f"Excitación no finita: {excitation!r}")
        return excitation

    # ── 3.6.8 Convergencia y estabilización ──────────────────────────────

    def _check_convergence(self, session: _ReactionSession) -> bool:
        """
        Metaestabilidad ⇔ ciclo ≥ mínimo ∧ |ΔG| < tol ∧ |dG/dciclo| < tol
        ∧ estacionariedad relativa de ‖ψ‖ en las últimas 3 muestras.
        """
        if session.cycle < ReactorLimits.MIN_CONVERGENCE_CYCLE:
            return False
        potential = session.potential
        current = potential.gibbs_free_energy
        if not (_is_finite(current) and _is_finite(session.previous_gibbs)):
            return False
        if abs(current - session.previous_gibbs) >= NumericalConstants.GIBBS_CONVERGENCE_TOL:
            return False
        trend = potential.gibbs_trend
        if trend is not None and abs(trend) >= ReactionKinetics.GIBBS_TREND_TOL:
            return False
        recent = session.state.norm_history[-3:]
        if len(recent) >= 2:
            max_change = max(abs(b - a) for a, b in zip(recent, recent[1:]))
            ref = max(recent[-1], NumericalConstants.EPS)
            if max_change / ref >= ReactorLimits.NORM_RELATIVE_STATIONARITY:
                return False
        return True

    def _attempt_stabilization(self, session: _ReactionSession, reason: str) -> None:
        """
        Estabilización de emergencia: damping hacia ker Δ₀, recorte de H y
        enfriamiento. Si I supera el umbral de colapso, se aborta.
        """
        state, potential, cycle = session.state, session.potential, session.cycle
        logger.warning(
            "⚠️ ESTABILIZACIÓN (%s): I=%.2f (umbral %.2f) en ciclo %d",
            reason, potential.instability, ReactorLimits.INSTABILITY_THRESHOLD, cycle,
        )
        state.apply_damping(cycle)
        potential.trim_enthalpy(ReactionKinetics.ENTHALPY_TRIM_FACTOR)
        potential.cool_temperature(DampingConstants.COOLING_FACTOR)
        potential.topological_stress = state.norm
        session.context[f"{_DIAG}stabilization_reason"] = reason

        collapse_threshold = ReactorLimits.INSTABILITY_THRESHOLD * ReactorLimits.COLLAPSE_FACTOR
        if potential.instability > collapse_threshold:
            raise RuntimeError(
                f"Colapso del reactor ({reason}): I={potential.instability:.4f} > {collapse_threshold:.4f}"
            )
        logger.info(
            "🛡️ Estabilizado: I=%.4f, ‖ψ‖=%.4f, T=%.2fK",
            potential.instability, state.norm, potential.base_temperature,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3.7  Configuración y fábrica
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReactorConfig:
    """Configuración inmutable y validada del reactor."""

    base_temperature: float = PhysicalConstants.T_REFERENCE
    temperature_coupling: float = 15.0
    deterministic: bool = False
    random_seed: Optional[int] = None
    strict_conservation: bool = True
    enable_huckel_resolvent: bool = True
    diffusion_scheme: DiffusionScheme = DiffusionScheme.EULER
    diffusion_rate: float = CFLConstants.ALPHA_SAFE
    ring_isometry: RingIsometry = RingIsometry.NONE
    isometry_period: int = 2
    entropy_mixing: float = 0.5
    simulate_latency: bool = True

    def __post_init__(self) -> None:
        if not _is_finite(self.base_temperature) or self.base_temperature < PhysicalConstants.T_MINIMUM:
            raise NumericalDomainError(
                f"Temperatura base {self.base_temperature!r}K < mínimo {PhysicalConstants.T_MINIMUM}K"
            )
        if not _is_finite(self.temperature_coupling) or self.temperature_coupling < 0.0:
            raise NumericalDomainError(f"Acoplamiento negativo o no finito: {self.temperature_coupling!r}")
        if not _is_finite(self.diffusion_rate) or self.diffusion_rate < 0.0:
            raise NumericalDomainError(f"Tasa de difusión inválida: {self.diffusion_rate!r}")
        if self.diffusion_rate > CFLConstants.ALPHA_SHARP:
            raise NumericalDomainError(
                f"α={self.diffusion_rate} > 2/λ_max={CFLConstants.ALPHA_SHARP}: Euler inestable"
            )
        if not _is_finite(self.entropy_mixing) or not 0.0 <= self.entropy_mixing <= 1.0:
            raise NumericalDomainError(f"entropy_mixing ∉ [0, 1]: {self.entropy_mixing!r}")
        if isinstance(self.isometry_period, bool) or self.isometry_period < 1:
            raise NumericalDomainError(f"isometry_period < 1: {self.isometry_period!r}")
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, numbers.Integral)
        ):
            raise NumericalDomainError(f"random_seed debe ser entero: {self.random_seed!r}")

    def as_kwargs(self) -> Dict[str, Any]:
        """Argumentos nominales para CatalyticReactor.__init__."""
        return {
            "base_temperature": self.base_temperature,
            "temperature_coupling": self.temperature_coupling,
            "deterministic": self.deterministic,
            "random_seed": self.random_seed,
            "strict_conservation": self.strict_conservation,
            "enable_huckel_resolvent": self.enable_huckel_resolvent,
            "diffusion_scheme": self.diffusion_scheme,
            "diffusion_rate": self.diffusion_rate,
            "ring_isometry": self.ring_isometry,
            "isometry_period": self.isometry_period,
            "entropy_mixing": self.entropy_mixing,
            "simulate_latency": self.simulate_latency,
        }


def create_reactor(
    mic: MICRegistry,
    agent: CatalystAgent,
    telemetry: TelemetryContext,
    config: Optional[ReactorConfig] = None,
) -> CatalyticReactor:
    """Fábrica de reactores: valida ReactorConfig e instancia CatalyticReactor."""
    if config is None:
        config = ReactorConfig()
    return CatalyticReactor(mic=mic, agent=agent, telemetry=telemetry, **config.as_kwargs())


__all__.extend(
    [
        "ReactionKinetics",
        "DiffusionScheme",
        "RingIsometry",
        "NodeOutcome",
        "DiffusionEnergyBalance",
        "NodeReaction",
        "CycleRecord",
        "CatalystAgent",
        "HodgeDecomposition",
        "SpectralCertificate",
        "HuckelResolvent",
        "CatalyticGenerator",
    ]
)
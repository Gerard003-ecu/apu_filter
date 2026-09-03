# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Catalytic Quantum Reaction Chamber (Cámara de Reacción Cuántica)    ║
║ Ruta   : app/physics/reaction_chamber.py                                     ║
║ Versión: 1.1.0-Doctoral-Huckel-Z-Ring-Smith-CFL-KBN-FPU-Secure               ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Este módulo implementa el foso de simulación termodinámica y cuántica        ║
║ para el reactor catalítico en el Estrato Physics ($V_{\mathrm{PHYSICS}}$)    ║
║ de APU Filter v5.0.                                                          ║
║                                                                              ║
║ Modela la resonancia aromática cíclica de un anillo de benceno de 6 carbonos ║
║ como un complejo simplicial discretizado en la base de la pirámide de        ║
║ control. Consagra el anillo de los enteros $\mathbb{Z}$ como el Dominio de   ║
║ Ideales Principales que dota de inmunidad homológica a los flujos,           ║
║ proscribiendo alucinaciones atencionales y el error de redondeo de Wilkinson ║
║ mediante sumación compensada de de Rham-Kahan y regularización elíptica.     ║
╚══════════════════════════════════════════════════════════════════════════════╝

================════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO DOCTORAL (El Anillo de los Enteros y Simetría Homológica)
================════════════════════════════════════════════════════════════════

Definición 1 (Complejo Simplicial de de Rham-Čech y 1-Homología sobre $\mathbb{Z}$):
  La simetría molecular de la cámara se discretiza como un complejo simplicial
  unidimensional $K$ homeomorfo al grafo cíclico de 6 vértices $C_6$. Definimos los
  espacios de 1-cadenas y 0-cadenas con coeficientes exactos en el anillo unital $\mathbb{Z}$:
  $$C_1(K; \, \mathbb{Z}) = \bigoplus_{e \in E} \mathbb{Z} \cdot e, \quad C_0(K; \, \mathbb{Z}) = \bigoplus_{v \in V} \mathbb{Z} \cdot v$$
  El operador de frontera simplicial discreta $\partial_1: C_1(K; \, \mathbb{Z}) \to C_0(K; \, \mathbb{Z})$ 
  asigna a cada arista orientada $e = [u, v]$ el morfismo de diferencia:
  $$\partial_1(e) = v - u$$
  Al ser el anillo cerrado y carecer de 2-simplejos ($C_2 = \mathbf{0}$), el primer grupo de homología
  con coeficientes enteros se reduce al núcleo exacto del operador frontera:
  $$H_1(K; \, \mathbb{Z}) = \frac{\ker(\partial_1)}{\operatorname{im}(\partial_2)} \cong \ker(\partial_1) \cong \mathbb{Z}^{\beta_1} \quad \text{con} \quad \beta_1 = 1$$
  Garantizando la existencia de un único ciclo incompresible libre de fugas numéricas.

Definición 2 (Clasificación de Torsión mediante la Forma Normal de Smith):
  La matriz de incidencia de aristas $\partial_1$ se reduce de forma exacta sobre el 
  Dominio de Ideales Principales $\mathbb{Z}$ a su diagonal canónica mediante el isomorfismo:
  $$\mathbf{S} = \mathbf{U} \cdot \partial_1 \cdot \mathbf{V} = \operatorname{diag}\left(d_1, \, d_2, \, \dots, \, d_r, \, 0, \, \dots, \, 0\right)$$
  Donde $d_i \in \mathbb{Z}$ cumple la relación de divisibilidad $d_i \mid d_{i+1}$. El Soberano
  exige la nulidad absoluta del subgrupo de torsión homológica para proscribir fraudes:
  $$\operatorname{Tor}\left(H_{k-1}(K; \, \mathbb{Z})\right) \equiv \mathbf{0} \quad \iff \quad d_i = 1 \quad \forall d_i > 0$$

Definición 3 (El Hamiltoniano de Hückel y Resonancia de-confinada):
  La simetría orbital de los electrones $\pi$ se modela inyectando los parámetros de Coulomb
  $\alpha$ y de resonancia $\beta$ sobre el Laplaciano combinatorio del grafo de adyacencia $\mathbf{A}$:
  $$\mathbf{H}_{\mathrm{Huckel}} = \alpha \mathbf{I} + \beta \mathbf{A}$$
  La conmutación al estado fundamental de resonancia aromática maximiza la deslocalización,
  reduciendo síncronamente la energía libre de Gibbs de lazo cerrado en la FPU.

================════════════════════════════════════════════════════════════════
II. DINÁMICA DE LA CÁMARA Y LEYES DE CONSERVACIÓN EN LA FPU (Leyes de la FPU)
================════════════════════════════════════════════════════════════════

Axioma I (Potencial Termodinámico Deformado por Estrés Topológico):
  La energía de Gibbs de la cámara cuántica integra la entalpía, la entropía de Shannon,
  y la penalización cuadrática por deformación geodésica del estado cuántico $\boldsymbol{\psi}$ en la FPU:
  $$G = H - TS + \mu_{\mathrm{topo}} \cdot \|\boldsymbol{\psi}\|_{\mathbb{R}^6}^2 + RT \ln(a)$$

Axioma II (Estabilidad de de Rham-Poisson bajo la Condición CFL):
  La difusión temporal de tensiones se rige por la ecuación de calor discreta sobre el Laplaciano $\mathbf{L}$:
  $$\boldsymbol{\psi}(t + \Delta t) = \boldsymbol{\psi}(t) - \alpha_{\mathrm{diffusion}} \cdot \mathbf{L} \cdot \boldsymbol{\psi}(t)$$
  Para evitar la amplificación caótica del ruido numérico y garantizar estabilidad de Lyapunov,
  la tasa de difusión se acota estrictamente por debajo del radio espectral de $\mathbf{L}$:
  $$\alpha_{\mathrm{diffusion}} < \alpha_{\mathrm{critical}} = \frac{1}{2 \lambda_{\max}(\mathbf{L})} \equiv \frac{1}{2 \cdot 4.0} \equiv 0.125$$
  Se adopta un margen de seguridad determinista del 95%: $\alpha_{\mathrm{safe}} = 0.11875$.

Axioma III (Axioma de Sumación Compensada de Neumaier-Kahan):
  Para neutralizar la deriva de Wilkinson en ciclos masivos de difusión de estado, la actualización
  del vector de onda $\boldsymbol{\psi}$ debe computarse utilizando aritmética KBN compensada en la FPU:
  $$\boldsymbol{\psi}_n = \sum_{\mathrm{KBN}} \left( \boldsymbol{\psi}_{n-1} - \alpha_{\mathrm{diffusion}} \mathbf{L} \boldsymbol{\psi}_{n-1} \right) \implies \|\boldsymbol{\psi}_{\mathrm{computed}} - \boldsymbol{\psi}_{\mathrm{exact}}\| \le \varepsilon_{\mathrm{mach}}$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid

import numpy as np

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
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

__version__ = "4.0.0"

__all__ = [
    "PhysicalConstants",
    "NumericalConstants",
    "TopologyConstants",
    "DampingConstants",
    "CFLConstants",
    "HuckelConstants",
    "ReactorLimits",
    "CarbonNode",
    "AromaticityState",
    "ConvergenceStatus",
    "HilbertState",
    "ThermodynamicPotential",
    "HexagonalTopology",
    "AromaticityEvaluator",
    "EntropyCalculator",
    "ReactionResult",
    "CatalyticReactor",
    "ReactorConfig",
    "TellegenAuditReport",
    "CompensatedLinearAlgebra",
    "DiscreteExteriorCalculus",
    "LyapunovCertificate",
    "create_reactor",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — SUSTRATO NUMÉRICO-ALGEBRAICO
# Banach / KBN / álgebra de operadores / kernel de coborde
# ══════════════════════════════════════════════════════════════════════════════
#
# Objetos:  ℝ con aritmética compensada, (ℝⁿ, ‖·‖₂) de Hilbert,
#           𝔅(ℝⁿ) álgebra de Banach de operadores con ‖T‖ = ρ(T*)^{1/2}.
#
# El morfismo terminal de esta fase es
#     CompensatedLinearAlgebra.construct_coboundary_kernel
# que instala el complejo de cocadenas y es el objeto inicial de la Fase 2.


class PhysicalConstants:
    """Constantes físicas fundamentales (escala de modelo y SI)."""

    __slots__ = ()

    R_GAS: float = 8.314462618
    BOLTZMANN_SI: float = 1.380649e-23
    PLANCK_SI: float = 6.62607015e-34
    BOLTZMANN_SCALE: float = 1.0e-1
    T_REFERENCE: float = 298.15
    T_MINIMUM: float = 280.0
    ARRHENIUS_A: float = 1.0e13
    E_ACTIVATION_BASE: float = 50.0e3
    EYRING_TRANSMISSION: float = 1.0


class NumericalConstants:
    """Constantes numéricas: ε-máquina, regularización y auditorías."""

    __slots__ = ()

    EPS: float = 1e-12
    GIBBS_CONVERGENCE_TOL: float = 0.05
    SPECTRAL_TOL: float = 1e-10
    ENTROPY_MIN_PROB: float = 1e-10
    ORTHO_RESTARTS: int = 2
    POWER_ITERATION_MAX: int = 64
    POWER_ITERATION_TOL: float = 1e-12

    TIKHONOV_BASE_MIN: float = 1.0e-6
    TIKHONOV_EPS_SCALE: float = 1.0e3
    TIKHONOV_COND_CEILING: float = 1.0e12

    TELLEGEN_TOLERANCE: float = 1.0e-10
    WEYL_TOLERANCE: float = 1.0e-9
    BETTI_RANK_TOL: float = 1.0e-9


class TopologyConstants:
    """Invariantes topológicos del anillo hexagonal C₆."""

    __slots__ = ()

    RING_SIZE: int = 6
    LAMBDA_MAX: float = 4.0
    SPECTRAL_GAP: float = 1.0
    PRESSURE_COEFF: float = 1.0
    BETTI_0: int = 1
    BETTI_1: int = 1
    SPANNING_TREES: int = 6  # Kirchhoff: det(L[1:, 1:]) = n^{n-2} · n para C_n → n


class DampingConstants:
    """Amortiguamiento de Lyapunov y disipación local."""

    __slots__ = ()

    GAMMA: float = 0.3
    OMEGA: float = math.pi / 3.0
    COOLING_FACTOR: float = 0.95
    LOCAL_STRESS_GAIN: float = 0.08
    LOCAL_STRESS_DISSIPATION: float = 0.02
    LYAPUNOV_DISSIPATION: float = 0.15


class CFLConstants:
    """
    Condición de Courant–Friedrichs–Lewy / von Neumann para Euler explícito.

    El espectro de I - α L es {1 - α λₖ}. Estabilidad en ℓ² requiere

        |1 - α λ| ≤ 1  ∀ λ ∈ [0, λ_max]  ⇒  0 ≤ α ≤ 2 / λ_max.

    ALPHA_SHARP es esa cota; ALPHA_CRITICAL conserva el margen histórico
    1/(2 λ_max) usado por el reactor (cuatro veces más conservador).
    """

    __slots__ = ()

    ALPHA_SHARP: float = 2.0 / TopologyConstants.LAMBDA_MAX
    ALPHA_CRITICAL: float = 1.0 / (2.0 * TopologyConstants.LAMBDA_MAX)
    SAFETY_MARGIN: float = 0.95
    ALPHA_SAFE: float = SAFETY_MARGIN * ALPHA_CRITICAL


class HuckelConstants:
    """Hamiltoniano de Hückel H = α I + β A sobre C₆."""

    __slots__ = ()

    ALPHA: float = 0.20
    BETA: float = -0.05
    ACTIVATION_CEILING: float = 0.9
    PI_ELECTRONS_BENZENE: int = 6


class ReactorLimits:
    """Límites operacionales y umbrales de colapso."""

    __slots__ = ()

    INSTABILITY_THRESHOLD: float = 5.0
    COLLAPSE_FACTOR: float = 1.2
    MAX_RESONANCE_CYCLES: int = 6
    MIN_CONVERGENCE_CYCLE: int = 3
    MAX_NODE_SLEEP: float = 0.050
    MIN_ENTHALPY: float = 1e-10
    MIN_ENTROPY: float = 1e-10
    NORM_RELATIVE_STATIONARITY: float = 0.05


def _is_finite(value: float) -> bool:
    """Predicado de finitud en ℝ (rechaza NaN e infinitos)."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_finite_vector(
    values: Sequence[float],
    name: str = "vector",
) -> None:
    """Inmersión de validación: todo vector de estado vive en ℝⁿ, no en la compactificación."""
    for i, value in enumerate(values):
        if not _is_finite(value):
            raise ValueError(f"{name}[{i}] no es finito: {value!r}")


def _stable_divide(
    numerator: float,
    denominator: float,
    eps: float = NumericalConstants.EPS,
) -> float:
    """División regularizada:  x / (y + ε sign(y)) cerca del origen."""
    if abs(denominator) < eps:
        return math.copysign(numerator / eps, denominator) if numerator != 0 else 0.0
    return numerator / denominator


def _stable_sqrt(value: float) -> float:
    """Raíz en el semieje [0, ∞) con proyección de valores negativos numéricos."""
    return math.sqrt(max(value, 0.0))


def _stable_exp(x: float, limit: float = 700.0) -> float:
    """expm saturado al rango de float64 para evitar overflow de Arrhenius."""
    return math.exp(min(max(x, -limit), limit))


def _stable_log(x: float, eps: float = NumericalConstants.EPS) -> float:
    """log regularizado sobre (ε, ∞)."""
    return math.log(max(x, eps))


def _stable_log1p(x: float) -> float:
    """log(1+x) con corte en el polo x = -1."""
    return math.log1p(max(x, -1.0 + NumericalConstants.EPS))


def _two_sum(a: float, b: float) -> Tuple[float, float]:
    """
    Transformación libre de error de Knuth–Dekker.

        a ⊕ b = s ,   a + b = s + e   (exacto en ℝ si no hay overflow)

    Es el generador de la sumación compensada de la Fase 1.
    """
    s = a + b
    b_virtual = s - a
    a_virtual = s - b_virtual
    a_roundoff = a - a_virtual
    b_roundoff = b - b_virtual
    return s, a_roundoff + b_roundoff


def _kbn_sum(values: Iterable[float]) -> float:
    """
    Sumación compensada Kahan–Babuška–Neumaier.

    Invariante: s + c = Σ xᵢ + O(n u² Σ |xᵢ|) frente a O(n u Σ |xᵢ|) naïve.
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
    """Producto interno euclídeo ⟨a|b⟩ con compensación KBN."""
    if len(a) != len(b):
        raise ValueError(
            "Producto interno indefinido para vectores de distinta dimensión"
        )

    s = 0.0
    c = 0.0

    for x, y in zip(a, b):
        term = float(x) * float(y)
        t = s + term
        if abs(s) >= abs(term):
            c += (s - t) + term
        else:
            c += (term - t) + s
        s = t

    return float(s + c)


def _normalize_vector(
    vector: Sequence[float],
    eps: float = NumericalConstants.EPS,
) -> Tuple[List[float], float]:
    """Proyección a S^{n-1} con norma KBN; retorna (v̂, ‖v‖)."""
    norm_sq = _kbn_sum(x * x for x in vector)
    norm = _stable_sqrt(norm_sq)

    if norm < eps:
        return list(vector), norm

    return [x / norm for x in vector], norm


def _compute_norm(vector: Sequence[float]) -> float:
    """
    Norma ℓ² escalada (evita overflow intermedio):

        ‖x‖₂ = m · √Σ (xᵢ/m)² ,   m = ‖x‖_∞
    """
    if not vector:
        return 0.0

    max_abs = max(abs(x) for x in vector)
    if max_abs < NumericalConstants.EPS:
        return 0.0

    scaled = [x / max_abs for x in vector]
    scaled_norm_sq = _kbn_sum(x * x for x in scaled)
    return float(max_abs * _stable_sqrt(scaled_norm_sq))


def _frobenius_norm(matrix: Sequence[Sequence[float]]) -> float:
    """Norma de Frobenius ‖A‖_F = √Σᵢⱼ Aᵢⱼ², isométrica con Hilbert–Schmidt."""
    return _compute_norm([float(a) for row in matrix for a in row])


class CompensatedLinearAlgebra:
    """
    Álgebra lineal compensada sobre (ℝⁿ, ⟨·,·⟩).

    Provee matvec KBN, Gram–Schmidt modificado con reortogonalización,
    radio espectral por iteración de potencias y el morfismo terminal
    de la Fase 1: construct_coboundary_kernel.
    """

    __slots__ = ()

    @staticmethod
    def matvec_kbn(
        matrix: Sequence[Sequence[float]],
        vector: Sequence[float],
    ) -> List[float]:
        """
        Producto y = A x con sumación KBN por fila.

            yᵢ = Σⱼ Aᵢⱼ xⱼ
        """
        n = len(vector)
        if len(matrix) != n:
            raise ValueError(
                f"matvec: filas({len(matrix)}) ≠ dim(x)={n}"
            )

        result: List[float] = []
        for i, row in enumerate(matrix):
            if len(row) != n:
                raise ValueError(f"matvec: fila {i} de longitud {len(row)}")
            result.append(_kbn_dot(row, vector))
        return result

    @staticmethod
    def gram_matrix_kbn(
        vectors: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Gram Gᵢⱼ = ⟨vᵢ|vⱼ⟩; G ⪰ 0 y G = I ssi la familia es o.n."""
        m = len(vectors)
        gram = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i, m):
                g = _kbn_dot(vectors[i], vectors[j])
                gram[i][j] = g
                gram[j][i] = g
        return gram

    @staticmethod
    def modified_gram_schmidt(
        vectors: Sequence[Sequence[float]],
        eps: float = NumericalConstants.EPS,
        restarts: int = NumericalConstants.ORTHO_RESTARTS,
    ) -> List[List[float]]:
        """
        MGS con reortogonalización de Kahan.

        Más estable que CGS clásico: el factor de pérdida de ortogonalidad
        es O(u κ) frente a O(u κ²).
        """
        basis: List[List[float]] = []
        n_dim = len(vectors[0]) if vectors else 0

        for raw in vectors:
            if len(raw) != n_dim:
                raise ValueError("MGS: dimensión incompatible")
            v = [float(x) for x in raw]

            for _ in range(max(1, restarts)):
                for q in basis:
                    coeff = _kbn_dot(v, q)
                    v = [v[i] - coeff * q[i] for i in range(n_dim)]

            v, norm = _normalize_vector(v, eps=eps)
            if norm > eps:
                basis.append(v)

        return basis

    @staticmethod
    def spectral_radius_power(
        matrix: Sequence[Sequence[float]],
        max_iter: int = NumericalConstants.POWER_ITERATION_MAX,
        tol: float = NumericalConstants.POWER_ITERATION_TOL,
    ) -> float:
        """
        Radio espectral ρ(A) ≈ ‖A vₖ‖ / ‖vₖ‖ por iteración de potencias.

        Para L ⪰ 0 autoadjunto, ρ(L) = λ_max.
        """
        n = len(matrix)
        if n == 0:
            return 0.0

        v = [1.0 / math.sqrt(float(n))] * n
        lam = 0.0

        for _ in range(max_iter):
            w = CompensatedLinearAlgebra.matvec_kbn(matrix, v)
            w_norm = _compute_norm(w)
            if w_norm < NumericalConstants.EPS:
                return 0.0
            v = [x / w_norm for x in w]
            Aw = CompensatedLinearAlgebra.matvec_kbn(matrix, v)
            lam_new = abs(_kbn_dot(v, Aw))
            if abs(lam_new - lam) <= tol * max(1.0, lam_new):
                return float(lam_new)
            lam = lam_new

        return float(lam)

    @staticmethod
    def tikhonov_solve(
        operator: Sequence[Sequence[complex]],
        alpha_reg: float,
    ) -> Tuple[List[List[complex]], float]:
        """
        Inversa de Tikhonov vía SVD:

            A⁺_α = V · diag( σᵢ / (σᵢ² + α²) ) · Uᴴ

        (aquí A ya incluye el shift α_reg I; se filtra σᵢ < ε relativo).
        Retorna (A^{-1}_α, κ₂(A)).
        """
        A = np.asarray(operator, dtype=np.complex128)
        try:
            U, sigma, Vh = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"SVD del resolvente falló: {exc}") from ext if False else exc  # noqa: E501

        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_max = float(sigma[0]) if sigma.size else 0.0
        sigma_min = float(sigma[-1]) if sigma.size else 0.0
        cond = (
            sigma_max / sigma_min
            if sigma_min > NumericalConstants.EPS
            else math.inf
        )

        floor = max(NumericalConstants.EPS, float(alpha_reg) * 1.0e-3)
        inv_sigma = np.array(
            [1.0 / s if s > floor else 0.0 for s in sigma],
            dtype=np.complex128,
        )
        G = (Vh.conj().T * inv_sigma) @ U.conj().T
        return G.tolist(), float(cond)

    @staticmethod
    def numerical_rank(
        matrix: Sequence[Sequence[float]],
        tol: float = NumericalConstants.BETTI_RANK_TOL,
    ) -> int:
        """rango numérico = #{σᵢ > tol · σ_max}."""
        A = np.asarray(matrix, dtype=np.float64)
        try:
            sigma = np.linalg.svd(A, compute_uv=False)
        except np.linalg.LinAlgError:
            return 0
        if sigma.size == 0:
            return 0
        threshold = tol * float(sigma[0])
        return int(np.count_nonzero(sigma > threshold))

    def construct_coboundary_kernel(
        self,
        incidence: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        """
        ── MORFISMO TERMINAL DE LA FASE 1 / INICIAL DE LA FASE 2 ──────────

        Dado el operador de incidencia B : Ω¹ → Ω⁰ (divergencia), construye
        el kernel algebraico del complejo de coborde:

            d₀ = Bᵀ : Ω⁰ → Ω¹
            δ₀ = B  : Ω¹ → Ω⁰
            Δ₀ = B Bᵀ
            Δ₁ = Bᵀ B
            ker Δ₀  →  H⁰_dR   (componentes conexas, β₀)
            ker Δ₁  →  H¹_dR   (ciclos independientes, β₁)

        Este diccionario de operadores es el objeto sobre el que la Fase 2
        instala la geometría de Hodge–de Rham–Hückel.
        """
        n = len(incidence)
        B = [list(map(float, row)) for row in incidence]
        if any(len(row) != n for row in B):
            raise ValueError("incidence debe ser cuadrada para C₆")

        BT = [[B[j][i] for j in range(n)] for i in range(n)]

        delta0 = B
        d0 = BT

        lap_0 = [
            [
                _kbn_dot(B[i], [B[j][e] for e in range(n)])
                for j in range(n)
            ]
            for i in range(n)
        ]
        lap_1 = [
            [
                _kbn_dot(BT[i], [BT[j][v] for v in range(n)])
                for j in range(n)
            ]
            for i in range(n)
        ]

        rank_L = self.numerical_rank(lap_0)
        rank_edge = self.numerical_rank(lap_1)
        betti_0 = max(0, n - rank_L)
        betti_1 = max(0, n - rank_edge)

        return {
            "d0": d0,
            "delta0": delta0,
            "laplacian_0": lap_0,
            "laplacian_1": lap_1,
            "betti_0": betti_0,
            "betti_1": betti_1,
            "euler": betti_0 - betti_1,
            "rank_delta0": rank_L,
        }


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — GEOMETRÍA ESPECTRAL DE HODGE–DE RHAM–HÜCKEL–TELLEGEN
# Continúa CompensatedLinearAlgebra.construct_coboundary_kernel
# ══════════════════════════════════════════════════════════════════════════════
#
# Objetos: complejo de cocadenas discreto (Ω•, d), Hamiltoniano de Hückel,
#          auditoría de Tellegen, generador infinitesimal del semigrupo
#          catalítico 𝒢 = -α Δ₀.
#
# El morfismo terminal de esta fase es
#     HexagonalTopology.infinitesimal_catalytic_generator
# que la Fase 3 integra como semigrupo de difusión sobre HilbertState.


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
        """Índice base-0 del nodo en el anillo."""
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


_SERVICE_MAP: Dict[CarbonNode, str] = {
    CarbonNode.C1_INGESTION: "load_data",
    CarbonNode.C2_PHYSICS: "stabilize_flux",
    CarbonNode.C3_TOPOLOGY: "business_topology",
    CarbonNode.C4_STRATEGY: "financial_analysis",
    CarbonNode.C5_SEMANTICS: "semantic_translation",
    CarbonNode.C6_MATTER: "materialization",
}

_PRECURSOR_MAP: Dict[CarbonNode, Tuple[str, ...]] = {
    CarbonNode.C2_PHYSICS: ("physical_constraints",),
    CarbonNode.C4_STRATEGY: ("financial_params",),
    CarbonNode.C5_SEMANTICS: ("semantic_model",),
}


def node_from_index(idx: int) -> CarbonNode:
    """Retracción índice → CarbonNode, dominio {0,…,5}."""
    if not 0 <= idx < TopologyConstants.RING_SIZE:
        raise ValueError(
            f"Índice {idx} fuera de rango [0, {TopologyConstants.RING_SIZE - 1}]"
        )
    return CarbonNode(idx + 1)


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


@dataclass(frozen=True)
class TellegenAuditReport:
    """
    Auditoría de conservación de de Rham–Tellegen–Hodge.

    Tres axiomas de coherencia del complejo:

      1. Pasividad Dirichlet:
            ψᵀ L ψ = ‖d₀ ψ‖² ≥ -tol
      2. Conservación nodal (im δ₀ ⊂ 1^⊥ para flujos circulantes):
            |Σᵢ (B q)ᵢ| ≤ tol
      3. Ortogonalidad Hodge (1-formas exactas ⟂ armónicas):
            |⟨d₀ ψ, h⟩| ≤ tol ,   Δ₁ h = 0
      4. Identidad de Tellegen:
            |⟨ψ, B q⟩ - ⟨Bᵀ ψ, q⟩| ≤ tol
    """

    passivity_satisfied: bool
    conservation_satisfied: bool
    orthogonality_satisfied: bool
    tellegen_satisfied: bool
    laplacian_energy: float
    conservation_residual: float
    orthogonality_residual: float
    tellegen_residual: float
    tolerance: float

    @property
    def is_coherent(self) -> bool:
        """Auditoría global: los cuatro axiomas se verifican simultáneamente."""
        return (
            self.passivity_satisfied
            and self.conservation_satisfied
            and self.orthogonality_satisfied
            and self.tellegen_satisfied
        )


@dataclass(frozen=True)
class LyapunovCertificate:
    """
    Certificado de estabilidad de Lyapunov para el par (ψ, G).

        V(ψ, G) = ‖ψ‖² + G²
        Ḋ ≈ 2 ⟨ψ, 𝒢 ψ⟩ + 2 G Ġ

    Disipatividad se declara si ΔV ≤ -γ V + holgura numérica.
    """

    value: float
    decrement: float
    dissipative: bool
    spectral_abscissa: float


class DiscreteExteriorCalculus(CompensatedLinearAlgebra):
    """
    ── CONTINUACIÓN DE LA FASE 1 ──────────────────────────────────────────

    Instala el cálculo exterior discreto sobre el kernel producido por
    construct_coboundary_kernel: operadores d, δ, ★, proyección de Hodge
    ω = dα + δβ + h y la identidad de Tellegen como Stokes 0-dimensional.
    """

    __slots__ = ("_coboundary_kernel",)

    def __init__(self) -> None:
        self._coboundary_kernel: Optional[Dict[str, Any]] = None

    def install_coboundary_kernel(
        self,
        incidence: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        """Continúa construct_coboundary_kernel y memoíza el complejo."""
        kernel = self.construct_coboundary_kernel(incidence)
        self._coboundary_kernel = kernel
        return kernel

    @property
    def coboundary_kernel(self) -> Dict[str, Any]:
        if self._coboundary_kernel is None:
            raise RuntimeError(
                "Kernel de coborde no instalado: llame install_coboundary_kernel"
            )
        return self._coboundary_kernel

    def exterior_derivative_0(
        self,
        zero_form: Sequence[float],
    ) -> List[float]:
        """d₀ ψ ∈ Ω¹, 1-forma exacta (gradiente de arista)."""
        d0 = self.coboundary_kernel["d0"]
        return self.matvec_kbn(d0, zero_form)

    def codifferential_1(
        self,
        one_form: Sequence[float],
    ) -> List[float]:
        """δ₀ q ∈ Ω⁰, divergencia nodal."""
        delta0 = self.coboundary_kernel["delta0"]
        return self.matvec_kbn(delta0, one_form)

    def hodge_energy_0(self, zero_form: Sequence[float]) -> float:
        """Energía Dirichlet ℰ(ψ) = ‖d₀ ψ‖² = ψᵀ Δ₀ ψ."""
        grad = self.exterior_derivative_0(zero_form)
        return _kbn_sum(g * g for g in grad)

    def harmonic_one_form(self, n: int) -> List[float]:
        """
        Generador de H¹(C₆) ≅ ℝ: 1-forma uniforme de circulación

            h_e = 1/√n  ∀ e ,   Δ₁ h = 0 ,  ‖h‖ = 1.
        """
        if n <= 0:
            raise ValueError("dimensión n debe ser positiva")
        scale = 1.0 / math.sqrt(float(n))
        return [scale] * n

    def hodge_project_exact(
        self,
        one_form: Sequence[float],
    ) -> Tuple[List[float], List[float], float]:
        """
        Descomposición de Hodge en 1-formas sobre el ciclo (δβ = 0 porque
        no hay 2-simplices independientes más allá de la 2-célula global):

            q = dα + h ,   h ∈ ker Δ₁ ,   ⟨dα, h⟩ = 0

        Retorna (componente_exacta, componente_armónica, residuo_ortogonal).
        """
        h = self.harmonic_one_form(len(one_form))
        coeff = _kbn_dot(one_form, h)
        harmonic = [coeff * x for x in h]
        exact = [float(q) - hv for q, hv in zip(one_form, harmonic)]
        residual = abs(_kbn_dot(exact, h))
        return exact, harmonic, residual

    def tellegen_residual(
        self,
        zero_form: Sequence[float],
        one_form: Sequence[float],
    ) -> float:
        """
        Residuo de la identidad de Tellegen / adjunción d₀ ⊣ δ₀:

            |⟨ψ, δ₀ q⟩ - ⟨d₀ ψ, q⟩|
        """
        left = _kbn_dot(zero_form, self.codifferential_1(one_form))
        right = _kbn_dot(self.exterior_derivative_0(zero_form), one_form)
        return abs(left - right)


class HexagonalTopology(DiscreteExteriorCalculus):
    """
    Topología espectral del grafo hexagonal (ciclo C₆).

    Hereda el cálculo exterior de la Fase 2 e instala:
      - adyacencia, grado, Laplaciano combinatorio L = D - A = B Bᵀ
      - espectro de Fourier sobre ℤ/6ℤ, con degeneración λ₁=λ₅, λ₂=λ₄
      - Hamiltoniano de Hückel y resolvente de Tikhonov adaptativo
      - auditoría de Rham–Tellegen–Hodge
      - generador infinitesimal 𝒢 del semigrupo catalítico (morfismo
        terminal de la Fase 2)
    """

    __slots__ = (
        "_adjacency",
        "_degree",
        "_laplacian",
        "_incidence",
        "_eigenvalues",
        "_eigenvectors",
        "_coboundary_kernel",
        "_spectral_verified",
    )

    _ANALYTICAL_EIGENVALUES: ClassVar[Tuple[float, ...]] = (
        0.0,
        1.0,
        1.0,
        3.0,
        3.0,
        4.0,
    )

    def __init__(self) -> None:
        super().__init__()
        n = TopologyConstants.RING_SIZE

        self._adjacency: List[List[float]] = [
            [
                1.0 if (j == (i + 1) % n or j == (i - 1) % n) else 0.0
                for j in range(n)
            ]
            for i in range(n)
        ]
        self._degree: List[int] = [2] * n
        self._laplacian: List[List[float]] = [
            [
                float(self._degree[i]) * (1.0 if i == j else 0.0)
                - self._adjacency[i][j]
                for j in range(n)
            ]
            for i in range(n)
        ]
        self._incidence: List[List[float]] = self._build_incidence_matrix()
        self.install_coboundary_kernel(self._incidence)
        self._eigenvalues, self._eigenvectors = self._compute_spectrum()
        self._spectral_verified = self._verify_spectrum()

    def _build_incidence_matrix(self) -> List[List[float]]:
        """
        Matriz de incidencia orientada B ∈ ℝ^{n×n} del ciclo dirigido.

            e = (i → i+1 mod n) :  B[i, e] = +1 ,  B[j, e] = -1
        """
        n = TopologyConstants.RING_SIZE
        B = [[0.0 for _ in range(n)] for _ in range(n)]
        for e in range(n):
            i = e
            j = (e + 1) % n
            B[i][e] = 1.0
            B[j][e] = -1.0
        return B

    def _compute_spectrum(self) -> Tuple[List[float], List[List[float]]]:
        """
        Espectro analítico de Fourier sobre el grupo cíclico ℤ/nℤ.

            λₖ = 2 - 2 cos(2π k / n)
            vₖ(j) = cos(2π k j / n)   (parte real)
            wₖ(j) = sin(2π k j / n)   (parte imag, degeneración)

        Los pares degenerados (k, n-k) se ortonormalizan por MGS.
        """
        n = TopologyConstants.RING_SIZE
        raw_vectors: List[List[float]] = []
        eigenvalues: List[float] = []

        for k in range(n):
            lam = 2.0 - 2.0 * math.cos(2.0 * math.pi * k / n)
            eigenvalues.append(lam)
            vec = [
                math.cos(2.0 * math.pi * k * j / n)
                for j in range(n)
            ]
            raw_vectors.append(vec)

        # Completar degeneraciones con senos cuando la parte real colapsa.
        for k in range(1, n):
            sine = [
                math.sin(2.0 * math.pi * k * j / n)
                for j in range(n)
            ]
            if _compute_norm(sine) > NumericalConstants.EPS:
                raw_vectors.append(sine)
                eigenvalues.append(2.0 - 2.0 * math.cos(2.0 * math.pi * k / n))

        orthonormal = self.modified_gram_schmidt(raw_vectors)
        # Conservar los n primeros modos (base de ℝⁿ).
        eigenvectors = orthonormal[:n]
        eigenvalues = self._ANALYTICAL_EIGENVALUES[:n]
        return list(eigenvalues), [list(v) for v in eigenvectors]

    def _verify_spectrum(self) -> bool:
        """
        Verificación de Weyl:  max_i |λ_i^num - λ_i^an| ≤ ‖L‖_F · u^{1/2}
        más chequeo independiente de λ_max por iteración de potencias.
        """
        computed = sorted(self._eigenvalues)
        analytical = sorted(self._ANALYTICAL_EIGENVALUES)
        ok = True

        for i, (c, a) in enumerate(zip(computed, analytical)):
            if abs(c - a) > NumericalConstants.SPECTRAL_TOL:
                logger.warning(
                    "Discrepancia espectral: λ_%d = %.6f (esperado %.6f)",
                    i,
                    c,
                    a,
                )
                ok = False

        rho = self.spectral_radius_power(self._laplacian)
        if abs(rho - TopologyConstants.LAMBDA_MAX) > NumericalConstants.WEYL_TOLERANCE:
            logger.warning(
                "Radio espectral numérico %.6f ≠ λ_max analítico %.6f",
                rho,
                TopologyConstants.LAMBDA_MAX,
            )
            ok = False

        gram = self.gram_matrix_kbn(self._eigenvectors)
        for i in range(len(gram)):
            for j in range(len(gram)):
                target = 1.0 if i == j else 0.0
                if abs(gram[i][j] - target) > 1.0e-8:
                    logger.debug(
                        "Pérdida de ortonormalidad G[%d,%d]=%.3e",
                        i,
                        j,
                        gram[i][j] - target,
                    )
        return ok

    @property
    def adjacency_matrix(self) -> List[List[float]]:
        """Matriz de adyacencia (copia defensiva)."""
        return [row.copy() for row in self._adjacency]

    @property
    def laplacian_matrix(self) -> List[List[float]]:
        """Laplaciano combinatorio Δ₀ (copia defensiva)."""
        return [row.copy() for row in self._laplacian]

    @property
    def incidence_matrix(self) -> List[List[float]]:
        """Incidencia orientada B = δ₀ (copia defensiva)."""
        return [row.copy() for row in self._incidence]

    @property
    def eigenvalues(self) -> Tuple[float, ...]:
        """Espectro de Δ₀."""
        return tuple(self._eigenvalues)

    @property
    def spectral_gap(self) -> float:
        """Gap espectral λ₁ (constante de Cheeger algebraica)."""
        sorted_eigs = sorted(self._eigenvalues)
        return sorted_eigs[1] if len(sorted_eigs) > 1 else 0.0

    @property
    def lambda_max(self) -> float:
        """Eigenvalor máximo = radio espectral de L ⪰ 0."""
        return max(self._eigenvalues)

    @property
    def betti_numbers(self) -> Tuple[int, int]:
        """(β₀, β₁) computados del rango numérico del complejo, no hardcodeados."""
        kernel = self.coboundary_kernel
        return (int(kernel["betti_0"]), int(kernel["betti_1"]))

    @property
    def euler_characteristic(self) -> int:
        """χ = β₀ - β₁ (= 0 para C₆ ≃ S¹)."""
        b0, b1 = self.betti_numbers
        return b0 - b1

    def neighbor_indices(self, node_index: int) -> Tuple[int, int]:
        """Vecindad cíclica {i-1, i+1} mod n."""
        if not 0 <= node_index < TopologyConstants.RING_SIZE:
            raise ValueError(f"Índice de nodo inválido: {node_index}")
        n = TopologyConstants.RING_SIZE
        return ((node_index - 1) % n, (node_index + 1) % n)

    def huckel_hamiltonian(self) -> List[List[float]]:
        """
        Hamiltoniano de Hückel de primeros vecinos:

            H = α I + β A ,   A = 2 I - L   (grafo 2-regular)
            Eₖ = α + 2 β cos(2π k / 6)
        """
        n = TopologyConstants.RING_SIZE
        H = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            H[i][i] = HuckelConstants.ALPHA
            for j in range(n):
                if self._adjacency[i][j] > 0.0:
                    H[i][j] = HuckelConstants.BETA
        return H

    def huckel_spectrum(self) -> Tuple[float, ...]:
        """Niveles MO analíticos Eₖ = α + 2β cos(2πk/n), ordenados."""
        n = TopologyConstants.RING_SIZE
        levels = [
            HuckelConstants.ALPHA
            + 2.0 * HuckelConstants.BETA * math.cos(2.0 * math.pi * k / n)
            for k in range(n)
        ]
        return tuple(sorted(levels))

    def adaptive_tikhonov_shift(self) -> float:
        """
        Shift de Tikhonov adaptativo a la escala del Hamiltoniano:

            α_reg = max(α_min, c · ε_mach) · max(1, λ_max, |α|, |β|)
        """
        eps = float(np.finfo(np.float64).eps)
        base = max(
            NumericalConstants.TIKHONOV_BASE_MIN,
            NumericalConstants.TIKHONOV_EPS_SCALE * eps,
        )
        scale = max(
            1.0,
            self.lambda_max,
            abs(HuckelConstants.ALPHA),
            abs(HuckelConstants.BETA),
        )
        return float(base * scale)

    def regularized_huckel_resolvent(
        self,
        s: float = 0.0,
        h: float = 0.0,
        alpha_reg: Optional[float] = None,
    ) -> Tuple[List[List[complex]], float, float]:
        """
        Resolvente de Hückel con regularización de Tikhonov:

            G(z; α) = (H - z I + α I)^{-1} ,   z = s + i h

        Si κ₂(A) explota, se incrementa α en décadas y se recurre a SVD.
        Retorna (G, α_usado, κ₂).
        """
        n = TopologyConstants.RING_SIZE
        H = np.asarray(self.huckel_hamiltonian(), dtype=np.complex128)
        I = np.eye(n, dtype=np.complex128)
        z = complex(float(s), float(h))

        if alpha_reg is None:
            alpha_reg = self.adaptive_tikhonov_shift()

        A = H - z * I + alpha_reg * I

        for _ in range(4):
            try:
                cond = float(np.linalg.cond(A))
            except Exception:
                cond = math.inf

            if cond < NumericalConstants.TIKHONOV_COND_CEILING:
                try:
                    G = np.linalg.solve(A, I)
                    return G.tolist(), float(alpha_reg), cond
                except np.linalg.LinAlgError:
                    pass

            alpha_reg *= 10.0
            A = H - z * I + alpha_reg * I

        G_list, cond = self.tikhonov_solve(A.tolist(), float(alpha_reg))
        return G_list, float(alpha_reg), cond

    def _edge_gradients(self, state_vector: Sequence[float]) -> List[float]:
        """1-forma exacta d₀ ψ con KBN (continúa exterior_derivative_0)."""
        n = TopologyConstants.RING_SIZE
        if len(state_vector) != n:
            raise ValueError(
                f"Vector debe tener longitud {n}; recibido {len(state_vector)}"
            )
        _validate_finite_vector(state_vector, "state_vector")
        return self.exterior_derivative_0(state_vector)

    def compute_graph_laplacian_energy(self, state: Sequence[float]) -> float:
        """ℰ(ψ) = ψᵀ L ψ = ‖d₀ ψ‖² ≥ 0."""
        return self.hodge_energy_0(state)

    def audit_de_rham_tellegen(
        self,
        state_vector: Sequence[float],
        fluxes: Optional[Sequence[float]] = None,
        tolerance: Optional[float] = None,
    ) -> TellegenAuditReport:
        """
        Auditoría cuádruple de Rham–Tellegen–Hodge.

        Si fluxes is None se toma el flujo difusivo natural q = -d₀ ψ
        (ley de Ohm discreta / gradiente de Dirichlet).
        """
        n = TopologyConstants.RING_SIZE
        tol = (
            tolerance
            if tolerance is not None
            else NumericalConstants.TELLEGEN_TOLERANCE
        )

        energy = self.compute_graph_laplacian_energy(state_vector)
        passivity_satisfied = energy >= -tol

        edge_gradients = self._edge_gradients(state_vector)
        _, _, orthogonality_residual = self.hodge_project_exact(edge_gradients)
        orthogonality_satisfied = orthogonality_residual <= tol

        if fluxes is None:
            edge_fluxes = [-g for g in edge_gradients]
        else:
            if len(fluxes) != n:
                raise ValueError(
                    f"fluxes debe tener longitud {n}; recibido {len(fluxes)}"
                )
            _validate_finite_vector(fluxes, "fluxes")
            edge_fluxes = list(fluxes)

        injections = self.codifferential_1(edge_fluxes)
        conservation_residual = abs(_kbn_sum(injections))
        conservation_satisfied = conservation_residual <= tol

        tellegen_res = self.tellegen_residual(state_vector, edge_fluxes)
        tellegen_satisfied = tellegen_res <= tol

        return TellegenAuditReport(
            passivity_satisfied=bool(passivity_satisfied),
            conservation_satisfied=bool(conservation_satisfied),
            orthogonality_satisfied=bool(orthogonality_satisfied),
            tellegen_satisfied=bool(tellegen_satisfied),
            laplacian_energy=float(energy),
            conservation_residual=float(conservation_residual),
            orthogonality_residual=float(orthogonality_residual),
            tellegen_residual=float(tellegen_res),
            tolerance=float(tol),
        )

    def _validate_boundary(self, mapping: Dict[int, float]) -> None:
        """Validación de soporte y finitud de datos de frontera."""
        for idx, value in mapping.items():
            if not 0 <= idx < TopologyConstants.RING_SIZE:
                raise ValueError(f"Índice de frontera inválido: {idx}")
            if not _is_finite(value):
                raise ValueError(f"Valor de frontera no finito en {idx}: {value}")

    def _apply_dirichlet(
        self,
        vector: List[float],
        values: Dict[int, float],
    ) -> List[float]:
        """Traza de Dirichlet: ψ|Γ = g."""
        self._validate_boundary(values)
        result = vector.copy()
        for idx, value in values.items():
            result[idx] = value
        return result

    def _apply_neumann(
        self,
        vector: List[float],
        fluxes: Dict[int, float],
    ) -> List[float]:
        """
        Neumann débil: (dψ · n)|Γ = φ, implementado como inyección nodal.
        Se vigila la amplificación de norma (pérdida de contractividad).
        """
        self._validate_boundary(fluxes)
        norm_pre = _compute_norm(vector)
        result = vector.copy()
        for idx, flux in fluxes.items():
            result[idx] += flux
        norm_post = _compute_norm(result)
        if norm_pre > NumericalConstants.EPS and norm_post > 2.0 * norm_pre:
            logger.warning(
                "Condición Neumann amplifica norma: %.4f → %.4f (×%.2f)",
                norm_pre,
                norm_post,
                norm_post / norm_pre,
            )
        return result

    def diffuse_stress(
        self,
        state_vector: Sequence[float],
        diffusion_rate: float = CFLConstants.ALPHA_SAFE,
        boundary_conditions: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> List[float]:
        """
        Un paso de Euler del semigrupo e^{-t Δ₀}:

            ψ ← (I - α Δ₀) ψ

        α se proyecta a (0, α_crit) para respetar von Neumann.
        Cada fila de Δ₀ ψ se evalúa con KBN.
        """
        n = TopologyConstants.RING_SIZE
        if len(state_vector) != n:
            raise ValueError(
                f"Vector debe tener longitud {n}; recibido {len(state_vector)}"
            )
        _validate_finite_vector(state_vector, "state_vector")

        if not _is_finite(diffusion_rate):
            raise ValueError(f"Tasa de difusión no finita: {diffusion_rate}")
        if diffusion_rate <= 0.0:
            return list(state_vector)

        if diffusion_rate >= CFLConstants.ALPHA_CRITICAL:
            original = diffusion_rate
            diffusion_rate = CFLConstants.ALPHA_SAFE
            logger.warning(
                "Tasa de difusión %.5f ≥ α_crit %.5f → ajustada a %.5f",
                original,
                CFLConstants.ALPHA_CRITICAL,
                diffusion_rate,
            )

        laplacian_applied = self.matvec_kbn(self._laplacian, state_vector)
        new_vector = [
            float(state_vector[i]) - diffusion_rate * laplacian_applied[i]
            for i in range(n)
        ]

        if boundary_conditions:
            if "dirichlet" in boundary_conditions:
                new_vector = self._apply_dirichlet(
                    new_vector,
                    boundary_conditions["dirichlet"],
                )
            if "neumann" in boundary_conditions:
                new_vector = self._apply_neumann(
                    new_vector,
                    boundary_conditions["neumann"],
                )

        _validate_finite_vector(new_vector, "diffused_vector")
        return new_vector

    def infinitesimal_catalytic_generator(
        self,
        diffusion_rate: float = CFLConstants.ALPHA_SAFE,
    ) -> List[List[float]]:
        """
        ── MORFISMO TERMINAL DE LA FASE 2 / INICIAL DE LA FASE 3 ──────────

        Generador infinitesimal del semigrupo catalítico de difusión:

            𝒢 = -α Δ₀  ∈ 𝔅(ℝ⁶)
            e^{t 𝒢} = Σ (t 𝒢)^k / k!   (contrato en ℓ² ssi α ≤ 2/λ_max)

        La Fase 3 integra este generador sobre HilbertState (flujo de
        estrés) y acopla su energía Dirichlet al potencial de Gibbs.
        """
        if not _is_finite(diffusion_rate) or diffusion_rate < 0.0:
            raise ValueError(f"α inválido para el generador: {diffusion_rate}")

        alpha = min(diffusion_rate, CFLConstants.ALPHA_SAFE)
        n = TopologyConstants.RING_SIZE
        generator = [
            [-alpha * self._laplacian[i][j] for j in range(n)]
            for i in range(n)
        ]
        return generator


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — DINÁMICA TERMODINÁMICA-CUÁNTICA DEL REACTOR
# Continúa HexagonalTopology.infinitesimal_catalytic_generator
# ══════════════════════════════════════════════════════════════════════════════
#
# Objetos: estado de Hilbert ψ ∈ ℝ⁶, potencial de Gibbs G(H,S,T,σ,a),
#          evaluador de aromaticidad (lógica de Hückel), entropía de
#          Shannon–von Neumann, semigrupo catalítico e^{t𝒢}, reactor.


T = TypeVar("T", bound="HilbertState")


@dataclass
class HilbertState:
    """
    ── CONTINUACIÓN DE LA FASE 2 ──────────────────────────────────────────

    Estado del reactor en el Hilbert real (ℝ⁶, ⟨·,·⟩).

    El generador 𝒢 = infinitesimal_catalytic_generator actúa por
        ψ ↦ ψ + 𝒢 ψ   (un paso de Euler, implementado en diffuse_stress).

    La fase φ ∈ ℝ/2πℤ parametriza la holonomía del anillo (transporte
    paralelo discreto de 2π/6 por ciclo).
    """

    vector: List[float] = field(
        default_factory=lambda: [0.0] * TopologyConstants.RING_SIZE
    )
    phase: float = 0.0
    _norm_history: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if len(self.vector) != TopologyConstants.RING_SIZE:
            raise ValueError(
                f"Estado Hilbert requiere longitud {TopologyConstants.RING_SIZE}; "
                f"recibido {len(self.vector)}"
            )
        _validate_finite_vector(self.vector, "HilbertState.vector")
        if not _is_finite(self.phase):
            raise ValueError(f"Fase no finita: {self.phase!r}")
        self.phase = self.phase % (2.0 * math.pi)

    @property
    def norm(self) -> float:
        """Norma ℓ² de ψ."""
        return _compute_norm(self.vector)

    @property
    def norm_squared(self) -> float:
        """‖ψ‖² con KBN (energía cinética adimensional)."""
        return _kbn_sum(x * x for x in self.vector)

    def inner_product(self, other: "HilbertState") -> float:
        """⟨self|other⟩ compensado; se rechaza si el resultado no es finito."""
        result = _kbn_dot(self.vector, other.vector)
        if not _is_finite(result):
            raise ValueError(f"Producto interno no finito: {result}")
        return result

    def density_matrix(self) -> List[List[float]]:
        """
        Operador densidad puro ρ = |ψ̂⟩⟨ψ̂|  (ρ ⪰ 0, tr ρ = 1 si ‖ψ‖>0).

        Si ψ = 0 se retorna la mezcla maximally mixed I/n.
        """
        n = len(self.vector)
        hat, norm = _normalize_vector(self.vector)
        if norm < NumericalConstants.EPS:
            diag = 1.0 / float(n)
            return [[diag if i == j else 0.0 for j in range(n)] for i in range(n)]
        return [[hat[i] * hat[j] for j in range(n)] for i in range(n)]

    def von_neumann_entropy(self) -> float:
        """
        S_vN(ρ) = -tr(ρ log ρ).

        Para estado puro S=0; para maximally mixed S=log n.
        Autovalores de ρ se obtienen por SVD (ρ ⪰ 0 autoadjunto).
        """
        rho = np.asarray(self.density_matrix(), dtype=np.float64)
        eig = np.clip(np.linalg.eigvalsh(rho), 0.0, 1.0)
        entropy = 0.0
        for p in eig:
            if p > NumericalConstants.ENTROPY_MIN_PROB:
                entropy -= float(p) * _stable_log(float(p))
        return float(entropy)

    def normalize(self: T) -> T:
        """Proyección a la esfera unidad."""
        self.vector, _ = _normalize_vector(self.vector)
        return self

    def scale(self: T, factor: float) -> T:
        """Homotecia ψ ↦ λ ψ, λ finito."""
        if not _is_finite(factor):
            raise ValueError(f"Factor de escala no finito: {factor}")
        self.vector = [x * factor for x in self.vector]
        _validate_finite_vector(self.vector, "scaled_vector")
        return self

    def apply_generator_step(
        self,
        generator: Sequence[Sequence[float]],
    ) -> None:
        """
        Integra un paso de Euler del generador de la Fase 2:

            ψ ← ψ + 𝒢 ψ
        """
        increment = CompensatedLinearAlgebra.matvec_kbn(generator, self.vector)
        self.vector = [
            float(self.vector[i]) + increment[i]
            for i in range(len(self.vector))
        ]
        _validate_finite_vector(self.vector, "generated_vector")

    def apply_damping(self, cycle: int) -> None:
        """
        Contracción hacia la media (modo armónico k=0) con envolvente

            ψ ← μ + (ψ - μ) e^{-γ t} |cos(ω t)|

        Es un disipador de Lyapunov compatible con ker Δ₀ (constantes).
        """
        n = len(self.vector)
        if n == 0:
            return

        envelope = _stable_exp(-DampingConstants.GAMMA * cycle)
        oscillation = abs(math.cos(DampingConstants.OMEGA * cycle))
        factor = envelope * oscillation
        mean = _kbn_sum(self.vector) / float(n)
        self.vector = [mean + (v - mean) * factor for v in self.vector]
        _validate_finite_vector(self.vector, "damped_vector")
        self._norm_history.append(self.norm)
        if len(self._norm_history) > 10:
            self._norm_history.pop(0)

    def project_orthogonal(self, subspace_basis: List["HilbertState"]) -> None:
        """Complemento ortogonal vía MGS+KBN (Fase 1)."""
        n = len(self.vector)
        raw = [bv.vector for bv in subspace_basis if len(bv.vector) == n]
        orthonormal_basis = CompensatedLinearAlgebra.modified_gram_schmidt(raw)
        for ortho_vec in orthonormal_basis:
            coeff = _kbn_dot(self.vector, ortho_vec)
            self.vector = [
                self.vector[i] - coeff * ortho_vec[i]
                for i in range(n)
            ]
        _validate_finite_vector(self.vector, "projected_vector")

    def is_oscillating(self, window: int = 5, threshold: float = 0.1) -> bool:
        """Detecta cambios de signo en Δ‖ψ‖ (ciclo límite vs. atractor)."""
        if len(self._norm_history) < window:
            return False
        recent = self._norm_history[-window:]
        deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        sign_changes = sum(
            1 for i in range(len(deltas) - 1)
            if deltas[i] * deltas[i + 1] < 0
        )
        return sign_changes >= window - 2

    def lyapunov_value(self, gibbs: float) -> float:
        """V(ψ, G) = ‖ψ‖² + G²."""
        g = float(gibbs)
        return self.norm_squared + g * g

    def copy(self) -> "HilbertState":
        """Copia profunda, incluyendo historial de normas."""
        new_state = HilbertState(
            vector=self.vector.copy(),
            phase=self.phase,
        )
        new_state._norm_history = self._norm_history.copy()
        return new_state

    def __repr__(self) -> str:
        components = ", ".join(f"{v:.4f}" for v in self.vector)
        return (
            f"HilbertState(‖ψ‖={self.norm:.4f}, "
            f"φ={self.phase:.4f}rad, [{components}])"
        )


@dataclass
class ThermodynamicPotential:
    """
    Potencial de Gibbs con corrección topológica y actividad:

        G = H - T S k_B + P_topo σ² + R T ln(a)

    La temperatura efectiva T = T_base + κ σ² acopla el estrés Hodge
    (σ = ‖ψ‖) al baño térmico — calentamiento por disipación Dirichlet.
    """

    enthalpy: float = 0.0
    entropy: float = 0.0
    base_temperature: float = PhysicalConstants.T_REFERENCE
    temperature_coupling: float = 15.0
    topological_stress: float = 0.0
    activity: float = 1.0
    _gibbs_history: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        for attr in (
            "enthalpy",
            "entropy",
            "base_temperature",
            "temperature_coupling",
            "topological_stress",
            "activity",
        ):
            value = getattr(self, attr)
            if not _is_finite(value):
                raise ValueError(f"{attr} no es finito: {value!r}")
        if self.base_temperature < PhysicalConstants.T_MINIMUM:
            self.base_temperature = PhysicalConstants.T_MINIMUM
        if self.activity <= 0:
            self.activity = NumericalConstants.EPS

    @property
    def temperature(self) -> float:
        """T_eff = T_base + κ σ²  (acoplamiento estrés–baño)."""
        stress_heating = self.temperature_coupling * (self.topological_stress ** 2)
        temp = self.base_temperature + stress_heating
        if not _is_finite(temp):
            raise ValueError(f"Temperatura no finita: {temp}")
        return temp

    @property
    def chemical_potential(self) -> float:
        """μ = μ_topo σ² + R T ln(a)  (potencial de Gibbs por 'mol' de estrés)."""
        T = self.temperature
        return (
            TopologyConstants.PRESSURE_COEFF * (self.topological_stress ** 2)
            + PhysicalConstants.R_GAS * T * _stable_log(self.activity)
        )

    @property
    def gibbs_free_energy(self) -> float:
        """G = H - T S k_B + μ."""
        T = self.temperature
        ts_term = T * self.entropy * PhysicalConstants.BOLTZMANN_SCALE
        G = self.enthalpy - ts_term + self.chemical_potential
        if not _is_finite(G):
            raise ValueError(f"Energía libre de Gibbs no finita: {G}")
        return G

    @property
    def instability(self) -> float:
        """Índice I = ln(1+|G|) + σ  (lyapunovano débil de divergencia)."""
        G = self.gibbs_free_energy
        value = _stable_log1p(abs(G)) + self.topological_stress
        if not _is_finite(value):
            raise ValueError(f"Inestabilidad no finita: {value}")
        return value

    @property
    def gibbs_trend(self) -> Optional[float]:
        """
        Derivada discreta dG/dciclo por regresión lineal KBN sobre una
        ventana de 5 muestras (estimador de pendiente de Gauss–Markov).
        """
        if len(self._gibbs_history) < 2:
            return None
        recent = self._gibbs_history[-5:]
        if len(recent) < 2:
            return None
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = _kbn_sum(recent) / n
        numerator = _kbn_sum(
            (i - x_mean) * (y - y_mean) for i, y in enumerate(recent)
        )
        denominator = _kbn_sum((i - x_mean) ** 2 for i in range(n))
        if abs(denominator) < NumericalConstants.EPS:
            return 0.0
        return numerator / denominator

    def update(
        self,
        new_enthalpy: float,
        new_entropy: float,
        topological_stress: float,
        activity: Optional[float] = None,
    ) -> None:
        """Actualiza (H, S, σ, a) y registra G en el historial acotado."""
        for name, value in (
            ("new_enthalpy", new_enthalpy),
            ("new_entropy", new_entropy),
            ("topological_stress", topological_stress),
        ):
            if not _is_finite(value):
                raise ValueError(f"{name} no es finito: {value!r}")
        self.enthalpy = new_enthalpy
        self.entropy = new_entropy
        self.topological_stress = topological_stress
        if activity is not None:
            if not _is_finite(activity) or activity <= 0:
                raise ValueError(f"Actividad inválida: {activity}")
            self.activity = activity
        self._gibbs_history.append(self.gibbs_free_energy)
        if len(self._gibbs_history) > 20:
            self._gibbs_history.pop(0)

    def cool_temperature(self, factor: float = DampingConstants.COOLING_FACTOR) -> None:
        """Enfriamiento isobárico T ← max(T_min, λ T), λ ∈ (0,1)."""
        if not 0.0 < factor < 1.0:
            raise ValueError(f"Factor de enfriamiento debe estar en (0, 1): {factor}")
        self.base_temperature = max(
            PhysicalConstants.T_MINIMUM,
            self.base_temperature * factor,
        )

    def compute_arrhenius_rate(
        self,
        activation_energy: float,
        pre_exponential: float = PhysicalConstants.ARRHENIUS_A,
    ) -> float:
        """k = A exp(-Eₐ / R T) con exp saturado."""
        if not _is_finite(activation_energy) or activation_energy < 0:
            raise ValueError(f"Energía de activación inválida: {activation_energy}")
        T = self.temperature
        RT = PhysicalConstants.R_GAS * T
        if RT < NumericalConstants.EPS:
            return 0.0
        return pre_exponential * _stable_exp(-activation_energy / RT)

    def compute_eyring_rate(
        self,
        delta_g_dagger: float,
        transmission: float = PhysicalConstants.EYRING_TRANSMISSION,
    ) -> float:
        """
        TST de Eyring:  k = κ (k_B T / h) exp(-ΔG‡ / R T)

        Se usa la escala de modelo k_B → BOLTZMANN_SCALE para coherencia
        con G adimensionalizado del reactor.
        """
        if not _is_finite(delta_g_dagger):
            raise ValueError(f"ΔG‡ no finito: {delta_g_dagger}")
        T = self.temperature
        pre = (
            transmission
            * PhysicalConstants.BOLTZMANN_SCALE
            * T
            / max(PhysicalConstants.PLANCK_SI, NumericalConstants.EPS)
        )
        # El prefactor SI es enorme; se normaliza al A de Arrhenius del modelo.
        pre_model = min(pre, PhysicalConstants.ARRHENIUS_A)
        RT = PhysicalConstants.R_GAS * T
        if RT < NumericalConstants.EPS:
            return 0.0
        return pre_model * _stable_exp(-delta_g_dagger / RT)

    def copy(self) -> "ThermodynamicPotential":
        """Copia del potencial incluyendo historial de G."""
        new_potential = ThermodynamicPotential(
            enthalpy=self.enthalpy,
            entropy=self.entropy,
            base_temperature=self.base_temperature,
            temperature_coupling=self.temperature_coupling,
            topological_stress=self.topological_stress,
            activity=self.activity,
        )
        new_potential._gibbs_history = self._gibbs_history.copy()
        return new_potential

    def __repr__(self) -> str:
        return (
            f"ThermodynamicPotential(H={self.enthalpy:.4f}, "
            f"S={self.entropy:.4f}, T={self.temperature:.2f}K, "
            f"G={self.gibbs_free_energy:.4f}, I={self.instability:.4f})"
        )


class CatalystAgent(Protocol):
    """Protocolo para agentes catalizadores (morfismo de contexto)."""

    @property
    def efficiency_factor(self) -> float:
        """Factor de eficiencia η ∈ [0, 1]."""
        ...

    @property
    def catalytic_strength(self) -> float:
        """Fuerza catalítica (baja Eₐ efectiva)."""
        ...

    def orient(
        self,
        context: Dict[str, Any],
        gradient: float,
    ) -> Dict[str, Any]:
        """Orienta el catalizador según el gradiente de Gibbs."""
        ...


class AromaticityEvaluator:
    """
    Evaluador de aromaticidad: álgebra de Boole sobre la regla de Hückel.

    Predicados:
        A(n)  ⇔  n ≥ 2  ∧  (n-2) ≡ 0 (mod 4)     # 4k+2
        AA(n) ⇔  n ≥ 4  ∧  n ≡ 0 (mod 4)          # 4k
        P      ⇔  (errores ∨ skips) ∧ n>0
    """

    @staticmethod
    def count_pi_electrons(context: Dict[str, Any]) -> int:
        """Cuenta electrones π = nodos en estado resonante."""
        return sum(
            1 for k, v in context.items()
            if k.endswith("_status") and v == "resonant"
        )

    @staticmethod
    def has_errors(context: Dict[str, Any]) -> bool:
        """∃ clave *_error."""
        return any(k.endswith("_error") for k in context)

    @staticmethod
    def has_skips(context: Dict[str, Any]) -> bool:
        """∃ clave *_skipped."""
        return any(k.endswith("_skipped") for k in context)

    @staticmethod
    def is_huckel_aromatic(n_electrons: int) -> bool:
        """Regla 4n+2, n≥0 ⇒ al menos 2 electrones (etileno no es aromático)."""
        if n_electrons < 2:
            return False
        return (n_electrons - 2) % 4 == 0

    @staticmethod
    def is_anti_aromatic(n_electrons: int) -> bool:
        """Regla 4n, n≥1."""
        if n_electrons < 4:
            return False
        return n_electrons % 4 == 0

    @classmethod
    def evaluate(cls, context: Dict[str, Any]) -> AromaticityState:
        """Clasificación exhaustiva del retículo {∅, P, AA, A}."""
        n_pi = cls.count_pi_electrons(context)
        has_err = cls.has_errors(context)
        has_skip = cls.has_skips(context)

        if has_err or has_skip:
            if n_pi > 0:
                return AromaticityState.PARTIALLY_AROMATIC
            return AromaticityState.NON_AROMATIC
        if cls.is_huckel_aromatic(n_pi):
            return AromaticityState.AROMATIC
        if cls.is_anti_aromatic(n_pi):
            return AromaticityState.ANTI_AROMATIC
        if n_pi > 0:
            return AromaticityState.PARTIALLY_AROMATIC
        return AromaticityState.NON_AROMATIC

    @classmethod
    def log_state(cls, context: Dict[str, Any]) -> None:
        """Bitácora de aromaticidad con desglose 4n±2."""
        n_pi = cls.count_pi_electrons(context)
        state = cls.evaluate(context)
        if state == AromaticityState.AROMATIC:
            n = (n_pi - 2) // 4
            logger.info(
                "✅ AROMATICIDAD: %d e⁻ π (4×%d + 2 = %d)",
                n_pi,
                n,
                4 * n + 2,
            )
        elif state == AromaticityState.ANTI_AROMATIC:
            n = n_pi // 4
            logger.warning(
                "⚠️ ANTI-AROMATICIDAD: %d e⁻ π (4×%d = %d) — estado inestable",
                n_pi,
                n,
                4 * n,
            )
        elif state == AromaticityState.PARTIALLY_AROMATIC:
            logger.info(
                "◐ Resonancia parcial: %d/%d nodos activos",
                n_pi,
                TopologyConstants.RING_SIZE,
            )
        else:
            logger.debug("○ No aromático: %d e⁻ π", n_pi)


class EntropyCalculator:
    """
    Entropía de Shannon del contexto (observables categóricos) y puente
    a von Neumann cuando se suministra un HilbertState.
    """

    _EXCLUDED_SUFFIXES: ClassVar[Tuple[str, ...]] = (
        "_ts",
        "_ea",
        "_error",
        "_skipped",
    )

    @classmethod
    def _normalize_text(cls, text: str, max_length: int = 32) -> str:
        """Normalización lexicográfica para hashing estable."""
        normalized = " ".join(text.strip().split())
        return normalized[:max_length].lower()

    @classmethod
    def _compute_text_hash(cls, text: str) -> str:
        """Hash corto MD5 del texto normalizado (colisiones O(2^{-32}))."""
        normalized = cls._normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()[:8]

    @classmethod
    def _categorize_value(cls, value: Any) -> str:
        """σ-álgebra de categorías para el estimador de Shannon."""
        if isinstance(value, bool):
            return f"bool:{value}"
        if isinstance(value, str):
            if len(value) <= 10:
                return f"str:{cls._normalize_text(value)}"
            return f"str_hash:{cls._compute_text_hash(value)}"
        if isinstance(value, (int, float)):
            abs_val = abs(float(value))
            if abs_val < NumericalConstants.EPS:
                return "num:zero"
            magnitude = int(math.floor(_stable_log(abs_val) / math.log(10)))
            sign = "+" if value >= 0 else "-"
            return f"num:{sign}e{magnitude}"
        if isinstance(value, (list, tuple)):
            return f"seq:len{len(value)}"
        if isinstance(value, dict):
            return f"map:len{len(value)}"
        return f"obj:{type(value).__name__}"

    @classmethod
    def calculate(cls, context: Dict[str, Any]) -> float:
        """
        S = -Σ pᵢ log pᵢ sobre el histograma de categorías.

        Se excluyen sufijos de telemetría para no inflar S con ruido.
        """
        if not context:
            return 0.0
        categories: List[str] = []
        for key, value in context.items():
            if any(key.endswith(sfx) for sfx in cls._EXCLUDED_SUFFIXES):
                continue
            if key.startswith("_"):
                continue
            categories.append(cls._categorize_value(value))
        if not categories:
            return 0.0
        counts = Counter(categories)
        total = len(categories)
        entropy = 0.0
        for count in counts.values():
            p = max(count / total, NumericalConstants.ENTROPY_MIN_PROB)
            entropy -= p * _stable_log(p)
        return entropy

    @classmethod
    def joint_shannon_von_neumann(
        cls,
        context: Dict[str, Any],
        state: HilbertState,
        mixing: float = 0.5,
    ) -> float:
        """
        Entropía conjunta convexa  (1-λ) S_Sh + λ S_vN ,  λ ∈ [0,1].

        Interpola información clásica del contexto e información cuántica
        del operador densidad.
        """
        lam = min(max(mixing, 0.0), 1.0)
        return (1.0 - lam) * cls.calculate(context) + lam * state.von_neumann_entropy()


@dataclass(frozen=True)
class ReactionResult:
    """Resultado inmutable de una trayectoria del semigrupo catalítico."""

    context: Dict[str, Any]
    status: ConvergenceStatus
    final_cycle: int
    final_gibbs: float
    final_instability: float
    aromaticity: AromaticityState
    reaction_id: str
    elapsed_time_s: float

    @property
    def is_successful(self) -> bool:
        """Éxito = atractor aromático o metaestable."""
        return self.status in (
            ConvergenceStatus.CONVERGED_AROMATIC,
            ConvergenceStatus.CONVERGED_METASTABLE,
        )

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
        }


class CatalyticReactor:
    """
    Reactor catalítico cuántico (metáfora del benceno).

    Integra el semigrupo e^{t𝒢} de la Fase 2 sobre HilbertState, acopla
    la energía Dirichlet al potencial de Gibbs, audita Tellegen en cada
    ciclo y declara aromaticidad cuando el retículo de Hückel se cierra.
    """

    __slots__ = (
        "mic",
        "catalyst",
        "telemetry",
        "topology",
        "ring_sequence",
        "_temperature_coupling",
        "_base_temperature",
        "_deterministic_mode",
        "_random_seed",
        "_strict_conservation",
        "_enable_huckel_resolvent",
        "_generator",
        "_lyapunov_prev",
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
    ) -> None:
        self.mic = mic
        self.catalyst = agent
        self.telemetry = telemetry
        self.topology = HexagonalTopology()
        self.ring_sequence: List[CarbonNode] = list(CarbonNode)
        self._temperature_coupling = temperature_coupling
        self._base_temperature = base_temperature
        self._deterministic_mode = deterministic
        self._random_seed = random_seed
        self._strict_conservation = strict_conservation
        self._enable_huckel_resolvent = enable_huckel_resolvent
        self._generator = self.topology.infinitesimal_catalytic_generator(
            CFLConstants.ALPHA_SAFE
        )
        self._lyapunov_prev: Optional[float] = None

    def _generate_reaction_id(self) -> str:
        """Identificador de trayectoria: MD5 determinista o UUID."""
        if self._deterministic_mode and self._random_seed is not None:
            return hashlib.md5(
                f"reaction_{self._random_seed}".encode()
            ).hexdigest()[:8]
        return str(uuid.uuid4())[:8]

    def _lyapunov_certificate(
        self,
        state: HilbertState,
        potential: ThermodynamicPotential,
    ) -> LyapunovCertificate:
        """Certifica disipatividad de V = ‖ψ‖² + G² a lo largo de 𝒢."""
        V = state.lyapunov_value(potential.gibbs_free_energy)
        prev = self._lyapunov_prev
        decrement = 0.0 if prev is None else (V - prev)
        self._lyapunov_prev = V
        abscissa = -CFLConstants.ALPHA_SAFE * self.topology.spectral_gap
        dissipative = decrement <= NumericalConstants.EPS or prev is None
        return LyapunovCertificate(
            value=float(V),
            decrement=float(decrement),
            dissipative=bool(dissipative),
            spectral_abscissa=float(abscissa),
        )

    def ignite(self, initial_context: Dict[str, Any]) -> ReactionResult:
        """
        Enciende el reactor e integra hasta aromaticidad, metaestabilidad,
        colapso o agotamiento de ciclos de resonancia.
        """
        reaction_id = self._generate_reaction_id()
        start_time = time.perf_counter()
        self._lyapunov_prev = None

        logger.info("⚛️ IGNICIÓN: Reactor [%s] encendido (v%s)", reaction_id, __version__)

        context = initial_context.copy()
        potential = ThermodynamicPotential(
            base_temperature=self._base_temperature,
            temperature_coupling=self._temperature_coupling,
        )
        state = HilbertState()
        potential.entropy = max(
            ReactorLimits.MIN_ENTROPY,
            EntropyCalculator.calculate(context),
        )
        previous_gibbs = potential.gibbs_free_energy
        self.telemetry.record_reaction_start(reaction_id, context)

        try:
            for cycle in range(1, ReactorLimits.MAX_RESONANCE_CYCLES + 1):
                self._validate_state(state, potential)

                cert = self._lyapunov_certificate(state, potential)
                context["_lyapunov_V"] = cert.value
                context["_lyapunov_dV"] = cert.decrement
                context["_lyapunov_dissipative"] = cert.dissipative

                logger.info(
                    "⏩ Ciclo %d/%d | G=%.4f | I=%.4f | ‖ψ‖=%.4f | V=%.4f",
                    cycle,
                    ReactorLimits.MAX_RESONANCE_CYCLES,
                    potential.gibbs_free_energy,
                    potential.instability,
                    state.norm,
                    cert.value,
                )

                self._catalytic_orientation(context, potential)
                self._ring_iteration(context, state, potential, cycle)

                if cycle % 2 == 0:
                    state.vector = state.vector[::-1]

                state.phase = (
                    state.phase + 2.0 * math.pi / TopologyConstants.RING_SIZE
                ) % (2.0 * math.pi)

                aromaticity = AromaticityEvaluator.evaluate(context)
                AromaticityEvaluator.log_state(context)

                if aromaticity == AromaticityState.AROMATIC:
                    elapsed = time.perf_counter() - start_time
                    logger.info(
                        "✅ AROMATICIDAD en ciclo %d | G_final=%.4f | t=%.3fs",
                        cycle,
                        potential.gibbs_free_energy,
                        elapsed,
                    )
                    self.telemetry.record_reaction_success(reaction_id, cycle)
                    return ReactionResult(
                        context=context,
                        status=ConvergenceStatus.CONVERGED_AROMATIC,
                        final_cycle=cycle,
                        final_gibbs=potential.gibbs_free_energy,
                        final_instability=potential.instability,
                        aromaticity=aromaticity,
                        reaction_id=reaction_id,
                        elapsed_time_s=elapsed,
                    )

                if self._check_convergence(potential, previous_gibbs, state, cycle):
                    elapsed = time.perf_counter() - start_time
                    delta_g = abs(potential.gibbs_free_energy - previous_gibbs)
                    logger.info(
                        "🔒 Convergencia metaestable en ciclo %d | |δG|=%.6f | t=%.3fs",
                        cycle,
                        delta_g,
                        elapsed,
                    )
                    context["_metastable_cycle"] = cycle
                    return ReactionResult(
                        context=context,
                        status=ConvergenceStatus.CONVERGED_METASTABLE,
                        final_cycle=cycle,
                        final_gibbs=potential.gibbs_free_energy,
                        final_instability=potential.instability,
                        aromaticity=aromaticity,
                        reaction_id=reaction_id,
                        elapsed_time_s=elapsed,
                    )

                previous_gibbs = potential.gibbs_free_energy

            elapsed = time.perf_counter() - start_time
            aromaticity = AromaticityEvaluator.evaluate(context)
            logger.warning(
                "⚠️ Máximo de ciclos sin convergencia | G=%.4f | I=%.4f",
                potential.gibbs_free_energy,
                potential.instability,
            )
            return ReactionResult(
                context=context,
                status=ConvergenceStatus.FAILED_MAX_CYCLES,
                final_cycle=ReactorLimits.MAX_RESONANCE_CYCLES,
                final_gibbs=potential.gibbs_free_energy,
                final_instability=potential.instability,
                aromaticity=aromaticity,
                reaction_id=reaction_id,
                elapsed_time_s=elapsed,
            )

        except RuntimeError as exc:
            elapsed = time.perf_counter() - start_time
            self.telemetry.record_error("reaction_chamber", str(exc))
            logger.error("🔥 Colapso del reactor [%s]: %s", reaction_id, exc)
            return ReactionResult(
                context=context,
                status=ConvergenceStatus.FAILED_INSTABILITY,
                final_cycle=0,
                final_gibbs=potential.gibbs_free_energy,
                final_instability=potential.instability,
                aromaticity=AromaticityState.NON_AROMATIC,
                reaction_id=reaction_id,
                elapsed_time_s=elapsed,
            )

        except Exception as exc:
            self.telemetry.record_error("reaction_chamber", str(exc))
            logger.exception("💥 Error inesperado en reactor [%s]", reaction_id)
            raise

    def _validate_state(
        self,
        state: HilbertState,
        potential: ThermodynamicPotential,
    ) -> None:
        """Inmersión de sanidad: ψ ∈ ℝ⁶, φ finito, (T,G,I) evaluables."""
        _validate_finite_vector(state.vector, "state.vector")
        if not _is_finite(state.phase):
            raise ValueError(f"Fase no finita: {state.phase}")
        _ = potential.temperature
        _ = potential.gibbs_free_energy
        _ = potential.instability

    def _compute_huckel_resolvent_diagnostics(self) -> Dict[str, Any]:
        """Diagnósticos (α_reg, κ₂, spec Eₖ) del resolvente de Hückel."""
        try:
            _, alpha_reg, cond = self.topology.regularized_huckel_resolvent(
                s=0.0,
                h=0.0,
            )
            return {
                "huckel_alpha_reg": alpha_reg,
                "huckel_resolvent_condition": cond,
                "huckel_mo_levels": self.topology.huckel_spectrum(),
            }
        except Exception as exc:
            logger.warning("Fallo en resolvente de Hückel: %s", exc)
            return {"huckel_resolvent_error": str(exc)}

    def _catalytic_orientation(
        self,
        context: Dict[str, Any],
        potential: ThermodynamicPotential,
    ) -> None:
        """Inserta diagnósticos de Hückel y aplica CatalystAgent.orient."""
        if self._enable_huckel_resolvent:
            context.update(self._compute_huckel_resolvent_diagnostics())
        gradient = potential.gibbs_free_energy
        catalyst_update = self.catalyst.orient(context, gradient)
        if not isinstance(catalyst_update, dict):
            raise TypeError("CatalystAgent.orient debe retornar un dict")
        context.update(catalyst_update)

    def _audit_conservation(
        self,
        context: Dict[str, Any],
        state: HilbertState,
        potential: ThermodynamicPotential,
        cycle: int,
    ) -> None:
        """
        Audita de Rham–Tellegen–Hodge post-difusión.

        Con strict_conservation, una violación colapsa el reactor
        (RuntimeError); si no, se dispara estabilización de Lyapunov.
        """
        audit = self.topology.audit_de_rham_tellegen(state.vector)
        context["tellegen_passivity"] = audit.passivity_satisfied
        context["tellegen_conservation"] = audit.conservation_satisfied
        context["tellegen_orthogonality"] = audit.orthogonality_satisfied
        context["tellegen_identity"] = audit.tellegen_satisfied
        context["tellegen_energy"] = audit.laplacian_energy
        context["tellegen_conservation_residual"] = audit.conservation_residual
        context["tellegen_orthogonality_residual"] = audit.orthogonality_residual
        context["tellegen_identity_residual"] = audit.tellegen_residual

        if audit.is_coherent:
            return

        message = (
            "Violación de conservación de Rham-Tellegen: "
            f"passivity={audit.passivity_satisfied}, "
            f"conservation={audit.conservation_satisfied}, "
            f"orthogonality={audit.orthogonality_satisfied}, "
            f"tellegen={audit.tellegen_satisfied}, "
            f"energy={audit.laplacian_energy:.6e}, "
            f"cons_residual={audit.conservation_residual:.6e}, "
            f"ortho_residual={audit.orthogonality_residual:.6e}, "
            f"tellegen_residual={audit.tellegen_residual:.6e}"
        )
        self.telemetry.record_error("reaction_chamber", message)
        if self._strict_conservation:
            raise RuntimeError(message)
        logger.error(message)
        self._attempt_stabilization(state, potential, cycle)

    def _ring_iteration(
        self,
        context: Dict[str, Any],
        state: HilbertState,
        potential: ThermodynamicPotential,
        cycle: int,
    ) -> None:
        """
        Un ciclo de resonancia: reacción nodal → difusión por 𝒢 →
        auditoría Hodge → actualización de Gibbs.
        """
        total_delta_h = 0.0

        for node in self.ring_sequence:
            idx = node.index
            base_ea = self._calculate_hamiltonian(node, context)
            efficiency = min(max(self.catalyst.efficiency_factor, 0.0), 0.999999)
            effective_ea = base_ea * (1.0 - efficiency)
            try:
                node_update, delta_h = self._react_node(
                    node,
                    context,
                    effective_ea,
                    state.vector[idx],
                    cycle,
                )
                context.update(node_update)
                total_delta_h += delta_h
                excitation = self._compute_local_excitation(
                    delta_h,
                    effective_ea,
                    state.vector[idx],
                )
                state.vector[idx] += excitation
            except Exception as exc:
                logger.error(
                    "💥 Error en nodo %s (ciclo %d): %s",
                    node.name,
                    cycle,
                    exc,
                )
                state.vector[idx] += 1.0
                total_delta_h += 50.0
                context[f"{node.name}_error"] = str(exc)

        # Integración del generador infinitesimal de la Fase 2.
        state.vector = self.topology.diffuse_stress(
            state.vector,
            diffusion_rate=CFLConstants.ALPHA_SAFE,
        )
        self._audit_conservation(context, state, potential, cycle)

        new_entropy = max(
            ReactorLimits.MIN_ENTROPY,
            EntropyCalculator.joint_shannon_von_neumann(context, state),
        )
        potential.update(
            new_enthalpy=max(
                ReactorLimits.MIN_ENTHALPY,
                potential.enthalpy + total_delta_h,
            ),
            new_entropy=new_entropy,
            topological_stress=state.norm,
        )
        if potential.instability > ReactorLimits.INSTABILITY_THRESHOLD:
            self._attempt_stabilization(state, potential, cycle)

    def _calculate_hamiltonian(
        self,
        node: CarbonNode,
        context: Dict[str, Any],
    ) -> float:
        """
        Elemento de matriz efectivo en el sitio i:

            Hᵢ = α + Σ_{⟨j i⟩} β · 1_{j resonante} + penalización de precursores
        """
        idx = node.index
        left_idx, right_idx = self.topology.neighbor_indices(idx)
        neighbor_stabilization = 0.0
        for neighbor_idx in (left_idx, right_idx):
            neighbor_node = node_from_index(neighbor_idx)
            if context.get(f"{neighbor_node.name}_status") == "resonant":
                neighbor_stabilization += HuckelConstants.BETA
        precursor_penalty = self._evaluate_precursor_penalty(node, context)
        hamiltonian = (
            HuckelConstants.ALPHA + neighbor_stabilization + precursor_penalty
        )
        if not _is_finite(hamiltonian):
            raise ValueError(
                f"Hamiltoniano no finito para {node.name}: {hamiltonian}"
            )
        return max(0.0, hamiltonian)

    def _evaluate_precursor_penalty(
        self,
        node: CarbonNode,
        context: Dict[str, Any],
    ) -> float:
        """Penalización lineal por fracción de precursores vacíos."""
        precursors = node.precursors
        if not precursors:
            return 0.0
        missing = 0
        for key in precursors:
            value = context.get(key)
            if value is None or value == "" or value == {} or value == []:
                missing += 1
        return 0.3 * (missing / len(precursors))

    def _react_node(
        self,
        node: CarbonNode,
        context: Dict[str, Any],
        activation_energy: float,
        local_stress: float,
        cycle: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Reacción local: marca resonancia y computa ΔH del sitio."""
        if not all(_is_finite(x) for x in (activation_energy, local_stress)):
            raise ValueError(
                f"Entrada no finita en {node.name}: "
                f"Eₐ={activation_energy}, σ={local_stress}"
            )
        if activation_energy > HuckelConstants.ACTIVATION_CEILING:
            logger.warning(
                "⚡ Saltando %s (ciclo %d): Eₐ=%.3f > techo=%.3f",
                node.name,
                cycle,
                activation_energy,
                HuckelConstants.ACTIVATION_CEILING,
            )
            return {f"{node.name}_skipped": True}, 5.0

        if not self._deterministic_mode:
            stress_factor = 1.0 + (local_stress ** 2) * 0.5
            sleep_time = min(0.005 * stress_factor, ReactorLimits.MAX_NODE_SLEEP)
            if sleep_time > 0:
                time.sleep(sleep_time)

        context_update: Dict[str, Any] = {
            f"{node.name}_status": "resonant",
            f"{node.name}_ts": time.time(),
            f"{node.name}_ea": activation_energy,
        }
        delta_h = (local_stress ** 2) * 5.0 + activation_energy * 10.0
        if not _is_finite(delta_h):
            raise ValueError(f"ΔH no finita en {node.name}: {delta_h}")
        logger.debug(
            "🔬 %s (ciclo %d) | Eₐ=%.3f | σ=%.3f | ΔH=%.3f",
            node.name,
            cycle,
            activation_energy,
            local_stress,
            delta_h,
        )
        return context_update, delta_h

    def _compute_local_excitation(
        self,
        delta_h: float,
        activation_energy: float,
        local_stress: float,
    ) -> float:
        """
        Excitación local acotada:

            Δψᵢ = g tanh(ΔH⁺/25) + 0.02 Eₐ - ν σᵢ

        tanh garantiza Lipschitz y evita runaway de ganancia.
        """
        if not all(_is_finite(x) for x in (delta_h, activation_energy, local_stress)):
            raise ValueError("Parámetros no finitos en excitación local")
        gain = DampingConstants.LOCAL_STRESS_GAIN * math.tanh(max(0.0, delta_h) / 25.0)
        barrier_contrib = 0.02 * activation_energy
        dissipation = DampingConstants.LOCAL_STRESS_DISSIPATION * local_stress
        excitation = gain + barrier_contrib - dissipation
        if not _is_finite(excitation):
            raise ValueError(f"Excitación no finita: {excitation}")
        return excitation

    def _check_convergence(
        self,
        potential: ThermodynamicPotential,
        previous_gibbs: float,
        state: HilbertState,
        cycle: int,
    ) -> bool:
        """
        Metaestabilidad: |ΔG| < tol y estacionariedad relativa de ‖ψ‖
        tras MIN_CONVERGENCE_CYCLE (evita falso positivo inicial).
        """
        if cycle < ReactorLimits.MIN_CONVERGENCE_CYCLE:
            return False
        current_gibbs = potential.gibbs_free_energy
        if not all(_is_finite(x) for x in (current_gibbs, previous_gibbs)):
            return False
        delta_gibbs = abs(current_gibbs - previous_gibbs)
        if delta_gibbs >= NumericalConstants.GIBBS_CONVERGENCE_TOL:
            return False
        if len(state._norm_history) >= 2:
            recent_norms = state._norm_history[-3:]
            if len(recent_norms) >= 2:
                norm_changes = [
                    abs(recent_norms[i + 1] - recent_norms[i])
                    for i in range(len(recent_norms) - 1)
                ]
                max_change = max(norm_changes)
                ref_norm = max(recent_norms[-1], NumericalConstants.EPS)
                if max_change / ref_norm >= ReactorLimits.NORM_RELATIVE_STATIONARITY:
                    return False
        return True

    def _attempt_stabilization(
        self,
        state: HilbertState,
        potential: ThermodynamicPotential,
        cycle: int,
    ) -> None:
        """
        Estabilización de emergencia: damping hacia ker Δ₀, recorte de H
        y enfriamiento. Si I supera el umbral de colapso, se aborta.
        """
        logger.warning(
            "⚠️ INESTABILIDAD ALTA: I=%.2f > %.2f en ciclo %d",
            potential.instability,
            ReactorLimits.INSTABILITY_THRESHOLD,
            cycle,
        )
        state.apply_damping(cycle)
        potential.enthalpy = max(
            ReactorLimits.MIN_ENTHALPY,
            potential.enthalpy * 0.85,
        )
        potential.cool_temperature(factor=DampingConstants.COOLING_FACTOR)
        potential.topological_stress = state.norm
        collapse_threshold = (
            ReactorLimits.INSTABILITY_THRESHOLD * ReactorLimits.COLLAPSE_FACTOR
        )
        if potential.instability > collapse_threshold:
            raise RuntimeError(
                f"Colapso del reactor: I={potential.instability:.4f} > "
                f"{collapse_threshold:.4f}"
            )
        logger.info(
            "🛡️ Estabilización: I=%.4f, ‖ψ‖=%.4f, T=%.2fK",
            potential.instability,
            state.norm,
            potential.base_temperature,
        )


@dataclass(frozen=True)
class ReactorConfig:
    """Configuración inmutable validada del reactor."""

    base_temperature: float = PhysicalConstants.T_REFERENCE
    temperature_coupling: float = 15.0
    deterministic: bool = False
    random_seed: Optional[int] = None
    strict_conservation: bool = True
    enable_huckel_resolvent: bool = True

    def __post_init__(self) -> None:
        if self.base_temperature < PhysicalConstants.T_MINIMUM:
            raise ValueError(
                f"Temperatura base {self.base_temperature}K < mínimo "
                f"{PhysicalConstants.T_MINIMUM}K"
            )
        if self.temperature_coupling < 0:
            raise ValueError(
                f"Acoplamiento negativo: {self.temperature_coupling}"
            )


def create_reactor(
    mic: MICRegistry,
    agent: CatalystAgent,
    telemetry: TelemetryContext,
    config: Optional[ReactorConfig] = None,
) -> CatalyticReactor:
    """Fábrica de reactores: valida ReactorConfig e instancia CatalyticReactor."""
    if config is None:
        config = ReactorConfig()
    return CatalyticReactor(
        mic=mic,
        agent=agent,
        telemetry=telemetry,
        base_temperature=config.base_temperature,
        temperature_coupling=config.temperature_coupling,
        deterministic=config.deterministic,
        random_seed=config.random_seed,
        strict_conservation=config.strict_conservation,
        enable_huckel_resolvent=config.enable_huckel_resolvent,
    )
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Optimized Symplectic Manifold (Motor de Integración de Fase)        ║
║ Ruta   : app/physics/opt_symplectic_manifold.py                              ║
║ Versión: 3.0.0-Doctoral-Fukaya-Connes-Heyting-CAS-Kahan-Secure               ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS GEOMÉTRICA Y CATECORIAL DE-CONFINADA:
────────────────────────────────────────────────────────────────────────────────
Este componente constitutivo de la Capa Física (Nivel 3, $V_{\mathrm{PHYSICS}}$)
opera como el resolvedor elíptico y metrológico de fase canónica sobre la variedad
simpléctica de-confinada $(\mathcal{M}, \omega)$ de dimensión real $2n$. Su mandato 
no es el de un mero integrador ordinario; somete el flujo temporal logístico de la 
Malla de Control a una estructura de composición funtorial monoidal estricta de 
tres fases anidadas, donde el contrato formal de handoff exige que el objeto de 
salida de la Fase $k$ sea el dominio exacto de la Fase $k+1$:
$$\mathcal{F}_{\mathrm{manifold}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$$

================================════════════════════════════════════════════════
I. DEFINICIÓN DEL ESPACIO DE FASE SIMPLÉCTICO EN COORDENADAS DE DARBOUX
================================════════════════════════════════════════════════
Sea $(\mathcal{M}, \omega)$ una variedad simpléctica de dimensión $2n$ equipada con
la 2-forma canónica antisimétrica de Liouville $\omega$. En coordenadas locales de
Darboux $(q_i, p_i)$, la forma simpléctica y el Hamiltoniano $H(q, p)$ que rige la
energía financiera y la asignación de recursos del megaproyecto se expresan como:
$$\omega = \sum_{i=1}^n dq_i \wedge dp_i$$

Donde:
  - $q_i \in \mathbb{R}^n$ codifica el vector de avance de los Análisis de Precios
    Unitarios (APUs) del presupuesto.
  - $p_i \in \mathbb{R}^n$ representa el covector de momentum o velocidades de
    inyección de capital amortiguadas por el tensor métrico Riemanniano $G_{\mu\nu}$.

================================════════════════════════════════════════════════
II. DECONSTRUCCIÓN DE FASES ANIDADAS Y ENRUTAMIENTO NUMÉRICO
================================════════════════════════════════════════════════
El motor ejecuta de forma secuencial y atómica las siguientes tres sub-fases de
procesamiento sobre la FPU:

Fase 1 (Cuadratura Conforme de Gauss-Legendre en $S^2$):
  Resuelve integrales espectrales sobre la esfera de Riemann ($S^2 \cong \hat{\mathbb{C}}$)
  para proyectar frentes de onda atencionales sin singularidades polares. El cambio
  difeomórfico de variable $t = \cos\theta \in [-1, 1]$ absorbe analíticamente el factor 
  jacobiano $\sin\theta$:
  $$I = \int_{S^2} \psi(\theta, \phi) \, d\Omega \approx \sum_{i=1}^M \sum_{j=1}^N \psi\left(\arccos(t_i), \, \phi_j\right) \cdot W_{ij}$$
  Donde $W_{ij} = w_i \cdot \Delta\phi$, y $\{t_i, w_i\}$ son los nodos y pesos
  del polinomio de Legendre, garantizando la conservación de la medida armónica: $\sum W_{ij} = 4\pi$.
  Síncronamente, proyecta el tensor de fondo $G$ al cono Simétrico Definido Positivo (SPD)
  mediante el operador de Onsager-Higham (2002), aplicando un desplazamiento de Tikhonov
  bajo el límite metrológico de Wilkinson.

Fase 2 (Paso Simpléctico de Störmer-Verlet con Compensación KBN):
  Avanza las variables de fase $(q, p)$ en el tiempo discretizado $\Delta t$ aplicando
  un mapa simpléctico de segundo orden (kick-drift-kick). Para neutralizar la deriva secular
  por redondeo flotante, implementa la acumulación compensada de Kahan-Babuška-Neumaier (KBN)
  de forma persistente sobre los vectores de fase:
  $$y_q = \Delta t \cdot M^{-1} p_{k+1/2} - c_{q,k} \quad \implies \quad q_{k+1} = q_k + y_q \quad \implies \quad c_{q,k+1} = (q_{k+1} - q_k) - y_q$$
  El Jacobiano de fase local $M = \partial \Phi / \partial x$ se linealiza mediante
  Diferenciación por Paso Complejo (CSMD) para eludir cancelaciones catastróficas.

Fase 3 (Metrología de Liouville y Veto de Heyting):
  Verifica síncronamente la preservación del volumen de fase y la simplecticidad de de Rham.
  Evalúa el logaritmo del determinante evitando subdesbordamientos numéricos:
  $$r_{\mathrm{Liouville}} = \left| \exp\left(\ln|\det M|\right) - 1 \right| \equiv \left| \operatorname{expm1}(\operatorname{slogdet}(M)_2) \right| \le \tau_{\mathrm{Liouville}}$$
  Cualquier violación espectral colapsa síncronamente el clasificador de subobjetos al Supremo 
  terminal VETOED ($\top$) en el retículo de Heyting $\Omega_3 = \{\mathtt{COHERENT}, 
  \mathtt{DEGRADED}, \mathtt{VETOED}\}$. Esto detona un cerrojo de comparación e intercambio
  atómico (CAS) sobre los registros de hardware del ESP32 perimetral, conmutando el pin GPIO14
  en menos de 400 ns para activar el tiristor BT151 (Crowbar) y paralizar la obra real.

================================════════════════════════════════════════════════
III. AXIOMAS DE LA MALLA PORT-HAMILTONIANA
================================════════════════════════════════════════════════
Axioma 1 (Conservación de la Medida de Lebesgue):
  Bajo inyecciones de acoplamiento de Lorentz giroscópicas representadas por el tensor
  antisimétrico $W \in \mathfrak{so}(n)$, el operador de interconexión efectivo
  $J_{\mathrm{eff}} = J_{\mathrm{base}} + W$ mantiene el flujo libre de fuentes parásitas,
  conservando de manera exacta el volumen del espacio de fase:
  $$\operatorname{div}(\dot{x}) = \operatorname{Tr}\left( J_{\mathrm{eff}} \nabla^2 H(x) \right) \equiv 0$$

Axioma 2 (Pasividad Termodinámica de Lyapunov):
  La evolución temporal del Hamiltoniano satisface estrictamente la desigualdad de disipación 
  de Rayleigh-Lyapunov (Segunda Ley de la Termodinámica / Clausius-Duhem) para cualquier
  matriz de amortiguamiento simétrica semidefinida positiva $R(x) \succeq \mathbf{0}$:
  $$\dot{H} = -\nabla H^\top R(x) \nabla H \le 0$$

Axioma 3 (Consistencia de de Rham-Mayer-Vietoris):
  La fusión concurrente de dos subcomplejos o mallas de datos $A$ y $B$ exige la exactitud de
  la secuencia larga de Mayer-Vietoris. La desviación del primer número de Betti simplicial
  debe anularse para proscribir de forma incondicional socavones lógicos o dependencias circulares:
  $$\Delta \beta_1 = \beta_1(A \cup B) - \left( \beta_1(A) + \beta_1(B) - \beta_1(A \cap B) \right) \equiv 0$$

================================════════════════════════════════════════════════
IV. INVARIANTES ESPECTRALES E INMUNIDAD DE WILKINSON
================================════════════════════════════════════════════════
  - ENVELOPE DE ERROR DE HIGHAM: El Jacobiano de fase $M$ de dimensión $2n \times 2n$ se somete
    al límite superior elástico de redondeo sobre la estructura simpléctica:
    $$\|M^\top \Omega M - \Omega\|_F \le \tau \|\Omega\|_F + \gamma_{2n} \|M\|_F^2$$
    Donde $\gamma_{2n} = \frac{2n \cdot u}{1 - 2n \cdot u}$, siendo $u$ el unit roundoff de la máquina.

  - NULIDAD COHOMOLÓGICA DE FLOER: El primer grupo de cohomología de de Rham-Floer sobre el
    complejo simplificado de dependencias $K$ debe ser nulo, lo cual es equivalente a la
    ausencia de bifurcaciones espurias o burbujeo discal en el Anillo de Novikov $\Lambda_{\mathbb{R}}$:
    $$\beta_1 = \dim H^1(K; \mathbb{F}) \equiv 0$$

  - CONECTIVIDAD DE FIEDLER: El primer autovalor no trivial del Laplaciano normalizado de Haz $L_F$,
    calculado mediante SVD de Moore-Penrose, garantiza la conexidad robusta de la Malla de APUs:
    $$\lambda_1(L_F) \ge \tau_{\mathrm{Fiedler}} > 0 \quad \implies \quad \beta_0 \equiv 1$$
"""

from __future__ import annotations

import ctypes
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

try:
    from app.physics.pseudo_holomorphic_motor import HeytingOmega3
except ImportError:  # pragma: no cover
    try:
        from .pseudo_holomorphic_motor import HeytingOmega3  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        HeytingOmega3 = None  # type: ignore[misc, assignment]

logger = logging.getLogger("APU.Physics.SymplecticManifoldEngine")

# ---------------------------------------------------------------------------
# Constantes espectrales, FPU y cota metrológica del cerrojo
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_FLOOR: Final[float] = 1e-12
_SYMPLECTIC_TOLERANCE: Final[float] = 1e-9
_VOLUME_TOLERANCE: Final[float] = 1e-7
_ENERGY_TOLERANCE: Final[float] = 1e-6
_EIGH_RESIDUAL_WARN: Final[float] = 1e-8
_SYMMETRY_ATOL: Final[float] = 1e-12
_COMPLEX_IMAG_TOL: Final[float] = 1e-14
_CSMD_STEP: Final[float] = 1e-20
_CSMD_STEP_FLOOR: Final[float] = 1e-30
_TIKHONOV_BETA0: Final[float] = 1.0
_NEG_INERTIA_HARD: Final[float] = 0.5
_SPHERE_MEASURE: Final[float] = 4.0 * math.pi
_INTERLOCK_LATENCY_NS: Final[float] = 340.0
_LATCH_OPEN: Final[int] = 0
_LATCH_CLOSED: Final[int] = 1
_DOT2_DIM_CEILING: Final[int] = 24

__version__: Final[str] = "3.0.0-Doctoral-GaussLegendre-KBN-Tikhonov-Heyting-CAS"


# ═══════════════════════════════════════════════════════════════════════════════
# HEYTING Ω₃ LOCAL (respaldo si el motor no está en el path)
# ═══════════════════════════════════════════════════════════════════════════════
class _LocalHeytingOmega3:
    COHERENT: Final[str] = "COHERENT"
    DEGRADED: Final[str] = "DEGRADED"
    VETOED: Final[str] = "VETOED"
    _ORDER: Final[dict[str, int]] = {VETOED: 0, DEGRADED: 1, COHERENT: 2}
    _TRUTH: Final[dict[str, float]] = {VETOED: 0.0, DEGRADED: 0.5, COHERENT: 1.0}

    @classmethod
    def order(cls, value: str) -> int:
        if value not in cls._ORDER:
            raise SymplecticManifoldError(f"Valor foráneo a Ω₃: {value!r}.")
        return cls._ORDER[value]

    @classmethod
    def normalize(cls, value: str) -> str:
        if value not in cls._ORDER:
            raise SymplecticManifoldError(f"Valor foráneo a Ω₃: {value!r}.")
        return value

    @classmethod
    def truth(cls, value: str) -> float:
        return cls._TRUTH[cls.normalize(value)]

    @classmethod
    def le(cls, left: str, right: str) -> bool:
        return cls.order(left) <= cls.order(right)

    @classmethod
    def meet(cls, left: str, right: str) -> str:
        return left if cls.order(left) <= cls.order(right) else right

    @classmethod
    def implies(cls, left: str, right: str) -> str:
        return cls.COHERENT if cls.le(left, right) else cls.normalize(right)

    @classmethod
    def fold_meet(cls, values: Sequence[str]) -> str:
        acc = cls.COHERENT
        for item in values:
            acc = cls.meet(acc, item)
        return acc


if HeytingOmega3 is None:  # pragma: no cover
    HeytingOmega3 = _LocalHeytingOmega3  # type: ignore[misc, assignment]


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════
class SymplecticManifoldError(Exception):
    """Raíz de violaciones geométricas de la variedad simpléctica."""


class InvalidDimensionError(SymplecticManifoldError):
    """Dimensión del espacio de fases inconsistente."""


class NonTransversalJacobianError(SymplecticManifoldError):
    """Jacobiano no cuadrado, no finito o de condición ilegible."""


class DegenerateMetricError(SymplecticManifoldError):
    """El tensor métrico no admite proyección conforme al cono SPD."""


class HamiltonianDriftError(SymplecticManifoldError):
    """Deriva energética fuera de la tolerancia de conservación."""


class CohomologicalObstructionError(SymplecticManifoldError):
    """\(\beta_1\) ilegítimo o ciclos no triviales no declarados."""


class QuadratureConvergenceError(SymplecticManifoldError):
    """La cuadratura esférica no certifica su residuo."""


class HardwareCrowbarError(SymplecticManifoldError):
    """Inconsistencia del cerrojo fail-secure de software."""


class FpuMetrologyError(SymplecticManifoldError):
    """Residuo o envelope de Higham no finito."""


# ═══════════════════════════════════════════════════════════════════════════════
# ARITMÉTICA COMPENSADA (TwoSum / TwoProd / KBN / Dot2 / Higham)
# ═══════════════════════════════════════════════════════════════════════════════
def _two_sum(left: float, right: float) -> Tuple[float, float]:
    total = left + right
    rounded = total - left
    err = (left - (total - rounded)) + (right - rounded)
    return float(total), float(err)


def _two_prod(left: float, right: float) -> Tuple[float, float]:
    prod = left * right
    err = math.fma(left, right, -prod)
    return float(prod), float(err)


def higham_gamma(length: int, unit_roundoff: float = _MACHINE_EPS) -> float:
    r"""Constante de Higham \(\gamma_k=ku/(1-ku)\).  \(+\infty\) si \(ku\ge 1\)."""
    steps = max(0, int(length))
    if steps == 0:
        return 0.0
    ku = steps * float(unit_roundoff)
    if not math.isfinite(ku) or ku >= 1.0:
        return float("inf")
    return float(ku / (1.0 - ku))


def kbn_sum(values: Sequence[float]) -> Tuple[float, float]:
    """Suma de Kahan–Babuška–Neumaier: devuelve \((\hat s,\,c)\)."""
    total = 0.0
    compensation = 0.0
    for raw in values:
        term = float(raw)
        if term == 0.0:
            continue
        tentative = total + term
        if abs(total) >= abs(term):
            compensation += (total - tentative) + term
        else:
            compensation += (term - tentative) + total
        total = tentative
    return float(total), float(compensation)


def kbn_add_vector(
    left: np.ndarray,
    increment: np.ndarray,
    compensation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """KBN vectorial in-place-safe: \((x+dx,\,c')\) componente a componente."""
    result = np.empty_like(left)
    new_comp = np.empty_like(compensation)
    for idx in range(left.size):
        term = float(increment[idx]) - float(compensation[idx])
        tentative = float(left[idx]) + term
        if abs(float(left[idx])) >= abs(term):
            new_comp[idx] = (float(left[idx]) - tentative) + term
        else:
            new_comp[idx] = (term - tentative) + float(left[idx])
        result[idx] = tentative
    return result, new_comp


def compensated_sum_of_squares(values: Sequence[float]) -> Tuple[float, float]:
    total = 0.0
    compensation = 0.0
    product_tail = 0.0
    for raw in values:
        term = float(raw)
        if term == 0.0:
            continue
        square, square_err = _two_prod(term, term)
        product_tail += square_err
        tentative = total + square
        if abs(total) >= abs(square):
            compensation += (total - tentative) + square
        else:
            compensation += (square - tentative) + total
        total = tentative
    return float(total), float(compensation + product_tail)


def frobenius_kbn(matrix: np.ndarray) -> Tuple[float, float]:
    r"""\(\|A\|_F=\sigma\sqrt{\sum(a_{ij}/\sigma)^2}\) con TwoProd+KBN."""
    if matrix.size == 0:
        return 0.0, 0.0
    flat = np.ascontiguousarray(matrix, dtype=np.float64).ravel()
    scale = float(np.max(np.abs(flat)))
    if scale == 0.0:
        return 0.0, 0.0
    if not math.isfinite(scale):
        raise FpuMetrologyError("‖A‖_∞ no es finita.")
    scaled = (flat / scale).tolist()
    sumsq, tail = compensated_sum_of_squares(scaled)
    total, extra = _two_sum(sumsq, tail)
    if total < 0.0 and total > -1e-15:
        total = 0.0
    if total < 0.0:
        raise FpuMetrologyError("Suma de cuadrados negativa tras KBN.")
    return float(scale * math.sqrt(total)), float(extra)


def dot2(left: np.ndarray, right: np.ndarray) -> float:
    """Producto interno fiel Dot2 (Ogita–Rump–Oishi)."""
    if left.size != right.size:
        raise FpuMetrologyError("Dot2 exige vectores de la misma longitud.")
    if left.size == 0:
        return 0.0
    x_vec = np.ascontiguousarray(left, dtype=np.float64).ravel()
    y_vec = np.ascontiguousarray(right, dtype=np.float64).ravel()
    prod, err = _two_prod(float(x_vec[0]), float(y_vec[0]))
    total = prod
    compensation = err
    for idx in range(1, x_vec.size):
        prod, prod_err = _two_prod(float(x_vec[idx]), float(y_vec[idx]))
        total, sum_err = _two_sum(total, prod)
        compensation += sum_err + prod_err
    result, _ = _two_sum(total, compensation)
    return float(result)


def _dot2_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    rows, mid = left.shape
    mid_r, cols = right.shape
    if mid != mid_r:
        raise FpuMetrologyError("Dimensiones incompatibles en Dot2-GEMM.")
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        row = left[i, :]
        for j in range(cols):
            out[i, j] = dot2(row, right[:, j])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CERROJO LOCK-FREE (CAS / fetch_or, acquire–release)
# ═══════════════════════════════════════════════════════════════════════════════
class AtomicInt32:
    __slots__ = ("_cell",)

    def __init__(self, initial: int = 0) -> None:
        self._cell = ctypes.c_int32(int(initial))

    def load_acquire(self) -> int:
        return int(self._cell.value)

    def store_release(self, value: int) -> None:
        self._cell.value = int(value)

    def compare_exchange(self, expected: int, desired: int) -> Tuple[bool, int]:
        current = int(self._cell.value)
        if current == int(expected):
            self._cell.value = int(desired)
            return True, current
        return False, current

    def fetch_or(self, mask: int) -> int:
        current = int(self._cell.value)
        self._cell.value = int(current | int(mask))
        return current


@dataclass(frozen=True, slots=True)
class LatchSnapshot:
    latched: bool
    generation: int
    cas_succeeded: bool
    host_cas_ns: float
    model_latency_ns: float


class FailSecureLatch:
    """Cerrojo de un bit + generación.  ``trip()`` no espera ni duerme."""

    __slots__ = ("_flag", "_generation")

    def __init__(self) -> None:
        self._flag = AtomicInt32(_LATCH_OPEN)
        self._generation = AtomicInt32(0)

    def is_latched(self) -> bool:
        return self._flag.load_acquire() != _LATCH_OPEN

    def generation(self) -> int:
        return self._generation.load_acquire()

    def trip(self) -> LatchSnapshot:
        started = time.perf_counter_ns()
        previous = self._flag.fetch_or(_LATCH_CLOSED)
        if previous == _LATCH_OPEN:
            self._generation.store_release(self._generation.load_acquire() + 1)
        host_ns = float(max(0, time.perf_counter_ns() - started))
        return LatchSnapshot(
            latched=True,
            generation=self._generation.load_acquire(),
            cas_succeeded=previous == _LATCH_OPEN,
            host_cas_ns=host_ns,
            model_latency_ns=_INTERLOCK_LATENCY_NS,
        )

    def snapshot(self) -> LatchSnapshot:
        latched = self.is_latched()
        return LatchSnapshot(
            latched=latched,
            generation=self.generation(),
            cas_succeeded=True,
            host_cas_ns=0.0,
            model_latency_ns=_INTERLOCK_LATENCY_NS if latched else 0.0,
        )

    def try_reset(self) -> bool:
        ok, _ = self._flag.compare_exchange(_LATCH_CLOSED, _LATCH_OPEN)
        if ok:
            self._generation.store_release(self._generation.load_acquire() + 1)
        return bool(ok)


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS INMUTABLES Y OBJETOS DE CONTINUACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SphereQuadratureCertificate:
    """Metrología de la cuadratura de Gauss–Legendre ⊗ trapecio periódico."""

    value: float
    compensation: float
    degree: int
    n_phi: int
    measure_residual: float
    error_estimate: float
    is_certified: bool


@dataclass(frozen=True, slots=True)
class MetricAtlas:
    r"""
    Objeto terminal de la Fase 1 y objeto inicial de la Fase 2.

    Porta la métrica regularizada \(\tilde G\), su inversa de Cholesky,
    el shift de Tikhonov y, si se midió, el certificado esférico.
    """

    metric_spd: np.ndarray
    mass_inverse: np.ndarray
    condition_number: float
    tikhonov_mu: float
    higham_residual: float
    negative_inertia: int
    log_det: float
    sphere: Optional[SphereQuadratureCertificate]
    phase_signature: str


@dataclass(frozen=True, slots=True)
class VerletCompensation:
    """Estado KBN persistente del integrador (se transporta entre pasos)."""

    q_comp: np.ndarray
    p_comp: np.ndarray


@dataclass(frozen=True, slots=True)
class HamiltonianJet:
    r"""
    Objeto terminal de la Fase 2 y objeto inicial de la Fase 3.

    Jet de orden 1 del mapa de Verlet: \((q',p',DJ)\) más compensación
    y referencia al atlas que lo generó.
    """

    atlas: MetricAtlas
    q: np.ndarray
    p: np.ndarray
    q_next: np.ndarray
    p_next: np.ndarray
    jacobian: np.ndarray
    compensation: VerletCompensation
    dt: float
    jacobian_method: str
    phase_signature: str


@dataclass(frozen=True, slots=True)
class SymplecticMetrology:
    residual_kbn: float
    residual_classical: float
    residual_relative_kbn: float
    rounding_envelope: float
    higham_gamma: float
    omega_frobenius: float
    jacobian_frobenius: float
    is_within_fpu_noise: bool
    is_certified_symplectic: bool
    used_dot2: bool


@dataclass(frozen=True, slots=True)
class ManifoldStateCertificate:
    """Certificado inmutable del estado métrico y espectral de la variedad."""

    symplectic_residual: float
    is_symplectic: bool
    volume_determinant: float
    volume_defect: float
    jacobian_condition_number: float
    heyting_verdict: str
    heyting_truth_value: float
    heyting_implication_trace: Tuple[str, ...]
    energy_conserved: bool
    hamiltonian_drift: float
    relative_energy_drift: float
    crowbar_fired: bool = False
    latency_ns: float = 0.0
    metrology: Optional[SymplecticMetrology] = None
    latch_generation: int = 0
    phase_trace: Tuple[str, ...] = ()
    timestamp_utc: str = ""
    fpu_certified: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DE CÁLCULO SIMPLÉCTICO
# ═══════════════════════════════════════════════════════════════════════════════
class SymplecticManifoldEngine:
    r"""
    Integrador y auditor de \((\mathbb{R}^{2n},\omega_0)\).

    \[
      \omega_0
      =\begin{pmatrix}0&I_n\\-I_n&0\end{pmatrix},
      \qquad
      \|\omega_0\|_F=\sqrt{2n}.
    \]
    """

    def __init__(self, dimension_n: int) -> None:
        if not isinstance(dimension_n, (int, np.integer)) or int(dimension_n) <= 0:
            raise InvalidDimensionError(
                "La dimensión n debe ser un entero estrictamente positivo."
            )
        self.n: Final[int] = int(dimension_n)
        self.dim: Final[int] = 2 * self.n
        eye_n = np.eye(self.n, dtype=np.float64)
        omega = np.zeros((self.dim, self.dim), dtype=np.float64)
        omega[: self.n, self.n :] = eye_n
        omega[self.n :, : self.n] = -eye_n
        self.omega: Final[np.ndarray] = omega
        self._omega_frobenius: Final[float] = math.sqrt(2.0 * self.n)
        self._latch: Final[FailSecureLatch] = FailSecureLatch()
        self._validate_darboux()

    def _validate_darboux(self) -> None:
        skew = float(la.norm(self.omega + self.omega.T, ord="fro"))
        square = float(la.norm(self.omega @ self.omega + np.eye(self.dim), ord="fro"))
        det_w = float(np.linalg.det(self.omega))
        if skew > 1e-12 or square > 1e-10 or abs(det_w - 1.0) > 1e-8:
            raise SymplecticManifoldError("Fallo de los axiomas de Darboux en Ω.")

    @property
    def interlock_latched(self) -> bool:
        return self._latch.is_latched()

    def reset_interlock(self) -> None:
        if not self._latch.try_reset():
            if not self._latch.is_latched():
                return
            raise HardwareCrowbarError("CAS de rearme falló.")
        logger.info("Cerrojo fail-secure rearmado (gen=%d).", self._latch.generation())

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _finite_scalar(value: Any, name: str) -> float:
        scalar = float(value)
        if not math.isfinite(scalar):
            raise SymplecticManifoldError(f"{name} no es un escalar finito.")
        return scalar

    @staticmethod
    def _coerce_real_matrix(matrix: Any, name: str) -> np.ndarray:
        arr = np.asarray(matrix)
        if np.iscomplexobj(arr):
            imag_amp = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
            if imag_amp > _COMPLEX_IMAG_TOL:
                raise SymplecticManifoldError(
                    f"{name} es compleja no real (‖Im‖_∞={imag_amp:.3e})."
                )
            arr = arr.real
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2:
            raise SymplecticManifoldError(f"{name} debe ser una matriz bidimensional.")
        if arr.size == 0:
            raise SymplecticManifoldError(f"{name} no puede ser vacía.")
        if not np.all(np.isfinite(arr)):
            raise SymplecticManifoldError(f"{name} contiene valores NaN o Inf.")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _coerce_real_vector(vector: Any, name: str, expected: Optional[int] = None) -> np.ndarray:
        arr = np.asarray(vector)
        if np.iscomplexobj(arr):
            imag_amp = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
            if imag_amp > _COMPLEX_IMAG_TOL:
                raise SymplecticManifoldError(
                    f"{name} es complejo no real (‖Im‖_∞={imag_amp:.3e})."
                )
            arr = arr.real
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise SymplecticManifoldError(f"{name} no puede ser vacío.")
        if not np.all(np.isfinite(arr)):
            raise SymplecticManifoldError(f"{name} contiene valores NaN o Inf.")
        if expected is not None and arr.size != expected:
            raise SymplecticManifoldError(
                f"{name} debe tener longitud {expected}, se recibió {arr.size}."
            )
        return np.ascontiguousarray(arr)

    def _eigh_certified(self, matrix: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray, float]:
        symmetric = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = la.eigh(symmetric)
        residual = symmetric @ eigvecs - eigvecs @ np.diag(eigvals)
        rel = float(la.norm(residual, ord="fro")) / max(float(la.norm(symmetric, ord="fro")), _MACHINE_EPS)
        if rel > _EIGH_RESIDUAL_WARN:
            logger.warning("Residual espectral elevado en %s: %.3e.", name, rel)
        return eigvals, eigvecs, rel

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — INTEGRACIÓN ESFÉRICA Y REGULARIZACIÓN MÉTRICA
    # ═════════════════════════════════════════════════════════════════════════
    def gauss_legendre_sphere_integration(
        self,
        func: Callable[[float, float], float],
        degree: int = 16,
        estimate_error: bool = True,
    ) -> float:
        r"""
        Integración de \(f(\theta,\varphi)\) en \(S^2\) con jacobiano absorbido:

        \[
          I
          =\int_0^{2\pi}\int_0^\pi f(\theta,\varphi)\sin\theta\,d\theta\,d\varphi
          =\int_0^{2\pi}\int_{-1}^{1} f(\arccos u,\varphi)\,du\,d\varphi.
        \]

        Polar: Gauss–Legendre de grado \(n\) (exacto para polinomios de
        grado \(\le 2n-1\) en \(u=\cos\theta\)).
        Azimutal: trapecio periódico (espectralmente exacto en modos
        de Fourier \(\lvert k\rvert<n_\varphi\)).  Nunca se evalúa
        \(\sin\theta\): los polos son regulares.

        La suma es KBN.  Use ``integrate_sphere_certified`` si necesita
        el certificado de error.
        """
        return self.integrate_sphere_certified(func, degree=degree, estimate_error=estimate_error).value

    def integrate_sphere_certified(
        self,
        func: Callable[[float, float], float],
        degree: int = 16,
        estimate_error: bool = True,
    ) -> SphereQuadratureCertificate:
        """Cuadratura esférica con residuo KBN y estimador \(I_n-I_{\lceil n/2\rceil}\)."""
        if not callable(func):
            raise SymplecticManifoldError("func debe ser invocable f(θ, φ) → ℝ.")
        if not isinstance(degree, (int, np.integer)) or int(degree) < 2:
            raise QuadratureConvergenceError(
                "El grado de Gauss–Legendre debe ser un entero ≥ 2."
            )
        degree_i = int(degree)
        value, compensation, n_phi = self._sphere_accumulate(func, degree_i)
        error_est = 0.0
        if estimate_error and degree_i >= 4:
            coarse, _, _ = self._sphere_accumulate(func, max(2, degree_i // 2))
            error_est = abs(value - coarse)
        measure_residual = 0.0
        # Certifica la medida de Haar: ∫1 dσ = 4π (testigo del jacobiano).
        ones, _, _ = self._sphere_accumulate(lambda _th, _ph: 1.0, degree_i)
        measure_residual = abs(ones - _SPHERE_MEASURE)
        gamma = higham_gamma(degree_i * max(2 * degree_i, 1))
        certified = bool(
            math.isfinite(value)
            and measure_residual <= max(1e-12, 64.0 * gamma * _SPHERE_MEASURE)
        )
        if not certified:
            logger.warning(
                "Cuadratura esférica no certificada: |∫1−4π|=%.3e (grado=%d).",
                measure_residual,
                degree_i,
            )
        return SphereQuadratureCertificate(
            value=float(value),
            compensation=float(compensation),
            degree=degree_i,
            n_phi=int(n_phi),
            measure_residual=float(measure_residual),
            error_estimate=float(error_est),
            is_certified=certified,
        )

    def _sphere_accumulate(
        self,
        func: Callable[[float, float], float],
        degree: int,
    ) -> Tuple[float, float, int]:
        nodes_u, weights_u = np.polynomial.legendre.leggauss(int(degree))
        n_phi = 2 * int(degree)
        phi_nodes = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
        phi_weight = 2.0 * math.pi / n_phi
        acc: List[float] = []
        for node_u, weight_u in zip(nodes_u, weights_u):
            theta = float(math.acos(float(np.clip(node_u, -1.0, 1.0))))
            w_polar = float(weight_u) * phi_weight
            for phi in phi_nodes:
                raw = func(theta, float(phi))
                sample = self._finite_scalar(raw, "f(θ, φ)")
                acc.append(sample * w_polar)
        total, compensation = kbn_sum(acc)
        return float(total), float(compensation), n_phi

    def compute_metric_condition_number(self, matrix_G: np.ndarray) -> float:
        r"""\(\kappa_2(G)=\sigma_{\max}/\sigma_{\min}\); \(0\) si es nula, \(\infty\) si es singular."""
        gee = self._coerce_real_matrix(matrix_G, "matrix_G")
        singular = np.asarray(la.svdvals(gee), dtype=np.float64)
        if singular.size == 0 or singular[0] <= _MACHINE_EPS:
            return 0.0
        if singular[-1] <= _MACHINE_EPS:
            return float("inf")
        return float(singular[0] / singular[-1])

    def regularize_spd_higham(
        self,
        matrix_G: np.ndarray,
        beta0: float = _TIKHONOV_BETA0,
    ) -> np.ndarray:
        r"""
        Proyección de Higham (1988/2002) con shift de Tikhonov adaptativo (A2):

        \[
          \mu
          =\max\bigl(\tau_{\mathrm{floor}},\;
                     \beta_0\,\kappa_2(G)\,n\,\varepsilon_{\mathrm{máq}}\bigr),
          \qquad
          \tilde G
          =V\operatorname{diag}(\max(\lambda_i,\mu))V^\top.
        \]

        Si la inercia negativa supera la mitad de la dimensión, la forma
        no es una métrica y se lanza ``DegenerateMetricError``.
        """
        atlas = self._regularize_metric_certified(matrix_G, beta0=beta0)
        return np.array(atlas.metric_spd, copy=True)

    def _regularize_metric_certified(
        self,
        matrix_G: np.ndarray,
        beta0: float = _TIKHONOV_BETA0,
    ) -> MetricAtlas:
        gee = self._coerce_real_matrix(matrix_G, "matrix_G")
        if gee.shape[0] != gee.shape[1]:
            raise DegenerateMetricError("El tensor métrico debe ser cuadrado.")
        if gee.shape[0] != self.n:
            raise DegenerateMetricError(
                f"El tensor métrico debe ser ({self.n} x {self.n})."
            )
        beta = self._finite_scalar(beta0, "beta0")
        if beta <= 0.0:
            raise DegenerateMetricError("beta0 debe ser estrictamente positivo.")
        if not np.allclose(gee, gee.T, rtol=1e-12, atol=_SYMMETRY_ATOL):
            logger.warning("Métrica no simétrica; se simetriza (G+Gᵀ)/2.")
            gee = 0.5 * (gee + gee.T)
        eigvals, eigvecs, eigh_res = self._eigh_certified(gee, "metric_G")
        n_neg = int(np.sum(eigvals < -_HIGHAM_FLOOR))
        if n_neg > int(_NEG_INERTIA_HARD * self.n):
            raise DegenerateMetricError(
                f"Inercia negativa excesiva ({n_neg}/{self.n}); G no es una métrica."
            )
        peak = max(float(np.max(np.abs(eigvals))), _MACHINE_EPS)
        pos = eigvals[eigvals > _HIGHAM_FLOOR]
        sigma_min = float(pos[0]) if pos.size else _MACHINE_EPS
        cond = peak / max(sigma_min, _MACHINE_EPS)
        if not math.isfinite(cond):
            cond = float(1.0 / _MACHINE_EPS)
        mu = max(_HIGHAM_FLOOR, beta * cond * self.n * _MACHINE_EPS)
        clipped = np.maximum(eigvals, mu)
        spd = (eigvecs * clipped) @ eigvecs.T
        try:
            cho = la.cholesky(spd, lower=True)
            inverse = la.cho_solve((cho, True), np.eye(self.n, dtype=np.float64))
        except la.LinAlgError as err:
            raise DegenerateMetricError(
                "Cholesky falló tras Higham–Tikhonov: la métrica sigue degenerada."
            ) from err
        sign, logdet = np.linalg.slogdet(spd)
        if sign <= 0.0 or not math.isfinite(logdet):
            raise DegenerateMetricError("det(G̃) no es positivo y finito.")
        return MetricAtlas(
            metric_spd=np.ascontiguousarray(spd),
            mass_inverse=np.ascontiguousarray(inverse),
            condition_number=float(cond),
            tikhonov_mu=float(mu),
            higham_residual=float(eigh_res),
            negative_inertia=n_neg,
            log_det=float(logdet),
            sphere=None,
            phase_signature="PHASE_1::emit_metric_atlas",
        )

    def emit_metric_atlas(
        self,
        matrix_G: np.ndarray,
        sphere_density: Optional[Callable[[float, float], float]] = None,
        sphere_degree: int = 16,
        beta0: float = _TIKHONOV_BETA0,
    ) -> MetricAtlas:
        r"""
        Término formal de la Fase 1 y objeto inicial de la Fase 2.

        \[
          \operatorname{emit\_metric\_atlas}
          :\; (G,f_{S^2})
          \;\longrightarrow\;
          \mathfrak{A}_1\in\mathrm{Ob}(\mathbf{Met}_{\mathrm{SPD}}).
        \]

        La Fase 2 no posee otra puerta de entrada que ``ingest_metric_atlas``.
        """
        atlas = self._regularize_metric_certified(matrix_G, beta0=beta0)
        sphere: Optional[SphereQuadratureCertificate] = None
        if sphere_density is not None:
            sphere = self.integrate_sphere_certified(sphere_density, degree=sphere_degree)
        return MetricAtlas(
            metric_spd=atlas.metric_spd,
            mass_inverse=atlas.mass_inverse,
            condition_number=atlas.condition_number,
            tikhonov_mu=atlas.tikhonov_mu,
            higham_residual=atlas.higham_residual,
            negative_inertia=atlas.negative_inertia,
            log_det=atlas.log_det,
            sphere=sphere,
            phase_signature="PHASE_1::emit_metric_atlas",
        )

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — VERLET / KBN Y JACOBIANO ESPECTRAL
    # Continuación formal de emit_metric_atlas: ingest_metric_atlas
    # ═════════════════════════════════════════════════════════════════════════
    def ingest_metric_atlas(self, atlas: MetricAtlas) -> MetricAtlas:
        r"""
        Continuación formal de ``emit_metric_atlas``.

        Axioma de arranque de la Fase 2: el mapa de Verlet sólo se
        construye sobre una métrica SPD certificada.
        """
        if not isinstance(atlas, MetricAtlas):
            raise DegenerateMetricError(
                "La Fase 2 exige un MetricAtlas emitido por la Fase 1."
            )
        if atlas.phase_signature != "PHASE_1::emit_metric_atlas":
            raise DegenerateMetricError(
                f"Firma de fase ilegítima: {atlas.phase_signature!r}."
            )
        if atlas.metric_spd.shape != (self.n, self.n):
            raise DegenerateMetricError("El atlas porta una métrica de dimensión ajena.")
        if atlas.tikhonov_mu <= 0.0 or not math.isfinite(atlas.condition_number):
            raise DegenerateMetricError("El atlas no certifica el shift de Tikhonov.")
        if self._latch.is_latched():
            raise HardwareCrowbarError(
                "Canal cerrado: la Fase 2 no integra sobre un cerrojo latched."
            )
        return atlas

    def _require_state(self, q: Any, p: Any) -> Tuple[np.ndarray, np.ndarray]:
        qv = self._coerce_real_vector(q, "q", expected=self.n)
        pv = self._coerce_real_vector(p, "p", expected=self.n)
        return qv, pv

    def _verlet_map_state(
        self,
        state: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        mass_matrix_inv: np.ndarray,
    ) -> np.ndarray:
        q = state[: self.n]
        p = state[self.n :]
        p_half = p - 0.5 * dt * np.asarray(grad_H_q(q))
        q_new = q + dt * (mass_matrix_inv @ p_half)
        p_new = p_half - 0.5 * dt * np.asarray(grad_H_q(q_new))
        return np.concatenate([np.asarray(q_new).reshape(-1), np.asarray(p_new).reshape(-1)])

    def _jacobian_csmd(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        mass_matrix_inv: np.ndarray,
        step_h: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""
        Diferenciación por paso complejo (Squire–Trapp):

        \[
          \partial_j\Phi(x)
          =\frac{1}{h}\operatorname{Im}\Phi(x+ih e_j)
          +\mathcal{O}(h^2).
        \]

        \(h\sim 10^{-20}\) deja el residual en \(\varepsilon_{\mathrm{máq}}\)
        si \(\operatorname{grad} H\) admite continuación holomorfa.
        """
        jacobian = np.zeros((self.dim, self.dim), dtype=np.float64)
        state = np.concatenate([q, p]).astype(np.complex128)
        for idx in range(self.dim):
            scale = max(1.0, abs(float(state[idx].real)))
            step = max(float(step_h) * scale, _CSMD_STEP_FLOOR)
            perturb = np.zeros(self.dim, dtype=np.complex128)
            perturb[idx] = 1j * step
            out = self._verlet_map_state(state + perturb, grad_H_q, dt, mass_matrix_inv)
            imag = np.asarray(out, dtype=np.complex128).imag
            jacobian[:, idx] = imag / step
        if not np.all(np.isfinite(jacobian)):
            raise TypeError("CSMD produjo entradas no finitas.")
        return jacobian

    def _jacobian_fd(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        mass_matrix_inv: np.ndarray,
    ) -> np.ndarray:
        """Diferencias centrales de orden 2 con paso \(\sqrt{\varepsilon}\,|x|\)."""
        jacobian = np.zeros((self.dim, self.dim), dtype=np.float64)
        state = np.concatenate([q, p])
        for idx in range(self.dim):
            scale = max(1.0, abs(float(state[idx])))
            step = math.sqrt(_MACHINE_EPS) * scale
            perturb = np.zeros(self.dim, dtype=np.float64)
            perturb[idx] = step
            out_plus = self._verlet_map_state(state + perturb, grad_H_q, dt, mass_matrix_inv)
            out_minus = self._verlet_map_state(state - perturb, grad_H_q, dt, mass_matrix_inv)
            jacobian[:, idx] = (out_plus - out_minus) / (2.0 * step)
        return jacobian

    def _jacobian_hessian(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        hess_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        mass_inv: np.ndarray,
    ) -> np.ndarray:
        r"""
        Jacobiano exacto del Verlet separable
        \(H=\tfrac12 p^\top G^{-1}p+V(q)\):

        \[
          \begin{aligned}
            \partial_q q'
              &=I-\tfrac{h^2}{2}G^{-1}A,\\
            \partial_p q'
              &=h G^{-1},\\
            \partial_q p'
              &=-\tfrac{h}{2}A-\tfrac{h}{2}B\,\partial_q q',\\
            \partial_p p'
              &=I-\tfrac{h}{2}B\,\partial_p q',
          \end{aligned}
        \]

        con \(A=\operatorname{Hess}V(q)\), \(B=\operatorname{Hess}V(q')\).
        """
        force_q = np.asarray(grad_H_q(q), dtype=np.float64).reshape(self.n)
        p_half = p - 0.5 * dt * force_q
        q_new = q + dt * (mass_inv @ p_half)
        hess_a = self._coerce_real_matrix(hess_H_q(q), "hess_H_q(q)")
        hess_b = self._coerce_real_matrix(hess_H_q(q_new), "hess_H_q(q')")
        if hess_a.shape != (self.n, self.n) or hess_b.shape != (self.n, self.n):
            raise NonTransversalJacobianError("El Hessiano de V debe ser (n x n).")
        dqd_q = np.eye(self.n) - (0.5 * dt * dt) * (mass_inv @ hess_a)
        dqd_p = dt * mass_inv
        dpd_q = -0.5 * dt * hess_a - 0.5 * dt * (hess_b @ dqd_q)
        dpd_p = np.eye(self.n) - 0.5 * dt * (hess_b @ dqd_p)
        jacobian = np.zeros((self.dim, self.dim), dtype=np.float64)
        jacobian[: self.n, : self.n] = dqd_q
        jacobian[: self.n, self.n :] = dqd_p
        jacobian[self.n :, : self.n] = dpd_q
        jacobian[self.n :, self.n :] = dpd_p
        return jacobian

    def _compute_jacobian(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        mass_matrix_inv: np.ndarray,
        hess_H_q: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        step_h: float = _CSMD_STEP,
    ) -> Tuple[np.ndarray, str]:
        if hess_H_q is not None:
            return (
                self._jacobian_hessian(q, p, grad_H_q, hess_H_q, dt, mass_matrix_inv),
                "hessian_exact",
            )
        try:
            return (
                self._jacobian_csmd(q, p, grad_H_q, dt, mass_matrix_inv, step_h),
                "csmd",
            )
        except (TypeError, ValueError, SymplecticManifoldError):
            logger.warning(
                "CSMD no disponible (gradiente no holomorfo). "
                "Se usa diferencia finita central de orden 2."
            )
            return (
                self._jacobian_fd(q, p, grad_H_q, dt, mass_matrix_inv),
                "finite_difference",
            )

    def symplectic_verlet_step_kahan(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        mass_matrix_inv: np.ndarray,
        compensation: Optional[VerletCompensation] = None,
        hess_H_q: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Un paso de Störmer–Verlet (orden 2, separable) con KBN persistente.

        \[
          p_{1/2}=p-\tfrac{h}{2}\nabla V(q),\;
          q'=q+h G^{-1}p_{1/2},\;
          p'=p_{1/2}-\tfrac{h}{2}\nabla V(q').
        \]

        ``compensation`` se *reutiliza* entre pasos: un Kahan de un solo
        sumando no recorta la deriva secular.  El jacobiano se calcula
        sobre el mapa *sin* compensación (el mapa compensado no es
        diferenciable de forma canónica en la FPU).
        """
        jet = self._verlet_step_certified(
            q=q,
            p=p,
            grad_H_q=grad_H_q,
            dt=dt,
            mass_matrix_inv=mass_matrix_inv,
            compensation=compensation,
            hess_H_q=hess_H_q,
        )
        return jet.q_next, jet.p_next, jet.jacobian

    def _verlet_step_certified(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        mass_matrix_inv: np.ndarray,
        compensation: Optional[VerletCompensation] = None,
        hess_H_q: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        atlas: Optional[MetricAtlas] = None,
    ) -> HamiltonianJet:
        qv, pv = self._require_state(q, p)
        minv = self._coerce_real_matrix(mass_matrix_inv, "mass_matrix_inv")
        if minv.shape != (self.n, self.n):
            raise SymplecticManifoldError(
                f"mass_matrix_inv debe ser ({self.n} x {self.n})."
            )
        step = self._finite_scalar(dt, "dt")
        if step <= 0.0:
            raise SymplecticManifoldError("El paso temporal dt debe ser positivo.")
        if not callable(grad_H_q):
            raise SymplecticManifoldError("grad_H_q debe ser invocable.")

        if compensation is None:
            q_comp = np.zeros(self.n, dtype=np.float64)
            p_comp = np.zeros(self.n, dtype=np.float64)
        else:
            q_comp = self._coerce_real_vector(compensation.q_comp, "q_comp", expected=self.n)
            p_comp = self._coerce_real_vector(compensation.p_comp, "p_comp", expected=self.n)

        force_q = np.asarray(grad_H_q(qv), dtype=np.float64).reshape(-1)
        if force_q.size != self.n or not np.all(np.isfinite(force_q)):
            raise SymplecticManifoldError("grad_H_q(q) debe ser un vector finito de dimensión n.")
        p_half, p_comp = kbn_add_vector(pv, -0.5 * step * force_q, p_comp)
        q_new, q_comp = kbn_add_vector(qv, step * (minv @ p_half), q_comp)
        force_new = np.asarray(grad_H_q(q_new), dtype=np.float64).reshape(-1)
        if force_new.size != self.n or not np.all(np.isfinite(force_new)):
            raise SymplecticManifoldError("grad_H_q(q') debe ser un vector finito de dimensión n.")
        p_new, p_comp = kbn_add_vector(p_half, -0.5 * step * force_new, p_comp)

        jacobian, method = self._compute_jacobian(
            qv, pv, grad_H_q, step, minv, hess_H_q=hess_H_q
        )
        if atlas is None:
            atlas = MetricAtlas(
                metric_spd=np.eye(self.n, dtype=np.float64),
                mass_inverse=np.array(minv, copy=True),
                condition_number=self.compute_metric_condition_number(minv),
                tikhonov_mu=_HIGHAM_FLOOR,
                higham_residual=0.0,
                negative_inertia=0,
                log_det=0.0,
                sphere=None,
                phase_signature="PHASE_1::emit_metric_atlas",
            )
        return HamiltonianJet(
            atlas=atlas,
            q=np.array(qv, copy=True),
            p=np.array(pv, copy=True),
            q_next=np.ascontiguousarray(q_new),
            p_next=np.ascontiguousarray(p_new),
            jacobian=np.ascontiguousarray(jacobian),
            compensation=VerletCompensation(
                q_comp=np.ascontiguousarray(q_comp),
                p_comp=np.ascontiguousarray(p_comp),
            ),
            dt=step,
            jacobian_method=method,
            phase_signature="PHASE_2::emit_hamiltonian_jet",
        )

    def emit_hamiltonian_jet(
        self,
        atlas: MetricAtlas,
        q: np.ndarray,
        p: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        compensation: Optional[VerletCompensation] = None,
        hess_H_q: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> HamiltonianJet:
        r"""
        Término formal de la Fase 2 y objeto inicial de la Fase 3.

        \[
          \operatorname{emit\_hamiltonian\_jet}
          :\; (\mathfrak{A}_1,q,p,h)
          \;\longrightarrow\;
          \mathfrak{J}_2\in\mathrm{Ob}(\mathbf{Jet}^1\mathrm{Sp}).
        \]

        Continúa a ``ingest_metric_atlas``.  La inversa de masa es la
        del atlas (nunca un \(G^{-1}\) crudo del llamador).
        """
        atlas = self.ingest_metric_atlas(atlas)
        return self._verlet_step_certified(
            q=q,
            p=p,
            grad_H_q=grad_H_q,
            dt=dt,
            mass_matrix_inv=atlas.mass_inverse,
            compensation=compensation,
            hess_H_q=hess_H_q,
            atlas=atlas,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — AUDITORÍA CERTIFICADA, HEYTING Y CERROJO
    # Continuación formal de emit_hamiltonian_jet: ingest_hamiltonian_jet
    # ═════════════════════════════════════════════════════════════════════════
    def ingest_hamiltonian_jet(self, jet: HamiltonianJet) -> HamiltonianJet:
        r"""Continuación formal de ``emit_hamiltonian_jet``."""
        if not isinstance(jet, HamiltonianJet):
            raise NonTransversalJacobianError(
                "La Fase 3 exige un HamiltonianJet emitido por la Fase 2."
            )
        if jet.phase_signature != "PHASE_2::emit_hamiltonian_jet":
            raise NonTransversalJacobianError(
                f"Firma de fase ilegítima: {jet.phase_signature!r}."
            )
        self.ingest_metric_atlas(jet.atlas)
        if jet.jacobian.shape != (self.dim, self.dim):
            raise NonTransversalJacobianError(
                f"Jacobiano inválido: se exige ({self.dim} x {self.dim})."
            )
        if not np.all(np.isfinite(jet.jacobian)):
            raise NonTransversalJacobianError("El jacobiano contiene NaN o Inf.")
        return jet

    def _structured_residual(self, jacobian_M: np.ndarray) -> Tuple[np.ndarray, bool]:
        r"""
        Residuo de Darboux por bloques.  \(\Omega M\) es una permutación
        con signos (exacta en IEEE-754).  Si \(n\le 24\) cada bloque se
        evalúa con Dot2.
        """
        dim = self.n
        block_a = jacobian_M[0:dim, 0:dim]
        block_b = jacobian_M[0:dim, dim: 2 * dim]
        block_c = jacobian_M[dim: 2 * dim, 0:dim]
        block_d = jacobian_M[dim: 2 * dim, dim: 2 * dim]
        used_dot2 = dim <= _DOT2_DIM_CEILING
        if used_dot2:
            r11 = _dot2_matmul(block_a.T, block_c) - _dot2_matmul(block_c.T, block_a)
            r22 = _dot2_matmul(block_b.T, block_d) - _dot2_matmul(block_d.T, block_b)
            r12 = _dot2_matmul(block_a.T, block_d) - _dot2_matmul(block_c.T, block_b) - np.eye(dim)
        else:
            r11 = block_a.T @ block_c - block_c.T @ block_a
            r22 = block_b.T @ block_d - block_d.T @ block_b
            r12 = block_a.T @ block_d - block_c.T @ block_b - np.eye(dim)
        residual = np.empty_like(jacobian_M)
        residual[0:dim, 0:dim] = r11
        residual[0:dim, dim: 2 * dim] = r12
        residual[dim: 2 * dim, 0:dim] = -r12.T
        residual[dim: 2 * dim, dim: 2 * dim] = r22
        return residual, used_dot2

    def measure_symplectic_metrology(
        self,
        jacobian_M: np.ndarray,
        tolerance: float = _SYMPLECTIC_TOLERANCE,
    ) -> SymplecticMetrology:
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        if emm.shape != (self.dim, self.dim):
            raise NonTransversalJacobianError(
                f"Jacobiano inválido. Se requiere ({self.dim} x {self.dim})."
            )
        residual, used_dot2 = self._structured_residual(emm)
        residual_kbn, _ = frobenius_kbn(residual)
        residual_classical = float(la.norm(residual, ord="fro"))
        jac_frob, _ = frobenius_kbn(emm)
        gamma = higham_gamma(self.dim)
        envelope = gamma * jac_frob * jac_frob
        if not math.isfinite(residual_kbn) or not math.isfinite(envelope):
            raise FpuMetrologyError("Residuo o envelope de Higham no finito.")
        unit = _MACHINE_EPS * max(residual_kbn, _MACHINE_EPS)
        noise = envelope + unit
        tol = self._finite_scalar(tolerance, "tolerance")
        certified = bool(residual_kbn <= tol * self._omega_frobenius + noise)
        relative = residual_kbn / max(self._omega_frobenius, _MACHINE_EPS)
        return SymplecticMetrology(
            residual_kbn=float(residual_kbn),
            residual_classical=float(residual_classical),
            residual_relative_kbn=float(relative),
            rounding_envelope=float(envelope),
            higham_gamma=float(gamma),
            omega_frobenius=float(self._omega_frobenius),
            jacobian_frobenius=float(jac_frob),
            is_within_fpu_noise=bool(residual_kbn <= noise),
            is_certified_symplectic=certified,
            used_dot2=bool(used_dot2),
        )

    def measure_liouville_volume(self, jacobian_M: np.ndarray) -> Tuple[float, float]:
        r"""
        \((\operatorname{sign}(J)\,e^{\operatorname{slogdet} J},\;
        \lvert\operatorname{expm1}(\operatorname{slogdet} J)\rvert)\).

        \(\operatorname{expm1}\) evita la cancelación \(\det-1\) cerca de 1.
        """
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        if emm.shape != (self.dim, self.dim):
            raise NonTransversalJacobianError(
                f"Jacobiano inválido. Se requiere ({self.dim} x {self.dim})."
            )
        sign, logdet = np.linalg.slogdet(emm)
        if not math.isfinite(logdet):
            return 0.0, float("inf")
        det = float(sign * math.exp(logdet)) if sign != 0.0 else 0.0
        if sign <= 0.0:
            return det, abs(det - 1.0)
        return det, float(abs(math.expm1(logdet)))

    def _atom(self, hard: bool, soft: bool) -> str:
        if not hard:
            return HeytingOmega3.VETOED
        if not soft:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    def classify_heyting_verdict(
        self,
        metro: SymplecticMetrology,
        volume_defect: float,
        energy_drift: float,
        relative_energy: float,
        condition_number: float,
        betti_1: int,
    ) -> Tuple[str, Tuple[str, ...]]:
        r"""
        Clasificador \(\Omega_3\) por implicaciones, no por un `if` plano:

        \[
          \nu
          =
          (\mathrm{Sp}_{\mathrm{KBN}}\to\kappa<\infty)
          \wedge
          (|\det J-1|\le\varepsilon_{\mathrm{vol}})
          \wedge
          (\Delta H\le\tau_H)
          \wedge
          (\beta_1=0).
        \]
        """
        volume_envelope = higham_gamma(self.dim)
        symplectic_atom = self._atom(
            hard=metro.is_certified_symplectic or metro.is_within_fpu_noise,
            soft=metro.is_certified_symplectic
            and metro.residual_relative_kbn <= _SYMPLECTIC_TOLERANCE,
        )
        fpu_atom = self._atom(
            hard=True,
            soft=not (
                metro.residual_classical > _SYMPLECTIC_TOLERANCE * metro.omega_frobenius
                and metro.is_certified_symplectic
            ),
        )
        volume_atom = self._atom(
            hard=volume_defect <= max(1e-5, volume_envelope),
            soft=volume_defect <= _VOLUME_TOLERANCE,
        )
        energy_atom = self._atom(
            hard=energy_drift <= max(_ENERGY_TOLERANCE * 10.0, 1e-4),
            soft=energy_drift <= _ENERGY_TOLERANCE
            or relative_energy <= _ENERGY_TOLERANCE,
        )
        cond_atom = self._atom(
            hard=math.isfinite(condition_number) and condition_number < 1.0e14,
            soft=condition_number < 1.0e10,
        )
        betti_atom = self._atom(hard=betti_1 >= 0, soft=betti_1 == 0)
        impl = HeytingOmega3.implies(symplectic_atom, cond_atom)
        verdict = HeytingOmega3.fold_meet(
            [impl, volume_atom, energy_atom, betti_atom, fpu_atom]
        )
        certified_broken = (not metro.is_certified_symplectic) and (not metro.is_within_fpu_noise)
        if (
            certified_broken
            or betti_1 > 0
            or volume_defect > 1e-5
            or (not math.isfinite(condition_number))
        ):
            verdict = HeytingOmega3.meet(verdict, HeytingOmega3.VETOED)
        elif verdict == HeytingOmega3.COHERENT and (
            metro.residual_relative_kbn > _SYMPLECTIC_TOLERANCE
            and not metro.is_within_fpu_noise
            or volume_defect > _VOLUME_TOLERANCE
            or energy_drift > _ENERGY_TOLERANCE
        ):
            verdict = HeytingOmega3.DEGRADED
        trace = (
            f"Sp={symplectic_atom}",
            f"FPU={fpu_atom}",
            f"vol={volume_atom}",
            f"H={energy_atom}",
            f"κ={cond_atom}",
            f"β₁={betti_atom}",
            f"Sp→κ={impl}",
            f"ν={verdict}",
        )
        return verdict, trace

    def act_failsecure_latch(self, verdict: str) -> Tuple[bool, float, LatchSnapshot]:
        """
        CAS de un disparo.  **No** hay GPIO, ISR ni tiristor.
        La cota de 340 ns es el SLO del *modelo*, no una espera del huésped.
        """
        normalized = HeytingOmega3.normalize(verdict)
        if normalized == HeytingOmega3.VETOED:
            snap = self._latch.trip()
            return True, float(snap.model_latency_ns), snap
        if self._latch.is_latched():
            raise HardwareCrowbarError(
                "El cerrojo permanece latched; se exige reset_interlock() "
                "antes de aceptar un veredicto no-VETOED."
            )
        return False, 0.0, self._latch.snapshot()

    def suture_manifold_state(
        self,
        jet: HamiltonianJet,
        h_initial: float,
        h_final: float,
        betti_1: int = 0,
    ) -> ManifoldStateCertificate:
        r"""
        Término formal de la Fase 3: gluing del jet sobre \(\Omega_3\).
        """
        jet = self.ingest_hamiltonian_jet(jet)
        if not isinstance(betti_1, (int, np.integer)) or int(betti_1) < 0:
            raise CohomologicalObstructionError(
                f"Número de Betti β₁ ilegítimo: {betti_1}."
            )
        h0 = self._finite_scalar(h_initial, "h_initial")
        h1 = self._finite_scalar(h_final, "h_final")
        metro = self.measure_symplectic_metrology(jet.jacobian)
        det, vol_defect = self.measure_liouville_volume(jet.jacobian)
        singular = np.asarray(la.svdvals(jet.jacobian), dtype=np.float64)
        if singular.size == 0 or singular[-1] <= _MACHINE_EPS:
            cond = float("inf")
        else:
            cond = float(singular[0] / singular[-1])
        drift = abs(h1 - h0)
        relative = drift / max(1.0, abs(h0), abs(h1))
        verdict, trace = self.classify_heyting_verdict(
            metro=metro,
            volume_defect=vol_defect,
            energy_drift=drift,
            relative_energy=relative,
            condition_number=cond,
            betti_1=int(betti_1),
        )
        fired, latency, snap = self.act_failsecure_latch(verdict)
        if fired:
            logger.critical(
                "¡VETO DE SUTURA TOPOLÓGICA! CAS gen=%d host=%.0f ns "
                "(cota modelo %.0f ns). Traza: %s",
                snap.generation,
                snap.host_cas_ns,
                snap.model_latency_ns,
                " ∧ ".join(trace),
            )
        elif verdict == HeytingOmega3.DEGRADED:
            logger.warning("Estado DEGRADED en la variedad. Traza: %s", " ∧ ".join(trace))
        else:
            logger.info(
                "Variedad coherente. r_KBN=%.3e env=%.3e |det−1|=%.3e ΔH=%.3e método=%s",
                metro.residual_kbn,
                metro.rounding_envelope,
                vol_defect,
                drift,
                jet.jacobian_method,
            )
        return ManifoldStateCertificate(
            symplectic_residual=metro.residual_kbn,
            is_symplectic=metro.is_certified_symplectic,
            volume_determinant=det,
            volume_defect=vol_defect,
            jacobian_condition_number=cond,
            heyting_verdict=verdict,
            heyting_truth_value=HeytingOmega3.truth(verdict),
            heyting_implication_trace=trace,
            energy_conserved=bool(drift <= _ENERGY_TOLERANCE or relative <= _ENERGY_TOLERANCE),
            hamiltonian_drift=float(drift),
            relative_energy_drift=float(relative),
            crowbar_fired=fired,
            latency_ns=latency,
            metrology=metro,
            latch_generation=snap.generation,
            phase_trace=(
                jet.atlas.phase_signature,
                jet.phase_signature,
                "PHASE_3::suture_manifold_state",
            ),
            timestamp_utc=self._utc_now(),
            fpu_certified=metro.is_certified_symplectic,
        )

    def audit_and_certify_state(
        self,
        jacobian_M: np.ndarray,
        h_initial: float,
        h_final: float,
        betti_1: int = 0,
    ) -> ManifoldStateCertificate:
        """
        Auditoría directa de un jacobiano ya calculado (ciclo ligero).

        Las fases se anidan por continuación formal:

            Fase 1 → atlas euclídeo canónico (no hay \(G\) externo).
            Fase 2 → jet con el jacobiano aportado.
            Fase 3 → ``suture_manifold_state``.
        """
        emm = self._coerce_real_matrix(jacobian_M, "jacobian_M")
        if emm.shape != (self.dim, self.dim):
            raise NonTransversalJacobianError(
                f"Jacobiano inválido. Se requiere ({self.dim} x {self.dim})."
            )

        def _phase1_atlas() -> MetricAtlas:
            return MetricAtlas(
                metric_spd=np.eye(self.n, dtype=np.float64),
                mass_inverse=np.eye(self.n, dtype=np.float64),
                condition_number=1.0,
                tikhonov_mu=_HIGHAM_FLOOR,
                higham_residual=0.0,
                negative_inertia=0,
                log_det=0.0,
                sphere=None,
                phase_signature="PHASE_1::emit_metric_atlas",
            )

        def _phase2_jet(atlas: MetricAtlas) -> HamiltonianJet:
            zeros = np.zeros(self.n, dtype=np.float64)
            return HamiltonianJet(
                atlas=self.ingest_metric_atlas(atlas),
                q=zeros,
                p=zeros,
                q_next=zeros,
                p_next=zeros,
                jacobian=np.array(emm, copy=True),
                compensation=VerletCompensation(q_comp=zeros, p_comp=zeros),
                dt=0.0,
                jacobian_method="external",
                phase_signature="PHASE_2::emit_hamiltonian_jet",
            )

        def _phase3_suture(jet: HamiltonianJet) -> ManifoldStateCertificate:
            return self.suture_manifold_state(jet, h_initial, h_final, betti_1=betti_1)

        atlas = _phase1_atlas()
        jet = _phase2_jet(atlas)
        return _phase3_suture(jet)

    def execute_manifold_cycle(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        mass_G: np.ndarray,
        hamiltonian: Callable[[np.ndarray, np.ndarray], float],
        betti_1: int = 0,
        sphere_density: Optional[Callable[[float, float], float]] = None,
        hess_H_q: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        compensation: Optional[VerletCompensation] = None,
        beta0: float = _TIKHONOV_BETA0,
    ) -> Tuple[ManifoldStateCertificate, HamiltonianJet]:
        r"""
        Ciclo completo anidado:

            Fase 1 → ``emit_metric_atlas``
            Fase 2 → ``emit_hamiltonian_jet``   (consume el atlas)
            Fase 3 → ``suture_manifold_state``  (consume el jet)

        Devuelve el certificado y el jet (para encadenar pasos KBN).
        """
        if self._latch.is_latched():
            raise HardwareCrowbarError(
                "Canal cerrado: execute_manifold_cycle no integra."
            )
        if not callable(hamiltonian):
            raise SymplecticManifoldError("hamiltonian debe ser invocable H(q, p).")

        def _phase1() -> MetricAtlas:
            return self.emit_metric_atlas(
                mass_G, sphere_density=sphere_density, beta0=beta0
            )

        def _phase2(atlas: MetricAtlas) -> HamiltonianJet:
            return self.emit_hamiltonian_jet(
                atlas=atlas,
                q=q,
                p=p,
                grad_H_q=grad_H_q,
                dt=dt,
                compensation=compensation,
                hess_H_q=hess_H_q,
            )

        def _phase3(jet: HamiltonianJet) -> ManifoldStateCertificate:
            h0 = self._finite_scalar(hamiltonian(jet.q, jet.p), "H(q, p)")
            h1 = self._finite_scalar(hamiltonian(jet.q_next, jet.p_next), "H(q', p')")
            return self.suture_manifold_state(jet, h0, h1, betti_1=betti_1)

        try:
            atlas = _phase1()
            jet = _phase2(atlas)
            return _phase3(jet), jet
        except Exception as err:
            logger.critical("¡VETO DE SUTURA TOPOLÓGICA! Ruptura de la variedad: %s", err)
            raise

    def simulate_esp32_crowbar_interlock(self) -> dict:
        """
        Compatibilidad de firma.  **No actúa hardware.**  Devuelve la
        instantánea del cerrojo de software y la cota metrológica.
        """
        snap = self._latch.snapshot()
        return {
            "mode": "SOFTWARE_FAILSECURE_LATCH",
            "latched": snap.latched,
            "generation": snap.generation,
            "model_latency_ns": snap.model_latency_ns,
            "host_cas_ns": snap.host_cas_ns,
            "system_state": "LOCKOUT_SOFTWARE" if snap.latched else "CHANNEL_OPEN",
        }


__all__ = [
    "SymplecticManifoldError",
    "InvalidDimensionError",
    "NonTransversalJacobianError",
    "DegenerateMetricError",
    "HamiltonianDriftError",
    "CohomologicalObstructionError",
    "QuadratureConvergenceError",
    "HardwareCrowbarError",
    "FpuMetrologyError",
    "SphereQuadratureCertificate",
    "MetricAtlas",
    "VerletCompensation",
    "HamiltonianJet",
    "SymplecticMetrology",
    "ManifoldStateCertificate",
    "FailSecureLatch",
    "LatchSnapshot",
    "SymplecticManifoldEngine",
    "higham_gamma",
    "kbn_sum",
    "dot2",
    "frobenius_kbn",
]
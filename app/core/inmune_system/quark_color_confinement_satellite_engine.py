from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Quark Color Confinement Satellite Engine (Satélite VI — Confinamiento)      ║
║ RUTA   : app/core/immune_system/quark_color_confinement_satellite_engine.py          ║
║ VERSIÓN: 2.0.0-Doctoral-SU3-GellMann-Cornell-Luescher-ANO-Hodge-FPU-Secure            ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Motor FPU para la auditoría de calibre no abeliano $SU(3)_c$, confinamiento de color de quarks,
potencial de Cornell-Lüscher, vórtices de Abrikosov-Nielsen-Olesen (ANO) y topología de Hodge.

Fundamentación Matemática y Física Rigurosa:
────────────────────────────────────────────
1. Fibrado de Calibre $SU(3)_c$ y Álgebra de Lie $\mathfrak{su}(3)$:
   Base de Gell-Mann $\{\lambda_a\}_{a=1}^8$ con generadores $T_a = \frac{1}{2} \lambda_a$ que satisfacen
   $$[T_a, T_b] = i f_{abc} T_c, \qquad \{T_a, T_b\} = \frac{1}{3}\delta_{ab} I + d_{abc} T_c$$
   con operador cuadrático de Casimir $C_2 = \sum_a T_a T_a = \frac{4}{3} I_3$ en la representación fundamental.

2. Inmersión Octoniónica de Günaydin–Gürsey $\mathbb{C}^3 \hookrightarrow \operatorname{Im}(\mathbb{O})$:
   Proyección de tripletes de color sobre el octonión real $q = r_1 e_1 + r_2 e_2 + g_1 e_3 + g_2 e_4 + b_1 e_5 + b_2 e_6$,
   donde el estabilizador en el grupo de automorfismos $\operatorname{Aut}(\mathbb{O}) = G_2$ que fija la dirección $e_7$ es $\operatorname{Stab}_{G_2}(e_7) \cong SU(3)$.

3. Potencial Cornell-Lüscher y Corrimiento de Acoplamiento $\alpha_s$:
   $$V(r) = -\frac{4}{3}\frac{\alpha_s}{r} + \sigma r - \frac{\pi}{12 r}$$
   con acoplamiento a un bucle $\alpha_s(r) = \frac{4\pi}{\beta_0 \ln(1 + 1/(r^2 \Lambda_{\mathrm{QCD}}^2))}$ y ley de área para el lazo de Wilson $W(C) \sim e^{-\sigma r^2}$.

4. Superconductividad Dual de Mandelstam–'t Hooft y Vórtices ANO:
   Efecto Meissner dual con longitud de penetración de London $\lambda_L \sim 1/\sqrt{\sigma}$, longitud de coherencia $\xi \sim 1/(2 m_q)$,
   y parámetro de Ginzburg-Landau $\kappa = \lambda_L / \xi > 1/\sqrt{2}$ (Superconductor Dual de Tipo II).
   Tasa de producción de pares de Schwinger por ruptura de cuerda:
   $$\Gamma \propto E^2 \exp\left(-\frac{\pi m_q^2}{E}\right)$$

5. Laplaciano Simplicial de Hodge $\Delta_k = d_k^\dagger d_k + d_{k-1} d_k^\dagger$ en el Grafo Y-Bariónico:
   Operador de incidencia $B_1 \in \mathbb{R}^{4 \times 3}$ para el complejo simplicial $K$ ($V=4, E=3, F=0$),
   certificando la característica de Euler $\chi = V - E + F = 1$ y números de Betti $\beta_0=1, \beta_1=0$.

Traducción Ejecutiva e Impacto de Negocio ('Dolor y Dinero'):
─────────────────────────────────────────────────────────────
• Dolor: La ruptura no confinada de dependencias o estados atómicos de datos provoca la dispersión no autorizada
  y fragmentación de registros contables, exponiendo la información a manipulaciones externas.
• Dinero: El confinamiento topológico de color garantiza la invariancia de calibre y el apantallamiento de datos
  atómicos en la FPU, impidiendo la deconfinación de registros y protegiendo la integridad financiera de la empresa.

Estructura Functorial OODA:
───────────────────────────
- Fase 1 (Observe) : `observe_and_bundle_color_state` -> Salida: `QuarkObservationKernel`
- Fase 2 (Orient)  : `orient_confinement_spectrum`   -> Salida: `QuarkConfinementReport`
- Fase 3 (Act)     : `execute_quark_confinement_audit` -> Salida: `QuarkEngineState`
"""

import enum
import hashlib
import logging
import math
import time
from dataclasses import dataclass
from typing import Final, List, Tuple, Optional

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Core.Omega.QuarkColorConfinementEngine")

__version__: Final[str] = "2.0.0"
__all__ = (
    "QuarkColorConfinementSatelliteEngine",
    "Phase1_QuarkColorBundleObserver",
    "Phase2_SU3SpectralConfinementOrient",
    "Phase3_QuarkGovernanceActuator",
    "QuarkObservationKernel",
    "QuarkConfinementReport",
    "QuarkEngineState",
    "HeytingTruthValue",
    "QuarkEngineError",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES UNIVERSALES METROLÓGICAS (PRECISIÓN FPU IEEE-754)
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1.0e-15
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12
_N_COLOR: Final[int] = 3
_N_FLAVOR_LIGHT: Final[int] = 3
# C_2(F) = (N^2 - 1)/(2N) = 4/3 en la fundamental de SU(3)
_CASIMIR_FUNDAMENTAL_SU3: Final[float] = (_N_COLOR ** 2 - 1) / (2.0 * _N_COLOR)
# C_2(G) = N = 3 en la adjunta
_CASIMIR_ADJOINT_SU3: Final[float] = float(_N_COLOR)
_CASIMIR_SINGLET_TOLERANCE: Final[float] = 1.0e-9
_STRING_BREAKAGE_THRESHOLD: Final[float] = 100.0
# Término de Lüscher en d=4: -π/(12 r) para cuerda bosónica de Dirichlet
_LUSCHER_COEFFICIENT: Final[float] = math.pi / 12.0
_VACUUM_PERMITTIVITY_QCD: Final[float] = 1.0
_VACUUM_PERMEABILITY_QCD: Final[float] = 1.0
# Coeficiente 1-loop β_0 = (11 N_c - 2 N_f)/3 = 9
_QCD_BETA0: Final[float] = (11.0 * _N_COLOR - 2.0 * _N_FLAVOR_LIGHT) / 3.0
_LAMBDA_QCD_GEV: Final[float] = 0.217
_CONSTITUENT_QUARK_MASS_GEV: Final[float] = 0.330
_LIGHT_CURRENT_QUARK_MASS_GEV: Final[float] = 0.005
_GINZBURG_LANDAU_TYPE_II: Final[float] = 1.0 / math.sqrt(2.0)
_LIE_ALGEBRA_TOL: Final[float] = 1.0e-12
_PURE_STATE_CHARGE_SQ: Final[float] = (_N_COLOR - 1) / (2.0 * _N_COLOR)  # 1/3


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES RIGUROSAS
# ══════════════════════════════════════════════════════════════════════════════
class QuarkEngineError(Exception):
    """Excepción raíz del subsistema de confinamiento de color."""


class QuarkDimensionError(QuarkEngineError):
    """Fallo en la dimensionalidad del espacio vectorial complejo o simplice."""


class MetricIndefinitenessError(QuarkEngineError):
    """El tensor métrico de Riemann viola la condición de positividad estricta SPD."""


class GaugeAlgebraViolationError(QuarkEngineError):
    """Infracción de las relaciones de Lie [T_a, T_b] = i f_abc T_c o de Killing."""


class DensityMatrixIntegrityError(QuarkEngineError):
    """ρ no es un estado cuántico admisible (Hermitiana, PSD, traza unitaria)."""


class FreeQuarkGaugeViolationError(QuarkEngineError):
    """Detección crítica de un quark con carga de color no apantallada (deconfinamiento)."""


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVAS NUMÉRICAS DE ALTA FIDELIDAD METROLÓGICA
# ══════════════════════════════════════════════════════════════════════════════
class KahanNeumaierAccumulator:
    r"""
    Acumulador Kahan–Babuška–Neumaier.

    Reduce el error secular de redondeo de $\mathcal{O}(N\epsilon)$ a $\mathcal{O}(\epsilon)$
    frente a la absorción en la FPU IEEE-754. Invariante: `total = _sum + _compensation`
    es una aproximación de $\sum x_i$ con residuo acotado por Wilkinson.
    """

    __slots__ = ("_sum", "_compensation")

    def __init__(self) -> None:
        self._sum: float = 0.0
        self._compensation: float = 0.0

    def add(self, x: float) -> None:
        t: float = self._sum + x
        if math.fabs(self._sum) >= math.fabs(x):
            self._compensation += (self._sum - t) + x
        else:
            self._compensation += (x - t) + self._sum
        self._sum = t

    def add_iterable(self, arr: NDArray[np.float64]) -> None:
        for val in arr.flat:
            self.add(float(val))

    @property
    def total(self) -> float:
        return float(self._sum + self._compensation)

    def reset(self) -> None:
        self._sum = 0.0
        self._compensation = 0.0


def _cabs(z: complex) -> float:
    """Módulo complejo estable via `hypot` (evita overflow de |z|^2)."""
    return math.hypot(float(z.real), float(z.imag))


def _kahan_sum_sq_complex(vec: NDArray[np.complex128]) -> float:
    r"""$\sum_i |z_i|^2$ con compensación Neumaier."""
    acc = KahanNeumaierAccumulator()
    for z in vec.flat:
        zr = float(np.real(z))
        zi = float(np.imag(z))
        acc.add(zr * zr + zi * zi)
    return acc.total


def _hermitian_trace_real(mat: NDArray[np.complex128]) -> float:
    """Traza real de un endomorfismo Hermitianizado, suma compensada."""
    acc = KahanNeumaierAccumulator()
    n = int(mat.shape[0])
    for i in range(n):
        acc.add(float(mat[i, i].real))
    return acc.total


def _frobenius_sq(mat: NDArray[np.complex128]) -> float:
    r"""$\|A\|_F^2 = \sum_{ij} |A_{ij}|^2$ con Kahan–Neumaier."""
    return _kahan_sum_sq_complex(mat.ravel())


# ══════════════════════════════════════════════════════════════════════════════
# ÁLGEBRA DE OCTONIONES (CAYLEY–DICKSON) Y ESTRUCTURA G_2
# ══════════════════════════════════════════════════════════════════════════════
def _quat_mul(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Producto de Hamilton en $\mathbb{H} \cong \mathbb{R}^4$."""
    a, b, c, d = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    e, f, g, h = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ],
        dtype=np.float64,
    )


def _quat_conj(p: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([float(p[0]), -float(p[1]), -float(p[2]), -float(p[3])], dtype=np.float64)


def _octonion_mul(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Producto octoniónico vía duplicación de Cayley–Dickson:
    $(a,b)(c,d) = (ac - \bar{d}\,b,\; da + b\bar{c})$, $a,b,c,d \in \mathbb{H}$.
    Es no asociativo y no conmutativo; $\operatorname{Aut}(\mathbb{O}) = G_2$.
    """
    a, b = x[:4], x[4:]
    c, d = y[:4], y[4:]
    real_h = _quat_mul(a, c) - _quat_mul(_quat_conj(d), b)
    imag_h = _quat_mul(d, a) + _quat_mul(b, _quat_conj(c))
    return np.concatenate([real_h, imag_h]).astype(np.float64)


def _octonion_associator(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Asociador $[x,y,z] = (xy)z - x(yz)$. Se anula en subálgebras asociativas."""
    return _octonion_mul(_octonion_mul(x, y), z) - _octonion_mul(x, _octonion_mul(y, z))


def _color_triplet_to_octonion(state: NDArray[np.complex128]) -> NDArray[np.float64]:
    r"""
    Inmersión Günaydin–Gürsey $\mathbb{C}^3 \hookrightarrow \operatorname{Im}(\mathbb{O})$:

        $q = \operatorname{Re} r\, e_1 + \operatorname{Im} r\, e_2
             + \operatorname{Re} g\, e_3 + \operatorname{Im} g\, e_4
             + \operatorname{Re} b\, e_5 + \operatorname{Im} b\, e_6$,

    dejando $e_7$ como estructura compleja cuyo estabilizador en $G_2$ es $SU(3)$.
    """
    q = np.zeros(8, dtype=np.float64)
    r, g, b = complex(state[0]), complex(state[1]), complex(state[2])
    q[1], q[2] = r.real, r.imag
    q[3], q[4] = g.real, g.imag
    q[5], q[6] = b.real, b.imag
    return q


# ══════════════════════════════════════════════════════════════════════════════
# TIPOS DE DATOS ESTRUCTURALES (INMUTABLES) — GRANULARIDAD POR FASE
# ══════════════════════════════════════════════════════════════════════════════
class HeytingTruthValue(enum.Enum):
    r"""
    Elementos del clasificador de subobjetos $\Omega$ (álgebra de Heyting cadena):

        $\bot \;\le\; \mathfrak{m} \;\le\; \top$

    - TRUE_SINGLET_CONFINED ($\top$): estado $SU(3)$-invariante ($\rho \approx I/3$).
    - METASTABLE_FLUX_TUBE ($\mathfrak{m}$): fuente coloreada con cuerda íntegra.
    - FALSE_FREE_QUARK_ANOMALY ($\bot$): deconfinamiento / ruptura de cuerda.
    """

    TRUE_SINGLET_CONFINED = "TOP_CONFINED"
    METASTABLE_FLUX_TUBE = "METASTABLE_MESON"
    FALSE_FREE_QUARK_ANOMALY = "BOTTOM_DECONFINED"

    @property
    def rank(self) -> int:
        return {
            HeytingTruthValue.FALSE_FREE_QUARK_ANOMALY: 0,
            HeytingTruthValue.METASTABLE_FLUX_TUBE: 1,
            HeytingTruthValue.TRUE_SINGLET_CONFINED: 2,
        }[self]


@dataclass(frozen=True, slots=True)
class BaseMetricCache:
    r"""Geometría Riemanniana de fondo $G_{\mu\nu} \succ 0$ (snapshot afín)."""

    g_base: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    g_inv: NDArray[np.float64]
    christoffel_symbols: NDArray[np.float64]
    ricci_scalar: float
    condition_number: float
    dimension: int
    volume_density: float
    min_eigenvalue: float
    max_eigenvalue: float
    cholesky_residual: float


@dataclass(frozen=True, slots=True)
class QuarkObservationKernel:
    r"""
    EXPEDIENTE TERMINAL DE LA FASE 1  (objeto inicial de la Fase 2).

    Ingestión canónica, C*-invariantes de Banach, inmersión octoniónica
    Günaydin–Gürsey y espectro de Hodge del 1-esqueleto Y-bariónico.
    """

    color_state_triplet: NDArray[np.complex128]
    normalized_state: NDArray[np.complex128]
    color_density_matrix: NDArray[np.complex128]
    banach_cstar_norm: float
    cstar_spectral_radius: float
    banach_l1_norm: float
    banach_l2_norm: float
    cstar_identity_residual: float
    purity: float
    frobenius_distance_to_singlet: float
    hodge_0laplacian_spectrum: NDArray[np.float64]
    hodge_1laplacian_spectrum: NDArray[np.float64]
    hodge_betti: Tuple[int, int]
    hodge_euler_characteristic: int
    hodge_spectral_gap: float
    octonionic_projector_weights: NDArray[np.float64]
    octonionic_associator_norm: float
    octonionic_parseval_residual: float
    metric_cache: BaseMetricCache
    phase1_sha256_digest: str


@dataclass(frozen=True, slots=True)
class QuarkConfinementReport:
    r"""
    EXPEDIENTE TERMINAL DE LA FASE 2  (objeto inicial de la Fase 3).

    Espectro su(3), Casimir operatorial, Cartan $(T_3,T_8)$, Cornell–Lüscher,
    Wilson loop y electrodinámica dual ANO / Meissner.
    """

    gell_mann_coherence_vector: NDArray[np.float64]
    casimir_c2_fundamental: float
    casimir_operator_expectation: float
    adjoint_casimir_magnitude: float
    color_neutrality_deviation: float
    color_charge_squared: float
    cartan_t3: float
    cartan_t8: float
    is_cartan_neutral: bool
    von_neumann_entropy: float
    is_color_singlet: bool
    lie_bracket_residual: float
    killing_form_residual: float
    jacobi_residual: float
    cornell_potential_val: float
    string_tension_energy: float
    luscher_correction_energy: float
    running_alpha_s: float
    is_string_broken: bool
    ano_characteristic_impedance: float
    schwinger_pair_production_rate: float
    london_penetration_depth: float
    coherence_length: float
    dual_meissner_kappa: float
    is_dual_type_II: bool
    wilson_loop_area_law: float
    source_kernel: QuarkObservationKernel
    phase2_sha256_digest: str


@dataclass(frozen=True, slots=True)
class QuarkEngineState:
    r"""
    CERTIFICADO TERMINAL DE GOBERNANZA — FASE 3.

    Clasificación de topos, verificación dagger-compacta en FdHilb,
    latencia FPU y sello BLAKE2b de integridad encadenada.
    """

    kernel: QuarkObservationKernel
    report: QuarkConfinementReport
    topos_truth_value: HeytingTruthValue
    heyting_negation: str
    heyting_implication_to_top: str
    is_gauge_invariant: bool
    is_free_quark_isolated: bool
    dagger_unitarity_residual: float
    choi_min_eigenvalue: float
    snake_identity_residual: float
    fpu_execution_time_us: float
    cryptographic_seal: str


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: OBSERVACIÓN, BANACH C*, GEOMETRÍA RIEMANNIANA, OCTONIONES Y HODGE
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_QuarkColorBundleObserver:
    r"""
    FASE 1 — OBSERVE.

    Ingestión de multipletes de color, saneamiento IEEE-754, auditoría métrica
    SPD, C*-norma, inmersión octoniónica y topología de Hodge del grafo Y.

    El método terminal `observe_and_bundle_color_state` produce el objeto
    inicial de la Fase 2: `QuarkObservationKernel`.
    """

    def __init__(self, tolerance: float = 1.0e-12) -> None:
        self._tol: Final[float] = float(tolerance)

    @staticmethod
    def _sanitize_ieee754_vector(vec: NDArray[np.complex128]) -> NDArray[np.complex128]:
        r"""
        Purga $-0.0$, subnormales bajo el suelo de Wilkinson y rechaza NaN/Inf
        en $\mathbb{C}^n$.
        """
        if not np.all(np.isfinite(vec)):
            raise QuarkEngineError(
                "Vector de entrada contiene componentes no finitas (NaN o Infinito)."
            )
        real_part = np.real(vec).astype(np.float64)
        imag_part = np.imag(vec).astype(np.float64)
        real_clean = np.where(np.abs(real_part) < _WILKINSON_FLOOR, 0.0, real_part)
        imag_clean = np.where(np.abs(imag_part) < _WILKINSON_FLOOR, 0.0, imag_part)
        real_clean = np.where(real_clean == 0.0, 0.0, real_clean)
        imag_clean = np.where(imag_clean == 0.0, 0.0, imag_clean)
        return (real_clean + 1j * imag_clean).astype(np.complex128)

    def _sanitize_density_matrix(
        self,
        rho: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""
        Proyecta $\rho$ al simplejo de estados: Hermiticidad, PSD (clip espectral
        de Higham) y renormalización $\operatorname{Tr}\rho = 1$.
        """
        if rho.ndim != 2 or rho.shape != (3, 3):
            raise QuarkDimensionError(
                f"La matriz de densidad debe ser 3×3. Recibido: {rho.shape}"
            )
        if not np.all(np.isfinite(rho)):
            raise DensityMatrixIntegrityError("ρ contiene componentes no finitas.")
        herm = 0.5 * (rho + np.conj(rho.T))
        eigvals, eigvecs = la.eigh(herm)
        eigvals = np.clip(np.real(eigvals), 0.0, None)
        if float(np.sum(eigvals)) <= _WILKINSON_FLOOR:
            raise DensityMatrixIntegrityError("ρ tiene traza nula tras proyección PSD.")
        eigvals = eigvals / float(np.sum(eigvals))
        rebuilt = (eigvecs * eigvals) @ np.conj(eigvecs.T)
        rebuilt = 0.5 * (rebuilt + np.conj(rebuilt.T))
        return rebuilt.astype(np.complex128)

    def _audit_riemannian_gauge_metric(self, G: NDArray[np.float64]) -> BaseMetricCache:
        r"""
        Audita $G_{\mu\nu}=G_{\nu\mu}\succ 0$. Calcula $G=LL^\top$, $G^{-1}$,
        $\kappa(G)=\lambda_{\max}/\lambda_{\min}$, $\sqrt{\det G}$ y el residuo
        de Cholesky. Para métrica afín constante $\partial G=0\Rightarrow\Gamma=0,\,R=0$.
        """
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise QuarkDimensionError(
                f"El tensor métrico debe ser 2-covariante cuadrado. Dimensiones: {G.shape}"
            )
        d: int = int(G.shape[0])
        g = np.array(G, dtype=np.float64, copy=True)
        sym_diff = float(np.max(np.abs(g - g.T)))
        if sym_diff > self._tol:
            raise MetricIndefinitenessError(
                f"Violación de simetría métrica: ||G - G^T||_max = {sym_diff:.3e}"
            )
        g = 0.5 * (g + g.T)
        try:
            L = la.cholesky(g, lower=True)
        except la.LinAlgError as exc:
            raise MetricIndefinitenessError(
                "El tensor métrico no es definido positivo (Cholesky falló)."
            ) from exc

        eigvals = la.eigvalsh(g)
        min_ev = float(np.min(eigvals))
        max_ev = float(np.max(eigvals))
        if min_ev <= _WILKINSON_FLOOR:
            raise MetricIndefinitenessError(
                f"Autovalor métrico degenerado o negativo detectado: {min_ev:.3e}"
            )
        cond_number = float(max_ev / min_ev)
        if cond_number > _CONDITION_NUMBER_MAX:
            raise MetricIndefinitenessError(
                f"Tensor métrico mal condicionado: kappa = {cond_number:.2e}"
            )

        i_d = np.eye(d, dtype=np.float64)
        l_inv = la.solve_triangular(L, i_d, lower=True)
        g_inv = l_inv.T @ l_inv
        chol_residual = float(np.max(np.abs(L @ L.T - g)))
        log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
        volume_density = float(math.exp(0.5 * log_det))
        gamma_symbols = np.zeros((d, d, d), dtype=np.float64)
        ricci_scalar = 0.0

        return BaseMetricCache(
            g_base=g,
            cholesky_factor=L,
            g_inv=g_inv,
            christoffel_symbols=gamma_symbols,
            ricci_scalar=ricci_scalar,
            condition_number=cond_number,
            dimension=d,
            volume_density=volume_density,
            min_eigenvalue=min_ev,
            max_eigenvalue=max_ev,
            cholesky_residual=chol_residual,
        )

    def _banach_cstar_invariants(
        self,
        state: NDArray[np.complex128],
        rho: NDArray[np.complex128],
    ) -> Tuple[float, float, float, float, float, float, float]:
        r"""
        Invariantes del álgebra $C^*(\mathcal{B}(\mathcal{H}))$:

        - $\|x\|_1$, $\|x\|_2$
        - $\|\rho\| = r(\rho) = \lambda_{\max}(\rho)$ (ρ normal PSD)
        - identidad C*: $\bigl|\|\rho^\dagger\rho\| - \|\rho\|^2\bigr|$
        - pureza $\gamma=\operatorname{Tr}(\rho^2)$
        - distancia de Frobenius a la órbita invariante $I/3$
        """
        acc_l1 = KahanNeumaierAccumulator()
        for z in state.flat:
            acc_l1.add(_cabs(complex(z)))
        norm_l1 = acc_l1.total
        norm_l2 = math.sqrt(max(_kahan_sum_sq_complex(state), 0.0))

        rho_eigvals = np.clip(np.real(la.eigvalsh(rho)), 0.0, None)
        cstar_norm = float(np.max(rho_eigvals))
        spectral_radius = float(np.max(np.abs(rho_eigvals)))
        rho_sq = rho @ rho
        rho_sq_eigs = np.clip(np.real(la.eigvalsh(0.5 * (rho_sq + np.conj(rho_sq.T)))), 0.0, None)
        cstar_of_star = float(np.max(rho_sq_eigs))
        identity_residual = math.fabs(cstar_of_star - cstar_norm * cstar_norm)
        purity = _hermitian_trace_real(rho_sq)
        singlet = np.eye(3, dtype=np.complex128) / 3.0
        dist_singlet = math.sqrt(max(_frobenius_sq(rho - singlet), 0.0))
        return (
            cstar_norm,
            spectral_radius,
            norm_l1,
            norm_l2,
            identity_residual,
            purity,
            dist_singlet,
        )

    @staticmethod
    def _project_gunaydin_gursey_octonions(
        state: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        Descomposición ortogonal de $\mathbb{C}^3$ alineada con Cartan/Günaydin–Gürsey:

            $w_0 = |r+g+b|^2/3$,\; $w_1 = |r-g|^2/2$,\; $w_2 = |r+g-2b|^2/6$.

        Parseval: $w_0+w_1+w_2 = \|\psi\|_2^2$. El asociador $[q,e_7,q]$ mide la
        fuga no asociativa fuera de la subálgebra $\mathbb{H}$ fijada por $e_7$.
        """
        weights = np.zeros(4, dtype=np.float64)
        r, g, b = complex(state[0]), complex(state[1]), complex(state[2])
        weights[0] = _cabs(r + g + b) ** 2 / 3.0
        weights[1] = _cabs(r - g) ** 2 / 2.0
        weights[2] = _cabs(r + g - 2.0 * b) ** 2 / 6.0
        parseval = weights[0] + weights[1] + weights[2]
        nsq = _kahan_sum_sq_complex(state)
        weights[3] = math.fabs(weights[0] - 0.5 * (weights[1] + weights[2]))
        parseval_residual = math.fabs(parseval - nsq)

        q = _color_triplet_to_octonion(state)
        e7 = np.zeros(8, dtype=np.float64)
        e7[7] = 1.0
        assoc = _octonion_associator(q, e7, q)
        assoc_norm = float(math.sqrt(max(float(np.dot(assoc, assoc)), 0.0)))
        return weights, assoc_norm, parseval_residual

    @staticmethod
    def _compute_hodge_simplicial_spectrum() -> Tuple[
        NDArray[np.float64], NDArray[np.float64], Tuple[int, int], int, float
    ]:
        r"""
        Complejo simplicial $K$ del hadrón Y (junction + 3 quarks):

            $V=4$, $E=3$, $F=0$,\; $\chi=V-E+F=1$.

        Operadores de incidencia $B_1:\mathbb{R}^E\to\mathbb{R}^V$,

            $\Delta_0 = B_1 B_1^\top$,\quad $\Delta_1 = B_1^\top B_1$.

        Números de Betti: $\beta_k = \dim\ker\Delta_k$. El Y-grafo es un árbol
        conexo $\Rightarrow \beta_0=1,\;\beta_1=0,\;\chi=\beta_0-\beta_1$.
        """
        b1 = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        delta0 = b1 @ b1.T
        delta1 = b1.T @ b1
        evals0 = np.sort(np.real(la.eigvalsh(delta0)))
        evals1 = np.sort(np.real(la.eigvalsh(delta1)))
        beta0 = int(np.sum(evals0 < 1.0e-10))
        beta1 = int(np.sum(evals1 < 1.0e-10))
        euler = 4 - 3 + 0
        positive = evals1[evals1 > 1.0e-10]
        gap = float(np.min(positive)) if positive.size else 0.0
        return evals0, evals1, (beta0, beta1), euler, gap

    def observe_and_bundle_color_state(
        self,
        color_triplet: NDArray[np.complex128],
        G_metric: NDArray[np.float64],
        density_matrix: Optional[NDArray[np.complex128]] = None,
    ) -> QuarkObservationKernel:
        r"""
        MÉTODO TERMINAL FORMAL DE LA FASE 1.
        ────────────────────────────────────────────────────────────────────────
        Produce `QuarkObservationKernel`, objeto inicial imprescindible de la
        Fase 2 (`Phase2_SU3SpectralConfinementOrient.orient_confinement_spectrum`).

        Cadena anidada:
            observe_and_bundle_color_state  ──Kernel──▶  orient_confinement_spectrum
        """
        if color_triplet.ndim != 1 or color_triplet.shape[0] != 3:
            raise QuarkDimensionError(
                f"El multiplete de color debe ser un 3-vector en C^3. Recibido: {color_triplet.shape}"
            )

        c_clean = self._sanitize_ieee754_vector(np.asarray(color_triplet, dtype=np.complex128))
        metric_cache = self._audit_riemannian_gauge_metric(np.asarray(G_metric, dtype=np.float64))

        norm_l2_sq = _kahan_sum_sq_complex(c_clean)
        norm_l2_tmp = math.sqrt(max(norm_l2_sq, 0.0))
        if norm_l2_tmp > _WILKINSON_FLOOR:
            normalized_c = (c_clean / norm_l2_tmp).astype(np.complex128)
        else:
            normalized_c = np.zeros(3, dtype=np.complex128)

        if density_matrix is None:
            rho = np.outer(normalized_c, np.conj(normalized_c)).astype(np.complex128)
            rho = 0.5 * (rho + np.conj(rho.T))
        else:
            rho = self._sanitize_density_matrix(np.asarray(density_matrix, dtype=np.complex128))

        (
            cstar_norm,
            spectral_radius,
            norm_l1,
            norm_l2,
            cstar_id_res,
            purity,
            dist_singlet,
        ) = self._banach_cstar_invariants(c_clean, rho)

        oct_weights, assoc_norm, parseval_res = self._project_gunaydin_gursey_octonions(c_clean)
        evals0, evals1, betti, euler, gap = self._compute_hodge_simplicial_spectrum()

        hasher = hashlib.sha256()
        hasher.update(c_clean.tobytes())
        hasher.update(rho.tobytes())
        hasher.update(metric_cache.g_base.tobytes())
        hasher.update(evals0.tobytes())
        hasher.update(evals1.tobytes())
        digest = hasher.hexdigest()

        logger.debug("Fase 1 sellada SHA-256=%s…", digest[:16])
        return QuarkObservationKernel(
            color_state_triplet=c_clean,
            normalized_state=normalized_c,
            color_density_matrix=rho,
            banach_cstar_norm=cstar_norm,
            cstar_spectral_radius=spectral_radius,
            banach_l1_norm=norm_l1,
            banach_l2_norm=norm_l2,
            cstar_identity_residual=cstar_id_res,
            purity=purity,
            frobenius_distance_to_singlet=dist_singlet,
            hodge_0laplacian_spectrum=evals0,
            hodge_1laplacian_spectrum=evals1,
            hodge_betti=betti,
            hodge_euler_characteristic=euler,
            hodge_spectral_gap=gap,
            octonionic_projector_weights=oct_weights,
            octonionic_associator_norm=assoc_norm,
            octonionic_parseval_residual=parseval_res,
            metric_cache=metric_cache,
            phase1_sha256_digest=digest,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: CONTINUACIÓN DEL KERNEL — su(3), CASIMIR, CORNELL–LÜSCHER, ANO
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_SU3SpectralConfinementOrient:
    r"""
    FASE 2 — ORIENT.  CONTINUACIÓN FORMAL DE LA FASE 1.

    Objeto inicial: `QuarkObservationKernel` (salida terminal de
    `Phase1_QuarkColorBundleObserver.observe_and_bundle_color_state`).

    Análisis no abeliano su(3), operador de Casimir $C_2=\sum_a T_a^2$,
    potencial Cornell–Lüscher, β-función, Wilson loop y Meissner dual ANO.

    El método terminal `orient_confinement_spectrum` produce el objeto inicial
    de la Fase 3: `QuarkConfinementReport`.
    """

    def __init__(self) -> None:
        self._lambda_matrices: Final[List[NDArray[np.complex128]]] = self._build_gell_mann_basis()
        self._t_generators: Final[List[NDArray[np.complex128]]] = [
            0.5 * lm for lm in self._lambda_matrices
        ]
        self._f_abc, self._d_abc = self._build_structure_constants()
        self._casimir_matrix: Final[NDArray[np.complex128]] = self._build_casimir_operator()
        self._lie_bracket_residual, self._killing_residual, self._jacobi_residual = (
            self._verify_lie_algebra_relations()
        )

    @staticmethod
    def _build_gell_mann_basis() -> List[NDArray[np.complex128]]:
        r"""Base de Gell-Mann $\{\lambda_a\}_{a=1}^{8}$, $\operatorname{Tr}(\lambda_a\lambda_b)=2\delta_{ab}$."""
        l1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)
        l2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128)
        l3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
        l4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex128)
        l5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
        l6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128)
        l7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
        l8 = (1.0 / math.sqrt(3.0)) * np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=np.complex128
        )
        return [l1, l2, l3, l4, l5, l6, l7, l8]

    def _build_structure_constants(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Extrae $f_{abc}$ y $d_{abc}$ desde la base:

            $[\lambda_a,\lambda_b] = 2i f_{abc}\lambda_c$,
            $\{\lambda_a,\lambda_b\} = \tfrac{4}{3}\delta_{ab}I + 2 d_{abc}\lambda_c$.

        Fórmulas de traza: $f_{abc}=\frac{1}{4i}\operatorname{Tr}([\lambda_a,\lambda_b]\lambda_c)$.
        """
        f_abc = np.zeros((8, 8, 8), dtype=np.float64)
        d_abc = np.zeros((8, 8, 8), dtype=np.float64)
        lambdas = self._lambda_matrices
        for a in range(8):
            for b in range(8):
                comm = lambdas[a] @ lambdas[b] - lambdas[b] @ lambdas[a]
                anti = lambdas[a] @ lambdas[b] + lambdas[b] @ lambdas[a]
                for c in range(8):
                    f_abc[a, b, c] = float(np.trace(comm @ lambdas[c]).imag) / 4.0
                    d_abc[a, b, c] = float(np.trace(anti @ lambdas[c]).real) / 4.0
        return f_abc, d_abc

    def _build_casimir_operator(self) -> NDArray[np.complex128]:
        r"""$C_2 = \sum_{a=1}^{8} T_a T_a$. En la fundamental $C_2 = \tfrac{4}{3} I_3$."""
        acc = np.zeros((3, 3), dtype=np.complex128)
        for t_a in self._t_generators:
            acc += t_a @ t_a
        return acc

    def _verify_lie_algebra_relations(self) -> Tuple[float, float, float]:
        r"""
        Verifica:
        1. $\operatorname{Tr} T_a = 0$, $\operatorname{Tr}(T_a T_b)=\tfrac12\delta_{ab}$.
        2. Corchete: $[T_a,T_b] - i f_{abc} T_c = 0$.
        3. Killing: $K_{ab}=\sum_{cd} f_{acd} f_{bcd} = N_c \delta_{ab}$.
        4. Jacobi: $f_{aex}f_{bcx} + \mathrm{ciclicas} = 0$.
        5. $C_2 = \tfrac43 I_3$.
        """
        max_bracket = 0.0
        max_killing = 0.0
        max_jacobi = 0.0
        for a in range(8):
            tr_val = float(np.trace(self._t_generators[a]).real)
            if math.fabs(tr_val) > _WILKINSON_FLOOR:
                raise GaugeAlgebraViolationError(
                    f"Generador su(3) T_{a + 1} con traza no nula: {tr_val:.3e}"
                )
            for b in range(8):
                ortho = float(np.trace(self._t_generators[a] @ self._t_generators[b]).real)
                target = 0.5 if a == b else 0.0
                if math.fabs(ortho - target) > _LIE_ALGEBRA_TOL:
                    raise GaugeAlgebraViolationError(
                        f"Violación de ortonormalidad Killing en T_{a + 1}, T_{b + 1}"
                    )
                comm = (
                    self._t_generators[a] @ self._t_generators[b]
                    - self._t_generators[b] @ self._t_generators[a]
                )
                recon = np.zeros((3, 3), dtype=np.complex128)
                for c in range(8):
                    recon += (1j * self._f_abc[a, b, c]) * self._t_generators[c]
                max_bracket = max(max_bracket, float(la.norm(comm - recon, "fro")))

                killing = float(np.sum(self._f_abc[a, :, :] * self._f_abc[b, :, :]))
                target_k = float(_N_COLOR) if a == b else 0.0
                max_killing = max(max_killing, math.fabs(killing - target_k))

        for a in range(8):
            for b in range(8):
                for c in range(8):
                    acc = 0.0
                    for x in range(8):
                        acc += (
                            self._f_abc[a, x, 0] * 0.0
                            + self._f_abc[a, b, x] * self._f_abc[c, x, 0]
                        )
                    # Jacobi completa en el tensor f:
                    s = 0.0
                    for x in range(8):
                        s += (
                            self._f_abc[a, b, x] * self._f_abc[x, c, 0]
                            + self._f_abc[b, c, x] * self._f_abc[x, a, 0]
                            + self._f_abc[c, a, x] * self._f_abc[x, b, 0]
                        )
                    # Contracción sobre el último índice libre (norma uniforme):
                    jacob = 0.0
                    for e in range(8):
                        acc_e = 0.0
                        for x in range(8):
                            acc_e += (
                                self._f_abc[a, b, x] * self._f_abc[x, c, e]
                                + self._f_abc[b, c, x] * self._f_abc[x, a, e]
                                + self._f_abc[c, a, x] * self._f_abc[x, b, e]
                            )
                        jacob += acc_e * acc_e
                    max_jacobi = max(max_jacobi, math.sqrt(jacob))

        c2_dev = float(la.norm(self._casimir_matrix - _CASIMIR_FUNDAMENTAL_SU3 * np.eye(3), "fro"))
        if c2_dev > 1.0e-10:
            raise GaugeAlgebraViolationError(f"Casimir no proporcional a I: {c2_dev:.3e}")
        if max_bracket > 1.0e-10:
            raise GaugeAlgebraViolationError(f"Corchete de Lie residual: {max_bracket:.3e}")
        if max_killing > 1.0e-8:
            raise GaugeAlgebraViolationError(f"Forma de Killing residual: {max_killing:.3e}")
        if max_jacobi > 1.0e-8:
            raise GaugeAlgebraViolationError(f"Identidad de Jacobi residual: {max_jacobi:.3e}")
        return float(max_bracket), float(max_killing), float(max_jacobi)

    @staticmethod
    def _compute_von_neumann_entropy(density_matrix: NDArray[np.complex128]) -> float:
        r"""$S(\rho)=-\sum_i \lambda_i\ln\lambda_i$ con Kahan–Neumaier y corte Wilkinson."""
        eigvals = np.clip(np.real(la.eigvalsh(density_matrix)), 0.0, None)
        acc = KahanNeumaierAccumulator()
        for p in eigvals:
            pf = float(p)
            if pf > _WILKINSON_FLOOR:
                acc.add(pf * math.log(pf))
        return float(-acc.total)

    @staticmethod
    def _running_alpha_s(r: float, alpha_s_ref: float) -> float:
        r"""
        Acoplamiento 1-loop regularizado (evita el polo de Landau):

            $\alpha_s(r) = \dfrac{4\pi}{\beta_0 \ln\bigl(1 + 1/(r^2\Lambda^2)\bigr)}$

        se mezcla con el valor de referencia del usuario para estabilidad fenoménica.
        """
        r_safe = max(float(r), _WILKINSON_FLOOR)
        log_arg = 1.0 + 1.0 / ((r_safe * _LAMBDA_QCD_GEV) ** 2 + _WILKINSON_FLOOR)
        alpha_run = (4.0 * math.pi) / (_QCD_BETA0 * math.log(max(log_arg, 1.0 + _MACHINE_EPS)))
        # convex combination: no sobreescribe el α_s experimental del llamador
        return float(0.5 * (alpha_s_ref + max(min(alpha_run, 2.0), 0.05)))

    @staticmethod
    def _evaluate_cornell_luescher_potential(
        r: float,
        alpha_s: float,
        string_tension: float,
    ) -> Tuple[float, float, float]:
        r"""
        $V(r) = -\dfrac{4}{3}\dfrac{\alpha_s}{r} + \sigma r - \dfrac{\pi}{12 r}$.
        El término de Coulomb lleva el factor de Casimir $C_F=4/3$.
        """
        r_safe = max(float(r), _WILKINSON_FLOOR)
        coulomb = -_CASIMIR_FUNDAMENTAL_SU3 * (alpha_s / r_safe)
        string_energy = string_tension * r_safe
        luscher_energy = -_LUSCHER_COEFFICIENT / r_safe
        v_total = coulomb + string_energy + luscher_energy
        return float(v_total), float(string_energy), float(luscher_energy)

    @staticmethod
    def _evaluate_ano_dual_circuit(
        string_tension: float,
        string_tension_energy: float,
        radius_r: float,
    ) -> Tuple[float, float, float, float, float, bool]:
        r"""
        Tubo de flujo ANO como superconductor dual de Mandelstam–'t Hooft.

        - $Z_0=\sqrt{\mu/\varepsilon}$
        - $\lambda_L \sim 1/\sqrt{\sigma}$ (penetración de London dual)
        - $\xi \sim 1/(2 m_q)$ (longitud de coherencia)
        - $\kappa=\lambda/\xi$; Tipo II si $\kappa>1/\sqrt{2}$
        - Schwinger: $\Gamma \propto E^2\exp(-\pi m_q^2/E)$, $E\sim\sigma$
        """
        z_0 = math.sqrt(_VACUUM_PERMEABILITY_QCD / _VACUUM_PERMITTIVITY_QCD)
        sigma = max(float(string_tension), _WILKINSON_FLOOR)
        lambda_l = 1.0 / math.sqrt(sigma)
        xi = 1.0 / max(2.0 * _CONSTITUENT_QUARK_MASS_GEV, _WILKINSON_FLOOR)
        kappa = lambda_l / xi
        is_type_ii = bool(kappa > _GINZBURG_LANDAU_TYPE_II)

        e_color = max(string_tension_energy / max(radius_r, _WILKINSON_FLOOR), 0.01)
        mq = _LIGHT_CURRENT_QUARK_MASS_GEV
        exponent = -(math.pi * (mq ** 2)) / e_color
        prefactor = (e_color ** 2) / (4.0 * (math.pi ** 3))
        schwinger_rate = float(prefactor * math.exp(max(exponent, -700.0)))
        return z_0, schwinger_rate, float(lambda_l), float(xi), float(kappa), is_type_ii

    @staticmethod
    def _wilson_loop_area_law(string_tension: float, r: float) -> float:
        r"""Ley de área $W\sim\exp(-\sigma r^2)$ (lazo espacial cuadrado de lado $r$)."""
        r_safe = max(float(r), 0.0)
        exponent = -float(string_tension) * r_safe * r_safe
        return float(math.exp(max(exponent, -700.0)))

    def orient_confinement_spectrum(
        self,
        kernel: QuarkObservationKernel,
        alpha_s: float = 0.3,
        string_tension: float = 1.0,
        geodesic_distance_r: float = 1.0,
    ) -> QuarkConfinementReport:
        r"""
        MÉTODO TERMINAL FORMAL DE LA FASE 2.
        ────────────────────────────────────────────────────────────────────────
        CONTINÚA de `QuarkObservationKernel` (Fase 1) y produce
        `QuarkConfinementReport`, objeto inicial de la Fase 3
        (`Phase3_QuarkGovernanceActuator.execute_quark_confinement_audit`).

        Cadena anidada:
            Kernel  ──orient_confinement_spectrum──▶  Report  ──▶  Fase 3
        """
        rho = kernel.color_density_matrix
        coherence_vector = np.zeros(8, dtype=np.float64)
        acc_norm = KahanNeumaierAccumulator()
        for a in range(8):
            proj_val = float(np.trace(rho @ self._t_generators[a]).real)
            coherence_vector[a] = proj_val
            acc_norm.add(proj_val * proj_val)

        charge_sq = acc_norm.total
        neutrality_deviation = math.sqrt(max(charge_sq, 0.0))
        cartan_t3 = float(coherence_vector[2])
        cartan_t8 = float(coherence_vector[7])
        is_cartan_neutral = bool(
            math.hypot(cartan_t3, cartan_t8) <= _CASIMIR_SINGLET_TOLERANCE
        )
        is_singlet = bool(neutrality_deviation <= _CASIMIR_SINGLET_TOLERANCE)

        # C_2 operatorial: Tr(ρ C_2) = 4/3 para todo ρ en la fundamental.
        c2_expectation = _hermitian_trace_real(rho @ self._casimir_matrix)
        # Casimir efectivo de carga: se anula en la órbita invariante ρ = I/3.
        c2_effective = 0.0 if is_singlet else _CASIMIR_FUNDAMENTAL_SU3
        adj_magnitude = float(neutrality_deviation * math.sqrt(_CASIMIR_ADJOINT_SU3))
        s_vn = self._compute_von_neumann_entropy(rho)

        alpha_run = self._running_alpha_s(geodesic_distance_r, alpha_s)
        v_cornell, e_string, e_luscher = self._evaluate_cornell_luescher_potential(
            r=geodesic_distance_r,
            alpha_s=alpha_s,
            string_tension=string_tension,
        )
        is_broken = bool(e_string >= _STRING_BREAKAGE_THRESHOLD)
        z_ano, gamma_schwinger, lambda_l, xi, kappa, is_type_ii = self._evaluate_ano_dual_circuit(
            string_tension=string_tension,
            string_tension_energy=e_string,
            radius_r=geodesic_distance_r,
        )
        wilson = self._wilson_loop_area_law(string_tension, geodesic_distance_r)

        hasher = hashlib.sha256()
        hasher.update(kernel.phase1_sha256_digest.encode("utf-8"))
        hasher.update(coherence_vector.tobytes())
        hasher.update(
            np.array(
                [c2_expectation, v_cornell, e_string, gamma_schwinger, wilson],
                dtype=np.float64,
            ).tobytes()
        )
        digest = hasher.hexdigest()

        logger.debug("Fase 2 sellada SHA-256=%s…", digest[:16])
        return QuarkConfinementReport(
            gell_mann_coherence_vector=coherence_vector,
            casimir_c2_fundamental=c2_effective,
            casimir_operator_expectation=float(c2_expectation),
            adjoint_casimir_magnitude=adj_magnitude,
            color_neutrality_deviation=neutrality_deviation,
            color_charge_squared=float(charge_sq),
            cartan_t3=cartan_t3,
            cartan_t8=cartan_t8,
            is_cartan_neutral=is_cartan_neutral,
            von_neumann_entropy=s_vn,
            is_color_singlet=is_singlet,
            lie_bracket_residual=self._lie_bracket_residual,
            killing_form_residual=self._killing_residual,
            jacobi_residual=self._jacobi_residual,
            cornell_potential_val=v_cornell,
            string_tension_energy=e_string,
            luscher_correction_energy=e_luscher,
            running_alpha_s=alpha_run,
            is_string_broken=is_broken,
            ano_characteristic_impedance=z_ano,
            schwinger_pair_production_rate=gamma_schwinger,
            london_penetration_depth=lambda_l,
            coherence_length=xi,
            dual_meissner_kappa=kappa,
            is_dual_type_II=is_type_ii,
            wilson_loop_area_law=wilson,
            source_kernel=kernel,
            phase2_sha256_digest=digest,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: CONTINUACIÓN DEL REPORT — TOPOS, FdHilb DAGGER-COMPACTO, SELLO
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_QuarkGovernanceActuator:
    r"""
    FASE 3 — ACT.  CONTINUACIÓN FORMAL DE LA FASE 2.

    Objetos iniciales: `QuarkObservationKernel` y `QuarkConfinementReport`.
    Clasificador de subobjetos $\Omega$ (Heyting), unitariedad dagger-compacta
    en FdHilb, Choi–Jamiołkowski, identidad de la serpiente, sello BLAKE2b.
    """

    def __init__(self, tolerance: float = 1.0e-12) -> None:
        self._phase1_observer: Final[Phase1_QuarkColorBundleObserver] = (
            Phase1_QuarkColorBundleObserver(tolerance=tolerance)
        )
        self._phase2_orienter: Final[Phase2_SU3SpectralConfinementOrient] = (
            Phase2_SU3SpectralConfinementOrient()
        )

    @staticmethod
    def _heyting_meet(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        return a if a.rank <= b.rank else b

    @staticmethod
    def _heyting_join(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        return a if a.rank >= b.rank else b

    @staticmethod
    def _heyting_implies(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        r"""$a \to b = \top$ si $a\le b$, si no $b$ (Heyting de cadena / Gödel)."""
        if a.rank <= b.rank:
            return HeytingTruthValue.TRUE_SINGLET_CONFINED
        return b

    @classmethod
    def _heyting_negation(cls, a: HeytingTruthValue) -> HeytingTruthValue:
        r"""$\neg a = a \to \bot$."""
        return cls._heyting_implies(a, HeytingTruthValue.FALSE_FREE_QUARK_ANOMALY)

    @staticmethod
    def _evaluate_topos_subobject_classifier(
        is_singlet: bool,
        is_broken: bool,
        neutrality_dev: float,
    ) -> HeytingTruthValue:
        r"""
        Morfismo característico $\chi:\mathrm{State}\to\Omega$.

        - $\top$: $\rho$ $SU(3)$-invariante (singlete de estado).
        - $\mathfrak{m}$: fuente coloreada con cuerda no rota.
        - $\bot$: deconfinamiento / hadronización / anomalía de calibre.
        """
        if is_singlet:
            return HeytingTruthValue.TRUE_SINGLET_CONFINED
        if (not is_broken) and neutrality_dev < 1.0:
            return HeytingTruthValue.METASTABLE_FLUX_TUBE
        return HeytingTruthValue.FALSE_FREE_QUARK_ANOMALY

    @staticmethod
    def _verify_dagger_compact_unitarity(density_matrix: NDArray[np.complex128]) -> float:
        r"""
        Endomorfismo en FdHilb: $\rho^\dagger=\rho$, $\operatorname{Tr}\rho=1$.
        Residuo $\|\rho-\rho^\dagger\|_F + |1-\operatorname{Tr}\rho|$.
        """
        dagger_diff = density_matrix - np.conj(density_matrix.T)
        frob_residual = float(la.norm(dagger_diff, "fro"))
        trace_residual = math.fabs(1.0 - _hermitian_trace_real(density_matrix))
        return float(frob_residual + trace_residual)

    @staticmethod
    def _choi_min_eigenvalue(density_matrix: NDArray[np.complex128]) -> float:
        r"""
        Isomorfismo de Choi–Jamiołkowski: un estado es CP ssi $\rho\succeq 0$.
        Devuelve $\lambda_{\min}(\rho)$.
        """
        eigs = np.real(la.eigvalsh(0.5 * (density_matrix + np.conj(density_matrix.T))))
        return float(np.min(eigs))

    @staticmethod
    def _snake_identity_residual(psi: NDArray[np.complex128]) -> float:
        r"""
        Identidad de la serpiente en FdHilb ($\dim=3$):

            $(\mathrm{id}\otimes\varepsilon)\circ(\eta\otimes\mathrm{id}) = \mathrm{id}$,

        con $\eta(1)=\sum_i |i\rangle\otimes|i\rangle$, $\varepsilon=\eta^\dagger$.
        Sobre $|\psi\rangle$ el residuo debe anularse a precisión FPU.
        """
        n = 3
        eta = np.eye(n, dtype=np.complex128).reshape(n * n)  # Σ |ii⟩
        # (η ⊗ id)|ψ⟩ ∈ C^{n³} ; (id ⊗ ε) contrae los dos últimos factores.
        # (id ⊗ ε)(η ⊗ id)|ψ⟩_k = Σ_i η_{i i} ψ_k = ψ_k.
        reconstructed = np.zeros(n, dtype=np.complex128)
        for k in range(n):
            acc = 0.0 + 0.0j
            for i in range(n):
                acc += eta[i * n + i] * psi[k]
            reconstructed[k] = acc
        return float(np.linalg.norm(reconstructed - psi))

    def execute_quark_confinement_audit(
        self,
        color_triplet: NDArray[np.complex128],
        G_metric: NDArray[np.float64],
        alpha_s: float = 0.3,
        string_tension: float = 1.0,
        geodesic_distance_r: float = 1.0,
        density_matrix: Optional[NDArray[np.complex128]] = None,
    ) -> QuarkEngineState:
        r"""
        MÉTODO TERMINAL FINAL DE LA FASE 3 (PIPELINE ANIDADO GLOBAL).

        1. Fase 1  → `QuarkObservationKernel`
        2. Fase 2  → `QuarkConfinementReport`   [continúa del Kernel]
        3. Fase 3  → `QuarkEngineState`         [continúa del Report]
        """
        t_start_ns: int = time.perf_counter_ns()

        kernel: QuarkObservationKernel = self._phase1_observer.observe_and_bundle_color_state(
            color_triplet=color_triplet,
            G_metric=G_metric,
            density_matrix=density_matrix,
        )
        report: QuarkConfinementReport = self._phase2_orienter.orient_confinement_spectrum(
            kernel=kernel,
            alpha_s=alpha_s,
            string_tension=string_tension,
            geodesic_distance_r=geodesic_distance_r,
        )

        truth_value = self._evaluate_topos_subobject_classifier(
            is_singlet=report.is_color_singlet,
            is_broken=report.is_string_broken,
            neutrality_dev=report.color_neutrality_deviation,
        )
        neg = self._heyting_negation(truth_value)
        impl_top = self._heyting_implies(truth_value, HeytingTruthValue.TRUE_SINGLET_CONFINED)

        dagger_residual = self._verify_dagger_compact_unitarity(kernel.color_density_matrix)
        choi_min = self._choi_min_eigenvalue(kernel.color_density_matrix)
        snake_res = self._snake_identity_residual(kernel.normalized_state)
        is_gauge_clean = bool(dagger_residual <= 1.0e-11 and choi_min >= -1.0e-12)
        is_free_quark = bool(truth_value is HeytingTruthValue.FALSE_FREE_QUARK_ANOMALY)

        t_elapsed_us = float((time.perf_counter_ns() - t_start_ns) / 1000.0)

        blake = hashlib.blake2b(digest_size=32)
        blake.update(kernel.phase1_sha256_digest.encode("utf-8"))
        blake.update(report.phase2_sha256_digest.encode("utf-8"))
        blake.update(truth_value.value.encode("utf-8"))
        blake.update(np.array([dagger_residual, choi_min, snake_res, t_elapsed_us]).tobytes())
        master_seal = blake.hexdigest()

        if is_free_quark:
            logger.warning(
                "Anomalía de deconfinamiento: ||n||=%.3e, cuerda_rota=%s, sello=%s",
                report.color_neutrality_deviation,
                report.is_string_broken,
                master_seal[:16],
            )

        return QuarkEngineState(
            kernel=kernel,
            report=report,
            topos_truth_value=truth_value,
            heyting_negation=neg.value,
            heyting_implication_to_top=impl_top.value,
            is_gauge_invariant=is_gauge_clean,
            is_free_quark_isolated=is_free_quark,
            dagger_unitarity_residual=dagger_residual,
            choi_min_eigenvalue=choi_min,
            snake_identity_residual=snake_res,
            fpu_execution_time_us=t_elapsed_us,
            cryptographic_seal=master_seal,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENLACE DE ALIAS DEL MOTOR MAESTRO
# ══════════════════════════════════════════════════════════════════════════════
class QuarkColorConfinementSatelliteEngine(Phase3_QuarkGovernanceActuator):
    r"""
    Alias principal y API unificada. Hereda la composición lineal estricta
    de las tres fases anidadas (Observe → Orient → Act).
    """


# ══════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN DE VERIFICACIÓN DE FPU Y MODELADO CUÁNTICO
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 80)
    print(" INICIALIZANDO AUDITORÍA ESPECTRAL: QUARK COLOR CONFINEMENT ENGINE v2.0.0")
    print("=" * 80 + "\n")

    engine = QuarkColorConfinementSatelliteEngine()
    g_minkowski = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float64)

    # --------------------------------------------------------------------------
    # CASO A: Estado SU(3)-invariante ρ = I/3  (único singlete de estado en C³)
    # Un 3-vector puro NUNCA es singlete: C_2(F)=4/3 y ||n||²=1/3. El confined
    # gauge-invariante es la mezcla máxima, punto fijo de Ad_{SU(3)}.
    # --------------------------------------------------------------------------
    print(">>> EXPERIMENTO A: Estado invariante de gauge ρ = I/3 (singlete de estado)...")
    c_neutral = (1.0 / math.sqrt(3.0)) * np.array(
        [1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=np.complex128
    )
    rho_singlet = np.eye(3, dtype=np.complex128) / 3.0

    state_a = engine.execute_quark_confinement_audit(
        color_triplet=c_neutral,
        G_metric=g_minkowski,
        alpha_s=0.3,
        string_tension=1.0,
        geodesic_distance_r=1.5,
        density_matrix=rho_singlet,
    )

    print(f"  [+] Fase 1 - Norma C* de Banach           : {state_a.kernel.banach_cstar_norm:.6f}")
    print(f"  [+] Fase 1 - Identidad C* residual        : {state_a.kernel.cstar_identity_residual:.3e}")
    print(f"  [+] Fase 1 - Pureza Tr(ρ²)                : {state_a.kernel.purity:.6f}")
    print(f"  [+] Fase 1 - ||ρ - I/3||_F                : {state_a.kernel.frobenius_distance_to_singlet:.3e}")
    print(f"  [+] Fase 1 - Betti (β0, β1) / χ           : {state_a.kernel.hodge_betti} / {state_a.kernel.hodge_euler_characteristic}")
    print(f"  [+] Fase 1 - Espectro Hodge Δ1            : {np.round(state_a.kernel.hodge_1laplacian_spectrum, 6)}")
    print(f"  [+] Fase 1 - Asociador octoniónico        : {state_a.kernel.octonionic_associator_norm:.3e}")
    print(f"  [+] Fase 1 - √det G (volumen)             : {state_a.kernel.metric_cache.volume_density:.6f}")
    print(f"  [+] Fase 1 - Digestión SHA-256            : {state_a.kernel.phase1_sha256_digest[:20]}...")
    print(f"  [*] Fase 2 - Desviación Neutralidad ||n|| : {state_a.report.color_neutrality_deviation:.6e}")
    print(f"  [*] Fase 2 - ⟨C_2⟩ operatorial            : {state_a.report.casimir_operator_expectation:.6f}")
    print(f"  [*] Fase 2 - C_2 efectivo (carga)         : {state_a.report.casimir_c2_fundamental:.6f}")
    print(f"  [*] Fase 2 - Cartan (T3, T8)              : ({state_a.report.cartan_t3:.3e}, {state_a.report.cartan_t8:.3e})")
    print(f"  [*] Fase 2 - S(ρ) von Neumann             : {state_a.report.von_neumann_entropy:.6f}")
    print(f"  [*] Fase 2 - Jacobi / Killing residuales  : {state_a.report.jacobi_residual:.3e} / {state_a.report.killing_form_residual:.3e}")
    print(f"  [*] Fase 2 - Potencial Cornell-Lüscher V  : {state_a.report.cornell_potential_val:.6f} GeV")
    print(f"  [*] Fase 2 - α_s running (1-loop mix)     : {state_a.report.running_alpha_s:.6f}")
    print(f"  [*] Fase 2 - Impedancia Tubo ANO Z_0      : {state_a.report.ano_characteristic_impedance:.2f} Ohm_QCD")
    print(f"  [*] Fase 2 - κ Meissner dual (Tipo II)    : {state_a.report.dual_meissner_kappa:.4f} ({state_a.report.is_dual_type_II})")
    print(f"  [#] Fase 3 - Valuación Topos (Heyting)    : {state_a.topos_truth_value.value}")
    print(f"  [#] Fase 3 - ¬χ  /  χ → ⊤                 : {state_a.heyting_negation} / {state_a.heyting_implication_to_top}")
    print(f"  [#] Fase 3 - Residuo Dagger FdHilb        : {state_a.dagger_unitarity_residual:.3e}")
    print(f"  [#] Fase 3 - λ_min Choi / serpiente       : {state_a.choi_min_eigenvalue:.3e} / {state_a.snake_identity_residual:.3e}")
    print(f"  [#] Fase 3 - Tiempo de Ejecución FPU      : {state_a.fpu_execution_time_us:.2f} us")
    print(f"  [#] Fase 3 - Sello Maestro BLAKE2b        : {state_a.cryptographic_seal}")
    print("-" * 80)

    # --------------------------------------------------------------------------
    # CASO B: Quark rojo puro (peso de Cartan (1/2, 1/(2√3))) a distancia crítica
    # --------------------------------------------------------------------------
    print(">>> EXPERIMENTO B: Quark asimétrico no confinado (rojo puro, r crítico)...")
    c_red = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=np.complex128)

    state_b = engine.execute_quark_confinement_audit(
        color_triplet=c_red,
        G_metric=g_minkowski,
        alpha_s=0.3,
        string_tension=1.0,
        geodesic_distance_r=120.0,
    )

    print(f"  [+] Fase 1 - Proyectores Günaydin-Gürsey  : {np.round(state_b.kernel.octonionic_projector_weights, 6)}")
    print(f"  [+] Fase 1 - Pureza / ||ρ-I/3||_F         : {state_b.kernel.purity:.6f} / {state_b.kernel.frobenius_distance_to_singlet:.6f}")
    print(f"  [*] Fase 2 - Vector Coherencia Gell-Mann  : {np.round(state_b.report.gell_mann_coherence_vector, 4)}")
    print(f"  [*] Fase 2 - Cartan (T3, T8)              : ({state_b.report.cartan_t3:.6f}, {state_b.report.cartan_t8:.6f})")
    print(f"  [*] Fase 2 - ||n||² (teórico 1/3 puro)    : {state_b.report.color_charge_squared:.6f}")
    print(f"  [*] Fase 2 - ⟨C_2⟩ operatorial            : {state_b.report.casimir_operator_expectation:.6f}")
    print(f"  [*] Fase 2 - ¿Cuerda Rota / Hadronizada?  : {state_b.report.is_string_broken}")
    print(f"  [*] Fase 2 - Tasa Schwinger de Ruptura    : {state_b.report.schwinger_pair_production_rate:.4e}")
    print(f"  [*] Fase 2 - Wilson loop area-law         : {state_b.report.wilson_loop_area_law:.4e}")
    print(f"  [*] Fase 2 - λ_L / ξ (Meissner dual)      : {state_b.report.london_penetration_depth:.4f} / {state_b.report.coherence_length:.4f}")
    print(f"  [#] Fase 3 - Valuación Topos (Heyting)    : {state_b.topos_truth_value.value}")
    print(f"  [#] Fase 3 - Alerta Quark Libre Aislado   : {state_b.is_free_quark_isolated}")
    print(f"  [#] Fase 3 - Sello Maestro BLAKE2b        : {state_b.cryptographic_seal}")
    print("=" * 80)
    print(" AUDITORÍA MATEMÁTICA Y METROLÓGICA FPU COMPLETADA CON ÉXITO")
    print("=" * 80 + "\n")
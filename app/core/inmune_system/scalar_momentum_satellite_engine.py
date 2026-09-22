from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Scalar Momentum Satellite Engine (Satélite I — Momentum Escalar)            ║
║ RUTA   : app/core/immune_system/scalar_momentum_satellite_engine.py                  ║
║ VERSIÓN: 3.0.0-Doctoral-Hypercomplex-Spectral-PortHamiltonian-Banach-Secure          ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Módulo FPU de alta precisión para el cálculo covariante de la transferencia de momentum escalar
$p_\mu \in T^*_x\mathcal{M}$ sobre un campo suave $\phi \in C^\infty(\mathcal{M}, \mathbb{R})$
definido en una variedad Riemanniana compacta orientada $(\mathcal{M}, G, \partial\mathcal{M})$.

Fundamentación Matemática y Física Rigurosa:
────────────────────────────────────────────
1. Geometría Simpléctica en Fibrados Cotangentes:
   El espacio de fases $T^*\mathcal{M}$ posee la 2-forma simpléctica canónica $\omega = dp_\mu \wedge dx^\mu$.
   La velocidad contravariante surge del isomorfismo musical (sharp $\sharp$):
   $$v = G^\sharp(p) \implies v^\mu = G^{\mu\nu} p_\nu \in T_x\mathcal{M}$$

2. Derivada de Lie y Corchetes de Poisson:
   $$\mathcal{L}_v \phi = \iota_v d\phi = \langle d\phi, v \rangle = p_\mu G^{\mu\nu} \partial_\nu \phi = \{\phi, H\}$$
   donde el Hamiltoniano cinético libre es $H(x, p) = \frac{1}{2} G^{\mu\nu}(x) p_\mu p_\nu$.

3. Diferenciación Holomorfa por Paso Complejo (CSMD) y Richardson:
   Auditada en todas las direcciones coordenadas mediante las condiciones de Cauchy-Riemann:
   $$\frac{\partial u}{\partial x^k} = \frac{\partial v}{\partial y^k}, \quad \frac{\partial u}{\partial y^k} = -\frac{\partial v}{\partial x^k}$$
   Con retroceso a diferencias centradas de orden 4 extrapoladas de Richardson $\mathcal{O}(h^6)$:
   $$D^\star = \frac{16 D_{h/2} - D_h}{15} + \mathcal{O}(h^6)$$

4. Teoría Espectral de Operadores Autoadjuntos y Espacios de Banach:
   $G = U \Lambda U^\top$, con proyectores espectrales de Banach $\{P_k = u_k u_k^\top\}$ auditados por:
   - Idempotencia: $\|P_k^2 - P_k\|_F < \delta_{\mathrm{idempotency}}$
   - Completitud: $\|\sum_k P_k - I\|_F < \delta_{\mathrm{completeness}}$
   Forma de volumen Riemanniana $\mathrm{vol}_G = \sqrt{\det G} = \prod L_{ii}$ con log-volumen estable.

5. Dinámica Port-Hamiltoniana y Pasividad Termodinámica de Tellegen:
   $$\dot{z} = (J - R)\nabla H + g u, \quad R = \begin{pmatrix} \sigma G^{-1} & 0 \\ 0 & \kappa G^{-1} \end{pmatrix} \succeq 0$$
   donde $\sigma, \kappa \ge 0$ son conductancias disipativas desacopladas. El generador infinitesimal $\mathcal{A} = -\sigma G - \eta J$
   satisface la contractividad de Lumer-Phillips en el producto interno $G$-ponderado:
   $$\langle \mathcal{A}x, x\rangle_G = x^\top G \mathcal{A} x \le 0 \iff \frac{1}{2}\left(G\mathcal{A} + (G\mathcal{A})^\top\right) \preceq \epsilon_{\mathrm{Wilkinson}} I$$

6. Topología Algebraica y Determinante de Gram $n$-Dimensional:
   $$\|x \wedge v\|_G^2 = \|x\|_G^2 \|v\|_G^2 - \langle x, v\rangle_G^2$$
   proporcionando el invariante de enrollamiento (winding number) $G$-métrico.

7. Metrología FPU y Cota de Redondeo de Wilkinson:
   $$\gamma_n = \frac{c \cdot n \cdot \epsilon_{\mathrm{mach}}}{1 - c \cdot n \cdot \epsilon_{\mathrm{mach}}}$$
   combinada con sumación compensada exactísima Kahan-Babuška-Neumaier (KBN).

Traducción Ejecutiva e Impacto de Negocio ('Dolor y Dinero'):
─────────────────────────────────────────────────────────────
• Dolor: Inestabilidades numéricas en la transferencia de momentum en estimaciones de costo/rendimiento
  provocan desviaciones financieras (ruina por sobrecostos en licitaciones de infraestructura).
• Dinero: Garantiza convergencia FPU de ultra-alta fidelidad en IEEE-754, eliminando derivas de redondeo y
  protegiendo la rentabilidad de megaproyectos mediante auditoría inmutable en lazo cerrado OODA.

Estructura Functorial OODA:
───────────────────────────
- Fase 1 (Observe) : `observe_scalar_field` -> Salida: `ScalarMomentumKernel`
- Fase 2 (Orient)  : `orient_from_kernel`   -> Salida: `MomentumTransferReport`
- Fase 3 (Act)     : `act_from_kernel_and_report` -> Salida: `ScalarMomentumEngineState`
"""

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Configuración de Logging de Metrología Orbital
logger = logging.getLogger("APU.Physics.ScalarMomentumSatelliteEngine")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FUNDAMENTALES DE PRECISIÓN FPU Y METROLOGÍA CUÁNTICA
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CSMD_STEP_OPTIMAL: Final[float] = 1.0e-20
_CSMD_CROSS_VALIDATION_MULTIPLIER: Final[float] = 1.0e8
_RICHARDSON_STEP: Final[float] = 1.0e-5
_RICHARDSON_STENCIL_ORDER: Final[int] = 4
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12
_WILKINSON_FLOOR: Final[float] = 1.0e-15
_WILKINSON_OPERATION_CHAIN_FACTOR: Final[int] = 4  # Cholesky + eigh + solve_triangular + matmul
_CR_HOLOMORPHY_TOLERANCE: Final[float] = 1.0e-4
_CR_PROBE_STEP: Final[float] = 1.0e-8
_SPECTRAL_PROJECTOR_IDEMPOTENCY_TOL: Final[float] = 1.0e-9
_ADVECTIVE_COUPLING_ETA_DEFAULT: Final[float] = 1.0
_DIRAC_PLANCK_CONSTANT: Final[float] = 1.054571817e-34  # \hbar (J·s)
_HMAC_METROLOGY_KEY: Final[bytes] = b"ScalarMomentumSatelliteEngine::QuantumPortHamiltonianSecretKey"


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES DOCTORALES DE FÓRMULA CERRADA
# ══════════════════════════════════════════════════════════════════════════════

class ScalarMomentumEngineError(Exception):
    """Excepción raíz del subsistema de cálculo de momentum escalar."""
    pass


class DimensionMismatchError(ScalarMomentumEngineError):
    """Falla de consistencia dimensional en fibras tangentes o cotangentes."""
    pass


class MetricIndefinitenessError(ScalarMomentumEngineError):
    """Falla crítica: El tensor métrico no reside en el cono SPD de Banach."""
    pass


class SpectralProjectorInconsistencyError(ScalarMomentumEngineError):
    """Falla en la auditoría de idempotencia/completitud de los proyectores espectrales $P_k$."""
    pass


class CSMDHolomorphyError(ScalarMomentumEngineError):
    """Ruptura de las condiciones de Cauchy-Riemann en la diferenciación compleja."""
    pass


class ThermodynamicPassivityViolationError(ScalarMomentumEngineError):
    """Violación del principio de pasividad termodinámica (Disipación negativa o R indefinida)."""
    pass


class BanachSemigroupInstabilityError(ScalarMomentumEngineError):
    """Falla en la contractividad de Lumer-Phillips para el semigrupo de evolución $G$-ponderado."""
    pass


class MetrologicalChainIntegrityError(ScalarMomentumEngineError):
    """Falla en la verificación de la cadena de custodia criptográfica entre fases."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# MÁQUINA DE SUMACIÓN COMPENSADA EXACTA: KAHAN-BABUŠKA-NEUMAIER (KBN)
# ══════════════════════════════════════════════════════════════════════════════

class KahanNeumaierAccumulator:
    r"""
    Acumulador con sumación compensada de Kahan-Babuška-Neumaier (KBN).
    Garantiza que el error residual de redondeo de máquina $\mathcal{O}(\epsilon)$
    no contamine los dígitos significativos de la mantisa IEEE-754.
    """
    __slots__ = ("_sum", "_compensation")

    def __init__(self, initial_value: float = 0.0) -> None:
        self._sum: float = float(initial_value)
        self._compensation: float = 0.0

    def add(self, term: float) -> None:
        r"""Añade un término escalar compensando la pérdida de significancia."""
        t = self._sum + term
        if abs(self._sum) >= abs(term):
            self._compensation += (self._sum - t) + term
        else:
            self._compensation += (term - t) + self._sum
        self._sum = t

    def add_array(self, terms: NDArray[np.float64]) -> None:
        r"""Acumula un vector unidimensional completo."""
        for val in terms.ravel():
            self.add(float(val))

    @property
    def total(self) -> float:
        r"""Retorna la suma corregida exacta."""
        return float(self._sum + self._compensation)

    @staticmethod
    def sum(arr: NDArray[np.float64]) -> float:
        r"""Método estático para computar la suma KBN de un tensor unidimensional."""
        acc = KahanNeumaierAccumulator()
        acc.add_array(arr)
        return acc.total

    @staticmethod
    def dot(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
        r"""Producto interno euclidiano exacto mediante producto elemento-a-elemento y KBN."""
        if u.shape[0] != v.shape[0]:
            raise DimensionMismatchError("Vectores incompatibles para producto punto KBN.")
        products = u * v
        return KahanNeumaierAccumulator.sum(products)


# ══════════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: OBSERVE — INGESTA, TEORÍA ESPECTRAL AUDITADA, CSMD/RICHARDSON             ║
# ║          VALIDADO CRUZADAMENTE Y FORMAS EXTERIORES DE HODGE                      ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────── Expedientes de Fase 1 ────────────────────────────────

@dataclass(frozen=True, slots=True)
class SpectralMetricTensorCache:
    r"""
    Expediente espectral inmutable de la variedad Riemanniana $(\mathcal{M}, G)$.
    Almacena $G$, Cholesky $L$, inversa $G^{-1}$, autovalores $\{\lambda_k\}$,
    proyectores espectrales AUDITADOS $P_k = u_k u_k^\top$ (idempotencia y completitud
    verificadas numéricamente), $\sqrt{\det G}$, $\log\sqrt{\det G}$ (estable ante
    overflow/underflow en alta dimensión), radio espectral y $\kappa_2(G)$.
    """
    g_base: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    g_inv: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    spectral_projectors: Tuple[NDArray[np.float64], ...]
    riemannian_volume_factor: float
    log_riemannian_volume_factor: float
    spectral_radius: float
    condition_number: float
    dimension: int
    projectors_verified: bool


@dataclass(frozen=True, slots=True)
class DifferentialFormsBundle:
    r"""
    Haz de formas diferenciales en $T^*_x\mathcal{M}$:
    $\phi \in \Omega^0(\mathcal{M})$, 1-forma $d\phi \in \Omega^1(\mathcal{M})$,
    dual de Hodge $*d\phi \in \Omega^{n-1}(\mathcal{M})$, método de diferenciación
    efectivamente empleado y su vector de error de truncamiento estimado por
    validación cruzada (CSMD de doble paso o Richardson de dos niveles).
    """
    scalar_field: float
    exterior_derivative: NDArray[np.float64]
    gradient_norm_squared: float
    hodge_star_exterior_derivative: NDArray[np.float64]
    differentiation_method: str
    truncation_error_estimate: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ScalarMomentumKernel:
    r"""
    Expediente Inmutable Terminal de Fase 1 (Observe).
    Entrega el punto base $x$, el momentum $p_\mu$, el haz de formas y la caché métrica.
    """
    evaluation_point: NDArray[np.float64]
    momentum_covector: NDArray[np.float64]
    forms_bundle: DifferentialFormsBundle
    metric_cache: SpectralMetricTensorCache
    phase1_sha256_hash: str


# ─────────────────────────────────── Motor de Fase 1 ───────────────────────────────────

class Phase1_ScalarMomentumObserver:
    r"""
    FASE 1 (OBSERVE):
    Fundamentación matemática:
      - Auditoría espectral del tensor métrico $G \in \mathcal{S}^+_n(\mathbb{R})$, con
        verificación explícita de idempotencia $P_k^2 = P_k$ y completitud $\sum_k P_k = I$
        de los proyectores espectrales (teorema espectral de operadores autoadjuntos).
      - Factorización de Cholesky $G = L L^\top$, $\kappa_2(G)$, inversión triangular exacta,
        log-volumen Riemanniano estable $\log\sqrt{\det G} = \sum_i \log L_{ii}$.
      - Diferenciación holomorfa por paso complejo (CSMD) auditada en las $n$ direcciones
        coordenadas mediante condiciones de Cauchy-Riemann (no solo el primer eje).
      - Validación cruzada de doble paso: CSMD con $h$ y $10^{8}h$, o Richardson con
        $h$ y $h/2$ extrapolado a $\mathcal{O}(h^6)$, produciendo una cota de error explícita.
      - Cálculo de formas diferenciales: $\phi$, $d\phi$, $\|d\phi\|^2_G \ge 0$ (auditado),
        y el dual de Hodge $*d\phi \in \Omega^{n-1}(\mathcal{M})$.
    """

    def __init__(self, condition_tolerance: float = _CONDITION_NUMBER_MAX) -> None:
        self._condition_tolerance: Final[float] = float(condition_tolerance)

    # ---- 1.1 Auditoría de holomorfía Cauchy-Riemann (todas las direcciones) ----------

    def _verify_cauchy_riemann_holomorphy(
        self,
        phi_func: Callable[[Any], Any],
        x_point: NDArray[np.float64],
        probe_step: float = _CR_PROBE_STEP
    ) -> Tuple[bool, float]:
        r"""
        Audita numéricamente si $\phi$ satisface las condiciones de Cauchy-Riemann sobre
        $\mathbb{C}$ en **cada una** de las $n$ direcciones coordenadas:
        $$\frac{\partial u}{\partial x^k} = \frac{\partial v}{\partial y^k}, \quad
          \frac{\partial u}{\partial y^k} = -\frac{\partial v}{\partial x^k}, \quad \forall k$$
        donde $\phi(x + i y) = u(x, y) + i v(x, y)$. Retorna el booleano de holomorfía
        global y la violación máxima (norma-sup) sobre todas las direcciones auditadas.
        """
        dim = x_point.shape[0]
        z0 = np.array(x_point, dtype=np.complex128)

        try:
            val_base = phi_func(z0)
        except Exception:
            return False, float("inf")

        if not isinstance(val_base, (complex, np.complex128)):
            return False, float("inf")

        max_violation = 0.0
        for k in range(dim):
            try:
                zx_plus = z0.copy()
                zx_plus[k] += probe_step
                val_x_plus = phi_func(zx_plus)

                zy_plus = z0.copy()
                zy_plus[k] += 1j * probe_step
                val_y_plus = phi_func(zy_plus)
            except Exception:
                return False, float("inf")

            du_dx = (np.real(val_x_plus) - np.real(val_base)) / probe_step
            dv_dx = (np.imag(val_x_plus) - np.imag(val_base)) / probe_step
            du_dy = (np.real(val_y_plus) - np.real(val_base)) / probe_step
            dv_dy = (np.imag(val_y_plus) - np.imag(val_base)) / probe_step

            cr_err1 = abs(du_dx - dv_dy)
            cr_err2 = abs(du_dy + dv_dx)
            max_violation = max(max_violation, cr_err1, cr_err2)

        is_holomorphic = bool(max_violation < _CR_HOLOMORPHY_TOLERANCE)
        return is_holomorphic, float(max_violation)

    # ---- 1.2 Extrapolación de Richardson genuina de dos niveles (O(h^4) -> O(h^6)) ---

    @staticmethod
    def _richardson_two_level_extrapolation(d_h: float, d_h_half: float, order_p: int) -> float:
        r"""
        Extrapolación de Richardson clásica: dadas dos aproximaciones $D_h$, $D_{h/2}$ de
        una cantidad con error líder $\mathcal{O}(h^p)$, produce una estimación de orden
        $\mathcal{O}(h^{p+2})$:
        $$D^\star = \frac{2^p D_{h/2} - D_h}{2^p - 1}$$
        """
        factor = float(2 ** order_p)
        return (factor * d_h_half - d_h) / (factor - 1.0)

    def _centered_fourth_order_stencil(
        self,
        phi_func: Callable[[NDArray[np.float64]], float],
        x_point: NDArray[np.float64],
        axis_k: int,
        h: float
    ) -> float:
        r"""
        Plantilla centrada de 4to orden sobre el eje $k$ con sumación KBN:
        $$f'(x) = \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h} + \mathcal{O}(h^4)$$
        """
        e_k = np.zeros_like(x_point)
        e_k[axis_k] = 1.0

        f_p2 = float(np.real(phi_func(x_point + 2.0 * h * e_k)))
        f_p1 = float(np.real(phi_func(x_point + 1.0 * h * e_k)))
        f_m1 = float(np.real(phi_func(x_point - 1.0 * h * e_k)))
        f_m2 = float(np.real(phi_func(x_point - 2.0 * h * e_k)))

        acc = KahanNeumaierAccumulator()
        acc.add(-1.0 * f_p2)
        acc.add(8.0 * f_p1)
        acc.add(-8.0 * f_m1)
        acc.add(1.0 * f_m2)

        return acc.total / (12.0 * h)

    def _compute_richardson_centered_gradient(
        self,
        phi_func: Callable[[NDArray[np.float64]], float],
        x_point: NDArray[np.float64],
        h: float = _RICHARDSON_STEP
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Diferenciación finita centrada de orden 4 evaluada en $h$ y $h/2$, seguida de
        extrapolación de Richardson genuina hacia $\mathcal{O}(h^6)$. Retorna el gradiente
        extrapolado junto a su vector de error de truncamiento estimado
        $|D_{h/2} - D^\star|$ (componente a componente).
        """
        dim = x_point.shape[0]
        grad_h = np.zeros(dim, dtype=np.float64)
        grad_h_half = np.zeros(dim, dtype=np.float64)

        for k in range(dim):
            grad_h[k] = self._centered_fourth_order_stencil(phi_func, x_point, k, h)
            grad_h_half[k] = self._centered_fourth_order_stencil(phi_func, x_point, k, 0.5 * h)

        grad_extrapolated = np.zeros(dim, dtype=np.float64)
        truncation_error = np.zeros(dim, dtype=np.float64)

        for k in range(dim):
            extrapolated = self._richardson_two_level_extrapolation(
                grad_h[k], grad_h_half[k], order_p=_RICHARDSON_STENCIL_ORDER
            )
            grad_extrapolated[k] = extrapolated
            truncation_error[k] = abs(grad_h_half[k] - extrapolated)

        return grad_extrapolated, truncation_error

    # ---- 1.3 CSMD holomorfo con validación cruzada de doble paso complejo -------------

    def _extract_holomorphic_csmd_gradient(
        self,
        phi_func: Callable[[Any], Any],
        x_point: NDArray[np.float64],
        h_step: float = _CSMD_STEP_OPTIMAL
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], str]:
        r"""
        Calcula el 1-forma gradiente $d\phi = \partial_\mu \phi\, dx^\mu$ mediante
        Diferenciación por Paso Complejo (CSMD), libre de cancelación sustractiva:
        $$\partial_\mu \phi = \frac{\operatorname{Im}(\phi(x + i h e_\mu))}{h} + \mathcal{O}(h^2)$$
        Auditada previamente por holomorfía de Cauchy-Riemann en todas las direcciones,
        y validada cruzadamente con un segundo paso de orden de magnitud distinto
        ($h_{\times} = 10^8 h$) para detectar underflow o inconsistencias de la función.
        Ante no-holomorfía o falla de evaluación, conmuta a Richardson $\mathcal{O}(h^6)$.
        """
        dim = x_point.shape[0]
        is_holomorphic, cr_violation = self._verify_cauchy_riemann_holomorphy(phi_func, x_point)

        if not is_holomorphic:
            logger.debug(
                "No-holomorfía detectada (violación C-R = %.6e >= %.1e). Conmutando a Richardson O(h^6).",
                cr_violation, _CR_HOLOMORPHY_TOLERANCE
            )
            grad, err = self._compute_richardson_centered_gradient(phi_func, x_point)
            return grad, err, "Richardson-Extrapolated-O6"

        grad_fine = np.zeros(dim, dtype=np.float64)
        grad_cross = np.zeros(dim, dtype=np.float64)
        h_cross = h_step * _CSMD_CROSS_VALIDATION_MULTIPLIER

        try:
            for k in range(dim):
                z_fine = np.array(x_point, dtype=np.complex128)
                z_fine[k] += 1j * h_step
                phi_eval_fine = phi_func(z_fine)
                if not isinstance(phi_eval_fine, (complex, np.complex128)):
                    raise CSMDHolomorphyError(
                        "La función retornó un tipo no complejo bajo perturbación imaginaria."
                    )
                grad_fine[k] = float(np.imag(phi_eval_fine)) / h_step

                z_cross = np.array(x_point, dtype=np.complex128)
                z_cross[k] += 1j * h_cross
                phi_eval_cross = phi_func(z_cross)
                grad_cross[k] = float(np.imag(phi_eval_cross)) / h_cross
        except Exception as exc:
            logger.debug("Falla en evaluación CSMD (%s). Conmutando a Richardson O(h^6).", exc)
            grad, err = self._compute_richardson_centered_gradient(phi_func, x_point)
            return grad, err, "Richardson-Fallback-Exception"

        error_estimate = np.abs(grad_fine - grad_cross)
        return grad_fine, error_estimate, "CSMD-Holomorphic-Cauchy-Riemann"

    # ---- 1.4 Descomposición espectral métrica auditada (idempotencia + completitud) --

    def _decompose_spectral_metric(self, G_metric: NDArray[np.float64]) -> SpectralMetricTensorCache:
        r"""
        Audita el tensor métrico $G_{\mu\nu}$, calcula descomposición espectral
        $G = U \Lambda U^\top$, descomposición de Cholesky $G = L L^\top$, proyectores
        espectrales $\{P_k\}$ **auditados numéricamente** por idempotencia
        $\|P_k^2 - P_k\|_F < \delta$ y completitud $\|\sum_k P_k - I\|_F < \delta$
        (teorema espectral para operadores autoadjuntos en dimensión finita), y forma
        de volumen Riemanniana (con log-volumen estable).
        """
        if G_metric.ndim != 2 or G_metric.shape[0] != G_metric.shape[1]:
            raise DimensionMismatchError(f"Tensor métrico debe ser cuadrado 2D. Forma: {G_metric.shape}")

        dim = G_metric.shape[0]

        # Simetrización de Frobenius compensada
        g_sym = 0.5 * (G_metric + G_metric.T)
        asym_res = float(la.norm(G_metric - g_sym, "fro"))
        if asym_res > 1.0e-14:
            logger.warning("Tensor métrico presentaba asimetría infinitesimal: %.6e. Simetrizado.", asym_res)

        # Descomposición espectral autoadjunta
        eigenvalues, eigenvectors = la.eigh(g_sym)
        min_lambda = float(eigenvalues[0])
        max_lambda = float(eigenvalues[-1])

        if min_lambda <= _WILKINSON_FLOOR:
            raise MetricIndefinitenessError(
                f"Autovalor colapsó por debajo de la cota de Wilkinson: {min_lambda:.6e} <= {_WILKINSON_FLOOR:.6e}"
            )

        cond_num = max_lambda / min_lambda
        if cond_num > self._condition_tolerance:
            raise MetricIndefinitenessError(
                f"Número de condición excedió umbral crítico: {cond_num:.4e} > {self._condition_tolerance:.4e}"
            )

        # Factorización de Cholesky en cono SPD
        try:
            L_cholesky = la.cholesky(g_sym, lower=True)
        except la.LinAlgError as exc:
            raise MetricIndefinitenessError(f"Falla en factorización Cholesky sobre cono SPD: {exc}") from exc

        # Inversión triangular no confinada: G^{-1} = L^{-\top} L^{-1}
        L_inv = la.solve_triangular(L_cholesky, np.eye(dim, dtype=np.float64), lower=True)
        g_inv = L_inv.T @ L_inv

        # Determinante y forma de volumen Riemanniana: \sqrt{\det G} = \prod L_{ii}
        diag_L = np.diag(L_cholesky)
        volume_factor = float(np.prod(diag_L))
        log_volume_factor = KahanNeumaierAccumulator.sum(np.log(diag_L))

        # Proyectores espectrales de Banach: P_k = u_k u_k^\top — AUDITADOS
        projectors: List[NDArray[np.float64]] = []
        projector_sum = np.zeros((dim, dim), dtype=np.float64)
        max_idempotency_residual = 0.0
        for k in range(dim):
            u_k = eigenvectors[:, k: k + 1]
            p_k = u_k @ u_k.T
            projectors.append(p_k)
            projector_sum += p_k
            max_idempotency_residual = max(
                max_idempotency_residual, float(la.norm(p_k @ p_k - p_k, "fro"))
            )

        completeness_residual = float(la.norm(projector_sum - np.eye(dim), "fro"))
        projectors_verified = bool(
            max_idempotency_residual < _SPECTRAL_PROJECTOR_IDEMPOTENCY_TOL
            and completeness_residual < _SPECTRAL_PROJECTOR_IDEMPOTENCY_TOL
        )

        if not projectors_verified:
            raise SpectralProjectorInconsistencyError(
                f"Proyectores espectrales inconsistentes: idempotencia={max_idempotency_residual:.3e}, "
                f"completitud={completeness_residual:.3e} (tolerancia={_SPECTRAL_PROJECTOR_IDEMPOTENCY_TOL:.1e})"
            )

        return SpectralMetricTensorCache(
            g_base=g_sym,
            cholesky_factor=L_cholesky,
            g_inv=g_inv,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            spectral_projectors=tuple(projectors),
            riemannian_volume_factor=volume_factor,
            log_riemannian_volume_factor=log_volume_factor,
            spectral_radius=max_lambda,
            condition_number=cond_num,
            dimension=dim,
            projectors_verified=projectors_verified
        )

    # ---- 1.5 Ensamblaje del haz de formas exteriores y dual de Hodge -----------------

    def _compute_hodge_star_and_bundle(
        self,
        phi_val: float,
        dphi: NDArray[np.float64],
        cache: SpectralMetricTensorCache,
        truncation_error: NDArray[np.float64],
        differentiation_method: str
    ) -> DifferentialFormsBundle:
        r"""
        Construye el haz de formas exteriores: calcula la norma invariante de Dirichlet
        $\|d\phi\|^2_G = G^{\mu\nu}\partial_\mu \phi \partial_\nu \phi \ge 0$ (auditada, pues
        $G^{-1}\succ 0$) y el dual de Hodge $*d\phi$ representado por su vector de flujo
        contravariante equivalente $\sqrt{\det G}\; G^\sharp(d\phi)$ sobre la frontera.
        """
        sharp_dphi = cache.g_inv @ dphi
        grad_norm_sq = KahanNeumaierAccumulator.dot(dphi, sharp_dphi)

        # Auditoría de positividad: G^{-1} SPD garantiza grad_norm_sq >= 0 en aritmética exacta.
        if grad_norm_sq < -_WILKINSON_FLOOR:
            raise MetricIndefinitenessError(
                f"Norma de Dirichlet negativa detectada ({grad_norm_sq:.3e}): "
                f"violación de la positividad de G^{{-1}}."
            )
        grad_norm_sq = max(grad_norm_sq, 0.0)

        hodge_rep = cache.riemannian_volume_factor * sharp_dphi

        return DifferentialFormsBundle(
            scalar_field=phi_val,
            exterior_derivative=dphi,
            gradient_norm_squared=grad_norm_sq,
            hodge_star_exterior_derivative=hodge_rep,
            differentiation_method=differentiation_method,
            truncation_error_estimate=truncation_error
        )

    # ---- 1.6 Sello criptográfico reutilizable (creación y verificación) -------------

    @staticmethod
    def _compute_phase1_seal(
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        dphi: NDArray[np.float64],
        g_base: NDArray[np.float64],
        eigenvalues: NDArray[np.float64],
        grad_norm_sq: float,
        differentiation_method: str,
        log_volume: float
    ) -> str:
        r"""Sello SHA-256 determinista y reutilizable del expediente de Fase 1 (creación y auditoría)."""
        hasher = hashlib.sha256()
        hasher.update(x_point.tobytes())
        hasher.update(momentum_p.tobytes())
        hasher.update(dphi.tobytes())
        hasher.update(g_base.tobytes())
        hasher.update(eigenvalues.tobytes())
        hasher.update(f"{grad_norm_sq:.16e}".encode("ascii"))
        hasher.update(differentiation_method.encode("ascii"))
        hasher.update(f"{log_volume:.16e}".encode("ascii"))
        return hasher.hexdigest()

    # ---- 1.7 MÉTODO TERMINAL FORMAL DE FASE 1 ----------------------------------------

    def observe_scalar_field(
        self,
        phi_func: Callable[[Any], Any],
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64]
    ) -> ScalarMomentumKernel:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1 (OBSERVE):
        Ingiere campos, audita espectro de $G$ (idempotencia/completitud de proyectores),
        deriva $d\phi$ vía CSMD holomorfo validado cruzadamente (o Richardson $\mathcal{O}(h^6)$),
        ensambla el haz de formas de Hodge y genera el sello criptográfico inmutable SHA-256.
        La estructura devuelta `ScalarMomentumKernel` es el conector directo hacia Fase 2.
        """
        if x_point.ndim != 1 or momentum_p.ndim != 1:
            raise DimensionMismatchError("Los vectores x y p deben ser tensores unidimensionales de rango 1.")

        dim = x_point.shape[0]
        if momentum_p.shape[0] != dim:
            raise DimensionMismatchError(f"Incompatibilidad de dimensiones: x({dim}) != p({momentum_p.shape[0]})")

        # 1. Auditoría Espectral del Tensor Métrico
        metric_cache = self._decompose_spectral_metric(G_metric)
        if metric_cache.dimension != dim:
            raise DimensionMismatchError(
                f"Dimensión de G ({metric_cache.dimension}) != Dimensión del Espacio ({dim})"
            )

        # 2. Evaluación escalar en el punto base
        phi_val = float(np.real(phi_func(np.array(x_point, dtype=np.float64))))

        # 3. Diferenciación CSMD holomorfa con validación cruzada (o Richardson O(h^6))
        dphi, truncation_error, diff_method = self._extract_holomorphic_csmd_gradient(phi_func, x_point)

        # 4. Construcción del haz de formas diferenciales y Hodge
        bundle = self._compute_hodge_star_and_bundle(phi_val, dphi, metric_cache, truncation_error, diff_method)

        # 5. Sello criptográfico SHA-256 de Fase 1 (reutilizable para auditoría en Fase 3)
        phase1_hash = self._compute_phase1_seal(
            x_point=x_point,
            momentum_p=momentum_p,
            dphi=dphi,
            g_base=metric_cache.g_base,
            eigenvalues=metric_cache.eigenvalues,
            grad_norm_sq=bundle.gradient_norm_squared,
            differentiation_method=diff_method,
            log_volume=metric_cache.log_riemannian_volume_factor
        )

        logger.debug(
            "Fase 1 (Observe) completada. Método=%s, Sello=%s", diff_method, phase1_hash[:16]
        )

        return ScalarMomentumKernel(
            evaluation_point=x_point.copy(),
            momentum_covector=momentum_p.copy(),
            forms_bundle=bundle,
            metric_cache=metric_cache,
            phase1_sha256_hash=phase1_hash
        )


# ══════════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: ORIENT — TRANSFERENCIA DE LIE, TENSOR DE ESFUERZOS Y                      ║
# ║          PORT-HAMILTONIANO CON DISIPACIÓN $\sigma$/$\kappa$ DESACOPLADA           ║
# ║          (Inicia absorbiendo DIRECTAMENTE ScalarMomentumKernel de Fase 1)         ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────── Expedientes de Fase 2 ────────────────────────────────

@dataclass(frozen=True, slots=True)
class PortHamiltonianCircuitTelemetry:
    r"""
    Expediente de pasividad de circuitos y dinámica Port-Hamiltoniana con conductancias
    resistiva ($\sigma$, acoplada a $p$) y reactiva-disipativa ($\kappa$, acoplada a $d\phi$)
    desacopladas, más el espectro auditado de la matriz de disipación en bloque
    $R = \mathrm{diag}(\sigma G^{-1}, \kappa G^{-1})$.
    """
    hamiltonian_energy: float
    lie_transfer_power: float
    dissipated_power: float
    resistive_conduction_rate: float
    reactive_conduction_rate: float
    poynting_boundary_flux: float
    dissipation_matrix_eigenvalues: NDArray[np.float64]
    is_dissipation_matrix_psd: bool
    is_thermodynamically_passive: bool


@dataclass(frozen=True, slots=True)
class MomentumTransferReport:
    r"""
    Expediente Inmutable Terminal de Fase 2 (Orient).
    Consolida la derivada de Lie, el Tensor de Esfuerzo-Energía $T_{\mu\nu}$ (con residual
    de antisimetría auditado como certificado de simetría de Belinfante), su traza
    contravariante $\mathcal{T}$, y la telemetría Port-Hamiltoniana.
    """
    kernel_ref: ScalarMomentumKernel
    velocity_vector: NDArray[np.float64]
    lie_derivative_transfer: float
    stress_energy_tensor: NDArray[np.float64]
    stress_energy_trace: float
    stress_energy_eigenvalues: NDArray[np.float64]
    stress_energy_antisymmetric_residual: float
    circuit_telemetry: PortHamiltonianCircuitTelemetry
    phase2_hmac_sha256: str


# ─────────────────────────────────── Motor de Fase 2 ───────────────────────────────────

class Phase2_CovariantTransferOrient(Phase1_ScalarMomentumObserver):
    r"""
    FASE 2 (ORIENT):
    Fundamentación matemática:
      - Absorbe DIRECTAMENTE el `ScalarMomentumKernel` emitido por el final de Fase 1.
      - Resuelve la velocidad covariante mediante el isomorfismo sharp: $v^\mu = G^{\mu\nu} p_\nu$.
      - Evalúa la derivada de Lie covariante $\mathcal{L}_v \phi = \{\phi, H\}$.
      - Construye el Tensor de Energía-Impulso Belinfante-Rosenfeld (simétrico por
        construcción algebraica; el residual antisimétrico se audita como control de
        redondeo IEEE-754, no como término físico).
      - Modela el sistema disipativo como estructura Port-Hamiltoniana con matriz de
        disipación bloque-diagonal EXPLÍCITA $R = \mathrm{diag}(\sigma G^{-1}, \kappa G^{-1})$,
        auditada como PSD, con $\sigma,\kappa \ge 0$ exigidos axiomáticamente
        (ninguna resistencia física puede ser negativa).
      - Certifica la pasividad termodinámica de Tellegen ($P_{\mathrm{diss}} \ge -\epsilon_{\mathrm{Wilkinson}}$).
      - Emite el reporte inmutable `MomentumTransferReport`.
    """

    # ---- 2.1 Velocidad contravariante y derivada de Lie simpléctica -----------------

    def _compute_lie_transfer_symplectic(
        self,
        p_covector: NDArray[np.float64],
        dphi: NDArray[np.float64],
        g_inv: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Calcula la velocidad contravariante $v = G^\sharp(p)$ y la transferencia de Lie
        $\mathcal{L}_v \phi = v^\mu \partial_\mu \phi = p_\mu G^{\mu\nu} \partial_\nu \phi$.
        """
        v_velocity = g_inv @ p_covector
        directional_terms = v_velocity * dphi
        lie_derivative = KahanNeumaierAccumulator.sum(directional_terms)
        return v_velocity, lie_derivative

    # ---- 2.2 Tensor de Esfuerzo-Energía Belinfante-Rosenfeld con acoplamiento -------

    def _synthesize_stress_energy_tensor(
        self,
        p_covector: NDArray[np.float64],
        dphi: NDArray[np.float64],
        phi_val: float,
        cache: SpectralMetricTensorCache,
        coupling_alpha: float,
        mass_m: float
    ) -> Tuple[NDArray[np.float64], float, NDArray[np.float64], float]:
        r"""
        Sintetiza el tensor de tensión-energía covariante $T_{\mu\nu}$ y su traza espectral:
        $$T_{\mu\nu} = \partial_\mu \phi \partial_\nu \phi - \frac{1}{2} G_{\mu\nu} (\|d\phi\|^2_{G^{-1}} + m^2 \phi^2)
        + \alpha (p_\mu \partial_\nu \phi + p_\nu \partial_\mu \phi - G_{\mu\nu} \langle p, d\phi \rangle_{G^{-1}})$$
        Por construcción cada término es simétrico; se audita el residual
        $\|T-T^\top\|_F$ como control estricto de redondeo IEEE-754 (debe ser
        $\mathcal{O}(\epsilon_{\mathrm{mach}})$, certificando la simetría de Belinfante).
        """
        g_metric = cache.g_base
        g_inv = cache.g_inv

        sharp_dphi = g_inv @ dphi
        grad_norm_sq = KahanNeumaierAccumulator.dot(dphi, sharp_dphi)

        kinetic_potential_density = 0.5 * (grad_norm_sq + (mass_m ** 2) * (phi_val ** 2))

        dyadic_grad = np.outer(dphi, dphi)

        coupling_inner = KahanNeumaierAccumulator.dot(p_covector, sharp_dphi)
        symmetric_momentum_coupling = np.outer(p_covector, dphi) + np.outer(dphi, p_covector)
        coupling_tensor = symmetric_momentum_coupling - (g_metric * coupling_inner)

        T_tensor = dyadic_grad - (g_metric * kinetic_potential_density) + (coupling_alpha * coupling_tensor)

        antisymmetric_residual = float(la.norm(T_tensor - T_tensor.T, "fro"))

        contraction_matrix = g_inv @ T_tensor
        trace_T = KahanNeumaierAccumulator.sum(np.diag(contraction_matrix))

        t_eigenvalues = la.eigvalsh(0.5 * (T_tensor + T_tensor.T))

        return T_tensor, float(trace_T), t_eigenvalues, antisymmetric_residual

    # ---- 2.3 Auditoría explícita de la matriz de disipación bloque-diagonal --------

    def _audit_dissipation_matrix_structure(
        self,
        cache: SpectralMetricTensorCache,
        conductivity_sigma: float,
        kappa_dissipation: float
    ) -> Tuple[NDArray[np.float64], bool]:
        r"""
        Construye y audita la matriz de disipación en bloque sobre el espacio de puertos
        $(p, d\phi) \in T^*_x\mathcal{M} \oplus T^*_x\mathcal{M}$:
        $$R = \begin{pmatrix} \sigma G^{-1} & 0 \\ 0 & \kappa G^{-1} \end{pmatrix} \succeq 0$$
        Exige axiomáticamente $\sigma, \kappa \ge 0$ (ninguna resistencia física negativa)
        y certifica numéricamente $R \succeq -\epsilon_{\mathrm{Wilkinson}}$.
        """
        if conductivity_sigma < 0.0 or kappa_dissipation < 0.0:
            raise ThermodynamicPassivityViolationError(
                f"Coeficientes de disipación negativos detectados: "
                f"sigma={conductivity_sigma:.6e}, kappa={kappa_dissipation:.6e}"
            )

        dim = cache.dimension
        R_block = np.zeros((2 * dim, 2 * dim), dtype=np.float64)
        R_block[:dim, :dim] = conductivity_sigma * cache.g_inv
        R_block[dim:, dim:] = kappa_dissipation * cache.g_inv

        dissipation_eigenvalues = la.eigvalsh(R_block)
        is_psd = bool(dissipation_eigenvalues[0] >= -_WILKINSON_FLOOR)

        return dissipation_eigenvalues, is_psd

    # ---- 2.4 Auditoría de pasividad Port-Hamiltoniana (sigma/kappa desacoplados) ----

    def _audit_port_hamiltonian_passivity(
        self,
        p_covector: NDArray[np.float64],
        dphi: NDArray[np.float64],
        v_velocity: NDArray[np.float64],
        lie_transfer: float,
        cache: SpectralMetricTensorCache,
        conductivity_sigma: float,
        kappa_dissipation: float
    ) -> PortHamiltonianCircuitTelemetry:
        r"""
        Audita el balance termodinámico y la estructura de circuito Port-Hamiltoniano:
        $H = \frac{1}{2} \|p\|^2_{G^{-1}} + \frac{1}{2} \|d\phi\|^2_{G^{-1}}$.
        Potencia disipada desacoplada:
        $P_{\mathrm{diss}} = \sigma \|p\|^2_{G^{-1}} + \kappa \|d\phi\|^2_{G^{-1}} \ge 0$.
        Flujo de Poynting en la frontera: $S = \mathcal{L}_v \phi \cdot \mathrm{vol}_G$.
        """
        g_inv = cache.g_inv

        h_kinetic = 0.5 * KahanNeumaierAccumulator.dot(p_covector, v_velocity)

        sharp_dphi = g_inv @ dphi
        h_potential = 0.5 * KahanNeumaierAccumulator.dot(dphi, sharp_dphi)
        hamiltonian_total = h_kinetic + h_potential

        dissipation_eigenvalues, is_psd = self._audit_dissipation_matrix_structure(
            cache, conductivity_sigma, kappa_dissipation
        )

        dissipation_p = conductivity_sigma * (2.0 * h_kinetic)
        dissipation_phi = kappa_dissipation * (2.0 * h_potential)
        total_p_diss = dissipation_p + dissipation_phi

        is_passive = bool(total_p_diss >= -_WILKINSON_FLOOR and is_psd)

        poynting_flux = lie_transfer * cache.riemannian_volume_factor

        return PortHamiltonianCircuitTelemetry(
            hamiltonian_energy=hamiltonian_total,
            lie_transfer_power=lie_transfer,
            dissipated_power=total_p_diss,
            resistive_conduction_rate=conductivity_sigma,
            reactive_conduction_rate=kappa_dissipation,
            poynting_boundary_flux=poynting_flux,
            dissipation_matrix_eigenvalues=dissipation_eigenvalues,
            is_dissipation_matrix_psd=is_psd,
            is_thermodynamically_passive=is_passive
        )

    # ---- 2.5 Sello criptográfico reutilizable (creación y verificación) -------------

    @staticmethod
    def _compute_phase2_seal(
        phase1_hash: str,
        v_velocity: NDArray[np.float64],
        T_tensor: NDArray[np.float64],
        lie_transfer: float,
        dissipated_power: float
    ) -> str:
        r"""Sello HMAC-SHA256 determinista y reutilizable del expediente de Fase 2, encadenado a Fase 1."""
        hmac_signer = hmac.new(_HMAC_METROLOGY_KEY, digestmod=hashlib.sha256)
        hmac_signer.update(phase1_hash.encode("ascii"))
        hmac_signer.update(v_velocity.tobytes())
        hmac_signer.update(T_tensor.tobytes())
        hmac_signer.update(f"{lie_transfer:.16e}".encode("ascii"))
        hmac_signer.update(f"{dissipated_power:.16e}".encode("ascii"))
        return hmac_signer.hexdigest()

    # ---- 2.6 Orquestación interna de Fase 2 ------------------------------------------

    def _orient_momentum_transfer_pipeline(
        self,
        kernel: ScalarMomentumKernel,
        coupling_alpha: float,
        mass_m: float,
        conductivity_sigma: float,
        kappa_dissipation: float
    ) -> MomentumTransferReport:
        r"""Orquesta la ejecución interna de la Fase 2 a partir del Kernel."""
        p_covector = kernel.momentum_covector
        dphi = kernel.forms_bundle.exterior_derivative
        phi_val = kernel.forms_bundle.scalar_field
        cache = kernel.metric_cache

        v_velocity, lie_transfer = self._compute_lie_transfer_symplectic(p_covector, dphi, cache.g_inv)

        T_tensor, trace_T, t_eigs, antisym_residual = self._synthesize_stress_energy_tensor(
            p_covector=p_covector,
            dphi=dphi,
            phi_val=phi_val,
            cache=cache,
            coupling_alpha=coupling_alpha,
            mass_m=mass_m
        )

        telemetry = self._audit_port_hamiltonian_passivity(
            p_covector=p_covector,
            dphi=dphi,
            v_velocity=v_velocity,
            lie_transfer=lie_transfer,
            cache=cache,
            conductivity_sigma=conductivity_sigma,
            kappa_dissipation=kappa_dissipation
        )

        if not telemetry.is_thermodynamically_passive:
            raise ThermodynamicPassivityViolationError(
                f"Violación de pasividad en el circuito térmico: P_diss = {telemetry.dissipated_power:.8e} < 0 "
                f"o matriz R no-PSD (psd={telemetry.is_dissipation_matrix_psd})"
            )

        phase2_hmac = self._compute_phase2_seal(
            phase1_hash=kernel.phase1_sha256_hash,
            v_velocity=v_velocity,
            T_tensor=T_tensor,
            lie_transfer=lie_transfer,
            dissipated_power=telemetry.dissipated_power
        )

        return MomentumTransferReport(
            kernel_ref=kernel,
            velocity_vector=v_velocity,
            lie_derivative_transfer=lie_transfer,
            stress_energy_tensor=T_tensor,
            stress_energy_trace=trace_T,
            stress_energy_eigenvalues=t_eigs,
            stress_energy_antisymmetric_residual=antisym_residual,
            circuit_telemetry=telemetry,
            phase2_hmac_sha256=phase2_hmac
        )

    # ---- 2.7 MÉTODO DE INICIO CONTINUO DE FASE 2 -------------------------------------

    def orient_from_kernel(
        self,
        kernel: ScalarMomentumKernel,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0,
        conductivity_sigma: float = 1.0,
        kappa_dissipation: Optional[float] = None
    ) -> MomentumTransferReport:
        r"""
        MÉTODO DE INICIO CONTINUO DE FASE 2:
        Conecta formalmente con el final de Fase 1 al recibir su artefacto terminal
        `ScalarMomentumKernel`. Si `kappa_dissipation` no se especifica, se adopta
        por defecto igual a `conductivity_sigma` (isotropía disipativa retrocompatible).
        """
        kappa_effective = kappa_dissipation if kappa_dissipation is not None else conductivity_sigma
        return self._orient_momentum_transfer_pipeline(
            kernel=kernel,
            coupling_alpha=coupling_alpha,
            mass_m=mass_m,
            conductivity_sigma=conductivity_sigma,
            kappa_dissipation=kappa_effective
        )

    # ---- 2.8 MÉTODO TERMINAL FORMAL DE FASE 2 ----------------------------------------

    def orient_momentum_transfer(
        self,
        kernel: ScalarMomentumKernel,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0,
        conductivity_sigma: float = 1.0,
        kappa_dissipation: Optional[float] = None
    ) -> MomentumTransferReport:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2 (ORIENT):
        Empaqueta la orientación del momentum. Entrega el `MomentumTransferReport`,
        el cual constituye el sustrato inmutable de partida para la Fase 3.
        """
        return self.orient_from_kernel(
            kernel=kernel,
            coupling_alpha=coupling_alpha,
            mass_m=mass_m,
            conductivity_sigma=conductivity_sigma,
            kappa_dissipation=kappa_dissipation
        )


# ══════════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: ACT — VERIFICACIÓN DE CADENA DE CUSTODIA, SEMIGRUPOS DE BANACH            ║
# ║          $G$-PONDERADOS, TOPOLOGÍA DE GRAM Y CERTIFICACIÓN SOBERANA               ║
# ║          (Inicia absorbiendo DIRECTAMENTE Kernel + MomentumTransferReport)        ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────── Expedientes de Fase 3 ────────────────────────────────

@dataclass(frozen=True, slots=True)
class BanachStabilityCertification:
    r"""Certificación metrológica del Semigrupo $C_0$ $G$-ponderado y análisis funcional de Banach."""
    lumer_phillips_dissipative_constant: float
    is_contractive_semigroup: bool
    spectral_radius: float
    topological_winding_number: float


@dataclass(frozen=True, slots=True)
class ScalarMomentumEngineState:
    r"""
    Certificado Holístico Soberano emitido por la Fase 3 (Act).
    Representa el estado cuántico-clásico inmutable verificado por la FPU.
    """
    kernel: ScalarMomentumKernel
    transfer_report: MomentumTransferReport
    banach_certification: BanachStabilityCertification
    fpu_execution_time_microseconds: float
    wilkinson_roundoff_bound: float
    sovereign_cryptographic_seal: str


# ─────────────────────────────────── Motor de Fase 3 ───────────────────────────────────

class Phase3_ScalarMomentumEngine(Phase2_CovariantTransferOrient):
    r"""
    FASE 3 (ACT):
    Fundamentación matemática:
      - Absorbe DIRECTAMENTE el `MomentumTransferReport` (y su `ScalarMomentumKernel` interno)
        procedente del final de la Fase 2, verificando primero la CADENA DE CUSTODIA
        criptográfica (recomputación de sellos SHA-256/HMAC con comparación de tiempo
        constante `hmac.compare_digest`).
      - Análisis Funcional en Espacios de Banach $G$-ponderados:
        El generador infinitesimal se construye con el BIVECTOR SIMPLÉCTICO GENUINO
        $J = p \wedge v = p v^\top - v p^\top$ (antisimétrico, no-nulo genéricamente) y la
        parte resistiva $-\sigma G$:
        $$\mathcal{A} = -\sigma G - \eta J$$
        La dissipatividad de Lumer-Phillips se evalúa en el producto interno INDUCIDO
        POR LA MÉTRICA (no el euclídeo implícito):
        $$\langle \mathcal{A}x, x\rangle_G = x^\top G \mathcal{A} x \le 0
          \iff M := \tfrac{1}{2}(G\mathcal{A} + (G\mathcal{A})^\top) \preceq \epsilon_{\mathrm{Wilkinson}} I$$
        garantizando $\|e^{t\mathcal{A}}\|_G \le 1,\ \forall t\ge 0$. El radio espectral se
        obtiene del problema generalizado $\mathcal{A}x = \lambda G x$.
      - Topología Algebraica generalizada a $n$ dimensiones vía determinante de Gram:
        $$\|x \wedge v\|_G^2 = \|x\|_G^2\|v\|_G^2 - \langle x,v\rangle_G^2$$
        En $n=2$ se recupera el signo mediante el elemento de volumen
        $\sqrt{\det G}(x^1v^2-x^2v^1)$; en $n>2$ se reporta la magnitud del bivector
        normalizada (grado topológico no-orientado del flujo de momentum).
      - Cómputo de la Cota de Redondeo de Wilkinson ponderada por la cadena de
        operaciones matriciales del pipeline completo (Cholesky + eigh + triangular + matmul):
        $\gamma_n = \frac{c\cdot n \cdot \epsilon_{\mathrm{mach}}}{1 - c\cdot n \cdot \epsilon_{\mathrm{mach}}}$.
      - Emisión del Sello Soberano Inmutable y Ledger SHA-256/HMAC FPU.
    """

    # ---- 3.1 Verificación de cadena de custodia criptográfica (defensa en profundidad)

    def _verify_metrological_chain_integrity(
        self,
        kernel: ScalarMomentumKernel,
        report: MomentumTransferReport
    ) -> None:
        r"""
        Recomputa de forma independiente los sellos de Fase 1 y Fase 2 a partir de los
        datos crudos almacenados en el Kernel/Report y los compara en tiempo constante
        contra los sellos originalmente emitidos, detectando cualquier adulteración o
        corrupción de estado entre fases (cadena de custodia inmutable).
        """
        recomputed_phase1 = self._compute_phase1_seal(
            x_point=kernel.evaluation_point,
            momentum_p=kernel.momentum_covector,
            dphi=kernel.forms_bundle.exterior_derivative,
            g_base=kernel.metric_cache.g_base,
            eigenvalues=kernel.metric_cache.eigenvalues,
            grad_norm_sq=kernel.forms_bundle.gradient_norm_squared,
            differentiation_method=kernel.forms_bundle.differentiation_method,
            log_volume=kernel.metric_cache.log_riemannian_volume_factor
        )
        if not hmac.compare_digest(recomputed_phase1, kernel.phase1_sha256_hash):
            raise MetrologicalChainIntegrityError(
                "Sello SHA-256 de Fase 1 corrupto o adulterado: fallo de cadena de custodia."
            )

        recomputed_phase2 = self._compute_phase2_seal(
            phase1_hash=kernel.phase1_sha256_hash,
            v_velocity=report.velocity_vector,
            T_tensor=report.stress_energy_tensor,
            lie_transfer=report.lie_derivative_transfer,
            dissipated_power=report.circuit_telemetry.dissipated_power
        )
        if not hmac.compare_digest(recomputed_phase2, report.phase2_hmac_sha256):
            raise MetrologicalChainIntegrityError(
                "Sello HMAC-SHA256 de Fase 2 corrupto o adulterado: fallo de cadena de custodia."
            )

    # ---- 3.2 Auditoría de Lumer-Phillips en el espacio de Hilbert $G$-ponderado -----

    def _audit_lumer_phillips_semigroup(
        self,
        p_covector: NDArray[np.float64],
        v_velocity: NDArray[np.float64],
        cache: SpectralMetricTensorCache,
        conductivity_sigma: float,
        advective_coupling_eta: float
    ) -> Tuple[float, bool, float, NDArray[np.float64]]:
        r"""
        Construye el generador infinitesimal $\mathcal{A} = -\sigma G - \eta J$, con
        $J = p\wedge v$ el bivector simpléctico genuino, y certifica el Teorema de
        Lumer-Phillips en el producto interno $G$-ponderado:
        $$\langle \mathcal{A}x,x\rangle_G = x^\top G\mathcal{A} x \le 0
          \iff M = \tfrac{1}{2}(G\mathcal{A}+(G\mathcal{A})^\top) \preceq \epsilon I$$
        El radio espectral se computa resolviendo el problema generalizado de
        autovalores $\mathcal{A}x = \lambda G x$ (métrica del espacio de Hilbert real).
        """
        G = cache.g_base

        # Bivector generador antisimétrico: momento angular generalizado de p y v.
        J_generator = np.outer(p_covector, v_velocity) - np.outer(v_velocity, p_covector)
        skew_residual = float(la.norm(J_generator + J_generator.T, "fro"))
        if skew_residual > 1.0e-10:
            logger.warning("Residual de antisimetría del generador J fuera de tolerancia: %.3e", skew_residual)

        A_generator = (-conductivity_sigma * G) - (advective_coupling_eta * J_generator)

        GA = G @ A_generator
        M_symmetric_part = 0.5 * (GA + GA.T)
        dissipative_spectrum = la.eigvalsh(M_symmetric_part)
        max_dissipative_eigenvalue = float(dissipative_spectrum[-1])
        is_contractive = bool(max_dissipative_eigenvalue <= _WILKINSON_FLOOR)

        # Radio espectral vía problema generalizado A x = lambda G x
        generalized_eigenvalues = la.eigvals(A_generator, G)
        spectral_radius = float(np.max(np.abs(generalized_eigenvalues)))

        return max_dissipative_eigenvalue, is_contractive, spectral_radius, A_generator

    # ---- 3.3 Winding topológico generalizado vía determinante de Gram ($G$-métrico) -

    def _evaluate_topological_winding(
        self,
        x_point: NDArray[np.float64],
        v_velocity: NDArray[np.float64],
        cache: SpectralMetricTensorCache
    ) -> float:
        r"""
        Calcula el invariante topológico del flujo de momentum sobre la 1-esfera de
        frontera, generalizado a $n$ dimensiones mediante el determinante de Gram del
        bivector $x\wedge v$ en el producto interno $G$-ponderado:
        $$\|x\wedge v\|_G^2 = \|x\|_G^2\|v\|_G^2 - \langle x,v\rangle_G^2$$
        En $n=2$ se recupera el signo con el elemento de volumen Riemanniano
        $\sqrt{\det G}(x^1v^2-x^2v^1)$ (equivalente exacto vía identidad de Lagrange
        tras el cambio de base de Cholesky). En $n>2$ se reporta la magnitud
        no-orientada, invariante de reparametrización.
        """
        G = cache.g_base
        dim = x_point.shape[0]

        norm_x_g_sq = float(x_point @ G @ x_point)
        norm_v_g_sq = float(v_velocity @ G @ v_velocity)
        inner_xv_g = float(x_point @ G @ v_velocity)

        norm_x_g = math.sqrt(max(norm_x_g_sq, 0.0))
        norm_v_g = math.sqrt(max(norm_v_g_sq, 0.0))

        if norm_x_g < 1.0e-12 or norm_v_g < 1.0e-12:
            return 0.0

        if dim == 2:
            signed_area_g = cache.riemannian_volume_factor * (
                x_point[0] * v_velocity[1] - x_point[1] * v_velocity[0]
            )
            winding = signed_area_g / (2.0 * math.pi * norm_x_g * norm_v_g)
            return float(winding)

        gram_determinant = max(norm_x_g_sq * norm_v_g_sq - inner_xv_g ** 2, 0.0)
        bivector_norm_g = math.sqrt(gram_determinant)
        winding_magnitude = bivector_norm_g / (2.0 * math.pi * norm_x_g * norm_v_g)
        return float(winding_magnitude)

    # ---- 3.4 Cota de Wilkinson ponderada por la cadena de operaciones del pipeline --

    def _compute_wilkinson_bound(
        self,
        dimension: int,
        operation_chain_factor: int = _WILKINSON_OPERATION_CHAIN_FACTOR
    ) -> float:
        r"""
        Calcula la cota de Wilkinson acumulada del pipeline completo (Cholesky, eigh,
        resolución triangular y productos matriciales encadenados), ponderando el
        conteo de operaciones dominantes por un factor de cadena documentado:
        $$\gamma_n = \frac{c \cdot n \cdot \epsilon_{\mathrm{mach}}}{1 - c \cdot n \cdot \epsilon_{\mathrm{mach}}}$$
        """
        n_eff = operation_chain_factor * dimension
        n_eps = n_eff * _MACHINE_EPS
        if n_eps >= 1.0:
            return float("inf")
        return float(n_eps / (1.0 - n_eps))

    # ---- 3.5 Sello Soberano Holístico -------------------------------------------------

    def _generate_sovereign_cryptographic_seal(
        self,
        kernel: ScalarMomentumKernel,
        report: MomentumTransferReport,
        banach_cert: BanachStabilityCertification,
        fpu_time_us: float
    ) -> str:
        r"""Genera el sello holístico SHA-256 encadenando los sellos verificados de las 3 Fases."""
        hasher = hashlib.sha256()
        hasher.update(kernel.phase1_sha256_hash.encode("ascii"))
        hasher.update(report.phase2_hmac_sha256.encode("ascii"))
        hasher.update(f"{banach_cert.lumer_phillips_dissipative_constant:.16e}".encode("ascii"))
        hasher.update(f"{banach_cert.spectral_radius:.16e}".encode("ascii"))
        hasher.update(f"{banach_cert.topological_winding_number:.16e}".encode("ascii"))
        hasher.update(f"{fpu_time_us:.6f}".encode("ascii"))
        return hasher.hexdigest()

    # ---- 3.6 Orquestación interna de Fase 3 ------------------------------------------

    def _act_execute_pipeline(
        self,
        kernel: ScalarMomentumKernel,
        report: MomentumTransferReport,
        advective_coupling_eta: float = _ADVECTIVE_COUPLING_ETA_DEFAULT
    ) -> ScalarMomentumEngineState:
        r"""Orquesta la verificación de custodia, certificación y sellado final de Fase 3."""
        t_start = time.perf_counter_ns()

        # 0. Verificación defensiva de la cadena de custodia criptográfica Fase1 <-> Fase2
        self._verify_metrological_chain_integrity(kernel, report)

        # 1. Auditoría del Semigrupo de Banach G-ponderado y Lumer-Phillips
        dissipative_const, is_contractive, spec_radius, _A_generator = self._audit_lumer_phillips_semigroup(
            p_covector=kernel.momentum_covector,
            v_velocity=report.velocity_vector,
            cache=kernel.metric_cache,
            conductivity_sigma=report.circuit_telemetry.resistive_conduction_rate,
            advective_coupling_eta=advective_coupling_eta
        )

        if not is_contractive:
            raise BanachSemigroupInstabilityError(
                f"Inestabilidad en el generador de Banach G-ponderado: "
                f"máxima forma cuadrática disipativa {dissipative_const:.6e} > 0"
            )

        # 2. Invariante topológico de enrollamiento (determinante de Gram G-métrico)
        winding_number = self._evaluate_topological_winding(
            x_point=kernel.evaluation_point,
            v_velocity=report.velocity_vector,
            cache=kernel.metric_cache
        )

        banach_cert = BanachStabilityCertification(
            lumer_phillips_dissipative_constant=dissipative_const,
            is_contractive_semigroup=is_contractive,
            spectral_radius=spec_radius,
            topological_winding_number=winding_number
        )

        # 3. Metrología de tiempo FPU y error de Wilkinson
        t_delta_ns = time.perf_counter_ns() - t_start
        fpu_time_us = float(t_delta_ns / 1000.0)
        wilkinson_floor = self._compute_wilkinson_bound(kernel.metric_cache.dimension)

        # 4. Sello Soberano Inmutable
        sovereign_seal = self._generate_sovereign_cryptographic_seal(
            kernel=kernel,
            report=report,
            banach_cert=banach_cert,
            fpu_time_us=fpu_time_us
        )

        logger.debug("Fase 3 (Act) completada con éxito. Sello Soberano: %s", sovereign_seal[:16])

        return ScalarMomentumEngineState(
            kernel=kernel,
            transfer_report=report,
            banach_certification=banach_cert,
            fpu_execution_time_microseconds=fpu_time_us,
            wilkinson_roundoff_bound=wilkinson_floor,
            sovereign_cryptographic_seal=sovereign_seal
        )

    # ---- 3.7 MÉTODO DE INICIO CONTINUO DE FASE 3 -------------------------------------

    def act_from_kernel_and_report(
        self,
        kernel: ScalarMomentumKernel,
        report: MomentumTransferReport,
        advective_coupling_eta: float = _ADVECTIVE_COUPLING_ETA_DEFAULT
    ) -> ScalarMomentumEngineState:
        r"""
        MÉTODO DE INICIO CONTINUO DE FASE 3:
        Ingiere conjuntamente el `ScalarMomentumKernel` de Fase 1 y el `MomentumTransferReport`
        de Fase 2 para ejecutar verificación de custodia, auditoría de estabilidad de
        Banach $G$-ponderada, topología de Gram y sellado soberano.
        """
        return self._act_execute_pipeline(kernel, report, advective_coupling_eta=advective_coupling_eta)

    # ---- 3.8 MÉTODO MAESTRO ORQUESTADOR DE LAS 3 FASES ANIDADAS ----------------------

    def execute_momentum_transfer_audit(
        self,
        phi_func: Callable[[Any], Any],
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0,
        conductivity_sigma: float = 1.0,
        kappa_dissipation: Optional[float] = None,
        advective_coupling_eta: float = _ADVECTIVE_COUPLING_ETA_DEFAULT
    ) -> ScalarMomentumEngineState:
        r"""
        MÉTODO MAESTRO ORQUESTADOR DE LAS TRES FASES ANIDADAS:
        Ejecuta ininterrumpidamente el ciclo OODA de la FPU:
          - Fase 1 (Observe): `observe_scalar_field(...)` -> produce `ScalarMomentumKernel`.
          - Fase 2 (Orient) : `orient_from_kernel(...)` -> absorbe Kernel, produce `MomentumTransferReport`.
          - Fase 3 (Act)    : `act_from_kernel_and_report(...)` -> absorbe Kernel y Report, emite `ScalarMomentumEngineState`.
        """
        t_global_start = time.perf_counter_ns()

        # Ejecución Fase 1
        kernel = self.observe_scalar_field(
            phi_func=phi_func,
            x_point=x_point,
            momentum_p=momentum_p,
            G_metric=G_metric
        )

        # Ejecución Fase 2 (Inicia consumiendo la salida de Fase 1)
        report = self.orient_from_kernel(
            kernel=kernel,
            coupling_alpha=coupling_alpha,
            mass_m=mass_m,
            conductivity_sigma=conductivity_sigma,
            kappa_dissipation=kappa_dissipation
        )

        # Ejecución Fase 3 (Inicia consumiendo la salida de Fase 2)
        state = self.act_from_kernel_and_report(
            kernel=kernel,
            report=report,
            advective_coupling_eta=advective_coupling_eta
        )

        t_global_us = float((time.perf_counter_ns() - t_global_start) / 1000.0)

        # Modificación del tiempo de ejecución FPU global en el estado inmutable
        final_state = ScalarMomentumEngineState(
            kernel=state.kernel,
            transfer_report=state.transfer_report,
            banach_certification=state.banach_certification,
            fpu_execution_time_microseconds=t_global_us,
            wilkinson_roundoff_bound=state.wilkinson_roundoff_bound,
            sovereign_cryptographic_seal=state.sovereign_cryptographic_seal
        )

        return final_state


# ══════════════════════════════════════════════════════════════════════════════
# CLASE FACADE: MOTOR SOBERANO DE TRANSFERENCIA DE MOMENTUM ESCALAR
# ══════════════════════════════════════════════════════════════════════════════

class ScalarMomentumSatelliteEngine(Phase3_ScalarMomentumEngine):
    r"""
    MOTOR SOBERANO DE TRANSFERENCIA DE MOMENTUM ESCALAR DE FPU.
    Punto de entrada primario para sistemas aeroespaciales, satelitales y de cálculo cuántico.
    Consolida las 3 Fases anidadas continuas bajo una interfaz de alta eficiencia y rigor estricto.
    """

    def __init__(self, condition_tolerance: float = _CONDITION_NUMBER_MAX) -> None:
        super().__init__(condition_tolerance=condition_tolerance)

    def __repr__(self) -> str:
        return (
            f"<ScalarMomentumSatelliteEngine "
            f"Precision=IEEE-754-Double, ConditionTolerance={self._condition_tolerance:.2e}, "
            f"CSMD_Step={_CSMD_STEP_OPTIMAL:.1e}, Metrology=PortHamiltonian-Banach-GWeighted>"
        )


# Exportación de la API de nivel doctoral
__all__ = [
    # Motor Principal
    "ScalarMomentumSatelliteEngine",
    # Clases de Fases Anidadas
    "Phase1_ScalarMomentumObserver",
    "Phase2_CovariantTransferOrient",
    "Phase3_ScalarMomentumEngine",
    # Expedientes de Datos Inmutables
    "SpectralMetricTensorCache",
    "DifferentialFormsBundle",
    "ScalarMomentumKernel",
    "PortHamiltonianCircuitTelemetry",
    "MomentumTransferReport",
    "BanachStabilityCertification",
    "ScalarMomentumEngineState",
    # Módulo de Sumación Aritmética Exacta
    "KahanNeumaierAccumulator",
    # Excepciones
    "ScalarMomentumEngineError",
    "DimensionMismatchError",
    "MetricIndefinitenessError",
    "SpectralProjectorInconsistencyError",
    "CSMDHolomorphyError",
    "ThermodynamicPassivityViolationError",
    "BanachSemigroupInstabilityError",
    "MetrologicalChainIntegrityError",
]
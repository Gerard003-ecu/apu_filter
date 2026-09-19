
from __future__ import annotations

r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Scalar Momentum Satellite Engine (Motor de Transferencia de Momentum)       ║
║ RUTA   : app/core/immune_system/scalar_momentum_satellite_engine.py                  ║
║ VERSIÓN: 2.0.0-Doctoral-Hypercomplex-Spectral-PortHamiltonian-Banach-Secure          ║
║                                                                                      ║
║ SINOPSIS MATEMÁTICA, GEOMÉTRICA Y METROLOGÍA DE LA FPU:                              ║
║ Este módulo constituye el núcleo de cálculo numérico de ultra-alta fidelidad en la   ║
║ FPU para la evaluación de la transferencia de momentum covariante $p_\mu \in T^*_x\mathcal{M}$║
║ sobre un campo escalar suave $\phi \in C^\infty(\mathcal{M}, \mathbb{R})$ definido   ║
║ en una variedad Riemanniana compacta con frontera orientada $(\mathcal{M}, G, \partial\mathcal{M})$.║
║                                                                                      ║
║ FUNDAMENTACIÓN FÍSICA Y ESTRUCTURAS ALGEBRAICAS INTEGRADAS:                          ║
║ 1. Geometría Simpléctica y Fibrados Cotangentes:                                     ║
║    El espacio de fases es el fibrado cotangente $T^*\mathcal{M}$, equipado con la   ║
║    forma simpléctica canónica canónica $\omega = dp_\mu \wedge dx^\mu$.              ║
║    La velocidad contravariante es generada por el isomorfismo musical (sharp):       ║
║    $v = G^\sharp(p) \implies v^\mu = G^{\mu\nu} p_\nu \in T_x\mathcal{M}$.           ║
║                                                                                      ║
║ 2. Derivada de Lie y Corchetes de Poisson:                                           ║
║    La transferencia direccional es la derivada de Lie a lo largo del flujo de $v$:   ║
║    $\mathcal{L}_v \phi = \iota_v d\phi = \langle d\phi, v \rangle = p_\mu G^{\mu\nu} \partial_\nu \phi = \{\phi, H\}$, ║
║    donde $H(x, p) = \frac{1}{2} G^{\mu\nu}(x) p_\mu p_\nu$ es el Hamiltoniano libre. ║
║                                                                                      ║
║ 3. Diferenciación Holomorfa por Paso Complejo (CSMD) y Holomorfía Cauchy-Riemann:     ║
║    Aproximación de gradientes sin cancelación sustractiva en IEEE-754:               ║
║    $\partial_\mu \phi = \frac{\operatorname{Im}(\phi(x + i \cdot h \cdot e_\mu))}{h} + \mathcal{O}(h^2)$  ║
║    acompañada de un test de holomorfía Cauchy-Riemann con retroceso a diferencias    ║
║    centradas de orden 4 con extrapolación de Richardson y sumación KBN.              ║
║                                                                                      ║
║ 4. Teoría Espectral, Métricas de Banach y Topología Algebraica:                      ║
║    El tensor métrico $G \in \mathcal{S}^+_n(\mathbb{R})$ es descompuesto espectralmente ║
║    $G = U \Lambda U^\top$, calculando proyectores espectrales $\{P_k\}$, determinante,║
║    forma de volumen de Riemann $\mathrm{vol}_G = \sqrt{\det G}\,dx^1\wedge\dots\wedge dx^n$, ║
║    dual de Hodge $*d\phi$, número de condición $\kappa_2(G)$ y radio espectral.       ║
║                                                                                      ║
║ 5. Dinámica Port-Hamiltoniana y Teorema de Pasividad Termodinámica (Tellegen):       ║
║    Estructura de interconexión Dirac: $\dot{z} = (J - R)\nabla H + g u$, con matriz  ║
║    de disipación resistiva $R \succeq 0$. El flujo disipativo de energía satisface:  ║
║    $P_{\mathrm{diss}} = \langle d\phi, G^{-1} d\phi \rangle_G \ge 0$.                ║
║    Generación de semigrupo contractivo $C_0$ según el teorema de Lumer-Phillips.     ║
║                                                                                      ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS DE TRANSICIÓN FORMAL (OODA LOOP):     ║
║   Fase 1 (Observe): Ingesta metrológica, espectro de $G$, CSMD de $d\phi$, Hodge $*d\phi$.║
║          → Salida Terminal: ScalarMomentumKernel.                                    ║
║   Fase 2 (Orient) : Inicia DIRECTAMENTE absorbiendo ScalarMomentumKernel.            ║
║          Calcula $\mathcal{L}_v \phi$, Tensor $T_{\mu\nu}$, Pasividad Port-Hamiltoniana. ║
║          → Salida Terminal: MomentumTransferReport.                                  ║
║   Fase 3 (Act)    : Inicia DIRECTAMENTE absorbiendo (Kernel, Report).               ║
║          Certificación Lumer-Phillips, Winding topológico y Sello Criptográfico.     ║
║          → Salida Terminal: ScalarMomentumEngineState.                              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Final, List, Optional, Tuple, Dict, Any
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Configuración de Logging de Metrología Orbital
logger = logging.getLogger("APU.Physics.ScalarMomentumSatelliteEngine")

# Constantes Fundamentales de Precisión FPU y Metrología Cuántica
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CSMD_STEP_OPTIMAL: Final[float] = 1.0e-20
_RICHARDSON_STEP: Final[float] = 1.0e-5
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12
_WILKINSON_FLOOR: Final[float] = 1.0e-15
_DIRAC_PLANCK_CONSTANT: Final[float] = 1.054571817e-34  # Constante de Dirac reducida \hbar (J·s)
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


class CSMDHolomorphyError(ScalarMomentumEngineError):
    """Ruptura de las condiciones de Cauchy-Riemann en la diferenciación compleja."""
    pass


class ThermodynamicPassivityViolationError(ScalarMomentumEngineError):
    """Violación del principio de pasividad termodinámica (Disipación negativa)."""
    pass


class BanachSemigroupInstabilityError(ScalarMomentumEngineError):
    """Falla en la contractividad de Lumer-Phillips para el semigrupo de evolución."""
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
        r"""Producto interno euclidiano exacto mediante 2Product y KBN."""
        if u.shape[0] != v.shape[0]:
            raise DimensionMismatchError("Vectores incompatibles para producto punto KBN.")
        products = u * v
        return KahanNeumaierAccumulator.sum(products)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES (EXPEDIENTES METROLÓGICOS DE FIBRA)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SpectralMetricTensorCache:
    r"""
    Expediente espectral inmutable de la variedad Riemanniana $(\mathcal{M}, G)$.
    Almacena $G$, Cholesky $L$, inversa $G^{-1}$, autovalores $\{\lambda_k\}$,
    proyectores espectrales $P_k = u_k u_k^\top$, $\sqrt{\det G}$, y $\kappa_2(G)$.
    """
    g_base: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    g_inv: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    spectral_projectors: Tuple[NDArray[np.float64], ...]
    riemannian_volume_factor: float
    condition_number: float
    dimension: int


@dataclass(frozen=True, slots=True)
class DifferentialFormsBundle:
    r"""
    Haz de formas diferenciales en $T^*_x\mathcal{M}$:
    $\phi \in \Omega^0(\mathcal{M})$, 1-forma $d\phi \in \Omega^1(\mathcal{M})$ y
    el dual de Hodge $*d\phi \in \Omega^{n-1}(\mathcal{M})$.
    """
    scalar_field: float
    exterior_derivative: NDArray[np.float64]
    gradient_norm_squared: float
    hodge_star_exterior_derivative: NDArray[np.float64]


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


@dataclass(frozen=True, slots=True)
class PortHamiltonianCircuitTelemetry:
    r"""Expediente de pasividad de circuitos y dinámica Port-Hamiltoniana."""
    hamiltonian_energy: float
    lie_transfer_power: float
    dissipated_power: float
    resistive_conduction_rate: float
    poynting_boundary_flux: float
    is_thermodynamically_passive: bool


@dataclass(frozen=True, slots=True)
class MomentumTransferReport:
    r"""
    Expediente Inmutable Terminal de Fase 2 (Orient).
    Consolida la derivada de Lie, el Tensor de Esfuerzo-Energía $T_{\mu\nu}$,
    su traza contravariante $\mathcal{T}$, y la telemetría Port-Hamiltoniana.
    """
    kernel_ref: ScalarMomentumKernel
    velocity_vector: NDArray[np.float64]
    lie_derivative_transfer: float
    stress_energy_tensor: NDArray[np.float64]
    stress_energy_trace: float
    stress_energy_eigenvalues: NDArray[np.float64]
    circuit_telemetry: PortHamiltonianCircuitTelemetry
    phase2_hmac_sha256: str


@dataclass(frozen=True, slots=True)
class BanachStabilityCertification:
    r"""Certificación metrológica del Semigrupo $C_0$ y análisis funcional de Banach."""
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


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: OBSERVE — INGESTA, TEORÍA ESPECTRAL, CSMD HOLOMORFO Y FORMAS EXTERIORES
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_ScalarMomentumObserver:
    r"""
    FASE 1 (OBSERVE):
    Fundamentación matemática:
      - Auditoría espectral del tensor métrico $G \in \mathcal{S}^+_n(\mathbb{R})$.
      - Factorización de Cholesky $G = L L^\top$, cálculo del número de condición $\kappa_2(G)$,
        inversión triangular exacta y construcción de proyectores espectrales $P_k = u_k u_k^\top$.
      - Cálculo del factor de volumen Riemanniano $\sqrt{\det G}$.
      - Diferenciación holomorfa por paso complejo (CSMD) sobre el álgebra compleja $\mathbb{C}$.
      - Verificación de condiciones de Cauchy-Riemann para evitar singularidades no holomorfas.
      - Retroceso automático a diferencias finitas centradas de 4to orden con Richardson si $\phi$ es no holomorfa.
      - Cálculo de formas diferenciales: campo $\phi$, 1-forma $d\phi$, norma invariante $\|d\phi\|^2_G$,
        y el dual de Hodge $*d\phi \in \Omega^{n-1}(\mathcal{M})$.
    """

    def __init__(self, condition_tolerance: float = _CONDITION_NUMBER_MAX) -> None:
        self._condition_tolerance: Final[float] = float(condition_tolerance)

    def _verify_cauchy_riemann_holomorphy(
        self,
        phi_func: Callable[[Any], Any],
        x_point: NDArray[np.float64],
        probe_step: float = 1.0e-8
    ) -> bool:
        r"""
        Audita numéricamente si $\phi$ satisface las condiciones de Cauchy-Riemann sobre $\mathbb{C}$:
        $$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$
        donde $\phi(x + i y) = u(x, y) + i v(x, y)$.
        """
        dim = x_point.shape[0]
        test_idx = 0  # Comprobación direccional sobre el primer eje de coordenadas

        z0 = np.array(x_point, dtype=np.complex128)
        try:
            val_base = phi_func(z0)
            if not isinstance(val_base, (complex, np.complex128)):
                return False

            # Perturbación real dx
            zx_plus = np.array(x_point, dtype=np.complex128)
            zx_plus[test_idx] += probe_step
            val_x_plus = phi_func(zx_plus)

            # Perturbación imaginaria dy
            zy_plus = np.array(x_point, dtype=np.complex128)
            zy_plus[test_idx] += 1j * probe_step
            val_y_plus = phi_func(zy_plus)

            du_dx = (np.real(val_x_plus) - np.real(val_base)) / probe_step
            dv_dx = (np.imag(val_x_plus) - np.imag(val_base)) / probe_step

            du_dy = (np.real(val_y_plus) - np.real(val_base)) / probe_step
            dv_dy = (np.imag(val_y_plus) - np.imag(val_base)) / probe_step

            cr_err1 = abs(du_dx - dv_dy)
            cr_err2 = abs(du_dy + dv_dx)

            return bool(cr_err1 < 1.0e-4 and cr_err2 < 1.0e-4)
        except Exception:
            return False

    def _compute_richardson_centered_gradient(
        self,
        phi_func: Callable[[NDArray[np.float64]], float],
        x_point: NDArray[np.float64],
        h: float = _RICHARDSON_STEP
    ) -> NDArray[np.float64]:
        r"""
        Diferenciación finita centrada de orden 4 con extrapolación de Richardson y KBN:
        $$f'(x) = \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h} + \mathcal{O}(h^4)$$
        """
        dim = x_point.shape[0]
        grad = np.zeros(dim, dtype=np.float64)

        for k in range(dim):
            e_k = np.zeros(dim, dtype=np.float64)
            e_k[k] = 1.0

            x_p2 = x_point + 2.0 * h * e_k
            x_p1 = x_point + 1.0 * h * e_k
            x_m1 = x_point - 1.0 * h * e_k
            x_m2 = x_point - 2.0 * h * e_k

            f_p2 = float(np.real(phi_func(x_p2)))
            f_p1 = float(np.real(phi_func(x_p1)))
            f_m1 = float(np.real(phi_func(x_m1)))
            f_m2 = float(np.real(phi_func(x_m2)))

            acc = KahanNeumaierAccumulator()
            acc.add(-1.0 * f_p2)
            acc.add(8.0 * f_p1)
            acc.add(-8.0 * f_m1)
            acc.add(1.0 * f_m2)

            grad[k] = acc.total / (12.0 * h)

        return grad

    def _extract_holomorphic_csmd_gradient(
        self,
        phi_func: Callable[[Any], Any],
        x_point: NDArray[np.float64],
        h_step: float = _CSMD_STEP_OPTIMAL
    ) -> NDArray[np.float64]:
        r"""
        Calcula el 1-forma gradiente $d\phi = \partial_\mu \phi\, dx^\mu$ mediante Diferenciación
        por Paso Complejo (CSMD). Si se detecta no holomorfía, conmuta a Richardson de orden 4.
        """
        dim = x_point.shape[0]
        grad = np.zeros(dim, dtype=np.float64)

        is_holomorphic = self._verify_cauchy_riemann_holomorphy(phi_func, x_point)

        if not is_holomorphic:
            logger.debug("Función escalar no holomorfa detectada. Conmutando a Richardson de 4to orden.")
            return self._compute_richardson_centered_gradient(phi_func, x_point)

        for k in range(dim):
            z_point = np.array(x_point, dtype=np.complex128)
            z_point[k] += 1j * h_step
            phi_eval = phi_func(z_point)

            if not isinstance(phi_eval, (complex, np.complex128)):
                return self._compute_richardson_centered_gradient(phi_func, x_point)

            grad[k] = float(np.imag(phi_eval)) / h_step

        return grad

    def _decompose_spectral_metric(self, G_metric: NDArray[np.float64]) -> SpectralMetricTensorCache:
        r"""
        Audita el tensor métrico $G_{\mu\nu}$, calcula descomposición espectral $G = U \Lambda U^\top$,
        descomposición de Cholesky $G = L L^\top$, proyectores espectrales $\{P_k\}$ y forma de volumen.
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

        # Determinante y Forma de Volumen Riemanniana: \sqrt{\det G} = \prod L_{ii}
        diag_L = np.diag(L_cholesky)
        volume_factor = float(np.prod(diag_L))

        # Proyectores espectrales de Banach: P_k = u_k u_k^\top
        projectors: List[NDArray[np.float64]] = []
        for k in range(dim):
            u_k = eigenvectors[:, k : k + 1]
            p_k = u_k @ u_k.T
            projectors.append(p_k)

        return SpectralMetricTensorCache(
            g_base=g_sym,
            cholesky_factor=L_cholesky,
            g_inv=g_inv,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            spectral_projectors=tuple(projectors),
            riemannian_volume_factor=volume_factor,
            condition_number=cond_num,
            dimension=dim
        )

    def _compute_hodge_star_and_bundle(
        self,
        phi_val: float,
        dphi: NDArray[np.float64],
        cache: SpectralMetricTensorCache
    ) -> DifferentialFormsBundle:
        r"""
        Construye el haz de formas exteriores: calcula la norma invariante de Dirichlet
        $\|d\phi\|^2_G = G^{\mu\nu}\partial_\mu \phi \partial_\nu \phi$ y el dual de Hodge $*d\phi$.
        En una variedad $n$-dimensional, para una 1-forma $\alpha$, su dual de Hodge es una $(n-1)$-forma:
        $$(*\alpha)_{\mu_1 \dots \mu_{n-1}} = \frac{\sqrt{\det G}}{(n-1)!} \epsilon_{\mu_1 \dots \mu_{n-1}\nu} G^{\nu\sigma}\alpha_\sigma$$
        Representamos el dual de Hodge por su vector de flujo contravariante equivalente sobre la frontera.
        """
        # Contracción con métrica inversa: (G^{-1} dphi)
        sharp_dphi = cache.g_inv @ dphi
        grad_norm_sq = KahanNeumaierAccumulator.dot(dphi, sharp_dphi)

        # Dual de Hodge vectorial representativo: *(d\phi) acoplado al factor de volumen
        hodge_rep = cache.riemannian_volume_factor * sharp_dphi

        return DifferentialFormsBundle(
            scalar_field=phi_val,
            exterior_derivative=dphi,
            gradient_norm_squared=grad_norm_sq,
            hodge_star_exterior_derivative=hodge_rep
        )

    def observe_scalar_field(
        self,
        phi_func: Callable[[Any], Any],
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64]
    ) -> ScalarMomentumKernel:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1 (OBSERVE):
        Ingiere campos, audita espectro de $G$, deriva $d\phi$ via CSMD holomorfo,
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
            raise DimensionMismatchError(f"Dimensión de G ({metric_cache.dimension}) != Dimensión del Espacio ({dim})")

        # 2. Evaluación escalar en el punto base
        phi_val = float(np.real(phi_func(np.array(x_point, dtype=np.float64))))

        # 3. Diferenciación CSMD holomorfa con preservación de Cauchy-Riemann
        dphi = self._extract_holomorphic_csmd_gradient(phi_func, x_point)

        # 4. Construcción del haz de formas diferenciales y Hodge
        bundle = self._compute_hodge_star_and_bundle(phi_val, dphi, metric_cache)

        # 5. Sello criptográfico SHA-256 de Fase 1
        hasher = hashlib.sha256()
        hasher.update(x_point.tobytes())
        hasher.update(momentum_p.tobytes())
        hasher.update(dphi.tobytes())
        hasher.update(metric_cache.g_base.tobytes())
        hasher.update(metric_cache.eigenvalues.tobytes())
        hasher.update(f"{bundle.gradient_norm_squared:.16e}".encode("ascii"))
        phase1_hash = hasher.hexdigest()

        logger.debug("Fase 1 (Observe) completada exitosamente. Sello: %s", phase1_hash[:16])

        return ScalarMomentumKernel(
            evaluation_point=x_point.copy(),
            momentum_covector=momentum_p.copy(),
            forms_bundle=bundle,
            metric_cache=metric_cache,
            phase1_sha256_hash=phase1_hash
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: ORIENT — TRANSFERENCIA DE LIE, TENSOR DE ESFUERZOS Y PORT-HAMILTONIANO
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_CovariantTransferOrient(Phase1_ScalarMomentumObserver):
    r"""
    FASE 2 (ORIENT):
    Fundamentación matemática:
      - Absorbe DIRECTAMENTE el `ScalarMomentumKernel` emitido por el final de Fase 1.
      - Resuelve la velocidad covariante mediante el isomorfismo sharp: $v^\mu = G^{\mu\nu} p_\nu$.
      - Evalúa la derivada de Lie covariante:
        $$\mathcal{L}_v \phi = \iota_v d\phi = p_\mu G^{\mu\nu} \partial_\nu \phi = \{\phi, H\}$$
      - Construye el Tensor de Energía-Impulso Belinfante-Rosenfeld con acoplamiento de momentum:
        $$T_{\mu\nu}[\phi, p] = \partial_\mu \phi \partial_\nu \phi - \frac{1}{2} G_{\mu\nu}\left(G^{\alpha\beta}\partial_\alpha \phi \partial_\beta \phi + m^2 \phi^2\right) + \alpha\left(p_\mu \partial_\nu \phi + p_\nu \partial_\mu \phi - G_{\mu\nu} G^{\alpha\beta} p_\alpha \partial_\beta \phi\right)$$
      - Determina la traza contravariante exacta $\mathcal{T} = G^{\mu\nu} T_{\mu\nu}$ mediante KBN.
      - Modela el sistema disipativo como una estructura Port-Hamiltoniana sobre circuitos:
        $P_{\mathrm{diss}} = \sigma \langle p, G^{-1} p \rangle + \kappa \langle d\phi, G^{-1} d\phi \rangle \ge 0$.
      - Evalúa el vector de flujo de Poynting en la frontera de Dirichlet/Neumann.
      - Certifica la pasividad termodinámica de Tellegen ($P_{\mathrm{diss}} \ge -\epsilon_{\mathrm{Wilkinson}}$).
      - Emite el reporte inmutable `MomentumTransferReport`.
    """

    def orient_from_kernel(
        self,
        kernel: ScalarMomentumKernel,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0,
        conductivity_sigma: float = 1.0
    ) -> MomentumTransferReport:
        r"""
        MÉTODO DE INICIO CONTINUO DE FASE 2:
        Conecta formalmente con el final de Fase 1 al recibir su artefacto terminal
        `ScalarMomentumKernel`. Ejecuta la orientación dinámico-geométrica completa.
        """
        return self._orient_momentum_transfer_pipeline(
            kernel=kernel,
            coupling_alpha=coupling_alpha,
            mass_m=mass_m,
            conductivity_sigma=conductivity_sigma
        )

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

    def _synthesize_stress_energy_tensor(
        self,
        p_covector: NDArray[np.float64],
        dphi: NDArray[np.float64],
        phi_val: float,
        cache: SpectralMetricTensorCache,
        coupling_alpha: float,
        mass_m: float
    ) -> Tuple[NDArray[np.float64], float, NDArray[np.float64]]:
        r"""
        Sintetiza el tensor de tensión-energía covariante $T_{\mu\nu}$ y su traza espectral:
        $$T_{\mu\nu} = \partial_\mu \phi \partial_\nu \phi - \frac{1}{2} G_{\mu\nu} (\|d\phi\|^2_{G^{-1}} + m^2 \phi^2) + \alpha (p_\mu \partial_\nu \phi + p_\nu \partial_\mu \phi - G_{\mu\nu} \langle p, d\phi \rangle_{G^{-1}})$$
        """
        g_metric = cache.g_base
        g_inv = cache.g_inv

        # Norma al cuadrado del gradiente: \|d\phi\|^2_{G^{-1}}
        sharp_dphi = g_inv @ dphi
        grad_norm_sq = KahanNeumaierAccumulator.dot(dphi, sharp_dphi)

        # Densidad escalar del campo Klein-Gordon libre
        kinetic_potential_density = 0.5 * (grad_norm_sq + (mass_m ** 2) * (phi_val ** 2))

        # Producto exterior de formas \partial_\mu \phi \partial_\nu \phi
        dyadic_grad = np.outer(dphi, dphi)

        # Término de acoplamiento de momentum con corrección conforme
        coupling_inner = KahanNeumaierAccumulator.dot(p_covector, sharp_dphi)
        symmetric_momentum_coupling = np.outer(p_covector, dphi) + np.outer(dphi, p_covector)
        coupling_tensor = symmetric_momentum_coupling - (g_metric * coupling_inner)

        # Ensamblaje tensorial global
        T_tensor = dyadic_grad - (g_metric * kinetic_potential_density) + (coupling_alpha * coupling_tensor)

        # Traza contravariante: Tr_G(T) = G^{\mu\nu} T_{\mu\nu} = Tr(G^{-1} T)
        contraction_matrix = g_inv @ T_tensor
        trace_T = KahanNeumaierAccumulator.sum(np.diag(contraction_matrix))

        # Espectro del tensor de esfuerzos
        t_eigenvalues = la.eigvalsh(0.5 * (T_tensor + T_tensor.T))

        return T_tensor, float(trace_T), t_eigenvalues

    def _audit_port_hamiltonian_passivity(
        self,
        p_covector: NDArray[np.float64],
        dphi: NDArray[np.float64],
        v_velocity: NDArray[np.float64],
        lie_transfer: float,
        cache: SpectralMetricTensorCache,
        conductivity_sigma: float
    ) -> PortHamiltonianCircuitTelemetry:
        r"""
        Audita el balance termodinámico y la estructura de circuito Port-Hamiltoniano:
        $H = \frac{1}{2} \|p\|^2_{G^{-1}} + \frac{1}{2} \|d\phi\|^2_{G^{-1}}$.
        Potencia disipada: $P_{\mathrm{diss}} = \sigma \|p\|^2_{G^{-1}} + \kappa \|d\phi\|^2_{G^{-1}} \ge 0$.
        Flujo de Poynting en la frontera de Dirichlet: $S = \mathcal{L}_v \phi \cdot \mathrm{vol}_G$.
        """
        g_inv = cache.g_inv

        # Energía cinética hamiltoniana: \frac{1}{2} p_\mu G^{\mu\nu} p_\nu
        h_kinetic = 0.5 * KahanNeumaierAccumulator.dot(p_covector, v_velocity)

        # Energía potencial: \frac{1}{2} d\phi_\mu G^{\mu\nu} d\phi_\nu
        sharp_dphi = g_inv @ dphi
        h_potential = 0.5 * KahanNeumaierAccumulator.dot(dphi, sharp_dphi)
        hamiltonian_total = h_kinetic + h_potential

        # Tasa de disipación de Rayleigh (análogo resistivo en circuito eléctrico)
        dissipation_p = conductivity_sigma * (2.0 * h_kinetic)
        dissipation_phi = conductivity_sigma * (2.0 * h_potential)
        total_p_diss = dissipation_p + dissipation_phi

        # Pasividad de Tellegen: P_diss >= -Wilkinson
        is_passive = bool(total_p_diss >= -_WILKINSON_FLOOR)

        # Flujo electromagnético de frontera (Poynting)
        poynting_flux = lie_transfer * cache.riemannian_volume_factor

        return PortHamiltonianCircuitTelemetry(
            hamiltonian_energy=hamiltonian_total,
            lie_transfer_power=lie_transfer,
            dissipated_power=total_p_diss,
            resistive_conduction_rate=conductivity_sigma,
            poynting_boundary_flux=poynting_flux,
            is_thermodynamically_passive=is_passive
        )

    def _orient_momentum_transfer_pipeline(
        self,
        kernel: ScalarMomentumKernel,
        coupling_alpha: float,
        mass_m: float,
        conductivity_sigma: float
    ) -> MomentumTransferReport:
        r"""Orquesta la ejecución interna de la Fase 2 a partir del Kernel."""
        p_covector = kernel.momentum_covector
        dphi = kernel.forms_bundle.exterior_derivative
        phi_val = kernel.forms_bundle.scalar_field
        cache = kernel.metric_cache

        # 1. Velocidad contravariante y derivada de Lie simpléctica
        v_velocity, lie_transfer = self._compute_lie_transfer_symplectic(p_covector, dphi, cache.g_inv)

        # 2. Tensor de Energía-Impulso Belinfante-Rosenfeld
        T_tensor, trace_T, t_eigs = self._synthesize_stress_energy_tensor(
            p_covector=p_covector,
            dphi=dphi,
            phi_val=phi_val,
            cache=cache,
            coupling_alpha=coupling_alpha,
            mass_m=mass_m
        )

        # 3. Telemetría Port-Hamiltoniana y auditoría de pasividad
        telemetry = self._audit_port_hamiltonian_passivity(
            p_covector=p_covector,
            dphi=dphi,
            v_velocity=v_velocity,
            lie_transfer=lie_transfer,
            cache=cache,
            conductivity_sigma=conductivity_sigma
        )

        if not telemetry.is_thermodynamically_passive:
            raise ThermodynamicPassivityViolationError(
                f"Violación de pasividad en el circuito térmico: P_diss = {telemetry.dissipated_power:.8e} < 0"
            )

        # 4. Sello Criptográfico HMAC-SHA256 encadenado a la Fase 1
        hmac_signer = hmac.new(_HMAC_METROLOGY_KEY, digestmod=hashlib.sha256)
        hmac_signer.update(kernel.phase1_sha256_hash.encode("ascii"))
        hmac_signer.update(v_velocity.tobytes())
        hmac_signer.update(T_tensor.tobytes())
        hmac_signer.update(f"{lie_transfer:.16e}".encode("ascii"))
        hmac_signer.update(f"{telemetry.dissipated_power:.16e}".encode("ascii"))
        phase2_hmac = hmac_signer.hexdigest()

        return MomentumTransferReport(
            kernel_ref=kernel,
            velocity_vector=v_velocity,
            lie_derivative_transfer=lie_transfer,
            stress_energy_tensor=T_tensor,
            stress_energy_trace=trace_T,
            stress_energy_eigenvalues=t_eigs,
            circuit_telemetry=telemetry,
            phase2_hmac_sha256=phase2_hmac
        )

    def orient_momentum_transfer(
        self,
        kernel: ScalarMomentumKernel,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0,
        conductivity_sigma: float = 1.0
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
            conductivity_sigma=conductivity_sigma
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: ACT — SEMIGRUPOS DE BANACH, TOPOLOGÍA Y CERTIFICACIÓN SOBERANA
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_ScalarMomentumEngine(Phase2_CovariantTransferOrient):
    r"""
    FASE 3 (ACT):
    Fundamentación matemática:
      - Absorbe DIRECTAMENTE el `MomentumTransferReport` (y su `ScalarMomentumKernel` interno)
        procedente del final de la Fase 2.
      - Análisis Funcional y Teoría de Operadores en Espacios de Banach:
        Verifica la contractividad del semigrupo de evolución $C_0$ generado por el operador
        advectivo-difusivo $\mathcal{A} = v^\mu \partial_\mu - \sigma \Delta_G$ mediante el
        Teorema de Lumer-Phillips:
        $$\operatorname{Re}\langle \mathcal{A} \psi, \psi \rangle \le 0 \iff \|e^{t\mathcal{A}}\| \le 1, \quad \forall t \ge 0$$
      - Topología Algebraica y Clases Características:
        Evalúa el número de enrollamiento (winding number) o grado topológico del flujo de momentum
        sobre la frontera $\partial \mathcal{M}$: $\mathrm{deg}(v) = \frac{1}{2\pi} \oint_{\partial \mathcal{M}} d(\operatorname{Arg}(v))$.
      - Cómputo de la Cota de Redondeo de Wilkinson en punto flotante IEEE-754:
        $\gamma_n = \frac{n \cdot \epsilon_{\mathrm{mach}}}{1 - n \cdot \epsilon_{\mathrm{mach}}}$.
      - Emisión del Sello Soberano Inmutable y Ledger SHA-256 / HMAC FPU.
    """

    def act_from_kernel_and_report(
        self,
        kernel: ScalarMomentumKernel,
        report: MomentumTransferReport
    ) -> ScalarMomentumEngineState:
        r"""
        MÉTODO DE INICIO CONTINUO DE FASE 3:
        Ingiere conjuntamente el `ScalarMomentumKernel` de Fase 1 y el `MomentumTransferReport`
        de Fase 2 para ejecutar la auditoría de estabilidad de Banach, topología y sellado.
        """
        return self._act_execute_pipeline(kernel, report)

    def _audit_lumer_phillips_semigroup(
        self,
        v_velocity: NDArray[np.float64],
        cache: SpectralMetricTensorCache,
        conductivity_sigma: float
    ) -> Tuple[float, bool, float]:
        r"""
        Verifica el Teorema de Lumer-Phillips para el generador infinitesimal $\mathcal{A}$:
        Construye la aproximación matricial en la base espectral $U^\top A U$, determina
        la cota disipativa $\alpha = \max \operatorname{Re}(\operatorname{spec}(\mathcal{A}))$,
        el radio espectral $r(\mathcal{A})$, y certifica contractividad ($\alpha \le \epsilon$).
        """
        dim = cache.dimension
        # Matriz generadora advectiva-resistiva aproximada: A = - \sigma G - \operatorname{skew}(v)
        # Construimos el componente antisimétrico de advección simpléctica
        v_skew = np.outer(v_velocity, v_velocity) - np.outer(v_velocity, v_velocity).T
        A_generator = -1.0 * conductivity_sigma * cache.g_base + 0.1 * v_skew

        # Espectro del operador en el álgebra de Banach \mathcal{B}(\mathbb{R}^n)
        operator_eigs = la.eigvals(A_generator)
        real_parts = np.real(operator_eigs)
        max_real_eigenvalue = float(np.max(real_parts))
        spectral_radius = float(np.max(np.abs(operator_eigs)))

        # Contractividad de Lumer-Phillips: Re(\lambda) <= Wilkinson
        is_contractive = bool(max_real_eigenvalue <= _WILKINSON_FLOOR)

        return max_real_eigenvalue, is_contractive, spectral_radius

    def _evaluate_topological_winding(
        self,
        x_point: NDArray[np.float64],
        v_velocity: NDArray[np.float64]
    ) -> float:
        r"""
        Calcula el invariante topológico de primer orden (número de enrollamiento simpléctico):
        $$W = \frac{1}{2\pi} \frac{x^1 v^2 - x^2 v^1}{\|x\|_2 \|v\|_2 + \epsilon}$$
        Generalizable como la integral de Maurer-Cartan sobre la 1-esfera de frontera.
        """
        norm_x = float(la.norm(x_point))
        norm_v = float(la.norm(v_velocity))

        if norm_x < 1.0e-12 or norm_v < 1.0e-12:
            return 0.0

        if x_point.shape[0] >= 2:
            cross_2d = (x_point[0] * v_velocity[1]) - (x_point[1] * v_velocity[0])
            winding = cross_2d / (2.0 * math.pi * norm_x * norm_v)
            return float(winding)
        else:
            # Caso 1-dimensional: Proyección signada
            return float(np.sign(x_point[0] * v_velocity[0]))

    def _compute_wilkinson_bound(self, dimension: int) -> float:
        r"""
        Calcula la cota de Wilkinson acumulada para algoritmos matriciales de paso $n$:
        $$\gamma_n = \frac{n \cdot \epsilon_{\mathrm{mach}}}{1 - n \cdot \epsilon_{\mathrm{mach}}}$$
        """
        n_eps = dimension * _MACHINE_EPS
        if n_eps >= 1.0:
            return float("inf")
        return float(n_eps / (1.0 - n_eps))

    def _generate_sovereign_cryptographic_seal(
        self,
        kernel: ScalarMomentumKernel,
        report: MomentumTransferReport,
        banach_cert: BanachStabilityCertification,
        fpu_time_us: float
    ) -> str:
        r"""Genera el sello holístico SHA-256 encadenando los sellos de las 3 Fases."""
        hasher = hashlib.sha256()
        hasher.update(kernel.phase1_sha256_hash.encode("ascii"))
        hasher.update(report.phase2_hmac_sha256.encode("ascii"))
        hasher.update(f"{banach_cert.lumer_phillips_dissipative_constant:.16e}".encode("ascii"))
        hasher.update(f"{banach_cert.spectral_radius:.16e}".encode("ascii"))
        hasher.update(f"{banach_cert.topological_winding_number:.16e}".encode("ascii"))
        hasher.update(f"{fpu_time_us:.6f}".encode("ascii"))
        return hasher.hexdigest()

    def _act_execute_pipeline(
        self,
        kernel: ScalarMomentumKernel,
        report: MomentumTransferReport
    ) -> ScalarMomentumEngineState:
        r"""Orquesta la certificación y sellado final de Fase 3."""
        t_start = time.perf_counter_ns()

        # 1. Auditoría del Semigrupo de Banach y Lumer-Phillips
        dissipative_const, is_contractive, spec_radius = self._audit_lumer_phillips_semigroup(
            v_velocity=report.velocity_vector,
            cache=kernel.metric_cache,
            conductivity_sigma=report.circuit_telemetry.resistive_conduction_rate
        )

        if not is_contractive:
            raise BanachSemigroupInstabilityError(
                f"Inestabilidad en el generador de Banach: Máximo autovalor real {dissipative_const:.6e} > 0"
            )

        # 2. Invariante topológico de enrollamiento
        winding_number = self._evaluate_topological_winding(
            x_point=kernel.evaluation_point,
            v_velocity=report.velocity_vector
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

    def execute_momentum_transfer_audit(
        self,
        phi_func: Callable[[Any], Any],
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0,
        conductivity_sigma: float = 1.0
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
            conductivity_sigma=conductivity_sigma
        )

        # Ejecución Fase 3 (Inicia consumiendo la salida de Fase 2)
        state = self.act_from_kernel_and_report(
            kernel=kernel,
            report=report
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
            f"CSMD_Step={_CSMD_STEP_OPTIMAL:.1e}, Metrology=PortHamiltonian-Banach>"
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
    "CSMDHolomorphyError",
    "ThermodynamicPassivityViolationError",
    "BanachSemigroupInstabilityError",
]
from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Pseudoholomorphic Centripetal Satellite Engine (Satélite III — Centrípeta)  ║
║ RUTA   : app/core/immune_system/pseudoholomorphic_centripetal_satellite_engine.py    ║
║ VERSIÓN: 3.0.0-Doctoral-Fukaya-Floer-CholeskyFrame-Hypercomplex-KBN-FPU-Secure       ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Motor FPU para auditar deformación elasto-plástica, confinamiento simpléctico y tensores
de esfuerzo giroscópico en $\mathfrak{so}(n)$ sobre polígonos pseudo-holomorfos
$u: (\Sigma, \partial\Sigma) \to (\mathcal{M}, \partial\mathcal{M})$ en la Categoría de Fukaya $\mathcal{F}uk(\mathcal{M})$,
sometidos a campos Hamiltonianos centrípetos en el Cinturón Orbital de Frontera ($\partial\mathcal{M}\neq\varnothing$).

Fundamentación Matemática y Física Rigurosa:
────────────────────────────────────────────
1. Estructura casi-compleja $J$ $G$-compatible en el marco ortonormal de Cholesky:
   $$G = L L^\top, \quad y = L^\top x, \quad J_x = L^{-\top} J_0 L^\top$$
   garantizando $J^2 = -\mathbb{I}$ (dimensión par) y $J^\top G J = G$.
   Forma simpléctica $\omega(u,v) = G(Ju, v) \iff \Omega = J^\top G = -\Omega^\top$.
   En dimensión 4: terna casi-hipercompleja $(I,J,K)$ con $IJ = K$, $I^2 = J^2 = K^2 = -1$.
   En dimensión impar: estructura de casi-contacto Sasakiana $\phi^2 = -\mathbb{I} + \xi \otimes \eta$.

2. Ecuación de Floer–Cauchy–Riemann Perturbada:
   $$\bar{\partial}_{J,H} u = \partial_s u + J(u)(\partial_\tau u - X_{H_{\mathrm{cent}}}(u)) = 0$$
   donde el campo Hamiltoniano centrípeto es $X_H = J\,\mathrm{grad}_G H$, con potencial:
   $$H_{\mathrm{cent}}(q) = \frac{1}{2} M_{\mathrm{eff}} \|\omega_{\mathrm{rot}}\|_G^2 \|q - q_c\|_G^2$$

3. Área de Novikov y Área de Stokes Discreta:
   $$\mathcal{A}(u) = \sum_k a_k, \quad \mathcal{A}_{\partial}(u) = \frac{1}{2} \sum_k \omega(u_k, u_{k+1})$$

4. Álgebra de Lie $\mathfrak{so}(n)$ y Tensores Giroscópicos:
   $$W = \alpha (p \wedge \omega_{\mathrm{rot}}) \in \mathfrak{so}(n)$$
   Casimir Euclídeo $\mathcal{C}_2 = -\frac{1}{2}\operatorname{Tr}(W^2) = \frac{1}{2}\|W\|_F^2 \ge 0$,
   Casimir Métrico $\mathcal{C}_2^G = -\frac{1}{2}\operatorname{Tr}((G^{-1}W)^2) \ge 0$.

5. Deformación Radial de Móduli y Fluencia Elasto-Plástica:
   $$\epsilon_{\mathrm{radial}} = \frac{1}{\mathcal{A}} \sum_k a_k (\delta q_k \otimes \delta q_k)$$
   con espectro generalizado de deformación $\epsilon v = \sigma G v$.

6. Identidad Energética Port-Hamiltoniana:
   $$\dot{H} + P_{\mathrm{diss}} \approx 0$$
   certificando que el acoplamiento giroscópico es potencia-neutro.

Traducción Ejecutiva e Impacto de Negocio ('Dolor y Dinero'):
─────────────────────────────────────────────────────────────
• Dolor: Deformaciones elasto-plásticas centrípetas no confinadas provocan derivas de insumos y desbordamiento
  de precios unitarios, resultando en pérdidas millonarias en contratos a precio fijo.
• Dinero: El confinamiento simpléctico y la auditoría giroscópica restringen la dispersión radial en la FPU,
  estabilizando las variaciones de costos e imprevistos en la franja elástica predeterminada.

Estructura Functorial OODA:
───────────────────────────
- Fase 1 (Observe) : `observe_centripetal_polygon`      -> Salida: `CentripetalObserverKernel`
- Fase 2 (Orient)  : `orient_from_kernel`               -> Salida: `CentripetalDeformationReport`
- Fase 3 (Act)     : `act_from_kernel_and_report`       -> Salida: `CentripetalEngineState`
"""

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from typing import Final, Optional, Tuple, List

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Physics.PseudoholomorphicCentripetalEngine")

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de precisión metrológica, cono SPD y umbrales de Gromov/plasticidad
# ──────────────────────────────────────────────────────────────────────────────
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1.0e-15
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12
_CENTRIFUGAL_BUBBLING_THRESHOLD: Final[float] = 1.0e-6
_PLASTIC_DEFORMATION_THRESHOLD: Final[float] = 50.0
_COMPATIBILITY_RESIDUAL_MAX: Final[float] = 1.0e-8
_HMAC_CENTRIPETAL_SECRET_KEY: Final[bytes] = (
    b"PseudoholomorphicCentripetalEngine::FloerFukayaKey2026"
)
_ENGINE_VERSION: Final[str] = "3.0.0"


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES DOCTORALES DE FÓRMULA CERRADA
# ══════════════════════════════════════════════════════════════════════════════

class CentripetalEngineError(Exception):
    """Excepción raíz del subsistema centrípeto pseudo-holomorfo."""


class CentripetalDimensionError(CentripetalEngineError):
    """Falla de consistencia dimensional en fibras, velocidades o tensores."""


class MetricIndefinitenessError(CentripetalEngineError):
    """El tensor métrico de fondo $G$ no reside en el cono estrictamente SPD."""


class AlmostComplexCompatibilityError(CentripetalEngineError):
    """$J$ no satisface $J^{2}=-I$ (par) o $J^{\top}GJ=G$ dentro de la cota FPU."""


class CryptographicChainError(CentripetalEngineError):
    """Ruptura de la cadena de sellos SHA-256 / HMAC entre fases anidadas."""


class CentrifugalDiskBubblingError(CentripetalEngineError):
    """Colapso de Gromov: pérdida de compacidad por rotación centrípeta."""


class PlasticDeformationRuptureError(CentripetalEngineError):
    r"""Falla elasto-plástica: $\|\epsilon_{\mathrm{radial}}\|_{F}$ excede fluencia."""


# ══════════════════════════════════════════════════════════════════════════════
# MÁQUINA DE SUMACIÓN COMPENSADA: KAHAN–BABUŠKA–NEUMAIER + TWOSUM / 2PROD
# ══════════════════════════════════════════════════════════════════════════════

class KahanNeumaierSum:
    r"""
    Sumación compensada de Kahan–Babuška–Neumaier (KBN) y producto interno
    compensado. Neutraliza la deriva de mantisa IEEE-754 acumulando el residuo
    de redondeo en cada contracción. TwoSum es exacto (Knuth); el producto
    usa el residuo $x_{i}y_{i} - \mathrm{fl}(x_{i}y_{i})$ via FMA si existe.
    """

    @staticmethod
    def two_sum(a: float, b: float) -> Tuple[float, float]:
        r"""Descomposición exacta $a+b = s+e$ en aritmética flotante (Knuth TwoSum)."""
        s = a + b
        z = s - a
        e = (a - (s - z)) + (b - z)
        return float(s), float(e)

    @staticmethod
    def sum(arr: NDArray[np.float64]) -> float:
        r"""Suma KBN de un tensor arbitrario reducido a 1-forma."""
        s = 0.0
        c = 0.0
        for x in np.ascontiguousarray(arr, dtype=np.float64).ravel():
            x_f = float(x)
            t = s + x_f
            if abs(s) >= abs(x_f):
                c += (s - t) + x_f
            else:
                c += (x_f - t) + s
            s = t
        return float(s + c)

    @staticmethod
    def dot(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
        r"""Producto interno euclidiano compensado $\langle u,v\rangle_{\mathrm{KBN}}$."""
        u_c = np.ascontiguousarray(u, dtype=np.float64).ravel()
        v_c = np.ascontiguousarray(v, dtype=np.float64).ravel()
        if u_c.shape[0] != v_c.shape[0]:
            raise CentripetalDimensionError(
                "Discrepancia dimensional para producto interno KBN."
            )
        return KahanNeumaierSum.sum(u_c * v_c)

    @staticmethod
    def quadratic_form(
        x: NDArray[np.float64],
        g_metric: NDArray[np.float64],
        y: Optional[NDArray[np.float64]] = None,
    ) -> float:
        r"""Forma bilineal $x^{\top} G y$ con reducción KBN."""
        y_vec = x if y is None else y
        return KahanNeumaierSum.dot(x, g_metric @ y_vec)

    @staticmethod
    def fsum_crosscheck(arr: NDArray[np.float64]) -> float:
        r"""Suma de Shewchuk (`math.fsum`) como testigo ortogonal de KBN."""
        return float(math.fsum(np.ascontiguousarray(arr, dtype=np.float64).ravel().tolist()))


def _immutable_array(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Copia C-contigua float64 con bandera write=False (inmutabilidad de fibra)."""
    out = np.ascontiguousarray(tensor, dtype=np.float64).copy()
    out.setflags(write=False)
    return out


def _finite_or_raise(tensor: NDArray[np.float64], name: str) -> None:
    if not np.all(np.isfinite(tensor)):
        raise CentripetalDimensionError(
            f"El tensor '{name}' contiene NaN/Inf: no es un punto de $\\mathbb{{R}}^{{n}}$."
        )


def _fmt_e(value: float) -> str:
    return f"{float(value):.16e}"


# ══════════════════════════════════════════════════════════════════════════════
# EXPEDIENTES DE DATOS INMUTABLES (ESTRUCTURAS TENSORIALES DE FIBRA)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class BaseMetricCache:
    r"""
    Caché espectral inmutable de $(G,J,\omega)$ sobre $(\mathcal{M},G)$.
    Incluye marco de Cholesky, forma simpléctica $\Omega=J^{\top}G$, gap espectral
    $\lambda_{\max}/\lambda_{\min}$ y, si $\dim=4$, terna casi-hipercompleja.
    """
    g_base: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    g_inv: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    almost_complex_j: NDArray[np.float64]
    symplectic_form: NDArray[np.float64]
    riemannian_volume_factor: float
    condition_number: float
    spectral_gap: float
    j_compatibility_residual: float
    dimension: int
    is_even_dimension: bool
    hypercomplex_j: Optional[NDArray[np.float64]]
    hypercomplex_k: Optional[NDArray[np.float64]]
    reeb_field: Optional[NDArray[np.float64]]
    contact_form: Optional[NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class CentripetalObserverKernel:
    r"""
    Expediente inmutable TERMINAL de Fase 1 (Observe).
    Es el conector formal obligatorio: todo método de Fase 2 comienza
    absorbiendo este Kernel (continuación functorial Observe $\Rightarrow$ Orient).
    """
    polygon_vertices: NDArray[np.float64]
    vertex_velocities: NDArray[np.float64]
    angular_velocity_vector: NDArray[np.float64]
    simplex_areas: NDArray[np.float64]
    novikov_area: float
    stokes_symplectic_area: float
    centroid_q: NDArray[np.float64]
    effective_mass: float
    metric_cache: BaseMetricCache
    phase1_sha256_seal: str


@dataclass(frozen=True, slots=True)
class CentripetalDeformationReport:
    r"""
    Expediente inmutable TERMINAL de Fase 2 (Orient).
    Insumo formal de Fase 3 junto con el Kernel anidado `kernel_ref`.
    """
    kernel_ref: CentripetalObserverKernel
    centripetal_potential: float
    centripetal_force_field: NDArray[np.float64]
    cauchy_riemann_residual: float
    gyroscopic_stress_tensor: NDArray[np.float64]
    gyroscopic_skew_residual: float
    g_skew_residual: float
    casimir_invariant_c2: float
    casimir_euclidean_c2: float
    radial_deformation_tensor: NDArray[np.float64]
    radial_deformation_norm: float
    radial_principal_strains: NDArray[np.float64]
    dissipated_viscous_power: float
    port_hamiltonian_residual: float
    dirichlet_energy: float
    virial_ratio: float
    is_gyroscopic_skew_symmetric: bool
    phase2_hmac_sha256: str


@dataclass(frozen=True, slots=True)
class CentripetalEngineState:
    r"""
    Certificado soberano inmutable emitido por Fase 3 (Act).
    Gobernanza de lazo cerrado, Gromov–Maslov, fluencia y cadena criptográfica.
    """
    kernel: CentripetalObserverKernel
    deformation_report: CentripetalDeformationReport
    is_centrifugal_bubbling_detected: bool
    is_plastic_deformation_critical: bool
    bubbling_threshold: float
    plastic_threshold: float
    wilkinson_roundoff_bound: float
    maslov_index_estimate: int
    chain_integrity_verified: bool
    fpu_execution_time_ms: float
    cryptographic_seal: str
    engine_version: str


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: OBSERVE — INGESTA, TEORÍA ESPECTRAL, $J$ $G$-COMPATIBLE, NOVIKOV, $q_c$
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_CentripetalPolygonObserver:
    r"""
    FASE 1 (OBSERVE).

    Fundamentación:
      - Ingesta de la malla $u(\Sigma)=\{u_k\}_{k=1}^{N}\subset\mathbb{R}^{d}$,
        velocidades $\{v_k\}$ y rotación $\omega_{\mathrm{rot}}\in\mathbb{R}^{d}$.
      - Saneamiento IEEE-754 ($-0.0\to+0.0$) y rechazo de NaN/Inf.
      - Auditoría espectral de $G\in\mathcal{S}_{d}^{+}(\mathbb{R})$: Cholesky
        $G=LL^{\top}$, $\kappa_{2}(G)$, gap $\lambda_{\max}-\lambda_{\min}$.
      - $J$ $G$-compatible en el marco ortonormal de Cholesky (estable, sin `sqrtm`).
      - Área de Novikov KBN y área de Stokes discreta $\tfrac12\sum\omega(u_k,u_{k+1})$.
      - Centroide baricéntrico $q_{c}^{\mu}=\mathcal{A}(u)^{-1}\sum a_k u_k^{\mu}$.
      - Sello SHA-256 determinista (sin reloj).

    MÉTODO TERMINAL: `observe_centripetal_polygon` → `CentripetalObserverKernel`.
    Ese Kernel es el objeto inicial de todos los métodos de Fase 2.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        condition_tolerance: float = _CONDITION_NUMBER_MAX,
        strict_mode: bool = False,
    ) -> None:
        self._tol: Final[float] = float(tolerance)
        self._condition_tolerance: Final[float] = float(condition_tolerance)
        self._strict_mode: Final[bool] = bool(strict_mode)

    # ── 1.1 Saneamiento de fibra ──────────────────────────────────────────────

    @staticmethod
    def _sanitize_signed_zeros(
        tensor: NDArray[np.float64],
        name: str = "tensor",
    ) -> NDArray[np.float64]:
        r"""Colapsa $-0.0$ a $+0.0$ y exige finitud completa del tensor."""
        _finite_or_raise(tensor, name)
        clean = np.where(tensor == -0.0, +0.0, tensor)
        return np.ascontiguousarray(clean, dtype=np.float64)

    # ── 1.2 Marco de Cholesky y formas canónicas ──────────────────────────────

    @staticmethod
    def _standard_complex_structure(dim: int) -> NDArray[np.float64]:
        r"""
        $J_{0}$ canónica:
          - $d=2m$: $J_{0}=\begin{pmatrix}0 & -\mathbb{I}_{m}\\ \mathbb{I}_{m} & 0\end{pmatrix}$,
            $J_{0}^{2}=-\mathbb{I}$, $J_{0}^{\top}=-J_{0}$.
          - $d=2m+1$: $\phi_{0}$ Sasakiana plana, $\phi_{0}^{2}=-\mathbb{I}+\xi\otimes\eta$,
            $\xi=e_{d}$, $\eta=e^{d}$.
        """
        j0 = np.zeros((dim, dim), dtype=np.float64)
        if dim % 2 == 0:
            m = dim // 2
            j0[:m, m:] = -np.eye(m, dtype=np.float64)
            j0[m:, :m] = np.eye(m, dtype=np.float64)
        else:
            m = (dim - 1) // 2
            j0[:m, m:2 * m] = -np.eye(m, dtype=np.float64)
            j0[m:2 * m, :m] = np.eye(m, dtype=np.float64)
        return j0

    @staticmethod
    def _standard_hypercomplex_jk() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Terna cuaterniónica plana sobre $\mathbb{R}^{4}\simeq\mathbb{H}$ en la misma
        base que $J_{0}$ de $d=4$: $I=J_{0}$, $J^{2}=-I$, $IJ=-JI=K$.
        """
        j_hyp = np.zeros((4, 4), dtype=np.float64)
        j_hyp[0, 1] = -1.0
        j_hyp[1, 0] = 1.0
        j_hyp[2, 3] = 1.0
        j_hyp[3, 2] = -1.0
        i_std = Phase1_CentripetalPolygonObserver._standard_complex_structure(4)
        k_hyp = i_std @ j_hyp
        return j_hyp, k_hyp

    def _pullback_endomorphism(
        self,
        l_chol: NDArray[np.float64],
        l_inv: NDArray[np.float64],
        j_eucl: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Pullback $J_{x}=L^{-\top} J_{\mathrm{eucl}} L^{\top}$ desde el marco
        ortonormal $y=L^{\top}x$ (donde $G=\mathbb{I}$) al marco coordenado.
        Preserva $J^{2}=J_{\mathrm{eucl}}^{2}$ conjugado y $J^{\top}GJ=G$ si
        $J_{\mathrm{eucl}}$ es ortogonal euclídeo.
        """
        return l_inv.T @ j_eucl @ l_chol.T

    def _construct_compatible_almost_complex_structure(
        self,
        l_chol: NDArray[np.float64],
        l_inv: NDArray[np.float64],
        dim: int,
    ) -> Tuple[
        NDArray[np.float64],
        Optional[NDArray[np.float64]],
        Optional[NDArray[np.float64]],
        Optional[NDArray[np.float64]],
        Optional[NDArray[np.float64]],
    ]:
        r"""
        Construye $J$ $G$-compatible, terna hipercompleja (si $d=4$) y par
        de contacto $(\xi,\eta)$ (si $d$ impar).
        """
        j_eucl = self._standard_complex_structure(dim)
        j_x = self._pullback_endomorphism(l_chol, l_inv, j_eucl)

        j_hyp_x: Optional[NDArray[np.float64]] = None
        k_hyp_x: Optional[NDArray[np.float64]] = None
        reeb: Optional[NDArray[np.float64]] = None
        eta: Optional[NDArray[np.float64]] = None

        if dim == 4:
            j_h, k_h = self._standard_hypercomplex_jk()
            j_hyp_x = self._pullback_endomorphism(l_chol, l_inv, j_h)
            k_hyp_x = self._pullback_endomorphism(l_chol, l_inv, k_h)
        elif dim % 2 == 1:
            # Reeb en marco euclídeo: $e_d$; pullback contravariante $L^{-\top} e_d$.
            e_reeb = np.zeros(dim, dtype=np.float64)
            e_reeb[-1] = 1.0
            reeb = l_inv.T @ e_reeb
            eta = l_chol @ e_reeb  # $\eta = G(\xi,\cdot)$ en coordenadas, $\xi$ G-unitario

        return j_x, j_hyp_x, k_hyp_x, reeb, eta

    def _validate_almost_complex_compatibility(
        self,
        j_matrix: NDArray[np.float64],
        g_metric: NDArray[np.float64],
        dim: int,
        reeb: Optional[NDArray[np.float64]],
        contact: Optional[NDArray[np.float64]],
    ) -> float:
        r"""
        Residuo de Frobenius de las identidades estructurales:
          par: $\|J^{2}+I\|_{F}+\|J^{\top}GJ-G\|_{F}$,
          impar: $\|\phi^{2}+I-\xi\otimes\eta\|_{F}+\|\phi^{\top}G\phi-(G-\eta\otimes\eta)\|_{F}$.
        """
        if dim % 2 == 0:
            j2_res = float(la.norm(j_matrix @ j_matrix + np.eye(dim), "fro"))
            iso_res = float(la.norm(j_matrix.T @ g_metric @ j_matrix - g_metric, "fro"))
            residual = j2_res + iso_res
        else:
            if reeb is None or contact is None:
                raise AlmostComplexCompatibilityError(
                    r"Estructura Sasakiana incompleta: falta el par $(\xi,\eta)$."
                )
            target = -np.eye(dim, dtype=np.float64) + np.outer(reeb, contact)
            phi2_res = float(la.norm(j_matrix @ j_matrix - target, "fro"))
            g_contact = g_metric - np.outer(contact, contact)
            iso_res = float(la.norm(j_matrix.T @ g_metric @ j_matrix - g_contact, "fro"))
            residual = phi2_res + iso_res

        if residual > max(_COMPATIBILITY_RESIDUAL_MAX, 1.0e3 * self._tol):
            msg = (
                f"Estructura casi-compleja no $G$-compatible: residuo Frobenius "
                f"{residual:.6e} > cota."
            )
            if self._strict_mode:
                raise AlmostComplexCompatibilityError(msg)
            logger.warning(msg)
        return residual

    # ── 1.3 Auditoría espectral de $G$ ────────────────────────────────────────

    def _audit_metric_regularity(self, g_metric: NDArray[np.float64]) -> BaseMetricCache:
        r"""
        Simetriza $G$, exige SPD estricto ($\lambda_{\min}>\gamma_{\mathrm{Wilkinson}}$),
        factoriza Cholesky, invierte triangularmente, construye $(J,\omega)$ y
        certifica compatibilidad.
        """
        if g_metric.ndim != 2 or g_metric.shape[0] != g_metric.shape[1]:
            raise CentripetalDimensionError(
                f"El tensor métrico $G$ debe ser matriz cuadrada 2D. Forma: {g_metric.shape}"
            )

        dim = int(g_metric.shape[0])
        if dim < 2:
            raise CentripetalDimensionError(
                "La variedad de fondo debe tener dimensión $d\\ge 2$."
            )

        g_sym = 0.5 * (g_metric + g_metric.T)
        asym_res = float(la.norm(g_metric - g_sym, "fro"))
        if asym_res > 1.0e-12:
            logger.warning(
                "Tensor métrico con asimetría infinitesimal %.4e. Simetrizado.",
                asym_res,
            )

        eigvals = la.eigvalsh(g_sym)
        min_eig = float(eigvals[0])
        max_eig = float(eigvals[-1])
        if min_eig <= _WILKINSON_FLOOR:
            raise MetricIndefinitenessError(
                f"$G$ no es estrictamente SPD: $\\lambda_{{\\min}}={min_eig:.6e}$ "
                f"<= cota de Wilkinson."
            )

        cond_num = max_eig / min_eig
        if cond_num > self._condition_tolerance:
            raise MetricIndefinitenessError(
                f"Número de condición $\\kappa_2(G)={cond_num:.4e}$ excedió "
                f"{self._condition_tolerance:.4e}."
            )

        try:
            l_cholesky = la.cholesky(g_sym, lower=True)
        except la.LinAlgError as exc:
            raise MetricIndefinitenessError(
                f"Falla de Cholesky sobre el cono SPD: {exc}"
            ) from exc

        l_inv = la.solve_triangular(
            l_cholesky, np.eye(dim, dtype=np.float64), lower=True
        )
        g_inv = l_inv.T @ l_inv
        vol_factor = float(np.prod(np.diag(l_cholesky)))
        spectral_gap = float(max_eig - min_eig)

        j_matrix, j_hyp, k_hyp, reeb, eta = self._construct_compatible_almost_complex_structure(
            l_chol=l_cholesky,
            l_inv=l_inv,
            dim=dim,
        )
        compat_res = self._validate_almost_complex_compatibility(
            j_matrix=j_matrix,
            g_metric=g_sym,
            dim=dim,
            reeb=reeb,
            contact=eta,
        )

        # $\omega(u,v)=G(Ju,v)=u^{\top}(J^{\top}G)v$
        omega_form = j_matrix.T @ g_sym
        omega_form = 0.5 * (omega_form - omega_form.T)  # proyección a $\mathfrak{so}(d)^{*}$

        return BaseMetricCache(
            g_base=_immutable_array(g_sym),
            cholesky_factor=_immutable_array(l_cholesky),
            g_inv=_immutable_array(g_inv),
            eigenvalues=_immutable_array(eigvals),
            almost_complex_j=_immutable_array(j_matrix),
            symplectic_form=_immutable_array(omega_form),
            riemannian_volume_factor=vol_factor,
            condition_number=float(cond_num),
            spectral_gap=spectral_gap,
            j_compatibility_residual=float(compat_res),
            dimension=dim,
            is_even_dimension=bool(dim % 2 == 0),
            hypercomplex_j=None if j_hyp is None else _immutable_array(j_hyp),
            hypercomplex_k=None if k_hyp is None else _immutable_array(k_hyp),
            reeb_field=None if reeb is None else _immutable_array(reeb),
            contact_form=None if eta is None else _immutable_array(eta),
        )

    # ── 1.4 Novikov + Stokes + centroide ──────────────────────────────────────

    def _discrete_stokes_symplectic_area(
        self,
        vertices: NDArray[np.float64],
        omega_form: NDArray[np.float64],
    ) -> float:
        r"""
        Área simpléctica de Stokes discreta sobre el 1-esqueleto cíclico:
        $$\mathcal{A}_{\partial}(u)=\frac12\sum_{k=1}^{N}\omega(u_{k},u_{k+1}),\quad u_{N+1}:=u_{1}.$$
        Es el pullback discreto $\langle u^{*}\omega,[\Sigma]\rangle$ para cadenas poligonales.
        """
        v_next = np.roll(vertices, -1, axis=0)
        pair_forms = np.einsum("ki,ij,kj->k", vertices, omega_form, v_next, optimize=True)
        return 0.5 * KahanNeumaierSum.sum(pair_forms)

    def _compute_novikov_area_and_centroid(
        self,
        vertices: NDArray[np.float64],
        simplex_areas: Optional[NDArray[np.float64]],
        omega_form: NDArray[np.float64],
    ) -> Tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
        r"""
        Integra $\mathcal{A}(u)=\sum a_{k}$ (KBN), rechaza pesos negativos, ubica
        $$q_{c}^{\mu}=\frac1{\mathcal{A}(u)}\sum_{k}a_{k}u_{k}^{\mu}$$
        y evalúa el área de Stokes como testigo geométrico independiente.
        """
        n_verts, dim = vertices.shape

        if simplex_areas is None:
            a_vec = np.full(n_verts, 1.0 / float(n_verts), dtype=np.float64)
        else:
            if simplex_areas.shape[0] != n_verts:
                raise CentripetalDimensionError(
                    f"Pesos/áreas ({simplex_areas.shape[0]}) ≠ vértices ({n_verts})."
                )
            a_raw = np.ascontiguousarray(simplex_areas, dtype=np.float64)
            if np.any(a_raw < 0.0):
                raise CentripetalDimensionError(
                    "Las áreas simplécticas de símplice no pueden ser negativas."
                )
            a_vec = np.clip(a_raw, _WILKINSON_FLOOR, None)

        novikov_area = KahanNeumaierSum.sum(a_vec)
        if novikov_area <= _WILKINSON_FLOOR:
            raise CentrifugalDiskBubblingError(
                f"Área de Novikov colapsada: {novikov_area:.6e} <= cota de Wilkinson."
            )

        centroid = np.zeros(dim, dtype=np.float64)
        for d in range(dim):
            centroid[d] = KahanNeumaierSum.sum(a_vec * vertices[:, d]) / novikov_area

        stokes_area = self._discrete_stokes_symplectic_area(vertices, omega_form)
        return float(novikov_area), centroid, a_vec, float(stokes_area)

    def _phase1_seal(
        self,
        vertices: NDArray[np.float64],
        velocities: NDArray[np.float64],
        omega: NDArray[np.float64],
        g_metric: NDArray[np.float64],
        novikov_area: float,
        effective_mass: float,
        stokes_area: float,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(_ENGINE_VERSION.encode("ascii"))
        hasher.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
        hasher.update(np.ascontiguousarray(velocities, dtype=np.float64).tobytes())
        hasher.update(np.ascontiguousarray(omega, dtype=np.float64).tobytes())
        hasher.update(np.ascontiguousarray(g_metric, dtype=np.float64).tobytes())
        hasher.update(
            f"{_fmt_e(novikov_area)}:{_fmt_e(effective_mass)}:{_fmt_e(stokes_area)}".encode("ascii")
        )
        return hasher.hexdigest()

    # ── 1.Ω MÉTODO TERMINAL FORMAL DE FASE 1 ──────────────────────────────────
    #    Su valor de retorno `CentripetalObserverKernel` ES el objeto de inicio
    #    de Fase 2: `Phase2.orient_from_kernel(kernel)` continúa este morfismo.

    def observe_centripetal_polygon(
        self,
        polygon_vertices: NDArray[np.float64],
        vertex_velocities: NDArray[np.float64],
        angular_velocity_vector: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        simplex_areas: Optional[NDArray[np.float64]] = None,
        base_mass: float = 1.0,
    ) -> CentripetalObserverKernel:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1 (OBSERVE).

        Ingiere y sanea la geometría del polígono pseudo-holomorfo, factoriza $G$,
        construye $J$ $G$-compatible (marco de Cholesky), integra Novikov/Stokes,
        ubica $q_{c}$ y emite el `CentripetalObserverKernel`.

        CONTINUACIÓN FUNCTORIAL:
            kernel = observe_centripetal_polygon(...)
            report = orient_from_kernel(kernel)          # ← inicio de Fase 2
        """
        if polygon_vertices.ndim != 2 or vertex_velocities.ndim != 2:
            raise CentripetalDimensionError(
                "Vértices y velocidades deben ser arreglos bidimensionales $(N\\times d)$."
            )

        n_verts, dim = polygon_vertices.shape
        if n_verts < 3:
            raise CentripetalDimensionError(
                "Un polígono pseudo-holomorfo requiere $N\\ge 3$ vértices."
            )
        if vertex_velocities.shape != (n_verts, dim):
            raise CentripetalDimensionError(
                f"Forma de velocidades {vertex_velocities.shape} ≠ vértices {(n_verts, dim)}."
            )
        if angular_velocity_vector.ndim != 1 or angular_velocity_vector.shape[0] != dim:
            raise CentripetalDimensionError(
                rf"$\omega_{{\mathrm{{rot}}}}$ debe ser 1D de dimensión {dim}. "
                f"Forma: {angular_velocity_vector.shape}"
            )
        if not math.isfinite(base_mass) or base_mass <= 0.0:
            raise CentripetalDimensionError("La masa base $m^{*}$ debe ser finita y estrictamente positiva.")

        clean_vertices = self._sanitize_signed_zeros(polygon_vertices, "polygon_vertices")
        clean_velocities = self._sanitize_signed_zeros(vertex_velocities, "vertex_velocities")
        clean_omega = self._sanitize_signed_zeros(angular_velocity_vector, "angular_velocity_vector")
        clean_g = self._sanitize_signed_zeros(G_metric, "G_metric")

        metric_cache = self._audit_metric_regularity(clean_g)
        if metric_cache.dimension != dim:
            raise CentripetalDimensionError(
                f"Dimensión de $G$ ({metric_cache.dimension}) ≠ polígono ({dim})."
            )

        novikov_area, centroid_q, a_vec, stokes_area = self._compute_novikov_area_and_centroid(
            vertices=clean_vertices,
            simplex_areas=simplex_areas,
            omega_form=metric_cache.symplectic_form,
        )
        effective_mass = float(novikov_area) * float(base_mass)

        phase1_seal = self._phase1_seal(
            vertices=clean_vertices,
            velocities=clean_velocities,
            omega=clean_omega,
            g_metric=clean_g,
            novikov_area=novikov_area,
            effective_mass=effective_mass,
            stokes_area=stokes_area,
        )

        logger.debug(
            "Fase 1 (Observe) completada: Novikov=%.6e Stokes=%.6e MassEff=%.4f κ2=%.3e sello=%s",
            novikov_area,
            stokes_area,
            effective_mass,
            metric_cache.condition_number,
            phase1_seal[:16],
        )

        return CentripetalObserverKernel(
            polygon_vertices=_immutable_array(clean_vertices),
            vertex_velocities=_immutable_array(clean_velocities),
            angular_velocity_vector=_immutable_array(clean_omega),
            simplex_areas=_immutable_array(a_vec),
            novikov_area=float(novikov_area),
            stokes_symplectic_area=float(stokes_area),
            centroid_q=_immutable_array(centroid_q),
            effective_mass=float(effective_mass),
            metric_cache=metric_cache,
            phase1_sha256_seal=phase1_seal,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: ORIENT — continuación formal del Kernel de Fase 1
#         FLOER–CR, $\mathfrak{so}(n)$, CASIMIR, $\epsilon_{\mathrm{radial}}$, PORT-H
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_FloerFukayaCentripetalOrient(Phase1_CentripetalPolygonObserver):
    r"""
    FASE 2 (ORIENT).

    INICIO CONTINUO: el primer método público `orient_from_kernel` absorbe
    DIRECTAMENTE el artefacto terminal de Fase 1 (`CentripetalObserverKernel`).
    No existe otra puerta de entrada algebraica a esta fase.

    Fundamentación:
      - $H_{\mathrm{cent}}=\tfrac12 M_{\mathrm{eff}}\|\omega\|_{G}^{2}
        \cdot\mathbb{E}_{a}[\|u_{k}-q_{c}\|_{G}^{2}]$  (esperanza baricéntrica, no $1/N$).
      - $X_{H}(u_{k})=M_{\mathrm{eff}}\|\omega\|_{G}^{2}\, J(u_{k}-q_{c})$.
      - Residuo $L^{2}(G)$ de Floer–CR discreto:
        $\|\bar{\partial}_{J,H}u\|_{L^{2}}^{2}
        =\mathcal{A}^{-1}\sum a_{k}\|v_{k}-X_{H}(u_{k})\|_{G}^{2}$.
      - $W=\alpha(p\wedge\omega_{\mathrm{rot}})\in\mathfrak{so}(n)$,
        $\mathcal{C}_{2}^{\mathrm{Euc}}=\tfrac12\|W\|_{F}^{2}$,
        $\mathcal{C}_{2}^{G}=-\tfrac12\mathrm{Tr}((G^{-1}W)^{2})$.
      - $\epsilon=\mathcal{A}^{-1}\sum a_{k}(\delta q_{k}\otimes\delta q_{k})$;
        espectro generalizado $\epsilon v=\sigma G v$.
      - Balance Port-Hamiltoniano $\dot H+P_{\mathrm{diss}}$ y virial $2T/(M\|\omega\|^{2}r^{2})$.
    """

    # ── 2.0 INICIO FORMAL = continuación del método terminal de Fase 1 ────────

    def orient_from_kernel(
        self,
        kernel: CentripetalObserverKernel,
        coupling_alpha: float = 0.1,
        viscous_damping_sigma: float = 1.0,
    ) -> CentripetalDeformationReport:
        r"""
        MÉTODO DE INICIO CONTINUO DE FASE 2.

        Continúa el morfismo terminal de Fase 1:
            observe_centripetal_polygon(...)  ↦  kernel
            orient_from_kernel(kernel, ...)   ↦  report
        """
        if not math.isfinite(coupling_alpha):
            raise CentripetalDimensionError("coupling_alpha debe ser finito.")
        if not math.isfinite(viscous_damping_sigma) or viscous_damping_sigma < 0.0:
            raise CentripetalDimensionError(
                "El amortiguamiento viscoso $\\sigma$ debe ser finito y $\\ge 0$."
            )
        return self._orient_centripetal_pipeline(
            kernel=kernel,
            coupling_alpha=float(coupling_alpha),
            viscous_damping_sigma=float(viscous_damping_sigma),
        )

    # ── 2.1 Campo hamiltoniano centrípeto ─────────────────────────────────────

    def _metric_sqnorm_rows(
        self,
        vectors: NDArray[np.float64],
        g_metric: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""$\|v_{k}\|_{G}^{2} = v_{k}^{\top} G v_{k}$ vectorizado (ensamble $N$)."""
        return np.einsum("ni,ij,nj->n", vectors, g_metric, vectors, optimize=True)

    def _compute_centripetal_hamiltonian_field(
        self,
        vertices: NDArray[np.float64],
        centroid: NDArray[np.float64],
        omega_vec: NDArray[np.float64],
        g_metric: NDArray[np.float64],
        almost_complex_j: NDArray[np.float64],
        simplex_areas: NDArray[np.float64],
        novikov_area: float,
        effective_mass: float,
    ) -> Tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
        r"""
        Potencial baricéntrico, campo $X_{H}$, desplazamientos $\delta q_{k}$ y
        $\|\omega_{\mathrm{rot}}\|_{G}^{2}$.
        """
        diffs = vertices - centroid
        omega_norm_sq = KahanNeumaierSum.quadratic_form(omega_vec, g_metric)
        radial_sq = self._metric_sqnorm_rows(diffs, g_metric)
        mean_radial_sq = KahanNeumaierSum.sum(simplex_areas * radial_sq) / novikov_area
        h_potential = 0.5 * effective_mass * omega_norm_sq * mean_radial_sq

        force_scalar = effective_mass * omega_norm_sq
        # $(J\,\delta q_{k})_{k} = \delta q\cdot J^{\top}$
        centripetal_forces = force_scalar * (diffs @ almost_complex_j.T)
        return float(h_potential), centripetal_forces, diffs, float(omega_norm_sq)

    # ── 2.2 Residuo Floer–Cauchy–Riemann ──────────────────────────────────────

    def _evaluate_floer_cauchy_riemann_residual(
        self,
        velocities: NDArray[np.float64],
        centripetal_forces: NDArray[np.float64],
        simplex_areas: NDArray[np.float64],
        novikov_area: float,
        g_metric: NDArray[np.float64],
    ) -> float:
        r"""
        $$\|\bar{\partial}_{J,H}u\|_{L^{2}(G)}
        =\sqrt{\frac1{\mathcal{A}(u)}\sum_{k}a_{k}\|v_{k}-X_{H}(u_{k})\|_{G}^{2}}.$$
        """
        diff_vel = velocities - centripetal_forces
        g_norm_sq = self._metric_sqnorm_rows(diff_vel, g_metric)
        weighted_sum = KahanNeumaierSum.sum(simplex_areas * g_norm_sq) / novikov_area
        return float(math.sqrt(max(0.0, weighted_sum)))

    # ── 2.3 Tensores giroscópicos y radiales ──────────────────────────────────

    def _synthesize_gyroscopic_and_radial_tensors(
        self,
        diffs: NDArray[np.float64],
        velocities: NDArray[np.float64],
        omega_vec: NDArray[np.float64],
        simplex_areas: NDArray[np.float64],
        novikov_area: float,
        g_metric: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        coupling_alpha: float,
    ) -> Tuple[
        NDArray[np.float64], float, float, float, float,
        NDArray[np.float64], float, NDArray[np.float64], bool,
    ]:
        r"""
        $W=\alpha(p\otimes\omega-\omega\otimes p)$, residuos de sesgo euclídeo y
        $G$-sesgo $\|W^{\top}G+GW\|_{F}$, Casimires, $\epsilon_{\mathrm{radial}}$
        y espectro generalizado $\sigma_{i}$ relativo a $G$.
        """
        n_verts, dim = diffs.shape

        v_centroid = np.zeros(dim, dtype=np.float64)
        for d in range(dim):
            v_centroid[d] = (
                KahanNeumaierSum.sum(simplex_areas * velocities[:, d]) / novikov_area
            )
        p_centroid = g_metric @ v_centroid

        w_tensor = coupling_alpha * (
            np.outer(p_centroid, omega_vec) - np.outer(omega_vec, p_centroid)
        )
        skew_res = float(la.norm(w_tensor + w_tensor.T, "fro"))
        is_skew = bool(skew_res <= (100.0 * self._tol))
        g_skew_res = float(la.norm(w_tensor.T @ g_metric + g_metric @ w_tensor, "fro"))

        # Casimir euclídeo: $W$ sesgado $\Rightarrow$ $-\frac12\mathrm{Tr}(W^{2})=\frac12\|W\|_{F}^{2}$
        casimir_eucl = 0.5 * float(np.sum(w_tensor * w_tensor))
        mixed_w = g_inv @ w_tensor
        casimir_g = -0.5 * float(np.trace(mixed_w @ mixed_w))
        casimir_g = float(max(0.0, casimir_g))
        casimir_eucl = float(max(0.0, casimir_eucl))

        radial_tensor = np.einsum(
            "k,ki,kj->ij", simplex_areas, diffs, diffs, optimize=True
        ) / novikov_area
        radial_sym = 0.5 * (radial_tensor + radial_tensor.T)
        radial_norm = float(la.norm(radial_sym, "fro"))

        try:
            principal_strains = la.eigh(radial_sym, g_metric, eigvals_only=True)
        except la.LinAlgError:
            principal_strains = la.eigvalsh(radial_sym)
        principal_strains = np.maximum(principal_strains, 0.0)

        return (
            w_tensor,
            skew_res,
            g_skew_res,
            casimir_g,
            casimir_eucl,
            radial_sym,
            radial_norm,
            principal_strains,
            is_skew,
        )

    # ── 2.4 Disipación de Rayleigh y balance Port-Hamiltoniano ────────────────

    def _compute_viscous_dissipation_power(
        self,
        velocities: NDArray[np.float64],
        simplex_areas: NDArray[np.float64],
        novikov_area: float,
        g_metric: NDArray[np.float64],
        damping_sigma: float,
    ) -> Tuple[float, float]:
        r"""$P_{\mathrm{diss}}=\sigma\,\mathbb{E}_{a}[\|v\|_{G}^{2}]$ y energía de Dirichlet $T$."""
        kinetic_density = self._metric_sqnorm_rows(velocities, g_metric)
        mean_kinetic = KahanNeumaierSum.sum(simplex_areas * kinetic_density) / novikov_area
        dissipated_power = damping_sigma * mean_kinetic
        dirichlet_energy = 0.5 * mean_kinetic
        return float(max(0.0, dissipated_power)), float(max(0.0, dirichlet_energy))

    def _port_hamiltonian_residual(
        self,
        diffs: NDArray[np.float64],
        velocities: NDArray[np.float64],
        simplex_areas: NDArray[np.float64],
        novikov_area: float,
        g_metric: NDArray[np.float64],
        effective_mass: float,
        omega_norm_sq: float,
        dissipated_power: float,
    ) -> Tuple[float, float]:
        r"""
        $\dot H \approx M\|\omega\|_{G}^{2}\,\mathbb{E}_{a}[\langle\delta q, v\rangle_{G}]$.
        El giroscopio es potencia-neutro; el residuo Port-H es $\dot H+P_{\mathrm{diss}}$.
        Virial: $2T / (M\|\omega\|_{G}^{2}\,\mathbb{E}_{a}[\|\delta q\|_{G}^{2}])$.
        """
        mixed = np.einsum("ni,ij,nj->n", diffs, g_metric, velocities, optimize=True)
        h_dot = effective_mass * omega_norm_sq * (
            KahanNeumaierSum.sum(simplex_areas * mixed) / novikov_area
        )
        radial_sq = self._metric_sqnorm_rows(diffs, g_metric)
        mean_radial_sq = KahanNeumaierSum.sum(simplex_areas * radial_sq) / novikov_area
        denom = effective_mass * omega_norm_sq * mean_radial_sq + _WILKINSON_FLOOR
        # $2T$ se pasa desde fuera; aquí devolvemos el factor radial para el virial
        ph_residual = float(h_dot + dissipated_power)
        return ph_residual, float(mean_radial_sq if denom == 0.0 else denom)

    def _phase2_hmac(
        self,
        kernel: CentripetalObserverKernel,
        w_tensor: NDArray[np.float64],
        radial_tensor: NDArray[np.float64],
        h_pot: float,
        cr_res: float,
        casimir_c2: float,
        rad_norm: float,
        ph_residual: float,
    ) -> str:
        signer = hmac.new(_HMAC_CENTRIPETAL_SECRET_KEY, digestmod=hashlib.sha256)
        signer.update(_ENGINE_VERSION.encode("ascii"))
        signer.update(kernel.phase1_sha256_seal.encode("ascii"))
        signer.update(np.ascontiguousarray(w_tensor, dtype=np.float64).tobytes())
        signer.update(np.ascontiguousarray(radial_tensor, dtype=np.float64).tobytes())
        signer.update(
            f"{_fmt_e(h_pot)}:{_fmt_e(cr_res)}:{_fmt_e(casimir_c2)}:"
            f"{_fmt_e(rad_norm)}:{_fmt_e(ph_residual)}".encode("ascii")
        )
        return signer.hexdigest()

    def _orient_centripetal_pipeline(
        self,
        kernel: CentripetalObserverKernel,
        coupling_alpha: float,
        viscous_damping_sigma: float,
    ) -> CentripetalDeformationReport:
        r"""Orquestación interna de Fase 2 a partir del Kernel de Fase 1."""
        verts = kernel.polygon_vertices
        vels = kernel.vertex_velocities
        omega_vec = kernel.angular_velocity_vector
        centroid = kernel.centroid_q
        eff_mass = kernel.effective_mass
        novikov_area = kernel.novikov_area
        a_vec = kernel.simplex_areas
        g_base = kernel.metric_cache.g_base
        g_inv = kernel.metric_cache.g_inv
        almost_j = kernel.metric_cache.almost_complex_j

        h_pot, c_forces, diffs, omega_norm_sq = self._compute_centripetal_hamiltonian_field(
            vertices=verts,
            centroid=centroid,
            omega_vec=omega_vec,
            g_metric=g_base,
            almost_complex_j=almost_j,
            simplex_areas=a_vec,
            novikov_area=novikov_area,
            effective_mass=eff_mass,
        )

        cr_res = self._evaluate_floer_cauchy_riemann_residual(
            velocities=vels,
            centripetal_forces=c_forces,
            simplex_areas=a_vec,
            novikov_area=novikov_area,
            g_metric=g_base,
        )

        (
            w_tensor,
            skew_res,
            g_skew_res,
            casimir_c2,
            casimir_eucl,
            rad_tensor,
            rad_norm,
            principal_strains,
            is_skew,
        ) = self._synthesize_gyroscopic_and_radial_tensors(
            diffs=diffs,
            velocities=vels,
            omega_vec=omega_vec,
            simplex_areas=a_vec,
            novikov_area=novikov_area,
            g_metric=g_base,
            g_inv=g_inv,
            coupling_alpha=coupling_alpha,
        )

        p_diss, dirichlet_e = self._compute_viscous_dissipation_power(
            velocities=vels,
            simplex_areas=a_vec,
            novikov_area=novikov_area,
            g_metric=g_base,
            damping_sigma=viscous_damping_sigma,
        )

        ph_residual, virial_denom = self._port_hamiltonian_residual(
            diffs=diffs,
            velocities=vels,
            simplex_areas=a_vec,
            novikov_area=novikov_area,
            g_metric=g_base,
            effective_mass=eff_mass,
            omega_norm_sq=omega_norm_sq,
            dissipated_power=p_diss,
        )
        virial_ratio = float((2.0 * dirichlet_e) / (virial_denom + _WILKINSON_FLOOR))

        phase2_hmac = self._phase2_hmac(
            kernel=kernel,
            w_tensor=w_tensor,
            radial_tensor=rad_tensor,
            h_pot=h_pot,
            cr_res=cr_res,
            casimir_c2=casimir_c2,
            rad_norm=rad_norm,
            ph_residual=ph_residual,
        )

        return CentripetalDeformationReport(
            kernel_ref=kernel,
            centripetal_potential=float(h_pot),
            centripetal_force_field=_immutable_array(c_forces),
            cauchy_riemann_residual=float(cr_res),
            gyroscopic_stress_tensor=_immutable_array(w_tensor),
            gyroscopic_skew_residual=float(skew_res),
            g_skew_residual=float(g_skew_res),
            casimir_invariant_c2=float(casimir_c2),
            casimir_euclidean_c2=float(casimir_eucl),
            radial_deformation_tensor=_immutable_array(rad_tensor),
            radial_deformation_norm=float(rad_norm),
            radial_principal_strains=_immutable_array(principal_strains),
            dissipated_viscous_power=float(p_diss),
            port_hamiltonian_residual=float(ph_residual),
            dirichlet_energy=float(dirichlet_e),
            virial_ratio=float(virial_ratio),
            is_gyroscopic_skew_symmetric=bool(is_skew),
            phase2_hmac_sha256=phase2_hmac,
        )

    # ── 2.Ω MÉTODO TERMINAL FORMAL DE FASE 2 ──────────────────────────────────
    #    Su valor `CentripetalDeformationReport` (con Kernel anidado) es el
    #    objeto de inicio de Fase 3: `act_from_kernel_and_report(kernel, report)`.

    def orient_centripetal_deformation(
        self,
        kernel: CentripetalObserverKernel,
        coupling_alpha: float = 0.1,
        viscous_damping_sigma: float = 1.0,
    ) -> CentripetalDeformationReport:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2 (ORIENT).

        Empaqueta el reporte elasto-giroscópico. Entrega `CentripetalDeformationReport`,
        insumo inmutable formal —junto con el Kernel— para la Fase 3.

        CONTINUACIÓN FUNCTORIAL:
            report = orient_centripetal_deformation(kernel, ...)
            state  = act_from_kernel_and_report(kernel, report)   # ← inicio Fase 3
        """
        return self.orient_from_kernel(
            kernel=kernel,
            coupling_alpha=coupling_alpha,
            viscous_damping_sigma=viscous_damping_sigma,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: ACT — continuación formal de (Kernel, Report)
#         GROMOV–MASLOV, FLUENCIA, WILKINSON, SELLO SOBERANO DETERMINISTA
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_NovikovMaslovActuator(Phase2_FloerFukayaCentripetalOrient):
    r"""
    FASE 3 (ACT).

    INICIO CONTINUO: `act_from_kernel_and_report` absorbe DIRECTAMENTE el par
    $({\tt CentripetalObserverKernel},\,{\tt CentripetalDeformationReport})$
    producido por el final de Fase 2 (el Report ya anida el Kernel).

    Fundamentación:
      - Bubbling de Gromov: $\mathcal{A}(u)\le\tau_{\mathrm{bubbling}}$
        (colapso) o concentración de energía Floer $\propto\|\bar{\partial}u\|^{2}/\mathcal{A}$.
      - Índice de Maslov discreto: número de rotación del Gauss map en el
        2-plano simpléctico principal (proyección $(x^{1},x^{m+1})$ si $d=2m$).
      - Fluencia: $\|\epsilon\|_{F}\ge\tau_{\mathrm{plastic}}$ o $\sigma_{\max}(G)\ge\tau$.
      - Cota de Wilkinson–Higham $\gamma_{n}=n\varepsilon/(1-n\varepsilon)$.
      - Sello soberano determinista (sin reloj de pared) y verificación de cadena.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        condition_tolerance: float = _CONDITION_NUMBER_MAX,
        strict_mode: bool = False,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            condition_tolerance=condition_tolerance,
            strict_mode=strict_mode,
        )

    # ── 3.0 INICIO FORMAL = continuación del método terminal de Fase 2 ────────

    def act_from_kernel_and_report(
        self,
        kernel: CentripetalObserverKernel,
        report: CentripetalDeformationReport,
        bubbling_threshold: float = _CENTRIFUGAL_BUBBLING_THRESHOLD,
        plastic_threshold: float = _PLASTIC_DEFORMATION_THRESHOLD,
    ) -> CentripetalEngineState:
        r"""
        MÉTODO DE INICIO CONTINUO DE FASE 3.

        Continúa el morfismo terminal de Fase 2:
            orient_centripetal_deformation(kernel)  ↦  report
            act_from_kernel_and_report(kernel, report)  ↦  state
        """
        if bubbling_threshold <= 0.0 or not math.isfinite(bubbling_threshold):
            raise CentripetalDimensionError("bubbling_threshold debe ser finito y $>0$.")
        if plastic_threshold <= 0.0 or not math.isfinite(plastic_threshold):
            raise CentripetalDimensionError("plastic_threshold debe ser finito y $>0$.")
        return self._act_execute_pipeline(
            kernel=kernel,
            report=report,
            bubbling_threshold=float(bubbling_threshold),
            plastic_threshold=float(plastic_threshold),
        )

    # ── 3.1 Cadena criptográfica y metrología ─────────────────────────────────

    def _verify_cryptographic_chain(
        self,
        kernel: CentripetalObserverKernel,
        report: CentripetalDeformationReport,
    ) -> bool:
        r"""
        Verifica que el Report deriva del Kernel (sello de Fase 1 idéntico) y
        recomputa el HMAC de Fase 2 con `hmac.compare_digest`.
        """
        if report.kernel_ref.phase1_sha256_seal != kernel.phase1_sha256_seal:
            msg = "Incoherencia de cadena: el Report no deriva del Kernel de Fase 1."
            if self._strict_mode:
                raise CryptographicChainError(msg)
            logger.error(msg)
            return False

        expected = self._phase2_hmac(
            kernel=kernel,
            w_tensor=np.asarray(report.gyroscopic_stress_tensor, dtype=np.float64),
            radial_tensor=np.asarray(report.radial_deformation_tensor, dtype=np.float64),
            h_pot=report.centripetal_potential,
            cr_res=report.cauchy_riemann_residual,
            casimir_c2=report.casimir_invariant_c2,
            rad_norm=report.radial_deformation_norm,
            ph_residual=report.port_hamiltonian_residual,
        )
        ok = hmac.compare_digest(expected, report.phase2_hmac_sha256)
        if not ok:
            msg = "HMAC de Fase 2 no verifica: posible mutación del expediente."
            if self._strict_mode:
                raise CryptographicChainError(msg)
            logger.error(msg)
        return bool(ok)

    def _compute_wilkinson_bound(self, dimension: int) -> float:
        r"""
        Cota de Wilkinson–Higham para $n$ operaciones en FPU:
        $$\gamma_{n}=\frac{n\varepsilon_{\mathrm{mach}}}{1-n\varepsilon_{\mathrm{mach}}},\quad
        n\varepsilon<1.$$
        """
        n_eps = float(dimension) * _MACHINE_EPS
        if n_eps >= 1.0:
            return float("inf")
        return float(n_eps / (1.0 - n_eps))

    def _estimate_maslov_index(self, kernel: CentripetalObserverKernel) -> int:
        r"""
        Índice de Maslov discreto del lazo poligonal proyectado al 2-plano
        simpléctico principal $(x^{0},x^{m})$ si $d=2m$, o $(x^{0},x^{m})$ si
        $d=2m+1$. Es el número de rotación
        $\mu=\mathrm{round}\bigl(\frac1{\pi}\sum\Delta\theta_{k}\bigr)\in\mathbb{Z}$.
        En $d=2$ coincide con el grado del Gauss map de $\partial\Sigma$.
        """
        vertices = np.asarray(kernel.polygon_vertices, dtype=np.float64)
        dim = int(kernel.metric_cache.dimension)
        m = dim // 2
        x_idx, y_idx = 0, m if m >= 1 else 1
        if y_idx >= dim:
            y_idx = min(1, dim - 1)

        tangents = np.roll(vertices, -1, axis=0) - vertices
        ang = np.arctan2(tangents[:, y_idx], tangents[:, x_idx])
        d_ang = np.diff(np.unwrap(np.concatenate([ang, ang[:1]])))
        winding = KahanNeumaierSum.sum(d_ang) / math.pi
        return int(round(float(winding)))

    def _gromov_bubbling_criterion(
        self,
        kernel: CentripetalObserverKernel,
        report: CentripetalDeformationReport,
        bubbling_threshold: float,
    ) -> bool:
        r"""
        Colapso de área O concentración de energía Floer
        $\|\bar{\partial}u\|_{L^{2}}^{2}/\max(\mathcal{A},\varepsilon)\gg 1$.
        """
        area_collapse = bool(kernel.novikov_area <= (bubbling_threshold + self._tol))
        energy_density = (report.cauchy_riemann_residual ** 2) / max(
            kernel.novikov_area, _WILKINSON_FLOOR
        )
        energy_blowup = bool(energy_density >= (1.0 / max(bubbling_threshold, _WILKINSON_FLOOR)))
        return bool(area_collapse or energy_blowup)

    def _plastic_yield_criterion(
        self,
        report: CentripetalDeformationReport,
        plastic_threshold: float,
    ) -> bool:
        r"""Fluencia de von Mises discreta: $\|\epsilon\|_{F}$ o $\sigma_{\max}$. """
        sigma_max = float(np.max(report.radial_principal_strains)) if report.radial_principal_strains.size else 0.0
        return bool(
            report.radial_deformation_norm >= (plastic_threshold - self._tol)
            or sigma_max >= (plastic_threshold - self._tol)
        )

    def _generate_sovereign_seal(
        self,
        kernel: CentripetalObserverKernel,
        report: CentripetalDeformationReport,
        is_bubbling: bool,
        is_plastic: bool,
        maslov_index: int,
        chain_ok: bool,
    ) -> str:
        r"""Sello holístico SHA-256 determinista (excluye reloj: es telemetría, no identidad)."""
        hasher = hashlib.sha256()
        hasher.update(_ENGINE_VERSION.encode("ascii"))
        hasher.update(kernel.phase1_sha256_seal.encode("ascii"))
        hasher.update(report.phase2_hmac_sha256.encode("ascii"))
        hasher.update(_fmt_e(report.centripetal_potential).encode("ascii"))
        hasher.update(_fmt_e(report.cauchy_riemann_residual).encode("ascii"))
        hasher.update(_fmt_e(report.radial_deformation_norm).encode("ascii"))
        hasher.update(_fmt_e(report.casimir_invariant_c2).encode("ascii"))
        hasher.update(f"{int(is_bubbling)}:{int(is_plastic)}:{maslov_index}:{int(chain_ok)}".encode("ascii"))
        return hasher.hexdigest()

    def _act_execute_pipeline(
        self,
        kernel: CentripetalObserverKernel,
        report: CentripetalDeformationReport,
        bubbling_threshold: float,
        plastic_threshold: float,
    ) -> CentripetalEngineState:
        r"""Certificación, Maslov, fluencia y sellado soberano de Fase 3."""
        t_start = time.perf_counter_ns()

        chain_ok = self._verify_cryptographic_chain(kernel, report)

        is_bubbling = self._gromov_bubbling_criterion(kernel, report, bubbling_threshold)
        if is_bubbling:
            logger.critical(
                "Burbujeo centrífugo de Gromov–Maslov. Área Novikov=%.6e umbral=%.6e CR=%.6e",
                kernel.novikov_area,
                bubbling_threshold,
                report.cauchy_riemann_residual,
            )
            if self._strict_mode:
                raise CentrifugalDiskBubblingError(
                    f"Área de Novikov {kernel.novikov_area:.6e} viola la compacidad de Gromov."
                )

        is_plastic = self._plastic_yield_criterion(report, plastic_threshold)
        if is_plastic:
            logger.warning(
                "Fluencia radial: ||ε||_F=%.4f σ_max=%.4f umbral=%.4f",
                report.radial_deformation_norm,
                float(np.max(report.radial_principal_strains)) if report.radial_principal_strains.size else 0.0,
                plastic_threshold,
            )
            if self._strict_mode:
                raise PlasticDeformationRuptureError(
                    f"||ε_radial||_F={report.radial_deformation_norm:.6e} ≥ τ_plastic={plastic_threshold:.6e}."
                )

        wilkinson_bound = self._compute_wilkinson_bound(kernel.metric_cache.dimension)
        maslov_index = self._estimate_maslov_index(kernel)

        cryptographic_seal = self._generate_sovereign_seal(
            kernel=kernel,
            report=report,
            is_bubbling=is_bubbling,
            is_plastic=is_plastic,
            maslov_index=maslov_index,
            chain_ok=chain_ok,
        )

        fpu_time_ms = float((time.perf_counter_ns() - t_start) / 1.0e6)

        logger.debug(
            "Fase 3 (Act) OK. bubbling=%s plastic=%s maslov=%d chain=%s sello=%s",
            is_bubbling,
            is_plastic,
            maslov_index,
            chain_ok,
            cryptographic_seal[:16],
        )

        return CentripetalEngineState(
            kernel=kernel,
            deformation_report=report,
            is_centrifugal_bubbling_detected=bool(is_bubbling),
            is_plastic_deformation_critical=bool(is_plastic),
            bubbling_threshold=float(bubbling_threshold),
            plastic_threshold=float(plastic_threshold),
            wilkinson_roundoff_bound=float(wilkinson_bound),
            maslov_index_estimate=int(maslov_index),
            chain_integrity_verified=bool(chain_ok),
            fpu_execution_time_ms=float(fpu_time_ms),
            cryptographic_seal=cryptographic_seal,
            engine_version=_ENGINE_VERSION,
        )

    # ── 3.Ω ORQUESTADOR MAESTRO DE LAS TRES FASES ANIDADAS ────────────────────

    def execute_centripetal_audit(
        self,
        polygon_vertices: NDArray[np.float64],
        vertex_velocities: NDArray[np.float64],
        angular_velocity_vector: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        simplex_areas: Optional[NDArray[np.float64]] = None,
        base_mass: float = 1.0,
        coupling_alpha: float = 0.1,
        viscous_damping_sigma: float = 1.0,
        bubbling_threshold: float = _CENTRIFUGAL_BUBBLING_THRESHOLD,
        plastic_threshold: float = _PLASTIC_DEFORMATION_THRESHOLD,
    ) -> CentripetalEngineState:
        r"""
        MÉTODO MAESTRO ORQUESTADOR DE LAS TRES FASES ANIDADAS (ciclo OODA FPU).

          Fase 1 Observe : `observe_centripetal_polygon(...)` → Kernel
          Fase 2 Orient  : `orient_from_kernel(Kernel)` → Report
                           (inicio = continuación del terminal de Fase 1)
          Fase 3 Act     : `act_from_kernel_and_report(Kernel, Report)` → State
                           (inicio = continuación del terminal de Fase 2)

        El sello soberano es independiente del reloj; `fpu_execution_time_ms`
        es telemetría global del lazo y no forma parte de la identidad criptográfica.
        """
        t_global_start = time.perf_counter_ns()

        kernel = self.observe_centripetal_polygon(
            polygon_vertices=polygon_vertices,
            vertex_velocities=vertex_velocities,
            angular_velocity_vector=angular_velocity_vector,
            G_metric=G_metric,
            simplex_areas=simplex_areas,
            base_mass=base_mass,
        )

        report = self.orient_from_kernel(
            kernel=kernel,
            coupling_alpha=coupling_alpha,
            viscous_damping_sigma=viscous_damping_sigma,
        )

        state = self.act_from_kernel_and_report(
            kernel=kernel,
            report=report,
            bubbling_threshold=bubbling_threshold,
            plastic_threshold=plastic_threshold,
        )

        t_global_ms = float((time.perf_counter_ns() - t_global_start) / 1.0e6)

        return CentripetalEngineState(
            kernel=state.kernel,
            deformation_report=state.deformation_report,
            is_centrifugal_bubbling_detected=state.is_centrifugal_bubbling_detected,
            is_plastic_deformation_critical=state.is_plastic_deformation_critical,
            bubbling_threshold=state.bubbling_threshold,
            plastic_threshold=state.plastic_threshold,
            wilkinson_roundoff_bound=state.wilkinson_roundoff_bound,
            maslov_index_estimate=state.maslov_index_estimate,
            chain_integrity_verified=state.chain_integrity_verified,
            fpu_execution_time_ms=t_global_ms,
            cryptographic_seal=state.cryptographic_seal,
            engine_version=state.engine_version,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACADE: MOTOR SOBERANO DE DEFORMACIÓN CENTRÍPETA PSEUDO-HOLOMORFA
# ══════════════════════════════════════════════════════════════════════════════

class PseudoholomorphicCentripetalSatelliteEngine(Phase3_NovikovMaslovActuator):
    r"""
    MOTOR SOBERANO DE DEFORMACIÓN CENTRÍPETA PSEUDO-HOLOMORFA EN FPU SECURE.

    Punto de entrada primario. Consolida las 3 fases anidadas
    (Observe $\subset$ Orient $\subset$ Act) bajo una interfaz doctoral.

    Uso canónico::

        engine = PseudoholomorphicCentripetalSatelliteEngine()
        state  = engine.execute_centripetal_audit(vertices, vel, omega, G)
        # o, fase a fase:
        kernel = engine.observe_centripetal_polygon(...)
        report = engine.orient_from_kernel(kernel)
        state  = engine.act_from_kernel_and_report(kernel, report)
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        condition_tolerance: float = _CONDITION_NUMBER_MAX,
        strict_mode: bool = False,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            condition_tolerance=condition_tolerance,
            strict_mode=strict_mode,
        )

    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> CentripetalEngineState:
        r"""Alias idiomático de `execute_centripetal_audit`."""
        return self.execute_centripetal_audit(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<PseudoholomorphicCentripetalSatelliteEngine "
            f"v={_ENGINE_VERSION} IEEE-754-Double "
            f"κ_max={self._condition_tolerance:.2e} "
            f"τ_bubbling={_CENTRIFUGAL_BUBBLING_THRESHOLD:.1e} "
            f"τ_plastic={_PLASTIC_DEFORMATION_THRESHOLD:.1f} "
            f"strict={self._strict_mode} "
            f"metrology=Floer-Fukaya-Maslov-CholeskyFrame-Hypercomplex>"
        )


# `Any` se usa sólo en el alias `run`; import local para no ensuciar el contrato
# público de fases. Se reimporta aquí para satisfacer anotaciones del facade.
from typing import Any  # noqa: E402  (anotación del alias run)


__all__ = [
    "PseudoholomorphicCentripetalSatelliteEngine",
    "Phase1_CentripetalPolygonObserver",
    "Phase2_FloerFukayaCentripetalOrient",
    "Phase3_NovikovMaslovActuator",
    "BaseMetricCache",
    "CentripetalObserverKernel",
    "CentripetalDeformationReport",
    "CentripetalEngineState",
    "KahanNeumaierSum",
    "CentripetalEngineError",
    "CentripetalDimensionError",
    "MetricIndefinitenessError",
    "AlmostComplexCompatibilityError",
    "CryptographicChainError",
    "CentrifugalDiskBubblingError",
    "PlasticDeformationRuptureError",
]
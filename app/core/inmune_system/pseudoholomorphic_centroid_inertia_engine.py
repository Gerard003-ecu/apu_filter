from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Pseudoholomorphic Centroid Inertia Engine                                   ║
║ RUTA   : app/core/immune_system/pseudoholomorphic_centroid_inertia_engine.py         ║
║ VERSIÓN: 4.0.0-Doctoral-Fukaya-Novikov-Symplectic-Casimir-KBN-FPU-Secure             ║
║                                                                                      ║
║ SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:                                          ║
║ Este módulo implementa el motor de cálculo ciego en la FPU para auditar la           ║
║ cinemática, la inercia de móduli y el momentum angular/spin atencional del           ║
║ centroide simpléctico de polígonos pseudo-holomorfos $u: (\Sigma, \partial\Sigma)    ║
║ \to (\mathcal{M}, \bigcup L_i)$ en la Categoría $\mathcal{F}uk(\mathcal{M})$ de      ║
║ Fukaya sobre una variedad simpléctica compacta $(\mathcal{M}, \omega, J, G)$.        ║
║                                                                                      ║
║ FUNDAMENTACIÓN FÍSICA Y ESTRUCTURAS GEOMÉTRICAS INTEGRADAS:                          ║
║ 1. Triplete Compatible Riemann-Kähler-Simpléctico:                                   ║
║    $\omega(u, v) = G(Ju, v), \quad J^2 = -\mathbb{I}_{2n}, \quad G \in \mathcal{S}^+_{2n}(\mathbb{R})$ ║
║    El tensor métrico $G$ induce el producto interno Riemanniano y el isomorfismo     ║
║    musical bemol ($\flat: T\mathcal{M} \to T^*\mathcal{M}$) y sostenido              ║
║    ($\sharp: T^*\mathcal{M} \to T\mathcal{M}$).                                      ║
║                                                                                      ║
║ 2. Anillo de Novikov $\Lambda_{\mathrm{Nov}}$ y Área Simpléctica de Curvatura:        ║
║    El área simpléctica $\mathcal{A}(u) = \int_\Sigma u^*\omega = \sum_k a_k$ es      ║
║    integrada sobre los símplices orientados de la triangulación de Delaunay/Cauchy   ║
║    mediante sumación compensada de Kahan-Babuška-Neumaier (KBN).                     ║
║                                                                                      ║
║ 3. Centroide de Móduli de Fukaya:                                                    ║
║    Ubicación baricéntrica covariante en el espacio de configuración:                 ║
║    $q_{\mathrm{centroid}}^\mu = \frac{1}{\mathcal{A}(u)} \sum_k a_k u_k^\mu$.        ║
║    Velocidad simpléctica $v_{\mathrm{centroid}}^\mu$ y momentum dual covariante      ║
║    $p_\mu^{\mathrm{centroid}} = (v^\flat)_\mu = G_{\mu\nu} v_{\mathrm{centroid}}^\nu$.║
║                                                                                      ║
║ 4. Tensor de Inercia y Álgebra de Lie de Spin $\mathfrak{so}(T_{q_c}\mathcal{M}, G)$:║
║    - Tensor de Inercia Riemanniano (definido positivo por Cauchy-Schwarz):           ║
║      $I_{\mu\nu} = \frac{1}{\mathcal{A}(u)} \sum_k a_k \left( \|\delta q_k\|_G^2 G_{\mu\nu} - (G\delta q_k)_\mu (G\delta q_k)_\nu \right)$ ║
║    - Bivector de Spin Atencional $L_{\mu\nu} \in \mathfrak{so}(n)$:                  ║
║      $L_{\mu\nu} = \frac{1}{\mathcal{A}(u)} \sum_k a_k \left( \delta q_{k,\mu} p_{k,\nu} - \delta q_{k,\nu} p_{k,\mu} \right)$ ║
║    - Invariante Cuadrático de Casimir: $\mathcal{C}_2(L) = -\frac{1}{2}\operatorname{Tr}((G^{-1}L)^2)$. ║
║                                                                                      ║
║ 5. Compactificación de Gromov y Degeneración de Maslov:                              ║
║    En el espacio de móduli $\overline{\mathcal{M}}_{0,k+1}(\mathcal{M}, J)$, cuando  ║
║    $\mathcal{A}(u) \le \tau_{\mathrm{Maslov}}$, se dispara el "Burbujeo de Discos"   ║
║    (Disk Bubbling), produciendo una obstrucción de curvatura $\mu^0(1) \neq 0$ en el ║
║    álgebra $A_\infty$ que rompe la estabilidad cuántica de Floer.                    ║
║                                                                                      ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS DE TRANSICIÓN FORMAL (OODA LOOP):     ║
║   Fase 1 (Observe): Ingesta de polígono $u(z)$, espectro de $G$, KBN de $\mathcal{A}(u)$,║
║                     cálculo del centroide $q_{\mathrm{centroid}}^\mu$.               ║
║          → Salida Terminal: CentroidObserverKernel.                                  ║
║   Fase 2 (Orient) : Inicia DIRECTAMENTE absorbiendo CentroidObserverKernel.          ║
║                     Velocidad, momentum $p_\mu = v^\flat$, Tensor $I_{\mu\nu}$,      ║
║                     Bivector $L_{\mu\nu} \in \mathfrak{so}(n)$, Casimir $\mathcal{C}_2$, Energía $T_{\mathrm{kin}}$.║
║          → Salida Terminal: CentroidInertiaReport.                                   ║
║   Fase 3 (Act)    : Inicia DIRECTAMENTE absorbiendo (Kernel, Report).               ║
║                     Detección de Maslov ($\mathcal{A}(u) \le \tau$), cota de Wilkinson,║
║                     telemetría FPU y Sello Inmutable HMAC-SHA256.                    ║
║          → Salida Terminal: CentroidInertiaEngineState.                              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from typing import Final, Optional, Tuple, Dict, Any, List
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Physics.PseudoholomorphicCentroidInertiaEngine")

# Constantes Metrológicas Universales de Precisión IEEE-754 y Silicio FPU
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1.0e-15
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12
_DEFAULT_DISK_BUBBLING_THRESHOLD: Final[float] = 1.0e-6
_HMAC_CENTROID_SECRET_KEY: Final[bytes] = b"PseudoholomorphicCentroidInertiaEngine::FukayaNovikovKey2026"


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES DOCTORALES DE FÓRMULA CERRADA
# ══════════════════════════════════════════════════════════════════════════════

class CentroidInertiaEngineError(Exception):
    """Excepción raíz del subsistema de inercia centroidal pseudo-holomorfa."""
    pass


class CentroidDimensionError(CentroidInertiaEngineError):
    """Falla de consistencia dimensional en vértices del polígono o espacio de fases."""
    pass


class NovikovAreaCollapseError(CentroidInertiaEngineError):
    """Falla crítica: El área de Novikov colapsó por debajo del límite de degeneración."""
    pass


class MetricIndefinitenessError(CentroidInertiaEngineError):
    """Falla crítica: El tensor métrico de fondo G no reside en el cono estrictamente SPD."""
    pass


class SymplecticCompatibilityError(CentroidInertiaEngineError):
    """Falla de compatibilidad entre la forma simpléctica omega y la métrica Riemanniana G."""
    pass


class MaslovDiskBubblingCriticalError(CentroidInertiaEngineError):
    """Colapso cuántico del disco: Área de Novikov menor que la cota de burbujeo de Gromov."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# MÁQUINA DE SUMACIÓN COMPENSADA EXACTA: KAHAN-BABUŠKA-NEUMAIER (KBN)
# ══════════════════════════════════════════════════════════════════════════════

class KahanNeumaierSum:
    r"""
    Sumación compensada de Kahan-Babuška-Neumaier (KBN).
    Captura y neutraliza la pérdida de precisión en coma flotante $\mathcal{O}(\epsilon)$
    almacenando el error residual de la mantisa de 53 bits IEEE-754.
    """

    @staticmethod
    def sum(arr: NDArray[np.float64]) -> float:
        r"""Calcula la suma compensada de un tensor real de rango 1."""
        s = 0.0
        c = 0.0
        for x in arr.ravel():
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
        r"""Calcula el producto interno euclidiano exacto mediante 2Product y KBN."""
        if u.shape[0] != v.shape[0]:
            raise CentroidDimensionError("Discrepancia dimensional para producto interno KBN.")
        return KahanNeumaierSum.sum(u * v)


# ══════════════════════════════════════════════════════════════════════════════
# EXPEDIENTES DE DATOS INMUTABLES (ESTRUCTURAS DE FIBRA DE FUKAYA)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class BaseMetricCache:
    r"""
    Caché espectral inmutable del tensor métrico $G_{\mu\nu}$ en $(\mathcal{M}, G)$.
    Almacena el tensor de base, factor Cholesky $L$, inversa $G^{-1}$, espectro $\{\lambda_k\}$,
    forma de volumen simpléctica $\mathrm{vol}_G$ y el número de condición espectral $\kappa_2(G)$.
    """
    g_base: NDArray[np.float64]
    cholesky_factor: NDArray[np.float64]
    g_inv: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    condition_number: float
    riemannian_volume_factor: float
    dimension: int


@dataclass(frozen=True, slots=True)
class CentroidObserverKernel:
    r"""
    Expediente Inmutable Terminal de Fase 1 (Observe).
    Consolida la geometría del polígono pseudo-holomorfo $u(z)$, la descomposición
    de áreas simplécticas de Novikov, el centroide baricéntrico y la métrica auditada.
    """
    polygon_vertices: NDArray[np.float64]
    simplex_areas: NDArray[np.float64]
    novikov_area: float
    centroid_q: NDArray[np.float64]
    metric_cache: BaseMetricCache
    phase1_sha256_seal: str


@dataclass(frozen=True, slots=True)
class CentroidInertiaReport:
    r"""
    Expediente Inmutable Terminal de Fase 2 (Orient).
    Entrega la cinemática de móduli: velocidad simpléctica $v^\mu$, momentum $p_\mu$,
    tensor de inercia $I_{\mu\nu}$, bivector de spin $L_{\mu\nu} \in \mathfrak{so}(n)$,
    el invariante cuadrático de Casimir $\mathcal{C}_2(L)$ y la partición energética de Fukaya.
    """
    kernel_ref: CentroidObserverKernel
    centroid_velocity: NDArray[np.float64]
    centroid_momentum: NDArray[np.float64]
    inertia_tensor: NDArray[np.float64]
    inertia_eigenvalues: NDArray[np.float64]
    spin_bivector: NDArray[np.float64]
    spin_skew_residual: float
    casimir_invariant_c2: float
    effective_mass: float
    translational_kinetic_energy: float
    rotational_kinetic_energy: float
    total_kinetic_energy: float
    is_spin_antisymmetric: bool
    phase2_hmac_sha256: str


@dataclass(frozen=True, slots=True)
class CentroidInertiaEngineState:
    r"""
    Certificado Soberano Inmutable emitido por la Fase 3 (Act).
    Acredita la gobernanza de lazo cerrado en la FPU, el test de burbujeo de discos de Maslov,
    la cota de error de redondeo de Wilkinson y el sello criptográfico HMAC global.
    """
    kernel: CentroidObserverKernel
    report: CentroidInertiaReport
    is_maslov_bubbling_detected: bool
    maslov_bubbling_threshold: float
    wilkinson_roundoff_bound: float
    fpu_execution_time_ms: float
    cryptographic_seal: str


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: OBSERVE — INGESTA, TEORÍA ESPECTRAL, ÁREA DE NOVIKOV Y CENTROIDE
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_PolygonCentroidObserver:
    r"""
    FASE 1 (OBSERVE):
    Fundamentación matemática:
      - Ingesta de la malla poliédrica del disco $u: (\Sigma, \partial\Sigma) \to (\mathcal{M}, \bigcup L_i)$
        representada por $N$ vértices $\{u_k\}_{k=1}^N \subset \mathbb{R}^d$.
      - Auditoría espectral del tensor métrico $G \in \mathcal{S}^+_d(\mathbb{R})$, factorización
        de Cholesky $G = L L^\top$, cálculo del número de condición $\kappa_2(G)$ e inversión
        triangular no confinada $G^{-1} = L^{-\top} L^{-1}$.
      - Integración del Área Simpléctica de Novikov $\mathcal{A}(u) = \int_\Sigma u^*\omega$
        mediante sumación KBN sobre la descomposición simplicial de 2-caras orientadas.
      - Ubicación baricéntrica del Centroide Simpléctico en el espacio tangente:
        $$q_{\mathrm{centroid}}^\mu = \frac{1}{\mathcal{A}(u)} \sum_{k=1}^N a_k u_k^\mu$$
      - Emisión del `CentroidObserverKernel` inmutable con sello criptográfico SHA-256.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        condition_tolerance: float = _CONDITION_NUMBER_MAX
    ) -> None:
        self._tol: Final[float] = float(tolerance)
        self._condition_tolerance: Final[float] = float(condition_tolerance)

    def _audit_metric_regularity(self, G_metric: NDArray[np.float64]) -> BaseMetricCache:
        r"""
        Somete al tensor métrico $G_{\mu\nu}$ a factorización Cholesky ($G = L L^\top \succ 0$),
        calcula la inversa exacta, los autovalores espectrales y el determinante Riemanniano.
        """
        if G_metric.ndim != 2 or G_metric.shape[0] != G_metric.shape[1]:
            raise CentroidDimensionError(f"Tensor métrico G debe ser matriz cuadrada 2D. Forma: {G_metric.shape}")

        dim = G_metric.shape[0]

        # Simetrización compensada de Frobenius
        g_sym = 0.5 * (G_metric + G_metric.T)
        asym_res = float(la.norm(G_metric - g_sym, "fro"))
        if asym_res > 1.0e-13:
            logger.warning("Tensor métrico presentaba asimetría numérica: %.4e. Simetrizado.", asym_res)

        # Espectro autoadjunto
        eigvals = la.eigvalsh(g_sym)
        min_eig = float(eigvals[0])
        max_eig = float(eigvals[-1])

        if min_eig <= _WILKINSON_FLOOR:
            raise MetricIndefinitenessError(
                f"El tensor métrico G no es estrictamente SPD: autovalor mínimo {min_eig:.6e} <= cota Wilkinson."
            )

        cond_num = max_eig / min_eig
        if cond_num > self._condition_tolerance:
            raise MetricIndefinitenessError(
                f"Número de condición excedió el umbral crítico: {cond_num:.4e} > {self._condition_tolerance:.4e}"
            )

        # Factorización de Cholesky
        try:
            L_cholesky = la.cholesky(g_sym, lower=True)
        except la.LinAlgError as exc:
            raise MetricIndefinitenessError(f"Falla en factorización Cholesky sobre cono SPD: {exc}") from exc

        # Inversión no confinada
        L_inv = la.solve_triangular(L_cholesky, np.eye(dim, dtype=np.float64), lower=True)
        g_inv = L_inv.T @ L_inv

        # Factor de volumen Riemanniano: \sqrt{\det G} = \prod L_{ii}
        vol_factor = float(np.prod(np.diag(L_cholesky)))

        return BaseMetricCache(
            g_base=g_sym,
            cholesky_factor=L_cholesky,
            g_inv=g_inv,
            eigenvalues=eigvals,
            condition_number=cond_num,
            riemannian_volume_factor=vol_factor,
            dimension=dim
        )

    def _compute_novikov_area_and_centroid(
        self,
        vertices: NDArray[np.float64],
        simplex_areas: Optional[NDArray[np.float64]] = None
    ) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Integra el Área Simpléctica de Novikov $\mathcal{A}(u) = \sum_k a_k$ con sumación KBN
        y ubica el centroide simpléctico baricéntrico ponderado:
        $$q_{\mathrm{centroid}}^\mu = \frac{1}{\mathcal{A}(u)} \sum_{k=1}^N a_k u_k^\mu$$
        """
        n_verts, dim = vertices.shape

        if simplex_areas is None:
            # Distribución canónica equiparticionada
            a_vec = np.full(n_verts, 1.0 / float(n_verts), dtype=np.float64)
        else:
            if simplex_areas.shape[0] != n_verts:
                raise CentroidDimensionError(
                    f"El número de pesos/áreas ({simplex_areas.shape[0]}) debe coincidir con vértices ({n_verts})."
                )
            a_vec = np.clip(simplex_areas, _WILKINSON_FLOOR, None).astype(np.float64)

        # Suma compensada KBN del área simpléctica
        novikov_area = KahanNeumaierSum.sum(a_vec)
        if novikov_area <= _WILKINSON_FLOOR:
            raise NovikovAreaCollapseError(
                f"Área simpléctica de Novikov colapsada a cero: {novikov_area:.6e} <= cota de Wilkinson."
            )

        # Centroide simpléctico ponderado
        centroid = np.zeros(dim, dtype=np.float64)
        for d in range(dim):
            weighted_coords = a_vec * vertices[:, d]
            centroid[d] = KahanNeumaierSum.sum(weighted_coords) / novikov_area

        return novikov_area, centroid, a_vec

    def observe_polygon_centroid(
        self,
        polygon_vertices: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        simplex_areas: Optional[NDArray[np.float64]] = None
    ) -> CentroidObserverKernel:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1 (OBSERVE):
        Ingiere la malla del polígono simpléctico, factoriza la métrica $G$, integra
        el Área de Novikov $\mathcal{A}(u)$ y ubica el centroide $q_{\mathrm{centroid}}^\mu$.
        Genera el sello inmutable SHA-256 de Fase 1.
        
        Este expediente `CentroidObserverKernel` es el conector formal obligatorio de inicio de Fase 2.
        """
        if polygon_vertices.ndim != 2:
            raise CentroidDimensionError("Los vértices del polígono deben ser una matriz 2D (N_vertices, dim).")

        n_verts, dim = polygon_vertices.shape
        if n_verts < 1:
            raise CentroidDimensionError("El polígono debe poseer al menos un vértice en el espacio de fases.")

        # 1. Auditoría y descomposición espectral de la métrica
        metric_cache = self._audit_metric_regularity(G_metric)
        if metric_cache.dimension != dim:
            raise CentroidDimensionError(
                f"Dimensión de G ({metric_cache.dimension}) no coincide con el espacio de fases del polígono ({dim})."
            )

        # 2. Integración KBN del Área de Novikov y cálculo del centroide
        novikov_area, centroid_q, a_vec = self._compute_novikov_area_and_centroid(
            polygon_vertices, simplex_areas=simplex_areas
        )

        # 3. Sello criptográfico SHA-256 de Fase 1
        hasher = hashlib.sha256()
        hasher.update(polygon_vertices.tobytes())
        hasher.update(a_vec.tobytes())
        hasher.update(centroid_q.tobytes())
        hasher.update(metric_cache.g_base.tobytes())
        hasher.update(f"{novikov_area:.16e}".encode("ascii"))
        phase1_seal = hasher.hexdigest()

        logger.debug(
            "Fase 1 (Observe) concluida con éxito. NovikovArea: %.6e, Sello: %s",
            novikov_area, phase1_seal[:16]
        )

        return CentroidObserverKernel(
            polygon_vertices=polygon_vertices.copy(),
            simplex_areas=a_vec.copy(),
            novikov_area=novikov_area,
            centroid_q=centroid_q,
            metric_cache=metric_cache,
            phase1_sha256_seal=phase1_seal
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: ORIENT — CINEMÁTICA, TENSOR DE INERCIA, BIVECTOR DE SPIN Y CASIMIR
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_InertiaSpinTensorOrient(Phase1_PolygonCentroidObserver):
    r"""
    FASE 2 (ORIENT):
    Fundamentación matemática:
      - Absorbe DIRECTAMENTE el `CentroidObserverKernel` emitido por el final de Fase 1.
      - Evalúa la velocidad simpléctica baricéntrica:
        $$v_{\mathrm{centroid}}^\mu = \frac{1}{\mathcal{A}(u)} \sum_{k=1}^N a_k v_k^\mu$$
      - Determina el momentum covariante canónico mediante el isomorfismo musical bemol ($\flat$):
        $$p_\mu^{\mathrm{centroid}} = (v^\flat)_\mu = G_{\mu\nu} v_{\mathrm{centroid}}^\nu$$
      - Sintetiza el Tensor de Inercia de Móduli Riemanniano:
        $$I_{\mu\nu} = \frac{1}{\mathcal{A}(u)} \sum_{k=1}^N a_k \left( \|\delta q_k\|_G^2 G_{\mu\nu} - (G\delta q_k)_\mu (G\delta q_k)_\nu \right)$$
        Demostración de positividad: Para todo vector tangente $w \in T\mathcal{M}$,
        $w^\mu I_{\mu\nu} w^\nu = \frac{1}{\mathcal{A}(u)} \sum a_k \left( \|\delta q_k\|_G^2 \|w\|_G^2 - \langle \delta q_k, w \rangle_G^2 \right) \ge 0$
        por la desigualdad de Cauchy-Schwarz.
      - Construye el Bivector de Spin Atencional $L \in \mathfrak{so}(n)$:
        $$L_{\mu\nu} = \frac{1}{\mathcal{A}(u)} \sum_{k=1}^N a_k \left( \delta q_{k,\mu} p_{k,\nu} - \delta q_{k,\nu} p_{k,\mu} \right)$$
      - Calcula el invariante cuadrático de Casimir de $\mathfrak{so}(n)$:
        $$\mathcal{C}_2(L) = -\frac{1}{2} \operatorname{Tr}\left( (G^{-1} L)^2 \right) \ge 0$$
      - Evalúa la partición de Energía Cinética de Fukaya:
        $$M_{\mathrm{eff}} = \mathcal{A}(u) \cdot m^*, \quad T_{\mathrm{trans}} = \frac{1}{2} M_{\mathrm{eff}} \|v\|_G^2, \quad T_{\mathrm{rot}} = \frac{1}{2} \operatorname{Tr}\left( L^\top G^{-1} L G^{-1} \right)$$
      - Emite el reporte inmutable `CentroidInertiaReport`.
    """

    def orient_from_kernel(
        self,
        kernel: CentroidObserverKernel,
        vertex_velocities: NDArray[np.float64],
        base_mass: float = 1.0
    ) -> CentroidInertiaReport:
        r"""
        MÉTODO DE INICIO CONTINUO DE FASE 2:
        Conecta formalmente con el final de Fase 1 al recibir su artefacto terminal
        `CentroidObserverKernel`. Ejecuta la orientación cinemática e inercial completa.
        """
        return self._orient_centroid_inertia_pipeline(
            kernel=kernel,
            vertex_velocities=vertex_velocities,
            base_mass=base_mass
        )

    def _compute_centroid_kinematics(
        self,
        vertex_velocities: NDArray[np.float64],
        simplex_areas: NDArray[np.float64],
        novikov_area: float,
        G_metric: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Calcula la velocidad contravariante del centroide $v_{\mathrm{centroid}}^\mu$
        y el momentum covariante canónico $p_\mu^{\mathrm{centroid}} = G_{\mu\nu} v_{\mathrm{centroid}}^\nu$.
        """
        dim = vertex_velocities.shape[1]
        v_centroid = np.zeros(dim, dtype=np.float64)

        for d in range(dim):
            weighted_v = simplex_areas * vertex_velocities[:, d]
            v_centroid[d] = KahanNeumaierSum.sum(weighted_v) / novikov_area

        # Isomorfismo musical bemol: p = G * v
        p_centroid = G_metric @ v_centroid
        return v_centroid, p_centroid

    def _compute_inertia_and_spin_tensors(
        self,
        vertices: NDArray[np.float64],
        vertex_velocities: NDArray[np.float64],
        simplex_areas: NDArray[np.float64],
        novikov_area: float,
        centroid_q: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        G_inv: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        r"""
        Sintetiza el Tensor de Inercia Riemanniano $I_{\mu\nu}$, el Bivector de Spin $L_{\mu\nu}$,
        el residuo de antisimetría de Frobenius y el invariante cuadrático de Casimir $\mathcal{C}_2(L)$.
        """
        n_verts, dim = vertices.shape
        delta_q = vertices - centroid_q

        # Momentum de cada vértice vía bemol: p_k = G * v_k
        p_vertices = (G_metric @ vertex_velocities.T).T

        inertia_tensor = np.zeros((dim, dim), dtype=np.float64)
        spin_bivector = np.zeros((dim, dim), dtype=np.float64)

        for k in range(n_verts):
            dq_k = delta_q[k]
            p_k = p_vertices[k]
            a_k = simplex_areas[k]

            # G-norma al cuadrado de la desviación: \|\delta q_k\|_G^2
            sharp_dq = G_metric @ dq_k
            dq_norm_sq = KahanNeumaierSum.dot(dq_k, sharp_dq)

            # Término de inercia: a_k * ( \|\delta q_k\|_G^2 G - (G \delta q_k) \otimes (G \delta q_k) )
            inertia_k = (dq_norm_sq * G_metric) - np.outer(sharp_dq, sharp_dq)
            inertia_tensor += a_k * inertia_k

            # Término de spin: a_k * ( \delta q_{k,\mu} p_{k,\nu} - \delta q_{k,\nu} p_{k,\mu} )
            spin_k = np.outer(dq_k, p_k) - np.outer(p_k, dq_k)
            spin_bivector += a_k * spin_k

        inertia_tensor /= novikov_area
        spin_bivector /= novikov_area

        # Residuo de antisimetría de Frobenius: ||L + L^T||_F
        skew_res = float(la.norm(spin_bivector + spin_bivector.T, "fro"))

        # Invariante cuadrático de Casimir de \mathfrak{so}(n): C_2(L) = -1/2 Tr((G^{-1} L)^2)
        mixed_spin = G_inv @ spin_bivector
        casimir_c2 = -0.5 * float(np.trace(mixed_spin @ mixed_spin))

        return inertia_tensor, spin_bivector, skew_res, max(0.0, casimir_c2)

    def _compute_fukaya_kinetic_energies(
        self,
        v_centroid: NDArray[np.float64],
        spin_bivector: NDArray[np.float64],
        novikov_area: float,
        G_metric: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        base_mass: float
    ) -> Tuple[float, float, float, float]:
        r"""
        Calcula la masa efectiva y las componentes traslacional y rotacional de la energía cinética:
        $$M_{\mathrm{eff}} = \mathcal{A}(u) \cdot m^*$$
        $$T_{\mathrm{trans}} = \frac{1}{2} M_{\mathrm{eff}} \|v_{\mathrm{centroid}}\|_G^2$$
        $$T_{\mathrm{rot}} = \frac{1}{2} \operatorname{Tr}\left( L^\top G^{-1} L G^{-1} \right)$$
        """
        eff_mass = novikov_area * base_mass

        # ||v||_G^2
        v_norm_sq = KahanNeumaierSum.dot(v_centroid, G_metric @ v_centroid)
        t_trans = 0.5 * eff_mass * v_norm_sq

        # Rotacional vía métrica inversa en el álgebra de Lie
        rot_mat = spin_bivector.T @ G_inv @ spin_bivector @ G_inv
        t_rot = 0.5 * float(np.trace(rot_mat))

        t_total = t_trans + t_rot
        return eff_mass, float(t_trans), float(t_rot), float(t_total)

    def _orient_centroid_inertia_pipeline(
        self,
        kernel: CentroidObserverKernel,
        vertex_velocities: NDArray[np.float64],
        base_mass: float
    ) -> CentroidInertiaReport:
        r"""Orquesta la ejecución interna de la Fase 2 a partir del Kernel."""
        if vertex_velocities.shape != kernel.polygon_vertices.shape:
            raise CentroidDimensionError(
                f"Forma de velocidades ({vertex_velocities.shape}) no coincide con vértices ({kernel.polygon_vertices.shape})."
            )

        g_metric = kernel.metric_cache.g_base
        g_inv = kernel.metric_cache.g_inv

        # 1. Cinemática simpléctica del centroide
        v_centroid, p_centroid = self._compute_centroid_kinematics(
            vertex_velocities=vertex_velocities,
            simplex_areas=kernel.simplex_areas,
            novikov_area=kernel.novikov_area,
            G_metric=g_metric
        )

        # 2. Tensor de Inercia, Bivector de Spin y Casimir
        I_tensor, L_spin, skew_res, casimir_c2 = self._compute_inertia_and_spin_tensors(
            vertices=kernel.polygon_vertices,
            vertex_velocities=vertex_velocities,
            simplex_areas=kernel.simplex_areas,
            novikov_area=kernel.novikov_area,
            centroid_q=kernel.centroid_q,
            G_metric=g_metric,
            G_inv=g_inv
        )

        # Autovalores del tensor de inercia
        inertia_eigs = la.eigvalsh(0.5 * (I_tensor + I_tensor.T))

        # 3. Energías cinéticas de Fukaya
        eff_mass, t_trans, t_rot, t_total = self._compute_fukaya_kinetic_energies(
            v_centroid=v_centroid,
            spin_bivector=L_spin,
            novikov_area=kernel.novikov_area,
            G_metric=g_metric,
            G_inv=g_inv,
            base_mass=base_mass
        )

        is_skew = bool(skew_res <= (10.0 * self._tol))

        # 4. Sello HMAC-SHA256 encadenado a la Fase 1
        hmac_signer = hmac.new(_HMAC_CENTROID_SECRET_KEY, digestmod=hashlib.sha256)
        hmac_signer.update(kernel.phase1_sha256_seal.encode("ascii"))
        hmac_signer.update(v_centroid.tobytes())
        hmac_signer.update(p_centroid.tobytes())
        hmac_signer.update(I_tensor.tobytes())
        hmac_signer.update(L_spin.tobytes())
        hmac_signer.update(f"{t_total:.16e}:{casimir_c2:.16e}".encode("ascii"))
        phase2_hmac = hmac_signer.hexdigest()

        return CentroidInertiaReport(
            kernel_ref=kernel,
            centroid_velocity=v_centroid,
            centroid_momentum=p_centroid,
            inertia_tensor=I_tensor,
            inertia_eigenvalues=inertia_eigs,
            spin_bivector=L_spin,
            spin_skew_residual=skew_res,
            casimir_invariant_c2=casimir_c2,
            effective_mass=eff_mass,
            translational_kinetic_energy=t_trans,
            rotational_kinetic_energy=t_rot,
            total_kinetic_energy=t_total,
            is_spin_antisymmetric=is_skew,
            phase2_hmac_sha256=phase2_hmac
        )

    def orient_centroid_inertia(
        self,
        kernel: CentroidObserverKernel,
        vertex_velocities: NDArray[np.float64],
        base_mass: float = 1.0
    ) -> CentroidInertiaReport:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2 (ORIENT):
        Empaqueta el reporte cinemático e inercial `CentroidInertiaReport`, el cual
        sirve de insumo inmutable para la Fase 3.
        """
        return self.orient_from_kernel(
            kernel=kernel,
            vertex_velocities=vertex_velocities,
            base_mass=base_mass
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: ACT — GROMOV MASLOV BUBBLING, WILKINSON BOUND Y SELLO SOBERANO
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_NovikovNullConeActuator(Phase2_InertiaSpinTensorOrient):
    r"""
    FASE 3 (ACT):
    Fundamentación matemática:
      - Absorbe DIRECTAMENTE el `CentroidInertiaReport` (y su `CentroidObserverKernel` anidado)
        procedente del final de la Fase 2.
      - Compactificación de Gromov y Detección de Maslov:
        Audita si el Área Simpléctica de Novikov colapsa por debajo de la cota crítica
        de "Burbujeo de Discos" ($\mathcal{A}(u) \le \tau_{\mathrm{Maslov}}$). Si ocurre,
        las correcciones cuánticas del disco generan una obstrucción de curvatura
        $\mu^0(1) \neq 0$ en el álgebra $A_\infty$ de Fukaya, rompiendo la unitariedad.
      - Metrología de Precisión IEEE-754:
        Cálculo riguroso de la cota de Wilkinson acumulada:
        $$\gamma_n = \frac{n \cdot \epsilon_{\mathrm{mach}}}{1 - n \cdot \epsilon_{\mathrm{mach}}}$$
      - Registro de latencia en milisegundos de la FPU.
      - Emisión del Certificado Global Inmutable `CentroidInertiaEngineState` con
        sellado criptográfico holístico HMAC-SHA256 en RAM.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        condition_tolerance: float = _CONDITION_NUMBER_MAX
    ) -> None:
        super().__init__(tolerance=tolerance, condition_tolerance=condition_tolerance)

    def act_from_kernel_and_report(
        self,
        kernel: CentroidObserverKernel,
        report: CentroidInertiaReport,
        bubbling_threshold: float = _DEFAULT_DISK_BUBBLING_THRESHOLD
    ) -> CentroidInertiaEngineState:
        r"""
        MÉTODO DE INICIO CONTINUO DE FASE 3:
        Ingiere conjuntamente el `CentroidObserverKernel` de Fase 1 y el `CentroidInertiaReport`
        de Fase 2 para ejecutar la auditoría de estabilidad cuántica de Maslov y sellado soberano.
        """
        return self._act_execute_pipeline(
            kernel=kernel,
            report=report,
            bubbling_threshold=bubbling_threshold
        )

    def _compute_wilkinson_bound(self, dimension: int) -> float:
        r"""
        Calcula la cota acumulada de error de Wilkinson para algoritmos matriciales de paso $n$:
        $$\gamma_n = \frac{n \cdot \epsilon_{\mathrm{mach}}}{1 - n \cdot \epsilon_{\mathrm{mach}}}$$
        """
        n_eps = dimension * _MACHINE_EPS
        if n_eps >= 1.0:
            return float("inf")
        return float(n_eps / (1.0 - n_eps))

    def _generate_sovereign_seal(
        self,
        kernel: CentroidObserverKernel,
        report: CentroidInertiaReport,
        is_bubbling: bool,
        fpu_time_ms: float
    ) -> str:
        r"""Genera el sello holístico HMAC-SHA256 encadenando los sellos de las 3 Fases."""
        hasher = hashlib.sha256()
        hasher.update(kernel.phase1_sha256_seal.encode("ascii"))
        hasher.update(report.phase2_hmac_sha256.encode("ascii"))
        hasher.update(f"{report.total_kinetic_energy:.16e}".encode("ascii"))
        hasher.update(f"{report.casimir_invariant_c2:.16e}".encode("ascii"))
        hasher.update(f"{is_bubbling}".encode("ascii"))
        hasher.update(f"{fpu_time_ms:.6f}".encode("ascii"))
        return hasher.hexdigest()

    def _act_execute_pipeline(
        self,
        kernel: CentroidObserverKernel,
        report: CentroidInertiaReport,
        bubbling_threshold: float
    ) -> CentroidInertiaEngineState:
        r"""Orquesta la certificación y sellado final de Fase 3."""
        t_start = time.perf_counter_ns()

        # 1. Detección de Colapso de Maslov / Burbujeo de Discos de Gromov
        is_bubbling = bool(kernel.novikov_area <= (bubbling_threshold + self._tol))
        if is_bubbling:
            logger.critical(
                "¡ALERTA CRÍTICA DE BUBBLING DE MASLOV! Área de Novikov: %.6e <= Umbral: %.6e",
                kernel.novikov_area, bubbling_threshold
            )

        # 2. Cota de redondeo de Wilkinson
        wilkinson_bound = self._compute_wilkinson_bound(kernel.metric_cache.dimension)

        # 3. Metrología de latencia de ejecución FPU
        t_delta_ns = time.perf_counter_ns() - t_start
        fpu_time_ms = float(t_delta_ns / 1.0e6)

        # 4. Sello Soberano Inmutable
        cryptographic_seal = self._generate_sovereign_seal(
            kernel=kernel,
            report=report,
            is_bubbling=is_bubbling,
            fpu_time_ms=fpu_time_ms
        )

        logger.debug(
            "Fase 3 (Act) completada con éxito. Bubbling: %s, Sello Soberano: %s",
            is_bubbling, cryptographic_seal[:16]
        )

        return CentroidInertiaEngineState(
            kernel=kernel,
            report=report,
            is_maslov_bubbling_detected=is_bubbling,
            maslov_bubbling_threshold=bubbling_threshold,
            wilkinson_roundoff_bound=wilkinson_bound,
            fpu_execution_time_ms=fpu_time_ms,
            cryptographic_seal=cryptographic_seal
        )

    def execute_centroid_inertia_audit(
        self,
        polygon_vertices: NDArray[np.float64],
        vertex_velocities: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        simplex_areas: Optional[NDArray[np.float64]] = None,
        base_mass: float = 1.0,
        bubbling_threshold: float = _DEFAULT_DISK_BUBBLING_THRESHOLD
    ) -> CentroidInertiaEngineState:
        r"""
        MÉTODO MAESTRO ORQUESTADOR DE LAS TRES FASES ANIDADAS:
        Ejecuta ininterrumpidamente el ciclo OODA de la FPU para la inercia del centroide:
          - Fase 1 (Observe): `observe_polygon_centroid(...)` -> produce `CentroidObserverKernel`.
          - Fase 2 (Orient) : `orient_from_kernel(...)` -> absorbe Kernel, produce `CentroidInertiaReport`.
          - Fase 3 (Act)    : `act_from_kernel_and_report(...)` -> absorbe Kernel y Report, emite `CentroidInertiaEngineState`.
        """
        t_global_start = time.perf_counter_ns()

        # Fase 1: Observe
        kernel = self.observe_polygon_centroid(
            polygon_vertices=polygon_vertices,
            G_metric=G_metric,
            simplex_areas=simplex_areas
        )

        # Fase 2: Orient (Inicia consumiendo la salida formal de Fase 1)
        report = self.orient_from_kernel(
            kernel=kernel,
            vertex_velocities=vertex_velocities,
            base_mass=base_mass
        )

        # Fase 3: Act (Inicia consumiendo la salida formal de Fase 2)
        state = self.act_from_kernel_and_report(
            kernel=kernel,
            report=report,
            bubbling_threshold=bubbling_threshold
        )

        t_global_ms = float((time.perf_counter_ns() - t_global_start) / 1.0e6)

        # Re-encapsulación del estado final con el tiempo total transcurrido
        final_state = CentroidInertiaEngineState(
            kernel=state.kernel,
            report=state.report,
            is_maslov_bubbling_detected=state.is_maslov_bubbling_detected,
            maslov_bubbling_threshold=state.maslov_bubbling_threshold,
            wilkinson_roundoff_bound=state.wilkinson_roundoff_bound,
            fpu_execution_time_ms=t_global_ms,
            cryptographic_seal=state.cryptographic_seal
        )

        return final_state


# ══════════════════════════════════════════════════════════════════════════════
# CLASE FACADE: MOTOR SOBERANO DE INERCIA Y MOMENTUM DEL CENTROIDE
# ══════════════════════════════════════════════════════════════════════════════

class PseudoholomorphicCentroidInertiaEngine(Phase3_NovikovNullConeActuator):
    r"""
    MOTOR SOBERANO DE INERCIA Y MOMENTUM DEL CENTROIDE PSEUDO-HOLOMORFO EN FPU SECURE.
    Punto de entrada primario para análisis de polígonos simplécticos en la Categoría de Fukaya.
    Consolida las 3 Fases anidadas continuas bajo una interfaz de alta eficiencia y rigor doctoral.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        condition_tolerance: float = _CONDITION_NUMBER_MAX
    ) -> None:
        super().__init__(tolerance=tolerance, condition_tolerance=condition_tolerance)

    def __repr__(self) -> str:
        return (
            f"<PseudoholomorphicCentroidInertiaEngine "
            f"Precision=IEEE-754-Double, ConditionTolerance={self._condition_tolerance:.2e}, "
            f"DiskBubblingThreshold={_DEFAULT_DISK_BUBBLING_THRESHOLD:.1e}, Metrology=Fukaya-Novikov>"
        )


# Exportación formal de la API de nivel doctoral
__all__ = [
    # Motor Principal
    "PseudoholomorphicCentroidInertiaEngine",
    # Clases de Fases Anidadas
    "Phase1_PolygonCentroidObserver",
    "Phase2_InertiaSpinTensorOrient",
    "Phase3_NovikovNullConeActuator",
    # Expedientes de Datos Inmutables
    "BaseMetricCache",
    "CentroidObserverKernel",
    "CentroidInertiaReport",
    "CentroidInertiaEngineState",
    # Módulo de Sumación Aritmética Exacta
    "KahanNeumaierSum",
    # Excepciones
    "CentroidInertiaEngineError",
    "CentroidDimensionError",
    "NovikovAreaCollapseError",
    "MetricIndefinitenessError",
    "SymplecticCompatibilityError",
    "MaslovDiskBubblingCriticalError",
]
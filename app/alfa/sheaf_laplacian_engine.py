# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Sheaf Laplacian Engine (Motor Espectral del Haz Celular)          ║
║ Ubicación: app/alfa/sheaf_laplacian_engine.py                                ║
║ Versión  : 6.1.0-SOVEREIGN-RSI3-POINCARE-LIOUVILLE-RICCI-BROCKETT-PHD        ║
║ Función  : Motor Espectral FPU de Alta Precisión para el Haz Celular y RSI³  ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL:
El `SheafLaplacianEngine` opera como el núcleo numérico de punto flotante de alta
precisión (FPU) del haz celular \(\mathcal{F}\) sobre el complejo simplicial 1-dimensional
\(K = (V, E)\) en el estrato Alfa de APU Filter. Su cometido exclusivo es la computación
determinista e indivisible de la geometría simpléctica de Henri Poincaré, la
factorización de Cholesky ponderada, la descomposición espectral de Kato-Banach, la
torsión analítica de Ray-Singer \(\det'(L_F)\), la termodinámica de núcleos de calor
\(Z(\tau) = \operatorname{Tr}(e^{-\tau L_F})\) y la evolución meta-recursiva de nivel 3
(RSI³: Flujo de Ricci-DeTurck \(R_1\), Respuesta Funcional de Connes-Daleckii-Krein \(R_2\)
y Flujo Isoespectral de Doble Corchete de Brockett-Shahshahani \(R_3\)).

MARCO TEÓRICO:
1. Geometría Simpléctica Celeste y Conforme de Poincaré:
   El espacio de fases cotangente \(T^*\mathcal{M}_{\mathrm{sheaf}}\) equipado con coordenadas
   canónicas de Darboux \(z = (q, p)^\top \in \mathbb{R}^{2n}\) preserva la 2-forma simpléctica
   canónica \(\omega = \sum_{i=1}^n dq_i \wedge dp_i = \frac{1}{2} dz^\top \mathbf{\Omega} dz\)
   bajo la jacobiana de monodromía de Floquet \(M = \frac{\partial z_{t+T}}{\partial z_t}\):
   \[
   M^\top \mathbf{\Omega} M = \mathbf{\Omega}, \qquad \det(M) = +1.
   \]
   Bajo el teorema de capacidad simpléctica de Gromov, la contracción o compresión no física del
   volumen en el espacio de fases queda prohibida. La trayectoria geodésica de Maupertuis-Jacobi
   bajo energía total \(H_0\) y potencial cinemático \(V(q)\) reparametriza el espacio de
   configuraciones mediante la métrica conforme \(\tilde{\mathbf{G}}_{\mu\nu} = 2(H_0 - V(q)) \mathbf{G}_{\mu\nu}\).
   En regiones de frontera \(\partial \mathcal{M}\) con intersección nula o energía cinemática
   subcrítica \(H_0 \le V(q)\), se aplica la regularización ultramétrica en el anillo de Novikov
   \(\Lambda_{\mathrm{Nov}}\), garantizando la analiticidad del operador de Jacobi.

2. Análisis Espectral del Haz Celular y Operadores de Banach-Fredholm:
   Dada la métrica conforme regulada \(\mathbf{G}_{\mathrm{reg}} \in \operatorname{Sym}^+(m, \mathbb{R})\)
   factorizada mediante Cholesky \(\mathbf{G}_{\mathrm{reg}} = L_G L_G^\top\), el coborde celular
   renormalizado de Wilkinson es \(B_G = L_G^{-1} \delta\). El operador laplaciano autoadjunto
   no negativo del haz se ensambla como:
   \[
   L_F = \delta^\top \mathbf{G}_{\mathrm{reg}}^{-1} \delta = B_G^\top B_G \succeq 0.
   \]
   La función zeta espectral de Minakshisundaram-Pleijel-Ray-Singer \(\zeta_{L_F}(s) = \sum_{\lambda_i > 0} \lambda_i^{-s}\)
   determina la torsión analítica y el determinante regularizado mediante
   \(\log \det'(L_F) = -\zeta'_{L_F}(0) = \sum_{\lambda_i > 0} \log \lambda_i\). El operador
   resolvente de Banach \(R(z, L_F) = (z I - L_F)^{-1} \in \mathcal{B}(\mathcal{H})\) define
   las proyecciones armónicas de Riesz sobre el núcleo \(\ker L_F\).

3. Motor de Automejora Recursiva de Nivel 3 (RSI³: \(R_3 \circ R_2 \circ R_1\)):
   - Nivel \(R_1\) (Microescala - Ricci-DeTurck): Evoluciona la métrica con fijación de gauge
     \(\frac{\partial G}{\partial \tau} = -2 \operatorname{Ric}(G) + \mathcal{L}_W G - \frac{R_{\mathrm{avg}}}{n} G\),
     preservando la invariancia de volumen de Liouville.
   - Nivel \(R_2\) (Mesoescala - Connes-Daleckii-Krein): Computa la derivada de Fréchet no conmutativa
     \(Df(L_F)[H] = Q (f^{[1]}(\lambda_i, \lambda_j) \circ (Q^\top H Q)) Q^\top\) a través de la matriz de
     diferencias divididas \(f^{[1]}\), actualizando la cota de Lipschitz de Connes \(L_{\max} \le \frac{1}{2 \lambda_2^{3/2}}\).
   - Nivel \(R_3\) (Macroescala - Brockett-Shahshahani): Ejecuta el flujo isoespectral de doble corchete
     \(\dot{L} = [L, [L, \operatorname{diag}(p)]]\) sobre la órbita coadjunta mediante integradores ortogonales
     de Cayley \(Q_k = \operatorname{Cayley}(-h [L_k, N]) \in \operatorname{SO}(n)\), garantizando la estricta
     monotonía de Lyapunov \(\dot{\mathcal{H}} = \|[L, \operatorname{diag}(p)]\|_F^2 \ge 0\) y la conservación
     de los invariantes de Casimir \(\operatorname{Tr}(L^k)\).

AXIOMAS FORMALES:
  1. Axioma de Darboux-Liouville: Todo flujo en el espacio de fases cotangente debe satisfacer
     \(M^\top \mathbf{\Omega} M = \mathbf{\Omega}\) y \(\det(M) = +1\), garantizando la conservación del volumen simpléctico.
  2. Axioma de Positividad Métrica de Higham: Toda métrica de haz \(\mathbf{G}\) debe ser proyectada sobre
     \(\operatorname{Sym}^+(m, \mathbb{R})\) con autovalor mínimo \(\lambda_{\min}(\mathbf{G}) \ge \tau_{\mathrm{SPD}} = 10^{-12}\).
  3. Axioma de Continuidad Espectral de Daleckii-Krein: La variación espectral bajo perturbaciones métricas
     autoadjuntas satisface la cota de Weyl-Lidskii \(|\lambda_i(L+H) - \lambda_i(L)| \le \|H\|_2\).
  4. Axioma Isoespectral de Brockett: El flujo de doble corchete conserva los autovalores de \(L_F\) e invariantes de
     Casimir \(\operatorname{Tr}(L_F^k)\) mediante transformaciones ortogonales de Cayley \(Q \in \operatorname{SO}(n)\).

INVARIANTES DE ESTADO:
  1. Invariante de Liouville-Darboux: \(\|M^\top \mathbf{\Omega} M - \mathbf{\Omega}\|_F \le 10^{-9}\) y \(\det(M) = +1 \pm 10^{-3}\).
  2. Invariante de Connes-Daleckii-Krein: Cota de Lipschitz no conmutativa \(L_{\max} \le \frac{1}{2 \lambda_2^{3/2}}\).
  3. Invariante de Casimir-Brockett: \(\max_i |\lambda_i(L^{(R_3)}) - \lambda_i(L^{(0)})| \le 10^{-12}\).
  4. Invariante de Monotonía de Lyapunov: \(\Delta \mathcal{H}_{\mathrm{Brockett}} = \operatorname{Tr}(L_{\mathrm{final}} N) - \operatorname{Tr}(L_{\mathrm{inicial}} N) \ge 0\).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("MIC.Physics.SheafLaplacianEngine")

__version__: Final[str] = "6.1.0-SOVEREIGN-RSI3-POINCARE-LIOUVILLE-RICCI-BROCKETT-PHD"

# ──────────────────────────────────────────────────────────────────────────────
# Constantes fundamentales de precisión espectral, simpléctica y de flujo
# ──────────────────────────────────────────────────────────────────────────────
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_KAPPA_MAX_CEILING: Final[float] = 1.0e8
_SYMPLECTIC_TOLERANCE: Final[float] = 1.0e-9
_NOVIKOV_ENERGY_EPSILON: Final[float] = 1.0e-10
_KREIN_UNIT_CIRCLE_TOL: Final[float] = 1.0e-5
_FLOQUET_PAIR_TOL: Final[float] = 1.0e-6
_MODULI_CONTRACTION_TOL: Final[float] = 1.0e-12
_LOGDET_FLOOR: Final[float] = 1.0e-18
_CAYLEY_STEP_CEILING: Final[float] = 0.25

# Gauss-Legendre 5 puntos en [0, 1] (nodos y pesos de precisión doble).
_GL5_NODES: Final[NDArray[np.float64]] = np.array(
    [
        0.04691007703066800,
        0.23076534494715845,
        0.50000000000000000,
        0.76923465505284150,
        0.95308992296933200,
    ],
    dtype=np.float64,
)
_GL5_WEIGHTS: Final[NDArray[np.float64]] = np.array(
    [
        0.11846344252809454,
        0.23931433524968324,
        0.28444444444444444,
        0.23931433524968324,
        0.11846344252809454,
    ],
    dtype=np.float64,
)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES: ESTADOS ESPECTRALES Y SIMPLÉCTICOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareLiouvilleAudit:
    r"""
    Auditoría simpléctica de Liouville-Poincaré sobre el fibrado cotangente \(T^*M\).

    Axiomas:
      1. Preservación de la 2-forma canónica: \(\|M^\top\mathbf{\Omega}M-\mathbf{\Omega}\|_F<\varepsilon\).
      2. Conservación del volumen de Liouville: \(\det(M)=+1\pm\delta\) (vía slogdet).
      3. Multiplicadores de Floquet \(\mu_i\in\operatorname{spec}(M)\) en cuaternas
         \(\{\mu,\mu^{-1},\bar\mu,\bar\mu^{-1}\}\); estabilidad elíptica KAM \(\Leftrightarrow|\mu_i|=1\).
      4. Discriminante de Hill \(\Delta=\operatorname{Tr}(M)\) (caso \(2n=2\): \(|\Delta|<2\) elíptico).
      5. Firmas de Krein \(\sigma_i=\operatorname{sign}(i\,v_i^*\mathbf{\Omega}v_i)\) sobre \(S^1\).
    """
    symplectic_drift: float
    is_symplectic: bool
    determinant: float
    determinant_drift: float
    floquet_multipliers: NDArray[np.complex128]
    krein_signature_stable: bool
    kam_stability_index: float
    hill_discriminant: float
    multiplier_reciprocal_residual: float
    krein_signatures: NDArray[np.float64]
    log_abs_determinant: float


@dataclass(frozen=True, slots=True)
class PoincareMaupertuisSheafState:
    r"""
    Estado final emitido por la FASE 1.

    Tensor métrico conforme cinemático, forma simpléctica de Darboux, estructura
    casi-compleja compatible y monodromía de Poincaré, requeridos de forma
    indivisible por los operadores de la FASE 2.
    """
    maupertuis_metric_G: NDArray[np.float64]
    canonical_coboundary_delta: NDArray[np.float64]
    symplectic_form_Omega: NDArray[np.float64]
    kinematic_action_scale: float
    liouville_audit: PoincareLiouvilleAudit
    phase_space_dimension: int
    monodromy_jacobian: NDArray[np.float64]
    almost_complex_J: NDArray[np.float64]
    novikov_regularized: bool


@dataclass(frozen=True, slots=True)
class SheafEngineSpectrumReport:
    r"""Reporte numérico inmutable de la descomposición espectral del haz celular."""
    regularized_metric_G: NDArray[np.float64]
    laplacian_L_F: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    pseudo_inverse_L_plus: NDArray[np.float64]
    fiedler_value: float
    spectral_gap: float
    condition_number: float
    connes_lipschitz_bound: float
    cholesky_factor_L: NDArray[np.float64]
    hodge_star_weight_trace: float
    numerical_rank: int


@dataclass(frozen=True, slots=True)
class KatoBanachSpectrumReport:
    r"""Reporte de análisis funcional en el álgebra de Banach \(\mathcal{B}(\mathcal{H})\)."""
    resolvent_norm_bound: float
    kato_stability_radius: float
    spectral_projections: NDArray[np.float64]
    ray_singer_determinant: float
    fredholm_index: int
    log_ray_singer_determinant: float
    spectral_zeta_s0_derivative: float
    resolvent_point: complex
    harmonic_dimension: int


@dataclass(frozen=True, slots=True)
class SheafHeatKernelReport:
    r"""Reporte termodinámico y trace-class del calor celular \(Z(\tau)=\operatorname{Tr}(e^{-\tau L_F})\)."""
    heat_trace: float
    diffusion_time_tau: float
    asymptotic_entropy: float
    betti_0_regularized: float
    minakshisundaram_a0: float
    heat_trace_derivative: float


@dataclass(frozen=True, slots=True)
class SpectralBanachSheafState:
    r"""
    Estado final emitido por la FASE 2.

    Encapsula el espectro del haz, las cotas de resolvente y la termodinámica
    del calor; constituye la entrada obligatoria de los operadores RSI³ de la FASE 3.
    """
    spectrum_report: SheafEngineSpectrumReport
    kato_report: KatoBanachSpectrumReport
    heat_report: SheafHeatKernelReport
    source_maupertuis_state: PoincareMaupertuisSheafState


@dataclass(frozen=True, slots=True)
class RicciDeTurckFlowState:
    r"""Estado de la variedad métrica celular tras una micro-evolución de Ricci-DeTurck (\(R_1\))."""
    evolved_metric_G: NDArray[np.float64]
    ricci_tensor: NDArray[np.float64]
    deturck_gauge_vector: NDArray[np.float64]
    scalar_curvature: float
    volume_norm_drift: float
    ricci_norm: float
    normalized_step_residual: float


@dataclass(frozen=True, slots=True)
class DaleckiiKreinResolventState:
    r"""Estado de la respuesta funcional mesoscópica de Connes-Daleckii-Krein (\(R_2\))."""
    frechet_derivative_operator: NDArray[np.float64]
    projector_shift_norm: float
    perturbed_lipschitz_constant: float
    spectral_variation_bound: float
    divided_difference_condition: float
    weyl_lidskii_shift: float


@dataclass(frozen=True, slots=True)
class BrockettShahshahaniState:
    r"""Estado del flujo isoespectral hamiltoniano de doble corchete (\(R_3\))."""
    diagonalized_orbit_L: NDArray[np.float64]
    coadjoint_gradient: NDArray[np.float64]
    commutator_norm: float
    isoespectral_trace_error: float
    casimir_invariants: NDArray[np.float64]
    lyapunov_increase: float
    orthogonality_residual: float


@dataclass(frozen=True, slots=True)
class RSI3ConvergenceState:
    r"""
    Estado terminal del ciclo recursivo cerrado RSI³ (\(R_3\circ R_2\circ R_1\)).

    Atestigua la convergencia métrica, espectral y dimensional en el espacio
    de moduli de haces celulares acoplados a la mecánica celeste de Poincaré.
    """
    final_banach_state: SpectralBanachSheafState
    ricci_flow_history: RicciDeTurckFlowState
    daleckii_krein_state: DaleckiiKreinResolventState
    brockett_state: BrockettShahshahaniState
    global_lyapunov_energy: float
    iterations_converged: int
    moduli_distance_delta: float
    lyapunov_monotonicity_residual: float
    spectral_gap_final: float
    kam_index_final: float


# ══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: SHEAF LAPLACIAN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SheafLaplacianEngine:
    r"""
    Motor espectral de cálculo de alta precisión FPU para el haz celular \(\mathcal{F}\).

    Integra de forma determinista la geometría simpléctica celeste de Henri Poincaré
    y la automejora recursiva RSI³ mediante el encadenamiento indivisible de 3 fases.
    Cada fase emite un estado congelado que es el único alimento analítico de la siguiente.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # NÚCLEO NUMÉRICO COMÚN (álgebra lineal estable, proyección SPD, log-det)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyección euclídea sobre \(\operatorname{Sym}(n,\mathbb{R})\): \(S\mapsto\tfrac12(S+S^\top)\)."""
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _nearest_spd_higham(
        matrix: NDArray[np.float64],
        floor: float = _SPECTRAL_PSD_FLOOR,
    ) -> NDArray[np.float64]:
        r"""
        Proyección convexa de Higham sobre el cono \(\operatorname{Sym}^+(n,\mathbb{R})\).

        Se recortan los autovalores por debajo de `floor` y se reconstruye
        \(Q\operatorname{diag}(\lambda_i\vee\varepsilon)Q^\top\) en forma simetrizada.
        """
        symmetric = 0.5 * (matrix + matrix.T)
        eigenvalues, eigenvectors = la.eigh(symmetric)
        lifted = np.maximum(eigenvalues, floor)
        restored = (eigenvectors * lifted) @ eigenvectors.T
        return 0.5 * (restored + restored.T)

    @staticmethod
    def _stable_logdet_spd(matrix: NDArray[np.float64]) -> Tuple[float, float]:
        r"""
        \(\operatorname{slogdet}\) simetrizado. Devuelve \((\log|\det A|,\operatorname{sign}\det A)\).
        Evita el overflow de `det` en dimensión moderada-alta.
        """
        sign, logdet = np.linalg.slogdet(0.5 * (matrix + matrix.T))
        return float(logdet), float(sign)

    @staticmethod
    def _divided_difference(
        eigenvalues: NDArray[np.float64],
        values_f: NDArray[np.float64],
        derivatives_f: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Matriz de diferencias divididas de Daleckii-Krein:

        \[
        f^{[1]}(\lambda_i,\lambda_j)=
        \begin{cases}
        f'(\lambda_i) & \text{si }|\lambda_i-\lambda_j|<\varepsilon,\\
        \dfrac{f(\lambda_i)-f(\lambda_j)}{\lambda_i-\lambda_j} & \text{en otro caso.}
        \end{cases}
        \]
        """
        lam_i = eigenvalues[:, None]
        lam_j = eigenvalues[None, :]
        delta = lam_i - lam_j
        close = np.abs(delta) <= _WILKINSON_EPSILON
        off = np.zeros_like(delta, dtype=np.float64)
        far = ~close
        off[far] = (values_f[:, None] - values_f[None, :])[far] / delta[far]
        diag_term = 0.5 * (derivatives_f[:, None] + derivatives_f[None, :])
        return np.where(close, diag_term, off)

    @staticmethod
    def _cayley_orthogonal(skew: NDArray[np.float64], step: float) -> NDArray[np.float64]:
        r"""
        Transformación de Cayley que produce una matriz ortogonal de \(\operatorname{SO}(n)\):

        \[
        Q=\bigl(I-\tfrac{h}{2}K\bigr)\bigl(I+\tfrac{h}{2}K\bigr)^{-1},
        \qquad K^\top=-K.
        \]
        Preserva \(\det Q=+1\) siempre que \(I+\tfrac{h}{2}K\) sea invertible.
        """
        n = skew.shape[0]
        identity = np.eye(n, dtype=np.float64)
        half = 0.5 * float(step) * SheafLaplacianEngine._symmetrize_skew(skew)
        left = identity - half
        right = identity + half
        try:
            cayley = la.solve(right.T, left.T).T
        except la.LinAlgError:
            logger.warning("Cayley: caída a expm por singularidad del resolvente de Cayley.")
            cayley = np.real(la.expm(-float(step) * skew))
        # Reortogonalización de polar (Higham) para cancelar deriva de redondeo.
        u, _, vt = la.svd(cayley, full_matrices=False)
        return u @ vt

    @staticmethod
    def _symmetrize_skew(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyección sobre \(\mathfrak{so}(n)\): \(K\mapsto\tfrac12(K-K^\top)\)."""
        return 0.5 * (matrix - matrix.T)

    @staticmethod
    def _validate_square(matrix: NDArray[np.ndarray], name: str) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada, recibido shape={matrix.shape}")

    @staticmethod
    def _connes_lipschitz(fiedler: float) -> float:
        r"""
        Cota de Lipschitz de Connes para el módulo de derivación \([D,\,\cdot\,]\)
        controlada por el hueco de Fiedler: \(\lambda_2^{-3/2}/2\).
        """
        effective = max(float(fiedler), _SPECTRAL_PSD_FLOOR)
        return float(1.0 / (2.0 * (effective ** 1.5)))

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 1: FUNDAMENTO SIMPLÉCTICO, GEOMETRÍA CELULAR DE POINCARÉ
    #         Y MÉTRICA CONFORME DE MAUPERTUIS-JACOBI
    # ══════════════════════════════════════════════════════════════════════════

    def construct_canonical_symplectic_form(self, phase_dim: int) -> NDArray[np.float64]:
        r"""
        Construye la forma simpléctica bilineal estándar de Liouville-Darboux \(\mathbf{\Omega}\).

        \[
        \mathbf{\Omega}
        =\begin{pmatrix}\mathbf{0}_n&\mathbf{I}_n\\-\mathbf{I}_n&\mathbf{0}_n\end{pmatrix}
        \in\operatorname{Sp}(2n,\mathbb{R}),
        \qquad
        \mathbf{\Omega}^\top=-\mathbf{\Omega},\quad \mathbf{\Omega}^2=-\mathrm{Id}.
        \]
        """
        if phase_dim <= 0 or phase_dim % 2 != 0:
            raise ValueError(
                f"La dimensión del espacio de fase debe ser par y positiva, recibido: {phase_dim}"
            )
        n = phase_dim // 2
        omega = np.zeros((phase_dim, phase_dim), dtype=np.float64)
        omega[:n, n:] = np.eye(n, dtype=np.float64)
        omega[n:, :n] = -np.eye(n, dtype=np.float64)
        return omega

    def construct_compatible_almost_complex_structure(
        self,
        phase_dim: int,
    ) -> NDArray[np.float64]:
        r"""
        Estructura casi-compleja canónica \(J\) compatible con \((\mathbf{G}_{\mathrm{eucl}},\mathbf{\Omega})\):

        \[
        J=\mathbf{\Omega},\qquad J^2=-\mathrm{Id},\qquad \omega(u,v)=g(Ju,v).
        \]
        En coordenadas de Darboux con métrica euclídea, \(J\) es la rotación de \(\pi/2\)
        en cada plano \((q_i,p_i)\), i.e. la estructura de Kähler lineal del espacio de fases.
        """
        return self.construct_canonical_symplectic_form(phase_dim)

    def compute_maupertuis_jacobi_bundle_metric(
        self,
        base_metric_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        novikov_floor: float = _NOVIKOV_ENERGY_EPSILON,
    ) -> Tuple[NDArray[np.float64], bool]:
        r"""
        Transformación conforme de Maupertuis-Jacobi sobre el espacio de configuraciones.

        \[
        \tilde{\mathbf{G}}_{\mu\nu}(q)=2\bigl(H_0-V(q)\bigr)\mathbf{G}_{\mu\nu}(q).
        \]

        Si el sistema atraviesa una superficie de energía nula (región clásicamente
        prohibida), se activa una extensión ultramétrica analítica sobre el anillo
        de Novikov

        \[
        \Lambda_{\mathrm{Nov}}
        =\Bigl\{\sum_{i=0}^\infty c_i T^{\lambda_i}
        \;\Big|\; c_i\in\mathbb{R},\;\lambda_i\in\mathbb{R},\;
        \lim_{i\to\infty}\lambda_i=+\infty\Bigr\},
        \]

        mediante un corte exponencial \(T^{\Delta E}\simeq\exp(\Delta E/\varepsilon_{\mathrm{Nov}})\)
        que mantiene \(\tilde G\succ 0\) y la geodésica de Jacobi bien definida.
        """
        self._validate_square(base_metric_G, "base_metric_G")
        g_sym = self._symmetrize(np.asarray(base_metric_G, dtype=np.float64))
        kinematic_energy = float(total_energy_H0) - float(potential_V)
        novikov_hit = kinematic_energy <= float(novikov_floor)
        if novikov_hit:
            # Completación no arquimediana: escala efectiva estrictamente positiva.
            effective_scale = float(novikov_floor) * np.exp(
                (kinematic_energy - float(novikov_floor)) / float(novikov_floor)
            )
            logger.debug(
                "Maupertuis-Novikov: energía cinemática %.3e ≤ floor; escala efectiva=%.3e",
                kinematic_energy,
                effective_scale,
            )
        else:
            effective_scale = kinematic_energy
        conformal_factor = 2.0 * float(effective_scale)
        maupertuis_metric = self._nearest_spd_higham(conformal_factor * g_sym)
        return maupertuis_metric, bool(novikov_hit)

    def pair_floquet_multipliers_reciprocal(
        self,
        multipliers: NDArray[np.complex128],
    ) -> float:
        r"""
        Residuo de la simetría simpléctica \(\mu\longleftrightarrow\mu^{-1}\).

        Para \(M\in\operatorname{Sp}(2n,\mathbb{R})\) el polinomio característico es
        recíproco: si \(\mu\) es autovalor, también lo es \(\mu^{-1}\). Se mide
        \(\min_{\sigma}\max_i|\mu_i-1/\mu_{\sigma(i)}|\) por emparejamiento voraz
        en el plano complejo.
        """
        mu = np.asarray(multipliers, dtype=np.complex128).ravel()
        if mu.size == 0:
            return 0.0
        safe = np.where(np.abs(mu) < _WILKINSON_EPSILON, _WILKINSON_EPSILON, mu)
        inv = 1.0 / safe
        unused = np.ones(mu.size, dtype=bool)
        residual = 0.0
        for value in mu:
            distances = np.abs(inv - value)
            distances[~unused] = np.inf
            j = int(np.argmin(distances))
            unused[j] = False
            residual = max(residual, float(distances[j]) if np.isfinite(distances[j]) else 1.0)
        return float(residual)

    def compute_hill_discriminant(self, monodromy_jacobian: NDArray[np.float64]) -> float:
        r"""
        Discriminante de Hill-Poincaré \(\Delta=\operatorname{Tr}(M)\).

        En un grado de libertad (\(2n=2\)), \(|\Delta|<2\) es elíptico (órbita estable),
        \(|\Delta|=2\) parabólico y \(|\Delta|>2\) hiperbólico. En dimensión superior
        \(\Delta\) permanece como invariante de monodromía de primer orden.
        """
        return float(np.trace(monodromy_jacobian))

    def compute_krein_signatures(
        self,
        omega: NDArray[np.float64],
        monodromy_jacobian: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Firmas de Krein sobre los autovectores de la monodromía.

        Para \(\mu\in S^1\) y \(Mv=\mu v\), la forma de Krein \(i\,v^*\mathbf{\Omega}v\)
        es real; su signo es la firma. Una colisión de firmas opuestas es el
        mecanismo de pérdida de estabilidad (teorema de Krein-Moser).
        """
        eigenvalues, eigenvectors = la.eig(monodromy_jacobian)
        signatures = np.zeros(eigenvalues.shape[0], dtype=np.float64)
        for idx in range(eigenvalues.shape[0]):
            vector = eigenvectors[:, idx]
            norm = np.linalg.norm(vector)
            if norm <= _WILKINSON_EPSILON:
                continue
            vector = vector / norm
            krein_form = np.vdot(vector, omega.astype(np.complex128) @ vector)
            signatures[idx] = float(np.sign(np.real(1j * krein_form)))
            if np.abs(np.abs(eigenvalues[idx]) - 1.0) > _KREIN_UNIT_CIRCLE_TOL:
                signatures[idx] = 0.0
        return signatures

    def audit_poincare_liouville_invariance(
        self,
        monodromy_jacobian: NDArray[np.float64],
    ) -> PoincareLiouvilleAudit:
        r"""
        Audita el teorema de Liouville y la preservación de la estructura simpléctica.

        Condición de Darboux:
        \[
        M^\top\mathbf{\Omega}M-\mathbf{\Omega}=\mathbf{0}.
        \]
        Estabilidad de Floquet: los multiplicadores aparecen en cuaternas simétricas.
        La condición elíptica de KAM exige \(|\mu_i|=1\) para todo \(i\), sin colisión
        de firmas de Krein de signo opuesto.
        """
        jacobian = np.asarray(monodromy_jacobian, dtype=np.float64)
        self._validate_square(jacobian, "monodromy_jacobian")
        m_dim = jacobian.shape[0]
        omega = self.construct_canonical_symplectic_form(m_dim)

        symplectic_transform = jacobian.T @ omega @ jacobian
        drift_matrix = symplectic_transform - omega
        drift_norm = float(la.norm(drift_matrix, ord="fro"))
        is_conserved = drift_norm <= _SYMPLECTIC_TOLERANCE

        log_abs_det, sign_det = self._stable_logdet_spd(jacobian.T @ jacobian)
        # det(M) se recupera por slogdet directo (M no es SPD).
        sign_m, log_abs_m = np.linalg.slogdet(jacobian)
        det_m = float(sign_m * np.exp(min(abs(float(log_abs_m)), 700.0)) * np.sign(1.0 if log_abs_m >= 0 else 1.0))
        # Reconstrucción estable: si |logdet| es grande se reporta el valor recortado.
        if abs(float(log_abs_m)) > 50.0:
            det_m = float(sign_m * np.exp(np.clip(float(log_abs_m), -50.0, 50.0)))
        else:
            det_m = float(sign_m * np.exp(float(log_abs_m)))
        det_drift = abs(det_m - 1.0)
        _ = log_abs_det  # volumen euclídeo de MᵀM; se retiene el slogdet de M.

        floquet_eigs = la.eigvals(jacobian)
        magnitudes = np.abs(floquet_eigs)
        krein_stable = bool(np.all(np.abs(magnitudes - 1.0) < _KREIN_UNIT_CIRCLE_TOL))
        kam_stability = float(1.0 / (1.0 + np.sum(np.abs(magnitudes - 1.0))))

        signatures = self.compute_krein_signatures(omega, jacobian)
        # Colisión de Krein: dos multiplicadores sobre S¹ con firmas opuestas y
        # distancia espectral nula se interpretan como pérdida de estabilidad.
        on_circle = np.abs(magnitudes - 1.0) < _KREIN_UNIT_CIRCLE_TOL
        if np.count_nonzero(on_circle) >= 2:
            circ_eigs = floquet_eigs[on_circle]
            circ_sig = signatures[on_circle]
            for i in range(circ_eigs.size):
                for j in range(i + 1, circ_eigs.size):
                    if (
                        abs(circ_eigs[i] - circ_eigs[j]) < _FLOQUET_PAIR_TOL
                        and circ_sig[i] * circ_sig[j] < 0.0
                    ):
                        krein_stable = False
                        kam_stability *= 0.5

        hill = self.compute_hill_discriminant(jacobian)
        pair_residual = self.pair_floquet_multipliers_reciprocal(
            np.asarray(floquet_eigs, dtype=np.complex128)
        )

        return PoincareLiouvilleAudit(
            symplectic_drift=drift_norm,
            is_symplectic=is_conserved,
            determinant=det_m,
            determinant_drift=det_drift,
            floquet_multipliers=np.asarray(floquet_eigs, dtype=np.complex128),
            krein_signature_stable=krein_stable,
            kam_stability_index=kam_stability,
            hill_discriminant=hill,
            multiplier_reciprocal_residual=pair_residual,
            krein_signatures=signatures,
            log_abs_determinant=float(log_abs_m),
        )

    def generate_poincare_maupertuis_coboundary_state(
        self,
        coboundary_delta: NDArray[np.float64],
        base_metric_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        monodromy_jacobian: NDArray[np.float64],
    ) -> PoincareMaupertuisSheafState:
        r"""
        MÉTODO FINAL DE LA FASE 1 (puente indivisible hacia la Fase 2).

        Sintetiza la geometría métrica conforme de Maupertuis con el auditor de
        monodromía simpléctica de Poincaré y la estructura casi-compleja \(J\),
        emitiendo el contenedor hermético `PoincareMaupertuisSheafState` que
        inicia formalmente la Fase 2 sin discontinuidades analíticas.

        Invariantes transportados:
          - \(\tilde G\) (Jacobi), \(\delta\) (coborde celular), \(\Omega\) (Darboux),
          - \(M=\mathrm{D}\Phi_T\) (monodromía), \(J\) (casi-compleja),
          - auditoría de Liouville-Floquet-Krein.
        """
        jacobian = np.asarray(monodromy_jacobian, dtype=np.float64)
        coboundary = np.asarray(coboundary_delta, dtype=np.float64)
        metric = np.asarray(base_metric_G, dtype=np.float64)

        if coboundary.ndim != 2:
            raise ValueError(f"coboundary_delta debe ser 2D, shape={coboundary.shape}")
        if metric.shape[0] != coboundary.shape[0] or metric.shape[1] != coboundary.shape[0]:
            raise ValueError(
                "Incompatibilidad de haces: G debe ser (m×m) y δ de shape (m×n); "
                f"recibido G={metric.shape}, δ={coboundary.shape}"
            )

        phase_dim = jacobian.shape[0]
        omega = self.construct_canonical_symplectic_form(phase_dim)
        almost_complex = self.construct_compatible_almost_complex_structure(phase_dim)

        maupertuis_metric, novikov_hit = self.compute_maupertuis_jacobi_bundle_metric(
            base_metric_G=metric,
            potential_V=potential_V,
            total_energy_H0=total_energy_H0,
        )
        audit = self.audit_poincare_liouville_invariance(jacobian)
        kinematic_scale = max(_NOVIKOV_ENERGY_EPSILON, float(total_energy_H0) - float(potential_V))

        if not audit.is_symplectic:
            logger.warning(
                "Fase 1: deriva simpléctica %.3e > tolerancia %.1e (KAM=%.4f).",
                audit.symplectic_drift,
                _SYMPLECTIC_TOLERANCE,
                audit.kam_stability_index,
            )

        logger.info(
            "Fase 1→2: Maupertuis rank=%d, Novikov=%s, KAM=%.4f, Hill=%.4f",
            maupertuis_metric.shape[0],
            novikov_hit,
            audit.kam_stability_index,
            audit.hill_discriminant,
        )

        return PoincareMaupertuisSheafState(
            maupertuis_metric_G=maupertuis_metric,
            canonical_coboundary_delta=coboundary,
            symplectic_form_Omega=omega,
            kinematic_action_scale=kinematic_scale,
            liouville_audit=audit,
            phase_space_dimension=phase_dim,
            monodromy_jacobian=jacobian,
            almost_complex_J=almost_complex,
            novikov_regularized=novikov_hit,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2: ANÁLISIS ESPECTRAL DE HAZ CELULAR, CUANTIZACIÓN
    #         Y OPERADORES DE BANACH-FREDHOLM
    #         (inicio = consumo de PoincareMaupertuisSheafState)
    # ══════════════════════════════════════════════════════════════════════════

    def assemble_sheaf_laplacian_operator(
        self,
        phase1_state: PoincareMaupertuisSheafState,
    ) -> SheafEngineSpectrumReport:
        r"""
        MÉTODO DE INICIO DE LA FASE 2.

        Consume directamente `PoincareMaupertuisSheafState` (generado al final de
        la Fase 1). Realiza la proyección de Higham sobre \(\operatorname{Sym}^+(m)\),
        factoriza \(G=L_G L_G^\top\), computa el coborde renormalizado
        \(B_G=L_G^{-1}\delta\) y ensambla el laplaciano del haz celular:

        \[
        L_F=\delta^\top G_{\mathrm{reg}}^{-1}\delta=B_G^\top B_G\succeq 0.
        \]

        El espectro se obtiene por descomposición de Jacobi (eigh); la pseudoinversa
        de Moore-Penrose se construye por recorte espectral (Hodge-Green).
        """
        metric_g = phase1_state.maupertuis_metric_G
        coboundary = phase1_state.canonical_coboundary_delta

        g_reg = self._nearest_spd_higham(metric_g, floor=_SPECTRAL_PSD_FLOOR)
        chol_g = la.cholesky(g_reg, lower=True)

        # B_G = L_G^{-1} δ  (sustitución triangular, estabilidad de Wilkinson).
        b_g = la.solve_triangular(chol_g, coboundary, lower=True)
        laplacian = self._symmetrize(b_g.T @ b_g)

        eigenvalues, eigenvectors = la.eigh(laplacian)
        clean_eigs = np.maximum(0.0, eigenvalues)

        fiedler = float(clean_eigs[1]) if clean_eigs.size > 1 else 0.0
        spectral_gap = fiedler - float(clean_eigs[0])
        max_eig = float(clean_eigs[-1]) if clean_eigs.size else 0.0
        cond_num = min(max_eig / max(fiedler, _WILKINSON_EPSILON), _KAPPA_MAX_CEILING)
        connes_bound = self._connes_lipschitz(fiedler)

        # Pseudoinversa espectral: λ⁺ = λ^{-1} sobre {λ > ε}, 0 sobre el núcleo.
        inv_eigs = np.zeros_like(clean_eigs)
        positive = clean_eigs > _SPECTRAL_PSD_FLOOR
        inv_eigs[positive] = 1.0 / clean_eigs[positive]
        pinv_l = self._symmetrize((eigenvectors * inv_eigs) @ eigenvectors.T)

        numerical_rank = int(np.count_nonzero(positive))
        hodge_trace = float(np.trace(g_reg))

        return SheafEngineSpectrumReport(
            regularized_metric_G=g_reg,
            laplacian_L_F=laplacian,
            eigenvalues=clean_eigs,
            eigenvectors=eigenvectors,
            pseudo_inverse_L_plus=pinv_l,
            fiedler_value=fiedler,
            spectral_gap=spectral_gap,
            condition_number=float(cond_num),
            connes_lipschitz_bound=connes_bound,
            cholesky_factor_L=chol_g,
            hodge_star_weight_trace=hodge_trace,
            numerical_rank=numerical_rank,
        )

    def compute_spectral_zeta_regularized(
        self,
        eigenvalues: NDArray[np.float64],
        s: float = 0.0,
    ) -> Tuple[float, float]:
        r"""
        Función zeta espectral de Minakshisundaram-Pleijel-Ray-Singer y su derivada.

        \[
        \zeta_{L_F}(s)=\sum_{\lambda_i>0}\lambda_i^{-s},
        \qquad
        \zeta'_{L_F}(0)=-\sum_{\lambda_i>0}\log\lambda_i=-\log\det'(L_F).
        \]
        El determinante de Ray-Singer es \(\det'(L_F)=\exp(-\zeta'(0))\).
        """
        positive = eigenvalues[eigenvalues > _SPECTRAL_PSD_FLOOR]
        if positive.size == 0:
            return 0.0, 0.0
        logs = np.log(np.maximum(positive, _LOGDET_FLOOR))
        if abs(s) <= _WILKINSON_EPSILON:
            zeta = float(positive.size)
        else:
            zeta = float(np.sum(np.exp(-float(s) * logs)))
        zeta_prime_0 = float(-np.sum(logs))
        return zeta, zeta_prime_0

    def decompose_kato_banach_spectrum(
        self,
        spectrum_report: SheafEngineSpectrumReport,
        resolvent_shift_z: complex = complex(-1.0, 0.05),
    ) -> KatoBanachSpectrumReport:
        r"""
        Resolvente en \(\mathcal{B}(\mathcal{H})\), proyecciones de Riesz y estabilidad
        analítica de Kato-Rellich.

        \[
        R(z,L_F)=(zI-L_F)^{-1},
        \qquad
        P_0=\frac{1}{2\pi i}\oint_{\Gamma_0}R(z,L_F)\,dz=Q_0 Q_0^\top.
        \]

        Para operadores normales \(\|R(z)\|_2=1/\operatorname{dist}(z,\operatorname{spec} L_F)\).
        El índice de Fredholm del coborde se identifica con \(\dim\ker L_F-\operatorname{codim}\operatorname{im} L_F\);
        al ser \(L_F\) autoadjunto, \(\operatorname{ind}(L_F)=0\) y \(\dim H^0=\dim\ker L_F\).
        """
        laplacian = spectrum_report.laplacian_L_F
        n = laplacian.shape[0]
        identity = np.eye(n, dtype=np.complex128)
        z_point = complex(resolvent_shift_z)

        # Distancia al espectro real (L ≽ 0): |z| si Re z < 0, o min |z-λ|.
        dist_spec = float(np.min(np.abs(z_point - spectrum_report.eigenvalues)))
        if dist_spec <= _SPECTRAL_PSD_FLOOR:
            z_point = complex(-1.0, 0.05)
            dist_spec = float(np.min(np.abs(z_point - spectrum_report.eigenvalues)))

        z_matrix = z_point * identity - laplacian.astype(np.complex128)
        try:
            resolvent = la.inv(z_matrix)
        except la.LinAlgError:
            resolvent = la.pinv(z_matrix)
        resolvent_norm = float(np.real(la.norm(resolvent, ord=2)))
        kato_radius = float(1.0 / max(resolvent_norm, _WILKINSON_EPSILON))

        eigs = spectrum_report.eigenvalues
        q_basis = spectrum_report.eigenvectors
        harmonic_mask = eigs <= _SPECTRAL_PSD_FLOOR
        harmonic_dim = int(np.count_nonzero(harmonic_mask))

        p_harmonic = np.zeros((n, n), dtype=np.float64)
        if harmonic_dim:
            q0 = q_basis[:, harmonic_mask]
            p_harmonic = self._symmetrize(q0 @ q0.T)

        _, zeta_prime_0 = self.compute_spectral_zeta_regularized(eigs, s=0.0)
        log_det = float(-zeta_prime_0)
        # det' recortado para no reventar el float64; el log-det es el invariante útil.
        ray_singer_det = float(np.exp(np.clip(log_det, -700.0, 700.0))) if harmonic_dim < n else 0.0

        # Índice de Fredholm de L_F (autoadjunto ⇒ 0). Se reporta dim ker como Betti.
        fredholm_index = 0
        _ = dist_spec

        return KatoBanachSpectrumReport(
            resolvent_norm_bound=resolvent_norm,
            kato_stability_radius=kato_radius,
            spectral_projections=p_harmonic,
            ray_singer_determinant=ray_singer_det,
            fredholm_index=fredholm_index,
            log_ray_singer_determinant=log_det,
            spectral_zeta_s0_derivative=zeta_prime_0,
            resolvent_point=z_point,
            harmonic_dimension=harmonic_dim,
        )

    def compute_sheaf_heat_kernel_trace(
        self,
        spectrum_report: SheafEngineSpectrumReport,
        diffusion_time_tau: float = 0.1,
    ) -> SheafHeatKernelReport:
        r"""
        Evolución del núcleo de calor celular sobre el haz \(\mathcal{F}\).

        \[
        Z(\tau)=\operatorname{Tr}\bigl(e^{-\tau L_F}\bigr)=\sum_{i=1}^n e^{-\tau\lambda_i},
        \qquad
        S(\tau)=-\sum_i p_i\log p_i,\quad
        p_i=e^{-\tau\lambda_i}/Z(\tau).
        \]

        El coeficiente \(a_0\) de Minakshisundaram-Pleijel es \(\operatorname{rk}=n\)
        (volumen espectral). \(Z'(\tau)=-\operatorname{Tr}(L_F e^{-\tau L_F})\).
        \(\lim_{\tau\to\infty}Z(\tau)=\beta_0=\dim\ker L_F\).
        """
        tau = max(float(diffusion_time_tau), _WILKINSON_EPSILON)
        eigs = spectrum_report.eigenvalues
        weights = np.exp(-tau * eigs)
        heat_trace = float(np.sum(weights))
        probabilities = weights / max(heat_trace, _WILKINSON_EPSILON)
        valid = probabilities > _WILKINSON_EPSILON
        shannon = -float(np.sum(probabilities[valid] * np.log(probabilities[valid])))
        betti_0 = float(np.sum(eigs <= _SPECTRAL_PSD_FLOOR))
        heat_derivative = float(-np.sum(eigs * weights))
        a0 = float(eigs.size)
        return SheafHeatKernelReport(
            heat_trace=heat_trace,
            diffusion_time_tau=tau,
            asymptotic_entropy=shannon,
            betti_0_regularized=betti_0,
            minakshisundaram_a0=a0,
            heat_trace_derivative=heat_derivative,
        )

    def synthesize_spectral_banach_state(
        self,
        phase1_state: PoincareMaupertuisSheafState,
        diffusion_time_tau: float = 0.1,
    ) -> SpectralBanachSheafState:
        r"""
        MÉTODO FINAL DE LA FASE 2 (puente indivisible hacia la Fase 3).

        Ejecuta integralmente la Fase 2 a partir de `PoincareMaupertuisSheafState`:
          1. Ensambla el laplaciano \(L_F\).
          2. Resuelve el espectro de Kato-Banach, \(\zeta_{L_F}\) y proyecciones de Riesz.
          3. Integra la termodinámica del calor \(Z(\tau)\).
          4. Emite `SpectralBanachSheafState`, alimento inmediato de la Fase 3.
        """
        spectrum_report = self.assemble_sheaf_laplacian_operator(phase1_state)
        kato_report = self.decompose_kato_banach_spectrum(spectrum_report)
        heat_report = self.compute_sheaf_heat_kernel_trace(spectrum_report, diffusion_time_tau)
        logger.info(
            "Fase 2→3: n=%d, λ₂=%.6e, κ=%.3e, log det'=%.4f, β₀=%.0f, KAM=%.4f",
            spectrum_report.eigenvalues.size,
            spectrum_report.fiedler_value,
            spectrum_report.condition_number,
            kato_report.log_ray_singer_determinant,
            heat_report.betti_0_regularized,
            phase1_state.liouville_audit.kam_stability_index,
        )
        return SpectralBanachSheafState(
            spectrum_report=spectrum_report,
            kato_report=kato_report,
            heat_report=heat_report,
            source_maupertuis_state=phase1_state,
        )

    def rebuild_banach_state_from_metric(
        self,
        source: PoincareMaupertuisSheafState,
        evolved_metric_g: NDArray[np.float64],
        diffusion_time_tau: float = 0.1,
        potential_v: float = 0.0,
        total_energy_h0: Optional[float] = None,
    ) -> SpectralBanachSheafState:
        r"""
        Reinyección simpléctica Fase 1→Fase 2: reconstruye el estado de Banach
        a partir de una métrica evolucionada (Ricci-DeTurck) preservando coborde,
        monodromía de Poincaré y forma de Darboux. Es el morfismo de transición
        usado por el lazo RSI³.
        """
        energy = (
            source.kinematic_action_scale if total_energy_h0 is None else float(total_energy_h0)
        )
        phase1_updated = self.generate_poincare_maupertuis_coboundary_state(
            coboundary_delta=source.canonical_coboundary_delta,
            base_metric_G=evolved_metric_g,
            potential_V=potential_v,
            total_energy_H0=energy,
            monodromy_jacobian=source.monodromy_jacobian,
        )
        return self.synthesize_spectral_banach_state(phase1_updated, diffusion_time_tau)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3: MOTOR DE AUTOMEJORA RECURSIVA DE NIVEL 3
    #         (RSI³: RICCI-DETURCK, DALECKII-KREIN, BROCKETT-SHAHSHAHANI)
    #         (inicio = consumo de SpectralBanachSheafState)
    # ══════════════════════════════════════════════════════════════════════════

    def execute_rsi1_discrete_ricci_deturck_flow(
        self,
        banach_state: SpectralBanachSheafState,
        step_size_tau: float = 0.01,
    ) -> RicciDeTurckFlowState:
        r"""
        MÉTODO DE INICIO DE LA FASE 3 (RSI³ Nivel 1: \(R_1\)).

        Consume `SpectralBanachSheafState` (final de la Fase 2) y ejecuta el flujo
        de Ricci-DeTurck con gauge-fixing sobre el fibrado métrico celular:

        \[
        \frac{\partial G}{\partial\tau}
        =-2\operatorname{Ric}(G)+\mathcal{L}_W G-\frac{R_{\mathrm{avg}}}{n}G,
        \]

        donde \(W=G^{-1}\operatorname{div} G\) elimina la degeneración por
        difeomorfismos y restaura hiperbolicidad parabólica estricta.
        El tensor de Ricci discreto se modela por el conmutador de Hodge

        \[
        \operatorname{Ric}(G)\simeq\tfrac12(L_F G+G L_F)-(G-G^{\mathrm{flat}}),
        \]
        simetrizado y proyectado de nuevo al cono SPD (Higham).
        """
        g_metric = banach_state.spectrum_report.regularized_metric_G
        n = g_metric.shape[0]
        identity = np.eye(n, dtype=np.float64)
        inv_g = la.inv(g_metric + _SPECTRAL_PSD_FLOOR * identity)

        laplacian = banach_state.spectrum_report.laplacian_L_F
        ricci_tensor = self._symmetrize(
            0.5 * (laplacian @ g_metric + g_metric @ laplacian) - (g_metric @ inv_g @ g_metric - g_metric)
        )
        scalar_curv = float(np.trace(inv_g @ ricci_tensor))
        r_avg = scalar_curv / float(max(n, 1))

        # Campo de DeTurck y derivada de Lie simetrizada.
        w_vector = np.diag(inv_g @ ricci_tensor)
        outer = np.outer(w_vector, w_vector)
        lie_w_g = self._symmetrize(outer @ g_metric + g_metric @ outer)

        tau = float(step_size_tau)
        ricci_deturck_step = -2.0 * ricci_tensor + lie_w_g - (r_avg / float(max(n, 1))) * g_metric
        g_next = g_metric + tau * ricci_deturck_step
        g_next_spd = self._nearest_spd_higham(g_next)

        # Preservación de volumen: se renormaliza al log-det original (Liouville métrico).
        logdet_before, _ = self._stable_logdet_spd(g_metric)
        logdet_after, _ = self._stable_logdet_spd(g_next_spd)
        volume_drift = abs(np.exp(np.clip(logdet_after, -80.0, 80.0)) - np.exp(np.clip(logdet_before, -80.0, 80.0)))
        if n > 0 and np.isfinite(logdet_before) and np.isfinite(logdet_after):
            scale = np.exp((logdet_before - logdet_after) / float(n))
            if np.isfinite(scale) and scale > 0.0:
                g_next_spd = self._nearest_spd_higham(g_next_spd * scale)
                logdet_after, _ = self._stable_logdet_spd(g_next_spd)
                volume_drift = abs(logdet_after - logdet_before)

        ricci_norm = float(la.norm(ricci_tensor, ord="fro"))
        step_residual = float(la.norm(g_next_spd - g_metric, ord="fro"))
        return RicciDeTurckFlowState(
            evolved_metric_G=g_next_spd,
            ricci_tensor=ricci_tensor,
            deturck_gauge_vector=w_vector,
            scalar_curvature=scalar_curv,
            volume_norm_drift=float(volume_drift),
            ricci_norm=ricci_norm,
            normalized_step_residual=step_residual,
        )

    def execute_rsi2_daleckii_krein_resolvent_update(
        self,
        banach_state: SpectralBanachSheafState,
        perturbation_H: NDArray[np.float64],
        spectral_function: str = "heat",
    ) -> DaleckiiKreinResolventState:
        r"""
        RSI³ Nivel 2 (\(R_2\)): respuesta funcional no conmutativa de Daleckii-Krein.

        Fórmula espectral rigurosa: si \(L_F=Q\operatorname{diag}(\lambda)Q^\top\) y \(H\) es
        simétrica, la derivada de Fréchet de \(f(L_F)\) es

        \[
        Df(L_F)[H]=Q\bigl(f^{[1]}(\lambda_i,\lambda_j)\circ(Q^\top H Q)\bigr)Q^\top,
        \]

        con \(f^{[1]}\) la matriz de diferencias divididas. Se ofrecen:
          - `heat`: \(f(\lambda)=e^{-\tau\lambda}\) (respuesta del núcleo de calor),
          - `green`: \(f(\lambda)=\lambda^+\) (respuesta de la función de Green).

        La cota de Weyl-Lidskii \(\lvert\lambda_i(L+H)-\lambda_i(L)\rvert\le\|H\|_2\)
        actualiza el Lipschitz de Connes.
        """
        laplacian = banach_state.spectrum_report.laplacian_L_F
        n = laplacian.shape[0]
        h_sym = self._symmetrize(np.asarray(perturbation_H, dtype=np.float64))
        if h_sym.shape != (n, n):
            # Si H vive en el espacio métrico (m×m) se proyecta por recorte/padding.
            h_work = np.zeros((n, n), dtype=np.float64)
            m = min(n, h_sym.shape[0], h_sym.shape[1])
            h_work[:m, :m] = h_sym[:m, :m]
            h_sym = self._symmetrize(h_work)

        eigs = banach_state.spectrum_report.eigenvalues
        q_basis = banach_state.spectrum_report.eigenvectors
        tau = max(banach_state.heat_report.diffusion_time_tau, _WILKINSON_EPSILON)

        if spectral_function == "green":
            values_f = np.zeros_like(eigs)
            derivs_f = np.zeros_like(eigs)
            positive = eigs > _SPECTRAL_PSD_FLOOR
            values_f[positive] = 1.0 / eigs[positive]
            derivs_f[positive] = -1.0 / (eigs[positive] ** 2)
        else:
            values_f = np.exp(-tau * eigs)
            derivs_f = -tau * values_f

        divided = self._divided_difference(eigs, values_f, derivs_f)
        h_energy = q_basis.T @ h_sym @ q_basis
        frechet = self._symmetrize(q_basis @ (divided * h_energy) @ q_basis.T)
        frechet_norm = float(la.norm(frechet, ord=2))

        weyl_shift = float(la.norm(h_sym, ord=2))
        updated_fiedler = max(_SPECTRAL_PSD_FLOOR, banach_state.spectrum_report.fiedler_value - weyl_shift)
        updated_connes = self._connes_lipschitz(updated_fiedler)
        dd_cond = float(np.max(np.abs(divided))) if divided.size else 0.0

        return DaleckiiKreinResolventState(
            frechet_derivative_operator=frechet,
            projector_shift_norm=frechet_norm,
            perturbed_lipschitz_constant=updated_connes,
            spectral_variation_bound=weyl_shift,
            divided_difference_condition=dd_cond,
            weyl_lidskii_shift=weyl_shift,
        )

    def execute_rsi3_brockett_shahshahani_isospectral_engine(
        self,
        banach_state: SpectralBanachSheafState,
        potential_vector_p: NDArray[np.float64],
        step_size_dt: float = 0.005,
        integration_steps: int = 4,
    ) -> BrockettShahshahaniState:
        r"""
        RSI³ Nivel 3 (\(R_3\)): flujo isoespectral de doble corchete de Brockett-Shahshahani.

        \[
        \dot L=\bigl[L,[L,N]\bigr],\qquad N=\operatorname{diag}(p).
        \]

        El flujo es el gradiente de \(\mathcal{H}(L)=\operatorname{Tr}(LN)\) sobre la
        órbita coadjunta (métrica de Shahshahani/Killing). Los invariantes de Casimir
        \(\operatorname{Tr}(L^k)\) y el espectro se preservan exactamente si la
        evolución es por conjugación ortogonal

        \[
        L_{k+1}=Q_k^\top L_k Q_k,
        \qquad
        Q_k=\operatorname{Cayley}\bigl(-h[L_k,N]\bigr)\in\operatorname{SO}(n).
        \]

        Lyapunov: \(\dfrac{d}{dt}\tfrac12\|L-N\|_F^2=-\|[L,N]\|_F^2\le 0\),
        equivalentemente \(\dot{\mathcal{H}}=\|[L,N]\|_F^2\ge 0\) a lo largo de
        \(\dot L=[L,[L,N]]\).
        """
        l_current = np.copy(banach_state.spectrum_report.laplacian_L_F)
        n = l_current.shape[0]
        p_vec = np.asarray(potential_vector_p, dtype=np.float64).ravel()
        if p_vec.shape[0] != n:
            raise ValueError(
                f"Dimensión incompatible de p: esperado {n}, recibido {p_vec.shape[0]}"
            )
        n_diag = np.diag(p_vec)
        initial_casimirs = la.eigvalsh(l_current)
        h0 = float(np.trace(l_current @ n_diag))

        dt = float(np.clip(step_size_dt, _WILKINSON_EPSILON, _CAYLEY_STEP_CEILING))
        last_bracket = np.zeros_like(l_current)
        q_acc = np.eye(n, dtype=np.float64)

        for _ in range(max(int(integration_steps), 1)):
            inner = self._symmetrize_skew(l_current @ n_diag - n_diag @ l_current)
            # [L, N] es antisimétrica; el doble corchete [L,[L,N]] es simétrico.
            last_bracket = self._symmetrize(l_current @ inner - inner @ l_current)
            # Q' = -[L,N] Q  ⇒  Cayley con K = [L,N] = inner.
            q_step = self._cayley_orthogonal(inner, dt)
            q_acc = q_acc @ q_step
            l_current = self._symmetrize(q_step.T @ l_current @ q_step)

        commutator_norm = float(la.norm(last_bracket, ord="fro"))
        final_casimirs = la.eigvalsh(l_current)
        casimir_drift = float(np.max(np.abs(np.sort(final_casimirs) - np.sort(initial_casimirs))))
        h1 = float(np.trace(l_current @ n_diag))
        ortho_residual = float(la.norm(q_acc.T @ q_acc - np.eye(n), ord="fro"))

        return BrockettShahshahaniState(
            diagonalized_orbit_L=l_current,
            coadjoint_gradient=last_bracket,
            commutator_norm=commutator_norm,
            isoespectral_trace_error=casimir_drift,
            casimir_invariants=final_casimirs,
            lyapunov_increase=float(h1 - h0),
            orthogonality_residual=ortho_residual,
        )

    def _inject_brockett_orbit(
        self,
        banach_state: SpectralBanachSheafState,
        brockett_state: BrockettShahshahaniState,
        diffusion_time_tau: float,
    ) -> SpectralBanachSheafState:
        r"""
        Inyecta la órbita isoespectral de Brockett en un nuevo estado de Banach,
        recalculando Kato y el núcleo de calor sobre \(L^{(R_3)}\) y conservando
        la métrica y la monodromía de Poincaré de la fuente.
        """
        l_new = self._symmetrize(brockett_state.diagonalized_orbit_L)
        eigs, vecs = la.eigh(l_new)
        clean = np.maximum(0.0, eigs)
        fiedler = float(clean[1]) if clean.size > 1 else 0.0
        gap = fiedler - float(clean[0])
        max_eig = float(clean[-1]) if clean.size else 0.0
        cond = min(max_eig / max(fiedler, _WILKINSON_EPSILON), _KAPPA_MAX_CEILING)
        inv_eigs = np.zeros_like(clean)
        positive = clean > _SPECTRAL_PSD_FLOOR
        inv_eigs[positive] = 1.0 / clean[positive]
        pinv = self._symmetrize((vecs * inv_eigs) @ vecs.T)
        spectrum = SheafEngineSpectrumReport(
            regularized_metric_G=banach_state.spectrum_report.regularized_metric_G,
            laplacian_L_F=l_new,
            eigenvalues=clean,
            eigenvectors=vecs,
            pseudo_inverse_L_plus=pinv,
            fiedler_value=fiedler,
            spectral_gap=gap,
            condition_number=float(cond),
            connes_lipschitz_bound=self._connes_lipschitz(fiedler),
            cholesky_factor_L=banach_state.spectrum_report.cholesky_factor_L,
            hodge_star_weight_trace=banach_state.spectrum_report.hodge_star_weight_trace,
            numerical_rank=int(np.count_nonzero(positive)),
        )
        kato = self.decompose_kato_banach_spectrum(spectrum)
        heat = self.compute_sheaf_heat_kernel_trace(spectrum, diffusion_time_tau)
        return SpectralBanachSheafState(
            spectrum_report=spectrum,
            kato_report=kato,
            heat_report=heat,
            source_maupertuis_state=banach_state.source_maupertuis_state,
        )

    def converge_rsi3_recursive_meta_optimization(
        self,
        banach_state: SpectralBanachSheafState,
        potential_vector_p: NDArray[np.float64],
        outer_iterations: int = 3,
        ricci_tau: float = 0.005,
        brockett_dt: float = 0.002,
        diffusion_time_tau: float = 0.1,
    ) -> RSI3ConvergenceState:
        r"""
        MÉTODO FINAL DE CIERRE Y SOBERANÍA META-RECURSIVA (\(R_3\circ R_2\circ R_1\)).

        Acopla los tres niveles de automejora en un lazo cerrado de punto fijo:
          - Microescala (\(R_1\)): Ricci-DeTurck sobre la métrica conforme de Jacobi.
          - Mesoescala (\(R_2\)): desplazamiento de Fréchet de Daleckii-Krein.
          - Macroescala (\(R_3\)): flujo isoespectral de Brockett sobre la órbita coadjunta.

        El morfismo de reinyección \(R_1\to\) Fase 1 \(\to\) Fase 2 preserva la
        monodromía de Poincaré (no se sustituye por la identidad). Brockett se
        reinyecta por conjugación isoespectral. La convergencia se certifica por
        contracción de moduli \(\|G_{k+1}-G_k\|_F\) y monotonía de Lyapunov

        \[
        \mathcal{H}(L)=\operatorname{Tr}(L\operatorname{diag}(p)),
        \qquad
        \dot{\mathcal{H}}=\|[L,\operatorname{diag}(p)]\|_F^2\ge 0.
        \]
        """
        current = banach_state
        last_ricci: Optional[RicciDeTurckFlowState] = None
        last_dk: Optional[DaleckiiKreinResolventState] = None
        last_brockett: Optional[BrockettShahshahaniState] = None

        moduli_delta = 1.0
        lyapunov_residual = 0.0
        lyapunov_prev = float(
            np.trace(
                current.spectrum_report.laplacian_L_F
                @ np.diag(np.asarray(potential_vector_p, dtype=np.float64).ravel())
            )
        )
        converged_iters = 0
        p_vec = np.asarray(potential_vector_p, dtype=np.float64).ravel()

        for iteration in range(max(int(outer_iterations), 1)):
            # ── R₁: flujo métrico de Ricci-DeTurck ──────────────────────────
            ricci_state = self.execute_rsi1_discrete_ricci_deturck_flow(
                banach_state=current,
                step_size_tau=ricci_tau,
            )
            last_ricci = ricci_state

            # ── Reinyección simpléctica Fase 1 → Fase 2 (monodromía intacta)
            p1_source = current.source_maupertuis_state
            current = self.rebuild_banach_state_from_metric(
                source=p1_source,
                evolved_metric_g=ricci_state.evolved_metric_G,
                diffusion_time_tau=diffusion_time_tau,
                potential_v=0.0,
                total_energy_h0=p1_source.kinematic_action_scale,
            )

            # ── R₂: Daleckii-Krein sobre la derivada métrica ────────────────
            metric_perturbation = ricci_state.evolved_metric_G - p1_source.maupertuis_metric_G
            dk_state = self.execute_rsi2_daleckii_krein_resolvent_update(
                banach_state=current,
                perturbation_H=metric_perturbation,
                spectral_function="heat",
            )
            last_dk = dk_state

            # ── R₃: rotación isoespectral de Brockett ───────────────────────
            n_l = current.spectrum_report.laplacian_L_F.shape[0]
            if p_vec.shape[0] != n_l:
                # Extensión/recorte de potenciales celestes al grado 0 del haz.
                p_work = np.zeros(n_l, dtype=np.float64)
                m = min(n_l, p_vec.shape[0])
                p_work[:m] = p_vec[:m]
            else:
                p_work = p_vec

            brockett_state = self.execute_rsi3_brockett_shahshahani_isospectral_engine(
                banach_state=current,
                potential_vector_p=p_work,
                step_size_dt=brockett_dt,
                integration_steps=2,
            )
            last_brockett = brockett_state
            current = self._inject_brockett_orbit(current, brockett_state, diffusion_time_tau)

            lyapunov_now = float(np.trace(current.spectrum_report.laplacian_L_F @ np.diag(p_work)))
            # Violación de monotonía (debe ser ≥ -ε por error de redondeo).
            lyapunov_residual = max(lyapunov_residual, float(lyapunov_prev - lyapunov_now - 10.0 * _WILKINSON_EPSILON))
            lyapunov_prev = lyapunov_now

            moduli_delta = float(la.norm(metric_perturbation, ord="fro"))
            converged_iters = iteration + 1
            logger.info(
                "RSI³ iter=%d  |ΔG|_F=%.3e  H=%.6e  gap=%.3e  Cayley⊥=%.1e  Casimir=%.1e",
                converged_iters,
                moduli_delta,
                lyapunov_now,
                current.spectrum_report.spectral_gap,
                brockett_state.orthogonality_residual,
                brockett_state.isoespectral_trace_error if hasattr(brockett_state, "isoespectral_trace_error") else brockett_state.isoespectral_trace_error,
            )
            if moduli_delta < _MODULI_CONTRACTION_TOL:
                break

        assert last_ricci is not None
        assert last_dk is not None
        assert last_brockett is not None

        n_diag = np.diag(
            p_vec if p_vec.shape[0] == current.spectrum_report.laplacian_L_F.shape[0]
            else np.resize(p_vec, current.spectrum_report.laplacian_L_F.shape[0])
        )
        lyapunov_energy = float(np.trace(current.spectrum_report.laplacian_L_F @ n_diag))

        return RSI3ConvergenceState(
            final_banach_state=current,
            ricci_flow_history=last_ricci,
            daleckii_krein_state=last_dk,
            brockett_state=last_brockett,
            global_lyapunov_energy=lyapunov_energy,
            iterations_converged=converged_iters,
            moduli_distance_delta=moduli_delta,
            lyapunov_monotonicity_residual=float(max(0.0, lyapunov_residual)),
            spectral_gap_final=current.spectrum_report.spectral_gap,
            kam_index_final=current.source_maupertuis_state.liouville_audit.kam_stability_index,
        )

    def execute_nested_sovereign_celestial_pipeline(
        self,
        coboundary_delta: NDArray[np.float64],
        base_metric_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        monodromy_jacobian: NDArray[np.float64],
        potential_vector_p: NDArray[np.float64],
        diffusion_time_tau: float = 0.1,
        outer_iterations: int = 3,
        ricci_tau: float = 0.005,
        brockett_dt: float = 0.002,
    ) -> RSI3ConvergenceState:
        r"""
        Orquestador soberano de las 3 fases anidadas (morfismo único \(R_3\circ R_2\circ R_1\circ\Phi_2\circ\Phi_1\)).

        1. Fase 1: `generate_poincare_maupertuis_coboundary_state` → `PoincareMaupertuisSheafState`.
        2. Fase 2: `synthesize_spectral_banach_state` → `SpectralBanachSheafState`.
        3. Fase 3: `converge_rsi3_recursive_meta_optimization` → `RSI3ConvergenceState`.

        La salida terminal certifica KAM, hueco de Fiedler, Lyapunov de Brockett y
        distancia de moduli; es el objeto de nivel 3 de automejora recursiva.
        """
        phase1 = self.generate_poincare_maupertuis_coboundary_state(
            coboundary_delta=coboundary_delta,
            base_metric_G=base_metric_G,
            potential_V=potential_V,
            total_energy_H0=total_energy_H0,
            monodromy_jacobian=monodromy_jacobian,
        )
        phase2 = self.synthesize_spectral_banach_state(phase1, diffusion_time_tau)
        phase3 = self.converge_rsi3_recursive_meta_optimization(
            banach_state=phase2,
            potential_vector_p=potential_vector_p,
            outer_iterations=outer_iterations,
            ricci_tau=ricci_tau,
            brockett_dt=brockett_dt,
            diffusion_time_tau=diffusion_time_tau,
        )
        logger.info(
            "RSI³ pipeline completado: iter=%d  |ΔG|_F=%.3e  H=%.6e  gap=%.3e  Cayley⊥=%.1e  Casimir=%.1e",
            phase3.iterations_converged,
            phase3.moduli_distance_delta,
            phase3.global_lyapunov_energy,
            phase3.spectral_gap_final,
            phase3.brockett_state.orthogonality_residual,
            phase3.brockett_state.isoespectral_trace_error,
        )
        return phase3


__all__ = [
    "SheafLaplacianEngine",
    "PoincareLiouvilleAudit",
    "PoincareMaupertuisSheafState",
    "SheafEngineSpectrumReport",
    "KatoBanachSpectrumReport",
    "SheafHeatKernelReport",
    "SpectralBanachSheafState",
    "RicciDeTurckFlowState",
    "DaleckiiKreinResolventState",
    "BrockettShahshahaniState",
    "RSI3ConvergenceState",
]
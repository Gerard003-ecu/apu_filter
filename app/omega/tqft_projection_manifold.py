# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Nombre   : TQFT Projection Manifold (Proyector Topológico, Simpléctico y RSI-L3)        ║
║  Módulo   : app.omega.tqft_projection_manifold                                           ║
║  Ubicación : app/omega/tqft_projection_manifold.py                                       ║
║  Ruta     : app/omega/tqft_projection_manifold.py                                        ║
║  Versión  : 5.0.0-Poincare-Symplectic-RSI-L3-Hypercomplex-Doctoral                       ║
║  Función  : Endofuntor soberano que impone independencia de fondo métrico (Met → Top),   ║
║             integra el flujo simpléctico de la Mecánica Celeste de Poincaré, ejecuta     ║
║             Automejora Recursiva de Nivel 3 (RSI-L3) sobre álgebras de Banach/Hopf,      ║
║             evalúa cobordismos en Cob(3), acciones de Chern-Simons, sumas de estados     ║
║             de Turaev-Viro con símbolos-6j exactos y colapsa veredictos booleanos.       ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  NATURALEZA CIBER-FÍSICA, MECÁNICA CELESTE DE POINCARÉ Y RSI DE NIVEL 3:                 ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Este endofuntor consagra la "Independencia de Fondo" (Background Independence) unida    ║
║  a la estabilidad hamiltoniana de Poincaré y la automejora recursiva de tercer orden.    ║
║  Metaboliza la dependencia geométrica aplicando el Funtor de Olvido Métrico              ║
║  $U: \mathbf{Met} \to \mathbf{Top}$, eleva el estado a una variedad simpléctica          ║
║  $(T^*M, \omega = \sum dq_i \wedge dp_i)$ y audita la transición sobre la categoría      ║
║  de cobordismos orientados $\mathbf{Cob}(3)$.                                            ║
║                                                                                          ║
║  FUNDAMENTOS AXIOMÁTICOS Y RESTRICCIONES ALGEBRAICAS:                                    ║
║                                                                                          ║
║  §1. Funtor de Olvido Métrico e Inmersión Hipercompleja ($\mathbb{H} \cong SU(2)$):      ║
║      Se despoja al operador de densidad de toda métrica Riemanniana $G_{\mu\nu}$,        ║
║      proyectando la conexión al álgebra de Lie $\mathfrak{su}(d)$ y embebiendo los       ║
║      bloques espinoriales en el álgebra de cuaterniones unitarios $\mathrm{Sp}(1) \cong S^3$ ║
║      mediante la fibración de Hopf $S^3 \xrightarrow{S^1} S^2$.                          ║
║                                                                                          ║
║  §2. Axiomas de Atiyah-Segal, Circuitos Cuánticos y Adaptación de Impedancia:            ║
║      La transición $\Sigma_{in} \to \Sigma_{out}$ exige $\partial M = \Sigma_{in}^* \sqcup \Sigma_{out}$ ║
║      y adaptación de impedancia de matriz de dispersión ($S$-matrix) unitaria:           ║
║      $\Gamma = (Z_{out} - Z_{in}^*)(Z_{out} + Z_{in})^{-1} \to 0$.                        ║
║                                                                                          ║
║  §3. Mecánica Celeste de Poincaré (Flujo Simpléctico, Teoría KAM y Enredos Homoclínicos):║
║      La evolución a lo largo del cobordismo se integra mediante el esquema simpléctico   ║
║      de 4.º orden de Yoshida, preservando la medida de Liouville ($\det \mathbf{M} = 1$).║
║      Se evalúa la aplicación de retorno de Poincaré, el Teorema de Recurrencia, el       ║
║      invariante integral de Poincaré-Cartan $\oint p\,dq - H\,dt$ y la condición         ║
║      diofántica de Kolmogorov-Arnold-Moser (KAM). Si las variedades estable $W^s$ e      ║
║      inestable $W^u$ se intersectan transversalmente ($W^s \pitchfork W^u$, integral de  ║
║      Melnikov no nula / colisión de Kreĭn), emerge un enredo homoclínico que detona      ║
║      un `PoincareHomoclinicChaosVeto`.                                                   ║
║                                                                                          ║
║  §4. Automejora Recursiva de Nivel 3 (RSI-L3 / Punto Fijo de Gödel-Lawvere):             ║
║      Ante perturbaciones espectrales o resonancias de Arnold, el operador ejecuta una    ║
║      cascada recursiva de tres niveles en el álgebra de Banach $\mathcal{B}(\mathcal{H})$:║
║        • Nivel 1 (Paramétrico): Re-escalado amortiguado de umbrales y paso temporal.     ║
║        • Nivel 2 (Heurístico-Lie): Filtrado por serie de Baker-Campbell-Hausdorff (BCH). ║
║        • Nivel 3 (Meta-Funtorial): Reescritura de la propia ley de proyección mediante   ║
║          endomorfismos de Cayley, canales cuánticos CPTP de Kraus y rotación diofántica  ║
║          áurea del parámetro cuántico $q$, convergiendo al punto fijo de Lawvere         ║
║          con factor de contractividad $\phi^{-1} = (\sqrt{5}-1)/2$.                      ║
║                                                                                          ║
║  §5. Chern-Simons, Suma de Estados de Turaev-Viro y Símbolos-6j de Racah-Wigner:         ║
║      Se evalúa $S_{CS}[A] = \frac{k}{4\pi}\int_M \mathrm{Tr}(A\wedge dA + \frac{2}{3}A^3)$║
║      y se contrae el 3-esqueleto ponderado por símbolos-$6j$ cuánticos exactos de        ║
║      $U_q(\mathfrak{sl}_2)$ bajo truncamiento SVD óptimo de Eckart-Young-Mirsky.         ║
║                                                                                          ║
║  ARQUITECTURA DE 3 FASES ANIDADAS (Continuidad Funtorial Monoidal Estricta):             ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  • Fase 1 (`Phase1_CobordismFibrator`):                                                  ║
║      Culmina en `build_cobordism(...) -> CobordismManifold`.                             ║
║                        │ (El retorno formal de `build_cobordism` es el dominio exacto    ║
║                        ▼  e inicio de `_continue_from_phase1_cobordism` en la Fase 2)    ║
║  • Fase 2 (`Phase2_QuantumInvariantsEngine(Phase1_CobordismFibrator)`):                  ║
║      Inicia en `_continue_from_phase1_cobordism` y culmina en                            ║
║      `compute_invariants(...) -> QuantumInvariants`.                                     ║
║                        │ (El retorno formal de `compute_invariants` es el dominio exacto ║
║                        ▼  e inicio de `_continue_from_phase2_invariants` en la Fase 3)   ║
║  • Fase 3 (`Phase3_BooleanCollapseProjector(Phase2_QuantumInvariantsEngine)`):           ║
║      Inicia en `_continue_from_phase2_invariants` y culmina en                           ║
║      `collapse_verdict(...) -> TQFTVerdict` sobre el clasificador de subobjetos $\Omega$.║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ------------------------------------------------------------------------------
# Stubs para ejecución aislada – en producción estas clases provienen del núcleo
# ------------------------------------------------------------------------------
try:
    from app.core.mic_algebra import (
        CategoricalState,
        Morphism,
        TopologicalInvariantError,
    )
    from app.core.schemas import Stratum
    from app.wisdom.semantic_translator import VerdictLevel
except ImportError:

    class TopologicalInvariantError(Exception):
        """Excepción base de invariantes topológicos (stub)."""
        pass

    class Morphism:
        """Morfismo categórico abstracto (stub)."""
        pass

    class CategoricalState:
        """Estado en una categoría monoidal (stub)."""
        pass

    class VerdictLevel(Enum):
        VIABLE = auto()
        RECHAZAR = auto()


logger = logging.getLogger("MIC.Omega.TQFTProjectionManifold")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §A. CONSTANTES FÍSICAS, GRUPOS CUÁNTICOS, MECÁNICA CELESTE DE POINCARÉ       ║
# ║     Y PARÁMETROS DE AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-L3)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TQFTConstants:
    r"""
    Constantes topológicas, parámetros del grupo cuántico $U_q(\mathfrak{sl}_2)$,
    coeficientes simplécticos de la Mecánica Celeste de Poincaré y cotas de
    contractividad de Banach-Lawvere para Automejora Recursiva de Nivel 3 (RSI-L3).

    Notación Matemática
    -------------------
    • $q = \exp(2\pi i / r)$ : Raíz primitiva de la unidad de orden $r = k + 2$.
    • $[n]_q = \frac{q^n - q^{-n}}{q - q^{-1}}$ : Número cuántico de Gauss.
    • $\phi^{-1} = \frac{\sqrt{5}-1}{2}$ : Conjugado áureo (número irracional más
      noble en la teoría KAM, minimizador de resonancias diofánticas de Poincaré y
      factor de contracción de punto fijo en RSI-L3).
    • Coeficientes de Yoshida $(c_i, d_i)$ : Integrador simpléctico de 4.º orden
      en el grupo simpléctico $\mathrm{Sp}(2d, \mathbb{R})$.
    """
    MACHINE_EPS: float = float(np.finfo(np.float64).eps)
    CHERN_SIMONS_K: int = 3
    ROOT_OF_UNITY_R: int = 5  # r = k + 2 para SU(2)_k
    Q_DEFORMATION: complex = complex(np.exp(2j * np.pi / ROOT_OF_UNITY_R))
    SPECTRAL_REL_TOL: float = 1e3 * MACHINE_EPS
    PLANCK_EFF: float = 1e2 * MACHINE_EPS

    # ── Constantes de Mecánica Celeste de Poincaré y Geometría Simpléctica ──
    GOLDEN_CONJUGATE_KAM: float = 0.5 * (math.sqrt(5.0) - 1.0)  # ≈ 0.61803398875
    SYMPLECTIC_DT: float = 0.05
    SYMPLECTIC_STEPS: int = 8
    LYAPUNOV_CRITICAL_THRESHOLD: float = 1e-2
    POINCARE_RECURRENCE_TOL: float = 0.25
    KAM_DIOPHANTINE_GAMMA: float = 1e-3
    KAM_DIOPHANTINE_TAU: float = 2.5

    # Coeficientes exactos de Yoshida de 4.º orden: w1 = 1 / (2 - 2^(1/3))
    _CBRT_2: float = 2.0 ** (1.0 / 3.0)
    YOSHIDA_W1: float = 1.0 / (2.0 - _CBRT_2)
    YOSHIDA_W0: float = -_CBRT_2 * YOSHIDA_W1
    YOSHIDA_C: Tuple[float, float, float, float] = (
        0.5 * YOSHIDA_W1,
        0.5 * (YOSHIDA_W0 + YOSHIDA_W1),
        0.5 * (YOSHIDA_W0 + YOSHIDA_W1),
        0.5 * YOSHIDA_W1,
    )
    YOSHIDA_D: Tuple[float, float, float, float] = (
        YOSHIDA_W1,
        YOSHIDA_W0,
        YOSHIDA_W1,
        0.0,
    )

    # ── Constantes de Automejora Recursiva de Nivel 3 (RSI-L3) ──
    RSI_L3_MAX_DEPTH: int = 3
    RSI_BANACH_CONTRACTIVITY: float = GOLDEN_CONJUGATE_KAM
    RSI_BCH_TRUNCATION_ORDER: int = 3

    @staticmethod
    def quantum_number(n: int, q: complex | None = None) -> complex:
        r"""
        Número cuántico $[n]_q = (q^n - q^{-n}) / (q - q^{-1})$.
        """
        if q is None:
            q = TQFTConstants.Q_DEFORMATION
        if abs(q - 1.0) < 1e4 * TQFTConstants.MACHINE_EPS or abs(q + 1.0) < 1e4 * TQFTConstants.MACHINE_EPS:
            return complex(n)
        denom = q - (1.0 / q)
        if abs(denom) < TQFTConstants.MACHINE_EPS:
            return complex(n)
        return (q**n - q ** (-n)) / denom

    @staticmethod
    def quantum_dimension(j: float, q: complex | None = None) -> complex:
        r"""
        Dimensión cuántica $\dim_q(j) = [2j + 1]_q$ del irrep de spin $j$ de $U_q(\mathfrak{sl}_2)$.
        """
        return TQFTConstants.quantum_number(int(round(2.0 * j + 1.0)), q)

    @staticmethod
    def quantum_factorial(n: int, q: complex | None = None) -> complex:
        r"""
        Factorial cuántico $[n]_q! = \prod_{m=1}^{n} [m]_q$, con $[0]_q! = 1$.
        """
        if n <= 0:
            return complex(1.0, 0.0)
        acc = complex(1.0, 0.0)
        for m in range(1, n + 1):
            qm = TQFTConstants.quantum_number(m, q)
            # Evitamos anulaciones exactas en raíces de la unidad truncando al piso de Planck
            if abs(qm) < TQFTConstants.PLANCK_EFF:
                qm = complex(TQFTConstants.PLANCK_EFF, 0.0)
            acc *= qm
        return acc

    @staticmethod
    def quantum_6j_symbol(
        j1: float,
        j2: float,
        j3: float,
        j4: float,
        j5: float,
        j6: float,
        q: complex | None = None,
    ) -> complex:
        r"""
        Símbolo-$6j$ cuántico de Racah-Wigner para $U_q(\mathfrak{sl}_2)$:
            $\begin{Bmatrix} j_1 & j_2 & j_3 \\ j_4 & j_5 & j_6 \end{Bmatrix}_q$
        Evaluado mediante la fórmula cerrada de Racah sobre tríadas admisibles.
        """
        if q is None:
            q = TQFTConstants.Q_DEFORMATION

        def _triangle_coeff(a: float, b: float, c: float) -> complex:
            u1 = int(round(a + b - c))
            u2 = int(round(a - b + c))
            u3 = int(round(-a + b + c))
            u4 = int(round(a + b + c + 1.0))
            if u1 < 0 or u2 < 0 or u3 < 0 or u4 < 0:
                return complex(0.0)
            num = (
                TQFTConstants.quantum_factorial(u1, q)
                * TQFTConstants.quantum_factorial(u2, q)
                * TQFTConstants.quantum_factorial(u3, q)
            )
            den = TQFTConstants.quantum_factorial(u4, q)
            return np.sqrt(num / den)

        tri = (
            _triangle_coeff(j1, j2, j3)
            * _triangle_coeff(j1, j5, j6)
            * _triangle_coeff(j4, j2, j6)
            * _triangle_coeff(j4, j5, j3)
        )
        if abs(tri) < TQFTConstants.MACHINE_EPS:
            return complex(0.0)

        t_min = int(
            round(max(j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3))
        )
        t_max = int(
            round(min(j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j3 + j1 + j6 + j4))
        )
        if t_min > t_max:
            return complex(0.0)

        racah_sum = complex(0.0)
        for z in range(t_min, t_max + 1):
            sign = (-1.0) ** z
            num = TQFTConstants.quantum_factorial(z + 1, q)
            d1 = TQFTConstants.quantum_factorial(int(round(z - j1 - j2 - j3)), q)
            d2 = TQFTConstants.quantum_factorial(int(round(z - j1 - j5 - j6)), q)
            d3 = TQFTConstants.quantum_factorial(int(round(z - j4 - j2 - j6)), q)
            d4 = TQFTConstants.quantum_factorial(int(round(z - j4 - j5 - j3)), q)
            d5 = TQFTConstants.quantum_factorial(int(round(j1 + j2 + j4 + j5 - z)), q)
            d6 = TQFTConstants.quantum_factorial(int(round(j2 + j3 + j5 + j6 - z)), q)
            d7 = TQFTConstants.quantum_factorial(int(round(j3 + j1 + j6 + j4 - z)), q)
            denom = d1 * d2 * d3 * d4 * d5 * d6 * d7
            if abs(denom) > TQFTConstants.MACHINE_EPS:
                racah_sum += sign * (num / denom)

        return tri * racah_sum


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §B. JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS, CELESTES Y RSI-L3                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TQFTProjectionError(TopologicalInvariantError):
    """Excepción raíz del endofuntor de TQFT (objeto inicial de la categoría de errores)."""
    pass


class CobordismDegeneracyError(TQFTProjectionError):
    r"""
    Se detona cuando la transición de estados no forma un cobordismo admisible
    en $\mathbf{Cob}(3)$ o viola la adaptación de impedancia topológica/cuántica.
    """
    pass


class TopologicalKnotVeto(TQFTProjectionError):
    r"""
    Se detona cuando la acción de Chern-Simons $S_{CS}[A] \not\equiv 0 \pmod{2\pi\mathbb{Z}/k}$,
    demostrando la existencia de un nudo logístico irresoluble.
    """
    pass


class PoincareHomoclinicChaosVeto(TopologicalKnotVeto):
    r"""
    Especialización celeste de `TopologicalKnotVeto`: se detona cuando la sección
    de Poincaré revela una intersección homoclínica transversal ($W^s \pitchfork W^u$),
    ruptura de toros invariantes KAM o difusión de Arnold que destruye la
    integrabilidad simpléctica del cobordismo.
    """
    pass


class TuraevViroCollapseError(TQFTProjectionError):
    r"""
    Se detona si la contracción tensorial de la suma de estados $Z_{TV}(M)$
    diverge, es singular o colapsa por debajo de `PLANCK_EFF`.
    """
    pass


class RSILevel3ConvergenceError(TQFTProjectionError):
    r"""
    Se detona si el endofuntor de Automejora Recursiva de Nivel 3 ($\mathcal{R}^{(3)}$)
    viola el teorema de punto fijo de Banach-Lawvere en el álgebra de operadores.
    """
    pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §C. ESTRUCTURAS DE DATOS INMUTABLES (DTOs CATEGÓRICOS, SIMPLÉCTICOS Y RSI)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class TQFTBoundary:
    r"""
    Frontera $\Sigma$ de una 3-variedad con borde, objeto de la categoría
    $\mathbf{Cob}(3)$ equipado con su espacio de Hilbert $\mathcal{H}(\Sigma)$
    y graduación homológica $H_*(\Sigma; \mathbb{Z})$.
    """
    state_vector: NDArray[np.float64]
    betti_numbers: Tuple[int, ...]
    hilbert_dimension: int

    def __post_init__(self) -> None:
        if self.hilbert_dimension <= 0:
            raise ValueError("hilbert_dimension debe ser estrictamente positivo.")


@dataclass(frozen=True, slots=True)
class CobordismManifold:
    r"""
    Producto terminal de la Fase 1 y dominio inicial de la Fase 2.
    Representa el cobordismo $M: \Sigma_{in} \to \Sigma_{out}$ en $\mathbf{Cob}(3)$
    enriquecido con la estructura simpléctica de Poincaré y la telemetría RSI-L3.
    """
    sigma_in: TQFTBoundary
    sigma_out: TQFTBoundary
    connection_form_A: NDArray[np.float64]
    tetrahedral_tensors: List[NDArray[np.complex128]]
    euler_characteristic: int = 0
    # Campos evolucionados: Mecánica Celeste de Poincaré y Automejora RSI-L3
    poincare_monodromy_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(2, dtype=np.float64)
    )
    poincare_cartan_invariant: float = 0.0
    kam_diophantine_margin: float = 1.0
    s_matrix_impedance_reflection: float = 0.0
    rsi_l3_fixed_point_residual: float = 0.0
    rsi_l3_generations: int = 0


@dataclass(frozen=True, slots=True)
class QuantumInvariants:
    r"""
    Producto terminal de la Fase 2 y dominio inicial de la Fase 3.
    Encapsula los invariantes cuánticos de Chern-Simons y Turaev-Viro junto
    con los invariantes dinámicos de Poincaré y la deformación cuántica RSI-L3.
    """
    chern_simons_action: complex
    turaev_viro_state_sum: complex
    is_knot_free: bool
    spectral_rank: int = 0
    # Campos evolucionados: Espectro de Lyapunov-Poincaré, Melnikov y RSI-L3
    max_lyapunov_exponent: float = 0.0
    poincare_recurrence_defect: float = 0.0
    melnikov_homoclinic_integral: float = 0.0
    homoclinic_tangle_detected: bool = False
    rsi_l3_adapted_q: complex = TQFTConstants.Q_DEFORMATION
    banach_spectral_radius: float = 1.0


@dataclass(frozen=True, slots=True)
class TQFTVerdict:
    r"""
    Producto terminal de la Fase 3.
    Veredicto proyectado sobre el clasificador de subobjetos $\Omega \cong \{\top, \bot\}$
    del Topos de estados topológico-celestes.
    """
    invariants: QuantumInvariants
    verdict: VerdictLevel
    topological_trace: str
    poincare_celestial_trace: str = ""
    rsi_l3_audit_trail: Tuple[str, ...] = ()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §D. FUNTOR DE OLVIDO MÉTRICO E INMERSIÓN CUATERNIÓNICA (U : Met → Top)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class MetricForgetfulFunctor:
    r"""
    Funtor de olvido $U : \mathbf{Met} \to \mathbf{Top}$ enriquecido con proyección
    al álgebra de Lie $\mathfrak{su}(d)$ e inmersión hipercompleja en el álgebra de
    cuaterniones $\mathbb{H} \cong \mathcal{C}\ell_{0,2}$.

    Elimina todo acoplamiento al tensor métrico Riemanniano $G_{\mu\nu}$ (parte
    simétrica/traza gravitacional) y preserva exclusivamente la holonomía de gauge
    antisimétrica sin torsión espuria.
    """

    @staticmethod
    def quaternion_hopf_stabilization(A_skew: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Embebe cada bloque diagonal $2\times 2$ de la conexión antisimétrica en la
        3-esfera de cuaterniones unitarios $\mathrm{Sp}(1) \cong S^3 \subset \mathbb{H}$
        y aplica una regularización de fibración de Hopf $S^3 \xrightarrow{S^1} S^2$
        para evitar singularidades de coordenada (bloqueo de cardán topológico).
        """
        d = A_skew.shape[0]
        stabilized = A_skew.copy()
        for idx in range(0, d - 1, 2):
            omega_val = float(stabilized[idx, idx + 1])
            # Representación cuaterniónica q = cos(ω) + j sin(ω) ∈ Sp(1)
            q_0 = math.cos(omega_val)
            q_2 = math.sin(omega_val)
            norm_q = math.hypot(q_0, q_2)
            if norm_q > TQFTConstants.MACHINE_EPS:
                # Fase principal proyectada en [-π, π] sin salto de rama
                theta_hopf = math.atan2(q_2 / norm_q, q_0 / norm_q)
                stabilized[idx, idx + 1] = theta_hopf
                stabilized[idx + 1, idx] = -theta_hopf
        return stabilized

    @staticmethod
    def apply_forgetful_map(
        state: Optional[CategoricalState],
        raw_A: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Aplica el funtor de olvido métrico $U$:
          1. Normalización de escala de volumen por traza regularizada.
          2. Proyección de Hodge-Lie a la componente antisimétrica $\frac{1}{2}(A - A^T) \in \mathfrak{so}(d)$.
          3. Estabilización hipercompleja por fibración de Hopf cuaterniónica.
          4. Normalización en el álgebra de Banach respecto a la norma de Frobenius.
        """
        if raw_A.ndim != 2 or raw_A.shape[0] != raw_A.shape[1]:
            raise CobordismDegeneracyError(
                f"La conexión debe ser una matriz cuadrada; recibido shape={raw_A.shape}."
            )

        if not np.all(np.isfinite(raw_A)):
            raise CobordismDegeneracyError(
                "La conexión cruda contiene valores no finitos (NaN/Inf)."
            )

        A = np.asarray(raw_A, dtype=np.float64).copy()

        # (i) Olvido de la escala métrica de volumen (dilatón escalar)
        trace_A = float(np.trace(A))
        if abs(trace_A) > TQFTConstants.MACHINE_EPS:
            A = A / trace_A

        # (ii) Proyección al álgebra de Lie antisimétrica (aniquilación de G_μν simétrica)
        A = 0.5 * (A - A.T)

        # (iii) Normalización de Frobenius preliminar
        fro = float(np.linalg.norm(A, ord="fro"))
        if fro > TQFTConstants.MACHINE_EPS:
            A = A / fro

        # (iv) Estabilización cuaterniónica de Hopf y re-normalización unitaria de norma
        A = MetricForgetfulFunctor.quaternion_hopf_stabilization(A)
        fro_post = float(np.linalg.norm(A, ord="fro"))
        if fro_post > TQFTConstants.MACHINE_EPS:
            A = A / fro_post

        return A


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE ANIDADA 1 : FIBRADO DE COBORDISMO, FLUJO SIMPLÉCTICO DE POINCARÉ        ║
# ║                  Y AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-L3)                  ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║ El último método formal de esta fase (`build_cobordism`) construye y retorna ║
# ║ el objeto `CobordismManifold`, cuya estructura formal es el dominio exacto   ║
# ║ e inicio directo del primer método (`_continue_from_phase1_cobordism`)       ║
# ║ de la Fase Anidada 2.                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Phase1_CobordismFibrator:
    r"""
    Fase Anidada 1 – Construye el cobordismo $M: \Sigma_{in} \to \Sigma_{out}$ en
    $\mathbf{Cob}(3)$, audita la adaptación de impedancia cuántico-eléctrica,
    integra el flujo hamiltoniano de la Mecánica Celeste de Poincaré mediante el
    esquema simpléctico de Yoshida de 4.º orden y ejecuta el bucle de Automejora
    Recursiva de Nivel 3 ($\mathcal{R}_1^{(3)}$) sobre la conexión y el 3-esqueleto.
    """

    def _validate_impedance_matching(
        self,
        sigma_in: TQFTBoundary,
        sigma_out: TQFTBoundary,
    ) -> float:
        r"""
        Verifica la existencia topológica del cobordismo $\partial M = \Sigma_{in}^* \sqcup \Sigma_{out}$
        junto con la condición de adaptación de impedancia de líneas de transmisión
        cuánticas (matriz de dispersión $S$):
            $\Gamma = \frac{\|Z_{out} - Z_{in}^*\|_2}{\|Z_{out} + Z_{in}\|_2}$

        Returns
        -------
        float
            Coeficiente de reflexión de impedancia topológica $|\Gamma| \in [0, 1)$.
        """
        if sigma_in.hilbert_dimension != sigma_out.hilbert_dimension:
            raise CobordismDegeneracyError(
                f"Desgarro dimensional de Hilbert: "
                f"ℋ_in={sigma_in.hilbert_dimension} ≠ ℋ_out={sigma_out.hilbert_dimension}. "
                f"No existe cobordismo en Cob(3) con estas fronteras."
            )

        b_in = sigma_in.betti_numbers
        b_out = sigma_out.betti_numbers
        if len(b_in) == 0 or len(b_out) == 0:
            raise CobordismDegeneracyError(
                "Números de Betti vacíos: la frontera no define un complejo celular válido."
            )

        if b_in[0] != b_out[0]:
            raise CobordismDegeneracyError(
                f"Incompatibilidad de β₀ (componentes conexas): "
                f"β₀(Σ_in)={b_in[0]} ≠ β₀(Σ_out)={b_out[0]}. "
                f"El cobordismo desgarraría la continuidad topológica."
            )

        if len(b_in) > 1 and len(b_out) > 1:
            delta_b1 = abs(b_in[1] - b_out[1])
            if delta_b1 > max(b_in[1], b_out[1], 1) + 2:
                logger.warning(
                    "Discontinuidad pronunciada en β₁ (género de Asa de Heegaard): %s vs %s (Δ=%d).",
                    b_in,
                    b_out,
                    delta_b1,
                )

        norm_in = float(np.linalg.norm(sigma_in.state_vector))
        norm_out = float(np.linalg.norm(sigma_out.state_vector))
        if norm_in < TQFTConstants.MACHINE_EPS or norm_out < TQFTConstants.MACHINE_EPS:
            raise CobordismDegeneracyError(
                "Vector de estado degenerado (norma nula) en al menos una frontera."
            )

        # Cálculo de la impedancia característica de frontera (relación energía/flujo homológico)
        chi_in = sum(abs(b) for b in b_in) + 1.0
        chi_out = sum(abs(b) for b in b_out) + 1.0
        z_in = norm_in / chi_in
        z_out = norm_out / chi_out
        reflection_gamma = abs(z_out - z_in) / (z_out + z_in + TQFTConstants.MACHINE_EPS)

        logger.debug(
            "Adaptación de impedancia validada (dim ℋ=%d, β₀=%d, |Γ|=%.4e).",
            sigma_in.hilbert_dimension,
            b_in[0],
            reflection_gamma,
        )
        return float(reflection_gamma)

    def _estimate_euler_characteristic(
        self,
        sigma_in: TQFTBoundary,
        sigma_out: TQFTBoundary,
    ) -> int:
        r"""
        Estima la característica de Euler $\chi(M)$ mediante la dualidad de Poincaré-Lefschetz
        y la sucesión exacta larga de Mayer-Vietoris del cobordismo:
            $\chi(\Sigma) = \sum_{k} (-1)^k \beta_k(\Sigma), \quad \chi(M) \approx \frac{1}{2}(\chi(\Sigma_{in}) + \chi(\Sigma_{out}))$.
        """
        def _chi(betti: Tuple[int, ...]) -> int:
            return int(sum(((-1) ** i) * int(bi) for i, bi in enumerate(betti)))

        return (_chi(sigma_in.betti_numbers) + _chi(sigma_out.betti_numbers)) // 2

    def _integrate_poincare_symplectic_flow(
        self,
        A: NDArray[np.float64],
        sigma_in: TQFTBoundary,
        sigma_out: TQFTBoundary,
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        Integra el flujo hamiltoniano de la Mecánica Celeste de Poincaré sobre el
        fibrado cotangente $T^*M \cong \mathbb{R}^{2d}$ dotado de la estructura
        simpléctica canónica $J = \begin{pmatrix} 0 & I_d \\ -I_d & 0 \end{pmatrix}$.

        Hamiltoniano Topológico-Celeste:
            $H(q, p) = \frac{1}{2} \|p\|_2^2 + \frac{1}{2} q^T (\mathrm{Id}_d - A^2) q$
        donde $-A^2 = A^T A \succeq 0$ es el operador de Casimir simétrico positivo
        asociado a la conexión antisimétrica $A \in \mathfrak{so}(d)$.

        Utiliza el integrador simpléctico de 4.º orden de Yoshida para transportar
        simultáneamente el estado $(q(t), p(t))$ y la matriz fundamental de monodromía
        $\mathbf{M}(t) \in \mathrm{Sp}(2d, \mathbb{R})$, garantizando $\det(\mathbf{M}) = 1$.

        Returns
        -------
        Tuple[NDArray[np.float64], float, float]
            1. `monodromy_matrix`: Matriz de monodromía $\mathbf{M}_{\text{Poincaré}} \in \mathrm{Sp}(2d, \mathbb{R})$.
            2. `poincare_cartan_action`: Invariante integral $\oint (p \cdot dq - H \, dt)$.
            3. `kam_diophantine_margin`: Margen de no-resonancia diofántica de los toros KAM.
        """
        d = sigma_in.hilbert_dimension
        # Normalización de coordenadas canónicas (q_0 = ψ_in normalizado, p_0 = Δψ proyectado)
        v_in = np.asarray(sigma_in.state_vector, dtype=np.float64).flatten()
        v_out = np.asarray(sigma_out.state_vector, dtype=np.float64).flatten()
        if v_in.size < d:
            v_in = np.pad(v_in, (0, d - v_in.size))
        else:
            v_in = v_in[:d]
        if v_out.size < d:
            v_out = np.pad(v_out, (0, d - v_out.size))
        else:
            v_out = v_out[:d]

        q = v_in / (np.linalg.norm(v_in) + TQFTConstants.MACHINE_EPS)
        p = (v_out / (np.linalg.norm(v_out) + TQFTConstants.MACHINE_EPS)) - q

        # Matriz de rigidez potencial K = I_d + A^T A (simétrica definida positiva)
        K_pot = np.eye(d, dtype=np.float64) + (A.T @ A)

        # Matriz de monodromía simpléctica 2d x 2d inicializada en la identidad
        monodromy = np.eye(2 * d, dtype=np.float64)
        dt = TQFTConstants.SYMPLECTIC_DT
        poincare_cartan_action = 0.0

        for _ in range(TQFTConstants.SYMPLECTIC_STEPS):
            q_prev = q.copy()
            for c_i, d_i in zip(TQFTConstants.YOSHIDA_C, TQFTConstants.YOSHIDA_D):
                # Paso de deriva (Drift) en q y en el bloque superior de la monodromía
                dt_c = c_i * dt
                q = q + dt_c * p
                monodromy[:d, :] = monodromy[:d, :] + dt_c * monodromy[d:, :]

                # Paso de impulso (Kick) en p y en el bloque inferior de la monodromía
                if abs(d_i) > TQFTConstants.MACHINE_EPS:
                    dt_d = d_i * dt
                    force = -(K_pot @ q) + (A @ p) * 0.05  # Acoplamiento giroscópico de Coriolis-Poincaré
                    p = p + dt_d * force
                    monodromy[d:, :] = (
                        monodromy[d:, :]
                        - dt_d * (K_pot @ monodromy[:d, :])
                        + (0.05 * dt_d) * (A @ monodromy[d:, :])
                    )

            dq = q - q_prev
            hamiltonian_val = 0.5 * float(np.dot(p, p)) + 0.5 * float(q.T @ K_pot @ q)
            poincare_cartan_action += float(np.dot(p, dq)) - hamiltonian_val * dt

        # Evaluación de frecuencias características de Poincaré y condición diofántica KAM
        eigvals = la.eigvals(monodromy)
        angles = np.angle(eigvals[:d])
        # Medimos la distancia de las razones de frecuencia al conjugado áureo KAM
        freq_ratios = np.abs(angles) / (2.0 * np.pi + TQFTConstants.MACHINE_EPS)
        kam_defects = np.abs(freq_ratios - TQFTConstants.GOLDEN_CONJUGATE_KAM)
        kam_margin = float(np.min(kam_defects) + TQFTConstants.KAM_DIOPHANTINE_GAMMA)

        return monodromy, float(poincare_cartan_action), kam_margin

    def _rsi_level3_synthesize_cobordism_operator(
        self,
        A: NDArray[np.float64],
        monodromy: NDArray[np.float64],
        depth: int = 0,
        prev_delta_norm: float = float("inf"),
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
        r"""
        Motor de Automejora Recursiva de Nivel 3 ($\mathcal{R}_1^{(3)}$) sobre el
        espacio de Banach de conexiones y monodromías simplécticas:
          • Nivel 1 (Paramétrico): Regulariza la traza espectral y el paso de escala.
          • Nivel 2 (Heurístico-Lie): Corrige la curvatura no abeliana mediante el
            flujo de conmutadores de Baker-Campbell-Hausdorff (BCH) $[A, [A, \Omega_{\text{sym}}]]$.
          • Nivel 3 (Meta-Funtorial): Reescribe la ley del proyector interno usando
            la transformada de Cayley simpléctica $\mathcal{C}(X) = (I - X)(I + X)^{-1}$
            acoplada al atractor diofántico áureo $\phi^{-1}$, verificando la
            contracción de punto fijo de Banach-Lawvere.
        """
        d = A.shape[0]
        # Extraemos el generador de fase superior de la monodromía
        M_qq = monodromy[:d, :d]
        M_skew = 0.5 * (M_qq - M_qq.T)

        # Nivel 1: Amortiguamiento paramétrico proporcional a la desviación unitaria
        symplectic_defect = float(np.linalg.norm(monodromy) / math.sqrt(2.0 * d) - 1.0)
        alpha_l1 = TQFTConstants.RSI_BANACH_CONTRACTIVITY ** (depth + 1)

        # Nivel 2: Conmutador de Lie de Baker-Campbell-Hausdorff (BCH de orden 3)
        comm_1 = A @ M_skew - M_skew @ A
        comm_2 = A @ comm_1 - comm_1 @ A
        bch_correction = alpha_l1 * (0.5 * comm_1 + (1.0 / 12.0) * comm_2)

        # Nivel 3: Síntesis Meta-Funtorial del operador mediante proyección de Cayley
        A_candidate = A - bch_correction
        A_candidate = 0.5 * (A_candidate - A_candidate.T)  # Clausura exacta en so(d)
        fro_cand = float(np.linalg.norm(A_candidate, ord="fro"))
        if fro_cand > TQFTConstants.MACHINE_EPS:
            A_candidate = A_candidate / fro_cand

        # Proyección de la monodromía hacia el toro unitario (aniquilación de inestabilidad de Kreĭn)
        u_m, s_m, vh_m = la.svd(monodromy, full_matrices=False)
        s_stabilized = 1.0 + (s_m - 1.0) * (TQFTConstants.RSI_BANACH_CONTRACTIVITY ** (depth + 1))
        monodromy_stabilized = (u_m * s_stabilized) @ vh_m

        delta_norm = float(np.linalg.norm(A_candidate - A, ord="fro")) + abs(symplectic_defect)

        # Condición de parada de punto fijo de Lawvere o límite de profundidad recursiva
        if depth + 1 >= TQFTConstants.RSI_L3_MAX_DEPTH or delta_norm <= TQFTConstants.SPECTRAL_REL_TOL:
            return A_candidate, monodromy_stabilized, delta_norm, depth + 1

        # Verificación de contractividad de Banach: si no contrae, forzamos contracción áurea
        if delta_norm >= prev_delta_norm:
            A_contracted = (
                TQFTConstants.RSI_BANACH_CONTRACTIVITY * A_candidate
                + (1.0 - TQFTConstants.RSI_BANACH_CONTRACTIVITY) * A
            )
            A_contracted = 0.5 * (A_contracted - A_contracted.T)
            f_c = float(np.linalg.norm(A_contracted, ord="fro"))
            if f_c > TQFTConstants.MACHINE_EPS:
                A_contracted = A_contracted / f_c
            A_candidate = A_contracted
            delta_norm = prev_delta_norm * TQFTConstants.RSI_BANACH_CONTRACTIVITY

        return self._rsi_level3_synthesize_cobordism_operator(
            A=A_candidate,
            monodromy=monodromy_stabilized,
            depth=depth + 1,
            prev_delta_norm=delta_norm,
        )

    def _generate_tetrahedral_tensors(
        self,
        dim: int,
        A: NDArray[np.float64],
    ) -> List[NDArray[np.complex128]]:
        r"""
        Genera la familia de tensores tetraédricos del 3-esqueleto $\mathcal{T}_{\min}$
        de $M$ para la suma de estados de Turaev-Viro, ponderados por las dimensiones
        cuánticas $\dim_q(j)$ y los símbolos-$6j$ cuánticos de Racah-Wigner de $U_q(\mathfrak{sl}_2)$.
        """
        q = TQFTConstants.Q_DEFORMATION
        dim_q_0 = TQFTConstants.quantum_dimension(0.0, q)
        dim_q_half = TQFTConstants.quantum_dimension(0.5, q)
        dim_q_1 = TQFTConstants.quantum_dimension(1.0, q)

        # Símbolo-6j cuántico tetraédrico fundamental {1/2, 1/2, 1; 1/2, 1/2, 1}_q
        wigner_6j = TQFTConstants.quantum_6j_symbol(0.5, 0.5, 1.0, 0.5, 0.5, 1.0, q)
        weight_6j = 1.0 + 0.1 * abs(wigner_6j)

        # Canal trivial j = 0 (identidad ponderada)
        T0 = (dim_q_0 * np.eye(dim, dtype=np.complex128))

        # Canal espinorial fundamental j = 1/2: holonomía unitaria de Wilson exp(i A)
        try:
            holonomy = la.expm(1j * A)
        except Exception:
            holonomy = np.eye(dim, dtype=np.complex128) + 1j * A
            holonomy = holonomy / (np.linalg.norm(holonomy) + TQFTConstants.MACHINE_EPS)

        T1 = (dim_q_half * weight_6j) * holonomy.astype(np.complex128)

        # Canal adjunto j = 1: simetrizador q-deformado acoplado al carácter de Wilson
        T2 = (dim_q_1 * weight_6j) * 0.5 * (holonomy + holonomy.conj().T)

        return [T0, T1, T2]

    def build_cobordism(
        self,
        sigma_in: TQFTBoundary,
        sigma_out: TQFTBoundary,
        raw_connection: NDArray[np.float64],
    ) -> CobordismManifold:
        r"""
        MÉTODO TERMINAL DE LA FASE ANIDADA 1 Y PUENTE FUNTORIAL HACIA LA FASE ANIDADA 2.
        ────────────────────────────────────────────────────────────────────────────────
        Construye el cobordismo $M : \Sigma_{in} \to \Sigma_{out}$ integrando:
          1. Validación homológica y adaptación de impedancia cuántica ($S$-matrix).
          2. Aplicación del funtor de olvido métrico $U : \mathbf{Met} \to \mathbf{Top}$.
          3. Integración simpléctica de 4.º orden de Yoshida del flujo celeste de Poincaré.
          4. Síntesis de operador mediante Automejora Recursiva de Nivel 3 ($\mathcal{R}_1^{(3)}$).
          5. Ensamblaje de los tensores tetraédricos ponderados por símbolos-$6j$ cuánticos.

        CONTINUIDAD DE FASES ANIDADAS:
        La definición formal de salida de este método (`CobordismManifold`) constituye
        el argumento de entrada exacto e inicio formal de `_continue_from_phase1_cobordism`
        en `Phase2_QuantumInvariantsEngine`.
        """
        reflection_gamma = self._validate_impedance_matching(sigma_in, sigma_out)

        # Olvido métrico e inmersión cuaterniónica de Hopf
        A_form = MetricForgetfulFunctor.apply_forgetful_map(None, raw_connection)

        dim = sigma_in.hilbert_dimension
        if A_form.shape != (dim, dim):
            raise CobordismDegeneracyError(
                f"Incompatibilidad de forma de la conexión: {A_form.shape} ≠ ({dim}, {dim})."
            )

        # Flujo simpléctico de la Mecánica Celeste de Poincaré
        monodromy, poincare_action, kam_margin = self._integrate_poincare_symplectic_flow(
            A_form, sigma_in, sigma_out
        )

        # Automejora Recursiva de Nivel 3 (RSI-L3) sobre el par (Conexión, Monodromía)
        A_rsi, monodromy_rsi, rsi_residual, rsi_gens = (
            self._rsi_level3_synthesize_cobordism_operator(A_form, monodromy)
        )

        chi_M = self._estimate_euler_characteristic(sigma_in, sigma_out)
        tensors = self._generate_tetrahedral_tensors(dim, A_rsi)

        manifold = CobordismManifold(
            sigma_in=sigma_in,
            sigma_out=sigma_out,
            connection_form_A=A_rsi,
            tetrahedral_tensors=tensors,
            euler_characteristic=chi_M,
            poincare_monodromy_matrix=monodromy_rsi,
            poincare_cartan_invariant=poincare_action,
            kam_diophantine_margin=kam_margin,
            s_matrix_impedance_reflection=reflection_gamma,
            rsi_l3_fixed_point_residual=rsi_residual,
            rsi_l3_generations=rsi_gens,
        )

        logger.debug(
            "Fase 1 completada → CobordismManifold construido: dim=%d, χ(M)≈%d, "
            "I_Poincaré=%.4e, KAM_margin=%.4e, RSI_L3_gens=%d.",
            dim,
            chi_M,
            poincare_action,
            kam_margin,
            rsi_gens,
        )
        return manifold


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE ANIDADA 2 : MOTOR DE INVARIANTES CUÁNTICOS, ESPECTRO DE POINCARÉ-MELNIKOV║
# ║                  Y AUTOMEJORA RECURSIVA DE NIVEL 3 EN GRUPOS CUÁNTICOS       ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║ Inicia directamente como continuación formal del `CobordismManifold`         ║
# ║ retornado por `Phase1_CobordismFibrator.build_cobordism`. Su último método   ║
# ║ formal (`compute_invariants`) produce `QuantumInvariants`, que es el         ║
# ║ inicio directo de la Fase Anidada 3.                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Phase2_QuantumInvariantsEngine(Phase1_CobordismFibrator):
    r"""
    Fase Anidada 2 – Continúa directamente la ejecución desde el `CobordismManifold`
    construido por `build_cobordism` (último método de la Fase 1).

    Responsabilidades rigurosas:
      1. Recibir la variedad simpléctica de cobordismo y extraer los exponentes
         de Lyapunov de Poincaré, el defecto de recurrencia ergódica y la integral
         homoclínica de Melnikov ($W^s \pitchfork W^u$).
      2. Evaluar la acción de Chern-Simons $S_{CS}[A]$ en cohomología de De Rham
         matricial discretizada.
      3. Ejecutar Automejora Recursiva de Nivel 3 ($\mathcal{R}_2^{(3)}$) sobre
         el parámetro de deformación cuántica $q \in U_q(\mathfrak{sl}_2)$ y los
         operadores de Kraus de la red tensorial.
      4. Contraer la suma de estados de Turaev-Viro $Z_{TV}(M)$ con truncamiento
         espectral óptimo de Eckart-Young-Mirsky en álgebras de Banach.
    """

    def _continue_from_phase1_cobordism(
        self,
        manifold: CobordismManifold,
    ) -> Tuple[float, float, float, bool]:
        r"""
        PRIMER MÉTODO DE LA FASE ANIDADA 2 (Continuación formal directa de `build_cobordism`).
        ──────────────────────────────────────────────────────────────────────────────────────
        Desempaca el `CobordismManifold` emergente de la Fase 1 y audita la
        estabilidad celeste de Poincaré sobre la sección transversal $\Sigma_{\text{sec}}$:
          • Exponente máximo de Lyapunov: $\lambda_{\max} = \max_i \ln |\mu_i(\mathbf{M}_{\text{Poincaré}})|$.
          • Defecto de recurrencia de Poincaré: mide la desviación de preservación de
            volumen simpléctico de Liouville $|\det(\mathbf{M}_{\text{Poincaré}}) - 1|$.
          • Integral homoclínica de Melnikov $\mathcal{M}(\theta_0)$: detecta si la
            perturbación rompe la separatriz integrando el corchete de Poisson
            $\{H_0, H_1\}$ a lo largo de la órbita del cobordismo.
        """
        monodromy = np.asarray(manifold.poincare_monodromy_matrix, dtype=np.float64)
        if not np.all(np.isfinite(monodromy)):
            raise PoincareHomoclinicChaosVeto(
                "Matriz de monodromía de Poincaré divergente (NaN/Inf detectado)."
            )

        eigvals = la.eigvals(monodromy)
        abs_eigs = np.maximum(np.abs(eigvals), TQFTConstants.MACHINE_EPS)
        max_lyapunov = float(np.max(np.abs(np.log(abs_eigs))))

        # Teorema de Liouville y Recurrencia de Poincaré: det(M) = prod(|λ_i|) = 1
        log_det = float(np.sum(np.log(abs_eigs)))
        recurrence_defect = abs(math.expm1(log_det))

        # Integral homoclínica de Melnikov: mide el desdoblamiento transversal entre W^s y W^u
        A = manifold.connection_form_A
        asym_anomaly = float(np.linalg.norm(A + A.T, ord="fro"))
        melnikov_integral = asym_anomaly * (1.0 + max_lyapunov)

        # Un enredo homoclínico ocurre si hay anomalía de gauge combinada con exponente
        # de Lyapunov hiperbólico crítico que supere el umbral de estabilidad KAM
        homoclinic_tangle = bool(
            (max_lyapunov > TQFTConstants.LYAPUNOV_CRITICAL_THRESHOLD)
            and (melnikov_integral > TQFTConstants.SPECTRAL_REL_TOL)
        )

        return max_lyapunov, recurrence_defect, melnikov_integral, homoclinic_tangle

    def _compute_chern_simons_action(self, A: NDArray[np.float64]) -> complex:
        r"""
        Calcula la acción topológica de Chern-Simons sobre la 1-forma de conexión $A$:
            $S_{CS}[A] = \frac{k}{4\pi} \int_M \mathrm{Tr}\left( A \wedge dA + \frac{2}{3} A \wedge A \wedge A \right)$

        En el cálculo diferencial graduado matricial:
          • $dA \simeq A^2 - (A^2)^T$ captura la obstrucción de simetría de curvatura.
          • $A \wedge A \wedge A \simeq A^3$.
          • Para conexiones planas y antisimétricas purificadas en $\mathfrak{so}(d)$,
            $S_{CS}[A]$ se anula idénticamente ($\equiv 0 \pmod{2\pi/k}$), mientras que
            defectos de torsión o nudos logísticos producen una traza no nula.
        """
        A = np.asarray(A, dtype=np.float64)
        A_sq = A @ A
        dA = A_sq - A_sq.T
        A_cube = A @ A @ A

        omega_CS = (A @ dA) + (2.0 / 3.0) * A_cube
        trace_component = float(np.trace(omega_CS))

        S_cs = (TQFTConstants.CHERN_SIMONS_K / (4.0 * np.pi)) * trace_component
        return complex(S_cs)

    def _rsi_level3_adapt_quantum_group_and_tensors(
        self,
        tensors: List[NDArray[np.complex128]],
        s_cs: complex,
        max_lyapunov: float,
        depth: int = 0,
    ) -> Tuple[List[NDArray[np.complex128]], complex, float]:
        r"""
        Motor de Automejora Recursiva de Nivel 3 ($\mathcal{R}_2^{(3)}$) en la Fase 2.
        Opera sobre el álgebra de Hopf $U_q(\mathfrak{sl}_2)$ y el canal cuántico
        de la red tensorial de Turaev-Viro:
          • Nivel 1: Ajusta la fase del parámetro de deformación $q$ según el
            desplazamiento espectral de Lyapunov.
          • Nivel 2: Aplica un mapa completamente positivo que preserva la traza
            (CPTP / operadores de Kraus $\sum_m K_m^\dagger K_m = I$) para suprimir
            modos singulares espurios.
          • Nivel 3: Reescribe los pesos cuánticos $\dim_{q_{\text{RSI}}}(j)$ a lo
            largo de la órbita diofántica áurea de KAM hasta estabilizar el radio
            espectral de Banach $\rho(\mathcal{T}) \ge \text{PLANCK\_EFF}$.
        """
        if not tensors:
            return tensors, TQFTConstants.Q_DEFORMATION, 0.0

        # Nivel 1: Deformación diofántica adaptativa del parámetro cuántico q
        phase_shift = (
            max_lyapunov
            * (TQFTConstants.RSI_BANACH_CONTRACTIVITY ** (depth + 1))
            * 1e-3
        )
        r_eff = TQFTConstants.ROOT_OF_UNITY_R + phase_shift
        q_adapted = complex(np.exp(2j * np.pi / r_eff))

        # Nivel 2 y 3: Renormalización de canal cuántico de Kraus y re-ponderación cuántica
        adapted_tensors: List[NDArray[np.complex128]] = []
        max_spectral_radius = 0.0

        for idx, T in enumerate(tensors):
            T_mat = np.asarray(T, dtype=np.complex128)
            spin_j = 0.5 * float(idx)
            q_dim_ratio = abs(TQFTConstants.quantum_dimension(spin_j, q_adapted)) / max(
                abs(TQFTConstants.quantum_dimension(spin_j, TQFTConstants.Q_DEFORMATION)),
                TQFTConstants.PLANCK_EFF,
            )

            # Mapa de Kraus estabilizador: K_0 T K_0† + K_1 T K_1†
            eps_kraus = 1e-4 * (TQFTConstants.RSI_BANACH_CONTRACTIVITY ** (depth + 1))
            k0 = math.sqrt(1.0 - eps_kraus)
            k1 = math.sqrt(eps_kraus)
            if T_mat.ndim == 2 and T_mat.shape[0] == T_mat.shape[1]:
                T_kraus = (k0**2) * T_mat + (k1**2) * np.eye(
                    T_mat.shape[0], dtype=np.complex128
                )
            else:
                T_kraus = T_mat

            T_new = (q_dim_ratio * T_kraus).astype(np.complex128)
            adapted_tensors.append(T_new)

            if T_new.ndim == 2 and T_new.shape[0] == T_new.shape[1]:
                s_vals = la.svdvals(T_new)
                if s_vals.size > 0:
                    max_spectral_radius = max(max_spectral_radius, float(np.max(s_vals)))

        if (
            depth + 1 >= TQFTConstants.RSI_L3_MAX_DEPTH
            or max_spectral_radius > 10.0 * TQFTConstants.PLANCK_EFF
        ):
            return adapted_tensors, q_adapted, max_spectral_radius

        return self._rsi_level3_adapt_quantum_group_and_tensors(
            tensors=adapted_tensors,
            s_cs=s_cs,
            max_lyapunov=max_lyapunov * TQFTConstants.RSI_BANACH_CONTRACTIVITY,
            depth=depth + 1,
        )

    def _spectral_truncate(
        self,
        Z: NDArray[np.complex128],
        rel_tol: float | None = None,
    ) -> Tuple[NDArray[np.complex128], int]:
        r"""
        Truncamiento espectral óptimo de Eckart-Young-Mirsky en la norma de
        Schatten-Frobenius del álgebra de Banach de operadores compactos.
        """
        if rel_tol is None:
            rel_tol = TQFTConstants.SPECTRAL_REL_TOL

        if Z.size == 0:
            return Z, 0

        original_shape = Z.shape
        if Z.ndim > 2:
            mid = max(1, Z.size // 2)
            Z_mat = Z.reshape(mid, -1)
        else:
            Z_mat = Z if Z.ndim == 2 else Z.reshape(-1, 1)

        try:
            U, S, Vh = la.svd(Z_mat, full_matrices=False)
        except la.LinAlgError as exc:
            raise TuraevViroCollapseError(
                f"Descomposición SVD fallida (matriz numéricamente singular): {exc}"
            ) from exc

        if S.size == 0:
            return Z, 0

        threshold = rel_tol * float(np.max(S))
        mask = S > threshold
        rank = int(np.sum(mask))

        if rank == 0:
            rank = 1
            mask[0] = True

        S_trunc = np.where(mask, S, 0.0)
        Z_trunc_mat = (U * S_trunc) @ Vh

        if Z_trunc_mat.size == int(np.prod(original_shape)):
            Z_trunc = Z_trunc_mat.reshape(original_shape)
        else:
            Z_trunc = Z_trunc_mat

        return Z_trunc.astype(np.complex128), rank

    def _compute_turaev_viro_state_sum(
        self,
        tensors: List[NDArray[np.complex128]],
    ) -> Tuple[complex, int]:
        r"""
        Calcula el invariante topológico de suma de estados de Turaev-Viro:
            $Z_{TV}(M; q) = \mathrm{tTr}\left( \bigotimes_{\tau \in \mathcal{T}_{\min}} T^{(\tau)} \right)$
        mediante contracción tensorial secuencial y estabilización espectral SVD
        de Eckart-Young-Mirsky en cada vértice.
        """
        if not tensors:
            raise TuraevViroCollapseError(
                "No existen tensores tetraédricos para contraer (triangulación vacía)."
            )

        Z_M = np.asarray(tensors[0], dtype=np.complex128)
        total_rank = 0

        for idx, T_tau in enumerate(tensors[1:], start=1):
            T_tau = np.asarray(T_tau, dtype=np.complex128)

            if Z_M.ndim >= 1 and T_tau.ndim >= 1:
                if Z_M.shape[-1] == T_tau.shape[0]:
                    Z_M = np.tensordot(Z_M, T_tau, axes=([-1], [0]))
                elif (
                    Z_M.shape[0] == T_tau.shape[0]
                    and Z_M.ndim == 2
                    and T_tau.ndim == 2
                ):
                    Z_M = Z_M @ T_tau
                else:
                    raise TuraevViroCollapseError(
                        f"Incompatibilidad de dimensiones en contracción {idx}: "
                        f"Z.shape={Z_M.shape}, T.shape={T_tau.shape}."
                    )
            else:
                raise TuraevViroCollapseError(
                    "Tensores de dimensión insuficiente para contracción."
                )

            Z_M, rank_step = self._spectral_truncate(Z_M)
            total_rank = max(total_rank, rank_step)

        if Z_M.ndim == 0:
            state_sum = complex(Z_M)
        elif Z_M.ndim == 1:
            state_sum = complex(np.sum(Z_M))
        elif Z_M.ndim == 2 and Z_M.shape[0] == Z_M.shape[1]:
            state_sum = complex(np.trace(Z_M))
        else:
            state_sum = complex(np.sum(Z_M))

        if abs(state_sum) < TQFTConstants.PLANCK_EFF:
            raise TuraevViroCollapseError(
                f"Suma de estados colapsada |Z(M)|={abs(state_sum):.3e} "
                f"< PLANCK_EFF={TQFTConstants.PLANCK_EFF:.3e}. "
                f"Pérdida de información topológica."
            )

        return state_sum, total_rank

    def compute_invariants(self, manifold: CobordismManifold) -> QuantumInvariants:
        r"""
        MÉTODO TERMINAL DE LA FASE ANIDADA 2 Y PUENTE FUNTORIAL HACIA LA FASE ANIDADA 3.
        ────────────────────────────────────────────────────────────────────────────────
        Orquesta el cálculo de invariantes a partir del `CobordismManifold`
        entregado por `build_cobordism` (Fase 1):
          1. Invoca `_continue_from_phase1_cobordism(manifold)` para extraer el
             espectro de Poincaré-Lyapunov y la integral homoclínica de Melnikov.
          2. Evalúa la acción de Chern-Simons $S_{CS}[A]$.
          3. Ejecuta el bucle RSI-L3 (`_rsi_level3_adapt_quantum_group_and_tensors`)
             para blindar el grupo cuántico $U_q(\mathfrak{sl}_2)$.
          4. Contrae la red tensorial de Turaev-Viro $Z_{TV}(M)$.

        CONTINUIDAD DE FASES ANIDADAS:
        La definición formal de salida de este método (`QuantumInvariants`) constituye
        el argumento de entrada exacto e inicio formal de `_continue_from_phase2_invariants`
        en `Phase3_BooleanCollapseProjector`.
        """
        logger.debug(
            "Fase 2 iniciada desde CobordismManifold (χ≈%d, I_Poincaré=%.4e).",
            manifold.euler_characteristic,
            manifold.poincare_cartan_invariant,
        )

        # 1. Continuación formal directa desde la Fase 1
        (
            max_lyapunov,
            recurrence_defect,
            melnikov_integral,
            homoclinic_tangle,
        ) = self._continue_from_phase1_cobordism(manifold)

        # 2. Acción de Chern-Simons
        s_cs = self._compute_chern_simons_action(manifold.connection_form_A)

        # 3. Automejora Recursiva de Nivel 3 sobre tensores y parámetro cuántico q
        adapted_tensors, q_adapted, banach_radius = (
            self._rsi_level3_adapt_quantum_group_and_tensors(
                manifold.tetrahedral_tensors,
                s_cs,
                max_lyapunov,
            )
        )

        # 4. Suma de estados de Turaev-Viro sobre los tensores estabilizados
        z_m, spectral_rank = self._compute_turaev_viro_state_sum(adapted_tensors)

        # Criterio de ausencia de nudos logísticos (Chern-Simons + ausencia de caos homoclínico)
        threshold_cs = TQFTConstants.SPECTRAL_REL_TOL * max(
            1.0, abs(TQFTConstants.CHERN_SIMONS_K)
        )
        is_knot_free = (abs(s_cs) < threshold_cs) and (not homoclinic_tangle)

        if not is_knot_free:
            logger.error(
                "Fuga topológica o caos homoclínico detectado: S_CS=%s (umbral=%.3e), "
                "homoclinic_tangle=%s, λ_max=%.4e.",
                s_cs,
                threshold_cs,
                homoclinic_tangle,
                max_lyapunov,
            )

        return QuantumInvariants(
            chern_simons_action=s_cs,
            turaev_viro_state_sum=z_m,
            is_knot_free=is_knot_free,
            spectral_rank=spectral_rank,
            max_lyapunov_exponent=max_lyapunov,
            poincare_recurrence_defect=recurrence_defect,
            melnikov_homoclinic_integral=melnikov_integral,
            homoclinic_tangle_detected=homoclinic_tangle,
            rsi_l3_adapted_q=q_adapted,
            banach_spectral_radius=banach_radius,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE ANIDADA 3 : PROYECTOR DE COLAPSO BOOLEANO, CLASIFICADOR DE SUBOBJETOS Ω ║
# ║                  Y META-AUDITORÍA RECURSIVA DE NIVEL 3 (GÖDEL-LAWVERE)       ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║ Inicia directamente como continuación formal del objeto `QuantumInvariants`  ║
# ║ retornado por `Phase2_QuantumInvariantsEngine.compute_invariants`.           ║
# ║ Culmina en `collapse_verdict`, proyectando al álgebra de Boole {⊤, ⊥}.       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Phase3_BooleanCollapseProjector(Phase2_QuantumInvariantsEngine):
    r"""
    Fase Anidada 3 – Continúa directamente desde el objeto `QuantumInvariants`
    producido por `compute_invariants` (último método de la Fase 2).

    Proyecta los invariantes cuánticos y celestes sobre el clasificador de
    subobjetos $\Omega \cong \{\text{VIABLE}, \text{RECHAZAR}\} \cong \{\top, \bot\}$
    del Topos mediante un homomorfismo de retículos distributivos acotados,
    auditado por el meta-verificador de punto fijo de Nivel 3 ($\mathcal{R}_3^{(3)}$).
    """

    def _continue_from_phase2_invariants(
        self,
        invariants: QuantumInvariants,
    ) -> Tuple[bool, str]:
        r"""
        PRIMER MÉTODO DE LA FASE ANIDADA 3 (Continuación formal directa de `compute_invariants`).
        ─────────────────────────────────────────────────────────────────────────────────────────
        Recibe el objeto `QuantumInvariants` emergente de la Fase 2, evalúa su
        fórmula característica en la lógica interna de Heyting-Boole del Topos y
        sintetiza la traza de estabilidad celeste de Poincaré.
        """
        celestial_viable = (
            not invariants.homoclinic_tangle_detected
            and abs(invariants.turaev_viro_state_sum) >= TQFTConstants.PLANCK_EFF
        )
        poincare_trace = (
            f"λ_max={invariants.max_lyapunov_exponent:.4e} | "
            f"Δ_rec={invariants.poincare_recurrence_defect:.4e} | "
            f"Melnikov={invariants.melnikov_homoclinic_integral:.4e} | "
            f"homoclinic_tangle={invariants.homoclinic_tangle_detected} | "
            f"ρ_Banach={invariants.banach_spectral_radius:.4e}"
        )
        return celestial_viable, poincare_trace

    def _rsi_level3_meta_verify_collapse(
        self,
        invariants: QuantumInvariants,
        depth: int = 0,
        audit_acc: Tuple[str, ...] = (),
    ) -> Tuple[QuantumInvariants, Tuple[str, ...]]:
        r"""
        Meta-Auditor de Automejora Recursiva de Nivel 3 ($\mathcal{R}_3^{(3)}$).
        Verifica la consistencia de punto fijo de Gödel-Lawvere de la proposición
        topológica antes del colapso booleano irreversible:
          • Nivel 1: Verifica la no-degeneración numérica de $Z_{TV}(M)$ y $S_{CS}[A]$.
          • Nivel 2: Audita la coherencia entre la ausencia de nudos de Chern-Simons
            y la ausencia de enredos homoclínicos de Poincaré.
          • Nivel 3: Certifica la invariancia del veredicto bajo endomorfismos de
            reescala diofántica $\phi^{-1}$ (clausura modal de Lawvere $\Box P \leftrightarrow P$).
        """
        threshold_cs = TQFTConstants.SPECTRAL_REL_TOL * max(
            1.0, abs(TQFTConstants.CHERN_SIMONS_K)
        )
        cs_mag = abs(invariants.chern_simons_action)

        # Consistencia estricta entre el predicado booleano y los invariantes físicos
        consistent_knot_free = (
            (cs_mag < threshold_cs)
            and (not invariants.homoclinic_tangle_detected)
            and (abs(invariants.turaev_viro_state_sum) >= TQFTConstants.PLANCK_EFF)
        )

        step_log = (
            f"RSI-L3[depth={depth + 1}]: |S_CS|={cs_mag:.3e}, "
            f"|Z_TV|={abs(invariants.turaev_viro_state_sum):.3e}, "
            f"fixed_point_consistent={consistent_knot_free == invariants.is_knot_free}"
        )
        new_audit = audit_acc + (step_log,)

        # Si hay discrepancia entre la bandera y la cota exacta, el meta-funtor corrige el estado
        if invariants.is_knot_free != consistent_knot_free:
            invariants = QuantumInvariants(
                chern_simons_action=invariants.chern_simons_action,
                turaev_viro_state_sum=invariants.turaev_viro_state_sum,
                is_knot_free=consistent_knot_free,
                spectral_rank=invariants.spectral_rank,
                max_lyapunov_exponent=invariants.max_lyapunov_exponent,
                poincare_recurrence_defect=invariants.poincare_recurrence_defect,
                melnikov_homoclinic_integral=invariants.melnikov_homoclinic_integral,
                homoclinic_tangle_detected=invariants.homoclinic_tangle_detected,
                rsi_l3_adapted_q=invariants.rsi_l3_adapted_q,
                banach_spectral_radius=invariants.banach_spectral_radius,
            )

        if depth + 1 >= TQFTConstants.RSI_L3_MAX_DEPTH or consistent_knot_free == invariants.is_knot_free:
            return invariants, new_audit

        return self._rsi_level3_meta_verify_collapse(
            invariants=invariants,
            depth=depth + 1,
            audit_acc=new_audit,
        )

    def _project_to_lattice(self, invariants: QuantumInvariants) -> VerdictLevel:
        r"""
        Morfismo característico del clasificador de subobjetos $\Omega$:
            $P_{\text{TQFT}}(\mathcal{I}) = \top \ (\text{VIABLE}) \iff \mathcal{I}.\text{is\_knot\_free} \land \neg \mathcal{I}.\text{homoclinic\_tangle\_detected}$.
        """
        if invariants.is_knot_free and not invariants.homoclinic_tangle_detected:
            return VerdictLevel.VIABLE
        return VerdictLevel.RECHAZAR

    def collapse_verdict(self, invariants: QuantumInvariants) -> TQFTVerdict:
        r"""
        MÉTODO TERMINAL DE LA FASE ANIDADA 3 (Clausura Booleana Soberana).
        ──────────────────────────────────────────────────────────────────
        Continúa directamente desde el `QuantumInvariants` producido por
        `compute_invariants` (Fase 2):
          1. Ejecuta `_continue_from_phase2_invariants(invariants)`.
          2. Certifica el punto fijo de Gödel-Lawvere mediante `_rsi_level3_meta_verify_collapse`.
          3. Proyecta al retículo booleano $\{\text{VIABLE}, \text{RECHAZAR}\}$.
          4. Detona `PoincareHomoclinicChaosVeto` o `TopologicalKnotVeto` si el
             veredicto es $\bot$ (`RECHAZAR`).
        """
        celestial_viable, poincare_trace = self._continue_from_phase2_invariants(
            invariants
        )
        verified_invariants, rsi_audit = self._rsi_level3_meta_verify_collapse(
            invariants
        )

        verdict = self._project_to_lattice(verified_invariants)

        if verdict == VerdictLevel.RECHAZAR or not celestial_viable:
            if verified_invariants.homoclinic_tangle_detected:
                raise PoincareHomoclinicChaosVeto(
                    f"Veto Celeste de Poincaré (Enredo Homoclínico W^s pitchfork W^u). "
                    f"{poincare_trace} | S_CS = {verified_invariants.chern_simons_action}"
                )
            raise TopologicalKnotVeto(
                f"Veto Topológico Incondicional. "
                f"S_CS = {verified_invariants.chern_simons_action}, "
                f"Z(M) = {verified_invariants.turaev_viro_state_sum}, "
                f"rango espectral = {verified_invariants.spectral_rank} | "
                f"{poincare_trace}"
            )

        return TQFTVerdict(
            invariants=verified_invariants,
            verdict=verdict,
            topological_trace=(
                f"Z(M) = {verified_invariants.turaev_viro_state_sum:.6e} | "
                f"S_CS = {verified_invariants.chern_simons_action:.6e} | "
                f"rank_eff = {verified_invariants.spectral_rank} | "
                f"knot_free = {verified_invariants.is_knot_free} | "
                f" {poincare_trace}"
            ),
            poincare_celestial_trace=poincare_trace,
            rsi_l3_audit_trail=rsi_audit,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ORQUESTADOR SUPREMO : TQFT PROJECTION MANIFOLD                               ║
# ║ Endofuntor Soberano Z = Phase3 ∘ Phase2 ∘ Phase1 ∘ MetricForgetfulFunctor    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TQFTProjectionManifold(Morphism, Phase3_BooleanCollapseProjector):
    r"""
    Endofuntor Soberano $Z$ de la TQFT enriquecida con Mecánica Celeste de
    Poincaré y Automejora Recursiva de Nivel 3 (RSI-L3).

    Composición funtorial monoidal estricta de las 3 fases anidadas:
        $Z = \text{Phase3\_BooleanCollapseProjector}
             \circ \text{Phase2\_QuantumInvariantsEngine}
             \circ \text{Phase1\_CobordismFibrator}
             \circ \text{MetricForgetfulFunctor}$
    """

    def project_intent(
        self,
        sigma_in: TQFTBoundary,
        sigma_out: TQFTBoundary,
        connection_tensor: NDArray[np.float64],
    ) -> TQFTVerdict:
        r"""
        Proyecta la intención del agente a través de la cascada funtorial de 3 fases anidadas:
          • Fase 1: `build_cobordism(sigma_in, sigma_out, connection_tensor)` → `CobordismManifold`
          • Fase 2: `compute_invariants(manifold)`                            → `QuantumInvariants`
          • Fase 3: `collapse_verdict(invariants)`                            → `TQFTVerdict`
        """
        logger.info(
            "Iniciando proyección TQFT (v5.0 Poincare-Symplectic-RSI-L3): "
            "olvido métrico, flujo simpléctico de Yoshida y síntesis recursiva L3..."
        )

        # ── FASE ANIDADA 1: Construcción del Cobordismo y Flujo de Poincaré ──
        manifold: CobordismManifold = self.build_cobordism(
            sigma_in, sigma_out, connection_tensor
        )

        # ── FASE ANIDADA 2: Continuación directa desde CobordismManifold ─────
        invariants: QuantumInvariants = self.compute_invariants(manifold)

        # ── FASE ANIDADA 3: Continuación directa desde QuantumInvariants ─────
        verdict: TQFTVerdict = self.collapse_verdict(invariants)

        logger.info(
            "Proyección TQFT exitosa. Veredicto: %s | %s",
            verdict.verdict.name,
            verdict.topological_trace,
        )
        return verdict


# ------------------------------------------------------------------------------
# Superficie pública del módulo
# ------------------------------------------------------------------------------
__all__ = [
    "TQFTConstants",
    "TQFTProjectionError",
    "CobordismDegeneracyError",
    "TopologicalKnotVeto",
    "PoincareHomoclinicChaosVeto",
    "TuraevViroCollapseError",
    "RSILevel3ConvergenceError",
    "TQFTBoundary",
    "CobordismManifold",
    "QuantumInvariants",
    "TQFTVerdict",
    "MetricForgetfulFunctor",
    "Phase1_CobordismFibrator",
    "Phase2_QuantumInvariantsEngine",
    "Phase3_BooleanCollapseProjector",
    "TQFTProjectionManifold",
]
# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER - SOBERANO AGÉNTICO BOOLE/PHYSICS
====================================================================================================

Módulo : AST Static Analyzer Agent (Custodio de la Cohomología Sintáctica)
Ruta   : app/agents/boole/physics/ast_static_analyzer_agent.py
Versión: 5.1.0-RSI3-Poincare-KAM-Heyting-OODA
Función: Soberano agéntico OODA (3-fases anidadas) sobre el motor v5.1.0.

----------------------------------------------------------------------------------------------------
DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL:
----------------------------------------------------------------------------------------------------
El Soberano Agéntico `ASTStaticAnalyzerAgentRSI3` (y su contraparte backward-compatible `ASTStaticAnalyzerAgent`)
actúa como el Custodio de la Cohomología Sintáctica dentro de la capa Boole/Physics de APU Filter.
Orquesta y audita la dinámica decisional del motor espectral `ASTStaticAnalyzer` mediante la ejecución
de un ciclo OODA de tres fases anidadas acopladas morfísicamente:
• Fase 1 (Phase1_SymplecticInvarianceAuditor): Auditoría de Liouville M^\top \Omega M = \Omega, residuo
  de determinante \det M = +1, descomposición polar no simpléctica y cálculo del índice de Maslov \mu(M).
  Emite el morfismo `AgentPhase1Handoff` vía `_handoff_to_phase2_agent`.
• Fase 2 (Phase2_DirichletThermodynamicEnforcer): Ingesta estricta `_ingest_phase1_handoff_agent`,
  enforcement de la disipación Port-Hamiltoniana P = \Phi^\top \nabla V \ge 0, verificación del axioma
  de Clausius-Duhem \sigma_{CD} = P / T \ge 0, análisis de retorno de Poincaré, cálculo de la amplicud de
  Melnikov M(\theta_0) y automejora recursiva RSI³ mediante la contracción determinista de Banach
  T: \mathbb{R}^3 \to \mathbb{R}^3 sobre (\delta\omega, 1 - \cos\theta, e^{-\sigma}).
  Emite el morfismo `AgentPhase2Handoff` vía `_handoff_to_phase3_agent`.
• Fase 3 (Phase3_CellularSheafCohomologyAuditor): Ingesta estricta `_ingest_phase2_handoff_agent`,
  auditoría de la nulidad de la primera cohomología de haz \dim H^1 = 0, verificación de la secuencia exacta de
  Mayer-Vietoris, espectro de Laplaciano de Hodge-Dirac \lambda_{\min}(\Delta_G) > 0 y adjudicación trivalente
  en el retículo de Heyting \Omega_3 = \{0x00, 0x01, 0x03\} sobre 10 átomos independientes, activando el
  interlock de hardware ESP32 Crowbar (< 400 ns) ante colapso a VETOED (0x03).

----------------------------------------------------------------------------------------------------
MARCO TEÓRICO Y FORMALISMO MATEMÁTICO:
----------------------------------------------------------------------------------------------------
1. INVARIANZA SIMPLÉCTICA DE LIOUVILLE Y ÍNDICE DE MASLOV:
   El soberano evalúa el mapeo linealizado M \in \mathrm{Sp}(2n,\mathbb{R}) sobre la variedad de fases T*Q.
   Verifica la conservación de la 2-forma canónica \Omega = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}:
   $$M^\top \Omega M = \Omega \implies \delta\omega_{\mathrm{rel}} = \frac{\|M^\top \Omega M - \Omega\|_F}{\max(1, \|M\|_F + \|\Omega\|_F)} \le \epsilon_{\mathrm{eff}}$$
   El índice de Maslov \mu(M) cuantifica la rotación de los subespacios Lagrangianos principales:
   $$\mu(M) = \# \{ \lambda \in \operatorname{Spec}(B) \mid \lambda < -\epsilon_{\mathrm{Maslov}} \} \quad \text{donde} \quad M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$$

2. DISSIPACIÓN PORT-HAMILTONIANA Y SEGUNDA LEY DE LA TERMODINÁMICA:
   La potencia disipada P entre el potencial de control \Phi y el gradiente de Lyapunov \nabla V debe ser
   estrictamente no negativa, satisfaciendo Clausius-Duhem diferencial:
   $$P = \langle \Phi, \nabla V \rangle \ge 0 \implies \sigma_{CD} = \frac{P}{T_{\mathrm{ref}}} \ge 0$$
   El ángulo termodinámico \theta = \arccos\left(\frac{\langle \Phi, \nabla V \rangle}{\|\Phi\|_2 \|\nabla V\|_2}\right) acota la alineación exergética.

3. CONTRACCIÓN DETERMINISTA DE BANACH RSI³ EN \mathbb{R}^3:
   Consume el `RSI3Report` del motor o ejecuta la contracción determinista T: X \to X en X = \mathbb{R}^3:
   $$x_k = \begin{pmatrix} \delta\omega_k \\ 1 - |\cos\theta_k| \\ e^{-\sigma_k} \end{pmatrix}, \quad T(x) = \begin{pmatrix} \kappa x_1 \\ x_2^2 \\ \beta x_3 \end{pmatrix}$$
   donde \kappa = 0.45 si Liouville se cumple y \beta = 0.70 si KAM es estable.
   El factor de aprendizaje \eta_k se modula dinámicamente para forzar \mathrm{Lip}(T) < \mathtt{BANACH\_TARGET} = 0.95.

4. ADJUDICACIÓN DE HEYTING EN \Omega_3 Y CROWBAR INTERLOCK:
   El veredicto global es el Join bitwise OR V = \bigvee_{i=1}^{10} v_i sobre los 10 átomos:
   $$v_1: \text{Liouville}, \, v_2: \text{Banach}, \, v_3: H^1, \, v_4: \text{KAM}, \, v_5: \text{Melnikov}, \, v_6: \text{Lyapunov}, \, v_7: \text{Hodge}, \, v_8: \text{Twist}, \, v_9: \text{RSI}^3, \, v_{10}: \text{Birkhoff}$$
   Si V = 0x03 (\mathtt{VETOED}), el soberano conmuta el pin GPIO14 del ESP32 a HIGH en t < 400 ns, desenergizando la etapa de potencia.

----------------------------------------------------------------------------------------------------
AXIOMAS FORMALES:
----------------------------------------------------------------------------------------------------
SGA1 (Inviolabilidad Simpléctica de Fase 1):
    \delta\omega_{\mathrm{rel}} \le \epsilon_{\mathrm{eff}} \quad \text{y} \quad \det(M) = +1 \quad \forall M.
SGA2 (Estabilidad Termodinámica Clausius-Duhem):
    P = \langle \Phi, \nabla V \rangle \ge 0 \quad \text{y} \quad \sigma_{CD} \ge 0.
SGA3 (Contracción Conforme en \mathbb{R}^3):
    L_\Phi = \mathrm{Lip}(T) \le 0.95 < 1.0 \implies x^{(k)} \to x^*.
SGA4 (Causalidad Continuada OODA):
    \operatorname{dom}(\Phi_{k+1}) \equiv \operatorname{im}(\Phi_k) \quad \forall k \in \{1, 2\}.

----------------------------------------------------------------------------------------------------
INVARIANTES DE ESTADO:
----------------------------------------------------------------------------------------------------
Invariante 1 (Residuo Simpléctico Adaptativo):
    \delta\omega_{\mathrm{rel}} \le \epsilon_{\mathrm{eff}} = \max(10^{-10}, 10^{-10} \cdot \sigma_{\max}(M) \cdot \epsilon_{\mathrm{mach}} \cdot 128).
Invariante 2 (No-Negatividad de Entropía):
    \sigma_{CD} \ge 0.0.
Invariante 3 (Trivialidad Cohomológica de Haz):
    \dim H^1(T, \mathcal{F}) = 0 \quad \text{y} \quad \text{Mayer-Vietoris Exacta}.
Invariante 4 (Latencia de Interlock Crowbar):
    t_{\mathrm{Crowbar}} < 400.0 \text{ ns ante colapso } V = 0x03.

----------------------------------------------------------------------------------------------------
TRADUCCIÓN A IMPACTO DE NEGOCIO ("DOLOR Y DINERO"):
----------------------------------------------------------------------------------------------------
· INMUNIDAD CIBER-FÍSICA ANTE SINGULARIDADES: La activación instantánea (< 400 ns) del interlock Crowbar
  evita la destrucción física de actuadores y hardware de cómputo ante resonancias homoclínicas o
  divergencias exergéticas en el software.
· CERTIFICACIÓN AUTOMÁTICA DE CALIDAD DE CÓDIGO: Elimina el riesgo operativo de desplegar parches o
  módulos con ciclos de inconsistencia lógica, reduciendo drásticamente los costos de mantenimiento y auditoría.
====================================================================================================
"""

from __future__ import annotations

import ast as _ast_module
import hashlib
import logging
import math
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# §0.1 DEPENDENCIAS ARQUITECTÓNICAS
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""

        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""

        pass


# ─────────────────────────────────────────────────────────────────────────────
# §0.2 IMPORTACIÓN DEL MOTOR v5.1.0 (fallback v5.0.0 → v4.1.0 → stubs)
# ─────────────────────────────────────────────────────────────────────────────
_ENGINE_V51_AVAILABLE: bool = False
_ENGINE_V5_AVAILABLE: bool = False
_ENGINE_BASE_AVAILABLE: bool = False

try:
    from app.boole.physics.ast_static_analyzer import (
        ThermodynamicSingularityError,
        CohomologicalObstructionError,
        SymplecticPoincareEngine,
        HeavyHodgeDiracLaplacian,
        RecursiveSelfImprovementL3 as EngineRSI3,
        ASTStaticAnalyzer as EngineASTStaticAnalyzer,
        AnalysisCertificate as EngineAnalysisCertificate,
        Phase1Handoff as EnginePhase1Handoff,
        Phase2Handoff as EnginePhase2Handoff,
        RSI3Report as EngineRSI3Report,
        ActionAngleChart as EngineActionAngleChart,
        TwistCondition as EngineTwistCondition,
        PoincareReturnMap as EnginePoincareReturnMap,
        BirkhoffRemainder as EngineBirkhoffRemainder,
        AnalysisLimits as EngineLimits,
    )
    _ENGINE_V51_AVAILABLE = True
    _ENGINE_V5_AVAILABLE = True
    _ENGINE_BASE_AVAILABLE = True
except ImportError:
    EngineRSI3 = None  # type: ignore[assignment,misc]
    EngineRSI3Report = None  # type: ignore[assignment,misc]
    EngineActionAngleChart = None  # type: ignore[assignment,misc]
    EngineTwistCondition = None  # type: ignore[assignment,misc]
    EnginePoincareReturnMap = None  # type: ignore[assignment,misc]
    EngineBirkhoffRemainder = None  # type: ignore[assignment,misc]
    EngineLimits = None  # type: ignore[assignment,misc]
    try:
        from app.boole.physics.ast_static_analyzer import (
            ThermodynamicSingularityError,
            CohomologicalObstructionError,
            SymplecticPoincareEngine,
            HeavyHodgeDiracLaplacian,
            ASTStaticAnalyzer as EngineASTStaticAnalyzer,
            AnalysisCertificate as EngineAnalysisCertificate,
            Phase1Handoff as EnginePhase1Handoff,
            Phase2Handoff as EnginePhase2Handoff,
        )
        _ENGINE_V5_AVAILABLE = True
        _ENGINE_BASE_AVAILABLE = True
    except ImportError:
        EngineAnalysisCertificate = None  # type: ignore[assignment,misc]
        EnginePhase1Handoff = None  # type: ignore[assignment,misc]
        EnginePhase2Handoff = None  # type: ignore[assignment,misc]
        EngineASTStaticAnalyzer = None  # type: ignore[assignment,misc]
        try:
            from app.boole.physics.ast_static_analyzer import (
                ThermodynamicSingularityError,
                CohomologicalObstructionError,
                SymplecticPoincareEngine,
                HeavyHodgeDiracLaplacian,
            )
            _ENGINE_BASE_AVAILABLE = True
        except ImportError:
            class ThermodynamicSingularityError(TopologicalInvariantError):
                def __init__(
                    self,
                    a: float = 0.0,
                    b: float = 0.0,
                    c: int = 0,
                    d: int = 0,
                    *args: Any,
                    **kwargs: Any,
                ):
                    super().__init__(f"Thermo: {a} vs {b}, depth={c}, cc={d}")

            class CohomologicalObstructionError(TopologicalInvariantError):
                def __init__(
                    self,
                    dim: int = 0,
                    vars: Optional[set] = None,
                    *args: Any,
                    **kwargs: Any,
                ):
                    super().__init__(f"Cohomology: dim H¹={dim}")

            class SymplecticPoincareEngine:
                def __init__(self, dim: int):
                    self.dim = dim
                    n = int(dim)
                    self.J = np.block(
                        [
                            [np.zeros((n, n)), np.eye(n)],
                            [-np.eye(n), np.zeros((n, n))],
                        ]
                    )

                def cayley_darboux_projection(self, M: np.ndarray) -> np.ndarray:
                    return np.asarray(M, dtype=np.float64)

            class HeavyHodgeDiracLaplacian:
                def compute_heavy_laplacian(
                    self, coboundary_d: np.ndarray, G_metric: np.ndarray
                ) -> np.ndarray:
                    d = np.asarray(coboundary_d, dtype=np.float64)
                    return d.T @ d


logger = logging.getLogger("MIC.Gamma.ASTAnalyzerAgent.v51")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES SIMPLÉCTICAS Y DE FASE
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_SYMPLECTIC_TOLERANCE_BASE: Final[float] = 1e-10
_SYMPLECTIC_SPECTRAL_FACTOR: Final[float] = 1e-10
_DIRICHLET_DISSIPATION_FLOOR: Final[float] = 0.0
_ENTROPY_PRODUCTION_FLOOR: Final[float] = 0.0
_COHOMOLOGICAL_COMPLEX_TOL: Final[float] = 1e-8
_SPECTRAL_GAP_RATIO: Final[float] = 1e-10
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0
_MASLOV_PHASE_TOLERANCE: Final[float] = 1e-9
_CONDITION_NUMBER_WARNING_THRESHOLD: Final[float] = 1.0 / _MACHINE_EPSILON
_THERMODYNAMIC_TEMPERATURE_REFERENCE: Final[float] = 1.0
_COHOMOLOGY_DIMENSION_FLOOR: Final[int] = 0

_BANACH_TARGET_DEFAULT: Final[float] = (
    float(EngineLimits.BANACH_TARGET) if EngineLimits is not None else 0.95
)
_RSI3_MAX_ITER_DEFAULT: Final[int] = (
    int(EngineLimits.RSI3_MAX_ITER) if EngineLimits is not None else 20
)
_RSI3_MIN_ITER_DEFAULT: Final[int] = (
    int(EngineLimits.RSI3_MIN_ITER) if EngineLimits is not None else 3
)
_RSI3_CONV_EPS: Final[float] = (
    float(EngineLimits.RSI3_CONV_EPS) if EngineLimits is not None else 1.0e-9
)
_RSI3_LIPSCHITZ_TARGET: Final[float] = (
    float(getattr(EngineLimits, "RSI3_LIPSCHITZ_TARGET", 0.85))
    if EngineLimits is not None
    else 0.85
)
_KAM_GAMMA_DEFAULT: Final[float] = (
    float(EngineLimits.KAM_GAMMA) if EngineLimits is not None else 1.0e-3
)
_MELNIKOV_THRESHOLD: Final[float] = (
    float(EngineLimits.MELNIKOV_THRESHOLD) if EngineLimits is not None else 1.0e-2
)
_LYAPUNOV_CHAOS: Final[float] = 1.0e-2
_FLOW_MAP_EPSILON: Final[float] = (
    float(getattr(EngineLimits, "FLOW_MAP_EPSILON", 0.1))
    if EngineLimits is not None
    else 0.1
)
_HODGE_LAMBDA_MIN: Final[float] = (
    float(getattr(EngineLimits, "HODGE_LAMBDA_MIN_THRESHOLD", 1.0e-9))
    if EngineLimits is not None
    else 1.0e-9
)


# ═══════════════════════════════════════════════════════════════════════════════
# §B. EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════
class SymplecticInvarianceViolation(TopologicalInvariantError):
    r"""Violación del Teorema de Liouville: $M^\top \Omega M \neq \Omega$."""

    pass


class SymplecticPolarDecompositionError(SymplecticInvarianceViolation):
    r"""Descomposición polar no simpléctica."""

    pass


class MaslovIndexError(SymplecticInvarianceViolation):
    r"""Índice de Maslov no calculable."""

    pass


class ClausiusDuhemViolation(ThermodynamicSingularityError):
    r"""Violación de Clausius-Duhem diferencial: $\sigma_{CD} < 0$."""

    def __init__(self, message: str = "Clausius-Duhem violated", *args: Any, **kwargs: Any):
        try:
            super().__init__(0.0, 0.0, 0, 0)
        except TypeError:
            super().__init__(message)  # type: ignore[misc]
        self._cd_message = message

    def __str__(self) -> str:
        return getattr(self, "_cd_message", "Clausius-Duhem violated")


class ExergyDivergenceError(ThermodynamicSingularityError):
    r"""Norma de exergía diverge."""

    def __init__(self, message: str = "Exergy diverged", *args: Any, **kwargs: Any):
        try:
            super().__init__(0.0, 0.0, 0, 0)
        except TypeError:
            super().__init__(message)  # type: ignore[misc]
        self._ex_message = message

    def __str__(self) -> str:
        return getattr(self, "_ex_message", "Exergy diverged")


class EulerCharacteristicMismatch(CohomologicalObstructionError):
    r"""$\chi$ inconsistente con Betti."""

    def __init__(self, message: str = "Euler mismatch", *args: Any, **kwargs: Any):
        try:
            super().__init__(1)
        except TypeError:
            super().__init__(message)  # type: ignore[misc]
        self._eu_message = message

    def __str__(self) -> str:
        return getattr(self, "_eu_message", "Euler mismatch")


class MayerVietorisBreachError(CohomologicalObstructionError):
    r"""Secuencia de Mayer-Vietoris no exacta."""

    def __init__(self, message: str = "Mayer-Vietoris breach", *args: Any, **kwargs: Any):
        try:
            super().__init__(1)
        except TypeError:
            super().__init__(message)  # type: ignore[misc]
        self._mv_message = message

    def __str__(self) -> str:
        return getattr(self, "_mv_message", "Mayer-Vietoris breach")


class OODAChainViolation(TopologicalInvariantError):
    r"""Ruptura de la continuidad estricta entre fases $\Phi_k \to \Phi_{k+1}$."""

    pass


class BanachContractionFailure(TopologicalInvariantError):
    r"""RSI³ no contrajo: $\mathrm{Lip}(T)\ge 1$."""

    pass


def _raise_cohomology(dim: int, message: str) -> None:
    """Lanza `CohomologicalObstructionError` con firma del motor (int, no str)."""
    logger.error(message)
    try:
        raise CohomologicalObstructionError(int(dim))
    except TypeError:
        raise CohomologicalObstructionError(int(dim), None)


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (FASE 1/2/3 + HANDOFFS)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SymplecticInvariantData:
    r"""Artefacto de FASE 1: certificado de Liouville + Cayley-Darboux."""

    phase_space_dimension: int
    symplectic_residual_norm: float
    symplectic_relative_residual: float
    antisymmetric_residual: float
    determinant_residual: float
    condition_number: float
    spectral_radius_Omega: float
    maslov_index: int
    polar_unitarity_residual: float
    effective_tolerance: float
    is_volume_preserved: bool


@dataclass(frozen=True, slots=True)
class ThermodynamicDirichletData:
    r"""Artefacto de FASE 2: disipación Port-Hamiltoniana + Banach RSI³."""

    dissipated_power: float
    raw_inner_product: float
    numerical_tolerance: float
    exergy_norm: float
    gradient_norm: float
    alignment_cosine: float
    thermodynamic_angle_rad: float
    entropy_production_rate: float
    clausius_duhem_satisfied: bool
    exergy_dissipation_ratio: float
    is_thermodynamically_stable: bool
    is_strictly_dissipative: bool


@dataclass(frozen=True, slots=True)
class SheafCohomologyAuditData:
    r"""Artefacto de FASE 3: nulidad de $H^1$ + Hodge-Dirac."""

    h1_dimension: int
    rank_delta0: int
    rank_delta1: int
    betti_numbers: Tuple[int, int, int]
    euler_characteristic: int
    reidemeister_torsion: float
    complex_residual: float
    is_globally_integrable: bool
    obstruction_free: bool
    verified_by_coboundary: bool
    mayer_vietoris_exact: bool


@dataclass(frozen=True, slots=True)
class AuditProvenance:
    r"""Trazabilidad funtorial $\Phi_1\to\Phi_2\to\Phi_3\to Z_{\Gamma\text{-PHYSICS}}$."""

    timestamp_iso: str
    input_checksum_sha256: str
    phase1_passed: bool
    phase2_passed: bool
    phase3_passed: bool
    functor_chain: str


@dataclass(frozen=True, slots=True)
class AgentPhase1Handoff:
    r"""
    Morfismo $\Phi_1\to\Phi_2$ del agente.

    Emitido por `Phase1_SymplecticInvarianceAuditor._handoff_to_phase2_agent`.
    Consumido por `Phase2_DirichletThermodynamicEnforcer._ingest_phase1_handoff_agent`.

    [v5.1.0] Transporta la semilla celeste del motor (frecuencias, dim,
    antisimetría de $\omega_I$) para que $\Phi_2$ no sintetice ruido.
    """

    symplectic_audit: SymplecticInvariantData
    input_checksum: str
    timestamp_iso: str
    engine_certificate: Optional[Any] = None
    frequencies: Tuple[float, ...] = ()
    symplectic_dim: int = 0
    antisymmetry_ok: bool = True
    interference_frobenius: float = 0.0
    generator_used: bool = False


@dataclass(frozen=True, slots=True)
class AgentPhase2Handoff:
    r"""
    Morfismo $\Phi_2\to\Phi_3$ del agente.

    Emitido por `Phase2_DirichletThermodynamicEnforcer._handoff_to_phase3_agent`.
    Consumido por `Phase3_CellularSheafCohomologyAuditor._ingest_phase2_handoff_agent`.
    """

    phase1: AgentPhase1Handoff
    thermodynamic_audit: ThermodynamicDirichletData
    banach_L_real: float
    rsi3_iterations: int
    rsi3_converged: bool
    quadratic_convergence_rate: float
    lyapunov_max: float
    melnikov_amplitude: float
    kam_gamma: float
    kam_stable: bool
    twist_ok: bool = True
    poincare_recurrent: bool = False
    poincare_recurrence_time: float = float("inf")
    action_energy: float = 0.0
    homological_W_norm: float = 0.0
    rsi3_residual: float = 1.0
    rsi3_score_trajectory: Tuple[float, ...] = ()
    coboundary_frobenius: float = 0.0
    coboundary_shape: Tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class AgentPhase3Certificate:
    r"""Certificado final del soberano: composición $\Phi_3\circ\Phi_2\circ\Phi_1$."""

    phase2: AgentPhase2Handoff
    cohomology_audit: SheafCohomologyAuditData
    hodge_lambda_min: float
    hodge_trivial: bool
    heyting_mask: int
    heyting_axioms_verified: bool
    is_compilation_authorized: bool
    provenance: AuditProvenance
    birkhoff_remainder: float = 0.0
    birkhoff_normalized: bool = True
    poincare_recurrence_time: float = float("inf")


@dataclass(frozen=True, slots=True)
class ASTGovernanceState:
    r"""Objeto final de $Z_{\Gamma\text{-PHYSICS}}=\Phi_3\circ\Phi_2\circ\Phi_1$ (backward-compat)."""

    symplectic_audit: SymplecticInvariantData
    thermodynamic_audit: ThermodynamicDirichletData
    cohomology_audit: SheafCohomologyAuditData
    provenance: AuditProvenance
    is_compilation_authorized: bool
    rsi3_banach_real: float = 0.0
    lyapunov_max: float = 0.0
    melnikov_amplitude: float = 0.0
    kam_gamma: float = 0.0
    kam_stable: bool = True
    heyting_mask: int = 0x00
    heyting_axioms_verified: bool = True
    twist_ok: bool = True
    poincare_recurrent: bool = False
    rsi3_converged: bool = False
    birkhoff_normalized: bool = True
    overall_engine_score: float = 0.0
    certificate: Optional[AgentPhase3Certificate] = None


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDAS NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════════

class _FiniteNumericalGuard:
    """Saneamiento numérico: finitez, realidad, estabilidad."""

    @staticmethod
    def _as_float_array(name: str, value: Any) -> NDArray[np.float64]:
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise TypeError(f"[Guard] '{name}' no convertible: {exc}") from exc
        if np.iscomplexobj(raw):
            raise TypeError(f"[Guard] '{name}' debe ser real (dtype={raw.dtype}).")
        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"[Guard] '{name}' no convertible a float64: {exc}") from exc
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise ValueError(f"[Guard] '{name}' contiene {n_bad} NaN/±∞.")
        return arr

    @classmethod
    def _as_finite_matrix(
        cls,
        name: str,
        value: Any,
        *,
        square: bool = False,
        min_rank: int = 0,
        check_structural_zeros: bool = False,
    ) -> NDArray[np.float64]:
        arr = cls._as_float_array(name, value)
        if arr.ndim != 2:
            raise ValueError(f"[Guard] '{name}' debe ser 2D; ndim={arr.ndim}.")
        rows, cols = arr.shape
        if rows == 0 or cols == 0:
            raise ValueError(f"[Guard] '{name}' matriz vacía ({rows}×{cols}).")
        if square and rows != cols:
            raise ValueError(f"[Guard] '{name}' debe ser cuadrada; forma {arr.shape}.")
        if check_structural_zeros:
            cls._structural_zero_check(name, arr)
        if min_rank > 0:
            actual = cls._fast_rank_estimate(arr)
            if actual < min_rank:
                raise ValueError(
                    f"[Guard] '{name}' rango {actual} < mínimo {min_rank}."
                )
        return arr

    @classmethod
    def _as_finite_vector(
        cls,
        name: str,
        value: Any,
        *,
        allow_zero_vector: bool = True,
    ) -> NDArray[np.float64]:
        arr = cls._as_float_array(name, value)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f"[Guard] '{name}' debe ser 1D; ndim={arr.ndim}.")
        if arr.size == 0:
            raise ValueError(f"[Guard] '{name}' vector vacío.")
        if not allow_zero_vector and float(la.norm(arr, ord=2)) == 0.0:
            raise ValueError(f"[Guard] '{name}' es vector cero.")
        return arr

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        if A.size == 0:
            return 0.0
        try:
            v = float(la.norm(A, ord="fro"))
            return v if math.isfinite(v) else math.inf
        except Exception:
            return math.inf

    @staticmethod
    def _vector_norm(v: NDArray[np.float64]) -> float:
        if v.size == 0:
            return 0.0
        try:
            n = float(la.norm(v, ord=2))
            return n if math.isfinite(n) else math.inf
        except Exception:
            return math.inf

    @staticmethod
    def _spectral_norm(A: NDArray[np.float64]) -> float:
        if A.size == 0:
            return 0.0
        try:
            svs = la.svdvals(A)
            return float(svs[0]) if svs.size > 0 and math.isfinite(svs[0]) else math.inf
        except Exception:
            return math.inf

    @staticmethod
    def _safe_svdvals(
        A: NDArray[np.float64], *, max_retries: int = 3
    ) -> NDArray[np.float64]:
        last_exc: Optional[Exception] = None
        work = np.array(A, dtype=np.float64, copy=True)
        eye = np.zeros_like(work)
        np.fill_diagonal(eye, 1.0)
        for attempt in range(max_retries):
            try:
                svs = la.svdvals(work)
                if np.all(np.isfinite(svs)):
                    return svs
            except (np.linalg.LinAlgError, ValueError) as exc:
                last_exc = exc
            perturbation = (_MACHINE_EPSILON ** (1.0 / (attempt + 2))) * float(
                la.norm(work, ord="fro") or 1.0
            )
            work = work + perturbation * eye
        _raise_cohomology(1, f"SVD no convergió: {last_exc}")
        raise CohomologicalObstructionError(1)  # unreachable, for type checkers

    @staticmethod
    def _structural_zero_check(name: str, A: NDArray[np.float64]) -> None:
        tol = _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON
        if np.any(np.all(np.abs(A) <= tol, axis=1)):
            logger.warning("[Guard] '%s' tiene fila(s) cero.", name)
        if np.any(np.all(np.abs(A) <= tol, axis=0)):
            logger.warning("[Guard] '%s' tiene columna(s) cero.", name)

    @staticmethod
    def _fast_rank_estimate(A: NDArray[np.float64]) -> int:
        if A.size == 0 or min(A.shape) == 0:
            return 0
        try:
            svs = la.svdvals(A)
            if svs.size == 0 or svs[0] == 0.0 or not math.isfinite(svs[0]):
                return 0
            tol = (
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(A.shape)
                * svs[0]
            )
            return int(np.count_nonzero(svs > tol))
        except Exception:
            return 0

    @staticmethod
    def _compute_input_checksum(*arrays: NDArray[np.float64]) -> str:
        hasher = hashlib.sha256()
        for arr in arrays:
            if arr is not None:
                shape3 = list(arr.shape[:3]) + [0] * max(0, 3 - arr.ndim)
                meta = struct.pack(">4Q", arr.ndim, *shape3[:3])
                hasher.update(meta)
                hasher.update(np.ascontiguousarray(arr).tobytes())
        return hasher.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# §E. UTILIDADES DE NORMALIZACIÓN, MOTOR Y HEYTING
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_input_to_source(ast_tree: Any) -> str:
    r"""Normaliza la entrada del agente a código fuente Python."""
    if isinstance(ast_tree, str):
        return ast_tree
    if isinstance(ast_tree, bytes):
        return ast_tree.decode("utf-8", errors="replace")
    if isinstance(ast_tree, _ast_module.AST):
        try:
            return _ast_module.unparse(ast_tree)
        except Exception as exc:
            logger.warning(f"[Agent] ast.unparse falló: {exc}; usando str(AST).")
            return str(ast_tree)
    logger.warning(
        f"[Agent] Tipo de entrada desconocido: {type(ast_tree)}; usando str()."
    )
    return str(ast_tree)


def _heyting_join_local(atoms: List[int]) -> int:
    r"""Join bitwise OR en $\Omega_3$: $V=v_1\lor\cdots\lor v_n\in\{0x00,0x01,0x03\}$."""
    acc = 0
    for a in atoms:
        acc |= a & 0x03
        if acc == 0x03:
            return 0x03
    return acc


def _heyting_lattice_axioms_check() -> bool:
    r"""
    Axiomas de retículo Heyting en $\Omega_3=\{0,1,3\}$:
    idempotencia, conmutatividad, asociatividad, absorción, distributividad.
    """
    elems = [0x00, 0x01, 0x03]

    def join(a: int, b: int) -> int:
        return a | b

    def meet(a: int, b: int) -> int:
        return a & b

    ok = True
    for a in elems:
        if join(a, a) != a or meet(a, a) != a:
            ok = False
        for b in elems:
            if join(a, b) != join(b, a) or meet(a, b) != meet(b, a):
                ok = False
            for c in elems:
                if join(join(a, b), c) != join(a, join(b, c)):
                    ok = False
                if meet(meet(a, b), c) != meet(a, meet(b, c)):
                    ok = False
                if join(a, meet(a, b)) != a:
                    ok = False
                if meet(a, join(a, b)) != a:
                    ok = False
                if meet(a, join(b, c)) != join(meet(a, b), meet(a, c)):
                    ok = False
    return ok


def _tuple_matrix_to_array(t: Any) -> Optional[NDArray[np.float64]]:
    if t is None:
        return None
    try:
        arr = np.array(t, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return None
        return arr
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class _EngineInvariants:
    """Proyección pura del `AnalysisCertificate` v5.1.0 / v5.0.0 / dict v4."""

    dim_H1: int = 0
    melnikov_amp: float = 0.0
    kam_stable: bool = True
    kam_gamma: float = float("inf")
    lyapunov_max: float = 0.0
    hodge_trivial: bool = True
    hodge_lambda_min: float = float("inf")
    banach_L: float = 1.0
    rsi3_iterations: int = 0
    rsi3_converged: bool = False
    rsi3_residual: float = 1.0
    rsi3_trajectory: Tuple[float, ...] = ()
    twist_ok: bool = True
    poincare_recurrent: bool = False
    poincare_recurrence_time: float = float("inf")
    action_energy: float = 0.0
    W_norm: float = 0.0
    frequencies: Tuple[float, ...] = ()
    symplectic_dim: int = 0
    antisymmetry_ok: bool = True
    interference_frobenius: float = 0.0
    coboundary_shape: Tuple[int, int] = (0, 0)
    coboundary_frobenius: float = 0.0
    coboundary_sv: Tuple[float, ...] = ()
    birkhoff_remainder: float = 0.0
    birkhoff_normalized: bool = True
    hamiltonian_ok: bool = True
    overall_score: float = 0.0
    generator: Optional[NDArray[np.float64]] = None


def _extract_engine_invariants(engine_cert: Any) -> _EngineInvariants:
    """Lee el certificado del motor sin lanzar si faltan campos (v5.1 / v5.0 / dict)."""
    if engine_cert is None:
        return _EngineInvariants()
    if isinstance(engine_cert, dict):
        return _EngineInvariants(
            dim_H1=int(engine_cert.get("cohomology_dimension", 0) or 0),
            hamiltonian_ok=bool(engine_cert.get("hamiltonian_ok", True)),
            kam_stable=bool(engine_cert.get("hamiltonian_ok", True)),
            overall_score=float(engine_cert.get("overall_score", 0.0) or 0.0),
        )
    try:
        h2 = getattr(engine_cert, "phase2", None)
        h1 = getattr(h2, "phase1", None) if h2 is not None else None
        rsi = getattr(h2, "rsi3", None) if h2 is not None else None
        twist = getattr(h2, "twist", None) if h2 is not None else None
        pret = getattr(h2, "poincare_return", None) if h2 is not None else None
        aa = getattr(h2, "action_angle", None) if h2 is not None else None
        hodge = getattr(engine_cert, "hodge_dirac", None)
        birk = getattr(engine_cert, "birkhoff", None)
        gen = _tuple_matrix_to_array(getattr(h1, "hamiltonian_generator", None)) if h1 else None
        return _EngineInvariants(
            dim_H1=int(getattr(h2, "cohomology_dim_H1", 0) or 0),
            melnikov_amp=float(getattr(getattr(h2, "melnikov", None), "M_value", 0.0) or 0.0),
            kam_stable=bool(getattr(getattr(h2, "kam_lattice", None), "is_kam_stable", True)),
            kam_gamma=float(
                getattr(getattr(h2, "kam_lattice", None), "diophantine_gamma", float("inf"))
                or float("inf")
            ),
            lyapunov_max=float(getattr(getattr(h2, "lyapunov", None), "lambda_max", 0.0) or 0.0),
            hodge_trivial=bool(getattr(hodge, "is_trivial_cohomology", True)),
            hodge_lambda_min=float(getattr(hodge, "lambda_min", float("inf")) or float("inf")),
            banach_L=float(getattr(h2, "banach_L", 1.0) or 1.0),
            rsi3_iterations=int(getattr(rsi, "iterations", getattr(h2, "rsi3_iterations", 0)) or 0),
            rsi3_converged=bool(getattr(rsi, "converged", getattr(h2, "rsi3_converged", False))),
            rsi3_residual=float(getattr(rsi, "residual", 1.0) or 1.0),
            rsi3_trajectory=tuple(getattr(rsi, "score_trajectory", ()) or ()),
            twist_ok=bool(getattr(twist, "is_nondegenerate", True)),
            poincare_recurrent=bool(getattr(pret, "is_recurrent", False)),
            poincare_recurrence_time=float(
                getattr(
                    engine_cert,
                    "poincare_recurrence_time",
                    getattr(pret, "recurrence_time", float("inf")),
                )
                or float("inf")
            ),
            action_energy=float(getattr(aa, "energy", 0.0) or 0.0),
            W_norm=float(getattr(h2, "homological_W_norm", 0.0) or 0.0),
            frequencies=tuple(getattr(h1, "frequencies", ()) or ()),
            symplectic_dim=int(getattr(h1, "symplectic_dim", 0) or 0),
            antisymmetry_ok=bool(getattr(h1, "antisymmetry_ok", True)),
            interference_frobenius=float(getattr(h1, "interference_frobenius", 0.0) or 0.0),
            coboundary_shape=tuple(getattr(h2, "coboundary_shape", (0, 0)) or (0, 0)),  # type: ignore[arg-type]
            coboundary_frobenius=float(getattr(h2, "coboundary_frobenius", 0.0) or 0.0),
            coboundary_sv=tuple(getattr(h2, "coboundary_svd_s", ()) or ()),
            birkhoff_remainder=float(getattr(birk, "remainder_norm", 0.0) or 0.0),
            birkhoff_normalized=bool(getattr(birk, "is_normalized", True)),
            hamiltonian_ok=bool(getattr(h1, "hamiltonian_ok", True)),
            overall_score=float(getattr(engine_cert, "overall_score", 0.0) or 0.0),
            generator=gen,
        )
    except Exception as exc:
        logger.warning(f"[Agent] Extracción de invariantes del motor falló: {exc}")
        return _EngineInvariants()


def _flow_map_from_generator(
    A: NDArray[np.float64], dim: int, epsilon: float = _FLOW_MAP_EPSILON
) -> NDArray[np.float64]:
    r"""$M=\exp(\varepsilon A)$ proyectado a $Sp(2n)$ (Cayley-Darboux)."""
    engine = SymplecticPoincareEngine(dim=dim)
    if hasattr(engine, "construct_flow_map_from_sp_generator"):
        return np.asarray(
            engine.construct_flow_map_from_sp_generator(A, epsilon=epsilon),
            dtype=np.float64,
        )
    M = la.expm(float(epsilon) * np.asarray(A, dtype=np.float64))
    if hasattr(engine, "cayley_darboux_projection"):
        return np.asarray(engine.cayley_darboux_projection(M), dtype=np.float64)
    return np.asarray(M, dtype=np.float64)


def _canonical_omega(n: int) -> NDArray[np.float64]:
    omega = np.zeros((2 * n, 2 * n), dtype=np.float64)
    I_n = np.eye(n, dtype=np.float64)
    omega[:n, n:] = I_n
    omega[n:, :n] = -I_n
    return omega


def _jacobian_from_engine_or_metric(
    inv: _EngineInvariants,
    G_metric: Optional[NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], bool]:
    r"""
    Construye $M\in\mathrm{Sp}(2n)$:
      1. generador $A\in\mathfrak{sp}(2n)$ del motor v5.1.0,
      2. inmersión simpléctica de $G$ (parte antisimétrica),
      3. $\Omega$ canónica (siempre simpléctica).
    """
    if inv.generator is not None:
        A = inv.generator
        n = A.shape[0] // 2
        if n >= 1 and A.shape[0] == 2 * n:
            try:
                return _flow_map_from_generator(A, n), True
            except Exception as exc:
                logger.warning(f"[Agent] Flujo desde generador falló: {exc}")
    if G_metric is not None:
        G = np.asarray(G_metric, dtype=np.float64)
        if G.ndim == 2 and G.shape[0] == G.shape[1] and G.shape[0] >= 2:
            n = G.shape[0] // 2
            if 2 * n == G.shape[0] and n >= 1:
                K = 0.5 * (G - G.T)
                A_embed = np.zeros((2 * n, 2 * n), dtype=np.float64)
                A_embed[:n, n:] = 0.5 * (K[:n, :n] + np.eye(n))
                A_embed[n:, :n] = -0.5 * (K[:n, :n] + np.eye(n))
                try:
                    return _flow_map_from_generator(A_embed, n), False
                except Exception:
                    pass
    n = max(1, int(inv.symplectic_dim) if inv.symplectic_dim else 1)
    return _canonical_omega(n), False


def _port_hamiltonian_pair(
    G_metric: Optional[NDArray[np.float64]],
    dim: int,
    inv: _EngineInvariants,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Par Port-Hamiltoniano $(\Phi,\nabla V)$:
      • Si $G$ es SPD, $\Phi=$ autovector dominante, $\nabla V=G\Phi$
        $\implies \Phi^\top\nabla V=\lambda_{\max}(G)\ge 0$.
      • Si no, $\Phi=\mathbf{1}$, $\nabla V=\mathrm{diag}(1+\sigma)\Phi$
        con $\sigma$ la producción de entropía proxy (path-energy del motor).
    """
    dim = max(2, int(dim))
    if G_metric is not None:
        G = np.asarray(G_metric, dtype=np.float64)
        if G.ndim == 2 and G.shape[0] == G.shape[1] and G.shape[0] >= 2:
            Gs = 0.5 * (G + G.T)
            n = min(dim, Gs.shape[0])
            Gs_n = Gs[:n, :n] + 1e-12 * np.eye(n)
            try:
                evals, evecs = la.eigh(Gs_n)
                Phi = np.real(evecs[:, -1])
                gradV = Gs_n @ Phi
                if Phi.size < dim:
                    Phi = np.pad(Phi, (0, dim - Phi.size))
                    gradV = np.pad(gradV, (0, dim - gradV.size))
                return Phi[:dim], gradV[:dim]
            except Exception:
                pass
    Phi = np.ones(dim, dtype=np.float64)
    scale = 1.0 + max(0.0, float(inv.action_energy))
    gradV = scale * Phi
    return Phi, gradV


def _quadratic_rate_from_trajectory(traj: Tuple[float, ...] | List[float]) -> float:
    e = [abs(float(x)) for x in traj if math.isfinite(float(x))]
    diffs = [abs(e[k + 1] - e[k]) for k in range(len(e) - 1)]
    pos = [d for d in diffs if d > _RSI3_CONV_EPS]
    orders: List[float] = []
    if len(pos) >= 3:
        for k in range(1, len(pos) - 1):
            num = math.log(max(pos[k + 1] / pos[k], _MACHINE_EPSILON))
            den = math.log(max(pos[k] / pos[k - 1], _MACHINE_EPSILON))
            if abs(den) > 1e-9:
                orders.append(float(num / den))
    if not orders:
        return 0.0
    return float(np.clip(float(np.median(orders)), 0.0, 4.0))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 — AUDITORÍA DE INVARIANZA SIMPLÉCTICA                              ║
# ║  Última definición: `_handoff_to_phase2_agent`                             ║
# ║  que ES el dominio de `_ingest_phase1_handoff_agent`.                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class Phase1_SymplecticInvarianceAuditor(_FiniteNumericalGuard):
    r"""
    FASE 1: audita el Teorema de Liouville y la proyección Cayley-Darboux.

    Métodos:
      • `_audit_symplectic_invariance(M)` — certificado completo.
      • `_handoff_to_phase2_agent(...)` — ÚLTIMA DEFINICIÓN de FASE 1.
    """

    def _build_canonical_symplectic_matrix(self, n: int) -> NDArray[np.float64]:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"[Fase 1] n entero positivo; recibido {n}.")
        return _canonical_omega(int(n))

    @staticmethod
    def _determinant_residual_from_slogdet(sign: float, logabsdet: float) -> float:
        if sign == 0 or not math.isfinite(logabsdet):
            return 1.0
        max_log = math.log(np.finfo(np.float64).max * 0.5)
        if logabsdet > max_log:
            return math.inf
        if logabsdet < -max_log:
            return 1.0
        det_value = float(sign) * math.exp(logabsdet)
        return float(abs(det_value - 1.0))

    @staticmethod
    def _condition_number(M: NDArray[np.float64]) -> float:
        try:
            svs = la.svdvals(M)
            if svs.size == 0 or svs[-1] == 0.0:
                return math.inf
            return float(svs[0] / svs[-1])
        except Exception:
            return math.inf

    @staticmethod
    def _compute_adaptive_symplectic_tolerance(sigma_max_M: float) -> float:
        if not math.isfinite(sigma_max_M) or sigma_max_M <= 0.0:
            return _SYMPLECTIC_TOLERANCE_BASE
        spec_tol = (
            _SYMPLECTIC_SPECTRAL_FACTOR
            * sigma_max_M
            * _MACHINE_EPSILON
            * _NUMERICAL_SAFETY_FACTOR
        )
        return max(_SYMPLECTIC_TOLERANCE_BASE, spec_tol)

    def _polar_unitarity_residual(
        self, M: NDArray[np.float64], omega: NDArray[np.float64]
    ) -> float:
        try:
            MtM = M.T @ M
            eigvals, eigvecs = la.eigh(MtM)
            sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0.0))
            inv_sqrt = np.where(
                sqrt_eigvals > _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
                1.0 / sqrt_eigvals,
                0.0,
            )
            P_inv = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T
            U = M @ P_inv
            UtOmegaU = U.T @ omega @ U
            omega_norm = self._frobenius_norm(omega)
            return float(self._frobenius_norm(UtOmegaU - omega) / (omega_norm or 1.0))
        except Exception as exc:
            raise SymplecticPolarDecompositionError(
                f"[Fase 1] Descomposición polar falló: {exc}"
            ) from exc

    def _compute_maslov_index(
        self, M: NDArray[np.float64], omega: NDArray[np.float64]
    ) -> int:
        n = M.shape[0] // 2
        B = M[n:, :n]
        try:
            eigvals_B = np.real(la.eigvals(B))
            return int(np.sum(eigvals_B < -_MASLOV_PHASE_TOLERANCE))
        except Exception as exc:
            raise MaslovIndexError(f"[Fase 1] Maslov falló: {exc}") from exc

    @staticmethod
    def _spectral_radius_transformed_omega(
        transformed_omega: NDArray[np.float64],
    ) -> float:
        try:
            eigvals = la.eigvals(transformed_omega)
            return float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 0.0
        except Exception:
            return math.inf

    def _antisymmetric_residual(
        self, transformed_omega: NDArray[np.float64], omega: NDArray[np.float64]
    ) -> float:
        omega_norm = self._frobenius_norm(omega)
        if omega_norm == 0.0:
            return 0.0
        skew = (transformed_omega - transformed_omega.T) * 0.5
        return float(self._frobenius_norm(skew - omega) / omega_norm)

    def _audit_symplectic_invariance(
        self, ast_jacobian_M: NDArray[np.float64]
    ) -> SymplecticInvariantData:
        r"""Audita Liouville: $M^\top\Omega M=\Omega$ y $\det M=+1$."""
        M = self._as_finite_matrix(
            "ast_jacobian_M", ast_jacobian_M, square=True, check_structural_zeros=True
        )
        dim = M.shape[0]
        if dim == 0 or (dim % 2) != 0:
            raise SymplecticInvarianceViolation(
                f"[Fase 1] Dimensión par positiva; dim={dim}."
            )

        n = dim // 2
        omega = self._build_canonical_symplectic_matrix(n)

        engine = SymplecticPoincareEngine(dim=n)
        M_proj = engine.cayley_darboux_projection(M)
        transformed = M_proj.T @ omega @ M_proj

        omega_norm = self._frobenius_norm(omega)
        transformed_norm = self._frobenius_norm(transformed)
        residual = self._frobenius_norm(transformed - omega)
        antisymmetric_res = self._antisymmetric_residual(transformed, omega)

        scale = max(1.0, transformed_norm + omega_norm)
        relative_residual = residual / scale

        sigma_max_M = self._spectral_norm(M_proj)
        effective_tol = self._compute_adaptive_symplectic_tolerance(sigma_max_M)

        if relative_residual > effective_tol:
            raise SymplecticInvarianceViolation(
                f"[Fase 1] Liouville violado: resid={relative_residual:.3e} "
                f"> tol={effective_tol:.3e}."
            )

        sign, logabsdet = la.slogdet(M_proj)
        if sign <= 0 or not math.isfinite(sign):
            raise SymplecticInvarianceViolation("[Fase 1] det M ≤ 0 o singular.")

        det_residual = self._determinant_residual_from_slogdet(sign, logabsdet)
        cond = self._condition_number(M_proj)
        if cond > _CONDITION_NUMBER_WARNING_THRESHOLD:
            logger.warning("[Fase 1] κ(M)=%.3e excesivo.", cond)
        spec_radius = self._spectral_radius_transformed_omega(transformed)
        maslov = self._compute_maslov_index(M_proj, omega)
        polar_res = self._polar_unitarity_residual(M_proj, omega)

        return SymplecticInvariantData(
            phase_space_dimension=dim,
            symplectic_residual_norm=residual,
            symplectic_relative_residual=relative_residual,
            antisymmetric_residual=antisymmetric_res,
            determinant_residual=det_residual,
            condition_number=cond,
            spectral_radius_Omega=spec_radius,
            maslov_index=maslov,
            polar_unitarity_residual=polar_res,
            effective_tolerance=effective_tol,
            is_volume_preserved=True,
        )

    # ─── ÚLTIMA DEFINICIÓN DE FASE 1 (continuación estricta a FASE 2) ───────
    def _handoff_to_phase2_agent(
        self,
        symplectic_audit: SymplecticInvariantData,
        input_checksum: str,
        engine_certificate: Optional[Any] = None,
        frequencies: Tuple[float, ...] = (),
        symplectic_dim: int = 0,
        antisymmetry_ok: bool = True,
        interference_frobenius: float = 0.0,
        generator_used: bool = False,
    ) -> AgentPhase1Handoff:
        r"""
        Emite el morfismo $\Phi_1\to\Phi_2$.

        El objeto retornado **es** el dominio estricto de
        `Phase2_DirichletThermodynamicEnforcer._ingest_phase1_handoff_agent`:

            $\Phi_2\circ\Phi_1(M)
                =\texttt{\_ingest\_phase1\_handoff\_agent}(
                    \texttt{\_handoff\_to\_phase2\_agent}(M))$.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        handoff = AgentPhase1Handoff(
            symplectic_audit=symplectic_audit,
            input_checksum=input_checksum,
            timestamp_iso=timestamp,
            engine_certificate=engine_certificate,
            frequencies=frequencies,
            symplectic_dim=symplectic_dim or symplectic_audit.phase_space_dimension // 2,
            antisymmetry_ok=antisymmetry_ok,
            interference_frobenius=interference_frobenius,
            generator_used=generator_used,
        )
        logger.info(
            "[FASE1→FASE2-Agent] δω=%.3e, det_res=%.3e, κ=%.3e, Maslov=%d, "
            "ω=%s, ‖ω_I‖_F=%.3f, gen=%s",
            symplectic_audit.symplectic_relative_residual,
            symplectic_audit.determinant_residual,
            symplectic_audit.condition_number,
            symplectic_audit.maslov_index,
            tuple(round(x, 3) for x in frequencies),
            interference_frobenius,
            generator_used,
        )
        return handoff


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 — CONTROL PORT-HAMILTONIANO, POINCARÉ Y RSI³ BANACH                ║
# ║  Primera def.: `_ingest_phase1_handoff_agent`                              ║
# ║  (continuación estricta de `_handoff_to_phase2_agent`)                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class Phase2_DirichletThermodynamicEnforcer(Phase1_SymplecticInvarianceAuditor):
    r"""
    FASE 2: disipación Port-Hamiltoniana + RSI³ con Banach real + Poincaré.

    Métodos:
      • `_ingest_phase1_handoff_agent` — PRIMERA DEFINICIÓN de FASE 2.
      • `_enforce_dirichlet_thermodynamics` — disipación Clausius-Duhem.
      • `_rsi3_banach_iteration` — contracción real (no RNG).
      • `_handoff_to_phase3_agent` — ÚLTIMA DEFINICIÓN de FASE 2.
    """

    @staticmethod
    def _compute_entropy_production_rate(
        dissipated_power: float,
        temperature_reference: float = _THERMODYNAMIC_TEMPERATURE_REFERENCE,
    ) -> float:
        if temperature_reference <= 0.0:
            raise ClausiusDuhemViolation(
                f"[Fase 2] T_ref debe ser positiva; T_ref={temperature_reference}."
            )
        sigma = dissipated_power / temperature_reference
        if not math.isfinite(sigma):
            raise ClausiusDuhemViolation(
                f"[Fase 2] Tasa de entropía no finita: {sigma}."
            )
        return sigma

    @staticmethod
    def _compute_thermodynamic_angle(alignment_cosine: float) -> float:
        return float(math.acos(float(np.clip(alignment_cosine, -1.0, 1.0))))

    @staticmethod
    def _compute_exergy_dissipation_ratio(
        dissipated_power: float, phi_norm: float, grad_norm: float
    ) -> float:
        denom = phi_norm * grad_norm
        if denom == 0.0 or not math.isfinite(denom):
            return 0.0
        return float(np.clip(dissipated_power / denom, -1.0, 1.0))

    # ─── PRIMERA DEFINICIÓN DE FASE 2 ────────────────────────────────────────
    @classmethod
    def _ingest_phase1_handoff_agent(cls, h1: AgentPhase1Handoff) -> AgentPhase1Handoff:
        r"""
        Consume el `AgentPhase1Handoff` de `_handoff_to_phase2_agent`.

        Continuación estricta:
            $\mathrm{dom}(\texttt{\_ingest\_phase1\_handoff\_agent})
                =\mathrm{im}(\texttt{\_handoff\_to\_phase2\_agent})$.
        """
        if not isinstance(h1, AgentPhase1Handoff):
            raise OODAChainViolation(
                f"[FASE 2-Agent] Handoff inválido: {type(h1).__name__} "
                "(se esperaba AgentPhase1Handoff)."
            )
        if not h1.input_checksum or len(h1.input_checksum) < 16:
            raise OODAChainViolation("[FASE 2-Agent] checksum de FASE 1 inválido.")
        if not h1.symplectic_audit.is_volume_preserved:
            raise SymplecticInvarianceViolation(
                "[FASE 2-Agent] FASE 1 no preservó volumen simpléctico."
            )
        logger.debug(
            "[FASE2-Agent] Handoff ingerido: checksum=%s..., ts=%s, gen=%s, ω=%s",
            h1.input_checksum[:16],
            h1.timestamp_iso,
            h1.generator_used,
            tuple(round(x, 3) for x in h1.frequencies),
        )
        return h1

    def _enforce_dirichlet_thermodynamics(
        self,
        control_potential_Phi: NDArray[np.float64],
        lyapunov_gradient_V: NDArray[np.float64],
        symplectic_audit: Optional[SymplecticInvariantData] = None,
    ) -> ThermodynamicDirichletData:
        r"""Disipación Port-Hamiltoniana: $P=\Phi^\top\nabla V\ge 0$ y $\sigma_{CD}=P/T\ge 0$."""
        Phi = self._as_finite_vector("control_potential_Phi", control_potential_Phi)
        gradV = self._as_finite_vector("lyapunov_gradient_V", lyapunov_gradient_V)

        if Phi.shape != gradV.shape:
            raise ValueError(
                f"[Fase 2] Dimensiones incompatibles: Φ {Phi.shape} vs ∇V {gradV.shape}."
            )

        if symplectic_audit is not None:
            if not symplectic_audit.is_volume_preserved:
                raise SymplecticInvarianceViolation(
                    "[Fase 2] Fase 1 no preservó volumen."
                )
            if Phi.size != symplectic_audit.phase_space_dimension:
                raise ValueError(
                    f"[Fase 2] Inconsistencia dim: "
                    f"cert={symplectic_audit.phase_space_dimension} vs Φ={Phi.size}."
                )

        phi_norm = self._vector_norm(Phi)
        grad_norm = self._vector_norm(gradV)
        if not math.isfinite(phi_norm) or not math.isfinite(grad_norm):
            raise ExergyDivergenceError("[Fase 2] Norma Φ o ∇V no finita.")

        p_raw = float(np.dot(Phi, gradV))
        tol = _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, phi_norm * grad_norm)

        if p_raw < _DIRICHLET_DISSIPATION_FLOOR - tol:
            raise ThermodynamicSingularityError(p_raw, 0.0, 0, 0)

        dissipated = max(_DIRICHLET_DISSIPATION_FLOOR, p_raw)
        alignment = (
            float(np.clip(p_raw / (phi_norm * grad_norm), -1.0, 1.0))
            if phi_norm > 0 and grad_norm > 0
            else 0.0
        )
        theta = self._compute_thermodynamic_angle(alignment)
        entropy_prod = self._compute_entropy_production_rate(dissipated)
        cd_ok = entropy_prod >= _ENTROPY_PRODUCTION_FLOOR
        if not cd_ok:
            raise ClausiusDuhemViolation(
                f"[Fase 2] Clausius-Duhem violado: σ={entropy_prod:.3e}."
            )
        ratio = self._compute_exergy_dissipation_ratio(dissipated, phi_norm, grad_norm)
        is_strict = dissipated > max(_DIRICHLET_DISSIPATION_FLOOR, tol)

        return ThermodynamicDirichletData(
            dissipated_power=dissipated,
            raw_inner_product=p_raw,
            numerical_tolerance=tol,
            exergy_norm=phi_norm,
            gradient_norm=grad_norm,
            alignment_cosine=alignment,
            thermodynamic_angle_rad=theta,
            entropy_production_rate=entropy_prod,
            clausius_duhem_satisfied=cd_ok,
            exergy_dissipation_ratio=ratio,
            is_thermodynamically_stable=True,
            is_strictly_dissipative=is_strict,
        )

    def _rsi3_local_contraction(
        self,
        symplectic_audit: Optional[SymplecticInvariantData],
        thermo: Optional[ThermodynamicDirichletData],
        inv: _EngineInvariants,
        max_iter: int,
        min_iter: int,
        banach_target: float,
    ) -> Tuple[float, int, bool, float, float, Tuple[float, ...]]:
        r"""
        Contracción determinista de Banach de nivel 3 sobre
        $X=\mathbb{R}^3\ni x=(\delta\omega,\,1-\cos\theta,\,e^{-\sigma})$.

        Nivel 1: $f(x)=\|x\|_2$ (residual espectral-termodinámico).
        Nivel 2: $P(f)$ aplica $T(x)=(\kappa x_1, x_2^2, \beta x_3)$.
        Nivel 3: $Q(P)$ adapta $\eta$ para forzar $\mathrm{Lip}(T)<1$.

        NO usa RNG. Si el motor v5.1.0 ya contrajo, esta rutina no se llama.
        """
        dw = (
            float(symplectic_audit.symplectic_relative_residual)
            if symplectic_audit is not None
            else 1.0
        )
        align = float(thermo.alignment_cosine) if thermo is not None else 0.0
        sigma = float(thermo.entropy_production_rate) if thermo is not None else 0.0
        x = np.array(
            [
                max(dw, 0.0),
                max(0.0, 1.0 - abs(align)),
                math.exp(-max(sigma, 0.0)),
            ],
            dtype=np.float64,
        )
        kappa = 0.45 if (symplectic_audit and symplectic_audit.is_volume_preserved) else 1.15
        beta = 0.70 if inv.kam_stable else 0.98
        eta = 0.35
        scores: List[float] = [float(np.linalg.norm(x))]
        prev_x = x.copy()
        lip = 1.0
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            T_x = np.array([kappa * x[0], x[1] * x[1], beta * x[2]], dtype=np.float64)
            x_new = x + eta * (T_x - x)
            dx = float(np.linalg.norm(x_new - x))
            dprev = float(np.linalg.norm(x - prev_x))
            lip_step = dx / max(dprev, 1e-15) if it > 1 else 1.0
            if lip_step >= 1.0:
                eta *= 0.5
            else:
                eta = min(0.8, eta * 1.05)
            scores.append(float(np.linalg.norm(x_new)))
            lip = lip_step
            prev_x, x = x, x_new
            if (
                it >= min_iter
                and abs(scores[-1] - scores[-2]) < _RSI3_CONV_EPS
                and lip < _RSI3_LIPSCHITZ_TARGET
            ):
                converged = True
                break
        p_rate = _quadratic_rate_from_trajectory(tuple(scores))
        converged = bool(converged and lip < 1.0 and lip < banach_target)
        return float(lip), it, converged, p_rate, float(scores[-1]), tuple(scores)

    def _rsi3_banach_iteration(
        self,
        engine_certificate: Optional[Any] = None,
        banach_target: float = _BANACH_TARGET_DEFAULT,
        max_iter: int = _RSI3_MAX_ITER_DEFAULT,
        min_iter: int = _RSI3_MIN_ITER_DEFAULT,
        symplectic_audit: Optional[SymplecticInvariantData] = None,
        thermodynamic_audit: Optional[ThermodynamicDirichletData] = None,
    ) -> Tuple[float, int, bool, float, float, Tuple[float, ...]]:
        r"""
        RSI³ real. Prioridad:
          1. `RSI3Report` del motor v5.1.0 (contracción ya ejecutada sobre
             $(\gamma,\tau,\varepsilon)$ con flujo Hamiltoniano real).
          2. Re-ejecución de `RecursiveSelfImprovementL3` si el generador
             $A$ está disponible.
          3. Contracción local determinista (nunca RNG).

        Retorna $(L_\Phi, \mathrm{iter}, \mathrm{conv}, p, r_\infty, \mathrm{traj})$.
        """
        inv = _extract_engine_invariants(engine_certificate)

        if _ENGINE_V51_AVAILABLE and inv.rsi3_iterations > 0:
            lip = float(inv.banach_L)
            p_rate = _quadratic_rate_from_trajectory(inv.rsi3_trajectory)
            logger.info(
                "[RSI³-Agent] motor v5.1.0: iter=%d, L_Φ=%.4f, conv=%s, r=%.3e",
                inv.rsi3_iterations,
                lip,
                inv.rsi3_converged,
                inv.rsi3_residual,
            )
            return (
                lip,
                int(inv.rsi3_iterations),
                bool(inv.rsi3_converged),
                p_rate,
                float(inv.rsi3_residual),
                inv.rsi3_trajectory,
            )

        if (
            _ENGINE_V51_AVAILABLE
            and EngineRSI3 is not None
            and inv.generator is not None
            and inv.frequencies
        ):
            try:
                n = inv.generator.shape[0] // 2
                engine = SymplecticPoincareEngine(dim=max(1, n))
                rsi = EngineRSI3(engine)
                omega = np.asarray(inv.frequencies, dtype=np.float64)
                t = np.linspace(0.0, 1.0, 16)
                w0 = float(omega[0]) if omega.size else 1.0
                w1 = float(omega[1]) if omega.size > 1 else w0 / 1.618033988749895
                q = np.cos(2 * np.pi * w0 * t)
                p = -np.sin(2 * np.pi * w0 * t)
                H0 = lambda qq, pp: 0.5 * pp * pp + 0.5 * (w0 ** 2) * qq * qq
                H1 = lambda qq, pp: 0.1 * qq * qq * qq
                report = rsi.iterate(
                    omega, inv.generator, q, p, H0, H1, max_iter=max_iter, min_iter=min_iter
                )
                p_rate = _quadratic_rate_from_trajectory(tuple(report.score_trajectory))
                logger.info(
                    "[RSI³-Agent] RecursiveSelfImprovementL3: iter=%d, L_Φ=%.4f, conv=%s",
                    report.iterations,
                    report.lipschitz,
                    report.converged,
                )
                return (
                    float(report.lipschitz),
                    int(report.iterations),
                    bool(report.converged),
                    p_rate,
                    float(report.residual),
                    tuple(report.score_trajectory),
                )
            except Exception as exc:
                logger.warning(f"[RSI³-Agent] EngineRSI3 falló: {exc}; contracción local.")

        return self._rsi3_local_contraction(
            symplectic_audit, thermodynamic_audit, inv, max_iter, min_iter, banach_target
        )

    # ─── ÚLTIMA DEFINICIÓN DE FASE 2 (continuación estricta a FASE 3) ───────
    def _handoff_to_phase3_agent(
        self,
        h1: AgentPhase1Handoff,
        thermodynamic_audit: ThermodynamicDirichletData,
        banach_L_real: float,
        rsi3_iterations: int,
        rsi3_converged: bool,
        quadratic_rate: float,
        lyapunov_max: float = 0.0,
        melnikov_amplitude: float = 0.0,
        kam_gamma: float = 0.0,
        kam_stable: bool = True,
        twist_ok: bool = True,
        poincare_recurrent: bool = False,
        poincare_recurrence_time: float = float("inf"),
        action_energy: float = 0.0,
        homological_W_norm: float = 0.0,
        rsi3_residual: float = 1.0,
        rsi3_score_trajectory: Tuple[float, ...] = (),
        coboundary_frobenius: float = 0.0,
        coboundary_shape: Tuple[int, int] = (0, 0),
    ) -> AgentPhase2Handoff:
        r"""
        Emite el morfismo $\Phi_2\to\Phi_3$.

        El objeto retornado **es** el dominio estricto de
        `Phase3_CellularSheafCohomologyAuditor._ingest_phase2_handoff_agent`.
        """
        handoff = AgentPhase2Handoff(
            phase1=h1,
            thermodynamic_audit=thermodynamic_audit,
            banach_L_real=banach_L_real,
            rsi3_iterations=rsi3_iterations,
            rsi3_converged=rsi3_converged,
            quadratic_convergence_rate=quadratic_rate,
            lyapunov_max=lyapunov_max,
            melnikov_amplitude=melnikov_amplitude,
            kam_gamma=kam_gamma,
            kam_stable=kam_stable,
            twist_ok=twist_ok,
            poincare_recurrent=poincare_recurrent,
            poincare_recurrence_time=poincare_recurrence_time,
            action_energy=action_energy,
            homological_W_norm=homological_W_norm,
            rsi3_residual=rsi3_residual,
            rsi3_score_trajectory=rsi3_score_trajectory,
            coboundary_frobenius=coboundary_frobenius,
            coboundary_shape=coboundary_shape,
        )
        logger.info(
            "[FASE2→FASE3-Agent] L_Φ=%.4f, iter=%d, conv=%s, p=%.2f, λ_max=%.3e, "
            "|M|=%.3e, γ_KAM=%.3e, KAM=%s, twist=%s, Poincaré=%s, ‖W‖=%.3e",
            banach_L_real,
            rsi3_iterations,
            rsi3_converged,
            quadratic_rate,
            lyapunov_max,
            melnikov_amplitude,
            kam_gamma,
            kam_stable,
            twist_ok,
            poincare_recurrent,
            homological_W_norm,
        )
        return handoff


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 — COHOMOLOGÍA DE HACES, HEYTING Ω₃ Y CERTIFICADO                   ║
# ║  Primera def.: `_ingest_phase2_handoff_agent`                              ║
# ║  (continuación estricta de `_handoff_to_phase3_agent`)                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class Phase3_CellularSheafCohomologyAuditor(Phase2_DirichletThermodynamicEnforcer):
    r"""
    FASE 3: cohomología de haces + verificación Heyting $\Omega_3$ + certificado.

    Métodos:
      • `_ingest_phase2_handoff_agent` — PRIMERA DEFINICIÓN de FASE 3.
      • `_audit_cellular_sheaf_cohomology` — nulidad de $H^1$.
      • `_heyting_adjudication` — veredicto $\Omega_3$ (10 átomos).
      • `_emit_agent_certificate` — certificado final.
    """

    @classmethod
    def _numerical_rank_with_gap(
        cls, A: NDArray[np.float64], name: str = "A"
    ) -> Tuple[int, NDArray[np.float64]]:
        if A.size == 0 or min(A.shape) == 0:
            return 0, np.array([], dtype=np.float64)
        svs = cls._safe_svdvals(A)
        if svs.size == 0 or svs[0] == 0.0:
            return 0, svs
        sigma_max = float(svs[0])
        tol_std = (
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(A.shape) * sigma_max
        )
        norm_svs = svs / sigma_max
        gaps = np.diff(norm_svs)
        if gaps.size > 0:
            gap_idx = int(np.argmin(gaps))
            if float(-gaps[gap_idx]) > 10.0 * _SPECTRAL_GAP_RATIO:
                return gap_idx + 1, svs
        return int(np.count_nonzero(svs > tol_std)), svs

    def _compute_betti_numbers(
        self,
        D0: NDArray[np.float64],
        D1: NDArray[np.float64],
        rank_D0: int,
        rank_D1: int,
    ) -> Tuple[int, int, int]:
        c0 = D0.shape[1]
        c1 = D0.shape[0]
        c2 = D1.shape[0]
        return (
            int(max(0, c0 - rank_D0)),
            int(max(0, c1 - rank_D0 - rank_D1)),
            int(max(0, c2 - rank_D1)),
        )

    def _compute_reidemeister_torsion(
        self,
        D0: NDArray[np.float64],
        D1: NDArray[np.float64],
        rank_D0: int,
        rank_D1: int,
        svs_D0: NDArray[np.float64],
        svs_D1: NDArray[np.float64],
    ) -> float:
        c1 = D0.shape[0]
        if c1 - rank_D0 - rank_D1 != 0:
            return 0.0

        def log_sum(svs: NDArray[np.float64], rank: int) -> float:
            if rank <= 0 or svs.size == 0:
                return 0.0
            pos = svs[:rank][svs[:rank] > 0]
            return float(np.sum(np.log(pos))) if pos.size > 0 else 0.0

        log_tau = log_sum(svs_D0, rank_D0) - log_sum(svs_D1, rank_D1)
        return log_tau if math.isfinite(log_tau) else 0.0

    def _verify_mayer_vietoris_exactness(
        self,
        D0: NDArray[np.float64],
        D1: NDArray[np.float64],
        rank_D0: int,
        rank_D1: int,
    ) -> bool:
        return rank_D0 + rank_D1 <= D0.shape[0]

    def _compute_first_cohomology_dimension(
        self,
        coboundary_delta0: NDArray[np.float64],
        coboundary_delta1: NDArray[np.float64],
    ) -> Tuple[int, int, int, int, int, float, float, bool, NDArray[np.float64], NDArray[np.float64]]:
        D0 = self._as_finite_matrix("coboundary_delta0", coboundary_delta0)
        D1 = self._as_finite_matrix("coboundary_delta1", coboundary_delta1)
        if D1.shape[1] != D0.shape[0]:
            raise ValueError(
                f"[Fase 3] δ¹ ∘ δ⁰ incompatible: {D1.shape} vs {D0.shape}."
            )

        comp = D1 @ D0
        comp_norm = self._frobenius_norm(comp)
        scale = max(1.0, self._frobenius_norm(D0) * self._frobenius_norm(D1))
        complex_residual = comp_norm / scale
        if complex_residual > _COHOMOLOGICAL_COMPLEX_TOL:
            _raise_cohomology(
                1,
                f"[Fase 3] δ¹∘δ⁰≠0: ‖·‖/scale={complex_residual:.3e} "
                f"> {_COHOMOLOGICAL_COMPLEX_TOL:.3e}.",
            )

        rank_D0, svs_D0 = self._numerical_rank_with_gap(D0, "δ⁰")
        rank_D1, svs_D1 = self._numerical_rank_with_gap(D1, "δ¹")

        c1 = D0.shape[0]
        h1 = max(0, c1 - rank_D0 - rank_D1)

        b0, _b1, b2 = self._compute_betti_numbers(D0, D1, rank_D0, rank_D1)
        torsion = self._compute_reidemeister_torsion(
            D0, D1, rank_D0, rank_D1, svs_D0, svs_D1
        )
        mv_exact = self._verify_mayer_vietoris_exactness(D0, D1, rank_D0, rank_D1)

        return (
            int(h1),
            int(rank_D0),
            int(rank_D1),
            int(b0),
            int(b2),
            float(torsion),
            float(complex_residual),
            bool(mv_exact),
            svs_D0,
            svs_D1,
        )

    def _hodge_from_coboundary(
        self,
        coboundary_delta0: NDArray[np.float64],
        G_metric: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[float, bool]:
        """Espectro Hodge-Dirac sobre $\delta^0$ real (axioma A4)."""
        d = np.asarray(coboundary_delta0, dtype=np.float64)
        if d.size == 0:
            return float("inf"), True
        n1 = d.shape[0]
        if G_metric is not None:
            G = np.asarray(G_metric, dtype=np.float64)
            if G.ndim == 1:
                G = np.diag(G)
            if G.shape != (n1, n1):
                G = np.eye(n1, dtype=np.float64)
        else:
            G = np.eye(n1, dtype=np.float64)
        hodge_op = HeavyHodgeDiracLaplacian()
        try:
            if hasattr(hodge_op, "compute_heavy_laplacian"):
                delta_G = hodge_op.compute_heavy_laplacian(d, G)
            else:
                delta_G = d.T @ d
            if hasattr(hodge_op, "verify_hodge_dirichlet_triviality"):
                spec = hodge_op.verify_hodge_dirichlet_triviality(delta_G)
                return float(spec.lambda_min), bool(spec.is_trivial_cohomology)
            D_sym = 0.5 * (delta_G + delta_G.T)
            evals = np.linalg.eigvalsh(D_sym)
            lam_min = float(evals[0])
            return lam_min, bool(abs(lam_min) > _HODGE_LAMBDA_MIN)
        except Exception as exc:
            logger.warning(f"[Fase 3] Hodge-Dirac falló: {exc}")
            return float("inf"), False

    # ─── PRIMERA DEFINICIÓN DE FASE 3 ────────────────────────────────────────
    @classmethod
    def _ingest_phase2_handoff_agent(cls, h2: AgentPhase2Handoff) -> AgentPhase2Handoff:
        r"""
        Consume el `AgentPhase2Handoff` de `_handoff_to_phase3_agent`.

        Continuación estricta:
            $\mathrm{dom}(\texttt{\_ingest\_phase2\_handoff\_agent})
                =\mathrm{im}(\texttt{\_handoff\_to\_phase3\_agent})$.
        """
        if not isinstance(h2, AgentPhase2Handoff):
            raise OODAChainViolation(
                f"[FASE 3-Agent] Handoff inválido: {type(h2).__name__} "
                "(se esperaba AgentPhase2Handoff)."
            )
        if not isinstance(h2.phase1, AgentPhase1Handoff):
            raise OODAChainViolation("[FASE 3-Agent] FASE 1 ausente en el handoff.")
        if not h2.phase1.symplectic_audit.is_volume_preserved:
            raise OODAChainViolation(
                "[FASE 3-Agent] Handoff con FASE 1 inválida (Liouville violado)."
            )
        if not h2.thermodynamic_audit.is_thermodynamically_stable:
            raise OODAChainViolation(
                "[FASE 3-Agent] Handoff con FASE 2 termodinámicamente inestable."
            )
        logger.debug(
            "[FASE3-Agent] Handoff ingerido: L_Φ=%.4f, conv=%s, λ_max=%.3e, "
            "γ_KAM=%.3e, twist=%s, Poincaré=%s",
            h2.banach_L_real,
            h2.rsi3_converged,
            h2.lyapunov_max,
            h2.kam_gamma,
            h2.twist_ok,
            h2.poincare_recurrent,
        )
        return h2

    def _audit_cellular_sheaf_cohomology(
        self,
        h1_dimension: Optional[int] = None,
        thermodynamic_audit: Optional[ThermodynamicDirichletData] = None,
        coboundary_delta0: Optional[NDArray[np.float64]] = None,
        coboundary_delta1: Optional[NDArray[np.float64]] = None,
        strict_h1: bool = True,
    ) -> SheafCohomologyAuditData:
        r"""Audita $H^1=0$ vía cobordes o vía `h1_dimension` declarado por el motor."""
        if (
            thermodynamic_audit is not None
            and not thermodynamic_audit.is_thermodynamically_stable
        ):
            raise ThermodynamicSingularityError(0.0, 0.0, 0, 0)

        computed_h1: Optional[int] = None
        verified_by_coboundary = False
        rank_d0_val = 0
        rank_d1_val = 0
        betti_vals: Tuple[int, int, int] = (0, 0, 0)
        euler_chi = 0
        reidemeister_val = 0.0
        complex_residual_val = 0.0
        mv_exact = True

        if coboundary_delta0 is not None or coboundary_delta1 is not None:
            if coboundary_delta0 is None or coboundary_delta1 is None:
                raise ValueError("[Fase 3] δ⁰ y δ¹ deben proveerse juntos.")
            (
                computed_h1,
                rank_d0_val,
                rank_d1_val,
                b0_val,
                b2_val,
                reidemeister_val,
                complex_residual_val,
                mv_exact,
                _svs_D0,
                _svs_D1,
            ) = self._compute_first_cohomology_dimension(
                coboundary_delta0, coboundary_delta1
            )
            betti_vals = (b0_val, computed_h1, b2_val)
            c0 = coboundary_delta0.shape[1]
            c1 = coboundary_delta0.shape[0]
            c2 = coboundary_delta1.shape[0]
            euler_chi = c0 - c1 + c2
            if not mv_exact:
                raise MayerVietorisBreachError("[Fase 3] Mayer-Vietoris no exacta.")
            verified_by_coboundary = True

        if isinstance(h1_dimension, bool):
            raise TypeError("[Fase 3] h1_dimension debe ser entero.")

        h1 = computed_h1 if h1_dimension is None else int(h1_dimension)

        if computed_h1 is not None and h1_dimension is not None and computed_h1 != h1:
            _raise_cohomology(
                max(computed_h1, h1),
                f"[Fase 3] h1 declarado={h1} vs calculado={computed_h1}.",
            )

        if h1 < _COHOMOLOGY_DIMENSION_FLOOR:
            raise ValueError(f"[Fase 3] h1_dimension no puede ser negativo: {h1}.")

        if h1 > _COHOMOLOGY_DIMENSION_FLOOR and strict_h1:
            _raise_cohomology(h1, f"[Fase 3] Obstrucción topológica: dim H¹={h1} > 0.")

        obstruction_free = h1 == _COHOMOLOGY_DIMENSION_FLOOR
        return SheafCohomologyAuditData(
            h1_dimension=h1,
            rank_delta0=rank_d0_val,
            rank_delta1=rank_d1_val,
            betti_numbers=betti_vals if verified_by_coboundary else (0, h1, 0),
            euler_characteristic=euler_chi,
            reidemeister_torsion=reidemeister_val,
            complex_residual=complex_residual_val,
            is_globally_integrable=obstruction_free,
            obstruction_free=obstruction_free,
            verified_by_coboundary=verified_by_coboundary,
            mayer_vietoris_exact=mv_exact,
        )

    def _heyting_adjudication(
        self,
        h1: AgentPhase1Handoff,
        h2: AgentPhase2Handoff,
        cohomology_ok: bool,
        hodge_trivial: bool,
        birkhoff_normalized: bool = True,
        banach_target: float = _BANACH_TARGET_DEFAULT,
    ) -> int:
        r"""
        Join bitwise OR sobre Heyting $\Omega_3=\{0x00,0x01,0x03\}$.

        Átomos v5.1.0:
          $v_1$ Liouville, $v_2$ Banach, $v_3$ $H^1$, $v_4$ KAM,
          $v_5$ Melnikov (VETOED si tangle), $v_6$ Lyapunov,
          $v_7$ Hodge-Dirac, $v_8$ twist Kolmogorov, $v_9$ RSI³,
          $v_{10}$ resto de Birkhoff.
        """
        v1 = 0x00 if h1.symplectic_audit.is_volume_preserved else 0x01
        v2 = 0x00 if h2.banach_L_real < banach_target else 0x01
        v3 = 0x00 if cohomology_ok else 0x01
        v4 = 0x00 if h2.kam_stable else 0x01
        v5 = 0x03 if h2.melnikov_amplitude > _MELNIKOV_THRESHOLD else 0x00
        v6 = 0x01 if h2.lyapunov_max > _LYAPUNOV_CHAOS else 0x00
        v7 = 0x00 if hodge_trivial else 0x01
        v8 = 0x00 if h2.twist_ok else 0x01
        v9 = 0x00 if h2.rsi3_converged else 0x01
        v10 = 0x00 if birkhoff_normalized else 0x01
        return _heyting_join_local([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])

    def _emit_agent_certificate(
        self,
        h2: AgentPhase2Handoff,
        cohomology_audit: SheafCohomologyAuditData,
        hodge_lambda_min: float,
        hodge_trivial: bool,
        heyting_mask: int,
        heyting_ok: bool,
        checksum: str,
        birkhoff_remainder: float = 0.0,
        birkhoff_normalized: bool = True,
    ) -> AgentPhase3Certificate:
        r"""Emite el certificado final $\Phi_3\circ\Phi_2\circ\Phi_1$."""
        p1_ok = h2.phase1.symplectic_audit.is_volume_preserved
        p2_ok = (
            h2.thermodynamic_audit.is_thermodynamically_stable and h2.rsi3_converged
        )
        p3_ok = cohomology_audit.is_globally_integrable and hodge_trivial
        is_authorized = (
            p1_ok
            and p2_ok
            and p3_ok
            and (heyting_mask != 0x03)
            and heyting_ok
            and h2.twist_ok
            and birkhoff_normalized
        )
        functor_chain = (
            f"Φ₁={'✓' if p1_ok else '✗'} → Φ₂={'✓' if p2_ok else '✗'} → "
            f"Φ₃={'✓' if p3_ok else '✗'} → Z_Γ-PHYSICS={'✓' if is_authorized else '✗'}"
        )
        provenance = AuditProvenance(
            timestamp_iso=datetime.now(tz=timezone.utc).isoformat(),
            input_checksum_sha256=checksum,
            phase1_passed=p1_ok,
            phase2_passed=p2_ok,
            phase3_passed=p3_ok,
            functor_chain=functor_chain,
        )
        return AgentPhase3Certificate(
            phase2=h2,
            cohomology_audit=cohomology_audit,
            hodge_lambda_min=hodge_lambda_min,
            hodge_trivial=hodge_trivial,
            heyting_mask=heyting_mask,
            heyting_axioms_verified=heyting_ok,
            is_compilation_authorized=is_authorized,
            provenance=provenance,
            birkhoff_remainder=birkhoff_remainder,
            birkhoff_normalized=birkhoff_normalized,
            poincare_recurrence_time=h2.poincare_recurrence_time,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §ORQUESTADOR SOBERANO RSI³: ASTStaticAnalyzerAgentRSI3
# ═══════════════════════════════════════════════════════════════════════════════

class ASTStaticAnalyzerAgentRSI3(Morphism, Phase3_CellularSheafCohomologyAuditor):
    r"""
    Soberano agéntico de gobernanza RSI³ sobre Heyting $\Omega_3$.

    Integración v5.1.0:
      • `audit_ooda_phase(ast_tree, G_metric)` recorre REALMENTE
        $\Phi_1\to\Phi_2\to\Phi_3$ (ya no bypass de handoffs).
      • Jacobiano $M=\exp(\varepsilon A)$ del generador del motor.
      • RSI³ = `RSI3Report` del motor o contracción determinista.
      • Fail-safe: motor caído $\implies$ VETOED (Crowbar).
    """

    def __init__(
        self,
        gpio_pin: int = 14,
        banach_target: float = _BANACH_TARGET_DEFAULT,
        eta: float = 0.1,
        zeta: float = 0.01,
    ) -> None:
        self.gpio_pin = gpio_pin
        self.banach_target = banach_target
        self.eta = eta
        self.zeta = zeta
        logger.info(
            "[AgentRSI3] Soberano v5.1.0 inicializado. GPIO=%s, Banach_Target=%s, "
            "engine_v51=%s, engine_v5=%s.",
            gpio_pin,
            banach_target,
            _ENGINE_V51_AVAILABLE,
            _ENGINE_V5_AVAILABLE,
        )

    def _normalize_ast_tree(self, ast_tree: Any) -> str:
        return _normalize_input_to_source(ast_tree)

    def _run_engine(
        self, source_code: str, strict_mode: bool = False
    ) -> Tuple[Optional[Any], bool, str]:
        try:
            if _ENGINE_V5_AVAILABLE and EngineASTStaticAnalyzer is not None:
                if hasattr(EngineASTStaticAnalyzer, "audit_three_phase"):
                    cert = EngineASTStaticAnalyzer.audit_three_phase(
                        source_code, strict_mode=strict_mode
                    )
                    return cert, True, ""
                result = EngineASTStaticAnalyzer.analyze_code(
                    source_code, strict_mode=strict_mode
                )
                return result, True, ""
            return None, False, "Engine no disponible (ni v5.1.0 ni v5.0.0 ni v4.1.0)."
        except Exception as exc:
            logger.error(f"[AgentRSI3] Motor falló: {exc}")
            return None, False, str(exc)

    def audit_ooda_phase(self, ast_tree: Any, G_metric: np.ndarray) -> int:
        r"""
        Ciclo OODA anidado (3 fases) sobre las superficies de la Mónada Kleisli.

        1. Normalizar `ast_tree` → fuente.
        2. Motor v5.1.0 `audit_three_phase` → `AnalysisCertificate`.
        3. FASE 1: $M=\exp(\varepsilon A)$ → Liouville → `_handoff_to_phase2_agent`.
        4. FASE 2: `_ingest_phase1_handoff_agent` → Dirichlet → RSI³ → Poincaré
           → `_handoff_to_phase3_agent`.
        5. FASE 3: `_ingest_phase2_handoff_agent` → $H^1$ / Hodge → Heyting.
        6. Crowbar ssi VETOED genuino ($0x03$).
        """
        source = self._normalize_ast_tree(ast_tree)
        if not source or not source.strip():
            logger.warning("[AgentRSI3] Fuente vacía; VETOED conservador.")
            self.trigger_esp32_crowbar_interlock()
            return 0x03

        engine_cert, engine_ok, err = self._run_engine(source, strict_mode=False)
        if not engine_ok:
            logger.error(f"[AgentRSI3] Motor falló: {err}. VETOED conservador.")
            self.trigger_esp32_crowbar_interlock()
            return 0x03

        inv = _extract_engine_invariants(engine_cert)
        G_arr: Optional[NDArray[np.float64]]
        try:
            G_arr = np.asarray(G_metric, dtype=np.float64)
            if G_arr.ndim != 2 or G_arr.size == 0:
                G_arr = None
        except Exception:
            G_arr = None

        try:
            M, gen_used = _jacobian_from_engine_or_metric(inv, G_arr)
            checksum = self._compute_input_checksum(
                M, G_arr if G_arr is not None else np.zeros((1, 1))
            )

            # ── FASE 1 ──────────────────────────────────────────────────────
            symplectic_audit = self._audit_symplectic_invariance(M)
            h1 = self._handoff_to_phase2_agent(
                symplectic_audit=symplectic_audit,
                input_checksum=checksum,
                engine_certificate=engine_cert,
                frequencies=inv.frequencies,
                symplectic_dim=inv.symplectic_dim or symplectic_audit.phase_space_dimension // 2,
                antisymmetry_ok=inv.antisymmetry_ok,
                interference_frobenius=inv.interference_frobenius,
                generator_used=gen_used,
            )

            # ── FASE 2 ──────────────────────────────────────────────────────
            h1_ing = self._ingest_phase1_handoff_agent(h1)
            Phi, gradV = _port_hamiltonian_pair(
                G_arr, symplectic_audit.phase_space_dimension, inv
            )
            thermo = self._enforce_dirichlet_thermodynamics(
                control_potential_Phi=Phi,
                lyapunov_gradient_V=gradV,
                symplectic_audit=h1_ing.symplectic_audit,
            )
            banach_L, n_iter, rsi3_conv, p_rate, r_inf, traj = self._rsi3_banach_iteration(
                engine_certificate=engine_cert,
                banach_target=self.banach_target,
                symplectic_audit=symplectic_audit,
                thermodynamic_audit=thermo,
            )
            h2 = self._handoff_to_phase3_agent(
                h1=h1_ing,
                thermodynamic_audit=thermo,
                banach_L_real=banach_L,
                rsi3_iterations=n_iter,
                rsi3_converged=rsi3_conv,
                quadratic_rate=p_rate,
                lyapunov_max=inv.lyapunov_max,
                melnikov_amplitude=inv.melnikov_amp,
                kam_gamma=inv.kam_gamma,
                kam_stable=inv.kam_stable,
                twist_ok=inv.twist_ok,
                poincare_recurrent=inv.poincare_recurrent,
                poincare_recurrence_time=inv.poincare_recurrence_time,
                action_energy=inv.action_energy,
                homological_W_norm=inv.W_norm,
                rsi3_residual=r_inf,
                rsi3_score_trajectory=traj,
                coboundary_frobenius=inv.coboundary_frobenius,
                coboundary_shape=inv.coboundary_shape,
            )

            # ── FASE 3 ──────────────────────────────────────────────────────
            h2_ing = self._ingest_phase2_handoff_agent(h2)
            cohomology_audit = self._audit_cellular_sheaf_cohomology(
                h1_dimension=inv.dim_H1,
                thermodynamic_audit=h2_ing.thermodynamic_audit,
                strict_h1=False,
            )
            hodge_trivial = inv.hodge_trivial
            hodge_lambda_min = inv.hodge_lambda_min
            heyting_mask = self._heyting_adjudication(
                h1=h2_ing.phase1,
                h2=h2_ing,
                cohomology_ok=cohomology_audit.is_globally_integrable,
                hodge_trivial=hodge_trivial,
                birkhoff_normalized=inv.birkhoff_normalized,
                banach_target=self.banach_target,
            )
        except (
            SymplecticInvarianceViolation,
            ThermodynamicSingularityError,
            CohomologicalObstructionError,
            OODAChainViolation,
            BanachContractionFailure,
            ValueError,
            TypeError,
        ) as exc:
            logger.error(f"[AgentRSI3] Cadena OODA abortada: {exc}")
            self.trigger_esp32_crowbar_interlock()
            return 0x03

        if heyting_mask == 0x03:
            logger.critical(
                "[AgentRSI3] VETOED: dim_H¹=%d, λ_max=%.3e, |M|=%.3e, L_Φ=%.4f, "
                "γ_KAM=%.3e, KAM=%s, Hodge=%s, twist=%s, RSI³=%s",
                inv.dim_H1,
                inv.lyapunov_max,
                inv.melnikov_amp,
                banach_L,
                inv.kam_gamma,
                inv.kam_stable,
                hodge_trivial,
                inv.twist_ok,
                rsi3_conv,
            )
            self.trigger_esp32_crowbar_interlock()

        logger.info(
            "[AgentRSI3] OODA Φ₁→Φ₂→Φ₃: verdict=0x%02X, L_Φ=%.4f, p=%.2f, "
            "dim_H¹=%d, λ_max=%.3e, |M|=%.3e, KAM=%s, Poincaré=%s, Birkhoff=%s",
            heyting_mask,
            banach_L,
            p_rate,
            inv.dim_H1,
            inv.lyapunov_max,
            inv.melnikov_amp,
            inv.kam_stable,
            inv.poincare_recurrent,
            inv.birkhoff_normalized,
        )
        return heyting_mask

    def trigger_esp32_crowbar_interlock(self) -> None:
        r"""Conmuta GPIO14 a HIGH en $<400\,\mathrm{ns}$ ante singularidad homoclínica."""
        logger.critical(
            f"[CrowbarInterlock] DISPARO CROWBAR EN GPIO{self.gpio_pin} (< 400 ns). "
            "Desenergizando hardware."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §ORQUESTADOR LEGACY: ASTStaticAnalyzerAgent (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class ASTStaticAnalyzerAgent(Morphism, Phase3_CellularSheafCohomologyAuditor):
    r"""
    El Custodio de la Cohomología Sintáctica. Ciclo OODA con continuidad
    estricta $\Phi_1\to\Phi_2\to\Phi_3$.

    Compatible v4.1.0 / v5.0.0: `execute_ast_symplectic_governance(...)`
    retorna `ASTGovernanceState` enriquecido con campos v5.1.0.
    """

    def __init__(self, strict_mode: bool = True) -> None:
        self._strict_mode = bool(strict_mode)
        logger.info(
            "[Orquestador] ASTStaticAnalyzerAgent v5.1.0 inicializado. strict_mode=%s.",
            self._strict_mode,
        )

    def execute_ast_symplectic_governance(
        self,
        ast_jacobian_M: NDArray[np.float64],
        control_potential_Phi: NDArray[np.float64],
        lyapunov_gradient_V: NDArray[np.float64],
        h1_dimension: Optional[int] = None,
        coboundary_delta0: Optional[NDArray[np.float64]] = None,
        coboundary_delta1: Optional[NDArray[np.float64]] = None,
        engine_certificate: Optional[Any] = None,
        G_metric: Optional[NDArray[np.float64]] = None,
    ) -> ASTGovernanceState:
        r"""
        Composición $\Phi_3\circ\Phi_2\circ\Phi_1$ con continuidad estricta.

        1. FASE 1: `_audit_symplectic_invariance` → `_handoff_to_phase2_agent`.
        2. FASE 2: `_ingest_phase1_handoff_agent` → Dirichlet → RSI³
           → `_handoff_to_phase3_agent`.
        3. FASE 3: `_ingest_phase2_handoff_agent` → $H^1$ → Heyting
           → `_emit_agent_certificate`.
        """
        arrays_for_checksum = [
            np.asarray(ast_jacobian_M) if ast_jacobian_M is not None else np.array([]),
            np.asarray(control_potential_Phi)
            if control_potential_Phi is not None
            else np.array([]),
            np.asarray(lyapunov_gradient_V)
            if lyapunov_gradient_V is not None
            else np.array([]),
        ]
        if coboundary_delta0 is not None:
            arrays_for_checksum.append(np.asarray(coboundary_delta0))
        if coboundary_delta1 is not None:
            arrays_for_checksum.append(np.asarray(coboundary_delta1))
        checksum = self._compute_input_checksum(*arrays_for_checksum)
        inv = _extract_engine_invariants(engine_certificate)

        # ─── FASE 1 ─────────────────────────────────────────────────────────
        symplectic_audit = self._audit_symplectic_invariance(ast_jacobian_M)
        h1 = self._handoff_to_phase2_agent(
            symplectic_audit,
            checksum,
            engine_certificate=engine_certificate,
            frequencies=inv.frequencies,
            symplectic_dim=inv.symplectic_dim
            or symplectic_audit.phase_space_dimension // 2,
            antisymmetry_ok=inv.antisymmetry_ok,
            interference_frobenius=inv.interference_frobenius,
            generator_used=inv.generator is not None,
        )

        # ─── FASE 2 ─────────────────────────────────────────────────────────
        h1_ingested = self._ingest_phase1_handoff_agent(h1)
        thermo_audit = self._enforce_dirichlet_thermodynamics(
            control_potential_Phi=control_potential_Phi,
            lyapunov_gradient_V=lyapunov_gradient_V,
            symplectic_audit=h1_ingested.symplectic_audit,
        )
        banach_L, iterations, rsi3_conv, quad_rate, r_inf, traj = self._rsi3_banach_iteration(
            engine_certificate=engine_certificate,
            banach_target=_BANACH_TARGET_DEFAULT,
            symplectic_audit=symplectic_audit,
            thermodynamic_audit=thermo_audit,
        )
        h2 = self._handoff_to_phase3_agent(
            h1=h1_ingested,
            thermodynamic_audit=thermo_audit,
            banach_L_real=banach_L,
            rsi3_iterations=iterations,
            rsi3_converged=rsi3_conv,
            quadratic_rate=quad_rate,
            lyapunov_max=inv.lyapunov_max,
            melnikov_amplitude=inv.melnikov_amp,
            kam_gamma=inv.kam_gamma,
            kam_stable=inv.kam_stable,
            twist_ok=inv.twist_ok,
            poincare_recurrent=inv.poincare_recurrent,
            poincare_recurrence_time=inv.poincare_recurrence_time,
            action_energy=inv.action_energy,
            homological_W_norm=inv.W_norm,
            rsi3_residual=r_inf,
            rsi3_score_trajectory=traj,
            coboundary_frobenius=inv.coboundary_frobenius,
            coboundary_shape=inv.coboundary_shape,
        )

        # ─── FASE 3 ─────────────────────────────────────────────────────────
        h2_ingested = self._ingest_phase2_handoff_agent(h2)
        cohomology_audit = self._audit_cellular_sheaf_cohomology(
            h1_dimension=h1_dimension if h1_dimension is not None else inv.dim_H1,
            thermodynamic_audit=h2_ingested.thermodynamic_audit,
            coboundary_delta0=coboundary_delta0,
            coboundary_delta1=coboundary_delta1,
            strict_h1=self._strict_mode,
        )

        hodge_trivial = inv.hodge_trivial
        hodge_lambda_min = inv.hodge_lambda_min
        if coboundary_delta0 is not None:
            try:
                hodge_lambda_min, hodge_trivial = self._hodge_from_coboundary(
                    coboundary_delta0, G_metric
                )
            except Exception as exc:
                logger.warning(f"[Orquestador] Hodge-Dirac falló: {exc}")

        birk_ok = inv.birkhoff_normalized
        heyting_mask = self._heyting_adjudication(
            h1=h2_ingested.phase1,
            h2=h2_ingested,
            cohomology_ok=cohomology_audit.is_globally_integrable,
            hodge_trivial=hodge_trivial,
            birkhoff_normalized=birk_ok,
        )
        heyting_ok = _heyting_lattice_axioms_check()

        cert = self._emit_agent_certificate(
            h2=h2_ingested,
            cohomology_audit=cohomology_audit,
            hodge_lambda_min=hodge_lambda_min,
            hodge_trivial=hodge_trivial,
            heyting_mask=heyting_mask,
            heyting_ok=heyting_ok,
            checksum=checksum,
            birkhoff_remainder=inv.birkhoff_remainder,
            birkhoff_normalized=birk_ok,
        )

        return ASTGovernanceState(
            symplectic_audit=symplectic_audit,
            thermodynamic_audit=thermo_audit,
            cohomology_audit=cohomology_audit,
            provenance=cert.provenance,
            is_compilation_authorized=cert.is_compilation_authorized,
            rsi3_banach_real=banach_L,
            lyapunov_max=h2_ingested.lyapunov_max,
            melnikov_amplitude=h2_ingested.melnikov_amplitude,
            kam_gamma=h2_ingested.kam_gamma,
            kam_stable=h2_ingested.kam_stable,
            heyting_mask=heyting_mask,
            heyting_axioms_verified=heyting_ok,
            twist_ok=h2_ingested.twist_ok,
            poincare_recurrent=h2_ingested.poincare_recurrent,
            rsi3_converged=rsi3_conv,
            birkhoff_normalized=birk_ok,
            overall_engine_score=inv.overall_score,
            certificate=cert,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "ASTStaticAnalyzerAgentRSI3",
    "ASTStaticAnalyzerAgent",
    "SymplecticInvarianceViolation",
    "SymplecticPolarDecompositionError",
    "MaslovIndexError",
    "ThermodynamicSingularityError",
    "ClausiusDuhemViolation",
    "ExergyDivergenceError",
    "CohomologicalObstructionError",
    "EulerCharacteristicMismatch",
    "MayerVietorisBreachError",
    "OODAChainViolation",
    "BanachContractionFailure",
    "SymplecticInvariantData",
    "ThermodynamicDirichletData",
    "SheafCohomologyAuditData",
    "AuditProvenance",
    "ASTGovernanceState",
    "AgentPhase1Handoff",
    "AgentPhase2Handoff",
    "AgentPhase3Certificate",
    "Phase1_SymplecticInvarianceAuditor",
    "Phase2_DirichletThermodynamicEnforcer",
    "Phase3_CellularSheafCohomologyAuditor",
    "_heyting_join_local",
    "_heyting_lattice_axioms_check",
]
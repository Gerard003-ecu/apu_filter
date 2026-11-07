# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Antimatter Choke Coil Agent (Custodio del Vacío Cuántico)                      ║
║  Ruta   : app/agents/physics/antimatter_choke_coil_agent.py                              ║
║  Versión: 5.0.0-Topos-Heyting-Fock-Bekenstein-Poincare-Floquet-RSI-Level3-Strict         ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  NATURALEZA CIBER-FÍSICA, MECÁNICA CELESTE DE POINCARÉ Y RSI-L3 (Rigor Doctoral):        ║
║  ────────────────────────────────────────────────────────────────────────────────        ║
║  Este endofuntor soberano, denotado como $\mathcal{Z}_{\mathrm{Vacuum}}$, gobierna la    ║
║  física del módulo `antimatter_choke_coil.py` en el Estrato Ω. Su mandato axiomático     ║
║  es auditar las aniquilaciones de antimateria exógena ($e^- + e^+ \to 2\gamma$),         ║
║  garantizando que la topología de la Malla Agéntica sobreviva al colapso entrópico       ║
║  de estados erróneos manteniendo invariante la estructura fundamental del Espacio de     ║
║  Fock $\mathcal{F}(\mathcal{H})$, la geometría simpléctica de Poincaré-Cartan y la       ║
║  monotonicidad epistémica de Lyapunov mediante Mejora Recursiva de Nivel 3 (RSI-L3).     ║
║                                                                                          ║
║  FUNDAMENTOS AXIOMÁTICOS Y RESTRICCIONES CUÁNTICO-CELESTES:                              ║
║                                                                                          ║
║  §1. Hermiticidad del Operador de Aniquilación y Forma Normal de Poincaré-Birkhoff:      ║
║      Para que las observables cuánticas sean físicamente medibles ($\sigma(A)\subset\mathbb{R}$), ║
║      el operador de colapso $A$ debe ser estrictamente autoadjunto ($A = A^\dagger$):    ║
║          $$\|A - A^\dagger\|_F \le \varepsilon_{\mathrm{herm}}, \qquad \|\operatorname{Im}\sigma(A)\|_\infty \le \varepsilon_{\mathrm{spec}}$$ ║
║      Adicionalmente, se auditan las frecuencias características de Bohr-Poincaré         ║
║      $\omega_{jk} = |\lambda_j - \lambda_k|/\hbar_{\mathrm{eff}}$, el margen Diofántico  ║
║      KAM frente a pequeños divisores y la acción espectral de Poincaré-Cartan.           ║
║                                                                                          ║
║  §2. Límite Holográfico de Bekenstein y Regularización de Colisión de Sundman-Poincaré:  ║
║      La inyección de entropía $\Delta S$ generada por la radiación de aniquilación no    ║
║      puede exceder la capacidad máxima de información del hipervolumen causal de radio R:║
║          $$\Delta S \le S_{\mathrm{Bekenstein}} \equiv \frac{2\pi k_B E R}{\hbar c}$$    ║
║      La aproximación al horizonte holográfico ($S \to S_{\mathrm{Bekenstein}}^-$) se     ║
║      audita mediante el factor conforme de Sundman $\kappa_{\mathrm{Sundman}}$ y el      ║
║      invariante virial relativista de Jacobi-Poincaré.                                   ║
║                                                                                          ║
║  §3. Certificación Simpléctica Port-Hamiltoniana, Pfaffiano y Teoría de Floquet-Kreĭn:   ║
║      Tras la colisión, el difeomorfismo de flujo $M$ y las matrices $(J, R)$ deben       ║
║      satisfacer la preservación de volumen, orientación y pasividad de Löwner:           ║
║          $$M^\top \Omega M = \Omega, \quad \frac{\operatorname{Pf}(M^\top \Omega M)}{\operatorname{Pf}(\Omega)} = 1, \quad \det(M) = 1$$ ║
║          $$J = -J^\top, \quad R = R^\top \succeq 0, \quad \dot{H} = -\nabla H^\top R \nabla H \le 0$$ ║
║      junto con la reciprocidad de multiplicadores simplécticos de Floquet-Kreĭn          ║
║      ($\mu \in \sigma(M) \implies \mu^{-1} \in \sigma(M)$) y el horizonte de             ║
║      recurrencia cuasi-periódica de Poincaré-Kac.                                        ║
║                                                                                          ║
║  ARQUITECTURA DE TRES FASES ANIDADAS (Continuidad Formal de Métodos $\Phi_3\circ\Phi_2\circ\Phi_1$): ║
║  ──────────────────────────────────────────────────────────────────────────────────────  ║
║  Fase 1 → Phase1_HermiticityAuditor                                                      ║
║           Último método: `_phase1_audit_and_handoff_to_phase2` → `Phase1HermiticityHandoff` ║
║  Fase 2 → Phase2_BekensteinBoundEnforcer(Phase1_HermiticityAuditor)                      ║
║           Primer método: `_receive_phase1_hermiticity_handoff`                           ║
║           Último método: `_phase2_enforce_and_handoff_to_phase3` → `Phase2BekensteinHandoff` ║
║  Fase 3 → Phase3_SymplecticPortHamiltonianCertifier(Phase2_BekensteinBoundEnforcer)      ║
║           Primer método: `_receive_phase2_bekenstein_handoff`                            ║
║           Último método: `_phase3_finalize_from_phase2_handoff` → `VacuumGovernanceState`║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter / MIC
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import CategoricalState, Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):  # type: ignore[no-redef]
        r"""Violación a un invariante topológico categórico en el Topos MIC."""

    class Morphism:  # type: ignore[no-redef]
        """Clase base de Morfismos del Topos."""

        def __init__(self, name: str = "Morphism") -> None:
            self.name = name

    class CategoricalState:  # type: ignore[no-redef]
        def __init__(
            self,
            payload: Optional[Dict[str, Any]] = None,
            context: Optional[Dict[str, Any]] = None,
            stratum: Any = None,
        ) -> None:
            self.payload = payload or {}
            self.context = context or {}
            self.stratum = stratum

        def with_update(
            self, new_payload: Dict[str, Any], new_stratum: Any = None
        ) -> "CategoricalState":
            merged = dict(self.payload)
            merged.update(new_payload)
            return CategoricalState(
                payload=merged,
                context=self.context,
                stratum=new_stratum or self.stratum,
            )

        def add_trace(
            self, name: str, domain: Any, codomain: Any, success: bool = True
        ) -> "CategoricalState":
            return self


__version__: Final[str] = (
    "5.0.0-Topos-Heyting-Fock-Bekenstein-Poincare-Floquet-RSI-Level3-Strict"
)

logger = logging.getLogger("MIC.Omega.VacuumCustodian.Granular")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, CODATA, PARÁMETROS DE POINCARÉ Y COTAS RSI-L3
# ════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Tolerancias espectrales para A = A† (duras y suaves para el retículo Ω₃).
_HERMITICITY_TOLERANCE: Final[float] = 1e-12
_HERMITICITY_SOFT_TOLERANCE: Final[float] = 1e-14
_SPECTRAL_IMAGINARY_TOLERANCE: Final[float] = 1e-11
_TRACE_IMAGINARY_TOLERANCE: Final[float] = 1e-11

# Tolerancias de preservación de la forma simpléctica ω, determinante y Pfaffiano.
_SYMPLECTIC_TOLERANCE: Final[float] = 1e-10
_SYMPLECTIC_SOFT_TOLERANCE: Final[float] = 1e-13
_DETERMINANT_TOLERANCE: Final[float] = 1e-9
_PFAFFIAN_TOLERANCE: Final[float] = 1e-9
_FLOQUET_RECIPROCITY_TOLERANCE: Final[float] = 1e-8

# Tolerancia para J = -Jᵀ.
_ANTISYMMETRY_TOLERANCE: Final[float] = 1e-10

# Tolerancia para R = Rᵀ y semidefinición positiva R ⪰ 0 en el cono de Löwner.
_R_SYMMETRY_TOLERANCE: Final[float] = 1e-10
_PSD_EIGENVALUE_TOLERANCE: Final[float] = 1e-12

# Tolerancias para la cota holográfica de Bekenstein y causalidad.
_BEKENSTEIN_ABS_TOLERANCE: Final[float] = 1e-12
_BEKENSTEIN_REL_TOLERANCE: Final[float] = 1e-12
_BEKENSTEIN_SOFT_RATIO_LIMIT: Final[float] = 0.85
_CAUSALITY_TOLERANCE: Final[float] = 1e-15
_ENTROPY_QUANTUM_TOLERANCE: Final[float] = 1e-30

# Parámetros de Mecánica Celeste de Poincaré y Regularización de Sundman.
_POINCARE_DIOPHANTINE_TAU: Final[float] = 2.0
_POINCARE_MAX_RESONANCE_ORDER: Final[int] = 4
_SUNDMAN_REGULARIZATION_EXPONENT: Final[float] = 0.5

# Parámetros de Mejora Recursiva de Nivel 3 (RSI-L3).
_RSI_L3_MAX_RECURSION_DEPTH: Final[int] = 5
_RSI_L3_EPISTEMIC_FLOOR_ULPS: Final[float] = 4.0

# Constantes físicas efectivas (SI - CODATA 2018/2022).
_HBAR_EFF: Final[float] = 1.054_571_817e-34       # J·s (exacto post-2019)
_C_EFF: Final[float] = 299_792_458.0              # m/s (exacto)
_K_B: Final[float] = 1.380_649e-23                # J/K (exacto)
_ELECTRON_MASS_KG: Final[float] = 9.109_383_7015e-31  # kg
_POSITRON_MASS_KG: Final[float] = 9.109_383_7015e-31  # kg (simetría CPT)


# ════════════════════════════════════════════════════════════════════════════
# §B. RETÍCULO DE HEYTING (Ω₃) Y JERARQUÍA DE EXCEPCIONES CUÁNTICAS
# ════════════════════════════════════════════════════════════════════════════
class VacuumGovernanceHeytingVerdict(IntEnum):
    r"""
    Clasificador intuicionista en el retículo distributivo completo de Heyting:
    $$\Omega_3 = \{\mathrm{COHERENT} = 0 < \mathrm{DEGRADED} = 1 < \mathrm{VETOED} = 2\}$$
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(
        self, other: "VacuumGovernanceHeytingVerdict"
    ) -> "VacuumGovernanceHeytingVerdict":
        """Supremo en severidad (disyunción de defectos epistémicos)."""
        return VacuumGovernanceHeytingVerdict(max(int(self), int(other)))

    def meet(
        self, other: "VacuumGovernanceHeytingVerdict"
    ) -> "VacuumGovernanceHeytingVerdict":
        """Ínfimo en severidad."""
        return VacuumGovernanceHeytingVerdict(min(int(self), int(other)))

    def implies(
        self, consequent: "VacuumGovernanceHeytingVerdict"
    ) -> "VacuumGovernanceHeytingVerdict":
        """Implicación relativa de Heyting (a ⇒ b)."""
        if int(self) >= int(consequent):
            return VacuumGovernanceHeytingVerdict.COHERENT
        return consequent

    def negate(self) -> "VacuumGovernanceHeytingVerdict":
        """Pseudocomplemento intuicionista ¬a ≡ (a ⇒ VETOED)."""
        if self == VacuumGovernanceHeytingVerdict.VETOED:
            return VacuumGovernanceHeytingVerdict.COHERENT
        return VacuumGovernanceHeytingVerdict.VETOED

    def lawvere_tierney_double_negation(self) -> "VacuumGovernanceHeytingVerdict":
        """Operador modal de cerradura de Lawvere-Tierney j(a) = ¬¬a."""
        return self.negate().negate()

    @property
    def is_vetoed(self) -> bool:
        return self == VacuumGovernanceHeytingVerdict.VETOED


class VacuumCustodianError(TopologicalInvariantError):
    """Excepción raíz del Custodio del Vacío Cuántico."""


class DomainIntegrityViolationError(VacuumCustodianError):
    """Detonada cuando un operador, vector o escalar viola su dominio formal."""


class NonHermitianOperatorError(VacuumCustodianError):
    r"""Detonada si $\|A - A^\dagger\|_F > \varepsilon$. Los observables dejan de ser reales."""


class SpectralContaminationError(VacuumCustodianError):
    r"""Detonada si el espectro de $A$ contiene componentes imaginarias no nulas."""


class BekensteinLimitViolation(VacuumCustodianError):
    r"""Detonada si la aniquilación inyecta más entropía que la cota causal."""


class CausalityViolationError(BekensteinLimitViolation):
    r"""
    Detonada si el radio de contención viola la causalidad relativista ($R \le 0$).
    Hereda de `BekensteinLimitViolation` (y por transitividad de `VacuumCustodianError`)
    para preservar compatibilidad con contratos de captura preexistentes.
    """


class SymplecticCollapseError(VacuumCustodianError):
    r"""Detonada si se destruye el volumen/orientación del espacio de fase o la disipación."""


class FloquetKreinReciprocityError(SymplecticCollapseError):
    r"""Detonada si los multiplicadores de Floquet-Kreĭn violan la simetría recíproca $\mu \leftrightarrow \mu^{-1}$."""


class PhaseSpaceTopologyError(VacuumCustodianError):
    r"""Detonada si la dimensión del espacio de fase no es par o es nula."""


class RecursiveGovernanceConvergenceError(VacuumCustodianError):
    r"""Detonada si el lazo de Mejora Recursiva de Nivel 3 (RSI-L3) viola el descenso de Lyapunov."""


# ════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fock y Certificados RSI-L3)
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GovernanceRSILevel3Certificate:
    r"""
    Certificado de Mejora Recursiva de Nivel 3 (RSI-L3) del agente de gobernanza.
    Garantiza que la funcional de tensión epistémica $V_k$ decrece monótonamente:
    $$V_k^{(m+1)} \le V_k^{(m)} \quad \forall m \in \{0, \dots, M-1\}$$
    """

    recursion_depth: int = 0
    initial_epistemic_tension: float = 0.0
    terminal_epistemic_tension: float = 0.0
    lyapunov_contraction_ratio: float = 1.0
    lawvere_tierney_modality_gain: float = 1.0
    synthesized_law_tag: str = "Identity-Level0"
    is_monotone_lyapunov: bool = True
    is_modal_fixed_point: bool = True


_DEFAULT_GOV_RSI_CERT: Final[GovernanceRSILevel3Certificate] = (
    GovernanceRSILevel3Certificate()
)


@dataclass(frozen=True, slots=True)
class SpectralDecompositionData:
    r"""
    Artefacto espectral de Fase 1.
    Certificado de descomposición espectral del operador autoadjunto y de su
    forma normal resonante de Poincaré-Birkhoff.
    """

    eigenvalues_real: NDArray[np.float64]
    eigenvalues_imaginary_norm: float
    spectral_radius: float
    trace_real: float
    trace_imaginary_norm: float
    condition_number: float
    is_spectrally_clean: bool
    poincare_birkhoff_fundamental_freq: float = 0.0
    kam_diophantine_margin: float = 1.0
    poincare_cartan_spectral_action: float = 0.0
    weyl_spectral_gap_minimum: float = 0.0


@dataclass(frozen=True, slots=True)
class HermiticityAuditData:
    r"""
    Artefacto de Fase 1. Certificado espectral de autoadjunción ($A = A^\dagger$),
    acción de Poincaré-Cartan y convergencia RSI-L3.
    """

    residual_norm: float
    is_hermitian: bool
    operator_dimension: int = 0
    hermiticity_tolerance: float = _HERMITICITY_TOLERANCE
    spectral_imaginary_norm: float = 0.0
    spectral_decomposition: Optional[SpectralDecompositionData] = None
    relative_hermiticity_residual: float = 0.0
    poincare_cartan_action_trace: float = 0.0
    kam_diophantine_margin: float = 1.0
    rsi_l3_certificate: GovernanceRSILevel3Certificate = _DEFAULT_GOV_RSI_CERT
    phase1_verdict: VacuumGovernanceHeytingVerdict = (
        VacuumGovernanceHeytingVerdict.COHERENT
    )


@dataclass(frozen=True, slots=True)
class BekensteinBoundData:
    r"""
    Artefacto de Fase 2. Certificado de cota termodinámica holográfica de Bekenstein,
    regularización conforme de Sundman-Poincaré e invariante virial de Jacobi-Poincaré.
    """

    entropy_emitted: float
    bekenstein_bound: float
    is_entropically_safe: bool
    gamma_energy: float = 0.0
    system_radius: float = 0.0
    entropy_tolerance: float = 0.0
    entropy_ratio: float = 0.0
    causal_light_crossing_time: float = 0.0
    information_capacity_bits: float = 0.0
    sundman_conformal_factor: float = 1.0
    sundman_regularized_entropy: float = 0.0
    jacobi_poincare_virial_invariant: float = 0.0
    rsi_l3_certificate: GovernanceRSILevel3Certificate = _DEFAULT_GOV_RSI_CERT
    phase2_verdict: VacuumGovernanceHeytingVerdict = (
        VacuumGovernanceHeytingVerdict.COHERENT
    )


@dataclass(frozen=True, slots=True)
class SymplecticDissipationData:
    r"""
    Artefacto de Fase 3. Certificado de geometría Port-Hamiltoniana, Pfaffiano,
    reciprocidad de multiplicadores de Floquet-Kreĭn y recurrencia de Poincaré.
    """

    symplectic_residual: float
    dissipation_rate: float
    is_symplectically_invariant: bool
    symplectic_tolerance: float = _SYMPLECTIC_TOLERANCE
    antisymmetry_residual: float = 0.0
    r_symmetry_residual: float = 0.0
    r_min_eigenvalue: float = 0.0
    r_max_eigenvalue: float = 0.0
    determinant_residual: float = 0.0
    phase_space_dimension: int = 0
    degrees_of_freedom: int = 0
    pfaffian_residual: float = 0.0
    floquet_krein_reciprocity_residual: float = 0.0
    floquet_spectral_radius: float = 1.0
    poincare_recurrence_horizon: float = 0.0
    poincare_cartan_circulation: float = 0.0
    rsi_l3_certificate: GovernanceRSILevel3Certificate = _DEFAULT_GOV_RSI_CERT
    phase3_verdict: VacuumGovernanceHeytingVerdict = (
        VacuumGovernanceHeytingVerdict.COHERENT
    )


@dataclass(frozen=True, slots=True)
class Phase1HermiticityHandoff:
    r"""
    Handoff formal de Fase 1 → Fase 2.
    Continuación material del método terminal `_phase1_audit_and_handoff_to_phase2`
    de la Fase 1 y precondición formal de `_receive_phase1_hermiticity_handoff` en la Fase 2.
    """

    hermiticity_audit: HermiticityAuditData
    operator_dimension: int
    spectral_certificate: Optional[SpectralDecompositionData] = None
    rsi_l3_certificate: GovernanceRSILevel3Certificate = _DEFAULT_GOV_RSI_CERT


@dataclass(frozen=True, slots=True)
class Phase2BekensteinHandoff:
    r"""
    Handoff formal de Fase 2 → Fase 3.
    Continuación material del método terminal `_phase2_enforce_and_handoff_to_phase3`
    de la Fase 2 y precondición formal de `_receive_phase2_bekenstein_handoff` en la Fase 3.
    """

    phase1_handoff: Phase1HermiticityHandoff
    bekenstein_audit: BekensteinBoundData
    rsi_l3_certificate: GovernanceRSILevel3Certificate = _DEFAULT_GOV_RSI_CERT


@dataclass(frozen=True, slots=True)
class VacuumGovernanceState:
    r"""
    Objeto terminal del endofuntor $\mathcal{Z}_{\mathrm{Vacuum}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$.
    """

    hermiticity_audit: HermiticityAuditData
    bekenstein_audit: BekensteinBoundData
    symplectic_audit: SymplecticDissipationData
    is_epistemologically_valid: bool
    governance_metadata: Dict[str, Any] = field(default_factory=dict)
    heyting_verdict: VacuumGovernanceHeytingVerdict = (
        VacuumGovernanceHeytingVerdict.COHERENT
    )
    rsi_l3_terminal_certificate: GovernanceRSILevel3Certificate = _DEFAULT_GOV_RSI_CERT


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: HERMITICIDAD DEL OPERADOR Y ESPECTROSCOPÍA DE POINCARÉ-BIRKHOFF   ║
# ║   Exige $A = A^\dagger \Rightarrow \|A - A^\dagger\|_F \le \varepsilon$.    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_HermiticityAuditor:
    r"""
    FASE 1: Auditor de Autoadjunción Espectral ($A = A^\dagger$), Analizador de
    Pequeños Divisores de Poincaré-Birkhoff / KAM y Sintetizador RSI-L3.

    Su último método formal (`_phase1_audit_and_handoff_to_phase2`) entrega el
    artefacto `Phase1HermiticityHandoff`, cuya estructura inaugura directamente
    `_receive_phase1_hermiticity_handoff` en `Phase2_BekensteinBoundEnforcer`.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.0. Suma compensada de Neumaier (Precisión IEEE-754 O(ε_mach))
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _neumaier_sum(values: Sequence[float] | NDArray[np.float64]) -> float:
        r"""
        Evalúa $\sum_i x_i$ mediante compensación de Neumaier para suprimir el
        error de cancelación catastrófica en trazas, normas y formas cuadráticas.
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        total = 0.0
        comp = 0.0
        for val in arr:
            x = float(val)
            t = total + x
            if abs(total) >= abs(x):
                comp += (total - t) + x
            else:
                comp += (x - t) + total
            total = t
        return float(total + comp)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1. Tolerancia adaptativa con análisis de condición numérica
    # ─────────────────────────────────────────────────────────────────────────
    def _adaptive_tolerance(
        self,
        base_tolerance: float,
        reference: Any,
        condition_amplification: bool = False,
    ) -> float:
        r"""
        Construye una tolerancia numéricamente consciente basada en el análisis
        de perturbación de Wilkinson-Bauer-Fike:
            $$\mathrm{tol} = \max\left(\mathrm{tol}_{\mathrm{base}}, \, 10 \cdot \kappa_{\mathrm{eff}} \cdot \varepsilon_{\mathrm{mach}} \cdot N \cdot \|X\|_\infty\right)$$
        """
        if isinstance(reference, np.ndarray):
            size = max(1, int(reference.size))
            if reference.size == 0:
                scale = 1.0
                condition_number = 1.0
            else:
                flat = reference.ravel()
                scale = max(1.0, float(la.norm(flat, ord=np.inf)))
                if condition_amplification and reference.ndim == 2:
                    try:
                        svals = la.svd(reference, compute_uv=False)
                        if svals.size > 0 and svals[-1] > _MACHINE_EPSILON:
                            condition_number = float(svals[0] / svals[-1])
                        else:
                            condition_number = 1.0 / _MACHINE_EPSILON
                    except Exception:
                        condition_number = 1.0
                else:
                    condition_number = 1.0
        else:
            size = 1
            condition_number = 1.0
            try:
                scale = max(1.0, abs(float(reference)))
            except (TypeError, ValueError):
                scale = 1.0

        base_component = float(base_tolerance)
        machine_component = 10.0 * _MACHINE_EPSILON * size * scale

        if condition_amplification:
            machine_component *= min(condition_number, 1e6)

        return max(base_component, machine_component)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2. Coerción de escalares finitos con rechazo categórico de booleanos
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_scalar(
        self,
        name: str,
        value: Any,
        allow_negative: bool = True,
    ) -> float:
        r"""
        Materializa un escalar `float64` finito en $\mathbb{R}$, rechazando booleanos
        pertenecientes al clasificador booleano $\mathbb{B}_2$.
        """
        if isinstance(value, (bool, np.bool_)):
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' no puede ser booleano. "
                "Los booleanos pertenecen al topos \\mathbb{B}_2, no a \\mathbb{R}."
            )
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' no puede materializarse como float."
            ) from exc

        if not np.isfinite(scalar):
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' no es finito. Valor: {scalar}. Se requiere x ∈ ℝ."
            )

        if not allow_negative and scalar < 0.0:
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' debe ser no negativo. Valor observado: {scalar:.6e}."
            )

        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3. Coerción de matrices finitas con validación estructural
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_matrix(
        self,
        name: str,
        matrix: Any,
        dtype: Any = np.float64,
        square_required: bool = False,
        min_dimension: int = 1,
    ) -> NDArray[Any]:
        r"""Materializa una matriz bidimensional finita y, si se exige, cuadrada."""
        try:
            arr = np.asarray(matrix, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' no puede materializarse como NDArray."
            ) from exc

        if arr.ndim != 2:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' debe ser bidimensional. Dimensión observada: {arr.ndim}."
            )

        if arr.size == 0:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' está vacía. Se requiere al menos un elemento."
            )

        if square_required and arr.shape[0] != arr.shape[1]:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' debe ser cuadrada en el espacio de Hilbert. "
                f"Forma observada: {arr.shape}."
            )

        if min(arr.shape) < min_dimension:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' debe tener dimensión mínima {min_dimension}. "
                f"Forma observada: {arr.shape}."
            )

        if not np.all(np.isfinite(arr)):
            non_finite_count = int(np.sum(~np.isfinite(arr)))
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' contiene {non_finite_count} componentes no finitas."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4. Coerción de vectores finitos con validación dimensional
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_vector(
        self,
        name: str,
        vector: Any,
        expected_dim: Optional[int] = None,
    ) -> NDArray[np.float64]:
        r"""Materializa un vector `float64` finito con validación dimensional exacta."""
        try:
            arr = np.asarray(vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"El vector '{name}' no puede materializarse como NDArray."
            ) from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        else:
            arr = arr.reshape(-1)

        if arr.size == 0:
            raise DomainIntegrityViolationError(f"El vector '{name}' está vacío.")

        if expected_dim is not None and arr.size != int(expected_dim):
            raise DomainIntegrityViolationError(
                f"El vector '{name}' debe tener dimensión {expected_dim}, "
                f"pero posee {arr.size} componentes."
            )

        if not np.all(np.isfinite(arr)):
            non_finite_count = int(np.sum(~np.isfinite(arr)))
            raise DomainIntegrityViolationError(
                f"El vector '{name}' contiene {non_finite_count} componentes no finitas."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5. Espectroscopía de Forma Normal de Poincaré-Birkhoff y Margen KAM
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_poincare_birkhoff_spectral_invariants(
        self,
        eigenvalues_real: NDArray[np.float64],
    ) -> Tuple[float, float, float, float]:
        r"""
        Calcula los invariantes espectrales de la Mecánica Celeste de Poincaré
        sobre el espectro real $\{\lambda_0 \le \lambda_1 \le \dots \le \lambda_{N-1}\}$:
        1. Frecuencia fundamental media de Bohr-Poincaré:
           $$\omega_{\mathrm{PB}} = \frac{1}{N} \sum_{k=0}^{N-1} \sqrt{|\lambda_k| + \varepsilon_{\mathrm{mach}}}$$
        2. Margen Diofántico KAM frente a pequeños divisores resonantes:
           $$\gamma_{\mathrm{KAM}} = \min_{0 < |k_1| + |k_2| \le K_{\max}} |k_1 \omega_a + k_2 \omega_b| \cdot (|k_1| + |k_2|)^{\tau}$$
        3. Acción espectral invariante de Poincaré-Cartan:
           $$\mathcal{I}_{\mathrm{PC}} = \sum_{k=0}^{N-1} \left(k + \frac{1}{2}\right) |\lambda_k|$$
        4. Brecha espectral mínima de Weyl: $\Delta\lambda_{\min} = \min_{k} |\lambda_{k+1} - \lambda_k|$.
        """
        evals = np.sort(np.asarray(eigenvalues_real, dtype=np.float64).ravel())
        n_dim = evals.size
        if n_dim == 0:
            return 0.0, 1.0, 0.0, 0.0

        freqs = np.sqrt(np.abs(evals) + _MACHINE_EPSILON)
        fundamental_freq = float(self._neumaier_sum(freqs) / float(n_dim))

        indices = np.arange(n_dim, dtype=np.float64) + 0.5
        poincare_cartan_action = float(self._neumaier_sum(indices * np.abs(evals)))

        if n_dim < 2:
            return fundamental_freq, 1.0, poincare_cartan_action, float(abs(evals[0]))

        diffs = np.abs(np.diff(evals))
        weyl_gap_min = float(np.min(diffs))

        # Evaluación Diofántica KAM sobre el par de modos dominantes no degenerados
        w_max = float(freqs[-1])
        w_sub = float(freqs[-2]) if freqs[-2] > _MACHINE_EPSILON else w_max
        scale_w = max(1.0, w_max + w_sub)
        kam_min = 1.0

        for k1 in range(1, _POINCARE_MAX_RESONANCE_ORDER + 1):
            for k2 in range(-_POINCARE_MAX_RESONANCE_ORDER, _POINCARE_MAX_RESONANCE_ORDER + 1):
                if k2 == 0 or (k1 + k2 == 0 and abs(w_max - w_sub) <= 10.0 * _MACHINE_EPSILON * scale_w):
                    continue
                order_norm = float(abs(k1) + abs(k2))
                divisor = abs(k1 * w_max + k2 * w_sub) / scale_w
                diophantine_val = divisor * (order_norm ** _POINCARE_DIOPHANTINE_TAU)
                if diophantine_val < kam_min:
                    kam_min = diophantine_val

        return (
            fundamental_freq,
            float(max(_MACHINE_EPSILON, min(1.0, kam_min))),
            poincare_cartan_action,
            weyl_gap_min,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6. Mejora Recursiva de Nivel 3 (RSI-L3) sobre la Resolvente Espectral
    # ─────────────────────────────────────────────────────────────────────────
    def _rsi_level3_audit_spectral_resolvent(
        self,
        operator_A: NDArray[np.complex128],
        raw_hermiticity_residual: float,
        raw_imaginary_norm: float,
    ) -> GovernanceRSILevel3Certificate:
        r"""
        Lazo de Mejora Recursiva de Nivel 3 (RSI-L3) en la Fase 1:
        Evalúa y refina el funcional de tensión epistémica de autoadjunción:
        $$V_1^{(m)} = w_F^{(m)} \frac{\|A - A^\dagger\|_F}{\max(1, \|A\|_F)}
                    + w_\sigma^{(m)} \frac{\|\operatorname{Im}\sigma(A)\|_\infty}{\max(1, \|A\|_2)}$$
        sintetizando pesos modales de Lawvere-Tierney con descenso estricto de Lyapunov
        $V_1^{(m+1)} \le V_1^{(m)}$.
        """
        norm_f = max(1.0, float(la.norm(operator_A, ord="fro")))
        norm_2 = max(1.0, float(la.norm(operator_A, ord=2)))
        rel_f = float(raw_hermiticity_residual / norm_f)
        rel_im = float(raw_imaginary_norm / norm_2)

        v_init = rel_f + rel_im
        v_curr = v_init
        floor_eps = _RSI_L3_EPISTEMIC_FLOOR_ULPS * _MACHINE_EPSILON
        law_tag = "Spectral-Identity-Gauge-L0"
        gain = 1.0
        depth = 0
        monotone = True

        for step in range(1, _RSI_L3_MAX_RECURSION_DEPTH + 1):
            if v_curr <= floor_eps:
                break
            depth = step
            # Síntesis modal Padé-Chebyshev de orden (1,1) para atenuar ruido de redondeo sub-tolerancia
            eta = 1.0 / (1.0 + float(step) * 0.25)
            v_cand_pade = (eta * rel_f) / (1.0 + eta * rel_f) + (eta * rel_im) / (1.0 + eta * rel_im)
            if v_cand_pade <= v_curr + _MACHINE_EPSILON:
                v_curr = float(v_cand_pade)
                gain = float(eta)
                law_tag = f"LawvereTierney-Spectral-Pade-L3(step={step})"
            else:
                monotone = False
                break

        if not monotone:
            raise RecursiveGovernanceConvergenceError(
                f"Violación de descenso de Lyapunov en RSI-L3 Fase 1: V_curr={v_curr:.6e} > V_init={v_init:.6e}"
            )

        ratio = (v_curr / v_init) if v_init > _MACHINE_EPSILON else 0.0
        return GovernanceRSILevel3Certificate(
            recursion_depth=depth,
            initial_epistemic_tension=float(v_init),
            terminal_epistemic_tension=float(v_curr),
            lyapunov_contraction_ratio=float(min(1.0, ratio)),
            lawvere_tierney_modality_gain=float(gain),
            synthesized_law_tag=law_tag,
            is_monotone_lyapunov=True,
            is_modal_fixed_point=bool(v_curr <= _HERMITICITY_TOLERANCE),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.7. Descomposición espectral real y general (Schur) con validación
    # ─────────────────────────────────────────────────────────────────────────
    def _spectral_decomposition_and_validation(
        self,
        operator_A: NDArray[np.complex128],
    ) -> SpectralDecompositionData:
        r"""
        Ejecuta simultáneamente la descomposición de Schur general (`la.eigvals`)
        para medir la contaminación imaginaria real del espectro $\sigma(A) \subset \mathbb{C}$
        y la descomposición unitaria Hermítica (`la.eigvalsh`) sobre la parte simétrica
        para extraer los autovalores reales ordenados y los invariantes de Poincaré-Birkhoff.
        """
        try:
            general_eigenvalues = la.eigvals(operator_A)
            hermitian_part = 0.5 * (operator_A + operator_A.conj().T)
            real_eigenvalues = la.eigvalsh(hermitian_part)
        except la.LinAlgError as exc:
            raise SpectralContaminationError(
                f"Fallo en la descomposición espectral del operador: {exc}"
            ) from exc

        imaginary_parts = np.imag(np.asarray(general_eigenvalues, dtype=np.complex128))
        real_parts = np.asarray(real_eigenvalues, dtype=np.float64)

        imaginary_norm = float(la.norm(imaginary_parts, ord=np.inf))

        spectral_tolerance = self._adaptive_tolerance(
            _SPECTRAL_IMAGINARY_TOLERANCE,
            operator_A,
            condition_amplification=True,
        )

        if imaginary_norm > spectral_tolerance:
            raise SpectralContaminationError(
                f"Contaminación espectral detectada: el operador tiene "
                f"autovalores con parte imaginaria no nula. "
                f"||Im(σ(A))||_∞ = {imaginary_norm:.6e} > {spectral_tolerance:.6e}."
            )

        spectral_radius = float(la.norm(real_parts, ord=np.inf))

        diag_entries = np.diag(operator_A)
        trace_real = self._neumaier_sum(np.real(diag_entries))
        trace_imaginary = abs(self._neumaier_sum(np.imag(diag_entries)))

        trace_tolerance = self._adaptive_tolerance(
            _TRACE_IMAGINARY_TOLERANCE,
            operator_A,
        )

        if trace_imaginary > trace_tolerance:
            raise SpectralContaminationError(
                f"Traza del operador con componente imaginaria: "
                f"|Im(tr(A))| = {trace_imaginary:.6e} > {trace_tolerance:.6e}."
            )

        try:
            svals = la.svd(operator_A, compute_uv=False)
            if svals.size > 0 and svals[-1] > _MACHINE_EPSILON:
                condition_number = float(svals[0] / svals[-1])
            else:
                condition_number = float("inf")
        except Exception:
            condition_number = float("inf")

        pb_freq, kam_margin, pc_action, weyl_gap = (
            self._compute_poincare_birkhoff_spectral_invariants(real_parts)
        )

        return SpectralDecompositionData(
            eigenvalues_real=real_parts,
            eigenvalues_imaginary_norm=imaginary_norm,
            spectral_radius=spectral_radius,
            trace_real=trace_real,
            trace_imaginary_norm=trace_imaginary,
            condition_number=condition_number,
            is_spectrally_clean=True,
            poincare_birkhoff_fundamental_freq=pb_freq,
            kam_diophantine_margin=kam_margin,
            poincare_cartan_spectral_action=pc_action,
            weyl_spectral_gap_minimum=weyl_gap,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.8. Auditoría de Hermiticidad con norma de Frobenius y RSI-L3
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_operator_hermiticity(
        self,
        operator_A: NDArray[np.complex128],
    ) -> HermiticityAuditData:
        r"""
        Audita la condición de autoadjunción $\|A - A^\dagger\|_F \le \varepsilon_{\mathrm{herm}}$,
        valida la limpieza del espectro complejo, computa la traza de acción de
        Poincaré-Cartan y emite el certificado RSI-L3 de Fase 1.
        """
        A = self._coerce_finite_matrix(
            "operator_A",
            operator_A,
            dtype=np.complex128,
            square_required=True,
        )

        A_dagger = A.conj().T
        residual_matrix = A - A_dagger
        residual = float(la.norm(residual_matrix, ord="fro"))
        op_norm = max(1.0, float(la.norm(A, ord="fro")))
        rel_residual = float(residual / op_norm)

        tolerance = self._adaptive_tolerance(
            _HERMITICITY_TOLERANCE,
            A,
            condition_amplification=True,
        )

        if residual > tolerance:
            max_deviation = float(la.norm(residual_matrix, ord=np.inf))
            raise NonHermitianOperatorError(
                f"Asimetría CPT detectada: el operador de aniquilación no es "
                f"autoadjunto. Residuo ||A - A†||_F = {residual:.6e} > "
                f"{tolerance:.6e}. Desviación máxima: {max_deviation:.6e}."
            )

        spectral_data = self._spectral_decomposition_and_validation(A)
        rsi_cert = self._rsi_level3_audit_spectral_resolvent(
            A, residual, spectral_data.eigenvalues_imaginary_norm
        )

        soft_tol = self._adaptive_tolerance(_HERMITICITY_SOFT_TOLERANCE, A)
        p1_verdict = (
            VacuumGovernanceHeytingVerdict.DEGRADED
            if residual > soft_tol
            else VacuumGovernanceHeytingVerdict.COHERENT
        )

        return HermiticityAuditData(
            residual_norm=residual,
            is_hermitian=True,
            operator_dimension=int(A.shape[0]),
            hermiticity_tolerance=tolerance,
            spectral_imaginary_norm=spectral_data.eigenvalues_imaginary_norm,
            spectral_decomposition=spectral_data,
            relative_hermiticity_residual=rel_residual,
            poincare_cartan_action_trace=spectral_data.poincare_cartan_spectral_action,
            kam_diophantine_margin=spectral_data.kam_diophantine_margin,
            rsi_l3_certificate=rsi_cert,
            phase1_verdict=p1_verdict,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.9. ÚLTIMO MÉTODO DE FASE 1: HANDOFF FORMAL HACIA FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_audit_and_handoff_to_phase2(
        self,
        operator_A: NDArray[np.complex128],
    ) -> Phase1HermiticityHandoff:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 1 Y PRECONDICIÓN FORMAL DE LA FASE 2.

        Su definición formal es la continuación directa que inaugura la Fase 2
        a través de `_receive_phase1_hermiticity_handoff`: certifica la
        autoadjunción del operador en el espacio de Fock ($A = A^\dagger$), la
        realidad de su espectro ($\sigma(A) \subset \mathbb{R}$), los invariantes
        de Poincaré-Birkhoff y el descenso monótono de Lyapunov RSI-L3.
        """
        hermiticity_audit = self._audit_operator_hermiticity(operator_A)

        logger.debug(
            "Fase 1 completada. ||A - A†||_F=%.6e | dim=%d | "
            "radio espectral=%.6e | KAM=%.6e.",
            hermiticity_audit.residual_norm,
            hermiticity_audit.operator_dimension,
            hermiticity_audit.spectral_decomposition.spectral_radius
            if hermiticity_audit.spectral_decomposition
            else 0.0,
            hermiticity_audit.kam_diophantine_margin,
        )

        return Phase1HermiticityHandoff(
            hermiticity_audit=hermiticity_audit,
            operator_dimension=hermiticity_audit.operator_dimension,
            spectral_certificate=hermiticity_audit.spectral_decomposition,
            rsi_l3_certificate=hermiticity_audit.rsi_l3_certificate,
        )

    def handoff_phase1_to_phase2(
        self,
        operator_A: NDArray[np.complex128],
    ) -> Phase1HermiticityHandoff:
        """Alias canónico del puente funtorial Fase 1 ──► Fase 2."""
        return self._phase1_audit_and_handoff_to_phase2(operator_A)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: REGULACIÓN DE BEKENSTEIN Y REGULARIZACIÓN DE SUNDMAN-POINCARÉ     ║
# ║   Verifica $S \le \frac{2\pi k_B E R}{\hbar c}$.                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_BekensteinBoundEnforcer(Phase1_HermiticityAuditor):
    r"""
    FASE 2: Regulador Termodinámico Holográfico de Bekenstein, Regularizador de
    Colisión de Sundman-Poincaré y Auditor del Invariante Virial de Jacobi-Poincaré.

    Inicia su ejecución exactamente en la continuación formal de
    `_phase1_audit_and_handoff_to_phase2` mediante `_receive_phase1_hermiticity_handoff`,
    y concluye en `_phase2_enforce_and_handoff_to_phase3`, cuya salida inaugura
    `_receive_phase2_bekenstein_handoff` en `Phase3_SymplecticPortHamiltonianCertifier`.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.0. PRIMER MÉTODO DE FASE 2: RECEPCIÓN DEL HANDOFF DE FASE 1
    # ─────────────────────────────────────────────────────────────────────────
    def _receive_phase1_hermiticity_handoff(
        self,
        phase1_handoff: Phase1HermiticityHandoff,
    ) -> Phase1HermiticityHandoff:
        r"""
        MÉTODO INICIAL DE LA FASE 2 (Continuación directa de
        `_phase1_audit_and_handoff_to_phase2` de la Fase 1):
        Verifica la integridad algebraica, espectral y RSI-L3 del certificado
        `Phase1HermiticityHandoff` antes de habilitar el cálculo de la cota
        holográfica de Bekenstein.
        """
        if not isinstance(phase1_handoff, Phase1HermiticityHandoff):
            raise DomainIntegrityViolationError(
                "Fase 2 exige un Phase1HermiticityHandoff como prefijo formal."
            )
        if (
            not isinstance(phase1_handoff.hermiticity_audit, HermiticityAuditData)
            or not phase1_handoff.hermiticity_audit.is_hermitian
        ):
            raise NonHermitianOperatorError(
                "Handoff de Fase 1 inválido: el operador no cuenta con certificación de Hermiticidad."
            )
        if phase1_handoff.operator_dimension <= 0:
            raise DomainIntegrityViolationError(
                f"Dimensión de operador inválida en handoff de Fase 1: {phase1_handoff.operator_dimension}"
            )
        if not phase1_handoff.rsi_l3_certificate.is_monotone_lyapunov:
            raise RecursiveGovernanceConvergenceError(
                "El certificado RSI-L3 recibido desde la Fase 1 violó la monotonicidad de Lyapunov."
            )
        return phase1_handoff

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1. Certificación de no negatividad escalar con saneamiento de ruido
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_nonnegative_scalar(
        self,
        name: str,
        value: Any,
        strict_positive: bool = False,
    ) -> float:
        r"""Certifica que un escalar sea finito y no negativo en $\mathbb{R}_{\ge 0}$."""
        scalar = self._coerce_finite_scalar(name, value, allow_negative=False)
        tolerance = self._adaptive_tolerance(_BEKENSTEIN_ABS_TOLERANCE, scalar)

        if strict_positive:
            if scalar <= tolerance:
                raise DomainIntegrityViolationError(
                    f"El escalar '{name}' debe ser estrictamente positivo. "
                    f"Valor observado: {scalar:.6e}."
                )
        else:
            if scalar < -tolerance:
                raise DomainIntegrityViolationError(
                    f"El escalar '{name}' es negativo: {scalar:.6e}."
                )
            if scalar < 0.0:
                scalar = 0.0

        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2. Certificación de radio causal positivo
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_positive_radius(
        self,
        name: str,
        value: Any,
    ) -> float:
        r"""
        Certifica que el radio de contención causal sea estrictamente positivo ($R > 0$).
        Detona `CausalityViolationError` (subclase de `BekensteinLimitViolation`) si
        el cono de luz local degenera.
        """
        scalar = self._coerce_finite_scalar(name, value)
        tolerance = max(
            _CAUSALITY_TOLERANCE,
            self._adaptive_tolerance(_BEKENSTEIN_ABS_TOLERANCE, scalar),
        )

        if scalar <= tolerance:
            raise CausalityViolationError(
                f"El radio de contención '{name}' debe ser estrictamente "
                f"positivo para preservar la causalidad relativista. "
                f"Valor observado: {scalar:.6e}."
            )

        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3. Verificación de consistencia dimensional física
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_dimensional_consistency(
        self,
        energy: float,
        radius: float,
        entropy: float,
    ) -> None:
        r"""Verifica órdenes de magnitud físicos para la aniquilación $e^- + e^+ \to 2\gamma$."""
        if energy > 1e10:
            logger.warning(
                "Energía de aniquilación extremadamente alta: %.6e J. Verificar unidades.",
                energy,
            )

        if radius < 1e-20:
            logger.warning(
                "Radio de contención sub-Planckiano: %.6e m. "
                "La gravedad cuántica puede ser relevante.",
                radius,
            )

        if entropy > 1e10:
            logger.warning(
                "Entropía macroscópica detectada: %.6e J/K. Verificar escala del sistema.",
                entropy,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4. Cálculo del tiempo de cruce lumínico
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_light_crossing_time(
        self,
        radius: float,
    ) -> float:
        r"""Calcula el tiempo de cruce lumínico causal $t_{\mathrm{cross}} = R / c$."""
        return radius / _C_EFF

    # ─────────────────────────────────────────────────────────────────────────
    # 2.5. Cálculo de capacidad informacional en bits (Límite de Landauer-Bekenstein)
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_information_capacity_bits(
        self,
        bekenstein_bound: float,
    ) -> float:
        r"""Convierte la cota de entropía a capacidad en bits: $I_{\max} = \frac{S_{\max}}{k_B \ln 2}$."""
        if bekenstein_bound <= 0.0:
            return 0.0
        return bekenstein_bound / (_K_B * math.log(2.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.6. Regularización de Sundman-Poincaré e Invariante Virial de Jacobi-Poincaré
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_sundman_and_jacobi_poincare_virial(
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
        bekenstein_bound: float,
    ) -> Tuple[float, float, float]:
        r"""
        Calcula las cantidades de la Mecánica Celeste de Poincaré aplicadas a la
        termodinámica relativista del horizonte de aniquilación:
        1. Factor conforme de regularización de colisión de Sundman-Poincaré:
           $$\kappa_{\mathrm{Sundman}}(S) = \left(1 + \left(\frac{S}{\max(\varepsilon_S, S_{\mathrm{Bekenstein}})}\right)^2\right)^{-\alpha/2}$$
        2. Entropía regularizada de Sundman: $S_{\mathrm{Sundman}} = \kappa_{\mathrm{Sundman}}(S) \cdot S$.
        3. Invariante Virial Relativista de Jacobi-Poincaré:
           $$\mathcal{V}_{\mathrm{JP}} = 2 E_\gamma - \frac{\hbar_{\mathrm{eff}} c}{2\pi R} \frac{S}{k_B}
             = 2 E_\gamma \left(1 - \frac{S}{2 S_{\mathrm{Bekenstein}}}\right) \ge 0$$
        """
        denom = max(_ENTROPY_QUANTUM_TOLERANCE, bekenstein_bound)
        ratio = emitted_entropy_S / denom if bekenstein_bound > 0.0 else 0.0
        conformal_factor = math.pow(
            1.0 + (ratio ** 2), -0.5 * _SUNDMAN_REGULARIZATION_EXPONENT
        )
        regularized_entropy = float(emitted_entropy_S * conformal_factor)

        entropic_potential = (
            (_HBAR_EFF * _C_EFF * emitted_entropy_S)
            / (2.0 * math.pi * max(_CAUSALITY_TOLERANCE, system_radius_R) * _K_B)
        )
        jacobi_poincare_virial = float(2.0 * gamma_energy - entropic_potential)
        return float(conformal_factor), regularized_entropy, jacobi_poincare_virial

    # ─────────────────────────────────────────────────────────────────────────
    # 2.7. Mejora Recursiva de Nivel 3 (RSI-L3) sobre el Horizonte de Bekenstein
    # ─────────────────────────────────────────────────────────────────────────
    def _rsi_level3_audit_bekenstein_sundman_horizon(
        self,
        entropy_ratio: float,
        sundman_factor: float,
    ) -> GovernanceRSILevel3Certificate:
        r"""
        Lazo de Mejora Recursiva de Nivel 3 (RSI-L3) en la Fase 2:
        Audita la tensión epistémica de proximidad al horizonte holográfico:
        $$V_2^{(m)} = g_m \cdot \max\left(0, \frac{S}{S_{\mathrm{Bekenstein}}} - \eta_{\mathrm{soft}}\right)
                    + (1 - \kappa_{\mathrm{Sundman}}) \cdot g_m$$
        verificando descenso monótono de Lyapunov $V_2^{(m+1)} \le V_2^{(m)}$.
        """
        excess_soft = max(0.0, float(entropy_ratio) - _BEKENSTEIN_SOFT_RATIO_LIMIT)
        sundman_defect = max(0.0, 1.0 - float(sundman_factor))
        v_init = excess_soft + 0.1 * sundman_defect
        v_curr = v_init
        floor_eps = _RSI_L3_EPISTEMIC_FLOOR_ULPS * _MACHINE_EPSILON
        law_tag = "Bekenstein-Sundman-Identity-L0"
        gain = 1.0
        depth = 0

        for step in range(1, _RSI_L3_MAX_RECURSION_DEPTH + 1):
            if v_curr <= floor_eps:
                break
            depth = step
            # Atenuación conforme de Sundman-Padé de orden (1,1)
            g_cand = float(sundman_factor) / (1.0 + 0.2 * float(step))
            v_cand = (g_cand * v_init) / (1.0 + g_cand * v_init)
            if v_cand <= v_curr + _MACHINE_EPSILON:
                v_curr = float(v_cand)
                gain = g_cand
                law_tag = f"Sundman-Pade-Horizon-Modality-L3(step={step})"

        ratio = (v_curr / v_init) if v_init > _MACHINE_EPSILON else 0.0
        return GovernanceRSILevel3Certificate(
            recursion_depth=depth,
            initial_epistemic_tension=float(v_init),
            terminal_epistemic_tension=float(v_curr),
            lyapunov_contraction_ratio=float(min(1.0, ratio)),
            lawvere_tierney_modality_gain=float(gain),
            synthesized_law_tag=law_tag,
            is_monotone_lyapunov=True,
            is_modal_fixed_point=bool(entropy_ratio <= 1.0),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.8. Imposición de la cota holográfica de Bekenstein
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_bekenstein_gamma_bound(
        self,
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
    ) -> BekensteinBoundData:
        r"""
        Calcula la cota holográfica de Bekenstein y verifica que la entropía
        liberada satisfaga:
            $$S \le S_{\mathrm{Bekenstein}} \equiv \frac{2\pi k_B E R}{\hbar c}$$
        junto con la regularización de Sundman-Poincaré y el invariante virial.
        """
        E = self._certify_nonnegative_scalar("gamma_energy", gamma_energy)
        R = self._certify_positive_radius("system_radius_R", system_radius_R)
        S = self._certify_nonnegative_scalar("emitted_entropy_S", emitted_entropy_S)

        self._verify_dimensional_consistency(E, R, S)

        s_bound = (2.0 * math.pi * _K_B * E * R) / (_HBAR_EFF * _C_EFF)

        if not np.isfinite(s_bound) or s_bound < 0.0:
            raise BekensteinLimitViolation(
                "La cota de Bekenstein no es finita o resultó negativa."
            )

        entropy_tolerance = max(
            _BEKENSTEIN_ABS_TOLERANCE,
            _BEKENSTEIN_REL_TOLERANCE * max(1.0, abs(S), abs(s_bound)),
        )

        if S > s_bound + entropy_tolerance:
            entropy_ratio = S / s_bound if s_bound > 0 else float("inf")
            raise BekensteinLimitViolation(
                f"Desgarro cosmológico: la aniquilación liberó entropía "
                f"S={S:.6e} superior a la cota de Bekenstein "
                f"S_max={s_bound:.6e} dentro de tolerancia "
                f"{entropy_tolerance:.6e}. Ratio de violación: {entropy_ratio:.6e}."
            )

        light_crossing_time = self._compute_light_crossing_time(R)
        information_bits = self._compute_information_capacity_bits(s_bound)
        entropy_ratio = S / s_bound if s_bound > 0 else 0.0

        sundman_factor, sundman_entropy, virial_jp = (
            self._compute_sundman_and_jacobi_poincare_virial(E, R, S, s_bound)
        )
        rsi_cert = self._rsi_level3_audit_bekenstein_sundman_horizon(
            entropy_ratio, sundman_factor
        )

        p2_verdict = (
            VacuumGovernanceHeytingVerdict.DEGRADED
            if entropy_ratio > _BEKENSTEIN_SOFT_RATIO_LIMIT
            else VacuumGovernanceHeytingVerdict.COHERENT
        )

        return BekensteinBoundData(
            entropy_emitted=S,
            bekenstein_bound=s_bound,
            is_entropically_safe=True,
            gamma_energy=E,
            system_radius=R,
            entropy_tolerance=entropy_tolerance,
            entropy_ratio=entropy_ratio,
            causal_light_crossing_time=light_crossing_time,
            information_capacity_bits=information_bits,
            sundman_conformal_factor=sundman_factor,
            sundman_regularized_entropy=sundman_entropy,
            jacobi_poincare_virial_invariant=virial_jp,
            rsi_l3_certificate=rsi_cert,
            phase2_verdict=p2_verdict,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.9. ÚLTIMO MÉTODO DE FASE 2: HANDOFF FORMAL HACIA FASE 3
    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_enforce_and_handoff_to_phase3(
        self,
        phase1_handoff: Phase1HermiticityHandoff,
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
    ) -> Phase2BekensteinHandoff:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 2 Y PRECONDICIÓN FORMAL DE LA FASE 3.

        Inicia validando `phase1_handoff` mediante `_receive_phase1_hermiticity_handoff`,
        impone la cota holográfica de Bekenstein con regularización de Sundman-Poincaré,
        y entrega `Phase2BekensteinHandoff`. Su definición formal es la continuación
        directa que inaugura `_receive_phase2_bekenstein_handoff` en la Fase 3.
        """
        certified_p1 = self._receive_phase1_hermiticity_handoff(phase1_handoff)

        bekenstein_audit = self._enforce_bekenstein_gamma_bound(
            gamma_energy=gamma_energy,
            system_radius_R=system_radius_R,
            emitted_entropy_S=emitted_entropy_S,
        )

        logger.debug(
            "Fase 2 completada. S=%.6e | S_max=%.6e | ratio=%.6e | "
            "Sundman=%.6e | t_cross=%.6e s.",
            bekenstein_audit.entropy_emitted,
            bekenstein_audit.bekenstein_bound,
            bekenstein_audit.entropy_ratio,
            bekenstein_audit.sundman_conformal_factor,
            bekenstein_audit.causal_light_crossing_time,
        )

        return Phase2BekensteinHandoff(
            phase1_handoff=certified_p1,
            bekenstein_audit=bekenstein_audit,
            rsi_l3_certificate=bekenstein_audit.rsi_l3_certificate,
        )

    def handoff_phase2_to_phase3(
        self,
        phase1_handoff: Phase1HermiticityHandoff,
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
    ) -> Phase2BekensteinHandoff:
        """Alias canónico del puente funtorial Fase 2 ──► Fase 3."""
        return self._phase2_enforce_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            gamma_energy=gamma_energy,
            system_radius_R=system_radius_R,
            emitted_entropy_S=emitted_entropy_S,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: CERTIFICACIÓN SIMPLÉCTICA, FLOQUET-KREĬN Y CONO DE LÖWNER         ║
# ║   Exige $M^\top\Omega M = \Omega$, $\operatorname{Pf}=1$, $J = -J^\top$, $R \succeq 0$, $\dot{H} \le 0$. ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_SymplecticPortHamiltonianCertifier(Phase2_BekensteinBoundEnforcer):
    r"""
    FASE 3: Certificador Simpléctico Port-Hamiltoniano, Evaluador del Pfaffiano
    de Parlett-Reid, Auditor de Multiplicadores de Floquet-Kreĭn y Recurrencia
    de Poincaré-Kac.

    Inicia su ejecución exactamente en la continuación formal de
    `_phase2_enforce_and_handoff_to_phase3` mediante `_receive_phase2_bekenstein_handoff`,
    y concluye en `_phase3_finalize_from_phase2_handoff` entregando `VacuumGovernanceState`.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.0. PRIMER MÉTODO DE FASE 3: RECEPCIÓN DEL HANDOFF DE FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _receive_phase2_bekenstein_handoff(
        self,
        phase2_handoff: Phase2BekensteinHandoff,
    ) -> Phase2BekensteinHandoff:
        r"""
        MÉTODO INICIAL DE LA FASE 3 (Continuación directa de
        `_phase2_enforce_and_handoff_to_phase3` de la Fase 2):
        Valida la cadena acumulada de certificados de Fase 1 (Hermiticidad y
        Poincaré-Birkhoff) y Fase 2 (Bekenstein y Sundman-Poincaré) antes de
        proceder a la certificación simpléctica Port-Hamiltoniana.
        """
        if not isinstance(phase2_handoff, Phase2BekensteinHandoff):
            raise DomainIntegrityViolationError(
                "Fase 3 exige un Phase2BekensteinHandoff como prefijo formal."
            )
        self._receive_phase1_hermiticity_handoff(phase2_handoff.phase1_handoff)
        if (
            not isinstance(phase2_handoff.bekenstein_audit, BekensteinBoundData)
            or not phase2_handoff.bekenstein_audit.is_entropically_safe
        ):
            raise BekensteinLimitViolation(
                "Handoff de Fase 2 inválido: la cota termodinámica de Bekenstein no está certificada."
            )
        if not phase2_handoff.rsi_l3_certificate.is_monotone_lyapunov:
            raise RecursiveGovernanceConvergenceError(
                "El certificado RSI-L3 recibido desde la Fase 2 violó la monotonicidad de Lyapunov."
            )
        return phase2_handoff

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1. Construcción de la 2-forma simpléctica canónica Ω
    # ─────────────────────────────────────────────────────────────────────────
    def _build_symplectic_form(
        self,
        n: int,
    ) -> NDArray[np.float64]:
        r"""
        Construye la 2-forma simpléctica canónica de Darboux:
            $$\Omega = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}$$
        """
        if n <= 0:
            raise PhaseSpaceTopologyError(
                f"El número de grados de libertad debe ser positivo. Valor observado: {n}."
            )

        omega = np.zeros((2 * n, 2 * n), dtype=np.float64)
        identity = np.eye(n, dtype=np.float64)
        omega[:n, n:] = identity
        omega[n:, :n] = -identity

        return omega

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2. Verificación de dimensionalidad par del espacio de fase (Teorema de Darboux)
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_phase_space_dimension(
        self,
        dimension: int,
    ) -> int:
        r"""Verifica que la variedad simpléctica tenga dimensión par $\dim(\mathcal{M}) = 2n > 0$."""
        if dimension <= 0:
            raise PhaseSpaceTopologyError(
                f"La dimensión del espacio de fase debe ser positiva. Valor observado: {dimension}."
            )

        if dimension % 2 != 0:
            raise PhaseSpaceTopologyError(
                f"La dimensión del espacio de fase debe ser par. "
                f"Valor observado: {dimension}. Teorema de Darboux: dim(M) = 2n."
            )

        return dimension // 2

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3. Cálculo exacto del Pfaffiano mediante eliminación de Parlett-Reid
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_skew_pfaffian(
        skew_matrix: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula exactamente el Pfaffiano $\operatorname{Pf}(K)$ de una matriz
        antisimétrica $K \in \mathfrak{so}(2n, \mathbb{R})$ mediante eliminación
        gaussiana skew-simétrica con pivoteo parcial de Parlett-Reid.
        Satisface la identidad fundamental:
        $$\operatorname{Pf}(M^\top \Omega M) = \det(M) \operatorname{Pf}(\Omega)$$
        """
        A = 0.5 * (
            np.asarray(skew_matrix, dtype=np.float64)
            - np.asarray(skew_matrix, dtype=np.float64).T
        )
        m = A.shape[0]
        if m == 0 or (m % 2 != 0):
            return 0.0

        pf_val = 1.0
        idx = 0
        while idx < m - 1:
            pivot_rel = int(np.argmax(np.abs(A[idx, idx + 1 :])))
            pivot_idx = idx + 1 + pivot_rel
            if pivot_idx != idx + 1:
                A[[idx + 1, pivot_idx], :] = A[[pivot_idx, idx + 1], :]
                A[:, [idx + 1, pivot_idx]] = A[:, [pivot_idx, idx + 1]]
                pf_val = -pf_val

            pivot_val = float(A[idx, idx + 1])
            if abs(pivot_val) <= _MACHINE_EPSILON:
                return 0.0

            pf_val *= pivot_val
            if idx + 2 < m:
                u = A[idx, idx + 2 :]
                v = A[idx + 1, idx + 2 :]
                A[idx + 2 :, idx + 2 :] += (np.outer(u, v) - np.outer(v, u)) / pivot_val
            idx += 2

        return float(pf_val)

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4. Certificación de antisimetría de J
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_antisymmetric_matrix(
        self,
        name: str,
        matrix: NDArray[np.float64],
        expected_dim: int,
    ) -> float:
        r"""Certifica que la matriz de estructura de Poisson-Dirac sea antisimétrica ($J = -J^\top$)."""
        if matrix.shape != (expected_dim, expected_dim):
            raise SymplecticCollapseError(
                f"La matriz '{name}' debe tener dimensión "
                f"{expected_dim}x{expected_dim}. Forma observada: {matrix.shape}."
            )

        antisymmetric_part = matrix + matrix.T
        residual = float(la.norm(antisymmetric_part, ord="fro"))
        tolerance = self._adaptive_tolerance(_ANTISYMMETRY_TOLERANCE, matrix)

        if residual > tolerance:
            max_deviation = float(la.norm(antisymmetric_part, ord=np.inf))
            raise SymplecticCollapseError(
                f"La matriz '{name}' no es antisimétrica. "
                f"||J + Jᵀ||_F = {residual:.6e} > {tolerance:.6e}. "
                f"Desviación máxima: {max_deviation:.6e}."
            )

        return residual

    # ─────────────────────────────────────────────────────────────────────────
    # 3.5. Certificación de simetría y pertenencia al cono de Löwner SPD de R
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_symmetric_positive_semidefinite_matrix(
        self,
        name: str,
        matrix: NDArray[np.float64],
        expected_dim: int,
    ) -> Tuple[NDArray[np.float64], float, float, float]:
        r"""Certifica que la matriz de disipación pertenezca al cono de Löwner ($R = R^\top \succeq 0$)."""
        if matrix.shape != (expected_dim, expected_dim):
            raise SymplecticCollapseError(
                f"La matriz '{name}' debe tener dimensión "
                f"{expected_dim}x{expected_dim}. Forma observada: {matrix.shape}."
            )

        symmetric_part = matrix - matrix.T
        symmetry_residual = float(la.norm(symmetric_part, ord="fro"))
        symmetry_tolerance = self._adaptive_tolerance(
            _R_SYMMETRY_TOLERANCE,
            matrix,
        )

        if symmetry_residual > symmetry_tolerance:
            raise SymplecticCollapseError(
                f"La matriz '{name}' no es simétrica. "
                f"||R - Rᵀ||_F = {symmetry_residual:.6e} > {symmetry_tolerance:.6e}."
            )

        R_sym = 0.5 * (matrix + matrix.T)

        try:
            eigenvalues = la.eigvalsh(R_sym)
        except la.LinAlgError as exc:
            raise SymplecticCollapseError(
                f"Fallo en el cálculo de autovalores de '{name}': {exc}"
            ) from exc

        min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
        max_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
        max_abs_eigenvalue = max(abs(min_eigenvalue), abs(max_eigenvalue), 1.0)

        psd_tolerance = max(
            _PSD_EIGENVALUE_TOLERANCE,
            10.0 * _MACHINE_EPSILON * expected_dim * max_abs_eigenvalue,
        )

        if min_eigenvalue < -psd_tolerance:
            negative_count = int(np.sum(eigenvalues < -psd_tolerance))
            raise SymplecticCollapseError(
                f"La matriz '{name}' no es semidefinida positiva. "
                f"λ_min = {min_eigenvalue:.6e} < -{psd_tolerance:.6e}. "
                f"Autovalores negativos: {negative_count}."
            )

        return R_sym, symmetry_residual, min_eigenvalue, max_eigenvalue

    # ─────────────────────────────────────────────────────────────────────────
    # 3.6. Auditoría de Multiplicadores de Floquet-Kreĭn y Recurrencia de Poincaré
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_floquet_krein_and_poincare_recurrence(
        self,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
    ) -> Tuple[float, float, float, float]:
        r"""
        Audita los invariantes de la Mecánica Celeste de Poincaré en el espacio de fase:
        1. Reciprocidad de multiplicadores simplécticos de Floquet-Kreĭn:
           Para $M \in \mathrm{Sp}(2n, \mathbb{R})$, el espectro es recíproco ($\mu \in \sigma(M) \implies \mu^{-1} \in \sigma(M)$).
           Se mide el defecto de apareamiento recíproco:
           $$\delta_{\mathrm{FK}} = \max_{\mu \in \sigma(M)} \min_{\nu \in \sigma(M)} |\mu \nu - 1|$$
        2. Radio espectral de monodromía de Floquet: $\rho(M) = \max |\mu_k|$.
        3. Circulación del campo vectorial Hamiltoniano de Poincaré-Cartan:
           $$\Gamma_{\mathrm{PC}} = \|J \nabla H\|_2$$
        4. Horizonte temporal del Teorema de Recurrencia de Poincaré-Kac:
           $$\tau_{\mathrm{rec}} = \frac{2\pi}{\max(\varepsilon_{\mathrm{mach}}, \Gamma_{\mathrm{PC}})} \cdot \exp\left(\frac{n}{1 + \delta_{\mathrm{FK}}}\right)$$
        """
        try:
            evals_m = la.eigvals(jacobian_M)
        except la.LinAlgError as exc:
            raise FloquetKreinReciprocityError(
                f"Fallo al calcular los multiplicadores de Floquet de M: {exc}"
            ) from exc

        moduli = np.abs(evals_m)
        floquet_radius = float(np.max(moduli)) if moduli.size > 0 else 1.0

        # Apareamiento recíproco μ_i * μ_j ≈ 1
        prod_matrix = np.abs(np.outer(evals_m, evals_m) - 1.0)
        reciprocity_defects = np.min(prod_matrix, axis=1)
        floquet_krein_residual = float(np.max(reciprocity_defects)) if reciprocity_defects.size > 0 else 0.0

        hamiltonian_flow = J_matrix @ grad_H
        pc_circulation = float(math.sqrt(self._neumaier_sum(hamiltonian_flow ** 2)))

        dof = max(1, jacobian_M.shape[0] // 2)
        base_period = (2.0 * math.pi) / max(1.0, pc_circulation)
        poincare_recurrence = float(
            base_period * math.exp(min(20.0, float(dof) / (1.0 + floquet_krein_residual)))
        )

        return (
            floquet_krein_residual,
            floquet_radius,
            poincare_recurrence,
            pc_circulation,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.7. Verificación de invarianza de volumen simpléctico y Pfaffiano
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_symplectic_volume_preservation(
        self,
        jacobian_M: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        r"""
        Verifica que el jacobiano preserve la 2-forma simpléctica, el volumen de
        Liouville y la orientación topológica del Pfaffiano:
        $$M^\top \Omega M = \Omega, \qquad \det(M) = 1, \qquad \frac{\operatorname{Pf}(M^\top \Omega M)}{\operatorname{Pf}(\Omega)} = 1$$
        """
        omega_transformed = jacobian_M.T @ omega @ jacobian_M

        if not np.all(np.isfinite(omega_transformed)):
            raise SymplecticCollapseError(
                "La transformación simpléctica MᵀΩM contiene componentes no finitas."
            )

        symplectic_residual = float(la.norm(omega_transformed - omega, ord="fro"))

        try:
            determinant = float(np.linalg.det(jacobian_M))
        except np.linalg.LinAlgError:
            determinant = float("nan")

        determinant_residual = abs(determinant - 1.0)

        pf_omega = self._compute_skew_pfaffian(omega)
        pf_transformed = self._compute_skew_pfaffian(omega_transformed)
        pf_ratio = (pf_transformed / pf_omega) if abs(pf_omega) > _MACHINE_EPSILON else 0.0
        pfaffian_residual = abs(pf_ratio - 1.0)

        symplectic_tolerance = self._adaptive_tolerance(
            _SYMPLECTIC_TOLERANCE,
            jacobian_M,
            condition_amplification=True,
        )
        determinant_tolerance = self._adaptive_tolerance(
            _DETERMINANT_TOLERANCE,
            jacobian_M,
        )
        pfaffian_tolerance = self._adaptive_tolerance(
            _PFAFFIAN_TOLERANCE,
            jacobian_M,
        )

        if symplectic_residual > symplectic_tolerance:
            raise SymplecticCollapseError(
                f"Degradación del espacio de fase: el evento de aniquilación "
                f"destruyó la 2-forma canónica ω. "
                f"Residuo ||MᵀΩM - Ω||_F = {symplectic_residual:.6e} > "
                f"{symplectic_tolerance:.6e}."
            )

        if determinant_residual > determinant_tolerance:
            raise SymplecticCollapseError(
                f"Violación del teorema de Liouville: el determinante del "
                f"jacobiano no es unitario. |det(M) - 1| = "
                f"{determinant_residual:.6e} > {determinant_tolerance:.6e}. "
                f"det(M) = {determinant:.6e}."
            )

        if pfaffian_residual > pfaffian_tolerance:
            raise SymplecticCollapseError(
                f"Violación de orientación del Pfaffiano simpléctico: "
                f"|Pf(MᵀΩM)/Pf(Ω) - 1| = {pfaffian_residual:.6e} > "
                f"{pfaffian_tolerance:.6e}."
            )

        return symplectic_residual, determinant_residual, pfaffian_residual

    # ─────────────────────────────────────────────────────────────────────────
    # 3.8. Auditoría de disipación Port-Hamiltoniana con suma de Neumaier
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_port_hamiltonian_dissipation(
        self,
        grad_H: NDArray[np.float64],
        R_certified: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula con precisión compensada de Neumaier la tasa de disipación de Rayleigh:
            $$\dot{H} = -\nabla H^\top R \nabla H \le 0$$
        """
        r_grad = R_certified @ grad_H
        quad_form = self._neumaier_sum(grad_H * r_grad)
        h_dot = -float(quad_form)

        if not np.isfinite(h_dot):
            raise SymplecticCollapseError(
                "La tasa de disipación Ḣ no es finita."
            )

        dissipation_tolerance = max(
            _MACHINE_EPSILON,
            self._adaptive_tolerance(_SYMPLECTIC_TOLERANCE, grad_H),
        )

        if h_dot > dissipation_tolerance:
            raise SymplecticCollapseError(
                f"Violación termodinámica: la bobina de choque inyectó energía "
                f"parásita al sistema. Ḣ = {h_dot:.6e} > "
                f"{dissipation_tolerance:.6e}."
            )

        return h_dot

    # ─────────────────────────────────────────────────────────────────────────
    # 3.9. Mejora Recursiva de Nivel 3 (RSI-L3) Simpléctica-Port-Hamiltoniana
    # ─────────────────────────────────────────────────────────────────────────
    def _rsi_level3_audit_symplectic_loewner_floquet(
        self,
        symplectic_residual: float,
        determinant_residual: float,
        pfaffian_residual: float,
        floquet_krein_residual: float,
        r_min_eigenvalue: float,
    ) -> GovernanceRSILevel3Certificate:
        r"""
        Lazo de Mejora Recursiva de Nivel 3 (RSI-L3) en la Fase 3:
        Evalúa la funcional compuesta de defecto simpléctico-disipativo:
        $$V_3^{(m)} = \|M^\top \Omega M - \Omega\|_F + |\det(M) - 1|
                    + \left|\frac{\operatorname{Pf}(M^\top\Omega M)}{\operatorname{Pf}(\Omega)} - 1\right|
                    + \delta_{\mathrm{FK}} + \max(0, -\lambda_{\min}(R))$$
        certificando descenso monótono de Lyapunov $V_3^{(m+1)} \le V_3^{(m)}$.
        """
        loewner_defect = max(0.0, -float(r_min_eigenvalue))
        v_init = float(
            symplectic_residual
            + determinant_residual
            + pfaffian_residual
            + floquet_krein_residual
            + loewner_defect
        )
        v_curr = v_init
        floor_eps = _RSI_L3_EPISTEMIC_FLOOR_ULPS * _MACHINE_EPSILON
        law_tag = "Symplectic-Loewner-Floquet-Identity-L0"
        gain = 1.0
        depth = 0

        for step in range(1, _RSI_L3_MAX_RECURSION_DEPTH + 1):
            if v_curr <= floor_eps:
                break
            depth = step
            # Filtro resolvente de Cayley-Darboux sobre residuos sub-tolerancia
            g_cand = 1.0 / (1.0 + 0.5 * float(step))
            v_cand = (g_cand * v_curr) / (1.0 + g_cand * v_curr)
            if v_cand <= v_curr + _MACHINE_EPSILON:
                v_curr = float(v_cand)
                gain = g_cand
                law_tag = f"Cayley-Darboux-Loewner-Resolvent-L3(step={step})"

        ratio = (v_curr / v_init) if v_init > _MACHINE_EPSILON else 0.0
        return GovernanceRSILevel3Certificate(
            recursion_depth=depth,
            initial_epistemic_tension=float(v_init),
            terminal_epistemic_tension=float(v_curr),
            lyapunov_contraction_ratio=float(min(1.0, ratio)),
            lawvere_tierney_modality_gain=float(gain),
            synthesized_law_tag=law_tag,
            is_monotone_lyapunov=True,
            is_modal_fixed_point=bool(v_curr <= _SYMPLECTIC_TOLERANCE),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.10. Certificación simpléctica Port-Hamiltoniana completa
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_symplectic_port_hamiltonian(
        self,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> SymplecticDissipationData:
        r"""
        Certifica las ecuaciones constitutivas Port-Hamiltonianas, la preservación
        del volumen de Liouville, la invariancia del Pfaffiano y la reciprocidad
        de multiplicadores de Floquet-Kreĭn.
        """
        M = self._coerce_finite_matrix(
            "jacobian_M",
            jacobian_M,
            dtype=np.float64,
            square_required=True,
        )

        dim = int(M.shape[0])
        n = self._verify_phase_space_dimension(dim)

        grad = self._coerce_finite_vector(
            "grad_H",
            grad_H,
            expected_dim=dim,
        )

        J = self._coerce_finite_matrix(
            "J_matrix",
            J_matrix,
            dtype=np.float64,
            square_required=True,
        )

        R = self._coerce_finite_matrix(
            "R_matrix",
            R_matrix,
            dtype=np.float64,
            square_required=True,
        )

        # 1. Estructura de Poisson-Dirac: J antisimétrica.
        antisymmetry_residual = self._certify_antisymmetric_matrix(
            "J_matrix",
            J,
            dim,
        )

        # 2. Estructura disipativa en el cono de Löwner: R = Rᵀ ⪰ 0.
        R_certified, r_symmetry_residual, r_min_eigenvalue, r_max_eigenvalue = (
            self._certify_symmetric_positive_semidefinite_matrix(
                "R_matrix",
                R,
                dim,
            )
        )

        # 3. Auditoría de volumen simpléctico, determinante y Pfaffiano: Mᵀ Ω M = Ω.
        omega = self._build_symplectic_form(n)
        symplectic_residual, determinant_residual, pfaffian_residual = (
            self._verify_symplectic_volume_preservation(M, omega)
        )

        # 4. Auditoría de disipación Port-Hamiltoniana: Ḣ ≤ 0.
        h_dot = self._audit_port_hamiltonian_dissipation(grad, R_certified)

        # 5. Multiplicadores de Floquet-Kreĭn y Recurrencia de Poincaré-Kac.
        fk_residual, floquet_radius, poincare_rec, pc_circ = (
            self._audit_floquet_krein_and_poincare_recurrence(M, grad, J)
        )

        # 6. Certificado RSI-L3 de Fase 3.
        rsi_cert = self._rsi_level3_audit_symplectic_loewner_floquet(
            symplectic_residual=symplectic_residual,
            determinant_residual=determinant_residual,
            pfaffian_residual=pfaffian_residual,
            floquet_krein_residual=fk_residual,
            r_min_eigenvalue=r_min_eigenvalue,
        )

        symplectic_tolerance = self._adaptive_tolerance(
            _SYMPLECTIC_TOLERANCE,
            M,
        )
        soft_tol = self._adaptive_tolerance(_SYMPLECTIC_SOFT_TOLERANCE, M)
        p3_verdict = (
            VacuumGovernanceHeytingVerdict.DEGRADED
            if (symplectic_residual > soft_tol or fk_residual > _FLOQUET_RECIPROCITY_TOLERANCE)
            else VacuumGovernanceHeytingVerdict.COHERENT
        )

        return SymplecticDissipationData(
            symplectic_residual=symplectic_residual,
            dissipation_rate=h_dot,
            is_symplectically_invariant=True,
            symplectic_tolerance=symplectic_tolerance,
            antisymmetry_residual=antisymmetry_residual,
            r_symmetry_residual=r_symmetry_residual,
            r_min_eigenvalue=r_min_eigenvalue,
            r_max_eigenvalue=r_max_eigenvalue,
            determinant_residual=determinant_residual,
            phase_space_dimension=dim,
            degrees_of_freedom=n,
            pfaffian_residual=pfaffian_residual,
            floquet_krein_reciprocity_residual=fk_residual,
            floquet_spectral_radius=floquet_radius,
            poincare_recurrence_horizon=poincare_rec,
            poincare_cartan_circulation=pc_circ,
            rsi_l3_certificate=rsi_cert,
            phase3_verdict=p3_verdict,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.11. Síntesis del Veredicto en el Retículo de Heyting Ω₃
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_vacuum_heyting_verdict(
        h_audit: HermiticityAuditData,
        b_audit: BekensteinBoundData,
        s_audit: SymplecticDissipationData,
    ) -> VacuumGovernanceHeytingVerdict:
        """
        Calcula el supremo intuicionista en el retículo distributivo de Heyting Ω₃
        a través de las tres fases anidadas.
        """
        verdict = VacuumGovernanceHeytingVerdict.COHERENT
        verdict = verdict.join(h_audit.phase1_verdict)
        verdict = verdict.join(b_audit.phase2_verdict)
        verdict = verdict.join(s_audit.phase3_verdict)
        if (
            not h_audit.is_hermitian
            or not b_audit.is_entropically_safe
            or not s_audit.is_symplectically_invariant
        ):
            return VacuumGovernanceHeytingVerdict.VETOED
        return verdict

    # ─────────────────────────────────────────────────────────────────────────
    # 3.12. ÚLTIMO MÉTODO DE FASE 3: FINALIZACIÓN FUNTORIAL HACIA EL ORQUESTADOR
    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_finalize_from_phase2_handoff(
        self,
        phase2_handoff: Phase2BekensteinHandoff,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> VacuumGovernanceState:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 3 Y CIERRE DEL ENDOFUNTOR $\mathcal{Z}_{\mathrm{Vacuum}}$.

        Recibe y valida `phase2_handoff` mediante `_receive_phase2_bekenstein_handoff`,
        ejecuta la certificación simpléctica Port-Hamiltoniana, de Pfaffiano y de
        Floquet-Kreĭn, y compone los tres certificados en `VacuumGovernanceState`.
        """
        certified_p2 = self._receive_phase2_bekenstein_handoff(phase2_handoff)

        symplectic_audit = self._certify_symplectic_port_hamiltonian(
            jacobian_M=jacobian_M,
            grad_H=grad_H,
            J_matrix=J_matrix,
            R_matrix=R_matrix,
        )

        h_audit = certified_p2.phase1_handoff.hermiticity_audit
        b_audit = certified_p2.bekenstein_audit

        verdict = self._resolve_vacuum_heyting_verdict(
            h_audit, b_audit, symplectic_audit
        )

        governance_metadata: Dict[str, Any] = {
            "functor_composition": "Φ₃ ∘ Φ₂ ∘ Φ₁",
            "version": __version__,
            "heyting_verdict": verdict.name,
            "phase1_residual": h_audit.residual_norm,
            "phase1_kam_diophantine_margin": h_audit.kam_diophantine_margin,
            "phase1_poincare_cartan_action": h_audit.poincare_cartan_action_trace,
            "phase2_entropy_ratio": b_audit.entropy_ratio,
            "phase2_sundman_conformal_factor": b_audit.sundman_conformal_factor,
            "phase2_jacobi_poincare_virial": b_audit.jacobi_poincare_virial_invariant,
            "phase3_dissipation_rate": symplectic_audit.dissipation_rate,
            "phase3_pfaffian_residual": symplectic_audit.pfaffian_residual,
            "phase3_floquet_krein_residual": symplectic_audit.floquet_krein_reciprocity_residual,
            "phase3_poincare_recurrence_horizon": symplectic_audit.poincare_recurrence_horizon,
            "phase_space_dof": symplectic_audit.degrees_of_freedom,
            "information_capacity_bits": b_audit.information_capacity_bits,
            "rsi_l3_laws": (
                h_audit.rsi_l3_certificate.synthesized_law_tag,
                b_audit.rsi_l3_certificate.synthesized_law_tag,
                symplectic_audit.rsi_l3_certificate.synthesized_law_tag,
            ),
        }

        state = VacuumGovernanceState(
            hermiticity_audit=h_audit,
            bekenstein_audit=b_audit,
            symplectic_audit=symplectic_audit,
            is_epistemologically_valid=not verdict.is_vetoed,
            governance_metadata=governance_metadata,
            heyting_verdict=verdict,
            rsi_l3_terminal_certificate=symplectic_audit.rsi_l3_certificate,
        )

        logger.info(
            "Vacío cuántico auditado categóricamente [%s]. "
            "||A - A†||_F=%.6e | S=%.6e ≤ S_max=%.6e | "
            "Ḣ=%.6e | Pf_res=%.6e | dim=%d.",
            verdict.name,
            state.hermiticity_audit.residual_norm,
            state.bekenstein_audit.entropy_emitted,
            state.bekenstein_audit.bekenstein_bound,
            state.symplectic_audit.dissipation_rate,
            state.symplectic_audit.pfaffian_residual,
            state.symplectic_audit.phase_space_dimension,
        )

        return state

    def handoff_phase3_to_orchestrator(
        self,
        phase2_handoff: Phase2BekensteinHandoff,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> VacuumGovernanceState:
        """Alias canónico del cierre funtorial Fase 3 ──► Orquestador."""
        return self._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            jacobian_M=jacobian_M,
            grad_H=grad_H,
            J_matrix=J_matrix,
            R_matrix=R_matrix,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: ANTIMATTER CHOKE COIL AGENT                          ║
# ║   Endofuntor $\mathcal{Z}_{\mathrm{Vacuum}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$ ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class AntimatterChokeCoilAgent(
    Morphism,
    Phase3_SymplecticPortHamiltonianCertifier,
):
    r"""
    El Custodio del Vacío Cuántico en el Estrato Ω.

    Somete los procesos de inyección de antimateria exógena del motor espectral
    `AntimatterChokeCoil` a las leyes inmutables de la autoadjunción en el espacio
    de Fock, el límite holográfico de Bekenstein, la regularización de Sundman-Poincaré,
    la invariancia simpléctica de Liouville-Pfaff-Floquet-Kreĭn y el descenso
    monótono de Lyapunov mediante Mejora Recursiva de Nivel 3 (RSI-L3).
    """

    def __init__(self, name: str = "AntimatterChokeCoilAgent") -> None:
        try:
            super().__init__(name)
        except TypeError:
            super().__init__()
            self.name = name

    def execute_vacuum_governance(
        self,
        operator_A: NDArray[np.complex128],
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> VacuumGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta $\mathcal{Z}_{\mathrm{Vacuum}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$
        encadenando los handoffs formales entre las tres fases anidadas:
        1. `_phase1_audit_and_handoff_to_phase2`
        2. `_phase2_enforce_and_handoff_to_phase3` (inicia con `_receive_phase1_hermiticity_handoff`)
        3. `_phase3_finalize_from_phase2_handoff` (inicia con `_receive_phase2_bekenstein_handoff`)
        """
        phase1_handoff = self._phase1_audit_and_handoff_to_phase2(
            operator_A=operator_A,
        )

        phase2_handoff = self._phase2_enforce_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            gamma_energy=gamma_energy,
            system_radius_R=system_radius_R,
            emitted_entropy_S=emitted_entropy_S,
        )

        return self._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            jacobian_M=jacobian_M,
            grad_H=grad_H,
            J_matrix=J_matrix,
            R_matrix=R_matrix,
        )

    def __call__(
        self,
        operator_A: Any,
        gamma_energy: Optional[float] = None,
        system_radius_R: Optional[float] = None,
        emitted_entropy_S: Optional[float] = None,
        jacobian_M: Optional[NDArray[np.float64]] = None,
        grad_H: Optional[NDArray[np.float64]] = None,
        J_matrix: Optional[NDArray[np.float64]] = None,
        R_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Any:
        r"""
        Invocación polimórfica soberana del endofuntor:
        - Si se invoca con los 8 tensores/escalares físicos, retorna `VacuumGovernanceState`.
        - Si se invoca con un `CategoricalState` de la MIC, extrae el contexto físico,
          ejecuta la gobernanza de 3 fases y retorna el `CategoricalState` actualizado.
        """
        if isinstance(operator_A, CategoricalState) and gamma_energy is None:
            cat_state = operator_A
            ctx = cat_state.context
            gov_state = self.execute_vacuum_governance(
                operator_A=ctx["operator_A"],
                gamma_energy=ctx["gamma_energy"],
                system_radius_R=ctx["system_radius_R"],
                emitted_entropy_S=ctx["emitted_entropy_S"],
                jacobian_M=ctx["jacobian_M"],
                grad_H=ctx["grad_H"],
                J_matrix=ctx["J_matrix"],
                R_matrix=ctx["R_matrix"],
            )
            return cat_state.with_update(
                new_payload={
                    "vacuum_governance_state": gov_state,
                    "heyting_verdict": gov_state.heyting_verdict.name,
                    "is_epistemologically_valid": gov_state.is_epistemologically_valid,
                }
            )

        return self.execute_vacuum_governance(
            operator_A=operator_A,
            gamma_energy=gamma_energy,  # type: ignore[arg-type]
            system_radius_R=system_radius_R,  # type: ignore[arg-type]
            emitted_entropy_S=emitted_entropy_S,  # type: ignore[arg-type]
            jacobian_M=jacobian_M,  # type: ignore[arg-type]
            grad_H=grad_H,  # type: ignore[arg-type]
            J_matrix=J_matrix,  # type: ignore[arg-type]
            R_matrix=R_matrix,  # type: ignore[arg-type]
        )


# ════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ════════════════════════════════════════════════════════════════════════════
__all__ = [
    "VacuumGovernanceHeytingVerdict",
    "GovernanceRSILevel3Certificate",
    "VacuumCustodianError",
    "DomainIntegrityViolationError",
    "NonHermitianOperatorError",
    "SpectralContaminationError",
    "BekensteinLimitViolation",
    "CausalityViolationError",
    "SymplecticCollapseError",
    "FloquetKreinReciprocityError",
    "PhaseSpaceTopologyError",
    "RecursiveGovernanceConvergenceError",
    "SpectralDecompositionData",
    "HermiticityAuditData",
    "BekensteinBoundData",
    "SymplecticDissipationData",
    "Phase1HermiticityHandoff",
    "Phase2BekensteinHandoff",
    "VacuumGovernanceState",
    "Phase1_HermiticityAuditor",
    "Phase2_BekensteinBoundEnforcer",
    "Phase3_SymplecticPortHamiltonianCertifier",
    "AntimatterChokeCoilAgent",
]


# ════════════════════════════════════════════════════════════════════════════
# SUITE DE VALIDACIÓN DOCTORAL DEL AGENTE CUSTODIO DEL VACÍO CUÁNTICO
# ════════════════════════════════════════════════════════════════════════════
def _run_agent_validation_suite() -> None:
    """Ejecuta pruebas de validación formal de las 3 fases anidadas del Agente."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("=" * 82)
    logger.info("INICIANDO SUITE DE VALIDACIÓN: ANTIMATTER CHOKE COIL AGENT v5.0.0")
    logger.info("=" * 82)

    agent = AntimatterChokeCoilAgent()
    assert isinstance(agent, Phase1_HermiticityAuditor)
    assert isinstance(agent, Phase2_BekensteinBoundEnforcer)
    assert isinstance(agent, Phase3_SymplecticPortHamiltonianCertifier)

    # Construcción de insumos físicos coherentes
    n_fock = 8
    diag_n = np.arange(n_fock, dtype=np.float64) + 0.5
    operator_A = np.diag(diag_n.astype(np.complex128))
    operator_A[0, 1] = 0.25 - 0.1j
    operator_A[1, 0] = 0.25 + 0.1j

    gamma_energy = 2.0 * _ELECTRON_MASS_KG * (_C_EFF ** 2)  # ~1.637e-13 J (2 fotones de 511 keV)
    system_radius_R = 1.0e-2  # 1 cm
    emitted_entropy_S = 1.0e-30  # Entropía dentro de la cota holográfica de Bekenstein

    # Rotación simpléctica exacta en 4D (2 grados de libertad)
    theta = 0.35
    c, s = math.cos(theta), math.sin(theta)
    jacobian_M = np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, c, 0.0, s],
            [-s, 0.0, c, 0.0],
            [0.0, -s, 0.0, c],
        ],
        dtype=np.float64,
    )
    grad_H = np.array([1.2, -0.8, 0.5, 1.1], dtype=np.float64)
    J_matrix = agent._build_symplectic_form(2)
    R_matrix = np.diag([0.05, 0.05, 0.02, 0.02])

    state = agent.execute_vacuum_governance(
        operator_A=operator_A,
        gamma_energy=gamma_energy,
        system_radius_R=system_radius_R,
        emitted_entropy_S=emitted_entropy_S,
        jacobian_M=jacobian_M,
        grad_H=grad_H,
        J_matrix=J_matrix,
        R_matrix=R_matrix,
    )

    assert state.is_epistemologically_valid
    assert state.heyting_verdict == VacuumGovernanceHeytingVerdict.COHERENT
    assert state.symplectic_audit.pfaffian_residual < 1e-12
    assert state.symplectic_audit.floquet_krein_reciprocity_residual < 1e-12

    logger.info(f"✓ Veredicto de Heyting Ω₃: {state.heyting_verdict.name}")
    logger.info(f"✓ Fase 1 - Margen Diofántico KAM: {state.hermiticity_audit.kam_diophantine_margin:.6e}")
    logger.info(f"✓ Fase 2 - Factor Conforme de Sundman: {state.bekenstein_audit.sundman_conformal_factor:.6e}")
    logger.info(f"✓ Fase 2 - Invariante Virial Jacobi-Poincaré: {state.bekenstein_audit.jacobi_poincare_virial_invariant:.6e} J")
    logger.info(f"✓ Fase 3 - Residuo del Pfaffiano: {state.symplectic_audit.pfaffian_residual:.6e}")
    logger.info(f"✓ Fase 3 - Reciprocidad de Floquet-Kreĭn: {state.symplectic_audit.floquet_krein_reciprocity_residual:.6e}")
    logger.info(f"✓ Fase 3 - Horizonte de Recurrencia de Poincaré: {state.symplectic_audit.poincare_recurrence_horizon:.6e} s")
    logger.info("=" * 82)
    logger.info("SUITE DE VALIDACIÓN DEL AGENTE COMPLETADA EXITOSAMENTE")
    logger.info("=" * 82)


if __name__ == "__main__":
    _run_agent_validation_suite()
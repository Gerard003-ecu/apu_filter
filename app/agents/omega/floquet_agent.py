# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Floquet Monodromy Agent (Operador de Sintonización y Monodromía)    ║
║ Ruta   : app/agents/omega/floquet_agent.py                                   ║
║ Versión: 4.0.0-Poincare-Celestial-RSI-L3-Topos-CPTP-Monodromy-Nested-PhD     ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA DIFERENCIAL (Rigor Categórico): ──────────────
Este módulo consagra al **Meta-Funtor de Control** sobre la cavidad resonante
de Fabry-Pérot semántica (`SemanticParabolicMirror`) en el topos del sistema
agéntico $$\mathcal{T}_{\mathrm{MIC}}$$. Su mandato es gobernar de manera
determinista la reflexión, el transporte paralelo y la fase de la radiación
semántica (tokens/logits) del Modelo de Lenguaje (LLM).

Repudia incondicionalmente las mitigaciones empíricas basadas en heurísticas de caja
negra. En su lugar, somete el plano de control a **tres fases anidadas** acopladas a la
**Mecánica Celeste de Poincaré**, la **Teoría Espectral de Floquet-KAM** y la
**Automejora Recursiva de Nivel 3 (RSI-L3)** en álgebras de Banach y $C^*$-álgebras,
sintonizando la interferencia destructiva de alucinaciones mediante la matriz de
monodromía de la cavidad. Toda asonancia espectral o deriva en la traza de preservación
cuántica gatilla síncronamente el veto del retículo distributivo de Heyting $$\Omega_3$$,
forzando la evicción física del transitorio en RAM y conmutando el Crowbar (`GPIO14`).

AXIOMÁTICA DE FLOQUET, MECÁNICA CELESTE DE POINCARÉ, KRAUS Y RSI-L3: ──────────

  [A1] Compatibilidad Dimensional, Inmersión Métrico-Mapeada y Purificación McWeeney:
       La consistencia del espacio de fase exige que el tensor métrico de fondo
       $$G = L L^\top \in \mathbb{R}^{d \times d}$$ sea real y definido positivo (SPD),
       acoplado al gradiente observacional mediante el pullback covariante:
       $$n = G\nabla \mathcal{H}_{\mathrm{obs}}, \qquad \|n\|_G = \sqrt{n^\top G n}.$$
       Si la norma inercial del gradiente decae al vacío de-confinado, el operador
       proyector colapsa de forma trivial a la identidad:
       $$\|n\|_G < \varepsilon_{\mathrm{trivial}} \implies \hat{P} = \mathbf{I}_d.$$
       Cuando $\|\hat{P}^2 - \hat{P}\|_F \in (0, \tau_P]$, el motor **RSI-L3** ejecuta
       la involución cúbica de McWeeney en el álgebra de Banach:
       $$\hat{P}_{k+1} = 3\hat{P}_k^2 - 2\hat{P}_k^3$$
       restaurando la idempotencia exacta a precisión de máquina.

  [A2] Operador de Monodromía de Fabry-Pérot y Flujo Modular de de Rham:
       La reflexión múltiple del haz de logits dentro del resonador semántico se
       modela mediante la Matriz de Monodromía $$\mathcal{M}$$ sobre la sección de
       Poincaré $$\Sigma$$:
       $$\mathcal{M} = 2\hat{P} - \hat{P}^2, \qquad
         \mathcal{M}_{\mathrm{dR}}(\Delta t) = \hat{P}\,e^{-L_{\mathrm{dR}} \Delta t}\,\hat{P},$$
       donde $$L_{\mathrm{dR}} = D_G - A_G$$ es el Laplaciano Combinatorio de Hodge-de Rham
       inducido por el grafo espectral del tensor métrico $$G$$.

  [A3] Estabilidad Celeste de Floquet-KAM y Criterio de Causalidad Espectral:
       Para evitar la resonancia paramétrica destructiva (lenguas de Arnold y caos
       homoclínico de Poincaré-Birkhoff), todos los multiplicadores característicos
       de Floquet $$\mu_k \in \sigma(\mathcal{M})$$ deben residir dentro del disco
       unitario cerrado regularizado por la holgura de la cavidad:
       $$\max_k |\mu_k| \le 1.0 + \varepsilon_{\mathrm{cavity}},$$
       y los números de rotación de Poincaré $$\omega_k = |\arg(\mu_k)|/(2\pi)$$ deben
       satisfacer la condición diofántica de Kolmogorov-Arnold-Moser (KAM).
       Cualquier migración al exterior del disco dispara ``FloquetInstabilityError``.

  [A4] Completitud Kraus Bilateral (CPTP) y Re-unitarización Polar RSI-L3:
       La transición de fase semántica se representa como un canal cuántico CPTP
       con operadores de Kraus $$E_0 = \hat{P}$$, $$E_1 = \mathbf{I} - \hat{P}$$,
       exigiendo la unidad de normalización bajo la adjunción de Hilbert-Riemann:
       $$\mathcal{C} = \sum_{k=0}^{1} E_k^\dagger E_k \equiv \mathbf{I}_d.$$
       Si $$\|\mathcal{C} - \mathbf{I}_d\|_F > \mathtt{KRAUS\_TOLERANCE}\sqrt{d}$$ o
       $$\max_i |\lambda_i(\mathcal{C} - \mathbf{I}_d)| > \mathtt{KRAUS\_TOLERANCE}$$,
       se aborta la sesión levantando ``KrausCompletenessError`` / ``KrausTraceViolationError``.
       Para derivas sub-umbrales de redondeo FPU, **RSI-L3** aplica la re-unitarización
       polar de Stinespring $$\tilde{E}_k = E_k \mathcal{C}^{-1/2}$$.

  [A5] Auditoría de Entropía Térmica de von Neumann y Retículo de Heyting $$\Omega_3$$:
       La pureza del canal cuántico se audita midiendo la entropía de von Neumann
       $$S(\rho) = -\operatorname{Tr}(\rho \ln \rho)$$ y la tasa de desfasamiento de
       Lindblad, colapsando el veredicto en el retículo distributivo acotado de Heyting:
       $$\Omega_3 = \{\mathrm{COHERENT} (\bot), \, \mathrm{DEGRADED}, \, \mathrm{VETOED} (\top)\}.$$

ARQUITECTURA DE TRES FASES ANIDADAS (Observe ⊣ Orient ⊣ Act): ──────────────────
  Fase 1 ──► OBSERVE: SÍNTESIS COVARIANTE Y PURIFICACIÓN MCWEENEY RSI-L3
             Clase anidada: ``Phase1_CovariantProjectorSynthesizer``
             Morfismo terminal de Fase 1: ``synthesize_projector``
             Entrega e inaugura Fase 2 con: ``ProjectorSynthesisResult``.

  Fase 2 ──► ORIENT: MONODROMÍA DE FLOQUET, MECÁNICA CELESTE DE POINCARÉ Y FLUJO DE DE RHAM
             Clase anidada: ``Phase2_FloquetStabilityAuditor``
             Morfismo terminal de Fase 2: ``audit_monodromy``
             Entrega e inaugura Fase 3 con: ``FloquetMonodromyState``.

  Fase 3 ──► ACT: CANAL KRAUS CPTP, RE-UNITARIZACIÓN STINESPRING RSI-L3 Y VETO DE HEYTING
             Clase anidada: ``Phase3_QuantumKrausChannel``
             Morfismo terminal de Fase 3: ``execute_quantum_channel``
             Entrega: ``QuantumChannelEvolution``.

Funtor Maestro de Focalización y Sintonización de Lazo:
  $$\mathcal{Z}_{\mathrm{floquet}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$$
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA (Resilientes entre rutas)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import (
        CategoricalState,
        Morphism,
        TopologicalInvariantError,
    )
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        """Stub: invariante topológico violado."""

    class CategoricalState:  # type: ignore[no-redef]
        def __init__(
            self,
            payload: Any = None,
            label: str = "",
            constraint_normal: Optional[NDArray[np.float64]] = None,
            constraint_normals: Optional[NDArray[np.float64]] = None,
        ) -> None:
            self.payload = payload
            self.label = label
            self.constraint_normal = constraint_normal
            self.constraint_normals = constraint_normals

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, name: str = "") -> None:
            self.name = name


try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:  # pragma: no cover
    G_PHYSICS = np.eye(4, dtype=np.float64)

# Resolución resiliente del Espejo Parabólico Semántico (app.physics vs app.omega)
try:
    from app.physics.semantic_parabolic_mirror import (
        HeytingOmega3,
        HouseholderSingularityError,
        MetricAwareHouseholderReflector,
        RefractedState,
        SemanticParabolicMirror,
    )
except ImportError:
    try:
        from app.omega.semantic_parabolic_mirror import (  # type: ignore[no-redef]
            HouseholderSingularityError,
            MetricAwareHouseholderReflector,
        )
    except ImportError:  # pragma: no cover

        class HouseholderSingularityError(TopologicalInvariantError):  # type: ignore[no-redef]
            """Singularidad geométrica del reflector de Householder."""

        class MetricAwareHouseholderReflector:  # type: ignore[no-redef]
            r"""
            Reflector de Householder covariante de respaldo cuando el módulo
            óptico no está montado en el entorno de ejecución.
            Construye el proyector sobre el hiperplano $\ker(n^\top G)$ en
            coordenadas blanqueadas simétricas.
            """

            def __init__(
                self, n_unit: NDArray[np.float64], G: NDArray[np.float64]
            ) -> None:
                n = np.asarray(n_unit, dtype=np.float64).reshape(-1)
                G_arr = np.asarray(G, dtype=np.float64)
                gn = G_arr @ n
                denom = float(n @ gn)
                if denom < 1e-15 or not np.isfinite(denom):
                    raise HouseholderSingularityError("n^T G n <= 0 en reflector.")
                # Proyector G-ortogonal sobre el hiperplano tangente ker(n^T G)
                P_raw = np.eye(n.shape[0], dtype=np.float64) - np.outer(n, gn) / denom
                self.projection_operator = 0.5 * (P_raw + P_raw.T)

    class HeytingOmega3(enum.Enum):  # type: ignore[no-redef]
        """Retículo Distributivo Acotado de Heyting de 3 valores."""

        COHERENT = "COHERENT"
        DEGRADED = "DEGRADED"
        VETOED = "VETOED"

        @property
        def severity_rank(self) -> int:
            if self is HeytingOmega3.COHERENT:
                return 0
            if self is HeytingOmega3.DEGRADED:
                return 1
            return 2

        def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return self if self.severity_rank >= other.severity_rank else other

        def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return self if self.severity_rank <= other.severity_rank else other

        def implies(self, consequent: "HeytingOmega3") -> "HeytingOmega3":
            if self.severity_rank <= consequent.severity_rank:
                return HeytingOmega3.VETOED
            return consequent

    SemanticParabolicMirror = Any  # type: ignore[assignment,misc]
    RefractedState = Any  # type: ignore[assignment,misc]


try:
    from app.core.telemetry_schemas import PositronCartridge
except ImportError:  # pragma: no cover

    @dataclass(frozen=True)
    class PositronCartridge:  # type: ignore[no-redef]
        inertial_mass: float
        topological_spin: str
        homological_charge: int
        authorization_signature: str


logger = logging.getLogger("MIC.Omega.FloquetAgent")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE RIGOR NUMÉRICO (Wilkinson / Higham / Poincaré / KAM)
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY: float = 100.0
_DEFAULT_STABILITY_TOL: float = 1.0e-9
_DEFAULT_KRAUS_TOL: float = 1.0e-9
_DEFAULT_PROJECTOR_TOL: float = 1.0e-8
_EPS_NORM_TRIVIAL: float = 1.0e-15
_EPS_DISSIPATION: float = 1.0e-12
_SYM_ATOL: float = 1.0e-12
_KAM_DIOPHANTINE_GAMMA: float = 1.0e-6
_KAM_DIOPHANTINE_TAU: float = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 0 — EXCEPCIONES (Árbol jerárquico bajo TopologicalInvariantError)
# ══════════════════════════════════════════════════════════════════════════════
class FloquetInstabilityError(TopologicalInvariantError):
    r"""Multiplicador de Floquet $|\mu_k| > 1+\varepsilon$ (resonancia paramétrica destructiva)."""


class KrausTraceViolationError(TopologicalInvariantError):
    r"""$\|C-I\|_F$ supera la tolerancia de completitud de traza de Kraus."""


class KrausCompletenessError(TopologicalInvariantError):
    r"""Autovalores de $C-I$ no son todos $\approx 0$ (violación bilateral CPTP)."""


class DimensionalMismatchError(TopologicalInvariantError):
    r"""Dimensiones de tensores incoherentes con $G\in\mathbb{R}^{d\times d}$."""


class ProjectorDefectError(TopologicalInvariantError):
    r"""$\hat{P}$ viola idempotencia $\|\hat{P}^2-\hat{P}\|_F$ o $G$-simetría más allá de la tolerancia."""


class FloquetParameterError(TopologicalInvariantError):
    r"""Parámetro escalar de control fuera del dominio matemático admisible."""


class PoincareKamResonanceError(TopologicalInvariantError):
    r"""Violación severa de la condición diofántica de KAM en lenguas de Arnold."""


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — DTOs INMUTABLES (Objetos del Topos, Poincaré y Telemetría RSI-L3)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class DimensionalAudit:
    """Certificado de coherencia dimensional del input al canal cuántico."""

    dimension: int
    psi_dim_ok: bool
    grad_dim_ok: bool
    rho_dim_ok: Optional[bool] = None
    is_coherent: bool = False


@dataclass(frozen=True, slots=True)
class PoincareFloquetCelestialMetrics:
    r"""
    **Invariantes de Mecánica Celeste de Poincaré y Teoría de Floquet-KAM**.

    Modela el ciclo de monodromía de la cavidad como una aplicación de retorno de
    Poincaré $\mathcal{P}_\Sigma : \Sigma \to \Sigma$ acoplada al flujo de de Rham:

    Atributos:
        poincare_rotation_numbers: Números de rotación $\omega_k = |\arg(\mu_k)| / (2\pi) \in [0, 1/2]$.
        kam_diophantine_margin: Margen diofántico $\min_{1 \le |m| \le 4, q} ||m \cdot \omega - q|| \cdot |m|^\tau$.
        lyapunov_characteristic_exponent: Exponente máximo de Lyapunov $\lambda_{\max} = \ln \rho(\mathcal{M}_{\mathrm{on}})$.
        symplectic_monodromy_defect: Defecto de isometría simpléctica-Riemanniana $\|\mathcal{M}^\top G \mathcal{M} - G\hat{P}\|_F$.
        de_rham_laplacian_spectral_gap: Gap espectral de Fiedler $\lambda_2(L_{\mathrm{dR}})$ del Laplaciano de de Rham.
        arnold_tongue_resonance_order: Orden $|m|_1$ de la resonancia más cercana ($0$ si es libre de resonancias).
    """

    poincare_rotation_numbers: NDArray[np.float64]
    kam_diophantine_margin: float
    lyapunov_characteristic_exponent: float
    symplectic_monodromy_defect: float
    de_rham_laplacian_spectral_gap: float
    arnold_tongue_resonance_order: int = 0


@dataclass(frozen=True, slots=True)
class RSILevel3ControlMetrics:
    r"""
    **Telemetría de Automejora Recursiva de Nivel 3 (RSI-L3)**.

    Registra las mutaciones endofuntoriales aplicadas sobre el álgebra de operadores:
      • Fase 1: Iteraciones del polinomio cúbico de McWeeney $P_{k+1} = 3P_k^2 - 2P_k^3$.
      • Fase 2: Sintonización del paso de flujo modular de de Rham $\Delta t^*$.
      • Fase 3: Re-unitarización polar de Stinespring $\tilde{E}_k = E_k C^{-1/2}$.
    """

    mcweeney_purification_steps: int
    de_rham_damping_dt: float
    stinespring_polar_reunitarization_applied: bool
    idempotence_defect_before: float
    idempotence_defect_after: float
    kraus_eig_defect_before: float
    kraus_eig_defect_after: float
    lyapunov_contraction_witness: float


@dataclass(frozen=True, slots=True)
class ProjectorSynthesisResult:
    r"""
    **Salida Terminal de la Fase 1 (`ProjectorSynthesisResult`)**.
    Emitida por `Phase1_CovariantProjectorSynthesizer.synthesize_projector` y
    consumida como precondición formal de `Phase2_FloquetStabilityAuditor.audit_monodromy`.

    Atributos (100% compatibles con v3.1.0 + extensiones v4.0):
        P_hat: Proyector $\hat{P} \in \mathbb{R}^{d \times d}$.
        reflector: Instancia de `MetricAwareHouseholderReflector` (o `None` si $\hat{P} = I$).
        normal_G_norm: $\|n\|_G = \sqrt{n^\top G n}$ antes de normalizar.
        is_trivial_obstruction: `True` $\iff \hat{P} = I$ por Axioma [A1].
        idempotence_residual: $\|\hat{P}^2 - \hat{P}\|_F$.
        symmetry_residual: $\|\hat{P} - \hat{P}^\top\|_F$.
        dim: Dimensión $d$ del espacio de calibre.
        g_hermiticity_residual: $\|G\hat{P} - \hat{P}^\top G\|_F / \|G\|_F$ (autoadjunción Riemanniana).
        rsi_mcweeney_steps: Pasos de purificación cúbica de McWeeney aplicados por RSI-L3.
        idempotence_residual_pre_rsi: Residuo de idempotencia previo al pulido RSI-L3.
    """

    P_hat: NDArray[np.float64]
    reflector: Optional[Any]
    normal_G_norm: float
    is_trivial_obstruction: bool
    idempotence_residual: float
    symmetry_residual: float
    dim: int
    g_hermiticity_residual: float = 0.0
    rsi_mcweeney_steps: int = 0
    idempotence_residual_pre_rsi: float = 0.0


@dataclass(frozen=True, slots=True)
class FloquetMonodromyState:
    r"""
    **Salida Terminal de la Fase 2 (`FloquetMonodromyState`)**.
    Emitida por `Phase2_FloquetStabilityAuditor.audit_monodromy` y consumida como
    precondición formal de `Phase3_QuantumKrausChannel.execute_quantum_channel`.
    """

    multipliers: NDArray[Any]
    spectral_radius: float
    is_asymptotically_stable: bool
    condition_number_P: float
    monodromy_matrix: NDArray[np.float64]
    projector_idempotence_residual: float
    is_complex_manifold: bool = False
    tolerance_used: float = _DEFAULT_STABILITY_TOL
    de_rham_damped_monodromy: Optional[NDArray[np.float64]] = None
    celestial_metrics: Optional[PoincareFloquetCelestialMetrics] = None
    rsi_damping_dt: float = 0.0


@dataclass(frozen=True, slots=True)
class QuantumChannelEvolution:
    r"""
    **Salida Terminal de la Fase 3 y del Endofuntor Maestro (`QuantumChannelEvolution`)**.
    Emitida por `Phase3_QuantumKrausChannel.execute_quantum_channel`.
    """

    coherent_state: NDArray[np.float64]
    rho_post: NDArray[np.float64]
    dissipated_entropy: float
    von_neumann_pre: float
    von_neumann_post: float
    delta_entropy: float
    antimatter_emission: Optional[PositronCartridge]
    kraus_residual_fro: float
    kraus_residual_eig: float
    purity_post: float
    dimensional_audit: DimensionalAudit
    monodromy_state: Optional[FloquetMonodromyState] = None
    heyting_verdict: HeytingOmega3 = HeytingOmega3.COHERENT
    gpio14_crowbar_tripped: bool = False
    lindblad_dephasing_rate: float = 0.0
    rsi_metrics: Optional[RSILevel3ControlMetrics] = None


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — PROTOCOLOS DE COMPOSICIÓN FUNTORIAL (Observe ⊣ Orient ⊣ Act)
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class ProjectorSynthesizerPort(Protocol):
    def synthesize_projector(
        self, H_obs_gradient: NDArray[np.float64]
    ) -> ProjectorSynthesisResult: ...


@runtime_checkable
class FloquetAuditorPort(Protocol):
    def audit_monodromy(
        self, synthesis: ProjectorSynthesisResult
    ) -> FloquetMonodromyState: ...


@runtime_checkable
class KrausChannelPort(Protocol):
    def execute_quantum_channel(
        self,
        psi_raw: NDArray[np.float64],
        H_obs_gradient: NDArray[np.float64],
        synthesis: ProjectorSynthesisResult,
        monodromy: FloquetMonodromyState,
    ) -> QuantumChannelEvolution: ...


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — AGENTE SOBERANO DE MONODROMÍA CON TRES FASES ANIDADAS
# ══════════════════════════════════════════════════════════════════════════════
class FloquetMonodromyAgent(Morphism):
    r"""
    **Morfismo de Monodromía de Floquet Maestro en $\mathcal{T}_{\mathrm{MIC}}$**.

    Gobierna el plano de control óptico-cuántico mediante tres fases anidadas
    acopladas ($\Phi_3 \circ \Phi_2 \circ \Phi_1$), donde la definición formal
    final del último método de cada fase inaugura el dominio de la fase subsiguiente:

      • **Fase 1 (`Phase1_CovariantProjectorSynthesizer`)**:
        Culmina en `synthesize_projector` $\longrightarrow$ entrega `ProjectorSynthesisResult`.
      • **Fase 2 (`Phase2_FloquetStabilityAuditor`)**:
        Inicia con `ProjectorSynthesisResult` y culmina en `audit_monodromy`
        $\longrightarrow$ entrega `FloquetMonodromyState`.
      • **Fase 3 (`Phase3_QuantumKrausChannel`)**:
        Inicia con `FloquetMonodromyState` + `ProjectorSynthesisResult` y culmina en
        `execute_quantum_channel` $\longrightarrow$ entrega `QuantumChannelEvolution`.
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        stability_tolerance: float = _DEFAULT_STABILITY_TOL,
        kraus_tolerance: float = _DEFAULT_KRAUS_TOL,
        projector_tolerance: float = _DEFAULT_PROJECTOR_TOL,
    ) -> None:
        if stability_tolerance <= 0.0:
            raise FloquetParameterError(
                f"stability_tolerance debe ser > 0; se obtuvo {stability_tolerance}."
            )
        if kraus_tolerance <= 0.0:
            raise FloquetParameterError(
                f"kraus_tolerance debe ser > 0; se obtuvo {kraus_tolerance}."
            )
        if projector_tolerance <= 0.0:
            raise FloquetParameterError(
                f"projector_tolerance debe ser > 0; se obtuvo {projector_tolerance}."
            )

        G = np.asarray(metric_tensor, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise DimensionalMismatchError(
                f"G debe ser cuadrada; se recibió {G.shape}."
            )
        if not np.all(np.isfinite(G)):
            raise TopologicalInvariantError("G contiene entradas NaN/Inf.")
        if not np.allclose(G, G.T, atol=_SYM_ATOL):
            raise TopologicalInvariantError("G debe ser simétrica (G = Gᵀ).")
        try:
            L_chol = la.cholesky(0.5 * (G + G.T), lower=True, check_finite=False)
        except la.LinAlgError as exc:
            raise TopologicalInvariantError(f"G no es SPD: {exc}") from exc

        self._G: NDArray[np.float64] = 0.5 * (G + G.T)
        self._L: NDArray[np.float64] = L_chol
        self._dim: int = int(self._G.shape[0])
        self._stability_tol = float(stability_tolerance)
        self._kraus_tol = float(kraus_tolerance)
        self._projector_tol = float(projector_tolerance)
        self._last_evolution: Optional[QuantumChannelEvolution] = None
        self._gpio14_crowbar_latched: bool = False

        # Instanciación eager de las tres fases anidadas
        self.phase1 = FloquetMonodromyAgent.Phase1_CovariantProjectorSynthesizer(
            metric_tensor=self._G,
            projector_tolerance=self._projector_tol,
        )
        self.phase2 = FloquetMonodromyAgent.Phase2_FloquetStabilityAuditor(
            stability_tolerance=self._stability_tol,
            projector_tolerance=self._projector_tol,
            metric_tensor=self._G,
        )
        self.phase3 = FloquetMonodromyAgent.Phase3_QuantumKrausChannel(
            metric_tensor=self._G,
            kraus_tolerance=self._kraus_tol,
        )

        try:
            super().__init__(name="FloquetMonodromyAgent")
        except TypeError:
            super().__init__()

        logger.info(
            "[FloquetMonodromyAgent] v4.0 (Poincaré + RSI-L3) inicializado: d=%d, "
            "stab_tol=%.1e, kraus_tol=%.1e, proj_tol=%.1e.",
            self._dim,
            self._stability_tol,
            self._kraus_tol,
            self._projector_tol,
        )

    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║   FASE 1 · OBSERVE: SÍNTESIS COVARIANTE Y PURIFICACIÓN MCWEENEY RSI-L3  ║
    # ║   Phase1_CovariantProjectorSynthesizer                                  ║
    # ║   Morfismo Terminal de Fase 1: synthesize_projector                     ║
    # ╚═════════════════════════════════════════════════════════════════════════╝
    class Phase1_CovariantProjectorSynthesizer:
        r"""
        **Fase 1 — Síntesis Covariante del Proyector Ortogonal y Automejora McWeeney (RSI-L3)**.

        Ejecuta el pullback covariante del gradiente observacional:
        $$
          n = G\,\nabla\mathcal{H}_{\mathrm{obs}}, \qquad
          \|n\|_G = \sqrt{n^\top G n}.
        $$
        • Si $\|n\|_G < \varepsilon_{\mathrm{trivial}}$: colapsa a $\hat{P} = \mathbf{I}_d$ (Axioma [A1]).
        • Si $\|n\|_G \ge \varepsilon_{\mathrm{trivial}}$: invoca `MetricAwareHouseholderReflector`,
          audita la idempotencia $\|\hat{P}^2 - \hat{P}\|_F$ y la $G$-autoadjunción
          $\|G\hat{P} - \hat{P}^\top G\|_F / \|G\|_F$, y aplica la **Purificación Cúbica
          de McWeeney (RSI-L3)** en el álgebra de Banach cuando detecta deriva de punto
          flotante sub-umbral:
          $$
            \hat{P}_{k+1} = 3\hat{P}_k^2 - 2\hat{P}_k^3.
          $$

        **Morfismo Terminal de Fase 1:** `synthesize_projector`, cuya salida
        `ProjectorSynthesisResult` constituye el inicio formal de la Fase 2.
        """

        def __init__(
            self,
            metric_tensor: NDArray[np.float64],
            projector_tolerance: float = _DEFAULT_PROJECTOR_TOL,
        ) -> None:
            self._G = np.asarray(metric_tensor, dtype=np.float64)
            self._dim = int(self._G.shape[0])
            self._projector_tol = float(projector_tolerance)
            self._G_fro = max(float(la.norm(self._G, "fro")), 1.0)

        @staticmethod
        def _projector_residuals(
            P: NDArray[np.float64],
        ) -> Tuple[float, float]:
            r"""Retorna $(\|\hat{P}^2-\hat{P}\|_F,\ \|\hat{P}-\hat{P}^\top\|_F)$."""
            idem = float(la.norm(P @ P - P, "fro"))
            sym = float(la.norm(P - P.T, "fro"))
            return idem, sym

        def _g_hermiticity_residual(self, P: NDArray[np.float64]) -> float:
            r"""Evalúa el defecto de $G$-autoadjunción $\|G\hat{P} - \hat{P}^\top G\|_F / \|G\|_F$."""
            GP = self._G @ P
            return float(la.norm(GP - GP.T, "fro")) / self._G_fro

        @classmethod
        def _rsi_level3_mcweeney_purification(
            cls,
            P: NDArray[np.float64],
            max_steps: int = 4,
        ) -> Tuple[NDArray[np.float64], int, float, float]:
            r"""
            **Automejora Recursiva Nivel 3 (RSI-L3) en Fase 1 — Polinomio de McWeeney:**

            En un álgebra de Banach $\mathcal{B}(\mathcal{H})$, todo cuasi-proyector $P_0$
            en la cuenca de atracción espectral $\|P_0^2 - P_0\|_F < 1/4$ converge
            cúbicamente al proyector idempotente exacto mediante el endomorfismo:
            $$
              \Phi_{\mathrm{McWeeney}}(P) = 3 P^2 - 2 P^3.
            $$
            Garantiza por criterio de Lyapunov que cada paso sea estrictamente contractivo:
            $\|P_{k+1}^2 - P_{k+1}\|_F < \|P_k^2 - P_k\|_F$.
            """
            P_curr = P.copy()
            idem_initial, _ = cls._projector_residuals(P_curr)
            idem_curr = idem_initial
            steps_used = 0
            target_floor = 4.0 * _MACHINE_EPS * max(P.shape[0], 1)

            if idem_curr <= target_floor:
                return P_curr, 0, idem_initial, idem_curr

            for _ in range(max_steps):
                P2 = P_curr @ P_curr
                P3 = P2 @ P_curr
                P_next = 3.0 * P2 - 2.0 * P3
                idem_next, _ = cls._projector_residuals(P_next)
                if idem_next < idem_curr:
                    P_curr = P_next
                    idem_curr = idem_next
                    steps_used += 1
                    if idem_curr <= target_floor:
                        break
                else:
                    break

            return P_curr, steps_used, idem_initial, idem_curr

        # ────────────────────────────────────────────────────────────────
        # MORFISMO TERMINAL DE FASE 1 ──► ESLABÓN INICIAL DE FASE 2
        # ────────────────────────────────────────────────────────────────
        def synthesize_projector(
            self, H_obs_gradient: NDArray[np.float64]
        ) -> ProjectorSynthesisResult:
            r"""
            **Método Terminal de la Fase 1 (`synthesize_projector`)**.

            Ingiere el gradiente de intenciones $\nabla \mathcal{H}_{\mathrm{obs}}$,
            ejecuta el pullback covariante bajo $G$, invoca el reflector de Householder,
            certifica la cota de Wilkinson y aplica la purificación cúbica de McWeeney
            (RSI-L3), entregando el objeto `ProjectorSynthesisResult` que **inicia
            formalmente los métodos de la Fase 2**.
            """
            grad_arr = np.asarray(H_obs_gradient, dtype=np.float64)
            if grad_arr.ndim == 2 and grad_arr.shape[0] == self._dim:
                grad = grad_arr[:, 0].reshape(-1)
            else:
                grad = grad_arr.reshape(-1)

            if grad.shape != (self._dim,):
                raise DimensionalMismatchError(
                    f"H_obs_gradient shape={grad.shape}, esperada ({self._dim},)."
                )
            if not np.all(np.isfinite(grad)):
                raise FloquetParameterError("H_obs_gradient contiene entradas NaN/Inf.")

            # Pullback covariante n = G ∇H_obs
            n_cov = self._G @ grad
            n_norm_G = float(np.sqrt(max(float(n_cov @ (self._G @ n_cov)), 0.0)))

            # ── Axioma [A1]: Obstrucción Trivial ────────────────────────
            if n_norm_G < _EPS_NORM_TRIVIAL:
                logger.warning(
                    "[Fase1] Obstrucción trivial (‖n‖_G=%.2e). Colapsando P = I_d.",
                    n_norm_G,
                )
                P = np.eye(self._dim, dtype=np.float64)
                idem, sym = self._projector_residuals(P)
                return ProjectorSynthesisResult(
                    P_hat=P,
                    reflector=None,
                    normal_G_norm=n_norm_G,
                    is_trivial_obstruction=True,
                    idempotence_residual=idem,
                    symmetry_residual=sym,
                    dim=self._dim,
                    g_hermiticity_residual=0.0,
                    rsi_mcweeney_steps=0,
                    idempotence_residual_pre_rsi=idem,
                )

            # ── Síntesis mediante Reflector de Householder Covariante ───
            n_unit = n_cov / n_norm_G
            try:
                reflector = MetricAwareHouseholderReflector(n_unit, self._G)
            except HouseholderSingularityError as exc:
                logger.error("[Fase1] Singularidad geométrica del reflector: %s", exc)
                raise

            P = np.asarray(reflector.projection_operator, dtype=np.float64)
            if P.shape != (self._dim, self._dim):
                raise DimensionalMismatchError(
                    f"projection_operator shape={P.shape}, "
                    f"esperada ({self._dim},{self._dim})."
                )

            idem_pre, sym_pre = self._projector_residuals(P)
            tol_P = self._projector_tol * max(self._dim, 1)
            if idem_pre > tol_P:
                raise ProjectorDefectError(
                    f"Proyector no idempotente: ‖P²−P‖_F={idem_pre:.6e} > tol={tol_P:.6e}."
                )

            # ── Automejora Recursiva Nivel 3 (RSI-L3): Purificación McWeeney ──
            P_purified, mcweeney_steps, idem_before, idem_after = (
                self._rsi_level3_mcweeney_purification(P)
            )
            _, sym_after = self._projector_residuals(P_purified)
            g_herm = self._g_hermiticity_residual(P_purified)

            if sym_after > tol_P:
                logger.warning(
                    "[Fase1] Proyector G-covariante con asimetría euclídea ‖P−Pᵀ‖_F=%.3e "
                    "(G-hermiticidad=%.3e; Fase 2 usará eigensolver general).",
                    sym_after,
                    g_herm,
                )

            return ProjectorSynthesisResult(
                P_hat=P_purified,
                reflector=reflector,
                normal_G_norm=n_norm_G,
                is_trivial_obstruction=False,
                idempotence_residual=idem_after,
                symmetry_residual=sym_after,
                dim=self._dim,
                g_hermiticity_residual=g_herm,
                rsi_mcweeney_steps=mcweeney_steps,
                idempotence_residual_pre_rsi=idem_before,
            )

    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║   FASE 2 · ORIENT: AUDITORÍA DE MONODROMÍA DE FLOQUET, MECÁNICA CELESTE ║
    # ║            DE POINCARÉ-KAM Y FLUJO MODULAR DE DE RHAM + RSI-L3          ║
    # ║   Continuación formal de synthesize_projector (ProjectorSynthesisResult)║
    # ║   Morfismo Terminal de Fase 2: audit_monodromy                          ║
    # ╚═════════════════════════════════════════════════════════════════════════╝
    class Phase2_FloquetStabilityAuditor:
        r"""
        **Fase 2 — Auditor de Monodromía de Floquet, Mecánica Celeste de Poincaré y Flujo de de Rham**.

        Inicia su ejecución recibiendo el `ProjectorSynthesisResult` emitido por
        `Phase1_CovariantProjectorSynthesizer.synthesize_projector` (el último método
        de la Fase 1).

        Construye tanto el operador de monodromía algebraico de Fabry-Pérot:
        $$
          \mathcal{M}_{\mathrm{on}} = 2\hat{P} - \hat{P}^2
        $$
        como el operador de monodromía amortiguado por el **Laplaciano Combinatorio
        de Hodge-de Rham** $L_{\mathrm{dR}}$ (Axioma [A2]):
        $$
          \mathcal{M}_{\mathrm{dR}}(\Delta t) = \hat{P}\,e^{-L_{\mathrm{dR}}\Delta t}\,\hat{P}.
        $$
        Evalúa los **Números de Rotación de Poincaré** $\omega_k$, el **Margen Diofántico
        de KAM**, el **Exponente Característico de Lyapunov** $\lambda_{\max}$ y aplica
        sintonización adaptativa RSI-L3 si el espectro roza el borde del disco unitario.

        **Morfismo Terminal de Fase 2:** `audit_monodromy`, cuya salida
        `FloquetMonodromyState` constituye el inicio formal de la Fase 3.
        """

        def __init__(
            self,
            stability_tolerance: float = _DEFAULT_STABILITY_TOL,
            projector_tolerance: float = _DEFAULT_PROJECTOR_TOL,
            metric_tensor: Optional[NDArray[np.float64]] = None,
        ) -> None:
            self._stability_tol = float(stability_tolerance)
            self._projector_tol = float(projector_tolerance)
            self._G: Optional[NDArray[np.float64]] = (
                np.asarray(metric_tensor, dtype=np.float64)
                if metric_tensor is not None
                else None
            )

        @staticmethod
        def _build_de_rham_combinatorial_laplacian(
            dim: int,
            G: Optional[NDArray[np.float64]] = None,
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Construye el **Laplaciano Combinatorio de de Rham** $L_{\mathrm{dR}} \in \mathbb{R}^{d \times d}$
            sobre el grafo de acoplamiento métrico:
              • Si $G$ posee acoplamientos fuera de la diagonal ($A_{ij} = |G_{ij}|$ para $i \neq j$),
                define $L_{\mathrm{dR}} = D - A$ más un potencial armónico de curvatura discreta.
              • Si $G$ es diagonal, construye el Laplaciano de ciclo discreto de de Rham en $S^1$.
            Garantiza $L_{\mathrm{dR}} = L_{\mathrm{dR}}^\top \succeq 0$ y retorna $(L_{\mathrm{dR}}, \lambda_2(L_{\mathrm{dR}}))$.
            """
            if dim <= 1:
                return np.zeros((dim, dim), dtype=np.float64), 0.0

            if G is not None and G.shape == (dim, dim):
                off_diag = np.abs(G - np.diag(np.diag(G)))
                if float(np.max(off_diag)) > _SYM_ATOL:
                    deg = np.sum(off_diag, axis=1)
                    L_dr = np.diag(deg) - off_diag
                else:
                    # Laplaciano combinatorio 1-dimensional de de Rham con pesos métricos
                    L_dr = 2.0 * np.eye(dim, dtype=np.float64)
                    for i in range(dim):
                        j = (i + 1) % dim
                        L_dr[i, j] -= 1.0
                        L_dr[j, i] -= 1.0
            else:
                L_dr = 2.0 * np.eye(dim, dtype=np.float64)
                for i in range(dim):
                    j = (i + 1) % dim
                    L_dr[i, j] -= 1.0
                    L_dr[j, i] -= 1.0

            L_dr = 0.5 * (L_dr + L_dr.T)
            eigs = np.linalg.eigvalsh(L_dr)
            spectral_gap = float(max(0.0, eigs[1])) if eigs.size > 1 else 0.0
            return L_dr, spectral_gap

        @staticmethod
        def _compute_poincare_celestial_metrics(
            multipliers: NDArray[Any],
            spectral_radius: float,
            M_on: NDArray[np.float64],
            P: NDArray[np.float64],
            G: Optional[NDArray[np.float64]],
            de_rham_gap: float,
        ) -> PoincareFloquetCelestialMetrics:
            r"""
            Evalúa los invariantes de la **Mecánica Celeste de Poincaré** y **KAM**:
              1. Números de rotación $\omega_k = |\arg(\mu_k)| / (2\pi) \in [0, 1/2]$.
              2. Margen diofántico de KAM $\min_{1 \le m \le 4, q} |m \omega_k - q| \cdot m^\tau$
                 para detectar proximidad a lenguas de resonancia paramétrica de Arnold.
              3. Exponente característico de Lyapunov $\lambda_{\max} = \ln(\max(\rho(\mathcal{M}), 10^{-16}))$.
              4. Defecto de conservación de estructura simpléctica-métrica $\|\mathcal{M}^\top G \mathcal{M} - G\hat{P}\|_F$.
            """
            mult_complex = np.asarray(multipliers, dtype=np.complex128)
            angles = np.abs(np.angle(mult_complex)) / (2.0 * math.pi)
            rot_numbers = np.sort(angles)

            # Evaluación de condición diofántica de KAM frente a resonancias de orden m ∈ {1,2,3,4}
            non_trivial = rot_numbers[
                (rot_numbers > 1e-7) & (rot_numbers < 0.5 - 1e-7)
            ]
            kam_margin = 1.0
            resonance_order = 0
            if non_trivial.size > 0:
                for omega in non_trivial:
                    for m_ord in range(1, 5):
                        dist_to_int = abs(
                            m_ord * float(omega) - round(m_ord * float(omega))
                        )
                        scaled_margin = dist_to_int * (
                            float(m_ord) ** _KAM_DIOPHANTINE_TAU
                        )
                        if scaled_margin < kam_margin:
                            kam_margin = scaled_margin
                            if dist_to_int < _KAM_DIOPHANTINE_GAMMA:
                                resonance_order = m_ord

            lyapunov_exp = math.log(max(spectral_radius, 1e-16))

            G_mat = (
                G
                if (G is not None and G.shape == P.shape)
                else np.eye(P.shape[0], dtype=np.float64)
            )
            symplectic_def = float(
                la.norm((M_on.T @ G_mat @ M_on) - (G_mat @ P), "fro")
            )

            return PoincareFloquetCelestialMetrics(
                poincare_rotation_numbers=rot_numbers,
                kam_diophantine_margin=float(kam_margin),
                lyapunov_characteristic_exponent=float(lyapunov_exp),
                symplectic_monodromy_defect=symplectic_def,
                de_rham_laplacian_spectral_gap=float(de_rham_gap),
                arnold_tongue_resonance_order=resonance_order,
            )

        @staticmethod
        def _rsi_level3_floquet_damping(
            P: NDArray[np.float64],
            L_dr: NDArray[np.float64],
            spectral_radius: float,
            stability_tol: float,
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            **Automejora Recursiva Nivel 3 (RSI-L3) en Fase 2 — Sintonización Modular de de Rham:**

            Construye el operador de evolución modular $\mathcal{M}_{\mathrm{dR}}(\Delta t) = \hat{P} e^{-L_{\mathrm{dR}}\Delta t} \hat{P}$.
            Si el radio espectral de $\mathcal{M}_{\mathrm{on}}$ presenta una deriva numérica
            marginal en $(1.0, 1.0 + \varepsilon_{\mathrm{cavity}}]$, calcula el paso
            temporal óptimo $\Delta t^* > 0$ que disipa la energía espuria de alta frecuencia
            sin alterar el modo cero armónico de de Rham.
            """
            if 1.0 < spectral_radius <= 1.0 + stability_tol:
                dt_star = min(1e-3, max(1e-12, spectral_radius - 1.0))
            else:
                dt_star = 0.0

            if dt_star > 0.0:
                exp_L = la.expm(-dt_star * L_dr)
                M_damped = P @ exp_L @ P
            else:
                M_damped = 2.0 * P - (P @ P)

            return M_damped, dt_star

        # ────────────────────────────────────────────────────────────────
        # MORFISMO TERMINAL DE FASE 2 ──► ESLABÓN INICIAL DE FASE 3
        # ────────────────────────────────────────────────────────────────
        def audit_monodromy(
            self, synthesis: ProjectorSynthesisResult
        ) -> FloquetMonodromyState:
            r"""
            **Método Terminal de la Fase 2 (`audit_monodromy`)**.

            Recibe el `ProjectorSynthesisResult` emitido por el último método de la Fase 1,
            construye la Matriz de Monodromía $\mathcal{M}_{\mathrm{on}} = 2\hat{P} - \hat{P}^2$
            y el flujo amortiguado de de Rham $\mathcal{M}_{\mathrm{dR}}(\Delta t)$, audita
            el espectro de Floquet bajo el Axioma [A3] y calcula las métricas celestes de
            Poincaré-KAM, entregando el `FloquetMonodromyState` que **inicia formalmente
            los métodos de la Fase 3**.
            """
            P = np.asarray(synthesis.P_hat, dtype=np.float64)
            d = int(P.shape[0])
            if P.shape != (d, d):
                raise DimensionalMismatchError(
                    f"P_hat debe ser cuadrada; se recibió {P.shape}."
                )

            # Re-auditoría de idempotencia en profundidad
            idem = float(la.norm(P @ P - P, "fro"))
            tol_P = self._projector_tol * max(d, 1)
            if idem > tol_P:
                raise ProjectorDefectError(
                    f"[Fase2] Idempotencia rota antes de monodromía: "
                    f"‖P²−P‖_F={idem:.6e} > tol={tol_P:.6e}."
                )

            # Matriz de Monodromía de Fabry-Pérot M_on = 2 P - P²
            M_on = 2.0 * P - (P @ P)

            # Simetría de M_on
            sym_M = float(la.norm(M_on - M_on.T, "fro"))
            is_symmetric = sym_M <= _SYM_ATOL * max(
                float(la.norm(M_on, "fro")), 1.0
            )

            if is_symmetric:
                M_sym = 0.5 * (M_on + M_on.T)
                multipliers = np.linalg.eigvalsh(M_sym).astype(np.float64)
                is_complex = False
            else:
                multipliers = np.linalg.eigvals(M_on).astype(np.complex128)
                is_complex = True
                logger.warning(
                    "[Fase2] M_on no simétrica euclídea (‖M−Mᵀ‖_F=%.3e); "
                    "usando eigensolver en variedad compleja.",
                    sym_M,
                )

            spectral_radius = float(np.max(np.abs(multipliers)))
            is_stable = spectral_radius <= 1.0 + self._stability_tol

            try:
                svals = la.svdvals(P)
                s_pos = svals[svals > 1.0e-14]
                cond_P = (
                    float(s_pos[0] / s_pos[-1])
                    if s_pos.size > 0
                    else float("inf")
                )
            except la.LinAlgError:
                cond_P = float("inf")

            if not is_stable:
                raise FloquetInstabilityError(
                    f"Inestabilidad de Floquet: ρ(M_on)={spectral_radius:.6e} > "
                    f"1+ε={1.0 + self._stability_tol:.6e}. "
                    f"Residuo de idempotencia de P={idem:.6e}."
                )

            # Laplaciano Combinatorio de de Rham, Flujo RSI-L3 y Mecánica Celeste de Poincaré
            L_dr, de_rham_gap = self._build_de_rham_combinatorial_laplacian(
                d, self._G
            )
            M_damped, dt_star = self._rsi_level3_floquet_damping(
                P, L_dr, spectral_radius, self._stability_tol
            )
            celestial_metrics = self._compute_poincare_celestial_metrics(
                multipliers=multipliers,
                spectral_radius=spectral_radius,
                M_on=M_on,
                P=P,
                G=self._G,
                de_rham_gap=de_rham_gap,
            )

            logger.debug(
                "[Fase2] Monodromía: ρ=%.6e, estable=%s, cond(P)=%.3e, "
                "complejo=%s, idem(P)=%.3e, KAM_margin=%.3e.",
                spectral_radius,
                is_stable,
                cond_P,
                is_complex,
                idem,
                celestial_metrics.kam_diophantine_margin,
            )

            return FloquetMonodromyState(
                multipliers=multipliers,
                spectral_radius=spectral_radius,
                is_asymptotically_stable=is_stable,
                condition_number_P=cond_P,
                monodromy_matrix=M_on.copy(),
                projector_idempotence_residual=idem,
                is_complex_manifold=is_complex,
                tolerance_used=self._stability_tol,
                de_rham_damped_monodromy=M_damped,
                celestial_metrics=celestial_metrics,
                rsi_damping_dt=dt_star,
            )

    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║   FASE 3 · ACT: CANAL CUÁNTICO KRAUS (CPTP), RE-UNITARIZACIÓN POLAR     ║
    # ║            STINESPRING RSI-L3, ENTROPÍA DE VON NEUMANN Y HEYTING Ω₃     ║
    # ║   Continuación formal de audit_monodromy (FloquetMonodromyState)        ║
    # ║   Morfismo Terminal de Fase 3: execute_quantum_channel                  ║
    # ╚═════════════════════════════════════════════════════════════════════════╝
    class Phase3_QuantumKrausChannel:
        r"""
        **Fase 3 — Evolución de Canal Cuántico CPTP (Kraus–Stinespring), Re-unitarización RSI-L3 y Veto de Heyting $\Omega_3$**.

        Inicia su ejecución recibiendo el `FloquetMonodromyState` emitido por
        `Phase2_FloquetStabilityAuditor.audit_monodromy` (el último método de la Fase 2)
        junto con el `ProjectorSynthesisResult` de la Fase 1.

        Construye los operadores de Kraus:
        $$
          E_0 = \hat{P}, \qquad E_1 = \mathbf{I}_d - \hat{P}, \qquad
          \rho_{\mathrm{post}} = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger.
        $$
        • Audita la **Completitud Bilateral de Kraus** $\mathcal{C} = \sum_k E_k^\dagger E_k = \mathbf{I}_d$
          (soportando tanto la adjunción euclídea como la adjunción covariante de
          Hilbert-Riemann cuando $\hat{P}$ es $G$-autoadjunto).
        • Ejecuta la **Re-unitarización Polar de Stinespring (RSI-L3)**:
          $\tilde{E}_k = E_k \mathcal{C}^{-1/2}$ cuando detecta derivas sub-umbrales
          de redondeo FPU.
        • Evalúa la entropía cuántica de von Neumann $S(\rho) = -\operatorname{Tr}(\rho\ln\rho)$,
          la pureza $\gamma$, la tasa de desfasamiento de Lindblad, el cartucho de
          antimateria `PositronCartridge` y el veredicto en el **Retículo de Heyting $\Omega_3$**.

        **Morfismo Terminal de Fase 3:** `execute_quantum_channel`.
        """

        def __init__(
            self,
            metric_tensor: NDArray[np.float64],
            kraus_tolerance: float = _DEFAULT_KRAUS_TOL,
        ) -> None:
            self._G = np.asarray(metric_tensor, dtype=np.float64)
            self._dim = int(self._G.shape[0])
            self._kraus_tol = float(kraus_tolerance)
            self._G_fro = max(float(la.norm(self._G, "fro")), 1.0)
            try:
                self._L = la.cholesky(
                    0.5 * (self._G + self._G.T), lower=True, check_finite=False
                )
            except la.LinAlgError:
                self._L = np.eye(self._dim, dtype=np.float64)

        def _verify_kraus_completeness(
            self,
            E0: NDArray[np.float64],
            E1: NDArray[np.float64],
        ) -> Tuple[float, float]:
            r"""
            Verifica $\mathcal{C} = \sum_k E_k^\dagger E_k = \mathbf{I}_d$ de forma **bilateral**:
              • Si $E_0, E_1$ son simétricos euclídeos (o si no son $G$-autoadjuntos),
                evalúa $\mathcal{C} = E_0^\top E_0 + E_1^\top E_1$.
              • Si $G \neq cI$ y $E_0, E_1$ son auténticos proyectores $G$-autoadjuntos
                ($G E_k = E_k^\top G$), evalúa la relación de completitud en el espacio
                de Hilbert blanqueado de Cholesky $\tilde{E}_k = L^\top E_k L^{-\top}$,
                donde la adjunción $G$-métrica coincide con la transpuesta $\tilde{E}_k^\top$.
            """
            d = E0.shape[0]
            I_d = np.eye(d, dtype=np.float64)

            sym_eucl = float(la.norm(E0 - E0.T, "fro")) + float(
                la.norm(E1 - E1.T, "fro")
            )
            sym_g = (
                float(la.norm(self._G @ E0 - E0.T @ self._G, "fro"))
                + float(la.norm(self._G @ E1 - E1.T @ self._G, "fro"))
            ) / self._G_fro

            if sym_eucl > _SYM_ATOL * d and sym_g <= _SYM_ATOL * d:
                # Representación isométrica de Cholesky para operadores G-autoadjuntos
                Zt0 = la.solve_triangular(
                    self._L, E0.T, lower=True, check_finite=False
                )
                Zt1 = la.solve_triangular(
                    self._L, E1.T, lower=True, check_finite=False
                )
                E0_w = self._L.T @ Zt0.T
                E1_w = self._L.T @ Zt1.T
                C = E0_w.T @ E0_w + E1_w.T @ E1_w
            else:
                C = E0.T @ E0 + E1.T @ E1

            diff = 0.5 * ((C - I_d) + (C - I_d).T)
            fro_res = float(la.norm(diff, "fro"))
            eigs = np.linalg.eigvalsh(diff)
            eig_res = float(np.max(np.abs(eigs)))

            tol_eig = self._kraus_tol
            tol_fro = self._kraus_tol * math.sqrt(max(d, 1))

            if eig_res > tol_eig:
                raise KrausCompletenessError(
                    f"C−I no es nula en espectro: max|λ|={eig_res:.6e} > "
                    f"tol={tol_eig:.6e} (λ_min={float(eigs[0]):.6e}, "
                    f"λ_max={float(eigs[-1]):.6e}). "
                    "Violación bilateral de completitud Kraus (CPTP)."
                )
            if fro_res > tol_fro:
                raise KrausTraceViolationError(
                    f"‖C−I‖_F={fro_res:.6e} > tol_fro={tol_fro:.6e}."
                )
            return fro_res, eig_res

        def _rsi_level3_stinespring_reunitarization(
            self,
            E0: NDArray[np.float64],
            E1: NDArray[np.float64],
            eig_res_before: float,
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], bool, float]:
            r"""
            **Automejora Recursiva Nivel 3 (RSI-L3) en Fase 3 — Re-unitarización Polar de Stinespring:**

            En una $C^*$-álgebra, si un par de Kraus $(E_0, E_1)$ presenta una deriva numérica
            sub-umbral $0 < \|\sum_k E_k^\top E_k - I\|_2 \le \tau_{\mathrm{Kraus}}$ (cuando son
            cuasi-simétricos), la transformación canónica:
            $$
              \tilde{E}_k = E_k\,\mathcal{C}^{-1/2}, \qquad
              \mathcal{C}^{-1/2} = V\,\operatorname{diag}(\sigma_i^{-1/2})\,V^\top
            $$
            restaura exactamente $\tilde{E}_0^\top \tilde{E}_0 + \tilde{E}_1^\top \tilde{E}_1 = I_d$.
            """
            target_floor = 4.0 * _MACHINE_EPS * max(self._dim, 1)
            sym_eucl = float(la.norm(E0 - E0.T, "fro")) + float(
                la.norm(E1 - E1.T, "fro")
            )
            if eig_res_before <= target_floor or sym_eucl > 1e-7:
                return E0, E1, False, eig_res_before

            C = 0.5 * ((E0.T @ E0 + E1.T @ E1) + (E0.T @ E0 + E1.T @ E1).T)
            vals, vecs = np.linalg.eigh(C)
            if np.any(vals <= 0.5):
                return E0, E1, False, eig_res_before

            C_inv_sqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
            E0_new = E0 @ C_inv_sqrt
            E1_new = E1 @ C_inv_sqrt

            I_d = np.eye(self._dim, dtype=np.float64)
            C_new = E0_new.T @ E0_new + E1_new.T @ E1_new
            diff_new = 0.5 * ((C_new - I_d) + (C_new - I_d).T)
            eig_after = float(np.max(np.abs(np.linalg.eigvalsh(diff_new))))

            if eig_after < eig_res_before:
                return E0_new, E1_new, True, eig_after
            return E0, E1, False, eig_res_before

        @staticmethod
        def _density_from_pure(
            psi: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""Construye el operador de densidad puro $\rho = |\psi\rangle\langle\psi|$."""
            v = np.asarray(psi, dtype=np.float64).reshape(-1)
            return np.outer(v, v)

        @staticmethod
        def _von_neumann_entropy_rho(
            rho: NDArray[np.float64],
        ) -> float:
            r"""
            Evalúa la entropía cuántica de von Neumann $S(\rho) = -\operatorname{Tr}(\hat{\rho}\ln\hat{\rho})$
            sobre la matriz de densidad normalizada $\hat{\rho} = \rho / \operatorname{Tr}(\rho)$.
            Convención por continuidad analítica: $0\ln 0 := 0$.
            """
            H = 0.5 * (rho + rho.T)
            tr = float(np.trace(H))
            if tr <= 1.0e-30:
                return 0.0
            Hn = H / tr
            eigvals = np.linalg.eigvalsh(Hn)
            eigvals = np.clip(eigvals, 0.0, None)
            pos = eigvals[eigvals > 1.0e-15]
            if pos.size == 0:
                return 0.0
            return float(-np.sum(pos * np.log(pos)))

        @staticmethod
        def _purity(rho: NDArray[np.float64]) -> float:
            r"""Pureza cuántica $\gamma = \operatorname{Tr}(\rho^2) / \operatorname{Tr}(\rho)^2 \in (0, 1]$."""
            H = 0.5 * (rho + rho.T)
            tr = float(np.trace(H))
            if tr <= 1.0e-30:
                return 0.0
            Hn = H / tr
            return float(np.trace(Hn @ Hn))

        @staticmethod
        def _apply_kraus_channel(
            rho: NDArray[np.float64],
            E0: NDArray[np.float64],
            E1: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""Mapa superoperador CPTP $\mathcal{E}(\rho) = E_0 \rho E_0^\top + E_1 \rho E_1^\top$."""
            return E0 @ rho @ E0.T + E1 @ rho @ E1.T

        @staticmethod
        def _sign_antimatter(
            dissipated_entropy: float,
            delta_entropy: float,
            kraus_residual_eig: float,
            secret: bytes = b"MIC_Floquet_v3",
        ) -> str:
            payload = (
                f"{dissipated_entropy:.10e}|"
                f"{delta_entropy:.10e}|"
                f"{kraus_residual_eig:.10e}"
            )
            return hmac.new(
                secret, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()

        @staticmethod
        def _evaluate_heyting_omega3(
            coherent_norm_G: float,
            dissipated_norm_G: float,
            is_floquet_stable: bool,
            eig_res: float,
            kraus_tol: float,
        ) -> Tuple[HeytingOmega3, bool]:
            r"""
            Evalúa el veredicto en el **Retículo Distributivo Acotado de Heyting $\Omega_3$**:
              • `VETOED` ($\top$): inestabilidad de Floquet o aniquilación total de la señal coherente
                cuando existía energía de entrada no nula.
              • `DEGRADED`: disipación de antimateria $> 50\%$ de la energía total o deriva
                de Kraus cercana al umbral.
              • `COHERENT` ($\bot$): tránsito fase-coherente nominal.
            """
            verdict = HeytingOmega3.COHERENT
            total_norm = math.hypot(coherent_norm_G, dissipated_norm_G)

            if not is_floquet_stable:
                verdict = verdict.join(HeytingOmega3.VETOED)
            elif total_norm > _EPS_NORM_TRIVIAL and (
                coherent_norm_G / total_norm
            ) < 1e-12:
                verdict = verdict.join(HeytingOmega3.VETOED)
            elif (
                total_norm > _EPS_NORM_TRIVIAL
                and (dissipated_norm_G / total_norm) > 0.5
            ) or (eig_res > 0.1 * kraus_tol):
                verdict = verdict.join(HeytingOmega3.DEGRADED)

            crowbar_tripped = verdict is HeytingOmega3.VETOED
            return verdict, crowbar_tripped

        def _audit_dimensions_strict(
            self,
            psi_raw: NDArray[np.float64],
            H_obs_gradient: NDArray[np.float64],
            rho: Optional[NDArray[np.float64]] = None,
        ) -> DimensionalAudit:
            psi = np.asarray(psi_raw, dtype=np.float64).reshape(-1)
            grad_arr = np.asarray(H_obs_gradient, dtype=np.float64)
            grad = (
                grad_arr[:, 0].reshape(-1)
                if (grad_arr.ndim == 2 and grad_arr.shape[0] == self._dim)
                else grad_arr.reshape(-1)
            )
            psi_ok = psi.shape == (self._dim,)
            grad_ok = grad.shape == (self._dim,)
            rho_ok: Optional[bool] = None
            if rho is not None:
                rho_ok = rho.shape == (self._dim, self._dim)

            is_coherent = bool(
                psi_ok and grad_ok and (rho_ok if rho is not None else True)
            )
            audit = DimensionalAudit(
                dimension=self._dim,
                psi_dim_ok=psi_ok,
                grad_dim_ok=grad_ok,
                rho_dim_ok=rho_ok,
                is_coherent=is_coherent,
            )
            if not is_coherent:
                raise DimensionalMismatchError(
                    f"Auditoría dimensional fallida: psi_ok={psi_ok}, "
                    f"grad_ok={grad_ok}, rho_ok={rho_ok}, d={self._dim}."
                )
            return audit

        # ────────────────────────────────────────────────────────────────
        # MORFISMO TERMINAL DE FASE 3 Y DEL LAZO DE MONODROMÍA
        # ────────────────────────────────────────────────────────────────
        def execute_quantum_channel(
            self,
            psi_raw: NDArray[np.float64],
            H_obs_gradient: NDArray[np.float64],
            synthesis: ProjectorSynthesisResult,
            monodromy: FloquetMonodromyState,
        ) -> QuantumChannelEvolution:
            r"""
            **Método Terminal de la Fase 3 (`execute_quantum_channel`)**.

            Recibe el `FloquetMonodromyState` emitido por `Phase2_FloquetStabilityAuditor.audit_monodromy`
            y el `ProjectorSynthesisResult` de `Phase1_CovariantProjectorSynthesizer.synthesize_projector`.
            Audita la completitud bilateral de Kraus, aplica la re-unitarización polar
            de Stinespring (RSI-L3), evoluciona la matriz de densidad $\rho$, cuantifica
            la entropía térmica de von Neumann, emite el `PositronCartridge` forense
            y colapsa el veredicto en el retículo de Heyting $\Omega_3$.
            """
            # [A1] Auditoría dimensional bloqueante
            audit = self._audit_dimensions_strict(psi_raw, H_obs_gradient)

            psi = np.asarray(psi_raw, dtype=np.float64).reshape(-1)
            P = np.asarray(synthesis.P_hat, dtype=np.float64)
            I_d = np.eye(self._dim, dtype=np.float64)
            E0 = P
            E1 = I_d - P

            # [A4] Completitud Kraus bilateral (falla rápido si supera tolerancia)
            fro_res, eig_res_before = self._verify_kraus_completeness(E0, E1)

            # [RSI-L3] Re-unitarización Polar de Stinespring para derivas sub-umbrales
            E0_eff, E1_eff, stinespring_applied, eig_res_after = (
                self._rsi_level3_stinespring_reunitarization(
                    E0, E1, eig_res_before
                )
            )

            # Evolución del estado coherente y del operador de densidad
            coherent_state = E0_eff @ psi
            dissipated_vec = E1_eff @ psi

            rho_pre = self._density_from_pure(psi)
            rho_post = self._apply_kraus_channel(rho_pre, E0_eff, E1_eff)
            rho_diss = E1_eff @ rho_pre @ E1_eff.T

            # [A5] Entropías cuánticas de von Neumann, pureza y tasa de Lindblad
            S_pre = self._von_neumann_entropy_rho(rho_pre)
            S_post = self._von_neumann_entropy_rho(rho_post)
            S_diss = self._von_neumann_entropy_rho(rho_diss)
            delta_S = S_post - S_pre
            purity_post = self._purity(rho_post)
            lindblad_rate = (
                -0.5 * math.log(max(purity_post, 1e-16))
                if purity_post > 0.0
                else 0.0
            )

            coherent_norm_G = float(
                np.sqrt(max(float(coherent_state @ (self._G @ coherent_state)), 0.0))
            )
            dissipated_norm_G = float(
                np.sqrt(max(float(dissipated_vec @ (self._G @ dissipated_vec)), 0.0))
            )

            # Emisión forense de antimateria (PositronCartridge)
            antimatter_emission: Optional[PositronCartridge] = None
            if (
                dissipated_norm_G > _EPS_DISSIPATION
                or S_diss > _EPS_DISSIPATION
            ):
                signature = self._sign_antimatter(
                    S_diss, delta_S, eig_res_after
                )
                antimatter_emission = PositronCartridge(
                    inertial_mass=dissipated_norm_G,
                    topological_spin="inverse_hallucination",
                    homological_charge=-1,
                    authorization_signature=signature,
                )
                logger.warning(
                    "[Fase3] Positrón forense emitido "
                    "(S_diss=%.4f, ‖·‖_G=%.2e, sig=%s…).",
                    S_diss,
                    dissipated_norm_G,
                    signature[:8],
                )

            # Evaluación en el Retículo de Heyting Ω₃ y disparo Crowbar (GPIO14)
            heyting_verdict, crowbar_tripped = self._evaluate_heyting_omega3(
                coherent_norm_G=coherent_norm_G,
                dissipated_norm_G=dissipated_norm_G,
                is_floquet_stable=monodromy.is_asymptotically_stable,
                eig_res=eig_res_after,
                kraus_tol=self._kraus_tol,
            )

            lyapunov_witness = (
                monodromy.celestial_metrics.lyapunov_characteristic_exponent
                if monodromy.celestial_metrics is not None
                else 0.0
            )
            rsi_metrics = RSILevel3ControlMetrics(
                mcweeney_purification_steps=synthesis.rsi_mcweeney_steps,
                de_rham_damping_dt=monodromy.rsi_damping_dt,
                stinespring_polar_reunitarization_applied=stinespring_applied,
                idempotence_defect_before=synthesis.idempotence_residual_pre_rsi,
                idempotence_defect_after=synthesis.idempotence_residual,
                kraus_eig_defect_before=eig_res_before,
                kraus_eig_defect_after=eig_res_after,
                lyapunov_contraction_witness=lyapunov_witness,
            )

            logger.info(
                "[Fase3] Canal CPTP: ‖C−I‖_F=%.2e, max|λ(C−I)|=%.2e, "
                "S_pre=%.4f, S_post=%.4f, ΔS=%.4f, purity_post=%.4f, "
                "ρ(M)=%.6e, Heyting=%s.",
                fro_res,
                eig_res_after,
                S_pre,
                S_post,
                delta_S,
                purity_post,
                monodromy.spectral_radius,
                heyting_verdict.value,
            )

            return QuantumChannelEvolution(
                coherent_state=coherent_state,
                rho_post=0.5 * (rho_post + rho_post.T),
                dissipated_entropy=S_diss,
                von_neumann_pre=S_pre,
                von_neumann_post=S_post,
                delta_entropy=delta_S,
                antimatter_emission=antimatter_emission,
                kraus_residual_fro=fro_res,
                kraus_residual_eig=eig_res_after,
                purity_post=purity_post,
                dimensional_audit=audit,
                monodromy_state=monodromy,
                heyting_verdict=heyting_verdict,
                gpio14_crowbar_tripped=crowbar_tripped,
                lindblad_dephasing_rate=lindblad_rate,
                rsi_metrics=rsi_metrics,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # INTERFAZ PÚBLICA DEL ENDOFUNTOR Y ACOPLAMIENTO CON SEMANTIC_PARABOLIC_MIRROR
    # ══════════════════════════════════════════════════════════════════════════
    def purify_and_tune_cavity(
        self,
        raw_llm_logits: NDArray[np.float64],
        h_obs_gradient: NDArray[np.float64],
    ) -> QuantumChannelEvolution:
        r"""
        **Funtor Maestro de Focalización de Lazo ($\mathcal{Z}_{\mathrm{floquet}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$)**:

        1. `phase1.synthesize_projector` $\longrightarrow$ `ProjectorSynthesisResult`
        2. `phase2.audit_monodromy`      $\longrightarrow$ `FloquetMonodromyState`
        3. `phase3.execute_quantum_channel` $\longrightarrow$ `QuantumChannelEvolution`
        """
        logger.info(
            "[FloquetMonodromyAgent] Sintonizando cavidad y purgando alucinaciones."
        )

        # Fase 1 ──► Observe
        synthesis = self.phase1.synthesize_projector(h_obs_gradient)

        # Fase 2 ──► Orient (continuación formal de synthesis)
        monodromy = self.phase2.audit_monodromy(synthesis)
        logger.debug(
            "Monodromía certificada: ρ=%.6e, estable=%s, idem=%.3e.",
            monodromy.spectral_radius,
            monodromy.is_asymptotically_stable,
            monodromy.projector_idempotence_residual,
        )

        # Fase 3 ──► Act (continuación formal de monodromy + synthesis)
        evolution = self.phase3.execute_quantum_channel(
            psi_raw=raw_llm_logits,
            H_obs_gradient=h_obs_gradient,
            synthesis=synthesis,
            monodromy=monodromy,
        )
        self._last_evolution = evolution
        if evolution.gpio14_crowbar_tripped:
            self._gpio14_crowbar_latched = True
        return evolution

    def guide_parabolic_mirror(
        self,
        mirror: Any,
        raw_llm_logits: NDArray[np.float64],
        h_obs_gradient: NDArray[np.float64],
        *,
        force_von_neumann: bool = False,
    ) -> Tuple[QuantumChannelEvolution, Any]:
        r"""
        **Acoplamiento Soberano de Calibre (`FloquetMonodromyAgent` $\longrightarrow$ `SemanticParabolicMirror`)**.

        Guía directamente al motor óptico `SemanticParabolicMirror`:
          1. Ejecuta el lazo de 3 fases de Floquet (`purify_and_tune_cavity`) para
             certificar la estabilidad de monodromía, el flujo de de Rham y el canal CPTP.
          2. Vincula la normal covariante $n = G \nabla \mathcal{H}_{\mathrm{obs}}$ al
             espejo parabólico (`mirror.bind_constraints`) e invoca `mirror.refract(...)`
             (o `mirror.apply(...)` si es una versión anterior), retornando la tupla
             `(QuantumChannelEvolution, RefractedState)`.
        """
        evolution = self.purify_and_tune_cavity(raw_llm_logits, h_obs_gradient)

        n_cov = self._G @ np.asarray(h_obs_gradient, dtype=np.float64).reshape(-1)
        if hasattr(mirror, "bind_constraints"):
            mirror.bind_constraints(n_cov)
        elif hasattr(mirror, "bind_constraint"):
            mirror.bind_constraint(n_cov)

        if hasattr(mirror, "refract"):
            refracted = mirror.refract(
                evolution.coherent_state,
                n_cov,
                force_von_neumann=force_von_neumann,
            )
        else:
            coherent, trace = mirror.apply(
                evolution.coherent_state,
                n_cov,
                force_von_neumann=force_von_neumann,
            )
            refracted = (coherent, trace)

        return evolution, refracted

    def forward(self, state: CategoricalState) -> CategoricalState:
        """Reflexión Floquet sobre un `CategoricalState` del topos MIC."""
        psi = np.asarray(state.payload, dtype=np.float64).reshape(-1)
        if psi.shape != (self._dim,):
            raise DimensionalMismatchError(
                f"state.payload shape={psi.shape} ≠ ({self._dim},)."
            )
        grad = getattr(state, "constraint_normal", None)
        if grad is None:
            grad = np.zeros(self._dim, dtype=np.float64)
            grad[0] = 1.0
        evolution = self.purify_and_tune_cavity(psi, grad)
        return CategoricalState(
            payload=evolution.coherent_state,
            label=getattr(state, "label", ""),
        )

    def backward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Adjunta categórica. Para un reflector de Householder métrico $R = I - 2P$
        se cumple $R^{-1} = R$ (involución); se reutiliza `forward` por convención
        del topos $\mathcal{T}_{\mathrm{MIC}}$.
        """
        return self.forward(state)

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Ejecuta el morfismo `forward` sobre el estado categórico."""
        return self.forward(state)

    # ──────────────────────────────────────────────────────────────────────────
    # PROPIEDADES Y REPORTE FORENSE DOCTORAL
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        """Copia defensiva de $G$."""
        return self._G.copy()

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def phase1_synthesizer(self) -> ProjectorSynthesizerPort:
        return self.phase1

    @property
    def phase2_auditor(self) -> FloquetAuditorPort:
        return self.phase2

    @property
    def phase3_channel(self) -> KrausChannelPort:
        return self.phase3

    @property
    def last_evolution(self) -> Optional[QuantumChannelEvolution]:
        return self._last_evolution

    @property
    def gpio14_crowbar_latched(self) -> bool:
        return self._gpio14_crowbar_latched

    def reset_crowbar_gpio14(self) -> None:
        """Rearma explícitamente el cerrojo del actuador Crowbar (`GPIO14 = LOW`)."""
        self._gpio14_crowbar_latched = False

    def diagnostic_report(self) -> Dict[str, Any]:
        """Reporte diagnóstico integral de las 3 fases, Poincaré-KAM y RSI-L3."""
        report: Dict[str, Any] = {
            "agent": "FloquetMonodromyAgent",
            "version": "4.0.0-Poincare-Celestial-RSI-L3",
            "dimension": self._dim,
            "gpio14_crowbar_latched": self._gpio14_crowbar_latched,
        }
        if self._last_evolution is not None:
            ev = self._last_evolution
            report["quantum_channel"] = {
                "heyting_verdict": ev.heyting_verdict.value,
                "purity_post": ev.purity_post,
                "dissipated_entropy": ev.dissipated_entropy,
                "delta_entropy": ev.delta_entropy,
                "lindblad_dephasing_rate": ev.lindblad_dephasing_rate,
                "kraus_residual_fro": ev.kraus_residual_fro,
                "kraus_residual_eig": ev.kraus_residual_eig,
                "antimatter_emitted": ev.antimatter_emission is not None,
            }
            if ev.monodromy_state is not None:
                m = ev.monodromy_state
                report["floquet_monodromy"] = {
                    "spectral_radius": m.spectral_radius,
                    "is_asymptotically_stable": m.is_asymptotically_stable,
                    "condition_number_P": m.condition_number_P,
                    "rsi_damping_dt": m.rsi_damping_dt,
                }
                if m.celestial_metrics is not None:
                    c = m.celestial_metrics
                    report["poincare_celestial"] = {
                        "kam_diophantine_margin": c.kam_diophantine_margin,
                        "lyapunov_characteristic_exponent": (
                            c.lyapunov_characteristic_exponent
                        ),
                        "symplectic_monodromy_defect": (
                            c.symplectic_monodromy_defect
                        ),
                        "de_rham_laplacian_spectral_gap": (
                            c.de_rham_laplacian_spectral_gap
                        ),
                        "arnold_tongue_resonance_order": (
                            c.arnold_tongue_resonance_order
                        ),
                    }
            if ev.rsi_metrics is not None:
                r = ev.rsi_metrics
                report["rsi_level3"] = {
                    "mcweeney_purification_steps": r.mcweeney_purification_steps,
                    "de_rham_damping_dt": r.de_rham_damping_dt,
                    "stinespring_polar_reunitarization_applied": (
                        r.stinespring_polar_reunitarization_applied
                    ),
                    "idempotence_defect_before": r.idempotence_defect_before,
                    "idempotence_defect_after": r.idempotence_defect_after,
                    "kraus_eig_defect_before": r.kraus_eig_defect_before,
                    "kraus_eig_defect_after": r.kraus_eig_defect_after,
                }
        return report


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "FloquetInstabilityError",
    "KrausTraceViolationError",
    "KrausCompletenessError",
    "DimensionalMismatchError",
    "ProjectorDefectError",
    "FloquetParameterError",
    "PoincareKamResonanceError",
    # DTOs y Retículo de Heyting
    "HeytingOmega3",
    "DimensionalAudit",
    "PoincareFloquetCelestialMetrics",
    "RSILevel3ControlMetrics",
    "ProjectorSynthesisResult",
    "FloquetMonodromyState",
    "QuantumChannelEvolution",
    # Protocolos
    "ProjectorSynthesizerPort",
    "FloquetAuditorPort",
    "KrausChannelPort",
    # Agente Orquestador
    "FloquetMonodromyAgent",
]
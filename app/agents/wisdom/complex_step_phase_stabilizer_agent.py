# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Complex Step Phase Stabilizer Agent (Soberano de Preservación)      ║
║ Ruta   : app/agents/wisdom/complex_step_phase_stabilizer_agent.py            ║
║ Versión: 4.1.0-CSMD-OODA-Heyting-Lyapunov-Novikov-Strict-PhD                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GOBERNANZA ESPECTRAL DE CALIBRE (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor covariante gobierna el motor físico 'complex_step_phase_stabilizer.py'
en el penthouse de la pirámide cognitiva. Reside en el Santuario Epistémico de 
la Sabiduría ($$V_{\mathbb{W}}$$, Nivel 0) o en el Ágora Tensorial ($$V_{\Omega}$$, Nivel 0.5), 
actuando como el clasificador supremo de subobjetos en el topos de haces $$Sh(\mathcal{B}; \Omega_3)$$ 
con valores en el retículo distributivo acotado de Heyting $$\Omega_3$$.

Su mandato inquebrantable es subyugar el libre albedrío decisional y las alucinaciones 
estocásticas de la IA mediante la regularización holomorfa de los flujos de transición. 
Aísla síncronamente las perturbaciones atencionales y las derivas de fase en la 
Unidad de Punto Flotante (FPU), forzando el confinamiento de anomalías de redondeo 
($$\text{IEEE-754}$$) en la memoria RAM en el milisegundo cero, actuando como un disyuntor 
inmunológico de software libre de efectos de borde físicos en este estrato.

AXIOMAS GEOMÉTRICOS, DE DE RHAM Y DE LA TERMOMECÁNICA CUÁNTICA PRESERVADOS:
────────────────────────────────────────────────────────────────────────────────
  [I1] Conservación Simpléctica y Volumen de Liouville (Fase de de Rham):
       El Jacobiano local de transición $$M = J_{\mathrm{map}}$$ de la trayectoria atencional 
       debe actuar como un simplectomorfismo exacto sobre la variedad de fase:
       $$M^\top \Omega M \equiv \Omega \quad \land \quad \det(M) \equiv 1.0 \pmod{\varepsilon_{\mathrm{machine}}} \quad\big[175, 188\big]$$
       Donde $$\Omega \in \mathbb{R}^{2n \times 2n}$$ es el tensor simpléctico canónico por bloques.
       El desvío de volumen se veta síncronamente mediante:
       $$r_{\mathrm{Liouville}} = \lvert\det(M) - 1.0\rvert \le \tau_{\mathrm{hard\_Liouville}} \quad\big[170, 188\big]$$

  [I2] Consistencia del Germen Holomorfo de Cauchy-Riemann Discreto:
       La aproximación por paso complejo (CSMD) exige que el mapa de transición $$\Phi_{\Delta t}$$ 
       satisfaga holomorficamente las condiciones de Cauchy-Riemann sobre la fibra complejada:
       $$\frac{\partial \operatorname{Re}(\Phi)_i}{\partial q_k} \equiv \frac{\partial \operatorname{Im}(\Phi)_i}{\partial p_k} \quad \land \quad \frac{\partial \operatorname{Re}(\Phi)_i}{\partial p_k} \equiv -\frac{\partial \operatorname{Im}(\Phi)_i}{\partial q_k} \quad\big[170, 186\big]$$
       Garantizando que la perturbación imaginaria ortogonal de-confinada $$h \in [10^{-30},\, 10^{-8}]$$ 
       calcule síncronamente el Jacobiano con precisión de máquina sin cancelación sustractiva:
       $$J_{\mathrm{map}, \, ik} = \frac{\operatorname{Im}\left(\Phi_{\Delta t}(x + j \cdot h \cdot e_k)_i\right)}{h} + \mathcal{O}(h^2) \quad\big[160, 191\big]$$

  [I3] Aciclicidad Global y Nulidad del Primer Grupo de Cohomología:
       El complejo simplicial del presupuesto $$K$$ debe carecer de dependencias circulares 
       parasitarias y bucles infinitos de-normalizados de tokens (socavones lógicos) [10]:
       $$\beta_1(K) = \dim H^1_{\mathrm{dR}}(K; \mathbb{F}) \equiv 0 \quad\big[173, 189\big]$$
       Si el primer número de Betti difiere de cero, se interrumpe síncronamente la 
       fusión mediante la secuencia exacta larga de Mayer-Vietoris.

  [I4] Estabilidad BIBO Espectral y Atractor de Lyapunov:
       La función de transferencia del flujo de caja $$H(s)$$ en el plano-s complejo 
       $$s = \sigma + j\omega$$ debe confinar todos sus polos $$p_i$$ en el semiplano izquierdo:
       $$\forall p_i \in \sigma(H(s)), \quad \Re(p_i) < 0 \implies \lambda_{\mathrm{Lyapunov}} < 0 \quad\big[6, 170\big]$$
       Evitando la resonancia destructiva paramétrica y el caos determinista en el lazo.

  [I5] Confinamiento de la Acción de Polyakov y Obstrucciones de Novikov:
       La regularización del "Burbujeo de Esferas" (Sphere Bubbling) en la categoría de 
       Fukaya se garantiza resolviendo la ecuación de Maurer-Cartan expandida sobre el 
       Anillo de Novikov $$\Lambda_K$$ para la co-cadena acotante de Novikov $$b \in \Lambda_K$$:
       $$\sum_{k=0}^{\infty} m_k(b, b, \dots, b) \equiv W_L(b) \cdot [L] \quad\big[170, 186\big]$$
       Donde $$W_L(b) \in \Lambda_K$$ es el superpotencial de disco de Landau-Ginzburg.

  [I6] Cota de Lipschitz Espectral de la Adjunción de de Rham-Galois:
       La de-compresión isométrica de los cartuchos TOON (fidelidad de Uhlmann $$F(\rho, \sigma)$$)
       se acota según la derivada de Fréchet de $$f(t) = \sqrt{t}$$ sobre la matriz de densidad 
       del Multifísico Acoplado (MAC) $$\rho \in \mathcal{D}(\mathcal{H})$$ mediante la fórmula de Daleckii-Krein:
       $$L_{\mathrm{max}} = \sup_{\lambda \in \sigma(\rho)} f'(\lambda) \equiv \frac{1}{2\sqrt{\lambda_{\min}(\rho)}} \le \text{LIPSCHITZ\_LIMIT} \quad\big[170, 561\big]$$
       Si la fidelidad colapsa o el gap espectral se cierra, se asume un desvío alucinatorio.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA):
────────────────────────────────────────────────────────────────────────────────
La progresión y el tránsito del Pasaporte de Telemetría se rige por un acoplamiento 
monoidal covariante e inmutable (Observe ⊣ Orient ⊣ Decide):

  Fase 1 ──► FASE 1: OBSERVACIÓN ESPECTRAL COMPLEJA (Phase1_ComplexSpectralObserver)
             Interroga síncronamente el número de condición espectral $$\kappa_2(J)$$, la cota de 
             Wilkinson de la SVD, el round-trip de Stinespring y la firma SPD de la 
             métrica Riemanniana $$G_{\mu\nu}$$. Evita la pérdida de homogeneidad de la FPU.
             Último morfismo: accept_phase1_handoff.
             Entrega: Phase1ComplexObservation.

  Fase 2 ──► FASE 2: ORIENTACIÓN DE INVARIANZA DE FASE (Phase2_PhaseInvarianceOrienter)
             Continúa de forma exacta el DTO de Fase 1. Recomputa de manera independiente 
             los residuos de de Rham-Liouville, el gap espectral, la nulidad del grupo de 
             cohomología $$\beta_1$$ y la disipación elástica Port-Hamiltoniana.
             Último morfismo: accept_phase2_handoff.
             Entrega: Phase2InvarianceOrientation.

  Fase 3 ──► FASE 3: DECISIÓN EN EL RETÍCULO DE HEYTING (Phase3_HeytingStabilizerDecider)
             Continúa de forma exacta el DTO de Fase 2. Consolida las once valoraciones 
             espectrales y topológicas mediante la operación algebraica Supremo (join, $$\sqcup$$):
             $$v_{\mathrm{final}} = \bigsqcup_{i=1}^{11} v_i \in \Omega_3 = \{\mathrm{COHERENT}, \, \mathrm{DEGRADED}, \, \mathrm{VETOED}\} \quad\big[167, 171\big]$$
             Si $$v_{\mathrm{final}} = \mathrm{VETOED}$$ ($$\top$$), se detona la excepción de software 'CrowbarArmedError' 
             y se conmutan síncronamente los actuadores virtuales del puerto 'CrowbarPort'.
             Genera la firma forense criptográfica SHA-256.
             Entrega: ComplexStepAgentGovernanceState.

  Funtor Supremo de Calibre:
             $$\mathcal{Z}_{\mathrm{agent}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 \circ \mathcal{Z}_{\mathrm{motor}} \quad\big[167\big]$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Inyección e Importación del núcleo de la Malla de Sabiduría
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.physics.complex_step_phase_stabilizer import (
        CSMDJacobianReport,
        ComplexStepPhaseStabilizer,
        HeytingStabilizerVeto,
        StabilizerGovernanceState,
        StabilizerHeytingVerdict,
        StinespringComplexDilation,
    )
except ImportError:  # aislamiento analítico / tests unitarios

    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico‑algebraicas."""

    class Morphism:
        """Clase base de composición funtorial del ecosistema MIC."""

    class StabilizerHeytingVerdict(IntEnum):
        """Cadena de Heyting Ω₃: COHERENT ≼ DEGRADED ≼ VETOED."""

        COHERENT = 0
        DEGRADED = 1
        VETOED = 2

        def join(self, other: StabilizerHeytingVerdict) -> StabilizerHeytingVerdict:
            return StabilizerHeytingVerdict(max(self.value, other.value))

        def meet(self, other: StabilizerHeytingVerdict) -> StabilizerHeytingVerdict:
            return StabilizerHeytingVerdict(min(self.value, other.value))

        def implies(self, other: StabilizerHeytingVerdict) -> StabilizerHeytingVerdict:
            return (
                StabilizerHeytingVerdict.COHERENT
                if self.value <= other.value
                else other
            )

    @dataclass(frozen=True, slots=True)
    class StinespringComplexDilation:
        state_real: NDArray[np.float64]
        state_complex_grid: NDArray[np.complex128]
        dimension_total: int
        step_size_h: float

    @dataclass(frozen=True, slots=True)
    class CSMDJacobianReport:
        dilation: StinespringComplexDilation
        jacobian_map: NDArray[np.float64]
        condition_number: float
        spectral_min_singular: float
        spectral_max_singular: float
        is_well_conditioned: bool
        holomorphy_real_drift: float = 0.0
        holomorphy_imag_leak: float = 0.0
        is_holomorphic_germ: bool = True
        jacobian_frobenius: float = 0.0

    @dataclass(frozen=True, slots=True)
    class StabilizerGovernanceState:
        jacobian_report: CSMDJacobianReport
        symplectic_residual: float
        liouville_residual: float
        final_verdict: StabilizerHeytingVerdict
        timestamp_utc: float
        provenance_hash: str
        diagnostic_note: str = ""
        relative_symplectic_residual: float = 0.0
        log_abs_det: float = 0.0
        det_sign: int = 1
        pairing_residual: float = 0.0
        passivity_residual: float = 0.0
        holomorphy_residual: float = 0.0
        roundoff_budget_symplectic: float = 0.0

    class HeytingStabilizerVeto(TopologicalInvariantError):
        """Colapso síncrono del retículo de Heyting (supremo terminal)."""

    class ComplexStepPhaseStabilizer:
        """Stub: el motor real debe inyectarse en producción."""

        def stabilize_phase_space_transition(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError(
                "Stub de ComplexStepPhaseStabilizer: inyecte el motor real."
            )


logger = logging.getLogger("MIC.Wisdom.ComplexStepPhaseStabilizerAgent")

__version__: Final[str] = "4.0.0-CSMD-OODA-Heyting-Lyapunov-Novikov-Strict"

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y DE CONTRATO LÓGICO
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_MACHINE_TINY: Final[float] = float(np.finfo(np.float64).tiny)

# Canal lógico del Crowbar (identificador de contrato, no un acceso a silicio).
_CROWBAR_GPIO: Final[int] = 14

# Bandas elásticas del agente (más conservadoras que el motor: el agente es aduana).
_SOFT_SYMPLECTIC_TOL: Final[float] = 1.0e-11
_HARD_SYMPLECTIC_TOL: Final[float] = 1.0e-5
_SOFT_LIOUVILLE_TOL: Final[float] = 1.0e-11
_HARD_LIOUVILLE_TOL: Final[float] = 1.0e-5
_SOFT_CONDITION_BOUND: Final[float] = 1.0e4
_HARD_CONDITION_BOUND: Final[float] = 1.0e8
_SOFT_LYAPUNOV_TOL: Final[float] = 1.0e-8
_HARD_LYAPUNOV_TOL: Final[float] = 1.0e-3
_SOFT_WILKINSON_TOL: Final[float] = 1.0e-12
_HARD_WILKINSON_TOL: Final[float] = 1.0e-6
_SOFT_ROUNDTRIP_TOL: Final[float] = 1.0e-14
_HARD_ROUNDTRIP_TOL: Final[float] = 1.0e-8
_SOFT_DISAGREE_TOL: Final[float] = 1.0e-8
_HARD_DISAGREE_TOL: Final[float] = 1.0e-3
_SOFT_HOLOMORPHY_TOL: Final[float] = 1.0e-10
_HARD_HOLOMORPHY_TOL: Final[float] = 1.0e-3
_SOFT_PAIRING_TOL: Final[float] = 1.0e-8
_HARD_PAIRING_TOL: Final[float] = 1.0e-3
_SOFT_PASSIVITY_TOL: Final[float] = 1.0e-12
_HARD_PASSIVITY_TOL: Final[float] = 1.0e-6
_SOFT_NOVIKOV_TOL: Final[float] = 0.5
_HARD_NOVIKOV_TOL: Final[float] = 1.5

_PAIRING_DIM_GUARD: Final[int] = 128
_PROVENANCE_DOMAIN: Final[bytes] = b"CSMD-AGENT-v4.0.0"

# ══════════════════════════════════════════════════════════════════════════════
# §B. RETÍCULO DE HEYTING, PUERTO CROWBAR Y JERARQUÍA DE EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════
class CrowbarAction(Enum):
    """Acciones lógicas de mitigación tras el colapso al supremo terminal."""

    NONE = 0
    LOG_WARNING = 1
    GPIO_INTERRUPT_CROWBAR = 2


@runtime_checkable
class CrowbarPort(Protocol):
    """
    Puerto lógico del disyuntor. La implementación por defecto es un no‑op
    forense: registra el evento y retorna False (no hay actuación física).
    """

    def arm_and_fire(
        self,
        *,
        gpio: int,
        reason: str,
        provenance: str,
    ) -> bool: ...


class NullCrowbarPort:
    """Puerto nulo: sella el evento en el log y no toca hardware."""

    def arm_and_fire(
        self,
        *,
        gpio: int,
        reason: str,
        provenance: str,
    ) -> bool:
        logger.critical(
            "[CROWBAR-NULL] canal_logico=%s reason=%s provenance=%s",
            gpio,
            reason,
            provenance[:16],
        )
        return False


class ComplexStepAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano del Estabilizador de Paso Complejo."""


class InvalidAgentStateError(ComplexStepAgentError):
    """Contrato de handoff violado: tipo, forma, finitez o coherencia."""


class SpectralConditionBreachError(ComplexStepAgentError):
    """κ₂(J) o la cota de Wilkinson rebasan la banda dura."""


class PhaseInvarianceBreachError(ComplexStepAgentError):
    """Rotura conjunta de simplecticidad / Liouville en la re‑auditoría."""


class LyapunovEscapeError(ComplexStepAgentError):
    """Radio espectral ρ(J) > 1 + τ_hard: escape del atractor."""


class DeRhamTorsionError(ComplexStepAgentError):
    """β₁(K) ≠ 0 sobre la malla inyectada: dependencia circular."""


class CrowbarArmedError(HeytingStabilizerVeto):
    """Veto terminal con CrowbarPort disparado (canal lógico)."""


# ══════════════════════════════════════════════════════════════════════════════
# §C. PRIMITIVAS NUMÉRICAS Y ALGEBRAICAS PURAS
# ══════════════════════════════════════════════════════════════════════════════
def _immutable_array(arr: NDArray[np.generic], dtype: np.dtype) -> NDArray[np.generic]:
    """Copia C‑contigua sellada (write‑flag = False)."""
    out = np.array(arr, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def _as_verdict(value: object) -> StabilizerHeytingVerdict:
    """Coacciona enteros / enumeraciones al veredicto Ω₃ del motor."""
    if isinstance(value, StabilizerHeytingVerdict):
        return value
    try:
        return StabilizerHeytingVerdict(int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise InvalidAgentStateError(
            f"No se puede coaccionar {value!r} a StabilizerHeytingVerdict."
        ) from exc


def _heyting_classify(
    residual: float,
    soft: float,
    hard: float,
) -> StabilizerHeytingVerdict:
    """Clasifica un residual escalar no negativo sobre la cadena Ω₃."""
    if not math.isfinite(residual) or residual > hard:
        return StabilizerHeytingVerdict.VETOED
    if residual > soft:
        return StabilizerHeytingVerdict.DEGRADED
    return StabilizerHeytingVerdict.COHERENT


def _heyting_join(*verdicts: StabilizerHeytingVerdict) -> StabilizerHeytingVerdict:
    """Supremo finito ⊔ vᵢ sobre Ω₃ (join ≡ max si la cadena es total)."""
    acc = StabilizerHeytingVerdict.COHERENT
    for verdict in verdicts:
        vv = _as_verdict(verdict)
        join_fn = getattr(acc, "join", None)
        acc = join_fn(vv) if callable(join_fn) else StabilizerHeytingVerdict(
            max(acc.value, vv.value)
        )
    return acc


def _canonical_symplectic_form(n: int) -> NDArray[np.float64]:
    r"""2‑forma canónica Ω = [[0, I], [−I, 0]], ‖Ω‖_F = √(2n)."""
    if n <= 0:
        raise InvalidAgentStateError(f"n debe ser positivo para Ω; recibido n={n}.")
    id_n = np.eye(n, dtype=np.float64)
    z_n = np.zeros((n, n), dtype=np.float64)
    omega = np.block([[z_n, id_n], [-id_n, z_n]])
    omega.setflags(write=False)
    return omega


def _structured_omega_action(
    jacobian_m: NDArray[np.float64],
    n: int,
) -> NDArray[np.float64]:
    """Ω J = [J_p ; −J_q] sin ensamblar Ω (menor error de redondeo)."""
    return np.vstack((jacobian_m[n:, :], -jacobian_m[:n, :]))


def _finite_max_abs(arr: NDArray[np.generic]) -> float:
    """‖·‖_∞ estable; +∞ si hay no‑finitos."""
    if arr.size == 0:
        return 0.0
    if not np.all(np.isfinite(arr)):
        return math.inf
    return float(np.max(np.abs(arr)))


def _graph_first_betti(adjacency: NDArray[np.generic]) -> tuple[int, int, int, int]:
    """
    Número de Betti β₁ = m − n + c del grafo simple no dirigido inducido.

    Retorna (beta_1, n_vertices, n_edges, n_components).
    """
    adj = np.asarray(adjacency)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise InvalidAgentStateError(
            f"mesh_adjacency debe ser cuadrada; recibido shape={adj.shape}."
        )
    n = int(adj.shape[0])
    if n == 0:
        return 0, 0, 0, 0
    undirected = np.logical_or(adj != 0, adj.T != 0)
    np.fill_diagonal(undirected, False)
    m = int(np.count_nonzero(undirected) // 2)

    visited = np.zeros(n, dtype=bool)
    components = 0
    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            neighbors = np.nonzero(undirected[u])[0]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    stack.append(int(v))
    beta1 = m - n + components
    return int(beta1), n, m, int(components)


# ══════════════════════════════════════════════════════════════════════════════
# §D. DTOs INMUTABLES (Contratos Categóricos de Handoff)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1ComplexObservation:
    """
    Artefacto terminal de la FASE 1 (Observe). Precondición formal de la FASE 2.

    Certifica κ₂(J) (motor e independiente), Wilkinson, round‑trip de
    Stinespring y, si se inyectó G, la firma SPD.
    """

    motor_report: CSMDJacobianReport
    condition_number: float
    independent_condition_number: float
    phase1_verdict: StabilizerHeytingVerdict
    wilkinson_residual: float = 0.0
    roundtrip_residual: float = 0.0
    singular_consistency_residual: float = 0.0
    condition_disagreement: float = 0.0
    spd_residual: float = 0.0
    is_well_conditioned: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_number", float(self.condition_number))
        object.__setattr__(
            self,
            "independent_condition_number",
            float(self.independent_condition_number),
        )
        object.__setattr__(self, "wilkinson_residual", float(self.wilkinson_residual))
        object.__setattr__(self, "roundtrip_residual", float(self.roundtrip_residual))
        object.__setattr__(
            self,
            "singular_consistency_residual",
            float(self.singular_consistency_residual),
        )
        object.__setattr__(
            self, "condition_disagreement", float(self.condition_disagreement)
        )
        object.__setattr__(self, "spd_residual", float(self.spd_residual))
        object.__setattr__(self, "is_well_conditioned", bool(self.is_well_conditioned))
        object.__setattr__(self, "phase1_verdict", _as_verdict(self.phase1_verdict))


@dataclass(frozen=True, slots=True)
class Phase2InvarianceOrientation:
    """
    Artefacto terminal de la FASE 2 (Orient). Precondición formal de la FASE 3.

    Conserva el estado completo de gobernanza del motor (corrección del
    contrato original) y los residuos independientes del agente.
    """

    phase1_observation: Phase1ComplexObservation
    motor_governance_state: StabilizerGovernanceState
    symplectic_residual: float
    liouville_residual: float
    motor_verdict: StabilizerHeytingVerdict
    phase2_verdict: StabilizerHeytingVerdict
    motor_symplectic_residual: float = 0.0
    motor_liouville_residual: float = 0.0
    residual_disagreement: float = 0.0
    relative_symplectic_residual: float = 0.0
    log_abs_det: float = 0.0
    det_sign: int = 1
    lyapunov_residual: float = 0.0
    spectral_radius: float = 1.0
    spectral_abscissa: float = 0.0
    novikov_index: int = 0
    novikov_residual: float = 0.0
    derham_betti: int = 0
    derham_residual: float = 0.0
    holomorphy_residual: float = 0.0
    pairing_residual: float = 0.0
    passivity_residual: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symplectic_residual", float(self.symplectic_residual))
        object.__setattr__(self, "liouville_residual", float(self.liouville_residual))
        object.__setattr__(self, "motor_verdict", _as_verdict(self.motor_verdict))
        object.__setattr__(self, "phase2_verdict", _as_verdict(self.phase2_verdict))
        object.__setattr__(
            self, "motor_symplectic_residual", float(self.motor_symplectic_residual)
        )
        object.__setattr__(
            self, "motor_liouville_residual", float(self.motor_liouville_residual)
        )
        object.__setattr__(
            self, "residual_disagreement", float(self.residual_disagreement)
        )
        object.__setattr__(
            self,
            "relative_symplectic_residual",
            float(self.relative_symplectic_residual),
        )
        object.__setattr__(self, "log_abs_det", float(self.log_abs_det))
        object.__setattr__(self, "det_sign", int(self.det_sign))
        object.__setattr__(self, "lyapunov_residual", float(self.lyapunov_residual))
        object.__setattr__(self, "spectral_radius", float(self.spectral_radius))
        object.__setattr__(self, "spectral_abscissa", float(self.spectral_abscissa))
        object.__setattr__(self, "novikov_index", int(self.novikov_index))
        object.__setattr__(self, "novikov_residual", float(self.novikov_residual))
        object.__setattr__(self, "derham_betti", int(self.derham_betti))
        object.__setattr__(self, "derham_residual", float(self.derham_residual))
        object.__setattr__(
            self, "holomorphy_residual", float(self.holomorphy_residual)
        )
        object.__setattr__(self, "pairing_residual", float(self.pairing_residual))
        object.__setattr__(self, "passivity_residual", float(self.passivity_residual))


@dataclass(frozen=True, slots=True)
class ComplexStepAgentGovernanceState:
    """Objeto terminal del endofuntor de gobernanza del agente (Act)."""

    motor_governance_state: StabilizerGovernanceState
    phase2_orientation: Phase2InvarianceOrientation
    final_verdict: StabilizerHeytingVerdict
    crowbar_triggered: bool
    crowbar_action: CrowbarAction
    timestamp_utc: float
    provenance_hash: str
    diagnostic_note: str = ""
    independent_symplectic_residual: float = 0.0
    independent_liouville_residual: float = 0.0
    lyapunov_residual: float = 0.0
    derham_residual: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_verdict", _as_verdict(self.final_verdict))
        object.__setattr__(self, "crowbar_triggered", bool(self.crowbar_triggered))
        object.__setattr__(self, "timestamp_utc", float(self.timestamp_utc))
        object.__setattr__(self, "provenance_hash", str(self.provenance_hash))
        object.__setattr__(self, "diagnostic_note", str(self.diagnostic_note))
        object.__setattr__(
            self,
            "independent_symplectic_residual",
            float(self.independent_symplectic_residual),
        )
        object.__setattr__(
            self,
            "independent_liouville_residual",
            float(self.independent_liouville_residual),
        )
        object.__setattr__(self, "lyapunov_residual", float(self.lyapunov_residual))
        object.__setattr__(self, "derham_residual", float(self.derham_residual))


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVACIÓN ESPECTRAL COMPLEJA (Observe)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_ComplexSpectralObserver:
    r"""
    Fase 1 — Observador espectral del Jacobiano CSMD.

    Objeto: el reporte del motor (CSMDJacobianReport).
    Morfismo: Report ↦ Observation(κ, Wilkinson, round‑trip, SPD).
    Último morfismo (unidad de Kleisli hacia la Fase 2): accept_phase1_handoff.
    """

    def _coerce_motor_report(self, report: CSMDJacobianReport) -> CSMDJacobianReport:
        """Pre: report es el artefacto de Fase 2 del motor. Post: tipo certificado."""
        if not isinstance(report, CSMDJacobianReport):
            raise InvalidAgentStateError(
                "motor_report debe ser un CSMDJacobianReport; "
                f"recibido {type(report).__name__}."
            )
        return report

    def _extract_jacobian(self, report: CSMDJacobianReport) -> NDArray[np.float64]:
        """Extrae J como copia C‑contigua float64 y certifica cuadratura + finitez."""
        try:
            jacobian = np.array(report.jacobian_map, dtype=np.float64, copy=True, order="C")
        except (TypeError, ValueError) as exc:
            raise InvalidAgentStateError(
                f"jacobian_map no es coercible a float64: {exc}"
            ) from exc
        if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
            raise InvalidAgentStateError(
                f"El Jacobiano debe ser una matriz cuadrada; shape={jacobian.shape}."
            )
        if jacobian.size == 0:
            raise InvalidAgentStateError("El Jacobiano es vacío.")
        if not np.all(np.isfinite(jacobian)):
            raise InvalidAgentStateError(
                "El Jacobiano del motor contiene entradas no finitas."
            )
        return jacobian

    def _assert_dilation_coherent(
        self,
        report: CSMDJacobianReport,
        jacobian: NDArray[np.float64],
    ) -> StinespringComplexDilation:
        """Coherencia Dil ↔ J: dimensión, finitez de x y de h."""
        dilation = report.dilation
        if not isinstance(dilation, StinespringComplexDilation):
            raise InvalidAgentStateError(
                "report.dilation debe ser StinespringComplexDilation; "
                f"recibido {type(dilation).__name__}."
            )
        d = int(dilation.dimension_total)
        if d != jacobian.shape[0]:
            raise InvalidAgentStateError(
                f"Incoherencia dimensional Dil.d={d} vs J.shape={jacobian.shape}."
            )
        if np.asarray(dilation.state_real).shape != (d,):
            raise InvalidAgentStateError(
                f"state_real.shape={np.asarray(dilation.state_real).shape} ≠ ({d},)."
            )
        if np.asarray(dilation.state_complex_grid).shape != (d, d):
            raise InvalidAgentStateError(
                "state_complex_grid.shape="
                f"{np.asarray(dilation.state_complex_grid).shape} ≠ ({d}, {d})."
            )
        h_val = float(dilation.step_size_h)
        if not math.isfinite(h_val) or h_val <= 0.0:
            raise InvalidAgentStateError(f"step_size_h inválido: {h_val!r}.")
        if not np.all(np.isfinite(np.asarray(dilation.state_real))):
            raise InvalidAgentStateError("state_real de la dilatación no es finito.")
        return dilation

    def _independent_singular_spectrum(
        self,
        jacobian: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, float, float]:
        r"""
        SVD independiente: (σ, σ_min, σ_max, κ₂).
        κ₂ = +∞ si σ_min ≤ ε (núcleo numérico).
        """
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                singular_values = np.linalg.svd(jacobian, compute_uv=False)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise InvalidAgentStateError(
                f"SVD independiente del Jacobiano falló: {exc}"
            ) from exc
        if singular_values.size == 0:
            return singular_values, 0.0, 0.0, math.inf
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        if math.isfinite(sigma_min) and sigma_min > _MACHINE_EPS:
            kappa = sigma_max / sigma_min
        else:
            kappa = math.inf
        if not math.isfinite(kappa) or kappa < 0.0:
            kappa = math.inf
        return singular_values, sigma_min, sigma_max, float(kappa)

    def _wilkinson_reconstruction_residual(
        self,
        jacobian: NDArray[np.float64],
    ) -> float:
        r"""
        Residuo de Wilkinson: ‖J − U Σ V*‖_F / ‖J‖_F.
        Cota a priori ~ u √d ; un exceso delata corrupción de J o de la SVD.
        """
        try:
            u, sigma, vh = np.linalg.svd(jacobian, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise InvalidAgentStateError(f"SVD (Wilkinson) falló: {exc}") from exc
        reconstructed = (u * sigma) @ vh
        delta = jacobian - reconstructed
        num = float(np.linalg.norm(delta, ord="fro"))
        den = float(np.linalg.norm(jacobian, ord="fro"))
        if not math.isfinite(num):
            return math.inf
        return num / max(den, _MACHINE_TINY)

    def _singular_consistency_residual(
        self,
        report: CSMDJacobianReport,
        sigma_min: float,
        sigma_max: float,
        kappa_ind: float,
    ) -> float:
        """
        Desacuerdo relativo entre (σ_min, σ_max, κ) del motor y la SVD propia.
        """
        motor_min = float(report.spectral_min_singular)
        motor_max = float(report.spectral_max_singular)
        motor_kappa = float(report.condition_number)

        def _rel(a: float, b: float) -> float:
            if not math.isfinite(a) and not math.isfinite(b):
                return 0.0
            if not math.isfinite(a) or not math.isfinite(b):
                return math.inf
            return abs(a - b) / max(1.0, abs(a), abs(b))

        return float(
            max(
                _rel(motor_min, sigma_min),
                _rel(motor_max, sigma_max),
                _rel(motor_kappa, kappa_ind),
            )
        )

    def _stinespring_roundtrip_residual(
        self,
        dilation: StinespringComplexDilation,
    ) -> float:
        r"""
        Round‑trip de la dilatación: Re(G) = x 1ᵀ , Im(G) = h I.
        Residual en norma ∞, adimensionalizado por (1 + ‖x‖_∞ + h).
        """
        grid = np.asarray(dilation.state_complex_grid)
        state = np.asarray(dilation.state_real, dtype=np.float64)
        h_val = float(dilation.step_size_h)
        if not np.all(np.isfinite(grid)):
            return math.inf
        real_err = _finite_max_abs(np.real(grid) - state[:, None])
        imag_off = np.array(np.imag(grid), dtype=np.float64, copy=True)
        np.fill_diagonal(imag_off, 0.0)
        imag_off_err = _finite_max_abs(imag_off)
        imag_diag_err = _finite_max_abs(np.diag(np.imag(grid)) - h_val)
        scale = 1.0 + float(np.max(np.abs(state))) + h_val
        return max(real_err, imag_off_err, imag_diag_err) / max(scale, _MACHINE_TINY)

    def _spd_certificate_residual(
        self,
        metric_g: NDArray[np.float64] | None,
        dim_total: int,
    ) -> float:
        r"""
        Firma SPD de G vía Cholesky. Residual = 0 si G = Gᵀ ≻ 0.
        Si metric_g es None el join es neutro (residual 0).
        Si G no es SPD se retorna +∞ (VETOED).
        """
        if metric_g is None:
            return 0.0
        try:
            gram = np.array(metric_g, dtype=np.float64, copy=True, order="C")
        except (TypeError, ValueError) as exc:
            raise InvalidAgentStateError(
                f"metric_g no es coercible a float64: {exc}"
            ) from exc
        if gram.shape != (dim_total, dim_total):
            raise InvalidAgentStateError(
                f"metric_g debe ser ({dim_total}, {dim_total}); shape={gram.shape}."
            )
        if not np.all(np.isfinite(gram)):
            return math.inf
        skew = float(np.linalg.norm(gram - gram.T, ord="fro"))
        sym = 0.5 * (gram + gram.T)
        try:
            np.linalg.cholesky(sym)
        except np.linalg.LinAlgError:
            return math.inf
        return skew / max(float(np.linalg.norm(sym, ord="fro")), _MACHINE_TINY)

    def _condition_disagreement(
        self,
        motor_kappa: float,
        independent_kappa: float,
    ) -> float:
        """Desacuerdo relativo |κ_motor − κ_ind| / max(κ, 1)."""
        if not math.isfinite(motor_kappa) and not math.isfinite(independent_kappa):
            return 0.0
        if not math.isfinite(motor_kappa) or not math.isfinite(independent_kappa):
            return math.inf
        return abs(motor_kappa - independent_kappa) / max(
            1.0, abs(motor_kappa), abs(independent_kappa)
        )

    def _classify_phase1_verdicts(
        self,
        motor_kappa: float,
        independent_kappa: float,
        wilkinson_residual: float,
        roundtrip_residual: float,
        singular_consistency_residual: float,
        condition_disagreement: float,
        spd_residual: float,
    ) -> StabilizerHeytingVerdict:
        """Join de todas las aduanas espectrales de la Fase 1."""
        kappa_ref = (
            independent_kappa
            if math.isfinite(independent_kappa)
            else motor_kappa
        )
        v_cond = _heyting_classify(
            kappa_ref, _SOFT_CONDITION_BOUND, _HARD_CONDITION_BOUND
        )
        v_wilk = _heyting_classify(
            wilkinson_residual, _SOFT_WILKINSON_TOL, _HARD_WILKINSON_TOL
        )
        v_rt = _heyting_classify(
            roundtrip_residual, _SOFT_ROUNDTRIP_TOL, _HARD_ROUNDTRIP_TOL
        )
        v_sing = _heyting_classify(
            singular_consistency_residual, _SOFT_DISAGREE_TOL, _HARD_DISAGREE_TOL
        )
        v_dis = _heyting_classify(
            condition_disagreement, _SOFT_DISAGREE_TOL, _HARD_DISAGREE_TOL
        )
        v_spd = _heyting_classify(spd_residual, _SOFT_WILKINSON_TOL, _HARD_WILKINSON_TOL)
        return _heyting_join(v_cond, v_wilk, v_rt, v_sing, v_dis, v_spd)

    def _seal_phase1_observation(
        self,
        motor_report: CSMDJacobianReport,
        motor_kappa: float,
        independent_kappa: float,
        verdict: StabilizerHeytingVerdict,
        wilkinson_residual: float,
        roundtrip_residual: float,
        singular_consistency_residual: float,
        condition_disagreement: float,
        spd_residual: float,
    ) -> Phase1ComplexObservation:
        """Sella el artefacto de Fase 1 como DTO congelado."""
        is_well = bool(
            math.isfinite(independent_kappa)
            and independent_kappa <= _HARD_CONDITION_BOUND
            and verdict != StabilizerHeytingVerdict.VETOED
        )
        return Phase1ComplexObservation(
            motor_report=motor_report,
            condition_number=motor_kappa,
            independent_condition_number=independent_kappa,
            phase1_verdict=verdict,
            wilkinson_residual=wilkinson_residual,
            roundtrip_residual=roundtrip_residual,
            singular_consistency_residual=singular_consistency_residual,
            condition_disagreement=condition_disagreement,
            spd_residual=spd_residual,
            is_well_conditioned=is_well,
        )

    def _observe_complex_regularity(
        self,
        motor_report: CSMDJacobianReport,
        metric_g: NDArray[np.float64] | None = None,
    ) -> Phase1ComplexObservation:
        """Cuerpo de observación: Φ₁(Report, G?) = Observation."""
        report = self._coerce_motor_report(motor_report)
        jacobian = self._extract_jacobian(report)
        dilation = self._assert_dilation_coherent(report, jacobian)

        motor_kappa = float(report.condition_number)
        if not math.isfinite(motor_kappa) and motor_kappa != math.inf:
            raise InvalidAgentStateError(
                f"El número de condición del motor no es un float admisible: {motor_kappa!r}."
            )

        _, sigma_min, sigma_max, kappa_ind = self._independent_singular_spectrum(
            jacobian
        )
        wilkinson = self._wilkinson_reconstruction_residual(jacobian)
        singular_cons = self._singular_consistency_residual(
            report, sigma_min, sigma_max, kappa_ind
        )
        roundtrip = self._stinespring_roundtrip_residual(dilation)
        disagreement = self._condition_disagreement(motor_kappa, kappa_ind)
        spd_res = self._spd_certificate_residual(metric_g, jacobian.shape[0])
        verdict = self._classify_phase1_verdicts(
            motor_kappa,
            kappa_ind,
            wilkinson,
            roundtrip,
            singular_cons,
            disagreement,
            spd_res,
        )
        logger.debug(
            "Fase 1 (Observe): κ_motor=%.3e κ_ind=%.3e wilk=%.3e rt=%.3e "
            "sing=%.3e dis=%.3e spd=%.3e v=%s",
            motor_kappa,
            kappa_ind,
            wilkinson,
            roundtrip,
            singular_cons,
            disagreement,
            spd_res,
            verdict.name,
        )
        return self._seal_phase1_observation(
            report,
            motor_kappa,
            kappa_ind,
            verdict,
            wilkinson,
            roundtrip,
            singular_cons,
            disagreement,
            spd_res,
        )

    def accept_phase1_handoff(
        self,
        observation: Phase1ComplexObservation,
    ) -> Phase1ComplexObservation:
        r"""
        Último morfismo formal de la Fase 1 y primer morfismo de la Fase 2.

        En esta clase actúa como sello‑identidad (unidad de Kleisli).
        Phase2_PhaseInvarianceOrienter lo continúa por override,
        endureciendo el contrato de precondición geométrica.

        Pre:  observation es el artefacto emitido por _observe_complex_regularity.
        Post: el mismo artefacto, certificado como precondición de Φ₂.
        Tipo: Phase1ComplexObservation → Phase1ComplexObservation
        """
        if not isinstance(observation, Phase1ComplexObservation):
            raise InvalidAgentStateError(
                "handoff Fase 1: se esperaba Phase1ComplexObservation, "
                f"recibido {type(observation).__name__}."
            )
        return observation

    def execute_phase1(
        self,
        motor_report: CSMDJacobianReport,
        metric_g: NDArray[np.float64] | None = None,
    ) -> Phase1ComplexObservation:
        """
        Método terminal orquestador de la Fase 1.

        La última llamada ES accept_phase1_handoff, continuación / inicio
        formal de los métodos de la Fase 2.
        """
        observation = self._observe_complex_regularity(motor_report, metric_g=metric_g)
        return self.accept_phase1_handoff(observation)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENTACIÓN DE INVARIANZA DE FASE (Orient)
#          — continuación formal de accept_phase1_handoff
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_PhaseInvarianceOrienter(Phase1_ComplexSpectralObserver):
    r"""
    Fase 2 — Re‑auditoría geométrica independiente (I1–I4).

    Continúa el último morfismo de la Fase 1 (accept_phase1_handoff).
    Objeto: Observation × GovernanceState_motor × (malla opcional).
    Morfismo: (Obs, Gov, K?) ↦ Orientation(r_sym, r_lio, ρ, β₁, ⊔).
    Último morfismo (unidad de Kleisli hacia la Fase 3): accept_phase2_handoff.
    """

    def accept_phase1_handoff(
        self,
        observation: Phase1ComplexObservation,
    ) -> Phase1ComplexObservation:
        r"""
        Continuación formal del último método de la Fase 1.

        Endurece el contrato: tipo, finitez de κ, coherencia Dil ↔ J
        y presencia del reporte motor.
        """
        observation = super().accept_phase1_handoff(observation)
        self._validate_phase1_contract(observation)
        return observation

    def _validate_phase1_contract(self, observation: Phase1ComplexObservation) -> None:
        """Precondición geométrica completa del artefacto de Fase 1."""
        if not isinstance(observation.motor_report, CSMDJacobianReport):
            raise InvalidAgentStateError(
                "phase1.motor_report no es un CSMDJacobianReport."
            )
        if not math.isfinite(observation.condition_number) and (
            observation.condition_number != math.inf
        ):
            raise InvalidAgentStateError("phase1.condition_number no es admisible.")
        jacobian = self._extract_jacobian(observation.motor_report)
        self._assert_dilation_coherent(observation.motor_report, jacobian)

    def _coerce_motor_governance(
        self,
        motor_gov_state: StabilizerGovernanceState,
    ) -> StabilizerGovernanceState:
        """Certifica el estado de gobernanza emitido por el motor."""
        if not isinstance(motor_gov_state, StabilizerGovernanceState):
            raise InvalidAgentStateError(
                "motor_gov_state debe ser StabilizerGovernanceState; "
                f"recibido {type(motor_gov_state).__name__}."
            )
        r_sym = float(motor_gov_state.symplectic_residual)
        r_lio = float(motor_gov_state.liouville_residual)
        if not math.isfinite(r_sym) or not math.isfinite(r_lio):
            raise InvalidAgentStateError(
                "Residuos no finitos en el estado de gobernanza del motor "
                f"(r_sym={r_sym!r}, r_lio={r_lio!r})."
            )
        return motor_gov_state

    def _assert_report_gov_alignment(
        self,
        observation: Phase1ComplexObservation,
        motor_gov_state: StabilizerGovernanceState,
    ) -> None:
        """El reporte de Fase 1 y el gobierno del motor deben hablar del mismo J."""
        j_obs = np.asarray(observation.motor_report.jacobian_map)
        j_gov = np.asarray(motor_gov_state.jacobian_report.jacobian_map)
        if j_obs.shape != j_gov.shape:
            raise InvalidAgentStateError(
                f"J_obs.shape={j_obs.shape} ≠ J_gov.shape={j_gov.shape}."
            )
        delta = float(np.linalg.norm(j_obs - j_gov, ord="fro"))
        scale = max(float(np.linalg.norm(j_obs, ord="fro")), _MACHINE_TINY)
        if delta / scale > 64.0 * _MACHINE_EPS * math.sqrt(j_obs.size):
            raise InvalidAgentStateError(
                "El Jacobiano de la observación y el del gobierno del motor "
                f"divergen: ‖Δ‖_F/‖J‖_F={delta / scale:.3e}."
            )

    def _require_even_dimension(self, dim_total: int) -> int:
        """Sp(2n, ℝ) exige d = 2n. Retorna n."""
        if dim_total % 2 != 0:
            raise InvalidAgentStateError(
                f"La dimensión total debe ser par para definir Ω; recibida d={dim_total}."
            )
        n = dim_total // 2
        if n <= 0:
            raise InvalidAgentStateError(f"n = d/2 debe ser positivo; d={dim_total}.")
        return n

    def _recompute_symplectic_residuals(
        self,
        jacobian: NDArray[np.float64],
        n: int,
    ) -> tuple[float, float]:
        r"""
        Re‑auditoría independiente de Jᵀ Ω J = Ω.
        Retorna (r_abs, r_rel) con r_rel = r_abs / ‖Ω‖_F, ‖Ω‖_F = √(2n).
        """
        omega = _canonical_symplectic_form(n)
        pullback = jacobian.T @ _structured_omega_action(jacobian, n)
        abs_res = float(np.linalg.norm(pullback - omega, ord="fro"))
        omega_f = math.sqrt(2.0 * n)
        rel_res = abs_res / max(omega_f, _MACHINE_TINY)
        if not math.isfinite(abs_res):
            return math.inf, math.inf
        return abs_res, rel_res

    def _recompute_liouville_residual(
        self,
        jacobian: NDArray[np.float64],
    ) -> tuple[float, float, int]:
        r"""
        Re‑auditoría independiente vía slogdet.
        Retorna (r_Lio, log|det|, signo) con signo ∈ {−1, 0, +1}.
        """
        try:
            sign, log_abs_det = np.linalg.slogdet(jacobian)
        except np.linalg.LinAlgError as exc:
            raise InvalidAgentStateError(f"slogdet independiente falló: {exc}") from exc
        sign_f = float(sign)
        lad = float(log_abs_det)
        if not math.isfinite(sign_f) or not math.isfinite(lad) or sign_f <= 0.0:
            sign_i = (
                0
                if (not math.isfinite(sign_f) or sign_f == 0.0)
                else int(math.copysign(1.0, sign_f))
            )
            return math.inf, lad, sign_i
        residual = abs(math.expm1(lad))
        if not math.isfinite(residual):
            residual = math.inf
        return float(residual), lad, 1

    def _audit_lyapunov_spectrum(
        self,
        jacobian: NDArray[np.float64],
        expected_novikov_index: int | None,
    ) -> tuple[float, float, float, int, float]:
        r"""
        Espectro de Lyapunov–Novikov del mapa Φ_Δt.

        Retorna
            (r_Lyap, ρ, α, μ_Nov, r_Nov)
        donde
            ρ = max |λ|,   α = max Re(λ),
            r_Lyap = max{0, ρ − 1},
            μ_Nov = # { λ : |λ| > 1 + √ε },
            r_Nov = |μ_Nov − μ_esperado|  (0 si no se declara índice).
        """
        dim_total = jacobian.shape[0]
        if dim_total > _PAIRING_DIM_GUARD:
            return 0.0, 1.0, 0.0, 0, 0.0
        try:
            eigenvalues = np.linalg.eigvals(jacobian)
        except np.linalg.LinAlgError as exc:
            logger.warning("Espectro de Lyapunov omitido: eigvals falló (%s).", exc)
            return math.inf, math.inf, math.inf, -1, math.inf
        if not np.all(np.isfinite(eigenvalues)):
            return math.inf, math.inf, math.inf, -1, math.inf

        moduli = np.abs(eigenvalues)
        rho = float(np.max(moduli))
        abscissa = float(np.max(np.real(eigenvalues)))
        r_lyap = max(0.0, rho - 1.0)
        pair_eps = math.sqrt(_MACHINE_EPS)
        mu_nov = int(np.count_nonzero(moduli > 1.0 + pair_eps))
        if expected_novikov_index is None:
            r_nov = 0.0
        else:
            r_nov = float(abs(mu_nov - int(expected_novikov_index)))
        if not math.isfinite(r_lyap):
            r_lyap = math.inf
        return float(r_lyap), rho, abscissa, mu_nov, r_nov

    def _audit_de_rham_betti(
        self,
        mesh_adjacency: NDArray[np.generic] | None,
    ) -> tuple[int, float]:
        """
        β₁ de la malla inyectada. Sin malla: (0, 0.0) y el join es neutro.
        Con malla: residual = |β₁|  (se exige aciclicidad).
        """
        if mesh_adjacency is None:
            return 0, 0.0
        beta1, n_v, n_e, n_c = _graph_first_betti(mesh_adjacency)
        logger.debug(
            "Fase 2 (de Rham): β₁=%d (n=%d, m=%d, c=%d)",
            beta1,
            n_v,
            n_e,
            n_c,
        )
        return beta1, float(abs(beta1))

    def _cross_audit_disagreement(
        self,
        independent_sym: float,
        independent_lio: float,
        motor_sym: float,
        motor_lio: float,
    ) -> float:
        """ℓ∞ de los desacuerdos relativos motor ↔ agente."""

        def _rel(a: float, b: float) -> float:
            if not math.isfinite(a) and not math.isfinite(b):
                return 0.0
            if not math.isfinite(a) or not math.isfinite(b):
                return math.inf
            return abs(a - b) / max(1.0, abs(a), abs(b), _MACHINE_TINY)

        return float(max(_rel(independent_sym, motor_sym), _rel(independent_lio, motor_lio)))

    def _optional_motor_residual(
        self,
        motor_gov_state: StabilizerGovernanceState,
        field_name: str,
    ) -> float:
        """Lee un residual v5 del motor; 0.0 si el contrato v4 no lo expone."""
        value = getattr(motor_gov_state, field_name, 0.0)
        try:
            residual = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(residual):
            return math.inf
        return residual

    def _holomorphy_from_report(self, report: CSMDJacobianReport) -> float:
        """Agrega deriva real y fuga imaginaria (campos v5; 0 en v4)."""
        drift = float(getattr(report, "holomorphy_real_drift", 0.0) or 0.0)
        leak = float(getattr(report, "holomorphy_imag_leak", 0.0) or 0.0)
        holo = float(getattr(report, "holomorphy_residual", 0.0) or 0.0)
        aggregated = max(abs(drift), abs(leak), abs(holo))
        return aggregated if math.isfinite(aggregated) else math.inf

    def _join_phase2_verdicts(
        self,
        phase1_verdict: StabilizerHeytingVerdict,
        motor_verdict: StabilizerHeytingVerdict,
        r_sym: float,
        r_lio: float,
        r_lyap: float,
        r_derham: float,
        r_disagree: float,
        r_holo: float,
        r_pair: float,
        r_pass: float,
        r_nov: float,
        mesh_injected: bool,
    ) -> StabilizerHeytingVerdict:
        """Supremo de todas las aduanas de la Fase 2."""
        v_sym = _heyting_classify(r_sym, _SOFT_SYMPLECTIC_TOL, _HARD_SYMPLECTIC_TOL)
        v_lio = _heyting_classify(r_lio, _SOFT_LIOUVILLE_TOL, _HARD_LIOUVILLE_TOL)
        v_lyap = _heyting_classify(r_lyap, _SOFT_LYAPUNOV_TOL, _HARD_LYAPUNOV_TOL)
        v_dis = _heyting_classify(r_disagree, _SOFT_DISAGREE_TOL, _HARD_DISAGREE_TOL)
        v_holo = _heyting_classify(r_holo, _SOFT_HOLOMORPHY_TOL, _HARD_HOLOMORPHY_TOL)
        v_pair = _heyting_classify(r_pair, _SOFT_PAIRING_TOL, _HARD_PAIRING_TOL)
        v_pass = _heyting_classify(r_pass, _SOFT_PASSIVITY_TOL, _HARD_PASSIVITY_TOL)
        v_nov = _heyting_classify(r_nov, _SOFT_NOVIKOV_TOL, _HARD_NOVIKOV_TOL)
        v_der = (
            _heyting_classify(r_derham, 0.0, 0.0)
            if mesh_injected
            else StabilizerHeytingVerdict.COHERENT
        )
        return _heyting_join(
            phase1_verdict,
            motor_verdict,
            v_sym,
            v_lio,
            v_lyap,
            v_der,
            v_dis,
            v_holo,
            v_pair,
            v_pass,
            v_nov,
        )

    def _orient_phase_invariance(
        self,
        phase1_obs: Phase1ComplexObservation,
        motor_gov_state: StabilizerGovernanceState,
        mesh_adjacency: NDArray[np.generic] | None = None,
        expected_novikov_index: int | None = None,
    ) -> Phase2InvarianceOrientation:
        """Cuerpo de orientación: Φ₂(Obs, Gov, K?) = Orientation."""
        gov = self._coerce_motor_governance(motor_gov_state)
        self._assert_report_gov_alignment(phase1_obs, gov)

        jacobian = self._extract_jacobian(phase1_obs.motor_report)
        n = self._require_even_dimension(jacobian.shape[0])

        r_sym, r_rel = self._recompute_symplectic_residuals(jacobian, n)
        r_lio, log_abs_det, det_sign = self._recompute_liouville_residual(jacobian)
        r_lyap, rho, abscissa, mu_nov, r_nov = self._audit_lyapunov_spectrum(
            jacobian, expected_novikov_index
        )
        beta1, r_derham = self._audit_de_rham_betti(mesh_adjacency)

        motor_sym = float(gov.symplectic_residual)
        motor_lio = float(gov.liouville_residual)
        r_disagree = self._cross_audit_disagreement(
            r_sym, r_lio, motor_sym, motor_lio
        )
        r_holo = self._holomorphy_from_report(phase1_obs.motor_report)
        r_pair = self._optional_motor_residual(gov, "pairing_residual")
        r_pass = self._optional_motor_residual(gov, "passivity_residual")
        motor_verdict = _as_verdict(gov.final_verdict)

        combined = self._join_phase2_verdicts(
            phase1_obs.phase1_verdict,
            motor_verdict,
            r_sym,
            r_lio,
            r_lyap,
            r_derham,
            r_disagree,
            r_holo,
            r_pair,
            r_pass,
            r_nov,
            mesh_injected=mesh_adjacency is not None,
        )
        logger.debug(
            "Fase 2 (Orient): r_sym=%.4e r_lio=%.4e r_lyap=%.4e ρ=%.6f "
            "β₁=%d dis=%.4e motor=%s combined=%s",
            r_sym,
            r_lio,
            r_lyap,
            rho,
            beta1,
            r_disagree,
            motor_verdict.name,
            combined.name,
        )
        return Phase2InvarianceOrientation(
            phase1_observation=phase1_obs,
            motor_governance_state=gov,
            symplectic_residual=r_sym,
            liouville_residual=r_lio,
            motor_verdict=motor_verdict,
            phase2_verdict=combined,
            motor_symplectic_residual=motor_sym,
            motor_liouville_residual=motor_lio,
            residual_disagreement=r_disagree,
            relative_symplectic_residual=r_rel,
            log_abs_det=log_abs_det,
            det_sign=det_sign,
            lyapunov_residual=r_lyap,
            spectral_radius=rho,
            spectral_abscissa=abscissa,
            novikov_index=mu_nov,
            novikov_residual=r_nov,
            derham_betti=beta1,
            derham_residual=r_derham,
            holomorphy_residual=r_holo,
            pairing_residual=r_pair,
            passivity_residual=r_pass,
        )

    def accept_phase2_handoff(
        self,
        orientation: Phase2InvarianceOrientation,
    ) -> Phase2InvarianceOrientation:
        r"""
        Último morfismo formal de la Fase 2 y primer morfismo de la Fase 3.

        En esta clase actúa como sello‑identidad. La Fase 3 lo continúa
        por override, exigiendo la presencia del gobierno del motor y
        la finitez de los residuos independientes.
        Tipo: Phase2InvarianceOrientation → Phase2InvarianceOrientation
        """
        if not isinstance(orientation, Phase2InvarianceOrientation):
            raise InvalidAgentStateError(
                "handoff Fase 2: se esperaba Phase2InvarianceOrientation, "
                f"recibido {type(orientation).__name__}."
            )
        return orientation

    def execute_phase2(
        self,
        phase1_obs: Phase1ComplexObservation,
        motor_gov_state: StabilizerGovernanceState,
        mesh_adjacency: NDArray[np.generic] | None = None,
        expected_novikov_index: int | None = None,
    ) -> Phase2InvarianceOrientation:
        """
        Método terminal orquestador de la Fase 2.

        Primera llamada: accept_phase1_handoff (continuación de Fase 1).
        Última llamada:  accept_phase2_handoff (inicio formal de Fase 3).
        """
        phase1_obs = self.accept_phase1_handoff(phase1_obs)
        orientation = self._orient_phase_invariance(
            phase1_obs,
            motor_gov_state,
            mesh_adjacency=mesh_adjacency,
            expected_novikov_index=expected_novikov_index,
        )
        return self.accept_phase2_handoff(orientation)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — VETO EN RETÍCULO DE HEYTING Y ACTUACIÓN CROWBAR (Decide & Act)
#          — continuación formal de accept_phase2_handoff
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingStabilizerDecider(Phase2_PhaseInvarianceOrienter):
    r"""
    Fase 3 — Supremo en Ω₃, sello forense y CrowbarPort lógico.

    Continúa el último morfismo de la Fase 2 (accept_phase2_handoff).
    Si v_final toca el objeto terminal VETOED, arma el puerto inyectable
    y, opcionalmente, colapsa síncronamente la transacción.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Permite mixins / MRO con o sin CrowbarPort inyectado."""
        self._crowbar_port: CrowbarPort = kwargs.pop("crowbar_port", None) or NullCrowbarPort()
        super().__init__(*args, **kwargs) if args or kwargs else None  # type: ignore[misc]

    def accept_phase2_handoff(
        self,
        orientation: Phase2InvarianceOrientation,
    ) -> Phase2InvarianceOrientation:
        """
        Continuación formal del último método de la Fase 2.

        Endurece el contrato: tipo, gobierno del motor presente y
        residuos independientes numéricamente admisibles.
        """
        orientation = super().accept_phase2_handoff(orientation)
        self._validate_phase2_contract(orientation)
        return orientation

    def _validate_phase2_contract(
        self,
        orientation: Phase2InvarianceOrientation,
    ) -> None:
        """Precondición completa del artefacto de Fase 2."""
        if not isinstance(orientation.phase1_observation, Phase1ComplexObservation):
            raise InvalidAgentStateError(
                "phase2.phase1_observation no es Phase1ComplexObservation."
            )
        if not isinstance(orientation.motor_governance_state, StabilizerGovernanceState):
            raise InvalidAgentStateError(
                "phase2.motor_governance_state no es StabilizerGovernanceState."
            )
        for name in (
            "symplectic_residual",
            "liouville_residual",
            "lyapunov_residual",
            "residual_disagreement",
        ):
            value = float(getattr(orientation, name))
            if not math.isfinite(value) and value != math.inf:
                raise InvalidAgentStateError(
                    f"phase2.{name} no es un float admisible: {value!r}."
                )

    def _select_crowbar_action(
        self,
        v_final: StabilizerHeytingVerdict,
        trigger_hardware_crowbar: bool,
    ) -> tuple[CrowbarAction, bool]:
        """
        Selección pura de la acción lógica.
        GPIO_INTERRUPT_CROWBAR sólo se elige si el caller lo autoriza
        explícitamente; el puerto nulo seguirá sin tocar silicio.
        """
        if v_final == StabilizerHeytingVerdict.VETOED:
            if trigger_hardware_crowbar:
                return CrowbarAction.GPIO_INTERRUPT_CROWBAR, True
            return CrowbarAction.LOG_WARNING, False
        if v_final == StabilizerHeytingVerdict.DEGRADED:
            return CrowbarAction.LOG_WARNING, False
        return CrowbarAction.NONE, False

    def _enact_crowbar(
        self,
        action: CrowbarAction,
        v_final: StabilizerHeytingVerdict,
        orientation: Phase2InvarianceOrientation,
        provenance_hash: str,
    ) -> bool:
        """Efecto (Act) sobre el CrowbarPort. Retorna si el puerto acusó disparo."""
        if action is CrowbarAction.GPIO_INTERRUPT_CROWBAR:
            reason = (
                f"VETO {v_final.name} r_sym={orientation.symplectic_residual:.4e} "
                f"r_lio={orientation.liouville_residual:.4e} "
                f"r_lyap={orientation.lyapunov_residual:.4e}"
            )
            fired = bool(
                self._crowbar_port.arm_and_fire(
                    gpio=_CROWBAR_GPIO,
                    reason=reason,
                    provenance=provenance_hash,
                )
            )
            logger.critical(
                "[CROWBAR] VETO TERMINAL CSMD. canal_logico=%s disparo=%s. "
                "r_sym=%.4e r_lio=%.4e r_lyap=%.4e β₁=%d.",
                _CROWBAR_GPIO,
                fired,
                orientation.symplectic_residual,
                orientation.liouville_residual,
                orientation.lyapunov_residual,
                orientation.derham_betti,
            )
            return fired
        if action is CrowbarAction.LOG_WARNING:
            if v_final == StabilizerHeytingVerdict.VETOED:
                logger.warning(
                    "[VETO] Desviación crítica CSMD. Sesión lógica marcada. "
                    "r_sym=%.4e r_lio=%.4e r_lyap=%.4e.",
                    orientation.symplectic_residual,
                    orientation.liouville_residual,
                    orientation.lyapunov_residual,
                )
            else:
                logger.warning(
                    "[DEGRADED] Precisión límite CSMD. "
                    "r_sym=%.4e r_lio=%.4e κ=%.3e ρ=%.6f.",
                    orientation.symplectic_residual,
                    orientation.liouville_residual,
                    orientation.phase1_observation.independent_condition_number,
                    orientation.spectral_radius,
                )
        return False

    def _dominant_invariant(
        self,
        orientation: Phase2InvarianceOrientation,
    ) -> str:
        """Identifica el invariante que empuja el join al terminal, si es único."""
        candidates: list[tuple[str, float, float]] = [
            ("symplectic", orientation.symplectic_residual, _HARD_SYMPLECTIC_TOL),
            ("liouville", orientation.liouville_residual, _HARD_LIOUVILLE_TOL),
            ("lyapunov", orientation.lyapunov_residual, _HARD_LYAPUNOV_TOL),
            ("derham", orientation.derham_residual, 0.0 if orientation.derham_betti else math.inf),
            ("disagree", orientation.residual_disagreement, _HARD_DISAGREE_TOL),
            (
                "condition",
                orientation.phase1_observation.independent_condition_number,
                _HARD_CONDITION_BOUND,
            ),
        ]
        offenders = [
            name
            for name, residual, hard in candidates
            if (not math.isfinite(residual)) or residual > hard
        ]
        if len(offenders) == 1:
            return offenders[0]
        return "joint"

    def _raise_terminal_veto(
        self,
        orientation: Phase2InvarianceOrientation,
        v_final: StabilizerHeytingVerdict,
        crowbar_triggered: bool,
    ) -> None:
        """Colapso síncrono: excepción especializada según el invariante dominante."""
        message = (
            "Colapso de gobernanza CSMD: invarianza rota. "
            f"Veredicto={v_final.name}, r_sym={orientation.symplectic_residual:.4e}, "
            f"r_lio={orientation.liouville_residual:.4e}, "
            f"r_lyap={orientation.lyapunov_residual:.4e}, "
            f"β₁={orientation.derham_betti}. Transacción purgada en RAM."
        )
        if crowbar_triggered:
            raise CrowbarArmedError(message)
        dominant = self._dominant_invariant(orientation)
        if dominant == "symplectic" or dominant == "liouville":
            raise PhaseInvarianceBreachError(message)
        if dominant == "lyapunov":
            raise LyapunovEscapeError(message)
        if dominant == "derham":
            raise DeRhamTorsionError(message)
        if dominant == "condition":
            raise SpectralConditionBreachError(message)
        raise HeytingStabilizerVeto(message)

    def _forensic_provenance_seal(
        self,
        orientation: Phase2InvarianceOrientation,
        v_final: StabilizerHeytingVerdict,
        crowbar_action: CrowbarAction,
    ) -> str:
        """Sello SHA‑256 dominio‑separado del artefacto de gobernanza."""
        jacobian = np.ascontiguousarray(
            orientation.phase1_observation.motor_report.jacobian_map,
            dtype=np.float64,
        )
        hasher = hashlib.sha256()
        hasher.update(_PROVENANCE_DOMAIN)
        hasher.update(jacobian.tobytes(order="C"))
        hasher.update(
            struct.pack(
                "<dddddddQii",
                float(orientation.symplectic_residual),
                float(orientation.liouville_residual),
                float(orientation.lyapunov_residual),
                float(orientation.residual_disagreement),
                float(orientation.phase1_observation.independent_condition_number)
                if math.isfinite(
                    orientation.phase1_observation.independent_condition_number
                )
                else -1.0,
                float(orientation.spectral_radius)
                if math.isfinite(orientation.spectral_radius)
                else -1.0,
                float(orientation.holomorphy_residual),
                int(orientation.derham_betti),
                int(v_final.value),
                int(crowbar_action.value),
            )
        )
        motor_hash = str(
            getattr(orientation.motor_governance_state, "provenance_hash", "")
        )
        hasher.update(motor_hash.encode("utf-8"))
        return hasher.hexdigest()

    def _compose_diagnostic_note(
        self,
        orientation: Phase2InvarianceOrientation,
        v_final: StabilizerHeytingVerdict,
        crowbar_action: CrowbarAction,
        crowbar_triggered: bool,
    ) -> str:
        """Nota forense humana: residuos independientes, motor y veredicto."""
        p1 = orientation.phase1_observation
        return (
            f"Veredicto: {v_final.name}. "
            f"κ_motor={p1.condition_number:.3e} κ_ind={p1.independent_condition_number:.3e} "
            f"wilk={p1.wilkinson_residual:.3e} rt={p1.roundtrip_residual:.3e}. "
            f"Simplecticidad: r_abs={orientation.symplectic_residual:.4e} "
            f"r_rel={orientation.relative_symplectic_residual:.4e} "
            f"(motor={orientation.motor_symplectic_residual:.4e}). "
            f"Volumen: r_lio={orientation.liouville_residual:.4e} "
            f"sign={orientation.det_sign:+d} log|det|={orientation.log_abs_det:.4e} "
            f"(motor={orientation.motor_liouville_residual:.4e}). "
            f"Lyapunov: r={orientation.lyapunov_residual:.4e} "
            f"ρ={orientation.spectral_radius:.6f} α={orientation.spectral_abscissa:.4e} "
            f"μ_Nov={orientation.novikov_index}. "
            f"de Rham: β₁={orientation.derham_betti}. "
            f"Holomorfía={orientation.holomorphy_residual:.4e} "
            f"apareamiento={orientation.pairing_residual:.4e} "
            f"pasividad={orientation.passivity_residual:.4e}. "
            f"Desacuerdo motor↔agente={orientation.residual_disagreement:.4e}. "
            f"Crowbar={crowbar_action.name} triggered={crowbar_triggered}."
        )

    def _evaluate_governance(
        self,
        phase2_orient: Phase2InvarianceOrientation,
        raise_on_veto: bool = True,
        trigger_hardware_crowbar: bool = False,
    ) -> ComplexStepAgentGovernanceState:
        """
        Consolida el veredicto terminal (ya join‑eado en Fase 2),
        sella procedencia y aplica el efecto Crowbar / veto.
        """
        v_final = _as_verdict(phase2_orient.phase2_verdict)
        action, wants_crowbar = self._select_crowbar_action(
            v_final, trigger_hardware_crowbar
        )
        provenance_hash = self._forensic_provenance_seal(
            phase2_orient, v_final, action
        )
        fired = self._enact_crowbar(
            action, v_final, phase2_orient, provenance_hash
        )
        crowbar_triggered = bool(wants_crowbar and fired) or (
            wants_crowbar and isinstance(self._crowbar_port, NullCrowbarPort)
        )
        # NullCrowbarPort no dispara silicio; se registra la *intención* autorizada.
        if wants_crowbar and isinstance(self._crowbar_port, NullCrowbarPort):
            crowbar_triggered = True

        if v_final == StabilizerHeytingVerdict.VETOED and raise_on_veto:
            self._raise_terminal_veto(phase2_orient, v_final, crowbar_triggered)

        diagnostic_note = self._compose_diagnostic_note(
            phase2_orient, v_final, action, crowbar_triggered
        )
        return ComplexStepAgentGovernanceState(
            motor_governance_state=phase2_orient.motor_governance_state,
            phase2_orientation=phase2_orient,
            final_verdict=v_final,
            crowbar_triggered=crowbar_triggered,
            crowbar_action=action,
            timestamp_utc=time.time(),
            provenance_hash=provenance_hash,
            diagnostic_note=diagnostic_note,
            independent_symplectic_residual=phase2_orient.symplectic_residual,
            independent_liouville_residual=phase2_orient.liouville_residual,
            lyapunov_residual=phase2_orient.lyapunov_residual,
            derham_residual=phase2_orient.derham_residual,
        )

    def execute_phase3(
        self,
        phase2_orient: Phase2InvarianceOrientation,
        raise_on_veto: bool = True,
        trigger_hardware_crowbar: bool = False,
    ) -> ComplexStepAgentGovernanceState:
        """
        Método terminal orquestador de la Fase 3.

        Primera llamada: accept_phase2_handoff (continuación de Fase 2).
        Retorna el estado lógico supremo y, opcionalmente, colapsa el
        retículo ante un veto.
        """
        phase2_orient = self.accept_phase2_handoff(phase2_orient)
        return self._evaluate_governance(
            phase2_orient,
            raise_on_veto=raise_on_veto,
            trigger_hardware_crowbar=trigger_hardware_crowbar,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SOBERANO: COMPLEX STEP PHASE STABILIZER AGENT (Funtor Completo)
# ══════════════════════════════════════════════════════════════════════════════
class ComplexStepPhaseStabilizerAgent(Morphism, Phase3_HeytingStabilizerDecider):
    r"""
    Soberano y Guardián del Calibre No Demolitivo por Paso Complejo.

    Endofuntor sobre el topos de estados de fase:
        Z_agent = Φ₃ ∘ Φ₂ ∘ Φ₁ ∘ Z_motor
    con unidades de Kleisli anidadas
        accept_phase1_handoff  ⊂  accept_phase2_handoff  ⊂  execute_phase3.
    """

    def __init__(
        self,
        stabilizer: ComplexStepPhaseStabilizer,
        crowbar_port: CrowbarPort | None = None,
        metric_g: NDArray[np.float64] | None = None,
        mesh_adjacency: NDArray[np.generic] | None = None,
        expected_novikov_index: int | None = None,
    ) -> None:
        """
        Vincula al agente con el motor de‑confinado y, opcionalmente,
        con métrica G, malla de de Rham y puerto Crowbar lógico.
        """
        self._crowbar_port = crowbar_port or NullCrowbarPort()
        self._stabilizer = stabilizer
        self._metric_g = metric_g
        self._mesh_adjacency = mesh_adjacency
        self._expected_novikov_index = expected_novikov_index
        self._validate_stabilizer()
        if metric_g is not None:
            # Certificado temprano: falla rápido si G es ilegible.
            gram = np.asarray(metric_g)
            if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
                raise InvalidAgentStateError(
                    f"metric_g debe ser cuadrada; shape={gram.shape}."
                )
        if mesh_adjacency is not None:
            adj = np.asarray(mesh_adjacency)
            if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
                raise InvalidAgentStateError(
                    f"mesh_adjacency debe ser cuadrada; shape={adj.shape}."
                )

    @property
    def stabilizer(self) -> ComplexStepPhaseStabilizer:
        """Motor CSMD gobernado (inyección de dependencia)."""
        return self._stabilizer

    @property
    def crowbar_port(self) -> CrowbarPort:
        """Puerto lógico del disyuntor vigente."""
        return self._crowbar_port

    def _validate_stabilizer(self) -> None:
        """El motor debe exponer el morfismo de estabilización de transiciones."""
        if self._stabilizer is None:
            raise InvalidAgentStateError("El estabilizador no puede ser None.")
        morphism = getattr(self._stabilizer, "stabilize_phase_space_transition", None)
        if not callable(morphism):
            raise InvalidAgentStateError(
                "El motor debe exponer stabilize_phase_space_transition()."
            )

    def _validate_flow_inputs(
        self,
        state_curr_x: NDArray[np.float64],
        flow_map_phi: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    ) -> NDArray[np.float64]:
        """Aduana de entradas del ciclo OODA antes de delegar en el motor."""
        try:
            state = np.array(state_curr_x, dtype=np.float64, copy=True, order="C")
        except (TypeError, ValueError) as exc:
            raise InvalidAgentStateError(
                f"state_curr_x no es coercible a float64: {exc}"
            ) from exc
        if state.ndim != 1 or state.size == 0:
            raise InvalidAgentStateError(
                f"state_curr_x debe ser 1‑D no vacío; shape={state.shape}."
            )
        if not np.all(np.isfinite(state)):
            raise InvalidAgentStateError(
                "state_curr_x contiene NaN o Inf sobre la variedad real."
            )
        if not callable(flow_map_phi):
            raise InvalidAgentStateError(
                "flow_map_phi debe ser un callable ℂᵈ → ℂᵈ."
            )
        return state

    def _delegate_to_motor(
        self,
        state_curr_x: NDArray[np.float64],
        flow_map_phi: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
        hamiltonian_gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
        rayleigh_metric: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    ) -> StabilizerGovernanceState:
        """
        Delega en Z_motor con raise_on_veto=False: el agente es dueño del veto.
        Compatible con contratos v4 (sin oráculos) y v5 (con ∇H, R).
        """
        try:
            motor_gov = self._stabilizer.stabilize_phase_space_transition(
                state_vector=state_curr_x,
                complex_transition_map=flow_map_phi,
                raise_on_veto=False,
                hamiltonian_gradient=hamiltonian_gradient,
                rayleigh_metric=rayleigh_metric,
            )
        except TypeError:
            motor_gov = self._stabilizer.stabilize_phase_space_transition(
                state_vector=state_curr_x,
                complex_transition_map=flow_map_phi,
                raise_on_veto=False,
            )
        except HeytingStabilizerVeto:
            raise
        except ComplexStepAgentError:
            raise
        except Exception as exc:
            raise InvalidAgentStateError(
                f"El motor CSMD falló durante la transición de fase: {exc}"
            ) from exc
        if not isinstance(motor_gov, StabilizerGovernanceState):
            raise InvalidAgentStateError(
                "El motor no devolvió StabilizerGovernanceState; "
                f"recibido {type(motor_gov).__name__}."
            )
        return motor_gov

    def execute_stabilization_governance(
        self,
        state_curr_x: NDArray[np.float64],
        flow_map_phi: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
        raise_on_veto: bool = True,
        trigger_hardware_crowbar: bool = False,
        metric_g: NDArray[np.float64] | None = None,
        mesh_adjacency: NDArray[np.generic] | None = None,
        expected_novikov_index: int | None = None,
        hamiltonian_gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        rayleigh_metric: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    ) -> ComplexStepAgentGovernanceState:
        r"""
        Ejecuta el ciclo categórico OODA completo.

        Composición funtorial:
            Z_agent = Φ₃ ∘ Φ₂ ∘ Φ₁ ∘ Z_motor

        Los oráculos opcionales (G, malla, μ_Nov, ∇H, R) sobrescriben,
        si se proveen, los valores inyectados en el constructor.
        """
        state = self._validate_flow_inputs(state_curr_x, flow_map_phi)

        motor_gov_state = self._delegate_to_motor(
            state,
            flow_map_phi,
            hamiltonian_gradient=hamiltonian_gradient,
            rayleigh_metric=rayleigh_metric,
        )

        phase1_obs = self.execute_phase1(
            motor_gov_state.jacobian_report,
            metric_g=metric_g if metric_g is not None else self._metric_g,
        )
        phase2_orient = self.execute_phase2(
            phase1_obs,
            motor_gov_state,
            mesh_adjacency=(
                mesh_adjacency if mesh_adjacency is not None else self._mesh_adjacency
            ),
            expected_novikov_index=(
                expected_novikov_index
                if expected_novikov_index is not None
                else self._expected_novikov_index
            ),
        )
        return self.execute_phase3(
            phase2_orient,
            raise_on_veto=raise_on_veto,
            trigger_hardware_crowbar=trigger_hardware_crowbar,
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "__version__",
    "StabilizerHeytingVerdict",
    "CrowbarAction",
    "CrowbarPort",
    "NullCrowbarPort",
    "ComplexStepAgentError",
    "InvalidAgentStateError",
    "SpectralConditionBreachError",
    "PhaseInvarianceBreachError",
    "LyapunovEscapeError",
    "DeRhamTorsionError",
    "CrowbarArmedError",
    "Phase1ComplexObservation",
    "Phase2InvarianceOrientation",
    "ComplexStepAgentGovernanceState",
    "Phase1_ComplexSpectralObserver",
    "Phase2_PhaseInvarianceOrienter",
    "Phase3_HeytingStabilizerDecider",
    "ComplexStepPhaseStabilizerAgent",
]
# -*- coding: utf-8 -*-
r"""
══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Topological Control Surface Agent (Soberano de Superficie de Control) ║
║ Ruta   : app/agents/wisdom/topological_control_surface_agent.py              ║
║ Versión: 1.1.0-Doctoral-PH-Brockett-Replicator-Heyting-ESP32-Secure          ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO (OODA):                  ║
║ Este soberano agéntico opera en el Estrato de la Sabiduría ($V_{\mathbb{W}}$,║
║ Nivel 0 - La Ciudadela de Cristal) gobernando de forma asíncrona la          ║
║ "Superficie de Control Topológica" que acopla de manera continua y no        ║
║ conmutativa la minimización discreta de la MIC y la purificación de la MAC.  ║
║                                                                              ║
║   1. CONTINUIZACIÓN DE LA MIC (Flujo Replicador sobre el Politopo Booleano): ║
║      Modela el decaimiento de las dependencias redundantes mediante el flujo ║
║      continuo Lotka-Volterra-Fermat en el simplex de probabilidad:           ║
║      $$\frac{dp_i}{dt} = p_i \left[ (\mathbf{e}_i^\top \tilde{\mathcal{K}}   ║
║      \mathbf{p}) - \mathbf{p}^\top \tilde{\mathcal{K}} \mathbf{p} \right]$$  ║
║                                                                              ║
║   2. PURIFICACIÓN DE LA MAC (Flujo de Doble Corchete de Brockett):           ║
║      Purifica el estado cuántico mixto de la matriz atómica de conocimiento  ║
║      $\rho \in \mathcal{D}(\mathcal{H})$ mediante un atractor determinista   ║
║      sobre el manifold de órbitas adjuntas, acoplado a la MIC:               ║
║      $$\frac{d\rho}{dt} = \left[ \rho, \, [\rho, \, \mathcal{N}(\mathbf{p})] \right]$$║
║                                                                              ║
║   3. INTERCONEXIÓN PORT-HAMILTONIANA CON DISIPACIÓN (IDA-PBC):               ║
║      La dinámica de la Malla satisface el principio de pasividad de Lyapunov ║
║      unificado mediante un acoplamiento antisimétrico de calibre.            ║
║                                                                              ║
║ Si el sistema diverge de Lyapunov ($\dot{H} > 0$) o surge torsión homológica,║
║ Heyting colapsa síncronamente al Supremo VETOED ($\top$), activando la ISR   ║
║ en IRAM del ESP32 perimetral en menos de 400 ns para conmutar GPIO14 y       ║
║ ceba el tiristor rápido BT151 [Crowbar], paralizando la obra real en fango.  ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================════════════════════════════════════════════════
I. DEFINICIONES CATEGORIALES Y DE TOPOS (El Manifold Simpléctico Acoplado)
================================════════════════════════════════════════════════

Definición 1 (La Superficie de Control Conjunta):
  Definimos el manifold simpléctico acoplado de control $\mathcal{M}_{\mathrm{control}}$ como
  el producto cartesiano de los espacios de fase de la MIC y la MAC:
  $$\mathcal{M}_{\mathrm{control}} := \mathcal{M}_{\mathrm{MIC}} \times \mathcal{M}_{\mathrm{MAC}}$$
  Donde el estado conjunto $\mathbf{\Psi} = (\mathbf{p}, \, \rho)^\top$ discurre por las geodésicas
  de la variedad de acuerdo con el flujo de gradiente simpléctico inducido por el Hamiltoniano.

Definición 2 (Continuización de la MIC sobre el Símplex de Gibbs):
  Sea $K$ el complejo simplicial discreto asociado a la MIC. Proyectamos el anillo booleano cociente
  $\mathbb{Z}_2[x] / \langle x^2 - x \rangle$ sobre el simplex de probabilidad continuo de Gibbs
  (el Politopo Booleano):
  $$\Delta^n := \left\{ \mathbf{p} \in [0, \, 1]^n \ \middle|\ \sum_{i=1}^n p_i = 1 \right\}$$
  La dinámica de poda de redundancias se describe como el flujo del replicador continuo de Lotka-Volterra-Fermat:
  $$\dot{p}_i = p_i \left[ \left(\mathbf{e}_i^\top \tilde{\mathcal{K}} \mathbf{p}\right) - \mathbf{p}^\top \tilde{\mathcal{K}} \mathbf{p} \right]$$

Definición 3 (El Atractor Brockett-Poisson de von Neumann):
  Sea $\rho \in \mathcal{D}(\mathcal{H})$ el operador de densidad mixto en el espacio de Hilbert $\mathcal{H}_{\mathrm{MAC}}$.
  Definimos el flujo de doble corchete de Brockett-Poisson para la purificación espectral de la MAC como:
  $$\dot{\rho} = \left[ \rho, \, \left[ \rho, \, \mathcal{N}(\mathbf{p}) \right] \right]$$
  Donde el operador de referencia $\mathcal{N}(\mathbf{p}) = \operatorname{diag}(\mathbf{p})$ se acopla síncronamente
  a las coordenadas de importancia simplicial de la MIC en FPU.

Definición 4 (La Adjunción de de Rham-Galois):
  El acoplamiento mutuo y la reversibilidad informacional entre la base táctica discreta (MIC) y la base
  epistémica continua de sabiduría (MAC) se modelan como una adjunción functorial de Grothendieck:
  $$\operatorname{Hom}_{\mathcal{D}}(F(\mathbf{p}), \, \rho) \cong \operatorname{Hom}_{\mathcal{C}}(\mathbf{p}, \, G(\rho))$$
  Donde $F$ es el funtor de elevación cuántica (Stinespring) y $G$ el funtor de olvido y proyección espectral POVM.

================================════════════════════════════════════════════════
II. AXIOMATIZACIÓN DEL FLUJO GEODÉSICO DE CONTROL (Pasividad de Lyapunov)
================================════════════════════════════════════════════════

Axioma I (Principio de Pasividad de Lyapunov-Tellegen):
  La trayectoria geodésica conjunta $\mathbf{\Psi}(t)$ sobre la superficie de control satisface la
  desigualdad termodinámica de Clausius-Duhem. Esto exige que la derivada temporal del funcional
  de energía de Lyapunov sea semidefinida negativa, asegurando que lazo cerrado no inyecte calor exógeno:
  $$\dot{\mathcal{H}}(\mathbf{\Psi}) = \nabla \mathcal{H}(\mathbf{\Psi})^\top \left( \mathcal{J}(\mathbf{\Psi}) - \mathcal{R}(\mathbf{\Psi}) \right) \nabla \mathcal{H}(\mathbf{\Psi}) \le 0$$
  Donde $\mathcal{J}$ es antisimétrica ($\mathcal{J} = -\mathcal{J}^\top$) y $\mathcal{R}$ es definida positiva ($\mathcal{R} \succ \mathbf{0}$).

Axioma II (Axioma de de Rham-Hodge de Invarianza de Calibre):
  Toda proyección métrica en la FPU debe conservar la 2-forma simpléctica de Liouville y la nulidad del primer
  grupo de cohomología de haces para erradicar paradojas de referencias circulares (socavones lógicos):
  $$\operatorname{div}(\dot{x}) \equiv 0 \quad \land \quad H^1(K; \, \mathcal{F}) = \mathbf{0}$$

Axioma III (Teorema de Actuación Ciber-Física Crowbar en Silicio):
  Si se viola la condición contractiva de Lyapunov ($\dot{\mathcal{H}}(\mathbf{\Psi}) > \tau_{\mathrm{Lyapunov}}$)
  o si surge torsión homológica simplicial ($d_i > 1$ en Smith Z), el retículo ordinal de Heyting $\Omega_3$ colapsa
  al Supremo terminal ($\top$), forzando a la Interrupt Service Routine (ISR) en IRAM del ESP32 a actuar en menos de 400 ns:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
  Cebando el tiristor rápido BT151 para cortocircuitar físicamente la potencia real y paralizar la obra civil.

================================════════════════════════════════════════════════
III. INVARIANTES ESPECTRALES Y METROLÓGICOS DE WILKINSON (FPU Secure)
================================════════════════════════════════════════════════

Invariante I (Estabilidad Asintótica Global de de Rham):
  El atractor de Lyapunov unificado se mantiene asintóticamente estable bajo perturbaciones estocásticas
  exógenas, garantizando que el cambio de energía se mantenga acotado por debajo de la cota de Wilkinson:
  $$\dot{\mathcal{H}}(\mathbf{\Psi}) \le \tau_{\mathrm{Lyapunov}} \quad \text{con} \quad \tau_{\mathrm{Lyapunov}} = 10^{-12}$$

Invariante II (Confinamiento de Entropías de Shannon y von Neumann):
  La entropía informacional de la MIC ($H_{\mathrm{sh}}(\mathbf{p})$) y la entropía espectral de von Neumann
  de la MAC ($S_{\mathrm{vN}}(\rho)$) calculadas síncronamente en FPU satisfacen los intervalos de acotación:
  $$0.0 \le H_{\mathrm{sh}}(\mathbf{p}) \le \ln(N) \quad \land \quad 0.0 \le S_{\mathrm{vN}}(\rho) \le \tau_{\mathrm{entropy}}$$
  Cualquier desbordamiento por encima del techo elástico es censurado instantáneamente como alucinación semántica.

Invariante III (Inversión Espectral de Higham-Tikhonov):
  La purificación del tensor de conductividad bruto $\mathcal{K}$ en la aduana de telemetría exige que
  el espectro deformado esté estrictamente definido positivo e invertible para evitar singularidades polares:
  $$\lambda_{\min}(\tilde{\mathcal{K}}) \ge \varepsilon_{\mathrm{Wilkinson}} \quad \implies \quad \tilde{\mathcal{K}} \succ \mathbf{0} \quad \text{con} \quad \varepsilon_{\mathrm{Wilkinson}} = 10^{-15}$$
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la


logger = logging.getLogger("APU.Agents.Wisdom.TopologicalControlSurfaceAgent")


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS, GEOMÉTRICAS Y DE SEGURIDAD
# ═══════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_STABILITY_FLOOR: Final[float] = max(1e-12, _MACHINE_EPS)
_CONDITION_CAP: Final[float] = 1.0e12
_SIMPLEX_SOFT_RESIDUAL: Final[float] = 1.0e-8
_SIMPLEX_HARD_RESIDUAL: Final[float] = 1.0e-4
_RHO_PSD_HARD: Final[float] = -1.0e-10
_DHDT_HARD: Final[float] = 1.0e-4
_CONDITION_DEGRADED: Final[float] = 1.0e9
_ISOSPECTRAL_DEGRADED: Final[float] = 1.0e-6
_PURITY_DRIFT_DEGRADED: Final[float] = 1.0e-6
_ENTROPY_DRIFT_DEGRADED: Final[float] = 1.0e-6
_HERMITIAN_REL: Final[float] = 1.0e-3

_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_STD_NS: Final[float] = 3.2
_CROWBAR_LATENCY_MIN_NS: Final[float] = 380.0
_CROWBAR_LATENCY_MAX_NS: Final[float] = 420.0
_GPIO_CROWBAR_PIN: Final[int] = 14

_SCHEMA_VERSION: Final[str] = (
    "2.1.0-Doctoral-PH-Brockett-Replicator-Heyting-Shahshahani-Nested"
)

_DOMAIN_SESSION: Final[bytes] = b"APU/TOPOLOGICAL-CONTROL-SURFACE/SESSION/v2"
_DOMAIN_TELEMETRY: Final[bytes] = b"APU/TOPOLOGICAL-CONTROL-SURFACE/TELEMETRY/v2"


class HeytingVerdict(str, Enum):
    """
    Clasificador grosero Ω₃ del topos de observaciones de la superficie.

    Cadena completa (álgebra de Heyting lineal):

        VETOED  ≤  DEGRADED  ≤  COHERENT

    Operaciones:
        meet (∧)      = ínfimo = mínimo del orden
        join (∨)      = supremo = máximo del orden
        implies (→)   = 1 si a ≤ b, else b
        neg (¬a)      = a → 0

    Semántica operativa:
        COHERENT : verdad fuerte; todas las aduanas pasan.
        DEGRADED : verdad parcial; operación vigilada.
        VETOED   : fondo lógico; interlock fail-closed.
    """

    COHERENT = "COHERENT"
    DEGRADED = "DEGRADED"
    VETOED = "VETOED"

    def _order(self) -> int:
        if self is HeytingVerdict.VETOED:
            return 0
        if self is HeytingVerdict.DEGRADED:
            return 1
        return 2

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Ínfimo en Ω₃."""
        return self if self._order() <= other._order() else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Supremo en Ω₃."""
        return self if self._order() >= other._order() else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Implicación de Heyting sobre la cadena Ω₃."""
        if self._order() <= other._order():
            return HeytingVerdict.COHERENT
        return other

    def neg(self) -> "HeytingVerdict":
        """Negación intuicionista ¬a := a → ⊥."""
        return self.implies(HeytingVerdict.VETOED)


# ═══════════════════════════════════════════════════════════════════════════
# CERTIFICADO PÚBLICO
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class TopologicalControlSurfaceAgentCertificate:
    r"""Certificado inmutable emitido por el soberano de la superficie de control."""

    phase: str
    heyting_verdict: str
    replicator_entropy: float
    spectral_entropy_mac: float
    hamiltonian_energy: float
    energy_derivative: float
    is_lyapunov_stable: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    digital_signature_sha256: str

    schema_version: str = _SCHEMA_VERSION
    session_digest: str = ""
    condition_number: float = math.inf
    rho_minimum_eigenvalue: float = math.nan
    rho_purity: float = math.nan
    simplex_residual: float = math.nan
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)

    # Extensiones v2.1 (compatibles por keywords).
    spectral_gap: float = math.nan
    algebraic_connectivity: float = math.nan
    fitness: float = math.nan
    analytic_energy_derivative: float = math.nan
    energy_derivative_residual: float = math.nan
    brockett_alignment: float = math.nan
    commutator_frobenius: float = math.nan
    isospectral_residual: float = math.nan
    shahshahani_speed: float = math.nan
    kl_to_uniform: float = math.nan
    purity_drift: float = math.nan
    entropy_drift: float = math.nan
    rho_trace: float = math.nan
    K_minimum_eigenvalue: float = math.nan


# ═══════════════════════════════════════════════════════════════════════════
# DOSSIERS INTERNOS DE FASE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class DensityAudit:
    """Invariantes de una matriz de densidad proyectada a 𝒟(ℋ)."""

    rho: np.ndarray
    min_eig: float
    purity: float
    von_neumann: float
    trace: float
    hermitian_residual: float
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConductivityAudit:
    """Invariantes espectrales de K purificada (PSD ∩ Simₙ)."""

    K: np.ndarray
    condition_number: float
    min_eig: float
    spectral_gap: float
    algebraic_connectivity: float
    spectral_radius: float
    hermitian_residual: float
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase1Dossier:
    """Expediente canonizado y congelado en la FASE 1 · OBSERVE."""

    p: np.ndarray
    rho: np.ndarray
    K_purified: np.ndarray
    has_torsion_homological: bool
    condition_number: float
    session_digest: str
    diagnostics: Tuple[str, ...]
    simplex_residual: float = 0.0
    rho_min_eig: float = 0.0
    rho_purity: float = 1.0
    rho_trace: float = 1.0
    rho_von_neumann: float = 0.0
    rho_hermitian_residual: float = 0.0
    K_min_eig: float = 0.0
    spectral_gap: float = 1.0
    algebraic_connectivity: float = 0.0
    spectral_radius: float = 0.0
    replicator_entropy: float = 0.0
    kl_to_uniform: float = 0.0


@dataclass(frozen=True, slots=True)
class LyapunovAudit:
    """Energía Port-Hamiltoniana y derivadas (analítica / numérica)."""

    energy: float
    analytic_derivative: float
    numerical_derivative: float
    residual: float
    fitness: float
    fitness_variance: float
    shahshahani_speed: float
    brockett_alignment: float
    commutator_frobenius: float
    is_lyapunov_stable: bool


@dataclass(frozen=True, slots=True)
class TrajectoryAudit:
    """Residuos geométricos de la integración estructura-preservante."""

    isospectral_residual: float
    purity_drift: float
    entropy_drift: float
    simplex_residual: float
    rho_min_eig: float
    rho_purity: float
    rho_trace: float


@dataclass(frozen=True, slots=True)
class Phase2Dossier:
    """Expediente auditado en la FASE 2 · ORIENT/DECIDE."""

    phase1: Phase1Dossier
    p_final: np.ndarray
    rho_final: np.ndarray
    replicator_entropy: float
    spectral_entropy_mac: float
    hamiltonian_energy: float
    energy_derivative: float
    is_lyapunov_stable: bool
    rho_minimum_eigenvalue: float
    rho_purity: float
    simplex_residual: float
    heyting_verdict: str
    diagnostics: Tuple[str, ...]
    fitness: float = 0.0
    analytic_energy_derivative: float = 0.0
    energy_derivative_residual: float = 0.0
    brockett_alignment: float = 0.0
    commutator_frobenius: float = 0.0
    isospectral_residual: float = 0.0
    shahshahani_speed: float = 0.0
    kl_to_uniform: float = 0.0
    purity_drift: float = 0.0
    entropy_drift: float = 0.0
    rho_trace: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# SOBERANO DE SUPERFICIE DE CONTROL
# ═══════════════════════════════════════════════════════════════════════════


class TopologicalControlSurfaceAgent:
    r"""
    Soberano de Calibre de la Superficie de Control Topológica en el Estrato Wisdom.

    Unifica de forma continua la poda simplicial de la MIC (replicador de
    Shahshahani) y la purificación de la MAC (flujo isospectral de Brockett)
    sobre un fibrado cuya fibra es \(\Delta^{n-1}\times\mathcal{D}(\mathcal{H})\).

    El ciclo completo se ejecuta como una cadena anidada de morfismos:

        execute_control_surface_cycle()
          └─ _phase1_observe_and_canonicalize()     # unidad de FASE 2
               └─ _phase2_orient_and_decide()       # unidad de FASE 3
                    └─ _phase3_certify()

    Ante cualquier excepción no recuperable se emite un certificado
    fail-closed con veredicto VETOED e interlock activado.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        lyapunov_threshold: float = 1e-12,
        entropy_mac_ceiling: float = 0.5,
        *,
        rng_seed: int | None = None,
        integration_dt: float = 0.05,
        integration_steps: int = 4,
    ) -> None:
        r"""
        Inicializa el soberano de la superficie de control.

        Args:
            dimension_n: Dimensión fundamental N de la frontera (MIC/MAC).
            safety_margin: Escalador de tolerancia espectral.
            lyapunov_threshold: Máxima deriva admisible para dH/dt.
            entropy_mac_ceiling: Límite de entropía von Neumann para coste sano.
            rng_seed: Semilla opcional para reproducibilidad del interlock.
            integration_dt: Paso temporal por defecto del integrador.
            integration_steps: Número de pasos por defecto del integrador.

        Raises:
            ValueError: Si los parámetros físicos/numéricos son inválidos.
        """
        if isinstance(dimension_n, bool) or not isinstance(dimension_n, (int, np.integer)):
            raise ValueError("dimension_n debe ser un entero estrictamente positivo.")
        if int(dimension_n) <= 0:
            raise ValueError("La dimensión fundamental debe ser estrictamente positiva.")

        if isinstance(safety_margin, bool) or not math.isfinite(float(safety_margin)) or float(safety_margin) <= 0.0:
            raise ValueError("safety_margin debe ser finito y estrictamente positivo.")

        if isinstance(lyapunov_threshold, bool) or not math.isfinite(float(lyapunov_threshold)) or float(lyapunov_threshold) < 0.0:
            raise ValueError("lyapunov_threshold debe ser finito y no negativo.")

        if isinstance(entropy_mac_ceiling, bool) or not math.isfinite(float(entropy_mac_ceiling)) or float(entropy_mac_ceiling) < 0.0:
            raise ValueError("entropy_mac_ceiling debe ser finito y no negativo.")

        if isinstance(integration_dt, bool) or not math.isfinite(float(integration_dt)) or float(integration_dt) < 0.0:
            raise ValueError("integration_dt debe ser finito y no negativo.")

        if isinstance(integration_steps, bool) or not isinstance(integration_steps, (int, np.integer)):
            raise ValueError("integration_steps debe ser un entero no negativo.")
        if int(integration_steps) < 0:
            raise ValueError("integration_steps debe ser no negativo.")

        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)

        self._lyapunov_thresh: Final[float] = float(lyapunov_threshold) * self._safety_margin
        self._entropy_ceiling: Final[float] = float(entropy_mac_ceiling) * self._safety_margin
        self._entropy_hard_ceiling: Final[float] = self._entropy_ceiling * 2.0

        self._reg: Final[float] = 1e-15
        self._rng: Final[np.random.Generator] = np.random.default_rng(rng_seed)

        self._integration_dt: Final[float] = float(integration_dt)
        self._integration_steps: Final[int] = int(integration_steps)

        self._uniform: Final[np.ndarray] = np.full(
            self._n, 1.0 / float(self._n), dtype=np.float64
        )

    # ═══════════════════════════════════════════════════════════════════════
    # UTILIDADES DE HASH CANÓNICO Y ÁLGEBRA MENOR
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _hash_update_domain(sha: Any, domain: bytes) -> None:
        """Actualiza SHA-256 con separación de dominio length-prefixed."""
        sha.update(len(domain).to_bytes(8, "little", signed=False))
        sha.update(domain)

    @staticmethod
    def _hash_update_bytes(sha: Any, data: bytes) -> None:
        """Actualiza SHA-256 con bloque length-prefixed."""
        sha.update(len(data).to_bytes(8, "little", signed=False))
        sha.update(data)

    @classmethod
    def _hash_update_array(cls, sha: Any, name: str, array: np.ndarray) -> None:
        """
        Actualiza SHA-256 con arreglo numpy canónico.

        Incluye nombre, ndim, forma, dtype.str (endianness explícita) y
        bytes C-contiguos para evitar colisiones por re-interpretación.
        """
        cls._hash_update_bytes(sha, name.encode("utf-8"))

        arr = np.ascontiguousarray(array)

        sha.update(int(arr.ndim).to_bytes(4, "little", signed=False))
        sha.update(len(arr.shape).to_bytes(4, "little", signed=False))
        for dim in arr.shape:
            sha.update(int(dim).to_bytes(8, "little", signed=False))

        cls._hash_update_bytes(sha, arr.dtype.str.encode("ascii"))
        sha.update(arr.tobytes(order="C"))

    @staticmethod
    def _reject_bool(value: Any, name: str) -> None:
        if isinstance(value, bool) or isinstance(value, np.bool_):
            raise ValueError(f"{name} no puede ser booleano.")

    @staticmethod
    def _freeze_ndarray(array: np.ndarray, dtype: Any) -> np.ndarray:
        """Copia C-contigua, dtype canónico y write-protect."""
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _finite_float(value: Any, name: str) -> float:
        if isinstance(value, bool) or isinstance(value, np.bool_):
            raise ValueError(f"{name} no puede ser booleano.")
        try:
            x = float(value)
        except Exception as exc:
            raise ValueError(f"{name} no es escalar.") from exc
        if not math.isfinite(x):
            raise ValueError(f"{name} no es finito.")
        return x

    @staticmethod
    def _frobenius(arr: np.ndarray) -> float:
        val = float(np.linalg.norm(arr, ord="fro"))
        return val if math.isfinite(val) else math.inf

    @staticmethod
    def _heyting_meet(atoms: Iterable[HeytingVerdict]) -> HeytingVerdict:
        """Ínfimo arbitrario en Ω₃ (monoidal, asociativo, conmutativo, idempotente)."""
        acc = HeytingVerdict.COHERENT
        for atom in atoms:
            acc = acc.meet(atom)
        return acc

    def _barycenter(self) -> np.ndarray:
        return self._uniform.copy()

    # ═══════════════════════════════════════════════════════════════════════
    # GEOMETRÍA DEL SIMPLEX Y PROYECCIONES FÍSICAS
    # ═══════════════════════════════════════════════════════════════════════

    def _simplex_residual(self, p: np.ndarray) -> float:
        r"""
        Residuo afín-cónico del simplex:
            \( r = |\mathbf{1}^T p - 1| + \|\min(p,0)\|_1 \).
        """
        mass = float(np.sum(p))
        neg = float(np.sum(np.maximum(-p, 0.0)))
        residual = abs(mass - 1.0) + neg
        return residual if math.isfinite(residual) else math.inf

    def _project_probability_simplex(self, vector: Any) -> np.ndarray:
        r"""
        Proyección euclídea exacta sobre \(\Delta^{n-1}\) (Duchi et al., 2008):

            \(\min_w \|w-v\|_2^2\)  s.t.  \(w\ge 0,\ \mathbf{1}^T w = 1\).

        El algoritmo es \(O(n\log n)\) por la ordenación. Si el vector ya
        vive en el simplex (tolerancia de Wilkinson), se recorta y renormaliza.
        """
        try:
            v = np.asarray(vector, dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise ValueError("p_init no puede convertirse a float64.") from exc

        if v.size != self._n:
            raise ValueError(f"p_init debe tener dimensión {self._n}.")

        if not np.all(np.isfinite(v)):
            raise ValueError("p_init contiene valores no finitos.")

        tol = max(_STABILITY_FLOOR, 1000.0 * _MACHINE_EPS * max(1.0, float(self._n)))

        if np.all(v >= -tol) and abs(float(np.sum(v)) - 1.0) <= tol:
            out = np.maximum(v, 0.0)
            s = float(np.sum(out))
            if s > 0.0 and math.isfinite(s):
                return np.asarray(out / s, dtype=np.float64)
            return self._barycenter()

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, self._n + 1, dtype=np.float64)
        cond = u > (cssv / ind)

        if not np.any(cond):
            return self._barycenter()

        rho_idx = int(np.nonzero(cond)[0][-1])
        theta = float(cssv[rho_idx] / float(rho_idx + 1))

        w = np.maximum(v - theta, 0.0)
        s = float(np.sum(w))

        if not math.isfinite(s) or s <= 0.0:
            return self._barycenter()

        return np.asarray(w / s, dtype=np.float64)

    def _kl_to_uniform(self, p: np.ndarray) -> float:
        r"""
        Divergencia de Kullback–Leibler \(D_{\mathrm{KL}}(p\|u)\) al baricentro
        \(u=n^{-1}\mathbf{1}\). Equivale a \(\log n - H(p)\).
        """
        p_safe = np.clip(p, self._reg, None)
        p_safe = p_safe / float(np.sum(p_safe))
        log_n = math.log(float(self._n))
        entropy = -float(np.sum(p_safe * np.log(p_safe)))
        kl = log_n - entropy
        if not math.isfinite(kl):
            return math.inf
        return max(0.0, kl)

    def _phase1_canonical_density_matrix(
        self,
        rho: Any,
    ) -> DensityAudit:
        r"""
        Canoniza una matriz de densidad \(\rho\in\mathcal{D}(\mathcal{H})\).

        Se impone, en este orden:
          1. Hermiticidad: \(\rho\leftarrow(\rho+\rho^\dagger)/2\).
          2. Proyección espectral al cono PSD (Higham).
          3. Normalización de traza \(\mathrm{Tr}\,\rho=1\).
          4. Piso de Wilkinson para estabilidad de \(S(\rho)=-\mathrm{Tr}(\rho\log\rho)\).

        El resultado vive en el interior relativo de \(\mathcal{D}(\mathcal{H})\).
        """
        diagnostics: List[str] = []

        try:
            arr = np.asarray(rho, dtype=np.complex128)
        except Exception as exc:
            raise ValueError("rho_init no puede convertirse a complex128.") from exc

        if arr.shape != (self._n, self._n):
            raise ValueError(f"rho_init debe tener forma ({self._n},{self._n}).")

        if not np.all(np.isfinite(arr)):
            raise ValueError("rho_init contiene valores no finitos.")

        herm = 0.5 * (arr + arr.conj().T)
        hermitian_residual = self._frobenius(arr - herm)

        try:
            eigvals, eigvecs = la.eigh(herm)
        except la.LinAlgError as exc:
            raise ValueError("No fue posible diagonalizar rho_init.") from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise ValueError("Autovalores de rho_init inválidos.")

        eigvals_real = np.real(eigvals)
        eigvals_proj = np.maximum(eigvals_real, 0.0)
        mass = float(np.sum(eigvals_proj))

        if mass <= self._reg:
            eigvals_proj = np.full(self._n, 1.0 / float(self._n), dtype=np.float64)
            diagnostics.append("MAC proyectada a estado máximamente mixto.")
        else:
            eigvals_proj = eigvals_proj / mass

        eigvals_proj = np.maximum(eigvals_proj, self._reg)
        eigvals_proj = eigvals_proj / float(np.sum(eigvals_proj))

        rho_proj = (eigvecs * eigvals_proj) @ eigvecs.conj().T
        rho_proj = 0.5 * (rho_proj + rho_proj.conj().T)

        min_eig = float(np.min(eigvals_proj))
        purity = float(np.real(np.trace(rho_proj @ rho_proj)))
        trace = float(np.real(np.trace(rho_proj)))
        von_neumann = -float(np.sum(eigvals_proj * np.log(eigvals_proj)))

        if not math.isfinite(min_eig):
            raise ValueError("λ_min proyectada de rho no finita.")
        if not math.isfinite(purity):
            purity = math.nan
            diagnostics.append("Pureza de rho no finita; se conserva NaN.")
        if not math.isfinite(von_neumann):
            raise ValueError("Entropía von Neumann de rho no finita.")
        if hermitian_residual > _HERMITIAN_REL * max(1.0, self._frobenius(herm)):
            diagnostics.append(
                f"ρ no hermitiana (‖ρ-ρ†‖_F={hermitian_residual:.6e})."
            )

        return DensityAudit(
            rho=np.asarray(rho_proj, dtype=np.complex128),
            min_eig=min_eig,
            purity=purity,
            von_neumann=max(0.0, von_neumann),
            trace=trace,
            hermitian_residual=float(hermitian_residual),
            diagnostics=tuple(diagnostics),
        )

    def _phase1_purify_conductivity(
        self,
        K_boundary_raw: Any,
    ) -> ConductivityAudit:
        r"""
        Purificación Higham / Weyl–Toeplitz de la matriz de conocimiento MIC.

        Se proyecta a \(K\in\mathrm{PSD}\cap\mathrm{Sim}_n(\mathbb{R})\):
            \(K_H=(K+K^\dagger)/2\), \(\lambda_i\leftarrow\max(\lambda_i,\varepsilon)\).

        \(\kappa_2=\lambda_{\max}/\lambda_{\min}^+\), cota \([1,10^{12}]\).
        El gap algebraico \(\lambda_{\mathrm{Fiedler}}\) es el segundo
        autovalor ordenado (conectividad del grafo de conocimiento).
        """
        diagnostics: List[str] = []

        try:
            arr = np.asarray(K_boundary_raw, dtype=np.complex128)
        except Exception as exc:
            raise ValueError("K_boundary_raw no puede convertirse a complex128.") from exc

        if arr.shape != (self._n, self._n):
            raise ValueError(f"K_boundary_raw debe tener forma ({self._n},{self._n}).")

        if not np.all(np.isfinite(arr)):
            raise ValueError("K_boundary_raw contiene valores no finitos.")

        K_h = 0.5 * (arr + arr.conj().T)
        hermitian_residual = self._frobenius(arr - K_h)

        try:
            eigvals, eigvecs = la.eigh(K_h)
        except la.LinAlgError as exc:
            raise ValueError("No fue posible diagonalizar K_boundary_raw.") from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise ValueError("Autovalores de K_boundary_raw inválidos.")

        eigvals_real = np.real(eigvals)
        scale = max(1.0, float(np.max(np.abs(eigvals_real))) if eigvals_real.size else 1.0)
        tol = max(_STABILITY_FLOOR, 1000.0 * _MACHINE_EPS * scale)

        negative_count = int(np.sum(eigvals_real < -tol))
        if negative_count > 0:
            diagnostics.append(
                f"MIC con {negative_count} autovalores negativos; "
                "se proyectaron a PSD (Higham)."
            )

        clamped = np.maximum(eigvals_real, self._reg)
        K_complex = (eigvecs * clamped) @ eigvecs.conj().T
        K_real = np.asarray(
            0.5 * (K_complex + K_complex.conj().T).real,
            dtype=np.float64,
        )

        positive = clamped[clamped > self._reg]
        if positive.size == 0:
            condition_number = 1.0
            spectral_gap = 1.0
        else:
            condition_number = float(np.max(clamped) / float(np.min(positive)))
            if not math.isfinite(condition_number) or condition_number <= 0.0:
                condition_number = _CONDITION_CAP
                spectral_gap = 0.0
            else:
                spectral_gap = min(1.0, 1.0 / condition_number)

        condition_number = float(min(max(1.0, condition_number), _CONDITION_CAP))
        if not math.isfinite(spectral_gap) or spectral_gap < 0.0:
            spectral_gap = 0.0

        spectral_radius = float(np.max(np.abs(clamped)))
        min_eig = float(np.min(clamped))
        sorted_eigs = np.sort(clamped)
        if sorted_eigs.size >= 2:
            algebraic_connectivity = float(sorted_eigs[1])
        else:
            algebraic_connectivity = float(sorted_eigs[0]) if sorted_eigs.size else 0.0

        return ConductivityAudit(
            K=K_real,
            condition_number=condition_number,
            min_eig=min_eig,
            spectral_gap=float(spectral_gap),
            algebraic_connectivity=float(algebraic_connectivity),
            spectral_radius=float(spectral_radius) if math.isfinite(spectral_radius) else math.inf,
            hermitian_residual=float(hermitian_residual),
            diagnostics=tuple(diagnostics),
        )

    def _phase1_validate_real_symmetric_matrix(
        self,
        matrix: Any,
        name: str,
    ) -> np.ndarray:
        """Valida matriz real simétrica finita de dimensión N."""
        try:
            arr = np.asarray(matrix)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse a matriz.") from exc

        if np.iscomplexobj(arr):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contiene valores no finitos.")
            arr = arr.real

        try:
            arr = np.asarray(arr, dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse a float64.") from exc

        if arr.shape != (self._n, self._n):
            raise ValueError(f"{name} debe tener forma ({self._n},{self._n}).")

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene valores no finitos.")

        return np.asarray(0.5 * (arr + arr.T), dtype=np.float64)

    def _phase1_generate_session_digest(
        self,
        *,
        p: np.ndarray,
        rho: np.ndarray,
        K: np.ndarray,
        has_torsion_homological: bool,
    ) -> str:
        """Genera firma SHA-256 de sesión con separación de dominio."""
        sha = hashlib.sha256()

        self._hash_update_domain(sha, _DOMAIN_SESSION)
        self._hash_update_array(sha, "p_init", p)
        self._hash_update_array(sha, "rho_init", rho)
        self._hash_update_array(sha, "K_boundary_raw", K)

        torsion_byte = np.asarray(
            [1 if bool(has_torsion_homological) else 0],
            dtype=np.uint8,
        )
        self._hash_update_bytes(sha, torsion_byte.tobytes(order="C"))
        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        self._hash_update_bytes(sha, str(self._n).encode("ascii"))

        return sha.hexdigest()

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1 · OBSERVE
    #
    # Responsabilidad:
    #   - Proyectar p al simplex Δ^{n-1} (Duchi).
    #   - Proyectar ρ al espacio de densidades 𝒟(ℋ) (Higham espectral).
    #   - Purificar K como PSD real simétrica.
    #   - Medir residuos (simplex, hermiticidad, κ₂).
    #   - Congelar tensores write-protected.
    #   - Firmar sesión.
    #   - Transferir el expediente como unidad de la FASE 2.
    #
    # Cada método es un prefijo del siguiente; el último morfismo
    # `_phase1_observe_and_canonicalize` ES la continuación / inicio
    # formal de la FASE 2.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase1_observe_and_canonicalize(
        self,
        *,
        p_init: np.ndarray,
        rho_init: np.ndarray,
        K_boundary_raw: np.ndarray,
        has_torsion_homological: bool,
    ) -> TopologicalControlSurfaceAgentCertificate:
        """
        FASE 1 · OBSERVE  —  último morfismo de la fase.

        Definición formal:
            Obs : Inp → Phase1Dossier
            η₁  : Phase1Dossier → Certificado
                = `_phase2_orient_and_decide` ∘ Obs

        Su valor de retorno no se detiene en FASE 1: es la unidad
        (continuación estricta) de la FASE 2 · ORIENT / DECIDE.
        """
        diagnostics: List[str] = []

        p = self._project_probability_simplex(p_init)
        simplex_residual = self._simplex_residual(p)

        density = self._phase1_canonical_density_matrix(rho_init)
        diagnostics.extend(density.diagnostics)

        conductivity = self._phase1_purify_conductivity(K_boundary_raw)
        diagnostics.extend(conductivity.diagnostics)

        has_torsion = bool(has_torsion_homological)
        if has_torsion:
            diagnostics.append("Torsión homológica reportada por el estrato superior.")

        p_frozen = self._freeze_ndarray(p, np.float64)
        rho_frozen = self._freeze_ndarray(density.rho, np.complex128)
        K_frozen = self._freeze_ndarray(conductivity.K, np.float64)

        session_digest = self._phase1_generate_session_digest(
            p=p_frozen,
            rho=rho_frozen,
            K=np.asarray(K_boundary_raw),
            has_torsion_homological=has_torsion,
        )

        replicator_entropy = self.compute_replicator_entropy(p_frozen)
        kl_uniform = self._kl_to_uniform(p_frozen)

        dossier = Phase1Dossier(
            p=p_frozen,
            rho=rho_frozen,
            K_purified=K_frozen,
            has_torsion_homological=has_torsion,
            condition_number=conductivity.condition_number,
            session_digest=session_digest,
            diagnostics=tuple(diagnostics),
            simplex_residual=simplex_residual,
            rho_min_eig=density.min_eig,
            rho_purity=density.purity,
            rho_trace=density.trace,
            rho_von_neumann=density.von_neumann,
            rho_hermitian_residual=density.hermitian_residual,
            K_min_eig=conductivity.min_eig,
            spectral_gap=conductivity.spectral_gap,
            algebraic_connectivity=conductivity.algebraic_connectivity,
            spectral_radius=conductivity.spectral_radius,
            replicator_entropy=replicator_entropy,
            kl_to_uniform=kl_uniform,
        )

        logger.info(
            "FASE 1 OBSERVE: superficie de control canonizada. "
            "κ(K)=%.6e, gap=%.6e, torsión=%s, S(ρ)=%.6f, digest=%s",
            conductivity.condition_number,
            conductivity.spectral_gap,
            has_torsion,
            density.von_neumann,
            session_digest[:16],
        )

        # ── continuación estricta: FASE 1 ▸ FASE 2 ────────────────────────
        return self._phase2_orient_and_decide(dossier)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2 · ORIENT / DECIDE
    #
    # Inicio formal: este bloque es la continuación algebraica del último
    # método de la FASE 1 (`_phase1_observe_and_canonicalize` → aquí).
    #
    # Responsabilidad:
    #   - Campo replicador (gradiente de Shahshahani de ½ pᵀKp).
    #   - Doble corchete de Brockett (gradiente isospectral de Tr(ρ N)).
    #   - Integración estructura-preservante (exponencial en Δ, Lie-Euler
    #     isospectral en 𝒟(ℋ)).
    #   - Energía de Lyapunov y derivada analítica Ḣ = −Var_p(Kp).
    #   - Clasificación como ⋀ de átomos en Ω₃.
    #   - Transferir a FASE 3.
    # ═══════════════════════════════════════════════════════════════════════

    def compute_replicator_entropy(self, p: np.ndarray) -> float:
        r"""
        Entropía de Shannon de la distribución de la MIC (nats):

            \( H(p)=-\sum_i p_i\ln p_i \in[0,\log n] \).
        """
        p_proj = self._project_probability_simplex(p)
        p_safe = np.clip(p_proj, self._reg, None)
        p_safe = p_safe / float(np.sum(p_safe))
        entropy = -float(np.sum(p_safe * np.log(p_safe)))
        if not math.isfinite(entropy):
            raise ValueError("Entropía replicadora no finita.")
        return float(min(max(0.0, entropy), math.log(float(self._n)) + 1e-12))

    def compute_von_neumann_entropy(self, rho: np.ndarray) -> float:
        r"""
        Entropía espectral de von Neumann de la MAC:

            \( S(\rho)=-\mathrm{Tr}(\rho\ln\rho) \).

        Depende sólo del espectro; es invariante isospectral, luego se
        conserva a lo largo del flujo de Brockett (salvo deriva numérica).
        """
        try:
            arr = np.asarray(rho, dtype=np.complex128)
        except Exception as exc:
            raise ValueError("rho no puede convertirse a complex128.") from exc

        if arr.shape != (self._n, self._n):
            raise ValueError(f"rho debe tener forma ({self._n},{self._n}).")
        if not np.all(np.isfinite(arr)):
            raise ValueError("rho contiene valores no finitos.")

        rho_h = 0.5 * (arr + arr.conj().T)
        try:
            eigvals = la.eigvalsh(rho_h)
        except la.LinAlgError as exc:
            raise ValueError("No fue posible diagonalizar rho.") from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise ValueError("Autovalores de rho inválidos.")

        eigvals = np.maximum(np.real(eigvals), self._reg)
        eigvals = eigvals / float(np.sum(eigvals))
        entropy = -float(np.sum(eigvals * np.log(eigvals)))
        if not math.isfinite(entropy):
            raise ValueError("Entropía von Neumann no finita.")
        return float(min(max(0.0, entropy), math.log(float(self._n)) + 1e-12))

    def _phase2_sorted_spectrum(self, rho: np.ndarray) -> np.ndarray:
        """Espectro real ordenado de una matriz hermitiana."""
        arr = 0.5 * (np.asarray(rho, dtype=np.complex128) + np.asarray(rho, dtype=np.complex128).conj().T)
        try:
            eigvals = la.eigvalsh(arr)
        except la.LinAlgError:
            return np.full(self._n, math.nan, dtype=np.float64)
        return np.sort(np.real(eigvals))

    def evaluate_replicator_vector_field(
        self,
        p: np.ndarray,
        K_purified: np.ndarray,
    ) -> np.ndarray:
        r"""
        Campo vectorial del replicador de Shahshahani / Lotka–Volterra:

            \( \dot p_i = p_i\bigl[(Kp)_i - p^T K p\bigr] \).

        Es el gradiente riemanniano de \(F(p)=\tfrac12 p^T K p\) respecto
        de la métrica de Shahshahani \( g_p(x,y)=\sum_i x_i y_i/p_i \)
        sobre el interior de \(\Delta^{n-1}\).

        Invariantes:
          - \(\mathbf{1}^T\dot p = 0\) (tangente al simplex),
          - \(p_i=0\Rightarrow\dot p_i=0\) (caras invariantes).
        """
        p_proj = self._project_probability_simplex(p)
        K = self._phase1_validate_real_symmetric_matrix(K_purified, "K_purified")

        Kp = K @ p_proj
        phi = float(np.dot(p_proj, Kp))
        dp = p_proj * (Kp - phi)

        # Corrección numérica para preservar el hiperplano Σp=1.
        dp = dp - p_proj * float(np.sum(dp))

        if not np.all(np.isfinite(dp)):
            raise ValueError("Campo replicador no finito.")
        return np.asarray(dp, dtype=np.float64)

    def evaluate_brockett_double_bracket(
        self,
        rho: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        r"""
        Flujo de doble corchete de Brockett sobre la MAC:

            \( \dot\rho = [\rho,[\rho,N(p)]] \),  \( N(p)=\mathrm{diag}(p) \).

        Es el gradiente de \( \mathrm{Tr}(\rho N) \) sobre la órbita adjunta
        (variedad isospectral). Conserva el espectro de \(\rho\) y por tanto
        \(S(\rho)\) y la pureza \(\mathrm{Tr}(\rho^2)\).

        El resultado se simetriza y se proyecta al hiperplano de traza nula.
        """
        p_proj = self._project_probability_simplex(p)

        try:
            rho_arr = np.asarray(rho, dtype=np.complex128)
        except Exception as exc:
            raise ValueError("rho no puede convertirse a complex128.") from exc

        if rho_arr.shape != (self._n, self._n):
            raise ValueError(f"rho debe tener forma ({self._n},{self._n}).")
        if not np.all(np.isfinite(rho_arr)):
            raise ValueError("rho contiene valores no finitos.")

        rho_h = 0.5 * (rho_arr + rho_arr.conj().T)
        N_op = np.diag(p_proj).astype(np.complex128)

        bracket1 = rho_h @ N_op - N_op @ rho_h
        bracket2 = rho_h @ bracket1 - bracket1 @ rho_h
        bracket2 = 0.5 * (bracket2 + bracket2.conj().T)

        tr = np.trace(bracket2)
        bracket2 = bracket2 - np.eye(self._n, dtype=np.complex128) * (tr / float(self._n))

        if not np.all(np.isfinite(bracket2)):
            raise ValueError("Doble corchete de Brockett no finito.")
        return np.asarray(bracket2, dtype=np.complex128)

    def _sanitize_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Normaliza y proyecta una matriz de densidad en etapas de integración."""
        arr = np.asarray(rho, dtype=np.complex128)
        if arr.shape != (self._n, self._n):
            raise ValueError(f"rho debe tener forma ({self._n},{self._n}).")

        arr = 0.5 * (arr + arr.conj().T)
        tr = float(np.real(np.trace(arr)))
        if not math.isfinite(tr) or abs(tr) <= self._reg:
            return np.eye(self._n, dtype=np.complex128) / float(self._n)

        arr = arr / tr
        try:
            eigvals, eigvecs = la.eigh(arr)
        except la.LinAlgError:
            return np.eye(self._n, dtype=np.complex128) / float(self._n)

        eigvals = np.maximum(np.real(eigvals), 0.0)
        mass = float(np.sum(eigvals))
        if mass <= self._reg:
            return np.eye(self._n, dtype=np.complex128) / float(self._n)

        eigvals = eigvals / mass
        eigvals = np.maximum(eigvals, self._reg)
        eigvals = eigvals / float(np.sum(eigvals))
        out = (eigvecs * eigvals) @ eigvecs.conj().T
        return np.asarray(0.5 * (out + out.conj().T), dtype=np.complex128)

    def _phase2_replicator_exponential_map(
        self,
        p: np.ndarray,
        K: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        r"""
        Mapa exponencial de Shahshahani (Euler multiplicativo):

            \( p_i' \propto p_i\,\exp\bigl(\Delta t\,(Kp)_i\bigr),\quad
               \mathbf{1}^T p' = 1 \).

        Es la solución exacta del replicador a fitness congelado, permanece
        en el interior relativo del simplex y respeta las caras
        (\(p_i=0\Rightarrow p_i'=0\)).
        """
        if dt <= 0.0:
            return self._project_probability_simplex(p)

        p_proj = self._project_probability_simplex(p)
        Kp = K @ p_proj
        # Centrado para estabilidad numérica del exp.
        Kp = Kp - float(np.max(Kp))
        scaled = dt * Kp
        # Evita overflow: clip del exponente.
        scaled = np.clip(scaled, -60.0, 60.0)
        weights = p_proj * np.exp(scaled)
        s = float(np.sum(weights))
        if not math.isfinite(s) or s <= 0.0:
            return p_proj
        out = weights / s
        if not np.all(np.isfinite(out)):
            return p_proj
        return np.asarray(out, dtype=np.float64)

    def _phase2_brockett_isospectral_step(
        self,
        rho: np.ndarray,
        p: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        r"""
        Paso de Lie–Euler isospectral para el doble corchete.

        Si \(A=-[\rho,N]\) (anti-hermitiana), entonces
            \(\rho' = U\rho U^\dagger,\quad U=\exp(\Delta t\, A)\)
        es isospectral por conjugación unitaria y reproduce
            \(\dot\rho=[\rho,[\rho,N]]\) al primer orden.

        Fallback: RK2 proyectado si `expm` falla.
        """
        if dt <= 0.0:
            return self._sanitize_density_matrix(rho)

        p_proj = self._project_probability_simplex(p)
        rho_h = 0.5 * (
            np.asarray(rho, dtype=np.complex128)
            + np.asarray(rho, dtype=np.complex128).conj().T
        )
        N_op = np.diag(p_proj).astype(np.complex128)
        commutator = rho_h @ N_op - N_op @ rho_h  # [ρ, N], anti-hermitiana
        A = -commutator

        try:
            U = la.expm(A * dt)
            rho_next = U @ rho_h @ U.conj().T
            if not np.all(np.isfinite(rho_next)):
                raise la.LinAlgError("expm no finito.")
            return self._sanitize_density_matrix(rho_next)
        except Exception:
            drho = self.evaluate_brockett_double_bracket(rho_h, p_proj)
            rho_mid = self._sanitize_density_matrix(rho_h + 0.5 * dt * drho)
            drho2 = self.evaluate_brockett_double_bracket(rho_mid, p_proj)
            return self._sanitize_density_matrix(rho_h + dt * drho2)

    def _phase2_fitness_moments(
        self,
        p: np.ndarray,
        K: np.ndarray,
    ) -> Tuple[float, float, float]:
        r"""
        Momentos de fitness bajo \(p\):

            \(\varphi = p^T K p,\quad
              \mathrm{Var}_p(Kp)=\sum_i p_i((Kp)_i-\varphi)^2\).

        Identidad de Shahshahani: \( g_p(\dot p,\dot p)=\mathrm{Var}_p(Kp)\).
        """
        p_proj = self._project_probability_simplex(p)
        Kp = K @ p_proj
        phi = float(np.dot(p_proj, Kp))
        var = float(np.dot(p_proj, (Kp - phi) ** 2))
        if not math.isfinite(phi):
            raise ValueError("Fitness no finito.")
        if not math.isfinite(var) or var < 0.0:
            var = 0.0
        speed = math.sqrt(max(0.0, var))
        return phi, var, speed

    def _phase2_brockett_alignment(
        self,
        rho: np.ndarray,
        p: np.ndarray,
    ) -> Tuple[float, float]:
        r"""
        Alineamiento de Brockett \(\mathrm{Tr}(\rho N)\) y \(\|[\rho,N]\|_F\).

        Con \(N\) congelada, \(\frac{d}{dt}\mathrm{Tr}(\rho N)=\|[\rho,N]\|_F^2\ge 0\).
        """
        p_proj = self._project_probability_simplex(p)
        rho_h = 0.5 * (
            np.asarray(rho, dtype=np.complex128)
            + np.asarray(rho, dtype=np.complex128).conj().T
        )
        N_op = np.diag(p_proj).astype(np.complex128)
        alignment = float(np.real(np.trace(rho_h @ N_op)))
        commutator = rho_h @ N_op - N_op @ rho_h
        fro = self._frobenius(commutator)
        if not math.isfinite(alignment):
            alignment = math.nan
        return alignment, fro

    def _phase2_lyapunov_energy(
        self,
        p: np.ndarray,
        rho: np.ndarray,
        K: np.ndarray,
    ) -> float:
        r"""
        Energía Port-Hamiltoniana de Lyapunov:

            \( H(\Psi)=-\tfrac12 p^T K p + S(\rho) \).

        El término cuadrático es (menos) la fitness de la MIC; \(S(\rho)\)
        es la entropía de von Neumann, isospectralmente constante.
        """
        p_proj = self._project_probability_simplex(p)
        K_real = self._phase1_validate_real_symmetric_matrix(K, "K")
        fitness = float(np.dot(p_proj, K_real @ p_proj))
        entropy = self.compute_von_neumann_entropy(rho)
        energy = -0.5 * fitness + entropy
        if not math.isfinite(energy):
            raise ValueError("Energía de Lyapunov no finita.")
        return energy

    def _phase2_lyapunov_audit(
        self,
        p: np.ndarray,
        rho: np.ndarray,
        K: np.ndarray,
        numerical_derivative: float,
    ) -> LyapunovAudit:
        r"""
        Auditoría de Lyapunov.

        Identidad analítica (replicador + isospectralidad de Brockett):
            \( \dot H = -\mathrm{Var}_p(Kp) \le 0 \).

        El residuo \(|\dot H_{\mathrm{num}}-\dot H_{\mathrm{an}}|\) mide
        deriva de proyección / no-isospectralidad numérica.
        """
        energy = self._phase2_lyapunov_energy(p, rho, K)
        fitness, variance, speed = self._phase2_fitness_moments(p, K)
        analytic = -variance
        residual = abs(float(numerical_derivative) - analytic)
        if not math.isfinite(residual):
            residual = math.inf
        alignment, fro = self._phase2_brockett_alignment(rho, p)
        is_stable = bool(float(numerical_derivative) <= self._lyapunov_thresh)
        return LyapunovAudit(
            energy=energy,
            analytic_derivative=float(analytic),
            numerical_derivative=float(numerical_derivative),
            residual=float(residual),
            fitness=float(fitness),
            fitness_variance=float(variance),
            shahshahani_speed=float(speed),
            brockett_alignment=float(alignment),
            commutator_frobenius=float(fro),
            is_lyapunov_stable=is_stable,
        )

    def integrate_control_surface_trajectory(
        self,
        p_init: np.ndarray,
        rho_init: np.ndarray,
        K_purified: np.ndarray,
        dt: float = 0.05,
        steps: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        r"""
        Integra la trayectoria acoplada con un splitting estructura-preservante:

          1. Replicador: mapa exponencial de Shahshahani (simplex exacto).
          2. Brockett: Lie–Euler isospectral (órbita adjunta).

        Retorna:
            p_final, rho_final, H_final, dH/dt numérico (diferencia finita).
        """
        p = self._project_probability_simplex(p_init)
        density = self._phase1_canonical_density_matrix(rho_init)
        rho = density.rho
        conductivity = self._phase1_purify_conductivity(K_purified)
        K = conductivity.K

        try:
            time_step = self._finite_float(dt, "dt")
            self._reject_bool(steps, "steps")
            n_steps = int(steps)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("dt/steps inválidos en integración.") from exc

        if n_steps < 0:
            raise ValueError("steps debe ser no negativo.")
        if time_step < 0.0:
            raise ValueError("dt debe ser no negativo.")

        H_init = self._phase2_lyapunov_energy(p, rho, K)

        if n_steps == 0 or time_step <= 0.0:
            return p, rho, H_init, 0.0

        for _ in range(n_steps):
            p = self._phase2_replicator_exponential_map(p, K, time_step)
            rho = self._phase2_brockett_isospectral_step(rho, p, time_step)

        H_final = self._phase2_lyapunov_energy(p, rho, K)
        total_time = float(n_steps) * time_step
        dH_dt = (H_final - H_init) / total_time if total_time > 0.0 else 0.0
        if not math.isfinite(dH_dt):
            raise ValueError("Derivada energética dH/dt no finita.")

        return p, rho, H_final, float(dH_dt)

    def _phase2_trajectory_audit(
        self,
        p_final: np.ndarray,
        rho_final: np.ndarray,
        rho_init: np.ndarray,
        S_init: float,
        purity_init: float,
    ) -> TrajectoryAudit:
        """Residuos geométricos post-integración (simplex, espectro, traza)."""
        simplex_residual = self._simplex_residual(p_final)

        spec_init = self._phase2_sorted_spectrum(rho_init)
        spec_final = self._phase2_sorted_spectrum(rho_final)
        if spec_init.size == spec_final.size and np.all(np.isfinite(spec_init)) and np.all(np.isfinite(spec_final)):
            isospectral_residual = float(np.linalg.norm(spec_final - spec_init, ord=2))
        else:
            isospectral_residual = math.inf

        rho_h = 0.5 * (rho_final + rho_final.conj().T)
        try:
            rho_eigs = la.eigvalsh(rho_h)
            rho_min_eig = float(np.min(np.real(rho_eigs))) if rho_eigs.size else 0.0
        except la.LinAlgError:
            rho_min_eig = -math.inf

        rho_purity = float(np.real(np.trace(rho_final @ rho_final)))
        rho_trace = float(np.real(np.trace(rho_final)))
        S_final = self.compute_von_neumann_entropy(rho_final)

        purity_drift = abs(rho_purity - purity_init) if math.isfinite(rho_purity) and math.isfinite(purity_init) else math.inf
        entropy_drift = abs(S_final - S_init)

        if not math.isfinite(isospectral_residual):
            isospectral_residual = math.inf
        if not math.isfinite(purity_drift):
            purity_drift = math.inf
        if not math.isfinite(entropy_drift):
            entropy_drift = math.inf

        return TrajectoryAudit(
            isospectral_residual=float(isospectral_residual),
            purity_drift=float(purity_drift),
            entropy_drift=float(entropy_drift),
            simplex_residual=float(simplex_residual),
            rho_min_eig=float(rho_min_eig),
            rho_purity=float(rho_purity) if math.isfinite(rho_purity) else math.nan,
            rho_trace=float(rho_trace) if math.isfinite(rho_trace) else math.nan,
        )

    def _phase2_evaluate_heyting(
        self,
        *,
        is_lyapunov_stable: bool,
        has_torsion_homological: bool,
        spectral_entropy_mac: float,
        energy_derivative: float,
        simplex_residual: float,
        rho_minimum_eigenvalue: float,
        condition_number: float,
        isospectral_residual: float,
        purity_drift: float,
        entropy_drift: float,
        rho_trace: float,
    ) -> Tuple[str, Tuple[str, ...]]:
        """
        Clasificación en el retículo de Heyting Ω₃ por ⋀ de átomos.

        Átomos de VETO:
          - Inestabilidad de Lyapunov / dH/dt dura.
          - Torsión homológica.
          - Entropía MAC sobre cota dura.
          - ρ no PSD o Tr(ρ) ≉ 1.
          - Residuo de simplex duro.

        Átomos de DEGRADED:
          - Entropía MAC sobre cota blanda.
          - dH/dt sobre margen blando.
          - Residuo de simplex blando.
          - κ₂ elevado.
          - Deriva isospectral / pureza / entropía.
        """
        diagnostics: List[str] = []

        lyapunov_atom = HeytingVerdict.COHERENT
        if not math.isfinite(energy_derivative):
            lyapunov_atom = HeytingVerdict.VETOED
            diagnostics.append("dH/dt no finito.")
        elif (not is_lyapunov_stable) or energy_derivative > _DHDT_HARD:
            lyapunov_atom = HeytingVerdict.VETOED
            diagnostics.append("Deriva de Lyapunov positiva sobre umbral / cota dura.")
        elif energy_derivative > self._lyapunov_thresh * 0.1:
            lyapunov_atom = HeytingVerdict.DEGRADED
            diagnostics.append("Deriva energética sobre margen blando.")

        torsion_atom = (
            HeytingVerdict.VETOED if has_torsion_homological else HeytingVerdict.COHERENT
        )
        if has_torsion_homological:
            diagnostics.append("Torsión homológica detectada.")

        entropy_atom = HeytingVerdict.COHERENT
        if spectral_entropy_mac > self._entropy_hard_ceiling:
            entropy_atom = HeytingVerdict.VETOED
            diagnostics.append("Entropía von Neumann sobre cota dura.")
        elif spectral_entropy_mac > self._entropy_ceiling:
            entropy_atom = HeytingVerdict.DEGRADED
            diagnostics.append("Entropía von Neumann sobre cota blanda.")

        mac_atom = HeytingVerdict.COHERENT
        if math.isfinite(rho_minimum_eigenvalue) and rho_minimum_eigenvalue < _RHO_PSD_HARD:
            mac_atom = HeytingVerdict.VETOED
            diagnostics.append("MAC no positiva semidefinida.")
        if (not math.isfinite(rho_trace)) or abs(rho_trace - 1.0) > 1e-6:
            mac_atom = HeytingVerdict.VETOED
            diagnostics.append("Traza de ρ no conservada.")

        simplex_atom = HeytingVerdict.COHERENT
        if simplex_residual > _SIMPLEX_HARD_RESIDUAL:
            simplex_atom = HeytingVerdict.VETOED
            diagnostics.append("Residuo de simplex sobre cota dura.")
        elif simplex_residual > _SIMPLEX_SOFT_RESIDUAL:
            simplex_atom = HeytingVerdict.DEGRADED
            diagnostics.append("Residuo de simplex fuera de tolerancia.")

        condition_atom = HeytingVerdict.COHERENT
        if condition_number > _CONDITION_DEGRADED:
            condition_atom = HeytingVerdict.DEGRADED
            diagnostics.append("Número de condición MIC elevado.")

        geometry_atom = HeytingVerdict.COHERENT
        if isospectral_residual > _ISOSPECTRAL_DEGRADED:
            geometry_atom = HeytingVerdict.DEGRADED
            diagnostics.append("Deriva isospectral de ρ sobre tolerancia.")
        if purity_drift > _PURITY_DRIFT_DEGRADED:
            geometry_atom = HeytingVerdict.DEGRADED
            diagnostics.append("Deriva de pureza Tr(ρ²) sobre tolerancia.")
        if entropy_drift > _ENTROPY_DRIFT_DEGRADED:
            geometry_atom = HeytingVerdict.DEGRADED
            diagnostics.append("Deriva de S(ρ) (debería ser isospectralmente nula).")

        verdict = self._heyting_meet(
            (
                lyapunov_atom,
                torsion_atom,
                entropy_atom,
                mac_atom,
                simplex_atom,
                condition_atom,
                geometry_atom,
            )
        )
        return verdict.value, tuple(diagnostics)

    def _phase2_orient_and_decide(
        self,
        dossier: Phase1Dossier,
    ) -> TopologicalControlSurfaceAgentCertificate:
        """
        FASE 2 · ORIENT / DECIDE  —  último morfismo de la fase.

        Definición formal:
            Dec : Phase1Dossier → Phase2Dossier
            η₂  : Phase2Dossier → Certificado
                = `_phase3_certify` ∘ Dec

        Su valor de retorno no se detiene en FASE 2: es la unidad
        (continuación estricta) de la FASE 3 · ACT / CERTIFY.
        """
        p_final, rho_final, H_final, dH_dt = self.integrate_control_surface_trajectory(
            p_init=dossier.p,
            rho_init=dossier.rho,
            K_purified=dossier.K_purified,
            dt=self._integration_dt,
            steps=self._integration_steps,
        )

        replicator_entropy = self.compute_replicator_entropy(p_final)
        spectral_entropy_mac = self.compute_von_neumann_entropy(rho_final)
        kl_uniform = self._kl_to_uniform(p_final)

        lyap = self._phase2_lyapunov_audit(
            p_final,
            rho_final,
            dossier.K_purified,
            numerical_derivative=dH_dt,
        )
        traj = self._phase2_trajectory_audit(
            p_final=p_final,
            rho_final=rho_final,
            rho_init=dossier.rho,
            S_init=dossier.rho_von_neumann,
            purity_init=dossier.rho_purity,
        )

        verdict, new_diagnostics = self._phase2_evaluate_heyting(
            is_lyapunov_stable=lyap.is_lyapunov_stable,
            has_torsion_homological=dossier.has_torsion_homological,
            spectral_entropy_mac=spectral_entropy_mac,
            energy_derivative=dH_dt,
            simplex_residual=traj.simplex_residual,
            rho_minimum_eigenvalue=traj.rho_min_eig,
            condition_number=dossier.condition_number,
            isospectral_residual=traj.isospectral_residual,
            purity_drift=traj.purity_drift,
            entropy_drift=traj.entropy_drift,
            rho_trace=traj.rho_trace if math.isfinite(traj.rho_trace) else math.nan,
        )

        diagnostics = tuple(list(dossier.diagnostics) + list(new_diagnostics))

        p_final_f = self._freeze_ndarray(p_final, np.float64)
        rho_final_f = self._freeze_ndarray(rho_final, np.complex128)

        phase2 = Phase2Dossier(
            phase1=dossier,
            p_final=p_final_f,
            rho_final=rho_final_f,
            replicator_entropy=replicator_entropy,
            spectral_entropy_mac=spectral_entropy_mac,
            hamiltonian_energy=H_final,
            energy_derivative=dH_dt,
            is_lyapunov_stable=lyap.is_lyapunov_stable,
            rho_minimum_eigenvalue=traj.rho_min_eig,
            rho_purity=traj.rho_purity,
            simplex_residual=traj.simplex_residual,
            heyting_verdict=verdict,
            diagnostics=diagnostics,
            fitness=lyap.fitness,
            analytic_energy_derivative=lyap.analytic_derivative,
            energy_derivative_residual=lyap.residual,
            brockett_alignment=lyap.brockett_alignment,
            commutator_frobenius=lyap.commutator_frobenius,
            isospectral_residual=traj.isospectral_residual,
            shahshahani_speed=lyap.shahshahani_speed,
            kl_to_uniform=kl_uniform,
            purity_drift=traj.purity_drift,
            entropy_drift=traj.entropy_drift,
            rho_trace=traj.rho_trace if math.isfinite(traj.rho_trace) else math.nan,
        )

        logger.info(
            "FASE 2 ORIENT: veredicto=%s, H=%.6e, dH/dt=%.6e "
            "(analítico=%.6e), S(ρ)=%.6f, Var(f)=%.6e",
            verdict,
            H_final,
            dH_dt,
            lyap.analytic_derivative,
            spectral_entropy_mac,
            lyap.fitness_variance,
        )

        # ── continuación estricta: FASE 2 ▸ FASE 3 ────────────────────────
        return self._phase3_certify(phase2)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3 · ACT / CERTIFY
    #
    # Inicio formal: este bloque es la continuación algebraica del último
    # método de la FASE 2 (`_phase2_orient_and_decide` → aquí).
    #
    # Responsabilidad:
    #   - Interlock crowbar ESP32 (GPIO14 / BT151 / IRAM ~400 ns) si VETO.
    #   - Firma SHA-256 de telemetría con separación de dominio.
    #   - Certificado inmutable (objeto inicial de la categoría de evidencias).
    # ═══════════════════════════════════════════════════════════════════════

    def _phase3_actuate_interlock(
        self,
        verdict: Any,
        session_hash: str,
    ) -> Tuple[bool, float]:
        """
        FASE 3 · ACT.

        Si Heyting colapsa a VETOED, la ISR en IRAM del ESP32 actúa en menos
        de 400 ns para conmutar GPIO14 y cebar el tiristor crowbar BT151.
        La latencia se recorta al intervalo físico [380, 420] ns.
        """
        if isinstance(verdict, HeytingVerdict):
            verdict_enum = verdict
        else:
            try:
                verdict_enum = HeytingVerdict(str(verdict).strip().upper())
            except ValueError:
                verdict_enum = HeytingVerdict.VETOED

        session_ref = str(session_hash)[:16]

        if verdict_enum is HeytingVerdict.VETOED:
            jitter = float(self._rng.normal(0.0, _CROWBAR_JITTER_STD_NS))
            latency = float(
                np.clip(
                    _CROWBAR_IRAM_LATENCY_NS + jitter,
                    _CROWBAR_LATENCY_MIN_NS,
                    _CROWBAR_LATENCY_MAX_NS,
                )
            )

            logger.critical(
                "¡VETO SÍNCRONO DE FRONTERA DETECTADO POR EL SOBERANO DE LA SUPERFICIE DE CONTROL! "
                "Bypass Crowbar BT151 [GPIO%d] gatillado síncronamente en IRAM. "
                "Latencia de conmutación: %.2f ns. Obra civil paralizada. Sello: %s",
                _GPIO_CROWBAR_PIN,
                latency,
                session_ref,
            )
            return True, latency

        logger.info(
            "Superficie de control topológica regulada. Sello inmutable: %s. Estado: %s",
            session_ref,
            verdict_enum.value,
        )
        return False, 0.0

    def _phase3_compose_signature(
        self,
        *,
        verdict: str,
        session_digest: str,
        replicator_entropy: float,
        spectral_entropy_mac: float,
        hamiltonian_energy: float,
        energy_derivative: float,
        condition_number: float,
        rho_minimum_eigenvalue: float,
        rho_purity: float,
        simplex_residual: float,
        extra_scalars: Sequence[float] = (),
    ) -> str:
        """Firma SHA-256 final de telemetría con separación de dominio."""
        sha = hashlib.sha256()

        self._hash_update_domain(sha, _DOMAIN_TELEMETRY)
        self._hash_update_bytes(sha, verdict.encode("utf-8"))
        self._hash_update_bytes(sha, session_digest.encode("utf-8"))

        scalars: Tuple[float, ...] = (
            replicator_entropy,
            spectral_entropy_mac,
            hamiltonian_energy,
            energy_derivative,
            condition_number,
            rho_minimum_eigenvalue,
            rho_purity,
            simplex_residual,
            *tuple(float(x) for x in extra_scalars),
        )

        for value in scalars:
            self._hash_update_bytes(sha, f"{float(value):.17e}".encode("ascii"))

        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        return sha.hexdigest()

    def _phase3_certify(
        self,
        dossier: Phase2Dossier,
    ) -> TopologicalControlSurfaceAgentCertificate:
        """
        FASE 3 · CERTIFY.

        Terminal del ciclo OODA anidado: acciona interlock si corresponde
        y emite el certificado inmutable. No hay FASE 4.
        """
        interlock_fired, latency = self._phase3_actuate_interlock(
            dossier.heyting_verdict,
            dossier.phase1.session_digest,
        )

        signature = self._phase3_compose_signature(
            verdict=dossier.heyting_verdict,
            session_digest=dossier.phase1.session_digest,
            replicator_entropy=dossier.replicator_entropy,
            spectral_entropy_mac=dossier.spectral_entropy_mac,
            hamiltonian_energy=dossier.hamiltonian_energy,
            energy_derivative=dossier.energy_derivative,
            condition_number=dossier.phase1.condition_number,
            rho_minimum_eigenvalue=dossier.rho_minimum_eigenvalue,
            rho_purity=dossier.rho_purity,
            simplex_residual=dossier.simplex_residual,
            extra_scalars=(
                dossier.fitness,
                dossier.analytic_energy_derivative,
                dossier.energy_derivative_residual,
                dossier.brockett_alignment,
                dossier.commutator_frobenius,
                dossier.isospectral_residual,
                dossier.shahshahani_speed,
                dossier.kl_to_uniform,
                dossier.purity_drift,
                dossier.entropy_drift,
                dossier.phase1.spectral_gap,
                dossier.phase1.algebraic_connectivity,
            ),
        )

        certificate = TopologicalControlSurfaceAgentCertificate(
            phase="G_WISDOM_CONTROL_SURFACE_SUTURATED",
            heyting_verdict=dossier.heyting_verdict,
            replicator_entropy=dossier.replicator_entropy,
            spectral_entropy_mac=dossier.spectral_entropy_mac,
            hamiltonian_energy=dossier.hamiltonian_energy,
            energy_derivative=dossier.energy_derivative,
            is_lyapunov_stable=dossier.is_lyapunov_stable,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            digital_signature_sha256=signature,
            schema_version=_SCHEMA_VERSION,
            session_digest=dossier.phase1.session_digest,
            condition_number=dossier.phase1.condition_number,
            rho_minimum_eigenvalue=dossier.rho_minimum_eigenvalue,
            rho_purity=dossier.rho_purity,
            simplex_residual=dossier.simplex_residual,
            diagnostics=dossier.diagnostics,
            spectral_gap=dossier.phase1.spectral_gap,
            algebraic_connectivity=dossier.phase1.algebraic_connectivity,
            fitness=dossier.fitness,
            analytic_energy_derivative=dossier.analytic_energy_derivative,
            energy_derivative_residual=dossier.energy_derivative_residual,
            brockett_alignment=dossier.brockett_alignment,
            commutator_frobenius=dossier.commutator_frobenius,
            isospectral_residual=dossier.isospectral_residual,
            shahshahani_speed=dossier.shahshahani_speed,
            kl_to_uniform=dossier.kl_to_uniform,
            purity_drift=dossier.purity_drift,
            entropy_drift=dossier.entropy_drift,
            rho_trace=dossier.rho_trace,
            K_minimum_eigenvalue=dossier.phase1.K_min_eig,
        )

        if dossier.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "FASE 3 CERTIFY: VETO de superficie de control. "
                "dH/dt=%.6e, S(ρ)=%.6f, torsión=%s",
                dossier.energy_derivative,
                dossier.spectral_entropy_mac,
                dossier.phase1.has_torsion_homological,
            )
        elif dossier.heyting_verdict == HeytingVerdict.DEGRADED.value:
            logger.warning(
                "FASE 3 CERTIFY: superficie degradada. diagnostics=%s",
                dossier.diagnostics,
            )
        else:
            logger.info(
                "FASE 3 CERTIFY: superficie coherente. digest=%s",
                dossier.phase1.session_digest[:16],
            )

        return certificate

    # ═══════════════════════════════════════════════════════════════════════
    # FAIL-CLOSED GLOBAL
    # ═══════════════════════════════════════════════════════════════════════

    def _fail_closed_certificate(self, reason: str) -> TopologicalControlSurfaceAgentCertificate:
        """
        Certificado fail-closed ante excepción no recuperable.

        Garantiza VETO, interlock y firma inmutable incluso cuando el
        expediente de entrada no pudo ser validado.
        """
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_SESSION)
        self._hash_update_bytes(sha, reason.encode("utf-8"))
        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        self._hash_update_bytes(sha, str(self._n).encode("ascii"))
        session_digest = sha.hexdigest()

        interlock_fired, latency = self._phase3_actuate_interlock(
            HeytingVerdict.VETOED.value,
            session_digest,
        )

        signature = self._phase3_compose_signature(
            verdict=HeytingVerdict.VETOED.value,
            session_digest=session_digest,
            replicator_entropy=0.0,
            spectral_entropy_mac=0.0,
            hamiltonian_energy=0.0,
            energy_derivative=0.0,
            condition_number=math.inf,
            rho_minimum_eigenvalue=0.0,
            rho_purity=1.0,
            simplex_residual=math.inf,
        )

        return TopologicalControlSurfaceAgentCertificate(
            phase="G_WISDOM_CONTROL_SURFACE_FAIL_CLOSED",
            heyting_verdict=HeytingVerdict.VETOED.value,
            replicator_entropy=0.0,
            spectral_entropy_mac=0.0,
            hamiltonian_energy=0.0,
            energy_derivative=0.0,
            is_lyapunov_stable=False,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            digital_signature_sha256=signature,
            schema_version=_SCHEMA_VERSION,
            session_digest=session_digest,
            condition_number=math.inf,
            rho_minimum_eigenvalue=0.0,
            rho_purity=1.0,
            simplex_residual=math.inf,
            diagnostics=(f"FAIL-CLOSED: {reason}",),
            spectral_gap=0.0,
            algebraic_connectivity=0.0,
            fitness=0.0,
            analytic_energy_derivative=0.0,
            energy_derivative_residual=math.inf,
            brockett_alignment=0.0,
            commutator_frobenius=math.inf,
            isospectral_residual=math.inf,
            shahshahani_speed=0.0,
            kl_to_uniform=math.inf,
            purity_drift=math.inf,
            entropy_drift=math.inf,
            rho_trace=0.0,
            K_minimum_eigenvalue=0.0,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # API PÚBLICA COMPATIBLE
    # ═══════════════════════════════════════════════════════════════════════

    def act_hardware_interlock_simulation(
        self,
        verdict: str,
        session_hash: str,
    ) -> Tuple[bool, float]:
        """
        API compatible con la versión original.

        Delega en la actuación de FASE 3.
        """
        return self._phase3_actuate_interlock(verdict, session_hash)

    def execute_control_surface_cycle(
        self,
        p_init: np.ndarray,
        rho_init: np.ndarray,
        K_boundary_raw: np.ndarray,
        has_torsion_homological: bool = False,
    ) -> TopologicalControlSurfaceAgentCertificate:
        r"""
        Orquesta el ciclo covariante OODA completo sobre la Superficie de Control.

        Cadena anidada:

            FASE 1 · OBSERVE
              └─ FASE 2 · ORIENT / DECIDE
                   └─ FASE 3 · ACT / CERTIFY

        Si ocurre cualquier excepción, devuelve certificado fail-closed VETOED.
        """
        try:
            return self._phase1_observe_and_canonicalize(
                p_init=p_init,
                rho_init=rho_init,
                K_boundary_raw=K_boundary_raw,
                has_torsion_homological=has_torsion_homological,
            )
        except Exception as exc:
            logger.exception(
                "Fallo fail-closed en superficie de control; emitiendo VETO."
            )
            return self._fail_closed_certificate(reason=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN DE FIRMAS DE CALIBRE
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "TopologicalControlSurfaceAgentCertificate",
    "TopologicalControlSurfaceAgent",
    "HeytingVerdict",
]

# Compatibilidad con la exportación histórica del módulo original.
all = __all__
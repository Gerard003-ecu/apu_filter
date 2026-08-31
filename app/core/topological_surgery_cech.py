# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Topological Surgery Čech (Cirugía Topológica de Haces de Čech)      ║
║ Ruta   : app/core/topological_surgery_cech.py                                ║
║ Versión: 1.1.0-Doctoral-Cech-Cohomology-Anisotropic-Surgery-FPU-Secure       ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Este módulo implementa el motor de cirugía topológica de-confinada sobre     ║
║ cubrimientos abiertos finos de Čech $\mathcal{U} = \{U_i\}$ en la frontera   ║
║ simplicial compacta $\partial K$. Ante la presencia de ruido analógico       ║
║ extremo (EMF, soldaduras, polvo analógico) en transductores locales, el      ║
║ sistema calcula el primer grupo de cohomología de Čech                       ║
║ $\check{H}^1(\mathcal{U}; \, \mathcal{F})$ para aislar y apagar de forma     ║
║ quirúrgica el subespacio de Hilbert ruidoso en el espacio de Fock,           ║
║ eludiendo falsos positivos y garantizando la continuidad geodésica del       ║
║ vaciado de concreto en los frentes de trabajo sanos de la obra civil.        ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================════════════════════════════════════════════════
I. DEFINICIONES DE LA CIRUGÍA TOPOLÓGICA (Cohomología de Čech y Variedades)
================================════════════════════════════════════════════════

Definición 1 (Cubrimiento de Čech y 1-Cocadenas de Obstrucción):
  Sea $K$ el complejo simplicial del presupuesto y $\mathcal{U} = \{U_i\}_{i=1}^M$
  un cubrimiento abierto del mismo. Las señales de telemetría local de transductores
  se modelan como secciones locales del haz celular $\mathcal{F}$. El desfase analógico
  entre intersecciones adyacentes $U_i \cap U_j$ define una 1-cocadena de Čech:
  $$\delta_{\mathrm{\check{C}ech}} \in C^1(\mathcal{U}; \, \mathcal{F}) \quad \implies \quad (\delta_{\mathrm{\check{C}ech}} \phi)_{ij} = \phi_i|_{U_i \cap U_j} - \phi_j|_{U_i \cap U_j}$$
  La obstrucción local a la convalidación global de datos (mismatch) se mide por la
  no-trivialidad del primer grupo de cohomología de Čech:
  $$\check{H}^1(\mathcal{U}; \, \mathcal{F}) = \ker(\delta_{\mathrm{\check{C}ech}}^1) / \operatorname{im}(\delta_{\mathrm{\check{C}ech}}^0) \neq \mathbf{0}$$
  El Laplaciano de Čech simétrico semidefinido positivo (SPSD) se expresa como:
  $$\mathbf{\Delta}_{\mathrm{\check{C}ech}} = \delta_{\mathrm{\check{C}ech}}^\top \delta_{\mathrm{\check{C}ech}}$$

Definición 2 (Pullback de Deformación Anisotrópica en de Rham):
  Si la discrepancia espectral supera la cota de Lipschitz del Triple Espectral de Connes,
  el sistema deforma de forma anisotrópica la métrica de fondo $\mathbf{G}$ (el tensor de
  conductancias de la red) aplicando un pullback de amortiguamiento perimetral:
  $$\mathbf{G}_{\mathrm{surgical}} = \mathbf{G} \odot (\mathbf{I} - \mathbf{P}_{\mathrm{noisy}})$$
  Donde $\mathbf{P}_{\mathrm{noisy}}$ es el proyector ortogonal sobre la carta local del
  sensor ruidoso, de-multiplicando los acoplamientos del nodo ruidoso hasta el límite de
  Wilkinson ($\approx 10^{-15}$), aislando la anomalía analógica sin comprometer la sismorresistencia global.

Definición 3 (Reducción de Fock y Conservación de Traza de von Neumann):
  Para aniquilar los polaritones de alucinación excitados por el ruido en Fock, el motor
  realiza un traceout cuántico parcial sobre el subespacio del nodo aislado, proyectando
  el operador densidad mixto $\rho \in \mathcal{D}(\mathcal{H})$ hacia el vacío de regularidad:
  $$\rho_{\mathrm{surgery}} = \operatorname{Tr}_{\mathrm{isolated}}\left( \mathbf{P}_{\mathrm{surg}} \rho \mathbf{P}_{\mathrm{surg}}^\top \right) \oplus \rho_{\mathrm{vacuum}}$$
  Garantizando síncronamente en la FPU la preservación de la traza de von Neumann y la
  pasividad del Hamiltoniano de Lyapunov en el resto de la Ciudadela:
  $$\operatorname{Tr}(\rho_{\mathrm{surgery}}) \equiv 1.0 \quad \implies \quad \|\operatorname{Tr}(\rho_{\mathrm{surgery}}) - 1.0\| \le \varepsilon_{\mathrm{Wilkinson}}$$

================================════════════════════════════════════════════════
II. AXIOMÁTICA INMUNILÓGICA DE CONTROL DE RUIDO (Leyes de Estabilidad)
================================════════════════════════════════════════════════

Axioma I (Principio de Conservación de la Conexidad de Fiedler):
  La cirugía de de Rham es convalidada única y exclusivamente si la conectividad algebraica
  del subcomplejo remanente se mantiene estrictamente positiva, asegurando que el presupuesto
  remanente conserve una única componente conexa ($\beta_0 = 1$):
  $$\lambda_2(\mathbf{L}_{\mathrm{remSub}}) \ge \tau_{\mathrm{Fiedler}} \quad \Longleftrightarrow \quad \beta_0 \equiv \dim H^0(K_{\mathrm{remanente}}; \, \mathbb{Z}) = 1$$
  Si la remoción del nodo comprometido fragmenta el complejo, el sistema colapsa síncronamente a VETOED.

Axioma II (Axioma de Confinamiento Espectral de de Rham-Čech):
  Toda señal perimetral se considera regular (coherente) si la obstrucción local se
  encuentra confinada por la cota de Lipschitz del operador de Dirac de Connes:
  $$\check{H}^1(\mathcal{U}; \, \mathcal{F}) \le L_{\max} \cdot \tau_{\mathrm{margin}} \quad \implies \quad \mathtt{heyting\_verdict} = \mathtt{COHERENT}$$

Axioma III (Teorema de Actuación Ciber-Física Crowbar de la Sonda Čech):
  Ante el colapso de Heyting al Supremo de veto ($\top$), la subrutina local isVerdictCoherent()
  del microcontrolador ESP32 despacha síncronamente la ISR en IRAM en menos de 400 ns:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
  Disparando el tiristor BT151 (Crowbar) para paralizar mecánicamente la obra en el milisegundo cero,
  protegiendo patrimonialmente el capital antes de liquidar la transacción ante el SECOP II.

================================════════════════════════════════════════════════
III. INVARIANTES DE PRECISIÓN DE PUNTO FLOTANTE (FPU Secure)
================================════════════════════════════════════════════════

Invariante I (Invarianza Simpléctica y Conservación de de Rham-Lyapunov):
  La evolución de la trayectoria de control conjunta $\mathbf{\Psi}(t) = (\mathbf{p}, \, \rho)^\top$ satisface la
  desigualdad de Clausius-Duhem y la contracción de Lyapunov en la FPU:
  $$\dot{\mathcal{H}}(\mathbf{\Psi}) = \nabla \mathcal{H}(\mathbf{\Psi})^top \left( \mathcal{J}(\mathbf{\Psi}) - \mathcal{R}(\mathbf{\Psi}) \right) \nabla \mathcal{H}(\mathbf{\Psi}) \le \tau_{\mathrm{Lyapunov}}$$
  Donde $\tau_{\mathrm{Lyapunov}} = 10^{-12}$ es la cota elástica de deriva en punto flotante de 64 bits.

Invariante II (Estabilidad de de Rham-Liouville del Espacio de Fase):
  El Jacobiano de transición $M$ de los solucionadores físicos debe preservar la 2-forma simpléctica canónica,
  garantizando la nulidad de la divergencia del flujo de fase (conservación del volumen de Liouville):
  $$\operatorname{div}(\dot{x}) \equiv 0 \quad \land \quad M^\top \Omega M = \Omega$$

Invariante III (Inmutabilidad del Pasaporte y Sello de Sesión SHA-256):
  Para prevenir inyecciones de estado o ataques de-normalización intermedia, el motor genera
  un sello inmutable unívoco para congelar la sesión en RAM en cada ciclo OODA:
  $$\mathtt{cryptographic\_seal} := \operatorname{SHA-256}\left(\delta_{\mathrm{\check{C}ech}} \oplus \mathbf{G}_{\mathrm{surgical}} \oplus \lambda_2 \oplus H_{\mathrm{ext}}\right)$$
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la


logger = logging.getLogger("APU.Physics.TopologicalSurgeryCech")


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS, ESPECTRALES Y DE SEGURIDAD
# ═══════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SQRT_EPS: Final[float] = float(np.sqrt(_MACHINE_EPS))

_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_STD_NS: Final[float] = 3.2
_CROWBAR_LATENCY_MIN_NS: Final[float] = 380.0
_CROWBAR_LATENCY_MAX_NS: Final[float] = 420.0
_GPIO_CROWBAR_PIN: Final[int] = 14

_SCHEMA_VERSION: Final[str] = (
    "2.1.0-Doctoral-Cech-Cohomology-Anisotropic-Hodge-FailClosed"
)

_DOMAIN_SESSION: Final[bytes] = b"APU/TOPOLOGICAL-CECH-SURGERY/SESSION/v2.1"
_DOMAIN_TELEMETRY: Final[bytes] = b"APU/TOPOLOGICAL-CECH-SURGERY/TELEMETRY/v2.1"
_DOMAIN_OVERRIDE: Final[bytes] = b"APU/TOPOLOGICAL-CECH-SURGERY/OVERRIDE/v2.1"
_DOMAIN_ENGINE: Final[bytes] = b"APU/TOPOLOGICAL-CECH-SURGERY/IDENTITY/v2.1"

_SPD_FLOOR: Final[float] = 1e-12
_HERMITIAN_REL_TOL: Final[float] = 1e-10
_KERNEL_REL_TOL: Final[float] = 1e-10
_IMAG_TOL: Final[float] = 1e-12
_SURGERY_CUT: Final[float] = 1e-15
_SURGERY_DIAG_FLOOR: Final[float] = 1e-12
_ACTIVE_ABS_FLOOR: Final[float] = 1e-11
_ACTIVE_REL_FLOOR: Final[float] = 1e-8
_COND_DEGRADED: Final[float] = 1e9
_COND_CAP: Final[float] = 1e12
_SHA256_HEX_LEN: Final[int] = 64
_OVERRIDE_TOKEN_MAX_BYTES: Final[int] = 4096
_DENSITY_REG: Final[float] = 1e-15


class HeytingVerdict(str, Enum):
    r"""
    Retículo de decisión de Heyting de tres niveles

        Ω₃ = { VETOED ≼ DEGRADED ≼ COHERENT }

    con meet ∧ = ínfimo (más restrictivo). La cirugía activa es ámbar
    (DEGRADED) y no se eleva a COHERENT ni siquiera con override: el
    override sólo autoriza el acto quirúrgico, no borra la cicatriz.
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"


_HEYTING_ORDER: Final[dict[str, int]] = {
    HeytingVerdict.VETOED.value: 0,
    HeytingVerdict.DEGRADED.value: 1,
    HeytingVerdict.COHERENT.value: 2,
}


class TopologicalSurgeryError(ValueError):
    """Error de canonización, álgebra lineal o invariante de Čech/Hodge."""


# ═══════════════════════════════════════════════════════════════════════════
# ESTADOS Y CERTIFICADOS PÚBLICOS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SurgeryState:
    r"""
    Estado cuántico y homológico tras la cirugía topológica de Čech.

    cech_laplacian_eigenvalues:
        Espectro real ordenado de Δ_Čech (Laplaciano del nervio).
    cohomological_mismatch:
        ‖δφ‖_{ℓ²} , obstrucción de pegado en H¹(𝒰, ℱ).
    surgical_metric_cond:
        κ(G_surgical) = λ_max⁺ / λ_min⁺.
    isolated_fock_trace:
        Masa de Fock remanente tr(P ρ P) *antes* de renormalizar.
    is_globally_coherent:
        λ₂(L_rem|activos) ≥ cota dura de Fiedler y |activos| > 1.
    fiedler_residual:
        λ₂ de la compresión de Rayleigh de L_rem a nodos activos.
    """

    cech_laplacian_eigenvalues: np.ndarray
    cohomological_mismatch: float
    surgical_metric_cond: float
    isolated_fock_trace: float
    is_globally_coherent: bool
    fiedler_residual: float

    schema_version: str = _SCHEMA_VERSION
    metric_spectral_gap: float = 0.0
    active_node_count: int = 0
    nerve_betti_0: int = 0
    cech_kernel_dimension: int = 0
    discarded_fock_mass: float = 0.0
    density_purity: float = math.nan
    hermitian_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class SurgeryCertificate:
    r"""Certificado criptográfico e inmunitario de la cirugía topológica."""

    heyting_verdict: str
    state: SurgeryState
    surgery_active: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    cryptographic_seal: str

    schema_version: str = _SCHEMA_VERSION
    session_digest: str = ""
    noisy_cover_id: int = -1
    lipschitz_bound: float = math.nan
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)
    engine_digest: str = ""
    override_present: bool = False
    override_valid: bool = False
    nerve_betti_0: int = 0
    active_nodes: Tuple[int, ...] = field(default_factory=tuple)


# ═══════════════════════════════════════════════════════════════════════════
# DOSSIERS INTERNOS DE FASE (objetos del morfismo anidado)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Phase1Dossier:
    """
    Expediente canonizado y congelado en la FASE 1 · OBSERVE.

    El token de override NUNCA viaja en claro: sólo su hash de dominio
    y banderas de presencia/validez morfológica.
    """

    boundary_matrix: np.ndarray
    global_metric_G: np.ndarray
    global_density_rho: np.ndarray
    local_signals: Dict[int, np.ndarray]
    lipschitz_bound_Lmax: float
    override_token_hash: Optional[str]
    override_present: bool
    override_is_valid: bool
    session_digest: str
    engine_digest: str
    density_purity: float
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase2Dossier:
    """Expediente auditado y operado en la FASE 2 · ORIENT/OPERATE."""

    phase1: Phase1Dossier
    cech_laplacian: np.ndarray
    cech_laplacian_eigenvalues: np.ndarray
    cohomological_mismatch: float
    noisy_cover_id: int
    surgery_required: bool
    G_surgical: np.ndarray
    rho_surgical: np.ndarray
    isolated_fock_trace: float
    discarded_fock_mass: float
    surgical_metric_cond: float
    metric_spectral_gap: float
    remnant_laplacian: np.ndarray
    active_nodes: Tuple[int, ...]
    fiedler_residual: float
    is_globally_coherent: bool
    nerve_betti_0: int
    cech_kernel_dimension: int
    hermitian_residual: float
    heyting_verdict: str
    diagnostics: Tuple[str, ...]


# ═══════════════════════════════════════════════════════════════════════════
# MOTOR DE CIRUGÍA TOPOLÓGICA DE ČECH
# ═══════════════════════════════════════════════════════════════════════════


class TopologicalSurgeryCech:
    r"""
    Motor de Cirugía Topológica de Čech.

    Identifica anomalías locales por ruido EMF analógico en cartas de Čech,
    deforma la métrica Riemanniana de fondo para aislar quirúrgicamente el
    canal ruidoso y proyecta el estado de Fock a un subespacio pasivo,
    evitando falsos positivos globales.

    Ciclo público, fases anidadas:

        execute_topological_surgery_cycle()
          └─ _phase1_observe_and_freeze()          # cierra FASE 1
               └─ _phase2_open_from_phase1()       # abre  FASE 2
                    └─ _phase2_audit_and_operate()
                         └─ _phase3_open_from_phase2()  # abre FASE 3
                              └─ _phase3_certify()

    Ante cualquier excepción no recuperable emite certificado fail-closed
    con veredicto VETOED e interlock activado.
    """

    def __init__(
        self,
        dimension_n: int,
        covering: Dict[int, List[int]],
        nominal_impedance: float = 50.0,
        safety_margin: float = 1.0,
        *,
        rng_seed: Optional[int] = None,
        fiedler_veto_threshold: float = 1e-4,
        fiedler_degraded_threshold: float = 1e-2,
    ) -> None:
        """
        Inicializa el motor de cirugía Čech.

        Args:
            dimension_n: Número de nodos del complejo simplicial.
            covering: Cubrimiento de Čech: cover_id → lista de nodos.
            nominal_impedance: Impedancia de referencia Z₀ (ohmios).
            safety_margin: Factor elástico de la cota de mismatch Lipschitz.
            rng_seed: Semilla opcional para reproducibilidad del jitter.
            fiedler_veto_threshold: Cota dura de conectividad algebraica.
            fiedler_degraded_threshold: Cota blanda de conectividad algebraica.
        """
        if dimension_n <= 0:
            raise TopologicalSurgeryError(
                "La dimensión del complejo simplicial debe ser estrictamente positiva."
            )

        if not math.isfinite(nominal_impedance) or nominal_impedance <= 0.0:
            raise TopologicalSurgeryError(
                "nominal_impedance debe ser finita y estrictamente positiva."
            )

        if not math.isfinite(safety_margin) or safety_margin <= 0.0:
            raise TopologicalSurgeryError(
                "safety_margin debe ser finito y estrictamente positivo."
            )

        if not math.isfinite(fiedler_veto_threshold) or fiedler_veto_threshold < 0.0:
            raise TopologicalSurgeryError(
                "fiedler_veto_threshold debe ser finita y no negativa."
            )

        if (
            not math.isfinite(fiedler_degraded_threshold)
            or fiedler_degraded_threshold < 0.0
        ):
            raise TopologicalSurgeryError(
                "fiedler_degraded_threshold debe ser finita y no negativa."
            )

        if not isinstance(covering, dict) or len(covering) == 0:
            raise TopologicalSurgeryError(
                "covering debe ser un diccionario no vacío de cartas de Čech."
            )

        validated_covering: Dict[int, Tuple[int, ...]] = {}
        covered_nodes: set[int] = set()

        for key, nodes in covering.items():
            try:
                cover_id = int(key)
            except (TypeError, ValueError) as exc:
                raise TopologicalSurgeryError(
                    "Toda clave del cubrimiento de Čech debe ser entera."
                ) from exc

            if nodes is None:
                raise TopologicalSurgeryError(
                    f"La carta de Čech {cover_id} no puede ser None."
                )

            canonical_nodes: List[int] = []
            seen: set[int] = set()

            for node in nodes:
                try:
                    node_id = int(node)
                except (TypeError, ValueError) as exc:
                    raise TopologicalSurgeryError(
                        f"Nodo inválido en la carta de Čech {cover_id}."
                    ) from exc

                if node_id < 0 or node_id >= dimension_n:
                    raise TopologicalSurgeryError(
                        f"Nodo {node_id} fuera del complejo simplicial de "
                        f"dimensión {dimension_n}."
                    )

                if node_id in seen:
                    continue
                seen.add(node_id)
                canonical_nodes.append(node_id)

            if not canonical_nodes:
                logger.warning("Carta Čech %s vacía tras canonización.", cover_id)

            validated_covering[cover_id] = tuple(canonical_nodes)
            covered_nodes.update(canonical_nodes)

        self._n: Final[int] = int(dimension_n)
        self._covering: Final[Dict[int, Tuple[int, ...]]] = validated_covering
        self._cover_ids: Final[Tuple[int, ...]] = tuple(sorted(validated_covering.keys()))
        self._covered_nodes: Final[Tuple[int, ...]] = tuple(sorted(covered_nodes))
        self._covering_is_total: Final[bool] = len(covered_nodes) == self._n

        self._z0: Final[float] = float(nominal_impedance)
        self._safety_margin: Final[float] = float(safety_margin)

        self._fiedler_veto: Final[float] = float(fiedler_veto_threshold)
        self._fiedler_degraded: Final[float] = float(fiedler_degraded_threshold)

        self._reg: Final[float] = max(_DENSITY_REG, _MACHINE_EPS)
        self._rng: Final[np.random.Generator] = np.random.default_rng(rng_seed)
        self._engine_digest: Final[str] = self._identity_digest()

        if not self._covering_is_total:
            logger.warning(
                "Cubrimiento de Čech no total: cubre %d/%d nodos.",
                len(covered_nodes),
                self._n,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # UTILIDADES CANÓNICAS (hash, congelación, retículo, espectro)
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
        """Actualiza SHA-256 con arreglo numpy C-contiguo y dtype explícito."""
        cls._hash_update_bytes(sha, name.encode("utf-8"))

        arr = np.ascontiguousarray(array)

        sha.update(len(arr.shape).to_bytes(4, "little", signed=False))
        for dim in arr.shape:
            sha.update(int(dim).to_bytes(8, "little", signed=False))

        cls._hash_update_bytes(sha, str(arr.dtype).encode("utf-8"))
        sha.update(arr.tobytes())

    @staticmethod
    def _freeze_array(array: np.ndarray, dtype: Any) -> np.ndarray:
        """Copia defensiva C-contigua e inmutable (write-flag desactivado)."""
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _freeze_signal_map(
        signals: Mapping[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """Congela cada señal local como vector float64 write-protected."""
        frozen: Dict[int, np.ndarray] = {}
        for cover_id, signal in signals.items():
            frozen[int(cover_id)] = TopologicalSurgeryCech._freeze_array(
                np.asarray(signal).reshape(-1),
                np.float64,
            )
        return frozen

    @staticmethod
    def _heyting_meet(verdicts: Sequence[str]) -> str:
        r"""Ínfimo en Ω₃: el veredicto más restrictivo (VETOED ≼ ... ≼ COHERENT)."""
        if not verdicts:
            return HeytingVerdict.COHERENT.value
        rank = min(_HEYTING_ORDER.get(str(v), 0) for v in verdicts)
        inverse = {
            0: HeytingVerdict.VETOED.value,
            1: HeytingVerdict.DEGRADED.value,
            2: HeytingVerdict.COHERENT.value,
        }
        return inverse[rank]

    @staticmethod
    def _spectral_abs_tol(eigenvalues: np.ndarray, rel: float) -> float:
        """Tolerancia absoluta dimensionada por el radio espectral."""
        if eigenvalues.size == 0:
            return max(_SPD_FLOOR, rel)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        return max(_SPD_FLOOR, rel * scale, 1000.0 * _MACHINE_EPS * scale)

    @staticmethod
    def _finite_norm(matrix: np.ndarray) -> float:
        """Norma de Frobenius finita; 0 si el tensor es vacío."""
        if matrix.size == 0:
            return 0.0
        value = float(la.norm(matrix, ord="fro"))
        if not math.isfinite(value):
            return math.inf
        return value

    def _identity_digest(self) -> str:
        """Huella de la identidad metrológica del motor (no de la sesión)."""
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_ENGINE)
        payload = (
            f"n={self._n}|z0={self._z0:.17e}|sm={self._safety_margin:.17e}|"
            f"fv={self._fiedler_veto:.17e}|fd={self._fiedler_degraded:.17e}|"
            f"covers={len(self._cover_ids)}|schema={_SCHEMA_VERSION}"
        )
        self._hash_update_bytes(sha, payload.encode("utf-8"))
        for cover_id in self._cover_ids:
            self._hash_update_bytes(sha, str(int(cover_id)).encode("utf-8"))
            nodes = np.asarray(self._covering[cover_id], dtype=np.int64)
            self._hash_update_array(sha, f"cover_{cover_id}", nodes)
        return sha.hexdigest()

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1 · OBSERVE
    #
    # Objeto inicial  : tensores crudos de frontera, métrica, densidad,
    #                   señales locales, cota de Lipschitz y token.
    # Objeto terminal : Phase1Dossier (inmutable, hasheado, SPD, traza 1).
    # Morfismo de cierre:
    #   _phase1_observe_and_freeze  →  _phase2_open_from_phase1
    #
    # Invariantes:
    #   (I1) B ∈ M_{M×n}(ℝ), finita, n columnas (nodos).
    #   (I2) G ∈ M_{M×M}(ℝ), G = Gᵀ ≽ ε I  (proyección espectral a SPD).
    #   (I3) ρ ∈ M_{n×n}(ℂ), ρ = ρᴴ ≽ 0, tr ρ = 1.
    #   (I4) señales locales finitas, indexadas por cover_id.
    #   (I5) L_max > 0 finito.
    #   (I6) override ∈ {∅} ∪ Digest₆₄; nunca se serializa en claro.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase1_canonical_boundary_matrix(self, boundary_matrix: Any) -> np.ndarray:
        r"""Valida la matriz de incidencia real δ : C_1 → C_0 con n columnas."""
        try:
            boundary = np.asarray(boundary_matrix, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TopologicalSurgeryError(
                "boundary_matrix no puede convertirse a float64."
            ) from exc

        if boundary.ndim != 2:
            raise TopologicalSurgeryError("boundary_matrix debe ser bidimensional.")

        if boundary.shape[0] == 0:
            raise TopologicalSurgeryError(
                "boundary_matrix no puede tener cero canales."
            )

        if boundary.shape[1] != self._n:
            raise TopologicalSurgeryError(
                f"boundary_matrix debe tener {self._n} columnas (nodos)."
            )

        if not np.all(np.isfinite(boundary)):
            raise TopologicalSurgeryError(
                "boundary_matrix contiene valores no finitos."
            )

        return np.ascontiguousarray(boundary, dtype=np.float64)

    def _phase1_project_metric_spd(
        self,
        metric: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        r"""
        Proyección espectral sobre el cono SPD real:

            G ↦ G_spd = ∑_i max(λ_i, ε) |u_i⟩⟨u_i|
        """
        diagnostics: List[str] = []
        symmetric = 0.5 * (metric + metric.T)

        try:
            eigvals, eigvecs = la.eigh(symmetric)
        except la.LinAlgError as exc:
            raise TopologicalSurgeryError(
                "No fue posible diagonalizar global_metric_G."
            ) from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise TopologicalSurgeryError("Autovalores de global_metric_G inválidos.")

        eigvals_real = np.real(eigvals)
        abs_tol = self._spectral_abs_tol(eigvals_real, 1e-12)
        negative_count = int(np.sum(eigvals_real < -abs_tol))
        if negative_count > 0:
            diagnostics.append(
                f"Métrica con {negative_count} autovalores negativos; "
                "proyección espectral a SPD."
            )

        floor = max(_SPD_FLOOR, _MACHINE_EPS)
        eigvals_clamped = np.maximum(eigvals_real, floor)
        reconstructed = (eigvecs * eigvals_clamped) @ eigvecs.T
        reconstructed = 0.5 * (reconstructed + reconstructed.T)

        if not np.all(np.isfinite(reconstructed)):
            raise TopologicalSurgeryError("Métrica regularizada no finita.")

        return reconstructed, tuple(diagnostics)

    def _phase1_canonical_metric(
        self,
        global_metric_G: Any,
        expected_dim: int,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        """Valida métrica real simétrica finita y la proyecta a SPD."""
        diagnostics: List[str] = []

        if np.iscomplexobj(global_metric_G):
            try:
                metric_complex = np.asarray(global_metric_G, dtype=np.complex128)
            except (TypeError, ValueError) as exc:
                raise TopologicalSurgeryError(
                    "global_metric_G no puede convertirse a complex128."
                ) from exc

            if not np.all(np.isfinite(metric_complex)):
                raise TopologicalSurgeryError(
                    "global_metric_G contiene valores no finitos."
                )

            imag_norm = (
                float(np.max(np.abs(metric_complex.imag)))
                if metric_complex.size
                else 0.0
            )
            if imag_norm > _IMAG_TOL:
                raise TopologicalSurgeryError(
                    "global_metric_G debe ser real dentro de tolerancia numérica."
                )
            if imag_norm > 0.0:
                diagnostics.append(
                    f"Parte imaginaria residual de G descartada (‖Im‖_∞={imag_norm:.3e})."
                )
            metric = np.ascontiguousarray(metric_complex.real, dtype=np.float64)
        else:
            try:
                metric = np.asarray(global_metric_G, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TopologicalSurgeryError(
                    "global_metric_G no puede convertirse a float64."
                ) from exc

        if metric.ndim != 2 or metric.shape != (expected_dim, expected_dim):
            raise TopologicalSurgeryError(
                f"global_metric_G debe tener forma ({expected_dim},{expected_dim})."
            )

        if not np.all(np.isfinite(metric)):
            raise TopologicalSurgeryError(
                "global_metric_G contiene valores no finitos."
            )

        metric_spd, spd_diag = self._phase1_project_metric_spd(metric)
        diagnostics.extend(spd_diag)
        return metric_spd, tuple(diagnostics)

    def _phase1_canonical_density_matrix(
        self,
        global_density_rho: Any,
    ) -> Tuple[np.ndarray, float, Tuple[str, ...]]:
        r"""
        Proyecta ρ al simplejo de densidades:

            ρ ↦ (ρ + ρᴴ)/2  →  clamp(σ, 0)  →  normaliza tr = 1.

        Retorna (ρ_proj, pureza tr(ρ²), diagnósticos).
        """
        diagnostics: List[str] = []

        try:
            density = np.asarray(global_density_rho, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise TopologicalSurgeryError(
                "global_density_rho no puede convertirse a complex128."
            ) from exc

        if density.ndim != 2 or density.shape != (self._n, self._n):
            raise TopologicalSurgeryError(
                f"global_density_rho debe tener forma ({self._n},{self._n})."
            )

        if not np.all(np.isfinite(density)):
            raise TopologicalSurgeryError(
                "global_density_rho contiene valores no finitos."
            )

        density = 0.5 * (density + density.conj().T)

        try:
            eigvals, eigvecs = la.eigh(density)
        except la.LinAlgError as exc:
            raise TopologicalSurgeryError(
                "No fue posible diagonalizar global_density_rho."
            ) from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise TopologicalSurgeryError(
                "Autovalores de global_density_rho inválidos."
            )

        eigvals_real = np.real(eigvals)
        abs_tol = self._spectral_abs_tol(eigvals_real, 1e-12)
        negative_count = int(np.sum(eigvals_real < -abs_tol))
        if negative_count > 0:
            diagnostics.append(
                f"Densidad con {negative_count} autovalores negativos; "
                "proyección al cono PSD."
            )

        eigvals_clamped = np.maximum(eigvals_real, 0.0)
        mass = float(np.sum(eigvals_clamped))

        if mass <= self._reg:
            diagnostics.append(
                "Traza de ρ numéricamente nula; se sustituye por el estado "
                "maximally mixed I/n."
            )
            eigvals_clamped = np.full(self._n, 1.0 / float(self._n), dtype=np.float64)
        else:
            eigvals_clamped = eigvals_clamped / mass

        eigvals_clamped = np.maximum(eigvals_clamped, self._reg)
        eigvals_clamped = eigvals_clamped / float(np.sum(eigvals_clamped))

        projected = (eigvecs * eigvals_clamped) @ eigvecs.conj().T
        projected = 0.5 * (projected + projected.conj().T)

        if not np.all(np.isfinite(projected)):
            raise TopologicalSurgeryError("Matriz de densidad proyectada no finita.")

        purity = float(np.real(np.trace(projected @ projected)))
        if not math.isfinite(purity):
            purity = math.nan

        return projected, purity, tuple(diagnostics)

    def _phase1_canonical_local_signals(
        self,
        local_signals: Any,
    ) -> Tuple[Dict[int, np.ndarray], Tuple[str, ...]]:
        """Canoniza señales locales de telemetría por carta de Čech."""
        diagnostics: List[str] = []

        if not isinstance(local_signals, dict):
            raise TopologicalSurgeryError(
                "local_signals debe ser un diccionario cover_id → señal."
            )

        coerced: Dict[int, Any] = {}
        for key, value in local_signals.items():
            try:
                coerced[int(key)] = value
            except (TypeError, ValueError) as exc:
                raise TopologicalSurgeryError(
                    "Claves de local_signals deben ser enteras."
                ) from exc

        out: Dict[int, np.ndarray] = {}
        for cover_id in self._cover_ids:
            if cover_id not in coerced:
                out[cover_id] = np.array([], dtype=np.float64)
                diagnostics.append(
                    f"Señal ausente en carta Čech {cover_id}; se registra vacía."
                )
                continue

            try:
                signal = np.asarray(coerced[cover_id], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TopologicalSurgeryError(
                    f"Señal local inválida en la carta Čech {cover_id}."
                ) from exc

            if signal.ndim == 0:
                signal = signal.reshape(1)
            else:
                signal = signal.reshape(-1)

            if not np.all(np.isfinite(signal)):
                raise TopologicalSurgeryError(
                    f"Señal local no finita en la carta Čech {cover_id}."
                )

            out[cover_id] = np.ascontiguousarray(signal, dtype=np.float64)

        extra = set(coerced.keys()) - set(self._cover_ids)
        if extra:
            diagnostics.append(
                f"Señales con cover_id ajenos al cubrimiento ignoradas: {sorted(extra)}."
            )

        return out, tuple(diagnostics)

    @staticmethod
    def _phase1_canonical_lipschitz_bound(lipschitz_bound_Lmax: Any) -> float:
        """Valida cota de Lipschitz como escalar finito estrictamente positivo."""
        try:
            value = float(lipschitz_bound_Lmax)
        except (TypeError, ValueError) as exc:
            raise TopologicalSurgeryError(
                "lipschitz_bound_Lmax debe ser escalar."
            ) from exc

        if not math.isfinite(value) or value <= 0.0:
            raise TopologicalSurgeryError(
                "lipschitz_bound_Lmax debe ser finito y estrictamente positivo."
            )

        return value

    @staticmethod
    def _phase1_normalize_override_token(override_token: Any) -> Optional[str]:
        """Normaliza el token a str sin revelar contenido; extraño ≡ ausencia."""
        if override_token is None:
            return None

        if isinstance(override_token, bytes):
            if len(override_token) > _OVERRIDE_TOKEN_MAX_BYTES:
                return None
            try:
                token = override_token.decode("utf-8")
            except UnicodeDecodeError:
                return None
        elif isinstance(override_token, str):
            if len(override_token.encode("utf-8", errors="ignore")) > (
                _OVERRIDE_TOKEN_MAX_BYTES
            ):
                return None
            token = override_token
        else:
            return None

        token = token.strip()
        return token or None

    @staticmethod
    def _phase1_token_is_sha256_hex(token: str) -> bool:
        """Morfología de digest SHA-256: exactamente 32 bytes en hex."""
        if len(token) != _SHA256_HEX_LEN:
            return False
        try:
            raw = bytes.fromhex(token)
        except ValueError:
            return False
        return len(raw) == 32

    def _phase1_hash_override_token(self, override_token: Optional[str]) -> Optional[str]:
        """Hashea el token de override con separación de dominio."""
        token = self._phase1_normalize_override_token(override_token)
        if token is None:
            return None

        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_OVERRIDE)
        self._hash_update_bytes(sha, token.encode("utf-8"))
        return sha.hexdigest()

    def _phase2_is_valid_override_token(self, override_token: Optional[str]) -> bool:
        """
        API de compatibilidad: validez morfológica SHA-256 hexadecimal.

        La validez operativa se congela en FASE 1 y viaja como bandera
        en el expediente (el token en claro no sobrevive al freeze).
        """
        token = self._phase1_normalize_override_token(override_token)
        if token is None:
            return False
        return self._phase1_token_is_sha256_hex(token)

    def _phase1_generate_session_digest(
        self,
        *,
        boundary_matrix: np.ndarray,
        global_metric_G: np.ndarray,
        global_density_rho: np.ndarray,
        local_signals: Dict[int, np.ndarray],
        lipschitz_bound_Lmax: float,
        override_token_hash: Optional[str],
        override_is_valid: bool,
    ) -> str:
        """Genera firma SHA-256 de sesión con separación de dominio."""
        sha = hashlib.sha256()

        self._hash_update_domain(sha, _DOMAIN_SESSION)
        self._hash_update_bytes(sha, self._engine_digest.encode("utf-8"))
        self._hash_update_array(sha, "boundary_matrix", boundary_matrix)
        self._hash_update_array(sha, "global_metric_G", global_metric_G)
        self._hash_update_array(sha, "global_density_rho", global_density_rho)

        for cover_id in self._cover_ids:
            self._hash_update_bytes(sha, str(int(cover_id)).encode("utf-8"))
            nodes = np.asarray(self._covering[cover_id], dtype=np.int64)
            self._hash_update_array(sha, f"cover_nodes_{cover_id}", nodes)

        for cover_id in self._cover_ids:
            self._hash_update_bytes(sha, str(int(cover_id)).encode("utf-8"))
            self._hash_update_array(
                sha,
                f"local_signal_{cover_id}",
                local_signals[cover_id],
            )

        self._hash_update_bytes(sha, f"{float(lipschitz_bound_Lmax):.17e}".encode("utf-8"))

        if override_token_hash is None:
            self._hash_update_bytes(sha, b"override:absent")
        else:
            self._hash_update_bytes(sha, b"override:present")
            self._hash_update_bytes(sha, override_token_hash.encode("utf-8"))
            self._hash_update_bytes(
                sha,
                b"override:valid" if override_is_valid else b"override:invalid",
            )

        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        return sha.hexdigest()

    def _phase1_observe_and_freeze(
        self,
        *,
        boundary_matrix: np.ndarray,
        global_metric_G: np.ndarray,
        global_density_rho: np.ndarray,
        local_signals: Dict[int, np.ndarray],
        lipschitz_bound_Lmax: float,
        override_token: Optional[str],
    ) -> SurgeryCertificate:
        r"""
        FASE 1 · OBSERVE  (morfismo de cierre).

        Canoniza, regulariza y congela el expediente de frontera. El valor de
        retorno no es un terminal de Fase 1: es el compuesto

            Φ₁₂ ∘ Observe  :  datos crudos  →  SurgeryCertificate

        realizado por la continuación formal `_phase2_open_from_phase1`,
        que es el primer morfismo de la FASE 2 · ORIENT/OPERATE.
        """
        diagnostics: List[str] = []

        boundary = self._phase1_canonical_boundary_matrix(boundary_matrix)

        metric, metric_diag = self._phase1_canonical_metric(
            global_metric_G,
            expected_dim=int(boundary.shape[0]),
        )
        diagnostics.extend(metric_diag)

        density, purity, density_diag = self._phase1_canonical_density_matrix(
            global_density_rho
        )
        diagnostics.extend(density_diag)

        signals, signal_diag = self._phase1_canonical_local_signals(local_signals)
        diagnostics.extend(signal_diag)

        lipschitz = self._phase1_canonical_lipschitz_bound(lipschitz_bound_Lmax)

        if not self._covering_is_total:
            diagnostics.append(
                f"Cubrimiento no total: {len(self._covered_nodes)}/{self._n} nodos."
            )

        token = self._phase1_normalize_override_token(override_token)
        token_hash = self._phase1_hash_override_token(token)
        override_present = token is not None
        override_is_valid = bool(
            token is not None and self._phase1_token_is_sha256_hex(token)
        )
        if override_present and override_is_valid:
            diagnostics.append(
                "Override presente; hash de token incorporado a sesión."
            )
        elif override_present:
            diagnostics.append(
                "Override presente pero morfológicamente inválido."
            )

        token = None
        override_token = None

        session_digest = self._phase1_generate_session_digest(
            boundary_matrix=boundary,
            global_metric_G=metric,
            global_density_rho=density,
            local_signals=signals,
            lipschitz_bound_Lmax=lipschitz,
            override_token_hash=token_hash,
            override_is_valid=override_is_valid,
        )

        dossier = Phase1Dossier(
            boundary_matrix=self._freeze_array(boundary, np.float64),
            global_metric_G=self._freeze_array(metric, np.float64),
            global_density_rho=self._freeze_array(density, np.complex128),
            local_signals=self._freeze_signal_map(signals),
            lipschitz_bound_Lmax=lipschitz,
            override_token_hash=token_hash,
            override_present=override_present,
            override_is_valid=override_is_valid,
            session_digest=session_digest,
            engine_digest=self._engine_digest,
            density_purity=purity,
            diagnostics=tuple(diagnostics),
        )

        logger.info(
            "FASE 1 OBSERVE: expediente Čech congelado. B=%s, covers=%d, digest=%s",
            boundary.shape,
            len(self._cover_ids),
            session_digest[:16],
        )

        # ── continuación anidada: el terminal de FASE 1 es el inicial de FASE 2
        return self._phase2_open_from_phase1(dossier)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2 · ORIENT / OPERATE
    #
    # Objeto inicial  : Phase1Dossier  (entregado por _phase1_observe_and_freeze)
    # Objeto terminal : Phase2Dossier
    # Morfismo de apertura: _phase2_open_from_phase1
    # Morfismo de cierre  : _phase2_audit_and_operate → _phase3_open_from_phase2
    #
    # Invariantes:
    #   (J1) Δ_Čech = δᵀδ sobre el 1-esqueleto del nervio N(𝒰).
    #   (J2) mismatch = ‖δφ‖_{ℓ²} con φ_i = sección local en U_i.
    #   (J3) cirugía ⇔ mismatch > L_max · safety_margin.
    #   (J4) G_surgical = Π_cut G Π_cut  (Π_cut diagonal sobre aristas ruidosas).
    #   (J5) ρ_surgical = Lüders_P(ρ) renormalizado; masa = tr(PρP).
    #   (J6) L_rem = AᵀA, A = G^{1/2} B; λ₂ sobre nodos activos.
    #   (J7) veredicto = ⋀ átomos de Heyting.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase2_open_from_phase1(self, dossier: Phase1Dossier) -> SurgeryCertificate:
        r"""
        Inicio formal de la FASE 2 · ORIENT/OPERATE.

        Continuación estricta del terminal de FASE 1: recibe el expediente
        congelado y abre la cohomología de Čech, la cirugía anisotrópica
        y la auditoría de Hodge remanente.
        """
        if not isinstance(dossier, Phase1Dossier):
            raise TopologicalSurgeryError(
                "FASE 2 exige un Phase1Dossier emitido por FASE 1."
            )
        return self._phase2_audit_and_operate(dossier)

    def _phase2_signal_as_nodal_field(
        self,
        cover_id: int,
        signal: np.ndarray,
    ) -> np.ndarray:
        r"""
        Interpreta la señal de la carta U_i como 0-cochain nodal.

        Convención:
          • len(φ) = n            → campo global, se usa tal cual.
          • len(φ) = |U_i|        → valores sobre los nodos de la carta, en orden.
          • otro                  → relleno parcial de los primeros min(|φ|,|U_i|)
                                    nodos de la carta; el resto queda NaN.
        """
        field = np.full(self._n, np.nan, dtype=np.float64)
        nodes = self._covering[cover_id]
        data = np.asarray(signal, dtype=np.float64).reshape(-1)

        if data.size == self._n:
            return np.ascontiguousarray(data, dtype=np.float64)

        if data.size == 0 or len(nodes) == 0:
            return field

        limit = min(data.size, len(nodes))
        for index in range(limit):
            field[int(nodes[index])] = float(data[index])
        return field

    def _phase2_build_nodal_fields(
        self,
        local_signals: Mapping[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """Construye el diccionario cover_id → campo nodal (n,)."""
        fields: Dict[int, np.ndarray] = {}
        for cover_id in self._cover_ids:
            signal = local_signals.get(cover_id, np.array([], dtype=np.float64))
            fields[cover_id] = self._phase2_signal_as_nodal_field(cover_id, signal)
        return fields

    def _phase2_intersection_nodes(
        self,
        cover_i: int,
        cover_j: int,
    ) -> Tuple[int, ...]:
        """Nodos de U_i ∩ U_j en orden canónico."""
        left = set(self._covering[cover_i])
        right = set(self._covering[cover_j])
        return tuple(sorted(left & right))

    def _phase2_coboundary_energy(
        self,
        field_i: np.ndarray,
        field_j: np.ndarray,
        intersection: Sequence[int],
    ) -> float:
        r"""
        Energía de 1-cocadena sobre una intersección:

            ∑_{v ∈ U_i ∩ U_j} (φ_j(v) − φ_i(v))²
        """
        if not intersection:
            return 0.0

        nodes = np.asarray(intersection, dtype=np.int64)
        left = field_i[nodes]
        right = field_j[nodes]
        mask = np.isfinite(left) & np.isfinite(right)
        if not np.any(mask):
            return 0.0

        diff = right[mask] - left[mask]
        energy = float(np.dot(diff, diff))
        if not math.isfinite(energy):
            raise TopologicalSurgeryError("Mismatch de Čech no finito.")
        return max(0.0, energy)

    def _phase2_compute_cech_laplacian_internal(
        self,
        local_signals: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, float, np.ndarray, Tuple[str, ...]]:
        r"""
        Laplaciano de Čech (nervio ponderado) y obstrucción ‖δφ‖_{ℓ²}.

        Para cada arista del nervio {i, j} con I = U_i ∩ U_j ≠ ∅:

            w_{ij} = |I|
            Δ_{ii} += w_{ij},   Δ_{ij} = −w_{ij}

        que es exactamente Δ = δᵀδ si δ tiene, por cada v ∈ I, la fila
        e_j − e_i. El mismatch es la norma ℓ² de esas filas aplicadas a φ.

        Retorna (Δ_Čech, mismatch, energía incidente por carta, diagnósticos).
        """
        diagnostics: List[str] = []
        cover_count = len(self._cover_ids)
        laplacian = np.zeros((cover_count, cover_count), dtype=np.float64)
        incident_energy = np.zeros(cover_count, dtype=np.float64)
        mismatch_sq = 0.0

        fields = self._phase2_build_nodal_fields(local_signals)

        for idx_i, cover_i in enumerate(self._cover_ids):
            for idx_j in range(idx_i + 1, cover_count):
                cover_j = self._cover_ids[idx_j]
                intersection = self._phase2_intersection_nodes(cover_i, cover_j)
                if not intersection:
                    continue

                energy = self._phase2_coboundary_energy(
                    fields[cover_i],
                    fields[cover_j],
                    intersection,
                )
                mismatch_sq += energy
                incident_energy[idx_i] += energy
                incident_energy[idx_j] += energy

                weight = float(len(intersection))
                laplacian[idx_i, idx_i] += weight
                laplacian[idx_j, idx_j] += weight
                laplacian[idx_i, idx_j] -= weight
                laplacian[idx_j, idx_i] -= weight

        laplacian = 0.5 * (laplacian + laplacian.T)
        if not np.all(np.isfinite(laplacian)):
            raise TopologicalSurgeryError("Laplaciano de Čech no finito.")

        mismatch = float(math.sqrt(max(0.0, mismatch_sq)))
        if cover_count == 1:
            diagnostics.append(
                "Nervio de Čech trivial (una sola carta); Δ_Čech = [0]."
            )

        return laplacian, mismatch, incident_energy, tuple(diagnostics)

    def compute_cech_laplacian(
        self,
        local_signals: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        r"""
        API pública compatible.

        Calcula el Laplaciano de Čech de las 1-cocadenas sobre las intersecciones:

            Δ_Čech = δ_Čechᵀ δ_Čech

        Retorna la matriz SPSD del Laplaciano de Čech y el mismatch global ‖δφ‖.
        """
        signals, _ = self._phase1_canonical_local_signals(local_signals)
        laplacian, mismatch, _, _ = self._phase2_compute_cech_laplacian_internal(
            signals
        )
        return laplacian, mismatch

    def _phase2_cech_spectral_audit(
        self,
        laplacian: np.ndarray,
    ) -> Tuple[np.ndarray, int, int, Tuple[str, ...]]:
        """Espectro de Δ_Čech, dim ker y β₀ del nervio."""
        diagnostics: List[str] = []

        if laplacian.size == 0:
            trivial = np.array([0.0], dtype=np.float64)
            return trivial, 1, 1, ("Laplaciano de Čech vacío; espectro trivial.",)

        try:
            eigvals = la.eigvalsh(laplacian)
        except la.LinAlgError:
            diagnostics.append(
                "Fallo espectral en Laplaciano de Čech; espectro trivial."
            )
            trivial = np.array([0.0], dtype=np.float64)
            return trivial, 1, 1, tuple(diagnostics)

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            diagnostics.append(
                "Espectro de Čech no finito; se colapsa a espectro trivial."
            )
            trivial = np.array([0.0], dtype=np.float64)
            return trivial, 1, 1, tuple(diagnostics)

        eigvals = np.sort(np.real(eigvals))
        abs_tol = self._spectral_abs_tol(eigvals, _KERNEL_REL_TOL)
        kernel_dim = int(np.sum(np.abs(eigvals) <= abs_tol))
        nerve_betti_0 = max(kernel_dim, 1 if eigvals.size == 1 and abs(eigvals[0]) <= abs_tol else kernel_dim)
        return eigvals, kernel_dim, int(nerve_betti_0), tuple(diagnostics)

    def _phase2_resolve_cover_id(self, noisy_cover_idx: Any) -> int:
        """
        Resuelve un identificador de carta Čech.

        Acepta un cover_id real del cubrimiento o un índice posicional
        0..len(covers)-1, por compatibilidad.
        """
        try:
            value = int(noisy_cover_idx)
        except (TypeError, ValueError) as exc:
            raise TopologicalSurgeryError(
                "noisy_cover_idx debe ser entero."
            ) from exc

        if value in self._covering:
            return value

        if 0 <= value < len(self._cover_ids):
            return self._cover_ids[value]

        raise TopologicalSurgeryError(
            f"noisy_cover_idx={value} no pertenece al cubrimiento de Čech."
        )

    def _regularize_spd_matrix(self, metric: np.ndarray) -> np.ndarray:
        """Regulariza una matriz real simétrica como SPD (proyección espectral)."""
        regularized, _ = self._phase1_project_metric_spd(np.asarray(metric, dtype=np.float64))
        return regularized

    def _phase2_noisy_edge_mask(
        self,
        boundary: np.ndarray,
        cover_id: int,
    ) -> np.ndarray:
        """Máscara booleana de aristas incidentes a nodos de la carta ruidosa."""
        edge_count = int(boundary.shape[0])
        mask = np.zeros(edge_count, dtype=bool)
        node_count = int(boundary.shape[1])

        for node in self._covering[cover_id]:
            if node < 0 or node >= node_count:
                continue
            incident = np.flatnonzero(boundary[:, node] != 0.0)
            if incident.size:
                mask[incident] = True
        return mask

    def _phase2_cut_projector(
        self,
        edge_count: int,
        noisy_edges: np.ndarray,
    ) -> np.ndarray:
        r"""
        Proyector anisotrópico diagonal Π_cut ∈ M_M(ℝ):

            (Π_cut)_{ee} = ε_cut  si e es ruidosa,  1  en caso contrario.
        """
        diagonal = np.ones(edge_count, dtype=np.float64)
        if noisy_edges.size:
            diagonal[noisy_edges] = _SURGERY_CUT
        return diagonal

    def perform_anisotropic_surgery(
        self,
        global_metric_G: np.ndarray,
        boundary_matrix: np.ndarray,
        noisy_cover_idx: int,
    ) -> np.ndarray:
        r"""
        API pública compatible.

        Deforma la métrica Riemanniana de fondo mediante cirugía anisotrópica:

            G_surgical = Π_cut G Π_cut

        con Π_cut diagonal sobre las aristas incidentes a la carta ruidosa,
        seguido de proyección espectral al cono SPD (pasividad / invertibilidad).
        """
        boundary = self._phase1_canonical_boundary_matrix(boundary_matrix)
        metric, _ = self._phase1_canonical_metric(
            global_metric_G,
            expected_dim=int(boundary.shape[0]),
        )
        cover_id = self._phase2_resolve_cover_id(noisy_cover_idx)
        return self._phase2_perform_anisotropic_surgery(metric, boundary, cover_id)

    def _phase2_perform_anisotropic_surgery(
        self,
        metric: np.ndarray,
        boundary: np.ndarray,
        cover_id: int,
    ) -> np.ndarray:
        """Núcleo de la cirugía anisotrópica sobre una métrica ya canonizada."""
        noisy_mask = self._phase2_noisy_edge_mask(boundary, cover_id)
        projector = self._phase2_cut_projector(int(metric.shape[0]), noisy_mask)
        surgical = (projector[:, None] * metric) * projector[None, :]

        noisy_indices = np.flatnonzero(noisy_mask)
        for edge in noisy_indices:
            surgical[edge, edge] = max(float(surgical[edge, edge]), _SURGERY_DIAG_FLOOR)

        surgical = 0.5 * (surgical + surgical.T)
        return self._regularize_spd_matrix(surgical)

    def project_fock_isolated_mode(
        self,
        global_density_rho: np.ndarray,
        noisy_cover_idx: int,
    ) -> Tuple[np.ndarray, float]:
        r"""
        API pública compatible.

        Ejecuta la proyección de Lüders del canal cuántico ruidoso:

            P = I − ∑_{v ∈ U_ruidosa} |v⟩⟨v|
            ρ ↦ P ρ P / tr(P ρ P)

        Retorna (ρ_renormalizada, masa remanente tr(PρP) *antes* de renormalizar).
        Si la masa colapsa, se emite el vacío de Fock |0⟩⟨0| y masa 0.
        """
        density, _, _ = self._phase1_canonical_density_matrix(global_density_rho)
        cover_id = self._phase2_resolve_cover_id(noisy_cover_idx)
        projected, remaining_mass, _ = self._phase2_project_fock_isolated_mode(
            density,
            cover_id,
        )
        return projected, remaining_mass

    def _phase2_project_fock_isolated_mode(
        self,
        density: np.ndarray,
        cover_id: int,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Núcleo de Lüders.

        Retorna (ρ_renorm, masa_remanente, masa_descartada).
        """
        dimension = int(density.shape[0])
        projector = np.eye(dimension, dtype=np.complex128)

        for node in self._covering[cover_id]:
            if 0 <= node < dimension:
                projector[node, node] = 0.0

        isolated = projector @ density @ projector.conj().T
        isolated = 0.5 * (isolated + isolated.conj().T)

        remaining_mass = float(np.real(np.trace(isolated)))
        if not math.isfinite(remaining_mass):
            raise TopologicalSurgeryError("Traza de ρ_surg no finita.")
        remaining_mass = max(0.0, remaining_mass)
        discarded = max(0.0, 1.0 - remaining_mass)

        try:
            eigvals, eigvecs = la.eigh(isolated)
        except la.LinAlgError as exc:
            raise TopologicalSurgeryError(
                "No fue posible diagonalizar rho_surg."
            ) from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise TopologicalSurgeryError("Autovalores de rho_surg inválidos.")

        eigvals_clamped = np.maximum(np.real(eigvals), 0.0)
        mass = float(np.sum(eigvals_clamped))

        if mass <= self._reg:
            vacuum = np.zeros((dimension, dimension), dtype=np.complex128)
            vacuum[0, 0] = 1.0
            return vacuum, remaining_mass, discarded

        eigvals_clamped = eigvals_clamped / mass
        eigvals_clamped = np.maximum(eigvals_clamped, self._reg)
        eigvals_clamped = eigvals_clamped / float(np.sum(eigvals_clamped))

        renormalized = (eigvecs * eigvals_clamped) @ eigvecs.conj().T
        renormalized = 0.5 * (renormalized + renormalized.conj().T)

        if not np.all(np.isfinite(renormalized)):
            raise TopologicalSurgeryError("ρ_surg renormalizada no finita.")

        return renormalized, remaining_mass, discarded

    def _phase2_identify_noisy_cover(
        self,
        local_signals: Dict[int, np.ndarray],
        incident_energy: np.ndarray,
    ) -> int:
        """
        Identifica la carta Čech de mayor obstrucción cohomológica incidente.

        Desempate: máxima varianza de telemetría local. Si todo es nulo,
        se elige la primera carta en orden canónico.
        """
        if incident_energy.size and float(np.max(incident_energy)) > 0.0:
            return self._cover_ids[int(np.argmax(incident_energy))]

        best_id = self._cover_ids[0]
        best_var = -1.0
        for cover_id in self._cover_ids:
            signal = local_signals.get(cover_id, np.array([], dtype=np.float64))
            if signal.size > 0:
                variance = float(np.var(signal))
                if not math.isfinite(variance):
                    variance = 0.0
            else:
                variance = 0.0
            if variance > best_var:
                best_var = variance
                best_id = cover_id
        return best_id

    def _phase2_metric_condition(
        self,
        metric: np.ndarray,
    ) -> Tuple[float, float]:
        """Número de condición y gap espectral relativo λ_min⁺/λ_max⁺."""
        try:
            eigvals = la.eigvalsh(metric)
        except la.LinAlgError as exc:
            raise TopologicalSurgeryError(
                "No fue posible diagonalizar la métrica quirúrgica."
            ) from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise TopologicalSurgeryError(
                "Autovalores de la métrica quirúrgica inválidos."
            )

        eigvals = np.real(eigvals)
        abs_tol = self._spectral_abs_tol(eigvals, 1e-12)
        positive = eigvals[eigvals > abs_tol]
        if positive.size == 0:
            return 1.0, 0.0

        max_pos = float(np.max(positive))
        min_pos = float(np.min(positive))
        if min_pos <= 0.0 or max_pos <= 0.0:
            return _COND_CAP, 0.0

        condition_number = max_pos / min_pos
        if not math.isfinite(condition_number):
            condition_number = _COND_CAP
        condition_number = float(min(max(1.0, condition_number), _COND_CAP))

        spectral_gap = min_pos / max_pos
        if not math.isfinite(spectral_gap):
            spectral_gap = 0.0
        return condition_number, float(max(0.0, spectral_gap))

    def _phase2_metric_sqrt(self, metric: np.ndarray) -> np.ndarray:
        r"""Raíz simétrica G^{1/2} vía cálculo funcional de Borel sobre σ(G) ⊂ [ε, ∞)."""
        try:
            eigvals, eigvecs = la.eigh(metric)
        except la.LinAlgError as exc:
            raise TopologicalSurgeryError(
                "No fue posible diagonalizar G_surgical para G^{1/2}."
            ) from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise TopologicalSurgeryError("Autovalores de G^{1/2} inválidos.")

        floor = max(_SPD_FLOOR, _MACHINE_EPS)
        sqrt_eigs = np.sqrt(np.maximum(np.real(eigvals), floor))
        root = (eigvecs * sqrt_eigs) @ eigvecs.T
        root = 0.5 * (root + root.T)
        if not np.all(np.isfinite(root)):
            raise TopologicalSurgeryError("G^{1/2} no finita.")
        return root

    def _phase2_remnant_laplacian(
        self,
        boundary: np.ndarray,
        metric: np.ndarray,
    ) -> np.ndarray:
        r"""
        Laplaciano de Hodge 0-cochain remanente en forma de Gram:

            A = G^{1/2} B ,   L_rem = Aᵀ A = Bᵀ G B
        """
        root = self._phase2_metric_sqrt(metric)
        factor = root @ boundary
        laplacian = factor.T @ factor
        laplacian = 0.5 * (laplacian + laplacian.T)
        if not np.all(np.isfinite(laplacian)):
            raise TopologicalSurgeryError("Laplaciano remanente no finito.")
        return laplacian

    def _phase2_active_nodes(
        self,
        boundary: np.ndarray,
        metric: np.ndarray,
    ) -> Tuple[int, ...]:
        """Nodos incidentes a aristas cuya rigidez diagonal sobrevive a la cirugía."""
        diag = np.real(np.diag(metric))
        scale = max(1.0, float(np.max(np.abs(diag))) if diag.size else 1.0)
        threshold = max(_ACTIVE_ABS_FLOOR, _ACTIVE_REL_FLOOR * scale)

        node_count = int(boundary.shape[1])
        active: List[int] = []
        for node in range(node_count):
            incident = np.flatnonzero(boundary[:, node] != 0.0)
            if incident.size == 0:
                continue
            if np.any(diag[incident] > threshold):
                active.append(node)
        return tuple(active)

    def _phase2_fiedler_of_compression(
        self,
        remnant: np.ndarray,
        active_nodes: Sequence[int],
    ) -> Tuple[float, Tuple[str, ...]]:
        """
        λ₂ de la compresión de Rayleigh de L_rem al subespacio de nodos activos.

        No es el Laplaciano inducido (se conservan los términos de corte ya
        absorbidos en G_surgical); es la forma cuadrática restringida.
        """
        diagnostics: List[str] = []

        if len(active_nodes) <= 1:
            return 0.0, tuple(diagnostics)

        index = np.asarray(active_nodes, dtype=np.int64)
        sub = remnant[np.ix_(index, index)]
        sub = 0.5 * (sub + sub.T)

        try:
            eigvals = la.eigvalsh(sub)
        except la.LinAlgError:
            diagnostics.append("Fallo espectral en Laplaciano remanente.")
            return -math.inf, tuple(diagnostics)

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            diagnostics.append("Espectro remanente no finito.")
            return -math.inf, tuple(diagnostics)

        sorted_eigs = np.sort(np.real(eigvals))
        if sorted_eigs.size > 1:
            fiedler = float(sorted_eigs[1])
        else:
            fiedler = float(sorted_eigs[0])

        if not math.isfinite(fiedler):
            return -math.inf, tuple(diagnostics)
        return fiedler, tuple(diagnostics)

    def _phase2_evaluate_heyting(
        self,
        *,
        is_globally_coherent: bool,
        surgery_required: bool,
        cohomological_mismatch: float,
        fiedler_residual: float,
        isolated_fock_trace: float,
        surgical_metric_cond: float,
        hermitian_residual: float,
        override_present: bool,
        override_valid: bool,
    ) -> Tuple[str, Tuple[str, ...]]:
        r"""
        Clasificador de Heyting Ω₃ por meet de átomos.

        VETO duro:
          • observables no finitos;
          • pérdida de conectividad global remanente;
          • residuo no simétrico incompatible con autoadjunción.

        DEGRADED:
          • cirugía activa (luz ámbar), con o sin override;
          • Fiedler bajo cota blanda;
          • número de condición métrico elevado.
        """
        diagnostics: List[str] = []
        atoms: List[str] = []

        def veto(message: str) -> None:
            atoms.append(HeytingVerdict.VETOED.value)
            diagnostics.append(message)

        def degrade(message: str) -> None:
            atoms.append(HeytingVerdict.DEGRADED.value)
            diagnostics.append(message)

        if not math.isfinite(cohomological_mismatch):
            veto("Mismatch cohomológico no finito.")
        if not math.isfinite(fiedler_residual):
            veto("Fiedler remanente no finito.")
        if not math.isfinite(isolated_fock_trace):
            veto("Traza de Fock no finita.")
        if not math.isfinite(surgical_metric_cond):
            veto("Número de condición quirúrgico no finito.")
        if not math.isfinite(hermitian_residual):
            veto("Residuo simétrico de L_rem no finito.")

        if not is_globally_coherent:
            veto(
                f"Malla remanente no sismorresistente. Fiedler={fiedler_residual:.6e}."
            )

        if hermitian_residual > 1e-6:
            veto("Residuo simétrico de L_rem incompatible con autoadjunción.")
        elif hermitian_residual > _HERMITIAN_REL_TOL:
            degrade("Residuo simétrico de L_rem elevado.")

        if surgery_required:
            if override_present and override_valid:
                degrade(
                    "¡POSITRÓN DE AUTORIZACIÓN HUMANA [e+] DETECTADO EN CIRUGÍA! "
                    "Aniquilando anomalía analítica local. Cicatriz topológica "
                    "conservada (DEGRADED)."
                )
            elif override_present:
                degrade("Override presente pero criptográficamente inválido.")
            else:
                degrade("Cirugía Čech activa bajo luz ámbar sin override.")

        if math.isfinite(fiedler_residual) and (
            fiedler_residual < self._fiedler_degraded
        ):
            degrade(
                f"Fiedler remanente {fiedler_residual:.6e} bajo cota blanda."
            )

        if math.isfinite(surgical_metric_cond) and (
            surgical_metric_cond > _COND_DEGRADED
        ):
            degrade("Número de condición de G_surgical elevado.")

        if not atoms:
            atoms.append(HeytingVerdict.COHERENT.value)

        return self._heyting_meet(atoms), tuple(diagnostics)

    def _phase2_audit_and_operate(self, dossier: Phase1Dossier) -> SurgeryCertificate:
        r"""
        FASE 2 · ORIENT/OPERATE  (morfismo de cierre).

        Compone nervio de Čech, cirugía anisotrópica, Lüders y Hodge remanente
        en un `Phase2Dossier`. El valor de retorno es el compuesto

            Φ₂₃ ∘ Operate  :  Phase1Dossier  →  SurgeryCertificate

        realizado por `_phase3_open_from_phase2`, primer morfismo de la FASE 3.
        """
        diagnostics: List[str] = list(dossier.diagnostics)

        laplacian, mismatch, incident_energy, cech_diag = (
            self._phase2_compute_cech_laplacian_internal(dossier.local_signals)
        )
        diagnostics.extend(cech_diag)

        eigvals, kernel_dim, nerve_betti_0, spectral_diag = (
            self._phase2_cech_spectral_audit(laplacian)
        )
        diagnostics.extend(spectral_diag)

        surgery_required = bool(
            mismatch > dossier.lipschitz_bound_Lmax * self._safety_margin
        )

        noisy_cover_id = -1
        discarded_mass = 0.0

        if surgery_required:
            noisy_cover_id = self._phase2_identify_noisy_cover(
                dossier.local_signals,
                incident_energy,
            )
            diagnostics.append(
                f"¡ANOMALÍA ELECTROMAGNÉTICA DETECTADA EN CARTA ČECH {noisy_cover_id}! "
                f"Mismatch H¹ ({mismatch:.6f}) excede la cota de Lipschitz "
                f"({dossier.lipschitz_bound_Lmax:.6f}). Procediendo con cirugía."
            )

            metric_surgical = self._phase2_perform_anisotropic_surgery(
                dossier.global_metric_G,
                dossier.boundary_matrix,
                noisy_cover_id,
            )
            density_surgical, remaining_mass, discarded_mass = (
                self._phase2_project_fock_isolated_mode(
                    dossier.global_density_rho,
                    noisy_cover_id,
                )
            )
        else:
            metric_surgical = self._regularize_spd_matrix(dossier.global_metric_G)
            density_surgical = np.array(
                dossier.global_density_rho,
                dtype=np.complex128,
                copy=True,
            )
            remaining_mass = float(np.real(np.trace(density_surgical)))
            discarded_mass = 0.0

        if not math.isfinite(remaining_mass):
            remaining_mass = 0.0
            diagnostics.append("Traza de Fock no finita; se colapsa a cero.")

        condition_number, spectral_gap = self._phase2_metric_condition(metric_surgical)

        remnant = self._phase2_remnant_laplacian(
            dossier.boundary_matrix,
            metric_surgical,
        )
        residual = self._finite_norm(remnant - remnant.T)
        denom = max(1.0, self._finite_norm(remnant))
        hermitian_residual = residual / denom
        if hermitian_residual > _HERMITIAN_REL_TOL:
            diagnostics.append(
                f"Residuo simétrico relativo de L_rem = {hermitian_residual:.3e}."
            )

        active_nodes = self._phase2_active_nodes(
            dossier.boundary_matrix,
            metric_surgical,
        )
        fiedler, fiedler_diag = self._phase2_fiedler_of_compression(
            remnant,
            active_nodes,
        )
        diagnostics.extend(fiedler_diag)

        is_globally_coherent = bool(
            len(active_nodes) > 1 and fiedler >= self._fiedler_veto
        )

        verdict, heyting_diag = self._phase2_evaluate_heyting(
            is_globally_coherent=is_globally_coherent,
            surgery_required=surgery_required,
            cohomological_mismatch=mismatch,
            fiedler_residual=fiedler,
            isolated_fock_trace=remaining_mass,
            surgical_metric_cond=condition_number,
            hermitian_residual=hermitian_residual,
            override_present=dossier.override_present,
            override_valid=dossier.override_is_valid,
        )
        diagnostics.extend(heyting_diag)

        phase2 = Phase2Dossier(
            phase1=dossier,
            cech_laplacian=self._freeze_array(laplacian, np.float64),
            cech_laplacian_eigenvalues=self._freeze_array(eigvals, np.float64),
            cohomological_mismatch=mismatch,
            noisy_cover_id=int(noisy_cover_id),
            surgery_required=surgery_required,
            G_surgical=self._freeze_array(metric_surgical, np.float64),
            rho_surgical=self._freeze_array(density_surgical, np.complex128),
            isolated_fock_trace=remaining_mass,
            discarded_fock_mass=discarded_mass,
            surgical_metric_cond=condition_number,
            metric_spectral_gap=spectral_gap,
            remnant_laplacian=self._freeze_array(remnant, np.float64),
            active_nodes=tuple(int(n) for n in active_nodes),
            fiedler_residual=fiedler,
            is_globally_coherent=is_globally_coherent,
            nerve_betti_0=int(nerve_betti_0),
            cech_kernel_dimension=int(kernel_dim),
            hermitian_residual=hermitian_residual,
            heyting_verdict=verdict,
            diagnostics=tuple(diagnostics),
        )

        logger.info(
            "FASE 2 ORIENT: veredicto=%s, mismatch=%.6f, Fiedler=%.6e, "
            "cirugía=%s, β₀(nervio)=%d",
            verdict,
            mismatch,
            fiedler,
            surgery_required,
            nerve_betti_0,
        )

        # ── continuación anidada: el terminal de FASE 2 es el inicial de FASE 3
        return self._phase3_open_from_phase2(phase2)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3 · ACT / CERTIFY
    #
    # Objeto inicial  : Phase2Dossier  (entregado por _phase2_audit_and_operate)
    # Objeto terminal : SurgeryCertificate (inmutable, sellado, fail-closed)
    # Morfismo de apertura: _phase3_open_from_phase2
    #
    # Invariantes:
    #   (K1) VETOED ⇒ interlock crowbar simulado y latencia ∈ [380, 420] ns.
    #   (K2) sello = SHA-256(dominio ‖ veredicto ‖ sesión ‖ observables).
    #   (K3) el certificado no muta tensores ni revela el token de override.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase3_open_from_phase2(self, dossier: Phase2Dossier) -> SurgeryCertificate:
        r"""
        Inicio formal de la FASE 3 · ACT/CERTIFY.

        Continuación estricta del terminal de FASE 2: recibe el expediente
        operado y abre actuación de interlock + certificación inmutable.
        """
        if not isinstance(dossier, Phase2Dossier):
            raise TopologicalSurgeryError(
                "FASE 3 exige un Phase2Dossier emitido por FASE 2."
            )
        return self._phase3_certify(dossier)

    def _phase3_parse_verdict(self, verdict: Any) -> HeytingVerdict:
        """Parseo total: cualquier etiqueta no reconocida colapsa a VETOED."""
        if isinstance(verdict, HeytingVerdict):
            return verdict
        try:
            return HeytingVerdict(str(verdict).strip().upper())
        except ValueError:
            return HeytingVerdict.VETOED

    def _phase3_actuate_interlock(
        self,
        verdict: Any,
        session_digest: str,
    ) -> Tuple[bool, float]:
        r"""
        FASE 3 · ACT.

        Si Heyting colapsa a VETOED, se simula la ISR en IRAM del ESP32:
        conmutación de GPIO14 y cebado del tiristor crowbar BT151 en
        400 ns ± jitter gaussiano recortado al intervalo de calibración.
        """
        verdict_enum = self._phase3_parse_verdict(verdict)
        session_ref = str(session_digest)[:16]

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
                "¡COLAPSO DE LA ENVOLVENTE EN CIRUGÍA TOPOLÓGICA! "
                "La malla remanente perdió consistencia global. "
                "Conmutando GPIO%d en IRAM en %.2f ns. Crowbar BT151 gatillado. "
                "Sello: %s",
                _GPIO_CROWBAR_PIN,
                latency,
                session_ref,
            )
            return True, latency

        logger.info(
            "Cirugía Čech regulada sin interlock. Estado=%s. Sello=%s",
            verdict_enum.value,
            session_ref,
        )
        return False, 0.0

    @staticmethod
    def _phase3_format_scalar(value: float) -> str:
        """Canoniza un escalar a 17 decimales; no-finitos → nan."""
        number = float(value)
        if not math.isfinite(number):
            number = float("nan")
        return f"{number:.17e}"

    def _phase3_compose_signature(
        self,
        *,
        verdict: str,
        session_digest: str,
        mismatch: float,
        condition_number: float,
        spectral_gap: float,
        fiedler: float,
        trace_fock: float,
        latency_ns: float,
        surgery_active: bool,
        noisy_cover_id: int,
        lipschitz_bound: float,
        engine_digest: str,
        nerve_betti_0: int,
        discarded_fock_mass: float,
        override_present: bool,
        override_valid: bool,
    ) -> str:
        """Firma SHA-256 final de telemetría con separación de dominio."""
        sha = hashlib.sha256()

        self._hash_update_domain(sha, _DOMAIN_TELEMETRY)
        self._hash_update_bytes(sha, verdict.encode("utf-8"))
        self._hash_update_bytes(sha, session_digest.encode("utf-8"))
        self._hash_update_bytes(sha, engine_digest.encode("utf-8"))

        for value in (
            mismatch,
            condition_number,
            spectral_gap,
            fiedler,
            trace_fock,
            latency_ns,
            float(noisy_cover_id),
            lipschitz_bound,
            float(nerve_betti_0),
            discarded_fock_mass,
        ):
            self._hash_update_bytes(
                sha,
                self._phase3_format_scalar(float(value)).encode("utf-8"),
            )

        for flag in (surgery_active, override_present, override_valid):
            self._hash_update_bytes(sha, b"1" if bool(flag) else b"0")

        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        return sha.hexdigest()

    def _phase3_certify(self, dossier: Phase2Dossier) -> SurgeryCertificate:
        """
        FASE 3 · CERTIFY.

        Emite el certificado inmutable y acciona el interlock si el veredicto
        de Heyting es VETOED. Terminal del morfismo anidado Φ₂₃ ∘ Φ₁₂.
        """
        interlock_fired, latency = self._phase3_actuate_interlock(
            dossier.heyting_verdict,
            dossier.phase1.session_digest,
        )

        seal = self._phase3_compose_signature(
            verdict=dossier.heyting_verdict,
            session_digest=dossier.phase1.session_digest,
            mismatch=dossier.cohomological_mismatch,
            condition_number=dossier.surgical_metric_cond,
            spectral_gap=dossier.metric_spectral_gap,
            fiedler=dossier.fiedler_residual,
            trace_fock=dossier.isolated_fock_trace,
            latency_ns=latency,
            surgery_active=dossier.surgery_required,
            noisy_cover_id=dossier.noisy_cover_id,
            lipschitz_bound=dossier.phase1.lipschitz_bound_Lmax,
            engine_digest=dossier.phase1.engine_digest,
            nerve_betti_0=dossier.nerve_betti_0,
            discarded_fock_mass=dossier.discarded_fock_mass,
            override_present=dossier.phase1.override_present,
            override_valid=dossier.phase1.override_is_valid,
        )

        state = SurgeryState(
            cech_laplacian_eigenvalues=dossier.cech_laplacian_eigenvalues,
            cohomological_mismatch=dossier.cohomological_mismatch,
            surgical_metric_cond=dossier.surgical_metric_cond,
            isolated_fock_trace=dossier.isolated_fock_trace,
            is_globally_coherent=dossier.is_globally_coherent,
            fiedler_residual=dossier.fiedler_residual,
            schema_version=_SCHEMA_VERSION,
            metric_spectral_gap=dossier.metric_spectral_gap,
            active_node_count=len(dossier.active_nodes),
            nerve_betti_0=dossier.nerve_betti_0,
            cech_kernel_dimension=dossier.cech_kernel_dimension,
            discarded_fock_mass=dossier.discarded_fock_mass,
            density_purity=dossier.phase1.density_purity,
            hermitian_residual=dossier.hermitian_residual,
        )

        certificate = SurgeryCertificate(
            heyting_verdict=dossier.heyting_verdict,
            state=state,
            surgery_active=dossier.surgery_required,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            cryptographic_seal=seal,
            schema_version=_SCHEMA_VERSION,
            session_digest=dossier.phase1.session_digest,
            noisy_cover_id=dossier.noisy_cover_id,
            lipschitz_bound=dossier.phase1.lipschitz_bound_Lmax,
            diagnostics=dossier.diagnostics,
            engine_digest=dossier.phase1.engine_digest,
            override_present=dossier.phase1.override_present,
            override_valid=dossier.phase1.override_is_valid,
            nerve_betti_0=dossier.nerve_betti_0,
            active_nodes=dossier.active_nodes,
        )

        if dossier.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "FASE 3 CERTIFY: VETO Čech. mismatch=%.6f, Fiedler=%.6e",
                dossier.cohomological_mismatch,
                dossier.fiedler_residual,
            )
        elif dossier.heyting_verdict == HeytingVerdict.DEGRADED.value:
            logger.warning(
                "FASE 3 CERTIFY: cirugía Čech degradada. diagnostics=%s",
                dossier.diagnostics,
            )
        else:
            logger.info(
                "FASE 3 CERTIFY: cirugía Čech coherente. digest=%s",
                dossier.phase1.session_digest[:16],
            )

        return certificate

    # ═══════════════════════════════════════════════════════════════════════
    # FAIL-CLOSED GLOBAL  (terminal de emergencia, isomorfo a FASE 3 VETOED)
    # ═══════════════════════════════════════════════════════════════════════

    def _fail_closed_certificate(self, reason: str) -> SurgeryCertificate:
        """
        Certificado fail-closed ante excepción no recuperable.

        Garantiza VETO, interlock y firma inmutable incluso cuando el expediente
        de entrada no pudo ser validado. Los observables imposibles se reportan
        como ∞ (no como 0, que simularía coherencia).
        """
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_SESSION)
        self._hash_update_bytes(sha, self._engine_digest.encode("utf-8"))
        self._hash_update_bytes(sha, reason.encode("utf-8"))
        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        session_digest = sha.hexdigest()

        interlock_fired, latency = self._phase3_actuate_interlock(
            HeytingVerdict.VETOED.value,
            session_digest,
        )

        seal = self._phase3_compose_signature(
            verdict=HeytingVerdict.VETOED.value,
            session_digest=session_digest,
            mismatch=math.inf,
            condition_number=math.inf,
            spectral_gap=0.0,
            fiedler=0.0,
            trace_fock=0.0,
            latency_ns=latency,
            surgery_active=False,
            noisy_cover_id=-1,
            lipschitz_bound=math.nan,
            engine_digest=self._engine_digest,
            nerve_betti_0=0,
            discarded_fock_mass=1.0,
            override_present=False,
            override_valid=False,
        )

        empty_real = self._freeze_array(np.array([], dtype=np.float64), np.float64)

        state = SurgeryState(
            cech_laplacian_eigenvalues=empty_real,
            cohomological_mismatch=math.inf,
            surgical_metric_cond=math.inf,
            isolated_fock_trace=0.0,
            is_globally_coherent=False,
            fiedler_residual=0.0,
            schema_version=_SCHEMA_VERSION,
            metric_spectral_gap=0.0,
            active_node_count=0,
            nerve_betti_0=0,
            cech_kernel_dimension=0,
            discarded_fock_mass=1.0,
            density_purity=math.nan,
            hermitian_residual=math.inf,
        )

        return SurgeryCertificate(
            heyting_verdict=HeytingVerdict.VETOED.value,
            state=state,
            surgery_active=False,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            cryptographic_seal=seal,
            schema_version=_SCHEMA_VERSION,
            session_digest=session_digest,
            noisy_cover_id=-1,
            lipschitz_bound=math.nan,
            diagnostics=(f"FAIL-CLOSED: {reason}",),
            engine_digest=self._engine_digest,
            override_present=False,
            override_valid=False,
            nerve_betti_0=0,
            active_nodes=(),
        )

    # ═══════════════════════════════════════════════════════════════════════
    # API PÚBLICA COMPATIBLE
    # ═══════════════════════════════════════════════════════════════════════

    def execute_topological_surgery_cycle(
        self,
        boundary_matrix: np.ndarray,
        global_metric_G: np.ndarray,
        global_density_rho: np.ndarray,
        local_signals: Dict[int, np.ndarray],
        lipschitz_bound_Lmax: float,
        override_token: Optional[str] = None,
    ) -> SurgeryCertificate:
        r"""
        Orquesta el lazo de cirugía de Čech para suprimir falsos positivos
        analógicos.

        Compone las tres fases anidadas

            Φ₂₃ ∘ Φ₁₂ ∘ Observe  :  datos  →  SurgeryCertificate

        Si ocurre cualquier excepción, devuelve certificado fail-closed VETOED
        (el tipo de retorno es total: nunca se propaga el fallo al llamador).
        """
        try:
            return self._phase1_observe_and_freeze(
                boundary_matrix=boundary_matrix,
                global_metric_G=global_metric_G,
                global_density_rho=global_density_rho,
                local_signals=local_signals,
                lipschitz_bound_Lmax=lipschitz_bound_Lmax,
                override_token=override_token,
            )
        except Exception as exc:  # noqa: BLE001 — contrato fail-closed
            logger.exception(
                "Fallo fail-closed en cirugía Čech; emitiendo VETO de frontera."
            )
            return self._fail_closed_certificate(reason=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN DE FIRMAS DE CALIBRE
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "TopologicalSurgeryCech",
    "SurgeryState",
    "SurgeryCertificate",
    "HeytingVerdict",
    "TopologicalSurgeryError",
    "Phase1Dossier",
    "Phase2Dossier",
]

# Compatibilidad con la exportación histórica del módulo original.
all = __all__
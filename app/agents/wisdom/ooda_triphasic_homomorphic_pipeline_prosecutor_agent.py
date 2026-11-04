# -*- coding: utf-8 -*-
r"""
Módulo: OODA Triphasic Homomorphic Pipeline Prosecutor Agent
ubicación: app/agents/wisdom/ooda_triphasic_homomorphic_pipeline_prosecutor_agent.py
Versión de referencia: 15.0.0-POINCARE-RSI-L3

Flujo:
    Fase I   ──► bridge_phase1_to_phase2_campaign_transfer
    Fase II  ──► bridge_phase2_to_phase3_adjudication_transfer
    Fase III ──► execute_phase3_sovereign_decision

Alcance y límites:
* La conexión, curvatura y cantidad de Chern–Simons son proxies matriciales.
  No sustituyen formas diferenciales ni una conexión definida sobre una variedad.
* Yoshida integra el Hamiltoniano discreto especificado. La auditoría KAM examina
  una truncación finita de divisores pequeños; no prueba un teorema KAM.
* El funcional denominado producción de Spohn es un proxy no negativo, no una
  demostración de la segunda ley para un semigrupo GKSL.
* RSI-L3 itera un operador escalar contractivo. No modifica el código fuente ni
  se autoentrena de forma autónoma.
* La aniquilación de Dirac y el interlock ESP32 son simulaciones. Nunca se escribe
  GPIO, MMIO ni el registro documental GPIO_OUT_W1TS_REG.
* La proyección financiera devuelve datos no calculados, no predicciones.
* La clave predeterminada sirve únicamente para demostraciones. En producción
  debe inyectarse una clave aleatoria y protegida externamente.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import itertools
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES Y DOMINIOS CRIPTOGRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-13
_SYMPLECTIC_DRIFT_TOL: Final[float] = 1.0e-7
_CLAIM_TAMPER_TOL: Final[float] = 0.35
_GATE_TEAR_INTEGRITY_FLOOR: Final[float] = 0.50
_FUNCTOR_DEFECT_TOL: Final[float] = 0.15
_DEFAULT_BANACH_CONTRACTION_LIP: Final[float] = 0.885
_MAX_KAM_FOURIER_ORDER: Final[int] = 5

_TWO_PI: Final[float] = 2.0 * math.pi
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0
_DIRAC_AUTH_FAILURE_BACK_ACTION_DB: Final[float] = 12.5
_DIRAC_PHASE_KICK: Final[float] = 0.5
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_ESP32_CROWBAR_MIN_LATENCY_NS: Final[float] = 42.0
_ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008  # Documental; nunca se escribe.

_CHAIN_HEX_LEN: Final[int] = 64
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0: Final[float] = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))
_PROSECUTOR_PERTURBATION_EPS: Final[float] = 0.025

# Solo para desarrollo. No usar en un despliegue real.
_DEFAULT_SECRET_KEY: Final[bytes] = (
    b"DEVELOPMENT_ONLY_OODA_POINCARE_RSI_L3_KEY_v15"
)

_VERSION_STRING: Final[str] = (
    "15.0.0-SOVEREIGN-POINCARE-KAM-DIRAC-ESP32-RSI-L3"
)

_HMAC_DOMAIN_P1: Final[bytes] = b"OODA-PROSECUTOR-P1-GAUGE-v15|"
_HMAC_DOMAIN_P1_TRANSFER: Final[bytes] = b"OODA-PROSECUTOR-P1-TRANSFER-v15|"
_HMAC_DOMAIN_P2: Final[bytes] = b"OODA-PROSECUTOR-P2-POINCARE-v15|"
_HMAC_DOMAIN_P2_TRANSFER: Final[bytes] = b"OODA-PROSECUTOR-P2-TRANSFER-v15|"
_HMAC_DOMAIN_P3: Final[bytes] = b"OODA-PROSECUTOR-P3-PASSPORT-v15|"

# El conteo de anomalías forma parte del mensaje autenticado: evita reutilizar
# una autorización para una cantidad distinta de electrones.
_HMAC_DOMAIN_DIRAC: Final[bytes] = (
    b"OODA-PROSECUTOR-DIRAC-COUNT-BOUND-v15|"
)


logger = logging.getLogger(
    "APU.Wisdom.OODATriphasicHomomorphicPipelineProsecutorAgent"
)


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES CANÓNICAS Y VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(value: float, ndigits: int = 8) -> str:
    return f"{float(value):.{ndigits}f}"


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Convierte a float finito; devuelve default ante datos inválidos."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _float_token(value: Any) -> str:
    """Representación estable y exacta de un float finito para firmas."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("No se puede firmar un valor flotante no finito.")
    return result.hex()


def _stable_jsonable(value: Any) -> Any:
    """Normaliza estructuras permitidas para una serialización determinista."""
    if isinstance(value, IntEnum):
        return int(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _float_token(value)
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError("No se puede serializar un complejo no finito.")
        return {"real": value.real.hex(), "imag": value.imag.hex()}
    if isinstance(value, np.ndarray):
        return _stable_jsonable(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _stable_jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_jsonable(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Tipo no admitido en serialización canónica: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _stable_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _readonly_array(
    value: Any,
    dtype: Any,
) -> NDArray[Any]:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _array_sha384(value: Any, complex_array: bool = False) -> str:
    """Hash de un array con forma y representación little-endian explícitas."""
    if complex_array:
        array = np.asarray(value, dtype=np.dtype("<c16"))
        dtype_name = "<c16"
    else:
        array = np.asarray(value, dtype=np.dtype("<f8"))
        dtype_name = "<f8"

    if not np.all(np.isfinite(array)):
        raise ValueError("No se puede firmar un array con NaN o infinito.")

    array = np.ascontiguousarray(array)
    header = _canonical_json(
        {"dtype": dtype_name, "shape": list(array.shape)}
    ).encode("utf-8")
    return hashlib.sha384(header + b"\x00" + array.tobytes(order="C")).hexdigest()


def _validated_anomaly_count(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("electron_anomaly_count debe ser un entero.")
    count = int(value)
    if count < 0 or count >= 2**63:
        raise ValueError("electron_anomaly_count fuera del dominio permitido.")
    return count


def compute_dirac_positron_hmac(
    secret_key: bytes,
    payload: str,
    electron_anomaly_count: int = 0,
) -> str:
    """
    HMAC-SHA384 de una autorización de simulación.

    La longitud y el conteo se codifican explícitamente para que el mensaje
    autenticado no tenga ambigüedades de concatenación. Para count > 0, el
    emisor debe pasar ese mismo conteo al calcular la firma.
    """
    if not isinstance(secret_key, (bytes, bytearray)) or not secret_key:
        raise ValueError("La clave HMAC debe ser bytes no vacíos.")
    if not isinstance(payload, str):
        raise TypeError("payload debe ser str.")

    count = _validated_anomaly_count(electron_anomaly_count)
    payload_bytes = payload.encode("utf-8")
    message = (
        _HMAC_DOMAIN_DIRAC
        + count.to_bytes(8, byteorder="big", signed=False)
        + len(payload_bytes).to_bytes(8, byteorder="big", signed=False)
        + payload_bytes
    )
    return hmac.new(bytes(secret_key), message, hashlib.sha384).hexdigest()


def _as_heyting(value: Any) -> "HeytingOmega4":
    if isinstance(value, HeytingOmega4):
        return value
    try:
        return HeytingOmega4(int(value))
    except (TypeError, ValueError, OverflowError):
        return HeytingOmega4.CRITICAL


def _as_status(value: Any) -> "OODAProsecutionStatus":
    if isinstance(value, OODAProsecutionStatus):
        return value
    try:
        return OODAProsecutionStatus(int(value))
    except (TypeError, ValueError, OverflowError):
        return OODAProsecutionStatus.PENDING


def _read_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _dirac_auth_flag(dirac: Any) -> bool:
    """Falla cerrada: la ausencia de evidencia no se interpreta como autorización."""
    if dirac is None:
        return False
    return bool(_read_field(dirac, "auth_ok", False))


# ══════════════════════════════════════════════════════════════════════════════
# RETÍCULO Y TIPOS DE TRANSFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega4(IntEnum):
    """
    Cadena finita de verdad:
        VETOED < CRITICAL < DEGRADED < COHERENT.

    Los valores enteros conservan la codificación histórica inversa.
    """

    COHERENT = 0
    DEGRADED = 1
    CRITICAL = 2
    VETOED = 3

    def truth_rank(self) -> int:
        return 3 - int(self)

    def meet(self, other: "HeytingOmega4") -> "HeytingOmega4":
        return HeytingOmega4(max(int(self), int(other)))

    def join(self, other: "HeytingOmega4") -> "HeytingOmega4":
        return HeytingOmega4(min(int(self), int(other)))

    def implies(self, other: "HeytingOmega4") -> "HeytingOmega4":
        if self.truth_rank() <= other.truth_rank():
            return HeytingOmega4.COHERENT
        return other


class OODAProsecutionStatus(IntEnum):
    PENDING = 0
    PURGED_DIRAC_VACUUM = 1
    VETOED_HOLONOMY_BREAKAGE = 2
    VETOED_DIRAC_AUTH_FAILURE = 3
    VETOED_KAM_CHAOTIC_RESONANCE = 4
    VETOED_SYMPLECTIC_DRIFT = 5
    ABSOLVED = 6
    VETOED_HOMOMORPHIC_GATE_TEAR = 7
    VETOED_INDICTMENT_TAMPER = 8


@dataclass(frozen=True, slots=True)
class OODAProsecutionIndictmentGermPhD:
    indictment_id: str
    trace_preservation_ok: bool
    connection_norm: float
    yang_mills_energy: float
    chern_simons_3form: float
    wilson_angle_rad: float
    instanton_number_k: int
    electron_anomaly_count: int
    positron_auth_signature: str
    transaction_payload: str
    density_matrix: NDArray[np.complex128]
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class PoincareProsecutionOrbitalCarrier:
    indictment_ref: Any
    sanitized_density_matrix: NDArray[np.complex128]
    coordinates_q: NDArray[np.float64]
    momenta_p: NDArray[np.float64]
    action_variables_I: NDArray[np.float64]
    angle_variables_theta: NDArray[np.float64]
    hamiltonian_energy: float
    dirac_result: Any
    spohn_entropy_production: float
    gauge_connection_A: NDArray[np.complex128]
    dimension_dof: int
    williamson_invariants: Tuple[float, ...] = ()
    su2_holonomy_trace: float = 0.0
    frequencies_omega: Tuple[float, ...] = ()
    claim_tamper_detected: bool = False
    gate_tear_detected: bool = False


@dataclass(frozen=True, slots=True)
class PoincareSectionResult:
    crossing_count: int
    crossing_points_qp: Tuple[Tuple[float, float], ...]
    section_name: str = "q[0]=0, p[0]>0"


@dataclass(frozen=True, slots=True)
class ProsecutionMonodromyResult:
    monodromy_matrix: NDArray[np.float64]
    floquet_multipliers: Tuple[complex, ...]
    max_lyapunov_exponent: float
    symplectic_defect: float
    period: float
    integration_steps: int = 0
    finite_time: bool = True
    largest_singular_value: float = 1.0


@dataclass(frozen=True, slots=True)
class ProsecutionKAMAuditResult:
    is_kam_torus_stable: bool
    diophantine_gamma: float
    diophantine_tau: float
    min_small_divisor: float
    nekhoroshev_time_bound: float
    poincare_recurrence_time_estimate: float
    max_lyapunov_exponent: float
    rotation_number: float
    fourier_order: int
    finite_order_certificate: bool = False
    resonance_detected: bool = False


@dataclass(frozen=True, slots=True)
class ProsecutionSymplecticFeedbackCarrier:
    orbital_carrier: PoincareProsecutionOrbitalCarrier
    monodromy_result: Any
    kam_audit: Any
    symplectic_drift_penalty: float
    holonomy_breakage_detected: bool
    intermediate_verdict: HeytingOmega4
    prosecution_status_tentative: OODAProsecutionStatus
    energy_drift: float = 0.0
    section_count: int = 0
    pairing_defect: float = 0.0
    symplectic_conserved: bool = True
    claim_tamper_detected: bool = False
    gate_tear_detected: bool = False


@dataclass(frozen=True, slots=True)
class OODAProsecutionExecutionCertificatePhD:
    certificate_id: str
    status: OODAProsecutionStatus
    global_heyting_verdict: HeytingOmega4
    crowbar_triggered: bool
    crowbar_triggered_simulated: bool
    crowbar_latency_ns: float
    electron_anomalies_purged: int
    released_energy_kev: float
    back_action_db: float
    rsi_level3_generation: int
    rsi_banach_contraction_lip: float
    rsi_meta_operator_lip: float
    rsi_cauchy_residual: float
    spohn_entropy_production: float
    is_kam_torus_stable: bool
    max_lyapunov_exponent: float
    symplectic_drift: float
    energy_drift: float
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class ProsecutorRSIConfiguration:
    generation: int
    banach_lipschitz: float
    meta_operator_lip: float
    cauchy_residual: float
    psd_floor: float
    symplectic_tolerance: float
    fixed_point_value: float = 0.0
    fixed_point_iterations: int = 0
    fixed_point_converged: bool = True
    fixed_point_error_bound: float = 0.0


@dataclass(frozen=True, slots=True)
class OODAProsecutorGaugeContractPhD:
    """Contrato P1. Los invariantes geométricos son proxies matriciales."""

    contract_id: str
    sovereign_id: str
    chern_simons_3form: float
    hopf_c1_invariant: float
    wilson_angle_rad: float
    connection_norm: float
    yang_mills_energy: float
    instanton_number_k: int
    su2_holonomy_trace: float
    purity: float
    dof_cotangent: int
    trace_preservation_ok: bool
    gate_tear_detected: bool
    isometry_verdict: HeytingOmega4
    chain_hash: str
    hmac_p1_signature: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class ProsecutorPhase1TransferCarrier:
    """Transferencia P1→P2; su firma cubre los datos numéricos consumidos por P2."""

    gauge_contract: OODAProsecutorGaugeContractPhD
    poincare_orbital_carrier: PoincareProsecutionOrbitalCarrier
    sanitized_density_matrix: NDArray[np.complex128]
    gauge_connection: NDArray[np.complex128]
    indictment_germ: Any
    dirac_auth_ok: bool
    claim_tamper_detected: bool
    williamson_invariants: Tuple[float, ...]
    energy_hamiltonian: float
    hmac_transfer_signature: str = ""


@dataclass(frozen=True, slots=True)
class OODAProsecutionCampaignResultPhD:
    """Auditoría P2, con resultados KAM explícitamente de orden finito."""

    campaign_id: str
    bound_contract_id: str
    lipschitz_contraction: float
    spohn_entropy_production: float
    is_kam_torus_stable: bool
    max_lyapunov_exponent: float
    poincare_recurrence_time: float
    symplectic_drift_penalty: float
    holonomy_breakage_detected: bool
    nekhoroshev_bound: float
    rotation_number: float
    section_count: int
    energy_drift: float
    pairing_defect: float
    symplectic_conserved: bool
    qnd_orient_verdict: HeytingOmega4
    prosecution_status_tentative: OODAProsecutionStatus
    chain_hash: str
    hmac_p2_signature: str
    timestamp_utc: float
    phase1_transfer_signature: str = ""
    dirac_auth_ok: bool = False
    claim_tamper_detected: bool = False
    gate_tear_detected: bool = False


@dataclass(frozen=True, slots=True)
class ProsecutorPhase2TransferCarrier:
    """Transferencia P2→P3; firma el feedback y los datos inspeccionados por P3."""

    gauge_contract: OODAProsecutorGaugeContractPhD
    campaign_result: OODAProsecutionCampaignResultPhD
    symplectic_feedback: ProsecutionSymplecticFeedbackCarrier
    sanitized_density_matrix: NDArray[np.complex128]
    gauge_connection: NDArray[np.complex128]
    indictment_germ: Any
    dirac_auth_ok: bool
    claim_tamper_detected: bool
    hmac_transfer_signature: str = ""


@dataclass(frozen=True, slots=True)
class OODAProsecutorSovereignPassportPhD:
    """Pasaporte con compromiso SHA-384 de todos los campos sustantivos."""

    passport_id: str
    sovereign_id: str
    campaign_id: str
    bound_contract_id: str
    global_verdict: HeytingOmega4
    prosecution_status: OODAProsecutionStatus
    crowbar_triggered: bool
    crowbar_triggered_simulated: bool
    crowbar_latency_ns: float
    electron_anomalies_purged: int
    released_energy_kev: float
    back_action_db: float
    rsi_level3_generation: int
    rsi_banach_contraction_lip: float
    rsi_meta_operator_lip: float
    rsi_cauchy_residual: float
    homomorphic_gate_phi12_integrity: float
    homomorphic_gate_phi23_integrity: float
    homomorphic_gate_phi13_integrity: float
    functoriality_defect: float
    energy_hamiltonian: float
    energy_drift: float
    nekhoroshev_bound: float
    williamson_min: float
    claim_tamper_detected: bool
    gate_tear_detected: bool
    esp32_gpio_w1ts_reg_documentary: int
    merkle_dag_sha384_root: str
    hmac_provenance_signature: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float
    phase1_transfer_signature: str = ""
    phase2_campaign_signature: str = ""
    phase2_transfer_signature: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# NÚCLEO NUMÉRICO DE REFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

class BanachGaugeDiracProsecutionCore:
    """
    Núcleo P1 de referencia.

    La proyección espectral es la proyección euclídea de los autovalores sobre
    el simplex probabilístico, después de hermitizar la matriz.
    """

    def __init__(self, dimension: int, secret_key: bytes) -> None:
        dim = max(4, int(dimension))
        if dim % 2:
            dim += 1
        self.dimension = dim
        self.dof = dim // 2
        self.secret_key = bytes(secret_key)

    @staticmethod
    def _project_eigenvalues_to_simplex(values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Proyección L2 de un vector real sobre {x≥0, sum(x)=1}."""
        y = np.asarray(values, dtype=np.float64)
        if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
            raise ValueError("Espectro inválido para proyección PSD.")

        sorted_y = np.sort(y)[::-1]
        cumulative = np.cumsum(sorted_y) - 1.0
        indices = np.arange(1, y.size + 1, dtype=np.float64)
        admissible = sorted_y - cumulative / indices > 0.0
        if not np.any(admissible):
            return np.full(y.size, 1.0 / y.size, dtype=np.float64)

        rho = int(np.nonzero(admissible)[0][-1])
        theta = float(cumulative[rho] / float(rho + 1))
        projected = np.maximum(y - theta, 0.0)
        total = float(np.sum(projected))
        if not math.isfinite(total) or total <= _WILKINSON_EPSILON:
            return np.full(y.size, 1.0 / y.size, dtype=np.float64)
        return projected / total

    def sanitize_density_matrix_higham(self, matrix: NDArray[Any]) -> NDArray[np.complex128]:
        """
        Hermitiza y proyecta al conjunto de matrices densidad.

        Si la entrada no es una matriz cuadrada finita o la descomposición
        espectral falla, retorna el estado maximálmente mezclado configurado.
        """
        try:
            array = np.asarray(matrix, dtype=np.complex128)
        except (TypeError, ValueError, OverflowError):
            array = np.empty((0, 0), dtype=np.complex128)

        if (
            array.ndim != 2
            or array.shape[0] != array.shape[1]
            or array.shape[0] == 0
            or not np.all(np.isfinite(array))
        ):
            n = self.dimension
            return np.eye(n, dtype=np.complex128) / float(n)

        array = 0.5 * (array + array.conj().T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(array)
            projected = self._project_eigenvalues_to_simplex(
                np.asarray(np.real(eigenvalues), dtype=np.float64)
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            n = self.dimension
            return np.eye(n, dtype=np.complex128) / float(n)

        rho = (eigenvectors * projected) @ eigenvectors.conj().T
        rho = 0.5 * (rho + rho.conj().T)
        trace = float(np.real(np.trace(rho)))
        if not math.isfinite(trace) or trace <= _WILKINSON_EPSILON:
            n = self.dimension
            return np.eye(n, dtype=np.complex128) / float(n)
        rho /= trace
        return np.asarray(rho, dtype=np.complex128)

    def extract_gauge_invariants(
        self,
        rho: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], float, float]:
        """
        Construye proxies matriciales.

        A = i(ρ − Tr(ρ)I/N) es antihermitiana y sin traza. Se toma dA=0 en este
        modelo finito, por lo que F=A². Esto no es una discretización completa
        de una conexión sobre una variedad.
        """
        matrix = np.asarray(rho, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
            raise ValueError("rho debe ser una matriz cuadrada no vacía.")

        n = int(matrix.shape[0])
        hermitian = 0.5 * (matrix + matrix.conj().T)
        traceless = hermitian - (
            np.trace(hermitian) / float(n)
        ) * np.eye(n, dtype=np.complex128)
        connection = 1.0j * traceless
        curvature_proxy = connection @ connection
        yang_mills_proxy = 0.5 * float(la.norm(curvature_proxy, ord="fro") ** 2)
        cubic = connection @ connection @ connection
        chern_simons_proxy = float(
            np.real(np.trace(connection @ curvature_proxy - (2.0 / 3.0) * cubic))
        )
        return connection, curvature_proxy, yang_mills_proxy, chern_simons_proxy

    def _dirac_auth_ok(
        self,
        payload: str,
        signature: str,
        electron_count: int,
    ) -> bool:
        if not isinstance(payload, str) or not payload.strip():
            return False
        if not isinstance(signature, str) or not signature:
            return False
        try:
            expected = compute_dirac_positron_hmac(
                self.secret_key,
                payload,
                electron_count,
            )
            return hmac.compare_digest(signature, expected)
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _spohn_entropy_proxy(rho: NDArray[np.complex128]) -> float:
        """
        Indicador no negativo basado en D(ρ||I/N) y distancia de Frobenius.
        No afirma ser la producción de Spohn de un generador GKSL.
        """
        n = int(rho.shape[0])
        try:
            eigenvalues = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
        except np.linalg.LinAlgError:
            return 0.0

        probabilities = np.maximum(np.real(eigenvalues), 0.0)
        total = float(np.sum(probabilities))
        if not math.isfinite(total) or total <= _WILKINSON_EPSILON:
            return 0.0
        probabilities /= total
        positive = probabilities > 0.0
        kl = float(
            np.sum(
                probabilities[positive]
                * np.log(probabilities[positive] * float(n))
            )
        )
        reference = np.eye(n, dtype=np.complex128) / float(n)
        distance = float(la.norm(rho - reference, ord="fro"))
        return max(0.0, 0.15 * kl * distance)

    @staticmethod
    def _frequencies(dof: int) -> NDArray[np.float64]:
        return 1.0 + 0.25 * np.sin(
            _TWO_PI * np.arange(dof, dtype=np.float64) / float(max(dof, 1))
        )

    @staticmethod
    def _action_variables(
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return 0.5 * (
            omega * q * q + p * p / np.maximum(omega, _WILKINSON_EPSILON)
        )

    def quantize_indictment_to_poincare_phase_space(
        self,
        indictment: Any,
    ) -> PoincareProsecutionOrbitalCarrier:
        rho = self.sanitize_density_matrix_higham(
            getattr(indictment, "density_matrix", None)
        )
        omega = self._frequencies(self.dof)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (rho + rho.conj().T))
        eigenvalues = np.maximum(np.real(eigenvalues), 0.0)
        ground = eigenvectors[:, -1]
        amplitude = math.sqrt(float(max(eigenvalues[-1], 0.0)))

        n = int(rho.shape[0])
        q = np.zeros(self.dof, dtype=np.float64)
        p = np.zeros(self.dof, dtype=np.float64)
        diagonal = np.real(np.diag(rho))
        for k in range(self.dof):
            q[k] = (
                float(np.real(ground[k % n]) * amplitude)
                + 0.10 * float(diagonal[k % n])
            )
            p[k] = (
                float(np.imag(ground[k % n]) * amplitude)
                + 0.10 * float(diagonal[(k + self.dof) % n])
            )

        payload = str(getattr(indictment, "transaction_payload", ""))
        signature = str(getattr(indictment, "positron_auth_signature", ""))
        electron_count = _validated_anomaly_count(
            getattr(indictment, "electron_anomaly_count", 0)
        )
        auth_ok = self._dirac_auth_ok(payload, signature, electron_count)

        if not auth_ok:
            q += _DIRAC_PHASE_KICK
            p += _DIRAC_PHASE_KICK

        coupling = float(np.sum(np.cos(q - np.roll(q, -1))))
        energy = float(
            0.5 * np.dot(p, p)
            + 0.5 * np.dot(omega * omega, q * q)
            + _PROSECUTOR_PERTURBATION_EPS * coupling
        )
        actions = self._action_variables(q, p, omega)
        angles = np.arctan2(p, q * omega)

        connection, _, _, _ = self.extract_gauge_invariants(rho)
        wilson_operator = la.expm(connection)
        wilson_trace = complex(np.trace(wilson_operator) / float(max(n, 1)))
        su2_trace_proxy = float(2.0 * math.cos(cmath.phase(wilson_trace)))

        dirac_result = {
            "auth_ok": bool(auth_ok),
            "back_action_db": (
                0.0 if auth_ok else _DIRAC_AUTH_FAILURE_BACK_ACTION_DB
            ),
            "electrons_purged": electron_count if auth_ok else 0,
            "released_energy_kev": (
                float(electron_count * _POSITRON_REST_MASS_KEV)
                if auth_ok
                else 0.0
            ),
            "phase_kick": 0.0 if auth_ok else _DIRAC_PHASE_KICK,
            "is_simulation": True,
        }

        # Compatibilidad histórica: son acciones armónicas, no autovalores
        # simplécticos de Williamson de una matriz de covarianza.
        pseudo_williamson = tuple(
            sorted((float(x) for x in actions), reverse=True)
        )

        return PoincareProsecutionOrbitalCarrier(
            indictment_ref=indictment,
            sanitized_density_matrix=_readonly_array(rho, np.complex128),
            coordinates_q=_readonly_array(q, np.float64),
            momenta_p=_readonly_array(p, np.float64),
            action_variables_I=_readonly_array(actions, np.float64),
            angle_variables_theta=_readonly_array(angles, np.float64),
            hamiltonian_energy=energy,
            dirac_result=dirac_result,
            spohn_entropy_production=self._spohn_entropy_proxy(rho),
            gauge_connection_A=_readonly_array(connection, np.complex128),
            dimension_dof=self.dof,
            williamson_invariants=pseudo_williamson,
            su2_holonomy_trace=su2_trace_proxy,
            frequencies_omega=tuple(float(x) for x in omega),
        )

    def audit_homomorphic_gate_tears(
        self,
        rho: NDArray[np.complex128],
    ) -> Tuple[bool, float, float]:
        matrix = np.asarray(rho, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
            return True, 0.0, 1.0

        n = int(matrix.shape[0])
        b = max(1, n // 3)

        def corner_integrity(i0: int, i1: int) -> float:
            i0 = max(0, min(i0, n - 1))
            i1 = max(i0 + 1, min(i1, n))
            projection = np.zeros((n, n), dtype=np.complex128)
            projection[i0:i1, i0:i1] = np.eye(i1 - i0, dtype=np.complex128)
            pi_rho = projection @ matrix @ projection
            pi_rho2 = projection @ (matrix @ matrix) @ projection
            numerator = float(
                la.norm(pi_rho2 - pi_rho @ pi_rho, ord="fro")
            )
            denominator = max(
                float(la.norm(matrix, ord="fro")),
                _WILKINSON_EPSILON,
            )
            return float(math.exp(-numerator / denominator))

        i12 = corner_integrity(0, min(n, 2 * b))
        i23 = corner_integrity(min(n - 1, b), n)
        i13 = corner_integrity(0, n)
        defect = abs(i13 - i12 * i23)
        tear = bool(
            min(i12, i23, i13) < _GATE_TEAR_INTEGRITY_FLOOR
            or defect > _FUNCTOR_DEFECT_TOL
        )
        return tear, float(min(i12, i23, i13)), float(defect)

    def audit_indictment_claims(
        self,
        indictment: Any,
        rho: NDArray[np.complex128],
    ) -> bool:
        """Devuelve True si una declaración no coincide con invariantes re-medidos."""
        connection, _, yang_mills, chern_simons = self.extract_gauge_invariants(rho)
        connection_norm = float(la.norm(connection, ord="fro"))
        wilson = la.expm(connection)
        wilson_trace = complex(
            np.trace(wilson) / float(max(connection.shape[0], 1))
        )
        wilson_angle = float(cmath.phase(wilson_trace))
        instanton = int(round(chern_simons / (8.0 * math.pi**2)))

        declared = np.asarray(
            [
                _finite_float(
                    getattr(indictment, "connection_norm", connection_norm),
                    math.nan,
                ),
                _finite_float(
                    getattr(indictment, "yang_mills_energy", yang_mills),
                    math.nan,
                ),
                _finite_float(
                    getattr(indictment, "chern_simons_3form", chern_simons),
                    math.nan,
                ),
                _finite_float(
                    getattr(indictment, "wilson_angle_rad", wilson_angle),
                    math.nan,
                ),
            ],
            dtype=np.float64,
        )
        measured = np.asarray(
            [connection_norm, yang_mills, chern_simons, wilson_angle],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(declared)):
            return True

        denominator = max(float(np.linalg.norm(measured)), _WILKINSON_EPSILON)
        defect = float(np.linalg.norm(declared - measured) / denominator)
        declared_k = int(getattr(indictment, "instanton_number_k", instanton))
        trace_claim = bool(
            getattr(indictment, "trace_preservation_ok", True)
        )
        trace_measured = abs(float(np.real(np.trace(rho))) - 1.0) < 1.0e-5

        return bool(
            defect > _CLAIM_TAMPER_TOL
            or declared_k != instanton
            or trace_claim != trace_measured
        )


# ══════════════════════════════════════════════════════════════════════════════
# NÚCLEO CELESTE: YOSHIDA, SECCIÓN DE POINCARÉ, MONODROMÍA Y KAM FINITO
# ══════════════════════════════════════════════════════════════════════════════

class PoincareCelestialProsecutorEngine:
    """
    Dinámica hamiltoniana de referencia para:

        H(q,p) = 1/2 p·p + 1/2 Σ ω_k² q_k²
                 + ε Σ cos(q_k − q_(k+1)).

    La monodromía es el jacobiano tangente del mapa discreto Yoshida, no una
    monodromía analítica de una órbita periódica certificada.
    """

    def __init__(self, dimension: int) -> None:
        dim = max(4, int(dimension))
        if dim % 2:
            dim += 1
        self.dimension = dim
        self.dof = dim // 2
        self.eps = _PROSECUTOR_PERTURBATION_EPS

    @staticmethod
    def _omega(n: int) -> NDArray[np.float64]:
        return 1.0 + 0.25 * np.sin(
            _TWO_PI * np.arange(n, dtype=np.float64) / float(max(n, 1))
        )

    def _grad_v(
        self,
        q: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        q = np.asarray(q, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)
        gradient = omega * omega * q
        n = int(q.size)
        for k in range(n):
            j = (k + 1) % n
            sine = self.eps * math.sin(float(q[k] - q[j]))
            gradient[k] -= sine
            gradient[j] += sine
        return gradient

    def _hessian_v(
        self,
        q: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        q = np.asarray(q, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)
        n = int(q.size)
        hessian = np.diag(omega * omega).astype(np.float64, copy=False)

        # Hessiano simétrico de ε cos(q_k − q_(k+1)):
        # -ε cos(d) [[1,-1],[-1,1]] en las coordenadas k y k+1.
        for k in range(n):
            j = (k + 1) % n
            coefficient = self.eps * math.cos(float(q[k] - q[j]))
            hessian[k, k] -= coefficient
            hessian[j, j] -= coefficient
            hessian[k, j] += coefficient
            hessian[j, k] += coefficient
        return hessian

    def _hamiltonian(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> float:
        coupling = float(np.sum(np.cos(q - np.roll(q, -1))))
        return float(
            0.5 * np.dot(p, p)
            + 0.5 * np.dot(omega * omega, q * q)
            + self.eps * coupling
        )

    def _verlet_step_with_jacobian(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        omega: NDArray[np.float64],
        dt: float,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Un paso de Störmer–Verlet y su jacobiano tangente exacto para el mapa
        discreto. Los dos kicks usan el Hessiano en sus respectivos estados.
        """
        q0 = np.asarray(q, dtype=np.float64)
        p0 = np.asarray(p, dtype=np.float64)
        h0 = self._hessian_v(q0, omega)

        p_half = p0 - 0.5 * dt * self._grad_v(q0, omega)
        q1 = q0 + dt * p_half

        h1 = self._hessian_v(q1, omega)
        p1 = p_half - 0.5 * dt * self._grad_v(q1, omega)

        n = int(q0.size)
        eye = np.eye(n, dtype=np.float64)
        zero = np.zeros((n, n), dtype=np.float64)

        kick0 = np.block(
            [
                [eye, zero],
                [-0.5 * dt * h0, eye],
            ]
        )
        drift = np.block(
            [
                [eye, dt * eye],
                [zero, eye],
            ]
        )
        kick1 = np.block(
            [
                [eye, zero],
                [-0.5 * dt * h1, eye],
            ]
        )
        jacobian = kick1 @ drift @ kick0
        return q1, p1, jacobian

    def _verlet_step(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        omega: NDArray[np.float64],
        dt: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        q1, p1, _ = self._verlet_step_with_jacobian(q, p, omega, dt)
        return q1, p1

    def _yoshida_step_with_jacobian(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        omega: NDArray[np.float64],
        dt: float,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        n = int(np.asarray(q).size)
        tangent = np.eye(2 * n, dtype=np.float64)
        qn = np.array(q, dtype=np.float64, copy=True)
        pn = np.array(p, dtype=np.float64, copy=True)

        for weight in (_YOSHIDA_W1, _YOSHIDA_W0, _YOSHIDA_W1):
            qn, pn, jacobian = self._verlet_step_with_jacobian(
                qn, pn, omega, weight * dt
            )
            tangent = jacobian @ tangent

        return qn, pn, tangent

    def _yoshida_step(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        omega: NDArray[np.float64],
        dt: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        qn, pn, _ = self._yoshida_step_with_jacobian(q, p, omega, dt)
        return qn, pn

    def _henon_section(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        omega: NDArray[np.float64],
        steps: int = 64,
        dt: float = 0.05,
    ) -> PoincareSectionResult:
        if steps < 1 or dt <= 0.0 or not math.isfinite(dt):
            raise ValueError("steps y dt deben ser positivos y finitos.")

        qn = np.array(q, dtype=np.float64, copy=True)
        pn = np.array(p, dtype=np.float64, copy=True)
        points: List[Tuple[float, float]] = []
        q_previous = float(qn[0])
        p_previous = float(pn[0])

        for _ in range(int(steps)):
            qn, pn = self._yoshida_step(qn, pn, omega, dt)
            q_current = float(qn[0])
            p_current = float(pn[0])

            if q_previous <= 0.0 < q_current:
                denominator = q_current - q_previous
                alpha = (
                    0.0
                    if abs(denominator) <= _WILKINSON_EPSILON
                    else -q_previous / denominator
                )
                p_cross = p_previous + alpha * (p_current - p_previous)
                if p_cross > 0.0:
                    points.append((float(q_current), float(p_cross)))

            q_previous = q_current
            p_previous = p_current

        return PoincareSectionResult(
            crossing_count=len(points),
            crossing_points_qp=tuple(points),
        )

    def _analytic_monodromy(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        omega: NDArray[np.float64],
        dt: float = 0.05,
        steps: int = 8,
    ) -> ProsecutionMonodromyResult:
        if dt <= 0.0 or not math.isfinite(dt) or steps < 1:
            raise ValueError("dt y steps deben ser positivos y finitos.")

        qn = np.array(q, dtype=np.float64, copy=True)
        pn = np.array(p, dtype=np.float64, copy=True)
        n = int(qn.size)
        tangent = np.eye(2 * n, dtype=np.float64)

        for _ in range(int(steps)):
            qn, pn, step_jacobian = self._yoshida_step_with_jacobian(
                qn, pn, omega, dt
            )
            tangent = step_jacobian @ tangent

        identity = np.eye(n, dtype=np.float64)
        zero = np.zeros((n, n), dtype=np.float64)
        canonical_j = np.block(
            [
                [zero, identity],
                [-identity, zero],
            ]
        )
        defect_matrix = tangent.T @ canonical_j @ tangent - canonical_j
        defect = float(
            la.norm(defect_matrix, ord="fro")
            / max(math.sqrt(float(2 * n)), 1.0)
        )

        eigenvalues = np.linalg.eigvals(tangent)
        singular_values = la.svdvals(tangent)
        largest_singular = float(max(singular_values[0], _WILKINSON_EPSILON))
        period = float(steps * dt)
        finite_time_lyapunov = float(
            math.log(largest_singular) / max(period, _WILKINSON_EPSILON)
        )

        return ProsecutionMonodromyResult(
            monodromy_matrix=_readonly_array(tangent, np.float64),
            floquet_multipliers=tuple(
                complex(value)
                for value in eigenvalues[: min(8, eigenvalues.size)]
            ),
            max_lyapunov_exponent=finite_time_lyapunov,
            symplectic_defect=defect,
            period=period,
            integration_steps=int(steps),
            finite_time=True,
            largest_singular_value=largest_singular,
        )

    def _kam_audit(
        self,
        omega: NDArray[np.float64],
        lyap: float,
    ) -> ProsecutionKAMAuditResult:
        """
        Auditoría diofántica de orden Fourier finito.

        El resultado solo informa sobre la truncación comprobada; no certifica
        la desigualdad para todos los vectores enteros ni las hipótesis completas
        de KAM o Nekhoroshev.
        """
        frequencies = np.asarray(omega, dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("omega debe ser un vector no vacío.")
        if not np.all(np.isfinite(frequencies)):
            raise ValueError("omega contiene valores no finitos.")

        n = int(min(4, frequencies.size))
        om = frequencies[:n]
        tau = float(n)
        gamma = 0.04 / float(max(n * n, 1))
        order = int(min(_MAX_KAM_FOURIER_ORDER, 5))

        min_divisor = math.inf
        resonance_detected = False

        for k_tuple in itertools.product(
            range(-order, order + 1),
            repeat=n,
        ):
            if all(k_value == 0 for k_value in k_tuple):
                continue

            k_vector = np.asarray(k_tuple, dtype=np.float64)
            divisor = abs(float(np.dot(k_vector, om)))
            norm_one = float(np.linalg.norm(k_vector, ord=1))
            threshold = gamma / max(norm_one**tau, 1.0)
            min_divisor = min(min_divisor, divisor)
            if divisor < threshold:
                resonance_detected = True

        if not math.isfinite(min_divisor):
            min_divisor = 0.0

        lyap_value = _finite_float(lyap, 1.0e6)
        finite_order_ok = not resonance_detected
        stable = bool(finite_order_ok and lyap_value < 0.15)

        epsilon = max(abs(self.eps), _WILKINSON_EPSILON)
        exponent = min(
            40.0,
            gamma / max(epsilon ** (1.0 / float(2 * n)), _WILKINSON_EPSILON),
        )
        nekhoroshev_proxy = float(math.exp(max(0.0, exponent)))

        min_frequency = float(np.min(np.abs(om)))
        recurrence_proxy = float(
            _TWO_PI / max(min_frequency, _WILKINSON_EPSILON)
        )
        rotation = float(
            om[0] / max(
                abs(float(om[1])) if n > 1 else 1.0,
                _WILKINSON_EPSILON,
            )
        )

        return ProsecutionKAMAuditResult(
            is_kam_torus_stable=stable,
            diophantine_gamma=gamma,
            diophantine_tau=tau,
            min_small_divisor=float(min_divisor),
            nekhoroshev_time_bound=nekhoroshev_proxy,
            poincare_recurrence_time_estimate=recurrence_proxy,
            max_lyapunov_exponent=lyap_value,
            rotation_number=rotation,
            fourier_order=order,
            finite_order_certificate=finite_order_ok,
            resonance_detected=resonance_detected,
        )

    def compute_henon_poincare_section(
        self,
        orbital: PoincareProsecutionOrbitalCarrier,
    ) -> PoincareSectionResult:
        q = np.asarray(orbital.coordinates_q, dtype=np.float64)
        p = np.asarray(orbital.momenta_p, dtype=np.float64)
        omega = np.asarray(orbital.frequencies_omega, dtype=np.float64)
        if omega.size != q.size:
            omega = self._omega(int(q.size))
        return self._henon_section(q, p, omega)

    def compute_analytic_monodromy(
        self,
        orbital: PoincareProsecutionOrbitalCarrier,
    ) -> ProsecutionMonodromyResult:
        q = np.asarray(orbital.coordinates_q, dtype=np.float64)
        p = np.asarray(orbital.momenta_p, dtype=np.float64)
        omega = np.asarray(orbital.frequencies_omega, dtype=np.float64)
        if omega.size != q.size:
            omega = self._omega(int(q.size))
        return self._analytic_monodromy(q, p, omega)

    def audit_kam_resonances(
        self,
        orbital: PoincareProsecutionOrbitalCarrier,
        monodromy: Any = None,
    ) -> ProsecutionKAMAuditResult:
        q = np.asarray(orbital.coordinates_q, dtype=np.float64)
        omega = np.asarray(orbital.frequencies_omega, dtype=np.float64)
        if omega.size != q.size:
            omega = self._omega(int(q.size))
        lyap = _finite_float(
            getattr(monodromy, "max_lyapunov_exponent", 0.0),
            1.0e6,
        )
        return self._kam_audit(omega, lyap)

    def synthesize_prosecution_symplectic_feedback(
        self,
        orbital_carrier: PoincareProsecutionOrbitalCarrier,
        kam: Any = None,
        monodromy: Any = None,
        section: Any = None,
    ) -> ProsecutionSymplecticFeedbackCarrier:
        q = np.asarray(orbital_carrier.coordinates_q, dtype=np.float64)
        p = np.asarray(orbital_carrier.momenta_p, dtype=np.float64)
        if q.ndim != 1 or p.ndim != 1 or q.size == 0 or q.size != p.size:
            raise ValueError("El estado orbital debe contener q y p compatibles.")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            raise ValueError("El estado orbital contiene valores no finitos.")

        omega = np.asarray(orbital_carrier.frequencies_omega, dtype=np.float64)
        if (
            omega.ndim != 1
            or omega.size != q.size
            or not np.all(np.isfinite(omega))
            or np.any(omega <= 0.0)
        ):
            omega = self._omega(int(q.size))

        e0 = self._hamiltonian(q, p, omega)
        qn = np.array(q, copy=True)
        pn = np.array(p, copy=True)
        integration_steps = 8
        dt = 0.05
        for _ in range(integration_steps):
            qn, pn = self._yoshida_step(qn, pn, omega, dt)
        e1 = self._hamiltonian(qn, pn, omega)
        energy_drift = abs(e1 - e0) / max(abs(e0), _WILKINSON_EPSILON)

        if monodromy is None:
            monodromy = self._analytic_monodromy(
                q, p, omega, dt=dt, steps=integration_steps
            )
        lyap = _finite_float(
            getattr(monodromy, "max_lyapunov_exponent", 0.0),
            1.0e6,
        )

        if kam is None:
            kam = self._kam_audit(omega, lyap)

        if section is None:
            section = self._henon_section(q, p, omega)

        section_count = int(
            _read_field(
                section,
                "crossing_count",
                _read_field(section, "section_count", 0),
            )
            or 0
        )
        pairing = max(
            0.0,
            _finite_float(
                getattr(monodromy, "symplectic_defect", 0.0),
                1.0e6,
            ),
        )
        drift_penalty = float(energy_drift + pairing)
        conserved = bool(
            drift_penalty < _SYMPLECTIC_DRIFT_TOL * 10.0
            and energy_drift < 1.0e-3
        )

        connection = np.asarray(
            orbital_carrier.gauge_connection_A,
            dtype=np.complex128,
        )
        connection_norm = float(la.norm(connection, ord="fro"))
        germ = orbital_carrier.indictment_ref
        trace_ok = bool(getattr(germ, "trace_preservation_ok", True))

        # Recalcula los proxies a partir del estado matricial; las declaraciones
        # del germen no se usan como si fueran mediciones independientes.
        try:
            _, _, measured_ym, measured_cs = (
                BanachGaugeDiracProsecutionCore(
                    max(4, 2 * int(orbital_carrier.dimension_dof)),
                    b"local-reference-key",
                ).extract_gauge_invariants(
                    orbital_carrier.sanitized_density_matrix
                )
            )
        except (ValueError, np.linalg.LinAlgError):
            measured_ym, measured_cs = 1.0e6, 1.0e6

        wilson_operator = la.expm(connection)
        wilson_trace = complex(
            np.trace(wilson_operator) / float(max(connection.shape[0], 1))
        )
        wilson_angle = float(cmath.phase(wilson_trace))
        instanton = int(round(measured_cs / (8.0 * math.pi**2)))

        holonomy_breakage = bool(
            (not trace_ok)
            or connection_norm > 2.5
            or measured_ym > 0.85
            or abs(wilson_angle) >= 0.50
            or abs(instanton) > 2
        )

        dirac = orbital_carrier.dirac_result
        auth_ok = _dirac_auth_flag(dirac)
        kam_ok = bool(
            getattr(kam, "is_kam_torus_stable", True)
        )
        claim_tamper = bool(orbital_carrier.claim_tamper_detected)
        gate_tear = bool(orbital_carrier.gate_tear_detected)

        if claim_tamper:
            status = OODAProsecutionStatus.VETOED_INDICTMENT_TAMPER
            verdict = HeytingOmega4.VETOED
        elif not auth_ok:
            status = OODAProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE
            verdict = HeytingOmega4.VETOED
        elif gate_tear:
            status = OODAProsecutionStatus.VETOED_HOMOMORPHIC_GATE_TEAR
            verdict = HeytingOmega4.VETOED
        elif holonomy_breakage:
            status = OODAProsecutionStatus.VETOED_HOLONOMY_BREAKAGE
            verdict = HeytingOmega4.VETOED
        elif not conserved:
            status = OODAProsecutionStatus.VETOED_SYMPLECTIC_DRIFT
            verdict = HeytingOmega4.VETOED
        elif not kam_ok:
            status = OODAProsecutionStatus.VETOED_KAM_CHAOTIC_RESONANCE
            verdict = HeytingOmega4.CRITICAL
        elif int(_read_field(dirac, "electrons_purged", 0) or 0) > 0:
            status = OODAProsecutionStatus.PURGED_DIRAC_VACUUM
            verdict = HeytingOmega4.COHERENT
        else:
            status = OODAProsecutionStatus.ABSOLVED
            verdict = HeytingOmega4.COHERENT

        return ProsecutionSymplecticFeedbackCarrier(
            orbital_carrier=orbital_carrier,
            monodromy_result=monodromy,
            kam_audit=kam,
            symplectic_drift_penalty=drift_penalty,
            holonomy_breakage_detected=holonomy_breakage,
            intermediate_verdict=verdict,
            prosecution_status_tentative=status,
            energy_drift=float(energy_drift),
            section_count=section_count,
            pairing_defect=pairing,
            symplectic_conserved=conserved,
            claim_tamper_detected=claim_tamper,
            gate_tear_detected=gate_tear,
        )


# ══════════════════════════════════════════════════════════════════════════════
# RSI-L3 Y MOTOR LOCAL DE REFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

class ProsecutionRSIL3DecisionCore:
    """
    Iterador de punto fijo escalar RSI-L3.

    Para datos de entrada fijos, define
        S(λ) = clip(κ λ + (1−κ) target, 0.50, 0.99),  κ=0.80.
    Como clip es 1-Lipschitz, Lip(S) ≤ κ < 1. El código itera hasta tolerancia
    o un máximo finito de iteraciones y registra el residuo de Cauchy.
    """

    def __init__(self, secret_key: bytes) -> None:
        self.secret_key = bytes(secret_key)
        self.generation = 0
        self.lam = _DEFAULT_BANACH_CONTRACTION_LIP
        self.contraction_modulus = 0.80
        self.lower_bound = 0.50
        self.upper_bound = 0.99

    def _meta_S(
        self,
        lam: float,
        residual: float,
        lip_emp: float,
        beta: float = 0.15,
        eta: float = 0.08,
    ) -> float:
        """Una iteración del operador contractivo RSI de nivel 3."""
        residual = float(np.clip(_finite_float(residual), 0.0, 1.0))
        lip_emp = float(np.clip(_finite_float(lip_emp), 0.0, 0.999999))
        target = float(
            np.clip(
                0.50 + 0.49 * lip_emp - beta * residual + eta * (1.0 - lip_emp),
                self.lower_bound,
                self.upper_bound,
            )
        )
        result = (
            self.contraction_modulus * float(lam)
            + (1.0 - self.contraction_modulus) * target
        )
        return float(np.clip(result, self.lower_bound, self.upper_bound))

    def trigger_recursive_step(
        self,
        lip_emp: float,
        residual: float,
        tolerance: float = 1.0e-12,
        max_iterations: int = 256,
    ) -> Tuple[ProsecutorRSIConfiguration, float, float]:
        if not math.isfinite(float(lip_emp)) or not math.isfinite(float(residual)):
            raise ValueError("Las entradas RSI deben ser finitas.")
        if tolerance <= 0.0 or max_iterations < 1:
            raise ValueError("tolerance y max_iterations deben ser positivos.")

        lip_emp = float(np.clip(lip_emp, 0.0, 0.999999))
        residual = float(np.clip(max(0.0, residual), 0.0, 1.0))
        current = float(np.clip(self.lam, self.lower_bound, self.upper_bound))
        delta = math.inf
        converged = False
        iterations = 0

        for iterations in range(1, int(max_iterations) + 1):
            next_value = self._meta_S(current, residual, lip_emp)
            delta = abs(next_value - current)
            current = next_value
            if delta <= tolerance:
                converged = True
                break

        self.lam = current
        self.generation += 1
        error_bound = delta / max(
            1.0 - self.contraction_modulus,
            _WILKINSON_EPSILON,
        )
        lip_t = min(lip_emp, current)

        config = ProsecutorRSIConfiguration(
            generation=self.generation,
            banach_lipschitz=float(lip_t),
            meta_operator_lip=float(self.contraction_modulus),
            cauchy_residual=float(delta),
            psd_floor=_SPECTRAL_PSD_FLOOR,
            symplectic_tolerance=_SYMPLECTIC_DRIFT_TOL,
            fixed_point_value=float(current),
            fixed_point_iterations=int(iterations),
            fixed_point_converged=bool(converged),
            fixed_point_error_bound=float(error_bound),
        )
        return config, float(lip_t), float(delta)

    def simulate_esp32_safety_interlock(
        self,
        vetoed: bool,
        seed_material: bytes,
    ) -> Tuple[bool, float]:
        """
        Simula una latencia determinista. No accede a GPIO, MMIO ni hardware.
        """
        if not vetoed:
            return False, 0.0
        digest = hashlib.sha384(
            b"ESP32-SIMULATION-ONLY-v15|" + bytes(seed_material)
        ).digest()
        unit = int.from_bytes(digest[:4], "big") / float(2**32)
        span = _ESP32_CROWBAR_MAX_LATENCY_NS - _ESP32_CROWBAR_MIN_LATENCY_NS
        latency = _ESP32_CROWBAR_MIN_LATENCY_NS + unit * span
        latency = float(
            min(
                max(latency, _ESP32_CROWBAR_MIN_LATENCY_NS),
                _ESP32_CROWBAR_MAX_LATENCY_NS,
            )
        )
        return True, latency

    def adjudicate_and_execute(
        self,
        feedback: ProsecutionSymplecticFeedbackCarrier,
    ) -> OODAProsecutionExecutionCertificatePhD:
        """
        Adjudicación local. La RSI-L3 se ejecuta una sola vez, posteriormente,
        en el cierre de la Fase III.
        """
        now = time.time()
        orbital = feedback.orbital_carrier
        dirac = orbital.dirac_result

        auth_ok = _dirac_auth_flag(dirac)
        back_action = _finite_float(
            _read_field(dirac, "back_action_db", 0.0),
            _DIRAC_AUTH_FAILURE_BACK_ACTION_DB,
        )
        purged = max(
            0,
            int(_read_field(dirac, "electrons_purged", 0) or 0),
        )
        released = max(
            0.0,
            _finite_float(_read_field(dirac, "released_energy_kev", 0.0)),
        )

        kam = feedback.kam_audit
        kam_ok = bool(getattr(kam, "is_kam_torus_stable", True))
        lyap = _finite_float(
            getattr(
                kam,
                "max_lyapunov_exponent",
                getattr(feedback.monodromy_result, "max_lyapunov_exponent", 0.0),
            ),
            1.0e6,
        )
        drift = max(
            0.0,
            _finite_float(feedback.symplectic_drift_penalty, 1.0e6),
        )
        energy_drift = max(
            0.0,
            _finite_float(feedback.energy_drift, 1.0e6),
        )

        if feedback.claim_tamper_detected:
            status, verdict = (
                OODAProsecutionStatus.VETOED_INDICTMENT_TAMPER,
                HeytingOmega4.VETOED,
            )
        elif not auth_ok:
            status, verdict = (
                OODAProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE,
                HeytingOmega4.VETOED,
            )
        elif feedback.gate_tear_detected:
            status, verdict = (
                OODAProsecutionStatus.VETOED_HOMOMORPHIC_GATE_TEAR,
                HeytingOmega4.VETOED,
            )
        elif feedback.holonomy_breakage_detected:
            status, verdict = (
                OODAProsecutionStatus.VETOED_HOLONOMY_BREAKAGE,
                HeytingOmega4.VETOED,
            )
        elif (
            drift > 1.0e-4
            or energy_drift > 1.0e-3
            or not feedback.symplectic_conserved
        ):
            status, verdict = (
                OODAProsecutionStatus.VETOED_SYMPLECTIC_DRIFT,
                HeytingOmega4.VETOED,
            )
        elif not kam_ok or lyap >= 0.15:
            status, verdict = (
                OODAProsecutionStatus.VETOED_KAM_CHAOTIC_RESONANCE,
                HeytingOmega4.CRITICAL,
            )
        elif purged > 0:
            status, verdict = (
                OODAProsecutionStatus.PURGED_DIRAC_VACUUM,
                HeytingOmega4.COHERENT,
            )
        else:
            status, verdict = (
                OODAProsecutionStatus.ABSOLVED,
                HeytingOmega4.COHERENT,
            )

        crowbar, latency = self.simulate_esp32_safety_interlock(
            verdict == HeytingOmega4.VETOED,
            f"{status.name}:{verdict.name}:{lyap:.12g}".encode("utf-8"),
        )

        return OODAProsecutionExecutionCertificatePhD(
            certificate_id=f"CERT-PROSECUTOR-v15-{uuid.uuid4().hex}",
            status=status,
            global_heyting_verdict=verdict,
            crowbar_triggered=crowbar,
            crowbar_triggered_simulated=crowbar,
            crowbar_latency_ns=float(latency),
            electron_anomalies_purged=purged,
            released_energy_kev=released,
            back_action_db=back_action,
            rsi_level3_generation=int(self.generation),
            rsi_banach_contraction_lip=float(
                min(_DEFAULT_BANACH_CONTRACTION_LIP, 0.999999)
            ),
            rsi_meta_operator_lip=float(self.contraction_modulus),
            rsi_cauchy_residual=0.0,
            spohn_entropy_production=max(
                0.0,
                _finite_float(orbital.spohn_entropy_production),
            ),
            is_kam_torus_stable=kam_ok,
            max_lyapunov_exponent=lyap,
            symplectic_drift=drift,
            energy_drift=energy_drift,
            timestamp_utc=now,
        )


class OODATriphasicHomomorphicPipelineProsecutorEnginePhD:
    """Motor local autocontenido; las fases de control siguen siendo simuladas."""

    def __init__(
        self,
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        dim = max(4, int(dimension))
        if dim % 2:
            dim += 1
        self.dimension = dim
        self.secret_key = bytes(secret_key)
        self.phase1_core = BanachGaugeDiracProsecutionCore(dim, self.secret_key)
        self.phase2_core = PoincareCelestialProsecutorEngine(dim)
        self.phase3_core = ProsecutionRSIL3DecisionCore(self.secret_key)


# ══════════════════════════════════════════════════════════════════════════════
# FASE I: OBSERVE — CALIBRE, ESTADO DENSIDAD Y AUTORIZACIÓN DIRAC
# El último método es el morfismo firmado que inicia la Fase II.
# ══════════════════════════════════════════════════════════════════════════════

class OODAPipelineProsecutorGaugeTopologyPhD:
    """
    Fase I — Observe.

    Sanitiza ρ, genera proxies matriciales de calibre, comprueba las declaraciones
    y crea un contenedor P1→P2 autenticado. El contrato y la transferencia tienen
    firmas distintas: la segunda cubre también las matrices y el estado orbital.
    """

    def __init__(
        self,
        sovereign_id: str = "SOVEREIGN-OODA-PROSECUTOR-v15",
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        engine: Optional[Any] = None,
    ) -> None:
        if not isinstance(secret_key, (bytes, bytearray)) or not secret_key:
            raise ValueError("secret_key debe ser bytes no vacíos.")

        dim = max(4, int(dimension))
        if dim % 2:
            dim += 1

        self.sovereign_id: Final[str] = str(sovereign_id)
        self.dimension: Final[int] = dim
        self.secret_key: Final[bytes] = bytes(secret_key)

        if engine is None:
            self.engine = OODATriphasicHomomorphicPipelineProsecutorEnginePhD(
                dimension=dim,
                secret_key=self.secret_key,
            )
            self._fallback_phase1_core = self.engine.phase1_core
            self._fallback_phase2_core = self.engine.phase2_core
            self._fallback_phase3_core = self.engine.phase3_core
        else:
            self.engine = engine
            # Las rutas locales se conservan como fallback verificable.
            self._fallback_phase1_core = BanachGaugeDiracProsecutionCore(
                dim, self.secret_key
            )
            self._fallback_phase2_core = PoincareCelestialProsecutorEngine(dim)
            self._fallback_phase3_core = ProsecutionRSIL3DecisionCore(
                self.secret_key
            )

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(
            self.secret_key,
            domain + payload.encode("utf-8"),
            hashlib.sha384,
        ).hexdigest()

    def _canonical_p1_payload(
        self,
        contract: OODAProsecutorGaugeContractPhD,
    ) -> str:
        return _canonical_json(
            {
                "contract_id": contract.contract_id,
                "sovereign_id": contract.sovereign_id,
                "chern_simons_3form": _float_token(contract.chern_simons_3form),
                "hopf_c1_invariant": _float_token(contract.hopf_c1_invariant),
                "wilson_angle_rad": _float_token(contract.wilson_angle_rad),
                "connection_norm": _float_token(contract.connection_norm),
                "yang_mills_energy": _float_token(contract.yang_mills_energy),
                "instanton_number_k": int(contract.instanton_number_k),
                "su2_holonomy_trace": _float_token(contract.su2_holonomy_trace),
                "purity": _float_token(contract.purity),
                "dof_cotangent": int(contract.dof_cotangent),
                "trace_preservation_ok": bool(contract.trace_preservation_ok),
                "gate_tear_detected": bool(contract.gate_tear_detected),
                "isometry_verdict": int(contract.isometry_verdict),
                "chain_hash": contract.chain_hash,
                "timestamp_utc": _float_token(contract.timestamp_utc),
            }
        )

    def sanitize_density_matrix(self, matrix: NDArray[Any]) -> NDArray[np.complex128]:
        """Proyección espectral estable al simplex de estados densidad."""
        return self._fallback_phase1_core.sanitize_density_matrix_higham(matrix)

    def extract_suN_gauge_bundle(
        self,
        rho: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], float, float]:
        return self._fallback_phase1_core.extract_gauge_invariants(rho)

    @staticmethod
    def compute_chern_simons_and_instanton(cs3_val: float) -> int:
        """Índice entero derivado de un proxy; no es un número de Pontryagin medido."""
        if not math.isfinite(float(cs3_val)):
            return 0
        return int(round(float(cs3_val) / (8.0 * math.pi**2)))

    @staticmethod
    def compute_wilson_holonomy(
        conn_a: NDArray[np.complex128],
    ) -> Tuple[float, float]:
        matrix = np.asarray(conn_a, dtype=np.complex128)
        n = max(int(matrix.shape[0]), 1)
        wilson_operator = la.expm(matrix)
        wilson_trace = complex(np.trace(wilson_operator) / float(n))
        angle = float(cmath.phase(wilson_trace))
        su2_trace_proxy = float(2.0 * math.cos(angle))
        return angle, su2_trace_proxy

    @staticmethod
    def adjudicate_heyting_isometry(
        tr_ok: bool,
        conn_norm: float,
        ym_energy: float,
        wilson_angle: float,
        instanton_k: int,
    ) -> HeytingOmega4:
        if (
            not tr_ok
            or conn_norm > 2.5
            or ym_energy > 0.85
            or abs(wilson_angle) >= 0.50
            or abs(instanton_k) > 2
        ):
            return HeytingOmega4.VETOED
        if conn_norm > 1.2 or ym_energy > 0.40:
            return HeytingOmega4.CRITICAL
        if conn_norm > 0.7 or ym_energy > 0.15:
            return HeytingOmega4.DEGRADED
        return HeytingOmega4.COHERENT

    def forge_prosecutor_gauge_contract(
        self,
        density_matrix: NDArray[Any],
    ) -> Tuple[
        OODAProsecutorGaugeContractPhD,
        NDArray[np.complex128],
        NDArray[np.complex128],
        bool,
    ]:
        now = time.time()
        rho = self.sanitize_density_matrix(density_matrix)
        trace_value = float(np.real(np.trace(rho)))
        trace_ok = abs(trace_value - 1.0) < 1.0e-5

        connection, curvature_proxy, ym_proxy, cs_proxy = (
            self.extract_suN_gauge_bundle(rho)
        )
        connection_norm = float(la.norm(connection, ord="fro"))
        wilson_angle, su2_trace = self.compute_wilson_holonomy(connection)
        hopf_proxy = float(np.real(np.trace(curvature_proxy)) / _TWO_PI)
        instanton_proxy = self.compute_chern_simons_and_instanton(cs_proxy)
        purity = float(np.real(np.trace(rho @ rho)))

        try:
            gate_tear, _, _ = (
                self._fallback_phase1_core.audit_homomorphic_gate_tears(rho)
            )
        except (TypeError, ValueError, AttributeError, np.linalg.LinAlgError):
            # Una auditoría que no puede ejecutarse falla cerrada.
            gate_tear = True

        verdict = self.adjudicate_heyting_isometry(
            trace_ok,
            connection_norm,
            ym_proxy,
            wilson_angle,
            instanton_proxy,
        )
        if gate_tear:
            verdict = HeytingOmega4.VETOED

        contract_id = f"CONTRACT-OODA-v15-{uuid.uuid4().hex}"
        chain_payload = _canonical_json(
            {
                "contract_id": contract_id,
                "sovereign_id": self.sovereign_id,
                "chern_simons_proxy": _float_token(cs_proxy),
                "wilson_angle": _float_token(wilson_angle),
                "instanton_proxy": int(instanton_proxy),
                "verdict": int(verdict),
            }
        )
        chain_hash = hashlib.sha384(
            chain_payload.encode("utf-8")
        ).hexdigest()[:_CHAIN_HEX_LEN]

        contract = OODAProsecutorGaugeContractPhD(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            chern_simons_3form=float(cs_proxy),
            hopf_c1_invariant=float(hopf_proxy),
            wilson_angle_rad=float(wilson_angle),
            connection_norm=float(connection_norm),
            yang_mills_energy=float(ym_proxy),
            instanton_number_k=int(instanton_proxy),
            su2_holonomy_trace=float(su2_trace),
            purity=float(purity),
            dof_cotangent=int(self.dimension // 2),
            trace_preservation_ok=bool(trace_ok),
            gate_tear_detected=bool(gate_tear),
            isometry_verdict=verdict,
            chain_hash=chain_hash,
            hmac_p1_signature="",
            timestamp_utc=float(now),
        )
        contract = replace(
            contract,
            hmac_p1_signature=self._compute_hmac(
                _HMAC_DOMAIN_P1,
                self._canonical_p1_payload(contract),
            ),
        )
        return (
            contract,
            _readonly_array(rho, np.complex128),
            _readonly_array(connection, np.complex128),
            bool(gate_tear),
        )

    def continue_to_phase2_prosecutor_gate(
        self,
        contract: OODAProsecutorGaugeContractPhD,
    ) -> bool:
        try:
            expected = self._compute_hmac(
                _HMAC_DOMAIN_P1,
                self._canonical_p1_payload(contract),
            )
            return hmac.compare_digest(
                str(contract.hmac_p1_signature),
                expected,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    def _construct_indictment_germ(
        self,
        **kwargs: Any,
    ) -> OODAProsecutionIndictmentGermPhD:
        return OODAProsecutionIndictmentGermPhD(**kwargs)

    def _audit_indictment_claims(
        self,
        indictment: Any,
        rho: NDArray[np.complex128],
    ) -> bool:
        try:
            return bool(
                self._fallback_phase1_core.audit_indictment_claims(
                    indictment,
                    rho,
                )
            )
        except (TypeError, ValueError, AttributeError, np.linalg.LinAlgError):
            return True

    def _quantize_indictment(
        self,
        indictment: Any,
    ) -> PoincareProsecutionOrbitalCarrier:
        return self._fallback_phase1_core.quantize_indictment_to_poincare_phase_space(
            indictment
        )

    def _canonical_phase1_transfer_payload(
        self,
        carrier: ProsecutorPhase1TransferCarrier,
    ) -> str:
        contract = carrier.gauge_contract
        orbital = carrier.poincare_orbital_carrier
        germ = carrier.indictment_germ
        dirac = orbital.dirac_result

        payload = str(getattr(germ, "transaction_payload", ""))
        signature = str(getattr(germ, "positron_auth_signature", ""))
        count = int(getattr(germ, "electron_anomaly_count", 0))

        return _canonical_json(
            {
                "contract_id": contract.contract_id,
                "contract_signature": contract.hmac_p1_signature,
                "indictment_id": str(getattr(germ, "indictment_id", "")),
                "transaction_payload_sha384": hashlib.sha384(
                    payload.encode("utf-8")
                ).hexdigest(),
                "positron_signature": signature,
                "electron_anomaly_count": count,
                "dirac_auth_ok": bool(carrier.dirac_auth_ok),
                "claim_tamper_detected": bool(carrier.claim_tamper_detected),
                "gate_tear_detected": bool(contract.gate_tear_detected),
                "density_matrix_sha384": _array_sha384(
                    carrier.sanitized_density_matrix,
                    complex_array=True,
                ),
                "gauge_connection_sha384": _array_sha384(
                    carrier.gauge_connection,
                    complex_array=True,
                ),
                "orbital_density_sha384": _array_sha384(
                    orbital.sanitized_density_matrix,
                    complex_array=True,
                ),
                "q_sha384": _array_sha384(orbital.coordinates_q),
                "p_sha384": _array_sha384(orbital.momenta_p),
                "actions_sha384": _array_sha384(orbital.action_variables_I),
                "angles_sha384": _array_sha384(orbital.angle_variables_theta),
                "hamiltonian_energy": _float_token(orbital.hamiltonian_energy),
                "spohn_proxy": _float_token(orbital.spohn_entropy_production),
                "dirac_result": dirac,
                "dof": int(orbital.dimension_dof),
                "williamson_action_proxy": [
                    _float_token(value)
                    for value in carrier.williamson_invariants
                ],
                "frequencies": [
                    _float_token(value)
                    for value in orbital.frequencies_omega
                ],
            }
        )

    def verify_phase1_transfer(
        self,
        carrier: ProsecutorPhase1TransferCarrier,
    ) -> bool:
        try:
            if not self.continue_to_phase2_prosecutor_gate(
                carrier.gauge_contract
            ):
                return False
            expected = self._compute_hmac(
                _HMAC_DOMAIN_P1_TRANSFER,
                self._canonical_phase1_transfer_payload(carrier),
            )
            return hmac.compare_digest(
                str(carrier.hmac_transfer_signature),
                expected,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # MORFISMO FINAL DE FASE I ──► INICIO DE FASE II
    # ──────────────────────────────────────────────────────────────────────────
    def bridge_phase1_to_phase2_campaign_transfer(
        self,
        density_matrix: NDArray[Any],
        electron_anomaly_count: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
    ) -> ProsecutorPhase1TransferCarrier:
        """
        Construye y firma el estado completo que consume la Fase II.

        La firma Dirac de simulación cubre tanto transaction_payload como
        electron_anomaly_count. La firma de transferencia P1 cubre además las
        matrices, el estado orbital y la firma del contrato.
        """
        electron_count = _validated_anomaly_count(electron_anomaly_count)
        if not isinstance(positron_auth_signature, str):
            raise TypeError("positron_auth_signature debe ser str.")
        if not isinstance(transaction_payload, str):
            raise TypeError("transaction_payload debe ser str.")

        contract, rho, connection, gate_tear = (
            self.forge_prosecutor_gauge_contract(density_matrix)
        )
        indictment = self._construct_indictment_germ(
            indictment_id=f"INDICT-{contract.contract_id}",
            trace_preservation_ok=contract.trace_preservation_ok,
            connection_norm=contract.connection_norm,
            yang_mills_energy=contract.yang_mills_energy,
            chern_simons_3form=contract.chern_simons_3form,
            wilson_angle_rad=contract.wilson_angle_rad,
            instanton_number_k=contract.instanton_number_k,
            electron_anomaly_count=electron_count,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
            density_matrix=rho,
            timestamp_utc=contract.timestamp_utc,
        )

        claim_tamper = self._audit_indictment_claims(indictment, rho)
        orbital = self._quantize_indictment(indictment)
        dirac_ok = _dirac_auth_flag(orbital.dirac_result)

        orbital = replace(
            orbital,
            claim_tamper_detected=bool(claim_tamper),
            gate_tear_detected=bool(gate_tear),
        )
        carrier = ProsecutorPhase1TransferCarrier(
            gauge_contract=contract,
            poincare_orbital_carrier=orbital,
            sanitized_density_matrix=rho,
            gauge_connection=connection,
            indictment_germ=indictment,
            dirac_auth_ok=bool(dirac_ok),
            claim_tamper_detected=bool(claim_tamper),
            williamson_invariants=tuple(
                float(value)
                for value in (orbital.williamson_invariants or ())
            ),
            energy_hamiltonian=float(orbital.hamiltonian_energy),
            hmac_transfer_signature="",
        )
        carrier = replace(
            carrier,
            hmac_transfer_signature=self._compute_hmac(
                _HMAC_DOMAIN_P1_TRANSFER,
                self._canonical_phase1_transfer_payload(carrier),
            ),
        )

        logger.debug(
            "P1→P2: contrato=%s DOF=%d energía=%.8g Dirac=%s tamper=%s tear=%s",
            contract.contract_id,
            orbital.dimension_dof,
            orbital.hamiltonian_energy,
            dirac_ok,
            claim_tamper,
            gate_tear,
        )
        return carrier


# ══════════════════════════════════════════════════════════════════════════════
# FASE II: ORIENT — MAPA DE POINCARÉ, MONODROMÍA Y AUDITORÍA KAM FINITA
# El último método firma la transferencia que consume la Fase III.
# ══════════════════════════════════════════════════════════════════════════════

class OODAProsecutionCampaignOrientorPhD(
    OODAPipelineProsecutorGaugeTopologyPhD
):
    """
    Fase II — Orient.

    Verifica el HMAC de la transferencia P1, integra Yoshida O(4), calcula una
    sección de Poincaré y el jacobiano tangente simpléctico, y registra una
    auditoría KAM truncada. No adjudica ni accede a hardware.
    """

    def _phase2_core(self) -> Any:
        core = getattr(self.engine, "phase2_core", None)
        return core if core is not None else self._fallback_phase2_core

    def _estimate_symplectic_lipschitz(
        self,
        drift: float,
        lyapunov: float,
        kam_ok: bool,
    ) -> float:
        drift = max(0.0, _finite_float(drift, 1.0e6))
        lyapunov = abs(_finite_float(lyapunov, 1.0e6))
        base = 0.50 + 0.49 * math.tanh(
            1.0 + drift / max(_SYMPLECTIC_DRIFT_TOL, 1.0e-18)
        )
        base = max(base, min(0.99, 0.70 + lyapunov))
        if not kam_ok:
            base = max(base, 0.92)
        cap = _DEFAULT_BANACH_CONTRACTION_LIP if kam_ok else 0.97
        return float(np.clip(min(base, cap, 0.99), 0.0, 0.999999))

    def _invoke_named(
        self,
        obj: Any,
        names: Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        for name in names:
            function = getattr(obj, name, None)
            if not callable(function):
                continue
            try:
                return function(*args, **kwargs)
            except TypeError:
                try:
                    return function(*args)
                except TypeError:
                    continue
            except Exception as exc:
                logger.warning("Fallo en método numérico %s: %s", name, exc)
                return None
        return None

    def _synthesize_feedback_resilient(
        self,
        orbital: PoincareProsecutionOrbitalCarrier,
        kam: Any,
        monodromy: Any,
        section: Any,
    ) -> ProsecutionSymplecticFeedbackCarrier:
        synthesizer = getattr(
            self._phase2_core(),
            "synthesize_prosecution_symplectic_feedback",
            None,
        )
        if not callable(synthesizer):
            synthesizer = getattr(
                self._fallback_phase2_core,
                "synthesize_prosecution_symplectic_feedback",
            )

        attempts = (
            lambda: synthesizer(
                orbital,
                kam=kam,
                monodromy=monodromy,
                section=section,
            ),
            lambda: synthesizer(orbital, kam, monodromy, section),
            lambda: synthesizer(orbital),
        )
        last_error: Optional[BaseException] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
        raise TypeError(
            "No se pudo invocar synthesize_prosecution_symplectic_feedback "
            f"con una firma compatible: {last_error}"
        )

    def _canonical_p2_payload(
        self,
        campaign: OODAProsecutionCampaignResultPhD,
    ) -> str:
        return _canonical_json(
            {
                "campaign_id": campaign.campaign_id,
                "bound_contract_id": campaign.bound_contract_id,
                "lipschitz_contraction": _float_token(
                    campaign.lipschitz_contraction
                ),
                "spohn_entropy_production": _float_token(
                    campaign.spohn_entropy_production
                ),
                "is_kam_torus_stable": bool(campaign.is_kam_torus_stable),
                "max_lyapunov_exponent": _float_token(
                    campaign.max_lyapunov_exponent
                ),
                "poincare_recurrence_time": _float_token(
                    campaign.poincare_recurrence_time
                ),
                "symplectic_drift_penalty": _float_token(
                    campaign.symplectic_drift_penalty
                ),
                "holonomy_breakage_detected": bool(
                    campaign.holonomy_breakage_detected
                ),
                "nekhoroshev_bound": _float_token(campaign.nekhoroshev_bound),
                "rotation_number": _float_token(campaign.rotation_number),
                "section_count": int(campaign.section_count),
                "energy_drift": _float_token(campaign.energy_drift),
                "pairing_defect": _float_token(campaign.pairing_defect),
                "symplectic_conserved": bool(campaign.symplectic_conserved),
                "qnd_orient_verdict": int(campaign.qnd_orient_verdict),
                "prosecution_status_tentative": int(
                    campaign.prosecution_status_tentative
                ),
                "chain_hash": campaign.chain_hash,
                "timestamp_utc": _float_token(campaign.timestamp_utc),
                "phase1_transfer_signature": campaign.phase1_transfer_signature,
                "dirac_auth_ok": bool(campaign.dirac_auth_ok),
                "claim_tamper_detected": bool(
                    campaign.claim_tamper_detected
                ),
                "gate_tear_detected": bool(campaign.gate_tear_detected),
            }
        )

    def continue_to_phase3_prosecutor_gate(
        self,
        campaign: OODAProsecutionCampaignResultPhD,
    ) -> bool:
        """Valida firma P2 y Lip(T) estrictamente menor que uno."""
        try:
            lip = float(campaign.lipschitz_contraction)
            if not math.isfinite(lip) or not (0.0 <= lip < 1.0):
                return False
            expected = self._compute_hmac(
                _HMAC_DOMAIN_P2,
                self._canonical_p2_payload(campaign),
            )
            return hmac.compare_digest(
                str(campaign.hmac_p2_signature),
                expected,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    def execute_prosecution_campaign_orientation(
        self,
        carrier1: ProsecutorPhase1TransferCarrier,
    ) -> Tuple[OODAProsecutionCampaignResultPhD, ProsecutionSymplecticFeedbackCarrier]:
        """Ejecuta la campaña Poincaré sin emitir todavía un veredicto final."""
        now = time.time()

        if not self.verify_phase1_transfer(carrier1):
            logger.critical("Transferencia P1 no auténtica o alterada.")
            raise ValueError("Falla de autenticidad en la transferencia Fase I→II.")

        contract = carrier1.gauge_contract
        orbital = carrier1.poincare_orbital_carrier
        phase2 = self._phase2_core()

        section = self._invoke_named(
            phase2,
            (
                "compute_henon_poincare_section",
                "poincare_section_henon",
                "extract_poincare_section",
            ),
            orbital,
        )
        monodromy = self._invoke_named(
            phase2,
            (
                "compute_analytic_monodromy",
                "analytic_monodromy",
                "_analytic_monodromy",
            ),
            orbital,
        )
        kam = self._invoke_named(
            phase2,
            ("audit_kam_resonances", "kam_audit", "_kam_audit"),
            orbital,
            monodromy,
        )

        feedback = self._synthesize_feedback_resilient(
            orbital,
            kam,
            monodromy,
            section,
        )

        kam_object = getattr(feedback, "kam_audit", kam)
        monodromy_object = getattr(feedback, "monodromy_result", monodromy)
        kam_ok = bool(
            getattr(kam_object, "is_kam_torus_stable", False)
        )
        lyap = _finite_float(
            getattr(
                kam_object,
                "max_lyapunov_exponent",
                getattr(monodromy_object, "max_lyapunov_exponent", 0.0),
            ),
            1.0e6,
        )
        recurrence = max(
            0.0,
            _finite_float(
                getattr(kam_object, "poincare_recurrence_time_estimate", 0.0)
            ),
        )
        nekhoroshev = max(
            0.0,
            _finite_float(
                getattr(
                    kam_object,
                    "nekhoroshev_time_bound",
                    getattr(kam_object, "nekhoroshev_bound", 0.0),
                )
            ),
        )
        rotation = _finite_float(
            getattr(kam_object, "rotation_number", 0.0)
        )
        drift = max(
            0.0,
            _finite_float(
                getattr(feedback, "symplectic_drift_penalty", 1.0e6),
                1.0e6,
            ),
        )
        energy_drift = max(
            0.0,
            _finite_float(getattr(feedback, "energy_drift", 1.0e6), 1.0e6),
        )
        pairing = max(
            0.0,
            _finite_float(getattr(feedback, "pairing_defect", 1.0e6), 1.0e6),
        )
        section_count = int(
            getattr(feedback, "section_count", 0) or 0
        )
        if section is not None and section_count == 0:
            section_count = int(
                _read_field(
                    section,
                    "crossing_count",
                    _read_field(section, "section_count", 0),
                )
                or 0
            )

        conserved = bool(
            getattr(
                feedback,
                "symplectic_conserved",
                drift < _SYMPLECTIC_DRIFT_TOL * 10.0,
            )
        )
        holonomy = bool(
            getattr(feedback, "holonomy_breakage_detected", False)
        )
        claim_tamper = bool(carrier1.claim_tamper_detected)
        gate_tear = bool(
            contract.gate_tear_detected
            or getattr(feedback, "gate_tear_detected", False)
        )
        dirac_ok = bool(carrier1.dirac_auth_ok)
        spohn = max(
            0.0,
            _finite_float(orbital.spohn_entropy_production),
        )

        lip = self._estimate_symplectic_lipschitz(
            drift,
            lyap,
            kam_ok,
        )
        orient_verdict = contract.isometry_verdict.meet(
            _as_heyting(
                getattr(
                    feedback,
                    "intermediate_verdict",
                    contract.isometry_verdict,
                )
            )
        )
        tentative = _as_status(
            getattr(
                feedback,
                "prosecution_status_tentative",
                OODAProsecutionStatus.PENDING,
            )
        )

        if claim_tamper:
            tentative = OODAProsecutionStatus.VETOED_INDICTMENT_TAMPER
            orient_verdict = HeytingOmega4.VETOED
        elif not dirac_ok:
            tentative = OODAProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE
            orient_verdict = HeytingOmega4.VETOED
        elif gate_tear:
            tentative = OODAProsecutionStatus.VETOED_HOMOMORPHIC_GATE_TEAR
            orient_verdict = HeytingOmega4.VETOED
        elif holonomy:
            tentative = OODAProsecutionStatus.VETOED_HOLONOMY_BREAKAGE
            orient_verdict = HeytingOmega4.VETOED
        elif not conserved:
            tentative = OODAProsecutionStatus.VETOED_SYMPLECTIC_DRIFT
            orient_verdict = HeytingOmega4.VETOED

        campaign_id = f"CAMP-OODA-v15-{uuid.uuid4().hex}"
        chain_hash = hashlib.sha384(
            _canonical_json(
                {
                    "contract_chain": contract.chain_hash,
                    "campaign_id": campaign_id,
                    "verdict": int(orient_verdict),
                    "status": int(tentative),
                    "p1_transfer_signature": carrier1.hmac_transfer_signature,
                }
            ).encode("utf-8")
        ).hexdigest()[:_CHAIN_HEX_LEN]

        campaign = OODAProsecutionCampaignResultPhD(
            campaign_id=campaign_id,
            bound_contract_id=contract.contract_id,
            lipschitz_contraction=float(lip),
            spohn_entropy_production=float(spohn),
            is_kam_torus_stable=bool(kam_ok),
            max_lyapunov_exponent=float(lyap),
            poincare_recurrence_time=float(recurrence),
            symplectic_drift_penalty=float(drift),
            holonomy_breakage_detected=bool(holonomy),
            nekhoroshev_bound=float(nekhoroshev),
            rotation_number=float(rotation),
            section_count=int(section_count),
            energy_drift=float(energy_drift),
            pairing_defect=float(pairing),
            symplectic_conserved=bool(conserved),
            qnd_orient_verdict=orient_verdict,
            prosecution_status_tentative=tentative,
            chain_hash=chain_hash,
            hmac_p2_signature="",
            timestamp_utc=float(now),
            phase1_transfer_signature=carrier1.hmac_transfer_signature,
            dirac_auth_ok=bool(dirac_ok),
            claim_tamper_detected=bool(claim_tamper),
            gate_tear_detected=bool(gate_tear),
        )
        campaign = replace(
            campaign,
            hmac_p2_signature=self._compute_hmac(
                _HMAC_DOMAIN_P2,
                self._canonical_p2_payload(campaign),
            ),
        )
        return campaign, feedback

    def _canonical_phase2_transfer_payload(
        self,
        carrier: ProsecutorPhase2TransferCarrier,
    ) -> str:
        feedback = carrier.symplectic_feedback
        orbital = feedback.orbital_carrier
        kam = feedback.kam_audit
        monodromy = feedback.monodromy_result
        germ = carrier.indictment_germ

        monodromy_matrix = getattr(
            monodromy,
            "monodromy_matrix",
            np.eye(2, dtype=np.float64),
        )
        dirac = orbital.dirac_result

        return _canonical_json(
            {
                "contract_id": carrier.gauge_contract.contract_id,
                "contract_signature": carrier.gauge_contract.hmac_p1_signature,
                "campaign_id": carrier.campaign_result.campaign_id,
                "campaign_signature": carrier.campaign_result.hmac_p2_signature,
                "phase1_transfer_signature": (
                    carrier.campaign_result.phase1_transfer_signature
                ),
                "indictment_id": str(getattr(germ, "indictment_id", "")),
                "transaction_payload_sha384": hashlib.sha384(
                    str(getattr(germ, "transaction_payload", "")).encode("utf-8")
                ).hexdigest(),
                "electron_anomaly_count": int(
                    getattr(germ, "electron_anomaly_count", 0)
                ),
                "positron_signature": str(
                    getattr(germ, "positron_auth_signature", "")
                ),
                "rho_sha384": _array_sha384(
                    carrier.sanitized_density_matrix,
                    complex_array=True,
                ),
                "connection_sha384": _array_sha384(
                    carrier.gauge_connection,
                    complex_array=True,
                ),
                "orbital_rho_sha384": _array_sha384(
                    orbital.sanitized_density_matrix,
                    complex_array=True,
                ),
                "q_sha384": _array_sha384(orbital.coordinates_q),
                "p_sha384": _array_sha384(orbital.momenta_p),
                "actions_sha384": _array_sha384(orbital.action_variables_I),
                "angles_sha384": _array_sha384(orbital.angle_variables_theta),
                "monodromy_sha384": _array_sha384(monodromy_matrix),
                "orbital_energy": _float_token(orbital.hamiltonian_energy),
                "spohn_proxy": _float_token(orbital.spohn_entropy_production),
                "dirac_result": dirac,
                "kam_stable": bool(
                    getattr(kam, "is_kam_torus_stable", False)
                ),
                "lyapunov": _float_token(
                    getattr(kam, "max_lyapunov_exponent", 0.0)
                ),
                "small_divisor": _float_token(
                    getattr(kam, "min_small_divisor", 0.0)
                ),
                "nekhoroshev_proxy": _float_token(
                    getattr(kam, "nekhoroshev_time_bound", 0.0)
                ),
                "feedback_drift": _float_token(
                    feedback.symplectic_drift_penalty
                ),
                "feedback_energy_drift": _float_token(feedback.energy_drift),
                "feedback_pairing_defect": _float_token(
                    feedback.pairing_defect
                ),
                "feedback_section_count": int(feedback.section_count),
                "feedback_conserved": bool(feedback.symplectic_conserved),
                "feedback_holonomy": bool(
                    feedback.holonomy_breakage_detected
                ),
                "feedback_verdict": int(feedback.intermediate_verdict),
                "feedback_status": int(
                    feedback.prosecution_status_tentative
                ),
                "transfer_dirac_auth_ok": bool(carrier.dirac_auth_ok),
                "transfer_claim_tamper": bool(
                    carrier.claim_tamper_detected
                ),
                "transfer_gate_tear": bool(
                    carrier.gauge_contract.gate_tear_detected
                ),
            }
        )

    def verify_phase2_transfer(
        self,
        carrier: ProsecutorPhase2TransferCarrier,
    ) -> bool:
        try:
            if not self.continue_to_phase3_prosecutor_gate(
                carrier.campaign_result
            ):
                return False
            expected = self._compute_hmac(
                _HMAC_DOMAIN_P2_TRANSFER,
                self._canonical_phase2_transfer_payload(carrier),
            )
            return hmac.compare_digest(
                str(carrier.hmac_transfer_signature),
                expected,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # MORFISMO FINAL DE FASE II ──► INICIO DE FASE III
    # ──────────────────────────────────────────────────────────────────────────
    def bridge_phase2_to_phase3_adjudication_transfer(
        self,
        carrier1: ProsecutorPhase1TransferCarrier,
    ) -> ProsecutorPhase2TransferCarrier:
        """
        Ejecuta P2 y firma el tensor de realimentación P2→P3.

        La firma cubre la campaña, la transferencia P1, el estado usado por la
        auditoría, la matriz de monodromía y los indicadores de adjudicación.
        """
        campaign, feedback = self.execute_prosecution_campaign_orientation(
            carrier1
        )
        carrier2 = ProsecutorPhase2TransferCarrier(
            gauge_contract=carrier1.gauge_contract,
            campaign_result=campaign,
            symplectic_feedback=feedback,
            sanitized_density_matrix=carrier1.sanitized_density_matrix,
            gauge_connection=carrier1.gauge_connection,
            indictment_germ=carrier1.indictment_germ,
            dirac_auth_ok=carrier1.dirac_auth_ok,
            claim_tamper_detected=carrier1.claim_tamper_detected,
            hmac_transfer_signature="",
        )
        carrier2 = replace(
            carrier2,
            hmac_transfer_signature=self._compute_hmac(
                _HMAC_DOMAIN_P2_TRANSFER,
                self._canonical_phase2_transfer_payload(carrier2),
            ),
        )
        logger.debug(
            "P2→P3: campaña=%s KAM=%s λ=%.6g ΔE=%.3e secciones=%d",
            campaign.campaign_id,
            campaign.is_kam_torus_stable,
            campaign.max_lyapunov_exponent,
            campaign.energy_drift,
            campaign.section_count,
        )
        return carrier2


# ══════════════════════════════════════════════════════════════════════════════
# FASE III: DECIDE & ACT — Ω₄, RSI-L3, CERTIFICADO Y PASAPORTE
# ══════════════════════════════════════════════════════════════════════════════

class OODAProsecutionDeciderPhD(OODAProsecutionCampaignOrientorPhD):
    """
    Fase III — Decide & Act.

    Comprueba P2, fusiona evidencia con el meet de Heyting, evalúa el defecto
    funtorial, itera RSI-L3 y emite un pasaporte firmado. El interlock es siempre
    simulado y no existe ruta de escritura MMIO en esta clase.
    """

    def _cstar_corner_integrity(
        self,
        rho: NDArray[np.complex128],
        i0: int,
        i1: int,
    ) -> float:
        matrix = np.asarray(rho, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return 0.0
        n = int(matrix.shape[0])
        if n == 0:
            return 0.0

        i0 = max(0, min(int(i0), n - 1))
        i1 = max(i0 + 1, min(int(i1), n))
        projection = np.zeros((n, n), dtype=np.complex128)
        projection[i0:i1, i0:i1] = np.eye(i1 - i0, dtype=np.complex128)

        pi_rho = projection @ matrix @ projection
        pi_rho2 = projection @ (matrix @ matrix) @ projection
        numerator = float(la.norm(pi_rho2 - pi_rho @ pi_rho, ord="fro"))
        denominator = max(
            float(la.norm(matrix, ord="fro")),
            _WILKINSON_EPSILON,
        )
        return float(math.exp(-numerator / denominator))

    def _compose_phi13_functoriality(
        self,
        rho: NDArray[np.complex128],
    ) -> Tuple[float, float, float, float]:
        """Retorna (i12, i23, i13, |i13 − i12*i23|)."""
        matrix = np.asarray(rho, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return 0.0, 0.0, 0.0, 1.0
        n = int(matrix.shape[0])
        if n == 0:
            return 0.0, 0.0, 0.0, 1.0

        b = max(1, n // 3)
        i12 = self._cstar_corner_integrity(matrix, 0, min(n, 2 * b))
        i23 = self._cstar_corner_integrity(
            matrix,
            min(n - 1, b),
            n,
        )
        i13 = self._cstar_corner_integrity(matrix, 0, n)
        defect = abs(i13 - i12 * i23)
        return float(i12), float(i23), float(i13), float(defect)

    def _apply_rsi_l3_meta_operator(
        self,
        lip_emp: float,
        residual: float,
    ) -> Tuple[int, float, float, float]:
        """Itera el operador RSI local una vez por adjudicación."""
        core = self._fallback_phase3_core
        config, lip_t, cauchy = core.trigger_recursive_step(
            float(np.clip(lip_emp, 0.0, 0.999999)),
            float(np.clip(max(0.0, residual), 0.0, 1.0)),
        )
        return (
            int(config.generation),
            float(config.banach_lipschitz),
            float(config.meta_operator_lip),
            float(cauchy),
        )

    def _project_to_visceral_business_phd(
        self,
        verdict: HeytingOmega4,
        status: OODAProsecutionStatus,
        spohn_entropy: float,
        lyapunov_exp: float,
        is_kam_stable: bool,
        crowbar_simulated: bool,
    ) -> Dict[str, Any]:
        """
        Proyección semántica informativa.

        No inventa WACC, liquidez, ahorros, compresión ni capital en riesgo:
        esos datos requieren un modelo externo que este módulo no implementa.
        """
        if crowbar_simulated:
            hardware_action = (
                "INTERLOCK ESP32 SIMULADO; sin MMIO ni activación de BT151. "
                f"Registro 0x{_ESP32_GPIO_OUT_W1TS_REG:08X} solo documental."
            )
        else:
            hardware_action = (
                "INTERLOCK INHIBIDO; no se realizó acción física ni acceso MMIO."
            )

        return {
            "model_scope": "SIMULACIÓN; no es un diagnóstico físico.",
            "financial_model_status": "NO_CALCULADO",
            "wacc_protection": "NO CALCULADO: no se suministró un modelo financiero.",
            "net_liquidity_impact": None,
            "kv_cache_compression_pct": None,
            "llm_cost_savings_pct": None,
            "hardware_action": hardware_action,
            "executive_summary": (
                f"Veredicto {verdict.name}; estado {status.name}; "
                f"exponente finito de Lyapunov={lyapunov_exp:+.6g}; "
                f"KAM finito estable={bool(is_kam_stable)}. "
                "Las magnitudes de este módulo son resultados de simulación."
            ),
            "visceral_financial_vector": {
                "spohn_entropy_proxy": round(
                    max(0.0, _finite_float(spohn_entropy)),
                    12,
                ),
                "kam_torus_rupture_in_finite_audit": not bool(is_kam_stable),
                "capital_at_risk_usd": None,
            },
        }

    @staticmethod
    def execute_cartesian_monad_kleisli(
        contract: OODAProsecutorGaugeContractPhD,
        campaign: OODAProsecutionCampaignResultPhD,
        exec_verdict: HeytingOmega4,
        exec_status: OODAProsecutionStatus,
    ) -> Tuple[HeytingOmega4, OODAProsecutionStatus]:
        """
        Composición monádica de veredictos para esta cadena finita.

        En Ω₄, el meet selecciona el menor grado de verdad. No se afirma que
        esta operación modele por sí sola una mónada categórica general.
        """
        fused = contract.isometry_verdict.meet(
            campaign.qnd_orient_verdict
        ).meet(exec_verdict)

        status = exec_status
        if (
            fused == HeytingOmega4.VETOED
            and status
            in (
                OODAProsecutionStatus.ABSOLVED,
                OODAProsecutionStatus.PURGED_DIRAC_VACUUM,
                OODAProsecutionStatus.PENDING,
            )
        ):
            status = OODAProsecutionStatus.VETOED_HOLONOMY_BREAKAGE
        return fused, status

    def _adjudicate_resilient(
        self,
        feedback: ProsecutionSymplecticFeedbackCarrier,
    ) -> OODAProsecutionExecutionCertificatePhD:
        # Se usa el núcleo local controlado. No se delega el interlock a un
        # núcleo externo que pudiera tener efectos laterales de hardware.
        core = self._fallback_phase3_core
        return core.adjudicate_and_execute(feedback)

    def _passport_commitment_data(
        self,
        passport: OODAProsecutorSovereignPassportPhD,
    ) -> Dict[str, Any]:
        """Todos los campos sustantivos salvo raíz y firma, que son circulares."""
        return {
            "passport_id": passport.passport_id,
            "sovereign_id": passport.sovereign_id,
            "campaign_id": passport.campaign_id,
            "bound_contract_id": passport.bound_contract_id,
            "global_verdict": int(passport.global_verdict),
            "prosecution_status": int(passport.prosecution_status),
            "crowbar_triggered": bool(passport.crowbar_triggered),
            "crowbar_triggered_simulated": bool(
                passport.crowbar_triggered_simulated
            ),
            "crowbar_latency_ns": passport.crowbar_latency_ns,
            "electron_anomalies_purged": int(
                passport.electron_anomalies_purged
            ),
            "released_energy_kev": passport.released_energy_kev,
            "back_action_db": passport.back_action_db,
            "rsi_level3_generation": int(passport.rsi_level3_generation),
            "rsi_banach_contraction_lip": passport.rsi_banach_contraction_lip,
            "rsi_meta_operator_lip": passport.rsi_meta_operator_lip,
            "rsi_cauchy_residual": passport.rsi_cauchy_residual,
            "homomorphic_gate_phi12_integrity": (
                passport.homomorphic_gate_phi12_integrity
            ),
            "homomorphic_gate_phi23_integrity": (
                passport.homomorphic_gate_phi23_integrity
            ),
            "homomorphic_gate_phi13_integrity": (
                passport.homomorphic_gate_phi13_integrity
            ),
            "functoriality_defect": passport.functoriality_defect,
            "energy_hamiltonian": passport.energy_hamiltonian,
            "energy_drift": passport.energy_drift,
            "nekhoroshev_bound": passport.nekhoroshev_bound,
            "williamson_min": passport.williamson_min,
            "claim_tamper_detected": bool(
                passport.claim_tamper_detected
            ),
            "gate_tear_detected": bool(passport.gate_tear_detected),
            "esp32_gpio_w1ts_reg_documentary": int(
                passport.esp32_gpio_w1ts_reg_documentary
            ),
            "visceral_report": passport.visceral_report,
            "timestamp_utc": passport.timestamp_utc,
            "phase1_transfer_signature": passport.phase1_transfer_signature,
            "phase2_campaign_signature": passport.phase2_campaign_signature,
            "phase2_transfer_signature": passport.phase2_transfer_signature,
        }

    def _merkle_seed(
        self,
        passport: OODAProsecutorSovereignPassportPhD,
    ) -> bytes:
        """
        Mantiene el nombre histórico del campo, pero calcula un compromiso
        canónico SHA-384 de un objeto; no construye por sí solo un DAG Merkle.
        """
        return _canonical_json(
            self._passport_commitment_data(passport)
        ).encode("utf-8")

    def _provenance_payload(
        self,
        passport: OODAProsecutorSovereignPassportPhD,
        root: str,
    ) -> str:
        return _canonical_json(
            {
                "version": _VERSION_STRING,
                "sha384_commitment": root,
                "timestamp_utc": _float_token(passport.timestamp_utc),
            }
        )

    def verify_merkle_provenance(
        self,
        passport: OODAProsecutorSovereignPassportPhD,
    ) -> bool:
        """Verifica el compromiso SHA-384 y su HMAC de procedencia."""
        try:
            root = hashlib.sha384(self._merkle_seed(passport)).hexdigest()
            if not hmac.compare_digest(
                str(passport.merkle_dag_sha384_root),
                root,
            ):
                return False
            expected = self._compute_hmac(
                _HMAC_DOMAIN_P3,
                self._provenance_payload(passport, root),
            )
            return hmac.compare_digest(
                str(passport.hmac_provenance_signature),
                expected,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False

    def execute_phase3_sovereign_decision(
        self,
        carrier2: ProsecutorPhase2TransferCarrier,
    ) -> OODAProsecutorSovereignPassportPhD:
        """
        Cierre P3: verifica ambas compuertas, adjudica, calcula RSI-L3 y firma
        el pasaporte. VETOED implica solo un interlock simulado.
        """
        now = time.time()
        contract = carrier2.gauge_contract
        campaign = carrier2.campaign_result
        feedback = carrier2.symplectic_feedback

        if not self.continue_to_phase2_prosecutor_gate(contract):
            raise ValueError("Contrato P1 inválido al iniciar Fase III.")
        if not self.verify_phase2_transfer(carrier2):
            raise ValueError("Transferencia P2→P3 inválida o alterada.")

        if (
            campaign.bound_contract_id != contract.contract_id
            or campaign.phase1_transfer_signature == ""
            or campaign.phase1_transfer_signature
            != getattr(campaign, "phase1_transfer_signature", "")
        ):
            raise ValueError("La campaña no está enlazada con el contrato P1.")

        if (
            bool(carrier2.dirac_auth_ok) != bool(campaign.dirac_auth_ok)
            or bool(carrier2.claim_tamper_detected)
            != bool(campaign.claim_tamper_detected)
        ):
            raise ValueError("Inconsistencia entre transferencia y campaña firmada.")

        execution = self._adjudicate_resilient(feedback)
        exec_verdict = _as_heyting(
            getattr(
                execution,
                "global_heyting_verdict",
                campaign.qnd_orient_verdict,
            )
        )
        exec_status = _as_status(
            getattr(
                execution,
                "status",
                campaign.prosecution_status_tentative,
            )
        )

        claim_tamper = bool(
            carrier2.claim_tamper_detected
            or campaign.claim_tamper_detected
            or feedback.claim_tamper_detected
        )
        auth_ok = bool(
            carrier2.dirac_auth_ok
            and campaign.dirac_auth_ok
            and _dirac_auth_flag(feedback.orbital_carrier.dirac_result)
        )
        gate_hard_evidence = bool(
            contract.gate_tear_detected
            or campaign.gate_tear_detected
            or feedback.gate_tear_detected
        )

        if claim_tamper:
            exec_verdict = HeytingOmega4.VETOED
            exec_status = OODAProsecutionStatus.VETOED_INDICTMENT_TAMPER
        elif not auth_ok:
            exec_verdict = HeytingOmega4.VETOED
            exec_status = OODAProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE
        elif gate_hard_evidence:
            exec_verdict = HeytingOmega4.VETOED
            exec_status = OODAProsecutionStatus.VETOED_HOMOMORPHIC_GATE_TEAR

        final_verdict, final_status = self.execute_cartesian_monad_kleisli(
            contract,
            campaign,
            exec_verdict,
            exec_status,
        )

        i12_raw, i23_raw, i13_raw, defect = (
            self._compose_phi13_functoriality(
                carrier2.sanitized_density_matrix
            )
        )

        # La autenticidad criptográfica ya fue verificada; estos indicadores
        # combinan la autenticidad de la compuerta y el control matricial.
        p1_gate_ok = self.continue_to_phase2_prosecutor_gate(contract)
        p2_gate_ok = self.continue_to_phase3_prosecutor_gate(campaign)
        i12 = min(i12_raw, 1.0 if p1_gate_ok else 0.0)
        i23 = min(i23_raw, 1.0 if p2_gate_ok else 0.0)
        i13 = float(i13_raw)

        gate_tear = bool(
            gate_hard_evidence
            or defect > _FUNCTOR_DEFECT_TOL
            or min(i12, i23, i13) < _GATE_TEAR_INTEGRITY_FLOOR
        )
        if gate_tear:
            # La ruptura medida es evidencia dura dentro de este modelo.
            final_verdict = HeytingOmega4.VETOED
            if not claim_tamper and auth_ok:
                final_status = (
                    OODAProsecutionStatus.VETOED_HOMOMORPHIC_GATE_TEAR
                )

        if claim_tamper:
            final_verdict = HeytingOmega4.VETOED
            final_status = OODAProsecutionStatus.VETOED_INDICTMENT_TAMPER
        elif not auth_ok:
            final_verdict = HeytingOmega4.VETOED
            final_status = OODAProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE

        lip_emp = float(
            np.clip(
                _finite_float(
                    campaign.lipschitz_contraction,
                    _DEFAULT_BANACH_CONTRACTION_LIP,
                ),
                0.0,
                0.999999,
            )
        )
        residual_input = max(
            0.0,
            _finite_float(campaign.symplectic_drift_penalty),
            _finite_float(campaign.energy_drift),
            _finite_float(campaign.pairing_defect),
        )
        generation, lip_t, meta_lip, cauchy = (
            self._apply_rsi_l3_meta_operator(
                lip_emp,
                min(residual_input, 1.0),
            )
        )

        crowbar_simulated = final_verdict == HeytingOmega4.VETOED
        if crowbar_simulated:
            simulation_core = self._fallback_phase3_core
            _, latency = simulation_core.simulate_esp32_safety_interlock(
                True,
                (
                    f"{contract.chain_hash}:"
                    f"{campaign.chain_hash}:"
                    f"{final_status.name}"
                ).encode("utf-8"),
            )
            latency = float(
                np.clip(
                    latency,
                    _ESP32_CROWBAR_MIN_LATENCY_NS,
                    _ESP32_CROWBAR_MAX_LATENCY_NS,
                )
            )
        else:
            latency = 0.0

        germ = carrier2.indictment_germ
        try:
            anomaly_count = _validated_anomaly_count(
                getattr(germ, "electron_anomaly_count", 0)
            )
        except (TypeError, ValueError):
            anomaly_count = 0
            claim_tamper = True
            final_verdict = HeytingOmega4.VETOED
            final_status = OODAProsecutionStatus.VETOED_INDICTMENT_TAMPER
            crowbar_simulated = True
            _, latency = self._fallback_phase3_core.simulate_esp32_safety_interlock(
                True,
                f"{contract.chain_hash}:INVALID-COUNT".encode("utf-8"),
            )

        purged = anomaly_count if auth_ok and not claim_tamper else 0
        released_energy = float(purged * _POSITRON_REST_MASS_KEV)
        back_action = (
            0.0 if auth_ok else _DIRAC_AUTH_FAILURE_BACK_ACTION_DB
        )

        visceral = self._project_to_visceral_business_phd(
            verdict=final_verdict,
            status=final_status,
            spohn_entropy=max(
                0.0,
                _finite_float(campaign.spohn_entropy_production),
            ),
            lyapunov_exp=_finite_float(
                campaign.max_lyapunov_exponent,
                1.0e6,
            ),
            is_kam_stable=bool(campaign.is_kam_torus_stable),
            crowbar_simulated=bool(crowbar_simulated),
        )

        orbital = feedback.orbital_carrier
        pseudo_williamson = tuple(
            getattr(orbital, "williamson_invariants", ()) or ()
        )
        williamson_min = (
            float(min(pseudo_williamson))
            if pseudo_williamson
            else 0.0
        )
        energy = _finite_float(orbital.hamiltonian_energy)

        passport = OODAProsecutorSovereignPassportPhD(
            passport_id=f"PASSPORT-OODA-v15-{uuid.uuid4().hex}",
            sovereign_id=self.sovereign_id,
            campaign_id=campaign.campaign_id,
            bound_contract_id=contract.contract_id,
            global_verdict=final_verdict,
            prosecution_status=final_status,
            crowbar_triggered=bool(crowbar_simulated),
            crowbar_triggered_simulated=bool(crowbar_simulated),
            crowbar_latency_ns=float(latency),
            electron_anomalies_purged=int(purged),
            released_energy_kev=float(released_energy),
            back_action_db=float(back_action),
            rsi_level3_generation=int(generation),
            rsi_banach_contraction_lip=float(lip_t),
            rsi_meta_operator_lip=float(meta_lip),
            rsi_cauchy_residual=float(cauchy),
            homomorphic_gate_phi12_integrity=float(i12),
            homomorphic_gate_phi23_integrity=float(i23),
            homomorphic_gate_phi13_integrity=float(i13),
            functoriality_defect=float(defect),
            energy_hamiltonian=float(energy),
            energy_drift=float(
                max(0.0, _finite_float(campaign.energy_drift))
            ),
            nekhoroshev_bound=float(
                max(0.0, _finite_float(campaign.nekhoroshev_bound))
            ),
            williamson_min=float(williamson_min),
            claim_tamper_detected=bool(claim_tamper),
            gate_tear_detected=bool(gate_tear),
            esp32_gpio_w1ts_reg_documentary=int(
                _ESP32_GPIO_OUT_W1TS_REG
            ),
            merkle_dag_sha384_root="",
            hmac_provenance_signature="",
            visceral_report=visceral,
            timestamp_utc=float(now),
            phase1_transfer_signature=campaign.phase1_transfer_signature,
            phase2_campaign_signature=campaign.hmac_p2_signature,
            phase2_transfer_signature=carrier2.hmac_transfer_signature,
        )

        root = hashlib.sha384(self._merkle_seed(passport)).hexdigest()
        passport = replace(passport, merkle_dag_sha384_root=root)
        signature = self._compute_hmac(
            _HMAC_DOMAIN_P3,
            self._provenance_payload(passport, root),
        )
        return replace(
            passport,
            hmac_provenance_signature=signature,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACHADA Y ORQUESTACIÓN DE LAS TRES FASES
# ══════════════════════════════════════════════════════════════════════════════

class OODATriphasicHomomorphicPipelineProsecutorAgentPhD(
    OODAProsecutionDeciderPhD
):
    """Fachada soberana que conserva la continuidad firmada P1→P2→P3."""

    def prosecute_ooda_pipeline_sovereign(
        self,
        density_matrix: NDArray[Any],
        electron_anomaly_count: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
    ) -> OODAProsecutorSovereignPassportPhD:
        carrier1 = self.bridge_phase1_to_phase2_campaign_transfer(
            density_matrix=density_matrix,
            electron_anomaly_count=electron_anomaly_count,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
        )
        carrier2 = self.bridge_phase2_to_phase3_adjudication_transfer(
            carrier1=carrier1
        )
        return self.execute_phase3_sovereign_decision(carrier2)


# ══════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN DE EJECUCIÓN LOCAL (SOLO SIMULACIÓN)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 88)
    print("OODA trifásico: Yoshida, auditoría KAM finita, RSI-L3 y ESP32 simulado")
    print(_VERSION_STRING)
    print("Ningún acceso MMIO; los proxies no son certificaciones físicas.")
    print("═" * 88)

    dimension = 56
    agent = OODATriphasicHomomorphicPipelineProsecutorAgentPhD(
        sovereign_id="SOVEREIGN-OODA-DEMO",
        dimension=dimension,
        # En producción, inyectar una clave aleatoria obtenida de un gestor seguro.
        secret_key=_DEFAULT_SECRET_KEY,
    )

    rng = np.random.default_rng(2026)
    psi = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    rho = psi @ psi.conj().T
    rho /= np.trace(rho)

    payload = "TX-OODA-DEMO-2026-001"
    anomaly_count = 2
    signature = compute_dirac_positron_hmac(
        agent.secret_key,
        payload,
        anomaly_count,
    )

    passport = agent.prosecute_ooda_pipeline_sovereign(
        density_matrix=rho,
        electron_anomaly_count=anomaly_count,
        positron_auth_signature=signature,
        transaction_payload=payload,
    )

    print(f"Pasaporte                 : {passport.passport_id}")
    print(f"Veredicto                 : {passport.global_verdict.name}")
    print(f"Estado                    : {passport.prosecution_status.name}")
    print(f"Anomalías de simulación   : {passport.electron_anomalies_purged}")
    print(f"Energía equivalente       : {passport.released_energy_kev:.1f} keV")
    print(f"Interlock simulado        : {passport.crowbar_triggered_simulated}")
    print(f"Latencia simulada         : {passport.crowbar_latency_ns:.3f} ns")
    print(
        "Integridad Φ12/Φ23/Φ13 : "
        f"{passport.homomorphic_gate_phi12_integrity:.6f} / "
        f"{passport.homomorphic_gate_phi23_integrity:.6f} / "
        f"{passport.homomorphic_gate_phi13_integrity:.6f}"
    )
    print(f"RSI-L3 generación         : {passport.rsi_level3_generation}")
    print(
        "RSI Lip(T)/Lip(S)         : "
        f"{passport.rsi_banach_contraction_lip:.6f} / "
        f"{passport.rsi_meta_operator_lip:.6f}"
    )
    print(
        "Procedencia verificada   : "
        f"{agent.verify_merkle_provenance(passport)}"
    )
    print("═" * 88)
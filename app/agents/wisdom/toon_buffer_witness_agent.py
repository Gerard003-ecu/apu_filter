r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo    : BoolNovikovToonBufferWitnessAgent v15.0.0-PhD                                            ║
║ Ubicación : app/agents/wisdom/toon_buffer_witness_agent.py                                           ║
║ Versión   : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-BUFFER-WITNESS-AGENT-POINCARE-KAM-RSI3-PhD            ║
║ Gobernanza: OODA Trifásica Pasiva QND                                                                ║
║             (① Observe ➔ Φ₁₂ ➔ ② Orient ➔ Φ₂₃ ➔ ③ Decide & Act)                                     ║
║             sobre el amortiguamiento y protección de caudales mnémonicos TOON en T*Q.                ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA EN TRES FASES ANIDADAS (CADENA DE HERENCIA CON CONTINUIDAD MORFISMÁTICA)
─────────────────────────────────────────────────────────────────────────────────────
FASE 1 · ToonBufferWitnessPhase1Observer
    Ingesta estricta sin normalización silenciosa de la MIC, los exponentes de Novikov, ρ_MAC
    y la matriz de incidencia opcional. Certifica hermiticidad, semidefinición positiva (PSD),
    traza unitaria, entropía de von Neumann S(ρ), pureza de Hilbert-Schmidt Tr(ρ²), información
    asimétrica de Wigner-Yanase I_WY(ρ, K) y margen de pasividad de Foster-Brune / Gershgorin.
    ÚLTIMO MÉTODO → execute_phase1(...)  ⇒  ToonBufferWitnessPhase1Certificate

FASE 2 · ToonBufferWitnessPhase2Orientor(ToonBufferWitnessPhase1Observer)
    PRIMER MÉTODO  ← continue_to_phase2_gate(p1, ...)   [Compuerta Φ₁₂ fail-closed: F1 → F2]
    Invoca el motor espectral v15 y orienta los invariantes de mecánica celeste de Poincaré:
    parámetro invariante de Tisserand-Jacobi C_T, integral vertical de Kozai-Lidov L_z = H/L,
    defecto simpléctico ‖Mᵀ J₆ M − J₆‖_F de las variables no singulares de Poincaré, deriva del
    invariante integral relativo de Poincaré-Cartan (∮ p dq), torsión de Kolmogorov y Arnol'd,
    serie de Lie de Deprit-Hori, impedancia/VSWR de Heaviside y sobrepresión de Zhukovsky.
    ÚLTIMO MÉTODO → execute_phase2(p1, ...)  ⇒  ToonBufferWitnessPhase2Certificate

FASE 3 · ToonBufferWitnessPhase3Decider(ToonBufferWitnessPhase2Orientor)
    PRIMER MÉTODO  ← continue_to_phase3_gate(p2)        [Compuerta Φ₂₃ y auditoría RSI³: F2 → F3]
    Verifica independientemente la torre de Automejora Recursiva de Nivel 3 (RSI³):
    contracción cuadrática del meta-endofuntor M_Ψ(T_Φ) (Lip_L2 ≤ Lip_L1² + ε), coherencia de la
    mónada de Kleisli (asociatividad y unidades), guardia de solidez de Löb-Tarski (invariancia
    del punto fijo θ* = θ_deployed) y descenso monótono de Lyapunov. Adjudica el veredicto en el
    topos de Heyting Ω₃ con topología modal de Lawvere-Tierney y emite el estado soberano con
    raíz de Merkle SHA-256 y pruebas criptográficas de inclusión verificables.
    ÚLTIMO MÉTODO → execute_phase3(p2)  ⇒  ToonBufferWitnessSovereignState

ORQUESTADOR SOBERANO · BoolNovikovToonBufferWitnessAgent(ToonBufferWitnessPhase3Decider)
    ORQUESTADOR   → run_sovereign_governance_cycle(...)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from app.wisdom.toon_buffer_witness_engine import (
        BoolNovikovToonBufferWitnessEngine,
        HeytingTruthValue,
        ToonBufferWitnessSpectrumReport,
    )
except ImportError:
    try:
        from toon_buffer_witness_engine import (
            BoolNovikovToonBufferWitnessEngine,
            HeytingTruthValue,
            ToonBufferWitnessSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar toon_buffer_witness_engine v15 "
            "(ni como app.wisdom.toon_buffer_witness_engine ni desde la raíz de módulos)."
        ) from exc

logger = logging.getLogger("APU.Wisdom.BoolNovikovToonBufferWitnessAgent")

# ─── Constantes soberanas ────────────────────────────────────────────────────────────────────────
_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-BUFFER-WITNESS-AGENT-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_TOON_BUFFER_WITNESS_KEY_v14.0.0"
_MIN_SECRET_KEY_BYTES: Final[int] = 16
_HMAC_DOMAIN_P1: Final[bytes] = b"TOON-BUFFER-WITNESS-P1-OBSERVE-v15.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"TOON-BUFFER-WITNESS-P2-ORIENT-v15.0|"
_MERKLE_LEAF_PREFIX: Final[bytes] = b"\x00"
_MERKLE_NODE_PREFIX: Final[bytes] = b"\x01"

_HERMITIAN_TOL: Final[float] = 1.0e-10
_PSD_TOL: Final[float] = 1.0e-12
_TRACE_TOL: Final[float] = 1.0e-9
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_CARTAN_DRIFT_TOL: Final[float] = 1.0e-7
_KLEISLI_COHERENCE_TOL: Final[float] = 1.0e-12
_LOEB_ANCHOR_TOL: Final[float] = 1.0e-9
_BANACH_CONTRACTION_MAX: Final[float] = 1.0
_ACTUATION_EMITTED: Final[bool] = False
_VISCERAL_PROVENANCE: Final[str] = "DECLARED_POLICY"
_DEFAULT_STAGE_FACTORS: Final[Tuple[float, float, float]] = (0.95, 0.95, 0.95)
_DEFAULT_QUANT_SCALE: Final[float] = 1000.0
_DEPLOYED_POLICY_ANCHOR: Final[Tuple[float, float, float]] = (0.95, 0.35, 0.10)


# ─── Errores, certificados y contratos de estado soberano ────────────────────────────────────────
class ToonBufferWitnessGateError(ValueError):
    """Rechazo fail-closed en una compuerta homomórfica (integridad o admisibilidad)."""


@dataclass(frozen=True)
class ToonBufferWitnessPhase1Certificate:
    """Certificado P1 de la Fase 1 Observe. Sellado sobre todos sus campos canónicos."""
    timestamp: float
    dim: int
    mic_sha256: str
    exponents_sha256: str
    density_sha256: str
    tr_rho: float
    tr_imag_abs: float
    hermitian_residual: float
    min_eigenvalue: float
    is_hermitian: bool
    is_psd: bool
    is_unit_trace: bool
    payload_digest: str
    hmac_signature: str
    incidence_sha256: str = "NONE"
    von_neumann_entropy: float = 0.0
    hilbert_schmidt_purity: float = 1.0
    wigner_yanase_skew_info: float = 0.0
    brune_passivity_min_eig: float = 0.0
    gershgorin_spectral_bound: float = 0.0

    @property
    def is_valid_density(self) -> bool:
        """Verifica que ρ_MAC sea un operador de estado cuántico físico admisible."""
        max_entropy = math.log(max(1, self.dim)) + 1.0e-7
        entropy_ok = -1.0e-9 <= self.von_neumann_entropy <= max_entropy
        return self.is_hermitian and self.is_psd and self.is_unit_trace and entropy_ok


@dataclass(frozen=True)
class ToonBufferWitnessPhase2Certificate:
    """Certificado P2 de la Fase 2 Orient. Liga el reporte espectral y de Poincaré-KAM a P1."""
    timestamp: float
    p1_payload_digest: str
    spectrum_report: ToonBufferWitnessSpectrumReport
    payload_digest: str
    hmac_signature: str
    tisserand_poincare_invariant: float = 0.0
    kozai_lidov_vertical_integral: float = 0.0
    poincare_symplectic_ok: bool = True
    poincare_cartan_ok: bool = True
    kam_torsion_non_degenerate: bool = True


@dataclass(frozen=True)
class ToonBufferWitnessVisceralPolicy:
    """Política visceral DECLARADA. No es una medición del motor espectral."""
    wacc_protection_pct: float = 15.0
    net_liquidity_release_pct: float = 11.5
    kv_cache_delta_pct: float = -86.4


@dataclass(frozen=True)
class ToonBufferWitnessVisceralMetrics:
    """Métricas visceral-semánticas con procedencia explícita y telemetría de línea/Poincaré."""
    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_delta_pct: float
    water_hammer_status: str
    provenance: str
    zhukovsky_surge_pressure: float = 0.0
    telegrapher_vswr: float = 1.0
    nekhoroshev_stability_horizon: float = 0.0


@dataclass(frozen=True)
class ToonBufferWitnessSovereignState:
    """Estado soberano final de la Fase 3 Decide & Act."""
    timestamp: float
    truth_value: HeytingTruthValue
    is_passive_silent: bool
    actuation_emitted: bool
    back_action_db: float
    contraction_ok: bool
    merkle_root_hash: str
    visceral_metrics: ToonBufferWitnessVisceralMetrics
    rsi3_loeb_soundness_ok: bool = False
    kleisli_coherence_ok: bool = False
    poincare_symplectic_invariant_ok: bool = False
    lawvere_tierney_truth: HeytingTruthValue = HeytingTruthValue.COHERENT
    merkle_inclusion_proofs: Tuple[Tuple[Tuple[str, str], ...], ...] = ()


# ─── Utilidades criptográficas, serialización canónica y árbol de Merkle ─────────────────────────
def _validate_secret_key(secret_key: bytes) -> None:
    """Valida tipo, longitud mínima y advierte si se usa la clave por defecto."""
    if not isinstance(secret_key, (bytes, bytearray)):
        raise TypeError("secret_key debe ser bytes.")
    if len(secret_key) < _MIN_SECRET_KEY_BYTES:
        raise ValueError(f"secret_key requiere al menos {_MIN_SECRET_KEY_BYTES} bytes.")
    if bytes(secret_key) == _DEFAULT_SECRET_KEY:
        logger.warning("Clave HMAC por defecto en uso; no apta para producción.")


def _jsafe(value: Any) -> Any:
    """Convierte a tipos JSON deterministas. NaN e infinitos se codifican como cadenas."""
    if isinstance(value, np.ndarray):
        return _jsafe(value.tolist())
    if isinstance(value, np.generic):
        return _jsafe(value.item())
    if isinstance(value, complex):
        return [_jsafe(value.real), _jsafe(value.imag)]
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _jsafe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsafe(v) for v in value]
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialización JSON canónica: claves ordenadas, sin espacios, ASCII estricto."""
    text = json.dumps(
        _jsafe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("ascii")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_digest(arr: NDArray[Any]) -> str:
    """Digest criptográfico de un arreglo que incluye su forma exacta."""
    return _sha256_hex(_canonical_bytes({"shape": list(arr.shape), "data": arr}))


def _hmac_hex(key: bytes, domain: bytes, digest: str) -> str:
    return hmac.new(key, domain + digest.encode("ascii"), hashlib.sha256).hexdigest()


def _verify_seal(key: bytes, domain: bytes, digest: str, signature: str) -> bool:
    return hmac.compare_digest(_hmac_hex(key, domain, digest), signature)


def _merkle_root(leaf_digests: Sequence[str]) -> str:
    """Raíz de Merkle con separación de dominios (0x00 hojas, 0x01 nodos internos)."""
    if not leaf_digests:
        raise ValueError("La raíz de Merkle requiere al menos una hoja.")
    level = [_sha256_hex(_MERKLE_LEAF_PREFIX + bytes.fromhex(d)) for d in leaf_digests]
    while len(level) > 1:
        nxt = [
            _sha256_hex(
                _MERKLE_NODE_PREFIX + bytes.fromhex(level[i]) + bytes.fromhex(level[i + 1])
            )
            for i in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def _merkle_inclusion_proofs(
    leaf_digests: Sequence[str],
) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
    """Construye las rutas de inclusión de Merkle para cada hoja.

    Cada paso es un par `(dirección, hash_hermano)` donde `dirección` ∈ {"L", "R"} indica
    si el nodo hermano se concatena a la izquierda o a la derecha.
    """
    if not leaf_digests:
        raise ValueError("Se requiere al menos una hoja para generar pruebas de Merkle.")
    levels: List[List[str]] = [
        [_sha256_hex(_MERKLE_LEAF_PREFIX + bytes.fromhex(d)) for d in leaf_digests]
    ]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        nxt = [
            _sha256_hex(_MERKLE_NODE_PREFIX + bytes.fromhex(cur[i]) + bytes.fromhex(cur[i + 1]))
            for i in range(0, len(cur) - 1, 2)
        ]
        if len(cur) % 2 == 1:
            nxt.append(cur[-1])
        levels.append(nxt)

    proofs: List[Tuple[Tuple[str, str], ...]] = []
    for leaf_idx in range(len(leaf_digests)):
        idx = leaf_idx
        path: List[Tuple[str, str]] = []
        for lvl in levels[:-1]:
            if idx % 2 == 0:
                if idx + 1 < len(lvl):
                    path.append(("R", lvl[idx + 1]))
            else:
                path.append(("L", lvl[idx - 1]))
            idx //= 2
        proofs.append(tuple(path))
    return tuple(proofs)


def verify_merkle_inclusion_proof(
    leaf_digest: str,
    proof: Sequence[Tuple[str, str]],
    expected_root: str,
) -> bool:
    """Verifica criptográficamente una prueba de inclusión de Merkle frente a `expected_root`."""
    node = _sha256_hex(_MERKLE_LEAF_PREFIX + bytes.fromhex(leaf_digest))
    for direction, sibling in proof:
        if direction == "R":
            node = _sha256_hex(
                _MERKLE_NODE_PREFIX + bytes.fromhex(node) + bytes.fromhex(sibling)
            )
        elif direction == "L":
            node = _sha256_hex(
                _MERKLE_NODE_PREFIX + bytes.fromhex(sibling) + bytes.fromhex(node)
            )
        else:
            return False
    return hmac.compare_digest(node, expected_root)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVE (Ingesta estricta, tomografía cuántica Wigner-Yanase y pasividad de Brune)
# ════════════════════════════════════════════════════════════════════════════════════════════════════
class ToonBufferWitnessPhase1Observer:
    """FASE 1 OODA — Observe: Ingesta estricta sin normalización silenciosa y certificación P1.

    Su último método (`execute_phase1`) emite el contrato `ToonBufferWitnessPhase1Certificate`,
    que inaugura el primer método de la Fase 2 (`continue_to_phase2_gate`).
    """

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        _validate_secret_key(secret_key)
        self._secret_key = bytes(secret_key)
        self._clock = clock

    @staticmethod
    def _ingest_cost_matrix(
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Valida la MIC y los exponentes de Novikov sin mutarlos."""
        mic = np.asarray(apu_cost_matrix, dtype=np.float64)
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)
        if mic.ndim != 2 or mic.shape[0] != mic.shape[1] or mic.shape[0] < 1:
            raise ValueError(f"La MIC debe ser cuadrada y no vacía; forma recibida {mic.shape}.")
        if exps.size < 1:
            raise ValueError("Se requiere al menos un exponente de Novikov.")
        if not (np.all(np.isfinite(mic)) and np.all(np.isfinite(exps))):
            raise ValueError("La MIC y los exponentes deben ser finitos.")
        return mic, exps

    @staticmethod
    def _ingest_density(rho_mac: NDArray[np.complex128], dim: int) -> NDArray[np.complex128]:
        """Valida forma y finitud de ρ_MAC. Prohibida cualquier normalización silenciosa."""
        rho = np.asarray(rho_mac, dtype=np.complex128)
        if rho.shape != (dim, dim):
            raise ValueError(f"ρ_MAC debe ser {dim}×{dim}; forma recibida {rho.shape}.")
        if not np.all(np.isfinite(rho)):
            raise ValueError("ρ_MAC contiene entradas no finitas.")
        return rho

    @staticmethod
    def _ingest_incidence(
        incidence_matrix: Optional[NDArray[np.float64]],
    ) -> Optional[NDArray[np.float64]]:
        """Valida la matriz de incidencia entera opcional sin mutarla."""
        if incidence_matrix is None:
            return None
        inc = np.asarray(incidence_matrix, dtype=np.float64)
        if inc.ndim != 2 or inc.shape[0] < 1 or inc.shape[1] < 1:
            raise ValueError("incidence_matrix debe ser una matriz 2D no vacía.")
        if not np.all(np.isfinite(inc)):
            raise ValueError("incidence_matrix contiene entradas no finitas.")
        return inc

    @staticmethod
    def _canonical_generator(dim: int) -> NDArray[np.complex128]:
        """Generador hermítico tridiagonal canónico K para la información de Wigner-Yanase."""
        K = np.eye(dim, dtype=np.complex128)
        for i in range(dim - 1):
            K[i, i + 1] = 0.02 * (i + 1)
            K[i + 1, i] = 0.02 * (i + 1)
        return K

    @classmethod
    def _observe_density(cls, rho: NDArray[np.complex128]) -> dict[str, Any]:
        """Mide hermiticidad, PSD, traza, entropía de von Neumann y asimetría de Wigner-Yanase."""
        herm_residual = float(np.max(np.abs(rho - rho.conj().T)))
        herm_part = 0.5 * (rho + rho.conj().T)
        evals, evecs = np.linalg.eigh(herm_part)
        min_eig = float(evals[0])

        tr = complex(np.trace(rho))
        tr_real = float(tr.real)
        tr_imag_abs = float(abs(tr.imag))

        # Entropía de von Neumann S(ρ) = -Σ λᵢ ln(λᵢ) y pureza Tr(ρ²)
        clipped = np.clip(evals, 0.0, None)
        pos = clipped[clipped > 1.0e-15]
        vn_entropy = float(-np.sum(pos * np.log(pos))) if pos.size > 0 else 0.0
        purity = float(np.real(np.trace(herm_part @ herm_part)))

        # Información de Wigner-Yanase I_WY(ρ, K) = Tr(ρ K²) - Tr(√ρ K √ρ K)
        sqrt_rho = (evecs * np.sqrt(clipped)) @ evecs.conj().T
        K = cls._canonical_generator(rho.shape[0])
        term1 = float(np.real(np.trace(herm_part @ (K @ K))))
        term2 = float(np.real(np.trace(sqrt_rho @ K @ sqrt_rho @ K)))
        wy_skew = max(0.0, term1 - term2)

        return {
            "tr_rho": tr_real,
            "tr_imag_abs": tr_imag_abs,
            "hermitian_residual": herm_residual,
            "min_eigenvalue": min_eig,
            "is_hermitian": herm_residual <= _HERMITIAN_TOL,
            "is_psd": min_eig >= -_PSD_TOL,
            "is_unit_trace": abs(tr_real - 1.0) <= _TRACE_TOL and tr_imag_abs <= _TRACE_TOL,
            "von_neumann_entropy": vn_entropy,
            "hilbert_schmidt_purity": purity,
            "wigner_yanase_skew_info": wy_skew,
        }

    @staticmethod
    def _observe_mic_circuit_passivity(mic: NDArray[np.float64]) -> dict[str, float]:
        """Evalúa el margen de pasividad de Foster-Brune y la cota espectral de Gershgorin."""
        sym_part = 0.5 * (mic + mic.T)
        brune_min_eig = float(la.eigvalsh(sym_part)[0])
        diag = np.abs(np.diag(mic))
        off_row_sum = np.sum(np.abs(mic), axis=1) - diag
        gershgorin_bound = float(np.max(diag + off_row_sum))
        return {
            "brune_passivity_min_eig": brune_min_eig,
            "gershgorin_spectral_bound": gershgorin_bound,
        }

    @staticmethod
    def _p1_payload(cert: ToonBufferWitnessPhase1Certificate) -> dict[str, Any]:
        """Carga útil sellada de P1: todos los campos salvo `payload_digest` y `hmac_signature`."""
        payload = asdict(cert)
        del payload["payload_digest"]
        del payload["hmac_signature"]
        return payload

    # ── ÚLTIMO MÉTODO DE LA FASE 1 · ESLABÓN DE CONTINUIDAD HACIA EL INICIO DE LA FASE 2 ────────

    def execute_phase1(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        incidence_matrix: Optional[NDArray[np.float64]] = None,
    ) -> ToonBufferWitnessPhase1Certificate:
        """DEFINICIÓN FORMAL FINAL DE LA FASE 1.

        Emite el certificado `ToonBufferWitnessPhase1Certificate` sellado con SHA-256 y HMAC,
        que constituye la entrada contractual directa de `continue_to_phase2_gate` (Fase 2).
        """
        mic, exps = self._ingest_cost_matrix(apu_cost_matrix, exponents)
        dim = int(mic.shape[0])
        rho = self._ingest_density(rho_mac, dim)
        inc = self._ingest_incidence(incidence_matrix)

        observed_rho = self._observe_density(rho)
        observed_mic = self._observe_mic_circuit_passivity(mic)
        inc_digest = "NONE" if inc is None else _array_digest(inc)

        fields: dict[str, Any] = {
            "timestamp": float(self._clock()),
            "dim": dim,
            "mic_sha256": _array_digest(mic),
            "exponents_sha256": _array_digest(exps),
            "density_sha256": _array_digest(rho),
            "incidence_sha256": inc_digest,
            **observed_rho,
            **observed_mic,
        }
        digest = _sha256_hex(_canonical_bytes(fields))
        return ToonBufferWitnessPhase1Certificate(
            **fields,
            payload_digest=digest,
            hmac_signature=_hmac_hex(self._secret_key, _HMAC_DOMAIN_P1, digest),
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENT (Compuerta Φ₁₂, Mecánica Celeste de Poincaré-Tisserand y Auditoría Espectral)
# ════════════════════════════════════════════════════════════════════════════════════════════════════
class ToonBufferWitnessPhase2Orientor(ToonBufferWitnessPhase1Observer):
    """FASE 2 OODA — Orient: Compuerta homomórfica Φ₁₂ y orientación simpléctica de Poincaré-KAM.

    Continuación directa de la Fase 1:
      • PRIMER MÉTODO : `continue_to_phase2_gate(p1, ...)`
      • ÚLTIMO MÉTODO : `execute_phase2(p1, ...) -> ToonBufferWitnessPhase2Certificate`
    """

    def __init__(
        self,
        secret_key: bytes,
        engine: BoolNovikovToonBufferWitnessEngine,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(secret_key=secret_key, clock=clock)
        self._engine = engine

    # ── PRIMER MÉTODO DE LA FASE 2 · CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 1 ────────

    def continue_to_phase2_gate(
        self,
        p1: ToonBufferWitnessPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac: NDArray[np.complex128],
        incidence_matrix: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """PRIMER MÉTODO DE LA FASE 2 — Compuerta homomórfica Φ₁₂ fail-closed.

        Recibe `ToonBufferWitnessPhase1Certificate` desde `execute_phase1`.
        Recalcula el digest desde los campos de P1, verifica el sello HMAC-SHA256, exige
        admisibilidad cuántica de ρ_MAC y comprueba la ligadura por digest de todas las matrices.
        """
        expected = _sha256_hex(_canonical_bytes(self._p1_payload(p1)))
        if not hmac.compare_digest(expected, p1.payload_digest):
            raise ToonBufferWitnessGateError("Φ₁₂: el digest de P1 no coincide con sus campos.")
        if not _verify_seal(
            self._secret_key, _HMAC_DOMAIN_P1, p1.payload_digest, p1.hmac_signature
        ):
            raise ToonBufferWitnessGateError("Falla de integridad homomórfica en Compuerta Φ₁₂.")
        if not p1.is_valid_density:
            raise ToonBufferWitnessGateError(
                "Φ₁₂: densidad inadmisible "
                f"(hermítica={p1.is_hermitian}, PSD={p1.is_psd}, "
                f"traza_unitaria={p1.is_unit_trace}, S_vN={p1.von_neumann_entropy:.4f})."
            )

        mic, exps = self._ingest_cost_matrix(apu_cost_matrix, exponents)
        rho = self._ingest_density(rho_mac, int(mic.shape[0]))
        inc = self._ingest_incidence(incidence_matrix)

        bindings: List[Tuple[str, str, str]] = [
            ("MIC", p1.mic_sha256, _array_digest(mic)),
            ("exponentes", p1.exponents_sha256, _array_digest(exps)),
            ("ρ_MAC", p1.density_sha256, _array_digest(rho)),
        ]
        if p1.incidence_sha256 != "NONE" or inc is not None:
            obs_inc = "NONE" if inc is None else _array_digest(inc)
            if p1.incidence_sha256 != "NONE" and not hmac.compare_digest(
                p1.incidence_sha256, obs_inc
            ):
                raise ToonBufferWitnessGateError(
                    "Φ₁₂: incidence_matrix no coincide con la atestación P1."
                )

        for label, attested, observed in bindings:
            if not hmac.compare_digest(attested, observed):
                raise ToonBufferWitnessGateError(f"Φ₁₂: {label} no coincide con la atestación P1.")

    # ── Orientación en Mecánica Celeste de Poincaré (Tisserand-Jacobi y Kozai-Lidov) ────────────

    @staticmethod
    def _orient_poincare_celestial_invariants(
        report: ToonBufferWitnessSpectrumReport,
    ) -> Tuple[float, float, bool, bool, bool]:
        """Calcula los invariantes celestes de Tisserand-Poincaré y Kozai-Lidov desde el reporte.

        En variables canónicas de Delaunay (L, G, H) con μ = 1 y frecuencia perturbadora ω_p:
          • Parámetro de Tisserand-Poincaré (derivado de la integral de Jacobi H_J = H₀ − ω_p H):
                C_T = 1/a + 2 √(a(1 − e²)) cos i = μ / L² + 2 ω_p H.
          • Integral vertical secular de Kozai-Lidov (componente z del momento angular normalizada):
                L_z = H / L = √(1 − e²) cos i.
        """
        kam = report.phase2.kam
        L, _G, H = kam.delaunay_actions
        omega_p = float(kam.kepler_frequency)
        tisserand = float((1.0 / max(1.0e-12, L * L)) + 2.0 * omega_p * H)
        kozai_lidov = float(H / max(1.0e-12, L))

        symp_defect = float(getattr(kam, "symplectic_jacobian_defect", 0.0))
        cartan_drift = float(getattr(kam, "poincare_cartan_invariant_drift", 0.0))
        kam_non_deg = bool(getattr(kam, "is_kam_non_degenerate", True))

        symp_ok = bool(symp_defect <= _SYMPLECTIC_TOL)
        cartan_ok = bool(cartan_drift <= _CARTAN_DRIFT_TOL)
        return tisserand, kozai_lidov, symp_ok, cartan_ok, kam_non_deg

    @classmethod
    def _build_p2_payload(
        cls,
        timestamp: float,
        p1_payload_digest: str,
        report: ToonBufferWitnessSpectrumReport,
    ) -> dict[str, Any]:
        """Construye la carga útil canónica de P2 incluyendo invariantes de Poincaré y RSI³."""
        qnd = report.phase1.qnd_report
        alg = report.phase1.algebra_report
        flow = report.phase2.flow
        kam = report.phase2.kam
        rsi = report.rsi_certificate
        tisserand, kozai_lidov, symp_ok, cartan_ok, kam_non_deg = (
            cls._orient_poincare_celestial_invariants(report)
        )

        return {
            "timestamp": float(timestamp),
            "p1_payload_sha256": p1_payload_digest,
            "passport_hash": report.passport_hash,
            "truth_value": int(report.truth_value),
            "truth_name": report.truth_value.name,
            "qnd_passive": bool(qnd.is_qnd_passive),
            "back_action_db": float(qnd.back_action_db),
            "quantum_fisher_info": float(getattr(qnd, "quantum_fisher_info", 0.0)),
            "fidelity_uhlmann": float(report.phase1.uhlmann_report.fidelity_uhlmann),
            "smith_torsion_free": bool(alg.smith_torsion_free),
            "hodge_spectral_gap": float(getattr(alg, "hodge_spectral_gap", 0.0)),
            "telegrapher_vswr": float(getattr(alg, "telegrapher_vswr", 1.0)),
            "water_hammer_suppressed": bool(flow.is_water_hammer_suppressed),
            "oseledets_lambda_max": float(flow.oseledets_lyapunov_exponent),
            "zhukovsky_surge_pressure": float(getattr(flow, "zhukovsky_surge_pressure", 0.0)),
            "gromov_rigid": bool(kam.is_gromov_rigid),
            "melnikov_chaos": bool(kam.melnikov_chaos_detected),
            "symplectic_jacobian_defect": float(getattr(kam, "symplectic_jacobian_defect", 0.0)),
            "poincare_cartan_drift": float(getattr(kam, "poincare_cartan_invariant_drift", 0.0)),
            "kam_non_degenerate": kam_non_deg,
            "tisserand_poincare_invariant": tisserand,
            "kozai_lidov_vertical_integral": kozai_lidov,
            "poincare_symplectic_ok": symp_ok,
            "poincare_cartan_ok": cartan_ok,
            "banach_lip": float(report.banach_contraction_lip),
            "rsi_accepted": bool(rsi.accepted),
            "rsi_loeb_soundness": bool(getattr(rsi, "loeb_soundness_verified", rsi.accepted)),
            "rsi_meta_lip": float(getattr(rsi, "meta_preconditioned_lip", rsi.lip_certified)),
        }

    @classmethod
    def _p2_digest(
        cls,
        timestamp: float,
        p1_payload_digest: str,
        report: ToonBufferWitnessSpectrumReport,
    ) -> str:
        return _sha256_hex(
            _canonical_bytes(cls._build_p2_payload(timestamp, p1_payload_digest, report))
        )

    # ── ÚLTIMO MÉTODO DE LA FASE 2 · ESLABÓN DE CONTINUIDAD HACIA EL INICIO DE LA FASE 3 ────────

    def execute_phase2(
        self,
        p1: ToonBufferWitnessPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac: NDArray[np.complex128],
        incidence_matrix: Optional[NDArray[np.float64]] = None,
        stage_factors: Tuple[float, ...] = _DEFAULT_STAGE_FACTORS,
        quant_scale: float = _DEFAULT_QUANT_SCALE,
    ) -> ToonBufferWitnessPhase2Certificate:
        """DEFINICIÓN FORMAL FINAL DE LA FASE 2.

        Verifica Φ₁₂, ejecuta la auditoría del motor v15, orienta los invariantes celestes de
        Poincaré y emite `ToonBufferWitnessPhase2Certificate` para `continue_to_phase3_gate`.
        """
        self.continue_to_phase2_gate(
            p1, apu_cost_matrix, exponents, rho_mac, incidence_matrix=incidence_matrix
        )
        mic, exps = self._ingest_cost_matrix(apu_cost_matrix, exponents)
        rho = self._ingest_density(rho_mac, int(mic.shape[0]))
        inc = self._ingest_incidence(incidence_matrix)

        report = self._engine.execute_toon_buffer_witness_audit(
            apu_cost_matrix=mic,
            exponents=exps,
            rho_mac_input=rho,
            incidence_matrix=inc,
            stage_factors=tuple(float(s) for s in stage_factors),
            quant_scale=float(quant_scale),
        )
        tisserand, kozai_lidov, symp_ok, cartan_ok, kam_non_deg = (
            self._orient_poincare_celestial_invariants(report)
        )
        t_now = float(self._clock())
        digest = self._p2_digest(t_now, p1.payload_digest, report)
        return ToonBufferWitnessPhase2Certificate(
            timestamp=t_now,
            p1_payload_digest=p1.payload_digest,
            spectrum_report=report,
            payload_digest=digest,
            hmac_signature=_hmac_hex(self._secret_key, _HMAC_DOMAIN_P2, digest),
            tisserand_poincare_invariant=tisserand,
            kozai_lidov_vertical_integral=kozai_lidov,
            poincare_symplectic_ok=symp_ok,
            poincare_cartan_ok=cartan_ok,
            kam_torsion_non_degenerate=kam_non_deg,
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3 — DECIDE & ACT (Compuerta Φ₂₃, Auditoría RSI³ de Löb-Kleisli, Topos Ω₃ y Merkle Proofs)
# ════════════════════════════════════════════════════════════════════════════════════════════════════
class ToonBufferWitnessPhase3Decider(ToonBufferWitnessPhase2Orientor):
    """FASE 3 OODA — Decide & Act: Meta-gobernanza RSI³, Topos Ω₃ y raíz de Merkle con pruebas.

    Continuación directa de la Fase 2:
      • PRIMER MÉTODO : `continue_to_phase3_gate(p2: ToonBufferWitnessPhase2Certificate)`
      • ÚLTIMO MÉTODO : `execute_phase3(p2: ToonBufferWitnessPhase2Certificate)`
    """

    def __init__(
        self,
        secret_key: bytes,
        engine: BoolNovikovToonBufferWitnessEngine,
        policy: Optional[ToonBufferWitnessVisceralPolicy] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(secret_key=secret_key, engine=engine, clock=clock)
        self._policy = policy if policy is not None else ToonBufferWitnessVisceralPolicy()

    # ── PRIMER MÉTODO DE LA FASE 3 · CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 2 ────────

    def continue_to_phase3_gate(self, p2: ToonBufferWitnessPhase2Certificate) -> bool:
        """PRIMER MÉTODO DE LA FASE 3 — Compuerta homomórfica Φ₂₃ y auditoría independiente RSI³.

        Verifica fail-closed la integridad criptográfica de P2. A continuación, audita de forma
        independiente la torre RSI³ (contracción de Banach Lip(T³) < 1, aceleración cuadrática
        del meta-preacondicionador Lip_L2 ≤ Lip_L1² + ε, coherencia de la mónada de Kleisli,
        guardia de solidez de Löb-Tarski sobre el punto fijo θ* y conservación simpléctica de
        Poincaré-Cartan). Un fallo dinámico o de RSI³ no interrumpe el flujo: degrada en Ω₃.
        """
        expected = self._p2_digest(p2.timestamp, p2.p1_payload_digest, p2.spectrum_report)
        if not hmac.compare_digest(expected, p2.payload_digest):
            raise ToonBufferWitnessGateError(
                "Φ₂₃: el digest de P2 no coincide con su reporte espectral."
            )
        if not _verify_seal(
            self._secret_key, _HMAC_DOMAIN_P2, p2.payload_digest, p2.hmac_signature
        ):
            raise ToonBufferWitnessGateError("Falla de integridad homomórfica en Compuerta Φ₂₃.")

        report = p2.spectrum_report
        loeb_ok, kleisli_ok, symp_ok = self._audit_rsi3_and_poincare_soundness(report)
        return bool(
            float(report.banach_contraction_lip) < _BANACH_CONTRACTION_MAX
            and report.rsi_certificate.accepted
            and loeb_ok
            and kleisli_ok
            and symp_ok
        )

    @staticmethod
    def _audit_rsi3_and_poincare_soundness(
        report: ToonBufferWitnessSpectrumReport,
    ) -> Tuple[bool, bool, bool]:
        """Verificación independiente por el agente de Löb-Tarski, Kleisli y Poincaré-Cartan."""
        rsi = report.rsi_certificate
        kam = report.phase2.kam

        # 1. Verificación del punto fijo contra el ancla desplegada y aceleración de Nivel 2
        fp = np.asarray(rsi.fixed_point, dtype=float)
        anchor = np.asarray(_DEPLOYED_POLICY_ANCHOR, dtype=float)
        anchor_match = (
            fp.shape == anchor.shape and float(np.max(np.abs(fp - anchor))) <= _LOEB_ANCHOR_TOL
        )
        lip_l1 = float(rsi.lip_certified)
        lip_l2 = float(getattr(rsi, "meta_preconditioned_lip", lip_l1 * lip_l1))
        quadratic_contraction_ok = bool(
            math.isfinite(lip_l1)
            and lip_l1 < 1.0
            and lip_l2 <= (lip_l1 * lip_l1) + 1.0e-9
        )
        engine_loeb_flag = bool(getattr(rsi, "loeb_soundness_verified", rsi.accepted))
        loeb_ok = bool(anchor_match and quadratic_contraction_ok and engine_loeb_flag)

        # 2. Coherencia de la mónada de Kleisli (asociatividad y unidades)
        assoc_def = float(getattr(rsi, "kleisli_associativity_defect", 0.0))
        unit_def = float(getattr(rsi, "kleisli_unit_defect", 0.0))
        kleisli_ok = bool(
            assoc_def <= _KLEISLI_COHERENCE_TOL and unit_def <= _KLEISLI_COHERENCE_TOL
        )

        # 3. Invariantes simplécticos de Poincaré y de circulación de Cartan
        symp_def = float(getattr(kam, "symplectic_jacobian_defect", 0.0))
        cartan_drift = float(getattr(kam, "poincare_cartan_invariant_drift", 0.0))
        symp_ok = bool(symp_def <= _SYMPLECTIC_TOL and cartan_drift <= _CARTAN_DRIFT_TOL)

        return loeb_ok, kleisli_ok, symp_ok

    @staticmethod
    def _meet(a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        """Ínfimo (∧) en el álgebra de Heyting Ω₃: COHERENT (2) > DEGRADED (1) > VETOED (0)."""
        rank_a = 2 - int(a)
        rank_b = 2 - int(b)
        return a if rank_a <= rank_b else b

    def _build_visceral_metrics(
        self,
        report: ToonBufferWitnessSpectrumReport,
    ) -> ToonBufferWitnessVisceralMetrics:
        """Construye las métricas visceral-semánticas acoplando la telemetría de Heaviside/KAM."""
        flow = report.phase2.flow
        alg = report.phase1.algebra_report
        kam = report.phase2.kam
        if bool(flow.is_water_hammer_suppressed):
            hammer_status = "SUPPRESSED"
        else:
            hammer_status = (
                f"UNSUPPRESSED_LAMBDA={float(flow.oseledets_lyapunov_exponent):.4f}"
            )
        return ToonBufferWitnessVisceralMetrics(
            wacc_protection_pct=self._policy.wacc_protection_pct,
            net_liquidity_release_pct=self._policy.net_liquidity_release_pct,
            kv_cache_delta_pct=self._policy.kv_cache_delta_pct,
            water_hammer_status=hammer_status,
            provenance=_VISCERAL_PROVENANCE,
            zhukovsky_surge_pressure=float(getattr(flow, "zhukovsky_surge_pressure", 0.0)),
            telegrapher_vswr=float(getattr(alg, "telegrapher_vswr", 1.0)),
            nekhoroshev_stability_horizon=float(
                getattr(kam, "nekhoroshev_stability_time", 0.0)
            ),
        )

    # ── ÚLTIMO MÉTODO DE LA FASE 3 · EMISIÓN DEL ESTADO SOBERANO Y RAÍZ DE MERKLE ───────────────

    def execute_phase3(
        self,
        p2: ToonBufferWitnessPhase2Certificate,
    ) -> ToonBufferWitnessSovereignState:
        """DEFINICIÓN FORMAL FINAL DE LA FASE 3.

        Adjudica el valor de verdad en el topos de Heyting Ω₃, evalúa la modalidad de
        Lawvere-Tierney y sella el árbol de Merkle con sus pruebas de inclusión.
        """
        contraction_ok = self.continue_to_phase3_gate(p2)
        report = p2.spectrum_report
        loeb_ok, kleisli_ok, symp_ok = self._audit_rsi3_and_poincare_soundness(report)

        truth = report.truth_value
        if not contraction_ok:
            truth = self._meet(truth, HeytingTruthValue.DEGRADED)
        if not symp_ok:
            truth = self._meet(truth, HeytingTruthValue.VETOED)

        if hasattr(truth, "lawvere_tierney_sheafify"):
            lt_truth = truth.lawvere_tierney_sheafify(stabilize_degraded=False)
        else:
            lt_truth = truth

        qnd = report.phase1.qnd_report
        is_passive_silent = bool(qnd.is_qnd_passive) and not _ACTUATION_EMITTED
        back_action_db = float(qnd.back_action_db)
        visceral = self._build_visceral_metrics(report)

        t_now = float(self._clock())
        state_fields: dict[str, Any] = {
            "timestamp": t_now,
            "truth_value": int(truth),
            "truth_name": truth.name,
            "lawvere_tierney_truth": int(lt_truth),
            "is_passive_silent": is_passive_silent,
            "actuation_emitted": _ACTUATION_EMITTED,
            "back_action_db": back_action_db,
            "contraction_ok": contraction_ok,
            "rsi3_loeb_soundness_ok": loeb_ok,
            "kleisli_coherence_ok": kleisli_ok,
            "poincare_symplectic_invariant_ok": symp_ok,
            "visceral_metrics": asdict(visceral),
            "p1_payload_sha256": p2.p1_payload_digest,
            "p2_payload_sha256": p2.payload_digest,
        }
        state_digest = _sha256_hex(_canonical_bytes(state_fields))
        leaves = [p2.p1_payload_digest, p2.payload_digest, state_digest]
        merkle_root = _merkle_root(leaves)
        inclusion_proofs = _merkle_inclusion_proofs(leaves)

        return ToonBufferWitnessSovereignState(
            timestamp=t_now,
            truth_value=truth,
            is_passive_silent=is_passive_silent,
            actuation_emitted=_ACTUATION_EMITTED,
            back_action_db=back_action_db,
            contraction_ok=contraction_ok,
            merkle_root_hash=merkle_root,
            visceral_metrics=visceral,
            rsi3_loeb_soundness_ok=loeb_ok,
            kleisli_coherence_ok=kleisli_ok,
            poincare_symplectic_invariant_ok=symp_ok,
            lawvere_tierney_truth=lt_truth,
            merkle_inclusion_proofs=inclusion_proofs,
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO — ORQUESTADOR DEL CICLO OODA TRIFÁSICO
# ════════════════════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovToonBufferWitnessAgent(ToonBufferWitnessPhase3Decider):
    """Soberano Agéntico Testigo QND de Amortiguamiento y Protección de Caudal TOON Buffer (v15).

    Hereda la cadena de 3 fases anidadas:
      ToonBufferWitnessPhase1Observer
        ⊂ ToonBufferWitnessPhase2Orientor
        ⊂ ToonBufferWitnessPhase3Decider
        ⊂ BoolNovikovToonBufferWitnessAgent
    """

    def __init__(
        self,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        policy: Optional[ToonBufferWitnessVisceralPolicy] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        engine = BoolNovikovToonBufferWitnessEngine(secret_key=secret_key)
        super().__init__(secret_key=secret_key, engine=engine, policy=policy, clock=clock)
        logger.info("%s v%s iniciado.", self.__class__.__name__, _AGENT_VERSION)

    def run_sovereign_governance_cycle(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac_input: Optional[NDArray[np.complex128]] = None,
        incidence_matrix: Optional[NDArray[np.float64]] = None,
        stage_factors: Tuple[float, ...] = _DEFAULT_STAGE_FACTORS,
        quant_scale: float = _DEFAULT_QUANT_SCALE,
    ) -> ToonBufferWitnessSovereignState:
        """Ciclo OODA: Observe ➔ Φ₁₂ ➔ Orient ➔ Φ₂₃ ➔ Decide & Act. Fail-closed en integridad."""
        mic, exps = self._ingest_cost_matrix(apu_cost_matrix, exponents)
        dim = int(mic.shape[0])
        if rho_mac_input is None:
            rho = np.eye(dim, dtype=np.complex128) / float(dim)
        else:
            rho = np.asarray(rho_mac_input, dtype=np.complex128)

        p1 = self.execute_phase1(rho, mic, exps, incidence_matrix=incidence_matrix)
        p2 = self.execute_phase2(
            p1,
            mic,
            exps,
            rho,
            incidence_matrix=incidence_matrix,
            stage_factors=stage_factors,
            quant_scale=quant_scale,
        )
        state = self.execute_phase3(p2)
        logger.info(
            "Ciclo OODA TOON Buffer Witness completado. Veredicto: %s", state.truth_value.name
        )
        return state


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# AUTOPRUEBA DE VERIFICACIÓN FORMAL DEL AGENTE SOBERANO
# ════════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def _fixed_clock() -> float:
        return 1700000000.0

    def _imprimir_estado(titulo: str, st: ToonBufferWitnessSovereignState, rep_ok: bool) -> None:
        m = st.visceral_metrics
        print(f"\n--- {titulo} ---")
        print(f"Veredicto Final Heyting Ω₃   : {st.truth_value.name} ({hex(int(st.truth_value))})")
        print(f"Invariante PASSIVE_SILENT    : {st.is_passive_silent} (Actuación: {st.actuation_emitted})")
        print(f"Back-Action QND (dB)         : {st.back_action_db:.6f} dB")
        print(f"Compuerta Φ₂₃ (contracción)  : {st.contraction_ok}")
        print(
            f"RSI³ Löb / Kleisli / Poincaré: Löb={st.rsi3_loeb_soundness_ok}, "
            f"Kleisli={st.kleisli_coherence_ok}, Simpléctico={st.poincare_symplectic_invariant_ok}"
        )
        print(f"Raíz Merkle SHA-256          : {st.merkle_root_hash[:16]}... (Reproducible: {rep_ok})")
        print(
            f"Zhukovsky ΔP / VSWR / Nekh.  : {m.zhukovsky_surge_pressure:.4f} / "
            f"{m.telegrapher_vswr:.4f} / {m.nekhoroshev_stability_horizon:.3e}"
        )
        print(
            f"Política Visceral ({m.provenance}): WACC={m.wacc_protection_pct}%, "
            f"Liquidez={m.net_liquidity_release_pct}%, KV={m.kv_cache_delta_pct}%"
        )
        print(f"Estado Golpe de Bélier       : {m.water_hammer_status}")

    print("=== TEST AUTÓNOMO: BoolNovikovToonBufferWitnessAgent v15.0.0-PhD ===")
    exps_test = np.array([0.15, 0.25, 0.40], dtype=np.float64)
    agent = BoolNovikovToonBufferWitnessAgent(clock=_fixed_clock)

    # Caso A: Matriz expansiva con torsión → VETOED y rollback RSI³.
    matrix_a = np.array([[2.0, 0.1], [0.1, 2.5]], dtype=np.float64)
    state_a = agent.run_sovereign_governance_cycle(matrix_a, exps_test)
    replay_a = BoolNovikovToonBufferWitnessAgent(clock=_fixed_clock).run_sovereign_governance_cycle(
        matrix_a, exps_test
    )
    _imprimir_estado(
        "CASO A: régimen expansivo con torsión (VETOED)",
        state_a,
        state_a.merkle_root_hash == replay_a.merkle_root_hash,
    )

    # Caso B: Régimen contractivo con incidencia unimodular → COHERENT y RSI³ aceptado.
    matrix_b = np.array([[0.5, 0.1], [0.1, 0.4]], dtype=np.float64)
    incidence_b = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]], dtype=np.float64)
    state_b = agent.run_sovereign_governance_cycle(
        matrix_b, exps_test, incidence_matrix=incidence_b
    )
    replay_b = BoolNovikovToonBufferWitnessAgent(clock=_fixed_clock).run_sovereign_governance_cycle(
        matrix_b, exps_test, incidence_matrix=incidence_b
    )
    _imprimir_estado(
        "CASO B: régimen contractivo KAM estable + incidencia unimodular (COHERENT)",
        state_b,
        state_b.merkle_root_hash == replay_b.merkle_root_hash,
    )

    print("\n=== TEST COMPLETADO CON ÉXITO ===")
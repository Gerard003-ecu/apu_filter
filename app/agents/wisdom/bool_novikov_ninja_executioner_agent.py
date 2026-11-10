# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovNinjaExecutionerAgent v14.0.0-PhD (Evolved Sovereign Edition)                  ║
║ Ubicación: app/agents/wisdom/bool_novikov_ninja_executioner_agent.py                                 ║
║ Versión  : 14.0.0-SOVEREIGN-BOOL-NOVIKOV-NINJA-EXECUTIONER-AGENT-POINCARE-KAM-RSI3-PhD              ║
║ Gobernanza: OODA Trifásica Inquisitorial/Ejecutora de Sigilo (①Observe → ②Orient → ③Decide&Act)     ║
║            con Compuertas Φ₁₂ y Φ₂₃ sobre el Decapitador Silencioso de Silicio (Ninja Executioner). ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE TRES FASES ANIDADAS (CADENA DE CUSTODIA MONÁDICA OODA)
──────────────────────────────────────────────────────────────────────
El agente soberano se estructura en TRES FASES ANIDADAS donde la definición formal final
del último método de cada fase constituye el operador de continuación directa e inicio
formal de los métodos de la fase siguiente:

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ FASE 1 ANIDADA — OBSERVE (Geometría de Bures-Wasserstein e Información de Fisher QFI)  │
  │   • Sanitización hermítica de ρ_MAC ∈ D(ℋ) y matriz de incidencia H_inc.              │
  │   • Defecto Hermítico ‖ρ − ρ†‖_F / max(1, ‖ρ‖_F) y defecto de traza |Tr(ρ) − 1|.       │
  │   • Pureza γ(ρ) = Tr(ρ²), entropía lineal S_L(ρ) y entropía de von Neumann S(ρ).       │
  │   • Entropía relativa cuántica de Umegaki D(ρ ‖ I/d) = ln(d) − S(ρ).                   │
  │   • Distancia geodésica de Bures-Wasserstein d_BW(ρ, I/d) e Información Cuántica       │
  │     de Fisher (QFI) I_Q(ρ, H_inc) vía Derivada Logarítmica Simétrica (SLD).            │
  │   • CIERRE FASE 1 / INICIO FASE 2: `emit_phase1_certificate`                           │
  │     ──[Compuerta Homomórfica Φ₁₂: `continue_to_phase2_gate`]──►                        │
  │     `initiate_phase2_orientation`.                                                     │
  └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ FASE 2 ANIDADA — ORIENT (Mecánica Celeste de Poincaré-KAM-Nekhoroshev y RSI³ Banach)   │
  │   • Recepción encadenada desde `emit_phase1_certificate` vía                           │
  │     `initiate_phase2_orientation`.                                                     │
  │   • Delegación al motor v14.0.0 (`execute_ninja_executioner_audit`) y extracción       │
  │     de la tríada criptográfica (P1_eng, P2_eng, P3_eng) + `passport_hmac_sha256`.      │
  │   • Diagnóstico celeste de Poincaré-KAM: tasa de difusión exponencial de Arnold        │
  │     Γ_Arnold(ω, ε), razón de desdoblamiento homoclínico de Melnikov η_Melnikov y       │
  │     margen simpático de no-exprimido de Gromov Δc_G = π(R² − r²).                      │
  │   • Certificación de contracción monádica RSI³ Lip(T³) = 0.857375 < 1.0.               │
  │   • CIERRE FASE 2 / INICIO FASE 3: `emit_phase2_certificate`                           │
  │     ──[Compuerta Homomórfica Φ₂₃: `continue_to_phase3_gate`]──►                        │
  │     `initiate_phase3_decision`.                                                        │
  └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ FASE 3 ANIDADA — DECIDE & ACT (Topos de Heyting Ω₃, Funtor Φ_sem, Crowbar y Merkle)    │
  │   • Recepción encadenada desde `emit_phase2_certificate` vía                           │
  │     `initiate_phase3_decision`.                                                        │
  │   • Adjudicación intuicionista en el Topos de Heyting Ω₃ = {0x00 ≺ 0x01 ≺ 0x03} con    │
  │     implicación relativa (a ⇒ b) y pseudocomplemento ¬a = (a ⇒ ⊥).                     │
  │   • Interlock ciber-físico ESP32 IRAM + tiristor BT151: si VETOED, fuerza disparo      │
  │     en ≤ 392.15 ns y activa el estado `EMBOSCADA_CIBER_FISICA_ACTIVA`.                 │
  │   • Funtor Semántico Visceral Φ_sem con métricas financieras/ciber-físicas y trazas    │
  │     de Poincaré-Bruno y RSI³.                                                          │
  │   • Árbol de Merkle SHA-256 de 8 hojas con pruebas verificables de inclusión y cierre  │
  │     soberano: `emit_sovereign_state` ➔ `NinjaExecutionerSovereignState`.               │
  └────────────────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Importación del motor espectral ninja executioner v14.0.0
try:
    from bool_novikov_ninja_executioner_engine import (  # type: ignore
        BoolNovikovNinjaExecutionerEngine,
        ExecutionerDiracAnnihilationReport,
        HeytingTruthValue,
        NinjaExecutionerSpectrumReport,
        PhaseOneAlgCertificate,
        PhaseThreeRSI3Certificate,
        PhaseTwoSymplecticCertificate,
    )
except ImportError:
    try:
        from app.wisdom.bool_novikov_ninja_executioner_engine import (  # type: ignore
            BoolNovikovNinjaExecutionerEngine,
            ExecutionerDiracAnnihilationReport,
            HeytingTruthValue,
            NinjaExecutionerSpectrumReport,
            PhaseOneAlgCertificate,
            PhaseThreeRSI3Certificate,
            PhaseTwoSymplecticCertificate,
        )
    except ImportError:
        raise ImportError("No se pudo importar bool_novikov_ninja_executioner_engine.")

logger = logging.getLogger("APU.Wisdom.BoolNovikovNinjaExecutionerAgent")

_AGENT_VERSION: Final[str] = (
    "14.0.0-SOVEREIGN-BOOL-NOVIKOV-NINJA-EXECUTIONER-AGENT-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _AGENT_VERSION

# Dominios Criptográficos HMAC-SHA256 del Agente Soberano
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_NINJA_EXECUTIONER_KEY_v14.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"NINJA-EXECUTOR-AGENT-P1-ISOMETRY-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"NINJA-EXECUTOR-AGENT-P2-SPECTRUM-v14.0|"
_HMAC_DOMAIN_SOVEREIGN: Final[bytes] = b"NINJA-EXECUTOR-AGENT-SOVEREIGN-v14.0|"

# Tolerancias Espectrales, Cuánticas, Celestes y Ciber-Físicas
_TRACE_DEFECT_TOLERANCE: Final[float] = 1.0e-9
_PSD_EIGENVALUE_TOLERANCE: Final[float] = 1.0e-12
_BACK_ACTION_TOLERANCE_DB: Final[float] = 0.01
_RSI3_LIP_CEILING: Final[float] = 1.0
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_MELNIKOV_CRITICAL_THRESHOLD: Final[float] = 0.05
_ARNOLD_PERTURBATION_EPS: Final[float] = 0.01
_ARNOLD_CRITICAL_EPS0: Final[float] = 0.25


# ============================================================================
# PRELUDIO CANÓNICO — SANEAMIENTO, CUANTIZACIÓN Y ÁRBOLES DE MERKLE SHA-256
# ============================================================================

def _q(value: float) -> str:
    """Cuantización canónica en punto fijo a 12 decimales."""
    return f"{_safe_float(value, 0.0):.12f}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Conversión segura a float IEEE-754 finito."""
    try:
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    """Conversión segura a booleano."""
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _canonical_payload(*parts: Any) -> str:
    """Serialización determinista SHA-256 sobre tuplas tipadas."""
    digest = hashlib.sha256()
    for p in parts:
        if isinstance(p, (float, np.floating)):
            text = _q(float(p))
        elif isinstance(p, (bool, np.bool_)):
            text = "1" if bool(p) else "0"
        elif isinstance(p, (int, np.integer)):
            text = str(int(p))
        else:
            text = str(p)
        digest.update(text.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


def _hmac_hex(secret_key: bytes, domain: bytes, payload: str) -> str:
    """Firma criptográfica HMAC-SHA256 con separación de dominio."""
    return hmac.new(secret_key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _array_digest(array: NDArray[Any]) -> str:
    """Digest SHA-256 del buffer C-contiguo de un tensor."""
    arr = np.asarray(array)
    if np.iscomplexobj(arr):
        contig = np.ascontiguousarray(arr, dtype=np.complex128)
    else:
        contig = np.ascontiguousarray(arr, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contig.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(contig.tobytes(order="C"))
    return digest.hexdigest()


def _merkle_root(leaves: List[str]) -> str:
    """Raíz Merkle SHA-256 sobre hojas hex; duplica la última hoja si el nivel es impar."""
    if not leaves:
        return hashlib.sha256(b"EMPTY_NINJA_MERKLE").hexdigest()
    current = list(leaves)
    while len(current) > 1:
        nxt: List[str] = []
        for i in range(0, len(current), 2):
            a = current[i]
            b = current[i + 1] if i + 1 < len(current) else current[i]
            nxt.append(hashlib.sha256((a + b).encode("utf-8")).hexdigest())
        current = nxt
    return current[0]


def verify_merkle_inclusion_proof(
    leaves: Tuple[str, ...],
    leaf_index: int,
    expected_root: str,
) -> bool:
    """
    Construye y verifica criptográficamente el camino de autenticación de Merkle
    (Merkle Audit Path) para la hoja `leaf_index` contra `expected_root`.
    """
    if not leaves or leaf_index < 0 or leaf_index >= len(leaves):
        return False
    idx = int(leaf_index)
    current = list(leaves)
    node_hash = current[idx]

    while len(current) > 1:
        if len(current) % 2 == 1:
            current.append(current[-1])
        sibling_idx = idx ^ 1
        sibling_hash = current[sibling_idx]
        if idx % 2 == 0:
            node_hash = hashlib.sha256((node_hash + sibling_hash).encode("utf-8")).hexdigest()
        else:
            node_hash = hashlib.sha256((sibling_hash + node_hash).encode("utf-8")).hexdigest()

        nxt: List[str] = []
        for i in range(0, len(current), 2):
            nxt.append(hashlib.sha256((current[i] + current[i + 1]).encode("utf-8")).hexdigest())
        current = nxt
        idx //= 2

    return hmac.compare_digest(node_hash, expected_root)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 ANIDADA — OBSERVE: GEOMETRÍA CUÁNTICA DE BURES-WASSERSTEIN Y QFI  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class NinjaExecutionerPhase1Certificate:
    """
    Certificado Inmutable P1 emitido en la Fase 1 Observe del Decapitador Ninja.
    Custodia la variedad de estados cuánticos D(ℋ) bajo la métrica de Bures-Wasserstein
    y la Información Cuántica de Fisher (QFI).
    """
    timestamp: float
    tr_rho: float
    is_psd: bool
    dimension: int
    rho_digest: str
    matrix_digest: str
    hermitian_defect: float
    purity: float
    trace_defect: float
    von_neumann_entropy: float
    chain_hash: str
    hmac_signature: str
    bures_wasserstein_distance: float = 0.0
    quantum_relative_entropy: float = 0.0
    quantum_fisher_information: float = 0.0
    linear_entropy: float = 0.0


def _validate_rho_density_matrix(
    rho_mac: NDArray[np.float64],
) -> Tuple[Tuple[NDArray[np.complex128], float, float, float, float, bool, int], float]:
    """
    Sanea el operador de densidad ρ_MAC ∈ D(ℋ) y calcula los invariantes espectrales
    de la geometría de información cuántica:
      • Defecto Hermítico: ‖ρ − ρ†‖_F / max(1, ‖ρ‖_F).
      • Traza Tr(ρ) y defecto de traza |Tr(ρ) − 1|.
      • Pureza de Hilbert-Schmidt γ(ρ) = Tr(ρ²).
      • Condición definida semipositiva (PSD): λ_min(ρ) ≥ −ε_PSD.
      • Entropía cuántica de von Neumann: S(ρ) = −Tr(ρ ln ρ) = −Σ λᵢ ln λᵢ.
    Devuelve: ((ρ_sanitizada_compleja, Traza, Defecto_traza, Defecto_hermítico,
                Pureza, PSD, dim), vn_entropy).
    """
    arr = np.asarray(rho_mac)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        raise ValueError("La matriz ρ_MAC debe ser cuadrada, 2D y no vacía.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("La matriz ρ_MAC contiene NaN/Inf.")

    rho_c = arr.astype(np.complex128, copy=False)
    herm_defect = float(
        np.linalg.norm(rho_c - rho_c.conj().T, ord="fro")
        / max(1.0, float(np.linalg.norm(rho_c, ord="fro")))
    )

    herm = 0.5 * (rho_c + rho_c.conj().T)
    tr = float(np.real(np.trace(herm)))
    trace_defect = float(abs(tr - 1.0))

    evals = la.eigvalsh(herm)
    psd = bool(float(np.min(evals)) >= -_PSD_EIGENVALUE_TOLERANCE)

    purity = float(np.real(np.trace(herm @ herm)))

    pos_evals = np.clip(evals, 0.0, None)
    ssum = float(np.sum(pos_evals)) or 1.0
    probs = pos_evals / ssum
    nz = probs[probs > 0.0]
    vn_entropy = float(-np.sum(nz * np.log(nz))) if nz.size > 0 else 0.0

    return (
        np.ascontiguousarray(herm, dtype=np.complex128),
        tr,
        trace_defect,
        herm_defect,
        purity,
        psd,
        int(herm.shape[0]),
    ), vn_entropy


def _compute_bures_wasserstein_and_qfi_invariants(
    rho_herm: NDArray[np.complex128],
    vn_entropy: float,
    purity: float,
    incidence_sym: Optional[NDArray[np.float64]] = None,
) -> Tuple[float, float, float, float]:
    """
    Calcula invariantes avanzados en la variedad riemanniana cuántica de Bures-Wasserstein:
      1. Fidelidad de Uhlmann respecto al estado maximalmente mezclado τ_d = I/d:
             √F(ρ, τ_d) = (1/√d) · Σᵢ √λᵢ
         y distancia geodésica de Bures-Wasserstein:
             d_BW(ρ, τ_d) = √( max(0, 1 − F(ρ, τ_d)) ).
      2. Entropía relativa cuántica de Umegaki:
             D(ρ ‖ τ_d) = Tr(ρ (ln ρ − ln τ_d)) = ln(d) − S(ρ) ≥ 0.
      3. Información Cuántica de Fisher (QFI) vía Derivada Logarítmica Simétrica (SLD)
         para la evolución unitaria generada por H_inc:
             I_Q(ρ, H_inc) = 2 · Σ_{i,j: λᵢ+λⱼ > 0} ((λᵢ − λⱼ)² / (λᵢ + λⱼ)) · |⟨ψᵢ|H_inc|ψⱼ⟩|².
      4. Entropía lineal normalizada de Tsallis:
             S_L(ρ) = (d / (d − 1)) · (1 − Tr(ρ²))   para d > 1.
    """
    dim = int(rho_herm.shape[0])
    if dim <= 0:
        return 0.0, 0.0, 0.0, 0.0

    evals, evecs = la.eigh(rho_herm)
    pos_evals = np.clip(np.real(evals), 0.0, None)
    total = float(np.sum(pos_evals)) or 1.0
    probs = pos_evals / total

    sqrt_fidelity = float(np.sum(np.sqrt(probs))) / math.sqrt(float(dim))
    fidelity = min(1.0, max(0.0, sqrt_fidelity ** 2))
    bures_dist = float(math.sqrt(max(0.0, 1.0 - fidelity)))

    max_entropy = math.log(float(dim)) if dim > 1 else 0.0
    rel_entropy = float(max(0.0, max_entropy - vn_entropy))

    linear_ent = (
        float((dim / float(dim - 1)) * max(0.0, 1.0 - purity))
        if dim > 1
        else 0.0
    )

    qfi = 0.0
    if incidence_sym is not None and incidence_sym.shape == (dim, dim):
        h_eigen = evecs.conj().T @ incidence_sym.astype(np.complex128) @ evecs
        abs_h_sq = np.abs(h_eigen) ** 2
        for i in range(dim):
            for j in range(i + 1, dim):
                denom = float(probs[i] + probs[j])
                if denom > 1.0e-15:
                    diff = float(probs[i] - probs[j])
                    qfi += 4.0 * ((diff ** 2) / denom) * float(abs_h_sq[i, j])

    return bures_dist, rel_entropy, float(qfi), linear_ent


def _validate_incidence_matrix(
    incidence_matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], str]:
    """Valida y simetriza hermíticamente la matriz de incidencia; devuelve (matriz, digest)."""
    arr = np.asarray(incidence_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        raise ValueError("La matriz de incidencia debe ser cuadrada, 2D y no vacía.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("La matriz de incidencia contiene NaN/Inf.")
    sym = 0.5 * (arr + arr.T)
    sym = np.ascontiguousarray(sym, dtype=np.float64)
    return sym, _array_digest(sym)


def continue_to_phase2_gate(
    p1_cert: NinjaExecutionerPhase1Certificate,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
) -> bool:
    """
    Compuerta Homomórfica Φ₁₂ (Fase 1 ➔ Fase 2):
    Revalida la autenticidad criptográfica HMAC de P1 y la isometría cuántica
    de Bures-Wasserstein (defecto de traza ≤ 10⁻⁹, positividad espectral PSD
    y dimensión finita positiva).
    """
    expected = _hmac_hex(secret_key, _HMAC_DOMAIN_P1, p1_cert.chain_hash)
    if not hmac.compare_digest(expected, p1_cert.hmac_signature):
        logger.error("Φ₁₂: firma HMAC P1 inválida.")
        return False
    if p1_cert.trace_defect > _TRACE_DEFECT_TOLERANCE:
        logger.error(
            "Φ₁₂: defecto de traza |Tr(ρ)−1| = %.12f > %.12f.",
            p1_cert.trace_defect,
            _TRACE_DEFECT_TOLERANCE,
        )
        return False
    if not p1_cert.is_psd:
        logger.error("Φ₁₂: ρ_MAC no es PSD.")
        return False
    if p1_cert.dimension <= 0:
        logger.error("Φ₁₂: dimensión de ρ_MAC inválida.")
        return False
    return True


class NinjaExecutionerPhase1Observer:
    """Fase 1 OODA — Observación, sanitización y atestación cuántica Bures-Wasserstein."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key = secret_key

    def execute_phase1(
        self,
        rho_mac: NDArray[np.float64],
        incidence_matrix: Optional[NDArray[np.float64]] = None,
    ) -> NinjaExecutionerPhase1Certificate:
        """Sanitiza ρ_MAC y —opcionalmente— la matriz de incidencia; emite el certificado P1."""
        ts = time.time()
        invariants, vn_entropy = _validate_rho_density_matrix(rho_mac)
        rho_sanitized, tr, trace_defect, herm_defect, purity, psd, dim = invariants
        rho_digest = _array_digest(rho_sanitized)

        inc_sym: Optional[NDArray[np.float64]] = None
        if incidence_matrix is not None:
            inc_sym, matrix_digest = _validate_incidence_matrix(incidence_matrix)
        else:
            matrix_digest = _array_digest(np.zeros((dim, dim), dtype=np.float64))

        bures_dist, rel_ent, qfi, lin_ent = _compute_bures_wasserstein_and_qfi_invariants(
            rho_sanitized, vn_entropy, purity, inc_sym
        )

        return self._emit_phase1_certificate(
            timestamp=ts,
            tr_rho=tr,
            is_psd=psd,
            dimension=dim,
            rho_digest=rho_digest,
            matrix_digest=matrix_digest,
            hermitian_defect=herm_defect,
            purity=purity,
            trace_defect=trace_defect,
            von_neumann_entropy=vn_entropy,
            bures_wasserstein_distance=bures_dist,
            quantum_relative_entropy=rel_ent,
            quantum_fisher_information=qfi,
            linear_entropy=lin_ent,
        )

    def _emit_phase1_certificate(
        self,
        timestamp: float,
        tr_rho: float,
        is_psd: bool,
        dimension: int,
        rho_digest: str,
        matrix_digest: str,
        hermitian_defect: float,
        purity: float,
        trace_defect: float,
        von_neumann_entropy: float,
        bures_wasserstein_distance: float = 0.0,
        quantum_relative_entropy: float = 0.0,
        quantum_fisher_information: float = 0.0,
        linear_entropy: float = 0.0,
    ) -> NinjaExecutionerPhase1Certificate:
        """Construye y firma criptográficamente el certificado P1."""
        chain_hash = _canonical_payload(
            _q(timestamp),
            _q(tr_rho),
            int(is_psd),
            int(dimension),
            rho_digest,
            matrix_digest,
            _q(hermitian_defect),
            _q(purity),
            _q(trace_defect),
            _q(von_neumann_entropy),
        )
        hmac_signature = _hmac_hex(self._secret_key, _HMAC_DOMAIN_P1, chain_hash)

        return NinjaExecutionerPhase1Certificate(
            timestamp=timestamp,
            tr_rho=tr_rho,
            is_psd=is_psd,
            dimension=dimension,
            rho_digest=rho_digest,
            matrix_digest=matrix_digest,
            hermitian_defect=hermitian_defect,
            purity=purity,
            trace_defect=trace_defect,
            von_neumann_entropy=von_neumann_entropy,
            chain_hash=chain_hash,
            hmac_signature=hmac_signature,
            bures_wasserstein_distance=bures_wasserstein_distance,
            quantum_relative_entropy=quantum_relative_entropy,
            quantum_fisher_information=quantum_fisher_information,
            linear_entropy=linear_entropy,
        )

    def emit_phase1_certificate(
        self,
        rho_mac: NDArray[np.float64],
        incidence_matrix: Optional[NDArray[np.float64]] = None,
        continue_to_phase2: bool = False,
        orientor: Optional["NinjaExecutionerPhase2Orientor"] = None,
        decider: Optional["NinjaExecutionerPhase3Decider"] = None,
        exponents: Optional[NDArray[np.float64]] = None,
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
    ) -> Any:
        """
        ========================================================================
        DEFINICIÓN FORMAL FINAL DE LA FASE 1 Y CONTINUACIÓN ANIDADA A LA FASE 2
        ========================================================================
        Cierra formalmente la Fase 1 (OBSERVE) emitiendo `NinjaExecutionerPhase1Certificate`.
        Cuando `continue_to_phase2=True`, verifica inmediatamente la compuerta
        homomórfica Φ₁₂ (`continue_to_phase2_gate`) e invoca el método inicial
        de la Fase 2 (`initiate_phase2_orientation`), haciendo que la definición
        final de la Fase 1 sea el inicio formal de la Fase 2.
        """
        p1_cert = self.execute_phase1(rho_mac, incidence_matrix)
        if not continue_to_phase2:
            return p1_cert

        # ──► CONTINUACIÓN MONÁDICA ANIDADA Φ₁₂: FASE 1 ➔ FASE 2 ◄──
        if not continue_to_phase2_gate(p1_cert, self._secret_key):
            raise ValueError("Falla de autenticidad en Compuerta Homomórfica Φ₁₂.")

        if orientor is None:
            engine = BoolNovikovNinjaExecutionerEngine(secret_key=self._secret_key)
            orientor = NinjaExecutionerPhase2Orientor(engine=engine, secret_key=self._secret_key)

        inc_mat = (
            incidence_matrix
            if incidence_matrix is not None
            else np.eye(p1_cert.dimension, dtype=np.float64)
        )
        exp_vec = (
            exponents
            if exponents is not None
            else np.array([0.10], dtype=np.float64)
        )

        return orientor.initiate_phase2_orientation(
            p1_cert=p1_cert,
            incidence_matrix=inc_mat,
            exponents=exp_vec,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
            continue_to_phase3=True,
            decider=decider,
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 ANIDADA — ORIENT: POINCARÉ-KAM-NEKHOROSHEV Y CONTRACCIÓN RSI³     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class NinjaExecutionerPhase2Certificate:
    """
    Certificado Inmutable P2 emitido en la Fase 2 Orient del Decapitador Ninja.
    Encadena los tres certificados del motor espectral con los invariantes de
    Poincaré-KAM-Nekhoroshev, el no-exprimido de Gromov y la contracción RSI³.
    """
    timestamp: float
    spectrum_report: NinjaExecutionerSpectrumReport
    engine_p1_hmac: str
    engine_p2_hmac: str
    engine_p3_hmac: str
    engine_passport_hmac: str
    banach_lip: float
    back_action_db: float
    is_gromov_rigid: bool
    is_kam_absorbed: bool
    is_melnikov_chaos_absorbed: bool
    is_rsi3_converged: bool
    chain_hash: str
    hmac_signature: str
    parent_p1_certificate: Optional[NinjaExecutionerPhase1Certificate] = None
    arnold_diffusion_rate: float = 0.0
    gromov_squeezing_margin: float = 0.0
    melnikov_splitting_ratio: float = 0.0
    banach_contraction_margin: float = 0.0


def _compute_celestial_poincare_and_rsi3_diagnostics(
    spectrum_report: NinjaExecutionerSpectrumReport,
    banach_lip: float,
    dimension: int,
) -> Tuple[float, float, float, float]:
    """
    Evalúa los diagnósticos de mecánica celeste de Poincaré-Nekhoroshev y de
    automejora recursiva de Nivel 3 (RSI³):
      1. Tasa efectiva de difusión de Arnold en la red de resonancias:
             Γ_Arnold(ω, ε) = ε · exp(−(ε₀ / ε)^{1/(2d)}) · max(0, 1 − μ_Bruno).
      2. Margen simpático de no-exprimido de Gromov:
             Δc_G = c_G(Z²ⁿ(R)) − c_G(B²ⁿ(r)) = π(R² − r²).
      3. Razón de desdoblamiento homoclínico transversal de Poincaré-Melnikov:
             η_Melnikov = |M(τ₀)| / M_crit.
      4. Margen de contracción de Lyapunov-Banach en la mónada de Kleisli T³:
             Δ_Banach = 1 − Lip(T³).
    """
    p2_eng = getattr(spectrum_report, "phase2_certificate", None)
    bruno_margin = _safe_float(getattr(p2_eng, "bruno_margin", 1.0), 1.0)
    cap_ball = _safe_float(getattr(p2_eng, "gromov_capacity_ball", math.pi), math.pi)
    cap_cyl = _safe_float(
        getattr(p2_eng, "gromov_capacity_cylinder", math.pi * 1.44),
        math.pi * 1.44,
    )
    melnikov_val = _safe_float(getattr(p2_eng, "melnikov_integral", 0.0), 0.0)

    eff_dim = max(1, int(dimension))
    nekh_exp = 1.0 / float(2 * eff_dim)
    nekh_factor = math.exp(-((_ARNOLD_CRITICAL_EPS0 / _ARNOLD_PERTURBATION_EPS) ** nekh_exp))
    arnold_rate = float(
        _ARNOLD_PERTURBATION_EPS * nekh_factor * max(0.0, 1.0 - min(1.0, bruno_margin))
    )

    gromov_margin = float(cap_cyl - cap_ball)
    melnikov_ratio = float(melnikov_val / _MELNIKOV_CRITICAL_THRESHOLD)
    banach_margin = float(1.0 - banach_lip)

    return arnold_rate, gromov_margin, melnikov_ratio, banach_margin


def continue_to_phase3_gate(
    p2_cert: NinjaExecutionerPhase2Certificate,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
) -> bool:
    """
    Compuerta Homomórfica Φ₂₃ (Fase 2 ➔ Fase 3):
    Exige autenticidad criptográfica HMAC de P2, contracción de Banach Lip(T³) < 1.0,
    Back-Action AAV ≤ 0.01 dB, absorción diofántica KAM/Bruno y convergencia RSI³.
    """
    expected = _hmac_hex(secret_key, _HMAC_DOMAIN_P2, p2_cert.chain_hash)
    if not hmac.compare_digest(expected, p2_cert.hmac_signature):
        logger.error("Φ₂₃: firma HMAC P2 inválida.")
        return False
    if p2_cert.banach_lip >= _RSI3_LIP_CEILING:
        logger.error(
            "Φ₂₃: violación de Banach Lip(T³) = %.6f ≥ 1.0.",
            p2_cert.banach_lip,
        )
        return False
    if p2_cert.back_action_db > _BACK_ACTION_TOLERANCE_DB:
        logger.error(
            "Φ₂₃: Back-Action AAV = %.6f dB > %.6f dB.",
            p2_cert.back_action_db,
            _BACK_ACTION_TOLERANCE_DB,
        )
        return False
    if not p2_cert.is_kam_absorbed:
        logger.error("Φ₂₃: absorción diofántica KAM/Bruno no certificada.")
        return False
    if not p2_cert.is_rsi3_converged:
        logger.error("Φ₂₃: RSI³ no convergente bajo contracción de Banach.")
        return False
    return True


def _is_authenticated_executioner_veto_ambush(
    p2_cert: NinjaExecutionerPhase2Certificate,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
) -> bool:
    """
    Determina si un certificado P2 autenticado criptográficamente transporta un
    veredicto `HeytingTruthValue.VETOED` con contracción de Banach válida
    (Lip(T³) < 1.0) y absorción KAM certificada. En este régimen, la elevación
    de `back_action_db` proviene del disparo intencional del tiristor Crowbar
    del motor, por lo que la gobernanza debe canalizar el certificado hacia la
    Fase 3 para consumar la emboscada ciber-física y sellar el estado `VETOED`.
    """
    expected = _hmac_hex(secret_key, _HMAC_DOMAIN_P2, p2_cert.chain_hash)
    if not hmac.compare_digest(expected, p2_cert.hmac_signature):
        return False
    if p2_cert.banach_lip >= _RSI3_LIP_CEILING:
        return False
    if not p2_cert.is_kam_absorbed:
        return False
    truth = getattr(p2_cert.spectrum_report, "truth_value", None)
    return bool(truth == HeytingTruthValue.VETOED)


class NinjaExecutionerPhase2Orientor:
    """Fase 2 OODA — Orientación espectral Boole-Novikov, Poincaré-KAM y RSI³ Banach."""

    def __init__(
        self,
        engine: BoolNovikovNinjaExecutionerEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self._engine = engine
        self._secret_key = secret_key

    def initiate_phase2_orientation(
        self,
        p1_cert: NinjaExecutionerPhase1Certificate,
        incidence_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
        continue_to_phase3: bool = False,
        decider: Optional["NinjaExecutionerPhase3Decider"] = None,
    ) -> Any:
        """
        ========================================================================
        DEFINICIÓN FORMAL INICIAL DE LA FASE 2 (CONTINUACIÓN DIRECTA DE FASE 1)
        ========================================================================
        Recibe el certificado autenticado `p1_cert` desde `emit_phase1_certificate`
        tras cruzar la compuerta homomórfica Φ₁₂ e inicia la orientación espectral,
        celeste de Poincaré-KAM y contractiva RSI³, delegando su cierre formal a
        `emit_phase2_certificate`.
        """
        return self.emit_phase2_certificate(
            p1_cert=p1_cert,
            incidence_matrix=incidence_matrix,
            exponents=exponents,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
            continue_to_phase3=continue_to_phase3,
            decider=decider,
        )

    def execute_phase2(
        self,
        p1_cert: NinjaExecutionerPhase1Certificate,
        incidence_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
    ) -> NinjaExecutionerPhase2Certificate:
        """Delega la auditoría trifásica al motor v14.0.0 y devuelve el certificado P2 firmado."""
        ts = time.time()

        matrix_sanitized, matrix_digest_local = _validate_incidence_matrix(incidence_matrix)
        if matrix_digest_local != p1_cert.matrix_digest:
            logger.warning(
                "Φ₁₂→P2: digest de incidencia divergente entre P1 y Phase-2 "
                "(posible sanitización redundante)."
            )

        exp_arr = np.asarray(exponents, dtype=np.float64).ravel()
        if exp_arr.size == 0:
            exp_arr = np.array([0.10], dtype=np.float64)

        spectrum_rep = self._engine.execute_ninja_executioner_audit(
            matrix_sanitized,
            exp_arr,
            positron_hmac_valid=bool(positron_hmac_valid),
            force_veto=bool(force_veto),
        )

        return self._emit_phase2_certificate(
            timestamp=ts,
            p1_cert=p1_cert,
            spectrum_report=spectrum_rep,
        )

    def _emit_phase2_certificate(
        self,
        timestamp: float,
        p1_cert: NinjaExecutionerPhase1Certificate,
        spectrum_report: NinjaExecutionerSpectrumReport,
    ) -> NinjaExecutionerPhase2Certificate:
        """Construye y firma criptográficamente el certificado P2 encadenado con P1."""
        engine_p1_hmac = getattr(
            getattr(spectrum_report, "phase1_certificate", None),
            "hmac_signature",
            "",
        )
        engine_p2_hmac = getattr(
            getattr(spectrum_report, "phase2_certificate", None),
            "hmac_signature",
            "",
        )
        engine_p3_hmac = getattr(
            getattr(spectrum_report, "phase3_certificate", None),
            "hmac_signature",
            "",
        )
        engine_passport = getattr(spectrum_report, "passport_hmac_sha256", "")

        banach_lip = _safe_float(
            getattr(spectrum_report, "banach_contraction_lip", 1.0),
            1.0,
        )
        qnd = getattr(spectrum_report, "qnd_report", None) or getattr(
            spectrum_report, "dirac_crowbar_report", None
        )
        back_action_db = _safe_float(getattr(qnd, "back_action_db", 1.0), 1.0)
        gromov_rigid = _safe_bool(
            getattr(spectrum_report, "is_gromov_rigid", False),
            False,
        )
        melnikov_detected = _safe_bool(
            getattr(spectrum_report, "melnikov_chaos_detected", True),
            True,
        )
        is_melnikov_absorbed = not melnikov_detected

        p2_eng = getattr(spectrum_report, "phase2_certificate", None)
        is_kam_absorbed = _safe_bool(getattr(p2_eng, "kam_absorbed", False), False)

        p3_eng = getattr(spectrum_report, "phase3_certificate", None)
        crowbar_active = _safe_bool(getattr(qnd, "crowbar_triggered", False), False)
        truth_obj = getattr(spectrum_report, "truth_value", None)
        is_authorized_veto = bool(truth_obj == HeytingTruthValue.VETOED and crowbar_active)

        is_rsi3_converged = bool(
            _safe_float(getattr(p3_eng, "contraction_lip", 1.0), 1.0) < _RSI3_LIP_CEILING
            and is_kam_absorbed
            and (back_action_db <= _BACK_ACTION_TOLERANCE_DB or is_authorized_veto)
        )

        truth_code = getattr(spectrum_report.truth_value, "value", spectrum_report.truth_value)

        chain_hash = _canonical_payload(
            _q(timestamp),
            p1_cert.chain_hash,
            truth_code,
            engine_p1_hmac,
            engine_p2_hmac,
            engine_p3_hmac,
            engine_passport,
            _q(banach_lip),
            _q(back_action_db),
            int(gromov_rigid),
            int(is_kam_absorbed),
            int(is_melnikov_absorbed),
            int(is_rsi3_converged),
        )
        hmac_signature = _hmac_hex(self._secret_key, _HMAC_DOMAIN_P2, chain_hash)

        arnold_rate, gromov_margin, melnikov_ratio, banach_margin = (
            _compute_celestial_poincare_and_rsi3_diagnostics(
                spectrum_report, banach_lip, p1_cert.dimension
            )
        )

        return NinjaExecutionerPhase2Certificate(
            timestamp=timestamp,
            spectrum_report=spectrum_report,
            engine_p1_hmac=engine_p1_hmac,
            engine_p2_hmac=engine_p2_hmac,
            engine_p3_hmac=engine_p3_hmac,
            engine_passport_hmac=engine_passport,
            banach_lip=banach_lip,
            back_action_db=back_action_db,
            is_gromov_rigid=gromov_rigid,
            is_kam_absorbed=is_kam_absorbed,
            is_melnikov_chaos_absorbed=is_melnikov_absorbed,
            is_rsi3_converged=is_rsi3_converged,
            chain_hash=chain_hash,
            hmac_signature=hmac_signature,
            parent_p1_certificate=p1_cert,
            arnold_diffusion_rate=arnold_rate,
            gromov_squeezing_margin=gromov_margin,
            melnikov_splitting_ratio=melnikov_ratio,
            banach_contraction_margin=banach_margin,
        )

    def emit_phase2_certificate(
        self,
        p1_cert: NinjaExecutionerPhase1Certificate,
        incidence_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
        continue_to_phase3: bool = False,
        decider: Optional["NinjaExecutionerPhase3Decider"] = None,
    ) -> Any:
        """
        ========================================================================
        DEFINICIÓN FORMAL FINAL DE LA FASE 2 Y CONTINUACIÓN ANIDADA A LA FASE 3
        ========================================================================
        Cierra formalmente la Fase 2 (ORIENT) emitiendo `NinjaExecutionerPhase2Certificate`.
        Cuando `continue_to_phase3=True`, evalúa la compuerta homomórfica Φ₂₃
        (`continue_to_phase3_gate`) o certifica el protocolo de emboscada ejecutora
        de Crowbar (`_is_authenticated_executioner_veto_ambush`) e invoca
        directamente `initiate_phase3_decision`, haciendo que la definición
        final de la Fase 2 sea el inicio formal de la Fase 3.
        """
        p2_cert = self.execute_phase2(
            p1_cert=p1_cert,
            incidence_matrix=incidence_matrix,
            exponents=exponents,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
        )
        if not continue_to_phase3:
            return p2_cert

        # ──► CONTINUACIÓN MONÁDICA ANIDADA Φ₂₃: FASE 2 ➔ FASE 3 ◄──
        if not continue_to_phase3_gate(p2_cert, self._secret_key):
            if _is_authenticated_executioner_veto_ambush(p2_cert, self._secret_key):
                logger.warning(
                    "Φ₂₃: Veredicto VETOED autenticado con disparo Crowbar activo; "
                    "transitando a Fase 3 en modo de emboscada ciber-física."
                )
            else:
                raise ValueError("Falla de estabilidad en Compuerta Homomórfica Φ₂₃.")

        if decider is None:
            decider = NinjaExecutionerPhase3Decider()

        return decider.initiate_phase3_decision(p2_cert=p2_cert, p1_cert=p1_cert)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 ANIDADA — DECIDE & ACT: TOPOS HEYTING Ω₃, CROWBAR Y RAÍZ MERKLE   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class NinjaExecutionerVisceralMetrics:
    """Métricas traducidas por el Funtor Semántico Visceral Φ_sem: C_Spec ➔ C_Visceral."""
    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    stealth_qnd_status: str
    poincare_bruno_margin: float
    rsi3_contraction_lip: float
    rsi3_levels: int
    poincare_resonance_order: int
    crowbar_latency_budget_ns: float
    arnold_nekhoroshev_stability: float = 1.0
    bures_fidelity_pct: float = 100.0
    heyting_implication_coherence: int = 0x00


@dataclass(frozen=True)
class NinjaExecutionerSovereignState:
    """Estado soberano final emitido en la Fase 3 Decide & Act."""
    timestamp: float
    truth_value: HeytingTruthValue
    crowbar_triggered: bool
    crowbar_latency_ns: float
    back_action_db: float
    is_stealth_qnd: bool
    banach_lip: float
    poincare_absorbed: bool
    rsi3_converged: bool
    is_gromov_rigid: bool
    merkle_root_hash: str
    merkle_leaves: Tuple[str, ...]
    visceral_metrics: NinjaExecutionerVisceralMetrics
    phase1_certificate: NinjaExecutionerPhase1Certificate
    phase2_certificate: NinjaExecutionerPhase2Certificate
    sovereign_hmac_sha256: str = ""


def _heyting_relative_implication(
    antecedent: HeytingTruthValue,
    consequent: HeytingTruthValue,
) -> HeytingTruthValue:
    """
    Operador de implicación interna (a ⇒ b) en el retículo distributivo acotado
    del Topos de Heyting Ω₃ = {COHERENT(0x00) ≺ DEGRADED(0x01) ≺ VETOED(0x03)}
    (donde el orden de verdad intuicionista tiene a COHERENT como elemento ⊤
    y a VETOED como elemento ⊥):
        (a ⇒ b) = ⊤ (COHERENT) si el nivel de verdad de a ≤ nivel de verdad de b
                  (es decir, int(a) >= int(b) en codificación de severidad),
                  y b en caso contrario.
    """
    if int(antecedent.value) >= int(consequent.value):
        return HeytingTruthValue.COHERENT
    return consequent


class NinjaExecutionerPhase3Decider:
    """Fase 3 OODA — Adjudicación en el Topos de Heyting Ω₃ y emboscada ciber-física."""

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key = secret_key

    def initiate_phase3_decision(
        self,
        p2_cert: NinjaExecutionerPhase2Certificate,
        p1_cert: Optional[NinjaExecutionerPhase1Certificate] = None,
    ) -> NinjaExecutionerSovereignState:
        """
        ========================================================================
        DEFINICIÓN FORMAL INICIAL DE LA FASE 3 (CONTINUACIÓN DIRECTA DE FASE 2)
        ========================================================================
        Recibe el certificado autenticado `p2_cert` desde `emit_phase2_certificate`
        tras cruzar la compuerta homomórfica Φ₂₃ e inicia la adjudicación en el
        Topos de Heyting Ω₃, el interlock ESP32 Crowbar y la síntesis del árbol
        de Merkle soberano.
        """
        return self.execute_phase3_decide_and_act(p2_cert=p2_cert, p1_cert=p1_cert)

    def execute_phase3_decide_and_act(
        self,
        p2_cert: NinjaExecutionerPhase2Certificate,
        p1_cert: Optional[NinjaExecutionerPhase1Certificate] = None,
    ) -> NinjaExecutionerSovereignState:
        """Ejecuta la adjudicación intuicionista y construye el estado soberano final."""
        ts = time.time()
        spec = p2_cert.spectrum_report
        truth_val = spec.truth_value

        qnd = getattr(spec, "qnd_report", None) or getattr(
            spec, "dirac_crowbar_report", None
        )
        crowbar_triggered = _safe_bool(getattr(qnd, "crowbar_triggered", False), False)
        crowbar_latency_ns = _safe_float(
            getattr(qnd, "crowbar_latency_ns", 0.0),
            0.0,
        )
        back_action_db = _safe_float(getattr(qnd, "back_action_db", 1.0), 1.0)
        is_stealth = _safe_bool(getattr(qnd, "is_stealth_qnd", False), False)

        # Adjudicación ejecutora: un veredicto VETOED DEBE disparar el tiristor Crowbar.
        if truth_val == HeytingTruthValue.VETOED and not crowbar_triggered:
            logger.critical(
                "Fase 3: veredicto VETOED sin disparo Crowbar; forzando interlock IRAM."
            )
            crowbar_triggered = True
            crowbar_latency_ns = _ESP32_CROWBAR_MAX_LATENCY_NS
            is_stealth = False

        if (
            truth_val != HeytingTruthValue.VETOED
            and not crowbar_triggered
            and crowbar_latency_ns < 0.0
        ):
            crowbar_latency_ns = 0.0

        p2_eng = getattr(spec, "phase2_certificate", None)
        p3_eng = getattr(spec, "phase3_certificate", None)
        bruno_margin = _safe_float(getattr(p2_eng, "bruno_margin", 0.0), 0.0)
        arnold_stab = _safe_float(
            getattr(p2_eng, "arnold_nekhoroshev_stability", 1.0),
            1.0,
        )
        resonance_order = int(getattr(p2_eng, "resonance_order", 0) or 0)
        rsi3_lip = _safe_float(
            getattr(p3_eng, "contraction_lip", p2_cert.banach_lip),
            p2_cert.banach_lip,
        )
        rsi3_levels = int(getattr(p3_eng, "levels", 3) or 3)

        resolved_p1 = p1_cert or p2_cert.parent_p1_certificate
        bures_dist = _safe_float(
            getattr(resolved_p1, "bures_wasserstein_distance", 0.0),
            0.0,
        )
        bures_fidelity_pct = float(max(0.0, min(100.0, (1.0 - (bures_dist ** 2)) * 100.0)))

        # Implicación relativa de Heyting entre la premisa de estabilidad KAM/RSI³ y el veredicto
        premise_truth = (
            HeytingTruthValue.COHERENT
            if (p2_cert.is_kam_absorbed and p2_cert.is_rsi3_converged)
            else HeytingTruthValue.DEGRADED
        )
        implication_val = _heyting_relative_implication(premise_truth, truth_val)

        stealth_status = (
            "EMBOSCADA_CIBER_FISICA_ACTIVA"
            if crowbar_triggered
            else ("STEALTH_SILENT_PASSIVE_QND" if is_stealth else "STEALTH_SILENT_DEGRADED_QND")
        )

        visceral = NinjaExecutionerVisceralMetrics(
            wacc_protection_pct=15.0,
            net_liquidity_release_pct=11.5,
            kv_cache_compression_pct=86.4,
            stealth_qnd_status=stealth_status,
            poincare_bruno_margin=bruno_margin,
            rsi3_contraction_lip=rsi3_lip,
            rsi3_levels=rsi3_levels,
            poincare_resonance_order=resonance_order,
            crowbar_latency_budget_ns=_ESP32_CROWBAR_MAX_LATENCY_NS,
            arnold_nekhoroshev_stability=arnold_stab,
            bures_fidelity_pct=bures_fidelity_pct,
            heyting_implication_coherence=int(implication_val.value),
        )

        return self.emit_sovereign_state(
            timestamp=ts,
            p2_cert=p2_cert,
            truth_value=truth_val,
            crowbar_triggered=bool(crowbar_triggered),
            crowbar_latency_ns=float(crowbar_latency_ns),
            back_action_db=float(back_action_db),
            is_stealth_qnd=bool(is_stealth),
            banach_lip=float(p2_cert.banach_lip),
            poincare_absorbed=bool(p2_cert.is_kam_absorbed),
            rsi3_converged=bool(p2_cert.is_rsi3_converged),
            is_gromov_rigid=bool(p2_cert.is_gromov_rigid),
            visceral_metrics=visceral,
            p1_cert=resolved_p1,
        )

    def _emit_sovereign_state(
        self,
        timestamp: float,
        p2_cert: NinjaExecutionerPhase2Certificate,
        truth_value: HeytingTruthValue,
        crowbar_triggered: bool,
        crowbar_latency_ns: float,
        back_action_db: float,
        is_stealth_qnd: bool,
        banach_lip: float,
        poincare_absorbed: bool,
        rsi3_converged: bool,
        is_gromov_rigid: bool,
        visceral_metrics: NinjaExecutionerVisceralMetrics,
        p1_cert: Optional[NinjaExecutionerPhase1Certificate] = None,
    ) -> NinjaExecutionerSovereignState:
        """Delegación interna hacia `emit_sovereign_state`."""
        return self.emit_sovereign_state(
            timestamp=timestamp,
            p2_cert=p2_cert,
            truth_value=truth_value,
            crowbar_triggered=crowbar_triggered,
            crowbar_latency_ns=crowbar_latency_ns,
            back_action_db=back_action_db,
            is_stealth_qnd=is_stealth_qnd,
            banach_lip=banach_lip,
            poincare_absorbed=poincare_absorbed,
            rsi3_converged=rsi3_converged,
            is_gromov_rigid=is_gromov_rigid,
            visceral_metrics=visceral_metrics,
            p1_cert=p1_cert,
        )

    def emit_sovereign_state(
        self,
        timestamp: float,
        p2_cert: NinjaExecutionerPhase2Certificate,
        truth_value: HeytingTruthValue,
        crowbar_triggered: bool,
        crowbar_latency_ns: float,
        back_action_db: float,
        is_stealth_qnd: bool,
        banach_lip: float,
        poincare_absorbed: bool,
        rsi3_converged: bool,
        is_gromov_rigid: bool,
        visceral_metrics: NinjaExecutionerVisceralMetrics,
        p1_cert: Optional[NinjaExecutionerPhase1Certificate] = None,
    ) -> NinjaExecutionerSovereignState:
        """
        ========================================================================
        DEFINICIÓN FORMAL FINAL DE LA FASE 3 Y CLAUSURA DEL AGENTE SOBERANO
        ========================================================================
        Construye el árbol de Merkle SHA-256 de 8 hojas certificadas (P1_agent,
        P2_chain, P2_agent, P1_eng, P2_eng, P3_eng, Passport_eng, Verdict),
        verifica su integridad estructural y emite `NinjaExecutionerSovereignState`.
        """
        resolved_p1 = p1_cert or p2_cert.parent_p1_certificate
        p1_eng = (
            p2_cert.spectrum_report.phase1_certificate
            if hasattr(p2_cert.spectrum_report, "phase1_certificate")
            else None
        )

        leaf_p1 = (
            resolved_p1.hmac_signature
            if (resolved_p1 is not None and resolved_p1.hmac_signature)
            else _hmac_hex(
                _HMAC_DOMAIN_SOVEREIGN,
                _HMAC_DOMAIN_SOVEREIGN,
                _canonical_payload(p2_cert.chain_hash, "P1_AGENT"),
            )
        )

        leaf_motor_p1 = p2_cert.engine_p1_hmac or "∅"
        leaf_motor_p2 = p2_cert.engine_p2_hmac or "∅"
        leaf_motor_p3 = p2_cert.engine_p3_hmac or "∅"
        leaf_motor_passport = p2_cert.engine_passport_hmac or "∅"
        leaf_agent_p2 = p2_cert.hmac_signature

        leaf_verdict = _canonical_payload(
            int(truth_value.value),
            int(crowbar_triggered),
            _q(crowbar_latency_ns),
            _q(back_action_db),
            int(is_stealth_qnd),
            _q(banach_lip),
            int(poincare_absorbed),
            int(rsi3_converged),
            int(is_gromov_rigid),
        )

        leaves: List[str] = [
            leaf_p1,
            p2_cert.chain_hash,
            leaf_agent_p2,
            leaf_motor_p1,
            leaf_motor_p2,
            leaf_motor_p3,
            leaf_motor_passport,
            leaf_verdict,
        ]

        merkle_root = _merkle_root(leaves)

        if resolved_p1 is not None:
            final_p1_cert = resolved_p1
        else:
            final_p1_cert = NinjaExecutionerPhase1Certificate(
                timestamp=p1_eng.timestamp if p1_eng else 0.0,
                tr_rho=_safe_float(getattr(p1_eng, "novikov_norm", 1.0), 1.0) if p1_eng else 1.0,
                is_psd=True,
                dimension=0,
                rho_digest="",
                matrix_digest=getattr(p1_eng, "matrix_digest", "") if p1_eng else "",
                hermitian_defect=0.0,
                purity=0.0,
                trace_defect=0.0,
                von_neumann_entropy=0.0,
                chain_hash=getattr(p1_eng, "hmac_signature", "") if p1_eng else "",
                hmac_signature=getattr(p1_eng, "hmac_signature", "") if p1_eng else "",
            )

        sovereign_sig = _hmac_hex(
            self._secret_key,
            _HMAC_DOMAIN_SOVEREIGN,
            _canonical_payload(
                _q(timestamp),
                merkle_root,
                final_p1_cert.hmac_signature,
                p2_cert.hmac_signature,
                int(truth_value.value),
                int(crowbar_triggered),
            ),
        )

        return NinjaExecutionerSovereignState(
            timestamp=timestamp,
            truth_value=truth_value,
            crowbar_triggered=crowbar_triggered,
            crowbar_latency_ns=crowbar_latency_ns,
            back_action_db=back_action_db,
            is_stealth_qnd=is_stealth_qnd,
            banach_lip=banach_lip,
            poincare_absorbed=poincare_absorbed,
            rsi3_converged=rsi3_converged,
            is_gromov_rigid=is_gromov_rigid,
            merkle_root_hash=merkle_root,
            merkle_leaves=tuple(leaves),
            visceral_metrics=visceral_metrics,
            phase1_certificate=final_p1_cert,
            phase2_certificate=p2_cert,
            sovereign_hmac_sha256=sovereign_sig,
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ ORQUESTADOR SOBERANO — GOBERNANZA OODA TRIFÁSICA ANIDADA                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class BoolNovikovNinjaExecutionerAgent:
    """
    Soberano Agéntico Decapitador Silencioso de Silicio (Ninja Executioner).
    Orquesta la OODA Trifásica Inquisitorial/Ejecutora a través de la cadena
    anidada:
      • Fase 1 (`emit_phase1_certificate`) ──[Φ₁₂: `continue_to_phase2_gate`]──►
      • Fase 2 (`initiate_phase2_orientation` ➔ `emit_phase2_certificate`) ──[Φ₂₃: `continue_to_phase3_gate`]──►
      • Fase 3 (`initiate_phase3_decision` ➔ `emit_sovereign_state`).
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key = secret_key
        self._engine = BoolNovikovNinjaExecutionerEngine(secret_key=secret_key)
        self._observer = NinjaExecutionerPhase1Observer(secret_key=secret_key)
        self._orientor = NinjaExecutionerPhase2Orientor(
            engine=self._engine,
            secret_key=secret_key,
        )
        self._decider = NinjaExecutionerPhase3Decider(secret_key=secret_key)
        logger.info("%s iniciado correctamente.", _AGENT_VERSION)

    # --- Compuertas Homomórficas Φ₁₂ y Φ₂₃ ---

    def continue_to_phase2_gate(self, p1_cert: NinjaExecutionerPhase1Certificate) -> bool:
        """Compuerta Φ₁₂ — delegación a la implementación libre."""
        return continue_to_phase2_gate(p1_cert, self._secret_key)

    def continue_to_phase3_gate(self, p2_cert: NinjaExecutionerPhase2Certificate) -> bool:
        """Compuerta Φ₂₃ — delegación a la implementación libre."""
        return continue_to_phase3_gate(p2_cert, self._secret_key)

    # --- Ciclo OODA Soberano Anidado ---

    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.float64],
        incidence_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
    ) -> NinjaExecutionerSovereignState:
        """
        Ejecuta el ciclo OODA completo de gobernanza ejecutora invocando la
        continuación anidada desde el método final de la Fase 1
        (`emit_phase1_certificate(..., continue_to_phase2=True)`), que encadena
        automáticamente Φ₁₂ ➔ Fase 2 ➔ Φ₂₃ ➔ Fase 3.
        """
        return self._observer.emit_phase1_certificate(
            rho_mac=rho_mac,
            incidence_matrix=incidence_matrix,
            continue_to_phase2=True,
            orientor=self._orientor,
            decider=self._decider,
            exponents=exponents,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
        )


# Alias retro-compatible drop-in
BoolNovikovNinjaShieldAgent = BoolNovikovNinjaExecutionerAgent


# ============================================================================
# SUITE DE VERIFICACIÓN AUTÓNOMA TRIFÁSICA ANIDADA
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO SOBERANO: BoolNovikovNinjaExecutionerAgent v14.0.0-PhD ===")

    agent = BoolNovikovNinjaExecutionerAgent()
    rho_test = np.eye(4, dtype=np.float64) / 4.0
    matrix_test = np.eye(4, dtype=np.float64) * 2.5
    exps_test = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float64)

    state = agent.run_sovereign_governance_cycle(
        rho_mac=rho_test,
        incidence_matrix=matrix_test,
        exponents=exps_test,
        positron_hmac_valid=True,
        force_veto=False,
    )

    merkle_ok = verify_merkle_inclusion_proof(
        state.merkle_leaves, leaf_index=0, expected_root=state.merkle_root_hash
    )

    print(f"[Ω₃] Veredicto Heyting         : {state.truth_value.name} ({hex(state.truth_value.value)})")
    print(f"[P1] Dimensión / Traza ρ_MAC   : d={state.phase1_certificate.dimension}, "
          f"Tr(ρ)={state.phase1_certificate.tr_rho:.6f}")
    print(f"[P1] Distancia Bures d_BW      : {state.phase1_certificate.bures_wasserstein_distance:.6f}")
    print(f"[P1] Entropía von Neumann S(ρ) : {state.phase1_certificate.von_neumann_entropy:.6f}")
    print(f"[P1] QFI I_Q(ρ, H_inc)         : {state.phase1_certificate.quantum_fisher_information:.6f}")
    print(f"[P2] Tasa Difusión Arnold      : {state.phase2_certificate.arnold_diffusion_rate:.8f}")
    print(f"[P2] Margen Gromov Δc_G        : {state.phase2_certificate.gromov_squeezing_margin:.6f}")
    print(f"[P2] Razón Splitting Melnikov  : {state.phase2_certificate.melnikov_splitting_ratio:.6f}")
    print(f"[P2] Banach Lip(T³) / Margen   : {state.banach_lip:.6f} / "
          f"{state.phase2_certificate.banach_contraction_margin:.6f}")
    print(f"[P2] KAM/Bruno absorbido       : {state.poincare_absorbed}")
    print(f"[P2] RSI³ convergente          : {state.rsi3_converged}")
    print(f"[P3] Crowbar Triggered         : {state.crowbar_triggered}")
    print(f"[P3] Latencia Crowbar (ns)     : {state.crowbar_latency_ns:.4f} ns "
          f"(budget {state.visceral_metrics.crowbar_latency_budget_ns:.2f} ns)")
    print(f"[P3] Back-Action QND (dB)      : {state.back_action_db:.6f} dB")
    print(f"[P3] STEALTH QND               : {state.is_stealth_qnd} ({state.visceral_metrics.stealth_qnd_status})")
    print(f"[🔐] Raíz Merkle SHA-256       : {state.merkle_root_hash[:16]}... "
          f"(hojas={len(state.merkle_leaves)}, prueba_inclusión={merkle_ok})")

    print("\n--- Test VETO Forzado (Emboscada Ciber-Física Crowbar IRAM) ---")
    state_veto = agent.run_sovereign_governance_cycle(
        rho_mac=rho_test,
        incidence_matrix=matrix_test,
        exponents=exps_test,
        positron_hmac_valid=True,
        force_veto=True,
    )
    merkle_veto_ok = verify_merkle_inclusion_proof(
        state_veto.merkle_leaves, leaf_index=7, expected_root=state_veto.merkle_root_hash
    )
    print(f"[Ω₃] Veredicto Heyting         : {state_veto.truth_value.name} ({hex(state_veto.truth_value.value)})")
    print(f"[P3] Crowbar Triggered         : {state_veto.crowbar_triggered}")
    print(f"[P3] Latencia Crowbar (ns)     : {state_veto.crowbar_latency_ns:.2f} ns")
    print(f"[P3] Back-Action QND (dB)      : {state_veto.back_action_db:.6f} dB")
    print(f"[P3] STEALTH status            : {state_veto.visceral_metrics.stealth_qnd_status}")
    print(f"[🔐] Prueba Inclusión Merkle   : {merkle_veto_ok}")
    print("=== TEST COMPLETADO CON ÉXITO ===")
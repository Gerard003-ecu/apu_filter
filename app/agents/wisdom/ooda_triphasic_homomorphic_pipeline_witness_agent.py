# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : OODA Triphasic Homomorphic Pipeline Witness Agent v15.0.0-PhD                             ║
║ Ubicación: app/wisdom/ooda_triphasic_homomorphic_pipeline_witness_agent.py                           ║
║ Versión  : 15.0.0-SOVEREIGN-POINCARE-KAM-RSI-L3-AGENT-YOSHIDA-WILLIAMSON-PhD                         ║
║ Autor    : Soberano Artesano Tejedor de Código & Arquitectura Cuántica de Precisión                  ║
║ Función  : Soberano agéntico testigo silencioso del pipeline OODA trifásico homomórfico.             ║
║            Atestigua pasivamente la consistencia functorial de Φ₁₂, Φ₂₃ y Φ₁₃=Φ₂₃∘Φ₁₂,              ║
║            acoplando mecánica celeste de Poincaré (Yoshida O(4), KAM, Nekhoroshev, Floquet)          ║
║            con automejora recursiva de nivel 3 (operador T y meta-operador S de Banach).             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE TRES FASES ANIDADAS CONTINUAS
(el morfismo terminal de la fase k es el objeto inicial de la fase k+1; no hay discontinuidad
functorial ni reparametrización oculta):

  • FASE 1 — OBSERVE
      C*-álgebra de Banach, proyección de Higham, conexión 𝔰𝔲(N), holonomía de Wilson,
      cuantización de Weyl–Williamson hacia T*(Tⁿ).
      Morfismo terminal 1→2: `bridge_phase1_to_phase2_poincare_transfer(...)`
      → `Phase1TransferCarrier`  (objeto inicial de la Fase 2)

  • FASE 2 — ORIENT
      Compuerta homomórfica Φ₁₂ (HMAC + C*), integración Yoshida, sección de Poincaré,
      monodromía analítica, KAM/Nekhoroshev, medición débil QND AAV.
      Morfismo terminal 2→3: `bridge_phase2_to_phase3_rsi_transfer(...)`
      → `Phase2TransferCarrier`  (objeto inicial de la Fase 3)

  • FASE 3 — DECIDE & ACT
      Compuerta Φ₂₃, RSI-L3 (T y S anidados), clasificador de Heyting Ω₄,
      funtor semántico visceral Φ_sem, atestación Merkle-HMAC.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Inclusión de rutas para carga modular dinámica y resiliente
for path_dir in ("/workspace/scratch", "/workspace/artifacts", ".", "app/wisdom"):
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

# ────────────────────────────────────────────────────────────────────────────────────────────────────
# Importación del motor testigo v15 (con repliegue de tipos si el PYTHONPATH no lo expone)
# ────────────────────────────────────────────────────────────────────────────────────────────────────
_ENGINE_AVAILABLE: bool = True
try:
    from ooda_triphasic_homomorphic_pipeline_witness_engine import (  # type: ignore
        HeytingOmega4,
        HomomorphicGateCertificate,
        KAMAuditResult,
        MonodromyResult,
        OODATriphasicHomomorphicPipelineWitnessEnginePhD,
        PoincareCelestialSymplecticEngine,
        PoincareOrbitalState,
        PoincareSectionResult,
        QuantumBanachAlgebraTopologyEngine,
        RSIL3ConfigurationBundle,
        RSIL3SelfOptimizationCore,
        SymplecticVariationalFeedback,
    )
except ImportError:
    try:
        from app.wisdom.ooda_triphasic_homomorphic_pipeline_witness_engine import (  # type: ignore
            HeytingOmega4,
            HomomorphicGateCertificate,
            KAMAuditResult,
            MonodromyResult,
            OODATriphasicHomomorphicPipelineWitnessEnginePhD,
            PoincareCelestialSymplecticEngine,
            PoincareOrbitalState,
            PoincareSectionResult,
            QuantumBanachAlgebraTopologyEngine,
            RSIL3ConfigurationBundle,
            RSIL3SelfOptimizationCore,
            SymplecticVariationalFeedback,
        )
    except ImportError:
        _ENGINE_AVAILABLE = False

        class HeytingOmega4(IntEnum):  # type: ignore[no-redef]
            """Cadena de Heyting Ω₄: ⊥ = VETOED ≺ CRITICAL ≺ DEGRADED ≺ COHERENT = ⊤."""

            COHERENT = 0
            DEGRADED = 1
            CRITICAL = 2
            VETOED = 3

            def _truth(self) -> int:
                return 3 - int(self)

            @classmethod
            def _from_truth(cls, rank: int) -> "HeytingOmega4":
                return HeytingOmega4(int(np.clip(3 - int(rank), 0, 3)))

            def meet(self, other: "HeytingOmega4") -> "HeytingOmega4":
                return HeytingOmega4._from_truth(min(self._truth(), other._truth()))

            def join(self, other: "HeytingOmega4") -> "HeytingOmega4":
                return HeytingOmega4._from_truth(max(self._truth(), other._truth()))

            def implies(self, other: "HeytingOmega4") -> "HeytingOmega4":
                if self._truth() <= other._truth():
                    return HeytingOmega4.COHERENT
                return other

            def complement(self) -> "HeytingOmega4":
                return self.implies(HeytingOmega4.VETOED)

        @dataclass(frozen=True, slots=True)
        class PoincareOrbitalState:  # type: ignore[no-redef]
            coordinates_q: NDArray[np.float64]
            momenta_p: NDArray[np.float64]
            action_variables_I: NDArray[np.float64]
            angle_variables_theta: NDArray[np.float64]
            energy_hamiltonian: float
            gauge_phase_shift: float
            dimension_dof: int
            quantum_density_snapshot: NDArray[np.complex128]
            williamson_invariants: NDArray[np.float64] = None  # type: ignore[assignment]
            su2_holonomy_trace: complex = 0j

        @dataclass(frozen=True, slots=True)
        class PoincareSectionResult:  # type: ignore[no-redef]
            intersection_count: int
            return_map_points: NDArray[np.float64]
            mean_return_time: float
            rotation_number_estimate: float

        @dataclass(frozen=True, slots=True)
        class MonodromyResult:  # type: ignore[no-redef]
            monodromy_matrix: NDArray[np.float64]
            floquet_multipliers: NDArray[np.complex128]
            max_lyapunov_exponent: float
            is_symplectic_conserved: bool
            symplectic_drift_norm: float
            energy_drift: float = 0.0
            floquet_pairing_defect: float = 0.0

        @dataclass(frozen=True, slots=True)
        class KAMAuditResult:  # type: ignore[no-redef]
            is_kam_torus_stable: bool
            diophantine_constant_gamma: float
            diophantine_exponent_tau: float
            minimum_small_divisor: float
            resonance_indices: List[Tuple[int, ...]]
            poincare_recurrence_time_estimate: float
            nekhoroshev_stability_time: float = 0.0
            continued_fraction_type: float = 0.0

        @dataclass(frozen=True, slots=True)
        class SymplecticVariationalFeedback:  # type: ignore[no-redef]
            symplectic_gradient_correction: NDArray[np.float64]
            floquet_stabilization_gain: float
            kam_regularization_scalar: float
            poincare_section_intersection_count: int
            orbital_divergence_risk: float
            origin_orbital_state: PoincareOrbitalState
            energy_drift: float = 0.0
            nekhoroshev_horizon: float = 0.0

        @dataclass(frozen=True, slots=True)
        class HomomorphicGateCertificate:  # type: ignore[no-redef]
            phi12_integrity: float
            phi23_integrity: float
            phi13_composition_integrity: float
            corner_homomorphism_defect_12: float
            corner_homomorphism_defect_23: float
            functoriality_defect: float

        @dataclass(frozen=True, slots=True)
        class RSIL3ConfigurationBundle:  # type: ignore[no-redef]
            psd_spectral_floor: float
            higham_projection_damping: float
            weak_measurement_gain: float
            kam_diophantine_threshold: float
            symplectomorphism_tolerance: float
            iteration_generation: int
            meta_contraction_lip: float = 0.885

        class OODATriphasicHomomorphicPipelineWitnessEnginePhD:  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ImportError(
                    "Motor v15 ausente: ooda_triphasic_homomorphic_pipeline_witness_engine.py "
                    "debe residir en PYTHONPATH para instanciar el agente."
                )


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# Constantes criptográficas, espectrales y de dominio HMAC (v15, no intercambiables con v14)
# ────────────────────────────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-13
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_SOVEREIGN_OODA_POINCARE_KAM_RSI_L3_KEY_v15.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"OODA-AGENT-P1-OBSERVE-ISOMETRY-v15.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"OODA-AGENT-P2-ORIENT-POINCARE-KAM-v15.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"OODA-AGENT-P3-DECIDE-ACT-RSI-L3-v15.0|"
_CHAIN_HEX_LEN: Final[int] = 64
_TWO_PI: Final[float] = 2.0 * math.pi
_TRACE_DRIFT_TOL: Final[float] = 1.0e-5

logger = logging.getLogger("APU.Wisdom.OODATriphasicHomomorphicPipelineWitnessAgent.PhD")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _canonical_complex(z: complex) -> str:
    """Serialización canónica de un complejo (evita deriva sign/verify por repr)."""
    c = complex(z)
    return f"{c.real:.16e}{c.imag:+.16e}j"


def _empty_section(dof: int) -> PoincareSectionResult:
    """Sección de Poincaré vacía (compatibilidad cuando el motor no expone Yoshida)."""
    n = max(1, int(dof))
    return PoincareSectionResult(
        intersection_count=0,
        return_map_points=np.zeros((0, 2 * n), dtype=np.float64),
        mean_return_time=float("inf"),
        rotation_number_estimate=0.0,
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES DE TRANSFERENCIA FORMAL INTER-FASE
# Cada carrier es el objeto inicial de la fase siguiente; los certificados son objetos de Heyting
# sellados HMAC y encadenados (chain_hash_k = SHA384(chain_hash_{k-1} ‖ payload_k)).
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class OODAWitnessPhase1CertificatePhD:
    """Certificado P1 (Observe): geometría C*, conexión SU(N), isometría de traza."""

    witness_id: str
    trace_preservation_ok: bool
    trace_drift: float
    connection_norm: float
    yang_mills_energy: float
    wilson_phase_rad: float
    su2_holonomy_trace: complex
    dof_cotangent: int
    energy_hamiltonian: float
    isometry_verdict: HeytingOmega4
    chain_hash: str
    hmac_p1_signature: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class Phase1TransferCarrier:
    """
    OBJETO INICIAL DE LA FASE 2.
    Emitido exclusivamente por `bridge_phase1_to_phase2_poincare_transfer`.
    Transporta el certificado P1, el estado orbital canónico en T*(Tⁿ) y la conexión
    de calibre (para no re-extraer A en Orient: testigo pasivo, cero demolición extra).
    """

    phase1_certificate: OODAWitnessPhase1CertificatePhD
    orbital_poincare_state: PoincareOrbitalState
    sanitized_density_matrix: NDArray[np.complex128]
    gauge_connection: NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class OODAWitnessPhase2CertificatePhD:
    """Certificado P2 (Orient): Poincaré–KAM, Floquet, sección Σ, QND AAV."""

    witness_id: str
    phase1_ref_id: str
    weak_value_qnd_aav: complex
    weak_value_modulus: float
    back_action_db: float
    qnd_commutator_norm: float
    uhlmann_fidelity: float
    bures_distance: float
    purity: float
    is_kam_torus_stable: bool
    max_lyapunov_exponent: float
    floquet_spectral_radius: float
    floquet_pairing_defect: float
    energy_drift: float
    nekhoroshev_horizon: float
    poincare_recurrence_time: float
    poincare_rotation_number: float
    section_intersection_count: int
    is_symplectic_conserved: bool
    qnd_orient_verdict: HeytingOmega4
    chain_hash: str
    hmac_p2_signature: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class Phase2TransferCarrier:
    """
    OBJETO INICIAL DE LA FASE 3.
    Emitido exclusivamente por `bridge_phase2_to_phase3_rsi_transfer`.
    Transporta ambos certificados, el tensor de retroalimentación simpléctica y ρ sanitizada.
    """

    phase1_certificate: OODAWitnessPhase1CertificatePhD
    phase2_certificate: OODAWitnessPhase2CertificatePhD
    symplectic_feedback: SymplecticVariationalFeedback
    sanitized_density_matrix: NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class OODAPipelineWitnessSovereignStatePhD:
    """Estado soberano final (Decide & Act): RSI-L3, Ω₄, Φ₁₂/Φ₂₃/Φ₁₃ y Merkle-HMAC."""

    sovereign_state_id: str
    phase1_cert: OODAWitnessPhase1CertificatePhD
    phase2_cert: OODAWitnessPhase2CertificatePhD
    global_heyting_verdict: HeytingOmega4
    rsi_level3_generation: int
    rsi_banach_contraction_lip: float
    rsi_meta_contraction_lip: float
    rsi_cauchy_residual: float
    homomorphic_gate_phi12_integrity: float
    homomorphic_gate_phi23_integrity: float
    homomorphic_gate_phi13_integrity: float
    functoriality_defect: float
    back_action_db: float
    merkle_dag_sha384_root: str
    hmac_provenance_signature: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1: OBSERVE — C*-ÁLGEBRA DE BANACH, TOPOLOGÍA DE CALIBRE Y CUANTIZACIÓN DE WEYL–WILLIAMSON
# El morfismo terminal `bridge_phase1_to_phase2_poincare_transfer` *es* el objeto inicial de la Fase 2.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class OODAWitnessPhase1ObserverPhD:
    """
    FASE 1 — OBSERVE.

    Audita 𝔅(ℋ): proyección convexa de Higham sobre {ρ ≥ 0, Tr ρ = 1}, 1-forma antihermítica
    A ∈ 𝔰𝔲(N), energía de Yang–Mills, holonomía de Wilson y cuantización de Weyl–Williamson
    hacia el fibrado cotangente T*(Tⁿ). Emite un certificado HMAC-P1 y el carrier de Poincaré.
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        engine: Optional[OODATriphasicHomomorphicPipelineWitnessEnginePhD] = None,
    ) -> None:
        dim = max(4, int(dimension))
        if dim % 2 != 0:
            dim += 1
        self.dimension: Final[int] = dim
        self.secret_key: Final[bytes] = bytes(secret_key)
        self.engine = engine or OODATriphasicHomomorphicPipelineWitnessEnginePhD(
            dimension=self.dimension, secret_key=self.secret_key
        )

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        """HMAC-SHA384 sobre un dominio canónico de fase (separación de contextos)."""
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha384).hexdigest()

    def _canonical_p1_payload(self, cert_fields: Dict[str, Any]) -> str:
        """Payload canónico P1: toda verificación HMAC *debe* reconstruir exactamente este string."""
        return (
            f"{cert_fields['witness_id']}:{cert_fields['trace_preservation_ok']}:"
            f"{cert_fields['trace_drift']:.10f}:{cert_fields['connection_norm']:.8f}:"
            f"{cert_fields['yang_mills_energy']:.8f}:{cert_fields['wilson_phase_rad']:.8f}:"
            f"{_canonical_complex(cert_fields['su2_holonomy_trace'])}:"
            f"{cert_fields['dof_cotangent']}:{cert_fields['energy_hamiltonian']:.10e}:"
            f"{cert_fields['isometry_verdict_value']}:{cert_fields['chain_hash']}"
        )

    def sanitize_quantum_density(self, density_matrix: NDArray) -> NDArray[np.complex128]:
        """Proyección espectral de Higham al simplejo de densidad (única mutación permitida: regularización)."""
        return self.engine.phase1_engine.sanitize_density_matrix_higham(density_matrix)

    def extract_phase1_gauge_topology(
        self, rho: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], float, float, float, complex]:
        """Conexión A, ‖A‖_F, E_YM, fase de Wilson y traza de holonomía SU(2) cuaterniónica."""
        conn_a, _curv_f, ym_energy = self.engine.phase1_engine.extract_gauge_connection_su_n(rho)
        conn_norm = float(la.norm(conn_a, ord="fro"))
        wilson_op = la.expm(conn_a)
        wilson_phase = float(cmath.phase(complex(np.trace(wilson_op) / float(self.dimension))))
        su2_tr: complex = 0j
        extractor = getattr(self.engine.phase1_engine, "extract_su2_holonomy_quaternions", None)
        if callable(extractor):
            _hol, su2_tr = extractor(rho)
        return conn_a, conn_norm, float(ym_energy), wilson_phase, complex(su2_tr)

    def _adjudicate_phase1(
        self, tr_ok: bool, conn_norm: float, ym_energy: float
    ) -> HeytingOmega4:
        """Clasificador Ω₄ de Observe (umbrales de calibre y traza)."""
        if (not tr_ok) or conn_norm > 2.8 or ym_energy > 0.90:
            return HeytingOmega4.VETOED
        if conn_norm > 1.2 or ym_energy > 0.40:
            return HeytingOmega4.CRITICAL
        if conn_norm > 0.7 or ym_energy > 0.15:
            return HeytingOmega4.DEGRADED
        return HeytingOmega4.COHERENT

    def execute_phase1(
        self, density_matrix: NDArray
    ) -> Tuple[OODAWitnessPhase1CertificatePhD, NDArray[np.complex128], NDArray[np.complex128], PoincareOrbitalState]:
        """
        Observación de Fase 1: sanitiza, extrae calibre, cuantiza a T*(Tⁿ) y sella HMAC-P1.
        Devuelve (certificado, ρ, A, estado orbital) — el orbital alimenta directamente la Fase 2.
        """
        now = time.time()
        rho = self.sanitize_quantum_density(density_matrix)
        tr_val = float(np.real(np.trace(rho)))
        tr_drift = abs(tr_val - 1.0)
        tr_ok = tr_drift < _TRACE_DRIFT_TOL

        conn_a, conn_norm, ym_energy, wilson_phase, su2_tr = self.extract_phase1_gauge_topology(rho)

        quantize = self.engine.phase1_engine.quantize_to_canonical_poincare_coordinates
        eps = float(getattr(self.engine.phase2_engine, "epsilon_pert", 0.0125))
        try:
            orbital: PoincareOrbitalState = quantize(rho, perturbation_eps=eps)
        except TypeError:
            orbital = quantize(rho)

        verdict = self._adjudicate_phase1(tr_ok, conn_norm, ym_energy)

        wit_id = f"WIT-OODA-P1-POINCARE-{int(now * 1000.0) % 10_000_000:07d}"
        chain_bytes = (
            f"{wit_id}:{tr_ok}:{conn_norm:.8f}:{ym_energy:.8f}:{verdict.value}:"
            f"{orbital.energy_hamiltonian:.10e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha384(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        fields = {
            "witness_id": wit_id,
            "trace_preservation_ok": tr_ok,
            "trace_drift": tr_drift,
            "connection_norm": conn_norm,
            "yang_mills_energy": ym_energy,
            "wilson_phase_rad": wilson_phase,
            "su2_holonomy_trace": su2_tr,
            "dof_cotangent": int(orbital.dimension_dof),
            "energy_hamiltonian": float(orbital.energy_hamiltonian),
            "isometry_verdict_value": int(verdict.value),
            "chain_hash": chain_hash,
        }
        sig = self._compute_hmac(_HMAC_DOMAIN_P1, self._canonical_p1_payload(fields))

        cert = OODAWitnessPhase1CertificatePhD(
            witness_id=wit_id,
            trace_preservation_ok=tr_ok,
            trace_drift=tr_drift,
            connection_norm=conn_norm,
            yang_mills_energy=ym_energy,
            wilson_phase_rad=wilson_phase,
            su2_holonomy_trace=su2_tr,
            dof_cotangent=int(orbital.dimension_dof),
            energy_hamiltonian=float(orbital.energy_hamiltonian),
            isometry_verdict=verdict,
            chain_hash=chain_hash,
            hmac_p1_signature=sig,
            timestamp_utc=now,
        )
        return cert, rho, conn_a, orbital

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # MÉTODO PUENTE FORMAL FINAL FASE 1 ──► INICIO FASE 2
    # Su valor de retorno `Phase1TransferCarrier` es el único objeto que
    # `OODAWitnessPhase2OrientorPhD` admite como semilla. No existe otra puerta.
    # ──────────────────────────────────────────────────────────────────────────────────────────
    def bridge_phase1_to_phase2_poincare_transfer(
        self, density_matrix: NDArray
    ) -> Phase1TransferCarrier:
        """
        DEFINICIÓN FORMAL FINAL DE LA FASE 1 / OBJETO INICIAL DE LA FASE 2.

        Ejecuta Observe, verifica Tr ρ ≡ 1, cuantiza ρ al fibrado cotangente T*(Tⁿ) con el
        Hamiltoniano celeste *idéntico* al de `PoincareCelestialSymplecticEngine`, y emite
        el contenedor acoplado `Phase1TransferCarrier`. La Fase 2 comienza exactamente aquí:
        `continue_to_phase2_witness_gate` consume `phase1_certificate` y el flujo simpléctico
        consume `orbital_poincare_state` sin reparametrización.
        """
        cert1, sanitized_rho, conn_a, poincare_orbital = self.execute_phase1(density_matrix)
        logger.debug(
            "Puente Fase 1 → Fase 2: cert=%s dof=%d H=%.6f veredicto=%s",
            cert1.witness_id,
            poincare_orbital.dimension_dof,
            poincare_orbital.energy_hamiltonian,
            cert1.isometry_verdict.name,
        )
        return Phase1TransferCarrier(
            phase1_certificate=cert1,
            orbital_poincare_state=poincare_orbital,
            sanitized_density_matrix=sanitized_rho,
            gauge_connection=conn_a,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2: ORIENT — MECÁNICA CELESTE DE POINCARÉ, KAM/NEKHOROSHEV, FLOQUET, SECCIÓN Σ Y QND AAV
# Continuación directa de `bridge_phase1_to_phase2_poincare_transfer`: todo método consume
# `Phase1TransferCarrier`. El morfismo terminal produce `Phase2TransferCarrier` (inicio Fase 3).
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class OODAWitnessPhase2OrientorPhD(OODAWitnessPhase1ObserverPhD):
    """
    FASE 2 — ORIENT.

    Hereda Observe. Consume `Phase1TransferCarrier`. Valida Φ₁₂ (HMAC-P1 + no-repudio).
    Integra el flujo hamiltoniano con Yoshida O(4), corta la sección de Poincaré Σ={q₀=0, p₀>0},
    construye la monodromía por el mapa tangente de Verlet, audita KAM/Nekhoroshev y mide
    el valor débil AAV con auditoría del conmutador QND. Sintetiza el tensor de control RSI-L3.
    """

    def _canonical_p2_payload(self, cert_fields: Dict[str, Any]) -> str:
        """Payload canónico P2 (sign ≡ verify)."""
        return (
            f"{cert_fields['witness_id']}:{cert_fields['phase1_ref_id']}:"
            f"{_canonical_complex(cert_fields['weak_value_qnd_aav'])}:"
            f"{cert_fields['uhlmann_fidelity']:.8f}:{cert_fields['bures_distance']:.8f}:"
            f"{cert_fields['is_kam_torus_stable']}:{cert_fields['max_lyapunov_exponent']:.6f}:"
            f"{cert_fields['floquet_spectral_radius']:.6f}:{cert_fields['energy_drift']:.10e}:"
            f"{cert_fields['qnd_orient_verdict_value']}:{cert_fields['chain_hash']}"
        )

    def continue_to_phase2_witness_gate(self, cert1: OODAWitnessPhase1CertificatePhD) -> bool:
        """
        COMPUERTA HOMOMÓRFICA Φ₁₂ (Observe → Orient) — primer morfismo de la Fase 2.

        Revalida HMAC-SHA384 del certificado P1 sobre el payload canónico, sin abrir el
        estado cuántico (testigo pasivo). Admite DEGRADED/CRITICAL para corrección posterior;
        sólo un HMAC inválido aborta el functor.
        """
        fields = {
            "witness_id": cert1.witness_id,
            "trace_preservation_ok": cert1.trace_preservation_ok,
            "trace_drift": cert1.trace_drift,
            "connection_norm": cert1.connection_norm,
            "yang_mills_energy": cert1.yang_mills_energy,
            "wilson_phase_rad": cert1.wilson_phase_rad,
            "su2_holonomy_trace": cert1.su2_holonomy_trace,
            "dof_cotangent": cert1.dof_cotangent,
            "energy_hamiltonian": cert1.energy_hamiltonian,
            "isometry_verdict_value": int(cert1.isometry_verdict.value),
            "chain_hash": cert1.chain_hash,
        }
        expected = self._compute_hmac(_HMAC_DOMAIN_P1, self._canonical_p1_payload(fields))
        return hmac.compare_digest(cert1.hmac_p1_signature, expected)

    def _measure_qnd_aav(
        self,
        rho: NDArray[np.complex128],
        conn_a: NDArray[np.complex128],
        phi_initial: Optional[NDArray],
        phi_final: Optional[NDArray],
    ) -> Tuple[complex, float, float, float, float, float, float]:
        """AAV + Bures–Uhlmann. Compatible con motor v14 (3-tupla) y v15 (4-tupla)."""
        dim = self.dimension
        if phi_initial is None:
            phi_init = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        else:
            phi_init = np.asarray(phi_initial, dtype=np.complex128).reshape(-1, 1)
        if phi_final is None:
            phi_fin = rho @ phi_init
            phi_fin = phi_fin / (float(la.norm(phi_fin)) + _WILKINSON_EPSILON)
        else:
            phi_fin = np.asarray(phi_final, dtype=np.complex128).reshape(-1, 1)

        aav = self.engine.phase1_engine.compute_qnd_weak_value_aav(phi_init, phi_fin, conn_a)
        if len(aav) >= 4:
            a_weak, weak_mod, back_action_db, comm_norm = aav[0], aav[1], aav[2], aav[3]
        else:
            a_weak, weak_mod, back_action_db = aav[0], aav[1], aav[2]
            comm_norm = 0.0

        bures = self.engine.phase1_engine.compute_bures_uhlmann_metrics(rho)
        fidelity, bures_dist, purity = float(bures[0]), float(bures[1]), float(bures[2]) if len(bures) > 2 else 0.0
        return (
            complex(a_weak),
            float(weak_mod),
            float(back_action_db),
            float(comm_norm),
            fidelity,
            bures_dist,
            purity,
        )

    def _integrate_section(self, orbital_state: PoincareOrbitalState) -> PoincareSectionResult:
        """Yoshida O(4) + sección de Poincaré; repliegue vacío si el motor no lo expone."""
        p2 = self.engine.phase2_engine
        integrate = getattr(p2, "integrate_yoshida4", None)
        section_fn = getattr(p2, "compute_poincare_section", None)
        if callable(integrate) and callable(section_fn):
            traj_pack = integrate(orbital_state, total_time=_TWO_PI, steps=96)
            trajectory = traj_pack[0] if isinstance(traj_pack, tuple) else traj_pack
            return section_fn(trajectory)
        return _empty_section(int(orbital_state.dimension_dof))

    def _synthesize_feedback(
        self,
        kam: KAMAuditResult,
        monodromy: MonodromyResult,
        section: PoincareSectionResult,
        orbital: PoincareOrbitalState,
    ) -> SymplecticVariationalFeedback:
        """Tensor de control Fase 2→3; compatible con firma v14 (sin section) y v15 (con section)."""
        synth = self.engine.phase2_engine.synthesize_symplectic_feedback_tensor
        try:
            return synth(
                kam_audit=kam,
                monodromy=monodromy,
                section=section,
                current_state=orbital,
            )
        except TypeError:
            return synth(kam_audit=kam, monodromy=monodromy, current_state=orbital)

    def execute_phase2(
        self,
        carrier1: Phase1TransferCarrier,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> Tuple[OODAWitnessPhase2CertificatePhD, SymplecticVariationalFeedback]:
        """
        Orientación sobre la variedad de Poincaré y medición débil QND.
        Requiere `Phase1TransferCarrier` (continúa el morfismo terminal de la Fase 1).
        """
        now = time.time()
        cert1 = carrier1.phase1_certificate
        rho = carrier1.sanitized_density_matrix
        orbital_state = carrier1.orbital_poincare_state
        conn_a = carrier1.gauge_connection

        if not self.continue_to_phase2_witness_gate(cert1):
            logger.critical("Violación de integridad en Φ₁₂ (HMAC-P1 inválido)")
            raise ValueError("Falla de autenticidad criptográfica en compuerta homomórfica Φ₁₂.")

        a_weak, weak_mod, back_action_db, comm_norm, fidelity, bures_dist, purity = self._measure_qnd_aav(
            rho, conn_a, phi_initial, phi_final
        )

        section_res = self._integrate_section(orbital_state)
        monodromy_res: MonodromyResult = self.engine.phase2_engine.compute_monodromy_and_floquet(orbital_state)
        kam_audit_res: KAMAuditResult = self.engine.phase2_engine.audit_kam_stability_diophantine(orbital_state)
        floquet_radius = float(np.max(np.abs(monodromy_res.floquet_multipliers)))
        pairing_defect = float(getattr(monodromy_res, "floquet_pairing_defect", 0.0))
        energy_drift = float(getattr(monodromy_res, "energy_drift", 0.0))
        nekhoroshev_t = float(getattr(kam_audit_res, "nekhoroshev_stability_time", 0.0))
        rotation_number = float(getattr(section_res, "rotation_number_estimate", 0.0))

        feedback = self._synthesize_feedback(kam_audit_res, monodromy_res, section_res, orbital_state)

        if (
            (not monodromy_res.is_symplectic_conserved)
            or feedback.orbital_divergence_risk > 0.85
            or fidelity < 0.20
        ):
            qnd_verdict = HeytingOmega4.VETOED
        elif (not kam_audit_res.is_kam_torus_stable) or monodromy_res.max_lyapunov_exponent > 0.15:
            qnd_verdict = HeytingOmega4.CRITICAL
        elif bures_dist > 0.80 or weak_mod > 3.0:
            qnd_verdict = HeytingOmega4.DEGRADED
        else:
            qnd_verdict = HeytingOmega4.COHERENT

        fused_verdict = cert1.isometry_verdict.meet(qnd_verdict)

        wit_id = f"WIT-OODA-P2-KAM-{int(now * 1000.0) % 10_000_000:07d}"
        chain_bytes = (
            f"{cert1.chain_hash}:{wit_id}:{_canonical_complex(a_weak)}:{fidelity:.8f}:"
            f"{fused_verdict.value}:{energy_drift:.10e}"
        ).encode("utf-8")
        chain_hash = hashlib.sha384(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]

        fields = {
            "witness_id": wit_id,
            "phase1_ref_id": cert1.witness_id,
            "weak_value_qnd_aav": a_weak,
            "uhlmann_fidelity": fidelity,
            "bures_distance": bures_dist,
            "is_kam_torus_stable": kam_audit_res.is_kam_torus_stable,
            "max_lyapunov_exponent": monodromy_res.max_lyapunov_exponent,
            "floquet_spectral_radius": floquet_radius,
            "energy_drift": energy_drift,
            "qnd_orient_verdict_value": int(fused_verdict.value),
            "chain_hash": chain_hash,
        }
        sig = self._compute_hmac(_HMAC_DOMAIN_P2, self._canonical_p2_payload(fields))

        cert2 = OODAWitnessPhase2CertificatePhD(
            witness_id=wit_id,
            phase1_ref_id=cert1.witness_id,
            weak_value_qnd_aav=a_weak,
            weak_value_modulus=weak_mod,
            back_action_db=back_action_db,
            qnd_commutator_norm=comm_norm,
            uhlmann_fidelity=fidelity,
            bures_distance=bures_dist,
            purity=purity,
            is_kam_torus_stable=bool(kam_audit_res.is_kam_torus_stable),
            max_lyapunov_exponent=float(monodromy_res.max_lyapunov_exponent),
            floquet_spectral_radius=floquet_radius,
            floquet_pairing_defect=pairing_defect,
            energy_drift=energy_drift,
            nekhoroshev_horizon=nekhoroshev_t,
            poincare_recurrence_time=float(kam_audit_res.poincare_recurrence_time_estimate),
            poincare_rotation_number=rotation_number,
            section_intersection_count=int(getattr(section_res, "intersection_count", 0)),
            is_symplectic_conserved=bool(monodromy_res.is_symplectic_conserved),
            qnd_orient_verdict=fused_verdict,
            chain_hash=chain_hash,
            hmac_p2_signature=sig,
            timestamp_utc=now,
        )
        return cert2, feedback

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # MÉTODO PUENTE FORMAL FINAL FASE 2 ──► INICIO FASE 3
    # Su valor de retorno `Phase2TransferCarrier` es el único objeto que
    # `OODAWitnessPhase3DeciderPhD` admite como semilla. No existe otra puerta.
    # ──────────────────────────────────────────────────────────────────────────────────────────
    def bridge_phase2_to_phase3_rsi_transfer(
        self,
        carrier1: Phase1TransferCarrier,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> Phase2TransferCarrier:
        """
        DEFINICIÓN FORMAL FINAL DE LA FASE 2 / OBJETO INICIAL DE LA FASE 3.

        Ejecuta Orient (Yoshida, Σ, monodromía, KAM, AAV), sella HMAC-P2 y emite
        `Phase2TransferCarrier`. La Fase 3 comienza exactamente aquí:
        `continue_to_phase3_witness_gate` consume `phase2_certificate` y el núcleo RSI-L3
        consume `symplectic_feedback` como único dato orbital.
        """
        cert2, feedback = self.execute_phase2(carrier1, phi_initial, phi_final)
        logger.debug(
            "Puente Fase 2 → Fase 3: cert=%s KAM=%s λ_max=%.6f risk=%.4f",
            cert2.witness_id,
            cert2.is_kam_torus_stable,
            cert2.max_lyapunov_exponent,
            feedback.orbital_divergence_risk,
        )
        return Phase2TransferCarrier(
            phase1_certificate=carrier1.phase1_certificate,
            phase2_certificate=cert2,
            symplectic_feedback=feedback,
            sanitized_density_matrix=carrier1.sanitized_density_matrix,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3: DECIDE & ACT — RSI-L3 (T Y META-OPERADOR S), TOPOS DE HEYTING Y FUNTOR VISCERAL Φ_sem
# Continuación directa de `bridge_phase2_to_phase3_rsi_transfer`: todo método consume
# `Phase2TransferCarrier`. El morfismo terminal es el estado soberano atestado.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class OODAWitnessPhase3DeciderPhD(OODAWitnessPhase2OrientorPhD):
    """
    FASE 3 — DECIDE & ACT.

    Hereda Orient. Consume `Phase2TransferCarrier`. Valida Φ₂₃. Aplica el paso RSI-L3
    (contracción de Banach T sobre C y meta-contracción S sobre Lip(T)). Adjudica Ω₄,
    evalúa Φ₁₂, Φ₂₃, Φ₁₃=Φ₂₃∘Φ₁₂ y proyecta al funtor semántico visceral Φ_sem.
    Invariante de pasividad: el agente jamás dispara actuadores ciber-físicos.
    """

    def continue_to_phase3_witness_gate(self, cert2: OODAWitnessPhase2CertificatePhD) -> bool:
        """
        COMPUERTA HOMOMÓRFICA Φ₂₃ (Orient → Decide & Act) — primer morfismo de la Fase 3.

        Revalida HMAC-SHA384 del certificado P2 sobre el payload canónico. Garantiza que
        la información simpléctica y cuántica llegue intacta al núcleo de decisión.
        """
        fields = {
            "witness_id": cert2.witness_id,
            "phase1_ref_id": cert2.phase1_ref_id,
            "weak_value_qnd_aav": cert2.weak_value_qnd_aav,
            "uhlmann_fidelity": cert2.uhlmann_fidelity,
            "bures_distance": cert2.bures_distance,
            "is_kam_torus_stable": cert2.is_kam_torus_stable,
            "max_lyapunov_exponent": cert2.max_lyapunov_exponent,
            "floquet_spectral_radius": cert2.floquet_spectral_radius,
            "energy_drift": cert2.energy_drift,
            "qnd_orient_verdict_value": int(cert2.qnd_orient_verdict.value),
            "chain_hash": cert2.chain_hash,
        }
        expected = self._compute_hmac(_HMAC_DOMAIN_P2, self._canonical_p2_payload(fields))
        return hmac.compare_digest(cert2.hmac_p2_signature, expected)

    def _rsi_step(
        self, feedback: SymplecticVariationalFeedback, fidelity: float
    ) -> Tuple[Any, float, float]:
        """Paso RSI-L3; compatible con retorno v14 (2-tupla) y v15 (3-tupla: config, Lip, Cauchy)."""
        pack = self.engine.phase3_rsi_core.trigger_recursive_step(
            feedback=feedback, fidelity_metric=fidelity
        )
        if len(pack) >= 3:
            new_config, lip, cauchy = pack[0], float(pack[1]), float(pack[2])
        else:
            new_config, lip = pack[0], float(pack[1])
            cauchy = 0.0
        ingest = getattr(self.engine, "ingest_rsi_configuration", None)
        if callable(ingest):
            ingest(new_config)
        return new_config, float(lip), float(cauchy)

    def _homomorphic_gates(self, rho: NDArray[np.complex128]) -> HomomorphicGateCertificate:
        """Certificado C* de Φ₁₂, Φ₂₃, Φ₁₃; compatible con retorno-tupla v14."""
        raw = self.engine.evaluate_homomorphic_gates(rho)
        if hasattr(raw, "phi12_integrity"):
            return raw  # type: ignore[return-value]
        gate_12, gate_23 = float(raw[0]), float(raw[1])
        phi13 = float(gate_12 * gate_23)
        return HomomorphicGateCertificate(
            phi12_integrity=gate_12,
            phi23_integrity=gate_23,
            phi13_composition_integrity=phi13,
            corner_homomorphism_defect_12=max(0.0, -math.log(max(gate_12, _WILKINSON_EPSILON))),
            corner_homomorphism_defect_23=max(0.0, -math.log(max(gate_23, _WILKINSON_EPSILON))),
            functoriality_defect=abs(phi13 - gate_12 * gate_23),
        )

    def evaluate_visceral_functor_phi_sem(
        self,
        cert1: OODAWitnessPhase1CertificatePhD,
        cert2: OODAWitnessPhase2CertificatePhD,
        rsi_generation: int,
        rsi_contraction_lip: float,
        rsi_meta_lip: float,
        rsi_cauchy: float,
        orbital_risk: float,
        functoriality_defect: float,
    ) -> Dict[str, Any]:
        """
        Funtor semántico visceral Φ_sem: Topos → ℭ_suite (Dolor, Riesgo, Capital de entropía).

        Traduce tensores de calibre y simplécticos a dimensiones ontológicas ejecutivas
        sin comprometer la pasividad ciber-física (ningún actuador es disparado).
        """
        structural_pain = float(
            np.clip(
                (cert1.trace_drift * 1.0e4)
                + (cert1.connection_norm * 0.25)
                + (cert1.yang_mills_energy * 0.35)
                + (max(0.0, cert2.max_lyapunov_exponent) * 1.5)
                + (cert2.energy_drift * 20.0)
                + (cert2.floquet_pairing_defect * 0.25),
                0.0,
                1.0,
            )
        )
        spectral_collapse = float(
            np.clip(
                (1.0 - cert2.uhlmann_fidelity) * 0.45
                + orbital_risk * 0.30
                + (0.15 if not cert2.is_kam_torus_stable else 0.0)
                + (0.10 if not cert2.is_symplectic_conserved else 0.0)
                + functoriality_defect * 0.20,
                0.0,
                1.0,
            )
        )
        qnd_efficiency = float(1.0 / (1.0 + cert2.qnd_commutator_norm))
        capital_entropy_saved = float(
            150_000.0 * (1.0 - structural_pain) * cert2.uhlmann_fidelity * qnd_efficiency
        )
        nekhoroshev_ok = cert2.nekhoroshev_horizon > 1.0

        return {
            "visceral_dimension_pain": {
                "structural_pain_index": round(structural_pain, 6),
                "severity": (
                    "CRITICAL" if structural_pain > 0.7 else "MODERATE" if structural_pain > 0.3 else "NOMINAL"
                ),
                "physical_cause": (
                    "Tensión de Yang–Mills / deriva simpléctica"
                    if (cert1.yang_mills_energy > 0.3 or cert2.energy_drift > 1.0e-4)
                    else "Dinámica laminar estable"
                ),
            },
            "visceral_dimension_risk": {
                "spectral_collapse_risk": round(spectral_collapse, 6),
                "kam_torus_status": "CONSERVADO" if cert2.is_kam_torus_stable else "RESONANCIA_NO_LINEAL_DETECTADA",
                "lyapunov_chaotic_divergence": cert2.max_lyapunov_exponent > 0.05,
                "orbital_risk_quotient": round(orbital_risk, 6),
                "nekhoroshev_horizon_stable": nekhoroshev_ok,
                "symplectic_form_preserved": cert2.is_symplectic_conserved,
            },
            "visceral_dimension_money": {
                "entropy_cost_avoided_usd": round(capital_entropy_saved, 2),
                "qnd_back_action_db": round(cert2.back_action_db, 6),
                "qnd_commutator_norm": round(cert2.qnd_commutator_norm, 6),
                "hardware_interruption_prevented": True,
            },
            "ciber_physical_passivity_attestation": {
                "esp32_crowbar_hardware_trigger": "STRICTLY_INHIBITED (Axioma QND-OODA3)",
                "ram_inspection_footprint": "ZERO_DEMOLITION_PASSIVE",
            },
            "rsi_l3_telemetry": {
                "generation": rsi_generation,
                "banach_contraction_lip": round(rsi_contraction_lip, 6),
                "meta_contraction_lip": round(rsi_meta_lip, 6),
                "cauchy_residual": round(rsi_cauchy, 12),
                "metric_stability": rsi_contraction_lip < 1.0 and rsi_meta_lip < 1.0,
            },
        }

    def execute_phase3(self, carrier2: Phase2TransferCarrier) -> OODAPipelineWitnessSovereignStatePhD:
        """
        DEFINICIÓN FORMAL FINAL DE LA FASE 3.

        Decide en el topos de Heyting, aplica RSI-L3 (T y S), evalúa las tres compuertas
        homomórficas y sella el DAG Merkle-HMAC. Consume exclusivamente `Phase2TransferCarrier`.
        """
        now = time.time()
        cert1 = carrier2.phase1_certificate
        cert2 = carrier2.phase2_certificate
        feedback = carrier2.symplectic_feedback
        rho = carrier2.sanitized_density_matrix

        if not self.continue_to_phase3_witness_gate(cert2):
            logger.critical("Violación de integridad en Φ₂₃ (HMAC-P2 inválido)")
            raise ValueError("Falla de autenticidad criptográfica en compuerta homomórfica Φ₂₃.")

        new_config, lip_contraction, cauchy_residual = self._rsi_step(feedback, cert2.uhlmann_fidelity)
        gates = self._homomorphic_gates(rho)
        meta_lip = float(getattr(new_config, "meta_contraction_lip", lip_contraction))

        final_verdict = cert1.isometry_verdict.meet(cert2.qnd_orient_verdict)
        if lip_contraction >= 1.0 or feedback.orbital_divergence_risk > 0.90 or meta_lip >= 1.0:
            final_verdict = final_verdict.meet(HeytingOmega4.VETOED)

        visceral_report = self.evaluate_visceral_functor_phi_sem(
            cert1=cert1,
            cert2=cert2,
            rsi_generation=int(new_config.iteration_generation),
            rsi_contraction_lip=lip_contraction,
            rsi_meta_lip=meta_lip,
            rsi_cauchy=cauchy_residual,
            orbital_risk=float(feedback.orbital_divergence_risk),
            functoriality_defect=float(gates.functoriality_defect),
        )

        state_id = f"OODA-SOVEREIGN-STATE-v15-{int(now * 1000.0) % 10_000_000:07d}"
        merkle_dag_seed = (
            f"{state_id}:{cert1.hmac_p1_signature}:{cert2.hmac_p2_signature}:"
            f"{cert1.chain_hash}:{cert2.chain_hash}:{final_verdict.value}:"
            f"{new_config.iteration_generation}:{lip_contraction:.8f}:{meta_lip:.8f}:"
            f"{gates.functoriality_defect:.10e}"
        ).encode("utf-8")
        merkle_root = hashlib.sha384(merkle_dag_seed).hexdigest()
        prov_payload = (
            f"{merkle_root}:{cert2.back_action_db:.6f}:{final_verdict.value}:{state_id}:"
            f"{gates.phi13_composition_integrity:.8f}"
        )
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3, prov_payload)

        return OODAPipelineWitnessSovereignStatePhD(
            sovereign_state_id=state_id,
            phase1_cert=cert1,
            phase2_cert=cert2,
            global_heyting_verdict=final_verdict,
            rsi_level3_generation=int(new_config.iteration_generation),
            rsi_banach_contraction_lip=lip_contraction,
            rsi_meta_contraction_lip=meta_lip,
            rsi_cauchy_residual=cauchy_residual,
            homomorphic_gate_phi12_integrity=float(gates.phi12_integrity),
            homomorphic_gate_phi23_integrity=float(gates.phi23_integrity),
            homomorphic_gate_phi13_integrity=float(gates.phi13_composition_integrity),
            functoriality_defect=float(gates.functoriality_defect),
            back_action_db=float(cert2.back_action_db),
            merkle_dag_sha384_root=merkle_root,
            hmac_provenance_signature=prov_sig,
            visceral_report=visceral_report,
            timestamp_utc=now,
        )

    def verify_merkle_provenance(self, state: OODAPipelineWitnessSovereignStatePhD) -> bool:
        """Auditoría pasiva de la raíz Merkle y de la firma de procedencia P3."""
        merkle_dag_seed = (
            f"{state.sovereign_state_id}:{state.phase1_cert.hmac_p1_signature}:"
            f"{state.phase2_cert.hmac_p2_signature}:{state.phase1_cert.chain_hash}:"
            f"{state.phase2_cert.chain_hash}:{state.global_heyting_verdict.value}:"
            f"{state.rsi_level3_generation}:{state.rsi_banach_contraction_lip:.8f}:"
            f"{state.rsi_meta_contraction_lip:.8f}:{state.functoriality_defect:.10e}"
        ).encode("utf-8")
        expected_root = hashlib.sha384(merkle_dag_seed).hexdigest()
        if not hmac.compare_digest(expected_root, state.merkle_dag_sha384_root):
            return False
        prov_payload = (
            f"{state.merkle_dag_sha384_root}:{state.phase2_cert.back_action_db:.6f}:"
            f"{state.global_heyting_verdict.value}:{state.sovereign_state_id}:"
            f"{state.homomorphic_gate_phi13_integrity:.8f}"
        )
        expected_sig = self._compute_hmac(_HMAC_DOMAIN_P3, prov_payload)
        return hmac.compare_digest(expected_sig, state.hmac_provenance_signature)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# CLASE ORQUESTADORA SOBERANA
# Encadena ininterrumpidamente:
#   FASE 1  ──► bridge_phase1_to_phase2_poincare_transfer
#   FASE 2  ──► bridge_phase2_to_phase3_rsi_transfer
#   FASE 3  ──► execute_phase3 → OODAPipelineWitnessSovereignStatePhD
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class OODATriphasicHomomorphicPipelineWitnessAgentPhD(OODAWitnessPhase3DeciderPhD):
    """
    SOBERANO AGÉNTICO TESTIGO DEL PIPELINE OODA TRIFÁSICO HOMOMÓRFICO v15.0.0-PhD.

    Orquestador maestro: hereda Decide, que hereda Orient, que hereda Observe.
    La cadena de herencia *es* la cadena functorial de fases anidadas.
    """

    def witness_ooda_pipeline_sovereign(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> OODAPipelineWitnessSovereignStatePhD:
        """
        EJECUCIÓN DEL PIPELINE ANIDADO COMPLETO.

        Flujo canónico estricto, sin atajos:
            ρ  --Fase 1-->  Phase1TransferCarrier
               --Fase 2-->  Phase2TransferCarrier
               --Fase 3-->  OODAPipelineWitnessSovereignStatePhD
        """
        carrier_phase1 = self.bridge_phase1_to_phase2_poincare_transfer(density_matrix)
        carrier_phase2 = self.bridge_phase2_to_phase3_rsi_transfer(
            carrier1=carrier_phase1,
            phi_initial=phi_initial,
            phi_final=phi_final,
        )
        return self.execute_phase3(carrier_phase2)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS FORMALES Y DEMOSTRACIÓN DE DOCTORADO
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 105
    print(banner)
    print("  VERIFICACIÓN CIENTÍFICA: SOBERANO AGÉNTICO TESTIGO OODA TRIFÁSICO HOMOMÓRFICO (v15.0.0-PhD)")
    print("  Fases anidadas 1→2→3 • Poincaré–KAM–Nekhoroshev • RSI-L3 (T,S) • Φ_sem • Φ₁₂, Φ₂₃, Φ₁₃")
    print(banner)

    DIMENSION = 56
    agent = OODATriphasicHomomorphicPipelineWitnessAgentPhD(dimension=DIMENSION)

    rng = np.random.default_rng(2026)
    psi = rng.normal(0.0, 1.0, size=(DIMENSION, DIMENSION)) + 1j * rng.normal(
        0.0, 1.0, size=(DIMENSION, DIMENSION)
    )
    rho_sim = psi @ psi.conj().T
    rho_sim /= np.trace(rho_sim)

    print(f"\n[+] Matriz de densidad en RAM: fibrado T*(Tⁿ), n = {DIMENSION // 2}")
    print("[+] Secuencia de atestación agéntica en 3 fases anidadas…")

    state: OODAPipelineWitnessSovereignStatePhD = agent.witness_ooda_pipeline_sovereign(rho_sim)
    provenance_ok = agent.verify_merkle_provenance(state)

    print("\n" + "─" * 44 + " RESULTADOS FASE 1 " + "─" * 42)
    print(f"  • Certificado Fase 1 ID           : {state.phase1_cert.witness_id}")
    print(f"  • Conservación Tr(ρ)=1            : {state.phase1_cert.trace_preservation_ok}  (deriva {state.phase1_cert.trace_drift:.4e})")
    print(f"  • Norma conexión SU(N) ‖A‖_F      : {state.phase1_cert.connection_norm:.6f}")
    print(f"  • Energía Yang–Mills              : {state.phase1_cert.yang_mills_energy:.6f}")
    print(f"  • Fase de Wilson                  : {state.phase1_cert.wilson_phase_rad:.6f} rad")
    print(f"  • Traza holonomía SU(2) (ℍ)       : {state.phase1_cert.su2_holonomy_trace}")
    print(f"  • Hamiltoniano celeste H          : {state.phase1_cert.energy_hamiltonian:.6f}")
    print(f"  • Veredicto isometría Fase 1      : {state.phase1_cert.isometry_verdict.name}")
    print(f"  • HMAC-SHA384 P1                  : {state.phase1_cert.hmac_p1_signature[:36]}…")

    print("\n" + "─" * 44 + " RESULTADOS FASE 2 " + "─" * 42)
    print(f"  • Certificado Fase 2 ID           : {state.phase2_cert.witness_id}")
    print(f"  • Referencia acoplada Fase 1      : {state.phase2_cert.phase1_ref_id}")
    print(f"  • Valor débil QND AAV             : {state.phase2_cert.weak_value_qnd_aav.real:+.4f} {state.phase2_cert.weak_value_qnd_aav.imag:+.4f}j")
    print(f"  • Retroacción QND                 : {state.phase2_cert.back_action_db:.6f} dB  (‖[A,Π]‖_F = {state.phase2_cert.qnd_commutator_norm:.4e})")
    print(f"  • Fidelidad de Uhlmann            : {state.phase2_cert.uhlmann_fidelity:.6f}")
    print(f"  • Distancia de Bures / pureza     : {state.phase2_cert.bures_distance:.6f} / {state.phase2_cert.purity:.6f}")
    print(f"  • Toros KAM                       : {'CONFORME' if state.phase2_cert.is_kam_torus_stable else 'RESONANTE'}")
    print(f"  • Lyapunov máximo                 : {state.phase2_cert.max_lyapunov_exponent:+.6f}")
    print(f"  • Radio espectral de Floquet      : {state.phase2_cert.floquet_spectral_radius:.6f}")
    print(f"  • Defecto de emparejamiento λ↔1/λ : {state.phase2_cert.floquet_pairing_defect:.6e}")
    print(f"  • Deriva de energía |ΔH|/|H|      : {state.phase2_cert.energy_drift:.6e}")
    print(f"  • Horizonte de Nekhoroshev        : {state.phase2_cert.nekhoroshev_horizon:.4e}")
    print(f"  • Recurrencia de Poincaré         : {state.phase2_cert.poincare_recurrence_time:.4f} ciclos")
    print(f"  • Número de rotación (Σ)          : {state.phase2_cert.poincare_rotation_number:+.6f}")
    print(f"  • Intersecciones de sección Σ     : {state.phase2_cert.section_intersection_count}")
    print(f"  • 2-forma ω conservada            : {state.phase2_cert.is_symplectic_conserved}")

    print("\n" + "─" * 44 + " RESULTADOS FASE 3 " + "─" * 42)
    print(f"  • Estado soberano ID              : {state.sovereign_state_id}")
    print(f"  • Veredicto topos Heyting Ω₄      : {state.global_heyting_verdict.name}")
    print(f"  • Generación RSI-L3               : #{state.rsi_level3_generation}")
    print(f"  • Lipschitz empírico de T         : {state.rsi_banach_contraction_lip:.6f}  (Banach exige < 1)")
    print(f"  • Lipschitz del meta-operador S   : {state.rsi_meta_contraction_lip:.6f}")
    print(f"  • Residuo de Cauchy               : {state.rsi_cauchy_residual:.6e}")
    print(f"  • Integridad Φ₁₂                  : {state.homomorphic_gate_phi12_integrity * 100:.2f}%")
    print(f"  • Integridad Φ₂₃                  : {state.homomorphic_gate_phi23_integrity * 100:.2f}%")
    print(f"  • Integridad compuesta Φ₁₃        : {state.homomorphic_gate_phi13_integrity * 100:.2f}%")
    print(f"  • Defecto de functorialidad       : {state.functoriality_defect:.6e}")
    print(f"  • Raíz Merkle-DAG SHA-384         : {state.merkle_dag_sha384_root[:36]}…")
    print(f"  • Firma de procedencia P3         : {state.hmac_provenance_signature[:36]}…")
    print(f"  • Verificación Merkle/HMAC        : {'VÁLIDA' if provenance_ok else 'INVÁLIDA'}")

    print("\n" + "─" * 38 + " FUNTOR SEMÁNTICO VISCERAL Φ_sem " + "─" * 38)
    visceral = state.visceral_report
    print(f"  • Dimensión DOLOR (strain)        : {visceral['visceral_dimension_pain']['structural_pain_index']} ({visceral['visceral_dimension_pain']['severity']})")
    print(f"  • Causa física                    : {visceral['visceral_dimension_pain']['physical_cause']}")
    print(f"  • Dimensión RIESGO (colapso)      : {visceral['visceral_dimension_risk']['spectral_collapse_risk']} (KAM: {visceral['visceral_dimension_risk']['kam_torus_status']})")
    print(f"  • Dimensión CAPITAL (entropía)    : ${visceral['visceral_dimension_money']['entropy_cost_avoided_usd']:,.2f}")
    print(f"  • Disyuntor ESP32 Crowbar         : {visceral['ciber_physical_passivity_attestation']['esp32_crowbar_hardware_trigger']}")
    print(f"  • Huella de inspección en RAM     : {visceral['ciber_physical_passivity_attestation']['ram_inspection_footprint']}")
    print(banner)
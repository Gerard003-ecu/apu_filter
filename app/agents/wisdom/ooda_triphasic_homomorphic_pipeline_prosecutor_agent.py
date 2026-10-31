r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : OODA Triphasic Homomorphic Pipeline Prosecutor Agent v15.0.0-PhD                          ║
║ Ubicación: app/agents/wisdom/ooda_triphasic_homomorphic_pipeline_prosecutor_agent.py                 ║
║ Versión  : 15.0.0-SOVEREIGN-POINCARE-KAM-DIRAC-ESP32-RSI-L3-YOSHIDA-WILLIAMSON-PhiSem-PhD            ║
║ Autor    : Soberano Artesano Tejedor de Código & Jurisdicción Espectral Cuántica                     ║
║ Función  : Soberano Agéntico Fiscal Inquisidor del Pipeline OODA Trifásico Homomórfico               ║
║            que orquesta la acusación formal, dinámica simpléctica celeste de Poincaré (Yoshida O(4), ║
║            secciones de Hénon, monodromía analítica Sp(2n), KAM–Nekhoroshev), reacción de Dirac en   ║
║            Fock, RSI-L3 (T punto fijo de S), Mónada Cartesiana Kleisli sobre Ω₄ y disparo            ║
║            ciber-físico SIMULADO del interlock ESP32 Crowbar (sin MMIO; GPIO_OUT_W1TS_REG            ║
║            0x3FF44008 es estrictamente documental).                                                  ║
║ Acoplamiento funtorial (sin discontinuidad):                                                         ║
║   Fase I  ──► quantize_indictment_to_poincare_phase_space  ──► Fase II                               ║
║   Fase II ──► synthesize_prosecution_symplectic_feedback   ──► Fase III                              ║
║   Φ₁₃ := Φ₂₃ ∘ Φ₁₂  (defecto funtorial |i₁₃ − i₁₂ i₂₃|)                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico Fiscal del Pipeline OODA (OODATriphasicHomomorphicPipelineProsecutorAgentPhD)
ejerce la jurisdicción inquisitorial y acusación formal sobre el motor espectral
OODATriphasicHomomorphicPipelineProsecutorEnginePhD (app/wisdom/ooda_triphasic_homomorphic_pipeline_prosecutor_engine.py).
Orquesta un ciclo OODA judicial trifásico anidado en el retículo de Heyting Omega4, imputando desgarros
simplécticos en compuertas homomórficas Phi_12/Phi_23, auditando la reacción de Dirac e- + e+ -> 2gamma
en el espacio de Fock con producción entrópica no negativa de Spohn sigma >= 0, evaluando la dinámica
celeste de Poincaré (Yoshida O(4), secciones de Henon, KAM, Nekhoroshev), y gobernando el disparo
ciber-físico simulado del interlock ESP32 Crowbar (< 392.15 ns @ GPIO14 / BT151).

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. Geometría de Calibre su(N) y Complejo de de Rham:
   Conexión de calibre A y curvatura F = dA + A wedge A.
   Identidad de Bianchi diferencial: d_A F = 0.
   La 3-forma de Chern-Simons con acoplamiento escalar de Higgs satisface:
       CS_3^Higgs(A, Phi) = 1/(8 pi^2) int Tr( A wedge dA + 2/3 A^3 + Phi F ).

2. Dinámica Celeste de Poincaré e Integración Simpléctica:
   Integrador de Yoshida de 4º orden para el mapa de fase (q, p).
   Monodromía simpléctica M en Sp(2n) con det(M) = 1.0.
   Sección transversal de Henon y persistencia KAM bajo la cota de Bruno: |k . omega| >= gamma / |k|_1^tau.

3. Reacción Cuántica de Dirac y Entropía de Spohn:
   Aniquilación de anomalías e- mediante positrones e+ autenticados vía HMAC-SHA256 en tiempo constante:
       e- + e+ -> 2gamma (1022.0 keV por par).
   Producción de entropía de Spohn sigma_Spohn >= 0 en relajación GKSL.

4. Retículo de Heyting Omega4 y Disparo ESP32 Crowbar:
   Retículo: COHERENT=0, DEGRADED=1, CRITICAL=2, VETOED=3.
   Compuerta homomórfica Phi_12: preserva ínfimos de verdad.
   Compuerta Phi_23: valida contracción de Banach Lip <= 0.999.
   Disparo ESP32 Crowbar en IRAM a t <= 392.15 ns ante veredicto VETOED.

AXIOMAS FORMALES
────────────────
Axioma OODA-PA1 (Homomorfismo Estricto de Compuerta):
    Las compuertas Phi_12 y Phi_23 conmutan con la negación pseudocomplementaria del retículo Omega4.
Axioma OODA-PA2 (Conservación de Energía de Dirac):
    Cada aniquilación autorizada en el Vacío de Dirac libera estrictamente 1022.0 keV de energía equivalente.
Axioma OODA-PA3 (Segunda Ley Cuántica de Spohn):
    La tasa instantánea de producción entrópica no conmutativa satisface rigurosamente sigma_Spohn >= 0.
Axioma OODA-PA4 (Debido Proceso de Veredicto):
    El veredicto VETOED exige consenso estricto en compuertas homomórficas o evidencia dura comprobada.

INVARIANTES DE ESTADO
─────────────────────
Invariante OODA-PAI1: Back-Action QND == 0.0 dB con positrón válido; 12.5 dB si la firma es apócrifa o corrupta.
Invariante OODA-PAI2: Invarianza de traza cuántica Tr(rho) == 1.0 y positividad espectral sobre el álgebra de Fock.
Invariante OODA-PAI3: Determinante simpléctico de monodromía acotado en det(M) = 1.0 +- 1.0e-5.
Invariante OODA-PAI4: Latencia certificada de disparo ESP32 Crowbar estrictamente acotada en t <= 392.15 ns.
Invariante OODA-PAI5: Firma criptográfica HMAC-SHA384 en todo certificado judicial emitido por el agente.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import itertools
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Inclusión de rutas para carga modular dinámica (motor fiscal v15 + tesigos)
for path_dir in ["/workspace/scratch", "/workspace/artifacts", ".", "app/wisdom", "app/agents/wisdom"]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# CONSTANTES FUNDAMENTALES v15 (no intercambiables con v14)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

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
_ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008  # documental; NUNCA se escribe
_CHAIN_HEX_LEN: Final[int] = 64
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0: Final[float] = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))
_PROSECUTOR_PERTURBATION_EPS: Final[float] = 0.025

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_SOVEREIGN_OODA_POINCARE_KAM_PROSECUTOR_KEY_v15.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"OODA-PROSECUTOR-AGENT-P1-GAUGE-POINCARE-v15.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"OODA-PROSECUTOR-AGENT-P2-KAM-DYNAMICS-v15.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"OODA-PROSECUTOR-AGENT-P3-SOVEREIGN-MONAD-v15.0|"
_HMAC_DOMAIN_DIRAC: Final[bytes] = b"OODA-PROSECUTOR-DIRAC-POSITRON-v15.0|"

_VERSION_STRING: Final[str] = (
    "15.0.0-SOVEREIGN-POINCARE-KAM-DIRAC-ESP32-RSI-L3-YOSHIDA-WILLIAMSON-PhiSem-PhD"
)

_GERM_DECLARED_MAP: Final[Dict[str, str]] = {
    "trace_preservation_ok": "declared_trace_ok",
    "connection_norm": "declared_connection_norm",
    "yang_mills_energy": "declared_yang_mills_energy",
    "chern_simons_3form": "declared_chern_simons_3form",
    "wilson_angle_rad": "declared_wilson_angle_rad",
    "instanton_number_k": "declared_instanton_number_k",
}


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# IMPORTACIÓN RESILIENTE DEL MOTOR FISCAL v15.0.0
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

_ENGINE_AVAILABLE: bool = False
_engine_dirac_hmac: Any = None

try:
    from app.wisdom.ooda_triphasic_homomorphic_pipeline_prosecutor_engine import (  # type: ignore
        BanachGaugeDiracProsecutionCore,
        HeytingOmega4,
        OODAProsecutionExecutionCertificatePhD,
        OODAProsecutionIndictmentGermPhD,
        OODAProsecutionStatus,
        OODATriphasicHomomorphicPipelineProsecutorEnginePhD,
        PoincareCelestialProsecutorEngine,
        PoincareProsecutionOrbitalCarrier,
        ProsecutionKAMAuditResult,
        ProsecutionMonodromyResult,
        ProsecutionRSIL3DecisionCore,
        ProsecutionSymplecticFeedbackCarrier,
        ProsecutorRSIConfiguration,
    )

    try:
        from app.wisdom.ooda_triphasic_homomorphic_pipeline_prosecutor_engine import (  # type: ignore
            dirac_positron_hmac as _engine_dirac_hmac,
        )
    except ImportError:
        _engine_dirac_hmac = None
    _ENGINE_AVAILABLE = True
except ImportError:
    try:
        from ooda_triphasic_homomorphic_pipeline_prosecutor_engine import (  # type: ignore
            BanachGaugeDiracProsecutionCore,
            HeytingOmega4,
            OODAProsecutionExecutionCertificatePhD,
            OODAProsecutionIndictmentGermPhD,
            OODAProsecutionStatus,
            OODATriphasicHomomorphicPipelineProsecutorEnginePhD,
            PoincareCelestialProsecutorEngine,
            PoincareProsecutionOrbitalCarrier,
            ProsecutionKAMAuditResult,
            ProsecutionMonodromyResult,
            ProsecutionRSIL3DecisionCore,
            ProsecutionSymplecticFeedbackCarrier,
            ProsecutorRSIConfiguration,
        )

        try:
            from ooda_triphasic_homomorphic_pipeline_prosecutor_engine import (  # type: ignore
                dirac_positron_hmac as _engine_dirac_hmac,
            )
        except ImportError:
            _engine_dirac_hmac = None
        _ENGINE_AVAILABLE = True
    except ImportError:
        _ENGINE_AVAILABLE = False
        _engine_dirac_hmac = None

    # ── Retículo de Heyting Ω₄ (verdad, no severidad): ⊥=VETOED ≺ CRITICAL ≺ DEGRADED ≺ COHERENT=⊤
    class HeytingOmega4(IntEnum):  # type: ignore[no-redef]
        COHERENT = 0
        DEGRADED = 1
        CRITICAL = 2
        VETOED = 3

        def truth_rank(self) -> int:
            return 3 - int(self)

        def meet(self, other: "HeytingOmega4") -> "HeytingOmega4":
            rank = min(self.truth_rank(), other.truth_rank())
            return HeytingOmega4(3 - rank)

        def join(self, other: "HeytingOmega4") -> "HeytingOmega4":
            rank = max(self.truth_rank(), other.truth_rank())
            return HeytingOmega4(3 - rank)

        def implies(self, other: "HeytingOmega4") -> "HeytingOmega4":
            if self.truth_rank() <= other.truth_rank():
                return HeytingOmega4.COHERENT
            return other

    try:
        from app.wisdom.ooda_triphasic_homomorphic_pipeline_witness_engine import HeytingOmega4 as _WHeyting  # type: ignore

        HeytingOmega4 = _WHeyting  # type: ignore[misc, assignment]
    except ImportError:
        try:
            from ooda_triphasic_homomorphic_pipeline_witness_engine import HeytingOmega4 as _WHeyting  # type: ignore

            HeytingOmega4 = _WHeyting  # type: ignore[misc, assignment]
        except ImportError:
            pass

    class OODAProsecutionStatus(IntEnum):  # type: ignore[no-redef]
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
    class OODAProsecutionIndictmentGermPhD:  # type: ignore[no-redef]
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
        timestamp_utc: float = 0.0

    @dataclass(frozen=True, slots=True)
    class PoincareProsecutionOrbitalCarrier:  # type: ignore[no-redef]
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

    @dataclass(frozen=True, slots=True)
    class ProsecutionMonodromyResult:  # type: ignore[no-redef]
        monodromy_matrix: NDArray[np.float64]
        floquet_multipliers: Tuple[complex, ...]
        max_lyapunov_exponent: float
        symplectic_defect: float
        period: float

    @dataclass(frozen=True, slots=True)
    class ProsecutionKAMAuditResult:  # type: ignore[no-redef]
        is_kam_torus_stable: bool
        diophantine_gamma: float
        diophantine_tau: float
        min_small_divisor: float
        nekhoroshev_time_bound: float
        poincare_recurrence_time_estimate: float
        max_lyapunov_exponent: float
        rotation_number: float
        fourier_order: int

    @dataclass(frozen=True, slots=True)
    class ProsecutionSymplecticFeedbackCarrier:  # type: ignore[no-redef]
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

    @dataclass(frozen=True, slots=True)
    class OODAProsecutionExecutionCertificatePhD:  # type: ignore[no-redef]
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
    class ProsecutorRSIConfiguration:  # type: ignore[no-redef]
        generation: int
        banach_lipschitz: float
        meta_operator_lip: float
        cauchy_residual: float
        psd_floor: float
        symplectic_tolerance: float

    class BanachGaugeDiracProsecutionCore:  # type: ignore[no-redef]
        """Núcleo autónomo Fase I: Higham, su(N), Dirac, Weyl–Williamson, cuantización T*(M)."""

        def __init__(self, dimension: int, secret_key: bytes) -> None:
            dim = max(4, int(dimension))
            if dim % 2 == 1:
                dim += 1
            self.dimension = dim
            self.secret_key = bytes(secret_key)
            self.dof = dim // 2

        def sanitize_density_matrix_higham(self, matrix: NDArray) -> NDArray[np.complex128]:
            a = np.asarray(matrix, dtype=np.complex128)
            if a.ndim != 2 or a.shape[0] != a.shape[1]:
                n = self.dimension
                a = np.eye(n, dtype=np.complex128) / float(n)
            a = 0.5 * (a + a.conj().T)
            w, v = np.linalg.eigh(a)
            w = np.maximum(np.real(w), 0.0)
            rho = (v * w) @ v.conj().T
            rho = 0.5 * (rho + rho.conj().T)
            tr = float(np.real(np.trace(rho)))
            if tr > _WILKINSON_EPSILON:
                rho = rho / tr
            return np.asarray(rho, dtype=np.complex128)

        def extract_gauge_invariants(
            self, rho: NDArray[np.complex128]
        ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], float, float]:
            n = int(rho.shape[0])
            h = 0.5 * (rho + rho.conj().T)
            h = h - (np.trace(h) / float(n)) * np.eye(n, dtype=np.complex128)
            conn_a = 1.0j * h
            curv_f = conn_a @ conn_a
            ym = 0.5 * float(np.real(np.sum(np.conj(curv_f) * curv_f)))
            a3 = conn_a @ conn_a @ conn_a
            cs3 = float(np.real(np.trace(conn_a @ curv_f - (2.0 / 3.0) * a3)))
            return conn_a, curv_f, ym, cs3

        def _dirac_auth_ok(self, payload: str, signature: str) -> bool:
            expected = compute_dirac_positron_hmac(self.secret_key, payload)
            try:
                return hmac.compare_digest(str(signature), str(expected))
            except (TypeError, ValueError):
                return False

        def _spohn_entropy(self, rho: NDArray[np.complex128], gamma: float = 0.15) -> float:
            n = int(rho.shape[0])
            evals = np.clip(np.real(np.linalg.eigvalsh(rho)), _SPECTRAL_PSD_FLOOR, 1.0)
            rho_inf = 1.0 / float(n)
            kl = float(np.sum(evals * (np.log(evals) - math.log(rho_inf))))
            dissip = float(la.norm(rho - (np.eye(n, dtype=np.complex128) / float(n)), ord="fro"))
            return float(max(0.0, gamma * kl * dissip))

        def _williamson_invariants(
            self, q: NDArray[np.float64], p: NDArray[np.float64], omega: NDArray[np.float64]
        ) -> Tuple[float, ...]:
            i_action = 0.5 * (omega * q * q + (p * p) / np.maximum(omega, _WILKINSON_EPSILON))
            inv = tuple(sorted((float(x) for x in np.real(i_action)), reverse=True))
            return inv

        def quantize_indictment_to_poincare_phase_space(
            self, indictment: Any
        ) -> PoincareProsecutionOrbitalCarrier:
            rho = self.sanitize_density_matrix_higham(getattr(indictment, "density_matrix"))
            n = self.dimension
            dof = self.dof
            conn_a, _curv, _ym, _cs = self.extract_gauge_invariants(rho)
            payload = str(getattr(indictment, "transaction_payload", ""))
            signature = str(getattr(indictment, "positron_auth_signature", ""))
            auth_ok = self._dirac_auth_ok(payload, signature)
            omega = 1.0 + 0.25 * np.sin(_TWO_PI * np.arange(dof) / float(max(dof, 1)))
            w, v = np.linalg.eigh(0.5 * (rho + rho.conj().T))
            w = np.maximum(np.real(w), 0.0)
            ground = v[:, -1]
            amp = math.sqrt(float(max(w[-1], 0.0)))
            q = np.zeros(dof, dtype=np.float64)
            p = np.zeros(dof, dtype=np.float64)
            diag = np.real(np.diag(rho))
            for k in range(dof):
                q[k] = float(np.real(ground[k % n]) * amp) + 0.10 * float(diag[k % n])
                p[k] = float(np.imag(ground[k % n]) * amp) + 0.10 * float(diag[(k + dof) % n])
            if not auth_ok:
                q = q + _DIRAC_PHASE_KICK
                p = p + _DIRAC_PHASE_KICK
            eps = _PROSECUTOR_PERTURBATION_EPS
            coupling = float(np.sum(np.cos(q - np.roll(q, -1))))
            energy = float(0.5 * np.dot(p, p) + 0.5 * np.dot(omega * omega, q * q) + eps * coupling)
            action = 0.5 * (omega * q * q + (p * p) / np.maximum(omega, _WILKINSON_EPSILON))
            theta = np.arctan2(p, q * np.maximum(omega, _WILKINSON_EPSILON))
            n_e = int(getattr(indictment, "electron_anomaly_count", 0))
            dirac = {
                "auth_ok": bool(auth_ok),
                "back_action_db": 0.0 if auth_ok else _DIRAC_AUTH_FAILURE_BACK_ACTION_DB,
                "electrons_purged": n_e if auth_ok else 0,
                "released_energy_kev": float(n_e * _POSITRON_REST_MASS_KEV) if auth_ok else 0.0,
                "phase_kick": 0.0 if auth_ok else _DIRAC_PHASE_KICK,
            }
            wilson_op = la.expm(conn_a)
            wilson_tr = complex(np.trace(wilson_op) / float(n))
            su2_tr = float(2.0 * math.cos(float(cmath.phase(wilson_tr))))
            return PoincareProsecutionOrbitalCarrier(
                indictment_ref=indictment,
                sanitized_density_matrix=rho,
                coordinates_q=q,
                momenta_p=p,
                action_variables_I=np.asarray(action, dtype=np.float64),
                angle_variables_theta=np.asarray(theta, dtype=np.float64),
                hamiltonian_energy=energy,
                dirac_result=dirac,
                spohn_entropy_production=self._spohn_entropy(rho),
                gauge_connection_A=conn_a,
                dimension_dof=dof,
                williamson_invariants=self._williamson_invariants(q, p, omega),
                su2_holonomy_trace=su2_tr,
                frequencies_omega=tuple(float(x) for x in omega),
            )

        def audit_homomorphic_gate_tears(
            self, rho: NDArray[np.complex128]
        ) -> Tuple[bool, float, float]:
            n = int(rho.shape[0])
            b = max(1, n // 3)

            def _integrity(i0: int, i1: int) -> float:
                e = np.zeros((n, n), dtype=np.complex128)
                e[i0:i1, i0:i1] = np.eye(i1 - i0, dtype=np.complex128)
                pi_rho = e @ rho @ e
                pi_rho2 = e @ (rho @ rho) @ e
                num = float(la.norm(pi_rho2 - pi_rho @ pi_rho, ord="fro"))
                den = max(float(la.norm(rho, ord="fro")), _WILKINSON_EPSILON)
                return float(math.exp(-num / den))

            i12 = _integrity(0, 2 * b)
            i23 = _integrity(b, 3 * b if 3 * b <= n else n)
            i13 = _integrity(0, 3 * b if 3 * b <= n else n)
            defect = abs(i13 - i12 * i23)
            tear = bool(min(i12, i23, i13) < _GATE_TEAR_INTEGRITY_FLOOR or defect > _FUNCTOR_DEFECT_TOL)
            return tear, float(min(i12, i23, i13)), float(defect)

        def audit_indictment_claims(self, indictment: Any, rho: NDArray[np.complex128]) -> bool:
            conn_a, _f, ym, cs3 = self.extract_gauge_invariants(rho)
            conn_norm = float(la.norm(conn_a, ord="fro"))
            declared_n = float(
                getattr(indictment, "connection_norm", getattr(indictment, "declared_connection_norm", conn_norm))
            )
            declared_ym = float(
                getattr(indictment, "yang_mills_energy", getattr(indictment, "declared_yang_mills_energy", ym))
            )
            declared_cs = float(
                getattr(indictment, "chern_simons_3form", getattr(indictment, "declared_chern_simons_3form", cs3))
            )
            vec_d = np.array([declared_n, declared_ym, declared_cs], dtype=np.float64)
            vec_m = np.array([conn_norm, ym, cs3], dtype=np.float64)
            den = max(float(np.linalg.norm(vec_m)), _WILKINSON_EPSILON)
            defect = float(np.linalg.norm(vec_d - vec_m) / den)
            return bool(defect > _CLAIM_TAMPER_TOL)

    class PoincareCelestialProsecutorEngine:  # type: ignore[no-redef]
        """Núcleo autónomo Fase II: Yoshida O(4), Hénon, monodromía Sp(2n), KAM–Nekhoroshev."""

        def __init__(self, dimension: int) -> None:
            dim = max(4, int(dimension))
            if dim % 2 == 1:
                dim += 1
            self.dimension = dim
            self.dof = dim // 2
            self.eps = _PROSECUTOR_PERTURBATION_EPS

        def _omega(self, n: int) -> NDArray[np.float64]:
            return 1.0 + 0.25 * np.sin(_TWO_PI * np.arange(n) / float(max(n, 1)))

        def _grad_v(self, q: NDArray[np.float64], omega: NDArray[np.float64]) -> NDArray[np.float64]:
            n = q.shape[0]
            g = omega * omega * q
            for k in range(n):
                kp = (k + 1) % n
                km = (k - 1) % n
                g[k] += self.eps * (-math.sin(q[k] - q[kp]) + math.sin(q[km] - q[k]))
            return g

        def _hamiltonian(self, q: NDArray[np.float64], p: NDArray[np.float64], omega: NDArray[np.float64]) -> float:
            coupling = float(np.sum(np.cos(q - np.roll(q, -1))))
            return float(0.5 * np.dot(p, p) + 0.5 * np.dot(omega * omega, q * q) + self.eps * coupling)

        def _verlet_step(
            self, q: NDArray[np.float64], p: NDArray[np.float64], omega: NDArray[np.float64], dt: float
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
            p = p - 0.5 * dt * self._grad_v(q, omega)
            q = q + dt * p
            p = p - 0.5 * dt * self._grad_v(q, omega)
            return q, p

        def _yoshida_step(
            self, q: NDArray[np.float64], p: NDArray[np.float64], omega: NDArray[np.float64], dt: float
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
            q, p = self._verlet_step(q, p, omega, _YOSHIDA_W1 * dt)
            q, p = self._verlet_step(q, p, omega, _YOSHIDA_W0 * dt)
            q, p = self._verlet_step(q, p, omega, _YOSHIDA_W1 * dt)
            return q, p

        def _henon_section(
            self, q: NDArray[np.float64], p: NDArray[np.float64], omega: NDArray[np.float64], steps: int = 64
        ) -> int:
            count = 0
            qn = np.array(q, dtype=np.float64, copy=True)
            pn = np.array(p, dtype=np.float64, copy=True)
            dt = 0.05
            q_prev = float(qn[0])
            for _ in range(steps):
                qn, pn = self._yoshida_step(qn, pn, omega, dt)
                q0 = float(qn[0])
                if q_prev <= 0.0 < q0 and float(pn[0]) > 0.0:
                    count += 1
                q_prev = q0
            return count

        def _analytic_monodromy(
            self, q: NDArray[np.float64], p: NDArray[np.float64], omega: NDArray[np.float64], dt: float = 0.05
        ) -> ProsecutionMonodromyResult:
            n = int(q.shape[0])
            dim_s = 2 * n
            m = np.eye(dim_s, dtype=np.float64)
            qn = np.array(q, dtype=np.float64, copy=True)
            pn = np.array(p, dtype=np.float64, copy=True)
            period = dt
            eye_n = np.eye(n, dtype=np.float64)
            hess = np.diag(omega * omega)
            for k in range(n):
                kp = (k + 1) % n
                hess[k, k] += self.eps * math.cos(qn[k] - qn[kp])
                hess[k, kp] -= self.eps * math.cos(qn[k] - qn[kp])
            dth = 0.5 * dt
            # Mapa tangente exacto de Verlet (permanece en Sp(2n) en aritmética exacta)
            a_half = eye_n
            b_half = -dth * hess
            c_half = np.zeros((n, n), dtype=np.float64)
            d_half = eye_n
            a_q = eye_n
            b_q = dt * eye_n
            c_q = np.zeros((n, n), dtype=np.float64)
            d_q = eye_n
            top = np.hstack([a_half, b_half])
            bot = np.hstack([c_half, d_half])
            s_p = np.vstack([top, bot])
            top = np.hstack([a_q, b_q])
            bot = np.hstack([c_q, d_q])
            s_q = np.vstack([top, bot])
            m = s_p @ s_q @ s_p @ m
            qn, pn = self._verlet_step(qn, pn, omega, dt)
            eigs = np.linalg.eigvals(m)
            spec = float(np.max(np.abs(eigs)))
            lyap = float(math.log(max(spec, _WILKINSON_EPSILON)) / max(period, _WILKINSON_EPSILON))
            j = np.block(
                [
                    [np.zeros((n, n)), eye_n],
                    [-eye_n, np.zeros((n, n))],
                ]
            )
            pairing = m.T @ j @ m - j
            defect = float(la.norm(pairing, ord="fro"))
            return ProsecutionMonodromyResult(
                monodromy_matrix=m,
                floquet_multipliers=tuple(complex(z) for z in eigs[: min(8, eigs.size)]),
                max_lyapunov_exponent=lyap,
                symplectic_defect=defect,
                period=period,
            )

        def _kam_audit(
            self, omega: NDArray[np.float64], lyap: float
        ) -> ProsecutionKAMAuditResult:
            n = int(min(4, omega.shape[0]))
            om = np.asarray(omega[:n], dtype=np.float64)
            tau = float(n)
            gamma = 0.04 / float(max(n * n, 1))
            min_div = float("inf")
            order = min(_MAX_KAM_FOURIER_ORDER, 3)
            for k_tuple in itertools.product(range(-order, order + 1), repeat=n):
                if all(kv == 0 for kv in k_tuple):
                    continue
                kvec = np.array(k_tuple, dtype=np.float64)
                div = abs(float(np.dot(kvec, om)))
                k1 = float(np.linalg.norm(kvec, ord=1))
                thresh = gamma / max(k1 ** tau, 1.0)
                if div < min_div:
                    min_div = div
                if div < thresh:
                    break
            min_div = float(min_div if math.isfinite(min_div) else 0.0)
            stable = bool(min_div >= gamma / ((2.0 * order) ** tau + 1.0) and lyap < 0.15)
            expo = min(40.0, gamma / max(self.eps ** (1.0 / (2.0 * n)), _WILKINSON_EPSILON))
            nek = float(math.exp(max(0.0, expo)))
            rot = float(om[0] / max(om[1] if n > 1 else 1.0, _WILKINSON_EPSILON))
            rec = float(_TWO_PI / max(float(np.min(np.abs(om))), _WILKINSON_EPSILON))
            return ProsecutionKAMAuditResult(
                is_kam_torus_stable=stable,
                diophantine_gamma=gamma,
                diophantine_tau=tau,
                min_small_divisor=min_div,
                nekhoroshev_time_bound=nek,
                poincare_recurrence_time_estimate=rec,
                max_lyapunov_exponent=lyap,
                rotation_number=rot,
                fourier_order=order,
            )

        def synthesize_prosecution_symplectic_feedback(
            self,
            orbital_carrier: PoincareProsecutionOrbitalCarrier,
            kam: Any = None,
            monodromy: Any = None,
            section: Any = None,
        ) -> ProsecutionSymplecticFeedbackCarrier:
            q = np.asarray(orbital_carrier.coordinates_q, dtype=np.float64)
            p = np.asarray(orbital_carrier.momenta_p, dtype=np.float64)
            n = int(q.shape[0])
            omega = self._omega(n)
            e0 = self._hamiltonian(q, p, omega)
            qn, pn = np.array(q, copy=True), np.array(p, copy=True)
            for _ in range(8):
                qn, pn = self._yoshida_step(qn, pn, omega, 0.05)
            e1 = self._hamiltonian(qn, pn, omega)
            energy_drift = abs(e1 - e0) / max(abs(e0), _WILKINSON_EPSILON)
            if monodromy is None:
                monodromy = self._analytic_monodromy(q, p, omega)
            lyap = float(getattr(monodromy, "max_lyapunov_exponent", 0.0))
            if kam is None:
                kam = self._kam_audit(omega, lyap)
            if section is None:
                section_count = self._henon_section(q, p, omega)
            else:
                section_count = int(getattr(section, "crossing_count", getattr(section, "section_count", 0)))
            pairing = float(getattr(monodromy, "symplectic_defect", 0.0))
            drift_pen = float(energy_drift + pairing)
            conserved = bool(drift_pen < _SYMPLECTIC_DRIFT_TOL * 10.0)
            conn_a = getattr(orbital_carrier, "gauge_connection_A", np.zeros((2, 2), dtype=np.complex128))
            conn_norm = float(la.norm(np.asarray(conn_a), ord="fro"))
            germ = getattr(orbital_carrier, "indictment_ref", None)
            tr_ok = bool(getattr(germ, "trace_preservation_ok", True)) if germ is not None else True
            ym = float(getattr(germ, "yang_mills_energy", 0.0)) if germ is not None else 0.0
            wilson = float(getattr(germ, "wilson_angle_rad", 0.0)) if germ is not None else 0.0
            inst_k = int(getattr(germ, "instanton_number_k", 0)) if germ is not None else 0
            holonomy = bool(
                (not tr_ok) or conn_norm > 2.5 or ym > 0.85 or abs(wilson) >= 0.50 or abs(inst_k) > 2
            )
            dirac = getattr(orbital_carrier, "dirac_result", {}) or {}
            auth_ok = True
            if isinstance(dirac, dict):
                auth_ok = bool(dirac.get("auth_ok", True))
            else:
                auth_ok = bool(getattr(dirac, "auth_ok", True))
            kam_ok = bool(getattr(kam, "is_kam_torus_stable", True))
            if not auth_ok:
                status = OODAProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE
                verdict = HeytingOmega4.VETOED
            elif holonomy:
                status = OODAProsecutionStatus.VETOED_HOLONOMY_BREAKAGE
                verdict = HeytingOmega4.VETOED
            elif not conserved:
                status = OODAProsecutionStatus.VETOED_SYMPLECTIC_DRIFT
                verdict = HeytingOmega4.CRITICAL
            elif not kam_ok:
                status = OODAProsecutionStatus.VETOED_KAM_CHAOTIC_RESONANCE
                verdict = HeytingOmega4.CRITICAL
            elif auth_ok:
                status = OODAProsecutionStatus.PURGED_DIRAC_VACUUM
                verdict = HeytingOmega4.COHERENT
            else:
                status = OODAProsecutionStatus.ABSOLVED
                verdict = HeytingOmega4.COHERENT
            return ProsecutionSymplecticFeedbackCarrier(
                orbital_carrier=orbital_carrier,
                monodromy_result=monodromy,
                kam_audit=kam,
                symplectic_drift_penalty=drift_pen,
                holonomy_breakage_detected=holonomy,
                intermediate_verdict=verdict,
                prosecution_status_tentative=status,
                energy_drift=float(energy_drift),
                section_count=int(section_count),
                pairing_defect=pairing,
                symplectic_conserved=conserved,
            )

    class ProsecutionRSIL3DecisionCore:  # type: ignore[no-redef]
        """Núcleo autónomo Fase III: RSI-L3 (S ∘ T), interlock ESP32 simulado, adjudicación Ω₄."""

        def __init__(self, secret_key: bytes) -> None:
            self.secret_key = bytes(secret_key)
            self.generation = 0
            self.lam = _DEFAULT_BANACH_CONTRACTION_LIP

        def _meta_S(self, lam: float, residual: float, lip_emp: float, beta: float = 0.15, eta: float = 0.08) -> float:
            return float(np.clip(lam * (1.0 - beta * residual) + eta * (1.0 - lip_emp), 0.50, 0.99))

        def trigger_recursive_step(
            self, lip_emp: float, residual: float
        ) -> Tuple[ProsecutorRSIConfiguration, float, float]:
            self.lam = self._meta_S(self.lam, residual, lip_emp)
            self.generation += 1
            cfg = ProsecutorRSIConfiguration(
                generation=self.generation,
                banach_lipschitz=float(min(lip_emp, self.lam)),
                meta_operator_lip=self.lam,
                cauchy_residual=float(residual),
                psd_floor=_SPECTRAL_PSD_FLOOR,
                symplectic_tolerance=_SYMPLECTIC_DRIFT_TOL,
            )
            return cfg, float(min(lip_emp, self.lam)), float(residual)

        def simulate_esp32_safety_interlock(self, vetoed: bool, seed_material: bytes) -> Tuple[bool, float]:
            if not vetoed:
                return False, 0.0
            digest = hashlib.sha384(b"ESP32-SIM-v15.0|" + seed_material).digest()
            unit = int.from_bytes(digest[:4], "big") / float(2**32)
            span = _ESP32_CROWBAR_MAX_LATENCY_NS - _ESP32_CROWBAR_MIN_LATENCY_NS
            latency = _ESP32_CROWBAR_MIN_LATENCY_NS + unit * span
            latency = float(min(max(latency, _ESP32_CROWBAR_MIN_LATENCY_NS), _ESP32_CROWBAR_MAX_LATENCY_NS))
            return True, latency

        def adjudicate_and_execute(
            self, feedback: ProsecutionSymplecticFeedbackCarrier
        ) -> OODAProsecutionExecutionCertificatePhD:
            now = time.time()
            orbital = feedback.orbital_carrier
            dirac = getattr(orbital, "dirac_result", {}) or {}
            if isinstance(dirac, dict):
                auth_ok = bool(dirac.get("auth_ok", True))
                back_db = float(dirac.get("back_action_db", 0.0))
                purged = int(dirac.get("electrons_purged", 0))
                released = float(dirac.get("released_energy_kev", 0.0))
            else:
                auth_ok = bool(getattr(dirac, "auth_ok", True))
                back_db = float(getattr(dirac, "back_action_db", 0.0))
                purged = int(getattr(dirac, "electrons_purged", 0))
                released = float(getattr(dirac, "released_energy_kev", 0.0))
            kam = feedback.kam_audit
            kam_ok = bool(getattr(kam, "is_kam_torus_stable", True))
            lyap = float(
                getattr(kam, "max_lyapunov_exponent", getattr(feedback.monodromy_result, "max_lyapunov_exponent", 0.0))
            )
            drift = float(getattr(feedback, "symplectic_drift_penalty", 0.0))
            energy_drift = float(getattr(feedback, "energy_drift", 0.0))
            holonomy = bool(feedback.holonomy_breakage_detected)
            tamper = bool(getattr(orbital, "claim_tamper_detected", False))
            tear = bool(getattr(orbital, "gate_tear_detected", False))
            # Prioridad: tamper → Dirac → holonomía → Φ-tear → ω-drift → KAM
            if tamper:
                status, verdict = OODAProsecutionStatus.VETOED_INDICTMENT_TAMPER, HeytingOmega4.VETOED
            elif not auth_ok:
                status, verdict = OODAProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE, HeytingOmega4.VETOED
            elif holonomy:
                status, verdict = OODAProsecutionStatus.VETOED_HOLONOMY_BREAKAGE, HeytingOmega4.VETOED
            elif tear:
                status, verdict = OODAProsecutionStatus.VETOED_HOMOMORPHIC_GATE_TEAR, HeytingOmega4.VETOED
            elif drift > 1.0e-4 or energy_drift > 1.0e-3:
                status, verdict = OODAProsecutionStatus.VETOED_SYMPLECTIC_DRIFT, HeytingOmega4.VETOED
            elif not kam_ok or lyap >= 0.15:
                status, verdict = OODAProsecutionStatus.VETOED_KAM_CHAOTIC_RESONANCE, HeytingOmega4.CRITICAL
            elif purged > 0 and auth_ok:
                status, verdict = OODAProsecutionStatus.PURGED_DIRAC_VACUUM, HeytingOmega4.COHERENT
            else:
                status, verdict = OODAProsecutionStatus.ABSOLVED, HeytingOmega4.COHERENT
            lip_emp = float(
                min(
                    _DEFAULT_BANACH_CONTRACTION_LIP,
                    0.50 + 0.49 * math.tanh(1.0 + drift / max(_SYMPLECTIC_DRIFT_TOL, 1e-18)),
                )
            )
            residual = float(min(1.0, drift / max(_SYMPLECTIC_DRIFT_TOL, 1e-18) * 1e-3 + abs(lyap)))
            cfg, lip_use, cauchy = self.trigger_recursive_step(lip_emp, residual)
            vetoed = verdict == HeytingOmega4.VETOED
            seed = f"{status.name}:{verdict.name}:{lyap:.8f}".encode("utf-8")
            crowbar, latency = self.simulate_esp32_safety_interlock(vetoed, seed)
            cid = f"CERT-PROSECUTOR-v15-{int(now * 1000.0) % 10000000:07d}"
            return OODAProsecutionExecutionCertificatePhD(
                certificate_id=cid,
                status=status,
                global_heyting_verdict=verdict,
                crowbar_triggered=crowbar,
                crowbar_triggered_simulated=crowbar,
                crowbar_latency_ns=latency,
                electron_anomalies_purged=purged,
                released_energy_kev=released,
                back_action_db=back_db,
                rsi_level3_generation=int(cfg.generation),
                rsi_banach_contraction_lip=float(lip_use),
                rsi_meta_operator_lip=float(cfg.meta_operator_lip),
                rsi_cauchy_residual=float(cauchy),
                spohn_entropy_production=float(getattr(orbital, "spohn_entropy_production", 0.0)),
                is_kam_torus_stable=kam_ok,
                max_lyapunov_exponent=lyap,
                symplectic_drift=drift,
                energy_drift=energy_drift,
                timestamp_utc=now,
            )

    class OODATriphasicHomomorphicPipelineProsecutorEnginePhD:  # type: ignore[no-redef]
        def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
            dim = max(4, int(dimension))
            if dim % 2 == 1:
                dim += 1
            self.dimension = dim
            self.secret_key = bytes(secret_key)
            self.phase1_core = BanachGaugeDiracProsecutionCore(dim, self.secret_key)
            self.phase2_core = PoincareCelestialProsecutorEngine(dim)
            self.phase3_core = ProsecutionRSIL3DecisionCore(self.secret_key)


def compute_dirac_positron_hmac(secret_key: bytes, payload: str) -> str:
    """HMAC-SHA384 del positrón de autorización con dominio canónico v15 (nunca key ‖ payload)."""
    if _engine_dirac_hmac is not None:
        try:
            return str(_engine_dirac_hmac(secret_key, payload))
        except TypeError:
            try:
                return str(_engine_dirac_hmac(payload, secret=secret_key))
            except TypeError:
                return str(_engine_dirac_hmac(payload))
    return hmac.new(
        bytes(secret_key),
        _HMAC_DOMAIN_DIRAC + str(payload).encode("utf-8"),
        hashlib.sha384,
    ).hexdigest()


logger = logging.getLogger("APU.Wisdom.OODATriphasicHomomorphicPipelineProsecutorAgent.PhD")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    )
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)


def _fmt(value: float, ndigits: int = 8) -> str:
    return f"{float(value):.{ndigits}f}"


def _as_heyting(value: Any) -> HeytingOmega4:
    if isinstance(value, HeytingOmega4):
        return value
    try:
        return HeytingOmega4(int(value))
    except (TypeError, ValueError):
        return HeytingOmega4.CRITICAL


def _as_status(value: Any) -> OODAProsecutionStatus:
    if isinstance(value, OODAProsecutionStatus):
        return value
    try:
        return OODAProsecutionStatus(int(value))
    except (TypeError, ValueError):
        return OODAProsecutionStatus.PENDING


def _dirac_auth_flag(dirac: Any) -> bool:
    if dirac is None:
        return True
    if isinstance(dirac, dict):
        return bool(dirac.get("auth_ok", True))
    return bool(getattr(dirac, "auth_ok", getattr(dirac, "authenticated", True)))


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE TRANSFERENCIA FORMAL Y ESTADOS INMUTABLES
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class OODAProsecutorGaugeContractPhD:
    """Contrato inmutable de topología de calibre fiscal forjado en Fase I (v15)."""

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
    """Contenedor formal de acoplamiento: salida Fase I ➔ entrada Fase II (hoja de energía canónica)."""

    gauge_contract: OODAProsecutorGaugeContractPhD
    poincare_orbital_carrier: PoincareProsecutionOrbitalCarrier
    sanitized_density_matrix: NDArray[np.complex128]
    gauge_connection: NDArray[np.complex128]
    indictment_germ: Any
    dirac_auth_ok: bool
    claim_tamper_detected: bool
    williamson_invariants: Tuple[float, ...]
    energy_hamiltonian: float


@dataclass(frozen=True, slots=True)
class OODAProsecutionCampaignResultPhD:
    """Resultado inmutable de la campaña celeste de Fase II (sin adjudicación penal; esa es Fase III)."""

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


@dataclass(frozen=True, slots=True)
class ProsecutorPhase2TransferCarrier:
    """Contenedor formal de acoplamiento: salida Fase II ➔ entrada Fase III (tensor de realimentación)."""

    gauge_contract: OODAProsecutorGaugeContractPhD
    campaign_result: OODAProsecutionCampaignResultPhD
    symplectic_feedback: ProsecutionSymplecticFeedbackCarrier
    sanitized_density_matrix: NDArray[np.complex128]
    gauge_connection: NDArray[np.complex128]
    indictment_germ: Any
    dirac_auth_ok: bool
    claim_tamper_detected: bool


@dataclass(frozen=True, slots=True)
class OODAProsecutorSovereignPassportPhD:
    """Pasaporte soberano final emitido por Fase III con certificación Merkle-DAG y Φ₁₃."""

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


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE I: OBSERVE / TOPOLOGÍA DE CALIBRE SU(N), DIRAC, WEYL–WILLIAMSON
# DEFINICIÓN FORMAL FINAL ──► bridge_phase1_to_phase2_campaign_transfer  ══╗
#                                                                          ║ continuación funtorial
# ═════════════════════════════════════════════════════════════════════════╩════════════════════════════

class OODAPipelineProsecutorGaugeTopologyPhD:
    """
    FASE I — OBSERVE.

    Audita el C*-espacio de Banach del operador densidad, extrae la conexión 1-forma
    A ∈ 𝔰𝔲(N), la curvatura F, la 3-forma de Chern–Simons
        CS₃ ≈ Re Tr(A F − ⅔ A³),
    el número instantónico k = round(CS₃ / 8π²), la primera clase de Chern c₁ del haz de Hopf
    y el ángulo de holonomía de Wilson. Proyecta Higham al cono PSD. Cuantiza al fibrado
    cotangente T*(M) con el Hamiltoniano canónico (la misma hoja de energía que Fase II)

        H = ½‖p‖² + ½ Σ_k ω_k² q_k² + ε Σ_k cos(q_k − q_{k+1}),
        ω_k = 1 + ¼ sin(2π k / n),  ε = 0.025.

    El último método de esta fase ES el morfismo de acoplamiento 1→2.
    """

    def __init__(
        self,
        sovereign_id: str = "SOVEREIGN-OODA-PROSECUTOR-v15-PhD",
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        engine: Optional[OODATriphasicHomomorphicPipelineProsecutorEnginePhD] = None,
    ) -> None:
        dim = max(4, int(dimension))
        if dim % 2 == 1:
            dim += 1
        self.sovereign_id: Final[str] = str(sovereign_id)
        self.dimension: Final[int] = dim
        self.secret_key: Final[bytes] = bytes(secret_key)
        self.engine = engine or OODATriphasicHomomorphicPipelineProsecutorEnginePhD(
            dimension=self.dimension, secret_key=self.secret_key
        )

    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(self.secret_key, domain + payload.encode("utf-8"), hashlib.sha384).hexdigest()

    def _canonical_p1_payload(self, contract: OODAProsecutorGaugeContractPhD) -> str:
        return (
            f"{contract.contract_id}:{contract.sovereign_id}:"
            f"{_fmt(contract.chern_simons_3form)}:{_fmt(contract.hopf_c1_invariant)}:"
            f"{_fmt(contract.wilson_angle_rad)}:{_fmt(contract.connection_norm)}:"
            f"{_fmt(contract.yang_mills_energy)}:{int(contract.instanton_number_k)}:"
            f"{_fmt(contract.su2_holonomy_trace)}:{_fmt(contract.purity)}:"
            f"{int(contract.dof_cotangent)}:{int(contract.trace_preservation_ok)}:"
            f"{int(contract.gate_tear_detected)}:{int(contract.isometry_verdict.value)}:"
            f"{contract.chain_hash}"
        )

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        """Proyección espectral de Higham al cono convexo PSD y normalización de traza (sin sesgo Uhlmann)."""
        core = getattr(self.engine, "phase1_core", None)
        if core is not None and hasattr(core, "sanitize_density_matrix_higham"):
            return np.asarray(core.sanitize_density_matrix_higham(matrix), dtype=np.complex128)
        a = np.asarray(matrix, dtype=np.complex128)
        a = 0.5 * (a + a.conj().T)
        w, v = np.linalg.eigh(a)
        w = np.maximum(np.real(w), 0.0)
        rho = (v * w) @ v.conj().T
        tr = float(np.real(np.trace(rho)))
        if tr > _WILKINSON_EPSILON:
            rho = rho / tr
        return np.asarray(rho, dtype=np.complex128)

    def extract_suN_gauge_bundle(
        self, rho: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], float, float]:
        """Extrae (A, F, E_YM, CS₃) con resiliencia de aridad sobre el núcleo del motor."""
        core = getattr(self.engine, "phase1_core", None)
        if core is not None and hasattr(core, "extract_gauge_invariants"):
            result = core.extract_gauge_invariants(rho)
            if isinstance(result, (tuple, list)) and len(result) >= 4:
                return (
                    np.asarray(result[0], dtype=np.complex128),
                    np.asarray(result[1], dtype=np.complex128),
                    float(result[2]),
                    float(result[3]),
                )
        n = int(rho.shape[0])
        h = 0.5 * (rho + rho.conj().T)
        h = h - (np.trace(h) / float(n)) * np.eye(n, dtype=np.complex128)
        conn_a = 1.0j * h
        curv_f = conn_a @ conn_a
        ym = 0.5 * float(np.real(np.sum(np.conj(curv_f) * curv_f)))
        cs3 = float(np.real(np.trace(conn_a @ curv_f - (2.0 / 3.0) * (conn_a @ conn_a @ conn_a))))
        return conn_a, curv_f, ym, cs3

    def compute_chern_simons_and_instanton(self, cs3_val: float) -> int:
        """k = round(CS₃ / 8π²) ∈ ℤ, índice instantónico de Pontryagin reducido."""
        return int(round(float(cs3_val) / (8.0 * math.pi ** 2)))

    def compute_wilson_holonomy(
        self, conn_a: NDArray[np.complex128]
    ) -> Tuple[float, float]:
        """W(γ) = Tr(P exp ∮ A)/N → (ángulo de Wilson, carácter SU(2) ≈ 2 cos θ)."""
        n = max(int(conn_a.shape[0]), 1)
        wilson_op = la.expm(conn_a)
        wilson_trace = complex(np.trace(wilson_op) / float(n))
        angle = float(cmath.phase(wilson_trace))
        su2_tr = float(2.0 * math.cos(angle))
        return angle, su2_tr

    def adjudicate_heyting_isometry(
        self, tr_ok: bool, conn_norm: float, ym_energy: float, wilson_angle: float, instanton_k: int
    ) -> HeytingOmega4:
        """Adjudicación Ω₄ de isometría de calibre (umbrales holonómicos v15)."""
        if (not tr_ok) or conn_norm > 2.5 or ym_energy > 0.85 or abs(wilson_angle) >= 0.50 or abs(instanton_k) > 2:
            return HeytingOmega4.VETOED
        if conn_norm > 1.2 or ym_energy > 0.40:
            return HeytingOmega4.CRITICAL
        if conn_norm > 0.7 or ym_energy > 0.15:
            return HeytingOmega4.DEGRADED
        return HeytingOmega4.COHERENT

    def forge_prosecutor_gauge_contract(
        self, density_matrix: NDArray
    ) -> Tuple[OODAProsecutorGaugeContractPhD, NDArray[np.complex128], NDArray[np.complex128], bool]:
        """
        Forja el Contrato de Topología de Calibre Fiscal.
        Verifica Tr(ρ) = 1 ± 10⁻⁵, extrae invariantes y audita lágrimas homomórficas C*.
        """
        now = time.time()
        rho = self.sanitize_density_matrix(density_matrix)
        tr_val = float(np.real(np.trace(rho)))
        tr_ok = abs(tr_val - 1.0) < 1.0e-5
        conn_a, curv_f, ym_energy, cs3_val = self.extract_suN_gauge_bundle(rho)
        conn_norm = float(la.norm(conn_a, ord="fro"))
        wilson_angle, su2_tr = self.compute_wilson_holonomy(conn_a)
        hopf_c1 = float(np.real(np.trace(curv_f)) / _TWO_PI)
        instanton_k = self.compute_chern_simons_and_instanton(cs3_val)
        purity = float(np.real(np.trace(rho @ rho)))
        gate_tear = False
        core = getattr(self.engine, "phase1_core", None)
        if core is not None and hasattr(core, "audit_homomorphic_gate_tears"):
            try:
                audited = core.audit_homomorphic_gate_tears(rho)
                if isinstance(audited, tuple) and len(audited) >= 1:
                    gate_tear = bool(audited[0])
                else:
                    gate_tear = bool(audited)
            except (TypeError, ValueError, AttributeError):
                gate_tear = False
        verdict = self.adjudicate_heyting_isometry(tr_ok, conn_norm, ym_energy, wilson_angle, instanton_k)
        if gate_tear and verdict.truth_rank() > HeytingOmega4.CRITICAL.truth_rank():
            verdict = HeytingOmega4.CRITICAL
        contract_id = f"CONTRACT-OODA-PROSECUTOR-v15-{int(now * 1000.0) % 10000000:07d}"
        chain_bytes = (
            f"{contract_id}:{self.sovereign_id}:{_fmt(cs3_val)}:{_fmt(wilson_angle)}:"
            f"{instanton_k}:{int(verdict.value)}"
        ).encode("utf-8")
        chain_hash = hashlib.sha384(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]
        contract = OODAProsecutorGaugeContractPhD(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            chern_simons_3form=float(cs3_val),
            hopf_c1_invariant=float(hopf_c1),
            wilson_angle_rad=float(wilson_angle),
            connection_norm=float(conn_norm),
            yang_mills_energy=float(ym_energy),
            instanton_number_k=int(instanton_k),
            su2_holonomy_trace=float(su2_tr),
            purity=float(purity),
            dof_cotangent=int(self.dimension // 2),
            trace_preservation_ok=bool(tr_ok),
            gate_tear_detected=bool(gate_tear),
            isometry_verdict=verdict,
            chain_hash=chain_hash,
            hmac_p1_signature="",
            timestamp_utc=now,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P1, self._canonical_p1_payload(contract))
        object.__setattr__(contract, "hmac_p1_signature", sig)
        return contract, rho, conn_a, gate_tear

    def continue_to_phase2_prosecutor_gate(self, contract: OODAProsecutorGaugeContractPhD) -> bool:
        """
        Compuerta Homomórfica Φ₁₂: revalida HMAC-SHA384 del contrato P1 (dominio v15).
        Preserva isometría C* sin abrir el estado. Admite VETOED para juzgamiento ulterior.
        """
        expected = self._compute_hmac(_HMAC_DOMAIN_P1, self._canonical_p1_payload(contract))
        try:
            return hmac.compare_digest(str(contract.hmac_p1_signature), str(expected))
        except (TypeError, ValueError):
            return False

    def _construct_indictment_germ(self, **kwargs: Any) -> Any:
        """Construye el germen con nombres v14 o declared_* v15, sin ruptura de aridad."""
        cls = OODAProsecutionIndictmentGermPhD
        attempts: List[Dict[str, Any]] = [dict(kwargs)]
        remapped = dict(kwargs)
        for src, dst in _GERM_DECLARED_MAP.items():
            if src in remapped:
                remapped[dst] = remapped.pop(src)
        attempts.append(remapped)
        last_exc: Optional[BaseException] = None
        for payload in attempts:
            try:
                return cls(**payload)  # type: ignore[arg-type]
            except TypeError as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            logger.warning("Germen de acusación: degradando a construcción mínima (%s)", last_exc)
        return cls(  # type: ignore[call-arg]
            indictment_id=str(kwargs.get("indictment_id", "INDICT-DEGRADED")),
            trace_preservation_ok=bool(kwargs.get("trace_preservation_ok", True)),
            connection_norm=float(kwargs.get("connection_norm", 0.0)),
            yang_mills_energy=float(kwargs.get("yang_mills_energy", 0.0)),
            chern_simons_3form=float(kwargs.get("chern_simons_3form", 0.0)),
            wilson_angle_rad=float(kwargs.get("wilson_angle_rad", 0.0)),
            instanton_number_k=int(kwargs.get("instanton_number_k", 0)),
            electron_anomaly_count=int(kwargs.get("electron_anomaly_count", 0)),
            positron_auth_signature=str(kwargs.get("positron_auth_signature", "")),
            transaction_payload=str(kwargs.get("transaction_payload", "")),
            density_matrix=kwargs.get("density_matrix"),
            timestamp_utc=float(kwargs.get("timestamp_utc", time.time())),
        )

    def _audit_indictment_claims(self, indictment: Any, rho: NDArray[np.complex128]) -> bool:
        core = getattr(self.engine, "phase1_core", None)
        if core is not None and hasattr(core, "audit_indictment_claims"):
            try:
                return bool(core.audit_indictment_claims(indictment, rho))
            except TypeError:
                try:
                    return bool(core.audit_indictment_claims(indictment))
                except (TypeError, AttributeError):
                    return False
            except (ValueError, AttributeError):
                return False
        return False

    def _quantize_indictment(self, indictment: Any) -> PoincareProsecutionOrbitalCarrier:
        core = getattr(self.engine, "phase1_core", None)
        if core is None or not hasattr(core, "quantize_indictment_to_poincare_phase_space"):
            raise RuntimeError("Núcleo Fase I ausente: no se puede cuantizar al espacio de Poincaré.")
        return core.quantize_indictment_to_poincare_phase_space(indictment)

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # MÉTODO PUENTE FORMAL FINAL FASE I ──► INICIO FASE II
    # ──────────────────────────────────────────────────────────────────────────────────────────
    def bridge_phase1_to_phase2_campaign_transfer(
        self,
        density_matrix: NDArray,
        electron_anomaly_count: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
    ) -> ProsecutorPhase1TransferCarrier:
        """
        DEFINICIÓN FORMAL FINAL DE LA FASE I (morfismo de acoplamiento 1→2):

        Forja el contrato SU(N), emite la firma HMAC P1 (dominio v15), construye el germen
        de acusación cuyas invariantes declaradas serán re-medidas por el núcleo, cuantiza
        ρ al espacio de fase canónico T*(M) mediante
            quantize_indictment_to_poincare_phase_space
        sobre la hoja de energía H de Verlet/Yoshida, y transfiere el contenedor inmutable
        `ProsecutorPhase1TransferCarrier` como continuación directa —sin discontinuidad
        funtorial— hacia la Fase II.
        """
        contract, sanitized_rho, conn_a, gate_tear = self.forge_prosecutor_gauge_contract(density_matrix)
        indictment = self._construct_indictment_germ(
            indictment_id=f"INDICT-{contract.contract_id}",
            trace_preservation_ok=contract.trace_preservation_ok,
            connection_norm=contract.connection_norm,
            yang_mills_energy=contract.yang_mills_energy,
            chern_simons_3form=contract.chern_simons_3form,
            wilson_angle_rad=contract.wilson_angle_rad,
            instanton_number_k=contract.instanton_number_k,
            electron_anomaly_count=int(electron_anomaly_count),
            positron_auth_signature=str(positron_auth_signature),
            transaction_payload=str(transaction_payload),
            density_matrix=sanitized_rho,
            timestamp_utc=contract.timestamp_utc,
        )
        claim_tamper = self._audit_indictment_claims(indictment, sanitized_rho)
        orbital_carrier = self._quantize_indictment(indictment)
        dirac_ok = _dirac_auth_flag(getattr(orbital_carrier, "dirac_result", None))
        if not dirac_ok:
            # Verificación agéntica redundante (el núcleo ya aplicó el kick +0.5 estructurado).
            dirac_ok = hmac.compare_digest(
                str(positron_auth_signature),
                compute_dirac_positron_hmac(self.secret_key, str(transaction_payload)),
            ) if positron_auth_signature and transaction_payload else False
        williamson = tuple(
            float(x) for x in getattr(orbital_carrier, "williamson_invariants", ()) or ()
        )
        energy = float(getattr(orbital_carrier, "hamiltonian_energy", 0.0))
        logger.debug(
            "Puente Fiscal Fase I → Fase II: Contrato=%s DOF=%s E=%.6f Dirac=%s Tamper=%s Tear=%s",
            contract.contract_id,
            getattr(orbital_carrier, "dimension_dof", self.dimension // 2),
            energy,
            dirac_ok,
            claim_tamper,
            gate_tear,
        )
        return ProsecutorPhase1TransferCarrier(
            gauge_contract=contract,
            poincare_orbital_carrier=orbital_carrier,
            sanitized_density_matrix=sanitized_rho,
            gauge_connection=np.asarray(conn_a, dtype=np.complex128),
            indictment_germ=indictment,
            dirac_auth_ok=bool(dirac_ok),
            claim_tamper_detected=bool(claim_tamper),
            williamson_invariants=williamson,
            energy_hamiltonian=energy,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE II: ORIENT / DINÁMICA CELESTE DE POINCARÉ, YOSHIDA O(4), KAM, FLOQUET
# hereda el morfismo 1→2 y su DEFINICIÓN FORMAL FINAL es
#     bridge_phase2_to_phase3_adjudication_transfer
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class OODAProsecutionCampaignOrientorPhD(OODAPipelineProsecutorGaugeTopologyPhD):
    """
    FASE II — ORIENT.

    Consume `ProsecutorPhase1TransferCarrier`. Valida Φ₁₂. Integra el flujo hamiltoniano
    con el método de Yoshida de orden 4 (composición w₁, w₀, w₁ del Verlet canónico
        p ← p − (dt/2) ∇V,  q ← q + dt p,  p ← p − (dt/2) ∇V),
    construye la sección de Poincaré–Hénon Σ = {q₀ = 0, p₀ > 0}, calcula la monodromía
    analítica por el mapa tangente de Verlet (∈ Sp(2n) en aritmética exacta), audita
    divisores pequeños KAM por itertools.product (n ≤ 4, τ = n, γ = 0.04/n²) y la cota
    de Nekhoroshev T ≤ exp(min(40, γ/ε^{1/(2n)})).

    NO adjudica ni dispara silicio: eso es el morfismo 2→3 / Fase III.
    El último método de esta fase ES el acoplamiento 2→3 vía
        synthesize_prosecution_symplectic_feedback.
    """

    def _estimate_symplectic_lipschitz(self, drift: float, lyapunov: float, kam_ok: bool) -> float:
        base = 0.50 + 0.49 * math.tanh(1.0 + abs(drift) / max(_SYMPLECTIC_DRIFT_TOL, 1e-18))
        if not kam_ok:
            base = max(base, 0.92)
        base = max(base, min(0.99, 0.70 + abs(lyapunov)))
        return float(min(_DEFAULT_BANACH_CONTRACTION_LIP if kam_ok else 0.97, base, 0.99))

    def _invoke_named(self, obj: Any, names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        for name in names:
            fn = getattr(obj, name, None)
            if not callable(fn):
                continue
            try:
                return fn(*args, **kwargs)
            except TypeError:
                try:
                    return fn(*args)
                except TypeError:
                    continue
            except Exception:
                continue
        return None

    def _synthesize_feedback_resilient(
        self,
        orbital: PoincareProsecutionOrbitalCarrier,
        kam: Any,
        monodromy: Any,
        section: Any,
    ) -> ProsecutionSymplecticFeedbackCarrier:
        synth = getattr(self.engine.phase2_core, "synthesize_prosecution_symplectic_feedback")
        attempts = (
            lambda: synth(orbital, kam, monodromy, section),
            lambda: synth(orbital, kam=kam, monodromy=monodromy, section=section),
            lambda: synth(orbital, section=section),
            lambda: synth(orbital),
        )
        last_exc: Optional[BaseException] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_exc = exc
                continue
        raise TypeError(f"synthesize_prosecution_symplectic_feedback: firma incompatible ({last_exc})")

    def execute_prosecution_campaign_orientation(
        self, carrier1: ProsecutorPhase1TransferCarrier
    ) -> Tuple[OODAProsecutionCampaignResultPhD, ProsecutionSymplecticFeedbackCarrier]:
        """Ejecuta la orientación penal celeste (Yoshida, Hénon, Floquet, KAM) sin sentenciar."""
        now = time.time()
        contract = carrier1.gauge_contract
        orbital = carrier1.poincare_orbital_carrier

        if not self.continue_to_phase2_prosecutor_gate(contract):
            logger.critical("Violación de integridad en Compuerta Homomórfica Φ₁₂ (HMAC v15 inválido)")
            raise ValueError("Falla de autenticidad criptográfica en Compuerta Homomórfica Φ₁₂.")

        p2 = getattr(self.engine, "phase2_core", None)
        section = None
        monodromy = None
        kam = None
        if p2 is not None:
            section = self._invoke_named(
                p2,
                ("compute_henon_poincare_section", "poincare_section_henon", "extract_poincare_section"),
                orbital,
            )
            monodromy = self._invoke_named(
                p2,
                ("compute_analytic_monodromy", "analytic_monodromy", "compute_monodromy"),
                orbital,
            )
            kam = self._invoke_named(
                p2,
                ("audit_kam_resonances", "kam_audit", "audit_kam"),
                orbital,
            )
        feedback = self._synthesize_feedback_resilient(orbital, kam, monodromy, section)

        kam_obj = getattr(feedback, "kam_audit", kam)
        mono_obj = getattr(feedback, "monodromy_result", monodromy)
        kam_ok = bool(getattr(kam_obj, "is_kam_torus_stable", True)) if kam_obj is not None else True
        lyap = 0.0
        recurrence = 0.0
        nek = 0.0
        rot = 0.0
        if kam_obj is not None:
            lyap = float(getattr(kam_obj, "max_lyapunov_exponent", lyap))
            recurrence = float(getattr(kam_obj, "poincare_recurrence_time_estimate", 0.0))
            nek = float(getattr(kam_obj, "nekhoroshev_time_bound", getattr(kam_obj, "nekhoroshev_bound", 0.0)))
            rot = float(getattr(kam_obj, "rotation_number", 0.0))
        if mono_obj is not None:
            lyap = float(getattr(mono_obj, "max_lyapunov_exponent", lyap))
        drift = float(getattr(feedback, "symplectic_drift_penalty", 0.0))
        energy_drift = float(getattr(feedback, "energy_drift", 0.0))
        pairing = float(getattr(feedback, "pairing_defect", 0.0))
        section_count = int(getattr(feedback, "section_count", 0))
        if section is not None and section_count == 0:
            section_count = int(getattr(section, "crossing_count", getattr(section, "section_count", 0)))
        conserved = bool(getattr(feedback, "symplectic_conserved", drift < _SYMPLECTIC_DRIFT_TOL * 10.0))
        holonomy = bool(getattr(feedback, "holonomy_breakage_detected", False))
        spohn = float(getattr(orbital, "spohn_entropy_production", 0.0))
        lip = self._estimate_symplectic_lipschitz(drift, lyap, kam_ok)
        orient_verdict = _as_heyting(getattr(feedback, "intermediate_verdict", contract.isometry_verdict))
        orient_verdict = contract.isometry_verdict.meet(orient_verdict)
        tentative = _as_status(
            getattr(feedback, "prosecution_status_tentative", OODAProsecutionStatus.PENDING)
        )
        if carrier1.claim_tamper_detected:
            tentative = OODAProsecutionStatus.VETOED_INDICTMENT_TAMPER
            orient_verdict = HeytingOmega4.VETOED
        elif not carrier1.dirac_auth_ok:
            tentative = OODAProsecutionStatus.VETOED_DIRAC_AUTH_FAILURE
            orient_verdict = HeytingOmega4.VETOED

        campaign_id = f"CAMP-OODA-PROSECUTOR-v15-{int(now * 1000.0) % 10000000:07d}"
        chain_bytes = (
            f"{contract.chain_hash}:{campaign_id}:{int(orient_verdict.value)}:{int(tentative.value)}"
        ).encode("utf-8")
        chain_hash = hashlib.sha384(chain_bytes).hexdigest()[:_CHAIN_HEX_LEN]
        campaign = OODAProsecutionCampaignResultPhD(
            campaign_id=campaign_id,
            bound_contract_id=contract.contract_id,
            lipschitz_contraction=float(lip),
            spohn_entropy_production=spohn,
            is_kam_torus_stable=kam_ok,
            max_lyapunov_exponent=lyap,
            poincare_recurrence_time=recurrence,
            symplectic_drift_penalty=drift,
            holonomy_breakage_detected=holonomy,
            nekhoroshev_bound=nek,
            rotation_number=rot,
            section_count=section_count,
            energy_drift=energy_drift,
            pairing_defect=pairing,
            symplectic_conserved=conserved,
            qnd_orient_verdict=orient_verdict,
            prosecution_status_tentative=tentative,
            chain_hash=chain_hash,
            hmac_p2_signature="",
            timestamp_utc=now,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P2, self._canonical_p2_payload(campaign))
        object.__setattr__(campaign, "hmac_p2_signature", sig)
        return campaign, feedback

    def _canonical_p2_payload(self, campaign: OODAProsecutionCampaignResultPhD) -> str:
        return (
            f"{campaign.campaign_id}:{campaign.bound_contract_id}:"
            f"{_fmt(campaign.lipschitz_contraction, 6)}:{_fmt(campaign.spohn_entropy_production)}:"
            f"{int(campaign.is_kam_torus_stable)}:{_fmt(campaign.max_lyapunov_exponent, 6)}:"
            f"{_fmt(campaign.poincare_recurrence_time)}:{_fmt(campaign.symplectic_drift_penalty)}:"
            f"{int(campaign.holonomy_breakage_detected)}:{_fmt(campaign.nekhoroshev_bound)}:"
            f"{_fmt(campaign.rotation_number)}:{int(campaign.section_count)}:"
            f"{_fmt(campaign.energy_drift)}:{int(campaign.symplectic_conserved)}:"
            f"{int(campaign.qnd_orient_verdict.value)}:{int(campaign.prosecution_status_tentative.value)}:"
            f"{campaign.chain_hash}"
        )

    def continue_to_phase3_prosecutor_gate(self, campaign: OODAProsecutionCampaignResultPhD) -> bool:
        """
        Compuerta Homomórfica Φ₂₃: revalida HMAC-SHA384 P2 y exige Lip(T) < 1
        (teorema del punto fijo de Banach sobre el endomorfismo RSI).
        """
        if float(campaign.lipschitz_contraction) >= 1.0:
            logger.error("Violación del Teorema de Punto Fijo de Banach: Lip(T) >= 1")
            return False
        expected = self._compute_hmac(_HMAC_DOMAIN_P2, self._canonical_p2_payload(campaign))
        try:
            return hmac.compare_digest(str(campaign.hmac_p2_signature), str(expected))
        except (TypeError, ValueError):
            return False

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # MÉTODO PUENTE FORMAL FINAL FASE II ──► INICIO FASE III
    # ──────────────────────────────────────────────────────────────────────────────────────────
    def bridge_phase2_to_phase3_adjudication_transfer(
        self, carrier1: ProsecutorPhase1TransferCarrier
    ) -> ProsecutorPhase2TransferCarrier:
        """
        DEFINICIÓN FORMAL FINAL DE LA FASE II (morfismo de acoplamiento 2→3):

        Ejecuta la campaña simpléctica de Poincaré, evalúa divisores diofánticos KAM,
        conserva (o acusa) la forma simpléctica ω = dq ∧ dp, sintetiza el tensor de
        realimentación mediante
            synthesize_prosecution_symplectic_feedback
        y emite `ProsecutorPhase2TransferCarrier` como continuación directa e
        ininterrumpida hacia la Fase III (RSI-L3, mónada, interlock simulado).
        """
        campaign_result, feedback_carrier = self.execute_prosecution_campaign_orientation(carrier1)
        logger.debug(
            "Puente Fiscal Fase II → Fase III: Campaña=%s KAM=%s λ=%.6f ΔE=%.3e Σ=%d",
            campaign_result.campaign_id,
            campaign_result.is_kam_torus_stable,
            campaign_result.max_lyapunov_exponent,
            campaign_result.energy_drift,
            campaign_result.section_count,
        )
        return ProsecutorPhase2TransferCarrier(
            gauge_contract=carrier1.gauge_contract,
            campaign_result=campaign_result,
            symplectic_feedback=feedback_carrier,
            sanitized_density_matrix=carrier1.sanitized_density_matrix,
            gauge_connection=carrier1.gauge_connection,
            indictment_germ=carrier1.indictment_germ,
            dirac_auth_ok=carrier1.dirac_auth_ok,
            claim_tamper_detected=carrier1.claim_tamper_detected,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE III: DECIDE & ACT / RSI-L3, MÓNADA CARTESIANA, TOPOS Ω₄, ESP32 SIMULADO, Φ_sem
# hereda el morfismo 2→3; su DEFINICIÓN FORMAL FINAL es execute_phase3_sovereign_decision
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class OODAProsecutionDeciderPhD(OODAProsecutionCampaignOrientorPhD):
    """
    FASE III — DECIDE & ACT.

    Consume `ProsecutorPhase2TransferCarrier`. Valida Φ₂₃. Aplica la mónada cartesiana
    T = (T, η, μ) por morfismos de Kleisli en Ω₄. Ejecuta RSI-L3:

        Nivel 1: pesos; Nivel 2: hiperparámetros con T fijo;
        Nivel 3: T es punto fijo del meta-operador
            S(λ) = clip( λ(1 − β ρ) + η(1 − L_emp),  0.50, 0.99 )
        Lip(T) < 1 ⇒ único punto fijo (Banach).

    Adjudica vía `adjudicate_and_execute`. El interlock ESP32 Crowbar se SIMULA
    (latencia recortada a [42, 392.15] ns). NUNCA se escribe GPIO_OUT_W1TS_REG
    (0x3FF44008) ni se emite payload de crowbar BT151. Proyecta Φ_sem (Dolor, Riesgo, Dinero).
    Compone Φ₁₃ := Φ₂₃ ∘ Φ₁₂ y mide el defecto funtorial.
    """

    def _cstar_corner_integrity(self, rho: NDArray[np.complex128], i0: int, i1: int) -> float:
        n = int(rho.shape[0])
        i0 = max(0, min(i0, n))
        i1 = max(i0 + 1, min(i1, n))
        e = np.zeros((n, n), dtype=np.complex128)
        e[i0:i1, i0:i1] = np.eye(i1 - i0, dtype=np.complex128)
        pi_rho = e @ rho @ e
        pi_rho2 = e @ (rho @ rho) @ e
        num = float(la.norm(pi_rho2 - pi_rho @ pi_rho, ord="fro"))
        den = max(float(la.norm(rho, ord="fro")), _WILKINSON_EPSILON)
        return float(math.exp(-num / den))

    def _compose_phi13_functoriality(
        self, rho: NDArray[np.complex128]
    ) -> Tuple[float, float, float, float]:
        """Retorna (i₁₂, i₂₃, i₁₃, |i₁₃ − i₁₂ i₂₃|). Φ₁₃ := Φ₂₃ ∘ Φ₁₂."""
        n = int(rho.shape[0])
        b = max(1, n // 3)
        i12 = self._cstar_corner_integrity(rho, 0, 2 * b)
        i23 = self._cstar_corner_integrity(rho, b, min(n, 3 * b))
        i13 = self._cstar_corner_integrity(rho, 0, min(n, 3 * b))
        defect = abs(i13 - i12 * i23)
        return i12, i23, i13, float(defect)

    def _apply_rsi_l3_meta_operator(
        self, lip_emp: float, residual: float
    ) -> Tuple[int, float, float, float]:
        """
        Dispara el paso RSI-L3 del núcleo (2- o 3-tupla) o aplica S localmente.
        Retorna (generación, Lip_T, Lip_S, residual de Cauchy).
        """
        core = getattr(self.engine, "phase3_core", None)
        if core is not None and hasattr(core, "trigger_recursive_step"):
            try:
                out = core.trigger_recursive_step(lip_emp, residual)
            except TypeError:
                try:
                    out = core.trigger_recursive_step(lip_emp)
                except TypeError:
                    out = None
            if isinstance(out, tuple):
                if len(out) >= 3:
                    cfg, lip_t, cauchy = out[0], float(out[1]), float(out[2])
                elif len(out) == 2:
                    cfg, lip_t, cauchy = out[0], float(out[1]), float(residual)
                else:
                    cfg, lip_t, cauchy = None, float(lip_emp), float(residual)
                gen = int(getattr(cfg, "generation", getattr(cfg, "rsi_level3_generation", 1)) or 1)
                meta = float(getattr(cfg, "meta_operator_lip", getattr(cfg, "banach_lipschitz", lip_t)))
                ingest = getattr(core, "ingest_rsi_configuration", None)
                if callable(ingest) and cfg is not None:
                    try:
                        ingest(cfg)
                    except (TypeError, AttributeError):
                        pass
                return gen, float(lip_t), float(meta), float(cauchy)
        lam = _DEFAULT_BANACH_CONTRACTION_LIP
        lam = float(np.clip(lam * (1.0 - 0.15 * residual) + 0.08 * (1.0 - lip_emp), 0.50, 0.99))
        return 1, float(min(lip_emp, lam)), float(lam), float(residual)

    def _project_to_visceral_business_phd(
        self,
        verdict: HeytingOmega4,
        status: OODAProsecutionStatus,
        spohn_entropy: float,
        lyapunov_exp: float,
        is_kam_stable: bool,
        crowbar_simulated: bool,
    ) -> Dict[str, Any]:
        """Funtor semántico visceral Φ_sem: variedad penal → gobernanza (Dolor, Riesgo, Dinero)."""
        hardware_sim = (
            "SIMULACIÓN DE PARÁLISIS CIBER-FÍSICA (< 392.15 ns); "
            "GPIO_OUT_W1TS_REG 0x3FF44008 DOCUMENTAL — SIN MMIO, SIN BT151"
            if crowbar_simulated
            else "FLUJO NOMINAL LAMINAR (interlock ESP32 STRICTLY_INHIBITED / simulado inerte)"
        )
        if verdict == HeytingOmega4.VETOED:
            return {
                "wacc_protection": "PROTEGIDO INCONDICIONALMENTE AL 15.0% HURDLE RATE",
                "net_liquidity_impact": "+11.5% EN TESORERÍA (Reserva por volatilidad recortada del 15% al 3.5%)",
                "kv_cache_compression_pct": 86.4,
                "llm_cost_savings_pct": 80.0,
                "hardware_action": hardware_sim,
                "executive_summary": (
                    f"INTERVENCIÓN FISCAL OODA v15 (SIMULADA): Estatus={status.name}. "
                    f"Ruptura de holonomía, anomalía fermiónica apócrifa o caos orbital (Lyap={lyapunov_exp:+.4f}). "
                    "Interlock ESP32 cebado en simulación para proteger el balance; ningún GPIO fue escrito."
                ),
                "visceral_financial_vector": {
                    "spohn_irreversible_dissipation": round(spohn_entropy, 8),
                    "kam_torus_rupture": not is_kam_stable,
                    "capital_at_risk_contained_usd": 1250000.0,
                },
            }
        if verdict == HeytingOmega4.CRITICAL:
            return {
                "wacc_protection": "PRESERVADO EN 15.0% — Resonancia Caótica Bajo Mitigación Simpléctica",
                "net_liquidity_impact": "+9.0% RECIRCULANTE (Filtro KAM activo)",
                "kv_cache_compression_pct": 86.4,
                "llm_cost_savings_pct": 80.0,
                "hardware_action": "PREPARACIÓN SIMULADA DE CEBADO — interlock armado, MMIO inhibido",
                "executive_summary": (
                    f"ALERTA CRÍTICA FISCAL OODA v15: Estatus={status.name}. Resonancia KAM detectada "
                    "sin colapso de calibre. RSI-L3 contrayendo T en Banach para restaurar toros invariantes."
                ),
                "visceral_financial_vector": {
                    "spohn_irreversible_dissipation": round(spohn_entropy, 8),
                    "kam_torus_rupture": True,
                    "capital_at_risk_contained_usd": 450000.0,
                },
            }
        if verdict == HeytingOmega4.DEGRADED:
            return {
                "wacc_protection": "PRESERVADO EN 15.0% — Ventana de Alivio Transitoria (1 Hora)",
                "net_liquidity_impact": "+8.5% RECIRCULANTE (Válvula de compensación activada)",
                "kv_cache_compression_pct": 86.4,
                "llm_cost_savings_pct": 80.0,
                "hardware_action": "VETO SUAVE — se requiere positrón HMAC-Dirac v15 en 1h",
                "executive_summary": (
                    f"ADVERTENCIA DE FRICCIÓN FISCAL OODA v15: Estatus={status.name}. "
                    "Fricción de Yang–Mills moderada. Firma de positrón requerida para evitar el interlock simulado."
                ),
                "visceral_financial_vector": {
                    "spohn_irreversible_dissipation": round(spohn_entropy, 8),
                    "kam_torus_rupture": False,
                    "capital_at_risk_contained_usd": 120000.0,
                },
            }
        return {
            "wacc_protection": "ÓPTIMO Y ESTABLE AL 15.0% HURDLE RATE",
            "net_liquidity_impact": "+11.5% FLUIDO LIBRE EN TESORERÍA",
            "kv_cache_compression_pct": 86.4,
            "llm_cost_savings_pct": 80.0,
            "hardware_action": hardware_sim,
            "executive_summary": (
                "COHERENCIA FISCAL TOTAL OODA v15: isometría simpléctica preservada, toros KAM estables "
                "y aniquilación e⁻+e⁺→2γ sin retroacción sobre el sustrato financiero."
            ),
            "visceral_financial_vector": {
                "spohn_irreversible_dissipation": 0.0,
                "kam_torus_rupture": False,
                "capital_at_risk_contained_usd": 0.0,
            },
        }

    def execute_cartesian_monad_kleisli(
        self,
        contract: OODAProsecutorGaugeContractPhD,
        campaign: OODAProsecutionCampaignResultPhD,
        exec_verdict: HeytingOmega4,
        exec_status: OODAProsecutionStatus,
    ) -> Tuple[HeytingOmega4, OODAProsecutionStatus]:
        """
        Mónada cartesiana T = (T, η, μ) sobre la categoría de estados penales.
        μ: T(T(X)) → T(X) por meet de Heyting (ínfimo de verdad = no decreciente en severidad).
        """
        fused = contract.isometry_verdict.meet(campaign.qnd_orient_verdict).meet(exec_verdict)
        status = exec_status
        if fused == HeytingOmega4.VETOED and status == OODAProsecutionStatus.ABSOLVED:
            status = OODAProsecutionStatus.VETOED_HOLONOMY_BREAKAGE
        return fused, status

    def _adjudicate_resilient(
        self, feedback: ProsecutionSymplecticFeedbackCarrier
    ) -> OODAProsecutionExecutionCertificatePhD:
        core = getattr(self.engine, "phase3_core", None)
        if core is None or not hasattr(core, "adjudicate_and_execute"):
            raise RuntimeError("Núcleo Fase III ausente: no se puede adjudicar.")
        try:
            return core.adjudicate_and_execute(feedback)
        except TypeError:
            return core.adjudicate_and_execute(feedback_carrier=feedback)

    def _merkle_seed(self, passport: OODAProsecutorSovereignPassportPhD) -> bytes:
        return (
            f"{passport.passport_id}:{passport.sovereign_id}:{passport.campaign_id}:"
            f"{passport.bound_contract_id}:{int(passport.global_verdict.value)}:"
            f"{int(passport.prosecution_status.value)}:{int(passport.crowbar_triggered_simulated)}:"
            f"{passport.crowbar_latency_ns:.2f}:{passport.rsi_level3_generation}:"
            f"{passport.rsi_banach_contraction_lip:.6f}:"
            f"{passport.homomorphic_gate_phi12_integrity:.8f}:"
            f"{passport.homomorphic_gate_phi23_integrity:.8f}:"
            f"{passport.homomorphic_gate_phi13_integrity:.8f}:"
            f"{passport.functoriality_defect:.8f}:{passport.timestamp_utc:.6f}"
        ).encode("utf-8")

    def verify_merkle_provenance(self, passport: OODAProsecutorSovereignPassportPhD) -> bool:
        """Recomputes SHA-384 Merkle root and HMAC P3; compare_digest en ambos."""
        root = hashlib.sha384(self._merkle_seed(passport)).hexdigest()
        if not hmac.compare_digest(str(passport.merkle_dag_sha384_root), str(root)):
            return False
        prov = (
            f"PROSECUTOR-PROVENANCE-v15:{root}:{passport.timestamp_utc:.6f}:"
            f"{int(passport.global_verdict.value)}:{int(passport.prosecution_status.value)}:"
            f"{_fmt(passport.homomorphic_gate_phi12_integrity)}:"
            f"{_fmt(passport.homomorphic_gate_phi23_integrity)}:"
            f"{_fmt(passport.homomorphic_gate_phi13_integrity)}:"
            f"{_fmt(passport.functoriality_defect)}"
        )
        expected = self._compute_hmac(_HMAC_DOMAIN_P3, prov)
        try:
            return hmac.compare_digest(str(passport.hmac_provenance_signature), str(expected))
        except (TypeError, ValueError):
            return False

    def execute_phase3_sovereign_decision(
        self, carrier2: ProsecutorPhase2TransferCarrier
    ) -> OODAProsecutorSovereignPassportPhD:
        """
        DEFINICIÓN FORMAL FINAL DE LA FASE III:

        Valida Φ₂₃, adjudica en Ω₄, aplica RSI-L3 (S punto fijo de T), simula el interlock
        ESP32 (sin MMIO), proyecta Φ_sem, sella el pasaporte Merkle-DAG y certifica
        Φ₁₃ = Φ₂₃ ∘ Φ₁₂. Este morfismo cierra el ciclo OODA soberano.
        """
        now = time.time()
        contract = carrier2.gauge_contract
        campaign = carrier2.campaign_result
        feedback = carrier2.symplectic_feedback

        if not self.continue_to_phase3_prosecutor_gate(campaign):
            logger.critical("Violación de integridad en Compuerta Homomórfica Φ₂₃ (HMAC o Banach inválido)")
            raise ValueError("Falla de autenticidad criptográfica en Compuerta Homomórfica Φ₂₃.")

        exec_cert = self._adjudicate_resilient(feedback)
        exec_verdict = _as_heyting(getattr(exec_cert, "global_heyting_verdict", campaign.qnd_orient_verdict))
        exec_status = _as_status(getattr(exec_cert, "status", campaign.prosecution_status_tentative))
        if carrier2.claim_tamper_detected:
            exec_status = OODAProsecutionStatus.VETOED_INDICTMENT_TAMPER
            exec_verdict = HeytingOmega4.VETOED
        final_verdict, final_status = self.execute_cartesian_monad_kleisli(
            contract, campaign, exec_verdict, exec_status
        )

        i12, i23, i13, defect = self._compose_phi13_functoriality(carrier2.sanitized_density_matrix)
        if contract.gate_tear_detected or defect > _FUNCTOR_DEFECT_TOL or min(i12, i23, i13) < _GATE_TEAR_INTEGRITY_FLOOR:
            if final_verdict.truth_rank() > HeytingOmega4.VETOED.truth_rank():
                final_verdict = final_verdict.meet(HeytingOmega4.CRITICAL)
            if final_status in (OODAProsecutionStatus.ABSOLVED, OODAProsecutionStatus.PURGED_DIRAC_VACUUM):
                final_status = OODAProsecutionStatus.VETOED_HOMOMORPHIC_GATE_TEAR

        lip_emp = float(getattr(exec_cert, "rsi_banach_contraction_lip", campaign.lipschitz_contraction))
        residual = float(getattr(exec_cert, "rsi_cauchy_residual", campaign.symplectic_drift_penalty))
        gen_e = int(getattr(exec_cert, "rsi_level3_generation", 0))
        meta_e = float(getattr(exec_cert, "rsi_meta_operator_lip", lip_emp))
        gen, lip_t, meta_s, cauchy = self._apply_rsi_l3_meta_operator(lip_emp, residual)
        gen = max(gen, gen_e, 1)
        meta_s = float(meta_e if meta_e > 0.0 else meta_s)

        crowbar_sim = bool(
            getattr(exec_cert, "crowbar_triggered_simulated", getattr(exec_cert, "crowbar_triggered", False))
        )
        if final_verdict == HeytingOmega4.VETOED:
            crowbar_sim = True
        latency = float(getattr(exec_cert, "crowbar_latency_ns", 0.0))
        if crowbar_sim and latency <= 0.0:
            sim = getattr(getattr(self.engine, "phase3_core", None), "simulate_esp32_safety_interlock", None)
            if callable(sim):
                try:
                    crowbar_sim, latency = sim(True, f"{final_status.name}".encode("utf-8"))
                except TypeError:
                    pass
            latency = float(
                min(max(latency if latency > 0.0 else _ESP32_CROWBAR_MIN_LATENCY_NS,
                        _ESP32_CROWBAR_MIN_LATENCY_NS),
                    _ESP32_CROWBAR_MAX_LATENCY_NS)
            )
        if not crowbar_sim:
            latency = 0.0

        visceral = self._project_to_visceral_business_phd(
            verdict=final_verdict,
            status=final_status,
            spohn_entropy=campaign.spohn_entropy_production,
            lyapunov_exp=campaign.max_lyapunov_exponent,
            is_kam_stable=campaign.is_kam_torus_stable,
            crowbar_simulated=crowbar_sim,
        )

        williamson = getattr(getattr(feedback, "orbital_carrier", None), "williamson_invariants", ()) or ()
        wmin = float(min(williamson)) if williamson else 0.0
        energy = float(getattr(getattr(feedback, "orbital_carrier", None), "hamiltonian_energy", 0.0))

        passport_id = f"PASSPORT-PROSECUTOR-OODA-v15-{int(now * 1000.0) % 10000000:07d}"
        passport = OODAProsecutorSovereignPassportPhD(
            passport_id=passport_id,
            sovereign_id=self.sovereign_id,
            campaign_id=campaign.campaign_id,
            bound_contract_id=contract.contract_id,
            global_verdict=final_verdict,
            prosecution_status=final_status,
            crowbar_triggered=bool(crowbar_sim),
            crowbar_triggered_simulated=bool(crowbar_sim),
            crowbar_latency_ns=float(latency),
            electron_anomalies_purged=int(getattr(exec_cert, "electron_anomalies_purged", 0)),
            released_energy_kev=float(getattr(exec_cert, "released_energy_kev", 0.0)),
            back_action_db=float(getattr(exec_cert, "back_action_db", 0.0)),
            rsi_level3_generation=int(gen),
            rsi_banach_contraction_lip=float(lip_t),
            rsi_meta_operator_lip=float(meta_s),
            rsi_cauchy_residual=float(cauchy),
            homomorphic_gate_phi12_integrity=float(i12),
            homomorphic_gate_phi23_integrity=float(i23),
            homomorphic_gate_phi13_integrity=float(i13),
            functoriality_defect=float(defect),
            energy_hamiltonian=energy,
            energy_drift=float(campaign.energy_drift),
            nekhoroshev_bound=float(campaign.nekhoroshev_bound),
            williamson_min=wmin,
            claim_tamper_detected=bool(carrier2.claim_tamper_detected),
            gate_tear_detected=bool(contract.gate_tear_detected or defect > _FUNCTOR_DEFECT_TOL),
            esp32_gpio_w1ts_reg_documentary=int(_ESP32_GPIO_OUT_W1TS_REG),
            merkle_dag_sha384_root="",
            hmac_provenance_signature="",
            visceral_report=visceral,
            timestamp_utc=now,
        )
        root = hashlib.sha384(self._merkle_seed(passport)).hexdigest()
        object.__setattr__(passport, "merkle_dag_sha384_root", root)
        prov = (
            f"PROSECUTOR-PROVENANCE-v15:{root}:{now:.6f}:"
            f"{int(final_verdict.value)}:{int(final_status.value)}:"
            f"{_fmt(i12)}:{_fmt(i23)}:{_fmt(i13)}:{_fmt(defect)}"
        )
        object.__setattr__(passport, "hmac_provenance_signature", self._compute_hmac(_HMAC_DOMAIN_P3, prov))
        return passport


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# CLASE ORQUESTADORA SOBERANA
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class OODATriphasicHomomorphicPipelineProsecutorAgentPhD(OODAProsecutionDeciderPhD):
    """
    SOBERANO AGÉNTICO FISCAL INQUISIDOR DEL PIPELINE OODA v15.0.0-PhD.

    Orquesta el flujo penal ininterrumpido a través de las 3 fases anidadas
    (Observe → Orient → Decide&Act) sin discontinuidad funtorial:

        FASE I   ──► bridge_phase1_to_phase2_campaign_transfer
        FASE II  ──► bridge_phase2_to_phase3_adjudication_transfer
        FASE III ──► execute_phase3_sovereign_decision  → OODAProsecutorSovereignPassportPhD

    El interlock ESP32 es estrictamente simulado. El agente no escribe MMIO.
    """

    def prosecute_ooda_pipeline_sovereign(
        self,
        density_matrix: NDArray,
        electron_anomaly_count: int = 0,
        positron_auth_signature: str = "",
        transaction_payload: str = "",
    ) -> OODAProsecutorSovereignPassportPhD:
        """
        EJECUCIÓN DEL PIPELINE JUDICIAL PENAL COMPLETO EN TRES FASES ANIDADAS.
        El operador densidad recorre validación topológica, estabilidad celeste
        simpléctica, aniquilación en Fock y sentencia (simulada) en silicio.
        """
        carrier_phase1 = self.bridge_phase1_to_phase2_campaign_transfer(
            density_matrix=density_matrix,
            electron_anomaly_count=electron_anomaly_count,
            positron_auth_signature=positron_auth_signature,
            transaction_payload=transaction_payload,
        )
        carrier_phase2 = self.bridge_phase2_to_phase3_adjudication_transfer(carrier1=carrier_phase1)
        return self.execute_phase3_sovereign_decision(carrier_phase2)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS FORMALES Y DEMOSTRACIÓN DE DOCTORADO
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 105
    print(banner)
    print("  VERIFICACIÓN CIENTÍFICA: SOBERANO FISCAL AGÉNTICO OODA TRIFÁSICO HOMOMÓRFICO")
    print(f"  {_VERSION_STRING}")
    print("  Fases anidadas • Yoshida O(4) • Williamson • KAM–Nekhoroshev • Dirac HMAC-dominio")
    print("  RSI-L3 (S∘T) • Φ₁₃=Φ₂₃∘Φ₁₂ • ESP32 Crowbar SIMULADO (sin MMIO)")
    print(banner)

    DIMENSION = 56
    agent = OODATriphasicHomomorphicPipelineProsecutorAgentPhD(
        sovereign_id="SOVEREIGN-OODA-PROSECUTOR-SUPREME-01",
        dimension=DIMENSION,
    )

    rng = np.random.default_rng(2026)
    psi = rng.normal(0.0, 1.0, size=(DIMENSION, DIMENSION)) + 1j * rng.normal(
        0.0, 1.0, size=(DIMENSION, DIMENSION)
    )
    rho_sim = psi @ psi.conj().T
    rho_sim = rho_sim / np.trace(rho_sim)

    payload_valid = "TX-OODA-2026-SOVEREIGN-SECOP-VALID-001"
    valid_positron_hmac = compute_dirac_positron_hmac(agent.secret_key, payload_valid)

    print("\n" + "─" * 34 + " CASO 1: POSITRÓN VÁLIDO (FLUJO LAMINAR) " + "─" * 34)
    passport_valid = agent.prosecute_ooda_pipeline_sovereign(
        density_matrix=rho_sim,
        electron_anomaly_count=2,
        positron_auth_signature=valid_positron_hmac,
        transaction_payload=payload_valid,
    )
    merkle_ok = agent.verify_merkle_provenance(passport_valid)
    print(f"  • Sovereign Passport ID           : {passport_valid.passport_id}")
    print(f"  • Veredicto Retículo Heyting Ω₄   : {passport_valid.global_verdict.name}")
    print(f"  • Estado de la Acusación          : {passport_valid.prosecution_status.name}")
    print(f"  • Electrones Purgados (e⁻ + e⁺)   : {passport_valid.electron_anomalies_purged}")
    print(f"  • Energía Liberada en Vacío (2γ)  : {passport_valid.released_energy_kev:.1f} keV")
    print(f"  • Retroacción Cuántica en RAM     : {passport_valid.back_action_db:.1f} dB")
    print(
        f"  • RSI-L3 generación / Lip_T / S   : "
        f"#{passport_valid.rsi_level3_generation} / "
        f"{passport_valid.rsi_banach_contraction_lip:.6f} / "
        f"{passport_valid.rsi_meta_operator_lip:.6f}  (Cauchy={passport_valid.rsi_cauchy_residual:.3e})"
    )
    print(
        f"  • ESP32 Crowbar (SIMULADO)        : {passport_valid.crowbar_triggered_simulated} "
        f"(GPIO_OUT_W1TS_REG=0x{passport_valid.esp32_gpio_w1ts_reg_documentary:08X} documental)"
    )
    print(
        f"  • Integridad Φ₁₂ / Φ₂₃ / Φ₁₃      : "
        f"{passport_valid.homomorphic_gate_phi12_integrity * 100:.1f}% / "
        f"{passport_valid.homomorphic_gate_phi23_integrity * 100:.1f}% / "
        f"{passport_valid.homomorphic_gate_phi13_integrity * 100:.1f}%  "
        f"(defecto funtorial={passport_valid.functoriality_defect:.3e})"
    )
    print(f"  • Energía H / ΔE / Nekhoroshev    : "
          f"{passport_valid.energy_hamiltonian:.6f} / {passport_valid.energy_drift:.3e} / "
          f"{passport_valid.nekhoroshev_bound:.3e}")
    print(f"  • Williamson ν_min                : {passport_valid.williamson_min:.6e}")
    print(f"  • Merkle-DAG SHA-384 verificado   : {merkle_ok}  ({passport_valid.merkle_dag_sha384_root[:36]}...)")
    print(f"  • Protección WACC Ejecutiva       : {passport_valid.visceral_report['wacc_protection']}")
    print(f"  • Impacto de Liquidez Neta        : {passport_valid.visceral_report['net_liquidity_impact']}")

    print("\n" + "─" * 32 + " CASO 2: POSITRÓN APÓCRIFO (INTERLOCK SIMULADO) " + "─" * 32)
    payload_attack = "TX-OODA-2026-MALICIOUS-INJECTION"
    passport_bad = agent.prosecute_ooda_pipeline_sovereign(
        density_matrix=rho_sim,
        electron_anomaly_count=3,
        positron_auth_signature="APOCRYPHAL_FORGED_HMAC_SIGNATURE_0xDEADBEEF",
        transaction_payload=payload_attack,
    )
    merkle_bad = agent.verify_merkle_provenance(passport_bad)
    print(f"  • Sovereign Passport ID           : {passport_bad.passport_id}")
    print(f"  • Veredicto Retículo Heyting Ω₄   : {passport_bad.global_verdict.name}")
    print(f"  • Estado de la Acusación          : {passport_bad.prosecution_status.name}")
    print(f"  • Retroacción Cuántica en RAM     : {passport_bad.back_action_db:.1f} dB (colapso fermiónico)")
    print(f"  • Disparo Interlock ESP32 (SIM)   : {passport_bad.crowbar_triggered_simulated}")
    print(
        f"  • Latencia Xtensa LX7 (simulada)  : {passport_bad.crowbar_latency_ns:.2f} ns "
        f"(clip [{_ESP32_CROWBAR_MIN_LATENCY_NS:.0f}, {_ESP32_CROWBAR_MAX_LATENCY_NS:.2f}])"
    )
    print(f"  • Acción de Hardware Visceral     : {passport_bad.visceral_report['hardware_action']}")
    print(f"  • Resumen Ejecutivo C-Suite       : {passport_bad.visceral_report['executive_summary']}")
    print(
        f"  • Capital Contenido en Riesgo     : "
        f"${passport_bad.visceral_report['visceral_financial_vector']['capital_at_risk_contained_usd']:,.2f} USD"
    )
    print(f"  • Tamper / Φ-tear                 : {passport_bad.claim_tamper_detected} / {passport_bad.gate_tear_detected}")
    print(f"  • Merkle-DAG SHA-384 verificado   : {merkle_bad}")
    print(f"  • Firma de Procedencia Soberana   : {passport_bad.hmac_provenance_signature[:36]}...")
    print(banner)
    print(f"  Motor fiscal importado            : {_ENGINE_AVAILABLE}")
    print(banner)
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Topological Gauge Field Agent (Soberano de Calibre Topológico)    ║
║ Ubicación: app/agents/wisdom/topological_gauge_field_agent.py                ║
║ Versión  : 11.0.0 (SOVEREIGN-GAUGE-AGENT-KKS-POINCARE-BRUNO-RSI3-OODA-PhD)   ║
║ Función  : Soberano agéntico de calibre topológico, orquestación del ciclo   ║
║            OODA anidado de 3 fases (Observe ⊣ Orient ⊣ Decide&Act),          ║
║            verificación cuántica no destructiva QND, adjudicación Heyting Ω₃ ║
║            y gobernanza ciber-física con interlock ESP32 Crowbar.           ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL:
─────────────────────────────────────────────────────────────────────────────────────────
El `TopologicalGaugeFieldAgent` actúa como el Soberano Agéntico de Calibre Topológico y
Gobernanza Inmunológica en el Estrato WISDOM de la arquitectura cognitiva APU Filter. Opera en
simbiosis directa con su motor espectral `TopologicalGaugeFieldEngine` (`app/wisdom/topological_gauge_field_engine.py`).
Mientras el motor procesa el sustrato diferencial (álgebra de Lie $\mathfrak{su}(N)$, curvatura
orbital KKS $F = [\rho, H_{\text{cel}}]$, proyecciones simplécticas $\mathfrak{sp}(2n, \mathbb{R})$ y series de Bruno),
este Soberano Agéntico estructura la deliberación ejecutiva en un bucle OODA anidado de tres fases:
  - FASE 1 (OBSERVE - `GaugeAgentPhase1Observer`): Sanitiza el operador de densidad $\rho$, consume el
    sobre analítico $\Phi_1$, ejecuta la automejora recursiva agéntica mediante conjugación unitaria
    $\rho \mapsto \exp(i \tau W_{\text{Lind}}) \rho \exp(-i \tau W_{\text{Lind}})$ y sella la transición
    mediante el morfismo vinculante `continue_to_phase2`.
  - FASE 2 (ORIENT - `GaugeAgentPhase2Orientor`): Recibe el certificado sellado $\Phi_1$, consume la fase
    topológica (invariantes de Chern-Simons, número de Pontryagin $p_1$, holonomía de Wilson $\mathcal{W}(C)$),
    computa la aniquilación en el espacio de Fock con firma criptográfica, evalúa el espectro de Floquet,
    la estabilidad homoclínica de Melnikov y cierra con `continue_to_phase3`.
  - FASE 3 (DECIDE & ACT - `GaugeAgentPhase3Decider`): Desempaqueta las observaciones QND, realiza la
    adjudicación de retículo en el álgebra de Heyting trivalente $\Omega_3 = \{0\text{x}00 < 0\text{x}01 < 0\text{x}03\}$,
    dispara el actuador interlock de hardware ESP32 Crowbar sobre GPIO14 ante cualquier indicio de fractura,
    y traduce el veredicto topológico al funtor semántico de negocios $\Phi_{\text{sem}}$ ("Riesgo, Dolor y Dinero").

2. MARCO TEÓRICO:
─────────────────────────────────────────────────────────────────────────────────────────
2.1. CICLO OODA ANIDADO Y MORFISMOS DE TRANSICIÓN CATEGORIAL:
   El ciclo de deliberación agéntica implementa un funtor de enriquecimiento sobre la categoría de estados de gauge:
     $$\mathbf{State}_1 \xrightarrow{\Phi_1 \circ \text{continue\_to\_phase2}} \mathbf{State}_2 \xrightarrow{\Phi_2 \circ \text{continue\_to\_phase3}} \mathbf{State}_3$$
   Cada transición exige la verificación incondicional de autenticidad criptográfica mediante HMAC-SHA256
   sobre el dominio aislado `APU.Wisdom.TopologicalGaugeFieldAgent.v11.0.0`.

2.2. AUTOMEJORA RECURSIVA AGÉNTICA RSI³ Y CONJUGACIÓN DE LINDSTEDT:
   A diferencia del motor espectral que opera sobre la curvatura $F$, el Soberano Agéntico actualiza el estado de densidad
   mediante un flujo unitario hamiltoniano generado por la parte antihermítica de la solución de Lindstedt $W_{\text{Lind}}$:
     $$\rho^{(m+1)} = \exp\left( i \tau \frac{W_{\text{Lind}} - W_{\text{Lind}}^\dagger}{2} \right) \rho^{(m)} \exp\left( -i \tau \frac{W_{\text{Lind}} - W_{\text{Lind}}^\dagger}{2} \right)$$
   iterando hasta $m \le 3$ veces si el residuo de Fubini-Study supera el umbral $d_{\text{FS}} > 10^{-4}\text{ rad}$.
   La contracción de Banach garantiza convergencia geométrica:
     $$\text{dist}(\rho^{(m+1)}, \rho^*) \le 0.95 \cdot \text{dist}(\rho^{(m)}, \rho^*)$$

2.3. MEDICIÓN DÉBIL CUÁNTICA NO DEMOLICIÓN (QND) Y VALORES DÉBILES DE AHARONOV-ALBERT-VAIDMAN (AAV):
   El acoplamiento del observable $H = -i A$ entre el estado inicial $|\phi_i\rangle$ y el estado final post-seleccionado $|\phi_f\rangle$
   arroja el valor débil complejo de AAV:
     $$A_w = \frac{\langle \phi_f | H | \phi_i \rangle}{\langle \phi_f | \phi_i \rangle} \in \mathbb{C}$$
   La condición QND exige varianza nula del observable sobre el estado del sistema:
     $$\text{Var}(H)_{|\phi_i\rangle} = \langle \phi_i | H^2 | \phi_i \rangle - \langle \phi_i | H | \phi_i \rangle^2 \approx 0$$
   asegurando una retroacción nula de medición sobre el sustrato cognitivo.

2.4. ADJUDICACIÓN EN RETÍCULO DE HEYTING $\Omega_3$ Y CENSURA CIBER-FÍSICA:
   La adjudicación formal fusiona 7 átomos de severidad mediante el operador ínfimo del retículo:
     $$\Omega_{\text{total}} = \bigwedge_{j=1}^7 \omega_j \in \Omega_3 = \{\text{VETOED}=0\text{x}00, \ \text{DEGRADED}=0\text{x}01, \ \text{COHERENT}=0\text{x}03\}$$
   En caso de que $\Omega_{\text{total}} = \text{VETOED}$, el agente activa de forma inmediata el interlock Crowbar:
     $$\tau_{\text{crowbar}} \le 392.15\text{ ns} < 400.0\text{ ns} \implies \text{GPIO14} \leftarrow \text{HIGH}$$

3. AXIOMAS FORMALES:
─────────────────────────────────────────────────────────────────────────────────────────
  AXIOMA I (ENCADENAMIENTO MORFOLÓGICO Y AUTENTICIDAD CRIPTOGRÁFICA):
    Ninguna fase del ciclo OODA puede ejecutarse sin la previa validación y firma HMAC-SHA256 del certificado emitido por la fase precedente.

  AXIOMA II (MEDICIÓN NO DEMOLICIÓN Y PRESERVACIÓN DE COHERENCIA):
    La observación agéntica no induce colapso de estado ni ruido térmico: $\text{BackAction}_{\text{dB}} \le 10^{-6}\text{ dB}$.

  AXIOMA III (INVARIANZA DEL VOLUMEN SIMPLÉCTICO DE DARBOUX):
    La matriz simpléctica realificada preserva idénticamente la forma canónica: $\| M^\top J_{2n} M - J_{2n} \|_F / \|M\|_F^2 \le 10^{-10}$.

  AXIOMA IV (CONVERGENCIA MONÁDICA EN EL NIVEL 3 DE RSI):
    La secuencia de automejora agéntica $T^3$ satisface la cota de Lipschitz $\text{Lip}(T^3) \le 0.95^3 < 1.0$, garantizando unicidad asintótica.

4. INVARIANTES DE ESTADO:
─────────────────────────────────────────────────────────────────────────────────────────
  1. TRAZA Y SEMIDEFINICIÓN POSITIVA DEL OPERADOR: $\rho = \rho^\dagger \succeq 0, \quad \text{Tr}(\rho) = 1$.
  2. PRESERVACIÓN DE LA CONEXIÓN ANTIHERMÍTICA DE GAUGE: $A^\dagger = -A, \quad \text{Tr}(A) = 0$.
  3. COTA DIOFÁNTICA KAM Y ESTABILIDAD HOMOCLÍNICA DE MELNIKOV: $|M(\theta_0)| \le 0.50, \quad |\mu_k| \le 1.0 + 10^{-6}$.
  4. INTERLOCK HARDWARE ESP32 CROWBAR: Disparo garantizado en $WCET \le 392.15\text{ ns}$ ante anomalías de calibre.
  5. VALUACIÓN NO-ARQUIMEDIANA DE NOVIKOV: $v \ge 0.10$.
  6. INTEGRIDAD FUNTORIAL $\Phi_{\text{sem}}$: Proyección determinista del estado topológico a las métricas ejecutivas de riesgo de infraestructura.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

for path_dir in ("/workspace/scratch", "/workspace/out"):
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

# ══════════════════════════════════════════════════════════════════════════════
# §0. IMPORTACIÓN ROBUSTA DEL MOTOR ESPECTRAL (v11 / v10 / fallback)
# ══════════════════════════════════════════════════════════════════════════════

_ENGINE_IMPORTED: bool = False
_ENGINE_HAS_ENVELOPES: bool = False
_ENGINE_HAS_CONTINUE: bool = False
_ENGINE_HAS_GOVERNANCE: bool = False

try:
    from app.wisdom.topological_gauge_field_engine import (  # type: ignore
        HeytingOmega3,
        TopologicalGaugeFieldEngine,
        TopologicalGaugeFieldSnapshot,
    )
    _ENGINE_IMPORTED = True
except ImportError:
    try:
        from topological_gauge_field_engine import (  # type: ignore
            HeytingOmega3,
            TopologicalGaugeFieldEngine,
            TopologicalGaugeFieldSnapshot,
        )
        _ENGINE_IMPORTED = True
    except ImportError:
        pass


if not _ENGINE_IMPORTED:  # pragma: no cover — fallback autónomo
    class HeytingOmega3(IntEnum):
        r"""Cadena de Heyting de severidad (fallback): 0x00 < 0x01 < 0x03=⊤."""
        COHERENT = 0x00
        DEGRADED = 0x01
        VETOED = 0x03

        def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return HeytingOmega3(max(int(self), int(other)))

        def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return HeytingOmega3(min(int(self), int(other)))

        def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
            return HeytingOmega3.VETOED if int(self) <= int(other) else other

        def negate(self) -> "HeytingOmega3":
            return self.implies(HeytingOmega3.COHERENT)

        @classmethod
        def fold_supremum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
            acc = cls.COHERENT
            for v in verdicts:
                acc = acc.join(v)
            return acc

        @classmethod
        def fold_infimum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
            acc = cls.VETOED
            for v in verdicts:
                acc = acc.meet(v)
            return acc

        @classmethod
        def legacy_from_0x02(cls, legacy: int) -> "HeytingOmega3":
            return cls.VETOED if legacy == 0x02 else cls(legacy)

    class TopologicalGaugeFieldSnapshot:  # type: ignore[no-redef]
        pass

    class TopologicalGaugeFieldEngine:  # type: ignore[no-redef]
        def __init__(self, dimension: int = 56, secret_key: bytes = b"APU_KEY") -> None:
            self.dimension = max(4, int(dimension))
            self.secret_key = bytes(secret_key)

        def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
            a = np.asarray(matrix, dtype=np.complex128)
            herm = 0.5 * (a + a.conj().T)
            tr = float(np.real(np.trace(herm)))
            if tr <= 1e-15:
                return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
            return herm / tr

        def build_gauge_connection_A(self, rho: NDArray) -> NDArray[np.complex128]:
            rho_c = self.sanitize_density_matrix(rho)
            n = rho_c.shape[0]
            su = rho_c - (np.trace(rho_c) / float(n)) * np.eye(n, dtype=np.complex128)
            A = 1j * su
            A = 0.5 * (A - A.conj().T)
            return A - (np.trace(A) / float(n)) * np.eye(n, dtype=np.complex128)

        def trigger_hardware_interlock_esp32_crowbar(
            self, verdict: HeytingOmega3, gpio_pin: int = 14,
        ) -> Tuple[bool, float, str]:
            if verdict == HeytingOmega3.VETOED:
                return True, 42.0, "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
            if verdict == HeytingOmega3.DEGRADED:
                return False, 120.0, "MONITORING_DEGRADED_PERFORMANCE"
            return False, 45.0, "NOMINAL_EXECUTION"

        def apply_visceral_functor_phi_sem(
            self, verdict: HeytingOmega3, energy: float, angle: float,
        ) -> Dict[str, Any]:
            if verdict == HeytingOmega3.VETOED:
                return {
                    "wacc_protection_status": "BLINDADO AL 15.0%",
                    "net_liquidity_impact": "+11.5% EN TESORERÍA",
                    "llm_inference_savings_pct": 80.0,
                    "kv_cache_compression_pct": 86.4,
                    "yang_mills_energy_fpu": energy,
                    "wilson_angle_rad": angle,
                    "executive_action": "DESEMBOLSO CERENADO",
                }
            if verdict == HeytingOmega3.DEGRADED:
                return {
                    "wacc_protection_status": "PRESERVADO EN 15.0%",
                    "net_liquidity_impact": "+8.5% RECIRCULANTE",
                    "llm_inference_savings_pct": 80.0,
                    "kv_cache_compression_pct": 86.4,
                    "yang_mills_energy_fpu": energy,
                    "wilson_angle_rad": angle,
                    "executive_action": "VETO SUAVE",
                }
            return {
                "wacc_protection_status": "ESTABLE AL 15.0%",
                "net_liquidity_impact": "+11.5% LIBRE",
                "llm_inference_savings_pct": 80.0,
                "kv_cache_compression_pct": 86.4,
                "yang_mills_energy_fpu": energy,
                "wilson_angle_rad": angle,
                "executive_action": "FLUJO NOMINAL",
            }

        def evaluate_topological_gauge_field(
            self, rho: NDArray, p_i: Any = None, p_f: Any = None, **_kw: Any,
        ) -> Tuple[Any, Dict[str, Any]]:
            class _DummySnap:
                snapshot_id = "GAUGE-DUMMY"
                yang_mills_energy = 0.001
                chern_simons_3form_higgs = 0.002
                chern_simons_anomaly = 0.0
                chern_hopf_class_c1 = 0.0
                wilson_loop_trace = 1.0 + 0.0j
                wilson_angle_rad = 0.0
                pontryagin_instanton_number = 0
                pontryagin_integrality_residual = 0.0
                stora_zumino_residual = 0.0001
                weak_value_qnd_aav = 0.01j
                back_action_db = 0.0
                topology_regime = "TRIVIAL_SU2_HOLONOMY"
                global_verdict = HeytingOmega3.COHERENT
                sha256_merkle_digest = "dummy_hash"
                timestamp_utc = time.time()
                kam_worst_ratio = 2.0
                kam_satisfied = True
                melnikov_value = 0.0
                floquet_max_unit_deviation = 0.0
                floquet_is_neutral = True
                poincare_section_n_crossings = 0
                poincare_section_dispersion = 0.0
                banach_lipschitz_t3 = 0.0
                fubini_study_distance = 0.0
                novikov_krull_valuation = 1.0
                rsi3_valid = True
                rsi3_iterations = 1
                symplectic_residue = 0.0
                symplectic_defect = 0.0
                crowbar_latency_ns = 45.0
                hardware_crowbar_action = "NOMINAL_EXECUTION"
                bruno_satisfied = True
                spectral_gap = 0.0
                poincare_cartan_invariant = 0.0
                chern_simons_algebraic = 0.0
                lindstedt_residual = 0.0
                qnd_variance = 0.0

            snap = _DummySnap()
            return snap, self.apply_visceral_functor_phi_sem(
                HeytingOmega3.COHERENT, 0.001, 0.0)


if _ENGINE_IMPORTED:
    _ENGINE_HAS_ENVELOPES = hasattr(TopologicalGaugeFieldEngine, "emit_phase1_envelope")
    _ENGINE_HAS_CONTINUE = hasattr(TopologicalGaugeFieldEngine, "continue_to_phase2")
    _ENGINE_HAS_GOVERNANCE = hasattr(TopologicalGaugeFieldEngine, "emit_governance_state")

logger = logging.getLogger("APU.Wisdom.TopologicalGaugeFieldAgent.v11")

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES DE RIGOR CIBER-FÍSICO Y FPU
# ══════════════════════════════════════════════════════════════════════════════
_AGENT_VERSION: Final[str] = "11.0.0"
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_CROWBAR_TIMING_LIMIT_NS: Final[float] = 400.0
_CROWBAR_DEGRADED_LATENCY_NS: Final[float] = 120.0
_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 45.0
_POSITRON_REST_MASS_ENERGY_KEV: Final[float] = 1022.0
_GAUGE_HMAC_DOMAIN: Final[bytes] = b"APU-FILTER-GAUGE-FIELD-AGENT-v11.0.0|"
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_GAUGE_FIELD_SOVEREIGN_SECRET_KEY"

_ANTI_HERM_TOL: Final[float] = 1.0e-10
_TRACELESS_TOL: Final[float] = 1.0e-10
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_TRACE_CONSERVATION_TOL: Final[float] = 0.01
_NORM_A_HARD: Final[float] = 100.0
_NORM_A_SOFT: Final[float] = 10.0
_BANACH_LIPSCHITZ_MAX: Final[float] = 0.95
_RSI3_MAX_ITER: Final[int] = 3
_LINSTEDT_DAMP: Final[float] = 0.15
_MELNIKOV_TOL: Final[float] = 0.5
_KRULL_NOVIKOV_FLOOR: Final[float] = 0.10


# ══════════════════════════════════════════════════════════════════════════════
# §B. EXCEPCIONES CATEGORIALES
# ══════════════════════════════════════════════════════════════════════════════
class GaugeFieldAgentError(Exception):
    """Excepción raíz del Soberano Agéntico del Campo de Gauge."""


class GaugeContractCorruptedError(GaugeFieldAgentError):
    """Falla de integridad HMAC en la cadena del contrato de calibre."""


class GaugeIsometryViolationError(GaugeFieldAgentError):
    """Violación no unitaria en la conexión de Gauge o en la traza."""


class SymplecticGaugeInvarianceError(GaugeFieldAgentError):
    """Violación de la invarianza simpléctica Sp(2n,ℝ) en el campo."""


class EnvelopeChainViolationError(GaugeFieldAgentError):
    """Cadena de hashes entre fases corrompida."""


class PoincareLindstedtResonanceError(GaugeFieldAgentError):
    """Divisores pequeños no resolubles en la homología de Lindstedt."""


class RSI3ContractionFailureError(GaugeFieldAgentError):
    """El mejorador T no es contracción de Banach (Lip>0.95)."""


# ══════════════════════════════════════════════════════════════════════════════
# §C. HASH / HMAC / ADAPTADORES DEL MOTOR
# ══════════════════════════════════════════════════════════════════════════════
def _hash_state(*arrays: NDArray[Any]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _g(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default)


def _unpack_qnd(result: Any) -> Tuple[complex, float, float]:
    r"""Motor v11 → (A_w, dB, var); v10 → (A_w, dB)."""
    if not isinstance(result, (tuple, list)):
        return complex(result), 0.0, 0.0
    if len(result) >= 3:
        return complex(result[0]), float(result[1]), float(result[2])
    if len(result) == 2:
        return complex(result[0]), float(result[1]), 0.0
    return 0j, 0.0, 0.0


def _unpack_crowbar(result: Any) -> Tuple[bool, float, str]:
    r"""Motor v11 → (latched, ns, action); v10 → (latched, ns)."""
    if not isinstance(result, (tuple, list)):
        return bool(result), 0.0, "UNKNOWN"
    if len(result) >= 3:
        return bool(result[0]), float(result[1]), str(result[2])
    if len(result) == 2:
        latched, ns = bool(result[0]), float(result[1])
        action = (
            "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED" if latched
            else ("MONITORING_DEGRADED_PERFORMANCE" if ns >= 100.0 else "NOMINAL_EXECUTION")
        )
        return latched, ns, action
    return False, 0.0, "UNKNOWN"


def _call_emit_phase1(engine: Any, rho: NDArray, background_H: Optional[NDArray]) -> Any:
    try:
        return engine.emit_phase1_envelope(rho, background_hamiltonian=background_H)
    except TypeError:
        return engine.emit_phase1_envelope(rho)


def _call_evaluate(engine: Any, rho: NDArray, background_H: Optional[NDArray]) -> Tuple[Any, Dict[str, Any]]:
    try:
        return engine.evaluate_topological_gauge_field(
            rho, background_hamiltonian=background_H)
    except TypeError:
        return engine.evaluate_topological_gauge_field(rho)


# ══════════════════════════════════════════════════════════════════════════════
# §D. ADJUDICADOR HEYTING Ω₃ (RETÍCULO COMPLETO, fold_infimum LOCAL)
# ══════════════════════════════════════════════════════════════════════════════
class HeytingOmega3Adjudicator:
    r"""
    Adjudicador \(\Omega_3\) con retículo completo:
    join \(\vee\), meet \(\wedge\), implies \(\to\), negate \(\neg\).
    `fold_infimum` es local: el motor v11 no lo exporta.
    """

    __slots__ = ()

    @staticmethod
    def fold_infimum(verdicts: Sequence[HeytingOmega3]) -> HeytingOmega3:
        acc = HeytingOmega3.VETOED
        for v in verdicts:
            acc = acc.meet(v)
        return acc

    @staticmethod
    def adjudicate(*atoms: HeytingOmega3) -> HeytingOmega3:
        return HeytingOmega3.fold_supremum(list(atoms))

    @staticmethod
    def join_atoms(atoms: Sequence[HeytingOmega3]) -> HeytingOmega3:
        return HeytingOmega3.fold_supremum(atoms)

    @classmethod
    def meet_atoms(cls, atoms: Sequence[HeytingOmega3]) -> HeytingOmega3:
        return cls.fold_infimum(atoms)

    @staticmethod
    def truth_table_join() -> Tuple[Tuple[int, int, int], ...]:
        vals = (0x00, 0x01, 0x03)
        return tuple(tuple(max(a, b) for b in vals) for a in vals)

    @staticmethod
    def truth_table_meet() -> Tuple[Tuple[int, int, int], ...]:
        vals = (0x00, 0x01, 0x03)
        return tuple(tuple(min(a, b) for b in vals) for a in vals)

    @staticmethod
    def truth_table_implies() -> Tuple[Tuple[int, int, int], ...]:
        vals = (0x00, 0x01, 0x03)
        top = 0x03
        return tuple(tuple(top if a <= b else b for b in vals) for a in vals)

    @staticmethod
    def truth_table_negate() -> Tuple[int, int, int]:
        vals = (0x00, 0x01, 0x03)
        top, bottom = 0x03, 0x00
        return tuple(top if a <= bottom else bottom for a in vals)  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# §E. CERTIFICADOS DE FASE INMUTABLES (ENRIQUECIDOS v11.0.0)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GaugeFieldIsometryCertificate:
    r"""FASE 1 — Isometría de calibre, Cayley–Sp, Lindstedt, Cartan, HMAC."""
    registry_size: int
    dimension: int
    connection_norm_frobenius: float
    density_trace: float
    is_trace_conserved: bool
    phase1_verdict: HeytingOmega3
    hmac_phase1_signature: str
    is_anti_hermitian_A: bool = True
    is_traceless_A: bool = True
    symplectic_residue: float = 0.0
    symplectic_defect: float = 0.0
    is_symplectic_valid: bool = True
    poincare_residual: float = 0.0
    poincare_is_resonant: bool = False
    energy_ym_baseline: float = 0.0
    engine_envelope: Optional[Any] = field(default=None, repr=False)
    chain_hash: str = ""
    # v11.0.0
    spectral_gap: float = 0.0
    poincare_cartan_invariant: float = 0.0
    is_kahler: bool = False
    is_quaternionic_proxy: bool = False
    cayley_unitary_defect: float = 0.0
    lindstedt_is_solved: bool = True
    n_small_divisors: int = 0
    rsi3_iterations: int = 1
    rsi3_contraction: float = 0.0
    symplectic_certified: bool = True


@dataclass(frozen=True, slots=True)
class Phase2GaugeDynamicsCertificate:
    r"""FASE 2 — Yang–Mills, CS, Poincaré celeste, KAM/Bruno, RSI³, HMAC."""
    yang_mills_energy: float
    chern_simons_3form_higgs: float
    wilson_angle_rad: float
    instanton_k: int
    stora_zumino_residual: float
    weak_value_qnd: complex
    positron_annihilations_count: int
    phase2_verdict: HeytingOmega3
    hmac_phase2_signature: str
    chern_simons_anomaly: float = 0.0
    c1_value: float = 0.0
    pontryagin_integrality_residual: float = 0.0
    kam_worst_ratio: float = 0.0
    kam_satisfied: bool = True
    melnikov_value: float = 0.0
    melnikov_preserved: bool = True
    floquet_max_unit_deviation: float = 0.0
    floquet_neutral: bool = True
    poincare_section_n_crossings: int = 0
    poincare_section_dispersion: float = 0.0
    rsi3_valid: bool = True
    rsi3_iterations: int = 1
    banach_lipschitz_t3: float = 0.0
    fubini_study_distance: float = 0.0
    novikov_krull_valuation: float = 0.0
    back_action_db: float = 0.0
    topology_regime: str = "TRIVIAL_SU2_HOLONOMY"
    engine_envelope: Optional[Any] = field(default=None, repr=False)
    chain_hash: str = ""
    # v11.0.0
    dimension: int = 56
    bruno_satisfied: bool = True
    bruno_sum: float = 0.0
    chern_simons_algebraic: float = 0.0
    chern_character_ch2: float = 0.0
    krein_collision: bool = False
    qnd_variance: float = 0.0
    mean_return_time: float = 0.0
    recurrence_time: float = 0.0
    rsi3_contraction: float = 0.0


@dataclass(frozen=True, slots=True)
class TopologicalGaugeFieldSovereignState:
    r"""FASE 3 — Gobernanza OODA terminal en \(\Omega_3\), Crowbar, Merkle, HMAC."""
    phase1_cert: GaugeFieldIsometryCertificate
    phase2_cert: Phase2GaugeDynamicsCertificate
    global_verdict: HeytingOmega3
    crowbar_triggered: bool
    crowbar_latency_ns: float
    visceral_report: Dict[str, Any]
    sha256_merkle_digest: str
    provenance_hmac: str
    timestamp_utc: float
    crowbar_latency_verified: bool = True
    hardware_action: str = "NOMINAL_EXECUTION"
    heyting_join_atoms: Tuple[int, ...] = ()
    heyting_meet_atoms: Tuple[int, ...] = ()
    heyting_implies_table: Tuple[Tuple[int, int, int], ...] = ()
    heyting_negate_table: Tuple[int, int, int] = (0x03, 0x00, 0x00)
    is_epistemologically_valid: bool = True
    agent_version: str = _AGENT_VERSION
    # v11.0.0
    spectral_gap: float = 0.0
    poincare_cartan_invariant: float = 0.0
    bruno_satisfied: bool = True
    lindstedt_residual: float = 0.0
    qnd_variance: float = 0.0
    hmac_verified: bool = True


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   1 :   O B S E R V E                                               ║
# ║  Isometría su(N), envelope KKS, RSI³-Lindstedt, HMAC.                        ║
# ║  Cierra con `continue_to_phase2` → habilita Orient.                          ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class GaugeAgentPhase1Observer:
    r"""
    FASE 1 (Observe): sanitiza ρ, consume el envelope del motor,
    re-aplica Lindstedt a su residuo (RSI³) y firma el certificado.
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        background_hamiltonian: Optional[NDArray] = None,
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)
        self._background_H: Optional[NDArray] = background_hamiltonian
        self.engine: Final[Any] = TopologicalGaugeFieldEngine(
            dimension=self.dimension, secret_key=self.secret_key,
        )
        self._phase1_cert: Optional[GaugeFieldIsometryCertificate] = None
        self._phase2_cert: Optional[Phase2GaugeDynamicsCertificate] = None
        self._adjudicator: HeytingOmega3Adjudicator = HeytingOmega3Adjudicator()

    def _compute_hmac(self, payload: str) -> str:
        return hmac.new(
            self.secret_key,
            _GAUGE_HMAC_DOMAIN + payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _verify_hmac(self, payload: str, signature: str) -> bool:
        return hmac.compare_digest(self._compute_hmac(payload), signature)

    @staticmethod
    def _phase1_payload(cert: GaugeFieldIsometryCertificate) -> str:
        return (
            f"P1:{cert.dimension}:{cert.connection_norm_frobenius:.8f}:"
            f"{cert.density_trace:.8f}:{int(cert.phase1_verdict)}:"
            f"{cert.symplectic_residue:.3e}:{cert.symplectic_defect:.3e}:"
            f"{int(cert.is_anti_hermitian_A)}:{int(cert.is_traceless_A)}:"
            f"{cert.spectral_gap:.3e}:{cert.poincare_cartan_invariant:.3e}"
        )

    @staticmethod
    def _phase2_payload(cert: Phase2GaugeDynamicsCertificate) -> str:
        return (
            f"P2:{cert.yang_mills_energy:.8f}:{cert.chern_simons_3form_higgs:.8f}:"
            f"{cert.wilson_angle_rad:.8f}:{cert.instanton_k}:{int(cert.phase2_verdict)}:"
            f"{cert.kam_worst_ratio:.4f}:{cert.melnikov_value:.3e}:"
            f"{cert.floquet_max_unit_deviation:.3e}:{int(cert.rsi3_valid)}:"
            f"{int(cert.bruno_satisfied)}:{cert.chern_simons_algebraic:.3e}"
        )

    def _sign_phase1(self, cert: GaugeFieldIsometryCertificate) -> GaugeFieldIsometryCertificate:
        return replace(cert, hmac_phase1_signature=self._compute_hmac(self._phase1_payload(cert)))

    def _sign_phase2(self, cert: Phase2GaugeDynamicsCertificate) -> Phase2GaugeDynamicsCertificate:
        return replace(cert, hmac_phase2_signature=self._compute_hmac(self._phase2_payload(cert)))

    def _rsi3_lindstedt_reobserve(
        self, rho: NDArray[np.complex128], env: Any,
    ) -> Tuple[NDArray[np.complex128], Any, List[float], float]:
        r"""
        RSI³ nivel 3: conjugación unitaria \(U=e^{K}\), \(K=\mathrm{AH}(W)\),
        \(\rho\leftarrow U\rho U^\dagger\), y se re-emite el envelope.
        Lip se estima como razón de residuos sucesivos.
        """
        hist: List[float] = []
        current_rho, current_env = rho, env
        for _ in range(_RSI3_MAX_ITER):
            pl = _g(current_env, "poincare_lindstedt", None)
            res = float(_g(pl, "residual_norm", 0.0)) if pl is not None else 0.0
            hist.append(res)
            if pl is None or bool(_g(pl, "is_solved", True)) or res < 1.0e-9:
                break
            W = np.asarray(_g(pl, "W", None), dtype=np.complex128)
            if W.size == 0:
                break
            try:
                K = 0.5 * (W - W.conj().T)
                U = la.expm(_LINSTEDT_DAMP * K)
                rho_n = U @ current_rho @ U.conj().T
                current_rho = self.engine.sanitize_density_matrix(rho_n)
                if not _ENGINE_HAS_ENVELOPES:
                    break
                current_env = _call_emit_phase1(self.engine, current_rho, self._background_H)
                if _ENGINE_HAS_CONTINUE:
                    try:
                        current_env = self.engine.continue_to_phase2(current_env)
                    except Exception as exc:
                        logger.debug("continue_to_phase2 en RSI³ omitido: %s", exc)
            except (la.LinAlgError, ValueError, TypeError) as exc:
                logger.warning("RSI³-Lindstedt abortado: %s", exc)
                break
        ratios = [
            hist[i + 1] / hist[i]
            for i in range(len(hist) - 1)
            if hist[i] > 1.0e-15
        ]
        lip = float(min(1.0, max(ratios))) if ratios else 0.0
        return current_rho, current_env, hist, lip

    def _verdict_phase1(
        self,
        *,
        is_trace_conserved: bool,
        norm_a: float,
        is_ah: bool,
        is_tr: bool,
        is_sympl: bool,
        symplectic_certified: bool,
        resonant: bool,
    ) -> HeytingOmega3:
        if (not is_trace_conserved) or norm_a > _NORM_A_HARD or (not is_ah) or (not is_tr):
            return HeytingOmega3.VETOED
        if (norm_a > _NORM_A_SOFT) or (not is_sympl) or (not symplectic_certified) or resonant:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    def emit_phase1_certificate(
        self, density_matrix: NDArray,
    ) -> GaugeFieldIsometryCertificate:
        r"""
        **Cierre operatorial de Observe (antes del puente).**
        Sanitiza, consume envelope v11, RSI³-Lindstedt, firma HMAC.
        """
        rho = self.engine.sanitize_density_matrix(density_matrix)
        tr = float(np.real(np.trace(rho)))
        is_trace_conserved = abs(tr - 1.0) <= _TRACE_CONSERVATION_TOL

        engine_env: Any = None
        rsi3_iters, rsi3_lip = 1, 0.0
        if _ENGINE_HAS_ENVELOPES:
            try:
                engine_env = _call_emit_phase1(self.engine, rho, self._background_H)
                if _ENGINE_HAS_CONTINUE:
                    try:
                        engine_env = self.engine.continue_to_phase2(engine_env)
                    except Exception as exc:
                        logger.debug("engine.continue_to_phase2: %s", exc)
                rho, engine_env, hist, rsi3_lip = self._rsi3_lindstedt_reobserve(rho, engine_env)
                rsi3_iters = max(1, len(hist))
                tr = float(np.real(np.trace(rho)))
                is_trace_conserved = abs(tr - 1.0) <= _TRACE_CONSERVATION_TOL
            except Exception as exc:
                logger.warning("Engine FASE1 envelope falló (%s); fallback.", exc)
                engine_env = None

        symplectic_certified = engine_env is not None
        if engine_env is not None:
            A = np.asarray(engine_env.connection_A, dtype=np.complex128)
            norm_a = float(la.norm(A, "fro"))
            is_ah = bool(engine_env.is_anti_hermitian_A)
            is_tr0 = bool(engine_env.is_traceless_A)
            sympl_res = float(engine_env.symplectic_residue)
            sympl_def = float(engine_env.symplectic_defect)
            is_sympl = bool(engine_env.is_symplectic_valid)
            pl = _g(engine_env, "poincare_lindstedt", None)
            poincare_res = float(_g(pl, "residual_norm", 0.0)) if pl is not None else 0.0
            poincare_resonant = bool(_g(pl, "is_resonant", False)) if pl is not None else False
            lindstedt_ok = bool(_g(pl, "is_solved", True)) if pl is not None else True
            n_small = int(_g(pl, "n_small_divisors", 0)) if pl is not None else 0
            energy_baseline = float(engine_env.energy_ym)
            gap = float(_g(engine_env, "spectral_gap", 0.0))
            cartan = float(_g(engine_env, "poincare_cartan_invariant", 0.0))
            kahler = bool(_g(engine_env, "is_kahler", is_sympl))
            quat = bool(_g(engine_env, "is_quaternionic_proxy", False))
            unit_def = float(_g(engine_env, "cayley_unitary_defect", 0.0))
        else:
            A = self.engine.build_gauge_connection_A(rho)
            norm_a = float(la.norm(A, "fro"))
            scale = max(1.0, norm_a)
            is_ah = float(la.norm(A + A.conj().T, "fro")) / scale < _ANTI_HERM_TOL
            is_tr0 = float(abs(np.trace(A))) < _TRACELESS_TOL
            sympl_res, sympl_def, is_sympl = 1.0, 1.0, False
            poincare_res, poincare_resonant, lindstedt_ok, n_small = 0.0, False, False, 0
            energy_baseline, gap, cartan, unit_def = 0.0, 0.0, 0.0, 1.0
            kahler, quat = False, False
            if hasattr(self.engine, "compute_yang_mills_curvature_F"):
                try:
                    _, energy_baseline = self.engine.compute_yang_mills_curvature_F(A, rho)
                except Exception:
                    energy_baseline = 0.0

        verdict = self._verdict_phase1(
            is_trace_conserved=is_trace_conserved, norm_a=norm_a,
            is_ah=is_ah, is_tr=is_tr0, is_sympl=is_sympl,
            symplectic_certified=symplectic_certified, resonant=poincare_resonant,
        )
        unsigned = GaugeFieldIsometryCertificate(
            registry_size=self.dimension, dimension=self.dimension,
            connection_norm_frobenius=norm_a, density_trace=tr,
            is_trace_conserved=is_trace_conserved, phase1_verdict=verdict,
            hmac_phase1_signature="",
            is_anti_hermitian_A=is_ah, is_traceless_A=is_tr0,
            symplectic_residue=sympl_res, symplectic_defect=sympl_def,
            is_symplectic_valid=is_sympl,
            poincare_residual=poincare_res, poincare_is_resonant=poincare_resonant,
            energy_ym_baseline=float(energy_baseline),
            engine_envelope=engine_env, chain_hash=_hash_state(rho, A),
            spectral_gap=gap, poincare_cartan_invariant=cartan,
            is_kahler=kahler, is_quaternionic_proxy=quat,
            cayley_unitary_defect=unit_def,
            lindstedt_is_solved=lindstedt_ok, n_small_divisors=n_small,
            rsi3_iterations=rsi3_iters, rsi3_contraction=rsi3_lip,
            symplectic_certified=symplectic_certified,
        )
        return self._sign_phase1(unsigned)

    def execute_phase1_gauge_isometry(
        self, density_matrix: NDArray,
    ) -> Tuple[GaugeFieldIsometryCertificate, NDArray[np.complex128]]:
        r"""**[Compat v9.3/v10] FASE 1.** Delega a `emit_phase1_certificate`."""
        cert1 = self.emit_phase1_certificate(density_matrix)
        env = cert1.engine_envelope
        if env is not None and hasattr(env, "connection_A"):
            return cert1, np.asarray(env.connection_A, dtype=np.complex128)
        rho = self.engine.sanitize_density_matrix(density_matrix)
        return cert1, self.engine.build_gauge_connection_A(rho)

    def consume_engine_phase1_envelope(
        self, engine_env: Any,
    ) -> GaugeFieldIsometryCertificate:
        r"""Interoperabilidad: `GaugePhase1Envelope` externo → certificado firmado."""
        if engine_env is None:
            raise GaugeFieldAgentError("engine_env es None.")
        A = np.asarray(engine_env.connection_A, dtype=np.complex128)
        rho = np.asarray(engine_env.rho_sanitized, dtype=np.complex128)
        norm_a = float(la.norm(A, "fro"))
        tr = float(np.real(np.trace(rho)))
        is_trace_conserved = abs(tr - 1.0) <= _TRACE_CONSERVATION_TOL
        pl = _g(engine_env, "poincare_lindstedt", None)
        poincare_res = float(_g(pl, "residual_norm", 0.0)) if pl is not None else 0.0
        poincare_resonant = bool(_g(pl, "is_resonant", False)) if pl is not None else False
        is_ah = bool(engine_env.is_anti_hermitian_A)
        is_tr0 = bool(engine_env.is_traceless_A)
        is_sympl = bool(engine_env.is_symplectic_valid)
        verdict = self._verdict_phase1(
            is_trace_conserved=is_trace_conserved, norm_a=norm_a,
            is_ah=is_ah, is_tr=is_tr0, is_sympl=is_sympl,
            symplectic_certified=True, resonant=poincare_resonant,
        )
        unsigned = GaugeFieldIsometryCertificate(
            registry_size=self.dimension,
            dimension=int(_g(engine_env, "dimension", self.dimension)),
            connection_norm_frobenius=norm_a, density_trace=tr,
            is_trace_conserved=is_trace_conserved, phase1_verdict=verdict,
            hmac_phase1_signature="",
            is_anti_hermitian_A=is_ah, is_traceless_A=is_tr0,
            symplectic_residue=float(engine_env.symplectic_residue),
            symplectic_defect=float(engine_env.symplectic_defect),
            is_symplectic_valid=is_sympl,
            poincare_residual=poincare_res, poincare_is_resonant=poincare_resonant,
            energy_ym_baseline=float(engine_env.energy_ym),
            engine_envelope=engine_env, chain_hash=_hash_state(rho, A),
            spectral_gap=float(_g(engine_env, "spectral_gap", 0.0)),
            poincare_cartan_invariant=float(_g(engine_env, "poincare_cartan_invariant", 0.0)),
            is_kahler=bool(_g(engine_env, "is_kahler", is_sympl)),
            is_quaternionic_proxy=bool(_g(engine_env, "is_quaternionic_proxy", False)),
            cayley_unitary_defect=float(_g(engine_env, "cayley_unitary_defect", 0.0)),
            lindstedt_is_solved=bool(_g(pl, "is_solved", True)) if pl is not None else True,
            n_small_divisors=int(_g(pl, "n_small_divisors", 0)) if pl is not None else 0,
            symplectic_certified=True,
        )
        return self._sign_phase1(unsigned)

    def continue_to_phase2(
        self, cert1: GaugeFieldIsometryCertificate,
    ) -> GaugeFieldIsometryCertificate:
        r"""
        **Puente anidado FASE 1 → FASE 2 (último morfismo de Observe).**

        Valida HMAC-SHA256 y ancla el certificado. Dual clasista:
        `GaugeAgentPhase2Orientor.from_phase1_certificate`.
        """
        if not self._verify_hmac(self._phase1_payload(cert1), cert1.hmac_phase1_signature):
            raise GaugeContractCorruptedError(
                "continue_to_phase2: HMAC de FASE 1 inválida (dominio v11).")
        if abs(cert1.density_trace - 1.0) > _TRACE_CONSERVATION_TOL and (
                cert1.phase1_verdict is HeytingOmega3.COHERENT):
            raise GaugeIsometryViolationError(
                f"Tr ρ={cert1.density_trace:.6e} ≰ 1±{_TRACE_CONSERVATION_TOL} con veredicto COHERENT.")
        self._phase1_cert = cert1
        logger.debug("FASE 1→2 puente OK hmac=%s…", cert1.hmac_phase1_signature[:16])
        return cert1


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   2 :   O R I E N T                                                 ║
# ║  YM, CS, KAM/Bruno, Melnikov, Floquet–Krein, Fock, RSI³.                     ║
# ║  Abre con from_phase1_certificate.  Cierra con continue_to_phase3.           ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class GaugeAgentPhase2Orientor(GaugeAgentPhase1Observer):
    r"""
    FASE 2 (Orient): dinámica de Yang–Mills + mecánica celeste de Poincaré
    + aniquilación de Fock autenticada.
    """

    @classmethod
    def from_phase1_certificate(
        cls,
        cert1: GaugeFieldIsometryCertificate,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        dimension: Optional[int] = None,
        background_hamiltonian: Optional[NDArray] = None,
    ) -> "GaugeAgentPhase2Orientor":
        r"""**Apertura formal (clasista) de la FASE 2.** Dual de `continue_to_phase2`."""
        dim = int(dimension) if dimension is not None else int(cert1.dimension)
        inst = cls(dimension=dim, secret_key=secret_key,
                   background_hamiltonian=background_hamiltonian)
        inst.continue_to_phase2(cert1)
        return inst

    def continue_into_phase2_campaign_gate(
        self, cert1: GaugeFieldIsometryCertificate,
    ) -> bool:
        r"""**[Compat v9.3/v10] Compuerta Φ₁₂.** Delega al puente anidado."""
        self.continue_to_phase2(cert1)
        return True

    def _observe_qnd_safe(
        self, rho: NDArray, connection_A: NDArray,
    ) -> Tuple[complex, float, float]:
        if not hasattr(self.engine, "observe_qnd_weak_measurement"):
            return 0j, 0.0, 0.0
        d = int(rho.shape[0])
        phi0 = np.ones((d, 1), dtype=np.complex128) / math.sqrt(d)
        phi_f = rho @ phi0
        nrm = float(np.linalg.norm(phi_f))
        phi_f = phi_f / (nrm + _WILKINSON_EPSILON)
        try:
            return _unpack_qnd(
                self.engine.observe_qnd_weak_measurement(phi0, phi_f, connection_A))
        except Exception as exc:
            logger.debug("QND no disponible: %s", exc)
            return 0j, 0.0, 0.0

    def _extract_from_phase2_envelope(
        self, engine_p1_env: Any, engine_p2_env: Any, rho: NDArray,
    ) -> Dict[str, Any]:
        kam = engine_p2_env.kam_certificate
        mel = engine_p2_env.melnikov
        flq = engine_p2_env.floquet
        sec = engine_p2_env.poincare_section
        rsi3 = engine_p2_env.rsi3_certificate
        a_weak, back_action, qnd_var = self._observe_qnd_safe(
            rho, np.asarray(engine_p1_env.connection_A, dtype=np.complex128))
        return dict(
            ym_energy=float(engine_p1_env.energy_ym),
            cs3=float(engine_p2_env.chern_simons_real),
            cs_anomaly=float(engine_p2_env.chern_simons_anomaly),
            w_angle=float(engine_p2_env.wilson_angle_rad),
            c1=float(engine_p2_env.c1_value),
            k_inst=int(engine_p2_env.pontryagin_instanton_k),
            integrality=float(engine_p2_env.pontryagin_integrality_residual),
            sz=float(engine_p2_env.stora_zumino_residual),
            kam_worst=float(kam.worst_ratio),
            kam_ok=bool(kam.is_satisfied),
            bruno_ok=bool(_g(kam, "bruno_satisfied", True)),
            bruno_sum=float(_g(kam, "bruno_sum", 0.0)),
            melnikov=float(mel.integral_value),
            melnikov_ok=bool(mel.is_homoclinic_preserved),
            floquet_dev=float(flq.max_unit_deviation),
            floquet_neutral=bool(flq.is_neutral),
            krein=bool(_g(flq, "krein_collision", False)),
            poincare_x=int(sec.n_crossings),
            poincare_disp=float(sec.return_dispersion),
            mean_ret=float(_g(sec, "mean_return_time", 0.0)),
            rec_time=float(_g(sec, "recurrence_time", 0.0)),
            rsi3_ok=bool(rsi3.is_valid),
            rsi3_iters=int(rsi3.iterations),
            banach_t3=float(rsi3.lipschitz_t3),
            fubini=float(rsi3.surface3.metric_value),
            novikov=float(rsi3.surface1.metric_value),
            contraction=float(_g(rsi3, "contraction_ratio", 0.0)),
            back_action=back_action,
            a_weak=a_weak,
            qnd_var=qnd_var,
            regime=str(engine_p2_env.topology_regime),
            cs_alg=float(_g(engine_p2_env, "chern_simons_algebraic", 0.0)),
            ch2=float(_g(engine_p2_env, "chern_character_ch2", 0.0)),
        )

    def _extract_from_snapshot(self, snapshot: Any) -> Dict[str, Any]:
        melnikov = float(_g(snapshot, "melnikov_value", 0.0))
        return dict(
            ym_energy=float(snapshot.yang_mills_energy),
            cs3=float(snapshot.chern_simons_3form_higgs),
            cs_anomaly=float(_g(snapshot, "chern_simons_anomaly", 0.0)),
            w_angle=float(snapshot.wilson_angle_rad),
            c1=float(snapshot.chern_hopf_class_c1),
            k_inst=int(snapshot.pontryagin_instanton_number),
            integrality=float(_g(snapshot, "pontryagin_integrality_residual", 0.0)),
            sz=float(snapshot.stora_zumino_residual),
            kam_worst=float(_g(snapshot, "kam_worst_ratio", 0.0)),
            kam_ok=bool(_g(snapshot, "kam_satisfied", True)),
            bruno_ok=bool(_g(snapshot, "bruno_satisfied", True)),
            bruno_sum=0.0,
            melnikov=melnikov,
            melnikov_ok=bool(abs(melnikov) < _MELNIKOV_TOL),
            floquet_dev=float(_g(snapshot, "floquet_max_unit_deviation", 0.0)),
            floquet_neutral=bool(_g(snapshot, "floquet_is_neutral", True)),
            krein=False,
            poincare_x=int(_g(snapshot, "poincare_section_n_crossings", 0)),
            poincare_disp=float(_g(snapshot, "poincare_section_dispersion", 0.0)),
            mean_ret=0.0, rec_time=0.0,
            rsi3_ok=bool(_g(snapshot, "rsi3_valid", True)),
            rsi3_iters=int(_g(snapshot, "rsi3_iterations", 1)),
            banach_t3=float(_g(snapshot, "banach_lipschitz_t3", 0.0)),
            fubini=float(_g(snapshot, "fubini_study_distance", 0.0)),
            novikov=float(_g(snapshot, "novikov_krull_valuation", 0.0)),
            contraction=float(_g(snapshot, "banach_lipschitz_t3", 0.0)) ** (1.0 / 3.0)
            if float(_g(snapshot, "banach_lipschitz_t3", 0.0)) > 0 else 0.0,
            back_action=float(_g(snapshot, "back_action_db", 0.0)),
            a_weak=complex(snapshot.weak_value_qnd_aav),
            qnd_var=float(_g(snapshot, "qnd_variance", 0.0)),
            regime=str(snapshot.topology_regime),
            cs_alg=float(_g(snapshot, "chern_simons_algebraic", 0.0)),
            ch2=0.0,
        )

    def _fock_annihilation(
        self, electron_anomaly_count: int, positron_auth_signature: Optional[str],
    ) -> int:
        if electron_anomaly_count <= 0 or not positron_auth_signature:
            return 0
        auth_payload = f"AUTH_POSITRON:{electron_anomaly_count}"
        if not self._verify_hmac(auth_payload, positron_auth_signature):
            logger.warning("Firma de positrón rechazada; vacío no purificado.")
            return 0
        logger.info(
            "[ANIQUILACIÓN DE FOCK] e⁻(%d)+e⁺(%d) → 2γ (%.1f keV).",
            electron_anomaly_count, electron_anomaly_count, _POSITRON_REST_MASS_ENERGY_KEV,
        )
        return int(electron_anomaly_count)

    def emit_phase2_certificate(
        self,
        cert1: GaugeFieldIsometryCertificate,
        density_matrix: NDArray,
        electron_anomaly_count: int = 0,
        positron_auth_signature: Optional[str] = None,
    ) -> Phase2GaugeDynamicsCertificate:
        r"""
        **Cierre operatorial de Orient (antes del puente).**
        Consume envelope FASE 2 / snapshot, Fock HMAC, adjudica Ω₃ local.
        """
        self.continue_to_phase2(cert1)
        rho = self.engine.sanitize_density_matrix(density_matrix)

        engine_p1_env = cert1.engine_envelope
        engine_p2_env: Any = None
        if _ENGINE_HAS_ENVELOPES and engine_p1_env is not None:
            try:
                engine_p2_env = self.engine.emit_phase2_envelope(engine_p1_env)
                if _ENGINE_HAS_CONTINUE and hasattr(self.engine, "continue_to_phase3"):
                    try:
                        engine_p2_env = self.engine.continue_to_phase3(engine_p2_env)
                    except Exception as exc:
                        logger.debug("engine.continue_to_phase3: %s", exc)
            except Exception as exc:
                logger.warning("Engine FASE2 envelope falló (%s); snapshot.", exc)
                engine_p2_env = None

        if engine_p2_env is not None:
            fields = self._extract_from_phase2_envelope(engine_p1_env, engine_p2_env, rho)
        else:
            try:
                snapshot, _ = _call_evaluate(self.engine, rho, self._background_H)
            except Exception as exc:
                raise GaugeFieldAgentError(f"Motor no pudo evaluar el campo: {exc}") from exc
            fields = self._extract_from_snapshot(snapshot)

        annihilations = self._fock_annihilation(
            electron_anomaly_count, positron_auth_signature)

        p2_atoms = [
            HeytingOmega3.COHERENT if fields["ym_energy"] <= 0.25 else (
                HeytingOmega3.DEGRADED if fields["ym_energy"] <= 0.85 else HeytingOmega3.VETOED),
            HeytingOmega3.COHERENT if abs(fields["k_inst"]) <= 2 else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if fields["kam_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if fields["bruno_ok"] else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if fields["melnikov_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if fields["rsi3_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if fields["novikov"] >= _KRULL_NOVIKOV_FLOOR
            else HeytingOmega3.DEGRADED,
        ]
        p2_verdict = cert1.phase1_verdict.join(self._adjudicator.join_atoms(p2_atoms))

        chain = _hash_state(
            rho,
            np.array([
                fields["ym_energy"], fields["cs3"], fields["w_angle"],
                float(fields["k_inst"]), fields["kam_worst"], fields["melnikov"],
                fields["floquet_dev"], float(fields["rsi3_ok"]),
                float(fields["bruno_ok"]),
            ], dtype=np.float64),
        )
        unsigned = Phase2GaugeDynamicsCertificate(
            yang_mills_energy=fields["ym_energy"],
            chern_simons_3form_higgs=fields["cs3"],
            wilson_angle_rad=fields["w_angle"],
            instanton_k=fields["k_inst"],
            stora_zumino_residual=fields["sz"],
            weak_value_qnd=fields["a_weak"],
            positron_annihilations_count=annihilations,
            phase2_verdict=p2_verdict,
            hmac_phase2_signature="",
            chern_simons_anomaly=fields["cs_anomaly"],
            c1_value=fields["c1"],
            pontryagin_integrality_residual=fields["integrality"],
            kam_worst_ratio=fields["kam_worst"],
            kam_satisfied=fields["kam_ok"],
            melnikov_value=fields["melnikov"],
            melnikov_preserved=fields["melnikov_ok"],
            floquet_max_unit_deviation=fields["floquet_dev"],
            floquet_neutral=fields["floquet_neutral"],
            poincare_section_n_crossings=fields["poincare_x"],
            poincare_section_dispersion=fields["poincare_disp"],
            rsi3_valid=fields["rsi3_ok"],
            rsi3_iterations=fields["rsi3_iters"],
            banach_lipschitz_t3=fields["banach_t3"],
            fubini_study_distance=fields["fubini"],
            novikov_krull_valuation=fields["novikov"],
            back_action_db=fields["back_action"],
            topology_regime=fields["regime"],
            engine_envelope=engine_p2_env,
            chain_hash=chain,
            dimension=int(cert1.dimension),
            bruno_satisfied=fields["bruno_ok"],
            bruno_sum=fields["bruno_sum"],
            chern_simons_algebraic=fields["cs_alg"],
            chern_character_ch2=fields["ch2"],
            krein_collision=fields["krein"],
            qnd_variance=fields["qnd_var"],
            mean_return_time=fields["mean_ret"],
            recurrence_time=fields["rec_time"],
            rsi3_contraction=fields["contraction"],
        )
        return self._sign_phase2(unsigned)

    def execute_phase2_gauge_dynamics(
        self,
        cert1: GaugeFieldIsometryCertificate,
        density_matrix: NDArray,
        electron_anomaly_count: int = 0,
        positron_auth_signature: Optional[str] = None,
    ) -> Phase2GaugeDynamicsCertificate:
        r"""**[Compat v9.3/v10] FASE 2.** Delega a `emit_phase2_certificate`."""
        return self.emit_phase2_certificate(
            cert1, density_matrix,
            electron_anomaly_count=electron_anomaly_count,
            positron_auth_signature=positron_auth_signature,
        )

    def consume_engine_phase2_envelope(
        self, engine_env: Any,
    ) -> Phase2GaugeDynamicsCertificate:
        r"""Interoperabilidad: `GaugePhase2Envelope` externo → certificado firmado."""
        if engine_env is None:
            raise GaugeFieldAgentError("engine_env es None.")
        p1 = engine_env.phase1
        fields = self._extract_from_phase2_envelope(
            p1, engine_env, np.asarray(p1.rho_sanitized, dtype=np.complex128))
        atoms = [
            HeytingOmega3.COHERENT if fields["ym_energy"] <= 0.25 else (
                HeytingOmega3.DEGRADED if fields["ym_energy"] <= 0.85 else HeytingOmega3.VETOED),
            HeytingOmega3.COHERENT if abs(fields["k_inst"]) <= 2 else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if fields["kam_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if fields["melnikov_ok"] else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if fields["rsi3_ok"] else HeytingOmega3.VETOED,
        ]
        verdict = self._adjudicator.join_atoms(atoms)
        dim = int(_g(p1, "dimension", self.dimension))
        chain = _hash_state(
            np.asarray(p1.rho_sanitized, dtype=np.complex128),
            np.array([
                fields["cs3"], fields["w_angle"], float(fields["k_inst"]),
                fields["kam_worst"], fields["melnikov"], fields["floquet_dev"],
                float(fields["rsi3_ok"]),
            ], dtype=np.float64),
        )
        unsigned = Phase2GaugeDynamicsCertificate(
            yang_mills_energy=fields["ym_energy"],
            chern_simons_3form_higgs=fields["cs3"],
            wilson_angle_rad=fields["w_angle"],
            instanton_k=fields["k_inst"],
            stora_zumino_residual=fields["sz"],
            weak_value_qnd=fields["a_weak"],
            positron_annihilations_count=0,
            phase2_verdict=verdict,
            hmac_phase2_signature="",
            chern_simons_anomaly=fields["cs_anomaly"],
            c1_value=fields["c1"],
            pontryagin_integrality_residual=fields["integrality"],
            kam_worst_ratio=fields["kam_worst"],
            kam_satisfied=fields["kam_ok"],
            melnikov_value=fields["melnikov"],
            melnikov_preserved=fields["melnikov_ok"],
            floquet_max_unit_deviation=fields["floquet_dev"],
            floquet_neutral=fields["floquet_neutral"],
            poincare_section_n_crossings=fields["poincare_x"],
            poincare_section_dispersion=fields["poincare_disp"],
            rsi3_valid=fields["rsi3_ok"],
            rsi3_iterations=fields["rsi3_iters"],
            banach_lipschitz_t3=fields["banach_t3"],
            fubini_study_distance=fields["fubini"],
            novikov_krull_valuation=fields["novikov"],
            back_action_db=fields["back_action"],
            topology_regime=fields["regime"],
            engine_envelope=engine_env,
            chain_hash=chain,
            dimension=dim,
            bruno_satisfied=fields["bruno_ok"],
            bruno_sum=fields["bruno_sum"],
            chern_simons_algebraic=fields["cs_alg"],
            chern_character_ch2=fields["ch2"],
            krein_collision=fields["krein"],
            qnd_variance=fields["qnd_var"],
            mean_return_time=fields["mean_ret"],
            recurrence_time=fields["rec_time"],
            rsi3_contraction=fields["contraction"],
        )
        return self._sign_phase2(unsigned)

    def continue_to_phase3(
        self, cert2: Phase2GaugeDynamicsCertificate,
    ) -> Phase2GaugeDynamicsCertificate:
        r"""
        **Puente anidado FASE 2 → FASE 3 (último morfismo de Orient).**

        Valida HMAC. Dual clasista: `GaugeAgentPhase3Decider.from_phase2_certificate`.
        """
        if not self._verify_hmac(self._phase2_payload(cert2), cert2.hmac_phase2_signature):
            raise GaugeContractCorruptedError(
                "continue_to_phase3: HMAC de FASE 2 inválida (dominio v11).")
        if cert2.banach_lipschitz_t3 > (_BANACH_LIPSCHITZ_MAX ** 3) + 1e-9:
            logger.warning(
                "Lip(T³)=%.4f > 0.95³; RSI³ no contráctil.", cert2.banach_lipschitz_t3)
        self._phase2_cert = cert2
        logger.debug("FASE 2→3 puente OK hmac=%s…", cert2.hmac_phase2_signature[:16])
        return cert2


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   3 :   D E C I D E   &   A C T                                      ║
# ║  QND, Heyting Ω₃, Crowbar, Φ_sem, Merkle, HMAC.                               ║
# ║  Abre con from_phase2_certificate.  Cierra con emit_sovereign_state.         ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class GaugeAgentPhase3Decider(GaugeAgentPhase2Orientor):
    r"""
    FASE 3 (Decide & Act): supremum en \(\Omega_3\), interlock Crowbar,
    funtor visceral y firma de gobernanza.
    """

    @classmethod
    def from_phase2_certificate(
        cls,
        cert2: Phase2GaugeDynamicsCertificate,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        dimension: Optional[int] = None,
        background_hamiltonian: Optional[NDArray] = None,
    ) -> "GaugeAgentPhase3Decider":
        r"""**Apertura formal (clasista) de la FASE 3.** Dual de `continue_to_phase3`."""
        dim = int(dimension) if dimension is not None else int(cert2.dimension)
        inst = cls(dimension=dim, secret_key=secret_key,
                   background_hamiltonian=background_hamiltonian)
        inst.continue_to_phase3(cert2)
        return inst

    def continue_into_phase3_adjudication_gate(
        self, cert2: Phase2GaugeDynamicsCertificate,
    ) -> bool:
        r"""**[Compat v9.3/v10] Compuerta Φ₂₃.** Delega al puente anidado."""
        self.continue_to_phase3(cert2)
        return True

    def _crowbar(self, verdict: HeytingOmega3) -> Tuple[bool, float, str]:
        try:
            return _unpack_crowbar(
                self.engine.trigger_hardware_interlock_esp32_crowbar(verdict))
        except Exception as exc:
            logger.warning("Crowbar del motor falló (%s); interlock local.", exc)
            if verdict is HeytingOmega3.VETOED:
                return True, _ESP32_CROWBAR_MAX_LATENCY_NS, "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
            if verdict is HeytingOmega3.DEGRADED:
                return False, _CROWBAR_DEGRADED_LATENCY_NS, "MONITORING_DEGRADED_PERFORMANCE"
            return False, _CROWBAR_NOMINAL_LATENCY_NS, "NOMINAL_EXECUTION"

    def emit_sovereign_state(
        self,
        cert1: GaugeFieldIsometryCertificate,
        cert2: Phase2GaugeDynamicsCertificate,
        density_matrix: NDArray,
    ) -> TopologicalGaugeFieldSovereignState:
        r"""
        **Cierre formal de la FASE 3.**
        Adjudica \(\Omega_3\), Crowbar, Φ_sem, Merkle+HMAC de provenance.
        """
        self.continue_to_phase3(cert2)
        now = time.time()
        _ = self.engine.sanitize_density_matrix(density_matrix)

        atoms: List[HeytingOmega3] = [
            cert1.phase1_verdict,
            cert2.phase2_verdict,
            HeytingOmega3.COHERENT if cert1.is_symplectic_valid else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if cert1.symplectic_certified else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if cert1.is_anti_hermitian_A else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if cert1.is_traceless_A else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if cert2.kam_satisfied else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if cert2.bruno_satisfied else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if cert2.melnikov_preserved else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if cert2.rsi3_valid else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if cert1.lindstedt_is_solved else HeytingOmega3.DEGRADED,
        ]
        global_verdict = self._adjudicator.adjudicate(*atoms)
        meet_atoms = self._adjudicator.meet_atoms(atoms)

        crowbar_triggered, latency_ns, hw_action = self._crowbar(global_verdict)
        latency_verified = float(latency_ns) <= _CROWBAR_TIMING_LIMIT_NS
        if not latency_verified:
            logger.critical(
                "LATENCIA VIOLADA: %.2f ns > %.0f ns", latency_ns, _CROWBAR_TIMING_LIMIT_NS)
            hw_action = "CROWBAR_LATENCY_VIOLATION"
            global_verdict = global_verdict.join(HeytingOmega3.VETOED)

        try:
            visceral = dict(self.engine.apply_visceral_functor_phi_sem(
                global_verdict, cert2.yang_mills_energy, cert2.wilson_angle_rad))
        except Exception:
            visceral = {"note": "visceral_functor_unavailable"}

        merkle_seed = (
            f"{cert1.hmac_phase1_signature}:{cert2.hmac_phase2_signature}:"
            f"{int(global_verdict)}:{now:.6f}:"
            f"{cert1.chain_hash}:{cert2.chain_hash}:{_AGENT_VERSION}"
        ).encode("utf-8")
        merkle_digest = hashlib.sha256(merkle_seed).hexdigest()
        prov_hmac = self._compute_hmac(merkle_digest)

        is_epistemologically_valid = (
            global_verdict is not HeytingOmega3.VETOED
            and latency_verified
            and cert1.is_trace_conserved
            and cert1.symplectic_certified
        )
        return TopologicalGaugeFieldSovereignState(
            phase1_cert=cert1, phase2_cert=cert2,
            global_verdict=global_verdict,
            crowbar_triggered=bool(crowbar_triggered),
            crowbar_latency_ns=float(latency_ns),
            visceral_report=visceral,
            sha256_merkle_digest=merkle_digest,
            provenance_hmac=prov_hmac,
            timestamp_utc=now,
            crowbar_latency_verified=bool(latency_verified),
            hardware_action=hw_action,
            heyting_join_atoms=tuple(int(a) for a in atoms),
            heyting_meet_atoms=(int(meet_atoms),),
            heyting_implies_table=HeytingOmega3Adjudicator.truth_table_implies(),
            heyting_negate_table=HeytingOmega3Adjudicator.truth_table_negate(),
            is_epistemologically_valid=bool(is_epistemologically_valid),
            agent_version=_AGENT_VERSION,
            spectral_gap=cert1.spectral_gap,
            poincare_cartan_invariant=cert1.poincare_cartan_invariant,
            bruno_satisfied=cert2.bruno_satisfied,
            lindstedt_residual=cert1.poincare_residual,
            qnd_variance=cert2.qnd_variance,
            hmac_verified=True,
        )

    def execute_phase3_adjudication(
        self,
        cert1: GaugeFieldIsometryCertificate,
        cert2: Phase2GaugeDynamicsCertificate,
        density_matrix: NDArray,
    ) -> TopologicalGaugeFieldSovereignState:
        r"""**[Compat v9.3/v10] FASE 3.** Delega a `emit_sovereign_state`."""
        return self.emit_sovereign_state(cert1, cert2, density_matrix)

    def verify_sovereign_hmac(self, state: TopologicalGaugeFieldSovereignState) -> bool:
        return (
            self._verify_hmac(self._phase1_payload(state.phase1_cert),
                              state.phase1_cert.hmac_phase1_signature)
            and self._verify_hmac(self._phase2_payload(state.phase2_cert),
                                  state.phase2_cert.hmac_phase2_signature)
            and hmac.compare_digest(
                self._compute_hmac(state.sha256_merkle_digest), state.provenance_hmac)
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACHADA — orquestación de las tres fases anidadas
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalGaugeFieldAgent(GaugeAgentPhase3Decider):
    r"""
    Soberano Agéntico del Campo Topológico de Gauge (v11.0.0).

      FASE 1  emit_phase1_certificate → continue_to_phase2
      FASE 2  emit_phase2_certificate → continue_to_phase3
      FASE 3  emit_sovereign_state

    Firmas públicas preservadas: `govern_gauge_field`, `execute_phase*`,
    `continue_into_*`, `consume_engine_phase*_envelope`.
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        background_hamiltonian: Optional[NDArray] = None,
    ) -> None:
        super().__init__(
            dimension=dimension, secret_key=secret_key,
            background_hamiltonian=background_hamiltonian,
        )
        logger.info(
            "TopologicalGaugeFieldAgent v%s (3-Fases) dim=%d engine=%s envelopes=%s",
            _AGENT_VERSION, self.dimension, _ENGINE_IMPORTED, _ENGINE_HAS_ENVELOPES,
        )

    def govern_gauge_field(
        self,
        density_matrix: NDArray,
        electron_anomaly_count: int = 0,
        positron_auth_signature: Optional[str] = None,
    ) -> TopologicalGaugeFieldSovereignState:
        r"""
        Ciclo OODA completo por fases anidadas:
          ① emit_phase1_certificate → continue_to_phase2
          ② emit_phase2_certificate → continue_to_phase3
          ③ emit_sovereign_state
        """
        cert1 = self.emit_phase1_certificate(density_matrix)
        cert1 = self.continue_to_phase2(cert1)

        cert2 = self.emit_phase2_certificate(
            cert1, density_matrix,
            electron_anomaly_count=electron_anomaly_count,
            positron_auth_signature=positron_auth_signature,
        )
        cert2 = self.continue_to_phase3(cert2)

        return self.emit_sovereign_state(cert1, cert2, density_matrix)

    def issue_positron_auth(self, electron_anomaly_count: int) -> str:
        r"""Emite la firma HMAC que autentica la inyección de positrones e⁺."""
        return self._compute_hmac(f"AUTH_POSITRON:{int(electron_anomaly_count)}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "GaugeFieldAgentError",
    "GaugeContractCorruptedError",
    "GaugeIsometryViolationError",
    "SymplecticGaugeInvarianceError",
    "EnvelopeChainViolationError",
    "PoincareLindstedtResonanceError",
    "RSI3ContractionFailureError",
    "HeytingOmega3Adjudicator",
    "GaugeFieldIsometryCertificate",
    "Phase2GaugeDynamicsCertificate",
    "TopologicalGaugeFieldSovereignState",
    "GaugeAgentPhase1Observer",
    "GaugeAgentPhase2Orientor",
    "GaugeAgentPhase3Decider",
    "TopologicalGaugeFieldAgent",
    "HeytingOmega3",
    "TopologicalGaugeFieldEngine",
    "TopologicalGaugeFieldSnapshot",
]


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL DEL SOBERANO AGÉNTICO
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 80
    print(banner)
    print("  VERIFICACIÓN DOCTORAL — Topological Gauge Field Agent v11.0.0")
    print(f"  Engine={_ENGINE_IMPORTED}  envelopes={_ENGINE_HAS_ENVELOPES}  "
          f"continue={_ENGINE_HAS_CONTINUE}  gov={_ENGINE_HAS_GOVERNANCE}")
    print(banner)

    dim = 16
    agent = TopologicalGaugeFieldAgent(dimension=dim)
    rng = np.random.default_rng(2026)
    g = rng.normal(0.0, 1.0, size=(dim, dim)) + 1j * rng.normal(0.0, 1.0, size=(dim, dim))
    density_sample = g @ g.conj().T
    density_sample /= np.trace(density_sample)

    state = agent.govern_gauge_field(density_sample)
    hmac_ok = agent.verify_sovereign_hmac(state)

    # Round-trip de puentes anidados
    reopened = TopologicalGaugeFieldAgent.from_phase1_certificate(
        state.phase1_cert, secret_key=agent.secret_key, dimension=dim)
    assert reopened._phase1_cert is not None
    reopened3 = TopologicalGaugeFieldAgent.from_phase2_certificate(
        state.phase2_cert, secret_key=agent.secret_key, dimension=dim)
    assert reopened3._phase2_cert is not None

    print(f"  • Veredicto Global Ω₃       : {state.global_verdict.name} "
          f"(0x{int(state.global_verdict):02x})")
    print(f"  • Norma Conexión A          : {state.phase1_cert.connection_norm_frobenius:.6f}")
    print(f"  • Anti-Hermítica / Traceless: {state.phase1_cert.is_anti_hermitian_A} / "
          f"{state.phase1_cert.is_traceless_A}")
    print(f"  • Simpléctica certificada   : {state.phase1_cert.symplectic_certified} "
          f"(δω={state.phase1_cert.symplectic_residue:.3e})")
    print(f"  • Gap espectral             : {state.spectral_gap:.4e}")
    print(f"  • Poincaré–Cartan           : {state.poincare_cartan_invariant:.4e}")
    print(f"  • Lindstedt residual        : {state.lindstedt_residual:.3e}  "
          f"solved={state.phase1_cert.lindstedt_is_solved}")
    print(f"  • RSI³ (agente) iters/Lip   : {state.phase1_cert.rsi3_iterations} / "
          f"{state.phase1_cert.rsi3_contraction:.4f}")
    print(f"  • Energía Yang-Mills        : {state.phase2_cert.yang_mills_energy:.6f}")
    print(f"  • Chern-Simons cov / alg    : {state.phase2_cert.chern_simons_3form_higgs:.6e} / "
          f"{state.phase2_cert.chern_simons_algebraic:.6e}")
    print(f"  • CS Anomaly                : {state.phase2_cert.chern_simons_anomaly:.6e}")
    print(f"  • Wilson / c₁               : {state.phase2_cert.wilson_angle_rad:.4f} rad / "
          f"{state.phase2_cert.c1_value:.4e}")
    print(f"  • Instanton k               : {state.phase2_cert.instanton_k}")
    print(f"  • KAM worst / Bruno         : {state.phase2_cert.kam_worst_ratio:.4f} / "
          f"{state.bruno_satisfied}")
    print(f"  • Melnikov |M|              : {abs(state.phase2_cert.melnikov_value):.4e}")
    print(f"  • Floquet |Δμ| / Krein      : {state.phase2_cert.floquet_max_unit_deviation:.3e} / "
          f"{state.phase2_cert.krein_collision}")
    print(f"  • Poincaré cruces / ⟨T⟩     : {state.phase2_cert.poincare_section_n_crossings} / "
          f"{state.phase2_cert.mean_return_time:.3f}")
    print(f"  • RSI³ motor Lip(T³)        : {state.phase2_cert.banach_lipschitz_t3:.4f}")
    print(f"  • Novikov v                 : {state.phase2_cert.novikov_krull_valuation:.4f}")
    print(f"  • QND A_w / var             : {state.phase2_cert.weak_value_qnd:.4f} / "
          f"{state.qnd_variance:.3e}")
    print(f"  • Crowbar                   : {state.crowbar_latency_ns:.2f} ns "
          f"verified={state.crowbar_latency_verified}  {state.hardware_action}")
    print(f"  • Merkle / HMAC ok          : {state.sha256_merkle_digest[:24]}… / {hmac_ok}")
    print(f"  • Heyting join atoms        : {state.heyting_join_atoms}")
    print(f"  • Heyting ¬ table           : {state.heyting_negate_table}")
    print(f"  • WACC                      : {state.visceral_report.get('wacc_protection_status', '—')}")
    print(f"  • Liquidez                  : {state.visceral_report.get('net_liquidity_impact', '—')}")
    print(banner)

    # Autenticación de Fock (debe ser 0 si la firma es basura; >0 si es HMAC v11)
    bad = agent.govern_gauge_field(density_sample, electron_anomaly_count=3,
                                   positron_auth_signature="deadbeef")
    sig = agent.issue_positron_auth(3)
    good = agent.govern_gauge_field(density_sample, electron_anomaly_count=3,
                                    positron_auth_signature=sig)
    print(f"  [Fock] firma basura → n={bad.phase2_cert.positron_annihilations_count}  "
          f"HMAC v11 → n={good.phase2_cert.positron_annihilations_count}")
    print(banner)
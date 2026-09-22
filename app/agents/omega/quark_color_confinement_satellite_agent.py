from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Quark Color Confinement Satellite Agent (Soberano del Satélite VI)          ║
║ RUTA   : app/agents/omega/quark_color_confinement_satellite_agent.py                 ║
║ VERSIÓN: 2.0.0-Doctoral-SU3-Berry-ANO-HeytingTopos-Fock-Crowbar-IRAM                 ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Agente supervisor ciber-físico en el Estrato Omega ($V_\Omega$). Gobierna en lazo cerrado al Motor
Satelital de Confinamiento de Color de Quarks en el Cinturón Orbital de Frontera.

Fundamentación Matemática, Categórica y Ciber-Física Rigurosa:
──────────────────────────────────────────────────────────────
1. Geometría de Fibrados Principales y Fase de Berry-Pancharatnam en $\mathbb{CP}^2$:
   Auditoría del invariante de Bargmann sobre el triángulo de Cartan $(T_3, T_8)$ en $\mathbb{CP}^2$:
   $$\Delta(\psi_0, \psi_1, \psi_2) = \langle\psi_0|\psi_1\rangle \langle\psi_1|\psi_2\rangle \langle\psi_2|\psi_0\rangle \implies \gamma_P = \arg \Delta$$
   complementado con el peso baricéntrico de Wilson $W_{\mathrm{bary}} = 3(\prod_i p_i)^{1/3}$ y la razón de participación $P_{\mathrm{part}} = (\sum_i p_i^2)^{-1}$.

2. Guía de Onda Telegrafista Dual ANO / 't Hooft–Mandelstam:
   Propagación de señal en el tubo de flujo cromodinámico:
   $$\gamma = \alpha + i\beta = \sqrt{(R' + i\omega L')(G' + i\omega C')}, \qquad Z_c = \sqrt{\frac{R' + i\omega L'}{G' + i\omega C'}}$$

3. Clasificador de Subobjetos en el Topos de Heyting $\Omega_3 = \{\bot, \mathfrak{m}, \top\}$:
   $$\bot = \mathtt{VETOED} (0), \quad \mathfrak{m} = \mathtt{DEGRADED} (1), \quad \top = \mathtt{COHERENT} (2)$$
   con combinación categórica por ínfimo reticular $a \wedge b = \min(a,b)$ entre el clasificador local y el motor.

4. Aniquilación de Fock ($e^- + e^+ \to 2\gamma$) y Crowbar BT151 (IRAM < 400 ns):
   - Inyección de Positrón validado vía HMAC-SHA512 con nonces SHA3-512 anti-replay.
   - Enclavamiento físico en silicio ESP32 (GPIO14) mediante ISR IRAM en latencia $\tau < 400\,\mathrm{ns}$ e integral térmica $I^2t$.

Traducción Ejecutiva e Impacto de Negocio ('Dolor y Dinero'):
─────────────────────────────────────────────────────────────
• Dolor: La deconfinación de registros atómicos o corrupción de estados de color en la base de datos acarrea
  inconsistencias críticas en transacciones distribuidas y quiebre de auditoría.
• Dinero: La censura inmediata del Satélite VI e interrupción en silicio (< 400 ns) impiden la propagación de
  datos deconfinados, blindando la integridad relacional y protegiendo el valor de los activos de la empresa.

Estructura Functorial OODA:
───────────────────────────
- Observe  : `observe_and_bundle_agent_kernel`  -> Salida: `QuarkAgentObservationKernel`
- Orient   : `orient_agent_confinement`          -> Salida: `QuarkAgentOrientationDocket`
- Act      : `decide_and_actuate_governance`    -> Salida: `QuarkAgentCertificate`
- Composición Síncrona Lazo Cerrado             : `audit_quark_confinement_cycle`
"""

import enum
import hashlib
import hmac
import logging
import math
import threading
import time
from dataclasses import dataclass, fields
from typing import Dict, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# RESOLUCIÓN RESILIENTE DE DEPENDENCIAS DEL SISTEMA INMUNE OMEGA
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class Morphism:
        """Morfismo base en la categoría de espacios fibrados."""

    class TopologicalInvariantError(Exception):
        """Violación crítica de invariancia topológica o clases características."""


try:
    from quark_color_confinement_satellite_engine import (
        QuarkColorConfinementSatelliteEngine,
        QuarkEngineState,
        QuarkObservationKernel,
        QuarkConfinementReport,
        BaseMetricCache,
        HeytingTruthValue,
        QuarkEngineError,
        MetricIndefinitenessError,
        QuarkDimensionError,
        GaugeAlgebraViolationError,
        FreeQuarkGaugeViolationError,
    )
except ImportError:
    try:
        from app.core.immune_system.quark_color_confinement_satellite_engine import (
            QuarkColorConfinementSatelliteEngine,
            QuarkEngineState,
            QuarkObservationKernel,
            QuarkConfinementReport,
            BaseMetricCache,
            HeytingTruthValue,
            QuarkEngineError,
            MetricIndefinitenessError,
            QuarkDimensionError,
            GaugeAlgebraViolationError,
            FreeQuarkGaugeViolationError,
        )
    except ImportError as exc:
        raise ImportError(
            "CRITICAL: No se pudo enlazar el motor cuántico subyacente "
            "'quark_color_confinement_satellite_engine.py'. "
            "Asegure su disponibilidad en PYTHONPATH."
        ) from exc

logger = logging.getLogger("APU.Agents.Omega.QuarkColorConfinementSatelliteAgent")

__version__: Final[str] = "2.0.0"
__all__ = (
    "QuarkColorConfinementSatelliteAgent",
    "Phase1_QuarkAgentObserver",
    "Phase2_QuarkAgentOrienter",
    "Phase3_QuarkAgentDecider",
    "QuarkAgentObservationKernel",
    "QuarkAgentOrientationDocket",
    "QuarkAgentCertificate",
    "QuarkHeytingVerdict",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, METROLÓGICAS Y DE SILICIO (IEEE-754 / HARDWARE ESP32)
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY_FLOOR: Final[float] = 1.0e-15
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12
_CASIMIR_SINGLET_TOLERANCE: Final[float] = 1.0e-9
_N_COLOR: Final[int] = 3

_STRING_BREAKAGE_THRESHOLD_ENERGY: Final[float] = 100.0
_ELASTIC_STRING_FLOOR_GEV: Final[float] = 30.0
_SCHWINGER_SOFT_THRESHOLD: Final[float] = 1.0e-3
_ANO_FLUX_QUANTUM_QCD: Final[float] = 2.0 * math.pi
_Z_VAC_SI_OHM: Final[float] = 376.730313461770

_CROWBAR_IRAM_LATENCY_MAX_NS: Final[float] = 400.0
_BT151_GATE_TRIGGER_CURRENT_MA: Final[float] = 15.0
_BT151_LATCHING_CURRENT_MA: Final[float] = 40.0
_BT151_ITSM_A: Final[float] = 120.0
_BT151_I2T_RATING: Final[float] = 72.0
_BT151_SURGE_WINDOW_S: Final[float] = 10.0e-3
_ESP32_GPIO14_PROPAGATION_DELAY_NS: Final[float] = 12.5
_ESP32_IRAM_WRITE_NS: Final[float] = 25.0
_BT151_AVALANCHE_NS: Final[float] = 350.0

_HMAC_TIMESTAMP_WINDOW_S: Final[float] = 300.0
_MAX_CONSUMED_NONCES: Final[int] = 4096
_POSITRON_DOES_NOT_PROVE_TOP: Final[bool] = True


# ══════════════════════════════════════════════════════════════════════════════
# ARITMÉTICA DE PUNTO FLOTANTE EXTENDIDA (KAHAN–BABUŠKA–NEUMAIER)
# ══════════════════════════════════════════════════════════════════════════════
class KahanNeumaierAccumulator:
    r"""
    Sumación compensada de Neumaier.

    Reduce el error secular de redondeo de \(\mathcal{O}(N\epsilon)\) a
    \(\mathcal{O}(\epsilon)\) frente a la absorción en la FPU IEEE-754.
    """

    __slots__ = ("_sum", "_compensation")

    def __init__(self) -> None:
        self._sum: float = 0.0
        self._compensation: float = 0.0

    def add(self, x: float) -> None:
        t: float = self._sum + x
        if math.fabs(self._sum) >= math.fabs(x):
            self._compensation += (self._sum - t) + x
        else:
            self._compensation += (x - t) + self._sum
        self._sum = t

    def add_iterable(self, arr: NDArray[np.float64]) -> None:
        for val in arr.flat:
            self.add(float(val))

    @property
    def total(self) -> float:
        return float(self._sum + self._compensation)


def _cabs(z: complex) -> float:
    r"""Módulo complejo estable vía `hypot` (evita overflow de \(|z|^2\))."""
    return math.hypot(float(z.real), float(z.imag))


def _kahan_sum_sq_complex(vec: NDArray[np.complex128]) -> float:
    r"""\(\sum_i |z_i|^2\) con compensación Neumaier."""
    acc = KahanNeumaierAccumulator()
    for z in vec.flat:
        zr = float(np.real(z))
        zi = float(np.imag(z))
        acc.add(zr * zr + zi * zi)
    return acc.total


def _const_eq_str(a: str, b: str) -> bool:
    """Igualdad a tiempo constante sobre digestos SHA-256 (longitud fija)."""
    return hmac.compare_digest(
        hashlib.sha256(a.encode("utf-8")).digest(),
        hashlib.sha256(b.encode("utf-8")).digest(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLASIFICADOR DE SUBOBJETOS: ÁLGEBRA DE HEYTING TRIVALENTE (TOPOS)
# ══════════════════════════════════════════════════════════════════════════════
class QuarkHeytingVerdict(enum.IntEnum):
    r"""
    Elementos del clasificador de subobjetos \(\Omega_3\) (cadena de Gödel):

        \(\bot \;\le\; \mathfrak{m} \;\le\; \top\)

    - VETOED (\(\bot\)): desconfinamiento, quark libre o ruptura de cuerda.
    - DEGRADED (\(\mathfrak{m}\)): tubo de flujo elástico metastable.
    - COHERENT (\(\top\)): estado invariante de calibre (\(\rho \approx I/3\)).

    El orden es lineal, luego el álgebra de Heyting es

        \(a \wedge b = \min(a,b),\; a \vee b = \max(a,b),\;
          a \Rightarrow b = \top\) si \(a\le b\), si no \(b\).
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @classmethod
    def meet(cls, a: "QuarkHeytingVerdict", b: "QuarkHeytingVerdict") -> "QuarkHeytingVerdict":
        r"""Ínfimo (conjunción intuicionista): \(a \wedge b = \min(a,b)\)."""
        return cls(min(int(a.value), int(b.value)))

    @classmethod
    def join(cls, a: "QuarkHeytingVerdict", b: "QuarkHeytingVerdict") -> "QuarkHeytingVerdict":
        r"""Supremo (disyunción intuicionista): \(a \vee b = \max(a,b)\)."""
        return cls(max(int(a.value), int(b.value)))

    @classmethod
    def implies(cls, a: "QuarkHeytingVerdict", b: "QuarkHeytingVerdict") -> "QuarkHeytingVerdict":
        r"""Implicación de Heyting: \(a \Rightarrow b = \bigvee\{c \mid a\wedge c \le b\}\)."""
        if int(a.value) <= int(b.value):
            return cls.COHERENT
        return b

    @classmethod
    def pseudocomplement(cls, a: "QuarkHeytingVerdict") -> "QuarkHeytingVerdict":
        r"""Negación intuicionista: \(\neg a = (a \Rightarrow \bot)\). \(\neg\neg\mathfrak{m}\ne\mathfrak{m}\)."""
        return cls.implies(a, cls.VETOED)

    @classmethod
    def from_engine_topos(cls, truth_value: object) -> "QuarkHeytingVerdict":
        r"""Morfismo de clasificadores \(\Omega_{\mathrm{engine}} \to \Omega_3\)."""
        raw = str(getattr(truth_value, "value", truth_value))
        mapping = {
            "TOP_CONFINED": cls.COHERENT,
            "METASTABLE_MESON": cls.DEGRADED,
            "BOTTOM_DECONFINED": cls.VETOED,
        }
        return mapping.get(raw, cls.VETOED)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL PIPELINE TRI-FÁSICO ANIDADO
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class QuarkAgentObservationKernel:
    r"""
    EXPEDIENTE TERMINAL DE LA FASE 1 (OBSERVE).
    Objeto inicial imprescindible de la Fase 2.

    Geometría del fibrado, fase de Berry–Pancharatnam en \(\mathbb{CP}^2\),
    Wilson baricéntrico, normas de Banach, fidelidad de Bures a \(I/3\)
    y auditoría Riemann–Cholesky.
    """

    color_triplet_vector: NDArray[np.complex128]
    normalized_state_cp2: NDArray[np.complex128]
    color_density_matrix: NDArray[np.complex128]
    banach_norm_l1: float
    banach_norm_l2: float
    banach_ratio: float
    banach_ratio_residual: float
    berry_pancharatnam_phase_rad: float
    berry_bargmann_modulus: float
    wilson_holonomy_trace: float
    wilson_barycentric_weight: float
    color_participation_ratio: float
    fubini_study_radius_to_center: float
    bures_distance_to_singlet: float
    metric_cache: BaseMetricCache
    phase1_sha256_seal: str


@dataclass(frozen=True, slots=True)
class QuarkAgentOrientationDocket:
    r"""
    EXPEDIENTE TERMINAL DE LA FASE 2 (ORIENT).
    Objeto inicial imprescindible de la Fase 3.

    Continúa del `QuarkAgentObservationKernel`. Encapsula la auditoría ciega
    del motor FPU, Casimir, Cornell–Lüscher, telegrafista ANO y Meissner dual.
    """

    observation_kernel: QuarkAgentObservationKernel
    engine_state: QuarkEngineState
    gell_mann_coherence: NDArray[np.float64]
    casimir_c2_value: float
    casimir_operator_expectation: float
    color_neutrality_deviation: float
    cartan_t3: float
    cartan_t8: float
    von_neumann_entropy: float
    is_color_singlet: bool
    cornell_potential_val: float
    string_tension_energy: float
    luscher_correction_energy: float
    running_alpha_s: float
    is_string_broken: bool
    ano_characteristic_impedance_natural: float
    ano_characteristic_impedance_ohms: float
    ano_telegrapher_attenuation_alpha: float
    ano_telegrapher_phase_beta: float
    ano_propagation_modulus: float
    schwinger_pair_production_rate: float
    dual_meissner_kappa: float
    wilson_loop_area_law: float
    engine_heyting: QuarkHeytingVerdict
    is_confinement_coherent: bool
    phase2_sha256_seal: str


@dataclass(frozen=True, slots=True)
class QuarkAgentCertificate:
    r"""
    CERTIFICADO TERMINAL DE GOBERNANZA — FASE 3 (ACT).

    Continúa del `QuarkAgentOrientationDocket`. Sella la clasificación en
    \(\Omega_3\), la aniquilación de Fock, la telemetría Crowbar BT151 y
    la cadena criptográfica SHA-256 → BLAKE2b-512.
    """

    governance_phase: str
    heyting_verdict: QuarkHeytingVerdict
    heyting_negation: str
    heyting_implication_to_top: str
    engine_topos_name: str
    casimir_c2_value: float
    is_color_singlet: bool
    cornell_potential_val: float
    string_tension_energy: float
    berry_pancharatnam_phase_rad: float
    wilson_barycentric_weight: float
    color_participation_ratio: float
    ano_telegrapher_attenuation_alpha: float
    is_free_quark_detected: bool
    is_string_breakage_critical: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    positron_annihilation_verified: bool
    positron_proves_top: bool
    hardware_interlock_fired: bool
    iram_budget_ok: bool
    actuation_latency_ns: float
    silicon_thermal_joule_integral: float
    i2t_rating_utilization: float
    time_grace_remaining_seconds: float
    phase1_sha256_seal: str
    phase2_sha256_seal: str
    engine_cryptographic_seal: str
    digital_signature_blake2b: str


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: OBSERVACIÓN, FIBRADOS PRINCIPALES, BERRY–PANCHARATNAM Y BANACH
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_QuarkAgentObserver:
    r"""
    FASE 1 — OBSERVE.

    Ingesta del triplete de color, saneamiento IEEE-754, holonomía geométrica
    de Berry–Pancharatnam en \(\mathbb{CP}^2\), equivalencia de normas de Banach
    y certificación métrica SPD.

    El método terminal `observe_and_bundle_agent_kernel` produce el objeto
    inicial de la Fase 2: `QuarkAgentObservationKernel`.
    """

    def __init__(self, tolerance: float = 1.0e-12) -> None:
        self._tol: Final[float] = float(tolerance)

    @staticmethod
    def _sanitize_ieee754_complex(arr: NDArray[np.complex128]) -> NDArray[np.complex128]:
        r"""Purga \(-0.0\), subnormales de Wilkinson y rechaza NaN/Inf en \(\mathbb{C}^n\)."""
        if not np.all(np.isfinite(arr)):
            raise QuarkEngineError(
                "El vector de estado cromodinámico contiene valores no finitos (NaN/Inf)."
            )
        real_part = np.real(arr).astype(np.float64)
        imag_part = np.imag(arr).astype(np.float64)
        real_clean = np.where(np.abs(real_part) < _WILKINSON_SAFETY_FLOOR, 0.0, real_part)
        imag_clean = np.where(np.abs(imag_part) < _WILKINSON_SAFETY_FLOOR, 0.0, imag_part)
        real_clean = np.where(real_clean == 0.0, 0.0, real_clean)
        imag_clean = np.where(imag_clean == 0.0, 0.0, imag_clean)
        return (real_clean + 1j * imag_clean).astype(np.complex128)

    def _sanitize_density_matrix(
        self,
        rho: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""Proyección de Higham al simplejo: Hermiticidad, PSD y \(\operatorname{Tr}\rho=1\)."""
        if rho.ndim != 2 or rho.shape != (3, 3):
            raise QuarkDimensionError(f"La matriz de densidad debe ser 3×3. Recibido: {rho.shape}")
        if not np.all(np.isfinite(rho)):
            raise QuarkEngineError("ρ contiene componentes no finitas.")
        herm = 0.5 * (rho + np.conj(rho.T))
        eigvals, eigvecs = la.eigh(herm)
        eigvals = np.clip(np.real(eigvals), 0.0, None)
        if float(np.sum(eigvals)) <= _WILKINSON_SAFETY_FLOOR:
            raise QuarkEngineError("ρ tiene traza nula tras proyección PSD.")
        eigvals = eigvals / float(np.sum(eigvals))
        rebuilt = (eigvecs * eigvals) @ np.conj(eigvecs.T)
        return (0.5 * (rebuilt + np.conj(rebuilt.T))).astype(np.complex128)

    def _audit_riemannian_gauge_metric(self, G: NDArray[np.float64]) -> BaseMetricCache:
        r"""
        Audita \(G_{\mu\nu}=G_{\nu\mu}\succ 0\). Calcula \(G=LL^\top\), \(G^{-1}\),
        \(\kappa(G)\), \(\sqrt{\det G}\) y el residuo de Cholesky.
        Compatible con `BaseMetricCache` v1 y v2 (filtrado por campos reales).
        """
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise QuarkDimensionError(
                f"El tensor métrico debe ser 2-covariante cuadrado. Dimensiones: {G.shape}"
            )
        d = int(G.shape[0])
        g = np.array(G, dtype=np.float64, copy=True)
        sym_diff = float(np.max(np.abs(g - g.T)))
        if sym_diff > self._tol:
            raise MetricIndefinitenessError(
                f"Violación de simetría en tensor métrico: ||G - G^T|| = {sym_diff:.3e}"
            )
        g = 0.5 * (g + g.T)
        try:
            l_cholesky = la.cholesky(g, lower=True)
        except la.LinAlgError as exc:
            raise MetricIndefinitenessError(
                "El tensor métrico viola la positividad estricta (Cholesky falló)."
            ) from exc

        eigvals = la.eigvalsh(g)
        min_ev = float(np.min(eigvals))
        max_ev = float(np.max(eigvals))
        if min_ev <= _WILKINSON_SAFETY_FLOOR:
            raise MetricIndefinitenessError(
                f"Autovalor nulo o negativo en tensor métrico: {min_ev:.3e}"
            )
        cond_num = float(max_ev / min_ev)
        if cond_num > _CONDITION_NUMBER_MAX:
            raise MetricIndefinitenessError(
                f"Tensor métrico mal condicionado: kappa(G) = {cond_num:.2e}"
            )

        i_d = np.eye(d, dtype=np.float64)
        l_inv = la.solve_triangular(l_cholesky, i_d, lower=True)
        g_inv = l_inv.T @ l_inv
        chol_residual = float(np.max(np.abs(l_cholesky @ l_cholesky.T - g)))
        log_det = 2.0 * float(np.sum(np.log(np.diag(l_cholesky))))
        volume_density = float(math.exp(0.5 * log_det))

        payload = {
            "g_base": g,
            "cholesky_factor": l_cholesky,
            "g_inv": g_inv,
            "christoffel_symbols": np.zeros((d, d, d), dtype=np.float64),
            "ricci_scalar": 0.0,
            "condition_number": cond_num,
            "dimension": d,
            "volume_density": volume_density,
            "min_eigenvalue": min_ev,
            "max_eigenvalue": max_ev,
            "cholesky_residual": chol_residual,
        }
        valid = {f.name for f in fields(BaseMetricCache)}
        return BaseMetricCache(**{k: v for k, v in payload.items() if k in valid})

    @staticmethod
    def _cartan_rotation(psi: NDArray[np.complex128], axis: str, theta: float) -> NDArray[np.complex128]:
        r"""
        Acción del toro de Cartan \(\exp(-i\theta T_a)\) sobre \(\mathbb{C}^3\):

            \(T_3=\operatorname{diag}(\tfrac12,-\tfrac12,0),\quad
              T_8=\operatorname{diag}(1,1,-2)/(2\sqrt{3})\).
        """
        if axis == "t3":
            weights = np.array([0.5, -0.5, 0.0], dtype=np.float64)
        elif axis == "t8":
            weights = np.array([1.0, 1.0, -2.0], dtype=np.float64) / (2.0 * math.sqrt(3.0))
        else:
            raise ValueError(f"Eje de Cartan desconocido: {axis}")
        phases = np.exp(-1j * float(theta) * weights)
        return (psi * phases).astype(np.complex128)

    @classmethod
    def _compute_berry_pancharatnam_phase(
        cls,
        c_norm: NDArray[np.complex128],
    ) -> Tuple[float, float]:
        r"""
        Invariante de Bargmann del triángulo de Cartan en \(\mathbb{CP}^2\):

            \(\Delta(\psi_0,\psi_1,\psi_2)=\langle\psi_0|\psi_1\rangle
              \langle\psi_1|\psi_2\rangle\langle\psi_2|\psi_0\rangle\),

        con \(\psi_1=e^{-i\frac{2\pi}{3}T_3}\psi_0\), \(\psi_2=e^{-i\frac{2\pi}{3}T_8}\psi_0\).

        La fase de Pancharatnam es \(\gamma_P=\arg\Delta\). El módulo \(|\Delta|\)
        degenera ssi el triángulo es nulo (estado en un eje de color).

        Nota: el producto \(\arg(c_1^*c_2\cdot c_2^*c_3\cdot c_3^*c_1)\) es idénticamente
        nulo (es \(\arg(|c_1|^2|c_2|^2|c_3|^2)\)); no es una fase geométrica.
        """
        if float(np.vdot(c_norm, c_norm).real) <= _WILKINSON_SAFETY_FLOOR:
            return 0.0, 0.0
        theta = 2.0 * math.pi / 3.0
        psi0 = c_norm
        psi1 = cls._cartan_rotation(c_norm, "t3", theta)
        psi2 = cls._cartan_rotation(c_norm, "t8", theta)
        invariant = np.vdot(psi0, psi1) * np.vdot(psi1, psi2) * np.vdot(psi2, psi0)
        modulus = _cabs(complex(invariant))
        if modulus < _WILKINSON_SAFETY_FLOOR:
            return 0.0, 0.0
        return float(np.angle(invariant)), float(modulus)

    @staticmethod
    def _evaluate_wilson_invariants(
        c_norm: NDArray[np.complex128],
    ) -> Tuple[float, float, float]:
        r"""
        Tres invariantes de holonomía / ocupación de color:

        1. Traza de Cartan: \(W=\frac13\operatorname{Re}\operatorname{Tr} U\),
           \(U=\operatorname{diag}(e^{i(\arg c_k-\bar{\arg})})\in SU(3)\).
        2. Peso baricéntrico (AM-GM): \(3(\prod_i p_i)^{1/3}\in[0,1]\),
           nulo sobre los ejes (quark de color puro).
        3. Razón de participación: \((\sum_i p_i^2)^{-1}\in[1,3]\).
        """
        args = np.angle(c_norm)
        args = args - float(np.mean(args))
        u_diag = np.exp(1j * args)
        wilson_trace = float(np.real(np.sum(u_diag)) / 3.0)

        weights = np.array([_cabs(complex(z)) ** 2 for z in c_norm], dtype=np.float64)
        acc = KahanNeumaierAccumulator()
        acc.add_iterable(weights)
        total = max(acc.total, _WILKINSON_SAFETY_FLOOR)
        probs = weights / total
        geo = float(math.prod(float(max(p, 0.0)) for p in probs) ** (1.0 / 3.0))
        barycentric = float(min(max(3.0 * geo, 0.0), 1.0))
        pur = float(np.dot(probs, probs))
        participation = 1.0 / max(pur, _WILKINSON_SAFETY_FLOOR)
        return wilson_trace, barycentric, float(participation)

    @staticmethod
    def _fubini_study_and_bures(
        c_norm: NDArray[np.complex128],
        rho: NDArray[np.complex128],
    ) -> Tuple[float, float]:
        r"""
        Radio de Fubini–Study a \((1,1,1)/\sqrt{3}\) y distancia de Bures a \(I/3\):

            \(d_{\mathrm{FS}}=\arccos|\langle n|\psi\rangle|\),
            \(F(\rho,I/3)=(\operatorname{Tr}\sqrt{\rho})^2/3\),
            \(d_B=\sqrt{2(1-\sqrt{F})}\).
        """
        center = (1.0 / math.sqrt(3.0)) * np.ones(3, dtype=np.complex128)
        overlap = min(1.0, _cabs(complex(np.vdot(center, c_norm))))
        fs = float(math.acos(overlap))
        eigs = np.clip(np.real(la.eigvalsh(rho)), 0.0, None)
        tr_sqrt = float(np.sum(np.sqrt(eigs)))
        fidelity = min(1.0, (tr_sqrt * tr_sqrt) / 3.0)
        bures = math.sqrt(max(0.0, 2.0 * (1.0 - math.sqrt(max(fidelity, 0.0)))))
        return fs, float(bures)

    def observe_and_bundle_agent_kernel(
        self,
        color_triplet: NDArray[np.complex128],
        G_metric: NDArray[np.float64],
        density_matrix: Optional[NDArray[np.complex128]] = None,
    ) -> QuarkAgentObservationKernel:
        r"""
        MÉTODO TERMINAL FORMAL DE LA FASE 1.
        ────────────────────────────────────────────────────────────────────────
        Produce `QuarkAgentObservationKernel`, objeto inicial imprescindible
        de la Fase 2 (`Phase2_QuarkAgentOrienter.orient_agent_confinement`).

        Cadena anidada:
            observe_and_bundle_agent_kernel  ──Kernel──▶  orient_agent_confinement
        """
        if color_triplet.ndim != 1 or color_triplet.shape[0] != 3:
            raise QuarkDimensionError(
                f"El vector de color debe pertenecer a C^3. Recibido: {color_triplet.shape}"
            )

        c_clean = self._sanitize_ieee754_complex(np.asarray(color_triplet, dtype=np.complex128))
        metric_cache = self._audit_riemannian_gauge_metric(np.asarray(G_metric, dtype=np.float64))

        acc_l1 = KahanNeumaierAccumulator()
        for z in c_clean.flat:
            acc_l1.add(_cabs(complex(z)))
        l1_norm = acc_l1.total
        l2_norm = math.sqrt(max(_kahan_sum_sq_complex(c_clean), 0.0))

        if l2_norm > _WILKINSON_SAFETY_FLOOR:
            banach_ratio = float(l1_norm / l2_norm)
            normalized_c = (c_clean / l2_norm).astype(np.complex128)
        else:
            banach_ratio = 0.0
            normalized_c = np.zeros(3, dtype=np.complex128)
            logger.warning("Vector de color nulo: proyección CP^2 degenerada.")

        ratio_ceiling = math.sqrt(float(_N_COLOR))
        ratio_residual = float(max(0.0, banach_ratio - ratio_ceiling))
        if ratio_residual > 1.0e-7:
            logger.warning(
                "Relación de Banach fuera de cota ||c||_1/||c||_2 <= sqrt(3): %.5f",
                banach_ratio,
            )

        if density_matrix is None:
            rho = np.outer(normalized_c, np.conj(normalized_c)).astype(np.complex128)
            rho = 0.5 * (rho + np.conj(rho.T))
        else:
            rho = self._sanitize_density_matrix(np.asarray(density_matrix, dtype=np.complex128))

        berry_phase, bargmann_mod = self._compute_berry_pancharatnam_phase(normalized_c)
        wilson_tr, bary, participation = self._evaluate_wilson_invariants(normalized_c)
        fs_radius, bures = self._fubini_study_and_bures(normalized_c, rho)

        hasher = hashlib.sha256()
        hasher.update(c_clean.tobytes())
        hasher.update(rho.tobytes())
        hasher.update(metric_cache.g_base.tobytes())
        hasher.update(
            np.array(
                [banach_ratio, berry_phase, wilson_tr, bary, bures],
                dtype=np.float64,
            ).tobytes()
        )
        seal = hasher.hexdigest()

        logger.debug("Fase 1 (agente) sellada SHA-256=%s…", seal[:16])
        return QuarkAgentObservationKernel(
            color_triplet_vector=c_clean,
            normalized_state_cp2=normalized_c,
            color_density_matrix=rho,
            banach_norm_l1=l1_norm,
            banach_norm_l2=l2_norm,
            banach_ratio=banach_ratio,
            banach_ratio_residual=ratio_residual,
            berry_pancharatnam_phase_rad=berry_phase,
            berry_bargmann_modulus=bargmann_mod,
            wilson_holonomy_trace=wilson_tr,
            wilson_barycentric_weight=bary,
            color_participation_ratio=participation,
            fubini_study_radius_to_center=fs_radius,
            bures_distance_to_singlet=bures,
            metric_cache=metric_cache,
            phase1_sha256_seal=seal,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: CONTINUACIÓN DEL KERNEL — MOTOR FPU, TELEGRAFISTA ANO, MEISSNER
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_QuarkAgentOrienter(Phase1_QuarkAgentObserver):
    r"""
    FASE 2 — ORIENT.  CONTINUACIÓN FORMAL DE LA FASE 1.

    Objeto inicial: `QuarkAgentObservationKernel` (salida terminal de
    `Phase1_QuarkAgentObserver.observe_and_bundle_agent_kernel`).

    Invoca al motor `QuarkColorConfinementSatelliteEngine`, resuelve el
    telegrafista del tubo ANO y evalúa el decaimiento de Schwinger.

    El método terminal `orient_agent_confinement` produce el objeto inicial
    de la Fase 3: `QuarkAgentOrientationDocket`.
    """

    def __init__(self, tolerance: float = 1.0e-12) -> None:
        super().__init__(tolerance=tolerance)
        try:
            self._engine: Final[QuarkColorConfinementSatelliteEngine] = (
                QuarkColorConfinementSatelliteEngine(tolerance=tolerance)
            )
        except TypeError:
            self._engine = QuarkColorConfinementSatelliteEngine()

    def _invoke_engine(
        self,
        kernel: QuarkAgentObservationKernel,
        alpha_s: float,
        string_tension: float,
        geodesic_distance_r: float,
    ) -> QuarkEngineState:
        """Invocación ciega compatible con firmas v1 y v2 del motor."""
        kwargs = {
            "color_triplet": kernel.color_triplet_vector,
            "G_metric": kernel.metric_cache.g_base,
            "alpha_s": float(alpha_s),
            "string_tension": float(string_tension),
            "geodesic_distance_r": float(geodesic_distance_r),
        }
        try:
            return self._engine.execute_quark_confinement_audit(
                density_matrix=kernel.color_density_matrix,
                **kwargs,
            )
        except TypeError:
            return self._engine.execute_quark_confinement_audit(**kwargs)

    @staticmethod
    def _solve_ano_telegrapher(
        string_energy: float,
        radius_r: float,
        alpha_s: float,
        string_tension: float,
    ) -> Tuple[float, float, float, float, float]:
        r"""
        Propagación transversal en la guía de onda del tubo ANO (unidades naturales).

            \(\gamma = \alpha + j\beta = \sqrt{(R'+j\omega L')(G'+j\omega C')}\),
            \(Z_c = \sqrt{(R'+j\omega L')/(G'+j\omega C')}\),

        con \(L'=C'=1\), \(\omega\sim\sigma\). Inmersión SI: \(Z_{\mathrm{SI}}=Z_{\mathrm{vac}}(1+\alpha_s)|Z_c|\).
        """
        r_safe = max(float(radius_r), _WILKINSON_SAFETY_FLOOR)
        omega = max(float(string_tension), _WILKINSON_SAFETY_FLOOR)
        l_prime = 1.0
        c_prime = 1.0
        r_prime = 0.05 * float(string_energy) / r_safe
        g_prime = 1.0e-4 * math.exp(min(float(string_energy) / 20.0, 50.0))

        z_series = complex(r_prime, omega * l_prime)
        y_shunt = complex(g_prime, omega * c_prime)
        gamma = (z_series * y_shunt) ** 0.5
        if abs(y_shunt) <= _WILKINSON_SAFETY_FLOOR:
            z_c = complex(1.0, 0.0)
        else:
            z_c = (z_series / y_shunt) ** 0.5

        z_natural = float(abs(z_c)) if abs(z_c) > _WILKINSON_SAFETY_FLOOR else 1.0
        z_si = float(_Z_VAC_SI_OHM * (1.0 + max(float(alpha_s), 0.0)) * z_natural)
        alpha_att = float(abs(gamma.real))
        beta_ph = float(abs(gamma.imag))
        return z_natural, z_si, alpha_att, beta_ph, float(abs(gamma))

    def orient_agent_confinement(
        self,
        kernel: QuarkAgentObservationKernel,
        alpha_s: float = 0.3,
        string_tension: float = 1.0,
        geodesic_distance_r: float = 1.0,
    ) -> QuarkAgentOrientationDocket:
        r"""
        MÉTODO TERMINAL FORMAL DE LA FASE 2.
        ────────────────────────────────────────────────────────────────────────
        CONTINÚA de `QuarkAgentObservationKernel` (Fase 1) y produce
        `QuarkAgentOrientationDocket`, objeto inicial de la Fase 3
        (`Phase3_QuarkAgentDecider.decide_and_actuate_governance`).

        Cadena anidada:
            Kernel  ──orient_agent_confinement──▶  Docket  ──▶  Fase 3
        """
        engine_state: QuarkEngineState = self._invoke_engine(
            kernel=kernel,
            alpha_s=alpha_s,
            string_tension=string_tension,
            geodesic_distance_r=geodesic_distance_r,
        )
        rep: QuarkConfinementReport = engine_state.report

        z_nat, z_si, alpha_ano, beta_ano, gamma_mod = self._solve_ano_telegrapher(
            string_energy=float(rep.string_tension_energy),
            radius_r=geodesic_distance_r,
            alpha_s=alpha_s,
            string_tension=string_tension,
        )

        engine_heyting = QuarkHeytingVerdict.from_engine_topos(engine_state.topos_truth_value)
        is_coherent = bool(
            rep.is_color_singlet
            and (not rep.is_string_broken)
            and engine_heyting is QuarkHeytingVerdict.COHERENT
        )

        hasher = hashlib.sha256()
        hasher.update(kernel.phase1_sha256_seal.encode("utf-8"))
        hasher.update(engine_state.cryptographic_seal.encode("utf-8"))
        hasher.update(rep.gell_mann_coherence_vector.tobytes())
        hasher.update(
            np.array(
                [z_nat, alpha_ano, float(rep.casimir_c2_fundamental)],
                dtype=np.float64,
            ).tobytes()
        )
        seal = hasher.hexdigest()

        logger.debug("Fase 2 (agente) sellada SHA-256=%s…", seal[:16])
        return QuarkAgentOrientationDocket(
            observation_kernel=kernel,
            engine_state=engine_state,
            gell_mann_coherence=np.array(rep.gell_mann_coherence_vector, copy=True),
            casimir_c2_value=float(rep.casimir_c2_fundamental),
            casimir_operator_expectation=float(
                getattr(rep, "casimir_operator_expectation", rep.casimir_c2_fundamental)
            ),
            color_neutrality_deviation=float(rep.color_neutrality_deviation),
            cartan_t3=float(getattr(rep, "cartan_t3", 0.0)),
            cartan_t8=float(getattr(rep, "cartan_t8", 0.0)),
            von_neumann_entropy=float(rep.von_neumann_entropy),
            is_color_singlet=bool(rep.is_color_singlet),
            cornell_potential_val=float(rep.cornell_potential_val),
            string_tension_energy=float(rep.string_tension_energy),
            luscher_correction_energy=float(rep.luscher_correction_energy),
            running_alpha_s=float(getattr(rep, "running_alpha_s", alpha_s)),
            is_string_broken=bool(rep.is_string_broken),
            ano_characteristic_impedance_natural=z_nat,
            ano_characteristic_impedance_ohms=z_si,
            ano_telegrapher_attenuation_alpha=alpha_ano,
            ano_telegrapher_phase_beta=beta_ano,
            ano_propagation_modulus=gamma_mod,
            schwinger_pair_production_rate=float(rep.schwinger_pair_production_rate),
            dual_meissner_kappa=float(getattr(rep, "dual_meissner_kappa", 0.0)),
            wilson_loop_area_law=float(getattr(rep, "wilson_loop_area_law", 0.0)),
            engine_heyting=engine_heyting,
            is_confinement_coherent=is_coherent,
            phase2_sha256_seal=seal,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: CONTINUACIÓN DEL DOCKET — TOPOS, FOCK/HMAC Y CROWBAR BT151
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_QuarkAgentDecider(Phase2_QuarkAgentOrienter):
    r"""
    FASE 3 — DECIDE & ACT.  CONTINUACIÓN FORMAL DE LA FASE 2.

    Objeto inicial: `QuarkAgentOrientationDocket`.
    Clasifica en \(\Omega_3\) por `meet` del topos del motor con el clasificador
    local, audita el positrón de autorización (HMAC-SHA512, anti-replay) y
    gobierna el Crowbar BT151 en IRAM/GPIO14.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        hmac_key: Optional[bytes] = None,
    ) -> None:
        super().__init__(tolerance=tolerance)
        self._safety_margin: Final[float] = float(safety_margin)
        self._grace_limit_seconds: Final[float] = float(grace_period_seconds)
        self._soft_veto_activation_timestamp: Optional[float] = None
        self._is_soft_veto_currently_engaged: bool = False
        self._canonical_authorized_positrons: Final[Tuple[str, ...]] = (
            "AUT_POS_SABIDURIA_777",
            "OVERRIDE_QUARK_CONFINEMENT_2026",
            "HMAC_SUTURA_FOCK_SECURE_QUARK",
        )
        self._canonical_digests: Final[Tuple[bytes, ...]] = tuple(
            hashlib.sha256(tok.encode("utf-8")).digest()
            for tok in self._canonical_authorized_positrons
        )
        self._hmac_key: Final[bytes] = hmac_key or b"APU_FILTER_FOCK_POSITRON_KEY_2026"
        self._consumed_positron_nonces: Dict[str, float] = {}
        self._lock: threading.RLock = threading.RLock()

    def reset_governance_state(self) -> None:
        """Reinicia gracia, veto suave y nonces (laboratorio / rearme)."""
        with self._lock:
            self._soft_veto_activation_timestamp = None
            self._is_soft_veto_currently_engaged = False
            self._consumed_positron_nonces.clear()

    def _remember_nonce(self, token_hash: str, now: float) -> None:
        self._consumed_positron_nonces[token_hash] = now
        overflow = len(self._consumed_positron_nonces) - _MAX_CONSUMED_NONCES
        if overflow > 0:
            for stale in list(self._consumed_positron_nonces.keys())[:overflow]:
                self._consumed_positron_nonces.pop(stale, None)

    def _canonical_hit(self, token: str) -> bool:
        token_digest = hashlib.sha256(token.encode("utf-8")).digest()
        hit = 0
        for allowed in self._canonical_digests:
            hit |= int(hmac.compare_digest(token_digest, allowed))
        return bool(hit)

    def _verify_hmac_payload(self, override_token: str, now: float) -> bool:
        payload, sep, signature = override_token.rpartition(":")
        if not sep or not payload or not signature:
            return False
        body = payload
        stamp_raw, has_stamp, rest = payload.rpartition(":")
        if has_stamp and stamp_raw and rest.lstrip("-").replace(".", "", 1).isdigit() is False:
            try:
                stamp = float(rest)
                if math.fabs(now - stamp) > _HMAC_TIMESTAMP_WINDOW_S:
                    return False
                body = stamp_raw
            except ValueError:
                body = payload
        expected = hmac.new(self._hmac_key, body.encode("utf-8"), hashlib.sha512).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _verify_fock_positron_annihilation(
        self,
        override_token: Optional[str],
        current_epoch: float,
    ) -> Tuple[bool, str]:
        r"""
        Audita el operador de override (metáfora \(e^++e^-\to 2\gamma\)).

        - Comparación a tiempo constante contra el conjunto canónico.
        - HMAC-SHA512 (`payload[:timestamp]:hmac`).
        - Nonces SHA3-512 anti-replay.

        Semántica intuicionista: la aniquilación **no** es una prueba de \(\top\);
        sólo impide el colapso \(\mathfrak{m}\to\bot\) por expiración de gracia.
        """
        if not override_token:
            return False, "Token de aniquilación nulo o no suministrado."

        token_hash = hashlib.sha3_512(override_token.encode("utf-8")).hexdigest()
        with self._lock:
            if token_hash in self._consumed_positron_nonces:
                return False, "Ataque de repetición detectado: nonce de positrón ya aniquilado."

            if self._canonical_hit(override_token):
                self._remember_nonce(token_hash, current_epoch)
                return True, "Positrón canónico absorbido: sutura de Fock aceptada (no prueba ⊤)."

            try:
                if self._verify_hmac_payload(override_token, current_epoch):
                    self._remember_nonce(token_hash, current_epoch)
                    return True, "Firma HMAC-SHA512 válida: aniquilación autorizada (no prueba ⊤)."
            except Exception as exc:
                logger.error("Fallo interno en validación HMAC de Fock: %s", exc)

        return False, "Firma de positrón espuria: operador no unitario en el álgebra de Fock."

    @staticmethod
    def _simulate_esp32_iram_crowbar_actuation(
        verdict: QuarkHeytingVerdict,
    ) -> Tuple[bool, float, float, bool, float]:
        r"""
        Conmutación ciber-física del ESP32 ante \(\bot\):

        1. ISR de alta prioridad en IRAM \(\sim 25\,\mathrm{ns}\).
        2. Escritura atómica `GPIO.out_w1ts = (1 << 14)` + pad \(12.5\,\mathrm{ns}\).
        3. Avalancha BT151-650R \(\sim 350\,\mathrm{ns}\) (\(I_{GT}>15\,\mathrm{mA}\), \(I_L=40\,\mathrm{mA}\)).
        4. Presupuesto total \(< 400\,\mathrm{ns}\).
        5. \(I^2t\) de semiciclo \(10\,\mathrm{ms}\) a \(I_{\mathrm{TSM}}=120\,\mathrm{A}\)
           (escala térmica, distinta de la latencia de disparo).
        """
        if verdict is not QuarkHeytingVerdict.VETOED:
            return False, 0.0, 0.0, True, 0.0

        total_latency_ns = (
            _ESP32_IRAM_WRITE_NS
            + _ESP32_GPIO14_PROPAGATION_DELAY_NS
            + _BT151_AVALANCHE_NS
        )
        iram_ok = bool(total_latency_ns <= _CROWBAR_IRAM_LATENCY_MAX_NS)
        if not iram_ok:
            logger.error(
                "Latencia Crowbar %.2f ns excede presupuesto IRAM %.1f ns.",
                total_latency_ns,
                _CROWBAR_IRAM_LATENCY_MAX_NS,
            )
        i2t = (_BT151_ITSM_A ** 2) * _BT151_SURGE_WINDOW_S / 2.0
        utilization = float(i2t / max(_BT151_I2T_RATING, _WILKINSON_SAFETY_FLOOR))
        return True, float(total_latency_ns), float(i2t), iram_ok, utilization

    def _local_heyting_classify(
        self,
        docket: QuarkAgentOrientationDocket,
    ) -> Tuple[QuarkHeytingVerdict, bool, bool]:
        r"""
        Clasificador local de calibre (independiente del motor, luego se `meet`).

        - \(\bot\): no singlete, quark libre, cuerda rota o \(E_\sigma\ge E_{\mathrm{brk}}\).
        - \(\mathfrak{m}\): singlete con cuerda elástica \(E_{\mathrm{el}}\le E_\sigma<E_{\mathrm{brk}}\)
          o Schwinger elevado.
        - \(\top\): singlete, cuerda corta, Schwinger bajo.

        Un 3-vector puro **nunca** es singlete (\(C_2(F)=4/3\)); el único estado
        \(SU(3)\)-invariante en \(M_3(\mathbb{C})\) es \(\rho=I/3\).
        """
        margin = max(float(self._safety_margin), _WILKINSON_SAFETY_FLOOR)
        break_thr = _STRING_BREAKAGE_THRESHOLD_ENERGY / margin
        elastic_lo = _ELASTIC_STRING_FLOOR_GEV * math.sqrt(margin)
        schwinger_soft = _SCHWINGER_SOFT_THRESHOLD / margin

        c2_val = docket.casimir_c2_value
        string_e = docket.string_tension_energy
        is_singlet = docket.is_color_singlet
        is_broken = docket.is_string_broken
        is_free = bool(docket.engine_state.is_free_quark_isolated)

        is_hard = bool(
            (not is_singlet)
            or is_free
            or is_broken
            or (c2_val > 0.05)
            or (string_e >= break_thr)
        )
        is_soft = bool(
            (not is_hard)
            and (
                (_CASIMIR_SINGLET_TOLERANCE < c2_val <= 0.05)
                or (elastic_lo < string_e < break_thr)
                or (docket.schwinger_pair_production_rate > schwinger_soft)
            )
        )
        if is_hard:
            return QuarkHeytingVerdict.VETOED, True, False
        if is_soft:
            return QuarkHeytingVerdict.DEGRADED, False, True
        return QuarkHeytingVerdict.COHERENT, False, False

    def decide_and_actuate_governance(
        self,
        docket: QuarkAgentOrientationDocket,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
    ) -> QuarkAgentCertificate:
        r"""
        MÉTODO TERMINAL FORMAL DE LA FASE 3 (DECIDE & ACT).
        ────────────────────────────────────────────────────────────────────────
        CONTINÚA de `QuarkAgentOrientationDocket` (Fase 2) y emite el
        `QuarkAgentCertificate` inviolable.

        Cadena anidada:
            Docket  ──decide_and_actuate_governance──▶  Certificate
        """
        curr_time = float(current_time if current_time is not None else time.time())
        local_h, is_hard, is_soft = self._local_heyting_classify(docket)
        composed = QuarkHeytingVerdict.meet(local_h, docket.engine_heyting)

        time_remaining = 0.0
        positron_annihilated = False
        grace_expired = False
        heyting_verdict = composed

        with self._lock:
            if heyting_verdict is QuarkHeytingVerdict.VETOED:
                self._is_soft_veto_currently_engaged = False
                self._soft_veto_activation_timestamp = None
                logger.error(
                    "VETO DURO: desconfinamiento SU(3)_c o ruptura ANO "
                    "(C2=%.3e, Eσ=%.3f, motor=%s, local=%s).",
                    docket.casimir_c2_value,
                    docket.string_tension_energy,
                    docket.engine_heyting.name,
                    local_h.name,
                )
            elif heyting_verdict is QuarkHeytingVerdict.DEGRADED:
                if not self._is_soft_veto_currently_engaged and not simulate_grace_expired:
                    self._is_soft_veto_currently_engaged = True
                    self._soft_veto_activation_timestamp = curr_time
                    time_remaining = self._grace_limit_seconds
                    logger.warning(
                        "VETO SUAVE: tubo ANO elástico (Eσ=%.3f GeV). Gracia=%.1f s.",
                        docket.string_tension_energy,
                        time_remaining,
                    )
                else:
                    elapsed = (
                        (curr_time - self._soft_veto_activation_timestamp)
                        if self._soft_veto_activation_timestamp is not None
                        else (self._grace_limit_seconds + 1.0)
                    )
                    time_remaining = max(0.0, self._grace_limit_seconds - elapsed)
                    if time_remaining <= self._tol or simulate_grace_expired:
                        heyting_verdict = QuarkHeytingVerdict.VETOED
                        grace_expired = True
                        is_hard = True
                        self._is_soft_veto_currently_engaged = False
                        logger.critical(
                            "Ventana de gracia expirada: colapso Heyting m → ⊥."
                        )
                    else:
                        heyting_verdict = QuarkHeytingVerdict.DEGRADED

                if (
                    override_token is not None
                    and heyting_verdict is QuarkHeytingVerdict.DEGRADED
                ):
                    is_valid, reason = self._verify_fock_positron_annihilation(
                        override_token, curr_time
                    )
                    if is_valid:
                        positron_annihilated = True
                        self._is_soft_veto_currently_engaged = False
                        self._soft_veto_activation_timestamp = None
                        time_remaining = 0.0
                        logger.info(
                            "Aniquilación de Fock aceptada (no prueba ⊤): %s", reason
                        )
                    else:
                        logger.error("Rechazo de operador de aniquilación: %s", reason)
            else:
                self._is_soft_veto_currently_engaged = False
                self._soft_veto_activation_timestamp = None
                heyting_verdict = QuarkHeytingVerdict.COHERENT

            soft_engaged = bool(self._is_soft_veto_currently_engaged)

        interlock, latency_ns, joule_heat, iram_ok, i2t_util = (
            self._simulate_esp32_iram_crowbar_actuation(heyting_verdict)
        )
        if interlock:
            logger.critical("CROWBAR BT151 DISPARADO EN SILICIO.")
            logger.critical(
                "  ISR IRAM+GPIO14+avalancha = %.2f ns (presupuesto %.1f ns, ok=%s)",
                latency_ns,
                _CROWBAR_IRAM_LATENCY_MAX_NS,
                iram_ok,
            )
            logger.critical(
                "  I_GT>%.1f mA, I_L=%.1f mA, I²t=%.4e A²s (utilización rating=%.3f)",
                _BT151_GATE_TRIGGER_CURRENT_MA,
                _BT151_LATCHING_CURRENT_MA,
                joule_heat,
                i2t_util,
            )

        neg = QuarkHeytingVerdict.pseudocomplement(heyting_verdict)
        impl_top = QuarkHeytingVerdict.implies(
            heyting_verdict, QuarkHeytingVerdict.COHERENT
        )
        kernel = docket.observation_kernel

        blake = hashlib.blake2b(digest_size=64)
        blake.update(docket.phase2_sha256_seal.encode("utf-8"))
        blake.update(heyting_verdict.name.encode("utf-8"))
        blake.update(
            np.array(
                [
                    docket.casimir_c2_value,
                    docket.string_tension_energy,
                    latency_ns,
                    joule_heat,
                    kernel.berry_pancharatnam_phase_rad,
                ],
                dtype=np.float64,
            ).tobytes()
        )
        master_seal = blake.hexdigest()

        return QuarkAgentCertificate(
            governance_phase="G_OMEGA_QUARK_CONFINEMENT_SUTURATED",
            heyting_verdict=heyting_verdict,
            heyting_negation=neg.name,
            heyting_implication_to_top=impl_top.name,
            engine_topos_name=docket.engine_heyting.name,
            casimir_c2_value=docket.casimir_c2_value,
            is_color_singlet=docket.is_color_singlet,
            cornell_potential_val=docket.cornell_potential_val,
            string_tension_energy=docket.string_tension_energy,
            berry_pancharatnam_phase_rad=kernel.berry_pancharatnam_phase_rad,
            wilson_barycentric_weight=kernel.wilson_barycentric_weight,
            color_participation_ratio=kernel.color_participation_ratio,
            ano_telegrapher_attenuation_alpha=docket.ano_telegrapher_attenuation_alpha,
            is_free_quark_detected=bool(
                docket.engine_state.is_free_quark_isolated or (not docket.is_color_singlet)
            ),
            is_string_breakage_critical=docket.is_string_broken,
            is_soft_veto_active=soft_engaged,
            override_grace_period_expired=bool(grace_expired or simulate_grace_expired),
            positron_annihilation_verified=positron_annihilated,
            positron_proves_top=bool(
                _POSITRON_DOES_NOT_PROVE_TOP and False
            ),
            hardware_interlock_fired=interlock,
            iram_budget_ok=iram_ok,
            actuation_latency_ns=latency_ns,
            silicon_thermal_joule_integral=joule_heat,
            i2t_rating_utilization=i2t_util,
            time_grace_remaining_seconds=time_remaining,
            phase1_sha256_seal=kernel.phase1_sha256_seal,
            phase2_sha256_seal=docket.phase2_sha256_seal,
            engine_cryptographic_seal=docket.engine_state.cryptographic_seal,
            digital_signature_blake2b=master_seal,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: SOBERANO DE CALIBRE Y GOBERNADOR DE CONFINAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
class QuarkColorConfinementSatelliteAgent(Phase3_QuarkAgentDecider):
    r"""
    AGENTE SOBERANO DE SUPERVISIÓN CIBER-FÍSICA EN LAZO CERRADO.

    Unifica las 3 fases anidadas por herencia lineal estricta:

        Observe (Fase 1) → Orient (Fase 2) → Decide & Act (Fase 3).
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        hmac_key: Optional[bytes] = None,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            safety_margin=safety_margin,
            grace_period_seconds=grace_period_seconds,
            hmac_key=hmac_key,
        )

    def audit_quark_confinement_cycle(
        self,
        color_triplet: NDArray[np.complex128],
        G_metric: NDArray[np.float64],
        alpha_s: float = 0.3,
        string_tension: float = 1.0,
        geodesic_distance_r: float = 1.0,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        density_matrix: Optional[NDArray[np.complex128]] = None,
        current_time: Optional[float] = None,
    ) -> QuarkAgentCertificate:
        r"""
        FACHADA PRINCIPAL DEL AGENTE — PIPELINE TRI-FÁSICO ANIDADO.

        1. Fase 1  → `QuarkAgentObservationKernel`
        2. Fase 2  → `QuarkAgentOrientationDocket`   [continúa del Kernel]
        3. Fase 3  → `QuarkAgentCertificate`         [continúa del Docket]
        """
        observation_kernel: QuarkAgentObservationKernel = self.observe_and_bundle_agent_kernel(
            color_triplet=color_triplet,
            G_metric=G_metric,
            density_matrix=density_matrix,
        )
        orientation_docket: QuarkAgentOrientationDocket = self.orient_agent_confinement(
            kernel=observation_kernel,
            alpha_s=alpha_s,
            string_tension=string_tension,
            geodesic_distance_r=geodesic_distance_r,
        )
        certificate: QuarkAgentCertificate = self.decide_and_actuate_governance(
            docket=orientation_docket,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
        )
        return certificate


# ══════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS DE LABORATORIO Y DEMOSTRACIÓN DE VERIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "═" * 80)
    print(" INICIALIZANDO SOBERANO: QUARK COLOR CONFINEMENT SATELLITE AGENT v2.0.0")
    print("═" * 80 + "\n")

    agent = QuarkColorConfinementSatelliteAgent()
    g_riemann = np.eye(4, dtype=np.float64)
    rho_singlet = np.eye(3, dtype=np.complex128) / 3.0
    c_center = (1.0 / math.sqrt(3.0)) * np.array(
        [1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=np.complex128
    )

    # --------------------------------------------------------------------------
    # ESCENARIO 1: Estado SU(3)-invariante ρ = I/3 (único singlete de estado)
    # Un 3-vector puro NUNCA es singlete: C_2(F)=4/3 y ||n||²=1/3.
    # --------------------------------------------------------------------------
    print(">>> ESCENARIO 1: Singlete de estado ρ = I/3, cuerda corta (r=1.2)...")
    cert_1 = agent.audit_quark_confinement_cycle(
        color_triplet=c_center,
        G_metric=g_riemann,
        alpha_s=0.3,
        string_tension=1.0,
        geodesic_distance_r=1.2,
        density_matrix=rho_singlet,
    )
    print(f"  [+] Fase 1 - Sello SHA-256                : {cert_1.phase1_sha256_seal[:20]}...")
    print(f"  [+] Fase 1 - Pancharatnam γ_P             : {cert_1.berry_pancharatnam_phase_rad:.6f} rad")
    print(f"  [+] Fase 1 - Wilson baricéntrico          : {cert_1.wilson_barycentric_weight:.6f}")
    print(f"  [+] Fase 1 - Participación de color       : {cert_1.color_participation_ratio:.6f}")
    print(f"  [*] Fase 2 - Operador Casimir C_2         : {cert_1.casimir_c2_value:.6e}")
    print(f"  [*] Fase 2 - Potencial Cornell V(r)       : {cert_1.cornell_potential_val:.4f} GeV")
    print(f"  [*] Fase 2 - α_p telegrafista ANO         : {cert_1.ano_telegrapher_attenuation_alpha:.4e}")
    print(f"  [*] Fase 2 - Topos del motor              : {cert_1.engine_topos_name}")
    print(f"  [#] Fase 3 - Veredicto Heyting            : {cert_1.heyting_verdict.name} ({cert_1.heyting_verdict.value})")
    print(f"  [#] Fase 3 - ¬χ / χ⇒⊤                     : {cert_1.heyting_negation} / {cert_1.heyting_implication_to_top}")
    print(f"  [#] Fase 3 - Crowbar disparado            : {cert_1.hardware_interlock_fired}")
    print(f"  [#] Fase 3 - Sello BLAKE2b-512            : {cert_1.digital_signature_blake2b[:32]}...")
    print("-" * 80)

    agent.reset_governance_state()

    # --------------------------------------------------------------------------
    # ESCENARIO 2: Singlete con cuerda elástica (35 GeV) + override de positrón
    # El positrón NO prueba ⊤; sólo impide el colapso m → ⊥.
    # --------------------------------------------------------------------------
    print(">>> ESCENARIO 2: Tubo ANO elástico (ρ=I/3, r=35) + override Fock...")
    cert_2 = agent.audit_quark_confinement_cycle(
        color_triplet=c_center,
        G_metric=g_riemann,
        alpha_s=0.3,
        string_tension=1.0,
        geodesic_distance_r=35.0,
        override_token="AUT_POS_SABIDURIA_777",
        density_matrix=rho_singlet,
    )
    print(f"  [*] Fase 2 - Energía tensión de cuerda    : {cert_2.string_tension_energy:.2f} GeV")
    print(f"  [#] Fase 3 - Veredicto Heyting            : {cert_2.heyting_verdict.name}")
    print(f"  [#] Fase 3 - Positrón aniquilado          : {cert_2.positron_annihilation_verified}")
    print(f"  [#] Fase 3 - ¿Positrón prueba ⊤?          : {cert_2.positron_proves_top}")
    print(f"  [#] Fase 3 - Crowbar disparado            : {cert_2.hardware_interlock_fired}")
    print("-" * 80)

    agent.reset_governance_state()

    # --------------------------------------------------------------------------
    # ESCENARIO 3: Quark rojo puro + ruptura de cuerda (120 GeV) → veto duro
    # --------------------------------------------------------------------------
    print(">>> ESCENARIO 3: Quark rojo aislado + ruptura de cuerda (r=120)...")
    c_free_red = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    cert_3 = agent.audit_quark_confinement_cycle(
        color_triplet=c_free_red,
        G_metric=g_riemann,
        alpha_s=0.3,
        string_tension=1.0,
        geodesic_distance_r=120.0,
    )
    print(f"  [+] Fase 1 - Wilson baricéntrico (eje)    : {cert_3.wilson_barycentric_weight:.3e}")
    print(f"  [+] Fase 1 - Participación de color       : {cert_3.color_participation_ratio:.6f}")
    print(f"  [*] Fase 2 - ¿Cuerda rota / hadronizada?  : {cert_3.is_string_breakage_critical}")
    print(f"  [*] Fase 2 - Quark libre detectado        : {cert_3.is_free_quark_detected}")
    print(f"  [#] Fase 3 - Veredicto Heyting            : {cert_3.heyting_verdict.name}")
    print(f"  [#] Fase 3 - Interlock Crowbar            : {cert_3.hardware_interlock_fired}")
    print(f"  [#] Fase 3 - Latencia IRAM                : {cert_3.actuation_latency_ns:.2f} ns (ok={cert_3.iram_budget_ok})")
    print(f"  [#] Fase 3 - I²t / utilización rating     : {cert_3.silicon_thermal_joule_integral:.4e} A²s / {cert_3.i2t_rating_utilization:.3f}")
    print(f"  [#] Fase 3 - Sello BLAKE2b-512            : {cert_3.digital_signature_blake2b[:32]}...")
    print("═" * 80)
    print(" VERIFICACIÓN CIBER-FÍSICA Y ESPECTRAL DEL SOBERANO AGENT COMPLETADA")
    print("═" * 80 + "\n")
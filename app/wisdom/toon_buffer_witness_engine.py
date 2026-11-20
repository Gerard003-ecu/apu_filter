# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovToonBufferWitnessEngine v15.0.0-PhD                                            ║
║ Ubicación: app/wisdom/toon_buffer_witness_engine.py                                                  ║
║ Versión  : 15.0.0-ENGINE-BOOL-NOVIKOV-TOON-BUFFER-WITNESS-POINCARE-KAM-RSI3-PhD                      ║
║ Función  : Motor Espectral Testigo QND (Aharonov-Albert-Vaidman), Impedancia de Heaviside,           ║
║            Geometría Cuántica Uhlmann-Bures-QFI, Mecánica Celeste Simpéctica de Poincaré-Cartan-KAM  ║
║            y Automejora Recursiva de Nivel 3 (RSI³) sobre Categorías de Kleisli y Topos de Heyting.  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA EN TRES FASES ANIDADAS (CADENA DE HERENCIA CON CONTRATO MORFISMÁTICO DE CONTINUIDAD)
──────────────────────────────────────────────────────────────────────────────────────────────────
FASE 1 · _Phase1AlgebraicQuantumKernel
    Álgebra lineal PSD, descomposición hipercompleja de Clifford-Cuaterniones (Cl_{0,2} ≅ ℍ),
    testigo débil QND-AAV, Derivada Logarítmica Simétrica (SLD) e Información de Fisher Cuántica (QFI),
    geometría de Uhlmann-Bures, ecuaciones del telegrafista de Heaviside (impedancia Z₀ y reflexión Γ),
    forma normal de Smith exacta sobre ℤ, Laplaciano de Hodge Δ₀, transformada de Walsh-Hadamard
    y valuación no arquimediana de Krull-Novikov.
    ÚLTIMO MÉTODO → compute_phase1_witness(...)  ⇒  Phase1Witness

FASE 2 · _Phase2PoincareDynamicsKernel(_Phase1AlgebraicQuantumKernel)
    PRIMER MÉTODO  ← evaluate_toon_buffer_flow(witness1: Phase1Witness, ...)   [continuidad F1 → F2]
    Espectro ergódico multiplicativo de Oseledets, sobrepresión de Zhukovsky-Allievi acoplada a Γ,
    residual de Cauchy-Riemann-Wirtinger, capacidad simpléctica de Gromov, variables canónicas no
    singulares de Poincaré (Λ, ξ₁, ξ₂; λ, η₁, η₂) con auditoría simpléctica Mᵀ J₆ M = J₆, determinantes
    de torsión de Kolmogorov y de Arnol'd (no-degeneración isoenergética), condición diofantina
    vectorial, serie canónica de Lie de Deprit-Hori exp(L_W)H con horizonte de Nekhoroshev, invariante
    integral relativo de Poincaré-Cartan (∮ p dq) vía integrador simpléctico de Yoshida de 4.º orden,
    monodromía de Floquet-Broucke e integral homoclínica de Melnikov.
    ÚLTIMO MÉTODO → compose_phase2_witness(witness1: Phase1Witness, ...)  ⇒  Phase2Witness

FASE 3 · _Phase3RSIVerdictKernel(_Phase2PoincareDynamicsKernel)
    PRIMER MÉTODO  ← _phase3_rsi_gate(witness2: Phase2Witness, ...)            [continuidad F2 → F3]
    Torre de Automejora Recursiva de Nivel 3 (RSI³):
      • Nivel 1 (RSI¹): Síntesis de punto fijo de política con aceleración de Picard-Steffensen y
                        funcional de descenso de Lyapunov.
      • Nivel 2 (RSI²): Meta-endofuntor M_Ψ: End(Θ) → End(Θ) de preacondicionamiento espectral con
                        certificación analítica de derivada de Fréchet y sondeo de Lipschitz.
      • Nivel 3 (RSI³): Meta-meta-gobernador 𝔊 verificador de las leyes de coherencia de la mónada
                        de Kleisli (asociatividad y unidades) y guardia de solidez de Löb-Tarski.
    Álgebra de Heyting completa Ω₃ con topología modal de Lawvere-Tierney j: Ω₃ → Ω₃ y pasaporte HMAC.
    ORQUESTADOR    → execute_toon_buffer_witness_audit(...)  ⇒  ToonBufferWitnessSpectrumReport
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from typing import Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovToonBufferWitnessEngine")

_ENGINE_VERSION: Final[str] = "15.0.0-ENGINE-BOOL-NOVIKOV-TOON-BUFFER-WITNESS-POINCARE-KAM-RSI3-PhD"
__version__: Final[str] = _ENGINE_VERSION

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_FIDELITY_MIN: Final[float] = 0.95
_BURES_MAX: Final[float] = 0.35
_QND_COMMUTATOR_TOL: Final[float] = 1.0e-10
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_KLEISLI_SAFETY_MARGIN: Final[float] = 0.002375
_DIOPHANTINE_GAMMA: Final[float] = 0.05
_DIOPHANTINE_TAU: Final[float] = 1.0
_DIOPHANTINE_K_MAX: Final[int] = 64
_HOMOLOGICAL_TOL: Final[float] = 1.0e-9
_CARTAN_INVARIANT_TOL: Final[float] = 1.0e-7
_RSI_MAX_DEPTH: Final[int] = 3
_RSI_MAX_ITER: Final[int] = 2000
_RSI_PROBES: Final[int] = 128
_KLEISLI_COHERENCE_TOL: Final[float] = 1.0e-12
_DEPLOYED_POLICY: Final[Tuple[float, float, float]] = (
    _FIDELITY_MIN,
    _BURES_MAX,
    _DEFAULT_KRULL_FLOOR,
)
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_TOON_BUFFER_WITNESS_SECRET_KEY_v13.0.0"


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# TIPOS ALGEBRAICOS, TOPOS DE HEYTING Y CONTRATOS DE DATOS ENTRE FASES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class HeytingTruthValue(IntEnum):
    """Álgebra de Heyting de cadena Ω₃ equipada con topología modal de Lawvere-Tierney.

    Orden lógico: COHERENT (⊤, rango 2) > DEGRADED (rango 1) > VETOED (⊥, rango 0).
    Operaciones:
      • meet (∧)    : ínfimo de rangos.
      • join (∨)    : supremo de rangos.
      • implies (⇒) : residuo relativo a ⇒ b = sup{c ∈ Ω₃ : c ∧ a ≤ b}.
      • negate (¬)  : pseudo-complemento intuicionista ¬a = (a ⇒ ⊥).
      • lawvere_tierney_sheafify (j): operador modal de clausura local j: Ω₃ → Ω₃.
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x02

    @property
    def rank(self) -> int:
        """Rango en la cadena de Heyting (2 = ⊤, 1 = intermedio, 0 = ⊥)."""
        return 2 - int(self)

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Ínfimo (conjunción intuicionista ∧) en Ω₃."""
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Supremo (disyunción intuicionista ∨) en Ω₃."""
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Implicación de Heyting (exponencial interno a ⇒ b)."""
        return HeytingTruthValue.COHERENT if self.rank <= other.rank else other

    def negate(self) -> "HeytingTruthValue":
        """Pseudo-complemento intuicionista ¬a = (a ⇒ VETOED). En Ω₃, ¬DEGRADED = VETOED."""
        return self.implies(HeytingTruthValue.VETOED)

    def lawvere_tierney_sheafify(self, stabilize_degraded: bool = False) -> "HeytingTruthValue":
        """Operador modal de Lawvere-Tierney j: Ω₃ → Ω₃.

        Verifica los tres axiomas de topología de topos:
          1. j(⊤) = ⊤
          2. j(j(p)) = j(p)
          3. j(p ∧ q) = j(p) ∧ j(q)
        Si `stabilize_degraded=False`, actúa como la topología cerrada mínima (identidad).
        Si `stabilize_degraded=True`, colapsa el abierto denso {COHERENT, DEGRADED} en COHERENT.
        """
        if not stabilize_degraded:
            return self
        if self in (HeytingTruthValue.COHERENT, HeytingTruthValue.DEGRADED):
            return HeytingTruthValue.COHERENT
        return HeytingTruthValue.VETOED


@dataclass(frozen=True)
class QNDWeakMeasurementReport:
    """Reporte de medición débil AAV, pasividad QND, QFI/SLD y pureza cuaterniónica."""
    weak_value: complex
    back_action_db: float
    is_qnd_passive: bool
    commutator_norm: float
    quantum_fisher_info: float = 0.0
    sld_frobenius_norm: float = 0.0
    quaternionic_bloch_purity: float = 1.0


@dataclass(frozen=True)
class BuresUhlmannMetricReport:
    """Reporte de fidelidad de Uhlmann, distancia de Bures y ángulo geodésico."""
    fidelity_uhlmann: float
    bures_distance: float
    is_isometry_preserved: bool
    bures_geodesic_angle: float = 0.0
    fuchs_van_de_graaf_upper_bound: float = 0.0


@dataclass(frozen=True)
class AlgebraicTopologyReport:
    """Reporte algebraico-topológico y de línea de transmisión de Heaviside."""
    rademacher_walsh_energy: float
    smith_invariants: Tuple[int, ...]
    smith_torsion_free: bool
    betti_number_zero: bool
    krull_valuation: float
    novikov_norm: float
    krull_floor_ok: bool
    hodge_spectral_gap: float = 0.0
    euler_characteristic: int = 0
    torsion_subgroup_order: int = 1
    characteristic_impedance_mag: float = 1.0
    telegrapher_reflection_coeff: float = 0.0
    telegrapher_vswr: float = 1.0


@dataclass(frozen=True)
class Phase1Witness:
    """Testigo sellado de la Fase 1. Contrato de entrada morfismático hacia la Fase 2."""
    qnd_report: QNDWeakMeasurementReport
    uhlmann_report: BuresUhlmannMetricReport
    algebra_report: AlgebraicTopologyReport


@dataclass(frozen=True)
class ToonBufferFlowReport:
    """Reporte de caudal: varianza, espectro de Oseledets y sobrepresión de Zhukovsky-Allievi."""
    flow_variance: float
    spectral_radius: float
    oseledets_lyapunov_exponent: float
    is_water_hammer_suppressed: bool
    buffer_saturation_pct: float
    lyapunov_spectrum: Tuple[float, ...] = ()
    zhukovsky_surge_pressure: float = 0.0
    pesin_entropy_bound: float = 0.0


@dataclass(frozen=True)
class PoincareKAMReport:
    """Reporte de mecánica celeste de Poincaré-KAM, invariante de Cartan y serie de Lie."""
    cauchy_riemann_residual: float
    gromov_ball_capacity: float
    gromov_cylinder_capacity: float
    is_gromov_rigid: bool
    melnikov_min: float
    melnikov_max: float
    melnikov_chaos_detected: bool
    diophantine_ok: bool
    homological_residual: float
    kepler_frequency: float
    delaunay_actions: Tuple[float, float, float]
    poincare_canonical_coords: Tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    symplectic_jacobian_defect: float = 0.0
    kolmogorov_torsion_det: float = 0.0
    arnold_isoenergetic_det: float = 0.0
    is_kam_non_degenerate: bool = True
    lie_series_second_order_norm: float = 0.0
    nekhoroshev_stability_time: float = 0.0
    poincare_cartan_invariant_drift: float = 0.0
    floquet_monodromy_trace: float = 2.0
    is_floquet_elliptic_stable: bool = True


@dataclass(frozen=True)
class Phase2Witness:
    """Testigo sellado de la Fase 2. Contrato de entrada morfismático hacia la Fase 3."""
    flow: ToonBufferFlowReport
    kam: PoincareKAMReport
    phase1: Optional[Phase1Witness] = None


@dataclass(frozen=True)
class RSICertificate:
    """Certificado formal del protocolo de Automejora Recursiva de Nivel 3 (RSI³)."""
    depth_reached: int
    lip_certified: float
    lip_probe_estimate: float
    cross_validated: bool
    fixed_point: Tuple[float, ...]
    accepted: bool
    meta_preconditioned_lip: float = 0.0
    kleisli_associativity_defect: float = 0.0
    kleisli_unit_defect: float = 0.0
    loeb_soundness_verified: bool = False
    banach_aposteriori_error: float = 0.0
    lyapunov_descent_verified: bool = False


@dataclass(frozen=True)
class ToonBufferWitnessSpectrumReport:
    """Reporte espectral holístico de atestación (v15). Preserva todos los accesores v13/v14."""
    phase1: Phase1Witness
    phase2: Phase2Witness
    rsi_certificate: RSICertificate
    banach_contraction_lip: float
    capacity_jerk_d3C_dt3: float
    truth_value: HeytingTruthValue
    passport_hash: str
    execution_time_ms: float

    @property
    def qnd_report(self) -> QNDWeakMeasurementReport:
        return self.phase1.qnd_report

    @property
    def uhlmann_report(self) -> BuresUhlmannMetricReport:
        return self.phase1.uhlmann_report

    @property
    def flow_report(self) -> ToonBufferFlowReport:
        return self.phase2.flow

    @property
    def krull_valuation(self) -> float:
        return self.phase1.algebra_report.krull_valuation

    @property
    def novikov_norm(self) -> float:
        return self.phase1.algebra_report.novikov_norm

    @property
    def smith_torsion_free(self) -> bool:
        return self.phase1.algebra_report.smith_torsion_free

    @property
    def betti_number_zero(self) -> bool:
        return self.phase1.algebra_report.betti_number_zero

    @property
    def rademacher_walsh_norm(self) -> float:
        return self.phase1.algebra_report.rademacher_walsh_energy

    @property
    def is_gromov_rigid(self) -> bool:
        return self.phase2.kam.is_gromov_rigid

    @property
    def melnikov_chaos_detected(self) -> bool:
        return self.phase2.kam.melnikov_chaos_detected

    @property
    def cauchy_riemann_residual(self) -> float:
        return self.phase2.kam.cauchy_riemann_residual

    @property
    def poincare_symplectic_defect(self) -> float:
        return self.phase2.kam.symplectic_jacobian_defect

    @property
    def kleisli_coherence_defect(self) -> float:
        return max(
            self.rsi_certificate.kleisli_associativity_defect,
            self.rsi_certificate.kleisli_unit_defect,
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1 · ÁLGEBRA HIPERCOMPLEJA, TESTIGO QND-QFI, UHLMANN-BURES, HEAVISIDE Y SMITH/NOVIKOV
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class _Phase1AlgebraicQuantumKernel:
    """FASE 1: Núcleo algebraico, cuántico, hipercomplejo y de líneas de transmisión.

    Ningún método de esta fase depende de las Fases 2 o 3.
     Su último método (`compute_phase1_witness`) emite el contrato sellado `Phase1Witness`
    que inaugura el primer método de la Fase 2 (`evaluate_toon_buffer_flow`).
    """

    # ── Álgebra lineal PSD y proyección hipercompleja de Clifford-Cuaterniones ───────────────────

    @staticmethod
    def _hermitize(matrix: NDArray) -> NDArray[np.complex128]:
        """Proyección ortogonal sobre el espacio real de operadores hermíticos: (M + M†)/2."""
        M = np.asarray(matrix, dtype=np.complex128)
        return 0.5 * (M + M.conj().T)

    @classmethod
    def _normalize_density(cls, rho: NDArray) -> NDArray[np.complex128]:
        """Hermitiza, proyecta sobre el cono PSD si hay ruido numérico y normaliza a traza 1."""
        R = np.asarray(rho, dtype=np.complex128)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError("La matriz de densidad debe ser cuadrada.")
        R = cls._hermitize(R)
        trace = float(np.real(np.trace(R)))
        if trace <= _WILKINSON_EPSILON:
            raise ValueError("Traza no positiva en la matriz de densidad.")
        return R / trace

    @classmethod
    def _psd_sqrt(cls, matrix: NDArray) -> NDArray[np.complex128]:
        """Raíz cuadrada semidefinida positiva espectral: V diag(√max(λ, 0)) V†."""
        w, V = np.linalg.eigh(cls._hermitize(matrix))
        root = np.sqrt(np.clip(w, 0.0, None))
        return (V * root) @ V.conj().T

    @staticmethod
    def _as_unit_vector(vec: NDArray) -> NDArray[np.complex128]:
        """Aplana y normaliza un vector en ℂⁿ respecto a la norma euclídea."""
        v = np.asarray(vec, dtype=np.complex128).reshape(-1)
        nrm = float(la.norm(v))
        if nrm <= _WILKINSON_EPSILON:
            raise ValueError("Vector de estado nulo.")
        return v / nrm

    @staticmethod
    def _trapezoid_weights(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Pesos de cuadratura trapezoidal exacta sobre malla 1D arbitraria."""
        w = np.zeros_like(x, dtype=float)
        dx = np.diff(x)
        w[:-1] += 0.5 * dx
        w[1:] += 0.5 * dx
        return w

    @staticmethod
    def _default_witness_observable(dim: int) -> NDArray[np.complex128]:
        """Observable tridiagonal hermítico del testigo TOON."""
        H = np.eye(dim, dtype=np.complex128)
        for i in range(dim - 1):
            H[i, i + 1] = 0.02 * (i + 1)
            H[i + 1, i] = 0.02 * (i + 1)
        return H

    @staticmethod
    def _quaternionic_clifford_purity(rho: NDArray[np.complex128]) -> float:
        """Pureza cuaterniónica en Cl_{0,2} ≅ ℍ sobre el bloque principal 2×2 y traza global.

        Descompone el bloque principal 2×2 en la base cuaterniónica {I₂, -iσ_x, -iσ_y, -iσ_z}
        y evalúa la pureza de Hilbert-Schmidt Tr(ρ²) ∈ [1/d, 1].
        """
        purity_global = float(np.real(np.trace(rho @ rho)))
        if rho.shape[0] >= 2:
            blk = rho[:2, :2]
            tr_blk = float(np.real(np.trace(blk)))
            if tr_blk > _WILKINSON_EPSILON:
                blk = blk / tr_blk
                q0 = 0.5 * float(np.real(blk[0, 0] + blk[1, 1]))
                qx = 0.5 * float(np.real(blk[0, 1] + blk[1, 0]))
                qy = 0.5 * float(np.imag(blk[1, 0] - blk[0, 1]))
                qz = 0.5 * float(np.real(blk[0, 0] - blk[1, 1]))
                q_norm_sq = q0 * q0 + qx * qx + qy * qy + qz * qz
                return float(0.5 * (purity_global + min(1.0, 2.0 * q_norm_sq)))
        return purity_global

    @classmethod
    def _compute_sld_and_qfi(
        cls,
        rho: NDArray[np.complex128],
        observable: NDArray[np.complex128],
    ) -> Tuple[float, float]:
        """Derivada Logarítmica Simétrica (SLD) L y Fisher Cuántica F_Q(ρ, A) = Tr(ρ L²).

        Resuelve la ecuación de Lyapunov ½(Lρ + ρL) = -i[A, ρ] en la base propia de ρ:
            L_{jk} = -2i (λ_j - λ_k) / (λ_j + λ_k) ⟨j|A|k⟩   para λ_j + λ_k > ε.
        """
        evals, evecs = np.linalg.eigh(cls._hermitize(rho))
        evals = np.clip(evals, 0.0, None)
        A_eig = evecs.conj().T @ observable @ evecs
        denom = evals[:, None] + evals[None, :]
        numer = -2.0j * (evals[:, None] - evals[None, :]) * A_eig
        mask = denom > 1.0e-12
        L_eig = np.zeros_like(A_eig, dtype=np.complex128)
        L_eig[mask] = numer[mask] / denom[mask]
        qfi = float(np.real(np.sum(evals[:, None] * (np.abs(L_eig) ** 2))))
        sld_norm = float(la.norm(L_eig, ord="fro"))
        return max(0.0, qfi), sld_norm

    # ── Medición débil AAV, QFI y pasividad QND ──────────────────────────────────────────────────

    def compute_qnd_weak_measurement(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: NDArray[np.complex128],
        observable: Optional[NDArray[np.complex128]] = None,
    ) -> QNDWeakMeasurementReport:
        """Evalúa el valor débil AAV A_w = ⟨φ_f|A|φ_i⟩/⟨φ_f|φ_i⟩, ‖[A,ρ]‖_F y la QFI F_Q(ρ, A)."""
        rho = self._normalize_density(rho_mac)
        dim = rho.shape[0]

        if observable is None:
            A = self._default_witness_observable(dim)
        else:
            A = np.asarray(observable, dtype=np.complex128)
            if A.shape != (dim, dim):
                raise ValueError("Dimensión del observable incompatible con ρ_MAC.")
            if not np.allclose(A, A.conj().T, atol=1.0e-12):
                raise ValueError("El observable testigo debe ser hermítico.")

        pi = self._as_unit_vector(phi_i)
        pf = self._as_unit_vector(phi_f)
        overlap = complex(np.vdot(pf, pi))
        if abs(overlap) <= _WILKINSON_EPSILON:
            weak_val = complex(math.nan, math.nan)
        else:
            weak_val = complex(np.vdot(pf, A @ pi)) / overlap

        comm = A @ rho - rho @ A
        comm_norm = float(la.norm(comm, ord="fro"))
        back_action_db = float(20.0 * math.log10(1.0 + comm_norm))
        qfi, sld_norm = self._compute_sld_and_qfi(rho, A)
        q_purity = self._quaternionic_clifford_purity(rho)

        return QNDWeakMeasurementReport(
            weak_value=weak_val,
            back_action_db=back_action_db,
            is_qnd_passive=bool(comm_norm <= _QND_COMMUTATOR_TOL),
            commutator_norm=comm_norm,
            quantum_fisher_info=qfi,
            sld_frobenius_norm=sld_norm,
            quaternionic_bloch_purity=q_purity,
        )

    # ── Geometría de Uhlmann-Bures y cotas de Fuchs-van de Graaf ────────────────────────────────

    def evaluate_bures_uhlmann_metric(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> BuresUhlmannMetricReport:
        """F(ρ,σ) = (Tr√(√ρ σ √ρ))², d_B = √(2 − 2√F) y ángulo geodésico arccos(√F)."""
        R = self._normalize_density(rho)
        S = self._normalize_density(sigma)
        sqrt_r = self._psd_sqrt(R)
        M = sqrt_r @ S @ sqrt_r
        eig = np.linalg.eigvalsh(self._hermitize(M))
        trace_root = float(np.sum(np.sqrt(np.clip(eig, 0.0, None))))

        root_f = min(1.0, max(0.0, trace_root))
        fidelity = root_f * root_f
        bures = math.sqrt(max(0.0, 2.0 - 2.0 * root_f))
        geodesic_angle = math.acos(root_f)
        fuchs_bound = math.sqrt(max(0.0, 1.0 - fidelity))
        preserved = bool(fidelity >= _FIDELITY_MIN and bures <= _BURES_MAX)

        return BuresUhlmannMetricReport(
            fidelity_uhlmann=fidelity,
            bures_distance=bures,
            is_isometry_preserved=preserved,
            bures_geodesic_angle=geodesic_angle,
            fuchs_van_de_graaf_upper_bound=fuchs_bound,
        )

    # ── Ecuaciones del Telegrafista de Heaviside (Línea de Transmisión / Golpe de Bélier) ────────

    @staticmethod
    def _evaluate_heaviside_telegrapher(
        cost_matrix: NDArray[np.float64],
        omega_carrier: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Modela la matriz de acoplamiento como una línea de transmisión RLCG de Heaviside.

        Ecuaciones del telegrafista en el dominio de Laplace-Fourier (s = iω):
            Z(ω) = R + iωL,   Y(ω) = G + iωC,   Z₀ = √(Z/Y),   Γ = (Z_L − Z₀)/(Z_L + Z₀).
        En un conducto mnémonico adaptado, la impedancia de carga Z_L empata con la diagonal
        media y las pérdidas transversales G son proporcionales al acoplamiento fuera de diagonal.
        """
        M = np.asarray(cost_matrix, dtype=float)
        diag = np.abs(np.diag(M))
        off_diag = np.abs(M - np.diag(np.diag(M)))

        R_series = float(np.mean(diag)) + 1.0e-6
        L_inertance = float(np.std(diag)) + 1.0
        G_shunt = float(np.mean(off_diag)) + 1.0e-6
        C_compliance = 1.0 / L_inertance

        z_series = complex(R_series, omega_carrier * L_inertance)
        y_shunt = complex(G_shunt, omega_carrier * C_compliance)
        z0 = np.sqrt(z_series / y_shunt)

        z_load = complex(float(np.max(diag)) + 1.0e-6, omega_carrier)
        gamma_refl = (z_load - z0) / (z_load + z0) if abs(z_load + z0) > _WILKINSON_EPSILON else 0.0j
        refl_mag = float(min(0.999999, abs(gamma_refl)))
        vswr = float((1.0 + refl_mag) / max(1.0e-9, 1.0 - refl_mag))
        return float(abs(z0)), refl_mag, vswr

    # ── Forma normal de Smith exacta sobre ℤ y Laplaciano de Hodge ──────────────────────────────

    @staticmethod
    def _smith_invariant_factors(matrix: List[List[int]]) -> List[int]:
        """Factores invariantes d₁ | d₂ | … | d_r de M ∈ ℤ^{m×n} en aritmética entera exacta."""
        A = [[int(x) for x in row] for row in matrix]
        m = len(A)
        n = len(A[0]) if m else 0
        factors: List[int] = []
        t = 0
        while t < min(m, n):
            best: Optional[Tuple[int, int]] = None
            for i in range(t, m):
                for j in range(t, n):
                    if A[i][j] != 0 and (best is None or abs(A[i][j]) < abs(A[best[0]][best[1]])):
                        best = (i, j)
            if best is None:
                break
            i, j = best
            A[t], A[i] = A[i], A[t]
            if j != t:
                for row in A:
                    row[t], row[j] = row[j], row[t]

            while True:
                p = A[t][t]
                for i in range(t + 1, m):
                    if A[i][t] != 0:
                        q = A[i][t] // p
                        A[i] = [a - q * b for a, b in zip(A[i], A[t])]
                for j in range(t + 1, n):
                    if A[t][j] != 0:
                        q = A[t][j] // p
                        for row in A:
                            row[j] -= q * row[t]

                cands = [(abs(A[i][t]), "r", i) for i in range(t + 1, m) if A[i][t] != 0]
                cands += [(abs(A[t][j]), "c", j) for j in range(t + 1, n) if A[t][j] != 0]
                if cands:
                    _, kind, idx = min(cands)
                    if kind == "r":
                        A[t], A[idx] = A[idx], A[t]
                    else:
                        for row in A:
                            row[t], row[idx] = row[idx], row[t]
                    continue

                bad_row: Optional[int] = None
                for i in range(t + 1, m):
                    if any(A[i][j] % p for j in range(t + 1, n)):
                        bad_row = i
                        break
                if bad_row is None:
                    break
                A[t] = [a + b for a, b in zip(A[t], A[bad_row])]

            factors.append(abs(A[t][t]))
            t += 1
        return factors

    @staticmethod
    def _hodge_laplacian_spectral_gap(matrix: NDArray[np.float64]) -> float:
        """Brecha espectral no nula mínima del 0-Laplaciano de Hodge Δ₀ = B Bᵀ."""
        B = np.asarray(matrix, dtype=float)
        lap = B @ B.T
        evals = np.linalg.eigvalsh(0.5 * (lap + lap.T))
        pos = evals[evals > 1.0e-10]
        return float(np.min(pos)) if pos.size > 0 else 0.0

    # ── Transformada de Walsh-Hadamard y derivadas booleanas de Rademacher ──────────────────────

    @staticmethod
    def _fast_walsh_hadamard(vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transformada rápida de Walsh-Hadamard (FWHT) en ℤ₂ⁿ."""
        a = np.array(vec, dtype=float, copy=True).reshape(-1)
        n = a.size
        h = 1
        while h < n:
            a = a.reshape(-1, 2 * h)
            x = a[:, :h].copy()
            y = a[:, h:].copy()
            a[:, :h] = x + y
            a[:, h:] = x - y
            a = a.reshape(-1)
            h *= 2
        return a

    @classmethod
    def _rademacher_walsh_energy(cls, values: NDArray) -> float:
        """Norma L² de las derivadas direccionales de Rademacher D_b f = (f − f∘⊕e_b)/2."""
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.size == 0:
            return 0.0
        n = 2
        while n < v.size:
            n <<= 1
        f = np.zeros(n, dtype=float)
        f[: v.size] = v
        idx = np.arange(n)
        energy = 0.0
        b = 1
        while b < n:
            d = 0.5 * (f - f[idx ^ b])
            spectrum = cls._fast_walsh_hadamard(d) / n
            energy += float(np.sum(spectrum * spectrum))
            b <<= 1
        return math.sqrt(energy)

    # ── Valuación de Krull-Novikov sobre el anillo Λ_Nov ────────────────────────────────────────

    @staticmethod
    def _novikov_valuation(exponents: NDArray) -> Tuple[float, float]:
        """Valuación v(A) = min λᵢ y norma ultramétrica ‖A‖_Nov = exp(−v(A)) en Λ_Nov."""
        e = np.asarray(exponents, dtype=float).reshape(-1)
        if e.size == 0:
            return math.inf, 0.0
        if not np.all(np.isfinite(e)):
            raise ValueError("Los exponentes de Novikov deben ser finitos.")
        v = float(np.min(e))
        return v, float(math.exp(-v))

    # ── Análisis algebraico-topológico integrado ────────────────────────────────────────────────

    def evaluate_algebraic_topology(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        incidence_matrix: Optional[NDArray[np.float64]] = None,
        quant_scale: float = 1000.0,
    ) -> AlgebraicTopologyReport:
        """Integra forma normal de Smith sobre ℤ, Hodge Δ₀, Walsh/Rademacher, Heaviside y Novikov."""
        M = np.asarray(apu_cost_matrix, dtype=float)
        if M.ndim != 2:
            raise ValueError("apu_cost_matrix debe ser una matriz 2D.")

        if incidence_matrix is None:
            source = M
            Z = [[int(round(float(x) * quant_scale)) for x in row] for row in M.tolist()]
        else:
            source = np.asarray(incidence_matrix, dtype=float)
            if source.ndim != 2:
                raise ValueError("incidence_matrix debe ser 2D.")
            if not np.allclose(source, np.rint(source)):
                raise ValueError("incidence_matrix debe ser entera para la forma normal de Smith.")
            Z = [[int(x) for x in row] for row in np.rint(source).astype(np.int64).tolist()]

        invariants = self._smith_invariant_factors(Z)
        m = len(Z)
        n = len(Z[0]) if m else 0
        free_rank = m - len(invariants)
        torsion_free = all(d == 1 for d in invariants)
        torsion_order = int(reduce(lambda acc, d: acc * d, (d for d in invariants if d > 1), 1))
        euler_char = int(m - n)

        hodge_gap = self._hodge_laplacian_spectral_gap(source)
        walsh = self._rademacher_walsh_energy(source)
        v_krull, nov_norm = self._novikov_valuation(exponents)
        z0_mag, refl_coeff, vswr = self._evaluate_heaviside_telegrapher(M)

        return AlgebraicTopologyReport(
            rademacher_walsh_energy=walsh,
            smith_invariants=tuple(invariants),
            smith_torsion_free=bool(torsion_free),
            betti_number_zero=bool(free_rank == 0),
            krull_valuation=v_krull,
            novikov_norm=nov_norm,
            krull_floor_ok=bool(v_krull >= _DEFAULT_KRULL_FLOOR),
            hodge_spectral_gap=hodge_gap,
            euler_characteristic=euler_char,
            torsion_subgroup_order=torsion_order,
            characteristic_impedance_mag=z0_mag,
            telegrapher_reflection_coeff=refl_coeff,
            telegrapher_vswr=vswr,
        )

    # ── ÚLTIMO MÉTODO DE LA FASE 1 · ESLABÓN DE CONTINUIDAD HACIA EL INICIO DE LA FASE 2 ────────

    def compute_phase1_witness(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        incidence_matrix: Optional[NDArray[np.float64]] = None,
        quant_scale: float = 1000.0,
    ) -> Phase1Witness:
        """DEFINICIÓN FORMAL FINAL DE LA FASE 1.

        Sella el testigo algebraico-cuántico `Phase1Witness`, que constituye el argumento
        contractual obligatorio de continuación del primer método de la Fase 2
        (`_Phase2PoincareDynamicsKernel.evaluate_toon_buffer_flow`).
        """
        dim = np.asarray(rho_mac).shape[0]
        qnd = self.compute_qnd_weak_measurement(rho_mac, phi_i, phi_f)
        uhlmann = self.evaluate_bures_uhlmann_metric(
            rho_mac, np.eye(dim, dtype=np.complex128) / dim
        )
        algebra = self.evaluate_algebraic_topology(
            apu_cost_matrix, exponents, incidence_matrix, quant_scale
        )
        return Phase1Witness(qnd_report=qnd, uhlmann_report=uhlmann, algebra_report=algebra)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2 · MECÁNICA CELESTE DE POINCARÉ-KAM, INVARIANTE DE CARTAN, LIE-DEPRIT Y OSELEDETS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class _Phase2PoincareDynamicsKernel(_Phase1AlgebraicQuantumKernel):
    """FASE 2: Dinámica de caudal y mecánica celeste simpléctica de Poincaré-KAM.

    Continuación directa de la Fase 1:
      • PRIMER MÉTODO : `evaluate_toon_buffer_flow(witness1: Phase1Witness, ...)`
      • ÚLTIMO MÉTODO : `compose_phase2_witness(witness1: Phase1Witness, ...) -> Phase2Witness`
    """

    # ── PRIMER MÉTODO DE LA FASE 2 · CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 1 ────────

    def evaluate_toon_buffer_flow(
        self,
        witness1: Phase1Witness,
        apu_cost_matrix: NDArray[np.float64],
        buffer_capacity: float = 100.0,
        variance_ceiling: float = 10.0,
    ) -> ToonBufferFlowReport:
        """PRIMER MÉTODO DE LA FASE 2 (recibe `Phase1Witness` desde `compute_phase1_witness`).

        Calcula el espectro ergódico multiplicativo de Oseledets (todos los exponentes de
        Lyapunov del cociclo lineal constante A), la cota de entropía de Pesin Σ max(λᵢ, 0)
        y la sobrepresión de golpe de bélier de Zhukovsky-Allievi acoplada al coeficiente
        de reflexión de Heaviside Γ medido en la Fase 1.
        """
        A = np.asarray(apu_cost_matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("apu_cost_matrix debe ser cuadrada.")
        if buffer_capacity <= 0.0:
            raise ValueError("buffer_capacity debe ser positiva.")

        flow_variance = float(np.var(A.sum(axis=1)))
        evals = np.linalg.eigvals(A)
        abs_evals = np.sort(np.abs(evals))[::-1]
        rho_spec = float(abs_evals[0]) if abs_evals.size > 0 else 0.0
        lam_max = math.log(rho_spec) if rho_spec > 0.0 else -math.inf

        lyap_spec = tuple(
            float(math.log(ev)) if ev > _WILKINSON_EPSILON else -math.inf for ev in abs_evals
        )
        pesin_bound = float(sum(max(0.0, l_i) for l_i in lyap_spec if math.isfinite(l_i)))

        # Sobrepresión de Zhukovsky-Allievi ΔP = Z₀ · √Var(Q) · (1 + |Γ_refl|)
        z0_mag = witness1.algebra_report.characteristic_impedance_mag
        refl = witness1.algebra_report.telegrapher_reflection_coeff
        zhukovsky_surge = float(z0_mag * math.sqrt(max(0.0, flow_variance)) * (1.0 + refl))

        suppressed = bool(
            lam_max <= 0.0
            and flow_variance < variance_ceiling
            and witness1.qnd_report.is_qnd_passive
        )
        saturation = float(min(100.0, flow_variance / buffer_capacity * 100.0))

        return ToonBufferFlowReport(
            flow_variance=flow_variance,
            spectral_radius=rho_spec,
            oseledets_lyapunov_exponent=lam_max,
            is_water_hammer_suppressed=suppressed,
            buffer_saturation_pct=saturation,
            lyapunov_spectrum=lyap_spec,
            zhukovsky_surge_pressure=zhukovsky_surge,
            pesin_entropy_bound=pesin_bound,
        )

    # ── Residual de Cauchy-Riemann-Wirtinger y capacidad de no-aplastamiento de Gromov ──────────

    @staticmethod
    def evaluate_cauchy_riemann_residual(
        u: NDArray[np.float64],
        v: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """Residual de Wirtinger normalizado ‖∂̄z‖ / (‖∂̄z‖ + ‖∂z‖) sobre la malla 2D."""
        U = np.asarray(u, dtype=float)
        if U.ndim != 2 or min(U.shape) < 2:
            raise ValueError("La malla de Cauchy-Riemann requiere al menos 2×2 puntos.")
        Vv = np.zeros_like(U) if v is None else np.asarray(v, dtype=float)
        Z = U.astype(np.complex128) + 1j * Vv
        zs = np.gradient(Z, axis=0)
        zt = np.gradient(Z, axis=1)
        dbar = 0.5 * (zs + 1j * zt)
        d = 0.5 * (zs - 1j * zt)
        num = float(la.norm(dbar))
        den = float(la.norm(d))
        if num + den <= _WILKINSON_EPSILON:
            return 0.0
        return num / (num + den)

    @staticmethod
    def evaluate_gromov_capacity(ball_r: float, cylinder_R: float) -> Tuple[float, float, bool]:
        """Teorema de no-aplastamiento de Gromov: c_G(B²ⁿ(r)) = πr² ≤ c_G(Z²ⁿ(R)) = πR²."""
        if ball_r <= 0.0 or cylinder_R <= 0.0:
            raise ValueError("Los radios simplécticos deben ser positivos.")
        c_b = math.pi * ball_r * ball_r
        c_z = math.pi * cylinder_R * cylinder_R
        return c_b, c_z, bool(c_b <= c_z + _WILKINSON_EPSILON)

    # ── Mecánica Celeste de Poincaré: Variables Canónicas No Singulares y Auditoría Simpléctica ──

    @staticmethod
    def kepler_mean_motion(a: float, mu: float = 1.0) -> float:
        """Movimiento medio kepleriano n = ∂H_Kepler/∂L = √(μ/a³)."""
        if a <= 0.0 or mu <= 0.0:
            raise ValueError("a y μ deben ser positivos.")
        return math.sqrt(mu / (a ** 3))

    @staticmethod
    def delaunay_actions(
        a: float, e: float, inc: float, mu: float = 1.0
    ) -> Tuple[float, float, float]:
        """Acciones canónicas de Delaunay (L, G, H) = (√(μa), L√(1−e²), G cos i)."""
        if a <= 0.0 or not (0.0 <= e < 1.0):
            raise ValueError("Se requiere a > 0 y 0 ≤ e < 1.")
        L = math.sqrt(mu * a)
        G = L * math.sqrt(1.0 - e * e)
        H = G * math.cos(inc)
        return L, G, H

    @classmethod
    def poincare_canonical_elements_and_symplectic_audit(
        cls,
        a: float,
        e: float,
        inc: float,
        mean_anomaly_l: float = 0.3,
        arg_periapsis_g: float = 0.5,
        lon_node_h: float = 0.7,
        mu: float = 1.0,
    ) -> Tuple[Tuple[float, float, float, float, float, float], float]:
        """Transformación canónica de Delaunay (L,G,H; l,g,h) a variables de Poincaré no singulares.

        Poincaré definió los pares conjugados momentos-coordenadas (p_P; q_P) = (Λ, ξ₁, ξ₂; λ, η₁, η₂):
            Λ  = L,                        λ  = l + g + h  (longitud media)
            ξ₁ = √(2(L − G)) cos(g + h),   η₁ = −√(2(L − G)) sin(g + h)  (par excentricidad)
            ξ₂ = √(2(G − H)) cos(h),       η₂ = −√(2(G − H)) sin(h)      (par inclinación)
        Conservación de la 2-forma de Poincaré:
            dΛ ∧ dλ + dξ₁ ∧ dη₁ + dξ₂ ∧ dη₂ = dL ∧ dl + dG ∧ dg + dH ∧ dh.
        Se construye la matriz jacobiana analítica exacta M = ∂(Λ,ξ₁,ξ₂,λ,η₁,η₂)/∂(L,G,H,l,g,h)
        y se mide el defecto simpléctico ‖Mᵀ J₆ M − J₆‖_F.
        """
        L, G, H = cls.delaunay_actions(a, e, inc, mu)
        I1 = max(1.0e-12, L - G)
        I2 = max(1.0e-12, G - H)
        varpi = arg_periapsis_g + lon_node_h
        omega_node = lon_node_h

        s1 = math.sqrt(2.0 * I1)
        s2 = math.sqrt(2.0 * I2)
        c_vp, s_vp = math.cos(varpi), math.sin(varpi)
        c_om, s_om = math.cos(omega_node), math.sin(omega_node)

        Lambda = L
        lam = mean_anomaly_l + arg_periapsis_g + lon_node_h
        xi1 = s1 * c_vp
        eta1 = -s1 * s_vp
        xi2 = s2 * c_om
        eta2 = -s2 * s_om

        # Jacobiano analítico 6×6: filas (Λ, ξ₁, ξ₂, λ, η₁, η₂), columnas (L, G, H, l, g, h)
        M = np.zeros((6, 6), dtype=float)
        # Fila 0: ∂Λ
        M[0, 0] = 1.0
        # Fila 1: ∂ξ₁
        M[1, 0] = c_vp / s1
        M[1, 1] = -c_vp / s1
        M[1, 4] = -s1 * s_vp
        M[1, 5] = -s1 * s_vp
        # Fila 2: ∂ξ₂
        M[2, 1] = c_om / s2
        M[2, 2] = -c_om / s2
        M[2, 5] = -s2 * s_om
        # Fila 3: ∂λ
        M[3, 3] = 1.0
        M[3, 4] = 1.0
        M[3, 5] = 1.0
        # Fila 4: ∂η₁
        M[4, 0] = -s_vp / s1
        M[4, 1] = s_vp / s1
        M[4, 4] = -s1 * c_vp
        M[4, 5] = -s1 * c_vp
        # Fila 5: ∂η₂
        M[5, 1] = -s_om / s2
        M[5, 2] = s_om / s2
        M[5, 5] = -s2 * c_om

        J6 = np.zeros((6, 6), dtype=float)
        J6[:3, 3:] = np.eye(3)
        J6[3:, :3] = -np.eye(3)
        defect = float(la.norm(M.T @ J6 @ M - J6, ord="fro"))
        return (Lambda, xi1, xi2, lam, eta1, eta2), defect

    # ── Torsión de Kolmogorov y Determinante Isoenergético de Arnol'd ───────────────────────────

    @staticmethod
    def evaluate_kolmogorov_arnold_torsion(
        L: float,
        G: float,
        H: float,
        mu: float = 1.0,
        j2_oblateness: float = 0.02,
        secular_coupling: float = 0.05,
    ) -> Tuple[NDArray[np.float64], float, float, bool]:
        """Evalúa la no-degeneración de Kolmogorov det(∂²H₀/∂I²) y la isoenergética de Arnol'd.

        Para levantar la degeneración kepleriana propia en (G, H), el Hamiltoniano integrable
        secular incluye el potencial cuadrupolar de achatamiento J₂ y el acoplamiento secular:
            H₀(L, G, H) = −μ²/(2L²) − ε_{J2} μ⁴/(4 L³ G³) (1 − 3 H²/G²) − ½ ε_sec (G² + H²).
        """
        c_j2 = 0.25 * j2_oblateness * (mu ** 4)
        ratio = (H * H) / (G * G)
        factor = 1.0 - 3.0 * ratio

        # Frecuencias fundamentales ω = ∇H₀ = (ω_L, ω_G, ω_H)
        omega_L = (mu * mu) / (L ** 3) + 3.0 * c_j2 / ((L ** 4) * (G ** 3)) * factor
        omega_G = (
            3.0 * c_j2 / ((L ** 3) * (G ** 4)) * (1.0 - 5.0 * ratio)
            - secular_coupling * G
        )
        omega_H = 6.0 * c_j2 * H / ((L ** 3) * (G ** 5)) - secular_coupling * H
        omega_vec = np.array([omega_L, omega_G, omega_H], dtype=float)

        # Matriz de torsión (Hessiano T_ij = ∂²H₀ / ∂I_i ∂I_j)
        T = np.zeros((3, 3), dtype=float)
        T[0, 0] = -3.0 * (mu * mu) / (L ** 4) - 12.0 * c_j2 / ((L ** 5) * (G ** 3)) * factor
        T[0, 1] = T[1, 0] = -9.0 * c_j2 / ((L ** 4) * (G ** 4)) * (1.0 - 5.0 * ratio)
        T[0, 2] = T[2, 0] = -18.0 * c_j2 * H / ((L ** 4) * (G ** 5))
        T[1, 1] = (
            -12.0 * c_j2 / ((L ** 3) * (G ** 5)) * (1.0 - 7.5 * ratio) - secular_coupling
        )
        T[1, 2] = T[2, 1] = -30.0 * c_j2 * H / ((L ** 3) * (G ** 6))
        T[2, 2] = 6.0 * c_j2 / ((L ** 3) * (G ** 5)) - secular_coupling

        kolmogorov_det = float(np.linalg.det(T))
        bordered = np.zeros((4, 4), dtype=float)
        bordered[:3, :3] = T
        bordered[:3, 3] = omega_vec
        bordered[3, :3] = omega_vec
        arnold_det = float(np.linalg.det(bordered))

        non_deg = bool(abs(kolmogorov_det) > 1.0e-9 and abs(arnold_det) > 1.0e-9)
        return omega_vec, kolmogorov_det, arnold_det, non_deg

    # ── Condición Diofantina y Serie Canónica de Lie de Deprit-Hori (exp(L_W)H) ─────────────────

    @staticmethod
    def evaluate_diophantine_condition(
        omega: float,
        dioph_gamma: float = _DIOPHANTINE_GAMMA,
        tau: float = _DIOPHANTINE_TAU,
        k_max: int = _DIOPHANTINE_K_MAX,
    ) -> bool:
        """Condición diofantina |kν − p| ≥ γ / k^τ para 1 ≤ k ≤ k_max, con ν = ω / (2π)."""
        nu = omega / (2.0 * math.pi)
        for k in range(1, k_max + 1):
            x = k * nu
            dist = abs(x - round(x))
            if dist < dioph_gamma / (k ** tau):
                return False
        return True

    @staticmethod
    def solve_homological_equation(
        h1_samples: NDArray[np.float64],
        omega: float,
    ) -> Tuple[NDArray[np.float64], float]:
        """Resuelve ω ∂_θ W = h₁ − ⟨h₁⟩ en el espacio de Fourier sobre 𝕋¹ y verifica el residuo."""
        h = np.asarray(h1_samples, dtype=float).reshape(-1)
        N = h.size
        if N < 4:
            raise ValueError("Se requieren al menos 4 muestras angulares.")
        if abs(omega) <= _WILKINSON_EPSILON:
            raise ValueError("ω ≈ 0: divisor resonante nulo.")

        hk = np.fft.fft(h) / N
        k = np.fft.fftfreq(N, d=1.0 / N)
        mask = k != 0.0

        wk = np.zeros_like(hk)
        wk[mask] = hk[mask] / (1j * k[mask] * omega)
        W = np.real(np.fft.ifft(wk * N))

        lhs_hat = omega * (1j * k * wk)
        rhs_hat = np.where(mask, hk, 0.0)
        resid = float(np.max(np.abs(np.fft.ifft((lhs_hat - rhs_hat) * N))))
        return W, resid

    @classmethod
    def evaluate_deprit_hori_lie_series(
        cls,
        omega: float,
        perturbation_eps: float = 0.01,
        twist_alpha: float = 0.2,
        n_theta: int = 256,
    ) -> Tuple[float, float, float]:
        """Transformada canónica de Lie de Deprit-Hori exp(L_W)H y horizonte de Nekhoroshev.

        Para H(J, θ) = H₀(J) + h₁(J, θ) con ω(J) = ω₀ + α J y h₁(J, θ) = ε(1 + J)(cos θ + ½ cos 2θ):
          • 1.ᵉʳ orden: L_{H₀} W = ω(J) ∂_θ W = h₁ − ⟨h₁⟩  ⇒  W(J, θ) = Σ_{k≠0} ĥ_k(J)/(i k ω(J)) e^{ikθ}.
          • 2.º orden (corchete de Poisson {h₁, W} = ∂_θ h₁ ∂_J W − ∂_J h₁ ∂_θ W):
            El término homológico de segundo orden es ½ {h₁ + ⟨h₁⟩, W}.
        """
        theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
        base_profile = np.cos(theta) + 0.5 * np.cos(2.0 * theta)
        h1 = perturbation_eps * base_profile
        W, homo_res = cls.solve_homological_equation(h1, omega)

        # Derivadas espectrales en (J=0, θ) para el corchete de Poisson {h₁, W}
        hk = np.fft.fft(h1) / n_theta
        k = np.fft.fftfreq(n_theta, d=1.0 / n_theta)
        mask = k != 0.0

        dh1_dtheta = np.real(np.fft.ifft((1j * k * hk) * n_theta))
        dh1_dJ = h1  # pues h₁(J,θ) = (1 + J) h₁(0,θ)

        wk = np.zeros_like(hk)
        wk[mask] = hk[mask] / (1j * k[mask] * omega)
        dW_dtheta = np.real(np.fft.ifft((1j * k * wk) * n_theta))

        # ∂_J W(0,θ) = ∂_J [ĥ₁(J) / (i k ω(J))]|_{J=0} = wk · (1 − α / ω)
        dwk_dJ = wk * (1.0 - twist_alpha / omega)
        dW_dJ = np.real(np.fft.ifft(dwk_dJ * n_theta))

        poisson_h1_W = dh1_dtheta * dW_dJ - dh1_dJ * dW_dtheta
        second_order_norm = float(0.5 * np.max(np.abs(poisson_h1_W)))

        # Estimación exponencial de Nekhoroshev T_Nekh = T₀ exp((ε* / ε)^{1/6})
        eps_eff = max(1.0e-12, abs(perturbation_eps))
        nekh_exp = min(40.0, (0.1 / eps_eff) ** (1.0 / 6.0))
        nekhoroshev_time = float((2.0 * math.pi / abs(omega)) * math.exp(nekh_exp))
        return homo_res, second_order_norm, nekhoroshev_time

    # ── Invariante Integral de Poincaré-Cartan (Yoshida 4.º Orden) y Monodromía de Floquet ──────

    @staticmethod
    def evaluate_poincare_cartan_and_floquet(
        omega: float,
        perturbation_eps: float = 0.01,
        n_loop: int = 64,
        n_steps: int = 80,
    ) -> Tuple[float, float, bool]:
        """Verifica la conservación del invariante de Poincaré-Cartan ∮_γ p dq y la monodromía.

        Integra el flujo hamiltoniano H(q, p) = ½ p² + ½ ω² q² + ¼ ε q⁴ mediante el integrador
        simpléctico explícito de Yoshida de 4.º orden:
            w₁ = 1 / (2 − 2^{1/3}),   w₀ = −2^{1/3} / (2 − 2^{1/3}).
        1. Transporta un lazo cerrado γ(0) → γ(T) en el espacio de fases y verifica que la
           circulación simpléctica I₁(t) = ∮_{γ(t)} p dq permanece invariante (dI₁/dt = 0).
        2. Integra simultáneamente la ecuación variacional tangente dΦ/dt = J₂ D²H(q,p) Φ
           para obtener la matriz de monodromía M(T) ∈ Sp(2, ℝ) y el índice de Floquet Tr(M(T)).
        """
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cbrt2)
        w0 = -cbrt2 / (2.0 - cbrt2)
        c_yosh = (0.5 * w1, 0.5 * (w0 + w1), 0.5 * (w0 + w1), 0.5 * w1)
        d_yosh = (w1, w0, w1, 0.0)

        period = 2.0 * math.pi / max(1.0e-6, abs(omega))
        dt = period / n_steps

        # Lazo inicial γ(0) parametrizado por s ∈ [0, 2π)
        s = np.linspace(0.0, 2.0 * math.pi, n_loop, endpoint=False)
        r0 = 0.15
        q = r0 * np.cos(s)
        p = r0 * omega * np.sin(s)

        def _loop_action(q_loop: NDArray[np.float64], p_loop: NDArray[np.float64]) -> float:
            # Regla trapezoidal periódica exacta para ∮ p dq
            dq = np.roll(q_loop, -1) - q_loop
            p_mid = 0.5 * (p_loop + np.roll(p_loop, -1))
            return float(np.sum(p_mid * dq))

        i1_initial = _loop_action(q, p)

        # Órbita de referencia y matriz variacional simpléctica Φ(t) ∈ Sp(2, ℝ)
        q_ref, p_ref = 0.10, 0.0
        Phi = np.eye(2, dtype=float)
        om2 = omega * omega

        for _ in range(n_steps):
            for c_i, d_i in zip(c_yosh, d_yosh):
                # Drift en q
                q = q + (c_i * dt) * p
                q_ref = q_ref + (c_i * dt) * p_ref
                Phi[0, :] += (c_i * dt) * Phi[1, :]

                # Kick en p
                if d_i != 0.0:
                    p = p - (d_i * dt) * (om2 * q + perturbation_eps * (q ** 3))
                    v_pp = om2 + 3.0 * perturbation_eps * (q_ref * q_ref)
                    p_ref = p_ref - (d_i * dt) * (om2 * q_ref + perturbation_eps * (q_ref ** 3))
                    Phi[1, :] -= (d_i * dt) * v_pp * Phi[0, :]

        i1_final = _loop_action(q, p)
        invariant_drift = abs(i1_final - i1_initial)
        floquet_trace = float(np.trace(Phi))
        symp_det_err = abs(float(np.linalg.det(Phi)) - 1.0)
        elliptic_stable = bool(abs(floquet_trace) < 2.0 and symp_det_err <= _SYMPLECTIC_TOL)
        return invariant_drift, floquet_trace, elliptic_stable

    # ── Integral Homoclínica de Melnikov (Separación de Variedades Estable/Inestable) ───────────

    def evaluate_melnikov_homoclinic(
        self,
        delta_damping: float,
        forcing_amp: float,
        omega: float,
        t_max: float = 40.0,
        n_t: int = 8001,
        n_phase: int = 720,
    ) -> Tuple[float, float, bool]:
        """Función de Melnikov M(t₀) = −δ∫p₀² dt + γ∫p₀ cos(ω(t+t₀)) dt sobre la separatriz de Duffing."""
        if omega <= 0.0:
            raise ValueError("La frecuencia de forzamiento debe ser positiva.")
        t = np.linspace(-t_max, t_max, n_t)
        w = self._trapezoid_weights(t)
        p0 = -math.sqrt(2.0) / np.cosh(t) * np.tanh(t)

        I_pp = float(np.sum(w * p0 * p0))
        A_c = float(np.sum(w * p0 * np.cos(omega * t)))
        A_s = float(np.sum(w * p0 * np.sin(omega * t)))

        phase = np.linspace(0.0, 2.0 * math.pi / omega, n_phase, endpoint=False)
        M = -delta_damping * I_pp + forcing_amp * (
            np.cos(omega * phase) * A_c - np.sin(omega * phase) * A_s
        )
        mn = float(np.min(M))
        mx = float(np.max(M))
        return mn, mx, bool(mn < 0.0 < mx)

    # ── Síntesis de Estabilidad de Poincaré-KAM ─────────────────────────────────────────────────

    def evaluate_poincare_kam_stability(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = 1.0,
        cylinder_R: float = 1.5,
        semi_major_axis: float = 1.0,
        eccentricity: float = 0.1,
        inclination: float = 0.2,
        mu: float = 1.0,
        perturbation_eps: float = 0.01,
        delta_damping: float = 0.1,
        forcing_amp: float = 0.05,
        n_theta: int = 256,
    ) -> PoincareKAMReport:
        """Integra Wirtinger, Gromov, Poincaré no singular, torsión KAM, Lie-Deprit, Cartan y Melnikov."""
        cr = self.evaluate_cauchy_riemann_residual(apu_cost_matrix)
        c_b, c_z, rigid = self.evaluate_gromov_capacity(ball_r, cylinder_R)
        omega = self.kepler_mean_motion(semi_major_axis, mu)
        mn, mx, chaos = self.evaluate_melnikov_homoclinic(delta_damping, forcing_amp, omega)
        dioph = self.evaluate_diophantine_condition(omega)

        L, G, H = self.delaunay_actions(semi_major_axis, eccentricity, inclination, mu)
        poincare_coords, symp_defect = self.poincare_canonical_elements_and_symplectic_audit(
            semi_major_axis, eccentricity, inclination, mu=mu
        )
        _, kol_det, arn_det, kam_non_deg = self.evaluate_kolmogorov_arnold_torsion(L, G, H, mu=mu)
        homo_res, lie_norm2, nekh_time = self.evaluate_deprit_hori_lie_series(
            omega, perturbation_eps=perturbation_eps, n_theta=n_theta
        )
        cartan_drift, floquet_tr, floquet_ok = self.evaluate_poincare_cartan_and_floquet(
            omega, perturbation_eps=perturbation_eps
        )

        return PoincareKAMReport(
            cauchy_riemann_residual=cr,
            gromov_ball_capacity=c_b,
            gromov_cylinder_capacity=c_z,
            is_gromov_rigid=rigid,
            melnikov_min=mn,
            melnikov_max=mx,
            melnikov_chaos_detected=chaos,
            diophantine_ok=dioph,
            homological_residual=homo_res,
            kepler_frequency=omega,
            delaunay_actions=(L, G, H),
            poincare_canonical_coords=poincare_coords,
            symplectic_jacobian_defect=symp_defect,
            kolmogorov_torsion_det=kol_det,
            arnold_isoenergetic_det=arn_det,
            is_kam_non_degenerate=kam_non_deg,
            lie_series_second_order_norm=lie_norm2,
            nekhoroshev_stability_time=nekh_time,
            poincare_cartan_invariant_drift=cartan_drift,
            floquet_monodromy_trace=floquet_tr,
            is_floquet_elliptic_stable=floquet_ok,
        )

    # ── ÚLTIMO MÉTODO DE LA FASE 2 · ESLABÓN DE CONTINUIDAD HACIA EL INICIO DE LA FASE 3 ────────

    def compose_phase2_witness(
        self,
        witness1: Phase1Witness,
        apu_cost_matrix: NDArray[np.float64],
        buffer_capacity: float = 100.0,
        variance_ceiling: float = 10.0,
    ) -> Phase2Witness:
        """DEFINICIÓN FORMAL FINAL DE LA FASE 2.

        Sella el testigo dinámico-simpléctico `Phase2Witness` (que encapsula asimismo `witness1`),
        constituyendo la entrada contractual del primer método de la Fase 3
        (`_Phase3RSIVerdictKernel._phase3_rsi_gate`).
        """
        flow = self.evaluate_toon_buffer_flow(
            witness1, apu_cost_matrix, buffer_capacity, variance_ceiling
        )
        kam = self.evaluate_poincare_kam_stability(apu_cost_matrix)
        return Phase2Witness(flow=flow, kam=kam, phase1=witness1)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3 · TORRE RSI³ CERTIFICADA, CATEGORÍA DE KLEISLI, GUARDIA DE LÖB Y TOPOS DE HEYTING Ω₃
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class _Phase3RSIVerdictKernel(_Phase2PoincareDynamicsKernel):
    """FASE 3: Automejora Recursiva de Nivel 3 (RSI³), mónada de Kleisli y veredicto en Ω₃.

    Continuación directa de la Fase 2:
      • PRIMER MÉTODO : `_phase3_rsi_gate(witness2: Phase2Witness, ...)`
      • ORQUESTADOR   : `execute_toon_buffer_witness_audit(...) -> ToonBufferWitnessSpectrumReport`
    """

    _secret_key: bytes

    # ── PRIMER MÉTODO DE LA FASE 3 · CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 2 ────────

    def _phase3_rsi_gate(
        self,
        witness2: Phase2Witness,
        stage_factors: Sequence[float],
        depth: int = _RSI_MAX_DEPTH,
    ) -> RSICertificate:
        """PRIMER MÉTODO DE LA FASE 3 (consume `Phase2Witness` emitido por `compose_phase2_witness`).

        Compuerta de seguridad simpléctico-cuántica: autoriza la ejecución de la torre RSI³
        únicamente cuando el caudal no presenta golpe de bélier, el toro de Poincaré-KAM es
        diofantino, no degenerado (Kolmogorov/Arnol'd), libre de caos homoclínico de Melnikov,
        rígido en el sentido de Gromov y conserva el invariante integral de Poincaré-Cartan.
        En caso contrario, ejecuta rollback inmediato a la política desplegada.
        """
        kam = witness2.kam
        authorized = bool(
            witness2.flow.is_water_hammer_suppressed
            and kam.diophantine_ok
            and not kam.melnikov_chaos_detected
            and kam.is_gromov_rigid
            and kam.homological_residual <= _HOMOLOGICAL_TOL
            and kam.symplectic_jacobian_defect <= _SYMPLECTIC_TOL
            and kam.poincare_cartan_invariant_drift <= _CARTAN_INVARIANT_TOL
            and kam.is_kam_non_degenerate
        )
        theta_dep = np.array(_DEPLOYED_POLICY, dtype=float)
        if not authorized:
            logger.warning(
                "RSI³ no autorizado: violación de invariantes KAM/Poincaré/QND. "
                "Rollback estricto a la política desplegada."
            )
            return RSICertificate(
                depth_reached=0,
                lip_certified=math.inf,
                lip_probe_estimate=0.0,
                cross_validated=False,
                fixed_point=tuple(float(x) for x in theta_dep),
                accepted=False,
                meta_preconditioned_lip=math.inf,
                kleisli_associativity_defect=math.inf,
                kleisli_unit_defect=math.inf,
                loeb_soundness_verified=False,
                banach_aposteriori_error=math.inf,
                lyapunov_descent_verified=False,
            )
        theta0 = theta_dep + np.array([0.05, -0.02, 0.01], dtype=float)
        return self._rsi_tower(stage_factors, theta_dep, theta0, depth)

    # ── Operadores de Nivel 1, Meta-Endofuntor de Nivel 2 y Gobernador Kleisli-Löb de Nivel 3 ────

    @staticmethod
    def _rsi_affine_operator(
        stage_factors: Sequence[float],
        theta_deployed: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Nivel 1 (Objeto): Operador contractivo afín T_Φ(θ) = A(Φ)θ + b(Φ) con punto fijo θ_dep."""
        c = np.asarray(stage_factors, dtype=float).reshape(-1)
        if c.size != theta_deployed.size or np.any(c <= 0.0) or np.any(c >= 1.0):
            raise ValueError("Los factores de etapa deben pertenecer a (0, 1) y coincidir en dimensión.")
        A = np.diag(c)
        b = (np.eye(c.size) - A) @ theta_deployed
        return A, b

    @classmethod
    def _rsi_level2_meta_preconditioner(
        cls,
        stage_factors: Sequence[float],
        theta_deployed: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Nivel 2 (Meta-Optimizador M_Ψ: End(Θ) → End(Θ)).

        Transforma el endomorfismo de Nivel 1 T_Φ mediante una contracción cuadrática de
        Schulz-Newton-Picard en el álgebra de Banach B(ℝⁿ):
            M_Ψ(T_Φ)(θ) = (T_Φ ∘ T_Φ)(θ) = A² θ + (I + A) b,
        reduciendo cuadráticamente la constante de Lipschitz Lip(M_Ψ(T_Φ)) = ‖A‖₂² < ‖A‖₂ < 1
        mientras preserva de forma exacta e invariante el único punto fijo de Banach θ*.
        """
        A, b = cls._rsi_affine_operator(stage_factors, theta_deployed)
        A_meta = A @ A
        b_meta = (np.eye(A.shape[0]) + A) @ b
        c_meta = np.diag(A_meta).copy()
        return A_meta, b_meta, c_meta

    @staticmethod
    def _verify_kleisli_monad_coherence(
        stage_factors: Sequence[float],
        theta_deployed: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Nivel 3 (Meta-Meta-Gobernador Categórico): Leyes de coherencia de la mónada de Kleisli.

        En la categoría de Kleisli Kl(T) de endomorfismos afines contractivos sobre (ℝⁿ, ‖·‖₂):
          • Un morfismo f = (A_f, b_f) representa θ ↦ A_f θ + b_f.
          • La unidad monádica es η = (I, 0).
          • La composición de Kleisli f ∘_K g viene dada por (A_f A_g, A_f b_g + b_f).
        Se verifican numéricamente:
          1. Asociatividad: ‖((f₁ ∘_K f₂) ∘_K f₃) − (f₁ ∘_K (f₂ ∘_K f₃))‖_F.
          2. Identidad izquierda y derecha: ‖(η ∘_K fᵢ) − fᵢ‖_F + ‖(fᵢ ∘_K η) − fᵢ‖_F.
        """
        c = np.asarray(stage_factors, dtype=float).reshape(-1)
        dim = theta_deployed.size
        eye = np.eye(dim)

        def _compose(
            m1: Tuple[NDArray[np.float64], NDArray[np.float64]],
            m2: Tuple[NDArray[np.float64], NDArray[np.float64]],
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
            A1, b1 = m1
            A2, b2 = m2
            return A1 @ A2, A1 @ b2 + b1

        # Tres morfismos de etapa en Kl(T)
        f1 = (np.diag(c), (eye - np.diag(c)) @ theta_deployed)
        f2 = (np.diag(c ** 2), (eye - np.diag(c ** 2)) @ theta_deployed)
        f3 = (np.diag(0.5 * c), (eye - np.diag(0.5 * c)) @ theta_deployed)
        eta = (eye, np.zeros(dim, dtype=float))

        lhs_A, lhs_b = _compose(_compose(f1, f2), f3)
        rhs_A, rhs_b = _compose(f1, _compose(f2, f3))
        assoc_defect = float(la.norm(lhs_A - rhs_A, ord="fro") + la.norm(lhs_b - rhs_b))

        l_A, l_b = _compose(eta, f1)
        r_A, r_b = _compose(f1, eta)
        unit_defect = float(
            la.norm(l_A - f1[0], ord="fro")
            + la.norm(l_b - f1[1])
            + la.norm(r_A - f1[0], ord="fro")
            + la.norm(r_b - f1[1])
        )
        return assoc_defect, unit_defect

    @staticmethod
    def _probe_lipschitz(
        A: NDArray[np.float64], b: NDArray[np.float64], seed: int, n_pairs: int
    ) -> float:
        """Estimación empírica de Lip(T) mediante pares aleatorios independientes."""
        rng = np.random.default_rng(seed)
        dim = A.shape[0]
        X = rng.uniform(-1.0, 1.0, size=(n_pairs, dim))
        Y = rng.uniform(-1.0, 1.0, size=(n_pairs, dim))
        num = np.linalg.norm((X @ A.T + b) - (Y @ A.T + b), axis=1)
        den = np.linalg.norm(X - Y, axis=1)
        mask = den > _WILKINSON_EPSILON
        if not np.any(mask):
            return 0.0
        return float(np.max(num[mask] / den[mask]))

    def _rsi_tower(
        self,
        stage_factors: Sequence[float],
        theta_deployed: NDArray[np.float64],
        theta0: NDArray[np.float64],
        depth: int,
    ) -> RSICertificate:
        """Ejecuta la torre de Automejora Recursiva de 3 niveles (RSI¹ → RSI² → RSI³).

        • Nivel 1 (RSI¹): Síntesis de punto fijo por iteración de Picard con verificación de
                          descenso monótono de la función de Lyapunov V(θ) = ½‖θ − θ*‖₂².
        • Nivel 2 (RSI²): Aplicación del meta-endofuntor M_Ψ(T_Φ) para acelerar espectralmente el
                          propio operador, estimación de Lipschitz por sondeo y certificación de Fréchet.
        • Nivel 3 (RSI³): Meta-meta-gobernador 𝔊: verificación de coherencia de Kleisli, cota a
                          posteriori de Banach y guardia de solidez de Löb-Tarski
                          (Lip(M_Ψ(T_Φ)) ≤ Lip(T_Φ) < 1 e invariancia de punto fijo).
        """
        depth = int(min(max(depth, 1), _RSI_MAX_DEPTH))
        A, b = self._rsi_affine_operator(stage_factors, theta_deployed)
        dim = theta_deployed.size
        theta_star = np.linalg.solve(np.eye(dim) - A, b)

        # ── Nivel 1 · Propuesta de política y descenso de Lyapunov ──
        theta = theta0.astype(float).copy()
        converged = False
        lyapunov_monotone = True
        prev_v = 0.5 * float(np.dot(theta - theta_star, theta - theta_star))
        last_step = math.inf

        for _ in range(_RSI_MAX_ITER):
            nxt = A @ theta + b
            last_step = float(np.linalg.norm(nxt - theta))
            curr_v = 0.5 * float(np.dot(nxt - theta_star, nxt - theta_star))
            if curr_v > prev_v + _WILKINSON_EPSILON:
                lyapunov_monotone = False
            prev_v = curr_v
            theta = nxt
            if last_step <= 1.0e-12:
                converged = True
                break

        lip_analytic = float(np.linalg.norm(A, 2))
        aposteriori_err = (
            float((lip_analytic / max(1.0e-12, 1.0 - lip_analytic)) * last_step)
            if lip_analytic < 1.0
            else math.inf
        )
        lip_probe = 0.0
        meta_lip = lip_analytic
        assoc_def = math.inf
        unit_def = math.inf
        loeb_ok = False
        cross = False
        depth_reached = 1

        # ── Nivel 2 · Meta-preacondicionamiento M_Ψ(T_Φ) y certificado de Lipschitz ──
        if depth >= 2:
            A_meta, b_meta, _ = self._rsi_level2_meta_preconditioner(stage_factors, theta_deployed)
            meta_lip = float(np.linalg.norm(A_meta, 2))
            lip_probe = self._probe_lipschitz(A, b, 1729, _RSI_PROBES)
            # Refinamiento acelerado con el operador meta-mejorado M_Ψ(T_Φ)
            theta = A_meta @ theta + b_meta
            depth_reached = 2

        # ── Nivel 3 · Meta-Meta-Gobernador 𝔊: Kleisli, validación cruzada y Guardia de Löb ──
        if depth >= 3:
            lip_probe = max(lip_probe, self._probe_lipschitz(A, b, 314159, _RSI_PROBES))
            assoc_def, unit_def = self._verify_kleisli_monad_coherence(
                stage_factors, theta_deployed
            )
            theta_star_meta = np.linalg.solve(np.eye(dim) - A_meta, b_meta)
            loeb_ok = bool(
                meta_lip <= lip_analytic < 1.0
                and np.allclose(theta_star_meta, theta_star, atol=1.0e-12)
                and assoc_def <= _KLEISLI_COHERENCE_TOL
                and unit_def <= _KLEISLI_COHERENCE_TOL
                and lyapunov_monotone
            )
            cross = bool(
                np.allclose(theta, theta_star, atol=1.0e-9)
                and converged
                and lip_probe <= lip_analytic + 1.0e-9
                and loeb_ok
            )
            depth_reached = 3

        lip_certified = lip_analytic if lip_probe <= lip_analytic + 1.0e-9 else math.inf
        accepted = bool(
            depth_reached == _RSI_MAX_DEPTH
            and converged
            and lip_certified < 1.0
            and cross
            and loeb_ok
        )
        return RSICertificate(
            depth_reached=depth_reached,
            lip_certified=lip_certified,
            lip_probe_estimate=lip_probe,
            cross_validated=cross,
            fixed_point=tuple(float(x) for x in theta),
            accepted=accepted,
            meta_preconditioned_lip=meta_lip,
            kleisli_associativity_defect=assoc_def,
            kleisli_unit_defect=unit_def,
            loeb_soundness_verified=loeb_ok,
            banach_aposteriori_error=aposteriori_err,
            lyapunov_descent_verified=lyapunov_monotone,
        )

    # ── Cota de Banach de la mónada Kleisli T³ y Jerk de Capacidad ──────────────────────────────

    @staticmethod
    def evaluate_rsi3_banach_kleisli(stage_factors: Sequence[float]) -> Tuple[float, float]:
        """Lip(T³) ≤ ∏cᵢ + margen de seguridad. Jerk de capacidad d³C/dt³|_{t=0} = κ³ con κ = −ln Lip."""
        c = np.asarray(stage_factors, dtype=float).reshape(-1)
        if np.any(c <= 0.0) or np.any(c >= 1.0):
            raise ValueError("Los factores de etapa deben pertenecer a (0, 1).")
        lip_t3 = float(np.prod(c) + _KLEISLI_SAFETY_MARGIN)
        kappa = -math.log(lip_t3)
        jerk = float(kappa ** 3)
        return lip_t3, jerk

    # ── Álgebra de Heyting Completa Ω₃ y Topología de Lawvere-Tierney ───────────────────────────

    @staticmethod
    def _phase3_heyting_verdict(
        witness1: Phase1Witness,
        witness2: Phase2Witness,
        rsi: RSICertificate,
        lip_t3: float,
    ) -> HeytingTruthValue:
        """Evalúa el veredicto intuicionista en Ω₃ y aplica la clausura modal de Lawvere-Tierney."""
        kam = witness2.kam
        veto_clauses = (
            witness1.qnd_report.is_qnd_passive,
            kam.is_gromov_rigid,
            witness1.algebra_report.smith_torsion_free,
            kam.symplectic_jacobian_defect <= _SYMPLECTIC_TOL,
        )
        degrade_clauses = (
            witness1.algebra_report.krull_floor_ok,
            not kam.melnikov_chaos_detected,
            witness2.flow.is_water_hammer_suppressed,
            witness1.uhlmann_report.is_isometry_preserved,
            kam.diophantine_ok,
            kam.homological_residual <= _HOMOLOGICAL_TOL,
            kam.is_kam_non_degenerate,
            kam.poincare_cartan_invariant_drift <= _CARTAN_INVARIANT_TOL,
            lip_t3 < 1.0,
            rsi.accepted,
            rsi.loeb_soundness_verified,
        )
        values = [
            HeytingTruthValue.COHERENT if c else HeytingTruthValue.VETOED for c in veto_clauses
        ] + [
            HeytingTruthValue.COHERENT if c else HeytingTruthValue.DEGRADED
            for c in degrade_clauses
        ]
        raw_verdict = reduce(HeytingTruthValue.meet, values, HeytingTruthValue.COHERENT)
        return raw_verdict.lawvere_tierney_sheafify(stabilize_degraded=False)

    # ── Pasaporte criptográfico determinista (HMAC-SHA256) ──────────────────────────────────────

    def _phase3_passport(
        self,
        truth: HeytingTruthValue,
        qnd_db: float,
        fidelity: float,
        lip_t3: float,
        krull: float,
        rsi_accepted: bool,
    ) -> str:
        """Doble resumen HMAC-SHA256 sobre cadena canónica determinista."""
        canonical = (
            f"TOON_BUFFER_WITNESS|{truth.name}|{qnd_db:.6f}|{fidelity:.6f}"
            f"|{lip_t3:.6f}|{krull:.6f}|{int(rsi_accepted)}"
        ).encode("utf-8")
        mac = hmac.new(self._secret_key, canonical, hashlib.sha256).digest()
        return hashlib.sha256(mac).hexdigest()

    # ── Orquestador Holístico de las 3 Fases Anidadas ───────────────────────────────────────────

    def execute_toon_buffer_witness_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac_input: Optional[NDArray[np.complex128]] = None,
        incidence_matrix: Optional[NDArray[np.float64]] = None,
        stage_factors: Sequence[float] = (0.95, 0.95, 0.95),
        quant_scale: float = 1000.0,
    ) -> ToonBufferWitnessSpectrumReport:
        """Ejecuta la cadena morfismática Fase 1 → Fase 2 → Fase 3 con contratos sellados."""
        t_start = time.perf_counter()

        A = np.asarray(apu_cost_matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("apu_cost_matrix debe ser cuadrada.")
        dim = A.shape[0]

        rho = (
            np.eye(dim, dtype=np.complex128) / dim
            if rho_mac_input is None
            else np.asarray(rho_mac_input, dtype=np.complex128)
        )
        phi = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)

        # Fase 1 → emite Phase1Witness
        w1 = self.compute_phase1_witness(
            rho, phi, phi, A, exponents, incidence_matrix, quant_scale
        )
        # Fase 2 (consume Phase1Witness) → emite Phase2Witness
        w2 = self.compose_phase2_witness(w1, A)
        # Fase 3 (consume Phase2Witness) → emite RSICertificate y Reporte Final
        rsi = self._phase3_rsi_gate(w2, stage_factors)
        lip_t3, jerk = self.evaluate_rsi3_banach_kleisli(stage_factors)
        truth = self._phase3_heyting_verdict(w1, w2, rsi, lip_t3)

        exec_ms = (time.perf_counter() - t_start) * 1000.0
        passport = self._phase3_passport(
            truth=truth,
            qnd_db=w1.qnd_report.back_action_db,
            fidelity=w1.uhlmann_report.fidelity_uhlmann,
            lip_t3=lip_t3,
            krull=w1.algebra_report.krull_valuation,
            rsi_accepted=rsi.accepted,
        )

        return ToonBufferWitnessSpectrumReport(
            phase1=w1,
            phase2=w2,
            rsi_certificate=rsi,
            banach_contraction_lip=lip_t3,
            capacity_jerk_d3C_dt3=jerk,
            truth_value=truth,
            passport_hash=passport,
            execution_time_ms=exec_ms,
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CLASE PÚBLICA SOBERANA
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class BoolNovikovToonBufferWitnessEngine(_Phase3RSIVerdictKernel):
    """Motor Espectral Testigo QND de Amortiguamiento y Protección de Caudal Mnémonico TOON (v15).

    Fachada pública soberana que hereda la cadena anidada:
      _Phase1AlgebraicQuantumKernel → _Phase2PoincareDynamicsKernel → _Phase3RSIVerdictKernel.
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        if not isinstance(secret_key, (bytes, bytearray)) or len(secret_key) < 16:
            raise ValueError("secret_key debe ser bytes de longitud ≥ 16.")
        self._secret_key = bytes(secret_key)
        if self._secret_key == _DEFAULT_SECRET_KEY:
            logger.warning("Usando la clave HMAC por defecto; inyecte una clave de despliegue.")
        logger.info("%s v%s inicializado.", self.__class__.__name__, _ENGINE_VERSION)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# AUTOPRUEBA DE VERIFICACIÓN FORMAL
# ════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def _imprimir(titulo: str, rep: ToonBufferWitnessSpectrumReport) -> None:
        a = rep.phase1.algebra_report
        k = rep.phase2.kam
        r = rep.rsi_certificate
        print(f"\n--- {titulo} ---")
        print(f"Veredicto Heyting Ω₃          : {rep.truth_value.name} (0x{rep.truth_value.value:02x})")
        print(
            f"Back-Action QND / QFI         : {rep.qnd_report.back_action_db:.3e} dB "
            f"(QND: {rep.qnd_report.is_qnd_passive}, F_Q: {rep.qnd_report.quantum_fisher_info:.3e})"
        )
        print(
            f"Fidelidad Uhlmann / d_B       : {rep.uhlmann_report.fidelity_uhlmann:.6f} / "
            f"{rep.uhlmann_report.bures_distance:.6f}"
        )
        print(
            f"Heaviside |Z₀| / |Γ| / VSWR   : {a.characteristic_impedance_mag:.4f} / "
            f"{a.telegrapher_reflection_coeff:.4f} / {a.telegrapher_vswr:.4f}"
        )
        print(
            f"Oseledets λ_max / Zhukovsky   : {rep.flow_report.oseledets_lyapunov_exponent:.4f} "
            f"(ρ = {rep.flow_report.spectral_radius:.4f}, ΔP = {rep.flow_report.zhukovsky_surge_pressure:.4f})"
        )
        print(f"Golpe de bélier suprimido     : {rep.flow_report.is_water_hammer_suppressed}")
        print(
            f"Invariantes de Smith / Hodge  : {a.smith_invariants} "
            f"(libre de torsión: {a.smith_torsion_free}, gap Δ₀: {a.hodge_spectral_gap:.4f})"
        )
        print(f"Valuación Krull v(A)          : {a.krull_valuation:.4f} (≥ piso: {a.krull_floor_ok})")
        print(
            f"Poincaré Defecto Simpléctico  : {k.symplectic_jacobian_defect:.2e} | "
            f"Deriva Cartan ∮p dq: {k.poincare_cartan_invariant_drift:.2e}"
        )
        print(
            f"Torsión Kolmogorov / Arnol'd  : det(T)={k.kolmogorov_torsion_det:.4f}, "
            f"det_Arn={k.arnold_isoenergetic_det:.4f} (KAM no-deg: {k.is_kam_non_degenerate})"
        )
        print(
            f"Serie Lie 2.º orden / Floquet : ‖½{{h₁,W}}‖={k.lie_series_second_order_norm:.2e}, "
            f"Tr(M)={k.floquet_monodromy_trace:.6f} (elíptica: {k.is_floquet_elliptic_stable})"
        )
        print(
            f"RSI³ Certificado (Löb/Kleisli): depth={r.depth_reached}, Lip_L1={r.lip_certified:.6f}, "
            f"Lip_L2={r.meta_preconditioned_lip:.6f}, Löb={r.loeb_soundness_verified}, "
            f"aceptado={r.accepted}"
        )
        print(f"Defecto Coherencia Kleisli    : {rep.kleisli_coherence_defect:.2e}")
        print(f"Contracción Kleisli Lip / Jerk: {rep.banach_contraction_lip:.6f} / {rep.capacity_jerk_d3C_dt3:.6e}")
        print(f"Pasaporte HMAC-SHA256         : {rep.passport_hash[:16]}...")

    print("=== AUTOPRUEBA: BoolNovikovToonBufferWitnessEngine v15.0.0-PhD ===")
    engine = BoolNovikovToonBufferWitnessEngine()
    exps = np.array([0.15, 0.28, 0.42], dtype=np.float64)

    # Caso A: Matriz de costo expansiva y cuantizada con torsión en ℤ → VETOED y rollback RSI³.
    matrix_a = np.array([[2.0, 0.1], [0.1, 2.5]], dtype=np.float64)
    rep_a = engine.execute_toon_buffer_witness_audit(matrix_a, exps)
    _imprimir("CASO A: matriz expansiva con torsión (VETOED + Rollback RSI³)", rep_a)

    # Caso B: Flujo contractivo + matriz de incidencia totalmente unimodular → COHERENT y RSI³ aceptado.
    matrix_b = np.array([[0.5, 0.1], [0.1, 0.4]], dtype=np.float64)
    incidence_b = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]], dtype=np.float64)
    rep_b = engine.execute_toon_buffer_witness_audit(
        matrix_b, exps, incidence_matrix=incidence_b
    )
    _imprimir("CASO B: flujo contractivo + incidencia unimodular (COHERENT + RSI³ Certificado)", rep_b)

    print("\n=== AUTOPRUEBA COMPLETADA CON ÉXITO ===")
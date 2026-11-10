
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : bool_novikov_pseudoholomorphic_witness_engine.py                                          ║
║ Versión  : 14.0.0-PhD-PHASES-QND-RSI3-POINCARE                                                       ║
║ Ubicación: app/wisdom/bool_novikov_pseudoholomorphic_witness_engine.py                               ║
║ Gobernanza: Motor Espectral de Atestación Pasiva QND (AAV) sobre el Fibrado Dual Boole-Novikov       ║
║            (R_Boole ⊗ Λ_Nov) en 3 FASES ANIDADAS con Cierre-Obertura de Certificados                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

MARCO FÍSICO-MATEMÁTICO DOCTORAL Y ESTRUCTURA TRIFÁSICA ANIDADA
───────────────────────────────────────────────────────────────
El Motor de Atestación Pasiva QND organiza su lógica en 3 FASES ANIDADAS:

  FASE I — ATESTACIÓN ALGEBRAICO-ESPECTRAL QND
    Invariantes evaluados:
      • Medición Débil AAV:        A_w = ⟨φ_f | A_witness | φ_i⟩ / ⟨φ_f | φ_i⟩ ∈ ℂ.
      • Conmutación QND:           ‖[A_witness, ρ_MAC]‖_F ≤ 10⁻¹⁰ ⟹ Back-Action ≡ 0 dB.
      • Fidelidad de Uhlmann:      F(ρ,σ) = (Tr√(√ρ σ √ρ))² ≥ 0.95 con PSD garantizada.
      • Distancia Bures:           d_B(ρ,σ) = √(2 − 2√F(ρ,σ)).
      • Anillo de Novikov:         v(A) = min{λ_i : a_i ≠ 0} ≥ 0.10 y ‖A‖_Nov = e^{−v(A)}.
    ÚLTIMO método:  `phase1_emit_witness_certificate` ➔ Phase1WitnessCertificate.
    Este retorno es la ENTRADA OBLIGATORIA del primer método de FASE II.

  FASE II — ATESTACIÓN GEOMÉTRICO-CELESTE
    Invariantes evaluados:
      • Superficie pseudoholomorfa con residual CR-Floer por curvatura media
        discreta sobre la gráfica de u: R² → R (proxy de holomorfía).
      • Rigidez Simpléctica de Gromov: c(B²ⁿ(r)) = πr² ≤ c(Z²ⁿ(R)) = πR².
      • Mecánica Celeste de Poincaré: coordenadas de Delaunay-Whittaker (J, θ).
      • Integral de Melnikov de la separatriz pendular forzada:
             M(τ) = 2π ε sech(πω/2) sin(ωτ).
      • Ecuación homológica de Poincaré-Lindstedt con suma de divisores pequeños.
      • Margen diofántico KAM: |ω·k| · |k|^τ.
    PRIMER método:  `phase2_open_from_phase1_certificate` (consume Phase1WitnessCertificate).
    ÚLTIMO método:  `phase2_emit_witness_certificate` ➔ Phase2WitnessCertificate.

  FASE III — RSI³ BANACH, ADJUDICACIÓN \Omega_3 Y EMISIÓN SOBERANA
    Invariantes evaluados:
      • Mónada Kleisli T³(x) = T(T(T(x))), T(x) = α·x + β·sin(x).
      • Cota Lipschitz teórica:     Lip(T³) ≤ (α+β)³ = 0.96³ ≈ 0.8847 < 1.
      • Residual Banach-Picard:     ‖T⁴(x) − T³(x)‖.
      • Jerk ciber-físico discreto: d³C/dt³ ≈ T³(x) − 3T²(x) + 3T(x) − x.
      • Adjudicación intuicionista en \Omega_3 = {0x00 ≺ 0x01 ≺ 0x03}.
    PRIMER método:  `phase3_open_from_phase2_certificate` (consume Phase2WitnessCertificate).
    ÚLTIMO método:  `phase3_emit_witness_passport` ➔ PseudoholomorphicWitnessReport.

La anidación garantiza:
  - Cierre criptográfico al final de cada fase (HMAC-SHA256 sobre payload determinista).
  - Apertura verificada al inicio de la fase siguiente (recalcula HMAC y cadena).
  - Trazabilidad forense completa: chain_hash P1 ⊕ chain_hash P2 ⊕ veredicto.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovPseudoholomorphicWitnessEngine")

_ENGINE_VERSION: Final[str] = (
    "14.0.0-ENGINE-WITNESS-PHASES-QND-RSI3-POINCARE-PhD"
)
__version__: Final[str] = _ENGINE_VERSION

# ──────────────────────────────────────────────────────────────────────────────
# Constantes numéricas y de gobernanza
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_QND_COMMUTATOR_BOUND: Final[float] = 1.0e-10
_QND_BACKACTION_BOUND_DB: Final[float] = 1.0e-3
_UHLMANN_FIDELITY_FLOOR: Final[float] = 0.95
_BANACH_ALPHA: Final[float] = 0.95
_BANACH_BETA: Final[float] = 0.01
_BANACH_LIP_T3: Final[float] = (_BANACH_ALPHA + _BANACH_BETA) ** 3  # ≈ 0.884736
_KAM_MARGIN_FLOOR: Final[float] = 1.0e-9
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_WITNESS_SECRET_KEY_v14.0.0"

_HMAC_DOMAIN_P1: Final[bytes] = b"WITNESS-P1-QND-ALGEBRAIC-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"WITNESS-P2-GEO-CELESTIAL-v14.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"WITNESS-P3-RSI3-SOVEREIGN-v14.0|"

_HEX_LEN: Final[int] = 64
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0


__all__ = [
    "HeytingTruthValue",
    "QNDWeakMeasurementReport",
    "BuresUhlmannMetricReport",
    "NovikovWitnessState",
    "PseudoholomorphicSurfaceWitness",
    "CelestialWitnessState",
    "RSI3WitnessState",
    "Phase1WitnessCertificate",
    "Phase2WitnessCertificate",
    "PseudoholomorphicWitnessReport",
    "WitnessGateViolationError",
    "WitnessParityViolationError",
    "BoolNovikovPseudoholomorphicWitnessEngine",
]


# ──────────────────────────────────────────────────────────────────────────────
# Excepciones específicas
# ──────────────────────────────────────────────────────────────────────────────
class WitnessGateViolationError(RuntimeError):
    r"""Falla de autenticidad o integridad en cualquiera de las compuertas Φ entre fases."""


class WitnessParityViolationError(RuntimeError):
    r"""Divergencia entre el veredicto de atestación y el esperado."""


# ──────────────────────────────────────────────────────────────────────────────
# Enum y estados atómicos
# ──────────────────────────────────────────────────────────────────────────────
class HeytingTruthValue(IntEnum):
    r"""Topos de Heyting \Omega_3 = {0x00 ≺ 0x01 ≺ 0x03}."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03


@dataclass(frozen=True)
class QNDWeakMeasurementReport:
    r"""Reporte de Medición Débil AAV y pasividad cuántica QND."""
    weak_value: complex
    back_action_db: float
    is_qnd_passive: bool
    commutator_norm: float
    overlap_probability: float
    weak_parameter_delta: float


@dataclass(frozen=True)
class BuresUhlmannMetricReport:
    r"""Reporte de Fidelidad de Uhlmann y Distancia Bures-Wasserstein."""
    fidelity_uhlmann: float
    bures_distance: float
    is_isometry_preserved: bool
    is_psd_rho: bool
    is_psd_sigma: bool


@dataclass(frozen=True)
class NovikovWitnessState:
    r"""Estado ultramétrico del anillo de Novikov."""
    exponents: Tuple[float, ...]
    coefficients: Tuple[complex, ...]
    krull_valuation: float
    ultrametric_norm: float
    is_valid_valuation: bool
    is_null_series: bool


@dataclass(frozen=True)
class PseudoholomorphicSurfaceWitness:
    r"""Atestación geométrica pseudoholomorfa y rigidez de Gromov."""
    grid_size: Tuple[int, int]
    cauchy_riemann_residual: float
    symplectic_area: float
    is_gromov_rigid: bool
    ball_radius_r: float
    cylinder_radius_R: float


@dataclass(frozen=True)
class CelestialWitnessState:
    r"""Dinámica hamiltoniana y estabilidad KAM (Poincaré-Delaunay-Whittaker)."""
    action_j: float
    angle_theta: float
    frequencies: Tuple[float, float]
    melnikov_integral_value: float
    melnikov_chaos_detected: bool
    kam_diophantine_margin: float
    poincare_lindstedt_residual: float
    secular_drift_norm: float
    twist_angle: float


@dataclass(frozen=True)
class RSI3WitnessState:
    r"""Estado RSI³ en la Mónada Kleisli-Banach de Nivel 3."""
    kleisli_input_norm: float
    kleisli_output_norm: float
    banach_contraction_constant: float
    fixed_point_error: float
    banach_residual: float
    capacity_jerk: float
    is_banach_contractive: bool
    jerk_strictly_positive: bool
    kleisli_depth: int = 3


# ──────────────────────────────────────────────────────────────────────────────
# Certificados de fase (P1 -> P2 -> final)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Phase1WitnessCertificate:
    r"""
    Certificado sellado de FASE I (ATESTACIÓN ALGEBRAICO-ESPECTRAL QND).

    Emitido por `phase1_emit_witness_certificate` (último método de FASE I).
    Consumido por `phase2_open_from_phase1_certificate` (primer método de FASE II).
    """
    timestamp: float
    qnd_report: QNDWeakMeasurementReport
    bures_report: BuresUhlmannMetricReport
    novikov_state: NovikovWitnessState
    chain_hash: str
    hmac_signature: str
    phase_ordinal: int = 1


@dataclass(frozen=True)
class Phase2WitnessCertificate:
    r"""
    Certificado sellado de FASE II (ATESTACIÓN GEOMÉTRICO-CELESTE).

    Emitido por `phase2_emit_witness_certificate` (último método de FASE II).
    Consumido por `phase3_open_from_phase2_certificate` (primer método de FASE III).
    """
    timestamp: float
    previous_chain_hash: str
    surface_witness: PseudoholomorphicSurfaceWitness
    celestial_witness: CelestialWitnessState
    chain_hash: str
    hmac_signature: str
    phase_ordinal: int = 2


@dataclass(frozen=True)
class PseudoholomorphicWitnessReport:
    r"""
    Reporte final soberano de atestación pasiva QND.

    Emitido por `phase3_emit_witness_passport` (último método de FASE III).
    Incluye trazabilidad forense completa: chain P1 ⊕ chain P2 ⊕ veredicto.
    """
    # ── Campos originales (compatibilidad v13) ──
    cauchy_riemann_residual: float
    krull_valuation: float
    novikov_norm: float
    is_gromov_rigid: bool
    melnikov_chaos_detected: bool
    banach_contraction_lip: float
    qnd_report: QNDWeakMeasurementReport
    bures_report: BuresUhlmannMetricReport
    truth_value: HeytingTruthValue
    passport_hash: str
    execution_time_ms: float
    # ── Campos enriquecidos (v14, con defaults para compatibilidad) ──
    celestial_state: Optional[CelestialWitnessState] = None
    rsi3_state: Optional[RSI3WitnessState] = None
    adjudication_reason: str = ""
    sovereign_hmac_sha256: str = ""
    phase1_chain_hash: str = ""
    phase2_chain_hash: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# MOTOR ESPECTRAL DE ATESTACIÓN PASIVA QND (3 FASES ANIDADAS)
# ──────────────────────────────────────────────────────────────────────────────
class BoolNovikovPseudoholomorphicWitnessEngine:
    r"""
    Motor Espectral de Atestación Pasiva QND sobre el Fibrado Dual Boole-Novikov.

    Estructura (contrato de anidamiento):
      FASE I  ➔ Phase1WitnessCertificate ➔
      FASE II ➔ Phase2WitnessCertificate ➔
      FASE III ➔ PseudoholomorphicWitnessReport.
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key: bytes = secret_key
        logger.info(
            "%s v%s inicializado.",
            self.__class__.__name__,
            _ENGINE_VERSION,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE I — ATESTACIÓN ALGEBRAICO-ESPECTRAL QND
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _sanitize_psd(matrix: NDArray[np.complex128]) -> Tuple[NDArray[np.complex128], bool]:
        r"""
        Fuerza hermiticidad y detecta PSD. Retorna (M_hermitizada, is_psd).
        """
        try:
            arr = np.atleast_2d(np.asarray(matrix, dtype=np.complex128))
        except Exception:
            return np.zeros((0, 0), dtype=np.complex128), False
        if arr.size == 0:
            return arr, False
        arr = 0.5 * (arr + arr.conj().T)
        try:
            evals = np.linalg.eigvalsh(arr)
            is_psd = bool(np.all(evals >= -1e-9))
        except np.linalg.LinAlgError:
            is_psd = False
        return arr, is_psd

    @staticmethod
    def _sqrtm_psd(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
        r"""Raíz cuadrada principal de una matriz PSD hermitiana vía eigh."""
        herm = 0.5 * (matrix + matrix.conj().T)
        try:
            evals, evecs = np.linalg.eigh(herm)
        except np.linalg.LinAlgError:
            return np.zeros_like(herm)
        evals = np.clip(evals.real, 0.0, None)
        return (evecs * np.sqrt(evals)) @ evecs.conj().T

    def phase1_witness_qnd_weak_measurement(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: Optional[NDArray[np.complex128]] = None,
        delta: float = 0.05,
    ) -> QNDWeakMeasurementReport:
        r"""
        Medición Débil AAV y verificación del invariante QND (Back-Action ≡ 0 dB).

        Implementación:
          • A_witness es diagonal en la base computacional (QND compatible).
          • Se añade una perturbación hermitiana off-diagonal escalada por delta
            para chequeo de robustez (passive-silent test).
          • Se aplica la fórmula AAV: A_w = ⟨φ_f|A|φ_i⟩/⟨φ_f|φ_i⟩.
          • Se mide ‖[A_witness, ρ_MAC]‖_F contra cota _QND_COMMUTATOR_BOUND.

        Nota crítica: con ρ_MAC = I/dim (estado máximamente mixto), el conmutador
        es exactamente nulo independientemente de A_witness, por lo cual la
        cota QND se cumple trivialmente. Para auditorías más estrictas se debe
        proporcionar ρ_MAC no uniforme.

        Parámetros
        ----------
        rho_mac : ndarray[complex128]
            Matriz de densidad del testigo MAC (debe ser PSD hermitiana).
        phi_i : ndarray[complex128]
            Vector de estado pre-seleccionado.
        phi_f : ndarray[complex128], optional
            Vector de estado post-seleccionado (por defecto phi_i).
        delta : float, optional
            Fuerza del acoplamiento débil perturbativo (default 0.05).

        Retorna
        -------
        QNDWeakMeasurementReport
            Reporte con weak_value, back_action_db, commutator_norm y overlap.
        """
        rho, _ = self._sanitize_psd(rho_mac)
        if rho.size == 0:
            return QNDWeakMeasurementReport(
                weak_value=0.0 + 0.0j,
                back_action_db=0.0,
                is_qnd_passive=True,
                commutator_norm=0.0,
                overlap_probability=0.0,
                weak_parameter_delta=float(delta),
            )

        dim = rho.shape[0]
        phi_i = np.asarray(phi_i, dtype=np.complex128).flatten()
        if phi_f is None:
            phi_f = phi_i.copy()
        phi_f = np.asarray(phi_f, dtype=np.complex128).flatten()

        if phi_i.size != dim or phi_f.size != dim:
            raise ValueError(
                f"[QND] Dimensión inconsistente: dim(rho)={dim}, "
                f"|phi_i|={phi_i.size}, |phi_f|={phi_f.size}."
            )

        # Operador testigo QND: diagonal en la base computacional.
        diag_weights = np.linspace(0.35, 1.65, dim).astype(np.complex128)
        A_witness = np.diag(diag_weights)

        # Perturbación hermitiana controlada por delta (robustez pasiva).
        if delta > 0.0 and dim > 1:
            perturb = np.zeros_like(A_witness)
            for k in range(dim - 1):
                c = (float(delta) ** 2) * (0.5 + 0.5j)
                perturb[k, k + 1] = c
                perturb[k + 1, k] = np.conj(c)
            A_witness = A_witness + perturb

        # Medición débil AAV
        overlap = np.vdot(phi_f, phi_i)
        if abs(overlap) < _SPECTRAL_PSD_FLOOR:
            overlap = _SPECTRAL_PSD_FLOOR + 0.0j
        matrix_elem = np.vdot(phi_f, A_witness @ phi_i)
        weak_val = complex(matrix_elem / overlap)

        # Conmutador QND
        comm = A_witness @ rho - rho @ A_witness
        comm_norm = float(la.norm(comm, ord="fro"))

        # Back-action (dB, escala logarítmica estricta)
        back_action = 10.0 * math.log10(1.0 + comm_norm)

        is_qnd = (
            comm_norm <= _QND_COMMUTATOR_BOUND
            and abs(back_action) <= _QND_BACKACTION_BOUND_DB
        )

        return QNDWeakMeasurementReport(
            weak_value=weak_val,
            back_action_db=float(back_action),
            is_qnd_passive=bool(is_qnd),
            commutator_norm=comm_norm,
            overlap_probability=float(abs(overlap) ** 2),
            weak_parameter_delta=float(delta),
        )

    def phase1_witness_bures_uhlmann(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> BuresUhlmannMetricReport:
        r"""
        Fidelidad cuántica de Uhlmann F(ρ,σ) y Distancia Bures-Wasserstein.

        Sanitización PSD con eigvalsh; raíces cuadradas principales vía eigh.
        F = (Tr √(√ρ σ √ρ))² ∈ [0, 1] por construcción.
        d_B = √(2 − 2√F).
        """
        rho_h, is_psd_rho = self._sanitize_psd(rho)
        sigma_h, is_psd_sigma = self._sanitize_psd(sigma)

        if rho_h.shape != sigma_h.shape or rho_h.size == 0:
            return BuresUhlmannMetricReport(
                fidelity_uhlmann=0.0,
                bures_distance=float(math.sqrt(2.0)),
                is_isometry_preserved=False,
                is_psd_rho=is_psd_rho,
                is_psd_sigma=is_psd_sigma,
            )

        if not (is_psd_rho and is_psd_sigma):
            return BuresUhlmannMetricReport(
                fidelity_uhlmann=0.0,
                bures_distance=float(math.sqrt(2.0)),
                is_isometry_preserved=False,
                is_psd_rho=is_psd_rho,
                is_psd_sigma=is_psd_sigma,
            )

        rho_sqrt = self._sqrtm_psd(rho_h)
        inner = rho_sqrt @ sigma_h @ rho_sqrt
        inner_sqrt = self._sqrtm_psd(inner)
        trace_val = float(np.real(np.trace(inner_sqrt)))
        fidelity = float(np.clip(trace_val ** 2, 0.0, 1.0))

        bures_dist = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fidelity)))
        fidelity_floor_distance = math.sqrt(
            max(0.0, 2.0 - 2.0 * math.sqrt(_UHLMANN_FIDELITY_FLOOR))
        )
        is_iso = (
            fidelity >= _UHLMANN_FIDELITY_FLOOR
            and bures_dist <= fidelity_floor_distance + 1e-9
        )

        return BuresUhlmannMetricReport(
            fidelity_uhlmann=fidelity,
            bures_distance=float(bures_dist),
            is_isometry_preserved=bool(is_iso),
            is_psd_rho=is_psd_rho,
            is_psd_sigma=is_psd_sigma,
        )

    def phase1_witness_novikov_krull(
        self,
        coefficients: Sequence[complex],
        exponents: Sequence[float],
    ) -> NovikovWitnessState:
        r"""
        Valuación de Krull v(A) = min{λ_i : |a_i| > 0} y norma ultramétrica.

        - Filtra coeficientes nulos (|a_i| ≤ _SPECTRAL_PSD_FLOOR).
        - Si todos nulos: serie nula ⟹ v ≡ +∞, válida por vacuidad.
        - Verifica v(A) ≥ _DEFAULT_KRULL_FLOOR.
        - ‖A‖_Nov = exp(−v) con piso numérico por estabilidad (v ≤ 700).
        """
        try:
            coeffs = tuple(complex(c) for c in coefficients)
            exps = tuple(float(e) for e in exponents)
        except Exception:
            return NovikovWitnessState(
                exponents=(),
                coefficients=(),
                krull_valuation=float("inf"),
                ultrametric_norm=0.0,
                is_valid_valuation=False,
                is_null_series=True,
            )

        if not coeffs or not exps or len(coeffs) != len(exps):
            return NovikovWitnessState(
                exponents=exps if exps else (),
                coefficients=coeffs if coeffs else (),
                krull_valuation=float("inf"),
                ultrametric_norm=0.0,
                is_valid_valuation=False,
                is_null_series=True,
            )

        valid_pairs: List[Tuple[complex, float]] = []
        for c, e in zip(coeffs, exps):
            if not (math.isfinite(c.real) and math.isfinite(c.imag) and math.isfinite(e)):
                return NovikovWitnessState(
                    exponents=exps,
                    coefficients=coeffs,
                    krull_valuation=float("inf"),
                    ultrametric_norm=0.0,
                    is_valid_valuation=False,
                    is_null_series=False,
                )
            if abs(c) > _SPECTRAL_PSD_FLOOR:
                valid_pairs.append((c, e))

        if not valid_pairs:
            return NovikovWitnessState(
                exponents=exps,
                coefficients=coeffs,
                krull_valuation=float("inf"),
                ultrametric_norm=0.0,
                is_valid_valuation=True,  # Vacuous truth
                is_null_series=True,
            )

        v_krull = min(e for _, e in valid_pairs)
        norm_nov = 0.0 if v_krull > 700.0 else math.exp(-v_krull)
        is_valid = v_krull >= _DEFAULT_KRULL_FLOOR

        return NovikovWitnessState(
            exponents=exps,
            coefficients=coeffs,
            krull_valuation=float(v_krull),
            ultrametric_norm=float(norm_nov),
            is_valid_valuation=bool(is_valid),
            is_null_series=False,
        )

    # ── Payload determinista P1 (idéntico en emisión y verificación) ──
    @staticmethod
    def _p1_payload_bytes(
        timestamp: float,
        qnd: QNDWeakMeasurementReport,
        bures: BuresUhlmannMetricReport,
        novikov: NovikovWitnessState,
    ) -> bytes:
        krull_str = "inf" if not math.isfinite(novikov.krull_valuation) else f"{novikov.krull_valuation:.12g}"
        return (
            f"WITNESS-P1|{timestamp:.12g}|"
            f"{qnd.weak_value.real:.12g}|{qnd.weak_value.imag:.12g}|"
            f"{qnd.commutator_norm:.12g}|{qnd.back_action_db:.12g}|{qnd.is_qnd_passive}|"
            f"{bures.fidelity_uhlmann:.12g}|{bures.bures_distance:.12g}|"
            f"{bures.is_isometry_preserved}|"
            f"{krull_str}|{novikov.ultrametric_norm:.12g}|"
            f"{novikov.is_valid_valuation}|{novikov.is_null_series}"
        ).encode("utf-8")

    def phase1_emit_witness_certificate(
        self,
        qnd_report: QNDWeakMeasurementReport,
        bures_report: BuresUhlmannMetricReport,
        novikov_state: NovikovWitnessState,
    ) -> Phase1WitnessCertificate:
        r"""
        ÚLTIMO MÉTODO DE LA FASE I.

        Ensambla el `Phase1WitnessCertificate` con:
          • chain_hash     = SHA-256(payload P1)
          • hmac_signature = HMAC-SHA256(_HMAC_DOMAIN_P1 + payload P1)

        El retorno es la ENTRADA OBLIGATORIA del primer método de FASE II:
        `phase2_open_from_phase1_certificate`.
        """
        now = time.time()
        payload = self._p1_payload_bytes(now, qnd_report, bures_report, novikov_state)
        chain = hashlib.sha256(payload).hexdigest()
        sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + payload,
            hashlib.sha256,
        ).hexdigest()

        return Phase1WitnessCertificate(
            timestamp=now,
            qnd_report=qnd_report,
            bures_report=bures_report,
            novikov_state=novikov_state,
            chain_hash=chain,
            hmac_signature=sig,
            phase_ordinal=1,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE II — ATESTACIÓN GEOMÉTRICO-CELESTE
    # ══════════════════════════════════════════════════════════════════════════

    def phase2_open_from_phase1_certificate(
        self,
        cert_p1: Phase1WitnessCertificate,
    ) -> Phase1WitnessCertificate:
        r"""
        PRIMER MÉTODO DE LA FASE II.

        Consume el `Phase1WitnessCertificate` emitido por el último método de
        FASE I, revalida la firma HMAC sobre el payload determinista y la
        longitud del chain_hash, y lo retorna como contexto operativo para
        los pasos siguientes de FASE II.

        Levanta `WitnessGateViolationError` si la autenticidad o la integridad
        estructural fallan (fail-closed).
        """
        payload = self._p1_payload_bytes(
            cert_p1.timestamp,
            cert_p1.qnd_report,
            cert_p1.bures_report,
            cert_p1.novikov_state,
        )
        expected_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + payload,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, cert_p1.hmac_signature):
            raise WitnessGateViolationError(
                r"Compuerta Φ₁₂: Firma HMAC P1 inválida."
            )
        if len(cert_p1.chain_hash) != _HEX_LEN:
            raise WitnessGateViolationError(
                r"Compuerta Φ₁₂: chain_hash P1 corrupto (no mide 64-hex)."
            )
        if not math.isfinite(cert_p1.timestamp) or cert_p1.timestamp <= 0.0:
            raise WitnessGateViolationError(
                r"Compuerta Φ₁₂: timestamp P1 no finito o no positivo."
            )
        return cert_p1

    def _compute_cr_residual(self, u: NDArray[np.float64]) -> float:
        r"""
        Residual CR-Floer discreto de una superficie pseudoholomorfa u: R² → R.

        Estrategia granulada:
          • Grid pequeño (n<3 o m<3): proxy de simetría de gradientes
            ‖‖∂_s u‖ − ‖∂_τ u‖‖ / max(‖∂_s u‖, ‖∂_τ u‖).
          • Grid grande (n,m ≥ 3): curvatura media discreta de la gráfica de u,
                H = [(1+g_τ²)u_ss − 2g_s g_τ u_sτ + (1+g_s²)u_ττ] / [2(1+g_s²+g_τ²)^{3/2}]
            que se anula sii la gráfica es un plano (proxy de holomorfía CR).
          • Residuo devuelto = ‖H‖_F / ‖u_interior‖_F, con piso numérico
            en _SPECTRAL_PSD_FLOOR para robustez.
        """
        n, m = u.shape
        if n < 2 or m < 2:
            return 0.0

        gs = np.gradient(u, axis=0) if n > 1 else np.zeros_like(u)
        gt = np.gradient(u, axis=1) if m > 1 else np.zeros_like(u)

        if n < 3 or m < 3:
            gs_norm = float(np.linalg.norm(gs))
            gt_norm = float(np.linalg.norm(gt))
            scale = max(gs_norm, gt_norm, _SPECTRAL_PSD_FLOOR)
            res = abs(gs_norm - gt_norm) / scale
            return max(res, 0.0)

        # Curvatura media discreta de la gráfica
        u_ss = u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]
        u_tt = u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]
        u_st = (u[2:, 2:] + u[:-2, :-2] - u[2:, :-2] - u[:-2, 2:]) / 4.0
        gs_i = gs[1:-1, 1:-1]
        gt_i = gt[1:-1, 1:-1]

        denom_base = 1.0 + gs_i ** 2 + gt_i ** 2
        H = (
            (1.0 + gt_i ** 2) * u_ss
            - 2.0 * gs_i * gt_i * u_st
            + (1.0 + gs_i ** 2) * u_tt
        ) / (2.0 * denom_base ** 1.5)

        H_norm = float(np.linalg.norm(H))
        u_norm = float(np.linalg.norm(u[1:-1, 1:-1])) + _SPECTRAL_PSD_FLOOR
        res = H_norm / u_norm
        return max(res, 0.0)

    def phase2_witness_cauchy_riemann_surface(
        self,
        cost_matrix: NDArray[np.float64],
        ball_r: float = 1.0,
        cylinder_R: float = 1.2,
    ) -> PseudoholomorphicSurfaceWitness:
        r"""
        Residual CR-Floer discreto y rigidez de Gromov:

            c(B²ⁿ(r)) = π r² ≤ c(Z²ⁿ(R)) = π R².

        Manipula NaN/Inf de la entrada y devuelve invariantes normalizados.
        """
        u = np.atleast_2d(np.asarray(cost_matrix, dtype=np.float64))
        n, m = u.shape if u.ndim == 2 else (0, 0)
        r = abs(float(ball_r)) if math.isfinite(ball_r) else 0.0
        R_c = abs(float(cylinder_R)) if math.isfinite(cylinder_R) else 0.0

        is_gromov = (math.pi * r * r) <= (math.pi * R_c * R_c) + 1e-9

        if u.size == 0:
            return PseudoholomorphicSurfaceWitness(
                grid_size=(n, m),
                cauchy_riemann_residual=0.0,
                symplectic_area=0.0,
                is_gromov_rigid=bool(is_gromov),
                ball_radius_r=r,
                cylinder_radius_R=R_c,
            )

        u = np.nan_to_num(u, nan=0.0, posinf=1e12, neginf=-1e12)
        res_cr = float(self._compute_cr_residual(u))
        symplectic_area = float(np.sum(u * u))

        return PseudoholomorphicSurfaceWitness(
            grid_size=(n, m),
            cauchy_riemann_residual=res_cr,
            symplectic_area=symplectic_area,
            is_gromov_rigid=bool(is_gromov),
            ball_radius_r=float(r),
            cylinder_radius_R=float(R_c),
        )

    def phase2_witness_poincare_kam(
        self,
        surface: PseudoholomorphicSurfaceWitness,
        pert_eps: float = 0.05,
    ) -> CelestialWitnessState:
        r"""
        Mecánica Celeste de Poincaré con rigor doctoral.

        Componentes:
          • Coordenadas Delaunay-Whittaker (J, θ) con J derivada del área
            simpléctica y θ con rotación áurea (evita resonancia).
          • Frecuencias Keplerianas ω₁ ~ J^{−3} y detuning áureo ω₂ = φω₁.
          • Integral de Melnikov para la separatriz pendular forzada:
                M(τ) = 2π ε sech(πω/2) sin(ωτ).
          • Ecuación homológica de Poincaré-Lindstedt con suma de pequeños divisores.
          • Margen diofántico KAM: min|ω·k|·|k|^τ con τ = 1.5.
          • Twist de Poincaré-Birkhoff: ω₁·2π mod 2π.
        """
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        eps = float(abs(pert_eps)) if math.isfinite(pert_eps) else 0.0

        action_j = max(
            1e-9,
            abs(float(surface.symplectic_area)) / (4.0 * math.pi * math.pi) + 0.5,
        )
        omega1 = 1.0 / (action_j ** 3 + 1e-12)
        omega1 = min(max(omega1, 1e-8), 1.0e6)
        omega2 = phi * omega1 * (1.0 + 1e-6)

        angle_theta = (action_j * phi) % (2.0 * math.pi)

        # Melnikov (separatriz pendular forzada)
        omega_pert = max(1e-6, min(omega1, 1.0e3))
        if omega_pert * math.pi / 2.0 > 700.0:
            sech_term = 0.0
        else:
            sech_term = 1.0 / math.cosh(0.5 * math.pi * omega_pert)

        melnikov_val = (
            2.0 * eps * math.pi * sech_term * math.sin(omega_pert * angle_theta)
        )
        has_chaos = abs(melnikov_val) > 0.35

        # Lindstedt: suma ponderada de pequeños divisores
        sum_inv = 0.0
        for k1 in range(-2, 3):
            for k2 in range(-2, 3):
                if k1 == 0 and k2 == 0:
                    continue
                denom = abs(k1 * omega1 + k2 * omega2)
                weight = 1.0 / (1.0 + k1 * k1 + k2 * k2)
                if denom < _WILKINSON_EPSILON:
                    sum_inv += 1e6
                elif denom < 0.02:
                    sum_inv += 10.0 * weight / denom
                else:
                    sum_inv += weight / denom
        lindstedt_residual = min(1.0, eps * sum_inv / (2.0 * math.pi))

        # Margen diofántico KAM
        tau = 1.5
        kam_margin = min(
            abs(k1 * omega1 + k2 * omega2) * ((k1 * k1 + k2 * k2) ** (0.5 * tau))
            for k1 in range(-2, 3)
            for k2 in range(-2, 3)
            if not (k1 == 0 and k2 == 0)
        )

        secular_drift = min(
            1.0,
            lindstedt_residual + abs(melnikov_val) / (1.0 + abs(melnikov_val)),
        )
        twist = (omega1 * 2.0 * math.pi) % (2.0 * math.pi)

        return CelestialWitnessState(
            action_j=float(action_j),
            angle_theta=float(angle_theta),
            frequencies=(float(omega1), float(omega2)),
            melnikov_integral_value=float(melnikov_val),
            melnikov_chaos_detected=bool(has_chaos),
            kam_diophantine_margin=float(kam_margin),
            poincare_lindstedt_residual=float(lindstedt_residual),
            secular_drift_norm=float(secular_drift),
            twist_angle=float(twist),
        )

    @staticmethod
    def _p2_payload_bytes(
        timestamp: float,
        previous_chain_hash: str,
        surface: PseudoholomorphicSurfaceWitness,
        celestial: CelestialWitnessState,
    ) -> bytes:
        return (
            f"WITNESS-P2|{timestamp:.12g}|{previous_chain_hash}|"
            f"{surface.cauchy_riemann_residual:.12g}|"
            f"{surface.symplectic_area:.12g}|{surface.is_gromov_rigid}|"
            f"{celestial.melnikov_integral_value:.12g}|"
            f"{celestial.melnikov_chaos_detected}|"
            f"{celestial.kam_diophantine_margin:.12g}|"
            f"{celestial.poincare_lindstedt_residual:.12g}|"
            f"{celestial.secular_drift_norm:.12g}"
        ).encode("utf-8")

    def phase2_emit_witness_certificate(
        self,
        cert_p1: Phase1WitnessCertificate,
        surface_witness: PseudoholomorphicSurfaceWitness,
        celestial_witness: CelestialWitnessState,
    ) -> Phase2WitnessCertificate:
        r"""
        ÚLTIMO MÉTODO DE LA FASE II.

        Ensambla el `Phase2WitnessCertificate` encadenando al P1 vía
        `previous_chain_hash = cert_p1.chain_hash`. Su retorno es la ENTRADA
        OBLIGATORIA del primer método de FASE III:
        `phase3_open_from_phase2_certificate`.
        """
        now = time.time()
        payload = self._p2_payload_bytes(
            now, cert_p1.chain_hash, surface_witness, celestial_witness
        )
        chain = hashlib.sha256(payload).hexdigest()
        sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + payload,
            hashlib.sha256,
        ).hexdigest()

        return Phase2WitnessCertificate(
            timestamp=now,
            previous_chain_hash=cert_p1.chain_hash,
            surface_witness=surface_witness,
            celestial_witness=celestial_witness,
            chain_hash=chain,
            hmac_signature=sig,
            phase_ordinal=2,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE III — RSI³ BANACH, ADJUDICACIÓN \Omega_3 Y EMISIÓN SOBERANA
    # ══════════════════════════════════════════════════════════════════════════

    def phase3_open_from_phase2_certificate(
        self,
        cert_p1: Phase1WitnessCertificate,
        cert_p2: Phase2WitnessCertificate,
    ) -> Tuple[Phase1WitnessCertificate, Phase2WitnessCertificate]:
        r"""
        PRIMER MÉTODO DE LA FASE III.

        Consume el `Phase2WitnessCertificate` emitido por el último método de
        FASE II. Revalida HMAC P2, longitud de hashes y encadenamiento
        P1 → P2. Levanta `WitnessGateViolationError` si falla (fail-closed).
        """
        payload = self._p2_payload_bytes(
            cert_p2.timestamp,
            cert_p2.previous_chain_hash,
            cert_p2.surface_witness,
            cert_p2.celestial_witness,
        )
        expected_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + payload,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, cert_p2.hmac_signature):
            raise WitnessGateViolationError(
                r"Compuerta Φ₂₃: Firma HMAC P2 inválida."
            )
        if len(cert_p2.chain_hash) != _HEX_LEN:
            raise WitnessGateViolationError(
                r"Compuerta Φ₂₃: chain_hash P2 corrupto (no mide 64-hex)."
            )
        if cert_p2.previous_chain_hash != cert_p1.chain_hash:
            raise WitnessGateViolationError(
                r"Compuerta Φ₂₃: ruptura de cadena P1 → P2."
            )
        return cert_p1, cert_p2

    def phase3_witness_rsi3_banach(
        self,
        state_vector: NDArray[np.float64],
        alpha: float = _BANACH_ALPHA,
        beta: float = _BANACH_BETA,
    ) -> RSI3WitnessState:
        r"""
        Automejora recursiva monádica de Nivel 3 (RSI³).

        Implementación rigurosa:
          • Operador: T(x) = α·x + β·sin(x).
          • Iteración monádica Kleisli: x₀ → x₁ → x₂ → x₃ → x₄.
          • Cota teórica de Lipschitz: L = α + β, Lip(T³) ≤ L³.
          • Residual Banach-Picard: ‖T⁴(x) − T³(x)‖.
          • Jerk ciber-físico: ‖T³(x) − 3T²(x) + 3T(x) − x‖.
          • Contracción verificada: L³ < 1 Y residual ≤ tol.
          • Jerk estrictamente positivo: jerk > piso numérico ampliado.
        """
        vec = np.atleast_1d(np.asarray(state_vector, dtype=np.float64)).flatten()
        L_theory = float((alpha + beta) ** 3)

        if vec.size == 0:
            return RSI3WitnessState(
                kleisli_input_norm=0.0,
                kleisli_output_norm=0.0,
                banach_contraction_constant=L_theory,
                fixed_point_error=0.0,
                banach_residual=0.0,
                capacity_jerk=_SPECTRAL_PSD_FLOOR,
                is_banach_contractive=(L_theory < 1.0),
                jerk_strictly_positive=False,
                kleisli_depth=3,
            )

        vec = np.nan_to_num(vec, nan=0.0, posinf=1e12, neginf=-1e12)
        x0 = vec.copy()
        input_norm = float(np.linalg.norm(x0))

        def T(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return alpha * x + beta * np.sin(x)

        x1 = T(x0)
        x2 = T(x1)
        x3 = T(x2)
        x4 = T(x3)

        output_norm = float(np.linalg.norm(x3))
        residual = float(np.linalg.norm(x4 - x3))
        jerk = float(np.linalg.norm(x3 - 3.0 * x2 + 3.0 * x1 - x0))
        capacity_jerk = max(jerk, _SPECTRAL_PSD_FLOOR)

        tol = max(1.0, 1.0e-6 * max(1.0, input_norm))
        is_contractive = (L_theory < 1.0) and (residual <= tol)
        jerk_positive = jerk > _SPECTRAL_PSD_FLOOR * 1.0e3

        return RSI3WitnessState(
            kleisli_input_norm=input_norm,
            kleisli_output_norm=output_norm,
            banach_contraction_constant=L_theory,
            fixed_point_error=residual,
            banach_residual=residual,
            capacity_jerk=float(capacity_jerk),
            is_banach_contractive=bool(is_contractive),
            jerk_strictly_positive=bool(jerk_positive),
            kleisli_depth=3,
        )

    def phase3_witness_adjudicate_heyting(
        self,
        cert_p1: Phase1WitnessCertificate,
        cert_p2: Phase2WitnessCertificate,
        rsi3_state: RSI3WitnessState,
    ) -> Tuple[HeytingTruthValue, str]:
        r"""
        Adjudicación en el Retículo de Heyting \Omega_3.

        Jerarquía intuicionista: COHERENT (0x00) ≺ DEGRADED (0x01) ≺ VETOED (0x03).

        Reglas VETOED (fallo estructural):
          - Valuación de Krull bajo el piso 0.10.
          - Back-action QND > cota (no pasividad pasiva).
          - Gromov no-squeezing violado.
          - CR residual no finito / weak value NaN.
        Reglas DEGRADED (fallo de calidad):
          - Fidelidad Uhlmann bajo el piso 0.95.
          - Caos homoclínico detectado por Melnikov.
          - Margen diofántico KAM < 1e-9.
          - Lindstedt residual > 0.50 (deriva secular excesiva).
          - RSI³ no contractiva (Lip(T³) ≥ 1).
        """
        n = cert_p1.novikov_state
        q = cert_p1.qnd_report
        b = cert_p1.bures_report
        s = cert_p2.surface_witness
        c = cert_p2.celestial_witness
        r = rsi3_state

        # ── VETOED ──
        if not n.is_valid_valuation:
            return HeytingTruthValue.VETOED, "NOVIKOV_KRULL_VALUATION_FLOOR_VIOLATION"
        if not q.is_qnd_passive:
            return HeytingTruthValue.VETOED, "QND_BACKACTION_VIOLATION"
        if not s.is_gromov_rigid:
            return HeytingTruthValue.VETOED, "GROMOV_NON_SQUEEZING_VIOLATION"
        if not math.isfinite(s.cauchy_riemann_residual):
            return HeytingTruthValue.VETOED, "CAUCHY_RIEMANN_NON_FINITE"
        if not (
            math.isfinite(q.weak_value.real) and math.isfinite(q.weak_value.imag)
        ):
            return HeytingTruthValue.VETOED, "WEAK_VALUE_NON_FINITE"

        # ── DEGRADED ──
        if not b.is_isometry_preserved:
            return HeytingTruthValue.DEGRADED, "UHLMANN_FIDELITY_BELOW_FLOOR"
        if c.melnikov_chaos_detected:
            return HeytingTruthValue.DEGRADED, "MELNIKOV_HOMOCLINIC_TANGLE"
        if c.kam_diophantine_margin < _KAM_MARGIN_FLOOR:
            return HeytingTruthValue.DEGRADED, "KAM_DIOPHANTINE_MARGIN_LOW"
        if c.poincare_lindstedt_residual > 0.50:
            return HeytingTruthValue.DEGRADED, "LINDSTEDT_SECULAR_DRIFT_EXCESSIVE"
        if not r.is_banach_contractive:
            return HeytingTruthValue.DEGRADED, "RSI3_BANACH_NON_CONTRACTIVE"

        return HeytingTruthValue.COHERENT, "ALL_WITNESS_INVARIANTS_SATISFIED"

    def phase3_emit_witness_passport(
        self,
        cert_p1: Phase1WitnessCertificate,
        cert_p2: Phase2WitnessCertificate,
        rsi3_state: RSI3WitnessState,
        truth_value: HeytingTruthValue,
        adjudication_reason: str,
        execution_time_ms: float,
    ) -> PseudoholomorphicWitnessReport:
        r"""
        ÚLTIMO MÉTODO DE LA FASE III.

        Emite el `PseudoholomorphicWitnessReport` final soberano con:
          • passport_hash = SHA-256(chain P1 ⊕ chain P2 ⊕ veredicto ⊕ RSI³ ⊕ razón)
          • sovereign_hmac_sha256 = HMAC-SHA256(_HMAC_DOMAIN_P3 + payload)
          • Trazabilidad forense completa: phase1_chain_hash ⊕ phase2_chain_hash.
        """
        payload = (
            f"WITNESS-FINAL|{cert_p1.chain_hash}|{cert_p2.chain_hash}|"
            f"{rsi3_state.banach_contraction_constant:.12g}|"
            f"{rsi3_state.fixed_point_error:.12g}|"
            f"{rsi3_state.capacity_jerk:.12g}|"
            f"{rsi3_state.is_banach_contractive}|"
            f"{truth_value.value}|{adjudication_reason}|"
            f"{execution_time_ms:.6f}"
        ).encode("utf-8")

        passport = hashlib.sha256(payload).hexdigest()
        sovereign_hmac = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P3 + payload,
            hashlib.sha256,
        ).hexdigest()

        return PseudoholomorphicWitnessReport(
            cauchy_riemann_residual=cert_p2.surface_witness.cauchy_riemann_residual,
            krull_valuation=cert_p1.novikov_state.krull_valuation,
            novikov_norm=cert_p1.novikov_state.ultrametric_norm,
            is_gromov_rigid=cert_p2.surface_witness.is_gromov_rigid,
            melnikov_chaos_detected=cert_p2.celestial_witness.melnikov_chaos_detected,
            banach_contraction_lip=rsi3_state.banach_contraction_constant,
            qnd_report=cert_p1.qnd_report,
            bures_report=cert_p1.bures_report,
            truth_value=truth_value,
            passport_hash=passport,
            execution_time_ms=float(execution_time_ms),
            celestial_state=cert_p2.celestial_witness,
            rsi3_state=rsi3_state,
            adjudication_reason=adjudication_reason,
            sovereign_hmac_sha256=sovereign_hmac,
            phase1_chain_hash=cert_p1.chain_hash,
            phase2_chain_hash=cert_p2.chain_hash,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ORQUESTADOR DE LAS 3 FASES ANIDADAS
    # ══════════════════════════════════════════════════════════════════════════

    def execute_witness_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
        coefficients: Optional[Sequence[complex]] = None,
        ball_r: float = 1.0,
        cylinder_R: float = 1.2,
        pert_eps: float = 0.05,
        delta_qnd: float = 0.05,
    ) -> PseudoholomorphicWitnessReport:
        r"""
        Orquestador completo de atestación pasiva QND.

        Encadena las 3 fases anidadas con sus dos compuertas:
            FASE I  ➔ [Φ₁₂] ➔  FASE II  ➔ [Φ₂₃] ➔  FASE III.

        Parámetros
        ----------
        apu_cost_matrix : ndarray[float64]
            Matriz de costos (n × m) interpretada como superficie u: R² → R.
        exponents : sequence[float]
            Exponentes λ_i de la serie de Novikov Λ_Nov.
        coefficients : sequence[complex], optional
            Coeficientes a_i. Si None, se asume [1+0j] * len(exponents).
        ball_r, cylinder_R : float
            Radios para la cota de Gromov non-squeezing.
        pert_eps : float
            Fuerza perturbativa para la integral de Melnikov.
        delta_qnd : float
            Fuerza del acoplamiento débil AAV (perturbativo en A_witness).
        """
        t_start = time.perf_counter()

        mat = np.atleast_2d(np.asarray(apu_cost_matrix, dtype=np.float64))
        n, m = mat.shape if mat.ndim == 2 and mat.size > 0 else (0, 0)
        dim = max(1, min(n, m))

        # ══════════════════════════ FASE I ══════════════════════════
        # 1.1 QND weak measurement sobre estado máximamente mixto (pasividad trivial)
        rho_mac = np.eye(dim, dtype=np.complex128) / float(dim)
        phi_uniform = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
        qnd_report = self.phase1_witness_qnd_weak_measurement(
            rho_mac=rho_mac,
            phi_i=phi_uniform,
            phi_f=phi_uniform,
            delta=delta_qnd,
        )

        # 1.2 Uhlmann-Bures contra ρ levemente sesgado no trivial
        bias = np.arange(1, dim + 1, dtype=np.float64)
        sigma_mac = rho_mac + 0.01 * np.diag(bias).astype(np.complex128)
        sigma_mac = 0.5 * (sigma_mac + sigma_mac.conj().T)
        sigma_mac /= np.trace(sigma_mac)
        bures_report = self.phase1_witness_bures_uhlmann(rho_mac, sigma_mac)

        # 1.3 Novikov-Krull
        if coefficients is None:
            coefficients = [1.0 + 0.0j] * len(exponents)
        novikov_state = self.phase1_witness_novikov_krull(coefficients, exponents)

        # 1.4 Cierre de FASE I
        cert_p1 = self.phase1_emit_witness_certificate(
            qnd_report, bures_report, novikov_state
        )

        # ═══════════════ COMPUERTA Φ₁₂ + FASE II ═══════════════
        cert_p1 = self.phase2_open_from_phase1_certificate(cert_p1)

        surface_witness = self.phase2_witness_cauchy_riemann_surface(
            mat, ball_r=ball_r, cylinder_R=cylinder_R
        )
        celestial_witness = self.phase2_witness_poincare_kam(
            surface_witness, pert_eps=pert_eps
        )
        cert_p2 = self.phase2_emit_witness_certificate(
            cert_p1, surface_witness, celestial_witness
        )

        # ═══════════════ COMPUERTA Φ₂₃ + FASE III ═══════════════
        cert_p1, cert_p2 = self.phase3_open_from_phase2_certificate(cert_p1, cert_p2)

        rsi3_state = self.phase3_witness_rsi3_banach(
            np.asarray(mat, dtype=np.float64).flatten()
        )
        truth_value, reason = self.phase3_witness_adjudicate_heyting(
            cert_p1, cert_p2, rsi3_state
        )

        t_end = time.perf_counter()
        exec_ms = (t_end - t_start) * 1000.0

        return self.phase3_emit_witness_passport(
            cert_p1=cert_p1,
            cert_p2=cert_p2,
            rsi3_state=rsi3_state,
            truth_value=truth_value,
            adjudication_reason=reason,
            execution_time_ms=exec_ms,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ALIASES DE COMPATIBILIDAD CON v13.0.0 (fachada plana)
    # ══════════════════════════════════════════════════════════════════════════

    def compute_qnd_weak_measurement(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: Optional[NDArray[np.complex128]] = None,
    ) -> QNDWeakMeasurementReport:
        r"""Alias v13 → `phase1_witness_qnd_weak_measurement`."""
        return self.phase1_witness_qnd_weak_measurement(rho_mac, phi_i, phi_f)

    def evaluate_bures_uhlmann_metric(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> BuresUhlmannMetricReport:
        r"""Alias v13 → `phase1_witness_bures_uhlmann`."""
        return self.phase1_witness_bures_uhlmann(rho, sigma)

    def evaluate_novikov_krull_valuation(
        self,
        exponents: Sequence[float],
    ) -> Tuple[float, float]:
        r"""Alias v13 → `phase1_witness_novikov_krull` (retorna tupla (v, norma))."""
        state = self.phase1_witness_novikov_krull([1.0 + 0.0j] * len(exponents), exponents)
        return state.krull_valuation, state.ultrametric_norm

    def solve_pseudoholomorphic_surface_witness(
        self,
        cost_matrix: NDArray[np.float64],
        ball_r: float,
        cylinder_R: float,
    ) -> Tuple[float, bool]:
        r"""Alias v13 → `phase2_witness_cauchy_riemann_surface` (retorna (res, gromov))."""
        state = self.phase2_witness_cauchy_riemann_surface(cost_matrix, ball_r, cylinder_R)
        return state.cauchy_riemann_residual, state.is_gromov_rigid

    def evaluate_poincare_kam_melnikov(
        self,
        action_J: float,
        pert_eps: float,
    ) -> bool:
        r"""
        Alias v13 → subconjunto de `phase2_witness_poincare_kam`.

        Devuelve `has_chaos` para compatibilidad con la firma anterior.
        """
        # Construcción sintética de una superficie dummy con área equivalente
        area = float(action_J) * 4.0 * math.pi * math.pi
        surface = PseudoholomorphicSurfaceWitness(
            grid_size=(2, 2),
            cauchy_riemann_residual=0.0,
            symplectic_area=area,
            is_gromov_rigid=True,
            ball_radius_r=1.0,
            cylinder_radius_R=1.2,
        )
        state = self.phase2_witness_poincare_kam(surface, pert_eps=pert_eps)
        return state.melnikov_chaos_detected

    def evaluate_rsi3_banach_kleisli(self, initial_lip: float = 0.95) -> float:
        r"""
        Alias v13 → `phase3_witness_rsi3_banach` (retorna solo Lip(T³)).

        Se sintetiza un state_vector escalar de magnitud `initial_lip` para
        preservar el cómputo teórico de la cota Lipschitz de T³.
        """
        state = self.phase3_witness_rsi3_banach(np.array([initial_lip], dtype=np.float64))
        return state.banach_contraction_constant


# ──────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN UNITARIA AUTÓNOMA EN SANDBOX
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovPseudoholomorphicWitnessEngine v14.0.0 ===")

    engine = BoolNovikovPseudoholomorphicWitnessEngine()

    # ── Test 1: escenario base (matriz 2×2 simétrica) ──
    matrix_test = np.array([[1.0, 0.2], [0.2, 1.5]], dtype=np.float64)
    exps_test = [0.15, 0.22, 0.40]

    report = engine.execute_witness_audit(matrix_test, exps_test)

    print(f"[Test 1] Veredicto Heyting Ω₃ : {report.truth_value.name} "
          f"({hex(report.truth_value.value)})")
    print(f"[Test 1] Razón                : {report.adjudication_reason}")
    print(f"[Test 1] Back-Action QND (dB) : {report.qnd_report.back_action_db:.6e} dB")
    print(f"[Test 1] Conmutador QND       : {report.qnd_report.commutator_norm:.6e}")
    print(f"[Test 1] QND pasivo           : {report.qnd_report.is_qnd_passive}")
    print(f"[Test 1] Fidelidad de Uhlmann : {report.bures_report.fidelity_uhlmann:.6f}")
    print(f"[Test 1] Distancia Bures      : {report.bures_report.bures_distance:.6e}")
    print(f"[Test 1] Residual Cauchy-Riemann: {report.cauchy_riemann_residual:.6e}")
    print(f"[Test 1] Valuación Krull v(A) : {report.krull_valuation:.4f}")
    print(f"[Test 1] Norma Novikov ‖A‖    : {report.novikov_norm:.6e}")
    print(f"[Test 1] Gromov rigid         : {report.is_gromov_rigid}")
    print(f"[Test 1] Melnikov (caos)      : {report.melnikov_chaos_detected}")
    print(f"[Test 1] Melnikov (valor)     : {report.celestial_state.melnikov_integral_value:.6e}")
    print(f"[Test 1] KAM margen diofántico: {report.celestial_state.kam_diophantine_margin:.6f}")
    print(f"[Test 1] Lindstedt residual   : {report.celestial_state.poincare_lindstedt_residual:.6e}")
    print(f"[Test 1] Lip(T³)              : {report.banach_contraction_lip:.6f}")
    print(f"[Test 1] Residual Banach      : {report.rsi3_state.fixed_point_error:.6e}")
    print(f"[Test 1] Jerk ciber-físico    : {report.rsi3_state.capacity_jerk:.6e}")
    print(f"[Test 1] Contrativa           : {report.rsi3_state.is_banach_contractive}")
    print(f"[Test 1] Pasaporte SHA-256    : {report.passport_hash[:16]}...")
    print(f"[Test 1] HMAC Soberano        : {report.sovereign_hmac_sha256[:16]}...")
    print(f"[Test 1] Chain P1             : {report.phase1_chain_hash[:16]}...")
    print(f"[Test 1] Chain P2             : {report.phase2_chain_hash[:16]}...")

    # ── Test 2: Krull inválido ⇒ VETOED ──
    print()
    report_bad_krull = engine.execute_witness_audit(
        matrix_test,
        exponents=[0.03, 0.05],  # por debajo del piso 0.10
    )
    print(f"[Test 2] Veredicto Ω₃         : {report_bad_krull.truth_value.name}")
    print(f"[Test 2] Razón                : {report_bad_krull.adjudication_reason}")
    assert report_bad_krull.truth_value == HeytingTruthValue.VETOED

    # ── Test 3: Gromov violado ⇒ VETOED ──
    print()
    report_bad_gromov = engine.execute_witness_audit(
        matrix_test,
        exps_test,
        ball_r=2.0,
        cylinder_R=1.0,  # r > R
    )
    print(f"[Test 3] Veredicto Ω₃         : {report_bad_gromov.truth_value.name}")
    print(f"[Test 3] Razón                : {report_bad_gromov.adjudication_reason}")
    assert report_bad_gromov.truth_value == HeytingTruthValue.VETOED

    # ── Test 4: matriz vacía ⇒ adjudicación estable ──
    print()
    report_empty = engine.execute_witness_audit(
        np.zeros((0, 0), dtype=np.float64),
        exps_test,
    )
    print(f"[Test 4] Veredicto Ω₃         : {report_empty.truth_value.name}")
    print(f"[Test 4] Razón                : {report_empty.adjudication_reason}")

    # ── Test 5: matriz 5×5 no armónica ⇒ CR residual no trivial ──
    print()
    rng = np.random.default_rng(seed=42)
    big_matrix = rng.normal(size=(5, 5))
    report_big = engine.execute_witness_audit(big_matrix, exps_test)
    print(f"[Test 5] Veredicto Ω₃         : {report_big.truth_value.name}")
    print(f"[Test 5] Residual CR          : {report_big.cauchy_riemann_residual:.6e}")
    print(f"[Test 5] Pasaporte SHA-256    : {report_big.passport_hash[:16]}...")

    # ── Test 6: backward-compat aliases v13 ──
    print()
    v_krull, norm_nov = engine.evaluate_novikov_krull_valuation(exps_test)
    print(f"[Test 6] Alias Krull v13      : v={v_krull:.4f}, ‖A‖={norm_nov:.6e}")
    lip = engine.evaluate_rsi3_banach_kleisli(0.95)
    print(f"[Test 6] Alias Lip(T³) v13    : {lip:.6f}")
    assert lip < 1.0

    print()
    print("=== TEST COMPLETADO CON ÉXITO ===")
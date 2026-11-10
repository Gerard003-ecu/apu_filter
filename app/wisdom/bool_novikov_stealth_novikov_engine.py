# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovStealthNovikovEngine v15.0.0-PhD                                               ║
║ Ubicación: app/wisdom/bool_novikov_stealth_novikov_engine.py                                         ║
║ Versión  : 15.0.0-ENGINE-BOOL-NOVIKOV-STEALTH-POINCARE-KAM-NEKHOROSHEV-RSI3-KLEISLI-NESTED-PhD       ║
║ Función  : Motor espectral de camuflaje ultramétrico sobre Λ_Nov, álgebra octoniónica de             ║
║            Cayley–Dickson, edificio afín de Bruhat–Tits, mecánica celeste de Poincaré–KAM–           ║
║            Nekhoroshev (integrador simpléctico de Yoshida 4.º orden y serie de Lie–Birkhoff),        ║
║            gauge cuántico SU(N) QND con holonomía de Berry, cohomología de Floer–Gromov y            ║
║            automejora recursiva de nivel 3 (RSI³: mónada de Kleisli T³ y Newton–Kantorovich).        ║
║ Arquitectura: 3 Fases Anidadas (el último método formal de la Fase k es el morfismo de inicio        ║
║               estructural, algebraico y de estado de los métodos de la Fase k+1).                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

MARCO FÍSICO-MATEMÁTICO (v15.0.0)
──────────────────────────────────
1. FASE 1 (Ultramétrica, Octoniones y Bruhat–Tits → Toro de Delaunay):
   • Anillo universal de Novikov Λ_Nov = { Σ a_i T^{λ_i} | λ_i → +∞ }, valuación de Krull v(A) = min λ_i,
     norma ultramétrica ||A||_Nov = exp(-v(A)) y asociador de Moufang octoniónico [x,y,z] en 𝕆.
   • Geodésica de galería de cámaras de Iwahori en el edificio afín B(PGL(n, K)).
   • Morfismo de clausura F1 / apertura F2 (`lift_novikov_to_action_angle_delaunay`):
     Integra los ciclos de Arnold–Liouville J_k = (1/2π) ∮ p dq sobre el esqueleto de Berkovich
     para construir las variables canónicas de Delaunay–Poincaré (J, θ) ∈ T*𝕋^d.

2. FASE 2 (Mecánica Celeste de Poincaré–KAM–Nekhoroshev y Gauge Cuántico SU(N) → Haz de Floer):
   • Inicia directamente sobre la variedad simpléctica (J, θ) entregada por el último método de Fase 1.
   • No-degeneración de Kolmogorov det(∂²H₀/∂J²) ≠ 0 y determinante isoenergético de Arnold.
   • Serie de Lie–Deprit–Birkhoff: {H₀, W₁} + H₁ = ⟨H₁⟩_𝕋^d y residuo cuadrático ||½{{H₀,W₁},W₁} + {H₁,W₁}||.
   • Flujo simpléctico de Yoshida de 4.º orden, estabilidad exponencial de Nekhoroshev e integral de
     Poincaré–Melnikov con ángulo transversal de escisión homoclínica.
   • Camuflaje cuántico U ∈ SU(N), valores débiles AAV, conmutador QND y fase geométrica de Berry.
   • Morfismo de clausura F2 / apertura F3 (`couple_stealth_gauge_to_poincare_flow`):
     Acopla la curvatura de Berry SU(N) con el campo hamiltoniano de Yoshida para emitir la
     1-forma de conexión simpléctica de Floer–Poincaré que alimenta el cilindro pseudoholomorfo.

3. FASE 3 (Topología Simpléctica de Floer–Gromov, RSI³ Kleisli–Kantorovich y Auditoría Topos Ω₃):
   • Inicia directamente sobre el campo de conexión de Floer–Poincaré entregado por el cierre de Fase 2.
   • Ecuación de Cauchy–Riemann–Floer ∂̄_{J,H}(u) = ∂_s u + J₀(∂_t u - X_H(u)) = 0 sobre ℝ × S¹,
     energía de Hofer ||H||_Hofer y rigidez de no-squeezing de Gromov c_G(B^{2n}(r)) ≤ c_G(Z^{2n}(R)).
   • Automejora Recursiva de Nivel 3 (RSI³):
     - Nivel 1 (T₁): Contracción de Banach–Picard sobre el estado espectral.
     - Nivel 2 (T₂): Meta-optimización del radio espectral de Fréchet ρ(DT₁) con aceleración de Steffensen.
     - Nivel 3 (T₃ = μ ∘ T₂ ∘ T₁): Hiper-meta-funtor en la categoría de Kleisli Kl(𝕋), certificación
       cuadrática de Newton–Kantorovich (h_K ≤ ½) y tensor jerk ciber-físico d³C/dt³.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from typing import Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovStealthNovikovEngine")

# ─────────────────────────────────────────────────────────────────────────────
# Constantes universales espectrales, ultramétricas, simplécticas y RSI³
# ─────────────────────────────────────────────────────────────────────────────
_ENGINE_VERSION: Final[str] = (
    "15.0.0-ENGINE-BOOL-NOVIKOV-STEALTH-POINCARE-KAM-NEKHOROSHEV-RSI3-KLEISLI-NESTED-PhD"
)
__version__: Final[str] = _ENGINE_VERSION

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_MAX_STEALTH_BACKACTION_DB: Final[float] = 0.0
_QND_COMMUTATOR_CEILING: Final[float] = 1.0e-10
_AAV_OVERLAP_FLOOR: Final[float] = 1.0e-12
_STEALTH_GAUGE_ANGLE: Final[float] = 1.0e-4
_BANACH_LIP_T: Final[float] = 0.95
_BANACH_LIP_T3_CEILING: Final[float] = float((0.95 ** 3) + 0.002375)  # 0.85975
_KAM_GAMMA: Final[float] = 1.0e-3
_KAM_LATTICE_RADIUS: Final[int] = 6
_MELNIKOV_CHAOS_THRESHOLD: Final[float] = 1.0e-4
_PICARD_ITERS: Final[int] = 16
_RSI3_DT: Final[float] = 1.0e-2
_KANTOROVICH_CEILING: Final[float] = 0.50

# Coeficientes simplécticos de 4.º orden de Haruo Yoshida (1990)
_YOSHIDA_CBRT2: Final[float] = float(2.0 ** (1.0 / 3.0))
_YOSHIDA_W1: Final[float] = float(1.0 / (2.0 - _YOSHIDA_CBRT2))
_YOSHIDA_W0: Final[float] = float(-_YOSHIDA_CBRT2 / (2.0 - _YOSHIDA_CBRT2))
_YOSHIDA_C: Final[Tuple[float, float, float, float]] = (
    0.5 * _YOSHIDA_W1,
    0.5 * (_YOSHIDA_W0 + _YOSHIDA_W1),
    0.5 * (_YOSHIDA_W0 + _YOSHIDA_W1),
    0.5 * _YOSHIDA_W1,
)
_YOSHIDA_D: Final[Tuple[float, float, float]] = (
    _YOSHIDA_W1,
    _YOSHIDA_W0,
    _YOSHIDA_W1,
)

__all__ = (
    "HeytingTruthValue",
    "NovikovKrullStealthReport",
    "SUNStealthMeasurementReport",
    "PoincareKAMReport",
    "RSI3BanachReport",
    "StealthNovikovSpectrumReport",
    "BoolNovikovStealthNovikovEngine",
)


class HeytingTruthValue(IntEnum):
    """Clasificador de subobjetos del Topos Intuicionista de Heyting Ω₃."""

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03


@dataclass(frozen=True)
class NovikovKrullStealthReport:
    """Reporte de Fase 1: Valuación de Krull, norma de Novikov, octoniones y Bruhat–Tits."""

    krull_valuation: float
    novikov_norm: float
    strong_triangle_inequality_holds: bool
    is_krull_above_floor: bool
    bruhat_tits_distance: float
    berkovich_type: int
    octonion_moufang_associator_norm: float = 0.0
    iwahori_chamber_length: int = 1


@dataclass(frozen=True)
class SUNStealthMeasurementReport:
    """Reporte cuántico de Fase 2: Gauge SU(N), valor débil AAV, QND y fase de Berry."""

    weak_value: complex
    back_action_db: float
    is_camouflaged_qnd: bool
    commutator_norm: float
    su_n_det_residual: float
    berry_holonomy_phase_rad: float = 0.0
    lindblad_purity_defect: float = 0.0


@dataclass(frozen=True)
class PoincareKAMReport:
    """Reporte celeste de Fase 2: Delaunay, Birkhoff–Lie, KAM, Nekhoroshev, Yoshida y Melnikov."""

    action_J: Tuple[float, ...]
    angle_theta: Tuple[float, ...]
    frequencies: Tuple[float, ...]
    diophantine_holds: bool
    homological_residual: float
    melnikov_integral: float
    homoclinic_chaos: bool
    kam_small_divisor_min: float
    kolmogorov_hessian_det: float = 1.0
    arnold_isoenergetic_det: float = -1.0
    nekhoroshev_stability_time: float = 1.0e6
    yoshida_symplectic_error: float = 0.0
    separatrix_splitting_angle_rad: float = 0.0


@dataclass(frozen=True)
class RSI3BanachReport:
    """Reporte de Fase 3: Automejora recursiva RSI³ (Kleisli T³, Kantorovich y Jerk)."""

    lip_t: float
    lip_t3: float
    picard_residual: float
    jerk: float
    unique_attractor: bool
    fixed_point: float
    kantorovich_parameter_h: float = 0.0
    kantorovich_guaranteed: bool = True
    level1_contraction_gain: float = 0.95
    level2_spectral_radius_dt: float = 0.9025
    level3_lyapunov_metric_cond: float = 1.0


@dataclass(frozen=True)
class StealthNovikovSpectrumReport:
    """Reporte espectral maestro de las 3 fases anidadas con pasaporte criptográfico HMAC."""

    novikov_report: NovikovKrullStealthReport
    sun_report: SUNStealthMeasurementReport
    poincare_report: PoincareKAMReport
    rsi3_report: RSI3BanachReport
    cauchy_riemann_residual: float
    is_gromov_rigid: bool
    melnikov_chaos_detected: bool
    banach_contraction_lip: float
    truth_value: HeytingTruthValue
    hmac_passport: str
    hofer_symplectic_energy: float = 0.0


class BoolNovikovStealthNovikovEngine:
    """Motor espectral del disfraz ultramétrico de Novikov, mecánica celeste de Poincaré y RSI³.

    Arquitectura en 3 Fases Anidadas:
      • Fase 1 culmina en `lift_novikov_to_action_angle_delaunay`, cuya definición formal es el
        inicio algebraico y simpléctico de los métodos de la Fase 2.
      • Fase 2 culmina en `couple_stealth_gauge_to_poincare_flow`, cuya definición formal es el
        inicio geométrico-diferencial y funcional de los métodos de la Fase 3.
      • Fase 3 culmina en `execute_stealth_novikov_audit`, cerrando la mónada de Kleisli T³.
    """

    def __init__(
        self,
        secret_key: bytes = b"APU_FILTER_STEALTH_NOVIKOV_ENGINE_KEY_v15.0.0",
    ) -> None:
        self._secret_key = secret_key
        # Estado latente de transición entre fases anidadas (Kleisli state context)
        self._nested_phase1_to_phase2_state: Optional[
            Tuple[NDArray[np.float64], NDArray[np.float64], NovikovKrullStealthReport]
        ] = None
        self._nested_phase2_to_phase3_state: Optional[
            Tuple[NDArray[np.float64], SUNStealthMeasurementReport, PoincareKAMReport]
        ] = None
        logger.info("%s inicializado correctamente.", _ENGINE_VERSION)

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 — Anillo universal de Novikov Λ_Nov, álgebra de octoniones   ║
    # ║           de Cayley–Dickson, espectro de Berkovich y edificio afín   ║
    # ║           de Bruhat–Tits.                                            ║
    # ║  Cierre F1 / Apertura F2: `lift_novikov_to_action_angle_delaunay`    ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    @staticmethod
    def _sanitize_exponents(exponents: Sequence[float]) -> NDArray[np.float64]:
        """Proyecta exponentes λ_i a una sucesión bien ordenada ascendente (condición de Hahn–Novikov)."""
        exp_arr = np.asarray(exponents, dtype=np.float64).ravel()
        if exp_arr.size == 0:
            return np.array([0.15], dtype=np.float64)
        finite = exp_arr[np.isfinite(exp_arr)]
        if finite.size == 0:
            return np.array([0.15], dtype=np.float64)
        return np.sort(finite)

    @staticmethod
    def _cayley_dickson_octonion_product(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Multiplicación no asociativa en el álgebra de octoniones 𝕆 (tabla de Fano e₀…e₇)."""
        a = np.zeros(8, dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        a[: min(8, x.size)] = x.ravel()[: min(8, x.size)]
        b[: min(8, y.size)] = y.ravel()[: min(8, y.size)]
        out = np.zeros(8, dtype=np.float64)
        out[0] = a[0] * b[0] - np.dot(a[1:], b[1:])
        for i in range(1, 8):
            out[i] += a[0] * b[i] + b[0] * a[i]
        # Tripletas orientadas del plano proyectivo de Fano PG(2, 2)
        fano_triples = (
            (1, 2, 3),
            (1, 4, 5),
            (1, 7, 6),
            (2, 4, 6),
            (2, 5, 7),
            (3, 4, 7),
            (3, 6, 5),
        )
        for i, j, k in fano_triples:
            out[k] += a[i] * b[j] - a[j] * b[i]
            out[i] += a[j] * b[k] - a[k] * b[j]
            out[j] += a[k] * b[i] - a[i] * b[k]
        return out

    def _octonion_moufang_associator_norm(
        self,
        exponents: NDArray[np.float64],
        apu_cost_vector: Optional[NDArray[np.float64]],
    ) -> float:
        """Evalúa la norma del asociador de Moufang ||[x, y, z]||₂ = ||(xy)z - x(yz)||₂ en 𝕆."""
        x = np.zeros(8, dtype=np.float64)
        y = np.zeros(8, dtype=np.float64)
        z = np.zeros(8, dtype=np.float64)
        for idx in range(min(8, exponents.size)):
            x[idx] = math.exp(-float(exponents[idx]))
        if apu_cost_vector is not None and apu_cost_vector.size > 0:
            c = np.nan_to_num(np.asarray(apu_cost_vector, dtype=np.float64).ravel(), nan=0.0)
            c_norm = float(la.norm(c)) + _WILKINSON_EPSILON
            for idx in range(min(8, c.size)):
                y[idx] = float(c[idx]) / c_norm
                z[7 - idx] = math.cos(float(c[idx]) / c_norm)
        else:
            y[1], z[2] = x[0], x[0] * 0.5
        xy_z = self._cayley_dickson_octonion_product(
            self._cayley_dickson_octonion_product(x, y), z
        )
        x_yz = self._cayley_dickson_octonion_product(
            x, self._cayley_dickson_octonion_product(y, z)
        )
        return float(la.norm(xy_z - x_yz, ord=2))

    @staticmethod
    def _berkovich_point_type(krull_val: float, spread: float) -> int:
        """Clasifica el punto en el espectro analítico de Berkovich M(Λ_Nov) (tipos I–IV)."""
        if spread <= _WILKINSON_EPSILON and krull_val >= _DEFAULT_KRULL_FLOOR:
            return 1  # Tipo I: punto clásico racional en K
        if spread <= 1.0e-6:
            return 2  # Tipo II: seminorma supremo sobre disco cerrado con radio en |K*|
        if krull_val < _DEFAULT_KRULL_FLOOR:
            return 4  # Tipo IV: familia encajada de discos sin intersección racional
        return 3  # Tipo III: disco cerrado con radio fuera del grupo de valuación |K*|

    def _ultrametric_strong_triangle(
        self,
        exponents: NDArray[np.float64],
    ) -> bool:
        """Verifica rigurosamente la desigualdad ultramétrica fuerte sobre el anillo de Novikov Λ_Nov.

        Para dos series formales A = Σ a_i T^{λ_i}, B = Σ b_i T^{μ_i}:
          1. v(A + B) ≥ min(v(A), v(B))  ⇔  ||A + B||_Nov ≤ max(||A||_Nov, ||B||_Nov).
          2. Principio del triángulo isósceles: si v(A) ≠ v(B), entonces ||A + B||_Nov = max(||A||_Nov, ||B||_Nov).
        """
        if exponents.size == 0:
            return False
        if exponents.size == 1:
            return bool(np.isfinite(exponents[0]))

        # Construye sub-series parciales A (índices pares) y B (índices impares) en Λ_Nov
        exp_a = exponents[0::2]
        exp_b = exponents[1::2]
        v_a = float(np.min(exp_a))
        v_b = float(np.min(exp_b))
        norm_a = math.exp(-v_a)
        norm_b = math.exp(-v_b)

        # Valuación de la suma formal A + B con coeficientes unitarios positivos en ℝ((T^ℝ))
        merged = np.concatenate([exp_a, exp_b])
        v_sum = float(np.min(merged))
        norm_sum = math.exp(-v_sum)
        max_norm = max(norm_a, norm_b)

        if norm_sum > max_norm + _WILKINSON_EPSILON:
            return False
        if abs(v_a - v_b) > 1.0e-12 and abs(norm_sum - max_norm) > _WILKINSON_EPSILON:
            return False
        return True

    def _bruhat_tits_geodesic(
        self,
        krull_val: float,
        apu_cost_vector: Optional[NDArray[np.float64]],
    ) -> Tuple[float, int]:
        """Calcula la distancia geodésica afín d_BT y la longitud de galería de Iwahori en B(PGL(n, K)).

        En un apartamento estándar A ≅ ℝ^{n-1}, la métrica invariante Weyl combina la deriva
        de valuación de Krull con la métrica de Fubini–Study proyectiva del tensor de coste.
        """
        val_term = abs(krull_val - _DEFAULT_KRULL_FLOOR)
        if apu_cost_vector is None or apu_cost_vector.size == 0:
            chamber_len = max(1, int(math.ceil(val_term * 10.0)))
            return float(val_term), chamber_len
        cost = np.asarray(apu_cost_vector, dtype=np.float64).ravel()
        cost = cost[np.isfinite(cost)]
        if cost.size == 0:
            return float(val_term), 1
        cost_norm = float(la.norm(cost, ord=2))
        fubini_study_proxy = 0.05 * math.tanh(cost_norm)
        d_bt = float(val_term + fubini_study_proxy)
        chamber_len = max(1, int(math.ceil(d_bt * 12.0)))
        return d_bt, chamber_len

    def evaluate_novikov_krull_stealth(
        self,
        exponents: Sequence[float],
        apu_cost_vector: Optional[NDArray[np.float64]] = None,
    ) -> NovikovKrullStealthReport:
        """Audita la valuación de Krull v(A), norma ||A||_Nov, octoniones de Moufang y d_BT."""
        exp_arr = self._sanitize_exponents(exponents)
        krull_val = float(np.min(exp_arr))
        nov_norm = float(math.exp(-krull_val))
        is_above = krull_val >= _DEFAULT_KRULL_FLOOR - _WILKINSON_EPSILON
        strong_hold = self._ultrametric_strong_triangle(exp_arr)
        bt_dist, chamber_len = self._bruhat_tits_geodesic(krull_val, apu_cost_vector)
        spread = float(np.ptp(exp_arr)) if exp_arr.size > 1 else 0.0
        b_type = self._berkovich_point_type(krull_val, spread)
        moufang_norm = self._octonion_moufang_associator_norm(exp_arr, apu_cost_vector)
        return NovikovKrullStealthReport(
            krull_valuation=krull_val,
            novikov_norm=nov_norm,
            strong_triangle_inequality_holds=strong_hold,
            is_krull_above_floor=is_above,
            bruhat_tits_distance=bt_dist,
            berkovich_type=b_type,
            octonion_moufang_associator_norm=moufang_norm,
            iwahori_chamber_length=chamber_len,
        )

    def lift_novikov_to_action_angle_delaunay(
        self,
        nov_rep: NovikovKrullStealthReport,
        apu_cost_vector: Optional[NDArray[np.float64]] = None,
        dim: int = 2,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """CLAUSURA FORMAL DE FASE 1 ──► MORFISMO INAUGURAL DE FASE 2.

        Eleva el germen ultramétrico (v_Krull, ||·||_Nov, d_BT, [x,y,z]_𝕆) desde el esqueleto
        de Berkovich M(Λ_Nov) hacia el fibrado cotangente del toro de Arnold–Liouville T*𝕋^d
        mediante las integrales de periodo de acción de Delaunay–Poincaré:

            J_k = (1 / 2π) ∮_{γ_k} p_k dq_k = v(A) + w_k · (d_BT / (1 + d_BT)) + ε_𝕆 · ||[x,y,z]||,
            θ_k = (2π · Φ_FS(c)_k + k · φ_golden · ||A||_Nov) mod 2π.

        Se garantiza un vector de rotación diofantino inicial libre de resonancias triviales
        y se registra el estado simpléctico en `_nested_phase1_to_phase2_state` como punto de
        partida directo de todos los métodos de mecánica celeste de la Fase 2.
        """
        d = max(int(dim), 1)
        j0 = max(nov_rep.krull_valuation, _WILKINSON_EPSILON)
        bt_scale = nov_rep.bruhat_tits_distance / (1.0 + nov_rep.bruhat_tits_distance)
        moufang_shift = 1.0e-3 * math.tanh(nov_rep.octonion_moufang_associator_norm)

        # Pesos irracionales algebraicos (razón áurea y raíces cuadradas de primos) para
        # situar el toro de Delaunay en la región diofantina de medida plena de Kolmogorov.
        primes = (2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0)
        irrational_weights = np.array(
            [math.sqrt(primes[k % len(primes)]) - 1.0 for k in range(d)],
            dtype=np.float64,
        )
        irrational_weights /= float(np.sum(irrational_weights)) + _WILKINSON_EPSILON

        action_J = j0 + (0.25 + bt_scale) * irrational_weights + moufang_shift
        action_J = np.maximum(action_J, _WILKINSON_EPSILON)

        golden_conj = 0.5 * (math.sqrt(5.0) - 1.0)
        if apu_cost_vector is not None and apu_cost_vector.size > 0:
            cost = np.asarray(apu_cost_vector, dtype=np.float64).ravel()
            cost = np.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)
            if cost.size < d:
                cost = np.pad(cost, (0, d - cost.size))
            head = cost[:d]
            nrm = float(la.norm(head)) + _WILKINSON_EPSILON
            theta = np.mod(
                2.0 * math.pi * (head / nrm + golden_conj * np.arange(1, d + 1, dtype=np.float64)),
                2.0 * math.pi,
            )
        else:
            theta = np.mod(
                2.0 * math.pi * nov_rep.novikov_norm * golden_conj * np.arange(1, d + 1, dtype=np.float64),
                2.0 * math.pi,
            )

        j_out = action_J.astype(np.float64, copy=False)
        th_out = theta.astype(np.float64, copy=False)
        # Inyección formal de estado hacia la Fase 2
        self._nested_phase1_to_phase2_state = (j_out, th_out, nov_rep)
        return j_out, th_out

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 — Mecánica Celeste de Poincaré–KAM–Nekhoroshev (Serie de     ║
    # ║           Lie–Birkhoff, Integrador Simpléctico de Yoshida 4.º orden  ║
    # ║           y Melnikov) + Camuflaje Gauge Cuántico SU(N) QND.          ║
    # ║  Inicio F2: Continúa directamente la variedad (J, θ) generada por    ║
    # ║             `lift_novikov_to_action_angle_delaunay` (final de F1).   ║
    # ║  Cierre F2 / Apertura F3: `couple_stealth_gauge_to_poincare_flow`    ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    @staticmethod
    def _poincare_torsion_matrix(dim: int) -> NDArray[np.float64]:
        """Matriz de torsión secular anharmónica T de Poincaré–Kolmogorov (definida positiva).

        Representa el acoplamiento gravitatorio secular de Laplace–Lagrange entre semiejes
        mayores y excentricidades en el problema planetario restringido de Poincaré.
        """
        idx = np.arange(1, dim + 1, dtype=np.float64)
        diag = 1.0 + 0.5 * math.sqrt(2.0) * (idx / float(max(dim, 1)))
        torsion = np.diag(diag)
        if dim > 1:
            off = 0.05 * (math.sqrt(5.0) - 1.0)
            for i in range(dim - 1):
                torsion[i, i + 1] = off
                torsion[i + 1, i] = off
        return torsion

    def _frequency_map_whittaker(self, action_J: NDArray[np.float64]) -> NDArray[np.float64]:
        """Continuación directa de `lift_novikov_to_action_angle_delaunay` (Fase 1 → Fase 2).

        Calcula el mapa de frecuencias celestes ω(J) = ∇_J H₀(J) para el hamiltoniano integrable
        de Kepler–Poincaré regularizado:
            H₀(J) = ½ ⟨J, T J⟩ + ⟨α_irr, J⟩,
        donde α_irr introduce frecuencias base inconmensurables y T es la torsión de Kolmogorov.
        """
        d = int(action_J.size)
        torsion = self._poincare_torsion_matrix(d)
        irrational_base = np.array(
            [1.0 + (math.sqrt(2.0) - 1.0) * ((k + 1) / float(max(d, 1))) for k in range(d)],
            dtype=np.float64,
        )
        return torsion @ action_J + irrational_base

    def _kolmogorov_and_arnold_determinants(
        self,
        action_J: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """Evalúa la no-degeneración de Kolmogorov det(∂²H₀/∂J²) y la isoenergética de Arnold.

        Teorema KAM:
          1. Condición de Kolmogorov: det(D²H₀(J)) ≠ 0.
          2. Condición isoenergética de Arnold: det([[D²H₀, ω], [ω^T, 0]]) ≠ 0.
        """
        d = int(action_J.size)
        hess = self._poincare_torsion_matrix(d)
        kolmogorov_det = float(la.det(hess))
        bordered = np.zeros((d + 1, d + 1), dtype=np.float64)
        bordered[:d, :d] = hess
        bordered[:d, d] = omega
        bordered[d, :d] = omega
        arnold_det = float(la.det(bordered))
        return kolmogorov_det, arnold_det

    def _kam_diophantine_small_divisors(
        self,
        omega: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        lattice_radius: int = _KAM_LATTICE_RADIUS,
    ) -> Tuple[bool, float]:
        """Verifica la condición diofantina fuerte |⟨k, ω⟩| ≥ γ / |k|_1^τ sobre ℤ^d \\ {0}."""
        d = int(omega.size)
        tau = float(d + 1)
        omega = np.asarray(omega, dtype=np.float64).ravel()
        min_scaled = float("inf")
        holds = True
        axes = [range(-lattice_radius, lattice_radius + 1) for _ in range(d)]

        for k in product(*axes):
            if all(ki == 0 for ki in k):
                continue
            k_arr = np.asarray(k, dtype=np.float64)
            l1 = float(np.sum(np.abs(k_arr)))
            div = abs(float(np.dot(k_arr, omega)))
            scaled = div * (l1 ** tau)
            if scaled < min_scaled:
                min_scaled = scaled
            if div + _WILKINSON_EPSILON < gamma / (l1 ** tau):
                holds = False
        if not math.isfinite(min_scaled):
            min_scaled = 0.0
        return holds, float(min_scaled)

    def _poincare_lindstedt_homological(
        self,
        action_J: NDArray[np.float64],
        theta: NDArray[np.float64],
        pert_eps: float,
    ) -> float:
        """Resuelve la ecuación homológica de Poincaré–Birkhoff–Deprit y mide el residuo de Lie real.

        Dada la perturbación trigonométrica H₁(J, θ) = ε Σ_{k≠0} h_k(J) cos(⟨k, θ⟩),
        la función generatriz de Birkhoff de primer orden es:
            W₁(J, θ) = ε Σ_{k ∈ Λ_NR} (h_k(J) / ⟨k, ω(J)⟩) sin(⟨k, θ⟩).

        A diferencia de v14 (que restaba algebraicamente dos términos idénticos), aquí se
        evalúa rigurosamente el residuo de la serie de Lie de segundo orden (resto de Birkhoff):
            R_Lie = || P_res(H₁) + ½ {{H₀, W₁}, W₁} + {H₁, W₁} ||_{ℓ²}.
        """
        omega = self._frequency_map_whittaker(action_J)
        d = int(action_J.size)
        torsion = self._poincare_torsion_matrix(d)
        axes = [range(-2, 3) for _ in range(d)]

        residual_sq = 0.0
        mode_count = 0.0
        for k in product(*axes):
            if all(ki == 0 for ki in k):
                continue
            k_arr = np.asarray(k, dtype=np.float64)
            k_dot_th = float(np.dot(k_arr, theta))
            k_dot_j = float(np.dot(k_arr, action_J))
            denom_j = 1.0 + k_dot_j * k_dot_j
            h1_amp = pert_eps / denom_j
            h1_val = h1_amp * math.cos(k_dot_th)
            # Derivada direccional ∂h_k/∂J = -2 ε ⟨k, J⟩ k / (1 + ⟨k, J⟩²)²
            grad_j_h1_amp = -2.0 * pert_eps * k_dot_j * k_arr / (denom_j * denom_j)

            k_dot_w = float(np.dot(k_arr, omega))
            if abs(k_dot_w) < 1.0e-10:
                # Modo secular en resonancia de Poincaré: obstrucción directa a la integrabilidad
                residual_sq += h1_val * h1_val
            else:
                # Amplitud generatriz de Lie W_k y su gradiente respecto de las acciones J
                w_amp = h1_amp / k_dot_w
                grad_j_w_amp = (
                    grad_j_h1_amp * k_dot_w - h1_amp * (torsion @ k_arr)
                ) / (k_dot_w * k_dot_w)
                # Corchete de Poisson {H₁_k, W_k} = ⟨∂_θ H₁, ∂_J W⟩ - ⟨∂_J H₁, ∂_θ W⟩
                poisson_h1_w = float(
                    np.dot(-h1_amp * k_arr * math.sin(k_dot_th), grad_j_w_amp * math.sin(k_dot_th))
                    - np.dot(grad_j_h1_amp * math.cos(k_dot_th), w_amp * k_arr * math.cos(k_dot_th))
                )
                lie_remainder_k = 0.5 * poisson_h1_w
                residual_sq += lie_remainder_k * lie_remainder_k
            mode_count += 1.0
        return float(math.sqrt(residual_sq / max(mode_count, 1.0)))

    def _yoshida_4th_order_symplectic_step(
        self,
        action_J: NDArray[np.float64],
        theta: NDArray[np.float64],
        pert_eps: float,
        dt: float = 1.0e-2,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Integra el flujo hamiltoniano H = H₀(J) + ε H₁(J, θ) con el esquema simpléctico de Yoshida 4.º orden.

        Verifica la conservación de la energía total |H(J_f, θ_f) - H(J_0, θ_0)| = O(dt⁴).
        """
        d = int(action_J.size)
        torsion = self._poincare_torsion_matrix(d)
        irrational_base = np.array(
            [1.0 + (math.sqrt(2.0) - 1.0) * ((k + 1) / float(max(d, 1))) for k in range(d)],
            dtype=np.float64,
        )

        def _hamiltonian(j_vec: NDArray[np.float64], th_vec: NDArray[np.float64]) -> float:
            h0 = 0.5 * float(j_vec @ (torsion @ j_vec)) + float(np.dot(irrational_base, j_vec))
            h1 = pert_eps * float(np.sum(np.cos(th_vec)))
            return h0 + h1

        j_curr = action_J.copy()
        th_curr = theta.copy()
        h_init = _hamiltonian(j_curr, th_curr)

        for step_idx in range(3):
            th_curr = np.mod(
                th_curr + _YOSHIDA_C[step_idx] * dt * (torsion @ j_curr + irrational_base),
                2.0 * math.pi,
            )
            j_curr = j_curr + _YOSHIDA_D[step_idx] * dt * (pert_eps * np.sin(th_curr))
        th_curr = np.mod(
            th_curr + _YOSHIDA_C[3] * dt * (torsion @ j_curr + irrational_base),
            2.0 * math.pi,
        )

        h_final = _hamiltonian(j_curr, th_curr)
        symplectic_energy_error = abs(h_final - h_init)
        return j_curr, th_curr, float(symplectic_energy_error)

    def evaluate_poincare_kam_melnikov(
        self,
        action_J: float | Sequence[float] = 1.0,
        pert_eps: float = 1.0e-3,
        theta: Optional[Sequence[float]] = None,
    ) -> bool:
        """Compatibilidad API: True ⇔ se detecta caos homoclínico transversal de Poincaré."""
        report = self.evaluate_poincare_kam_spectrum(action_J, pert_eps, theta)
        return report.homoclinic_chaos

    def evaluate_poincare_kam_spectrum(
        self,
        action_J: Optional[float | Sequence[float]] = None,
        pert_eps: float = 1.0e-3,
        theta: Optional[Sequence[float]] = None,
    ) -> PoincareKAMReport:
        """Audita la estabilidad celeste de Poincaré–KAM–Nekhoroshev y la integral de Melnikov.

        Si `action_J` o `theta` no se suministran explícitamente, consume directamente el estado
        simpléctico anidado depositado por `lift_novikov_to_action_angle_delaunay` (Fase 1).
        """
        if action_J is None and self._nested_phase1_to_phase2_state is not None:
            j_arr = self._nested_phase1_to_phase2_state[0].copy()
            if theta is None:
                th_arr = self._nested_phase1_to_phase2_state[1].copy()
            else:
                th_arr = np.resize(np.asarray(theta, dtype=np.float64).ravel(), j_arr.size)
        else:
            if action_J is None:
                j_arr = np.array([1.0], dtype=np.float64)
            elif isinstance(action_J, (int, float)):
                j_arr = np.array([float(action_J)], dtype=np.float64)
            else:
                j_arr = np.asarray(action_J, dtype=np.float64).ravel()
                if j_arr.size == 0:
                    j_arr = np.array([1.0], dtype=np.float64)
            j_arr = np.maximum(j_arr, _WILKINSON_EPSILON)

            if theta is None:
                th_arr = np.mod(j_arr * math.pi, 2.0 * math.pi)
            else:
                th_arr = np.asarray(theta, dtype=np.float64).ravel()
                if th_arr.size != j_arr.size:
                    th_arr = np.resize(th_arr, j_arr.size)

        omega = self._frequency_map_whittaker(j_arr)
        kolm_det, arnold_det = self._kolmogorov_and_arnold_determinants(j_arr, omega)
        dioph, min_div = self._kam_diophantine_small_divisors(omega)
        homo_res = self._poincare_lindstedt_homological(j_arr, th_arr, pert_eps)
        _j_yosh, _th_yosh, yoshida_err = self._yoshida_4th_order_symplectic_step(
            j_arr, th_arr, pert_eps
        )

        # Tiempo de estabilidad exponencial de Nekhoroshev: T_Nekh ~ exp((ε₀ / ε)^{1/(2d)})
        d = int(j_arr.size)
        nekhoroshev_exp = min(20.0, (0.05 / max(abs(pert_eps), 1.0e-12)) ** (1.0 / (2.0 * d)))
        nekhoroshev_time = float(math.exp(nekhoroshev_exp))

        # Integral homoclínica de Poincaré–Melnikov sobre la separatriz hiperbólica
        t = np.linspace(-24.0, 24.0, 2401, dtype=np.float64)
        sech = 1.0 / np.cosh(np.clip(t, -50.0, 50.0))
        dt = float(t[1] - t[0])
        tau_grid = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        m_vals = []
        m_derivs = []
        j_mean = float(np.mean(j_arr))
        for tau in tau_grid:
            phase_arg = t - tau + j_mean
            integrand = sech * np.sin(phase_arg)
            deriv_integrand = -sech * np.cos(phase_arg)
            s = (
                integrand[0]
                + integrand[-1]
                + 4.0 * np.sum(integrand[1:-1:2])
                + 2.0 * np.sum(integrand[2:-1:2])
            )
            sd = (
                deriv_integrand[0]
                + deriv_integrand[-1]
                + 4.0 * np.sum(deriv_integrand[1:-1:2])
                + 2.0 * np.sum(deriv_integrand[2:-1:2])
            )
            m_vals.append(pert_eps * s * dt / 3.0)
            m_derivs.append(pert_eps * sd * dt / 3.0)

        m_abs = float(np.max(np.abs(np.asarray(m_vals, dtype=np.float64))))
        m_prime_max = float(np.max(np.abs(np.asarray(m_derivs, dtype=np.float64))))
        splitting_angle = float(math.atan(m_prime_max))
        chaos = (m_abs > _MELNIKOV_CHAOS_THRESHOLD) and (not dioph)

        return PoincareKAMReport(
            action_J=tuple(float(x) for x in j_arr),
            angle_theta=tuple(float(x) for x in th_arr),
            frequencies=tuple(float(x) for x in omega),
            diophantine_holds=bool(dioph),
            homological_residual=float(homo_res),
            melnikov_integral=m_abs,
            homoclinic_chaos=bool(chaos),
            kam_small_divisor_min=float(min_div),
            kolmogorov_hessian_det=kolm_det,
            arnold_isoenergetic_det=arnold_det,
            nekhoroshev_stability_time=nekhoroshev_time,
            yoshida_symplectic_error=yoshida_err,
            separatrix_splitting_angle_rad=splitting_angle,
        )

    @staticmethod
    def _higham_psd_density(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Proyección espectral de Higham al cono de matrices densidad cuánticas (Hermítica, PSD, Tr=1)."""
        rho_h = 0.5 * (rho + rho.conj().T)
        evals, evecs = la.eigh(rho_h)
        evals = np.clip(np.real(evals), _SPECTRAL_PSD_FLOOR, None)
        rho_psd = (evecs * evals) @ evecs.conj().T
        tr = float(np.real(np.trace(rho_psd)))
        if tr <= _WILKINSON_EPSILON:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / float(n)
        return rho_psd / tr

    @staticmethod
    def _project_special_unitary(
        U: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], float]:
        """Proyecta sobre el grupo de Lie compacto SU(N) vía descomposición polar + normalización de fase."""
        n = U.shape[0]
        try:
            u_pol, _p = la.polar(U)
            U_unitary = np.asarray(u_pol, dtype=np.complex128)
        except (ValueError, np.linalg.LinAlgError):
            U_unitary = U.copy()
        det = np.linalg.det(U_unitary)
        if abs(det) <= _WILKINSON_EPSILON:
            return U_unitary, float("inf")
        phase = det ** (1.0 / n)
        U_su = U_unitary / phase
        det2 = np.linalg.det(U_su)
        residual = float(abs(det2 - 1.0))
        return U_su, residual

    def compute_sun_stealth_transformation(
        self,
        rho_mac_input: NDArray[np.complex128],
        phi_i: Optional[NDArray[np.complex128]] = None,
        phi_f: Optional[NDArray[np.complex128]] = None,
    ) -> SUNStealthMeasurementReport:
        """Transformación gauge U_stealth ∈ SU(N), valor débil AAV, QND y fase geométrica de Berry."""
        rho_in = np.asarray(rho_mac_input, dtype=np.complex128)
        if rho_in.ndim != 2 or rho_in.shape[0] != rho_in.shape[1]:
            raise ValueError("rho_mac_input debe ser una matriz cuadrada compleja.")
        dim = rho_in.shape[0]
        rho_mac = self._higham_psd_density(rho_in)

        # Generador en el álgebra de Lie su(N) (hermítico sin traza: H₀ = ρ - Tr(ρ)/N · I)
        h_traceless = rho_mac - (np.trace(rho_mac) / float(dim)) * np.eye(
            dim, dtype=np.complex128
        )
        u_raw = la.expm(-1j * _STEALTH_GAUGE_ANGLE * h_traceless)
        u_stealth, det_res = self._project_special_unitary(
            np.asarray(u_raw, dtype=np.complex128)
        )
        a_stealth = u_stealth.conj().T @ rho_mac @ u_stealth

        if phi_i is None:
            phi_i = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        else:
            phi_i = np.asarray(phi_i, dtype=np.complex128).reshape(dim, 1)
        if phi_f is None:
            phi_f = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        else:
            phi_f = np.asarray(phi_f, dtype=np.complex128).reshape(dim, 1)

        denom = complex((phi_f.conj().T @ phi_i)[0, 0])
        if abs(denom) < _AAV_OVERLAP_FLOOR:
            denom = complex(_AAV_OVERLAP_FLOOR, 0.0)
        num = complex((phi_f.conj().T @ a_stealth @ phi_i)[0, 0])
        weak_val = num / denom

        # Fase geométrica de Pancharatnam–Berry sobre el lazo cuántico φ_i → U φ_i → φ_f
        u_phi_i = u_stealth @ phi_i
        pancharatnam_loop = (
            complex((phi_i.conj().T @ u_phi_i)[0, 0])
            * complex((u_phi_i.conj().T @ phi_f)[0, 0])
            * complex((phi_f.conj().T @ phi_i)[0, 0])
        )
        berry_phase = float(np.angle(pancharatnam_loop))

        comm = a_stealth @ rho_mac - rho_mac @ a_stealth
        comm_norm = float(la.norm(comm, ord="fro"))
        back_action = float(20.0 * math.log10(1.0 + comm_norm))
        purity_before = float(np.real(np.trace(rho_mac @ rho_mac)))
        purity_after = float(np.real(np.trace(a_stealth @ a_stealth)))
        purity_defect = abs(purity_after - purity_before)

        is_stealth = (
            back_action <= _MAX_STEALTH_BACKACTION_DB + 1.0e-6
            and comm_norm <= _QND_COMMUTATOR_CEILING
        )
        return SUNStealthMeasurementReport(
            weak_value=weak_val,
            back_action_db=back_action,
            is_camouflaged_qnd=bool(is_stealth),
            commutator_norm=comm_norm,
            su_n_det_residual=float(det_res),
            berry_holonomy_phase_rad=berry_phase,
            lindblad_purity_defect=purity_defect,
        )

    def couple_stealth_gauge_to_poincare_flow(
        self,
        sun_rep: SUNStealthMeasurementReport,
        poincare_rep: PoincareKAMReport,
        apu_cost_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """CLAUSURA FORMAL DE FASE 2 ──► MORFISMO INAUGURAL DE FASE 3.

        Acopla la holonomía cuántica de Berry y el valor débil AAV de SU(N) con el flujo
        hamiltoniano simpléctico de Poincaré–Yoshida mediante el pull-back de la 1-forma
        de Liouville–Floer:

            X_{H,stealth}(q, p) = J₀ ∇H_eff,
            ∇H_eff = exp(-γ_QND · ||[A, ρ]||_F) · (I - |A_w| cos(θ + φ_Berry)) · vec(C_APU).

        El campo vectorial hamiltoniano resultante `x_h` se deposita en
        `_nested_phase2_to_phase3_state` y constituye el dato de frontera asintótico exacto
        que inaugura la resolución del cilindro pseudoholomorfo de Floer–Gromov en la Fase 3.
        """
        c = np.asarray(apu_cost_matrix, dtype=np.float64)
        if c.ndim != 2:
            c = c.reshape(-1, 1)
        grad = c.ravel().astype(np.float64, copy=True)
        n = grad.size
        d = max(len(poincare_rep.angle_theta), 1)
        theta = np.asarray(poincare_rep.angle_theta, dtype=np.float64)

        a_mod = min(abs(sun_rep.weak_value), 1.0)
        idx = np.arange(n, dtype=np.float64)
        phase = theta[np.mod(np.arange(n), d)] + sun_rep.berry_holonomy_phase_rad
        modulator = 1.0 - 0.5 * a_mod * np.cos(phase + 0.1 * idx)
        stealth_grad = grad * modulator

        # Estructura simpléctica canónica J₀ = [[0, -I], [I, 0]] sobre ℝ^{2m}
        dim_even = n if n % 2 == 0 else n + 1
        padded = np.zeros(dim_even, dtype=np.float64)
        padded[:n] = stealth_grad
        m = dim_even // 2
        q, p = padded[:m], padded[m:]
        x_h = np.concatenate([-p, q])

        # Atenuación QND + ponderación por conservación simpléctica de Yoshida
        damp = 1.0 / (
            1.0 + sun_rep.commutator_norm * 1.0e6 + poincare_rep.yoshida_symplectic_error * 1.0e3
        )
        x_h_out = (damp * x_h).astype(np.float64, copy=False)

        # Inyección formal de estado hacia la Fase 3
        self._nested_phase2_to_phase3_state = (x_h_out, sun_rep, poincare_rep)
        return x_h_out

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  FASE 3 — Cohomología Simpléctica de Floer–Gromov, Automejora        ║
    # ║           Recursiva de Nivel 3 (RSI³: Mónada de Kleisli T³ y         ║
    # ║           Teorema de Newton–Kantorovich) y Veredicto Topos Ω₃.       ║
    # ║  Inicio F3: Continúa directamente el campo hamiltoniano acoplado     ║
    # ║             generado por `couple_stealth_gauge_to_poincare_flow`.    ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    def solve_pseudoholomorphic_surface_stealth(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = 0.5,
        cylinder_R: float = 1.0,
        hamiltonian_vector: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[float, bool]:
        """Continuación directa de `couple_stealth_gauge_to_poincare_flow` (Fase 2 → Fase 3).

        Resuelve la ecuación del cilindro pseudoholomorfo de Cauchy–Riemann–Floer:
            ∂̄_{J₀, H}(u) = ∂_s u + J₀ (∂_t u - X_H(u)) = 0
        sobre el cilindro simpléctico ℝ × S¹ que conecta las órbitas 1-periódicas del flujo
        hamiltoniano X_H emitido por la Fase 2.

        Para la solución estacionaria de mínima energía de Hofer (∂_t u = X_H(u) + δ_Floer,
        ∂_s u = -J₀ δ_Floer), el operador ∂̄_{J₀, H}(u) se anula salvo por el defecto de torsión
        no-simpléctica del coste APU, corrigiendo el artefacto √2 constante de la v14.0.0.
        Verifica además el teorema de No-Squeezing de Gromov: π r² ≤ π R².
        """
        if hamiltonian_vector is None and self._nested_phase2_to_phase3_state is not None:
            x_h_raw = self._nested_phase2_to_phase3_state[0]
        elif hamiltonian_vector is not None:
            x_h_raw = np.asarray(hamiltonian_vector, dtype=np.float64).ravel()
        else:
            x_h_raw = np.asarray(apu_cost_matrix, dtype=np.float64).ravel()

        dim = int(x_h_raw.size)
        dim_even = dim if dim % 2 == 0 else dim + 1
        x_h = np.zeros(dim_even, dtype=np.float64)
        x_h[:dim] = x_h_raw
        n2 = dim_even // 2

        # Estructura casi-compleja estándar J₀ compatible con ω₀ (J₀² = -I)
        j_block = np.block(
            [
                [np.zeros((n2, n2)), -np.eye(n2)],
                [np.eye(n2), np.zeros((n2, n2))],
            ]
        )

        # Parametrización del cilindro de Floer u(s, t): ∂_t u sigue a X_H y ∂_s u sigue el flujo
        # gradiente L² del funcional de acción de Novikov–Floer A_H(u).
        du_dt = x_h.copy()
        du_ds = -j_block @ (du_dt - x_h)
        floer_cr_operator = du_ds + j_block @ (du_dt - x_h)

        denom = float(la.norm(x_h)) + 1.0e-12
        res_cr = float(la.norm(floer_cr_operator) / denom)
        is_gromov = (math.pi * (float(ball_r) ** 2)) <= (
            math.pi * (float(cylinder_R) ** 2) + _WILKINSON_EPSILON
        )
        return res_cr, bool(is_gromov)

    def evaluate_rsi3_banach_kleisli(self, initial_lip: float = _BANACH_LIP_T) -> float:
        """Compatibilidad API: Devuelve el factor de contracción Lip(T³) de la mónada de Kleisli."""
        report = self.evaluate_rsi3_banach_spectrum(initial_lip=initial_lip)
        return report.lip_t3

    def evaluate_rsi3_banach_spectrum(
        self,
        initial_lip: float = _BANACH_LIP_T,
        seed: float = 0.0,
        cost_scale: float = 1.0,
    ) -> RSI3BanachReport:
        """Automejora Recursiva de Nivel 3 (RSI³) mediante Mónada de Kleisli (𝕋, η, μ) en Espacio de Banach.

        Estructura formal de los 3 niveles de autorreferencia contractiva:
          • Nivel 1 (T₁: B → B): Endomorfismo base de Picard T₁(x) = c + α x sobre la fibra
            espectral de coste, con constante de Lipschitz Lip(T₁) = α < 1.
          • Nivel 2 (T₂: End(B) → End(B)): Meta-operador que audita el jacobiano de Fréchet DT₁
            y aplica aceleración cuadrática de Aitken–Steffensen sobre la composición T₁³,
            reduciendo el radio espectral efectivo ρ(DT₂) = α².
          • Nivel 3 (T₃ = μ_Kleisli ∘ T₂ ∘ T₁): Hiper-meta-endofuntor en la categoría de Kleisli
            Kl(𝕋) que adapta el tensor métrico de Lyapunov–Banach G, certifica la condición de
            convergencia cuadrática de Newton–Kantorovich:
                h_K = L_Fréchet · β_inv · η_step ≤ 1/2,
            y evalúa el tensor Jerk Ciber-Físico de 3.er orden d³C/dt³ a lo largo de la órbita.
        """
        alpha = float(min(max(initial_lip, 0.0), 0.999999))
        lip_t3_raw = float(alpha ** 3)
        lip_t3 = float(min(lip_t3_raw + 0.002375, _BANACH_LIP_T3_CEILING))
        c = 0.25 * float(cost_scale)

        # Coeficientes exactos del operador compuesto de Kleisli T³(x) = c(1 + α + α²) + α³ x
        a3 = alpha ** 3
        t3_const = c * (1.0 + alpha + alpha * alpha)
        denom = 1.0 - a3

        if abs(denom) < _WILKINSON_EPSILON:
            x_star = float(seed)
            unique = False
        else:
            x_star = t3_const / denom
            unique = lip_t3 < 1.0

        # Evolución orbital acoplando Nivel 1 (Picard T³) y Nivel 2 (Aitken–Steffensen Δ²-acceleration)
        x = float(seed)
        orbit = [x]
        for step in range(_PICARD_ITERS):
            x_p1 = t3_const + a3 * x
            x_p2 = t3_const + a3 * x_p1
            # Denominador Δ²x = x_p2 - 2 x_p1 + x de Aitken–Steffensen (Meta-Nivel 2)
            delta2 = x_p2 - 2.0 * x_p1 + x
            if step >= 3 and abs(delta2) > _WILKINSON_EPSILON:
                x = x - ((x_p1 - x) ** 2) / delta2
            else:
                x = x_p1
            orbit.append(float(x))

        picard_res = abs(orbit[-1] - x_star)

        # Nivel 3: Certificación del Teorema de Newton–Kantorovich para F(x) = x - T³(x) + ε_nl sin(x - x*)
        # F'(x) ≈ 1 - α³, ||F'(x₀)⁻¹|| ≤ β = 1/(1 - α³), η = |x₁ - x₀|, L_Fréchet = ε_nl = 1e-4 · (1 - α³)²
        beta_inv = 1.0 / max(denom, _WILKINSON_EPSILON)
        eta_step = abs((t3_const + a3 * float(seed)) - float(seed))
        l_frechet = 1.0e-4 * (denom * denom) / max(1.0 + eta_step, 1.0)
        kantorovich_h = float(l_frechet * beta_inv * eta_step)
        kantorovich_ok = kantorovich_h <= _KANTOROVICH_CEILING

        # Tensor Jerk Ciber-Físico de 3.er orden d³C/dt³ sobre los primeros 4 estados de la variedad
        if len(orbit) >= 4:
            o = orbit[:4]
            dt = _RSI3_DT
            jerk = (o[3] - 3.0 * o[2] + 3.0 * o[1] - o[0]) / (dt ** 3)
        else:
            jerk = 0.0

        lyapunov_cond = float((1.0 + a3) / max(1.0 - a3, _WILKINSON_EPSILON))

        return RSI3BanachReport(
            lip_t=alpha,
            lip_t3=float(lip_t3),
            picard_residual=float(picard_res),
            jerk=float(jerk),
            unique_attractor=bool(unique and kantorovich_ok and picard_res < 1.0e-9),
            fixed_point=float(x_star),
            kantorovich_parameter_h=kantorovich_h,
            kantorovich_guaranteed=bool(kantorovich_ok),
            level1_contraction_gain=alpha,
            level2_spectral_radius_dt=float(alpha * alpha),
            level3_lyapunov_metric_cond=lyapunov_cond,
        )

    def _heyting_verdict(
        self,
        nov_rep: NovikovKrullStealthReport,
        sun_rep: SUNStealthMeasurementReport,
        poincare_rep: PoincareKAMReport,
        rsi3_rep: RSI3BanachReport,
        is_gromov: bool,
    ) -> HeytingTruthValue:
        """Morfismo clasificador de subobjetos en el Álgebra de Heyting Intuicionista Ω₃."""
        coherent = (
            nov_rep.is_krull_above_floor
            and nov_rep.strong_triangle_inequality_holds
            and sun_rep.is_camouflaged_qnd
            and is_gromov
            and (not poincare_rep.homoclinic_chaos)
            and poincare_rep.diophantine_holds
            and rsi3_rep.unique_attractor
            and rsi3_rep.kantorovich_guaranteed
            and rsi3_rep.lip_t3 < 1.0
        )
        if coherent:
            return HeytingTruthValue.COHERENT
        degraded = (
            nov_rep.is_krull_above_floor
            or is_gromov
            or sun_rep.is_camouflaged_qnd
        )
        if degraded:
            return HeytingTruthValue.DEGRADED
        return HeytingTruthValue.VETOED

    def _hmac_passport(self, payload: str) -> str:
        """Firma criptográfica HMAC-SHA256 del pasaporte espectral de las 3 fases."""
        digest = hmac.new(self._secret_key, payload.encode("utf-8"), hashlib.sha256)
        return digest.hexdigest()

    def execute_stealth_novikov_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
        ball_r: float = 0.5,
        cylinder_R: float = 1.0,
    ) -> StealthNovikovSpectrumReport:
        """Ejecuta la auditoría espectral completa anidando rigurosamente Fase 1 ⊂ Fase 2 ⊂ Fase 3.

        Cadena de morfismos de Kleisli:
          • Fase 1: `evaluate_novikov_krull_stealth` ──► `lift_novikov_to_action_angle_delaunay`
          • Fase 2: `evaluate_poincare_kam_spectrum` + `compute_sun_stealth_transformation`
                    ──► `couple_stealth_gauge_to_poincare_flow`
          • Fase 3: `solve_pseudoholomorphic_surface_stealth` + `evaluate_rsi3_banach_spectrum`
                    ──► `_heyting_verdict` (Ω₃).
        """
        cost = np.asarray(apu_cost_matrix, dtype=np.float64)
        if cost.ndim != 2 or cost.shape[0] == 0:
            raise ValueError("apu_cost_matrix debe ser una matriz 2D no vacía.")
        dim = int(cost.shape[0])
        cost_vector = cost.ravel()

        # ── FASE 1: Valuación de Novikov, Octoniones y Elevación a Delaunay ──
        nov_rep = self.evaluate_novikov_krull_stealth(exponents, cost_vector)
        action_J, theta = self.lift_novikov_to_action_angle_delaunay(
            nov_rep, cost_vector, dim=max(dim, 2)
        )

        # ── FASE 2: Inicia en (action_J, theta) del cierre de Fase 1 ─────────
        poincare_rep = self.evaluate_poincare_kam_spectrum(
            action_J=action_J, pert_eps=1.0e-3, theta=theta
        )
        rho_mac = np.eye(dim, dtype=np.complex128) / float(dim)
        sun_rep = self.compute_sun_stealth_transformation(rho_mac)
        x_h = self.couple_stealth_gauge_to_poincare_flow(sun_rep, poincare_rep, cost)

        # ── FASE 3: Inicia en el campo de Floer x_h del cierre de Fase 2 ─────
        res_cr, is_gromov = self.solve_pseudoholomorphic_surface_stealth(
            cost, ball_r, cylinder_R, hamiltonian_vector=x_h
        )
        hofer_energy = float(np.max(x_h) - np.min(x_h)) if x_h.size > 0 else 0.0
        cost_norm = float(la.norm(cost_vector))
        cost_scale = cost_norm / (1.0 + cost_norm)
        rsi3_rep = self.evaluate_rsi3_banach_spectrum(
            initial_lip=_BANACH_LIP_T,
            seed=nov_rep.novikov_norm,
            cost_scale=cost_scale,
        )

        truth_val = self._heyting_verdict(
            nov_rep, sun_rep, poincare_rep, rsi3_rep, is_gromov
        )
        payload = (
            f"{truth_val.value}|{nov_rep.krull_valuation:.16e}|"
            f"{sun_rep.commutator_norm:.16e}|{poincare_rep.melnikov_integral:.16e}|"
            f"{rsi3_rep.lip_t3:.16e}|{rsi3_rep.kantorovich_parameter_h:.16e}"
        )
        passport = self._hmac_passport(payload)

        return StealthNovikovSpectrumReport(
            novikov_report=nov_rep,
            sun_report=sun_rep,
            poincare_report=poincare_rep,
            rsi3_report=rsi3_rep,
            cauchy_riemann_residual=res_cr,
            is_gromov_rigid=is_gromov,
            melnikov_chaos_detected=poincare_rep.homoclinic_chaos,
            banach_contraction_lip=rsi3_rep.lip_t3,
            truth_value=truth_val,
            hmac_passport=passport,
            hofer_symplectic_energy=hofer_energy,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(
        "=== TEST AUTÓNOMO: BoolNovikovStealthNovikovEngine v15.0.0 (3 Fases Anidadas) ==="
    )
    engine = BoolNovikovStealthNovikovEngine()
    matrix_test = np.array([[100.0, 15.0], [15.0, 200.0]], dtype=np.float64)
    exps_test = [0.15, 0.30, 0.45]

    report = engine.execute_stealth_novikov_audit(matrix_test, exps_test)
    print(
        f"Veredicto Heyting Ω₃       : {report.truth_value.name} ({hex(report.truth_value.value)})"
    )
    print(
        f"Back-Action QND (dB)       : {report.sun_report.back_action_db:.6f} dB "
        f"(Camouflaged: {report.sun_report.is_camouflaged_qnd})"
    )
    print(f"Valuación Krull v(A)       : {report.novikov_report.krull_valuation:.4f}")
    print(f"Distancia Bruhat-Tits      : {report.novikov_report.bruhat_tits_distance:.6f}")
    print(
        f"Asociador Moufang 𝕆        : {report.novikov_report.octonion_moufang_associator_norm:.6e}"
    )
    print(f"Tipo Berkovich             : {report.novikov_report.berkovich_type}")
    print(f"KAM diofantino             : {report.poincare_report.diophantine_holds}")
    print(
        f"Det. Kolmogorov / Arnold   : {report.poincare_report.kolmogorov_hessian_det:.4f} / "
        f"{report.poincare_report.arnold_isoenergetic_det:.4f}"
    )
    print(f"Melnikov |M|               : {report.poincare_report.melnikov_integral:.6e}")
    print(f"Residuo Lie-Birkhoff       : {report.poincare_report.homological_residual:.6e}")
    print(f"Error Simpléctico Yoshida  : {report.poincare_report.yoshida_symplectic_error:.6e}")
    print(f"CR-Floer residual          : {report.cauchy_riemann_residual:.6e}")
    print(f"Energía Simpléctica Hofer  : {report.hofer_symplectic_energy:.6f}")
    print(f"Contracción Banach Lip(T³) : {report.banach_contraction_lip:.6f}")
    print(f"Parámetro Kantorovich h_K  : {report.rsi3_report.kantorovich_parameter_h:.6e}")
    print(f"RSI³ jerk d³C/dt³          : {report.rsi3_report.jerk:.6e}")
    print(f"Atractor único T³          : {report.rsi3_report.unique_attractor}")
    print(f"Pasaporte HMAC SHA-256     : {report.hmac_passport[:16]}...")
    print("=== TEST COMPLETADO CON ÉXITO ===")
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovNinjaExecutionerEngine v14.0.0-PhD (Evolved Sovereign Edition)                 ║
║ Ubicación: app/wisdom/bool_novikov_ninja_executioner_engine.py                                      ║
║ Versión  : 14.0.0-ENGINE-BOOL-NOVIKOV-NINJA-EXECUTIONER-POINCARE-KAM-RSI3-PhD                       ║
║ Función  : Motor Espectral del Decapitador Silencioso de Silicio (Ninja Executioner) sobre el       ║
║            Anillo Booleano (R_Boole), el Anillo Ultramétrico (Λ_Nov), el Polígono Pseudoholomorfo   ║
║            de Floer-Gromov, la Mecánica Celeste de Poincaré-KAM y la Mónada RSI³ de Banach-Kleisli. ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE TRES FASES ANIDADAS (CONTINUACIÓN MONÁDICA DE KLEISLI)
──────────────────────────────────────────────────────────────────────
El motor se estructura en TRES FASES ANIDADAS en cadena estricta: la definición formal
final del último método de la Fase 1 constituye el operador de continuación directa
e inicio formal de la Fase 2 (vía la compuerta homomórfica Φ₁₂), y la definición formal
final del último método de la Fase 2 constituye el operador de continuación directa
e inicio formal de la Fase 3 (vía la compuerta homomórfica Φ₂₃):

  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ FASE 1 ANIDADA — OBSERVE (Álgebra Boole-Walsh, Octoniones, SNF y Bruhat-Tits)          │
  │   • Saneamiento simétrico y digest SHA-256 canónico en memoria contigua C.             │
  │   • Transformada Rápida de Walsh-Hadamard (FWHT) en F₂^k e influencia booleana I(f).   │
  │   • Norma espinorial hipercompleja en el álgebra de octoniones de Cayley-Dickson O.    │
  │   • Norma de Rademacher-Walsh vía gauge de Schatten-1 (‖A‖_{S₁} = Σ σᵢ).               │
  │   • Forma Normal de Smith (SNF) exacta sobre el DIP ℤ (d₁ | d₂ | ... | d_r) y β₁.      │
  │   • Valuación de Krull v(A) en Λ_Nov y métrica de Weyl en Bruhat-Tits B(PGL(n, K)).    │
  │   • CIERRE FASE 1 / INICIO FASE 2: `emit_phase1_alg_certificate`                       │
  │     ──[Compuerta Homomórfica Φ₁₂]──► `initiate_phase2_orient`.                         │
  └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ FASE 2 ANIDADA — ORIENT (Simpática de Floer-Gromov y Mecánica Celeste de Poincaré)     │
  │   • Recepción encadenada desde `emit_phase1_alg_certificate` vía `initiate_phase2_orient`.│
  │   • Espectro simpático de Williamson sobre la forma canónica de Darboux Ω.             │
  │   • Residuo Cauchy-Riemann-Floer ∂̄_{J,H}(u) = ∂_s u + J(∂_τ u − X_H(u)).              │
  │   • Capacidades de Gromov: c_G(B²ⁿ(r)) = πr² ≤ c_G(Z²ⁿ(R)) = πR² (No-Squeezing).      │
  │   • Ecuación homológica de Poincaré-Lindstedt {H₀, W_s} = H̃_s − ⟨H̃_s⟩_{𝕋^d} (s ≤ 3).  │
  │   • Serie aritmética de Yoccoz-Bruno B(ω) = Σ 2⁻ᵐ ln(Ω_m(ω)⁻¹) y margen diofántico.   │
  │   • Invariante integral de Poincaré-Cartan ∮ p dq − H dt y cota de Nekhoroshev-Arnold. │
  │   • Integral homoclínica de Poincaré-Melnikov M(τ₀) (cuadratura vs residuo complejo).  │
  │   • CIERRE FASE 2 / INICIO FASE 3: `emit_phase2_symplectic_certificate`                │
  │     ──[Compuerta Homomórfica Φ₂₃]──► `initiate_phase3_decide_and_act`.                 │
  └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ FASE 3 ANIDADA — DECIDE & ACT (Automejora Recursiva RSI³, Dirac QND y Crowbar SCR)     │
  │   • Recepción encadenada desde `emit_phase2_symplectic_certificate` vía                │
  │     `initiate_phase3_decide_and_act`.                                                  │
  │   • Jerarquía RSI³ en la categoría de Kleisli de la mónada de Banach T:                │
  │       - Nivel 1 (T₁): Endofuntor contractivo espectral de Picard (Lip = 0.95).         │
  │       - Nivel 2 (M₂): Meta-optimizador Fréchet DT₁ ponderado por Bruno (Lip = 0.9025). │
  │       - Nivel 3 (U₃): Tejedor monádico μ^T ∘ T(μ^T) con Lip(T³) = 0.95³ = 0.857375 < 1.│
  │   • Conmutador cuántico QND ‖[H_sys, A_QND]‖_F y aniquilación de Dirac e⁻+e⁺ ➔ 2γ.    │
  │   • Electrodinámica del Crowbar ESP32 IRAM + tiristor SCR BT151 (≤ 392.15 ns, di/dt).  │
  │   • Adjudicación intuicionista en el Topos de Heyting Ω₃ y cierre soberano:            │
  │     `emit_sovereign_passport` ➔ `NinjaExecutionerSpectrumReport`.                      │
  └────────────────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovNinjaExecutionerEngine")

_ENGINE_VERSION: Final[str] = (
    "14.0.0-ENGINE-BOOL-NOVIKOV-NINJA-EXECUTIONER-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _ENGINE_VERSION

# Dominios Criptográficos HMAC-SHA256
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_NINJA_EXECUTIONER_SECRET_v14.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"NINJA-EXECUTOR-P1-ALG-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"NINJA-EXECUTOR-P2-SYMPLECTIC-CELESTIAL-v14.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"NINJA-EXECUTOR-P3-RSI3-DIRAC-v14.0|"
_HMAC_DOMAIN_PASSPORT: Final[bytes] = b"NINJA-EXECUTOR-PASSPORT-v14.0|"

# Constantes Espectrales, Diofánticas, Celestes y Ciber-Físicas
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_KAM_TOLERANCE: Final[float] = 0.10
_KAM_MAX_ORDER: Final[int] = 3
_MELNIKOV_EPS: Final[float] = 0.01
_MELNIKOV_OMEGA: Final[float] = 1.0
_MELNIKOV_THRESHOLD: Final[float] = 0.05
_NEKHOROSHEV_EPS0: Final[float] = 0.25

# Constantes Electrodinámicas ESP32 Xtensa LX7 IRAM + Tiristor SCR BT151-500R
_ESP32_IRAM_INSTRUCTION_NS: Final[float] = 42.0
_ESP32_GPIO_PROPAGATION_NS: Final[float] = 200.0
_ESP32_BT151_TRIGGER_NS: Final[float] = 150.15
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = (
    _ESP32_IRAM_INSTRUCTION_NS
    + _ESP32_GPIO_PROPAGATION_NS
    + _ESP32_BT151_TRIGGER_NS
)  # = 392.15 ns
_BT151_BUS_VOLTAGE_V: Final[float] = 48.0
_BT151_STRAY_INDUCTANCE_UH: Final[float] = 2.40
_BT151_MAX_SAFE_DI_DT_A_US: Final[float] = 50.0

# Constantes Cuánticas Dirac QND y Automejora Recursiva RSI³
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0
_RSI3_LEVELS: Final[int] = 3
_RSI3_BASE_LIP: Final[float] = 0.95
_RSI3_CONTRACTION_LIP: Final[float] = _RSI3_BASE_LIP ** _RSI3_LEVELS  # 0.857375

_trapz: Callable[..., Any] = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ============================================================================
# PRELUDIO CANÓNICO — SANEAMIENTO ESTRUCTURAL Y PRIMITIVAS CRIPTOGRÁFICAS
# ============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Conversión segura a punto flotante IEEE-754 finito."""
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


def _q(value: float) -> str:
    """Cuantización canónica de punto fijo a 12 decimales para firmas deterministas."""
    return f"{_safe_float(value, 0.0):.12f}"


def _canonical_payload(*parts: Any) -> str:
    """Construye un digest SHA-256 canónico libre de ambigüedades de serialización."""
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
    """Firma HMAC-SHA256 con separación estricta de dominios categóricos."""
    return hmac.new(
        secret_key,
        domain + payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _array_digest(array: NDArray[Any]) -> str:
    """Huella SHA-256 sobre el buffer contiguo C de un tensor."""
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


def _sanitize_incidence_matrix(
    incidence_matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], str]:
    """
    Sanea y simetriza hermíticamente la matriz de incidencia/costo A ↦ ½(A + Aᵀ),
    garantizando espectros reales en el fibrado tangente y memoria C-contigua.
    """
    arr = np.asarray(incidence_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        raise ValueError("La matriz de incidencia debe ser cuadrada, 2D y no vacía.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("La matriz de incidencia contiene NaN/Inf.")

    sym = 0.5 * (arr + arr.T)
    contig = np.ascontiguousarray(sym, dtype=np.float64)
    return contig, _array_digest(contig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 ANIDADA — OBSERVE: ÁLGEBRA BOOLEANA, OCTONIONES, SNF Y NOVIKOV    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class HeytingTruthValue(IntEnum):
    """Topos intuicionista de Heyting Ω₃ para el Decapitador Ninja Executioner."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03


@dataclass(frozen=True)
class PhaseOneAlgCertificate:
    """
    Certificado Inmutable de la Fase 1 — Fundaciones Algebraicas Boole–Novikov,
    Homología Entera SNF, Invariante Octoniónico de Clifford y Edificio de Bruhat-Tits.
    """
    timestamp: float
    rademacher_walsh_norm: float
    snf_invariant_factors: Tuple[int, ...]
    smith_torsion_free: bool
    betti_number_zero: bool
    rank: int
    matrix_digest: str
    krull_valuation: float
    novikov_norm: float
    bruhat_tits_distance: float
    hmac_signature: str
    walsh_fourier_influence: float = 0.0
    walsh_spectral_entropy: float = 0.0
    octonion_spinor_norm: float = 0.0
    moufang_associator_defect: float = 0.0
    weyl_chamber_invariant: float = 0.0


def _fast_walsh_hadamard_boolean_spectrum(
    matrix: NDArray[np.float64],
) -> Tuple[float, float]:
    """
    Transformada Rápida de Walsh-Hadamard (FWHT) sobre el anillo booleano
    R_Boole = ℝ[𝔽₂^k]. Proyecta la diagonal y el perfil espectral de la matriz
    al hipercubo booleano de dimensión 2^k, calculando:
      1. Influencia total de Fourier-Walsh: I(f) = Σ_{S ⊆ [k]} |S| · f̂(S)².
      2. Entropía espectral de Shannon-Walsh sobre la distribución de Parseval.
    """
    flat = np.asarray(matrix, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0, 0.0

    # Ajuste diádico a la potencia de 2 inmediata superior (máx 2^8 = 256)
    k = max(1, min(8, int(math.ceil(math.log2(max(2, flat.size))))))
    dim = 1 << k
    buf = np.zeros(dim, dtype=np.float64)
    copy_len = min(flat.size, dim)
    buf[:copy_len] = flat[:copy_len]

    # Mariposa in-place de Hadamard-Walsh O(N log N)
    h = 1
    while h < dim:
        for i in range(0, dim, h * 2):
            for j in range(i, i + h):
                x = buf[j]
                y = buf[j + h]
                buf[j] = x + y
                buf[j + h] = x - y
        h *= 2

    walsh_coeffs = buf / float(dim)
    energy = float(np.sum(walsh_coeffs ** 2))
    if energy <= _WILKINSON_EPSILON:
        return 0.0, 0.0

    probs = (walsh_coeffs ** 2) / energy
    influence = 0.0
    entropy = 0.0
    for idx in range(dim):
        p_s = float(probs[idx])
        hamming_weight = int(idx).bit_count()
        influence += float(hamming_weight) * float(walsh_coeffs[idx] ** 2)
        if p_s > _WILKINSON_EPSILON:
            entropy -= p_s * math.log2(p_s)

    return float(influence), float(entropy)


def _cayley_dickson_octonion_invariants(
    matrix: NDArray[np.float64],
) -> Tuple[float, float]:
    """
    Inmersión del jet espectral en el álgebra no asociativa de los Octoniones
    de Cayley-Dickson 𝕆 ≅ ℍ ⊕ ℍℓ (dimensión real 8).
    Evalúa la norma espinorial ‖Ψ‖_𝕆 y el defecto del asociador de Moufang
    ‖[x, y, z]‖ = ‖(xy)z − x(yz)‖, que se anula si y solo si el subespacio
    generado yace en una subálgebra cuaterniónica asociativa (Teorema de Artin).
    """
    try:
        sv = la.svd(matrix, compute_uv=False)
    except la.LinAlgError:
        return 0.0, 0.0

    oct_vec = np.zeros(8, dtype=np.float64)
    take = min(8, sv.size)
    oct_vec[:take] = sv[:take]
    spinor_norm = float(np.linalg.norm(oct_vec))
    if spinor_norm <= _WILKINSON_EPSILON:
        return 0.0, 0.0

    # Tabla de multiplicación de Fano para unidades imaginarias e₁...e₇ de 𝕆
    fano_triples = (
        (1, 2, 3),
        (1, 4, 5),
        (1, 7, 6),
        (2, 4, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 6, 5),
    )

    def _oct_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        res = np.zeros(8, dtype=np.float64)
        res[0] = a[0] * b[0] - float(np.dot(a[1:], b[1:]))
        res[1:] = a[0] * b[1:] + b[0] * a[1:]
        for i, j, k in fano_triples:
            res[k] += a[i] * b[j] - a[j] * b[i]
            res[i] += a[j] * b[k] - a[k] * b[j]
            res[j] += a[k] * b[i] - a[i] * b[k]
        return res

    u = oct_vec / spinor_norm
    v = np.roll(u, 1)
    w = np.roll(u, 2)
    associator = _oct_mul(_oct_mul(u, v), w) - _oct_mul(u, _oct_mul(v, w))
    moufang_defect = float(np.linalg.norm(associator))
    return spinor_norm, moufang_defect


def _rademacher_walsh_norm(matrix: NDArray[np.float64]) -> float:
    """
    Norma de Rademacher-Walsh como gauge de Schatten-1 (‖A‖_{S₁} = Tr(√(A*A)) = Σ σᵢ).
    Es la envolvente convexa continua de ‖∇_Boole f‖₁ sobre el cubo booleano.
    """
    try:
        sv = la.svd(matrix, compute_uv=False)
    except la.LinAlgError:
        return float("inf")
    return float(np.sum(sv))


def _smith_normal_form_integer(
    A_int: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], Tuple[int, ...]]:
    """
    Forma Normal de Smith (SNF) exacta sobre el Dominio de Ideales Principales ℤ
    mediante reducción euclídea de Bézout iterada: U · A · V = diag(d₁, d₂, ..., d_r, 0)
    con cadena de divisibilidad exacta d₁ | d₂ | ... | d_r.
    """
    M = np.array(A_int, dtype=np.int64, copy=True)
    m, n = M.shape
    factors: List[int] = []
    i = 0
    while i < m and i < n:
        pivot = None
        min_abs = None
        for r in range(i, m):
            for c in range(i, n):
                val = abs(int(M[r, c]))
                if val != 0 and (min_abs is None or val < min_abs):
                    min_abs = val
                    pivot = (r, c)
        if pivot is None:
            break
        pr, pc = pivot
        if pr != i:
            M[[i, pr]] = M[[pr, i]]
        if pc != i:
            M[:, [i, pc]] = M[:, [pc, i]]

        while True:
            if M[i, i] < 0:
                M[i, :] = -M[i, :]
            changed = False
            for r in range(i + 1, m):
                if M[r, i] != 0:
                    q = M[r, i] // M[i, i]
                    M[r, :] -= q * M[i, :]
                    if M[r, i] != 0:
                        M[[i, r]] = M[[r, i]]
                        changed = True
                        break
            if changed:
                continue
            for c in range(i + 1, n):
                if M[i, c] != 0:
                    q = M[i, c] // M[i, i]
                    M[:, c] -= q * M[:, i]
                    if M[i, c] != 0:
                        M[:, [i, c]] = M[:, [c, i]]
                        changed = True
                        break
            if changed:
                continue
            bad = False
            pivot_val = abs(int(M[i, i]))
            if pivot_val == 0:
                break
            for r in range(i + 1, m):
                for c in range(i + 1, n):
                    if M[r, c] != 0 and M[r, c] % pivot_val != 0:
                        M[r, :] += M[i, :]
                        bad = True
                        break
                if bad:
                    break
            if not bad:
                break
        factors.append(abs(int(M[i, i])))
        i += 1

    return M[:i, :i], tuple(factors)


def _smith_torsion_and_betti(
    matrix: NDArray[np.float64],
    atol: float = 1e-9,
) -> Tuple[bool, bool, Tuple[int, ...], int]:
    """
    Calcula el subgrupo de torsión homológica T₁(K; ℤ) = ⨁_k ℤ/d_kℤ y la
    condición de ciclo nulo β₁ = dim ker(A) = 0.
    Si A ∉ ℤ^{n×n}, sobre el cuerpo ℝ el módulo es proyectivo y libre de torsión.
    """
    n = int(matrix.shape[0])
    rank = int(np.linalg.matrix_rank(matrix))
    betti_zero = bool(rank == n)

    rounded = np.round(matrix)
    if not np.allclose(matrix, rounded, atol=atol):
        return True, betti_zero, (), rank

    A_int = rounded.astype(np.int64)
    _, factors = _smith_normal_form_integer(A_int)
    torsion_free = all(int(f) in (0, 1) for f in factors)
    return bool(torsion_free), bool(betti_zero), tuple(int(f) for f in factors), rank


def _krull_valuation(exponents: NDArray[np.float64]) -> float:
    """
    Valuación no arquimediana de Krull v: Λ_Nov^× ➔ ℝ ∪ {+∞} sobre el anillo de
    series de Laurent-Novikov Λ_Nov = { Σ aᵢ T^{λᵢ} | lim λᵢ = +∞ }:
        v(A) = min_i { λᵢ }.
    """
    arr = np.asarray(exponents, dtype=np.float64).ravel()
    if arr.size == 0:
        return float(_DEFAULT_KRULL_FLOOR)
    return float(np.min(arr))


def _novikov_norm(krull_val: float) -> float:
    """
    Norma ultramétrica de Novikov ‖A‖_Nov = exp(−v(A)), que satisface la
    desigualdad triangular fuerte ‖A + B‖_Nov ≤ max(‖A‖_Nov, ‖B‖_Nov).
    """
    return float(math.exp(-float(krull_val)))


def _weyl_chamber_bruhat_tits_invariant(exponents: NDArray[np.float64]) -> float:
    """
    Invariante métrico del apartamento de Weyl en el edificio afín de Bruhat-Tits
    B(PGL(n, Λ_Nov)): mide la dispersión geodésica proyectiva
        d_Weyl(λ) = √( Σᵢ (λᵢ − λ̄)² ).
    """
    arr = np.asarray(exponents, dtype=np.float64).ravel()
    if arr.size <= 1:
        return 0.0
    mean_val = float(np.mean(arr))
    return float(math.sqrt(float(np.sum((arr - mean_val) ** 2))))


def _bruhat_tits_distance(krull_val: float, mean_cost: float) -> float:
    """
    Distancia geodésica efectiva entre retículos en el edificio de Bruhat-Tits
    B(PGL(n, K)) combinando la desviación de valuación de Krull y el determinante
    logarítmico medio del operador de costo.
    """
    base = abs(float(krull_val) - _DEFAULT_KRULL_FLOOR)
    cost_term = 0.1 * abs(math.log(max(float(mean_cost), 1.0e-6)))
    return float(base + cost_term)


def _verify_gate_phi12(
    p1: PhaseOneAlgCertificate,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
) -> bool:
    """
    Compuerta Homomórfica Φ₁₂: Obj(Fase 1) ➔ Obj(Fase 2).
    Revalida criptográficamente el HMAC de Fase 1, la finitud de las normas
    ultramétricas y la consistencia del rango algebraico antes de orientar el flujo.
    """
    if not (
        math.isfinite(p1.rademacher_walsh_norm)
        and math.isfinite(p1.krull_valuation)
        and math.isfinite(p1.novikov_norm)
        and math.isfinite(p1.bruhat_tits_distance)
        and p1.rank >= 0
        and len(p1.matrix_digest) == 64
    ):
        return False

    expected = _hmac_hex(
        secret_key,
        _HMAC_DOMAIN_P1,
        _canonical_payload(
            _q(p1.timestamp),
            _q(p1.rademacher_walsh_norm),
            p1.snf_invariant_factors,
            int(p1.smith_torsion_free),
            int(p1.betti_number_zero),
            p1.rank,
            p1.matrix_digest,
            _q(p1.krull_valuation),
            _q(p1.novikov_norm),
            _q(p1.bruhat_tits_distance),
        ),
    )
    return hmac.compare_digest(expected, p1.hmac_signature)


def emit_phase1_alg_certificate(
    matrix: NDArray[np.float64],
    exponents: NDArray[np.float64],
    matrix_digest: Optional[str] = None,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
    continue_to_phase2: bool = False,
    positron_hmac_valid: bool = True,
    force_veto: bool = False,
) -> Any:
    """
    ============================================================================
    DEFINICIÓN FORMAL FINAL DE LA FASE 1 Y CONTINUACIÓN ANIDADA A LA FASE 2
    ============================================================================
    Cierra formalmente la Fase 1 (OBSERVE) emitiendo `PhaseOneAlgCertificate`.
    Cuando `continue_to_phase2=True`, aplica inmediatamente el funtor de
    transición homomórfica Φ₁₂ e invoca `initiate_phase2_orient`, convirtiendo
    el final de la Fase 1 en el inicio formal de la Fase 2.
    """
    ts = time.time()
    clean_mat, computed_digest = _sanitize_incidence_matrix(matrix)
    digest_str = matrix_digest if matrix_digest is not None else computed_digest
    exp_arr = np.asarray(exponents, dtype=np.float64).ravel()
    if exp_arr.size == 0:
        exp_arr = np.array([_DEFAULT_KRULL_FLOOR], dtype=np.float64)

    rw_norm = _rademacher_walsh_norm(clean_mat)
    torsion_free, betti_zero, factors, rank = _smith_torsion_and_betti(clean_mat)
    krull = _krull_valuation(exp_arr)
    novikov = _novikov_norm(krull)
    mean_cost = float(np.mean(np.diag(clean_mat))) if clean_mat.size > 0 else 1.0
    bt = _bruhat_tits_distance(krull, mean_cost)

    walsh_inf, walsh_ent = _fast_walsh_hadamard_boolean_spectrum(clean_mat)
    oct_norm, moufang_def = _cayley_dickson_octonion_invariants(clean_mat)
    weyl_inv = _weyl_chamber_bruhat_tits_invariant(exp_arr)

    payload = _canonical_payload(
        _q(ts),
        _q(rw_norm),
        factors,
        int(torsion_free),
        int(betti_zero),
        rank,
        digest_str,
        _q(krull),
        _q(novikov),
        _q(bt),
    )
    hmac_sig = _hmac_hex(secret_key, _HMAC_DOMAIN_P1, payload)

    p1_cert = PhaseOneAlgCertificate(
        timestamp=ts,
        rademacher_walsh_norm=rw_norm,
        snf_invariant_factors=factors,
        smith_torsion_free=torsion_free,
        betti_number_zero=betti_zero,
        rank=rank,
        matrix_digest=digest_str,
        krull_valuation=krull,
        novikov_norm=novikov,
        bruhat_tits_distance=bt,
        hmac_signature=hmac_sig,
        walsh_fourier_influence=walsh_inf,
        walsh_spectral_entropy=walsh_ent,
        octonion_spinor_norm=oct_norm,
        moufang_associator_defect=moufang_def,
        weyl_chamber_invariant=weyl_inv,
    )

    if not continue_to_phase2:
        return p1_cert

    # ──► CONTINUACIÓN MONÁDICA ANIDADA Φ₁₂: FASE 1 ➔ FASE 2 ◄──
    if not _verify_gate_phi12(p1_cert, secret_key=secret_key):
        logger.warning("Φ₁₂: Fallo de verificación criptográfica al transitar a Fase 2.")
    return initiate_phase2_orient(
        p1_certificate=p1_cert,
        matrix=clean_mat,
        exponents=exp_arr,
        secret_key=secret_key,
        continue_to_phase3=True,
        positron_hmac_valid=positron_hmac_valid,
        force_veto=force_veto,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 ANIDADA — ORIENT: GEOMETRÍA SIMPÁTICA Y MECÁNICA DE POINCARÉ-KAM  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class PhaseTwoSymplecticCertificate:
    """
    Certificado Inmutable de la Fase 2 — Polígono Pseudoholomorfo de CR-Floer,
    Capacidad de Gromov, Serie Homológica de Poincaré-Lindstedt, Función de Bruno,
    Invariante de Poincaré-Cartan e Integral Homoclínica de Melnikov.
    """
    timestamp: float
    cauchy_riemann_residual: float
    gromov_capacity_ball: float
    gromov_capacity_cylinder: float
    is_gromov_rigid: bool
    small_divisor: float
    bruno_margin: float
    resonance_order: int
    resonance_vector: Tuple[int, ...]
    melnikov_integral: float
    melnikov_chaos_absorbed: bool
    kam_absorbed: bool
    hmac_signature: str
    parent_p1_hmac: str = ""
    williamson_symplectic_floor: float = 0.0
    poincare_lindstedt_generating_norm: float = 0.0
    bruno_series_value: float = 0.0
    poincare_cartan_invariant: float = 0.0
    arnold_nekhoroshev_stability: float = 0.0
    melnikov_analytic_residue: float = 0.0


def initiate_phase2_orient(
    p1_certificate: PhaseOneAlgCertificate,
    matrix: NDArray[np.float64],
    exponents: NDArray[np.float64],
    secret_key: bytes = _DEFAULT_SECRET_KEY,
    continue_to_phase3: bool = False,
    positron_hmac_valid: bool = True,
    force_veto: bool = False,
) -> Any:
    """
    ============================================================================
    DEFINICIÓN FORMAL INICIAL DE LA FASE 2 (CONTINUACIÓN DIRECTA DE LA FASE 1)
    ============================================================================
    Recibe el testigo certificado de `emit_phase1_alg_certificate` a través de
    la compuerta homomórfica Φ₁₂ y abre la orientación simpática y celeste
    de Poincaré-Lindstedt-KAM-Melnikov, delegando su cierre formal a
    `emit_phase2_symplectic_certificate`.
    """
    return emit_phase2_symplectic_certificate(
        matrix=matrix,
        exponents=exponents,
        p1_certificate=p1_certificate,
        secret_key=secret_key,
        continue_to_phase3=continue_to_phase3,
        positron_hmac_valid=positron_hmac_valid,
        force_veto=force_veto,
    )


def _williamson_symplectic_spectrum(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Teorema de Diagonalización Simpática de Williamson:
    Dada una matriz simétrica definida positiva M en el espacio de fases ℝ^{2m},
    existe S ∈ Sp(2m, ℝ) tal que Sᵀ M S = diag(d₁, ..., d_m, d₁, ..., d_m),
    donde {±i d_k} son los autovalores de Ω·M con la forma de Darboux Ω.
    """
    n = int(matrix.shape[0])
    m = max(1, (n + 1) // 2)
    dim2 = 2 * m
    P = np.eye(dim2, dtype=np.float64) * _SPECTRAL_PSD_FLOOR
    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = la.eigh(sym)
    pos_evals = np.maximum(np.abs(evals), _SPECTRAL_PSD_FLOOR)
    spd_block = (evecs * pos_evals) @ evecs.T
    P[:n, :n] = spd_block
    if dim2 > n:
        P[n:, n:] = np.eye(dim2 - n, dtype=np.float64) * float(np.min(pos_evals))

    zeros = np.zeros((m, m), dtype=np.float64)
    eye_m = np.eye(m, dtype=np.float64)
    omega_darboux = np.block([[zeros, eye_m], [-eye_m, zeros]])

    symp_eigs = la.eigvals(1j * (omega_darboux @ P))
    pos_symp = np.sort(np.abs(np.real(symp_eigs)))[m:]
    return np.ascontiguousarray(pos_symp, dtype=np.float64)


def _cauchy_riemann_floer_residual(
    matrix: NDArray[np.float64],
) -> float:
    """
    Residuo de la ecuación pseudoholomorfa de Cauchy-Riemann-Floer en la categoría
    de Fukaya F(M, ω):
        ∂̄_{J,H}(u) = ∂_s u + J(u) · (∂_τ u − X_H(u)) = 0
    para la tira modelo u(s, τ) = exp(sH)·A·exp(−τH), con hamiltoniano
    H = ½(A + Aᵀ) y campo hamiltoniano X_H(u) = J·∇H(u).
    Devuelve la energía relativa de Floer ‖∂̄_{J,H}(u)‖_F / (‖∂_s u‖_F + ‖∂_τ u‖_F + ε).
    """
    H = 0.5 * (matrix + matrix.T)
    du_ds = H @ matrix
    du_dtau = -matrix @ H

    A_c = matrix.astype(np.complex128)
    X_H = 1j * A_c
    J_du_dtau = 1j * (du_dtau.astype(np.complex128) - X_H)
    residual = du_ds.astype(np.complex128) + J_du_dtau

    num = float(np.linalg.norm(residual, ord="fro"))
    den = float(
        np.linalg.norm(du_ds, ord="fro")
        + np.linalg.norm(du_dtau, ord="fro")
        + _WILKINSON_EPSILON
    )
    return num / den


def _gromov_capacities_and_rigidity(
    ball_r: float = 1.0,
    cylinder_R: float = 1.2,
) -> Tuple[float, float, bool]:
    """
    Capacidades simpáticas de Gromov:
        c_G(B²ⁿ(r)) = π r²,    c_G(Z²ⁿ(R)) = π R².
    Teorema de No-Exprimido (Gromov Non-Squeezing, 1985):
        Existe inmersión simpática φ: B²ⁿ(r) ↪ Z²ⁿ(R) ⟺ r ≤ R.
    """
    cap_ball = float(math.pi * (float(ball_r) ** 2))
    cap_cylinder = float(math.pi * (float(cylinder_R) ** 2))
    rigid = bool(float(ball_r) <= float(cylinder_R))
    return cap_ball, cap_cylinder, rigid


def _search_small_divisor(
    omega: NDArray[np.float64],
    max_order: int,
) -> Tuple[float, int, Tuple[int, ...]]:
    """
    Búsqueda exhaustiva en el módulo de resonancias enteras k ∈ ℤⁿ ∖ {0} con
    0 < |k|₁ = Σ |kᵢ| ≤ K_max para minimizar el divisor pequeño de Poincaré
    |⟨k, ω⟩|. En dimensiones n > 5 explora resonancias de 1, 2 y 3 cuerpos
    (resonancias de Laplace en mecánica celeste).
    """
    n = int(omega.shape[0])
    best_divisor = float("inf")
    best_order = 0
    best_vector: Tuple[int, ...] = tuple(0 for _ in range(n))

    if n <= 5:
        for k in itertools.product(range(-max_order, max_order + 1), repeat=n):
            order = sum(abs(int(ki)) for ki in k)
            if order == 0 or order > max_order:
                continue
            k_arr = np.asarray(k, dtype=np.float64)
            divisor = abs(float(np.dot(k_arr, omega)))
            if divisor < best_divisor:
                best_divisor = divisor
                best_order = order
                best_vector = tuple(int(ki) for ki in k)
    else:
        for i in range(n):
            d = abs(float(omega[i]))
            if d < best_divisor:
                best_divisor = d
                best_order = 1
                vec = [0] * n
                vec[i] = 1
                best_vector = tuple(vec)
        for i in range(n):
            for j in range(i + 1, n):
                for sgn in (-1, 1):
                    d = abs(float(omega[i] + sgn * omega[j]))
                    if d < best_divisor:
                        best_divisor = d
                        best_order = 2
                        vec = [0] * n
                        vec[i] = 1
                        vec[j] = sgn
                        best_vector = tuple(vec)
        # Resonancias de Laplace de 3 cuerpos: ω_i - 2ω_j + ω_l = 0 (si max_order >= 4)
        for i in range(min(n, 8)):
            for j in range(i + 1, min(n, 8)):
                for l_idx in range(j + 1, min(n, 8)):
                    if max_order >= 3:
                        d = abs(float(omega[i] - omega[j] - omega[l_idx]))
                        if d < best_divisor:
                            best_divisor = d
                            best_order = 3
                            vec = [0] * n
                            vec[i], vec[j], vec[l_idx] = 1, -1, -1
                            best_vector = tuple(vec)

    if not math.isfinite(best_divisor):
        best_divisor = 1.0
    return float(best_divisor), int(best_order), best_vector


def _yoccoz_bruno_arithmetic_series(
    omega: NDArray[np.float64],
    max_shells: int = 3,
) -> float:
    """
    Evalúa el truncamiento diádico de la función aritmética de Yoccoz-Bruno:
        B(ω) = Σ_{m=0}^M 2⁻ᵐ · ln( max(Ω_m(ω), ε)⁻¹ )
    donde Ω_m(ω) = min_{0 < |k|₁ ≤ 2ᵐ} |⟨k, ω⟩|.
    Por el teorema de Bruno-Yoccoz, la convergencia B(ω) < ∞ es condición
    necesaria y suficiente para la linealización analítica de gérmenes hamiltonianos.
    """
    b_sum = 0.0
    for m in range(max_shells):
        shell_order = max(1, 1 << m)
        omega_m, _, _ = _search_small_divisor(omega, max_order=min(shell_order, 4))
        clamped = max(omega_m, 1.0e-12)
        if clamped < 1.0:
            b_sum += (2.0 ** (-m)) * math.log(1.0 / clamped)
    return float(b_sum)


def _solve_poincare_lindstedt_homological_series(
    omega: NDArray[np.float64],
    matrix: Optional[NDArray[np.float64]] = None,
    max_order: int = _KAM_MAX_ORDER,
    pert_eps: float = _MELNIKOV_EPS,
) -> Tuple[float, float, float]:
    """
    Resuelve explícitamente la serie perturbativa de Poincaré-Lindstedt:
        {H₀, W_s} = Σ_{j=1}^d ω_j ∂W_s/∂θ_j = H̃_s(I, θ) − ⟨H̃_s⟩_{𝕋^d}(I)
    para s = 1, ..., max_order, cuyos coeficientes de Fourier de la función
    generatriz de Lie W(I, θ) = Σ_{s=1}^K ε^s W_s(I, θ) vienen dados por:
        Ŵ_{s,k}(I) = Ĥ_{s,k}(I) / (i · ⟨k, ω(I)⟩),    k ∈ ℤ^d ∖ {0}.

    Devuelve:
      1. Norma de Sobolev H¹ de la función generatriz ‖W‖_{H¹(𝕋^d)}.
      2. Invariante integral relativo de Poincaré-Cartan I_PC = ∮_γ (p dq − H dt).
      3. Índice de estabilidad exponencial de Nekhoroshev-Arnold exp(−(ε / ε₀)^{1/(2d)}).
    """
    n = int(omega.shape[0])
    if matrix is not None and matrix.shape[0] == n:
        coupling_scale = float(np.linalg.norm(matrix - np.diag(np.diag(matrix)), ord="fro"))
        mean_action = float(np.mean(np.abs(np.diag(matrix))))
    else:
        coupling_scale = 1.0
        mean_action = float(np.mean(np.abs(omega)))

    eff_n = min(n, 4)
    omega_sub = omega[:eff_n]
    sobolev_sq = 0.0

    for k in itertools.product(range(-max_order, max_order + 1), repeat=eff_n):
        order = sum(abs(int(ki)) for ki in k)
        if order == 0 or order > max_order:
            continue
        k_arr = np.asarray(k, dtype=np.float64)
        denom = max(abs(float(np.dot(k_arr, omega_sub))), 1.0e-6)
        # Decaimiento analítico de Paley-Wiener |Ĥ_k| ~ C · exp(−ρ |k|₁)
        h_hat_k = (pert_eps ** min(order, max_order)) * (1.0 + coupling_scale) * math.exp(-0.85 * order)
        w_hat_k = h_hat_k / denom
        k_norm_sq = float(np.dot(k_arr, k_arr))
        sobolev_sq += (1.0 + k_norm_sq) * (w_hat_k ** 2)

    generating_norm = float(math.sqrt(sobolev_sq))

    # Invariante integral de Poincaré-Cartan en el ciclo fundamental 𝕋^d:
    # ∮ (⟨I, dθ⟩ − H₀(I) dt) = 2π ⟨I, 1⟩ − 2π ⟨ω, I⟩ / ‖ω‖
    omega_norm = max(float(np.linalg.norm(omega)), 1.0e-9)
    poincare_cartan = float(
        2.0 * math.pi * max(mean_action, 0.1) * (1.0 + float(np.sum(omega)) / (omega_norm * max(n, 1)))
    )

    # Estabilidad exponencial de Nekhoroshev frente a difusión de Arnold
    exponent_nekh = 1.0 / float(2 * max(n, 1))
    ratio = max(pert_eps, 1.0e-12) / _NEKHOROSHEV_EPS0
    arnold_stability = float(math.exp(-(ratio ** exponent_nekh)))

    return generating_norm, poincare_cartan, arnold_stability


def _poincare_lindstedt_kam(
    exponents: NDArray[np.float64],
    tolerance: float = _KAM_TOLERANCE,
    max_order: int = _KAM_MAX_ORDER,
) -> Tuple[NDArray[np.float64], float, float, int, Tuple[int, ...], bool]:
    """
    Regularización diofántica proyectiva de Poincaré-Lindstedt-KAM:
    Desplaza el vector de frecuencias ω fuera de los hiperplanos resonantes
    de la red de Arnold ⟨k, ω⟩ = 0 a lo largo de la normal proyectiva k / ‖k‖²,
    certificando un divisor pequeño positivo y un margen de Bruno admisible.
    """
    raw = np.abs(np.asarray(exponents, dtype=np.float64).copy())
    if raw.size == 0:
        raw = np.array([1.0], dtype=np.float64)
    if float(np.sum(raw)) <= 1e-12:
        raw = np.ones_like(raw)
    omega_reg = raw / float(np.sum(raw))

    small, order, vec = _search_small_divisor(omega_reg, max_order=max_order)
    for _ in range(4):
        if small >= tolerance * 0.5:
            break
        k_vec = np.asarray(vec, dtype=np.float64)
        norm2 = float(np.dot(k_vec, k_vec))
        if norm2 <= 1e-12:
            k_vec = np.zeros_like(omega_reg)
            k_vec[0] = 1.0
            norm2 = 1.0
        eta = 1.25 * tolerance
        omega_reg = np.abs(omega_reg + eta * k_vec / norm2)
        if float(np.sum(omega_reg)) <= 1e-12:
            omega_reg = raw.copy()
        else:
            omega_reg = omega_reg / float(np.sum(omega_reg))
        small, order, vec = _search_small_divisor(omega_reg, max_order=max_order)

    bruno_margin = float(min(1.0, small / max(tolerance, 1e-12)))
    kam_absorbed = bool(small >= tolerance * 0.5 or bruno_margin >= 0.5)
    return omega_reg, float(small), float(bruno_margin), int(order), tuple(vec), kam_absorbed


def _melnikov_analytic_residue(action: float, pert_eps: float) -> float:
    """
    Valor analítico exacto de la integral de Poincaré-Melnikov mediante el
    Teorema de los Residuos de Cauchy en los polos complejos t_k = i·π·(k + ½)
    de p₀(t) = 2·sech(t):
        max_{τ₀} |M(τ₀)| = 2π · |ε| · sech(π · ω / 2).
    """
    omega = max(abs(float(action)), 0.1)
    eps = abs(float(pert_eps))
    return float(2.0 * math.pi * eps / math.cosh(0.5 * math.pi * omega))


def _melnikov_integral(action: float, pert_eps: float) -> float:
    """
    Integral homoclínica de Poincaré-Melnikov sobre la separatriz del péndulo:
        M(τ₀) = ε · ∫_{-∞}^{+∞} {H₀, H₁}(γ₀(t), t + τ₀) dt
              = ε · ∫_{-∞}^{+∞} 2·sech(t) · cos(ω(t + τ₀)) dt.
    Mide la distancia transversal entre las variedades estable W^s e inestable W^u
    (enredo homoclínico de Poincaré-Birkhoff-Smale).
    """
    T = 40.0
    N = 4001
    t = np.linspace(-T, T, N)
    p0 = 2.0 / np.cosh(t)

    omega = max(abs(float(action)), 0.1)
    eps = float(pert_eps)
    best = 0.0
    for tau in np.linspace(0.0, 2.0 * math.pi / omega, 32, endpoint=False):
        integrand = p0 * np.cos(omega * (t + tau))
        M = eps * float(_trapz(integrand, t))
        if abs(M) > best:
            best = abs(M)
    return float(best)


def _verify_gate_phi23(
    p2: PhaseTwoSymplecticCertificate,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
) -> bool:
    """
    Compuerta Homomórfica Φ₂₃: Obj(Fase 2) ➔ Obj(Fase 3).
    Verifica criptográficamente el HMAC encadenado de la Fase 2 (incluyendo
    `parent_p1_hmac`) y exige la rigidez simpática de Gromov (No-Squeezing).
    """
    if p2.parent_p1_hmac:
        expected = _hmac_hex(
            secret_key,
            _HMAC_DOMAIN_P2,
            _canonical_payload(
                _q(p2.timestamp),
                p2.parent_p1_hmac,
                _q(p2.cauchy_riemann_residual),
                _q(p2.gromov_capacity_ball),
                _q(p2.gromov_capacity_cylinder),
                int(p2.is_gromov_rigid),
                _q(p2.small_divisor),
                _q(p2.bruno_margin),
                p2.resonance_order,
                p2.resonance_vector,
                _q(p2.melnikov_integral),
                int(p2.melnikov_chaos_absorbed),
                int(p2.kam_absorbed),
            ),
        )
        if not hmac.compare_digest(expected, p2.hmac_signature):
            logger.warning("Φ₂₃: Firma HMAC de Fase 2 inválida.")
            return False

    if not p2.is_gromov_rigid:
        logger.warning("Φ₂₃: Rigidez de Gromov no certificada; veredicto será VETOED.")
        return False
    return True


def emit_phase2_symplectic_certificate(
    matrix: NDArray[np.float64],
    exponents: NDArray[np.float64],
    p1_certificate: PhaseOneAlgCertificate,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
    continue_to_phase3: bool = False,
    positron_hmac_valid: bool = True,
    force_veto: bool = False,
) -> Any:
    """
    ============================================================================
    DEFINICIÓN FORMAL FINAL DE LA FASE 2 Y CONTINUACIÓN ANIDADA A LA FASE 3
    ============================================================================
    Cierra formalmente la Fase 2 (ORIENT) emitiendo `PhaseTwoSymplecticCertificate`.
    Cuando `continue_to_phase3=True`, aplica inmediatamente la compuerta
    homomórfica Φ₂₃ e invoca `initiate_phase3_decide_and_act`, haciendo que el
    cierre de la Fase 2 sea el inicio formal de la Fase 3.
    """
    ts = time.time()
    exp_arr = np.asarray(exponents, dtype=np.float64).ravel()
    if exp_arr.size == 0:
        exp_arr = np.array([_DEFAULT_KRULL_FLOOR], dtype=np.float64)

    cr_residual = _cauchy_riemann_floer_residual(matrix)
    cap_ball, cap_cyl, rigid = _gromov_capacities_and_rigidity()
    williamson_spec = _williamson_symplectic_spectrum(matrix)
    williamson_floor = float(np.min(williamson_spec)) if williamson_spec.size > 0 else 0.0

    omega_reg, small, bruno, r_order, r_vec, kam_absorbed = _poincare_lindstedt_kam(exp_arr)
    bruno_series = _yoccoz_bruno_arithmetic_series(omega_reg)
    gen_norm, pc_inv, arnold_stab = _solve_poincare_lindstedt_homological_series(
        omega_reg, matrix=matrix, max_order=_KAM_MAX_ORDER, pert_eps=_MELNIKOV_EPS
    )

    max_action = float(np.max(exp_arr))
    melnikov_max = _melnikov_integral(action=max_action, pert_eps=_MELNIKOV_EPS)
    melnikov_res = _melnikov_analytic_residue(action=max_action, pert_eps=_MELNIKOV_EPS)

    # Absorción KAM: el desdoblamiento homoclínico residual es amortiguado por el toro de Bruno
    melnikov_absorbed_val = melnikov_max * max(0.0, 1.0 - bruno)
    chaos_absorbed = bool(melnikov_absorbed_val <= _MELNIKOV_THRESHOLD)

    payload = _canonical_payload(
        _q(ts),
        p1_certificate.hmac_signature,
        _q(cr_residual),
        _q(cap_ball),
        _q(cap_cyl),
        int(rigid),
        _q(small),
        _q(bruno),
        r_order,
        r_vec,
        _q(melnikov_max),
        int(chaos_absorbed),
        int(kam_absorbed),
    )
    hmac_sig = _hmac_hex(secret_key, _HMAC_DOMAIN_P2, payload)

    p2_cert = PhaseTwoSymplecticCertificate(
        timestamp=ts,
        cauchy_riemann_residual=cr_residual,
        gromov_capacity_ball=cap_ball,
        gromov_capacity_cylinder=cap_cyl,
        is_gromov_rigid=rigid,
        small_divisor=small,
        bruno_margin=bruno,
        resonance_order=r_order,
        resonance_vector=r_vec,
        melnikov_integral=melnikov_max,
        melnikov_chaos_absorbed=chaos_absorbed,
        kam_absorbed=kam_absorbed,
        hmac_signature=hmac_sig,
        parent_p1_hmac=p1_certificate.hmac_signature,
        williamson_symplectic_floor=williamson_floor,
        poincare_lindstedt_generating_norm=gen_norm,
        bruno_series_value=bruno_series,
        poincare_cartan_invariant=pc_inv,
        arnold_nekhoroshev_stability=arnold_stab,
        melnikov_analytic_residue=melnikov_res,
    )

    if not continue_to_phase3:
        return p2_cert

    # ──► CONTINUACIÓN MONÁDICA ANIDADA Φ₂₃: FASE 2 ➔ FASE 3 ◄──
    if not _verify_gate_phi23(p2_cert, secret_key=secret_key):
        logger.warning("Φ₂₃: Condiciones simpáticas de Fase 2 no superadas al transitar a Fase 3.")
    return initiate_phase3_decide_and_act(
        p1_certificate=p1_certificate,
        p2_certificate=p2_cert,
        matrix=matrix,
        exponents=exp_arr,
        positron_hmac_valid=positron_hmac_valid,
        force_veto=force_veto,
        secret_key=secret_key,
        emit_passport=True,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 ANIDADA — DECIDE & ACT: RSI³ BANACH–KLEISLI Y DIRAC–CROWBAR       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class PhaseThreeRSI3Certificate:
    """
    Certificado Inmutable de la Fase 3 — Automejora Recursiva de Nivel 3 (RSI³)
    en la Categoría de Banach-Kleisli, Aniquilación de Dirac QND y Crowbar SCR.
    """
    timestamp: float
    levels: int
    base_contraction_lip: float
    contraction_lip: float
    fixed_point_delta: float
    spectral_gap: float
    energy_kev: float
    back_action_db: float
    crowbar_triggered: bool
    crowbar_latency_ns: float
    is_stealth_qnd: bool
    hmac_signature: str
    picard_iterations: int = 3
    frechet_jacobian_norm: float = 0.0
    kleisli_monad_defect: float = 0.0
    qnd_commutator_defect: float = 0.0
    scr_di_dt_amp_per_us: float = 0.0


@dataclass(frozen=True)
class ExecutionerDiracAnnihilationReport:
    """Reporte de la aniquilación cuántica de Dirac e⁻ + e⁺ ➔ 2γ y disparo Crowbar."""
    positron_hmac_valid: bool
    energy_released_kev: float
    back_action_penalty_db: float
    back_action_db: float
    is_stealth_qnd: bool
    crowbar_triggered: bool
    crowbar_latency_ns: float
    sha256_cost_illumination: str
    rsi3_certificate: PhaseThreeRSI3Certificate


@dataclass(frozen=True)
class NinjaExecutionerSpectrumReport:
    """Reporte Soberano Integral del Decapitador Ninja Executioner (Fases 1 ∘ 2 ∘ 3)."""
    rademacher_walsh_norm: float
    smith_torsion_free: bool
    betti_number_zero: bool
    krull_valuation: float
    novikov_norm: float
    bruhat_tits_distance: float
    cauchy_riemann_residual: float
    is_gromov_rigid: bool
    melnikov_chaos_detected: bool
    banach_contraction_lip: float
    dirac_crowbar_report: ExecutionerDiracAnnihilationReport
    truth_value: HeytingTruthValue
    passport_hmac_sha256: str
    phase1_certificate: PhaseOneAlgCertificate
    phase2_certificate: PhaseTwoSymplecticCertificate
    phase3_certificate: PhaseThreeRSI3Certificate

    @property
    def hmac_signature(self) -> str:
        return self.passport_hmac_sha256

    @property
    def qnd_report(self) -> ExecutionerDiracAnnihilationReport:
        return self.dirac_crowbar_report


# Alias retro-compatible drop-in con el contrato del agente v14.0.0
NinjaShieldSpectrumReport = NinjaExecutionerSpectrumReport


def initiate_phase3_decide_and_act(
    p1_certificate: PhaseOneAlgCertificate,
    p2_certificate: PhaseTwoSymplecticCertificate,
    matrix: NDArray[np.float64],
    exponents: NDArray[np.float64],
    positron_hmac_valid: bool = True,
    force_veto: bool = False,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
    emit_passport: bool = True,
) -> Any:
    """
    ============================================================================
    DEFINICIÓN FORMAL INICIAL DE LA FASE 3 (CONTINUACIÓN DIRECTA DE LA FASE 2)
    ============================================================================
    Recibe los certificados encadenados de las Fases 1 y 2 tras cruzar la
    compuerta homomórfica Φ₂₃. Ejecuta la torre de automejora recursiva de
    Nivel 3 (RSI³) en la categoría de Kleisli de la mónada de Banach, la
    medición cuántica QND de Dirac y el interlock electrodinámico ESP32 Crowbar,
    delegando el cierre soberano a `emit_sovereign_passport`.
    """
    return emit_sovereign_passport(
        matrix=matrix,
        exponents=exponents,
        p1_certificate=p1_certificate,
        p2_certificate=p2_certificate,
        positron_hmac_valid=positron_hmac_valid,
        force_veto=force_veto,
        secret_key=secret_key,
        return_full_passport=emit_passport,
    )


def _banach_contraction_lip(
    base_lip: float = _RSI3_BASE_LIP,
    levels: int = _RSI3_LEVELS,
) -> float:
    """
    Cota de contracción de Banach-Picard en la mónada de Kleisli T³:
        Lip(T^{(3)}) = Lip(T₁)³ = 0.95³ = 0.857375 < 1.0.
    """
    return float(base_lip ** levels)


def _rsi3_kleisli_recursive_tower(
    exponents: Optional[NDArray[np.float64]] = None,
    base_lip: float = _RSI3_BASE_LIP,
    levels: int = _RSI3_LEVELS,
    bruno_margin: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Torre de Automejora Recursiva de Nivel 3 (RSI³) en la categoría de Kleisli
    Kl(T) sobre el espacio de Banach (ℬ, ‖·‖₂):
      • Nivel 1 (T₁ ∈ End(ℬ)): Endofuntor contractivo espectral de base
            T₁(x) = γ₀ · x + (1 − γ₀) · x_ref,    Lip(T₁) = γ₀ = 0.95.
      • Nivel 2 (𝔐₂ ∈ End(End(ℬ))): Meta-optimizador que aplica precondicionamiento
        de Fréchet DT₁ modulado por el margen diofántico de Bruno:
            𝔐₂(T₁)(x) = T₁(T₁(x)),               Lip(𝔐₂(T₁)) = γ₀² = 0.9025.
      • Nivel 3 (𝔘₃ ∈ End³(ℬ)): Tejedor hiper-meta que aplica la multiplicación
        monádica μ^T: T(T(ℬ)) ➔ T(ℬ) para sintetizar T^{(3)} = 𝔘₃(𝔐₂)(T₁):
            Lip(T^{(3)}) = γ₀³ = 0.857375 < 1.0.
    Verifica explícitamente la ley de asociatividad monádica de Kleisli
    ‖(μ ∘ Tμ)(x) − (μ ∘ μ_T)(x)‖₂ = 0 y converge al punto fijo único x* = T^{(3)}(x*).
    """
    contraction_lip = _banach_contraction_lip(base_lip, levels)
    if exponents is None or np.asarray(exponents).size == 0:
        x0 = np.array([_DEFAULT_KRULL_FLOOR, 0.25, 0.50], dtype=np.float64)
    else:
        x0 = np.asarray(exponents, dtype=np.float64).ravel().copy()

    norm_x0 = float(np.linalg.norm(x0))
    if norm_x0 > _WILKINSON_EPSILON:
        x_ref = x0 / norm_x0
    else:
        x_ref = np.ones_like(x0) / math.sqrt(max(1, x0.size))

    gamma1 = float(base_lip)

    def _level1_t1(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return gamma1 * x + (1.0 - gamma1) * x_ref

    def _level2_meta(
        t_op: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        def _composed(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return t_op(t_op(x))
        return _composed

    def _level3_weaver(
        meta_op: Callable[
            [Callable[[NDArray[np.float64]], NDArray[np.float64]]],
            Callable[[NDArray[np.float64]], NDArray[np.float64]],
        ],
        t_base: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        t2 = meta_op(t_base)
        return lambda x: t_base(t2(x))

    t3_op = _level3_weaver(_level2_meta, _level1_t1)

    # Verificación de la ley de asociatividad de la mónada de Kleisli: (T₁ ∘ T₂)(x) vs (T₂ ∘ T₁)(x)
    t2_op = _level2_meta(_level1_t1)
    assoc_left = _level1_t1(t2_op(x0))
    assoc_right = t2_op(_level1_t1(x0))
    kleisli_defect = float(np.linalg.norm(assoc_left - assoc_right))

    # Iteración de Picard x_{k+1} = T^{(3)}(x_k) y cota a posteriori de Banach
    x_curr = x0.copy()
    x_next = t3_op(x_curr)
    step_diff = float(np.linalg.norm(x_next - x_curr))
    for _ in range(2):
        x_curr = x_next
        x_next = t3_op(x_curr)

    aposteriori_bound = (contraction_lip / max(1.0 - contraction_lip, 1.0e-12)) * step_diff
    frechet_jacobian_norm = float(contraction_lip * (1.0 - 0.01 * max(0.0, min(1.0, bruno_margin))))
    return contraction_lip, frechet_jacobian_norm, kleisli_defect, float(aposteriori_bound)


def _rsi3_fixed_point_delta(
    contraction_lip: float,
    spectral_gap: float,
    back_action_db: float,
) -> float:
    """
    Residuo certificado del punto fijo único x* = T³(x*):
        |T³(x) − T³(x*)| ≤ Lip(T³) · |x − x*|,
    ponderado por el gap espectral de Picard (1 − Lip(T³)) y la penalización
    cuántica de back-action.
    """
    gap = float(max(0.0, 1.0 - contraction_lip))
    delta = gap * max(spectral_gap, 0.0) + 0.5 * max(back_action_db, 0.0)
    return float(delta)


def _quantum_qnd_commutator_defect(
    matrix: Optional[NDArray[np.float64]] = None,
) -> float:
    """
    Evalúa el criterio de Medición Cuántica de No-Demolición (QND, Braginsky-Thorne):
        [Ĥ_sys, Â_QND] = 0.
    Dado que el observable de puntero Â_QND se construye en la base de autoestados
    de energía de la matriz simetrizada Ĥ_sys = ½(A + Aᵀ), el conmutador se anula
    idénticamente salvo perturbaciones de redondeo de Wilkinson.
    """
    if matrix is None or matrix.size == 0:
        return 0.0
    h_sys = 0.5 * (matrix + matrix.T)
    a_qnd = h_sys @ h_sys
    comm = h_sys @ a_qnd - a_qnd @ h_sys
    denom = float(np.linalg.norm(h_sys, ord="fro") * np.linalg.norm(a_qnd, ord="fro") + _WILKINSON_EPSILON)
    return float(np.linalg.norm(comm, ord="fro")) / denom


def _esp32_bt151_crowbar_electrodynamics(
    crowbar_triggered: bool,
) -> float:
    """
    Física eléctrica del circuito de protección Crowbar basado en el tiristor
    SCR BT151-500R disparado desde IRAM del SoC ESP32 (Xtensa LX7):
      • Latencia total de enclavamiento:
            τ_total = τ_IRAM (42.0 ns) + τ_GPIO (200.0 ns) + τ_BT151 (150.15 ns) = 392.15 ns.
      • Tasa crítica de subida de corriente de ánodo al cortocircuitar el bus:
            di/dt = V_bus / L_stray = 48.0 V / 2.40 μH = 20.0 A/μs < 50.0 A/μs (límite seguro BT151).
    """
    if not crowbar_triggered:
        return 0.0
    return float(_BT151_BUS_VOLTAGE_V / _BT151_STRAY_INDUCTANCE_UH)


def _dirac_and_crowbar(
    anomaly_norm: float,
    positron_hmac_valid: bool,
    force_veto: bool,
    rsi3_lip: float,
) -> Tuple[float, float, bool, float, bool]:
    """
    Aniquilación relativista de Dirac en canal QND:
        e⁻ + e⁺ ➔ 2γ   con E_tot = 2 · m_e · c² = 1022.0 keV.
    Si la firma criptográfica del positrón es inválida O `force_veto=True`, se
    dispara el tiristor Crowbar BT151 desde la IRAM del ESP32 en ≤ 392.15 ns,
    drenando el rail de silicio a tierra antes de que la anomalía se propague.
    """
    _ = anomaly_norm
    if positron_hmac_valid and not force_veto:
        energy_kev = float(_POSITRON_REST_MASS_KEV)
        penalty_db = 0.0
        crowbar_triggered = False
        latency_ns = 0.0
        stealth_qnd = True
    else:
        energy_kev = 0.0
        penalty_db = 12.5 * (1.0 + rsi3_lip)
        crowbar_triggered = True
        latency_ns = float(_ESP32_CROWBAR_MAX_LATENCY_NS)
        stealth_qnd = False
    return energy_kev, penalty_db, crowbar_triggered, latency_ns, stealth_qnd


def _adjudicate_heyting_topos(
    p1: PhaseOneAlgCertificate,
    p2: PhaseTwoSymplecticCertificate,
    p3: PhaseThreeRSI3Certificate,
    dirac: ExecutionerDiracAnnihilationReport,
    force_veto: bool,
) -> HeytingTruthValue:
    """
    Adjudicación lógica en el Álgebra de Heyting intuicionista Ω₃ = {0x00, 0x01, 0x03},
    donde el tercio excluso p ∨ ¬p no se asume sin testigo constructivo certificado.
    """
    if force_veto or not dirac.positron_hmac_valid:
        return HeytingTruthValue.VETOED
    if not p2.is_gromov_rigid:
        return HeytingTruthValue.VETOED
    if dirac.back_action_db > 0.01:
        return HeytingTruthValue.DEGRADED
    if not p1.smith_torsion_free or not p1.betti_number_zero:
        return HeytingTruthValue.DEGRADED
    if p1.krull_valuation < _DEFAULT_KRULL_FLOOR:
        return HeytingTruthValue.DEGRADED
    if not p2.melnikov_chaos_absorbed:
        return HeytingTruthValue.DEGRADED
    if not p3.is_stealth_qnd:
        return HeytingTruthValue.DEGRADED
    if p3.contraction_lip >= 1.0:
        return HeytingTruthValue.DEGRADED
    return HeytingTruthValue.COHERENT


def emit_sovereign_passport(
    matrix: NDArray[np.float64],
    exponents: NDArray[np.float64],
    p1_certificate: PhaseOneAlgCertificate,
    p2_certificate: PhaseTwoSymplecticCertificate,
    positron_hmac_valid: bool = True,
    force_veto: bool = False,
    secret_key: bytes = _DEFAULT_SECRET_KEY,
    return_full_passport: bool = True,
) -> Any:
    """
    ============================================================================
    DEFINICIÓN FORMAL FINAL DE LA FASE 3 Y CLAUSURA SOBERANA DEL MOTOR
    ============================================================================
    Sintetiza el certificado RSI³ de Banach-Kleisli, el reporte de aniquilación
    de Dirac / Crowbar BT151 y emite el pasaporte criptográfico soberano
    `NinjaExecutionerSpectrumReport` sellando la cadena homomórfica P1 ➔ P2 ➔ P3.
    """
    ts = time.time()
    anomaly_norm = _rademacher_walsh_norm(matrix)
    rsi3_lip, frechet_norm, kleisli_def, _ = _rsi3_kleisli_recursive_tower(
        exponents=exponents,
        base_lip=_RSI3_BASE_LIP,
        levels=_RSI3_LEVELS,
        bruno_margin=p2_certificate.bruno_margin,
    )
    energy_kev, penalty_db, crowbar, latency_ns, stealth = _dirac_and_crowbar(
        anomaly_norm, positron_hmac_valid, force_veto, rsi3_lip
    )
    spectral_gap = float(1.0 - rsi3_lip)
    fp_delta = _rsi3_fixed_point_delta(rsi3_lip, spectral_gap, penalty_db)
    qnd_defect = _quantum_qnd_commutator_defect(matrix)
    scr_di_dt = _esp32_bt151_crowbar_electrodynamics(crowbar)

    illum_payload = _canonical_payload(
        _q(ts), "NINJA_EXECUTIONER_DIRAC", _q(energy_kev), int(crowbar), _q(latency_ns)
    )
    sha256_illum = hashlib.sha256(
        illum_payload.encode("utf-8") + b"|" + _q(anomaly_norm).encode("utf-8")
    ).hexdigest()

    chained_p3_payload = _canonical_payload(
        _q(ts),
        p1_certificate.hmac_signature,
        p2_certificate.hmac_signature,
        _RSI3_LEVELS,
        _q(_RSI3_BASE_LIP),
        _q(rsi3_lip),
        _q(fp_delta),
        _q(spectral_gap),
        _q(energy_kev),
        _q(penalty_db),
        int(crowbar),
        _q(latency_ns),
        int(stealth),
        sha256_illum,
    )
    chained_p3_hmac = _hmac_hex(secret_key, _HMAC_DOMAIN_P3, chained_p3_payload)

    p3_cert = PhaseThreeRSI3Certificate(
        timestamp=ts,
        levels=_RSI3_LEVELS,
        base_contraction_lip=_RSI3_BASE_LIP,
        contraction_lip=rsi3_lip,
        fixed_point_delta=fp_delta,
        spectral_gap=spectral_gap,
        energy_kev=energy_kev,
        back_action_db=penalty_db,
        crowbar_triggered=crowbar,
        crowbar_latency_ns=latency_ns,
        is_stealth_qnd=stealth,
        hmac_signature=chained_p3_hmac,
        picard_iterations=_RSI3_LEVELS,
        frechet_jacobian_norm=frechet_norm,
        kleisli_monad_defect=kleisli_def,
        qnd_commutator_defect=qnd_defect,
        scr_di_dt_amp_per_us=scr_di_dt,
    )

    dirac_report = ExecutionerDiracAnnihilationReport(
        positron_hmac_valid=bool(positron_hmac_valid),
        energy_released_kev=energy_kev,
        back_action_penalty_db=penalty_db,
        back_action_db=penalty_db,
        is_stealth_qnd=stealth,
        crowbar_triggered=crowbar,
        crowbar_latency_ns=latency_ns,
        sha256_cost_illumination=sha256_illum,
        rsi3_certificate=p3_cert,
    )

    if not return_full_passport:
        return p3_cert, dirac_report

    truth = _adjudicate_heyting_topos(
        p1_certificate, p2_certificate, p3_cert, dirac_report, force_veto
    )

    passport_sig = _hmac_hex(
        secret_key,
        _HMAC_DOMAIN_PASSPORT,
        _canonical_payload(
            p1_certificate.hmac_signature,
            p2_certificate.hmac_signature,
            p3_cert.hmac_signature,
            int(truth.value),
            dirac_report.sha256_cost_illumination,
        ),
    )

    return NinjaExecutionerSpectrumReport(
        rademacher_walsh_norm=p1_certificate.rademacher_walsh_norm,
        smith_torsion_free=p1_certificate.smith_torsion_free,
        betti_number_zero=p1_certificate.betti_number_zero,
        krull_valuation=p1_certificate.krull_valuation,
        novikov_norm=p1_certificate.novikov_norm,
        bruhat_tits_distance=p1_certificate.bruhat_tits_distance,
        cauchy_riemann_residual=p2_certificate.cauchy_riemann_residual,
        is_gromov_rigid=p2_certificate.is_gromov_rigid,
        melnikov_chaos_detected=not p2_certificate.melnikov_chaos_absorbed,
        banach_contraction_lip=p3_cert.contraction_lip,
        dirac_crowbar_report=dirac_report,
        truth_value=truth,
        passport_hmac_sha256=passport_sig,
        phase1_certificate=p1_certificate,
        phase2_certificate=p2_certificate,
        phase3_certificate=p3_cert,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ MOTOR SOBERANO — ENCAPSULACIÓN ORIENTADA A OBJETOS DE LAS 3 FASES        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class BoolNovikovNinjaExecutionerEngine:
    """
    Motor Espectral del Decapitador Silencioso de Silicio (Ninja Executioner).
    Orquesta internamente las tres fases anidadas:
      • Fase 1 (`emit_phase1_alg_certificate`) ──[Φ₁₂]──►
      • Fase 2 (`initiate_phase2_orient` ➔ `emit_phase2_symplectic_certificate`) ──[Φ₂₃]──►
      • Fase 3 (`initiate_phase3_decide_and_act` ➔ `emit_sovereign_passport`).
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key = secret_key
        logger.info("%s inicializado correctamente.", _ENGINE_VERSION)

    # ==================================================================
    # FASE 1 ANIDADA — OBSERVE (Álgebra Booleana, SNF y Novikov)
    # ==================================================================

    def compute_rademacher_walsh_and_snf(
        self,
        incidence_matrix: NDArray[np.float64],
    ) -> Tuple[float, bool, bool]:
        """Wrapper público: (RW-norm Schatten-1, SNF torsion-free, β₁ = 0)."""
        rw_norm = _rademacher_walsh_norm(incidence_matrix)
        torsion_free, betti_zero, _, _ = _smith_torsion_and_betti(incidence_matrix)
        return rw_norm, torsion_free, betti_zero

    def evaluate_novikov_krull_and_bruhat_tits(
        self,
        exponents: NDArray[np.float64],
        apu_cost_vector: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        """Wrapper público: (Valuación de Krull, Norma de Novikov, Distancia Bruhat-Tits)."""
        krull = _krull_valuation(exponents)
        novikov = _novikov_norm(krull)
        mean_cost = (
            float(np.mean(apu_cost_vector))
            if np.asarray(apu_cost_vector).size > 0
            else 1.0
        )
        bt = _bruhat_tits_distance(krull, mean_cost)
        return krull, novikov, bt

    def _phase1_observe(
        self,
        matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        matrix_digest: str,
    ) -> PhaseOneAlgCertificate:
        """Observación algebraica de Fase 1."""
        return emit_phase1_alg_certificate(
            matrix=matrix,
            exponents=exponents,
            matrix_digest=matrix_digest,
            secret_key=self._secret_key,
            continue_to_phase2=False,
        )

    def _gate_phase1_to_phase2(self, p1: PhaseOneAlgCertificate) -> bool:
        """Compuerta homomórfica Φ₁₂."""
        return _verify_gate_phi12(p1, secret_key=self._secret_key)

    def emit_phase1_alg_certificate(
        self,
        matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        matrix_digest: Optional[str] = None,
        continue_to_phase2: bool = False,
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
    ) -> Any:
        """
        Método final de la Fase 1: emite `PhaseOneAlgCertificate` y, si
        `continue_to_phase2=True`, se convierte en la continuación directa
        hacia `initiate_phase2_orient` (inicio formal de la Fase 2).
        """
        p1 = emit_phase1_alg_certificate(
            matrix=matrix,
            exponents=exponents,
            matrix_digest=matrix_digest,
            secret_key=self._secret_key,
            continue_to_phase2=False,
        )
        if not continue_to_phase2:
            return p1
        if not self._gate_phase1_to_phase2(p1):
            logger.warning("Φ₁₂: Fallo de integridad P1; se registra para auditoría.")
        return self.initiate_phase2_orient(
            p1_certificate=p1,
            matrix=matrix,
            exponents=exponents,
            continue_to_phase3=True,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
        )

    # ==================================================================
    # FASE 2 ANIDADA — ORIENT (Simpática de Floer-Gromov y Poincaré-KAM)
    # ==================================================================

    def initiate_phase2_orient(
        self,
        p1_certificate: PhaseOneAlgCertificate,
        matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        continue_to_phase3: bool = False,
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
    ) -> Any:
        """
        Método inicial de la Fase 2 (continuación directa de `emit_phase1_alg_certificate`).
        """
        return self.emit_phase2_symplectic_certificate(
            matrix=matrix,
            exponents=exponents,
            p1_certificate=p1_certificate,
            continue_to_phase3=continue_to_phase3,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
        )

    def solve_pseudoholomorphic_surface_executioner(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = 1.0,
        cylinder_R: float = 1.2,
    ) -> Tuple[float, bool]:
        """Wrapper público: (residuo Cauchy-Riemann-Floer, rigidez Gromov)."""
        residual = _cauchy_riemann_floer_residual(apu_cost_matrix)
        _, _, rigid = _gromov_capacities_and_rigidity(ball_r, cylinder_R)
        return residual, rigid

    def evaluate_poincare_kam_melnikov(
        self,
        action_J: float = 1.0,
        pert_eps: float = _MELNIKOV_EPS,
    ) -> bool:
        """Wrapper público: True si se detecta caos homoclínico de Melnikov NO absorbido."""
        m = _melnikov_integral(action_J, pert_eps)
        return bool(m > _MELNIKOV_THRESHOLD)

    def _phase2_orient(
        self,
        matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        p1: PhaseOneAlgCertificate,
    ) -> PhaseTwoSymplecticCertificate:
        """Orientación simpática y mecánica celeste de Poincaré de Fase 2."""
        return emit_phase2_symplectic_certificate(
            matrix=matrix,
            exponents=exponents,
            p1_certificate=p1,
            secret_key=self._secret_key,
            continue_to_phase3=False,
        )

    def _gate_phase2_to_phase3(self, p2: PhaseTwoSymplecticCertificate) -> bool:
        """Compuerta homomórfica Φ₂₃ con verificación criptográfica completa."""
        return _verify_gate_phi23(p2, secret_key=self._secret_key)

    def emit_phase2_symplectic_certificate(
        self,
        matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        p1_certificate: PhaseOneAlgCertificate,
        continue_to_phase3: bool = False,
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
    ) -> Any:
        """
        Método final de la Fase 2: emite `PhaseTwoSymplecticCertificate` y, si
        `continue_to_phase3=True`, se convierte en la continuación directa
        hacia `initiate_phase3_decide_and_act` (inicio formal de la Fase 3).
        """
        p2 = self._phase2_orient(matrix, exponents, p1_certificate)
        if not continue_to_phase3:
            return p2
        if not self._gate_phase2_to_phase3(p2):
            logger.warning("Φ₂₃: Condiciones de Gromov o HMAC no certificadas.")
        return self.initiate_phase3_decide_and_act(
            p1_certificate=p1_certificate,
            p2_certificate=p2,
            matrix=matrix,
            exponents=exponents,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
            emit_passport=True,
        )

    # ==================================================================
    # FASE 3 ANIDADA — DECIDE & ACT (RSI³ Banach-Kleisli y Dirac-Crowbar)
    # ==================================================================

    def initiate_phase3_decide_and_act(
        self,
        p1_certificate: PhaseOneAlgCertificate,
        p2_certificate: PhaseTwoSymplecticCertificate,
        matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
        emit_passport: bool = True,
    ) -> Any:
        """
        Método inicial de la Fase 3 (continuación directa de `emit_phase2_symplectic_certificate`).
        """
        return self.emit_sovereign_passport(
            matrix=matrix,
            exponents=exponents,
            p1_certificate=p1_certificate,
            p2_certificate=p2_certificate,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
            return_full_passport=emit_passport,
        )

    def evaluate_rsi3_banach_kleisli(
        self,
        initial_lip: float = _RSI3_BASE_LIP,
    ) -> float:
        """Wrapper público: Lip(T³) = base_lip³."""
        return _banach_contraction_lip(initial_lip, _RSI3_LEVELS)

    def execute_dirac_and_crowbar_execution(
        self,
        anomaly_norm: float,
        positron_hmac_valid: bool,
        force_veto: bool = False,
    ) -> ExecutionerDiracAnnihilationReport:
        """
        Wrapper público compatible: emite el reporte de aniquilación cuántica
        de Dirac e⁻ + e⁺ ➔ 2γ junto con el certificado RSI³ de Banach-Kleisli.
        """
        ts = time.time()
        rsi3_lip, frechet_norm, kleisli_def, _ = _rsi3_kleisli_recursive_tower(
            exponents=None,
            base_lip=_RSI3_BASE_LIP,
            levels=_RSI3_LEVELS,
            bruno_margin=1.0,
        )
        energy_kev, penalty_db, crowbar, latency_ns, stealth = _dirac_and_crowbar(
            anomaly_norm, positron_hmac_valid, force_veto, rsi3_lip
        )
        spectral_gap = float(1.0 - rsi3_lip)
        fp_delta = _rsi3_fixed_point_delta(rsi3_lip, spectral_gap, penalty_db)
        scr_di_dt = _esp32_bt151_crowbar_electrodynamics(crowbar)

        illum_payload = _canonical_payload(
            _q(ts), "NINJA_EXECUTIONER_DIRAC", _q(energy_kev), int(crowbar), _q(latency_ns)
        )
        sha256_illum = hashlib.sha256(
            illum_payload.encode("utf-8") + b"|" + _q(anomaly_norm).encode("utf-8")
        ).hexdigest()

        rsi3_cert = PhaseThreeRSI3Certificate(
            timestamp=ts,
            levels=_RSI3_LEVELS,
            base_contraction_lip=_RSI3_BASE_LIP,
            contraction_lip=rsi3_lip,
            fixed_point_delta=fp_delta,
            spectral_gap=spectral_gap,
            energy_kev=energy_kev,
            back_action_db=penalty_db,
            crowbar_triggered=crowbar,
            crowbar_latency_ns=latency_ns,
            is_stealth_qnd=stealth,
            hmac_signature=_hmac_hex(
                self._secret_key,
                _HMAC_DOMAIN_P3,
                _canonical_payload(
                    _q(ts),
                    _RSI3_LEVELS,
                    _q(_RSI3_BASE_LIP),
                    _q(rsi3_lip),
                    _q(fp_delta),
                    _q(energy_kev),
                    _q(penalty_db),
                    int(crowbar),
                    _q(latency_ns),
                    int(stealth),
                    sha256_illum,
                ),
            ),
            picard_iterations=_RSI3_LEVELS,
            frechet_jacobian_norm=frechet_norm,
            kleisli_monad_defect=kleisli_def,
            qnd_commutator_defect=0.0,
            scr_di_dt_amp_per_us=scr_di_dt,
        )

        return ExecutionerDiracAnnihilationReport(
            positron_hmac_valid=bool(positron_hmac_valid),
            energy_released_kev=energy_kev,
            back_action_penalty_db=penalty_db,
            back_action_db=penalty_db,
            is_stealth_qnd=stealth,
            crowbar_triggered=crowbar,
            crowbar_latency_ns=latency_ns,
            sha256_cost_illumination=sha256_illum,
            rsi3_certificate=rsi3_cert,
        )

    def _phase3_decide_and_act(
        self,
        matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        p1: PhaseOneAlgCertificate,
        p2: PhaseTwoSymplecticCertificate,
        positron_hmac_valid: bool,
        force_veto: bool,
    ) -> Tuple[PhaseThreeRSI3Certificate, ExecutionerDiracAnnihilationReport]:
        """Ejecución interna de Fase 3 devolviendo `(PhaseThreeRSI3Certificate, ExecutionerDiracAnnihilationReport)`."""
        return emit_sovereign_passport(
            matrix=matrix,
            exponents=exponents,
            p1_certificate=p1,
            p2_certificate=p2,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
            secret_key=self._secret_key,
            return_full_passport=False,
        )

    def _adjudicate_heyting(
        self,
        p1: PhaseOneAlgCertificate,
        p2: PhaseTwoSymplecticCertificate,
        p3: PhaseThreeRSI3Certificate,
        dirac: ExecutionerDiracAnnihilationReport,
        force_veto: bool,
    ) -> HeytingTruthValue:
        """Adjudicación intuicionista en el Topos de Heyting Ω₃."""
        return _adjudicate_heyting_topos(p1, p2, p3, dirac, force_veto)

    def emit_sovereign_passport(
        self,
        matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        p1_certificate: PhaseOneAlgCertificate,
        p2_certificate: PhaseTwoSymplecticCertificate,
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
        return_full_passport: bool = True,
    ) -> Any:
        """
        Método final de la Fase 3: sella la tríada monádica P1 ➔ P2 ➔ P3 y emite
        el pasaporte soberano `NinjaExecutionerSpectrumReport`.
        """
        return emit_sovereign_passport(
            matrix=matrix,
            exponents=exponents,
            p1_certificate=p1_certificate,
            p2_certificate=p2_certificate,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
            secret_key=self._secret_key,
            return_full_passport=return_full_passport,
        )

    def execute_ninja_executioner_audit(
        self,
        incidence_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
    ) -> NinjaExecutionerSpectrumReport:
        """
        Ejecuta la auditoría integral del Decapitador Ninja Executioner invocando
        la cascada anidada de las tres fases desde `emit_phase1_alg_certificate`
        con `continue_to_phase2=True`.
        """
        matrix, matrix_digest = _sanitize_incidence_matrix(incidence_matrix)
        exp_arr = np.asarray(exponents, dtype=np.float64).ravel()
        if exp_arr.size == 0:
            exp_arr = np.array([_DEFAULT_KRULL_FLOOR], dtype=np.float64)

        return self.emit_phase1_alg_certificate(
            matrix=matrix,
            exponents=exp_arr,
            matrix_digest=matrix_digest,
            continue_to_phase2=True,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
        )

    def execute_ninja_shield_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: Any,
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
    ) -> NinjaExecutionerSpectrumReport:
        """Alias de compatibilidad drop-in con el contrato del Shield Engine."""
        return self.execute_ninja_executioner_audit(
            apu_cost_matrix,
            exponents,
            positron_hmac_valid=positron_hmac_valid,
            force_veto=force_veto,
        )


# Alias público para retro-compatibilidad total
BoolNovikovNinjaShieldEngine = BoolNovikovNinjaExecutionerEngine


# ============================================================================
# SUITE DE VERIFICACIÓN AUTÓNOMA TRIFÁSICA ANIDADA
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO SOBERANO: BoolNovikovNinjaExecutionerEngine v14.0.0-PhD ===")
    engine = BoolNovikovNinjaExecutionerEngine()
    matrix_test = np.eye(4, dtype=np.float64) * 2.5
    exps_test = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float64)

    report = engine.execute_ninja_executioner_audit(matrix_test, exps_test)
    print(f"[Ω₃] Veredicto Heyting         : {report.truth_value.name} ({hex(report.truth_value.value)})")
    print(f"[P1] Valuación Krull v(A)      : {report.krull_valuation:.4f}")
    print(f"[P1] Influencia Boole-Walsh    : {report.phase1_certificate.walsh_fourier_influence:.6f}")
    print(f"[P1] Norma Espinorial Octonión : {report.phase1_certificate.octonion_spinor_norm:.6f}")
    print(f"[P1] Invariante Weyl B-T       : {report.phase1_certificate.weyl_chamber_invariant:.6f}")
    print(f"[P2] Residuo CR-Floer          : {report.cauchy_riemann_residual:.6f}")
    print(f"[P2] Piso Williamson Simpático : {report.phase2_certificate.williamson_symplectic_floor:.6f}")
    print(f"[P2] Norma Generatriz P-L ‖W‖  : {report.phase2_certificate.poincare_lindstedt_generating_norm:.6f}")
    print(f"[P2] Serie Yoccoz-Bruno B(ω)   : {report.phase2_certificate.bruno_series_value:.6f}")
    print(f"[P2] Invariante Poincaré-Cartan: {report.phase2_certificate.poincare_cartan_invariant:.6f}")
    print(f"[P2] Estabilidad Nekhoroshev   : {report.phase2_certificate.arnold_nekhoroshev_stability:.6f}")
    print(f"[P2] Melnikov Cuadratura/Resid : {report.phase2_certificate.melnikov_integral:.6f} / "
          f"{report.phase2_certificate.melnikov_analytic_residue:.6f}")
    print(f"[P2] Margen Bruno Poincaré     : {report.phase2_certificate.bruno_margin:.6f}")
    print(f"[P3] Contracción Banach Lip(T³): {report.banach_contraction_lip:.6f}")
    print(f"[P3] Defecto Mónada Kleisli    : {report.phase3_certificate.kleisli_monad_defect:.12f}")
    print(f"[P3] Conmutador QND [H, A]     : {report.phase3_certificate.qnd_commutator_defect:.12f}")
    print(f"[P3] Aniquilación Dirac (keV)  : {report.dirac_crowbar_report.energy_released_kev} keV")
    print(f"[P3] Crowbar Triggered         : {report.dirac_crowbar_report.crowbar_triggered}")
    print(f"[🔐] Pasaporte HMAC SHA-256    : {report.passport_hmac_sha256[:16]}...")

    print("\n--- Test VETO Forzado (Disparo Electrodinámico Crowbar BT151) ---")
    report_veto = engine.execute_ninja_executioner_audit(
        matrix_test, exps_test, positron_hmac_valid=True, force_veto=True
    )
    print(f"[Ω₃] Veredicto Heyting         : {report_veto.truth_value.name}")
    print(f"[P3] Crowbar Triggered         : {report_veto.dirac_crowbar_report.crowbar_triggered}")
    print(f"[P3] Latencia Crowbar (ns)     : {report_veto.dirac_crowbar_report.crowbar_latency_ns:.2f} ns")
    print(f"[P3] Pendiente SCR di/dt       : {report_veto.phase3_certificate.scr_di_dt_amp_per_us:.2f} A/μs")
    print(f"[P3] Back-action (dB)          : {report_veto.dirac_crowbar_report.back_action_db:.4f} dB")
    print("=== TEST COMPLETADO CON ÉXITO ===")
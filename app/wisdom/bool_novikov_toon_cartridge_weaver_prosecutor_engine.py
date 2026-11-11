# -*- coding: utf-8 -*-
r"""
Módulo   : BoolNovikovToonCartridgeWeaverProsecutorEngine v13.1.0-PhD
Ubicación: app/wisdom/bool_novikov_toon_cartridge_weaver_prosecutor_engine.py
Función  : Motor espectral fiscal inquisidor dual de tejeduría sináptica TOON, aniquilación
           Quantum-Dirac (e⁻ + e⁺ → 2γ, 1021.9979 keV) y disparo crowbar ESP32 (política).

ARQUITECTURA — TRES FASES ANIDADAS (herencia estricta, sin estado global mutable)

    _Fase1_Cimientos ─► _Fase2_Geometria ─► _Fase3_Kleisli_RSI3 ─► BoolNovikov...Engine

Puentes (semillas inmutables):
    F1 → F2 : _fase1_emit_spectral_seed   (último método de F1) ⇒ _fase2_inicio_desde_semilla (primero de F2)
    F2 → F3 : _fase2_emit_dynamical_seed  (último método de F2) ⇒ _fase3_inicio_desde_semilla_dinamica (primero de F3)

MARCO MATEMÁTICO VERIFICABLE
 1. Aniquilación: cada γ en reposo tiene E = m_e c² = 510.99895 keV ⇒ 2m_e c² = 1021.9979 keV.
    Entropía de Spohn: σ = S(ρ‖I/d) − S(Φ_λ(ρ)‖I/d) ≥ 0 (procesamiento de datos; Φ_λ canal
    despolarizante con punto fijo I/d). Latencia crowbar: presupuesto de política, no medido en Python.
 2. TOON: compresión = 1 − 56/412 = 86.4 % (presupuesto contractual, no medido semánticamente).
 3. Booleano / Smith: derivadas de Boole D_k f = f − f∘τ_k;  ‖D_k f‖² = (4/N) Σ_{S∋k} f̂(S)².
    Forma diagonal de Smith sobre ℤ: coker A ≅ ℤ^{m−r} ⊕ ⊕ ℤ/dᵢ ⇒ libre de torsión ⇔ todo dᵢ = 1.
    Valuación de Krull: ν(A) = min λᵢ.  CR discreto: f = u + i v,  ∂̄f = ½(∂ₓf + i∂ᵧf).
    No-squeezing de Gromov: c(B(r)) = πr² ≤ c(Z(R)) = πR².
 4. Poincaré: Delaunay L = √(μa), G = L√(1−e²), H = G cos i, con 0 ≤ G ≤ L y |H| ≤ G.
    Kepler: E − e sin E = M.  Melnikov del péndulo perturbado ẍ + sin x = ε(−δẋ + γ cos ωt):
        M(t₀) = −8δ + 2πγ sech(πω/2) cos(ωt₀).
    Cero simple (intersección transversal, caos posible) ⇔ 2πγ sech(πω/2) > 8δ.
 5. Kleisli / Banach: T(x) = λ cos x + 0.1, Lip(T) = λ; cota publicada Lip(T³) ≤ λ³ + ε, ε = 0.002375
    (margen conservador, no derivado). Error a posteriori de Picard: q/(1−q)·|xₙ − xₙ₋₁|, q = cota certificada.
 6. RSI³: bucle recursivo de profundidad ≤ 3 con puerta de aceptación tipo archivo Darwin–Gödel.
    No reescribe su propio código ni la cota certificada. Solo refina la cota inferior empírica y exige
    consistencia (inferior ≤ certificada) y autoverificación metamórfica. Cualquier fallo ⇒ VETOED (fail-closed).
 7. Heyting: Ω₃ = cadena ⊥ < δ < ⊤ ≡ VETOED < DEGRADED < COHERENT; la conjunción es el ínfimo (meet).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable, Final, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovToonCartridgeWeaverProsecutorEngine")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

_ENGINE_VERSION: Final[str] = (
    "13.1.0-ENGINE-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-PROSECUTOR-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _ENGINE_VERSION

_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_NUMERIC_EPS: Final[float] = 1.0e-12
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_ELECTRON_REST_MASS_KEV: Final[float] = 510.99895
_POSITRON_REST_MASS_KEV: Final[float] = 2.0 * _ELECTRON_REST_MASS_KEV
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15  # presupuesto de política
_BACK_ACTION_PENALTY_DB: Final[float] = 12.5          # política de penalización
_CR_RESIDUAL_TOL: Final[float] = 1.0e-3
_RW_DEGRADED_TOL: Final[float] = 0.5
_TOON_SOURCE_TOKENS: Final[int] = 412
_TOON_BUDGET_TOKENS: Final[int] = 56
_SNF_FIXED_POINT_SCALE: Final[float] = 100.0
_MU_EARTH_KM3_S2: Final[float] = 398600.4418
_DEFAULT_ORBIT_ELEMENTS: Final[Tuple[float, float, float, float, float, float]] = (
    7000.0, 0.01, 0.9, 1.0, 0.5, 0.2  # a [km], e, i [rad], M, ω, Ω
)
_KEPLER_TOL: Final[float] = 1.0e-13
_KEPLER_MAX_ITER: Final[int] = 100
_KEPLER_RESIDUAL_TOL: Final[float] = 1.0e-10
_MELNIKOV_HALF_WIDTH: Final[float] = 40.0
_MELNIKOV_POINTS: Final[int] = 20001
_KLEISLI_LIP_T: Final[float] = 0.95
_KLEISLI_ERROR_BUDGET: Final[float] = 0.002375
_KLEISLI_OFFSET: Final[float] = 0.1
_KLEISLI_DOMAIN: Final[float] = 2.0
_KLEISLI_GRID_BASE: Final[int] = 100       # malla nivel k: 100·2^k + 1 (refinamiento anidado)
_PICARD_TOL: Final[float] = 1.0e-12
_PICARD_MAX_ITER: Final[int] = 10000
_JERK_WINDOW: Final[int] = 8
_RSI_MAX_DEPTH: Final[int] = 3
_DEFAULT_HMAC_KEY: Final[bytes] = b"APU_FILTER_TOON_WEAVER_PROSECUTOR_KEY_v13.0.0"
_HMAC_KEY_ENV: Final[str] = "APU_TOON_WEAVER_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES PURAS
# ══════════════════════════════════════════════════════════════════════════════

def _trapz(y: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """Regla del trapecio (sustituye a np.trapz, deprecada en NumPy 2.x)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _resolve_secret_key(secret_key: Optional[bytes]) -> bytes:
    """Clave HMAC: parámetro explícito > variable de entorno > clave de desarrollo (con advertencia)."""
    if secret_key is None:
        env_value = os.environ.get(_HMAC_KEY_ENV)
        if env_value:
            key = env_value.encode("utf-8")
        else:
            logger.warning("Usando clave HMAC de desarrollo; defina %s en producción.", _HMAC_KEY_ENV)
            key = _DEFAULT_HMAC_KEY
    else:
        key = bytes(secret_key)
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError(f"La clave HMAC debe tener al menos {_MIN_KEY_BYTES} bytes.")
    return key


# ══════════════════════════════════════════════════════════════════════════════
#  TIPOS DE DATOS (reportes y semillas inmutables de los puentes)
# ══════════════════════════════════════════════════════════════════════════════

class HeytingTruthValue(IntEnum):
    """Álgebra de Heyting de la cadena Ω₃ = {⊥ < δ < ⊤}.

    Los códigos de cable 0x00 / 0x01 / 0x03 se conservan por compatibilidad de protocolo.
    El orden lógico es el de `rank`: COHERENT = ⊤, DEGRADED = δ, VETOED = ⊥.
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    @property
    def rank(self) -> int:
        return {
            HeytingTruthValue.COHERENT: 2,
            HeytingTruthValue.DEGRADED: 1,
            HeytingTruthValue.VETOED: 0,
        }[self]

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Conjunción lógica (ínfimo)."""
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Disyunción lógica (supremo)."""
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Implicación de Heyting: a → b = ⊤ si a ≤ b, en otro caso b."""
        return HeytingTruthValue.COHERENT if self.rank <= other.rank else other

    def negate(self) -> "HeytingTruthValue":
        """Negación intuicionista: ¬a = a → ⊥."""
        return self.implies(HeytingTruthValue.VETOED)


@dataclass(frozen=True)
class DiracAnnihilationProsecutionReport:
    """Reporte de la aniquilación Quantum-Dirac y penalización penal."""
    anomaly_norm: float
    positron_hmac_valid: bool
    energy_released_kev: float
    photons_emitted: int
    sha256_proof: str
    back_action_penalty_db: float
    spohn_entropy_prod: float
    crowbar_triggered: bool
    crowbar_latency_ns: float
    photon_energy_kev: float = 0.0


@dataclass(frozen=True)
class ToonCartridgeWeaverProsecutorSpectrumReport:
    """Reporte completo de auditoría fiscal inquisitorial espectral TOON."""
    rademacher_walsh_norm: float
    smith_torsion_free: bool
    betti_number_zero: bool
    snf_diagonal_factors: Tuple[int, ...]
    krull_valuation: float
    novikov_norm: float
    raw_token_count: int
    compressed_token_count: int
    kv_cache_compression_pct: float
    cauchy_riemann_residual: float
    is_gromov_violated: bool
    gromov_capacity_ball: float
    gromov_capacity_cylinder: float
    melnikov_chaos_detected: bool
    melnikov_amplitude: float
    melnikov_offset: float
    delaunay_LGH: Tuple[float, float, float]
    delaunay_valid: bool
    banach_contraction_lip: float
    banach_fixed_point: float
    banach_a_posteriori_error: float
    capacity_jerk_d3C_dt3: float
    spohn_entropy_prod: float
    rsi_accepted_depth: int
    metamorphic_selftest_passed: bool
    dirac_report: DiracAnnihilationProsecutionReport
    truth_value: HeytingTruthValue
    passport_hash: str
    execution_time_ms: float
    schema_version: str = _ENGINE_VERSION


@dataclass(frozen=True)
class _SpectralSeed:
    """Semilla de Fase 1 (puente F1 → F2)."""
    matrix: NDArray[np.float64]
    exponents: NDArray[np.float64]
    rw_norm: float
    torsion_free: bool
    betti_zero: bool
    snf_diagonal_factors: Tuple[int, ...]
    krull: float
    novikov_norm: float
    compressed_tokens: int
    kv_pct: float
    raw_token_count: int
    positron_valid: bool
    delaunay_LGH: Tuple[float, float, float]
    delaunay_valid: bool


@dataclass(frozen=True)
class _DynamicalSeed:
    """Semilla de Fase 2 (puente F2 → F3)."""
    spectral: _SpectralSeed
    cr_residual: float
    capacity_ball: float
    capacity_cylinder: float
    gromov_violated: bool
    melnikov_chaos: bool
    melnikov_amplitude: float
    melnikov_offset: float
    spohn_production: float
    anomaly_norm: float
    force_veto: bool


@dataclass(frozen=True)
class _KleisliState:
    """Estado del bucle Kleisli / RSI³ (Fase 3)."""
    lip_T: float
    lip_certified: float
    lip_lower: float
    fixed_point: float
    a_posteriori_error: float
    jerk: float
    selftest_passed: bool
    consistency_ok: bool
    depth_accepted: int
    archive: Tuple[Tuple[int, int, float, bool], ...]


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 1 — CIMIENTOS ALGEBRAICOS, TOPOLÓGICOS Y CELESTES (base de la jerarquía)
# ══════════════════════════════════════════════════════════════════════════════

class _Fase1_Cimientos:
    """FASE 1: Smith sobre ℤ, Walsh–Hadamard, Novikov/TOON, HMAC, Kepler y Delaunay.

    El último método de la fase, `_fase1_emit_spectral_seed`, emite la semilla que
    consume `_fase2_inicio_desde_semilla` (primer método de la Fase 2).
    """

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        self._secret_key: bytes = _resolve_secret_key(secret_key)
        logger.info("%s v%s inicializado.", self.__class__.__name__, _ENGINE_VERSION)

    # ── 1.1 Cuantización y reducción diagonal de Smith sobre ℤ ───────────────

    def _fase1_quantize(self, matrix: NDArray[np.float64]) -> List[List[int]]:
        """Cuantiza a punto fijo entero: round(scale·A) ∈ ℤ^{m×n}."""
        if not np.all(np.isfinite(matrix)):
            raise ValueError("La matriz de coste contiene valores no finitos.")
        scaled = np.rint(matrix * _SNF_FIXED_POINT_SCALE).astype(np.int64)
        return [[int(v) for v in row] for row in scaled]

    @staticmethod
    def _fase1_integer_pivots(matrix: List[List[int]]) -> List[int]:
        """Diagonaliza una matriz entera por operaciones elementales unimodulares.

        Devuelve |dᵢ| para los pivotes no nulos. El grupo cokernel es
        ℤ^{m−r} ⊕ ⊕ ℤ/dᵢ, así que la forma no normalizada por divisibilidad tiene el mismo
        tipo de isomorfismo que la SNF. El pivote de cada paso decrece estrictamente en valor
        absoluto, lo que garantiza terminación.
        """
        A = [row[:] for row in matrix]
        m = len(A)
        n = len(A[0]) if m else 0
        pivots: List[int] = []
        for t in range(min(m, n)):
            best: Optional[Tuple[int, int]] = None
            for i in range(t, m):
                for j in range(t, n):
                    v = A[i][j]
                    if v != 0 and (best is None or abs(v) < abs(A[best[0]][best[1]])):
                        best = (i, j)
            if best is None:
                break
            i0, j0 = best
            A[t], A[i0] = A[i0], A[t]
            for row in A:
                row[t], row[j0] = row[j0], row[t]
            while True:
                clean = True
                p = A[t][t]
                for i in range(t + 1, m):
                    if A[i][t] != 0:
                        q = A[i][t] // p
                        if q != 0:
                            A[i] = [a - q * b for a, b in zip(A[i], A[t])]
                        if A[i][t] != 0:
                            A[t], A[i] = A[i], A[t]
                            clean = False
                            break
                if not clean:
                    continue
                p = A[t][t]
                for j in range(t + 1, n):
                    if A[t][j] != 0:
                        q = A[t][j] // p
                        if q != 0:
                            for row in A:
                                row[j] -= q * row[t]
                        if A[t][j] != 0:
                            for row in A:
                                row[t], row[j] = row[j], row[t]
                            clean = False
                            break
                if clean:
                    break
            pivots.append(abs(A[t][t]))
        return pivots

    def _fase1_snf_audit(self, matrix: NDArray[np.float64]) -> Tuple[Tuple[int, ...], bool, bool]:
        """Devuelve (factores diagonales, libre de torsión, rango pleno)."""
        pivots = self._fase1_integer_pivots(self._fase1_quantize(matrix))
        m = matrix.shape[0]
        torsion_free = all(p == 1 for p in pivots)
        betti_zero = len(pivots) == m
        return tuple(pivots), bool(torsion_free), bool(betti_zero)

    # ── 1.2 Walsh–Hadamard y derivadas de Rademacher-Walsh ───────────────────

    @staticmethod
    def _fase1_fwht(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """WHT no normalizada: f̂(S) = Σ_x f(x)(−1)^{S·x}, con N = 2^n."""
        a = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
        n = a.size
        if n == 0 or (n & (n - 1)) != 0:
            raise ValueError("La WHT requiere longitud potencia de 2.")
        h = 1
        while h < n:
            a = a.reshape(-1, 2 * h)
            x = a[:, :h].copy()
            y = a[:, h:].copy()
            a[:, :h] = x + y
            a[:, h:] = x - y
            a = a.reshape(-1)
            h <<= 1
        return a

    def _fase1_rademacher_walsh_norm(self, matrix: NDArray[np.float64]) -> float:
        """Promedio sobre k de ‖D_k f‖ / ‖f‖ con f aplanada y acolchada a 2^n."""
        flat = matrix.reshape(-1)
        n_bits = max(0, (flat.size - 1).bit_length())
        if n_bits == 0:
            return 0.0
        size = 1 << n_bits
        f = np.zeros(size, dtype=np.float64)
        f[: flat.size] = flat
        idx = np.arange(size)
        f_norm = float(np.linalg.norm(f)) + _NUMERIC_EPS
        norms = [float(np.linalg.norm(f - f[idx ^ (1 << k)])) / f_norm for k in range(n_bits)]
        return float(np.mean(norms))

    def compute_rademacher_walsh_and_snf(self, apu_cost_matrix: NDArray[np.float64]) -> Tuple[float, bool, bool]:
        """Devuelve (norma RW, libre de torsión sobre ℤ, rango pleno)."""
        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        rw = self._fase1_rademacher_walsh_norm(M)
        _, torsion_free, betti_zero = self._fase1_snf_audit(M)
        return rw, torsion_free, betti_zero

    # ── 1.3 Novikov / Krull / TOON ───────────────────────────────────────────

    def evaluate_novikov_krull_and_toon_prosecution(
        self,
        exponents: NDArray[np.float64],
        raw_token_count: int = _TOON_SOURCE_TOKENS,
    ) -> Tuple[float, float, int, float]:
        """Valuación de Krull ν = min λᵢ, norma e^{−ν} y compresión TOON (presupuesto de 56 tokens)."""
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)
        if exps.size == 0 or not np.all(np.isfinite(exps)):
            raise ValueError("Exponentes de Novikov vacíos o no finitos.")
        if raw_token_count < _TOON_BUDGET_TOKENS:
            raise ValueError("raw_token_count debe ser ≥ presupuesto TOON (56 tokens).")
        v_krull = float(np.min(exps))
        nov_norm = math.exp(min(-v_krull, 700.0))
        kv_pct = (1.0 - _TOON_BUDGET_TOKENS / float(raw_token_count)) * 100.0
        return v_krull, nov_norm, _TOON_BUDGET_TOKENS, float(kv_pct)

    # ── 1.4 Autorización HMAC del positrón ───────────────────────────────────

    def sign_positron(self, message: bytes) -> str:
        """Etiqueta HMAC-SHA256 hexadecimal del mensaje de autorización humana."""
        return hmac.new(self._secret_key, message, hashlib.sha256).hexdigest()

    def verify_positron_tag(self, message: bytes, tag_hex: str) -> bool:
        """Verificación en tiempo constante de la etiqueta del positrón."""
        if not isinstance(tag_hex, str):
            return False
        expected = self.sign_positron(message).encode("ascii")
        received = tag_hex.strip().lower().encode("utf-8")
        return hmac.compare_digest(expected, received)

    def _fase1_resolve_positron(
        self,
        positron_hmac_valid: Optional[bool],
        positron_message: Optional[bytes],
        positron_tag: Optional[str],
    ) -> bool:
        """Prioriza la verificación criptográfica. Sin datos, falla cerrado (False)."""
        if positron_message is not None and positron_tag is not None:
            return self.verify_positron_tag(positron_message, positron_tag)
        if positron_hmac_valid is None:
            return False
        if positron_hmac_valid:
            logger.warning("Positrón aceptado por bandera booleana sin verificación HMAC (modo legado).")
        return bool(positron_hmac_valid)

    # ── 1.5 Mecánica celeste: Kepler y Delaunay ──────────────────────────────

    @staticmethod
    def _fase1_reduce_angle(angle: float) -> float:
        two_pi = 2.0 * math.pi
        r = math.fmod(angle, two_pi)
        return r + two_pi if r < 0.0 else r

    def _fase1_kepler_solve(self, mean_anomaly: float, e: float) -> float:
        """Resuelve E − e sin E = M por Newton–Raphson (0 ≤ e < 1)."""
        if not (0.0 <= e < 1.0):
            raise ValueError("Excentricidad fuera de dominio elíptico.")
        M = self._fase1_reduce_angle(mean_anomaly)
        E = M if e < 0.8 else math.pi
        for _ in range(_KEPLER_MAX_ITER):
            f = E - e * math.sin(E) - M
            dE = f / (1.0 - e * math.cos(E))
            E -= dE
            if abs(dE) < _KEPLER_TOL:
                return E
        raise RuntimeError("Newton–Kepler no convergió.")

    def _fase1_kepler_residual(self, mean_anomaly: float, e: float) -> float:
        E = self._fase1_kepler_solve(mean_anomaly, e)
        M = self._fase1_reduce_angle(mean_anomaly)
        return abs(E - e * math.sin(E) - M)

    @staticmethod
    def _fase1_delaunay_actions(mu: float, a: float, e: float, inc: float) -> Tuple[float, float, float]:
        """Acciones de Delaunay (L, G, H) a partir de (μ, a, e, i)."""
        if mu <= 0.0 or a <= 0.0 or not (0.0 <= e < 1.0):
            raise ValueError("Elementos orbitales fuera de dominio elíptico.")
        L = math.sqrt(mu * a)
        G = L * math.sqrt(1.0 - e * e)
        H = G * math.cos(inc)
        return L, G, H

    @staticmethod
    def _fase1_delaunay_is_physical(L: float, G: float, H: float) -> bool:
        return (0.0 <= G <= L + _NUMERIC_EPS) and (abs(H) <= G + _NUMERIC_EPS)

    # ── 1.6 ÚLTIMO MÉTODO DE FASE 1: emisión de la semilla espectral (puente F1 → F2) ──

    def _fase1_emit_spectral_seed(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        raw_token_count: int,
        positron_hmac_valid: Optional[bool],
        positron_message: Optional[bytes],
        positron_tag: Optional[str],
        orbital_elements: Optional[Tuple[float, float, float, float, float, float]],
    ) -> _SpectralSeed:
        """Ensambla la semilla espectral inmutable que consume `_fase2_inicio_desde_semilla`."""
        matrix = np.asarray(apu_cost_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Se requiere una matriz de coste cuadrada.")
        exps = np.asarray(exponents, dtype=np.float64).reshape(-1)

        rw_norm = self._fase1_rademacher_walsh_norm(matrix)
        snf_factors, torsion_free, betti_zero = self._fase1_snf_audit(matrix)
        v_krull, nov_norm, comp, kv_pct = self.evaluate_novikov_krull_and_toon_prosecution(exps, raw_token_count)

        a, e, inc, mean_anom, _omega, _node = (
            orbital_elements if orbital_elements is not None else _DEFAULT_ORBIT_ELEMENTS
        )
        L, G, H = self._fase1_delaunay_actions(_MU_EARTH_KM3_S2, a, e, inc)
        delaunay_valid = (
            self._fase1_delaunay_is_physical(L, G, H)
            and self._fase1_kepler_residual(mean_anom, e) < _KEPLER_RESIDUAL_TOL
        )
        positron_valid = self._fase1_resolve_positron(positron_hmac_valid, positron_message, positron_tag)

        return _SpectralSeed(
            matrix=matrix,
            exponents=exps,
            rw_norm=rw_norm,
            torsion_free=torsion_free,
            betti_zero=betti_zero,
            snf_diagonal_factors=snf_factors,
            krull=v_krull,
            novikov_norm=nov_norm,
            compressed_tokens=comp,
            kv_pct=kv_pct,
            raw_token_count=raw_token_count,
            positron_valid=positron_valid,
            delaunay_LGH=(L, G, H),
            delaunay_valid=bool(delaunay_valid),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 2 — GEOMETRÍA SIMPLÉCTICA, DINÁMICA DE POINCARÉ Y ANIQUILACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class _Fase2_Geometria(_Fase1_Cimientos):
    """FASE 2: CR discreto, no-squeezing, Melnikov, Spohn y disparo Dirac/crowbar.

    Primer método: `_fase2_inicio_desde_semilla` (consume la semilla de Fase 1).
    Último método: `_fase2_emit_dynamical_seed` (emite la semilla para Fase 3).
    """

    # ── 2.0 INICIO DE FASE 2 (puente F1 → F2) ────────────────────────────────

    def _fase2_inicio_desde_semilla(
        self,
        seed: _SpectralSeed,
        ball_r: float,
        cylinder_R: float,
    ) -> Tuple[float, float, float, bool]:
        """Consume la semilla espectral y audita CR y no-squeezing.

        Devuelve (residuo ∂̄, capacidad de la bola, capacidad del cilindro, violación de Gromov).
        """
        res_cr, violated = self.solve_cauchy_riemann_floer_prosecution(seed.matrix, ball_r, cylinder_R)
        cap_ball, cap_cyl, _ = self._fase2_nonsqueezing(ball_r, cylinder_R)
        return res_cr, cap_ball, cap_cyl, violated

    # ── 2.1 Cauchy–Riemann discreto y no-squeezing de Gromov ─────────────────

    @staticmethod
    def _fase2_cauchy_riemann_residual(matrix: NDArray[np.float64]) -> float:
        """Residuo normalizado de ∂̄f = 0 para f = u + iv, con u = M y v = Mᵀ.

        Defecto de CR: (uₓ − v_y) + i(v_x + u_y). Se normaliza por la suma de normas de las
        derivadas parciales. Es cero si y solo si se cumplen las ecuaciones de CR en la malla.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            raise ValueError("Se requiere una malla cuadrada de al menos 2×2 para ∂̄.")
        u = matrix
        v = matrix.T
        u_x = np.gradient(u, axis=0)
        u_y = np.gradient(u, axis=1)
        v_x = np.gradient(v, axis=0)
        v_y = np.gradient(v, axis=1)
        defect = (u_x - v_y) + 1j * (v_x + u_y)
        num = float(np.linalg.norm(defect))
        den = float(
            np.linalg.norm(u_x) + np.linalg.norm(u_y) + np.linalg.norm(v_x) + np.linalg.norm(v_y)
        ) + _NUMERIC_EPS
        return num / den

    @staticmethod
    def _fase2_nonsqueezing(ball_r: float, cylinder_R: float) -> Tuple[float, float, bool]:
        """Capacidades de Gromov: c(B) = πr², c(Z) = πR². Violación ⇔ c(B) > c(Z)."""
        if ball_r <= 0.0 or cylinder_R <= 0.0:
            raise ValueError("Los radios de Gromov deben ser positivos.")
        c_ball = math.pi * ball_r * ball_r
        c_cyl = math.pi * cylinder_R * cylinder_R
        return c_ball, c_cyl, c_ball > c_cyl

    def solve_cauchy_riemann_floer_prosecution(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = 0.25,
        cylinder_R: float = 0.20,
    ) -> Tuple[float, bool]:
        """Devuelve (residuo ∂̄ discreto, violación de no-squeezing)."""
        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        res = self._fase2_cauchy_riemann_residual(M)
        _, _, violated = self._fase2_nonsqueezing(ball_r, cylinder_R)
        return res, violated

    # ── 2.2 Melnikov del péndulo perturbado (Poincaré–KAM) ───────────────────

    @staticmethod
    def _fase2_melnikov_coefficients(
        melnikov_delta: float,
        melnikov_gamma: float,
        melnikov_omega: float,
    ) -> Tuple[float, float]:
        """Devuelve (amplitud A, desplazamiento c) de M(t₀) = c + A cos(ωt₀ + φ).

        Separatriz del péndulo: ẋ_h(t) = 2 sech t. Para la perturbación ε(−δẋ + γ cos ωt):
            c = −δ ∫ẋ_h² dt ≈ −8δ,   A = γ·|∫ẋ_h e^{iωt} dt| ≈ 2πγ sech(πω/2).
        """
        if melnikov_omega <= 0.0:
            raise ValueError("La frecuencia de forzamiento debe ser positiva.")
        t = np.linspace(-_MELNIKOV_HALF_WIDTH, _MELNIKOV_HALF_WIDTH, _MELNIKOV_POINTS)
        xdot = 2.0 / np.cosh(t)
        offset = -melnikov_delta * _trapz(xdot * xdot, t)
        i_cos = _trapz(xdot * np.cos(melnikov_omega * t), t)
        i_sin = _trapz(xdot * np.sin(melnikov_omega * t), t)
        amplitude = abs(melnikov_gamma) * math.hypot(i_cos, i_sin)
        return float(amplitude), float(offset)

    def evaluate_poincare_kam_melnikov(
        self,
        melnikov_delta: float = 0.01,
        melnikov_gamma: float = 0.02,
        melnikov_omega: float = 1.0,
    ) -> bool:
        """True si M tiene cero simple (intersección homoclínica transversal, caos posible)."""
        amp, off = self._fase2_melnikov_coefficients(melnikov_delta, melnikov_gamma, melnikov_omega)
        return amp > abs(off)

    # ── 2.3 Entropía relativa y producción de Spohn ──────────────────────────

    @staticmethod
    def _fase2_relative_entropy_maximally_mixed(p: NDArray[np.float64]) -> float:
        """S(ρ‖I/d) = log d − H(p) = log d + Σ pᵢ log pᵢ para ρ diagonal (p normalizada)."""
        d = p.size
        return float(math.log(d) + float(np.sum(p * np.log(p))))

    def _fase2_spohn_production(self, weights: NDArray[np.float64], depolarizing_rate: float = 0.1) -> float:
        """σ = S(ρ‖I/d) − S(Φ_λ(ρ)‖I/d) = H(p_after) − H(p) ≥ 0 (DPI, punto fijo I/d)."""
        if not (0.0 <= depolarizing_rate <= 1.0):
            raise ValueError("La tasa del canal despolarizante debe estar en [0, 1].")
        p = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), _SPECTRAL_PSD_FLOOR)
        p = p / float(np.sum(p))
        d = p.size
        p_after = (1.0 - depolarizing_rate) * p + depolarizing_rate / d
        s_before = self._fase2_relative_entropy_maximally_mixed(p)
        s_after = self._fase2_relative_entropy_maximally_mixed(p_after)
        return float(s_before - s_after)

    # ── 2.4 Aniquilación Dirac y disparo crowbar ─────────────────────────────

    def execute_dirac_and_crowbar_prosecution(
        self,
        anomaly_norm: float,
        positron_hmac_valid: bool,
        force_veto: bool = False,
        spohn_production: float = 0.0,
    ) -> DiracAnnihilationProsecutionReport:
        """Aniquilación e⁻ + e⁺ → 2γ (2m_e c²) si el positrón es válido y no hay veto; crowbar en otro caso."""
        if anomaly_norm < 0.0:
            raise ValueError("La norma de anomalía no puede ser negativa.")
        if positron_hmac_valid and not force_veto:
            energy_kev = _POSITRON_REST_MASS_KEV
            photons = 2
            photon_kev = _ELECTRON_REST_MASS_KEV
            penalty_db = 0.0
            crowbar = False
            latency_ns = 0.0
            proof_str = (
                f"DIRAC_ANNIHILATION_SHA256_VALID_E={energy_kev:.4f}_keV|anomaly={anomaly_norm:.6e}"
            )
        else:
            energy_kev = 0.0
            photons = 0
            photon_kev = 0.0
            penalty_db = _BACK_ACTION_PENALTY_DB
            crowbar = True
            latency_ns = _ESP32_CROWBAR_MAX_LATENCY_NS
            proof_str = (
                f"DIRAC_ANNIHILATION_VETOED_ESP32_CROWBAR_TRIGGERED_GPIO14|anomaly={anomaly_norm:.6e}"
            )
        sha256_proof = hashlib.sha256(proof_str.encode("utf-8")).hexdigest()
        return DiracAnnihilationProsecutionReport(
            anomaly_norm=float(anomaly_norm),
            positron_hmac_valid=bool(positron_hmac_valid),
            energy_released_kev=energy_kev,
            photons_emitted=photons,
            sha256_proof=sha256_proof,
            back_action_penalty_db=penalty_db,
            spohn_entropy_prod=float(spohn_production),
            crowbar_triggered=crowbar,
            crowbar_latency_ns=latency_ns,
            photon_energy_kev=photon_kev,
        )

    # ── 2.5 ÚLTIMO MÉTODO DE FASE 2: emisión de la semilla dinámica (puente F2 → F3) ──

    def _fase2_emit_dynamical_seed(
        self,
        seed: _SpectralSeed,
        ball_r: float,
        cylinder_R: float,
        melnikov_delta: float,
        melnikov_gamma: float,
        melnikov_omega: float,
    ) -> _DynamicalSeed:
        """Agrega CR, Gromov, Melnikov, Spohn y el veto estructural en una semilla inmutable."""
        res_cr, cap_ball, cap_cyl, gromov_violated = self._fase2_inicio_desde_semilla(seed, ball_r, cylinder_R)
        mel_amp, mel_off = self._fase2_melnikov_coefficients(melnikov_delta, melnikov_gamma, melnikov_omega)
        melnikov_chaos = mel_amp > abs(mel_off)
        spohn = self._fase2_spohn_production(seed.exponents)
        anomaly = float(np.linalg.norm(seed.matrix) * 0.1)
        force_veto = (
            gromov_violated
            or (not seed.torsion_free)
            or melnikov_chaos
            or (seed.krull < _DEFAULT_KRULL_FLOOR)
            or (not seed.delaunay_valid)
        )
        return _DynamicalSeed(
            spectral=seed,
            cr_residual=res_cr,
            capacity_ball=cap_ball,
            capacity_cylinder=cap_cyl,
            gromov_violated=bool(gromov_violated),
            melnikov_chaos=bool(melnikov_chaos),
            melnikov_amplitude=mel_amp,
            melnikov_offset=mel_off,
            spohn_production=spohn,
            anomaly_norm=anomaly,
            force_veto=bool(force_veto),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 3 — MÓNADA KLEISLI, BANACH–PICARD, RSI³ Y ADJUDICACIÓN HEYTING
# ══════════════════════════════════════════════════════════════════════════════

class _Fase3_Kleisli_RSI3(_Fase2_Geometria):
    """FASE 3: operador Kleisli T³, Picard con cota a posteriori, RSI³ y veredicto final.

    Primer método: `_fase3_inicio_desde_semilla_dinamica` (consume la semilla de Fase 2).
    Último método: `execute_toon_prosecutor_audit` (API pública).
    """

    # ── 3.0 INICIO DE FASE 3 (puente F2 → F3) ────────────────────────────────

    def _fase3_inicio_desde_semilla_dinamica(
        self,
        dyn: _DynamicalSeed,
        rsi_max_depth: int,
        initial_lip: float,
    ) -> _KleisliState:
        """Consume la semilla dinámica. Si hay veto, no gasta profundidad de auto-mejora (fail-fast)."""
        state = self._fase3_kleisli_full(initial_lip)
        selftest_ok = self._fase3_metamorphic_selftest()
        state = replace(state, selftest_passed=selftest_ok)
        depth = 0 if dyn.force_veto else max(0, min(int(rsi_max_depth), _RSI_MAX_DEPTH))
        return self._fase3_rsi_recurse(1, state, depth)

    # ── 3.1 Operador Kleisli, Picard y medición de Lipschitz ─────────────────

    @staticmethod
    def _fase3_kleisli_cubed(lip_T: float) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Devuelve T³ con T(x) = λ cos x + 0.1, Lip(T) = λ."""
        def T(x):
            return lip_T * np.cos(x) + _KLEISLI_OFFSET

        def T3(x):
            return T(T(T(x)))

        return T3

    @staticmethod
    def _fase3_picard(
        op: Callable[[float], float],
        x0: float,
        tol: float = _PICARD_TOL,
        max_iter: int = _PICARD_MAX_ITER,
    ) -> Tuple[float, List[float]]:
        """Iteración de Picard–Banach xₙ₊₁ = op(xₙ) con criterio de parada |Δx| < tol."""
        x = float(x0)
        iterates = [x]
        for _ in range(max_iter):
            x_next = float(op(x))
            iterates.append(x_next)
            if abs(x_next - x) < tol:
                return x_next, iterates
            x = x_next
        raise RuntimeError("Picard–Banach no convergió dentro de max_iter.")

    @staticmethod
    def _fase3_empirical_lipschitz(op: Callable[[NDArray[np.float64]], NDArray[np.float64]], n_points: int) -> float:
        """Cota inferior empírica de Lip(op) sobre una malla uniforme en [−2, 2].

        Con n = 100·2^k + 1 las mallas están anidadas (step_k divide a step_{k−1}), así que la
        cota inferior es monotónica en k.
        """
        xs = np.linspace(-_KLEISLI_DOMAIN, _KLEISLI_DOMAIN, n_points)
        ys = op(xs)
        dx = np.abs(xs[:, None] - xs[None, :])
        dy = np.abs(ys[:, None] - ys[None, :])
        mask = dx > 0.0
        return float(np.max(dy[mask] / dx[mask]))

    def _fase3_kleisli_full(self, initial_lip: float) -> _KleisliState:
        """Calcula la cota certificada λ³ + ε, el punto fijo, el error a posteriori y el jerk transitorio."""
        if not (0.0 <= initial_lip < 1.0):
            raise ValueError("La constante de Lipschitz base debe estar en [0, 1).")
        t3 = self._fase3_kleisli_cubed(initial_lip)
        lip_cert = initial_lip ** 3 + _KLEISLI_ERROR_BUDGET
        x_star, iterates = self._fase3_picard(t3, 0.0)
        a_post = (
            (lip_cert / (1.0 - lip_cert)) * abs(iterates[-1] - iterates[-2])
            if len(iterates) >= 2
            else float("inf")
        )
        capacities = [math.pi * x * x for x in iterates[:_JERK_WINDOW]]
        if len(capacities) >= 4:
            jerk = capacities[-1] - 3.0 * capacities[-2] + 3.0 * capacities[-3] - capacities[-4]
        else:
            jerk = 0.0
        lower0 = self._fase3_empirical_lipschitz(t3, _KLEISLI_GRID_BASE + 1)
        return _KleisliState(
            lip_T=initial_lip,
            lip_certified=lip_cert,
            lip_lower=lower0,
            fixed_point=x_star,
            a_posteriori_error=float(a_post),
            jerk=float(jerk),
            selftest_passed=False,
            consistency_ok=bool(lower0 <= lip_cert + _NUMERIC_EPS),
            depth_accepted=0,
            archive=(),
        )

    def evaluate_rsi3_banach_kleisli(self, initial_lip: float = _KLEISLI_LIP_T) -> Tuple[float, float]:
        """Devuelve (cota certificada de Lip(T³), jerk discreto transitorio de la capacidad de Gromov)."""
        st = self._fase3_kleisli_full(initial_lip)
        return st.lip_certified, st.jerk

    # ── 3.2 Verificación del verificador (autoprueba metamórfica) ────────────

    def _fase3_metamorphic_selftest(self) -> bool:
        """Invariantes algebraicos y analíticos. Cualquier fallo ⇒ el veredicto queda VETOED."""
        checks: List[bool] = []
        try:
            # (1) Smith: diag(2,3) ⇒ ℤ/6 (con torsión); unimodular ⇒ libre de torsión
            checks.append(self._fase1_integer_pivots([[2, 0], [0, 3]]) == [2, 3])
            checks.append(self._fase1_integer_pivots([[1, 1], [0, 1]]) == [1, 1])
            # (2) Invariante: ∏ dᵢ = |det A| bajo operaciones unimodulares
            A = [[2, 4], [6, 8]]
            piv = self._fase1_integer_pivots(A)
            det_int = round(float(np.linalg.det(np.array(A, dtype=np.float64))))
            checks.append(int(np.prod(piv)) == abs(det_int))
            # (3) Parseval de WHT y derivadas de Boole en el dominio espectral
            rng = np.random.default_rng(20250101)
            f = rng.standard_normal(16)
            F = self._fase1_fwht(f)
            checks.append(abs(float(np.sum(F ** 2)) - 16.0 * float(np.sum(f ** 2))) < 1e-9)
            idx = np.arange(16)
            for k in range(4):
                direct = float(np.sum((f - f[idx ^ (1 << k)]) ** 2))
                spectral = (4.0 / 16.0) * float(np.sum((F ** 2)[((idx >> k) & 1) == 1]))
                checks.append(abs(direct - spectral) < 1e-9)
            # (4) Kepler: residuo de la ecuación de Kepler
            E = self._fase1_kepler_solve(1.0, 0.3)
            checks.append(abs(E - 0.3 * math.sin(E) - 1.0) < 1e-10)
            # (5) Melnikov: amplitud numérica frente a 2πγ sech(πω/2) y offset frente a −8δ
            amp, off = self._fase2_melnikov_coefficients(0.01, 0.02, 1.0)
            amp_an = 2.0 * math.pi * 0.02 / math.cosh(math.pi / 2.0)
            checks.append(abs(amp - amp_an) <= 1e-6 * amp_an and abs(off + 0.08) < 1e-6)
            # (6) Spohn ≥ 0 (desigualdad de procesamiento de datos)
            checks.append(self._fase2_spohn_production(np.array([0.15, 0.22, 0.35, 0.48])) >= -1e-12)
            # (7) Delaunay: 0 ≤ G ≤ L y |H| ≤ G
            L, G, H = self._fase1_delaunay_actions(_MU_EARTH_KM3_S2, 7000.0, 0.01, 0.9)
            checks.append(self._fase1_delaunay_is_physical(L, G, H))
            # (8) Cota publicada: 0.95³ + 0.002375 = 0.85975
            checks.append(abs((_KLEISLI_LIP_T ** 3 + _KLEISLI_ERROR_BUDGET) - 0.85975) < 1e-12)
        except Exception:  # fail-closed: una excepción en la autoprueba invalida el verificador
            logger.exception("Autoverificación metamórfica: excepción durante la ejecución.")
            return False
        ok = all(checks)
        if not ok:
            logger.error("Autoverificación metamórfica FALLIDA: %s", checks)
        return bool(ok)

    # ── 3.3 RSI³: bucle recursivo acotado con puerta de aceptación ───────────

    def _fase3_rsi_recurse(self, level: int, state: _KleisliState, max_depth: int) -> _KleisliState:
        """Nivel k: refina la cota inferior empírica en la malla 100·2^k+1 y acepta solo si
        (a) es monótona, (b) es consistente con la certificada y (c) la autoprueba pasó.
        La recursión termina en `max_depth` ≤ 3. Nunca altera la cota certificada.
        """
        if level > max_depth:
            return state
        n_points = _KLEISLI_GRID_BASE * (1 << level) + 1
        t3 = self._fase3_kleisli_cubed(state.lip_T)
        lower = self._fase3_empirical_lipschitz(t3, n_points)
        monotone = lower >= state.lip_lower - _NUMERIC_EPS
        consistent = lower <= state.lip_certified + _NUMERIC_EPS
        accepted = bool(state.selftest_passed and monotone and consistent)
        new_state = replace(
            state,
            lip_lower=max(state.lip_lower, lower) if accepted else state.lip_lower,
            consistency_ok=bool(state.consistency_ok and consistent),
            depth_accepted=state.depth_accepted + (1 if accepted else 0),
            archive=state.archive + ((level, n_points, float(lower), accepted),),
        )
        return self._fase3_rsi_recurse(level + 1, new_state, max_depth)

    # ── 3.4 Adjudicación Heyting y pasaporte HMAC ────────────────────────────

    def _fase3_heyting_adjudicate(
        self,
        seed: _SpectralSeed,
        dyn: _DynamicalSeed,
        kleisli: _KleisliState,
        dirac: DiracAnnihilationProsecutionReport,
    ) -> HeytingTruthValue:
        """Veredicto = meet(espectral, dinámico, verificador). Crowbar ⇒ ⊥."""
        H = HeytingTruthValue
        if dirac.crowbar_triggered:
            return H.VETOED
        spectral = (
            H.VETOED
            if (not seed.torsion_free or seed.krull < _DEFAULT_KRULL_FLOOR)
            else (H.DEGRADED if seed.rw_norm > _RW_DEGRADED_TOL else H.COHERENT)
        )
        dynamical = (
            H.VETOED
            if (dyn.gromov_violated or dyn.melnikov_chaos or not seed.delaunay_valid)
            else (H.DEGRADED if dyn.cr_residual > _CR_RESIDUAL_TOL else H.COHERENT)
        )
        verifier = H.COHERENT if (kleisli.selftest_passed and kleisli.consistency_ok) else H.VETOED
        return spectral.meet(dynamical).meet(verifier)

    def _fase3_sign_passport(
        self,
        truth: HeytingTruthValue,
        dirac: DiracAnnihilationProsecutionReport,
        seed: _SpectralSeed,
        kleisli: _KleisliState,
    ) -> str:
        """Pasaporte HMAC determinista (sin tiempos de ejecución)."""
        payload = "|".join(
            [
                _ENGINE_VERSION,
                str(int(truth.value)),
                dirac.sha256_proof,
                f"{seed.krull:.6f}",
                f"{kleisli.lip_certified:.6f}",
                ",".join(str(p) for p in seed.snf_diagonal_factors),
                str(kleisli.depth_accepted),
            ]
        )
        return hmac.new(self._secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    # ── 3.5 ÚLTIMO MÉTODO DE FASE 3 — API pública de auditoría ───────────────

    def execute_toon_prosecutor_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        raw_token_count: int = _TOON_SOURCE_TOKENS,
        positron_hmac_valid: Optional[bool] = None,
        ball_r: float = 0.25,
        cylinder_R: float = 0.20,
        *,
        positron_message: Optional[bytes] = None,
        positron_tag: Optional[str] = None,
        orbital_elements: Optional[Tuple[float, float, float, float, float, float]] = None,
        melnikov_delta: float = 0.01,
        melnikov_gamma: float = 0.02,
        melnikov_omega: float = 1.0,
        rsi_max_depth: int = _RSI_MAX_DEPTH,
        kleisli_initial_lip: float = _KLEISLI_LIP_T,
    ) -> ToonCartridgeWeaverProsecutorSpectrumReport:
        """Pipeline completo F1 → F2 → F3. Para autorizar el positrón use `positron_message` y `positron_tag`."""
        t_start = time.perf_counter()

        seed = self._fase1_emit_spectral_seed(
            apu_cost_matrix, exponents, raw_token_count,
            positron_hmac_valid, positron_message, positron_tag, orbital_elements,
        )
        dyn = self._fase2_emit_dynamical_seed(
            seed, ball_r, cylinder_R, melnikov_delta, melnikov_gamma, melnikov_omega,
        )
        kleisli = self._fase3_inicio_desde_semilla_dinamica(dyn, rsi_max_depth, kleisli_initial_lip)

        verifier_ok = kleisli.selftest_passed and kleisli.consistency_ok
        dirac = self.execute_dirac_and_crowbar_prosecution(
            anomaly_norm=dyn.anomaly_norm,
            positron_hmac_valid=seed.positron_valid,
            force_veto=(dyn.force_veto or not verifier_ok),
            spohn_production=dyn.spohn_production,
        )
        truth = self._fase3_heyting_adjudicate(seed, dyn, kleisli, dirac)
        passport = self._fase3_sign_passport(truth, dirac, seed, kleisli)
        exec_ms = (time.perf_counter() - t_start) * 1000.0

        return ToonCartridgeWeaverProsecutorSpectrumReport(
            rademacher_walsh_norm=seed.rw_norm,
            smith_torsion_free=seed.torsion_free,
            betti_number_zero=seed.betti_zero,
            snf_diagonal_factors=seed.snf_diagonal_factors,
            krull_valuation=seed.krull,
            novikov_norm=seed.novikov_norm,
            raw_token_count=seed.raw_token_count,
            compressed_token_count=seed.compressed_tokens,
            kv_cache_compression_pct=seed.kv_pct,
            cauchy_riemann_residual=dyn.cr_residual,
            is_gromov_violated=dyn.gromov_violated,
            gromov_capacity_ball=dyn.capacity_ball,
            gromov_capacity_cylinder=dyn.capacity_cylinder,
            melnikov_chaos_detected=dyn.melnikov_chaos,
            melnikov_amplitude=dyn.melnikov_amplitude,
            melnikov_offset=dyn.melnikov_offset,
            delaunay_LGH=seed.delaunay_LGH,
            delaunay_valid=seed.delaunay_valid,
            banach_contraction_lip=kleisli.lip_certified,
            banach_fixed_point=kleisli.fixed_point,
            banach_a_posteriori_error=kleisli.a_posteriori_error,
            capacity_jerk_d3C_dt3=kleisli.jerk,
            spohn_entropy_prod=dyn.spohn_production,
            rsi_accepted_depth=kleisli.depth_accepted,
            metamorphic_selftest_passed=kleisli.selftest_passed,
            dirac_report=dirac,
            truth_value=truth,
            passport_hash=passport,
            execution_time_ms=exec_ms,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

class BoolNovikovToonCartridgeWeaverProsecutorEngine(_Fase3_Kleisli_RSI3):
    """Motor espectral fiscal inquisidor dual de tejeduría sináptica TOON (API pública v13.1.0)."""


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOTEST DE DEMOSTRACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _print_report(title: str, rep: ToonCartridgeWeaverProsecutorSpectrumReport) -> None:
    print(f"\n--- {title} ---")
    print(f"Veredicto Heyting Ω₃        : {rep.truth_value.name} ({hex(rep.truth_value.value)})")
    print(f"Crowbar Triggered           : {rep.dirac_report.crowbar_triggered}")
    print(f"Smith (factores diagonales) : {rep.snf_diagonal_factors} | libre_torsión={rep.smith_torsion_free}")
    print(f"Compresión KV-Cache         : {rep.kv_cache_compression_pct:.1f}% ({rep.compressed_token_count} tokens)")
    print(f"Valuación Krull v(A)        : {rep.krull_valuation:.4f}")
    print(f"Residuo ∂̄ (CR discreto)     : {rep.cauchy_riemann_residual:.3e}")
    print(f"Gromov c(B) vs c(Z)         : {rep.gromov_capacity_ball:.4f} vs {rep.gromov_capacity_cylinder:.4f}"
          f" | violado={rep.is_gromov_violated}")
    print(f"Melnikov A vs |offset|      : {rep.melnikov_amplitude:.5f} vs {abs(rep.melnikov_offset):.5f}"
          f" | caos={rep.melnikov_chaos_detected}")
    print(f"Delaunay (L,G,H) válido     : {tuple(round(x, 3) for x in rep.delaunay_LGH)} | {rep.delaunay_valid}")
    print(f"Contracción Banach Lip      : {rep.banach_contraction_lip:.6f}"
          f" | punto fijo={rep.banach_fixed_point:.6f} | err_post={rep.banach_a_posteriori_error:.2e}")
    print(f"Jerk discreto d³C/dt³       : {rep.capacity_jerk_d3C_dt3:.4e}")
    print(f"Entropía de Spohn σ ≥ 0     : {rep.spohn_entropy_prod:.6e}")
    print(f"RSI³ profundidad aceptada   : {rep.rsi_accepted_depth} | selftest={rep.metamorphic_selftest_passed}")
    print(f"Aniquilación Dirac (keV)    : {rep.dirac_report.energy_released_kev} keV")
    print(f"Pasaporte HMAC SHA-256      : {rep.passport_hash[:16]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = BoolNovikovToonCartridgeWeaverProsecutorEngine(secret_key=b"demo-key-0123456789abcdef")

    exps_test = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)
    auth_message = b"APU|HUMAN_AUTH|ticket=0001"
    auth_tag = engine.sign_positron(auth_message)

    print("=== TEST AUTÓNOMO: BoolNovikovToonCartridgeWeaverProsecutorEngine v13.1.0 ===")
    print(f"HMAC tag válido             : {engine.verify_positron_tag(auth_message, auth_tag)}")
    tampered = auth_tag[:-1] + ("0" if auth_tag[-1] != "0" else "1")
    print(f"HMAC tag alterado           : {engine.verify_positron_tag(auth_message, tampered)}")

    # Escenario A: r > R ⇒ violación de no-squeezing ⇒ VETOED con crowbar.
    rep_a = engine.execute_toon_prosecutor_audit(
        np.eye(4, dtype=np.float64) * 1.5, exps_test,
        positron_message=auth_message, positron_tag=auth_tag,
        ball_r=0.25, cylinder_R=0.20,
    )
    _print_report("Escenario A: r > R (Gromov violado)", rep_a)

    # Escenario B: r ≤ R, cuantización unimodular ⇒ libre de torsión.
    rep_b = engine.execute_toon_prosecutor_audit(
        np.array([[0.01, 0.01], [0.0, 0.01]], dtype=np.float64), exps_test,
        positron_message=auth_message, positron_tag=auth_tag,
        ball_r=0.15, cylinder_R=0.20,
    )
    _print_report("Escenario B: r ≤ R, unimodular", rep_b)

    print("\n=== TEST COMPLETADO ===")
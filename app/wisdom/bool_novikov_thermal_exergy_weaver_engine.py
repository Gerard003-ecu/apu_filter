# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovThermalExergyWeaverEngine  v14.0.0-PhD (reedición rigurosa)              ║
║ Ubicación: app/wisdom/bool_novikov_thermal_exergy_weaver_engine.py                             ║
║ Función  : Tejedor espectral y custodio de gradientes termodinámicos sobre el fibrado dual      ║
║            Boole–Novikov, con mecánica celeste de Poincaré y automejora recursiva RSI³.        ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

DOCTRINA DEL MÓDULO
───────────────────
Ningún nombre de teorema se usa como adorno: cada afirmación matemática del módulo se acompaña
de (i) su hipótesis explícita, (ii) un residual numérico calculado y (iii) una cota de tolerancia
justificada (error de redondeo, error de truncamiento o error estadístico). Lo que es *política*
de custodia se declara como política y jamás se disfraza de identidad matemática.

ARQUITECTURA EN TRES FASES ANIDADAS
───────────────────────────────────
El anidamiento es doble y literal:

  (a) Anidamiento de capacidades (cadena de herencia):
          PhaseIAlgebraicSubstrate
              └── PhaseIIThermoSymplectic
                      └── PhaseIIICelestialRSI
                              └── BoolNovikovThermalExergyWeaverEngine   (fachada pública)

  (b) Anidamiento de datos y de flujo (handoffs):
          PhaseIHandoff  ⊂  PhaseIIHandoff  ⊂  ThermalExergyWeaverSpectrumReport
      y el ÚLTIMO método de cada fase es el PRIMERO que invoca la fase siguiente:
          Fase I  §I.6  build_phase_i_handoff(...)      ──► Fase II §II.1 accept_phase_i_handoff
          Fase II §II.7 build_phase_ii_handoff(...)     ──► Fase III §III.1 accept_phase_ii_handoff
          Fase III §III.8 execute_thermal_exergy_weaver_audit  (método terminal / orquestador)

FASE I  — SUSTRATO ALGEBRAICO-DISCRETO (Boole / ℤ / grafos / gradientes discretos)
  §I.1 Regularización espectral del núcleo K (proyección PSD, número de condición).
  §I.2 Smith Normal Form exacta sobre ℤ ⇒ torsión, β₀, β₁ (homología celular real).
  §I.3 Análisis de Rademacher–Walsh en el grupo booleano (ℤ/2)^n (Plancherel, secuencia).
  §I.4 Tellegen–Kirchhoff: im(Bᵀ) ⊥ ker(B), balance de potencia, Joule ↔ Gouy–Stodola.
  §I.5 Gradiente discreto Itoh–Abe baricéntrico w*=1/2 ≡ punto medio de González; ley exacta
       de disipación ΔE = −h‖∇̄E‖² (integrador de gradiente discreto / transformada de Cayley).
  §I.6 [TERMINAL] build_phase_i_handoff  ── continuación directa de la Fase II.

FASE II — TERMODINÁMICA ESPECTRAL Y GEOMETRÍA SIMPLÉCTICA (KMS / Novikov / Floer / Gromov)
  §II.1 accept_phase_i_handoff (entrada: certificado terminal de Fase I).
  §II.2 Estado de Gibbs, condición KMS de Tomita–Takesaki, entropía relativa de Umegaki.
  §II.3 Jarzynski–Crooks: log-sum-exp, ESS de Kish, BAR (MLE), identidad de disipación.
  §II.4 Anillo ultramétrico de Novikov: valuación, norma, axiomas verificados, filtración.
  §II.5 Ecuación de Cauchy–Riemann–Floer con cota de truncamiento y control no holomorfo.
  §II.6 Capacidades de Gromov–Hofer–Zehnder–Ekeland–Hofer: no-squeezing (condición necesaria).
  §II.7 [TERMINAL] build_phase_ii_handoff ── continuación directa de la Fase III.

FASE III — MECÁNICA CELESTE DE POINCARÉ, RSI³ Y CUSTODIA FINAL
  §III.1 accept_phase_ii_handoff (entrada: certificado terminal de Fase II).
  §III.2 Ecuación homológica de Poincaré–Lindstedt (divisores pequeños, Fourier en 𝕋^d).
  §III.3 Certificado KAM diophantino en coordenadas de Delaunay (twist, isoenergético).
  §III.4 Integral de Melnikov homoclínica: ceros simples ⇒ herradura de Smale–Birkhoff.
  §III.5 Flujo simpléctico de Yoshida⁴: invariante integral de Poincaré, deriva de energía.
  §III.6 RSI³: Banach + Kleisli (Writer sobre (Ω₃,∧,⊤)), Lip(T³) ≤ q³, cotas a priori/a posteriori.
  §III.7 Aniquilación de Dirac (cinemática relativista CODATA) y pasaporte HMAC determinista.
  §III.8 [TERMINAL] execute_thermal_exergy_weaver_audit: adjudicación en Heyting Ω₃.

MIGRACIÓN DESDE v13 (rupturas deliberadas)
──────────────────────────────────────────
• Ω₃ se reordena: VETOED=⊥=0 < DEGRADED=½=1 < COHERENT=⊤=2 (antes COHERENT era 0 = ⊥).
• El pasaporte ya NO incluye el tiempo de ejecución (era irreproducible); el tiempo es telemetría.
• `positron_hmac_valid: bool` desaparece: se exige un token HMAC verificado en tiempo constante.
• Los informes se renombran y anidan; `legacy_summary()` provee una proyección compatible.

LIMITACIONES DECLARADAS (honestidad epistémica)
───────────────────────────────────────────────
1. no-squeezing y las capacidades EH se evalúan como OBSTRICCIONES NECESARIAS: su cumplimiento
   no prueba la existencia de un encaje simpléctico.
2. Melnikov es un método de O(ε): la existencia de ceros simples prueba caos para ε pequeño
   suficiente, no para ε arbitrario.
3. La cota KAM de pequeñez ε‖W‖_∞ ≪ 1 es HEURÍSTICA cuantitativa (las constantes rigurosas
   dependen del teorema KAM concreto); la degenerancia kepleriana se reporta, no se oculta.
4. RSI³ es una meta-iteración numérica certificada (sintonía → refinamiento → contracción),
   no una afirmación sobre auto-modificación de agentes.
5. La auditoría Floer usa una familia exacta con S = cI; el caso anisótropo requiere trayectorias
   numéricas y queda fuera del certificado.

REFERENCIAS: Itoh–Abe (1988); González (1996); McLachlan–Quispel–Robidoux (1999); Tellegen (1952);
Jarzynski (1997); Crooks (1999); Bennett (1976); Shirts et al. (2003); Kish (1965);
Bratteli–Robinson / Tomita–Takesaki; Umegaki (1962); Gromov (1985); Hofer–Zehnder (1990);
Ekeland–Hofer (1990); Floer (1988); Poincaré (1892); Melnikov (1963); Guckenheimer–Holmes (1983);
Arnold (1963); Moser (1962); Delaunay; Yoshida (1990); Banach (1922); Higham (2008);
Savitzky–Golay (1964); CODATA 2022.
"""
from __future__ import annotations

import dataclasses
import hashlib
import heapq
import hmac
import itertools
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

__all__ = [
    "HeytingTruthValue",
    "Evidence",
    "Certified",
    "WeaverError",
    "WeaverValidationError",
    "PhaseIAlgebraicSubstrate",
    "PhaseIIThermoSymplectic",
    "PhaseIIICelestialRSI",
    "BoolNovikovThermalExergyWeaverEngine",
    "ThermalExergyWeaverSpectrumReport",
    "__version__",
]

logger = logging.getLogger("APU.Wisdom.BoolNovikovThermalExergyWeaver")

__version__: Final[str] = (
    "14.0.0-ENGINE-BOOL-NOVIKOV-THERMAL-EXERGY-WEAVER-POINCARE-KAM-RSI3-PhD"
)
version: Final[str] = __version__

# ─────────────────────────── Constantes físicas y numéricas ───────────────────────────
_EPS: Final[float] = float(np.finfo(np.float64).eps)          # 2.220446049250313e-16
_WILKINSON: Final[float] = 1.0e-15                            # holgura relativa de redondeo
_PSD_FLOOR: Final[float] = 1.0e-12                            # piso espectral del núcleo
_KRULL_FILTRATION_FLOOR: Final[float] = 0.10                  # POLÍTICA de filtración de energía
_NOVIKOV_CUTOFF: Final[float] = 64.0                          # corte Λ_max de la serie truncada
_ME_C2_KEV: Final[float] = 510.99895069                       # CODATA 2022: m_e c² [keV]
_TWO_ME_C2_KEV: Final[float] = 2.0 * _ME_C2_KEV               # 1021.99790138 keV (no "1022.0")
_KB_J_PER_K: Final[float] = 1.380649e-23                      # exacto (SI 2019)
_T0_KELVIN: Final[float] = 300.0                              # temperatura ambiente de referencia
_MODULAR_LOG_CAP: Final[float] = 600.0                        # guarda de rango dinámico modular


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE I · §I.0  SUSTRATO LÓGICO, METROLÓGICO Y DE VALIDACIÓN
#  (Topos Ω₃, evidencia certificada, mónada Writer, utilidades exactas)
# ══════════════════════════════════════════════════════════════════════════════════════
class WeaverError(Exception):
    """Error base del tejido."""


class WeaverValidationError(WeaverError, ValueError):
    """Violación de hipótesis matemáticas de entrada (fail-fast, sin degradación silenciosa)."""


class HeytingTruthValue(IntEnum):
    r"""
    Álgebra de Heyting de 3 elementos (cadena de Gödel) Ω₃ = {⊥ < ½ < ⊤}.

    Realización topos-teórica: Ω₃ ≅ 𝒪(Sierpiński) = {∅, {1}, {0,1}}, retículo de abiertos del
    espacio de Sierpiński; es un *frame* y por tanto un álgebra de Heyting completa.

    Operaciones (cadena de Gödel):
        a ∧ b = min(a,b),   a ∨ b = max(a,b),
        a → b = ⊤ si a ≤ b, y b en caso contrario,      ¬a = (a → ⊥).
    No es booleana: ¬½ = ⊥ y ¬¬½ = ⊤ ≠ ½ (falla el tercero excluido). El núcleo de doble
    negación ¬¬ es el morfismo de núcleos que colapsa Ω₃ onto su parte booleana {⊥, ⊤}.

    NOTA CRÍTICA v13: allí COHERENT=0x00 era el fondo ⊥, invirtiendo el orden intuicionista.
    """

    VETOED = 0     # ⊥  (falso / custodia violada)
    DEGRADED = 1   # ½  (frontera / evidencia blanda insuficiente)
    COHERENT = 2   # ⊤  (verdadero / certificado)

    # ---- estructura de retículo de Heyting -------------------------------------------------
    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return HeytingTruthValue(min(int(self), int(other)))

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return HeytingTruthValue(max(int(self), int(other)))

    def implies(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return HeytingTruthValue.COHERENT if int(self) <= int(other) else other

    def negate(self) -> "HeytingTruthValue":
        return self.implies(HeytingTruthValue.VETOED)

    def double_negation(self) -> "HeytingTruthValue":
        return self.negate().negate()

    @property
    def is_decidable(self) -> bool:
        """¬¬a = a ⟺ a es booleano (⊥ o ⊤)."""
        return self.double_negation() == self

    @classmethod
    def infimum(cls, values: Sequence["HeytingTruthValue"]) -> "HeytingTruthValue":
        """Ínfimo (conjunción) — monótono y absorbente en ⊥; base de la adjudicación."""
        acc = cls.COHERENT
        for v in values:
            acc = acc.meet(v)
        return acc


@dataclass(frozen=True)
class Evidence:
    r"""
    Átomo de custodia: (residual, tolerancia, dureza).

    `hard=True`  ⇒ fallo produce ⊥ (VETOED): se violó una hipótesis estructural o una política
                   de seguridad no negociable.
    `hard=False` ⇒ fallo produce ½ (DEGRADED): la evidencia es estadística, de discretización
                   o de calidad de estimador, no una contradicción matemática.
    """

    label: str
    residual: float
    tolerance: float
    hard: bool = False
    note: str = ""

    @property
    def passed(self) -> bool:
        return bool(math.isfinite(self.residual) and self.residual <= self.tolerance)

    @property
    def truth(self) -> HeytingTruthValue:
        if self.passed:
            return HeytingTruthValue.COHERENT
        return HeytingTruthValue.VETOED if self.hard else HeytingTruthValue.DEGRADED

    @classmethod
    def boolean(cls, label: str, ok: bool, *, hard: bool = False, note: str = "") -> "Evidence":
        """Codifica un predicado como residual 0/1 con tolerancia 0 (sin magia numérica)."""
        return cls(label=label, residual=0.0 if ok else 1.0, tolerance=0.0, hard=hard, note=note)


@dataclass(frozen=True)
class Certified:
    r"""
    Mónada Writer T = (−) × M sobre el monoide conmutativo M = (Ω₃, ∧, ⊤) × (Ledger, ++, ε).

    unit(a) = (a, ε, ⊤);  bind((a, ℓ, t), f) = (b, ℓ ++ ℓ', t ∧ t') con f(a) = (b, ℓ', t').
    Las leyes monádicas se siguen de los axiomas de monoide; §III.6 las verifica numéricamente.
    La métrica de la categoría de Kleisli se toma sobre el portador `value` (el ledger es
    bookkeeping y no interviene en la constante de Lipschitz).
    """

    value: Any
    ledger: Tuple[str, ...] = ()
    truth: HeytingTruthValue = HeytingTruthValue.COHERENT

    @staticmethod
    def unit(value: Any) -> "Certified":
        return Certified(value)

    def bind(self, f: Callable[[Any], "Certified"]) -> "Certified":
        out = f(self.value)
        return Certified(out.value, self.ledger + out.ledger, self.truth.meet(out.truth))


def kleisli_compose(
    f: Callable[[Any], Certified], g: Callable[[Any], Certified]
) -> Callable[[Any], Certified]:
    """Composición de Kleisli (f >=> g)(x) = bind(f(x), g)."""
    return lambda x: f(x).bind(g)


# ─────────────────────────── utilidades numéricas exactas ───────────────────────────
def _as_matrix(value: Any, name: str, *, square: bool = False, complex_ok: bool = False) -> NDArray:
    dtype = np.complex128 if complex_ok else np.float64
    try:
        arr = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise WeaverValidationError(f"{name}: no convertible a matriz {dtype}.") from exc
    if arr.ndim != 2 or 0 in arr.shape:
        raise WeaverValidationError(f"{name}: se requiere matriz 2-D no vacía, recibido {arr.shape}.")
    if square and arr.shape[0] != arr.shape[1]:
        raise WeaverValidationError(f"{name}: se requiere matriz cuadrada, recibido {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise WeaverValidationError(f"{name}: contiene NaN/Inf (hipótesis de finitud violada).")
    return arr


def _as_vector(value: Any, name: str, *, allow_empty: bool = False) -> NDArray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0 and not allow_empty:
        raise WeaverValidationError(f"{name}: vector vacío.")
    if not np.all(np.isfinite(arr)):
        raise WeaverValidationError(f"{name}: contiene NaN/Inf.")
    return arr


def _logsumexp(v: NDArray) -> float:
    m = float(np.max(v))
    if not math.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(v - m))))


def _expit(x: NDArray) -> NDArray:
    """Logística estable (sin overflow ni warnings de rama)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _xlogx(p: NDArray) -> NDArray:
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    mask = p > 0.0
    out[mask] = p[mask] * np.log(p[mask])
    return out


def _savgol_derivative_coeffs(window: int, poly: int, deriv: int, step: float = 1.0) -> NDArray:
    r"""
    Coeficientes de Savitzky–Golay para la derivada d-ésima en el centro de la ventana.
    Ajuste local p(t)=Σ a_j t^j por mínimos cuadrados (Vandermonde + pseudoinversa); la
    derivada en 0 es d!·a_d / step^d. Requiere window impar, poly ≥ deriv, window > poly.
    """
    if window % 2 == 0 or window <= poly or deriv > poly:
        raise WeaverValidationError("Savitzky–Golay: ventana/orden inválidos.")
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    vand = np.vander(t, poly + 1, increasing=True)
    coeffs = np.linalg.pinv(vand)[deriv] * math.factorial(deriv) / (step ** deriv)
    return coeffs


def _array_digest(arr: NDArray) -> str:
    """Digest canónico (forma, dtype little-endian, bytes) — reproducible entre plataformas."""
    a = np.ascontiguousarray(arr)
    dt = np.dtype(a.dtype).newbyteorder("<")
    a = a.astype(dt, copy=False)
    h = hashlib.sha256()
    h.update(b"ARR|")
    h.update(str(a.shape).encode("ascii"))
    h.update(b"|")
    h.update(dt.str.encode("ascii"))
    h.update(b"|")
    h.update(a.tobytes())
    return h.hexdigest()


def _freeze_primitives(obj: Any) -> Any:
    """Proyección determinista a primitivas JSON (arrays → digest, flotantes → 15 dígitos)."""
    if isinstance(obj, np.ndarray):
        return {"ndarray_sha256": _array_digest(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _freeze_primitives(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, HeytingTruthValue):
        return {"omega3": int(obj), "name": obj.name}
    if isinstance(obj, IntEnum):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [_freeze_primitives(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _freeze_primitives(v) for k, v in obj.items()}
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        return float(f"{x:.15g}")
    if isinstance(obj, bytes):
        return obj.hex()
    return str(obj)


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _smith_invariants(integer_matrix: NDArray) -> Tuple[int, ...]:
    r"""
    Forma normal de Smith sobre ℤ con aritmética entera exacta (int de Python, sin overflow).

    Devuelve los factores invariantes d₁ | d₂ | … | d_r (d_i > 0) de la matriz. Consecuencia
    homológica: coker(A) ≅ ℤ^{n−r} ⊕ ⊕_{i≤r} ℤ/d_i. Por tanto:
        libre de torsión ⟺ todos los d_i = 1.
    (v13 usaba `matrix_rank` en ℝ y lo llamaba "rango en F₂": ni F₂ ni torsión.)
    """
    m_rows = [[int(round(float(v))) for v in row] for row in np.asarray(integer_matrix)]
    n = len(m_rows)
    m = len(m_rows[0]) if n else 0
    invariants: List[int] = []
    k = 0
    while k < min(n, m):
        pivot: Optional[Tuple[int, int]] = None
        for i in range(k, n):
            for j in range(k, m):
                if m_rows[i][j] != 0 and (
                    pivot is None or abs(m_rows[i][j]) < abs(m_rows[pivot[0]][pivot[1]])
                ):
                    pivot = (i, j)
        if pivot is None:
            break
        i0, j0 = pivot
        m_rows[k], m_rows[i0] = m_rows[i0], m_rows[k]
        for row in m_rows:
            row[k], row[j0] = row[j0], row[k]
        while True:
            column_clean = True
            for i in range(k + 1, n):
                if m_rows[i][k] == 0:
                    continue
                q = m_rows[i][k] // m_rows[k][k]
                for j in range(k, m):
                    m_rows[i][j] -= q * m_rows[k][j]
                if m_rows[i][k] != 0:
                    m_rows[k], m_rows[i] = m_rows[i], m_rows[k]
                    column_clean = False
            if not column_clean:
                continue
            row_clean = True
            for j in range(k + 1, m):
                if m_rows[k][j] == 0:
                    continue
                q = m_rows[k][j] // m_rows[k][k]
                for i in range(k, n):
                    m_rows[i][j] -= q * m_rows[i][k]
                if m_rows[k][j] != 0:
                    for row in m_rows:
                        row[k], row[j] = row[j], row[k]
                    row_clean = False
            if not row_clean:
                continue
            piv = m_rows[k][k]
            offender: Optional[int] = None
            for i in range(k + 1, n):
                if any(m_rows[i][j] % piv != 0 for j in range(k + 1, m)):
                    offender = i
                    break
            if offender is None:
                break
            for j in range(k, m):
                m_rows[k][j] += m_rows[offender][j]
        invariants.append(abs(m_rows[k][k]))
        k += 1
    return tuple(invariants)


def _sequency_walsh_matrix(n_bits: int) -> NDArray:
    r"""
    Matriz de Walsh en orden de secuencia (sequency) construida como producto de funciones
    de Rademacher: con g = gray(s) = s ⊕ (s≫1),
        w_s(i) = Π_{j: g_j=1} r_{j+1}(i),    r_{j+1}(i) = (−1)^{bit_{n−1−j}(i)}.
    Propiedades verificadas en §I.3: W Wᵀ = N·I y #cambios de signo de w_s = s.
    Es la transformada de Fourier sobre el grupo booleano (ℤ/2)^n reordenada por secuencia.
    """
    n_samples = 1 << n_bits
    idx = np.arange(n_samples)
    bits = (idx[:, None] >> np.arange(n_bits)[None, :]) & 1
    walsh = np.ones((n_samples, n_samples), dtype=np.float64)
    for s in range(n_samples):
        g = s ^ (s >> 1)
        row = np.ones(n_samples, dtype=np.float64)
        for j in range(n_bits):
            if (g >> j) & 1:
                row *= 1.0 - 2.0 * bits[:, n_bits - 1 - j]
        walsh[s] = row
    return walsh


def _eh_capacities(axes_squared: Sequence[float], count: int) -> Tuple[float, ...]:
    r"""
    Capacidades de Ekeland–Hofer del elipsoide E(a₁,…,a_n) ⊂ ℝ^{2n}, a_i = r_i²:
        Σ(E) = { m·π·a_i : m ∈ ℕ_{≥1}, i = 1..n }  (multiconjunto ordenado),  c_k = k-ésimo.
    Para la bola B(r): a_i = r² ∀i ⇒ c_k(B(r)) = π r² ⌈k/n⌉ (verificado en el auto-test).
    """
    heap: List[Tuple[float, int, int]] = []
    for i, a in enumerate(axes_squared):
        if a <= 0.0:
            raise WeaverValidationError("Ekeland–Hofer: ejes al cuadrado deben ser positivos.")
        heapq.heappush(heap, (math.pi * float(a), i, 1))
    caps: List[float] = []
    for _ in range(count):
        val, i, mult = heapq.heappop(heap)
        caps.append(val)
        heapq.heappush(heap, (math.pi * float(axes_squared[i]) * (mult + 1), i, mult + 1))
    return tuple(caps)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE I · INFORMES (contratos de datos que la Fase II consume)
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class KernelSpectrumReport:
    """§I.1 — Espectro del núcleo regularizado (proyección PSD y condicionamiento)."""

    dimension: int
    symmetry_residual: float
    min_eigenvalue_raw: float
    min_eigenvalue_reg: float
    max_eigenvalue_reg: float
    condition_number: float
    spectral_shift_applied: float


@dataclass(frozen=True)
class TellegenKirchhoffReport:
    """§I.2+§I.4 — Homología entera del grafo y teorema de Tellegen (KVL ⊥ KCL)."""

    n_nodes: int
    n_edges: int
    smith_invariants: Tuple[int, ...]
    smith_rank: int
    torsion_free: bool
    is_oriented_incidence: bool
    betti_0_components: int
    betti_1_cycles: int
    kcl_residual: float
    kvl_cycle_residual: float
    tellegen_power_residual: float
    cross_tellegen_residual: float
    joule_power_w: float
    entropy_production_w_per_k: float
    gouy_stodola_destroyed_w: float


@dataclass(frozen=True)
class DiscreteGradientReport:
    """§I.5 — Gradiente discreto baricéntrico y ley exacta de disipación."""

    barycentric_weight_w: float
    secant_residual_max: float
    barycentric_symmetry_residual: float
    gonzalez_midpoint_equivalence_residual: float
    energy_law_residual_max: float
    energy_monotone: bool
    euler_control_monotone: bool
    integration_step_h: float
    integration_steps: int
    cayley_spectral_radius: float
    optimal_step_h: float
    optimal_contraction_q: float
    tellegen_energy_delta: float


@dataclass(frozen=True)
class WalshRademacherReport:
    """§I.3 — Análisis armónico sobre (ℤ/2)^n con verificación de Plancherel y secuencia."""

    n_bits: int
    n_samples: int
    orthogonality_residual: float
    sequency_order_defect: int
    parseval_residual: float
    low_sequency_energy_ratio: float
    normalized_l2: float


@dataclass(frozen=True)
class PhaseIHandoff:
    """
    §I.6 [TERMINAL FASE I] — Certificado de salida y CONTINUACIÓN de la Fase II.
    Los arrays viajan como portadores internos (repr=False) y se serializan como digest.
    """

    kernel: KernelSpectrumReport
    tellegen: TellegenKirchhoffReport
    discrete_gradient: DiscreteGradientReport
    walsh: WalshRademacherReport
    evidences: Tuple[Evidence, ...]
    truth_value: HeytingTruthValue
    kernel_digest: str
    kernel_reg: NDArray = field(repr=False)
    eigenvalues: NDArray = field(repr=False)
    eigenvectors: NDArray = field(repr=False)
    incidence: NDArray = field(repr=False)


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE I · MOTOR: SUSTRATO ALGEBRAICO-DISCRETO
# ══════════════════════════════════════════════════════════════════════════════════════
class PhaseIAlgebraicSubstrate:
    r"""
    FASE I — Boole / ℤ / grafos / gradientes discretos.

    Responsabilidad: convertir una matriz de coste arbitraria en un objeto matemáticamente
    admisible (núcleo simétrico PSD), extraer sus invariantes enteros y armónicos booleanos,
    certificar la física de circuitos (Tellegen) y la ley exacta de disipación del gradiente
    discreto. Su método terminal `build_phase_i_handoff` es el punto de entrada de la Fase II.
    """

    PHASE_LABEL: Final[str] = "I·SUSTRATO-ALGEBRAICO-DISCRETO"

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        *,
        ambient_temperature_k: float = _T0_KELVIN,
        seed: int = 20_261_111,
    ) -> None:
        if secret_key is None:
            env = os.environ.get("APU_WEAVER_HMAC_KEY")
            if env:
                secret_key = env.encode("utf-8")
            else:
                secret_key = os.urandom(32)
                logger.warning(
                    "Clave HMAC efímera generada: los pasaportes NO serán verificables entre "
                    "procesos. Defina APU_WEAVER_HMAC_KEY o pase secret_key explícitamente."
                )
        if len(secret_key) < 16:
            raise WeaverValidationError("La clave HMAC debe tener ≥ 128 bits de material.")
        if not (math.isfinite(ambient_temperature_k) and ambient_temperature_k > 0.0):
            raise WeaverValidationError("T₀ debe ser finita y positiva (tercer principio).")
        self._secret_key = bytes(secret_key)          # jamás se registra ni se filtra
        self._t0 = float(ambient_temperature_k)
        self._seed = int(seed)
        logger.info("%s v%s listo (T₀=%.3f K, semilla=%d).",
                    type(self).__name__, __version__, self._t0, self._seed)

    # ───────────────────────── §I.1 Regularización espectral ─────────────────────────
    def regularize_kernel(
        self, apu_cost_matrix: NDArray, *, psd_floor: float = _PSD_FLOOR
    ) -> Tuple[KernelSpectrumReport, NDArray, NDArray, NDArray]:
        r"""
        Proyección al cono PSD: S = ½(K+Kᵀ) = U diag(λ) Uᵀ;  K_reg = U diag(max(λ, ε_psd)) Uᵀ.

        Es la proyección de Frobenius sobre {S ⪰ ε_psd I} (Higham 1988): mínima perturbación
        que garantiza positividad definida estricta, requisito de la teoría de gradientes
        discretos disipativos y de la contracción de Cayley.
        """
        k_raw = _as_matrix(apu_cost_matrix, "apu_cost_matrix", square=True)
        n = k_raw.shape[0]
        fro = float(la.norm(k_raw, ord="fro")) + _EPS
        sym_res = float(la.norm(k_raw - k_raw.T, ord="fro")) / fro
        sym = 0.5 * (k_raw + k_raw.T)
        eigvals, eigvecs = la.eigh(sym)
        eigvals_reg = np.maximum(eigvals, float(psd_floor))
        k_reg = (eigvecs * eigvals_reg) @ eigvecs.T
        k_reg = 0.5 * (k_reg + k_reg.T)
        report = KernelSpectrumReport(
            dimension=n,
            symmetry_residual=sym_res,
            min_eigenvalue_raw=float(eigvals[0]),
            min_eigenvalue_reg=float(eigvals_reg[0]),
            max_eigenvalue_reg=float(eigvals_reg[-1]),
            condition_number=float(eigvals_reg[-1] / eigvals_reg[0]),
            spectral_shift_applied=float(np.max(eigvals_reg - eigvals)),
        )
        return report, k_reg, eigvals_reg, eigvecs

    # ───────────────────────── §I.2 Smith / homología entera ─────────────────────────
    def smith_normal_form_invariants(self, integer_matrix: NDArray) -> Tuple[int, ...]:
        r"""Factores invariantes d₁|…|d_r sobre ℤ (aritmética exacta). Ver `_smith_invariants`."""
        mat = _as_matrix(integer_matrix, "integer_matrix")
        if not np.allclose(mat, np.rint(mat), rtol=0.0, atol=1e-9):
            raise WeaverValidationError("SNF: la matriz debe ser (casi) entera; no se redondea en silencio.")
        return _smith_invariants(np.rint(mat))

    @staticmethod
    def _oriented_incidence(k_mat: NDArray, edge_tol: float) -> Tuple[NDArray, NDArray]:
        r"""
        Grafo de acoplamientos: aristas (i<j) con |K_ij| > edge_tol; incidencia orientada
        B ∈ {−1,0,+1}^{n×m} (una +1 y una −1 por columna ⇒ unimodularidad total, teorema
        clásico de matrices de incidencia: todos los menores ∈ {0,±1} ⇒ d_i = 1 ⇒ sin torsión).
        Conductancias g_e = |K_ij| > 0 (relación constitutiva óhmica lineal).
        """
        n = k_mat.shape[0]
        rows, cols = np.triu_indices(n, k=1)
        mask = np.abs(k_mat[rows, cols]) > edge_tol
        rows, cols = rows[mask], cols[mask]
        m = int(rows.size)
        incidence = np.zeros((n, m), dtype=np.float64)
        if m:
            incidence[rows, np.arange(m)] = 1.0
            incidence[cols, np.arange(m)] = -1.0
        return incidence, np.abs(k_mat[rows, cols])

    # ───────────────────────── §I.3 Rademacher–Walsh ─────────────────────────
    def walsh_rademacher_analysis(self, potential: NDArray) -> WalshRademacherReport:
        r"""
        Análisis de Walsh–Rademacher del vector de potencial p = diag(K) embebido diádicamente.

        Se verifica: (i) W Wᵀ = N I; (ii) la secuencia de w_s es exactamente s (orden de
        secuencia correcto, no el orden natural de Sylvester); (iii) Plancherel
        ‖Wx/√N‖₂ = ‖x‖₂. La energía en baja secuencia mide la "suavidad booleana" del paisaje
        de coste: es una métrica armónica, NO un invariante topológico (crítica a v13).
        """
        x = _as_vector(potential, "potential")
        n_bits = max(1, int(math.ceil(math.log2(max(2, x.size)))))
        n_samples = 1 << n_bits
        padded = np.zeros(n_samples, dtype=np.float64)
        padded[: x.size] = x
        walsh = _sequency_walsh_matrix(n_bits)
        orth_res = float(la.norm(walsh @ walsh.T / n_samples - np.eye(n_samples), ord="fro")) / math.sqrt(n_samples)
        sequency_defect = max(
            abs(int(np.count_nonzero(np.diff(walsh[s]) != 0.0)) - s) for s in range(n_samples)
        )
        coeffs = (walsh @ padded) / math.sqrt(n_samples)
        norm_x = float(la.norm(padded))
        parseval = abs(float(la.norm(coeffs)) - norm_x) / (norm_x + _EPS)
        energy = coeffs ** 2
        total = float(np.sum(energy)) + _EPS
        low_band = float(np.sum(energy[: max(1, n_samples // 4)])) / total
        return WalshRademacherReport(
            n_bits=n_bits,
            n_samples=n_samples,
            orthogonality_residual=orth_res,
            sequency_order_defect=int(sequency_defect),
            parseval_residual=parseval,
            low_sequency_energy_ratio=low_band,
            normalized_l2=norm_x / math.sqrt(n_samples),
        )

    # ───────────────────────── §I.4 Tellegen–Kirchhoff ─────────────────────────
    def tellegen_kirchhoff_certificate(
        self, k_reg: NDArray, *, edge_tol: float = 1e-12
    ) -> Tuple[TellegenKirchhoffReport, NDArray]:
        r"""
        Teorema de Tellegen (1952) — puramente topológico, independiente de la constitución:

            v = Bᵀe (KVL)  y  Bj = i_inj (KCL)  ⇒  vᵀj = eᵀB j = eᵀ i_inj.

        Equivalentemente im(Bᵀ) ⊥ ker(B): el espacio de tensiones (cut-set) es el ortogonal
        del espacio de corrientes de lazo. Se certifica además el *cross-Tellegen* (cuasi-
        potencia): v de una red ⋅ j circulante de OTRA red con la misma topología = 0.

        Termodinámica: Φ = Σ g_e v_e² ≥ 0 (Joule), σ = Φ/T₀ ≥ 0 (producción de entropía,
        Clausius–Duhem) y X_destroyed = T₀σ = Φ (Gouy–Stodola). Esto sustituye las fórmulas
        decorativas de v13 por una cadena de identidades verificables.
        """
        incidence, conductances = self._oriented_incidence(k_reg, edge_tol)
        n, m = incidence.shape
        invariants = self.smith_normal_form_invariants(incidence) if m else ()
        rank = len(invariants)
        torsion_free = all(d == 1 for d in invariants)
        oriented = bool(m == 0 or (
            np.all(np.sum(incidence == 1.0, axis=0) == 1)
            and np.all(np.sum(incidence == -1.0, axis=0) == 1)
            and np.all(np.sum(np.abs(incidence), axis=0) == 2)
        ))
        if m == 0:
            report = TellegenKirchhoffReport(
                n_nodes=n, n_edges=0, smith_invariants=(), smith_rank=0, torsion_free=True,
                is_oriented_incidence=True, betti_0_components=n, betti_1_cycles=0,
                kcl_residual=0.0, kvl_cycle_residual=0.0, tellegen_power_residual=0.0,
                cross_tellegen_residual=0.0, joule_power_w=0.0,
                entropy_production_w_per_k=0.0, gouy_stodola_destroyed_w=0.0,
            )
            return report, incidence

        laplacian = incidence @ (conductances[:, None] * incidence.T)
        pseudo = np.linalg.pinv(laplacian, rcond=1e-13)
        rng = np.random.default_rng(self._seed)
        injection = laplacian @ (pseudo @ rng.standard_normal(n))   # KCL soluble por componente
        potentials = pseudo @ injection
        voltages = incidence.T @ potentials
        currents = conductances * voltages
        scale_i = float(la.norm(injection)) + _EPS
        scale_v = float(la.norm(voltages)) + _EPS
        kcl_res = float(la.norm(incidence @ currents - injection)) / scale_i

        _, sing, vt = la.svd(incidence, full_matrices=True)
        num_rank = int(np.sum(sing > 1e-12 * max(1.0, float(sing[0]))))
        cycle_basis = vt[num_rank:].T if num_rank < m else np.zeros((m, 0))
        if cycle_basis.shape[1]:
            kvl_res = float(la.norm(cycle_basis.T @ voltages)) / scale_v
            circulating = cycle_basis @ np.ones(cycle_basis.shape[1])
            cross_res = abs(float(voltages @ circulating)) / (scale_v * float(la.norm(circulating)) + _EPS)
        else:
            kvl_res, cross_res = 0.0, 0.0

        delivered = float(potentials @ injection)
        joule = float(np.sum(conductances * voltages ** 2))
        tellegen_res = abs(float(voltages @ currents) - delivered) / (abs(delivered) + joule + _EPS)
        sigma = joule / self._t0
        report = TellegenKirchhoffReport(
            n_nodes=n, n_edges=m, smith_invariants=invariants, smith_rank=rank,
            torsion_free=bool(torsion_free), is_oriented_incidence=oriented,
            betti_0_components=int(n - num_rank), betti_1_cycles=int(m - num_rank),
            kcl_residual=kcl_res, kvl_cycle_residual=kvl_res,
            tellegen_power_residual=tellegen_res, cross_tellegen_residual=cross_res,
            joule_power_w=joule, entropy_production_w_per_k=sigma,
            gouy_stodola_destroyed_w=self._t0 * sigma,
        )
        return report, incidence

    # ───────────────────────── §I.5 Gradiente discreto Itoh–Abe ─────────────────────────
    @staticmethod
    def _energy(k_mat: NDArray, x: NDArray) -> float:
        return 0.5 * float(x @ (k_mat @ x))

    @staticmethod
    def _itoh_abe_quadratic(k_mat: NDArray, x: NDArray, y: NDArray) -> NDArray:
        r"""
        Gradiente discreto de Itoh–Abe (1988) coordenado, en forma ANALÍTICA libre de
        cancelación para E(x)=½xᵀKx con K simétrica:

            ∇̄_IA(x,y)_i = (K z^{(i−1)})_i + ½ K_ii Δ_i,   Δ = y−x,
            z^{(i−1)} = (y₁,…,y_{i−1}, x_i,…,x_n).

        Se sigue de E(z^{(i)}) − E(z^{(i−1)}) = Δ_i (K z^{(i−1)})_i + ½K_ii Δ_i², de donde la
        identidad secante Σ_i ∇̄_i Δ_i = E(y) − E(x) es EXACTA (telescópica), y el límite
        Δ_i → 0 reproduce la derivada parcial (consistencia).
        """
        delta = y - x
        z = x.copy()
        grad = np.empty_like(x)
        for i in range(x.size):
            grad[i] = float(k_mat[i] @ z) + 0.5 * float(k_mat[i, i]) * delta[i]
            z[i] = y[i]
        return grad

    def _barycentric_discrete_gradient(
        self, k_mat: NDArray, x: NDArray, y: NDArray, w: float
    ) -> NDArray:
        r"""
        Familia baricéntrica ∇̄_w = w ∇̄_IA(x,y) + (1−w) ∇̄_IA(y,x).

        LEMA (rescate riguroso del "w*=1/2" de v13): ambos sumandos satisfacen la identidad
        secante, y la combinación es afín con pesos que suman 1 ⇒ ∇̄_w es gradiente discreto
        para TODO w ∈ [0,1]. En w = ½ es simétrico bajo x↔y (esquema auto-adjunto) y, para
        formas cuadráticas, coincide EXACTAMENTE con el punto medio de González:
            ∇̄_IA(x,y)_i = (Kx)_i + Σ_{j<i}K_ij Δ_j + ½K_ii Δ_i
            ∇̄_IA(y,x)_i = (Kx)_i + Σ_{j>i}K_ij Δ_j + ½K_ii Δ_i
        ⇒ ½(suma) = (Kx)_i + ½(KΔ)_i = (K(x+y)/2)_i.   ∎
        (v13 escribía w·d + (1−w)·d ≡ d: peso vacuo y además con factor 2 erróneo.)
        """
        if not 0.0 <= w <= 1.0:
            raise WeaverValidationError("El peso baricéntrico w debe estar en [0,1].")
        forward = self._itoh_abe_quadratic(k_mat, x, y)
        backward = self._itoh_abe_quadratic(k_mat, y, x)
        return w * forward + (1.0 - w) * backward

    def discrete_gradient_certificate(
        self,
        k_reg: NDArray,
        eigenvalues: NDArray,
        eigenvectors: NDArray,
        *,
        w: float = 0.5,
        steps: int = 24,
    ) -> DiscreteGradientReport:
        r"""
        Certificados del gradiente discreto y del integrador disipativo:

        (1) identidad secante exacta para w ∈ {0,¼,½,¾,1};
        (2) simetría en w=½ y equivalencia con González (lema anterior);
        (3) esquema implícito x_{k+1} = x_k − h ∇̄_½(x_k,x_{k+1}) ⇒ sistema lineal
            (I + hK/2) x_{k+1} = (I − hK/2) x_k  (transformada de Cayley),
            con ley EXACTA ΔE = −h‖∇̄‖² ≤ 0 para todo h>0 (disipación incondicional);
        (4) control: Euler explícito con h > 2/λ_max viola la monotonía (contraste honesto);
        (5) radio espectral de Cayley q(h) = max_i |(1−hλ_i/2)/(1+hλ_i/2)| < 1 y parámetro
            óptimo h* = 2/√(λ_min λ_max), q* = (√κ−1)/(√κ+1)  ← insumo del RSI³ nivel 1.
        """
        n = k_reg.shape[0]
        rng = np.random.default_rng(self._seed + 17)
        secant_max, sym_res, mid_res = 0.0, 0.0, 0.0
        for _ in range(24):
            x = rng.standard_normal(n)
            y = rng.standard_normal(n)
            e_x, e_y = self._energy(k_reg, x), self._energy(k_reg, y)
            scale = abs(e_x) + abs(e_y) + 1.0
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                grad = self._barycentric_discrete_gradient(k_reg, x, y, weight)
                secant_max = max(secant_max, abs(float(grad @ (y - x)) - (e_y - e_x)) / scale)
            g_xy = self._barycentric_discrete_gradient(k_reg, x, y, 0.5)
            g_yx = self._barycentric_discrete_gradient(k_reg, y, x, 0.5)
            mid = k_reg @ (0.5 * (x + y))
            nrm = float(la.norm(g_xy)) + _EPS
            sym_res = max(sym_res, float(la.norm(g_xy - g_yx)) / nrm)
            mid_res = max(mid_res, float(la.norm(g_xy - mid)) / nrm)

        lam_min, lam_max = float(eigenvalues[0]), float(eigenvalues[-1])
        h_demo = 2.0 / lam_max
        a_mat = np.eye(n) + 0.5 * h_demo * k_reg
        b_mat = np.eye(n) - 0.5 * h_demo * k_reg
        x_k = np.diag(k_reg).copy()
        energies = [self._energy(k_reg, x_k)]
        law_res = 0.0
        monotone = True
        for _ in range(int(steps)):
            x_next = la.solve(a_mat, b_mat @ x_k, assume_a="pos")
            grad = self._barycentric_discrete_gradient(k_reg, x_k, x_next, 0.5)
            e_next = self._energy(k_reg, x_next)
            law_res = max(
                law_res,
                abs((e_next - energies[-1]) + h_demo * float(grad @ grad))
                / (abs(e_next - energies[-1]) + _EPS),
            )
            monotone = monotone and (e_next <= energies[-1] * (1.0 + 1e-12) + 1e-15)
            energies.append(e_next)
            x_k = x_next

        h_euler = 2.5 / lam_max                      # fuera del disco de estabilidad de Euler
        x_e = np.diag(k_reg).copy()
        e_euler = [self._energy(k_reg, x_e)]
        for _ in range(int(steps)):
            x_e = x_e - h_euler * (k_reg @ x_e)
            e_euler.append(self._energy(k_reg, x_e))
        euler_monotone = all(
            e_euler[i + 1] <= e_euler[i] * (1.0 + 1e-12) + 1e-15 for i in range(len(e_euler) - 1)
        )

        half = 0.5 * h_demo * eigenvalues
        cayley_q = float(np.max(np.abs((1.0 - half) / (1.0 + half))))
        h_opt = 2.0 / math.sqrt(lam_min * lam_max)
        sqrt_kappa = math.sqrt(lam_max / lam_min)
        q_opt = (sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)
        return DiscreteGradientReport(
            barycentric_weight_w=float(w),
            secant_residual_max=secant_max,
            barycentric_symmetry_residual=sym_res,
            gonzalez_midpoint_equivalence_residual=mid_res,
            energy_law_residual_max=law_res,
            energy_monotone=bool(monotone),
            euler_control_monotone=bool(euler_monotone),
            integration_step_h=h_demo,
            integration_steps=int(steps),
            cayley_spectral_radius=cayley_q,
            optimal_step_h=h_opt,
            optimal_contraction_q=q_opt,
            tellegen_energy_delta=energies[-1] - energies[0],
        )

    # ─────────── §I.6 [TERMINAL FASE I] → CONTINUACIÓN / ENTRADA DE LA FASE II ───────────
    def build_phase_i_handoff(
        self,
        apu_cost_matrix: NDArray,
        *,
        psd_floor: float = _PSD_FLOOR,
        edge_tol: float = 1e-12,
        dg_steps: int = 24,
    ) -> PhaseIHandoff:
        r"""
        MÉTODO TERMINAL DE LA FASE I. Ensambla §I.1→§I.5, adjudica en Ω₃ por ínfimo y entrega
        `PhaseIHandoff`, que es EXACTAMENTE el argumento de `PhaseIIThermoSymplectic.
        accept_phase_i_handoff` (§II.1). No hay otra puerta de entrada a la Fase II.
        """
        kernel_rep, k_reg, eigvals, eigvecs = self.regularize_kernel(apu_cost_matrix, psd_floor=psd_floor)
        tellegen_rep, incidence = self.tellegen_kirchhoff_certificate(k_reg, edge_tol=edge_tol)
        dg_rep = self.discrete_gradient_certificate(k_reg, eigvals, eigvecs, steps=dg_steps)
        walsh_rep = self.walsh_rademacher_analysis(np.diag(k_reg))

        evidences: Tuple[Evidence, ...] = (
            Evidence("I.psd_estricto", -kernel_rep.min_eigenvalue_reg, 0.0, hard=True),
            Evidence("I.simetria_nucleo", kernel_rep.symmetry_residual, 1e-10),
            Evidence("I.smith_sin_torsion", 0.0 if tellegen_rep.torsion_free else 1.0, 0.0, hard=True),
            Evidence("I.incidencia_orientada_TU", 0.0 if tellegen_rep.is_oriented_incidence else 1.0, 0.0, hard=True),
            Evidence("I.betti0_consistente",
                     abs(tellegen_rep.betti_0_components - (tellegen_rep.n_nodes - tellegen_rep.smith_rank))
                     if tellegen_rep.n_edges else 0.0, 0.0, hard=True),
            Evidence("I.kcl", tellegen_rep.kcl_residual, 1e-9, hard=True),
            Evidence("I.kvl_espacio_de_lazos", tellegen_rep.kvl_cycle_residual, 1e-9, hard=True),
            Evidence("I.tellegen_potencia", tellegen_rep.tellegen_power_residual, 1e-9, hard=True),
            Evidence("I.cross_tellegen", tellegen_rep.cross_tellegen_residual, 1e-9),
            Evidence.boolean("I.joule_no_negativo", tellegen_rep.joule_power_w >= 0.0, hard=True),
            Evidence("I.secante_itoh_abe", dg_rep.secant_residual_max, 1e-12, hard=True),
            Evidence("I.baricentro_simetrico_w05", dg_rep.barycentric_symmetry_residual, 1e-14),
            Evidence("I.ia05_equiv_gonzalez", dg_rep.gonzalez_midpoint_equivalence_residual, 1e-12, hard=True),
            Evidence("I.ley_disipacion_exacta", dg_rep.energy_law_residual_max, 1e-9, hard=True),
            Evidence.boolean("I.energia_monotona", dg_rep.energy_monotone, hard=True),
            Evidence("I.cayley_contractivo", dg_rep.cayley_spectral_radius - 1.0, 0.0, hard=True),
            Evidence("I.walsh_ortogonalidad", walsh_rep.orthogonality_residual, 1e-12, hard=True),
            Evidence("I.walsh_secuencia", float(walsh_rep.sequency_order_defect), 0.0, hard=True),
            Evidence("I.walsh_plancherel", walsh_rep.parseval_residual, 1e-12, hard=True),
        )
        truth = HeytingTruthValue.infimum([e.truth for e in evidences])
        handoff = PhaseIHandoff(
            kernel=kernel_rep, tellegen=tellegen_rep, discrete_gradient=dg_rep, walsh=walsh_rep,
            evidences=evidences, truth_value=truth, kernel_digest=_array_digest(k_reg),
            kernel_reg=k_reg, eigenvalues=eigvals, eigenvectors=eigvecs, incidence=incidence,
        )
        logger.info("Fase I terminada: Ω₃=%s (κ=%.3e, β₀=%d, β₁=%d).",
                    truth.name, kernel_rep.condition_number,
                    tellegen_rep.betti_0_components, tellegen_rep.betti_1_cycles)
        return handoff  # ──► Fase II §II.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE II · INFORMES
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GibbsKMSReport:
    """§II.2 — Gibbs, KMS (Tomita–Takesaki) y entropía relativa."""

    beta: float
    log_partition: float
    free_energy_j: float
    von_neumann_entropy: float
    stationarity_residual: float
    kms_residual: float
    gibbs_relative_entropy: float
    tomita_j_squared_residual: float
    tomita_j_delta_j_residual: float
    modular_selfadjoint_residual: float
    modular_log_range: float
    rho_hermiticity_residual: float
    rho_min_eigenvalue: float
    is_kms_satisfied: bool
    is_gibbs_input: bool


@dataclass(frozen=True)
class JarzynskiCrooksReport:
    """§II.3 — Jarzynski–Crooks con estimadores estables y diagnóstico de muestra efectiva."""

    beta: float
    n_forward: int
    n_reverse: int
    work_mean_j: float
    work_sigma_j: float
    delta_f_true_j: float
    delta_f_jarzynski_j: float
    delta_f_bar_j: float
    delta_f_bar_sigma_j: float
    delta_f_crooks_fit_j: float
    log_jarzynski_lhs: float
    log_jarzynski_rhs: float
    analytic_identity_residual: float
    irreversible_work_j: float
    ess_fraction: float
    crooks_slope: float
    crooks_kl_residual: float
    is_jarzynski_valid: bool


@dataclass(frozen=True)
class NovikovReport:
    """§II.4 — Valuación de rango 1, norma ultramétrica y política de filtración."""

    valuation: float
    novikov_norm: float
    is_unit: bool
    n_terms: int
    cutoff: float
    ultrametric_residual: float
    multiplicativity_residual: float
    filtration_floor: float
    filtration_compliant: bool


@dataclass(frozen=True)
class FloerGromovReport:
    """§II.5+§II.6 — Residual CR-Floer con cota de truncamiento y obstrucciones de capacidad."""

    hamiltonian_coupling_c: float
    cr_residual: float
    cr_truncation_bound: float
    cr_control_residual: float
    detection_margin: float
    energy_area_residual: float
    ball_radius: float
    cylinder_radius: float
    c1_ball: float
    c1_cylinder: float
    ellipsoid_c1: float
    ellipsoid_c4: float
    ball_c4: float
    nonsqueezing_ok: bool
    ellipsoid_into_ball_ok: bool


@dataclass(frozen=True)
class PhaseIIHandoff:
    """§II.7 [TERMINAL FASE II] — anida PhaseIHandoff; entrada de la Fase III (§III.1)."""

    phase_i: PhaseIHandoff
    gibbs_kms: GibbsKMSReport
    jarzynski_crooks: JarzynskiCrooksReport
    novikov: NovikovReport
    floer_gromov: FloerGromovReport
    evidences: Tuple[Evidence, ...]
    truth_value: HeytingTruthValue
    hamiltonian_digest: str


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE II · MOTOR: TERMODINÁMICA ESPECTRAL Y GEOMETRÍA SIMPLÉCTICA  (anida la Fase I)
# ══════════════════════════════════════════════════════════════════════════════════════
class PhaseIIThermoSymplectic(PhaseIAlgebraicSubstrate):
    r"""
    FASE II — Anida la Fase I por herencia y por handoff. Su primer método consume el último
    método de la Fase I; su último método alimenta el primero de la Fase III.
    """

    PHASE_LABEL: Final[str] = "II·TERMODINAMICA-ESPECTRAL-SIMPLECTICA"

    # ─────────── §II.1 ENTRADA: continuación exacta del terminal de Fase I ───────────
    def accept_phase_i_handoff(self, handoff: PhaseIHandoff) -> PhaseIHandoff:
        r"""
        Valida el contrato de la Fase I. La adjudicación es monótona: si Fase I es ⊥, el
        ínfimo propagará ⊥ (⊥ ∧ x = ⊥); la auditoría continúa para reportar TODA la evidencia.
        """
        if not isinstance(handoff, PhaseIHandoff):
            raise WeaverValidationError("Fase II requiere el PhaseIHandoff de §I.6.")
        if _array_digest(handoff.kernel_reg) != handoff.kernel_digest:
            raise WeaverValidationError("Handoff Fase I: digest del núcleo inconsistente (tampering).")
        if handoff.truth_value is HeytingTruthValue.VETOED:
            logger.warning("Fase I en ⊥: la Fase II audita, pero el veredicto final será ⊥.")
        return handoff

    # ───────────────────────── §II.2 Gibbs / KMS / Tomita ─────────────────────────
    def gibbs_kms_certificate(
        self,
        hamiltonian: NDArray,
        *,
        beta: float = 1.0,
        rho_mac_input: Optional[NDArray] = None,
        n_probe_pairs: int = 6,
    ) -> GibbsKMSReport:
        r"""
        Condición KMS a temperatura inversa β (unidades k_B=1):

            ρ_β = e^{−βH}/Z,   σ_t(B) = e^{iHt} B e^{−iHt},   σ_{iβ}(B) = e^{−βH} B e^{βH},
            ω(A σ_{iβ}(B)) = ω(B A)      ∀ A,B  (KMS, Bratteli–Robinson §5.3).

        Cálculo SIN overflow por ciclicidad: con D = diag(e^{−β(E−E_min)}),
            Tr(ρ A e^{−βH} B e^{βH}) = Tr(D⁻¹ ρ̃ Ã D · B̃),
        y para ρ = ρ_β se reduce a Tr(Ã B̃)/Z (identidad exacta, cancelación-free).

        Tomita–Takesaki en dimensión finita (base propia de ρ_β, p_i>0):
            Δ(X)_ij = (p_i/p_j) X_ij,   J(X)_ij = √(p_i/p_j) · conj(X_ji),
            J² = id,  JΔJ = Δ⁻¹,  Δ = Δ* > 0 en el producto de Hilbert–Schmidt.

        CRÍTICA v13: usaba ‖[ρ,H]‖_F como "conmutador modular" y aprobaba ρ=I/d, que NO es
        KMS de un H no degenerado a β>0. Aquí I/d se detecta (½) vía S(ρ‖ρ_β) > 0.
        """
        h_mat = _as_matrix(hamiltonian, "hamiltonian", square=True, complex_ok=True)
        if not (math.isfinite(beta) and beta > 0.0):
            raise WeaverValidationError("β debe ser finito y positivo.")
        dim = h_mat.shape[0]
        herm_res_h = float(la.norm(h_mat - h_mat.conj().T, ord="fro")) / (float(la.norm(h_mat, ord="fro")) + _EPS)
        if herm_res_h > 1e-10:
            raise WeaverValidationError("H no hermítico: KMS no está definido.")
        energies, u_mat = la.eigh(h_mat)
        shift = float(energies[0])
        log_p = -beta * (energies - shift)
        log_z_shift = _logsumexp(log_p)
        log_p = log_p - log_z_shift
        probs = np.exp(log_p)
        log_z = log_z_shift - beta * shift
        rho_gibbs = (u_mat * probs) @ u_mat.conj().T

        if rho_mac_input is None:
            rho = rho_gibbs
            is_gibbs_input = True
            rho_herm, rho_min = 0.0, float(np.min(probs))
        else:
            rho_in = _as_matrix(rho_mac_input, "rho_mac_input", square=True, complex_ok=True)
            if rho_in.shape != (dim, dim):
                raise WeaverValidationError("rho_mac_input: dimensión incompatible con H.")
            rho_herm = float(la.norm(rho_in - rho_in.conj().T, ord="fro")) / (float(la.norm(rho_in, ord="fro")) + _EPS)
            rho_in = 0.5 * (rho_in + rho_in.conj().T)
            trace = float(np.real(np.trace(rho_in)))
            if trace <= 0.0:
                raise WeaverValidationError("rho_mac_input: traza no positiva (no es estado).")
            rho = rho_in / trace
            rho_min = float(np.min(la.eigvalsh(rho)))
            is_gibbs_input = False

        h_fro = float(la.norm(h_mat, ord="fro")) + _EPS
        r_fro = float(la.norm(rho, ord="fro")) + _EPS
        stationarity = float(la.norm(rho @ h_mat - h_mat @ rho, ord="fro")) / (r_fro * h_fro)

        modular_range = float(beta * (energies[-1] - energies[0]))
        clip = min(modular_range, _MODULAR_LOG_CAP)
        d_vec = np.exp(np.clip(-beta * (energies - shift), -clip, clip))
        rho_t = u_mat.conj().T @ rho @ u_mat
        rng = np.random.default_rng(self._seed + 101)
        kms_res = 0.0
        for _ in range(int(n_probe_pairs)):
            a_r = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
            b_r = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
            a_mat = 0.5 * (a_r + a_r.conj().T)
            b_mat = 0.5 * (b_r + b_r.conj().T)
            a_t = u_mat.conj().T @ a_mat @ u_mat
            b_t = u_mat.conj().T @ b_mat @ u_mat
            lhs = complex(np.trace(((rho_t @ a_t) / d_vec[:, None] * d_vec[None, :]) @ b_t))
            rhs = complex(np.trace(rho_t @ b_t @ a_t))
            kms_res = max(kms_res, abs(lhs - rhs) / (abs(lhs) + abs(rhs) + _EPS))

        # Tomita–Takesaki sobre ρ_β (rango completo por construcción)
        log_ratio = np.clip(log_p[:, None] - log_p[None, :], -clip, clip)
        ratio = np.exp(log_ratio)
        x_r = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        y_r = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        j_op = lambda z: np.sqrt(ratio) * np.conj(z.T)                # noqa: E731
        delta_op = lambda z: ratio * z                                # noqa: E731
        x_norm = float(la.norm(x_r, ord="fro")) + _EPS
        j2_res = float(la.norm(j_op(j_op(x_r)) - x_r, ord="fro")) / x_norm
        jdj_res = float(la.norm(j_op(delta_op(j_op(x_r))) - x_r / ratio, ord="fro")) / x_norm
        lhs_sa = complex(np.vdot(y_r, delta_op(x_r)))
        rhs_sa = complex(np.vdot(delta_op(y_r), x_r))
        sa_res = abs(lhs_sa - rhs_sa) / (abs(lhs_sa) + abs(rhs_sa) + _EPS)

        # Entropía relativa de Umegaki: S(ρ‖ρ_β) = Tr ρ log ρ − Σ_i log p_i · (U†ρU)_ii ≥ 0
        rho_eigs = np.maximum(la.eigvalsh(rho), 0.0)
        tr_rho_log_rho = float(np.sum(_xlogx(rho_eigs)))
        tr_rho_log_gibbs = float(np.sum(log_p * np.real(np.diag(rho_t))))
        rel_entropy = tr_rho_log_rho - tr_rho_log_gibbs

        return GibbsKMSReport(
            beta=float(beta), log_partition=float(log_z), free_energy_j=float(-log_z / beta),
            von_neumann_entropy=-tr_rho_log_rho, stationarity_residual=stationarity,
            kms_residual=kms_res, gibbs_relative_entropy=rel_entropy,
            tomita_j_squared_residual=j2_res, tomita_j_delta_j_residual=jdj_res,
            modular_selfadjoint_residual=sa_res, modular_log_range=modular_range,
            rho_hermiticity_residual=rho_herm, rho_min_eigenvalue=rho_min,
            is_kms_satisfied=bool(kms_res <= 1e-10), is_gibbs_input=is_gibbs_input,
        )

    # ───────────────────────── §II.3 Jarzynski–Crooks / BAR ─────────────────────────
    def jarzynski_crooks_certificate(
        self,
        *,
        beta: float = 1.0,
        work_mean: float = 5.0,
        delta_f: float = 4.8,
        work_forward: Optional[NDArray] = None,
        work_reverse: Optional[NDArray] = None,
        n_samples: int = 20_000,
        n_bins: int = 60,
    ) -> JarzynskiCrooksReport:
        r"""
        Igualdad de Jarzynski ⟨e^{−βW}⟩ = e^{−βΔF} y teorema de Crooks
            P_F(W)/P_R(−W) = e^{β(W−ΔF)}.

        Modelo sintético EXACTO (respuesta lineal / armónico): W_F ~ 𝒩(ΔF + βσ²/2, σ²),
        W_R ~ 𝒩(−ΔF + βσ²/2, σ²) satisface Crooks y Jarzynski analíticamente; σ se calibra
        con los parámetros heredados de v13: βσ²/2 = ⟨W⟩ − ΔF (=0.2 J por defecto).

        Estimadores:
        • Jarzynski en log-dominio: ΔF̂ = −(1/β)[logsumexp(−βW_F) − ln N] (estable).
        • Jensen MUESTRAL (desigualdad determinista, no estadística): ΔF̂ ≤ ⟨W⟩_emp ⇒
          W_irr ≥ 0 siempre; se certifica como evidencia dura.
        • ESS de Kish: (Σw)²/(N Σw²) con w ∝ e^{−βW}: diagnostica dominancia de eventos raros.
        • BAR = MLE del modelo de tilt exponencial inducido por Crooks. Con x_F=W_F,
          x_R=−W_R y M=ln(n_F/n_R), π(x)=σ(β(x−ΔF)+M); la ecuación de score
              Σ_F σ(−(βx_F−βΔF+M)) = Σ_R σ(βx_R−βΔF+M)
          es monótona en ΔF ⇒ bisección con raíz única; σ_BAR = I_Fisher^{−1/2}.
        • Identidad de disipación: β(⟨W⟩−ΔF) = D(P_F ‖ P̃_R) ≥ 0 (Crooks ⇒ 2ª ley).
        """
        if not (math.isfinite(beta) and beta > 0.0):
            raise WeaverValidationError("β debe ser finito y positivo.")
        if work_mean < delta_f - 1e-12:
            raise WeaverValidationError("⟨W⟩ < ΔF viola la segunda ley (Jensen) para el modelo.")
        rng = np.random.default_rng(self._seed + 211)
        sigma = math.sqrt(2.0 * max(work_mean - delta_f, 1e-6) / beta)
        if work_forward is None or work_reverse is None:
            mu_f = delta_f + 0.5 * beta * sigma ** 2
            mu_r = -delta_f + 0.5 * beta * sigma ** 2
            w_f = rng.normal(mu_f, sigma, int(n_samples))
            w_r = rng.normal(mu_r, sigma, int(n_samples))
            analytic_res = abs(-beta * mu_f + 0.5 * beta ** 2 * sigma ** 2 + beta * delta_f)
        else:
            w_f = _as_vector(work_forward, "work_forward")
            w_r = _as_vector(work_reverse, "work_reverse")
            analytic_res = float("nan")   # sin modelo: no hay identidad analítica que auditar
        n_f, n_r = int(w_f.size), int(w_r.size)

        log_lhs = _logsumexp(-beta * w_f) - math.log(n_f)
        log_rhs = -beta * delta_f
        df_jar = -log_lhs / beta
        mean_w = float(np.mean(w_f))
        w_irr = mean_w - df_jar
        weights = np.exp(-beta * w_f - float(np.max(-beta * w_f)))
        ess = float(np.sum(weights) ** 2 / (n_f * np.sum(weights ** 2)))

        # ---- BAR por bisección del score monótono -------------------------------------------
        x_f, x_r = beta * w_f, -beta * w_r
        offset = math.log(n_f / n_r)

        def score(df: float) -> float:
            b = beta * df
            return float(np.sum(_expit(-(x_f - b + offset))) - np.sum(_expit(x_r - b + offset)))

        span = float(np.max(np.abs(w_f)) + np.max(np.abs(w_r)) + 10.0 / beta)
        lo, hi = -span, span
        for _ in range(60):
            if score(lo) <= 0.0 <= score(hi):
                break
            lo, hi = 2.0 * lo, 2.0 * hi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if score(mid) > 0.0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-13 / beta:
                break
        df_bar = 0.5 * (lo + hi)
        b_bar = beta * df_bar
        pi_f = _expit(x_f - b_bar + offset)
        pi_r = _expit(x_r - b_bar + offset)
        fisher = beta ** 2 * float(np.sum(pi_f * (1 - pi_f)) + np.sum(pi_r * (1 - pi_r)))
        sigma_bar = 1.0 / math.sqrt(max(fisher, _EPS))

        # ---- Pendiente de Crooks e identidad de disipación (histogramas) --------------------
        edge_lo = float(min(w_f.min(), (-w_r).min()))
        edge_hi = float(max(w_f.max(), (-w_r).max()))
        edges = np.linspace(edge_lo, edge_hi, int(n_bins) + 1)
        h_f, _ = np.histogram(w_f, bins=edges)
        h_r, _ = np.histogram(-w_r, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask = (h_f >= 8) & (h_r >= 8)
        if int(np.sum(mask)) >= 4:
            x_fit = beta * centers[mask]
            y_fit = np.log(h_f[mask] / h_r[mask])
            w_fit = 1.0 / (1.0 / h_f[mask] + 1.0 / h_r[mask])
            design = np.vstack([x_fit, np.ones_like(x_fit)]).T
            wls = la.lstsq(design * np.sqrt(w_fit)[:, None], y_fit * np.sqrt(w_fit))[0]
            slope, intercept = float(wls[0]), float(wls[1])
            df_crooks = -intercept / beta
            p_f = h_f[mask] / n_f
            p_r = h_r[mask] / n_r
            kl_emp = float(np.sum(p_f * np.log(p_f / p_r)))
            kl_res = abs(kl_emp - beta * (mean_w - df_bar)) / (1.0 + abs(beta * (mean_w - df_bar)))
        else:
            slope, df_crooks, kl_res = float("nan"), float("nan"), float("nan")

        return JarzynskiCrooksReport(
            beta=float(beta), n_forward=n_f, n_reverse=n_r, work_mean_j=mean_w,
            work_sigma_j=float(np.std(w_f, ddof=1)), delta_f_true_j=float(delta_f),
            delta_f_jarzynski_j=float(df_jar), delta_f_bar_j=float(df_bar),
            delta_f_bar_sigma_j=float(sigma_bar), delta_f_crooks_fit_j=float(df_crooks),
            log_jarzynski_lhs=float(log_lhs), log_jarzynski_rhs=float(log_rhs),
            analytic_identity_residual=float(analytic_res), irreversible_work_j=float(w_irr),
            ess_fraction=ess, crooks_slope=slope, crooks_kl_residual=kl_res,
            is_jarzynski_valid=bool(w_irr >= -1e-12),
        )

    # ───────────────────────── §II.4 Anillo de Novikov ─────────────────────────
    def novikov_valuation_certificate(
        self,
        exponents: NDArray,
        *,
        coefficients: Optional[NDArray] = None,
        cutoff: float = _NOVIKOV_CUTOFF,
        filtration_floor: float = _KRULL_FILTRATION_FLOOR,
    ) -> NovikovReport:
        r"""
        Anillo universal de Novikov Λ = { Σ a_i T^{λ_i} : λ_i ∈ ℝ, λ_i → +∞ } con valuación
        de rango 1 v(A) = min{λ_i : a_i ≠ 0} (v(0)=+∞) y norma no arquimediana ‖A‖ = e^{−v(A)}.

        Axiomas VERIFICADOS con series truncadas a Λ_max:
            ‖A+B‖ ≤ max(‖A‖,‖B‖)      (desigualdad ultramétrica fuerte)
            v(A·B) = v(A) + v(B)       (multiplicatividad; requiere Λ_max ≥ v(A)+v(B))
        El anillo de valuación Λ_{≥0} es local con ideal maximal {v>0}; A es unidad ⟺ v(A)=0.
        Terminología: v es una valuación real de rango 1; el anillo asociado tiene dimensión
        de Krull 1 (v13 llamaba "valuación de Krull" al mínimo de exponentes).

        CRÍTICA v13: `max(min_exp, 0.10)` FALSIFICABA v. Aquí v es el mínimo real y el umbral
        0.10 es una POLÍTICA de filtración de energía (convergencia de sumas de Floer).
        """
        lam = _as_vector(exponents, "exponents", allow_empty=True)
        if lam.size == 0:
            return NovikovReport(
                valuation=float("inf"), novikov_norm=0.0, is_unit=False, n_terms=0,
                cutoff=float(cutoff), ultrametric_residual=0.0, multiplicativity_residual=0.0,
                filtration_floor=float(filtration_floor), filtration_compliant=False,
            )
        coef = np.ones_like(lam) if coefficients is None else _as_vector(coefficients, "coefficients")
        if coef.size != lam.size:
            raise WeaverValidationError("Novikov: exponents y coefficients deben ser paralelos.")
        keep = lam <= cutoff
        lam, coef = lam[keep], coef[keep]
        order = np.argsort(lam, kind="stable")
        lam, coef = lam[order], coef[order]

        def valuation(l_vec: NDArray, c_vec: NDArray) -> float:
            nz = np.abs(c_vec) > 0.0
            return float(np.min(l_vec[nz])) if np.any(nz) else float("inf")

        val_a = valuation(lam, coef)
        norm_a = math.exp(-val_a) if math.isfinite(val_a) else 0.0
        rng = np.random.default_rng(self._seed + 307)
        lam_b = np.sort(rng.uniform(0.0, min(cutoff, 4.0), size=lam.size))
        coef_b = rng.standard_normal(lam.size)
        val_b = valuation(lam_b, coef_b)

        merged_l = np.concatenate([lam, lam_b])
        merged_c = np.concatenate([coef, coef_b])
        uniq = np.unique(merged_l)
        summed = np.array([np.sum(merged_c[merged_l == u]) for u in uniq])
        val_sum = valuation(uniq, summed)
        norm_sum = math.exp(-val_sum) if math.isfinite(val_sum) else 0.0
        ultra_res = max(0.0, norm_sum - max(norm_a, math.exp(-val_b))) / (max(norm_a, math.exp(-val_b)) + _EPS)

        prod_l = (lam[:, None] + lam_b[None, :]).reshape(-1)
        prod_c = (coef[:, None] * coef_b[None, :]).reshape(-1)
        trunc = prod_l <= cutoff
        val_prod = valuation(prod_l[trunc], prod_c[trunc])
        if math.isfinite(val_a + val_b) and (val_a + val_b) <= cutoff:
            mult_res = abs(val_prod - (val_a + val_b)) / (1.0 + abs(val_a + val_b))
        else:
            mult_res = float("nan")

        return NovikovReport(
            valuation=val_a, novikov_norm=norm_a, is_unit=bool(abs(val_a) <= _WILKINSON),
            n_terms=int(lam.size), cutoff=float(cutoff), ultrametric_residual=ultra_res,
            multiplicativity_residual=mult_res, filtration_floor=float(filtration_floor),
            filtration_compliant=bool(val_a >= filtration_floor),
        )

    # ───────────────────────── §II.5 Cauchy–Riemann–Floer ─────────────────────────
    def cauchy_riemann_floer_certificate(
        self,
        kernel_mean_diagonal: float,
        *,
        hamiltonian_scale: float = 0.1,
        n_tau: int = 32,
        n_s: int = 65,
        s_span: Tuple[float, float] = (-1.0, 0.0),
        modes: Sequence[Tuple[int, float, float]] = ((1, 0.7, 0.0), (2, 0.3, -0.2)),
    ) -> Tuple[float, float, float, float, float]:
        r"""
        Ecuación de Cauchy–Riemann–Floer en ℝ² ≅ ℂ con J = [[0,−1],[1,0]] y H(z)=½c‖z‖²:

            ∂_s u + J(∂_τ u − X_H(u)) = 0,   X_H = −J∇H  ⇒  ∂_s u + J∂_τ u − c u = 0.

        Familia EXACTA: u = e^{cs}·v con v holomorfa en z=s+iτ (v = Σ c_k e^{kz}), pues
        ∂_s u + J∂_τ u = c u + e^{cs}(∂_s v + J∂_τ v) = c u. La holomorfía equivale a las
        ecuaciones de Cauchy–Riemann f_s = g_τ, g_s = −f_τ.

        Operador auditado: derivada ESPECTRAL en τ (exacta para modos < n_τ/2) + diferencias
        centrales de 4º orden en s en el interior. COTA RIGUROSA de truncamiento:
            |error| ≤ (h_s⁴/30)·max|∂_s⁵ u|   (×4 de holgura),
        y se excluyen los cierres de frontera del certificado (declarado, no ocultado).
        Control: se añade un campo no holomorfo ⇒ residual O(1) ≫ cota (no vacuidad).

        Identidad energía-área: para u J-holomorfa, E(u)=∫|∂_s u|² = ∫u*ω, con
        ω(v,w)=⟨Jv,w⟩; cuadratura de Gauss–Legendre en s y trapecio exacto en τ.

        CRÍTICA v13: su "residual" ‖g+Jg‖/‖g‖ ≡ √2 para TODO g (J ortogonal antisimétrica).
        """
        c_coupling = float(hamiltonian_scale) * float(kernel_mean_diagonal)
        s_grid = np.linspace(s_span[0], s_span[1], int(n_s))
        h_s = float(s_grid[1] - s_grid[0])
        tau = np.arange(int(n_tau)) * (2.0 * math.pi / int(n_tau))
        s_mesh, t_mesh = np.meshgrid(s_grid, tau, indexing="ij")

        def build(s_v: NDArray, t_v: NDArray) -> Tuple[NDArray, NDArray, NDArray, float]:
            u = np.zeros(s_v.shape + (2,))
            du_s = np.zeros_like(u)
            du_t = np.zeros_like(u)
            fifth = 0.0
            for k, a_k, b_k in modes:
                lam = k + c_coupling
                env = np.exp(lam * s_v)
                cos_k, sin_k = np.cos(k * t_v), np.sin(k * t_v)
                px, py = a_k * cos_k - b_k * sin_k, a_k * sin_k + b_k * cos_k
                u[..., 0] += env * px
                u[..., 1] += env * py
                du_s[..., 0] += lam * env * px
                du_s[..., 1] += lam * env * py
                du_t[..., 0] += env * (-k) * (a_k * sin_k + b_k * cos_k)
                du_t[..., 1] += env * k * (a_k * cos_k - b_k * sin_k)
                fifth += abs(lam) ** 5 * float(np.max(env)) * math.hypot(a_k, b_k)
            return u, du_s, du_t, fifth

        u_grid, du_s_an, du_t_an, fifth_bound = build(s_mesh, t_mesh)

        def j_action(v: NDArray) -> NDArray:
            return np.stack([-v[..., 1], v[..., 0]], axis=-1)

        def spectral_d_tau(f: NDArray) -> NDArray:
            n = f.shape[1]
            wave = np.fft.fftfreq(n, d=1.0 / n)
            spec = np.fft.fft(f, axis=1)
            deriv = 1j * wave[None, :, None] * spec
            if n % 2 == 0:
                deriv[:, n // 2, :] = 0.0
            return np.real(np.fft.ifft(deriv, axis=1))

        def fd4_d_s(f: NDArray) -> NDArray:
            return (-f[4:] + 8.0 * f[3:-1] - 8.0 * f[1:-3] + f[:-4]) / (12.0 * h_s)

        interior = slice(2, -2)
        d_s_disc = fd4_d_s(u_grid)
        d_t_disc = spectral_d_tau(u_grid)[interior]
        u_int = u_grid[interior]
        residual_field = d_s_disc + j_action(d_t_disc) - c_coupling * u_int
        denom = (float(la.norm(du_s_an[interior])) + float(la.norm(du_t_an[interior]))
                 + abs(c_coupling) * float(la.norm(u_int)) + _EPS)
        cr_residual = float(la.norm(residual_field)) / denom
        cr_bound = 4.0 * (h_s ** 4 / 30.0) * fifth_bound * math.sqrt(residual_field.size) / denom

        delta = 0.5 * float(np.max(np.abs(du_s_an)))
        u_ctrl = u_grid.copy()
        u_ctrl[..., 1] += delta * s_mesh
        ctrl_field = fd4_d_s(u_ctrl) + j_action(spectral_d_tau(u_ctrl)[interior]) - c_coupling * u_ctrl[interior]
        cr_control = float(la.norm(ctrl_field)) / denom

        nodes, weights = np.polynomial.legendre.leggauss(48)
        s_gl = 0.5 * (s_span[1] - s_span[0]) * nodes + 0.5 * (s_span[1] + s_span[0])
        w_gl = 0.5 * (s_span[1] - s_span[0]) * weights
        s_gl_mesh, t_gl_mesh = np.meshgrid(s_gl, tau, indexing="ij")
        _, du_s_gl, du_t_gl, _ = build(s_gl_mesh, t_gl_mesh)
        d_tau = 2.0 * math.pi / int(n_tau)
        integrand_e = np.sum(du_s_gl ** 2, axis=-1)
        j_ds = j_action(du_s_gl)
        integrand_a = np.sum(j_ds * du_t_gl, axis=-1)
        energy = float(np.sum(w_gl[:, None] * d_tau * integrand_e))
        area = float(np.sum(w_gl[:, None] * d_tau * integrand_a))
        energy_area_res = abs(energy - area) / (abs(energy) + abs(area) + _EPS)
        return c_coupling, cr_residual, cr_bound, cr_control, energy_area_res

    # ───────────────────────── §II.6 Gromov / capacidades ─────────────────────────
    def floer_gromov_certificate(
        self,
        kernel_mean_diagonal: float,
        *,
        ball_r: float = 1.0,
        cylinder_r: float = 1.5,
        ellipsoid_axes_squared: Optional[Sequence[float]] = None,
    ) -> FloerGromovReport:
        r"""
        Rigidez de Gromov (1985) como OBSTRUCCIÓN NECESARIA de capacidad:
            B^{2n}(r) ↪ Z^{2n}(R) = B²(R)×ℝ^{2n−2}  ⇒  c(B(r)) = πr² ≤ πR² = c(Z(R)).
        Se añade la monotonía de las capacidades de Ekeland–Hofer para E(a) ↪ B(r):
            c_k(E(a)) ≤ c_k(B(r)) ∀k, con c_k(B(r)) = πr²⌈k/n⌉.
        c_HZ de un elipsoide convexo = c₁ = π min a_i. NO se afirma existencia de encajes.
        """
        if ball_r <= 0.0 or cylinder_r <= 0.0:
            raise WeaverValidationError("Radios de bola/cilindro deben ser positivos.")
        c_coupling, cr_res, cr_bound, cr_ctrl, ea_res = self.cauchy_riemann_floer_certificate(
            kernel_mean_diagonal
        )
        n_dim = 2
        axes = tuple(ellipsoid_axes_squared) if ellipsoid_axes_squared is not None else (
            float(ball_r) ** 2, (0.8 * float(ball_r)) ** 2
        )
        caps_e = _eh_capacities(axes, 4)
        caps_b = _eh_capacities([float(ball_r) ** 2] * n_dim, 4)
        c1_ball = math.pi * float(ball_r) ** 2
        c1_cyl = math.pi * float(cylinder_r) ** 2
        return FloerGromovReport(
            hamiltonian_coupling_c=c_coupling, cr_residual=cr_res, cr_truncation_bound=cr_bound,
            cr_control_residual=cr_ctrl,
            detection_margin=cr_ctrl / max(cr_bound, _EPS), energy_area_residual=ea_res,
            ball_radius=float(ball_r), cylinder_radius=float(cylinder_r),
            c1_ball=c1_ball, c1_cylinder=c1_cyl, ellipsoid_c1=caps_e[0], ellipsoid_c4=caps_e[3],
            ball_c4=caps_b[3],
            nonsqueezing_ok=bool(c1_ball <= c1_cyl * (1.0 + 1e-12)),
            ellipsoid_into_ball_ok=bool(all(e <= b * (1.0 + 1e-12) for e, b in zip(caps_e, caps_b))),
        )

    # ─────────── §II.7 [TERMINAL FASE II] → CONTINUACIÓN / ENTRADA DE LA FASE III ───────────
    def build_phase_ii_handoff(
        self,
        handoff_i: PhaseIHandoff,
        exponents: NDArray,
        *,
        beta: float = 1.0,
        rho_mac_input: Optional[NDArray] = None,
        work_mean: float = 5.0,
        delta_f: float = 4.8,
        work_forward: Optional[NDArray] = None,
        work_reverse: Optional[NDArray] = None,
        ball_r: float = 1.0,
        cylinder_r: float = 1.5,
        ellipsoid_axes_squared: Optional[Sequence[float]] = None,
    ) -> PhaseIIHandoff:
        r"""
        MÉTODO TERMINAL DE LA FASE II: consume §I.6 vía §II.1, ejecuta §II.2→§II.6 y entrega
        `PhaseIIHandoff`, argumento EXACTO de `PhaseIIICelestialRSI.accept_phase_ii_handoff`.
        """
        handoff_i = self.accept_phase_i_handoff(handoff_i)
        gibbs = self.gibbs_kms_certificate(handoff_i.kernel_reg, beta=beta, rho_mac_input=rho_mac_input)
        jarz = self.jarzynski_crooks_certificate(
            beta=beta, work_mean=work_mean, delta_f=delta_f,
            work_forward=work_forward, work_reverse=work_reverse,
        )
        novikov = self.novikov_valuation_certificate(exponents)
        floer = self.floer_gromov_certificate(
            float(np.mean(np.diag(handoff_i.kernel_reg))),
            ball_r=ball_r, cylinder_r=cylinder_r, ellipsoid_axes_squared=ellipsoid_axes_squared,
        )
        evidences: Tuple[Evidence, ...] = (
            Evidence("II.rho_hermitico", gibbs.rho_hermiticity_residual, 1e-10, hard=True),
            Evidence("II.rho_positivo", -gibbs.rho_min_eigenvalue, 1e-10, hard=True),
            Evidence("II.estacionariedad", gibbs.stationarity_residual, 1e-10),
            Evidence("II.kms_tomita_takesaki", gibbs.kms_residual, 1e-10),
            Evidence("II.entropia_relativa_klein", -gibbs.gibbs_relative_entropy, 1e-10, hard=True),
            Evidence("II.tomita_J2", gibbs.tomita_j_squared_residual, 1e-10, hard=True),
            Evidence("II.tomita_JDJ", gibbs.tomita_j_delta_j_residual, 1e-9, hard=True),
            Evidence("II.modular_autoadjunto", gibbs.modular_selfadjoint_residual, 1e-12, hard=True),
            Evidence("II.rango_modular", gibbs.modular_log_range - _MODULAR_LOG_CAP, 0.0),
            Evidence.boolean("II.jarzynski_jensen", jarz.is_jarzynski_valid, hard=True),
            Evidence("II.jarzynski_analitico",
                     jarz.analytic_identity_residual if math.isfinite(jarz.analytic_identity_residual) else 0.0,
                     1e-12, hard=True),
            Evidence("II.ess_kish", max(0.0, 0.05 - jarz.ess_fraction), 0.0),
            Evidence("II.bar_vs_jarzynski",
                     abs(jarz.delta_f_bar_j - jarz.delta_f_jarzynski_j),
                     3.0 * jarz.delta_f_bar_sigma_j + 1e-9),
            Evidence("II.crooks_pendiente",
                     abs(jarz.crooks_slope - 1.0) if math.isfinite(jarz.crooks_slope) else 0.0, 0.15),
            Evidence("II.crooks_kl_disipacion",
                     jarz.crooks_kl_residual if math.isfinite(jarz.crooks_kl_residual) else 0.0, 0.15),
            Evidence("II.novikov_ultrametrica", novikov.ultrametric_residual, 1e-12, hard=True),
            Evidence("II.novikov_multiplicatividad",
                     novikov.multiplicativity_residual if math.isfinite(novikov.multiplicativity_residual) else 0.0,
                     1e-12, hard=True),
            Evidence.boolean("II.novikov_filtracion", novikov.filtration_compliant),
            Evidence("II.floer_cr_vs_cota", floer.cr_residual - floer.cr_truncation_bound, 0.0, hard=True),
            Evidence.boolean("II.floer_control_detectado",
                             floer.cr_control_residual > 10.0 * floer.cr_truncation_bound, hard=True),
            Evidence("II.floer_energia_area", floer.energy_area_residual, 1e-10, hard=True),
            Evidence.boolean("II.gromov_no_squeezing", floer.nonsqueezing_ok, hard=True),
            Evidence.boolean("II.eh_elipsoide_en_bola", floer.ellipsoid_into_ball_ok),
        )
        truth = HeytingTruthValue.infimum([handoff_i.truth_value] + [e.truth for e in evidences])
        handoff_ii = PhaseIIHandoff(
            phase_i=handoff_i, gibbs_kms=gibbs, jarzynski_crooks=jarz, novikov=novikov,
            floer_gromov=floer, evidences=evidences, truth_value=truth,
            hamiltonian_digest=_array_digest(handoff_i.kernel_reg),
        )
        logger.info("Fase II terminada: Ω₃=%s (KMS=%.2e, ΔF_BAR=%.4f±%.4f, v=%.4f).",
                    truth.name, gibbs.kms_residual, jarz.delta_f_bar_j,
                    jarz.delta_f_bar_sigma_j, novikov.valuation)
        return handoff_ii  # ──► Fase III §III.1


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE III · INFORMES
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PoincareKAMReport:
    """§III.2+§III.3 — Divisores pequeños, ecuación homológica y no degenerancia KAM."""

    frequencies: Tuple[float, ...]
    diophantine_tau: float
    diophantine_gamma: float
    scan_order: int
    min_small_divisor: float
    resonance_detected: bool
    homological_residual: float
    lindstedt_sup_norm: float
    small_divisor_amplification: float
    twist_hessian_det: float
    twist_hessian_rank: int
    isoenergetic_bordered_det: float
    kepler_hessian_rank: int
    kepler_degenerate: bool
    kam_smallness_parameter: float
    is_kam_admissible: bool


@dataclass(frozen=True)
class MelnikovReport:
    """§III.4 — Integral de Melnikov homoclínica y umbral de caos transversal."""

    omega: float
    gamma_forcing: float
    delta_damping: float
    perturbation_epsilon: float
    analytic_amplitude: float
    damping_offset: float
    critical_ratio: float
    actual_ratio: float
    numeric_analytic_residual: float
    simple_zero_count: int
    min_slope_at_zeros: float
    chaos_detected: bool


@dataclass(frozen=True)
class SymplecticFlowReport:
    """§III.5 — Invariante integral de Poincaré y ausencia de deriva secular."""

    integrator_order: int
    step_h: float
    steps: int
    energy_max_deviation: float
    energy_secular_drift_per_step: float
    symplecticity_residual: float
    reversibility_residual: float


@dataclass(frozen=True)
class RSI3Report:
    """§III.6 — Automejora recursiva de nivel 3 certificada (Banach ⊗ Kleisli)."""

    level1_optimal_step: float
    level1_contraction_q: float
    level1_grid_relative_agreement: float
    level2_initial_residual: float
    level2_final_residual: float
    level2_quadratic_law_residual: float
    level2_burn_in_iterations: int
    invariant_ball_radius: float
    composite_lipschitz_q: float
    lip_t3_theoretical_bound: float
    lip_t3_empirical_probe: float
    banach_apriori_bound: float
    banach_aposteriori_bound: float
    observed_fixed_point_error: float
    contraction_iterations: int
    capacity_d1: float
    capacity_d2: float
    capacity_d3: float
    superlinear_regime: bool
    kleisli_monad_laws_ok: bool
    kleisli_ledger: Tuple[str, ...]


@dataclass(frozen=True)
class DiracAnnihilationReport:
    """§III.7 — Cinemática relativista e⁻e⁺ → 2γ y autorización criptográfica real."""

    electron_rest_energy_kev: float
    threshold_kev: float
    positron_kinetic_kev: float
    sqrt_s_kev: float
    photon_energy_cm_kev: float
    anomaly_norm: float
    token_verified: bool
    token_self_issued: bool
    event_authorized: bool
    signature: str


@dataclass(frozen=True)
class ThermalExergyWeaverSpectrumReport:
    """§III.8 [TERMINAL] — Informe final anidado: Fase I ⊂ Fase II ⊂ Fase III."""

    engine_version: str
    phase_ii: PhaseIIHandoff
    poincare_kam: PoincareKAMReport
    melnikov: MelnikovReport
    symplectic_flow: SymplecticFlowReport
    rsi3: RSI3Report
    dirac: DiracAnnihilationReport
    evidences: Tuple[Evidence, ...]
    truth_value: HeytingTruthValue
    input_digest: str
    passport_hmac: str
    execution_time_ms: float

    def to_json(self) -> str:
        """Serialización determinista (arrays → digest; sin telemetría temporal)."""
        return json.dumps(_freeze_primitives(self), sort_keys=True, indent=2, ensure_ascii=True)

    def legacy_summary(self) -> Dict[str, Any]:
        """Proyección de compatibilidad con los nombres de v13 (sólo lectura)."""
        p1, p2 = self.phase_ii.phase_i, self.phase_ii
        return {
            "truth_value": self.truth_value.name,
            "itoh_abe_tellegen_exact": p1.discrete_gradient.secant_residual_max <= 1e-12,
            "tellegen_power_residual": p1.tellegen.tellegen_power_residual,
            "irreversible_work_j": p2.jarzynski_crooks.irreversible_work_j,
            "kms_commutator_norm": p2.gibbs_kms.kms_residual,
            "rademacher_walsh_norm": p1.walsh.normalized_l2,
            "smith_torsion_free": p1.tellegen.torsion_free,
            "betti_number_zero": p1.tellegen.betti_0_components,
            "krull_valuation": p2.novikov.valuation,
            "novikov_norm": p2.novikov.novikov_norm,
            "cauchy_riemann_residual": p2.floer_gromov.cr_residual,
            "is_gromov_rigid": p2.floer_gromov.nonsqueezing_ok,
            "melnikov_chaos_detected": self.melnikov.chaos_detected,
            "banach_contraction_lip": self.rsi3.lip_t3_theoretical_bound,
            "capacity_jerk_d3C_dt3": self.rsi3.capacity_d3,
            "annihilation_energy_kev": self.dirac.sqrt_s_kev,
            "annihilation_signature": self.dirac.signature,
            "passport_hash": self.passport_hmac,
        }


# ══════════════════════════════════════════════════════════════════════════════════════
#  FASE III · MOTOR: POINCARÉ + RSI³ + CUSTODIA  (anida las Fases II y I)
# ══════════════════════════════════════════════════════════════════════════════════════
class PhaseIIICelestialRSI(PhaseIIThermoSymplectic):
    r"""
    FASE III — Mecánica celeste de Poincaré (Delaunay, Lindstedt, Melnikov, flujo simpléctico),
    automejora recursiva de nivel 3 con certificados de Banach y Kleisli, y custodia final
    (Dirac + pasaporte HMAC). Anida la Fase II por herencia y por handoff.
    """

    PHASE_LABEL: Final[str] = "III·POINCARE-KAM-RSI3-CUSTODIA"

    # ─────────── §III.1 ENTRADA: continuación exacta del terminal de Fase II ───────────
    def accept_phase_ii_handoff(self, handoff: PhaseIIHandoff) -> PhaseIIHandoff:
        """Valida el contrato anidado Fase I ⊂ Fase II antes de la mecánica celeste."""
        if not isinstance(handoff, PhaseIIHandoff):
            raise WeaverValidationError("Fase III requiere el PhaseIIHandoff de §II.7.")
        self.accept_phase_i_handoff(handoff.phase_i)
        if handoff.hamiltonian_digest != handoff.phase_i.kernel_digest:
            raise WeaverValidationError("Handoff Fase II: digest Hamiltoniano inconsistente.")
        return handoff

    # ─────────── §III.2+§III.3 Poincaré–Lindstedt y certificado KAM diophantino ───────────
    def poincare_kam_certificate(
        self,
        *,
        frequencies: Optional[Sequence[float]] = None,
        diophantine_tau: float = 1.05,
        scan_order: int = 48,
        twist_matrix: Optional[NDArray] = None,
        perturbation_epsilon: float = 0.01,
        n_theta: int = 32,
        kepler_mu: float = 1.0,
        kepler_l: float = 1.0,
    ) -> PoincareKAMReport:
        r"""
        Ecuación homológica de Poincaré–Lindstedt en coordenadas acción-ángulo (J,θ):
            {H₀,W} = H₁ − ⟨H₁⟩_{𝕋^d},  con {H₀,W} = −⟨ω, ∂_θ W⟩, ω = ∂H₀/∂J.
        En Fourier: −i⟨k,ω⟩ W_k = H̃_{1,k} ⇒ W_k = i H̃_{1,k}/⟨k,ω⟩ (k≠0). Los DIVISORES
        PEQUEÑOS se controlan con la condición diophantina
            |⟨k,ω⟩| ≥ γ/|k|₁^τ   para 0<|k|₁ ≤ K,
        que se VERIFICA por barrido entero exhaustivo (γ estimado = min |⟨k,ω⟩|·|k|₁^τ).

        No degenerancia (Kolmogorov): det ∂²H₀/∂J² ≠ 0; isoenergética (Arnold–Moser):
            det [[H₀'', ω],[ωᵀ, 0]] ≠ 0.
        Degenerancia kepleriana: H₀(L)=−μ²/(2L²) depende sólo de L ⇒ Hessiano de rango 1 en
        (L,G,H): el problema planetario PURO es degenerado y KAM requiere promediado secular
        (Arnold 1963). Se reporta explícitamente en lugar de asumirse "estabilidad KAM".
        """
        omega = np.asarray(frequencies if frequencies is not None else (1.0, 0.5 * (math.sqrt(5.0) - 1.0)),
                           dtype=np.float64)
        d = int(omega.size)
        if d < 1 or diophantine_tau <= 0.0:
            raise WeaverValidationError("KAM: d≥1 y τ>0 requeridos.")
        k_order = int(scan_order)
        gamma_est, min_div = float("inf"), float("inf")
        for k_vec in itertools.product(range(-k_order, k_order + 1), repeat=d):
            l1 = sum(abs(v) for v in k_vec)
            if l1 == 0 or l1 > k_order:
                continue
            div = abs(float(np.dot(np.asarray(k_vec, dtype=np.float64), omega)))
            min_div = min(min_div, div)
            gamma_est = min(gamma_est, div * (l1 ** diophantine_tau))
        resonance = bool(min_div < 1e-12)

        # ---- ecuación homológica en 𝕋^d ------------------------------------------------------
        shape = (int(n_theta),) * d
        grids = np.meshgrid(*[np.arange(n_theta) * (2.0 * math.pi / n_theta)] * d, indexing="ij")
        h1 = np.cos(grids[0])
        if d >= 2:
            h1 = h1 + 0.5 * np.cos(grids[0] - grids[1]) + 0.25 * np.sin(2.0 * grids[1])
        h1_hat = np.fft.fftn(h1)
        h1_hat -= np.mean(h1) * h1.size if False else 0.0     # media ya nula por construcción
        wave_axes = [np.fft.fftfreq(n_theta, d=1.0 / n_theta)] * d
        k_mesh = np.meshgrid(*wave_axes, indexing="ij")
        divisor = np.zeros(shape, dtype=np.float64)
        for i in range(d):
            divisor += omega[i] * k_mesh[i]
        nyquist = np.zeros(shape, dtype=bool)
        for i in range(d):
            nyquist |= np.abs(k_mesh[i]) == (n_theta // 2)
        safe = (np.abs(divisor) > 1e-14) & (~nyquist)
        w_hat = np.zeros(shape, dtype=np.complex128)
        w_hat[safe] = 1j * h1_hat[safe] / divisor[safe]
        w_real = np.real(np.fft.ifftn(w_hat))
        reconstructed = np.real(np.fft.ifftn(-1j * divisor * w_hat))
        target = h1 - float(np.mean(h1))
        hom_res = float(la.norm(reconstructed - target)) / (float(la.norm(target)) + _EPS)
        amp = float(np.max(np.abs(w_hat))) / (float(np.max(np.abs(h1_hat))) + _EPS)
        sup_w = float(np.max(np.abs(w_real)))

        # ---- no degenerancia -----------------------------------------------------------------
        twist = np.diag(np.linspace(1.0, 1.3, d)) if twist_matrix is None else _as_matrix(
            twist_matrix, "twist_matrix", square=True
        )
        if twist.shape[0] != d:
            raise WeaverValidationError("KAM: twist_matrix incompatible con d.")
        twist_det = float(la.det(twist))
        twist_rank = int(np.linalg.matrix_rank(twist))
        bordered = np.block([[twist, omega[:, None]], [omega[None, :], np.zeros((1, 1))]])
        iso_det = float(la.det(bordered))
        kepler_hess = np.diag([-3.0 * kepler_mu ** 2 / kepler_l ** 4] + [0.0] * max(0, d - 1))
        kepler_rank = int(np.linalg.matrix_rank(kepler_hess))
        smallness = float(perturbation_epsilon) * sup_w
        admissible = (not resonance) and abs(twist_det) > 1e-12 and abs(iso_det) > 1e-12 and smallness < 0.1
        return PoincareKAMReport(
            frequencies=tuple(float(v) for v in omega), diophantine_tau=float(diophantine_tau),
            diophantine_gamma=float(gamma_est), scan_order=k_order, min_small_divisor=float(min_div),
            resonance_detected=resonance, homological_residual=hom_res, lindstedt_sup_norm=sup_w,
            small_divisor_amplification=amp, twist_hessian_det=twist_det, twist_hessian_rank=twist_rank,
            isoenergetic_bordered_det=iso_det, kepler_hessian_rank=kepler_rank,
            kepler_degenerate=bool(kepler_rank < d), kam_smallness_parameter=smallness,
            is_kam_admissible=bool(admissible),
        )

    # ───────────────────────── §III.4 Melnikov homoclínico ─────────────────────────
    def melnikov_homoclinic_certificate(
        self,
        *,
        omega: float = 1.0,
        gamma_forcing: float = 0.05,
        delta_damping: float = 0.25,
        perturbation_epsilon: float = 0.01,
        t_max: float = 40.0,
        n_t: int = 1 << 14,
        n_phase: int = 256,
    ) -> MelnikovReport:
        r"""
        Duffing forzado y amortiguado:  ẍ − x + x³ = ε(γ cos ωt − δ ẋ).
        Órbita homoclínica del sistema no perturbado: x₀(t)=√2 sech t, ẋ₀(t)=−√2 sech t tanh t.

        Integral de Melnikov (Melnikov 1963; Guckenheimer–Holmes §4.5):
            M(τ) = ∫_{ℝ} ẋ₀(t)[γ cos(ω(t+τ)) − δ ẋ₀(t)] dt
                 = √2 π γ ω sech(πω/2) · sin(ωτ) − 4δ/3,
        usando ∫sech t e^{−iωt}dt = π sech(πω/2) y ∫sech²t tanh²t dt = 2/3.

        TEOREMA: si M tiene CEROS SIMPLES (M(τ*)=0, M'(τ*)≠0), para ε pequeño la variedad
        estable e inestable se intersectan transversalmente ⇒ herradura de Smale–Birkhoff ⇒
        caos. Umbral exacto:  γ/δ > R_crit(ω) = 4 cosh(πω/2)/(3√2 π ω).
        Se verifica la fórmula analítica contra cuadratura trapecial (convergencia exponencial
        para integrando analítico con decaimiento e^{−2|t|}).

        CRÍTICA v13: integraba ε sinθ cos(Jθ) en [0,2π], que es IDÉNTICAMENTE 0 para J entero:
        su detector de caos era vacío. Aquí se prueba detección positiva Y negativa.
        """
        if omega <= 0.0 or gamma_forcing < 0.0 or delta_damping < 0.0:
            raise WeaverValidationError("Melnikov: ω>0, γ≥0, δ≥0 requeridos.")
        sech_term = 1.0 / math.cosh(math.pi * omega / 2.0)
        amplitude = math.sqrt(2.0) * math.pi * gamma_forcing * omega * sech_term
        offset = 4.0 * delta_damping / 3.0
        critical_ratio = 4.0 * math.cosh(math.pi * omega / 2.0) / (3.0 * math.sqrt(2.0) * math.pi * omega)
        actual_ratio = gamma_forcing / delta_damping if delta_damping > 0.0 else float("inf")

        t_grid = np.linspace(-t_max, t_max, int(n_t))
        dt = float(t_grid[1] - t_grid[0])
        xdot0 = -math.sqrt(2.0) / np.cosh(t_grid) * np.tanh(t_grid)
        phases = np.linspace(0.0, 2.0 * math.pi / omega, 8, endpoint=False)
        num_res = 0.0
        for tau in phases:
            integrand = xdot0 * (gamma_forcing * np.cos(omega * (t_grid + tau)) - delta_damping * xdot0)
            numeric = float(np.sum(0.5 * (integrand[:-1] + integrand[1:])) * dt)
            analytic = amplitude * math.sin(omega * tau) - offset
            num_res = max(num_res, abs(numeric - analytic) / (amplitude + offset + 1e-12))

        tau_grid = np.linspace(0.0, 2.0 * math.pi / omega, int(n_phase), endpoint=False)
        m_vals = amplitude * np.sin(omega * tau_grid) - offset
        zeros: List[float] = []
        for i in range(tau_grid.size):
            j = (i + 1) % tau_grid.size
            if m_vals[i] == 0.0:
                zeros.append(float(tau_grid[i]))
            elif m_vals[i] * m_vals[j] < 0.0:
                lo, hi = float(tau_grid[i]), float(tau_grid[i]) + (2.0 * math.pi / omega) / int(n_phase)
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if (amplitude * math.sin(omega * lo) - offset) * (amplitude * math.sin(omega * mid) - offset) <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                zeros.append(0.5 * (lo + hi))
        slopes = [abs(amplitude * omega * math.cos(omega * z)) for z in zeros]
        simple = [s for s in slopes if s > 1e-10]
        chaos = bool(amplitude > offset + 1e-12 and len(simple) >= 2)
        return MelnikovReport(
            omega=float(omega), gamma_forcing=float(gamma_forcing), delta_damping=float(delta_damping),
            perturbation_epsilon=float(perturbation_epsilon), analytic_amplitude=amplitude,
            damping_offset=offset, critical_ratio=critical_ratio, actual_ratio=actual_ratio,
            numeric_analytic_residual=num_res, simple_zero_count=len(simple),
            min_slope_at_zeros=min(simple) if simple else 0.0, chaos_detected=chaos,
        )

    # ───────────────────────── §III.5 Flujo simpléctico (Poincaré) ─────────────────────────
    def poincare_symplectic_flow_certificate(
        self, *, step_h: float = 0.05, steps: int = 2000, q0: float = 0.4, p0: float = 0.0
    ) -> SymplecticFlowReport:
        r"""
        Péndulo H(q,p) = p²/2 − cos q ( autonomous, 1 grado de libertad). Integrador de
        Yoshida⁴ por triple salto del Störmer–Verlet: Φ_h = SV(x₁h)∘SV(x₀h)∘SV(x₁h),
        x₁ = 1/(2−2^{1/3}), x₀ = −2^{1/3}/(2−2^{1/3}).

        Certificados: (i) simplecticidad DΦᵀ J DΦ = J (invariante integral de Poincaré,
        Jacobiano por diferencias centrales); (ii) reversibilidad Φ_{−h}∘Φ_h = id;
        (iii) error de energía ACOTADO O(h⁴) sin deriva secular (Hamiltoniano sombra, BCH).
        """
        x1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        x0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))

        def leapfrog(z: NDArray, h: float) -> NDArray:
            q, p = float(z[0]), float(z[1])
            p -= 0.5 * h * math.sin(q)
            q += h * p
            p -= 0.5 * h * math.sin(q)
            return np.array([q, p])

        def yoshida(z: NDArray, h: float) -> NDArray:
            return leapfrog(leapfrog(leapfrog(z, x1 * h), x0 * h), x1 * h)

        energy = lambda z: 0.5 * float(z[1]) ** 2 - math.cos(float(z[0]))   # noqa: E731
        z = np.array([q0, p0], dtype=np.float64)
        e0 = energy(z)
        trace = [e0]
        for _ in range(int(steps)):
            z = yoshida(z, step_h)
            trace.append(energy(z))
        trace_arr = np.asarray(trace)
        max_dev = float(np.max(np.abs(trace_arr - e0)))
        drift = float(np.polyfit(np.arange(trace_arr.size), trace_arr, 1)[0])

        j2 = np.array([[0.0, 1.0], [-1.0, 0.0]])
        delta = 1e-6
        jac = np.zeros((2, 2))
        z_ref = np.array([q0, p0])
        for j in range(2):
            e_j = np.zeros(2)
            e_j[j] = delta
            jac[:, j] = (yoshida(z_ref + e_j, step_h) - yoshida(z_ref - e_j, step_h)) / (2.0 * delta)
        symp_res = float(la.norm(jac.T @ j2 @ jac - j2, ord="fro"))
        rev_res = float(la.norm(yoshida(yoshida(z_ref, step_h), -step_h) - z_ref))
        return SymplecticFlowReport(
            integrator_order=4, step_h=float(step_h), steps=int(steps), energy_max_deviation=max_dev,
            energy_secular_drift_per_step=drift, symplecticity_residual=symp_res,
            reversibility_residual=rev_res,
        )

    # ───────────────────────── §III.6 RSI³: Banach ⊗ Kleisli ─────────────────────────
    def rsi3_banach_kleisli_certificate(
        self,
        k_reg: NDArray,
        eigenvalues: NDArray,
        eigenvectors: NDArray,
        *,
        invariant_ball_radius: float = 0.25,
        burn_in_max: int = 96,
        contraction_steps: int = 12,
        probe_pairs: int = 48,
        sg_window: int = 7,
    ) -> RSI3Report:
        r"""
        AUTOMEJORA RECURSIVA DE NIVEL 3, con tres niveles ANIDADOS y certificados reales:

        NIVEL 1 (sintonía de parámetro). Sobre el flujo de gradiente discreto de §I.5, la
        transformada de Cayley tiene factor q(h)=max_i|(1−hλ_i/2)/(1+hλ_i/2)|. Minimización
        exacta: h* = 2/√(λ₋λ₊), q* = (√κ−1)/(√κ+1). Se verifica contra barrido geométrico.

        NIVEL 2 (refinamiento de operador). Newton–Schulz para A = I + h*K/2:
            X_{k+1} = X_k(2I − A X_k),  R_k = I − A X_k ⇒ R_{k+1} = R_k²  (convergencia
            cuadrática), con X₀ = 2/(λ₋(A)+λ₊(A))·I ⇒ ‖R₀‖₂ = (κ_A−1)/(κ_A+1) < 1.
            Se mide la ley cuadrática ‖R_{k+1}‖ = ‖R_k‖² y la traza de CAPACIDAD
            C_k = −log₁₀‖R_k‖₂ ≈ 2^k·C₀ (dígitos de exactitud ganados por iteración).

        NIVEL 3 (meta-contracción certificada). En el espacio de Banach producto
            𝔅 = ℝ^n × {X : ‖I−AX‖₂ ≤ r},  ‖(x,X)‖ = max(‖x‖₂, ‖X‖₂),
        el mapa T(x,X) = (Cayley(h*)x, X(2I−AX)) es autoaplicación (R↦R² ⇒ bola invariante)
        con Lip(T) ≤ q = max(q*, 2r) < 1, pues DT₂[E] = E(I−AX) + (I−XA)E. Por
        submultiplicatividad Lip(T³) ≤ q³. Se comprueba: sondeo empírico de Lip(T³),
        cotas de Banach a priori q_S^k/(1−q_S)‖z₁−z₀‖ y a posteriori q_S/(1−q_S)‖z_k−z_{k−1}‖,
        y el error real contra el punto fijo exacto (0, A⁻¹).

        Jerk de capacidad: dC/dt, d²C/dt², d³C/dt³ se MIDEN con Savitzky–Golay (ventana 7,
        grado 3) sobre la traza real, en ticks de iteración. En régimen cuadrático C_k ~ 2^k C₀
        ⇒ las tres derivadas son positivas: el "jerk > 0" de v13 se vuelve un HECHO MEDIDO
        (y no una constante inventada); en régimen lineal d³C/dt³ ≈ 0 y así se reporta.

        Kleisli: los niveles se componen con la mónada Certified (Writer sobre (Ω₃,∧,⊤));
        se verifican las tres leyes monádicas y T³ = level >=> level >=> level.
        """
        n = k_reg.shape[0]
        lam_min, lam_max = float(eigenvalues[0]), float(eigenvalues[-1])
        sqrt_kappa = math.sqrt(lam_max / lam_min)
        h_opt = 2.0 / math.sqrt(lam_min * lam_max)
        q_opt = (sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)

        def cayley_q(h: float) -> float:
            half = 0.5 * h * eigenvalues
            return float(np.max(np.abs((1.0 - half) / (1.0 + half))))

        grid = np.geomspace(0.05 * h_opt, 20.0 * h_opt, 401)
        h_grid = float(grid[int(np.argmin([cayley_q(h) for h in grid]))])
        grid_agreement = abs(h_grid - h_opt) / h_opt

        a_mat = np.eye(n) + 0.5 * h_opt * k_reg
        a_eigs = 1.0 + 0.5 * h_opt * eigenvalues
        a_inv = (eigenvectors * (1.0 / a_eigs)) @ eigenvectors.T
        identity = np.eye(n)
        x_pre = (2.0 / (float(a_eigs[0]) + float(a_eigs[-1]))) * identity

        residuals: List[float] = []
        quad_res = 0.0
        r_norm = float(la.norm(identity - a_mat @ x_pre, 2))
        residuals.append(r_norm)
        burn = 0
        while r_norm > invariant_ball_radius and burn < int(burn_in_max):
            x_pre = x_pre @ (2.0 * identity - a_mat @ x_pre)
            new_r = float(la.norm(identity - a_mat @ x_pre, 2))
            quad_res = max(quad_res, abs(new_r - r_norm ** 2) / (r_norm ** 2 + _EPS))
            r_norm = new_r
            residuals.append(r_norm)
            burn += 1
        x_pre = x_pre @ (2.0 * identity - a_mat @ x_pre)      # un paso extra: interior estricto
        residuals.append(float(la.norm(identity - a_mat @ x_pre, 2)))

        res_arr = np.asarray(residuals)
        capacity = -np.log10(np.maximum(res_arr, 1e-300))
        active = np.where(res_arr > 1e-14)[0]
        d1 = d2 = d3 = 0.0
        if active.size >= sg_window:
            center = int(active[active.size // 2])
            half = sg_window // 2
            center = min(max(center, half), capacity.size - half - 1)
            segment = capacity[center - half: center + half + 1]
            d1 = float(_savgol_derivative_coeffs(sg_window, 3, 1) @ segment)
            d2 = float(_savgol_derivative_coeffs(sg_window, 3, 2) @ segment)
            d3 = float(_savgol_derivative_coeffs(sg_window, 3, 3) @ segment)

        q_composite = max(q_opt, 2.0 * invariant_ball_radius)
        q_s = q_composite ** 3
        cayley_mat = a_inv @ (np.eye(n) - 0.5 * h_opt * k_reg)

        def level_map(z: Tuple[NDArray, NDArray]) -> Tuple[NDArray, NDArray]:
            x_v, x_m = z
            return cayley_mat @ x_v, x_m @ (2.0 * identity - a_mat @ x_m)

        def triple(z: Tuple[NDArray, NDArray]) -> Tuple[NDArray, NDArray]:
            return level_map(level_map(level_map(z)))

        def prod_norm(z: Tuple[NDArray, NDArray]) -> float:
            return max(float(la.norm(z[0])), float(la.norm(z[1], 2)))

        def prod_dist(a: Tuple[NDArray, NDArray], b: Tuple[NDArray, NDArray]) -> float:
            return max(float(la.norm(a[0] - b[0])), float(la.norm(a[1] - b[1], 2)))

        rng = np.random.default_rng(self._seed + 401)
        z_state = (0.05 * rng.standard_normal(n), x_pre.copy())
        fixed = (np.zeros(n), a_inv)
        z_prev = z_state
        z_state = triple(z_state)
        first_gap = prod_dist(z_state, z_prev)
        observed = prod_dist(z_state, fixed)
        for _ in range(int(contraction_steps) - 1):
            z_prev = z_state
            z_state = triple(z_state)
            observed = prod_dist(z_state, fixed)
        last_gap = prod_dist(z_state, z_prev)
        apriori = (q_s ** int(contraction_steps)) / (1.0 - q_s) * first_gap
        aposteriori = q_s / (1.0 - q_s) * last_gap

        probe_scale = 1e-6 * invariant_ball_radius / (float(la.norm(a_mat, 2)) + 1.0)
        lip_emp = 0.0
        for _ in range(int(probe_pairs)):
            pert = (rng.standard_normal(n) * probe_scale, rng.standard_normal((n, n)) * probe_scale)
            z_a = (z_state[0] + pert[0], z_state[1] + pert[1])
            denom = prod_dist(z_a, z_state)
            if denom > 0.0:
                lip_emp = max(lip_emp, prod_dist(triple(z_a), triple(z_state)) / denom)

        # ---- leyes de la mónada Writer / composición de Kleisli -------------------------------
        def f_cert(v: NDArray) -> Certified:
            return Certified(v * 0.5, ("f",), HeytingTruthValue.COHERENT)

        def g_cert(v: NDArray) -> Certified:
            return Certified(v + 1.0, ("g",), HeytingTruthValue.COHERENT)

        sample = rng.standard_normal(4)
        m_val = Certified(sample, ("m",), HeytingTruthValue.DEGRADED)
        left = Certified.unit(sample).bind(f_cert)
        left_ok = np.allclose(left.value, f_cert(sample).value) and left.ledger == f_cert(sample).ledger
        right = m_val.bind(Certified.unit)
        right_ok = np.allclose(right.value, m_val.value) and right.ledger == m_val.ledger and right.truth == m_val.truth
        assoc_l = m_val.bind(f_cert).bind(g_cert)
        assoc_r = m_val.bind(lambda v: f_cert(v).bind(g_cert))
        assoc_ok = (np.allclose(assoc_l.value, assoc_r.value) and assoc_l.ledger == assoc_r.ledger
                    and assoc_l.truth == assoc_r.truth)
        laws_ok = bool(left_ok and right_ok and assoc_ok)

        level_cert = lambda z: Certified(level_map(z), ("rsi-level",), HeytingTruthValue.COHERENT)  # noqa: E731
        triple_cert = kleisli_compose(kleisli_compose(level_cert, level_cert), level_cert)
        certified_run = triple_cert((0.05 * rng.standard_normal(n), x_pre.copy()))
        return RSI3Report(
            level1_optimal_step=h_opt, level1_contraction_q=q_opt,
            level1_grid_relative_agreement=grid_agreement,
            level2_initial_residual=float(res_arr[0]), level2_final_residual=float(res_arr[-1]),
            level2_quadratic_law_residual=quad_res, level2_burn_in_iterations=burn,
            invariant_ball_radius=float(invariant_ball_radius), composite_lipschitz_q=q_composite,
            lip_t3_theoretical_bound=q_s, lip_t3_empirical_probe=lip_emp,
            banach_apriori_bound=apriori, banach_aposteriori_bound=aposteriori,
            observed_fixed_point_error=observed, contraction_iterations=int(contraction_steps),
            capacity_d1=d1, capacity_d2=d2, capacity_d3=d3,
            superlinear_regime=bool(d1 > 0.0 and d2 > 0.0 and d3 > 0.0),
            kleisli_monad_laws_ok=laws_ok, kleisli_ledger=certified_run.ledger,
        )

    # ───────────────────────── §III.7 Dirac + pasaporte HMAC ─────────────────────────
    def mint_positron_token(self, context: bytes) -> str:
        """Emite un token HMAC-SHA256 con separación de dominio (la clave nunca viaja en el mensaje)."""
        return hmac.new(self._secret_key, b"POSITRON-v14|" + context, hashlib.sha256).hexdigest()

    def verify_positron_token(self, token: str, context: bytes) -> bool:
        """Verificación en tiempo constante (`compare_digest`), inmune a temporización."""
        expected = self.mint_positron_token(context)
        return hmac.compare_digest(expected, str(token))

    def dirac_annihilation_certificate(
        self,
        anomaly_norm: float,
        context: bytes,
        *,
        positron_token: Optional[str] = None,
        positron_kinetic_kev: float = 0.0,
        anomaly_threshold: float = 0.05,
    ) -> DiracAnnihilationReport:
        r"""
        Cinemática relativista e⁻ + e⁺ → 2γ con electrón en reposo (CODATA 2022):
            s = (p₋+p₊)² = 2m²c⁴ + 2mc²E₊,  E₊ = mc² + K  ⇒  √s = √(2mc²(mc²+E₊)),
        umbral √s ≥ 2m_ec² = 1021.99790138 keV (v13 escribía "1022.0" como constante mágica),
        y energía de cada fotón en el centro de masas = √s/2.

        Autorización: se exige TOKEN HMAC verificado (v13 confiaba en un booleano y además
        calculaba sha256(mensaje ‖ clave.hex()), que no es un MAC y expone material de clave).
        En modo autónomo el motor emite y verifica su propio token, marcado explícitamente.
        """
        if positron_kinetic_kev < 0.0:
            raise WeaverValidationError("La energía cinética del positrón no puede ser negativa.")
        self_issued = positron_token is None
        token = self.mint_positron_token(context) if self_issued else str(positron_token)
        verified = self.verify_positron_token(token, context)
        authorized = bool(verified and float(anomaly_norm) > anomaly_threshold)
        e_plus = _ME_C2_KEV + float(positron_kinetic_kev)
        sqrt_s = math.sqrt(2.0 * _ME_C2_KEV * (_ME_C2_KEV + e_plus))
        if authorized:
            payload = _canonical_json({
                "anomaly": float(f"{float(anomaly_norm):.15g}"),
                "sqrt_s_kev": float(f"{sqrt_s:.15g}"),
                "context": context.hex(),
            })
            signature = hmac.new(self._secret_key, b"DIRAC-v14|" + payload, hashlib.sha256).hexdigest()
        else:
            signature = "NO_ANNIHILATION_EVENT"
        return DiracAnnihilationReport(
            electron_rest_energy_kev=_ME_C2_KEV, threshold_kev=_TWO_ME_C2_KEV,
            positron_kinetic_kev=float(positron_kinetic_kev), sqrt_s_kev=sqrt_s,
            photon_energy_cm_kev=0.5 * sqrt_s, anomaly_norm=float(anomaly_norm),
            token_verified=bool(verified), token_self_issued=bool(self_issued),
            event_authorized=authorized, signature=signature,
        )

    def _passport_payload(self, report: ThermalExergyWeaverSpectrumReport) -> Dict[str, Any]:
        """
        Payload determinista del pasaporte: excluye telemetría temporal y el artefacto Dirac
        dependiente de nonce (la firma es un objeto no-repudiable independiente).
        """
        frozen = _freeze_primitives(report)
        for key in ("passport_hmac", "execution_time_ms"):
            frozen.pop(key, None)
        frozen["phase_ii"]["floer_gromov"].pop("detection_margin", None)
        dirac = frozen["dirac"]
        dirac.pop("signature", None)
        return frozen

    def verify_report_passport(self, report: ThermalExergyWeaverSpectrumReport) -> bool:
        """Reverifica el pasaporte HMAC del informe (determinista y a prueba de manipulación)."""
        expected = hmac.new(
            self._secret_key, b"PASSPORT-v14|" + _canonical_json(self._passport_payload(report)),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, report.passport_hmac)

    # ─────────── §III.8 [TERMINAL FASE III] ORQUESTADOR Y ADJUDICACIÓN Ω₃ ───────────
    def execute_thermal_exergy_weaver_audit(
        self,
        apu_cost_matrix: NDArray,
        exponents: NDArray,
        *,
        beta: float = 1.0,
        rho_mac_input: Optional[NDArray] = None,
        work_mean: float = 5.0,
        delta_f: float = 4.8,
        work_forward: Optional[NDArray] = None,
        work_reverse: Optional[NDArray] = None,
        ball_r: float = 1.0,
        cylinder_r: float = 1.5,
        ellipsoid_axes_squared: Optional[Sequence[float]] = None,
        kam_frequencies: Optional[Sequence[float]] = None,
        kam_scan_order: int = 48,
        diophantine_tau: float = 1.05,
        melnikov_omega: float = 1.0,
        melnikov_gamma: float = 0.05,
        melnikov_delta: float = 0.25,
        melnikov_epsilon: float = 0.01,
        positron_token: Optional[str] = None,
        positron_kinetic_kev: float = 0.0,
        anomaly_norm: Optional[float] = None,
        psd_floor: float = _PSD_FLOOR,
    ) -> ThermalExergyWeaverSpectrumReport:
        r"""
        MÉTODO TERMINAL DEL TEJIDO. Ejecuta la cadena anidada completa:

            §I.6 build_phase_i_handoff → §II.1 → §II.7 build_phase_ii_handoff → §III.1
            → §III.2..§III.7 → adjudicación Ω₃ → pasaporte HMAC determinista.

        ADJUDICACIÓN (tabla de decisión en la cadena de Gödel Ω₃):
            ⊥ VETOED   : fallo de evidencia DURA (hipótesis estructural, identidad exacta,
                         política de seguridad: PSD, torsión, Tellegen, secante, KMS-Tomita
                         estructural, no-squeezing, caos transversal, q ≥ 1).
            ½ DEGRADED : fallo de evidencia BLANDA (estadística, discretización marginal,
                         estado de entrada no Gibbs, filtración de Novikov, token no emitido).
            ⊤ COHERENT : todas las evidencias satisfechas.
        El ínfimo es monótono y absorbente: ninguna fase posterior puede "reparar" un ⊥ previo.
        """
        t_start = time.perf_counter()

        # ── Fase I (terminal) ──► Fase II (entrada) ──► Fase II (terminal) ──► Fase III (entrada)
        handoff_i = self.build_phase_i_handoff(apu_cost_matrix, psd_floor=psd_floor)
        handoff_ii = self.build_phase_ii_handoff(
            handoff_i, exponents, beta=beta, rho_mac_input=rho_mac_input, work_mean=work_mean,
            delta_f=delta_f, work_forward=work_forward, work_reverse=work_reverse,
            ball_r=ball_r, cylinder_r=cylinder_r, ellipsoid_axes_squared=ellipsoid_axes_squared,
        )
        handoff_ii = self.accept_phase_ii_handoff(handoff_ii)

        # ── Fase III: Poincaré, RSI³, Dirac ────────────────────────────────────────────────
        kam = self.poincare_kam_certificate(
            frequencies=kam_frequencies, diophantine_tau=diophantine_tau, scan_order=kam_scan_order,
            perturbation_epsilon=melnikov_epsilon,
        )
        melnikov = self.melnikov_homoclinic_certificate(
            omega=melnikov_omega, gamma_forcing=melnikov_gamma, delta_damping=melnikov_delta,
            perturbation_epsilon=melnikov_epsilon,
        )
        flow = self.poincare_symplectic_flow_certificate()
        rsi3 = self.rsi3_banach_kleisli_certificate(
            handoff_i.kernel_reg, handoff_i.eigenvalues, handoff_i.eigenvectors
        )
        anomaly = float(handoff_i.walsh.normalized_l2 if anomaly_norm is None else anomaly_norm)
        input_digest = hashlib.sha256(
            (_array_digest(handoff_i.kernel_reg) + "|" + _array_digest(np.asarray(exponents, dtype=np.float64))).encode("ascii")
        ).hexdigest()
        dirac = self.dirac_annihilation_certificate(
            anomaly, bytes.fromhex(input_digest),
            positron_token=positron_token, positron_kinetic_kev=positron_kinetic_kev,
        )

        evidences: Tuple[Evidence, ...] = (
            Evidence.boolean("III.sin_resonancia", not kam.resonance_detected, hard=True),
            Evidence("III.gamma_diophantino", max(0.0, 1e-3 - kam.diophantine_gamma), 0.0),
            Evidence("III.ecuacion_homologica", kam.homological_residual, 1e-10, hard=True),
            Evidence.boolean("III.twist_no_degenerado", abs(kam.twist_hessian_det) > 1e-12, hard=True),
            Evidence.boolean("III.isoenergetico_no_degenerado", abs(kam.isoenergetic_bordered_det) > 1e-12, hard=True),
            Evidence("III.pequenez_kam_heuristica", kam.kam_smallness_parameter, 0.1),
            Evidence("III.melnikov_analitico_vs_cuadratura", melnikov.numeric_analytic_residual, 1e-8, hard=True),
            Evidence.boolean("III.sin_caos_homoclinico", not melnikov.chaos_detected, hard=True,
                             note="cero simple de M ⇒ herradura de Smale (política: veto)"),
            Evidence("III.melnikov_validez_eps", melnikov.perturbation_epsilon, 0.05),
            Evidence("III.energia_simpléctica_acotada", flow.energy_max_deviation, 1e-3, hard=True),
            Evidence("III.deriva_secular", abs(flow.energy_secular_drift_per_step), 1e-9),
            Evidence("III.simplecticidad_poincare", flow.symplecticity_residual, 1e-6, hard=True),
            Evidence("III.reversibilidad", flow.reversibility_residual, 1e-10, hard=True),
            Evidence("III.rsi1_optimo_vs_barrido", rsi3.level1_grid_relative_agreement, 0.05),
            Evidence("III.rsi2_ley_cuadratica", rsi3.level2_quadratic_law_residual, 1e-8, hard=True),
            Evidence("III.banach_q_menor_1", rsi3.composite_lipschitz_q - 1.0, 0.0, hard=True),
            Evidence("III.lip_t3_cota", rsi3.lip_t3_empirical_probe - rsi3.lip_t3_theoretical_bound * (1.0 + 1e-9),
                     0.0, hard=True),
            Evidence("III.banach_apriori", rsi3.observed_fixed_point_error - rsi3.banach_apriori_bound, 1e-12, hard=True),
            Evidence("III.banach_aposteriori", rsi3.observed_fixed_point_error - rsi3.banach_aposteriori_bound, 1e-12, hard=True),
            Evidence.boolean("III.kleisli_leyes_monadicas", rsi3.kleisli_monad_laws_ok, hard=True),
            Evidence.boolean("III.dirac_token_verificado", dirac.token_verified),
            Evidence.boolean("III.dirac_umbral_cinematico", dirac.sqrt_s_kev >= dirac.threshold_kev - 1e-9, hard=True),
        )
        truth = HeytingTruthValue.infimum([handoff_ii.truth_value] + [e.truth for e in evidences])
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        draft = ThermalExergyWeaverSpectrumReport(
            engine_version=__version__, phase_ii=handoff_ii, poincare_kam=kam, melnikov=melnikov,
            symplectic_flow=flow, rsi3=rsi3, dirac=dirac, evidences=evidences, truth_value=truth,
            input_digest=input_digest, passport_hmac="", execution_time_ms=elapsed_ms,
        )
        passport = hmac.new(
            self._secret_key, b"PASSPORT-v14|" + _canonical_json(self._passport_payload(draft)),
            hashlib.sha256,
        ).hexdigest()
        report = dataclasses.replace(draft, passport_hmac=passport)
        logger.info("Fase III terminada: Ω₃=%s en %.2f ms (caos=%s, Lip(T³)≤%.9f).",
                    truth.name, elapsed_ms, melnikov.chaos_detected, rsi3.lip_t3_theoretical_bound)
        return report  # ── fin del tejido (informe anidado I ⊂ II ⊂ III)


class BoolNovikovThermalExergyWeaverEngine(PhaseIIICelestialRSI):
    """
    Fachada pública (nombre conservado por compatibilidad de integración).
    Herencia anidada: Fase I ⊂ Fase II ⊂ Fase III ⊂ Fachada.
    """

    PHASE_LABEL: Final[str] = "FACHADA·BOOL-NOVIKOV-THERMAL-EXERGY-WEAVER"


# ══════════════════════════════════════════════════════════════════════════════════════
#  AUTO-TEST: cada teorema se prueba, y cada detector se prueba también en su caso NEGATIVO
# ══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    engine = BoolNovikovThermalExergyWeaverEngine(secret_key=b"APU_WEAVER_SELFTEST_KEY_32BYTES!!")

    k_test = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    exps_test = np.array([0.15, 0.25, 0.40], dtype=np.float64)

    # ── 1) Régimen nominal: debe ser ⊤ COHERENT ─────────────────────────────────────────
    rep = engine.execute_thermal_exergy_weaver_audit(k_test, exps_test)
    p1, p2 = rep.phase_ii.phase_i, rep.phase_ii
    assert rep.truth_value is HeytingTruthValue.COHERENT, rep.truth_value
    assert p1.discrete_gradient.secant_residual_max < 1e-12          # identidad secante exacta
    assert p1.discrete_gradient.gonzalez_midpoint_equivalence_residual < 1e-12  # lema w*=1/2
    assert p1.tellegen.tellegen_power_residual < 1e-9                # Tellegen real
    assert p1.tellegen.torsion_free and p1.tellegen.smith_invariants == (1,)
    assert p1.walsh.sequency_order_defect == 0
    assert p2.gibbs_kms.kms_residual < 1e-10                         # KMS de Gibbs
    assert p2.jarzynski_crooks.irreversible_work_j >= 0.0            # Jensen muestral
    assert abs(p2.jarzynski_crooks.delta_f_bar_j - 4.8) < 0.1
    assert p2.novikov.valuation == 0.15                              # sin clamp falsificador
    assert p2.floer_gromov.cr_residual <= p2.floer_gromov.cr_truncation_bound
    assert p2.floer_gromov.detection_margin > 1e3                    # control no holomorfo
    assert not rep.melnikov.chaos_detected
    assert rep.rsi3.composite_lipschitz_q < 1.0
    assert rep.rsi3.lip_t3_empirical_probe <= rep.rsi3.lip_t3_theoretical_bound * (1 + 1e-9)
    assert rep.rsi3.kleisli_monad_laws_ok
    assert rep.dirac.token_verified and abs(rep.dirac.threshold_kev - 1021.99790138) < 1e-6
    assert engine.verify_report_passport(rep)

    # ── 2) No vacuidad de los detectores: Ω₃ = ⊥ ────────────────────────────────────────
    bad = engine.execute_thermal_exergy_weaver_audit(
        k_test, exps_test,
        rho_mac_input=np.eye(2, dtype=np.complex128) / 2.0,   # I/d NO es KMS de H no degenerado
        melnikov_gamma=1.0, melnikov_delta=0.05,              # γ/δ ≫ R_crit ⇒ caos transversal
        ball_r=2.0, cylinder_r=1.0,                           # viola no-squeezing
    )
    assert bad.truth_value is HeytingTruthValue.VETOED
    assert bad.melnikov.chaos_detected and bad.melnikov.simple_zero_count >= 2
    assert not bad.phase_ii.floer_gromov.nonsqueezing_ok
    assert bad.phase_ii.gibbs_kms.gibbs_relative_entropy > 1e-3

    # ── 3) Sólo evidencia blanda fallida: Ω₃ = ½ (no booleano) ──────────────────────────
    mid = engine.execute_thermal_exergy_weaver_audit(
        k_test, exps_test, rho_mac_input=np.eye(2, dtype=np.complex128) / 2.0
    )
    assert mid.truth_value is HeytingTruthValue.DEGRADED
    assert not mid.truth_value.is_decidable
    assert HeytingTruthValue.DEGRADED.negate() is HeytingTruthValue.VETOED
    assert HeytingTruthValue.DEGRADED.double_negation() is HeytingTruthValue.COHERENT

    # ── 4) Pasaporte determinista y detección de manipulación ───────────────────────────
    rep2 = engine.execute_thermal_exergy_weaver_audit(k_test, exps_test)
    assert rep2.passport_hmac == rep.passport_hmac                    # sin tiempo en el MAC
    tampered = dataclasses.replace(rep, truth_value=HeytingTruthValue.DEGRADED)
    assert not engine.verify_report_passport(tampered)

    # ── 5) Identidades auxiliares: capacidades EH de la bola y Melnikov analítico ───────
    caps_ball = _eh_capacities([1.0, 1.0], 6)
    assert all(abs(caps_ball[k] - math.pi * math.ceil((k + 1) / 2)) < 1e-12 for k in range(6))
    mel_hi = engine.melnikov_homoclinic_certificate(gamma_forcing=1.0, delta_damping=0.05)
    assert mel_hi.chaos_detected and mel_hi.numeric_analytic_residual < 1e-9

    print("=== AUTO-TEST BoolNovikovThermalExergyWeaverEngine v14 ===")
    print(f"Veredicto Ω₃                : {rep.truth_value.name} ({hex(rep.truth_value.value)})")
    print(f"Tellegen (potencia)         : {p1.tellegen.tellegen_power_residual:.3e}")
    print(f"Secante Itoh–Abe (max)      : {p1.discrete_gradient.secant_residual_max:.3e}")
    print(f"IA(w=½) ≡ González          : {p1.discrete_gradient.gonzalez_midpoint_equivalence_residual:.3e}")
    print(f"Smith / β₀ / β₁             : {p1.tellegen.smith_invariants} / "
          f"{p1.tellegen.betti_0_components} / {p1.tellegen.betti_1_cycles}")
    print(f"KMS (Gibbs)                 : {p2.gibbs_kms.kms_residual:.3e}")
    print(f"ΔF BAR [J]                  : {p2.jarzynski_crooks.delta_f_bar_j:.5f} "
          f"± {p2.jarzynski_crooks.delta_f_bar_sigma_j:.5f}  (ESS={p2.jarzynski_crooks.ess_fraction:.3f})")
    print(f"W_irr [J]                   : {p2.jarzynski_crooks.irreversible_work_j:.5f}")
    print(f"Valuación Novikov v(A)      : {p2.novikov.valuation:.4f}  ‖A‖={p2.novikov.novikov_norm:.6f}")
    print(f"CR-Floer / cota / control   : {p2.floer_gromov.cr_residual:.3e} / "
          f"{p2.floer_gromov.cr_truncation_bound:.3e} / {p2.floer_gromov.cr_control_residual:.3e}")
    print(f"Melnikov A / 4δ/3 / R_crit  : {rep.melnikov.analytic_amplitude:.5f} / "
          f"{rep.melnikov.damping_offset:.5f} / {rep.melnikov.critical_ratio:.5f}")
    print(f"KAM γ (τ={rep.poincare_kam.diophantine_tau}) : {rep.poincare_kam.diophantine_gamma:.5f}"
          f"  (kepleriano degenerado: {rep.poincare_kam.kepler_degenerate})")
    print(f"Lip(T³) cota / empírico     : {rep.rsi3.lip_t3_theoretical_bound:.9f} / "
          f"{rep.rsi3.lip_t3_empirical_probe:.9f}")
    print(f"Capacidad dC, d²C, d³C      : {rep.rsi3.capacity_d1:.4f}, "
          f"{rep.rsi3.capacity_d2:.4f}, {rep.rsi3.capacity_d3:.4f} "
          f"(superlineal={rep.rsi3.superlinear_regime})")
    print(f"√s Dirac [keV]              : {rep.dirac.sqrt_s_kev:.6f} (umbral {rep.dirac.threshold_kev:.6f})")
    print(f"Pasaporte HMAC (16)         : {rep.passport_hmac[:16]}…  verificado="
          f"{engine.verify_report_passport(rep)}")
    print(f"Controles ⊥ / ½             : {bad.truth_value.name} / {mid.truth_value.name}")
    print(f"Tiempo total [ms]           : {rep.execution_time_ms:.2f}")
    print("=== TODOS LOS CERTIFICADOS Y CONTRA-EJEMPLOS SATISFECHOS ===")
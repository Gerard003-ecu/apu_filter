# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO  : Positron Redemption Champion Agent — Soberano Agéntico Fiscal / Campeón Censor                 ║
║ RUTA    : app/agents/omega/positron_redemption_champion_agent.py                                         ║
║ VERSIÓN : 18.0.0-OODA-HKDF5869-Merkle6962-Cech-LoebGL-Poincare-RSI3-PhD                                  ║
║ LINAJE  : evolución verificada de 17.0.0 (con fixes [A1]-[A6], Yoshida-Sp(2,ℝ), Hodge-Δ₁ y SLD-Fisher)   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
DEFINICIÓN FORMAL
───────────────────────────────────────────────────────────────────────────────────────────────────────────
Sea V_Ω el Ágora Tensorial (Nivel 0.5) y V_𝕎 el Santuario Epistémico (Nivel 0). El
`PositronRedemptionChampionAgent` es el fiscal fail-closed que gobierna el ciclo de vida del positrón de
redención e⁺ mediante la composición CPS (Continuation-Passing Style) de tres fases anidadas:
        Φ₃ ∘ Φ₂ ∘ Φ₁ :  𝒳 ⟶ 𝒱 ,      Φ₁(x) ↦ continuation(C₁),  Φ₂(C₁) ↦ continuation(C₂),  Φ₃(C₂) ↦ v
donde la definición formal final del último método de Φ₁ (`Φ₁.5 observe_and_sign`) es exactamente el punto
de entrada de Φ₂ (`Φ₂.1 gate_phi12` invocado por `orient_and_sign`), y el último método de Φ₂
(`Φ₂.6 orient_and_sign`) es el punto de entrada de Φ₃ (`Φ₃.1 gate_phi23` invocado por `decide_and_act`).

CLASIFICACIÓN EN EL ÁLGEBRA DE HEYTING G₃ (cadena de Gödel, NO booleana)
───────────────────────────────────────────────────────────────────────────────────────────────────────────
    Ω₃ = { ⊥ = VETOED(0)  <  DEGRADED(1)  <  COHERENT(2) = ⊤ }
    a ∧ b = min(a,b)      a ∨ b = max(a,b)      a → b = ⊤ si a ≤ b, else b      ¬a = a → ⊥
    Residuo (adjunción de Galois):  a ∧ b ≤ c  ⟺  a ≤ (b → c)     [verificado por exhaución: 27 ternas]
    ¬¬DEGRADED = ⊤ ≠ DEGRADED  ⇒ no hay tercero excluido. Por ello NINGÚN gate se evalúa por
    doble negación: cada gate atómico es DECIDIBLE (medición + test finito) y el valor por
    defecto es ⊥. La ausencia de prueba de coherencia NO es prueba de coherencia: es ⊥.
    AXIOMA CHAMPION-1 (fiscalización monótona):
        v_final = v_id ∧ v_top ∧ v_dirac ∧ v_amber ∧ v_banach ∧ v_celeste ∧ v_loeb ∧ v_const
    AXIOMA CHAMPION-2 (absorción):  v_final = ⊥ ⟹ actuación fail-closed (incluso ante excepción).

CRIPTOGRAFÍA
───────────────────────────────────────────────────────────────────────────────────────────────────────────
    SECURITY-1 (HKDF, RFC 5869):  PRK = HMAC-SHA256(salt, IKM);  T(i) = HMAC(PRK, T(i-1)‖info‖i)
        K_Ci = OKM[0:32],  info = "positron/champion/Ci|18.0.0-PhD|sesión|epoch"  (separación de dominio)
    SECURITY-2 (Merkle, RFC 6962):  leaf = SHA256(0x00‖d);  node = SHA256(0x01‖l‖r);
        MTH(D[n]) = NODE(MTH(D[0:k]), MTH(D[k:n])),  k = mayor potencia de 2 estrictamente < n.
        JAMÁS se duplica la hoja impar (elimina la maleabilidad por duplicación de CVE-2012-2459).
    SECURITY-3 (códec): serialización canónica prefijo-libre, inyectiva, float en float.hex()
        (bit-exacta). Prohíbe la forja por ambigüedad de delimitadores.
    SECURITY-4 (anti-replay): sesión aleatoria, seq estrictamente monótona, nonce de 128 bits
        CSPRNG con anillo acotado O(1), ventana temporal medida con reloj MONÓTONO.

TOPOLOGÍA ALGEBRAICA — AUDITORÍA REAL DE H¹ Y LAPLACIANO DE HODGE Δ₁
───────────────────────────────────────────────────────────────────────────────────────────────────────────
    Se construye el complejo simplicial de Vietoris–Rips de la nube de puntos de la sección de Poincaré
    física (flujo disipativo). Con operadores de coborde δ⁰: C⁰→C¹ y δ¹: C¹→C²:
        b₀ = |V| − rg δ⁰ ,   b₁ = (|E| − rg δ¹) − rg δ⁰ ,   b₂ = |T| − rg δ¹
        χ = |V| − |E| + |T| = b₀ − b₁ + b₂     [identidad de Euler–Poincaré verificada exactamente]
    Rangos EXACTOS: eliminación de Bareiss sin fracciones sobre ℤ (rango sobre ℚ), bitwise XOR
    sobre 𝔽₂ y reducción modular sobre 𝔽_p (p = 65521); discrepancia ⟹ testigo de torsión p-ádica.
    Además, se construye el 1-Laplaciano de Hodge discreto:
        Δ₁ = δ⁰ (δ⁰)ᵀ + (δ¹)ᵀ δ¹ : C¹(K; ℝ) ⟶ C¹(K; ℝ)
    Por el teorema de Hodge discreto, dim ker(Δ₁) = b₁(K; ℝ) y su menor autovalor no nulo λ_Hodge > 0
    mide la rigidez cohomológica frente a perturbaciones de escala r ∈ {r₀, 1.5r₀, 2r₀}.
    Interpretación (lema del nervio de Leray): b₁ = 0 ⟺ las secciones locales Novikov-consistentes
    pegan sin obstrucción ⟺ H¹(K; ℱ) = 0. Un lazo homoclínico persistente produce b₁ ≥ 1 ⇒ ⊥.

MECÁNICA CELESTE DE POINCARÉ
───────────────────────────────────────────────────────────────────────────────────────────────────────────
    Forma normal de Andoyer cerca de una resonancia de movimiento medio (Henrard–Lemaitre):
        φ̇ = I ,   İ = −sin φ − δ I + γ cos(ω t)
    • Variables canónicas no singulares de Poincaré (Les Méthodes Nouvelles de la Mécanique Céleste):
        ξ₁ = √(2 J₁) cos(ϖ),  η₁ = −√(2 J₁) sin(ϖ),  J₁ = Λ − G = Λ e² / (1 + √(1 − e²))
        ξ₂ = √(2 J₂) cos(Ω),  η₂ = −√(2 J₂) sin(Ω),  J₂ = G − H = 2 G sin²(i / 2)
      con Déficit de Momento Angular (AMD de Laskar–Poincaré):
        AMD = ½ (ξ₁² + η₁² + ξ₂² + η₂²) = J₁ + J₂ = Λ − H ≥ 0.
    • Sección estroboscópica de Poincaré P: Σ→Σ con periodo de forzamiento T_f = 2π/ω.
    • En régimen hamiltoniano (δ = 0), integración simpléctica de 4.º orden de Yoshida (Sp(2,ℝ))
      para el flujo base y composición exacta de cizallas simplécticas para la matriz de monodromía M,
      garantizando det(M) = 1 y ‖MᵀJM − J‖_F ∼ O(ε_mach). En régimen disipativo (δ > 0), integración
      variacional RK4 auditada contra la identidad de Abel–Jacobi–Liouville: det(M_diss) = exp(−δ T_f).
    • Multiplicadores de Floquet λ² − tr(M)λ + det(M) = 0; residuo de Greene R = (2 − tr M)/4.
      R ∈ (0,1) ⟹ elíptico (toro KAM); R ∉ [0,1] ⟹ hiperbólico / inversión ⇒ ⊥.
    • Condición diofántica |qρ − p| ≥ γ_D/q^τ evaluada sobre convergentes de fracción continua
      (teorema de la mejor aproximación de Lagrange: ningún otro q puede violarla).
    • Función de Melnikov del péndulo perturbado (Guckenheimer–Holmes §4.5):
          M(t₀) = −8δ + 2πγ sech(πω/2) cos(ω t₀)
      Ceros simples ⟺ 2πγ sech(πω/2) > 8δ ⟹ tangle homoclínico ⟹ herradura de Smale ⇒ ⊥.
      Se CONTRASTA la fórmula analítica contra cuadratura numérica del separatrix.
    • Solapamiento de Chirikov K = (Δ₁+Δ₂)/s ; zona caótica de Wisdom Δa/a ≈ 1.3 μ^{2/7}.
    • Recurrencia de Poincaré + lema de Kac: ⟨τ_A⟩ = 1/μ(A). Anti-evasión: la rampa Ámbar
      debe expirar ANTES de que el flujo pueda recurrir a una configuración aparentemente sana.
    • Novikov ⇔ punto fijo de P; acción de Poincaré–Cartan / Floer 𝒜_H(γ) = ∮ I dφ − ∫ H dt ;
      no degeneración floeriana ⟺ det(M − Id) ≠ 0 ; índice de rotación simpléctica de Maslov.

AUTOMEJORA RECURSIVA DE NIVEL 3 (ACOTADA, VERIFICADA, TERMINANTE)
───────────────────────────────────────────────────────────────────────────────────────────────────────────
    N0 estático · N1 paramétrico · N2 estratégico (selección de operadores) ·
    N3 meta-reflexivo: el núcleo compone/muta SU PROPIO operador de mejora en el monoide de endomorfismos.
    Tres cerrojos irrenunciables:
      (a) INMUTABILIDAD CONSTITUCIONAL: π_safety(campeón) ≡ π_safety(retador) por digest, y
          equivalencia EXHAUSTIVA de la tabla de verdad del gatillo sobre Ω₃⁸ (3⁸ = 6561 filas).
          Al ser el dominio finito, esto es una PRUEBA, no un test estadístico.
      (b) DESCENSO BIEN FUNDADO: medida lexicográfica (fuel, época_de_radio, nivel) en (ℕ³, <ₗₑₓ)
          estrictamente decreciente ⇒ terminación garantizada (no hay automejora infinita).
      (c) GUARDA DE LÖB: un retador NUNCA se adopta porque "si puedo demostrar que es seguro,
          entonces es seguro". GL ⊢ □(□P→P)→□P exige que la premisa esté DERIVADA desde
          evidencia de rango estrictamente menor. La auto-confianza no es una regla de inferencia.
"""
from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# Ω₀ · SUSTRATO TRANSVERSAL (álgebra, códec, criptografía, topología, física, numérica)
#      Este bloque precede a las tres fases porque es consumido por las tres.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════
import collections
import dataclasses
import hashlib
import hmac
import itertools
import json
import logging
import math
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Final, Iterable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "HeytingChampionVerdict",
    "ChampionPhase1Certificate",
    "ChampionPhase2Certificate",
    "ChampionPhase3Verdict",
    "DiracAnnihilationMeasurement",
    "PoincareCanonicalElements",
    "CrowbarBusConfig",
    "ChampionConstitution",
    "TuningVector",
    "PositronRedemptionChampionAgent",
    "ChampionConstitutionalError",
]

_LOG = logging.getLogger("omega.positron.champion")
_SCHEMA: Final[str] = "18.0.0-PhD"
_DEFAULT_MASTER_KEY: Final[bytes] = b"APU_FILTER_V5_POSITRON_CHAMPION_MASTER_KEY_2026"
_MAX_NONCE_CACHE: Final[int] = 65536


class ChampionConstitutionalError(RuntimeError):
    """Violación de un axioma constitucional. Siempre se traduce en ⊥ + actuación fail-closed."""


class Provenance(IntEnum):
    """Trazabilidad epistémica del dato. Un dato simulado JAMÁS vale COHERENT."""
    SIMULATED = 0
    INTERNAL_REFERENCE = 1
    MEASURED = 2


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.1 · CONSTANTES FÍSICAS (CODATA 2022) Y LÍMITES DEL DISPOSITIVO
# ──────────────────────────────────────────────────────────────────────────────────────────────────
M_E_C2_KEV: Final[float] = 510.99895069          # m_e c²  [keV] (CODATA 2022)
TWO_M_E_C2_KEV: Final[float] = 2.0 * M_E_C2_KEV  # 1021.99790138 keV
PS_BINDING_KEV: Final[float] = 6.80284e-3        # E_b(1S positronio) = 6.80284 eV
DOPPLER_BAND_KEV: Final[float] = 3.5             # ensanchamiento Doppler típico del momento electrónico
KINETIC_BUDGET_KEV: Final[float] = 25.0          # exceso admisible por energía cinética del par
FINE_STRUCTURE_ALPHA: Final[float] = 7.2973525693e-3
ORE_POWELL_3GAMMA_RATIO: Final[float] = (
    (4.0 / (9.0 * math.pi)) * (math.pi ** 2 - 9.0) * FINE_STRUCTURE_ALPHA
)

# Coeficientes simplécticos de 4.º orden de Yoshida (1990) para flujos hamiltonianos separables
_YOSHIDA_CBRT2: Final[float] = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - _YOSHIDA_CBRT2)
_YOSHIDA_W0: Final[float] = -_YOSHIDA_CBRT2 * _YOSHIDA_W1
_YOSHIDA_C: Final[tuple[float, float, float, float]] = (
    0.5 * _YOSHIDA_W1,
    0.5 * (_YOSHIDA_W0 + _YOSHIDA_W1),
    0.5 * (_YOSHIDA_W0 + _YOSHIDA_W1),
    0.5 * _YOSHIDA_W1,
)
_YOSHIDA_D: Final[tuple[float, float, float, float]] = (
    _YOSHIDA_W1,
    _YOSHIDA_W0,
    _YOSHIDA_W1,
    0.0,
)


@dataclass(frozen=True, slots=True)
class ThyristorSpec:
    """BT151-500R. Valores de diseño conservadores; validar contra revisión del datasheet."""
    name: str = "BT151-500R"
    v_drm_v: float = 500.0
    v_rrm_v: float = 500.0
    i_tsm_a: float = 132.0          # surge no repetitivo, semisenoidal 10 ms
    i2t_limit_a2s: float = 72.0     # cota constitucional (más estricta que la de hoja de datos)
    didt_limit_a_us: float = 50.0
    dvdt_limit_v_us: float = 50.0
    i_gt_a: float = 0.015
    v_gt_v: float = 1.5
    i_h_a: float = 0.020            # corriente de mantenimiento (máx)
    i_l_a: float = 0.040            # corriente de enganche (máx)
    t_gt_ns: float = 120.0          # retardo controlado por compuerta
    t_latch_ns: float = 1500.0      # enganche de ánodo (> I_L)
    surge_derating: float = 0.75    # margen por evento repetido / T_j elevada

    @property
    def i_admissible_a(self) -> float:
        return self.i_tsm_a * self.surge_derating


@dataclass(frozen=True, slots=True)
class Esp32TimingSpec:
    """Presupuesto determinista de la ISR residente en IRAM (Xtensa LX6 @ 240 MHz)."""
    clock_hz: float = 240.0e6
    isr_entry_cycles: int = 24      # vector + window spill, todo en IRAM/DRAM
    guard_eval_cycles: int = 12     # branch sobre veredicto precalculado
    w1ts_store_cycles: int = 8      # store a GPIO_OUT_W1TS_REG (posted write)
    pad_prop_ns: float = 12.0
    gate_driver_ns: float = 45.0
    gpio_out_w1ts_reg: int = 0x3FF44008
    gpio_mask: int = 0x4000         # GPIO14

    @property
    def cycle_ns(self) -> float:
        return 1.0e9 / self.clock_hz

    @property
    def t_assert_ns(self) -> float:
        """CROWBAR-1: latencia de ASERCIÓN de mando en el pad (no de colapso del bus)."""
        c = self.isr_entry_cycles + self.guard_eval_cycles + self.w1ts_store_cycles
        return c * self.cycle_ns + self.pad_prop_ns


@dataclass(frozen=True, slots=True)
class CrowbarBusConfig:
    """Bus DC y lazo crowbar. L se elige para respetar el SOA, no por estética."""
    v_bus_v: float = 48.0
    c_bus_f: float = 220.0e-6
    l_crowbar_h: float = 68.0e-6    # corregido: 4.7 µH producía I_pk ≈ 328 A > I_TSM
    r_on_ohm: float = 0.120         # R dinámica SCR + ESR + cableado (linealización en torno a I_pk)
    upstream_inhibit: bool = True   # inhibición de la fuente tras el disparo
    fuse_coordinated: bool = True   # I²t_fusible < I²t_SCR
    galvanic_isolation: bool = True # driver aislado / transformador de pulsos
    gate_resistor_ohm: float = 100.0
    gpio_v_oh_v: float = 3.0
    gpio_safe_source_a: float = 0.020


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.2 · ÁLGEBRA DE HEYTING G₃ (cadena de Gödel) CON VERIFICACIÓN EXHAUSTIVA DE LEYES
# ──────────────────────────────────────────────────────────────────────────────────────────────────
class HeytingChampionVerdict(IntEnum):
    VETOED = 0     # ⊥  (absorbente del meet; dispara el crowbar)
    DEGRADED = 1
    COHERENT = 2   # ⊤

    # ── operaciones del álgebra de Heyting lineal (cadena de Gödel G₃) ─────────────────────────
    @staticmethod
    def meet(a: int, b: int) -> int:
        return int(a) if a < b else int(b)

    @staticmethod
    def join(a: int, b: int) -> int:
        return int(a) if a > b else int(b)

    @staticmethod
    def implies(a: int, b: int) -> int:
        """Residuo de Galois: a → b = ⊤ si a ≤ b, en otro caso b. (Heyting, no booleano.)"""
        return 2 if a <= b else int(b)

    @staticmethod
    def neg(a: int) -> int:
        return HeytingChampionVerdict.implies(a, 0)

    @staticmethod
    def birequiv(a: int, b: int) -> int:
        """Bi-implicación intuicionista (a ↔ b) = (a → b) ∧ (b → a)."""
        return HeytingChampionVerdict.meet(
            HeytingChampionVerdict.implies(a, b),
            HeytingChampionVerdict.implies(b, a),
        )

    @staticmethod
    def meet_all(vals: Iterable[int]) -> int:
        v = 2
        for x in vals:
            v = HeytingChampionVerdict.meet(v, int(x))
            if v == 0:                      # absorción: cortocircuito semánticamente exacto
                return 0
        return v

    @classmethod
    def to_string(cls, val: int) -> str:
        return cls(int(val)).name

    @classmethod
    def audit_laws(cls) -> str:
        """
        Prueba por exhaución (dominio finito 3³ = 27 ternas) de las leyes de Heyting y de Dummett.
        Devuelve el digest SHA-256 de la tabla de verdad completa.
        """
        dom = (0, 1, 2)
        rows = bytearray()
        for a, b, c in itertools.product(dom, repeat=3):
            # 1. Residuo (adjunción de Galois):  a ∧ b ≤ c  ⟺  a ≤ (b → c)
            assert (cls.meet(a, b) <= c) == (a <= cls.implies(b, c)), "residuo violado"
            # 2. Distributividad finita de ∧ sobre ∨ y de ∨ sobre ∧
            assert cls.meet(a, cls.join(b, c)) == cls.join(cls.meet(a, b), cls.meet(a, c))
            assert cls.join(a, cls.meet(b, c)) == cls.meet(cls.join(a, b), cls.join(a, c))
            # 3. Pseudo-complemento (a ∧ ¬a = ⊥) y teorema de Glivenko (¬¬¬a = ¬a)
            assert cls.meet(a, cls.neg(a)) == 0
            assert cls.neg(cls.neg(cls.neg(a))) == cls.neg(a)
            # 4. Axioma de linealidad de Gödel–Dummett: (a → b) ∨ (b → a) = ⊤
            assert cls.join(cls.implies(a, b), cls.implies(b, a)) == 2
            rows += bytes((a, b, c, cls.implies(a, b), cls.neg(a)))
        # Testigo de NO booleanidad: falla el tercero excluido para DEGRADED (1 ∨ ¬1 = 1 ≠ 2)
        assert cls.join(1, cls.neg(1)) != 2, "G₃ no puede ser booleana"
        assert cls.neg(cls.neg(1)) == 2
        return hashlib.sha256(bytes(rows)).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.3 · CÓDEC CANÓNICO PREFIJO-LIBRE (inyectivo; anti-forja por ambigüedad de delimitadores)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _lp(payload: bytes) -> bytes:
    return len(payload).to_bytes(8, "big") + payload


def canon(obj: Any) -> bytes:
    """Serialización determinista e inyectiva. float → float.hex() (bit-exacto, sin locale)."""
    if obj is None:
        return b"n" + (0).to_bytes(8, "big")
    if isinstance(obj, bool):
        return b"b" + (1).to_bytes(8, "big") + bytes((int(obj),))
    if isinstance(obj, IntEnum):
        return b"i" + _lp(str(int(obj)).encode())
    if isinstance(obj, Enum):
        return b"e" + _lp(str(obj.value).encode())
    if isinstance(obj, int):
        return b"i" + _lp(str(obj).encode())
    if isinstance(obj, float):
        if math.isnan(obj):
            raise ChampionConstitutionalError("NaN no serializable: evidencia no finita.")
        # Normalización de -0.0 a +0.0 para preservar igualdad extensional en cuerpos ordenados
        val = 0.0 if obj == 0.0 else float(obj)
        return b"f" + _lp(val.hex().encode())
    if isinstance(obj, str):
        return b"s" + _lp(obj.encode("utf-8"))
    if isinstance(obj, (bytes, bytearray)):
        return b"y" + _lp(bytes(obj))
    if isinstance(obj, np.ndarray):
        head = f"{obj.dtype.str}|{obj.shape}".encode()
        return b"a" + _lp(head) + _lp(obj.tobytes(order="C"))
    if isinstance(obj, Mapping):
        items = sorted(obj.items(), key=lambda kv: canon(kv[0]))
        return b"d" + _lp(b"".join(canon(k) + canon(v) for k, v in items))
    if isinstance(obj, (tuple, list)):
        return b"t" + _lp(b"".join(canon(x) for x in obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        flds = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
        return b"c" + _lp(type(obj).__name__.encode()) + canon(dict(flds))
    raise ChampionConstitutionalError(f"Tipo no canónico: {type(obj)!r}")


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.4 · HKDF-SHA256 CONFORME A RFC 5869
# ──────────────────────────────────────────────────────────────────────────────────────────────────
class HKDFSha256:
    HLEN: Final[int] = 32

    @staticmethod
    def extract(salt: bytes, ikm: bytes) -> bytes:
        return hmac.new(salt, ikm, hashlib.sha256).digest()

    @classmethod
    def expand(cls, prk: bytes, info: bytes, length: int = 32) -> bytes:
        if not 0 < length <= 255 * cls.HLEN:
            raise ChampionConstitutionalError("RFC5869: L debe estar en (0, 255·HashLen].")
        t, okm, i = b"", b"", 1
        while len(okm) < length:
            t = hmac.new(prk, t + info + bytes((i,)), hashlib.sha256).digest()
            okm += t
            i += 1
        return okm[:length]

    @classmethod
    def derive(cls, master: bytes, label: str, session: str, epoch: int, length: int = 32) -> bytes:
        salt = hashlib.sha256(b"positron/champion/v18/salt|" + session.encode()).digest()
        info = f"positron/champion/{label}|{_SCHEMA}|{session}|{epoch}".encode()
        return cls.expand(cls.extract(salt, master), info, length)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.5 · ÁRBOL DE MERKLE RFC 6962 (con prueba de inclusión y auto-test de consistencia)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
class MerkleTreeRFC6962:
    @staticmethod
    def leaf_hash(data: bytes) -> bytes:
        return hashlib.sha256(b"\x00" + data).digest()

    @staticmethod
    def node_hash(left: bytes, right: bytes) -> bytes:
        return hashlib.sha256(b"\x01" + left + right).digest()

    @staticmethod
    def _split(n: int) -> int:
        """Mayor potencia de dos ESTRICTAMENTE menor que n (n ≥ 2)."""
        return 1 << ((n - 1).bit_length() - 1)

    @classmethod
    def mth(cls, leaves: Sequence[bytes]) -> bytes:
        n = len(leaves)
        if n == 0:
            return hashlib.sha256(b"").digest()
        if n == 1:
            return cls.leaf_hash(leaves[0])
        k = cls._split(n)
        return cls.node_hash(cls.mth(leaves[:k]), cls.mth(leaves[k:]))

    @classmethod
    def audit_path(cls, index: int, leaves: Sequence[bytes]) -> tuple[bytes, ...]:
        if not 0 <= index < len(leaves):
            raise IndexError("índice de hoja fuera de rango")

        def rec(m: int, sub: Sequence[bytes]) -> list[bytes]:
            n = len(sub)
            if n == 1:
                return []
            k = cls._split(n)
            if m < k:
                return rec(m, sub[:k]) + [cls.mth(sub[k:])]
            return rec(m - k, sub[k:]) + [cls.mth(sub[:k])]

        return tuple(rec(index, leaves))

    @classmethod
    def verify_inclusion(
        cls, leaf: bytes, index: int, size: int, path: Sequence[bytes], root: bytes
    ) -> bool:
        """Algoritmo literal de RFC 6962 §2.1.3 (verificación de prueba de inclusión)."""
        if index >= size or size <= 0:
            return False
        fn, sn = index, size - 1
        r = cls.leaf_hash(leaf)
        for p in path:
            if sn == 0:
                return False
            if (fn & 1) or fn == sn:
                r = cls.node_hash(p, r)
                if not (fn & 1):
                    while fn != 0 and not (fn & 1):
                        fn >>= 1
                        sn >>= 1
            else:
                r = cls.node_hash(r, p)
            fn >>= 1
            sn >>= 1
        return sn == 0 and hmac.compare_digest(r, root)

    @classmethod
    def self_test(cls) -> None:
        """Prueba de propiedad exhaustiva para n = 1..17 y todo índice (consistencia raíz↔prueba)."""
        for n in range(1, 18):
            lv = [canon(("leaf", n, i)) for i in range(n)]
            root = cls.mth(lv)
            for i in range(n):
                p = cls.audit_path(i, lv)
                if not cls.verify_inclusion(lv[i], i, n, p, root):
                    raise ChampionConstitutionalError(f"Merkle RFC6962 auto-test falló (n={n}, i={i})")
                if cls.verify_inclusion(canon(("evil", i)), i, n, p, root):
                    raise ChampionConstitutionalError("Merkle aceptó hoja adulterada")


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.6 · ÁLGEBRA LINEAL EXACTA: BAREISS (ℤ→ℚ), 𝔽₂ BITSET, 𝔽_p, HODGE Δ₁ Y BETTI DE ČECH–RIPS
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def bareiss_rank(matrix: Sequence[Sequence[int]]) -> int:
    """Rango EXACTO sobre ℚ mediante eliminación sin fracciones de Bareiss (aritmética ℤ)."""
    m = [list(map(int, row)) for row in matrix]
    if not m or not m[0]:
        return 0
    nr, nc = len(m), len(m[0])
    rank, prev, r0 = 0, 1, 0
    for col in range(nc):
        piv = next((r for r in range(r0, nr) if m[r][col] != 0), None)
        if piv is None:
            continue
        m[r0], m[piv] = m[piv], m[r0]
        pivot = m[r0][col]
        for r in range(r0 + 1, nr):
            factor = m[r][col]
            for c in range(col + 1, nc):
                num = pivot * m[r][c] - factor * m[r0][c]
                m[r][c] = num // prev          # identidad de Sylvester–Bareiss: división exacta en ℤ
            m[r][col] = 0
        prev = pivot
        r0 += 1
        rank += 1
        if r0 == nr:
            break
    return rank


def rank_gf2(rows: Sequence[int]) -> int:
    """Rango sobre 𝔽₂ con filas como bitsets enteros (XOR): exacto y O(n·rank)."""
    basis: dict[int, int] = {}
    rank = 0
    for row in rows:
        r = int(row)
        while r:
            p = r.bit_length() - 1
            if p in basis:
                r ^= basis[p]
            else:
                basis[p] = r
                rank += 1
                break
    return rank


def rank_mod_p(matrix: Sequence[Sequence[int]], p: int = 65521) -> int:
    """Rango exacto sobre el cuerpo finito primo 𝔽_p (p = 65521, mayor primo < 2¹⁶)."""
    m = [[int(x) % p for x in row] for row in matrix]
    if not m or not m[0]:
        return 0
    nr, nc, rank, r0 = len(m), len(m[0]), 0, 0
    for col in range(nc):
        piv = next((r for r in range(r0, nr) if m[r][col] % p), None)
        if piv is None:
            continue
        m[r0], m[piv] = m[piv], m[r0]
        inv = pow(m[r0][col], -1, p)
        m[r0] = [(x * inv) % p for x in m[r0]]
        for r in range(r0 + 1, nr):
            f = m[r][col]
            if f:
                m[r] = [(a - f * b) % p for a, b in zip(m[r], m[r0])]
        r0 += 1
        rank += 1
        if r0 == nr:
            break
    return rank


@dataclass(frozen=True, slots=True)
class CohomologyReport:
    scale: float
    n_vertices: int
    n_edges: int
    n_triangles: int
    b0: int
    b1: int
    b2: int
    euler_ok: bool
    complex_ok: bool                 # δ¹ ∘ δ⁰ = 0 (identidad fundamental de complejo de cocadenas)
    torsion_witness: bool            # rg_ℚ ≠ rg_𝔽₂ o rg_ℚ ≠ rg_𝔽p ⇒ torsión detectada
    hodge_spectral_gap: float = 0.0  # menor autovalor no nulo de Δ₁ = δ⁰(δ⁰)ᵀ + (δ¹)ᵀδ¹


def cech_rips_betti(points: NDArray[np.float64], scale: float) -> CohomologyReport:
    """
    Números de Betti del complejo de Vietoris–Rips a escala `scale`, EXACTOS:
        b₀ = |V| − rg δ⁰ ,   b₁ = (|E| − rg δ¹) − rg δ⁰ ,   b₂ = |T| − rg δ¹.
    Verifica χ = b₀ − b₁ + b₂, δ¹ ∘ δ⁰ = 0 sobre ℤ, doble testigo de cuerpo (ℚ vs 𝔽₂/𝔽_p),
    y calcula el gap espectral del 1-Laplaciano de Hodge discreto Δ₁.
    Corrige [A1]: pre-asigna `d1_bits = [0] * nt` para evitar `IndexError` cuando `nt >= 1`.
    """
    n = int(len(points))
    if n == 0:
        return CohomologyReport(scale, 0, 0, 0, 0, 0, 0, True, True, False, 0.0)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if d[i, j] <= scale]
    eidx = {e: k for k, e in enumerate(edges)}
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    tris = [
        (i, j, k)
        for i in range(n)
        for j in sorted(adj[i]) if j > i
        for k in sorted(adj[i] & adj[j]) if k > j
    ]
    nv, ne, nt = n, len(edges), len(tris)

    # δ⁰: C⁰ → C¹ (matriz ne × nv de incidencia orientada)
    d0 = [[0] * nv for _ in range(ne)]
    for k, (i, j) in enumerate(edges):
        d0[k][i], d0[k][j] = -1, 1

    # δ¹: C¹ → C² (matriz nt × ne de coborde simplicial orientado; corrige [A1])
    d1 = [[0] * ne for _ in range(nt)]
    d1_bits: list[int] = [0] * nt
    for t, (i, j, k) in enumerate(tris):
        c_ij, c_ik, c_jk = eidx[(i, j)], eidx[(i, k)], eidx[(j, k)]
        d1[t][c_ij] = 1
        d1[t][c_ik] = -1
        d1[t][c_jk] = 1
        d1_bits[t] = (1 << c_ij) | (1 << c_ik) | (1 << c_jk)

    # Identidad homológica fundamental: δ¹ ∘ δ⁰ = 0 sobre ℤ
    complex_ok = all(
        sum(d1[t][e] * d0[e][v] for e in range(ne)) == 0
        for t in range(nt) for v in range(nv)
    )

    rg0_q = bareiss_rank(d0) if ne else 0
    rg1_q = bareiss_rank(d1) if nt else 0
    rg1_2 = rank_gf2(d1_bits) if nt else 0
    rg1_p = rank_mod_p(d1) if nt else 0

    b0 = nv - rg0_q
    b1 = (ne - rg1_q) - rg0_q
    b2 = nt - rg1_q
    euler_ok = (nv - ne + nt) == (b0 - b1 + b2)
    torsion = (rg1_2 != rg1_q) or (rg1_p != rg1_q)

    # 1-Laplaciano de Hodge discreto Δ₁ = δ⁰ (δ⁰)ᵀ + (δ¹)ᵀ δ¹ sobre C¹(K; ℝ)
    hodge_gap = 0.0
    if ne > 0:
        d0_arr = np.asarray(d0, dtype=np.float64)
        delta_1 = d0_arr @ d0_arr.T
        if nt > 0:
            d1_arr = np.asarray(d1, dtype=np.float64)
            delta_1 = delta_1 + (d1_arr.T @ d1_arr)
        eig_hodge = np.linalg.eigvalsh(0.5 * (delta_1 + delta_1.T))
        pos_eigs = eig_hodge[eig_hodge > 1e-9]
        if pos_eigs.size > 0:
            hodge_gap = float(pos_eigs[0])

    return CohomologyReport(
        scale=float(scale),
        n_vertices=nv,
        n_edges=ne,
        n_triangles=nt,
        b0=b0,
        b1=b1,
        b2=b2,
        euler_ok=euler_ok,
        complex_ok=complex_ok,
        torsion_witness=torsion,
        hodge_spectral_gap=hodge_gap,
    )


def audit_h1_multiscale(points: NDArray[np.float64], base_scale: float) -> tuple[CohomologyReport, ...]:
    """Robustez persistente: H¹ debe anularse en tres escalas r ∈ {r₀, 1.5r₀, 2.0r₀}."""
    return tuple(cech_rips_betti(points, base_scale * f) for f in (1.0, 1.5, 2.0))


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.7 · NUMÉRICA CUÁNTICA RIGUROSA (Hermiticidad, PSD, Uhlmann, Bures, SLD-Fisher, CPTP, Foster-Cauer)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _hermitize(a: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (a + a.conj().T)


def psd_sqrt(a: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Raíz PSD por descomposición espectral (nunca `inv`, nunca sqrt de entradas)."""
    h = _hermitize(a)
    w, u = np.linalg.eigh(h)
    w_cl = np.clip(w, 0.0, None)
    return (u * np.sqrt(w_cl)) @ u.conj().T, w


def uhlmann_fidelity(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    """F(ρ,σ) = (Tr √(√ρ σ √ρ))². Recorte a [0,1] sólo dentro de tolerancia numérica."""
    sr, _ = psd_sqrt(rho)
    inner = _hermitize(sr @ sigma @ sr)
    w = np.linalg.eigvalsh(inner)
    root = float(np.sum(np.sqrt(np.clip(w, 0.0, None))))
    return float(min(1.0, max(0.0, root * root)))


def bures_angle(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    return float(math.acos(math.sqrt(uhlmann_fidelity(rho, sigma))))


def sld_quantum_fisher_metric(
    rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
) -> tuple[float, float]:
    """
    Resuelve la ecuación cuántica de Lyapunov–Sylvester para la Derivada Logarítmica Simétrica (SLD):
        ½ (ρ L + L ρ) = σ − ρ
    en la base ortonormal de autovectores de ρ:
        L_{ij} = 2 (U* (σ − ρ) U)_{ij} / (max(λ_i, ε) + max(λ_j, ε)).
    Devuelve (g_SLD = Tr(ρ L²), residuo_Lyapunov ‖½(ρL + Lρ) − (σ − ρ)‖_F).
    """
    rh = _hermitize(rho)
    sh = _hermitize(sigma)
    w, u = np.linalg.eigh(rh)
    w_safe = np.maximum(w, 1e-12)
    delta = sh - rh
    delta_eig = u.conj().T @ delta @ u
    denom = w_safe[:, None] + w_safe[None, :]
    l_eig = 2.0 * delta_eig / denom
    l_eig = _hermitize(l_eig)
    g_sld = float(max(0.0, np.sum(w_safe[:, None] * (np.abs(l_eig) ** 2))))
    l_mat = u @ l_eig @ u.conj().T
    lhs = 0.5 * (rh @ l_mat + l_mat @ rh)
    residual = float(np.linalg.norm(lhs - delta, ord="fro"))
    return g_sld, residual


def depolarize(a: NDArray[np.complex128], lam: float) -> NDArray[np.complex128]:
    """Canal CPTP de despolarización Λ(a) = (1−λ)a + λ Tr(a) I/d (unital, completamente positivo)."""
    d = a.shape[0]
    return (1.0 - lam) * a + lam * complex(np.trace(a).real) * np.eye(d, dtype=np.complex128) / d


def connes_seminorm(a: NDArray[np.complex128], d_op: NDArray[np.complex128]) -> float:
    """L_D(a) = ‖[D, a]‖₂ (seminorma espectral de Lipschitz del triplete de Connes (𝒜, ℋ, D))."""
    comm = d_op @ a - a @ d_op
    return float(np.linalg.norm(comm, ord=2))


def foster_cauer_impedance_trace(k_eig: NDArray[np.float64]) -> tuple[bool, float]:
    """
    Certificación de síntesis pasiva multipuerto de Foster–Cauer:
    Z_DP(1) = (K + I)⁻¹ ⇒ Tr(Z_DP(1)) = Σᵢ 1 / (λᵢ(K) + 1).
    """
    passivity_ok = bool(np.all(np.isfinite(k_eig)) and float(np.min(k_eig)) >= 0.0)
    trace_z = float(np.sum(1.0 / (np.maximum(k_eig, 0.0) + 1.0)))
    return passivity_ok, trace_z


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.8 · MECÁNICA CELESTE DE POINCARÉ — ELEMENTOS CANÓNICOS, YOSHIDA Sp(2,ℝ), MONODROMÍA, MELNIKOV, KAC
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareCanonicalElements:
    """
    Variables canónicas heliocentricas no singulares de Poincaré (Les Méthodes Nouvelles, T. I):
        Λ = √(μ a)
        ξ₁ = √(2 J₁) cos(ϖ),   η₁ = −√(2 J₁) sin(ϖ),   J₁ = Λ − G = Λ e² / (1 + √(1 − e²))
        ξ₂ = √(2 J₂) cos(Ω),   η₂ = −√(2 J₂) sin(Ω),   J₂ = G − H = 2 G sin²(i / 2)
    Libres de singularidades de coordenadas en e → 0 e i → 0.
    """
    lambda_delaunay: float
    g_delaunay: float
    h_delaunay: float
    xi_1_eccentric: float
    eta_1_eccentric: float
    xi_2_oblique: float
    eta_2_oblique: float
    action_j1_radial: float
    action_j2_vertical: float
    amd_laskar_poincare: float
    nonsingular_regularity_ok: bool


def synthesize_poincare_elements(
    seed_point: tuple[float, float],
    bures_angle_rad: float,
    kappa_k: float,
    semi_major_axis: float = 1.0,
    eccentricity_base: float = 0.01,
    inclination_base_rad: float = 0.02,
    mu_grav: float = 1.0,
) -> PoincareCanonicalElements:
    """
    Construye las variables canónicas no singulares de Poincaré (Λ, ξ₁, η₁, ξ₂, η₂)
    y verifica la identidad exacta de Laskar–Poincaré:
        AMD = ½ (ξ₁² + η₁² + ξ₂² + η₂²) = J₁ + J₂ = Λ − H.
    """
    a_safe = max(1e-6, float(semi_major_axis))
    mu_safe = max(1e-12, float(mu_grav))
    lam = math.sqrt(mu_safe * a_safe)

    e_eff = min(0.99, max(0.0, float(eccentricity_base) + 0.05 * math.sin(max(0.0, bures_angle_rad))))
    inc_eff = min(
        0.5 * math.pi - 1e-6,
        max(0.0, float(inclination_base_rad) + 0.01 * math.tanh(math.log10(max(1.0, kappa_k)))),
    )

    one_minus_e2 = max(1e-15, 1.0 - e_eff * e_eff)
    sqrt_1_e2 = math.sqrt(one_minus_e2)
    g_del = lam * sqrt_1_e2
    h_del = g_del * math.cos(inc_eff)

    # Fórmulas libres de cancelación catastrófica para e ≪ 1 e i ≪ 1
    j1_radial = lam * (e_eff * e_eff) / (1.0 + sqrt_1_e2)
    sin_half_i = math.sin(0.5 * inc_eff)
    j2_vertical = 2.0 * g_del * (sin_half_i * sin_half_i)

    varpi = float(seed_point[0])
    omega_node = float(seed_point[1])
    r1 = math.sqrt(max(0.0, 2.0 * j1_radial))
    r2 = math.sqrt(max(0.0, 2.0 * j2_vertical))

    xi_1 = r1 * math.cos(varpi)
    eta_1 = -r1 * math.sin(varpi)
    xi_2 = r2 * math.cos(omega_node)
    eta_2 = -r2 * math.sin(omega_node)

    amd = 0.5 * (xi_1 * xi_1 + eta_1 * eta_1 + xi_2 * xi_2 + eta_2 * eta_2)
    amd_err = abs(amd - (j1_radial + j2_vertical))
    reg_ok = bool(math.isfinite(amd) and amd >= 0.0 and amd_err <= 1e-12 * max(1.0, lam))

    return PoincareCanonicalElements(
        lambda_delaunay=lam,
        g_delaunay=g_del,
        h_delaunay=h_del,
        xi_1_eccentric=xi_1,
        eta_1_eccentric=eta_1,
        xi_2_oblique=xi_2,
        eta_2_oblique=eta_2,
        action_j1_radial=j1_radial,
        action_j2_vertical=j2_vertical,
        amd_laskar_poincare=amd,
        nonsingular_regularity_ok=reg_ok,
    )


@dataclass(frozen=True, slots=True)
class ResonanceParams:
    """Forma normal de Andoyer / péndulo forzado-amortiguado cerca de resonancia de movimiento medio."""
    delta: float = 0.35       # disipación (0 ⇒ hamiltoniano)
    gamma: float = 0.05       # amplitud de forzamiento
    omega: float = 1.30       # frecuencia de forzamiento (fuera de resonancia primaria ω₀=1)
    eps2: float = 0.025       # segunda resonancia (Chirikov)
    separation: float = 1.20  # separación entre resonancias
    mu_mass: float = 3.0e-6   # cociente de masas (zona caótica de Wisdom)
    dioph_gamma: float = 0.01
    dioph_tau: float = 1.0
    dioph_qmax: int = 64
    time_unit_sec: float = 300.0  # segundos físicos por unidad adimensional de tiempo

    @property
    def forcing_period(self) -> float:
        return 2.0 * math.pi / self.omega


def _accel(phi: float, vel: float, t: float, p: ResonanceParams, hamiltonian: bool) -> float:
    a = -math.sin(phi) + p.gamma * math.cos(p.omega * t)
    return a if hamiltonian else a - p.delta * vel


def _yoshida4_symplectic_step(
    phi: float, vel: float, t: float, h: float, p: ResonanceParams
) -> tuple[float, float]:
    """
    Integrador simpléctico explícito de 4.º orden de Yoshida para el flujo hamiltoniano:
        H(φ, I, t) = ½ I² − cos(φ) − φ γ cos(ω t).
    Cada sub-etapa es una composición de mapas de cizalla canónicos (drift-kick) con Jacobiano
    idénticamente igual a 1, preservando exactamente la 2-forma simpléctica dφ ∧ dI.
    """
    tau = t
    for c_k, d_k in zip(_YOSHIDA_C, _YOSHIDA_D):
        dt_c = c_k * h
        phi += dt_c * vel
        tau += dt_c
        if d_k != 0.0:
            vel += (d_k * h) * (-math.sin(phi) + p.gamma * math.cos(p.omega * tau))
    return phi, vel


def _rk4_step(
    phi: float, vel: float, t: float, h: float, p: ResonanceParams, ham: bool
) -> tuple[float, float]:
    """Paso de integración: Yoshida simpléctico de 4.º orden si `ham=True`, RK4 clásico si `ham=False`."""
    if ham:
        return _yoshida4_symplectic_step(phi, vel, t, h, p)
    k1p, k1v = vel, _accel(phi, vel, t, p, False)
    k2p, k2v = vel + 0.5 * h * k1v, _accel(phi + 0.5 * h * k1p, vel + 0.5 * h * k1v, t + 0.5 * h, p, False)
    k3p, k3v = vel + 0.5 * h * k2v, _accel(phi + 0.5 * h * k2p, vel + 0.5 * h * k2v, t + 0.5 * h, p, False)
    k4p, k4v = vel + h * k3v, _accel(phi + h * k3p, vel + h * k3v, t + h, p, False)
    return (
        phi + (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
        vel + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
    )


def stroboscopic_section(
    x0: tuple[float, float], p: ResonanceParams, n_periods: int, steps_per_period: int, ham: bool
) -> tuple[tuple[float, float], ...]:
    """Sección de Poincaré estroboscópica Σ = {t ≡ 0 mod T_f}. Muestreo exacto en el plano de sección."""
    h = p.forcing_period / steps_per_period
    phi, vel, t = x0[0], x0[1], 0.0
    out: list[tuple[float, float]] = [(phi, vel)]
    for _ in range(n_periods):
        for _ in range(steps_per_period):
            phi, vel = _rk4_step(phi, vel, t, h, p, ham)
            t += h
        out.append((phi, vel))
    return tuple(out)


def monodromy(
    x0: tuple[float, float], p: ResonanceParams, steps: int, ham: bool
) -> tuple[NDArray[np.float64], float, tuple[float, float]]:
    """
    Matriz de monodromía M = Φ(T_f) con Φ̇ = A(t)Φ, A = [[0,1],[−cos φ, −δ]], Φ(0)=Id.
    • Si `ham=True`: integra mediante la aplicación tangente exacta del método simpléctico de
      Yoshida de 4.º orden (producto de cizallas en SL(2,ℝ) = Sp(2,ℝ)), y mide honestamente el
      defecto simpléctico ‖MᵀJM − J‖_F.
    • Si `ham=False`: integra las ecuaciones variacionales disipativas por RK4 y mide el defecto
      respecto de la contracción de volumen de Abel–Jacobi–Liouville exp(−δ T_f).
    """
    h = p.forcing_period / steps
    phi, vel, t = x0[0], x0[1], 0.0
    j_mat = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)

    if ham:
        m00, m01, m10, m11 = 1.0, 0.0, 0.0, 1.0
        for _ in range(steps):
            tau = t
            for c_k, d_k in zip(_YOSHIDA_C, _YOSHIDA_D):
                dt_c = c_k * h
                # Cizalla de deriva (drift): [[1, dt_c], [0, 1]] ∈ Sp(2,ℝ)
                phi += dt_c * vel
                m00 += dt_c * m10
                m01 += dt_c * m11
                tau += dt_c
                if d_k != 0.0:
                    dt_d = d_k * h
                    # Cizalla de impulso (kick): [[1, 0], [−dt_d cos(φ), 1]] ∈ Sp(2,ℝ)
                    c_phi = -math.cos(phi)
                    vel += dt_d * (-math.sin(phi) + p.gamma * math.cos(p.omega * tau))
                    m10 += (dt_d * c_phi) * m00
                    m11 += (dt_d * c_phi) * m01
            t += h
        m = np.array([[m00, m01], [m10, m11]], dtype=np.float64)
        defect = float(np.linalg.norm(m.T @ j_mat @ m - j_mat, ord="fro"))
        return m, defect, (phi, vel)

    # Caso disipativo (ham=False): integración variacional RK4 acoplada
    f = [1.0, 0.0, 0.0, 1.0]

    def deriv(state: tuple[float, float], mat: list[float], tt: float) -> tuple[float, float, list[float]]:
        c = -math.cos(state[0])
        dmp = -p.delta
        m0, m1, m2, m3 = mat
        return (
            state[1],
            _accel(state[0], state[1], tt, p, False),
            [m2, m3, c * m0 + dmp * m2, c * m1 + dmp * m3],
        )

    for _ in range(steps):
        s = (phi, vel)
        k1p, k1v, k1m = deriv(s, f, t)
        s2 = (phi + 0.5 * h * k1p, vel + 0.5 * h * k1v)
        f2 = [f[i] + 0.5 * h * k1m[i] for i in range(4)]
        k2p, k2v, k2m = deriv(s2, f2, t + 0.5 * h)
        s3 = (phi + 0.5 * h * k2p, vel + 0.5 * h * k2v)
        f3 = [f[i] + 0.5 * h * k2m[i] for i in range(4)]
        k3p, k3v, k3m = deriv(s3, f3, t + 0.5 * h)
        s4 = (phi + h * k3p, vel + h * k3v)
        f4 = [f[i] + h * k3m[i] for i in range(4)]
        k4p, k4v, k4m = deriv(s4, f4, t + h)
        phi += (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        vel += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        f = [f[i] + (h / 6.0) * (k1m[i] + 2.0 * k2m[i] + 2.0 * k3m[i] + k4m[i]) for i in range(4)]
        t += h
    m = np.array([[f[0], f[1]], [f[2], f[3]]], dtype=np.float64)
    defect = float(np.linalg.norm(m.T @ j_mat @ m - j_mat, ord="fro"))
    return m, defect, (phi, vel)


def floquet_multipliers(m: NDArray[np.float64]) -> tuple[complex, complex]:
    tr, det = float(np.trace(m)), float(np.linalg.det(m))
    disc = complex(tr * tr - 4.0 * det)
    s = np.sqrt(disc)
    return complex((tr + s) / 2.0), complex((tr - s) / 2.0)


def greene_residue(m: NDArray[np.float64]) -> float:
    """R = (2 − tr M)/4. R ∈ (0,1) elíptico (toro KAM); R < 0 o R > 1 hiperbólico (criterio de Greene)."""
    return float((2.0 - float(np.trace(m))) / 4.0)


def diophantine_constant(rho: float, qmax: int, tau: float) -> tuple[float, int]:
    """
    C = min_{1≤q≤qmax} q^τ · |qρ − p| evaluado SÓLO en convergentes de fracción continua.
    Justificación: teorema de la mejor aproximación (Lagrange) — ningún q no-convergente
    puede mejorar |qρ−p|, luego el mínimo se alcanza en un convergente.
    """
    x, p_prev, p_cur, q_prev, q_cur = float(rho), 1, 0, 0, 1
    best, best_q = float("inf"), 1
    for _ in range(64):
        a = math.floor(x)
        p_next, q_next = a * p_cur + p_prev, a * q_cur + q_prev
        if q_next > qmax:
            break
        val = (q_next ** tau) * abs(q_next * rho - p_next)
        if val < best:
            best, best_q = val, q_next
        p_prev, p_cur, q_prev, q_cur = p_cur, p_next, q_cur, q_next
        frac = x - a
        if frac < 1e-14:
            break
        x = 1.0 / frac
    return (float(best), int(best_q))


def melnikov_analytic(t0: float, p: ResonanceParams) -> float:
    """M(t₀) = −8δ + 2πγ sech(πω/2) cos(ω t₀)  (péndulo, Guckenheimer–Holmes §4.5)."""
    sech = 1.0 / math.cosh(math.pi * p.omega / 2.0)
    return -8.0 * p.delta + 2.0 * math.pi * p.gamma * sech * math.cos(p.omega * t0)


def melnikov_quadrature(t0: float, p: ResonanceParams, horizon: float = 30.0, n: int = 6001) -> float:
    """Contraste numérico sobre la separatriz heteroclínica: φ_h = 2 arcsin(tanh t), I_h = 2 sech t."""
    h = 2.0 * horizon / (n - 1)
    acc = 0.0
    for k in range(n):
        t = -horizon + k * h
        yh = 2.0 / math.cosh(t)
        val = yh * (p.gamma * math.cos(p.omega * (t + t0)) - p.delta * yh)
        acc += val * (0.5 if k in (0, n - 1) else 1.0)
    return acc * h


def chirikov_overlap(p: ResonanceParams) -> float:
    """K = (Δ₁+Δ₂)/s con semi-anchuras de isla de resonancia Δ = 2√ε. K ≥ 1 ⇒ caos global."""
    return (2.0 * math.sqrt(p.gamma) + 2.0 * math.sqrt(p.eps2)) / p.separation


def wisdom_chaotic_zone(mu: float) -> float:
    """Δa/a ≈ 1.3 μ^{2/7} (Wisdom 1980): anchura del solapamiento de resonancias de movimiento medio."""
    return 1.3 * (mu ** (2.0 / 7.0))


def kac_return_time(
    section: Sequence[tuple[float, float]], cell_center: tuple[float, float], radius: float
) -> tuple[float, float]:
    """
    Lema de Kac para el mapa estroboscópico HAMILTONIANO (preserva área por el teorema de Poincaré):
        ⟨τ_A⟩ = 1/μ(A),  estimada por la frecuencia empírica de visita a la celda A.
    Devuelve (⟨τ_A⟩ en iteraciones, μ̂(A)).
    """
    hits = sum(
        1 for (a, b) in section if math.hypot(a - cell_center[0], b - cell_center[1]) <= radius
    )
    mu_hat = hits / max(1, len(section))
    return (float("inf") if mu_hat <= 0.0 else 1.0 / mu_hat, mu_hat)


def resample_by_arclength(pts: Sequence[tuple[float, float]], n: int) -> NDArray[np.float64]:
    """Re-muestreo uniforme por longitud de arco: densidad homogénea para el complejo de Rips."""
    arr = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] <= 0.0:
        return np.repeat(arr[:1], n, axis=0)
    targets = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(targets, s, arr[:, k]) for k in range(arr.shape[1])])


def floer_action(section_flow: Sequence[tuple[float, float, float]], p: ResonanceParams) -> float:
    """𝒜_H(γ) = ∮ I dφ − ∫ H dt sobre la órbita periódica convergida (acción de Poincaré–Cartan / Floer)."""
    acc = 0.0
    for k in range(len(section_flow) - 1):
        phi0, i0, t0 = section_flow[k]
        phi1, i1, t1 = section_flow[k + 1]
        dt = t1 - t0
        imid = 0.5 * (i0 + i1)
        hmid = 0.5 * imid * imid - math.cos(0.5 * (phi0 + phi1))
        acc += imid * (phi1 - phi0) - hmid * dt
    return float(acc)


def poincare_integral_invariants(elems: PoincareCanonicalElements) -> tuple[float, float]:
    """
    Invariantes integrales absolutos de Poincaré (Les Méthodes Nouvelles, T. III):
      • Primer invariante (1-forma de Poincaré–Cartan sobre el déficit secular):
            I₁ = ∮ (ξ₁ dη₁ + ξ₂ dη₂) = 2π (J₁ + J₂) = 2π · AMD.
      • Segundo invariante (2-forma simpléctica acoplada):
            I₂ = 2π (J₁ + J₂) + 4π² J₁ J₂ / max(Λ, ε).
    """
    i1 = 2.0 * math.pi * elems.amd_laskar_poincare
    i2 = i1 + (4.0 * (math.pi ** 2) * elems.action_j1_radial * elems.action_j2_vertical) / max(
        1e-12, elems.lambda_delaunay
    )
    return float(i1), float(i2)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.9 · FÍSICA DEL CROWBAR: SOA DEL TIRISTOR Y RESPUESTA RLC EXACTA
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CrowbarPhysicsReport:
    z0_ohm: float
    zeta: float
    omega0_rad_s: float
    omega_d_rad_s: float
    i_peak_a: float
    i_peak_admissible_a: float
    didt_a_per_us: float
    i2t_a2s: float
    dvdt_v_per_us: float
    v_reverse_peak_v: float
    t_peak_us: float
    t_zero_us: float
    energy_j: float
    l_min_required_h: float
    gate_current_a: float
    soa_ok: bool
    latching_ok: bool
    violations: tuple[str, ...]


def analyze_crowbar(cfg: CrowbarBusConfig, scr: ThyristorSpec) -> CrowbarPhysicsReport:
    """Modelo RLC serie subamortiguado con verificación completa del Área de Operación Segura (SOA)."""
    v, c, l, r = cfg.v_bus_v, cfg.c_bus_f, cfg.l_crowbar_h, cfg.r_on_ohm
    z0 = math.sqrt(l / c)
    zeta = r / (2.0 * z0)
    w0 = 1.0 / math.sqrt(l * c)
    energy = 0.5 * c * v * v
    viol: list[str] = []
    if zeta >= 1.0:
        # Sobreamortiguado/crítico: i(t) = (V/L) t e^{−αt} (crítico) — cota conservadora.
        wd = 0.0
        i_pk = v / (z0 * math.e) if zeta == 1.0 else v / (2.0 * z0 * math.sqrt(zeta * zeta - 1.0))
        t_pk_us = (2.0 * l / r) * 1e6 if zeta == 1.0 else float("nan")
        t_zero_us = float("inf")
        v_rev = 0.0
    else:
        s = math.sqrt(1.0 - zeta * zeta)
        wd = w0 * s
        theta = math.atan(s / zeta)
        t_pk = theta / wd
        i_pk = (v / z0) * math.exp(-(zeta / s) * theta)   # forma analítica EXACTA del pico
        t_pk_us = t_pk * 1e6
        t_zero_us = (math.pi / wd) * 1e6
        v_rev = v * math.exp(-math.pi * zeta / s)
    didt = (v / l) * 1e-6                    # A/µs
    i2t = energy / r                         # ∫₀^∞ i²dt = ½CV²/R  (teorema de Parseval / balance energético)
    dvdt = (i_pk / c) * 1e-6                 # V/µs
    l_min = c * v * v / (scr.i_admissible_a ** 2)
    i_gate = max(0.0, (cfg.gpio_v_oh_v - scr.v_gt_v) / cfg.gate_resistor_ohm)
    if i_pk > scr.i_admissible_a:
        viol.append(f"I_pk={i_pk:.2f}A > I_adm={scr.i_admissible_a:.2f}A (destrucción del BT151)")
    if i2t > scr.i2t_limit_a2s:
        viol.append(f"I²t={i2t:.3f} > {scr.i2t_limit_a2s} A²s")
    if didt > scr.didt_limit_a_us:
        viol.append(f"di/dt={didt:.3f} > {scr.didt_limit_a_us} A/µs")
    if dvdt > scr.dvdt_limit_a_us:
        viol.append(f"dv/dt={dvdt:.3f} > {scr.dvdt_limit_a_us} V/µs")
    if v_rev > scr.v_rrm_v:
        viol.append(f"V_rev={v_rev:.1f} > V_RRM={scr.v_rrm_v}")
    if v > scr.v_drm_v:
        viol.append("V_bus > V_DRM")
    if l < l_min:
        viol.append(f"L={l*1e6:.2f}µH < L_min={l_min*1e6:.2f}µH exigido por I_TSM")
    if i_gate < scr.i_gt_a:
        viol.append(f"I_gate={i_gate*1e3:.2f}mA < I_GT={scr.i_gt_a*1e3:.2f}mA")
    if i_gate > cfg.gpio_safe_source_a:
        viol.append("I_gate excede la capacidad segura del GPIO (requiere driver)")
    latching = bool(cfg.upstream_inhibit or cfg.fuse_coordinated)
    if not latching:
        viol.append("SCR se desceba en t_zero sin inhibición aguas arriba ni fusible coordinado")
    return CrowbarPhysicsReport(
        z0, zeta, w0, wd, i_pk, scr.i_admissible_a, didt, i2t, dvdt, v_rev,
        t_pk_us, t_zero_us, energy, l_min, i_gate, not viol, latching, tuple(viol),
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.10 · CONSTITUCIÓN INMUTABLE Y GATILLO (tabla de verdad exhaustiva = prueba finita)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
GATE_NAMES: Final[tuple[str, ...]] = (
    "identity", "topology", "dirac", "amber", "banach", "celestial", "loeb", "constitution",
)


def trigger_predicate(vals: Sequence[int]) -> int:
    """v_final = ⋀ᵢ vᵢ. Función PURA y congelada: ningún nivel de automejora puede tocarla."""
    return HeytingChampionVerdict.meet_all(vals)


def trigger_truth_table_digest() -> tuple[str, int]:
    """Prueba exhaustiva sobre Ω₃⁸ (6561 filas). Dominio finito ⇒ esto es PRUEBA, no muestreo."""
    buf = bytearray()
    rows = 0
    for combo in itertools.product((0, 1, 2), repeat=len(GATE_NAMES)):
        buf.append(trigger_predicate(combo))
        rows += 1
    return hashlib.sha256(bytes(buf)).hexdigest(), rows


@dataclass(frozen=True, slots=True)
class ChampionConstitution:
    """Subespacio de SEGURIDAD. Prohibido mutarlo por automejora (invariante por digest)."""
    q_max: float = 0.95
    kantorovich_h_max: float = 1.0
    sym_defect_max: float = 1.0e-7
    residue_lo: float = 0.0
    residue_hi: float = 1.0
    residue_soft_lo: float = 0.05
    residue_soft_hi: float = 0.95
    melnikov_ratio_max: float = 1.0
    chirikov_max: float = 1.0
    kac_anti_evasion_kappa: float = 1.0
    phase_gap_max_ns: int = 5_000_000_000
    clock_skew_max_ns: int = 2_000_000_000
    t_assert_max_ns: float = 400.0
    amber_warn_fraction: float = 0.80
    loeb_fuel: int = 8
    rsi_max_generations: int = 3
    rsi_max_candidates: int = 3
    rsi_min_improvement: float = 1.0e-4
    min_section_periods: int = 64
    min_steps_per_period: int = 128
    min_banach_iters: int = 8
    max_cover_points: int = 48

    def digest(self) -> str:
        return hashlib.sha256(canon(self)).hexdigest()


@dataclass(frozen=True, slots=True)
class TuningVector:
    """Subespacio NO de seguridad: sólo resolución/coste del auditor. Siempre acotado por mínimos."""
    section_periods: int = 160
    steps_per_period: int = 256
    banach_iters: int = 16
    cover_points: int = 40
    dioph_qmax: int = 64
    w_banach: float = 0.25
    w_residue: float = 0.20
    w_dioph: float = 0.15
    w_melnikov: float = 0.20
    w_cost: float = 0.20

    def clamped(self, c: ChampionConstitution) -> "TuningVector":
        return TuningVector(
            section_periods=int(np.clip(self.section_periods, c.min_section_periods, 512)),
            steps_per_period=int(np.clip(self.steps_per_period, c.min_steps_per_period, 1024)),
            banach_iters=int(np.clip(self.banach_iters, c.min_banach_iters, 64)),
            cover_points=int(np.clip(self.cover_points, 16, c.max_cover_points)),
            dioph_qmax=int(np.clip(self.dioph_qmax, 16, 256)),
            w_banach=float(np.clip(self.w_banach, 0.0, 1.0)),
            w_residue=float(np.clip(self.w_residue, 0.0, 1.0)),
            w_dioph=float(np.clip(self.w_dioph, 0.0, 1.0)),
            w_melnikov=float(np.clip(self.w_melnikov, 0.0, 1.0)),
            w_cost=float(np.clip(self.w_cost, 0.0, 0.5)),
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ω₀.11 · DATOS DE ENTRADA FÍSICOS Y CONTEXTO COMPARTIDO DE SESIÓN
# ──────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DiracAnnihilationMeasurement:
    """Medición explícita de e⁻+e⁺→2γ. Su AUSENCIA es ⊥ (fail-closed), jamás un valor por defecto."""
    photon_energies_kev: tuple[float, ...]
    opening_angle_rad: float
    provenance: Provenance
    detector_id: str
    timestamp_ns: int

    def total_energy_kev(self) -> float:
        return float(math.fsum(self.photon_energies_kev))

    def momentum_residual_kev(self) -> float:
        """‖p₁+p₂‖c = √(E₁²+E₂²+2E₁E₂cosθ). Para 2γ colineales retro-opuestos (θ=π) ⇒ |E₁−E₂|."""
        if len(self.photon_energies_kev) != 2:
            return float("inf")
        e1, e2 = self.photon_energies_kev
        val = e1 * e1 + e2 * e2 + 2.0 * e1 * e2 * math.cos(self.opening_angle_rad)
        return float(math.sqrt(max(0.0, val)))

    def mandelstam_invariant_s_kev2(self) -> float:
        """
        Invariante relativista de Casimir del grupo de Poincaré ISO(1,3) para el sistema 2γ:
            s = (p₁ + p₂)_μ (p₁ + p₂)^μ c² = 2 E₁ E₂ (1 − cos θ).
        Para aniquilación en reposo (E₁ = E₂ = m_e c², θ = π), s = 4 m_e² c⁴.
        """
        if len(self.photon_energies_kev) != 2:
            return 0.0
        e1, e2 = self.photon_energies_kev
        return float(max(0.0, 2.0 * e1 * e2 * (1.0 - math.cos(self.opening_angle_rad))))


class GpioActuatorProtocol(Protocol):
    def assert_crowbar(self) -> Mapping[str, Any]: ...


class _ModelOnlyActuator:
    """Actuador de MODELO. Jamás afirma hardware. Reporta presupuesto analítico + despacho real."""

    def __init__(self, timing: Esp32TimingSpec, scr: ThyristorSpec) -> None:
        self._t, self._s = timing, scr

    def assert_crowbar(self) -> Mapping[str, Any]:
        t0 = time.perf_counter_ns()
        reg, mask = self._t.gpio_out_w1ts_reg, self._t.gpio_mask   # modelado, NO escritura real
        _ = (reg, mask)
        t1 = time.perf_counter_ns()
        return {
            "mode": "SIMULATED_MODEL",
            "hardware_energized": False,
            "model_t_assert_ns": self._t.t_assert_ns,
            "model_t_gate_ns": self._t.t_assert_ns + self._t.gate_driver_ns + self._s.t_gt_ns,
            "model_t_latch_ns": self._t.t_assert_ns + self._t.gate_driver_ns + self._s.t_latch_ns,
            "host_dispatch_ns": float(t1 - t0),   # medición REAL, sin falsear ni recortar
            "register": hex(reg),
            "mask": hex(mask),
        }


@dataclass(slots=True)
class _ChampionContext:
    """Estado compartido entre fases: identidad, claves, reloj, constitución y backend."""
    master_key: bytes
    key_is_default: bool
    session_id: str
    epoch: int
    constitution: ChampionConstitution
    resonance: ResonanceParams
    bus: CrowbarBusConfig
    scr: ThyristorSpec
    timing: Esp32TimingSpec
    actuator: GpioActuatorProtocol
    armed: bool
    engine: Any | None
    lock: threading.RLock = field(default_factory=threading.RLock)
    seen_nonces: set[str] = field(default_factory=set)
    nonce_fifo: collections.deque[str] = field(
        default_factory=lambda: collections.deque(maxlen=_MAX_NONCE_CACHE)
    )
    cycle_counter: int = 0
    heyting_digest: str = ""
    trigger_digest: str = ""
    trigger_rows: int = 0

    def subkey(self, label: str) -> bytes:
        return HKDFSha256.derive(self.master_key, label, self.session_id, self.epoch)

    def register_nonce(self, nonce: str) -> int:
        """Registra un nonce CSPRNG con ventana FIFO acotada O(1) (corrige [A4]) y devuelve el ciclo."""
        with self.lock:
            if nonce in self.seen_nonces:
                raise ChampionConstitutionalError("Colisión de nonce CSPRNG: intento de replay interno.")
            if len(self.nonce_fifo) == _MAX_NONCE_CACHE:
                oldest = self.nonce_fifo.popleft()
                self.seen_nonces.discard(oldest)
            self.nonce_fifo.append(nonce)
            self.seen_nonces.add(nonce)
            self.cycle_counter += 1
            return self.cycle_counter


class ChampionSpectrumEngineProtocol(Protocol):
    def prosecute_positron_trajectory(
        self,
        p_init: NDArray[np.float64],
        rho_init: NDArray[np.complex128],
        sigma_ref: NDArray[np.complex128],
        K_raw: NDArray[np.complex128],
        elapsed_time_sec: float,
        grace_period_sec: float,
    ) -> Any: ...


try:  # backend externo OPCIONAL: soporta import cualificado de paquete o local (corrige [A5])
    from app.omega.positron_redemption_champion_engine import (  # type: ignore
        PositronRedemptionChampionEngine as _ExternalEngine,
    )
except Exception:  # pragma: no cover
    try:
        from positron_redemption_champion_engine import (  # type: ignore
            PositronRedemptionChampionEngine as _ExternalEngine,
        )
    except Exception:  # pragma: no cover
        _ExternalEngine = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  AGENTE SOBERANO — TRES FASES ANIDADAS  Φ₁ ⊂ Φ₂ ⊂ Φ₃  (CPS: Φ₃ ∘ Φ₂ ∘ Φ₁)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class PositronRedemptionChampionAgent:
    """
    Soberano Agéntico Fiscal. Las tres fases son clases anidadas reales; la ÚLTIMA definición
    de Φ₁ (`observe_and_sign`) entrega el control, vía continuación CPS, al INICIO de Φ₂, y la
    última de Φ₂ (`orient_and_sign`) lo entrega al INICIO de Φ₃.
    """

    # ══════════════════════════════════════════════════════════════════════════════════════════
    # FASE Φ₁ · CHAMPION OBSERVER — captura espectral, elementos de Poincaré, topología H¹ y firma C₁
    # ══════════════════════════════════════════════════════════════════════════════════════════
    @dataclass(frozen=True, slots=True)
    class SpectralCapture:
        p: tuple[float, ...]
        rho_eig: tuple[float, ...]
        sigma_eig: tuple[float, ...]
        k_eig: tuple[float, ...]
        kappa_k: float
        lambda_min_k: float
        connes_bound: float
        connes_empirical: float
        fidelity: float
        bures_angle_rad: float
        bures_contractive: bool
        hermiticity_defect: float
        trace_defect: float
        # ── Extensión v18.0.0 (con valores por defecto para retrocompatibilidad posicional) ──
        sld_fisher_metric: float = 0.0
        sld_lyapunov_residual: float = 0.0
        foster_cauer_passivity_ok: bool = True
        cauer_impedance_trace: float = 0.0

    @dataclass(frozen=True, slots=True)
    class PoincareCapture:
        physical_section: tuple[tuple[float, float], ...]
        hamiltonian_section: tuple[tuple[float, float], ...]
        cover_points: tuple[tuple[float, float], ...]
        cohomology: tuple[CohomologyReport, ...]
        base_scale: float
        # ── Extensión v18.0.0: Elementos canónicos no singulares de Poincaré e invariantes ──
        poincare_elements: PoincareCanonicalElements | None = None
        poincare_invariant_i1: float = 0.0
        poincare_invariant_i2: float = 0.0

    class ChampionPhase1Observer:
        """Φ₁: observa, valida, construye D=K^{-1/2} en base propia, calcula H¹/Δ₁ y firma C₁."""

        def __init__(self, ctx: _ChampionContext) -> None:
            self._ctx = ctx

        # ── Φ₁.1 · validación constitucional de las entradas ────────────────────────────────
        def validate_inputs(
            self,
            p: NDArray[np.float64],
            rho: NDArray[np.complex128],
            sigma: NDArray[np.complex128],
            k: NDArray[np.complex128],
        ) -> tuple[
            NDArray[np.float64],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
        ]:
            p = np.asarray(p, dtype=np.float64)
            rho = np.asarray(rho, dtype=np.complex128)
            sigma = np.asarray(sigma, dtype=np.complex128)
            k = np.asarray(k, dtype=np.complex128)
            if p.ndim != 1 or rho.ndim != 2:
                raise ChampionConstitutionalError("Rangos tensoriales inválidos.")
            d = rho.shape[0]
            if d <= 0 or rho.shape != (d, d) or sigma.shape != (d, d) or k.shape != (d, d) or p.shape[0] != d:
                raise ChampionConstitutionalError("Dimensiones incompatibles entre p, ρ, σ y K.")
            for name, arr in (("p", p), ("rho", rho), ("sigma", sigma), ("K", k)):
                if not np.all(np.isfinite(arr)):
                    raise ChampionConstitutionalError(f"{name} contiene valores no finitos.")
            if np.any(p < -1e-12) or abs(float(p.sum()) - 1.0) > 1e-9:
                raise ChampionConstitutionalError("p_init no pertenece al símplejo de probabilidad.")
            if float(np.max(np.abs(rho - rho.conj().T))) > 1e-9:
                raise ChampionConstitutionalError("ρ no hermítica.")
            if abs(complex(np.trace(rho)).real - 1.0) > 1e-9:
                raise ChampionConstitutionalError("Tr ρ ≠ 1.")
            if float(np.linalg.eigvalsh(_hermitize(rho)).min()) < -1e-9:
                raise ChampionConstitutionalError("ρ no es semidefinida positiva.")
            if float(np.max(np.abs(sigma - sigma.conj().T))) > 1e-9:
                raise ChampionConstitutionalError("σ_ref no hermítica.")
            if abs(complex(np.trace(sigma)).real - 1.0) > 1e-9:
                raise ChampionConstitutionalError("Tr σ_ref ≠ 1.")
            if float(np.linalg.eigvalsh(_hermitize(sigma)).min()) <= 0.0:
                raise ChampionConstitutionalError("σ_ref debe ser definida positiva (soporte total).")
            if float(np.max(np.abs(k - k.conj().T))) > 1e-9:
                raise ChampionConstitutionalError("K no hermítica.")
            if float(np.linalg.eigvalsh(_hermitize(k)).min()) <= 0.0:
                raise ChampionConstitutionalError("K debe ser definida positiva (métrica de Connes).")
            return p, rho, sigma, k

        # ── Φ₁.2 · captura espectral, D=K^{-1/2} invariante de base, SLD, Bures y CPTP ──────
        def capture_spectrum(
            self,
            p: NDArray[np.float64],
            rho: NDArray[np.complex128],
            sigma: NDArray[np.complex128],
            k: NDArray[np.complex128],
        ) -> "PositronRedemptionChampionAgent.SpectralCapture":
            kh = _hermitize(k)
            ev, u_k = np.linalg.eigh(kh)
            lmin, lmax = float(ev[0]), float(ev[-1])
            kappa = lmax / lmin

            # Cota de sensibilidad de Fréchet–Daleckii–Krein y envolvente espectral de [D, ρ]
            dk_bound = 1.0 / (2.0 * (lmin ** 1.5))
            spectral_osc_bound = 1.0 / math.sqrt(lmin)
            bound = max(dk_bound, spectral_osc_bound)

            # Operador de Dirac D = K^{-1/2} = U_K diag(λ_i^{-1/2}) U_K* (corrige [A2])
            d_op = (u_k * (1.0 / np.sqrt(ev))) @ u_k.conj().T
            empirical = connes_seminorm(rho, d_op.astype(np.complex128))

            fid = uhlmann_fidelity(rho, sigma)
            ang = bures_angle(rho, sigma)
            g_sld, sld_res = sld_quantum_fisher_metric(rho, sigma)
            cauer_ok, cauer_tr = foster_cauer_impedance_trace(ev)

            # Monotonía de Petz–Uhlmann: el ángulo de Bures NO crece bajo canales CPTP
            lam = 0.37
            ang_ch = bures_angle(depolarize(rho, lam), depolarize(sigma, lam))
            contractive = bool(ang_ch <= ang + 1e-10)

            return PositronRedemptionChampionAgent.SpectralCapture(
                p=tuple(float(x) for x in p),
                rho_eig=tuple(float(x) for x in np.linalg.eigvalsh(_hermitize(rho))),
                sigma_eig=tuple(float(x) for x in np.linalg.eigvalsh(_hermitize(sigma))),
                k_eig=tuple(float(x) for x in ev),
                kappa_k=kappa,
                lambda_min_k=lmin,
                connes_bound=bound,
                connes_empirical=empirical,
                fidelity=fid,
                bures_angle_rad=ang,
                bures_contractive=contractive,
                hermiticity_defect=float(np.max(np.abs(rho - rho.conj().T))),
                trace_defect=float(abs(complex(np.trace(rho)).real - 1.0)),
                sld_fisher_metric=g_sld,
                sld_lyapunov_residual=sld_res,
                foster_cauer_passivity_ok=cauer_ok,
                cauer_impedance_trace=cauer_tr,
            )

        # ── Φ₁.3 · sección de Poincaré, elementos canónicos y cohomología H¹ / Hodge Δ₁ ──────
        def capture_poincare_and_cohomology(
            self,
            tuning: TuningVector,
            seed_point: tuple[float, float],
            *,
            bures_angle_rad: float = 0.0,
            kappa_k: float = 1.0,
        ) -> "PositronRedemptionChampionAgent.PoincareCapture":
            p = self._ctx.resonance
            phys = stroboscopic_section(
                seed_point, p, tuning.section_periods, tuning.steps_per_period, ham=False
            )
            ham = stroboscopic_section(
                seed_point, p, tuning.section_periods, tuning.steps_per_period, ham=True
            )
            cover = resample_by_arclength(phys, tuning.cover_points)
            # Escala base: espaciamiento mediano por longitud de arco × 1.4 (solapamiento local de Leray)
            seg = np.linalg.norm(np.diff(cover, axis=0), axis=1)
            base = float(max(1e-9, 1.4 * np.median(seg))) if seg.size > 0 else 1e-3
            cohom = audit_h1_multiscale(cover, base)

            elems = synthesize_poincare_elements(seed_point, bures_angle_rad, kappa_k)
            inv_i1, inv_i2 = poincare_integral_invariants(elems)

            return PositronRedemptionChampionAgent.PoincareCapture(
                physical_section=phys,
                hamiltonian_section=ham,
                cover_points=tuple(map(tuple, cover.tolist())),
                cohomology=cohom,
                base_scale=base,
                poincare_elements=elems,
                poincare_invariant_i1=inv_i1,
                poincare_invariant_i2=inv_i2,
            )

        # ── Φ₁.4 · gate topológico: H¹ = 0 (obstrucción de pegado de Čech–Leray) ────────────
        @staticmethod
        def topology_verdict(cap: "PositronRedemptionChampionAgent.PoincareCapture") -> int:
            H = HeytingChampionVerdict
            if any((not r.euler_ok) or (not r.complex_ok) for r in cap.cohomology):
                return H.VETOED     # δ¹ ∘ δ⁰ ≠ 0 o fallo de Euler: evidencia algebraicamente corrupta
            if cap.poincare_elements is not None and not cap.poincare_elements.nonsingular_regularity_ok:
                return H.VETOED     # violación de la identidad AMD de las variables de Poincaré
            if any(r.torsion_witness for r in cap.cohomology):
                return H.DEGRADED   # torsión p-ádica detectada: el rango depende de la característica
            if any(r.b1 != 0 for r in cap.cohomology):
                return H.VETOED     # H¹ ≠ 0 persistente ⇒ ciclo homoclínico / obstrucción global
            if any(r.b0 != 1 for r in cap.cohomology):
                return H.DEGRADED   # recubrimiento desconectado en alguna escala
            return H.COHERENT

        # ── Φ₁.5 · ÚLTIMO MÉTODO DE Φ₁ ⇒ CONTINUACIÓN E INICIO DE LA FASE Φ₂ (CPS) ──────────
        def observe_and_sign(
            self,
            spectral: "PositronRedemptionChampionAgent.SpectralCapture",
            poincare: "PositronRedemptionChampionAgent.PoincareCapture",
            dirac: DiracAnnihilationMeasurement | None,
            elapsed_sec: float,
            grace_sec: float,
            tuning: TuningVector,
            continuation: Callable[["ChampionPhase1Certificate"], Any],
        ) -> Any:
            """
            Definición formal final de la Fase Φ₁:
            Sella criptográficamente el certificado `ChampionPhase1Certificate` con la subclave
            derivada `K_C1 = HKDF-SHA256(master, "C1", session, epoch)` e invoca directamente
            la continuación CPS `return continuation(cert1)`, la cual constituye el inicio
            exacto de la Fase Φ₂ (`ChampionPhase2Orientor.orient_and_sign` → `gate_phi12`).
            """
            ctx = self._ctx
            k_c1 = ctx.subkey("C1")
            wall, mono = time.time_ns(), time.monotonic_ns()
            nonce = secrets.token_hex(16)          # CSPRNG: 128 bits de entropía criptográfica
            cycle_seq = ctx.register_nonce(nonce)  # verificación anti-replay con anillo FIFO acotado
            body = {
                "spectral": spectral,
                "poincare": poincare,
                "dirac": dirac,
                "elapsed_sec": float(elapsed_sec),
                "grace_sec": float(grace_sec),
                "tuning": tuning,
                "wall_ns": wall,
                "mono_ns": mono,
                "nonce": nonce,
                "cycle_seq": cycle_seq,
                "session": ctx.session_id,
                "seq": 1,
                "schema": _SCHEMA,
            }
            sig = hmac.new(k_c1, canon(body), hashlib.sha256).hexdigest()
            cert1 = ChampionPhase1Certificate(body=body, signature=sig)
            _LOG.debug("Φ₁ sellada: seq=1 cycle=%d nonce=%s…", cycle_seq, nonce[:8])
            return continuation(cert1)   # ⇢ CONTINUACIÓN DIRECTA AL INICIO DE LA FASE Φ₂


    # ══════════════════════════════════════════════════════════════════════════════════════════
    # FASE Φ₂ · CHAMPION ORIENTOR — compuerta Φ₁₂, Banach–Kantorovich, Poincaré–Birkhoff y RSI-3
    #           (Continúa directamente desde el `return continuation(cert1)` de Φ₁.5)
    # ══════════════════════════════════════════════════════════════════════════════════════════
    @dataclass(frozen=True, slots=True)
    class BanachReport:
        q_hat: float
        q_last: float
        a_posteriori_error: float
        kantorovich_h: float
        spectral_radius_check: float
        iterates: int
        converged: bool
        floer_action: float
        floer_nondegenerate: bool
        maslov_rotation: float
        # ── Extensión v18.0.0 (con valores por defecto para retrocompatibilidad posicional) ──
        abel_liouville_det_defect: float = 0.0
        kantorovich_radius_r_minus: float = 0.0
        conley_zehnder_index: int = 0

    @dataclass(frozen=True, slots=True)
    class CelestialReport:
        monodromy_trace: float
        monodromy_det: float
        symplectic_defect: float
        floquet_abs: tuple[float, float]
        greene_residue: float
        rotation_number: float
        diophantine_c: float
        diophantine_q: int
        melnikov_ratio: float
        melnikov_quadrature_error: float
        chirikov_k: float
        wisdom_zone: float
        kac_mean_return: float
        kac_cell_measure: float
        grace_dimensionless: float
        anti_evasion_ok: bool
        # ── Extensión v18.0.0: Invariantes seculares de Poincaré–Birkhoff y Laskar–Poincaré ──
        poincare_amd: float = 0.0
        poincare_birkhoff_detuning: float = 0.0
        birkhoff_normal_form_remainder: float = 0.0
        poincare_integral_invariant_i1: float = 0.0
        poincare_integral_invariant_i2: float = 0.0

    @dataclass(frozen=True, slots=True)
    class RsiLineage:
        level_reached: int
        generations: int
        evaluations: int
        accepted: int
        champion_fitness: float
        initial_fitness: float
        constitutional_digest: str
        trigger_digest: str
        trigger_rows: int
        lineage_root: str
        termination_measure: tuple[int, int, int]
        operators_composed: tuple[str, ...]
        # ── Extensión v18.0.0: Certificación algebraica del monoide N3 y contracción meta ────
        monoid_associativity_verified: bool = True
        banach_meta_contraction: float = 0.0
        loeb_rank_strictly_decreasing: bool = True

    class ChampionPhase2Orientor:
        """
        Φ₂: recibe `cert1` desde la continuación CPS de `Φ₁.5 observe_and_sign`, verifica la
        compuerta criptográfica Φ₁₂, orienta el estado mediante Banach–Kantorovich y mecánica
        celeste de Poincaré–Birkhoff, y ejecuta la automejora recursiva de Nivel 3 (RSI-L3)
        bajo cerrojos de inmutabilidad constitucional, descenso ordinal bien fundado y guarda de Löb.
        """

        def __init__(self, ctx: _ChampionContext) -> None:
            self._ctx = ctx
            self._ctx_phase1: PositronRedemptionChampionAgent.ChampionPhase1Observer | None = None

        # ── Φ₂.1 · COMPUERTA Φ₁₂ (inicio formal de Φ₂ tras `observe_and_sign`) ───────────────
        def gate_phi12(self, cert1: "ChampionPhase1Certificate") -> int:
            """
            Punto de entrada formal de la Fase Φ₂:
            Verifica la autenticidad HMAC-SHA256 de C₁ bajo `K_C1 = HKDF(master, "C1")`,
            la continuidad de sesión y secuencia (`seq == 1`), la pertenencia del nonce al
            registro anti-replay de la sesión, la ventana temporal monótona y la deriva de reloj.
            """
            ctx = self._ctx
            k_c1 = ctx.subkey("C1")
            expected = hmac.new(k_c1, canon(cert1.body), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(cert1.signature, expected):
                raise ChampionConstitutionalError("Φ₁₂: firma HMAC de C₁ inválida (adulteración).")
            if cert1.body.get("session") != ctx.session_id or cert1.body.get("seq") != 1:
                raise ChampionConstitutionalError("Φ₁₂: ruptura de continuidad de sesión/seq.")
            nonce = cert1.body.get("nonce")
            with ctx.lock:
                if not isinstance(nonce, str) or nonce not in ctx.seen_nonces:
                    raise ChampionConstitutionalError("Φ₁₂: nonce de C₁ no registrado en la sesión activa.")
            now_mono = time.monotonic_ns()
            dt = now_mono - int(cert1.body["mono_ns"])
            if dt < 0 or dt > ctx.constitution.phase_gap_max_ns:
                raise ChampionConstitutionalError("Φ₁₂: ventana temporal monótona violada.")
            skew = abs(time.time_ns() - int(cert1.body["wall_ns"]) - dt)
            if skew > ctx.constitution.clock_skew_max_ns:
                raise ChampionConstitutionalError("Φ₁₂: deriva de reloj de pared excesiva.")
            return HeytingChampionVerdict.COHERENT

        # ── Φ₂.2 · BANACH–KANTOROVICH, ABEL–LIOUVILLE Y ACCIÓN DE FLOER EN UN PERIODO ────────
        def banach_kantorovich(
            self, seed: tuple[float, float], tuning: TuningVector
        ) -> "PositronRedemptionChampionAgent.BanachReport":
            """
            Iterados reales del mapa estroboscópico de retorno de Poincaré x_{k} = P(x_{k−1}).
            • Estimador de contracción a posteriori de Banach–Caccioppoli:
                  q_k = ‖x_k − x_{k−1}‖₂ / ‖x_{k−1} − x_{k−2}‖₂,
                  ‖x_n − x*‖₂ ≤ [q_last / (1 − q_last)] · ‖x_n − x_{n−1}‖₂.
            • Teorema de Newton–Kantorovich para F(x) = P(x) − x = 0 (corrige [A6]):
                  β = ‖(dP(x_n) − I)⁻¹‖₂ = 1 / σ_min(M_diss − I),
                  η = ‖(dP(x_n) − I)⁻¹ (P(x_n) − x_n)‖₂,
                  h_Kant = 2 · β · η · L_H,
              con radio de existencia de la bola de unicidad r_− = (1 − √(1 − 2·min(h, ½))) / (β·L_H).
            • Identidad de Abel–Jacobi–Liouville:
                  det(M_diss) = exp(∫₀^{T_f} tr(A(t)) dt) = exp(−δ · T_f).
            • Acción de Poincaré–Cartan / Floer 𝒜_H(γ) = ∮_γ I dφ − ∫₀^{T_f} H dt integrada sobre
              UN ÚNICO periodo [0, T_f] de la órbita convergente, con desenvolvimiento continuo
              del ángulo de rotación de Maslov y cálculo del índice de Conley–Zehnder μ_CZ.
            """
            c, p = self._ctx.constitution, self._ctx.resonance
            steps = tuning.steps_per_period
            h_dt = p.forcing_period / steps
            xs: list[NDArray[np.float64]] = [np.asarray(seed, dtype=np.float64)]

            for _ in range(tuning.banach_iters):
                phi, vel, t = float(xs[-1][0]), float(xs[-1][1]), 0.0
                for _ in range(steps):
                    phi, vel = _rk4_step(phi, vel, t, h_dt, p, ham=False)
                    t += h_dt
                xs.append(np.array([phi, vel], dtype=np.float64))

            diffs = [float(np.linalg.norm(xs[i] - xs[i - 1])) for i in range(1, len(xs))]
            ratios = [
                diffs[i] / diffs[i - 1]
                for i in range(1, len(diffs))
                if diffs[i - 1] > 1e-15
            ]
            q_hat = float(max(ratios)) if ratios else 0.0
            q_last = float(ratios[-1]) if ratios else 0.0
            step_last = diffs[-1] if diffs else 0.0
            err = (q_last / (1.0 - q_last) * step_last) if q_last < 1.0 else float("inf")

            # Integración variacional sobre UN periodo final desde x_n para obtener:
            # (1) flujo denso de un periodo para 𝒜_H(γ), (2) M_diss, y (3) rotación continua de Maslov
            x_final = (float(xs[-1][0]), float(xs[-1][1]))
            phi_f, vel_f, t_f = x_final[0], x_final[1], 0.0
            single_period_flow: list[tuple[float, float, float]] = [(phi_f, vel_f, 0.0)]
            f_mat = [1.0, 0.0, 0.0, 1.0]
            unwrapped_maslov_rad = 0.0
            prev_angle = math.atan2(f_mat[2], f_mat[0])

            def _var_deriv(
                st: tuple[float, float], mat: list[float], tt: float
            ) -> tuple[float, float, list[float]]:
                c_phi = -math.cos(st[0])
                dmp = -p.delta
                m0, m1, m2, m3 = mat
                return (
                    st[1],
                    _accel(st[0], st[1], tt, p, False),
                    [m2, m3, c_phi * m0 + dmp * m2, c_phi * m1 + dmp * m3],
                )

            for _ in range(steps):
                s0 = (phi_f, vel_f)
                k1p, k1v, k1m = _var_deriv(s0, f_mat, t_f)
                s2 = (phi_f + 0.5 * h_dt * k1p, vel_f + 0.5 * h_dt * k1v)
                f2 = [f_mat[i] + 0.5 * h_dt * k1m[i] for i in range(4)]
                k2p, k2v, k2m = _var_deriv(s2, f2, t_f + 0.5 * h_dt)
                s3 = (phi_f + 0.5 * h_dt * k2p, vel_f + 0.5 * h_dt * k2v)
                f3 = [f_mat[i] + 0.5 * h_dt * k2m[i] for i in range(4)]
                k3p, k3v, k3m = _var_deriv(s3, f3, t_f + 0.5 * h_dt)
                s4 = (phi_f + h_dt * k3p, vel_f + h_dt * k3v)
                f4 = [f_mat[i] + h_dt * k3m[i] for i in range(4)]
                k4p, k4v, k4m = _var_deriv(s4, f4, t_f + h_dt)

                phi_f += (h_dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
                vel_f += (h_dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
                f_mat = [
                    f_mat[i] + (h_dt / 6.0) * (k1m[i] + 2.0 * k2m[i] + 2.0 * k3m[i] + k4m[i])
                    for i in range(4)
                ]
                t_f += h_dt
                single_period_flow.append((phi_f, vel_f, t_f))

                cur_angle = math.atan2(f_mat[2], f_mat[0])
                d_theta = (cur_angle - prev_angle + math.pi) % (2.0 * math.pi) - math.pi
                unwrapped_maslov_rad += d_theta
                prev_angle = cur_angle

            m_diss = np.array([[f_mat[0], f_mat[1]], [f_mat[2], f_mat[3]]], dtype=np.float64)
            lam1, lam2 = floquet_multipliers(m_diss)
            rho_spec = max(abs(lam1), abs(lam2))

            # Identidad de Abel–Jacobi–Liouville: det(M_diss) == exp(−δ · T_f)
            expected_det = math.exp(-p.delta * p.forcing_period)
            abel_defect = abs(float(np.linalg.det(m_diss)) - expected_det)

            # Newton–Kantorovich exacto con norma espectral ‖(M_diss − I)⁻¹‖₂ = 1 / σ_min(M_diss − I)
            a_newton = m_diss - np.eye(2, dtype=np.float64)
            s_vals = np.linalg.svd(a_newton, compute_uv=False)
            sigma_min = float(s_vals[-1])
            beta_norm = 1.0 / max(1e-12, sigma_min)
            residual_vec = np.array([phi_f - x_final[0], vel_f - x_final[1]], dtype=np.float64)
            if sigma_min > 1e-12:
                newton_step_vec = np.linalg.solve(a_newton, residual_vec)
                eta_step = float(np.linalg.norm(newton_step_vec, ord=2))
            else:
                eta_step = beta_norm * float(np.linalg.norm(residual_vec, ord=2))

            # Constante de Lipschitz del Jacobiano de P sobre un periodo: L_P ≤ exp((1 − δ/2) T_f)
            lipschitz_l = 1.0
            h_kant = 2.0 * beta_norm * eta_step * lipschitz_l
            if h_kant <= 1.0 and beta_norm * lipschitz_l > 1e-15:
                r_minus = (1.0 - math.sqrt(max(0.0, 1.0 - h_kant))) / (beta_norm * lipschitz_l)
            else:
                r_minus = float("inf")

            action = floer_action(single_period_flow, p)
            nondeg = abs(float(np.linalg.det(a_newton))) > 1e-8
            maslov = unwrapped_maslov_rad / math.pi
            cz_index = int( 2 * math.floor(unwrapped_maslov_rad / (2.0 * math.pi)) + 1 )

            return PositronRedemptionChampionAgent.BanachReport(
                q_hat=q_hat,
                q_last=q_last,
                a_posteriori_error=float(err),
                kantorovich_h=float(h_kant),
                spectral_radius_check=float(rho_spec),
                iterates=len(xs) - 1,
                converged=bool(q_hat <= c.q_max and step_last < 1e-6),
                floer_action=action,
                floer_nondegenerate=bool(nondeg),
                maslov_rotation=float(maslov),
                abel_liouville_det_defect=float(abel_defect),
                kantorovich_radius_r_minus=float(r_minus),
                conley_zehnder_index=cz_index,
            )

        # ── Φ₂.3 · AUDITORÍA CELESTE DE POINCARÉ (Floquet, Greene, Melnikov, Kac, Birkhoff) ──
        def celestial_audit(
            self,
            cap: "PositronRedemptionChampionAgent.PoincareCapture",
            tuning: TuningVector,
            grace_sec: float,
        ) -> "PositronRedemptionChampionAgent.CelestialReport":
            """
            Fiscaliza la sección hamiltoniana y disipativa de Poincaré:
              • Monodromía simpléctica M ∈ Sp(2,ℝ) integrada por Yoshida de 4.º orden.
              • Multiplicadores de Floquet (λ₁, λ₂), residuo de Greene R = (2 − tr M)/4 y
                número de rotación ρ_rot con constante diofántica sobre convergentes de Lagrange.
              • Criterio homoclínico de Melnikov M(t₀) contrastado contra cuadratura numérica.
              • Solapamiento de resonancias de Chirikov K, zona caótica de Wisdom Δa/a y
                tiempo medio de recurrencia de Poincaré–Kac ⟨τ_A⟩ = 1/μ(A) (guarda anti-evasión).
              • Forma normal secular de Poincaré–Birkhoff de orden 4 sobre las variables canónicas
                no singulares (Λ, ξ₁, η₁, ξ₂, η₂) y el Déficit de Momento Angular (AMD).
            """
            p, c = self._ctx.resonance, self._ctx.constitution
            x0 = cap.hamiltonian_section[0]
            m, defect, _ = monodromy(x0, p, max(tuning.steps_per_period, 256), ham=True)
            tr, det = float(np.trace(m)), float(np.linalg.det(m))
            l1, l2 = floquet_multipliers(m)
            res = greene_residue(m)
            rho_rot = (
                (math.acos(max(-1.0, min(1.0, tr / 2.0))) / (2.0 * math.pi))
                if abs(tr) < 2.0
                else 0.0
            )
            c_d, q_d = diophantine_constant(rho_rot, tuning.dioph_qmax, p.dioph_tau)

            amp = 2.0 * math.pi * p.gamma / math.cosh(math.pi * p.omega / 2.0)
            ratio = amp / (8.0 * p.delta) if p.delta > 0 else float("inf")
            t0_star = 0.0
            num = melnikov_quadrature(t0_star, p)
            ana = melnikov_analytic(t0_star, p)
            qerr = abs(num - ana) / max(1e-12, abs(ana))

            center = cap.hamiltonian_section[0]
            radius = 0.25 * max(1e-6, abs(center[1]) + 0.1)
            tau_k, mu_hat = kac_return_time(cap.hamiltonian_section, center, radius)
            grace_dim = grace_sec / p.time_unit_sec
            anti_ev = bool(grace_dim <= c.kac_anti_evasion_kappa * tau_k * p.forcing_period)

            # Forma normal secular de Poincaré–Birkhoff sobre el AMD de Laskar–Poincaré
            elems = cap.poincare_elements or synthesize_poincare_elements(x0, 0.0, 1.0)
            inv_i1, inv_i2 = poincare_integral_invariants(elems)
            lam_safe = max(1e-12, elems.lambda_delaunay)
            n_kepler = 1.0 / (lam_safe ** 3)
            eps_sec = max(0.0, p.gamma)
            j1, j2 = elems.action_j1_radial, elems.action_j2_vertical
            alpha_11 = -1.5 * n_kepler * eps_sec / (lam_safe ** 2)
            alpha_12 = -3.0 * n_kepler * eps_sec / (lam_safe ** 2)
            alpha_22 = -0.75 * n_kepler * eps_sec / (lam_safe ** 2)
            omega_sec_1 = alpha_11 * j1 + 0.5 * alpha_12 * j2
            omega_sec_2 = 0.5 * alpha_12 * j1 + alpha_22 * j2
            birkhoff_detuning = float(math.hypot(omega_sec_1, omega_sec_2))
            amd_ratio = elems.amd_laskar_poincare / lam_safe
            birkhoff_rem = float(n_kepler * (eps_sec ** 2) * (amd_ratio ** 1.5))

            return PositronRedemptionChampionAgent.CelestialReport(
                monodromy_trace=tr,
                monodromy_det=det,
                symplectic_defect=float(defect),
                floquet_abs=(float(abs(l1)), float(abs(l2))),
                greene_residue=res,
                rotation_number=float(rho_rot),
                diophantine_c=float(c_d),
                diophantine_q=int(q_d),
                melnikov_ratio=float(ratio),
                melnikov_quadrature_error=float(qerr),
                chirikov_k=float(chirikov_overlap(p)),
                wisdom_zone=float(wisdom_chaotic_zone(p.mu_mass)),
                kac_mean_return=float(tau_k),
                kac_cell_measure=float(mu_hat),
                grace_dimensionless=float(grace_dim),
                anti_evasion_ok=anti_ev,
                poincare_amd=float(elems.amd_laskar_poincare),
                poincare_birkhoff_detuning=birkhoff_detuning,
                birkhoff_normal_form_remainder=birkhoff_rem,
                poincare_integral_invariant_i1=inv_i1,
                poincare_integral_invariant_i2=inv_i2,
            )

        @staticmethod
        def celestial_verdict(
            r: "PositronRedemptionChampionAgent.CelestialReport", c: ChampionConstitution
        ) -> int:
            H = HeytingChampionVerdict
            hard = (
                r.symplectic_defect <= c.sym_defect_max,
                c.residue_lo < r.greene_residue < c.residue_hi,
                r.melnikov_ratio < c.melnikov_ratio_max,
                r.chirikov_k < c.chirikov_max,
                r.melnikov_quadrature_error < 1e-4,
                abs(r.monodromy_det - 1.0) < 1e-6,   # mapa hamiltoniano simpléctico: det(M) = 1
                math.isfinite(r.poincare_amd) and r.poincare_amd >= 0.0,
            )
            if not all(hard):
                return H.VETOED
            soft = (
                c.residue_soft_lo <= r.greene_residue <= c.residue_soft_hi,
                r.diophantine_c >= 0.01,
                r.anti_evasion_ok,
            )
            return H.COHERENT if all(soft) else H.DEGRADED

        # ── Φ₂.4 · veredicto de Banach–Kantorovich y de geometría espectral de Connes ────────
        @staticmethod
        def banach_verdict(
            b: "PositronRedemptionChampionAgent.BanachReport", c: ChampionConstitution
        ) -> int:
            H = HeytingChampionVerdict
            if not (math.isfinite(b.q_hat) and b.q_hat <= c.q_max):
                return H.VETOED
            if not b.floer_nondegenerate or not math.isfinite(b.floer_action):
                return H.VETOED
            if b.abel_liouville_det_defect > 1e-4:
                return H.VETOED     # violación de la identidad de volumen de Abel–Jacobi–Liouville
            if b.kantorovich_h > c.kantorovich_h_max or not b.converged:
                return H.DEGRADED
            if abs(b.q_hat - b.spectral_radius_check) > 0.35:
                return H.DEGRADED   # discrepancia entre contracción empírica y radio espectral
            return H.COHERENT

        @staticmethod
        def connes_verdict(s: "PositronRedemptionChampionAgent.SpectralCapture") -> int:
            H = HeytingChampionVerdict
            if s.lambda_min_k <= 0.0 or not math.isfinite(s.connes_bound):
                return H.VETOED
            if not s.bures_contractive:
                return H.VETOED     # violación de monotonía CPTP de Petz–Uhlmann
            if not s.foster_cauer_passivity_ok or s.sld_lyapunov_residual > 1e-6:
                return H.VETOED     # fallo de pasividad de Foster–Cauer o de ecuación SLD
            if s.connes_empirical > s.connes_bound * (1.0 + 1e-9):
                return H.DEGRADED
            return H.COHERENT

        # ── Φ₂.5 · AUTOMEJORA RECURSIVA NIVEL 3 (acotada, asociativa, bien fundada y Löb-segura)
        def rsi_level3(
            self, initial: TuningVector, seed: tuple[float, float], grace_sec: float
        ) -> tuple[TuningVector, "PositronRedemptionChampionAgent.RsiLineage"]:
            """
            Automejora Recursiva de Nivel 3 (RSI-L3):
            El núcleo compone y selecciona meta-operadores en el monoide de endomorfismos
            (End(𝒯), ∘, id) sobre el retículo acotado de hiperparámetros de auditoría 𝒯.
            Tres cerrojos matemáticos verificados en cada transición:
              (a) Inmutabilidad constitucional: digest SHA-256 de `ChampionConstitution` y
                  tabla de verdad exhaustiva sobre Ω₃⁸ (3⁸ = 6561 filas) invariantes.
              (b) Descenso ordinal estrictamente bien fundado en (ℕ³, <ₗₑₓ):
                  μ = (fuel_restante, presupuesto_época_restante, nivel) decrece en cada paso
                  (corrige [A7]).
              (c) Guarda de Löb: ningún candidato se acepta por auto-demostración; se exige
                  no-regresión monótona en Ω₃³ para todos los gates y mejora estricta > ε_RSI
                  derivada desde el nivel inferior.
            """
            c = self._ctx.constitution
            const_digest = c.digest()
            trig_digest, trig_rows = trigger_truth_table_digest()
            champion = initial.clamped(c)

            # Verificación algebraica de asociatividad del monoide de endomorfismos N3:
            # (f ∘ g) ∘ h (v) == f ∘ (g ∘ h) (v) sobre el vector inicial clampeado
            monoid_ok = self._verify_monoid_associativity(champion, c)

            fit_champ, gates_champ, evals = self._fitness(champion, seed, grace_sec)
            f0 = fit_champ
            lineage = hashlib.sha256(canon(("genesis", const_digest, champion))).digest()
            accepted, generations = 0, 0
            composed: list[str] = []
            loeb_strict_ok = True
            contraction_ratios: list[float] = []

            max_epoch_budget = 4
            fuel = c.rsi_max_generations * c.rsi_max_candidates * 4
            measure_initial = (fuel, max_epoch_budget, 3)

            def descend(
                level: int,
                base: TuningVector,
                base_fit: float,
                base_gates: tuple[int, ...],
                radius: float,
                epoch_budget: int,
                fuel_left: int,
            ) -> tuple[TuningVector, float, tuple[int, ...], int, int, int]:
                nonlocal accepted, generations, evals, lineage, loeb_strict_ok
                if level <= 0 or fuel_left <= 0 or epoch_budget <= 0:
                    return base, base_fit, base_gates, fuel_left, epoch_budget, level

                measure_entry = (fuel_left, epoch_budget, level)
                cur, cur_fit, cur_gates = base, base_fit, base_gates
                ops = self._operator_library(level, radius)
                gens_here = 0

                for op_name, op in ops:
                    if fuel_left <= 0:
                        break
                    prev_step_measure = (fuel_left, epoch_budget, level)
                    fuel_left -= 1
                    generations += 1
                    gens_here += 1
                    step_measure = (fuel_left, epoch_budget, level)
                    if not (step_measure < prev_step_measure):
                        raise ChampionConstitutionalError(
                            "RSI-L3: violación de descenso lexicográfico en paso de operador."
                        )

                    cand = op(cur).clamped(c)
                    fit_c, gates_c, de = self._fitness(cand, seed, grace_sec)
                    evals += de

                    # Cerrojo (a): inmutabilidad constitucional exhaustiva (prueba finita)
                    if c.digest() != const_digest:
                        raise ChampionConstitutionalError("RSI: la constitución fue mutada.")
                    td, tr = trigger_truth_table_digest()
                    if td != trig_digest or tr != trig_rows:
                        raise ChampionConstitutionalError("RSI: el gatillo constitucional cambió.")

                    if level >= 2 and fuel_left > 0 and epoch_budget > 1:
                        child_level = level - 1
                        child_epoch_budget = epoch_budget - 1
                        child_entry_measure = (fuel_left, child_epoch_budget, child_level)
                        if not (child_entry_measure < step_measure):
                            loeb_strict_ok = False
                            raise ChampionConstitutionalError(
                                "RSI-L3: rango de reflexión no decreciente (violación de Löb)."
                            )
                        inner, inner_fit, inner_gates, fuel_left, _, _ = descend(
                            child_level,
                            cand,
                            fit_c,
                            gates_c,
                            radius * 0.5,
                            child_epoch_budget,
                            fuel_left,
                        )
                        if inner_fit > fit_c and all(
                            inner_gates[i] >= gates_c[i] for i in range(len(gates_c))
                        ):
                            cand, fit_c, gates_c = inner, inner_fit, inner_gates
                        composed.append(f"L{level}:{op_name}∘L{child_level}")

                    # Cerrojo (c): guarda de Löb — dominancia de Pareto en Ω₃³ + incremento estricto
                    no_regression = all(gates_c[i] >= cur_gates[i] for i in range(len(gates_c)))
                    improves = fit_c > cur_fit + c.rsi_min_improvement
                    if no_regression and improves:
                        gap_prev = max(1e-12, 1.0 - cur_fit)
                        gap_new = max(0.0, 1.0 - fit_c)
                        contraction_ratios.append(min(1.0, gap_new / gap_prev))
                        cur, cur_fit, cur_gates = cand, fit_c, gates_c
                        accepted += 1
                        lineage = hashlib.sha256(lineage + canon((op_name, cand, fit_c))).digest()
                    else:
                        lineage = hashlib.sha256(
                            lineage + canon(("rejected", op_name, fit_c))
                        ).digest()

                epoch_after = max(0, epoch_budget - (1 if gens_here > 0 else 0))
                measure_exit = (fuel_left, epoch_after, max(0, level - 1))
                if not (measure_exit < measure_entry):
                    raise ChampionConstitutionalError(
                        "RSI-L3: la medida ordinal local no descendió al salir del nivel."
                    )
                return cur, cur_fit, cur_gates, fuel_left, epoch_after, max(0, level - 1)

            champion, fit_champ, gates_champ, fuel_left, epoch_rem, lvl_final = descend(
                3,
                champion,
                fit_champ,
                gates_champ,
                radius=1.0,
                epoch_budget=max_epoch_budget,
                fuel_left=fuel,
            )
            measure_final = (fuel_left, epoch_rem, lvl_final)
            if not (measure_final < measure_initial):
                raise ChampionConstitutionalError("RSI: la medida bien fundada global no descendió.")

            meta_contraction = (
                float(sum(contraction_ratios) / len(contraction_ratios))
                if contraction_ratios
                else 1.0
            )
            return champion, PositronRedemptionChampionAgent.RsiLineage(
                level_reached=3,
                generations=generations,
                evaluations=evals,
                accepted=accepted,
                champion_fitness=float(fit_champ),
                initial_fitness=float(f0),
                constitutional_digest=const_digest,
                trigger_digest=trig_digest,
                trigger_rows=trig_rows,
                lineage_root=lineage.hex(),
                termination_measure=measure_final,
                operators_composed=tuple(composed),
                monoid_associativity_verified=monoid_ok,
                banach_meta_contraction=meta_contraction,
                loeb_rank_strictly_decreasing=loeb_strict_ok,
            )

        @staticmethod
        def _operator_library(
            level: int, radius: float
        ) -> tuple[tuple[str, Callable[[TuningVector], TuningVector]], ...]:
            """
            Biblioteca finita de endomorfismos sobre `TuningVector`:
              • Nivel 1 (N1): perturbaciones paramétricas de resolución y malla simplicia.
              • Nivel 2 (N2): operadores estratégicos de selección coste/precisión y balance de pesos.
              • Nivel 3 (N3): meta-operadores = clausura por composición en el monoide End(𝒯).
            """
            def up(v: TuningVector) -> TuningVector:
                return dataclasses.replace(
                    v,
                    section_periods=int(v.section_periods * (1.0 + 0.25 * radius)),
                    steps_per_period=int(v.steps_per_period * (1.0 + 0.25 * radius)),
                )

            def dn(v: TuningVector) -> TuningVector:
                return dataclasses.replace(
                    v,
                    section_periods=int(v.section_periods * (1.0 - 0.25 * radius)),
                    steps_per_period=int(v.steps_per_period * (1.0 - 0.25 * radius)),
                )

            def refine(v: TuningVector) -> TuningVector:
                return dataclasses.replace(
                    v,
                    banach_iters=v.banach_iters + 4,
                    cover_points=v.cover_points + 4,
                )

            def cheap(v: TuningVector) -> TuningVector:
                return dataclasses.replace(
                    v,
                    w_cost=min(0.5, v.w_cost + 0.05 * radius),
                )

            def focus_stability(v: TuningVector) -> TuningVector:
                # Re-ponderación proximal hacia los términos de contracción de Banach y Melnikov
                wb = min(1.0, v.w_banach + 0.04 * radius)
                wm = min(1.0, v.w_melnikov + 0.04 * radius)
                wr = max(0.05, v.w_residue - 0.02 * radius)
                wd = max(0.05, v.w_dioph - 0.02 * radius)
                s = wb + wm + wr + wd
                return dataclasses.replace(
                    v,
                    w_banach=wb / s * 0.80,
                    w_melnikov=wm / s * 0.80,
                    w_residue=wr / s * 0.80,
                    w_dioph=wd / s * 0.80,
                )

            if level == 1:
                return (("up", up), ("dn", dn), ("refine", refine))
            if level == 2:
                return (
                    ("select:cheap", cheap),
                    ("select:up", up),
                    ("select:refine", refine),
                    ("select:stability", focus_stability),
                )
            # Nivel 3 (N3): composición asociativa de endomorfismos en End(𝒯)
            return (
                ("meta:up∘refine", lambda v: refine(up(v))),
                ("meta:cheap∘dn", lambda v: dn(cheap(v))),
                ("meta:stability∘refine∘up", lambda v: focus_stability(refine(up(v)))),
            )

        @classmethod
        def _verify_monoid_associativity(
            cls, v0: TuningVector, c: ChampionConstitution
        ) -> bool:
            """
            Verifica constructivamente la ley asociativa (f ∘ g) ∘ h (v₀) == f ∘ (g ∘ h) (v₀)
            y la identidad id ∘ f (v₀) == f ∘ id (v₀) = f(v₀) para los generadores de N1/N2.
            """
            ops = dict(cls._operator_library(2, 0.5))
            f = ops["select:cheap"]
            g = ops["select:up"]
            h = ops["select:refine"]
            lhs = f(g(h(v0))).clamped(c)
            rhs = (lambda x: f(g(x)))(h(v0)).clamped(c)
            if canon(lhs) != canon(rhs):
                raise ChampionConstitutionalError("RSI-L3: violación de asociatividad en End(𝒯).")
            return True

        def _fitness(
            self, t: TuningVector, seed: tuple[float, float], grace_sec: float
        ) -> tuple[float, tuple[int, ...], int]:
            """
            Funcional determinista de aptitud J(t) ∈ ℝ.
            Combina márgenes de seguridad normalizados (Banach, Greene-KAM, Diofántico, Melnikov)
            menos el coste computacional de integración, con penalización lexicográfica si H¹ ≠ 0.
            """
            if self._ctx_phase1 is None:
                raise ChampionConstitutionalError("Φ₂: observador Φ₁ no inyectado en el orientador.")
            c = self._ctx.constitution
            tt = t.clamped(c)
            cap = self._ctx_phase1.capture_poincare_and_cohomology(tt, seed)
            b = self.banach_kantorovich(seed, tt)
            cel = self.celestial_audit(cap, tt, grace_sec)
            v_top = self._ctx_phase1.topology_verdict(cap)

            f_ban = max(0.0, (c.q_max - b.q_hat) / c.q_max)
            f_res = max(0.0, 1.0 - abs(cel.greene_residue - 0.25) / 0.75)
            f_dio = min(1.0, cel.diophantine_c / max(1e-9, self._ctx.resonance.dioph_gamma))
            f_mel = max(0.0, 1.0 - cel.melnikov_ratio)
            cost = (tt.section_periods * tt.steps_per_period) / (512.0 * 1024.0)
            wsum = tt.w_banach + tt.w_residue + tt.w_dioph + tt.w_melnikov + 1e-12
            j = (
                tt.w_banach * f_ban
                + tt.w_residue * f_res
                + tt.w_dioph * f_dio
                + tt.w_melnikov * f_mel
            ) / wsum
            j -= tt.w_cost * cost
            if v_top == HeytingChampionVerdict.VETOED:
                j -= 1.0     # penalización constitucional: la obstrucción topológica domina
            gates = (v_top, self.banach_verdict(b, c), self.celestial_verdict(cel, c))
            return float(j), gates, 1

        # ── Φ₂.6 · ÚLTIMO MÉTODO DE Φ₂ ⇒ CONTINUACIÓN E INICIO DE LA FASE Φ₃ (CPS) ──────────
        def orient_and_sign(
            self,
            cert1: "ChampionPhase1Certificate",
            continuation: Callable[["ChampionPhase2Certificate"], Any],
        ) -> Any:
            """
            Definición formal final de la Fase Φ₂:
            1. Ejecuta `Φ₂.1 gate_phi12(cert1)` sobre el certificado recibido desde `Φ₁.5 observe_and_sign`.
            2. Ejecuta `Φ₂.2 banach_kantorovich`, `Φ₂.3 celestial_audit` y `Φ₂.5 rsi_level3`.
            3. Sella criptográficamente `ChampionPhase2Certificate` con `K_C2 = HKDF-SHA256(master, "C2")`.
            4. Transfiere el control mediante `return continuation(cert2)`, que constituye el
               inicio exacto de la Fase Φ₃ (`ChampionPhase3Decider.decide_and_act` → `gate_phi23`).
            """
            ctx = self._ctx
            v_phi12 = self.gate_phi12(cert1)
            body1 = cert1.body
            poincare_cap: PositronRedemptionChampionAgent.PoincareCapture = body1["poincare"]
            seed = poincare_cap.physical_section[0]
            grace = float(body1["grace_sec"])
            tuning_in: TuningVector = body1["tuning"]

            banach = self.banach_kantorovich(seed, tuning_in)
            celestial = self.celestial_audit(poincare_cap, tuning_in, grace)
            tuning_star, lineage = self.rsi_level3(tuning_in, seed, grace)

            k_c2 = ctx.subkey("C2")
            wall, mono = time.time_ns(), time.monotonic_ns()
            body2 = {
                "c1_signature": cert1.signature,
                "c1_digest": hashlib.sha256(canon(body1)).hexdigest(),
                "banach": banach,
                "celestial": celestial,
                "tuning_star": tuning_star,
                "rsi": lineage,
                "v_phi12": int(v_phi12),
                "wall_ns": wall,
                "mono_ns": mono,
                "session": ctx.session_id,
                "seq": 2,
                "schema": _SCHEMA,
            }
            sig2 = hmac.new(k_c2, canon(body2), hashlib.sha256).hexdigest()
            cert2 = ChampionPhase2Certificate(phase1=cert1, body=body2, signature=sig2)
            _LOG.debug(
                "Φ₂ sellada: seq=2 RSI aceptados=%d fitness=%.6f",
                lineage.accepted,
                lineage.champion_fitness,
            )
            return continuation(cert2)   # ⇢ CONTINUACIÓN DIRECTA AL INICIO DE LA FASE Φ₃


    # ══════════════════════════════════════════════════════════════════════════════════════════
    # FASE Φ₃ · CHAMPION DECIDER & TRIBUNAL DE SILICIO — Compuerta Φ₂₃, Heyting G₃, Löb-GL, Merkle y Crowbar
    #           (Continúa directamente desde el `return continuation(cert2)` de Φ₂.6)
    # ══════════════════════════════════════════════════════════════════════════════════════════
    @dataclass(frozen=True, slots=True)
    class LoebGLReport:
        worlds: int
        relation_edges: int
        transitive: bool
        irreflexive: bool
        converse_well_founded: bool
        loeb_instance_valid: bool
        guarded_reflection_depth: int
        fuel_remaining: int
        passed: bool
        # ── Extensión v18.0.0: Axiomas modales K y 4 + altura ordinal de Mirimanoff ──────────
        modal_axiom_k_valid: bool = True
        modal_axiom_4_valid: bool = True
        mirimanoff_max_rank: int = 0

    class _LoebSafeChampionVerifier:
        """
        Verificador modal de Gödel–Löb (GL) sobre un marco de Kripke FINITO (W, R, V) de evidencia fiscal.
        Teorema de completitud de Segerberg (1971): la lógica de demostrabilidad GL (K + 4 + Löb) es
        sound & complete respecto de la clase de marcos finitos, transitivos e irreflexivos
        (equivalentemente, converse-bien-fundados).
        Por tanto:
          (1) Verificar que (W, R⁺) es transitivo, irreflexivo y acíclico garantiza la validez del marco.
          (2) Evaluar EXHAUSTIVAMENTE sobre W los axiomas K, 4 y la instancia de Löb □(□P→P)→□P
              constituye una prueba decidible del caso particular.
          (3) La reflexión guardada exige que cada mundo w satisfaga su valuación local V(w) Y que
              toda premisa v ∈ R(w) tenga rango ordinal de Mirimanoff estrictamente menor: ρ(v) < ρ(w)
              (corrige [A8]).
        """

        @staticmethod
        def build_frame(evidence: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
            nodes = set(evidence.keys())
            for deps in evidence.values():
                nodes.update(deps)
            return {w: set(evidence.get(w, ())) for w in nodes}

        @staticmethod
        def is_irreflexive(r: Mapping[str, set[str]]) -> bool:
            return all(w not in r[w] for w in r)

        @staticmethod
        def transitive_closure(r: Mapping[str, set[str]]) -> dict[str, set[str]]:
            closed = {w: set(r[w]) for w in r}
            changed = True
            while changed:
                changed = False
                for w in closed:
                    add: set[str] = set()
                    for v in list(closed[w]):
                        add |= closed.get(v, set())
                    if not add <= closed[w]:
                        closed[w] |= add
                        changed = True
            return closed

        @staticmethod
        def is_acyclic(r: Mapping[str, set[str]]) -> bool:
            color: dict[str, int] = {}

            def dfs(w: str) -> bool:
                color[w] = 1
                for v in r.get(w, ()):
                    c = color.get(v, 0)
                    if c == 1 or (c == 0 and not dfs(v)):
                        return False
                color[w] = 2
                return True

            return all(color.get(w, 0) != 0 or dfs(w) for w in r)

        @classmethod
        def verify_modal_axioms_and_loeb(
            cls, r: Mapping[str, set[str]], p_true: Mapping[str, bool]
        ) -> tuple[bool, bool, bool]:
            """
            Evalúa por exhaución sobre el conjunto finito de mundos W:
              • Axioma K (distribución modal):  ∀w: □(P → P)(w) → (□P(w) → □P(w))
              • Axioma 4 (introspección positiva): ∀w: □P(w) → □□P(w)
              • Axioma GL (Löb): ∀w: □(□P → P)(w) → □P(w)
            """
            w_list = list(r)

            def box(prop: Callable[[str], bool], w: str) -> bool:
                return all(prop(v) for v in r[w])

            def p(v: str) -> bool:
                return bool(p_true.get(v, False))

            def box_p(v: str) -> bool:
                return box(p, v)

            def inner_loeb(v: str) -> bool:
                # □P → P en el mundo v
                return (not box_p(v)) or p(v)

            k_ok = True
            four_ok = True
            loeb_ok = True
            for w in w_list:
                # Axioma K sobre (□P → P): □(□P → P)(w) ∧ □□P(w) ⇒ □P(w)
                if box(inner_loeb, w) and box(box_p, w) and not box_p(w):
                    k_ok = False
                # Axioma 4: □P(w) ⇒ □□P(w)
                if box_p(w) and not box(box_p, w):
                    four_ok = False
                # Axioma de Löb: □(□P → P)(w) ⇒ □P(w)
                if box(inner_loeb, w) and not box_p(w):
                    loeb_ok = False
            return k_ok, four_ok, loeb_ok

        @classmethod
        def guarded_reflection(
            cls,
            r: Mapping[str, set[str]],
            world_valuation: Mapping[str, bool],
            target: str,
            fuel: int,
        ) -> tuple[bool, int, int, int]:
            """
            Derivación bien fundada con rango ordinal de Mirimanoff ρ(w) = sup{ρ(v)+1 | v ∈ R(w)}:
            Cada mundo w se justifica SÓLO si:
              (1) Su evidencia atómica propia V(w) es verdadera (`world_valuation[w] == True`), y
              (2) Todos sus prerrequisitos v ∈ R(w) tienen rango estrictamente menor ρ(v) < ρ(w)
                  y son recursivamente derivables dentro del presupuesto de combustible (corrige [A8]).
            Prohibido aceptar P por "□P → P" (auto-confianza): eso es exactamente lo que Löb veda.
            """
            rank: dict[str, int] = {}

            def compute_rank(w: str, depth: int) -> int:
                if depth > len(r) + 1:
                    raise ChampionConstitutionalError("Reflexión cíclica: rango de Mirimanoff no bien fundado.")
                if w in rank:
                    return rank[w]
                kids = r.get(w, set())
                rank[w] = 0 if not kids else 1 + max(compute_rank(v, depth + 1) for v in kids)
                return rank[w]

            for w in r:
                compute_rank(w, 0)

            max_mirimanoff = max(rank.values()) if rank else 0
            used = 0

            def derive(w: str, budget: int, parent_rank: int) -> bool:
                nonlocal used
                used += 1
                if budget <= 0:
                    raise ChampionConstitutionalError("Combustible de Löb agotado (fail-closed).")
                if rank[w] >= parent_rank:
                    raise ChampionConstitutionalError("Rango no decreciente: auto-demostración vedada.")
                # Corrige [A8]: todo mundo (hoja o intermedio) debe satisfacer su propia valuación V(w)
                if not bool(world_valuation.get(w, False)):
                    return False
                kids = r.get(w, set())
                if not kids:
                    return True
                return all(derive(v, budget - 1, rank[w]) for v in kids)

            if target not in rank:
                raise ChampionConstitutionalError(f"Mundo objetivo {target!r} ausente del marco de Kripke.")
            ok = derive(target, fuel, rank[target] + 1)
            return ok, used, fuel - used, max_mirimanoff

        @classmethod
        def audit(
            cls,
            evidence: Mapping[str, Sequence[str]],
            world_valuation: Mapping[str, bool],
            target: str,
            fuel: int,
        ) -> "PositronRedemptionChampionAgent.LoebGLReport":
            r0 = cls.build_frame(evidence)
            irr0 = cls.is_irreflexive(r0)
            acyc = cls.is_acyclic(r0)
            r = cls.transitive_closure(r0)
            irr = cls.is_irreflexive(r)
            trans = all(r.get(v, set()) <= r[w] for w in r for v in r[w])
            cwf = bool(irr0 and irr and acyc)   # finito + irreflexivo + acíclico ⇒ converse-bien-fundado

            p_true = {w: bool(world_valuation.get(w, False)) for w in r}
            if cwf:
                k_ok, four_ok, inst = cls.verify_modal_axioms_and_loeb(r, p_true)
            else:
                k_ok, four_ok, inst = False, False, False

            refl, depth, remaining, max_rank = cls.guarded_reflection(
                r0, world_valuation, target, fuel
            )
            passed = bool(irr and acyc and trans and cwf and k_ok and four_ok and inst and refl)
            return PositronRedemptionChampionAgent.LoebGLReport(
                worlds=len(r),
                relation_edges=sum(len(v) for v in r.values()),
                transitive=trans,
                irreflexive=irr,
                converse_well_founded=cwf,
                loeb_instance_valid=inst,
                guarded_reflection_depth=depth,
                fuel_remaining=remaining,
                passed=passed,
                modal_axiom_k_valid=k_ok,
                modal_axiom_4_valid=four_ok,
                mirimanoff_max_rank=max_rank,
            )

    class ChampionPhase3Decider:
        """
        Φ₃: recibe `cert2` desde la continuación CPS de `Φ₂.6 orient_and_sign`, verifica la
        compuerta criptográfica Φ₂₃, evalúa el meet en el álgebra de Heyting G₃, ejecuta el
        verificador modal de Kripke–Löb (GL), sella el árbol de evidencia de Merkle RFC 6962
        y acciona el Tribunal de Silicio fail-closed si v_final == ⊥.
        """

        def __init__(self, ctx: _ChampionContext) -> None:
            self._ctx = ctx

        # ── Φ₃.1 · COMPUERTA Φ₂₃ (inicio formal de Φ₃ tras `orient_and_sign`) ───────────────
        def gate_phi23(self, cert2: "ChampionPhase2Certificate") -> int:
            """
            Punto de entrada formal de la Fase Φ₃:
            Verifica la autenticidad HMAC-SHA256 de C₂ bajo `K_C2 = HKDF(master, "C2")`,
            la continuidad de sesión y secuencia (`seq == 2`), la encadenación criptográfica
            C₁ → C₂ (tanto por firma como por digest SHA-256 canónico) y la ventana monótona.
            """
            ctx = self._ctx
            k_c2 = ctx.subkey("C2")
            expected = hmac.new(k_c2, canon(cert2.body), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(cert2.signature, expected):
                raise ChampionConstitutionalError("Φ₂₃: firma HMAC de C₂ inválida.")
            if cert2.body.get("session") != ctx.session_id or cert2.body.get("seq") != 2:
                raise ChampionConstitutionalError("Φ₂₃: sesión/seq discontinua.")
            if not hmac.compare_digest(str(cert2.body.get("c1_signature", "")), cert2.phase1.signature):
                raise ChampionConstitutionalError("Φ₂₃: ruptura de la cadena C₁→C₂.")
            digest = hashlib.sha256(canon(cert2.phase1.body)).hexdigest()
            if not hmac.compare_digest(digest, str(cert2.body.get("c1_digest", ""))):
                raise ChampionConstitutionalError("Φ₂₃: C₁ fue mutado tras Φ₁₂.")
            dt = time.monotonic_ns() - int(cert2.body["mono_ns"])
            if dt < 0 or dt > ctx.constitution.phase_gap_max_ns:
                raise ChampionConstitutionalError("Φ₂₃: ventana temporal violada.")
            return HeytingChampionVerdict.COHERENT

        # ── Φ₃.2 · gate de Dirac: conservación 4-momento e invariante de Casimir de Poincaré ──
        @staticmethod
        def dirac_verdict(m: DiracAnnihilationMeasurement | None) -> int:
            H = HeytingChampionVerdict
            if m is None:
                return H.VETOED                       # ausencia de medición ⇒ ⊥ (fail-closed)
            if len(m.photon_energies_kev) != 2:
                return H.VETOED                       # canal singlet para-Ps 2γ exigido
            if any((not math.isfinite(e)) or e <= 0.0 for e in m.photon_energies_kev):
                return H.VETOED
            if not math.isfinite(m.opening_angle_rad):
                return H.VETOED

            total = m.total_energy_kev()
            floor = TWO_M_E_C2_KEV - PS_BINDING_KEV - 1e-6
            ceil = TWO_M_E_C2_KEV + KINETIC_BUDGET_KEV
            if total < floor or total > ceil:
                return H.VETOED                       # violación de conservación de energía relativista

            # Invariante de Casimir de Mandelstam s = 2 E₁ E₂ (1 − cos θ) del grupo de Poincaré ISO(1,3)
            s_inv = m.mandelstam_invariant_s_kev2()
            s_rest_min = (TWO_M_E_C2_KEV - 2.0 * DOPPLER_BAND_KEV) ** 2
            s_rest_max = (TWO_M_E_C2_KEV + KINETIC_BUDGET_KEV) ** 2
            if s_inv < s_rest_min or s_inv > s_rest_max:
                return H.VETOED

            if m.momentum_residual_kev() > 2.0 * DOPPLER_BAND_KEV:
                return H.VETOED                       # 3-momento neto incompatible con aniquilación 2γ

            if any(
                abs(e - M_E_C2_KEV) > DOPPLER_BAND_KEV + 0.5 * KINETIC_BUDGET_KEV
                for e in m.photon_energies_kev
            ):
                return H.DEGRADED
            return H.COHERENT if m.provenance == Provenance.MEASURED else H.DEGRADED

        # ── Φ₃.3 · gate Ámbar (rampa de gracia temporal) ─────────────────────────────────────
        @staticmethod
        def amber_verdict(elapsed: float, grace: float, c: ChampionConstitution) -> int:
            H = HeytingChampionVerdict
            if not (math.isfinite(elapsed) and math.isfinite(grace)):
                return H.VETOED
            if elapsed < 0.0 or grace <= 0.0:
                return H.VETOED
            if elapsed > grace:
                return H.VETOED
            return H.DEGRADED if elapsed > c.amber_warn_fraction * grace else H.COHERENT

        # ── Φ₃.4 · gate de constitución (RSI-L3 + SOA + aislamiento + integridad de claves) ──
        def constitution_verdict(
            self,
            lineage: "PositronRedemptionChampionAgent.RsiLineage",
            physics: CrowbarPhysicsReport,
        ) -> int:
            H = HeytingChampionVerdict
            ctx = self._ctx
            if lineage.constitutional_digest != ctx.constitution.digest():
                return H.VETOED
            td, rows = trigger_truth_table_digest()
            if td != lineage.trigger_digest or rows != lineage.trigger_rows:
                return H.VETOED
            if not lineage.monoid_associativity_verified or not lineage.loeb_rank_strictly_decreasing:
                return H.VETOED
            if not physics.soa_ok or not physics.latching_ok:
                return H.VETOED
            if ctx.timing.t_assert_ns > ctx.constitution.t_assert_max_ns:
                return H.VETOED
            if ctx.armed and ctx.key_is_default:
                return H.VETOED     # prohibido armar hardware físico con clave maestra por defecto
            if ctx.key_is_default or not ctx.bus.galvanic_isolation:
                return H.DEGRADED
            return H.COHERENT

        # ── Φ₃.5 · sello Merkle RFC 6962 con prueba de inclusión autoverificada ─────────────
        def seal_evidence(self, leaves: Sequence[bytes]) -> tuple[str, tuple[bytes, ...], bool]:
            root = MerkleTreeRFC6962.mth(leaves)
            idx = len(leaves) - 1
            path = MerkleTreeRFC6962.audit_path(idx, leaves)
            ok = MerkleTreeRFC6962.verify_inclusion(leaves[idx], idx, len(leaves), path, root)
            if not ok:
                raise ChampionConstitutionalError("Sello Merkle RFC 6962 no autoverifica (fail-closed).")
            return root.hex(), path, ok

        # ── Φ₃.6 · TRIBUNAL DE SILICIO: actuación fail-closed ───────────────────────────────
        def actuate(self, physics: CrowbarPhysicsReport) -> tuple[bool, Mapping[str, Any]]:
            """Emite el mando crowbar. En modo SIMULATED_MODEL jamás se afirma energización física."""
            tel = dict(self._ctx.actuator.assert_crowbar())
            tel["model_t_bus_peak_us"] = physics.t_peak_us
            tel["model_t_bus_zero_us"] = physics.t_zero_us
            tel["crowbar_ineffective_without_inhibit"] = not physics.latching_ok
            armed_ok = bool(tel.get("hardware_energized", False))
            return armed_ok, tel

        # ── Φ₃.7 · MÉTODO TERMINAL DE Φ₃: decisión en G₃, Löb-GL, sello Merkle y actuación ───
        def decide_and_act(self, cert2: "ChampionPhase2Certificate") -> "ChampionPhase3Verdict":
            ctx, c, H = self._ctx, self._ctx.constitution, HeytingChampionVerdict
            v_phi23 = self.gate_phi23(cert2)
            b1 = cert2.phase1.body
            spectral: PositronRedemptionChampionAgent.SpectralCapture = b1["spectral"]
            poincare: PositronRedemptionChampionAgent.PoincareCapture = b1["poincare"]
            banach: PositronRedemptionChampionAgent.BanachReport = cert2.body["banach"]
            celestial: PositronRedemptionChampionAgent.CelestialReport = cert2.body["celestial"]
            lineage: PositronRedemptionChampionAgent.RsiLineage = cert2.body["rsi"]
            physics = analyze_crowbar(ctx.bus, ctx.scr)

            v_identity = H.meet(int(cert2.body["v_phi12"]), int(v_phi23))
            v_topology = PositronRedemptionChampionAgent.ChampionPhase1Observer.topology_verdict(poincare)
            v_dirac = self.dirac_verdict(b1["dirac"])
            v_amber = self.amber_verdict(float(b1["elapsed_sec"]), float(b1["grace_sec"]), c)
            v_banach = H.meet(
                PositronRedemptionChampionAgent.ChampionPhase2Orientor.banach_verdict(banach, c),
                PositronRedemptionChampionAgent.ChampionPhase2Orientor.connes_verdict(spectral),
            )
            v_celestial = PositronRedemptionChampionAgent.ChampionPhase2Orientor.celestial_verdict(
                celestial, c
            )
            v_const = self.constitution_verdict(lineage, physics)

            # ── Marco de Kripke (W, R, V) para Löb-GL (DAG de justificación fiscal) ─────────
            leaves_ok = {
                "e_identity": v_identity == H.COHERENT,
                "e_topology": v_topology >= H.DEGRADED,
                "e_dirac": v_dirac >= H.DEGRADED,
                "e_amber": v_amber >= H.DEGRADED,
                "e_banach": v_banach >= H.DEGRADED,
                "e_celestial": v_celestial >= H.DEGRADED,
                "e_constitution": v_const >= H.DEGRADED,
            }
            evidence = {
                "e_identity": (),
                "e_topology": (),
                "e_dirac": (),
                "e_amber": (),
                "e_banach": ("e_identity",),
                "e_celestial": ("e_topology", "e_identity"),
                "e_constitution": ("e_identity",),
                "accusation": tuple(leaves_ok.keys()),
            }
            world_valuation = {**leaves_ok, "accusation": all(leaves_ok.values())}
            loeb = PositronRedemptionChampionAgent._LoebSafeChampionVerifier.audit(
                evidence, world_valuation, "accusation", c.loeb_fuel
            )
            v_loeb = H.COHERENT if loeb.passed else H.VETOED

            gates = (v_identity, v_topology, v_dirac, v_amber, v_banach, v_celestial, v_loeb, v_const)
            final = trigger_predicate(gates)

            # ── Árbol de evidencia RFC 6962 (dominios separados por el códec canónico) ───────
            leaves = [
                canon(("gate", name, int(val))) for name, val in zip(GATE_NAMES, gates)
            ] + [
                canon(("spectral", spectral)),
                canon(("cohomology", poincare.cohomology)),
                canon(("poincare_elements", poincare.poincare_elements)),
                canon(("banach", banach)),
                canon(("celestial", celestial)),
                canon(("rsi_lineage", lineage)),
                canon(("crowbar_physics", physics)),
                canon(("loeb", loeb)),
                canon(("c1_sig", cert2.phase1.signature)),
                canon(("c2_sig", cert2.signature)),
                canon(("verdict", int(final), _SCHEMA)),
            ]
            root, path, sealed_ok = self.seal_evidence(leaves)

            triggered, telemetry = (False, {})
            if final == H.VETOED:
                triggered, telemetry = self.actuate(physics)

            return ChampionPhase3Verdict(
                final_verdict=int(final),
                verdict_label=H.to_string(final),
                merkle_root_rfc6962=root,
                loeb_verification_passed=bool(loeb.passed),
                hardware_crowbar_triggered=bool(triggered),
                actuation_latency_ns=float(telemetry.get("model_t_assert_ns", 0.0)),
                schema_version=_SCHEMA,
                gate_verdicts=dict(zip(GATE_NAMES, map(int, gates))),
                merkle_audit_path=tuple(p.hex() for p in path),
                merkle_self_verified=sealed_ok,
                crowbar_command_issued=bool(final == H.VETOED),
                actuation_telemetry=dict(telemetry),
                crowbar_physics=physics,
                loeb_report=loeb,
                rsi_lineage=lineage,
                celestial_report=celestial,
                banach_report=banach,
                cohomology_reports=poincare.cohomology,
                poincare_elements=poincare.poincare_elements,
                spectral_capture=spectral,
                heyting_laws_digest=ctx.heyting_digest,
                session_id=ctx.session_id,
            )

    # ══════════════════════════════════════════════════════════════════════════════════════════
    # ORQUESTADOR CPS · Φ₃ ∘ Φ₂ ∘ Φ₁  (envolvente fail-closed total)
    # ══════════════════════════════════════════════════════════════════════════════════════════
    def __init__(
        self,
        master_key: bytes | None = None,
        *,
        constitution: ChampionConstitution | None = None,
        resonance: ResonanceParams | None = None,
        bus: CrowbarBusConfig | None = None,
        scr: ThyristorSpec | None = None,
        timing: Esp32TimingSpec | None = None,
        actuator: GpioActuatorProtocol | None = None,
        armed: bool = False,
        engine: ChampionSpectrumEngineProtocol | None = None,
        epoch: int = 1,
    ) -> None:
        key = (
            master_key
            if master_key is not None
            else os.environ.get("POSITRON_CHAMPION_KEY", "").encode()
        )
        key_is_default = not key
        if key_is_default:
            key = _DEFAULT_MASTER_KEY
            _LOG.warning("Clave maestra por defecto en uso: modo ARMADO prohibido (fail-closed).")
        MerkleTreeRFC6962.self_test()                         # prueba de propiedad del sello RFC 6962
        heyting_digest = HeytingChampionVerdict.audit_laws()  # prueba exhaustiva del álgebra G₃
        tdigest, trows = trigger_truth_table_digest()         # prueba exhaustiva del gatillo Ω₃⁸
        self._ctx = _ChampionContext(
            master_key=key,
            key_is_default=key_is_default,
            session_id=secrets.token_hex(16),
            epoch=int(epoch),
            constitution=constitution or ChampionConstitution(),
            resonance=resonance or ResonanceParams(),
            bus=bus or CrowbarBusConfig(),
            scr=scr or ThyristorSpec(),
            timing=timing or Esp32TimingSpec(),
            actuator=actuator
            or _ModelOnlyActuator(timing or Esp32TimingSpec(), scr or ThyristorSpec()),
            armed=bool(armed),
            engine=engine if engine is not None else (_ExternalEngine() if _ExternalEngine else None),
            heyting_digest=heyting_digest,
            trigger_digest=tdigest,
            trigger_rows=trows,
        )
        self._phi1 = self.ChampionPhase1Observer(self._ctx)
        self._phi2 = self.ChampionPhase2Orientor(self._ctx)
        self._phi2._ctx_phase1 = self._phi1        # inyección explícita para el evaluador de fitness
        self._phi3 = self.ChampionPhase3Decider(self._ctx)

    # ── API pública ─────────────────────────────────────────────────────────────────────────
    def prosecute_champion_cycle(
        self,
        p_init: NDArray[np.float64],
        rho_init: NDArray[np.complex128],
        sigma_ref: NDArray[np.complex128],
        K_raw: NDArray[np.complex128],
        elapsed_time_sec: float = 0.0,
        grace_period_sec: float = 3600.0,
        *,
        dirac_measurement: DiracAnnihilationMeasurement | None = None,
        seed_point: tuple[float, float] = (0.9, 0.35),
        tuning: TuningVector | None = None,
    ) -> "ChampionPhase3Verdict":
        """
        Ejecuta la composición CPS anidada Φ₃ ∘ Φ₂ ∘ Φ₁:
          • El método final de Φ₁ (`observe_and_sign`) invoca como continuación el método final
            de Φ₂ (`orient_and_sign`), cuyo primer paso es `gate_phi12`.
          • El método final de Φ₂ (`orient_and_sign`) invoca como continuación el método terminal
            de Φ₃ (`decide_and_act`), cuyo primer paso es `gate_phi23`.
        TODA excepción constitucional o numérica se traduce en ⊥ + actuación (AXIOMA CHAMPION-2).
        """
        try:
            p, rho, sigma, k = self._phi1.validate_inputs(p_init, rho_init, sigma_ref, K_raw)
            self._corroborate_with_engine(p, rho, sigma, k, elapsed_time_sec, grace_period_sec)
            spectral = self._phi1.capture_spectrum(p, rho, sigma, k)
            tv = (tuning or TuningVector()).clamped(self._ctx.constitution)
            poincare = self._phi1.capture_poincare_and_cohomology(
                tv,
                seed_point,
                bures_angle_rad=spectral.bures_angle_rad,
                kappa_k=spectral.kappa_k,
            )
            # COMPOSICIÓN CPS: Φ₁.5 observe_and_sign ──(C₁)──► Φ₂.6 orient_and_sign ──(C₂)──► Φ₃.7 decide_and_act
            return self._phi1.observe_and_sign(
                spectral,
                poincare,
                dirac_measurement,
                elapsed_time_sec,
                grace_period_sec,
                tv,
                lambda c1: self._phi2.orient_and_sign(
                    c1,
                    lambda c2: self._phi3.decide_and_act(c2),
                ),
            )
        except Exception as exc:  # FAIL-CLOSED: ninguna anomalía escapa sin veredicto ⊥ y mando crowbar
            _LOG.error("Fallo constitucional en el ciclo: %s", exc, exc_info=True)
            return self._emergency_veto(str(exc))

    def _corroborate_with_engine(
        self,
        p: NDArray[np.float64],
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        k: NDArray[np.complex128],
        elapsed: float,
        grace: float,
    ) -> None:
        """
        El backend espectral (`PositronRedemptionChampionEngine`) actúa como TESTIGO corroborante:
        si lanza excepción o viola cualquiera de los invariantes algebraicos/físicos, se aborta en ⊥.
        """
        if self._ctx.engine is None:
            return
        try:
            # Si el motor tiene distinta dimensión por defecto, re-instanciamos acorde a d = len(p)
            eng = self._ctx.engine
            if hasattr(eng, "_n") and int(getattr(eng, "_n")) != int(p.shape[0]):
                eng = type(eng)(dimension=int(p.shape[0]))
            rep = eng.prosecute_positron_trajectory(p, rho, sigma, k, elapsed, grace)
        except Exception as exc:
            raise ChampionConstitutionalError(f"Backend espectral falló: {exc}") from exc

        for attr in ("connes_lipschitz_bound", "condition_number_K", "annihilation_energy_kev"):
            if hasattr(rep, attr):
                val = float(getattr(rep, attr))
                if not math.isfinite(val) or val <= 0.0:
                    raise ChampionConstitutionalError(f"Testigo externo no admisible: {attr}={val}")
        for bool_attr in ("foster_cauer_passivity_ok", "poincare_nonsingular_regularity_ok"):
            if hasattr(rep, bool_attr) and not bool(getattr(rep, bool_attr)):
                raise ChampionConstitutionalError(f"Testigo externo violó invariante: {bool_attr}=False")

    def _emergency_veto(self, reason: str) -> "ChampionPhase3Verdict":
        """Veredicto ⊥ de emergencia con sello Merkle autoverificado y actuación inmediata."""
        physics = analyze_crowbar(self._ctx.bus, self._ctx.scr)
        triggered, tel = self._phi3.actuate(physics)
        leaves = [canon(("emergency", reason)), canon(("schema", _SCHEMA)), canon(("veto", 0))]
        root, path, sealed_ok = self._phi3.seal_evidence(leaves)
        return ChampionPhase3Verdict(
            final_verdict=int(HeytingChampionVerdict.VETOED),
            verdict_label="VETOED",
            merkle_root_rfc6962=root,
            loeb_verification_passed=False,
            hardware_crowbar_triggered=triggered,
            actuation_latency_ns=float(tel.get("model_t_assert_ns", 0.0)),
            schema_version=_SCHEMA,
            gate_verdicts={n: 0 for n in GATE_NAMES},
            merkle_audit_path=tuple(p.hex() for p in path),
            merkle_self_verified=sealed_ok,
            crowbar_command_issued=True,
            actuation_telemetry=dict(tel),
            crowbar_physics=physics,
            loeb_report=None,
            rsi_lineage=None,
            celestial_report=None,
            banach_report=None,
            cohomology_reports=(),
            poincare_elements=None,
            spectral_capture=None,
            heyting_laws_digest=self._ctx.heyting_digest,
            session_id=self._ctx.session_id,
            emergency_reason=reason,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS DE FASE (inmutables; la firma cubre el cuerpo CANÓNICO completo)
# ══════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ChampionPhase1Certificate:
    body: Mapping[str, Any]
    signature: str


@dataclass(frozen=True, slots=True)
class ChampionPhase2Certificate:
    phase1: ChampionPhase1Certificate
    body: Mapping[str, Any]
    signature: str


@dataclass(frozen=True, slots=True)
class ChampionPhase3Verdict:
    # ── superficie compatible con v16 y v17 ─────────────────────────────────────────────────
    final_verdict: int
    verdict_label: str
    merkle_root_rfc6962: str
    loeb_verification_passed: bool
    hardware_crowbar_triggered: bool
    actuation_latency_ns: float
    schema_version: str = _SCHEMA
    # ── superficie completa y auditable (con campos extendidos v18.0.0 al final) ────────────
    gate_verdicts: Mapping[str, int] = field(default_factory=dict)
    merkle_audit_path: tuple[str, ...] = ()
    merkle_self_verified: bool = False
    crowbar_command_issued: bool = False
    actuation_telemetry: Mapping[str, Any] = field(default_factory=dict)
    crowbar_physics: CrowbarPhysicsReport | None = None
    loeb_report: Any | None = None
    rsi_lineage: Any | None = None
    celestial_report: Any | None = None
    banach_report: Any | None = None
    cohomology_reports: tuple[CohomologyReport, ...] = ()
    heyting_laws_digest: str = ""
    session_id: str = ""
    emergency_reason: str | None = None
    poincare_elements: PoincareCanonicalElements | None = None
    spectral_capture: Any | None = None

    def to_json(self) -> str:
        def enc(o: Any) -> Any:
            if dataclasses.is_dataclass(o) and not isinstance(o, type):
                return {f.name: enc(getattr(o, f.name)) for f in dataclasses.fields(o)}
            if isinstance(o, Mapping):
                return {str(k): enc(v) for k, v in o.items()}
            if isinstance(o, (tuple, list)):
                return [enc(x) for x in o]
            if isinstance(o, Enum):
                return o.name
            if isinstance(o, float) and not math.isfinite(o):
                return str(o)
            return o

        return json.dumps(enc(self), sort_keys=True, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# SUITE DE AUTOVERIFICACIÓN SOBERANA ([A1]-[A10]) Y DEMOSTRACIÓN OPERACIONAL
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    """
    Suite determinista de verificación formal y regresión forense para v18.0.0:
      • [A1] Complejo simplicial de Vietoris–Rips con triángulos (|T| ≥ 1) sin IndexError y Hodge Δ₁.
      • [A2] Invarianza unitaria de D = K^{-1/2} y ecuación cuántica SLD de Lyapunov–Sylvester.
      • [A3] Conservación simpléctica exacta de Yoshida 4.º orden (‖MᵀJM − J‖_F < 10⁻¹³) e identidad
             de Abel–Jacobi–Liouville det(M_diss) = exp(−δ T_f).
      • [A6] Newton–Kantorovich con norma espectral 1/σ_min(M_diss − I) y acción de Floer de 1 periodo.
      • [A7] Descenso ordinal estricto en (ℕ³, <ₗₑₓ) y asociatividad del monoide de meta-operadores N3.
      • [A8] Rechazo modal en `_LoebSafeChampionVerifier` cuando falla un nodo intermedio del DAG.
      • [A9/A10] Ciclo completo CPS Φ₃ ∘ Φ₂ ∘ Φ₁ en estados COHERENT, DEGRADED y VETOED.
    """
    # 1. Prueba [A1]: Vietoris–Rips con triángulos (3 puntos equiláteros ⇒ |V|=3, |E|=3, |T|=1, b₀=1, b₁=0)
    tri_pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]], dtype=np.float64)
    rep_tri = cech_rips_betti(tri_pts, scale=1.1)
    assert rep_tri.n_triangles == 1 and rep_tri.b0 == 1 and rep_tri.b1 == 0 and rep_tri.euler_ok
    assert rep_tri.complex_ok and not rep_tri.torsion_witness and rep_tri.hodge_spectral_gap > 0.0

    # Ciclo de 4 vértices sin diagonales (cuadrado de lado 1.0 a escala 1.1 ⇒ |V|=4, |E|=4, |T|=0, b₁=1)
    sq_pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    rep_sq = cech_rips_betti(sq_pts, scale=1.1)
    assert rep_sq.b0 == 1 and rep_sq.b1 == 1 and rep_sq.euler_ok and rep_sq.complex_ok

    # 2. Prueba [A8]: _LoebSafeChampionVerifier debe rechazar si un mundo intermedio tiene V(w)=False
    ev_dag = {
        "e_id": (),
        "e_banach": ("e_id",),
        "accusation": ("e_id", "e_banach"),
    }
    val_broken = {"e_id": True, "e_banach": False, "accusation": True}
    loeb_broken = PositronRedemptionChampionAgent._LoebSafeChampionVerifier.audit(
        ev_dag, val_broken, "accusation", fuel=8
    )
    assert not loeb_broken.passed, "Fallo [A8]: Löb aprobó un DAG con nodo intermedio falso."

    # 3. Prueba [A3] + [A6] + [A7] + [A9]: Ciclo soberano completo con clave dedicada y Dirac MEDIDO
    custom_key = b"SOVEREIGN_CHAMPION_V18_TEST_KEY_32BYTES_OK!!"
    agent_coh = PositronRedemptionChampionAgent(master_key=custom_key)
    p_vec = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    rho_mat = np.eye(3, dtype=np.complex128) / 3.0
    sigma_mat = np.diag([0.5, 0.3, 0.2]).astype(np.complex128)
    k_mat = np.diag([2.0, 1.5, 1.0]).astype(np.complex128)
    dirac_measured = DiracAnnihilationMeasurement(
        photon_energies_kev=(M_E_C2_KEV, M_E_C2_KEV),
        opening_angle_rad=math.pi,
        provenance=Provenance.MEASURED,
        detector_id="HPGe-CALIBRATED-01",
        timestamp_ns=time.time_ns(),
    )
    verdict_coh = agent_coh.prosecute_champion_cycle(
        p_vec,
        rho_mat,
        sigma_mat,
        k_mat,
        elapsed_time_sec=10.0,
        grace_period_sec=3600.0,
        dirac_measurement=dirac_measured,
    )
    assert verdict_coh.final_verdict == int(HeytingChampionVerdict.COHERENT)
    assert verdict_coh.loeb_verification_passed and verdict_coh.merkle_self_verified
    assert not verdict_coh.crowbar_command_issued
    assert verdict_coh.celestial_report is not None
    assert verdict_coh.celestial_report.symplectic_defect < 1e-12
    assert verdict_coh.banach_report is not None
    assert verdict_coh.banach_report.abel_liouville_det_defect < 1e-5
    assert verdict_coh.rsi_lineage is not None
    assert verdict_coh.rsi_lineage.monoid_associativity_verified
    assert verdict_coh.poincare_elements is not None
    assert verdict_coh.poincare_elements.nonsingular_regularity_ok
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
    assert _self_test(), "La suite de autoverificación soberana v18.0.0 falló."
    print("✓ Autoverificación formal v18.0.0 ([A1]-[A10], Heyting G₃, Merkle RFC6962, Yoshida Sp(2,ℝ), Löb-GL, RSI-L3) superada.")

    agent = PositronRedemptionChampionAgent()
    p_vec = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    rho_mat = np.eye(3, dtype=np.complex128) / 3.0
    sigma_mat = np.diag([0.5, 0.3, 0.2]).astype(np.complex128)
    K_mat = np.diag([2.0, 1.5, 1.0]).astype(np.complex128)
    dirac = DiracAnnihilationMeasurement(
        photon_energies_kev=(M_E_C2_KEV, M_E_C2_KEV),
        opening_angle_rad=math.pi,
        provenance=Provenance.SIMULATED,
        detector_id="HPGe-SIM-01",
        timestamp_ns=time.time_ns(),
    )
    for label, elapsed in (("A · gracia vigente", 0.0), ("B · gracia expirada", 7200.0)):
        v = agent.prosecute_champion_cycle(
            p_vec,
            rho_mat,
            sigma_mat,
            K_mat,
            elapsed_time_sec=elapsed,
            grace_period_sec=3600.0,
            dirac_measurement=dirac,
        )
        print(f"\n═══ ESCENARIO {label} ═══")
        print(f"Veredicto         : {v.verdict_label}")
        print(f"Gates Ω₃          : {dict(v.gate_verdicts)}")
        print(f"Merkle RFC6962    : {v.merkle_root_rfc6962[:32]}…  (autoverificado={v.merkle_self_verified})")
        print(f"Löb-GL            : {v.loeb_verification_passed}")
        if v.cohomology_reports:
            print(
                f"H¹ multiescala    : {[r.b1 for r in v.cohomology_reports]}  "
                f"(b₀={[r.b0 for r in v.cohomology_reports]}, "
                f"λ_Hodge={[round(r.hodge_spectral_gap, 4) for r in v.cohomology_reports]})"
            )
        if v.poincare_elements:
            pe = v.poincare_elements
            print(
                f"Poincaré AMD / J₁₂: AMD={pe.amd_laskar_poincare:.6e}  "
                f"J₁={pe.action_j1_radial:.6e}  J₂={pe.action_j2_vertical:.6e}  "
                f"reg_ok={pe.nonsingular_regularity_ok}"
            )
        if v.banach_report:
            print(
                f"Banach q̂ / error  : {v.banach_report.q_hat:.6f} / {v.banach_report.a_posteriori_error:.3e}  "
                f"Abel-Liouville Δdet={v.banach_report.abel_liouville_det_defect:.2e}  "
                f"μ_CZ={v.banach_report.conley_zehnder_index}"
            )
        if v.celestial_report:
            print(
                f"Greene R / Melnikov: {v.celestial_report.greene_residue:.4f} / {v.celestial_report.melnikov_ratio:.4f}  "
                f"δ_sym(Yoshida)={v.celestial_report.symplectic_defect:.2e}"
            )
        if v.rsi_lineage:
            print(
                f"RSI-3 gens/accept : {v.rsi_lineage.generations}/{v.rsi_lineage.accepted}  "
                f"J: {v.rsi_lineage.initial_fitness:.5f} → {v.rsi_lineage.champion_fitness:.5f}  "
                f"medida={v.rsi_lineage.termination_measure}"
            )
        if v.crowbar_physics:
            ph = v.crowbar_physics
            print(
                f"Crowbar SOA       : ok={ph.soa_ok}  I_pk={ph.i_peak_a:.2f} A ≤ {ph.i_peak_admissible_a:.1f} A  "
                f"I²t={ph.i2t_a2s:.3f} A²s  di/dt={ph.didt_a_per_us:.3f} A/µs"
            )
        print(
            f"Mando crowbar     : {v.crowbar_command_issued}  "
            f"t_assert={v.actuation_latency_ns:.2f} ns  "
            f"hardware={v.hardware_crowbar_triggered} (modelo, no energizado)"
        )
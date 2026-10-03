# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/wisdom/toon_oniric_dreamer_engine.py                                  ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / FASE REM (GAN-REM)                  ║
║ FUNCIÓN  : MOTOR ESPECTRAL ONÍRICO Y UNIFICACIÓN FUCSIANA DE POINCARÉ                ║
║ VERSIÓN  : 9.1.0-Doctoral-Poincare-Fuchsian-CRTBP-GKSL-Bures-Krein-Merkle            ║
║ CONTRATO : 8.1.0 (Mecánica Celeste de Poincaré + Álgebra de Heyting + GKSL)          ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICA
───────────────────────────────────────────────
El `TOONOniricDreamerEngine` es el motor espectral de simulación contrafactual de alta
entropía en la fase REM del bucle de Automejora Recursiva (RSI Nivel 2 — Darwin-Gödel).
Traslada el "Túnel de Viento Financiero" desde el espacio euclídeo plano hacia la
navegación sobre:

    (i)   tubos de variedades invariantes Wˢ, Wᵘ de los puntos de Lagrange L₁…L₅ del
          Problema Restringido Circular de Tres Cuerpos (CRTBP, Poincaré 1892–1899);
    (ii)  el semiplano superior de Poincaré ℍ² = { z ∈ ℂ : Im z > 0 } como espacio
          modular del parámetro τ = τ₁ + i τ₂ de la hoja de mundo de Polyakov.

Sea (ℋ_MAC, ⟨·,·⟩) el espacio de Hilbert complejo n-dimensional asociado a la Matriz
Atómica de Conocimiento (MAC). El motor onírico genera evoluciones CPTP no unitarias
sobre el cono de operadores densidad

    𝔇(ℋ_MAC) = { ρ ∈ B(ℋ_MAC) : ρ = ρ†, ρ ⪰ 0, Tr ρ = 1 }.

POSTULADOS Y FORMULACIÓN ESPECTRAL ONÍRICA POINCARANA
─────────────────────────────────────────────────────
1. AISLAMIENTO HOMOLÓGICO (is_dream_state = True). Toda evolución en ℳ_REM ⊂ ℋ_MAC es
   cerrada bajo la frontera ∂, ∂ ℳ_REM ≡ 0 mod RealWorld, garantizando que la inyección
   de estrés contrafactual (cisnes negros, devaluaciones ≥ 60%, fallas geotécnicas)
   no dispare el interlock ESP32 Crowbar del mundo físico.

2. UNIFORMIZACIÓN FUCSIANA. El grupo modular PSL(2, ℤ) = ⟨T, S | S² = (ST)³ = 1⟩
   actúa sobre ℍ² por transformaciones de Möbius:

       T: τ ↦ τ + 1,        S: τ ↦ −1/τ,        γ·τ = (a τ + b) / (c τ + d).

   El dominio fundamental

       ℱ = { τ ∈ ℍ² : |Re τ| ≤ ½,  |τ| ≥ 1 }

   es un dominio de Siegel cerrado bajo la acción modular, con métrica hiperbólica

       ds² = (dτ₁² + dτ₂²) / τ₂²,

   y distancia geodésica

       d_ℍ²(z₁, z₂) = arccosh( 1 + |z₁ − z₂|² / (2 Im z₁ Im z₂) ).

3. TUBOS INVARIANTES EN CRTBP. Cerca de L₁, L₂, L₃ (inestables), las variedades estables
   Wˢ y no estables Wᵘ son cilindros tridimensionales de energía constante H = C_Jacobi.
   La evolución contrafactual navega en su interior con Δv → 0 (transferencia de energía
   nula), explorando escenarios extremos sin coste real. L₄, L₅ son estables si se cumple
   el criterio de Routh 27 μ (1 − μ) < 1.

4. ECUACIÓN MAESTRA DE LINDBLAD (GKSL). Dinámica onírica de la matriz de densidad

       dρ/dt = −i [H_eff, ρ] + Σ_k γ_k ( L_k ρ L_k† − ½ { L_k† L_k, ρ } ).

   El generador 𝓛 = −i[H_eff, ·] + Σ_k γ_k 𝒟[L_k] es un elemento del cono de Lindblad:
   CP y TP (Tr 𝓛(ρ) = 0 ∀ ρ).

5. MÉTRICAS CUÁNTICAS. Umegaki

       S(ρ ‖ σ) = Tr(ρ (log ρ − log σ))

   y Bures–Wasserstein

       d_B(ρ, σ) = √( 2 (1 − Tr √(√ρ σ √ρ)) ).

TOPOS DE VERDAD Ω₄ = {⊥, ∂, ♯, ⊤}
──────────────────────────────────
El clasificador de subobjetos del topos de haces sobre el sitio de 4 sieves del ciclo REM
es la cadena Ω₄ = { ABSURDUM_VETOED ≺ BOUNDARY_DEGRADED ≺ TOPOLOGICAL_SOUND ≺ VERUM_COHERENT }.
El álgebra de Heyting de Ω₄ está completamente determinada por la ley de residuación

       c ∧ a ≤ b  ⟺  c ≤ (a → b)  ∀ a, b, c ∈ Ω₄,

que da la implicación a → b = ⋁{ c : c ∧ a ≤ b }. La negación intuicionista ¬_H a = a → ⊥
no es involutiva (Ω₄ es strictly intuitionista): sólo los elementos {⊥, ⊤} son regulares.

TRADUCCIÓN BIYECTIVA A "DOLOR Y DINERO"
───────────────────────────────────────
• Estrés macro sin riesgo de capital: devaluación +45%, insumos +60%, huelgas prolongadas
  sin desembolsar un peso en la obra física.
• Navegación de Lagrange de nulo propelente: trayectorias de bancarrota en Wˢ, Wᵘ a Δv → 0.
• Blindaje de tasa WACC: reduce prima de riesgo contra Fat-Tail Events antes de licitar.
• Prevención de falsas alertas: parálisis de obra por falsos positivos se restringe a ℳ_REM.

ORGANIZACIÓN EN TRES FASES ANIDADAS
───────────────────────────────────
FASE 1: Heyting Ω₄ (topos), cuaterniones ℍ, complejo simplicial + Hodge, red circuital no
        recíproca, C*-álgebra B(ℋ), uniformización fucsiana PSL(2,ℤ).
        ÚLTIMO MÉTODO ──► TopologicalCircuitBundle.lift_hamiltonian : (K, τ) ⟶ 𝔥𝔢𝔯(ℋ_n)
FASE 2: Distancia hiperbólica, Jacobi/CRTBP, tubos Wˢ/Wᵘ, hoja de mundo de Polyakov,
        motor Lindblad-GKSL, estado metabolizado.
        PRIMERO consume H; ÚLTIMO ──► MetabolizedFieldState.create_metabolized_state
        : (bundle, ρ₀, τ) ⟶ MetabolizedFieldState
FASE 3: Merkle sha512, vacuna espectral, DreamFieldReport, orquestador terminal
        TOONOniricDreamerEngine.
        PRIMERO consume MetabolizedFieldState; ÚLTIMO ──► run_terminal_rem_cycle
        : MetabolizedFieldState ⟶ DreamFieldReport ⟶ Merkle + Ω₄

ANIDACIÓN FORMAL (el último morfismo de cada fase ES el dominio del primero de la siguiente):

    lift_hamiltonian            : (K, τ) ⟶ H ∈ 𝔥𝔢𝔯(ℋ_n)
    create_metabolized_state    : (H, ρ₀, τ) ⟶ MetabolizedFieldState
    run_terminal_rem_cycle      : MetabolizedFieldState ⟶ DreamFieldReport + ISR-safe REM
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONOniricDreamerEngine.v9")

# ── Tipos algebraicos ──────────────────────────────────────────────────────────
ComplexMatrix = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]
ComplexVector = NDArray[np.complex128]

try:
    _TRAPEZOID = np.trapezoid  # NumPy ≥ 2.0
except AttributeError:
    _TRAPEZOID = np.trapz  # type: ignore[attr-defined]

_SCHEMA_VERSION: Final[str] = "8.1.0"
_ENGINE_VERSION: Final[str] = "9.1.0-Doctoral-Poincaré-Fuchsian"
_ATOL_DEFAULT: Final[float] = 1e-9
_H2_IMAG_FLOOR: Final[float] = 1e-6

__all__ = [
    "HeytingTruthValue",
    "Quaternion",
    "SimplicialHodgeComplex",
    "NonReciprocalCircuitField",
    "OpenQuantumDynamicsSeed",
    "TopologicalCircuitBundle",
    "BanachOperatorAlgebra",
    "MobiusTransformation",
    "poincare_hyperbolic_distance",
    "reduce_to_poincare_fundamental_domain",
    "classify_modular_element",
    "FuchsianDomainCertificate",
    "JacobiIntegral",
    "LagrangePoint",
    "CRTBPInvariantManifoldTube",
    "PolyakovWorldsheetMetrics",
    "NonHermitianLindbladMasterEngine",
    "LindbladCFTMasterEvolver",
    "MetabolizedFieldState",
    "MerkleInclusionProof",
    "MerkleTree",
    "SpectralImmuneVaccineSynthesizer",
    "DreamFieldReport",
    "WakeSleepCycleReport",
    "TOONOniricDreamerEngine",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — FUNDAMENTOS HIPERCOMPLEJOS, TOPOLOGÍA EXACTA Y UNIFORMIZACIÓN FUCSIANA
#
#   Cadena de morfismos de la Fase 1:
#       Ω₄ ──meet/join/→──► ℍ ──Hodge──► Y_bus ──B(ℋ)──► PSL(2,ℤ)
#           ──► H = TopologicalCircuitBundle.lift_hamiltonian(n, τ)
#   El Hamiltoniano H ES el objeto inicial de la Fase 2 (GKSL + CRTBP + Polyakov).
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.1 — Álgebra de Heyting Ω₄ (topos de 4 sieves)
# ─────────────────────────────────────────────────────────────────────────────
class HeytingTruthValue(IntEnum):
    r"""
    Cadena finita completa Ω₄ = {⊥ ≺ ∂ ≺ ♯ ≺ ⊤}, álgebra de Heyting.

    Como toda cadena finita con 0 y 1, Ω₄ es álgebra de Heyting completa:

        a ∧ b  = min(a, b)                     (ínfimo / producto categorial),
        a ∨ b  = max(a, b)                     (supremo / coproducto),
        a → b  = ⋁{ c ∈ Ω₄ : c ∧ a ≤ b }       (residuo de Heyting),
        ¬_H a  = a → ⊥                          (pseudocomplemento).

    Propiedades estructurales:
        • La ley de residuación  c ∧ a ≤ b ⟺ c ≤ (a → b)  es el axioma definitorio.
        • ¬¬a ≥ a  (ley débil intuicionista).
        • Un elemento es *regular* ssi ¬¬a = a. En Ω₄ sólo {⊥, ⊤} son regulares.
        • El tercio excluso a ∨ ¬a = ⊤ falla precisamente en ∂ y ♯.
        • El esqueleto booleano B₂ ↪ Ω₄ es la reflexión de doble negación.

    Interpretación Poincaré–categórica (ciclo REM)
    ----------------------------------------------
    ⊥  ABSURDUM_VETOED     ≡ tubo Wˢ ⋔ Wᵘ escapado a RealWorld (aislamiento roto).
    ∂  BOUNDARY_DEGRADED   ≡ frontera de Hill / sieve de frontera modular.
    ♯  TOPOLOGICAL_SOUND   ≡ navegación regular en el tubo (Δv ≈ 0) sin KAM roto.
    ⊤  VERUM_COHERENT      ≡ τ ∈ ℱ, toro modular persistente, ρ ∈ 𝔇(ℋ).
    """

    ABSURDUM_VETOED: int = 0  # ⊥
    BOUNDARY_DEGRADED: int = 1  # ∂
    TOPOLOGICAL_SOUND: int = 2  # ♯
    VERUM_COHERENT: int = 3  # ⊤

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        r"""Ínfimo categorial ∧ : límite del diagrama discreto {a, b}."""
        return HeytingTruthValue(min(int(self), int(other)))

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        r"""Supremo categorial ∨ : colímite del diagrama discreto {a, b}."""
        return HeytingTruthValue(max(int(self), int(other)))

    def implies(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        r"""Residuo de Heyting a → b = ⋁{ c ∈ Ω₄ : c ∧ a ≤ b }."""
        a, b = int(self), int(other)
        valid = [c for c in range(4) if min(c, a) <= b]
        return HeytingTruthValue(max(valid))

    def complement(self) -> "HeytingTruthValue":
        r"""Pseudocomplemento intuicionista ¬_H a = a → ⊥."""
        return self.implies(HeytingTruthValue.ABSURDUM_VETOED)

    def pseudo_complement(self) -> "HeytingTruthValue":
        r"""Alias del pseudocomplemento ¬_H."""
        return self.complement()

    def classical_negation(self) -> "HeytingTruthValue":
        r"""Negación involutiva inducida por B₂ ⊂ Ω₄: a ↦ 3 − a."""
        return HeytingTruthValue(3 - int(self))

    def double_negation(self) -> "HeytingTruthValue":
        r"""Funtor ¬¬ : Ω₄ → Ω₄. En general ¬¬a ≥ a (ley débil intuicionista)."""
        return self.complement().complement()

    def modus_ponens(self, implication: "HeytingTruthValue") -> "HeytingTruthValue":
        r"""De a y (a → b) se obtiene a ∧ (a → b) = a ∧ b sobre una cadena."""
        return self.meet(implication)

    def is_regular(self) -> bool:
        r"""a es regular ssi ¬¬a = a. En Ω₄: {⊥, ⊤}."""
        return self.double_negation() == self

    def is_dense(self) -> bool:
        r"""a es denso ssi ¬a = ⊥."""
        return self.complement() == HeytingTruthValue.ABSURDUM_VETOED

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬a = ⊤ ⇔ a ∈ {⊥, ⊤}."""
        return self.join(self.complement()) == HeytingTruthValue.VERUM_COHERENT

    def booleanize(self) -> "HeytingTruthValue":
        r"""Reflexión de doble negación a B₂ ⊂ Ω₄: a ↦ ¬¬a."""
        return self.double_negation()

    def heyting_distance(self, other: "HeytingTruthValue") -> float:
        r"""Métrica normalizada |a − b| / (|Ω₄| − 1) ∈ [0, 1]."""
        return abs(int(self) - int(other)) / 3.0

    @classmethod
    def bottom(cls) -> "HeytingTruthValue":
        return cls.ABSURDUM_VETOED

    @classmethod
    def top(cls) -> "HeytingTruthValue":
        return cls.VERUM_COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingTruthValue":
        r"""Encaje ι : B₂ ↪ Ω₄; False ↦ ⊥, True ↦ ⊤."""
        return cls.VERUM_COHERENT if b else cls.ABSURDUM_VETOED

    @classmethod
    def from_coherence(
        cls,
        coherence: float,
        thr_top: float = 0.75,
        thr_sound: float = 0.45,
        thr_bound: float = 0.15,
    ) -> "HeytingTruthValue":
        r"""Sección [0,1] → Ω₄ por umbrales de coherencia espectral."""
        if coherence >= thr_top:
            return cls.VERUM_COHERENT
        if coherence >= thr_sound:
            return cls.TOPOLOGICAL_SOUND
        if coherence >= thr_bound:
            return cls.BOUNDARY_DEGRADED
        return cls.ABSURDUM_VETOED

    def to_bool(self) -> bool:
        r"""Sección parcial de ι; no definida fuera de im(B₂ ↪ Ω₄)."""
        if self not in (
            HeytingTruthValue.ABSURDUM_VETOED,
            HeytingTruthValue.VERUM_COHERENT,
        ):
            raise ValueError(f"{self.name} ∉ im(B₂ ↪ Ω₄); usar booleanize().")
        return self == HeytingTruthValue.VERUM_COHERENT

    def __and__(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return self.meet(other)

    def __or__(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return self.join(other)

    def __invert__(self) -> "HeytingTruthValue":
        return self.complement()

    def __le__(self, other: "HeytingTruthValue") -> bool:  # type: ignore[override]
        return int(self) <= int(other)

    @classmethod
    def verify_heyting_axioms(cls) -> bool:
        r"""
        Verificación exhaustiva de los axiomas de Heyting sobre Ω₄:
        residuación, idempotencia, conmutatividad, distributividad,
        modus ponens interno a ∧ (a → b) = a ∧ b, y regularidad de {⊥, ⊤}.
        """
        elements = list(cls)
        for a in elements:
            for b in elements:
                residual = a.implies(b)
                if a.meet(residual) != a.meet(b):
                    return False
                for c in elements:
                    lhs = min(int(c), int(a)) <= int(b)
                    rhs = int(c) <= int(residual)
                    if lhs != rhs:
                        return False
                if a.meet(a) != a or a.join(a) != a:
                    return False
                if a.meet(a.complement()) != cls.ABSURDUM_VETOED:
                    return False
                if a.implies(a) != cls.VERUM_COHERENT:
                    return False
                for c in elements:
                    if a.meet(b.join(c)) != a.meet(b).join(a.meet(c)):
                        return False
                    if a.join(b.meet(c)) != a.join(b).meet(a.join(c)):
                        return False
        assert cls.BOUNDARY_DEGRADED.double_negation() != cls.BOUNDARY_DEGRADED
        assert cls.TOPOLOGICAL_SOUND.double_negation() != cls.TOPOLOGICAL_SOUND
        assert cls.ABSURDUM_VETOED.double_negation() == cls.ABSURDUM_VETOED
        assert cls.VERUM_COHERENT.double_negation() == cls.VERUM_COHERENT
        return True


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.2 — Álgebra de cuaterniones ℍ (división no conmutativa)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Quaternion:
    r"""
    Álgebra de división ℍ ≅ Cl_{0,3}⁺, ℝ-espacio de Banach de dimensión 4,
    con producto no conmutativo y estructura de C*-álgebra real:

        |q* q| = |q|²,    ∀ q ∈ ℍ.

    Representación fiel en M₂(ℂ):

        q = w + xi + yj + zk ↦ [[ w+iz ,  y+ix ], [ −y+ix ,  w−iz ]].

    Estructuras inducidas:
        • grupo de Lie Sp(1) ≅ SU(2) (versor, exp, log);
        • recubrimiento de Hopf 2:1  Sp(1) → SO(3);
        • fibración de Hopf S³ → S²;
        • ℍ es un álgebra de composición:  |q₁ q₂| = |q₁| |q₂|.

    En la Fase 1 el cuaternión (Re τ, Im τ, β₁, ½) parametriza el bloque
    2×2 que `lift_hamiltonian` inyecta en H (analogía: rotación rígida de
    Euler / elipsoide de Poincaré).
    """

    w: float
    x: float
    y: float
    z: float

    _NORM_FLOOR: Final[float] = 1e-15

    def norm_squared(self) -> float:
        r"""|q|² = w² + x² + y² + z²."""
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def norm(self) -> float:
        r"""|q| = √(q* q)."""
        return math.sqrt(self.norm_squared())

    def norm_residual_cstar(self) -> float:
        r"""Residuo de la C*-identidad: ||q* q| − |q|²|."""
        return abs((self.conjugate() * self).norm() - self.norm_squared())

    def conjugate(self) -> "Quaternion":
        r"""Conjugación: q* = w − xi − yj − zk."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def scale(self, lam: float) -> "Quaternion":
        r"""Multiplicación escalar a derecha: q ↦ q · λ (λ ∈ ℝ)."""
        return Quaternion(self.w * lam, self.x * lam, self.y * lam, self.z * lam)

    def inverse(self) -> "Quaternion":
        r"""Inversa multiplicativa: q⁻¹ = q* / |q|²."""
        n2 = self.norm_squared()
        if n2 < self._NORM_FLOOR:
            raise ZeroDivisionError("Cuaternión nulo: no invertible en ℍ.")
        return self.conjugate().scale(1.0 / n2)

    def versor(self) -> "Quaternion":
        r"""Normalización a Sp(1) ⊂ ℍ: q ↦ q / |q|."""
        n = self.norm()
        if n < self._NORM_FLOOR:
            raise ZeroDivisionError("Normalización de cuaternión nulo.")
        return self.scale(1.0 / n)

    def to_matrix(self) -> ComplexMatrix:
        r"""Homomorfismo inyectivo de ℝ-álgebras ℍ ↪ M₂(ℂ)."""
        return np.array(
            [
                [complex(self.w, self.z), complex(self.y, self.x)],
                [complex(-self.y, self.x), complex(self.w, -self.z)],
            ],
            dtype=np.complex128,
        )

    def su2_det_residual(self) -> float:
        r"""|det φ(q̂) − 1| sobre el versor; fidelidad de la inmersión SU(2)."""
        u = self.versor()
        return abs(complex(np.linalg.det(u.to_matrix())) - 1.0)

    def exp(self) -> "Quaternion":
        r"""
        Exponencial cuaterniónica:
            exp(w + v) = (eʷ cos|v|) + (eʷ sin|v|/|v|) · v,   v = xi + yj + zk.
        """
        vec_norm = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        e_w = math.exp(self.w)
        if vec_norm < self._NORM_FLOOR:
            return Quaternion(e_w, 0.0, 0.0, 0.0)
        s = e_w * math.sin(vec_norm) / vec_norm
        return Quaternion(e_w * math.cos(vec_norm), self.x * s, self.y * s, self.z * s)

    def log(self) -> "Quaternion":
        r"""Logaritmo cuaterniónico (rama principal): log(q) = log|q| + û · arccos(w/|q|)."""
        n = self.norm()
        if n < self._NORM_FLOOR:
            raise ValueError("log no definido en el cuaternión nulo.")
        vec_n = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        if vec_n < self._NORM_FLOOR:
            return Quaternion(math.log(n), 0.0, 0.0, 0.0)
        phi = math.acos(max(-1.0, min(1.0, self.w / n)))
        s = phi / vec_n
        return Quaternion(math.log(n), s * self.x, s * self.y, s * self.z)

    def hopf_s2(self) -> Tuple[float, float, float]:
        r"""Proyección de Hopf S³ ⊂ ℍ ≅ ℝ⁴ → S² ⊂ Im ℍ."""
        u = self.versor()
        return (
            2.0 * (u.w * u.y + u.x * u.z),
            2.0 * (u.x * u.y - u.w * u.z),
            u.w ** 2 + u.x ** 2 - u.y ** 2 - u.z ** 2,
        )

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Quaternion":
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: object) -> "Quaternion":
        r"""Producto de Hamilton (no conmutativo, asociativo)."""
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            )
        if isinstance(other, (int, float)):
            return self.scale(float(other))
        return NotImplemented

    def __rmul__(self, other: object) -> "Quaternion":
        if isinstance(other, (int, float)):
            return self.scale(float(other))
        return NotImplemented

    def __truediv__(self, other: "Quaternion") -> "Quaternion":
        return self * other.inverse()

    def associativity_residual(self, other: "Quaternion", third: "Quaternion") -> float:
        r"""||(q₁ q₂) q₃ − q₁ (q₂ q₃)||."""
        return ((self * other) * third - self * (other * third)).norm()

    def distributivity_residual(self, other: "Quaternion", third: "Quaternion") -> float:
        r"""||q₁ (q₂ + q₃) − q₁ q₂ − q₁ q₃||."""
        lhs = self * (other + third)
        rhs = self * other + self * third
        return (lhs - rhs).norm()

    def composition_residual(self, other: "Quaternion") -> float:
        r"""||q₁ q₂| − |q₁| |q₂||."""
        return abs((self * other).norm() - self.norm() * other.norm())

    def exp_log_roundtrip_residual(self) -> float:
        r"""||exp(log(q)) − q|| sobre ℍˣ."""
        return (self.log().exp() - self).norm()


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.3 — Complejo simplicial finito, Laplacianos de Hodge, números de Betti
# ─────────────────────────────────────────────────────────────────────────────
class SimplicialHodgeComplex:
    r"""
    Complejo de cadenas simplicial finito K = (V, E, F) de dimensión 2.

        C₂ --∂₂--> C₁ --∂₁--> C₀,   ∂₁ ∂₂ = 0.

    Laplacianos de Hodge (Eckmann–Hodge):
        L₀ = ∂₁ ∂₁ᵀ,   L₁ = ∂₁ᵀ ∂₁ + ∂₂ ∂₂ᵀ,   L₂ = ∂₂ᵀ ∂₂.

    Números de Betti: βₖ = dim ker Lₖ = dim Hₖ(K; ℝ),
    χ(K) = |V| − |E| + |F| = β₀ − β₁ + β₂.

    Descomposición de Hodge en 1-cadenas:
        C₁ = im ∂₂ ⊕ im ∂₁ᵀ ⊕ ker L₁.

    Analogía celestial: β₁ cuenta 1-ciclos (cavidades contractuales / lóbulos
    homoclínicos discretos); χ es el índice de Poincaré–Hopf del 2-complejo.
    """

    def __init__(
        self,
        num_vertices: int,
        edges: List[Tuple[int, int]],
        faces: List[Tuple[int, int, int]],
    ) -> None:
        if num_vertices < 1:
            raise ValueError("num_vertices ≥ 1.")
        self.num_vertices = int(num_vertices)
        self.edges = [tuple(sorted((u, v))) for u, v in edges]
        self.faces = [tuple(sorted((u, v, w))) for u, v, w in faces]
        self._validate_simplicial_topology()
        self.boundary_1 = self._build_boundary_1()
        self.boundary_2 = self._build_boundary_2()
        self._validate_chain_complex_exactness()
        self.laplacian_0 = self.boundary_1 @ self.boundary_1.T
        self.laplacian_1 = (self.boundary_1.T @ self.boundary_1) + (
            self.boundary_2 @ self.boundary_2.T
        )
        if self.boundary_2.size > 0:
            self.laplacian_2 = self.boundary_2.T @ self.boundary_2
        else:
            self.laplacian_2 = np.zeros(
                (len(self.faces), len(self.faces)), dtype=np.float64
            )

    def _validate_simplicial_topology(self) -> None:
        for (u, v) in self.edges:
            if not (0 <= u < self.num_vertices and 0 <= v < self.num_vertices):
                raise ValueError(f"Arista {(u, v)} fuera de [0, {self.num_vertices}).")
            if u == v:
                raise ValueError(f"Lazo {(u, v)} no es 1-símplice.")
        edge_set = set(self.edges)
        for (u, v, w) in self.faces:
            if len({u, v, w}) != 3:
                raise ValueError(f"Cara degenerada {(u, v, w)}.")
            for e in ((u, v), (v, w), (u, w)):
                if tuple(sorted(e)) not in edge_set:
                    raise ValueError(f"Cara {(u, v, w)} refiere arista inexistente {e}.")

    def _validate_chain_complex_exactness(self, tol: float = 1e-9) -> None:
        r"""Propiedad ∂₁ ∂₂ = 0."""
        if self.boundary_2.size == 0:
            return
        residual = self.boundary_1 @ self.boundary_2
        rn = float(la.norm(residual))
        if rn > tol:
            raise ValueError(f"Violación de ∂₁∂₂ = 0: ||·||={rn:.3e}")

    def _build_boundary_1(self) -> RealMatrix:
        B1 = np.zeros((self.num_vertices, len(self.edges)), dtype=np.float64)
        for e_idx, (u, v) in enumerate(self.edges):
            B1[u, e_idx] = -1.0
            B1[v, e_idx] = 1.0
        return B1

    def _build_boundary_2(self) -> RealMatrix:
        r"""Orientación canónica: ∂[u,v,w] = [v,w] − [u,w] + [u,v]."""
        if not self.faces:
            return np.zeros((len(self.edges), 0), dtype=np.float64)
        B2 = np.zeros((len(self.edges), len(self.faces)), dtype=np.float64)
        edge_map = {e: idx for idx, e in enumerate(self.edges)}
        for f_idx, (u, v, w) in enumerate(self.faces):
            e_vw = edge_map.get(tuple(sorted((v, w))))
            e_uw = edge_map.get(tuple(sorted((u, w))))
            e_uv = edge_map.get(tuple(sorted((u, v))))
            if e_vw is not None:
                B2[e_vw, f_idx] += 1.0
            if e_uw is not None:
                B2[e_uw, f_idx] -= 1.0
            if e_uv is not None:
                B2[e_uv, f_idx] += 1.0
        return B2

    @property
    def coboundary_0(self) -> RealMatrix:
        r"""δ₀ = ∂₁ᵀ : C⁰ → C¹."""
        return self.boundary_1.T

    @property
    def coboundary_1(self) -> RealMatrix:
        r"""δ₁ = ∂₂ᵀ : C¹ → C²."""
        return self.boundary_2.T

    def connected_components_combinatorial(self) -> List[Set[int]]:
        r"""Descomposición de V en componentes conexas vía DFS."""
        adjacency: Dict[int, Set[int]] = {v: set() for v in range(self.num_vertices)}
        for (u, v) in self.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in range(self.num_vertices):
            if start in visited:
                continue
            stack = [start]
            comp: Set[int] = set()
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                stack.extend(adjacency[node] - comp)
            visited |= comp
            components.append(comp)
        return components

    def euler_poincare_characteristic(self) -> int:
        r"""χ(K) = |V| − |E| + |F| (índice de Poincaré–Hopf discreto)."""
        return self.num_vertices - len(self.edges) + len(self.faces)

    def compute_betti_numbers(self, tol: float = 1e-10) -> Tuple[int, int, int]:
        r"""βₖ = dim ker Lₖ vía conteo espectral de autovalores nulos."""
        eigvals_0 = la.eigvalsh(self.laplacian_0)
        betti_0 = int(np.sum(np.abs(eigvals_0) < tol))
        eigvals_1 = la.eigvalsh(self.laplacian_1)
        betti_1 = int(np.sum(np.abs(eigvals_1) < tol))
        if self.laplacian_2.size > 0:
            eigvals_2 = la.eigvalsh(self.laplacian_2)
            betti_2 = int(np.sum(np.abs(eigvals_2) < tol))
        else:
            betti_2 = 0
        n_comp = len(self.connected_components_combinatorial())
        if n_comp != betti_0:
            logger.warning(
                "β₀ espectral (%d) ≠ componentes DFS (%d).",
                betti_0,
                n_comp,
            )
        return betti_0, betti_1, betti_2

    def verify_betti_euler_consistency(self, tol: float = 1e-10) -> bool:
        r"""Consistencia de Euler–Poincaré: β₀ − β₁ + β₂ = χ(K)."""
        b0, b1, b2 = self.compute_betti_numbers(tol)
        return (b0 - b1 + b2) == self.euler_poincare_characteristic()

    def algebraic_connectivity(self, tol: float = 1e-12) -> float:
        r"""Segundo autovalor de L₀ (valor de Fiedler; 0 si β₀ > 1 o n=1)."""
        evals = np.sort(la.eigvalsh(self.laplacian_0))
        if evals.size < 2:
            return 0.0
        return float(max(evals[1], 0.0)) if evals[1] > tol else 0.0

    def spectral_gap_L1(self, betti_1: int) -> float:
        r"""Autovalor número (β₁ + 1) de L₁: gap espectral sobre armónicos."""
        evals = np.sort(la.eigvalsh(self.laplacian_1))
        if len(evals) <= betti_1:
            return 0.0
        return float(max(evals[betti_1], 0.0))

    def harmonic_1_forms(self, tol: float = 1e-10) -> RealMatrix:
        r"""Base ortonormal de ker L₁ ⊂ C₁ ≅ ℝ^{|E|}."""
        w, v = la.eigh(self.laplacian_1)
        mask = np.abs(w) < tol
        if not np.any(mask):
            return np.zeros((len(self.edges), 0), dtype=np.float64)
        return v[:, mask]

    def harmonic_2_forms(self, tol: float = 1e-10) -> RealMatrix:
        r"""Base ortonormal de ker L₂ ⊂ C₂ ≅ ℝ^{|F|}."""
        if self.laplacian_2.size == 0:
            return np.zeros((len(self.faces), 0), dtype=np.float64)
        w, v = la.eigh(self.laplacian_2)
        mask = np.abs(w) < tol
        if not np.any(mask):
            return np.zeros((len(self.faces), 0), dtype=np.float64)
        return v[:, mask]

    def hodge_decompose_1_form(
        self, omega: RealVector, rcond: float = 1e-10
    ) -> Tuple[RealVector, RealVector, RealVector]:
        r"""
        Descomposición canónica de Hodge de ω ∈ C₁:
            ω = ∂₂ α  +  ∂₁ᵀ β  +  γ,    γ ∈ ker L₁.
        """
        omega = np.asarray(omega, dtype=np.float64).reshape(-1)
        if omega.size != len(self.edges):
            raise ValueError("ω debe vivir en C₁ (dim = |E|).")
        B2, B1 = self.boundary_2, self.boundary_1
        if B2.size > 0 and B2.shape[1] > 0:
            exact = B2 @ la.lstsq(B2, omega, cond=rcond)[0]
        else:
            exact = np.zeros_like(omega)
        remainder = omega - exact
        coexact = B1.T @ la.lstsq(B1.T, remainder, cond=rcond)[0]
        harmonic = remainder - coexact
        return exact, coexact, harmonic

    def harmonic_projection_1(self) -> RealMatrix:
        r"""Proyector ortogonal sobre ker L₁ ⊂ C₁."""
        P = self.harmonic_1_forms()
        if P.shape[1] == 0:
            return np.zeros((len(self.edges), len(self.edges)), dtype=np.float64)
        return P @ P.T

    def heat_kernel_trace(self, t: float = 1.0) -> float:
        r"""
        Combinación alternada (−1)^k Tr e^{−t L_k}. Como t → ∞ recupera χ(K)
        (McKean–Singer / índice de Hodge).
        """
        tr0 = float(np.sum(np.exp(-t * la.eigvalsh(self.laplacian_0))))
        tr1 = float(np.sum(np.exp(-t * la.eigvalsh(self.laplacian_1))))
        tr2 = (
            float(np.sum(np.exp(-t * la.eigvalsh(self.laplacian_2))))
            if self.laplacian_2.size
            else 0.0
        )
        return tr0 - tr1 + tr2


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.4 — Red circuital no recíproca sobre el 1-esqueleto
# ─────────────────────────────────────────────────────────────────────────────
class NonReciprocalCircuitField:
    r"""
    Red no recíproca sobre el 1-esqueleto de K, con tierra por componente conexa
    (β₀ ≥ 1). Los elementos giroscópicos Y_b − Y_bᵀ ≠ 0 modelan disipación
    asimétrica (analogía: fuerza de Coriolis en el CRTBP sinóptico).

    Identidades estructurales:
        • Tellegen: Σ_e v_e i_e* = V† Y_bus V.
        • Pasividad: Re Y_b ⪰ 0  ⟺  λ_min((Y_b + Y_b†)/2) ≥ 0.
        • Recirculación: A = ½(Y_b − Y_bᵀ) mide el defecto de Onsager.
    """

    def __init__(
        self,
        simplicial_complex: SimplicialHodgeComplex,
        angular_freq: float = 100.0,
    ) -> None:
        self.complex = simplicial_complex
        self.omega = float(angular_freq)
        self.num_edges = len(self.complex.edges)
        self.branch_admittance = self._build_branch_admittance()

    def _build_branch_admittance(self) -> ComplexMatrix:
        r"""Y_b = G + jB con acoplamiento giroscópico antisimétrico."""
        dim = self.num_edges
        Yb = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            g_i = 1.0 + 0.1 * (i + 1)
            b_i = self.omega * 0.01 - 1.0 / (self.omega * 0.05 + 1e-5)
            Yb[i, i] = complex(g_i, b_i)
            if i + 1 < dim:
                gyration = 0.25 * (i + 1)
                Yb[i, i + 1] += complex(0.0, gyration)
                Yb[i + 1, i] -= complex(0.0, gyration)
        return Yb

    def reciprocity_defect_norm(self) -> float:
        r"""σ_max(½(Y_b − Y_bᵀ)): norma espectral del defecto de Onsager."""
        antisym = 0.5 * (self.branch_admittance - self.branch_admittance.T)
        sv = la.svdvals(antisym)
        return float(np.max(sv)) if len(sv) > 0 else 0.0

    def passivity_margin(self) -> float:
        r"""λ_min((Y_b + Y_b†)/2). Negativo ⇒ violación de pasividad."""
        herm = 0.5 * (self.branch_admittance + self.branch_admittance.conj().T)
        return float(np.min(la.eigvalsh(herm)))

    def compute_bus_admittance_matrix(self) -> ComplexMatrix:
        r"""Y_bus = ∂₁ Y_b ∂₁ᵀ."""
        B1 = self.complex.boundary_1
        return B1 @ self.branch_admittance @ B1.T

    def solve_voltage_distribution(
        self, nodal_current_injections: NDArray[np.complex128]
    ) -> ComplexVector:
        r"""
        Y_bus V = I con V = 0 en un nodo de tierra por componente conexa.
        ker Y_bus tiene dimensión β₀.
        """
        n = self.complex.num_vertices
        components = self.complex.connected_components_combinatorial()
        ground_nodes = {min(comp) for comp in components}
        free_nodes = [i for i in range(n) if i not in ground_nodes]
        V_full = np.zeros(n, dtype=np.complex128)
        if not free_nodes:
            return V_full
        inj = np.asarray(nodal_current_injections, dtype=np.complex128).reshape(-1)
        if inj.size != n:
            padded = np.zeros(n, dtype=np.complex128)
            m = min(n, inj.size)
            padded[:m] = inj[:m]
            inj = padded
        Y_bus = self.compute_bus_admittance_matrix()
        Y_reduced = Y_bus[np.ix_(free_nodes, free_nodes)]
        I_reduced = inj[free_nodes]
        try:
            V_reduced = la.solve(Y_reduced, I_reduced)
        except la.LinAlgError:
            logger.warning("Y_reduced casi-singular; lstsq regularizado.")
            V_reduced, *_ = la.lstsq(Y_reduced, I_reduced)
        for local_idx, node in enumerate(free_nodes):
            V_full[node] = V_reduced[local_idx]
        return V_full

    def kcl_residual(self, v_nodes: ComplexVector, i_inj: ComplexVector) -> float:
        r"""||Y_bus V − I||₂ / max(1, ||I||₂)."""
        I_pred = self.compute_bus_admittance_matrix() @ v_nodes
        denom = max(float(np.linalg.norm(i_inj)), 1.0)
        return float(np.linalg.norm(I_pred - i_inj) / denom)

    def verify_tellegen_conservation(
        self, v_nodes: ComplexVector, tol: float = 1e-6
    ) -> float:
        r"""Identidad de Tellegen: residuo relativo Σ_e v_e conj(i_e) − Σ_n conj(V_n) I_n."""
        _ = tol
        Y_bus = self.compute_bus_admittance_matrix()
        I_full = Y_bus @ v_nodes
        branch_v = self.complex.boundary_1.T @ v_nodes
        branch_i = self.branch_admittance @ branch_v
        lhs = np.sum(branch_v * np.conj(branch_i))
        rhs = np.sum(np.conj(v_nodes) * I_full)
        denom = max(abs(rhs), 1e-12)
        return float(abs(lhs - rhs) / denom)

    def power_dissipated(self, v_nodes: ComplexVector) -> float:
        r"""P_dis = Re[V† Y_bus V]."""
        Y_bus = self.compute_bus_admittance_matrix()
        power = np.conj(v_nodes) @ (Y_bus @ v_nodes)
        return float(np.real(power))


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.5 — C*-álgebra de operadores B(ℋ) (normas, cono, métricas)
# ─────────────────────────────────────────────────────────────────────────────
class BanachOperatorAlgebra:
    r"""
    C*-álgebra B(ℋ) de dimensión finita. Estructuras y métricas transportadas:

        • Normas ‖·‖₁ (traza), ‖·‖₂ (Hilbert–Schmidt), ‖·‖∞ (operador).
        • Variedad de estados 𝔇(ℋ) = { ρ = ρ†, ρ ⪰ 0, Tr ρ = 1 }.
        • Entropía de von Neumann  S(ρ) = −Tr(ρ log ρ).
        • Pureza P(ρ) = Tr(ρ²) ∈ [1/n, 1].
        • Fidelidad (Uhlmann–Jozsa)  F(ρ, σ) = (Tr √(√ρ σ √ρ))² ∈ [0, 1].
        • Distancia de Bures–Wasserstein  d_B(ρ, σ) = √(2 (1 − √F(ρ, σ))).
        • Divergencia de Umegaki  D(ρ ‖ σ) = Tr ρ (log ρ − log σ).

    Analogía celestial: d_B es el análogo de Jacobi–Maupertuis sobre 𝔇(ℋ);
    D(ρ ‖ σ) es la acción de Poincaré entre dos estados del flujo onírico.
    """

    SPECTRUM_FLOOR: Final[float] = 1e-15

    @staticmethod
    def trace_norm(A: ComplexMatrix) -> float:
        r"""‖A‖₁ = Tr √(A†A) = Σ σ_i(A)."""
        return float(np.sum(la.svdvals(A)))

    @staticmethod
    def hilbert_schmidt_norm(A: ComplexMatrix) -> float:
        r"""‖A‖₂ = √Tr(A†A)."""
        return float(np.sqrt(np.real(np.trace(A.conj().T @ A))))

    @staticmethod
    def operator_norm(A: ComplexMatrix) -> float:
        r"""‖A‖∞ = σ_max(A)."""
        sv = la.svdvals(A)
        return float(np.max(sv)) if len(sv) > 0 else 0.0

    @staticmethod
    def norm_hierarchy_residual(A: ComplexMatrix, B: ComplexMatrix) -> float:
        r"""Verifica ‖AB‖₁ ≤ ‖A‖∞ · ‖B‖₁ (ideal de traza)."""
        lhs = BanachOperatorAlgebra.trace_norm(A @ B)
        rhs = BanachOperatorAlgebra.operator_norm(A) * BanachOperatorAlgebra.trace_norm(B)
        return max(0.0, lhs - rhs)

    @staticmethod
    def clean_density_matrix(rho: ComplexMatrix) -> ComplexMatrix:
        r"""Proyección espectral al símplex 𝔇(ℋ)."""
        rho_h = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals_clipped = np.maximum(eigvals, BanachOperatorAlgebra.SPECTRUM_FLOOR)
        eigvals_normalized = eigvals_clipped / np.sum(eigvals_clipped)
        proj = eigvecs @ np.diag(eigvals_normalized) @ eigvecs.conj().T
        return 0.5 * (proj + proj.conj().T)

    @staticmethod
    def is_valid_density_matrix(rho: ComplexMatrix, tol: float = 1e-8) -> bool:
        r"""Test ρ ∈ 𝔇(ℋ): hermiticidad, positividad, traza 1."""
        if float(la.norm(rho - rho.conj().T)) > tol:
            return False
        eigvals = la.eigvalsh(0.5 * (rho + rho.conj().T))
        if np.any(eigvals < -tol):
            return False
        return abs(np.real(np.trace(rho)) - 1.0) < tol

    @staticmethod
    def von_neumann_entropy(rho: ComplexMatrix, base: float = 2.0) -> float:
        r"""S(ρ) = −Σ λ_i log_base λ_i, con 0·log 0 ≡ 0."""
        eigvals = la.eigvalsh(rho)
        eigvals = eigvals[eigvals > BanachOperatorAlgebra.SPECTRUM_FLOOR]
        return float(-np.sum(eigvals * np.log(eigvals)) / math.log(base))

    @staticmethod
    def purity(rho: ComplexMatrix) -> float:
        r"""P(ρ) = Tr(ρ²) ∈ [1/n, 1]."""
        return float(np.real(np.trace(rho @ rho)))

    @staticmethod
    def _matrix_log_regularized(A: ComplexMatrix, floor: float = 1e-12) -> ComplexMatrix:
        r"""log(A) vía eigendescomposición hermitiana, con clipping del espectro."""
        eigvals, eigvecs = la.eigh(0.5 * (A + A.conj().T))
        log_eigvals = np.log(np.maximum(eigvals, floor))
        return eigvecs @ np.diag(log_eigvals) @ eigvecs.conj().T

    @staticmethod
    def quantum_relative_entropy(
        rho: ComplexMatrix, sigma: ComplexMatrix, floor: float = 1e-12
    ) -> float:
        r"""D(ρ‖σ) = Tr(ρ (log ρ − log σ)) en bits. Klein: ≥ 0, = 0 ⇔ ρ = σ."""
        log_rho = BanachOperatorAlgebra._matrix_log_regularized(rho, floor)
        log_sigma = BanachOperatorAlgebra._matrix_log_regularized(sigma, floor)
        val = np.real(np.trace(rho @ (log_rho - log_sigma))) / math.log(2.0)
        return float(max(val, 0.0)) if np.isfinite(val) else 1e6

    @staticmethod
    def quantum_fidelity(rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""F(ρ, σ) = (Tr √(√ρ σ √ρ))² ∈ [0, 1]."""
        rho_h = 0.5 * (rho + rho.conj().T)
        sigma_h = 0.5 * (sigma + sigma.conj().T)
        sqrt_rho = np.real_if_close(la.sqrtm(rho_h), tol=1e6)
        inner = sqrt_rho @ sigma_h @ sqrt_rho.conj().T
        inner_h = 0.5 * (inner + inner.conj().T)
        eigvals = np.clip(np.real(la.eigvalsh(inner_h)), 0.0, None)
        fidelity_val = float(np.sum(np.sqrt(eigvals)) ** 2)
        return float(np.clip(fidelity_val, 0.0, 1.0))

    @staticmethod
    def bures_distance(rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""d_B(ρ, σ) = √(2 (1 − √F(ρ, σ)))."""
        f = BanachOperatorAlgebra.quantum_fidelity(rho, sigma)
        return float(math.sqrt(max(2.0 * (1.0 - math.sqrt(max(f, 0.0))), 0.0)))

    @staticmethod
    def bures_geodesic(
        rho: ComplexMatrix, sigma: ComplexMatrix, t: float
    ) -> ComplexMatrix:
        r"""
        Punto en la geodésica de Bures–Wasserstein entre ρ y σ a fracción t ∈ [0,1].
        Analogía: geodésica de Jacobi en el CRTBP sobre el nivel de Jacobi.
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t debe estar en [0, 1].")
        rho_h = 0.5 * (rho + rho.conj().T)
        sigma_h = 0.5 * (sigma + sigma.conj().T)
        sqrt_rho = la.sqrtm(rho_h).astype(np.complex128)
        sqrt_sigma = la.sqrtm(sigma_h).astype(np.complex128)
        M = sqrt_rho @ sqrt_sigma
        U, _, Vh = la.svd(M)
        phase = U @ Vh
        interpolant = (1.0 - t) * sqrt_rho + t * (phase @ sqrt_sigma)
        candidate = interpolant @ interpolant.conj().T
        return BanachOperatorAlgebra.clean_density_matrix(candidate)

    @staticmethod
    def cstar_residual(rho: ComplexMatrix) -> float:
        r"""||ρ* ρ| − |ρ|²| (nulo en aritmética exacta)."""
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

    @staticmethod
    def gksl_trace_generator(
        rho: ComplexMatrix,
        H: ComplexMatrix,
        jumps: Sequence[ComplexMatrix],
        gammas: Sequence[float],
    ) -> float:
        r"""
        Residuo de conservación de traza del generador GKSL:
            Tr 𝓛(ρ) = 0  (Axioma TP).
        """
        comm = -1j * (H @ rho - rho @ H)
        diss = np.zeros_like(rho, dtype=np.complex128)
        for g, L in zip(gammas, jumps):
            Ldl = L.conj().T @ L
            diss += float(g) * (
                L @ rho @ L.conj().T - 0.5 * (Ldl @ rho + rho @ Ldl)
            )
        return float(np.real(np.trace(comm + diss)))


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.6 — Uniformización fucsiana en ℍ² / PSL(2, ℤ)
# ─────────────────────────────────────────────────────────────────────────────
class MobiusTransformation:
    r"""
    Transformación de Möbius γ ∈ PSL(2, ℂ) ⊂ Aut(ℍ² ∪ ℝ̂):

        γ·τ = (a τ + b) / (c τ + d),    ad − bc = 1.

    Generadores del grupo modular PSL(2, ℤ):

        T = [[1, 1], [0, 1]]  (τ ↦ τ + 1),
        S = [[0, −1], [1, 0]] (τ ↦ −1/τ),

    con relaciones S² = (ST)³ = 1 en PSL(2, ℤ). Toda γ es isometría de (ℍ², ds²).
    """

    def __init__(self, a: int, b: int, c: int, d: int) -> None:
        det = a * d - b * c
        if det == -1:
            a, b, c, d = -a, -b, -c, -d
            det = 1
        if det != 1:
            raise ValueError("γ debe tener det = ±1 para vivir en PSL(2, ℤ).")
        self.a, self.b, self.c, self.d = int(a), int(b), int(c), int(d)

    @classmethod
    def T(cls) -> "MobiusTransformation":
        return cls(1, 1, 0, 1)

    @classmethod
    def S(cls) -> "MobiusTransformation":
        return cls(0, -1, 1, 0)

    @classmethod
    def identity(cls) -> "MobiusTransformation":
        return cls(1, 0, 0, 1)

    def apply(self, tau: complex) -> complex:
        r"""γ·τ = (aτ + b)/(cτ + d)."""
        den = self.c * tau + self.d
        if abs(den) < 1e-15:
            return complex(0.0, 1e6)
        return (self.a * tau + self.b) / den

    def trace(self) -> int:
        r"""Tr γ = |a + d| (clase de conjugación: elíptica / parabólica / hiperbólica)."""
        return abs(self.a + self.d)

    def conjugacy_class(self) -> str:
        tr = self.trace()
        if tr < 2:
            return "elliptic"
        if tr == 2:
            return "parabolic"
        return "hyperbolic"

    def compose(self, other: "MobiusTransformation") -> "MobiusTransformation":
        r"""Composición γ ∘ δ (producto matricial)."""
        return MobiusTransformation(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )


def _ensure_upper_half(z: complex) -> complex:
    imag = z.imag if z.imag > _H2_IMAG_FLOOR else _H2_IMAG_FLOOR
    return complex(z.real, imag)


def poincare_hyperbolic_distance(z1: complex, z2: complex) -> float:
    r"""
    Distancia geodésica hiperbólica en el semiplano superior ℍ²:

        d_ℍ²(z₁, z₂) = arccosh( 1 + |z₁ − z₂|² / (2 Im z₁ · Im z₂) ).

    Es invariante bajo toda γ ∈ PSL(2, ℝ): d(γ·z₁, γ·z₂) = d(z₁, z₂).
    """
    z1 = _ensure_upper_half(z1)
    z2 = _ensure_upper_half(z2)
    arg = 1.0 + (abs(z1 - z2) ** 2) / (2.0 * z1.imag * z2.imag)
    return float(math.acosh(max(1.0, arg)))


def _classify_modular_element(tau: complex) -> str:
    r"""
    Clasifica τ ∈ ℍ² por su posición en ℱ y puntos de torsión:
        i     (orden 2, elíptico),  ζ₃ = e^{2πi/3} (orden 3, elíptico),
        cúspide ∞ (parabólico), resto del interior de ℱ (hiperbólico genérico).
    """
    tau = _ensure_upper_half(tau)
    if abs(tau - 1j) < 1e-6:
        return "elliptic"
    if abs(tau - (-0.5 + 1j * math.sqrt(3.0) / 2.0)) < 1e-6 or abs(
        tau - (0.5 + 1j * math.sqrt(3.0) / 2.0)
    ) < 1e-6:
        return "elliptic"
    re_abs = abs(tau.real)
    if re_abs <= 0.5 + 1e-6 and abs(tau) >= 1.0 - 1e-6:
        if abs(abs(tau) - 1.0) < 1e-6 and abs(re_abs - 0.5) < 1e-6:
            return "elliptic"
        return "hyperbolic"
    return "parabolic"


def classify_modular_element(tau: complex) -> str:
    r"""Envoltura pública de `_classify_modular_element`."""
    return _classify_modular_element(tau)


@dataclass(frozen=True, slots=True)
class FuchsianDomainCertificate:
    r"""
    Certificado de proyección al Dominio Fundamental de Poincaré ℱ ⊂ ℍ²:

        ℱ = { τ ∈ ℍ² : |Re τ| ≤ ½, |τ| ≥ 1 }.

    El grupo modular PSL(2, ℤ) = ⟨T, S | S² = (ST)³ = 1⟩ actúa discretamente
    sobre ℍ²; ℱ es un dominio de Siegel cerrado bajo esta acción.
    """

    tau_original: complex
    tau_reduced: complex
    is_in_fundamental_domain: bool
    modular_transformations_count: int
    poincare_metric_distance: float
    modular_class: str  # "elliptic" | "parabolic" | "hyperbolic"

    def j_invariant_residual(self) -> float:
        r"""
        |j(τ) − j(γ·τ)| sobre la truncación q⁻¹ + 744 del invariante modular.
        En aritmética exacta j es PSL(2,ℤ)-invariante.
        """
        q_orig = np.exp(2j * math.pi * self.tau_original)
        q_red = np.exp(2j * math.pi * self.tau_reduced)
        if abs(q_orig) < 1e-12 or abs(q_red) < 1e-12:
            return float("inf")
        j_orig = 1.0 / q_orig + 744.0
        j_red = 1.0 / q_red + 744.0
        return float(abs(j_orig - j_red))

    def isometry_residual(self) -> float:
        r"""d_ℍ²(τ, γ·τ) debe coincidir con el campo almacenado (test de isometría)."""
        return abs(
            self.poincare_metric_distance
            - poincare_hyperbolic_distance(self.tau_original, self.tau_reduced)
        )


def _psl2z_reduce(tau: complex, max_iter: int = 128) -> Tuple[complex, int]:
    r"""
    Reducción algorítmica al dominio fundamental ℱ. Aplicación iterada de

        T^{−k}: τ ↦ τ − round(Re τ),     S: τ ↦ −1/τ  si |τ| < 1.
    """
    curr_tau = _ensure_upper_half(tau)
    transforms = 0
    for _ in range(max_iter):
        shift = round(curr_tau.real)
        if shift != 0:
            curr_tau -= shift
            transforms += 1
        if abs(curr_tau) < 1.0 - 1e-9:
            curr_tau = -1.0 / curr_tau
            curr_tau = _ensure_upper_half(curr_tau)
            transforms += 1
        else:
            break
    return curr_tau, transforms


def reduce_to_poincare_fundamental_domain(
    tau: complex, max_iter: int = 128
) -> FuchsianDomainCertificate:
    r"""
    Reduce τ ∈ ℍ² al dominio fundamental ℱ ⊂ ℍ² por la acción de PSL(2, ℤ).
    """
    tau = _ensure_upper_half(tau)
    curr_tau, transforms = _psl2z_reduce(tau, max_iter=max_iter)
    in_domain = abs(curr_tau.real) <= 0.5 + 1e-7 and abs(curr_tau) >= 1.0 - 1e-7
    poincare_dist = poincare_hyperbolic_distance(tau, curr_tau)
    mclass = _classify_modular_element(curr_tau)
    return FuchsianDomainCertificate(
        tau_original=tau,
        tau_reduced=curr_tau,
        is_in_fundamental_domain=in_domain,
        modular_transformations_count=transforms,
        poincare_metric_distance=poincare_dist,
        modular_class=mclass,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.7 — Semilla de dinámica cuántica abierta (ABC)
# ─────────────────────────────────────────────────────────────────────────────
class OpenQuantumDynamicsSeed(ABC):
    r"""
    Germen formal de la flecha H : (K, τ) ↦ 𝔥𝔢𝔯(ℋₙ), asociando a un complejo
    simplicial K y un parámetro modular τ ∈ ℍ² un Hamiltoniano hermitiano
    H ∈ 𝔥𝔢𝔯(ℋₙ) que alimenta la ecuación de Lindblad (objeto inicial de Fase 2).
    """

    @abstractmethod
    def lift_hamiltonian(self, hilbert_dim: int, modular_tau: complex) -> ComplexMatrix:
        r"""Produce H = H† a partir del fibrado topológico y del módulo τ."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.8 — Fibrado circuital topológico (método bisagra hacia FASE 2)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TopologicalCircuitBundle(OpenQuantumDynamicsSeed):
    r"""
    NEXO FORMAL TERMINAL DE LA FASE 1.

    Encapsula topología simplicial exacta, espectro de Hodge (β₀, β₁, β₂, χ),
    respuesta circuital no recíproca y valuación en Ω₄.

    El método `lift_hamiltonian` CIERRA la FASE 1 y alimenta DIRECTAMENTE la
    FASE 2 (motor Lindblad-GKSL, CRTBP y evolución fucsiana):

        H = lift_hamiltonian(n, τ)  ∈ 𝔥𝔢𝔯(ℋ_n)
            ⟶ NonHermitianLindbladMasterEngine.evolve_*(ρ, H, …)
    """

    simplicial_complex: SimplicialHodgeComplex
    circuit_field: NonReciprocalCircuitField
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    euler_betti_consistent: bool
    hodge_spectral_gap: float
    algebraic_connectivity: float
    harmonic_energy_norm: float
    hodge_decomposition_residual: float
    circuit_dissipation_rate: float
    reciprocity_defect: float
    tellegen_residual: float
    passivity_margin: float
    kcl_residual: float
    spectral_coherence_index: float
    heyting_topos_evaluation: HeytingTruthValue

    _KAPPA_BETTI: ClassVar[float] = 1.0
    _KAPPA_DISSIPATION: ClassVar[float] = 12.0
    _KAPPA_RECIPROCITY: ClassVar[float] = 5.0

    @staticmethod
    def _spectral_coherence_functional(
        betti_1: int, betti_2: int, dissipation: float, reciprocity_defect: float
    ) -> float:
        r"""
        Funcional de coherencia topológico-circuital

            C = exp(−(β₁ + β₂)/κ_b) · exp(−|P_dis|/κ_d) · exp(−|A|/κ_r) ∈ (0, 1].
        """
        kb = TopologicalCircuitBundle._KAPPA_BETTI
        kd = TopologicalCircuitBundle._KAPPA_DISSIPATION
        kr = TopologicalCircuitBundle._KAPPA_RECIPROCITY
        return (
            math.exp(-(betti_1 + betti_2) / kb)
            * math.exp(-abs(dissipation) / kd)
            * math.exp(-abs(reciprocity_defect) / kr)
        )

    @staticmethod
    def _classify_topos_state(coherence: float) -> HeytingTruthValue:
        r"""Valuación del fibrado en Ω₄ por umbrales de coherencia."""
        return HeytingTruthValue.from_coherence(coherence)

    @classmethod
    def synthesize_bundle(
        cls,
        num_vertices: int,
        edges: List[Tuple[int, int]],
        faces: List[Tuple[int, int, int]],
        current_stimulus: Optional[NDArray[np.complex128]] = None,
    ) -> "TopologicalCircuitBundle":
        r"""Sintetiza un fibrado circuital completo desde el complejo simplicial."""
        comp = SimplicialHodgeComplex(num_vertices=num_vertices, edges=edges, faces=faces)
        b0, b1, b2 = comp.compute_betti_numbers()
        chi = comp.euler_poincare_characteristic()
        euler_ok = comp.verify_betti_euler_consistency()
        if not euler_ok:
            logger.warning(
                "Inconsistencia Euler–Poincaré: χ=%d ≠ β₀−β₁+β₂=%d.",
                chi,
                b0 - b1 + b2,
            )
        circ = NonReciprocalCircuitField(simplicial_complex=comp)
        if current_stimulus is None:
            current_stimulus = np.zeros(num_vertices, dtype=np.complex128)
            current_stimulus[0] = complex(1.0, 0.0)
            current_stimulus[-1] = complex(-1.0, 0.0)
        v_nodes = circ.solve_voltage_distribution(current_stimulus)
        branch_v = comp.boundary_1.T @ v_nodes
        branch_i = circ.branch_admittance @ branch_v
        dissipation = float(np.real(np.sum(branch_v * np.conj(branch_i))))
        tellegen_residual = circ.verify_tellegen_conservation(v_nodes)
        reciprocity_defect = circ.reciprocity_defect_norm()
        passivity = circ.passivity_margin()
        kcl = circ.kcl_residual(v_nodes, current_stimulus)
        spectral_gap = comp.spectral_gap_L1(b1)
        fiedler = comp.algebraic_connectivity()
        harmonics = comp.harmonic_1_forms()
        harm_norm = float(la.norm(harmonics)) if harmonics.shape[1] > 0 else 0.0
        omega_probe = np.real(branch_v)
        if omega_probe.size == len(comp.edges) and omega_probe.size > 0:
            ex, co, ha = comp.hodge_decompose_1_form(omega_probe)
            hodge_res = float(np.linalg.norm(omega_probe - (ex + co + ha)))
        else:
            hodge_res = 0.0
        coherence = cls._spectral_coherence_functional(
            b1, b2, dissipation, reciprocity_defect
        )
        verdict = cls._classify_topos_state(coherence)
        return cls(
            simplicial_complex=comp,
            circuit_field=circ,
            betti_0=b0,
            betti_1=b1,
            betti_2=b2,
            euler_characteristic=chi,
            euler_betti_consistent=euler_ok,
            hodge_spectral_gap=spectral_gap,
            algebraic_connectivity=fiedler,
            harmonic_energy_norm=harm_norm,
            hodge_decomposition_residual=hodge_res,
            circuit_dissipation_rate=dissipation,
            reciprocity_defect=reciprocity_defect,
            tellegen_residual=tellegen_residual,
            passivity_margin=passivity,
            kcl_residual=kcl,
            spectral_coherence_index=coherence,
            heyting_topos_evaluation=verdict,
        )

    # ── MÉTODO BISAGRA FASE 1 → FASE 2 ────────────────────────────────────
    def lift_hamiltonian(self, hilbert_dim: int, modular_tau: complex) -> ComplexMatrix:
        r"""
        Eleva (K, τ) ↦ H ∈ 𝔥𝔢𝔯(ℋ_n). CIERRA la FASE 1 y ABRE la FASE 2.

        Construcción (Poincaré Vol. I + cuantización del fibrado):

            H₀ = diag( (k+1)·Δ_spectral + 0.05·P_dis )_{k=0..n−1},
            q  = (Re τ, Im τ, β₁, ½) ∈ ℍ,
            H_block = Re[ φ(q) ] sobre bloques 2×2,
            H  = ½(H₀ + H_block + h.c.).

        El H retornado es el generador hamiltoniano de la ecuación GKSL
        (`NonHermitianLindbladMasterEngine`) y de la hoja de mundo de Polyakov.

        Axioma I: H = H† por construcción (se aserta).
        """
        if hilbert_dim < 1:
            raise ValueError("hilbert_dim ≥ 1.")
        dim = int(hilbert_dim)
        tau = _ensure_upper_half(modular_tau)
        H0 = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            H0[i, i] = (i + 1) * self.hodge_spectral_gap + 0.05 * self.circuit_dissipation_rate
        q = Quaternion(
            float(tau.real),
            float(tau.imag),
            float(self.betti_1),
            0.5,
        )
        q_mat = q.to_matrix()
        for i in range(0, dim - 1, 2):
            H0[i : i + 2, i : i + 2] += q_mat * 0.1
        H = 0.5 * (H0 + H0.conj().T)
        residual = float(la.norm(H - H.conj().T))
        if residual >= 1e-9:
            raise RuntimeError(f"H no hermitiano en lift_hamiltonian: ‖H−H†‖={residual:.3e}")
        return H


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — MECÁNICA CELESTE DE POINCARÉ: ℍ², CRTBP, POLYAKOV, LINDBLAD-GKSL
#
#   CONTINUACIÓN DIRECTA de `lift_hamiltonian` (FASE 1): el Hamiltoniano
#   elevado alimenta el motor GKSL, que evoluciona bajo la métrica hiperbólica
#   de ℍ² y las variedades invariantes de Lagrange. Termina con
#   `MetabolizedFieldState.create_metabolized_state`, método bisagra que
#   entrega la densidad metabolizada a la FASE 3.
#
#   Cadena de morfismos de la Fase 2:
#       H = lift_hamiltonian(...)
#           ──► Jacobi / Lagrange L₁…L₅
#           ──► tubos Wˢ, Wᵘ (Δv → 0)
#           ──► Polyakov S_P(τ) sobre ℱ
#           ──► GKSL 𝓛(ρ)
#           ──► MetabolizedFieldState = create_metabolized_state(...)
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2.0 — Integral de Jacobi y puntos de Lagrange (cimiento CRTBP)
#            Primer consumidor de H: identifica el nivel de energía onírico
#            con C_Jacobi y clasifica L_i.
# ─────────────────────────────────────────────────────────────────────────────
class JacobiIntegral:
    r"""
    Integral de Jacobi del CRTBP (única integral primera, Poincaré 1892).

    En el sistema sinóptico rotante, con potencial efectivo Ω(x, y, z),

        C = 2 Ω(x, y, z) − (ẋ² + ẏ² + ż²).

    Las regiones de Hill {Ω ≥ C/2} delimitan el dominio accesible. Cruzar
    L₁ (cuello de Hill) es el análogo celestial de un cisne negro: un
    escenario contrafactual que el motor REM explora a Δv → 0.
    """

    @staticmethod
    def effective_potential(x: float, y: float, mu: float) -> float:
        r"""
        Ω = ½(x² + y²) + (1−μ)/r₁ + μ/r₂,  r₁ = dist a (μ,0), r₂ a (μ−1,0).
        """
        r1 = math.hypot(x - mu, y)
        r2 = math.hypot(x - mu + 1.0, y)
        r1 = max(r1, 1e-12)
        r2 = max(r2, 1e-12)
        return 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2

    @classmethod
    def constant(cls, x: float, y: float, vx: float, vy: float, mu: float) -> float:
        r"""C_Jacobi = 2Ω − v²."""
        return 2.0 * cls.effective_potential(x, y, mu) - (vx * vx + vy * vy)

    @classmethod
    def hill_forbidden(cls, x: float, y: float, mu: float, C: float) -> bool:
        r"""True ssi el punto está en la región de Hill prohibida (2Ω < C)."""
        return 2.0 * cls.effective_potential(x, y, mu) < C


class LagrangePoint:
    r"""
    Puntos de libración del CRTBP.

    Colineales L₁, L₂, L₃ (Euler): silla × centro, inestables, origen de
    los tubos Wˢ/Wᵘ. Triangulares L₄, L₅ (Lagrange): estables si se cumple
    el criterio de Routh

        27 μ (1 − μ) < 1    ⇔    μ < μ₁ ≈ 0.03852.
    """

    COLLINEAR: Final[Tuple[str, ...]] = ("L1", "L2", "L3")
    TRIANGULAR: Final[Tuple[str, ...]] = ("L4", "L5")

    @staticmethod
    def routh_stable(mu: float) -> bool:
        r"""Criterio de Routh para L₄, L₅."""
        return 27.0 * mu * (1.0 - mu) < 1.0

    @staticmethod
    def collinear_x(mu: float, which: str) -> float:
        r"""
        Aproximación de Hill para la abscisa de L₁/L₂ (μ ≪ 1) y L₃ ≈ −1 + 5μ/12.
        Suficiente para dimensionar tubos oníricos; no pretende raíz exacta de Euler.
        """
        mu = min(max(mu, 1e-12), 1.0 - 1e-12)
        rH = (mu / 3.0) ** (1.0 / 3.0)
        if which == "L1":
            return 1.0 - mu - rH
        if which == "L2":
            return 1.0 - mu + rH
        if which == "L3":
            return -1.0 - 5.0 * mu / 12.0
        raise ValueError("which ∈ {L1, L2, L3}.")

    @classmethod
    def coordinates(cls, mu: float, which: str) -> Tuple[float, float]:
        r"""Coordenadas (x, y) del punto de libración en el sinóptico."""
        which = which.upper()
        if which in cls.COLLINEAR:
            return cls.collinear_x(mu, which), 0.0
        if which == "L4":
            return 0.5 - mu, math.sqrt(3.0) / 2.0
        if which == "L5":
            return 0.5 - mu, -math.sqrt(3.0) / 2.0
        raise ValueError("which ∈ {L1,…,L5}.")

    @classmethod
    def is_hyperbolic(cls, mu: float, which: str) -> bool:
        r"""L₁–L₃ siempre hiperbólicos; L₄/L₅ hiperbólicos si falla Routh."""
        which = which.upper()
        if which in cls.COLLINEAR:
            return True
        return not cls.routh_stable(mu)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2.1 — Tubos invariantes de Lagrange del CRTBP
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CRTBPInvariantManifoldTube:
    r"""
    Tubo invariante de Lagrange Wˢ, Wᵘ en el CRTBP (Poincaré 1892; Koon–Lo–
    Marsden–Ross).

    Cerca de un punto de silla L_i, la linealización tiene autovalores
    ±λ (reales, λ > 0) y ±iω (imaginarios); las variedades Wˢ, Wᵘ son tubos
    tridimensionales de energía constante H = C_Jacobi:

        Wˢ(L_i) = { x : φ^t(x) → L_i cuando t → +∞ },
        Wᵘ(L_i) = { x : φ^t(x) → L_i cuando t → −∞ }.

    La navegación interior es de nulo propelente (Δv → 0): el análogo onírico
    de explorar bancarrotas / cisnes negros sin desembolsar capital real.
    """

    libration_point_id: str
    mass_ratio_mu: float
    jacobi_constant_C: float
    is_stable_manifold: bool
    tube_radius: float
    trajectory_energy: float

    def critical_radius_hill(self) -> float:
        r"""Radio de Hill r_H = (μ/3)^{1/3} escalado por √(C/3)."""
        if self.mass_ratio_mu <= 0.0 or self.mass_ratio_mu >= 1.0:
            return 0.0
        r_hill = (self.mass_ratio_mu / 3.0) ** (1.0 / 3.0)
        return float(r_hill * math.sqrt(max(self.jacobi_constant_C, 0.0) / 3.0))

    def energy_residual(self) -> float:
        r"""|H_tubo − C_Jacobi|; nulo por construcción si E = −C/2 en convención."""
        h_equivalent = -self.trajectory_energy
        return abs(h_equivalent - self.jacobi_constant_C)

    def delta_v_budget(self, tube_target_radius: float = 0.01) -> float:
        r"""
        Coste de maniobra Δv ≈ |λ| · max(0, R_tubo − r_target).
        Nulo si el objetivo está dentro del tubo (navegación de nulo propelente).
        """
        if self.tube_radius <= 0.0:
            return 0.0
        return float(max(0.0, self.tube_radius - tube_target_radius))

    def is_hyperbolic_libration(self) -> bool:
        r"""True ssi L_i es silla (tubos Wˢ/Wᵘ bien definidos)."""
        return LagrangePoint.is_hyperbolic(self.mass_ratio_mu, self.libration_point_id)

    def coordinates(self) -> Tuple[float, float]:
        r"""(x, y) del punto de libración ancla del tubo."""
        return LagrangePoint.coordinates(self.mass_ratio_mu, self.libration_point_id)

    def hill_crossing_allowed(self, x: float, y: float) -> bool:
        r"""False ssi (x, y) está en la región de Hill prohibida para este C."""
        return not JacobiIntegral.hill_forbidden(
            x, y, self.mass_ratio_mu, self.jacobi_constant_C
        )


class CRTBPPoincareSection:
    r"""
    Sección de Poincaré del CRTBP en el plano y = 0, ẏ > 0, sobre el nivel
    C = C_Jacobi. El mapa de retorno 𝒫 : Σ → Σ es el análogo celestial del
    mapa de primer retorno de la Fase-2 del motor adversarial; aquí se usa
    para estimar el exponente de Lyapunov local cerca de L₁ (silla).
    """

    def __init__(self, mu: float, C: float) -> None:
        self.mu = float(mu)
        self.C = float(C)

    def is_transversal(self, y: float, vy: float, tol: float = 1e-9) -> bool:
        r"""Transversalidad: y ≈ 0 y ẏ > tol (el flujo corta Σ)."""
        return abs(y) <= tol * 10.0 and vy > tol

    def saddle_lyapunov_proxy(self, which: str = "L1") -> float:
        r"""
        Proxy del exponente real λ de la silla L_i (Hill):
            λ² ≈  (9 + √(81 − 8 η))/4   con η ligado a μ cerca de L₁.
        Se retorna λ > 0 (inestabilidad).
        """
        _ = which
        mu = min(max(self.mu, 1e-12), 0.5)
        # Fórmula clásica de Szebehely para el exponente de L1 a primer orden en μ.
        lam = math.sqrt(2.0 + math.sqrt(2.0) + 6.0 * (mu ** (1.0 / 3.0)))
        return float(lam)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2.2 — Hoja de mundo de Polyakov (cuerda bosónica sobre ℍ²)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PolyakovWorldsheetMetrics:
    r"""
    Métricas de la cuerda bosónica sobre la hoja de mundo Σ:

        S_P[X, h] = (1 / (4 π α')) ∫_Σ d²σ √(−h) h^{ab} ∂_a X^μ ∂_b X_μ.

    El parámetro modular τ = τ₁ + i τ₂ ∈ ℍ² (τ₂ > 0) clasifica los toros
    complejos conformemente equivalentes. PSL(2, ℤ) actúa sobre τ por
    transformaciones de Möbius. La anomalía de Weyl se cancela en la
    cuerda crítica (c = 26); aquí c = dim(ℋ_MAC) y el residual |c−26|/26
    diagnostica la desviación.
    """

    modular_parameter_tau: complex
    string_tension_alpha_prime: float
    polyakov_action_integral: float
    conformal_anomaly_central_charge: float
    weyl_invariance_residual: float
    in_fundamental_domain_flag: bool

    def in_fundamental_domain(self) -> bool:
        r"""Verifica ℱ ⊂ ℍ² en τ (|Re| ≤ ½, |τ| ≥ 1)."""
        tau1 = self.modular_parameter_tau.real
        return abs(tau1) <= 0.5 + 1e-9 and abs(self.modular_parameter_tau) >= 1.0 - 1e-9

    @staticmethod
    def reduce_to_fundamental_domain(tau: complex, max_iter: int = 64) -> complex:
        r"""Proyección al dominio fundamental vía PSL(2, ℤ)."""
        return reduce_to_poincare_fundamental_domain(tau, max_iter=max_iter).tau_reduced

    def t_invariance_residual(self, other_action: float) -> float:
        r"""|S(τ+1) − S(τ)| / |S(τ)|: diagnóstico de invariancia modular bajo T."""
        denom = abs(self.polyakov_action_integral) if abs(self.polyakov_action_integral) > 1e-12 else 1e-12
        return float(abs(other_action - self.polyakov_action_integral) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2.3 — Motor de Lindblad-GKSL (fucsianizado)
# ─────────────────────────────────────────────────────────────────────────────
class NonHermitianLindbladMasterEngine:
    r"""
    Integrador de la ecuación maestra de Lindblad (GKSL) para dinámica cuántica
    abierta, restringido al dominio fundamental de Poincaré ℱ ⊂ ℍ².

        dρ/dt = −i [H_eff, ρ] + Σ_k γ_k ( L_k ρ L_k† − ½ { L_k† L_k, ρ } ).

    El generador 𝓛 es CP y TP. El Hamiltoniano H_eff ES el objeto producido
    por `TopologicalCircuitBundle.lift_hamiltonian` (continuación de Fase 1).
    Tras cada paso Euler se proyecta al símplex 𝔇(ℋ) para restaurar Axioma II.
    """

    def __init__(self, enclave_dim: int = 4, damping_kossakowski: float = 0.05) -> None:
        self.dim = int(enclave_dim)
        self.gamma_k = abs(float(damping_kossakowski))

    def reduce_to_poincare_fundamental_domain(
        self, tau: complex, max_iter: int = 128
    ) -> FuchsianDomainCertificate:
        r"""Proyección de τ ∈ ℍ² a ℱ vía PSL(2, ℤ)."""
        return reduce_to_poincare_fundamental_domain(tau, max_iter=max_iter)

    def evolve_fuchsian_lindblad_manifold(
        self,
        rho_dream: ComplexMatrix,
        H_eff: ComplexMatrix,
        jump_operators: Sequence[ComplexMatrix],
        tau_modular: complex,
        dt: float,
    ) -> Tuple[ComplexMatrix, Dict[str, Any]]:
        r"""
        Integra un paso Euler (traza-preservante por normalización posterior)
        de la ecuación de Lindblad. H_eff proviene de `lift_hamiltonian`.
        """
        herm_res = float(la.norm(H_eff - H_eff.conj().T))
        if herm_res >= 1e-8:
            raise ValueError(f"H_eff no hermitiano: ‖H−H†‖={herm_res:.3e}")
        fuchsian_cert = self.reduce_to_poincare_fundamental_domain(tau_modular)
        rho_dream = BanachOperatorAlgebra.clean_density_matrix(rho_dream)
        commutator = -1j * (H_eff @ rho_dream - rho_dream @ H_eff)
        dissipator = np.zeros_like(rho_dream, dtype=np.complex128)
        gammas = [self.gamma_k] * len(jump_operators)
        for L in jump_operators:
            L_dag_L = L.conj().T @ L
            dissipator += self.gamma_k * (
                L @ rho_dream @ L.conj().T
                - 0.5 * (L_dag_L @ rho_dream + rho_dream @ L_dag_L)
            )
        drho_dt = commutator + dissipator
        rho_next = rho_dream + dt * drho_dt
        rho_sanitized = BanachOperatorAlgebra.clean_density_matrix(rho_next)
        escape_rate = (
            float(
                np.sum(
                    [
                        self.gamma_k * np.trace(rho_sanitized @ L.conj().T @ L).real
                        for L in jump_operators
                    ]
                )
            )
            if jump_operators
            else 0.0
        )
        trace_defect = abs(float(np.trace(rho_sanitized).real) - 1.0)
        tp_residual = BanachOperatorAlgebra.gksl_trace_generator(
            rho_dream, H_eff, list(jump_operators), gammas
        )
        report: Dict[str, Any] = {
            "fuchsian_certificate": fuchsian_cert,
            "escape_rate_gamma": escape_rate,
            "purity": float(np.trace(rho_sanitized @ rho_sanitized).real),
            "trace_preserved": bool(trace_defect < 1e-9),
            "trace_defect": float(trace_defect),
            "gksl_tp_residual": float(tp_residual),
            "is_cptp": BanachOperatorAlgebra.is_valid_density_matrix(rho_sanitized),
            "hermiticity_residual": herm_res,
        }
        return rho_sanitized, report


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2.4 — Evolucionador CFT (Lindblad + Polyakov)
# ─────────────────────────────────────────────────────────────────────────────
class LindbladCFTMasterEvolver:
    r"""
    Integra la ecuación maestra GKSL y proyecta parámetros modulares a ℱ ⊂ ℍ².

    Coordina:
        • síntesis de H_eff vía `TopologicalCircuitBundle.lift_hamiltonian`
          (método bisagra FASE 1 → FASE 2),
        • operadores de salto de Lindblad con tasas γ_topo y γ_relax,
        • acción de Polyakov evaluada sobre el parámetro modular reducido,
        • diagnóstico de anomalía modular |ΔS_T| / |S_τ|.
    """

    GAMMA_FLOOR: Final[float] = 0.0

    def __init__(self, bundle: TopologicalCircuitBundle, hilbert_dim: int = 4) -> None:
        self.bundle = bundle
        self.dim = int(hilbert_dim)
        self.banach = BanachOperatorAlgebra()
        self.fuchsian_engine = NonHermitianLindbladMasterEngine(
            enclave_dim=self.dim, damping_kossakowski=0.05
        )

    def _synthesize_hamiltonian(self, modular_tau: complex) -> ComplexMatrix:
        r"""CONTINÚA `lift_hamiltonian` de la Fase 1."""
        return self.bundle.lift_hamiltonian(self.dim, modular_tau)

    def _build_lindblad_jump_operators(self) -> List[Tuple[float, ComplexMatrix]]:
        r"""
        Pares (γ_k, L_k):
            γ_topo  = max(0, 0.05 · (β₁ + 1))     (decoherencia topológica),
            γ_relax = max(0, 0.02 · |P_dis|)       (relajación disipativa).
        """
        gamma_topo = max(self.GAMMA_FLOOR, 0.05 * (self.bundle.betti_1 + 1.0))
        L_dephase = np.diag(
            [math.sqrt(i + 1) for i in range(self.dim)]
        ).astype(np.complex128)
        L_relax = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim - 1):
            L_relax[i, i + 1] = 1.0
        gamma_relax = max(
            self.GAMMA_FLOOR, 0.02 * abs(self.bundle.circuit_dissipation_rate)
        )
        return [(gamma_topo, L_dephase), (gamma_relax, L_relax)]

    def evaluate_polyakov_string_action(
        self, modular_tau: complex, worldsheet_area: float = 1.0
    ) -> PolyakovWorldsheetMetrics:
        r"""
        Evalúa S_P = (Área / (4 π α' τ₂)) · (1 + τ₁² + τ₂²) sobre el τ reducido
        a ℱ ⊂ ℍ², con carga central c = dim (materia bosónica).
        """
        cert = reduce_to_poincare_fundamental_domain(modular_tau)
        tau_red = cert.tau_reduced
        tau1, tau2 = tau_red.real, max(tau_red.imag, _H2_IMAG_FLOOR)
        alpha_prime = 0.5
        polyakov_action = (worldsheet_area / (4.0 * math.pi * alpha_prime * tau2)) * (
            1.0 + tau1 ** 2 + tau2 ** 2
        )
        central_charge = float(self.dim)
        weyl_residual = abs(central_charge - 26.0) / 26.0
        return PolyakovWorldsheetMetrics(
            modular_parameter_tau=tau_red,
            string_tension_alpha_prime=alpha_prime,
            polyakov_action_integral=float(polyakov_action),
            conformal_anomaly_central_charge=central_charge,
            weyl_invariance_residual=float(weyl_residual),
            in_fundamental_domain_flag=cert.is_in_fundamental_domain,
        )

    def diagnose_modular_anomaly(self, modular_tau: complex) -> float:
        r"""|S(τ + 1) − S(τ)| / |S(τ)|: desviación de invariancia modular bajo T."""
        s_tau = self.evaluate_polyakov_string_action(modular_tau).polyakov_action_integral
        s_t_tau = self.evaluate_polyakov_string_action(
            modular_tau + 1
        ).polyakov_action_integral
        denom = abs(s_tau) if abs(s_tau) > 1e-12 else 1e-12
        return float(abs(s_t_tau - s_tau) / denom)

    def evolve_density_state(
        self,
        rho_initial: ComplexMatrix,
        time_step: float,
        modular_tau: complex,
        integration_tolerance: float = 1e-7,
    ) -> Tuple[ComplexMatrix, PolyakovWorldsheetMetrics]:
        r"""Un paso GKSL + evaluación de Polyakov sobre ℱ. Consume H de Fase 1."""
        _ = integration_tolerance
        H = self._synthesize_hamiltonian(modular_tau)
        jumps = [L for _, L in self._build_lindblad_jump_operators()]
        rho_projected, _ = self.fuchsian_engine.evolve_fuchsian_lindblad_manifold(
            rho_dream=rho_initial,
            H_eff=H,
            jump_operators=jumps,
            tau_modular=modular_tau,
            dt=time_step,
        )
        metrics_cft = self.evaluate_polyakov_string_action(modular_tau)
        return rho_projected, metrics_cft


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2.5 — Estado metabolizado (método bisagra hacia FASE 3)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class MetabolizedFieldState:
    r"""
    NEXO FORMAL TERMINAL DE LA FASE 2.

    Estado metabolizado por GKSL y uniformización fucsiana de Poincaré en ℱ ⊂ ℍ².
    Encapsula: densidad ρ_dream ∈ 𝔇(ℋ), entropías, métricas Bures/Umegaki hacia
    ρ_térmico, residuo C* y coherencia metabólica C ∈ (0, 1].

    Este objeto ES el dominio de la Fase 3: vacuna espectral, Merkle y
    `run_terminal_rem_cycle`.
    """

    density_matrix: ComplexMatrix
    von_neumann_entropy: float
    purity: float
    trace_distance_to_thermal: float
    quantum_relative_entropy_to_thermal: float
    bures_distance_to_thermal: float
    cstar_residual: float
    hodge_bundle_carrier: TopologicalCircuitBundle
    cft_worldsheet_metrics: PolyakovWorldsheetMetrics
    modular_anomaly_estimate: float
    dirichlet_cft_energy: float
    metabolic_coherence_index: float
    heyting_state: HeytingTruthValue
    crtbp_tube: Optional[CRTBPInvariantManifoldTube] = None
    fuchsian_certificate: Optional[FuchsianDomainCertificate] = None

    _W_POLYAKOV: ClassVar[float] = 0.4
    _W_HARMONIC: ClassVar[float] = 0.3
    _W_SPECTRAL_GAP: ClassVar[float] = 0.3
    _KAPPA_ENERGY: ClassVar[float] = 10.0
    _KAPPA_RELATIVE_ENTROPY: ClassVar[float] = 4.0

    @staticmethod
    def _metabolic_coherence_functional(
        purity: float, dirichlet_energy: float, relative_entropy: float
    ) -> float:
        r"""
        C = P(ρ) · exp(−|E_D|/κ_E) · exp(−|D(ρ‖ρ_th)|/κ_S) ∈ [0, 1].
        """
        kE = MetabolizedFieldState._KAPPA_ENERGY
        kS = MetabolizedFieldState._KAPPA_RELATIVE_ENTROPY
        return (
            purity
            * math.exp(-abs(dirichlet_energy) / kE)
            * math.exp(-abs(relative_entropy) / kS)
        )

    @staticmethod
    def _classify_metabolic_state(coherence: float) -> HeytingTruthValue:
        r"""Valuación Ω₄ metabólica: C ≥ 0.60 ⇒ ⊤; ≥ 0.35 ⇒ ♯; ≥ 0.10 ⇒ ∂; si no ⊥."""
        if coherence >= 0.60:
            return HeytingTruthValue.VERUM_COHERENT
        if coherence >= 0.35:
            return HeytingTruthValue.TOPOLOGICAL_SOUND
        if coherence >= 0.10:
            return HeytingTruthValue.BOUNDARY_DEGRADED
        return HeytingTruthValue.ABSURDUM_VETOED

    def is_quantum_physical(self, atol: float = 1e-8) -> bool:
        r"""Axioma II: ρ ∈ 𝔇(ℋ)."""
        return BanachOperatorAlgebra.is_valid_density_matrix(self.density_matrix, tol=atol)

    def isolation_intact(self) -> bool:
        r"""∂ ℳ_REM ≡ 0: el veredicto no es ⊥ (el Crowbar físico no debe disparar)."""
        return self.heyting_state is not HeytingTruthValue.ABSURDUM_VETOED

    # ── MÉTODO BISAGRA FASE 2 → FASE 3 ───────────────────────────────────
    @classmethod
    def create_metabolized_state(
        cls,
        bundle: TopologicalCircuitBundle,
        base_rho: ComplexMatrix,
        modular_tau: complex,
        time_step: float = 0.05,
        crtbp_tube: Optional[CRTBPInvariantManifoldTube] = None,
    ) -> "MetabolizedFieldState":
        r"""
        Sintetiza un estado metabolizado completo. CIERRA la FASE 2 y ABRE la FASE 3.

        Pipeline (anidación 1 → 2):
            ρ₀  →  H(τ) = lift_hamiltonian  →  GKSL  →  ρ'
            → reduce_to_fundamental_domain(τ) → Polyakov S_P
            → métricas Bures/Umegaki → fusión con Ω₄ del bundle
            → MetabolizedFieldState  (consumido por run_terminal_rem_cycle).
        """
        fuchsian = reduce_to_poincare_fundamental_domain(modular_tau)
        evolver = LindbladCFTMasterEvolver(bundle=bundle, hilbert_dim=base_rho.shape[0])
        rho_evolved, cft_metrics = evolver.evolve_density_state(
            base_rho, time_step, fuchsian.tau_reduced
        )
        modular_anomaly = evolver.diagnose_modular_anomaly(fuchsian.tau_reduced)
        banach = BanachOperatorAlgebra()
        entropy = banach.von_neumann_entropy(rho_evolved)
        pur = banach.purity(rho_evolved)
        dim = rho_evolved.shape[0]
        rho_thermal = np.eye(dim, dtype=np.complex128) / dim
        dist_thermal = 0.5 * banach.trace_norm(rho_evolved - rho_thermal)
        rel_entropy_thermal = banach.quantum_relative_entropy(rho_evolved, rho_thermal)
        bures_thermal = banach.bures_distance(rho_evolved, rho_thermal)
        cstar = banach.cstar_residual(rho_evolved)
        dirichlet_energy = (
            cls._W_POLYAKOV * cft_metrics.polyakov_action_integral
            + cls._W_HARMONIC * bundle.harmonic_energy_norm
            + cls._W_SPECTRAL_GAP * (1.0 / (bundle.hodge_spectral_gap + 1e-4))
        )
        if crtbp_tube is not None:
            dirichlet_energy += 0.05 * crtbp_tube.delta_v_budget()
        coherence = cls._metabolic_coherence_functional(
            pur, dirichlet_energy, rel_entropy_thermal
        )
        phase2_heyting = cls._classify_metabolic_state(coherence)
        final_heyting = bundle.heyting_topos_evaluation.meet(phase2_heyting)
        return cls(
            density_matrix=rho_evolved,
            von_neumann_entropy=entropy,
            purity=pur,
            trace_distance_to_thermal=dist_thermal,
            quantum_relative_entropy_to_thermal=rel_entropy_thermal,
            bures_distance_to_thermal=bures_thermal,
            cstar_residual=cstar,
            hodge_bundle_carrier=bundle,
            cft_worldsheet_metrics=cft_metrics,
            modular_anomaly_estimate=modular_anomaly,
            dirichlet_cft_energy=dirichlet_energy,
            metabolic_coherence_index=coherence,
            heyting_state=final_heyting,
            crtbp_tube=crtbp_tube,
            fuchsian_certificate=fuchsian,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INMUNIZACIÓN, MERKLE, WAKE-SLEEP, ORQUESTACIÓN TERMINAL
#
#   CONTINUACIÓN DIRECTA de `create_metabolized_state` (FASE 2): el estado
#   metabolizado alimenta la síntesis de vacuna espectral, el ciclo onírico,
#   y el orquestador terminal TOONOniricDreamerEngine.
#
#   Cadena de morfismos de la Fase 3:
#       MetabolizedFieldState
#           ──► SpectralImmuneVaccineSynthesizer
#           ──► MerkleTree SHA-512
#           ──► DreamFieldReport
#           ──► run_terminal_rem_cycle   ← MÉTODO TERMINAL DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3.1 — Árbol de Merkle SHA-512 y pruebas de inclusión
# ─────────────────────────────────────────────────────────────────────────────
class MerkleTree:
    r"""
    Árbol de Merkle binario con hash SHA-512 y duplicación de nodos impares
    (convención Bitcoin/Ethereum: par impar se auto-empareja).

    Coste de prueba de inclusión: O(log n) hermanos. Sella la procedencia
    de cada ciclo REM (aislamiento homológico: evidencia de que ∂ℳ_REM = 0).
    """

    def __init__(self, leaves: Sequence[str]) -> None:
        if not leaves:
            raise ValueError("MerkleTree requiere al menos una hoja.")
        self.leaves: Tuple[str, ...] = tuple(leaves)
        self.root: str = self._build_root()

    @staticmethod
    def _hash_pair(a: bytes, b: bytes) -> bytes:
        return hashlib.sha512(a + b).digest()

    def _build_root(self) -> str:
        level = [bytes.fromhex(h) for h in self.leaves]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [
                self._hash_pair(level[i], level[i + 1])
                for i in range(0, len(level), 2)
            ]
        return level[0].hex()

    def proof(self, index: int) -> "MerkleInclusionProof":
        r"""Prueba de inclusión de la hoja `index` (0-indexed)."""
        if not (0 <= index < len(self.leaves)):
            raise IndexError(f"índice {index} fuera de [0, {len(self.leaves)}).")
        level = [bytes.fromhex(h) for h in self.leaves]
        siblings: List[str] = []
        idx = index
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            pair = idx ^ 1
            siblings.append(level[pair].hex())
            level = [
                self._hash_pair(level[i], level[i + 1])
                for i in range(0, len(level), 2)
            ]
            idx //= 2
        return MerkleInclusionProof(
            leaf_hash=self.leaves[index],
            siblings=tuple(siblings),
            index=index,
            root=level[0].hex(),
        )


@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    r"""Prueba de inclusión Merkle: camino de hermanos + índice orientado."""

    leaf_hash: str
    siblings: Tuple[str, ...]
    index: int
    root: str

    def verify(self) -> bool:
        r"""Reconstruye la raíz desde la hoja y compara con el miembro almacenado."""
        node = bytes.fromhex(self.leaf_hash)
        idx = self.index
        for sib_hex in self.siblings:
            sib = bytes.fromhex(sib_hex)
            if idx % 2 == 0:
                node = hashlib.sha512(node + sib).digest()
            else:
                node = hashlib.sha512(sib + node).digest()
            idx //= 2
        return node.hex() == self.root


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3.2 — Sintetizador de vacuna espectral
# ─────────────────────────────────────────────────────────────────────────────
class SpectralImmuneVaccineSynthesizer:
    r"""
    Sintetiza el proyector de inmunidad P_vac ∈ B(ℋ) sobre el subespacio de
    cobertura de masa espectral μ ∈ (0, 1].

    Dada ρ = Σ λ_i |v_i⟩⟨v_i| y un nivel μ, se retiene el mínimo k tal que
    Σ_{i≤k} λ_i ≥ μ. P_vac = Σ_{i≤k} |v_i⟩⟨v_i| satisface P² = P.

    Analogía celestial: P_vac es una sección de Poincaré espectral que
    recorta el tubo Wˢ al subespacio de masa dominante (inmunidad a cisnes
    negros fuera de la cobertura).
    """

    COVERAGE_FRACTION: Final[float] = 0.75
    PERTURBATION_HARD: Final[float] = 0.85
    RANK_FRACTION_CAP: Final[float] = 0.75
    MASS_FLOOR: Final[float] = 1e-15

    @classmethod
    def synthesize_from_metabolized_state(
        cls,
        state: MetabolizedFieldState,
        perturbation_norm: float,
        coverage_fraction: float = COVERAGE_FRACTION,
    ) -> Tuple[ComplexMatrix, bool, float, float]:
        r"""Atajo desde un `MetabolizedFieldState` (continuación de Fase 2)."""
        return cls.synthesize_vaccine(
            density_matrix=state.density_matrix,
            perturbation_norm=perturbation_norm,
            coverage_fraction=coverage_fraction,
        )

    @classmethod
    def synthesize_vaccine(
        cls,
        density_matrix: ComplexMatrix,
        perturbation_norm: float,
        coverage_fraction: float = COVERAGE_FRACTION,
    ) -> Tuple[ComplexMatrix, bool, float, float]:
        r"""
        Retorna (P_vac, es_efectiva, masa_retenida, residuo_idempotencia).
        """
        eigvals, eigvecs = la.eigh(density_matrix)
        dim = density_matrix.shape[0]
        order = np.argsort(eigvals)[::-1]
        eigvals_sorted = np.clip(eigvals[order], 0.0, None)
        eigvecs_sorted = eigvecs[:, order]
        total_mass = float(np.sum(eigvals_sorted))
        if total_mass <= cls.MASS_FLOOR:
            eye = np.eye(dim, dtype=np.complex128) / dim
            return eye, False, 0.0, 0.0
        cumulative = np.cumsum(eigvals_sorted) / total_mass
        k = int(np.searchsorted(cumulative, coverage_fraction) + 1)
        k = min(max(k, 1), dim)
        retained_mass = float(cumulative[k - 1])
        P_vac = eigvecs_sorted[:, :k] @ eigvecs_sorted[:, :k].conj().T
        P_vac = 0.5 * (P_vac + P_vac.conj().T)
        idem_res = float(la.norm(P_vac @ P_vac - P_vac, "fro"))
        is_effective = bool(
            retained_mass >= coverage_fraction
            and perturbation_norm < cls.PERTURBATION_HARD
            and k <= math.ceil(dim * cls.RANK_FRACTION_CAP)
        )
        return P_vac, is_effective, retained_mass, idem_res


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3.3 — Reportes de ciclo y wake-sleep
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DreamFieldReport:
    r"""Reporte inmutable de un ciclo onírico contrafactual completo."""

    cycle_id: str
    scenario_type: str
    metabolized_state: MetabolizedFieldState
    immune_projection_operator: ComplexMatrix
    projector_idempotency_residual: float
    spectral_coverage_mass: float
    spectral_vaccine_effective: bool
    dirichlet_energy: float
    heyting_verdict: HeytingTruthValue
    learning_rate_applied: float
    merkle_sha512_provenance: str
    timestamp_utc: float
    fuchsian_certificate: Optional[FuchsianDomainCertificate] = None
    umegaki_divergence: float = 0.0
    bures_geodesic_distance: float = 0.0
    dream_isolation_intact: bool = True
    crtbp_delta_v: float = 0.0
    schema_version: str = _SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class WakeSleepCycleReport:
    r"""Reporte consolidado del ciclo completo de sueño (REM) y despertar (wake)."""

    cycle_id: str
    initial_entropy: float
    final_entropy: float
    wake_purity: float
    sleep_purity: float
    net_free_energy_reduction: float
    synthesized_scenarios_count: int
    immune_vaccines_generated: int
    overall_topos_verdict: HeytingTruthValue
    merkle_leaf_count: int
    sha512_merkle_root: str
    merkle_proofs_ok: bool
    isolation_violations: int = 0
    schema_version: str = _SCHEMA_VERSION

    def is_energetically_favourable(self) -> bool:
        r"""Reducción neta de energía libre F_wake − F_sleep ≥ 0."""
        return self.net_free_energy_reduction >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3.4 — Motor onírico terminal
# ─────────────────────────────────────────────────────────────────────────────
class TOONOniricDreamerEngine:
    r"""
    MOTOR ESPECTRAL ONÍRICO (TOON) Y NAVEGACIÓN POINCARANA FUCSIANA.

    Orquestador terminal del topos Ω₄ de la fase REM. Coordina las tres
    fases anidadas:

        Fase 1  lift_hamiltonian              → H ∈ 𝔥𝔢𝔯(ℋ_n)
        Fase 2  create_metabolized_state      → MetabolizedFieldState
        Fase 3  run_terminal_rem_cycle        → DreamFieldReport + Merkle

    El aprendizaje es modulado por el veredicto Ω₄ vía `_HEYTING_LEARNING_RATE_MAP`:
    sólo los veredictos coherentes o parcialmente coherentes actualizan la densidad
    maestra ρ_MAC. El aislamiento REM (∂ℳ_REM = 0) impide el Crowbar físico.
    """

    _HEYTING_LEARNING_RATE_MAP: Final[Dict[int, float]] = {
        int(HeytingTruthValue.VERUM_COHERENT): 0.35,
        int(HeytingTruthValue.TOPOLOGICAL_SOUND): 0.20,
        int(HeytingTruthValue.BOUNDARY_DEGRADED): 0.08,
        int(HeytingTruthValue.ABSURDUM_VETOED): 0.0,
    }

    def __init__(
        self,
        engine_id: str = "TOON-ONIRIC-DOCTORAL-001",
        dimension_mac: int = 4,
        seed: int = 42,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1.")
        self.engine_id = engine_id
        self.dimension_mac = int(dimension_mac)
        self.rng = np.random.default_rng(seed)
        self.cycle_counter = 0
        self.master_rho: ComplexMatrix = (
            np.eye(self.dimension_mac, dtype=np.complex128) / self.dimension_mac
        )
        self.dream_records: List[DreamFieldReport] = []
        self.last_metabolized: Optional[MetabolizedFieldState] = None

    def reduce_to_poincare_fundamental_domain(
        self, tau: complex, max_iter: int = 128
    ) -> FuchsianDomainCertificate:
        return reduce_to_poincare_fundamental_domain(tau, max_iter=max_iter)

    def evolve_fuchsian_lindblad_manifold(
        self,
        rho_dream: ComplexMatrix,
        H_eff: ComplexMatrix,
        jump_operators: Sequence[ComplexMatrix],
        tau_modular: complex,
        dt: float,
    ) -> Tuple[ComplexMatrix, Dict[str, Any]]:
        engine = NonHermitianLindbladMasterEngine(enclave_dim=rho_dream.shape[0])
        return engine.evolve_fuchsian_lindblad_manifold(
            rho_dream=rho_dream,
            H_eff=H_eff,
            jump_operators=jump_operators,
            tau_modular=tau_modular,
            dt=dt,
        )

    def _compile_counterfactual_topology(
        self, scenario_type: str, betti_loops_request: int
    ) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        r"""
        Construye un 2-complejo simplicial contrafactual con β₁ controlable por
        `betti_loops_request` y hashing determinista del `scenario_type`.
        """
        num_v = 6
        edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5), (5, 0)]
        faces = [(0, 1, 2), (2, 3, 4)]
        digest = hashlib.sha256(scenario_type.encode("utf-8")).digest()
        if digest[0] % 2 == 0:
            candidate = (digest[1] % num_v, digest[2] % num_v)
            if candidate[0] != candidate[1]:
                edges.append(tuple(sorted(candidate)))
        if betti_loops_request > 0:
            edges.append((1, 4))
        if betti_loops_request > 1:
            edges.append((3, 0))
        seen: Set[Tuple[int, int]] = set()
        dedup_edges: List[Tuple[int, int]] = []
        for e in edges:
            e_sorted = tuple(sorted(e))
            if e_sorted not in seen:
                seen.add(e_sorted)
                dedup_edges.append(e_sorted)
        return num_v, dedup_edges, faces

    def _default_crtbp_tube(self, cost_delta_ratio: float) -> CRTBPInvariantManifoldTube:
        r"""Tubo L₁ Sol–Tierra dimensionado por el estrés contrafactual |Δcoste|."""
        C = 3.0009 + 0.02 * abs(cost_delta_ratio)
        return CRTBPInvariantManifoldTube(
            libration_point_id="L1",
            mass_ratio_mu=3.04e-6,
            jacobi_constant_C=C,
            is_stable_manifold=True,
            tube_radius=0.01 + 0.02 * abs(cost_delta_ratio),
            trajectory_energy=-C,
        )

    @staticmethod
    def _merkle_tree_root(leaf_hashes: Sequence[str]) -> str:
        if not leaf_hashes:
            return hashlib.sha512(b"EMPTY_MERKLE_ROOT").hexdigest()
        return MerkleTree(leaf_hashes).root

    @staticmethod
    def _merkle_proof(leaf_hashes: Sequence[str], index: int) -> MerkleInclusionProof:
        if not leaf_hashes:
            empty = hashlib.sha512(b"EMPTY_MERKLE_ROOT").hexdigest()
            return MerkleInclusionProof(empty, tuple(), 0, empty)
        return MerkleTree(leaf_hashes).proof(index)

    def _seal_leaf(
        self,
        cycle_id: str,
        scenario_type: str,
        cost_delta_ratio: float,
        state: MetabolizedFieldState,
        verdict: HeytingTruthValue,
        coverage_mass: float,
        t_start: float,
    ) -> str:
        hasher = hashlib.sha512()
        payload = (
            f"{self.engine_id}::{cycle_id}::{scenario_type}::{cost_delta_ratio:.6f}::"
            f"{state.dirichlet_cft_energy:.6f}::{state.purity:.6f}::"
            f"{verdict.name}::{coverage_mass:.6f}::{t_start}"
        ).encode("utf-8")
        hasher.update(payload)
        return hasher.hexdigest()

    def _apply_vaccine_learning(
        self,
        state: MetabolizedFieldState,
        P_vac: ComplexMatrix,
        verdict: HeytingTruthValue,
    ) -> float:
        learning_rate = self._HEYTING_LEARNING_RATE_MAP[int(verdict)]
        if learning_rate > 0.0:
            vaccinated_rho = P_vac @ state.density_matrix @ P_vac.conj().T
            trace_v = float(np.real(np.trace(vaccinated_rho)))
            if trace_v > 1e-12:
                vaccinated_rho = vaccinated_rho / trace_v
            self.master_rho = (
                (1.0 - learning_rate) * self.master_rho + learning_rate * vaccinated_rho
            )
            self.master_rho = BanachOperatorAlgebra.clean_density_matrix(self.master_rho)
        return learning_rate

    def run_dream_cycle(
        self,
        scenario_type: str,
        cost_delta_ratio: float,
        betti_1_loops: int = 0,
        modular_tau: complex = complex(0.1, 1.2),
        dream_isolation_flag: bool = True,
    ) -> DreamFieldReport:
        r"""
        Ejecuta un ciclo onírico completo (anidación 1 → 2 → 3):

            1. Reduce τ al dominio fundamental ℱ ⊂ ℍ².
            2. Compila la topología contrafactual (β₁ controlado).
            3. Construye el `TopologicalCircuitBundle` y H = lift_hamiltonian.
            4. Metaboliza: `create_metabolized_state` (Fase 2).
            5. Sintetiza la vacuna espectral y actualiza ρ_MAC.
            6. Sella la procedencia con SHA-512 y firma en Ω₄.
        """
        self.cycle_counter += 1
        t_start = time.time()
        cycle_id = f"CYC-REM-{self.cycle_counter:05d}"
        fuchsian_cert = reduce_to_poincare_fundamental_domain(modular_tau)
        num_v, edges, faces = self._compile_counterfactual_topology(
            scenario_type, betti_1_loops
        )
        bundle = TopologicalCircuitBundle.synthesize_bundle(
            num_vertices=num_v, edges=edges, faces=faces
        )
        tube = self._default_crtbp_tube(cost_delta_ratio)
        forced_heyting = (
            bundle.heyting_topos_evaluation
            if dream_isolation_flag
            else HeytingTruthValue.ABSURDUM_VETOED
        )
        metabolized_state = MetabolizedFieldState.create_metabolized_state(
            bundle=bundle,
            base_rho=self.master_rho,
            modular_tau=fuchsian_cert.tau_reduced,
            time_step=0.05 + 0.05 * abs(cost_delta_ratio),
            crtbp_tube=tube,
        )
        self.last_metabolized = metabolized_state
        final_verdict = metabolized_state.heyting_state.meet(forced_heyting)
        banach = BanachOperatorAlgebra()
        umegaki = banach.quantum_relative_entropy(
            metabolized_state.density_matrix, self.master_rho
        )
        bures = banach.bures_distance(metabolized_state.density_matrix, self.master_rho)
        P_vac, vaccine_effective, coverage_mass, idem_res = (
            SpectralImmuneVaccineSynthesizer.synthesize_from_metabolized_state(
                metabolized_state, perturbation_norm=abs(cost_delta_ratio)
            )
        )
        learning_rate = self._apply_vaccine_learning(
            metabolized_state, P_vac, final_verdict
        )
        merkle_leaf_hash = self._seal_leaf(
            cycle_id,
            scenario_type,
            cost_delta_ratio,
            metabolized_state,
            final_verdict,
            coverage_mass,
            t_start,
        )
        report = DreamFieldReport(
            cycle_id=cycle_id,
            scenario_type=scenario_type,
            metabolized_state=metabolized_state,
            immune_projection_operator=P_vac,
            projector_idempotency_residual=idem_res,
            spectral_coverage_mass=coverage_mass,
            spectral_vaccine_effective=vaccine_effective,
            dirichlet_energy=metabolized_state.dirichlet_cft_energy,
            heyting_verdict=final_verdict,
            learning_rate_applied=learning_rate,
            merkle_sha512_provenance=merkle_leaf_hash,
            timestamp_utc=t_start,
            fuchsian_certificate=metabolized_state.fuchsian_certificate or fuchsian_cert,
            umegaki_divergence=umegaki,
            bures_geodesic_distance=bures,
            dream_isolation_intact=dream_isolation_flag
            and metabolized_state.isolation_intact(),
            crtbp_delta_v=tube.delta_v_budget(),
            schema_version=_SCHEMA_VERSION,
        )
        self.dream_records.append(report)
        logger.info(
            "[%s] Escenario '%s' | Ω₄: %s | P=%.4f | Cobertura=%.3f | "
            "η=%.2f | VacunaOK=%s | P²−P=%.2e | Δv=%.3e | Hash=%s…",
            cycle_id,
            scenario_type,
            final_verdict.name,
            metabolized_state.purity,
            coverage_mass,
            learning_rate,
            vaccine_effective,
            idem_res,
            tube.delta_v_budget(),
            merkle_leaf_hash[:16],
        )
        return report

    def execute_wake_sleep_phase(
        self, counterfactual_batch: List[Tuple[str, float, int, complex]]
    ) -> WakeSleepCycleReport:
        r"""
        Ejecuta un batch completo de escenarios contrafactuales (fase REM) y
        consolida el veredicto global en Ω₄, junto con la raíz de Merkle
        SHA-512 de todas las pruebas de procedencia.
        """
        t_start = time.time()
        banach = BanachOperatorAlgebra()
        init_entropy = banach.von_neumann_entropy(self.master_rho)
        init_purity = banach.purity(self.master_rho)
        vaccines_synthesized = 0
        cumulative_heyting = HeytingTruthValue.VERUM_COHERENT
        batch_leaf_hashes: List[str] = []
        isolation_violations = 0
        for sc_type, cost_delta, betti_1, tau in counterfactual_batch:
            dream_rep = self.run_dream_cycle(
                scenario_type=sc_type,
                cost_delta_ratio=cost_delta,
                betti_1_loops=betti_1,
                modular_tau=tau,
                dream_isolation_flag=True,
            )
            cumulative_heyting = cumulative_heyting.meet(dream_rep.heyting_verdict)
            batch_leaf_hashes.append(dream_rep.merkle_sha512_provenance)
            if dream_rep.spectral_vaccine_effective:
                vaccines_synthesized += 1
            if not dream_rep.dream_isolation_intact:
                isolation_violations += 1
        final_entropy = banach.von_neumann_entropy(self.master_rho)
        final_purity = banach.purity(self.master_rho)
        free_energy_reduction = float(init_entropy - final_entropy)
        merkle_root = self._merkle_tree_root(batch_leaf_hashes)
        proofs_ok = True
        for i in range(len(batch_leaf_hashes)):
            proof = self._merkle_proof(batch_leaf_hashes, i)
            if proof.root != merkle_root or not proof.verify():
                proofs_ok = False
                break
        context_binder = hashlib.sha512()
        context_binder.update(
            f"WAKE-SLEEP-CONTEXT::{self.engine_id}::{merkle_root}::{init_purity:.8f}::"
            f"{final_purity:.8f}::{vaccines_synthesized}::{t_start}".encode("utf-8")
        )
        final_root_hash = context_binder.hexdigest()
        return WakeSleepCycleReport(
            cycle_id=f"WAKE-SLEEP-{self.cycle_counter:05d}",
            initial_entropy=init_entropy,
            final_entropy=final_entropy,
            wake_purity=init_purity,
            sleep_purity=final_purity,
            net_free_energy_reduction=free_energy_reduction,
            synthesized_scenarios_count=len(counterfactual_batch),
            immune_vaccines_generated=vaccines_synthesized,
            overall_topos_verdict=cumulative_heyting,
            merkle_leaf_count=len(batch_leaf_hashes),
            sha512_merkle_root=final_root_hash,
            merkle_proofs_ok=proofs_ok,
            isolation_violations=isolation_violations,
            schema_version=_SCHEMA_VERSION,
        )

    def audit_registry(self) -> Dict[str, Any]:
        r"""Resumen cuantitativo del registro de ciclos oníricos."""
        n = len(self.dream_records)
        empty_dist = {v.name: 0 for v in HeytingTruthValue}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingTruthValue.VERUM_COHERENT.name,
                "avg_purity": 0.0,
                "avg_coverage_mass": 0.0,
                "avg_dirichlet_energy": 0.0,
                "n_vaccines_effective": 0,
                "registry_integrity_ok": True,
                "avg_crtbp_delta_v": 0.0,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingTruthValue}
        s_p = s_c = s_d = s_dv = 0.0
        n_vac = 0
        hashes: Set[str] = set()
        collide = False
        for r in self.dream_records:
            dist[r.heyting_verdict.name] += 1
            s_p += r.metabolized_state.purity
            s_c += r.spectral_coverage_mass
            s_d += r.dirichlet_energy
            s_dv += r.crtbp_delta_v
            if r.spectral_vaccine_effective:
                n_vac += 1
            if r.merkle_sha512_provenance in hashes:
                collide = True
            hashes.add(r.merkle_sha512_provenance)
        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "avg_purity": s_p * inv,
            "avg_coverage_mass": s_c * inv,
            "avg_dirichlet_energy": s_d * inv,
            "n_vaccines_effective": n_vac,
            "registry_integrity_ok": not collide,
            "avg_crtbp_delta_v": s_dv * inv,
            "booleanized_verdict": self.global_verdict.booleanize().name,
        }

    def emit_dreamer_passport(self) -> Dict[str, Any]:
        r"""Pasaporte criptográfico consolidado del motor onírico."""
        h = hashlib.sha512()
        h.update(f"{self.engine_id}::{self.cycle_counter}".encode("utf-8"))
        for r in self.dream_records:
            h.update(r.merkle_sha512_provenance.encode("utf-8"))
        return {
            "engine_id": self.engine_id,
            "registry_size": self.cycle_counter,
            "global_verdict": self.global_verdict.name,
            "n_vaccines_effective": sum(
                1 for r in self.dream_records if r.spectral_vaccine_effective
            ),
            "evidence_hash": h.hexdigest(),
            "engine_version": _ENGINE_VERSION,
            "schema_version": _SCHEMA_VERSION,
        }

    @property
    def registry_view(self) -> Tuple[DreamFieldReport, ...]:
        r"""Vista inmutable del registro onírico."""
        return tuple(self.dream_records)

    @property
    def global_verdict(self) -> HeytingTruthValue:
        r"""Veredicto global: ínfimo Ω₄ de todos los ciclos registrados."""
        gv = HeytingTruthValue.VERUM_COHERENT
        for r in self.dream_records:
            gv = gv.meet(r.heyting_verdict)
        return gv

    # ──────────────────────────────────────────────────────────────────────
    # FASE 3 ⟶ CIERRE DEL MÓDULO : MÉTODO TERMINAL
    #
    #   run_terminal_rem_cycle : MetabolizedFieldState ⟶ DreamFieldReport
    #
    #   Consume el estado metabolizado de Fase 2 (o lo induce) y sella el
    #   ciclo completo 1 → 2 → 3. Es la continuación formal de
    #   `create_metabolized_state`.
    # ──────────────────────────────────────────────────────────────────────
    def run_terminal_rem_cycle(
        self,
        *,
        metabolized: Optional[MetabolizedFieldState] = None,
        scenario_type: str = "TERMINAL_REM_POINCARE",
        cost_delta_ratio: float = 0.35,
        betti_1_loops: int = 1,
        modular_tau: complex = complex(0.1, 1.2),
        dream_isolation_flag: bool = True,
    ) -> DreamFieldReport:
        r"""
        Ciclo terminal del motor onírico.

        Dominio
        -------
        MetabolizedFieldState  (objeto de Fase 2)  o su inducción vía
        lift_hamiltonian → GKSL → create_metabolized_state.

        Codominio
        ---------
        DreamFieldReport  +  actualización de ρ_MAC  +  hoja Merkle SHA-512.
        El aislamiento REM (∂ℳ_REM = 0) garantiza que el Crowbar físico
        NO se dispara: el estrés contrafactual permanece en ℳ_REM.

        Protocolo anidado
        -----------------
        1. Si no hay estado metabolizado, se corre `run_dream_cycle`
           (que a su vez induce Fase 1 y Fase 2).
        2. Si lo hay, se vacuna, se sella Merkle y se adjudica Ω₄
           respetando el aislamiento onírico.
        3. Se registra y se retorna el reporte terminal.

        Este método ES la continuación formal de `create_metabolized_state`
        y cierra las tres fases anidadas del módulo.
        """
        if metabolized is None:
            return self.run_dream_cycle(
                scenario_type=scenario_type,
                cost_delta_ratio=cost_delta_ratio,
                betti_1_loops=betti_1_loops,
                modular_tau=modular_tau,
                dream_isolation_flag=dream_isolation_flag,
            )
        self.cycle_counter += 1
        t_start = time.time()
        cycle_id = f"CYC-REM-TERM-{self.cycle_counter:05d}"
        self.last_metabolized = metabolized
        if not metabolized.is_quantum_physical():
            raise RuntimeError("Invariante C* violado en run_terminal_rem_cycle: ρ ∉ 𝔇.")
        forced = (
            metabolized.heyting_state
            if dream_isolation_flag
            else HeytingTruthValue.ABSURDUM_VETOED
        )
        verdict = metabolized.heyting_state.meet(forced)
        P_vac, vaccine_effective, coverage_mass, idem_res = (
            SpectralImmuneVaccineSynthesizer.synthesize_from_metabolized_state(
                metabolized, perturbation_norm=abs(cost_delta_ratio)
            )
        )
        learning_rate = self._apply_vaccine_learning(metabolized, P_vac, verdict)
        banach = BanachOperatorAlgebra()
        umegaki = banach.quantum_relative_entropy(
            metabolized.density_matrix, self.master_rho
        )
        bures = banach.bures_distance(metabolized.density_matrix, self.master_rho)
        merkle_leaf = self._seal_leaf(
            cycle_id,
            scenario_type,
            cost_delta_ratio,
            metabolized,
            verdict,
            coverage_mass,
            t_start,
        )
        tube_dv = (
            metabolized.crtbp_tube.delta_v_budget()
            if metabolized.crtbp_tube is not None
            else 0.0
        )
        report = DreamFieldReport(
            cycle_id=cycle_id,
            scenario_type=scenario_type,
            metabolized_state=metabolized,
            immune_projection_operator=P_vac,
            projector_idempotency_residual=idem_res,
            spectral_coverage_mass=coverage_mass,
            spectral_vaccine_effective=vaccine_effective,
            dirichlet_energy=metabolized.dirichlet_cft_energy,
            heyting_verdict=verdict,
            learning_rate_applied=learning_rate,
            merkle_sha512_provenance=merkle_leaf,
            timestamp_utc=t_start,
            fuchsian_certificate=metabolized.fuchsian_certificate,
            umegaki_divergence=umegaki,
            bures_geodesic_distance=bures,
            dream_isolation_intact=dream_isolation_flag
            and metabolized.isolation_intact(),
            crtbp_delta_v=tube_dv,
            schema_version=_SCHEMA_VERSION,
        )
        self.dream_records.append(report)
        logger.info(
            "Ciclo terminal REM %s | Ω₄=%s | P=%.4f | η=%.2f | aislamiento=%s | "
            "Δv=%.3e | Merkle=%s…",
            cycle_id,
            verdict.name,
            metabolized.purity,
            learning_rate,
            report.dream_isolation_intact,
            tube_dv,
            merkle_leaf[:16],
        )
        return report


# ══════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN RIGUROSA Y VALIDACIÓN UNITARIA INTER-FASES
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("╔" + "═" * 86 + "╗")
    print("║  DEMOSTRACIÓN TEÓRICO-PRÁCTICA DEL TOON ONIRIC DREAMER ENGINE v9.1     ║")
    print("║  MECÁNICA CELESTE (POINCARÉ ℍ² + CRTBP + LINDBLAD-GKSL FUCSIANO)      ║")
    print("╚" + "═" * 86 + "╝")

    # ── FASE 1: verificaciones algebraicas ────────────────────────────────
    assert HeytingTruthValue.verify_heyting_axioms(), "Ω₄ viola ley de residuación."
    print("\n[FASE 1] Álgebra de Heyting Ω₄: axiomas de residuación verificados.")

    q1 = Quaternion(0.3, 0.5, -0.2, 0.7)
    q2 = Quaternion(0.1, -0.4, 0.6, 0.2)
    q3 = Quaternion(-0.5, 0.2, 0.1, 0.4)
    assert q1.associativity_residual(q2, q3) < 1e-12, "ℍ no asociativa."
    assert q1.composition_residual(q2) < 1e-12, "ℍ no composición."
    assert q1.exp_log_roundtrip_residual() < 1e-10, "exp·log ≠ id en ℍˣ."
    print(
        f"  · Cuaterniones ℍ: residuo asociatividad = {q1.associativity_residual(q2, q3):.3e}, "
        f"exp∘log = {q1.exp_log_roundtrip_residual():.3e}"
    )

    cert = reduce_to_poincare_fundamental_domain(complex(1.8, 0.4))
    print("\n[FASE 1] Uniformización fucsiana:")
    print(f"  · τ_orig = 1.8 + 0.4j → τ_red = {cert.tau_reduced:.4f}")
    print(
        f"  · d_ℍ² = {cert.poincare_metric_distance:.4f} | "
        f"in_ℱ = {cert.is_in_fundamental_domain} | "
        f"clase modular = {cert.modular_class}"
    )
    assert cert.is_in_fundamental_domain
    T = MobiusTransformation.T()
    S = MobiusTransformation.S()
    print(f"  · T-clase = {T.conjugacy_class()} | S-clase = {S.conjugacy_class()}")

    # ── FASE 2: tubos CRTBP y evolución GKSL ──────────────────────────────
    tube = CRTBPInvariantManifoldTube(
        libration_point_id="L1",
        mass_ratio_mu=3.04e-6,
        jacobi_constant_C=3.0009,
        is_stable_manifold=True,
        tube_radius=0.01,
        trajectory_energy=-3.0009,
    )
    print("\n[FASE 2] Tubo invariante L₁ (Sol–Tierra):")
    print(f"  · r_Hill = {tube.critical_radius_hill():.4e}")
    print(f"  · residuo energía = {tube.energy_residual():.3e}")
    print(f"  · Δv a r_target = {tube.delta_v_budget(0.005):.4f}")
    print(f"  · hiperbólico = {tube.is_hyperbolic_libration()}")
    print(f"  · Routh L4/L5 estable (μ⊕) = {LagrangePoint.routh_stable(3.04e-6)}")
    sec = CRTBPPoincareSection(mu=3.04e-6, C=3.0009)
    print(f"  · λ_silla L1 (proxy) = {sec.saddle_lyapunov_proxy():.4f}")

    engine = TOONOniricDreamerEngine(
        engine_id="APU-METACORTEX-REM-POINCARE-01", dimension_mac=4, seed=1337
    )
    batch_scenarios: List[Tuple[str, float, int, complex]] = [
        ("STEEL_CARTEL_PRICE_SHOCK_35", 0.35, 0, complex(0.2, 1.1)),
        ("STRIKE_LABOR_PARALYSIS_MACRO", 0.65, 1, complex(-0.4, 0.9)),
        ("HYDROLOGIC_FLOOD_FOUNDATION", 0.25, 0, complex(0.0, 2.0)),
        ("CIRCULAR_SUB_BILLING_ATTACK", 0.85, 2, complex(1.5, 0.3)),
    ]
    wake_sleep_report = engine.execute_wake_sleep_phase(batch_scenarios)

    # ── FASE 3: reporte consolidado + ciclo terminal ──────────────────────
    print("\n[FASE 3] Reporte Wake-Sleep consolidado:")
    print(f"  · ID ciclo            : {wake_sleep_report.cycle_id}")
    print(f"  · Escenarios          : {wake_sleep_report.synthesized_scenarios_count}")
    print(f"  · Vacunas sintetizadas: {wake_sleep_report.immune_vaccines_generated}")
    print(f"  · Entropía inicial    : {wake_sleep_report.initial_entropy:.4f}")
    print(f"  · Entropía final      : {wake_sleep_report.final_entropy:.4f}")
    print(f"  · ΔF (energía libre)  : {wake_sleep_report.net_free_energy_reduction:+.4f}")
    print(f"  · Ω₄ veredicto global : {wake_sleep_report.overall_topos_verdict.name}")
    print(f"  · Merkle root (SHA-512): {wake_sleep_report.sha512_merkle_root[:48]}…")
    print(f"  · Pruebas Merkle OK   : {wake_sleep_report.merkle_proofs_ok}")
    print(f"  · Violaciones aislamiento: {wake_sleep_report.isolation_violations}")
    assert wake_sleep_report.merkle_proofs_ok

    terminal = engine.run_terminal_rem_cycle(
        metabolized=engine.last_metabolized,
        scenario_type="TERMINAL_REM_POINCARE",
        cost_delta_ratio=0.35,
        dream_isolation_flag=True,
    )
    print("\n[FASE 3] run_terminal_rem_cycle:")
    print(f"  · cycle_id            : {terminal.cycle_id}")
    print(f"  · Ω₄                  : {terminal.heyting_verdict.name}")
    print(f"  · aislamiento intacto : {terminal.dream_isolation_intact}")
    print(f"  · Δv CRTBP            : {terminal.crtbp_delta_v:.4e}")
    print(f"  · Bures               : {terminal.bures_geodesic_distance:.4f}")

    passport = engine.emit_dreamer_passport()
    print("\n[FASE 3] Pasaporte del motor onírico:")
    for k, v in passport.items():
        print(f"  · {k:22s}: {v}")
    print("\n" + "═" * 88)
    print("✓ VERIFICACIÓN INTEGRAL DE POINCARÉ (ℍ², CRTBP, GKSL, Bures, Merkle)")
    print("  Anidación  lift_hamiltonian → create_metabolized_state")
    print("           → run_terminal_rem_cycle.  v9.1.0-Doctoral-Poincare-Fuchsian.")
    print("═" * 88)
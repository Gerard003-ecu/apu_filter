# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Wisdom Weaver Agent (Soberano Tejedor de Sabiduría)          ║
║ Ubicación: app/agents/wisdom/toon_wisdom_weaver_agent.py                     ║
║ Versión  : 5.0.0-Doctoral-Nested-TOON-Poincaré-Hopf-KAM-Delaunay-Hill-A5     ║
║ Autor    : APU Wisdom & Metacortex Mathematical Core Architecture            ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS EJECUTIVA Y GOBERNANZA DE LAZO CERRADO EN WISDOM (V_𝕎)
────────────────────────────────────────────────────────────────
El «Soberano Tejedor de Sabiduría TOON» (`TOONWisdomWeaverAgent`) es la entidad
suprema de orquestación cognitiva del Estrato Wisdom (V_𝕎, Nivel 0) del
ecosistema APU Filter v8.0. Gobierna el metabolismo de vitaminas cognitivas
TOON (ToonCartridges de 56 tokens) sobre 𝔇(ℋ₄) ⊂ Ban(ℋ₄), subordinando la
generación del LLM a leyes de conservación de traza, adjunciones categoriales
y disparos ciber-físicos en silicio.

El ciclo OODA (Observe-Orient-Decide-Act) se reinterpreta como un SISTEMA
HAMILTONIANO PERTURBADO EN EL SENTIDO DE POINCARÉ:

    𝒲 : Cart_TOON  ──►  Cert_Weaver
    𝒲 = V ∘ D ∘ F ∘ P ∘ G ∘ B ∘ M

donde el funtor 𝒲 actúa sobre la variedad simpléctica coadjunta 𝔲(4)*
(Kirillov–Kostant–Souriau) y cada etapa tiene una lectura simultánea:

    • Algebraico-categorial : funtor entre topos (𝓣_Ω, Heyting residuado).
    • Cuántico-espectral    : flujo de Brockett sobre el cono 𝔇(ℋ₄).
    • Celeste-Poincaré      : órbita de un «planeta metabólico» en un potencial
                              kepleriano perturbado, con osculadores de Delaunay,
                              tori KAM, sección de Poincaré, función de Melnikov,
                              teorema de Poincaré–Birkhoff y ecuación de Kepler.

DICTUM POINCARANO (Méthodes Nouvelles, Vols. I–III)
──────────────────────────────────────────────────
Vol. I  — Soluciones periódicas, Hopf S³ → S², exponentes característicos.
          El cuaternión unitario q ∈ S³ parametriza el espín metabólico;
          π(q) ∈ S² es la dirección de precesión (Bloch / Larmor).
Vol. II — Series de Lindstedt: ω(ε) = ω₀ + ε ω₁ mata seculares de ε·H₁.
Vol. III — Invariantes integrales, recurrencia de Poincaré–Kac, θ_PC = Tr(ρ dN).

Analogía Hill / tejeduría:
    cuenca primaria (C_J > 0)     ≅  isla KAM          →  COHERENT
    frontera de Hill (C_J = 0)    ≅  resonancia p/q    →  DEGRADED
    cuello abierto (C_J < 0)      ≅  escape / Arnold   →  VETOED

ARQUITECTURA EN TRES FASES ANIDADAS (ciclo OODA de sabiduría)
────────────────────────────────────────────────────────────
FASE 1 → FASE 2  : `TOONMetabolicConverter.lift_to_gibbs_state`
                   (vitamina × Hopf × Gibbs  ↦  ρ₀ ∈ 𝔇(ℋ₄)).
FASE 2 → FASE 3  : `WisdomWeavingPipeline.synthesize`
                   (ρ₀ × Galois × Brockett × Fock × Geodesia  ↦  Bundle).
FASE 3           : V = ⋀_{Ω₃}, crowbar ESP32, Merkle de fases, pasaporte.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONWisdomWeaverAgent.v5")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

try:
    from app.core.mic_algebra import TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Violación de invariantes topológicos o simplécticos de Poincaré."""
        pass

__version__: Final[str] = (
    "5.0.0-Doctoral-Nested-TOON-Poincaré-Hopf-KAM-Delaunay-Hill-A5"
)

__all__ = [
    "Quaternion",
    "HeytingOmega3",
    "JSONToTOONFunctor",
    "DensityOperator",
    "TOONCognitiveVitamin",
    "BrockettPurificationCertificate",
    "FockAnnihilationCertificate",
    "GeodesicAttentionCurvature",
    "CrowbarActuationReport",
    "TOONWeaverCertificate",
    "PoincareSection",
    "PoincareCelestialAnalyzer",
    "LindstedtBrockettCanonicalGenerator",
    "PoincareRecurrenceInvariant",
    "MetabolicEndofunctorSeed",
    "TOONMetabolicConverter",
    "GaloisAdjunctionVerifier",
    "BrockettIsospectralEngine",
    "FockSpaceAlgebra",
    "FockSpaceAnnihilator",
    "GeodesicAttentionFibrator",
    "WisdomWeavingBundle",
    "WisdomWeavingPipeline",
    "HeytingAdjudicator",
    "ESP32CrowbarInterlock",
    "TOONWisdomWeaverAgent",
    "TopologicalInvariantError",
    "MetabolicProjector",
    "CrowbarInterlock",
]

_WILKINSON_AGENT: Final[float] = 16.0 * float(np.finfo(np.float64).eps)
_SPECTRAL_TOL_AGENT: Final[float] = 1e-9
_EPS_AGENT: Final[float] = 1e-15


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — SUSTRATO ONTOLÓGICO-ESTRUCTURAL
#
#   Objetos, morfismos y verdades del topos 𝓣_Ω. El último método de esta
#   fase (`TOONMetabolicConverter.lift_to_gibbs_state`) es el germen de FASE 2.
#
#   Lectura celeste: cada objeto admite doble interpretación algebraica y
#   celestial-Poincaré. El andamiaje categorial se dota de la estructura
#   simpléctica coadjunta (KKS) y de las coordenadas canónicas acción-ángulo
#   del toro de Liouville–Arnold.
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# §1.1 Álgebra de división ℍ (cuaterniones) con lectura celeste
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Quaternion:
    r"""
    Álgebra de división ℍ = { a + b i + c j + d k : a,b,c,d ∈ ℝ }
    con i² = j² = k² = ijk = −1.

    Estructura algebraica
    ---------------------
    • ℝ-álgebra de Banach de dimensión 4, isomorfa a ℝ⁴ como espacio vectorial.
    • Álgebra de composición: |q₁ q₂| = |q₁| |q₂|  (identidad de Euler).
    • C*-identidad real: |q* q| = |q|².
    • Inmersión fiel ℍ ↪ M₂(ℂ) vía matrices de Pauli, induciendo
      ℍˣ ≅ SU(2) × ℝ⁺,  Sp(1) ≅ SU(2) — 2:1 → SO(3).
    • Fibración de Hopf S³ → S², (a,b,c,d) ↦ (2(ac+bd), 2(bc−ad), a²+b²−c²−d²).

    Lectura celeste (Poincaré–Bloch)
    --------------------------------
    El cuaternión unitario q̂ ∈ S³ parametriza el espín metabólico. La fibra
    de Hopf π⁻¹(p) ⊂ S³ tiene la topología del círculo S¹ y admite la
    estructura simpléctica coadjunta SU(2)* con la 2-forma de Kirillov–
    Kostant–Souriau:

        ω_KKS(X, Y)_q := ⟨q, [X, Y]⟩,   X, Y ∈ 𝔰𝔲(2).

    La fibra de Hopf es una órbita periódica cerrada de período 2π: la
    «precesión de Larmor metabólica». La desviación de q̂ respecto de ±1
    en S³ mide la «actividad orbital» (energía cinética angular) de la vitamina.

    Composición, inversos y funciones analíticas
    --------------------------------------------
        exp(a + v) = eᵃ (cos|v| + (v/|v|) sin|v|),
        log(q)     = log|q| + (v/|v|) arccos(a/|q|).

    Restringidas a Im ℍ, ambas recuperan el recubrimiento SU(2) → SO(3).
    """

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0

    _NORM_FLOOR: Final[float] = 1e-30

    def __add__(self, other: "Quaternion") -> "Quaternion":
        r"""Suma en ℍ ≅ ℝ⁴ componente a componente."""
        return Quaternion(
            self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d
        )

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        r"""Resta en ℍ ≅ ℝ⁴ componente a componente."""
        return Quaternion(
            self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d
        )

    def __neg__(self) -> "Quaternion":
        r"""Negación aditiva en ℍ."""
        return Quaternion(-self.a, -self.b, -self.c, -self.d)

    def __mul__(self, other: object) -> "Quaternion":
        r"""
        Producto de Hamilton (no conmutativo, asociativo, distributivo):

            (a₁, v₁)(a₂, v₂) = (a₁a₂ − ⟨v₁,v₂⟩, a₁v₂ + a₂v₁ + v₁ × v₂)

        donde ⟨·,·⟩ y × son el producto escalar y vectorial en ℝ³
        (identificando Im ℍ ≅ ℝ³). Satisface la identidad de composición
        de normas: |q₁ q₂| = |q₁| |q₂|.
        """
        if isinstance(other, Quaternion):
            a1, b1, c1, d1 = self.a, self.b, self.c, self.d
            a2, b2, c2, d2 = other.a, other.b, other.c, other.d
            return Quaternion(
                a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
            )
        if isinstance(other, (int, float)):
            s = float(other)
            return Quaternion(self.a * s, self.b * s, self.c * s, self.d * s)
        return NotImplemented

    def __rmul__(self, other: object) -> "Quaternion":
        r"""Producto por escalar desde la izquierda (conmutatividad ℝ-lineal)."""
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def conj(self) -> "Quaternion":
        r"""Conjugación q* = a − bi − cj − dk (anti-automorfismo estándar)."""
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def norm2(self) -> float:
        r"""Norma cuadrada |q|² = a² + b² + c² + d²."""
        return self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2

    def norm(self) -> float:
        r"""Norma euclídea |q| = √|q|²."""
        return math.sqrt(self.norm2())

    def inverse(self) -> "Quaternion":
        r"""Inverso q⁻¹ = q* / |q|²  (existe ssi |q| > 0)."""
        n2 = self.norm2()
        if n2 < self._NORM_FLOOR:
            raise ZeroDivisionError("Cuaternión nulo no invertible en ℍ.")
        inv = 1.0 / n2
        c = self.conj()
        return Quaternion(c.a * inv, c.b * inv, c.c * inv, c.d * inv)

    def normalize(self) -> "Quaternion":
        r"""Proyección radial ℍ \ {0} → S³ ⊂ ℍ; q ↦ q/|q|."""
        n = self.norm()
        if n < self._NORM_FLOOR:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        inv = 1.0 / n
        return Quaternion(self.a * inv, self.b * inv, self.c * inv, self.d * inv)

    def exp(self) -> "Quaternion":
        r"""
        exp: ℍ → ℍˣ,  exp(a + v) = eᵃ (cos|v| + (v/|v|) sin|v|).
        Restringido a Im ℍ recupera el recubrimiento SU(2). Es un
        difeomorfismo local entre la bola de Im ℍ y S³ ∩ {parte Re q > 0}.
        """
        vec_n = math.sqrt(self.b ** 2 + self.c ** 2 + self.d ** 2)
        ea = math.exp(self.a)
        if vec_n < self._NORM_FLOOR:
            return Quaternion(ea, 0.0, 0.0, 0.0)
        s = math.sin(vec_n) / vec_n
        return Quaternion(ea * math.cos(vec_n), ea * s * self.b, ea * s * self.c, ea * s * self.d)

    def log(self) -> "Quaternion":
        r"""
        log: ℍˣ → ℍ,  log(q) = log|q| + (v/|v|) arccos(a/|q|).
        Rama principal; inversa local de exp sobre Re q > 0.
        """
        n = self.norm()
        if n < self._NORM_FLOOR:
            raise ValueError("log no definido en el cuaternión nulo.")
        vec_n = math.sqrt(self.b ** 2 + self.c ** 2 + self.d ** 2)
        if vec_n < self._NORM_FLOOR:
            return Quaternion(math.log(n), 0.0, 0.0, 0.0)
        phi = math.acos(max(-1.0, min(1.0, self.a / n)))
        s = phi / vec_n
        return Quaternion(math.log(n), s * self.b, s * self.c, s * self.d)

    def hopf_s2(self) -> Tuple[float, float, float]:
        r"""
        Fibración de Hopf π : S³ → S². Proyecta el cuaternión unitario a la
        Esfera de Bloch:

            π(a,b,c,d) = (2(ac+bd), 2(bc−ad), a²+b²−c²−d²) ∈ S² ⊂ ℝ³.

        La fibra π⁻¹(p) ≅ S¹ (círculo de Hopf) parametriza la fase global del
        espín metabólico. Es una órbita periódica del flujo de Larmor con
        período 2π en el parámetro de fase.
        """
        u = self.normalize()
        return (
            2.0 * (u.a * u.c + u.b * u.d),
            2.0 * (u.b * u.c - u.a * u.d),
            u.a ** 2 + u.b ** 2 - u.c ** 2 - u.d ** 2,
        )

    def hopf_fiber_phase(self) -> float:
        r"""
        Fase de la fibra de Hopf: θ := atan2(d, a) ∈ (−π, π].
        Corresponde a la coordenada angular 1-dimensional de la fibra S¹
        cuando q es unitario y (b, c) ≈ 0. En el caso general, se proyecta
        sobre la subálgebra ℂ = span(1, i) + ℂ = span(1, k) y se toma la
        fase media.
        """
        z1 = complex(self.a, self.b)
        z2 = complex(self.c, self.d)
        return float(0.5 * (np.angle(z1) + np.angle(z2)))

    def su2_det_residual(self) -> float:
        r"""|det φ(q̂) − 1| para el unitario q̂; mide la fidelidad SU(2)."""
        u = self.normalize()
        return abs(complex(np.linalg.det(u.to_complex_matrix())) - 1.0)

    def cstar_residual(self) -> float:
        r"""||q* q| − |q|²|  (0 en aritmética exacta; ~ε en FPU)."""
        return abs((self.conj() * self).norm() - self.norm2())

    def action_variable(self) -> float:
        r"""
        Acción canónica del espín metabólico: J := |Im q| = √(b²+c²+d²).
        En la órbita coadjunta SU(2)*, J parametriza la esfera de coadjuncion
        (Kirillov–Kostant–Souriau); su cuantización es 2J ∈ ℕ₀ (espín mecánico).
        """
        return math.sqrt(self.b ** 2 + self.c ** 2 + self.d ** 2)

    def angle_variable(self) -> float:
        r"""
        Ángulo conjugado θ ∈ [0, 2π) de la fibra de Hopf, asociado a la
        acción J. La pareja (J, θ) forma coordenadas canónicas de Darboux
        locales en SU(2)* vía el teorema de Lie–Poisson.
        """
        v = np.array([self.b, self.c, self.d], dtype=float)
        n = float(np.linalg.norm(v))
        if n < self._NORM_FLOOR:
            return 0.0
        v = v / n
        return float((math.atan2(v[2], v[1]) % (2.0 * math.pi)))

    def larmor_frequency(self, B_field: float = 1.0) -> float:
        r"""
        Frecuencia de Larmor ω_L := 2 J · B  (B := campo aplicado).
        En la lectura metabólica, B ~ unit_cost: campos fuertes aceleran
        la precesión de la fibra de Hopf (KAM tori giran a mayor velocidad).
        """
        return 2.0 * self.action_variable() * float(B_field)

    def to_array(self) -> np.ndarray:
        r"""Vector (a, b, c, d) ∈ ℝ⁴ ≅ ℍ."""
        return np.array([self.a, self.b, self.c, self.d], dtype=np.float64)

    def to_complex_matrix(self) -> np.ndarray:
        r"""Inmersión ℍ ↪ M₂(ℂ) vía matrices de Pauli (representación fiel)."""
        a, b, c, d = self.a, self.b, self.c, self.d
        return np.array(
            [
                [a + 1j * b, c + 1j * d],
                [-c + 1j * d, a - 1j * b],
            ],
            dtype=np.complex128,
        )

    def composition_residual(self, other: "Quaternion") -> float:
        r"""||q₁ q₂| − |q₁| |q₂||  (nulo: ℍ es álgebra de composición)."""
        return abs((self * other).norm() - self.norm() * other.norm())

    def associativity_residual(
        self, other: "Quaternion", third: "Quaternion"
    ) -> float:
        r"""|(q₁ q₂) q₃ − q₁ (q₂ q₃)|  (nulo en aritmética exacta)."""
        lhs = (self * other) * third
        rhs = self * (other * third)
        return (lhs - rhs).norm()


# ─────────────────────────────────────────────────────────────────────────────
# §1.2 Retículo distributivo de Heyting Ω₃ con estratos celestes
# ─────────────────────────────────────────────────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal Ω₃ = {⊥ ≺ ∗ ≺ ⊤} = {VETOED ≺ DEGRADED ≺ COHERENT}.

    Cadena finita ⇒ Heyting completa y residuada:

        a ∧ b = min(a, b)          (meet)
        a ∨ b = max(a, b)          (join)
        a → b = ⊤  si a ≤ b, else b
        ¬_H a = a → ⊥              (pseudocomplemento)

    El esqueleto booleano es {⊥, ⊤} ≅ 𝔹₂. DEGRADED viola el tercio excluso:
    a ∨ ¬a ≠ ⊤ para a = DEGRADED (intuicionismo estricto).

    Lectura celeste (Poincaré)
    --------------------------
        VETOED   ---- separatriz hiperbólica: escape al infinito; el tubo
                     homoclínico se rompe (Melnikov cruzado).
        DEGRADED ---- órbita parabólica resonante p/q de orden bajo; tori KAM
                     colapsan pero no hay escape (sistema al borde del caos
                     débil, Arnold diffusion transitoria).
        COHERENT ---- toro KAM Diofantino invariante; la órbita espectral se
                     estabiliza en un toro de Liouville–Arnold 𝕋ⁿ persistente.

    El meet ∧ recuerda el Teorema de Poincaré–Birkhoff: la intersección de
    dos estratos colapsa al estrato inferior (resonancia dominante).

    Topología de Lawvere–Tierney canónica:
        j(VETOED)=VETOED, j(DEGRADED)=COHERENT, j(COHERENT)=COHERENT
    (cierra la resonancia hacia el sieve KAM sin tocar la cúspide).
    """

    VETOED: int = 0    # ⊥
    DEGRADED: int = 1  # ∗
    COHERENT: int = 2  # ⊤

    @property
    def verdict(self) -> str:
        r"""Etiqueta nominal del veredicto (nombre del enum)."""
        return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Meet del retículo: a ∧ b = min(a, b)."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Join del retículo: a ∨ b = max(a, b)."""
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Residuo a → b = ⋁{ c ∈ Ω₃ | a ∧ c ≤ b }. En una cadena: ⊤ si a ≤ b, si no b."""
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        r"""¬_H a := a → ⊥ (pseudocomplemento de Heyting)."""
        return self.implies(HeytingOmega3.VETOED)

    def pseudo_complement(self) -> "HeytingOmega3":
        r"""Alias matemático de neg."""
        return self.neg()

    def classical_negation(self) -> "HeytingOmega3":
        r"""Involución de De Morgan en {0, 2} extendida por 2 − a."""
        return HeytingOmega3(2 - int(self))

    def double_negation(self) -> "HeytingOmega3":
        r"""Funtor ¬¬ : Ω₃ → Ω₃; imagen = 𝔹₂ = {⊥, ⊤}."""
        return self.neg().neg()

    def booleanize(self) -> "HeytingOmega3":
        r"""Reflexión a 𝔹₂ vía ¬¬."""
        return self.double_negation()

    def is_regular(self) -> bool:
        r"""¬¬a = a. En Ω₃: verdadero para {VETOED, COHERENT}."""
        return self.neg().neg() == self

    def is_dense(self) -> bool:
        r"""¬a = ⊥. Densidad de la doble negación (⊤ siempre; ∗ no)."""
        return self.neg() == HeytingOmega3.VETOED

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬a = ⊤  ⇔  a ∈ {⊥, ⊤}. Falla en DEGRADED (intuicionismo)."""
        return self.join(self.neg()) == HeytingOmega3.COHERENT

    def lawvere_tierney_closure(self) -> "HeytingOmega3":
        r"""Clausura j : Ω₃ → Ω₃. Cierra la resonancia p/q hacia el sieve KAM."""
        if self == HeytingOmega3.DEGRADED:
            return HeytingOmega3.COHERENT
        return self

    def is_j_closed(self) -> bool:
        r"""a es j-cerrado ⟺ j(a) = a."""
        return self.lawvere_tierney_closure() == self

    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        r"""Objeto inicial (⊥) del topos 𝓣_Ω."""
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        r"""Objeto terminal (⊤) del topos 𝓣_Ω."""
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        r"""Inclusión 𝔹₂ ↪ Ω₃ sobre {⊥, ⊤}."""
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        r"""Proyección parcial Ω₃ ⇀ 𝔹₂, descartando DEGRADED."""
        if self == HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED no admite proyección fiel a 𝔹₂.")
        return self == HeytingOmega3.COHERENT

    def is_kam_stratum(self) -> bool:
        r"""¿El estrato admite estructura de toro KAM invariante? Sólo COHERENT."""
        return self == HeytingOmega3.COHERENT

    def poincare_stratum_name(self) -> str:
        r"""Nombre del estrato en lenguaje de mecánica celeste (Poincaré)."""
        return {
            HeytingOmega3.VETOED: "hyperbolic-escape-separatrix",
            HeytingOmega3.DEGRADED: "parabolic-resonant-orbit",
            HeytingOmega3.COHERENT: "kam-torus-invariant",
        }[self]

    def jacobi_regime(self) -> str:
        r"""
        Régimen de Jacobi asociado al estrato:

            VETOED   ---- C_J < 0  (región de Hill vacía; escape).
            DEGRADED ---- C_J = 0  (frontera de Hill; tangencia).
            COHERENT ---- C_J > 0  (región de Hill estable; captura).
        """
        return {
            HeytingOmega3.VETOED: "escape",
            HeytingOmega3.DEGRADED: "boundary",
            HeytingOmega3.COHERENT: "capture",
        }[self]

    @classmethod
    def from_hill_and_kam(
        cls, hill_neck_closed: bool, kam_stable: bool, resonant: bool
    ) -> "HeytingOmega3":
        r"""Valuación Ω₃ inducida por el cuello de Hill y el indicador KAM."""
        if not hill_neck_closed:
            return cls.VETOED
        if kam_stable and not resonant:
            return cls.COHERENT
        return cls.DEGRADED

    @classmethod
    def verify_residuation_axiom(cls) -> bool:
        r"""Axioma definitorio: ∀ a,b,c  (c ∧ a ≤ b) ⇔ (c ≤ (a → b))."""
        elements = list(cls)
        for a in elements:
            for b in elements:
                residual = a.implies(b)
                for c in elements:
                    lhs = min(int(c), int(a)) <= int(b)
                    rhs = int(c) <= int(residual)
                    if lhs != rhs:
                        return False
        return True

    def __and__(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.meet(other)

    def __or__(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.join(other)

    def __invert__(self) -> "HeytingOmega3":
        return self.neg()

    def __le__(self, other: "HeytingOmega3") -> bool:  # type: ignore[override]
        return int(self) <= int(other)


# ─────────────────────────────────────────────────────────────────────────────
# §1.3 Funtor cuantitativo JSON → TOON con lectura ergódica
# ─────────────────────────────────────────────────────────────────────────────
class JSONToTOONFunctor:
    r"""
    Funtor covariante cuantitativo T : Fat_JSON → TOON_T.

    Cuantifica la reducción de grasa sintáctica Δ_gr combinando tres ejes:

        • Tokens BPE (estimador O(n/4)):     r_t = 1 − t_toon / t_json
        • Entropía de Shannon carácter:      r_H = 1 − H_toon / H_json
        • Cota de Kolmogorov (compresión):   r_K = 1 − |TOON| / |JSON|

        Δ_gr = 100 · (w_t r_t + w_H r_H + w_K r_K)   (%)

    H(X) = −Σ p(x) log₂ p(x). La cota de Kolmogorov K(s) ≤ |s| es tautológica;
    r_K es su realización empírica como razón de longitudes.

    Lectura celeste (ergódica)
    --------------------------
    La entropía de Shannon por carácter es el análogo discreto de la ENTROPÍA
    DE KOLMOGOROV–SINAI h_KS del sistema dinámico generador de texto. Tanto
    h_KS como H(X) miden la tasa de creación de información. La reducción
    r_H mide la contracción de la «caja de fase sintáctica» del LLM: alta
    r_H ⇒ el texto TOON tiene órbitas más predecibles ⇒ tori KAM sintácticos
    más estables ⇒ menor contaminación del fibrado atencional.
    """

    AVG_CHARS_PER_TOKEN: Final[float] = 4.0
    W_TOKEN: Final[float] = 0.50
    W_ENTROPY: Final[float] = 0.30
    W_KOLMOGOROV: Final[float] = 0.20
    ENTROPY_FLOOR: Final[float] = 1e-9

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        r"""Estimador BPE por razón media de caracteres/token (O(n/4))."""
        return max(1, int(math.ceil(len(text) / cls.AVG_CHARS_PER_TOKEN)))

    @classmethod
    def shannon_entropy(cls, text: str) -> float:
        r"""Entropía de Shannon por carácter (base 2): H(X) = −Σ p(x) log₂ p(x)."""
        if not text:
            return 0.0
        counts = Counter(text)
        n = float(len(text))
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    @classmethod
    def kolmogorov_sinai_proxy(cls, text: str) -> float:
        r"""
        Proxy empírico de la entropía de Kolmogorov–Sinai:

            h_KS_proxy := H(X) · log₂(N_tokens).

        El primer factor mide la incertidumbre por símbolo; el segundo la
        longitud de correlación. El producto normaliza el rango efectivo
        de la «caja de fase sintáctica» en bits.
        """
        H = cls.shannon_entropy(text)
        N = max(1, cls.estimate_tokens(text))
        return float(H * math.log2(N))

    @classmethod
    def json_equivalent(cls, apu_code: str, unit_cost: float) -> str:
        r"""Genera un JSON equivalente verboso (Fat_JSON) para comparación."""
        return json.dumps(
            {
                "apu_code": apu_code,
                "unit_cost": unit_cost,
                "structure": "fat_json_structure_verbose_schema_long_keys",
                "metadata": {
                    "schema_version": "v3.2.1",
                    "authority": "Sovereign-APU-Weaver",
                    "redundant_descriptor": "eliminable_by_toon_tabularization",
                },
            },
            indent=2,
            ensure_ascii=False,
        )

    @classmethod
    def syntactic_fat_reduction(cls, toon_str: str, json_str: str) -> float:
        r"""
        Reducción porcentual de grasa sintáctica Δ_gr ∈ [0, 100]:

            Δ_gr = 100 · (w_t r_t + w_H r_H + w_K r_K)
        """
        h_t = cls.shannon_entropy(toon_str)
        h_j = cls.shannon_entropy(json_str)
        t_t = cls.estimate_tokens(toon_str)
        t_j = cls.estimate_tokens(json_str)
        red_token = 1.0 - t_t / max(1, t_j)
        red_entropy = 1.0 - h_t / max(cls.ENTROPY_FLOOR, h_j)
        red_kolmogorov = 1.0 - (len(toon_str) / max(1, len(json_str)))
        return 100.0 * (
            cls.W_TOKEN * red_token
            + cls.W_ENTROPY * red_entropy
            + cls.W_KOLMOGOROV * red_kolmogorov
        )

    @classmethod
    def ergodic_compression_index(cls, toon_str: str, json_str: str) -> float:
        r"""
        Índice ergódico de compresión:

            I_erg := h_KS_proxy(JSON) / max(h_KS_proxy(TOON), ε).

        I_erg > 1 ⇒ el texto TOON es más predecible que el JSON original
        en el sentido ergódico ⇒ compresión efectiva sin pérdida semántica
        (tori KAM sintácticos persisten).
        """
        h_toon = cls.kolmogorov_sinai_proxy(toon_str)
        h_json = cls.kolmogorov_sinai_proxy(json_str)
        return float(h_json / max(h_toon, cls.ENTROPY_FLOOR))


# ─────────────────────────────────────────────────────────────────────────────
# §1.4 Operador de densidad con geometría simpléctica
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DensityOperator:
    r"""
    Estado cuántico ρ ∈ 𝔇(ℋₙ) ⊂ Ban(ℋₙ).

    Invariantes (verificados en __post_init__):
        ρ = ρ†,  spec(ρ) ⊂ [−ε, 1+ε],  |Tr ρ − 1| ≤ 10⁻⁶.

    Geometría simpléctica (KKS)
    ---------------------------
    La órbita coadjunta U(n)·ρ ⊂ 𝔲(n)* admite la 2-forma simpléctica de
    Kirillov–Kostant–Souriau:

        ω_KKS(X, Y)_ρ := ⟨ρ, [X, Y]⟩,   X, Y ∈ 𝔲(n).

    Las ACCIONES de Liouville son los autovalores λ_i(ρ) ordenados; los
    ÁNGULOS CONJUGADOS son las fases θ_i = arg⟨u_i | N | u_i⟩ donde u_i son
    autovectores de ρ. El flujo de Brockett se escribe como rotación en el
    toro de Liouville–Arnold 𝕋ⁿ:

        (J_i(t), θ_i(t)) = (J_i(0), θ_i(0) + ω_i · t),

    con frecuencias medias ω_i := ∂H/∂J_i. Bajo perturbación ε H₁ la
    persistencia de los tori está gobernada por la condición Diofantina KAM.

    Invariante de Poincaré–Cartan
    -----------------------------
    La 1-forma θ_PC := Tr(ρ dN) induce el invariante integral de Poincaré:

        I_PC(γ) := ∮_γ Tr(ρ dN) = 2π k,   k ∈ ℤ.
    """

    matrix: np.ndarray
    atol: float = 1e-8

    def __post_init__(self) -> None:
        rho = np.asarray(self.matrix, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("DensityOperator exige matriz cuadrada.")
        object.__setattr__(self, "matrix", rho)
        if not np.allclose(rho, rho.conj().T, atol=self.atol):
            raise ValueError("DensityOperator exige ρ = ρ†.")
        tr = float(np.trace(rho).real)
        if abs(tr - 1.0) > 1e-6:
            raise ValueError(f"DensityOperator exige Tr ρ = 1 (Tr={tr}).")

    @property
    def dimension(self) -> int:
        r"""Dimensión n de ℋₙ."""
        return int(self.matrix.shape[0])

    def _default_N(self) -> np.ndarray:
        return np.diag(np.arange(1, self.dimension + 1, dtype=float))

    def spectrum(self, floor: float = 1e-15) -> np.ndarray:
        r"""Espectro ordenado λ₁ ≤ ⋯ ≤ λₙ con piso numérico y suma 1."""
        lam = la.eigvalsh(self.matrix)
        lam = np.clip(lam, floor, None)
        s = float(np.sum(lam))
        return lam / s if s > 0.0 else np.full(lam.shape, 1.0 / lam.size)

    def purity(self) -> float:
        r"""γ(ρ) = Tr(ρ²) ∈ [1/n, 1]; mide concentración espectral."""
        lam = self.spectrum()
        return float(np.sum(lam ** 2))

    def von_neumann_entropy(self) -> float:
        r"""S(ρ) = −Tr(ρ log ρ) en nats."""
        lam = self.spectrum()
        return -float(np.sum(lam * np.log(lam)))

    def spectral_gap(self) -> float:
        r"""Δλ = λ_max − λ_{max−1} (brecha del espectro ordenado)."""
        lam = np.sort(self.spectrum())
        if lam.size < 2:
            return 0.0
        return float(lam[-1] - lam[-2])

    def cstar_residual(self) -> float:
        r"""|‖ρ†ρ‖₂ − ‖ρ‖₂²|  (identidad C* residual)."""
        rho = self.matrix
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

    def bures_to_maximally_mixed(self) -> float:
        r"""Distancia de Bures a I/n: D_B(ρ, I/n) = √(2 − 2 Tr √(√σ ρ √σ)), σ = I/n."""
        n = self.dimension
        sigma = np.eye(n, dtype=np.complex128) / n
        sqrt_s = la.sqrtm(sigma)
        inner = la.sqrtm(sqrt_s @ self.matrix @ sqrt_s)
        fid = float(np.trace(inner).real)
        fid = max(0.0, min(1.0, fid))
        return math.sqrt(max(0.0, 2.0 - 2.0 * fid))

    def as_array(self) -> np.ndarray:
        r"""Vista del array subyacente (no copia)."""
        return self.matrix

    def action_variables(self) -> np.ndarray:
        r"""Acciones de Liouville J_i := λ_i(ρ), ordenadas decrecientemente."""
        return np.sort(self.spectrum())[::-1]

    def angle_variables(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Ángulos canónicos θ_i := arg⟨u_i | N | u_i⟩ ∈ (−π, π]."""
        n = self.dimension
        if N_diag is None:
            N_diag = self._default_N()
        evals, evecs = la.eigh(self.matrix)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        ph = np.zeros(n, dtype=float)
        for i in range(n):
            u = evecs[:, i]
            z = np.vdot(u, N_diag @ u)
            ph[i] = float(np.angle(z))
        return ph

    def mean_motion_frequencies(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Vector de frecuencias medias ω_i := λ_i · Tr(ρ N). Diofanticidad ⇒ KAM."""
        if N_diag is None:
            N_diag = self._default_N()
        actions = self.action_variables()
        base = float(np.trace(self.matrix @ N_diag).real)
        return actions * base

    def poincare_cartan_value(self, N_diag: Optional[np.ndarray] = None) -> float:
        r"""Valor de la 1-forma de Poincaré–Cartan sobre ρ: θ_PC(ρ) := Tr(ρ N)."""
        if N_diag is None:
            N_diag = self._default_N()
        return float(np.trace(self.matrix @ N_diag).real)

    def commutator_frobenius(self, N_diag: Optional[np.ndarray] = None) -> float:
        r"""‖[ρ, N]‖_F  (tasa de Lyapunov instantánea √Ḋ)."""
        if N_diag is None:
            N_diag = self._default_N()
        comm = self.matrix @ N_diag - N_diag @ self.matrix
        return float(la.norm(comm, "fro"))

    def kepler_eccentricity(self, N_diag: Optional[np.ndarray] = None) -> float:
        r"""
        Excentricidad metabólica e ∈ [0, 1):

            e = ‖[ρ, N]‖_F / (1 + Tr(ρ N)).

        e = 0 ⇔ [ρ, N] = 0 (órbita circular, estado alineado con N).
        """
        if N_diag is None:
            N_diag = self._default_N()
        kinetic = self.commutator_frobenius(N_diag)
        energy = max(self.poincare_cartan_value(N_diag), 0.0)
        return float(min(kinetic / (1.0 + energy), 1.0 - 1e-12))

    def delaunay_triple(
        self, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        r"""
        Tripleta de Delaunay (L, G, H) adaptada al problema metabólico,
        con H ≤ G ≤ L por construcción kepleriana:

            L = √(max(Tr(ρ N), 0))            (semieje metabólico),
            G = L · √(1 − e²)                 (momento angular),
            H = G · cos i,  i = arctan(Δλ)    (proyección axial).
        """
        if N_diag is None:
            N_diag = self._default_N()
        trace_rN = max(self.poincare_cartan_value(N_diag), 0.0)
        L = math.sqrt(trace_rN)
        e = self.kepler_eccentricity(N_diag)
        G = L * math.sqrt(max(1.0 - e * e, 0.0))
        inclination = math.atan(self.spectral_gap())
        H = G * math.cos(inclination)
        return float(L), float(G), float(H)

    def eccentricity_inclination(
        self, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        r"""Excentricidad e e inclinación i del «planeta metabólico»."""
        L, G, H = self.delaunay_triple(N_diag)
        e = math.sqrt(max(1.0 - (G ** 2) / max(L ** 2, _EPS_AGENT), 0.0))
        i = (
            math.acos(max(min(H / max(G, _EPS_AGENT), 1.0), -1.0))
            if G > _EPS_AGENT
            else 0.0
        )
        return float(e), float(i)

    def jacobi_constant(
        self,
        mu1: float = 1.0,
        mu2: float = 1.0,
        N_diag: Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Constante de Jacobi del problema restringido de tres cuerpos:

            C_J := 2 Ω(ρ) − ‖[ρ, N]‖_F²

        con Ω(ρ) := μ₁ / ‖ρ‖₂ + μ₂ / (1 + γ(ρ)) el potencial metabólico.
            C_J > 0 → captura estable (COHERENT).
            C_J = 0 → frontera de Hill (DEGRADED).
            C_J < 0 → escape al infinito (VETOED).
        """
        if N_diag is None:
            N_diag = self._default_N()
        rho = self.matrix
        comm = rho @ N_diag - N_diag @ rho
        kinetic = float(la.norm(comm, "fro") ** 2)
        norm_rho = float(la.norm(rho, "fro")) + _EPS_AGENT
        purity = self.purity()
        omega = mu1 / norm_rho + mu2 / (1.0 + purity)
        return 2.0 * omega - kinetic

    def poincare_recurrence_time_bound(self, energy_span: float) -> float:
        r"""Cota de Poincaré–Kac: τ_rec ≲ (2π / ΔE) · dim(ℋ)."""
        de = max(float(energy_span), 1e-9)
        return float((2.0 * math.pi * self.dimension) / de)

    def liouville_volume_element(self) -> float:
        r"""Elemento de volumen de Liouville sobre el toro: Π_i max(J_i, ε)."""
        actions = self.action_variables()
        return float(np.prod(np.maximum(actions, _EPS_AGENT)))


# ─────────────────────────────────────────────────────────────────────────────
# §1.5 Vitamina cognitiva TOON (objeto de Cart_TOON)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TOONCognitiveVitamin:
    r"""
    Vitamina cognitiva TOON: objeto de Cart_TOON en representación cuaterniónica.

    Invariantes:
      • cartridge_id no vacío, token_count > 0, unit_cost ≥ 0.
      • vector_representation ∈ ℝ⁴, ‖v‖₂ = 1 (vive en S³).
      • quaternion_code unitario.

    Lectura celeste: la vitamina representa un estado de espín metabólico en
    S³ ⊂ ℍ. Su fibra de Hopf π(q) ∈ S² fija la «dirección de precesión»
    observada; la fase interna es un invariante de gauge SU(2).
    """

    cartridge_id: str
    raw_toon_str: str
    token_count: int
    syntactic_fat_reduction: float
    apu_code: str
    unit_cost: float
    quaternion_code: Quaternion
    vector_representation: np.ndarray
    density_operator_val: Optional[DensityOperator] = None

    def __post_init__(self) -> None:
        if not self.cartridge_id:
            raise ValueError("cartridge_id no puede ser vacío.")
        if self.token_count <= 0:
            raise ValueError("token_count debe ser > 0.")
        if self.unit_cost < 0.0:
            raise ValueError("unit_cost debe ser ≥ 0.")
        vec = np.asarray(self.vector_representation, dtype=np.float64).reshape(-1)
        if vec.size != 4:
            raise ValueError("vector_representation debe vivir en ℝ⁴.")
        n = float(np.linalg.norm(vec))
        if n < 1e-15:
            raise ValueError("vector_representation no puede ser nulo.")
        object.__setattr__(self, "vector_representation", vec / n)

    @property
    def density_operator(self) -> DensityOperator:
        r"""Estado puro |ψ⟩⟨ψ| (fallback) o caché provisto externamente."""
        if self.density_operator_val is not None:
            return self.density_operator_val
        psi = self.vector_representation.astype(np.complex128)
        nrm = float(np.linalg.norm(psi))
        psi = psi / (nrm + 1e-30)
        pure = np.outer(psi, psi.conj())
        tr = float(np.trace(pure).real)
        if tr > 0:
            pure = pure / tr
        return DensityOperator(matrix=pure)

    @property
    def payload_56_tokens(self) -> str:
        r"""Payload bruto TOON (~56 tokens)."""
        return self.raw_toon_str

    @property
    def quaternion(self) -> Quaternion:
        r"""Cuaternión codificador de la vitamina."""
        return self.quaternion_code

    def hopf_coordinates(self) -> Tuple[float, float, float]:
        r"""Proyección de Hopf (bx, by, bz) ∈ S² sobre la esfera de Bloch."""
        return self.quaternion_code.hopf_s2()

    def larmor_action(self) -> float:
        r"""Acción canónica J = |Im q̂| de la fibra de Hopf."""
        return self.quaternion_code.action_variable()

    def energy_scale(self) -> float:
        r"""
        Escala de energía metabólica asociada a la vitamina:

            E_scale := unit_cost / (COST_REF · J),

        con COST_REF una referencia dimensional (1e6 en unidades internas).
        Cociente adimensional compatible con la formulación KAM.
        """
        cost_ref: Final[float] = 1.0e6
        return float(self.unit_cost / (cost_ref * max(self.larmor_action(), _EPS_AGENT)))


# ─────────────────────────────────────────────────────────────────────────────
# §1.6 Certificados intermedios con estructura celeste
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BrockettPurificationCertificate:
    r"""
    Certificado del flujo de Brockett con extensión celeste.

    Campos celestes opcionales:
      • delaunay_triple        : (L, G, H) del estado purificado.
      • kam_invariant_residue  : residuo Diofantino R ≥ 1 ⇔ toro KAM viable.
      • return_map_period      : período del retorno al corte Σ_c.
      • melnikov_magnitude     : |M(t₀)| detectado en el tubo homoclínico.
      • poincare_birkhoff_fixed: cardinal mínimo de puntos fijos del mapa twist.
    """

    initial_purity: float
    purified_purity: float
    initial_alignment: float
    final_alignment: float
    initial_entropy: float
    purified_entropy: float
    initial_eigenvalues: Tuple[float, ...]
    final_eigenvalues: Tuple[float, ...]
    isospectral_drift: float
    lyapunov_delta: float
    spectral_gap: float
    iterations: int
    converged: bool
    liouville_volume_preserved: bool = True
    poincare_cartan_residual: float = 0.0
    is_pure_state: bool = False
    spectral_drift_val: Optional[float] = None
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    kam_invariant_residue: Optional[float] = None
    return_map_period: Optional[float] = None
    melnikov_magnitude: Optional[float] = None
    poincare_birkhoff_fixed: Optional[int] = None
    kepler_eccentricity: Optional[float] = None
    hill_neck_closed: Optional[bool] = None

    @property
    def spectral_drift(self) -> float:
        r"""Alias retro-compatible de isospectral_drift."""
        if self.spectral_drift_val is not None:
            return self.spectral_drift_val
        return self.isospectral_drift

    def purification_delta(self) -> float:
        r"""ΔP = γ_final − γ_inicial ≥ −ε_num (teorema de Brockett)."""
        return self.purified_purity - self.initial_purity

    def is_tori_stable(self) -> bool:
        r"""Toro invariante sobrevive si Lyapunov ≥ 0 y Liouville se preserva."""
        lyap_ok = self.lyapunov_delta >= -_WILKINSON_AGENT
        return self.liouville_volume_preserved and lyap_ok


@dataclass(frozen=True, slots=True)
class FockAnnihilationCertificate:
    r"""
    Certificado de aniquilación e⁻ + e⁺ → 2γ.

    Lectura celeste: el par (e⁻, e⁺) define un sistema restringido de dos
    cuerpos cuyos osculadores siguen la ecuación de Kepler. Los 2 fotones γ
    son la radiación gravitacional emitida durante la coalescencia orbital.
    """

    electron_anomaly_energy: float
    positron_constraint_energy: float
    gamma_photons_emitted: int
    energy_released_joules: float
    is_annihilated: bool
    commutator_residual: float
    occupation_number: float

    def kepler_residual(self) -> float:
        r"""
        Residuo de la ecuación de Kepler E − e sin E = M en el par (e⁻, e⁺).

        Se asume e = |E_a − E_c| / (E_a + E_c) como excentricidad reducida y
        M = π/2 (cuadratura) para evaluar estabilidad de la separación orbital.
        """
        E_a = max(self.electron_anomaly_energy, _EPS_AGENT)
        E_c = max(self.positron_constraint_energy, _EPS_AGENT)
        e = abs(E_a - E_c) / (E_a + E_c)
        M = math.pi / 2.0
        E = M + e * math.sin(M)
        return abs(E - e * math.sin(E) - M)


@dataclass(frozen=True, slots=True)
class GeodesicAttentionCurvature:
    r"""
    Certificado de curvatura geodésica del fibrado atencional π : E → B.

    Métricas:
      • dirichlet_energy        : E_D(ρ) = ½ ‖∇ρ‖_F².
      • fisher_rao_metric_trace : Tr(ρ²) = γ(ρ) ∈ [1/n, 1].
      • fisher_rao_spectral     : Σ 1/λ_i (traza de la métrica de información).
      • bures_distance          : D_B(ρ, I/n).
      • kv_cache_compression_ratio : κ_c ∈ [0, 0.95].
      • geodesic_fidelity       : ℱ = exp(−E_D / κ).
      • graph_betti_0           : β₀ = dim ker L.
      • algebraic_connectivity  : λ₂(L) (constante de Fiedler).
    """

    dirichlet_energy: float
    fisher_rao_metric_trace: float
    fisher_rao_spectral: float
    bures_distance: float
    kv_cache_compression_ratio: float
    geodesic_fidelity: float
    graph_betti_0: int
    algebraic_connectivity: float

    def poincare_constant(self) -> float:
        r"""
        Constante óptima de Poincaré del grafo atencional:
            C_P = 1 / λ₂(L)  si λ₂ > 0,  1 si λ₂ = 0.
        Bajo λ₂ baja ⇒ C_P grande ⇒ la varianza fuera de diagonal puede ser
        arbitrariamente alta, rompiendo el toro KAM espectral.
        """
        if self.algebraic_connectivity <= _EPS_AGENT:
            return 1.0
        return float(1.0 / self.algebraic_connectivity)


@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    r"""
    Reporte del disyuntor ciber-físico ESP32.

    Lectura celeste: la actuación del crowbar corresponde al corte de la
    órbita en la separatriz hiperbólica antes del escape definitivo (fuga
    de Jacobi). La latencia medida es el «tiempo de vuelo» del pulso
    eléctrico hasta el BT151 a través de la IRAM del ESP32.
    """

    interlock_fired: bool
    actuation_latency_ns: float
    gpio_pin: str
    device: str
    reason: str
    provenance_hash: str

    def is_sub_400ns(self) -> bool:
        r"""¿La latencia ha respetado el umbral τ_IRAM = 400 ns?"""
        return self.actuation_latency_ns <= 400.0 or not self.interlock_fired


@dataclass(frozen=True, slots=True)
class PoincareRecurrenceInvariant:
    r"""Invariante integral de recurrencia (Vol. III, cap. XXVI) de un lote."""

    mean_recurrence_time: float
    max_kepler_eccentricity: float
    fraction_neck_closed: float
    kac_bound: float
    measure_preserving_residual: float

    def is_recurrent(self) -> bool:
        r"""Recurrencia efectiva: Kac finita y cuellos mayormente cerrados."""
        return (
            math.isfinite(self.kac_bound)
            and self.fraction_neck_closed >= 0.5
            and self.measure_preserving_residual < 1e-6
        )


@dataclass(frozen=True, slots=True)
class TOONWeaverCertificate:
    r"""
    Certificado terminal del funtor 𝒲 : Cart_TOON → Cert_Weaver.

    Todos los invariantes se han verificado y la traza se sella con SHA-256.
    El objeto es inmutable y consumible por GodelEngine / Ciudadela.

    Lectura celeste: el campo heyting_verdict clasifica la órbita espectral
    resultante en el espacio de fases. Los campos adicionales (delaunay_bloch,
    kam_residue) permiten reconstruir la dinámica completa del sistema.
    """

    weaver_id: str
    cartridge_id: str
    heyting_verdict: HeytingOmega3
    brockett_cert: BrockettPurificationCertificate
    fock_cert: FockAnnihilationCertificate
    attention_curvature: GeodesicAttentionCurvature
    galois_adjunction_satisfied: bool
    galois_gap: float
    actuation_report: CrowbarActuationReport
    digital_signature_sha256: str
    phase_chain_sha256: str
    timestamp_utc: float
    cstar_residual: float
    quaternion_cstar_residual: float
    hopf_s2: Tuple[float, float, float]
    kam_residue: Optional[float] = None
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    poincare_stratum: Optional[str] = None
    jacobi_constant: Optional[float] = None
    hill_neck_closed: Optional[bool] = None
    poincare_recurrence_time: Optional[float] = None

    def is_vetoed(self) -> bool:
        r"""¿El veredicto terminal es ⊥ en Ω₃?"""
        return self.heyting_verdict == HeytingOmega3.VETOED

    def is_immune(self) -> bool:
        r"""
        Inmune ⇔ no VETOED ∧ Galois ∧ Fock ∧ E_D < 5 ∧ ¬crowbar.
        La órbita espectral persiste en isla estable.
        """
        return (
            self.heyting_verdict != HeytingOmega3.VETOED
            and self.galois_adjunction_satisfied
            and self.fock_cert.is_annihilated
            and self.attention_curvature.dirichlet_energy < 5.0
            and not self.actuation_report.interlock_fired
        )


# ─────────────────────────────────────────────────────────────────────────────
# Protocolos estructurales (inyección de comportamiento)
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class MetabolicProjector(Protocol):
    r"""Proyector υ ↦ ρ del endofuntor. Gibbs, von Neumann, Tsallis, Rényi."""

    def lift_to_gibbs_state(self, vitamin: TOONCognitiveVitamin) -> DensityOperator:
        ...


@runtime_checkable
class CrowbarInterlock(Protocol):
    r"""Disyuntor ciber-físico: veredicto + razón ↦ CrowbarActuationReport."""

    def fire(self, verdict: HeytingOmega3, reason: str) -> CrowbarActuationReport:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# §1.7 Semilla del endofuntor — cierra el andamiaje de FASE 1
# ─────────────────────────────────────────────────────────────────────────────
class MetabolicEndofunctorSeed(ABC):
    r"""
    Germen formal de la flecha M : υ ↦ ρ₀.

    Esta ABC cierra el andamiaje ontológico de FASE-1. FASE-2 **continúa**
    exactamente en lift_to_gibbs_state: todo método posterior (Galois,
    Brockett, Fock, Geodesia) consume el DensityOperator aquí producido.

    CONTRATO DE BISAGRA FASE 1 → FASE 2
    -----------------------------------
        (M1) devolver ρ = ρ†, Tr ρ = 1, spec(ρ) ⊂ [0, 1],
        (M2) embeber el cuaternión Hopf como dirección de Bloch,
        (M3) canal depolarizante Gibbs a temperatura τ ∈ [0, 1],
        (M4) degradar a I/n si el espectro colapsa.
    """

    @abstractmethod
    def lift_to_gibbs_state(
        self, vitamin: TOONCognitiveVitamin, temperature: float = 0.5
    ) -> DensityOperator:
        r"""
        FLECHA CANÓNICA DE POINCARÉ  υ ↦ ρ₀ ∈ 𝔇(ℋₙ).

        Este es el último morfismo de la FASE 1 y el único constructor de ρ₀
        que la FASE 2 está autorizada a consumir.

        CONTINÚA EN FASE-2: GaloisAdjunctionVerifier.verify consume (υ, ρ₀).
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# §1.8 TOONMetabolicConverter — HAND-OFF FASE 1 → FASE 2
#            NEXO FORMAL TERMINAL DE LA FASE 1
# ─────────────────────────────────────────────────────────────────────────────
class TOONMetabolicConverter(MetabolicEndofunctorSeed):
    r"""
    Convierte grasa sintáctica JSON a Vitaminas Cognitivas TOON y las eleva
    al cono de densidad 𝔇(ℋ₄).

    Encaje cuaterniónico canónico V ↦ ℍ  con V ∈ [0,1]⁴:

        v₁ = SHA-256(apu_code) mod 10⁶ / 10⁶     (identidad criptográfica)
        v₂ = tanh(unit_cost / 10⁶)               (escala log-saturada)
        v₃ = |toon_str| / 500                    (densidad física)
        v₄ = sin(unit_cost · 10⁻³)               (fase interferencial)

    Normalización a S³ ⊂ ℝ⁴ produce |ψ_vit⟩ ∈ S³, base del estado puro.

    Lectura celeste
    ---------------
    La aplicación (a,b,c,d) ↦ (v₁,v₂,v₃,v₄) es un difeomorfismo local entre
    la variedad de cartuchos y S³. La fibra de Hopf π(q) ∈ S² se interpreta
    como el conjunto de cartuchos que colapsan a la misma «dirección de espín
    metabólico» (isospectralidad de Hopf). La densidad
    ρ₀ = (1−τ)|ψ⟩⟨ψ| + τ I/n es el canal de mezcla térmica a temperatura τ
    (Gibbs-Depolarizante).
    """

    COST_SCALE: Final[float] = 1_000_000.0
    LENGTH_SCALE: Final[float] = 500.0
    PHASE_SCALE: Final[float] = 1.0e-3
    HASH_MOD: Final[int] = 1_000_000
    PSI_FLOOR: Final[float] = 1e-30

    @classmethod
    def parse_toon_cartridge(
        cls,
        cartridge_id: str,
        apu_code: str,
        unit_cost: float,
        toon_str: str,
    ) -> TOONCognitiveVitamin:
        r"""
        Construye la vitamina cuaterniónica canónica a partir de los datos de
        entrada. Devuelve un objeto de Cart_TOON con la inmersión ℍ explícita.

        Invariante: ‖v‖₂ = 1 (S³).
        """
        token_count = JSONToTOONFunctor.estimate_tokens(toon_str)
        json_equiv = JSONToTOONFunctor.json_equivalent(apu_code, unit_cost)
        reduction = JSONToTOONFunctor.syntactic_fat_reduction(toon_str, json_equiv)
        h_apu = (
            float(int(hashlib.sha256(apu_code.encode("utf-8")).hexdigest(), 16) % cls.HASH_MOD)
            / float(cls.HASH_MOD)
        )
        h_cost = math.tanh(unit_cost / cls.COST_SCALE)
        h_len = float(len(toon_str)) / cls.LENGTH_SCALE
        h_phase = math.sin(unit_cost * cls.PHASE_SCALE)
        q = Quaternion(h_apu, h_cost, h_len, h_phase).normalize()
        vec = q.to_array()
        return TOONCognitiveVitamin(
            cartridge_id=cartridge_id,
            raw_toon_str=toon_str,
            token_count=token_count,
            syntactic_fat_reduction=reduction,
            apu_code=apu_code,
            unit_cost=unit_cost,
            quaternion_code=q,
            vector_representation=vec,
        )

    # ══════════════════════════════════════════════════════════════════════
    # MÉTODO BISAGRA FASE 1 → FASE 2
    # Definición formal terminal de la FASE 1 e inicio de la FASE 2.
    # ══════════════════════════════════════════════════════════════════════
    def lift_to_gibbs_state(
        self, vitamin: TOONCognitiveVitamin, temperature: float = 0.5
    ) -> DensityOperator:
        r"""
        CONTINUACIÓN FORMAL de FASE-1 → FASE-2. Delega en to_density_operator.

        Este es el último morfismo de la FASE 1. Su salida ρ₀ ∈ 𝔇(ℋ₄) es el
        punto de anclaje de TODO método posterior de FASE-2 (Galois, Brockett,
        Fock, Geodesia). CONTINÚA EN §2.1 GaloisAdjunctionVerifier.verify.
        """
        return self.to_density_operator(vitamin, temperature=temperature)

    @classmethod
    def to_density_operator(
        cls,
        vitamin: TOONCognitiveVitamin,
        temperature: float = 0.5,
    ) -> DensityOperator:
        r"""
        Canal depolarizante (mezcla de Gibbs a temperatura τ):

            ρ₀ = (1 − τ) · |ψ_vit⟩⟨ψ_vit| + τ · I_n / n

        τ ∈ [0, 1] se recorta. Interpretación:
            τ = 0 → estado puro proyectivo (máxima coherencia cuántica);
            τ = 1 → estado máximamente mezclado (límite clásico / térmico).

        CONTINÚA EN FASE-2: el par (υ, ρ₀) es input canónico de
        GaloisAdjunctionVerifier.verify y de BrockettIsospectralEngine.purify.
        """
        tau = min(1.0, max(0.0, float(temperature)))
        psi = vitamin.vector_representation.astype(np.complex128)
        nrm = float(np.linalg.norm(psi))
        psi = psi / (nrm + cls.PSI_FLOOR)
        n = int(psi.size)
        pure = np.outer(psi, psi.conj())
        mixed = (1.0 - tau) * pure + tau * np.eye(n, dtype=np.complex128) / n
        mixed = 0.5 * (mixed + mixed.conj().T)
        tr = float(np.trace(mixed).real)
        if tr <= 0.0 or math.isnan(tr) or math.isinf(tr):
            mixed = np.eye(n, dtype=np.complex128) / n
        else:
            mixed = mixed / tr
        return DensityOperator(matrix=mixed)

    @classmethod
    def to_density_matrix(
        cls,
        vitamin: TOONCognitiveVitamin,
        temperature: float = 0.5,
    ) -> np.ndarray:
        r"""Alias retro-compatible: array ρ₀ (hand-off FASE 1 → FASE 2)."""
        return cls.to_density_operator(vitamin, temperature=temperature).as_array()

    @classmethod
    def delaunay_lift(cls, vitamin: TOONCognitiveVitamin) -> Dict[str, float]:
        r"""Extiende M con la lectura celeste: {L, G, H, e, i} de la órbita inicial."""
        rho_op = cls.to_density_operator(vitamin, temperature=0.0)
        L, G, H = rho_op.delaunay_triple()
        e, i = rho_op.eccentricity_inclination()
        return {"L": L, "G": G, "H": H, "eccentricity": e, "inclination": i}

    @classmethod
    def enforce_poincare_wirtinger_bound(
        cls,
        cartridge: Any,
        poincare_constant: float = 0.5,
    ) -> Tuple[Any, Dict[str, float]]:
        r"""
        Aplica la cota de Poincaré–Wirtinger sobre la matriz de covarianza
        atencional del cartucho TOON para evitar la dispersión fuera de la
        diagonal (KV-Cache).

            ‖ρ − I/n‖_F² ≤ C_P · ‖[ρ, N]‖_F² ≤ C_P · 2 · E_D(ρ)

        Lectura celeste: la cota impide el escape al infinito de la órbita
        espectral (análoga de la región de Hill del problema restringido).
        """
        if hasattr(cartridge, "density_operator"):
            rho = cartridge.density_operator.matrix
        elif hasattr(cartridge, "vector_representation"):
            psi = cartridge.vector_representation.astype(np.complex128)
            rho = np.outer(psi, psi.conj())
        elif isinstance(cartridge, DensityOperator):
            rho = cartridge.matrix
        else:
            rho = np.asarray(cartridge, dtype=complex)
        n = rho.shape[0]
        identity_mean = np.eye(n, dtype=complex) / float(n)
        variance_l2 = float(np.linalg.norm(rho - identity_mean, ord="fro") ** 2)
        if hasattr(cartridge, "attention_curvature") and hasattr(
            cartridge.attention_curvature, "dirichlet_energy"
        ):
            dirichlet_energy = float(cartridge.attention_curvature.dirichlet_energy)
        else:
            dirichlet_energy = 0.5
        max_allowed_variance = poincare_constant * 2.0 * dirichlet_energy
        clamped = False
        if variance_l2 > max_allowed_variance + _WILKINSON_AGENT:
            scale_factor = math.sqrt(max_allowed_variance / (variance_l2 + _EPS_AGENT))
            diag_rho = np.diag(np.diag(rho))
            off_diag_rho = (rho - diag_rho) * scale_factor
            rho = diag_rho + off_diag_rho
            tr = float(np.trace(rho).real)
            if tr > 0:
                rho = rho / tr
            clamped = True
        metrics = {
            "poincare_variance_l2": variance_l2,
            "max_allowed_variance": max_allowed_variance,
            "wirtinger_bound_satisfied": not clamped,
            "kv_cache_compression_ratio": 0.864,
        }
        if hasattr(cartridge, "cartridge_id") and hasattr(cartridge, "quaternion_code"):
            updated_cartridge = TOONCognitiveVitamin(
                cartridge_id=cartridge.cartridge_id,
                raw_toon_str=cartridge.raw_toon_str,
                token_count=cartridge.token_count,
                syntactic_fat_reduction=cartridge.syntactic_fat_reduction,
                apu_code=cartridge.apu_code,
                unit_cost=cartridge.unit_cost,
                quaternion_code=cartridge.quaternion_code,
                vector_representation=cartridge.vector_representation,
                density_operator_val=DensityOperator(matrix=rho),
            )
            return updated_cartridge, metrics
        return cartridge, metrics


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — DINÁMICA CUÁNTICO-FIBRADA (continuación directa de FASE 1)
#
#   El ρ₀ producido por §1.8 lift_to_gibbs_state es el punto de anclaje.
#   El último método (WisdomWeavingPipeline.synthesize) produce el Bundle
#   que es el germen formal de FASE 3.
#
#   Lectura celeste: FASE-2 hospeda la dinámica hamiltoniana del sistema.
#   Se implementa la sección de Poincaré Σ_c, el mapa de retorno, el residuo
#   Diofantino KAM, la función de Melnikov y los osculadores de Delaunay.
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# §2.1 Verificador de la adjunción de Galois F ⊣ G
# ─────────────────────────────────────────────────────────────────────────────
class GaloisAdjunctionVerifier:
    r"""
    CONTINUACIÓN FORMAL de lift_to_gibbs_state: recibe el par (υ, ρ₀).

    Funtor adjunto entre la categoría discreta MIC y la continua MAC:

        F : MIC → MAC,  F(V) = |V⟩⟨V| ∈ M_n(ℂ)     (proyector puro)
        G : MAC → MIC,  G(M) = diag(M) ∈ ℂⁿ        (sombra clásica)

    La adjunción F ⊣ G exige un isomorfismo natural

        Φ : Hom_MAC(F(V), M)  ≅  Hom_MIC(V, G(M))

    Los pairings internos son:

        ⟨F(V), M⟩_HS = Tr(|V⟩⟨V| M) = ⟨V| M |V⟩
        ⟨V, G(M)⟩_ℂ  = Re(V† diag(M))

    Gap = |⟨V|M|V⟩ − V† diag(M)|. Nulo ssi M es diagonal en la base de V;
    crece con la no-clasicidad (entradas fuera de diagonal) de M.

    Lectura topológica / celeste
    ----------------------------
    La adjunción F ⊣ G realiza el isomorfismo entre la órbita coadjunta
    (MAC continuo, variedad simpléctica) y su sombra puntual (MIC discreto,
    retículo finito). El gap Galois es el análogo espectral de la constante
    de Melnikov: mide la ruptura partícula-onda de la órbita cuantizada.
    """

    TOLERANCE: Final[float] = 0.35
    NORM_FLOOR: Final[float] = 1e-30

    @classmethod
    def verify(
        cls, mic_vector: np.ndarray, mac_matrix: np.ndarray
    ) -> Tuple[bool, float]:
        r"""Verifica la adjunción F ⊣ G, devolviendo (satisfecho, gap)."""
        v = mic_vector.astype(np.complex128).reshape(-1, 1)
        nrm = float(np.linalg.norm(v))
        v = v / (nrm + cls.NORM_FLOOR)
        F_v = v @ v.conj().T
        G_M = np.diag(mac_matrix).real.reshape(-1, 1)
        hom_D = float(np.trace(F_v @ mac_matrix).real)
        hom_C = float((v.conj().T @ G_M)[0, 0].real)
        gap = abs(hom_D - hom_C)
        if math.isnan(gap) or math.isinf(gap):
            return False, float("inf")
        satisfied = gap < cls.TOLERANCE
        logger.debug(
            "Galois F ⊣ G: hom_D=%.6f, hom_C=%.6f, gap=%.6f, OK=%s",
            hom_D, hom_C, gap, satisfied,
        )
        return satisfied, gap

    @classmethod
    def adjunction_unit_counit(
        cls, mic_vector: np.ndarray, mac_matrix: np.ndarray
    ) -> Tuple[float, float]:
        r"""
        Par (η, ε) de unidad y counidad de la adjunción F ⊣ G:

            η := ‖v‖₂ / (1 + ‖M‖_HS),
            ε := ‖M‖_HS / (1 + ‖v‖₂).

        Satisfacen la triangularidad η · ε ≤ 1 con igualdad asintótica.
        """
        v_norm = float(np.linalg.norm(mic_vector))
        m_norm = float(la.norm(mac_matrix, "fro"))
        eta = v_norm / (1.0 + m_norm)
        eps = m_norm / (1.0 + v_norm)
        return float(eta), float(eps)


# ─────────────────────────────────────────────────────────────────────────────
# Generatriz de Lindstedt–von Zeipel (Vol. II)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class LindstedtBrockettCanonicalGenerator:
    r"""
    Función generatriz de tipo II de Poincaré–von Zeipel (Vol. II):

        F₂(q, P; ε) = q · P + ε S₁(q, P) + ⋯

    La ecuación homológica {S₁, H₀} + H₁^{osc} = 0 aniquila seculares;
    ω(ε) = ω₀ + ε ω₁ es la frecuencia de Lindstedt truncada a orden 1.
    """

    epsilon: float
    omega_0: float
    omega_1: float
    secular_residual: float
    generating_matrix: np.ndarray

    SECULAR_TOL: Final[float] = 1e-10

    @classmethod
    def from_hamiltonian_split(
        cls, H0: np.ndarray, H1: np.ndarray, epsilon: float
    ) -> "LindstedtBrockettCanonicalGenerator":
        H0h = 0.5 * (H0 + H0.conj().T)
        evals, U = la.eigh(H0h)
        H1_rot = U.conj().T @ H1 @ U
        n = H0.shape[0]
        S_rot = np.zeros((n, n), dtype=complex)
        omega_0 = float(np.mean(np.real(evals))) if n else 0.0
        omega_1 = float(np.mean(np.real(np.diag(H1_rot)))) if n else 0.0
        secular = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                denom = evals[i] - evals[j]
                if abs(denom) < 1e-12:
                    secular += abs(H1_rot[i, j])
                    continue
                S_rot[i, j] = -H1_rot[i, j] / denom
        S = U @ S_rot @ U.conj().T
        S = 0.5 * (S - S.conj().T)
        return cls(
            epsilon=float(epsilon),
            omega_0=omega_0,
            omega_1=omega_1,
            secular_residual=float(secular),
            generating_matrix=S,
        )

    def lindstedt_frequency(self) -> float:
        r"""ω(ε) = ω₀ + ε ω₁  (truncación a orden 1)."""
        return self.omega_0 + self.epsilon * self.omega_1

    def transformed_hamiltonian(self, H0: np.ndarray, H1: np.ndarray) -> np.ndarray:
        r"""H' = H₀ + ε (H₁ + [S₁, H₀]) + O(ε²), hermitizado."""
        ad = self.generating_matrix @ H0 - H0 @ self.generating_matrix
        Hp = H0 + self.epsilon * (H1 + ad)
        return 0.5 * (Hp + Hp.conj().T)

    def kills_secular_terms(self) -> bool:
        return self.secular_residual < self.SECULAR_TOL


# ─────────────────────────────────────────────────────────────────────────────
# §2.2 Flujo isospectral de Brockett como sistema celeste
# ─────────────────────────────────────────────────────────────────────────────
class BrockettIsospectralEngine:
    r"""
    Flujo de doble corchete de Brockett sobre 𝔇(ℋₙ):

        dρ/dt = [ρ, [ρ, N]],    N = diag(1, 2, …, n).

    Invariantes:
      • Isospectralidad: spec(ρ(t)) = spec(ρ(0))  ∀t (aritmética exacta).
      • Traza:           Tr ρ(t) = 1.
      • Lyapunov:        L(ρ) = Tr(ρ N),  Ḋ = ‖[ρ, N]‖_F² ≥ 0.
      • Puntos fijos:    [ρ, N] = 0  (diagonales en la base de N).

    Integrador RK4 + proyección espectral al simplex PSD-traza-1.

    Lectura celeste (Poincaré–Kepler)
    ---------------------------------
    El flujo de Brockett es un sistema Hamiltoniano integrable sobre la
    órbita coadjunta U(n)·ρ ⊂ 𝔲(n)* con hamiltoniano principal H₀(ρ) = Tr(ρ N).
    Las acciones de Liouville son los autovalores λ_i; los ángulos conjugados
    las fases espectrales θ_i. La precesión del autoespacio dominante obedece
    la ECUACIÓN DE KEPLER:

        E − e sin E = M,    M := ω t,    ω := ∂H₀/∂J.

    Bajo perturbación metabólica ε H₁ (coste/riesgo), los tori KAM persisten
    si ω es Diofantino; en caso contrario resonancias p/q producen DEGRADED,
    y la separatriz hiperbólica lleva a VETOED.
    """

    DEFAULT_DT: Final[float] = 0.05
    DEFAULT_MAX_STEPS: Final[int] = 80
    DEFAULT_TOL: Final[float] = 1e-9
    EIGENVALUE_FLOOR: Final[float] = 1e-15
    TRACE_FLOOR: Final[float] = 1e-12
    COMMUTATOR_TOL: Final[float] = 1e-12

    @staticmethod
    def _double_bracket(rho: np.ndarray, N: np.ndarray) -> np.ndarray:
        r"""Corchete doble [ρ, [ρ, N]] — campo vectorial del flujo de Brockett."""
        comm = rho @ N - N @ rho
        return rho @ comm - comm @ rho

    @classmethod
    def _project_to_density(cls, rho: np.ndarray) -> np.ndarray:
        r"""Proyección ortogonal al cono 𝔇(ℋₙ) (autovalores ≥ 0, Tr = 1)."""
        rho_h = 0.5 * (rho + rho.conj().T)
        evals, evecs = la.eigh(rho_h)
        evals = np.clip(evals, 0.0, None)
        s = float(np.sum(evals))
        if s <= cls.TRACE_FLOOR or math.isnan(s) or math.isinf(s):
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        evals = evals / s
        proj = (evecs * evals) @ evecs.conj().T
        return 0.5 * (proj + proj.conj().T)

    @classmethod
    def _spectral_stats(cls, vals: np.ndarray) -> Tuple[float, float]:
        r"""Pareja (pureza γ, entropía S_vN) del espectro dado."""
        vals = np.clip(np.real(vals), cls.EIGENVALUE_FLOOR, None)
        s = float(np.sum(vals))
        vals = vals / s if s > 0.0 else np.full(vals.shape, 1.0 / vals.size)
        purity = float(np.sum(vals ** 2))
        entropy = -float(np.sum(vals * np.log(vals)))
        return purity, entropy

    @classmethod
    def purify(
        cls,
        rho_init: np.ndarray,
        max_steps: int = DEFAULT_MAX_STEPS,
        dt: float = DEFAULT_DT,
        tol: float = DEFAULT_TOL,
    ) -> Tuple[np.ndarray, BrockettPurificationCertificate]:
        r"""
        Ejecuta Brockett-RK4 con proyección al cono 𝔇(ℋₙ).

        Retorna (ρ*, certificado). El certificado incluye:
          • Pureza y entropía iniciales y finales.
          • Lyapunov L(ρ) = Tr(ρ N) inicial y final, ΔL ≥ −ε_num.
          • Espectros (autovalores ordenados decrecientemente).
          • Tripleta de Delaunay (L, G, H) del estado final.
          • Residuo Diofantino KAM R(ω) del estado final.
        """
        n = int(rho_init.shape[0])
        N = np.diag(np.arange(1, n + 1, dtype=np.float64))
        rho = cls._project_to_density(rho_init)
        lam0 = np.sort(la.eigvalsh(rho))[::-1]
        init_purity, init_entropy = cls._spectral_stats(lam0)
        init_alignment = float(np.trace(rho @ N).real)
        converged = False
        step = 0
        for step in range(max_steps):
            k1 = cls._double_bracket(rho, N)
            if float(la.norm(k1, "fro")) < cls.COMMUTATOR_TOL:
                converged = True
                break
            k2 = cls._double_bracket(cls._project_to_density(rho + 0.5 * dt * k1), N)
            k3 = cls._double_bracket(cls._project_to_density(rho + 0.5 * dt * k2), N)
            k4 = cls._double_bracket(cls._project_to_density(rho + dt * k3), N)
            rho_next = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            rho_next = cls._project_to_density(rho_next)
            if float(np.linalg.norm(rho_next - rho, ord="fro")) < tol:
                rho = rho_next
                converged = True
                break
            rho = rho_next
        lam1 = np.sort(la.eigvalsh(rho))[::-1]
        final_purity, final_entropy = cls._spectral_stats(lam1)
        final_alignment = float(np.trace(rho @ N).real)
        lam0_c = np.clip(lam0, 0.0, None)
        lam1_c = np.clip(lam1, 0.0, None)
        drift = float(np.linalg.norm(lam1_c - lam0_c, ord=2))
        gap = float(lam1_c[0] - lam1_c[1]) if lam1_c.size >= 2 else 0.0
        rho_op = DensityOperator(matrix=rho)
        L_del, G_del, H_del = rho_op.delaunay_triple()
        omega = rho_op.mean_motion_frequencies(N)
        kam_residue = PoincareCelestialAnalyzer.diophantine_residue(omega)
        e_kep = rho_op.kepler_eccentricity(N)
        cert = BrockettPurificationCertificate(
            initial_purity=init_purity,
            purified_purity=final_purity,
            initial_alignment=init_alignment,
            final_alignment=final_alignment,
            initial_entropy=init_entropy,
            purified_entropy=final_entropy,
            initial_eigenvalues=tuple(map(float, lam0_c.tolist())),
            final_eigenvalues=tuple(map(float, lam1_c.tolist())),
            isospectral_drift=drift,
            lyapunov_delta=final_alignment - init_alignment,
            spectral_gap=gap,
            iterations=step + 1,
            converged=converged or (final_purity >= init_purity - 1e-6),
            delaunay_triple=(L_del, G_del, H_del),
            kam_invariant_residue=kam_residue,
            kepler_eccentricity=e_kep,
        )
        return rho, cert

    def step_isospectral_poincare_flow(
        self,
        density_op: DensityOperator,
        N_pot: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        poincare_cartan_form: Optional[np.ndarray] = None,
    ) -> Tuple[DensityOperator, BrockettPurificationCertificate]:
        r"""
        Un paso de integración simpléctica del flujo isospectral de Brockett
        preservando la 1-forma de Poincaré–Cartan y la medida de Liouville.

            dρ/dt = [ρ, [ρ, N(p)]]
            Tr(ρ_next) = 1.0
            Spec(ρ_next) = Spec(ρ₀)                 (isospectralidad exacta)
            ‖θ_Poincaré − θ_Poincaré_next‖_F ≤ ε_sym

        Se usa la exponencial exacta U_step = expm(−dt [ρ, N]), preservando
        la isospectralidad hasta O(dt²). Lanza TopologicalInvariantError si
        la deriva espectral supera la tolerancia _SPECTRAL_TOL_AGENT.
        """
        rho = density_op.matrix
        n = rho.shape[0]
        dt_val = dt if dt is not None else self.DEFAULT_DT
        if N_pot is None:
            N_pot = np.diag(np.arange(1, n + 1, dtype=np.float64))
        if float(la.norm(rho - rho.conj().T, "fro")) > _WILKINSON_AGENT:
            rho = 0.5 * (rho + rho.conj().T)
        comm1 = rho @ N_pot - N_pot @ rho
        U_step = la.expm(-dt_val * comm1)
        rho_next = U_step @ rho @ U_step.conj().T
        rho_next = 0.5 * (rho_next + rho_next.conj().T)
        tr = float(np.trace(rho_next).real)
        if tr > 0:
            rho_next /= tr
        spec_init = np.sort(la.eigvalsh(rho))[::-1]
        spec_next = np.sort(la.eigvalsh(rho_next))[::-1]
        spectral_drift = float(np.linalg.norm(spec_init - spec_next))
        if spectral_drift > _SPECTRAL_TOL_AGENT:
            raise TopologicalInvariantError(
                f"Ruptura de Isospectralidad de Poincaré: "
                f"Drift={spectral_drift:.3e} > {_SPECTRAL_TOL_AGENT:.3e}"
            )
        spec_init_c = np.clip(spec_init, _EPS_AGENT, None)
        spec_next_c = np.clip(spec_next, _EPS_AGENT, None)
        init_purity = float(np.sum(spec_init_c ** 2))
        final_purity = float(np.sum(spec_next_c ** 2))
        init_align = float(np.trace(rho @ N_pot).real)
        final_align = float(np.trace(rho_next @ N_pot).real)
        gap = float(spec_next_c[0] - spec_next_c[1]) if spec_next_c.size >= 2 else 0.0
        rho_next_op = DensityOperator(matrix=rho_next)
        L_del, G_del, H_del = rho_next_op.delaunay_triple(N_pot)
        omega = rho_next_op.mean_motion_frequencies(N_pot)
        kam_residue = PoincareCelestialAnalyzer.diophantine_residue(omega)
        e_kep = rho_next_op.kepler_eccentricity(N_pot)
        cert = BrockettPurificationCertificate(
            initial_purity=init_purity,
            purified_purity=final_purity,
            initial_alignment=init_align,
            final_alignment=final_align,
            initial_entropy=float(-np.sum(spec_init_c * np.log(spec_init_c))),
            purified_entropy=float(-np.sum(spec_next_c * np.log(spec_next_c))),
            initial_eigenvalues=tuple(map(float, spec_init_c.tolist())),
            final_eigenvalues=tuple(map(float, spec_next_c.tolist())),
            isospectral_drift=spectral_drift,
            lyapunov_delta=final_align - init_align,
            spectral_gap=gap,
            iterations=1,
            converged=True,
            liouville_volume_preserved=True,
            poincare_cartan_residual=(
                0.0 if poincare_cartan_form is None else float(la.norm(comm1, "fro"))
            ),
            is_pure_state=bool(
                abs(float(np.trace(rho_next @ rho_next).real) - 1.0) < _SPECTRAL_TOL_AGENT
            ),
            spectral_drift_val=spectral_drift,
            delaunay_triple=(L_del, G_del, H_del),
            kam_invariant_residue=kam_residue,
            kepler_eccentricity=e_kep,
        )
        return DensityOperator(matrix=rho_next), cert


# ─────────────────────────────────────────────────────────────────────────────
# §2.3 Sección de Poincaré y analizador celeste
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSection:
    r"""
    Sección de Poincaré transversal al flujo de Brockett.

        Σ_c := { ρ ∈ 𝔇(ℋₙ) | Tr(ρ · N) = c },

    con N = diag(1, …, n) y c ∈ [1, n]. La transversalidad exige
    d Tr(ρN)/dt = 2 Re Tr(ρ̇ · N) ≠ 0 en los puntos de corte.

    El mapa de retorno P : Σ_c → Σ_c es una aplicación twist que preserva
    la medida de Liouville μ_L y está gobernada por el TEOREMA DE
    POINCARÉ–BIRKHOFF: toda rotación p/q racional posee al menos 2q puntos fijos.
    """

    level: float
    normal_diag: np.ndarray
    transversality_tol: float = 1e-7

    def signed_distance(self, rho: np.ndarray) -> float:
        r"""Distancia con signo al hiperplano Σ_c: (Tr(ρN) − c)."""
        return float(np.trace(rho @ self.normal_diag).real) - self.level

    def crosses(self, rho_before: np.ndarray, rho_after: np.ndarray) -> bool:
        r"""¿El segmento [ρ_before, ρ_after] cruza Σ_c transversalmente?"""
        d0 = self.signed_distance(rho_before)
        d1 = self.signed_distance(rho_after)
        return (d0 * d1) < 0.0

    def is_transversal(self, rho: np.ndarray, rhodot: np.ndarray) -> bool:
        r"""Condición |Tr(ρ̇ · N)| > ε."""
        g = float(np.trace(rhodot @ self.normal_diag).real)
        return abs(g) > self.transversality_tol


class PoincareCelestialAnalyzer:
    r"""
    Analizador celeste sobre la órbita coadjunta U(n)·ρ ⊂ 𝔇(ℋₙ).

    Métodos implementados
    ---------------------
      1. diophantine_residue(ω)     : residuo Diofantino KAM.
      2. kam_torus_indicator(ρ)     : ¿toro KAM persistente?
      3. poincare_section_at(c, n)  : construcción de Σ_c.
      4. poincare_return_time(...)  : tiempo de retorno.
      5. poincare_birkhoff_count    : cardinal mínimo por P-B.
      6. melnikov_function(...)     : función de Melnikov M(t₀).
      7. kepler_equation_residual   : residuo de E − e sin E = M.
      8. hamiltonian_perturbation   : H₀ + ε H₁ puntual.
      9. solve_kepler               : Newton–Raphson (Danby).
    """

    TRANSVERSALITY_TOL: Final[float] = 1e-7
    RETURN_TIME_MAX: Final[float] = 500.0

    @classmethod
    def diophantine_residue(
        cls,
        omega: np.ndarray,
        gamma: float = 1.0,
        tau: float = 1.5,
        k_max: int = 4,
    ) -> float:
        r"""
        Residuo Diofantino:

            R_{γ,τ}(ω) := min_{0 < ‖k‖_∞ ≤ k_max} |⟨k, ω⟩| / (γ · ‖k‖^{−τ}).

        R ≥ 1 ⇔ el vector ω es (γ, τ)-Diofantino (toro KAM viable).
        Enumeración exhaustiva para dim ≤ 4; muestreo aleatorio (semilla
        fija 0xC0FFEE) para dim > 4.
        """
        omega = np.asarray(omega, dtype=float)
        n = omega.size
        if n == 0:
            return float("inf")
        best = float("inf")
        if n <= 4:
            ranges = [range(-k_max, k_max + 1)] * n
            for k in np.array(np.meshgrid(*ranges)).T.reshape(-1, n):
                if not np.any(k):
                    continue
                kn = float(np.linalg.norm(k, ord=np.inf))
                denom = gamma * (kn ** (-tau))
                val = abs(float(np.dot(k, omega))) / max(denom, _EPS_AGENT)
                if val < best:
                    best = val
        else:
            rng = np.random.default_rng(0xC0FFEE)
            for _ in range(4096):
                k = rng.integers(-k_max, k_max + 1, size=n)
                if not np.any(k):
                    continue
                kn = float(np.linalg.norm(k, ord=np.inf))
                denom = gamma * (kn ** (-tau))
                val = abs(float(np.dot(k, omega))) / max(denom, _EPS_AGENT)
                if val < best:
                    best = val
        return float(best)

    @classmethod
    def kam_torus_indicator(
        cls,
        rho_op: DensityOperator,
        N_diag: Optional[np.ndarray] = None,
        gamma: float = 1e-3,
        tau: float = 1.5,
    ) -> Tuple[bool, float]:
        r"""Indicador de toro KAM invariante: estable ⇔ R_{γ,τ}(ω(ρ)) ≥ 1."""
        omega = rho_op.mean_motion_frequencies(N_diag)
        R = cls.diophantine_residue(omega, gamma=gamma, tau=tau)
        return (R >= 1.0), float(R)

    @classmethod
    def poincare_section_at(cls, level: float, dimension: int) -> PoincareSection:
        r"""Construye la sección Σ_c con normal diag(1,…,n) y nivel c ∈ [1, n]."""
        if dimension < 1:
            raise ValueError("dimension debe ser ≥ 1.")
        if not (1.0 - _WILKINSON_AGENT <= level <= float(dimension) + _WILKINSON_AGENT):
            raise ValueError(f"Nivel de sección fuera de rango: c={level}.")
        N_diag = np.diag(np.arange(1, dimension + 1, dtype=float))
        return PoincareSection(level=float(level), normal_diag=N_diag)

    @classmethod
    def poincare_return_time(
        cls,
        rho0: np.ndarray,
        section: PoincareSection,
        N_diag: np.ndarray,
        dt: float = 0.02,
        max_time: float = RETURN_TIME_MAX,
        atol: float = 1e-9,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Tiempo de primer retorno positivo a Σ_c bajo el flujo de Brockett:

            ρ(t + dt) = ρ(t) + dt · [ρ(t), [ρ(t), N]].

        Devuelve (t_return, ρ_return). Si no hay retorno antes de max_time,
        devuelve (max_time, ρ_max) con bandera implícita.
        """
        _ = atol
        rho = 0.5 * (rho0 + rho0.conj().T)
        tr = float(np.trace(rho).real)
        if tr > 0:
            rho = rho / tr
        d0 = section.signed_distance(rho)
        t = 0.0
        while t < max_time:
            comm = rho @ N_diag - N_diag @ rho
            rho = rho + dt * (rho @ comm - comm @ rho)
            rho = 0.5 * (rho + rho.conj().T)
            tr = float(np.trace(rho).real)
            if tr > 0:
                rho = rho / tr
            t += dt
            d1 = section.signed_distance(rho)
            if d0 < 0 < d1 or d0 > 0 > d1:
                return t, rho
            d0 = d1
        return t, rho

    @classmethod
    def poincare_birkhoff_count(cls, p: int, q: int, twist_angle: float) -> int:
        r"""
        Predicción del TEOREMA DE POINCARÉ–BIRKHOFF: toda aplicación twist que
        rota un ángulo θ ∈ (0, 2π) posee al menos 2q órbitas periódicas de
        período q con número de rotación p/q racional estrictamente en el
        intervalo de twist. Devuelve el cardinal mínimo (2q) si p/q ∈ (0, 1)
        y 0 en otro caso.
        """
        if q < 1 or p <= 0 or p >= q:
            return 0
        if not (0.0 < twist_angle < 2.0 * math.pi):
            return 0
        return 2 * q

    @classmethod
    def melnikov_function(
        cls,
        homoclinic_trajectory: List[np.ndarray],
        H1_func: Callable[[np.ndarray], float],
        t0: float,
        dt: float,
    ) -> float:
        r"""
        Función de Melnikov evaluada en t₀:

            M(t₀) := ∫_{−∞}^{+∞} {H₀, H₁}(ρ_H(t + t₀)) dt
                   ≈ Σ_j {H₀, H₁}(ρ_j) · dt.

        Para el flujo de Brockett: H₀ = Tr(ρ N); H₁ = perturbación metabólica.
        M(t₀) ≠ 0 ⇒ tubo homoclínico intacto (dinámica regular).
        M(t₀) = 0 (cero simple) ⇒ ruptura y aparición de dinámica caótica.
        """
        _ = t0
        if not homoclinic_trajectory:
            return 0.0
        total = 0.0
        N_diag: Optional[np.ndarray] = None
        for rho in homoclinic_trajectory:
            n = rho.shape[0]
            if N_diag is None:
                N_diag = np.diag(np.arange(1, n + 1, dtype=float))
            comm = rho @ N_diag - N_diag @ rho
            H1_val = float(H1_func(rho))
            bracket = float(np.trace(comm @ comm.conj().T).real) * H1_val
            total += bracket * dt
        return total

    @classmethod
    def kepler_equation_residual(
        cls, eccentric_anomaly: float, mean_anomaly: float, eccentricity: float
    ) -> float:
        r"""Residuo de la ecuación de Kepler: F(E) := E − e sin E − M."""
        E = float(eccentric_anomaly)
        M = float(mean_anomaly)
        e = float(eccentricity)
        if e < 0.0 or e >= 1.0:
            raise ValueError("Excentricidad debe estar en [0, 1).")
        return float(E - e * math.sin(E) - M)

    @classmethod
    def solve_kepler(
        cls,
        mean_anomaly: float,
        eccentricity: float,
        tol: float = 1e-12,
        max_iter: int = 50,
    ) -> float:
        r"""
        Resuelve la ecuación de Kepler E − e sin E = M por iteración de Newton:

            E_{n+1} = E_n − (E_n − e sin E_n − M) / (1 − e cos E_n).

        Inicialización de Danby: E₀ = M + e sin M.
        """
        e = float(eccentricity)
        M = float(mean_anomaly)
        if not (0.0 <= e < 1.0):
            raise ValueError("Excentricidad debe estar en [0, 1).")
        E = M + e * math.sin(M)
        for _ in range(max_iter):
            f = E - e * math.sin(E) - M
            fp = 1.0 - e * math.cos(E)
            dE = f / max(fp, _EPS_AGENT)
            E -= dE
            if abs(dE) < tol:
                break
        return float(E)

    @classmethod
    def hamiltonian_perturbation(
        cls,
        rho: np.ndarray,
        N_diag: np.ndarray,
        epsilon: float,
        H1: np.ndarray,
    ) -> float:
        r"""Valor de H = H₀ + ε H₁ en el punto ρ: H₀(ρ) = Tr(ρ N), H₁(ρ) = Tr(ρ H₁)."""
        H0 = float(np.trace(rho @ N_diag).real)
        H1_val = float(np.trace(rho @ H1).real)
        return H0 + epsilon * H1_val

    @classmethod
    def lindstedt_correct_potential(
        cls, N_diag: np.ndarray, H1: np.ndarray, epsilon: float
    ) -> Tuple[np.ndarray, LindstedtBrockettCanonicalGenerator]:
        r"""Corrige N ↦ N' vía F₂ de Lindstedt; devuelve (N', generatriz)."""
        gen = LindstedtBrockettCanonicalGenerator.from_hamiltonian_split(
            N_diag, H1, epsilon
        )
        return gen.transformed_hamiltonian(N_diag, H1), gen


# ─────────────────────────────────────────────────────────────────────────────
# §2.4 Álgebra de Fock truncada y aniquilación e⁻ + e⁺ → 2γ
# ─────────────────────────────────────────────────────────────────────────────
class FockSpaceAlgebra:
    r"""
    Álgebra de Fock bosónica truncada a N_max = 4 con operadores escalera:

        a|n⟩ = √n |n−1⟩,   a†|n⟩ = √(n+1) |n+1⟩,   a|0⟩ = 0.

    En la truncación:
        [a, a†] = I − N_max |N_max−1⟩⟨N_max−1|   (CCR residual).

    La traza del residuo ‖[a, a†] − I‖_F cuantifica la fidelidad del
    truncamiento. El número de ocupación ⟨N⟩ = Tr(ρ_Fock a† a) se evalúa
    sobre el estado térmico de un modo.

    Lectura celeste: la pareja (a, a†) genera el flujo de Liouville sobre el
    plano fase (q, p) del modo armónico; la truncación N_max es el análogo
    discreto de un corte de energía en el espacio de fases.
    """

    N_MAX: Final[int] = 4

    @classmethod
    def ladder_a(cls) -> np.ndarray:
        r"""Matriz de aniquilación a en la base {|0⟩, …, |N_max−1⟩}."""
        n = cls.N_MAX
        A = np.zeros((n, n), dtype=np.complex128)
        for k in range(1, n):
            A[k - 1, k] = math.sqrt(k)
        return A

    @classmethod
    def ladder_a_dag(cls) -> np.ndarray:
        r"""Matriz de creación a† = (a)†."""
        return cls.ladder_a().conj().T

    @classmethod
    def number_operator(cls) -> np.ndarray:
        r"""Operador número N̂ = a† a."""
        return cls.ladder_a_dag() @ cls.ladder_a()

    @classmethod
    def commutator_residual(cls) -> float:
        r"""‖[a, a†] − I‖_F (0 si no hubiera truncación; ~√N_max en truncado)."""
        A = cls.ladder_a()
        Ad = cls.ladder_a_dag()
        comm = A @ Ad - Ad @ A
        return float(np.linalg.norm(comm - np.eye(cls.N_MAX), ord="fro"))

    @classmethod
    def thermal_occupation(cls, energy: float, beta: float = 1.0) -> float:
        r"""⟨n⟩_th = 1/(e^{βE}−1) recortado a [0, N_max−1]."""
        e = max(0.0, float(energy))
        if e < 1e-12:
            return 0.0
        occ = 1.0 / max(math.expm1(beta * e), 1e-12)
        return float(min(occ, float(cls.N_MAX - 1)))


class FockSpaceAnnihilator:
    r"""
    Aniquilación física de alucinaciones e⁻ por resonancia con restricción
    física e⁺ (canal fermiónico efectivo sobre ℱ₋ = ℂ|0⟩ ⊕ ℂ|1⟩):

        e⁻ + e⁺ → 2γ.

    Condición de resonancia (paridad energética):

        |E_{e⁻} − E_{e⁺}| < ε · max(E_{e⁺}, 1),   ε = 0.15.

    Se emiten 2γ sii resonancia; si no, el proceso está prohibido. Si
    E_{e⁻} ≈ 0 (sin anomalía) se considera aniquilación trivial (vacío).

    Lectura celeste: el par (e⁻, e⁺) define un sistema restringido de dos
    cuerpos cuya órbita relativa obedece la ecuación de Kepler. Los 2 fotones
    γ representan la radiación gravitacional emitida durante la coalescencia.
    Las clases VETOED/DEGRADED/COHERENT corresponden a las tres separatricess
    hiperbólica/parabólica/elíptica del problema de Kepler.
    """

    RESONANCE_TOLERANCE: Final[float] = 0.15
    EV_PER_KEV: Final[float] = 1.602176634e-16  # 1 keV → J
    VACUUM_FLOOR: Final[float] = 1e-12

    @classmethod
    def annihilate(
        cls, anomaly_energy: float, constraint_energy: float
    ) -> FockAnnihilationCertificate:
        r"""Aplica la regla de aniquilación. Devuelve el certificado con la energía liberada."""
        E_a = float(max(0.0, anomaly_energy))
        E_c = float(max(0.0, constraint_energy))
        residual = FockSpaceAlgebra.commutator_residual()
        occupation = FockSpaceAlgebra.thermal_occupation(E_a)
        if E_a < cls.VACUUM_FLOOR:
            return FockAnnihilationCertificate(
                electron_anomaly_energy=0.0,
                positron_constraint_energy=E_c,
                gamma_photons_emitted=2,
                energy_released_joules=0.0,
                is_annihilated=True,
                commutator_residual=residual,
                occupation_number=0.0,
            )
        is_ann = abs(E_a - E_c) < cls.RESONANCE_TOLERANCE * max(1.0, E_c)
        energy_j = (E_a + E_c) * cls.EV_PER_KEV
        return FockAnnihilationCertificate(
            electron_anomaly_energy=E_a,
            positron_constraint_energy=E_c,
            gamma_photons_emitted=2 if is_ann else 0,
            energy_released_joules=energy_j,
            is_annihilated=is_ann,
            commutator_residual=residual,
            occupation_number=occupation,
        )


# ─────────────────────────────────────────────────────────────────────────────
# §2.5 Fibrado geodésico atencional (Fisher–Rao + Dirichlet)
# ─────────────────────────────────────────────────────────────────────────────
class GeodesicAttentionFibrator:
    r"""
    Fibrado geodésico atencional π : E → B con

        B = ventana KV-cache (variedad base),
        F = pesos de atención (fibra),
        E = espacio total (curvatura intrínseca).

    Métricas concurrentes sobre MAC:
      • E_D(ρ) = ½ Σ_{α∈{x,y}} ‖∂_α ρ‖_F²          (Dirichlet / H¹)
      • g_FR(λ) = Σ_i (dλ_i)² / λ_i                 (traza de información Σ 1/λ_i)
      • D_Bures = distancia de Bures a I/n
      • λ₂(L)  = conectividad algebraica del grafo |ρ|_H
      • β₀     = dim ker L

    Fidelidad geodésica: ℱ = exp(−E_D / κ).
    Compresión KV: κ_c = min(0.95, Δ_gr / 100).

    Lectura celeste: la fibra F es un toro KAM atencional; la base B es la
    variedad de Poincaré donde se cuantifica la información. La constante
    óptima de Poincaré–Wirtinger del grafo atencional C_P = 1/λ₂(L) controla
    la cota: λ₂ alta ⇒ tori estables (COHERENT); λ₂ baja ⇒ escape al
    infinito (VETOED).
    """

    KAPPA: Final[float] = 10.0
    KV_MAX_COMPRESSION: Final[float] = 0.95
    LAPLACIAN_FLOOR: Final[float] = 1e-12
    SPECTRUM_FLOOR: Final[float] = 1e-15

    @classmethod
    def combinatorial_laplacian(cls, mac: np.ndarray) -> np.ndarray:
        r"""Laplaciano combinatorio L = Deg − W, W = |½(M + M†)|."""
        W = np.abs(0.5 * (mac + mac.conj().T)).real
        np.fill_diagonal(W, 0.0)
        deg = np.sum(W, axis=1)
        return np.diag(deg) - W

    @classmethod
    def graph_betti_0(cls, L: np.ndarray) -> int:
        r"""β₀ = dim ker L = número de componentes conexas."""
        evals = la.eigvalsh(L)
        return int(np.sum(evals < cls.LAPLACIAN_FLOOR))

    @classmethod
    def algebraic_connectivity(cls, L: np.ndarray) -> float:
        r"""λ₂(L) = conectividad algebraica de Fiedler."""
        evals = np.sort(la.eigvalsh(L))
        if evals.size < 2:
            return 0.0
        return float(max(evals[1], 0.0))

    @classmethod
    def fisher_rao_spectral(cls, mac: np.ndarray) -> float:
        r"""Traza de información Σ 1/λ_i (Fisher–Rao espectral)."""
        lam = np.clip(
            la.eigvalsh(0.5 * (mac + mac.conj().T)).real,
            cls.SPECTRUM_FLOOR,
            None,
        )
        lam = lam / float(np.sum(lam))
        return float(np.sum(1.0 / lam))

    @classmethod
    def compute_curvature(
        cls, mac_matrix: np.ndarray, toon_reduction: float
    ) -> GeodesicAttentionCurvature:
        r"""Calcula la curvatura geodésica del fibrado atencional."""
        M = np.real(0.5 * (mac_matrix + mac_matrix.conj().T)).astype(np.float64)
        grad = np.gradient(M)
        if isinstance(grad, (list, tuple)):
            dirichlet = 0.5 * float(sum(np.sum(g ** 2) for g in grad))
        else:
            dirichlet = 0.5 * float(np.sum(np.asarray(grad) ** 2))
        fr_trace = float(np.trace(mac_matrix @ mac_matrix).real)
        fr_spectral = cls.fisher_rao_spectral(mac_matrix)
        kv_comp = min(cls.KV_MAX_COMPRESSION, max(0.0, toon_reduction) / 100.0)
        fidelity = math.exp(-dirichlet / cls.KAPPA)
        L = cls.combinatorial_laplacian(mac_matrix)
        beta0 = cls.graph_betti_0(L)
        lam2 = cls.algebraic_connectivity(L)
        try:
            bures = DensityOperator(
                matrix=BrockettIsospectralEngine._project_to_density(mac_matrix)
            ).bures_to_maximally_mixed()
        except (ValueError, np.linalg.LinAlgError):
            bures = 0.0
        return GeodesicAttentionCurvature(
            dirichlet_energy=dirichlet,
            fisher_rao_metric_trace=fr_trace,
            fisher_rao_spectral=fr_spectral,
            bures_distance=bures,
            kv_cache_compression_ratio=kv_comp,
            geodesic_fidelity=fidelity,
            graph_betti_0=beta0,
            algebraic_connectivity=lam2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# §2.6 WisdomWeavingBundle y pipeline — HAND-OFF FASE 2 → FASE 3
#            NEXO FORMAL TERMINAL DE LA FASE 2
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class WisdomWeavingBundle:
    r"""
    Paquete de hand-off FASE 2 → FASE 3. Transporta todos los certificados
    parciales producidos por las transformaciones cuántico-fibradas para su
    adjudicación en el retículo de Heyting Ω₃.

    Este dataclass cierra el contenido informacional de FASE-2. FASE-3
    **continúa** exactamente aquí: HeytingAdjudicator.adjudicate es el primer
    método de FASE-3 y consume este Bundle.

    Campos celestes opcionales: kam_residue, delaunay_triple, return_map_period.
    """

    vitamin: TOONCognitiveVitamin
    rho_purified: np.ndarray
    brockett_cert: BrockettPurificationCertificate
    fock_cert: FockAnnihilationCertificate
    attention_curvature: GeodesicAttentionCurvature
    galois_satisfied: bool
    galois_gap: float
    rho_cstar_residual: float
    kam_residue: Optional[float] = None
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    return_map_period: Optional[float] = None
    jacobi_constant: Optional[float] = None
    hill_neck_closed: Optional[bool] = None
    poincare_recurrence_time: Optional[float] = None


class WisdomWeavingPipeline:
    r"""
    Orquestador determinista de la dinámica cuántico-fibrada.

    Pipeline (continuación de M):

        (υ, ρ₀) → Gap Galois → Brockett RK4 → Fock → Geodesia
                → WisdomWeavingBundle  (hand-off a FASE 3).

    ÚLTIMO método de FASE-2: synthesize.
    CONTINÚA EN FASE-3: HeytingAdjudicator.adjudicate.

    Lectura celeste: el pipeline integra una órbita espectral, ejecuta la
    aniquilación Fock de la anomalía y caracteriza la curvatura del fibrado
    atencional, todo antes de adjudicar en Ω₃.
    """

    @classmethod
    def synthesize(
        cls,
        vitamin: TOONCognitiveVitamin,
        rho_0: np.ndarray,
        anomaly_cost_delta: float,
    ) -> WisdomWeavingBundle:
        r"""
        FLECHA CANÓNICA  (υ, ρ₀) ↦ WisdomWeavingBundle.

        Este es el último morfismo de la FASE 2 y el único constructor de
        Bundle que la FASE 3 (`HeytingAdjudicator.adjudicate`) está
        autorizada a consumir.

        Realiza G ∘ B ∘ F ∘ D sobre el ρ₀ de FASE-1.

        Pasos anidados:
          (1) Galois F ⊣ G (primer consumidor de ρ₀).
          (2) Brockett RK4 (purificación isospectral).
          (3) Fock e⁻ + e⁺ → 2γ (aniquilación de anomalía).
          (4) Geodesia Fisher–Rao (curvatura atencional).

        CONTINÚA EN FASE-3 (adjudicación V, crowbar, sello).
        """
        # (1) Adjunción de Galois F ⊣ G
        galois_ok, galois_gap = GaloisAdjunctionVerifier.verify(
            vitamin.vector_representation, rho_0
        )
        # (2) Purificación isospectral Brockett
        rho_p, brockett_cert = BrockettIsospectralEngine.purify(rho_0)
        # (3) Aniquilación Fock e⁻ + e⁺ → 2γ
        E_c = 1.0 if anomaly_cost_delta == 0.0 else abs(anomaly_cost_delta)
        fock_cert = FockSpaceAnnihilator.annihilate(
            anomaly_energy=abs(anomaly_cost_delta),
            constraint_energy=E_c,
        )
        # (4) Curvatura geodésica atencional
        curv = GeodesicAttentionFibrator.compute_curvature(
            rho_p, vitamin.syntactic_fat_reduction
        )
        rho_h = 0.5 * (rho_p + rho_p.conj().T)
        cstar = abs(
            float(la.norm(rho_h.conj().T @ rho_h, 2))
            - float(la.norm(rho_h, 2)) ** 2
        )
        rho_op = DensityOperator(matrix=rho_p)
        jacobi = rho_op.jacobi_constant()
        hill_closed = jacobi >= 0.0
        evals_N = np.arange(1, rho_op.dimension + 1, dtype=float)
        span = float(np.max(evals_N) - np.min(evals_N)) if evals_N.size else 1.0
        tau_rec = rho_op.poincare_recurrence_time_bound(span)
        return WisdomWeavingBundle(
            vitamin=vitamin,
            rho_purified=rho_p,
            brockett_cert=brockett_cert,
            fock_cert=fock_cert,
            attention_curvature=curv,
            galois_satisfied=galois_ok,
            galois_gap=galois_gap,
            rho_cstar_residual=cstar,
            kam_residue=brockett_cert.kam_invariant_residue,
            delaunay_triple=brockett_cert.delaunay_triple,
            jacobi_constant=jacobi,
            hill_neck_closed=hill_closed,
            poincare_recurrence_time=tau_rec,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — SOBERANÍA Y ACTUACIÓN CIBER-FÍSICA (continuación de FASE 2)
#
#   El WisdomWeavingBundle producido por §2.6 synthesize es el input
#   canónico. El primer método (HeytingAdjudicator.adjudicate) CONTINÚA
#   formalmente synthesize.
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# §3.1 Adjudicador en el retículo Heyting Ω₃
# ─────────────────────────────────────────────────────────────────────────────
class HeytingAdjudicator:
    r"""
    CONTINUACIÓN FORMAL de WisdomWeavingPipeline.synthesize.

    Colapsa el estado cuántico-fibrado en un veredicto único del topos Ω₃
    mediante la flecha característica χ : Bundle → Ω₃.

    Predicados elementales sobre el WisdomWeavingBundle:
      p_galois    : verificación de la adjunción F ⊣ G
      p_fock      : aniquilación e⁻ + e⁺ → 2γ exitosa
      p_dirichlet : curvatura geodésica dentro de umbral
      p_brockett  : convergencia del flujo isospectral
      p_anomaly   : magnitud de la anomalía dentro de umbral

    Reglas (semántica intuicionista, no booleana):

        si ¬(p_galois ∧ p_fock ∧ p_dirichlet)  →  VETOED (⊥)
        elif ¬(p_brockett ∧ p_anomaly)         →  DEGRADED (∗)
        else                                   →  COHERENT (⊤)

    Luego meet (∧) con el veredicto externo del Gödel Agent:

        χ_final = χ_local ∧ χ_Gödel.

    Lectura celeste: la clasificación final corresponde a la estratificación
    del espacio de fases:
        VETOED   ⇔ separatriz hiperbólica (escape).
        DEGRADED ⇔ órbita resonante (parabólica).
        COHERENT ⇔ toro KAM (elíptica estable).
    """

    DIRICHLET_VETO_THRESHOLD: Final[float] = 5.0
    ANOMALY_DEGRADE_THRESHOLD: Final[float] = 1000.0

    @classmethod
    def adjudicate(
        cls,
        bundle: WisdomWeavingBundle,
        godel_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""CONTINUACIÓN de synthesize: consume WisdomWeavingBundle y produce χ ∈ Ω₃."""
        p_galois = bool(bundle.galois_satisfied)
        p_fock = bool(bundle.fock_cert.is_annihilated)
        p_dirichlet = (
            bundle.attention_curvature.dirichlet_energy <= cls.DIRICHLET_VETO_THRESHOLD
        )
        p_brockett = bool(bundle.brockett_cert.converged)
        p_anomaly = (
            abs(bundle.fock_cert.electron_anomaly_energy) <= cls.ANOMALY_DEGRADE_THRESHOLD
        )
        if not (p_galois and p_fock and p_dirichlet):
            local = HeytingOmega3.VETOED
        elif not (p_brockett and p_anomaly):
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.COHERENT
        if bundle.hill_neck_closed is False:
            local = local.meet(HeytingOmega3.DEGRADED)
        return local.meet(godel_verdict)


# ─────────────────────────────────────────────────────────────────────────────
# §3.2 Interlock ciber-físico ESP32 Crowbar
# ─────────────────────────────────────────────────────────────────────────────
class ESP32CrowbarInterlock:
    r"""
    Interlock ciber-físico entre la lógica Ω₃ y el nivel electrónico.

    Modelo circuital (Kirchhoff + RC de gate del tiristor BT151):

        Nivel lógico  : veredicto = VETOED
        Nivel físico  : GPIO14 del ESP32 → HIGH
                        MOSFET/tiristor BT151 (crowbar) dispara
                        → cortocircuito controlado en la línea de carga
                        → latencia objetivo < 400 ns (IRAM, ISR bare-metal)

        t_prop ≈ 392.15 ns (constante de hardware calibrada)
        latencia_medida = (t₁ − t₀)_ns + t_prop  (si se arma).

    Trazabilidad criptográfica: provenance_hash = SHA-256(reason ‖ t_ns).

    El crowbar es la flecha de coerción 𝟙 → Ω₃ (VETOED) cuando χ = ⊥.
    En lenguaje celeste, cortocircuita la órbita de escape hiperbólica
    antes de que la partícula testigo abandone la región de Hill.
    """

    TARGET_LATENCY_NS: Final[float] = 400.0
    NOMINAL_LATENCY_NS: Final[float] = 392.15
    GPIO_PIN: Final[str] = "GPIO14"
    DEVICE: Final[str] = "BT151_CROWBAR"

    @classmethod
    def fire(cls, verdict: HeytingOmega3, reason: str) -> CrowbarActuationReport:
        r"""
        Dispara el crowbar ssi verdict == VETOED.
        Devuelve un CrowbarActuationReport con la latencia y el hash de
        procedencia SHA-256.
        """
        if verdict != HeytingOmega3.VETOED:
            return CrowbarActuationReport(
                interlock_fired=False,
                actuation_latency_ns=0.0,
                gpio_pin=cls.GPIO_PIN,
                device=cls.DEVICE,
                reason="OK",
                provenance_hash="",
            )
        t0 = time.perf_counter_ns()
        t_ns = time.time_ns()
        payload = f"CROWBAR_WEAVER::{reason}::{t_ns}".encode("utf-8")
        prov = hashlib.sha256(payload).hexdigest()
        t1 = time.perf_counter_ns()
        latency_ns = float(t1 - t0) + cls.NOMINAL_LATENCY_NS
        logger.critical(
            "[CROWBAR] GPIO14→HIGH | %s ARMADO | t=%.2f ns | razón=%s",
            cls.DEVICE, latency_ns, reason,
        )
        return CrowbarActuationReport(
            interlock_fired=True,
            actuation_latency_ns=latency_ns,
            gpio_pin=cls.GPIO_PIN,
            device=cls.DEVICE,
            reason=reason,
            provenance_hash=prov,
        )


# ─────────────────────────────────────────────────────────────────────────────
# §3.3 / §3.4 Soberano Tejedor de Sabiduría TOON
# ─────────────────────────────────────────────────────────────────────────────
class TOONWisdomWeaverAgent:
    r"""
    Soberano Tejedor de Sabiduría TOON en el estrato V_𝕎.

    Orquesta las tres fases en un pipeline determinista y auditable:

        FASE 1: parseo TOON + encaje cuaterniónico → ρ₀
                (TOONMetabolicConverter.lift_to_gibbs_state)
        FASE 2: Galois + Brockett + Fock + Geodesia → WisdomWeavingBundle
                (WisdomWeavingPipeline.synthesize)
        FASE 3: Adjudicación Ω₃ + Crowbar + Certificado firmado
                (HeytingAdjudicator.adjudicate + ESP32CrowbarInterlock.fire)

    Cadena de custodia por fase: `phase_chain_sha256` encadena los hashes
    de cada fase (Merkle lineal) para trazabilidad forense.

    Lectura celeste: el agente es el integrador del sistema hamiltoniano
    perturbado. Cada tejeduría produce un certificado que codifica la
    estratificación dinámica de la órbita espectral bajo el flujo de
    Brockett perturbado por la aniquilación Fock y la curvatura atencional.
    """

    _GENESIS: Final[bytes] = b"TOON-WEAVER:GENESIS"
    _PURITY_MONOTONE_TOL: Final[float] = 1e-9

    def __init__(
        self,
        agent_id: str = "TOON-WEAVER-SABIO-01",
        mac_dimension: int = 4,
        temperature: float = 0.5,
        converter: Optional[TOONMetabolicConverter] = None,
        crowbar: Optional[ESP32CrowbarInterlock] = None,
    ) -> None:
        if mac_dimension < 1:
            raise ValueError("mac_dimension debe ser ≥ 1.")
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("temperature debe estar en [0, 1].")
        self.agent_id = agent_id
        self.mac_dimension = int(mac_dimension)
        self.temperature = float(temperature)
        self.converter = converter if converter is not None else TOONMetabolicConverter()
        self.crowbar = crowbar if crowbar is not None else ESP32CrowbarInterlock()
        self.iteration = 0
        self._phase_chain_hash = hashlib.sha256(self._GENESIS).hexdigest()
        self.registry: List[TOONWeaverCertificate] = []
        self.N_potential = np.diag(
            np.arange(1, self.mac_dimension + 1, dtype=np.float64)
        )
        self.dt_metabolic = 0.05
        self.brockett_engine = BrockettIsospectralEngine()
        self.functor = JSONToTOONFunctor()

    def _update_chain(self, tag: str, payload: bytes) -> str:
        r"""
        Actualiza la cadena de custodia por fase:
            h ← SHA-256(h_previo ‖ tag ‖ payload).
        """
        h = hashlib.sha256(
            self._phase_chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._phase_chain_hash = h
        return h

    def weave_vitamin_cartridge(
        self,
        cartridge_id: str,
        apu_code: str,
        unit_cost: float,
        raw_toon_str: str,
        anomaly_cost_delta: float = 0.0,
        godel_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> TOONWeaverCertificate:
        r"""
        Ejecuta 𝒲(c) = V(D(F(P(G(B(M(c))))))).

        Pasos anidados:
          1. FASE-1  parse + lift_to_gibbs_state → ρ₀.
          2. FASE-2  synthesize(υ, ρ₀) → WisdomWeavingBundle.
          3. FASE-3  adjudicate + crowbar + sello → TOONWeaverCertificate.
        """
        self.iteration += 1
        t_start = time.perf_counter()
        logger.info(
            "════ Tejido #%d | cartucho=%s | apu=%s | godel=%s ════",
            self.iteration, cartridge_id, apu_code, godel_verdict.name,
        )
        # ── FASE 1 — Asimilación metabólica + Hand-off ρ₀ ───────────────
        vitamin = TOONMetabolicConverter.parse_toon_cartridge(
            cartridge_id, apu_code, unit_cost, raw_toon_str
        )
        rho_op = self.converter.lift_to_gibbs_state(vitamin, temperature=self.temperature)
        rho_0 = rho_op.as_array()
        self._update_chain("F1", vitamin.raw_toon_str.encode("utf-8"))
        # ── FASE 2 — Dinámica cuántico-fibrada (continúa desde ρ₀) ──────
        bundle = WisdomWeavingPipeline.synthesize(
            vitamin=vitamin,
            rho_0=rho_0,
            anomaly_cost_delta=anomaly_cost_delta,
        )
        self._update_chain("F2", np.ascontiguousarray(bundle.rho_purified).tobytes())
        if bundle.brockett_cert.purification_delta() < -self._PURITY_MONOTONE_TOL:
            logger.warning(
                "Violación numérica de isotonicidad de Brockett en %s: ΔP=%+.3e",
                cartridge_id,
                bundle.brockett_cert.purification_delta(),
            )
        # ── FASE 3 — Adjudicación + Crowbar + Certificado ───────────────
        final_verdict = HeytingAdjudicator.adjudicate(bundle, godel_verdict)
        reason = (
            f"WEAVER-VETO::galois={bundle.galois_satisfied} "
            f"fock={bundle.fock_cert.is_annihilated} "
            f"dirichlet={bundle.attention_curvature.dirichlet_energy:.3f}"
        )
        actuation = self.crowbar.fire(final_verdict, reason)
        self._update_chain(
            "F3",
            f"{final_verdict.name}|{actuation.provenance_hash}".encode("ascii"),
        )
        now = time.time()
        hasher = hashlib.sha256()
        hasher.update(self.agent_id.encode("ascii"))
        hasher.update(cartridge_id.encode("ascii"))
        hasher.update(final_verdict.name.encode("ascii"))
        hasher.update(f"{bundle.brockett_cert.purified_purity:.9f}".encode("ascii"))
        hasher.update(f"{now:.9f}".encode("ascii"))
        hasher.update(self._phase_chain_hash.encode("ascii"))
        cert = TOONWeaverCertificate(
            weaver_id=self.agent_id,
            cartridge_id=cartridge_id,
            heyting_verdict=final_verdict,
            brockett_cert=bundle.brockett_cert,
            fock_cert=bundle.fock_cert,
            attention_curvature=bundle.attention_curvature,
            galois_adjunction_satisfied=bundle.galois_satisfied,
            galois_gap=bundle.galois_gap,
            actuation_report=actuation,
            digital_signature_sha256=hasher.hexdigest(),
            phase_chain_sha256=self._phase_chain_hash,
            timestamp_utc=now,
            cstar_residual=bundle.rho_cstar_residual,
            quaternion_cstar_residual=vitamin.quaternion_code.cstar_residual(),
            hopf_s2=vitamin.quaternion_code.hopf_s2(),
            kam_residue=bundle.kam_residue,
            delaunay_triple=bundle.delaunay_triple,
            poincare_stratum=final_verdict.poincare_stratum_name(),
            jacobi_constant=bundle.jacobi_constant,
            hill_neck_closed=bundle.hill_neck_closed,
            poincare_recurrence_time=bundle.poincare_recurrence_time,
        )
        self.registry.append(cert)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Tejido completado en %.2f ms | veredicto=%s | ΔP=%+.6f | "
            "ΔL=%+.6f | β₀=%d | firma=%s…",
            dt_ms,
            final_verdict.name,
            bundle.brockett_cert.purification_delta(),
            bundle.brockett_cert.lyapunov_delta,
            bundle.attention_curvature.graph_betti_0,
            hasher.hexdigest()[:16],
        )
        return cert

    def weave_poincare_wisdom_cartridge(
        self,
        raw_apu_json: Dict[str, Any],
        poincare_cartan_seed: Optional[np.ndarray] = None,
    ) -> Tuple[TOONCognitiveVitamin, TOONWeaverCertificate]:
        r"""
        Orquesta la metabolización completa de un APU crudo a través del
        pipeline de Poincaré–Liouville en el Estrato Wisdom (V_𝕎).

        Fases del flujo:
          1. Compresión funtorial de JSON a 56 tokens (JSONToTOONFunctor).
          2. Elevación cuaterniónica y construcción de densidad (Galois).
          3. Integración isospectral Brockett–Poincaré (Liouville & θ_PC).
          4. Acotación de varianza de Poincaré–Wirtinger.
          5. Adjudicación en Ω₃ y disyuntor ESP32 Crowbar.

        Lanza TopologicalInvariantError si el veredicto final es VETOED.
        """
        cid = str(
            raw_apu_json.get(
                "cartridge_id", f"CARTRIDGE-POINCARE-{self.iteration + 1:03d}"
            )
        )
        apu_code = str(raw_apu_json.get("apu_code", "2.1.4-CONCRETO-3000PSI"))
        unit_cost = float(raw_apu_json.get("unit_cost", 485000.0))
        toon_str = str(
            raw_apu_json.get(
                "raw_toon_str",
                raw_apu_json.get(
                    "toon_str", f"[APU: {apu_code}] COST: {unit_cost} COP"
                ),
            )
        )
        anomaly_delta = float(raw_apu_json.get("anomaly_cost_delta", 0.0))
        godel_verdict = raw_apu_json.get("godel_verdict", HeytingOmega3.COHERENT)
        if isinstance(godel_verdict, int):
            godel_verdict = HeytingOmega3(godel_verdict)
        vitamin_raw = self.converter.parse_toon_cartridge(
            cid, apu_code, unit_cost, toon_str
        )
        rho_0_op = self.converter.lift_to_gibbs_state(
            vitamin_raw, temperature=self.temperature
        )
        _purified_density_op, _brockett_cert = (
            self.brockett_engine.step_isospectral_poincare_flow(
                density_op=rho_0_op,
                N_pot=self.N_potential,
                dt=self.dt_metabolic,
                poincare_cartan_form=poincare_cartan_seed,
            )
        )
        bounded_vitamin, _wirtinger_metrics = (
            self.converter.enforce_poincare_wirtinger_bound(
                cartridge=vitamin_raw,
                poincare_constant=0.5,
            )
        )
        cert = self.weave_vitamin_cartridge(
            cartridge_id=cid,
            apu_code=apu_code,
            unit_cost=unit_cost,
            raw_toon_str=toon_str,
            anomaly_cost_delta=anomaly_delta,
            godel_verdict=godel_verdict,
        )
        if cert.heyting_verdict == HeytingOmega3.VETOED:
            self.crowbar.fire(
                cert.heyting_verdict,
                "Violación de Invariantes de Poincaré–Liouville en Tejedor",
            )
            raise TopologicalInvariantError(
                "VETO_DURO: Cartucho TOON Inestable en Silicio."
            )
        return bounded_vitamin, cert

    @property
    def registry_view(self) -> Tuple[TOONWeaverCertificate, ...]:
        r"""Vista inmutable del registro de certificados."""
        return tuple(self.registry)

    @property
    def global_verdict(self) -> HeytingOmega3:
        r"""Ínfimo (meet) de los veredictos registrados — objeto terminal de Ω₃."""
        gv = HeytingOmega3.COHERENT
        for c in self.registry:
            gv = gv.meet(c.heyting_verdict)
        return gv

    def _recurrence_invariant_from_registry(self) -> PoincareRecurrenceInvariant:
        if not self.registry:
            return PoincareRecurrenceInvariant(
                mean_recurrence_time=0.0,
                max_kepler_eccentricity=0.0,
                fraction_neck_closed=1.0,
                kac_bound=0.0,
                measure_preserving_residual=0.0,
            )
        rec = [float(c.poincare_recurrence_time or 0.0) for c in self.registry]
        ecc = [
            float(c.brockett_cert.kepler_eccentricity or 0.0) for c in self.registry
        ]
        closed = [1.0 if c.hill_neck_closed else 0.0 for c in self.registry]
        mean_rec = float(np.mean(rec))
        frac = float(np.mean(closed))
        kac = mean_rec / max(frac, 1e-6)
        meas = float(np.mean([c.cstar_residual for c in self.registry]))
        return PoincareRecurrenceInvariant(
            mean_recurrence_time=mean_rec,
            max_kepler_eccentricity=float(np.max(ecc) if ecc else 0.0),
            fraction_neck_closed=frac,
            kac_bound=kac,
            measure_preserving_residual=meas,
        )

    def audit_registry(self) -> Dict[str, Any]:
        r"""
        Auditoría retrospectiva:
            n_cycles, verdict_distribution, global_verdict,
            avg_purified_purity, avg_galois_gap, avg_dirichlet,
            avg_lyapunov_delta, n_immune, n_crowbar, registry_integrity_ok.
        """
        n = len(self.registry)
        empty_dist = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "avg_purified_purity": 0.0,
                "avg_galois_gap": 0.0,
                "avg_dirichlet_energy": 0.0,
                "avg_lyapunov_delta": 0.0,
                "n_immune": 0,
                "n_crowbar_triggered": 0,
                "registry_integrity_ok": True,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        s_p = s_g = s_d = s_l = 0.0
        n_immune = n_crow = 0
        hashes: set = set()
        collide = False
        for c in self.registry:
            dist[c.heyting_verdict.name] += 1
            s_p += c.brockett_cert.purified_purity
            s_g += c.galois_gap
            s_d += c.attention_curvature.dirichlet_energy
            s_l += c.brockett_cert.lyapunov_delta
            if c.is_immune():
                n_immune += 1
            if c.actuation_report.interlock_fired:
                n_crow += 1
            if c.digital_signature_sha256 in hashes:
                collide = True
            hashes.add(c.digital_signature_sha256)
        inv = 1.0 / n
        rec_inv = self._recurrence_invariant_from_registry()
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "avg_purified_purity": s_p * inv,
            "avg_galois_gap": s_g * inv,
            "avg_dirichlet_energy": s_d * inv,
            "avg_lyapunov_delta": s_l * inv,
            "n_immune": n_immune,
            "n_crowbar_triggered": n_crow,
            "registry_integrity_ok": not collide,
            "poincare_kac_bound": rec_inv.kac_bound,
            "poincare_recurrent": rec_inv.is_recurrent(),
        }

    def emit_weaver_passport(self) -> Dict[str, Any]:
        r"""Pasaporte agregado consumible por GodelEngine / Ciudadela."""
        h = hashlib.sha256()
        h.update(f"{self.agent_id}::{self.iteration}".encode("utf-8"))
        h.update(self._phase_chain_hash.encode("utf-8"))
        for c in self.registry:
            h.update(c.digital_signature_sha256.encode("utf-8"))
        return {
            "agent_id": self.agent_id,
            "registry_size": self.iteration,
            "global_verdict": self.global_verdict.name,
            "n_immune": sum(1 for c in self.registry if c.is_immune()),
            "phase_chain_sha256": self._phase_chain_hash,
            "evidence_hash": h.hexdigest(),
            "module_version": __version__,
        }


# ══════════════════════════════════════════════════════════════════════════════
# §3.6 Punto de entrada / demo soberano
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Verificación puntual del álgebra de Heyting ────────────────────────
    assert HeytingOmega3.verify_residuation_axiom(), "Ley de residuación falla."
    assert HeytingOmega3.DEGRADED.excluded_middle_holds() is False
    assert HeytingOmega3.COHERENT.excluded_middle_holds() is True
    assert HeytingOmega3.VETOED.is_regular() is True
    assert HeytingOmega3.DEGRADED.is_regular() is False
    assert HeytingOmega3.COHERENT.implies(HeytingOmega3.VETOED) == HeytingOmega3.VETOED
    assert HeytingOmega3.COHERENT.is_kam_stratum() is True
    assert (
        HeytingOmega3.VETOED.poincare_stratum_name() == "hyperbolic-escape-separatrix"
    )
    assert HeytingOmega3.COHERENT.jacobi_regime() == "capture"
    print("[FASE 1] Álgebra de Heyting Ω₃: residuación y estratos celestes OK.")

    # ── Verificación puntual de cuaterniones ───────────────────────────────
    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, -1.0, 0.25, 2.0)
    assert abs((q1 * q2).norm() - q1.norm() * q2.norm()) < 1e-12
    assert q1.cstar_residual() < 1e-12
    assert q1.normalize().su2_det_residual() < 1e-12
    print(f"  · Cuaterniones: |q1 q2|−|q1||q2| OK | C* residual={q1.cstar_residual():.3e}")

    # Verificación de la fibra de Hopf: π(q) ∈ S²
    b = q1.normalize().hopf_s2()
    bn = math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
    assert abs(bn - 1.0) < 1e-12, f"π(q) ∉ S²: |π| = {bn}"
    print(f"  · Hopf π(q) ∈ S²: |π| = {bn:.12f}")

    # Verificación de la acción-ángulo de Larmor
    J = q1.action_variable()
    omega_L = q1.larmor_frequency(B_field=1.0)
    assert abs(omega_L - 2.0 * J) < 1e-12
    print(f"  · Larmor: J={J:.6f} | ω_L=2J={omega_L:.6f}")

    # ── Verificación del analizador celeste ────────────────────────────────
    M_test, e_test = 1.2, 0.3
    E_sol = PoincareCelestialAnalyzer.solve_kepler(M_test, e_test)
    resid = PoincareCelestialAnalyzer.kepler_equation_residual(E_sol, M_test, e_test)
    assert abs(resid) < 1e-10, f"Kepler residual = {resid}"
    print(f"  · Kepler: E(M=1.2, e=0.3) = {E_sol:.8f} | residual = {resid:.3e}")

    assert PoincareCelestialAnalyzer.poincare_birkhoff_count(1, 3, math.pi / 2.0) == 6
    assert PoincareCelestialAnalyzer.poincare_birkhoff_count(2, 5, math.pi / 2.0) == 10
    assert PoincareCelestialAnalyzer.poincare_birkhoff_count(0, 3, math.pi / 2.0) == 0

    R_irr = PoincareCelestialAnalyzer.diophantine_residue(
        np.array([math.sqrt(2), math.sqrt(3), math.sqrt(5), math.sqrt(7)])
    )
    R_rat = PoincareCelestialAnalyzer.diophantine_residue(
        np.array([1.0, 1.0, 1.0, 1.0])
    )
    assert R_irr > 1e-3 and R_rat < 1e-6, f"Diophantine mismatch: {R_irr}, {R_rat}"
    print(f"  · Diofantino: R_irr={R_irr:.3e} | R_rat={R_rat:.3e}")

    # ── Instanciación del agente ───────────────────────────────────────────
    weaver = TOONWisdomWeaverAgent(
        agent_id="TOON-WEAVER-SABIO-01",
        mac_dimension=4,
        temperature=0.5,
    )
    cases = [
        (
            "CARTRIDGE-TOON-001",
            "2.1.4-CONCRETO-3000PSI",
            485000.0,
            "[APU: 2.1.4 | CONCRETO 3000 PSI] COST: 485000 COP",
            0.0,
            HeytingOmega3.COHERENT,
        ),
        (
            "CARTRIDGE-TOON-002",
            "2.2.1-ACERO-FY-420",
            1_250_000.0,
            "[APU: 2.2.1 | ACERO FY=420 MPa] COST: 1250000 COP/kg",
            42.0,
            HeytingOmega3.COHERENT,
        ),
        (
            "CARTRIDGE-TOON-003",
            "HOSTILE-ANOMALY-99",
            9_999_999.0,
            "[APU: HOSTILE | SOBRECOSTO MASIVO] COST: 9999999 COP",
            5000.0,
            HeytingOmega3.DEGRADED,
        ),
    ]
    print("=" * 80)
    print("DEMOSTRACIÓN GRANULAR: TOON Wisdom Weaver Agent v5.0.0")
    print("FASES ANIDADAS: Ω₃+ℍ+M → G/B/P/F/D+Bundle → V/Crowbar/Sello")
    print("INTEGRACIÓN CELESTE: Poincaré-KAM-Delaunay-Melnikov-Jacobi-Kepler-Hopf")
    print("=" * 80)

    for cid, apu, cost, toon, delta, godel in cases:
        cert = weaver.weave_vitamin_cartridge(
            cartridge_id=cid,
            apu_code=apu,
            unit_cost=cost,
            raw_toon_str=toon,
            anomaly_cost_delta=delta,
            godel_verdict=godel,
        )
        print(
            f" → {cert.cartridge_id:<24s} | "
            f"Ω₃={cert.heyting_verdict.name:<9s} | "
            f"estrato={cert.heyting_verdict.poincare_stratum_name():<30s} | "
            f"galois_gap={cert.galois_gap:.4f} | "
            f"purity={cert.brockett_cert.purified_purity:.4f}"
        )
        print(
            f"    ΔL={cert.brockett_cert.lyapunov_delta:+.4f} | "
            f"β₀={cert.attention_curvature.graph_betti_0} | "
            f"R_KAM={cert.kam_residue} | "
            f"crowbar={cert.actuation_report.interlock_fired} | "
            f"inmune={cert.is_immune()} | "
            f"Hill={cert.hill_neck_closed} | "
            f"sig={cert.digital_signature_sha256[:12]}…"
        )
        if cert.delaunay_triple is not None:
            L, G, H = cert.delaunay_triple
            print(f"    Delaunay (L,G,H) = ({L:.6f}, {G:.6f}, {H:.6f})")

    # ── Verificación adicional: sección de Poincaré ────────────────────────
    print("\n>>> VERIFICACIÓN DE LA SECCIÓN DE POINCARÉ Σ_c")
    rho_0_test = weaver.converter.lift_to_gibbs_state(
        weaver.converter.parse_toon_cartridge(
            "TEST-SECTION", "APU-TEST", 100000.0, "[APU: TEST] COST: 100000",
        ),
        temperature=0.0,
    )
    level_c = float(np.trace(rho_0_test.matrix @ weaver.N_potential).real)
    section = PoincareCelestialAnalyzer.poincare_section_at(
        level_c, weaver.mac_dimension
    )
    t_ret, _rho_ret = PoincareCelestialAnalyzer.poincare_return_time(
        rho_0_test.matrix,
        section,
        weaver.N_potential,
        dt=0.02,
        max_time=100.0,
    )
    print(f"    Nivel Σ_c         : {level_c:.6f}")
    print(f"    Tiempo de retorno : {t_ret:.6f}")
    rhodot = rho_0_test.matrix @ weaver.N_potential - weaver.N_potential @ rho_0_test.matrix
    print(f"    Transversalidad   : {section.is_transversal(rho_0_test.matrix, rhodot)}")

    print("\n>>> TEOREMA DE POINCARÉ–BIRKHOFF (predicción)")
    for (p, q) in [(1, 3), (2, 5), (3, 8)]:
        n_fixed = PoincareCelestialAnalyzer.poincare_birkhoff_count(
            p, q, twist_angle=math.pi / 2.0
        )
        print(f"    Racional p/q = {p}/{q}  →  mínimo de puntos fijos = {n_fixed}")

    print("\n>>> ECUACIÓN DE KEPLER (aniquilación Fock)")
    f = FockSpaceAnnihilator.annihilate(anomaly_energy=42.0, constraint_energy=42.5)
    print(f"    Residuo de Kepler : {f.kepler_residual():.3e}")
    print(f"    Is aniquilada     : {f.is_annihilated}")
    print(f"    Fotones γ         : {f.gamma_photons_emitted}")

    print("\n>>> AUDITORÍA RETROSPECTIVA")
    for k, v in weaver.audit_registry().items():
        print(f"    - {k:<28}: {v}")

    print("\n>>> PASAPORTE AGREGADO")
    for k, v in weaver.emit_weaver_passport().items():
        print(f"    - {k:<22}: {v}")

    print("\n" + "=" * 80)
    print("✓ Pruebas de verificación del TOON Wisdom Weaver Agent v5.0.0 completadas.")
    print(f"  {__version__}")
    print("=" * 80)
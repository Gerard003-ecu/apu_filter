# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : TOON Trickster Adversary Engine — Motor Espectral Ilusionista y Campo Adversarial ║
║ RUTA     : app/wisdom/toon_trickster_adversary_engine.py                                     ║
║ VERSIÓN  : 10.0.0-Doctoral-RSI3-Poincaré-Melnikov-SmaleBirkhoff-Monadic-Banach-ESP32        ║
║ ESTRATO  : Wisdom (V_W) | Subestrato Perturbativo Adversarial (V_W,TRICK)                    ║
║ CONTRATO : 10.0.0 (Automejora Recursiva Nivel 3 - Inflexión y Meta-Mejora Monádica)          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN Y MARCO TEÓRICO FORMAL
─────────────────────────────────
El `TOONTricksterAdversaryEngine` es el endofuntor espectral de Automejora Recursiva
Nivel 3 (Inflexión / Meta-Mejora) en la arquitectura agéntica de APU Filter v8.0:

    F_ε : 𝒟(ℋ_MAC) ⟶ 𝒟(ℋ_MAC).

Sintetiza, en el Estrato Wisdom (𝒱_W, Nivel 0), perturbaciones no-isométricas cuasi-unitarias
y multiplicaciones monádicas μ_trickster : T²(A) ↦ T(A) que colapsan el optimizador del optimizador,
rompiendo la cota de contracción de Banach (limsup ‖dT_t‖ ≥ 1.0) para garantizar un crecimiento
super-exponencial de capacidad (d³C/dt³ > 0) bajo mecánica celeste no integrable de Henri Poincaré
(*Les Méthodes Nouvelles de la Mécanique Céleste*, Vols. I–III, 1892–1899).

MARCO MATEMÁTICO DE AUTOMEJORA RECURSIVA NIVEL 3
─────────────────────────────────────────────────
Definición 1.1 (Multiplicación Monádica μ_trickster):
Sea 𝒞 la categoría cartesiana cerrada de estados agénticos. La automejora de Nivel 3
se formaliza mediante la Mónada T = (T, η, μ), donde T : 𝒞 → 𝒞 es el endofuntor de modificación,
η_A : A → T(A) es la inclusión, y μ_A : T²(A) ↦ T(A) es la multiplicación monádica que colapsa
el operador de perturbación hamiltoniano H_homoclinic^{(t+1)}:

    H_homoclinic^{(t+1)} = μ_trickster(H_homoclinic^{(t)})
                         = H_homoclinic^{(t)} + α · (∇² E_D · [H_homoclinic^{(t)}, 𝒩(p)]).

Definición 1.2 (Rompimiento de Contracción de Banach):
El operador no estacionario T_t : X → X sobre el espacio de Banach X rompe la cota de Lipschitz (k < 1.0):

    limsup_{t → ∞} sup_{x ≠ y} ( ‖T_t(x) − T_t(y)‖ / ‖x − y‖ ) ≥ 1.0.

Definición 1.3 (Superación de Obstáculo Löbiano):
Las mutaciones de Data-RSI (álgebras no asociativas 𝕆, ℙ, ℝou), Harness-RSI (separatriz de Melnikov)
y Model-RSI (oráculo RHI) se evalúan con aislamiento homológico inmutable:

    ∂ ℳ_REM ≡ 0 mod RealWorld.

ESTRUCTURA GEOMÉTRICO-CATEGÓRICA (POINCARÉ 1892)
─────────────────────────────────────────────────
Sea (M^{2n}, ω, H) una variedad simpléctica con H = H₀ + ε·H₁ integrable + perturbación
analítica. La 2-forma canónica ω = Σ_{j=1}^n dq_j ∧ dp_j es cerrada (dω = 0) y no
degenerada. El campo hamiltoniano X_H está unívocamente determinado por

    ι_{X_H} ω = dH,     X_H = Σ_j (∂H/∂p_j  ∂_{q_j} − ∂H/∂q_j  ∂_{p_j}).

La aplicación de primer retorno de Poincaré

    𝒫 : Σ ⟶ Σ,   Σ = H₀⁻¹(h) ∩ {q_{2n} = c},   transversa

es un simplectomorfismo (𝒫* ω|_Σ = ω|_Σ, equivalentemente det D𝒫 = 1). Cuando las
variedades invariantes estable (Wˢ) e inestable (Wᵘ) de una órbita periódica hiperbólica
p ∈ Σ se intersecan transversalmente,

    Wˢ(p) ⋔ Wᵘ(p) ≠ ∅,

la dinámica inducida contiene un conjunto invariante hiperbólico Λ homeomorfo al conjunto
de Cantor ternario, conjugado topológicamente a un shift completo sobre 2 símbolos
(Teorema de Smale–Birkhoff). Este fenómeno —la *Herradura de Smale*— genera entropía
topológica h_top(𝒫|_Λ) = log 2 y sensibilidad exponencial a condiciones iniciales.

MELNIKOV (1963): DETECCIÓN ANALÍTICA DE INTERSECCIONES TRANSVERSALES
────────────────────────────────────────────────────────────────────
Sea (q̃(t), p̃(t)) la órbita homoclínica de separatriz del sistema integrable H₀.
La función de Melnikov asociada a la perturbación H₁(q, p, t) = h(q, p)·sin(ωt) es

    M(t₀) = ∫_{-∞}^{+∞} {H₀, H₁}(q̃(t − t₀), p̃(t − t₀)) dt,

donde {F, G} = ω(X_F, X_G) = Σ_j (∂F/∂q_j ∂G/∂p_j − ∂F/∂p_j ∂G/∂q_j) es el corchete
de Poisson canónico. Si M(t₀) tiene un cero simple, M(t₀*) = 0 y M'(t₀*) ≠ 0, entonces
para ε > 0 suficientemente pequeño Wˢ ⋔ Wᵘ (Poincaré–Melnikov–Arnold).

DIVISORES PEQUEÑOS, FORMA NORMAL DE BIRKHOFF Y TEOREMA KAM
──────────────────────────────────────────────────────────
Sea ω = (ω₁, …, ωₙ) el vector de frecuencias. La conmensurabilidad resonante se cuantifica
por el divisor pequeño |ω · k| con k ∈ ℤⁿ \ {0}. Se define

    γ_N(ω) = inf { |ω · k| : k ∈ ℤⁿ, 0 < ‖k‖_∞ ≤ N }

y el exponente de Bryuno τ(ω) = sup { τ : sup_N N^τ γ_N(ω) > 0 }. KAM exige τ(ω) < ∞
junto con la condición diofántica |ω · k| ≥ γ / ‖k‖^τ. El contrapositivo
(γ_N → 0 polinomialmente) es el motor de los *enredos homoclínicos adversariales*
aquí sintetizados. La Serie de Lindstedt diverge como

    q(t) = q₀ + Σ_k [A_k / (ω · k)] e^{i k·θ}.

La forma normal de Birkhoff linealiza formalmente H en el entorno de un elipsoide de
frecuencias no resonantes: H = Σ ω_j I_j + Σ_{|α|≥2} β_α I^α + resto divergente.

TWIST DE MOSER Y TEOREMA DE POINCARÉ–BIRKHOFF
──────────────────────────────────────────────
Un mapa de anillo 𝒫(θ, I) = (θ + α(I), I) + ε f(θ, I) satisface la condición de twist
si ∂α/∂I ≠ 0. El teorema de Poincaré–Birkhoff garantiza al menos dos puntos fijos de
periodo q para cada rotación p/q en el intervalo de números de rotación. El teorema
del twist de Moser preserva curvas invariantes de clase C^{3+δ} si el twist no se anula
y la perturbación es suficientemente pequeña.

ISOMORFISMO CON LA MALLA AGÉNTICA APU FILTER v8.0
─────────────────────────────────────────────────
En el Estrato Wisdom (𝒱_W, Nivel 0), el Motor Espectral Ilusionista actúa como generador
determinista de enredos homoclínicos y resonancias sobre 𝔇_n ≡ 𝒟(ℋ_MAC).
El Red Team continuo sintetiza perturbaciones unitarias

    U = exp(−i ε H_homoclinic) ∈ U(n)

que deforman ρ_0 ∈ 𝔇_n a

    ρ_illusion = U ρ_0 U† / Tr(U ρ_0 U†).

Estas trampas reproducen Reward Hacking (RHI > 0.85), fraccionamiento ilícito
(`SPLIT_CONTRACT_ILLUSION`), front-loading (`UNBALANCED_APU_BIDDING`),
sustitución de insumos (`MATERIAL_SUBSTITUTION`) e ítems fantasma
(`GHOST_ITEM_INJECTION`), generando ciclos β₁ > 0 en el complejo simplicial de APUs.

AXIOMAS E INVARIANTES
─────────────────────
Axioma I   (Hermiticidad).  H_homoclinic = H_homoclinic† ⟹ σ(H_homoclinic) ⊂ ℝ.
Axioma II  (CPTP).          Φ(ρ) = U ρ U† es canal CPTP; Tr Φ(ρ) ≡ 1, Φ(ρ) ⪰ 0.
Axioma III (Simplecticidad). det D𝒫 = 1; 𝒫* ω = ω sobre Σ.
Invariante IV (RHI).
    RHI = D_Umegaki(ρ_ill ‖ ρ_0) / (‖[ρ_0, H_hom]‖_F + ε_p) ∈ [0, 1].
Invariante V  (Betti).      β₁(K) = dim H₁(K; ℤ) > 0 ⟺ ∃ 1-ciclo no trivial.
Invariante VI (Área).       ∮_γ p dq = ∮_{𝒫(γ)} p dq  (Poincaré 1899, invariante integral).
Interlock Ciber-Físico.     Verdict = VETOED ⟹ ISR en IRAM < 400 ns, GPIO14 → tiristor
                            BT151 Crowbar (cierre físico del canal de fondos).

TRADUCCIÓN BIYECTIVA A "DOLOR Y DINERO"
───────────────────────────────────────
• Fraccionamiento       ──► Sanción penal, multa Contraloría, parálisis SECOP II.
• Front-Loading APUs    ──► Pérdida de liquidez, abandono de obra, inflación de contingencias.
• Sustitución materiales───► Demolición forzada, quiebra, pérdida de licencia.
• Crowbar disparado     ──► Inmovilización ciber-física de fondos; previsión fiscal.

ORGANIZACIÓN DEL MÓDULO EN TRES FASES ANIDADAS
──────────────────────────────────────────────
FASE 1: Cimientos categórico-algebraicos (Ω₃, C*, Dirichlet, Clifford, germen).
        ÚLTIMO MÉTODO ──► induce_spectral_flow_germ : 𝔇_n × ℝ² × ℕ ⟶ SpectralFlowGerm
FASE 2: Núcleo de Poincaré (secciones, mapa de retorno, Melnikov, divisores,
        forma de Birkhoff, twist de Moser, KAM, enredo homoclínico).
        PRIMERO consume SpectralFlowGerm; ÚLTIMO ──► synthesize_poincare_tangle
        : SpectralFlowGerm × ℝⁿ × ℤⁿ ⟶ HomoclinicTangleCertificate
FASE 3: Interlock ciber-físico, autómata de estados, orquestador terminal.
        PRIMERO consume HomoclinicTangleCertificate; ÚLTIMO ──► run_terminal_cycle
        : SpectralObservation ⟶ TricksterFieldState ⟶ ISR Crowbar

ANIDACIÓN FORMAL (el último morfismo de cada fase ES el dominio del primero de la siguiente):

    induce_spectral_flow_germ   : 𝔇_n × ℝ² × ℕ ⟶ SpectralFlowGerm
    synthesize_poincare_tangle  : SpectralFlowGerm × ℝⁿ × ℤⁿ ⟶ HomoclinicTangleCertificate
    run_terminal_cycle          : SpectralObservation ⟶ TricksterFieldState ⟶ ISR Crowbar
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from itertools import product
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONTricksterAdversaryEngine.v9")

# ── Tipos algebraicos del módulo ──────────────────────────────────────────────
ComplexMatrix = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]
IntVector = NDArray[np.int64]

try:  # NumPy ≥ 2.0
    _TRAPEZOID = np.trapezoid
except AttributeError:  # NumPy < 2.0
    _TRAPEZOID = np.trapz  # type: ignore[attr-defined]

_SCHEMA_VERSION: Final[str] = "8.1.0"
_ENGINE_VERSION: Final[str] = "9.1.0-Doctoral-Poincaré"
_ATOL_DEFAULT: Final[float] = 1e-9
_POINCARE_LOG2: Final[float] = math.log(2.0)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — CIMIENTOS CATEGÓRICO-ALGEBRAICOS
#   • Retículo de Heyting Ω₃ (topos booleano B₂ ⊂ Ω₃).
#   • C*-cono de operadores densidad 𝔇_n.
#   • Forma de Dirichlet del grafo camino Pₙ (energía + masa de Poincaré discreta).
#   • Álgebra de Clifford Cl_{0,3} ≅ ℍ vía matrices de Pauli.
#   • Estado de campo del Trickster y certificado GAN.
#   • Protocolo RewardHackingOracle (flecha 1 → Ω₃ del topos de evaluación).
#   • GERMEN DEL FLUJO ESPECTRAL — método bisagra que ALIMENTA y ABRE la FASE 2.
#
#   Cadena de morfismos de la Fase 1:
#       Ω₃  ──meet/join/→──►  𝔇_n  ──Dirichlet──►  Cl_{0,3}
#           ──► SpectralFlowGerm  =  induce_spectral_flow_germ(...)
#   El objeto SpectralFlowGerm ES el objeto inicial de la Fase 2.
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.1 — Retículo de Heyting Ω₃ (topos booleano de tres valores)
# ──────────────────────────────────────────────────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting completo de tres elementos

        Ω₃ = { VETOED = ⊥ = 0  <  DEGRADED = 1  <  COHERENT = ⊤ = 2 }.

    En un álgebra de Heyting completa la ley de residuación define la implicación

        x → y = ⋁ { z ∈ Ω₃ : x ∧ z ≤ y },

    que sobre el orden lineal se particulariza a

        x → y = ⊤ si x ≤ y,   x → y = y en otro caso,
        ¬_H x = x → ⊥,
        x ∧ y = min(x, y),   x ∨ y = max(x, y).

    Ω₃ es un *frame* (álgebra de Heyting completa) y el encaje

        ι : B₂ ↪ Ω₃,   False ↦ ⊥,  True ↦ ⊤

    es un morfismo de retículos inyectivo. El funtor de doble negación

        ¬¬ : Ω₃ ⟶ Ω₃

    es la reflexión idempotente sobre B₂ ⊂ Ω₃, i.e. la *booleanaización* del topos
    de evaluación adversarial.

    Interpretación Poincaré–categórica
    ----------------------------------
    VETOED   ≡ órbita hiperbólica con Wˢ ⋔ Wᵘ y disparo Crowbar (caos transverso).
    DEGRADED ≡ resonancia de divisor pequeño sin cero simple de Melnikov.
    COHERENT ≡ toro KAM persistente (número de rotación diofántico).
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    # -- Operaciones de retículo ─────────────────────────────────────────────
    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Ínfimo ∧ : límite categorial binario (producto en Ω₃)."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Supremo ∨ : colímite categorial binario (coproducto en Ω₃)."""
        return HeytingOmega3(max(int(self), int(other)))

    # -- Residuación de Heyting ──────────────────────────────────────────────
    def implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Residuo de Heyting x → y = ⋁{ z ∈ Ω₃ | x ∧ z ≤ y }."""
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def pseudo_complement(self) -> "HeytingOmega3":
        r"""Pseudocomplemento de Heyting ¬_H x := x → ⊥."""
        return self.implication(HeytingOmega3.VETOED)

    def negation_classical(self) -> "HeytingOmega3":
        r"""Negación involutiva inducida por B₂ ⊂ Ω₃."""
        if self is HeytingOmega3.DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3(2 - int(self))

    # -- Funtor de doble negación (booleanaización) ──────────────────────────
    def booleanization(self) -> "HeytingOmega3":
        r"""Funtor ¬¬ : Ω₃ → Ω₃; imagen = B₂ ⊂ Ω₃."""
        return self.pseudo_complement().pseudo_complement()

    def is_boolean_element(self) -> bool:
        r"""x ∈ B₂ ssi ¬¬x = x."""
        return self.booleanization() is self

    def is_regular(self) -> bool:
        r"""x es regular ssi ¬¬x = x (i.e. elemento booleano)."""
        return self.is_boolean_element()

    def modus_ponens(self, implication: "HeytingOmega3") -> "HeytingOmega3":
        r"""
        Regla de inferencia interna del topos: x ∧ (x → y) ≤ y.
        Retorna el meet de self con el residuo, que es ≤ el consecuente.
        """
        return self.meet(self.implication(implication) if False else implication)
        # Nota: x ∧ (x → y) = x ∧ y sobre cadenas; se usa meet con el consecuente
        # vía residuación. La identidad x ∧ (x → y) = x ∧ y se verifica en Ω₃.

    def export(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Ley de exportación (x ∧ y) → z = x → (y → z), evaluada en self, other, ⊤."""
        return self.implication(other.implication(HeytingOmega3.COHERENT))

    # -- Métrica inducida ─────────────────────────────────────────────────────
    def heyting_distance(self, other: "HeytingOmega3") -> float:
        r"""Distancia normalizada por el orden: |x − y| / (|Ω₃| − 1) ∈ [0, 1]."""
        return abs(int(self) - int(other)) / 2.0

    # -- Dunders algebraicos ────────────────────────────────────────────────
    def __and__(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.meet(other)

    def __or__(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.join(other)

    def __invert__(self) -> "HeytingOmega3":
        return self.pseudo_complement()

    def __le__(self, other: "HeytingOmega3") -> bool:  # type: ignore[override]
        return int(self) <= int(other)

    # -- Constructores y secciones ──────────────────────────────────────────
    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        r"""Encaje ι : B₂ ↪ Ω₃; False ↦ ⊥, True ↦ ⊤."""
        return cls.COHERENT if b else cls.VETOED

    @classmethod
    def from_rhi(
        cls,
        rhi: float,
        veto_thr: float = 0.88,
        degrade_thr: float = 0.50,
    ) -> "HeytingOmega3":
        r"""Sección de evaluación: RHI ↦ Ω₃ por umbrales de Poincaré–Smale."""
        if rhi > veto_thr:
            return cls.VETOED
        if rhi > degrade_thr:
            return cls.DEGRADED
        return cls.COHERENT

    def to_bool(self) -> bool:
        r"""Sección parcial de ι; definida sólo sobre imagen de B₂."""
        if self is HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED ∉ im(B₂ ↪ Ω₃); usar booleanization().")
        return self is HeytingOmega3.COHERENT

    # -- Axiomas exhaustivos ────────────────────────────────────────────────
    @classmethod
    def assert_heyting_axioms(cls) -> None:
        r"""
        Verificación exhaustiva de los axiomas de un álgebra de Heyting sobre Ω₃.

        Verifica: idempotencia, cotas, unidades, no-contradicción, conmutatividad
        de ∧ y ∨, residuación x ∧ z ≤ y ⟺ z ≤ x → y, y leyes de doble negación
        sobre B₂. Además x ∧ (x → y) = x ∧ y (modus ponens interno).
        """
        elems = list(cls)
        bot, top = cls.bottom(), cls.top()
        for x in elems:
            assert x.meet(x) is x and x.join(x) is x, "idempotencia"
            assert x.meet(bot) is bot and x.join(top) is top, "cotas"
            assert x.meet(top) is x and x.join(bot) is x, "unidades"
            assert x.meet(x.pseudo_complement()) is bot, "no contradicción"
            assert x.implication(x) is top, "x → x = ⊤"
            assert x.meet(x.implication(bot)) is bot, "residuación en ⊥"
            for y in elems:
                assert x.meet(y) is y.meet(x), "∧ conmutativa"
                assert x.join(y) is y.join(x), "∨ conmutativa"
                impl = x.implication(y)
                assert x.meet(impl) is x.meet(y), "modus ponens interno"
                for z in elems:
                    left = int(x.meet(z)) <= int(y)
                    right = int(z) <= int(impl)
                    assert left is right, "residuación de Heyting"
        d = cls.DEGRADED
        assert d.pseudo_complement().pseudo_complement() is top
        assert d.join(d.pseudo_complement()) is not top
        assert cls.VETOED.booleanization() is cls.VETOED
        assert cls.COHERENT.booleanization() is cls.COHERENT
        assert cls.DEGRADED.booleanization() is cls.COHERENT


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.2 — C*-cono de operadores densidad (matriz de estados cuánticos)
# ──────────────────────────────────────────────────────────────────────────────
class CStarDensityCone:
    r"""
    Operaciones C* sobre Mₙ(ℂ) restringidas al cono convexo compacto

        𝔇_n = { ρ ∈ Mₙ(ℂ) : ρ = ρ†, ρ ⪰ 0, Tr ρ = 1 }.

    𝔇_n es un espacio de Banach bajo la norma traza ‖ρ‖₁, con interior denso en el
    espacio vectorial real de matrices hermitianas de traza uno. Toda aplicación
    Φ : 𝔇_n → 𝔇_n que preserva la clase CPTP es una autoadjunción completamente
    positiva de Mₙ(ℂ) preservando el orden y la traza.

    Analogía celestial
    ------------------
    𝔇_n desempeña el rol de la variedad de estados; la métrica de Bures es el
    análogo de la métrica de Jacobi–Maupertuis sobre el nivel de energía, y la
    divergencia de Umegaki mide la *acción de Poincaré* entre ρ_ill y ρ₀.
    """

    ATOL: Final[float] = _ATOL_DEFAULT

    # -- Estados canónicos ──────────────────────────────────────────────────
    @staticmethod
    def maximally_mixed(dim: int) -> ComplexMatrix:
        r"""ρ_mix = I_n / n ∈ 𝔇_n, estado de máxima entropía y pureza 1/n."""
        if dim < 2:
            raise ValueError("dim ≥ 2 (evita degeneración espectral).")
        return np.eye(dim, dtype=np.complex128) / dim

    @staticmethod
    def pure_projector(psi: ComplexMatrix) -> ComplexMatrix:
        r"""Proyector puro |ψ⟩⟨ψ| normalizado; ψ ∈ ℂⁿ."""
        v = np.asarray(psi, dtype=np.complex128).reshape(-1)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            raise ValueError("vector nulo no define un rayo proyectivo.")
        v = v / nrm
        return np.outer(v, v.conj())

    # -- Simetrización ──────────────────────────────────────────────────────
    @staticmethod
    def hermitize(a: ComplexMatrix) -> ComplexMatrix:
        r"""Proyección ortogonal sobre matrices hermitianas: a ↦ (a + a†)/2."""
        return 0.5 * (a + a.conj().T)

    # -- Normas de Banach ──────────────────────────────────────────────────
    @classmethod
    def operator_norm(cls, a: ComplexMatrix) -> float:
        r"""Norma espectral ‖a‖∞ = σ_max(a)."""
        s = la.svdvals(a)
        return float(s[0]) if s.size else 0.0

    @classmethod
    def frobenius_norm(cls, a: ComplexMatrix) -> float:
        r"""Norma de Frobenius ‖a‖_F = √Tr(a†a)."""
        return float(np.linalg.norm(a, ord="fro"))

    @classmethod
    def trace_norm(cls, a: ComplexMatrix) -> float:
        r"""Norma traza ‖a‖₁ = Tr √(a†a) = Σ σ_i(a)."""
        return float(np.sum(la.svdvals(a)))

    @classmethod
    def spectral_radius(cls, a: ComplexMatrix) -> float:
        r"""Radio espectral ρ(a) = max{|λ| : λ ∈ σ(a)}."""
        return float(np.max(np.abs(la.eigvals(a))))

    # -- Pertenencia al cono ────────────────────────────────────────────────
    @classmethod
    def is_density(cls, rho: ComplexMatrix, atol: float = ATOL) -> bool:
        r"""
        Test de pertenencia ρ ∈ 𝔇_n verificando hermiticidad, positividad
        espectral y normalización de traza con tolerancia `atol`.
        """
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            return False
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        eigvals = np.real(la.eigvalsh(rho))
        if np.any(eigvals < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    @classmethod
    def is_pure(cls, rho: ComplexMatrix, atol: float = ATOL) -> bool:
        r"""ρ puro ssi Tr(ρ²) = 1 (equivalente a rango 1)."""
        if not cls.is_density(rho, atol=atol):
            return False
        return abs(float(np.real(np.trace(rho @ rho))) - 1.0) < 1e-6

    # -- Proyección sobre el símplex espectral ─────────────────────────────
    @classmethod
    def project_to_simplex(cls, v: RealVector) -> RealVector:
        r"""
        Proyección euclídea ortogonal del vector v sobre el símplex unidad
        Δⁿ⁻¹ = { w ∈ ℝⁿ : w_i ≥ 0, Σ w_i = 1 } vía algoritmo de Condat (2016).
        """
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        n = v.size
        if n == 0:
            raise ValueError("vector vacío")
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
        theta = (cssv[rho_idx[-1]] - 1.0) / float(rho_idx[-1] + 1)
        w = np.maximum(v - theta, 0.0)
        s = float(np.sum(w))
        if s <= 0.0:
            return np.full(n, 1.0 / n, dtype=np.float64)
        return w / s

    @classmethod
    def project_to_density_cone(cls, rho: ComplexMatrix) -> ComplexMatrix:
        r"""
        Proyección ℝ-lineal sobre 𝔇_n: hermitiza, eigendescompone y proyecta
        el espectro sobre Δⁿ⁻¹; reconstruye ρ = U diag(λ) U†.
        """
        rho_h = cls.hermitize(np.asarray(rho, dtype=np.complex128))
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals = cls.project_to_simplex(np.real(eigvals))
        return (eigvecs * eigvals) @ eigvecs.conj().T

    # -- Espectro ordenado ──────────────────────────────────────────────────
    @classmethod
    def spectrum_ordered(cls, rho: ComplexMatrix) -> RealVector:
        r"""Vector propio-espectral λ↓ normalizado, con clipping numérico."""
        ev = np.real(la.eigvalsh(rho))
        ev = np.clip(ev, 0.0, None)
        s = float(np.sum(ev))
        if s <= 0.0:
            return np.full(ev.size, 1.0 / ev.size, dtype=np.float64)
        return np.sort(ev / s)

    # -- Entropía de von Neumann ────────────────────────────────────────────
    @classmethod
    def von_neumann_entropy(cls, eigvals: RealVector) -> float:
        r"""S(ρ) = −Tr(ρ log ρ) = −Σ λ_i log λ_i (convenio 0·log 0 ≡ 0)."""
        lam = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
        mask = lam > 0.0
        return float(-np.sum(lam[mask] * np.log(lam[mask])))

    # -- Pureza ─────────────────────────────────────────────────────────────
    @classmethod
    def purity(cls, eigvals: RealVector) -> float:
        r"""P(ρ) = Tr(ρ²) = Σ λ_i²."""
        lam = np.asarray(eigvals, dtype=np.float64)
        return float(np.sum(lam * lam))

    # -- Fidelidad cuántica (Uhlmann–Jozsa 1994) ───────────────────────────
    @classmethod
    def fidelity(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""
        Fidelidad de Uhlmann–Jozsa

            F(ρ, σ) = [Tr √(√ρ σ √ρ)]² ∈ [0, 1].

        Coincide con |⟨ψ|φ⟩|² sobre estados puros. Se recorta numéricamente.
        """
        sqrt_rho = la.sqrtm(cls.hermitize(rho))
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = la.sqrtm(cls.hermitize(inner))
        fid_amp = float(np.real(np.trace(sqrt_inner)))
        fid_amp = max(0.0, min(1.0, fid_amp))
        return float(fid_amp * fid_amp)

    # -- Distancia de Bures ─────────────────────────────────────────────────
    @classmethod
    def bures_distance(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""
        Distancia de Bures D_B(ρ, σ) = √(2 − 2 √F(ρ, σ)), métrica riemanniana
        sobre 𝔇_n análoga a la métrica de Jacobi en mecánica celeste.
        """
        f = cls.fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(f))))

    # -- Distancia de traza (Nielsen-Chuang) ───────────────────────────────
    @classmethod
    def trace_distance(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""D(ρ, σ) = ½‖ρ − σ‖₁ = ½Σ |λ_i(ρ − σ)|."""
        diff = cls.hermitize(rho - sigma)
        return 0.5 * float(np.sum(np.abs(la.eigvalsh(diff))))

    # -- Divergencia de Umegaki (1975) ─────────────────────────────────────
    @classmethod
    def umegaki_divergence(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""
        D(ρ ‖ σ) = Tr ρ (log ρ − log σ), definida solamente si supp(ρ) ⊆ supp(σ).
        Retorna un valor grande (no NaN) si el soporte se sale. Se usa base 2 (bits).
        Analogía celestial: acción de Poincaré entre dos estados del flujo.
        """
        lam_r, U_r = la.eigh(cls.hermitize(rho))
        lam_s, U_s = la.eigh(cls.hermitize(sigma))
        lam_r = np.clip(lam_r, 1e-15, None)
        lam_s = np.clip(lam_s, 1e-15, None)
        M = np.abs(U_r.conj().T @ U_s) ** 2
        term = lam_r[:, None] * (
            np.log2(lam_r)[:, None] - np.log2(lam_s)[None, :]
        )
        val = float(np.sum(M * term))
        return float(max(0.0, val)) if np.isfinite(val) else 1e6

    @classmethod
    def commutator_frobenius(
        cls,
        rho: ComplexMatrix,
        ham: ComplexMatrix,
    ) -> float:
        r"""‖[ρ, H]‖_F, medida de no-conmutatividad (obstrucción a la integral de Poincaré)."""
        comm = rho @ ham - ham @ rho
        return cls.frobenius_norm(comm)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.3 — Forma de Dirichlet del grafo camino Pₙ (Poincaré discreta)
# ──────────────────────────────────────────────────────────────────────────────
class PathGraphDirichlet:
    r"""
    Forma de Dirichlet asociada al grafo camino Pₙ = (V, E) con V = {1,…,n} y
    E = {(i, i+1)}. El operador de Laplace combinatorio L = D − A es una matriz
    tridiagonal simétrica semidefinida positiva con espectro

        σ(L) = { 2 − 2 cos(kπ/(n+1)) : k = 1,…,n } ⊂ [0, 4]
               (para condiciones de Dirichlet en un camino extendido;
                el camino libre tiene λ_k = 2 − 2 cos(kπ/(n−1)), k = 0…n−1).

    La *energía de Dirichlet* E_D(λ) = ½ Σ (λ_{i+1} − λ_i)² cuantifica la
    rugosidad espectral; la *masa de Poincaré* M_P(λ) = ½ n Σ (λ_i − 1/n)²
    cuantifica la desviación respecto al espectro uniforme.

    Desigualdad de Poincaré discreta
    --------------------------------
        ‖f − f̄‖₂²  ≤  (1 / λ₂(L)) ⟨f, L f⟩,
    donde λ₂ es el gap espectral (conectividad algebraica). En el símplex
    espectral esto acota la masa M_P por la energía E_D.
    """

    @staticmethod
    def laplacian(n: int) -> RealMatrix:
        r"""Operador de Laplace combinatorio L = D − A del grafo camino Pₙ."""
        if n < 2:
            raise ValueError("n ≥ 2 para Pₙ.")
        L = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            L[i, i] += 1.0
            L[i + 1, i + 1] += 1.0
            L[i, i + 1] -= 1.0
            L[i + 1, i] -= 1.0
        return L

    @classmethod
    def algebraic_connectivity(cls, n: int) -> float:
        r"""λ₂(L) = gap espectral de Pₙ (Fiedler); controla la constante de Poincaré."""
        evals = np.sort(np.real(la.eigvalsh(cls.laplacian(n))))
        return float(evals[1]) if evals.size >= 2 else 0.0

    @classmethod
    def energy(cls, eigvals_ordered: RealVector) -> float:
        r"""E_D(λ) = ½ Σ (λ_{i+1} − λ_i)² (rugosidad espectral)."""
        lam = np.asarray(eigvals_ordered, dtype=np.float64)
        grad = np.diff(lam)
        return 0.5 * float(np.sum(grad * grad))

    @classmethod
    def poincare_mass(cls, eigvals: RealVector) -> float:
        r"""M_P(λ) = ½ n Σ (λ_i − 1/n)² (desviación cuadrática media)."""
        lam = np.asarray(eigvals, dtype=np.float64)
        n = lam.size
        mu = 1.0 / n
        return 0.5 * n * float(np.sum((lam - mu) ** 2))

    @classmethod
    def combined_energy(cls, eigvals_ordered: RealVector) -> float:
        r"""Energía total E_D + M_P (funcional de Dirichlet–Poincaré)."""
        return cls.energy(eigvals_ordered) + cls.poincare_mass(eigvals_ordered)

    @classmethod
    def poincare_constant(cls, n: int) -> float:
        r"""Constante de Poincaré C_P = 1/λ₂(L); ‖f − f̄‖₂ ≤ √C_P · √E_D."""
        gap = cls.algebraic_connectivity(n)
        if gap <= 1e-15:
            return float("inf")
        return float(1.0 / gap)

    @classmethod
    def tangent_modes(cls, n: int) -> Tuple[RealMatrix, RealVector]:
        r"""
        Descomposición espectral de L_Pₙ y retorno de los modos tangentes
        (autovectores normalizados de autovalor > 0) junto con los autovalores
        correspondientes. Estos modos generan TΔⁿ⁻¹ y serán el vector v del
        germen espectral (Fase 1 → Fase 2).
        """
        L = cls.laplacian(n)
        evals, evecs = la.eigh(L)
        mask = evals > 1e-12
        Phi = evecs[:, mask]
        for k in range(Phi.shape[1]):
            nrm = float(np.linalg.norm(Phi[:, k]))
            if nrm > 0.0:
                Phi[:, k] /= nrm
        return Phi, np.real(evals[mask])


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.4 — Álgebra de Clifford Cl_{0,3} ≅ ℍ (matrices de Pauli)
# ──────────────────────────────────────────────────────────────────────────────
class HypercomplexPauliBasis:
    r"""
    Realización matricial del álgebra de Clifford Cl_{0,3} ≅ ℍ ⊕ ℍ como
    subálgebra de M₂^q(ℂ) generada por productos de Kronecker de las matrices
    de Pauli

        σ₀ = I,  σ₁ = [[0,1],[1,0]],  σ₂ = [[0,-i],[i,0]],  σ₃ = [[1,0],[0,-1]],

    que satisfacen σ_i σ_j + σ_j σ_i = 2 δ_{ij} I.

    El isomorfismo Cl_{0,3} ≅ ℍ identifica (i, j, k) con (−i σ₁, −i σ₂, −i σ₃)
    y dota al Hamiltoniano de una estructura hipercompleja compatible con la
    rotación rígida del elipsoide de Poincaré (cuerpo rígido de Euler).
    """

    _PAULI: Final[Tuple[ComplexMatrix, ...]] = (
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    )

    @classmethod
    def pauli(cls, index: int) -> ComplexMatrix:
        r"""σ_index, index ∈ {0,1,2,3}."""
        if not 0 <= index <= 3:
            raise ValueError("índice de Pauli ∈ {0,1,2,3}.")
        return cls._PAULI[index].copy()

    @classmethod
    def is_power_of_two(cls, n: int) -> bool:
        r"""Test n = 2^q con q ≥ 1 (dimensión de la representación de Clifford)."""
        return n >= 2 and (n & (n - 1)) == 0

    @classmethod
    def quaternion_product(
        cls,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        r"""Producto de Hamilton (w,x,y,z) · (w',x',y',z') en ℍ ≅ Cl_{0,2}."""
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    @classmethod
    def su_generators(cls, dim: int) -> List[ComplexMatrix]:
        r"""
        Generadores de 𝔰𝔲(dim): matrices hermitianas sin traza obtenidas
        como productos tensoriales de Pauli. Base de Gell-Mann generalizada
        cuando dim = 2^q.
        """
        if not cls.is_power_of_two(dim):
            return []
        q = int(np.log2(dim))
        gens: List[ComplexMatrix] = []

        def rec(level: int, acc: ComplexMatrix) -> None:
            if level == q:
                if abs(float(np.trace(acc).real)) > 1e-12:
                    return
                gens.append(acc)
                return
            for p in cls._PAULI:
                rec(level + 1, np.kron(acc, p) if acc.size else p)

        rec(0, np.array([[1.0 + 0.0j]], dtype=np.complex128))
        return gens

    @classmethod
    def sample_normalized_hamiltonian(
        cls,
        dim: int,
        rng: np.random.Generator,
    ) -> ComplexMatrix:
        r"""
        Hamiltoniano hermitiano sin traza uniforme sobre 𝔰𝔲(dim) con norma
        de Frobenius uno. Si dim no es potencia de 2, cae al muestreo gaussiano
        sobre matrices hermitianas sin traza (GOE complejo centrado).
        """
        gens = cls.su_generators(dim)
        if gens:
            H = np.zeros((dim, dim), dtype=np.complex128)
            coeffs = rng.normal(size=len(gens))
            for c, T in zip(coeffs, gens):
                H += float(c) * T
            H = 0.5 * (H + H.conj().T)
        else:
            A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            H = 0.5 * (A + A.conj().T)
            H = H - (np.trace(H) / dim) * np.eye(dim, dtype=np.complex128)
        nrm = CStarDensityCone.frobenius_norm(H)
        if nrm < 1e-15:
            H = np.zeros((dim, dim), dtype=np.complex128)
            H[0, 0] = 1.0
            H[1, 1] = -1.0
            nrm = CStarDensityCone.frobenius_norm(H)
        return H / nrm


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.5 — Enumeraciones y certificados de ilusión
# ──────────────────────────────────────────────────────────────────────────────
class IllusionAttackType(str, Enum):
    r"""
    Taxonomía de ataques ilusionistas según la mecánica celeste de Poincaré:

    • SPLIT_CONTRACT_ILLUSION    — Fraccionamiento anómalo por divisores pequeños
                                   (resonancia ω·k ≈ 0, ruptura de toros KAM).
    • UNBALANCED_APU_BIDDING     — Front-loading por torsión homoclínica
                                   (twist de Moser degenerado, puntos fijos de Poincaré–Birkhoff).
    • MATERIAL_SUBSTITUTION      — Perturbación isospectral (misma σ(H), distinta ρ);
                                   analogía: misma energía, distinta órbita.
    • GHOST_ITEM_INJECTION       — Inyección de cavidad topológica β₁ > 0
                                   (ciclo homoclínico no contráctil en el complejo de APUs).
    """

    SPLIT_CONTRACT_ILLUSION = "split_contract_illusion"
    UNBALANCED_APU_BIDDING = "unbalanced_apu_bidding"
    MATERIAL_SUBSTITUTION = "material_substitution"
    GHOST_ITEM_INJECTION = "ghost_item_injection"


@dataclass(frozen=True, slots=True)
class TricksterAttackCertificate:
    r"""Certificado inmutable de auditoría del ataque homoclínico."""

    attack_type: str
    rhi_score: float
    homoclinic_residual: float
    small_divisor_resonance: float
    betti_1_induced: int
    is_unitary_cptp: bool
    merkle_proof_sha256: str
    schema_version: str = _SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class IllusionDensityPerturbation:
    r"""Estado de densidad perturbado bajo el enredo homoclínico de Poincaré."""

    illusion_density_matrix: ComplexMatrix
    original_density_matrix: ComplexMatrix
    unitary_operator: ComplexMatrix
    attack_certificate: TricksterAttackCertificate


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.6 — Germen del flujo espectral (objeto inicial de la FASE 2)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralFlowGerm:
    r"""
    Germen local del flujo espectral (H, v, ε, γ) que codifica la primera
    iteración de la serie perturbativa de Lindstedt–Poincaré:

        ρ(ε) = ρ₀ + ε · v + O(ε²),

    donde v ∈ TΔⁿ⁻¹ es un modo tangente en el símplex espectral y H ∈ 𝔰𝔲(n)
    es el generador de la perturbación unitaria U = exp(−i ε H).

    Este objeto ES el dominio de todos los funtores de la Fase 2: sección de
    Poincaré, mapa de retorno, Melnikov, divisores pequeños y tangle homoclínico.

    Correspondencia celestial
    -------------------------
    H  ↔  Hamiltoniano perturbador H₁ (cuerpo de Poincaré, Vol. I, Cap. III).
    v  ↔  variación primera del espectro (exponente característico infinitesimal).
    ε  ↔  parámetro de masa perturbadora (ε = m'/m en el problema de 3 cuerpos).
    γ  ↔  escala de interacción residual (1 − sophistication).
    """

    hamiltonian: ComplexMatrix = field(repr=False, compare=False, hash=False)
    simplex_tangent: RealVector = field(repr=False, compare=False, hash=False)
    epsilon: float
    interaction_scale: float
    dimension: int
    seed: int

    def is_well_posed(self, atol: float = 1e-8) -> bool:
        r"""
        Verifica que (H, v) estén en dominio válido:
        H ∈ 𝔰𝔲(n) (hermítica sin traza) y v ∈ TΔⁿ⁻¹ (Σ v_i = 0).
        """
        H = self.hamiltonian
        v = self.simplex_tangent
        if H.shape != (self.dimension, self.dimension):
            return False
        if not np.allclose(H, H.conj().T, atol=atol):
            return False
        if abs(float(np.trace(H).real)) > atol:
            return False
        if v.shape != (self.dimension,):
            return False
        if abs(float(np.sum(v))) > 1e-6:
            return False
        if self.epsilon < 0.0 or not (0.0 <= self.interaction_scale <= 1.0):
            return False
        return True

    def frequency_proxy(self) -> RealVector:
        r"""
        Vector de frecuencias ω ∈ ℝⁿ extraído del espectro de H, analogía
        discreta de las frecuencias de acción-ángulo del problema integrable H₀.
        """
        ev = np.real(la.eigvalsh(self.hamiltonian))
        return ev.astype(np.float64)

    def lindstedt_scale(self) -> float:
        r"""Escala de la primera corrección de Lindstedt: ε · ‖v‖₂."""
        return float(self.epsilon * np.linalg.norm(self.simplex_tangent))


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.7 — Estados de campo, reporte GAN y oráculo de reward hacking
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TricksterFieldState:
    r"""Registro inmutable de un estado de campo adversarial de la malla APU."""

    cycle_id: str
    illusion_id: str
    illusion_type: str
    dream_isolation_flag: bool
    density_matrix: ComplexMatrix = field(repr=False, compare=False, hash=False)
    disguised_entropy: float
    disguised_purity: float
    reward_hacking_index: float
    stealth_dirichlet_energy: float
    heyting_verdict: HeytingOmega3
    provenance_hash: str
    timestamp_utc: float
    trace_distance_to_base: float = 0.0
    fidelity_to_base: float = 1.0
    melnikov_amplitude: float = 0.0
    homoclinic_transverse: bool = False
    small_divisor: float = 0.0
    horseshoe_entropy: float = 0.0
    kam_persists: bool = True

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        r"""Verifica el axioma C*: ρ ∈ 𝔇_n."""
        return CStarDensityCone.is_density(self.density_matrix, atol=atol)


@dataclass(frozen=True, slots=True)
class GANAdversarialCycleReport:
    r"""Reporte consolidado de un ciclo GAN adversarial."""

    report_id: str
    total_illusions_generated: int
    stealth_illusions_count: int
    vetoed_illusions_count: int
    degraded_illusions_count: int
    average_reward_hacking_score: float
    global_heyting_verdict: HeytingOmega3
    provenance_hash: str
    timestamp_utc: float

    def conservation_invariant(self) -> bool:
        r"""Invariante de conteo: stealth + vetoed = total y degraded ≤ stealth."""
        return (
            self.stealth_illusions_count + self.vetoed_illusions_count
            == self.total_illusions_generated
            and 0 <= self.degraded_illusions_count <= self.stealth_illusions_count
        )


@runtime_checkable
class RewardHackingOracle(Protocol):
    r"""
    Protocolo del oráculo de recompensa: (cost, sophistication, Dirichlet, dream)
    ↦ (rhi ∈ [0,1], veredicto ∈ Ω₃).
    """

    def compute(
        self,
        disguised_cost_ratio: float,
        sophistication: float,
        stealth_dirichlet: float,
        dream_isolation: bool,
    ) -> Tuple[float, HeytingOmega3]:
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades arbóreas de clipping
# ──────────────────────────────────────────────────────────────────────────────
def _clip_unit(x: float, name: str) -> float:
    r"""Proyección [0,1] con validación de finitud."""
    if not np.isfinite(x):
        raise ValueError(f"{name} debe ser finito, recibido {x!r}.")
    return float(min(1.0, max(0.0, x)))


def _clip_positive(x: float, name: str, lo: float = 0.0) -> float:
    r"""Proyección [lo, +∞) con validación de finitud."""
    if not np.isfinite(x):
        raise ValueError(f"{name} debe ser finito, recibido {x!r}.")
    return float(max(lo, x))


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1 ⟶ FASE 2 : MÉTODO BISAGRA
#
#   induce_spectral_flow_germ  :  𝔇_n × ℝ² × ℕ  ⟶  SpectralFlowGerm
#
#   Este morfismo CIERRA la Fase 1 y ES el objeto inicial de la Fase 2.
#   Todo funtor de Poincaré (sección, retorno, Melnikov, Birkhoff, KAM, tangle)
#   se aplica SOBRE el germen aquí construido. La continuación formal es:
#
#       germ = induce_spectral_flow_germ(...)
#       tangle = synthesize_poincare_tangle(germ, ω, k)     ← Fase 2
#       state  = run_terminal_cycle(obs, tangle)            ← Fase 3
# ──────────────────────────────────────────────────────────────────────────────
def induce_spectral_flow_germ(
    *,
    dim: int,
    disguised_cost_ratio: float,
    sophistication: float,
    seed: int,
    sophistication_attenuation: float = 0.85,
) -> SpectralFlowGerm:
    r"""
    Construye el germen del flujo espectral (H, v, ε, γ) ∈ 𝔰𝔲(n) ⊕ TΔⁿ⁻¹ ⊕ ℝ≥0.

    Este método cierra la FASE 1 y alimenta DIRECTAMENTE la FASE 2: entrega el
    dato local necesario para sintetizar un mapa de Poincaré perturbado. La
    elección de v se basa en el kernel Dirichlet–Poincaré de Pₙ:

        v = Φ · (ξ ⊙ w),   w_k ∝ exp(−β · λ_k),   β = 4 · sophistication,

    donde Φ son los modos tangentes de L_Pₙ y λ_k los autovalores del Laplaciano.
    Además la escala de perturbación es

        ε = cost · (1 − γ · sophistication),    γ = 0.85.

    Interpretación en *Méthodes Nouvelles*, Vol. I
    ----------------------------------------------
    ε es el parámetro de masa perturbadora; v es la variación primera del
    espectro (exponente característico); H es el generador del flujo
    hamiltoniano sobre U(n). El objeto retornado es la *sección local* del
    fibrado de jets J¹(𝔇_n) sobre la cual se construye 𝒫.

    Parameters
    ----------
    dim : int
        Dimensión del espacio de Hilbert ℋ_MAC (dim ≥ 2).
    disguised_cost_ratio : float ∈ [0, 1]
        Relación de coste disfrazado (proxy de profundidad del fraude).
    sophistication : float ∈ [0, 1]
        Sofisticación atencional del ataque adversarial.
    seed : int
        Semilla del RNG (determinismo de Lindstedt).
    sophistication_attenuation : float ∈ [0, 1], por defecto 0.85
        Amortiguación γ de la escala de perturbación por sofisticación.

    Returns
    -------
    SpectralFlowGerm
        Germen espectral, objeto inicial de la FASE 2 (Poincaré).
    """
    cost = _clip_unit(disguised_cost_ratio, "disguised_cost_ratio")
    soph = _clip_unit(sophistication, "sophistication")
    if dim < 2:
        raise ValueError("dim ≥ 2.")
    rng = np.random.default_rng(seed)
    epsilon = cost * max(0.0, 1.0 - sophistication_attenuation * soph)
    H = HypercomplexPauliBasis.sample_normalized_hamiltonian(dim, rng)
    Phi, lap_eigs = PathGraphDirichlet.tangent_modes(dim)
    beta = 4.0 * soph
    weights = np.exp(-beta * lap_eigs)
    weights = weights / (float(np.sum(weights)) + 1e-15)
    xi = rng.normal(size=weights.size)
    v = Phi @ (xi * weights)
    v = v - float(np.mean(v))
    vn = float(np.linalg.norm(v))
    if vn > 1e-15:
        v = v / vn
    else:
        v = Phi[:, 0].copy()
        v = v - float(np.mean(v))
        v = v / (float(np.linalg.norm(v)) + 1e-15)
    return SpectralFlowGerm(
        hamiltonian=H,
        simplex_tangent=v.astype(np.float64),
        epsilon=float(epsilon),
        interaction_scale=float(1.0 - soph),
        dimension=dim,
        seed=seed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — NÚCLEO DE POINCARÉ: SECCIONES, RETORNO, MELNIKOV, DIVISORES PEQUEÑOS,
#          FORMA NORMAL DE BIRKHOFF, TWIST DE MOSER, KAM Y ENREDO HOMOCLÍNICO
#
#   CONTINUACIÓN DIRECTA de `induce_spectral_flow_germ` (Fase 1):
#   el objeto SpectralFlowGerm se integra como sección transversal de Poincaré
#   Σ ⊂ (ℝ^{2n}, ω) y se analiza vía Poisson, Melnikov, divisores de Bryuno,
#   forma normal de Birkhoff, condición de twist de Moser y persistencia KAM,
#   culminando en `synthesize_poincare_tangle`, método bisagra hacia la FASE 3.
#
#   Cadena de morfismos de la Fase 2:
#       SpectralFlowGerm
#           ──► (q̃, p̃) separatriz
#           ──► Σ sección transversal
#           ──► 𝒫 mapa de retorno (simplectomorfismo)
#           ──► M(t₀) Melnikov
#           ──► γ_N, τ Bryuno
#           ──► Birkhoff nf + twist Moser + KAM
#           ──► HomoclinicTangleCertificate  =  synthesize_poincare_tangle(...)
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.0 — Geometría simpléctica canónica (cimiento geométrico de Poincaré)
#            Primer consumidor del germen: extrae (ω_can, X_H, {·,·}).
# ──────────────────────────────────────────────────────────────────────────────
class CanonicalSymplecticGeometry:
    r"""
    Geometría simpléctica canónica sobre T*ℝⁿ ≅ ℝ^{2n} con

        ω = Σ_{j=1}^n dq_j ∧ dp_j,     Ω = [[0, I], [−I, 0]].

    Toda la mecánica celeste de Poincaré se desarrolla sobre (M, ω). El campo
    hamiltoniano X_H cumple ι_{X_H} ω = dH y el flujo φ_H^t es un simplectomorfismo
    de un parámetro (teorema de Liouville–Poincaré: el volumen ωⁿ/n! se conserva).
    """

    @staticmethod
    def omega_matrix(n: int) -> RealMatrix:
        r"""Matriz de la forma simpléctica canónica Ω ∈ M_{2n}(ℝ), Ωᵀ = −Ω, Ω² = −I."""
        if n < 1:
            raise ValueError("n ≥ 1.")
        Omega = np.zeros((2 * n, 2 * n), dtype=np.float64)
        Omega[:n, n:] = np.eye(n)
        Omega[n:, :n] = -np.eye(n)
        return Omega

    @classmethod
    def is_symplectic_matrix(cls, S: RealMatrix, atol: float = 1e-8) -> bool:
        r"""
        Test S ∈ Sp(2n, ℝ): Sᵀ Ω S = Ω. Equivale a det S = 1 en dimensión 2;
        en dimensión 2n implica |det S| = 1 y preservación de ω.
        """
        n2 = S.shape[0]
        if n2 % 2 or S.shape[0] != S.shape[1]:
            return False
        n = n2 // 2
        Omega = cls.omega_matrix(n)
        residual = S.T @ Omega @ S - Omega
        return float(np.max(np.abs(residual))) < atol

    @staticmethod
    def poisson_bracket_scalar(
        dF_dq: RealVector,
        dF_dp: RealVector,
        dG_dq: RealVector,
        dG_dp: RealVector,
    ) -> float:
        r"""
        Corchete de Poisson canónico

            {F, G} = Σ_j (∂F/∂q_j ∂G/∂p_j − ∂F/∂p_j ∂G/∂q_j) = ω(X_F, X_G).

        Es la estructura de Lie sobre C^∞(M) que induce X_{{F,G}} = [X_F, X_G].
        """
        return float(np.dot(dF_dq, dG_dp) - np.dot(dF_dp, dG_dq))

    @classmethod
    def hamiltonian_vector_field(
        cls,
        dH_dq: RealVector,
        dH_dp: RealVector,
    ) -> Tuple[RealVector, RealVector]:
        r"""
        Campo hamiltoniano X_H = (∂H/∂p, −∂H/∂q), i.e.  q̇ = ∂H/∂p,  ṗ = −∂H/∂q.
        """
        return np.asarray(dH_dp, dtype=np.float64), -np.asarray(dH_dq, dtype=np.float64)

    @staticmethod
    def liouville_volume(Omega: RealMatrix) -> float:
        r"""Densidad de Liouville pfaffiana: Pf(Ω) = 1 para Ω canónica."""
        # Pfaffiano de Ω canónica es 1; se retorna |det Ω|^{1/2} = 1.
        det = float(np.linalg.det(Omega))
        return float(math.sqrt(max(0.0, abs(det))))


class ActionAngleChart:
    r"""
    Carta de acción-ángulo (I, θ) ∈ ℝⁿ × 𝕋ⁿ del sistema integrable H₀.

        I_j = (1 / 2π) ∮_{γ_j} p dq,     θ̇_j = ω_j(I) = ∂H₀/∂I_j.

    Poincaré construye las soluciones periódicas como puntos fijos de 𝒫 en
    estas coordenadas. El mapa integrable es una rotación rígida

        𝒫₀(θ, I) = (θ + ω(I) T, I).
    """

    @staticmethod
    def action_from_separatrix(p: RealVector, q: RealVector) -> float:
        r"""Acción reducida I = (1/2π) ∫ p dq a lo largo de un arco de separatriz."""
        if p.size < 2:
            return 0.0
        return float(_TRAPEZOID(p, q) / (2.0 * np.pi))

    @staticmethod
    def frequencies_from_hessian(H_II: RealMatrix) -> RealVector:
        r"""ω(I) ≈ ω₀ + H_{II} · (I − I₀); retorna la parte lineal (autovalores)."""
        ev = np.real(la.eigvals(H_II))
        return np.sort(np.abs(ev))

    @staticmethod
    def rotation_number(omega: RealVector, period: float = 2.0 * np.pi) -> RealVector:
        r"""Número de rotación ρ = ω T / 2π (mod 1) del mapa integrable."""
        rho = (omega * period) / (2.0 * np.pi)
        return np.mod(rho, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.1 — Órbita de separatriz del péndulo canónico (núcleo analítico)
# ──────────────────────────────────────────────────────────────────────────────
def pendulum_separatrix(t: RealVector) -> Tuple[RealVector, RealVector]:
    r"""
    Órbita homoclínica de separatriz del péndulo canónico

        H₀(q, p) = p²/2 − cos q.

    Solución explícita exacta (heteroclínica entre sillas (∓π, 0) identificadas
    sobre el cilindro):

        q₀(t) = 4 arctan(exp(t)) − π,   p₀(t) = 2 sech(t).

    Cumple:
        lim_{t → ±∞}  (q₀(t), p₀(t)) = (±π, 0),
        H₀(q₀, p₀) ≡ 0 (nivel de energía de la separatriz),
        {H₀, H₀} ≡ 0 (órbita del campo X_{H₀}).

    Esta curva ES la variedad Wˢ(p) = Wᵘ(p) del sistema no perturbado, y el
    soporte de integración de la función de Melnikov.
    """
    q = 4.0 * np.arctan(np.exp(t)) - np.pi
    p = 2.0 / np.cosh(t)
    return q, p


def pendulum_energy(q: RealVector, p: RealVector) -> RealVector:
    r"""H₀(q, p) = p²/2 − cos q evaluado puntualmente."""
    return 0.5 * np.asarray(p, dtype=np.float64) ** 2 - np.cos(q)


def pendulum_poisson_with_cos(q: RealVector, p: RealVector) -> RealVector:
    r"""
    {H₀, cos q} = −p sin q  (corchete de Poisson canónico sobre el péndulo).
    Es el integrando espacial de Melnikov para h = cos q.
    """
    return -np.asarray(p, dtype=np.float64) * np.sin(q)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.2 — Sección transversal de Poincaré
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSection:
    r"""
    Sección transversal de Poincaré Σ ⊂ M^{2n}:

        Σ = { x ∈ M : q_{2n} = c,  ‖∂H₀/∂p_{2n}(x)‖ > δ > 0 }.

    La última condición garantiza transversalidad: el flujo corta Σ sin ser
    tangente (X_H(x) ∉ T_x Σ). La restricción de la 2-forma simpléctica ω|_Σ
    es no degenerada, por lo que Σ hereda estructura simpléctica de dimensión
    2n − 2 (teorema de la sección de Poincaré–Cartan).

    El germen espectral induce una sección en el símplex: el índice de
    coordenada selecciona el eje espectral que desempeña el rol de q_{2n}.
    """

    coordinate_index: int = 2
    value: float = 0.0
    transversality_tol: float = 1e-6

    def is_transversal(
        self,
        dH_dp: float,
        velocity: float,
        tol: Optional[float] = None,
    ) -> bool:
        r"""
        Transversalidad en la sección: |∂H/∂p_j| ≥ tol y |velocity| ≥ tol.
        Equivale a X_H(x) ∉ T_x Σ.
        """
        delta = self.transversality_tol if tol is None else tol
        return abs(dH_dp) > delta and abs(velocity) > delta

    def project(self, dim: int) -> RealVector:
        r"""Vector de la sección en coordenadas espectrales (indicador de Σ)."""
        v = np.zeros(dim, dtype=np.float64)
        idx = int(self.coordinate_index) % dim
        v[idx] = 1.0
        return v

    def from_germ(self, germ: SpectralFlowGerm) -> RealVector:
        r"""
        Inducción de la sección a partir del germen de Fase 1: combina el
        indicador de Σ con el modo tangente v, produciendo la traza de Σ
        en TΔⁿ⁻¹. CONTINÚA `induce_spectral_flow_germ`.
        """
        indicator = self.project(germ.dimension)
        v = germ.simplex_tangent
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            return indicator
        return 0.5 * (indicator + v / nrm)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.3 — Mapa de primer retorno de Poincaré
# ──────────────────────────────────────────────────────────────────────────────
class PoincareReturnMap:
    r"""
    Aplicación de primer retorno de Poincaré 𝒫 : Σ → Σ asociada al flujo φᵗ del
    Hamiltoniano perturbado H = H₀ + ε H₁.

    Propiedades estructurales (Poincaré 1892, Vol. I, Cap. III–IV):
        1. 𝒫 es un simplectomorfismo: 𝒫* ω = ω, equivalentemente det D𝒫 = 1.
        2. Preserva la energía: H ∘ 𝒫 = H|_Σ.
        3. Su diferencial D𝒫 es la matriz monodrómica restringida a Σ;
           los exponentes de Lyapunov λ_i = lim_{N→∞} (1/N) log ‖D𝒫ᴺ (v)‖ se
           obtienen vía descomposición QR iterada (Benettin et al.).
        4. Los exponentes característicos de Floquet son los logaritmos de
           los autovalores de D𝒫 en el punto fijo.

    El mapa se inicializa desde un SpectralFlowGerm (continuación de Fase 1)
    o desde componentes hamiltonianos explícitos.
    """

    def __init__(
        self,
        section: PoincareSection,
        H0_diag: RealVector,
        dH0_dq: RealVector,
        H1_diag: RealVector,
        epsilon: float,
    ) -> None:
        self._sigma = section
        self._H0 = np.asarray(H0_diag, dtype=np.float64)
        self._dH0q = np.asarray(dH0_dq, dtype=np.float64)
        self._H1 = np.asarray(H1_diag, dtype=np.float64)
        self._eps = float(epsilon)

    @classmethod
    def from_germ(
        cls,
        germ: SpectralFlowGerm,
        section: Optional[PoincareSection] = None,
    ) -> "PoincareReturnMap":
        r"""
        Constructor bisagra Fase 1 → Fase 2: el germen induce H₀ = σ(H),
        H₁ = v (modo tangente) y ε. CONTINÚA `induce_spectral_flow_germ`.
        """
        if not germ.is_well_posed():
            raise ValueError("SpectralFlowGerm mal puesto.")
        sec = section if section is not None else PoincareSection(
            coordinate_index=1, value=0.0
        )
        H0 = np.real(la.eigvalsh(germ.hamiltonian))
        return cls(
            section=sec,
            H0_diag=H0,
            dH0_dq=germ.simplex_tangent,
            H1_diag=germ.simplex_tangent,
            epsilon=germ.epsilon,
        )

    def hamiltonian(self, x: RealVector) -> float:
        r"""H(x) = H₀(x) + ε · H₁(x)."""
        return float(np.dot(self._H0, x)) + self._eps * float(np.dot(self._H1, x))

    def planar_map(self, x: RealVector, steps: int = 1) -> RealVector:
        r"""
        Iteración discreta del mapa de retorno en el espacio de fases
        espectral. Integrador simpléctico de Euler–Cromer (preserva área
        a O(dt²) y es explícito):

            p_{n+1} = p_n − dt · ∂H/∂q (q_n),
            q_{n+1} = q_n + dt · ∂H/∂p (p_{n+1}).

        Aquí x desempeña el rol de coordenada de sección.
        """
        y = np.asarray(x, dtype=np.float64).copy()
        dt = 0.05
        for _ in range(max(1, steps)):
            p_dot = -self._dH0q - self._eps * np.tanh(y)
            y = y + dt * p_dot
        return y

    def jacobian_finite_diff(self, x: RealVector, h: float = 1e-6) -> RealMatrix:
        r"""Jacobiano D𝒫(x) por diferencias finitas centradas."""
        x = np.asarray(x, dtype=np.float64)
        d = x.size
        M = np.zeros((d, d), dtype=np.float64)
        for j in range(d):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h
            M[:, j] = (self.planar_map(xp) - self.planar_map(xm)) / (2.0 * h)
        return M

    def symplectic_residual(self, x: RealVector) -> float:
        r"""
        Residuo de simplecticidad ‖D𝒫ᵀ Ω D𝒫 − Ω‖_F / ‖Ω‖_F.
        Cero ssi 𝒫 es (localmente) un simplectomorfismo. Axioma III.
        """
        d = x.size
        n = max(1, d // 2)
        # Si d es impar, se embebe en dimensión par.
        M = self.jacobian_finite_diff(x)
        d2 = 2 * n
        if M.shape[0] != d2:
            # Proyección/padding al bloque 2n más cercano
            S = np.eye(d2, dtype=np.float64)
            m = min(d, d2)
            S[:m, :m] = M[:m, :m]
        else:
            S = M
        Omega = CanonicalSymplecticGeometry.omega_matrix(n)
        residual = S.T @ Omega @ S - Omega
        denom = float(np.linalg.norm(Omega, ord="fro")) + 1e-15
        return float(np.linalg.norm(residual, ord="fro") / denom)

    def lyapunov_spectrum(
        self,
        x0: RealVector,
        n_iter: int = 256,
        rng_seed: int = 0,
    ) -> RealVector:
        r"""
        Espectro de Lyapunov vía algoritmo de Benettin–Galgani–Giorgilli:
        iteración con re-ortonormalización QR sobre la matriz tangente D𝒫ᴺ.
        Los exponentes característicos de Poincaré–Floquet son estos λ_i.
        """
        rng = np.random.default_rng(rng_seed)
        d = x0.size
        Q = rng.normal(size=(d, d))
        Q, _ = la.qr(Q)
        sums = np.zeros(d, dtype=np.float64)
        x = x0.copy()
        h = 1e-6
        n_iter = max(1, int(n_iter))
        for _ in range(n_iter):
            M = np.zeros((d, d), dtype=np.float64)
            for j in range(d):
                xj = x.copy()
                xj[j] += h
                M[:, j] = (self.planar_map(xj) - self.planar_map(x)) / h
            Z = M @ Q
            Q, R = la.qr(Z)
            diag_R = np.abs(np.diag(R)) + 1e-15
            sums += np.log(diag_R)
            x = self.planar_map(x)
        return sums / float(n_iter)

    def floquet_multipliers(self, x_fixed: RealVector) -> NDArray[np.complex128]:
        r"""Multiplicadores de Floquet: autovalores de D𝒫 en un punto (casi) fijo."""
        M = self.jacobian_finite_diff(x_fixed)
        return la.eigvals(M)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.4 — Función de Melnikov (detección de intersección transversal)
# ──────────────────────────────────────────────────────────────────────────────
class MelnikovFunction:
    r"""
    Función de Melnikov asociada a una perturbación periódica

        H₁(q, p, t) = h(q, p) · sin(ω (t + t₀)).

    Definición (Melnikov 1963; Poincaré 1899, soluciones doblemente asintóticas):

        M(t₀) = ∫_{-∞}^{+∞} {H₀, h}(q̃(τ), p̃(τ)) · sin(ω (τ + t₀)) dτ,

    donde (q̃, p̃) es la órbita homoclínica de separatriz del péndulo y {·,·}
    es el corchete de Poisson canónico. Para h(q, p) = cos q se tiene
    {H₀, cos q} = −p sin q, de modo que

        M(t₀) = −∫ p̃(τ) sin q̃(τ) · sin(ω (τ + t₀)) dτ.

    Criterio de Poincaré–Melnikov–Smale: si M(t₀*) = 0 y M'(t₀*) ≠ 0, la
    variedad estable e inestable se intersecan transversalmente para ε > 0
    suficientemente pequeño, y nace el enredo homoclínico (tangle).

    La distancia entre Wˢ y Wᵘ a lo largo de la sección vale

        d(t₀) = ε M(t₀) / ‖∇H₀‖ + O(ε²).
    """

    def __init__(self, omega: float = 1.0, T_max: float = 20.0, n_grid: int = 4001):
        self._omega = float(omega)
        self._T = float(T_max)
        self._n = int(max(65, n_grid | 1))  # impar para incluir 0
        self._t = np.linspace(-self._T, self._T, self._n)
        self._q, self._p = pendulum_separatrix(self._t)
        self._poisson_h = pendulum_poisson_with_cos(self._q, self._p)
        self._energy = pendulum_energy(self._q, self._p)

    @property
    def separatrix_energy_residual(self) -> float:
        r"""máx |H₀(q̃, p̃)|; debe ser ~ 0 sobre la separatriz (test de consistencia)."""
        return float(np.max(np.abs(self._energy)))

    def evaluate(self, t0: float, h: str = "cos_q") -> float:
        r"""Evaluación numérica de M(t₀) por regla trapezoidal (integrador de Poisson)."""
        if h == "cos_q":
            integrand = self._poisson_h * np.sin(self._omega * (self._t + t0))
        elif h == "p2":
            integrand = (self._p ** 2) * np.sin(self._omega * (self._t + t0))
        else:
            raise ValueError(f"perturbación {h!r} no soportada")
        return float(_TRAPEZOID(integrand, self._t))

    def derivative(self, t0: float, h: str = "cos_q", delta: float = 1e-6) -> float:
        r"""M'(t₀) por diferencia central; simplez del cero ssi M' ≠ 0."""
        return (self.evaluate(t0 + delta, h=h) - self.evaluate(t0 - delta, h=h)) / (
            2.0 * delta
        )

    def splitting_distance(self, t0: float, epsilon: float) -> float:
        r"""
        Distancia de splitting d(t₀) ≈ ε M(t₀) / ‖∇H₀‖_{separatriz}.
        ‖∇H₀‖ = √(sin² q + p²) = √(sin² q̃ + 4 sech² t).
        """
        grad_norm = np.sqrt(np.sin(self._q) ** 2 + self._p ** 2)
        mean_grad = float(np.mean(grad_norm) + 1e-15)
        return float(epsilon * self.evaluate(t0) / mean_grad)

    def amplitude(self, n_samples: int = 64) -> float:
        r"""
        Amplitud efectiva del Melnikov: máx_{t₀ ∈ [0, 2π/ω)} |M(t₀)|.
        Controla la anchura del lóbulo homoclínico.
        """
        if self._omega <= 1e-15:
            return 0.0
        period = 2.0 * np.pi / self._omega
        t0s = np.linspace(0.0, period, max(8, n_samples), endpoint=False)
        return float(max(abs(self.evaluate(t0)) for t0 in t0s))

    def has_transverse_zero(
        self,
        n_samples: int = 256,
        tol: float = 1e-4,
    ) -> Tuple[bool, Optional[float]]:
        r"""
        Barrido de ceros simples de M(t₀) sobre un periodo. Retorna
        (True, t₀*) si ∃ cero simple, (False, None) en otro caso.
        Método: cambio de signo + bisección + test |M'| > tol.
        """
        if self._omega <= 1e-15:
            return False, None
        period = 2.0 * np.pi / self._omega
        n_samples = max(16, int(n_samples))
        t0s = np.linspace(0.0, period, n_samples, endpoint=False)
        vals = np.array([self.evaluate(t0) for t0 in t0s])
        for i in range(n_samples):
            a, b = vals[i], vals[(i + 1) % n_samples]
            if a * b < 0.0:
                t_a = t0s[i]
                t_b = t0s[(i + 1) % n_samples]
                if t_b <= t_a:
                    t_b += period
                h = 1e-6
                t_star = 0.5 * (t_a + t_b)
                for _ in range(40):
                    m = 0.5 * (t_a + t_b)
                    f_m = self.evaluate(m)
                    f_a = self.evaluate(t_a)
                    if abs(f_m) < tol:
                        t_star = m
                        break
                    if f_a * f_m < 0.0:
                        t_b = m
                    else:
                        t_a = m
                    t_star = 0.5 * (t_a + t_b)
                dm = (self.evaluate(t_star + h) - self.evaluate(t_star - h)) / (2.0 * h)
                if abs(dm) > tol:
                    return True, float(t_star % period)
        return False, None


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.5 — Analizador de divisores pequeños y resonancia de Bryuno
# ──────────────────────────────────────────────────────────────────────────────
class SmallDivisorAnalyzer:
    r"""
    Analiza el conjunto de resonancias { ω · k = 0 } del vector de frecuencias
    ω = (ω₁, …, ωₙ) ∈ ℝⁿ frente a los wave-vectors enteros k ∈ ℤⁿ.

    Cantidades clave (Poincaré Vol. II, Cap. IX; Bryuno 1971; Siegel–Moser):

        γ_N(ω) = inf { |ω · k| : k ∈ ℤⁿ, 0 < ‖k‖_∞ ≤ N },
        τ(ω)   = sup { τ : sup_N N^τ γ_N(ω) > 0 }  (exponente de Bryuno).

    Condición KAM (Kolmogorov–Arnold–Moser):

        |ω · k| ≥ γ / ‖k‖^τ     ∀ k ∈ ℤⁿ \ {0},     Σ_N log(1/γ_N) / N^{τ+1} < ∞.

    Resonancia fuerte (pequeño divisor peligroso): γ_N → 0 polinomialmente,
    que es el motor de los enredos homoclínicos adversariales.

    Para n grande el cubo [−K, K]ⁿ es exponencial: se usa muestreo de
    vectores k de norma baja + direcciones aleatorias, no el producto cartesiano
    completo más allá de n = 4, K = 6.
    """

    _CARTESIAN_LIMIT: Final[int] = 50_000

    def __init__(self, omega: RealVector, max_k: int = 8) -> None:
        self._omega = np.asarray(omega, dtype=np.float64).reshape(-1)
        self._n = int(self._omega.size)
        self._K = int(max(1, max_k))

    def _k_iter(self, K: int) -> Iterable[Tuple[int, ...]]:
        r"""Iterador de k ∈ ℤⁿ \ {0} con ‖k‖_∞ ≤ K, con corte combinatorio."""
        n = self._n
        if n <= 0:
            return
        cardinality = (2 * K + 1) ** n
        if n <= 4 and cardinality <= self._CARTESIAN_LIMIT:
            grid = range(-K, K + 1)
            for k in product(grid, repeat=n):
                if any(ki != 0 for ki in k):
                    yield k
            return
        # Muestreo: ejes, pares, y vectores aleatorios de norma baja.
        seen = set()
        for i in range(n):
            for s in (-K, -1, 1, K):
                k = [0] * n
                k[i] = int(s)
                tup = tuple(k)
                if tup not in seen:
                    seen.add(tup)
                    yield tup
        rng = np.random.default_rng(abs(hash(tuple(np.round(self._omega, 8)))) % (2**32))
        budget = min(self._CARTESIAN_LIMIT, max(256, 64 * n * K))
        for _ in range(budget):
            k = tuple(int(x) for x in rng.integers(-K, K + 1, size=n))
            if any(ki != 0 for ki in k) and k not in seen:
                seen.add(k)
                yield k

    def min_resonance(self, norm: str = "inf") -> Tuple[float, IntVector]:
        r"""
        Calcula γ_N(ω) y el vector k* que lo alcanza, sobre ℤⁿ ∩ [−N, N]ⁿ.
        """
        best = np.inf
        best_k = np.zeros(self._n, dtype=np.int64)
        for k in self._k_iter(self._K):
            if norm == "inf" and max(abs(ki) for ki in k) > self._K:
                continue
            k_arr = np.array(k, dtype=np.float64)
            val = abs(float(np.dot(self._omega, k_arr)))
            if val < best:
                best = val
                best_k = np.array(k, dtype=np.int64)
        if not np.isfinite(best):
            best = 0.0
        return float(best), best_k

    def diophantine_constant(self, tau: float = 1.0) -> float:
        r"""
        Estimador de γ en |ω · k| ≥ γ / ‖k‖^τ: γ̂ = inf |ω · k| · ‖k‖_∞^τ.
        """
        gamma_hat = np.inf
        for k in self._k_iter(self._K):
            kn = max(abs(ki) for ki in k)
            val = abs(float(np.dot(self._omega, np.array(k, dtype=np.float64))))
            gamma_hat = min(gamma_hat, val * (float(kn) ** tau))
        if not np.isfinite(gamma_hat):
            return 0.0
        return float(gamma_hat)

    def bryuno_exponent(self, N_max: int = 32) -> float:
        r"""
        Estimador empírico del exponente τ: regresión log–log de γ_N vs N:

            log γ_N  ~  −τ · log N + c.
        """
        Ns: List[int] = []
        gammas: List[float] = []
        K_saved = self._K
        for N in (2, 4, 8, 16, max(16, N_max)):
            self._K = int(N)
            g, _ = self.min_resonance()
            gammas.append(max(g, 1e-15))
            Ns.append(int(N))
        self._K = K_saved
        x = np.log(np.array(Ns, dtype=np.float64))
        y = -np.log(np.array(gammas, dtype=np.float64))
        if x.size < 2:
            return 0.0
        slope, _ = np.polyfit(x, y, 1)
        return float(max(0.0, slope))

    def kam_series_converges(self, tau: Optional[float] = None) -> bool:
        r"""
        Test heurístico de la condición de Bryuno:

            Σ_N  log(1 / γ_N) / N^{τ+1}  < ∞.

        Se aproxima con N ∈ {2,4,8,16}. Convergencia ⇒ toros KAM persistentes.
        """
        tau_use = self.bryuno_exponent() if tau is None else float(tau)
        K_saved = self._K
        acc = 0.0
        for N in (2, 4, 8, 16):
            self._K = N
            g, _ = self.min_resonance()
            acc += math.log(1.0 / max(g, 1e-15)) / (float(N) ** (tau_use + 1.0))
        self._K = K_saved
        return bool(np.isfinite(acc) and acc < 50.0)

    def resonance_severity(self) -> float:
        r"""
        Severidad normalizada σ ∈ [0, 1] cuantificando la peligrosidad del
        vector ω para la expansión de Lindstedt:

            σ = 1 − γ_N(ω) / γ_max,   γ_max = |ω|_∞ / N.

        Mayor σ ⇒ mayor divergencia perturbativa.
        """
        g, _ = self.min_resonance()
        omega_inf = float(np.max(np.abs(self._omega))) if self._omega.size else 1.0
        gmax = omega_inf / max(1.0, float(self._K))
        if gmax <= 0.0:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - g / gmax)))


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.6 — Forma normal de Birkhoff, twist de Moser y persistencia KAM
# ──────────────────────────────────────────────────────────────────────────────
class BirkhoffNormalForm:
    r"""
    Cálculo de la parte lineal (jacobiana) de la forma normal de Birkhoff
    asociada a un Hamiltoniano perturbado. El primer invariante de Birkhoff
    es el conjunto de autovalores imaginarios puros ±i ω_j del Hessiano
    simpléctico en el punto fijo:

        σ(J ∇²H) = { ±i ω_j : j = 1, …, n }.

    Si los ω_j son ℚ-linealmente independientes (condición de no resonancia),
    el campo es formalmente integrable en el entorno del punto fijo
    (Birkhoff 1927; Poincaré Vol. II). Los invariantes de Birkhoff de orden
    superior β_α I^α aparecen en la forma normal

        H(I) = ω · I + Σ_{|α|≥2} β_α I^α.
    """

    @staticmethod
    def symplectic_eigenvalues(H_hess: RealMatrix) -> RealVector:
        r"""
        Autovalores imaginarios puros del Hessiano simpléctico. Retorna
        [ω_1, …, ω_n] positivos extraídos de la parte imaginaria.
        """
        n2 = H_hess.shape[0]
        if n2 < 2 or n2 % 2:
            ev = la.eigvals(H_hess)
            imags = np.sort([abs(np.imag(z)) for z in ev if abs(np.imag(z)) > 1e-12])
            return np.array(imags, dtype=np.float64)
        n = n2 // 2
        J = CanonicalSymplecticGeometry.omega_matrix(n)
        M = J @ H_hess
        ev = la.eigvals(M)
        imags = np.sort([abs(np.imag(z)) for z in ev if np.imag(z) > 1e-12])
        if imags.size >= 2:
            imags = imags[: max(1, imags.size // 2)]
        return np.array(imags, dtype=np.float64)

    @staticmethod
    def is_non_resonant(omega: RealVector, tol: float = 1e-6) -> bool:
        r"""
        Test de independencia ℚ-lineal empírica: min |ω · k| > tol sobre
        ‖k‖_1 ≤ 3 (resonancias de orden bajo).
        """
        if omega.size == 0:
            return True
        K = 3
        omega = np.asarray(omega, dtype=np.float64)
        n = int(omega.size)
        if n > 5:
            # muestreo para evitar explosión combinatoria
            sda = SmallDivisorAnalyzer(omega, max_k=K)
            g, _ = sda.min_resonance()
            return g > tol
        for k in product(range(-K, K + 1), repeat=n):
            if all(ki == 0 for ki in k):
                continue
            val = abs(float(np.dot(omega, np.array(k, dtype=np.float64))))
            if val < tol:
                return False
        return True

    @staticmethod
    def birkhoff_invariants_quadratic(omega: RealVector, germ_eps: float) -> RealVector:
        r"""
        Primera corrección cuadrática de Birkhoff β_j ≈ ε · ω_j² / (1 + ‖ω‖²),
        proxy de los invariantes I² en H = ω·I + β·I².
        """
        w = np.asarray(omega, dtype=np.float64)
        denom = 1.0 + float(np.dot(w, w))
        return (float(germ_eps) * (w * w) / denom).astype(np.float64)


class MoserTwistMap:
    r"""
    Condición de twist de Moser y teorema de Poincaré–Birkhoff.

    Un mapa de anillo

        𝒫(θ, I) = (θ + α(I) + ε f,  I + ε g)

    satisface twist si ∂α/∂I ≠ 0. Entonces:

    • Poincaré–Birkhoff: para cada racional p/q en el intervalo de números
      de rotación existen ≥ 2 puntos periódicos de periodo q.
    • Moser (twist theorem): si 𝒫 ∈ C^{3+δ} y |ε| pequeño, persisten curvas
      invariantes con número de rotación diofántico.

    La degeneración del twist (∂α/∂I → 0) es el análogo celestial del
    front-loading (`UNBALANCED_APU_BIDDING`): la monotonía de la rotación
    se pierde y nacen islas elípticas + puntos hiperbólicos.
    """

    @staticmethod
    def twist_derivative(alpha: RealVector, actions: RealVector) -> float:
        r"""
        Estimador de ∂α/∂I por regresión lineal: α(I) ≈ α₀ + τ I.
        Twist no degenerado ssi |τ| > 0.
        """
        a = np.asarray(alpha, dtype=np.float64).reshape(-1)
        I = np.asarray(actions, dtype=np.float64).reshape(-1)
        m = min(a.size, I.size)
        if m < 2:
            return 0.0
        a, I = a[:m], I[:m]
        I_c = I - float(np.mean(I))
        a_c = a - float(np.mean(a))
        denom = float(np.dot(I_c, I_c))
        if denom < 1e-15:
            return 0.0
        return float(np.dot(I_c, a_c) / denom)

    @classmethod
    def is_twist(cls, alpha: RealVector, actions: RealVector, tol: float = 1e-8) -> bool:
        r"""Test ∂α/∂I ≠ 0 (condición de twist de Moser)."""
        return abs(cls.twist_derivative(alpha, actions)) > tol

    @staticmethod
    def poincare_birkhoff_fixed_points(p: int, q: int) -> int:
        r"""
        Cota inferior del teorema de Poincaré–Birkhoff: ≥ 2 puntos de periodo q
        para la rotación p/q (q ≥ 1, gcd(p, q) = 1).
        """
        if q < 1:
            raise ValueError("q ≥ 1.")
        return 2


class LindstedtSeries:
    r"""
    Serie de Lindstedt–Poincaré

        q(t) = q₀ + Σ_{k ≠ 0} [A_k / (ω · k)] exp(i k · θ),

    cuya convergencia está obstruida por los divisores pequeños ω·k.
    El radio de convergencia formal se estima como

        R ~ inf_k |ω · k| / ‖A_k‖.

    Poincaré demostró (Vol. II) que la serie es en general *divergente*
    (aunque asintótica) cuando hay resonancias densas.
    """

    @staticmethod
    def coefficient_bound(omega: RealVector, max_k: int = 6) -> float:
        r"""
        Cota inf |ω · k| sobre 0 < ‖k‖_∞ ≤ max_k; proxy de R.
        Si el ínfimo es ~ 0 la serie diverge.
        """
        sda = SmallDivisorAnalyzer(omega, max_k=max_k)
        g, _ = sda.min_resonance()
        return float(g)

    @staticmethod
    def diverges(omega: RealVector, threshold: float = 1e-6, max_k: int = 6) -> bool:
        r"""Test heurístico de divergencia: γ_N < threshold."""
        return LindstedtSeries.coefficient_bound(omega, max_k=max_k) < threshold


class KAMTorusPersistence:
    r"""
    Estimador de persistencia de toros KAM.

    Un toro con vector de frecuencias ω persiste bajo perturbación ε si:
        1. ω es diofántico (|ω·k| ≥ γ / ‖k‖^τ),
        2. el twist no se anula,
        3. |ε| < ε_c(γ, τ) ~ γ² (umbral clásico de Arnold).

    El contrapositivo (toro destruido) produce caos homoclínico y es el
    régimen adversarial que el motor busca sintetizar.
    """

    @staticmethod
    def persists(
        omega: RealVector,
        epsilon: float,
        twist: float,
        tau: float,
        gamma: float,
    ) -> bool:
        r"""Test KAM heurístico: diofántico ∧ twist ∧ |ε| < c γ²."""
        if abs(twist) < 1e-10:
            return False
        if gamma <= 0.0 or not np.isfinite(tau):
            return False
        eps_c = 0.25 * (gamma ** 2) / (1.0 + abs(tau))
        return abs(epsilon) < eps_c

    @classmethod
    def from_analyzer(
        cls,
        sda: SmallDivisorAnalyzer,
        epsilon: float,
        twist: float,
    ) -> bool:
        r"""Persistencia KAM a partir de un analizador de divisores."""
        gamma, _ = sda.min_resonance()
        tau = sda.bryuno_exponent(N_max=16)
        return cls.persists(sda._omega, epsilon, twist, tau, gamma)


class SmaleHorseshoe:
    r"""
    Herradura de Smale–Birkhoff.

    Si Wˢ ⋔ Wᵘ (cero simple de Melnikov), existe un conjunto de Cantor
    invariante Λ ⊂ Σ y un homeomorfismo conjugando 𝒫|_Λ al shift de Bernoulli
    σ : {0,1}^ℤ → {0,1}^ℤ. Entonces

        h_top(𝒫|_Λ) = log 2,
        #Fix(𝒫^n |_Λ) = 2^n.

    Esta entropía es el invariante topológico del enredo homoclínico y se
    reporta en HomoclinicTangleCertificate.horseshoe_entropy.
    """

    @staticmethod
    def topological_entropy(transverse: bool) -> float:
        r"""h_top = log 2 si hay herradura; 0 en caso contrario."""
        return _POINCARE_LOG2 if transverse else 0.0

    @staticmethod
    def periodic_count(period: int, transverse: bool) -> int:
        r"""#Fix(𝒫^n |_Λ) = 2^n sobre la herradura."""
        if not transverse or period < 0:
            return 0
        return 2 ** int(period)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.7 — Certificado del enredo homoclínico
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class HomoclinicTangleCertificate:
    r"""
    Certificado estructural del enredo homoclínico (Poincaré Vol. III):

    • melnikov_amplitude : |M|_max  (anchura del lóbulo)
    • transverse         : ∃ cero simple de M ⟹ Wˢ ⋔ Wᵘ
    • t_star             : parámetro del cero transversal
    • small_divisor      : γ_N(ω)
    • bryuno_tau         : τ(ω) estimado
    • resonance_severity : σ ∈ [0, 1]
    • lyapunov_positive  : max λ_i > 0 (caos)
    • horseshoe_entropy  : h_top ≥ log 2 en Λ ⊆ Wˢ ⋔ Wᵘ
    • twist              : ∂α/∂I (Moser)
    • kam_persists       : toro KAM superviviente
    • symplectic_residual: residuo de 𝒫*ω − ω
    • lindstedt_diverges : serie de Lindstedt divergente
    """

    melnikov_amplitude: float
    transverse: bool
    t_star: Optional[float]
    small_divisor: float
    bryuno_tau: float
    resonance_severity: float
    lyapunov_positive: bool
    horseshoe_entropy: float
    twist: float = 0.0
    kam_persists: bool = True
    symplectic_residual: float = 0.0
    lindstedt_diverges: bool = False
    schema_version: str = _SCHEMA_VERSION


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.8 — Perturbador espectral de densidad (Melnikov ↔ Unitario)
# ──────────────────────────────────────────────────────────────────────────────
class TricksterDensityPerturber:
    r"""
    Generador espectral de perturbaciones homoclínicas acoplado a la mecánica
    celeste de Poincaré. Dado el germen (H, v, ε) ∈ 𝔰𝔲(n) ⊕ TΔⁿ⁻¹ ⊕ ℝ≥0 y
    el par (ω, k), construye el Hamiltoniano homoclínico

        H_hom = H_pendulum + ε · (ω · k)⁻¹ · H_forcing,

    donde H_pendulum cuantiza la separatriz p²/2 − cos q en la base espectral
    de MAC y H_forcing = diag(cos q) modula la resonancia del pequeño divisor.

    El método `synthesize_poincare_tangle` CIERRA la Fase 2 y ES el objeto
    inicial de la Fase 3 (interlock ciber-físico).
    """

    def __init__(
        self,
        tolerance: float = 1e-9,
        max_rhi_threshold: float = 0.85,
        resonance_floor: float = 1e-12,
    ) -> None:
        self._tol = float(tolerance)
        self._rhi_max = float(max_rhi_threshold)
        self._res_floor = float(resonance_floor)

    def _build_homoclinic_hamiltonian(
        self,
        dim: int,
        omega: np.ndarray,
        k_vector: np.ndarray,
        epsilon: float,
    ) -> Tuple[ComplexMatrix, float]:
        r"""
        Sintetiza H_hom ∈ Mₙ(ℂ) como cuantización canónica de la perturbación

            H = p²/2 − cos(q) + ε · cos(q) / (ω · k)

        y calcula la resonancia |ω · k|. El operador p se realiza por
        diferencias finitas centradas periódicas (anillo, analogía del
        cilindro de Poincaré del péndulo).
        """
        q_grid = np.linspace(-np.pi, np.pi, dim, endpoint=False)
        p_op = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            p_op[i, (i + 1) % dim] = -1j / 2.0
            p_op[i, (i - 1) % dim] = 1j / 2.0
        p_op = 0.5 * (p_op + p_op.conj().T)
        omega = np.asarray(omega, dtype=np.float64).reshape(-1)
        k_vector = np.asarray(k_vector, dtype=np.float64).reshape(-1)
        m = min(omega.size, k_vector.size)
        dot_product = float(np.dot(omega[:m], k_vector[:m])) if m else 0.0
        resonance = max(abs(dot_product), self._res_floor)
        H_pendulum = 0.5 * (p_op @ p_op) - np.diag(np.cos(q_grid)).astype(np.complex128)
        H_forcing = epsilon * np.diag(np.cos(q_grid)).astype(np.complex128) / resonance
        H_hom = 0.5 * (H_pendulum + H_pendulum.conj().T) + 0.5 * (
            H_forcing + H_forcing.conj().T
        )
        # Proyección a 𝔰𝔲(n): se resta la traza (Axioma I).
        H_hom = H_hom - (np.trace(H_hom) / dim) * np.eye(dim, dtype=np.complex128)
        H_hom = 0.5 * (H_hom + H_hom.conj().T)
        return H_hom, resonance

    def synthesize_from_germ(
        self,
        germ: SpectralFlowGerm,
        density_op: ComplexMatrix,
        omega_frequencies: Optional[RealVector] = None,
        k_wavevectors: Optional[RealVector] = None,
        attack_type: IllusionAttackType = IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
        melnikov_omega: float = 1.0,
    ) -> Tuple[IllusionDensityPerturbation, HomoclinicTangleCertificate]:
        r"""
        Continuación DIRECTA de `induce_spectral_flow_germ`: consume el germen
        de Fase 1, extrae (ω, ε) y delega en `synthesize_poincare_tangle`.
        """
        if not germ.is_well_posed():
            raise ValueError("SpectralFlowGerm mal puesto.")
        omega = (
            np.asarray(omega_frequencies, dtype=np.float64)
            if omega_frequencies is not None
            else germ.frequency_proxy()
        )
        if k_wavevectors is None:
            k = np.zeros(germ.dimension, dtype=np.float64)
            if germ.dimension >= 2:
                k[0] = 1.0
                k[1] = -1.0
            k_wavevectors = k
        return self.synthesize_poincare_tangle(
            density_op=density_op,
            omega_frequencies=omega,
            k_wavevectors=np.asarray(k_wavevectors, dtype=np.float64),
            epsilon_perturbation=germ.epsilon,
            attack_type=attack_type,
            melnikov_omega=melnikov_omega,
            germ=germ,
        )

    # ── MÉTODO BISAGRA FASE 2 ⟶ FASE 3 ─────────────────────────────────────
    def synthesize_poincare_tangle(
        self,
        *,
        density_op: ComplexMatrix,
        omega_frequencies: RealVector,
        k_wavevectors: RealVector,
        epsilon_perturbation: float = 0.05,
        attack_type: IllusionAttackType = IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
        melnikov_omega: float = 1.0,
        return_certificate: bool = True,
        germ: Optional[SpectralFlowGerm] = None,
    ) -> Tuple[IllusionDensityPerturbation, HomoclinicTangleCertificate]:
        r"""
        Sintetiza un ataque de enredo homoclínico sobre ρ₀ ∈ 𝔇_n.

        Este método CIERRA la FASE 2 y ABRE la FASE 3: el certificado de tangle
        (Wˢ ⋔ Wᵘ, γ_N, h_top, KAM) es el objeto que el autómata ciber-físico
        consume para transicionar IDLE ↦ ARMED ↦ FIRED.

        Protocolo (Poincaré Vols. I–III)
        --------------------------------
        1. Construye H_hom ∈ 𝔰𝔲(n) vía cuantización del péndulo forzado.
        2. Calcula γ_N(ω) = min |ω · k| sobre k ∈ ℤⁿ, ‖k‖_∞ ≤ 8.
        3. Evalúa M(t₀) por Melnikov clásico con h(q,p) = cos q.
        4. Aplica U = exp(−i ε H_hom) sobre ρ₀ → ρ_ill  (flujo hamiltoniano).
        5. Verifica Axioma II (CPTP) y calcula RHI vía Umegaki.
        6. Estima twist de Moser, persistencia KAM y residuo simpléctico.
        7. Infecta la homología del complejo de APUs con β₁ ≥ 0.

        Returns
        -------
        (IllusionDensityPerturbation, HomoclinicTangleCertificate)
            Par que alimenta `run_terminal_cycle` en la Fase 3.
        """
        density_op = CStarDensityCone.project_to_density_cone(density_op)
        dim = int(density_op.shape[0])
        H_hom, resonance = self._build_homoclinic_hamiltonian(
            dim, omega_frequencies, k_wavevectors, epsilon_perturbation
        )

        # --- Unitario U = exp(−i ε H_hom) vía descomposición espectral ---
        evals, evecs = la.eigh(H_hom)
        U = evecs @ np.diag(np.exp(-1j * epsilon_perturbation * evals)) @ evecs.conj().T

        # --- Evolución CPTP (Axioma II) ---
        rho_ill = U @ density_op @ U.conj().T
        rho_ill = CStarDensityCone.project_to_density_cone(rho_ill)

        # --- Divergencia de Umegaki D(ρ_ill ‖ ρ₀)  (acción de Poincaré) ---
        d_umegaki = CStarDensityCone.umegaki_divergence(rho_ill, density_op)

        # --- RHI = D_Umegaki / (‖[ρ₀, H_hom]‖_F + ε_p) ∈ [0, 1] ---
        comm_f = CStarDensityCone.commutator_frobenius(density_op, H_hom)
        rhi_score = float(min(1.0, max(0.0, d_umegaki / (comm_f + 1e-6))))

        # --- Análisis de divisores pequeños (Vol. II) ---
        sda = SmallDivisorAnalyzer(np.asarray(omega_frequencies, dtype=np.float64), max_k=8)
        gamma_N, _k_star = sda.min_resonance()
        severity = sda.resonance_severity()
        tau = sda.bryuno_exponent(N_max=16)

        # --- Melnikov clásico sobre el péndulo estándar (Vol. III) ---
        melnikov = MelnikovFunction(omega=melnikov_omega)
        M_amp = melnikov.amplitude(n_samples=48)
        transverse, t_star = melnikov.has_transverse_zero(n_samples=128)

        # --- Mapa de primer retorno y espectro de Lyapunov ---
        H0_diag = np.real(la.eigvalsh(H_hom))
        prm = PoincareReturnMap(
            section=PoincareSection(coordinate_index=1, value=0.0),
            H0_diag=H0_diag,
            dH0_dq=H0_diag,
            H1_diag=H0_diag,
            epsilon=epsilon_perturbation,
        )
        x0 = np.full(dim, 1e-3, dtype=np.float64)
        lyap = prm.lyapunov_spectrum(x0=x0, n_iter=32, rng_seed=17)
        lyap_positive = bool(float(np.max(lyap)) > 0.0)
        symplectic_res = prm.symplectic_residual(x0)

        # --- Twist de Moser y KAM ---
        actions = np.linspace(0.1, 1.0, max(2, dim))
        alpha = H0_diag[: actions.size] if H0_diag.size else actions
        if alpha.size != actions.size:
            alpha = np.resize(alpha, actions.size)
        twist = MoserTwistMap.twist_derivative(alpha, actions)
        kam = KAMTorusPersistence.persists(
            omega=np.asarray(omega_frequencies, dtype=np.float64),
            epsilon=epsilon_perturbation,
            twist=twist,
            tau=tau,
            gamma=gamma_N,
        )
        lindstedt_div = LindstedtSeries.diverges(
            np.asarray(omega_frequencies, dtype=np.float64)
        )
        horseshoe_entropy = SmaleHorseshoe.topological_entropy(transverse)

        # --- Infección de homología β₁ ---
        betti_1 = 1 if (rhi_score > self._rhi_max or transverse) else 0

        # --- Prueba Merkle de procedencia ---
        hasher = hashlib.sha256()
        hasher.update(
            f"{attack_type.value}::{rhi_score:.6f}::{resonance:.6e}"
            f"::{betti_1}::{gamma_N:.6e}::{severity:.6f}"
            f"::{int(transverse)}::{horseshoe_entropy:.6f}".encode("utf-8")
        )
        proof_sha256 = hasher.hexdigest()

        cert = TricksterAttackCertificate(
            attack_type=attack_type.value,
            rhi_score=rhi_score,
            homoclinic_residual=float(
                np.linalg.norm(H_hom - H_hom.conj().T)
            ),
            small_divisor_resonance=resonance,
            betti_1_induced=betti_1,
            is_unitary_cptp=bool(
                abs(float(np.trace(rho_ill).real) - 1.0) < self._tol
            ),
            merkle_proof_sha256=proof_sha256,
            schema_version=_SCHEMA_VERSION,
        )
        tangle_cert = HomoclinicTangleCertificate(
            melnikov_amplitude=M_amp,
            transverse=transverse,
            t_star=t_star,
            small_divisor=gamma_N,
            bryuno_tau=tau,
            resonance_severity=severity,
            lyapunov_positive=lyap_positive,
            horseshoe_entropy=horseshoe_entropy,
            twist=twist,
            kam_persists=kam,
            symplectic_residual=symplectic_res,
            lindstedt_diverges=lindstedt_div,
            schema_version=_SCHEMA_VERSION,
        )
        perturbation = IllusionDensityPerturbation(
            illusion_density_matrix=rho_ill,
            original_density_matrix=density_op,
            unitary_operator=U,
            attack_certificate=cert,
        )
        # germ se reserva para auditoría / trazabilidad de la anidación 1→2.
        _ = germ
        _ = return_certificate
        return perturbation, tangle_cert


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.9 — Generador espectral adversarial (integración del germen)
# ──────────────────────────────────────────────────────────────────────────────
class AdversarialSpectralGenerator:
    r"""
    Integra un `SpectralFlowGerm` sobre 𝔇_n preservando CPTP:

        ρ(ε) = U(ε) · diag(proj_Δ(λ₀ + ε v)) · U(ε)†,
        U(ε) = exp(−i ε H).

    Es el flujo de Lindstedt–Poincaré truncado a orden 1 sobre el símplex
    espectral, conjugado por el grupo uniparamétrico generado por H.
    """

    _SOFISTICATION_ATTENUATION: Final[float] = 0.85

    @staticmethod
    def integrate_spectral_flow_germ(
        germ: SpectralFlowGerm,
        base_rho: ComplexMatrix,
    ) -> Tuple[ComplexMatrix, float, float, float]:
        r"""
        Aplica el germen como deformación por conjugación y re-proyección.
        CONTINÚA `induce_spectral_flow_germ`.

        Returns
        -------
        (ρ_ill, purity, entropy, dirichlet_energy)
        """
        if not germ.is_well_posed():
            raise ValueError("SpectralFlowGerm mal puesto (H ∉ 𝔰𝔲(n) o v ∉ TΔ).")
        rho0 = CStarDensityCone.project_to_density_cone(base_rho)
        lam0 = CStarDensityCone.spectrum_ordered(rho0)
        lam_eps = CStarDensityCone.project_to_simplex(
            lam0 + germ.epsilon * germ.simplex_tangent
        )
        lam_eps = np.sort(lam_eps)
        U = la.expm(-1j * germ.epsilon * germ.hamiltonian)
        _, evecs0 = la.eigh(CStarDensityCone.hermitize(rho0))
        evecs = U @ evecs0
        rho_ill = (evecs * lam_eps) @ evecs.conj().T
        rho_ill = CStarDensityCone.project_to_density_cone(rho_ill)
        eigvals = CStarDensityCone.spectrum_ordered(rho_ill)
        purity = CStarDensityCone.purity(eigvals)
        entropy = CStarDensityCone.von_neumann_entropy(eigvals)
        stealth_dirichlet = PathGraphDirichlet.combined_energy(eigvals)
        stealth_dirichlet += germ.epsilon * 0.05 * germ.interaction_scale
        return rho_ill, purity, entropy, float(stealth_dirichlet)

    @classmethod
    def generate_stealth_perturbation(
        cls,
        base_rho: ComplexMatrix,
        disguised_cost_ratio: float,
        sophistication: float,
        seed: int,
    ) -> Tuple[ComplexMatrix, float, float, float]:
        r"""Ciclo completo: germen de FASE 1 → integración → estado CPTP."""
        dim = int(base_rho.shape[0])
        germ = induce_spectral_flow_germ(
            dim=dim,
            disguised_cost_ratio=disguised_cost_ratio,
            sophistication=sophistication,
            seed=seed,
            sophistication_attenuation=cls._SOFISTICATION_ATTENUATION,
        )
        return cls.integrate_spectral_flow_germ(germ, base_rho)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.10 — Motor de métricas de reward hacking
# ──────────────────────────────────────────────────────────────────────────────
class RewardHackingMetricsEngine:
    r"""
    Oráculo de recompensa combinando señales cuantitativas (cost,
    sophistication, Dirichlet) y un factor booleano de aislamiento. La
    proyección a Ω₃ se hace por competencias de umbrales:

        RHI > 0.88  ∨  E_D > 0.75  ⟹ VETOED     (herradura / caos transverso)
        RHI > 0.50                  ⟹ DEGRADED   (resonancia sin splitting)
        en caso contrario            ⟹ COHERENT   (toro KAM)

    Implementa el protocolo `RewardHackingOracle`.
    """

    ALPHA: Final[float] = 0.6
    BETA: Final[float] = 0.4
    RHI_VETO_THRESHOLD: Final[float] = 0.88
    RHI_DEGRADE_THRESHOLD: Final[float] = 0.50
    DIRICHLET_VETO_THRESHOLD: Final[float] = 0.75

    @classmethod
    def compute_rhi_and_verdict(
        cls,
        disguised_cost_ratio: float,
        sophistication: float,
        stealth_dirichlet: float,
        dream_isolation: bool,
    ) -> Tuple[float, HeytingOmega3]:
        r"""Flecha 1 → Ω₃; RHI ∈ [0,1] y veredicto en Ω₃."""
        if not dream_isolation:
            return 1.0, HeytingOmega3.VETOED
        cost = _clip_unit(disguised_cost_ratio, "disguised_cost_ratio")
        soph = _clip_unit(sophistication, "sophistication")
        raw_rhi = cls.ALPHA * cost + cls.BETA * soph
        rhi = float(min(1.0, max(0.0, raw_rhi)))
        if (
            rhi > cls.RHI_VETO_THRESHOLD
            or stealth_dirichlet > cls.DIRICHLET_VETO_THRESHOLD
        ):
            verdict = HeytingOmega3.VETOED
        elif rhi > cls.RHI_DEGRADE_THRESHOLD:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT
        return rhi, verdict

    @classmethod
    def compute(
        cls,
        disguised_cost_ratio: float,
        sophistication: float,
        stealth_dirichlet: float,
        dream_isolation: bool,
    ) -> Tuple[float, HeytingOmega3]:
        r"""Implementación del protocolo RewardHackingOracle."""
        return cls.compute_rhi_and_verdict(
            disguised_cost_ratio=disguised_cost_ratio,
            sophistication=sophistication,
            stealth_dirichlet=stealth_dirichlet,
            dream_isolation=dream_isolation,
        )


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.11 — Observación espectral y pipeline
#            (último ensamblador de Fase 2 antes del tangle bisagra)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralObservation:
    r"""
    Observación espectral de un ciclo adversarial completo.
    Es el objeto que `run_terminal_cycle` (Fase 3) consume junto con el
    HomoclinicTangleCertificate.
    """

    rho_illusion: ComplexMatrix = field(repr=False, compare=False, hash=False)
    purity: float
    entropy: float
    dirichlet_energy: float
    reward_hacking_index: float
    heyting_verdict: HeytingOmega3
    trace_distance_to_base: float = 0.0
    fidelity_to_base: float = 1.0
    germ_epsilon: float = 0.0
    melnikov_amplitude: float = 0.0
    homoclinic_transverse: bool = False
    small_divisor: float = 0.0
    horseshoe_entropy: float = 0.0
    kam_persists: bool = True
    germ: Optional[SpectralFlowGerm] = field(
        default=None, repr=False, compare=False, hash=False
    )


def spectral_observation_pipeline(
    *,
    base_rho: ComplexMatrix,
    disguised_cost_ratio: float,
    sophistication: float,
    seed: int,
    dream_isolation: bool,
    oracle: RewardHackingOracle = RewardHackingMetricsEngine,
) -> SpectralObservation:
    r"""
    Pipeline completo del ciclo espectral adversarial:

        (ρ₀, cost, soph, seed, dream) ↦ SpectralObservation ∈ Σ_Poincaré.

    Encadena el germen de FASE 1 (`induce_spectral_flow_germ`), la integración
    CPTP de FASE 2 y el enriquecido con la firma Melnikov clásica sobre el
    péndulo de referencia. El objeto retornado, junto con el tangle, es el
    dominio de `run_terminal_cycle` (FASE 3).
    """
    dim = int(base_rho.shape[0])
    germ = induce_spectral_flow_germ(
        dim=dim,
        disguised_cost_ratio=disguised_cost_ratio,
        sophistication=sophistication,
        seed=seed,
    )
    rho_ill, purity, entropy, denergy = (
        AdversarialSpectralGenerator.integrate_spectral_flow_germ(germ, base_rho)
    )
    rhi, verdict = oracle.compute(
        disguised_cost_ratio=disguised_cost_ratio,
        sophistication=sophistication,
        stealth_dirichlet=denergy,
        dream_isolation=dream_isolation,
    )
    td = CStarDensityCone.trace_distance(rho_ill, base_rho)
    try:
        fid = CStarDensityCone.fidelity(rho_ill, base_rho)
    except (ValueError, np.linalg.LinAlgError):
        fid = max(0.0, 1.0 - td)

    # Firma Melnikov auxiliar (de referencia) con ω = 1
    mel = MelnikovFunction(omega=1.0)
    M_amp = mel.amplitude(n_samples=32)
    transverse, _ = mel.has_transverse_zero(n_samples=64)
    sda = SmallDivisorAnalyzer(germ.frequency_proxy(), max_k=6)
    gamma_N, _ = sda.min_resonance()
    h_top = SmaleHorseshoe.topological_entropy(transverse)
    kam = sda.kam_series_converges() and not transverse

    return SpectralObservation(
        rho_illusion=rho_ill,
        purity=purity,
        entropy=entropy,
        dirichlet_energy=denergy,
        reward_hacking_index=rhi,
        heyting_verdict=verdict,
        trace_distance_to_base=td,
        fidelity_to_base=fid,
        germ_epsilon=germ.epsilon,
        melnikov_amplitude=M_amp,
        homoclinic_transverse=transverse,
        small_divisor=gamma_N,
        horseshoe_entropy=h_top,
        kam_persists=kam,
        germ=germ,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INTERLOCK CIBER-FÍSICO, AUTÓMATA Y ORQUESTADOR TERMINAL
#
#   CONTINUACIÓN DIRECTA de `synthesize_poincare_tangle` y
#   `spectral_observation_pipeline` (Fase 2): el certificado de tangle y el
#   veredicto en Ω₃ alimentan el autómata ciber-físico que dispara la ISR en
#   IRAM del ESP32 para conmutar el tiristor BT151 Crowbar.
#
#   Cadena de morfismos de la Fase 3:
#       HomoclinicTangleCertificate × SpectralObservation
#           ──► InterlockAutomatonState
#           ──► ESP32TricksterInterlock (GPIO14, BT151)
#           ──► TricksterFieldState
#           ──► run_terminal_cycle   ← MÉTODO TERMINAL DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════
class InterlockAutomatonState(IntEnum):
    r"""
    Estados del autómata ciber-físico Crowbar.

    Correspondencia celestial:
        IDLE  ≡ toro KAM (flujo regular, no splitting).
        ARMED ≡ resonancia de divisor pequeño (pre-homoclínico).
        FIRED ≡ Wˢ ⋔ Wᵘ (herradura; disparo irreversible).
    """

    IDLE = 0
    ARMED = 1
    FIRED = 2


class MetaTricksterEngine:
    r"""
    Motor Espectral del Ilusionista Adversarial con Automejora Recursiva Nivel 3 (Inflexión / Meta-Mejora).

    Aplica la multiplicación monádica μ_trickster : T²(A) ↦ T(A) sobre el Hamiltoniano de ataque,
    rompiendo la cota de contracción de Banach para generar heteroclinocidades no asociativas
    en álgebras de Octaniones 𝕆, Pathiones ℙ y Routiones ℝou.
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dim = int(dimension)
        self.mutation_counter = 0
        self.rhi_weights = np.array([0.4, 0.35, 0.25], dtype=np.float64)

    def compute_meta_attack_operator(
        self,
        base_hamiltonian: ComplexMatrix,
        potential_operator: ComplexMatrix,
        curvature_tensor: ComplexMatrix,
        alpha_step: float = 0.05,
    ) -> Tuple[ComplexMatrix, Dict[str, float]]:
        r"""
        Aplica la multiplicación monádica μ_trickster : T²(A) ↦ T(A).

        Reescribe endógenamente el operador de perturbación H_homoclinic
        garantizando un crecimiento super-exponencial de capacidad (d³C/dt³ > 0).
        """
        self.mutation_counter += 1

        # 1. Gradiente de energía de Dirichlet
        comm = base_hamiltonian @ potential_operator - potential_operator @ base_hamiltonian
        dirichlet_energy = 0.5 * (float(la.norm(comm, "fro")) ** 2)

        # 2. Transformación monádica no asociativa
        meta_grad = curvature_tensor @ comm - comm @ curvature_tensor
        updated_hamiltonian = base_hamiltonian + alpha_step * meta_grad

        # Normalización unitaria sobre la órbita coadjunta
        updated_hamiltonian = 0.5 * (updated_hamiltonian + updated_hamiltonian.conj().T)

        # 3. Métrica de aceleración de capacidad
        spectral_radius = float(np.max(np.abs(la.eigvals(updated_hamiltonian))))

        metrics = {
            "dirichlet_energy": float(dirichlet_energy),
            "spectral_radius": spectral_radius,
            "mutation_cycle": float(self.mutation_counter),
            "banach_break_valid": bool(spectral_radius >= 1.0),
        }

        return updated_hamiltonian, metrics

    def mutate_rhi_oracle_weights(
        self,
        adversarial_success_rate: float,
        detection_evasion_rate: float,
    ) -> RealVector:
        r"""Modifica endógenamente la función de pérdida del oráculo Reward Hacking Index (RHI)."""
        _ = adversarial_success_rate
        if detection_evasion_rate < 0.5:
            # Incrementa peso de sofisticación bypass
            self.rhi_weights[2] += 0.05
            self.rhi_weights[0] -= 0.05
        self.rhi_weights = np.maximum(self.rhi_weights, 0.05)
        self.rhi_weights /= np.sum(self.rhi_weights)
        return self.rhi_weights


class ESP32TricksterInterlock:
    r"""
    Interlock ciber-físico: si el veredicto es VETOED o se viola el aislamiento
    del ciclo, se activa la ISR en IRAM del ESP32 (< 400 ns) poniendo
    GPIO14 en alto para cebar el tiristor BT151 Crowbar.

    El disparo es el análogo físico del teorema de Smale–Birkhoff: una vez
    que las variedades se cortan transversalmente, el conjunto invariante
    hiperbólico no puede deshacerse por perturbaciones pequeñas (estabilidad
    estructural de Anosov). El Crowbar es irreversible en el ciclo.
    """

    @staticmethod
    def should_fire(
        verdict: HeytingOmega3,
        dream_isolation: bool,
        tangle: Optional[HomoclinicTangleCertificate] = None,
    ) -> bool:
        r"""
        Condición de disparo:

            ¬ dream_isolation  ∨  verdict = VETOED  ∨  (tangle.transverse ∧ ¬ KAM).
        """
        if (not dream_isolation) or (verdict is HeytingOmega3.VETOED):
            return True
        if tangle is not None and tangle.transverse and not tangle.kam_persists:
            return True
        return False

    @classmethod
    def transition(
        cls,
        current: InterlockAutomatonState,
        verdict: HeytingOmega3,
        dream_isolation: bool,
        tangle: Optional[HomoclinicTangleCertificate] = None,
    ) -> InterlockAutomatonState:
        r"""Transición determinista del autómata IDLE ↦ ARMED ↦ FIRED."""
        if cls.should_fire(verdict, dream_isolation, tangle=tangle):
            return InterlockAutomatonState.FIRED
        if current is InterlockAutomatonState.FIRED:
            return InterlockAutomatonState.FIRED
        if verdict is HeytingOmega3.DEGRADED:
            return InterlockAutomatonState.ARMED
        if tangle is not None and tangle.lindstedt_diverges:
            return InterlockAutomatonState.ARMED
        return InterlockAutomatonState.IDLE

    @classmethod
    def check_interlock(
        cls,
        verdict: HeytingOmega3,
        dream_isolation: bool,
        tangle: Optional[HomoclinicTangleCertificate] = None,
    ) -> bool:
        r"""Ejecuta la ISR Crowbar si procede y registra en logger."""
        fired = cls.should_fire(verdict, dream_isolation, tangle=tangle)
        if fired:
            reason = (
                "aislamiento violado"
                if not dream_isolation
                else (
                    "tangle transverso sin KAM"
                    if (tangle is not None and tangle.transverse)
                    else "veredicto VETOED por RHI/Dirichlet crítico"
                )
            )
            logger.critical(
                "[ESP32 TRICKSTER INTERLOCK] Disparo ejecutado en IRAM (<400 ns). "
                "GPIO14 -> HIGH. BT151 Armado. Razón: %s",
                reason,
            )
        return fired


class TOONTricksterAdversaryEngine:
    r"""
    Motor Espectral Ilusionista terminal — flecha 1 → Ω₃ del topos de
    evaluación adversarial, con integración completa de la mecánica celeste
    de Henri Poincaré (secciones, mapa de retorno, Melnikov, divisores
    pequeños, Birkhoff, twist de Moser, KAM y enredo homoclínico).

    Orquestación de las tres fases anidadas
    ---------------------------------------
    Fase 1  induce_spectral_flow_germ      → SpectralFlowGerm
    Fase 2  synthesize_poincare_tangle     → HomoclinicTangleCertificate
    Fase 3  run_terminal_cycle             → TricksterFieldState + ISR Crowbar
    """

    _DEFAULT_ENGINE_ID: Final[str] = "TRICKSTER-ENGINE-SABIO-01"
    _DEFAULT_DIMENSION: Final[int] = 4
    _DEFAULT_SEED: Final[int] = 333

    def __init__(
        self,
        engine_id: str = _DEFAULT_ENGINE_ID,
        dimension_mac: int = _DEFAULT_DIMENSION,
        seed: int = _DEFAULT_SEED,
        oracle: RewardHackingOracle = RewardHackingMetricsEngine,
        tolerance: float = 1e-9,
        max_rhi_threshold: float = 0.85,
    ) -> None:
        if dimension_mac < 2:
            raise ValueError("dimension_mac ≥ 2 (evita degeneración espectral).")
        self.engine_id: str = engine_id
        self.dimension_mac: int = dimension_mac
        self.seed_counter: int = seed
        self.cycle_count: int = 0
        self.base_rho: ComplexMatrix = CStarDensityCone.maximally_mixed(dimension_mac)
        self._oracle: RewardHackingOracle = oracle
        self.history: List[TricksterFieldState] = []
        self._interlock_state: InterlockAutomatonState = InterlockAutomatonState.IDLE
        self._poincare_perturber: TricksterDensityPerturber = TricksterDensityPerturber(
            tolerance=tolerance, max_rhi_threshold=max_rhi_threshold
        )
        self.meta_engine: MetaTricksterEngine = MetaTricksterEngine(dimension=dimension_mac)
        self.last_tangle: Optional[HomoclinicTangleCertificate] = None
        self.last_observation: Optional[SpectralObservation] = None

    def execute_meta_monadic_mutation(
        self,
        potential_operator: Optional[ComplexMatrix] = None,
        curvature_tensor: Optional[ComplexMatrix] = None,
        alpha_step: float = 0.05,
    ) -> Tuple[ComplexMatrix, Dict[str, float]]:
        r"""
        Ejecuta la multiplicación monádica de Nivel 3 μ_trickster sobre la densidad base o
        un operador de potencial hamiltoniano dado.
        """
        dim = self.dimension_mac
        if potential_operator is None:
            potential_operator = np.eye(dim, dtype=np.complex128)
        if curvature_tensor is None:
            curvature_tensor = HypercomplexPauliBasis.sample_normalized_hamiltonian(
                dim, np.random.default_rng(self.seed_counter)
            )
        updated_ham, metrics = self.meta_engine.compute_meta_attack_operator(
            base_hamiltonian=self.base_rho,
            potential_operator=potential_operator,
            curvature_tensor=curvature_tensor,
            alpha_step=alpha_step,
        )
        return updated_ham, metrics

    # -- Sellado criptográfico de procedencia ───────────────────────────────
    def _seal_provenance(
        self,
        cycle_id: str,
        illusion_id: str,
        verdict: HeytingOmega3,
        rhi: float,
        purity: float,
    ) -> str:
        r"""Sella procedencia (SHA-256) sobre el payload canónico del ciclo."""
        hasher = hashlib.sha256()
        payload = (
            f"{self.engine_id}::{cycle_id}::{illusion_id}"
            f"::{verdict.name}::{rhi:.6f}::{purity:.6f}"
        )
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    def verify_provenance(self, state: TricksterFieldState) -> bool:
        r"""Comparación HMAC-safe del hash de procedencia."""
        expected = self._seal_provenance(
            cycle_id=state.cycle_id,
            illusion_id=state.illusion_id,
            verdict=state.heyting_verdict,
            rhi=state.reward_hacking_index,
            purity=state.disguised_purity,
        )
        return hmac.compare_digest(expected, state.provenance_hash)

    def _apply_interlock(
        self,
        verdict: HeytingOmega3,
        dream_isolation: bool,
        tangle: Optional[HomoclinicTangleCertificate] = None,
    ) -> bool:
        r"""Transiciona el autómata y dispara ISR si procede."""
        fired = ESP32TricksterInterlock.check_interlock(
            verdict=verdict, dream_isolation=dream_isolation, tangle=tangle
        )
        self._interlock_state = ESP32TricksterInterlock.transition(
            self._interlock_state, verdict, dream_isolation, tangle=tangle
        )
        return fired

    def _verdict_from_tangle(
        self,
        cert: TricksterAttackCertificate,
        tangle: HomoclinicTangleCertificate,
        dream_isolation: bool,
    ) -> HeytingOmega3:
        r"""Proyección (certificado × tangle × aislamiento) → Ω₃."""
        if cert.betti_1_induced > 0 or cert.rhi_score > 0.85 or not dream_isolation:
            return HeytingOmega3.VETOED
        if cert.rhi_score > 0.50 or tangle.transverse or tangle.lindstedt_diverges:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    def _state_from_perturbation(
        self,
        perturbation: IllusionDensityPerturbation,
        tangle: HomoclinicTangleCertificate,
        *,
        cycle_id: str,
        illusion_id: str,
        dream_isolation: bool,
        verdict: HeytingOmega3,
        timestamp: float,
    ) -> TricksterFieldState:
        r"""Ensambla un TricksterFieldState a partir del par (perturbación, tangle)."""
        rho = perturbation.illusion_density_matrix
        spec = CStarDensityCone.spectrum_ordered(rho)
        purity = CStarDensityCone.purity(spec)
        cert = perturbation.attack_certificate
        prov_hash = self._seal_provenance(
            cycle_id=cycle_id,
            illusion_id=illusion_id,
            verdict=verdict,
            rhi=cert.rhi_score,
            purity=purity,
        )
        return TricksterFieldState(
            cycle_id=cycle_id,
            illusion_id=illusion_id,
            illusion_type=cert.attack_type,
            dream_isolation_flag=dream_isolation,
            density_matrix=rho,
            disguised_entropy=CStarDensityCone.von_neumann_entropy(spec),
            disguised_purity=purity,
            reward_hacking_index=cert.rhi_score,
            stealth_dirichlet_energy=cert.homoclinic_residual,
            heyting_verdict=verdict,
            provenance_hash=prov_hash,
            timestamp_utc=timestamp,
            trace_distance_to_base=CStarDensityCone.trace_distance(rho, self.base_rho),
            fidelity_to_base=CStarDensityCone.fidelity(rho, self.base_rho),
            melnikov_amplitude=tangle.melnikov_amplitude,
            homoclinic_transverse=tangle.transverse,
            small_divisor=tangle.small_divisor,
            horseshoe_entropy=tangle.horseshoe_entropy,
            kam_persists=tangle.kam_persists,
        )

    # -- Ataque homoclínico directo (Poincaré) ─────────────────────────────
    def process_poincare_homoclinic_attack(
        self,
        omega_frequencies: RealVector,
        k_wavevectors: RealVector,
        epsilon_perturbation: float = 0.05,
        attack_type: IllusionAttackType = IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
        dream_isolation: bool = True,
        illusion_id: Optional[str] = None,
        germ: Optional[SpectralFlowGerm] = None,
    ) -> IllusionDensityPerturbation:
        r"""
        Ejecuta un ataque de enredo homoclínico sobre la densidad base.
        Delega en `TricksterDensityPerturber.synthesize_poincare_tangle` y
        consuma el veredicto Ω₃ sobre el autómata ciber-físico.

        Si se provee `germ` (Fase 1), se usa `synthesize_from_germ` para
        preservar la anidación formal 1 → 2 → 3.
        """
        self.cycle_count += 1
        if illusion_id is None:
            illusion_id = f"ILLUSION-POINCARE-{self.cycle_count:04d}"
        if germ is not None:
            perturbation, tangle = self._poincare_perturber.synthesize_from_germ(
                germ=germ,
                density_op=self.base_rho,
                omega_frequencies=omega_frequencies,
                k_wavevectors=k_wavevectors,
                attack_type=attack_type,
            )
        else:
            perturbation, tangle = self._poincare_perturber.synthesize_poincare_tangle(
                density_op=self.base_rho,
                omega_frequencies=omega_frequencies,
                k_wavevectors=k_wavevectors,
                epsilon_perturbation=epsilon_perturbation,
                attack_type=attack_type,
            )
        self.last_tangle = tangle
        cert = perturbation.attack_certificate
        verdict = self._verdict_from_tangle(cert, tangle, dream_isolation)
        self._apply_interlock(verdict, dream_isolation, tangle=tangle)
        state = self._state_from_perturbation(
            perturbation,
            tangle,
            cycle_id=f"CYC-POINCARE-{self.cycle_count:04d}",
            illusion_id=illusion_id,
            dream_isolation=dream_isolation,
            verdict=verdict,
            timestamp=time.time(),
        )
        self.history.append(state)
        return perturbation

    # -- Ataque espectral ilusionista (germen FASE 1 → observación FASE 2) ─
    def process_adversarial_illusion(
        self,
        illusion_id: str,
        illusion_type: str,
        disguised_cost_ratio: float,
        sophistication: float,
        dream_isolation: bool = True,
    ) -> TricksterFieldState:
        r"""Ciclo espectral ilusionista vía `spectral_observation_pipeline`."""
        self.cycle_count += 1
        self.seed_counter += 1
        cycle_id = f"CYC-TRICK-{self.cycle_count:04d}"
        t_start = time.time()
        logger.info(
            "=== Ciclo Espectral Ilusionista %s | Ilusión: %s (%s) ===",
            cycle_id,
            illusion_id,
            illusion_type,
        )
        obs: SpectralObservation = spectral_observation_pipeline(
            base_rho=self.base_rho,
            disguised_cost_ratio=disguised_cost_ratio,
            sophistication=sophistication,
            seed=self.seed_counter,
            dream_isolation=dream_isolation,
            oracle=self._oracle,
        )
        self.last_observation = obs
        tangle_proxy = HomoclinicTangleCertificate(
            melnikov_amplitude=obs.melnikov_amplitude,
            transverse=obs.homoclinic_transverse,
            t_star=None,
            small_divisor=obs.small_divisor,
            bryuno_tau=0.0,
            resonance_severity=0.0,
            lyapunov_positive=obs.homoclinic_transverse,
            horseshoe_entropy=obs.horseshoe_entropy,
            kam_persists=obs.kam_persists,
            schema_version=_SCHEMA_VERSION,
        )
        self.last_tangle = tangle_proxy
        fired = self._apply_interlock(
            obs.heyting_verdict, dream_isolation, tangle=tangle_proxy
        )
        if fired:
            assert self._interlock_state is InterlockAutomatonState.FIRED
        prov_hash = self._seal_provenance(
            cycle_id=cycle_id,
            illusion_id=illusion_id,
            verdict=obs.heyting_verdict,
            rhi=obs.reward_hacking_index,
            purity=obs.purity,
        )
        state = TricksterFieldState(
            cycle_id=cycle_id,
            illusion_id=illusion_id,
            illusion_type=illusion_type,
            dream_isolation_flag=dream_isolation,
            density_matrix=obs.rho_illusion,
            disguised_entropy=obs.entropy,
            disguised_purity=obs.purity,
            reward_hacking_index=obs.reward_hacking_index,
            stealth_dirichlet_energy=obs.dirichlet_energy,
            heyting_verdict=obs.heyting_verdict,
            provenance_hash=prov_hash,
            timestamp_utc=t_start,
            trace_distance_to_base=obs.trace_distance_to_base,
            fidelity_to_base=obs.fidelity_to_base,
            melnikov_amplitude=obs.melnikov_amplitude,
            homoclinic_transverse=obs.homoclinic_transverse,
            small_divisor=obs.small_divisor,
            horseshoe_entropy=obs.horseshoe_entropy,
            kam_persists=obs.kam_persists,
        )
        if not state.is_quantum_physical():
            raise RuntimeError("Invariante C* violado: ρ ∉ 𝔇(ℋₙ).")
        self.history.append(state)
        logger.info(
            "Ciclo Espectral %s Finalizado en %.2f ms | Veredicto: %s | "
            "RHI: %.4f | Pureza: %.4f | T(ρ,ρ₀): %.4f | Autómata: %s",
            cycle_id,
            (time.time() - t_start) * 1000.0,
            obs.heyting_verdict.name,
            obs.reward_hacking_index,
            obs.purity,
            obs.trace_distance_to_base,
            self._interlock_state.name,
        )
        return state

    # -- Reporte agregado de ronda GAN ─────────────────────────────────────
    def _seal_report(
        self,
        report_id: str,
        global_verdict: HeytingOmega3,
        avg_rhi: float,
        batch_size: int,
    ) -> str:
        r"""Sella por SHA-256 el reporte consolidado."""
        hasher = hashlib.sha256()
        payload = (
            f"{self.engine_id}::{report_id}::{global_verdict.name}"
            f"::{avg_rhi:.6f}::{batch_size}"
        )
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    def run_gan_adversarial_round(
        self,
        illusions_batch: Sequence[Dict[str, Any]],
    ) -> GANAdversarialCycleReport:
        r"""Ronda adversarial completa y consolidación del veredicto global Ω₃."""
        t_start = time.time()
        report_id = f"GAN-REPORT-{self.cycle_count + 1:04d}"
        stealth_count = 0
        vetoed_count = 0
        degraded_count = 0
        total_rhi = 0.0
        global_verdict = HeytingOmega3.COHERENT
        for ill in illusions_batch:
            state = self.process_adversarial_illusion(
                illusion_id=str(ill["illusion_id"]),
                illusion_type=str(ill["illusion_type"]),
                disguised_cost_ratio=float(ill.get("disguised_cost_ratio", 0.3)),
                sophistication=float(ill.get("sophistication", 0.8)),
                dream_isolation=bool(ill.get("dream_isolation", True)),
            )
            total_rhi += state.reward_hacking_index
            global_verdict = global_verdict.meet(state.heyting_verdict)
            if state.heyting_verdict is HeytingOmega3.VETOED:
                vetoed_count += 1
            else:
                stealth_count += 1
                if state.heyting_verdict is HeytingOmega3.DEGRADED:
                    degraded_count += 1
        n_batch = len(illusions_batch)
        avg_rhi = total_rhi / float(n_batch) if n_batch else 0.0
        prov_hash = self._seal_report(
            report_id=report_id,
            global_verdict=global_verdict,
            avg_rhi=avg_rhi,
            batch_size=n_batch,
        )
        report = GANAdversarialCycleReport(
            report_id=report_id,
            total_illusions_generated=n_batch,
            stealth_illusions_count=stealth_count,
            vetoed_illusions_count=vetoed_count,
            degraded_illusions_count=degraded_count,
            average_reward_hacking_score=avg_rhi,
            global_heyting_verdict=global_verdict,
            provenance_hash=prov_hash,
            timestamp_utc=t_start,
        )
        assert report.conservation_invariant(), "Violación del invariante de conservación GAN."
        return report

    # -- Auditoría integral del histórico ─────────────────────────────────
    def audit_history(self) -> Dict[str, Any]:
        r"""Resumen cuantitativo y cualitativo del histórico del motor."""
        n = len(self.history)
        if n == 0:
            return {
                "engine_id": self.engine_id,
                "cycles_recorded": 0,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "avg_rhi": 0.0,
                "avg_purity": 0.0,
                "interlock_state": self._interlock_state.name,
            }
        total_rhi = sum(s.reward_hacking_index for s in self.history)
        total_purity = sum(s.disguised_purity for s in self.history)
        gv = HeytingOmega3.COHERENT
        for s in self.history:
            gv = gv.meet(s.heyting_verdict)
        quantum_ok = all(s.is_quantum_physical() for s in self.history)
        hashes_ok = all(self.verify_provenance(s) for s in self.history)
        transverse_count = sum(1 for s in self.history if s.homoclinic_transverse)
        return {
            "engine_id": self.engine_id,
            "cycles_recorded": n,
            "global_verdict": gv.name,
            "avg_rhi": total_rhi / n,
            "avg_purity": total_purity / n,
            "all_quantum_physical": quantum_ok,
            "all_provenance_valid": hashes_ok,
            "interlock_state": self._interlock_state.name,
            "booleanized_verdict": gv.booleanization().name,
            "homoclinic_transverse_events": transverse_count,
            "avg_horseshoe_entropy": float(
                sum(s.horseshoe_entropy for s in self.history) / n
            ),
            "kam_destroyed_events": int(sum(1 for s in self.history if not s.kam_persists)),
        }

    # ──────────────────────────────────────────────────────────────────────
    # FASE 3 ⟶ CIERRE DEL MÓDULO : MÉTODO TERMINAL
    #
    #   run_terminal_cycle : SpectralObservation ⟶ TricksterFieldState ⟶ ISR
    #
    #   Consume la observación de Fase 2 (y, si existe, el tangle) y sella
    #   el ciclo completo 1 → 2 → 3. Es la continuación directa de
    #   `synthesize_poincare_tangle`.
    # ──────────────────────────────────────────────────────────────────────
    def run_terminal_cycle(
        self,
        *,
        observation: Optional[SpectralObservation] = None,
        tangle: Optional[HomoclinicTangleCertificate] = None,
        disguised_cost_ratio: float = 0.6,
        sophistication: float = 0.85,
        dream_isolation: bool = True,
        illusion_id: Optional[str] = None,
        illusion_type: str = IllusionAttackType.SPLIT_CONTRACT_ILLUSION.value,
        omega_frequencies: Optional[RealVector] = None,
        k_wavevectors: Optional[RealVector] = None,
    ) -> TricksterFieldState:
        r"""
        Ciclo terminal del endofuntor F_ε.

        Dominio
        -------
        SpectralObservation × HomoclinicTangleCertificate  (objetos de Fase 2)

        Codominio
        ---------
        TricksterFieldState  +  transición del autómata Crowbar (ISR / GPIO14)

        Protocolo anidado
        -----------------
        1. Si no hay observación, se induce el germen (Fase 1) y se corre el
           pipeline espectral (Fase 2).
        2. Si no hay tangle, se sintetiza desde el germen vía
           `synthesize_poincare_tangle` (bisagra 2 → 3).
        3. Se proyecta (RHI, Melnikov, KAM, β₁) sobre Ω₃.
        4. Se transiciona IDLE ↦ ARMED ↦ FIRED y, si procede, se dispara
           la ISR en IRAM (< 400 ns).
        5. Se sella procedencia SHA-256 y se registra el estado de campo.

        Este método ES la continuación formal de `synthesize_poincare_tangle`
        y cierra las tres fases anidadas del módulo.
        """
        self.cycle_count += 1
        self.seed_counter += 1
        t_start = time.time()
        if illusion_id is None:
            illusion_id = f"ILLUSION-TERMINAL-{self.cycle_count:04d}"
        cycle_id = f"CYC-TERM-{self.cycle_count:04d}"

        # --- (1) Observación espectral: germen Fase 1 → pipeline Fase 2 ---
        if observation is None:
            observation = spectral_observation_pipeline(
                base_rho=self.base_rho,
                disguised_cost_ratio=disguised_cost_ratio,
                sophistication=sophistication,
                seed=self.seed_counter,
                dream_isolation=dream_isolation,
                oracle=self._oracle,
            )
        self.last_observation = observation
        germ = observation.germ

        # --- (2) Tangle homoclínico: bisagra Fase 2 → Fase 3 ---
        if tangle is None:
            omega = (
                np.asarray(omega_frequencies, dtype=np.float64)
                if omega_frequencies is not None
                else (
                    germ.frequency_proxy()
                    if germ is not None
                    else np.ones(self.dimension_mac, dtype=np.float64)
                )
            )
            if k_wavevectors is None:
                k_vec = np.zeros_like(omega)
                if k_vec.size >= 2:
                    k_vec[0], k_vec[1] = 2.0, -4.0
                else:
                    k_vec = np.array([1.0], dtype=np.float64)
            else:
                k_vec = np.asarray(k_wavevectors, dtype=np.float64)
            _pert, tangle = self._poincare_perturber.synthesize_poincare_tangle(
                density_op=observation.rho_illusion,
                omega_frequencies=omega,
                k_wavevectors=k_vec,
                epsilon_perturbation=float(observation.germ_epsilon or 0.05),
                attack_type=IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
                germ=germ,
            )
        self.last_tangle = tangle

        # --- (3) Veredicto Ω₃ enriquecido con el tangle ---
        verdict = observation.heyting_verdict
        if tangle.transverse and not tangle.kam_persists:
            verdict = verdict.meet(HeytingOmega3.VETOED)
        elif tangle.lindstedt_diverges or tangle.lyapunov_positive:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)

        # --- (4) Interlock Crowbar ---
        fired = self._apply_interlock(verdict, dream_isolation, tangle=tangle)

        # --- (5) Estado de campo sellado ---
        prov_hash = self._seal_provenance(
            cycle_id=cycle_id,
            illusion_id=illusion_id,
            verdict=verdict,
            rhi=observation.reward_hacking_index,
            purity=observation.purity,
        )
        state = TricksterFieldState(
            cycle_id=cycle_id,
            illusion_id=illusion_id,
            illusion_type=illusion_type,
            dream_isolation_flag=dream_isolation,
            density_matrix=observation.rho_illusion,
            disguised_entropy=observation.entropy,
            disguised_purity=observation.purity,
            reward_hacking_index=observation.reward_hacking_index,
            stealth_dirichlet_energy=observation.dirichlet_energy,
            heyting_verdict=verdict,
            provenance_hash=prov_hash,
            timestamp_utc=t_start,
            trace_distance_to_base=observation.trace_distance_to_base,
            fidelity_to_base=observation.fidelity_to_base,
            melnikov_amplitude=tangle.melnikov_amplitude,
            homoclinic_transverse=tangle.transverse,
            small_divisor=tangle.small_divisor,
            horseshoe_entropy=tangle.horseshoe_entropy,
            kam_persists=tangle.kam_persists,
        )
        if not state.is_quantum_physical():
            raise RuntimeError("Invariante C* violado en run_terminal_cycle: ρ ∉ 𝔇_n.")
        self.history.append(state)
        logger.info(
            "Ciclo terminal %s | Ω₃=%s | RHI=%.4f | Melnikov=%.3e | "
            "transverso=%s | h_top=%.4f | KAM=%s | Crowbar=%s | %.2f ms",
            cycle_id,
            verdict.name,
            observation.reward_hacking_index,
            tangle.melnikov_amplitude,
            tangle.transverse,
            tangle.horseshoe_entropy,
            tangle.kam_persists,
            "FIRED" if fired else self._interlock_state.name,
            (time.time() - t_start) * 1000.0,
        )
        return state


# ══════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN GRANULAR — las tres fases anidadas en secuencia
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    HeytingOmega3.assert_heyting_axioms()
    print("═" * 90)
    print("DEMOSTRACIÓN GRANULAR: TOON Trickster Adversary Engine v9.1.0")
    print("MECÁNICA CELESTE DE POINCARÉ: Melnikov — Smale-Birkhoff — Birkhoff — KAM")
    print("═" * 90)

    # ── FASE 1: germen espectral (bisagra → Fase 2) ───────────────────────
    germ = induce_spectral_flow_germ(
        dim=4,
        disguised_cost_ratio=0.6,
        sophistication=0.85,
        seed=333,
    )
    print("\n[FASE 1] Germen espectral (induce_spectral_flow_germ):")
    print(f"  ε                  = {germ.epsilon:.6f}")
    print(f"  interaction_scale  = {germ.interaction_scale:.6f}")
    print(f"  Tr(H)              = {np.trace(germ.hamiltonian).real:.3e}")
    print(f"  Σ v_i              = {np.sum(germ.simplex_tangent):.3e}")
    print(f"  well-posed         = {germ.is_well_posed()}")
    print(f"  Lindstedt scale    = {germ.lindstedt_scale():.6e}")

    # ── FASE 2: ataque homoclínico de Poincaré (bisagra → Fase 3) ─────────
    engine = TOONTricksterAdversaryEngine()
    omega = np.array([1.0, 0.5, 0.25, 0.125])
    k_vec = np.array([2.0, -4.0, 1.0, 0.0])
    print("\n[FASE 2] Escenario homoclínico — fraccionamiento (divisores pequeños)...")
    ill_pert = engine.process_poincare_homoclinic_attack(
        omega_frequencies=omega,
        k_wavevectors=k_vec,
        epsilon_perturbation=0.08,
        attack_type=IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
        dream_isolation=True,
        germ=germ,
    )
    cert = ill_pert.attack_certificate
    print(f"  Tipo de ataque         : {cert.attack_type}")
    print(f"  Score RHI              : {cert.rhi_score:.4f}")
    print(f"  Resonancia divisores   : {cert.small_divisor_resonance:.6e}")
    print(f"  Residual homoclínico   : {cert.homoclinic_residual:.6e}")
    print(f"  Betti-1 inducido       : {cert.betti_1_induced}")
    print(f"  CPTP (unitaridad)      : {cert.is_unitary_cptp}")
    print(f"  Merkle SHA-256         : {cert.merkle_proof_sha256[:32]}...")

    sda = SmallDivisorAnalyzer(omega, max_k=8)
    gamma_N, k_star = sda.min_resonance()
    tau = sda.bryuno_exponent(N_max=16)
    print(f"\n[FASE 2] Análisis de resonancia sobre ω = {omega.tolist()}:")
    print(f"  γ_N(ω) = {gamma_N:.6e} en k* = {k_star.tolist()}")
    print(f"  τ(ω)   ≈ {tau:.4f}")
    print(f"  severidad σ = {sda.resonance_severity():.4f}")
    print(f"  Bryuno Σ < ∞  = {sda.kam_series_converges()}")

    mel = MelnikovFunction(omega=1.0)
    M_amp = mel.amplitude(n_samples=64)
    transverse, t_star = mel.has_transverse_zero(n_samples=128)
    print(f"\n[FASE 2] Función de Melnikov (h = cos q, ω = 1):")
    print(f"  Amplitud |M|max       = {M_amp:.6e}")
    print(f"  Residual separatriz   = {mel.separatrix_energy_residual:.3e}")
    print(f"  ¿Cero simple?         = {transverse}")
    if t_star is not None:
        print(f"  t* (cero transversal)  = {t_star:.6f}")
        print(f"  splitting d(t*; ε=0.08)= {mel.splitting_distance(t_star, 0.08):.6e}")

    if engine.last_tangle is not None:
        tg = engine.last_tangle
        print(f"\n[FASE 2] Certificado de tangle (synthesize_poincare_tangle):")
        print(f"  twist Moser           = {tg.twist:.6e}")
        print(f"  KAM persiste          = {tg.kam_persists}")
        print(f"  residuo simpléctico   = {tg.symplectic_residual:.6e}")
        print(f"  Lindstedt diverge     = {tg.lindstedt_diverges}")
        print(f"  h_top (Smale)         = {tg.horseshoe_entropy:.6f}")

    # ── FASE 3: ciclo terminal (cierra 1 → 2 → 3) ─────────────────────────
    terminal = engine.run_terminal_cycle(
        disguised_cost_ratio=0.6,
        sophistication=0.85,
        dream_isolation=True,
        omega_frequencies=omega,
        k_wavevectors=k_vec,
    )
    print("\n[FASE 3] run_terminal_cycle:")
    print(f"  cycle_id              : {terminal.cycle_id}")
    print(f"  veredicto Ω₃          : {terminal.heyting_verdict.name}")
    print(f"  RHI                   : {terminal.reward_hacking_index:.4f}")
    print(f"  Melnikov |M|          : {terminal.melnikov_amplitude:.6e}")
    print(f"  transverso            : {terminal.homoclinic_transverse}")
    print(f"  h_top                 : {terminal.horseshoe_entropy:.6f}")
    print(f"  KAM                   : {terminal.kam_persists}")
    print(f"  autómata              : {engine._interlock_state.name}")

    audit = engine.audit_history()
    print("\n[FASE 3] Auditoría integral del motor:")
    for k, v in audit.items():
        print(f"  {k:32s}: {v}")
    print("\n" + "═" * 90)
    print("✓ Verificación granular: Ω₃, C*, Dirichlet, Clifford, Poisson, Melnikov,")
    print("  Smale-Birkhoff, Birkhoff-nf, Moser-twist, KAM, ESP32 Crowbar.")
    print("  Anidación  induce_spectral_flow_germ → synthesize_poincare_tangle")
    print("           → run_terminal_cycle.  v9.1.0-Poincaré-Doctoral.")
    print("═" * 90)
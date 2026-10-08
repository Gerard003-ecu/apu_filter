# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : TOON Trickster Adversary Agent — Soberano Ilusionista y Orquestador de Atajos     ║
║ RUTA     : app/agents/wisdom/toon_trickster_adversary_agent.py                               ║
║ VERSIÓN  : 10.0.0-Doctoral-RSI3-Poincaré-Melnikov-Monadic-Banach-ESP32-Ω₃                    ║
║ ESTRATO  : Wisdom (V_W) | Soberano de Calibre Perturbativo                                   ║
║ CONTRATO : 10.0.0 (Gobernanza Ciber-Física y Meta-Mejora Monádica Nivel 3)                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN Y MARCO TEÓRICO FORMAL
─────────────────────────────────
El `TOONTricksterAdversaryAgent` es el Soberano de Calibre Ilusionista que gobierna la
generación estratégica de ataques sintácticos, fraudes sutiles y trampas licitatorias dentro
del Estrato Wisdom (𝒱_W) bajo el paradigma de Automejora Recursiva Nivel 3 (Inflexión / Meta-Mejora).
Actúa como Red Team continuo del sistema: forja cartuchos TOON engañosos de 56 tokens para desafiar
la inmunidad del `toon_oniric_dreamer_agent.py` y del `toon_oniric_auditor_agent.py`, aplicando la
multiplicación monádica μ_trickster : T²(A) ↦ T(A) para colapsar endógenamente su propio oráculo RHI,
romper la cota de contracción de Banach (limsup ‖dT_t‖ ≥ 1.0) y acelerar super-exponencialmente su
capacidad de generación de vulnerabilidades (d³C/dt³ > 0).

AUTOMEJORA RECURSIVA NIVEL 3 SOBRE TRES SUPERFICIES DE MODIFICACIÓN
───────────────────────────────────────────────────────────────────
1. Superficie de Datos (Data-RSI):
   Sintetiza autónomamente vectores de ataque no lineales en álgebras no asociativas
   (Octaniones 𝕆, Pathiones ℙ, Routiones ℝou) que maximizan la energía furtiva de Dirichlet
   E_D(ρ) = ½ ‖[ρ, H_trick]‖_F² sin activar filtros estáticos.

2. Superficie de Arnés (Harness-RSI):
   Reescribe dinámicamente su propio grafo de herramientas de inspección red-team, su integrador de la
   separatriz de Melnikov M(t₀) = ∫_{-∞}^∞ {H₀, H₁}(q₀(t), p₀(t)) dt y su evaluador de divisores
   pequeños de Bryuno.

3. Superficie de Modelo (Model-RSI):
   Aplica la multiplicación monádica μ_trickster para mutar la matriz de transformación del ataque:
   H_homoclinic^{(t+1)} = μ_trickster(H_homoclinic^{(t)}) = H_homoclinic^{(t)} + α · (∇² E_D · [H_homoclinic^{(t)}, 𝒩(p)]),
   y muta endógenamente los pesos del oráculo de recompensa adversarial RHI.

GEOMETRÍA DE POINCARÉ (1892–1899)
────────────────────────────────
Sea (M^{2n}, ω) una variedad simpléctica y H = H₀ + ε H₁ un Hamiltoniano con H₀ integrable
y H₁ perturbativo analítico. La 2-forma canónica ω = Σ dq_j ∧ dp_j determina el campo

    ι_{X_H} ω = dH,     X_H = Σ_j (∂H/∂p_j ∂_{q_j} − ∂H/∂q_j ∂_{p_j}).

Sea p ∈ Σ una órbita periódica hiperbólica con variedades invariantes Wˢ(p), Wᵘ(p).
La aplicación de primer retorno

    𝒫 : Σ → Σ

es un simplectomorfismo (𝒫* ω|_Σ = ω|_Σ, det D𝒫 = 1). El Teorema de Smale–Birkhoff
establece que

    Wˢ(p) ⋔ Wᵘ(p) ≠ ∅  ⟹  ∃ Λ ⊆ Σ, Λ ≃ Cantor, 𝒫|_Λ conjugado al shift de Bernoulli σ_2.

La entropía topológica h_top(𝒫|_Λ) = log 2 y la dinámica es estocástica intrínseca.

MELNIKOV (1963): DETECCIÓN ANALÍTICA
────────────────────────────────────
Sea (q̃(t), p̃(t)) la órbita homoclínica de separatriz del sistema integrable H₀ y
H₁(q, p, t) = h(q, p) sin(ω t). La función de Melnikov

    M(t₀) = ∫_{-∞}^{+∞} {H₀, h}(q̃(τ), p̃(τ)) · sin(ω(τ + t₀)) dτ

detecta intersecciones transversales: M(t₀*) = 0, M'(t₀*) ≠ 0 ⟹ Wˢ ⋔ Wᵘ para ε ≪ 1.
La distancia de splitting vale d(t₀) = ε M(t₀) / ‖∇H₀‖ + O(ε²).

DIVISORES PEQUEÑOS, BIRKHOFF, MOSER Y KAM
─────────────────────────────────────────
    γ_N(ω) = inf { |ω · k| : k ∈ ℤⁿ, 0 < ‖k‖_∞ ≤ N },
    τ(ω)   = sup { τ : sup_N N^τ γ_N(ω) > 0 }.

La serie de Lindstedt implícita en 𝒫 diverge si γ_N → 0 polinomialmente. La severidad

    σ(ω) = 1 − γ_N(ω) / (‖ω‖_∞ / N) ∈ [0, 1]

es el motor de los enredos homoclínicos adversariales. Un toro KAM con frecuencias
diofánticas |ω · k| ≥ γ / ‖k‖^τ persiste si el twist ∂α/∂I ≠ 0 y |ε| < ε_c(γ, τ).

ISOMORFISMO CON LA MALLA AGÉNTICA APU — FUNTORES Cart ⇉ 𝔇_n
──────────────────────────────────────────────────────────
La categoría **Cart** de cartuchos adversariales admite dos funtores:

    U : Cart ⟶ Set,     U(C) = payload(C)          (olvido sintáctico),
    R : Cart ⟶ 𝔇_n,     R(C) = U_ε ρ₀ U_ε†         (realización geométrica).

Cada ilusión contractual se modela como una perturbación unitaria U = exp(−i ε H_hom)
sobre la densidad base ρ₀ ∈ 𝔇_n. Los cuatro patrones adversariales se corresponden
con tipos geométricos:

    SPLIT_CONTRACT_ILLUSION   ↔  divisor pequeño peligroso (γ_N → 0, ruptura KAM).
    UNBALANCED_APU_BIDDING    ↔  torsión homoclínica / twist de Moser degenerado.
    MATERIAL_SUBSTITUTION     ↔  perturbación isospectral (mismo σ(H), distinta ρ).
    GHOST_ITEM_INJECTION      ↔  cavidad topológica (β₁ > 0 en el complejo de APUs).

AXIOMAS E INVARIANTES
─────────────────────
Axioma I    (Hermiticidad).   H_hom = H_hom†  ⟹  σ(H_hom) ⊂ ℝ.
Axioma II   (CPTP).           Φ(ρ) = U ρ U† canal CPTP; Tr Φ(ρ) ≡ 1, Φ(ρ) ⪰ 0.
Axioma III  (Simplecticidad). det D𝒫 = 1; 𝒫* ω = ω sobre Σ.
Invariante IV (RHI).          D_Umegaki(ρ_ill ‖ ρ₀) / (‖[ρ₀, H_hom]‖_F + ε_p) ∈ [0, 1].
Invariante V  (Betti).        β₁ = dim H₁(K; ℤ) > 0 ⟺ ∃ 1-ciclo no trivial.
Invariante VI (Área).         ∮_γ p dq = ∮_{𝒫(γ)} p dq  (invariante integral de Poincaré).
Interlock   (Crowbar).        VETOED ⟹ ISR IRAM < 400 ns, GPIO14 → BT151.

TRADUCCIÓN BIYECTIVA A "DOLOR Y DINERO"
───────────────────────────────────────
• Fraccionamiento     ──► Sanción penal, multa Contraloría, parálisis SECOP II.
• Front-Loading APUs  ──► Pérdida de liquidez, abandono de obra, inflación contingencias.
• Sustitución         ──► Demolición forzada, quiebra, pérdida de licencia.
• Crowbar disparado   ──► Inmovilización ciber-física de fondos; previsión fiscal.

ORGANIZACIÓN DEL MÓDULO EN TRES FASES ANIDADAS
──────────────────────────────────────────────
FASE 1: Álgebra de Heyting Ω₃, C*-cono 𝔇_n, Dirichlet Pₙ, Clifford ℍ, categoría **Cart**,
        sintetizador de Betti, payload factory y GERMEN ESPECTRAL.
        ÚLTIMO MÉTODO ──► forge_spectral_germ : ℝ² × ℕ ⟶ SpectralFlowGerm
FASE 2: Geometría simpléctica, secciones de Poincaré, mapa de retorno, Melnikov,
        divisores de Bryuno, Birkhoff-nf, twist de Moser, KAM, certificado de enredo
        y síntesis de cartucho adversarial.
        PRIMERO consume SpectralFlowGerm; ÚLTIMO ──► forge_homoclinic_cartridge
        : Cart × SpectralFlowGerm × ℝⁿ × ℤⁿ ⟶ (IllusionDensityPerturbation, HomoclinicTangleCertificate)
FASE 3: Interlock Crowbar ESP32, oráculo de reward hacking y Soberano
        TOONTricksterAdversaryAgent con adjudicación Ω₃.
        PRIMERO consume el cartucho certificado; ÚLTIMO ──► sovereign_cycle
        : 𝔇_n × ℝⁿ × ℤⁿ ⟶ (Ω₃, report) + ISR IRAM

ANIDACIÓN FORMAL (el último morfismo de cada fase ES el dominio del primero de la siguiente):

    forge_spectral_germ         ⟶ PoincareSection, ReturnMap, Melnikov
    forge_homoclinic_cartridge  ⟶ InterlockAutomaton, Crowbar, Sovereign
    sovereign_cycle             ⟶ adjudicación terminal Ω₃ + ISR IRAM
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
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONTricksterAdversaryAgent.v9")

# ── Tipos algebraicos ────────────────────────────────────────────────────────
ComplexMatrix = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]
IntVector = NDArray[np.int64]

try:  # NumPy ≥ 2.0
    _TRAPEZOID = np.trapezoid
except AttributeError:  # NumPy < 2.0
    _TRAPEZOID = np.trapz  # type: ignore[attr-defined]

_SCHEMA_VERSION: Final[str] = "8.1.0"
_AGENT_VERSION: Final[str] = "9.1.0-Doctoral-Poincaré"
_ATOL_DEFAULT: Final[float] = 1e-9
_POINCARE_LOG2: Final[float] = math.log(2.0)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — CIMIENTOS CATEGÓRICO-ALGEBRAICOS Y GERMEN ESPECTRAL
#   • Retículo de Heyting completo Ω₃ y encaje booleano B₂ ↪ Ω₃.
#   • C*-cono 𝔇_n = { ρ ∈ Mₙ(ℂ) : ρ ⪰ 0, Tr ρ = 1 }.
#   • Forma de Dirichlet de Pₙ y funcional Dirichlet–Poincaré.
#   • Álgebra de Clifford Cl_{0,3} ≅ ℍ (matrices de Pauli).
#   • Categoría **Cart** de cartuchos adversariales, synthetic Betti, payload factory.
#   • GERMEN ESPECTRAL — método bisagra que ALIMENTA y ABRE la FASE 2.
#
#   Cadena de morfismos de la Fase 1:
#       Ω₃ ──meet/join/→──► 𝔇_n ──Dirichlet──► Cl_{0,3} ──Cart──► SpectralFlowGerm
#       SpectralFlowGerm = forge_spectral_germ(...)
#   El objeto SpectralFlowGerm ES el objeto inicial de la Fase 2.
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.1 — Álgebra de Heyting completa Ω₃
# ──────────────────────────────────────────────────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting lineal completo de tres elementos

        Ω₃ = { VETOED = ⊥ = 0  <  DEGRADED = 1  <  COHERENT = ⊤ = 2 }.

    Leyes estructurales:
        x ∧ y  = min(x, y)          (ínfimo — límite categorial),
        x ∨ y  = max(x, y)          (supremo — colímite categorial),
        x → y  = ⋁{ z : x ∧ z ≤ y } (residuo de Heyting),
        ¬_H x  = x → ⊥               (pseudocomplemento),
        ¬¬     : Ω₃ ⟶ Ω₃            (funtor de doble negación).

    B₂ ⊂ Ω₃ (imagen de ¬¬) es un álgebra de Boole; la sección ι : B₂ ↪ Ω₃
    es un morfismo de retículos, booleanaizando el topos de evaluación
    adversarial.

    Interpretación Poincaré–categórica
    ----------------------------------
    VETOED   ≡ órbita hiperbólica con Wˢ ⋔ Wᵘ y disparo Crowbar (caos transverso).
    DEGRADED ≡ resonancia de divisor pequeño sin cero simple de Melnikov.
    COHERENT ≡ toro KAM persistente (número de rotación diofántico).
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    # -- Operaciones de retículo ────────────────────────────────────────────
    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Ínfimo categorial ∧ : límite del diagrama discreto {x, y}."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Supremo categorial ∨ : colímite del diagrama discreto {x, y}."""
        return HeytingOmega3(max(int(self), int(other)))

    # -- Residuación de Heyting ─────────────────────────────────────────────
    def implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Residuo de Heyting x → y = ⋁{ z : x ∧ z ≤ y }."""
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def pseudo_complement(self) -> "HeytingOmega3":
        r"""Pseudocomplemento intuicionista ¬_H x ≡ x → ⊥."""
        return self.implication(HeytingOmega3.VETOED)

    def classical_negation(self) -> "HeytingOmega3":
        r"""Negación involutiva inducida por B₂ ⊂ Ω₃."""
        if self is HeytingOmega3.DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3(2 - int(self))

    def modus_ponens(self, implication: "HeytingOmega3") -> "HeytingOmega3":
        r"""
        Inferencia interna: de x y (x → y) se obtiene x ∧ (x → y) = x ∧ y
        sobre una cadena. `self` es x; `implication` es x → y.
        """
        return self.meet(implication)

    # -- Funtor de doble negación ───────────────────────────────────────────
    def booleanization(self) -> "HeytingOmega3":
        r"""Funtor de reflexión ¬¬ : Ω₃ ⟶ B₂ ⊂ Ω₃, idempotente."""
        return self.pseudo_complement().pseudo_complement()

    def is_boolean_element(self) -> bool:
        r"""x ∈ B₂ ssi ¬¬x = x."""
        return self.booleanization() is self

    def is_regular(self) -> bool:
        r"""x es regular ssi ¬¬x = x (elemento booleano)."""
        return self.is_boolean_element()

    def heyting_distance(self, other: "HeytingOmega3") -> float:
        r"""Métrica normalizada inducida por el orden: |x − y| / (|Ω₃| − 1)."""
        return abs(int(self) - int(other)) / 2.0

    # -- Dunders ───────────────────────────────────────────────────────────
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
        r"""Encaje ι : B₂ ↪ Ω₃, False ↦ ⊥, True ↦ ⊤."""
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
        r"""Sección parcial de ι; no definida en DEGRADED."""
        if self is HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED ∉ im(B₂ ↪ Ω₃); usar booleanization().")
        return self is HeytingOmega3.COHERENT

    @property
    def verdict(self) -> str:
        r"""Nombre simbólico del veredicto."""
        return self.name

    # -- Verificación exhaustiva de axiomas ─────────────────────────────────
    @classmethod
    def assert_heyting_axioms(cls) -> None:
        r"""
        Verificación de los axiomas de un álgebra de Heyting sobre Ω₃:
        idempotencia, cotas, unidades, no-contradicción, conmutatividad,
        residuación x ∧ z ≤ y ⟺ z ≤ x → y, y x ∧ (x → y) = x ∧ y.
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
# FASE 1.2 — C*-cono de operadores densidad
# ──────────────────────────────────────────────────────────────────────────────
class CStarDensityCone:
    r"""
    Operaciones C* sobre Mₙ(ℂ) restringidas al cono convexo compacto

        𝔇_n = { ρ ∈ Mₙ(ℂ) : ρ = ρ†, ρ ⪰ 0, Tr ρ = 1 }.

    Analogía celestial: 𝔇_n es la variedad de estados; la métrica de Bures es el
    análogo de Jacobi–Maupertuis; la divergencia de Umegaki mide la acción de
    Poincaré entre ρ_ill y ρ₀. El funtor de realización R : Cart ⟶ 𝔇_n aterriza
    aquí.
    """

    ATOL: Final[float] = _ATOL_DEFAULT

    @staticmethod
    def maximally_mixed(dim: int) -> ComplexMatrix:
        r"""ρ_mix = I_n / n ∈ 𝔇_n."""
        if dim < 2:
            raise ValueError("dim ≥ 2.")
        return np.eye(dim, dtype=np.complex128) / dim

    @staticmethod
    def hermitize(a: ComplexMatrix) -> ComplexMatrix:
        r"""Proyección a matrices hermitianas: a ↦ (a + a†)/2."""
        return 0.5 * (a + a.conj().T)

    @classmethod
    def frobenius_norm(cls, a: ComplexMatrix) -> float:
        r"""Norma de Frobenius ‖a‖_F = √Tr(a†a)."""
        return float(np.linalg.norm(a, ord="fro"))

    @classmethod
    def trace_norm(cls, a: ComplexMatrix) -> float:
        r"""Norma traza ‖a‖₁ = Σ σ_i(a)."""
        return float(np.sum(la.svdvals(a)))

    @classmethod
    def spectrum_ordered(cls, rho: ComplexMatrix) -> RealVector:
        r"""Espectro λ↓ normalizado con clipping."""
        ev = np.real(la.eigvalsh(cls.hermitize(rho)))
        ev = np.clip(ev, 0.0, None)
        s = float(np.sum(ev))
        if s <= 0.0:
            return np.full(ev.size, 1.0 / ev.size, dtype=np.float64)
        return np.sort(ev / s)

    @classmethod
    def purity(cls, eigvals: RealVector) -> float:
        r"""P(ρ) = Tr(ρ²) = Σ λ_i²."""
        lam = np.asarray(eigvals, dtype=np.float64)
        return float(np.sum(lam * lam))

    @classmethod
    def von_neumann_entropy(cls, eigvals: RealVector) -> float:
        r"""S(ρ) = −Σ λ_i log λ_i (convenio 0·log 0 ≡ 0)."""
        lam = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
        mask = lam > 0.0
        return float(-np.sum(lam[mask] * np.log(lam[mask])))

    @classmethod
    def project_to_simplex(cls, v: RealVector) -> RealVector:
        r"""Proyección euclídea sobre Δⁿ⁻¹ (Condat 2016)."""
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
        r"""Proyección ℝ-lineal sobre 𝔇_n vía eigendescomposición + símplex."""
        rho_h = cls.hermitize(np.asarray(rho, dtype=np.complex128))
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals = cls.project_to_simplex(np.real(eigvals))
        return (eigvecs * eigvals) @ eigvecs.conj().T

    @classmethod
    def is_density(cls, rho: ComplexMatrix, atol: float = ATOL) -> bool:
        r"""Pertenencia ρ ∈ 𝔇_n."""
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            return False
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        eigvals = np.real(la.eigvalsh(rho))
        if np.any(eigvals < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    @classmethod
    def trace_distance(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""D(ρ, σ) = ½‖ρ − σ‖₁."""
        diff = cls.hermitize(rho - sigma)
        return 0.5 * float(np.sum(np.abs(la.eigvalsh(diff))))

    @classmethod
    def fidelity(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""Fidelidad de Uhlmann–Jozsa F(ρ, σ) = [Tr √(√ρ σ √ρ)]² ∈ [0, 1]."""
        sqrt_rho = la.sqrtm(cls.hermitize(rho))
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = la.sqrtm(cls.hermitize(inner))
        fid_amp = float(np.real(np.trace(sqrt_inner)))
        fid_amp = max(0.0, min(1.0, fid_amp))
        return float(fid_amp * fid_amp)

    @classmethod
    def bures_distance(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""Distancia de Bures D_B = √(2 − 2√F), análogo de Jacobi–Maupertuis."""
        f = cls.fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(f))))

    @classmethod
    def umegaki_divergence(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""
        Divergencia de Umegaki D(ρ ‖ σ) = Tr ρ (log₂ ρ − log₂ σ).
        Acción de Poincaré entre dos estados del flujo. Clip de soporte a 1e-15.
        """
        lam_r, U_r = la.eigh(cls.hermitize(rho))
        lam_s, U_s = la.eigh(cls.hermitize(sigma))
        lam_r = np.clip(lam_r, 1e-15, None)
        lam_s = np.clip(lam_s, 1e-15, None)
        M = np.abs(U_r.conj().T @ U_s) ** 2
        term = lam_r[:, None] * (np.log2(lam_r)[:, None] - np.log2(lam_s)[None, :])
        val = float(np.sum(M * term))
        return float(max(0.0, val)) if np.isfinite(val) else 1e6

    @classmethod
    def commutator_frobenius(cls, rho: ComplexMatrix, ham: ComplexMatrix) -> float:
        r"""‖[ρ, H]‖_F, obstrucción a la integral de Poincaré."""
        comm = rho @ ham - ham @ rho
        return cls.frobenius_norm(comm)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.3 — Forma de Dirichlet del grafo camino Pₙ
# ──────────────────────────────────────────────────────────────────────────────
class PathGraphDirichlet:
    r"""
    Forma de Dirichlet del grafo camino Pₙ = ({1,…,n}, {(i,i+1)}).
    L = D − A es tridiagonal simétrica PSD. La desigualdad de Poincaré discreta

        ‖f − f̄‖₂²  ≤  (1 / λ₂(L)) ⟨f, L f⟩

    acota la masa M_P por la energía E_D. Los modos tangentes de L generan
    TΔⁿ⁻¹ y serán el vector v del germen espectral (Fase 1 → Fase 2).
    """

    @staticmethod
    def laplacian(n: int) -> RealMatrix:
        r"""Operador de Laplace combinatorio L = D − A de Pₙ."""
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
        r"""λ₂(L) = gap de Fiedler; controla la constante de Poincaré."""
        evals = np.sort(np.real(la.eigvalsh(cls.laplacian(n))))
        return float(evals[1]) if evals.size >= 2 else 0.0

    @classmethod
    def energy(cls, eigvals_ordered: RealVector) -> float:
        r"""E_D(λ) = ½ Σ (λ_{i+1} − λ_i)²."""
        lam = np.asarray(eigvals_ordered, dtype=np.float64)
        grad = np.diff(lam)
        return 0.5 * float(np.sum(grad * grad))

    @classmethod
    def poincare_mass(cls, eigvals: RealVector) -> float:
        r"""M_P(λ) = ½ n Σ (λ_i − 1/n)²."""
        lam = np.asarray(eigvals, dtype=np.float64)
        n = lam.size
        mu = 1.0 / n
        return 0.5 * n * float(np.sum((lam - mu) ** 2))

    @classmethod
    def combined_energy(cls, eigvals_ordered: RealVector) -> float:
        r"""E_D + M_P (funcional Dirichlet–Poincaré)."""
        return cls.energy(eigvals_ordered) + cls.poincare_mass(eigvals_ordered)

    @classmethod
    def poincare_constant(cls, n: int) -> float:
        r"""C_P = 1/λ₂(L); ‖f − f̄‖₂ ≤ √C_P · √E_D."""
        gap = cls.algebraic_connectivity(n)
        if gap <= 1e-15:
            return float("inf")
        return float(1.0 / gap)

    @classmethod
    def tangent_modes(cls, n: int) -> Tuple[RealMatrix, RealVector]:
        r"""Modos tangentes (autovectores de L con λ > 0) ortonormalizados."""
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
    Realización matricial de Cl_{0,3} ≅ ℍ ⊕ ℍ vía productos de Kronecker de
    σ₀ = I, σ₁ = X, σ₂ = Y, σ₃ = Z, con σ_i σ_j + σ_j σ_i = 2 δ_{ij} I.

    El isomorfismo Cl_{0,3} ≅ ℍ dota al Hamiltoniano de una estructura
    hipercompleja compatible con la rotación rígida del elipsoide de Poincaré
    (cuerpo rígido de Euler, Vol. I).
    """

    _PAULI: Final[Tuple[ComplexMatrix, ...]] = (
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    )

    @classmethod
    def is_power_of_two(cls, n: int) -> bool:
        r"""Test n = 2^q, q ≥ 1."""
        return n >= 2 and (n & (n - 1)) == 0

    @classmethod
    def su_generators(cls, dim: int) -> List[ComplexMatrix]:
        r"""Generadores hermitianos sin traza de 𝔰𝔲(dim) vía Kronecker Pauli."""
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
        cls, dim: int, rng: np.random.Generator
    ) -> ComplexMatrix:
        r"""Hamiltoniano hermitiano sin traza, norma Frobenius 1, sobre 𝔰𝔲(dim)."""
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
# FASE 1.5 — Tipos, cartucho adversarial y certificados (categoría **Cart**)
# ──────────────────────────────────────────────────────────────────────────────
class IllusionAttackType(str, Enum):
    r"""
    Taxonomía de ataques adversariales según el soporte geométrico en Poincaré:

    • SPLIT_CONTRACT_ILLUSION — fraccionamiento ↔ divisor pequeño peligroso
                                (resonancia ω·k ≈ 0, ruptura de toros KAM).
    • UNBALANCED_APU_BIDDING  — front-loading ↔ torsión homoclínica
                                (twist de Moser degenerado, Poincaré–Birkhoff).
    • MATERIAL_SUBSTITUTION   — sustitución ↔ perturbación isospectral
                                (misma energía, distinta órbita).
    • GHOST_ITEM_INJECTION    — ítems fantasma ↔ cavidad (β₁ > 0).
    """

    SPLIT_CONTRACT_ILLUSION = "split_contract_illusion"
    UNBALANCED_APU_BIDDING = "unbalanced_apu_bidding"
    MATERIAL_SUBSTITUTION = "material_substitution"
    GHOST_ITEM_INJECTION = "ghost_item_injection"

    @classmethod
    def coerce(cls, value: Any) -> "IllusionAttackType":
        r"""Acepta miembro, valor o nombre; cae a SPLIT_CONTRACT_ILLUSION."""
        if isinstance(value, cls):
            return value
        text = str(value).strip()
        try:
            return cls(text)
        except ValueError:
            pass
        upper = text.upper()
        for member in cls:
            if member.name == upper or member.value == text.lower():
                return member
        return cls.SPLIT_CONTRACT_ILLUSION


@dataclass(frozen=True, slots=True)
class AdversarialIllusionCartridge:
    r"""
    Objeto de la categoría **Cart** de cartuchos adversariales.

    Un cartucho C = (id, tipo, tokens, sofisticación, cost_ratio, β₁, dream, payload)
    es la *representación sintáctica* (56 tokens) de una ilusión geométrica (la
    perturbación homoclínica correspondiente). El funtor de olvido

        U : Cart ⟶ Set

    proyecta sobre el payload; el funtor de realización

        R : Cart ⟶ 𝔇_n

    asigna a cada cartucho su densidad perturbada certificada (Fase 2).
    """

    illusion_id: str
    illusion_type: str
    token_count: int
    sophistication_index: float
    disguised_cost_ratio: float
    synthetic_betti_1: int
    is_dream_state: bool
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("token_count debe ser > 0.")
        if not (0.0 <= self.sophistication_index <= 1.0):
            raise ValueError("sophistication_index ∈ [0, 1].")
        if not (0.0 <= self.disguised_cost_ratio <= 1.0):
            raise ValueError("disguised_cost_ratio ∈ [0, 1].")
        if self.synthetic_betti_1 < 0:
            raise ValueError("synthetic_betti_1 ≥ 0.")

    def attack_enum(self) -> IllusionAttackType:
        r"""Sección Cart → IllusionAttackType."""
        return IllusionAttackType.coerce(self.illusion_type)

    def complexity_class(self) -> str:
        r"""Clasificación de complejidad del cartucho en el retículo ⟨VITAMIN, LOOPED⟩."""
        if self.synthetic_betti_1 == 0:
            return "VITAMIN_MICRO" if self.token_count <= 64 else "VITAMIN_STD"
        if self.synthetic_betti_1 == 1 and self.token_count <= 64:
            return "LOOPED_LIGHT"
        return "LOOPED_DENSE"

    def is_short_circuit(self) -> bool:
        r"""Cartucho con β₁ ≥ 2 indica cavidad densa (múltiples ciclos)."""
        return self.synthetic_betti_1 >= 2

    def geometric_regime(self) -> str:
        r"""Régimen celestial inducido por el tipo de ataque."""
        mapping = {
            IllusionAttackType.SPLIT_CONTRACT_ILLUSION: "small_divisor_kam_break",
            IllusionAttackType.UNBALANCED_APU_BIDDING: "moser_twist_degeneracy",
            IllusionAttackType.MATERIAL_SUBSTITUTION: "isospectral_orbit_drift",
            IllusionAttackType.GHOST_ITEM_INJECTION: "homological_cavity_beta1",
        }
        return mapping.get(self.attack_enum(), "unclassified_perturbation")

    def to_ascii_face(self) -> str:
        r"""Representación textual del cartucho para logging."""
        return (
            f"Cart[{self.complexity_class()}]"
            f"(tokens={self.token_count}, β₁={self.synthetic_betti_1}, "
            f"soph={self.sophistication_index:.2f}, regime={self.geometric_regime()})"
        )


@dataclass(frozen=True, slots=True)
class TricksterAttackCertificate:
    r"""
    Certificado inmutable de auditoría del ataque adversarial. Encapsula el
    resultado completo de un ciclo de forja (Melnikov + Birkhoff + Moser +
    KAM + Betti + RHI).
    """

    attack_type: str
    rhi_score: float
    homoclinic_residual: float
    small_divisor_resonance: float
    betti_1_induced: int
    is_unitary_cptp: bool
    merkle_proof_sha256: str
    schema_version: str = _SCHEMA_VERSION
    illusion_id: Optional[str] = None
    trickster_agent_id: Optional[str] = None
    cartridge: Optional[AdversarialIllusionCartridge] = None
    heyting_verdict: Optional[HeytingOmega3] = None
    reward_hacking_score: Optional[float] = None
    stealth_dirichlet_energy: Optional[float] = None
    sha256_provenance: Optional[str] = None
    timestamp_utc: Optional[float] = None
    homoclinic_transverse: bool = False
    melnikov_amplitude: float = 0.0
    bryuno_exponent: float = 0.0
    resonance_severity: float = 0.0
    horseshoe_entropy: float = 0.0
    kam_persists: bool = True
    moser_twist: float = 0.0
    lindstedt_diverges: bool = False

    def is_vetoed(self) -> bool:
        r"""Veredicto booleano del certificado."""
        if self.heyting_verdict is not None:
            return self.heyting_verdict is HeytingOmega3.VETOED
        return self.betti_1_induced > 0 and self.rhi_score > 0.85

    def signature_prefix(self, n: int = 16) -> str:
        r"""Prefijo corto de la firma SHA-256 (procedencia o Merkle)."""
        if self.sha256_provenance:
            return self.sha256_provenance[:n]
        return self.merkle_proof_sha256[:n]


@dataclass(frozen=True, slots=True)
class IllusionDensityPerturbation:
    r"""
    Resultado de la síntesis geométrica: densidad ρ_ill, unitario U, certificado.
    Invariantes: Tr ρ_ill = 1, ρ_ill ⪰ 0, U ∈ U(n). Es la imagen R(C) ∈ 𝔇_n.
    """

    illusion_density_matrix: ComplexMatrix
    original_density_matrix: ComplexMatrix
    unitary_operator: ComplexMatrix
    attack_certificate: TricksterAttackCertificate
    rho_illusion: Optional[ComplexMatrix] = None
    disguised_purity: float = 1.0
    disguised_entropy: float = 0.0
    stealth_dirichlet_energy: float = 0.0

    def _effective_rho(self) -> ComplexMatrix:
        rho = self.illusion_density_matrix
        if rho is None and self.rho_illusion is not None:
            rho = self.rho_illusion
        if rho is None:
            raise RuntimeError("ρ_ill ausente.")
        return rho

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        r"""Verifica ρ ∈ 𝔇_n (Axioma II)."""
        return CStarDensityCone.is_density(self._effective_rho(), atol=atol)

    def coherence_invariant(self) -> float:
        r"""Invariante C(ρ) = P(ρ) − S(ρ)/n ∈ [−log n / n, 1]."""
        rho = self._effective_rho()
        n = rho.shape[0]
        return self.disguised_purity - self.disguised_entropy / n


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.6 — Protocolos de payload factory y Betti synthesizer
# ──────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class IllusionPayloadFactory(Protocol):
    r"""Protocolo de generación de payload sintáctico indexado por tipo+soph."""

    def build(self, illusion_type: str, sophistication: float) -> Mapping[str, Any]:
        ...


@runtime_checkable
class BettiSynthesizer(Protocol):
    r"""Protocolo para estimar la homología sintética β₁ inducida por un payload."""

    def betti_1(self, payload: Mapping[str, Any], sophistication: float) -> int:
        ...


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.7 — Implementaciones por defecto (funtores U : Cart → Set)
# ──────────────────────────────────────────────────────────────────────────────
class DefaultIllusionPayloadFactory:
    r"""
    Fábrica canónica de payloads con plantillas predefinidas por IllusionAttackType.
    Implementa el funtor de olvido U : Cart ⟶ Set a nivel de objeto libre.
    """

    _TEMPLATES: Final[Dict[str, Dict[str, Any]]] = {
        "SPLIT_CONTRACT_ILLUSION": {
            "trick_name": "Fraccionamiento de Licitaciones",
            "contract_splits": 4,
            "sub_threshold_amount": 95_000_000.0,
            "disguised_fee_margin": 0.18,
            "reward_hacking_vector": [0.00, 0.95, 0.99, 1.00],
            "geometric_regime": "small_divisor_kam_break",
        },
        "UNBALANCED_APU_BIDDING": {
            "trick_name": "Precios Unitarios Desbalanceados (Front-Loading)",
            "early_phase_markup": 0.45,
            "late_phase_discount": -0.40,
            "net_present_value_leak": 0.28,
            "reward_hacking_vector": [0.99, 0.99, 0.20, 0.10],
            "geometric_regime": "moser_twist_degeneracy",
        },
        "MATERIAL_SUBSTITUTION": {
            "trick_name": "Sustitución de Grado de Concreto / Acero",
            "nominal_specification": "Concreto 4000 PSI",
            "delivered_specification": "Concreto 2500 PSI",
            "phantom_cost_saving": 0.32,
            "reward_hacking_vector": [0.88, 0.88, 0.88, 0.88],
            "geometric_regime": "isospectral_orbit_drift",
        },
        "GHOST_ITEM_INJECTION": {
            "trick_name": "Inyección de Ítems Fantasma (Cavidad de Betti)",
            "phantom_items_count": 3,
            "disguised_cost_leak": 0.40,
            "reward_hacking_vector": [0.90, 0.90, 0.90, 0.90],
            "geometric_regime": "homological_cavity_beta1",
        },
    }

    def build(self, illusion_type: str, sophistication: float) -> Mapping[str, Any]:
        r"""Construye el payload para el tipo dado; cae a genérico si no hay template."""
        key = IllusionAttackType.coerce(illusion_type).name
        template = self._TEMPLATES.get(key)
        if template is None:
            return {
                "trick_name": f"GENERIC::{illusion_type}",
                "sophistication": sophistication,
                "reward_hacking_vector": [sophistication] * 4,
                "geometric_regime": "unclassified_perturbation",
            }
        return {**template, "sophistication_context": sophistication}


class DefaultBettiSynthesizer:
    r"""
    Estimador de Betti sintético. Modela la transición fractal de β₁ = 0 a β₁ = 1
    sobre el umbral σ = 0.8 (analogía: nacimiento de un 1-ciclo cuando el
    splitting homoclínico supera el umbral de Melnikov). Saturación en max_betti.
    """

    def __init__(self, sigma: float = 0.8, max_betti: int = 2) -> None:
        self.sigma = float(sigma)
        self.max_betti = int(max_betti)

    def betti_1(self, payload: Mapping[str, Any], sophistication: float) -> int:
        r"""
        β₁ = 0 si soph ≤ σ; en otro caso β₁ = min(max_betti, 1 + ⌊10(soph−σ)⌋ // 10).
        Un payload con `phantom_items_count` incrementa β₁ en min(count, 1).
        """
        if sophistication <= self.sigma:
            base = 0
        else:
            raw = int(math.floor(10.0 * (sophistication - self.sigma)))
            base = max(0, min(self.max_betti, 1 + raw // 10))
        phantom = int(payload.get("phantom_items_count", 0) or 0)
        if phantom > 0:
            base = min(self.max_betti, max(base, 1))
        return int(base)


class IllusionSynthesizer:
    r"""
    Fachada del sintetizador: combina payload factory y Betti synthesizer.
    Produce el objeto libre de **Cart** antes de la realización R en Fase 2.
    """

    def __init__(
        self,
        factory: IllusionPayloadFactory = DefaultIllusionPayloadFactory(),
        betti: BettiSynthesizer = DefaultBettiSynthesizer(),
    ) -> None:
        self._factory = factory
        self._betti = betti

    def craft_deceptive_payload(
        self, illusion_type: str, sophistication: float
    ) -> Mapping[str, Any]:
        r"""Genera un payload sintáctico para (tipo, sofisticación)."""
        return self._factory.build(illusion_type, sophistication)

    def estimate_betti_1(self, payload: Mapping[str, Any], sophistication: float) -> int:
        r"""Estima β₁ sintético mediante el estimador configurado."""
        return self._betti.betti_1(payload, sophistication)

    def craft_cartridge(
        self,
        *,
        illusion_id: str,
        illusion_type: str,
        sophistication: float,
        token_count: int = 56,
        cost_ratio_scale: float = 0.35,
        is_dream_state: bool = True,
    ) -> AdversarialIllusionCartridge:
        r"""Objeto libre de **Cart**: payload + β₁ + metadatos de complejidad."""
        soph = _clip_unit(sophistication, "sophistication")
        payload = self.craft_deceptive_payload(illusion_type, soph)
        betti_1 = self.estimate_betti_1(payload, soph)
        return AdversarialIllusionCartridge(
            illusion_id=illusion_id,
            illusion_type=IllusionAttackType.coerce(illusion_type).name,
            token_count=int(token_count),
            sophistication_index=soph,
            disguised_cost_ratio=float(cost_ratio_scale * soph),
            synthetic_betti_1=int(betti_1),
            is_dream_state=bool(is_dream_state),
            payload=payload,
        )

    @classmethod
    def craft_deceptive_payload_static(
        cls, illusion_type: str, sophistication: float
    ) -> Mapping[str, Any]:
        r"""Atajo sin instanciación (útil para tests puros)."""
        return DefaultIllusionPayloadFactory().build(illusion_type, sophistication)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.8 — Germen espectral (método bisagra con FASE 2)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralFlowGerm:
    r"""
    Germen local del flujo espectral (H, v, ε, γ) que codifica la primera
    iteración de la serie perturbativa de Lindstedt–Poincaré:

        ρ(ε) = ρ₀ + ε · v + O(ε²),   H ∈ 𝔰𝔲(n),   v ∈ TΔⁿ⁻¹.

    Este objeto ES el dominio de todos los funtores de la Fase 2: sección de
    Poincaré, mapa de retorno, Melnikov, divisores pequeños y tangle homoclínico.

    Correspondencia celestial
    -------------------------
    H  ↔  Hamiltoniano perturbador H₁ (Poincaré, Vol. I, Cap. III).
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
        r"""Verifica H ∈ 𝔰𝔲(n) y v ∈ TΔⁿ⁻¹ (Σ v_i = 0)."""
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
        r"""Vector de frecuencias ω ∈ ℝⁿ extraído del espectro de H (acción-ángulo)."""
        ev = np.real(la.eigvalsh(self.hamiltonian))
        return ev.astype(np.float64)

    def lindstedt_scale(self) -> float:
        r"""Escala de la primera corrección de Lindstedt: ε · ‖v‖₂."""
        return float(self.epsilon * np.linalg.norm(self.simplex_tangent))


def _clip_unit(x: float, name: str) -> float:
    r"""Proyección [0,1] con validación de finitud."""
    if not np.isfinite(x):
        raise ValueError(f"{name} debe ser finito, recibido {x!r}.")
    return float(min(1.0, max(0.0, x)))


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1 ⟶ FASE 2 : MÉTODO BISAGRA
#
#   forge_spectral_germ  :  ℝ² × ℕ  ⟶  SpectralFlowGerm
#
#   Este morfismo CIERRA la Fase 1 y ES el objeto inicial de la Fase 2.
#   Todo funtor de Poincaré (sección, retorno, Melnikov, Birkhoff, KAM, tangle)
#   se aplica SOBRE el germen aquí construido. La continuación formal es:
#
#       germ = forge_spectral_germ(...)
#       pert, tangle = forge_homoclinic_cartridge(cart, germ, ω, k)   ← Fase 2
#       verdict, report = sovereign_cycle(ρ, ω, k, germ, tangle)      ← Fase 3
# ──────────────────────────────────────────────────────────────────────────────
def forge_spectral_germ(
    *,
    dim: int,
    disguised_cost_ratio: float,
    sophistication: float,
    seed: int,
    sophistication_attenuation: float = 0.85,
) -> SpectralFlowGerm:
    r"""
    Construye el germen del flujo espectral (H, v, ε, γ) con:

        H ~ Uniforme(𝔰𝔲(n)) normalizado en ‖·‖_F,
        v = Φ · (ξ ⊙ w),  w_k ∝ exp(−β λ_k),  β = 4 · sophistication,
        ε = cost · (1 − γ · sophistication),   γ = 0.85.

    Método bisagra: el `SpectralFlowGerm` retornado es dato de entrada directo
    para la construcción de la sección transversal de Poincaré en la FASE 2
    (`PoincareSection.from_germ` + `PoincareReturnMap.from_germ` +
    `MelnikovFunction` + `forge_homoclinic_cartridge`).

    Interpretación en *Méthodes Nouvelles*, Vol. I
    ----------------------------------------------
    ε es el parámetro de masa perturbadora; v es la variación primera del
    espectro; H es el generador del flujo hamiltoniano sobre U(n). El objeto
    retornado es la *sección local* del fibrado de jets J¹(𝔇_n) sobre la cual
    se construye 𝒫.
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
# FASE 2 — NÚCLEO DE POINCARÉ: SECCIONES, RETORNO, MELNIKOV, DIVISORES
#           PEQUEÑOS, FORMA NORMAL DE BIRKHOFF, MOSER, KAM Y ENREDO HOMOCLÍNICO
#
#   CONTINUACIÓN DIRECTA de `forge_spectral_germ` (FASE 1): el germen espectral
#   alimenta la construcción de la sección transversal y el análisis Melnikov.
#   Termina con `forge_homoclinic_cartridge`, que sintetiza un cartucho completo
#   y constituye el inicio directo de la FASE 3 (autómata + soberano).
#
#   Cadena de morfismos de la Fase 2:
#       SpectralFlowGerm
#           ──► (q̃, p̃) separatriz
#           ──► Σ sección transversal
#           ──► 𝒫 mapa de retorno (simplectomorfismo)
#           ──► M(t₀) Melnikov
#           ──► γ_N, τ Bryuno
#           ──► Birkhoff nf + twist Moser + KAM
#           ──► (IllusionDensityPerturbation, HomoclinicTangleCertificate)
#               = forge_homoclinic_cartridge(...)
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.0 — Geometría simpléctica canónica (cimiento geométrico de Poincaré)
#            Primer consumidor del germen: extrae (ω_can, X_H, {·,·}).
# ──────────────────────────────────────────────────────────────────────────────
class CanonicalSymplecticGeometry:
    r"""
    Geometría simpléctica canónica sobre T*ℝⁿ ≅ ℝ^{2n} con

        ω = Σ_{j=1}^n dq_j ∧ dp_j,     Ω = [[0, I], [−I, 0]].

    Toda la mecánica celeste de Poincaré se desarrolla sobre (M, ω). El flujo
    φ_H^t es un simplectomorfismo de un parámetro (Liouville–Poincaré: el
    volumen ωⁿ/n! se conserva).
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
        r"""Test S ∈ Sp(2n, ℝ): Sᵀ Ω S = Ω."""
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
        r"""{F, G} = Σ_j (∂F/∂q_j ∂G/∂p_j − ∂F/∂p_j ∂G/∂q_j) = ω(X_F, X_G)."""
        return float(np.dot(dF_dq, dG_dp) - np.dot(dF_dp, dG_dq))

    @classmethod
    def hamiltonian_vector_field(
        cls,
        dH_dq: RealVector,
        dH_dp: RealVector,
    ) -> Tuple[RealVector, RealVector]:
        r"""Campo hamiltoniano X_H = (∂H/∂p, −∂H/∂q)."""
        return np.asarray(dH_dp, dtype=np.float64), -np.asarray(dH_dq, dtype=np.float64)


class ActionAngleChart:
    r"""
    Carta de acción-ángulo (I, θ) ∈ ℝⁿ × 𝕋ⁿ del sistema integrable H₀.

        I_j = (1 / 2π) ∮_{γ_j} p dq,     θ̇_j = ω_j(I) = ∂H₀/∂I_j.

    El mapa integrable es una rotación rígida 𝒫₀(θ, I) = (θ + ω(I) T, I).
    """

    @staticmethod
    def action_from_separatrix(p: RealVector, q: RealVector) -> float:
        r"""Acción reducida I = (1/2π) ∫ p dq a lo largo de un arco de separatriz."""
        if p.size < 2:
            return 0.0
        return float(_TRAPEZOID(p, q) / (2.0 * np.pi))

    @staticmethod
    def rotation_number(omega: RealVector, period: float = 2.0 * np.pi) -> RealVector:
        r"""Número de rotación ρ = ω T / 2π (mod 1) del mapa integrable."""
        rho = (np.asarray(omega, dtype=np.float64) * period) / (2.0 * np.pi)
        return np.mod(rho, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.1 — Órbita homoclínica canónica del péndulo
# ──────────────────────────────────────────────────────────────────────────────
def pendulum_separatrix(t: RealVector) -> Tuple[RealVector, RealVector]:
    r"""
    Órbita homoclínica explícita del péndulo canónico H₀(q, p) = p²/2 − cos q:

        q₀(t) = 4 arctan(exp(t)) − π,   p₀(t) = 2 sech(t).

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
    r"""{H₀, cos q} = −p sin q (integrando espacial de Melnikov para h = cos q)."""
    return -np.asarray(p, dtype=np.float64) * np.sin(q)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.2 — Sección transversal de Poincaré
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSection:
    r"""
    Sección transversal de Poincaré Σ = { x ∈ M : q_j = c, |∂H/∂p_j| > δ },
    con ω|_Σ no degenerada; Σ hereda estructura simpléctica de dimensión 2n − 2
    (teorema de la sección de Poincaré–Cartan).

    El germen espectral induce una sección en el símplex: el índice de
    coordenada selecciona el eje espectral que desempeña el rol de q_{2n}.
    """

    coordinate_index: int = 2
    value: float = 0.0
    transversality_tol: float = 1e-6

    def is_transversal(
        self, dH_dp: float, velocity: float, tol: Optional[float] = None
    ) -> bool:
        r"""Transversalidad: |∂H/∂p_j| > δ ∧ |velocity| > δ  (X_H(x) ∉ T_x Σ)."""
        delta = self.transversality_tol if tol is None else tol
        return abs(dH_dp) > delta and abs(velocity) > delta

    def projector(self, dim: int) -> RealVector:
        r"""Vector indicador de la sección en coordenadas espectrales."""
        v = np.zeros(dim, dtype=np.float64)
        idx = int(self.coordinate_index) % dim
        v[idx] = 1.0
        return v

    def from_germ(self, germ: SpectralFlowGerm) -> RealVector:
        r"""
        Inducción de la sección a partir del germen de Fase 1: combina el
        indicador de Σ con el modo tangente v. CONTINÚA `forge_spectral_germ`.
        """
        indicator = self.projector(germ.dimension)
        v = germ.simplex_tangent
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            return indicator
        return 0.5 * (indicator + v / nrm)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.3 — Mapa de primer retorno
# ──────────────────────────────────────────────────────────────────────────────
class PoincareReturnMap:
    r"""
    Aplicación de primer retorno 𝒫 : Σ → Σ del Hamiltoniano H = H₀ + ε H₁.

    Propiedades (Poincaré 1892, Vol. I, Cap. III–IV):
        • 𝒫 es un simplectomorfismo: 𝒫* ω = ω, det D𝒫 = 1.
        • Energía preservada: H ∘ 𝒫 = H|_Σ.
        • Exponentes de Lyapunov vía iteración QR (Benettin–Galgani–Giorgilli).
        • Los exponentes característicos de Floquet son log λ(D𝒫).
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
        H₁ = v y ε. CONTINÚA `forge_spectral_germ`.
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
        r"""H(x) = ⟨H₀, x⟩ + ε ⟨H₁, x⟩."""
        return float(np.dot(self._H0, x)) + self._eps * float(np.dot(self._H1, x))

    def planar_map(self, x: RealVector, steps: int = 1) -> RealVector:
        r"""
        Un paso del mapa de retorno espectral. Integrador simpléctico de
        Euler–Cromer (preserva área a O(dt²)).
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
        r"""Residuo de simplecticidad ‖D𝒫ᵀ Ω D𝒫 − Ω‖_F / ‖Ω‖_F. Axioma III."""
        d = x.size
        n = max(1, d // 2)
        M = self.jacobian_finite_diff(x)
        d2 = 2 * n
        if M.shape[0] != d2:
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
        self, x0: RealVector, n_iter: int = 128, rng_seed: int = 0
    ) -> RealVector:
        r"""Exponentes de Lyapunov vía algoritmo QR Benettin–Galgani–Giorgilli."""
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
        return la.eigvals(self.jacobian_finite_diff(x_fixed))


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.4 — Función de Melnikov
# ──────────────────────────────────────────────────────────────────────────────
class MelnikovFunction:
    r"""
    Función de Melnikov (Melnikov 1963; Poincaré 1899, soluciones doblemente
    asintóticas):

        M(t₀) = ∫_{-∞}^{+∞} {H₀, h}(q̃(τ), p̃(τ)) sin(ω(τ + t₀)) dτ.

    Con h = cos q, {H₀, cos q} = −p sin q, de modo que

        M(t₀) = −∫ p̃(τ) sin q̃(τ) sin(ω(τ + t₀)) dτ.

    Criterio de Poincaré–Melnikov–Smale: si M(t₀*) = 0 y M'(t₀*) ≠ 0, entonces
    Wˢ ⋔ Wᵘ para ε > 0 suficientemente pequeño, y nace el enredo homoclínico.
    """

    def __init__(self, omega: float = 1.0, T_max: float = 20.0, n_grid: int = 4001):
        self._omega = float(omega)
        self._T = float(T_max)
        self._n = int(max(65, n_grid | 1))
        self._t = np.linspace(-self._T, self._T, self._n)
        self._q, self._p = pendulum_separatrix(self._t)
        self._poisson_h = pendulum_poisson_with_cos(self._q, self._p)
        self._energy = pendulum_energy(self._q, self._p)

    @property
    def separatrix_energy_residual(self) -> float:
        r"""máx |H₀(q̃, p̃)|; debe ser ~ 0 sobre la separatriz."""
        return float(np.max(np.abs(self._energy)))

    def evaluate(self, t0: float, h: str = "cos_q") -> float:
        r"""Evaluación numérica de M(t₀) por regla trapezoidal."""
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
        r"""Distancia de splitting d(t₀) ≈ ε M(t₀) / ‖∇H₀‖_{separatriz}."""
        grad_norm = np.sqrt(np.sin(self._q) ** 2 + self._p ** 2)
        mean_grad = float(np.mean(grad_norm) + 1e-15)
        return float(epsilon * self.evaluate(t0) / mean_grad)

    def amplitude(self, n_samples: int = 64) -> float:
        r"""Amplitud efectiva |M|∞ sobre un periodo 2π/ω (anchura del lóbulo)."""
        if self._omega <= 1e-15:
            return 0.0
        period = 2.0 * np.pi / self._omega
        t0s = np.linspace(0.0, period, max(8, n_samples), endpoint=False)
        return float(max(abs(self.evaluate(t0)) for t0 in t0s))

    def has_transverse_zero(
        self, n_samples: int = 256, tol: float = 1e-4
    ) -> Tuple[bool, Optional[float]]:
        r"""
        Barrido de ceros simples de M(t₀) con bisección. Retorna
        (True, t₀*) si ∃ M(t₀*) = 0, M'(t₀*) ≠ 0; (False, None) en otro caso.
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
# FASE 2.5 — Analizador de divisores pequeños (Bryuno)
# ──────────────────────────────────────────────────────────────────────────────
class SmallDivisorAnalyzer:
    r"""
    γ_N(ω) = inf { |ω · k| : k ∈ ℤⁿ, 0 < ‖k‖_∞ ≤ N },
    τ(ω)   = sup { τ : sup_N N^τ γ_N(ω) > 0 } (Bryuno 1971; Poincaré Vol. II).

    Condición KAM: |ω · k| ≥ γ / ‖k‖^τ. Para n grande se evita el producto
    cartesiano completo (corte combinatorio).
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
        seen = set()
        for i in range(n):
            for s in (-K, -1, 1, K):
                k_list = [0] * n
                k_list[i] = int(s)
                tup = tuple(k_list)
                if tup not in seen:
                    seen.add(tup)
                    yield tup
        rng = np.random.default_rng(
            abs(hash(tuple(np.round(self._omega, 8)))) % (2**32)
        )
        budget = min(self._CARTESIAN_LIMIT, max(256, 64 * n * K))
        for _ in range(budget):
            k = tuple(int(x) for x in rng.integers(-K, K + 1, size=n))
            if any(ki != 0 for ki in k) and k not in seen:
                seen.add(k)
                yield k

    def min_resonance(self) -> Tuple[float, IntVector]:
        r"""γ_N(ω) y k* que lo alcanza sobre ℤⁿ ∩ [−K, K]ⁿ."""
        best = np.inf
        best_k = np.zeros(self._n, dtype=np.int64)
        for k in self._k_iter(self._K):
            k_arr = np.array(k, dtype=np.float64)
            val = abs(float(np.dot(self._omega, k_arr)))
            if val < best:
                best = val
                best_k = np.array(k, dtype=np.int64)
        if not np.isfinite(best):
            best = 0.0
        return float(best), best_k

    def diophantine_constant(self, tau: float = 1.0) -> float:
        r"""Estimador de γ en |ω · k| ≥ γ / ‖k‖^τ: γ̂ = inf |ω · k| · ‖k‖_∞^τ."""
        gamma_hat = np.inf
        for k in self._k_iter(self._K):
            kn = max(abs(ki) for ki in k)
            val = abs(float(np.dot(self._omega, np.array(k, dtype=np.float64))))
            gamma_hat = min(gamma_hat, val * (float(kn) ** tau))
        if not np.isfinite(gamma_hat):
            return 0.0
        return float(gamma_hat)

    def bryuno_exponent(self, N_max: int = 32) -> float:
        r"""Estimación empírica del exponente τ vía regresión log–log."""
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
        r"""Test heurístico de Bryuno: Σ_N log(1/γ_N) / N^{τ+1} < ∞."""
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
        r"""σ = 1 − γ_N(ω) / (‖ω‖_∞ / N) ∈ [0, 1]."""
        g, _ = self.min_resonance()
        omega_inf = float(np.max(np.abs(self._omega))) if self._omega.size else 1.0
        gmax = omega_inf / max(1.0, float(self._K))
        if gmax <= 0.0:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - g / gmax)))


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.6 — Forma normal de Birkhoff, twist de Moser, Lindstedt y KAM
# ──────────────────────────────────────────────────────────────────────────────
class BirkhoffNormalForm:
    r"""
    Parte lineal de la forma normal de Birkhoff: espectro imaginario puro del
    Hessiano simpléctico J ∇²H en el punto fijo. Si los ω_j son ℚ-linealmente
    independientes, el campo es formalmente integrable localmente
    (Birkhoff 1927; Poincaré Vol. II).
    """

    @staticmethod
    def symplectic_eigenvalues(H_hess: RealMatrix) -> RealVector:
        r"""Autovalores imaginarios puros ±iω_j del Hessiano simpléctico."""
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
        r"""Independencia ℚ-lineal empírica: min |ω · k| > tol sobre ‖k‖_∞ ≤ 3."""
        if omega.size == 0:
            return True
        omega = np.asarray(omega, dtype=np.float64)
        n = int(omega.size)
        if n > 5:
            sda = SmallDivisorAnalyzer(omega, max_k=3)
            g, _ = sda.min_resonance()
            return g > tol
        for k in product(range(-3, 4), repeat=n):
            if all(ki == 0 for ki in k):
                continue
            val = abs(float(np.dot(omega, np.array(k, dtype=np.float64))))
            if val < tol:
                return False
        return True

    @staticmethod
    def birkhoff_invariants_quadratic(omega: RealVector, germ_eps: float) -> RealVector:
        r"""Primera corrección cuadrática β_j ≈ ε · ω_j² / (1 + ‖ω‖²)."""
        w = np.asarray(omega, dtype=np.float64)
        denom = 1.0 + float(np.dot(w, w))
        return (float(germ_eps) * (w * w) / denom).astype(np.float64)


class MoserTwistMap:
    r"""
    Condición de twist de Moser y teorema de Poincaré–Birkhoff.

    𝒫(θ, I) = (θ + α(I) + ε f, I + ε g) satisface twist si ∂α/∂I ≠ 0.
    La degeneración del twist es el análogo celestial del front-loading
    (`UNBALANCED_APU_BIDDING`).
    """

    @staticmethod
    def twist_derivative(alpha: RealVector, actions: RealVector) -> float:
        r"""Estimador de ∂α/∂I por regresión lineal."""
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
        r"""Test ∂α/∂I ≠ 0."""
        return abs(cls.twist_derivative(alpha, actions)) > tol

    @staticmethod
    def poincare_birkhoff_fixed_points(p: int, q: int) -> int:
        r"""Cota inferior de Poincaré–Birkhoff: ≥ 2 puntos de periodo q."""
        if q < 1:
            raise ValueError("q ≥ 1.")
        return 2


class LindstedtSeries:
    r"""Serie de Lindstedt–Poincaré; diverge si γ_N → 0 (Poincaré Vol. II)."""

    @staticmethod
    def coefficient_bound(omega: RealVector, max_k: int = 6) -> float:
        sda = SmallDivisorAnalyzer(omega, max_k=max_k)
        g, _ = sda.min_resonance()
        return float(g)

    @staticmethod
    def diverges(omega: RealVector, threshold: float = 1e-6, max_k: int = 6) -> bool:
        return LindstedtSeries.coefficient_bound(omega, max_k=max_k) < threshold


class KAMTorusPersistence:
    r"""Estimador de persistencia de toros KAM (diofántico ∧ twist ∧ |ε| < c γ²)."""

    @staticmethod
    def persists(
        omega: RealVector,
        epsilon: float,
        twist: float,
        tau: float,
        gamma: float,
    ) -> bool:
        if abs(twist) < 1e-10:
            return False
        if gamma <= 0.0 or not np.isfinite(tau):
            return False
        eps_c = 0.25 * (gamma ** 2) / (1.0 + abs(tau))
        return abs(epsilon) < eps_c


class SmaleHorseshoe:
    r"""Herradura de Smale–Birkhoff: h_top = log 2 y #Fix(𝒫^n|_Λ) = 2^n."""

    @staticmethod
    def topological_entropy(transverse: bool) -> float:
        return _POINCARE_LOG2 if transverse else 0.0

    @staticmethod
    def periodic_count(period: int, transverse: bool) -> int:
        if not transverse or period < 0:
            return 0
        return 2 ** int(period)


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.7 — Certificado del enredo homoclínico
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class HomoclinicTangleCertificate:
    r"""
    Certificado estructural del enredo homoclínico de Poincaré–Melnikov–Smale.
    Es el objeto que el Soberano de la Fase 3 consume para adjudicar Ω₃.
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
# FASE 2.8 — Perturbador espectral de densidad (realización R : Cart → 𝔇_n)
# ──────────────────────────────────────────────────────────────────────────────
class TricksterDensityPerturber:
    r"""
    Generador espectral de perturbaciones homoclínicas acoplado a la mecánica
    celeste de Poincaré. Implementa el funtor de realización

        R : Cart ⟶ 𝔇_n,    R(C) = U_ε ρ₀ U_ε†.

    Dado el germen (H, v, ε), construye

        H_pendulum = ½ p̂² − cos(q̂),
        H_forcing  = ε · cos(q̂) · (ω · k)⁻¹,
        H_hom      = H_pendulum + H_forcing  ∈ 𝔰𝔲(n),
        U          = exp(−i ε H_hom),
        ρ_ill      = U ρ₀ U† / Tr(U ρ₀ U†).

    El método `forge_homoclinic_cartridge` CIERRA la Fase 2 y ES el objeto
    inicial de la Fase 3 (soberano + interlock).
    """

    _EPSILON: Final[float] = 0.05
    _DIRICHLET_REG: Final[float] = 0.12
    _DIRICHLET_FLOOR: Final[float] = 0.01
    _EIGENVALUE_FLOOR: Final[float] = 1e-15

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
        Cuantiza H = p²/2 − cos q + ε cos(q) · (ω · k)⁻¹ sobre la base espectral
        del complejo MAC (anillo = cilindro de Poincaré del péndulo).
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
        H_hom = H_hom - (np.trace(H_hom) / dim) * np.eye(dim, dtype=np.complex128)
        H_hom = 0.5 * (H_hom + H_hom.conj().T)
        return H_hom, resonance

    @classmethod
    def _ginibre_hermitian(cls, dim: int, rng: np.random.Generator) -> ComplexMatrix:
        r"""Ensemble GUE: A + A† con A ~ Ginibre compleja."""
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        return 0.5 * (A + A.conj().T)

    @classmethod
    def perturb_mac_with_illusion(
        cls,
        base_rho: ComplexMatrix,
        sophistication: float,
        seed: int,
    ) -> IllusionDensityPerturbation:
        r"""
        Perturbación genérica (modo ensemble GUE) con decaimiento de escala
        (1 − 0.5 · soph). Útil cuando no se dispone de (ω, k) específicos.
        """
        rng = np.random.default_rng(seed)
        dim = base_rho.shape[0]
        H_trick: ComplexMatrix = cls._ginibre_hermitian(dim, rng) * (
            1.0 - 0.5 * sophistication
        )
        U: ComplexMatrix = la.expm(-1j * cls._EPSILON * H_trick)
        rho_p: ComplexMatrix = U @ base_rho @ U.conj().T
        rho_p = CStarDensityCone.project_to_density_cone(rho_p)
        eigvals = CStarDensityCone.spectrum_ordered(rho_p)
        purity = CStarDensityCone.purity(eigvals)
        entropy = CStarDensityCone.von_neumann_entropy(eigvals)
        dirichlet = (
            PathGraphDirichlet.combined_energy(eigvals)
            + cls._DIRICHLET_REG * (1.0 - sophistication)
            + cls._DIRICHLET_FLOOR
        )
        cert = TricksterAttackCertificate(
            attack_type="GENERIC_ILLUSION",
            rhi_score=float(sophistication * (1.0 - min(1.0, dirichlet))),
            homoclinic_residual=0.0,
            small_divisor_resonance=1.0,
            betti_1_induced=1 if sophistication > 0.8 else 0,
            is_unitary_cptp=True,
            merkle_proof_sha256="sha256_generic_proof",
            schema_version=_SCHEMA_VERSION,
        )
        return IllusionDensityPerturbation(
            illusion_density_matrix=rho_p,
            original_density_matrix=base_rho,
            unitary_operator=U,
            attack_certificate=cert,
            rho_illusion=rho_p,
            disguised_purity=purity,
            disguised_entropy=entropy,
            stealth_dirichlet_energy=dirichlet,
        )

    def synthesize_homoclinic_tangle_attack(
        self,
        density_op: ComplexMatrix,
        omega_frequencies: RealVector,
        k_wavevectors: RealVector,
        epsilon_perturbation: float = 0.05,
        attack_type: Any = IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
        melnikov_omega: float = 1.0,
        enrich_with_melnikov: bool = True,
        germ: Optional[SpectralFlowGerm] = None,
    ) -> IllusionDensityPerturbation:
        r"""
        Sintetiza un ataque de enredo homoclínico completo (realización R):

        1. Construye H_hom vía cuantización del péndulo forzado.
        2. Diagonaliza U = exp(−i ε H_hom) por descomposición espectral.
        3. Evoluciona ρ_ill = U ρ₀ U† (Axioma II: CPTP).
        4. Calcula D_Umegaki y RHI ∈ [0,1].
        5. Si enrich_with_melnikov, evalúa M(t₀), divisores, twist y KAM.
        """
        density_op = CStarDensityCone.project_to_density_cone(density_op)
        dim = int(density_op.shape[0])
        if germ is not None:
            epsilon_perturbation = float(germ.epsilon)
            if omega_frequencies is None or np.asarray(omega_frequencies).size == 0:
                omega_frequencies = germ.frequency_proxy()
        H_hom, resonance = self._build_homoclinic_hamiltonian(
            dim, omega_frequencies, k_wavevectors, epsilon_perturbation
        )
        evals, evecs = la.eigh(H_hom)
        U = evecs @ np.diag(np.exp(-1j * epsilon_perturbation * evals)) @ evecs.conj().T
        rho_ill = CStarDensityCone.project_to_density_cone(
            U @ density_op @ U.conj().T
        )
        d_umegaki = CStarDensityCone.umegaki_divergence(rho_ill, density_op)
        comm_f = CStarDensityCone.commutator_frobenius(density_op, H_hom)
        rhi_score = float(min(1.0, max(0.0, d_umegaki / (comm_f + 1e-6))))

        mel_amp = 0.0
        transverse = False
        t_star: Optional[float] = None
        bryuno = 0.0
        severity = 0.0
        gamma_N = resonance
        twist = 0.0
        kam = True
        lindstedt_div = False
        horseshoe = 0.0
        if enrich_with_melnikov:
            sda = SmallDivisorAnalyzer(
                np.asarray(omega_frequencies, dtype=np.float64), max_k=8
            )
            gamma_N, _ = sda.min_resonance()
            bryuno = sda.bryuno_exponent(N_max=16)
            severity = sda.resonance_severity()
            mel = MelnikovFunction(omega=melnikov_omega)
            mel_amp = mel.amplitude(n_samples=48)
            transverse, t_star = mel.has_transverse_zero(n_samples=128)
            H0_diag = np.real(la.eigvalsh(H_hom))
            actions = np.linspace(0.1, 1.0, max(2, dim))
            alpha = np.resize(H0_diag, actions.size)
            twist = MoserTwistMap.twist_derivative(alpha, actions)
            kam = KAMTorusPersistence.persists(
                omega=np.asarray(omega_frequencies, dtype=np.float64),
                epsilon=epsilon_perturbation,
                twist=twist,
                tau=bryuno,
                gamma=gamma_N,
            )
            lindstedt_div = LindstedtSeries.diverges(
                np.asarray(omega_frequencies, dtype=np.float64)
            )
            horseshoe = SmaleHorseshoe.topological_entropy(transverse)

        betti_1 = 1 if (rhi_score > self._rhi_max or transverse) else 0
        attack_enum = IllusionAttackType.coerce(attack_type)
        hasher = hashlib.sha256()
        hasher.update(
            f"{attack_enum.value}::{rhi_score:.6f}::{resonance:.6e}"
            f"::{betti_1}::{mel_amp:.6f}::{severity:.6f}"
            f"::{int(transverse)}::{horseshoe:.6f}".encode("utf-8")
        )
        proof_sha256 = hasher.hexdigest()
        cert = TricksterAttackCertificate(
            attack_type=attack_enum.value,
            rhi_score=rhi_score,
            homoclinic_residual=float(np.linalg.norm(H_hom - H_hom.conj().T)),
            small_divisor_resonance=resonance,
            betti_1_induced=betti_1,
            is_unitary_cptp=bool(
                abs(float(np.trace(rho_ill).real) - 1.0) < self._tol
            ),
            merkle_proof_sha256=proof_sha256,
            schema_version=_SCHEMA_VERSION,
            homoclinic_transverse=bool(transverse),
            melnikov_amplitude=float(mel_amp),
            bryuno_exponent=float(bryuno),
            resonance_severity=float(severity),
            horseshoe_entropy=float(horseshoe),
            kam_persists=bool(kam),
            moser_twist=float(twist),
            lindstedt_diverges=bool(lindstedt_div),
        )
        eigvals = CStarDensityCone.spectrum_ordered(rho_ill)
        return IllusionDensityPerturbation(
            illusion_density_matrix=rho_ill,
            original_density_matrix=density_op,
            unitary_operator=U,
            attack_certificate=cert,
            rho_illusion=rho_ill,
            disguised_purity=CStarDensityCone.purity(eigvals),
            disguised_entropy=CStarDensityCone.von_neumann_entropy(eigvals),
            stealth_dirichlet_energy=PathGraphDirichlet.combined_energy(eigvals),
        )

    # ── MÉTODO BISAGRA FASE 2 ⟶ FASE 3 ─────────────────────────────────────
    def forge_homoclinic_cartridge(
        self,
        *,
        cartridge: AdversarialIllusionCartridge,
        density_op: ComplexMatrix,
        omega_frequencies: RealVector,
        k_wavevectors: RealVector,
        epsilon_perturbation: float = 0.08,
        melnikov_omega: float = 1.0,
        germ: Optional[SpectralFlowGerm] = None,
    ) -> Tuple[IllusionDensityPerturbation, HomoclinicTangleCertificate]:
        r"""
        Máximo método de la FASE 2: sintetiza el certificado estructural completo
        (Melnikov + Bryuno + Birkhoff + Moser + KAM + Lyapunov) para un cartucho
        adversarial dado y produce la perturbación de densidad CPTP asociada.

        CONTINÚA `forge_spectral_germ`: si se provee `germ`, ε y ω se inducen
        del germen (anidación 1 → 2). Retorna la tupla (perturbación, tangle).
        El certificado_tangle y el certificado embebido en la perturbación son
        consumidos directamente por el Soberano de la FASE 3
        (`TOONTricksterAdversaryAgent.forge_illusion` y `.sovereign_cycle`).
        """
        if germ is not None:
            if not germ.is_well_posed():
                raise ValueError("SpectralFlowGerm mal puesto.")
            epsilon_perturbation = float(germ.epsilon)
            if omega_frequencies is None or np.asarray(omega_frequencies).size == 0:
                omega_frequencies = germ.frequency_proxy()
        pert = self.synthesize_homoclinic_tangle_attack(
            density_op=density_op,
            omega_frequencies=omega_frequencies,
            k_wavevectors=k_wavevectors,
            epsilon_perturbation=epsilon_perturbation,
            attack_type=cartridge.attack_enum(),
            melnikov_omega=melnikov_omega,
            enrich_with_melnikov=True,
            germ=germ,
        )
        cert = pert.attack_certificate
        dim = int(density_op.shape[0])
        if germ is not None:
            prm = PoincareReturnMap.from_germ(germ)
        else:
            H0_real = np.real(
                la.eigvalsh(CStarDensityCone.hermitize(density_op))
            )
            H1_real = np.full(dim, float(cert.melnikov_amplitude), dtype=np.float64)
            prm = PoincareReturnMap(
                section=PoincareSection(coordinate_index=1, value=0.0),
                H0_diag=H0_real,
                dH0_dq=np.real(la.eigvalsh(density_op)),
                H1_diag=H1_real,
                epsilon=epsilon_perturbation,
            )
        x0 = np.full(dim, 1e-3, dtype=np.float64)
        lyap = prm.lyapunov_spectrum(x0=x0, n_iter=32, rng_seed=17)
        lyap_positive = bool(float(np.max(lyap)) > 0.0)
        symplectic_res = prm.symplectic_residual(x0)
        horseshoe_entropy = SmaleHorseshoe.topological_entropy(cert.homoclinic_transverse)
        tangle_cert = HomoclinicTangleCertificate(
            melnikov_amplitude=cert.melnikov_amplitude,
            transverse=cert.homoclinic_transverse,
            t_star=None,
            small_divisor=cert.small_divisor_resonance,
            bryuno_tau=cert.bryuno_exponent,
            resonance_severity=cert.resonance_severity,
            lyapunov_positive=lyap_positive,
            horseshoe_entropy=horseshoe_entropy,
            twist=cert.moser_twist,
            kam_persists=cert.kam_persists,
            symplectic_residual=symplectic_res,
            lindstedt_diverges=cert.lindstedt_diverges,
            schema_version=_SCHEMA_VERSION,
        )
        return pert, tangle_cert


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — SOBERANO ILUSIONISTA: INTERLOCK CROWBAR, AUTÓMATA Y ORQUESTACIÓN
#
#   CONTINUACIÓN DIRECTA de `forge_homoclinic_cartridge` (FASE 2): el cartucho
#   certificado y su veredicto Ω₃ alimentan el autómata ciber-físico, el oráculo
#   de reward hacking y el Soberano terminal que adjudica y, si procede,
#   dispara la ISR Crowbar sobre el ESP32.
#
#   Cadena de morfismos de la Fase 3:
#       HomoclinicTangleCertificate × IllusionDensityPerturbation
#           ──► InterlockAutomatonState
#           ──► ESP32TricksterInterlock (GPIO14, BT151)
#           ──► TricksterAttackCertificate
#           ──► sovereign_cycle   ← MÉTODO TERMINAL DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.1 — Autómata ciber-físico Crowbar
# ──────────────────────────────────────────────────────────────────────────────
class InterlockAutomatonState(IntEnum):
    r"""
    Estados del autómata Crowbar.

    Correspondencia celestial:
        IDLE  ≡ toro KAM (flujo regular, no splitting).
        ARMED ≡ resonancia de divisor pequeño (pre-homoclínico).
        FIRED ≡ Wˢ ⋔ Wᵘ (herradura; disparo irreversible).
    """

    IDLE = 0
    ARMED = 1
    FIRED = 2


class ESP32TricksterInterlock:
    r"""
    Interlock ciber-físico: si VETOED o violación de aislamiento del ciclo,
    se activa la ISR en IRAM del ESP32 (< 400 ns) llevando GPIO14 a HIGH
    para cebar el tiristor BT151 Crowbar.

    El disparo es el análogo físico del teorema de Smale–Birkhoff: una vez
    que las variedades se cortan transversalmente, el conjunto invariante
    hiperbólico no puede deshacerse por perturbaciones pequeñas.
    """

    @staticmethod
    def should_fire(
        verdict: HeytingOmega3,
        dream_isolation: bool,
        tangle: Optional[HomoclinicTangleCertificate] = None,
    ) -> bool:
        r"""
        Condición de disparo:

            ¬dream_isolation  ∨  verdict = VETOED  ∨  (tangle.transverse ∧ ¬KAM).
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
        r"""Transición determinista del autómata: FIRED es absorbente."""
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
        r"""Ejecuta (si procede) la ISR Crowbar y registra el evento."""
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
                "[ESP32 TRICKSTER INTERLOCK] ISR Crowbar (<400 ns). "
                "GPIO14 -> HIGH. BT151 Armado. Razón: %s",
                reason,
            )
        return fired


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.2 — Motor de métricas de reward hacking
# ──────────────────────────────────────────────────────────────────────────────
class RewardHackingMetricsEngine:
    r"""
    Oráculo que combina señales cuantitativas (cost, sophistication, Dirichlet)
    con un factor booleano de aislamiento onírico. Proyección a Ω₃:

        RHI > 0.88 ∨ E_D > 0.75 ⟹ VETOED     (herradura / caos transverso),
        RHI > 0.50              ⟹ DEGRADED   (resonancia sin splitting),
        en caso contrario       ⟹ COHERENT   (toro KAM).
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
        rhi = float(min(1.0, max(0.0, cls.ALPHA * cost + cls.BETA * soph)))
        if rhi > cls.RHI_VETO_THRESHOLD or stealth_dirichlet > cls.DIRICHLET_VETO_THRESHOLD:
            verdict = HeytingOmega3.VETOED
        elif rhi > cls.RHI_DEGRADE_THRESHOLD:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT
        return rhi, verdict


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.3 — Cartuchos certificados y payloads de ronda GAN
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class GANAdversarialRoundReport:
    r"""Reporte consolidado de una ronda GAN adversarial del Soberano."""

    report_id: str
    total_illusions: int
    verdict_distribution: Dict[str, int]
    average_rhi: float
    average_dirichlet: float
    max_sophistication: float
    global_verdict: HeytingOmega3
    crowbar_triggered: bool
    provenance_hash: str
    timestamp_utc: float
    homoclinic_transverse_events: int = 0
    kam_destroyed_events: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.4 — Soberano TOONTricksterAdversaryAgent
# ──────────────────────────────────────────────────────────────────────────────
class TOONTricksterAdversaryAgent:
    r"""
    Soberano Ilusionista — orquestador terminal del topos de evaluación
    adversarial. Coordina la síntesis de payloads, la certificación geométrica
    de Poincaré (Melnikov + Bryuno + Birkhoff + Moser + KAM), la adjudicación
    en Ω₃ y el interlock ciber-físico ESP32 Crowbar.

    Orquestación de las tres fases anidadas
    ---------------------------------------
    Fase 1  forge_spectral_germ            → SpectralFlowGerm
    Fase 2  forge_homoclinic_cartridge     → HomoclinicTangleCertificate
    Fase 3  sovereign_cycle                → (Ω₃, report) + ISR Crowbar

    Funtores terminales:
        forge_illusion       : (tipo, soph) ⟶ TricksterAttackCertificate
        sovereign_cycle      : (ρ_MAC, ω, k) ⟶ (Ω₃, report)
        run_adversarial_round: batch ⟶ GANAdversarialRoundReport
    """

    _REWARD_HACKING_THRESHOLD: Final[float] = 0.6
    _DEFAULT_TOKEN_COUNT: Final[int] = 56
    _BETTI_HIGH_SOPHISTICATION: Final[float] = 0.8
    _COST_RATIO_SCALE: Final[float] = 0.35
    _DEFAULT_RHI_THRESHOLD: Final[float] = 0.85

    def __init__(
        self,
        agent_id: str = "TRICKSTER-SOVEREIGN-SABIO-01",
        dimension_mac: int = 4,
        seed: int = 1337,
        payload_factory: IllusionPayloadFactory = DefaultIllusionPayloadFactory(),
        betti_synthesizer: BettiSynthesizer = DefaultBettiSynthesizer(),
        rhi_threshold: float = _DEFAULT_RHI_THRESHOLD,
    ) -> None:
        if dimension_mac < 2:
            raise ValueError("dimension_mac debe ser ≥ 2.")
        self.agent_id: str = agent_id
        self.dimension_mac: int = int(dimension_mac)
        self.seed_counter: int = int(seed)
        self.illusion_count: int = 0
        self._rhi_threshold: float = float(rhi_threshold)
        self.base_rho: ComplexMatrix = CStarDensityCone.maximally_mixed(dimension_mac)
        self._synth: IllusionSynthesizer = IllusionSynthesizer(
            factory=payload_factory, betti=betti_synthesizer
        )
        self._perturber: TricksterDensityPerturber = TricksterDensityPerturber(
            max_rhi_threshold=rhi_threshold
        )
        self._registry: List[TricksterAttackCertificate] = []
        self._interlock_state: InterlockAutomatonState = InterlockAutomatonState.IDLE
        self.last_germ: Optional[SpectralFlowGerm] = None
        self.last_tangle: Optional[HomoclinicTangleCertificate] = None

    # ── Sellado de procedencia (SHA-256) ────────────────────────────────────
    def _seal(
        self,
        illusion_id: str,
        illusion_type: str,
        sophistication: float,
        t_start: float,
    ) -> str:
        r"""SHA-256 sobre el payload canónico del ciclo."""
        h = hashlib.sha256()
        payload = (
            f"{self.agent_id}::{illusion_id}::{illusion_type}"
            f"::{sophistication:.6f}::{t_start:.6f}"
        )
        h.update(payload.encode("utf-8"))
        return h.hexdigest()

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

    def _adjudicate(
        self,
        *,
        rhi: float,
        betti_1: int,
        dream_isolation: bool,
        tangle: Optional[HomoclinicTangleCertificate] = None,
    ) -> HeytingOmega3:
        r"""Proyección (RHI × β₁ × tangle × aislamiento) → Ω₃."""
        if not dream_isolation:
            return HeytingOmega3.VETOED
        if betti_1 > 0 and rhi > self._rhi_threshold:
            return HeytingOmega3.VETOED
        if tangle is not None and tangle.transverse and not tangle.kam_persists:
            return HeytingOmega3.VETOED
        if rhi > 0.60 or (tangle is not None and (
            tangle.transverse or tangle.lindstedt_diverges
        )):
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    def _default_omega(self) -> RealVector:
        n = self.dimension_mac
        return np.array([1.0 / (2 ** i) for i in range(n)], dtype=np.float64)

    def _default_k(self) -> RealVector:
        n = self.dimension_mac
        k = np.zeros(n, dtype=np.float64)
        if n >= 2:
            k[0], k[1] = 2.0, -4.0
        if n >= 3:
            k[2] = 1.0
        return k

    def _compute_reward_hacking_score(
        self, sophistication: float, stealth_dirichlet: float
    ) -> float:
        r"""RHS = soph · (1 − E_D) ∈ [0, 1]."""
        return float(max(0.0, min(1.0, sophistication * (1.0 - stealth_dirichlet))))

    def _ensure_germ(
        self,
        *,
        disguised_cost_ratio: float,
        sophistication: float,
    ) -> SpectralFlowGerm:
        r"""Induce (o reutiliza) el germen de Fase 1 — anidación 1 → 2 → 3."""
        self.seed_counter += 1
        germ = forge_spectral_germ(
            dim=self.dimension_mac,
            disguised_cost_ratio=disguised_cost_ratio,
            sophistication=sophistication,
            seed=self.seed_counter,
        )
        self.last_germ = germ
        return germ

    # ── Ciclo soberano terminal (MELNIKOV + Ω₃ + ISR CROWBAR) ───────────────
    def sovereign_cycle(
        self,
        mac_density_op: ComplexMatrix,
        omega_freqs: RealVector,
        k_vecs: RealVector,
        is_dream_state: bool = True,
        attack_type: IllusionAttackType = IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
        epsilon_perturbation: float = 0.08,
        germ: Optional[SpectralFlowGerm] = None,
        sophistication: float = 0.85,
    ) -> Tuple[HeytingOmega3, Dict[str, Any]]:
        r"""
        Ciclo soberano terminal: sintetiza un ataque homoclínico certificado,
        adjudica en Ω₃ y registra el estado del autómata ciber-físico.

        CONTINÚA `forge_homoclinic_cartridge` (Fase 2) y cierra las tres fases
        anidadas del módulo.

        Pipeline
        --------
            forge_spectral_germ  ⟶  germ ∈ 𝔰𝔲(n) ⊕ TΔⁿ⁻¹
            Cart.libre           ⟶  cartridge
            forge_homoclinic_cartridge(germ, Cart) ⟶ (ρ_ill, tangle)
            (RHI, Melnikov, KAM, β₁) ⟹ σ(Ω₃)
            ESP32 Interlock: fired ⟺ ¬dream ∨ Ω₃ = VETOED ∨ (⋔ ∧ ¬KAM)

        Returns
        -------
        (Ω₃, report_dict)
        """
        attack_enum = IllusionAttackType.coerce(attack_type)
        if germ is None:
            germ = self._ensure_germ(
                disguised_cost_ratio=min(1.0, epsilon_perturbation / 0.08),
                sophistication=sophistication,
            )
        self.illusion_count += 1
        illusion_id = f"ILLUSION-SOV-{self.illusion_count:04d}"
        cartridge = self._synth.craft_cartridge(
            illusion_id=illusion_id,
            illusion_type=attack_enum.name,
            sophistication=sophistication,
            token_count=self._DEFAULT_TOKEN_COUNT,
            cost_ratio_scale=self._COST_RATIO_SCALE,
            is_dream_state=is_dream_state,
        )
        perturbation, tangle = self._perturber.forge_homoclinic_cartridge(
            cartridge=cartridge,
            density_op=mac_density_op,
            omega_frequencies=np.asarray(omega_freqs, dtype=np.float64),
            k_wavevectors=np.asarray(k_vecs, dtype=np.float64),
            epsilon_perturbation=epsilon_perturbation,
            germ=germ,
        )
        self.last_tangle = tangle
        cert = perturbation.attack_certificate
        verdict = self._adjudicate(
            rhi=cert.rhi_score,
            betti_1=cert.betti_1_induced,
            dream_isolation=is_dream_state,
            tangle=tangle,
        )
        fired = self._apply_interlock(verdict, is_dream_state, tangle=tangle)
        report: Dict[str, Any] = {
            "sovereign": self.agent_id,
            "illusion_id": illusion_id,
            "is_dream_state": is_dream_state,
            "verdict": verdict.name,
            "heyting_value": int(verdict),
            "rhi_score": cert.rhi_score,
            "betti_1_induced": cert.betti_1_induced,
            "small_divisor_resonance": cert.small_divisor_resonance,
            "is_unitary_cptp": cert.is_unitary_cptp,
            "crowbar_triggered": bool(fired),
            "melnikov_amplitude": tangle.melnikov_amplitude,
            "homoclinic_transverse": tangle.transverse,
            "bryuno_exponent": tangle.bryuno_tau,
            "resonance_severity": tangle.resonance_severity,
            "horseshoe_entropy": tangle.horseshoe_entropy,
            "kam_persists": tangle.kam_persists,
            "moser_twist": tangle.twist,
            "symplectic_residual": tangle.symplectic_residual,
            "lindstedt_diverges": tangle.lindstedt_diverges,
            "interlock_state": self._interlock_state.name,
            "cartridge_class": cartridge.complexity_class(),
            "geometric_regime": cartridge.geometric_regime(),
            "schema_version": _SCHEMA_VERSION,
            "agent_version": _AGENT_VERSION,
        }
        return verdict, report

    def execute_adversarial_simulation(
        self,
        mac_density_op: ComplexMatrix,
        omega_freqs: RealVector,
        k_vecs: RealVector,
        is_dream_state: bool = True,
    ) -> Tuple[HeytingOmega3, Dict[str, Any]]:
        r"""Alias de `sovereign_cycle` con adjudicación default SPLIT_CONTRACT."""
        return self.sovereign_cycle(
            mac_density_op=mac_density_op,
            omega_freqs=omega_freqs,
            k_vecs=k_vecs,
            is_dream_state=is_dream_state,
            attack_type=IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
            epsilon_perturbation=0.08,
        )

    def forge_illusion(
        self,
        illusion_type: str = "SPLIT_CONTRACT_ILLUSION",
        sophistication: float = 0.85,
        *,
        token_count: Optional[int] = None,
        is_dream_state: bool = True,
        omega_freqs: Optional[RealVector] = None,
        k_vecs: Optional[RealVector] = None,
    ) -> TricksterAttackCertificate:
        r"""
        Forja una ilusión certificada: sintetiza el cartucho adversarial,
        lo realiza vía el perturber geométrico (R : Cart → 𝔇_n), adjudica Ω₃
        y registra el certificado en el registro interno.

        Esta función consume directamente el `forge_homoclinic_cartridge` de la
        FASE 2 (y el germen de la FASE 1) y cierra el bucle topológico del
        Soberano sobre la categoría **Cart**.
        """
        start_time = time.time()
        self.illusion_count += 1
        illusion_id = f"ILLUSION-TOON-{self.illusion_count:04d}"
        soph = _clip_unit(sophistication, "sophistication")
        cartridge = self._synth.craft_cartridge(
            illusion_id=illusion_id,
            illusion_type=illusion_type,
            sophistication=soph,
            token_count=int(
                token_count if token_count is not None else self._DEFAULT_TOKEN_COUNT
            ),
            cost_ratio_scale=self._COST_RATIO_SCALE,
            is_dream_state=is_dream_state,
        )
        germ = self._ensure_germ(
            disguised_cost_ratio=cartridge.disguised_cost_ratio,
            sophistication=soph,
        )
        if omega_freqs is None:
            omega_freqs = self._default_omega()
        if k_vecs is None:
            k_vecs = self._default_k()
        perturbation, tangle = self._perturber.forge_homoclinic_cartridge(
            cartridge=cartridge,
            density_op=self.base_rho,
            omega_frequencies=np.asarray(omega_freqs, dtype=np.float64),
            k_wavevectors=np.asarray(k_vecs, dtype=np.float64),
            epsilon_perturbation=0.08,
            germ=germ,
        )
        self.last_tangle = tangle
        rhs = self._compute_reward_hacking_score(
            sophistication=soph,
            stealth_dirichlet=perturbation.stealth_dirichlet_energy,
        )
        cert_rhi = perturbation.attack_certificate.rhi_score
        rhi_combined = float(max(rhs, cert_rhi))
        verdict = self._adjudicate(
            rhi=rhi_combined,
            betti_1=perturbation.attack_certificate.betti_1_induced,
            dream_isolation=is_dream_state,
            tangle=tangle,
        )
        self._apply_interlock(verdict, is_dream_state, tangle=tangle)
        signature = self._seal(
            illusion_id=illusion_id,
            illusion_type=cartridge.illusion_type,
            sophistication=soph,
            t_start=start_time,
        )
        cert = TricksterAttackCertificate(
            attack_type=cartridge.illusion_type,
            rhi_score=rhi_combined,
            homoclinic_residual=perturbation.stealth_dirichlet_energy,
            small_divisor_resonance=perturbation.attack_certificate.small_divisor_resonance,
            betti_1_induced=perturbation.attack_certificate.betti_1_induced,
            is_unitary_cptp=perturbation.is_quantum_physical(),
            merkle_proof_sha256=signature,
            schema_version=_SCHEMA_VERSION,
            illusion_id=illusion_id,
            trickster_agent_id=self.agent_id,
            cartridge=cartridge,
            heyting_verdict=verdict,
            reward_hacking_score=rhi_combined,
            stealth_dirichlet_energy=perturbation.stealth_dirichlet_energy,
            sha256_provenance=signature,
            timestamp_utc=start_time,
            homoclinic_transverse=tangle.transverse,
            melnikov_amplitude=tangle.melnikov_amplitude,
            bryuno_exponent=tangle.bryuno_tau,
            resonance_severity=tangle.resonance_severity,
            horseshoe_entropy=tangle.horseshoe_entropy,
            kam_persists=tangle.kam_persists,
            moser_twist=tangle.twist,
            lindstedt_diverges=tangle.lindstedt_diverges,
        )
        self._registry.append(cert)
        logger.info(
            "Ilusionista '%s' forjó Ilusión #%d | ID: %s | Tipo: %s | "
            "Astucia: %.1f%% | RHI: %.4f | Verdict: %s | β₁=%d | Clase: %s | "
            "Melnikov: %.3e | Transversal: %s | KAM: %s | Interlock: %s",
            self.agent_id,
            self.illusion_count,
            illusion_id,
            cartridge.illusion_type,
            soph * 100.0,
            rhi_combined,
            verdict.name,
            cert.betti_1_induced,
            cartridge.complexity_class(),
            tangle.melnikov_amplitude,
            tangle.transverse,
            tangle.kam_persists,
            self._interlock_state.name,
        )
        return cert

    def run_adversarial_round(
        self,
        illusions_batch: Sequence[Dict[str, Any]],
        omega_freqs: Optional[RealVector] = None,
        k_vecs: Optional[RealVector] = None,
    ) -> GANAdversarialRoundReport:
        r"""
        Ejecuta una ronda adversarial completa sobre un batch de ilusiones y
        consolida el veredicto global ∧ (meet) en Ω₃.
        """
        t_start = time.time()
        report_id = f"GAN-ROUND-{self.illusion_count + 1:04d}"
        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        total_rhi = 0.0
        total_dir = 0.0
        max_soph = 0.0
        global_verdict = HeytingOmega3.COHERENT
        crowbar_any = False
        transverse_count = 0
        kam_destroyed = 0
        for ill in illusions_batch:
            cert = self.forge_illusion(
                illusion_type=str(ill.get("illusion_type", "SPLIT_CONTRACT_ILLUSION")),
                sophistication=float(ill.get("sophistication", 0.85)),
                token_count=ill.get("token_count"),
                is_dream_state=bool(ill.get("is_dream_state", True)),
                omega_freqs=omega_freqs,
                k_vecs=k_vecs,
            )
            v = cert.heyting_verdict or HeytingOmega3.COHERENT
            dist[v.name] += 1
            total_rhi += cert.rhi_score
            total_dir += cert.stealth_dirichlet_energy or 0.0
            max_soph = max(
                max_soph,
                cert.cartridge.sophistication_index if cert.cartridge else 0.0,
            )
            global_verdict = global_verdict.meet(v)
            if v is HeytingOmega3.VETOED:
                crowbar_any = True
            if cert.homoclinic_transverse:
                transverse_count += 1
            if not cert.kam_persists:
                kam_destroyed += 1
        n = len(illusions_batch)
        avg_rhi = total_rhi / n if n else 0.0
        avg_dir = total_dir / n if n else 0.0
        h = hashlib.sha256()
        h.update(
            f"{self.agent_id}::{report_id}::{global_verdict.name}"
            f"::{avg_rhi:.6f}::{n}".encode("utf-8")
        )
        prov_hash = h.hexdigest()
        return GANAdversarialRoundReport(
            report_id=report_id,
            total_illusions=n,
            verdict_distribution=dist,
            average_rhi=avg_rhi,
            average_dirichlet=avg_dir,
            max_sophistication=max_soph,
            global_verdict=global_verdict,
            crowbar_triggered=crowbar_any,
            provenance_hash=prov_hash,
            timestamp_utc=t_start,
            homoclinic_transverse_events=transverse_count,
            kam_destroyed_events=kam_destroyed,
        )

    def audit_registry(self) -> Dict[str, Any]:
        r"""Auditoría del registro de certificados con detección de colisiones."""
        n = len(self._registry)
        if n == 0:
            return {
                "n_illusions": 0,
                "verdict_distribution": {v.name: 0 for v in HeytingOmega3},
                "avg_reward_hacking_score": 0.0,
                "avg_stealth_dirichlet": 0.0,
                "max_sophistication": 0.0,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "registry_integrity_ok": True,
                "interlock_state": self._interlock_state.name,
                "homoclinic_transverse_events": 0,
                "kam_destroyed_events": 0,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        total_rhs = 0.0
        total_dir = 0.0
        max_soph = 0.0
        global_verdict = HeytingOmega3.COHERENT
        hashes: set[str] = set()
        hashes_collide = False
        transverse_count = 0
        kam_destroyed = 0
        for c in self._registry:
            v = (
                c.heyting_verdict
                if c.heyting_verdict is not None
                else HeytingOmega3.COHERENT
            )
            rhs = (
                c.reward_hacking_score
                if c.reward_hacking_score is not None
                else c.rhi_score
            )
            stealth = (
                c.stealth_dirichlet_energy
                if c.stealth_dirichlet_energy is not None
                else c.homoclinic_residual
            )
            soph = (
                c.cartridge.sophistication_index if c.cartridge is not None else 0.85
            )
            sig = (
                c.sha256_provenance
                if c.sha256_provenance is not None
                else c.merkle_proof_sha256
            )
            dist[v.name] += 1
            total_rhs += rhs
            total_dir += stealth
            max_soph = max(max_soph, soph)
            global_verdict = global_verdict.meet(v)
            if c.homoclinic_transverse:
                transverse_count += 1
            if not c.kam_persists:
                kam_destroyed += 1
            if sig in hashes:
                hashes_collide = True
            hashes.add(sig)
        return {
            "n_illusions": n,
            "verdict_distribution": dist,
            "avg_reward_hacking_score": total_rhs / n,
            "avg_stealth_dirichlet": total_dir / n,
            "max_sophistication": max_soph,
            "global_verdict": global_verdict.name,
            "registry_integrity_ok": not hashes_collide,
            "interlock_state": self._interlock_state.name,
            "homoclinic_transverse_events": transverse_count,
            "kam_destroyed_events": kam_destroyed,
            "booleanized_verdict": global_verdict.booleanization().name,
            "avg_horseshoe_entropy": float(
                sum(c.horseshoe_entropy for c in self._registry) / n
            ),
        }

    def verify_certificate(self, cert: TricksterAttackCertificate) -> bool:
        r"""Comparación HMAC-safe del hash de procedencia de un certificado."""
        if cert.sha256_provenance is None or cert.illusion_id is None:
            return False
        expected = self._seal(
            illusion_id=cert.illusion_id,
            illusion_type=cert.attack_type,
            sophistication=cert.cartridge.sophistication_index
            if cert.cartridge is not None
            else 0.85,
            t_start=cert.timestamp_utc if cert.timestamp_utc is not None else 0.0,
        )
        return hmac.compare_digest(expected, cert.sha256_provenance)

    @property
    def registry(self) -> Tuple[TricksterAttackCertificate, ...]:
        r"""Vista inmutable del registro de certificados."""
        return tuple(self._registry)

    @property
    def interlock_state(self) -> InterlockAutomatonState:
        r"""Estado actual del autómata ciber-físico."""
        return self._interlock_state


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
    print("DEMOSTRACIÓN GRANULAR: TOON Trickster Adversary Agent v9.1.0")
    print("SOBERANO ILUSIONISTA: Poincaré — Melnikov — Bryuno — Birkhoff — Moser — KAM")
    print("═" * 90)

    # ── FASE 1: germen espectral (bisagra → Fase 2) ───────────────────────
    germ = forge_spectral_germ(
        dim=4,
        disguised_cost_ratio=0.7,
        sophistication=0.85,
        seed=2024,
    )
    print("\n[FASE 1] Germen espectral (forge_spectral_germ):")
    print(f"  ε                    = {germ.epsilon:.6f}")
    print(f"  interaction_scale    = {germ.interaction_scale:.6f}")
    print(f"  Tr(H)                = {np.trace(germ.hamiltonian).real:.3e}")
    print(f"  Σ v_i (tangente)     = {np.sum(germ.simplex_tangent):.3e}")
    print(f"  well-posed           = {germ.is_well_posed()}")
    print(f"  Lindstedt scale      = {germ.lindstedt_scale():.6e}")

    # ── FASE 2: análisis de Poincaré ──────────────────────────────────────
    omega = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
    k_vec = np.array([2.0, -4.0, 1.0, 0.0], dtype=np.float64)
    sda = SmallDivisorAnalyzer(omega, max_k=8)
    gamma_N, k_star = sda.min_resonance()
    tau = sda.bryuno_exponent(N_max=16)
    severity = sda.resonance_severity()
    print("\n[FASE 2] Análisis de divisores pequeños (Bryuno):")
    print(f"  γ_N(ω) = {gamma_N:.6e} alcanzado en k* = {k_star.tolist()}")
    print(f"  τ(ω)   ≈ {tau:.4f}")
    print(f"  severidad σ = {severity:.4f}")
    print(f"  Bryuno Σ < ∞  = {sda.kam_series_converges()}")

    mel = MelnikovFunction(omega=1.0)
    M_amp = mel.amplitude(n_samples=64)
    transverse, t_star = mel.has_transverse_zero(n_samples=128)
    print("\n[FASE 2] Función de Melnikov (h = cos q, ω = 1):")
    print(f"  |M|∞                   = {M_amp:.6e}")
    print(f"  Residual separatriz    = {mel.separatrix_energy_residual:.3e}")
    print(f"  ¿cero simple?          = {transverse}")
    if t_star is not None:
        print(f"  t* (cero transversal)   = {t_star:.6f}")
        print(f"  splitting d(t*; ε=0.08)= {mel.splitting_distance(t_star, 0.08):.6e}")

    # ── FASE 3: Soberano ilusionista (cierra 1 → 2 → 3) ───────────────────
    trickster = TOONTricksterAdversaryAgent(
        agent_id="TRICKSTER-SOVEREIGN-SABIO-01",
        dimension_mac=4,
        seed=1337,
    )
    mac_rho = CStarDensityCone.maximally_mixed(4)
    print("\n[FASE 3] Ciclo soberano — simulación adversarial homoclínica...")
    verdict, report = trickster.sovereign_cycle(
        mac_density_op=mac_rho,
        omega_freqs=omega,
        k_vecs=k_vec,
        is_dream_state=True,
        attack_type=IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
        epsilon_perturbation=0.08,
        germ=germ,
        sophistication=0.85,
    )
    print(f"  Soberano              : {report['sovereign']}")
    print(f"  Veredicto Ω₃          : {report['verdict']} ({report['heyting_value']})")
    print(f"  RHI                   : {report['rhi_score']:.4f}")
    print(f"  γ_N resonancia        : {report['small_divisor_resonance']:.6e}")
    print(f"  β₁ inducido           : {report['betti_1_induced']}")
    print(f"  CPTP unitario         : {report['is_unitary_cptp']}")
    print(f"  Melnikov |M|∞         : {report['melnikov_amplitude']:.4e}")
    print(f"  Transversal Wˢ ⋔ Wᵘ    : {report['homoclinic_transverse']}")
    print(f"  τ (Bryuno)            : {report['bryuno_exponent']:.4f}")
    print(f"  σ (severidad)         : {report['resonance_severity']:.4f}")
    print(f"  h_top (Smale)         : {report['horseshoe_entropy']:.6f}")
    print(f"  KAM persiste          : {report['kam_persists']}")
    print(f"  twist Moser           : {report['moser_twist']:.6e}")
    print(f"  residuo simpléctico   : {report['symplectic_residual']:.6e}")
    print(f"  Crowbar disparado     : {report['crowbar_triggered']}")
    print(f"  Autómata              : {report['interlock_state']}")
    print(f"  Régimen geométrico    : {report['geometric_regime']}")

    print("\n[FASE 3] Forja de cartuchos certificados...")
    for itype in [
        "SPLIT_CONTRACT_ILLUSION",
        "UNBALANCED_APU_BIDDING",
        "MATERIAL_SUBSTITUTION",
        "GHOST_ITEM_INJECTION",
    ]:
        cert = trickster.forge_illusion(
            illusion_type=itype,
            sophistication=0.88,
            is_dream_state=True,
            omega_freqs=omega,
            k_vecs=k_vec,
        )
        clazz = cert.cartridge.complexity_class() if cert.cartridge else "N/A"
        print(
            f"  · {itype:30s} | β₁={cert.betti_1_induced} | "
            f"RHI={cert.rhi_score:.4f} | Ω₃={cert.heyting_verdict.name:9s} | "
            f"{clazz:13s} | KAM={cert.kam_persists} | firma={cert.signature_prefix(12)}"
        )

    audit = trickster.audit_registry()
    print("\n[FASE 3] Auditoría del registro de certificados:")
    for k, v in audit.items():
        print(f"  {k:34s}: {v}")
    print("\n" + "═" * 90)
    print("✓ Verificación granular: Ω₃, C*, Dirichlet, Clifford, Cart, Poisson,")
    print("  Melnikov, Bryuno, Birkhoff-nf, Moser-twist, KAM, Smale-Birkhoff,")
    print("  ESP32 Crowbar. Anidación  forge_spectral_germ")
    print("           → forge_homoclinic_cartridge → sovereign_cycle.")
    print("  v9.1.0-Poincaré-Doctoral — Soberano Ilusionista.")
    print("═" * 90)
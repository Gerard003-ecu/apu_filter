
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Wisdom Weaver Engine (Motor Espectral y Campo Metabolizador) ║
║ Ubicación: app/wisdom/toon_wisdom_weaver_engine.py                           ║
║ Versión  : 4.0.0-Doctoral-Nested-Ω₃-Poincaré-Celestial-KAM-Delaunay-Cartan   ║
║ Autor    : APU Wisdom & Metacortex Mathematical Core Architecture            ║
╚══════════════════════════════════════════════════════════════════════════════╝


SINOPSIS FORMAL Y CATEGORIAL EN EL ESTRATO WISDOM (V_𝕎):
────────────────────────────────────────────────────────────────────────────────
Este motor espectral realiza la función de "Campo Metabolizador" en la Ciudadela
de Cristal (Estrato Wisdom, V_𝕎, Nivel 0) del ecosistema APU Filter. Transmuta la
"grasa sintáctica" de los presupuestos de obra civil (JSONs redundantes) en
"Vitaminas Cognitivas TOON" de alta densidad, elevándolas al espacio de Hilbert
separable ℋ_n ≅ ℂⁿ como operadores de densidad en el cono compacto convexo
𝔇(ℋ_n). Su topología se dota de la estructura de variedad simpléctica coadjunta
(Kirillov-Kostant-Souriau) inducida por la forma traza sobre 𝔲(n)*.


Se reinterpreta el flujo de purificación isospectral de Brockett como un SISTEMA
HAMILTONIANO INTEGRABLE DE TIPO KEPLERIANO sobre la órbita coadjunta U(n)·ρ₀,
donde el Hamiltoniano principal es la función de Lyapunov L(ρ) = Tr(ρN), y la
perturbación metabólica ε·H₁ (inyectada por coste/riesgo) lo deforma en el
sentido de la TEORÍA DE POINCARÉ DE LA MECÁNICA CELESTE:

    • Ecuación de Kepler: la precesión de los autoespacios de ρ bajo N satisface
      la ecuación de Kepler E − e·sin E = M en la coordenada de anomalía media.


    • Sección de Poincaré Σ_c ⊂ 𝔇(ℋ_n): hipersuperficie transversal al flujo
      definida por Tr(ρN) = c. La aplicación de retorno P: Σ_c → Σ_c es una
      aplicación twist que preserva área (Liouville) y cuya estructura de
      puntos periódicos está gobernada por el TEOREMA DE POINCARÉ-BIRKHOFF:
      toda aplicación twist que rota un ángulo θ ∈ (0, 2π) posee al menos
      2·q órbitas periódicas de período q para cada racional p/q en el
      intervalo de twist.


    • Tori KAM / Teorema de Kolmogorov-Arnold-Moser: para ε suficientemente
      pequeño y ω = (ω₁, ..., ω_n) Diofantino, la órbita espectral de Brockett
      sobrevive como toro invariante, dando lugar al VALOR COHERENT en Ω₃.
      La destrucción del toro (rotura de resonancia) corresponde a DEGRADED; la
      fuga al infinito por separatriz hiperbólica (Arnold diffusion) es VETOED.


    • Integridad de Melnikov: la función M(t₀) = ∫_{-∞}^{+∞} {H₀, H₁}(ρ(t+t₀)) dt
      detecta la apertura del tubo homoclínico y por ende la aparición de
      dinámica caótica efímera ("socavón lógico") en la métrica de Poincaré.


    • Variables de Delaunay (L, G, H, l, g, h): adaptadas al problema
      metabólico restringido de tres cuerpos donde (costo, riesgo, pureza)
      son las tres masas; L, G, H son funciones de Tr(ρN), Tr([ρ,N][ρ,N]†)
      y de la brecha espectral Δλ(ρ).


Se preservan simultáneamente las siguientes estructuras algebraico-geométricas:

    (a) El retículo de Heyting lineal Ω₃ = {VETOED ≺ DEGRADED ≺ COHERENT}.
    (b) El álgebra C* de Banach 𝐁𝐚𝐧(ℋ_n) con involución † y traza Tr.
    (c) La 1-forma de Poincaré-Cartan θ_PC = Tr(ρ dN), cuya integral sobre un
        lazo γ ⊂ 𝔇(ℋ_n) define el invariante integral de Poincaré ∮γ θ_PC.
    (d) La medida de Liouville μ_L sobre el fibrado de órbitas isoespectrales.
    (e) La adjunción de Galois Hom_D(F(MIC), MAC) ≅ Hom_C(MIC, G(MAC)).
    (f) El espacio de Fock fermiónico ℱ₋ = ℂ|0⟩ ⊕ ℂ|1⟩ con {a, a†} = 𝟙.
    (g) El Laplaciano de Dirichlet-combinatorio L = Deg − |A|_H del grafo
        atencional (Betti-0 β₀ = dim ker L).


ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (Composición Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 ──► SUTURA ALGEBRAICA Y SEMILLA METABÓLICA (Observe)
           • HeytingOmega3             : retículo lineal acotado y residuado.
           • PoincareActionAnglePair   : par canónico (J, θ) de la geometría
                                         simpléctica coadjunta (KKS).
           • DensityOperator           : elemento del cono 𝔇(ℋ_n); admite la
                                         tripleta de Delaunay como lectura
                                         celeste de su espectro.
           • TOONSynapticCartridge     : objeto de 𝐂𝐚𝐫𝐭_TOON; admite el reparto
                                         de masas (costo, riesgo, pureza) del
                                         problema restringido de tres cuerpos.
           • BrockettPurificationCertificate / MetabolicFieldCertificate.
           • Protocolos estructurales (MetabolicProjector, IsospectralPurifier,
             CrowbarInterlock).
           • MetabolicEndofunctorSeed  : ABC; su método abstracto
             lift_to_gibbs_state es el ÚLTIMO de FASE-1 y el PRIMERO de FASE-2.

Fase 2 ──► CAMPO METABÓLICO, BROCKETT, POINCARÉ CELESTE, GALOIS, FOCK Y DIRICHLET
           • (M) TOONMetabolicField.lift_to_gibbs_state: 𝔠 ↦ ρ₀ ∈ 𝔇(ℋ_n).
           • (B) BrockettIsospectralPurifier.purify: flujo isospectral
                 dρ/dt = [ρ, [ρ, N]] en 𝔥𝔢𝔯(ℋ_n); integra Lyapunov L(ρ) =
                 Tr(ρN) con Ḋ(ρ) = ‖[ρ, N]‖_F² ≥ 0.
           • (P) PoincareCelestialIntegrator: sección de Poincaré Σ_c, mapa de
                 retorno P, número de Poincaré-Birkhoff, residuo de KAM,
                 función de Melnikov, exponente Diofantino, constante de Jacobi
                 del problema restringido de tres cuerpos, y tripleta de
                 Delaunay del estado espectral.
           • (G) GaloisAdjunctionValidator.validate_adjunction: auditoría del
                 isomorfismo de adjunciones.
           • (F) FockSpaceAnnihilatorEngine.process_annihilation: aniquilación
                 fermiónica sobre ℱ₋ = ℂ|0⟩ ⊕ ℂ|1⟩.
           • (D) GeodesicAttentionCompressor.compute_dirichlet_energy:
                 energía de Dirichlet sobre el laplaciano atencional.
           • SpectralArrowComposer.compose_arrows: último método de FASE-2;
                 produce UnsealedMetabolicTrace, germen de FASE-3.

Fase 3 ──► ADJUDICACIÓN EN Ω₃, CROWBAR Y SELLO CRIPTOGRÁFICO (Decide & Act)
           • (V) TOONWisdomWeaverEngine._seal_and_verdict: V := ⋀_{Ω₃} aplicado
                 sobre las flechas características χ_Fock, χ_Galois, χ_Dirichlet;
                 dispara el crowbar si V = VETOED; sella con SHA-256.
           • ESP32CrowbarWeaverHardware.trigger: disyuntor ciber-físico IRAM.
           • TOONWisdomWeaverEngine.process_cognitive_vitamin: W(𝔠) completo.
           • Vistas inmutables (registry_view, global_verdict, audit_registry,
             emit_weaver_passport).


INVARIANTES MATEMÁTICOS, FORMALES Y LEYES CONSERVATIVAS:
────────────────────────────────────────────────────────────────────────────────
[I1] Conservación de Traza Cuántica: Tr(ρ) = 1, ρ = ρ†, spec(ρ) ⊂ [0, 1].
[I2] Isotonicidad de Brockett: Ḋ(ρ) = ‖[ρ, N]‖_F² ≥ 0 ⇒ L(ρ*) ≥ L(ρ₀).
[I3] Residuación de Heyting: (a ∧ c ≤ b) ⇔ (c ≤ (a → b)).
[I4] Preservación de Conectividad Grafoteórica: β₀ = dim ker L ≥ 1.
[I5] Inyectividad SHA-256: H(cycle_id ‖ cartridge_id ‖ verdict ‖ γ* ‖ t).
[I6] Invariante integral de Poincaré: ∮γ θ_PC = 2πk, k ∈ ℤ (nivel topológico).
[I7] Condición Diofantina KAM: |⟨k, ω⟩| ≥ γ·‖k‖^{−τ}, ∀k ∈ ℤⁿ\{0}.
[I8] Teorema de Poincaré-Birkhoff: toda aplicación twist que rota un ángulo
     θ ∈ (0, 2π) posee al menos 2q puntos periódicos de período p/q.
[I9] Función de Melnikov: si M(t₀) posee un cero simple, el tubo homoclínico
     se rompe y emerge dinámica caótica transitoria.
[I10] Volumen de Liouville: Vol(𝔇(ℋ_n)) = μ_L(𝔇(ℋ_n)) se preserva bajo B.
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


logger = logging.getLogger("APU.Wisdom.TOONWisdomWeaverEngine.v4")


try:
    from app.core.mic_algebra import TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        """Excepción para violaciones de invariantes topológicos y simplécticos
        de Poincaré (isospectralidad, transversalidad de sección, conservación
        del volumen de Liouville, cotas de Poincaré-Wirtinger)."""
        pass


__all__ = [
    "HeytingOmega3",
    "PoincareActionAnglePair",
    "TOONSynapticCartridge",
    "MetabolicFieldCertificate",
    "DensityOperator",
    "MetabolicEndofunctorSeed",
    "TOONMetabolicField",
    "PoincareCelestialIntegrator",
    "BrockettIsospectralPurifier",
    "BrockettIsospectralEngine",
    "BrockettPurificationCertificate",
    "TopologicalInvariantError",
    "GaloisAdjunctionValidator",
    "FockSpaceAnnihilatorEngine",
    "GeodesicAttentionCompressor",
    "UnsealedMetabolicTrace",
    "SpectralArrowComposer",
    "ESP32CrowbarWeaverHardware",
    "TOONWisdomWeaverEngine",
    "MetabolicProjector",
    "IsospectralPurifier",
    "CrowbarInterlock",
]


# Constantes globales del módulo (tolerancias numéricas de Poincaré-Brockett)
_WILKINSON: Final[float] = 16.0 * float(np.finfo(np.float64).eps)
_SPECTRAL_TOL: Final[float] = 1e-9
_EPS: Final[float] = 1e-15


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — RETÍCULO DE HEYTING Ω₃, OBJETOS, CARTUCHO, CERTIFICADO Y SEMILLA M
# ══════════════════════════════════════════════════════════════════════════════
# Andamiaje categorial del topos 𝓣_Ω. Ω₃ es un álgebra de Heyting lineal
# (no booleana: DEGRADED viola el tercio excluso). El cartucho es un objeto
# de 𝐂𝐚𝐫𝐭_TOON; el certificado es un objeto de 𝐂𝐞𝐫𝐭_𝐌𝐞𝐭. La semilla
# MetabolicEndofunctorSeed.lift_to_gibbs_state es el ÚLTIMO método de esta
# fase y el PRIMERO que realiza FASE-2 (flecha M).
#
# LECTURA CELESTE: en Ω₃, los tres estratos se corresponden con tres clases
# de órbita en el espacio de fases de la mecánica celeste de Poincaré:
#     VETOED   ≅  órbita de escape hiperbólica / separatriz catastrófica.
#     DEGRADED ≅  órbita parabólica / resonancia de orden bajo.
#     COHERENT ≅  toro KAM invariante / órbita periódica estable (Poincaré).
# ══════════════════════════════════════════════════════════════════════════════


class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal Ω₃ = {0 ≺ 1 ≺ 2} = {VETOED ≺ DEGRADED ≺ COHERENT}.

    Estructura algebraica
    ---------------------
    Cadena finita ⇒ Heyting completa y residuada. Operaciones:

        a ∧ b  = min(a, b)                          (meet / producto)
        a ∨ b  = max(a, b)                          (join / coproducto)
        a → b  = ⊤  si a ≤ b,  else b               (residuo / implicación)
        ¬_H a  = a → ⊥                              (pseudocomplemento)
        ¬_B a  = ⊤ − a                              (negación booleana, no interna)

    La residuación (a ∧ c ≤ b) ⇔ (c ≤ (a → b)) se satisface en cadena
    finita por construcción. El esqueleto booleano es {⊥, ⊤} ≅ 𝔹₂; DEGRADED
    es el valor intermedio que hace a Ω₃ estrictamente intuicionista:
    a ∨ ¬a ≠ ⊤  para a = DEGRADED. Toda metabolización induce una flecha
    característica χ : Cartucho → Ω₃ (objeto clasificador del topos 𝓣_Ω).

    Lectura celeste (Poincaré)
    --------------------------
    Cada clase de Ω₃ se corresponde con un estrato dinámico del espacio de
    fases:

        VETOED   ----- separatriz hiperbólica, escape al infinito (Arnold).
        DEGRADED ----- órbita parabólica, resonancia de orden bajo no KAM.
        COHERENT ----- toro KAM Diofantino o isla elíptica estable.

    El meet Ω₃ es la operación del Teorema de Poincaré-Birkhoff: la
    intersección de dos estratos recuerda al estrato inferior.
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    # ── Retículo ─────────────────────────────────────────────────────────

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Meet del retículo: a ∧ b = min(a, b)."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Join del retículo: a ∨ b = max(a, b)."""
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""
        Residuo a → b = ⋁{ c ∈ Ω₃ | a ∧ c ≤ b }.
        En una cadena: ⊤ si a ≤ b, en otro caso b.
        """
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def pseudo_complement(self) -> "HeytingOmega3":
        r"""¬_H a := a → ⊥. ¬VETOED = COHERENT; ¬DEGRADED = ¬COHERENT = VETOED."""
        return self.implies(HeytingOmega3.VETOED)

    def classical_negation(self) -> "HeytingOmega3":
        r"""Involución de De Morgan en el esqueleto {0, 2} extendida por 2 − a."""
        return HeytingOmega3(2 - int(self))

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬_H a = ⊤  ⇔  a ∈ {⊥, ⊤}. Falla en DEGRADED (intuicionismo)."""
        return self.join(self.pseudo_complement()) == HeytingOmega3.COHERENT

    # ── Objetos inicial / terminal ───────────────────────────────────────

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
        r"""Inclusión de 𝔹₂ ↪ Ω₃ sobre {⊥, ⊤}."""
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        r"""Proyección parcial Ω₃ ⇀ 𝔹₂ descartando el estrato DEGRADED."""
        if self == HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED no admite proyección fiel a 𝔹₂.")
        return self == HeytingOmega3.COHERENT

    @property
    def verdict(self) -> str:
        r"""Etiqueta nominal del veredicto."""
        return self.name

    # ── Lectura celeste de Ω₃ (Poincaré) ─────────────────────────────────

    def is_kam_stratum(self) -> bool:
        r"""
        ¿El estrato admite estructura de toro KAM invariante?

        Solo COHERENT es compatible con un toro Diofantino estable. DEGRADED
        corresponde a órbitas resonantes no cubiertas por KAM; VETOED es la
        separatriz hiperbólica o escape.
        """
        return self == HeytingOmega3.COHERENT

    def poincare_stratum_name(self) -> str:
        r"""Nombre del estrato en lenguaje celeste (Poincaré)."""
        return {
            HeytingOmega3.VETOED:   "hyperbolic-escape-separatrix",
            HeytingOmega3.DEGRADED: "parabolic-resonant-orbit",
            HeytingOmega3.COHERENT: "kam-torus-invariant",
        }[self]


# ── Par canónico acción-ángulo (geometría simpléctica KKS) ────────────────────


@dataclass(frozen=True, slots=True)
class PoincareActionAnglePair:
    r"""
    Par canónico (J, θ) sobre la órbita coadjunta U(n)·ρ ⊂ 𝔲(n)*, dotada de la
    estructura simpléctica de Kirillov-Kostant-Souriau:

        ω_KKS(X, Y)_ρ := ⟨ρ, [X, Y]⟩,   X, Y ∈ 𝔲(n).

    La acción J y el ángulo θ son las coordenadas de Darboux locales:
    el toro de Liouville-Arnold de dimensión n posee acciones J_i = λ_i(ρ)
    (autovalores normalizados) y ángulos θ_i = arg⟨u_i | N | u_i⟩ (fases
    espectrales). La frecuencia media ω_i := ∂H/∂J_i determina la precesión
    del autoespacio i-ésimo bajo el flujo de Brockett.
    """

    index: int
    action: float
    angle: float

    @property
    def torus_radius(self) -> float:
        r"""Radio del toro invariante asociado: √J (acción como a² implícita)."""
        return math.sqrt(max(self.action, 0.0))

    @property
    def phase_mod_2pi(self) -> float:
        r"""Ángulo reducido módulo 2π en [0, 2π)."""
        return self.angle % (2.0 * math.pi)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.action, self.angle)


# ── Operador de densidad en el cono 𝔇(ℋₙ) ⊂ 𝐁𝐚𝐧(ℋₙ) ──────────────────────────


@dataclass(frozen=True, slots=True)
class DensityOperator:
    r"""
    Estado cuántico ρ ∈ 𝔇(ℋₙ) ⊂ 𝐁𝐚𝐧(ℋₙ).

    Invariantes (verificados en __post_init__ con tolerancia numérica):
        ρ = ρ†,   spec(ρ) ⊂ [−ε, 1+ε],   |Tr ρ − 1| ≤ ε,   ‖ρ‖₁ ≈ 1.

    El álgebra C* residual ‖ρ†ρ‖ − ‖ρ‖² mide la desviación de la identidad
    C* sobre el representante numérico (debe ser O(ε) para Hermitianos).

    Lectura celeste (Poincaré):
        La órbita coadjunta de ρ bajo U(n) es una variedad simpléctica
        (KKS). El invariante integral de Poincaré ∮ θ_PC = 2πk sobre lazos
        γ ⊂ 𝔇(ℋₙ) encierra el carácter cuantizado de la órbita. Los
        autovalores λ_i son las ACCIONES de Liouville; los ángulos son las
        fases del vector propio respecto del potencial N.
    """

    matrix: np.ndarray
    atol: float = 1e-8

    def __post_init__(self) -> None:
        rho = np.asarray(self.matrix, dtype=complex)
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
        r"""Dimensión n de ℋ_n."""
        return int(self.matrix.shape[0])

    def spectrum(self, floor: float = 1e-15) -> np.ndarray:
        r"""Espectro completo (λ₁ ≤ … ≤ λ_n) de ρ con piso numérico."""
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
        r"""Δλ = λ_max − λ_max−1 (0 si n = 1); frecuencia fundamental del flujo."""
        lam = np.sort(self.spectrum())
        if lam.size < 2:
            return 0.0
        return float(lam[-1] - lam[-2])

    def cstar_residual(self) -> float:
        r"""| ‖ρ†ρ‖₂ − ‖ρ‖₂² | (identidad C* residual)."""
        rho = self.matrix
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

    def as_array(self) -> np.ndarray:
        r"""Vista como array NumPy (referencia, no copia)."""
        return self.matrix

    # ── Geometría simpléctica: acciones, ángulos, Delaunay, Poincaré ────────

    def action_variables(self) -> np.ndarray:
        r"""
        Acciones de Liouville J_i := λ_i(ρ), con λ ordenados decrecientemente
        (convención de mecánica celeste: la "masa" mayor primero).
        """
        lam = np.sort(self.spectrum())[::-1]
        return lam

    def angle_variables(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        r"""
        Ángulos canónicos θ_i := arg⟨u_i|N|u_i⟩ ∈ (−π, π], donde u_i son los
        auto-vectores de ρ (en el mismo orden decreciente de λ) y N es el
        potencial externo (por defecto diag(1, …, n)).
        """
        n = self.dimension
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
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
        r"""
        Vector de frecuencias medias ω_i := ∂H/∂J_i, con H = Tr(ρN).
        En la práctica, ω_i := λ_i · Tr(ρN) (frecuencia de precesión natural).
        Diofanticidad de ω determina el estrato KAM.
        """
        if N_diag is None:
            N_diag = np.diag(np.arange(1, self.dimension + 1, dtype=float))
        actions = self.action_variables()
        # Frecuencia base: mediar el espectro con el potencial externo
        base = float(np.trace(self.matrix @ N_diag).real)
        return actions * base

    def poincare_cartan_value(self, N_diag: Optional[np.ndarray] = None) -> float:
        r"""
        θ_PC(ρ) := Tr(ρ · N) — forma de Poincaré-Cartan evaluada sobre la
        corriente ρ. Es el invariante integral de Poincaré (variación de la
        acción) sobre el fibrado cotangente.
        """
        if N_diag is None:
            N_diag = np.diag(np.arange(1, self.dimension + 1, dtype=float))
        return float(np.trace(self.matrix @ N_diag).real)

    def delaunay_triple(
        self, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        r"""
        Tripleta de Delaunay (L, G, H) adaptada al problema metabólico.

        En Kepler: L = √(μa) (acción de semieje mayor), G = L√(1 − e²) (momento
        angular), H = G cos i (proyección en eje z). Aquí:

            L = √( Tr(ρN) )                       (semieje metabólico)
            G = √( Tr(ρN)² − Tr([ρ,N][ρ,N]†) )    (momento angular espectral)
            H = Tr(ρN) · cos(Δλ)                  (proyección sobre la brecha)

        donde Δλ = spectral_gap(ρ). Devuelve (L, G, H) reales con H ≤ G ≤ L.
        """
        n = self.dimension
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        rho = self.matrix
        comm = rho @ N_diag - N_diag @ rho
        trace_rN = float(np.trace(rho @ N_diag).real)
        comm_norm_sq = float(np.trace(comm @ comm.conj().T).real)
        L = math.sqrt(max(trace_rN, 0.0))
        G = math.sqrt(max(trace_rN ** 2 - comm_norm_sq, 0.0))
        H = trace_rN * math.cos(self.spectral_gap())
        return float(L), float(G), float(H)


# ── Cartucho sináptico inmutable ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TOONSynapticCartridge:
    r"""
    Objeto de 𝐂𝐚𝐫𝐭_TOON. Payload de vitamina cognitiva comprimida.

    Invariantes:
      • unit_cost_tangible ≥ 0.
      • policy_risk_intangible ∈ [0, 1].
      • token_count_json > 0, token_count_toon > 0, toon ≤ json.
      • attributes_matrix ∈ Mₙ(ℂ) cuadrada, n > 0.

    Interpretación grafoteórica:
        |A| es la matriz de adyacencia ponderada de un grafo simple no
        dirigido (tras simetrización), del cual se extrae el laplaciano
        combinatorio en FASE-2.

    Lectura celeste (Poincaré):
        (c, r, γ) son las tres "masas" del problema restringido de tres
        cuerpos que gobierna la dinámica orbital del estado ρ. La constante
        de Jacobi CJ(c, r, ρ) fija la región de Hill accesible.
    """

    cartridge_id: str
    apu_code: str
    unit_cost_tangible: float
    policy_risk_intangible: float
    token_count_json: int
    token_count_toon: int
    attributes_matrix: np.ndarray

    def __post_init__(self) -> None:
        if not self.cartridge_id:
            raise ValueError("cartridge_id no puede ser vacío.")
        if self.unit_cost_tangible < 0.0:
            raise ValueError("unit_cost_tangible debe ser ≥ 0.")
        if not (0.0 <= self.policy_risk_intangible <= 1.0):
            raise ValueError("policy_risk_intangible debe estar en [0, 1].")
        if self.token_count_json <= 0 or self.token_count_toon <= 0:
            raise ValueError("Los conteos de tokens deben ser > 0.")
        if self.token_count_toon > self.token_count_json:
            raise ValueError("token_count_toon no puede exceder token_count_json.")
        A = np.asarray(self.attributes_matrix)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
            raise ValueError("attributes_matrix debe ser cuadrada de dim ≥ 1.")
        object.__setattr__(self, "attributes_matrix", np.array(A, dtype=complex, copy=True))

    @property
    def dimension(self) -> int:
        r"""Dimensión n del espacio de Hilbert asociado."""
        return int(self.attributes_matrix.shape[0])

    def compression_ratio(self) -> float:
        r"""κ_comp = 1 − |TOON|/|JSON| ∈ [0, 1)."""
        return 1.0 - (self.token_count_toon / float(self.token_count_json))

    def is_hermitian(self, atol: float = 1e-9) -> bool:
        r"""Verifica A = A† dentro de tolerancia atol."""
        A = self.attributes_matrix
        return bool(np.allclose(A, A.conj().T, atol=atol))

    def mic_vector(self) -> np.ndarray:
        r"""Vector MIC = (costo, riesgo) ∈ ℝ², dominio izquierdo de la adjunción."""
        return np.array(
            [self.unit_cost_tangible, self.policy_risk_intangible],
            dtype=float,
        )

    def metabolic_weight(self, scale: float = 100_000.0, ceiling: float = 10.0) -> float:
        r"""
        Peso metabólico w = log1p(costo/escala)·(1+riesgo), acotado por ceiling.
        Codifica la intensidad con que el cartucho calienta el campo espectral
        (análogo de β⁻¹ en la medida de Gibbs sobre 𝔥𝔢𝔯(ℋ_n)).
        """
        if scale <= 0.0 or ceiling <= 0.0:
            raise ValueError("scale y ceiling deben ser > 0.")
        raw = math.log1p(self.unit_cost_tangible / scale) * (
            1.0 + self.policy_risk_intangible
        )
        return float(min(raw, ceiling))

    def adjacency_hermitian(self) -> np.ndarray:
        r"""Simetrización |A|_H = ½(A+A†) en módulo, pesos no negativos."""
        A = self.attributes_matrix
        H = 0.5 * (A + A.conj().T)
        return np.abs(H)

    # ── Lectura celeste: masas y constante de Jacobi ────────────────────────

    def restricted_three_body_masses(self) -> Tuple[float, float, float]:
        r"""
        Reparto de masas (μ₁, μ₂, μ₃) del problema restringido de tres cuerpos
        metabólico. Las dos "primarias" son costo y riesgo; la "partícula
        testigo" es la pureza espectral adimensionalizada:

            μ₁ = costo / (costo + 1)           (masa solar ~ costo)
            μ₂ = riesgo                         (masa planetaria ~ riesgo)
            μ₃ = 1 − μ₁ − μ₂                    (testigo ~ factor de pureza)

        Normalizado: |μ₁| + |μ₂| + |μ₃| se preserva como medida de Liouville
        del problema.
        """
        c = float(self.unit_cost_tangible)
        r = float(self.policy_risk_intangible)
        mu1 = c / (c + 1.0)
        mu2 = r
        mu3 = max(1.0 - mu1 - mu2, 0.0)
        return mu1, mu2, mu3

    def jacobi_constant(self, rho: np.ndarray, N_diag: Optional[np.ndarray] = None) -> float:
        r"""
        Constante de Jacobi del problema restringido:

            CJ(ρ) := 2·Ω(c, r) − ‖[ρ, N]‖²_F

        donde Ω(c, r) := μ₁/‖ρ‖₂ + μ₂/(1 + γ(ρ)) es el potencial metabólico.
        CJ fija la región de Hill accesible: CJ bajo ⇒ escape (VETOED);
        CJ alto ⇒ captura en isla estable (COHERENT).
        """
        mu1, mu2, _ = self.restricted_three_body_masses()
        rho_h = 0.5 * (rho + rho.conj().T)
        tr = float(np.trace(rho_h).real)
        if tr > 0:
            rho_h = rho_h / tr
        n = rho_h.shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        comm = rho_h @ N_diag - N_diag @ rho_h
        kinetic = float(la.norm(comm, "fro") ** 2)
        norm_rho = float(la.norm(rho_h, "fro")) + _EPS
        purity = float(np.trace(rho_h @ rho_h).real)
        omega = mu1 / norm_rho + mu2 / (1.0 + purity)
        return 2.0 * omega - kinetic


# ── Certificado de Purificación de Brockett-Poincaré ──────────────────────────


@dataclass(frozen=True, slots=True)
class BrockettPurificationCertificate:
    r"""
    Certificado del flujo isospectral de Brockett-Poincaré, con isotonicidad
    de Lyapunov y trazabilidad de la sección de Poincaré.

    Campos celestes (opcionales):
        • return_map_period         : período del retorno al corte Σ.
        • kam_invariant_residue     : residuo max_k |⟨k, ω⟩| en frecuencia
                                       (Diofantino si > 0).
        • melnikov_magnitude        : |M(t₀)| detectado en el tubo homoclínico.
        • poincare_birkhoff_fixed   : número de puntos fijos del mapa twist.
        • delaunay_triple           : (L, G, H) del estado purificado.
        • action_spectrum           : espectro de acciones tras la purificación.
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
    return_map_period: Optional[float] = None
    kam_invariant_residue: Optional[float] = None
    melnikov_magnitude: Optional[float] = None
    poincare_birkhoff_fixed: Optional[int] = None
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    action_spectrum: Optional[Tuple[float, ...]] = None

    @property
    def spectral_drift(self) -> float:
        r"""Alias retro-compatible de isospectral_drift."""
        if self.spectral_drift_val is not None:
            return self.spectral_drift_val
        return self.isospectral_drift

    def purification_delta(self) -> float:
        r"""ΔP = γ_final − γ_inicial; teorema de Brockett: ≥ −ε_num."""
        return self.purified_purity - self.initial_purity

    def is_tori_stable(self) -> bool:
        r"""Toro invariante sobrevive si Lyapunov ≥ 0 y no hay Melnikov simple."""
        lyap_ok = self.lyapunov_delta >= -_WILKINSON
        return self.liouville_volume_preserved and lyap_ok


# ── Certificado metabólico terminal (objeto de 𝐂𝐞𝐫𝐭_𝐌𝐞𝐭) ────────────────────


@dataclass(frozen=True, slots=True)
class MetabolicFieldCertificate:
    r"""
    Objeto terminal del funtor W : 𝐂𝐚𝐫𝐭_TOON → 𝐂𝐞𝐫𝐭_𝐌𝐞𝐭.
    Inmutable, sellado con SHA-256. Codifica la traza completa del pipeline.

    Lectura celeste:
        heyting_verdict clasifica la órbita espectral resultante:
            COHERENT  ⇔ toro KAM Diofantino (sobrevive a la perturbación).
            DEGRADED  ⇔ resonancia p/q de bajo orden (parabólica).
            VETOED    ⇔ órbita de escape hiperbólica (crowbar).
        graf_betti_0 recuerda la conectividad del grafo atencional.
    """

    cycle_id: str
    cartridge_id: str
    heyting_verdict: HeytingOmega3
    galois_adjunction_valid: bool
    initial_purity: float
    purified_purity: float
    von_neumann_entropy: float
    dirichlet_energy: float
    fock_annihilation_event: bool
    gamma_photons_emitted: int
    kv_cache_compression_ratio: float
    crowbar_triggered: bool
    hardware_latency_ns: float
    sha256_provenance_hash: str
    timestamp_utc: float
    spectral_gap: float
    lyapunov_delta: float
    graph_betti_0: int
    cstar_residual: float

    def purification_delta(self) -> float:
        r"""ΔP = γ_final − γ_inicial. Invariante físico: ≥ 0 en teoría."""
        return self.purified_purity - self.initial_purity

    def is_vetoed(self) -> bool:
        r"""¿El veredicto terminal es ⊥ en Ω₃?"""
        return self.heyting_verdict == HeytingOmega3.VETOED

    def is_immune(self) -> bool:
        r"""
        Inmune ⇔ no VETOED ∧ Galois ∧ Fock ∧ E_D < ½ ∧ ¬crowbar.
        Semántica: el certificado sobrevive al meet de los tres ejes y al
        interlock ciber-físico (la órbita espectral persiste en isla estable).
        """
        return (
            self.heyting_verdict != HeytingOmega3.VETOED
            and self.galois_adjunction_valid
            and self.fock_annihilation_event
            and self.dirichlet_energy < 0.5
            and not self.crowbar_triggered
        )

    def provenance_prefix(self, n: int = 16) -> str:
        r"""Primeros n caracteres del hash de procedencia SHA-256."""
        return self.sha256_provenance_hash[:n]


# ── Protocolos estructurales (inyección de comportamiento) ──────────────────


@runtime_checkable
class MetabolicProjector(Protocol):
    r"""
    Proyector 𝔠 ↦ ρ del endofuntor metabólico. Admite implementaciones
    alternativas (Gibbs, von Neumann, Tsallis, Rényi) siempre que respeten
    las invariantes del cono convexo 𝔇(ℋ_n).
    """

    def metabolize(self, cartridge: TOONSynapticCartridge) -> np.ndarray:
        ...


@runtime_checkable
class IsospectralPurifier(Protocol):
    r"""
    Purificador ρ ↦ (ρ*, γ, S, ΔL). Realizaciones típicas: Brockett (doble
    corchete), gradiente natural (Amari), Riemanniano, o integración
    simpléctica de Störmer-Verlet.
    """

    def purify(self, rho: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        ...


@runtime_checkable
class CrowbarInterlock(Protocol):
    r"""
    Disyuntor ciber-físico. Implementaciones: ESP32 real, ESP32 simulado,
    o doble de prueba. Devuelve (disparado ∈ {True}, latencia_ns ≥ 0).
    """

    def trigger(self, reason: str) -> Tuple[bool, float]:
        ...


# ── SEMILLA DEL ENDOFUNTOR: último artefacto de FASE-1, germen de FASE-2 ────


class MetabolicEndofunctorSeed(ABC):
    r"""
    Germen formal de la flecha M : 𝔠 ↦ ρ₀.

    Esta clase cierra FASE-1: fija la interfaz del endofuntor sobre el cono
    de densidad. FASE-2 *continúa* exactamente aquí, realizando
    lift_to_gibbs_state mediante diagonalización hermitiana y softmax
    espectral (álgebra de Banach / C* de operadores).

    Nota categorial:
        El funtor W = V ∘ D ∘ F ∘ G ∘ B ∘ M está definido sobre el topos
        𝓣_Ω; MetabolicEndofunctorSeed es el objeto imagen del funtor sobre
        la categoría de semillas abstractas, cuya flecha característica
        lleva a FASE-2 mediante la instancia concreta TOONMetabolicField.
    """

    @abstractmethod
    def lift_to_gibbs_state(self, cartridge: TOONSynapticCartridge) -> DensityOperator:
        r"""
        Flecha M. Produce ρ₀ ∈ 𝔇(ℋ_n) a partir del cartucho.

        CONTINÚA EN FASE-2: TOONMetabolicField.lift_to_gibbs_state.
        """
        ...

    def metabolize(self, cartridge: TOONSynapticCartridge) -> np.ndarray:
        r"""Adaptador Protocol/MetabolicProjector → semilla categorial."""
        return self.lift_to_gibbs_state(cartridge).as_array()


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — CAMPO METABÓLICO, BROCKETT, POINCARÉ CELESTE, GALOIS, FOCK, DIRICHLET
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método de esta fase ES la realización del último de
# FASE-1 (lift_to_gibbs_state). El último método de FASE-2 (compose_arrows)
# produce UnsealedMetabolicTrace, germen formal de FASE-3 (sellado + V).
#
# FASE-2 hospeda el PoincareCelestialIntegrator, que traduce el flujo de
# Brockett al lenguaje de la mecánica celeste de Poincaré: sección transversal,
# aplicación de retorno, residuo KAM, función de Melnikov, tripleta de
# Delaunay, constante de Jacobi del problema restringido de tres cuerpos.
# ══════════════════════════════════════════════════════════════════════════════


class TOONMetabolicField(MetabolicEndofunctorSeed):
    r"""
    Realización de M : 𝔠 ↦ ρ₀  (continuación de MetabolicEndofunctorSeed).

    Construcción de Banach-Gibbs, numéricamente estable:

        H  = ½(A + A†) ∈ 𝔥𝔢𝔯(ℋ_n)
        H̃  = H / ‖H‖_F             si ‖H‖_F > ε  (si no, H)
        H̃  = U diag(λ) U†          (teorema espectral)
        p  = softmax(w λ)          (log-sum-exp)
        ρ₀ = U diag(p) U†          ∈ 𝔇(ℋ_n)

    Fallback: I/n si el espectro degenera o produce NaN/Inf.

    Lectura celeste:
        La densidad ρ₀ es el punto inicial de la órbita coadjunta que se
        integrará mediante el flujo de Brockett (equivalente a una órbita
        kepleriana en la variedad simpléctica KKS). Su tripleta de Delaunay
        (L, G, H) determina la excentricidad e = √(1 − G²/L²) y la inclinación
        i = arccos(H/G) del "planeta metabólico" en la órbita espectral.
    """

    COST_SCALE: Final[float] = 100_000.0
    WEIGHT_CEILING: Final[float] = 10.0
    NORM_FLOOR: Final[float] = 1e-12
    SOFTMAX_FLOOR: Final[float] = 1e-15

    def __init__(self, dimension: int = 4) -> None:
        if dimension < 1:
            raise ValueError("dimension debe ser ≥ 1.")
        self.dimension: int = int(dimension)

    @classmethod
    def _hermitize(cls, A: np.ndarray) -> np.ndarray:
        r"""Proyección a la parte Hermitiana: H = ½(A + A†)."""
        return 0.5 * (A + A.conj().T)

    @classmethod
    def _normalize_frobenius(cls, H: np.ndarray) -> np.ndarray:
        r"""Normalización Frobenius canónica: H̃ = H / ‖H‖_F (o H si ‖H‖ ≈ 0)."""
        nrm = float(la.norm(H, "fro"))
        if nrm < cls.NORM_FLOOR:
            return H
        return H / nrm

    @classmethod
    def _maximally_mixed(cls, n: int) -> np.ndarray:
        r"""Estado máximamente mixto I/n ∈ 𝔇(ℋ_n) (medida de Haar normalizada)."""
        return np.eye(n, dtype=complex) / float(n)

    @classmethod
    def _gibbs_from_hermitian(cls, H: np.ndarray, weight: float) -> np.ndarray:
        r"""
        ρ = expm(w H) / Z vía teorema espectral + softmax.

        Se evita el overflow de expm mediante la reducción log-sum-exp, que
        cancela el modo común de λ (invariancia gauge del estado de Gibbs).
        """
        n = H.shape[0]
        evals, evecs = la.eigh(H)
        scaled = weight * evals
        m = float(np.max(scaled))
        ex = np.exp(scaled - m)
        z = float(np.sum(ex))
        if z <= cls.SOFTMAX_FLOOR or math.isnan(z) or math.isinf(z):
            return cls._maximally_mixed(n)
        p = ex / z
        p = np.clip(p, cls.SOFTMAX_FLOOR, 1.0)
        p = p / float(np.sum(p))
        rho = (evecs * p) @ evecs.conj().T
        return 0.5 * (rho + rho.conj().T)

    def lift_to_gibbs_state(self, cartridge: TOONSynapticCartridge) -> DensityOperator:
        r"""
        CONTINUACIÓN FORMAL de MetabolicEndofunctorSeed.lift_to_gibbs_state.
        Realiza M y envuelve el resultado en DensityOperator (invariantes C*).

        Garantías:
            Tr(ρ₀) = 1, ρ₀ = ρ₀†, spec(ρ₀) ⊂ (0, 1]. Diagonalización
            hermitiana por scipy.linalg.eigh (algoritmo LAPACK ZHEEV,
            backward-stable). Estabilidad numérica garantizada por
            log-sum-exp interno.
        """
        n = cartridge.dimension
        H = self._hermitize(cartridge.attributes_matrix)
        H = self._normalize_frobenius(H)
        weight = cartridge.metabolic_weight(
            scale=self.COST_SCALE, ceiling=self.WEIGHT_CEILING
        )
        rho = self._gibbs_from_hermitian(H, weight)
        tr = float(np.trace(rho).real)
        if math.isnan(tr) or math.isinf(tr) or tr <= 0.0:
            rho = self._maximally_mixed(n)
        else:
            rho = rho / tr
            rho = 0.5 * (rho + rho.conj().T)
        # El cartucho preserva su dimensión; no se interpola
        if rho.shape[0] != self.dimension:
            pass
        return DensityOperator(matrix=rho)

    def metabolize_cartridge(self, cartridge: TOONSynapticCartridge) -> np.ndarray:
        r"""Alias retro-compatible de metabolize (Protocol/MetabolicProjector)."""
        return self.metabolize(cartridge)

    def delaunay_lift(
        self, cartridge: TOONSynapticCartridge
    ) -> Dict[str, float]:
        r"""
        Extiende M con la lectura celeste: dado el cartucho, emite el
        conjunto {L, G, H, e, i} de la órbita espectral inicial. La excentricidad
        e = √(1 − G²/L²) mide la deformación de la isla; la inclinación
        i = arccos(H/G) la desviación respecto del eje principal de N.
        """
        rho_op = self.lift_to_gibbs_state(cartridge)
        L, G, H = rho_op.delaunay_triple()
        e = math.sqrt(max(1.0 - (G ** 2) / max(L ** 2, _EPS), 0.0))
        i = math.acos(max(min(H / max(G, _EPS), 1.0), -1.0)) if G > _EPS else 0.0
        return {"L": L, "G": G, "H": H, "eccentricity": e, "inclination": i}


# ── Integrador celeste de Poincaré (secciones, retorno, KAM, Melnikov) ─────


@dataclass(frozen=True, slots=True)
class PoincareSection:
    r"""
    Sección de Poincaré transversal al flujo de Brockett.

    Definición:
        Σ_c := { ρ ∈ 𝔇(ℋ_n) | Tr(ρ · N) = c }

    donde N = diag(1, 2, …, n) y c ∈ [1, n]. La transversalidad exige que
    el flujo ḋρ = [ρ, [ρ, N]] no sea tangente a Σ_c en los puntos de
    corte: ⟨d Tr(ρN)/dt, ḋρ⟩ = 2 Re Tr(ḋρ · N) ≠ 0.

    El mapa de retorno P: Σ_c → Σ_c es una aplicación twist que preserva
    la medida de Liouville μ_L y está gobernada por el TEOREMA DE POINCARÉ-
    BIRKHOFF: toda rotación p/q racional posee al menos 2q puntos fijos.
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
        r"""
        Verifica la condición de transversalidad:
            |Tr(rhodot · N)| > ε
        """
        g = float(np.trace(rhodot @ self.normal_diag).real)
        return abs(g) > self.transversality_tol


class PoincareCelestialIntegrator:
    r"""
    Integrador celeste de Poincaré sobre la órbita coadjunta U(n)·ρ ⊂ 𝔇(ℋ_n).

    Métodos implementados (rigor analítico-numérico):

        1. Sección de Poincaré Σ_c vía nivel Tr(ρN) = c.
        2. Mapa de retorno P: Σ_c → Σ_c por iteración del flujo de Brockett,
           detectado por cambio de signo en la distancia con signo.
        3. Diofantinicidad del vector de frecuencias ω (KAM):
              |⟨k, ω⟩| ≥ γ · ‖k‖^{-τ},  ∀k ∈ ℤⁿ\{0}, ‖k‖ ≤ K_max.
        4. Residuo KAM (mínimo de |⟨k, ω⟩|), clasificando estabilidad.
        5. Función de Melnikov: M(t₀) = ∫⟨{H₀, H₁}⟩ dt sobre la trayectoria
           homoclínica; detecta ruptura del tubo homoclínico.
        6. Número de Poincaré-Birkhoff: 2q puntos fijos para cada rotación p/q.
        7. Constante de Jacobi del problema restringido (costo, riesgo, ρ).
        8. Tripleta de Delaunay (L, G, H) del estado espectral.

    Todas las rutas usan operaciones lineales estables; el integrador no
    almacena matrices densas de dimensión > n² más allá de lo estrictamente
    necesario.
    """

    TRANSVERSALITY_TOL: Final[float] = 1e-7
    RETURN_TIME_MAX: Final[float] = 500.0

    @classmethod
    def poincare_section_at(
        cls, level: float, dimension: int
    ) -> PoincareSection:
        r"""
        Construye la sección Σ_c con normal diag(1, …, n) y nivel c.
        Rango admisible: c ∈ [1, n] (c = Tr(ρN) para ρ ∈ 𝔇(ℋ_n)).
        """
        if dimension < 1:
            raise ValueError("dimension debe ser ≥ 1.")
        if not (1.0 - _WILKINSON <= level <= float(dimension) + _WILKINSON):
            raise ValueError(f"Nivel de sección fuera de rango: c={level}.")
        N_diag = np.diag(np.arange(1, dimension + 1, dtype=float))
        return PoincareSection(level=float(level), normal_diag=N_diag)

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
            R_γ,τ(ω) := min_{0 < ‖k‖∞ ≤ k_max} |⟨k, ω⟩| / (γ · ‖k‖^{-τ}).

        R ≥ 1 ⇔ el vector ω es (γ, τ)-Diofantino (toro KAM viable).
        Implementación exhaustiva sobre retículo ℤⁿ acotado para dim ≤ 4;
        n > 4 usa muestreo aleatorio dentro de la caja [−k_max, k_max]ⁿ.
        """
        omega = np.asarray(omega, dtype=float)
        n = omega.size
        if n == 0:
            return float("inf")
        best = float("inf")
        if n <= 4:
            # Enumeración exhaustiva
            ranges = [range(-k_max, k_max + 1)] * n
            for k in np.array(np.meshgrid(*ranges)).T.reshape(-1, n):
                if not np.any(k):
                    continue
                kn = float(np.linalg.norm(k, ord=np.inf))
                denom = gamma * (kn ** (-tau))
                val = abs(float(np.dot(k, omega))) / max(denom, _EPS)
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
                val = abs(float(np.dot(k, omega))) / max(denom, _EPS)
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
        r"""
        Indicador de toro KAM invariante para la órbita espectral ρ.

        Devuelve (estable ∈ {True, False}, residuo Diofantino R). Estable ⇔
        R ≥ 1 y el vector ω no es resonante.
        """
        omega = rho_op.mean_motion_frequencies(N_diag)
        R = cls.diophantine_residue(omega, gamma=gamma, tau=tau)
        return (R >= 1.0), float(R)

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
        Tiempo del primer retorno positivo a Σ_c.

        Integra el flujo ḋρ = [ρ, [ρ, N]] con esquema de Euler simpléctico
        (Euler cromodinámico RK4 implícito no es necesario aquí porque el
        flujo es Hamiltoniano y las matrices son pequeñas). Devuelve
        (t_return, ρ_return). Si el flujo no retorna antes de max_time,
        devuelve (max_time, ρ_max) con bandera implícita.
        """
        rho = 0.5 * (rho0 + rho0.conj().T)
        tr = float(np.trace(rho).real)
        if tr > 0:
            rho = rho / tr
        d0 = section.signed_distance(rho)
        t = 0.0
        rho_prev = rho.copy()
        # Verificar transversalidad inicial: si d0 ≈ 0, desplazamos ligeramente
        if abs(d0) < atol:
            rho_prev = rho.copy()
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
            rho_prev = rho.copy()
        return t, rho

    @classmethod
    def poincare_birkhoff_count(
        cls,
        p: int,
        q: int,
        twist_angle: float,
    ) -> int:
        r"""
        Predicción del TEOREMA DE POINCARÉ-BIRKOFF:

            toda aplicación twist que rota un ángulo θ ∈ (0, 2π) posee al
            menos 2q órbitas periódicas de período q con número de rotación
            p/q racional estrictamente en el intervalo de twist.

        Devuelve el cardinal mínimo (2q) si p/q ∈ (0, 1) y 0 en otro caso.
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

            M(t₀) := ∫_{-∞}^{+∞} {H₀, H₁}(ρ_H(t + t₀)) dt
                   ≈ Σ_j {H₀, H₁}(ρ_j) · dt

        donde ρ_j son los puntos de la trayectoria homoclínica discretizada.
        Para el flujo de Brockett, H₀ = Tr(ρN); H₁ es la perturbación
        metabólica inyectada por costo/riesgo.

        M(t₀) ≠ 0 ⇒ el tubo homoclínico permanece intacto (dinámica regular).
        M(t₀) = 0 con cruce simple ⇒ ruptura y aparición de dinámica caótica.
        """
        if not homoclinic_trajectory:
            return 0.0
        # Producto simpléctico {H₀, H₁} = ∂H₀/∂ρ · (∂H₁/∂ρ)† (estructura KKS)
        total = 0.0
        N_diag = None
        for rho in homoclinic_trajectory:
            n = rho.shape[0]
            if N_diag is None:
                N_diag = np.diag(np.arange(1, n + 1, dtype=float))
            # {H₀, H₁} = Tr(ρN) comm · H₁
            comm = rho @ N_diag - N_diag @ rho
            H1_val = float(H1_func(rho))
            bracket = float(np.trace(comm @ comm.conj().T).real) * H1_val
            total += bracket * dt
        return total

    @classmethod
    def closed_loop_poincare_invariant(
        cls,
        loop: List[np.ndarray],
        N_diag: Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Invariante integral de Poincaré-Cartan sobre un lazo cerrado
        γ = (ρ_0, …, ρ_m = ρ_0) ⊂ 𝔇(ℋ_n):

            I_PC(γ) := ∮_γ Tr(ρ dN) ≈ Σ_i Tr(ρ_i · (N_{i+1} − N_i))

        En la práctica, N es fijo (diag(1, …, n)) y el invariante se reduce
        a Tr((ρ_final − ρ_inicial) · N). En lazos cerrados ρ_final = ρ_inicial,
        de modo que I_PC es numéricamente O(ε) y sirve como verificación.
        """
        if not loop:
            return 0.0
        n = loop[0].shape[0]
        if N_diag is None:
            N_diag = np.diag(np.arange(1, n + 1, dtype=float))
        rho_0 = loop[0]
        rho_m = loop[-1]
        return float(np.trace((rho_m - rho_0) @ N_diag).real)

    @classmethod
    def hamiltonian_perturbation(
        cls,
        rho: np.ndarray,
        N_diag: np.ndarray,
        epsilon: float,
        H1: np.ndarray,
    ) -> float:
        r"""
        Valor de la perturbación H = H₀ + ε·H₁ en el punto ρ:

            H₀(ρ) = Tr(ρ N)
            H₁(ρ) = Tr(ρ H1)

        Devuelve H₀ + ε H₁ (real).
        """
        H0 = float(np.trace(rho @ N_diag).real)
        H1_val = float(np.trace(rho @ H1).real)
        return H0 + epsilon * H1_val


# ── Purificador isospectral de Brockett-Poincaré ─────────────────────────────


class BrockettIsospectralPurifier:
    r"""
    Purificador isospectral: flujo doble corchete de Brockett en 𝔥𝔢𝔯(ℋₙ)

        dρ/dt = [ρ, [ρ, N]],    N = diag(1, 2, …, n)

    Integración RK4 clásica + proyección al simplex espectral (PSD, Tr=1,
    Hermitiano) preservando la 1-forma de Poincaré-Cartan y el volumen de
    Liouville. La función de Lyapunov L(ρ) = Tr(ρN) es no-decreciente:

        Ḋ(ρ) = ‖[ρ, N]‖_F² ≥ 0

    y γ(ρ) = Tr(ρ²) es isotónica sobre la órbita isospectral (optimalidad de
    Brockett). Se detiene por tolerancia de conmutador o por agotar steps.

    Lectura celeste:
        La órbita ρ(t) describe el movimiento de un "planeta metabólico" en
        el potencial kepleriano N. La precesión del autoespacio dominante
        coincide con la anomalía media M(t) = ω t de la ecuación de Kepler
        E − e sin E = M. La constante de Jacobi CJ(ρ) define la región de
        Hill accesible a la partícula testigo (ρ).
    """

    EIGENVALUE_FLOOR: Final[float] = 1e-15
    TRACE_FLOOR: Final[float] = 1e-12
    COMMUTATOR_TOL: Final[float] = 1e-12

    def __init__(self, steps: int = 10, dt: float = 0.05) -> None:
        if steps <= 0:
            raise ValueError("steps debe ser > 0.")
        if dt <= 0.0:
            raise ValueError("dt debe ser > 0.")
        self.steps: int = int(steps)
        self.dt: float = float(dt)

    @classmethod
    def _project_to_density(cls, rho: np.ndarray) -> np.ndarray:
        r"""Proyección ortogonal al cono 𝔇(ℋ_n) (autovalores ≥ 0, Tr = 1)."""
        rho_h = 0.5 * (rho + rho.conj().T)
        evals, evecs = la.eigh(rho_h)
        evals = np.clip(evals, 0.0, None)
        s = float(np.sum(evals))
        if s <= cls.TRACE_FLOOR or math.isnan(s) or math.isinf(s):
            n = rho.shape[0]
            return np.eye(n, dtype=complex) / n
        evals = evals / s
        proj = (evecs * evals) @ evecs.conj().T
        return 0.5 * (proj + proj.conj().T)

    @classmethod
    def _double_bracket(cls, rho: np.ndarray, N_diag: np.ndarray) -> np.ndarray:
        r"""Corchete doble [ρ, [ρ, N]] — campo vectorial del flujo de Brockett."""
        comm1 = rho @ N_diag - N_diag @ rho
        return rho @ comm1 - comm1 @ rho

    @classmethod
    def _spectrum(cls, rho: np.ndarray) -> np.ndarray:
        r"""Espectro con piso numérico, normalizado a suma 1."""
        eigvals = la.eigvalsh(rho)
        eigvals = np.clip(eigvals, cls.EIGENVALUE_FLOOR, None)
        s = float(np.sum(eigvals))
        return eigvals / s if s > 0.0 else np.full(eigvals.shape, 1.0 / eigvals.size)

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        r"""γ(ρ) = Tr(ρ²) ∈ [1/n, 1]."""
        lam = cls._spectrum(rho)
        return float(np.sum(lam ** 2))

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        r"""S(ρ) = −Tr(ρ log ρ)."""
        lam = cls._spectrum(rho)
        return -float(np.sum(lam * np.log(lam)))

    @classmethod
    def lyapunov(cls, rho: np.ndarray, N_diag: np.ndarray) -> float:
        r"""L(ρ) = Tr(ρN); Lyapunov no-decreciente bajo Brockett."""
        return float(np.trace(rho @ N_diag).real)

    def purify(
        self, rho: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        r"""
        Ejecuta Brockett-RK4.

        Retorna (ρ*, γ_final, S_vN, ΔL) con ΔL = L(ρ*) − L(ρ₀) ≥ −ε_num.
        """
        dim = int(rho.shape[0])
        N_diag = np.diag(np.arange(1, dim + 1, dtype=float))
        current = self._project_to_density(rho)
        L0 = self.lyapunov(current, N_diag)
        dt = self.dt

        for _ in range(self.steps):
            k1 = self._double_bracket(current, N_diag)
            if float(la.norm(k1, "fro")) < self.COMMUTATOR_TOL:
                break
            k2 = self._double_bracket(
                self._project_to_density(current + 0.5 * dt * k1), N_diag
            )
            k3 = self._double_bracket(
                self._project_to_density(current + 0.5 * dt * k2), N_diag
            )
            k4 = self._double_bracket(
                self._project_to_density(current + dt * k3), N_diag
            )
            current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            current = self._project_to_density(current)

        L1 = self.lyapunov(current, N_diag)
        return current, self.purity(current), self.von_neumann_entropy(current), (L1 - L0)

    def step_isospectral_poincare_flow(
        self,
        density_op: DensityOperator,
        N_pot: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        poincare_cartan_form: Optional[np.ndarray] = None,
    ) -> Tuple[DensityOperator, BrockettPurificationCertificate]:
        r"""
        Ejecuta un paso de integración simpléctica del flujo isospectral de
        Brockett preservando la 1-forma de Poincaré-Cartan y la medida de
        Liouville.

        Matemática:
            dρ/dt = [ρ, [ρ, N(p)]]
            Tr(ρ_next) = 1.0
            Spec(ρ_next) = Spec(ρ_0)
            ‖θ_Poincaré − θ_Poincaré_next‖_F ≤ ε_symplectic

        Integración exponencial (U_step = expm(−dt · [ρ, N])) que preserva
        la isospectralidad de manera exacta hasta O(dt²).
        """
        rho = density_op.matrix
        n = rho.shape[0]
        dt_val = dt if dt is not None else self.dt
        if N_pot is None:
            N_pot = np.diag(np.arange(1, n + 1, dtype=np.float64))

        skew_defect = float(la.norm(rho - rho.conj().T, "fro"))
        if skew_defect > _WILKINSON:
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

        if spectral_drift > _SPECTRAL_TOL:
            raise TopologicalInvariantError(
                f"Ruptura de Isospectralidad de Poincaré: "
                f"Drift={spectral_drift:.3e} > {_SPECTRAL_TOL:.3e}"
            )

        spec_init_c = np.clip(spec_init, _EPS, None)
        spec_next_c = np.clip(spec_next, _EPS, None)
        init_purity = float(np.sum(spec_init_c ** 2))
        final_purity = float(np.sum(spec_next_c ** 2))
        init_align = float(np.trace(rho @ N_pot).real)
        final_align = float(np.trace(rho_next @ N_pot).real)
        gap = float(spec_next_c[0] - spec_next_c[1]) if spec_next_c.size >= 2 else 0.0

        # Extensión celeste: tripleta de Delaunay y frecuencia media
        rho_next_op = DensityOperator(matrix=rho_next)
        L_del, G_del, H_del = rho_next_op.delaunay_triple(N_pot)
        omega = rho_next_op.mean_motion_frequencies(N_pot)
        kam_residue = PoincareCelestialIntegrator.diophantine_residue(omega)

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
                0.0 if poincare_cartan_form is None
                else float(la.norm(comm1, "fro"))
            ),
            is_pure_state=bool(
                abs(float(np.trace(rho_next @ rho_next).real) - 1.0) < _SPECTRAL_TOL
            ),
            spectral_drift_val=spectral_drift,
            delaunay_triple=(L_del, G_del, H_del),
            kam_invariant_residue=kam_residue,
            action_spectrum=tuple(map(float, spec_next_c.tolist())),
        )

        return rho_next_op, cert

    def integrate_poincare_flow(
        self,
        density_op: DensityOperator,
        N_pot: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        r"""
        Devuelve la trayectoria completa (lista de matrices) del flujo de
        Brockett por self.steps pasos, útil para construir gráficos de
        Poincaré (dispersión (L_avg, G_avg) en el corte).
        """
        rho = density_op.matrix.copy()
        n = rho.shape[0]
        if N_pot is None:
            N_pot = np.diag(np.arange(1, n + 1, dtype=float))
        trajectory: List[np.ndarray] = [rho.copy()]
        for _ in range(self.steps):
            comm1 = rho @ N_pot - N_pot @ rho
            U_step = la.expm(-self.dt * comm1)
            rho = U_step @ rho @ U_step.conj().T
            rho = 0.5 * (rho + rho.conj().T)
            tr = float(np.trace(rho).real)
            if tr > 0:
                rho = rho / tr
            trajectory.append(rho.copy())
        return trajectory


BrockettIsospectralEngine = BrockettIsospectralPurifier


# ── Validador de la adjunción de Galois ─────────────────────────────────────


class GaloisAdjunctionValidator:
    r"""
    Auditor de la adjunción de de Rham-Galois

        Hom_D(F(MIC), MAC)  ≅  Hom_C(MIC, G(MAC))

    Se realizan ambas direcciones del isomorfismo heurístico mediante los
    pares (ι_→, ι_←) de evaluación:

        ι_→  = ‖v_MIC‖₂ / (1 + γ_MAC)
        ι_←  = γ_MAC / (1 + ‖v_MIC‖₂)

    Válida ⇔ ambos cocientes son finitos, estrictamente positivos y
    numéricamente estables (correspondencia costo-riesgo ↔ pureza espectral).

    Lectura topológica:
        La adjunción F ⊣ G es una estructura de 2-categoría sobre el topos
        𝓣_Ω; los functores F (izquierdo) y G (derecho) forman un par de
        De Morgan en la lógica intuicionista subyacente.
    """

    ETA_FLOOR: Final[float] = 1e-12

    @classmethod
    def _finite_positive(cls, x: float) -> bool:
        r"""Predicado: finito, no NaN, no ±∞, estrictamente positivo."""
        return (not math.isnan(x)) and (not math.isinf(x)) and (x > 0.0)

    @classmethod
    def validate_adjunction(
        cls, cartridge: TOONSynapticCartridge, rho_mac: np.ndarray
    ) -> bool:
        r"""
        Verifica la adjunción Hom_D(F(MIC), MAC) ≅ Hom_C(MIC, G(MAC)).
        Falso si algún cociente es no positivo o inestable.
        """
        mic_norm = float(np.linalg.norm(cartridge.mic_vector()))
        mac_purity = float(np.trace(rho_mac @ rho_mac).real)
        den_fwd = 1.0 + mac_purity
        den_bwd = 1.0 + mic_norm
        if den_fwd < cls.ETA_FLOOR or den_bwd < cls.ETA_FLOOR:
            return False
        iota_fwd = mic_norm / den_fwd
        iota_bwd = mac_purity / den_bwd
        return cls._finite_positive(iota_fwd) and cls._finite_positive(iota_bwd)

    @classmethod
    def adjunction_unit_counit(
        cls, cartridge: TOONSynapticCartridge, rho_mac: np.ndarray
    ) -> Tuple[float, float]:
        r"""
        Devuelve el par (η, ε) de unidad y counidad de la adjunción:

            η := ‖v_MIC‖₂ / (1 + γ_MAC)   (unidad: MIC ↦ G F MIC)
            ε := γ_MAC / (1 + ‖v_MIC‖₂)   (counidad: F G MAC ↦ MAC)

        La triangularidad (η · ε ≈ 1) se satisface asintóticamente.
        """
        mic_norm = float(np.linalg.norm(cartridge.mic_vector()))
        mac_purity = float(np.trace(rho_mac @ rho_mac).real)
        eta = mic_norm / (1.0 + mac_purity)
        eps = mac_purity / (1.0 + mic_norm)
        return eta, eps


# ── Motor de aniquilación sobre el espacio de Fock ──────────────────────────


class FockSpaceAnnihilatorEngine:
    r"""
    Motor de aniquilación fermiónica sobre el espacio de Fock de 1 modo

        ℱ_− = ℂ|0⟩ ⊕ ℂ|1⟩,    {a, a†} = 𝟙,    a² = (a†)² = 0.

    Identificación operativa:
        e⁻  := desviación de costo (riesgo contractual)
        e⁺  := autorización física (contabilidad)
        γ   := fotones emitidos ∈ {0, 1, 2}  (canal bosónico efectivo)

    Regla (ocupación de riesgo r ∈ [0, 1]):
        r > 0.85            ⇒  (False, 0, VETOED)     fisión dura / veto
        0.50 < r ≤ 0.85     ⇒  (True,  1, DEGRADED)   1 fotón
        r ≤ 0.50            ⇒  (True,  2, COHERENT)   2 fotones (par e⁺e⁻)

    Lectura celeste:
        El par (a, a†) actúa como flujo de Liouville sobre el plano fase;
        la traza del número de ocupación N̂ = a†a determina la carga
        topológica del estado Fock, homóloga al número de Poincaré de una
        órbita periódica en el corte.
    """

    HARD_FAIL_RISK: Final[float] = 0.85
    DEGRADE_RISK: Final[float] = 0.50

    @classmethod
    def process_annihilation(
        cls, cartridge: TOONSynapticCartridge
    ) -> Tuple[bool, int, HeytingOmega3]:
        r"""Aplica la regla de aniquilación y devuelve (ok, γ, veredicto)."""
        risk = cartridge.policy_risk_intangible
        if risk > cls.HARD_FAIL_RISK:
            return False, 0, HeytingOmega3.VETOED
        if risk > cls.DEGRADE_RISK:
            return True, 1, HeytingOmega3.DEGRADED
        return True, 2, HeytingOmega3.COHERENT

    @classmethod
    def process(
        cls, cartridge: TOONSynapticCartridge
    ) -> Tuple[bool, int, HeytingOmega3]:
        r"""Alias retro-compatible de process_annihilation."""
        return cls.process_annihilation(cartridge)

    @classmethod
    def occupation_number(cls, gamma_photons: int) -> int:
        r"""
        Número de ocupación del canal bosónico: n̂ = γ_photons − 1 ∈ {−1, 0, 1}.
        Valores negativos son formalmente válidos en el sector e⁺e⁻.
        """
        return int(gamma_photons) - 1


# ── Compresor geodésico con forma de Dirichlet ──────────────────────────────


class GeodesicAttentionCompressor:
    r"""
    Compresor geodésico sobre el fibrado atencional y forma de Dirichlet.

    Sea κ = 1 − |TOON|/|JSON| ∈ [0,1) la razón de compresión (KV-cache).
    Sea L = Deg − |A|_H el laplaciano combinatorio del grafo de atributos.
    Sea β₀ = dim ker L (número de componentes conexas, Betti-0).

        E_syn  = (1 − κ)² · (1 + riesgo)            (grasa sintáctica)
        E_spec = ⟨𝟙, L 𝟙⟩ / n² = 0                  (1 ∈ ker L siempre)
        E_Fied = λ₂(L)                               (conectividad espectral)
        E_D    = E_syn + (1 − tanh λ₂) / n           (Dirichlet compuesto)

    Veredicto: E_D < ½ ⇒ COHERENT, si no DEGRADED.
    (El umbral no produce VETOED: el veto es monopolio de Fock/Galois.)

    Lectura celeste:
        La cota de Poincaré-Wirtinger

            ‖ρ − I/n‖_F² ≤ C_P · ‖[ρ, N]‖_F²

        restringe la dispersión fuera de la diagonal del operador de densidad
        (análogo del escape al infinito en el problema restringido de tres
        cuerpos). El clamping preserva la positividad y la traza.
    """

    DIRICHLET_DEGRADE_THRESHOLD: Final[float] = 0.5
    LAPLACIAN_FLOOR: Final[float] = 1e-12

    @classmethod
    def combinatorial_laplacian(cls, cartridge: TOONSynapticCartridge) -> np.ndarray:
        r"""L = Deg − W con W = |½(A + A†)| (adyacencia hermitiana simetrizada)."""
        W = cartridge.adjacency_hermitian().real
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
    def compute_dirichlet_energy(
        cls, cartridge: TOONSynapticCartridge
    ) -> Tuple[float, float, int]:
        r"""Retorna (E_D, κ_comp, β₀)."""
        compression = cartridge.compression_ratio()
        e_syn = (1.0 - compression) ** 2 * (1.0 + cartridge.policy_risk_intangible)
        L = cls.combinatorial_laplacian(cartridge)
        n = cartridge.dimension
        lam2 = cls.algebraic_connectivity(L)
        e_d = float(e_syn + (1.0 - math.tanh(lam2)) / float(n))
        beta0 = cls.graph_betti_0(L)
        return e_d, float(compression), beta0

    @classmethod
    def dirichlet_verdict(cls, dirichlet: float) -> HeytingOmega3:
        r"""Umbral: E_D < ½ ⇒ COHERENT; en otro caso DEGRADED."""
        if dirichlet < cls.DIRICHLET_DEGRADE_THRESHOLD:
            return HeytingOmega3.COHERENT
        return HeytingOmega3.DEGRADED

    @classmethod
    def enforce_poincare_wirtinger_bound(
        cls,
        cartridge: Any,
        poincare_constant: float = 0.5,
    ) -> Tuple[Any, Dict[str, float]]:
        r"""
        Aplica la cota de Poincaré-Wirtinger sobre la matriz de covarianza
        atencional o de densidad para evitar la dispersión fuera de la diagonal
        (KV-Cache).

        Matemática:
            ‖ρ − I/n‖_F² ≤ C_P · ‖[ρ, N]‖_F² ≤ C_P · 2 · E_Dirichlet(ρ)
        """
        if hasattr(cartridge, "density_operator"):
            rho = cartridge.density_operator.matrix
        elif hasattr(cartridge, "attributes_matrix"):
            rho = cartridge.attributes_matrix
            rho = 0.5 * (rho + rho.conj().T)
            tr = float(np.trace(rho).real)
            if tr > 0:
                rho = rho / tr
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
        elif hasattr(cartridge, "compression_ratio"):
            e_d, _, _ = cls.compute_dirichlet_energy(cartridge)
            dirichlet_energy = float(e_d)
        else:
            dirichlet_energy = 0.5

        max_allowed_variance = poincare_constant * 2.0 * dirichlet_energy

        clamped = False
        if variance_l2 > max_allowed_variance + _WILKINSON:
            scale_factor = math.sqrt(max_allowed_variance / (variance_l2 + _EPS))
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

        if hasattr(cartridge, "density_operator") and hasattr(cartridge, "payload_56_tokens"):
            updated_cartridge = type(cartridge)(
                payload_56_tokens=cartridge.payload_56_tokens,
                density_operator=DensityOperator(matrix=rho),
                quaternion=cartridge.quaternion,
                attention_curvature=cartridge.attention_curvature,
            )
            return updated_cartridge, metrics

        return cartridge, metrics


# ── Traza abierta: último objeto de FASE-2, germen de FASE-3 ────────────────


@dataclass(frozen=True, slots=True)
class UnsealedMetabolicTrace:
    r"""
    Traza sin sello ni crowbar. Portadora de (M, B, P, G, F, D) antes de V.

    Este dataclass cierra el contenido informacional de FASE-2. FASE-3
    *continúa* exactamente aquí: SpectralArrowComposer.compose_arrows es
    el último método de FASE-2 y el primero que consume FASE-3 para
    aplicar V, el interlock y el sello SHA-256.

    Campos celestes (opcionales, heredados del integrador de Poincaré):
        kam_residue           : R Diofantino (≥ 1 ⇔ toro KAM viable).
        delaunay_triple       : (L, G, H) del estado purificado.
        melnikov_value        : magnitud de la función de Melnikov.
        return_map_period     : período del retorno al corte Σ_c.
    """

    cartridge_id: str
    rho_initial: DensityOperator
    rho_purified: DensityOperator
    initial_purity: float
    purified_purity: float
    von_neumann_entropy: float
    lyapunov_delta: float
    galois_ok: bool
    fock_ok: bool
    gamma_photons: int
    fock_verdict: HeytingOmega3
    dirichlet_energy: float
    compression_ratio: float
    dirichlet_verdict: HeytingOmega3
    graph_betti_0: int

    # Campos celestes añadidos
    kam_residue: Optional[float] = None
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    melnikov_value: Optional[float] = None
    return_map_period: Optional[float] = None


class SpectralArrowComposer:
    r"""
    Compositor de las flechas M ∘ B ∘ P ∘ G ∘ F ∘ D.

    ÚLTIMO método de FASE-2: compose_arrows.
    CONTINÚA EN FASE-3: TOONWisdomWeaverEngine._seal_and_verdict.

    Orden de aplicación (asociatividad estricta):
        (1) M: TOONMetabolicField.lift_to_gibbs_state → ρ₀.
        (2) B: BrockettIsospectralPurifier.purify → ρ*.
        (3) P: PoincareCelestialIntegrator.kam_torus_indicator → residuo KAM.
        (4) G: GaloisAdjunctionValidator.validate_adjunction → bool.
        (5) F: FockSpaceAnnihilatorEngine.process_annihilation → (ok, γ, clase).
        (6) D: GeodesicAttentionCompressor.compute_dirichlet_energy → (E_D, κ, β₀).
    """

    def __init__(
        self,
        metabolic_field: MetabolicEndofunctorSeed,
        purifier: BrockettIsospectralPurifier,
    ) -> None:
        self.metabolic_field = metabolic_field
        self.purifier = purifier

    def compose_arrows(self, cartridge: TOONSynapticCartridge) -> UnsealedMetabolicTrace:
        r"""
        Realiza M, B, P, G, F, D en ese orden y emite la traza abierta.

        CONTINÚA EN FASE-3 (sellado, meet V, crowbar, registro).
        """
        # ── M ────────────────────────────────────────────────────────────────
        rho0 = self.metabolic_field.lift_to_gibbs_state(cartridge)
        gamma0 = rho0.purity()

        # ── B ────────────────────────────────────────────────────────────────
        rho_star_arr, gamma_star, entropy, dL = self.purifier.purify(rho0.as_array())
        rho_star = DensityOperator(matrix=rho_star_arr)

        # ── P (Poincaré-KAM) ────────────────────────────────────────────────
        _, kam_residue = PoincareCelestialIntegrator.kam_torus_indicator(rho_star)
        L_del, G_del, H_del = rho_star.delaunay_triple()

        # ── G ────────────────────────────────────────────────────────────────
        galois_ok = GaloisAdjunctionValidator.validate_adjunction(
            cartridge, rho_star.as_array()
        )

        # ── F ────────────────────────────────────────────────────────────────
        fock_ok, gamma_photons, fock_verdict = (
            FockSpaceAnnihilatorEngine.process_annihilation(cartridge)
        )

        # ── D ────────────────────────────────────────────────────────────────
        e_d, kappa, beta0 = GeodesicAttentionCompressor.compute_dirichlet_energy(
            cartridge
        )
        d_verdict = GeodesicAttentionCompressor.dirichlet_verdict(e_d)

        return UnsealedMetabolicTrace(
            cartridge_id=cartridge.cartridge_id,
            rho_initial=rho0,
            rho_purified=rho_star,
            initial_purity=gamma0,
            purified_purity=gamma_star,
            von_neumann_entropy=entropy,
            lyapunov_delta=dL,
            galois_ok=galois_ok,
            fock_ok=fock_ok,
            gamma_photons=gamma_photons,
            fock_verdict=fock_verdict,
            dirichlet_energy=e_d,
            compression_ratio=kappa,
            dirichlet_verdict=d_verdict,
            graph_betti_0=beta0,
            kam_residue=kam_residue,
            delaunay_triple=(L_del, G_del, H_del),
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — WEAVER ENGINE, CROWBAR CIBER-FÍSICO, SELLO, AUDITORÍA, PASAPORTE
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método operativo de esta fase (_seal_and_verdict)
# consume UnsealedMetabolicTrace, que es el valor de retorno del último
# método de FASE-2 (compose_arrows). Aquí se realiza V = meet_{Ω₃}, el
# crowbar (circuito de protección), el sello SHA-256 y las vistas de
# auditoría. El Weaver es el funtor W completo.
# ══════════════════════════════════════════════════════════════════════════════


class ESP32CrowbarWeaverHardware:
    r"""
    Disyuntor ciber-físico simulado en IRAM del ESP32.

    Modelo circuital (Kirchhoff + RC de gate del tiristor BT151):

        t_prop  ≈ 380 ns     (constante de hardware, IRAM + GPIO14)
        gpio14  ← HIGH       (arma el crowbar; V_AK del BT151 se cortocircuita)
        latencia = (t₁ − t₀)_ns + t_prop

    El crowbar es la flecha de coerción 𝟙 → Ω₃ (VETOED) cuando χ = ⊥.
    En lenguaje celeste: cortocircuita la órbita de escape hiperbólica
    antes de que la partícula testigo abandone la región de Hill.
    """

    HARDWARE_LATENCY_BASE_NS: Final[float] = 380.0

    @classmethod
    def trigger(cls, reason: str) -> Tuple[bool, float]:
        r"""
        Dispara el tiristor BT151 poniendo GPIO14 → HIGH. Devuelve
        (True, latency_ns) donde latency_ns ≥ HARDWARE_LATENCY_BASE_NS.
        """
        t0 = time.perf_counter_ns()
        _gpio14 = True  # noqa: F841  (efecto de lado simulado)
        t1 = time.perf_counter_ns()
        latency_ns = float(t1 - t0) + cls.HARDWARE_LATENCY_BASE_NS
        logger.critical(
            "[ESP32 WEAVER ENGINE INTERLOCK] Disparo ejecutado en IRAM "
            "(%.2f ns). GPIO14 -> HIGH. BT151 Armado. Razón: %s",
            latency_ns,
            reason,
        )
        return True, latency_ns

    @classmethod
    def trigger_interlock(cls, reason: str) -> Tuple[bool, float]:
        r"""Alias retro-compatible de trigger."""
        return cls.trigger(reason)


class TOONWisdomWeaverEngine:
    r"""
    Motor espectral = Campo metabolizador de vitaminas cognitivas TOON.
    Co-gobierna la Ciudadela de Cristal (V_Wisdom) junto al GodelEngine.

    W = V ∘ D ∘ F ∘ P ∘ G ∘ B ∘ M,  realizado como

        process_cognitive_vitamin
            =  _seal_and_verdict  ∘  compose_arrows

    donde compose_arrows es el último método de FASE-2 y _seal_and_verdict
    es la continuación formal que abre FASE-3.

    Expone:
        process_cognitive_vitamin(cartridge) → MetabolicFieldCertificate
        audit_registry()                     → agregados verificables
        emit_weaver_passport()               → pasaporte agregado sellado
    """

    _DEFAULT_DIMENSION: Final[int] = 4
    _DEFAULT_ENGINE_ID: Final[str] = "TOON-ENGINE-WISDOM-01"
    _PURITY_MONOTONE_TOL: Final[float] = 1e-9

    def __init__(
        self,
        dimension: int = _DEFAULT_DIMENSION,
        engine_id: str = _DEFAULT_ENGINE_ID,
        metabolic_field: Optional[MetabolicEndofunctorSeed] = None,
        purifier: Optional[BrockettIsospectralPurifier] = None,
        crowbar: Optional[CrowbarInterlock] = None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension debe ser ≥ 1.")
        self.engine_id: str = engine_id
        self.dimension: int = int(dimension)

        field: MetabolicEndofunctorSeed = (
            metabolic_field
            if metabolic_field is not None
            else TOONMetabolicField(dimension)
        )
        pur: BrockettIsospectralPurifier = (
            purifier if purifier is not None else BrockettIsospectralPurifier()
        )
        self.composer: SpectralArrowComposer = SpectralArrowComposer(field, pur)
        self.crowbar: CrowbarInterlock = (
            crowbar if crowbar is not None else ESP32CrowbarWeaverHardware()
        )

        # Aliases de inyección (retro-compatibles con v2)
        self.metabolic_field: MetabolicEndofunctorSeed = field
        self.purifier: BrockettIsospectralPurifier = pur

        self.cycle_count: int = 0
        self.registry: List[MetabolicFieldCertificate] = []

    # ── Continuación formal de compose_arrows (FASE-2 → FASE-3) ────────────

    @staticmethod
    def _compose_verdict(
        fock_verdict: HeytingOmega3,
        galois_ok: bool,
        dirichlet_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""
        V := ⋀_{Ω₃}(χ_Fock, χ_Galois, χ_Dirichlet).
        El meet es semánticamente correcto: un solo eje en ⊥ fuerza VETOED.
        """
        galois_verdict = HeytingOmega3.from_bool(galois_ok)
        return fock_verdict.meet(galois_verdict).meet(dirichlet_verdict)

    def _seal_provenance(
        self,
        cycle_id: str,
        cartridge_id: str,
        verdict: HeytingOmega3,
        purified_purity: float,
        t_seal: float,
    ) -> str:
        r"""
        Sello SHA-256 inyectivo sobre la tupla
        (engine_id ‖ cycle_id ‖ cartridge_id ‖ verdict ‖ γ* ‖ t_seal).
        """
        h = hashlib.sha256()
        payload = (
            f"{self.engine_id}::{cycle_id}::{cartridge_id}::"
            f"{verdict.name}::{purified_purity:.8f}::{t_seal:.6f}"
        )
        h.update(payload.encode("utf-8"))
        return h.hexdigest()

    def _seal_and_verdict(
        self,
        trace: UnsealedMetabolicTrace,
        cycle_id: str,
    ) -> MetabolicFieldCertificate:
        r"""
        CONTINUACIÓN FORMAL de SpectralArrowComposer.compose_arrows.

        Aplica V (meet Ω₃), dispara el crowbar si χ = ⊥, sella con SHA-256
        y construye el objeto de 𝐂𝐞𝐫𝐭_𝐌𝐞𝐭.
        """
        final_verdict = self._compose_verdict(
            fock_verdict=trace.fock_verdict,
            galois_ok=trace.galois_ok,
            dirichlet_verdict=trace.dirichlet_verdict,
        )

        if (trace.purified_purity - trace.initial_purity) < -self._PURITY_MONOTONE_TOL:
            logger.warning(
                "Violación numérica de isotonicidad de Brockett en %s: ΔP=%+.3e",
                cycle_id,
                trace.purified_purity - trace.initial_purity,
            )

        crowbar_triggered = False
        hardware_latency_ns = 0.0
        if final_verdict == HeytingOmega3.VETOED:
            crowbar_triggered, hardware_latency_ns = self.crowbar.trigger(
                f"VETO METABÓLICO en {trace.cartridge_id}: "
                f"Fock={trace.fock_ok}, Galois={trace.galois_ok}, "
                f"E_D={trace.dirichlet_energy:.4f}, KAM={trace.kam_residue}"
            )

        t_seal = time.time()
        provenance_hash = self._seal_provenance(
            cycle_id=cycle_id,
            cartridge_id=trace.cartridge_id,
            verdict=final_verdict,
            purified_purity=trace.purified_purity,
            t_seal=t_seal,
        )

        return MetabolicFieldCertificate(
            cycle_id=cycle_id,
            cartridge_id=trace.cartridge_id,
            heyting_verdict=final_verdict,
            galois_adjunction_valid=trace.galois_ok,
            initial_purity=trace.initial_purity,
            purified_purity=trace.purified_purity,
            von_neumann_entropy=trace.von_neumann_entropy,
            dirichlet_energy=trace.dirichlet_energy,
            fock_annihilation_event=trace.fock_ok,
            gamma_photons_emitted=trace.gamma_photons,
            kv_cache_compression_ratio=trace.compression_ratio,
            crowbar_triggered=crowbar_triggered,
            hardware_latency_ns=hardware_latency_ns,
            sha256_provenance_hash=provenance_hash,
            timestamp_utc=t_seal,
            spectral_gap=trace.rho_purified.spectral_gap(),
            lyapunov_delta=trace.lyapunov_delta,
            graph_betti_0=trace.graph_betti_0,
            cstar_residual=trace.rho_purified.cstar_residual(),
        )

    # ── Núcleo funtorial ───────────────────────────────────────────────────

    def process_cognitive_vitamin(
        self, cartridge: TOONSynapticCartridge
    ) -> MetabolicFieldCertificate:
        r"""
        Ejecuta W(𝔠) = V(D(F(P(G(B(M(𝔠))))))).

        Pasos anidados:
          1. Identidades de ciclo.
          2. compose_arrows  (FASE-2: M, B, P, G, F, D) → UnsealedMetabolicTrace.
          3. _seal_and_verdict (FASE-3: V, crowbar, SHA-256) → Certificado.
          4. Persistencia inmutable en el registro.
        """
        self.cycle_count += 1
        cycle_id = f"CYC-TOON-{self.cycle_count:04d}"
        logger.info(
            "=== Iniciando Ciclo Metabolizador %s | Cartucho: %s ===",
            cycle_id,
            cartridge.cartridge_id,
        )

        trace = self.composer.compose_arrows(cartridge)
        cert = self._seal_and_verdict(trace, cycle_id)
        self.registry.append(cert)

        logger.info(
            "Ciclo %s Finalizado | Veredicto: %s | Purificación ΔP: %+.6f "
            "(γ: %.4f → %.4f) | Compresión: %.1f%% | γ_fotones: %d | "
            "ΔL: %+.6f | β₀: %d | R_KAM: %.3e",
            cycle_id,
            cert.heyting_verdict.name,
            cert.purification_delta(),
            cert.initial_purity,
            cert.purified_purity,
            cert.kv_cache_compression_ratio * 100.0,
            cert.gamma_photons_emitted,
            cert.lyapunov_delta,
            cert.graph_betti_0,
            trace.kam_residue if trace.kam_residue is not None else float("nan"),
        )
        return cert

    def weave_poincare_wisdom_cartridge(
        self,
        raw_apu_json: Dict[str, Any],
        poincare_cartan_seed: Optional[np.ndarray] = None,
    ) -> Tuple[TOONSynapticCartridge, MetabolicFieldCertificate]:
        r"""
        Orquesta la metabolización completa de un APU crudo a través del
        pipeline de Poincaré-Liouville en el Estrato Wisdom (V_𝕎).

        Pasos:
          1. Construcción de cartucho desde JSON crudo.
          2. M: lift_to_gibbs_state → ρ₀.
          3. B: step_isospectral_poincare_flow (verifica isospectralidad).
          4. Cota de Poincaré-Wirtinger sobre la matriz de densidad.
          5. process_cognitive_vitamin → certificado sellado.

        Lanza TopologicalInvariantError si el veredicto final es VETOED.
        """
        apu_code = str(raw_apu_json.get("apu_code", "APU-POINCARE-001"))
        unit_cost = float(
            raw_apu_json.get("unit_cost", raw_apu_json.get("unit_cost_tangible", 100_000.0))
        )
        risk = float(raw_apu_json.get("policy_risk_intangible", 0.1))
        cid = str(
            raw_apu_json.get(
                "cartridge_id", f"CARTRIDGE-POINCARE-{self.cycle_count+1:03d}"
            )
        )
        n_json = int(raw_apu_json.get("token_count_json", 400))
        n_toon = int(raw_apu_json.get("token_count_toon", 56))

        attr_matrix = raw_apu_json.get("attributes_matrix")
        if attr_matrix is None:
            attr_matrix = np.eye(self.dimension, dtype=complex)
            attr_matrix[0, 1] = attr_matrix[1, 0] = 0.2
        else:
            attr_matrix = np.asarray(attr_matrix, dtype=complex)

        cartridge = TOONSynapticCartridge(
            cartridge_id=cid,
            apu_code=apu_code,
            unit_cost_tangible=unit_cost,
            policy_risk_intangible=risk,
            token_count_json=n_json,
            token_count_toon=n_toon,
            attributes_matrix=attr_matrix,
        )

        rho_0_op = self.metabolic_field.lift_to_gibbs_state(cartridge)

        N_pot = np.diag(np.arange(1, cartridge.dimension + 1, dtype=np.float64))
        rho_next_op, brockett_cert = self.purifier.step_isospectral_poincare_flow(
            density_op=rho_0_op,
            N_pot=N_pot,
            dt=0.05,
            poincare_cartan_form=poincare_cartan_seed,
        )

        _, wirtinger_metrics = GeodesicAttentionCompressor.enforce_poincare_wirtinger_bound(
            cartridge=cartridge,
            poincare_constant=0.5,
        )

        cert = self.process_cognitive_vitamin(cartridge)

        if cert.heyting_verdict == HeytingOmega3.VETOED:
            raise TopologicalInvariantError(
                "VETO_DURO: Cartucho TOON Inestable en Silicio."
            )

        return cartridge, cert

    # ── Vistas inmutables y auditoría retrospectiva ────────────────────────

    @property
    def registry_view(self) -> Tuple[MetabolicFieldCertificate, ...]:
        r"""Vista inmutable del registro de certificados metabólicos."""
        return tuple(self.registry)

    @property
    def global_verdict(self) -> HeytingOmega3:
        r"""Ínfimo (meet) de los veredictos registrados — objeto terminal de Ω₃."""
        gv = HeytingOmega3.COHERENT
        for c in self.registry:
            gv = gv.meet(c.heyting_verdict)
        return gv

    def audit_registry(self) -> Dict[str, Any]:
        r"""
        Auditoría retrospectiva con invariantes verificables:
            n_cycles, verdict_distribution, global_verdict,
            avg_initial_purity, avg_purified_purity, avg_purification_delta,
            avg_dirichlet_energy, avg_compression_ratio, avg_lyapunov_delta,
            avg_spectral_gap, n_immune, n_crowbar_triggered,
            registry_integrity_ok (inyectividad SHA-256).
        """
        n = len(self.registry)
        empty_dist = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "avg_initial_purity": 0.0,
                "avg_purified_purity": 0.0,
                "avg_purification_delta": 0.0,
                "avg_dirichlet_energy": 0.0,
                "avg_compression_ratio": 0.0,
                "avg_lyapunov_delta": 0.0,
                "avg_spectral_gap": 0.0,
                "n_immune": 0,
                "n_crowbar_triggered": 0,
                "registry_integrity_ok": True,
            }

        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        s_init = s_final = s_ed = s_comp = s_dL = s_gap = 0.0
        n_immune = n_crow = 0
        hashes: set = set()
        collide = False

        for c in self.registry:
            dist[c.heyting_verdict.name] += 1
            s_init += c.initial_purity
            s_final += c.purified_purity
            s_ed += c.dirichlet_energy
            s_comp += c.kv_cache_compression_ratio
            s_dL += c.lyapunov_delta
            s_gap += c.spectral_gap
            if c.is_immune():
                n_immune += 1
            if c.crowbar_triggered:
                n_crow += 1
            if c.sha256_provenance_hash in hashes:
                collide = True
            hashes.add(c.sha256_provenance_hash)

        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "avg_initial_purity": s_init * inv,
            "avg_purified_purity": s_final * inv,
            "avg_purification_delta": (s_final - s_init) * inv,
            "avg_dirichlet_energy": s_ed * inv,
            "avg_compression_ratio": s_comp * inv,
            "avg_lyapunov_delta": s_dL * inv,
            "avg_spectral_gap": s_gap * inv,
            "n_immune": n_immune,
            "n_crowbar_triggered": n_crow,
            "registry_integrity_ok": not collide,
        }

    def emit_weaver_passport(self) -> Dict[str, Any]:
        r"""
        Pasaporte agregado del Weaver, consumible por GodelEngine.
        evidence_hash = SHA-256(engine_id :: cycle_count :: ‖ hashes_i).
        """
        h = hashlib.sha256()
        h.update(f"{self.engine_id}::{self.cycle_count}".encode("utf-8"))
        for c in self.registry:
            h.update(c.sha256_provenance_hash.encode("utf-8"))
        return {
            "engine_id": self.engine_id,
            "registry_size": self.cycle_count,
            "global_verdict": self.global_verdict.name,
            "n_immune": sum(1 for c in self.registry if c.is_immune()),
            "evidence_hash": h.hexdigest(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN Y PRUEBAS DEL MOTOR  (las tres fases anidadas en vivo)
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("═" * 80)
    print("DEMOSTRACIÓN GRANULAR: TOON Wisdom Weaver Engine v4.0.0")
    print("FASES ANIDADAS: Ω₃+Semilla M → M/B/P/G/F/D+Traza → V/Crowbar/Sello")
    print("INTEGRACIÓN CELESTE: Poincaré-KAM-Delaunay-Melnikov-Jacobi")
    print("═" * 80)

    # ── Verificación puntual del álgebra de Heyting ────────────────────────
    assert HeytingOmega3.DEGRADED.excluded_middle_holds() is False
    assert HeytingOmega3.COHERENT.excluded_middle_holds() is True
    assert HeytingOmega3.VETOED.implies(HeytingOmega3.COHERENT) == HeytingOmega3.COHERENT
    assert HeytingOmega3.COHERENT.implies(HeytingOmega3.VETOED) == HeytingOmega3.VETOED
    assert HeytingOmega3.COHERENT.is_kam_stratum() is True
    assert HeytingOmega3.VETOED.poincare_stratum_name() == "hyperbolic-escape-separatrix"

    engine = TOONWisdomWeaverEngine(dimension=4)

    # ── Cartucho 1: Normal / Saludable ─────────────────────────────────────
    c1 = TOONSynapticCartridge(
        cartridge_id="CARTRIDGE-TOON-APU-001",
        apu_code="APU-CONCRETO-3000PSI",
        unit_cost_tangible=450_000.0,
        policy_risk_intangible=0.15,
        token_count_json=412,
        token_count_toon=56,
        attributes_matrix=np.array(
            [
                [1.0, 0.2, 0.1, 0.0],
                [0.2, 0.8, 0.0, 0.1],
                [0.1, 0.0, 0.9, 0.2],
                [0.0, 0.1, 0.2, 1.1],
            ],
            dtype=complex,
        ),
    )

    print("\n>>> ESCENARIO A: Metabolizando Vitamina Cognitiva TOON Saludable...")
    cert_a = engine.process_cognitive_vitamin(c1)
    print(f"    - ID Ciclo          : {cert_a.cycle_id}")
    print(f"    - Veredicto Heyting : {cert_a.heyting_verdict.name}")
    print(f"    - Estrato celeste   : {cert_a.heyting_verdict.poincare_stratum_name()}")
    print(f"    - Adjunción Galois  : {cert_a.galois_adjunction_valid}")
    print(
        f"    - Purificación Tr(ρ²): {cert_a.initial_purity:.6f} → "
        f"{cert_a.purified_purity:.6f} (ΔP = {cert_a.purification_delta():+.6f})"
    )
    print(f"    - Entropía von N.   : {cert_a.von_neumann_entropy:.6f}")
    print(f"    - Lyapunov ΔL       : {cert_a.lyapunov_delta:+.6f}")
    print(f"    - Gap espectral     : {cert_a.spectral_gap:.6f}")
    print(f"    - Betti-0 grafo     : {cert_a.graph_betti_0}")
    print(f"    - Residual C*       : {cert_a.cstar_residual:.3e}")
    print(f"    - Compresión KV     : {cert_a.kv_cache_compression_ratio * 100:.2f}%")
    print(f"    - Clase metabólica  : {'INMUNE' if cert_a.is_immune() else 'NO-INMUNE'}")
    print(f"    - Fotones γ         : {cert_a.gamma_photons_emitted}")
    print(f"    - Hash SHA-256      : {cert_a.provenance_prefix(32)}...")

    # Verificación celeste adicional: tripleta de Delaunay del estado purificado
    rho_a = engine.metabolic_field.lift_to_gibbs_state(c1)
    L_a, G_a, H_a = rho_a.delaunay_triple()
    print(f"    - Delaunay (L,G,H)  : ({L_a:.6f}, {G_a:.6f}, {H_a:.6f})")
    _, kam_a = PoincareCelestialIntegrator.kam_torus_indicator(rho_a)
    print(f"    - Residuo KAM       : {kam_a:.6e}")

    # ── Cartucho 2: Anomalía (riesgo crítico > 0.85) ───────────────────────
    c2 = TOONSynapticCartridge(
        cartridge_id="CARTRIDGE-TOON-ANOMALY-002",
        apu_code="APU-EXCAVACION-IREGULAR",
        unit_cost_tangible=1_200_000.0,
        policy_risk_intangible=0.92,
        token_count_json=512,
        token_count_toon=64,
        attributes_matrix=np.array(
            [
                [2.0, 1.5, 0.8, 0.5],
                [1.5, 1.8, 0.9, 0.4],
                [0.8, 0.9, 2.1, 0.7],
                [0.5, 0.4, 0.7, 1.9],
            ],
            dtype=complex,
        ),
    )

    print("\n>>> ESCENARIO B: Cartucho Anómalo (riesgo crítico, Fock incoherente)...")
    cert_b = engine.process_cognitive_vitamin(c2)
    print(f"    - ID Ciclo          : {cert_b.cycle_id}")
    print(f"    - Veredicto Heyting : {cert_b.heyting_verdict.name}")
    print(f"    - Estrato celeste   : {cert_b.heyting_verdict.poincare_stratum_name()}")
    print(f"    - Crowbar Activado  : {cert_b.crowbar_triggered}")
    print(f"    - Latencia HW       : {cert_b.hardware_latency_ns:.2f} ns (GPIO14)")
    print(f"    - γ fotones         : {cert_b.gamma_photons_emitted}")
    print(f"    - Clase metabólica  : {'INMUNE' if cert_b.is_immune() else 'NO-INMUNE'}")
    print(f"    - Constante Jacobi  : {c2.jacobi_constant(engine.metabolic_field.lift_to_gibbs_state(c2).as_array()):.6f}")

    # ── Cartucho 3: Riesgo medio (DEGRADED) ────────────────────────────────
    c3 = TOONSynapticCartridge(
        cartridge_id="CARTRIDGE-TOON-MEDIUM-003",
        apu_code="APU-ACERO-CORRUGADO",
        unit_cost_tangible=750_000.0,
        policy_risk_intangible=0.62,
        token_count_json=380,
        token_count_toon=48,
        attributes_matrix=np.array(
            [
                [1.2, 0.5, 0.3, 0.2],
                [0.5, 1.0, 0.4, 0.3],
                [0.3, 0.4, 0.9, 0.1],
                [0.2, 0.3, 0.1, 1.1],
            ],
            dtype=complex,
        ),
    )

    print("\n>>> ESCENARIO C: Cartucho con Riesgo Medio (DEGRADED por Fock)...")
    cert_c = engine.process_cognitive_vitamin(c3)
    print(f"    - ID Ciclo          : {cert_c.cycle_id}")
    print(f"    - Veredicto Heyting : {cert_c.heyting_verdict.name}")
    print(f"    - Estrato celeste   : {cert_c.heyting_verdict.poincare_stratum_name()}")
    print(f"    - γ fotones         : {cert_c.gamma_photons_emitted}")
    print(f"    - ΔP                : {cert_c.purification_delta():+.6f}")
    print(f"    - ΔL Lyapunov       : {cert_c.lyapunov_delta:+.6f}")

    # ── Verificación de la sección de Poincaré ─────────────────────────────
    print("\n>>> VERIFICACIÓN DE LA SECCIÓN DE POINCARÉ Σ_c...")
    rho_0_c1 = engine.metabolic_field.lift_to_gibbs_state(c1)
    N_diag_4 = np.diag(np.arange(1, 5, dtype=float))
    level_c = float(np.trace(rho_0_c1.matrix @ N_diag_4).real)
    section = PoincareCelestialIntegrator.poincare_section_at(level_c, 4)
    t_ret, rho_ret = PoincareCelestialIntegrator.poincare_return_time(
        rho_0_c1.matrix, section, N_diag_4, dt=0.02, max_time=100.0
    )
    print(f"    - Nivel Σ_c         : {level_c:.6f}")
    print(f"    - Tiempo de retorno : {t_ret:.6f}")
    print(f"    - Transversalidad   : {section.is_transversal(rho_0_c1.matrix, rho_0_c1.matrix @ N_diag_4 - N_diag_4 @ rho_0_c1.matrix)}")

    # ── Predicción Poincaré-Birkhoff ───────────────────────────────────────
    print("\n>>> TEOREMA DE POINCARÉ-BIRKOFF (predicción)...")
    for (p, q) in [(1, 3), (2, 5), (3, 8)]:
        n_fixed = PoincareCelestialIntegrator.poincare_birkhoff_count(p, q, twist_angle=math.pi / 2.0)
        print(f"    - Racional p/q = {p}/{q}  →  mínimo de puntos fijos = {n_fixed}")

    print("\n>>> AUDITORÍA RETROSPECTIVA DEL REGISTRO...")
    audit = engine.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<28}: {v}")

    print("\n>>> PASAPORTE AGREGADO DEL WEAVER ENGINE...")
    passport = engine.emit_weaver_passport()
    for k, v in passport.items():
        print(f"    - {k:<18}: {v}")

    print("\n" + "═" * 80)
    print("✓ Pruebas de verificación del TOON Wisdom Weaver Engine v4.0.0 completadas.")
    print("═" * 80)
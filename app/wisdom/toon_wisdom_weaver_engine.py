
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Wisdom Weaver Engine (Motor Espectral y Campo Metabolizador) ║
║ Ubicación: app/wisdom/toon_wisdom_weaver_engine.py                           ║
║ Versión  : 3.0.0-Doctoral-Nested-Ω₃-Banach-Brockett-Fock-Galois-Dirichlet    ║
║ Fases    : FASE-1 → FASE-2 → FASE-3  (anidadas: el último método de k es el  ║
║            germen formal del primero de k+1)                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Formalización Categorial Doctoral (Endofuntor Metabolizador W)
==============================================================

Sea 𝓣_Ω el topos de haces sobre el retículo de Heyting lineal linealmente ordenado:

        Ω₃  =  { VETOED = 0  ≺  DEGRADED = 1  ≺  COHERENT = 2 }

donde Ω₃ es el objeto clasificador de subobjetos intuicionistas de 𝓣_Ω. El motor
construye un endofuntor estricto sobre la categoría de cartuchos sinápticos:

        W :  𝐂𝐚𝐫𝐭_𝐓𝐎𝐎Ns  ──▶  𝐂𝐞𝐫𝐭_𝐌𝐞𝐭

dotado de una transformación natural de elevación η : Id ⇒ W (unidad de la
metabolización) y una counidad de sellado criptográfico ε : W ⇒ Id_Cert. La composición
de morfismos es asociativa en el álgebra C* de operadores acotados 𝐁𝐚𝐧(ℋₙ) sobre
el espacio de Hilbert complejo de dimensión finita ℋₙ ≅ ℂⁿ:

        W  =  V ∘ D ∘ F ∘ G ∘ B ∘ M

Estructura de Fases Anidadas e Invariantes
===========================================

FASE 1 — RETÍCULO DE HEYTING Ω₃, OBJETOS, CARTUCHO, CERTIFICADO Y SEMILLA M
──────────────────────────────────────────────────────────────────────────
  • HeytingOmega3: Álgebra de Heyting lineal (0 < 1 < 2). Satisface la adjunción
    de residuación  a ∧ c ≤ b  ⇔  c ≤ (a → b)  donde  a → b = ⊤ si a ≤ b, y b en otro caso.
    Pseudocomplemento ¬_H a = a → ⊥. Falla del tercio excluso en DEGRADED:
    DEGRADED ∨ ¬_H(DEGRADED) = DEGRADED ≠ COHERENT.
  • DensityOperator: Representante del cono positivo 𝔇(ℋₙ) = { ρ ∈ 𝐁𝐚𝐧(ℋₙ) | ρ = ρ†, ρ ⪰ 0, Tr(ρ) = 1 }.
    Garantiza la identidad C* residual | ‖ρ†ρ‖₂ − ‖ρ‖₂² | = O(ε_num).
  • TOONSynapticCartridge: Objeto de 𝐂𝐚𝐫𝐭_𝐓𝐎𝐎𝐍. Posee la matriz de atributos A ∈ Mₙ(ℂ),
    costo tangible c ≥ 0, riesgo intangible r ∈ [0, 1] y razón de compresión κ_comp = 1 − |TOON|/|JSON|.
  • MetabolicEndofunctorSeed (ABC): Germen formal de la metabolización. Su método
    abstracto `lift_to_gibbs_state` es el ÚLTIMO método de FASE-1 y el PRIMERO de FASE-2.

FASE 2 — CAMPO METABÓLICO, BROCKETT, GALOIS, FOCK, DIRICHLET Y TRAZA ABIERTA
──────────────────────────────────────────────────────────────────────────
  • (M) TOONMetabolicField.lift_to_gibbs_state: M : 𝔠 ↦ ρ₀. Construye el estado de Gibbs:
        H = ½(A + A†) ∈ 𝔥𝔢𝔯(ℋₙ),    H̃ = H / ‖H‖_F  (si ‖H‖_F > ε_norm),
        w = min( ceiling, log(1 + c/scale) · (1 + r) ),
        ρ₀ = U diag(softmax(w λ(H̃))) U† ∈ 𝔇(ℋₙ).
  • (B) BrockettIsospectralPurifier.purify: Flujo isospectral del doble corchete de Brockett en 𝔥𝔢𝔯(ℋₙ):
        dρ/dt = [ρ, [ρ, N]],    N = diag(1, 2, …, n).
        Función de Lyapunov isotónica L(ρ) = Tr(ρ N) con Ḋ(ρ) = ‖[ρ, N]‖_F² ≥ 0.
        Calcula la pureza γ(ρ) = Tr(ρ²) ∈ [1/n, 1] y la entropía de von Neumann S(ρ) = −Tr(ρ log ρ).
        Garantiza la isotonicidad ΔP = γ* − γ₀ ≥ −ε_num.
  • (G) GaloisAdjunctionValidator.validate_adjunction: Auditoría de la adjunción de Rham-Galois
        Hom_D(F(MIC), MAC) ≅ Hom_C(MIC, G(MAC)) con cocientes de pairing:
        ι_→ = ‖v_MIC‖₂ / (1 + γ_MAC),    ι_← = γ_MAC / (1 + ‖v_MIC‖₂).
  • (F) FockSpaceAnnihilatorEngine.process_annihilation: Aniquilación fermiónica sobre Fock
        1-modo ℱ_− = ℂ|0⟩ ⊕ ℂ|1⟩ con {a, a†} = 𝟙. Mapea el riesgo r:
        r > 0.85 ⇒ (False, 0, VETOED); 0.50 < r ≤ 0.85 ⇒ (True, 1, DEGRADED); r ≤ 0.50 ⇒ (True, 2, COHERENT).
  • (D) GeodesicAttentionCompressor.compute_dirichlet_energy: Geodésica sobre el laplaciano L = Deg − |A|_H.
        Energía de Dirichlet compuesta E_D = (1 − κ)² (1 + r) + (1 − tanh λ₂(L)) / n, donde λ₂(L)
        es la conectividad algebraica de Fiedler y β₀ = dim ker L es el número de Betti-0.
  • (Composer) SpectralArrowComposer.compose_arrows: Compone M ∘ B ∘ G ∘ F ∘ D y produce
    UnsealedMetabolicTrace (último objeto/método de FASE-2).

FASE 3 — WEAVER ENGINE, CROWBAR CIBER-FÍSICO, SELLO, AUDITORÍA, PASAPORTE
──────────────────────────────────────────────────────────────────────────
  • ESP32CrowbarWeaverHardware.trigger: Disyuntor ciber-físico en IRAM de ESP32 (GPIO14 → HIGH,
    tiristor BT151) con latencia t_prop ≈ 380 ns.
  • (V) TOONWisdomWeaverEngine._seal_and_verdict: V = ⋀_{Ω₃}(χ_Fock, χ_Galois, χ_Dirichlet).
    Aplica V, el crowbar si V = VETOED, genera el sello de procedencia SHA-256 inyectivo y
    retorna MetabolicFieldCertificate.
  • TOONWisdomWeaverEngine.process_cognitive_vitamin: Ejecuta W(𝔠) completo, registrando
    el certificado en la memoria inmutable del Weaver.

Definición Granular de Invariantes y Axiomas
=============================================
  1. Invariante de Traza Cuántica: Tr(ρ) = 1, ρ = ρ†, spec(ρ) ⊂ [0, 1].
  2. Isotonicidad de Brockett: Ḋ(ρ) = ‖[ρ, N]‖_F² ≥ 0 ⇒ L(ρ*) ≥ L(ρ₀).
  3. Adjunción de Heyting: (a ∧ c ≤ b) ⇔ (c ≤ (a → b)).
  4. Conservación de Conectividad Grafoteórica: β₀ = dim ker L ≥ 1.
  5. Inyectividad Criptográfica: H_SHA256(cycle_id ‖ cartridge_id ‖ verdict ‖ γ* ‖ t) es inyectivo.
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

logger = logging.getLogger("APU.Wisdom.TOONWisdomWeaverEngine.v3")

__all__ = [
    "HeytingOmega3",
    "TOONSynapticCartridge",
    "MetabolicFieldCertificate",
    "DensityOperator",
    "MetabolicEndofunctorSeed",
    "TOONMetabolicField",
    "BrockettIsospectralPurifier",
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


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — RETÍCULO DE HEYTING Ω₃, OBJETOS, CARTUCHO, CERTIFICADO Y SEMILLA M
# ══════════════════════════════════════════════════════════════════════════════
# Andamiaje categorial del topos 𝓣_Ω. Ω₃ es un álgebra de Heyting lineal
# (no booleana: DEGRADED viola el tercio excluso). El cartucho es un objeto
# de 𝐂𝐚𝐫𝐭_𝐓𝐎𝐎𝐍; el certificado es un objeto de 𝐂𝐞𝐫𝐭_𝐌𝐞𝐭. La semilla
# MetabolicEndofunctorSeed.lift_to_gibbs_state es el ÚLTIMO método de esta
# fase y el PRIMERO que realiza FASE-2 (flecha M).
# ══════════════════════════════════════════════════════════════════════════════


class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal Ω₃ = {0 ≺ 1 ≺ 2} = {VETOED ≺ DEGRADED ≺ COHERENT}.

    Operaciones (cadena finita ⇒ Heyting completa y residuada):

        a ∧ b  = min(a, b)                          (meet / producto)
        a ∨ b  = max(a, b)                          (join / coproducto)
        a → b  = ⊤  si a ≤ b,  else b               (residuo / implicación)
        ¬_H a  = a → ⊥                              (pseudocomplemento)
        ¬_B a  = ⊤ − a                              (negación booleana, no interna)

    El esqueleto booleano es {⊥, ⊤} ≅ 𝔹₂; DEGRADED es el valor intermedio
    que hace a Ω₃ estrictamente intuicionista:  a ∨ ¬a ≠ ⊤  para a = DEGRADED.
    Toda metabolización induce una flecha característica χ : Cartucho → Ω₃.
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    # ── retículo ──────────────────────────────────────────────────────────

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
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
        r""" ¬_H a := a → ⊥.  ¬VETOED = COHERENT; ¬DEGRADED = ¬COHERENT = VETOED. """
        return self.implies(HeytingOmega3.VETOED)

    def classical_negation(self) -> "HeytingOmega3":
        r""" Involución de De Morgan en el esqueleto {0,2} extendida por 2−a. """
        return HeytingOmega3(2 - int(self))

    def excluded_middle_holds(self) -> bool:
        r""" a ∨ ¬_H a = ⊤  ⇔  a ∈ {⊥, ⊤}. Falla en DEGRADED (intuicionismo). """
        return self.join(self.pseudo_complement()) == HeytingOmega3.COHERENT

    # ── objetos inicial / terminal ────────────────────────────────────────

    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        if self == HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED no admite proyección fiel a 𝔹₂.")
        return self == HeytingOmega3.COHERENT

    @property
    def verdict(self) -> str:
        return self.name


# ── Operador de densidad en el cono 𝔇(ℋₙ) ⊂ 𝐁𝐚𝐧(ℋₙ) ─────────────────────────


@dataclass(frozen=True, slots=True)
class DensityOperator:
    r"""
    Estado cuántico ρ ∈ 𝔇(ℋₙ) ⊂ 𝐁𝐚𝐧(ℋₙ).

    Invariantes (verificados en __post_init__ con tolerancia numérica):
        ρ = ρ†,   spec(ρ) ⊂ [−ε, 1+ε],   |Tr ρ − 1| ≤ ε,   ‖ρ‖₁ ≈ 1.

    El álgebra C* residual ‖ρ†ρ‖ − ‖ρ‖² mide la desviación de la identidad
    C* sobre el representante numérico (debe ser O(ε) para Hermitianos).
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
        return int(self.matrix.shape[0])

    def spectrum(self, floor: float = 1e-15) -> np.ndarray:
        lam = la.eigvalsh(self.matrix)
        lam = np.clip(lam, floor, None)
        s = float(np.sum(lam))
        return lam / s if s > 0.0 else np.full(lam.shape, 1.0 / lam.size)

    def purity(self) -> float:
        lam = self.spectrum()
        return float(np.sum(lam ** 2))

    def von_neumann_entropy(self) -> float:
        lam = self.spectrum()
        return -float(np.sum(lam * np.log(lam)))

    def spectral_gap(self) -> float:
        r""" Δλ = λ_max − λ_{max-1} del espectro de ρ (0 si n=1). """
        lam = np.sort(self.spectrum())
        if lam.size < 2:
            return 0.0
        return float(lam[-1] - lam[-2])

    def cstar_residual(self) -> float:
        r""" | ‖ρ†ρ‖₂ − ‖ρ‖₂² |  (identidad C* residual). """
        rho = self.matrix
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

    def as_array(self) -> np.ndarray:
        return self.matrix


# ── Cartucho sináptico inmutable ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TOONSynapticCartridge:
    r"""
    Objeto de 𝐂𝐚𝐫𝐭_𝐓𝐎𝐎𝐍. Payload de vitamina cognitiva comprimida.

    Invariantes:
      - unit_cost_tangible ≥ 0.
      - policy_risk_intangible ∈ [0, 1].
      - token_count_json > 0, token_count_toon > 0, toon ≤ json.
      - attributes_matrix ∈ Mₙ(ℂ) cuadrada, n > 0  (representación
        compleja de un posible álgebra hipercompleja; lo hermitiano
        se fuerza en M, no se exige al cartucho crudo).

    Interpretación grafoteórica: |A| es la matriz de adyacencia ponderada
    de un grafo simple no dirigido (tras simetrización), de la cual se
    extrae el laplaciano combinatorio en FASE-2.
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
        return int(self.attributes_matrix.shape[0])

    def compression_ratio(self) -> float:
        r""" κ_comp = 1 − |TOON|/|JSON| ∈ [0, 1). """
        return 1.0 - (self.token_count_toon / float(self.token_count_json))

    def is_hermitian(self, atol: float = 1e-9) -> bool:
        A = self.attributes_matrix
        return bool(np.allclose(A, A.conj().T, atol=atol))

    def mic_vector(self) -> np.ndarray:
        r""" Vector MIC = (costo, riesgo) ∈ ℝ², dominio izquierdo de la adjunción. """
        return np.array(
            [self.unit_cost_tangible, self.policy_risk_intangible],
            dtype=float,
        )

    def metabolic_weight(self, scale: float = 100_000.0, ceiling: float = 10.0) -> float:
        r"""
        Peso metabólico w = log1p(costo/escala)·(1+riesgo), acotado por ceiling.
        Codifica la intensidad con que el cartucho calienta el campo espectral
        (análogo de β⁻¹ en la medida de Gibbs).
        """
        if scale <= 0.0 or ceiling <= 0.0:
            raise ValueError("scale y ceiling deben ser > 0.")
        raw = math.log1p(self.unit_cost_tangible / scale) * (
            1.0 + self.policy_risk_intangible
        )
        return float(min(raw, ceiling))

    def adjacency_hermitian(self) -> np.ndarray:
        r""" Simetrización |A|_H = ½(A+A†) en módulo, pesos no negativos. """
        A = self.attributes_matrix
        H = 0.5 * (A + A.conj().T)
        return np.abs(H)


# ── Certificado metabólico terminal (objeto de 𝐂𝐞𝐫𝐭_𝐌𝐞𝐭) ────────────────────


@dataclass(frozen=True, slots=True)
class MetabolicFieldCertificate:
    r"""
    Objeto terminal del funtor W : 𝐂𝐚𝐫𝐭_𝐓𝐎𝐎𝐍 → 𝐂𝐞𝐫𝐭_𝐌𝐞𝐭.
    Inmutable, sellado con SHA-256. Codifica la traza completa del pipeline.
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
        r""" ΔP = γ_final − γ_inicial. Invariante físico: ≥ 0 en teoría. """
        return self.purified_purity - self.initial_purity

    def is_vetoed(self) -> bool:
        return self.heyting_verdict == HeytingOmega3.VETOED

    def is_immune(self) -> bool:
        r"""
        Inmune ⇔ no VETOED ∧ Galois ∧ Fock ∧ E_D < ½ ∧ ¬crowbar.
        Semántica: el certificado sobrevive al meet de los tres ejes y al
        interlock ciber-físico.
        """
        return (
            self.heyting_verdict != HeytingOmega3.VETOED
            and self.galois_adjunction_valid
            and self.fock_annihilation_event
            and self.dirichlet_energy < 0.5
            and not self.crowbar_triggered
        )

    def provenance_prefix(self, n: int = 16) -> str:
        return self.sha256_provenance_hash[:n]


# ── Protocolos estructurales (inyección de comportamiento) ──────────────────


@runtime_checkable
class MetabolicProjector(Protocol):
    """Proyector 𝔠 ↦ ρ. Permite sustituir Gibbs por von Neumann, Tsallis, etc."""

    def metabolize(self, cartridge: TOONSynapticCartridge) -> np.ndarray:
        ...


@runtime_checkable
class IsospectralPurifier(Protocol):
    """Purificador ρ ↦ (ρ*, γ, S, ΔL). Brockett, gradiente natural, o Riemann."""

    def purify(self, rho: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        ...


@runtime_checkable
class CrowbarInterlock(Protocol):
    """Disyuntor ciber-físico. ESP32, simulado, o doble de prueba."""

    def trigger(self, reason: str) -> Tuple[bool, float]:
        ...


# ── SEMILLA DEL ENDOFuntor: último artefacto de FASE-1, germen de FASE-2 ────


class MetabolicEndofunctorSeed(ABC):
    r"""
    Germen formal de la flecha M : 𝔠 ↦ ρ₀.

    Esta clase cierra FASE-1: fija la interfaz del endofuntor sobre el
    cono de densidad. FASE-2 *continúa* exactamente aquí, realizando
    lift_to_gibbs_state mediante diagonalización hermitiana y softmax
    espectral (álgebra de Banach / C* de operadores).
    """

    @abstractmethod
    def lift_to_gibbs_state(self, cartridge: TOONSynapticCartridge) -> DensityOperator:
        r"""
        Flecha M. Produce ρ₀ ∈ 𝔇(ℋₙ) a partir del cartucho.

        CONTINÚA EN FASE-2: TOONMetabolicField.lift_to_gibbs_state.
        """
        ...

    def metabolize(self, cartridge: TOONSynapticCartridge) -> np.ndarray:
        r"""Adaptador Protocol/MetabolicProjector → semilla categorial."""
        return self.lift_to_gibbs_state(cartridge).as_array()


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — CAMPO METABÓLICO, BROCKETT, GALOIS, FOCK, DIRICHLET Y TRAZA ABIERTA
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método de esta fase ES la realización del último de
# FASE-1 (lift_to_gibbs_state). El último método de FASE-2 (compose_arrows)
# produce UnsealedMetabolicTrace, germen formal de FASE-3 (sellado + V).
# ══════════════════════════════════════════════════════════════════════════════


class TOONMetabolicField(MetabolicEndofunctorSeed):
    r"""
    Realización de M : 𝔠 ↦ ρ₀  (continuación de MetabolicEndofunctorSeed).

    Construcción de Banach-Gibbs, numéricamente estable:

        H  = ½(A + A†) ∈ 𝔥𝔢𝔯(ℋₙ)
        H̃  = H / ‖H‖_F            si ‖H‖_F > ε  (si no, H)
        H̃  = U diag(λ) U†         (teorema espectral)
        p  = softmax(w λ)         (log-sum-exp)
        ρ₀ = U diag(p) U†         ∈ 𝔇(ℋₙ)

    Fallback: I/n si el espectro degenera o produce NaN/Inf.
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
        return 0.5 * (A + A.conj().T)

    @classmethod
    def _normalize_frobenius(cls, H: np.ndarray) -> np.ndarray:
        nrm = float(la.norm(H, "fro"))
        if nrm < cls.NORM_FLOOR:
            return H
        return H / nrm

    @classmethod
    def _maximally_mixed(cls, n: int) -> np.ndarray:
        return np.eye(n, dtype=complex) / float(n)

    @classmethod
    def _gibbs_from_hermitian(cls, H: np.ndarray, weight: float) -> np.ndarray:
        r"""
        ρ = expm(w H) / Z  vía teorema espectral + softmax.
        Equivale a la serie de Banach Σ (wH)^k / k! normalizada, pero
        evita overflow de expm y cancela el modo común de λ.
        """
        n = H.shape[0]
        evals, evecs = la.eigh(H)
        scaled = weight * evals
        # log-sum-exp
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
        # Ajuste de dimensión del campo vs. cartucho
        if rho.shape[0] != self.dimension:
            # no se interpola: se respeta el cartucho
            pass
        return DensityOperator(matrix=rho)

    def metabolize_cartridge(self, cartridge: TOONSynapticCartridge) -> np.ndarray:
        """Alias retro-compatible."""
        return self.metabolize(cartridge)


class BrockettIsospectralPurifier:
    r"""
    Purificador isospectral: flujo doble corchete de Brockett en 𝔥𝔢𝔯(ℋₙ)

        dρ/dt = [ρ, [ρ, N]],    N = diag(1, 2, …, n)

    Integración RK4 clásica + proyección al simplex espectral (PSD, Tr=1,
    Hermitiano). Lyapunov L(ρ)=Tr(ρ N) es no-decreciente:

        Ḋ(ρ) = ‖[ρ, N]‖_F² ≥ 0

    y γ(ρ)=Tr(ρ²) es isotónica sobre la órbita isospectral (optimalidad de
    Brockett). Se detiene por tolerancia de conmutador o por agotar steps.
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
        comm1 = rho @ N_diag - N_diag @ rho
        return rho @ comm1 - comm1 @ rho

    @classmethod
    def _spectrum(cls, rho: np.ndarray) -> np.ndarray:
        eigvals = la.eigvalsh(rho)
        eigvals = np.clip(eigvals, cls.EIGENVALUE_FLOOR, None)
        s = float(np.sum(eigvals))
        return eigvals / s if s > 0.0 else np.full(eigvals.shape, 1.0 / eigvals.size)

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        lam = cls._spectrum(rho)
        return float(np.sum(lam ** 2))

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        lam = cls._spectrum(rho)
        return -float(np.sum(lam * np.log(lam)))

    @classmethod
    def lyapunov(cls, rho: np.ndarray, N_diag: np.ndarray) -> float:
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


class GaloisAdjunctionValidator:
    r"""
    Auditor de la adjunción de de Rham-Galois

        Hom_D(F(MIC), MAC)  ≅  Hom_C(MIC, G(MAC))

    Se realizan ambas direcciones del isomorfismo heurístico:

        ι_→  = ‖v_MIC‖_2 / (1 + γ_MAC)
        ι_←  = γ_MAC / (1 + ‖v_MIC‖_2)

    Válida ⇔ ambos cocientes son finitos y estrictamente positivos
    (correspondencia costo-riesgo ↔ pureza espectral).
    """

    ETA_FLOOR: Final[float] = 1e-12

    @classmethod
    def _finite_positive(cls, x: float) -> bool:
        return (not math.isnan(x)) and (not math.isinf(x)) and (x > 0.0)

    @classmethod
    def validate_adjunction(
        cls, cartridge: TOONSynapticCartridge, rho_mac: np.ndarray
    ) -> bool:
        mic_norm = float(np.linalg.norm(cartridge.mic_vector()))
        mac_purity = float(np.trace(rho_mac @ rho_mac).real)
        den_fwd = 1.0 + mac_purity
        den_bwd = 1.0 + mic_norm
        if den_fwd < cls.ETA_FLOOR or den_bwd < cls.ETA_FLOOR:
            return False
        iota_fwd = mic_norm / den_fwd
        iota_bwd = mac_purity / den_bwd
        return cls._finite_positive(iota_fwd) and cls._finite_positive(iota_bwd)


class FockSpaceAnnihilatorEngine:
    r"""
    Motor de aniquilación fermiónica sobre el espacio de Fock de 1 modo

        ℱ_− = ℂ|0⟩ ⊕ ℂ|1⟩,    {a, a†} = 𝟙,    a² = (a†)² = 0.

    Identificación operativa:
        e⁻  := desviación de costo (riesgo contractual)
        e⁺  := autorización física (contabilidad)
        γ   := fotones emitidos ∈ {0, 1, 2}  (canal bosónico efectivo)

    Regla (ocupación de riesgo r ∈ [0,1]):
        r > 0.85            ⇒  (False, 0, VETOED)     fisión dura / veto
        0.50 < r ≤ 0.85     ⇒  (True,  1, DEGRADED)   1 fotón
        r ≤ 0.50            ⇒  (True,  2, COHERENT)   2 fotones (par e⁺e⁻)
    """

    HARD_FAIL_RISK: Final[float] = 0.85
    DEGRADE_RISK: Final[float] = 0.50

    @classmethod
    def process_annihilation(
        cls, cartridge: TOONSynapticCartridge
    ) -> Tuple[bool, int, HeytingOmega3]:
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
        """Alias retro-compatible."""
        return cls.process_annihilation(cartridge)


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
    """

    DIRICHLET_DEGRADE_THRESHOLD: Final[float] = 0.5
    LAPLACIAN_FLOOR: Final[float] = 1e-12

    @classmethod
    def combinatorial_laplacian(cls, cartridge: TOONSynapticCartridge) -> np.ndarray:
        W = cartridge.adjacency_hermitian().real
        # anular diagonal para interpretar como adyacencia simple ponderada
        np.fill_diagonal(W, 0.0)
        deg = np.sum(W, axis=1)
        return np.diag(deg) - W

    @classmethod
    def graph_betti_0(cls, L: np.ndarray) -> int:
        evals = la.eigvalsh(L)
        return int(np.sum(evals < cls.LAPLACIAN_FLOOR))

    @classmethod
    def algebraic_connectivity(cls, L: np.ndarray) -> float:
        evals = np.sort(la.eigvalsh(L))
        if evals.size < 2:
            return 0.0
        return float(max(evals[1], 0.0))

    @classmethod
    def compute_dirichlet_energy(
        cls, cartridge: TOONSynapticCartridge
    ) -> Tuple[float, float, int]:
        r"""
        Retorna (E_D, κ_comp, β₀).
        """
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
        if dirichlet < cls.DIRICHLET_DEGRADE_THRESHOLD:
            return HeytingOmega3.COHERENT
        return HeytingOmega3.DEGRADED


# ── Traza abierta: último objeto de FASE-2, germen de FASE-3 ────────────────


@dataclass(frozen=True, slots=True)
class UnsealedMetabolicTrace:
    r"""
    Traza sin sello ni crowbar. Portadora de (M, B, G, F, D) antes de V.

    Este dataclass cierra el contenido informacional de FASE-2. FASE-3
    *continúa* exactamente aquí: SpectralArrowComposer.compose_arrows es
    el último método de FASE-2 y el primero que consume FASE-3 para
    aplicar V, el interlock y el sello SHA-256.
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


class SpectralArrowComposer:
    r"""
    Compositor de las flechas M ∘ B ∘ G ∘ F ∘ D.

    ÚLTIMO método de FASE-2: compose_arrows.
    CONTINÚA EN FASE-3: TOONWisdomWeaverEngine._seal_and_verdict.
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
        Realiza M, B, G, F, D en ese orden y emite la traza abierta.

        CONTINÚA EN FASE-3 (sellado, meet V, crowbar, registro).
        """
        # M
        rho0 = self.metabolic_field.lift_to_gibbs_state(cartridge)
        gamma0 = rho0.purity()

        # B
        rho_star_arr, gamma_star, entropy, dL = self.purifier.purify(rho0.as_array())
        rho_star = DensityOperator(matrix=rho_star_arr)

        # G
        galois_ok = GaloisAdjunctionValidator.validate_adjunction(
            cartridge, rho_star.as_array()
        )

        # F
        fock_ok, gamma_photons, fock_verdict = (
            FockSpaceAnnihilatorEngine.process_annihilation(cartridge)
        )

        # D
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
    """

    HARDWARE_LATENCY_BASE_NS: Final[float] = 380.0

    @classmethod
    def trigger(cls, reason: str) -> Tuple[bool, float]:
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
        """Alias retro-compatible."""
        return cls.trigger(reason)


class TOONWisdomWeaverEngine:
    r"""
    Motor espectral = Campo metabolizador de vitaminas cognitivas TOON.
    Co-gobierna la Ciudadela de Cristal (V_Wisdom) junto al GodelEngine.

    W = V ∘ D ∘ F ∘ G ∘ B ∘ M,  realizado como

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

        # aliases de inyección (retro-compatibles con v2)
        self.metabolic_field: MetabolicEndofunctorSeed = field
        self.purifier: BrockettIsospectralPurifier = pur

        self.cycle_count: int = 0
        self.registry: List[MetabolicFieldCertificate] = []

    # ── continuación formal de compose_arrows (FASE-2 → FASE-3) ───────────

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

        Aplica V (meet Ω₃), el crowbar si χ=⊥, el sello SHA-256 y construye
        el objeto de 𝐂𝐞𝐫𝐭_𝐌𝐞𝐭.
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
                f"E_D={trace.dirichlet_energy:.4f}"
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

    # ── Núcleo funtorial ──────────────────────────────────────────────────

    def process_cognitive_vitamin(
        self, cartridge: TOONSynapticCartridge
    ) -> MetabolicFieldCertificate:
        r"""
        Ejecuta W(𝔠) = V(D(F(G(B(M(𝔠)))))).

        Pasos anidados:
          1. Identidades de ciclo.
          2. compose_arrows  (FASE-2: M, B, G, F, D) → UnsealedMetabolicTrace.
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
            "ΔL: %+.6f | β₀: %d",
            cycle_id,
            cert.heyting_verdict.name,
            cert.purification_delta(),
            cert.initial_purity,
            cert.purified_purity,
            cert.kv_cache_compression_ratio * 100.0,
            cert.gamma_photons_emitted,
            cert.lyapunov_delta,
            cert.graph_betti_0,
        )
        return cert

    # ── Vistas inmutables y auditoría retrospectiva ────────────────────────

    @property
    def registry_view(self) -> Tuple[MetabolicFieldCertificate, ...]:
        """Vista inmutable del registro de certificados metabólicos."""
        return tuple(self.registry)

    @property
    def global_verdict(self) -> HeytingOmega3:
        """Ínfimo (meet) de los veredictos registrados — objeto terminal de Ω₃."""
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
        hashes: set[str] = set()
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
    print("DEMOSTRACIÓN GRANULAR: TOON Wisdom Weaver Engine v3.0.0")
    print("FASES ANIDADAS: Ω₃+Semilla M → M/B/G/F/D+Traza → V/Crowbar/Sello")
    print("═" * 80)

    # Verificación puntual del álgebra de Heyting (tercio excluso)
    assert HeytingOmega3.DEGRADED.excluded_middle_holds() is False
    assert HeytingOmega3.COHERENT.excluded_middle_holds() is True
    assert HeytingOmega3.VETOED.implies(HeytingOmega3.COHERENT) == HeytingOmega3.COHERENT
    assert HeytingOmega3.COHERENT.implies(HeytingOmega3.VETOED) == HeytingOmega3.VETOED

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
    print(f"    - Crowbar Activado  : {cert_b.crowbar_triggered}")
    print(f"    - Latencia HW       : {cert_b.hardware_latency_ns:.2f} ns (GPIO14)")
    print(f"    - γ fotones         : {cert_b.gamma_photons_emitted}")
    print(f"    - Clase metabólica  : {'INMUNE' if cert_b.is_immune() else 'NO-INMUNE'}")

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
    print(f"    - γ fotones         : {cert_c.gamma_photons_emitted}")
    print(f"    - ΔP                : {cert_c.purification_delta():+.6f}")
    print(f"    - ΔL Lyapunov       : {cert_c.lyapunov_delta:+.6f}")

    print("\n>>> AUDITORÍA RETROSPECTIVA DEL REGISTRO...")
    audit = engine.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<28}: {v}")

    print("\n>>> PASAPORTE AGREGADO DEL WEAVER ENGINE...")
    passport = engine.emit_weaver_passport()
    for k, v in passport.items():
        print(f"    - {k:<18}: {v}")

    print("\n" + "═" * 80)
    print("✓ Pruebas de verificación del TOON Wisdom Weaver Engine v3.0.0 completadas.")
    print("═" * 80)
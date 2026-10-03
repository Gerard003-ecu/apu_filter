# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/agents/wisdom/toon_oniric_dreamer_agent.py                            ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / FASE REM (GAN-REM)                  ║
║ FUNCIÓN  : SOBERANO SIMULADOR ONÍRICO REM Y METABOLIZADOR POINCARANO                 ║
║ VERSIÓN  : 9.1.0-Doctoral-Poincare-Celeste-CRTBP-Lindstedt-Hill-Homoclinic-A4        ║
║ CONTRATO : 9.1.0 (Mecánica Celeste de Poincaré Vols. I–III + Topos + C* + GKSL)      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICA
───────────────────────────────────────────────
El `TOONOniricDreamerAgent` es la autoridad soberana responsable de orquestar la
generación de escenarios contrafactuales y la inyección de estrés controlado durante
la Fase REM del ecosistema agéntico APU Filter.

Inspirado en *Les Méthodes Nouvelles de la Mécanique Céleste* (Vols. I–III, 1892–1899)
de Henri Poincaré, este soberano simula escenarios de colapso navegando los tubos de
variedades invariantes de Lagrange (L₁ … L₅) en el Problema Restringido Circular de
Tres Cuerpos (CRTBP) y proyectando parámetros modulares sobre el dominio fundamental
de Poincaré ℱ ⊂ ℍ². Forma el vértice generador del bucle GAN-REM:

    [Ilusionista (Trickster)] → [Soñador (Dreamer)] → [Auditor Onírico] → [Testigo]

DICTUM POINCARANO (traducción operacional al enclave onírico)
────────────────────────────────────────────────────────────
Vol. I  — Soluciones periódicas y exponentes característicos.
          Toda órbita de Lyapunov alrededor de Lₖ induce un mapa de primer retorno
          P : Σ → Σ cuya linealización D_x P tiene espectro {λ, 1/λ, …} (simplecticidad).
Vol. II — Series de Lindstedt: se expande simultáneamente la solución y la frecuencia
          ω = ω₀ + ε ω₁ + ε² ω₂ + ⋯ para aniquilar términos seculares.
Vol. III — Invariantes integrales y recurrencia: si Φ_t preserva una medida finita μ,
          μ-casi todo x retorna a todo entorno de x infinitas veces.

Analogía Hill / enclave:
    cuenca primaria (C_J > C_{L₁})  ≅  ℋ_physical = im P_p
    cuenca secundaria               ≅  ℋ_dream    = im P_d
    cuello de Hill en L₁ abierto    ≅  leakage ‖[ρ, P_p]‖∞ > 0
    cuello cerrado (C_J ≥ C_{L₁})   ≅  aislamiento homológico estricto.

ESTRUCTURA CATEGÓRICO-ALGEBRAICA
────────────────────────────────
    (i)   Topos de verdad Ω₄ = {⊥ ≺ ∂ ≺ ♯ ≺ ⊤} con topología de Lawvere–Tierney j.
    (ii)  Álgebra de Clifford Cl⁺_{1,3}(ℝ) ≅ ℍ ⊗_ℝ ℂ ≅ M₂(ℂ).
    (iii) Complejo de cadenas simplicial K = (C₀, C₁, C₂) con Laplacianos de Hodge
          y dualidad de Poincaré débil |β₀ − β₂| (obstrucción de cerradura).
    (iv)  C*-álgebra B(ℋ) con 𝔇(ℋ) y métricas Umegaki / Bures–Wasserstein.
    (v)   Semiplano ℍ² con PSL(2, ℤ) y dominio fundamental ℱ.
    (vi)  CRTBP canónico (q, p) ∈ T*ℝ³ con integral de Jacobi C_J = 2Ω − ‖p‖²
          y tubos W^{s/u}(γ_{Lₖ}) de Koon–Lo–Marsden–Ross.

POSTULADOS DE LA GOBERNANZA AGÉNTICA POINCARANA
──────────────────────────────────────────────
1. Aislamiento homológico ⇔ cuello de Hill cerrado:
       C_J(ρ_dream) ≥ C_{L₁}  ⟹  ∂(ρ_dream) ≡ 0 mod RealWorld.
2. Uniformización fucsiana: γ·τ = (aτ+b)/(cτ+d), τ ∈ ℱ.
3. Inoculación afín Φ_η y vacuna espectral P_vac moduladas por Ω₄ y por el
   residuo homoclínico de la sección de Poincaré.

ORGANIZACIÓN EN TRES FASES ANIDADAS (BISAGRAS CANÓNICAS)
────────────────────────────────────────────────────────
FASE 1 → FASE 2  : `CategoricalCircuitCartridge.lift_enclave_hamiltonian`
                   (cartucho × CRTBP × F₂ de Poincaré  ↦  H ∈ 𝔥𝔢𝔯(ℋ)).
FASE 2 → FASE 3  : `MetabolizedPerturbationField.evolve_and_certify`
                   (H × ρ₀  ↦  ρ_dream metabolizada + recurrencia de Poincaré).
FASE 3           : vacuna espectral, Merkle SHA-512, wake-sleep, pasaporte.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

from app.wisdom.toon_oniric_dreamer_engine import (
    FuchsianDomainCertificate,
    NonHermitianLindbladMasterEngine,
    poincare_hyperbolic_distance,
    reduce_to_poincare_fundamental_domain,
)

logger = logging.getLogger("APU.Wisdom.TOONOniricDreamerAgent.v10")

# ── Tipos algebraicos ─────────────────────────────────────────────────────────
ComplexMatrix = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]

__version__: Final[str] = "10.0.0-Doctoral-Poincare-Celeste-CRTBP-Lindstedt-Hill-Homoclinic-A4"

__all__ = [
    "HeytingToposAlgebra",
    "BiquaternionClifford",
    "SimplicialHodgeGraph",
    "PoincareCRTBPPhaseSpace",
    "PoincareInvariantManifoldTube",
    "LindstedtPoincareCanonicalGenerator",
    "NonReciprocalTellegenNetwork",
    "OpenQuantumDynamicsSeed",
    "CategoricalCircuitCartridge",
    "CartridgeMorphism",
    "BanachSpectralAlgebra",
    "NonCommutativeNoSignalingEnclave",
    "MetabolizedPerturbationField",
    "MerkleTree",
    "MerkleInclusionProof",
    "OniricScenarioCertificate",
    "PoincareOniricScenarioCertificate",
    "PoincareRecurrenceInvariant",
    "MacroWakeSleepAuditReport",
    "SpectralImmuneVaccineEngine",
    "TOONOniricDreamerAgent",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — TOPOS Ω₄, CLIFFORD, HODGE, CRTBP DE POINCARÉ, TELLEGEN Y SEMILLA H
#
#   Germen terminal: CategoricalCircuitCartridge.lift_enclave_hamiltonian
#   (H ∈ 𝔥𝔢𝔯(ℋ) construido por la función generatriz F₂ de Poincaré sobre el
#    fibrado cotangente del CRTBP).  Ese H es el único input hamiltoniano de
#    la FASE 2.
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.1 — Álgebra de Heyting Ω₄ con topología de Lawvere–Tierney
# ─────────────────────────────────────────────────────────────────────────────
class HeytingToposAlgebra(IntEnum):
    r"""
    Cadena finita completa Ω₄ = {⊥ ≺ ∂ ≺ ♯ ≺ ⊤}, clasificador de subobjetos del
    topos de haces sobre el sitio finito de 4 sieves del ciclo REM.

    Como toda cadena finita con 0 y 1, Ω₄ es un álgebra de Heyting completa:

        a ∧ b  = min(a, b)                          (ínfimo / límite categorial),
        a ∨ b  = max(a, b)                          (supremo / colímite),
        a → b  = ⋁ { c ∈ Ω₄ : c ∧ a ≤ b }           (residuo de Heyting),
        ¬_H a  = a → ⊥                               (pseudocomplemento).

    Interpretación celestial de Poincaré (Vol. I, §§36–47: estabilidad lineal):

        ⊤  ≅  L₄/L₅ linealmente estables (μ < μ_Routh) — órbita acotada,
        ♯  ≅  L₂/L₃ silla con cuello de Hill estrecho — transporte lento,
        ∂  ≅  L₁ silla-gateway — frontera de las cuencas de Hill,
        ⊥  ≅  C_J < C_{L₁} — cuello abierto, escape hiperbólico (caos).

    Topología de Lawvere–Tierney canónica:
        j(⊥)=⊥, j(∂)=♯, j(♯)=♯, j(⊤)=⊤
    (cierra la frontera L₁ hacia un sieve submáximo sin tocar la cúspide).
    """

    VETOED_ABSURDUM: int = 0      # ⊥
    BOUNDARY_CRITICAL: int = 1    # ∂  (sieve de frontera / L₁)
    TOPOLOGICAL_STABLE: int = 2   # ♯  (sieve submáximo / L₂–L₃)
    VERUM_COHERENT: int = 3       # ⊤  (L₄/L₅ o recurrencia regular)

    @property
    def verdict(self) -> str:
        r"""Nombre simbólico del veredicto."""
        return self.name

    def meet(self, other: "HeytingToposAlgebra") -> "HeytingToposAlgebra":
        r"""Ínfimo categorial ∧ : límite del diagrama discreto {a, b}."""
        return HeytingToposAlgebra(min(int(self), int(other)))

    def join(self, other: "HeytingToposAlgebra") -> "HeytingToposAlgebra":
        r"""Supremo categorial ∨ : colímite del diagrama discreto {a, b}."""
        return HeytingToposAlgebra(max(int(self), int(other)))

    def implies(self, other: "HeytingToposAlgebra") -> "HeytingToposAlgebra":
        r"""Residuo de Heyting a → b = ⋁{ c ∈ Ω₄ : c ∧ a ≤ b }."""
        a, b = int(self), int(other)
        candidates = [c for c in range(4) if min(c, a) <= b]
        return HeytingToposAlgebra(max(candidates))

    def pseudo_complement(self) -> "HeytingToposAlgebra":
        r"""Pseudocomplemento intuicionista ¬_H a = a → ⊥."""
        return self.implies(HeytingToposAlgebra.VETOED_ABSURDUM)

    def classical_negation(self) -> "HeytingToposAlgebra":
        r"""Negación involutiva inducida por B₂ ⊂ Ω₄: a ↦ 3 − a."""
        return HeytingToposAlgebra(3 - int(self))

    def double_negation(self) -> "HeytingToposAlgebra":
        r"""Funtor ¬¬ : Ω₄ → Ω₄; imagen = B₂ = {⊥, ⊤}."""
        return self.pseudo_complement().pseudo_complement()

    def booleanize(self) -> "HeytingToposAlgebra":
        r"""Reflexión a B₂ vía ¬¬."""
        return self.double_negation()

    def is_regular(self) -> bool:
        r"""a regular ⟺ ¬¬a = a. En Ω₄ sólo {⊥, ⊤} son regulares."""
        return self.double_negation() == self

    def is_dense(self) -> bool:
        r"""a denso ⟺ ¬a = ⊥."""
        return self.pseudo_complement() == HeytingToposAlgebra.VETOED_ABSURDUM

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬a = ⊤  ⇔  a ∈ {⊥, ⊤}."""
        return self.join(self.pseudo_complement()) == HeytingToposAlgebra.VERUM_COHERENT

    def lawvere_tierney_closure(self) -> "HeytingToposAlgebra":
        r"""
        Operador de clausura j : Ω₄ → Ω₄ de la topología interna canónica:
            j(⊥) = ⊥,  j(∂) = ♯,  j(♯) = ♯,  j(⊤) = ⊤.
        Celestialmente: cierra el gateway L₁ hacia un sieve submáximo L₂/L₃.
        """
        val = int(self)
        if val == 0:
            return HeytingToposAlgebra.VETOED_ABSURDUM
        if val == 1:
            return HeytingToposAlgebra.TOPOLOGICAL_STABLE
        return self

    def is_j_closed(self) -> bool:
        r"""a es j-cerrado ⟺ j(a) = a."""
        return self.lawvere_tierney_closure() == self

    def is_j_dense(self) -> bool:
        r"""a es j-denso ⟺ j(a) = ⊤."""
        return self.lawvere_tierney_closure() == HeytingToposAlgebra.VERUM_COHERENT

    @classmethod
    def bottom(cls) -> "HeytingToposAlgebra":
        return cls.VETOED_ABSURDUM

    @classmethod
    def top(cls) -> "HeytingToposAlgebra":
        return cls.VERUM_COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingToposAlgebra":
        r"""Encaje ι : B₂ ↪ Ω₄; False ↦ ⊥, True ↦ ⊤."""
        return cls.VERUM_COHERENT if b else cls.VETOED_ABSURDUM

    def to_bool(self) -> bool:
        r"""Sección parcial de ι; no definida fuera de im(B₂ ↪ Ω₄)."""
        if self not in (
            HeytingToposAlgebra.VETOED_ABSURDUM,
            HeytingToposAlgebra.VERUM_COHERENT,
        ):
            raise ValueError(f"{self.name} ∉ im(B₂ ↪ Ω₄); usar booleanize().")
        return self == HeytingToposAlgebra.VERUM_COHERENT

    @classmethod
    def from_lagrange_stability(
        cls, lagrange_index: int, mu: float, hill_neck_closed: bool
    ) -> "HeytingToposAlgebra":
        r"""
        Valuación Ω₄ inducida por el índice de Lagrange y el criterio de Routh.

        Poincaré Vol. I: L₄, L₅ son linealmente estables ssi
            μ < μ_Routh = ½(1 − √(23/27)) ≈ 0.03852.
        Si el cuello de Hill está abierto, el veredicto colapsa a ⊥
        (existe transporte heteroclínico L₁ → escape).
        """
        if not hill_neck_closed:
            return cls.VETOED_ABSURDUM
        if lagrange_index in (4, 5):
            mu_routh = PoincareCRTBPPhaseSpace.MU_ROUTH
            return cls.VERUM_COHERENT if mu < mu_routh else cls.TOPOLOGICAL_STABLE
        if lagrange_index in (2, 3):
            return cls.TOPOLOGICAL_STABLE
        if lagrange_index == 1:
            return cls.BOUNDARY_CRITICAL
        return cls.VETOED_ABSURDUM

    @classmethod
    def verify_residuation_axiom(cls) -> bool:
        r"""
        Axioma definitorio de Heyting: ∀ a, b, c ∈ Ω₄,
            c ∧ a ≤ b  ⟺  c ≤ (a → b).
        """
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

    @classmethod
    def verify_lawvere_tierney_topology_axioms(cls) -> bool:
        r"""Verifica los axiomas (LT1)–(LT4) de j como topología de Lawvere–Tierney."""
        elements = list(cls)
        if cls.VERUM_COHERENT.lawvere_tierney_closure() != cls.VERUM_COHERENT:
            return False
        for a in elements:
            ja = a.lawvere_tierney_closure()
            if int(a) > int(ja):
                return False
            if ja.lawvere_tierney_closure() != ja:
                return False
            for b in elements:
                lhs = a.meet(b).lawvere_tierney_closure()
                rhs = ja.meet(b.lawvere_tierney_closure())
                if lhs != rhs:
                    return False
                lhs_join = a.join(b).lawvere_tierney_closure()
                rhs_join = ja.join(b.lawvere_tierney_closure())
                if int(lhs_join) < int(rhs_join):
                    return False
        return True

    def __and__(self, other: "HeytingToposAlgebra") -> "HeytingToposAlgebra":
        return self.meet(other)

    def __or__(self, other: "HeytingToposAlgebra") -> "HeytingToposAlgebra":
        return self.join(other)

    def __invert__(self) -> "HeytingToposAlgebra":
        return self.pseudo_complement()

    def __le__(self, other: "HeytingToposAlgebra") -> bool:  # type: ignore[override]
        return int(self) <= int(other)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.2 — Bicuaterniones: Cl⁺_{1,3}(ℝ) ≅ ℍ ⊗_ℝ ℂ ≅ M₂(ℂ)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BiquaternionClifford:
    r"""
    Elemento del álgebra de bicuaterniones 𝔹 ≅ ℍ ⊗_ℝ ℂ ≅ Cl⁺_{1,3}(ℝ) ≅ M₂(ℂ),
    subálgebra par del álgebra de Clifford del espaciotiempo de Minkowski
    (+, −, −, −).

    Realización matricial fiel:
        q = w + x i + y j + z k    con w, x, y, z ∈ ℂ
        φ(q) = [[w + iz , y + ix], [−y + ix , w − iz]] ∈ M₂(ℂ).

    Recubrimiento 2:1  SL(2, ℂ) → SO⁺(1, 3)  (Lorentz propias ortócronas),
    que es el grupo de simetrías del fibrado cotangente del CRTBP embebido
    en el espacio de Minkowski auxiliar de Poincaré (Vol. I, cap. I).
    """

    w_re: float
    w_im: float
    x_re: float
    x_im: float
    y_re: float
    y_im: float
    z_re: float
    z_im: float

    _NORM_FLOOR: Final[float] = 1e-15

    def _as_complex_tuple(self) -> Tuple[complex, complex, complex, complex]:
        r"""Componentes (w, x, y, z) ∈ ℂ⁴."""
        return (
            complex(self.w_re, self.w_im),
            complex(self.x_re, self.x_im),
            complex(self.y_re, self.y_im),
            complex(self.z_re, self.z_im),
        )

    @classmethod
    def from_complex(
        cls, w: complex, x: complex, y: complex, z: complex
    ) -> "BiquaternionClifford":
        r"""Constructor desde componentes complejas (w, x, y, z) ∈ ℂ⁴."""
        return cls(
            w.real, w.imag,
            x.real, x.imag,
            y.real, y.imag,
            z.real, z.imag,
        )

    @classmethod
    def identity(cls) -> "BiquaternionClifford":
        r"""Unidad multiplicativa 1 + 0i + 0j + 0k."""
        return cls(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def zero(cls) -> "BiquaternionClifford":
        r"""Elemento neutro aditivo."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def to_matrix(self) -> ComplexMatrix:
        r"""Homomorfismo inyectivo de ℂ-álgebras 𝔹 ↪ M₂(ℂ)."""
        w, x, y, z = self._as_complex_tuple()
        return np.array(
            [
                [w + 1j * z, y + 1j * x],
                [-y + 1j * x, w - 1j * z],
            ],
            dtype=np.complex128,
        )

    def reduced_norm_complex(self) -> complex:
        r"""Norma reducida N(q) = det φ(q) = w² + x² + y² + z² ∈ ℂ."""
        return complex(np.linalg.det(self.to_matrix()))

    def frobenius_operator_norm(self) -> float:
        r"""Norma de Frobenius ‖φ(q)‖_F = √Tr(φ(q)† φ(q))."""
        M = self.to_matrix()
        return float(np.sqrt(np.real(np.trace(M.conj().T @ M))))

    def spectral_norm(self) -> float:
        r"""Norma espectral ‖φ(q)‖∞ = σ_max(φ(q))."""
        M = self.to_matrix()
        return float(np.max(la.svdvals(M))) if M.size else 0.0

    def hermitian_part(self) -> ComplexMatrix:
        r"""½(φ(q) + φ(q)†): componente hermítica (traslaciones en ℝ^{1,3})."""
        M = self.to_matrix()
        return 0.5 * (M + M.conj().T)

    def anti_hermitian_part(self) -> ComplexMatrix:
        r"""½(φ(q) − φ(q)†): componente antihermítica (rotaciones de Lorentz)."""
        M = self.to_matrix()
        return 0.5 * (M - M.conj().T)

    def conjugate(self) -> "BiquaternionClifford":
        r"""Conjugación cuaterniónica tensorial: q* = w − xi − yj − zk."""
        return BiquaternionClifford(
            self.w_re, self.w_im,
            -self.x_re, -self.x_im,
            -self.y_re, -self.y_im,
            -self.z_re, -self.z_im,
        )

    def complex_conjugate(self) -> "BiquaternionClifford":
        r"""Conjugación de la estructura ℂ: (w,x,y,z) ↦ (w̄,x̄,ȳ,z̄)."""
        return BiquaternionClifford(
            self.w_re, -self.w_im,
            self.x_re, -self.x_im,
            self.y_re, -self.y_im,
            self.z_re, -self.z_im,
        )

    def poincare_adjoint(self) -> "BiquaternionClifford":
        r"""
        Adjunta de Poincaré (análogo CPT): q ↦ (q*)̄.
        Intercambia la hoja futura/pasada del cono de luz y es involutiva.
        """
        return self.conjugate().complex_conjugate()

    def hermitian_conjugate_matrix(self) -> ComplexMatrix:
        r"""φ(q)† = φ(q*) como operador."""
        return self.conjugate().to_matrix()

    def inverse(self) -> "BiquaternionClifford":
        r"""Inversa multiplicativa q⁻¹ = q* / N(q); error si N(q) ≈ 0."""
        n = self.reduced_norm_complex()
        if abs(n) < self._NORM_FLOOR:
            raise ZeroDivisionError(f"Bicuaternión singular: N(q) = {n}.")
        qc = self.conjugate()
        w, x, y, z = qc._as_complex_tuple()
        return BiquaternionClifford.from_complex(w / n, x / n, y / n, z / n)

    def __add__(self, other: "BiquaternionClifford") -> "BiquaternionClifford":
        return BiquaternionClifford(
            self.w_re + other.w_re, self.w_im + other.w_im,
            self.x_re + other.x_re, self.x_im + other.x_im,
            self.y_re + other.y_re, self.y_im + other.y_im,
            self.z_re + other.z_re, self.z_im + other.z_im,
        )

    def __sub__(self, other: "BiquaternionClifford") -> "BiquaternionClifford":
        return BiquaternionClifford(
            self.w_re - other.w_re, self.w_im - other.w_im,
            self.x_re - other.x_re, self.x_im - other.x_im,
            self.y_re - other.y_re, self.y_im - other.y_im,
            self.z_re - other.z_re, self.z_im - other.z_im,
        )

    def __neg__(self) -> "BiquaternionClifford":
        return BiquaternionClifford(
            -self.w_re, -self.w_im, -self.x_re, -self.x_im,
            -self.y_re, -self.y_im, -self.z_re, -self.z_im,
        )

    def __mul__(self, other: object) -> "BiquaternionClifford":
        r"""Producto de bicuaterniones (ℂ-lineal, asociativo, no conmutativo)."""
        if isinstance(other, BiquaternionClifford):
            w1, x1, y1, z1 = self._as_complex_tuple()
            w2, x2, y2, z2 = other._as_complex_tuple()
            return BiquaternionClifford.from_complex(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        if isinstance(other, (int, float, complex)):
            s = complex(other)
            w, x, y, z = self._as_complex_tuple()
            return BiquaternionClifford.from_complex(s * w, s * x, s * y, s * z)
        return NotImplemented

    def __rmul__(self, other: object) -> "BiquaternionClifford":
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def reduced_norm_multiplicativity_residual(
        self, other: "BiquaternionClifford"
    ) -> float:
        r"""|N(q₁ q₂) − N(q₁) N(q₂)| (nulo: 𝔹 es álgebra de composición)."""
        n_prod = (self * other).reduced_norm_complex()
        n_sep = self.reduced_norm_complex() * other.reduced_norm_complex()
        return abs(n_prod - n_sep)

    def associativity_residual(
        self, other: "BiquaternionClifford", third: "BiquaternionClifford"
    ) -> float:
        r"""‖(q₁ q₂) q₃ − q₁ (q₂ q₃)‖_F (nulo en aritmética exacta)."""
        lhs = ((self * other) * third).to_matrix()
        rhs = (self * (other * third)).to_matrix()
        return float(la.norm(lhs - rhs, "fro"))

    def cstar_residual(self) -> float:
        r"""|‖φ(q)†φ(q)‖∞ − ‖φ(q)‖∞²|  (C*-identidad universal)."""
        M = self.to_matrix()
        op = float(la.norm(M.conj().T @ M, 2))
        nrm = float(la.norm(M, 2))
        return abs(op - nrm * nrm)

    def hopf_coordinates(self) -> Tuple[complex, complex]:
        r"""Coordenadas de Hopf complejas (η₁, η₂) ∈ ℂ² sobre S³_ℂ."""
        w, x, y, z = self._as_complex_tuple()
        return w + 1j * z, y + 1j * x

    def matrix_exp(self) -> ComplexMatrix:
        r"""exp(φ(q)) vía descomposición espectral de scipy.linalg.expm."""
        return la.expm(self.to_matrix())

    def matrix_log(self) -> ComplexMatrix:
        r"""log(φ(q)) vía eigendescomposición matricial (rama principal)."""
        M = self.to_matrix()
        eigvals, eigvecs = la.eig(M)
        log_eigvals = np.log(eigvals + 1e-30)
        return eigvecs @ np.diag(log_eigvals) @ la.inv(eigvecs)

    def spin_lorentz_residual(self) -> float:
        r"""
        Residuo del recubrimiento 2:1: |det φ(q) − N(q)|.
        Nulo ssi la realización matricial es un homomorfismo de álgebras.
        """
        return abs(complex(np.linalg.det(self.to_matrix())) - self.reduced_norm_complex())


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.3 — Complejo de Hodge simplicial con índice McKean–Singer
# ─────────────────────────────────────────────────────────────────────────────
class SimplicialHodgeGraph:
    r"""
    Complejo de cadenas simplicial finito K = (C₀, C₁, C₂) de dimensión 2.

        C₂ --∂₂--> C₁ --∂₁--> C₀,    ∂₁ ∂₂ = 0.

    Laplacianos de Hodge (Eckmann):
        L₀ = ∂₁ ∂₁ᵀ,   L₁ = ∂₁ᵀ ∂₁ + ∂₂ ∂₂ᵀ,   L₂ = ∂₂ᵀ ∂₂.

    Dualidad de Poincaré débil (obstrucción de variedad cerrada):
        δ_PD = |β₀ − β₂|.  δ_PD = 0 ssi K se comporta numéricamente como
        una 2-variedad cerrada orientable (Poincaré, Analysis Situs, 1895).

    Índice McKean–Singer: Tr(e^{−tL₀}) − Tr(e^{−tL₁}) + Tr(e^{−tL₂}) = χ(K).
    """

    def __init__(
        self,
        num_vertices: int,
        edges: List[Tuple[int, int]],
        faces: List[Tuple[int, int, int]],
    ) -> None:
        if num_vertices < 1:
            raise ValueError("num_vertices debe ser ≥ 1.")
        self.num_vertices = int(num_vertices)
        self.edges = [tuple(sorted((u, v))) for u, v in edges]
        self.faces = [tuple(sorted((u, v, w))) for u, v, w in faces]
        self._validate_topology()
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

    def _validate_topology(self) -> None:
        for (u, v) in self.edges:
            if not (0 <= u < self.num_vertices and 0 <= v < self.num_vertices):
                raise ValueError(f"Arista {(u, v)} fuera de [0, {self.num_vertices}).")
            if u == v:
                raise ValueError(f"Lazo {(u, v)} no es un 1-símplice.")
        edge_set = set(self.edges)
        for (u, v, w) in self.faces:
            if len({u, v, w}) != 3:
                raise ValueError(f"Cara degenerada {(u, v, w)}.")
            for e in ((u, v), (v, w), (u, w)):
                if tuple(sorted(e)) not in edge_set:
                    raise ValueError(f"Cara {(u, v, w)} refiere arista inexistente {e}.")

    def _validate_chain_complex_exactness(self, tol: float = 1e-9) -> None:
        r"""Propiedad ∂₁∂₂ = 0."""
        if self.boundary_2.size == 0:
            return
        residual_norm = float(la.norm(self.boundary_1 @ self.boundary_2))
        if residual_norm > tol:
            raise ValueError(f"Violación de ∂₁∂₂ = 0: ‖·‖ = {residual_norm:.3e}")

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

    def connected_components(self) -> List[Set[int]]:
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
            stack, comp = [start], set()
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                stack.extend(adjacency[node] - comp)
            visited |= comp
            components.append(comp)
        return components

    def euler_characteristic(self) -> int:
        r"""χ(K) = |V| − |E| + |F|."""
        return self.num_vertices - len(self.edges) + len(self.faces)

    def compute_betti_numbers(self, tol: float = 1e-9) -> Tuple[int, int, int]:
        r"""βₖ = dim ker Lₖ por conteo espectral de autovalores nulos."""
        eigvals_0 = la.eigvalsh(self.laplacian_0)
        betti_0 = int(np.sum(np.abs(eigvals_0) < tol))
        eigvals_1 = la.eigvalsh(self.laplacian_1)
        betti_1 = int(np.sum(np.abs(eigvals_1) < tol))
        betti_2 = 0
        if self.laplacian_2.size > 0:
            eigvals_2 = la.eigvalsh(self.laplacian_2)
            betti_2 = int(np.sum(np.abs(eigvals_2) < tol))
        n_comp = len(self.connected_components())
        if n_comp != betti_0:
            logger.warning(
                "Discrepancia β₀ espectral (%d) vs. combinatoria DFS (%d).",
                betti_0, n_comp,
            )
        return betti_0, betti_1, betti_2

    def poincare_duality_residual(self, tol: float = 1e-9) -> int:
        r"""
        Obstrucción de dualidad de Poincaré en dimensión 2: |β₀ − β₂|.
        Cero ssi el 2-esqueleto se comporta como variedad cerrada orientable.
        """
        b0, _, b2 = self.compute_betti_numbers(tol)
        return abs(b0 - b2)

    def verify_betti_euler_consistency(self, tol: float = 1e-9) -> bool:
        r"""Consistencia Euler–Poincaré: β₀ − β₁ + β₂ = χ(K)."""
        b0, b1, b2 = self.compute_betti_numbers(tol)
        return (b0 - b1 + b2) == self.euler_characteristic()

    def algebraic_connectivity(self, tol: float = 1e-12) -> float:
        r"""Segundo autovalor de L₀ (valor de Fiedler)."""
        evals = np.sort(la.eigvalsh(self.laplacian_0))
        if evals.size < 2:
            return 0.0
        return float(max(evals[1], 0.0)) if evals[1] > tol else 0.0

    def compute_spectral_gap(self, tol: float = 1e-9) -> float:
        r"""Menor autovalor positivo de L₁ (gap espectral Hodge en aristas)."""
        eigvals_1 = np.sort(la.eigvalsh(self.laplacian_1))
        non_zero = eigvals_1[eigvals_1 > tol]
        return float(non_zero[0]) if len(non_zero) > 0 else 0.0

    def harmonic_1_forms(self, tol: float = 1e-9) -> RealMatrix:
        r"""Base ortonormal de ker L₁ ⊂ C₁ ≅ ℝ^{|E|}."""
        w, v = la.eigh(self.laplacian_1)
        mask = np.abs(w) < tol
        if not np.any(mask):
            return np.zeros((len(self.edges), 0), dtype=np.float64)
        return v[:, mask]

    def harmonic_2_forms(self, tol: float = 1e-9) -> RealMatrix:
        r"""Base ortonormal de ker L₂ ⊂ C₂ ≅ ℝ^{|F|}."""
        if self.laplacian_2.size == 0:
            return np.zeros((len(self.faces), 0), dtype=np.float64)
        w, v = la.eigh(self.laplacian_2)
        mask = np.abs(w) < tol
        if not np.any(mask):
            return np.zeros((len(self.faces), 0), dtype=np.float64)
        return v[:, mask]

    def harmonic_projection_operator(self) -> RealMatrix:
        r"""Proyector ortogonal P_H : C₁ → ker L₁ ⊂ C₁."""
        P = self.harmonic_1_forms()
        if P.shape[1] == 0:
            return np.zeros((len(self.edges), len(self.edges)), dtype=np.float64)
        return P @ P.T

    def hodge_decompose_1_form(
        self, omega: RealVector, rcond: float = 1e-10
    ) -> Tuple[RealVector, RealVector, RealVector]:
        r"""
        Descomposición canónica de Hodge en 1-cadenas:
            ω = ∂₂ α + ∂₁ᵀ β + γ,    γ ∈ ker L₁.
        """
        omega = np.asarray(omega, dtype=np.float64).reshape(-1)
        if omega.size != len(self.edges):
            raise ValueError("ω debe vivir en C₁ (dim = |E|).")
        B2 = self.boundary_2
        B1 = self.boundary_1
        if B2.size > 0 and B2.shape[1] > 0:
            exact = B2 @ la.lstsq(B2, omega, cond=rcond)[0]
        else:
            exact = np.zeros_like(omega)
        remainder = omega - exact
        coexact = B1.T @ la.lstsq(B1.T, remainder, cond=rcond)[0]
        harmonic = remainder - coexact
        return exact, coexact, harmonic

    def heat_kernel_trace(self, t: float = 1.0) -> float:
        r"""χ_t(K) = Tr(e^{−tL₀}) − Tr(e^{−tL₁}) + Tr(e^{−tL₂}) = χ(K) ∀ t > 0."""
        tr0 = float(np.sum(np.exp(-t * la.eigvalsh(self.laplacian_0))))
        tr1 = float(np.sum(np.exp(-t * la.eigvalsh(self.laplacian_1))))
        tr2 = (
            float(np.sum(np.exp(-t * la.eigvalsh(self.laplacian_2))))
            if self.laplacian_2.size
            else 0.0
        )
        return tr0 - tr1 + tr2

    def verify_mckean_singer_index(self, t: float = 1.0, tol: float = 1e-6) -> bool:
        r"""Verifica la identidad χ_t(K) = χ(K) del índice de McKean–Singer."""
        return abs(self.heat_kernel_trace(t) - self.euler_characteristic()) < tol


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.4 — CRTBP de Poincaré: Jacobi, Lagrange, Hill, exponentes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareCRTBPPhaseSpace:
    r"""
    Espacio de fases del Problema Restringido Circular de Tres Cuerpos (CRTBP)
    en el marco sinódico, según Poincaré, *Méthodes Nouvelles*, Vol. I, cap. III
    y Szebehely, *Theory of Orbits* (1967).

    Hamiltoniano sinódico (unidades canónicas n = 1, G(m₁+m₂) = 1):

        H(q, p) = ½‖p‖² − Ω(q),
        Ω(x,y,z) = ½(x²+y²) + (1−μ)/r₁ + μ/r₂ + ½μ(1−μ),

    con primarios en (−μ, 0, 0) y (1−μ, 0, 0).  La integral de Jacobi

        C_J = −2H = 2Ω − ‖ẋ‖²

    folia T*ℝ³ en hipersuperficies de energía; las curvas de velocidad nula
    {2Ω = C_J} acotan las regiones de Hill.

    Criterio de aislamiento onírico (cuello de Hill en L₁):
        C_J ≥ C_{L₁}  ⟺  las cuencas primaria/secundaria son disjuntas
                       ⟺  no hay transporte heteroclínico dream ↔ physical.
    """

    mu: float = 0.0121505856  # Earth–Moon canónico (DE440)

    MU_ROUTH: ClassVar[float] = 0.5 * (1.0 - math.sqrt(23.0 / 27.0))
    MU_EARTH_MOON: ClassVar[float] = 0.0121505856
    MU_SUN_EARTH: ClassVar[float] = 3.0034806e-6
    NEWTON_ITERS: ClassVar[int] = 48
    NEWTON_TOL: ClassVar[float] = 1e-14

    def __post_init__(self) -> None:
        if not (0.0 < float(self.mu) <= 0.5):
            raise ValueError("μ del CRTBP debe vivir en (0, 1/2].")

    # -- Geometría de primarios ────────────────────────────────────────────
    def primary_positions(self) -> Tuple[RealVector, RealVector]:
        r"""Posiciones sinódicas (m₁, m₂) ∈ ℝ³ × ℝ³."""
        mu = float(self.mu)
        m1 = np.array([-mu, 0.0, 0.0], dtype=np.float64)
        m2 = np.array([1.0 - mu, 0.0, 0.0], dtype=np.float64)
        return m1, m2

    def _radii(self, q: RealVector) -> Tuple[float, float]:
        m1, m2 = self.primary_positions()
        r1 = float(np.linalg.norm(np.asarray(q, dtype=np.float64) - m1))
        r2 = float(np.linalg.norm(np.asarray(q, dtype=np.float64) - m2))
        return max(r1, 1e-16), max(r2, 1e-16)

    # -- Potencial efectivo y Jacobi ───────────────────────────────────────
    def effective_potential(self, q: RealVector) -> float:
        r"""Ω(q) = ½(x²+y²) + (1−μ)/r₁ + μ/r₂ + ½μ(1−μ)."""
        q = np.asarray(q, dtype=np.float64).reshape(3)
        mu = float(self.mu)
        r1, r2 = self._radii(q)
        return float(
            0.5 * (q[0] ** 2 + q[1] ** 2)
            + (1.0 - mu) / r1
            + mu / r2
            + 0.5 * mu * (1.0 - mu)
        )

    def jacobi_constant(self, q: RealVector, qdot: RealVector) -> float:
        r"""C_J(q, q̇) = 2Ω(q) − ‖q̇‖²  (integral primera del CRTBP)."""
        speed2 = float(np.sum(np.asarray(qdot, dtype=np.float64) ** 2))
        return 2.0 * self.effective_potential(q) - speed2

    def hill_region_contains(self, q: RealVector, jacobi_C: float) -> bool:
        r"""q ∈ región de Hill de nivel C  ⟺  2Ω(q) ≥ C."""
        return 2.0 * self.effective_potential(q) + 1e-15 >= float(jacobi_C)

    # -- Puntos de Lagrange ────────────────────────────────────────────────
    def _collinear_force(self, x: float) -> Tuple[float, float]:
        r"""(∂Ω/∂x, ∂²Ω/∂x²) sobre el eje sinódico y = z = 0."""
        mu = float(self.mu)
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        r1 = max(r1, 1e-16)
        r2 = max(r2, 1e-16)
        s1 = 1.0 if (x + mu) >= 0.0 else -1.0
        s2 = 1.0 if (x - 1.0 + mu) >= 0.0 else -1.0
        # (x-a)/|x-a|³ = s / r²
        f = x - (1.0 - mu) * s1 / (r1 ** 2) - mu * s2 / (r2 ** 2)
        df = (
            1.0
            + 2.0 * (1.0 - mu) / (r1 ** 3)
            + 2.0 * mu / (r2 ** 3)
        )
        return float(f), float(df)

    def _collinear_root(self, kind: int) -> float:
        r"""Newton sobre el eje colineal. kind ∈ {1,2,3} → L₁, L₂, L₃."""
        mu = float(self.mu)
        cube = (mu / 3.0) ** (1.0 / 3.0)
        if kind == 1:
            x = 1.0 - mu - cube
        elif kind == 2:
            x = 1.0 - mu + cube
        else:
            x = -1.0 - 5.0 * mu / 12.0
        for _ in range(self.NEWTON_ITERS):
            f, df = self._collinear_force(x)
            if abs(df) < 1e-18:
                break
            x_new = x - f / df
            if abs(x_new - x) < self.NEWTON_TOL:
                x = x_new
                break
            x = x_new
        return float(x)

    def lagrange_point(self, index: int) -> RealVector:
        r"""
        Coordenadas sinódicas de L_index, index ∈ {1,2,3,4,5}.
        L₄, L₅ son equiláteros exactos; L₁–L₃ se resuelven por Newton.
        """
        if index not in (1, 2, 3, 4, 5):
            raise ValueError("Índice de Lagrange ∈ {1,2,3,4,5}.")
        mu = float(self.mu)
        if index == 4:
            return np.array([0.5 - mu, math.sqrt(3.0) / 2.0, 0.0], dtype=np.float64)
        if index == 5:
            return np.array([0.5 - mu, -math.sqrt(3.0) / 2.0, 0.0], dtype=np.float64)
        x = self._collinear_root(index)
        return np.array([x, 0.0, 0.0], dtype=np.float64)

    def jacobi_at_lagrange(self, index: int) -> float:
        r"""C_{Lₖ} = 2Ω(Lₖ)  (velocidad nula en el equilibrio)."""
        return 2.0 * self.effective_potential(self.lagrange_point(index))

    def hill_neck_is_closed(self, jacobi_C: float) -> bool:
        r"""
        Cuello de Hill en L₁ cerrado  ⟺  C ≥ C_{L₁}.
        Equivale a la imposibilidad de transporte heteroclínico entre
        las cuencas de m₁ y m₂ (Koon–Lo–Marsden–Ross).
        """
        return float(jacobi_C) + 1e-12 >= self.jacobi_at_lagrange(1)

    def routh_stability(self) -> bool:
        r"""L₄, L₅ linealmente estables ssi μ < μ_Routh (Poincaré–Gascheau)."""
        return float(self.mu) < self.MU_ROUTH

    # -- Linearización y exponentes característicos (Vol. I, §§57–69) ─────
    def variational_matrix_planar(self, q: RealVector) -> RealMatrix:
        r"""
        Jacobiano 4×4 del CRTBP planar en (x, y, ẋ, ẏ):

            d/dt (δx, δy, δẋ, δẏ) = A(q) · δz,

        con términos de Coriolis ±2 y hessiano de Ω.  Los exponentes
        característicos de Poincaré son los autovalores de A(Lₖ).
        """
        q = np.asarray(q, dtype=np.float64).reshape(3)
        mu = float(self.mu)
        r1, r2 = self._radii(q)
        dx1, dy1 = q[0] + mu, q[1]
        dx2, dy2 = q[0] - 1.0 + mu, q[1]
        # Hessiano de (1−μ)/r₁ + μ/r₂
        def _hess_body(dx: float, dy: float, r: float, mass: float) -> Tuple[float, float, float]:
            r5 = r ** 5
            Uxx = mass * (3.0 * dx * dx / r5 - 1.0 / (r ** 3))
            Uyy = mass * (3.0 * dy * dy / r5 - 1.0 / (r ** 3))
            Uxy = mass * (3.0 * dx * dy / r5)
            return Uxx, Uyy, Uxy

        Uxx1, Uyy1, Uxy1 = _hess_body(dx1, dy1, r1, 1.0 - mu)
        Uxx2, Uyy2, Uxy2 = _hess_body(dx2, dy2, r2, mu)
        Omega_xx = 1.0 + Uxx1 + Uxx2
        Omega_yy = 1.0 + Uyy1 + Uyy2
        Omega_xy = Uxy1 + Uxy2
        A = np.zeros((4, 4), dtype=np.float64)
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[2, 0] = Omega_xx
        A[2, 1] = Omega_xy
        A[2, 3] = 2.0
        A[3, 0] = Omega_xy
        A[3, 1] = Omega_yy
        A[3, 2] = -2.0
        return A

    def characteristic_exponents(self, lagrange_index: int) -> NDArray[np.complex128]:
        r"""Espectro de A(Lₖ): exponentes característicos de Poincaré."""
        A = self.variational_matrix_planar(self.lagrange_point(lagrange_index))
        return np.asarray(la.eigvals(A), dtype=np.complex128)

    def lyapunov_spectral_radius(self, lagrange_index: int) -> float:
        r"""Radio espectral de A(Lₖ): max |Re λ| (tasa de escape hiperbólico)."""
        eigs = self.characteristic_exponents(lagrange_index)
        return float(np.max(np.abs(np.real(eigs))))

    def symplectic_spectrum_residual(self, lagrange_index: int, tol: float = 1e-8) -> float:
        r"""
        Un flujo hamiltoniano tiene espectro simétrico {λ, −λ, λ̄, −λ̄}.
        Residuo: distancia del multiconjunto σ al conjunto −σ.
        """
        eigs = np.sort_complex(self.characteristic_exponents(lagrange_index))
        neg = np.sort_complex(-eigs)
        return float(np.max(np.abs(eigs - neg)))

    # -- Escala de Jacobi inducida por un escenario adversarial ────────────
    def scenario_jacobi_energy(
        self,
        cost_delta_ratio: float,
        dissipation_watts: float,
        betti_1: int,
    ) -> float:
        r"""
        Inmersión del escenario presupuestal en una hoja de Jacobi:

            C_sc = C_{L₁} + α(1 − |δ|) − β |P_dis| − γ β₁.

        |δ| grande o disipación alta empujan C bajo C_{L₁} (cuello abierto).
        """
        c_l1 = self.jacobi_at_lagrange(1)
        alpha, beta, gamma = 0.45, 0.020, 0.12
        return float(
            c_l1
            + alpha * (1.0 - min(abs(cost_delta_ratio), 1.5))
            - beta * abs(dissipation_watts)
            - gamma * float(betti_1)
        )

    def classify_lagrange_gateway(self, jacobi_C: float) -> int:
        r"""
        Índice de Lagrange 'activo' según el nivel de Jacobi:
            C ≥ C_{L₄} → 4 (cuencas totalmente disjuntas + islas equiláteras),
            C ≥ C_{L₂} → 2,  C ≥ C_{L₁} → 1,  si no → 0 (escape).
        """
        c1 = self.jacobi_at_lagrange(1)
        c2 = self.jacobi_at_lagrange(2)
        c4 = self.jacobi_at_lagrange(4)
        C = float(jacobi_C)
        if C >= c4:
            return 4
        if C >= c2:
            return 2
        if C >= c1:
            return 1
        return 0


@dataclass(frozen=True, slots=True)
class PoincareInvariantManifoldTube:
    r"""
    Tubo de variedad invariante W^{s/u}(γ_{Lₖ}) en el sentido de
    Koon–Lo–Marsden–Ross (heteroclínicas del CRTBP).

    En el enclave onírico el tubo es el canal de transporte entre
    im P_d e im P_p.  La sección de Poincaré Σ = {y = 0, ẏ > 0}
    reduce el flujo a un mapa P : Σ → Σ (Vol. I, §§36–40).
    """

    lagrange_index: int
    jacobi_C: float
    stable_exponent: complex
    unstable_exponent: complex
    neck_closed: bool
    homoclinic_residue: float
    section_area: float

    @classmethod
    def from_crtbp(
        cls,
        crtbp: PoincareCRTBPPhaseSpace,
        lagrange_index: int,
        jacobi_C: float,
    ) -> "PoincareInvariantManifoldTube":
        eigs = crtbp.characteristic_exponents(max(lagrange_index, 1) if lagrange_index else 1)
        real_parts = np.real(eigs)
        unstable = complex(eigs[int(np.argmax(real_parts))])
        stable = complex(eigs[int(np.argmin(real_parts))])
        neck = crtbp.hill_neck_is_closed(jacobi_C)
        # Residuo homoclínico (Melnikov discreto): apertura del cuello × |λ_u|.
        gap = max(crtbp.jacobi_at_lagrange(1) - jacobi_C, 0.0)
        hom = float(gap * abs(unstable.real))
        # Área de sección (Liouville): proporcional a (C_L1 − C)_+ nula si cerrado.
        area = 0.0 if neck else float(math.pi * gap)
        return cls(
            lagrange_index=int(lagrange_index),
            jacobi_C=float(jacobi_C),
            stable_exponent=stable,
            unstable_exponent=unstable,
            neck_closed=neck,
            homoclinic_residue=hom,
            section_area=area,
        )

    def poincare_return_multiplier(self) -> complex:
        r"""Multiplicador de Floquet λ_u de la silla (mapa de primer retorno)."""
        return self.unstable_exponent

    def is_hyperbolic_saddle(self, tol: float = 1e-9) -> bool:
        r"""Silla hiperbólica ⟺ |Re λ_u| > 0 y |Re λ_s| > 0 con signos opuestos."""
        return (
            abs(self.unstable_exponent.real) > tol
            and abs(self.stable_exponent.real) > tol
            and (self.unstable_exponent.real * self.stable_exponent.real) < 0.0
        )


@dataclass(frozen=True, slots=True)
class LindstedtPoincareCanonicalGenerator:
    r"""
    Función generatriz de tipo II de Poincaré–von Zeipel (Vol. II, caps. VIII–X):

        F₂(q, P; ε) = q · P + ε S₁(q, P) + ε² S₂(q, P) + ⋯

    induce la transformación canónica cercana a la identidad

        p = ∂F₂/∂q,    Q = ∂F₂/∂P,

    y la corrección de Lindstedt de la frecuencia

        ω(ε) = ω₀ + ε ω₁ + ε² ω₂ + ⋯

    elegida para aniquilar términos seculares en la serie de la solución.
    En dimensión finita, S₁ se toma como el generador antihermítico
    que diagonaliza el bloque perturbativo de H en la base de H₀.
    """

    epsilon: float
    omega_0: float
    omega_1: float
    secular_residual: float
    generating_matrix: ComplexMatrix

    SECULAR_TOL: ClassVar[float] = 1e-10

    @classmethod
    def from_hamiltonian_split(
        cls,
        H0: ComplexMatrix,
        H1: ComplexMatrix,
        epsilon: float,
    ) -> "LindstedtPoincareCanonicalGenerator":
        r"""
        Construye S₁ por la ecuación homológica de Poincaré:

            {S₁, H₀} + H₁^{osc} = 0,     ω₁ = ⟨H₁⟩_{torus}.

        En matrices: (ad_{H₀}) S₁ = −H₁^{off-diag}, ω₁ = media diagonal de H₁.
        """
        H0h = 0.5 * (H0 + H0.conj().T)
        evals, U = la.eigh(H0h)
        H1_rot = U.conj().T @ H1 @ U
        n = H0.shape[0]
        S_rot = np.zeros((n, n), dtype=np.complex128)
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
        S = 0.5 * (S - S.conj().T)  # antihermítica (flujo unitario)
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

    def transformed_hamiltonian(self, H0: ComplexMatrix, H1: ComplexMatrix) -> ComplexMatrix:
        r"""
        H' = H₀ + ε (H₁ + [S₁, H₀]) + O(ε²).
        La homológica garantiza que H' es diagonal a orden ε salvo residuo secular.
        """
        ad = self.generating_matrix @ H0 - H0 @ self.generating_matrix
        Hp = H0 + self.epsilon * (H1 + ad)
        return 0.5 * (Hp + Hp.conj().T)

    def kills_secular_terms(self) -> bool:
        r"""True ssi el residuo secular está bajo tolerancia."""
        return self.secular_residual < self.SECULAR_TOL


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.5 — Red circuital no recíproca (Tellegen–Onsager–Casimir)
# ─────────────────────────────────────────────────────────────────────────────
class NonReciprocalTellegenNetwork:
    r"""
    Red AC no recíproca sobre el 1-esqueleto de K, con tierra por componente
    conexa (β₀ ≥ 1). Elementos giroscópicos Y_b − Y_bᵀ ≠ 0 modelan disipación
    asimétrica (análogo circuital de las fuerzas de Coriolis del CRTBP).

    Identidades estructurales:
        • Tellegen: Σ_e v_e conj(i_e) = V† Y_bus V.
        • Pasividad: Re Y_b ⪰ 0.
        • Onsager–Casimir: A = ½(Y_b − Y_bᵀ) mide el defecto de inversión temporal.
    """

    def __init__(
        self,
        graph: SimplicialHodgeGraph,
        frequency_rad_s: float = 50.0,
    ) -> None:
        self.graph = graph
        self.omega = float(frequency_rad_s)
        self.num_edges = len(self.graph.edges)
        self.branch_admittance = self._build_admittance_matrix()

    def _build_admittance_matrix(self) -> ComplexMatrix:
        r"""Y_b = G + jB + j·I_gyro (acoplamiento giroscópico antisimétrico)."""
        dim = self.num_edges
        Yb = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            r_k = 0.8 + 0.05 * (i + 1)
            x_k = self.omega * 0.02 - 1.0 / (self.omega * 0.08 + 1e-6)
            Yb[i, i] = 1.0 / complex(r_k, x_k)
            if i + 1 < dim:
                gyration = 0.15 * (i + 1)
                Yb[i, i + 1] += complex(0.0, gyration)
                Yb[i + 1, i] -= complex(0.0, gyration)
        return Yb

    def reciprocity_defect_norm(self) -> float:
        r"""‖½(Y_b − Y_bᵀ)‖∞ : norma espectral del defecto de Onsager."""
        antisym = 0.5 * (self.branch_admittance - self.branch_admittance.T)
        sv = la.svdvals(antisym)
        return float(np.max(sv)) if len(sv) > 0 else 0.0

    def passivity_margin(self) -> float:
        r"""λ_min((Y_b + Y_b†)/2). Negativo ⇒ violación de pasividad."""
        herm = 0.5 * (self.branch_admittance + self.branch_admittance.conj().T)
        return float(np.min(la.eigvalsh(herm)))

    def symmetric_part_norm(self) -> float:
        r"""‖½(Y_b + Y_b†)‖∞ : contribución simétrica (disipativa)."""
        herm = 0.5 * (self.branch_admittance + self.branch_admittance.conj().T)
        sv = la.svdvals(herm)
        return float(np.max(sv)) if len(sv) > 0 else 0.0

    def compute_bus_admittance_matrix(self) -> ComplexMatrix:
        r"""Y_bus = ∂₁ Y_b ∂₁ᵀ."""
        B1 = self.graph.boundary_1
        return B1 @ self.branch_admittance @ B1.T

    def solve_network_dissipation(
        self, current_stimulus: RealVector
    ) -> Tuple[float, ComplexMatrix, ComplexMatrix]:
        r"""
        Resuelve Y_bus V = I con V = 0 en un nodo de tierra por componente conexa.
        dissipation = Re[V† Y_bus V] = Re[Σ_e v_e* i_e].
        """
        n = self.graph.num_vertices
        components = self.graph.connected_components()
        ground_nodes = {min(c) for c in components}
        free_nodes = [i for i in range(n) if i not in ground_nodes]
        B1 = self.graph.boundary_1
        Y_bus = B1 @ self.branch_admittance @ B1.T
        V_nodes = np.zeros(n, dtype=np.complex128)
        if free_nodes:
            Y_red = Y_bus[np.ix_(free_nodes, free_nodes)]
            I_red = current_stimulus[free_nodes]
            try:
                V_red = la.solve(Y_red, I_red)
            except la.LinAlgError:
                logger.warning("Y_red casi-singular: lstsq regularizado.")
                V_red, *_ = la.lstsq(Y_red, I_red)
            for local_idx, node in enumerate(free_nodes):
                V_nodes[node] = V_red[local_idx]
        branch_v = B1.T @ V_nodes
        branch_i = self.branch_admittance @ branch_v
        dissipation = float(np.real(np.sum(branch_v * np.conj(branch_i))))
        return dissipation, branch_v, V_nodes

    def kcl_residual(self, v_nodes: ComplexMatrix, i_inj: ComplexMatrix) -> float:
        r"""‖Y_bus V − I‖₂ / max(1, ‖I‖₂)."""
        I_pred = self.compute_bus_admittance_matrix() @ v_nodes
        denom = max(float(np.linalg.norm(i_inj)), 1.0)
        return float(np.linalg.norm(I_pred - i_inj) / denom)

    def verify_tellegen_conservation(
        self, branch_v: ComplexMatrix, v_nodes: ComplexMatrix
    ) -> float:
        r"""Identidad de Tellegen auto-consistente; residuo relativo."""
        B1 = self.graph.boundary_1
        Y_bus = B1 @ self.branch_admittance @ B1.T
        I_full = Y_bus @ v_nodes
        branch_i = self.branch_admittance @ branch_v
        lhs = np.sum(branch_v * np.conj(branch_i))
        rhs = np.sum(np.conj(v_nodes) * I_full)
        denom = max(abs(rhs), 1e-12)
        return float(abs(lhs - rhs) / denom)

    def coriolis_gyroscopic_norm(self) -> float:
        r"""
        Norma del bloque giroscópico (análogo circuital de ±2 ẋ ∧ ẑ del CRTBP).
        Coincide con el defecto de reciprocidad de Onsager–Casimir.
        """
        return self.reciprocity_defect_norm()


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.6 — Semilla ABC de dinámica cuántica abierta (germen de la bisagra)
# ─────────────────────────────────────────────────────────────────────────────
class OpenQuantumDynamicsSeed(ABC):
    r"""
    Germen formal de la flecha canónica de Poincaré

        Cart × CRTBP × F₂  ──lift_enclave_hamiltonian──►  𝔥𝔢𝔯(ℋ_dream)

    que eleva la estructura categórico-topológica de un cartucho adversarial
    a un Hamiltoniano hermitiano, único combustible del motor GKSL (FASE 2).

    CONTRATO DE BISAGRA FASE 1 → FASE 2
    -----------------------------------
    Toda implementación DEBE:

        (H1) devolver H = H†  (‖H − H†‖_F < 10⁻⁹),
        (H2) embeber la escala de Jacobi C_J como gap espectral,
        (H3) inyectar la función generatriz F₂ (Lindstedt) como corrección
             de orden ε sobre el bloque 2×2 cliffordiano,
        (H4) anular el flujo a través del cuello de Hill cuando
             `is_homologically_isolated` es verdadero
             (off-diagonales dream–physical ≡ 0).
    """

    @abstractmethod
    def lift_enclave_hamiltonian(self, hilbert_dim: int) -> ComplexMatrix:
        r"""Produce H = H† a partir del cartucho, su CRTBP interno y F₂."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1.7 — Categoría **Cart** de cartuchos adversariales
#            NEXO FORMAL TERMINAL DE LA FASE 1
#            (último método = inicio formal de la FASE 2)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CartridgeMorphism:
    r"""
    Morfismo f : A → B entre cartuchos adversariales, preservando Hodge y
    Tellegen. Se representa por una permutación de aristas donde A.csr ⊂ B.csr
    y la masa espectral no aumenta.  Simplecticidad discreta: el jacobiano
    combinatorio de f tiene det ±1 (análogo de PSL(2, ℤ) sobre ℱ).
    """

    source_id: str
    target_id: str
    edge_permutation: Tuple[Tuple[int, int], ...]
    cost_ratio_diff: float
    betti_1_diff: int

    def is_monomorphism(self) -> bool:
        r"""f inyectivo ⟺ todos los índices de la permutación son distintos."""
        flat = [x for pair in self.edge_permutation for x in pair]
        return len(flat) == len(set(flat))


@dataclass(frozen=True, slots=True)
class CategoricalCircuitCartridge(OpenQuantumDynamicsSeed):
    r"""
    NEXO FORMAL TERMINAL DE LA FASE 1.

    Objeto de la categoría **Cart** de cartuchos adversariales:

        Ob(Cart) = { C = (id, escenario, Hodge, Tellegen, Clifford, CRTBP) },
        Hom(C, C') = { morfismos que preservan Hodge, Tellegen y Jacobi }.

    Encapsula simplicidad topológica (βₖ, χ), espectro de Hodge, respuesta
    circuital no recíproca, gauge cliffordiano, valuación Ω₄ y los invariantes
    celestes de Poincaré (C_J, Lₖ, cuello de Hill, tubos W^{s/u}, F₂).

    El método terminal `lift_enclave_hamiltonian` ES la flecha canónica
    Cart → 𝔥𝔢𝔯(ℋ) que inaugura la FASE 2: no existe otro constructor de H.
    """

    cartridge_id: str
    scenario_type: str
    cost_delta_ratio: float
    token_count: int
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    euler_betti_consistent: bool
    hodge_spectral_gap: float
    algebraic_connectivity: float
    hodge_decomposition_residual: float
    circuit_dissipation_watts: float
    reciprocity_defect: float
    passivity_margin: float
    tellegen_residual: float
    kcl_residual: float
    clifford_gauge_perturbation: BiquaternionClifford
    clifford_cstar_residual: float
    hodge_graph: SimplicialHodgeGraph
    tellegen_network: NonReciprocalTellegenNetwork
    spectral_coherence_index: float
    topos_heyting_sieve: HeytingToposAlgebra
    is_homologically_isolated: bool
    metadata_payload: Dict[str, Any]
    # --- Invariantes celestes de Poincaré (Vols. I–III) ---
    crtbp: PoincareCRTBPPhaseSpace
    poincare_jacobi_constant: float
    lagrange_gateway_index: int
    hill_neck_closed: bool
    lyapunov_spectral_radius: float
    symplectic_spectrum_residual: float
    invariant_tube: PoincareInvariantManifoldTube
    poincare_duality_residual: int

    _KAPPA_BETTI: ClassVar[float] = 1.2
    _KAPPA_DISSIPATION: ClassVar[float] = 14.0
    _KAPPA_COST: ClassVar[float] = 0.55
    _LINDSTEDT_EPSILON: ClassVar[float] = 0.18

    @staticmethod
    def _spectral_coherence_functional(
        betti_1: int,
        betti_2: int,
        dissipation: float,
        cost_delta_ratio: float,
        is_dream_state: bool,
    ) -> float:
        r"""
        C = 𝟙{dream} · exp(−(β₁+β₂)/κ_b) · exp(−|P_dis|/κ_d) · exp(−|δ|/κ_c).
        """
        if not is_dream_state:
            return 0.0
        kb = CategoricalCircuitCartridge._KAPPA_BETTI
        kd = CategoricalCircuitCartridge._KAPPA_DISSIPATION
        kc = CategoricalCircuitCartridge._KAPPA_COST
        return (
            math.exp(-(betti_1 + betti_2) / kb)
            * math.exp(-abs(dissipation) / kd)
            * math.exp(-abs(cost_delta_ratio) / kc)
        )

    @staticmethod
    def _classify_sieve(coherence: float) -> HeytingToposAlgebra:
        r"""Valuación Ω₄: C ≥ 0.70 ⇒ ⊤; ≥ 0.40 ⇒ ♯; ≥ 0.12 ⇒ ∂; si no ⊥."""
        if coherence >= 0.70:
            return HeytingToposAlgebra.VERUM_COHERENT
        if coherence >= 0.40:
            return HeytingToposAlgebra.TOPOLOGICAL_STABLE
        if coherence >= 0.12:
            return HeytingToposAlgebra.BOUNDARY_CRITICAL
        return HeytingToposAlgebra.VETOED_ABSURDUM

    @classmethod
    def synthesize_cartridge(
        cls,
        cartridge_id: str,
        scenario_type: str,
        cost_delta_ratio: float,
        synthetic_betti_1: int,
        is_dream_state: bool,
        payload: Dict[str, Any],
        deterministic_extra_edge: Optional[Tuple[int, int]] = None,
        mu: float = PoincareCRTBPPhaseSpace.MU_EARTH_MOON,
    ) -> "CategoricalCircuitCartridge":
        r"""
        Sintetiza un cartucho adversarial y lo sumerge en una hoja de Jacobi
        del CRTBP Earth–Moon (o μ prescrito).  El cuello de Hill determina
        el aislamiento homológico efectivo.
        """
        num_v = 6
        edges = [
            (0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5), (5, 0)
        ]
        faces = [(0, 1, 2), (2, 3, 4)]
        if deterministic_extra_edge is not None:
            u, v = deterministic_extra_edge
            if u != v:
                edges.append(tuple(sorted((u, v))))
        if synthetic_betti_1 >= 1:
            edges.append((1, 4))
        if synthetic_betti_1 >= 2:
            edges.append((3, 0))
        if synthetic_betti_1 >= 3:
            edges.append((1, 5))
        seen: Set[Tuple[int, int]] = set()
        dedup_edges: List[Tuple[int, int]] = []
        for e in edges:
            e_s = tuple(sorted(e))
            if e_s not in seen:
                seen.add(e_s)
                dedup_edges.append(e_s)
        graph = SimplicialHodgeGraph(
            num_vertices=num_v, edges=dedup_edges, faces=faces
        )
        b0, b1, b2 = graph.compute_betti_numbers()
        chi = graph.euler_characteristic()
        euler_ok = graph.verify_betti_euler_consistency()
        if not euler_ok:
            logger.warning(
                "Inconsistencia Euler–Poincaré en cartucho %s: χ=%d.",
                cartridge_id, chi,
            )
        spectral_gap = graph.compute_spectral_gap()
        fiedler = graph.algebraic_connectivity()
        pd_res = graph.poincare_duality_residual()
        tellegen = NonReciprocalTellegenNetwork(graph=graph)
        stimulus = np.zeros(num_v, dtype=np.complex128)
        stimulus[0] = complex(1.0 + cost_delta_ratio, 0.2)
        stimulus[-1] = -complex(1.0 + cost_delta_ratio, 0.2)
        dissipation, branch_v, v_nodes = tellegen.solve_network_dissipation(stimulus)
        reciprocity_defect = tellegen.reciprocity_defect_norm()
        passivity = tellegen.passivity_margin()
        tellegen_res = tellegen.verify_tellegen_conservation(branch_v, v_nodes)
        kcl = tellegen.kcl_residual(v_nodes, stimulus)
        omega_probe = np.real(branch_v)
        if omega_probe.size == len(graph.edges) and omega_probe.size > 0:
            ex, co, ha = graph.hodge_decompose_1_form(omega_probe)
            hodge_res = float(np.linalg.norm(omega_probe - (ex + co + ha)))
        else:
            hodge_res = 0.0
        clifford_gauge = BiquaternionClifford(
            w_re=1.0,
            w_im=cost_delta_ratio * 0.1,
            x_re=cost_delta_ratio * 0.5,
            x_im=0.0,
            y_re=0.0,
            y_im=cost_delta_ratio * 0.2,
            z_re=float(b1) * 0.3,
            z_im=0.0,
        )
        crtbp = PoincareCRTBPPhaseSpace(mu=float(mu))
        jacobi_C = crtbp.scenario_jacobi_energy(cost_delta_ratio, dissipation, b1)
        neck_closed = crtbp.hill_neck_is_closed(jacobi_C) and is_dream_state
        lg_index = crtbp.classify_lagrange_gateway(jacobi_C)
        tube = PoincareInvariantManifoldTube.from_crtbp(
            crtbp, max(lg_index, 1), jacobi_C
        )
        lyap = crtbp.lyapunov_spectral_radius(max(lg_index, 1))
        symp_res = crtbp.symplectic_spectrum_residual(max(lg_index, 1))
        coherence = cls._spectral_coherence_functional(
            b1, b2, dissipation, cost_delta_ratio, is_dream_state
        )
        sieve_coh = cls._classify_sieve(coherence)
        sieve_lag = HeytingToposAlgebra.from_lagrange_stability(
            lg_index if lg_index else 1, float(mu), neck_closed
        )
        sieve = sieve_coh.meet(sieve_lag).lawvere_tierney_closure()
        token_count = max(1, 24 + int(abs(cost_delta_ratio) * 40) + 8 * b1)
        return cls(
            cartridge_id=cartridge_id,
            scenario_type=scenario_type,
            cost_delta_ratio=cost_delta_ratio,
            token_count=token_count,
            betti_0=b0,
            betti_1=b1,
            betti_2=b2,
            euler_characteristic=chi,
            euler_betti_consistent=euler_ok,
            hodge_spectral_gap=spectral_gap,
            algebraic_connectivity=fiedler,
            hodge_decomposition_residual=hodge_res,
            circuit_dissipation_watts=dissipation,
            reciprocity_defect=reciprocity_defect,
            passivity_margin=passivity,
            tellegen_residual=tellegen_res,
            kcl_residual=kcl,
            clifford_gauge_perturbation=clifford_gauge,
            clifford_cstar_residual=clifford_gauge.cstar_residual(),
            hodge_graph=graph,
            tellegen_network=tellegen,
            spectral_coherence_index=coherence,
            topos_heyting_sieve=sieve,
            is_homologically_isolated=bool(neck_closed and is_dream_state),
            metadata_payload=payload,
            crtbp=crtbp,
            poincare_jacobi_constant=jacobi_C,
            lagrange_gateway_index=lg_index,
            hill_neck_closed=neck_closed,
            lyapunov_spectral_radius=lyap,
            symplectic_spectrum_residual=symp_res,
            invariant_tube=tube,
            poincare_duality_residual=pd_res,
        )

    # -- Generadores granulares de la bisagra ──────────────────────────────
    def _unperturbed_number_hamiltonian(self, dim: int) -> ComplexMatrix:
        r"""
        H₀ = diag( (i+1)·Δ_hodge + 0.02·P_dis + κ_J·|C_J − C_{L₁}| ).
        El gap de Jacobi desplaza el espectro exactamente como la energía
        de una órbita de Lyapunov alrededor de L₁ (Vol. I, §47).
        """
        H0 = np.zeros((dim, dim), dtype=np.complex128)
        gap = self.hodge_spectral_gap
        c_l1 = self.crtbp.jacobi_at_lagrange(1)
        jacobi_shift = 0.08 * abs(self.poincare_jacobi_constant - c_l1)
        for i in range(dim):
            H0[i, i] = (
                (i + 1) * gap
                + (self.circuit_dissipation_watts * 0.02)
                + jacobi_shift
            )
        return H0

    def _clifford_perturbation_block(self, dim: int) -> ComplexMatrix:
        r"""H₁ : inmersión del gauge cliffordiano (bloque 2×2) y ceros."""
        H1 = np.zeros((dim, dim), dtype=np.complex128)
        herm = self.clifford_gauge_perturbation.hermitian_part()
        block = min(2, dim)
        H1[0:block, 0:block] = herm[:block, :block]
        return H1

    def _lindstedt_generator(
        self, dim: int
    ) -> LindstedtPoincareCanonicalGenerator:
        r"""F₂ de Poincaré–Lindstedt a orden ε sobre (H₀, H₁)."""
        H0 = self._unperturbed_number_hamiltonian(dim)
        H1 = self._clifford_perturbation_block(dim)
        return LindstedtPoincareCanonicalGenerator.from_hamiltonian_split(
            H0, H1, epsilon=self._LINDSTEDT_EPSILON
        )

    def _enforce_hill_neck_blockade(self, H: ComplexMatrix) -> ComplexMatrix:
        r"""
        (H4) Si el cuello de Hill está cerrado, se anulan las coherencias
        entre el primer tercio (dream) y el último tercio (physical) del
        espectro — analogon matricial de P_d H P_p = 0.
        """
        if not self.hill_neck_closed:
            return H
        n = H.shape[0]
        if n < 3:
            return H
        cut_d = max(1, n // 3)
        cut_p = n - max(1, n // 3)
        H = H.copy()
        H[:cut_d, cut_p:] = 0.0
        H[cut_p:, :cut_d] = 0.0
        return H

    # ══════════════════════════════════════════════════════════════════════
    # MÉTODO BISAGRA FASE 1 → FASE 2
    # Definición formal terminal de la FASE 1 e inicio de la FASE 2.
    # ══════════════════════════════════════════════════════════════════════
    def lift_enclave_hamiltonian(self, hilbert_dim: int) -> ComplexMatrix:
        r"""
        FLECHA CANÓNICA DE POINCARÉ  Cart × CRTBP × F₂  →  𝔥𝔢𝔯(ℋ_dream).

        Este es el último morfismo de la FASE 1 y el único constructor de H
        que la FASE 2 (`MetabolizedPerturbationField.evolve_and_certify`)
        está autorizada a consumir.

        Construcción (contratos H1–H4 del germen `OpenQuantumDynamicsSeed`):

            (1)  H₀ = N_Hodge + κ_J |C_J − C_{L₁}|          (integrable),
            (2)  H₁ = Re φ(q_Clifford)  sobre el bloque 2×2  (perturbación),
            (3)  F₂ : S₁ resuelve {S₁, H₀} + H₁^{osc} = 0    (Lindstedt),
            (4)  H  = H₀ + ε (H₁ + [S₁, H₀])                 (von Zeipel),
            (5)  si cuello de Hill cerrado: P_d H P_p = 0    (no-signaling),
            (6)  H  ← ½(H + H†)  con ‖H − H†‖_F < 10⁻⁹      (hermiticidad).

        El espectro de H hereda los exponentes característicos de Lₖ
        a través del radio de Lyapunov y del gap de Jacobi, de modo que
        el flujo GKSL de la FASE 2 es un análogo cuántico del mapa de
        primer retorno de Poincaré sobre la sección Σ = {y=0, ẏ>0}.
        """
        if hilbert_dim < 1:
            raise ValueError("hilbert_dim debe ser ≥ 1.")
        dim = int(hilbert_dim)
        H0 = self._unperturbed_number_hamiltonian(dim)
        H1 = self._clifford_perturbation_block(dim)
        generator = self._lindstedt_generator(dim)
        H = generator.transformed_hamiltonian(H0, H1)
        # Acoplamiento débil por el radio de Lyapunov (silla Lₖ).
        n = min(dim, 2)
        H[:n, :n] = H[:n, :n] + 0.05 * self.lyapunov_spectral_radius * np.eye(n)
        H = self._enforce_hill_neck_blockade(H)
        H = 0.5 * (H + H.conj().T)
        herm_res = float(la.norm(H - H.conj().T))
        assert herm_res < 1e-9, f"H no hermitiano: ‖H−H†‖={herm_res:.3e}"
        return H


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — C*-ÁLGEBRA, ENCLAVE NO-SIGNALING, GKSL FUCSIANO Y CAMPO METABOLIZADO
#
#   Continuación DIRECTA y ÚNICA de
#       CategoricalCircuitCartridge.lift_enclave_hamiltonian
#   de la FASE 1: el Hamiltoniano elevado H_Poincaré alimenta el motor GKSL
#   dentro del enclave P_d ℋ P_d, se estroboscopía por el mapa de primer
#   retorno de Poincaré, y termina en
#       MetabolizedPerturbationField.evolve_and_certify
#   — método bisagra hacia la FASE 3.
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# FASE 2.1 — C*-álgebra B(ℋ): normas, cono, métricas cuánticas
# ─────────────────────────────────────────────────────────────────────────────
class BanachSpectralAlgebra:
    r"""
    C*-álgebra B(ℋ) de dimensión finita.

    Normas de Schatten:
        ‖A‖₁ = Tr √(A†A),   ‖A‖₂ = √Tr(A†A),   ‖A‖∞ = σ_max(A).

    Cono de operadores densidad:
        𝔇(ℋ) = { ρ ∈ B(ℋ) : ρ = ρ†, ρ ⪰ 0, Tr ρ = 1 }.

    Métricas cuánticas (la distancia de Bures es la geodésica de Fisher–Rao
    cuántica, análogo de la métrica de Poincaré en ℍ²):
        F(ρ, σ) = (Tr √(√ρ σ √ρ))²,
        d_B(ρ, σ) = √(2(1 − √F)),
        D(ρ ‖ σ) = Tr(ρ (log ρ − log σ)).
    """

    SPECTRUM_FLOOR: Final[float] = 1e-15

    @staticmethod
    def project_to_state_manifold(rho: ComplexMatrix) -> ComplexMatrix:
        r"""Proyección ℝ-lineal a 𝔇(ℋ) vía eigendescomposición + clipping."""
        rho_h = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals = np.maximum(np.real(eigvals), BanachSpectralAlgebra.SPECTRUM_FLOOR)
        eigvals /= np.sum(eigvals)
        proj = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        return 0.5 * (proj + proj.conj().T)

    @staticmethod
    def is_valid_density_matrix(rho: ComplexMatrix, tol: float = 1e-8) -> bool:
        r"""Test ρ ∈ 𝔇(ℋ): hermiticidad, positividad y traza 1."""
        if float(la.norm(rho - rho.conj().T)) > tol:
            return False
        eigvals = la.eigvalsh(0.5 * (rho + rho.conj().T))
        if np.any(eigvals < -tol):
            return False
        return abs(np.real(np.trace(rho)) - 1.0) < tol

    @staticmethod
    def schatten_1_norm(A: ComplexMatrix) -> float:
        r"""‖A‖₁ = Σ σ_i(A)."""
        return float(np.sum(la.svdvals(A)))

    @staticmethod
    def schatten_2_norm(A: ComplexMatrix) -> float:
        r"""‖A‖₂ = √Tr(A†A)."""
        return float(np.sqrt(np.real(np.trace(A.conj().T @ A))))

    @staticmethod
    def schatten_infty_norm(A: ComplexMatrix) -> float:
        r"""‖A‖∞ = σ_max(A)."""
        s = la.svdvals(A)
        return float(s[0]) if len(s) > 0 else 0.0

    @staticmethod
    def von_neumann_entropy(rho: ComplexMatrix) -> float:
        r"""S(ρ) = −Σ λ_i log₂ λ_i en bits, con 0·log 0 ≡ 0."""
        eigvals = la.eigvalsh(rho)
        eigvals = eigvals[eigvals > BanachSpectralAlgebra.SPECTRUM_FLOOR]
        return -float(np.sum(eigvals * np.log2(eigvals)))

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
        r"""D(ρ ‖ σ) = Tr(ρ (log ρ − log σ)) en bits. Klein: ≥ 0, = 0 ⇔ ρ = σ."""
        log_rho = BanachSpectralAlgebra._matrix_log_regularized(rho, floor)
        log_sigma = BanachSpectralAlgebra._matrix_log_regularized(sigma, floor)
        val = np.real(np.trace(rho @ (log_rho - log_sigma))) / math.log(2.0)
        return float(max(val, 0.0))

    @staticmethod
    def quantum_fidelity(rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""F(ρ, σ) = (Tr √(√ρ σ √ρ))² ∈ [0, 1] (Jozsa 1994)."""
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
        f = BanachSpectralAlgebra.quantum_fidelity(rho, sigma)
        return float(math.sqrt(max(2.0 * (1.0 - math.sqrt(max(f, 0.0))), 0.0)))

    @staticmethod
    def bures_geodesic(
        rho: ComplexMatrix, sigma: ComplexMatrix, t: float
    ) -> ComplexMatrix:
        r"""
        Punto sobre la geodésica de Bures–Wasserstein a fracción t ∈ [0, 1]:
            γ(t) = ((1−t) √ρ + t U √σ)² / Tr(...),
        con U la fase unitaria óptima de SVD(√ρ · √σ).
        Análogo cuántico de la geodésica hiperbólica d_ℍ².
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
        interp = (1.0 - t) * sqrt_rho + t * (phase @ sqrt_sigma)
        candidate = interp @ interp.conj().T
        return BanachSpectralAlgebra.project_to_state_manifold(candidate)

    @staticmethod
    def cstar_residual(rho: ComplexMatrix) -> float:
        r"""| ‖ρ†ρ‖∞ − ‖ρ‖∞² |  (nulo en aritmética exacta)."""
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

    @staticmethod
    def data_processing_inequality_residual(
        rho: ComplexMatrix,
        sigma: ComplexMatrix,
        kraus_ops: Sequence[ComplexMatrix],
        floor: float = 1e-12,
    ) -> float:
        r"""
        DPI cuántica: D(ρ ‖ σ) ≥ D(𝒩(ρ) ‖ 𝒩(σ)),  𝒩(X) = Σ_k K_k X K_k†.
        Retorna (D_orig − D_next) ≥ 0 en aritmética exacta.
        """
        d_orig = BanachSpectralAlgebra.quantum_relative_entropy(rho, sigma, floor)
        rho_next = sum(K @ rho @ K.conj().T for K in kraus_ops)
        sigma_next = sum(K @ sigma @ K.conj().T for K in kraus_ops)
        rho_next = BanachSpectralAlgebra.project_to_state_manifold(rho_next)
        sigma_next = BanachSpectralAlgebra.project_to_state_manifold(sigma_next)
        d_next = BanachSpectralAlgebra.quantum_relative_entropy(rho_next, sigma_next, floor)
        return float(d_orig - d_next)

    @staticmethod
    def poincare_recurrence_time_bound(
        rho: ComplexMatrix, energy_span: float, dim: Optional[int] = None
    ) -> float:
        r"""
        Cota de recurrencia de Poincaré–Kac para un flujo unitario en 𝔇(ℋ):

            τ_rec ≲ (2π / ΔE) · dim(ℋ)

        (análogo cuántico del teorema de recurrencia, Vol. III, cap. XXVI).
        ΔE se toma como el span espectral del Hamiltoniano efectivo.
        """
        n = int(dim if dim is not None else rho.shape[0])
        de = max(float(energy_span), 1e-9)
        return float((2.0 * math.pi * n) / de)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2.2 — Enclave no-signaling por proyectores ortogonales
#            (cuencas de Hill dream / physical)
# ─────────────────────────────────────────────────────────────────────────────
class NonCommutativeNoSignalingEnclave:
    r"""
    Enclave cuántico definido por proyectores ortogonales P_d (dream) y P_p
    (physical) sobre ℋ = ℋ_dream ⊕ ℋ_physical, con P_d P_p = 0 y P_d + P_p = I.

    Analogía celestial: P_d y P_p son las cuencas de Hill de m₂ y m₁;
    el conmutador [A, P_p] mide el flujo a través del cuello L₁.

    Canal de pinching (Davies):
        Φ_pinch(ρ) = P_d ρ P_d + P_p ρ P_p
    elimina coherencias inter-cuenca (no-signaling reforzado).
    """

    LEAKAGE_TOL: Final[float] = 1e-12

    def __init__(self, hilbert_dim: int = 4, num_physical_modes: int = 1) -> None:
        if hilbert_dim < 2:
            raise ValueError("hilbert_dim ≥ 2 para partir dream/physical.")
        self.dim = int(hilbert_dim)
        self.num_physical = max(1, min(num_physical_modes, hilbert_dim - 1))
        cutoff = hilbert_dim - self.num_physical
        self.P_dream = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.P_physical = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(cutoff):
            self.P_dream[i, i] = 1.0
        for i in range(cutoff, hilbert_dim):
            self.P_physical[i, i] = 1.0

    def verify_projector_completeness(self, tol: float = 1e-12) -> bool:
        r"""Verifica P_d + P_p = I, P_d P_p = 0, P_d² = P_d, P_p² = P_p."""
        identity_defect = float(
            la.norm(self.P_dream + self.P_physical - np.eye(self.dim))
        )
        orthogonality_defect = float(la.norm(self.P_dream @ self.P_physical))
        idem_d = float(la.norm(self.P_dream @ self.P_dream - self.P_dream))
        idem_p = float(la.norm(self.P_physical @ self.P_physical - self.P_physical))
        return (
            identity_defect < tol
            and orthogonality_defect < tol
            and idem_d < tol
            and idem_p < tol
        )

    def verify_isolation_commutator(
        self, dream_operator: ComplexMatrix
    ) -> Tuple[bool, float]:
        r"""Aislamiento inter-cuenca: ‖[A, P_p]‖∞ < ε.  Retorna (ok, leakage)."""
        comm = dream_operator @ self.P_physical - self.P_physical @ dream_operator
        leakage = BanachSpectralAlgebra.schatten_infty_norm(comm)
        return bool(leakage < self.LEAKAGE_TOL), leakage

    def enforce_strict_enclave(self, dream_operator: ComplexMatrix) -> ComplexMatrix:
        r"""P_d A P_d : restricción estricta al subespacio dream."""
        return self.P_dream @ dream_operator @ self.P_dream

    def pinching_channel(self, rho: ComplexMatrix) -> ComplexMatrix:
        r"""Φ_pinch(ρ) = P_d ρ P_d + P_p ρ P_p (canal CPTP idempotente)."""
        return self.P_dream @ rho @ self.P_dream + self.P_physical @ rho @ self.P_physical

    def kraus_operators(self) -> Tuple[ComplexMatrix, ...]:
        r"""Operadores de Kraus (K₁, K₂) = (P_d, P_p) del canal de pinching."""
        return (self.P_dream, self.P_physical)

    def channel_is_trace_preserving(
        self, rho: ComplexMatrix, tol: float = 1e-10
    ) -> bool:
        r"""Verifica Tr Φ_pinch(ρ) = Tr ρ."""
        out = self.pinching_channel(rho)
        return abs(float(np.real(np.trace(out)) - np.real(np.trace(rho)))) < tol

    def hill_neck_leakage(
        self, rho: ComplexMatrix, tube: PoincareInvariantManifoldTube
    ) -> float:
        r"""
        Flujo heteroclínico cuántico a través del cuello L₁:
            Λ = ‖[ρ, P_p]‖∞ + 𝟙{¬neck} · |λ_u| · area(Σ).
        """
        _, leak = self.verify_isolation_commutator(rho)
        if tube.neck_closed:
            return leak
        return float(leak + abs(tube.unstable_exponent.real) * tube.section_area)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2.3 — Campo metabolizado
#            NEXO FORMAL TERMINAL DE LA FASE 2
#            (último método = inicio formal de la FASE 3)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class MetabolizedPerturbationField:
    r"""
    NEXO FORMAL TERMINAL DE LA FASE 2.

    Estado metabolizado por GKSL fucsiano dentro del enclave P_d ℋ P_d,
    estroboscopiado por el mapa de primer retorno de Poincaré, con
    certificación de aislamiento, coherencia metabólica y recurrencia.

    El método terminal `evolve_and_certify` ES la flecha

        Cart × ρ₀  --(H = lift_enclave_hamiltonian)-->  (ρ_dream, inv. Poincaré)

    que inaugura la FASE 3: no existe otro constructor de campo metabolizado.
    """

    source_cartridge: CategoricalCircuitCartridge
    rho_perturbed: ComplexMatrix
    dirichlet_spectral_energy: float
    von_neumann_entropy: float
    purity: float
    relative_entropy_to_base: float
    bures_distance_to_base: float
    cstar_residual: float
    no_signaling_leakage_norm: float
    projector_completeness_ok: bool
    is_strictly_isolated: bool
    jump_rate: float
    mean_first_jump_time: float
    metabolic_coherence_index: float
    topos_heyting_evaluation: HeytingToposAlgebra
    # --- Invariantes de recurrencia de Poincaré (Vol. III) ---
    poincare_recurrence_time: float
    homoclinic_residue: float
    stroboscopic_bures_increment: float
    fuchsian_tau_reduced: complex
    escape_rate_gamma: float

    _KAPPA_ENERGY: ClassVar[float] = 12.0
    _KAPPA_RELATIVE_ENTROPY: ClassVar[float] = 5.0

    @staticmethod
    def _metabolic_coherence_functional(
        purity: float,
        dirichlet_energy: float,
        relative_entropy: float,
        is_isolated: bool,
    ) -> float:
        r"""
        C_meta = 𝟙{isolated} · P(ρ) · exp(−|E_D|/κ_E) · exp(−|D(ρ‖ρ₀)|/κ_S).
        """
        if not is_isolated:
            return 0.0
        kE = MetabolizedPerturbationField._KAPPA_ENERGY
        kS = MetabolizedPerturbationField._KAPPA_RELATIVE_ENTROPY
        return (
            purity
            * math.exp(-abs(dirichlet_energy) / kE)
            * math.exp(-abs(relative_entropy) / kS)
        )

    @staticmethod
    def _classify_metabolic_state(coherence: float) -> HeytingToposAlgebra:
        r"""Valuación Ω₄: C ≥ 0.55 ⇒ ⊤; ≥ 0.30 ⇒ ♯; ≥ 0.08 ⇒ ∂; si no ⊥."""
        if coherence >= 0.55:
            return HeytingToposAlgebra.VERUM_COHERENT
        if coherence >= 0.30:
            return HeytingToposAlgebra.TOPOLOGICAL_STABLE
        if coherence >= 0.08:
            return HeytingToposAlgebra.BOUNDARY_CRITICAL
        return HeytingToposAlgebra.VETOED_ABSURDUM

    @staticmethod
    def _stroboscopic_poincare_section(
        rho_before: ComplexMatrix, rho_after: ComplexMatrix
    ) -> float:
        r"""
        Incremento de Bures bajo un golpe de primer retorno (sección Σ).
        d_B(ρ_t, ρ_{t+Δ}) mide la apertura del mapa P : Σ → Σ.
        """
        return BanachSpectralAlgebra.bures_distance(rho_before, rho_after)

    @staticmethod
    def _dirichlet_energy(rho: ComplexMatrix, H: ComplexMatrix, dissipation: float) -> float:
        r"""
        Energía de Dirichlet espectral:
            E_D = ½ Tr(ρ H) + ½ ‖[ρ, H]‖₂² + 0.1 P_dis.
        Análogo cuántico de ½‖∇f‖²_{L²} sobre la variedad de Jacobi.
        """
        comm_h = rho @ H - H @ rho
        return (
            0.5 * float(np.real(np.trace(rho @ H)))
            + 0.5 * (BanachSpectralAlgebra.schatten_2_norm(comm_h) ** 2)
            + 0.1 * dissipation
        )

    # ══════════════════════════════════════════════════════════════════════
    # MÉTODO BISAGRA FASE 2 → FASE 3
    # Definición formal terminal de la FASE 2 e inicio de la FASE 3.
    # Consume OBLIGATORIAMENTE lift_enclave_hamiltonian (FASE 1).
    # ══════════════════════════════════════════════════════════════════════
    @classmethod
    def evolve_and_certify(
        cls, cartridge: CategoricalCircuitCartridge, base_rho: ComplexMatrix
    ) -> "MetabolizedPerturbationField":
        r"""
        FLECHA CANÓNICA  (C, ρ₀) ↦ (ρ_dream, invariantes de Poincaré).

        Este es el último morfismo de la FASE 2 y el único constructor de
        campo metabolizado que la FASE 3 (vacuna espectral, Merkle,
        `TOONOniricDreamerAgent.dream_scenario`) está autorizada a consumir.

        Pipeline (continuación directa de `lift_enclave_hamiltonian`):

            C  --lift_enclave_hamiltonian-->  H_Poincaré ∈ 𝔥𝔢𝔯(ℋ)
               --GKSL fucsiano (τ ∈ ℱ ⊂ ℍ²)-->  ρ'
               --sección de Poincaré estroboscópica-->  d_B(ρ₀, ρ')
               --enclave Hill (P_d, P_p)-->  leakage = ‖[ρ', P_p]‖∞
               --recurrencia de Poincaré–Kac-->  τ_rec
               --fusión Ω₄ (cartucho ∧ metabolismo ∧ Lₖ)-->  veredicto.

        El parámetro modular τ se reduce al dominio fundamental ℱ por la
        acción de PSL(2, ℤ) (uniformización fucsiana, Vol. III + Klein).
        """
        dim = int(base_rho.shape[0])
        engine = NonHermitianLindbladMasterEngine(enclave_dim=dim)
        # ---- BISAGRA FASE 1: único origen de H ----
        H_eff = cartridge.lift_enclave_hamiltonian(dim)
        # Uniformización fucsiana del parámetro modular.
        tau_raw = complex(0.1, 1.2)
        try:
            tau_mod = reduce_to_poincare_fundamental_domain(tau_raw)
        except Exception:  # noqa: BLE001 — el motor puede no exponer la firma
            tau_mod = tau_raw
        jumps: List[ComplexMatrix] = []
        rho_p, report = engine.evolve_fuchsian_lindblad_manifold(
            rho_dream=base_rho,
            H_eff=H_eff,
            jump_operators=jumps,
            tau_modular=tau_mod,
            dt=0.08,
        )
        rho_p = BanachSpectralAlgebra.project_to_state_manifold(rho_p)
        enclave = NonCommutativeNoSignalingEnclave(hilbert_dim=dim)
        completeness_ok = enclave.verify_projector_completeness()
        is_isolated, leakage = enclave.verify_isolation_commutator(rho_p)
        hill_leak = enclave.hill_neck_leakage(rho_p, cartridge.invariant_tube)
        leakage = max(leakage, hill_leak if not cartridge.hill_neck_closed else leakage)
        strict_isolated = bool(
            is_isolated
            and cartridge.is_homologically_isolated
            and completeness_ok
            and cartridge.hill_neck_closed
        )
        purity = BanachSpectralAlgebra.purity(rho_p)
        entropy = BanachSpectralAlgebra.von_neumann_entropy(rho_p)
        rel_entropy = BanachSpectralAlgebra.quantum_relative_entropy(rho_p, base_rho)
        bures = BanachSpectralAlgebra.bures_distance(rho_p, base_rho)
        cstar = BanachSpectralAlgebra.cstar_residual(rho_p)
        dirichlet_energy = cls._dirichlet_energy(
            rho_p, H_eff, cartridge.circuit_dissipation_watts
        )
        strobo = cls._stroboscopic_poincare_section(base_rho, rho_p)
        evals_H = np.real(la.eigvalsh(H_eff))
        energy_span = float(np.max(evals_H) - np.min(evals_H)) if evals_H.size else 1.0
        tau_rec = BanachSpectralAlgebra.poincare_recurrence_time_bound(
            rho_p, energy_span, dim
        )
        escape_gamma = float(
            report.get("escape_rate_gamma", cartridge.lyapunov_spectral_radius)
            if isinstance(report, dict)
            else cartridge.lyapunov_spectral_radius
        )
        tau_reduced = tau_mod
        if isinstance(report, dict):
            fcert = report.get("fuchsian_certificate")
            if fcert is not None and hasattr(fcert, "tau_reduced"):
                tau_reduced = fcert.tau_reduced
        coherence = cls._metabolic_coherence_functional(
            purity, dirichlet_energy, rel_entropy, strict_isolated
        )
        heyting_verdict = cls._classify_metabolic_state(coherence)
        final_verdict = cartridge.topos_heyting_sieve.meet(heyting_verdict)
        if not cartridge.hill_neck_closed:
            final_verdict = final_verdict.meet(HeytingToposAlgebra.BOUNDARY_CRITICAL)
        return cls(
            source_cartridge=cartridge,
            rho_perturbed=rho_p,
            dirichlet_spectral_energy=dirichlet_energy,
            von_neumann_entropy=entropy,
            purity=purity,
            relative_entropy_to_base=rel_entropy,
            bures_distance_to_base=bures,
            cstar_residual=cstar,
            no_signaling_leakage_norm=leakage,
            projector_completeness_ok=completeness_ok,
            is_strictly_isolated=strict_isolated,
            jump_rate=0.05 + 0.02 * cartridge.invariant_tube.homoclinic_residue,
            mean_first_jump_time=float(max(tau_rec, 1e-6)),
            metabolic_coherence_index=coherence,
            topos_heyting_evaluation=final_verdict,
            poincare_recurrence_time=tau_rec,
            homoclinic_residue=cartridge.invariant_tube.homoclinic_residue,
            stroboscopic_bures_increment=strobo,
            fuchsian_tau_reduced=complex(tau_reduced),
            escape_rate_gamma=escape_gamma,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — VACUNA ESPECTRAL, MERKLE, RECURRENCIA, WAKE-SLEEP Y ORQUESTACIÓN
#
#   Continuación DIRECTA y ÚNICA de
#       MetabolizedPerturbationField.evolve_and_certify
#   de la FASE 2: el campo metabolizado (ρ_dream + invariantes de Poincaré)
#   alimenta la síntesis de vacuna, la certificación Merkle SHA-512, el
#   invariante de recurrencia y el ciclo wake-sleep del soberano
#   TOONOniricDreamerAgent.
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# FASE 3.1 — Árbol de Merkle SHA-512
# ─────────────────────────────────────────────────────────────────────────────
class MerkleTree:
    r"""
    Árbol de Merkle binario con hash SHA-512 y duplicación de nodos impares
    (convención Bitcoin/Ethereum: hoja impar se auto-empareja).

    Coste de verificación O(log n); colisión-resistencia bajo preimagen de
    SHA-512.  El digest de la raíz es el invariante integral discreto del
    lote onírico (análogo combinatorio del invariante integral de Poincaré,
    Vol. III, cap. XXII).
    """

    def __init__(self, leaves: Sequence[str]) -> None:
        if not leaves:
            raise ValueError("MerkleTree requiere ≥ 1 hoja.")
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
        r"""Reconstruye la raíz desde la hoja; HMAC-compare-digest anti-timing."""
        node = bytes.fromhex(self.leaf_hash)
        idx = self.index
        for sib_hex in self.siblings:
            sib = bytes.fromhex(sib_hex)
            if idx % 2 == 0:
                node = hashlib.sha512(node + sib).digest()
            else:
                node = hashlib.sha512(sib + node).digest()
            idx //= 2
        return hmac.compare_digest(node.hex(), self.root)


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3.2 — Certificados oníricos, Poincaré y recurrencia
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class OniricScenarioCertificate:
    r"""Certificado forense de un sueño contrafactual REM."""

    dream_id: str
    dreamer_agent_id: str
    cartridge_id: str
    scenario_type: str
    heyting_verdict: HeytingToposAlgebra
    dirichlet_energy: float
    purity: float
    entropy: float
    quantum_fidelity: float
    bures_distance: float
    cstar_residual: float
    spectral_coverage_mass: float
    projector_idempotency_residual: float
    isolation_guarantee: bool
    no_signaling_leakage: float
    jump_rate: float
    mean_first_jump_time: float
    immune_vaccine_effective: bool
    learning_rate_applied: float
    merkle_sha512_provenance: str
    timestamp_utc: float
    fuchsian_tau_reduced: Optional[complex] = None
    umegaki_divergence: float = 0.0
    poincare_jacobi_constant: float = 0.0
    lagrange_gateway_index: int = 0
    hill_neck_closed: bool = False
    poincare_recurrence_time: float = 0.0
    homoclinic_residue: float = 0.0

    def is_vetoed(self) -> bool:
        r"""Veredicto booleano: ⊥ significa veto absoluto."""
        return self.heyting_verdict == HeytingToposAlgebra.VETOED_ABSURDUM

    def is_immune(self) -> bool:
        r"""Criterio de inmunidad efectiva: no veto + aislamiento + vacuna."""
        return (
            not self.is_vetoed()
            and self.isolation_guarantee
            and self.immune_vaccine_effective
        )

    def is_hill_isolated(self) -> bool:
        r"""Aislamiento celestial: cuello de Hill cerrado y leakage nulo."""
        return self.hill_neck_closed and self.isolation_guarantee


@dataclass(frozen=True, slots=True)
class PoincareOniricScenarioCertificate:
    r"""Certificado Poincaré específico: navegación de tubos de Lagrange en ℱ."""

    scenario_id: str
    dream_state_isolated: bool
    fuchsian_tau_reduced: complex
    umegaki_divergence: float
    bures_geodesic_distance: float
    escape_rate_gamma: float
    vaccine_coverage_ratio: float
    heyting_verdict_omega4: int
    merkle_sha512_root: str
    jacobi_constant: float = 0.0
    lagrange_index: int = 1
    hill_neck_closed: bool = False
    recurrence_time: float = 0.0

    def is_coherent(self) -> bool:
        r"""Coherencia a nivel Ω₄ (veredicto = 3 ⇒ ⊤)."""
        return self.heyting_verdict_omega4 == 3


@dataclass(frozen=True, slots=True)
class PoincareRecurrenceInvariant:
    r"""
    Invariante integral de recurrencia (Vol. III, cap. XXVI) asociado a un
    lote wake-sleep.  Si el flujo Φ_t preserva μ y μ(ℋ) < ∞, entonces
    μ-casi todo estado retorna a todo entorno; el tiempo medio de primer
    retorno es la cota de Kac τ_Kac = μ(X) / μ(A).
    """

    mean_recurrence_time: float
    max_homoclinic_residue: float
    fraction_neck_closed: float
    kac_bound: float
    measure_preserving_residual: float

    def is_recurrent(self) -> bool:
        r"""Recurrencia efectiva: cota de Kac finita y cuellos mayormente cerrados."""
        return (
            math.isfinite(self.kac_bound)
            and self.fraction_neck_closed >= 0.5
            and self.measure_preserving_residual < 1e-6
        )


@dataclass(frozen=True, slots=True)
class MacroWakeSleepAuditReport:
    r"""Reporte consolidado del ciclo REM con transición de fase wake-sleep."""

    audit_cycle_id: str
    total_scenarios_simulated: int
    vaccines_absorbed_count: int
    wake_entropy: float
    sleep_entropy: float
    entropy_reduction_ratio: float
    overall_topos_verdict: HeytingToposAlgebra
    merkle_leaf_count: int
    merkle_root_hash: str
    merkle_proofs_ok: bool
    execution_duration_ms: float
    recurrence_invariant: Optional[PoincareRecurrenceInvariant] = None

    def is_energy_favourable(self) -> bool:
        r"""Criterio físico: F_wake − F_sleep ≥ 0 ⇔ S_wake ≥ S_sleep."""
        return self.wake_entropy >= self.sleep_entropy

    def is_topos_coherent(self) -> bool:
        r"""Consistencia topológica: veredicto global = ⊤."""
        return self.overall_topos_verdict == HeytingToposAlgebra.VERUM_COHERENT


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3.3 — Vacuna espectral (consumidor de evolve_and_certify)
# ─────────────────────────────────────────────────────────────────────────────
class SpectralImmuneVaccineEngine:
    r"""
    Proyector de inmunidad P_vac ∈ B(ℋ) sobre el subespacio de cobertura de
    masa espectral μ ∈ (0, 1].

    Dada ρ = Σ λ_i |v_i⟩⟨v_i| con λ_1 ≥ λ_2 ≥ …, se retiene el mínimo k tal que
        Σ_{i≤k} λ_i / Tr(ρ) ≥ μ.
    P_vac = Σ_{i≤k} |v_i⟩⟨v_i|  satisface P_vac² = P_vac y Tr(P_vac ρ) = masa.

    Inspiración celestial: P_vac es la sección transversal que intercepta
    el tubo W^s(γ_{Lₖ}) — vacuna = taponamiento del canal heteroclínico.
    """

    COVERAGE_FRACTION: Final[float] = 0.75
    RANK_FRACTION_CAP: Final[float] = 0.75
    MASS_FLOOR: Final[float] = 1e-15

    @classmethod
    def construct_from_metabolized_field(
        cls,
        perturbed_state: MetabolizedPerturbationField,
        coverage_fraction: float = COVERAGE_FRACTION,
    ) -> Tuple[ComplexMatrix, bool, float, float]:
        r"""
        Atajo desde un campo metabolizado.

        CONTRATO DE BISAGRA FASE 2 → FASE 3: este método es el primer
        consumidor autorizado de `evolve_and_certify`.
        """
        return cls.construct_spectral_vaccine(perturbed_state, coverage_fraction)

    @classmethod
    def construct_spectral_vaccine(
        cls,
        perturbed_state: MetabolizedPerturbationField,
        coverage_fraction: float = COVERAGE_FRACTION,
    ) -> Tuple[ComplexMatrix, bool, float, float]:
        r"""
        Retorna (P_vac, es_efectiva, masa_retenida, residuo_idempotencia).

        Efectividad simultánea: veredicto ≠ ⊥, aislamiento estricto del
        enclave (cuello de Hill cerrado), masa ≥ cobertura, rango ≤ ⌈0.75 n⌉
        y residuo homoclínico bajo (tubo taponado).
        """
        rho = perturbed_state.rho_perturbed
        dim = rho.shape[0]
        if perturbed_state.topos_heyting_evaluation == HeytingToposAlgebra.VETOED_ABSURDUM:
            eye = np.eye(dim, dtype=np.complex128) / dim
            return eye, False, 0.0, 0.0
        eigvals, eigvecs = la.eigh(rho)
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
        tube_capped = perturbed_state.homoclinic_residue < 0.35
        is_effective = bool(
            retained_mass >= coverage_fraction
            and perturbed_state.is_strictly_isolated
            and k <= math.ceil(dim * cls.RANK_FRACTION_CAP)
            and tube_capped
        )
        return P_vac, is_effective, retained_mass, idem_res


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3.4 — Soberano orquestador onírico
# ─────────────────────────────────────────────────────────────────────────────
class TOONOniricDreamerAgent:
    r"""
    SOBERANO AGENTE SIMULADOR ONÍRICO (FASE REM).

    Autoridad terminal del bucle GAN-REM. Coordina, en orden canónico:

        1. `generate_synthetic_cartridge`          (FASE 1: objeto de Cart),
        2. `MetabolizedPerturbationField.evolve_and_certify`
           → consume `lift_enclave_hamiltonian`    (FASE 1 → 2),
        3. `SpectralImmuneVaccineEngine.construct_from_metabolized_field`
           → consume el campo metabolizado         (FASE 2 → 3),
        4. Φ_η(ρ_MAC) afín, Merkle SHA-512, pasaporte.

    El aprendizaje es modulado por Ω₄ vía `_HEYTING_LEARNING_RATE_MAP` y
    atenuado si la energía de Dirichlet supera el umbral o si el residuo
    homoclínico indica un tubo de escape abierto.
    """

    _HEYTING_LEARNING_RATE_MAP: Final[Dict[int, float]] = {
        int(HeytingToposAlgebra.VERUM_COHERENT): 0.30,
        int(HeytingToposAlgebra.TOPOLOGICAL_STABLE): 0.15,
        int(HeytingToposAlgebra.BOUNDARY_CRITICAL): 0.05,
        int(HeytingToposAlgebra.VETOED_ABSURDUM): 0.0,
    }

    def __init__(
        self,
        agent_id: str = "TOON-DREAMER-SABIO-01",
        dimension_mac: int = 4,
        energy_threshold: float = 18.0,
        seed: int = 777,
        enclave_engine: Optional[NonHermitianLindbladMasterEngine] = None,
        crtbp_mu: float = PoincareCRTBPPhaseSpace.MU_EARTH_MOON,
    ) -> None:
        if dimension_mac < 2:
            raise ValueError("dimension_mac ≥ 2 (partición dream/physical).")
        if energy_threshold <= 0.0:
            raise ValueError("energy_threshold debe ser > 0.")
        self.agent_id = agent_id
        self.dimension_mac = int(dimension_mac)
        self.energy_threshold = float(energy_threshold)
        self.seed_counter = int(seed)
        self.dream_count = 0
        self.crtbp_mu = float(crtbp_mu)
        self.base_rho: ComplexMatrix = (
            np.eye(dimension_mac, dtype=np.complex128) / dimension_mac
        )
        self.dream_certificates_history: List[OniricScenarioCertificate] = []
        self.engine = (
            enclave_engine
            if enclave_engine is not None
            else NonHermitianLindbladMasterEngine(enclave_dim=dimension_mac)
        )

    def compute_umegaki_and_bures_metrics(
        self,
        rho_dream: ComplexMatrix,
        rho_base: ComplexMatrix,
    ) -> Tuple[float, float]:
        r"""Divergencia de Umegaki y distancia geodésica de Bures–Wasserstein."""
        umegaki = BanachSpectralAlgebra.quantum_relative_entropy(rho_dream, rho_base)
        bures = BanachSpectralAlgebra.bures_distance(rho_dream, rho_base)
        return max(0.0, umegaki), bures

    def synthesize_spectral_vaccine_projection(
        self,
        rho_dream: ComplexMatrix,
        coverage_target: float = 0.90,
    ) -> Tuple[ComplexMatrix, float]:
        r"""P_vac = Σ_{i≤k} |v_i⟩⟨v_i| cubriendo la masa espectral μ objetivo."""
        evals, evecs = la.eigh(rho_dream)
        idx = np.argsort(evals)[::-1]
        evals_sorted = np.clip(evals[idx], 0.0, None)
        evecs_sorted = evecs[:, idx]
        total_mass = float(np.sum(evals_sorted))
        if total_mass <= 1e-15:
            dim = rho_dream.shape[0]
            return np.eye(dim, dtype=np.complex128) / dim, 0.0
        cum_mass = np.cumsum(evals_sorted) / total_mass
        k = int(np.searchsorted(cum_mass, coverage_target)) + 1
        k = min(k, len(evals))
        P_vac = np.zeros_like(rho_dream, dtype=np.complex128)
        for i in range(k):
            v_i = evecs_sorted[:, i:i + 1]
            P_vac += v_i @ v_i.conj().T
        coverage_achieved = float(cum_mass[k - 1])
        return P_vac, coverage_achieved

    def run_fuchsian_counterfactual_simulation(
        self,
        scenario_id: str,
        rho_base: ComplexMatrix,
        H_eff: ComplexMatrix,
        jump_ops: Sequence[ComplexMatrix],
        tau_modular: complex,
        dt: float = 0.01,
        is_dream_state: bool = True,
    ) -> PoincareOniricScenarioCertificate:
        r"""
        Simulación de cisnes negros navegando los tubos de Lagrange en ℱ ⊂ ℍ²,
        sin invocar la maquinaria completa de cartucho (vía corta de auditoría).
        """
        if not is_dream_state:
            return PoincareOniricScenarioCertificate(
                scenario_id=scenario_id,
                dream_state_isolated=False,
                fuchsian_tau_reduced=tau_modular,
                umegaki_divergence=float("inf"),
                bures_geodesic_distance=float("inf"),
                escape_rate_gamma=float("inf"),
                vaccine_coverage_ratio=0.0,
                heyting_verdict_omega4=0,
                merkle_sha512_root="0" * 128,
            )
        try:
            tau_mod = reduce_to_poincare_fundamental_domain(tau_modular)
        except Exception:  # noqa: BLE001
            tau_mod = tau_modular
        rho_dream, report = self.engine.evolve_fuchsian_lindblad_manifold(
            rho_dream=rho_base.copy(),
            H_eff=H_eff,
            jump_operators=jump_ops,
            tau_modular=tau_mod,
            dt=dt,
        )
        rho_dream = BanachSpectralAlgebra.project_to_state_manifold(rho_dream)
        umegaki, bures = self.compute_umegaki_and_bures_metrics(rho_dream, rho_base)
        _, coverage = self.synthesize_spectral_vaccine_projection(rho_dream)
        escape = float(
            report["escape_rate_gamma"] if isinstance(report, dict) else 0.0
        )
        if bures > 1.2 or escape > 5.0:
            verdict = 0
        elif bures > 0.6:
            verdict = 1
        elif bures > 0.2:
            verdict = 2
        else:
            verdict = 3
        hasher = hashlib.sha512()
        hasher.update(scenario_id.encode("utf-8"))
        hasher.update(rho_dream.tobytes())
        hasher.update(str(verdict).encode("utf-8"))
        merkle_root = hasher.hexdigest()
        tau_reduced = tau_mod
        if isinstance(report, dict):
            fcert = report.get("fuchsian_certificate")
            if fcert is not None and hasattr(fcert, "tau_reduced"):
                tau_reduced = fcert.tau_reduced
        crtbp = PoincareCRTBPPhaseSpace(mu=self.crtbp_mu)
        evals_H = np.real(la.eigvalsh(0.5 * (H_eff + H_eff.conj().T)))
        span = float(np.max(evals_H) - np.min(evals_H)) if evals_H.size else 1.0
        tau_rec = BanachSpectralAlgebra.poincare_recurrence_time_bound(
            rho_dream, span, rho_dream.shape[0]
        )
        # Hoja de Jacobi de referencia (sueño aislado ⇒ cuello cerrado).
        jacobi_C = crtbp.jacobi_at_lagrange(1) + 0.2 * (1.0 - min(bures, 1.0))
        return PoincareOniricScenarioCertificate(
            scenario_id=scenario_id,
            dream_state_isolated=True,
            fuchsian_tau_reduced=complex(tau_reduced),
            umegaki_divergence=umegaki,
            bures_geodesic_distance=bures,
            escape_rate_gamma=escape,
            vaccine_coverage_ratio=coverage,
            heyting_verdict_omega4=verdict,
            merkle_sha512_root=merkle_root,
            jacobi_constant=jacobi_C,
            lagrange_index=crtbp.classify_lagrange_gateway(jacobi_C),
            hill_neck_closed=crtbp.hill_neck_is_closed(jacobi_C),
            recurrence_time=tau_rec,
        )

    @staticmethod
    def _deterministic_topology_perturbation(
        scenario_type: str, num_vertices: int
    ) -> Optional[Tuple[int, int]]:
        r"""Arista adicional determinista desde el hash del escenario."""
        digest = hashlib.sha256(scenario_type.encode("utf-8")).digest()
        if digest[0] % 2 != 0:
            return None
        u, v = digest[1] % num_vertices, digest[2] % num_vertices
        return None if u == v else (u, v)

    def generate_synthetic_cartridge(
        self,
        scenario_type: str,
        cost_delta_ratio: float,
        synthetic_betti_1: int = 0,
        is_dream_state: bool = True,
    ) -> CategoricalCircuitCartridge:
        r"""Sintetiza un cartucho adversarial CRTBP-inmerso; incrementa `dream_count`."""
        self.dream_count += 1
        cartridge_id = f"SYNTH-TOON-DREAM-{self.dream_count:05d}"
        payload = {
            "synthetic_apu_code": "APU-SIM-9999",
            "material_variance_pct": cost_delta_ratio * 100.0,
            "simulated_labor_strike_prob": min(1.0, max(0.0, cost_delta_ratio * 0.6)),
            "simulated_supply_chain_latency_days": int(abs(cost_delta_ratio) * 30),
            "isolation_token": f"DREAM_STATE_HOMOLOGICAL_LOCK_{self.dream_count:04d}",
            "crtbp_mu": self.crtbp_mu,
        }
        extra_edge = self._deterministic_topology_perturbation(scenario_type, 6)
        return CategoricalCircuitCartridge.synthesize_cartridge(
            cartridge_id=cartridge_id,
            scenario_type=scenario_type,
            cost_delta_ratio=cost_delta_ratio,
            synthetic_betti_1=synthetic_betti_1,
            is_dream_state=is_dream_state,
            payload=payload,
            deterministic_extra_edge=extra_edge,
            mu=self.crtbp_mu,
        )

    def _adaptive_learning_rate(
        self,
        verdict: HeytingToposAlgebra,
        dirichlet_energy: float,
        homoclinic_residue: float = 0.0,
    ) -> float:
        r"""
        η adaptativo: η_base(Ω₄) atenuado × 0.25 si E_D supera el umbral,
        y × 0.5 adicional si el residuo homoclínico indica tubo abierto.
        """
        eta = self._HEYTING_LEARNING_RATE_MAP[int(verdict)]
        if dirichlet_energy > self.energy_threshold:
            eta *= 0.25
        if homoclinic_residue > 0.35:
            eta *= 0.5
        return float(eta)

    def dream_scenario(
        self,
        scenario_type: str,
        cost_delta_ratio: float,
        synthetic_betti_1: int = 0,
        force_isolation_breach: bool = False,
    ) -> OniricScenarioCertificate:
        r"""
        Ciclo onírico completo (tres fases anidadas en un único morfismo):

            1. Sintetiza el cartucho adversarial (FASE 1, objeto de Cart).
            2. Metaboliza vía `evolve_and_certify`, que internamente invoca
               `lift_enclave_hamiltonian` (BISAGRA 1→2).
            3. Sintetiza la vacuna espectral (BISAGRA 2→3) y actualiza ρ_MAC
               si η > 0.  Sella procedencia SHA-512.

        Un 'sueño de bancarrota' con cuello de Hill abierto no actualiza ρ_MAC
        (postulado 1: aislamiento homológico).
        """
        start_time = time.time()
        self.seed_counter += 1
        is_dream_state = not force_isolation_breach
        cartridge = self.generate_synthetic_cartridge(
            scenario_type=scenario_type,
            cost_delta_ratio=cost_delta_ratio,
            synthetic_betti_1=synthetic_betti_1,
            is_dream_state=is_dream_state,
        )
        # ---- BISAGRA FASE 2: único origen del campo metabolizado ----
        metabolized_field = MetabolizedPerturbationField.evolve_and_certify(
            cartridge=cartridge, base_rho=self.base_rho
        )
        # ---- BISAGRA FASE 3: único origen de P_vac ----
        P_vac, vaccine_effective, coverage_mass, idem_res = (
            SpectralImmuneVaccineEngine.construct_from_metabolized_field(
                metabolized_field
            )
        )
        learning_rate = self._adaptive_learning_rate(
            metabolized_field.topos_heyting_evaluation,
            metabolized_field.dirichlet_spectral_energy,
            metabolized_field.homoclinic_residue,
        )
        if learning_rate > 0.0 and vaccine_effective:
            vaccinated_rho = P_vac @ metabolized_field.rho_perturbed @ P_vac.conj().T
            trace_v = float(np.real(np.trace(vaccinated_rho)))
            if trace_v > 1e-12:
                vaccinated_rho = vaccinated_rho / trace_v
            self.base_rho = (
                (1.0 - learning_rate) * self.base_rho
                + learning_rate * vaccinated_rho
            )
            self.base_rho = BanachSpectralAlgebra.project_to_state_manifold(self.base_rho)
        quantum_fidelity = BanachSpectralAlgebra.quantum_fidelity(
            self.base_rho, metabolized_field.rho_perturbed
        )
        umegaki, bures = self.compute_umegaki_and_bures_metrics(
            metabolized_field.rho_perturbed, self.base_rho
        )
        hasher = hashlib.sha512()
        signature_payload = (
            f"{self.agent_id}::{cartridge.cartridge_id}::{scenario_type}::"
            f"{metabolized_field.topos_heyting_evaluation.name}::"
            f"{metabolized_field.dirichlet_spectral_energy:.8f}::"
            f"{metabolized_field.purity:.8f}::{metabolized_field.is_strictly_isolated}::"
            f"{coverage_mass:.6f}::{cartridge.poincare_jacobi_constant:.8f}::{start_time}"
        ).encode("utf-8")
        hasher.update(signature_payload)
        merkle_provenance = hasher.hexdigest()
        cert = OniricScenarioCertificate(
            dream_id=cartridge.cartridge_id,
            dreamer_agent_id=self.agent_id,
            cartridge_id=cartridge.cartridge_id,
            scenario_type=scenario_type,
            heyting_verdict=metabolized_field.topos_heyting_evaluation,
            dirichlet_energy=metabolized_field.dirichlet_spectral_energy,
            purity=metabolized_field.purity,
            entropy=metabolized_field.von_neumann_entropy,
            quantum_fidelity=quantum_fidelity,
            bures_distance=bures,
            cstar_residual=metabolized_field.cstar_residual,
            spectral_coverage_mass=coverage_mass,
            projector_idempotency_residual=idem_res,
            isolation_guarantee=metabolized_field.is_strictly_isolated,
            no_signaling_leakage=metabolized_field.no_signaling_leakage_norm,
            jump_rate=metabolized_field.jump_rate,
            mean_first_jump_time=metabolized_field.mean_first_jump_time,
            immune_vaccine_effective=vaccine_effective,
            learning_rate_applied=learning_rate,
            merkle_sha512_provenance=merkle_provenance,
            timestamp_utc=start_time,
            fuchsian_tau_reduced=metabolized_field.fuchsian_tau_reduced,
            umegaki_divergence=umegaki,
            poincare_jacobi_constant=cartridge.poincare_jacobi_constant,
            lagrange_gateway_index=cartridge.lagrange_gateway_index,
            hill_neck_closed=cartridge.hill_neck_closed,
            poincare_recurrence_time=metabolized_field.poincare_recurrence_time,
            homoclinic_residue=metabolized_field.homoclinic_residue,
        )
        self.dream_certificates_history.append(cert)
        logger.info(
            "Sueño REM #%04d [%s] ➔ Ω₄: %s | E_D: %.4f | C_J: %.4f | L%d | "
            "Hill: %s | τ_rec: %.3f | Cobertura: %.3f | η: %.2f | "
            "VacunaOK: %s | Isol: %s | Hom: %.3e | Hash: %s…",
            self.dream_count,
            scenario_type,
            cert.heyting_verdict.name,
            cert.dirichlet_energy,
            cert.poincare_jacobi_constant,
            cert.lagrange_gateway_index,
            cert.hill_neck_closed,
            cert.poincare_recurrence_time,
            coverage_mass,
            learning_rate,
            cert.immune_vaccine_effective,
            cert.isolation_guarantee,
            cert.homoclinic_residue,
            merkle_provenance[:16],
        )
        return cert

    @staticmethod
    def _merkle_tree_root(leaf_hashes: Sequence[str]) -> str:
        r"""Raíz Merkle SHA-512 sobre las hojas."""
        if not leaf_hashes:
            return hashlib.sha512(b"EMPTY_MERKLE_ROOT").hexdigest()
        return MerkleTree(leaf_hashes).root

    @staticmethod
    def _merkle_proof(
        leaf_hashes: Sequence[str], index: int
    ) -> MerkleInclusionProof:
        r"""Prueba de inclusión de la hoja `index`."""
        if not leaf_hashes:
            empty = hashlib.sha512(b"EMPTY_MERKLE_ROOT").hexdigest()
            return MerkleInclusionProof(empty, tuple(), 0, empty)
        return MerkleTree(leaf_hashes).proof(index)

    def _recurrence_invariant_from_batch(
        self, batch_certs: Sequence[OniricScenarioCertificate]
    ) -> PoincareRecurrenceInvariant:
        r"""Invariante integral de Poincaré–Kac del lote wake-sleep."""
        if not batch_certs:
            return PoincareRecurrenceInvariant(
                mean_recurrence_time=0.0,
                max_homoclinic_residue=0.0,
                fraction_neck_closed=1.0,
                kac_bound=0.0,
                measure_preserving_residual=0.0,
            )
        rec = [c.poincare_recurrence_time for c in batch_certs]
        hom = [c.homoclinic_residue for c in batch_certs]
        closed = [1.0 if c.hill_neck_closed else 0.0 for c in batch_certs]
        mean_rec = float(np.mean(rec))
        frac = float(np.mean(closed))
        # Cota de Kac: μ(X)/μ(A) ≈ 1 / fracción de cuellos cerrados.
        kac = mean_rec / max(frac, 1e-6)
        # Residuo de preservación de medida: variación relativa de tr(ρ)=1
        # se certifica por cstar medio (proxy de isometría C*).
        meas = float(np.mean([c.cstar_residual for c in batch_certs]))
        return PoincareRecurrenceInvariant(
            mean_recurrence_time=mean_rec,
            max_homoclinic_residue=float(np.max(hom)),
            fraction_neck_closed=frac,
            kac_bound=kac,
            measure_preserving_residual=meas,
        )

    def execute_macro_wake_sleep_audit(
        self, batch_scenarios: List[Tuple[str, float, int]]
    ) -> MacroWakeSleepAuditReport:
        r"""
        Batch de escenarios (fase REM) consolidado:

            • veredicto global ínfimo en Ω₄,
            • raíz Merkle SHA-512 y verificación de pruebas,
            • balance entrópico wake → sleep,
            • invariante de recurrencia de Poincaré–Kac,
            • tiempo total de ejecución.
        """
        t0 = time.time()
        initial_entropy = BanachSpectralAlgebra.von_neumann_entropy(self.base_rho)
        vaccines_absorbed = 0
        cumulative_topos = HeytingToposAlgebra.VERUM_COHERENT
        batch_leaf_hashes: List[str] = []
        batch_certs: List[OniricScenarioCertificate] = []
        for sc_type, cost_delta, betti_1 in batch_scenarios:
            cert = self.dream_scenario(
                scenario_type=sc_type,
                cost_delta_ratio=cost_delta,
                synthetic_betti_1=betti_1,
            )
            cumulative_topos = cumulative_topos.meet(cert.heyting_verdict)
            batch_leaf_hashes.append(cert.merkle_sha512_provenance)
            batch_certs.append(cert)
            if cert.immune_vaccine_effective:
                vaccines_absorbed += 1
        final_entropy = BanachSpectralAlgebra.von_neumann_entropy(self.base_rho)
        entropy_ratio = float(
            (initial_entropy - final_entropy) / (initial_entropy + 1e-12)
        )
        duration = (time.time() - t0) * 1000.0
        merkle_root = self._merkle_tree_root(batch_leaf_hashes)
        proofs_ok = True
        for i in range(len(batch_leaf_hashes)):
            proof = self._merkle_proof(batch_leaf_hashes, i)
            if proof.root != merkle_root or not proof.verify():
                proofs_ok = False
                break
        context_binder = hashlib.sha512()
        context_binder.update(
            f"AUDIT-CONTEXT::{self.agent_id}::{merkle_root}::{initial_entropy:.8f}::"
            f"{final_entropy:.8f}::{vaccines_absorbed}::{t0}".encode("utf-8")
        )
        root_hash = context_binder.hexdigest()
        rec_inv = self._recurrence_invariant_from_batch(batch_certs)
        return MacroWakeSleepAuditReport(
            audit_cycle_id=f"AUDIT-REM-{self.dream_count:05d}",
            total_scenarios_simulated=len(batch_scenarios),
            vaccines_absorbed_count=vaccines_absorbed,
            wake_entropy=initial_entropy,
            sleep_entropy=final_entropy,
            entropy_reduction_ratio=entropy_ratio,
            overall_topos_verdict=cumulative_topos,
            merkle_leaf_count=len(batch_leaf_hashes),
            merkle_root_hash=root_hash,
            merkle_proofs_ok=proofs_ok,
            execution_duration_ms=duration,
            recurrence_invariant=rec_inv,
        )

    def audit_registry(self) -> Dict[str, Any]:
        r"""Resumen cuantitativo del registro de certificados oníricos."""
        n = len(self.dream_certificates_history)
        empty_dist = {v.name: 0 for v in HeytingToposAlgebra}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingToposAlgebra.VERUM_COHERENT.name,
                "avg_purity": 0.0,
                "avg_coverage_mass": 0.0,
                "avg_dirichlet_energy": 0.0,
                "avg_jacobi_C": 0.0,
                "fraction_hill_closed": 1.0,
                "n_immune": 0,
                "n_vaccines_effective": 0,
                "registry_integrity_ok": True,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingToposAlgebra}
        s_p = s_c = s_d = s_j = 0.0
        n_imm = n_vac = n_hill = 0
        hashes: Set[str] = set()
        collide = False
        for c in self.dream_certificates_history:
            dist[c.heyting_verdict.name] += 1
            s_p += c.purity
            s_c += c.spectral_coverage_mass
            s_d += c.dirichlet_energy
            s_j += c.poincare_jacobi_constant
            if c.hill_neck_closed:
                n_hill += 1
            if c.is_immune():
                n_imm += 1
            if c.immune_vaccine_effective:
                n_vac += 1
            if c.merkle_sha512_provenance in hashes:
                collide = True
            hashes.add(c.merkle_sha512_provenance)
        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "avg_purity": s_p * inv,
            "avg_coverage_mass": s_c * inv,
            "avg_dirichlet_energy": s_d * inv,
            "avg_jacobi_C": s_j * inv,
            "fraction_hill_closed": n_hill * inv,
            "n_immune": n_imm,
            "n_vaccines_effective": n_vac,
            "registry_integrity_ok": not collide,
        }

    def emit_dreamer_passport(self) -> Dict[str, Any]:
        r"""Pasaporte criptográfico consolidado del motor onírico."""
        h = hashlib.sha512()
        h.update(f"{self.agent_id}::{self.dream_count}::{self.crtbp_mu:.12f}".encode("utf-8"))
        for c in self.dream_certificates_history:
            h.update(c.merkle_sha512_provenance.encode("utf-8"))
        return {
            "agent_id": self.agent_id,
            "registry_size": self.dream_count,
            "global_verdict": self.global_verdict.name,
            "n_immune": sum(1 for c in self.dream_certificates_history if c.is_immune()),
            "crtbp_mu": self.crtbp_mu,
            "evidence_hash": h.hexdigest(),
            "module_version": __version__,
        }

    @property
    def registry_view(self) -> Tuple[OniricScenarioCertificate, ...]:
        r"""Vista inmutable del registro onírico."""
        return tuple(self.dream_certificates_history)

    @property
    def global_verdict(self) -> HeytingToposAlgebra:
        r"""Veredicto global: ínfimo Ω₄ de todos los ciclos registrados."""
        gv = HeytingToposAlgebra.VERUM_COHERENT
        for c in self.dream_certificates_history:
            gv = gv.meet(c.heyting_verdict)
        return gv


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN RIGUROSA Y AUDITORÍA UNITARIA END-TO-END
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("╔" + "═" * 86 + "╗")
    print("║   DEMOSTRACIÓN DOCTORAL DEL TOON ONIRIC DREAMER AGENT v10.0            ║")
    print("║  POINCARÉ CRTBP + HILL + LINDSTEDT + LAWVERE–TIERNEY + GKSL + MERKLE   ║")
    print("╚" + "═" * 86 + "╝")

    # ── Verificaciones algebraicas (FASE 1) ────────────────────────────────
    assert HeytingToposAlgebra.verify_residuation_axiom(), "Ley de residuación falla."
    assert (
        HeytingToposAlgebra.verify_lawvere_tierney_topology_axioms()
    ), "Axiomas Lawvere–Tierney fallan."
    print("\n[FASE 1] Álgebra de Heyting Ω₄: residuación y Lawvere–Tierney OK.")

    q1 = BiquaternionClifford(0.3, 0.1, 0.5, -0.2, 0.2, 0.1, -0.4, 0.3)
    q2 = BiquaternionClifford(0.2, -0.1, 0.4, 0.2, -0.3, 0.1, 0.5, -0.2)
    q3 = BiquaternionClifford(-0.1, 0.2, 0.3, -0.3, 0.4, 0.1, -0.2, 0.2)
    assoc = q1.associativity_residual(q2, q3)
    mult = q1.reduced_norm_multiplicativity_residual(q2)
    inv_res = float(la.norm((q1 * q1.inverse()).to_matrix() - BiquaternionClifford.identity().to_matrix()))
    print(
        f"  · Bicuaterniones: ‖asoc‖={assoc:.3e} | "
        f"N(q1q2)−N₁N₂={mult:.3e} | ‖q q⁻¹−1‖={inv_res:.3e}"
    )
    assert assoc < 1e-10 and mult < 1e-10 and inv_res < 1e-10

    graph_probe = SimplicialHodgeGraph(
        num_vertices=5,
        edges=[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)],
        faces=[(0, 1, 2), (2, 3, 4)],
    )
    assert graph_probe.verify_mckean_singer_index(t=1.0), "Índice McKean–Singer falla."
    b0, b1, b2 = graph_probe.compute_betti_numbers()
    print(
        f"\n[FASE 1] Hodge simplicial: β=({b0},{b1},{b2}) | "
        f"χ={graph_probe.euler_characteristic()} | "
        f"δ_PD={graph_probe.poincare_duality_residual()} | McKean–Singer OK"
    )

    # CRTBP Earth–Moon
    crtbp = PoincareCRTBPPhaseSpace()
    L4 = crtbp.lagrange_point(4)
    c1 = crtbp.jacobi_at_lagrange(1)
    c4 = crtbp.jacobi_at_lagrange(4)
    symp = crtbp.symplectic_spectrum_residual(1)
    print(
        f"\n[FASE 1] CRTBP Earth–Moon: μ={crtbp.mu:.6f} | "
        f"L₄=({L4[0]:.4f},{L4[1]:.4f}) | C_L1={c1:.4f} | C_L4={c4:.4f}"
    )
    print(
        f"  · Routh estable: {crtbp.routh_stability()} | "
        f"residuo simpléctico L₁={symp:.3e} | "
        f"cuello C=C_L1+0.1 cerrado: {crtbp.hill_neck_is_closed(c1 + 0.1)}"
    )
    assert crtbp.routh_stability()
    assert symp < 1e-6
    assert crtbp.hill_neck_is_closed(c1 + 0.1)
    assert not crtbp.hill_neck_is_closed(c1 - 0.5)

    # ── FASE 2: motor soberano ─────────────────────────────────────────────
    dreamer_agent = TOONOniricDreamerAgent(
        agent_id="TOON-DREAMER-SABIO-01",
        dimension_mac=4,
        energy_threshold=18.0,
        seed=10101,
    )
    cert_fuchsian = dreamer_agent.run_fuchsian_counterfactual_simulation(
        scenario_id="TEST_CRTBP_LAGRANGE_L1",
        rho_base=dreamer_agent.base_rho,
        H_eff=np.eye(4, dtype=np.complex128),
        jump_ops=[],
        tau_modular=complex(0.2, 1.3),
        dt=0.01,
        is_dream_state=True,
    )
    print(
        f"\n[FASE 2] Simulación Fucsiana: {cert_fuchsian.scenario_id} | "
        f"Ω₄={cert_fuchsian.heyting_verdict_omega4} | "
        f"D_Umegaki={cert_fuchsian.umegaki_divergence:.4f} | "
        f"C_J={cert_fuchsian.jacobi_constant:.4f} | "
        f"Hill={cert_fuchsian.hill_neck_closed}"
    )
    assert cert_fuchsian.dream_state_isolated

    # ── FASE 3: ciclos oníricos y wake-sleep ───────────────────────────────
    cert1 = dreamer_agent.dream_scenario(
        scenario_type="BLACK_SWAN_STEEL_SPIKE",
        cost_delta_ratio=0.35,
        synthetic_betti_1=0,
    )
    print(
        f"\n[FASE 3] Sueño REM: {cert1.dream_id} | "
        f"Ω₄: {cert1.heyting_verdict.name} | "
        f"D(ρ‖ρ₀): {cert1.umegaki_divergence:.4f} | "
        f"C_J: {cert1.poincare_jacobi_constant:.4f} | "
        f"L{cert1.lagrange_gateway_index} | "
        f"Inmunidad: {cert1.is_immune()}"
    )

    batch_scenarios = [
        ("STEEL_CARTEL_PRICE_SHOCK", 0.35, 0),
        ("STRIKE_LABOR_PARALYSIS", 0.65, 1),
        ("HYDROLOGIC_FLOOD_FOUNDATION", 0.25, 0),
        ("CIRCULAR_SUB_BILLING_ATTACK", 0.85, 2),
    ]
    audit_report = dreamer_agent.execute_macro_wake_sleep_audit(batch_scenarios)
    print("\n[FASE 3] Auditoría Wake-Sleep consolidada:")
    print(f"  · Ciclo ID                : {audit_report.audit_cycle_id}")
    print(f"  · Escenarios simulados    : {audit_report.total_scenarios_simulated}")
    print(f"  · Vacunas absorbidas      : {audit_report.vaccines_absorbed_count}")
    print(f"  · Entropía wake (bits)    : {audit_report.wake_entropy:.4f}")
    print(f"  · Entropía sleep (bits)   : {audit_report.sleep_entropy:.4f}")
    print(f"  · Reducción entrópica     : {audit_report.entropy_reduction_ratio:+.4f}")
    print(f"  · Ω₄ global               : {audit_report.overall_topos_verdict.name}")
    print(f"  · Merkle root (SHA-512)   : {audit_report.merkle_root_hash[:48]}…")
    print(f"  · Pruebas Merkle OK       : {audit_report.merkle_proofs_ok}")
    print(f"  · Duración total (ms)     : {audit_report.execution_duration_ms:.2f}")
    if audit_report.recurrence_invariant is not None:
        ri = audit_report.recurrence_invariant
        print(
            f"  · Recurrencia Poincaré    : τ̄={ri.mean_recurrence_time:.3f} | "
            f"Kac={ri.kac_bound:.3f} | Hill={ri.fraction_neck_closed:.2f} | "
            f"recurrente={ri.is_recurrent()}"
        )
    assert audit_report.merkle_proofs_ok

    passport = dreamer_agent.emit_dreamer_passport()
    print("\n[FASE 3] Pasaporte criptográfico del soberano:")
    for k, v in passport.items():
        print(f"  · {k:22s}: {v}")
    print("\n" + "═" * 88)
    print("✓ AUDITORÍA Y METABOLIZACIÓN ONÍRICA CONCLUIDA SATISFACTORIAMENTE.")
    print(f"  {__version__}")
    print("═" * 88)
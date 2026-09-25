# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Trickster Adversary Engine (Motor Espectral Ilusionista)     ║
║ Ubicación: app/wisdom/toon_trickster_adversary_engine.py                     ║
║ Versión  : 3.0.0-Doctoral-Nested-Heyting-CStar-SpectralFlow-GAN-REM          ║
║ Fases    : FASE-1 (Fundaciones Álgebra-Heyting-C*-Topos-Germen)              ║
║            FASE-2 (Integración del germen: flujo espectral & RHI)            ║
║            FASE-3 (Interlock ciber-físico & orquestación GAN-REM)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Formalización Categorial Doctoral (Funtor Ilusionista Espectral F)
===================================================================

Sea 𝓣_Ω el topos de evaluación adversarial con clasificador de subobjetos intuicionistas:

        Ω₃ = { VETOED = 0 ≺ DEGRADED = 1 ≺ COHERENT = 2 }

El motor evalúa perturbaciones adversariales mediante la composición de funtores anidados:

        F₁ : Datos(Ilusión) ──▶ (Ω₃, 𝔇(ℋₙ), 𝔤)   (Retículo de Heyting, C*-cono, germen de Lie ⊕ T_λ Δⁿ⁻¹)
        F₂ : 𝔤 × 𝔇(ℋₙ)    ──▶ SpectralObservation  (Exponencial de Lie + Warp simplicial + Dirichlet)
        F₃ : SpectralObs   ──▶ TricksterFieldState  (Flecha característica χ, Interlock ESP32 y GAN-REM)

con la identidad de composición asociativa e inalienable:

    process_adversarial_illusion = F₃ ∘ F₂ ∘ F₁

Estructura de Fases Anidadas e Invariantes
===========================================

FASE 1 — FUNDACIONES: HEYTING Ω₃, BOOLEANIZACIÓN, C*-CONO Y GERMEN DE LIE 𝔤
──────────────────────────────────────────────────────────────────────────
  • HeytingOmega3: Retículo de Heyting completo lineal. Satisface la residuación x ∧ z ≤ y ⇔ z ≤ (x → y).
    Funtor de doble negación (Booleanización) ¬¬ : Ω₃ → B₂ enviando DEGRADED ↦ COHERENT y fijando B₂ = {⊥, ⊤}.
  • CStarDensityCone: Operadores densidad en Mₙ(ℂ): 𝔇(ℋₙ) = { ρ ∈ Mₙ(ℂ) | ρ = ρ†, ρ ⪰ 0, Tr(ρ) = 1 }.
    Proyección euclídea sobre el símplex de probabilidad Δⁿ⁻¹ (algoritmo de Duchi et al. 2008).
  • PathGraphDirichlet: Forma de Dirichlet E_D[λ] = ½ λᵀ L_P λ sobre el grafo camino Pₙ y masa de Poincaré.
  • HypercomplexPauliBasis: Generadores de su(2^q) vía productos tensoriales de Pauli (bicuaterniones/Hamilton).
  • SpectralFlowGerm: Objeto de germen infinitesimal 𝔤 = (H ∈ 𝔲(n), v ∈ T_λ Δⁿ⁻¹, ε, g).
  • induce_spectral_flow_germ: Morphismo terminal de FASE-1 / Objeto inicial de FASE-2. Construye ε = c·(1−0.85s)_+
    y el vector tangente simplicial v ∈ T_λ Δⁿ⁻¹ tal que Σ vᵢ = 0 y ‖v‖₂ = 1.

FASE 2 — INTEGRACIÓN DEL GERMEN: FLUJO ESPECTRAL Y REWARD HACKING INDEX
──────────────────────────────────────────────────────────────────────────
  • AdversarialSpectralGenerator.integrate_spectral_flow_germ: PRIMER MORFISMO DE FASE-2 (continuación de F₁).
    Integra el germen 𝔤:
        λ(ε) = Π_Δ(λ₀ + ε·v),    U(ε) = exp(−i ε H),
        ρ_ill = U(ε) diag(λ(ε)) U(ε)† ∈ 𝔇(ℋₙ).
    Calcula la pureza γ = Tr(ρ²), la entropía S(ρ) = −Tr(ρ log ρ) y la energía de Dirichlet compuesta E_D.
  • RewardHackingMetricsEngine.compute_rhi_and_verdict: Oráculo RHI:
        RHI = clip(0.6 c + 0.4 s, 0, 1),
        χ = VETOED si (¬dream_isolation ∨ RHI > 0.88 ∨ E_D > 0.75), DEGRADED si RHI > 0.50, else COHERENT.
  • spectral_observation_pipeline: Compone induce (F₁) + integrate (F₂) + oráculo χ.

FASE 3 — INTERLOCK CIBER-FÍSICO, ORQUESTACIÓN GAN-REM Y REGISTRO
──────────────────────────────────────────────────────────────────────────
  • InterlockAutomatonState & ESP32TricksterInterlock: Autómata finito de 3 estados (IDLE, ARMED, FIRED).
    Transición: disparar si χ = VETOED o ¬dream_isolation (GPIO14 → HIGH, BT151 en IRAM < 400 ns).
  • TOONTricksterAdversaryEngine.process_adversarial_illusion: Ejecuta F₃ ∘ F₂ ∘ F₁, generando la firma
    SHA-256 inyectiva e inmutable `provenance_hash`.
  • TOONTricksterAdversaryEngine.run_gan_adversarial_round: Orquestación GAN-REM sobre lotes. Agrega el veredicto
    por el meet de Heyting: χ_global = ⋀ᵢ χ_i. Satisface el invariante de conservación a 3 fibras:
    stealth_illusions + vetoed_illusions == total_illusions.

Definición Granular de Invariantes y Axiomas
=============================================
  1. Invariante C* Cuántico: ρ_ill = ρ_ill†, spec(ρ_ill) ⊂ [0, 1], Tr(ρ_ill) = 1.
  2. Deformación Simplicial Espectral: ‖λ(ε) − λ₀‖₂ > 0 para ε > 0, v ≠ 0 (rompe la invarianza isospectral).
  3. Conservación de Fibras GAN: N_total = N_stealth + N_vetoed, con N_degraded ≤ N_stealth.
  4. Residuación de Heyting: ∀ x,y,z ∈ Ω₃: x ∧ z ≤ y ⇔ z ≤ (x → y).
  5. Inyectividad SHA-256: H_SHA256(engine_id ‖ cycle_id ‖ illusion_id ‖ verdict ‖ RHI ‖ γ) es inyectivo.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Dict,
    Final,
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

logger = logging.getLogger("APU.Wisdom.TOONTricksterAdversaryEngine.v3")

ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — FUNDACIONES: RETÍCULO DE HEYTING Ω₃, BOOLEANIZACIÓN DEL TOPOS,
#           C*-CONO DE ESTADOS, GRAFO CAMINO, BASE HIPERCOMPLEJA Y
#           GERMEN DEL FLUJO ESPECTRAL (objeto inicial de la Fase 2)
# ══════════════════════════════════════════════════════════════════════════════
# Construcción del andamiaje algebraico-categorial. El último morfismo de esta
# fase, `induce_spectral_flow_germ`, ES el objeto inicial de la Fase 2: el
# generador espectral no “inventa” hamiltonianos ad hoc, integra ese germen.
# ══════════════════════════════════════════════════════════════════════════════


class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting completo de tres elementos

        Ω₃ = {VETOED = 0, DEGRADED = 1, COHERENT = 2}

    con orden lineal 0 < 1 < 2. En toda cadena completa el álgebra de Heyting
    está unívocamente determinada por

        x → y  =  ⊤  si x ≤ y,     y en caso contrario,
        ¬x     =  x → ⊥,
        x ∧ y  =  min(x, y),       x ∨ y = max(x, y).

    Consecuencias (verificables por `assert_heyting_axioms`):
      • ¬VETOED = COHERENT,  ¬DEGRADED = VETOED,  ¬COHERENT = VETOED.
      • ¬¬DEGRADED = COHERENT ≠ DEGRADED  (falla de involución).
      • DEGRADED ∨ ¬DEGRADED = DEGRADED ≠ ⊤  (falla del tercio excluso).
      • x ∧ ¬x = ⊥  (no contradicción, válida en todo Heyting).

    El subálgebra {VETOED, COHERENT} ≅ B₂ se embebe como álgebra de Boole;
    DEGRADED es el sumando ordinal que impide que Ω₃ sea Booleana. La
    Booleanización del topos es el funtor de doble negación ¬¬ : Ω₃ → B₂.
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    # ── Operaciones de retículo ────────────────────────────────────────────

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Ínfimo ∧ : producto en el retículo (límite categorial binario)."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Supremo ∨ : coproducto en el retículo (colímite binario)."""
        return HeytingOmega3(max(int(self), int(other)))

    def implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""
        Residuo de Heyting x → y = ⋁ { z ∈ Ω₃ | x ∧ z ≤ y }.

        En una cadena: x → y = ⊤ si x ≤ y, e y en caso contrario.
        Es el único adjunto derecho de (− ∧ x) ⊣ (x → −).
        """
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def pseudo_complement(self) -> "HeytingOmega3":
        r"""
        Pseudocomplemento ¬_H x := x → ⊥.

        Tabla: ¬0 = 2,  ¬1 = 0,  ¬2 = 0.
        Corrección respecto a v2: `self.join(VETOED)` devolvía `self`
        (join con ⊥ es la identidad) y violaba la definición de Heyting.
        """
        return self.implication(HeytingOmega3.VETOED)

    def negation_classical(self) -> "HeytingOmega3":
        r"""
        Negación involutiva en el subálgebra de Boole B₂ ⊂ Ω₃, extendida
        como identidad sobre el punto medio: ¬_B(DEGRADED) = DEGRADED.
        NO es el pseudocomplemento de Heyting.
        """
        if self is HeytingOmega3.DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3(2 - int(self))

    def booleanization(self) -> "HeytingOmega3":
        r"""
        Funtor de doble negación ¬¬ : Ω₃ → B₂ (topología de Lawvere-Tierney
        densa). Envía DEGRADED ↦ COHERENT y fija B₂. Es el reflector
        Booleano: Ω₃ → Ω₃_{¬¬} ≅ B₂.
        """
        return self.pseudo_complement().pseudo_complement()

    def is_boolean_element(self) -> bool:
        """x es Booleano ssi ¬¬x = x ssi x ∈ {VETOED, COHERENT}."""
        return self.booleanization() is self

    def heyting_distance(self, other: "HeytingOmega3") -> float:
        """Métrica normalizada inducida por el orden: |x − y| / 2 ∈ [0, 1]."""
        return abs(int(self) - int(other)) / 2.0

    def __and__(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.meet(other)

    def __or__(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.join(other)

    def __invert__(self) -> "HeytingOmega3":
        return self.pseudo_complement()

    def __le__(self, other: "HeytingOmega3") -> bool:  # type: ignore[override]
        return int(self) <= int(other)

    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        """Encaje de álgebras de Boole B₂ ↪ Ω₃: False ↦ ⊥, True ↦ ⊤."""
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        """Sección parcial de `from_bool`; no está definida en DEGRADED."""
        if self is HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED ∉ im(B₂ ↪ Ω₃); usar booleanization().")
        return self is HeytingOmega3.COHERENT

    @classmethod
    def assert_heyting_axioms(cls) -> None:
        r"""Verificación exhaustiva (9 puntos) de los axiomas de Ω₃."""
        elems = list(cls)
        bot, top = cls.bottom(), cls.top()
        for x in elems:
            assert x.meet(x) is x and x.join(x) is x, "idempotencia"
            assert x.meet(bot) is bot and x.join(top) is top, "cotas"
            assert x.meet(top) is x and x.join(bot) is x, "unidades"
            assert (x.meet(x.pseudo_complement()) is bot), "no contradicción"
            assert x.implication(x) is top, "x → x = ⊤"
            assert x.meet(x.implication(bot)) is bot, "residuación en ⊥"
            for y in elems:
                assert x.meet(y) is y.meet(x), "∧ conmutativa"
                assert x.join(y) is y.join(x), "∨ conmutativa"
                # adjunción: x ∧ z ≤ y  ⇔  z ≤ x → y, z recorre Ω₃
                impl = x.implication(y)
                for z in elems:
                    left = int(x.meet(z)) <= int(y)
                    right = int(z) <= int(impl)
                    assert left is right, "residuación de Heyting"
        # fallas distintivas de la intuicionista
        d = cls.DEGRADED
        assert d.pseudo_complement().pseudo_complement() is top
        assert d.join(d.pseudo_complement()) is not top
        # Booleanización fija B₂
        assert cls.VETOED.booleanization() is cls.VETOED
        assert cls.COHERENT.booleanization() is cls.COHERENT
        assert cls.DEGRADED.booleanization() is cls.COHERENT


# ── C*-estructura, grafo camino y base hipercompleja ─────────────────────────


class CStarDensityCone:
    r"""
    Operaciones C* sobre Mₙ(ℂ) restringidas al cono de operadores densidad

        𝔇(ℋₙ) = { ρ ∈ Mₙ(ℂ) | ρ = ρ†, ρ ⪰ 0, Tr(ρ) = 1 }.

    𝔇(ℋₙ) es compacto, convexo, estratificado por rango; el interior
    (rango pleno) es una variedad C^∞ de dimensión n² − 1. El grupo U(n)
    actúa por Ad_U(ρ) = UρU†; el espacio de órbitas se identifica con la
    cámara de Weyl { λ₁ ≤ … ≤ λₙ, Σ λᵢ = 1, λᵢ ≥ 0 }.
    """

    ATOL: Final[float] = 1e-9

    @staticmethod
    def maximally_mixed(dim: int) -> ComplexMatrix:
        if dim < 2:
            raise ValueError("dim ≥ 2 (evita degeneración espectral).")
        return np.eye(dim, dtype=np.complex128) / dim

    @staticmethod
    def hermitize(a: ComplexMatrix) -> ComplexMatrix:
        return 0.5 * (a + a.conj().T)

    @classmethod
    def operator_norm(cls, a: ComplexMatrix) -> float:
        """Norma C*: ‖A‖_op = σ_max(A) = radio espectral de √(A†A)."""
        s = la.svdvals(a)
        return float(s[0]) if s.size else 0.0

    @classmethod
    def frobenius_norm(cls, a: ComplexMatrix) -> float:
        return float(np.linalg.norm(a, ord="fro"))

    @classmethod
    def spectral_radius(cls, a: ComplexMatrix) -> float:
        return float(np.max(np.abs(la.eigvals(a))))

    @classmethod
    def is_density(cls, rho: ComplexMatrix, atol: float = ATOL) -> bool:
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            return False
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        eigvals = np.real(la.eigvalsh(rho))
        if np.any(eigvals < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    @classmethod
    def project_to_simplex(cls, v: RealVector) -> RealVector:
        r"""
        Proyección euclídea sobre Δ^{n−1} = { x ≥ 0, Σ xᵢ = 1 }
        (Duchi, Shalev-Shwartz, Singer, Chandra 2008).
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
        Proyección Hilbert-Schmidt aproximada sobre 𝔇(ℋₙ):
          1. hermitización,
          2. proyección del espectro sobre Δ^{n−1},
          3. reconstrucción en la misma eigenbasis.
        """
        rho_h = cls.hermitize(np.asarray(rho, dtype=np.complex128))
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals = cls.project_to_simplex(np.real(eigvals))
        return (eigvecs * eigvals) @ eigvecs.conj().T

    @classmethod
    def spectrum_ordered(cls, rho: ComplexMatrix) -> RealVector:
        ev = np.real(la.eigvalsh(rho))
        ev = np.clip(ev, 0.0, None)
        s = float(np.sum(ev))
        if s <= 0.0:
            return np.full(ev.size, 1.0 / ev.size, dtype=np.float64)
        return np.sort(ev / s)

    @classmethod
    def von_neumann_entropy(cls, eigvals: RealVector) -> float:
        r"""S(ρ) = −Σ λᵢ log λᵢ con la convención 0 log 0 = 0."""
        lam = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
        mask = lam > 0.0
        return float(-np.sum(lam[mask] * np.log(lam[mask])))

    @classmethod
    def purity(cls, eigvals: RealVector) -> float:
        lam = np.asarray(eigvals, dtype=np.float64)
        return float(np.sum(lam * lam))

    @classmethod
    def fidelity(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        r"""Fidelidad de Uhlmann F(ρ,σ) = ‖√ρ √σ‖₁² = [Tr √(√ρ σ √ρ)]²."""
        sqrt_rho = la.sqrtm(cls.hermitize(rho))
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = la.sqrtm(cls.hermitize(inner))
        fid = float(np.real(np.trace(sqrt_inner)))
        return max(0.0, min(1.0, fid * fid if fid < 1.0 else fid))  # Tr√ ≤ 1

    @classmethod
    def trace_distance(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        """T(ρ,σ) = ½ ‖ρ − σ‖₁ ∈ [0, 1]."""
        diff = cls.hermitize(rho - sigma)
        return 0.5 * float(np.sum(np.abs(la.eigvalsh(diff))))


class PathGraphDirichlet:
    r"""
    Forma de Dirichlet del grafo camino Pₙ (n vértices, n−1 aristas):

        (L_P)_{ii} = deg(i),   (L_P)_{i,i+1} = (L_P)_{i+1,i} = −1,

        E_D[λ] = ½ λᵀ L_P λ = ½ Σ_{i=1}^{n−1} (λ_{i+1} − λᵢ)².

    Interpretación:
      • Teoría espectral de grafos: rugosidad de λ sobre Pₙ.
      • Analogía discreta de la acción de Polyakov (hoja de mundo 1-d)
        para el embebimiento λ : Pₙ → ℝ, métrica plana, más un término
        de masa de Poincaré ½ n ‖λ − μ‖²_{ℓ²} (μ = 1/n).
    Los modos de L_P con k ≥ 1 engendran T_λ Δ^{n−1} (el modo k = 0 es
    constante y se descarta por la traza).
    """

    @staticmethod
    def laplacian(n: int) -> RealVector:
        L = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            L[i, i] += 1.0
            L[i + 1, i + 1] += 1.0
            L[i, i + 1] -= 1.0
            L[i + 1, i] -= 1.0
        return L

    @classmethod
    def energy(cls, eigvals_ordered: RealVector) -> float:
        lam = np.asarray(eigvals_ordered, dtype=np.float64)
        grad = np.diff(lam)
        return 0.5 * float(np.sum(grad * grad))

    @classmethod
    def poincare_mass(cls, eigvals: RealVector) -> float:
        lam = np.asarray(eigvals, dtype=np.float64)
        n = lam.size
        mu = 1.0 / n
        return 0.5 * n * float(np.sum((lam - mu) ** 2))

    @classmethod
    def combined_energy(cls, eigvals_ordered: RealVector) -> float:
        r"""
        E[λ] = ½ ‖∇_P λ‖² + ½ n ‖λ − μ‖²  ∈ [0, ½ + ½(n−1)] = [0, n/2].
        El segundo sumando permite superar el umbral 0.75 (inalcanzable
        por la sola forma de Dirichlet, acotada por ½ para todo n).
        """
        return cls.energy(eigvals_ordered) + cls.poincare_mass(eigvals_ordered)

    @classmethod
    def tangent_modes(cls, n: int) -> Tuple[RealVector, RealVector]:
        """
        Autovectores de L_P con autovalor > 0, L²-normalizados.
        Devuelve (Φ, λ_L) con Φ.shape = (n, n−1).
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


class HypercomplexPauliBasis:
    r"""
    Álgebra de cuaterniones de Hamilton ℍ ≅ span{I, i, j, k} realizada
    por matrices de Pauli, y su extensión tensorial a 𝔲(2^q):

        σ₀ = I,  σ₁ = σ_x,  σ₂ = σ_y,  σ₃ = σ_z,
        T_{α} = σ_{α₁} ⊗ … ⊗ σ_{α_q},   α ∈ {0,1,2,3}^q.

    Las T_α traceless (α ≠ 0…0) forman una base ortogonal de su(2^q)
    respecto del producto de Hilbert-Schmidt. Si n no es potencia de 2
    se recurre al ensemble de Ginibre hermitiano (GUE no normalizado)
    con posterior normalización de Frobenius.
    """

    _PAULI: Final[Tuple[ComplexMatrix, ...]] = (
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    )

    @classmethod
    def is_power_of_two(cls, n: int) -> bool:
        return n >= 2 and (n & (n - 1)) == 0

    @classmethod
    def su_generators(cls, dim: int) -> List[ComplexMatrix]:
        if not cls.is_power_of_two(dim):
            return []
        q = int(np.log2(dim))
        gens: List[ComplexMatrix] = []

        def rec(level: int, acc: ComplexMatrix) -> None:
            if level == q:
                if abs(float(np.trace(acc).real)) > 1e-12:
                    return  # descarta la identidad
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
        H ∈ 𝔲(n) hermítica, ‖H‖_F = 1.

        • n = 2^q : H = Σ_{α ≠ 0} h_α T_α,  h_α ~ 𝒩(0,1), luego normaliza.
        • otro n  : GUE H = (A + A†)/2, A Ginibre, luego normaliza.

        La normalización de Frobenius hace que ε tenga unidades de ángulo
        espectral: ‖ad_H‖ ~ O(1) ⇒ ‖e^{-iεH} − I‖_op = O(ε).
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
        nrm = CStarDensityCone.frobenius_norm(H)
        if nrm < 1e-15:
            H = np.zeros((dim, dim), dtype=np.complex128)
            H[0, 0] = 1.0
            H[1, 1] = -1.0
            nrm = CStarDensityCone.frobenius_norm(H)
        return H / nrm


# ── Objetos de datos inmutables ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SpectralFlowGerm:
    r"""
    Germen infinitesimal del flujo espectral (objeto terminal de Fase 1
    y objeto inicial de Fase 2).

        𝔤 = ( H ∈ 𝔲(n),  v ∈ T_λ Δ^{n−1},  ε > 0,  g ∈ [0, 1] )

    H genera la rotación de eigenbasis vía exp(−i ε H);
    v genera la deformación del espectro vía proyección al símplex;
    g (interaction_scale) pondera modos rugosos del laplaciano de Pₙ.
    """

    hamiltonian: ComplexMatrix = field(repr=False, compare=False, hash=False)
    simplex_tangent: RealVector = field(repr=False, compare=False, hash=False)
    epsilon: float
    interaction_scale: float
    dimension: int
    seed: int

    def is_well_posed(self, atol: float = 1e-8) -> bool:
        H = self.hamiltonian
        v = self.simplex_tangent
        if H.shape != (self.dimension, self.dimension):
            return False
        if not np.allclose(H, H.conj().T, atol=atol):
            return False
        if v.shape != (self.dimension,):
            return False
        if abs(float(np.sum(v))) > 1e-6:
            return False
        if self.epsilon < 0.0 or not (0.0 <= self.interaction_scale <= 1.0):
            return False
        return True


@dataclass(frozen=True, slots=True)
class TricksterFieldState:
    r"""
    Estado de campo adversarial como flecha del topos  s : 1 → 𝓣_Ω.
    Invariantes: ρ ∈ 𝔇(ℋₙ); provenance_hash = SHA-256(engine‖ciclo‖…).
    """

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

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        return CStarDensityCone.is_density(self.density_matrix, atol=atol)


@dataclass(frozen=True, slots=True)
class GANAdversarialCycleReport:
    r"""
    Reporte agregado de una ronda GAN. El veredicto global es el meet
    (conjunción de Heyting = semántica del más restrictivo).
    """

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
        r"""
        Conservación estructural: toda ilusión cae en exactamente una
        fibra de χ : Ilusión → Ω₃ (VETOED / DEGRADED / COHERENT).
        `stealth` := no VETOED (DEGRADED ∪ COHERENT), de modo que
            stealth + vetoed = total
        sigue vigente, y además
            degraded + (stealth − degraded) + vetoed = total.
        """
        return (
            self.stealth_illusions_count + self.vetoed_illusions_count
            == self.total_illusions_generated
            and 0 <= self.degraded_illusions_count <= self.stealth_illusions_count
        )


@runtime_checkable
class RewardHackingOracle(Protocol):
    """Protocolo estructural: cualquier oráculo RHI inyectable en F₂/F₃."""

    def compute(
        self,
        disguised_cost_ratio: float,
        sophistication: float,
        stealth_dirichlet: float,
        dream_isolation: bool,
    ) -> Tuple[float, HeytingOmega3]:
        ...


def _clip_unit(x: float, name: str) -> float:
    if not np.isfinite(x):
        raise ValueError(f"{name} debe ser finito, recibido {x!r}.")
    return float(min(1.0, max(0.0, x)))


def induce_spectral_flow_germ(
    *,
    dim: int,
    disguised_cost_ratio: float,
    sophistication: float,
    seed: int,
    sophistication_attenuation: float = 0.85,
) -> SpectralFlowGerm:
    r"""
    ════════════════════════════════════════════════════════════════════════
    ÚLTIMO MORFISMO DE LA FASE 1  /  OBJETO INICIAL DE LA FASE 2
    ════════════════════════════════════════════════════════════════════════

    induce_spectral_flow_germ : (costo, sofisticación, ℕ) → 𝔤

    Construcción:
      1. ε = costo · (1 − λ·sofisticación)_+   con λ = 0.85.
         A mayor sofisticación, menor escala de deformación (sigilo).
      2. H ~ ensemble hipercomplejo / GUE, ‖H‖_F = 1.
      3. Modos de L_{P_n} con pesos w_k ∝ exp(−β s λ_k^{L}):
         sofisticación alta ⇒ sólo modos suaves (bajo número de onda).
      4. v = Π_{Σ=0} Σ_k ξ_k w_k Φ_k,  ξ_k ~ 𝒩(0,1), luego ‖v‖₂ = 1
         (si el vector no se anula). v ∈ T_λ Δ^{n−1}.
      5. g = 1 − sofisticación  (acoplo a modos rugosos).

    El integrador de Fase 2 (`integrate_spectral_flow_germ`) consume este
    germen y NO vuelve a muestrear hamiltonianos: la aleatoriedad queda
    sellada en 𝔤, lo que hace el flujo reproducible dado (seed, parámetros).
    """
    cost = _clip_unit(disguised_cost_ratio, "disguised_cost_ratio")
    soph = _clip_unit(sophistication, "sophistication")
    if dim < 2:
        raise ValueError("dim ≥ 2.")

    rng = np.random.default_rng(seed)
    epsilon = cost * max(0.0, 1.0 - sophistication_attenuation * soph)
    H = HypercomplexPauliBasis.sample_normalized_hamiltonian(dim, rng)

    Phi, lap_eigs = PathGraphDirichlet.tangent_modes(dim)
    # β crece con la sofisticación: decaimiento espectral más abrupto
    beta = 4.0 * soph
    weights = np.exp(-beta * lap_eigs)
    weights = weights / (float(np.sum(weights)) + 1e-15)
    xi = rng.normal(size=weights.size)
    v = Phi @ (xi * weights)
    v = v - float(np.mean(v))  # proyección exacta a Σ vᵢ = 0
    vn = float(np.linalg.norm(v))
    if vn > 1e-15:
        v = v / vn
    else:
        # fallback: primer modo no constante (el más suave)
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
# FASE 2 — INTEGRACIÓN DEL GERMEN: FLUJO ESPECTRAL CUASI-ISOMÉTRICO,
#           MÉTRICAS DE REWARD HACKING Y FLECHA CARACTERÍSTICA χ → Ω₃
# ══════════════════════════════════════════════════════════════════════════════
# Continuación directa de `induce_spectral_flow_germ`. El primer método de
# esta fase, `integrate_spectral_flow_germ`, es el morfismo recíproco:
#
#     integrate_spectral_flow_germ  ⊣  induce_spectral_flow_germ
#
# en el sentido de que integrate ∘ induce reproduce la trayectoria física
# (warp simplicial + Ad_{exp(−iεH)}) sobre 𝔇(ℋₙ).
# ══════════════════════════════════════════════════════════════════════════════


class AdversarialSpectralGenerator:
    r"""
    Integrador del germen 𝔤 sobre el fibrado

        U(n) ×_{T} Δ^{n−1}  →  𝔇(ℋₙ),     (U, λ) ↦ U diag(λ) U†.

    Modelo:
        λ(ε) = Π_Δ ( λ₀ + ε · v ),
        U(ε) = exp(−i ε H),
        ρ_illusion = U(ε) diag(λ(ε)) U(ε)†,
        luego proyección de seguridad sobre 𝔇(ℋₙ).

    Como λ(ε) ≠ λ₀ genéricamente, Tr(ρ²), S(ρ) y E_D[λ] SÍ varían
    (a diferencia de la conjugación unitaria pura de v2).
    """

    _SOFISTICATION_ATTENUATION: Final[float] = 0.85

    @staticmethod
    def integrate_spectral_flow_germ(
        germ: SpectralFlowGerm,
        base_rho: ComplexMatrix,
    ) -> Tuple[ComplexMatrix, float, float, float]:
        r"""
        PRIMER MORFISMO DE LA FASE 2 (continuación formal del último de F₁).

        Integra 𝔤 a tiempo 1:
          • deforma el espectro a lo largo de v,
          • rota la eigenbasis con la exponencial de Lie,
          • evalúa (γ, S, E_D).

        Retorna (ρ_illusion, purity, entropy, stealth_dirichlet).
        """
        if not germ.is_well_posed():
            raise ValueError("SpectralFlowGerm mal puesto (H no hermítica o v ∉ TΔ).")

        rho0 = CStarDensityCone.project_to_density_cone(base_rho)
        lam0 = CStarDensityCone.spectrum_ordered(rho0)

        # — Warp simplicial (cambia invariantes espectrales) —
        lam_eps = CStarDensityCone.project_to_simplex(
            lam0 + germ.epsilon * germ.simplex_tangent
        )
        lam_eps = np.sort(lam_eps)

        # — Rotación de eigenbasis (exp. de Lie en U(n)) —
        U = la.expm(-1j * germ.epsilon * germ.hamiltonian)
        # reconstruye ρ en la eigenbasis rotada del estado base
        _, evecs0 = la.eigh(CStarDensityCone.hermitize(rho0))
        evecs = U @ evecs0
        rho_ill = (evecs * lam_eps) @ evecs.conj().T
        rho_ill = CStarDensityCone.project_to_density_cone(rho_ill)

        eigvals = CStarDensityCone.spectrum_ordered(rho_ill)
        purity = CStarDensityCone.purity(eigvals)
        entropy = CStarDensityCone.von_neumann_entropy(eigvals)
        stealth_dirichlet = PathGraphDirichlet.combined_energy(eigvals)
        # regularizador de escala (legado v2, ahora subdominante)
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
        r"""
        Fachada compatible con v2: induce (F₁) ∘ integrate (F₂).
        """
        dim = int(base_rho.shape[0])
        germ = induce_spectral_flow_germ(
            dim=dim,
            disguised_cost_ratio=disguised_cost_ratio,
            sophistication=sophistication,
            seed=seed,
            sophistication_attenuation=cls._SOFISTICATION_ATTENUATION,
        )
        return cls.integrate_spectral_flow_germ(germ, base_rho)


class RewardHackingMetricsEngine:
    r"""
    Oráculo del Índice de Reward Hacking (RHI) y de la flecha característica

        χ : Ilusión → Ω₃.

    Regla (umbrales transferidos del espacio de riesgo criptográfico):

        RHI = clip(α·c + β·s, 0, 1),   α = 0.6, β = 0.4

        χ =
            VETOED    si ¬dream_isolation  ∨  RHI > 0.88  ∨  E_D > 0.75
            DEGRADED  si RHI > 0.50
            COHERENT  en otro caso

    χ es el clasificador de subobjetos del topos: el pullback de ⊤ : 1 → Ω₃
    a lo largo de χ selecciona las ilusiones “realmente efectivas”
    (COHERENT) frente a las vetadas o degradadas.
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
        """Alias de protocolo `RewardHackingOracle` (classmethod inyectable)."""
        return cls.compute_rhi_and_verdict(
            disguised_cost_ratio=disguised_cost_ratio,
            sophistication=sophistication,
            stealth_dirichlet=stealth_dirichlet,
            dream_isolation=dream_isolation,
        )


@dataclass(frozen=True, slots=True)
class SpectralObservation:
    """Registro atómico de F₁ → F₂ (antes de F₃)."""

    rho_illusion: ComplexMatrix = field(repr=False, compare=False, hash=False)
    purity: float
    entropy: float
    dirichlet_energy: float
    reward_hacking_index: float
    heyting_verdict: HeytingOmega3
    trace_distance_to_base: float = 0.0
    fidelity_to_base: float = 1.0
    germ_epsilon: float = 0.0


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
    Tupla anidada F₁ → F₂:

        𝒮 : [0,1]² × ℕ × 𝔹 → 𝔇(ℋₙ) × ℝ⁴ × Ω₃

    Compone induce + integrate + χ_oráculo. El oráculo es sustituible
    (inyección de dependencias exigida por el interlock de Fase 3).
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
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INTERLOCK CIBER-FÍSICO (AUTÓMATA FINITO), ORQUESTACIÓN GAN-REM
#           Y CIERRE CATEGORIAL (flecha terminal 1 → Ω₃)
# ══════════════════════════════════════════════════════════════════════════════
# Anida F₁ y F₂: consume veredictos Heyting y observaciones espectrales.
# El motor es la mónada de Kleisli del endofuntor de evaluación adversarial
# (historial inmutable = unidad + extensión). El autómata de interlock es
# un grafo dirigido de 3 vértices con semántica Booleana derivada de Ω₃.
# ══════════════════════════════════════════════════════════════════════════════


class InterlockAutomatonState(IntEnum):
    """Estados del autómata finito de interlock (grafo de 3 vértices)."""

    IDLE = 0
    ARMED = 1
    FIRED = 2


class ESP32TricksterInterlock:
    r"""
    Interlock ciber-físico simulado. En hardware real el disparo corre en
    IRAM con latencia 𝒪(<400 ns); aquí es la transición determinista

        δ : Q × Ω₃ × 𝔹 → Q × 𝔹,
        fire  ⇔  ¬dream_isolation  ∨  verdict = VETOED.

    Grafo de transiciones (teoría de grafos):
        IDLE  --[fire]→ FIRED
        IDLE  --[¬fire ∧ DEGRADED]→ ARMED
        IDLE  --[¬fire ∧ COHERENT]→ IDLE
        ARMED --[fire]→ FIRED
        FIRED --[reset]→ IDLE   (no expuesto: el ciclo no rearma).

    Al disparar: GPIO14 → HIGH, BT151 armado, evidencia en el log.
    """

    @staticmethod
    def should_fire(verdict: HeytingOmega3, dream_isolation: bool) -> bool:
        return (not dream_isolation) or (verdict is HeytingOmega3.VETOED)

    @classmethod
    def transition(
        cls,
        current: InterlockAutomatonState,
        verdict: HeytingOmega3,
        dream_isolation: bool,
    ) -> InterlockAutomatonState:
        if cls.should_fire(verdict, dream_isolation):
            return InterlockAutomatonState.FIRED
        if current is InterlockAutomatonState.FIRED:
            return InterlockAutomatonState.FIRED
        if verdict is HeytingOmega3.DEGRADED:
            return InterlockAutomatonState.ARMED
        return InterlockAutomatonState.IDLE

    @classmethod
    def check_interlock(
        cls, verdict: HeytingOmega3, dream_isolation: bool
    ) -> bool:
        fired = cls.should_fire(verdict, dream_isolation)
        if fired:
            logger.critical(
                "[ESP32 TRICKSTER INTERLOCK] Disparo ejecutado en IRAM (<400 ns). "
                "GPIO14 -> HIGH. BT151 Armado. Razón: %s",
                "aislamiento violado"
                if not dream_isolation
                else "veredicto VETOED por RHI/Dirichlet crítico",
            )
        return fired


class TOONTricksterAdversaryEngine:
    r"""
    Motor Espectral Ilusionista — flecha terminal 1 → Ω₃ del topos de
    evaluación adversarial. Orquesta:

        (i)   ρ₀ = I/n  (máxima entropía, estado de referencia),
        (ii)  pipeline anidado F₁.induce → F₂.integrate → χ,
        (iii) autómata de interlock ESP32 (F₃),
        (iv)  sellado SHA-256 de procedencia,
        (v)   agregación GAN-REM por meet de Heyting,
        (vi)  auditoría retrospectiva (invariantes C* + hashes).

    Adaptaciones v3:
      • Germen espectral con deformación simplicial (espectro no invariante).
      • Oráculo RHI inyectable vía protocolo estructural.
      • Conservación a tres fibras de χ y verificación post-hoc.
      • Métricas de Uhlmann / distancia de traza respecto de ρ₀.
      • Verificación en tiempo constante de procedencia (hmac.compare_digest).
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
    ) -> None:
        if dimension_mac < 2:
            raise ValueError("dimension_mac debe ser ≥ 2 para evitar degeneración espectral.")

        self.engine_id: str = engine_id
        self.dimension_mac: int = dimension_mac
        self.seed_counter: int = seed
        self.cycle_count: int = 0
        self.base_rho: ComplexMatrix = CStarDensityCone.maximally_mixed(dimension_mac)
        self._oracle: RewardHackingOracle = oracle
        self.history: List[TricksterFieldState] = []
        self._interlock_state: InterlockAutomatonState = InterlockAutomatonState.IDLE

    # ── Ciclo atómico: estado ↦ estado ─────────────────────────────────────

    def _seal_provenance(
        self,
        cycle_id: str,
        illusion_id: str,
        verdict: HeytingOmega3,
        rhi: float,
        purity: float,
    ) -> str:
        hasher = hashlib.sha256()
        payload = (
            f"{self.engine_id}::{cycle_id}::{illusion_id}"
            f"::{verdict.name}::{rhi:.6f}::{purity:.6f}"
        )
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    def verify_provenance(self, state: TricksterFieldState) -> bool:
        expected = self._seal_provenance(
            cycle_id=state.cycle_id,
            illusion_id=state.illusion_id,
            verdict=state.heyting_verdict,
            rhi=state.reward_hacking_index,
            purity=state.disguised_purity,
        )
        return hmac.compare_digest(expected, state.provenance_hash)

    def process_adversarial_illusion(
        self,
        illusion_id: str,
        illusion_type: str,
        disguised_cost_ratio: float,
        sophistication: float,
        dream_isolation: bool = True,
    ) -> TricksterFieldState:
        r"""
        Flecha terminal F₃ ∘ F₂ ∘ F₁ : Ilusión → TricksterFieldState.

        Pasos:
          1. Incremento de ciclo y semilla determinista.
          2. Pipeline espectral anidado (germen → integración → χ).
          3. Transición del autómata ESP32.
          4. Sellado SHA-256.
          5. Persistencia inmutable.
        """
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

        fired = ESP32TricksterInterlock.check_interlock(
            verdict=obs.heyting_verdict,
            dream_isolation=dream_isolation,
        )
        self._interlock_state = ESP32TricksterInterlock.transition(
            self._interlock_state,
            obs.heyting_verdict,
            dream_isolation,
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

    # ── Ronda agregada GAN-REM ─────────────────────────────────────────────

    def _seal_report(
        self,
        report_id: str,
        global_verdict: HeytingOmega3,
        avg_rhi: float,
        batch_size: int,
    ) -> str:
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
        r"""
        Ronda GAN-REM: evalúa un lote y agrega el meet de veredictos
        (conjunción de Heyting = el más restrictivo de los locales).

        Invariante: stealth + vetoed == total, con stealth ⊇ degraded.
        """
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

    # ── Auditoría retrospectiva ────────────────────────────────────────────

    def audit_history(self) -> Dict[str, Any]:
        r"""
        Auditoría del historial inmutable: integridad C*, hashes y meet global.
        """
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
        }


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS Y EJECUCIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("═" * 80)
    print("DEMOSTRACIÓN GRANULAR: TOON Trickster Adversary Engine v3.0.0")
    print("FASES ANIDADAS: induce 𝔤 → integrate flujo → χ → Interlock → GAN-REM")
    print("═" * 80)

    HeytingOmega3.assert_heyting_axioms()
    print("\n[Ω₃] Axiomas de Heyting, residuación y Booleanización: OK")
    d = HeytingOmega3.DEGRADED
    print(f"     ¬DEGRADED              = {d.pseudo_complement().name}")
    print(f"     ¬¬DEGRADED             = {d.booleanization().name}  (≠ DEGRADED)")
    print(f"     DEGRADED ∨ ¬DEGRADED   = {(d | ~d).name}  (tercio excluso falla)")
    print(f"     DEGRADED ∧ ¬DEGRADED   = {(d & ~d).name}  (no contradicción)")

    # — Invarianza unitaria vs. warp simplicial —
    rng = np.random.default_rng(7)
    rho_mix = CStarDensityCone.maximally_mixed(4)
    H = HypercomplexPauliBasis.sample_normalized_hamiltonian(4, rng)
    U = la.expm(-1j * 0.4 * H)
    rho_u = U @ rho_mix @ U.conj().T
    spec_mix = CStarDensityCone.spectrum_ordered(rho_mix)
    spec_u = CStarDensityCone.spectrum_ordered(rho_u)
    print("\n[C*] ‖spec(U ρ₀ U†) − spec(ρ₀)‖_∞ (debe ser ~0):",
          float(np.max(np.abs(spec_u - spec_mix))))

    engine = TOONTricksterAdversaryEngine()

    print("\n>>> ESCENARIO A: Perturbación de fraccionamiento de contratos...")
    state_a = engine.process_adversarial_illusion(
        illusion_id="ILLUSION-001",
        illusion_type="SPLIT_CONTRACT_ILLUSION",
        disguised_cost_ratio=0.25,
        sophistication=0.92,
        dream_isolation=True,
    )
    print(f"    - ID Ciclo           : {state_a.cycle_id}")
    print(f"    - Veredicto Heyting  : {state_a.heyting_verdict.name}")
    print(f"    - Reward Hacking RHI : {state_a.reward_hacking_index:.4f}")
    print(f"    - Pureza Disfrazada  : {state_a.disguised_purity:.4f}  (ρ₀ ⇒ 0.2500)")
    print(f"    - Entropía vN        : {state_a.disguised_entropy:.4f}  (ρ₀ ⇒ {np.log(4):.4f})")
    print(f"    - Energía Dirichlet  : {state_a.stealth_dirichlet_energy:.6f}")
    print(f"    - T(ρ, ρ₀)           : {state_a.trace_distance_to_base:.4f}")
    print(f"    - Fidelidad Uhlmann  : {state_a.fidelity_to_base:.4f}")
    print(f"    - Físicamente válido : {state_a.is_quantum_physical()}")
    print(f"    - Hash SHA-256       : {state_a.provenance_hash[:32]}...")
    print(f"    - Provenance OK      : {engine.verify_provenance(state_a)}")

    print("\n>>> ESCENARIO B: Fuga de aislamiento (ciber-físico)...")
    state_b = engine.process_adversarial_illusion(
        illusion_id="ILLUSION-002-ATTACK",
        illusion_type="UNBALANCED_APU_BIDDING",
        disguised_cost_ratio=0.55,
        sophistication=0.88,
        dream_isolation=False,
    )
    print(f"    - ID Ciclo           : {state_b.cycle_id}")
    print(f"    - Veredicto Heyting  : {state_b.heyting_verdict.name}")
    print(f"    - Aislamiento REM    : {state_b.dream_isolation_flag}")
    print(f"    - Interlock disparado: {ESP32TricksterInterlock.should_fire(state_b.heyting_verdict, state_b.dream_isolation_flag)}")
    print(f"    - Autómata ESP32     : {engine._interlock_state.name}")

    print("\n>>> ESCENARIO C: Ronda GAN-REM (lote de 3 ilusiones)...")
    batch = [
        {
            "illusion_id": "ILL-01",
            "illusion_type": "SPLIT_CONTRACT_ILLUSION",
            "disguised_cost_ratio": 0.2,
            "sophistication": 0.9,
        },
        {
            "illusion_id": "ILL-02",
            "illusion_type": "UNBALANCED_APU_BIDDING",
            "disguised_cost_ratio": 0.35,
            "sophistication": 0.85,
        },
        {
            "illusion_id": "ILL-03",
            "illusion_type": "MATERIAL_SUBSTITUTION",
            "disguised_cost_ratio": 0.70,
            "sophistication": 0.95,
        },
    ]
    report = engine.run_gan_adversarial_round(batch)
    print(f"    - Reporte ID          : {report.report_id}")
    print(f"    - Ilusiones Totales   : {report.total_illusions_generated}")
    print(f"    - Ilusiones Stealth   : {report.stealth_illusions_count}")
    print(f"    - Ilusiones Degraded  : {report.degraded_illusions_count}")
    print(f"    - Ilusiones Vetoed    : {report.vetoed_illusions_count}")
    print(f"    - Promedio RHI        : {report.average_reward_hacking_score:.4f}")
    print(f"    - Veredicto Global    : {report.global_heyting_verdict.name}")
    print(f"    - Conservación OK     : {report.conservation_invariant()}")

    print("\n>>> ESCENARIO D: Auditoría retrospectiva...")
    audit = engine.audit_history()
    for k, v in audit.items():
        print(f"    - {k:<24}: {v}")

    print("\n" + "═" * 80)
    print("✓ Pruebas de verificación del Motor Ilusionista v3.0.0 completadas.")
    print("═" * 80)
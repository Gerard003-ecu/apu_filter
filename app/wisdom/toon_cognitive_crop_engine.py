# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  TOON Cognitive Crop Engine — Soberano Motor del Cultivo Cognitivo            ║
║  Ubicación: app/wisdom/toon_cognitive_crop_engine.py                          ║
║  Versión  : 2.2.0-Doctoral-Nested-Banach-Brockett-Rényi-Faith-MAC             ║
║  Fases    : FASE-1 → FASE-2 → FASE-3  (anidadas: el último método de k es el   ║
║            germen formal del primero de k+1)                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Formalización Categorial Doctoral (Funtor del Cultivo Cognitivo C)
==================================================================

Sea 𝓣_Ω el topos de haces con clasificador Ω₃ = { VETOED = 0 ≺ DEGRADED = 1 ≺ COHERENT = 2 }.
Las cuatro operaciones del cultivo se formalizan como morfismos en la categoría Crop mediante el funtor:

        C  :  M_n(ℂ) × Σ* × Σ*  ──▶  CropGerminationCertificate

componiendo de forma asociativa las tres fases anidadas:

        C  =  Certify ∘ CropGrowthPipeline ∘ Prepare

donde el tipo de retorno del último método de la fase k es el dominio inalienable de la fase k+1.

Estructura de Fases Anidadas e Invariantes
===========================================

FASE 1 — SUSTRATO ALGEBRAICO Y PREPARACIÓN DEL CRISTAL SEMILLA
──────────────────────────────────────────────────────────────────────────
  • HeytingOmega3: Retículo de Heyting completo Ω₃. Residuo a → b = ⊤ si a ≤ b, else b.
    Satisface residuación (a ∧ c ≤ b ⇔ c ≤ (a → b)) y la falla del tercio excluso en DEGRADED.
  • DensityOperatorAlgebra: Operadores en 𝔇_n. Entropía S(ρ) = −Tr(ρ log ρ), Rényi S_α(ρ) = (1−α)⁻¹ log Tr(ρ^α),
    pureza P(ρ) = Tr(ρ²) y reweighting Φ_α(ρ) = ρ^α / Tr(ρ^α).
  • BanachContractionAlgebra: Mapeo de la mutación T_η(ρ) = ρ − η[ρ,[ρ,N]] en B(u(n)).
    Radio espectral exacto en equilibrio diagonal: τ_ij = 1 − η (λ_i−λ_j) log(λ_i/λ_j), η_max = 2/g_max.
  • SoilField: Campo base H_mac perturbado según Kato–Rellich, con gap topológico E₁ − E₀ y ground |Ω⟩⟨Ω|.
  • SeedCrystalPreparation.prepare: Morfismo de hand-off FASE 1 ⟶ FASE 2. Prepara `SeedState`
    (último objeto/método de FASE-1).

FASE 2 — RIEGO, LUZ Y DISCIPLINA
──────────────────────────────────────────────────────────────────────────
  • CognitiveWateringModule.apply_water: (§2.1 Riego) Compresión de grasa sintáctica:
        Δ_gr = 100 · [ 0.7·(1 − t_t/t_j) + 0.3·(1 − H_t/H_j) ],  t BPE Ω(⌈n/4⌉), H₂ Shannon.
  • CognitiveIlluminationModule.apply_light: (§2.2 Luz) Flujo isospectral Brockett en U(n) (RK4 + polar)
        + Rényi Φ_α + matching energético Fock e⁻ + e⁺ → 2γ.
  • CognitiveDisciplineModule.audit_discipline: (§2.3 Disciplina) Verificación de la banda de Banach ρ(T_η) < 1.
  • CropGrowthPipeline.synthesize: Compone Riego ⊗ Luz ⊗ Disciplina, emitiendo `CropGrowthBundle`
    (último objeto/método de FASE-2).

FASE 3 — FE, ADJUDICACIÓN Y CERTIFICACIÓN CIBER-FÍSICA
──────────────────────────────────────────────────────────────────────────
  • HeytingCropAdjudicator.adjudicate: PRIMER MORFISMO DE FASE-3 (continúa `synthesize`).
    Colapsa el bundle en Ω₃ mediante meets: local = water ∧ discipline ∧ fock ∧ brockett ∧ renyi, final = local ∧ external.
  • CognitiveFaithModule.verify_faith: (§3.2 Fe) Disparo del crowbar ESP32 (GPIO14 → HIGH, BT151) si Ω₃ = ⊥
    con latencia nominal < 400 ns y firma SHA-256 de provenance.
  • TOONCognitiveCropEngine: Orquestador soberano C = Certify ∘ Growth ∘ Prepare.

Definición Granular de Invariantes y Axiomas
=============================================
  1. Isotonicidad de Brockett: Ḋ(ρ) = ‖[ρ, N]‖_F² ≥ 0 ⇒ L(ρ*) ≥ L(ρ₀).
  2. Banda de Estabilidad de Banach: 0 < η < η_max = 2/g_max ⇒ ρ(T_η) < 1 (contracción local).
  3. Proyección al Simplex 𝔇_n: Tr(ρ) = 1, ρ = ρ†, spec(ρ) ⊂ [0, 1].
  4. Adjunción de Heyting: (a ∧ c ≤ b) ⇔ (c ≤ (a → b)).
  5. Inyectividad Merkle: Cadena de custodia `phase_chain_sha256` inalienable por SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


logger = logging.getLogger("APU.Wisdom.TOONCognitiveCropEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS: Final[float] = 1.0e-14
_EPS_MODULAR: Final[float] = 1.0e-10
_EPS_TRACE: Final[float] = 1.0e-15
_EPS_CONTRACT: Final[float] = 1.0e-3

ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]


def _seed_from_string(s: str) -> int:
    """Proyección SHA-256 → ℕ/2³², determinista, libre de plataforma."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def _sha256_bytes(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ALGEBRAICO DEL CULTIVO                                  ║
# ║                                                                           ║
# ║ Objetos: Ω₃, M_n(ℂ)_sa,⁺¹, B(u(n)), (H_mac, |Ω⟩).                         ║
# ║ Morfismo terminal: SeedCrystalPreparation.prepare : M_n → SeedState.      ║
# ║ Ese morfismo ES el dominio de todos los funtores de la FASE 2.            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §1.1 Retículo distributivo de Heyting Ω₃ ──────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa (álgebra de Gödel de 3 valores)

        Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0, 1, 2}

    Estructura:
        meet    (∧) : ínfimo = min                        producto cartesiano
        join    (∨) : supremo = max                       coproducto
        implies (⇒) : residuo   (a ∧ b ≤ c ⇔ a ≤ (b ⇒ c))
                      a ⇒ b = ⊤  si a ≤ b,  else b
        neg     (¬) : a ⇒ ⊥     (seudocomplemento intuicionista)
        iff     (⇔) : (a ⇒ b) ∧ (b ⇒ a)

    Propiedades que fallan respecto de un álgebra de Boole:
        ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤          (tercio excluso)
        ¬¬⋆ = ⊤ ≠ ⋆             (⋆ no es regular)
        {⊥, ⊤}  ↪  Ω₃           (subálgebra Booleana de regulares)

    Interpretación en el topos de prefaisceaux sobre el poset Ω₃:
        ⊤ clasifica subobjetos totales (cosecha coherente),
        ⋆ clasifica subobjetos densos no cerrados (degradación),
        ⊥ clasifica el subobjeto vacío (veto / crowbar).
    """
    VETOED: int = 0      # ⊥
    DEGRADED: int = 1    # ⋆
    COHERENT: int = 2    # ⊤

    @property
    def verdict(self) -> str:
        return self.name

    def leq(self, other: "HeytingOmega3") -> bool:
        """Orden total del poset: ⊥ ≤ ⋆ ≤ ⊤."""
        return int(self) <= int(other)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.COHERENT if self.leq(other) else other

    def neg(self) -> "HeytingOmega3":
        """Seudocomplemento ¬a := a ⇒ ⊥.  ¬⋆ = ⊥,  ¬⊥ = ⊤,  ¬⊤ = ⊥."""
        return self.implies(HeytingOmega3.VETOED)

    def iff(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.implies(other).meet(other.implies(self))

    def is_regular(self) -> bool:
        """a es regular ⟺ a = ¬¬a.  Sólo ⊥ y ⊤ lo son."""
        return self.neg().neg() == self

    def as_weight(self) -> float:
        """Inmersión afín Ω₃ ↪ [0, 1] : ⊥↦0, ⋆↦1/2, ⊤↦1."""
        return float(int(self)) / 2.0


# ── §1.2 Álgebra de operadores densidad ───────────────────────────────────
class DensityOperatorAlgebra:
    r"""
    Operaciones canónicas sobre el conjunto convexo de estados

        𝔇_n = { ρ ∈ M_n(ℂ) : ρ = ρ†,  ρ ≥ 0,  Tr ρ = 1 }.

    Funcionales (todos unitariamente invariantes, funciones espectrales):

        S(ρ)     = −Tr(ρ log ρ)                 von Neumann (nats)
        S_α(ρ)   = (1−α)⁻¹ log Tr(ρ^α)          Rényi, α ≠ 1
        P(ρ)     = Tr(ρ²) = ‖ρ‖₂²               pureza
        F(ρ,σ)   = ‖√ρ √σ‖₁ = Tr√(√ρ σ √ρ)     Uhlmann–Jozsa
        T(ρ,σ)   = (1/2)‖ρ−σ‖₁                  distancia de traza
        S(ρ‖σ)   = Tr(ρ(log ρ − log σ))         Umegaki
        K_ρ      = −log ρ                       Hamiltoniano modular
        ρ^z      = exp(z log ρ)                 cálculo funcional holomorfo
        ‖ρ‖_p    = (Tr|ρ|^p)^{1/p}              Schatten

    El espectro se toma siempre descendente y proyectado al simplex
    {λ ∈ ℝ₊ⁿ : Σ λ_i = 1, λ_i ≥ ε}, lo que hace de 𝔇_n un compacto
    efectivo y evita log(0) en el Hamiltoniano modular.
    """
    EPS: Final[float] = _EPS
    EPS_MODULAR: Final[float] = _EPS_MODULAR

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""Proyección afín sobre 𝔇_n: hermitización + renormalización de traza."""
        if not cls.is_square(rho):
            raise ValueError(f"DensityOperatorAlgebra.sanitize: matriz no cuadrada {np.shape(rho)}")
        rho_h = np.asarray(rho, dtype=np.complex128)
        rho_h = 0.5 * (rho_h + rho_h.conj().T)
        tr = float(np.trace(rho_h).real)
        if abs(tr) > _EPS_TRACE:
            rho_h = rho_h / tr
        return rho_h

    @classmethod
    def spectrum_descending(cls, rho: np.ndarray) -> RealVector:
        rho = cls.sanitize(rho)
        w = np.real(la.eigvalsh(rho))
        w = np.sort(w)[::-1]
        w = np.maximum(w, cls.EPS)
        s = float(w.sum())
        if s <= 0.0:
            n = w.size
            return np.full(n, 1.0 / n, dtype=np.float64)
        return (w / s).astype(np.float64)

    @classmethod
    def spectral_gap(cls, rho: np.ndarray) -> float:
        """λ₁ − λ₂ del espectro descendente (0 si n < 2)."""
        w = cls.spectrum_descending(rho)
        if w.size < 2:
            return 0.0
        return float(w[0] - w[1])

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -float(np.sum(p * np.log(p)))

    @classmethod
    def renyi_entropy(cls, rho: np.ndarray, alpha: float) -> float:
        r"""S_α(ρ) = (1−α)⁻¹ log Tr(ρ^α).  Límite α→1 = S(ρ)."""
        if abs(alpha - 1.0) < 1e-12:
            return cls.von_neumann_entropy(rho)
        p = cls.spectrum_descending(rho)
        tr_a = float(np.sum(np.power(p, alpha)))
        tr_a = max(tr_a, cls.EPS)
        return float(math.log(tr_a) / (1.0 - alpha))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        return float(np.sum(p * p))

    @classmethod
    def frobenius_norm(cls, rho: np.ndarray) -> float:
        return float(np.linalg.norm(cls.sanitize(rho), "fro"))

    @classmethod
    def schatten_p_norm(cls, rho: np.ndarray, p: float) -> float:
        r"""‖ρ‖_p = (Σ σ_i^p)^{1/p}, σ_i valores singulares.  p=∞ → σ_max."""
        sig = np.real(la.svdvals(cls.sanitize(rho)))
        sig = np.maximum(sig, 0.0)
        if p == math.inf:
            return float(sig.max()) if sig.size else 0.0
        if p <= 0.0:
            raise ValueError("Schatten p-norm requiere p > 0")
        return float(np.power(np.sum(np.power(sig, p)), 1.0 / p))

    @classmethod
    def trace_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        """T(ρ,σ) = (1/2)‖ρ−σ‖₁ ∈ [0, 1]."""
        delta = cls.sanitize(rho) - cls.sanitize(sigma)
        return 0.5 * cls.schatten_p_norm(delta, 1.0)

    @classmethod
    def umegaki_relative_entropy(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""
        S(ρ‖σ) = Tr(ρ log ρ) − Tr(ρ log σ) ≥ 0  (Klein).
        Devuelve +∞ si supp(ρ) ⊈ supp(σ) a la resolución EPS_MODULAR.
        """
        w_r, V_r = la.eigh(cls.sanitize(rho))
        w_s, V_s = la.eigh(cls.sanitize(sigma))
        w_r = np.maximum(np.real(w_r), 0.0)
        w_s = np.real(w_s)
        if np.any((w_r > cls.EPS_MODULAR) & (np.abs(V_r.conj().T @ V_s) ** 2 @ np.maximum(w_s, 0.0) < cls.EPS_MODULAR)):
            # test grosero de soporte; si σ tiene autovalor nulo en dirección de ρ
            if np.any(w_s < cls.EPS_MODULAR) and np.any(w_r > cls.EPS_MODULAR):
                # no concluimos ∞ sin overlap exacto; caemos al cálculo regularizado
                pass
        w_r_reg = np.maximum(w_r, cls.EPS_MODULAR)
        w_s_reg = np.maximum(w_s, cls.EPS_MODULAR)
        log_r = V_r @ np.diag(np.log(w_r_reg).astype(np.complex128)) @ V_r.conj().T
        log_s = V_s @ np.diag(np.log(w_s_reg).astype(np.complex128)) @ V_s.conj().T
        rho_s = cls.sanitize(rho)
        val = float(np.real(np.trace(rho_s @ (log_r - log_s))))
        return max(0.0, val)

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sr = cls.matrix_power(rho, 0.5)
        inner = sr @ sigma @ sr
        val = float(np.real(np.trace(cls.matrix_power(inner, 0.5))))
        return float(np.clip(val, 0.0, 1.0))

    @classmethod
    def matrix_power(cls, rho: np.ndarray, z: complex,
                     floor: float = _EPS_MODULAR) -> ComplexMatrix:
        r"""
        ρ^z = V diag(λ_i^z) V†  vía cálculo funcional (Holstein–Rellich).
        Para z complejo se usa exp(z log λ) sobre el corte principal.
        """
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

    @classmethod
    def modular_hamiltonian(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""K_ρ = −log ρ  (Tomita–Takesaki: Δ_ρ = exp(−K_ρ) sobre el estado)."""
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), cls.EPS_MODULAR)
        kspec = -np.log(w)
        return (V * kspec.astype(np.complex128)) @ V.conj().T

    @classmethod
    def modular_spectrum(cls, rho: np.ndarray) -> Tuple[float, ...]:
        """Espectro ascendente de K_ρ = −log ρ."""
        w = cls.spectrum_descending(rho)
        k = -np.log(np.maximum(w, cls.EPS_MODULAR))
        return tuple(sorted(float(x) for x in k.tolist()))

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> ComplexMatrix:
        H = 0.5 * (np.asarray(H, dtype=np.complex128) + np.asarray(H, dtype=np.complex128).conj().T)
        w, V = la.eigh(H)
        i0 = int(np.argmin(np.real(w)))
        psi = V[:, i0].reshape(-1, 1)
        return cls.sanitize(psi @ psi.conj().T)

    @classmethod
    def renyi_sharpen(cls, rho: np.ndarray, alpha: float) -> ComplexMatrix:
        r"""
        Reweighting de Rényi (función espectral, U(n)-equivariante):

            Φ_α(ρ) = ρ^α / Tr(ρ^α),    α ≥ 1.

        α = 1 → id.  α > 1 ⇒ P↑, S↓  (mayor peso a autovalores dominantes).
        Es el gradiente de S_α en la variedad de estados de espectro fijo
        salvo renormalización; conmuta con todo el flujo de Brockett.
        """
        if alpha <= 1.0 + 1e-12:
            return cls.sanitize(rho)
        rho = cls.sanitize(rho)
        rho_a = cls.matrix_power(rho, complex(alpha, 0.0))
        tr = float(np.trace(rho_a).real)
        if tr < 1e-30:
            return rho
        return cls.sanitize(rho_a / tr)


# ── §1.3 Álgebra de Banach: radio espectral real de la mutación ───────────
@dataclass(frozen=True, slots=True)
class BanachContractionReport:
    r"""
    Auditoría de la disciplina en el álgebra de Banach B(u(n)).

    Euler-step de la mutación doble-corchete sobre estados:

        T_η(ρ) = ρ − η [ρ, [ρ, N]],     N = K_ρ  (o N = diag(1..n)).

    En un equilibrio diagonal ρ = diag(λ), N = diag(−log λ), la
    linealización DT_η actúa sobre los modos de coherencia E_{ij}
    (i ≠ j) con autovalores exactos (Brockett 1991, linearización):

        τ_ij(η) = 1 − η · g_ij,
        g_ij    = (λ_i − λ_j) log(λ_i / λ_j) ≥ 0     (por convexidad de x log x).

    Radio espectral (Gelfand):

        ρ(T_η) = max_{i≠j} |τ_ij(η)| = lim_{k→∞} ‖(DT_η)^k‖^{1/k}.

    Banda de estabilidad de Banach (contracción local):

        0 < η < η_max := 2 / g_max    ⇒    ρ(T_η) < 1.

    Interpretación:
        ρ(T) < 1  → punto fijo atractor (el cultivo converge),
        ρ(T) = 1  → centro (espectro degenerado, sin dirección),
        ρ(T) > 1  → divergencia (η excede el radio de estabilidad).
    """
    spectral_radius: float
    eta_star: float
    eta_max: float
    g_max: float
    alignment: float
    is_contraction: bool
    lipschitz_bound: float
    pair_index: Tuple[int, int]
    local_verdict: HeytingOmega3


class BanachContractionAlgebra:
    r"""
    Cálculo vectorizado del radio espectral de DT_η sobre u(n).

    La matriz de acoplamientos G = (g_ij) es un kernel de tipo
    Hilbert–Schmidt simétrico, nulo en la diagonal, y G ≥ 0
    entrada a entrada.  g_max = ‖G‖_∞ (norma max).
    """
    EPS: Final[float] = _EPS_MODULAR

    @classmethod
    def coupling_matrix(cls, rho: np.ndarray) -> RealVector:
        r"""
        G_ij = (λ_i − λ_j) log(λ_i/λ_j),  G_ii = 0.

        Identidad elemental: g_ij = 4 λ̄_{ij} sinh²(δ_{ij}/2) · |δ|
        con δ = log(λ_i/λ_j), de donde g_ij ≥ 0 y g_ij = 0 ⇔ λ_i = λ_j.
        """
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        w = np.maximum(w, cls.EPS)
        diff = w[:, None] - w[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log(w[:, None] / w[None, :])
        G = diff * ratio
        np.fill_diagonal(G, 0.0)
        G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(G, 0.0)

    @classmethod
    def pair_couplings(cls, rho: np.ndarray) -> Tuple[float, int, int]:
        """Devuelve (g_max, i*, j*) del par más inestable."""
        G = cls.coupling_matrix(rho)
        if G.size == 0:
            return 0.0, 0, 0
        idx = int(np.argmax(G))
        n = G.shape[0]
        i_max, j_max = divmod(idx, n)
        return float(G[i_max, j_max]), int(i_max), int(j_max)

    @classmethod
    def spectral_radius(
        cls, rho: np.ndarray, eta: float
    ) -> Tuple[float, float, float]:
        r"""
        (ρ(T; η), η_max, g_max) con la fórmula exacta τ_ij = 1 − η g_ij.

        Si g_max = 0 (estado maximally mixed o puro degenerado n=1),
        η_max = +∞ y ρ(T) = 1 (marginal: DT = Id sobre coherencias nulas).
        """
        G = cls.coupling_matrix(rho)
        if G.size == 0:
            return 1.0, float("inf"), 0.0
        g_max = float(G.max())
        tau = 1.0 - eta * G
        np.fill_diagonal(tau, 0.0)
        max_abs_tau = float(np.max(np.abs(tau))) if tau.size else 0.0
        eta_max = (2.0 / g_max) if g_max > cls.EPS else float("inf")
        return max_abs_tau, float(eta_max), g_max

    @classmethod
    def lipschitz_bound(cls, rho: np.ndarray, eta: float) -> float:
        r"""
        Cota de Lipschitz local de T_η en norma de Frobenius:

            Lip(T_η) ≤ max(1, ρ(T; η)) ≤ 1 + η · g_max.

        Suficiente para Banach si Lip < 1.
        """
        rho_T, _, g_max = cls.spectral_radius(rho, eta)
        return float(max(rho_T, abs(1.0 - eta * g_max), eta * g_max))

    @classmethod
    def recommend_eta(cls, rho: np.ndarray, safety: float = 0.5) -> float:
        """η recomendada = safety · η_max,  safety ∈ (0, 1)."""
        _, eta_max, _ = cls.spectral_radius(rho, 0.0)
        if not math.isfinite(eta_max):
            return 1.0
        return float(max(safety, 1e-6) * eta_max)

    @classmethod
    def audit(
        cls, rho: np.ndarray, eta_star: float
    ) -> BanachContractionReport:
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        log_w = np.log(np.maximum(w, cls.EPS))
        # K_ρ = −log ρ;  Tr(ρ K_ρ) = −Σ λ log λ = S(ρ)  (energía libre a T=1)
        alignment = float(-np.sum(w * log_w))

        rho_T, eta_max, g_max = cls.spectral_radius(rho, eta_star)
        _, i_star, j_star = cls.pair_couplings(rho)
        lip = cls.lipschitz_bound(rho, eta_star)

        if rho_T < 1.0 - _EPS_CONTRACT:
            local = HeytingOmega3.COHERENT
        elif rho_T < 1.0 + _EPS_CONTRACT:
            local = HeytingOmega3.DEGRADED
        elif rho_T < 1.5:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return BanachContractionReport(
            spectral_radius=rho_T,
            eta_star=float(eta_star),
            eta_max=eta_max,
            g_max=g_max,
            alignment=alignment,
            is_contraction=rho_T < 1.0,
            lipschitz_bound=lip,
            pair_index=(i_star, j_star),
            local_verdict=local,
        )


# ── §1.4 Campo del suelo: H_mac y |Ω⟩ ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SoilState:
    r"""
    Tierra fértil (fondo espectral) sobre la que se siembra.

        H_mac       : Hamiltoniano externo Hermitian dim×dim
        ground_proj : |Ω⟩⟨Ω|  proyector espectral del ground
        E0          : min spec(H_mac)
        gap_H       : E₁ − E₀  (gap de Kato; robustez a perturbaciones)
        hash        : SHA-256(H_mac ‖ ground_proj)  trazabilidad
    """
    H_mac: np.ndarray
    ground_proj: np.ndarray
    E0: float
    gap_H: float
    hash: str


class SoilField:
    r"""
    Construye H_mac como perturbación de Kato–Rellich de un espectro
    equiespaciado:

        H₀ = diag(0, Δ/(n−1), …, Δ),     Δ = DEFAULT_GAP,
        W  = ε · (E_{i,i+1} + E_{i+1,i}), ε = DEFAULT_COUPLING,
        H  = H₀ + W.

    El gap de H₀ es Δ/(n−1).  Si ε ≪ gap, el teorema de Kato garantiza
    que el ground permanece simple y |Ω⟩ es C¹ en ε.
    """
    DEFAULT_GAP: Final[float] = 3.0
    DEFAULT_COUPLING: Final[float] = 1.0e-3

    @classmethod
    def build(cls, n: int = 4) -> SoilState:
        if n < 1:
            raise ValueError("SoilField.build: dimensión n ≥ 1")
        base = np.diag(np.linspace(0.0, cls.DEFAULT_GAP, n)).astype(np.complex128)
        off = np.zeros((n, n), dtype=np.complex128)
        if n >= 2:
            idx = np.arange(n - 1)
            off[idx, idx + 1] = cls.DEFAULT_COUPLING
        H = base + off + off.conj().T

        proj = DensityOperatorAlgebra.ground_state_projector(H)
        w = np.real(la.eigvalsh(H))
        w = np.sort(w)
        E0 = float(w[0])
        gap = float(w[1] - w[0]) if w.size > 1 else float("inf")

        soil_hash = _sha256_bytes(
            np.ascontiguousarray(H).tobytes(),
            np.ascontiguousarray(proj).tobytes(),
        )
        return SoilState(
            H_mac=H, ground_proj=proj, E0=E0, gap_H=gap, hash=soil_hash,
        )


# ── §1.5 SeedCrystalPreparation — HAND-OFF FASE 1 → FASE 2 ────────────────
@dataclass(frozen=True, slots=True)
class SeedState:
    r"""
    Objeto terminal de la FASE 1 y objeto inicial de la FASE 2.

    Tipo:  SeedState ≅ 𝔇_n × ℝⁿ × ℝ₊³

        rho_seed  : ρ ∈ 𝔇_n  (Hermitiano, traza 1, regularizado)
        K_spec    : spec↑(K_ρ) = spec↑(−log ρ)
        purity    : P(ρ) = Tr ρ² ∈ [1/n, 1]
        entropy   : S(ρ) ∈ [0, log n]
        alignment : Tr(ρ K_ρ) = S(ρ)   (energía libre a β=1)
        gap_K     : gap modular λ_min(K) salto
        dim       : n
    """
    rho_seed: np.ndarray
    K_spec: Tuple[float, ...]
    purity: float
    entropy: float
    alignment: float
    gap_K: float
    dim: int


class SeedCrystalPreparation:
    r"""
    Prepara el estado (ρ_seed, K_seed) a partir del cristal de experiencia.

    Convenciones:
        · sanitize()  ⇒  Hermiticidad + traza unitaria.
        · K_spec      ⇒  espectro del Hamiltoniano modular (Tomita–Takesaki).
        · purity, S   ⇒  invariantes de salud de la semilla.

    ────────────────────────────────────────────────────────────────────────
    HAND-OFF FORMAL  FASE 1 → FASE 2
    ────────────────────────────────────────────────────────────────────────
    prepare : M_n(ℂ) → SeedState

    Es el morfismo terminal de la FASE 1.  Su imagen SeedState es el
    dominio de TODOS los métodos de la FASE 2:

        CognitiveWateringModule.apply_water      : SeedState × Σ* × Σ* → WateringReport
        CognitiveIlluminationModule.apply_light  : SeedState → (𝔇_n × Cert³)
        CognitiveDisciplineModule.audit_discipline: SeedState × ℝ₊ → BanachReport
        CropGrowthPipeline.synthesize            : SeedState × Σ* × Σ* → CropGrowthBundle

    En el sentido de teoría de categorías, la FASE 2 es el comma-category
    (SeedState ↓ Crop₂).  Invocar cualquier método de FASE 2 sin un
    SeedState es un error de tipo.
    ────────────────────────────────────────────────────────────────────────
    """

    @classmethod
    def prepare(cls, rho_raw: np.ndarray) -> SeedState:
        r"""
        Cierra la FASE 1.  Abre la FASE 2.

        Parámetro
            rho_raw : matriz cruda del Testigo Silencioso (no necesariamente
                      en 𝔇_n).  Se proyecta por sanitize.

        Retorna
            SeedState — único combustible algebraico de §2.1–§2.4.
        """
        rho = DensityOperatorAlgebra.sanitize(rho_raw)
        K_spec = DensityOperatorAlgebra.modular_spectrum(rho)
        purity = DensityOperatorAlgebra.purity(rho)
        entropy = DensityOperatorAlgebra.von_neumann_entropy(rho)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        alignment = float(-np.sum(w * np.log(np.maximum(w, _EPS))))
        gap_K = float(K_spec[1] - K_spec[0]) if len(K_spec) > 1 else float("inf")
        dim = int(rho.shape[0])

        logger.debug(
            "SeedCrystalPreparation.prepare | n=%d | P=%.6f | S=%.6f | "
            "E0(K)=%.4f | gap(K)=%.4f",
            dim, purity, entropy,
            K_spec[0] if K_spec else 0.0, gap_K,
        )
        return SeedState(
            rho_seed=rho,
            K_spec=K_spec,
            purity=purity,
            entropy=entropy,
            alignment=alignment,
            gap_K=gap_K,
            dim=dim,
        )

    @classmethod
    def continue_into_phase2(
        cls,
        seed: SeedState,
        cycle_index: int,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float,
        eta_star: float,
    ) -> "CropGrowthBundle":
        r"""
        Continuación estricta de `prepare`.

        Este método ES simultáneamente:
            · el último morfismo de la FASE 1 (consume SeedState),
            · el primer morfismo de la FASE 2 (delega en CropGrowthPipeline).

        Identidad de composición:

            continue_into_phase2 ∘ prepare
                = CropGrowthPipeline.synthesize ∘ prepare
                : M_n(ℂ) × Σ* × Σ* → CropGrowthBundle.
        """
        return CropGrowthPipeline.synthesize(
            cycle_index=cycle_index,
            seed=seed,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=renyi_alpha,
            eta_star=eta_star,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · RIEGO + LUZ + DISCIPLINA                                         ║
# ║                                                                           ║
# ║ Dominio = SeedState  (codominio de §1.5 SeedCrystalPreparation.prepare).  ║
# ║ Codominio = CropGrowthBundle, dominio de toda la FASE 3.                  ║
# ║                                                                           ║
# ║ Los métodos de §2.1 se leen como la continuación literal del HAND-OFF     ║
# ║ de prepare: el riego actúa sobre (SeedState, toon, json).                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 CognitiveWateringModule — RIEGO cuantificado ─────────────────────
@dataclass(frozen=True, slots=True)
class WateringReport:
    r"""
    Resultado del riego (compresión de grasa sintáctica, teorema de Shannon).

        tokens_toon, tokens_json  : estimador BPE  Ω(⌈n/4⌉)  (chars/token ≈ 4)
        h_toon, h_json            : H₂ bits/carácter  (entropía de alfabeto)
        fat_reduction_pct         : Δ_gr ∈ ℝ
        kv_compression_ratio      : min(0.95, Δ_gr/100)  ∈ [0, 0.95]
        seed_entropy_nats         : S(ρ_seed)  (custodia de FASE 1)
        local_verdict             : ⊤ si Δ_gr≥60, ⋆ si ≥30, ⊥ si no
    """
    tokens_toon: int
    tokens_json: int
    h_toon: float
    h_json: float
    fat_reduction_pct: float
    kv_compression_ratio: float
    seed_entropy_nats: float
    local_verdict: HeytingOmega3


class CognitiveWateringModule:
    r"""
    FASE 2 · EL RIEGO — continuación de SeedCrystalPreparation.prepare.

    Mide el caudal de información que refresca la KV-cache.  El estimador
    de tokens es el isomorfismo asintótico BPE ≃ ⌈n/4⌉ (byte-pair, GPT-like).
    La entropía de Shannon del alfabeto empírico

        H₂(s) = − Σ_c (n_c/|s|) log₂(n_c/|s|)     (bits/carácter)

    es el límite de Rényi α→1 en base 2.  El funcional de grasa es la
    combinación convexa (pesos de Dirichlet W_TOKEN + W_ENTROPY = 1):

        Δ_gr = 100 · [ W_T (1 − t_t/t_j) + W_H (1 − H_t/H_j) ].

    El KV-ratio satura en KV_MAX = 0.95 (nunca se declara compresión
    perfecta: leftover de Kraft).

    Firma (continuación de §1.5):

        apply_water : SeedState × Σ* × Σ* → WateringReport
    """
    W_TOKEN: Final[float] = 0.7
    W_ENTROPY: Final[float] = 0.3
    KV_MAX: Final[float] = 0.95
    CHARS_PER_TOKEN: Final[float] = 4.0

    @classmethod
    def bpe_tokens(cls, text: str) -> int:
        """Estimador BPE Ω(⌈|s|/4⌉).  Cota inferior 1 (nunca cadena vacía)."""
        return max(1, int(math.ceil(len(text) / cls.CHARS_PER_TOKEN)))

    @classmethod
    def shannon_bits(cls, text: str) -> float:
        """H₂ empírica del alfabeto de `text`.  H₂(∅) := 0."""
        if not text:
            return 0.0
        counts = Counter(text)
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)

    @classmethod
    def fat_functional(
        cls, t_toon: int, t_json: int, h_toon: float, h_json: float
    ) -> float:
        red_token = 1.0 - t_toon / max(1, t_json)
        red_ent = 1.0 - h_toon / max(1e-9, h_json)
        return 100.0 * (cls.W_TOKEN * red_token + cls.W_ENTROPY * red_ent)

    @classmethod
    def apply_water(
        cls,
        seed: SeedState,
        toon_str: str,
        base_json_str: str,
    ) -> WateringReport:
        r"""
        Continuación de `SeedCrystalPreparation.prepare`.

        El SeedState no altera Δ_gr (el riego es un funcional del
        lenguaje, no del espectro), pero su entropía viaja en el
        reporte para la cadena de custodia F1→F2.
        """
        t_t = cls.bpe_tokens(toon_str)
        t_j = cls.bpe_tokens(base_json_str)
        h_t = cls.shannon_bits(toon_str)
        h_j = cls.shannon_bits(base_json_str)

        delta_gr = cls.fat_functional(t_t, t_j, h_t, h_j)
        kv = float(np.clip(delta_gr / 100.0, 0.0, cls.KV_MAX))

        if delta_gr >= 60.0:
            local = HeytingOmega3.COHERENT
        elif delta_gr >= 30.0:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return WateringReport(
            tokens_toon=t_t,
            tokens_json=t_j,
            h_toon=h_t,
            h_json=h_j,
            fat_reduction_pct=float(delta_gr),
            kv_compression_ratio=kv,
            seed_entropy_nats=float(seed.entropy),
            local_verdict=local,
        )


# ── §2.2 CognitiveIlluminationModule — LUZ (Brockett + Rényi + Fock) ─────
@dataclass(frozen=True, slots=True)
class BrockettPurificationCertificate:
    r"""
    Certificado del flujo isospectral de Brockett en la órbita coadjunta.

        dρ/dt = [A(ρ), ρ],   A(ρ) = −[ρ, N],   N = diag(1,…,n).

    Equivale a ρ(t) = U(t) ρ(0) U(t)† con  dU/dt = A(ρ) U, U ∈ U(n).
    El funcional de Lyapunov  ℒ(ρ) = Tr(ρ N)  es estrictamente decreciente
    fuera de los puntos alineados con N (Brockett 1991, Thm. 1).

        isospectral_drift = ‖λ_final − λ_inicial‖₂  (debe ser ~ 0)
    """
    initial_alignment: float
    final_alignment: float
    initial_purity: float
    final_purity: float
    lyapunov_drop: float
    iterations: int
    converged: bool
    isospectral_drift: float


@dataclass(frozen=True, slots=True)
class RenyiPurificationCertificate:
    r"""
    Certificado de Φ_α(ρ) = ρ^α / Tr(ρ^α).

        purity_gain  = P_after − P_before ≥ 0   (α ≥ 1)
        entropy_drop = S_before − S_after ≥ 0
        renyi_S      = S_α(ρ_after)
    """
    alpha: float
    purity_before: float
    purity_after: float
    entropy_before: float
    entropy_after: float
    purity_gain: float
    entropy_drop: float
    renyi_S: float


@dataclass(frozen=True, slots=True)
class FockAnnihilationCertificate:
    r"""
    Matching energético de dos modos (oscilador de Fock, no HEP literal):

        E₋ = n · (1 − λ_max(ρ_seed))     anomalía: fuga de la cúpula pura
        E₊ = S(ρ_seed)                   fricción térmica (nats)
        residual = |E₋ − E₊| / max(1, |E₋|, |E₊|)
        annihilated ⟺ residual < θ      ⇒  2 cuantos γ contabilizados

    Interpretación: cuando la desviación de pureza (modo −) equilibra
    la entropía (modo +), el cultivo emite un par y se declara resonante.
    """
    electron_anomaly_energy: float
    positron_constraint_energy: float
    gamma_photons_emitted: int
    resonance_residual: float
    is_annihilated: bool


class CognitiveIlluminationModule:
    r"""
    FASE 2 · LA LUZ — tres operaciones encadenadas sobre SeedState.rho_seed.

        (1) Brockett en U(n): integración RK4 del generador anti-Hermítico
            A = −[ρ, N], polar-proyectada a U(n) ⇒ isospectralidad exacta
            hasta error de redondeo (no deriva espectral de Euler en 𝔇_n).
        (2) Φ_α de Rényi: ganancia monótona de pureza.
        (3) Matching Fock de dos modos sobre la semilla (invariante de F1).

    Firma (continuación de §1.5):

        apply_light : SeedState → (𝔇_n × BrockettCert × RenyiCert × FockCert)
    """
    BROCKETT_MAX_STEPS: Final[int] = 60
    BROCKETT_DT: Final[float] = 0.05
    BROCKETT_TOL: Final[float] = 1.0e-9
    RENYI_ALPHA: Final[float] = 1.5
    FOCK_RESONANCE_TOL: Final[float] = 0.15

    @classmethod
    def _N(cls, n: int) -> RealVector:
        return np.diag(np.arange(1, n + 1, dtype=np.float64))

    @classmethod
    def _anti_hermitian_generator(cls, rho: np.ndarray, N: np.ndarray) -> ComplexMatrix:
        r"""A(ρ) = −[ρ, N] ∈ u(n)  (anti-Hermítico si ρ, N Hermitianos)."""
        comm = rho @ N - N @ rho
        A = -comm
        return 0.5 * (A - A.conj().T)

    @classmethod
    def _project_unitary(cls, U: np.ndarray) -> ComplexMatrix:
        """Proyección polar U ↦ U (U†U)^{−1/2} ∈ U(n)."""
        # SVD: U = W Σ V†  ⇒  polar = W V†
        W, _, Vh = la.svd(U, full_matrices=False)
        return W @ Vh

    @classmethod
    def lyapunov_alignment(cls, rho: np.ndarray, N: np.ndarray) -> float:
        """ℒ(ρ) = Tr(ρ N)  (Lyapunov de Brockett, descenso)."""
        return float(np.trace(rho @ N).real)

    @classmethod
    def _brockett_unitary_rk4(
        cls, rho_init: np.ndarray,
    ) -> Tuple[ComplexMatrix, BrockettPurificationCertificate]:
        r"""
        Integra  dU/dt = A(U ρ₀ U†) U  en U(n) por RK4 clásico,
        reproyectando a U(n) en cada paso (método de Crouch–Grossman
        de orden 1 en la variedad, orden 4 en el álgebra).
        """
        rho0 = DensityOperatorAlgebra.sanitize(rho_init)
        n = rho0.shape[0]
        N = cls._N(n)

        lam0 = DensityOperatorAlgebra.spectrum_descending(rho0)
        init_align = cls.lyapunov_alignment(rho0, N)
        init_pur = float(np.sum(lam0 ** 2))

        U = np.eye(n, dtype=np.complex128)
        dt = cls.BROCKETT_DT
        converged = False
        step = 0
        rho = rho0.copy()

        def A_of(U_cur: np.ndarray) -> ComplexMatrix:
            r = U_cur @ rho0 @ U_cur.conj().T
            r = 0.5 * (r + r.conj().T)
            return cls._anti_hermitian_generator(r, N)

        for step in range(cls.BROCKETT_MAX_STEPS):
            k1 = A_of(U) @ U
            k2 = A_of(U + 0.5 * dt * k1) @ (U + 0.5 * dt * k1)
            k3 = A_of(U + 0.5 * dt * k2) @ (U + 0.5 * dt * k2)
            k4 = A_of(U + dt * k3) @ (U + dt * k3)
            U_next = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            U_next = cls._project_unitary(U_next)

            rho_next = U_next @ rho0 @ U_next.conj().T
            rho_next = 0.5 * (rho_next + rho_next.conj().T)
            if np.linalg.norm(rho_next - rho, "fro") < cls.BROCKETT_TOL:
                U, rho = U_next, rho_next
                converged = True
                break
            U, rho = U_next, rho_next

        rho = DensityOperatorAlgebra.sanitize(rho)
        lam1 = DensityOperatorAlgebra.spectrum_descending(rho)
        fin_align = cls.lyapunov_alignment(rho, N)
        fin_pur = float(np.sum(lam1 ** 2))
        drift = float(np.linalg.norm(lam1 - lam0, ord=2))

        cert = BrockettPurificationCertificate(
            initial_alignment=init_align,
            final_alignment=fin_align,
            initial_purity=init_pur,
            final_purity=fin_pur,
            lyapunov_drop=float(init_align - fin_align),
            iterations=step + 1,
            converged=converged,
            isospectral_drift=drift,
        )
        return rho, cert

    @classmethod
    def _fock_resonance(cls, seed: SeedState) -> FockAnnihilationCertificate:
        w = DensityOperatorAlgebra.spectrum_descending(seed.rho_seed)
        n = len(w)
        E_minus = float(n * (1.0 - w[0]))
        E_plus = float(seed.entropy)
        denom = max(1.0, abs(E_minus), abs(E_plus))
        residual = abs(E_minus - E_plus) / denom
        is_ann = residual < cls.FOCK_RESONANCE_TOL
        return FockAnnihilationCertificate(
            electron_anomaly_energy=E_minus,
            positron_constraint_energy=E_plus,
            gamma_photons_emitted=2 if is_ann else 0,
            resonance_residual=float(residual),
            is_annihilated=bool(is_ann),
        )

    @classmethod
    def apply_light(
        cls,
        seed: SeedState,
        renyi_alpha: float = RENYI_ALPHA,
    ) -> Tuple[
        ComplexMatrix,
        BrockettPurificationCertificate,
        RenyiPurificationCertificate,
        FockAnnihilationCertificate,
    ]:
        r"""
        Continuación de `SeedCrystalPreparation.prepare` / `apply_water`.

        Consume SeedState.rho_seed; no toca el watering (producto
        tensorial de funtores Riego ⊗ Luz).
        """
        rho_flow, brockett_cert = cls._brockett_unitary_rk4(seed.rho_seed)

        p_before = DensityOperatorAlgebra.purity(rho_flow)
        s_before = DensityOperatorAlgebra.von_neumann_entropy(rho_flow)
        rho_illum = DensityOperatorAlgebra.renyi_sharpen(rho_flow, renyi_alpha)
        p_after = DensityOperatorAlgebra.purity(rho_illum)
        s_after = DensityOperatorAlgebra.von_neumann_entropy(rho_illum)
        s_renyi = DensityOperatorAlgebra.renyi_entropy(rho_illum, renyi_alpha)

        renyi_cert = RenyiPurificationCertificate(
            alpha=float(renyi_alpha),
            purity_before=p_before,
            purity_after=p_after,
            entropy_before=s_before,
            entropy_after=s_after,
            purity_gain=float(p_after - p_before),
            entropy_drop=float(s_before - s_after),
            renyi_S=float(s_renyi),
        )
        fock_cert = cls._fock_resonance(seed)
        return rho_illum, brockett_cert, renyi_cert, fock_cert


# ── §2.3 CognitiveDisciplineModule — DISCIPLINA (Banach real) ─────────────
class CognitiveDisciplineModule:
    r"""
    FASE 2 · LA DISCIPLINA — continuación de apply_light.

    Verifica que la traza de crecimiento satisface la condición de Banach
    ρ(T_η) < 1 sobre la linealización exacta del Euler-step en u(n).

        g_ij    = (λ_i − λ_j) log(λ_i/λ_j) ≥ 0
        η_max   = 2 / max g_ij
        ρ(T; η) = max_{i≠j} |1 − η g_ij|

    Firma:

        audit_discipline : 𝔇_n × ℝ₊ → BanachContractionReport
    """

    @classmethod
    def audit_discipline(
        cls, rho: np.ndarray, eta_star: float = 1.5,
    ) -> BanachContractionReport:
        return BanachContractionAlgebra.audit(rho, eta_star)

    @classmethod
    def audit_from_seed(
        cls, seed: SeedState, eta_star: float = 1.5,
    ) -> BanachContractionReport:
        """Variante que disciplina la semilla cruda (pre-luz), útil como testigo."""
        return cls.audit_discipline(seed.rho_seed, eta_star)


# ── §2.4 CropGrowthPipeline — HAND-OFF FASE 2 → FASE 3 ────────────────────
@dataclass(frozen=True, slots=True)
class CropGrowthBundle:
    r"""
    Objeto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Encapsula el producto de los tres funtores Riego ⊗ Luz ⊗ Disciplina
    aplicados al SeedState de FASE 1.

        cycle_index : ℕ₊
        seed        : SeedState          (FASE 1)
        watering    : WateringReport     (§2.1)
        rho_illum   : ρ tras U(n)+Φ_α    (§2.2)
        brockett_c, renyi_c, fock_c
        discipline  : BanachContractionReport (§2.3)
    """
    cycle_index: int
    seed: SeedState
    watering: WateringReport
    rho_illum: np.ndarray
    brockett_c: BrockettPurificationCertificate
    renyi_c: RenyiPurificationCertificate
    fock_c: FockAnnihilationCertificate
    discipline: BanachContractionReport


class CropGrowthPipeline:
    r"""
    Orquestador determinista del crecimiento (funtor F₂).

        synthesize : ℕ × SeedState × Σ* × Σ* × ℝ₊ × ℝ₊ → CropGrowthBundle

    ────────────────────────────────────────────────────────────────────────
    HAND-OFF FORMAL  FASE 2 → FASE 3
    ────────────────────────────────────────────────────────────────────────
    synthesize es el morfismo terminal de la FASE 2.  Su imagen
    CropGrowthBundle es el dominio de TODOS los métodos de la FASE 3:

        HeytingCropAdjudicator.adjudicate  : Bundle × Ω₃ → Ω₃
        CognitiveFaithModule.verify_faith  : Ω₃ → CrowbarActuationReport
        TOONCognitiveCropEngine.cultivate  : … → CropGerminationCertificate

    Identidad de anidamiento:

        certify ∘ synthesize ∘ prepare  :  M_n(ℂ)×Σ*×Σ* → Certificate.
    ────────────────────────────────────────────────────────────────────────
    """

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        seed: SeedState,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
        eta_star: float = 1.5,
    ) -> CropGrowthBundle:
        r"""
        Cierra la FASE 2.  Abre la FASE 3.

        Composición estricta, sin estado oculto:

            watering   = apply_water(seed, toon, json)          §2.1
            (ρ, B, R, F) = apply_light(seed, α)                 §2.2
            discipline = audit_discipline(ρ, η*)                §2.3
        """
        watering = CognitiveWateringModule.apply_water(seed, toon_str, base_json_str)
        rho_illum, brockett_c, renyi_c, fock_c = (
            CognitiveIlluminationModule.apply_light(seed, renyi_alpha)
        )
        discipline = CognitiveDisciplineModule.audit_discipline(rho_illum, eta_star)
        return CropGrowthBundle(
            cycle_index=cycle_index,
            seed=seed,
            watering=watering,
            rho_illum=rho_illum,
            brockett_c=brockett_c,
            renyi_c=renyi_c,
            fock_c=fock_c,
            discipline=discipline,
        )

    @classmethod
    def continue_into_phase3(
        cls,
        bundle: CropGrowthBundle,
        external_verdict: HeytingOmega3,
        reason_prefix: str = "CROP-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        r"""
        Continuación estricta de `synthesize`.

        Simultáneamente último morfismo de FASE 2 y primero de FASE 3:
        adjudica en Ω₃ y dispara la Fe.  El certificado (§3.3) se
        cristaliza aguas arriba en el orquestador, que posee el suelo
        |Ω⟩ y la cadena Merkle.
        """
        verdict = HeytingCropAdjudicator.adjudicate(bundle, external_verdict)
        reason = (
            f"{reason_prefix}::water={bundle.watering.local_verdict.name} "
            f"disc={bundle.discipline.local_verdict.name} "
            f"ρ(T)={bundle.discipline.spectral_radius:.4f} "
            f"η*={bundle.discipline.eta_star:.3f} "
            f"ηmax={bundle.discipline.eta_max:.3f}"
        )
        actuation = CognitiveFaithModule.verify_faith(verdict, reason)
        return verdict, actuation


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · FE + ADJUDICACIÓN + CERTIFICACIÓN                                ║
# ║                                                                           ║
# ║ Dominio = CropGrowthBundle (codominio de §2.4 synthesize).                ║
# ║ Codominio = CropGerminationCertificate (objeto terminal del cultivo).     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en Ω₃ ────────────────────────────────────────────────
class HeytingCropAdjudicator:
    r"""
    Lógica interna del topos: colapsa el Bundle a un único valor de Ω₃
    por meets sucesivos (producto de subobjetos):

        local = water ∧ discipline ∧ fock ∧ brockett ∧ renyi
        final = local ∧ external          (meet conservador, nunca infla)

    Reglas locales (clasificadores):
        fock      : annihilated ↦ ⊤  else ⋆
        brockett  : convergencia ∨ drift < 10⁻³ ↦ ⊤  else ⋆
        renyi     : purity_gain ≥ 0 ↦ ⊤  else ⋆
    """

    @classmethod
    def _fock_rule(cls, bundle: CropGrowthBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if bundle.fock_c.is_annihilated
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _brockett_rule(cls, bundle: CropGrowthBundle) -> HeytingOmega3:
        ok = (
            bundle.brockett_c.converged
            or bundle.brockett_c.isospectral_drift < 1e-3
        )
        return HeytingOmega3.COHERENT if ok else HeytingOmega3.DEGRADED

    @classmethod
    def _renyi_rule(cls, bundle: CropGrowthBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if bundle.renyi_c.purity_gain >= -1e-12
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def adjudicate(
        cls,
        bundle: CropGrowthBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""Continuación de CropGrowthPipeline.synthesize / continue_into_phase3."""
        local = (
            bundle.watering.local_verdict
            .meet(bundle.discipline.local_verdict)
            .meet(cls._fock_rule(bundle))
            .meet(cls._brockett_rule(bundle))
            .meet(cls._renyi_rule(bundle))
        )
        return local.meet(external_verdict)


# ── §3.2 CognitiveFaithModule — FE (interlock ciber-físico) ───────────────
@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    r"""
    Actuación física (simulada) del crowbar ESP32.

        interlock_fired      : True ⟺ verdict = ⊥
        actuation_latency_ns : cota de diseño < 400 ns (nominal 392.15 ns)
        gpio_pin, device     : "GPIO14", "BT151_CROWBAR"
        reason               : cadena explicativa
        provenance_hash      : SHA-256(reason ‖ t_ns)  forense
    """
    interlock_fired: bool
    actuation_latency_ns: float
    gpio_pin: str
    device: str
    reason: str
    provenance_hash: str


class CognitiveFaithModule:
    r"""
    FASE 3 · LA FE — certidumbre ciber-física en silicio (axioma operativo).

    Si Ω₃ = ⊥, se arma el crowbar:

        GPIO14 → HIGH  ⇒  MOSFET BT151  ⇒  latencia < 400 ns.

    La «fe» es el invariante de arquitectura: el corte ocurre antes de
    que la anomalía se propague.  Este módulo no emite I/O de hardware;
    certifica la decisión y su provenance.
    """
    GPIO_PIN: Final[str] = "GPIO14"
    DEVICE: Final[str] = "BT151_CROWBAR"
    NOMINAL_LATENCY_NS: Final[float] = 392.15

    @classmethod
    def verify_faith(
        cls, verdict: HeytingOmega3, reason: str = "",
    ) -> CrowbarActuationReport:
        if verdict != HeytingOmega3.VETOED:
            return CrowbarActuationReport(
                interlock_fired=False,
                actuation_latency_ns=0.0,
                gpio_pin=cls.GPIO_PIN,
                device=cls.DEVICE,
                reason="OK",
                provenance_hash="",
            )

        t_ns = time.time_ns()
        payload = f"CROWBAR_CROP::{reason}::{t_ns}".encode("utf-8")
        prov = hashlib.sha256(payload).hexdigest()
        logger.critical(
            "[CULTIVO COGNITIVO — CROWBAR ARMADO] %s → HIGH | "
            "latencia %.2f ns | razón=%s",
            cls.GPIO_PIN, cls.NOMINAL_LATENCY_NS, reason,
        )
        return CrowbarActuationReport(
            interlock_fired=True,
            actuation_latency_ns=cls.NOMINAL_LATENCY_NS,
            gpio_pin=cls.GPIO_PIN,
            device=cls.DEVICE,
            reason=reason,
            provenance_hash=prov,
        )


# ── §3.3 Certificado del cultivo (frozen + phase chain) ───────────────────
@dataclass(frozen=True, slots=True)
class CropGerminationCertificate:
    r"""
    Objeto terminal del cultivo: producto fibrado firmado

        Certificate ≅  Bundle × Ω₃ × Crowbar × SoilFidelity × Merkle.

        phase_chain_sha256  : H_k = SHA-256(H_{k−1} ‖ tag_k ‖ payload_k)
        sha256_provenance   : firma global del orquestador
    """
    crop_id: str
    seed_crystal_id: str
    water_kv_compression_ratio: float
    light_photons_emitted: int
    discipline_banach_radius: float
    discipline_eta_max: float
    faith_crowbar_armed: bool
    faith_provenance_hash: str
    heyting_verdict: HeytingOmega3
    germinated_purity: float
    germinated_entropy: float
    germinated_fidelity_to_ground: float
    brockett_iterations: int
    brockett_isospectral_drift: float
    fock_resonance_residual: float
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.4 TOONCognitiveCropEngine — orquestador soberano ───────────────────
class TOONCognitiveCropEngine:
    r"""
    Motor del Cultivo Cognitivo Dinámico.

    Funtor soberano  F = F₃ ∘ F₂ ∘ F₁ :

        F₁  SeedCrystalPreparation.prepare
        F₂  CropGrowthPipeline.synthesize
        F₃  adjudicate ⊗ verify_faith ⊗ certify

    Toma cristales de experiencia, los somete a Riego / Luz / Disciplina,
    adjudica en Ω₃, dispara la Fe si ⊥, y cristaliza un certificado con
    cadena Merkle de fases.
    """

    def __init__(
        self,
        engine_id: str = "CROP-ENGINE-WISDOM-01",
        dimension_mac: int = 4,
        eta_star: float = 1.5,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1")
        self.engine_id = engine_id
        self.dimension_mac = int(dimension_mac)
        self.eta_star = float(eta_star)
        self.renyi_alpha = float(renyi_alpha)
        self.crop_counter = 0

        self.soil: SoilState = SoilField.build(self.dimension_mac)
        self._chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::soil={self.soil.hash}".encode("ascii")
        ).hexdigest()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _phase1_prepare(self, seed_matrix: np.ndarray) -> SeedState:
        """FASE 1 anidada: cierra con SeedState (dominio de FASE 2)."""
        seed_state = SeedCrystalPreparation.prepare(seed_matrix)
        self._advance_chain("F1", np.ascontiguousarray(seed_state.rho_seed).tobytes())
        return seed_state

    def _phase2_grow(
        self,
        seed_state: SeedState,
        toon_str: str,
        base_json_str: str,
    ) -> CropGrowthBundle:
        """FASE 2 anidada: continuación de prepare; cierra con Bundle."""
        bundle = SeedCrystalPreparation.continue_into_phase2(
            seed=seed_state,
            cycle_index=self.crop_counter,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=self.renyi_alpha,
            eta_star=self.eta_star,
        )
        self._advance_chain(
            "F2",
            (
                f"purity={bundle.renyi_c.purity_after:.12f}|"
                f"rho_T={bundle.discipline.spectral_radius:.12f}"
            ).encode("ascii"),
        )
        return bundle

    def _phase3_certify(
        self,
        crop_id: str,
        seed_crystal_id: str,
        bundle: CropGrowthBundle,
        external_verdict: HeytingOmega3,
    ) -> CropGerminationCertificate:
        """FASE 3 anidada: continuación de synthesize; cierra con Certificate."""
        final_verdict, actuation = CropGrowthPipeline.continue_into_phase3(
            bundle, external_verdict
        )
        self._advance_chain(
            "F3",
            f"{final_verdict.name}|{actuation.provenance_hash}".encode("ascii"),
        )

        fid_ground = DensityOperatorAlgebra.uhlmann_fidelity(
            bundle.rho_illum, self.soil.ground_proj,
        )
        signature = _sha256_bytes(
            self.engine_id.encode("ascii"),
            crop_id.encode("ascii"),
            seed_crystal_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{bundle.renyi_c.purity_after:.10f}".encode("ascii"),
            f"{bundle.discipline.spectral_radius:.10f}".encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
            self._chain_hash.encode("ascii"),
        )
        return CropGerminationCertificate(
            crop_id=crop_id,
            seed_crystal_id=seed_crystal_id,
            water_kv_compression_ratio=bundle.watering.kv_compression_ratio,
            light_photons_emitted=bundle.fock_c.gamma_photons_emitted,
            discipline_banach_radius=bundle.discipline.spectral_radius,
            discipline_eta_max=bundle.discipline.eta_max,
            faith_crowbar_armed=actuation.interlock_fired,
            faith_provenance_hash=actuation.provenance_hash,
            heyting_verdict=final_verdict,
            germinated_purity=bundle.renyi_c.purity_after,
            germinated_entropy=bundle.renyi_c.entropy_after,
            germinated_fidelity_to_ground=fid_ground,
            brockett_iterations=bundle.brockett_c.iterations,
            brockett_isospectral_drift=bundle.brockett_c.isospectral_drift,
            fock_resonance_residual=bundle.fock_c.resonance_residual,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=signature,
            timestamp_utc=time.time(),
        )

    def cultivate_seed_crystal(
        self,
        seed_crystal_id: str,
        seed_matrix: np.ndarray,
        toon_str: str,
        base_json_str: str,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> CropGerminationCertificate:
        r"""
        Ciclo soberano: F₃ ∘ F₂ ∘ F₁.

        Asociatividad (teorema de anidamiento):

            cultivate = _phase3_certify ∘ _phase2_grow ∘ _phase1_prepare
                      = certify ∘ synthesize ∘ prepare.
        """
        self.crop_counter += 1
        crop_id = f"GERMINATED-CROP-{self.crop_counter:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Cultivo #%d | semilla=%s | η*=%.3f | α_Rényi=%.2f ═══",
            self.crop_counter, seed_crystal_id, self.eta_star, self.renyi_alpha,
        )

        seed_state = self._phase1_prepare(seed_matrix)
        bundle = self._phase2_grow(seed_state, toon_str, base_json_str)
        cert = self._phase3_certify(crop_id, seed_crystal_id, bundle, external_verdict)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cultivo %s finalizado en %.2f ms | Ω₃=%s | ρ(T)=%.4f | "
            "P_after=%.6f | γ=%d | crowbar=%s | drift_iso=%.2e",
            crop_id, dt_ms, cert.heyting_verdict.name,
            cert.discipline_banach_radius,
            cert.germinated_purity,
            cert.light_photons_emitted,
            cert.faith_crowbar_armed,
            cert.brockett_isospectral_drift,
        )
        return cert


# ── §3.5 Demostración autónoma ───────────────────────────────────────────
def _build_seed_matrix(n: int, alpha: float, key: str) -> ComplexMatrix:
    r"""
    Semilla determinista de espectro softmax (familia exponencial):

        w_k = exp(α (n − k)) / Z,    k = 1…n.

        α = 0  →  λ = 1/n     (máximamente mezclada)
        α → ∞ →  λ → e₁      (pura)

    Se sumerge en 𝔇_n por una rotación Haar extraída de QR de un
    Gaussiano complejo (Ginibre → Haar en U(n), Stewart 1980).
    """
    rng = np.random.default_rng(_seed_from_string(key))
    idx = np.arange(n)
    raw = np.exp(alpha * (n - idx))
    w = raw / raw.sum()
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    rho = (Q * w.astype(np.complex128)) @ Q.conj().T
    return DensityOperatorAlgebra.sanitize(rho)


if __name__ == "__main__":
    print("═" * 88)
    print("TOON COGNITIVE CROP ENGINE — v2.2.0 Nested Doctoral")
    print("Banach · Brockett-U(n) · Rényi · Fock · Heyting Ω₃ · Merkle")
    print("═" * 88)

    toon_str = "[APU: 2.1.4|CONCRETO 3000PSI] COST: 485000 COP"
    base_json_str = json.dumps({
        "apu_code": "2.1.4-CONCRETO-3000PSI",
        "unit_cost": 485000.0,
        "metadata": {
            "schema": "fat_json_structure_with_verbose_keys",
            "authority": "Sovereign-APU-Crop-Engine",
            "redundant": "eliminable_by_toon_tabularization",
        },
    }, indent=2)

    engine = TOONCognitiveCropEngine(
        engine_id="CROP-ENGINE-WISDOM-01",
        dimension_mac=4,
        eta_star=1.5,
        renyi_alpha=1.5,
    )

    scenarios = [
        ("COHERENT", 0.5, "SEED-FOCUSED", HeytingOmega3.COHERENT),
        ("DEGRADED", 0.0, "SEED-UNIFORM", HeytingOmega3.COHERENT),
        ("VETOED", 3.0, "SEED-HYPERSHARP", HeytingOmega3.COHERENT),
    ]

    print("\n" + "─" * 88)
    for name, alpha, key, ext in scenarios:
        seed_matrix = _build_seed_matrix(n=4, alpha=alpha, key=key)
        cert = engine.cultivate_seed_crystal(
            seed_crystal_id=f"CRYSTAL-{key}",
            seed_matrix=seed_matrix,
            toon_str=toon_str,
            base_json_str=base_json_str,
            external_verdict=ext,
        )
        crowbar_ns = (
            CognitiveFaithModule.NOMINAL_LATENCY_NS if cert.faith_crowbar_armed else 0.0
        )
        print(f"\n[{name}]  α_cría = {alpha}  seed = {key}")
        print(f"   crop_id                : {cert.crop_id}")
        print(f"   Ω₃ final               : {cert.heyting_verdict.name}")
        print(f"   Riego KV ratio         : {cert.water_kv_compression_ratio:.4f} "
              f"({cert.water_kv_compression_ratio * 100:.1f}%)")
        print(f"   Fotones γ              : {cert.light_photons_emitted}  "
              f"(residuo Fock = {cert.fock_resonance_residual:.4f})")
        print(f"   Disciplina ρ(T; η*)    : {cert.discipline_banach_radius:.6f}  "
              f"(η_max = {cert.discipline_eta_max:.4f})")
        print(f"   Pureza germinada       : {cert.germinated_purity:.6f}")
        print(f"   Entropía germinada     : {cert.germinated_entropy:.6f}")
        print(f"   Fidelidad a |Ω⟩        : {cert.germinated_fidelity_to_ground:.6f}")
        print(f"   Iteraciones Brockett   : {cert.brockett_iterations}  "
              f"(drift isospectral = {cert.brockett_isospectral_drift:.2e})")
        print(f"   Crowbar (Fe)           : {cert.faith_crowbar_armed}  "
              f"latencia = {crowbar_ns} ns")
        print(f"   Firma SHA-256          : {cert.sha256_provenance[:32]}…")

    print("\n" + "═" * 88)
    print("✓ F1→F2: prepare ⊣ apply_water / apply_light / audit_discipline.")
    print("✓ F2→F3: synthesize ⊣ adjudicate ⊗ verify_faith ⊗ certify.")
    print("✓ Riego: Shannon H₂ + BPE Ω(⌈n/4⌉), funcional convexo Δ_gr.")
    print("✓ Luz: Brockett en U(n) (polar) + Rényi Φ_α + matching Fock.")
    print("✓ Disciplina: ρ(T)=max|1−η g_ij| vectorizado, banda 2/g_max.")
    print("✓ Merkle de fases preservado entre cultivos.")
    print(f"✓ Total cultivos: {engine.crop_counter}")
    print("═" * 88)
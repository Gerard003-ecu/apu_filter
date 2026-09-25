# -*- coding: utf-8 -*-
r"""Soberano de Calibre del Cultivo Cognitivo Dinámico.

Ubicación: app/agents/wisdom/toon_cognitive_crop_agent.py
Versión  : 2.2.0-Doctoral-Nested-Banach-U(n)-Rényi-MAC-Merkle

Este módulo implementa el "Soberano del Cultivo Cognitivo", agente encargado de
cultivar, purificar y calibrar estados semilla (Seed Crystals) en la Memoria de
Alto Contenido (MAC) de la arquitectura COGNITIVE TOON / APU Filter.

================================================================================
I. FORMALIZACIÓN MATEMÁTICA Y ESPACIOS BANACH / HILBERT
================================================================================

1. Espacio de Estados Densidad y Álgebra de Banach:
   El estado semilla se representa sobre la C*-álgebra $M_n(\mathbb{C})$ mediante el compacto convexo
   $$\mathfrak{D}_n = \{ \rho \in M_n(\mathbb{C}) : \rho = \rho^\dagger, \, \rho \ge 0, \, \mathrm{Tr}(\rho) = 1 \}$$
   La pureza se define como $\mathcal{P}(\rho) = \mathrm{Tr}(\rho^2) = \|\rho\|_2^2$, la entropía de von Neumann
   como $S(\rho) = -\mathrm{Tr}(\rho \log \rho)$, y la entropía de Rényi como $S_\alpha(\rho) = \frac{1}{1-\alpha} \log \mathrm{Tr}(\rho^\alpha)$.

2. Módulo de Riego y Reducción de Grasa Sintáctica (Shannon + BPE):
   Dado un texto en formato TOON y su equivalente en JSON, la reducción de grasa sintáctica $\Delta_{\mathrm{gr}}$ es:
       $$\Delta_{\mathrm{gr}} = 100 \cdot \left[ w_T \left(1 - \frac{t_{\mathrm{toon}}}{t_{\mathrm{json}}}\right) + w_H \left(1 - \frac{H_{\mathrm{toon}}}{H_{\mathrm{json}}}\right) \right]$$
   donde $t_x = \lceil |x|/4 \rceil$ es la estimación de tokens BPE y $H_x$ es la entropía de Shannon en bits.

3. Módulo de Luz (Flujo Isospectral de Brockett y Sharpening de Rényi):
   La evolución isospectral en el grupo unitario $U(n)$ bajo la función de Lyapunov $L(\rho) = \mathrm{Tr}(\rho N)$
   ($N = \mathrm{diag}(1, 2, \dots, n)$) es:
       $$\frac{d\rho}{dt} = [\rho, [\rho, N]]$$
   Seguida por la transformación de Rényi $U(n)$-equivariante $\Phi_\alpha(\rho) = \frac{\rho^\alpha}{\mathrm{Tr}(\rho^\alpha)}$ ($\alpha \ge 1$),
   que incrementa la pureza de forma monótona.

4. Módulo de Disciplina (Contracción de Banach en $\mathfrak{u}(n)$ y Canal MAC):
   En la linealización $T_\eta(\rho) = \rho - \eta [\rho, [\rho, K_\rho]]$, el radio espectral en $\mathfrak{u}(n)$ es:
       $$\rho(T_\eta) = \max_{i \ne j} |1 - \eta \cdot g_{ij}|, \quad g_{ij} = (\lambda_i - \lambda_j) \log\left(\frac{\lambda_i}{\lambda_j}\right) \ge 0$$
   Para el canal convexo de la MAC $\Phi_\gamma(\rho) = (1-\gamma)\rho + \gamma \rho_{\mathrm{target}}$, la constante de Lipschitz es
   $\mathrm{Lip}_{\|\cdot\|_1}(\Phi_\gamma) = |1 - \gamma| < 1$.

5. Adjudicación de Heyting $\Omega_3$ y Pasaporte de Gobernanza Merkle:
   El veredicto final en $\Omega_3 = \{\bot (\mathrm{VETOED}) < \star (\mathrm{DEGRADED}) < \top (\mathrm{COHERENT})\}$
   aplica meet ($\land$) sobre los indicadores de Riego, Luz, Disciplina y la auditoría de vacío.
   Las cosechas se firman mediante árboles de Merkle SHA-256.

================================================================================
II. ESTRUCTURA FUNTORIAL Y ARQUITECTURA
================================================================================

El Soberano opera como la composición estricta del funtor $F$:
    $$F : \mathbf{SeedCrystal} \times \Sigma^* \times \Sigma^* \longrightarrow \mathbf{CropHarvestYield} \times \mathbf{Passport}$$
    $$F = F_3 \circ F_2 \circ F_1$$

  • $F_1$ (`SeedHandoff.build`): $\mathbf{SeedCrystal} \times \mathbf{MAC} \to \mathrm{SeedHandoff}$.
    Sanitización $C^*$, proyección sobre $\mathfrak{D}_n$, espectro modular $K_\rho$ y verificación del estado fundamental.
  • $F_2$ (`CropGrowthPipeline.synthesize`): $\mathrm{SeedHandoff} \to \mathrm{CropGrowthBundle}$.
    Riego sintáctico $\Delta_{\mathrm{gr}}$, flujo de Brockett en $U(n)$, sharpening $\Phi_\alpha$ y radio de Banach $\rho(T_\eta)$.
  • $F_3$ (`TOONCognitiveCropAgent._phase3_harvest`): $\mathrm{CropGrowthBundle} \to \mathrm{CropHarvestYield}$.
    Adjudicación en $\Omega_3$, inoculación afín $\Phi_\gamma$, crowbar ESP32 si $\bot$ y pasaporte Merkle.

================================================================================
III. INVARIANTES FORMALES Y AXIOMAS DEL SISTEMA
================================================================================

- Axioma 1 (Isotonicidad de Brockett): La purificación preserva el espectro $\sigma(\rho_t) = \sigma(\rho_0)$ con $\dot{L} = \|[\rho, N]\|_F^2 \ge 0$.
- Axioma 2 (Contracción de Banach): $\rho(T_\eta) < 1$ para todo $\eta < 2 / g_{\max}$.
- Axioma 3 (Coherencia Lógica del Vacío): Si se declara $\mathrm{is\_vacuum\_pure} = \mathrm{True}$ pero $F(\rho, |\Omega\rangle\langle\Omega|) \le 1 - \varepsilon$, se marca $\mathrm{false\_vacuum\_claim} = \mathrm{True}$ y el veredicto es $\bot (\mathrm{VETOED})$.
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


logger = logging.getLogger("APU.Wisdom.TOONCognitiveCropAgent")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS: Final[float] = 1.0e-14
_EPS_MOD: Final[float] = 1.0e-12
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
# ║ FASE 1 · SUSTRATO ALGEBRAICO + SANEAMIENTO DE LA SEMILLA                  ║
# ║                                                                           ║
# ║ Objetos: Ω₃, 𝔇_n, B(u(n)), Φ_γ, (H_mac, |Ω⟩).                             ║
# ║ Morfismo terminal: SeedHandoff.build / continue_into_phase2.              ║
# ║ Ese morfismo ES el dominio de todos los funtores de la FASE 2.            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §1.1 Retículo distributivo de Heyting Ω₃ ──────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa (álgebra de Gödel de 3 valores)

        Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0, 1, 2}

    Estructura:
        meet    (∧) : ínfimo = min
        join    (∨) : supremo = max
        implies (⇒) : residuo  (a ∧ b ≤ c ⇔ a ≤ (b ⇒ c))
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
        """Inmersión afín Ω₃ ↪ [0, 1] : ⊥↦0, ⋆↦½, ⊤↦1."""
        return float(int(self)) / 2.0


# ── §1.2 Álgebra de operadores densidad ───────────────────────────────────
class DensityOperatorAlgebra:
    r"""
    Operaciones canónicas sobre el compacto convexo de estados

        𝔇_n = { ρ ∈ M_n(ℂ) : ρ = ρ†,  ρ ≥ 0,  Tr ρ = 1 }.

    Funcionales (unitariamente invariantes, funciones espectrales):

        S(ρ)     = −Tr(ρ log ρ)                 von Neumann (nats)
        S_α(ρ)   = (1−α)⁻¹ log Tr(ρ^α)          Rényi, α ≠ 1
        P(ρ)     = Tr(ρ²) = ‖ρ‖₂²               pureza ∈ [1/n, 1]
        F(ρ,σ)   = ‖√ρ √σ‖₁                     Uhlmann–Jozsa
        T(ρ,σ)   = (1/2)‖ρ−σ‖₁                  distancia de traza
        K_ρ      = −log ρ                       Hamiltoniano modular
        ρ^z      = exp(z log ρ)                 cálculo funcional
        Φ_α(ρ)   = ρ^α / Tr(ρ^α)                reweighting de Rényi

    sanitize proyecta afínmente sobre 𝔇_n: hermitización + PSD-clip
    (autovalores negativos → ε) + renormalización de traza.  El clip
    es la proyección espectral de Dykstra sobre el cono PSD.
    """
    EPS: Final[float] = _EPS
    EPS_MODULAR: Final[float] = _EPS_MOD

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""Proyección afín sobre 𝔇_n: Hermitiza, PSD-clip, Tr = 1."""
        if not cls.is_square(rho):
            raise ValueError(
                f"DensityOperatorAlgebra.sanitize: no cuadrada {np.shape(rho)}"
            )
        rho_h = np.asarray(rho, dtype=np.complex128)
        rho_h = 0.5 * (rho_h + rho_h.conj().T)
        w, V = la.eigh(rho_h)
        w = np.maximum(np.real(w), cls.EPS)
        rho_h = (V * w) @ V.conj().T
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
        r"""‖ρ‖_p = (Σ σ_i^p)^{1/p}.  p=∞ → σ_max."""
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
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sr = cls.matrix_power(rho, 0.5)
        inner = sr @ sigma @ sr
        val = float(np.real(np.trace(cls.matrix_power(inner, 0.5))))
        return float(np.clip(val, 0.0, 1.0))

    @classmethod
    def matrix_power(
        cls, rho: np.ndarray, z: complex, floor: float = _EPS_MOD
    ) -> ComplexMatrix:
        r"""ρ^z = V diag(λ_i^z) V†  (Holstein–Rellich, corte principal)."""
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

    @classmethod
    def modular_hamiltonian(cls, rho: np.ndarray) -> ComplexMatrix:
        """K_ρ = −log ρ  (Tomita–Takesaki: Δ_ρ = exp(−K_ρ))."""
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
        H = np.asarray(H, dtype=np.complex128)
        H = 0.5 * (H + H.conj().T)
        w, V = la.eigh(H)
        i0 = int(np.argmin(np.real(w)))
        psi = V[:, i0].reshape(-1, 1)
        return cls.sanitize(psi @ psi.conj().T)

    @classmethod
    def renyi_sharpen(cls, rho: np.ndarray, alpha: float) -> ComplexMatrix:
        r"""
        Reweighting de Rényi (función espectral, U(n)-equivariante):

            Φ_α(ρ) = ρ^α / Tr(ρ^α),    α ≥ 1.

        α = 1 → id.  α > 1 ⇒ P↑, S↓.  Conmuta con el flujo de Brockett.
        """
        if alpha <= 1.0 + 1e-12:
            return cls.sanitize(rho)
        rho = cls.sanitize(rho)
        rho_a = cls.matrix_power(rho, complex(alpha, 0.0))
        tr = float(np.trace(rho_a).real)
        if tr < 1e-30:
            return rho
        return cls.sanitize(rho_a / tr)


# ── §1.3 Álgebra de Banach: radio espectral real ⊕ Lip(Φ_γ) ───────────────
@dataclass(frozen=True, slots=True)
class BanachContractionReport:
    r"""
    Auditoría conjunta semilla ⊕ MAC en B(u(n)) × CPTP(𝔇_n).

    (A) Euler-step de la mutación doble-corchete sobre estados:

            T_η(ρ) = ρ − η [ρ, [ρ, N]],     N = K_ρ.

        En un equilibrio diagonal ρ = diag(λ), la linealización DT_η
        actúa sobre los modos E_{ij} (i ≠ j) con autovalores exactos
        (Brockett 1991):

            τ_ij(η) = 1 − η · g_ij,
            g_ij    = (λ_i − λ_j) log(λ_i/λ_j) ≥ 0.

        ρ(T_η) = max_{i≠j} |τ_ij(η)|.
        Banda de Banach: 0 < η < η_max := 2/g_max  ⇒  ρ(T_η) < 1.

    (B) Canal convexo de la MAC (Birkhoff 1957, coeficiente de Hilbert):

            Φ_γ(ρ) = (1−γ) ρ + γ ρ_target,     γ ∈ [0, 1].

        Lip_{‖·‖₁}(Φ_γ) = |1−γ|  exactamente (afín, no heurístico).
        Contracción estricta ⟺ γ ∈ (0, 1].
    """
    seed_spectral_radius: float
    seed_eta_max: float
    seed_g_max: float
    seed_pair_index: Tuple[int, int]
    mac_contraction_coef: float
    mac_gamma: float
    is_seed_contraction: bool
    is_mac_contraction: bool
    lipschitz_bound: float
    local_verdict: HeytingOmega3


class BanachContractionAlgebra:
    r"""
    Cálculo vectorizado de ρ(T_η) sobre u(n) y de Lip(Φ_γ) sobre 𝔇_n.

    G = (g_ij) es un kernel Hilbert–Schmidt simétrico, nulo en la
    diagonal, G ≥ 0 entrada a entrada.  g_max = ‖G‖_∞.
    """
    EPS: Final[float] = _EPS_MOD

    @classmethod
    def coupling_matrix(cls, rho: np.ndarray) -> RealVector:
        r"""
        G_ij = (λ_i − λ_j) log(λ_i/λ_j),  G_ii = 0.

        Identidad: g_ij = 0 ⇔ λ_i = λ_j;  g_ij ≥ 0 por convexidad de x log x.
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
        """(g_max, i*, j*) del par más inestable."""
        G = cls.coupling_matrix(rho)
        if G.size == 0:
            return 0.0, 0, 0
        idx = int(np.argmax(G))
        n = G.shape[0]
        i_max, j_max = divmod(idx, n)
        return float(G[i_max, j_max]), int(i_max), int(j_max)

    @classmethod
    def seed_spectral_radius(
        cls, rho: np.ndarray, eta_star: float,
    ) -> Tuple[float, float, float]:
        r"""
        (ρ(T; η*), η_max, g_max).

        n = 1            → ρ(T) = 0  (sin modos de coherencia: vacuamente contractivo)
        n > 1, g_max = 0 → ρ(T) = 1  (DT = Id, marginal)
        """
        G = cls.coupling_matrix(rho)
        n = int(G.shape[0]) if G.size else 0
        if n <= 1:
            return 0.0, float("inf"), 0.0
        g_max = float(G.max())
        if g_max < cls.EPS:
            return 1.0, float("inf"), 0.0
        tau = 1.0 - float(eta_star) * G
        np.fill_diagonal(tau, 0.0)
        rho_T = float(np.max(np.abs(tau)))
        eta_max = 2.0 / g_max
        return rho_T, float(eta_max), g_max

    @classmethod
    def mac_lipschitz(cls, gamma: float) -> float:
        r"""Lip_{‖·‖₁}(Φ_γ) = |1 − clip(γ,[0,1])|."""
        g = float(np.clip(gamma, 0.0, 1.0))
        return abs(1.0 - g)

    @classmethod
    def audit(
        cls,
        rho_seed: np.ndarray,
        eta_star: float,
        mac_gamma: float,
    ) -> BanachContractionReport:
        rho_T, eta_max, g_max = cls.seed_spectral_radius(rho_seed, eta_star)
        _, i_star, j_star = cls.pair_couplings(rho_seed)
        mac_coef = cls.mac_lipschitz(mac_gamma)
        lip = float(max(rho_T, mac_coef))

        p_seed = rho_T < 1.0
        p_mac = mac_coef < 1.0

        if p_seed and p_mac:
            if rho_T < 1.0 - _EPS_CONTRACT:
                local = HeytingOmega3.COHERENT
            else:
                local = HeytingOmega3.DEGRADED
        elif p_seed or p_mac:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return BanachContractionReport(
            seed_spectral_radius=rho_T,
            seed_eta_max=eta_max,
            seed_g_max=g_max,
            seed_pair_index=(i_star, j_star),
            mac_contraction_coef=mac_coef,
            mac_gamma=float(np.clip(mac_gamma, 0.0, 1.0)),
            is_seed_contraction=p_seed,
            is_mac_contraction=p_mac,
            lipschitz_bound=lip,
            local_verdict=local,
        )


# ── §1.4 SeedCrystal + sanitizador dimensional ────────────────────────────
@dataclass(frozen=True, slots=True)
class SeedCrystal:
    r"""
    Cristal de experiencia emitido por el Testigo Silencioso.

        density_matrix    : ρ_cruda ∈ M_n(ℂ)  (se sanea en F₁)
        experience_vector : invariantes declarados por el Testigo
            convención: [verdict_norm, dirichlet_energy, purity, crystal_count]
        is_vacuum_pure    : flag DECLARADO — el motor lo confronta con
            P(ρ) y F(ρ, |Ω⟩⟨Ω|).  «Vacuum» = ground de H_mac, no un
            puro Haar-aleatorio.
    """
    crystal_id: str
    origin_agent: str
    density_matrix: np.ndarray
    experience_vector: np.ndarray
    is_vacuum_pure: bool


@dataclass(frozen=True, slots=True)
class SeedAuditReport:
    r"""
    Auditoría dimensional de la semilla.

        verified_vacuum_pure := P(ρ) > 1−ε_P  ∧  F(ρ,|Ω⟩⟨Ω|) > 1−ε_F
        consistent           := (verified == declared)
        false_vacuum_claim   := declared ∧ ¬verified   (testimonio falso)
        consistency_residual := ‖êv − ŝig‖₂ / √2       (chordal en S^{k−1})
    """
    purity: float
    entropy: float
    fidelity_to_ground: float
    lambda_min: float
    lambda_max: float
    spectral_gap: float
    declared_vacuum_pure: bool
    verified_vacuum_pure: bool
    consistent: bool
    false_vacuum_claim: bool
    consistency_residual: float
    spectral_hash: str


class SeedCrystalSanitizer:
    r"""
    Proyecta ρ_cruda → 𝔇_n, calcula la signatura dimensional y verifica
    la coherencia lógica entre el flag declarado y la física del estado.

    Signatura (coordenadas adimensionales en [0,1]⁴):

        sig(ρ) = [ P(ρ),  S(ρ)/log n,  F(ρ,|Ω⟩⟨Ω|),  λ_min/(λ_min+λ_max) ]

    El residual es la distancia euclídea entre versores (invariante de
    escala del experience_vector).  Un residual 0 es alineación perfecta;
    1 es antipodal en el hemisferio positivo.
    """
    EPS_PURITY: Final[float] = 1.0e-6
    EPS_FID: Final[float] = 1.0e-6

    @classmethod
    def signature(cls, rho: np.ndarray, rho_ground: np.ndarray) -> RealVector:
        P = DensityOperatorAlgebra.purity(rho)
        S = DensityOperatorAlgebra.von_neumann_entropy(rho)
        n = int(rho.shape[0])
        S_max = math.log(n) if n > 1 else 1.0
        F = DensityOperatorAlgebra.uhlmann_fidelity(rho, rho_ground)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        lam_min, lam_max = float(w[-1]), float(w[0])
        lam_ratio = lam_min / (lam_min + lam_max + 1e-30)
        return np.array([P, S / max(S_max, 1e-30), F, lam_ratio], dtype=np.float64)

    @classmethod
    def sanitize(
        cls,
        seed: SeedCrystal,
        rho_ground: np.ndarray,
    ) -> Tuple[ComplexMatrix, SeedAuditReport]:
        rho = DensityOperatorAlgebra.sanitize(seed.density_matrix)
        w = DensityOperatorAlgebra.spectrum_descending(rho)

        P = float(np.sum(w ** 2))
        S = float(-np.sum(w * np.log(np.maximum(w, _EPS))))
        F = DensityOperatorAlgebra.uhlmann_fidelity(rho, rho_ground)
        lam_min, lam_max = float(w[-1]), float(w[0])
        gap = float(w[0] - w[1]) if w.size > 1 else 0.0

        verified = (P > 1.0 - cls.EPS_PURITY) and (F > 1.0 - cls.EPS_FID)
        declared = bool(seed.is_vacuum_pure)
        consistent = bool(verified == declared)
        false_claim = bool(declared and (not verified))

        sig = cls.signature(rho, rho_ground)
        ev = np.asarray(seed.experience_vector, dtype=np.float64).reshape(-1)
        k = min(int(ev.size), int(sig.size))
        if k == 0:
            residual = 1.0
        else:
            ev_k = ev[:k]
            sig_k = sig[:k]
            n_ev = float(np.linalg.norm(ev_k))
            n_sg = float(np.linalg.norm(sig_k))
            ev_u = ev_k / (n_ev + 1e-30)
            sg_u = sig_k / (n_sg + 1e-30)
            residual = float(np.linalg.norm(ev_u - sg_u) / math.sqrt(2.0))

        spec_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(w).tobytes(),
        )
        audit = SeedAuditReport(
            purity=P,
            entropy=S,
            fidelity_to_ground=F,
            lambda_min=lam_min,
            lambda_max=lam_max,
            spectral_gap=gap,
            declared_vacuum_pure=declared,
            verified_vacuum_pure=bool(verified),
            consistent=consistent,
            false_vacuum_claim=false_claim,
            consistency_residual=residual,
            spectral_hash=spec_hash,
        )
        return rho, audit


# ── §1.5 Estado MAC + operador de actualización auditado ──────────────────
@dataclass(frozen=True, slots=True)
class MacUpdateCertificate:
    r"""
    Certificado del canal convexo de la MAC:

        Φ_γ(ρ) = (1−γ)·ρ + γ·ρ_target,     γ ∈ [0, 1].

    Propiedades (todas teoremas, no heurísticas):
        · Traza:   Tr Φ_γ(ρ) = 1  ∀ ρ ∈ 𝔇_n
        · PSD:     Φ_γ(ρ) ⪰ 0     ∀ ρ ⪰ 0
        · CPTP:    Kraus {√(1−γ) I, √γ I}  (canal de mezcla)
        · Lip₁:    ‖Φ_γ(ρ₁) − Φ_γ(ρ₂)‖₁ = |1−γ| · ‖ρ₁−ρ₂‖₁
        · Punto fijo: Φ_γ(ρ_target) = ρ_target  (si γ = 1, constante)
        · Unital:  Φ_γ(I/n) = (1−γ) I/n + γ ρ_target
    """
    gamma: float
    contraction_coef: float
    fixed_point_fidelity: float
    trace_residual: float
    min_eigenvalue: float
    is_contractive: bool
    is_trace_preserving: bool
    is_positive_preserving: bool
    mutated: bool
    local_verdict: HeytingOmega3


class MacStateField:
    r"""
    Estado continuo de la Matriz Atómica de Conocimiento (MAC).

    Cada `update` aplica Φ_γ, verifica CPTP numéricamente y emite un
    certificado.  `peek` audita sin mutar (útil cuando el veredicto
    no autoriza inoculación).
    """

    def __init__(self, rho_init: np.ndarray, gamma: float = 0.2) -> None:
        self.rho: ComplexMatrix = DensityOperatorAlgebra.sanitize(rho_init)
        self.gamma: float = float(np.clip(gamma, 0.0, 1.0))

    def _apply(self, rho_target: np.ndarray) -> ComplexMatrix:
        rho_t = DensityOperatorAlgebra.sanitize(rho_target)
        return (1.0 - self.gamma) * self.rho + self.gamma * rho_t

    def _certify(self, rho_new: np.ndarray, rho_target: np.ndarray,
                 mutated: bool) -> MacUpdateCertificate:
        tr_res = abs(float(np.trace(rho_new).real) - 1.0)
        tr_ok = tr_res < 1e-9
        w_new = np.real(la.eigvalsh(0.5 * (rho_new + rho_new.conj().T)))
        min_ev = float(w_new.min()) if w_new.size else 0.0
        psd_ok = bool(min_ev > -1e-8)
        coef = abs(1.0 - self.gamma)
        fid = DensityOperatorAlgebra.uhlmann_fidelity(rho_new, rho_target)

        if coef < 1.0 and tr_ok and psd_ok:
            verdict = HeytingOmega3.COHERENT
        elif abs(coef - 1.0) <= 1e-15 and tr_ok and psd_ok:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.VETOED

        return MacUpdateCertificate(
            gamma=self.gamma,
            contraction_coef=coef,
            fixed_point_fidelity=fid,
            trace_residual=tr_res,
            min_eigenvalue=min_ev,
            is_contractive=coef < 1.0,
            is_trace_preserving=tr_ok,
            is_positive_preserving=psd_ok,
            mutated=mutated,
            local_verdict=verdict,
        )

    def peek(self, rho_target: np.ndarray) -> MacUpdateCertificate:
        """Audita Φ_γ(ρ, ρ_target) sin mutar el estado MAC."""
        rho_new = self._apply(rho_target)
        return self._certify(rho_new, rho_target, mutated=False)

    def update(self, rho_target: np.ndarray) -> MacUpdateCertificate:
        """Aplica Φ_γ y sanea el resultado sobre 𝔇_n."""
        rho_new = self._apply(rho_target)
        cert = self._certify(rho_new, rho_target, mutated=True)
        self.rho = DensityOperatorAlgebra.sanitize(rho_new)
        return cert

    def idle_certificate(self) -> MacUpdateCertificate:
        """Certificado de punto fijo: Φ_γ(ρ, ρ) = ρ (no-op auditado)."""
        return self._certify(self.rho, self.rho, mutated=False)


# ── §1.6 SeedHandoff — HAND-OFF FASE 1 → FASE 2 ──────────────────────────
@dataclass(frozen=True, slots=True)
class SeedHandoff:
    r"""
    Objeto terminal de la FASE 1 y objeto inicial de la FASE 2.

    Tipo:  SeedHandoff ≅ 𝔇_n × Audit × spec(K_ρ) × |Ω⟩⟨Ω| × MAC

        seed_id, origin_agent : procedencia
        rho_seed              : semilla saneada (Hermítica, PSD, Tr=1)
        seed_audit            : P, S, F, flag, residual
        K_spec_seed           : spec↑(−log ρ)
        ground_projector      : |Ω⟩⟨Ω|
        mac_snapshot, mac_gamma
        spectral_hash         : SHA-256(ρ ‖ λ)  custodia forense
    """
    seed_id: str
    origin_agent: str
    rho_seed: np.ndarray
    seed_audit: SeedAuditReport
    K_spec_seed: Tuple[float, ...]
    ground_projector: np.ndarray
    mac_snapshot: np.ndarray
    mac_gamma: float
    spectral_hash: str

    @classmethod
    def build(
        cls,
        seed: SeedCrystal,
        mac_field: MacStateField,
        ground_projector: np.ndarray,
    ) -> "SeedHandoff":
        r"""
        Cierra la FASE 1 como objeto.  El morfismo de continuación
        hacia FASE 2 es `continue_into_phase2`.
        """
        rho_s, audit = SeedCrystalSanitizer.sanitize(seed, ground_projector)
        K_spec = DensityOperatorAlgebra.modular_spectrum(rho_s)
        return cls(
            seed_id=seed.crystal_id,
            origin_agent=seed.origin_agent,
            rho_seed=rho_s,
            seed_audit=audit,
            K_spec_seed=K_spec,
            ground_projector=DensityOperatorAlgebra.sanitize(ground_projector),
            mac_snapshot=mac_field.rho.copy(),
            mac_gamma=mac_field.gamma,
            spectral_hash=audit.spectral_hash,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "seed_id": self.seed_id,
            "origin_agent": self.origin_agent,
            "purity": self.seed_audit.purity,
            "entropy": self.seed_audit.entropy,
            "fid_ground": self.seed_audit.fidelity_to_ground,
            "verified_pure": self.seed_audit.verified_vacuum_pure,
            "consistent": self.seed_audit.consistent,
            "false_vacuum_claim": self.seed_audit.false_vacuum_claim,
            "mac_gamma": self.mac_gamma,
            "spectral_hash": self.spectral_hash,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    # ══════════════════════════════════════════════════════════════════════
    def continue_into_phase2(
        self,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float,
        eta_star: float,
    ) -> "CropGrowthBundle":
        r"""
        Último morfismo de la FASE 1  ∧  primer morfismo de la FASE 2.

        Identidad de composición:

            continue_into_phase2 ∘ build
                = CropGrowthPipeline.synthesize ∘ build
                : Crystal × MAC × |Ω⟩ × Σ* × Σ* → CropGrowthBundle.

        En el sentido de categorías, la FASE 2 es el comma-category
        (SeedHandoff ↓ Crop₂).  Invocar Riego/Luz/Disciplina sin un
        SeedHandoff es un error de tipo.
        """
        return CropGrowthPipeline.synthesize(
            handoff=self,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=renyi_alpha,
            eta_star=eta_star,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · RIEGO + LUZ + DISCIPLINA                                         ║
# ║                                                                           ║
# ║ Dominio = SeedHandoff (codominio de §1.6).                                ║
# ║ Codominio = CropGrowthBundle, dominio de toda la FASE 3.                  ║
# ║                                                                           ║
# ║ §2.1 se lee como la continuación literal de continue_into_phase2.         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 RIEGO — Shannon + BPE ────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class WateringReport:
    r"""
    Compresión de grasa sintáctica (teorema de Shannon + estimador BPE).

        tokens_*              : Ω(⌈|s|/4⌉)
        h_*                   : H₂ bits/carácter
        fat_reduction_pct     : Δ_gr
        kv_compression_ratio  : min(0.95, Δ_gr/100)
        seed_entropy_nats     : S(ρ_seed)  (custodia F1→F2)
        local_verdict         : ⊤ si Δ_gr≥60, ⋆ si ≥30, ⊥ si no
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
    FASE 2 · EL RIEGO — continuación de SeedHandoff.continue_into_phase2.

    Estimador BPE ≃ ⌈n/4⌉ (chars/token GPT-like).  Entropía empírica:

        H₂(s) = − Σ_c (n_c/|s|) log₂(n_c/|s|)

    Funcional de grasa (pesos de Dirichlet W_T + W_H = 1):

        Δ_gr = 100 · [ W_T (1 − t_t/t_j) + W_H (1 − H_t/H_j) ].

    KV-ratio satura en KV_MAX = 0.95 (leftover de Kraft: nunca 100%).

        apply_water : SeedHandoff × Σ* × Σ* → WateringReport
    """
    W_TOKEN: Final[float] = 0.7
    W_ENTROPY: Final[float] = 0.3
    KV_MAX: Final[float] = 0.95
    CHARS_PER_TOKEN: Final[float] = 4.0

    @classmethod
    def bpe_tokens(cls, text: str) -> int:
        return max(1, int(math.ceil(len(text) / cls.CHARS_PER_TOKEN)))

    @classmethod
    def shannon_bits(cls, text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)

    @classmethod
    def fat_functional(
        cls, t_toon: int, t_json: int, h_toon: float, h_json: float
    ) -> float:
        red_t = 1.0 - t_toon / max(1, t_json)
        red_h = 1.0 - h_toon / max(1e-9, h_json)
        return 100.0 * (cls.W_TOKEN * red_t + cls.W_ENTROPY * red_h)

    @classmethod
    def apply_water(
        cls,
        handoff: SeedHandoff,
        toon_str: str,
        base_json_str: str,
    ) -> WateringReport:
        r"""
        Continuación de `SeedHandoff.continue_into_phase2`.

        Δ_gr es un funcional del lenguaje; la entropía de la semilla
        viaja en el reporte para la cadena de custodia F1→F2.
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
            seed_entropy_nats=float(handoff.seed_audit.entropy),
            local_verdict=local,
        )


# ── §2.2 LUZ — Brockett U(n) + Rényi + Fock ───────────────────────────────
@dataclass(frozen=True, slots=True)
class BrockettPurificationCertificate:
    r"""
    Flujo isospectral de Brockett en la órbita coadjunta:

        dρ/dt = [A(ρ), ρ],   A(ρ) = −[ρ, N],   N = diag(1,…,n)

    Equivale a ρ(t) = U(t) ρ(0) U(t)†,  U ∈ U(n).
    ℒ(ρ) = Tr(ρ N) es Lyapunov estrictamente decreciente fuera de
    los puntos alineados con N (Brockett 1991, Thm. 1).
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
    r"""Φ_α(ρ) = ρ^α/Tr(ρ^α).  purity_gain ≥ 0 y entropy_drop ≥ 0 si α ≥ 1."""
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

        E₋ = n · (1 − λ_max(ρ_seed))     fuga de la cúpula pura
        E₊ = S(ρ_seed)                   fricción térmica (nats)
        residual = |E₋ − E₊| / max(1, |E₋|, |E₊|)
        annihilated ⟺ residual < θ      ⇒  2 cuantos γ
    """
    electron_anomaly_energy: float
    positron_constraint_energy: float
    gamma_photons_emitted: int
    resonance_residual: float
    is_annihilated: bool


class CognitiveIlluminationModule:
    r"""
    FASE 2 · LA LUZ — tres operaciones encadenadas sobre SeedHandoff.rho_seed.

        (1) Brockett en U(n): RK4 del generador anti-Hermítico A = −[ρ, N],
            polar-proyectado a U(n) ⇒ isospectralidad exacta hasta redondeo.
        (2) Φ_α de Rényi: ganancia monótona de pureza.
        (3) Matching Fock de dos modos sobre la semilla (invariante de F1).

        apply_light : SeedHandoff → (𝔇_n × BrockettCert × RenyiCert × FockCert)
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
    def _anti_hermitian_generator(
        cls, rho: np.ndarray, N: np.ndarray
    ) -> ComplexMatrix:
        """A(ρ) = −[ρ, N] ∈ u(n)."""
        comm = rho @ N - N @ rho
        A = -comm
        return 0.5 * (A - A.conj().T)

    @classmethod
    def _project_unitary(cls, U: np.ndarray) -> ComplexMatrix:
        """Proyección polar U ↦ U (U†U)^{−1/2} ∈ U(n) vía SVD."""
        W, _, Vh = la.svd(U, full_matrices=False)
        return W @ Vh

    @classmethod
    def lyapunov_alignment(cls, rho: np.ndarray, N: np.ndarray) -> float:
        return float(np.trace(rho @ N).real)

    @classmethod
    def _brockett_unitary_rk4(
        cls, rho_init: np.ndarray,
    ) -> Tuple[ComplexMatrix, BrockettPurificationCertificate]:
        r"""
        Integra dU/dt = A(U ρ₀ U†) U en U(n) por RK4 + reproyección polar
        (Crouch–Grossman: orden 1 en la variedad, 4 en el álgebra).
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
    def _fock_resonance(cls, handoff: SeedHandoff) -> FockAnnihilationCertificate:
        w = DensityOperatorAlgebra.spectrum_descending(handoff.rho_seed)
        n = len(w)
        E_minus = float(n * (1.0 - w[0]))
        E_plus = float(handoff.seed_audit.entropy)
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
        handoff: SeedHandoff,
        renyi_alpha: float = RENYI_ALPHA,
    ) -> Tuple[
        ComplexMatrix,
        BrockettPurificationCertificate,
        RenyiPurificationCertificate,
        FockAnnihilationCertificate,
    ]:
        r"""Continuación de apply_water.  Producto tensorial Riego ⊗ Luz."""
        rho_flow, brockett_cert = cls._brockett_unitary_rk4(handoff.rho_seed)

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
        fock_cert = cls._fock_resonance(handoff)
        return rho_illum, brockett_cert, renyi_cert, fock_cert


# ── §2.3 DISCIPLINA — Banach + MAC contractive audit ─────────────────────
class CognitiveDisciplineModule:
    r"""
    FASE 2 · LA DISCIPLINA — continuación de apply_light.

        (1) Contracción espectral de la semilla: ρ(T_seed; η*) < 1.
        (2) Contracción del canal MAC: Lip(Φ_γ) = |1−γ| < 1.

    El veredicto local es el meet de ambos predicados (delegado a
    BanachContractionAlgebra.audit).

        audit : SeedHandoff × ℝ₊ → BanachContractionReport
    """

    @classmethod
    def audit(
        cls,
        handoff: SeedHandoff,
        eta_star: float = 1.5,
    ) -> BanachContractionReport:
        return BanachContractionAlgebra.audit(
            handoff.rho_seed, eta_star, handoff.mac_gamma,
        )


# ── §2.4 CropGrowthPipeline — HAND-OFF FASE 2 → FASE 3 ───────────────────
@dataclass(frozen=True, slots=True)
class CropGrowthBundle:
    r"""
    Objeto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Producto de los funtores Riego ⊗ Luz ⊗ Disciplina aplicados al
    SeedHandoff de FASE 1.
    """
    handoff: SeedHandoff
    watering: WateringReport
    rho_illum: np.ndarray
    brockett_c: BrockettPurificationCertificate
    renyi_c: RenyiPurificationCertificate
    fock_c: FockAnnihilationCertificate
    discipline: BanachContractionReport

    def phase_content_bytes(self) -> bytes:
        """Digest firmable para la cadena Merkle de fases."""
        return hashlib.sha256(
            self.handoff.spectral_hash.encode("ascii")
            + np.ascontiguousarray(self.rho_illum).tobytes()
            + f"{self.watering.fat_reduction_pct:.10f}".encode("ascii")
            + f"{self.discipline.seed_spectral_radius:.10f}".encode("ascii")
            + f"{self.discipline.mac_contraction_coef:.10f}".encode("ascii")
            + f"{self.renyi_c.purity_after:.10f}".encode("ascii")
            + f"{self.fock_c.resonance_residual:.10f}".encode("ascii")
        ).digest()


class CropGrowthPipeline:
    r"""
    Orquestador determinista del crecimiento (funtor F₂).

        synthesize : SeedHandoff × Σ* × Σ* × ℝ₊ × ℝ₊ → CropGrowthBundle

    ────────────────────────────────────────────────────────────────────────
    HAND-OFF FORMAL  FASE 2 → FASE 3
    ────────────────────────────────────────────────────────────────────────
    synthesize es el morfismo terminal de la FASE 2.  Su imagen
    CropGrowthBundle es el dominio de TODOS los métodos de la FASE 3.

    Identidad de anidamiento:

        certify ∘ synthesize ∘ build  :  Crystal×Σ*×Σ* → Harvest.
    """

    @classmethod
    def synthesize(
        cls,
        handoff: SeedHandoff,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
        eta_star: float = 1.5,
    ) -> CropGrowthBundle:
        r"""
        Cierra la FASE 2.  Abre la FASE 3.

            watering     = apply_water(handoff, toon, json)     §2.1
            (ρ, B, R, F) = apply_light(handoff, α)              §2.2
            discipline   = audit(handoff, η*)                   §2.3
        """
        watering = CognitiveWateringModule.apply_water(
            handoff, toon_str, base_json_str,
        )
        rho_illum, brockett_c, renyi_c, fock_c = (
            CognitiveIlluminationModule.apply_light(handoff, renyi_alpha)
        )
        discipline = CognitiveDisciplineModule.audit(handoff, eta_star)
        return CropGrowthBundle(
            handoff=handoff,
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
        Último morfismo de la FASE 2  ∧  primero de la FASE 3.

        Adjudica en Ω₃ y dispara la Fe.  La cosecha y el pasaporte se
        cristalizan aguas arriba en el soberano (posee MAC y Merkle).
        """
        verdict = HeytingCropAdjudicator.adjudicate(bundle, external_verdict)
        audit = bundle.handoff.seed_audit
        reason = (
            f"{reason_prefix}::water={bundle.watering.local_verdict.name} "
            f"disc={bundle.discipline.local_verdict.name} "
            f"ρ_seed={bundle.discipline.seed_spectral_radius:.4f} "
            f"ρ_MAC={bundle.discipline.mac_contraction_coef:.4f} "
            f"fock_res={bundle.fock_c.resonance_residual:.4f} "
            f"flag_ok={audit.consistent} "
            f"false_vac={audit.false_vacuum_claim} "
            f"pure_ok={audit.verified_vacuum_pure}"
        )
        actuation = CognitiveFaithModule.fire(verdict, reason)
        return verdict, actuation


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · FE + ADJUDICACIÓN + PASAPORTE                                    ║
# ║                                                                           ║
# ║ Dominio = CropGrowthBundle (codominio de §2.4 synthesize).                ║
# ║ Codominio = CropHarvestYield × CropSovereignGovernancePassport.           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en Ω₃ ────────────────────────────────────────────────
class HeytingCropAdjudicator:
    r"""
    Lógica interna del topos: colapsa el Bundle a un único valor de Ω₃
    por meets sucesivos (producto de subobjetos):

        local = water ∧ discipline ∧ fock ∧ brockett ∧ renyi
                ∧ flag ∧ vacuum_claim
        final = local ∧ external          (meet conservador, nunca infla)

    Clasificadores:
        fock           : annihilated ↦ ⊤  else ⋆
        brockett       : convergencia ∨ drift < 10⁻³ ↦ ⊤  else ⋆
        renyi          : purity_gain ≥ 0 ↦ ⊤  else ⋆
        flag           : consistent ↦ ⊤  else ⋆
        vacuum_claim   : false_vacuum_claim ↦ ⊥  else ⊤
                         (el testimonio falso es el único veto de semilla;
                          no ser vacuum-pure NO es defecto: la mayoría
                          de estados de 𝔇_n no lo son)
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
    def _flag_rule(cls, bundle: CropGrowthBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if bundle.handoff.seed_audit.consistent
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _vacuum_claim_rule(cls, bundle: CropGrowthBundle) -> HeytingOmega3:
        if bundle.handoff.seed_audit.false_vacuum_claim:
            return HeytingOmega3.VETOED
        return HeytingOmega3.COHERENT

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
            .meet(cls._flag_rule(bundle))
            .meet(cls._vacuum_claim_rule(bundle))
        )
        return local.meet(external_verdict)


# ── §3.2 FE — interlock ciber-físico ESP32 Crowbar ────────────────────────
@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    r"""
    Actuación física (simulada) del crowbar ESP32.

        interlock_fired      : True ⟺ verdict = ⊥
        actuation_latency_ns : cota de diseño < 400 ns (nominal 392.15 ns)
        provenance_hash      : SHA-256(reason ‖ t_ns)
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

    Si Ω₃ = ⊥ se arma el crowbar:

        GPIO14 → HIGH  ⇒  MOSFET BT151  ⇒  latencia < 400 ns.

    Este módulo no emite I/O de hardware; certifica la decisión y su
    provenance.  La «fe» es el invariante de arquitectura: el corte
    ocurre antes de que la anomalía se propague.
    """
    GPIO_PIN: Final[str] = "GPIO14"
    DEVICE: Final[str] = "BT151_CROWBAR"
    NOMINAL_LATENCY_NS: Final[float] = 392.15

    @classmethod
    def fire(
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
            "[CROP COGNITIVO — CROWBAR] %s → HIGH | %.2f ns | razón=%s",
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


# ── §3.3 Cosecha + Pasaporte con Merkle root ─────────────────────────────
@dataclass(frozen=True, slots=True)
class CropHarvestYield:
    r"""Cosecha firmada de un ciclo F₃∘F₂∘F₁."""
    crop_id: str
    seed_crystal_id: str
    origin_agent: str
    purity_gain: float
    entropy_reduction: float
    mac_update_certificate: MacUpdateCertificate
    fidelity_after_update: float
    is_inoculated_into_mac: bool
    watering_kv_ratio: float
    banach_seed_radius: float
    banach_mac_coef: float
    brockett_isospectral_drift: float
    fock_photons: int
    heyting_verdict: HeytingOmega3
    crowbar_report: CrowbarActuationReport
    content_hash: str
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class CropSovereignGovernancePassport:
    r"""Pasaporte de gobernanza: agregado MAC × Merkle × Ω₃ global."""
    passport_id: str
    sovereign_agent_id: str
    total_crops_cultivated: int
    active_coherent_crops: int
    coherent_fraction: float
    aggregate_purity: float
    aggregate_entropy: float
    aggregate_fidelity_to_ground: float
    mac_spectral_gap: float
    global_heyting_verdict: HeytingOmega3
    crowbar_protection_active: bool
    field_merkle_root: str
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


def _merkle_root(hashes: List[str]) -> str:
    r"""
    Merkle-SHA-256 sobre hashes hex.  Lista vacía → SHA-256(b"EMPTY").
    Capa impar: se duplica la hoja derecha (Bitcoin-style), de modo que
    el árbol sea siempre binario completo.  Las hojas se consumen en el
    orden de inserción (custodia temporal, no lexicográfica).
    """
    if not hashes:
        return hashlib.sha256(b"EMPTY").hexdigest()
    layer = [bytes.fromhex(h) for h in hashes]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0].hex()


# ── §3.4 Soberano del Cultivo Cognitivo ──────────────────────────────────
class TOONCognitiveCropAgent:
    r"""
    Soberano de Calibre del Cultivo Cognitivo Dinámico.

    Funtor soberano  F = F₃ ∘ F₂ ∘ F₁ :

        F₁  SeedHandoff.build
        F₂  CropGrowthPipeline.synthesize
        F₃  adjudicate ⊗ fire ⊗ inoculate ⊗ certify

    Asociatividad (teorema de anidamiento):

        sow_and_cultivate
            = _phase3_harvest ∘ _phase2_grow ∘ _phase1_handoff
            = certify ∘ synthesize ∘ build.
    """

    def __init__(
        self,
        agent_id: str = "CROP-SOVEREIGN-SABIO-01",
        dimension_mac: int = 4,
        mac_gamma: float = 0.2,
        eta_star: float = 1.5,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
        field_coupling: float = 1e-3,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1")
        self.agent_id = agent_id
        self.dimension_mac = int(dimension_mac)
        self.mac_gamma = float(np.clip(mac_gamma, 0.0, 1.0))
        self.eta_star = float(eta_star)
        self.renyi_alpha = float(renyi_alpha)

        n = self.dimension_mac
        base = np.diag(np.linspace(0.0, 3.0, n)).astype(np.complex128)
        off = np.zeros((n, n), dtype=np.complex128)
        if n >= 2:
            idx = np.arange(n - 1)
            off[idx, idx + 1] = float(field_coupling)
        self.H_mac: ComplexMatrix = base + off + off.conj().T
        self.ground_projector: ComplexMatrix = (
            DensityOperatorAlgebra.ground_state_projector(self.H_mac)
        )

        rho0 = np.eye(n, dtype=np.complex128) / n
        self.mac_field = MacStateField(rho0, gamma=self.mac_gamma)
        self.cultivated_crops_history: List[CropHarvestYield] = []
        self._chain_hash = hashlib.sha256(
            f"{agent_id}::GENESIS::n={n}::γ={self.mac_gamma}".encode("ascii")
        ).hexdigest()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _content_hash(
        self, bundle: CropGrowthBundle, verdict: HeytingOmega3
    ) -> str:
        return hashlib.sha256(
            self.agent_id.encode("ascii")
            + bundle.handoff.seed_id.encode("ascii")
            + bundle.handoff.spectral_hash.encode("ascii")
            + bundle.phase_content_bytes()
            + verdict.name.encode("ascii")
        ).hexdigest()

    def _phase1_handoff(self, seed_crystal: SeedCrystal) -> SeedHandoff:
        """FASE 1 anidada: cierra con SeedHandoff (dominio de FASE 2)."""
        handoff = SeedHandoff.build(
            seed=seed_crystal,
            mac_field=self.mac_field,
            ground_projector=self.ground_projector,
        )
        self._advance_chain("F1", bytes.fromhex(handoff.spectral_hash))
        return handoff

    def _phase2_grow(
        self,
        handoff: SeedHandoff,
        toon_str: str,
        base_json_str: str,
    ) -> CropGrowthBundle:
        """FASE 2 anidada: continuación de build; cierra con Bundle."""
        bundle = handoff.continue_into_phase2(
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=self.renyi_alpha,
            eta_star=self.eta_star,
        )
        self._advance_chain("F2", bundle.phase_content_bytes())
        return bundle

    def _phase3_harvest(
        self,
        bundle: CropGrowthBundle,
        external_verdict: HeytingOmega3,
    ) -> CropHarvestYield:
        """FASE 3 anidada: continuación de synthesize; cierra con Harvest."""
        final_verdict, actuation = CropGrowthPipeline.continue_into_phase3(
            bundle, external_verdict
        )
        self._advance_chain(
            "F3",
            f"{final_verdict.name}|{actuation.provenance_hash}".encode("ascii"),
        )

        is_inoculate = (final_verdict == HeytingOmega3.COHERENT)
        if is_inoculate:
            mac_cert = self.mac_field.update(bundle.rho_illum)
        else:
            mac_cert = self.mac_field.idle_certificate()

        fid_after = DensityOperatorAlgebra.uhlmann_fidelity(
            self.mac_field.rho, bundle.rho_illum,
        )
        p_before = bundle.brockett_c.initial_purity
        p_after = DensityOperatorAlgebra.purity(bundle.rho_illum)
        purity_gain = float(p_after - p_before)
        s_before = float(bundle.handoff.seed_audit.entropy)
        s_after = DensityOperatorAlgebra.von_neumann_entropy(bundle.rho_illum)
        entropy_reduction = float(s_before - s_after)

        content_hash = self._content_hash(bundle, final_verdict)
        provenance = _sha256_bytes(
            self.agent_id.encode("ascii"),
            bundle.handoff.seed_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{purity_gain:.10f}".encode("ascii"),
            f"{entropy_reduction:.10f}".encode("ascii"),
            self._chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )
        crop_id = f"CROP-SOVEREIGN-{len(self.cultivated_crops_history) + 1:04d}"
        harvest = CropHarvestYield(
            crop_id=crop_id,
            seed_crystal_id=bundle.handoff.seed_id,
            origin_agent=bundle.handoff.origin_agent,
            purity_gain=purity_gain,
            entropy_reduction=entropy_reduction,
            mac_update_certificate=mac_cert,
            fidelity_after_update=fid_after,
            is_inoculated_into_mac=is_inoculate,
            watering_kv_ratio=bundle.watering.kv_compression_ratio,
            banach_seed_radius=bundle.discipline.seed_spectral_radius,
            banach_mac_coef=bundle.discipline.mac_contraction_coef,
            brockett_isospectral_drift=bundle.brockett_c.isospectral_drift,
            fock_photons=bundle.fock_c.gamma_photons_emitted,
            heyting_verdict=final_verdict,
            crowbar_report=actuation,
            content_hash=content_hash,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )
        self.cultivated_crops_history.append(harvest)
        return harvest

    def sow_and_cultivate(
        self,
        seed_crystal: SeedCrystal,
        toon_str: str,
        base_json_str: str,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> CropHarvestYield:
        r"""Ciclo soberano: F₃ ∘ F₂ ∘ F₁."""
        t_start = time.perf_counter()
        logger.info(
            "═══ Siembra | semilla=%s | origen=%s | γ_MAC=%.2f ═══",
            seed_crystal.crystal_id, seed_crystal.origin_agent, self.mac_gamma,
        )
        handoff = self._phase1_handoff(seed_crystal)
        bundle = self._phase2_grow(handoff, toon_str, base_json_str)
        harvest = self._phase3_harvest(bundle, external_verdict)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cosecha %s | Ω₃=%s | ΔP=%+.6f | ΔS=%+.6f | inoculada=%s | %.2f ms",
            harvest.crop_id, harvest.heyting_verdict.name,
            harvest.purity_gain, harvest.entropy_reduction,
            harvest.is_inoculated_into_mac, dt_ms,
        )
        return harvest

    def audit_field_governance(self) -> CropSovereignGovernancePassport:
        total = len(self.cultivated_crops_history)
        coherent = sum(
            1 for c in self.cultivated_crops_history
            if c.heyting_verdict == HeytingOmega3.COHERENT
        )
        coherent_frac = (coherent / total) if total > 0 else 0.0

        w = DensityOperatorAlgebra.spectrum_descending(self.mac_field.rho)
        aggregate_purity = float(np.sum(w ** 2))
        aggregate_entropy = float(-np.sum(w * np.log(np.maximum(w, _EPS))))
        aggregate_fidelity = DensityOperatorAlgebra.uhlmann_fidelity(
            self.mac_field.rho, self.ground_projector,
        )
        mac_gap = float(w[0] - w[1]) if w.size > 1 else float("inf")

        crowbar_active = any(
            c.heyting_verdict == HeytingOmega3.VETOED
            for c in self.cultivated_crops_history
        )
        if total == 0:
            global_verdict = HeytingOmega3.DEGRADED
        elif crowbar_active:
            global_verdict = HeytingOmega3.VETOED
        elif coherent_frac >= 0.7:
            global_verdict = HeytingOmega3.COHERENT
        else:
            global_verdict = HeytingOmega3.DEGRADED

        merkle = _merkle_root(
            [c.content_hash for c in self.cultivated_crops_history]
        )
        passport_id = f"PASSPORT-WISDOM-CROP-{int(time.time())}"
        signature = _sha256_bytes(
            self.agent_id.encode("ascii"),
            passport_id.encode("ascii"),
            global_verdict.name.encode("ascii"),
            merkle.encode("ascii"),
            f"{aggregate_purity:.10f}".encode("ascii"),
            f"{aggregate_entropy:.10f}".encode("ascii"),
            f"{coherent_frac:.10f}".encode("ascii"),
            self._chain_hash.encode("ascii"),
        )
        return CropSovereignGovernancePassport(
            passport_id=passport_id,
            sovereign_agent_id=self.agent_id,
            total_crops_cultivated=total,
            active_coherent_crops=coherent,
            coherent_fraction=coherent_frac,
            aggregate_purity=aggregate_purity,
            aggregate_entropy=aggregate_entropy,
            aggregate_fidelity_to_ground=aggregate_fidelity,
            mac_spectral_gap=mac_gap,
            global_heyting_verdict=global_verdict,
            crowbar_protection_active=crowbar_active,
            field_merkle_root=merkle,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=signature,
            timestamp_utc=time.time(),
        )


# ── §3.5 Demostración autónoma ───────────────────────────────────────────
def _build_seed_matrix(
    n: int,
    alpha: float,
    key: str,
    align_with: Optional[np.ndarray] = None,
    ground_mix: float = 0.0,
) -> ComplexMatrix:
    r"""
    Semilla determinista de espectro softmax (familia exponencial):

        w_k = exp(α (n − k)) / Z,    k = 1…n.

        α = 0  →  λ = 1/n     (máximamente mezclada)
        α → ∞ →  λ → e₁      (espectralmente pura)

    Se sumerge en 𝔇_n por QR de Ginibre → Haar en U(n) (Stewart 1980).
    Si `align_with` es un proyector y ground_mix ∈ (0,1], se interpola

        ρ ← (1−μ) ρ_Haar + μ |Ω⟩⟨Ω|

    para que «vacuum-pure» sea físicamente el ground de H_mac, no un
    puro Haar-aleatorio (cuya F(ρ,|Ω⟩) es 1/n en esperanza).
    """
    rng = np.random.default_rng(_seed_from_string(key))
    idx = np.arange(n)
    raw = np.exp(alpha * (n - idx))
    w = raw / raw.sum()
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    rho = (Q * w.astype(np.complex128)) @ Q.conj().T
    rho = DensityOperatorAlgebra.sanitize(rho)
    mu = float(np.clip(ground_mix, 0.0, 1.0))
    if align_with is not None and mu > 0.0:
        ground = DensityOperatorAlgebra.sanitize(align_with)
        rho = DensityOperatorAlgebra.sanitize((1.0 - mu) * rho + mu * ground)
    return rho


if __name__ == "__main__":
    print("═" * 92)
    print("SOBERANO DEL CULTIVO — v2.2.0 Nested Doctoral")
    print("Banach · Brockett-U(n) · Rényi · MAC-Φ_γ · Heyting Ω₃ · Merkle")
    print("═" * 92)

    sovereign = TOONCognitiveCropAgent(
        agent_id="CROP-SOVEREIGN-SABIO-01",
        dimension_mac=4,
        mac_gamma=0.2,
        eta_star=1.5,
        renyi_alpha=1.5,
        field_coupling=1e-3,
    )

    toon_str = "[APU: 2.1.4|CONCRETO 3000PSI] COST: 485000 COP"
    base_json_str = json.dumps({
        "apu_code": "2.1.4-CONCRETO-3000PSI",
        "unit_cost": 485000.0,
        "metadata": {
            "schema": "fat_json_structure_verbose_keys",
            "authority": "Sovereign-Crop-Agent",
            "redundant": "eliminable_by_toon_tabularization",
        },
    }, indent=2)

    print("\n─── Sanity-check del campo MAC ───")
    print(f"H_mac  ground E₀ = {float(np.linalg.eigvalsh(sovereign.H_mac).min()):.6f}")
    print(
        f"MAC γ           = {sovereign.mac_gamma}  "
        f"(Lipschitz coef = {abs(1.0 - sovereign.mac_gamma):.4f})"
    )

    scenarios = [
        ("COHERENT-VACUUM  (α=6, flag=True, mix=|Ω⟩)",
         6.0, "SEED-VACUUM", True, HeytingOmega3.COHERENT, 0.999),
        ("DEGRADED-MIXED   (α=0, flag=False)",
         0.0, "SEED-MIXED", False, HeytingOmega3.COHERENT, 0.0),
        ("VETOED-FALSE-FLAG (α=0, flag=True)",
         0.0, "SEED-FALSE-FLAG", True, HeytingOmega3.COHERENT, 0.0),
    ]

    for name, alpha, key, flag, ext, gmix in scenarios:
        rho_crystal = _build_seed_matrix(
            n=4, alpha=alpha, key=key,
            align_with=sovereign.ground_projector,
            ground_mix=gmix,
        )
        seed = SeedCrystal(
            crystal_id=f"CRYSTAL-{key}",
            origin_agent="TOON-SILENT-WITNESS-01",
            density_matrix=rho_crystal,
            experience_vector=np.array(
                [2.0, 0.89, 0.9999 if flag else 0.25, 1.0], dtype=np.float64
            ),
            is_vacuum_pure=flag,
        )
        harvest = sovereign.sow_and_cultivate(
            seed_crystal=seed,
            toon_str=toon_str,
            base_json_str=base_json_str,
            external_verdict=ext,
        )
        print(f"\n[{name}]")
        print(f"   crop_id               : {harvest.crop_id}")
        print(f"   Ω₃ final              : {harvest.heyting_verdict.name}")
        print(f"   Riego KV ratio        : {harvest.watering_kv_ratio:.4f}")
        print(
            f"   Banach ρ(T_seed)      : {harvest.banach_seed_radius:.6f}  "
            f"Lip(Φ_γ)={harvest.banach_mac_coef:.4f}"
        )
        print(f"   ΔPureza               : {harvest.purity_gain:+.6f}")
        print(f"   ΔEntropía             : {harvest.entropy_reduction:+.6f}")
        print(f"   Fotones γ             : {harvest.fock_photons}")
        print(
            f"   Drift isospectral     : {harvest.brockett_isospectral_drift:.2e}"
        )
        print(f"   Inoculada en MAC      : {harvest.is_inoculated_into_mac}")
        print(f"   Fid MAC↔illum         : {harvest.fidelity_after_update:.6f}")
        print(
            f"   Crowbar               : {harvest.crowbar_report.interlock_fired}  "
            f"({harvest.crowbar_report.actuation_latency_ns:.2f} ns)"
        )
        print(f"   Firma SHA-256         : {harvest.sha256_provenance[:32]}…")

    print("\n" + "─" * 92)
    print("─── PASAPORTE DE GOBERNANZA GLOBAL ───")
    passport = sovereign.audit_field_governance()
    print(f"   passport_id           : {passport.passport_id}")
    print(
        f"   total / coherentes    : {passport.total_crops_cultivated} / "
        f"{passport.active_coherent_crops}"
    )
    print(f"   fracción coherente    : {passport.coherent_fraction:.4f}")
    print(f"   P(ρ_MAC)              : {passport.aggregate_purity:.6f}")
    print(f"   S(ρ_MAC)              : {passport.aggregate_entropy:.6f}")
    print(f"   F(ρ_MAC, |Ω⟩)         : {passport.aggregate_fidelity_to_ground:.6f}")
    print(f"   gap(ρ_MAC)            : {passport.mac_spectral_gap:.6f}")
    print(f"   Ω₃ GLOBAL             : {passport.global_heyting_verdict.name}")
    print(f"   Crowbar protegido     : {passport.crowbar_protection_active}")
    print(f"   Merkle root (field)   : {passport.field_merkle_root[:32]}…")
    print(f"   Firma global          : {passport.sha256_provenance[:32]}…")

    print("\n" + "═" * 92)
    print("✓ F1→F2: build ⊣ continue_into_phase2 = synthesize.")
    print("✓ F2→F3: synthesize ⊣ continue_into_phase3 = adjudicate ⊗ fire.")
    print("✓ Riego: Shannon H₂ + BPE Ω(⌈n/4⌉), funcional convexo Δ_gr.")
    print("✓ Luz: Brockett en U(n) (polar) + Rényi Φ_α + matching Fock.")
    print("✓ Disciplina: ρ(T)=max|1−η g_ij| vectorizado ⊕ Lip(Φ_γ)=|1−γ|.")
    print("✓ Flag vacuum-pure: P∧F(|Ω⟩); falso testimonio ⇒ ⊥.")
    print("✓ Merkle-SHA-256 sobre cosechas + cadena de fases.")
    print("═" * 92)
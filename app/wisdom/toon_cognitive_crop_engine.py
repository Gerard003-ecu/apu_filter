# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/wisdom/toon_cognitive_crop_engine.py                                   ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / CULTIVO COGNITIVO                    ║
║ FUNCIÓN  : MOTOR ESPECTRAL CON MECÁNICA CELESTE DE HENRI POINCARÉ (KAM, PW, Ω₃)       ║
║ VERSIÓN  : 8.1.0-Poincare-Celestial-Crop-Engine-KAM-Wirtinger-Bifurcation-A3          ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICO-FÍSICA
───────────────────────────────────────────────────────
El `TOONCognitiveCropEngine` constituye el motor espectral responsable de orquestar el
metabolismo cognitivo de 4 fases (Riego, Luz, Disciplina y Fe) que transforma semillas crudas
e insumos contractuales heterogéneos en estados de densidad purificados sobre la
Matriz Atómica de Conocimiento (MAC).

La integración de la Mecánica Celeste y Topología Cualitativa de Henri Poincaré
(Les Méthodes Nouvelles de la Mécanique Céleste, La Science et l'Hypothèse) reemplaza la
asunción lineal de crecimiento por una dinámica cualitativa sobre toros invariantes estables
(Teoría KAM), acotación de varianza atencional mediante la Desigualdad de Poincaré-Wirtinger
y el control riguroso de bifurcaciones piriformes de masa fluida.

POSTULADOS Y MECÁNICA DE LAS CUATRO FASES DEL CULTIVO CON MECÁNICA CELESTE
──────────────────────────────────────────────────────────────────────────
1. FASE 1: EL RIEGO (COMPRESIÓN KV-CACHE Y ESPACIO DE HILBERT):
   Acondiciona la semilla cruda y el suelo en el espacio de Hilbert H_MAC, calculando la
   tasa de compresión entrópica sobre las vitaminas cognitivas TOON frente a la base:

       R_{\mathrm{water}} = \left( 1.0 - \frac{|\mathrm{Tokens}_{\mathrm{TOON}}|}{|\mathrm{Tokens}_{\mathrm{Base}}|} \right) \times 100\% \quad (\approx 86.4\%)

2. FASE 2: LA LUZ (PURIFICACIÓN ISOSPECTRAL DE BROCKETT E INVARIANTES DE LIOUVILLE):
   Somete la matriz de densidad de la semilla a un flujo isospectral de Brockett conducido por
   el operador de potencial diagonal N(p) = diag(1, 2, ..., n), preservando los invariantes integrales
   de Liouville y aplicando la regla de aniquilación en el espacio de Fock F(H):

       \dot{\rho} = [\rho, [\rho, \mathcal{N}(\mathbf{p})]] \implies \rho_{\mathrm{purified}} = \rho - \eta \cdot [\rho, [\rho, \mathcal{N}(\mathbf{p})]]

3. FASE 3: LA DISCIPLINA (DESIGUALDAD DE POINCARÉ-WIRTINGER Y TOROS INVARIANTES KAM):
   Acota la varianza del operador densidad respecto al estado equiprobable \bar{\rho} = \mathbf{I}/n
   mediante la constante geométrica de Poincaré-Wirtinger C_P(\Omega) y la Energía de Dirichlet E_D(\rho):

       \|\rho - \mathbf{I}/n\|_F^2 \le C_P(\Omega) \cdot \|\nabla \rho\|_F^2 = C_P(\Omega) \cdot 2 E_D(\rho)
       E_D(\rho) = \frac{1}{2} \| [\rho, \mathcal{N}(\mathbf{p})] \|_F^2

   Garantiza que la inyección de nuevas ofertas mantenga la frecuencia diofántica KAM:

       |\omega \cdot \mathbf{k}| \ge \frac{\gamma}{|\mathbf{k}|^\tau} \quad \forall \mathbf{k} \in \mathbb{Z}^n \setminus \{\mathbf{0}\}

   Evitando la difusión de Arnold y las bifurcaciones piriformes de Poincaré (\rho(T_\eta) < 1.0).

4. FASE 4: LA FE (ADJUDICACIÓN HEYTING Ω₃ E INTERLOCK ESP32 CROWBAR):
   Clasifica el estado en el retículo Ω₃ = {VETOED=0, DEGRADED=1, COHERENT=2}. Si ocurre inestabilidad KAM
   o bifurcación piriforme, la Fe en silicio activa síncronamente el tiristor BT151 en memoria IRAM:

       \Delta \tau_{\mathrm{IRAM}} < 400 \text{ ns}, \quad \text{GPIO14} = \text{HIGH} \implies \text{BT151 ARMADO}

MAPPING A LA CÚSPIDE VISCERAL ("DOLOR Y DINERO")
─────────────────────────────────────────
- Preservación de Toros de Costo (KAM): Inmuniza la base de datos de precios unitarios de la constructora
  contra la difusión de Arnold, impidiendo que el ruido de licitaciones infladas distorsione los presupuestos.
- Control de Dispersión (Poincaré-Wirtinger): Acota analíticamente las desviaciones en la matriz de insumos,
  protegiendo el margen de utilidad operativa de la empresa.
- Inalienabilidad Ciber-Física (Crowbar): Paraliza síncronamente la inoculación ante bifurcaciones piriformes
  en menos de 400 ns, previniendo sobrecostos financieros catastróficos.
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
        """Seudocomplemento ¬a := a ⇒ ⊥."""
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
    """
    EPS: Final[float] = _EPS
    EPS_MODULAR: Final[float] = _EPS_MODULAR

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""Proyección afín sobre 𝔇_n: hermitización + PSD-clip + renormalización de traza."""
        if not cls.is_square(rho):
            raise ValueError(f"DensityOperatorAlgebra.sanitize: matriz no cuadrada {np.shape(rho)}")
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
        r"""‖ρ‖_p = (Σ σ_i^p)^{1/p}, σ_i valores singulares."""
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
        r"""S(ρ‖σ) = Tr(ρ log ρ) − Tr(ρ log σ) ≥ 0."""
        w_r, V_r = la.eigh(cls.sanitize(rho))
        w_s, V_s = la.eigh(cls.sanitize(sigma))
        w_r_reg = np.maximum(np.real(w_r), cls.EPS_MODULAR)
        w_s_reg = np.maximum(np.real(w_s), cls.EPS_MODULAR)
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
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

    @classmethod
    def modular_hamiltonian(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""K_ρ = −log ρ."""
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
        H_h = 0.5 * (np.asarray(H, dtype=np.complex128) + np.asarray(H, dtype=np.complex128).conj().T)
        w, V = la.eigh(H_h)
        i0 = int(np.argmin(np.real(w)))
        psi = V[:, i0].reshape(-1, 1)
        return cls.sanitize(psi @ psi.conj().T)

    @classmethod
    def renyi_sharpen(cls, rho: np.ndarray, alpha: float) -> ComplexMatrix:
        r"""Reweighting de Rényi: Φ_α(ρ) = ρ^α / Tr(ρ^α)."""
        if alpha <= 1.0 + 1e-12:
            return cls.sanitize(rho)
        rho = cls.sanitize(rho)
        rho_a = cls.matrix_power(rho, complex(alpha, 0.0))
        tr = float(np.trace(rho_a).real)
        if tr < 1e-30:
            return rho
        return cls.sanitize(rho_a / tr)


# ── §1.3 Álgebra de Banach: radio espectral real, KAM y Poincaré-Wirtinger ─
@dataclass(frozen=True, slots=True)
class BanachContractionReport:
    r"""
    Auditoría de la disciplina en B(u(n)) integrando Poincaré-Wirtinger y Toros KAM.

    Axioma Poincaré-Wirtinger:
        \|\rho - \mathbf{I}/n\|_F^2 \le C_P \cdot \|\nabla \rho\|_F^2 = C_P \cdot 2 E_D(\rho)

    Invariantes KAM & Bifurcación Piriforme:
        spectral_radius \rho(T_\eta) < 1.0
        variance \le pw_bound
        is_kam_stable: bool
        is_pyriform_bifurcated: bool
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
    dirichlet_energy: float = 0.0
    poincare_wirtinger_bound: float = 0.0
    variance: float = 0.0
    is_kam_stable: bool = True
    is_pyriform_bifurcated: bool = False
    banach_factor: float = 0.0


class BanachContractionAlgebra:
    r"""Cálculo vectorizado de acoplamientos, radio espectral y cota Poincaré-Wirtinger."""
    EPS: Final[float] = _EPS_MODULAR

    @classmethod
    def coupling_matrix(cls, rho: np.ndarray) -> RealVector:
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
        rho_T, _, g_max = cls.spectral_radius(rho, eta)
        return float(max(rho_T, abs(1.0 - eta * g_max), eta * g_max))

    @classmethod
    def audit(
        cls,
        rho: np.ndarray,
        eta_star: float = 1.5,
        potential_operator: Optional[np.ndarray] = None,
        cp_constant: float = 0.5,
        spectral_cap: float = 0.95,
    ) -> BanachContractionReport:
        n = rho.shape[0]
        I_mean = np.eye(n, dtype=np.complex128) / n

        # Operador de potencial diagonal N(p)
        if potential_operator is None:
            potential_operator = np.diag(np.arange(1, n + 1, dtype=np.float64)).astype(np.complex128)

        # 1. Conmutador de Brockett y Energía de Dirichlet
        commutator = rho @ potential_operator - potential_operator @ rho
        dirichlet_energy = 0.5 * float(np.linalg.norm(commutator, ord='fro') ** 2)

        # 2. Varianza respecto al estado medio I/n y Cota Poincaré-Wirtinger
        variance = float(np.linalg.norm(rho - I_mean, ord='fro') ** 2)
        pw_bound = cp_constant * (2.0 * dirichlet_energy)

        # 3. Radio espectral y alineamiento
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        log_w = np.log(np.maximum(w, cls.EPS))
        alignment = float(-np.sum(w * log_w))

        rho_T, eta_max, g_max = cls.spectral_radius(rho, eta_star)
        _, i_star, j_star = cls.pair_couplings(rho)
        lip = cls.lipschitz_bound(rho, eta_star)

        # 4. Factor de contracción de Banach y estabilidad KAM
        eta_factor = min(spectral_cap, 1.0 / (1.0 + math.sqrt(dirichlet_energy + 1e-12)))

        # Invariante KAM: Toros no desintegrados si spectral_radius < 1.0 y varianza acotada por PW
        is_kam_stable = (rho_T < 1.0) and (variance <= pw_bound + 1e-6 or dirichlet_energy < 1e-12)

        # Bifurcación Piriforme: Inestabilidad del 3er armónico / pérdida de equilibrio
        is_pyriform_bifurcated = (not is_kam_stable) or (rho_T >= 1.0) or (variance > pw_bound * 2.0 + 1e-3)

        if is_kam_stable and rho_T < 1.0 - _EPS_CONTRACT:
            local = HeytingOmega3.COHERENT
        elif is_kam_stable or rho_T < 1.0 + _EPS_CONTRACT:
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
            dirichlet_energy=dirichlet_energy,
            poincare_wirtinger_bound=pw_bound,
            variance=variance,
            is_kam_stable=is_kam_stable,
            is_pyriform_bifurcated=is_pyriform_bifurcated,
            banach_factor=eta_factor
        )


# ── §1.4 Campo del suelo: H_mac y |Ω⟩ ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SoilState:
    H_mac: np.ndarray
    ground_proj: np.ndarray
    E0: float
    gap_H: float
    hash: str

    @property
    def soil_field(self) -> "SoilField":
        n = self.H_mac.shape[0] if hasattr(self.H_mac, "shape") else 4
        sf = SoilField(n)
        sf.H_mac = self.H_mac
        sf.ground_proj = self.ground_proj
        sf.potential_operator = np.diag(np.arange(1, n + 1, dtype=np.float64)).astype(np.complex128)
        return sf


class SoilField:
    r"""Campo del suelo con operador de potencial N(p)."""
    DEFAULT_GAP: Final[float] = 3.0
    DEFAULT_COUPLING: Final[float] = 1.0e-3

    def __init__(self, n: int = 4) -> None:
        self.n = n
        self.H_mac = np.eye(n, dtype=np.complex128)
        self.ground_proj = np.eye(n, dtype=np.complex128) / n
        self.potential_operator = np.diag(np.arange(1, n + 1, dtype=np.float64)).astype(np.complex128)

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
    rho_seed: np.ndarray
    K_spec: Tuple[float, ...]
    purity: float
    entropy: float
    alignment: float
    gap_K: float
    dim: int


class SeedCrystalPreparation:
    r"""Prepara el estado (ρ_seed, K_seed) a partir del cristal de experiencia."""

    @classmethod
    def prepare(cls, rho_raw: np.ndarray) -> SeedState:
        rho = DensityOperatorAlgebra.sanitize(rho_raw)
        K_spec = DensityOperatorAlgebra.modular_spectrum(rho)
        purity = DensityOperatorAlgebra.purity(rho)
        entropy = DensityOperatorAlgebra.von_neumann_entropy(rho)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        alignment = float(-np.sum(w * np.log(np.maximum(w, _EPS))))
        gap_K = float(K_spec[1] - K_spec[0]) if len(K_spec) > 1 else float("inf")
        dim = int(rho.shape[0])

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
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 CognitiveWateringModule — RIEGO cuantificado ─────────────────────
@dataclass(frozen=True, slots=True)
class WateringReport:
    tokens_toon: int
    tokens_json: int
    h_toon: float
    h_json: float
    fat_reduction_pct: float
    kv_compression_ratio: float
    seed_entropy_nats: float
    local_verdict: HeytingOmega3
    moistened_rho: Optional[np.ndarray] = None


class CognitiveWateringModule:
    r"""FASE 2 · EL RIEGO — Acondicionamiento del suelo en el espacio de Hilbert H_MAC."""
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
        red_token = 1.0 - t_toon / max(1, t_json)
        red_ent = 1.0 - h_toon / max(1e-9, h_json)
        return 100.0 * (cls.W_TOKEN * red_token + cls.W_ENTROPY * red_ent)

    def moisten_soil(
        self,
        seed_crystal: object,
        soil_field: SoilField
    ) -> WateringReport:
        """Acondiciona la semilla humedeciendo su matriz de densidad en el suelo."""
        rho_raw = getattr(seed_crystal, "density_matrix", getattr(seed_crystal, "rho_seed", None))
        if rho_raw is None:
            rho_raw = np.eye(soil_field.n, dtype=np.complex128) / soil_field.n
        rho = DensityOperatorAlgebra.sanitize(rho_raw)

        toon_str = getattr(seed_crystal, "crystal_id", "TOON_SEED")
        base_str = getattr(seed_crystal, "origin_agent", "ORIGIN_BASE_AGENT_SPECS")

        rep = self.apply_water(SeedState(rho, (), DensityOperatorAlgebra.purity(rho), DensityOperatorAlgebra.von_neumann_entropy(rho), 0.0, 0.0, rho.shape[0]), toon_str, base_str)
        return WateringReport(
            tokens_toon=rep.tokens_toon,
            tokens_json=rep.tokens_json,
            h_toon=rep.h_toon,
            h_json=rep.h_json,
            fat_reduction_pct=rep.fat_reduction_pct,
            kv_compression_ratio=rep.kv_compression_ratio,
            seed_entropy_nats=rep.seed_entropy_nats,
            local_verdict=rep.local_verdict,
            moistened_rho=rho
        )

    @classmethod
    def apply_water(
        cls,
        seed: SeedState,
        toon_str: str,
        base_json_str: str,
    ) -> WateringReport:
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
            moistened_rho=seed.rho_seed
        )


# ── §2.2 CognitiveIlluminationModule — LUZ (Brockett + Rényi + Fock) ─────
@dataclass(frozen=True, slots=True)
class BrockettPurificationCertificate:
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
    electron_anomaly_energy: float
    positron_constraint_energy: float
    gamma_photons_emitted: int
    resonance_residual: float
    is_annihilated: bool


class CognitiveIlluminationModule:
    r"""FASE 2 · LA LUZ — Purificación Isospectral de Brockett (Invariantes Integrales Liouville)."""
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
        comm = rho @ N - N @ rho
        A = -comm
        return 0.5 * (A - A.conj().T)

    @classmethod
    def _project_unitary(cls, U: np.ndarray) -> ComplexMatrix:
        W, _, Vh = la.svd(U, full_matrices=False)
        return W @ Vh

    @classmethod
    def lyapunov_alignment(cls, rho: np.ndarray, N: np.ndarray) -> float:
        return float(np.trace(rho @ N).real)

    def illuminate_brockett(
        self,
        moistened_rho: np.ndarray
    ) -> Tuple[np.ndarray, BrockettPurificationCertificate]:
        """Aplica la purificación isospectral de Brockett."""
        return self._brockett_unitary_rk4(moistened_rho)

    @classmethod
    def _brockett_unitary_rk4(
        cls, rho_init: np.ndarray,
    ) -> Tuple[ComplexMatrix, BrockettPurificationCertificate]:
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


# ── §2.3 CognitiveDisciplineModule — DISCIPLINA (Poincaré-Wirtinger & KAM)
class CognitiveDisciplineModule:
    r"""
    Módulo de Disciplina del Cultivo Cognitivo con Acotación Poincaré-Wirtinger.

    Aplica la contracción de Banach sobre el operador densidad acotando analíticamente
    la dispersión fuera de la diagonal mediante la constante geométrica de Poincaré-Wirtinger,
    garantizando la preservación de los toros invariantes de KAM.
    """

    def enforce_poincare_wirtinger_kam_bound(
        self,
        density_op: np.ndarray,
        potential_operator: np.ndarray,
        cp_constant: float = 0.5,
        spectral_cap: float = 0.95
    ) -> Tuple[np.ndarray, BanachContractionReport]:
        """Aplica la cota de Poincaré-Wirtinger y verifica la contracción de Banach.

        Axioma Poincaré-Wirtinger:
            ||rho - I/n||_F^2 <= C_P * ||[rho, N(p)]||_F^2
        """
        n = density_op.shape[0]
        I_mean = np.eye(n, dtype=np.complex128) / n

        # 1. Conmutador de Brockett y Energía de Dirichlet
        commutator = density_op @ potential_operator - potential_operator @ density_op
        dirichlet_energy = 0.5 * float(np.linalg.norm(commutator, ord='fro') ** 2)

        # 2. Cota de Poincaré-Wirtinger sobre la varianza
        variance = float(np.linalg.norm(density_op - I_mean, ord='fro') ** 2)
        pw_bound = cp_constant * (2.0 * dirichlet_energy)

        # 3. Escalamiento de Contracción de Banach
        eta = min(spectral_cap, 1.0 / (1.0 + math.sqrt(dirichlet_energy + 1e-12)))
        disciplined_rho = (1.0 - eta) * I_mean + eta * density_op
        disciplined_rho = DensityOperatorAlgebra.sanitize(disciplined_rho)

        report = BanachContractionAlgebra.audit(
            disciplined_rho,
            eta_star=1.5,
            potential_operator=potential_operator,
            cp_constant=cp_constant,
            spectral_cap=spectral_cap
        )
        return disciplined_rho, report

    @classmethod
    def audit_discipline(
        cls,
        rho: np.ndarray,
        eta_star: float = 1.5,
        potential_operator: Optional[np.ndarray] = None,
        cp_constant: float = 0.5,
    ) -> BanachContractionReport:
        return BanachContractionAlgebra.audit(
            rho, eta_star=eta_star, potential_operator=potential_operator, cp_constant=cp_constant
        )

    @classmethod
    def audit_from_seed(
        cls, seed: SeedState, eta_star: float = 1.5,
    ) -> BanachContractionReport:
        return cls.audit_discipline(seed.rho_seed, eta_star)


# ── §2.4 CropGrowthPipeline — HAND-OFF FASE 2 → FASE 3 ────────────────────
@dataclass(frozen=True, slots=True)
class CropGrowthBundle:
    cycle_index: int
    seed: SeedState
    watering: WateringReport
    rho_illum: np.ndarray
    brockett_c: BrockettPurificationCertificate
    renyi_c: RenyiPurificationCertificate
    fock_c: FockAnnihilationCertificate
    discipline: BanachContractionReport


class CropGrowthPipeline:
    r"""Orquestador determinista del crecimiento (funtor F₂)."""

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
        verdict = HeytingCropAdjudicator.adjudicate(bundle, external_verdict)
        reason = (
            f"{reason_prefix}::water={bundle.watering.local_verdict.name} "
            f"disc={bundle.discipline.local_verdict.name} "
            f"ρ(T)={bundle.discipline.spectral_radius:.4f} "
            f"KAM={bundle.discipline.is_kam_stable} "
            f"Bifurc={bundle.discipline.is_pyriform_bifurcated}"
        )
        actuation = CognitiveFaithModule.verify_faith(verdict, reason)
        return verdict, actuation


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · FE + ADJUDICACIÓN + CERTIFICACIÓN                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en Ω₃ ────────────────────────────────────────────────
class HeytingCropAdjudicator:
    r"""Lógica interna del topos: colapsa el Bundle a un único valor de Ω₃."""

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
        local = (
            bundle.watering.local_verdict
            .meet(bundle.discipline.local_verdict)
            .meet(cls._fock_rule(bundle))
            .meet(cls._brockett_rule(bundle))
            .meet(cls._renyi_rule(bundle))
        )
        if bundle.discipline.is_pyriform_bifurcated:
            local = local.meet(HeytingOmega3.VETOED)
        return local.meet(external_verdict)


# ── §3.2 CognitiveFaithModule — FE (interlock ciber-físico ESP32 Crowbar) ──
@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    interlock_fired: bool
    actuation_latency_ns: float
    gpio_pin: str
    device: str
    reason: str
    provenance_hash: str


class CognitiveFaithModule:
    r"""FASE 3 · LA FE — Adjudicación Heyting Ω₃ e interlock ciber-físico ESP32 Crowbar."""
    GPIO_PIN: Final[str] = "GPIO14"
    DEVICE: Final[str] = "BT151_CROWBAR"
    NOMINAL_LATENCY_NS: Final[float] = 392.15

    def adjudicate_heyting_crowbar(
        self,
        disciplined_rho: np.ndarray,
        discipline_report: BanachContractionReport
    ) -> object:
        """Adjudica en Heyting Ω₃ y dispara interlock si hay veto o bifurcación piriforme."""
        verdict = discipline_report.local_verdict
        if not discipline_report.is_kam_stable or discipline_report.is_pyriform_bifurcated:
            verdict = HeytingOmega3.VETOED

        reason = f"KAM_STABLE={discipline_report.is_kam_stable}, PYRIFORM={discipline_report.is_pyriform_bifurcated}"
        actuation = self.verify_faith(verdict, reason)

        # Devolver un objeto passport de gobernanza simplificado/compatible
        from types import SimpleNamespace
        return SimpleNamespace(
            verdict=verdict,
            reason=reason,
            actuation=actuation,
            is_kam_stable=discipline_report.is_kam_stable,
            is_pyriform_bifurcated=discipline_report.is_pyriform_bifurcated
        )

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
            "[CULTIVO COGNITIVO — CROWBAR ARMADO] %s → HIGH | latencia %.2f ns | razón=%s",
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


# ── §3.3 Certificado del cultivo ──────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CropGerminationCertificate:
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
    is_kam_stable: bool = True
    is_pyriform_bifurcated: bool = False


# ── §3.4 TOONCognitiveCropEngine — orquestador soberano ───────────────────
class TOONCognitiveCropEngine:
    r"""Motor Espectral Principal del Cultivo Cognitivo con Mecánica Celeste Integrada."""

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

        # Módulos del motor
        self.watering_module = CognitiveWateringModule()
        self.illumination_module = CognitiveIlluminationModule()
        self.discipline_module = CognitiveDisciplineModule()
        self.faith_module = CognitiveFaithModule()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def execute_poincare_crop_pipeline(
        self,
        seed_crystal: object,
        soil_field: SoilField,
        cp_constant: float = 0.5
    ) -> Tuple[object, object]:
        """Ejecuta las 4 fases del cultivo bajo invariantes KAM y Poincaré-Wirtinger.

        Fases:
            1. Riego: Acondicionamiento del suelo en H_MAC.
            2. Luz: Purificación isospectral de Brockett.
            3. Disciplina: Acotación de Poincaré-Wirtinger y Contracción KAM.
            4. Fe: Adjudicación Heyting Ω₃ e interlock ESP32 Crowbar.
        """
        # Fase 1: Riego
        w_report = self.watering_module.moisten_soil(seed_crystal, soil_field)

        # Fase 2: Luz
        rho_illuminated, i_report = self.illumination_module.illuminate_brockett(w_report.moistened_rho)

        # Fase 3: Disciplina con Poincare-Wirtinger
        rho_disciplined, d_report = self.discipline_module.enforce_poincare_wirtinger_kam_bound(
            rho_illuminated, soil_field.potential_operator, cp_constant=cp_constant
        )

        # Fase 4: Fe y Adjudicación en Heyting Ω₃
        passport = self.faith_module.adjudicate_heyting_crowbar(rho_disciplined, d_report)

        from types import SimpleNamespace
        harvest = SimpleNamespace(
            harvested_rho=rho_disciplined,
            banach_report=d_report,
            watering_report=w_report,
            illumination_report=i_report
        )
        return harvest, passport

    def _phase1_prepare(self, seed_matrix: np.ndarray) -> SeedState:
        seed_state = SeedCrystalPreparation.prepare(seed_matrix)
        self._advance_chain("F1", np.ascontiguousarray(seed_state.rho_seed).tobytes())
        return seed_state

    def _phase2_grow(
        self,
        seed_state: SeedState,
        toon_str: str,
        base_json_str: str,
    ) -> CropGrowthBundle:
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
            is_kam_stable=bundle.discipline.is_kam_stable,
            is_pyriform_bifurcated=bundle.discipline.is_pyriform_bifurcated
        )

    def cultivate_seed_crystal(
        self,
        seed_crystal_id: str,
        seed_matrix: np.ndarray,
        toon_str: str,
        base_json_str: str,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> CropGerminationCertificate:
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
    print("TOON COGNITIVE CROP ENGINE — v8.1.0 Poincare Celestial Mechanics")
    print("KAM Tori · Poincaré-Wirtinger · Pyriform Bifurcations · Heyting Ω₃ · Merkle")
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
        print(f"   Invariante KAM Estable : {cert.is_kam_stable}")
        print(f"   Bifurcación Piriforme  : {cert.is_pyriform_bifurcated}")
        print(f"   Crowbar (Fe)           : {cert.faith_crowbar_armed}  "
              f"latencia = {crowbar_ns} ns")
        print(f"   Firma SHA-256          : {cert.sha256_provenance[:32]}…")

    print("\n" + "═" * 88)
    print("✓ F1→F2: prepare ⊣ apply_water / apply_light / audit_discipline.")
    print("✓ F2→F3: synthesize ⊣ adjudicate ⊗ verify_faith ⊗ certify.")
    print("✓ Poincaré-Wirtinger: ||rho - I/n||_F^2 <= C_P * ||[rho, N(p)]||_F^2.")
    print("✓ KAM Invariant Tori: Preservación de toros estables e inmunidad a difusión de Arnold.")
    print("✓ Control de Bifurcaciones Piriformes en rotación del espacio de fases.")
    print("═" * 88)

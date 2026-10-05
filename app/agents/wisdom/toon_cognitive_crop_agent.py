# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/agents/wisdom/toon_cognitive_crop_agent.py                            ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / CULTIVO COGNITIVO                   ║
║ FUNCIÓN  : SOBERANO DEL CULTIVO COGNITIVO CON MECÁNICA CELESTE DE POINCARÉ           ║
║ VERSIÓN  : 9.1.0-Poincare-Cartan-Melnikov-KAM-Bruno-CZ-Celestial-Crop-Agent-PhD      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICO-FÍSICA
───────────────────────────────────────────────────────
El `TOONCognitiveCropAgent` es el Soberano de Calibre que gobierna el campo de
cultivo cognitivo, sanea semillas, autoriza la inoculación CPTP en la MAC y
certifica la cosecha con Merkle + Crowbar BT151.

Composición anidada (funtores Φ_I ⊣ Φ_II ⊣ Φ_III):
  Φ_I   : SeedCrystalSanitizer.synthesize_poincare_cartan_germ
          → SeedCelestialHandoff.hand_off_germ_to_phase2
          ↦ 𝒢_I  = _PoincareCartanSeedGerm ⊗ SeedCelestialHandoff
          ≡ objeto inicial de Fase II.
  Φ_II  : CropGrowthPipeline.synthesize_celestial_bundle
          → CropGrowthPipeline.hand_off_bundle_to_phase3
          ↦ 𝒢_II = _PoincareCelestialGrowthBundle
          ≡ objeto inicial de Fase III.
  Φ_III : Ω₃⁷ ⊕ canal celeste + Crowbar + inoculación MAC + pasaporte Merkle
          ↦ CropHarvestYield ⊗ CropSovereignGovernancePassport.

FASE I  — Sustrato algebraico, saneamiento, geometría de la fase, MAC CPTP.
FASE II — Riego (Hill) + Luz (Brockett/Cayley) + Disciplina + KAM/Bruno + Melnikov + CZ.
FASE III — Fe Ω₃⁷⊕celeste, Crowbar BT151, inoculación MAC, gobernanza Merkle.

POSTULADOS DE GOBERNANZA
────────────────────────
1. Toros KAM: |⟨k,ω⟩| ≥ γ/|k|^τ  y  Bruno ∑ 2^{-ν} log(1/Ω(2^ν)) < ∞.
2. Poincaré-Wirtinger: ‖ρ − I/n‖_F² ≤ C_P · 2 E_D(ρ).
3. Contracción MAC: Lip(Φ_γ) = |1−γ| < 1 (CPTP convexo).
4. Melnikov: cero simple ⇒ fractura homoclínica (veto).
5. Conley–Zehnder + Nekhoroshev T ≥ exp(c ε^{-1/(2n)}).
6. Inoculación MAC sólo si Ω₃ = ⊤; crowbar si Ω₃ = ⊥.
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
from typing import Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from app.wisdom.toon_cognitive_crop_engine import (  # type: ignore[import]
        TOONCognitiveCropEngine,
        SoilField,
        SoilState,
    )
except ImportError:  # pragma: no cover
    try:
        from toon_cognitive_crop_engine import (  # type: ignore[no-redef]
            TOONCognitiveCropEngine,
            SoilField,
            SoilState,
        )
    except ImportError:  # pragma: no cover — motor ausente: stubs mínimos
        TOONCognitiveCropEngine = None  # type: ignore[misc, assignment]

        class SoilField:  # type: ignore[no-redef]
            def __init__(self, n: int = 4) -> None:
                self.n = n
                self.potential_operator = np.diag(
                    np.arange(1, n + 1, dtype=np.float64)
                ).astype(np.complex128)

        @dataclass(frozen=True)
        class SoilState:  # type: ignore[no-redef]
            H_mac: np.ndarray
            ground_proj: np.ndarray
            E0: float
            gap_H: float
            hash: str

            @property
            def soil_field(self) -> "SoilField":
                return SoilField(self.H_mac.shape[0] if hasattr(self.H_mac, "shape") else 4)

logger = logging.getLogger("APU.Wisdom.TOONCognitiveCropAgent")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

__version__: Final[str] = (
    "9.1.0-Poincare-Cartan-Melnikov-KAM-Bruno-CZ-Celestial-Crop-Agent-PhD"
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════
_EPS: Final[float] = 1.0e-14
_EPS_MOD: Final[float] = 1.0e-12
_EPS_TRACE: Final[float] = 1.0e-15
_EPS_CONTRACT: Final[float] = 1.0e-3
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DRIFT_LIMIT: Final[float] = 1.0e-9
_HILL_MARGIN_FLOOR: Final[float] = 1.0e-12
_KAM_TAU_FLOOR: Final[float] = 1.0
_KAM_GAMMA_FLOOR: Final[float] = 1.0e-12
_KAM_THRESHOLD: Final[float] = 1.0e-9
_MELNIKOV_THRESHOLD: Final[float] = 1.0e-6
_MELNIKOV_T_INF: Final[float] = 25.0
_MELNIKOV_QUAD_NODES: Final[int] = 513
_FLOQUET_PARABOLIC: Final[float] = 1.0e-6
_LYAPUNOV_CLIP: Final[float] = 700.0
_CSMD_STEP: Final[float] = 1.0e-8
_LOG_EXP_CLIP: Final[float] = 700.0
_DEGRADATION_FACTOR: Final[float] = 1.0e-2
_BRUNO_MAX_OCTAVES: Final[int] = 12
_RICHARDSON_STEPS: Final[int] = 3
_GELFAND_POWERS: Final[int] = 8
_NEKHOROSHEV_C: Final[float] = 0.25
_SYMPLECTIC_SKEW_TOL: Final[float] = 1.0e-12
_BIRKHOFF_TWIST_FLOOR: Final[float] = 1.0e-9
_MAC_TRACE_TOL: Final[float] = 1.0e-9
_MAC_PSD_TOL: Final[float] = 1.0e-8

ComplexMatrix = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]


def _seed_from_string(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def _sha256_bytes(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


def _kahan_neumaier_sum(values: Sequence[float]) -> float:
    acc = 0.0
    comp = 0.0
    for x in values:
        t = acc + x
        if abs(acc) >= abs(x):
            comp += (acc - t) + x
        else:
            comp += (x - t) + acc
        acc = t
    return acc + comp


def _clip_log_exp(x: float) -> float:
    return float(np.clip(x, -_LOG_EXP_CLIP, _LOG_EXP_CLIP))


def _hermitian_part(a: np.ndarray) -> ComplexMatrix:
    z = np.asarray(a, dtype=np.complex128)
    return 0.5 * (z + z.conj().T)


def _merkle_root(hashes: List[str]) -> str:
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ALGEBRAICO + SANEAMIENTO + GEOMETRÍA + MAC CPTP         ║
# ║                                                                           ║
# ║ Objetos: Ω₃, 𝔇_n, B(u(n))×CPTP, SeedCrystal, (T*Q, Ω, g̃, λ_PC), MAC.     ║
# ║ Morfismo terminal (I.ω) : SeedCrystalSanitizer.synthesize_poincare_       ║
# ║                           cartan_germ ↦ 𝒢_I.                              ║
# ║ Morfismo de empalme (I.ω⁺): SeedCelestialHandoff.hand_off_germ_to_phase2  ║
# ║     ≡ objeto inicial / continuación de Fase II.                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── §1.1 Retículo de Heyting Ω₃ ───────────────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa (álgebra de Gödel G₃)
        Ω₃ = {⊥ ≺ ⋆ ≺ ⊤} ≅ {0, 1, 2}.
    Locale finito: ∧ distribuye sobre ∨. Regulares = {⊥, ⊤}.
    """
    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    @property
    def verdict(self) -> str:
        return self.name

    @property
    def godel_value(self) -> float:
        return float(int(self)) / 2.0

    def leq(self, other: "HeytingOmega3") -> bool:
        return int(self) <= int(other)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.COHERENT if self.leq(other) else other

    def neg(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.VETOED)

    def iff(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.implies(other).meet(other.implies(self))

    def is_regular(self) -> bool:
        return self.neg().neg() == self

    def as_weight(self) -> float:
        return self.godel_value

    @classmethod
    def from_godel(cls, value: float) -> "HeytingOmega3":
        v = float(value)
        if v >= 0.75:
            return cls.COHERENT
        if v >= 0.25:
            return cls.DEGRADED
        return cls.VETOED

    @classmethod
    def infimum(cls, *values: "HeytingOmega3") -> "HeytingOmega3":
        acc = cls.COHERENT
        for v in values:
            acc = acc.meet(v)
        return acc

    @classmethod
    def supremum(cls, *values: "HeytingOmega3") -> "HeytingOmega3":
        acc = cls.VETOED
        for v in values:
            acc = acc.join(v)
        return acc

    @classmethod
    def from_name_safe(cls, name: Optional[str]) -> "HeytingOmega3":
        if not name:
            return cls.VETOED
        try:
            return cls[name]
        except KeyError:
            return cls.VETOED


# ── §1.2 Álgebra de operadores densidad ───────────────────────────────────
class DensityOperatorAlgebra:
    r"""Operaciones canónicas sobre 𝔇_n = {ρ = ρ†, ρ ≥ 0, Tr ρ = 1}."""
    EPS: Final[float] = _EPS
    EPS_MOD: Final[float] = _EPS_MOD
    EPS_MODULAR: Final[float] = _EPS_MOD

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        if not cls.is_square(rho):
            raise ValueError(
                f"DensityOperatorAlgebra.sanitize: no cuadrada {np.shape(rho)}"
            )
        rho_h = _hermitian_part(rho)
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
        if abs(alpha - 1.0) < 1e-12:
            return cls.von_neumann_entropy(rho)
        p = cls.spectrum_descending(rho)
        tr_a = max(float(np.sum(np.power(p, alpha))), cls.EPS)
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
        sig = np.maximum(np.real(la.svdvals(cls.sanitize(rho))), 0.0)
        if p == math.inf:
            return float(sig.max()) if sig.size else 0.0
        if p <= 0.0:
            raise ValueError("Schatten p-norm requiere p > 0")
        return float(np.power(np.sum(np.power(sig, p)), 1.0 / p))

    @classmethod
    def trace_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        delta = cls.sanitize(rho) - cls.sanitize(sigma)
        return 0.5 * cls.schatten_p_norm(delta, 1.0)

    @classmethod
    def umegaki_relative_entropy(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        w_r, V_r = la.eigh(cls.sanitize(rho))
        w_s, V_s = la.eigh(cls.sanitize(sigma))
        w_r_reg = np.maximum(np.real(w_r), cls.EPS_MOD)
        w_s_reg = np.maximum(np.real(w_s), cls.EPS_MOD)
        log_r = V_r @ np.diag(np.log(w_r_reg).astype(np.complex128)) @ V_r.conj().T
        log_s = V_s @ np.diag(np.log(w_s_reg).astype(np.complex128)) @ V_s.conj().T
        val = float(np.real(np.trace(cls.sanitize(rho) @ (log_r - log_s))))
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
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        fid = cls.uhlmann_fidelity(rho, sigma)
        return float(np.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid))))

    @classmethod
    def matrix_power(
        cls, rho: np.ndarray, z: complex, floor: float = _EPS_MOD
    ) -> ComplexMatrix:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        powered = np.exp(complex(z) * np.log(w.astype(np.complex128)))
        return (V * powered) @ V.conj().T

    @classmethod
    def modular_hamiltonian(cls, rho: np.ndarray) -> ComplexMatrix:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), cls.EPS_MOD)
        return (V * (-np.log(w)).astype(np.complex128)) @ V.conj().T

    @classmethod
    def modular_spectrum(cls, rho: np.ndarray) -> Tuple[float, ...]:
        w = cls.spectrum_descending(rho)
        k = -np.log(np.maximum(w, cls.EPS_MOD))
        return tuple(sorted(float(x) for x in k.tolist()))

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> ComplexMatrix:
        H_h = _hermitian_part(H)
        w, V = la.eigh(H_h)
        i0 = int(np.argmin(np.real(w)))
        psi = V[:, i0].reshape(-1, 1)
        return cls.sanitize(psi @ psi.conj().T)

    @classmethod
    def renyi_sharpen(cls, rho: np.ndarray, alpha: float) -> ComplexMatrix:
        if alpha <= 1.0 + 1e-12:
            return cls.sanitize(rho)
        rho = cls.sanitize(rho)
        rho_a = cls.matrix_power(rho, complex(alpha, 0.0))
        tr = float(np.trace(rho_a).real)
        if tr < 1e-30:
            return rho
        return cls.sanitize(rho_a / tr)


# ── §1.3 Álgebra de Banach: semilla ⊕ MAC, Poincaré-Wirtinger, Gelfand ────
@dataclass(frozen=True, slots=True)
class BanachContractionReport:
    r"""
    Auditoría conjunta semilla ⊕ MAC en B(u(n)) × CPTP(𝔇_n).
    ρ(T_η) por cota ∞ y Gelfand; Lip(Φ_γ) = |1−γ|; C_P Poincaré-Wirtinger.
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
    dirichlet_energy: float = 0.0
    poincare_wirtinger_bound: float = 0.0
    variance: float = 0.0
    is_kam_stable: bool = True
    is_pyriform_bifurcated: bool = False
    banach_factor: float = 0.0
    gelfand_radius: float = 0.0
    neumann_remainder: float = 0.0


class BanachContractionAlgebra:
    r"""ρ(T_η) sobre u(n), Lip(Φ_γ) sobre 𝔇_n, Poincaré-Wirtinger y Gelfand."""
    EPS: Final[float] = _EPS_MOD

    @classmethod
    def coupling_matrix(cls, rho: np.ndarray) -> RealMatrix:
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
    def seed_spectral_radius(
        cls, rho: np.ndarray, eta_star: float
    ) -> Tuple[float, float, float]:
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
    def gelfand_spectral_radius(
        cls, rho: np.ndarray, eta: float, powers: int = _GELFAND_POWERS
    ) -> float:
        G = cls.coupling_matrix(rho)
        if G.size == 0:
            return 1.0
        T = np.eye(G.shape[0], dtype=np.float64) - eta * G
        np.fill_diagonal(T, 0.0)
        Tk = T.copy()
        radius = float(np.linalg.norm(Tk, 2))
        for k in range(1, max(1, powers) + 1):
            Tk = Tk @ T
            nk = float(np.linalg.norm(Tk, 2))
            if nk <= 0.0:
                return 0.0
            radius = nk ** (1.0 / (k + 1))
        return float(radius)

    @classmethod
    def neumann_series_remainder(cls, rho_T: float, order: int = 8) -> float:
        if rho_T >= 1.0:
            return float("inf")
        return float((rho_T ** order) / max(1.0 - rho_T, _MACHINE_EPS))

    @classmethod
    def mac_lipschitz(cls, gamma: float) -> float:
        return abs(1.0 - float(np.clip(gamma, 0.0, 1.0)))

    @classmethod
    def audit(
        cls,
        rho_seed: np.ndarray,
        eta_star: float,
        mac_gamma: float,
        potential_operator: Optional[np.ndarray] = None,
        cp_constant: float = 0.5,
    ) -> BanachContractionReport:
        n = rho_seed.shape[0]
        I_mean = np.eye(n, dtype=np.complex128) / n
        if potential_operator is None:
            potential_operator = np.diag(
                np.arange(1, n + 1, dtype=np.float64)
            ).astype(np.complex128)
        commutator = rho_seed @ potential_operator - potential_operator @ rho_seed
        dirichlet_energy = 0.5 * float(np.linalg.norm(commutator, ord="fro") ** 2)
        variance = float(np.linalg.norm(rho_seed - I_mean, ord="fro") ** 2)
        pw_bound = cp_constant * (2.0 * dirichlet_energy)
        rho_T, eta_max, g_max = cls.seed_spectral_radius(rho_seed, eta_star)
        gelfand = cls.gelfand_spectral_radius(rho_seed, eta_star)
        rho_eff = max(rho_T, gelfand)
        _, i_star, j_star = cls.pair_couplings(rho_seed)
        mac_coef = cls.mac_lipschitz(mac_gamma)
        lip = float(max(rho_eff, mac_coef))
        eta_factor = min(0.95, 1.0 / (1.0 + math.sqrt(dirichlet_energy + 1e-12)))
        remainder = cls.neumann_series_remainder(rho_eff)
        is_kam_stable = (rho_eff < 1.0) and (
            variance <= pw_bound + 1e-6 or dirichlet_energy < 1e-12
        )
        is_pyriform_bifurcated = (
            (not is_kam_stable)
            or (rho_eff >= 1.0)
            or (variance > pw_bound * 2.0 + 1e-3)
        )
        p_seed = rho_eff < 1.0
        p_mac = mac_coef < 1.0
        if p_seed and p_mac and is_kam_stable:
            local = (
                HeytingOmega3.COHERENT
                if rho_eff < 1.0 - _EPS_CONTRACT
                else HeytingOmega3.DEGRADED
            )
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
            dirichlet_energy=dirichlet_energy,
            poincare_wirtinger_bound=pw_bound,
            variance=variance,
            is_kam_stable=is_kam_stable,
            is_pyriform_bifurcated=is_pyriform_bifurcated,
            banach_factor=eta_factor,
            gelfand_radius=gelfand,
            neumann_remainder=remainder,
        )


# ── §1.4 Geometría de la fase: Maupertuis-Jacobi, Hill, Poincaré-Cartan ────
@dataclass(frozen=True, slots=True)
class MaupertuisJacobiGerm:
    r"""g̃_{jk}(q) = 2(H₀ − V(q)) g_{jk}(q) = n(q)² g_{jk}(q)."""
    conformal_factor: float
    refractive_index: float
    hill_margin: float
    is_in_hill_region: bool
    min_eigenvalue: float
    is_positive_definite: bool
    condition_number: float = 1.0


@dataclass(frozen=True, slots=True)
class PoincareCartanGerm:
    r"""λ = p_i dq^i − H dt ∈ Ω¹(T*Q × ℝ); dλ = ω − dH ∧ dt."""
    lambda_vector: np.ndarray
    hamiltonian_value: float
    dim: int
    two_n: int
    symplectic_skew_residual: float
    contact_dim: int = 0
    liouville_volume: float = 1.0


class PoincareCelestialGeometry:
    r"""Fase I (bloque geométrico). Álgebra de mecánica celeste de Poincaré."""

    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"Dimensión simpléctica dim={dim} debe ser par y positiva.")
        half = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:half, half:] = np.eye(half, dtype=np.float64)
        omega[half:, :half] = -np.eye(half, dtype=np.float64)
        return omega

    @classmethod
    def symplectic_form_audit(cls, omega: np.ndarray) -> Tuple[float, float, bool]:
        om = np.asarray(omega, dtype=np.float64)
        if om.ndim != 2 or om.shape[0] != om.shape[1]:
            return float("inf"), float("inf"), False
        skew = float(np.linalg.norm(om + om.T, "fro"))
        try:
            det = float(np.real(la.det(om)))
        except la.LinAlgError:
            det = float("nan")
        darboux = float(np.linalg.norm(om @ om + np.eye(om.shape[0]), "fro"))
        ok = (
            skew <= _SYMPLECTIC_SKEW_TOL
            and abs(abs(det) - 1.0) <= 1e-8
            and darboux <= 1e-8
        )
        return skew, darboux, ok

    @classmethod
    def compute_maupertuis_conformal_metric(
        cls,
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
        base_metric_g: np.ndarray,
    ) -> Tuple[np.ndarray, MaupertuisJacobiGerm]:
        g = np.asarray(base_metric_g, dtype=np.float64)
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError("base_metric_g debe ser cuadrada.")
        h0 = float(hamiltonian_energy_H0)
        v = float(potential_energy_V)
        if not (np.isfinite(h0) and np.isfinite(v)):
            raise ValueError("H₀ y V deben ser finitos.")
        free_energy = 2.0 * (h0 - v)
        in_hill = bool(free_energy > _HILL_MARGIN_FLOOR)
        phi = max(free_energy, _HILL_MARGIN_FLOOR)
        refractive = float(np.sqrt(phi))
        gt = phi * g
        try:
            evals = la.eigvalsh(0.5 * (gt + gt.T))
            real_ev = np.real(evals) if evals.size else np.array([0.0])
            min_eig = float(np.min(real_ev))
            max_eig = float(np.max(real_ev))
            cond = float(max_eig / max(min_eig, _HILL_MARGIN_FLOOR))
        except la.LinAlgError:
            min_eig = 0.0
            cond = float("inf")
        germ = MaupertuisJacobiGerm(
            conformal_factor=float(phi),
            refractive_index=refractive,
            hill_margin=float(h0 - v),
            is_in_hill_region=in_hill,
            min_eigenvalue=min_eig,
            is_positive_definite=bool(min_eig > _HILL_MARGIN_FLOOR),
            condition_number=cond,
        )
        return gt, germ

    @staticmethod
    def compute_hill_region_margin(potential_V: float, total_energy_H0: float) -> float:
        return float(total_energy_H0 - potential_V)

    @classmethod
    def compute_christoffel_conformal_symbols(
        cls,
        grad_V: np.ndarray,
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: np.ndarray,
    ) -> np.ndarray:
        r"""Γ̃^i_{jk} vectorizado: δ^i_j ∂_k φ + δ^i_k ∂_j φ − g_{jk} g^{il} ∂_l φ."""
        grad_V = np.asarray(grad_V, dtype=np.float64).ravel()
        g = np.asarray(g_base_metric, dtype=np.float64)
        n_dim = grad_V.size
        if g.shape != (n_dim, n_dim):
            raise ValueError(f"g_base_metric debe ser {n_dim}×{n_dim}.")
        headroom = 2.0 * (total_energy_H0 - potential_V)
        if headroom <= _HILL_MARGIN_FLOOR:
            raise ValueError("[CROP_AGENT_VETO] Cero energía cinética: invasión de pozo.")
        grad_phi = -grad_V / (headroom + _HILL_MARGIN_FLOOR)
        g_inv = la.inv(g)
        term1 = np.eye(n_dim)[:, :, None] * grad_phi[None, None, :]
        term2 = np.eye(n_dim)[:, None, :] * grad_phi[None, :, None]
        ginv_dphi = g_inv @ grad_phi
        term3 = g[None, :, :] * ginv_dphi[:, None, None]
        return term1 + term2 - term3

    @classmethod
    def compute_poincare_cartan_lambda(
        cls,
        x: np.ndarray,
        hamiltonian_value: float = 0.0,
        omega: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, PoincareCartanGerm]:
        xv = np.asarray(x, dtype=np.float64).ravel()
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("λ de Poincaré-Cartan exige dim par (T*Q).")
        n = dim // 2
        p = xv[n:]
        h_val = float(hamiltonian_value) if np.isfinite(hamiltonian_value) else 0.0
        lam = np.concatenate(
            [p, np.zeros(n, dtype=np.float64), np.array([-h_val], dtype=np.float64)]
        )
        if omega is None:
            omega = cls.generate_canonical_symplectic_form(dim)
        omega_arr = np.asarray(omega, dtype=np.float64)
        skew_res, _, _ = cls.symplectic_form_audit(omega_arr)
        try:
            vol = float(abs(la.det(omega_arr)) ** 0.5) if omega_arr.size else 1.0
        except la.LinAlgError:
            vol = 0.0
        germ = PoincareCartanGerm(
            lambda_vector=lam,
            hamiltonian_value=h_val,
            dim=dim,
            two_n=dim,
            symplectic_skew_residual=skew_res,
            contact_dim=2 * n + 1,
            liouville_volume=vol,
        )
        return lam, germ

    @staticmethod
    def compute_gradient_csmd(
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        xv = np.asarray(x, dtype=np.float64).ravel()
        if not np.isfinite(h) or h == 0.0:
            raise ValueError("El paso CSMD h debe ser finito y no nulo.")
        h = float(abs(h))
        dim = xv.size
        grad = np.zeros(dim, dtype=np.float64)
        holomorphic = True
        try:
            probe = func(xv.astype(np.complex128))
            holomorphic = np.isfinite(np.real(probe)) or np.isfinite(np.imag(probe))
        except (TypeError, ValueError, FloatingPointError):
            holomorphic = False
        if holomorphic:
            for i in range(dim):
                xp = xv.astype(np.complex128)
                xp[i] += 1j * h
                try:
                    val = func(xp)
                    imag = float(np.imag(val))
                except Exception:
                    holomorphic = False
                    break
                if not np.isfinite(imag):
                    holomorphic = False
                    break
                grad[i] = imag / h
        if not holomorphic:
            for i in range(dim):
                xp = xv.copy()
                xm = xv.copy()
                xp[i] += h
                xm[i] -= h
                try:
                    fp = float(np.real(func(xp)))
                    fm = float(np.real(func(xm)))
                except Exception:
                    fp = fm = 0.0
                grad[i] = (fp - fm) / (2.0 * h)
        return grad

    @classmethod
    def richardson_csmd_gradient(
        cls,
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
        steps: int = _RICHARDSON_STEPS,
    ) -> np.ndarray:
        table: List[np.ndarray] = []
        hh = float(abs(h))
        for s in range(max(1, steps)):
            table.append(cls.compute_gradient_csmd(func, x, hh / (2 ** s)))
        for k in range(1, len(table)):
            four_k = 4.0 ** k
            for s in range(len(table) - k):
                table[s] = (four_k * table[s + 1] - table[s]) / (four_k - 1.0)
        return table[0]

    @classmethod
    def poisson_bracket(
        cls,
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        x: np.ndarray,
        omega: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> float:
        xv = np.asarray(x, dtype=np.float64).ravel()
        if xv.size % 2 != 0:
            raise ValueError("El corchete de Poisson exige dim par (Darboux).")
        grad0 = cls.compute_gradient_csmd(hamiltonian_0, xv, h)
        grad1 = cls.compute_gradient_csmd(hamiltonian_1, xv, h)
        return float(grad0 @ omega @ grad1)

    @classmethod
    def cartan_magic_identity_residual(
        cls,
        hamiltonian: Callable[[np.ndarray], float],
        x: np.ndarray,
        omega: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> float:
        xv = np.asarray(x, dtype=np.float64).ravel()
        grad = cls.compute_gradient_csmd(hamiltonian, xv, h)
        residual = omega @ (omega @ grad) + grad
        return float(np.linalg.norm(residual, 2))

    @staticmethod
    def stormer_verlet_step(
        q: np.ndarray,
        p: np.ndarray,
        grad_v: Callable[[np.ndarray], np.ndarray],
        mass_inv: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        qn = np.asarray(q, dtype=np.float64)
        pn = np.asarray(p, dtype=np.float64)
        p_half = pn - 0.5 * dt * np.asarray(grad_v(qn), dtype=np.float64)
        q_next = qn + dt * mass_inv * p_half
        p_next = p_half - 0.5 * dt * np.asarray(grad_v(q_next), dtype=np.float64)
        return q_next, p_next

    @classmethod
    def gromov_capacity_proxy(cls, hill_margin: float) -> float:
        return float(math.pi * max(float(hill_margin), 0.0))


# ── §1.5 SeedCrystal + sanitizador + gérmen de Poincaré-Cartan ────────────
@dataclass(frozen=True, slots=True)
class SeedCrystal:
    crystal_id: str
    origin_agent: str
    density_matrix: np.ndarray
    experience_vector: np.ndarray
    is_vacuum_pure: bool


@dataclass(frozen=True, slots=True)
class SeedAuditReport:
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
    bures_to_ground: float = 0.0


@dataclass(frozen=True, slots=True)
class _PoincareCartanSeedGerm:
    r"""
    GÉRMEN DE POINCARÉ-CARTAN (terminal de Fase I, inicial de Fase II).
    Compone (ρ_seed, seed_audit, ground, mac) ⊗ (g̃, λ_PC, Ω, Γ̃, c_G).
    """
    seed_id: str
    origin_agent: str
    rho_seed: np.ndarray
    seed_audit: SeedAuditReport
    ground_projector: np.ndarray
    mac_snapshot: np.ndarray
    mac_gamma: float
    K_spec_seed: Tuple[float, ...]
    maupertuis_germ: MaupertuisJacobiGerm
    cartan_germ: PoincareCartanGerm
    omega: np.ndarray
    base_metric_g: np.ndarray
    conformal_metric_gt: np.ndarray
    christoffel_conformal: np.ndarray
    symplectic_capacity: float
    hamiltonian_energy_H0: float
    potential_V: float
    reg_floor: float
    seed_crystal: Optional[SeedCrystal] = None


class SeedCrystalSanitizer:
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
        cls, seed: SeedCrystal, rho_ground: np.ndarray
    ) -> Tuple[ComplexMatrix, SeedAuditReport]:
        rho = DensityOperatorAlgebra.sanitize(seed.density_matrix)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        P = float(np.sum(w ** 2))
        S = float(-np.sum(w * np.log(np.maximum(w, _EPS))))
        F = DensityOperatorAlgebra.uhlmann_fidelity(rho, rho_ground)
        bures = DensityOperatorAlgebra.bures_distance(rho, rho_ground)
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
            residual = float(
                np.linalg.norm(ev_k / (n_ev + 1e-30) - sig_k / (n_sg + 1e-30))
                / math.sqrt(2.0)
            )
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
            bures_to_ground=bures,
        )
        return rho, audit

    # ── I.ω  MORFISMO TERMINAL DE LA FASE I ──────────────────────────────
    @classmethod
    def synthesize_poincare_cartan_germ(
        cls,
        seed: SeedCrystal,
        rho_ground: np.ndarray,
        mac_snapshot: np.ndarray,
        mac_gamma: float,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        base_metric_g: Optional[np.ndarray] = None,
        grad_V: Optional[np.ndarray] = None,
    ) -> _PoincareCartanSeedGerm:
        r"""
        MORFISMO TERMINAL DE LA FASE I ≅ OBJETO INICIAL DE LA FASE II.
        Compone SeedCrystal saneado ⊗ (Maupertuis ⊕ Hill ⊕ λ_PC ⊕ Ω ⊕ Γ̃ ⊕ c_G).
        """
        rho_s, audit = cls.sanitize(seed, rho_ground)
        K_spec = DensityOperatorAlgebra.modular_spectrum(rho_s)
        n = int(rho_s.shape[0])
        two_n = 2 * n
        omega = PoincareCelestialGeometry.generate_canonical_symplectic_form(two_n)
        if base_metric_g is None:
            base_metric_g = np.eye(n, dtype=np.float64)
        gt, mau_germ = PoincareCelestialGeometry.compute_maupertuis_conformal_metric(
            hamiltonian_energy_H0, potential_energy_V, base_metric_g
        )
        if grad_V is None:
            grad_V = np.zeros(n, dtype=np.float64)
        try:
            christoffel = PoincareCelestialGeometry.compute_christoffel_conformal_symbols(
                grad_V=grad_V,
                potential_V=potential_energy_V,
                total_energy_H0=hamiltonian_energy_H0,
                g_base_metric=np.asarray(base_metric_g, dtype=np.float64),
            )
        except ValueError:
            christoffel = np.zeros((n, n, n), dtype=np.float64)
        x0 = np.zeros(two_n, dtype=np.float64)
        _lam, cartan_germ = PoincareCelestialGeometry.compute_poincare_cartan_lambda(
            x0, hamiltonian_value=potential_energy_V, omega=omega
        )
        capacity = PoincareCelestialGeometry.gromov_capacity_proxy(mau_germ.hill_margin)
        reg_floor = max(
            float(np.linalg.norm(omega, "fro")) * _MACHINE_EPS * 10.0,
            _HILL_MARGIN_FLOOR,
        )
        return _PoincareCartanSeedGerm(
            seed_id=seed.crystal_id,
            origin_agent=seed.origin_agent,
            rho_seed=rho_s,
            seed_audit=audit,
            ground_projector=DensityOperatorAlgebra.sanitize(rho_ground),
            mac_snapshot=np.asarray(mac_snapshot).copy(),
            mac_gamma=float(np.clip(mac_gamma, 0.0, 1.0)),
            K_spec_seed=K_spec,
            maupertuis_germ=mau_germ,
            cartan_germ=cartan_germ,
            omega=omega,
            base_metric_g=np.asarray(base_metric_g, dtype=np.float64),
            conformal_metric_gt=gt,
            christoffel_conformal=christoffel,
            symplectic_capacity=float(capacity),
            hamiltonian_energy_H0=float(hamiltonian_energy_H0),
            potential_V=float(potential_energy_V),
            reg_floor=float(reg_floor),
            seed_crystal=seed,
        )


# ── §1.6 Estado MAC + operador de actualización CPTP auditado ─────────────
@dataclass(frozen=True, slots=True)
class MacUpdateCertificate:
    r"""Certificado CPTP de Φ_γ(ρ) = (1−γ) ρ_MAC + γ ρ_target. Lip = |1−γ|."""
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
    bures_to_target: float = 0.0
    is_cptp: bool = True


class MacStateField:
    def __init__(self, rho_init: np.ndarray, gamma: float = 0.2) -> None:
        self.rho: ComplexMatrix = DensityOperatorAlgebra.sanitize(rho_init)
        self.gamma: float = float(np.clip(gamma, 0.0, 1.0))

    def _apply(self, rho_target: np.ndarray) -> ComplexMatrix:
        rho_t = DensityOperatorAlgebra.sanitize(rho_target)
        return (1.0 - self.gamma) * self.rho + self.gamma * rho_t

    def _certify(
        self, rho_new: np.ndarray, rho_target: np.ndarray, mutated: bool
    ) -> MacUpdateCertificate:
        tr_res = abs(float(np.trace(rho_new).real) - 1.0)
        tr_ok = tr_res < _MAC_TRACE_TOL
        w_new = np.real(la.eigvalsh(_hermitian_part(rho_new)))
        min_ev = float(w_new.min()) if w_new.size else 0.0
        psd_ok = bool(min_ev > -_MAC_PSD_TOL)
        coef = abs(1.0 - self.gamma)
        fid = DensityOperatorAlgebra.uhlmann_fidelity(rho_new, rho_target)
        bures = DensityOperatorAlgebra.bures_distance(rho_new, rho_target)
        cptp = bool(tr_ok and psd_ok)
        if coef < 1.0 and cptp:
            verdict = HeytingOmega3.COHERENT
        elif abs(coef - 1.0) <= 1e-15 and cptp:
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
            bures_to_target=bures,
            is_cptp=cptp,
        )

    def peek(self, rho_target: np.ndarray) -> MacUpdateCertificate:
        return self._certify(self._apply(rho_target), rho_target, mutated=False)

    def update(self, rho_target: np.ndarray) -> MacUpdateCertificate:
        rho_new = self._apply(rho_target)
        cert = self._certify(rho_new, rho_target, mutated=True)
        self.rho = DensityOperatorAlgebra.sanitize(rho_new)
        return cert

    def idle_certificate(self) -> MacUpdateCertificate:
        return self._certify(self.rho, self.rho, mutated=False)


# ── §1.7 SeedHandoff (clásico, retrocompatible) ────────────────────────────
@dataclass(frozen=True, slots=True)
class SeedHandoff:
    seed_id: str
    origin_agent: str
    rho_seed: np.ndarray
    seed_audit: SeedAuditReport
    K_spec_seed: Tuple[float, ...]
    ground_projector: np.ndarray
    mac_snapshot: np.ndarray
    mac_gamma: float
    spectral_hash: str
    seed_crystal: Optional[SeedCrystal] = None

    @classmethod
    def build(
        cls,
        seed: SeedCrystal,
        mac_field: MacStateField,
        ground_projector: np.ndarray,
    ) -> "SeedHandoff":
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
            seed_crystal=seed,
        )

    def continue_into_phase2(
        self,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float,
        eta_star: float,
    ) -> "CropGrowthBundle":
        return CropGrowthPipeline.synthesize(
            handoff=self,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=renyi_alpha,
            eta_star=eta_star,
        )


# ── §1.8 SeedCelestialHandoff — HAND-OFF FASE 1 → FASE 2 ──────────────────
@dataclass(frozen=True, slots=True)
class SeedCelestialHandoff:
    r"""
    HANDOFF CELESTE (objeto terminal de Fase I, inicial de Fase II).
    Extiende SeedHandoff con el gérmen 𝒢_I = _PoincareCartanSeedGerm.
    """
    base_handoff: SeedHandoff
    celestial_germ: _PoincareCartanSeedGerm

    @classmethod
    def build(
        cls,
        seed: SeedCrystal,
        mac_field: MacStateField,
        ground_projector: np.ndarray,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        base_metric_g: Optional[np.ndarray] = None,
    ) -> "SeedCelestialHandoff":
        base = SeedHandoff.build(seed, mac_field, ground_projector)
        germ = SeedCrystalSanitizer.synthesize_poincare_cartan_germ(
            seed=seed,
            rho_ground=ground_projector,
            mac_snapshot=mac_field.rho,
            mac_gamma=mac_field.gamma,
            hamiltonian_energy_H0=hamiltonian_energy_H0,
            potential_energy_V=potential_energy_V,
            base_metric_g=base_metric_g,
        )
        return cls(base_handoff=base, celestial_germ=germ)

    def continue_into_phase2_celestial(
        self,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float,
        eta_star: float,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        period_T: float = 1.0,
        novikov_valuation_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> "_PoincareCelestialGrowthBundle":
        return CropGrowthPipeline.synthesize_celestial_bundle(
            celestial_handoff=self,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=renyi_alpha,
            eta_star=eta_star,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            period_T=period_T,
            novikov_valuation_T=novikov_valuation_T,
            safety_margin=safety_margin,
        )

    # ── I.ω⁺  EMPALME FORMAL FASE I → FASE II ────────────────────────────
    def hand_off_germ_to_phase2(
        self,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float = 1.5,
        eta_star: float = 1.5,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        period_T: float = 1.0,
        novikov_valuation_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> "_PoincareCelestialGrowthBundle":
        r"""
        ═══════════════════════════════════════════════════════════════════════
        ÚLTIMO MÉTODO DE LA FASE I ≡ PRIMER MORFISMO OPERATIVO DE LA FASE II.
        ═══════════════════════════════════════════════════════════════════════
        Consume 𝒢_I = SeedCelestialHandoff (gérmen Poincaré-Cartan + MAC) y
        delega en CropGrowthPipeline.synthesize_celestial_bundle
        (Riego Hill + Luz + Disciplina + KAM/Bruno + Melnikov + CZ).
        """
        return self.continue_into_phase2_celestial(
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=renyi_alpha,
            eta_star=eta_star,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            period_T=period_T,
            novikov_valuation_T=novikov_valuation_T,
            safety_margin=safety_margin,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · RIEGO + LUZ + DISCIPLINA + KAM/BRUNO + MELNIKOV + CZ + RETORNO   ║
# ║                                                                           ║
# ║ Continuación directa de SeedCelestialHandoff.hand_off_germ_to_phase2.     ║
# ║ Morfismo terminal (II.ω): CropGrowthPipeline.synthesize_celestial_bundle  ║
# ║     ↦ 𝒢_II = _PoincareCelestialGrowthBundle.                              ║
# ║ Morfismo de empalme (II.ω⁺): hand_off_bundle_to_phase3                    ║
# ║     ≡ objeto inicial / continuación de Fase III.                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── §2.1 CognitiveWateringModule — RIEGO ──────────────────────────────────
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
    hill_compatible: bool = True


class CognitiveWateringModule:
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
        cls, handoff: SeedHandoff, toon_str: str, base_json_str: str
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
            seed_entropy_nats=float(handoff.seed_audit.entropy),
            local_verdict=local,
        )

    @classmethod
    def moisten_from_germ(
        cls,
        celestial_handoff: SeedCelestialHandoff,
        toon_str: str,
        base_json_str: str,
    ) -> WateringReport:
        r"""Primer acto de Fase II sobre 𝒢_I: riego condicionado a la región de Hill."""
        report = cls.apply_water(celestial_handoff.base_handoff, toon_str, base_json_str)
        hill_ok = bool(celestial_handoff.celestial_germ.maupertuis_germ.is_in_hill_region)
        local = report.local_verdict if hill_ok else report.local_verdict.meet(
            HeytingOmega3.DEGRADED
        )
        return WateringReport(
            tokens_toon=report.tokens_toon,
            tokens_json=report.tokens_json,
            h_toon=report.h_toon,
            h_json=report.h_json,
            fat_reduction_pct=report.fat_reduction_pct,
            kv_compression_ratio=report.kv_compression_ratio,
            seed_entropy_nats=report.seed_entropy_nats,
            local_verdict=local,
            hill_compatible=hill_ok,
        )


# ── §2.2 CognitiveIlluminationModule — LUZ ────────────────────────────────
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
    unitary_drift: float = 0.0
    integrator: str = "rk4"


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
    BROCKETT_MAX_STEPS: Final[int] = 60
    BROCKETT_DT: Final[float] = 0.05
    BROCKETT_TOL: Final[float] = 1.0e-9
    RENYI_ALPHA: Final[float] = 1.5
    FOCK_RESONANCE_TOL: Final[float] = 0.15

    @classmethod
    def _N(cls, n: int) -> RealMatrix:
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
    def _cayley_step(cls, U: np.ndarray, A: np.ndarray, dt: float) -> ComplexMatrix:
        n = U.shape[0]
        I = np.eye(n, dtype=np.complex128)
        half = 0.5 * dt * A
        try:
            U_next = la.solve(I - half, (I + half) @ U)
        except la.LinAlgError:
            U_next = U + dt * (A @ U)
        return cls._project_unitary(U_next)

    @classmethod
    def lyapunov_alignment(cls, rho: np.ndarray, N: np.ndarray) -> float:
        return float(np.trace(rho @ N).real)

    @classmethod
    def _brockett_unitary_rk4(
        cls, rho_init: np.ndarray
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
            r = _hermitian_part(U_cur @ rho0 @ U_cur.conj().T)
            return cls._anti_hermitian_generator(r, N)

        for step in range(cls.BROCKETT_MAX_STEPS):
            k1 = A_of(U) @ U
            k2 = A_of(U + 0.5 * dt * k1) @ (U + 0.5 * dt * k1)
            k3 = A_of(U + 0.5 * dt * k2) @ (U + 0.5 * dt * k2)
            k4 = A_of(U + dt * k3) @ (U + dt * k3)
            U_next = cls._project_unitary(U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
            rho_next = _hermitian_part(U_next @ rho0 @ U_next.conj().T)
            if np.linalg.norm(rho_next - rho, "fro") < cls.BROCKETT_TOL:
                U, rho = U_next, rho_next
                converged = True
                break
            U, rho = U_next, rho_next
        rho = DensityOperatorAlgebra.sanitize(rho)
        lam1 = DensityOperatorAlgebra.spectrum_descending(rho)
        cert = BrockettPurificationCertificate(
            initial_alignment=init_align,
            final_alignment=cls.lyapunov_alignment(rho, N),
            initial_purity=init_pur,
            final_purity=float(np.sum(lam1 ** 2)),
            lyapunov_drop=float(init_align - cls.lyapunov_alignment(rho, N)),
            iterations=step + 1,
            converged=converged,
            isospectral_drift=float(np.linalg.norm(lam1 - lam0, ord=2)),
            unitary_drift=float(np.linalg.norm(U.conj().T @ U - np.eye(n), "fro")),
            integrator="rk4-polar",
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


class CognitiveDisciplineModule:
    @classmethod
    def audit(cls, handoff: SeedHandoff, eta_star: float = 1.5) -> BanachContractionReport:
        return BanachContractionAlgebra.audit(
            handoff.rho_seed, eta_star, handoff.mac_gamma
        )

    @classmethod
    def sharp_poincare_constant(cls, n: int) -> float:
        return 0.5 if n <= 1 else 0.5


# ── §2.4 PoincareCelestialVerifier — KAM/Bruno + MELNIKOV + CZ ────────────
@dataclass(frozen=True, slots=True)
class KAMAudit:
    min_divisor: float
    novikov_weight: float
    maurercartan_residual: float
    volume_drift: float
    resonance_gap: float
    tau: float
    gamma: float
    is_diophantine: bool
    is_kam_stable: bool
    local_verdict: HeytingOmega3
    engine_ok: bool = True
    bruno_sum: float = 0.0
    is_bruno: bool = True
    twist_determinant: float = 1.0
    is_kolmogorov_nondegenerate: bool = True
    nekhoroshev_time: float = 0.0


@dataclass(frozen=True, slots=True)
class MelnikovAudit:
    melnikov_value: float
    melnikov_derivative: float
    is_simple_zero: bool
    homoclinic_splitting: float
    local_verdict: HeytingOmega3
    engine_ok: bool = True
    quadrature_nodes: int = 0
    l_infinity_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class ReturnMapAudit:
    floquet_multipliers: np.ndarray
    lyapunov_spectrum: np.ndarray
    max_lyapunov: float
    floquet_parabolic: float
    is_hyperbolic: bool
    is_elliptic: bool
    is_parabolic: bool
    trace_M: float
    det_M: float
    local_verdict: HeytingOmega3
    engine_ok: bool = True
    conley_zehnder_index: int = 0
    is_nondegenerate: bool = True
    birkhoff_twist: float = 0.0
    symplectic_residual: float = 0.0


class PoincareCelestialVerifier:
    r"""Verificador celeste. Recibe 𝒢_I como dato obligatorio."""

    def __init__(self, germ: _PoincareCartanSeedGerm) -> None:
        self._germ = germ

    @property
    def germ(self) -> _PoincareCartanSeedGerm:
        return self._germ

    @staticmethod
    def compute_bruno_sum(
        frequency_vector_omega: np.ndarray,
        max_octaves: int = _BRUNO_MAX_OCTAVES,
    ) -> Tuple[float, bool]:
        omega = np.asarray(frequency_vector_omega, dtype=np.float64).ravel()
        n = int(omega.size)
        if n == 0:
            return float("inf"), False
        acc = 0.0
        finite = True
        for nu in range(max(1, max_octaves)):
            K = 1 << nu
            if n > 3 or K > 8:
                candidates = [np.eye(n, dtype=np.int64), -np.eye(n, dtype=np.int64)]
                extra = []
                for i in range(n):
                    for j in range(i + 1, n):
                        v = np.zeros(n, dtype=np.int64)
                        v[i] = 1
                        v[j] = 1
                        extra.append(v)
                        extra.append(-v)
                k_all = np.vstack(candidates + extra)
            else:
                ranges = [np.arange(-K, K + 1) for _ in range(n)]
                grid = np.meshgrid(*ranges, indexing="ij")
                k_all = np.stack([g.ravel() for g in grid], axis=1).astype(np.int64)
            norms = np.abs(k_all).sum(axis=1)
            keep = (norms > 0) & (norms <= K)
            if not np.any(keep):
                omega_k = _WILKINSON_DRIFT_LIMIT
            else:
                inner = np.abs(k_all[keep] @ omega)
                omega_k = float(np.min(inner)) if inner.size else _WILKINSON_DRIFT_LIMIT
            omega_k = max(omega_k, _MACHINE_EPS)
            acc += (2.0 ** (-nu)) * math.log(1.0 / omega_k)
            if not np.isfinite(acc):
                finite = False
                break
        return float(acc), bool(finite and acc < 1.0e6)

    @staticmethod
    def compute_twist_determinant(jacobian_M: np.ndarray) -> Tuple[float, bool]:
        M = np.asarray(jacobian_M, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.size == 0:
            return 0.0, False
        try:
            det_m = float(np.real(la.det(M)))
            det_mi = float(np.real(la.det(M - np.eye(M.shape[0]))))
        except la.LinAlgError:
            return 0.0, False
        twist = float(abs(det_mi))
        nondeg = bool(twist > _BIRKHOFF_TWIST_FLOOR and abs(det_m - 1.0) < 1e-3)
        return twist, nondeg

    @classmethod
    def compute_nekhoroshev_time(cls, eps_pert: float, n_dim: int) -> float:
        eps = max(float(abs(eps_pert)), _MACHINE_EPS)
        expo = 1.0 / max(2 * max(n_dim, 1), 1)
        return float(math.exp(_clip_log_exp(_NEKHOROSHEV_C / (eps ** expo))))

    def compute_poincare_small_divisors_spectrum(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        jacobian_M: np.ndarray,
        tau: float = _KAM_TAU_FLOOR,
        gamma: float = _KAM_GAMMA_FLOOR,
        novikov_valuation_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> KAMAudit:
        try:
            omega = np.asarray(frequency_vector_omega, dtype=np.float64).ravel()
            wave_k = np.asarray(wave_vectors_k, dtype=np.float64)
            jac_m = np.asarray(jacobian_M, dtype=np.float64)
            n_freq = max(int(omega.size), 1)
            tau_eff = max(float(tau), float(n_freq - 1) + 1e-6)
            if wave_k.ndim == 1:
                divisors = np.abs(np.dot(wave_k, omega))
                k_norms = np.abs(wave_k)
                min_divisor = float(divisors)
                argmin_idx: Optional[int] = 0
                resonance_gap = float(divisors)
            else:
                divisors = np.abs(wave_k @ omega)
                k_norms = np.linalg.norm(wave_k, axis=1)
                if divisors.size == 0:
                    min_divisor = 1.0
                    argmin_idx = None
                    resonance_gap = 1.0
                else:
                    argmin_idx = int(np.argmin(divisors))
                    min_divisor = float(divisors[argmin_idx])
                    resonance_gap = min_divisor
            novikov_weight = float(
                np.exp(
                    -np.clip(
                        novikov_valuation_T / (_WILKINSON_DRIFT_LIMIT + min_divisor),
                        0.0,
                        _LOG_EXP_CLIP,
                    )
                )
            )
            mc_residual = float(abs(min_divisor * novikov_weight))
            if argmin_idx is not None and wave_k.ndim > 1 and wave_k.size > 0:
                k_norm_min = float(max(k_norms[argmin_idx], 1.0))
                is_diophantine = bool(min_divisor * (k_norm_min ** tau_eff) >= gamma)
            else:
                is_diophantine = bool(min_divisor >= _WILKINSON_DRIFT_LIMIT)
            if jac_m.ndim == 2 and jac_m.shape[0] == jac_m.shape[1] and jac_m.size > 0:
                volume_drift = float(abs(float(la.det(jac_m)) - 1.0))
            else:
                volume_drift = 0.0
            bruno_sum, is_bruno = self.compute_bruno_sum(omega)
            twist_det, is_twist = self.compute_twist_determinant(jac_m)
            eps_proxy = max(volume_drift, 1.0 - min(min_divisor, 1.0), _MACHINE_EPS)
            t_nek = self.compute_nekhoroshev_time(eps_proxy, n_freq)
            is_kam_stable = bool(
                (min_divisor >= _KAM_THRESHOLD * safety_margin)
                and (volume_drift <= _WILKINSON_DRIFT_LIMIT)
                and is_diophantine
                and is_bruno
            )
            if min_divisor <= _WILKINSON_DRIFT_LIMIT or volume_drift > _WILKINSON_DRIFT_LIMIT:
                local = HeytingOmega3.VETOED
            elif not is_diophantine or not is_bruno:
                local = HeytingOmega3.DEGRADED
            elif is_kam_stable:
                local = HeytingOmega3.COHERENT
            elif min_divisor >= (_KAM_THRESHOLD * safety_margin) * _DEGRADATION_FACTOR:
                local = HeytingOmega3.DEGRADED
            else:
                local = HeytingOmega3.VETOED
            return KAMAudit(
                min_divisor=min_divisor,
                novikov_weight=novikov_weight,
                maurercartan_residual=mc_residual,
                volume_drift=volume_drift,
                resonance_gap=resonance_gap,
                tau=float(tau_eff),
                gamma=float(gamma),
                is_diophantine=is_diophantine,
                is_kam_stable=is_kam_stable,
                local_verdict=local,
                engine_ok=True,
                bruno_sum=bruno_sum,
                is_bruno=is_bruno,
                twist_determinant=twist_det,
                is_kolmogorov_nondegenerate=is_twist,
                nekhoroshev_time=t_nek,
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_small_divisors_spectrum: %s", exc)
            return KAMAudit(
                min_divisor=float("inf"),
                novikov_weight=0.0,
                maurercartan_residual=float("inf"),
                volume_drift=float("inf"),
                resonance_gap=float("inf"),
                tau=float(tau),
                gamma=float(gamma),
                is_diophantine=False,
                is_kam_stable=False,
                local_verdict=HeytingOmega3.VETOED,
                engine_ok=False,
                bruno_sum=float("inf"),
                is_bruno=False,
                twist_determinant=0.0,
                is_kolmogorov_nondegenerate=False,
                nekhoroshev_time=0.0,
            )

    def compute_arnold_resonance_lattice(
        self,
        frequency_vector_omega: np.ndarray,
        max_order: int = 4,
        tol: float = 1e-6,
    ) -> np.ndarray:
        omega = np.asarray(frequency_vector_omega, dtype=np.float64).ravel()
        n = omega.size
        if n == 0 or max_order < 1:
            return np.zeros((0, max(n, 1)), dtype=np.int64)
        ranges = [np.arange(-max_order, max_order + 1) for _ in range(n)]
        grid = np.meshgrid(*ranges, indexing="ij")
        k_all = np.stack([g.ravel() for g in grid], axis=1).astype(np.int64)
        norms = np.abs(k_all).sum(axis=1)
        keep = (norms > 0) & (norms <= max_order)
        if not np.any(keep):
            return np.zeros((0, n), dtype=np.int64)
        k_cand = k_all[keep]
        inner = np.abs(k_cand @ omega)
        return k_cand[inner < tol]

    def compute_melnikov_function(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float = _MELNIKOV_T_INF,
        n_quad: int = _MELNIKOV_QUAD_NODES,
        safety_margin: float = 1.0,
    ) -> MelnikovAudit:
        try:
            t0s = np.asarray(t0_grid, dtype=np.float64).ravel()
            if t0s.size == 0:
                raise ValueError("t0_grid no puede estar vacío.")
            n_quad = int(max(16, n_quad))
            nodes, weights = np.polynomial.legendre.leggauss(n_quad)
            t_nodes = t_inf * nodes
            w_nodes = t_inf * weights
            omega = self._germ.omega
            melnikov_vals = np.zeros(t0s.size, dtype=np.float64)
            for i, t0 in enumerate(t0s):
                acc = 0.0
                comp = 0.0
                for t_shift, w in zip(t_nodes, w_nodes):
                    try:
                        x = np.asarray(homoclinic_flow(float(t_shift - t0)), dtype=np.float64)
                    except Exception:
                        continue
                    if x.size != omega.shape[0]:
                        continue
                    try:
                        pb = PoincareCelestialGeometry.poisson_bracket(
                            hamiltonian_0, hamiltonian_1, x, omega
                        )
                    except (TypeError, ValueError, FloatingPointError):
                        pb = 0.0
                    if not np.isfinite(pb):
                        pb = 0.0
                    y = pb * float(w) - comp
                    t = acc + y
                    comp = (t - acc) - y
                    acc = t
                melnikov_vals[i] = acc
            idx_min = int(np.argmin(np.abs(melnikov_vals)))
            m_val = float(melnikov_vals[idx_min])
            linf = float(np.max(np.abs(melnikov_vals))) if melnikov_vals.size else 0.0
            if t0s.size >= 2 and 0 < idx_min < t0s.size - 1:
                dm = (melnikov_vals[idx_min + 1] - melnikov_vals[idx_min - 1]) / (
                    t0s[idx_min + 1] - t0s[idx_min - 1]
                )
            elif t0s.size >= 2:
                dm = (melnikov_vals[-1] - melnikov_vals[0]) / max(
                    t0s[-1] - t0s[0], _MACHINE_EPS
                )
            else:
                dm = 0.0
            is_simple = bool(
                abs(m_val) < _WILKINSON_DRIFT_LIMIT and abs(dm) > _WILKINSON_DRIFT_LIMIT
            )
            try:
                x_star = np.asarray(homoclinic_flow(float(t0s[idx_min])), dtype=np.float64)
                grad_h0 = PoincareCelestialGeometry.compute_gradient_csmd(
                    hamiltonian_0, x_star
                )
                grad_norm = float(np.linalg.norm(grad_h0, 2))
            except Exception:
                grad_norm = 0.0
            splitting = float(abs(m_val) / max(grad_norm, _WILKINSON_DRIFT_LIMIT))
            tol = _MELNIKOV_THRESHOLD * safety_margin
            if is_simple:
                local = HeytingOmega3.VETOED
            elif abs(m_val) >= tol:
                local = HeytingOmega3.COHERENT
            elif abs(m_val) >= tol * _DEGRADATION_FACTOR:
                local = HeytingOmega3.DEGRADED
            else:
                local = HeytingOmega3.VETOED
            return MelnikovAudit(
                melnikov_value=m_val,
                melnikov_derivative=float(dm),
                is_simple_zero=is_simple,
                homoclinic_splitting=splitting,
                local_verdict=local,
                engine_ok=True,
                quadrature_nodes=n_quad,
                l_infinity_norm=linf,
            )
        except Exception as exc:
            logger.error("Fallo en compute_melnikov_function: %s", exc)
            return MelnikovAudit(
                melnikov_value=float("nan"),
                melnikov_derivative=float("nan"),
                is_simple_zero=False,
                homoclinic_splitting=float("nan"),
                local_verdict=HeytingOmega3.VETOED,
                engine_ok=False,
            )

    @staticmethod
    def compute_conley_zehnder_index(M: np.ndarray) -> Tuple[int, bool]:
        A = np.asarray(M, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] % 2 != 0:
            return 0, False
        n = A.shape[0] // 2
        try:
            det_mi = float(np.real(la.det(A - np.eye(A.shape[0]))))
        except la.LinAlgError:
            return 0, False
        nondeg = bool(abs(det_mi) > _FLOQUET_PARABOLIC)
        try:
            P = la.sqrtm(A.T @ A).real
            U = A @ la.inv(P) if abs(la.det(P)) > _MACHINE_EPS else A
            ev = la.eigvals(U)
        except (la.LinAlgError, ValueError):
            ev = la.eigvals(A)
        args = np.angle(ev)
        pos = int(np.sum(args > _FLOQUET_PARABOLIC))
        neg = int(np.sum(args < -_FLOQUET_PARABOLIC))
        return int(n - pos + neg), nondeg

    def compute_poincare_return_map(
        self,
        jacobian_M: np.ndarray,
        period_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> ReturnMapAudit:
        try:
            M = np.asarray(jacobian_M, dtype=np.float64)
            if M.ndim == 1:
                side = int(round(np.sqrt(M.size)))
                if side * side != M.size:
                    raise ValueError("jacobian_M plano no es cuadrado perfecto.")
                M = M.reshape(side, side)
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError("jacobian_M debe ser cuadrada.")
            ev = la.eigvals(M)
            magnitudes = np.abs(ev)
            lyap = np.log(np.maximum(magnitudes, _MACHINE_EPS)) / max(
                abs(period_T), _MACHINE_EPS
            )
            lyap = np.clip(lyap, -_LYAPUNOV_CLIP, _LYAPUNOV_CLIP)
            max_lyap = float(np.max(lyap)) if lyap.size else 0.0
            on_circle = np.abs(magnitudes - 1.0) <= _FLOQUET_PARABOLIC
            is_parabolic = bool(
                np.any(np.abs(ev - 1.0) < _FLOQUET_PARABOLIC)
                or np.any(np.abs(ev + 1.0) < _FLOQUET_PARABOLIC)
            )
            is_elliptic = bool(np.all(on_circle) and not is_parabolic)
            is_hyperbolic = bool(
                (not is_elliptic) and (not is_parabolic) and np.any(~on_circle)
            )
            if not (is_elliptic or is_parabolic or is_hyperbolic):
                is_hyperbolic = bool(np.any(~on_circle))
            floq_par = float(np.max(np.abs(np.abs(ev) - 1.0))) if ev.size else 0.0
            cz, nondeg = self.compute_conley_zehnder_index(M)
            twist, _ = self.compute_twist_determinant(M)
            try:
                om_m = PoincareCelestialGeometry.generate_canonical_symplectic_form(
                    M.shape[0]
                )
                symp_res = float(np.linalg.norm(M.T @ om_m @ M - om_m, "fro"))
            except ValueError:
                symp_res = float("inf")
            tol = _FLOQUET_PARABOLIC * safety_margin
            if is_elliptic and abs(max_lyap) <= tol * _DEGRADATION_FACTOR and nondeg:
                local = HeytingOmega3.COHERENT
            elif is_parabolic or floq_par <= tol:
                local = HeytingOmega3.DEGRADED
            elif abs(max_lyap) <= tol / max(_DEGRADATION_FACTOR, _MACHINE_EPS):
                local = HeytingOmega3.DEGRADED
            else:
                local = HeytingOmega3.VETOED
            return ReturnMapAudit(
                floquet_multipliers=ev,
                lyapunov_spectrum=lyap,
                max_lyapunov=max_lyap,
                floquet_parabolic=floq_par,
                is_hyperbolic=is_hyperbolic,
                is_elliptic=is_elliptic,
                is_parabolic=is_parabolic,
                trace_M=float(np.trace(M)),
                det_M=float(np.real(la.det(M))),
                local_verdict=local,
                engine_ok=True,
                conley_zehnder_index=cz,
                is_nondegenerate=nondeg,
                birkhoff_twist=twist,
                symplectic_residual=symp_res,
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_return_map: %s", exc)
            return ReturnMapAudit(
                floquet_multipliers=np.array([], dtype=np.complex128),
                lyapunov_spectrum=np.array([], dtype=np.float64),
                max_lyapunov=float("inf"),
                floquet_parabolic=float("inf"),
                is_hyperbolic=False,
                is_elliptic=False,
                is_parabolic=False,
                trace_M=float("nan"),
                det_M=float("nan"),
                local_verdict=HeytingOmega3.VETOED,
                engine_ok=False,
            )


# ── §2.5 CropGrowthBundle + _PoincareCelestialGrowthBundle ────────────────
@dataclass(frozen=True, slots=True)
class CropGrowthBundle:
    handoff: SeedHandoff
    watering: WateringReport
    rho_illum: np.ndarray
    brockett_c: BrockettPurificationCertificate
    renyi_c: RenyiPurificationCertificate
    fock_c: FockAnnihilationCertificate
    discipline: BanachContractionReport

    def phase_content_bytes(self) -> bytes:
        return hashlib.sha256(
            self.handoff.spectral_hash.encode("ascii")
            + np.ascontiguousarray(self.rho_illum).tobytes()
            + f"{self.watering.fat_reduction_pct:.10f}".encode("ascii")
            + f"{self.discipline.seed_spectral_radius:.10f}".encode("ascii")
            + f"{self.discipline.mac_contraction_coef:.10f}".encode("ascii")
            + f"{self.renyi_c.purity_after:.10f}".encode("ascii")
            + f"{self.fock_c.resonance_residual:.10f}".encode("ascii")
        ).digest()


@dataclass(frozen=True, slots=True)
class _PoincareCelestialGrowthBundle:
    r"""
    GÉRMEN CELESTE DE CRECIMIENTO (terminal de Fase II, inicial de Fase III).
    Extiende CropGrowthBundle con KAM/Bruno, Melnikov, retorno CZ.
    """
    base_bundle: CropGrowthBundle
    celestial_handoff: SeedCelestialHandoff
    kam_audit: Optional[KAMAudit]
    melnikov_audit: Optional[MelnikovAudit]
    return_map: Optional[ReturnMapAudit]
    celestial_verdict: HeytingOmega3

    def celestial_content_bytes(self) -> bytes:
        parts = [
            np.ascontiguousarray(self.base_bundle.rho_illum).tobytes(),
            self.celestial_verdict.name.encode("ascii"),
        ]
        if self.kam_audit is not None:
            parts.append(f"{self.kam_audit.min_divisor:.12f}".encode("ascii"))
            parts.append(self.kam_audit.local_verdict.name.encode("ascii"))
            parts.append(f"{self.kam_audit.bruno_sum:.12f}".encode("ascii"))
        if self.melnikov_audit is not None:
            parts.append(f"{self.melnikov_audit.melnikov_value:.12f}".encode("ascii"))
            parts.append(self.melnikov_audit.local_verdict.name.encode("ascii"))
        if self.return_map is not None:
            parts.append(f"{self.return_map.max_lyapunov:.12f}".encode("ascii"))
            parts.append(self.return_map.local_verdict.name.encode("ascii"))
            parts.append(f"CZ={self.return_map.conley_zehnder_index}".encode("ascii"))
        return hashlib.sha256(b"|".join(parts)).digest()


class CropGrowthPipeline:
    @classmethod
    def synthesize(
        cls,
        handoff: SeedHandoff,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
        eta_star: float = 1.5,
    ) -> CropGrowthBundle:
        watering = CognitiveWateringModule.apply_water(handoff, toon_str, base_json_str)
        rho_illum, brockett_c, renyi_c, fock_c = CognitiveIlluminationModule.apply_light(
            handoff, renyi_alpha
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
    def synthesize_celestial_bundle(
        cls,
        celestial_handoff: SeedCelestialHandoff,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
        eta_star: float = 1.5,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        period_T: float = 1.0,
        novikov_valuation_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> _PoincareCelestialGrowthBundle:
        r"""
        MORFISMO TERMINAL DE LA FASE II ≅ OBJETO INICIAL DE LA FASE III.
        Riego-desde-gérmen + luz + disciplina + KAM/Bruno + Melnikov + CZ.
        Canales ausentes no votan (neutros = ⊤).
        """
        watering = CognitiveWateringModule.moisten_from_germ(
            celestial_handoff, toon_str, base_json_str
        )
        rho_illum, brockett_c, renyi_c, fock_c = CognitiveIlluminationModule.apply_light(
            celestial_handoff.base_handoff, renyi_alpha
        )
        discipline = CognitiveDisciplineModule.audit(
            celestial_handoff.base_handoff, eta_star
        )
        base_bundle = CropGrowthBundle(
            handoff=celestial_handoff.base_handoff,
            watering=watering,
            rho_illum=rho_illum,
            brockett_c=brockett_c,
            renyi_c=renyi_c,
            fock_c=fock_c,
            discipline=discipline,
        )
        verifier = PoincareCelestialVerifier(celestial_handoff.celestial_germ)
        kam_audit: Optional[KAMAudit] = None
        melnikov_audit: Optional[MelnikovAudit] = None
        return_map_audit: Optional[ReturnMapAudit] = None
        if (
            frequency_vector_omega is not None
            and wave_vectors_k is not None
            and jacobian_M is not None
        ):
            kam_audit = verifier.compute_poincare_small_divisors_spectrum(
                frequency_vector_omega=frequency_vector_omega,
                wave_vectors_k=wave_vectors_k,
                jacobian_M=jacobian_M,
                novikov_valuation_T=novikov_valuation_T,
                safety_margin=safety_margin,
            )
        if (
            homoclinic_flow is not None
            and hamiltonian_0 is not None
            and hamiltonian_1 is not None
            and t0_grid is not None
        ):
            melnikov_audit = verifier.compute_melnikov_function(
                homoclinic_flow=homoclinic_flow,
                hamiltonian_0=hamiltonian_0,
                hamiltonian_1=hamiltonian_1,
                t0_grid=t0_grid,
                safety_margin=safety_margin,
            )
        if jacobian_M is not None:
            return_map_audit = verifier.compute_poincare_return_map(
                jacobian_M=jacobian_M, period_T=period_T, safety_margin=safety_margin
            )
        celestial_verdict = HeytingOmega3.infimum(
            *[
                a.local_verdict
                for a in (kam_audit, melnikov_audit, return_map_audit)
                if a is not None
            ]
        )
        return _PoincareCelestialGrowthBundle(
            base_bundle=base_bundle,
            celestial_handoff=celestial_handoff,
            kam_audit=kam_audit,
            melnikov_audit=melnikov_audit,
            return_map=return_map_audit,
            celestial_verdict=celestial_verdict,
        )

    @classmethod
    def continue_into_phase3(
        cls,
        bundle: CropGrowthBundle,
        external_verdict: HeytingOmega3,
        reason_prefix: str = "CROP-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
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
            f"pure_ok={audit.verified_vacuum_pure} "
            f"KAM={bundle.discipline.is_kam_stable} "
            f"Bifurc={bundle.discipline.is_pyriform_bifurcated}"
        )
        actuation = CognitiveFaithModule.fire(verdict, reason)
        return verdict, actuation

    @classmethod
    def continue_celestial_into_phase3(
        cls,
        bundle: _PoincareCelestialGrowthBundle,
        external_verdict: HeytingOmega3,
        reason_prefix: str = "CROP-CELESTIAL-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        return HeytingCropAdjudicator.adjudicate_celestial(
            bundle, external_verdict, reason_prefix=reason_prefix
        )

    # ── II.ω⁺  EMPALME FORMAL FASE II → FASE III ─────────────────────────
    @classmethod
    def hand_off_bundle_to_phase3(
        cls,
        bundle: _PoincareCelestialGrowthBundle,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
        reason_prefix: str = "CROP-CELESTIAL-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        ÚLTIMO MÉTODO DE LA FASE II ≡ PRIMER MORFISMO OPERATIVO DE LA FASE III.
        ═══════════════════════════════════════════════════════════════════════
        Consume 𝒢_II y abre la adjudicación Heyting Ω₃⁷⊕celeste + Crowbar.
        """
        return cls.continue_celestial_into_phase3(
            bundle, external_verdict=external_verdict, reason_prefix=reason_prefix
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · FE + ADJUDICACIÓN Ω₃⁷⊕CELESTE + CROWBAR + MAC + PASAPORTE        ║
# ║                                                                           ║
# ║ Continuación directa de CropGrowthPipeline.hand_off_bundle_to_phase3.     ║
# ║ Objeto inicial: 𝒢_II. Terminal: CropHarvestYield ⊗ Passport Merkle.       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── §3.1 Adjudicador Ω₃⁷ ⊕ canal celeste ──────────────────────────────────
class HeytingCropAdjudicator:
    r"""
    Ω₃⁷ = producto de Heyting de 7 canales clásicos:
      1. Riego (KV / Hill)  2. Disciplina (Banach ⊕ MAC)
      3. Fock  4. Brockett  5. Rényi  6. Flag consistencia  7. Vacío falso
    Canal 8 (celeste) = ínfimo KAM ∧ Melnikov ∧ Retorno, ⋀-reducido a Ω₃.
    El veredicto global es el ínfimo del frame.
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
        ok = bundle.brockett_c.converged or bundle.brockett_c.isospectral_drift < 1e-3
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
    def channel_vector_omega37(
        cls,
        bundle: CropGrowthBundle,
        celestial: Optional[HeytingOmega3] = None,
    ) -> Tuple[HeytingOmega3, ...]:
        classic = (
            bundle.watering.local_verdict,
            bundle.discipline.local_verdict,
            cls._fock_rule(bundle),
            cls._brockett_rule(bundle),
            cls._renyi_rule(bundle),
            cls._flag_rule(bundle),
            cls._vacuum_claim_rule(bundle),
        )
        if celestial is None:
            return classic
        return classic + (celestial,)

    @classmethod
    def adjudicate(
        cls, bundle: CropGrowthBundle, external_verdict: HeytingOmega3
    ) -> HeytingOmega3:
        local = HeytingOmega3.infimum(*cls.channel_vector_omega37(bundle))
        if bundle.discipline.is_pyriform_bifurcated:
            local = local.meet(HeytingOmega3.VETOED)
        return local.meet(external_verdict)

    @classmethod
    def adjudicate_omega37_celestial(
        cls,
        celestial_bundle: _PoincareCelestialGrowthBundle,
        external_verdict: HeytingOmega3,
    ) -> Tuple[HeytingOmega3, Tuple[HeytingOmega3, ...]]:
        r"""Primer morfismo operativo de Fase III sobre 𝒢_II."""
        channels = cls.channel_vector_omega37(
            celestial_bundle.base_bundle, celestial_bundle.celestial_verdict
        )
        local = HeytingOmega3.infimum(*channels)
        if celestial_bundle.base_bundle.discipline.is_pyriform_bifurcated:
            local = local.meet(HeytingOmega3.VETOED)
        if celestial_bundle.celestial_handoff.celestial_germ.symplectic_capacity <= 0.0:
            local = local.meet(HeytingOmega3.DEGRADED)
        return local.meet(external_verdict), channels

    @classmethod
    def adjudicate_celestial(
        cls,
        bundle: _PoincareCelestialGrowthBundle,
        external_verdict: HeytingOmega3,
        reason_prefix: str = "CROP-CELESTIAL-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        final_verdict, channels = cls.adjudicate_omega37_celestial(
            bundle, external_verdict
        )
        ch_names = [c.name for c in channels]
        reason = (
            f"{reason_prefix}::Ω₃⁷⊕C={ch_names} "
            f"water={bundle.base_bundle.watering.local_verdict.name} "
            f"disc={bundle.base_bundle.discipline.local_verdict.name} "
            f"celestial={bundle.celestial_verdict.name} "
            f"KAM={(bundle.kam_audit.local_verdict.name if bundle.kam_audit else 'n/a')} "
            f"Bruno={(bundle.kam_audit.is_bruno if bundle.kam_audit else 'n/a')} "
            f"Mel={(bundle.melnikov_audit.local_verdict.name if bundle.melnikov_audit else 'n/a')} "
            f"Ret={(bundle.return_map.local_verdict.name if bundle.return_map else 'n/a')} "
            f"CZ={(bundle.return_map.conley_zehnder_index if bundle.return_map else 'n/a')} "
            f"Mau_n(q)={bundle.celestial_handoff.celestial_germ.maupertuis_germ.refractive_index:.4f}"
        )
        actuation = CognitiveFaithModule.fire(final_verdict, reason)
        return final_verdict, actuation


# ── §3.2 CognitiveFaithModule — FE / Crowbar BT151 ────────────────────────
@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    interlock_fired: bool
    actuation_latency_ns: float
    gpio_pin: str
    device: str
    reason: str
    provenance_hash: str


class CognitiveFaithModule:
    GPIO_PIN: Final[str] = "GPIO14"
    DEVICE: Final[str] = "BT151_CROWBAR"
    NOMINAL_LATENCY_NS: Final[float] = 392.15

    @classmethod
    def fire(cls, verdict: HeytingOmega3, reason: str = "") -> CrowbarActuationReport:
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
            cls.GPIO_PIN,
            cls.NOMINAL_LATENCY_NS,
            reason,
        )
        return CrowbarActuationReport(
            interlock_fired=True,
            actuation_latency_ns=cls.NOMINAL_LATENCY_NS,
            gpio_pin=cls.GPIO_PIN,
            device=cls.DEVICE,
            reason=reason,
            provenance_hash=prov,
        )

    def trigger_esp32_hardware_veto(
        self, reason: str = "HARDWARE_VETO"
    ) -> CrowbarActuationReport:
        return self.fire(HeytingOmega3.VETOED, reason)


# ── §3.3 Cosecha y pasaporte de gobernanza ────────────────────────────────
@dataclass(frozen=True, slots=True)
class CropHarvestYield:
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
    is_kam_stable: bool = True
    is_pyriform_bifurcated: bool = False
    celestial_verdict: Optional[str] = None
    celestial_kam_verdict: Optional[str] = None
    celestial_kam_min_divisor: Optional[float] = None
    celestial_kam_is_diophantine: Optional[bool] = None
    celestial_kam_bruno_sum: Optional[float] = None
    celestial_kam_is_bruno: Optional[bool] = None
    celestial_nekhoroshev_time: Optional[float] = None
    celestial_melnikov_verdict: Optional[str] = None
    celestial_melnikov_value: Optional[float] = None
    celestial_melnikov_is_simple_zero: Optional[bool] = None
    celestial_return_verdict: Optional[str] = None
    celestial_return_max_lyapunov: Optional[float] = None
    celestial_return_is_elliptic: Optional[bool] = None
    celestial_conley_zehnder: Optional[int] = None
    maupertuis_refractive_index: Optional[float] = None
    maupertuis_hill_margin: Optional[float] = None
    gromov_capacity: Optional[float] = None
    gelfand_radius: Optional[float] = None
    omega37_channels: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True, slots=True)
class CropSovereignGovernancePassport:
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
    verdict: Optional[HeytingOmega3] = None
    reason: str = ""
    celestial_coherent_fraction: Optional[float] = None
    celestial_field_merkle_root: Optional[str] = None
    aggregate_celestial_verdict: Optional[str] = None
    aggregate_bruno_ok_fraction: Optional[float] = None
    mac_cptp_ok: bool = True


# ── §3.4 Soberano del Cultivo Cognitivo ───────────────────────────────────
class TOONCognitiveCropAgent:
    r"""
    Soberano de Calibre del Cultivo Cognitivo con mecánica celeste.
        Φ_I   : synthesize_poincare_cartan_germ → hand_off_germ_to_phase2
        Φ_II  : synthesize_celestial_bundle     → hand_off_bundle_to_phase3
        Φ_III : Ω₃⁷⊕celeste + Crowbar + MAC CPTP + Merkle
    """

    def __init__(
        self,
        agent_id: str = "CROP-SOVEREIGN-SABIO-01",
        dimension_mac: int = 4,
        mac_gamma: float = 0.2,
        eta_star: float = 1.5,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
        field_coupling: float = 1e-3,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        novikov_valuation_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1")
        self.agent_id = agent_id
        self.dimension_mac = int(dimension_mac)
        self.mac_gamma = float(np.clip(mac_gamma, 0.0, 1.0))
        self.eta_star = float(eta_star)
        self.renyi_alpha = float(renyi_alpha)
        self._H0 = float(hamiltonian_energy_H0)
        self._V = float(potential_energy_V)
        self._novikov_T = float(novikov_valuation_T)
        self._safety_margin = float(safety_margin)
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
        self.engine = None
        if TOONCognitiveCropEngine is not None:
            try:
                self.engine = TOONCognitiveCropEngine(
                    engine_id=f"ENGINE-{agent_id}",
                    dimension_mac=dimension_mac,
                    eta_star=eta_star,
                    renyi_alpha=renyi_alpha,
                    hamiltonian_energy_H0=hamiltonian_energy_H0,
                    potential_energy_V=potential_energy_V,
                    novikov_valuation_T=novikov_valuation_T,
                    safety_margin=safety_margin,
                )
            except TypeError:
                try:
                    self.engine = TOONCognitiveCropEngine(
                        engine_id=f"ENGINE-{agent_id}",
                        dimension_mac=dimension_mac,
                        eta_star=eta_star,
                        renyi_alpha=renyi_alpha,
                    )
                except Exception:
                    self.engine = None
        self.crowbar_interlock = CognitiveFaithModule()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _content_hash(self, bundle: CropGrowthBundle, verdict: HeytingOmega3) -> str:
        return hashlib.sha256(
            self.agent_id.encode("ascii")
            + bundle.handoff.seed_id.encode("ascii")
            + bundle.handoff.spectral_hash.encode("ascii")
            + bundle.phase_content_bytes()
            + verdict.name.encode("ascii")
        ).hexdigest()

    def _content_hash_celestial(
        self, celestial: _PoincareCelestialGrowthBundle, verdict: HeytingOmega3
    ) -> str:
        return hashlib.sha256(
            self.agent_id.encode("ascii")
            + celestial.base_bundle.handoff.seed_id.encode("ascii")
            + celestial.base_bundle.handoff.spectral_hash.encode("ascii")
            + celestial.base_bundle.phase_content_bytes()
            + celestial.celestial_content_bytes()
            + verdict.name.encode("ascii")
        ).hexdigest()

    def _phase1_handoff(self, seed_crystal: SeedCrystal) -> SeedHandoff:
        handoff = SeedHandoff.build(
            seed=seed_crystal,
            mac_field=self.mac_field,
            ground_projector=self.ground_projector,
        )
        self._advance_chain("F1", bytes.fromhex(handoff.spectral_hash))
        return handoff

    def _phase1_celestial_handoff(self, seed_crystal: SeedCrystal) -> SeedCelestialHandoff:
        handoff = SeedCelestialHandoff.build(
            seed=seed_crystal,
            mac_field=self.mac_field,
            ground_projector=self.ground_projector,
            hamiltonian_energy_H0=self._H0,
            potential_energy_V=self._V,
        )
        self._advance_chain("F1-CEL", bytes.fromhex(handoff.base_handoff.spectral_hash))
        return handoff

    def _phase2_grow(
        self, handoff: SeedHandoff, toon_str: str, base_json_str: str
    ) -> CropGrowthBundle:
        bundle = handoff.continue_into_phase2(
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=self.renyi_alpha,
            eta_star=self.eta_star,
        )
        self._advance_chain("F2", bundle.phase_content_bytes())
        return bundle

    def _phase2_grow_celestial(
        self,
        celestial_handoff: SeedCelestialHandoff,
        toon_str: str,
        base_json_str: str,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        period_T: float = 1.0,
    ) -> _PoincareCelestialGrowthBundle:
        bundle = celestial_handoff.hand_off_germ_to_phase2(
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=self.renyi_alpha,
            eta_star=self.eta_star,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            period_T=period_T,
            novikov_valuation_T=self._novikov_T,
            safety_margin=self._safety_margin,
        )
        self._advance_chain("F2-CEL", bundle.celestial_content_bytes())
        return bundle

    def _phase3_harvest(
        self, bundle: CropGrowthBundle, external_verdict: HeytingOmega3
    ) -> CropHarvestYield:
        final_verdict, actuation = CropGrowthPipeline.continue_into_phase3(
            bundle, external_verdict
        )
        return self._finalize_harvest(bundle, final_verdict, actuation, None)

    def _phase3_harvest_celestial(
        self,
        celestial: _PoincareCelestialGrowthBundle,
        external_verdict: HeytingOmega3,
    ) -> CropHarvestYield:
        final_verdict, actuation = CropGrowthPipeline.hand_off_bundle_to_phase3(
            celestial, external_verdict
        )
        return self._finalize_harvest(
            celestial.base_bundle, final_verdict, actuation, celestial
        )

    def _finalize_harvest(
        self,
        bundle: CropGrowthBundle,
        final_verdict: HeytingOmega3,
        actuation: CrowbarActuationReport,
        celestial_bundle: Optional[_PoincareCelestialGrowthBundle],
    ) -> CropHarvestYield:
        self._advance_chain(
            "F3", f"{final_verdict.name}|{actuation.provenance_hash}".encode("ascii")
        )
        is_inoculate = final_verdict == HeytingOmega3.COHERENT
        mac_cert = (
            self.mac_field.update(bundle.rho_illum)
            if is_inoculate
            else self.mac_field.idle_certificate()
        )
        fid_after = DensityOperatorAlgebra.uhlmann_fidelity(
            self.mac_field.rho, bundle.rho_illum
        )
        p_before = bundle.brockett_c.initial_purity
        p_after = DensityOperatorAlgebra.purity(bundle.rho_illum)
        purity_gain = float(p_after - p_before)
        s_before = float(bundle.handoff.seed_audit.entropy)
        s_after = DensityOperatorAlgebra.von_neumann_entropy(bundle.rho_illum)
        entropy_reduction = float(s_before - s_after)
        if celestial_bundle is not None:
            content_hash = self._content_hash_celestial(celestial_bundle, final_verdict)
        else:
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
        celestial_verdict_s = kam_v = kam_md = kam_diof = None
        kam_bruno = kam_is_bruno = nek_t = None
        mel_v = mel_val = mel_simple = None
        ret_v = ret_max_lyap = ret_ell = cz_idx = None
        mau_refr = mau_hill = gromov = None
        omega37: Optional[Tuple[str, ...]] = None
        gelfand = float(bundle.discipline.gelfand_radius)
        if celestial_bundle is not None:
            celestial_verdict_s = celestial_bundle.celestial_verdict.name
            germ = celestial_bundle.celestial_handoff.celestial_germ
            mau_refr = germ.maupertuis_germ.refractive_index
            mau_hill = germ.maupertuis_germ.hill_margin
            gromov = germ.symplectic_capacity
            _, channels = HeytingCropAdjudicator.adjudicate_omega37_celestial(
                celestial_bundle, final_verdict
            )
            omega37 = tuple(c.name for c in channels)
            if celestial_bundle.kam_audit is not None:
                kam_v = celestial_bundle.kam_audit.local_verdict.name
                kam_md = celestial_bundle.kam_audit.min_divisor
                kam_diof = celestial_bundle.kam_audit.is_diophantine
                kam_bruno = celestial_bundle.kam_audit.bruno_sum
                kam_is_bruno = celestial_bundle.kam_audit.is_bruno
                nek_t = celestial_bundle.kam_audit.nekhoroshev_time
            if celestial_bundle.melnikov_audit is not None:
                mel_v = celestial_bundle.melnikov_audit.local_verdict.name
                mel_val = celestial_bundle.melnikov_audit.melnikov_value
                mel_simple = celestial_bundle.melnikov_audit.is_simple_zero
            if celestial_bundle.return_map is not None:
                ret_v = celestial_bundle.return_map.local_verdict.name
                ret_max_lyap = celestial_bundle.return_map.max_lyapunov
                ret_ell = celestial_bundle.return_map.is_elliptic
                cz_idx = celestial_bundle.return_map.conley_zehnder_index
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
            is_kam_stable=bundle.discipline.is_kam_stable,
            is_pyriform_bifurcated=bundle.discipline.is_pyriform_bifurcated,
            celestial_verdict=celestial_verdict_s,
            celestial_kam_verdict=kam_v,
            celestial_kam_min_divisor=kam_md,
            celestial_kam_is_diophantine=kam_diof,
            celestial_kam_bruno_sum=kam_bruno,
            celestial_kam_is_bruno=kam_is_bruno,
            celestial_nekhoroshev_time=nek_t,
            celestial_melnikov_verdict=mel_v,
            celestial_melnikov_value=mel_val,
            celestial_melnikov_is_simple_zero=mel_simple,
            celestial_return_verdict=ret_v,
            celestial_return_max_lyapunov=ret_max_lyap,
            celestial_return_is_elliptic=ret_ell,
            celestial_conley_zehnder=cz_idx,
            maupertuis_refractive_index=mau_refr,
            maupertuis_hill_margin=mau_hill,
            gromov_capacity=gromov,
            gelfand_radius=gelfand,
            omega37_channels=omega37,
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
        t_start = time.perf_counter()
        logger.info(
            "═══ Siembra clásica | semilla=%s | origen=%s | γ_MAC=%.2f ═══",
            seed_crystal.crystal_id,
            seed_crystal.origin_agent,
            self.mac_gamma,
        )
        handoff = self._phase1_handoff(seed_crystal)
        bundle = self._phase2_grow(handoff, toon_str, base_json_str)
        harvest = self._phase3_harvest(bundle, external_verdict)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cosecha %s | Ω₃=%s | ΔP=%+.6f | ΔS=%+.6f | inoculada=%s | %.2f ms",
            harvest.crop_id,
            harvest.heyting_verdict.name,
            harvest.purity_gain,
            harvest.entropy_reduction,
            harvest.is_inoculated_into_mac,
            dt_ms,
        )
        return harvest

    def sow_and_cultivate_celestial(
        self,
        seed_crystal: SeedCrystal,
        toon_str: str,
        base_json_str: str,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        period_T: float = 1.0,
    ) -> CropHarvestYield:
        r"""Cultivo celestial: 𝒢_I → 𝒢_II → cosecha Ω₃⁷⊕celeste vía hand-offs anidados."""
        t_start = time.perf_counter()
        logger.info(
            "═══ Siembra celestial | semilla=%s | H₀=%.3f | V=%.3f ═══",
            seed_crystal.crystal_id,
            self._H0,
            self._V,
        )
        celestial_handoff = self._phase1_celestial_handoff(seed_crystal)
        celestial_bundle = self._phase2_grow_celestial(
            celestial_handoff=celestial_handoff,
            toon_str=toon_str,
            base_json_str=base_json_str,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            period_T=period_T,
        )
        harvest = self._phase3_harvest_celestial(celestial_bundle, external_verdict)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cosecha celestial %s | Ω₃=%s | celestial=%s | KAM=%s | Bruno=%s | "
            "Mel=%s | CZ=%s | %.2f ms",
            harvest.crop_id,
            harvest.heyting_verdict.name,
            harvest.celestial_verdict,
            harvest.celestial_kam_verdict,
            harvest.celestial_kam_is_bruno,
            harvest.celestial_melnikov_verdict,
            harvest.celestial_conley_zehnder,
            dt_ms,
        )
        return harvest

    def _germ_from_seed(self, seed_crystal: SeedCrystal) -> _PoincareCartanSeedGerm:
        return SeedCrystalSanitizer.synthesize_poincare_cartan_germ(
            seed=seed_crystal,
            rho_ground=self.ground_projector,
            mac_snapshot=self.mac_field.rho,
            mac_gamma=self.mac_gamma,
            hamiltonian_energy_H0=self._H0,
            potential_energy_V=self._V,
        )

    def audit_poincare_kam_stability(
        self,
        seed_crystal: SeedCrystal,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        jacobian_M: np.ndarray,
        tau: float = _KAM_TAU_FLOOR,
        gamma: float = _KAM_GAMMA_FLOOR,
    ) -> KAMAudit:
        verifier = PoincareCelestialVerifier(self._germ_from_seed(seed_crystal))
        return verifier.compute_poincare_small_divisors_spectrum(
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            tau=tau,
            gamma=gamma,
            novikov_valuation_T=self._novikov_T,
            safety_margin=self._safety_margin,
        )

    def audit_melnikov_homoclinic_splitting(
        self,
        seed_crystal: SeedCrystal,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float = _MELNIKOV_T_INF,
    ) -> MelnikovAudit:
        verifier = PoincareCelestialVerifier(self._germ_from_seed(seed_crystal))
        return verifier.compute_melnikov_function(
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            t_inf=t_inf,
            safety_margin=self._safety_margin,
        )

    def audit_poincare_return_map(
        self,
        seed_crystal: SeedCrystal,
        jacobian_M: np.ndarray,
        period_T: float = 1.0,
    ) -> ReturnMapAudit:
        verifier = PoincareCelestialVerifier(self._germ_from_seed(seed_crystal))
        return verifier.compute_poincare_return_map(
            jacobian_M=jacobian_M,
            period_T=period_T,
            safety_margin=self._safety_margin,
        )

    def compute_poincare_cartan_lambda(
        self, seed_crystal: SeedCrystal, x: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, PoincareCartanGerm]:
        germ = self._germ_from_seed(seed_crystal)
        if x is None:
            x = np.zeros(germ.omega.shape[0], dtype=np.float64)
        return PoincareCelestialGeometry.compute_poincare_cartan_lambda(
            x, hamiltonian_value=self._V, omega=germ.omega
        )

    def compute_maupertuis_jacobi_metric(
        self,
        seed_crystal: SeedCrystal,
        base_metric_g: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, MaupertuisJacobiGerm]:
        if base_metric_g is None:
            base_metric_g = np.eye(self.dimension_mac, dtype=np.float64)
        _ = seed_crystal
        return PoincareCelestialGeometry.compute_maupertuis_conformal_metric(
            self._H0, self._V, base_metric_g
        )

    def compute_poisson_bracket(
        self,
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        x: np.ndarray,
    ) -> float:
        omega = PoincareCelestialGeometry.generate_canonical_symplectic_form(
            2 * self.dimension_mac
        )
        return PoincareCelestialGeometry.poisson_bracket(
            hamiltonian_0, hamiltonian_1, x, omega
        )

    def stormer_verlet_step(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_v: Callable[[np.ndarray], np.ndarray],
        mass_inv: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return PoincareCelestialGeometry.stormer_verlet_step(q, p, grad_v, mass_inv, dt)

    def audit_field_governance(self) -> CropSovereignGovernancePassport:
        total = len(self.cultivated_crops_history)
        coherent = sum(
            1
            for c in self.cultivated_crops_history
            if c.heyting_verdict == HeytingOmega3.COHERENT
        )
        coherent_frac = (coherent / total) if total > 0 else 0.0
        celestial_harvests = [
            c for c in self.cultivated_crops_history if c.celestial_verdict is not None
        ]
        celestial_coherent = sum(
            1 for c in celestial_harvests if c.celestial_verdict == "COHERENT"
        )
        celestial_frac = (
            (celestial_coherent / len(celestial_harvests)) if celestial_harvests else None
        )
        bruno_ok = [
            c for c in celestial_harvests if c.celestial_kam_is_bruno is True
        ]
        bruno_frac = (
            (len(bruno_ok) / len(celestial_harvests)) if celestial_harvests else None
        )
        w = DensityOperatorAlgebra.spectrum_descending(self.mac_field.rho)
        aggregate_purity = float(np.sum(w ** 2))
        aggregate_entropy = float(-np.sum(w * np.log(np.maximum(w, _EPS))))
        aggregate_fidelity = DensityOperatorAlgebra.uhlmann_fidelity(
            self.mac_field.rho, self.ground_projector
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
        merkle = _merkle_root([c.content_hash for c in self.cultivated_crops_history])
        celestial_merkle = (
            _merkle_root([c.content_hash for c in celestial_harvests])
            if celestial_harvests
            else None
        )
        agg_celestial = None
        if celestial_harvests:
            acc = HeytingOmega3.COHERENT
            for c in celestial_harvests:
                acc = acc.meet(HeytingOmega3.from_name_safe(c.celestial_verdict))
            agg_celestial = acc.name
        idle = self.mac_field.idle_certificate()
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
            (agg_celestial or "n/a").encode("ascii"),
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
            celestial_coherent_fraction=celestial_frac,
            celestial_field_merkle_root=celestial_merkle,
            aggregate_celestial_verdict=agg_celestial,
            aggregate_bruno_ok_fraction=bruno_frac,
            mac_cptp_ok=bool(idle.is_cptp),
        )

    def process_seed_harvest_governance(
        self,
        seed_handoff: SeedHandoff,
        soil_state: SoilState,
    ) -> CropSovereignGovernancePassport:
        seed_crystal = getattr(seed_handoff, "seed_crystal", None)
        if seed_crystal is None:
            seed_crystal = SeedCrystal(
                crystal_id=seed_handoff.seed_id,
                origin_agent=seed_handoff.origin_agent,
                density_matrix=seed_handoff.rho_seed,
                experience_vector=np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float64),
                is_vacuum_pure=seed_handoff.seed_audit.verified_vacuum_pure,
            )
        _ = (
            soil_state.soil_field
            if hasattr(soil_state, "soil_field")
            else SoilField(self.dimension_mac)
        )
        harvest = self.sow_and_cultivate(
            seed_crystal=seed_crystal,
            toon_str=getattr(seed_crystal, "crystal_id", "TOON_SEED"),
            base_json_str=getattr(seed_crystal, "origin_agent", "ORIGIN_BASE_AGENT_SPECS"),
            external_verdict=HeytingOmega3.COHERENT,
        )
        verdict = harvest.heyting_verdict
        reason = harvest.crowbar_report.reason if harvest.crowbar_report.interlock_fired else "OK"
        if verdict == HeytingOmega3.VETOED:
            self.crowbar_interlock.trigger_esp32_hardware_veto(
                reason=f"KAM_INSTABILITY_OR_BIFURCATION: {reason}"
            )
        field_passport = self.audit_field_governance()
        return CropSovereignGovernancePassport(
            passport_id=field_passport.passport_id,
            sovereign_agent_id=self.agent_id,
            total_crops_cultivated=field_passport.total_crops_cultivated,
            active_coherent_crops=field_passport.active_coherent_crops,
            coherent_fraction=field_passport.coherent_fraction,
            aggregate_purity=field_passport.aggregate_purity,
            aggregate_entropy=field_passport.aggregate_entropy,
            aggregate_fidelity_to_ground=field_passport.aggregate_fidelity_to_ground,
            mac_spectral_gap=field_passport.mac_spectral_gap,
            global_heyting_verdict=verdict,
            crowbar_protection_active=(
                verdict == HeytingOmega3.VETOED or field_passport.crowbar_protection_active
            ),
            field_merkle_root=field_passport.field_merkle_root,
            phase_chain_sha256=field_passport.phase_chain_sha256,
            sha256_provenance=field_passport.sha256_provenance,
            timestamp_utc=time.time(),
            verdict=verdict,
            reason=reason,
            celestial_coherent_fraction=field_passport.celestial_coherent_fraction,
            celestial_field_merkle_root=field_passport.celestial_field_merkle_root,
            aggregate_celestial_verdict=field_passport.aggregate_celestial_verdict,
            aggregate_bruno_ok_fraction=field_passport.aggregate_bruno_ok_fraction,
            mac_cptp_ok=field_passport.mac_cptp_ok,
        )


# ── §3.5 Demostración autónoma ───────────────────────────────────────────
def _build_seed_matrix(
    n: int,
    alpha: float,
    key: str,
    align_with: Optional[np.ndarray] = None,
    ground_mix: float = 0.0,
) -> ComplexMatrix:
    rng = np.random.default_rng(_seed_from_string(key))
    idx = np.arange(n)
    raw = np.exp(alpha * (n - idx))
    w = raw / raw.sum()
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    rho = (Q * w.astype(np.complex128)) @ Q.conj().T
    rho = DensityOperatorAlgebra.sanitize(rho)
    mix = float(np.clip(ground_mix, 0.0, 1.0))
    if align_with is not None and mix > 0.0:
        g = DensityOperatorAlgebra.sanitize(align_with)
        rho = DensityOperatorAlgebra.sanitize((1.0 - mix) * rho + mix * g)
    return rho


def _make_seed(
    crystal_id: str,
    origin: str,
    rho: np.ndarray,
    vacuum_claim: bool,
) -> SeedCrystal:
    P = DensityOperatorAlgebra.purity(rho)
    S = DensityOperatorAlgebra.von_neumann_entropy(rho)
    n = int(rho.shape[0])
    S_max = math.log(n) if n > 1 else 1.0
    w = DensityOperatorAlgebra.spectrum_descending(rho)
    lam_ratio = float(w[-1] / (w[-1] + w[0] + 1e-30))
    experience = np.array([P, S / max(S_max, 1e-30), 1.0 if vacuum_claim else 0.5, lam_ratio])
    return SeedCrystal(
        crystal_id=crystal_id,
        origin_agent=origin,
        density_matrix=rho,
        experience_vector=experience,
        is_vacuum_pure=vacuum_claim,
    )


__all__ = [
    "HeytingOmega3",
    "DensityOperatorAlgebra",
    "BanachContractionReport",
    "BanachContractionAlgebra",
    "MaupertuisJacobiGerm",
    "PoincareCartanGerm",
    "PoincareCelestialGeometry",
    "SeedCrystal",
    "SeedAuditReport",
    "_PoincareCartanSeedGerm",
    "SeedCrystalSanitizer",
    "MacUpdateCertificate",
    "MacStateField",
    "SeedHandoff",
    "SeedCelestialHandoff",
    "WateringReport",
    "CognitiveWateringModule",
    "BrockettPurificationCertificate",
    "RenyiPurificationCertificate",
    "FockAnnihilationCertificate",
    "CognitiveIlluminationModule",
    "CognitiveDisciplineModule",
    "KAMAudit",
    "MelnikovAudit",
    "ReturnMapAudit",
    "PoincareCelestialVerifier",
    "CropGrowthBundle",
    "_PoincareCelestialGrowthBundle",
    "CropGrowthPipeline",
    "HeytingCropAdjudicator",
    "CrowbarActuationReport",
    "CognitiveFaithModule",
    "CropHarvestYield",
    "CropSovereignGovernancePassport",
    "TOONCognitiveCropAgent",
]


if __name__ == "__main__":
    print("═" * 88)
    print("TOON COGNITIVE CROP AGENT — v9.1.0 Poincare-Cartan-Melnikov-KAM-Bruno-CZ")
    print("Soberano MAC · Ω₃⁷⊕celeste · Crowbar BT151 · Merkle · Nekhoroshev")
    print("═" * 88)
    toon_str = "[APU: 2.1.4|CONCRETO 3000PSI] COST: 485000 COP"
    base_json_str = json.dumps(
        {
            "apu_code": "2.1.4-CONCRETO-3000PSI",
            "unit_cost": 485000.0,
            "metadata": {
                "schema": "fat_json_structure_with_verbose_keys",
                "authority": "Sovereign-APU-Crop-Agent",
                "redundant": "eliminable_by_toon_tabularization",
            },
        },
        indent=2,
    )
    agent = TOONCognitiveCropAgent(
        agent_id="CROP-SOVEREIGN-SABIO-01",
        dimension_mac=4,
        mac_gamma=0.2,
        eta_star=1.5,
        renyi_alpha=1.5,
        hamiltonian_energy_H0=1.0,
        potential_energy_V=0.0,
    )
    omega_freq = np.array([1.6180339887, 2.7182818285])
    wave_k = np.array([[1, 1], [1, -1], [2, 1]])
    jacobian_M = np.array(
        [
            [np.cos(0.3), -np.sin(0.3), 0.0, 0.0],
            [np.sin(0.3), np.cos(0.3), 0.0, 0.0],
            [0.0, 0.0, np.cos(0.5), -np.sin(0.5)],
            [0.0, 0.0, np.sin(0.5), np.cos(0.5)],
        ]
    )
    scenarios = [
        ("COHERENT-CELESTIAL", 0.8, "SEED-FOCUSED", False, 0.15),
        ("DEGRADED-CELESTIAL", 0.0, "SEED-UNIFORM", False, 0.0),
        ("FALSE-VACUUM", 0.2, "SEED-MIXED", True, 0.0),
    ]
    print("\n" + "─" * 88)
    for name, alpha, key, vac_claim, gmix in scenarios:
        rho = _build_seed_matrix(
            n=4,
            alpha=alpha,
            key=key,
            align_with=agent.ground_projector,
            ground_mix=gmix,
        )
        seed = _make_seed(f"CRYSTAL-{key}", "ORIGIN-WISDOM", rho, vac_claim)
        harvest = agent.sow_and_cultivate_celestial(
            seed_crystal=seed,
            toon_str=toon_str,
            base_json_str=base_json_str,
            external_verdict=HeytingOmega3.COHERENT,
            frequency_vector_omega=omega_freq,
            wave_vectors_k=wave_k,
            jacobian_M=jacobian_M,
        )
        print(f"\n[{name}]  α={alpha}  vacuum_claim={vac_claim}")
        print(f"   crop_id                 : {harvest.crop_id}")
        print(f"   Ω₃ final                : {harvest.heyting_verdict.name}")
        print(f"   Ω₃⁷⊕C canales           : {harvest.omega37_channels}")
        print(f"   Celestial / KAM / Bruno : {harvest.celestial_verdict} / "
              f"{harvest.celestial_kam_verdict} / {harvest.celestial_kam_is_bruno}")
        print(f"   Nekhoroshev T / CZ      : {harvest.celestial_nekhoroshev_time} / "
              f"{harvest.celestial_conley_zehnder}")
        print(f"   Melnikov / Return       : {harvest.celestial_melnikov_verdict} / "
              f"{harvest.celestial_return_verdict}")
        print(f"   n(q) / Hill / c_G       : {harvest.maupertuis_refractive_index} / "
              f"{harvest.maupertuis_hill_margin} / {harvest.gromov_capacity}")
        print(f"   ΔP / ΔS / inoculada     : {harvest.purity_gain:+.6f} / "
              f"{harvest.entropy_reduction:+.6f} / {harvest.is_inoculated_into_mac}")
        print(f"   ρ_seed / Lip_MAC        : {harvest.banach_seed_radius:.6f} / "
              f"{harvest.banach_mac_coef:.6f}")
        print(f"   Gelfand / Crowbar       : {harvest.gelfand_radius} / "
              f"{harvest.crowbar_report.interlock_fired}")
        print(f"   Firma SHA-256           : {harvest.sha256_provenance[:32]}…")
    passport = agent.audit_field_governance()
    print("\n" + "─" * 88)
    print(f"PASAPORTE {passport.passport_id}")
    print(f"   veredicto global        : {passport.global_heyting_verdict.name}")
    print(f"   coherente frac / celeste: {passport.coherent_fraction:.3f} / "
          f"{passport.celestial_coherent_fraction}")
    print(f"   Bruno frac / CPTP MAC   : {passport.aggregate_bruno_ok_fraction} / "
          f"{passport.mac_cptp_ok}")
    print(f"   Merkle campo            : {passport.field_merkle_root[:32]}…")
    print(f"   celestial agregado      : {passport.aggregate_celestial_verdict}")
    print("\n" + "═" * 88)
    print("✓ F1→F2: synthesize_poincare_cartan_germ ⊣ hand_off_germ_to_phase2.")
    print("✓ F2→F3: synthesize_celestial_bundle ⊣ hand_off_bundle_to_phase3.")
    print("✓ Ω₃⁷⊕celeste: 7 canales clásicos + 1 canal KAM∧Melnikov∧CZ.")
    print("✓ MAC CPTP: Φ_γ(ρ)=(1−γ)ρ+γρ★, Lip=|1−γ|, inoculación sólo si ⊤.")
    print("✓ KAM+Bruno+Nekhoroshev · Melnikov · Conley–Zehnder · Crowbar BT151.")
    print("═" * 88)
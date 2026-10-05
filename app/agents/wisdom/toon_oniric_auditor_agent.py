# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/agents/toon_oniric_auditor_agent.py                                   ║
║ ESTRATO  : WISDOM (V_𝕎) — CIUDADELA DE CRISTAL / AUDITORÍA ONÍRICA TQFT              ║
║ FUNCIÓN  : SOBERANO AUDITOR DE ESCENARIOS ONÍRICOS Y CERTIFICADOR DE INMUNIZACIÓN    ║
║ VERSIÓN  : 9.1.0-Doctoral-Poincare-Celeste-KAM-Nekhoroshev-Hannay-ESP32              ║
║ AUTOR    : APU Wisdom & Metacortex Mathematical Core Architecture                    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICA POINCARÉ
─────────────────────────────────────────────────────────
El `OniricDreamAuditorAgent` es la autoridad soberana responsable de auditar,
certificar e inmunizar el ecosistema agéntico frente a escenarios de estrés
contrafactual y ataques adversariales procesados en la Fase REM.

Su acción se formaliza como un endofuntor de gobernanza:

    𝒜 : 𝐒𝐜𝐞𝐧𝐚𝐫𝐢𝐨  ──▶  𝐂𝐞𝐫𝐭_𝐈𝐧𝐦𝐮𝐧𝐞
    𝒜 = Seal ∘ Crowbar ∘ V ∘ Isol ∘ I_GW ∘ Spec

sobre el topos 𝓣_Ω (Ω₃ de Heyting) y la variedad simpléctica coadjunta 𝔲(n)*
dotada de la 2-forma de Kirillov–Kostant–Souriau y de las coordenadas
acción-ángulo de Liouville–Arnold.

CORRECCIÓN CELESTE CANÓNICA (Poincaré 1890, 1892, 1899)
───────────────────────────────────────────────────────
El flujo de Brockett \(\dot\rho=[\rho,[\rho,N]]\) es un flujo *gradiente* de
\(\operatorname{Tr}(\rho N)\): no preserva Liouville y no admite recurrencia.
El flujo celeste es el de von Neumann

    \(\dot\rho=-i[N,\rho]\),

hamiltoniano respecto de \(\omega_{\mathrm{KKS}}\). Si \(N=\operatorname{diag}(\nu)\),

    \(\rho_{jk}(t)=\rho_{jk}(0)\,e^{-i(\nu_j-\nu_k)t}\).

\(\operatorname{Tr}(\rho N)\) es integral primera. La sección de Poincaré es un
corte angular transversal *sobre* la superficie de energía, no la superficie
misma. KAM, Nekhoroshev, Greene, Chirikov, Brjuno y Hannay se aplican a este flujo.

ARQUITECTURA EN TRES FASES ANIDADAS (Composición Estricta de Funtores)
──────────────────────────────────────────────────────────────────────
Fase 1 ──► Ω₃, DENSIDAD, MEDIDA, SECCIÓN, RESONANCIA, PAYLOAD, CERTIFICADOS
           Cierra con OniricAuditSeed.extract_spectral_measure.
Fase 2 ──► GW/TQFT, LEFSCHETZ, VON NEUMANN, KAM, MELNIKOV, GUARDIÁN
           Abre  con GromovWittenOniricAuditor.extract_spectral_measure.
           Cierra con UnsealedOniricAuditTrace (germen de Fase 3).
Fase 3 ──► SOBERANO, SELLO DUAL, HOLONOMÍA BERRY–HANNAY, MERKLE, PASAPORTE
           Abre  con OniricDreamAuditorAgent._seal_and_classify
           (único consumidor formal de UnsealedOniricAuditTrace).
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONOniricAuditor.v91")

_WILKINSON: Final[float] = 16.0 * float(np.finfo(np.float64).eps)
_SPECTRAL_TOL: Final[float] = 1e-9
_EPS: Final[float] = 1e-15
_EIG_FLOOR: Final[float] = 1e-15
_TRACE_TOL: Final[float] = 1e-6
_MESH_CAP: Final[int] = 80_000

__all__ = [
    "HeytingOmega3",
    "DensityOperator",
    "PoincareCelestialElements",
    "FloquetSpectrum",
    "SpectralMeasure",
    "PoincareSectionOniric",
    "ResonanceStructure",
    "OniricAuditSeed",
    "OniricScenarioPayload",
    "OniricIsolationCertificate",
    "ImmunizationCertificate",
    "IsolationAuditor",
    "GWInvariantEvaluator",
    "GromovWittenOniricAuditor",
    "OniricIsolationGuard",
    "UnsealedOniricAuditTrace",
    "OniricAuditArrowComposer",
    "MerkleInclusionProof",
    "OniricDreamAuditorAgent",
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES ALGEBRAICAS (redes de Poincaré, fracciones continuas, Hermiticidad)
# ══════════════════════════════════════════════════════════════════════════════

def _hermitize_trace_one(rho: np.ndarray) -> np.ndarray:
    r"""Proyección ortogonal al cono \(\mathfrak{D}(\mathcal{H}_n)\)."""
    raw = np.asarray(rho, dtype=np.complex128)
    rho_h = 0.5 * (raw + raw.conj().T)
    evals, evecs = la.eigh(rho_h)
    evals = np.clip(evals, 0.0, None)
    s = float(np.sum(evals))
    if s <= _EIG_FLOOR:
        n = rho_h.shape[0]
        return np.eye(n, dtype=np.complex128) / n
    evals = evals / s
    rho_h = (evecs * evals) @ evecs.conj().T
    return 0.5 * (rho_h + rho_h.conj().T)


def _diag_potential(n: int, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Potencial cartaniano \(N=\operatorname{diag}(1,\dots,n)\) o diagonal extraída."""
    if N_diag is None:
        return np.arange(1, n + 1, dtype=float)
    arr = np.asarray(N_diag, dtype=float)
    if arr.ndim == 2:
        return np.real(np.diag(arr))
    return arr.reshape(-1)


def _iterate_integer_lattice(n: int, order: int) -> List[np.ndarray]:
    r"""
    Genera \(\{k\in\mathbb{Z}^n:\|k\|_1=\textit{order}\}\) con signos completos.

    Corrección: \(n=1\) produce \(\{\pm\textit{order}\}\) y el residuo nulo
    produce el vector cero, de modo que \((2,0)\) y \((-2,0)\) existen.
    """
    if n < 1 or order < 0:
        return []
    if n == 1:
        if order == 0:
            return [np.array([0], dtype=int)]
        return [np.array([order], dtype=int), np.array([-order], dtype=int)]
    vectors: List[np.ndarray] = []
    for k1 in range(-order, order + 1):
        rest = order - abs(k1)
        for tail in _iterate_integer_lattice(n - 1, rest):
            vectors.append(np.concatenate(([k1], tail)))
    return vectors


def _iter_inf_lattice(n: int, k_max: int) -> Iterator[Tuple[int, ...]]:
    r"""Itera \(k\in\{-k_{\max},\dots,k_{\max}\}^n\setminus\{0\}\)."""
    if n < 1 or k_max < 1:
        return
    stack: List[Tuple[int, ...]] = [()]
    while stack:
        prefix = stack.pop()
        if len(prefix) == n:
            if any(prefix):
                yield prefix
            continue
        for ki in range(-k_max, k_max + 1):
            stack.append(prefix + (ki,))


def _continued_fraction(x: float, max_terms: int = 16, tol: float = 1e-14) -> List[int]:
    r"""Fracción continua \([a_0;a_1,\dots]\) (algoritmo de Gauss)."""
    a: List[int] = []
    if not math.isfinite(x):
        return [0]
    y = float(x)
    for _ in range(max_terms):
        ai = int(math.floor(y))
        a.append(ai)
        frac = y - ai
        if abs(frac) < tol:
            break
        y = 1.0 / frac
        if abs(y) > 1e12:
            break
    return a


def _convergents(cf: Sequence[int]) -> List[Tuple[int, int]]:
    r"""Convergentes \(p_k/q_k\) de una fracción continua."""
    p_m2, q_m2 = 0, 1
    p_m1, q_m1 = 1, 0
    out: List[Tuple[int, int]] = []
    for a in cf:
        p = a * p_m1 + p_m2
        q = a * q_m1 + q_m2
        out.append((p, q))
        p_m2, q_m2 = p_m1, q_m1
        p_m1, q_m1 = p, q
    return out


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-30:
        return default
    return num / den


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Ω₃, DENSIDAD, MEDIDA ESPECTRAL, SECCIÓN, RESONANCIA, PAYLOAD,
#          CERTIFICADOS Y SEMILLA Spec
# ══════════════════════════════════════════════════════════════════════════════
# Andamiaje ontológico del topos 𝓣_Ω y del fibrado simpléctico 𝔇(ℋ_n) → S²_Poincaré.
# FASE-1 cierra con OniricAuditSeed.extract_spectral_measure, cuyo primer
# consumidor real (y por tanto continuación formal) es GromovWittenOniricAuditor
# en FASE-2.
# ══════════════════════════════════════════════════════════════════════════════

# ── §1.1 Retículo de Heyting Ω₃ con estratificación celeste ────────────────
class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal \(\Omega_3=\{0\prec 1\prec 2\}=\{\mathsf{VETOED}\prec\mathsf{DEGRADED}\prec\mathsf{COHERENT}\}\).

    Operaciones: \(a\wedge b=\min(a,b)\), \(a\vee b=\max(a,b)\),
    \(a\to b=\top\) si \(a\le b\) else \(b\); \(\neg_H a=a\to\bot\).

    Estratificación celeste de Poincaré / Lyapunov / Floquet
    --------------------------------------------------------
        VETOED   ⇔ separatriz hiperbólica (Melnikov, Greene \(|R|>1\)), \(\sigma_+>0\).
        DEGRADED ⇔ resonancia \(p/q\) de orden bajo, \(\sigma\approx 0\), \(|R|\approx 1\).
        COHERENT ⇔ toro KAM diofantino, \(\sigma_-<0\), \(|R|<1\), Nekhoroshev largo.
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Meet: \(a\wedge b=\min(a,b)\)."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Join: \(a\vee b=\max(a,b)\)."""
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Residuo \(a\to b=\bigvee\{c:a\wedge c\le b\}\)."""
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def pseudo_complement(self) -> "HeytingOmega3":
        r"""\(\neg_H a:=a\to\bot\)."""
        return self.implies(HeytingOmega3.VETOED)

    def classical_negation(self) -> "HeytingOmega3":
        r"""Involución booleana \(2-a\) (no interna)."""
        return HeytingOmega3(2 - int(self))

    def double_negation(self) -> "HeytingOmega3":
        r"""\(\neg\neg_H a\) (no idempotente sobre DEGRADED)."""
        return self.pseudo_complement().pseudo_complement()

    def is_regular(self) -> bool:
        r"""¿\(\neg\neg a=a\)? Verdadero en \(\{\bot,\top\}\)."""
        return self.double_negation() == self

    def excluded_middle_holds(self) -> bool:
        r"""\(a\vee\neg a=\top\) ⇔ \(a\in\{\bot,\top\}\). Falla en DEGRADED."""
        return self.join(self.pseudo_complement()) == HeytingOmega3.COHERENT

    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        r"""Inclusión \(\mathbb{B}_2\hookrightarrow\Omega_3\)."""
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        r"""Proyección parcial \(\Omega_3\rightharpoonup\mathbb{B}_2\)."""
        if self == HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED no admite proyección fiel a 𝔹₂.")
        return self == HeytingOmega3.COHERENT

    @property
    def verdict(self) -> str:
        return self.name

    @classmethod
    def verify_residuation_axiom(cls) -> bool:
        r"""Verifica \((c\wedge a\le b)\Leftrightarrow(c\le(a\to b))\) \(\forall a,b,c\)."""
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

    def poincare_stratum_name(self) -> str:
        r"""Nombre del estrato en el lenguaje de la mecánica celeste de Poincaré."""
        return {
            HeytingOmega3.VETOED: "hyperbolic-escape-separatrix",
            HeytingOmega3.DEGRADED: "parabolic-resonant-orbit",
            HeytingOmega3.COHERENT: "kam-torus-invariant",
        }[self]

    def is_kam_stratum(self) -> bool:
        return self == HeytingOmega3.COHERENT

    def lyapunov_exponent_sign(self) -> int:
        r"""Signo del exponente de Lyapunov máximo: \(+1,0,-1\)."""
        return -1 + int(self)

    def symplectic_regime(self) -> str:
        return {
            HeytingOmega3.VETOED: "symplectic-escape",
            HeytingOmega3.DEGRADED: "resonant-cascade",
            HeytingOmega3.COHERENT: "integrable-kam",
        }[self]

    def floquet_stability(self) -> str:
        r"""Clasificación de Floquet–Poincaré del multiplicador dominante."""
        return {
            HeytingOmega3.VETOED: "hyperbolic-unstable",
            HeytingOmega3.DEGRADED: "parabolic-critical",
            HeytingOmega3.COHERENT: "elliptic-stable",
        }[self]

    def greene_residue_regime(self) -> str:
        r"""Régimen del residuo de Greene \(R=(2-\operatorname{Tr} M)/4\)."""
        return {
            HeytingOmega3.VETOED: "|R|>1 hyperbolic",
            HeytingOmega3.DEGRADED: "|R|~1 bifurcation",
            HeytingOmega3.COHERENT: "|R|<1 elliptic",
        }[self]


# ── §1.2 Elementos celestes de Poincaré (Delaunay → Poincaré → equinocciales)
@dataclass(frozen=True, slots=True)
class PoincareCelestialElements:
    r"""
    Carta canónica de Poincaré sobre la órbita coadjunta.

    Delaunay \((L,G,H,l,g,h)\) con \(\mu=1\):
        \(L=\sqrt{a}\), \(G=L\sqrt{1-e^2}\), \(H=G\cos i\).

    Variables de Poincaré (regulares en \(e=0\), \(i=0\)):
        \(\Lambda=L\), \(\lambda=l+g+h\),
        \(\xi=\sqrt{2(L-G)}\cos\varpi\), \(\eta=-\sqrt{2(L-G)}\sin\varpi\),
        \(p=\sqrt{2(G-H)}\cos h\), \(q=-\sqrt{2(G-H)}\sin h\).

    Tisserand (respecto a \(a_p=1\)):
        \(T=a_p/a+2\sqrt{a/a_p\,(1-e^2)}\cos i\).
    Jacobi espectral: \(C_J=2\operatorname{Tr}(\rho N)-\gamma(\rho)\).
    """

    L: float
    G: float
    H: float
    mean_anomaly: float
    arg_periapsis: float
    long_node: float
    Lambda: float
    mean_longitude: float
    xi: float
    eta: float
    p: float
    q: float
    eccentricity: float
    inclination: float
    tisserand: float
    jacobi_integral: float
    energy: float
    mean_motion: float


# ── §1.3 Espectro de Floquet / residuo de Greene ──────────────────────────
@dataclass(frozen=True, slots=True)
class FloquetSpectrum:
    r"""
    Multiplicadores de Floquet de la monodromía \(M\) y residuo de Greene
    \(R=(2-\operatorname{Tr} M)/4\). \(|R|<1\) elíptica, \(|R|=1\) parabólica,
    \(|R|>1\) hiperbólica.
    """

    multipliers: np.ndarray
    spectral_radius: float
    greene_residue: float
    trace_monodromy: float
    is_elliptic: bool
    is_symplectic: bool


# ── §1.4 Operador de densidad con estructura simpléctica y Wigner ──────────
@dataclass(frozen=True, slots=True)
class DensityOperator:
    r"""
    Estado cuántico \(\rho\in\mathfrak{D}(\mathcal{H}_n)\subset B(\mathcal{H}_n)\).

    Invariantes: \(\rho=\rho^\dagger\), \(\operatorname{spec}(\rho)\subset[-\varepsilon,1+\varepsilon]\),
    \(|\operatorname{Tr}\rho-1|\le 10^{-6}\).

    Flujo hamiltoniano exacto: si \(N=\operatorname{diag}(\nu)\),
    \(\rho_{jk}(t)=\rho_{jk}(0)e^{-i(\nu_j-\nu_k)t}\). Preserva \(\omega_{\mathrm{KKS}}\)
    y el volumen de Liouville (invariante integral absoluto de Poincaré).
    """

    matrix: np.ndarray
    atol: float = 1e-8

    def __post_init__(self) -> None:
        rho = np.asarray(self.matrix, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("DensityOperator exige matriz cuadrada.")
        object.__setattr__(self, "matrix", np.array(rho, copy=True))
        if not np.allclose(rho, rho.conj().T, atol=self.atol):
            raise ValueError("DensityOperator exige ρ = ρ†.")
        tr = float(np.trace(rho).real)
        if abs(tr - 1.0) > _TRACE_TOL:
            raise ValueError(f"DensityOperator exige Tr ρ = 1 (Tr={tr}).")

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def as_array(self) -> np.ndarray:
        return self.matrix

    def cstar_residual(self) -> float:
        r"""Residuo \(C^*\): \(\big|\|\rho^\dagger\rho\|_2-\|\rho\|_2^2\big|\)."""
        rho = self.matrix
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

    def spectrum(self, floor: float = _EIG_FLOOR) -> np.ndarray:
        r"""Autovalores normalizados a suma 1."""
        lam = la.eigvalsh(self.matrix)
        lam = np.clip(np.real(lam), floor, None)
        s = float(np.sum(lam))
        return lam / s if s > 0.0 else np.full(lam.shape, 1.0 / lam.size)

    def purity(self) -> float:
        r"""\(\gamma(\rho)=\operatorname{Tr}(\rho^2)\in[1/n,1]\)."""
        lam = self.spectrum()
        return float(np.sum(lam ** 2))

    def von_neumann_entropy(self) -> float:
        r"""\(S(\rho)=-\operatorname{Tr}(\rho\log\rho)\) (nats)."""
        lam = self.spectrum()
        return -float(np.sum(lam * np.log(lam)))

    def spectral_gap(self) -> float:
        r"""\(\Delta\lambda=\lambda_{\max}-\lambda_{\max-1}\)."""
        lam = np.sort(self.spectrum())
        if lam.size < 2:
            return 0.0
        return float(lam[-1] - lam[-2])

    @classmethod
    def from_array(cls, rho: np.ndarray, atol: float = 1e-8) -> "DensityOperator":
        r"""Proyección al cono \(\mathfrak{D}(\mathcal{H}_n)\) y envoltura inmutable."""
        return cls(matrix=_hermitize_trace_one(rho), atol=atol)

    def action_variables(self) -> np.ndarray:
        r"""Acciones de Liouville \(J_i:=\lambda_i(\rho)\) decrecientes."""
        return np.sort(self.spectrum())[::-1]

    def angle_variables(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Ángulos canónicos \(\theta_i:=\arg\langle u_i|N|u_i\rangle\in(-\pi,\pi]\)."""
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        evals, evecs = la.eigh(self.matrix)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        ph = np.zeros(n, dtype=float)
        for i in range(n):
            u = evecs[:, i]
            z = np.vdot(u, N @ u)
            ph[i] = float(np.angle(z))
        return ph

    def mean_motion_frequencies(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Frecuencias metabólicas \(\omega_i:=J_i\cdot\operatorname{Tr}(\rho N)\)."""
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        actions = self.action_variables()
        base = float(np.trace(self.matrix @ N).real)
        return actions * base

    def keplerian_frequency_matrix(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Matriz de frecuencias de Poincaré \(\omega_{jk}=\nu_j-\nu_k\)."""
        nu = _diag_potential(self.dimension, N_diag)
        return nu[:, None] - nu[None, :]

    def delaunay_triple(self, N_diag: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        r"""
        Tripleta de Delaunay \((L,G,H)\):
            \(L=\sqrt{\max(\operatorname{Tr}(\rho N),0)}\),
            \(G=\sqrt{\max(\operatorname{Tr}(\rho N)^2-\|[ \rho,N]\|_F^2,0)}\),
            \(H=\operatorname{Tr}(\rho N)\cdot\cos\Delta\lambda\).
        """
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        comm = self.matrix @ N - N @ self.matrix
        trace_rN = float(np.trace(self.matrix @ N).real)
        comm_norm_sq = float(np.trace(comm @ comm.conj().T).real)
        L = math.sqrt(max(trace_rN, 0.0))
        G = math.sqrt(max(trace_rN ** 2 - comm_norm_sq, 0.0))
        H = trace_rN * math.cos(self.spectral_gap())
        return float(L), float(G), float(H)

    def eccentricity_inclination(
        self, N_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        r"""Excentricidad \(e=\sqrt{1-G^2/L^2}\) e inclinación \(i=\arccos(H/G)\)."""
        L, G, H = self.delaunay_triple(N_diag)
        e = math.sqrt(max(1.0 - (G ** 2) / max(L ** 2, _EPS), 0.0))
        i = (
            math.acos(max(min(_safe_div(H, G, 1.0), 1.0), -1.0))
            if G > _EPS
            else 0.0
        )
        return float(e), float(i)

    def celestial_elements(
        self, N_diag: Optional[np.ndarray] = None
    ) -> PoincareCelestialElements:
        r"""Carta completa Delaunay → Poincaré, Tisserand y Jacobi."""
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        L, G, H = self.delaunay_triple(N)
        e, inc = self.eccentricity_inclination(N)
        th = self.angle_variables(N)
        l = float(th[0]) if n >= 1 else 0.0
        g = float(th[1] - th[0]) if n >= 2 else 0.0
        h = float(th[2] - th[1]) if n >= 3 else 0.0
        varpi = g + h
        rt_e = math.sqrt(max(2.0 * (L - G), 0.0))
        rt_i = math.sqrt(max(2.0 * (G - H), 0.0))
        energy = float(np.trace(self.matrix @ N).real)
        a = max(L * L, 1e-30)
        a_p = 1.0
        tisserand = a_p / a + 2.0 * math.sqrt(max(a / a_p * (1.0 - e * e), 0.0)) * math.cos(inc)
        jacobi = 2.0 * energy - self.purity()
        mean_motion = _safe_div(1.0, a ** 1.5, 0.0)
        return PoincareCelestialElements(
            L=float(L),
            G=float(G),
            H=float(H),
            mean_anomaly=l,
            arg_periapsis=g,
            long_node=h,
            Lambda=float(L),
            mean_longitude=l + g + h,
            xi=float(rt_e * math.cos(varpi)),
            eta=float(-rt_e * math.sin(varpi)),
            p=float(rt_i * math.cos(h)),
            q=float(-rt_i * math.sin(h)),
            eccentricity=float(e),
            inclination=float(inc),
            tisserand=float(tisserand),
            jacobi_integral=float(jacobi),
            energy=float(energy),
            mean_motion=float(mean_motion),
        )

    def poincare_integral_invariant(self) -> float:
        r"""Invariante integral relativo de Poincaré \(\sum_i J_i\theta_i\)."""
        return float(np.dot(self.action_variables(), self.angle_variables()))

    def hamiltonian_evolve(self, t: float, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Flujo de von Neumann exacto \(\rho(t)=e^{-iNt}\rho e^{iNt}\)."""
        nu = _diag_potential(self.dimension, N_diag)
        phase = np.exp(-1j * float(t) * (nu[:, None] - nu[None, :]))
        rho_t = self.matrix * phase
        return 0.5 * (rho_t + rho_t.conj().T)

    def first_return_phase_time(
        self, i: int = 0, j: int = 1, N_diag: Optional[np.ndarray] = None
    ) -> float:
        r"""Periodo kepleriano del par: \(T=2\pi/|\nu_i-\nu_j|\)."""
        nu = _diag_potential(self.dimension, N_diag)
        n = self.dimension
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return float("inf")
        omega = float(nu[i] - nu[j])
        if abs(omega) < 1e-15:
            return float("inf")
        return float(2.0 * math.pi / abs(omega))

    def wigner_function(self) -> np.ndarray:
        r"""
        Wigner discreto vía FFT:
            \(W_\rho(q,p)=\frac1n\sum_x e^{-2\pi i px/n}\rho_{q+x,\,q-x}\).
        """
        n = self.dimension
        rho = self.matrix
        W = np.empty((n, n), dtype=np.complex128)
        idx = np.arange(n)
        for q in range(n):
            disp = rho[(q + idx) % n, (q - idx) % n]
            W[q, :] = np.fft.fft(disp) / n
        return np.real(W)

    def husimi_function(self, sigma: float = 0.5) -> np.ndarray:
        r"""Husimi \(Q_\rho=\mathcal{G}_\sigma * W_\rho\) (kernel gaussiano periódico)."""
        W = self.wigner_function()
        n = self.dimension
        ax = np.arange(n)
        dx = np.minimum(ax, n - ax)
        g = np.exp(-0.5 * (dx ** 2) / max(sigma * sigma, 1e-12))
        g = g / np.sum(g)
        G = np.outer(g, g)
        Q = np.real(np.fft.ifft2(np.fft.fft2(W) * np.fft.fft2(np.fft.ifftshift(G))))
        s = float(np.sum(Q))
        return Q / s if abs(s) > 1e-15 else Q

    def wigner_marginals(self) -> Tuple[np.ndarray, np.ndarray]:
        W = self.wigner_function()
        return np.sum(W, axis=1), np.sum(W, axis=0)

    def wigner_negativity(self) -> float:
        r"""Negatividad de Wigner \(N_W(\rho)=\sum|W|-1\)."""
        W = self.wigner_function()
        return float(np.sum(np.abs(W)) - 1.0)

    def symplectic_capacity_gromov(self) -> float:
        r"""Capacidad de Gromov heurística \(c_G(\rho)=4/(\mathrm{Var}_q+\mathrm{Var}_p)\)."""
        q_marg, p_marg = self.wigner_marginals()
        n = self.dimension
        idx = np.arange(n, dtype=float)
        mean_q = float(np.dot(idx, q_marg))
        mean_p = float(np.dot(idx, p_marg))
        var_q = float(np.dot((idx - mean_q) ** 2, q_marg))
        var_p = float(np.dot((idx - mean_p) ** 2, p_marg))
        if var_q + var_p < 1e-12:
            return float("inf")
        return float(4.0 / (var_q + var_p))

    def symplectic_capacity_ratio(
        self, rho_base: Optional["DensityOperator"] = None
    ) -> float:
        r"""Ratio \(c_G(\rho)/c_G(\rho_{\mathrm{base}})\). \(>1.25\) ⇒ violación de rigidez."""
        c_rho = self.symplectic_capacity_gromov()
        if rho_base is None:
            return 1.0
        c_base = rho_base.symplectic_capacity_gromov()
        if c_base <= 1e-12:
            return float("inf")
        return float(c_rho / c_base)

    def maslov_index(self, N_diag: Optional[np.ndarray] = None) -> int:
        r"""Índice de Maslov (cruces de fase proyectiva respecto de \(N\))."""
        n = self.dimension
        N = np.diag(_diag_potential(n, N_diag))
        _, evecs = la.eigh(self.matrix)
        count = 0
        for k in range(n):
            u = evecs[:, k]
            overlap = abs(np.vdot(u, N @ u))
            if overlap > 1e-12:
                ph = float(np.angle(np.vdot(u, N @ u)))
                if ph > math.pi / 2.0 or ph < -math.pi / 2.0:
                    count += 1
        return int(count)

    def poincare_recurrence_time(self, N_diag: Optional[np.ndarray] = None) -> float:
        r"""
        Recurrencia de Poincaré: mínimo finito entre
        \(\tau_{\mathrm{spec}}=1/\min|\lambda_i-\lambda_j|\),
        \(\tau_{\mathrm{Kac}}=1/\prod\lambda_i\) y
        \(T_{\mathrm{Kep}}=2\pi/\min|\nu_j-\nu_k|\).
        """
        lam = np.sort(self.spectrum())
        tau_spec = float("inf")
        if lam.size >= 2:
            min_gap = float(np.min(np.abs(np.diff(lam))))
            if min_gap >= _EPS:
                tau_spec = 1.0 / min_gap
        vol = float(np.prod(np.clip(lam, 1e-30, None)))
        tau_kac = 1.0 / vol if vol > 0.0 else float("inf")
        nu = _diag_potential(self.dimension, N_diag)
        diffs = np.abs(nu[:, None] - nu[None, :])
        np.fill_diagonal(diffs, np.inf)
        min_nu = float(np.min(diffs))
        tau_kep = (2.0 * math.pi / min_nu) if min_nu < 1e30 and min_nu > 1e-15 else float("inf")
        return float(min(tau_spec, tau_kac, tau_kep))

    def berry_phase_curve(self, curve: Sequence[np.ndarray], closed: bool = True) -> float:
        r"""Fase de Berry–Pancharatnam a lo largo de \(\gamma=(\rho_0,\dots,\rho_m)\)."""
        if len(curve) < 2:
            return 0.0
        phis: List[np.ndarray] = []
        for rho_k in curve:
            rho_h = 0.5 * (rho_k + rho_k.conj().T)
            _, evecs = la.eigh(rho_h)
            phis.append(evecs[:, -1])
        total_arg = 0.0
        nphi = len(phis)
        if closed:
            for k in range(nphi):
                total_arg += float(np.angle(np.vdot(phis[k], phis[(k + 1) % nphi])))
        else:
            for k in range(nphi - 1):
                total_arg += float(np.angle(np.vdot(phis[k], phis[k + 1])))
        return float(total_arg)

    def hannay_angle(self, N_diag: Optional[np.ndarray] = None) -> float:
        r"""Ángulo de Hannay reducido al invariante relativo módulo \(2\pi\)."""
        I = self.poincare_integral_invariant()
        return float((I + math.pi) % (2.0 * math.pi) - math.pi)

    def chirikov_overlap(self, N_diag: Optional[np.ndarray] = None) -> float:
        r"""Parámetro de solapamiento de Chirikov \(s\). \(s\gtrsim 1\) ⇒ caos."""
        n = self.dimension
        if n < 2:
            return 0.0
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
        comm = self.matrix @ N - N @ self.matrix
        eps = float(la.norm(comm, "fro"))
        J = self.action_variables()
        omega = nu * float(np.trace(self.matrix @ N).real)
        order = np.argsort(omega)
        s_max = 0.0
        for a, b in zip(order[:-1], order[1:]):
            dw = abs(float(omega[b] - omega[a]))
            half = math.sqrt(max(abs(J[min(a, n - 1)] * eps), 0.0)) + math.sqrt(
                max(abs(J[min(b, n - 1)] * eps), 0.0)
            )
            s = _safe_div(half, dw, 0.0)
            if s > s_max:
                s_max = s
        return float(s_max)

    def siegel_brjuno_sum(self, N_diag: Optional[np.ndarray] = None, max_cf: int = 12) -> float:
        r"""Suma de Brjuno \(B(\alpha)=\sum(\log q_{k+1})/q_k\) del cociente \(\omega_1/\omega_0\)."""
        n = self.dimension
        if n < 2:
            return 0.0
        nu = _diag_potential(n, N_diag)
        if abs(nu[0]) < 1e-15:
            return float("inf")
        alpha = float(nu[1] / nu[0])
        conv = _convergents(_continued_fraction(alpha, max_terms=max_cf))
        if len(conv) < 2:
            return 0.0
        acc = 0.0
        for k in range(len(conv) - 1):
            qk = max(abs(conv[k][1]), 1)
            qk1 = max(abs(conv[k + 1][1]), 1)
            acc += math.log(qk1) / qk
        return float(acc)


# ── §1.5 Medida espectral con estructura Liouvilliana ─────────────────────
@dataclass(frozen=True, slots=True)
class SpectralMeasure:
    r"""
    Medida espectral \(\lambda\in\Delta^{n-1}\) con observables derivados.
    Volumen de Liouville \(\mathrm{Vol}_L(\lambda)=\prod_i\lambda_i\).
    """

    eigenvalues: np.ndarray
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    dirichlet_energy: float
    dirac_total_variation: float
    dimension: int
    cstar_residual: float

    def __post_init__(self) -> None:
        lam = np.asarray(self.eigenvalues, dtype=np.float64).reshape(-1)
        if lam.size < 1:
            raise ValueError("SpectralMeasure exige al menos un autovalor.")
        object.__setattr__(self, "eigenvalues", lam)

    def as_simplex(self) -> np.ndarray:
        return self.eigenvalues

    def liouville_volume(self) -> float:
        r"""Volumen de Liouville discreto \(\prod\lambda_i\)."""
        lam = np.clip(self.eigenvalues, 1e-30, None)
        return float(np.prod(lam))

    def fisher_rao_norm(self, tangent: np.ndarray) -> float:
        r"""Norma Fisher–Rao \(\|v\|_g^2=\sum_i v_i^2/\lambda_i\)."""
        lam = np.clip(self.eigenvalues, 1e-30, None)
        v = np.asarray(tangent, dtype=float).reshape(-1)
        if v.size != lam.size:
            raise ValueError("tangent incompatible con la medida.")
        return float(np.sqrt(np.sum((v * v) / lam)))

    def poincare_recurrence_time(self) -> float:
        r"""\(\tau_{\mathrm{rec}}=1/\min_{i\neq j}|\lambda_i-\lambda_j|\)."""
        lam = np.sort(self.eigenvalues)
        if lam.size < 2:
            return float("inf")
        min_gap = float(np.min(np.abs(np.diff(lam))))
        if min_gap < _EPS:
            return float("inf")
        return 1.0 / min_gap

    def kac_recurrence_time(self) -> float:
        r"""Tiempo medio de Kac \(\tau_{\mathrm{Kac}}=1/\mathrm{Vol}_L(\lambda)\)."""
        vol = self.liouville_volume()
        if vol <= 0.0:
            return float("inf")
        return 1.0 / vol

    def resonance_order(self, k_max: int = 3) -> int:
        r"""Orden resonante mínimo \(r^*=\min\{\|k\|_1:|\langle k,\lambda\rangle|<\varepsilon\}\)."""
        lam = self.eigenvalues
        n = lam.size
        for order in range(1, k_max + 1):
            for k in _iterate_integer_lattice(n, order):
                if float(abs(np.dot(k, lam))) < 1e-6:
                    return order
        return 0


# ── §1.6 Sección de Poincaré onírica y estructura de resonancias ──────────
@dataclass(frozen=True, slots=True)
class PoincareSectionOniric:
    r"""
    Sección de Poincaré *transversal* al flujo de von Neumann, contenida en
    la superficie de energía \(M_c=\{\operatorname{Tr}(\rho N)=c\}\).

    \(M_c\) es invariante. El corte \(\Sigma=\{\rho\in M_c:\arg\rho_{ij}=\phi_*\}\)
    es transversal ssi \(\nu_i\neq\nu_j\). El mapa de primer retorno
    \(P:\Sigma\to\Sigma\) es un twist que preserva el área (Poincaré–Cartan).
    """

    level: float
    normal_diag: np.ndarray
    transversality_tol: float = 1e-7
    i_cut: int = 0
    j_cut: int = 1
    phase_cut: float = 0.0

    def energy(self, rho: np.ndarray) -> float:
        r"""Hamiltoniano \(H(\rho)=\operatorname{Tr}(\rho N)\)."""
        return float(np.trace(rho @ self.normal_diag).real)

    def signed_distance(self, rho: np.ndarray) -> float:
        r"""Distancia con signo al hiperplano de energía (diagnóstico, *no* sección)."""
        return self.energy(rho) - float(self.level)

    def phase(self, rho: np.ndarray) -> float:
        r"""Ángulo \(\arg\rho_{ij}\) del corte transversal."""
        return float(np.angle(rho[self.i_cut, self.j_cut]))

    def signed_phase(self, rho: np.ndarray) -> float:
        r"""Fase reducida a \((-\pi,\pi]\) relativa a `phase_cut`."""
        d = self.phase(rho) - self.phase_cut
        return float((d + math.pi) % (2.0 * math.pi) - math.pi)

    def crosses(self, rho_before: np.ndarray, rho_after: np.ndarray) -> bool:
        r"""¿El segmento cruza el corte de fase transversalmente?"""
        return (self.signed_phase(rho_before) * self.signed_phase(rho_after)) < 0.0

    def is_transversal(self, rho: np.ndarray, rhodot: np.ndarray) -> bool:
        r"""Transversalidad: \(\operatorname{Im}(\overline{\rho_{ij}}(\dot\rho)_{ij})\neq 0\)."""
        z = rho[self.i_cut, self.j_cut]
        zdot = rhodot[self.i_cut, self.j_cut]
        g = float(np.imag(np.conj(z) * zdot))
        return abs(g) > self.transversality_tol

    def twist_condition(self, omega: np.ndarray) -> bool:
        r"""Condición de twist de Poincaré–Birkhoff / Moser: \(\mathrm{std}(\omega)>0\)."""
        return float(np.std(np.asarray(omega, dtype=float))) > self.transversality_tol

    def greene_residue_from_trace(self, trace_M: float) -> float:
        r"""Residuo de Greene \(R=(2-\operatorname{Tr} M)/4\)."""
        return float((2.0 - trace_M) / 4.0)


@dataclass(frozen=True, slots=True)
class ResonanceStructure:
    r"""
    Estructura de resonancias del vector de frecuencias \(\omega\).
    \(k\in\mathbb{Z}^n\setminus\{0\}\) es resonancia de orden \(\|k\|_1\) si
    \(|\langle k,\omega\rangle|<\varepsilon\).
    """

    vector: Tuple[int, ...]
    order: int
    residue_min: float
    kam_residue: float
    is_kam_safe: bool
    continued_fraction: Tuple[int, ...] = field(default_factory=tuple)
    convergent: Tuple[int, int] = (0, 1)
    brjuno_sum: float = 0.0
    chirikov_overlap: float = 0.0


# ── §1.7 Payload del escenario onírico ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class OniricScenarioPayload:
    r"""
    Payload de un escenario onírico contrafactual. Objeto de 𝐒𝐜𝐞𝐧𝐚𝐫𝐢𝐨.
    Incluye matrices de frontera para la dualidad relativa de Poincaré–Lefschetz.
    """

    scenario_id: str
    dream_state_flag: bool
    synthetic_cartridge_id: str
    density_matrix: np.ndarray
    dirichlet_energy: float
    betti_1_loop_count: int
    betti_0_components: int
    betti_2_cavities: int
    simulated_risk_factor: float
    timestamp_utc: float
    rho_base: Optional[np.ndarray] = None
    boundary_stalk_matrix: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id no puede ser vacío.")
        if not self.synthetic_cartridge_id:
            raise ValueError("synthetic_cartridge_id no puede ser vacío.")
        if self.dirichlet_energy < 0.0:
            raise ValueError("dirichlet_energy debe ser ≥ 0.")
        if (
            self.betti_1_loop_count < 0
            or self.betti_0_components < 0
            or self.betti_2_cavities < 0
        ):
            raise ValueError("Los números de Betti deben ser ≥ 0.")
        if not (0.0 <= self.simulated_risk_factor <= 1.0):
            raise ValueError("simulated_risk_factor debe estar en [0, 1].")
        rho = DensityOperator.from_array(self.density_matrix).as_array()
        object.__setattr__(self, "density_matrix", rho)
        n = rho.shape[0]
        if self.rho_base is None:
            object.__setattr__(self, "rho_base", np.eye(n, dtype=np.complex128) / float(n))
        else:
            object.__setattr__(
                self, "rho_base", DensityOperator.from_array(self.rho_base).as_array()
            )
        if self.boundary_stalk_matrix is None:
            object.__setattr__(self, "boundary_stalk_matrix", np.eye(n, dtype=np.complex128))
        else:
            object.__setattr__(
                self,
                "boundary_stalk_matrix",
                np.asarray(self.boundary_stalk_matrix, dtype=np.complex128),
            )

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        r"""¿\(\rho\) hermitiano, PSD y de traza 1?"""
        rho = self.density_matrix
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        if np.any(la.eigvalsh(rho) < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    def euler_characteristic(self) -> int:
        r"""\(\chi=b_0-b_1+b_2\) (Euler–Poincaré)."""
        return (
            int(self.betti_0_components)
            - int(self.betti_1_loop_count)
            + int(self.betti_2_cavities)
        )

    def topological_class(self) -> str:
        r"""Clasificación discreta del 1-esqueleto: POINT / ARC / LOOPED_LIGHT / LOOPED_DENSE."""
        if self.betti_1_loop_count == 0:
            return "POINT" if self.betti_0_components == 1 else "ARC"
        if self.betti_1_loop_count == 1:
            return "LOOPED_LIGHT"
        return "LOOPED_DENSE"

    def density_operator(self) -> DensityOperator:
        return DensityOperator(matrix=self.density_matrix)


# ── §1.8 Certificados con estructura celeste ──────────────────────────────
@dataclass(frozen=True, slots=True)
class OniricIsolationCertificate:
    r"""
    Certificado de aislamiento homológico.
    Axioma: \(\mathsf{is\_fully\_isolated}\equiv\mathsf{dream\_verified}\wedge\neg\mathsf{hardware\_leak}\).
    Lectura celeste: su negación es una separatriz hiperbólica (VETOED).
    """

    is_fully_isolated: bool
    dream_state_verified: bool
    hardware_leak_risk: bool
    proof_hash: str
    leak_threshold: float

    def logical_consistency(self) -> bool:
        expected = self.dream_state_verified and (not self.hardware_leak_risk)
        return self.is_fully_isolated == expected

    def as_heyting(self) -> HeytingOmega3:
        r"""\(\Lambda_{\mathrm{isolation}}:\mathbb{F}_2\times\mathbb{F}_2\to\Omega_3\)."""
        return HeytingOmega3.from_bool(self.is_fully_isolated)


@dataclass(frozen=True, slots=True)
class ImmunizationCertificate:
    r"""
    Certificado terminal del funtor \(\mathcal{A}:\mathbf{Scenario}\to\mathbf{Cert}_{\mathrm{Inmune}}\).

    Invariantes: dualidad Poincaré–Lefschetz, rigidez de Gromov, \(\Omega_3\),
    sello SHA-256 / prueba SHA-512, y campos celestes v9.1
    (Delaunay–Poincaré, KAM, Brjuno, Chirikov, Greene, Nekhoroshev, Hannay).
    """

    immunization_id: str
    scenario_id: str
    heyting_verdict: HeytingOmega3
    gw_relative_invariant: float
    poincare_lefschetz_defect: float
    symplectic_capacity_ratio: float
    is_boundary_consistent: bool
    crowbar_triggered: bool
    gpio14_signal: str
    proof_merkle_sha512: str
    gromov_witten_invariant: float
    tqft_amplitude: float
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    dirac_total_variation: float
    cstar_residual: float
    dirichlet_bound_valid: bool
    dirichlet_residual: float
    euler_characteristic: int
    topological_class: str
    isolation_cert: OniricIsolationCertificate
    immunization_payload_hash: str
    digital_signature_sha256: str
    timestamp_utc: float
    holonomy_partial: float
    wilson_phase: complex
    delaunay_triple: Optional[Tuple[float, float, float]] = None
    eccentricity: float = 0.0
    inclination: float = 0.0
    kam_residue: float = float("inf")
    resonance_order: int = 0
    maslov_index: int = 0
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_stratum: str = "kam-torus-invariant"
    lyapunov_sign: int = -1
    hannay_angle: float = 0.0
    greene_residue: float = 0.0
    chirikov_overlap: float = 0.0
    nekhoroshev_time: float = float("inf")
    brjuno_sum: float = 0.0
    tisserand: float = 0.0
    jacobi_integral: float = 0.0
    floquet_radius: float = 1.0
    mean_motion: float = 0.0
    poincare_recurrence_time: float = float("inf")

    def is_vetoed(self) -> bool:
        return self.heyting_verdict == HeytingOmega3.VETOED

    def is_immune(self) -> bool:
        r"""
        Inmune ⇔ \(\neg\)VETOED \(\wedge\) aislamiento \(\wedge\) frontera \(\wedge\)
        Gromov \(\wedge\) Dirichlet \(\wedge\) KAM \(\wedge\) Chirikov \(s<1\)
        \(\wedge\) Greene \(|R|\le 1\).
        """
        return (
            self.heyting_verdict != HeytingOmega3.VETOED
            and self.isolation_cert.is_fully_isolated
            and self.isolation_cert.logical_consistency()
            and self.dirichlet_bound_valid
            and self.is_boundary_consistent
            and self.symplectic_capacity_ratio <= 1.25
            and self.kam_residue >= 1.0
            and self.chirikov_overlap < 1.0
            and abs(self.greene_residue) <= 1.0 + 1e-9
        )

    def signature_prefix(self, n: int = 16) -> str:
        return self.digital_signature_sha256[:n]


# ── §1.9 Protocolos estructurales ─────────────────────────────────────────
@runtime_checkable
class IsolationAuditor(Protocol):
    r"""Protocolo de auditoría del aislamiento homológico del escenario onírico."""

    def audit_isolation(self, payload: OniricScenarioPayload) -> OniricIsolationCertificate:
        ...


@runtime_checkable
class GWInvariantEvaluator(Protocol):
    r"""Protocolo de evaluación del invariante relativo Gromov–Witten sintético."""

    def compute_gw_invariant(
        self,
        density_matrix: np.ndarray,
        betti_1: int,
        *,
        dirichlet_energy: Optional[float] = None,
        entropy: float = 0.0,
        dimension: int = 1,
        euler_characteristic: int = 1,
    ) -> float:
        ...


# ── §1.10 Semilla abstracta Spec — cierra FASE-1 ──────────────────────────
class OniricAuditSeed(ABC):
    r"""
    Germen formal de la flecha \(\mathrm{Spec}:\rho\mapsto\lambda\in\Delta^{n-1}\).

    Esta ABC cierra el andamiaje ontológico de FASE-1. FASE-2 **continúa**
    exactamente aquí: `GromovWittenOniricAuditor.extract_spectral_measure` es
    la primera realización real de la semilla.
    """

    @abstractmethod
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        r"""
        Extrae la medida espectral \(\lambda\in\Delta^{n-1}\) del estado \(\rho\in\mathfrak{D}(\mathcal{H}_n)\).
        CONTINÚA EN FASE-2: GromovWittenOniricAuditor.extract_spectral_measure.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — GW/TQFT, DUALIDAD POINCARÉ–LEFSCHETZ, VON NEUMANN, KAM, MELNIKOV
#          Y GUARDIÁN
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método de esta fase ES la realización del último de
# FASE-1 (extract_spectral_measure). El último artefacto de FASE-2
# (UnsealedOniricAuditTrace) es el germen formal de FASE-3
# (_seal_and_classify).
# ══════════════════════════════════════════════════════════════════════════════
class GromovWittenOniricAuditor(OniricAuditSeed):
    r"""
    Auditor TQFT con dualidad relativa de Poincaré–Lefschetz sobre variedades
    abiertas, rigidez de Gromov y análisis celeste completo.

    Doble rol:
        • Realización de la semilla Spec : ρ ↦ λ (FASE 1 → FASE 2).
        • Auditor del invariante relativo Gromov–Witten con cofrontera de borde.
    """

    EIGENVALUE_FLOOR: Final[float] = _EIG_FLOOR
    DIOPHANTINE_GAMMA: Final[float] = 1e-3
    DIOPHANTINE_TAU: Final[float] = 1.5
    DIOPHANTINE_KMAP: Final[int] = 4
    NEKHOROSHEV_EPS_STAR: Final[float] = 0.15
    SYMPLECTIC_RATIO_MAX: Final[float] = 1.25

    def __init__(
        self,
        gw_threshold: float = 0.15,
        lefschetz_tolerance: float = 1e-6,
        capacity_floor: float = 1e-8,
    ) -> None:
        self.gw_threshold = float(gw_threshold)
        self.lefschetz_tolerance = float(lefschetz_tolerance)
        self.capacity_floor = float(capacity_floor)

    @classmethod
    def _project_spectrum(cls, eigvals: np.ndarray) -> np.ndarray:
        r"""Proyección al simplex \(\Delta^{n-1}\)."""
        eigvals = np.clip(np.real(eigvals), cls.EIGENVALUE_FLOOR, None)
        s = float(np.sum(eigvals))
        if s < cls.EIGENVALUE_FLOOR:
            n = eigvals.shape[0]
            return np.full(n, 1.0 / max(n, 1))
        return eigvals / s

    # ── §2.1 Realización de la semilla (HAND-OFF FASE-1 → FASE-2) ────────
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        r"""
        CONTINUACIÓN FORMAL de OniricAuditSeed.extract_spectral_measure.

        Extrae \(\lambda\in\Delta^{n-1}\), pureza \(\gamma\), entropía \(S_{\mathrm{vN}}\),
        brecha \(\Delta\lambda\), energía de Dirichlet \(E_D\), variación total de
        Dirac \(\mathrm{TV}\) y residuo \(C^*\) del estado \(\rho\).
        """
        rho_h = _hermitize_trace_one(rho)
        eigvals = la.eigvalsh(rho_h)
        lam = np.sort(self._project_spectrum(eigvals))
        purity = float(np.sum(lam ** 2))
        entropy = -float(np.sum(lam * np.log(lam)))
        gap = float(lam[-1] - lam[-2]) if lam.size >= 2 else 0.0
        if lam.size < 2:
            ed = 0.0
            tv = 0.0
        else:
            diff = np.diff(lam)
            ed = 0.5 * float(np.sum(diff ** 2))
            tv = float(np.sum(np.abs(diff)))
        op = float(la.norm(rho_h.conj().T @ rho_h, 2))
        nrm = float(la.norm(rho_h, 2))
        return SpectralMeasure(
            eigenvalues=lam,
            purity=purity,
            von_neumann_entropy=entropy,
            spectral_gap=gap,
            dirichlet_energy=ed,
            dirac_total_variation=tv,
            dimension=int(lam.size),
            cstar_residual=abs(op - nrm * nrm),
        )

    # ── §2.2 Diofantinicidad, resonancias, KAM, Brjuno, Nekhoroshev ──────
    @classmethod
    def diophantine_residue(
        cls,
        omega: np.ndarray,
        gamma: Optional[float] = None,
        tau: Optional[float] = None,
        k_max: Optional[int] = None,
    ) -> float:
        r"""
        Residuo diofantino
            \(R_{\gamma,\tau}(\omega)=\min_{0<\|k\|_\infty\le k_{\max}}
              |\langle k,\omega\rangle|/(\gamma\|k\|_\infty^{-\tau})\).
        \(R\ge 1\) ⇔ \(\omega\) es \((\gamma,\tau)\)-diofantino.
        Enumeración por generador (sin `meshgrid`); recorte si \((2k+1)^n>_MESH_CAP\).
        """
        gamma = cls.DIOPHANTINE_GAMMA if gamma is None else float(gamma)
        tau = cls.DIOPHANTINE_TAU if tau is None else float(tau)
        k_max = cls.DIOPHANTINE_KMAP if k_max is None else int(k_max)
        omega = np.asarray(omega, dtype=float).reshape(-1)
        n = int(omega.size)
        if n == 0:
            return float("inf")
        span = 2 * k_max + 1
        while n > 0 and span ** n > _MESH_CAP and k_max > 1:
            k_max -= 1
            span = 2 * k_max + 1
        best = float("inf")
        for k in _iter_inf_lattice(n, k_max):
            kn = float(max(abs(x) for x in k))
            denom = gamma * (kn ** (-tau))
            val = abs(float(np.dot(k, omega))) / max(denom, _EPS)
            if val < best:
                best = val
        return float(best)

    @classmethod
    def detect_resonances(
        cls,
        omega: np.ndarray,
        k_max: int = 3,
        epsilon: float = 1e-6,
        chirikov: float = 0.0,
    ) -> ResonanceStructure:
        r"""Detección exhaustiva de resonancias \(\|k\|_1\le 2k_{\max}\), más CF/Brjuno."""
        omega = np.asarray(omega, dtype=float).reshape(-1)
        n = omega.size
        best_k: Tuple[int, ...] = ()
        best_order = 0
        best_residue = float("inf")
        for order in range(1, 2 * k_max + 1):
            for k_list in _iterate_integer_lattice(n, order):
                val = abs(float(np.dot(k_list, omega)))
                if val < best_residue:
                    best_residue = val
                    best_k = tuple(int(x) for x in k_list)
                    if val < epsilon:
                        best_order = order
        kam_residue = cls.diophantine_residue(omega)
        cf: Tuple[int, ...] = ()
        conv = (0, 1)
        brjuno = 0.0
        if n >= 2 and abs(omega[0]) > 1e-15:
            alpha = float(omega[1] / omega[0])
            cfl = _continued_fraction(alpha)
            cf = tuple(cfl)
            convs = _convergents(cfl)
            if convs:
                conv = convs[-1]
            if len(convs) >= 2:
                acc = 0.0
                for i in range(len(convs) - 1):
                    qk = max(abs(convs[i][1]), 1)
                    qk1 = max(abs(convs[i + 1][1]), 1)
                    acc += math.log(qk1) / qk
                brjuno = acc
        return ResonanceStructure(
            vector=best_k,
            order=best_order,
            residue_min=best_residue,
            kam_residue=kam_residue,
            is_kam_safe=bool(kam_residue >= 1.0),
            continued_fraction=cf,
            convergent=(int(conv[0]), int(conv[1])),
            brjuno_sum=float(brjuno),
            chirikov_overlap=float(chirikov),
        )

    @classmethod
    def kam_torus_indicator(cls, omega: np.ndarray) -> Tuple[bool, float]:
        r"""Indicador binario de toro KAM: \((R\ge 1,\; R)\)."""
        R = cls.diophantine_residue(omega)
        return (R >= 1.0), float(R)

    @classmethod
    def nekhoroshev_stability_time(
        cls, epsilon: float, n_dof: int, eps_star: Optional[float] = None
    ) -> float:
        r"""Tiempo de Nekhoroshev \(T_N\sim\exp((\varepsilon^*/\varepsilon)^{1/(2n)})\)."""
        eps_star = cls.NEKHOROSHEV_EPS_STAR if eps_star is None else float(eps_star)
        eps = max(float(epsilon), 1e-30)
        if eps >= eps_star:
            return 1.0
        expo = min((eps_star / eps) ** (1.0 / max(2 * max(n_dof, 1), 1)), 80.0)
        return float(math.exp(expo))

    # ── §2.3 Wigner, capacidad y Gromov ─────────────────────────────────
    @classmethod
    def wigner_function(cls, rho: np.ndarray) -> np.ndarray:
        r"""Función de Wigner discreta de \(\rho\) (proyección previa)."""
        return DensityOperator.from_array(rho).wigner_function()

    @classmethod
    def symplectic_capacity_gromov(cls, rho: np.ndarray) -> float:
        r"""Capacidad de Gromov heurística \(c_G(\rho)\)."""
        return DensityOperator.from_array(rho).symplectic_capacity_gromov()

    # ── §2.4 Sección de Poincaré y retorno (flujo de von Neumann exacto)
    @classmethod
    def poincare_section_at(
        cls, level: float, dimension: int, i_cut: int = 0, j_cut: int = 1
    ) -> PoincareSectionOniric:
        r"""Construye \(\Sigma\subset M_c\) con normal \(\operatorname{diag}(1,\dots,n)\)."""
        if dimension < 1:
            raise ValueError("dimension debe ser ≥ 1.")
        N_diag = np.diag(np.arange(1, dimension + 1, dtype=float))
        return PoincareSectionOniric(
            level=float(level),
            normal_diag=N_diag,
            i_cut=int(i_cut) % dimension,
            j_cut=int(j_cut) % max(dimension, 1),
        )

    @staticmethod
    def poincare_return_time(
        rho0: np.ndarray,
        section: PoincareSectionOniric,
        N_diag: np.ndarray,
        dt: float = 0.02,
        max_time: float = 100.0,
    ) -> Tuple[float, np.ndarray]:
        r"""
        Primer retorno positivo al corte de *fase* bajo el flujo de von Neumann
        exacto \(\rho_{jk}(t)=\rho_{jk}(0)e^{-i(\nu_j-\nu_k)t}\).
        Si \(\omega=\nu_i-\nu_j\neq 0\), el retorno es analítico: \(T=2\pi/|\omega|\).
        """
        rho = _hermitize_trace_one(rho0)
        nu = _diag_potential(rho.shape[0], N_diag)
        i, j = section.i_cut, section.j_cut
        omega = float(nu[i] - nu[j]) if i != j else 0.0
        if abs(omega) > 1e-15:
            T = 2.0 * math.pi / abs(omega)
            if T <= max_time:
                phase = np.exp(-1j * T * (nu[:, None] - nu[None, :]))
                rho_T = 0.5 * ((rho * phase) + (rho * phase).conj().T)
                return float(T), rho_T
        t = 0.0
        d0 = section.signed_phase(rho)
        phase_dt = np.exp(-1j * dt * (nu[:, None] - nu[None, :]))
        current = rho
        while t < max_time:
            current = 0.5 * ((current * phase_dt) + (current * phase_dt).conj().T)
            t += dt
            d1 = section.signed_phase(current)
            if d0 * d1 < 0.0:
                return t, current
            d0 = d1
        return t, current

    @classmethod
    def poincare_birkhoff_count(cls, p: int, q: int, twist_angle: float) -> int:
        r"""
        Último teorema geométrico de Poincaré (Birkhoff 1913):
        un twist de área en el anillo con \(0<p/q<1\) y \(\theta\in(0,2\pi)\)
        posee al menos \(2q\) puntos periódicos de periodo \(q\).
        """
        if q < 1 or p <= 0 or p >= q:
            return 0
        if not (0.0 < twist_angle < 2.0 * math.pi):
            return 0
        return 2 * q

    @classmethod
    def moser_twist_admissible(
        cls, omega: np.ndarray, actions: np.ndarray, atol: float = 1e-9
    ) -> bool:
        r"""Condición de twist de Moser: \(\det(\partial\omega/\partial J)\neq 0\)."""
        w = np.asarray(omega, dtype=float).reshape(-1)
        J = np.asarray(actions, dtype=float).reshape(-1)
        m = min(w.size, J.size)
        if m < 2:
            return False
        dJ = np.diff(J[:m])
        dw = np.diff(w[:m])
        if np.all(np.abs(dJ) < atol):
            return False
        slope = dw / np.where(np.abs(dJ) < atol, np.nan, dJ)
        slope = slope[np.isfinite(slope)]
        return bool(slope.size > 0 and abs(float(np.mean(slope))) > atol)

    @classmethod
    def monodromy_and_greene(
        cls,
        rho: np.ndarray,
        section: PoincareSectionOniric,
        N_diag: np.ndarray,
        eps: float = 1e-6,
    ) -> FloquetSpectrum:
        r"""Monodromía 2D del mapa de Poincaré, residuo de Greene y radio de Floquet."""
        rho0 = _hermitize_trace_one(rho)
        n = rho0.shape[0]
        i, j = section.i_cut, section.j_cut
        T, _ = cls.poincare_return_time(rho0, section, N_diag)
        nu = _diag_potential(n, N_diag)

        def chart(m: np.ndarray) -> np.ndarray:
            z = m[i, j]
            return np.array([float(np.real(z)), float(np.imag(z))], dtype=float)

        def push(m: np.ndarray) -> np.ndarray:
            phase = np.exp(-1j * T * (nu[:, None] - nu[None, :]))
            return 0.5 * ((m * phase) + (m * phase).conj().T)

        base = chart(rho0)
        cols = []
        for ax in range(2):
            dlt = np.zeros(2, dtype=float)
            dlt[ax] = eps
            pert = rho0.copy()
            pert[i, j] = (base[0] + dlt[0]) + 1j * (base[1] + dlt[1])
            pert[j, i] = np.conj(pert[i, j])
            pert = _hermitize_trace_one(pert)
            cols.append((chart(push(pert)) - chart(push(rho0))) / eps)
        M = np.column_stack(cols) if cols else np.eye(2)
        tr = float(np.trace(M).real)
        ev = np.linalg.eigvals(M)
        radius = float(np.max(np.abs(ev)))
        R = (2.0 - tr) / 4.0
        detM = float(np.real(np.linalg.det(M)))
        return FloquetSpectrum(
            multipliers=ev,
            spectral_radius=radius,
            greene_residue=float(R),
            trace_monodromy=tr,
            is_elliptic=bool(abs(R) < 1.0),
            is_symplectic=bool(abs(detM - 1.0) < 0.15),
        )

    # ── §2.5 Ecuación de Kepler (Halley + arranque de Danby) ─────────────
    @classmethod
    def solve_kepler(
        cls,
        mean_anomaly: float,
        eccentricity: float,
        tol: float = 1e-12,
        max_iter: int = 50,
    ) -> float:
        r"""Kepler \(E-e\sin E=M\) por Halley con arranque de Danby."""
        e = float(eccentricity)
        M = float(mean_anomaly)
        if not (0.0 <= e < 1.0):
            raise ValueError("Excentricidad debe estar en [0, 1).")
        M = (M + math.pi) % (2.0 * math.pi) - math.pi
        if e < 0.8:
            denom = 1.0 - math.sin(M + e) + math.sin(M)
            E = M + e * math.sin(M) / max(denom, 1e-12)
        else:
            E = math.pi * math.copysign(1.0, M) if abs(M) > 1e-15 else math.pi
        for _ in range(max_iter):
            s = math.sin(E)
            c = math.cos(E)
            f = E - e * s - M
            fp = 1.0 - e * c
            fpp = e * s
            den = fp - 0.5 * f * fpp / max(fp, _EPS)
            dE = f / max(den, _EPS)
            E -= dE
            if abs(dE) < tol:
                break
        return float(E)

    @classmethod
    def kepler_residual(
        cls,
        eccentric_anomaly: float,
        mean_anomaly: float,
        eccentricity: float,
    ) -> float:
        r"""Residuo \(F(E)=E-e\sin E-M\)."""
        return float(
            eccentric_anomaly
            - eccentricity * math.sin(eccentric_anomaly)
            - mean_anomaly
        )

    @classmethod
    def gauss_planetary_rates(
        cls, elements: PoincareCelestialElements, pert_eps: float = 0.0
    ) -> Dict[str, float]:
        r"""Ecuaciones planetarias de Gauss a perturbación radial isotrópica."""
        a = max(elements.L ** 2, 1e-30)
        R = float(pert_eps)
        a_dot = 2.0 * math.sqrt(a ** 3) * R
        n = elements.mean_motion
        return {
            "a_dot": float(a_dot),
            "e_dot": 0.0,
            "i_dot": 0.0,
            "mean_motion_dot": float(-1.5 * n / a * a_dot) if a > 0 else 0.0,
        }

    # ── §2.6 Forma normal de Birkhoff–Dulac ──────────────────────────────
    @classmethod
    def birkhoff_normal_form(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        order: int = 2,
    ) -> np.ndarray:
        r"""
        Forma normal de Birkhoff–Dulac hasta orden `order`.
        Linealización \(\mathrm{ad}_N^2(\rho)\) filtrada por Schur; si
        `order>=2` se añade el residuo cuadrático proyectado.
        """
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        N = np.diag(_diag_potential(n, N_diag))
        comm1 = N @ rho_h - rho_h @ N
        X1 = N @ comm1 - comm1 @ N
        tr_num = float(np.real(np.trace(X1 @ rho_h.conj().T)))
        tr_den = float(np.real(np.trace(rho_h @ rho_h.conj().T)))
        if tr_den > _EPS:
            X1 = X1 - (tr_num / tr_den) * rho_h
        if order >= 2:
            comm2 = N @ X1 - X1 @ N
            X2 = N @ comm2 - comm2 @ N
            tr2 = float(np.real(np.trace(X2 @ rho_h.conj().T)))
            if tr_den > _EPS:
                X2 = X2 - (tr2 / tr_den) * rho_h
            X1 = X1 + 0.5 * X2
        return X1

    # ── §2.7 Función de Melnikov (Poisson sobre el flujo hamiltoniano) ───
    @classmethod
    def melnikov_function(
        cls,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        n_steps: int = 32,
        dt: float = 0.02,
    ) -> float:
        r"""
        Integral de Melnikov a lo largo de la órbita hamiltoniana exacta:
            \(M=\int\{H_0,H_1\}(\rho(t))\,dt\),
        \(\{H_0,H_1\}=-i\operatorname{Tr}([N,\rho]\,\rho_{\mathrm{base}})\).
        \(M\neq 0\) ⇒ tubo homoclínico intacto; \(M=0\) ⇒ ruptura.
        """
        rho_h = _hermitize_trace_one(rho_dream)
        n = rho_h.shape[0]
        nu = _diag_potential(n, N_diag)
        base = _hermitize_trace_one(rho_base)
        phase_dt = np.exp(-1j * dt * (nu[:, None] - nu[None, :]))
        total = 0.0
        current = rho_h
        N = np.diag(nu)
        for _ in range(n_steps):
            comm = N @ current - current @ N
            poisson = float(np.real(-1j * np.trace(comm @ base)))
            total += poisson * dt
            current = 0.5 * ((current * phase_dt) + (current * phase_dt).conj().T)
        return float(total)

    # ── §2.8 Auditoría Gromov–Witten relativo con Poincaré–Lefschetz ─────
    def evaluate_poincare_lefschetz_gw_invariant(
        self,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        boundary_stalk_matrix: np.ndarray,
        betti_1_cycles: int = 0,
        dirichlet_energy: Optional[float] = None,
        dream_isolation: bool = True,
        scenario_id: str = "DREAM-POINCARE-EVAL",
        isolation_cert: Optional[OniricIsolationCertificate] = None,
        **kwargs: Any,
    ) -> ImmunizationCertificate:
        r"""
        Audita la rigidez simpléctica y la dualidad de Poincaré–Lefschetz
        entre el sueño y la frontera real, y propaga el análisis celeste.
        """
        measure_dream = self.extract_spectral_measure(rho_dream)
        measure_base = self.extract_spectral_measure(rho_base)
        n = int(np.asarray(rho_dream).shape[0])
        purity = measure_dream.purity
        entropy = measure_dream.von_neumann_entropy
        ed = (
            measure_dream.dirichlet_energy
            if dirichlet_energy is None
            else float(dirichlet_energy)
        )
        rho_d = _hermitize_trace_one(rho_dream)
        rho_b = _hermitize_trace_one(rho_base)
        stalk = np.asarray(boundary_stalk_matrix, dtype=np.complex128)
        boundary_defect = float(la.norm(rho_d @ stalk - stalk @ rho_b, ord="fro"))
        is_boundary_consistent = boundary_defect <= self.lefschetz_tolerance
        evals_dream = measure_dream.eigenvalues
        evals_base = measure_base.eigenvalues
        r_dream_max = float(np.max(evals_dream)) if evals_dream.size > 0 else 1.0
        R_base_min = (
            float(np.min(evals_base[evals_base > 1e-12]))
            if np.any(evals_base > 1e-12)
            else 1.0
        )
        capacity_ratio = r_dream_max / max(R_base_min, self.capacity_floor)
        denom = 1.0 + float(max(betti_1_cycles, 0))
        entropy_damp = math.exp(-abs(entropy) / max(n, 1))
        relative_factor = 1.0 / (1.0 + boundary_defect)
        gw_rel = (purity * math.exp(-ed) * entropy_damp / denom) * relative_factor
        gw_rel = float(min(max(gw_rel, 0.0), 1.0))
        if not dream_isolation:
            verdict = HeytingOmega3.VETOED
        elif (
            gw_rel >= self.gw_threshold
            and is_boundary_consistent
            and capacity_ratio <= 1.05
        ):
            verdict = HeytingOmega3.COHERENT
        elif gw_rel >= self.gw_threshold * 0.5 and capacity_ratio <= self.SYMPLECTIC_RATIO_MAX:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.VETOED
        crowbar_triggered = verdict == HeytingOmega3.VETOED
        gpio14_signal = "HIGH" if crowbar_triggered else "LOW"
        proof_str = (
            f"{gw_rel:.8f}|{boundary_defect:.8f}|{capacity_ratio:.8f}|{verdict.value}"
        )
        proof_sha512 = hashlib.sha512(proof_str.encode("utf-8")).hexdigest()
        imm_id = f"IMM-PL-{hashlib.md5(proof_str.encode('utf-8')).hexdigest()[:8]}"
        t_seal = time.time()
        sig_str = f"{scenario_id}::{imm_id}::{proof_sha512}::{t_seal:.6f}"
        sig_sha256 = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()
        isol = (
            isolation_cert
            if isolation_cert is not None
            else OniricIsolationCertificate(
                is_fully_isolated=dream_isolation,
                dream_state_verified=dream_isolation,
                hardware_leak_risk=not dream_isolation,
                proof_hash=sig_sha256,
                leak_threshold=0.8,
            )
        )
        L_del = G_del = H_del = 0.0
        e_del = i_del = 0.0
        kam_res = float("inf")
        res_order = 0
        wigner_neg = 0.0
        capacity = 1.0
        maslov = 0
        hannay = 0.0
        brjuno = 0.0
        greene = 0.0
        floquet_r = 1.0
        t_nek = float("inf")
        tisserand = 0.0
        jacobi = 0.0
        mean_motion = 0.0
        chirikov = 0.0
        tau_rec = float("inf")
        try:
            rho_op = DensityOperator.from_array(rho_d)
            L_del, G_del, H_del = rho_op.delaunay_triple()
            e_del, i_del = rho_op.eccentricity_inclination()
            omega = rho_op.mean_motion_frequencies()
            chirikov = rho_op.chirikov_overlap()
            res_struct = self.detect_resonances(omega, chirikov=chirikov)
            kam_res = res_struct.kam_residue
            res_order = res_struct.order
            wigner_neg = rho_op.wigner_negativity()
            capacity = rho_op.symplectic_capacity_gromov()
            maslov = rho_op.maslov_index()
            hannay = rho_op.hannay_angle()
            brjuno = rho_op.siegel_brjuno_sum()
            tau_rec = rho_op.poincare_recurrence_time()
            elems = rho_op.celestial_elements()
            tisserand = elems.tisserand
            jacobi = elems.jacobi_integral
            mean_motion = elems.mean_motion
            N4 = np.diag(_diag_potential(n))
            section = self.poincare_section_at(elems.energy, n)
            floq = self.monodromy_and_greene(rho_op.matrix, section, N4)
            greene = floq.greene_residue
            floquet_r = floq.spectral_radius
            eps_comm = float(la.norm(rho_op.matrix @ N4 - N4 @ rho_op.matrix, "fro"))
            t_nek = self.nekhoroshev_stability_time(eps_comm, n)
        except Exception:
            logger.exception("Fallo en el análisis celeste; se aplican neutros.")
        return ImmunizationCertificate(
            immunization_id=imm_id,
            scenario_id=scenario_id,
            heyting_verdict=verdict,
            gw_relative_invariant=gw_rel,
            poincare_lefschetz_defect=boundary_defect,
            symplectic_capacity_ratio=capacity_ratio,
            is_boundary_consistent=is_boundary_consistent,
            crowbar_triggered=crowbar_triggered,
            gpio14_signal=gpio14_signal,
            proof_merkle_sha512=proof_sha512,
            gromov_witten_invariant=gw_rel,
            tqft_amplitude=gw_rel,
            purity=purity,
            von_neumann_entropy=entropy,
            spectral_gap=measure_dream.spectral_gap,
            dirac_total_variation=measure_dream.dirac_total_variation,
            cstar_residual=measure_dream.cstar_residual,
            dirichlet_bound_valid=ed <= 0.85,
            dirichlet_residual=0.0,
            euler_characteristic=1 - betti_1_cycles,
            topological_class=("POINT" if betti_1_cycles == 0 else "LOOPED_LIGHT"),
            isolation_cert=isol,
            immunization_payload_hash=proof_sha512,
            digital_signature_sha256=sig_sha256,
            timestamp_utc=t_seal,
            holonomy_partial=gw_rel,
            wilson_phase=complex(math.cos(gw_rel), math.sin(gw_rel)),
            delaunay_triple=(L_del, G_del, H_del),
            eccentricity=e_del,
            inclination=i_del,
            kam_residue=kam_res,
            resonance_order=res_order,
            maslov_index=maslov,
            wigner_negativity=wigner_neg,
            symplectic_capacity=capacity,
            poincare_stratum=verdict.poincare_stratum_name(),
            lyapunov_sign=verdict.lyapunov_exponent_sign(),
            hannay_angle=hannay,
            greene_residue=greene,
            chirikov_overlap=chirikov,
            nekhoroshev_time=t_nek,
            brjuno_sum=brjuno,
            tisserand=tisserand,
            jacobi_integral=jacobi,
            floquet_radius=floquet_r,
            mean_motion=mean_motion,
            poincare_recurrence_time=tau_rec,
        )

    @classmethod
    def compute_purity(cls, density_matrix: np.ndarray) -> float:
        r"""\(\gamma(\rho)=\operatorname{Tr}(\rho^2)\) vía proyección espectral canónica."""
        eigvals = la.eigvalsh(_hermitize_trace_one(density_matrix))
        lam = cls._project_spectrum(eigvals)
        return float(np.sum(lam ** 2))

    def compute_gw_invariant(
        self,
        density_matrix: np.ndarray,
        betti_1: int,
        *,
        dirichlet_energy: Optional[float] = None,
        entropy: float = 0.0,
        dimension: int = 1,
        euler_characteristic: int = 1,
    ) -> float:
        r"""Alias flexible del invariante GW sintético."""
        measure = self.extract_spectral_measure(density_matrix)
        ed = measure.dirichlet_energy if dirichlet_energy is None else float(dirichlet_energy)
        return self.compute_gromov_witten_invariant(
            density_matrix,
            betti_1,
            dirichlet_energy=ed,
            entropy=measure.von_neumann_entropy if entropy == 0.0 else entropy,
            dimension=measure.dimension if dimension == 1 else dimension,
            euler_characteristic=euler_characteristic,
        )

    @classmethod
    def compute_gromov_witten_invariant(
        cls,
        density_matrix: np.ndarray,
        betti_1: int,
        *,
        dirichlet_energy: Optional[float] = None,
        entropy: float = 0.0,
        dimension: int = 1,
        euler_characteristic: int = 1,
    ) -> float:
        r"""Invariante GW sintético con factor de Euler–Poincaré."""
        if betti_1 < 0:
            raise ValueError("b₁ debe ser ≥ 0.")
        gamma = cls.compute_purity(density_matrix)
        denom = 1.0 + float(betti_1)
        if dirichlet_energy is None:
            return float(min(max(gamma / denom, 0.0), 1.0))
        n = max(int(dimension), 1)
        chi_pos = max(int(euler_characteristic), 0)
        chi_abs = abs(int(euler_characteristic))
        euler_factor = (1.0 + chi_pos) / (1.0 + chi_abs)
        entropy_damp = math.exp(-abs(entropy) / n)
        raw = gamma * math.exp(-float(dirichlet_energy)) * entropy_damp / denom * euler_factor
        return float(min(max(raw, 0.0), 1.0))


# ── §2.9 Guardián del aislamiento homológico ─────────────────────────────
class OniricIsolationGuard:
    r"""
    Guardián de aislamiento homológico.
    Axioma: \(\mathsf{is\_fully\_isolated}\equiv\mathsf{dream\_verified}\wedge\neg\mathsf{hardware\_leak}\),
    con \(\mathsf{hardware\_leak}:=(\neg\mathsf{dream})\wedge(\mathrm{risk}>0.8)\).
    """

    LEAK_RISK_THRESHOLD: Final[float] = 0.8

    def audit_isolation(self, payload: OniricScenarioPayload) -> OniricIsolationCertificate:
        r"""Aplica el axioma de aislamiento y sella con SHA-256."""
        dream_verified = bool(payload.dream_state_flag)
        hardware_leak = (not dream_verified) and (
            payload.simulated_risk_factor > self.LEAK_RISK_THRESHOLD
        )
        is_isolated = dream_verified and (not hardware_leak)
        h = hashlib.sha256()
        payload_bytes = (
            f"{payload.scenario_id}::{payload.dream_state_flag}::"
            f"{is_isolated}::{payload.synthetic_cartridge_id}::"
            f"{payload.simulated_risk_factor:.6f}"
        )
        h.update(payload_bytes.encode("utf-8"))
        cert = OniricIsolationCertificate(
            is_fully_isolated=is_isolated,
            dream_state_verified=dream_verified,
            hardware_leak_risk=hardware_leak,
            proof_hash=h.hexdigest(),
            leak_threshold=self.LEAK_RISK_THRESHOLD,
        )
        if not cert.logical_consistency():
            raise RuntimeError("Violación del axioma de aislamiento homológico.")
        return cert


# ── §2.10 Traza abierta: último artefacto de FASE-2, germen de FASE-3 ────
@dataclass(frozen=True, slots=True)
class UnsealedOniricAuditTrace:
    r"""
    Traza abierta portadora de \((\mathrm{Spec},D,I_{GW},\mathrm{Isol},\mathrm{celeste})\)
    antes de \(V/\mathrm{Seal}\).

    Cierra FASE-2. FASE-3 **continúa** exactamente aquí:
    `OniricDreamAuditorAgent._seal_and_classify` consume esta traza.
    """

    payload: OniricScenarioPayload
    measure: SpectralMeasure
    gromov_witten_invariant: float
    tqft_amplitude: float
    dirichlet_bound_valid: bool
    dirichlet_residual: float
    isolation_cert: OniricIsolationCertificate
    poincare_cert: Optional[ImmunizationCertificate] = None


# ── §2.11 Compositor de las flechas Spec → I_GW → Isol → V ────────────────
class OniricAuditArrowComposer:
    r"""
    Compositor de las flechas \(\mathrm{Isol}\circ I_{GW}\circ D\circ\mathrm{Spec}\)
    con dualidad Poincaré–Lefschetz.

    Produce `UnsealedOniricAuditTrace`, germen formal de FASE-3.
    """

    DIRICHLET_CONSISTENCY_TOL: Final[float] = 1e-3

    def __init__(
        self,
        spectra: GromovWittenOniricAuditor,
        isolation_auditor: IsolationAuditor,
        energy_threshold: float,
        include_dirichlet_attenuation: bool,
    ) -> None:
        self.spectra = spectra
        self.isolation_auditor = isolation_auditor
        self.energy_threshold = float(energy_threshold)
        self.include_dirichlet_attenuation = bool(include_dirichlet_attenuation)

    def compose_audit_arrows(self, payload: OniricScenarioPayload) -> UnsealedOniricAuditTrace:
        r"""
        Compone las flechas en orden (Spec → D → I_GW → Isol → V).
        En caso de aislamiento fallido, fuerza \((I_{GW}=0,\;\mathrm{dirichlet\_valid}=\mathrm{False})\)
        y deja que FASE-3 aplique VETOED + crowbar.
        """
        measure = self.spectra.extract_spectral_measure(payload.density_matrix)
        ed_residual = abs(payload.dirichlet_energy - measure.dirichlet_energy)
        isolation = self.isolation_auditor.audit_isolation(payload)
        n = payload.density_matrix.shape[0]
        poincare_cert = self.spectra.evaluate_poincare_lefschetz_gw_invariant(
            rho_dream=payload.density_matrix,
            rho_base=(
                payload.rho_base
                if payload.rho_base is not None
                else np.eye(n, dtype=np.complex128) / float(n)
            ),
            boundary_stalk_matrix=(
                payload.boundary_stalk_matrix
                if payload.boundary_stalk_matrix is not None
                else np.eye(n, dtype=np.complex128)
            ),
            betti_1_cycles=payload.betti_1_loop_count,
            dirichlet_energy=payload.dirichlet_energy,
            dream_isolation=isolation.is_fully_isolated,
            scenario_id=payload.scenario_id,
            isolation_cert=isolation,
        )
        if not isolation.is_fully_isolated:
            return UnsealedOniricAuditTrace(
                payload=payload,
                measure=measure,
                gromov_witten_invariant=0.0,
                tqft_amplitude=0.0,
                dirichlet_bound_valid=False,
                dirichlet_residual=ed_residual,
                isolation_cert=isolation,
                poincare_cert=poincare_cert,
            )
        gw = poincare_cert.gw_relative_invariant
        dirichlet_valid = payload.dirichlet_energy <= self.energy_threshold
        return UnsealedOniricAuditTrace(
            payload=payload,
            measure=measure,
            gromov_witten_invariant=gw,
            tqft_amplitude=gw,
            dirichlet_bound_valid=dirichlet_valid,
            dirichlet_residual=ed_residual,
            isolation_cert=isolation,
            poincare_cert=poincare_cert,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — SOBERANO AUDITOR, SELLO DUAL, HOLONOMÍA BERRY–HANNAY, MERKLE
#          Y PASAPORTE
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método operativo de esta fase (_seal_and_classify)
# consume UnsealedOniricAuditTrace (último artefacto de FASE-2). Aquí se
# completa el funtor 𝒜 = Seal ∘ Crowbar ∘ V ∘ Isol ∘ I_GW ∘ Spec.
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    r"""
    Prueba de inclusión Merkle SHA-256 (convención CT/Bitcoin).

        node := leaf; idx := index
        para cada hermano: node := SHA256(node‖sib) si idx par, else SHA256(sib‖node);
        idx //= 2
        assert node == root
    """

    leaf_hash: str
    siblings: Tuple[str, ...]
    index: int
    root: str

    def verify(self) -> bool:
        r"""Recalcula el camino de inclusión y compara con la raíz."""
        try:
            node = bytes.fromhex(self.leaf_hash)
        except ValueError:
            return False
        idx = self.index
        for sib_hex in self.siblings:
            try:
                sib = bytes.fromhex(sib_hex)
            except ValueError:
                return False
            if idx % 2 == 0:
                node = hashlib.sha256(node + sib).digest()
            else:
                node = hashlib.sha256(sib + node).digest()
            idx //= 2
        return node.hex() == self.root


class OniricDreamAuditorAgent:
    r"""
    Soberano auditor de escenarios contrafactuales del Onírico.

    Realiza el funtor \(\mathcal{A}=\mathrm{Seal}\circ\mathrm{Crowbar}\circ V\circ\mathrm{Isol}\circ I_{GW}\circ\mathrm{Spec}\).
    Co-gobierna el Estrato Wisdom (\(V_\mathbb{W}\)) junto al
    `TOONWisdomWeaverAgent` y al `TOONOniricAuditorEngine`.

    Cadena de custodia: cada certificado se indexa en el árbol Merkle global;
    la holonomía Wilson \(\sum\gamma_{GW}\) y la de Hannay \(\sum\theta_H\)
    se acumulan sobre los ciclos; la fase de Berry–Pancharatnam se integra
    sobre la curva \(\rho_0\to\cdots\to\rho_n\) de estados auditados.
    """

    _DEFAULT_ENERGY_THRESHOLD: Final[float] = 0.85
    _DEFAULT_GW_TOLERANCE: Final[float] = 0.15
    _DEFAULT_BETTI_MAX: Final[int] = 3
    _DEFAULT_DIRAC_TV_MAX: Final[float] = 1.50

    def __init__(
        self,
        agent_id: str = "ONIRIC-AUDITOR-SABIO-01",
        energy_threshold: float = _DEFAULT_ENERGY_THRESHOLD,
        gw_tolerance: float = _DEFAULT_GW_TOLERANCE,
        betti_max: int = _DEFAULT_BETTI_MAX,
        dirac_tv_max: float = _DEFAULT_DIRAC_TV_MAX,
        isolation_auditor: Optional[IsolationAuditor] = None,
        gw_evaluator: Optional[GromovWittenOniricAuditor] = None,
        *,
        include_dirichlet_attenuation: bool = True,
    ) -> None:
        if not (0.0 < energy_threshold <= 2.0):
            raise ValueError("energy_threshold debe estar en (0, 2].")
        if not (0.0 <= gw_tolerance <= 1.0):
            raise ValueError("gw_tolerance debe estar en [0, 1].")
        if betti_max < 0:
            raise ValueError("betti_max debe ser ≥ 0.")
        if dirac_tv_max < 0.0:
            raise ValueError("dirac_tv_max debe ser ≥ 0.")
        self.agent_id: str = agent_id
        self.energy_threshold: float = float(energy_threshold)
        self.gw_tolerance: float = float(gw_tolerance)
        self.betti_max: int = int(betti_max)
        self.dirac_tv_max: float = float(dirac_tv_max)
        self._include_dirichlet_attenuation: bool = bool(include_dirichlet_attenuation)
        spectra = (
            gw_evaluator
            if gw_evaluator is not None
            else GromovWittenOniricAuditor(gw_threshold=gw_tolerance)
        )
        isolator: IsolationAuditor = (
            isolation_auditor if isolation_auditor is not None else OniricIsolationGuard()
        )
        self.composer: OniricAuditArrowComposer = OniricAuditArrowComposer(
            spectra=spectra,
            isolation_auditor=isolator,
            energy_threshold=self.energy_threshold,
            include_dirichlet_attenuation=self._include_dirichlet_attenuation,
        )
        self._isolation_auditor = isolator
        self._gw_evaluator = spectra
        self.audit_count: int = 0
        self.immunization_registry: List[ImmunizationCertificate] = []
        self._density_history: List[np.ndarray] = []
        self._holonomy_accum: float = 0.0
        self._hannay_accum: float = 0.0

    # ── §3.1 HAND-OFF FASE-2 → FASE-3: sello y clasificación ─────────────
    def _seal_and_classify(self, trace: UnsealedOniricAuditTrace) -> ImmunizationCertificate:
        r"""
        CONTINUACIÓN FORMAL de OniricAuditArrowComposer.compose_audit_arrows.

        Consume `UnsealedOniricAuditTrace`, aplica \(V\) (flecha característica
        \(\chi:\mathrm{Trace}\to\Omega_3\)), sella con SHA-256/SHA-512, acumula
        holonomías Wilson y Hannay, y construye el `ImmunizationCertificate`
        inmutable.
        """
        payload = trace.payload
        isolation = trace.isolation_cert
        measure = trace.measure
        verdict = self._classify(
            betti_1=payload.betti_1_loop_count,
            gw_inv=trace.gromov_witten_invariant,
            dirichlet_valid=trace.dirichlet_bound_valid,
            isolation=isolation,
            dirac_tv=measure.dirac_total_variation,
            poincare_cert=trace.poincare_cert,
        )
        imm_id = f"IMM-ONIRIC-{self.audit_count:04d}"
        t_seal = time.time()
        p_cert = trace.poincare_cert
        boundary_defect = p_cert.poincare_lefschetz_defect if p_cert else 0.0
        cap_ratio = p_cert.symplectic_capacity_ratio if p_cert else 1.0
        is_boundary_consistent = p_cert.is_boundary_consistent if p_cert else True
        crowbar_triggered = verdict == HeytingOmega3.VETOED
        gpio14_signal = "HIGH" if crowbar_triggered else "LOW"
        delaunay = p_cert.delaunay_triple if p_cert else None
        eccentricity = p_cert.eccentricity if p_cert else 0.0
        inclination = p_cert.inclination if p_cert else 0.0
        kam_res = p_cert.kam_residue if p_cert else float("inf")
        res_order = p_cert.resonance_order if p_cert else 0
        maslov = p_cert.maslov_index if p_cert else 0
        wigner_neg = p_cert.wigner_negativity if p_cert else 0.0
        symplectic_cap = p_cert.symplectic_capacity if p_cert else 1.0
        hannay = p_cert.hannay_angle if p_cert else 0.0
        greene = p_cert.greene_residue if p_cert else 0.0
        chirikov = p_cert.chirikov_overlap if p_cert else 0.0
        t_nek = p_cert.nekhoroshev_time if p_cert else float("inf")
        brjuno = p_cert.brjuno_sum if p_cert else 0.0
        tisserand = p_cert.tisserand if p_cert else 0.0
        jacobi = p_cert.jacobi_integral if p_cert else 0.0
        floquet_r = p_cert.floquet_radius if p_cert else 1.0
        mean_motion = p_cert.mean_motion if p_cert else 0.0
        tau_rec = p_cert.poincare_recurrence_time if p_cert else float("inf")
        stratum = verdict.poincare_stratum_name()
        lyap = verdict.lyapunov_exponent_sign()
        proof_str = (
            f"{payload.scenario_id}|{trace.gromov_witten_invariant:.8f}|"
            f"{boundary_defect:.8f}|{cap_ratio:.8f}|{verdict.value}"
        )
        merkle_sha512 = hashlib.sha512(proof_str.encode("utf-8")).hexdigest()
        h_payload = hashlib.sha256(proof_str.encode("utf-8")).hexdigest()
        h_sig = hashlib.sha256(
            f"{self.agent_id}::{imm_id}::{h_payload}::{t_seal:.6f}".encode("utf-8")
        ).hexdigest()
        self._holonomy_accum += trace.gromov_witten_invariant
        self._hannay_accum += hannay
        wilson = complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))
        return ImmunizationCertificate(
            immunization_id=imm_id,
            scenario_id=payload.scenario_id,
            heyting_verdict=verdict,
            gw_relative_invariant=float(trace.gromov_witten_invariant),
            poincare_lefschetz_defect=boundary_defect,
            symplectic_capacity_ratio=cap_ratio,
            is_boundary_consistent=is_boundary_consistent,
            crowbar_triggered=crowbar_triggered,
            gpio14_signal=gpio14_signal,
            proof_merkle_sha512=merkle_sha512,
            gromov_witten_invariant=float(trace.gromov_witten_invariant),
            tqft_amplitude=float(trace.tqft_amplitude),
            purity=measure.purity,
            von_neumann_entropy=measure.von_neumann_entropy,
            spectral_gap=measure.spectral_gap,
            dirac_total_variation=measure.dirac_total_variation,
            cstar_residual=measure.cstar_residual,
            dirichlet_bound_valid=bool(trace.dirichlet_bound_valid),
            dirichlet_residual=float(trace.dirichlet_residual),
            euler_characteristic=payload.euler_characteristic(),
            topological_class=payload.topological_class(),
            isolation_cert=isolation,
            immunization_payload_hash=h_payload,
            digital_signature_sha256=h_sig,
            timestamp_utc=t_seal,
            holonomy_partial=self._holonomy_accum,
            wilson_phase=wilson,
            delaunay_triple=delaunay,
            eccentricity=eccentricity,
            inclination=inclination,
            kam_residue=kam_res,
            resonance_order=res_order,
            maslov_index=maslov,
            wigner_negativity=wigner_neg,
            symplectic_capacity=symplectic_cap,
            poincare_stratum=stratum,
            lyapunov_sign=lyap,
            hannay_angle=hannay,
            greene_residue=greene,
            chirikov_overlap=chirikov,
            nekhoroshev_time=t_nek,
            brjuno_sum=brjuno,
            tisserand=tisserand,
            jacobi_integral=jacobi,
            floquet_radius=floquet_r,
            mean_motion=mean_motion,
            poincare_recurrence_time=tau_rec,
        )

    def _classify(
        self,
        *,
        betti_1: int,
        gw_inv: float,
        dirichlet_valid: bool,
        isolation: OniricIsolationCertificate,
        dirac_tv: float,
        poincare_cert: Optional[ImmunizationCertificate] = None,
    ) -> HeytingOmega3:
        r"""
        Flecha característica \(\chi:\mathrm{Trace}\to\Omega_3\).
        Reglas intuicionistas (no booleanas): aislamiento o \(b_1>b_{\max}\)
        fuerzan VETOED; Dirichlet/TV fallidos degradan; el resto hereda el
        veredicto Lefschetz o asciende a COHERENT.
        """
        if poincare_cert is not None:
            if not isolation.is_fully_isolated or betti_1 > self.betti_max:
                return HeytingOmega3.VETOED
            if (
                not poincare_cert.is_boundary_consistent
                or poincare_cert.symplectic_capacity_ratio > 1.25
            ):
                return HeytingOmega3.VETOED
            return poincare_cert.heyting_verdict
        if not isolation.is_fully_isolated:
            return HeytingOmega3.VETOED
        if betti_1 > self.betti_max or gw_inv < self.gw_tolerance:
            return HeytingOmega3.VETOED
        if not dirichlet_valid or dirac_tv > self.dirac_tv_max:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    # ── §3.2 Auditoría de un escenario ───────────────────────────────────
    def audit_dream_scenario(self, payload: OniricScenarioPayload) -> ImmunizationCertificate:
        r"""Ejecuta \(\mathcal{A}(\mathrm{payload})=\mathrm{Seal}\circ\mathrm{Crowbar}\circ V\circ\mathrm{Isol}\circ I_{GW}\circ\mathrm{Spec}\)."""
        self.audit_count += 1
        t_start = time.time()
        logger.info(
            "=== Auditando Escenario Onírico #%d | ID: %s ===",
            self.audit_count,
            payload.scenario_id,
        )
        trace = self.composer.compose_audit_arrows(payload)
        cert = self._seal_and_classify(trace)
        self.immunization_registry.append(cert)
        self._density_history.append(np.array(payload.density_matrix, copy=True))
        logger.info(
            "Auditoría Onírica Finalizada en %.2f ms | Veredicto: %s | "
            "I_GW=%.6f | Defect=%.6f | R_KAM=%.3e | Chirikov=%.3f | "
            "Greene=%.3f | Stratum=%s | Crowbar=%s | GPIO14=%s",
            (time.time() - t_start) * 1000.0,
            cert.heyting_verdict.name,
            cert.gw_relative_invariant,
            cert.poincare_lefschetz_defect,
            cert.kam_residue,
            cert.chirikov_overlap,
            cert.greene_residue,
            cert.poincare_stratum,
            cert.crowbar_triggered,
            cert.gpio14_signal,
        )
        return cert

    # ── §3.3 Vistas y propiedades agregadas ──────────────────────────────
    @property
    def registry(self) -> Tuple[ImmunizationCertificate, ...]:
        return tuple(self.immunization_registry)

    @property
    def holonomy_accum(self) -> float:
        return self._holonomy_accum

    @property
    def wilson_loop(self) -> complex:
        r"""Lazo de Wilson \(e^{i\sum\gamma_{GW}}\in U(1)\)."""
        return complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))

    @property
    def hannay_loop(self) -> complex:
        r"""Holonomía de Hannay \(e^{i\sum\theta_H}\in U(1)\)."""
        return complex(math.cos(self._hannay_accum), math.sin(self._hannay_accum))

    @property
    def global_verdict(self) -> HeytingOmega3:
        r"""Ínfimo (meet) de los veredictos — objeto terminal de \(\Omega_3\)."""
        gv = HeytingOmega3.COHERENT
        for c in self.immunization_registry:
            gv = gv.meet(c.heyting_verdict)
        return gv

    @property
    def berry_phase_along_history(self) -> float:
        r"""Fase de Berry–Pancharatnam cerrada a lo largo de la historia de \(\rho\)."""
        if len(self._density_history) < 2:
            return 0.0
        return DensityOperator.from_array(self._density_history[0]).berry_phase_curve(
            self._density_history, closed=True
        )

    # ── §3.4 Árbol de Merkle ─────────────────────────────────────────────
    @staticmethod
    def _merkle_tree_root(leaf_hashes: Sequence[str]) -> str:
        r"""Raíz Merkle SHA-256 (duplica la última hoja si el nivel es impar)."""
        if not leaf_hashes:
            return hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
        level = [bytes.fromhex(h) for h in leaf_hashes]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [
                hashlib.sha256(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
        return level[0].hex()

    @staticmethod
    def _merkle_proof(leaf_hashes: Sequence[str], index: int) -> MerkleInclusionProof:
        r"""Prueba de inclusión coherente con el padding de duplicado impar."""
        if not leaf_hashes:
            empty = hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
            return MerkleInclusionProof(empty, tuple(), 0, empty)
        if not (0 <= index < len(leaf_hashes)):
            raise IndexError("índice de hoja Merkle fuera de rango")
        level = [bytes.fromhex(h) for h in leaf_hashes]
        siblings: List[str] = []
        idx = index
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            pair = idx ^ 1
            siblings.append(level[pair].hex())
            next_level = [
                hashlib.sha256(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
            level = next_level
            idx //= 2
        return MerkleInclusionProof(
            leaf_hash=leaf_hashes[index],
            siblings=tuple(siblings),
            index=index,
            root=level[0].hex(),
        )

    def merkle_root(self) -> str:
        return self._merkle_tree_root(
            [c.digital_signature_sha256 for c in self.immunization_registry]
        )

    def merkle_proofs_ok(self) -> bool:
        leaves = [c.digital_signature_sha256 for c in self.immunization_registry]
        root = self._merkle_tree_root(leaves)
        for i in range(len(leaves)):
            proof = self._merkle_proof(leaves, i)
            if proof.root != root or not proof.verify():
                return False
        return True

    # ── §3.5 Auditoría retrospectiva y pasaporte ─────────────────────────
    def audit_registry(self) -> Dict[str, Any]:
        r"""Auditoría retrospectiva del registro, holonomías y Merkle."""
        n = len(self.immunization_registry)
        empty_dist = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_certificates": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "holonomy_accum": 0.0,
                "wilson_loop": 1.0 + 0.0j,
                "hannay_loop": 1.0 + 0.0j,
                "berry_phase": 0.0,
                "avg_gw_invariant": 0.0,
                "avg_purity": 0.0,
                "avg_dirac_tv": 0.0,
                "avg_dirichlet_residual": 0.0,
                "avg_kam_residue": 0.0,
                "avg_capacity": 0.0,
                "avg_chirikov": 0.0,
                "avg_greene": 0.0,
                "n_immune": 0,
                "n_hardware_leak_risk": 0,
                "registry_integrity_ok": True,
                "merkle_proofs_ok": True,
                "isolation_axioms_ok": True,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        total_gw = total_p = total_tv = total_ed_res = 0.0
        total_kam = total_cap = total_ch = total_gr = 0.0
        immune_count = leak_count = 0
        signatures: Set[str] = set()
        collide = False
        axioms_ok = True
        for c in self.immunization_registry:
            dist[c.heyting_verdict.name] += 1
            total_gw += c.gromov_witten_invariant
            total_p += c.purity
            total_tv += c.dirac_total_variation
            total_ed_res += c.dirichlet_residual
            total_kam += c.kam_residue if math.isfinite(c.kam_residue) else 0.0
            total_cap += c.symplectic_capacity
            total_ch += c.chirikov_overlap
            total_gr += c.greene_residue
            if c.isolation_cert.hardware_leak_risk:
                leak_count += 1
            if c.is_immune():
                immune_count += 1
            if not c.isolation_cert.logical_consistency():
                axioms_ok = False
            if c.digital_signature_sha256 in signatures:
                collide = True
            signatures.add(c.digital_signature_sha256)
        inv = 1.0 / n
        return {
            "n_certificates": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "hannay_loop": self.hannay_loop,
            "berry_phase": self.berry_phase_along_history,
            "avg_gw_invariant": total_gw * inv,
            "avg_purity": total_p * inv,
            "avg_dirac_tv": total_tv * inv,
            "avg_dirichlet_residual": total_ed_res * inv,
            "avg_kam_residue": total_kam * inv,
            "avg_capacity": total_cap * inv,
            "avg_chirikov": total_ch * inv,
            "avg_greene": total_gr * inv,
            "n_immune": immune_count,
            "n_hardware_leak_risk": leak_count,
            "registry_integrity_ok": not collide,
            "merkle_proofs_ok": self.merkle_proofs_ok(),
            "isolation_axioms_ok": axioms_ok,
        }

    def emit_immunization_passport(self) -> Dict[str, Any]:
        r"""Pasaporte criptográfico agregado (GodelEngine / Ciudadela de Cristal)."""
        h = hashlib.sha256()
        h.update(
            f"{self.agent_id}::{self.audit_count}::{self._holonomy_accum:.10f}".encode("utf-8")
        )
        for c in self.immunization_registry:
            h.update(c.digital_signature_sha256.encode("utf-8"))
        return {
            "agent_id": self.agent_id,
            "registry_size": self.audit_count,
            "global_verdict": self.global_verdict.name,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "hannay_loop": self.hannay_loop,
            "berry_phase": self.berry_phase_along_history,
            "n_immune": sum(1 for c in self.immunization_registry if c.is_immune()),
            "merkle_root": self.merkle_root(),
            "evidence_hash": h.hexdigest(),
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
    print("DEMOSTRACIÓN GRANULAR: TOON Oniric Auditor Agent v9.1.0-Poincare-Celeste")
    print("FASES: Ω₃+Spec+Wigner → GW/von Neumann/KAM/Isol → V/Crowbar/Merkle")
    print("═" * 80)

    print("\n[§0] VERIFICACIÓN FORMAL DE Ω₃")
    assert HeytingOmega3.verify_residuation_axiom()
    assert HeytingOmega3.DEGRADED.excluded_middle_holds() is False
    assert HeytingOmega3.COHERENT.excluded_middle_holds() is True
    assert HeytingOmega3.VETOED.is_regular() is True
    assert HeytingOmega3.DEGRADED.is_regular() is False
    assert HeytingOmega3.VETOED.poincare_stratum_name() == "hyperbolic-escape-separatrix"
    assert HeytingOmega3.COHERENT.is_kam_stratum() is True
    assert HeytingOmega3.COHERENT.lyapunov_exponent_sign() == -1
    print("  • Residuación, tercio excluso, regularidad y estratos: OK")

    lat = _iterate_integer_lattice(2, 2)
    assert any(np.array_equal(v, np.array([2, 0])) for v in lat)
    assert any(np.array_equal(v, np.array([-2, 0])) for v in lat)
    print("  • Red ℤ² de norma ℓ¹=2 contiene (±2,0): OK")

    rng = np.random.default_rng(20250321)
    auditor = OniricDreamAuditorAgent(
        agent_id="ONIRIC-AUDITOR-POINCARE-01",
        include_dirichlet_attenuation=True,
    )
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real

    rho_op = DensityOperator.from_array(rho)
    W = rho_op.wigner_function()
    assert W.shape == (4, 4)
    assert abs(float(np.sum(W)) - 1.0) < 1e-6
    print(f"  • Wigner FFT (4×4): ΣW={np.sum(W):.6f}, N_W={rho_op.wigner_negativity():.6e}")
    print(f"  • Capacidad Gromov c_G(ρ)={rho_op.symplectic_capacity_gromov():.6f}")
    print(f"  • Recurrencia Poincaré τ={rho_op.poincare_recurrence_time():.6f}")
    elems = rho_op.celestial_elements()
    print(
        f"  • Delaunay (L,G,H)=({elems.L:.4f},{elems.G:.4f},{elems.H:.4f}) "
        f"e={elems.eccentricity:.4f} Tiss={elems.tisserand:.4f}"
    )

    E_sol = GromovWittenOniricAuditor.solve_kepler(mean_anomaly=1.2, eccentricity=0.3)
    resid = GromovWittenOniricAuditor.kepler_residual(E_sol, 1.2, 0.3)
    assert abs(resid) < 1e-10
    print(f"  • Kepler-Halley E={E_sol:.8f}, resid={resid:.3e}")
    assert GromovWittenOniricAuditor.poincare_birkhoff_count(1, 3, math.pi / 2) == 6
    print("  • Poincaré–Birkhoff p/q=1/3 ⇒ 2q=6 puntos fijos")

    omega_irr = np.array([math.sqrt(2), math.sqrt(3), math.sqrt(5), math.sqrt(7)])
    omega_rat = np.array([1.0, 1.0, 1.0, 1.0])
    R_irr = GromovWittenOniricAuditor.diophantine_residue(omega_irr)
    R_rat = GromovWittenOniricAuditor.diophantine_residue(omega_rat)
    print(f"  • R_irr={R_irr:.6e}, R_rat={R_rat:.6e}")
    T_ret = rho_op.first_return_phase_time()
    print(f"  • Retorno kepleriano exacto T=2π/|ν₀-ν₁|={T_ret:.6f}")

    print("\n>>> ESCENARIO 1: Escenario Onírico Coherente...")
    s1 = OniricScenarioPayload(
        scenario_id="DREAM-POINCARE-001",
        dream_state_flag=True,
        synthetic_cartridge_id="CARTRIDGE-SYNTH-STRESS-01",
        density_matrix=rho,
        dirichlet_energy=0.25,
        betti_1_loop_count=0,
        betti_0_components=1,
        betti_2_cavities=0,
        simulated_risk_factor=0.30,
        timestamp_utc=time.time(),
    )
    cert1 = auditor.audit_dream_scenario(s1)
    print(f"    - ID Certificado          : {cert1.immunization_id}")
    print(f"    - Veredicto Heyting       : {cert1.heyting_verdict.name}")
    print(f"    - Estrato Poincaré        : {cert1.poincare_stratum}")
    print(f"    - GW Relativo Invariant   : {cert1.gw_relative_invariant:.6f}")
    print(f"    - Defect Poincaré-Lefsch  : {cert1.poincare_lefschetz_defect:.6e}")
    print(f"    - Cap. Gromov / ratio     : {cert1.symplectic_capacity:.4f} / {cert1.symplectic_capacity_ratio:.4f}")
    print(f"    - Delaunay (L,G,H)        : {cert1.delaunay_triple}")
    print(f"    - Chirikov s / Greene R   : {cert1.chirikov_overlap:.6f} / {cert1.greene_residue:.6f}")
    print(f"    - Nekhoroshev T_N         : {cert1.nekhoroshev_time:.6e}")
    print(f"    - Hannay θ_H / Tisserand  : {cert1.hannay_angle:.6f} / {cert1.tisserand:.6f}")
    print(f"    - Crowbar / GPIO14        : {cert1.crowbar_triggered} / {cert1.gpio14_signal}")
    assert s1.is_quantum_physical()
    assert cert1.isolation_cert.logical_consistency()

    print("\n>>> ESCENARIO 2: Fuga de aislamiento (dream=False, risk=0.95)...")
    rho_leak = np.eye(4, dtype=np.complex128) / 4.0
    s2 = OniricScenarioPayload(
        scenario_id="DREAM-LEAK-002",
        dream_state_flag=False,
        synthetic_cartridge_id="CARTRIDGE-LEAK-02",
        density_matrix=rho_leak,
        dirichlet_energy=0.20,
        betti_1_loop_count=0,
        betti_0_components=1,
        betti_2_cavities=0,
        simulated_risk_factor=0.95,
        timestamp_utc=time.time(),
    )
    cert2 = auditor.audit_dream_scenario(s2)
    print(f"    - Veredicto / Estrato     : {cert2.heyting_verdict.name} / {cert2.poincare_stratum}")
    print(f"    - Hardware leak / Crowbar : {cert2.isolation_cert.hardware_leak_risk} / {cert2.crowbar_triggered}")
    assert cert2.heyting_verdict == HeytingOmega3.VETOED
    assert cert2.is_immune() is False

    print("\n>>> ESCENARIO 3: Topología sintáctica compleja (b₁ = 5)...")
    s3 = OniricScenarioPayload(
        scenario_id="DREAM-COMPLEX-003",
        dream_state_flag=True,
        synthetic_cartridge_id="CARTRIDGE-COMPLEX-03",
        density_matrix=rho,
        dirichlet_energy=0.30,
        betti_1_loop_count=5,
        betti_0_components=2,
        betti_2_cavities=1,
        simulated_risk_factor=0.30,
        timestamp_utc=time.time(),
    )
    cert3 = auditor.audit_dream_scenario(s3)
    print(f"    - Veredicto / Estrato     : {cert3.heyting_verdict.name} / {cert3.poincare_stratum}")
    assert cert3.heyting_verdict == HeytingOmega3.VETOED

    print("\n>>> SECCIÓN DE POINCARÉ (corte de fase sobre M_c)...")
    N_diag_4 = np.diag(np.arange(1, 5, dtype=float))
    level_c = float(np.trace(rho @ N_diag_4).real)
    section = GromovWittenOniricAuditor.poincare_section_at(level_c, 4)
    t_ret, rho_ret = GromovWittenOniricAuditor.poincare_return_time(
        rho, section, N_diag_4, dt=0.02, max_time=100.0
    )
    print(f"    - Nivel M_c               : {level_c:.6f}")
    print(f"    - Tiempo de retorno       : {t_ret:.6f}  (analítico {T_ret:.6f})")
    rhodot = -1j * (N_diag_4 @ rho - rho @ N_diag_4)
    print(f"    - Transversalidad         : {section.is_transversal(rho, rhodot)}")

    print("\n>>> FUNCIÓN DE MELNIKOV (Poisson hamiltoniano)...")
    mel = GromovWittenOniricAuditor.melnikov_function(rho, rho_leak)
    print(f"    - M(t₀)                   : {mel:.6e}")
    print(f"    - Estabilidad del tubo    : {'intacto' if abs(mel) > 1e-9 else 'roto'}")

    print("\n>>> HOLONOMÍAS BERRY–HANNAY / WILSON...")
    print(f"    - γ_Berry                 : {auditor.berry_phase_along_history:.6f} rad")
    print(f"    - Wilson loop             : {auditor.wilson_loop}")
    print(f"    - Hannay loop             : {auditor.hannay_loop}")

    print("\n>>> AUDITORÍA RETROSPECTIVA DEL REGISTRO...")
    audit = auditor.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<28}: {v}")
    assert audit["registry_integrity_ok"]
    assert audit["merkle_proofs_ok"]
    assert audit["isolation_axioms_ok"]

    print("\n>>> PASAPORTE AGREGADO DEL SOBERANO ONÍRICO...")
    passport = auditor.emit_immunization_passport()
    for k, v in passport.items():
        print(f"    - {k:<22}: {v}")

    print("\n" + "═" * 80)
    print("✓ Verificación del Soberano Auditor Onírico Poincaré v9.1.0 completada.")
    print("═" * 80)
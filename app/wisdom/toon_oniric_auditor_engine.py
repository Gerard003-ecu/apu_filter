# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/wisdom/toon_oniric_auditor_engine.py                                  ║
║ ESTRATO  : WISDOM (V_𝕎) — CIUDADELA DE CRISTAL / AUDITORÍA ONÍRICA TQFT             ║
║ FUNCIÓN  : MOTOR ESPECTRAL AUDITOR DE SUEÑOS Y CAMPO TQFT GROMOV-WITTEN POINCARÉ     ║
║ VERSIÓN  : 9.1.0-Doctoral-Poincare-Celeste-KAM-Nekhoroshev-Hannay-ESP32              ║
║ AUTOR    : APU Wisdom & Metacortex Mathematical Core Architecture                    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICA POINCARÉ
─────────────────────────────────────────────────────────
El `TOONOniricAuditorEngine` es el motor espectral de auditoría topológica y evaluación
de la Teoría Cuántica de Campos Topológicos (TQFT) sobre escenarios sintéticos
contrafactuales generados en la fase REM del ecosistema APU Filter.

ARQUITECTURA EN TRES FASES ANIDADAS (Composición Estricta de Funtores)
──────────────────────────────────────────────────────────────────────
Fase 1 ──► SUSTRATO ONTOLÓGICO, Ω₃, DENSIDAD, MEDIDA ESPECTRAL Y ESTRUCTURAS
           POINCARÉANAS (SECCIÓN, RESONANCIA, RECURRENCIA, ELEMENTOS CELESTES)
              Cierra con SpectralMeasureSeed.extract_spectral_measure.
Fase 2 ──► TQFT, GROMOV-WITTEN RELATIVO, FLUJO DE VON NEUMANN, KAM/NEKHOROSHEV
              Abre  con OniricSpectraEngine.extract_spectral_measure
              (realización estricta de la semilla de Fase 1).
              Cierra con UnsealedOniricTrace (germen de Fase 3).
Fase 3 ──► AUDITOR, SELLO, HOLONOMÍA BERRY-HANNAY, MERKLE, INMUNIZACIÓN
              Abre  con TOONOniricAuditorEngine._seal_and_accumulate
              (único consumidor formal de UnsealedOniricTrace).

CORRECCIÓN CELESTE CANÓNICA (Poincaré 1890, 1892, 1899)
───────────────────────────────────────────────────────
El flujo de Brockett \(\dot\rho=[\rho,[\rho,N]]\) es un flujo *gradiente* de
\(\operatorname{Tr}(\rho N)\) sobre la órbita isoespectral: no preserva la medida de
Liouville y no admite recurrencia de Poincaré. El flujo celeste es el de von Neumann

    \(\dot\rho = -i[N,\rho]\),

que es hamiltoniano respecto de la forma de Kirillov–Kostant–Souriau, preserva el
volumen de Liouville y, si \(N=\operatorname{diag}(\nu_1,\dots,\nu_n)\), se integra
en forma cerrada:

    \(\rho_{jk}(t)=\rho_{jk}(0)\,e^{-i(\nu_j-\nu_k)t}\).

\(\operatorname{Tr}(\rho N)\) es *integral primera* (energía). La sección de Poincaré
es un corte angular transversal *sobre* la superficie de energía, no la superficie
misma. El teorema de recurrencia, el último teorema geométrico, KAM, Nekhoroshev,
Greene, Chirikov y Hannay se aplican a este flujo.
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

logger = logging.getLogger("APU.Wisdom.TOONOniricAuditorEngine.v91")

# ── Importación opcional del agente con fallback autocontenido ──────────────
try:
    from app.agents.wisdom.toon_oniric_auditor_agent import (
        GromovWittenOniricAuditor,
        ImmunizationCertificate,
    )
except ImportError:

    @dataclass(frozen=True, slots=True)
    class ImmunizationCertificate:
        """Fallback local del certificado de inmunización (interfaz pública)."""

        gw_relative_invariant: float = 0.0
        poincare_lefschetz_defect: float = 0.0
        symplectic_capacity_ratio: float = 1.0
        heyting_verdict: Any = None
        is_boundary_consistent: bool = True
        crowbar_triggered: bool = False
        gpio14_signal: str = "LOW"
        proof_merkle_sha512: str = ""

    class GromovWittenOniricAuditor:
        """
        Fallback Gromov–Witten relativo con dualidad Poincaré–Lefschetz y
        cota de no-aplastamiento de Gromov. HeytingOmega3 se resuelve en
        tiempo de llamada (el fallback se define *antes* que el enum).
        """

        ETA_RESONANCE: Final[float] = 0.20
        SYMPLECTIC_RATIO_MAX: Final[float] = 1.25

        @classmethod
        def _project_density(cls, rho: np.ndarray) -> np.ndarray:
            rho_h = 0.5 * (rho + rho.conj().T)
            evals, evecs = la.eigh(rho_h)
            evals = np.clip(evals, 0.0, None)
            s = float(np.sum(evals))
            if s <= 1e-15:
                n = rho_h.shape[0]
                return np.eye(n, dtype=np.complex128) / n
            evals = evals / s
            return (evecs * evals) @ evecs.conj().T

        def evaluate_poincare_lefschetz_gw_invariant(
            self,
            rho_dream: np.ndarray,
            rho_base: np.ndarray,
            boundary_stalk_matrix: np.ndarray,
            betti_1_cycles: int = 0,
            dirichlet_energy: float = 0.0,
            *,
            dream_isolation: bool = True,
            scenario_id: str = "DREAM-EVAL",
            **kwargs: Any,
        ) -> ImmunizationCertificate:
            r"""Invariante relativo Gromov–Witten con dualidad Poincaré–Lefschetz."""
            rho_d = self._project_density(np.asarray(rho_dream, dtype=np.complex128))
            rho_b = self._project_density(np.asarray(rho_base, dtype=np.complex128))
            stalk = np.asarray(boundary_stalk_matrix, dtype=np.complex128)
            evals_d = la.eigvalsh(rho_d)
            evals_d = np.clip(evals_d, 1e-15, None)
            evals_d /= np.sum(evals_d)
            purity = float(np.sum(evals_d ** 2))
            ent_n = -float(np.sum(evals_d * np.log(evals_d)))
            comm = rho_d @ stalk - stalk @ rho_d
            defect = float(la.norm(comm, "fro"))
            n = rho_d.shape[0]
            X = np.diag(np.arange(n, dtype=float))
            mean_X = float(np.real(np.trace(rho_d @ X)))
            var_X = float(np.real(np.trace(rho_d @ X @ X))) - mean_X ** 2
            mean_X_b = float(np.real(np.trace(rho_b @ X)))
            var_X_b = float(np.real(np.trace(rho_b @ X @ X))) - mean_X_b ** 2
            capacity_ratio = float(max(var_X_b, 1e-12) / max(var_X, 1e-12))
            beta1 = max(0, int(betti_1_cycles))
            hom_dim = 1 + beta1
            I_GW = (
                (purity * math.exp(-dirichlet_energy) * math.exp(-ent_n / max(n, 1)))
                / (1.0 + beta1)
            ) * (1.0 / hom_dim)
            ok = (
                I_GW > 0.05
                and dream_isolation
                and capacity_ratio <= self.SYMPLECTIC_RATIO_MAX
            )
            # Resolución tardía: HeytingOmega3 ya existe en el momento de la llamada.
            verdict = HeytingOmega3.COHERENT if ok else HeytingOmega3.VETOED
            return ImmunizationCertificate(
                gw_relative_invariant=float(I_GW),
                poincare_lefschetz_defect=defect,
                symplectic_capacity_ratio=capacity_ratio,
                heyting_verdict=verdict,
                is_boundary_consistent=bool(defect < 0.5),
                crowbar_triggered=bool(verdict == HeytingOmega3.VETOED),
                gpio14_signal="HIGH" if verdict == HeytingOmega3.VETOED else "LOW",
                proof_merkle_sha512=hashlib.sha512(
                    f"{scenario_id}|{I_GW:.10f}".encode("utf-8")
                ).hexdigest(),
            )


__all__ = [
    "HeytingOmega3",
    "DensityOperator",
    "PoincareCelestialElements",
    "SpectralMeasure",
    "PoincareSectionOniric",
    "ResonanceStructure",
    "FloquetSpectrum",
    "SpectralMeasureSeed",
    "ImmunizationPassport",
    "OniricFieldState",
    "OniricSpectraEngine",
    "UnsealedOniricTrace",
    "MerkleInclusionProof",
    "TOONOniricAuditorEngine",
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES ALGEBRAICAS (redes de Poincaré, fracciones continuas, Hermiticidad)
# ══════════════════════════════════════════════════════════════════════════════

_EIG_FLOOR: Final[float] = 1e-15
_TRACE_TOL: Final[float] = 1e-6
_MESH_CAP: Final[int] = 80_000


def _hermitize_trace_one(rho: np.ndarray) -> np.ndarray:
    r"""Proyección ortogonal al cono \(\mathfrak{D}(\mathcal{H}_n)\): \(\rho=\rho^\dagger\), PSD, \(\operatorname{Tr}\rho=1\)."""
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

    Corrección respecto a v9.0.0: el caso \(n=1\) ahora produce \(\{\pm\textit{order}\}\)
    y el residuo nulo produce el vector cero (necesario para ceros internos,
    p.ej. \((2,0)\) y \((-2,0)\)).
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
    r"""Fracción continua \([a_0;a_1,\dots]\) de \(x\in\mathbb{R}\) (algoritmo de Gauss)."""
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
# FASE 1 — RETÍCULO Ω₃, DENSIDAD, MEDIDA ESPECTRAL, ESTRUCTURAS POINCARÉANAS
#          Y SEMILLA Spec
# ══════════════════════════════════════════════════════════════════════════════
# Andamiaje ontológico del topos 𝓣_Ω y del fibrado simpléctico 𝔇(ℋ_n) → S²_Poincaré.
# FASE-1 cierra con SpectralMeasureSeed.extract_spectral_measure, cuyo primer
# consumidor real (y por tanto continuación formal) es OniricSpectraEngine en FASE-2.
# ══════════════════════════════════════════════════════════════════════════════

# ── §1.1 Retículo de Heyting Ω₃ con estratificación celeste ────────────────
class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal \(\Omega_3=\{0\prec 1\prec 2\}=\{\mathsf{VETOED}\prec\mathsf{DEGRADED}\prec\mathsf{COHERENT}\}\).

    Operaciones:
        \(a\wedge b=\min(a,b)\), \(a\vee b=\max(a,b)\),
        \(a\to b=\top\) si \(a\le b\), else \(b\);
        \(\neg_H a=a\to\bot\), \(\neg_B a=\top-a\) (no interna).

    Estratificación celeste de Poincaré / Lyapunov / Floquet
    --------------------------------------------------------
        VETOED   ⇔ separatriz hiperbólica: tubo homoclínico roto (Melnikov \(\neq 0\)
                   y residuo de Greene \(|R|>1\)), exponente de Lyapunov \(\sigma_+>0\),
                   multiplicador de Floquet fuera del círculo unidad.
        DEGRADED ⇔ órbita parabólica / resonancia \(p/q\) de orden bajo: toro KAM
                   colapsado, difusión de Arnold transitoria, \(\sigma\approx 0\),
                   \(|R|\approx 1\) (bifurcación).
        COHERENT ⇔ toro KAM diofantino invariante (teorema KAM), \(\sigma_-<0\),
                   \(|R|<1\) (elíptico), tiempo de Nekhoroshev exponencialmente largo.
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
        r"""Residuo \(a\to b=\bigvee\{c:a\wedge c\le b\}\). En cadena: \(\top\) si \(a\le b\), si no \(b\)."""
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def pseudo_complement(self) -> "HeytingOmega3":
        r"""\(\neg_H a:=a\to\bot\)."""
        return self.implies(HeytingOmega3.VETOED)

    def classical_negation(self) -> "HeytingOmega3":
        r"""Negación booleana extendida por \(2-a\) (no interna)."""
        return HeytingOmega3(2 - int(self))

    def double_negation(self) -> "HeytingOmega3":
        r"""\(\neg\neg_H a\) (no idempotente sobre DEGRADED)."""
        return self.pseudo_complement().pseudo_complement()

    def is_regular(self) -> bool:
        r"""¿\(\neg\neg a=a\)? Verdadero en \(\{\bot,\top\}\), falso en DEGRADED."""
        return self.double_negation() == self

    def excluded_middle_holds(self) -> bool:
        r"""\(a\vee\neg a=\top\) ⇔ \(a\in\{\bot,\top\}\). Falla en DEGRADED."""
        return self.join(self.pseudo_complement()) == HeytingOmega3.COHERENT

    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        r"""Objeto inicial \(\bot\)."""
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        r"""Objeto terminal \(\top\)."""
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        r"""Inclusión \(\mathbb{B}_2\hookrightarrow\Omega_3\)."""
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        r"""Proyección parcial \(\Omega_3\rightharpoonup\mathbb{B}_2\) (rechaza DEGRADED)."""
        if self == HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED no admite proyección fiel a 𝔹₂.")
        return self == HeytingOmega3.COHERENT

    @property
    def verdict(self) -> str:
        return self.name

    @classmethod
    def verify_residuation_axiom(cls) -> bool:
        r"""Verifica \((c\wedge a\le b)\Leftrightarrow(c\le(a\to b))\) \(\forall a,b,c\in\Omega_3\)."""
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
        r"""¿Admite toro KAM invariante persistente? Solo COHERENT."""
        return self == HeytingOmega3.COHERENT

    def lyapunov_exponent_sign(self) -> int:
        r"""Signo del exponente de Lyapunov máximo: \(+1,0,-1\)."""
        return -1 + int(self)

    def symplectic_regime(self) -> str:
        r"""Régimen simpléctico (fuga / cascada resonante / integrable-KAM)."""
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
        \(p=\sqrt{2(G-H)}\cos h\), \(q=-\sqrt{2(G-H)}\sin h\),
        \(\varpi=g+h\).

    Tisserand (respecto a un perturbador de semieje \(a_p=1\)):
        \(T=a_p/a+2\sqrt{a/a_p\,(1-e^2)}\cos i\).

    Integral de Jacobi (heurística espectral):
        \(C_J=2\operatorname{Tr}(\rho N)-\gamma(\rho)\).
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
    Multiplicadores de Floquet de la monodromía \(M\) del mapa de Poincaré
    y residuo de Greene \(R=(2-\operatorname{Tr} M)/4\) (órbita 2-dimensional).

        \(|R|<1\) elíptica, \(|R|=1\) parabólica, \(|R|>1\) hiperbólica.
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

    Invariantes en `__post_init__`: \(\rho=\rho^\dagger\), \(\operatorname{spec}(\rho)\subset[-\varepsilon,1+\varepsilon]\),
    \(|\operatorname{Tr}\rho-1|\le 10^{-6}\).

    Estructura simpléctica (KKS)
    ----------------------------
    La órbita coadjunta \(U(n)\cdot\rho\subset\mathfrak{u}(n)^*\) lleva
        \(\omega_{\mathrm{KKS}}(X,Y)_\rho=\langle\rho,[X,Y]\rangle\).
    Autovalores \(\lambda_i\) = acciones de Liouville \(J_i\); fases
    \(\theta_i=\arg\langle u_i|N|u_i\rangle\) = ángulos conjugados.

    Flujo hamiltoniano exacto
    -------------------------
    Si \(N=\operatorname{diag}(\nu)\), \(\rho_{jk}(t)=\rho_{jk}(0)e^{-i(\nu_j-\nu_k)t}\).
    Este flujo preserva \(\omega_{\mathrm{KKS}}\) y el volumen de Liouville
    (teorema de Liouville–Poincaré: el invariante integral absoluto
    \(\iint dp\wedge dq\) es constante).
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

    def spectrum(self, floor: float = _EIG_FLOOR) -> np.ndarray:
        r"""Autovalores normalizados a suma 1 (orden de `eigh`: creciente)."""
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
        r"""\(\Delta\lambda=\lambda_{\max}-\lambda_{\max-1}\) (0 si \(n=1\))."""
        lam = np.sort(self.spectrum())
        if lam.size < 2:
            return 0.0
        return float(lam[-1] - lam[-2])

    def cstar_residual(self) -> float:
        r"""Residuo de la identidad \(C^*\): \(\big|\|\rho^\dagger\rho\|_2-\|\rho\|_2^2\big|\)."""
        rho = self.matrix
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

    @classmethod
    def from_array(cls, rho: np.ndarray, atol: float = 1e-8) -> "DensityOperator":
        r"""Proyección al cono \(\mathfrak{D}(\mathcal{H}_n)\) y envoltura inmutable."""
        return cls(matrix=_hermitize_trace_one(rho), atol=atol)

    # ── Geometría simpléctica: acciones, ángulos, Delaunay, Poincaré ─────
    def action_variables(self) -> np.ndarray:
        r"""Acciones de Liouville \(J_i:=\lambda_i(\rho)\) decrecientes."""
        return np.sort(self.spectrum())[::-1]

    def angle_variables(self, N_diag: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Ángulos canónicos \(\theta_i:=\arg\langle u_i|N|u_i\rangle\in(-\pi,\pi]\)."""
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
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
        r"""
        Frecuencias keplerianas exactas del flujo de von Neumann:
            \(\omega_{jk}=\nu_j-\nu_k\), compactadas como vector \(\omega_j=\nu_j\cdot\bar E\),
        más el análogo metabólico \(J_i\cdot\operatorname{Tr}(\rho N)\) (v9).
        Se devuelve el vector metabólico (compatibilidad) — usar
        `keplerian_frequency_matrix` para el espectro de frecuencias exacto.
        """
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
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
            \(H=\operatorname{Tr}(\rho N)\cdot\cos\Delta\lambda\),
        con \(H\le G\le L\) en el régimen kepleriano no degenerado.
        """
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
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
        e = math.sqrt(max(1.0 - (G ** 2) / max(L ** 2, 1e-30), 0.0))
        i = (
            math.acos(max(min(_safe_div(H, G, 1.0), 1.0), -1.0))
            if G > 1e-30
            else 0.0
        )
        return float(e), float(i)

    def celestial_elements(
        self, N_diag: Optional[np.ndarray] = None
    ) -> PoincareCelestialElements:
        r"""
        Carta completa Delaunay → Poincaré, Tisserand y Jacobi.

        Convención angular: \(l=\theta_0\) (anomalía media espectral),
        \(g=\theta_1-\theta_0\), \(h=\theta_2-\theta_1\) (cíclicos en \(n\ge 3\);
        se anulan las diferencias no definidas).
        """
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
        L, G, H = self.delaunay_triple(N)
        e, inc = self.eccentricity_inclination(N)
        th = self.angle_variables(N)
        l = float(th[0]) if n >= 1 else 0.0
        g = float(th[1] - th[0]) if n >= 2 else 0.0
        h = float(th[2] - th[1]) if n >= 3 else 0.0
        varpi = g + h
        two_LG = max(2.0 * (L - G), 0.0)
        two_GH = max(2.0 * (G - H), 0.0)
        rt_e = math.sqrt(two_LG)
        rt_i = math.sqrt(two_GH)
        xi = rt_e * math.cos(varpi)
        eta = -rt_e * math.sin(varpi)
        p = rt_i * math.cos(h)
        q = -rt_i * math.sin(h)
        energy = float(np.trace(self.matrix @ N).real)
        a = max(L * L, 1e-30)
        a_p = 1.0
        tisserand = a_p / a + 2.0 * math.sqrt(max(a / a_p * (1.0 - e * e), 0.0)) * math.cos(inc)
        jacobi = 2.0 * energy - self.purity()
        mean_motion = _safe_div(1.0, a ** 1.5, 0.0)  # 3ª ley de Kepler, μ=1
        return PoincareCelestialElements(
            L=float(L),
            G=float(G),
            H=float(H),
            mean_anomaly=l,
            arg_periapsis=g,
            long_node=h,
            Lambda=float(L),
            mean_longitude=l + g + h,
            xi=float(xi),
            eta=float(eta),
            p=float(p),
            q=float(q),
            eccentricity=float(e),
            inclination=float(inc),
            tisserand=float(tisserand),
            jacobi_integral=float(jacobi),
            energy=float(energy),
            mean_motion=float(mean_motion),
        )

    def poincare_integral_invariant(self) -> float:
        r"""
        Invariante integral relativo de Poincaré \(\sum_i J_i\theta_i\)
        (acción de Hilbert–Poincaré–Cartan \(\oint p\,dq\) en carta de Darboux).
        """
        J = self.action_variables()
        th = self.angle_variables()
        return float(np.dot(J, th))

    # ── Flujo hamiltoniano exacto (von Neumann) ──────────────────────────
    def hamiltonian_evolve(
        self, t: float, N_diag: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""
        Flujo de von Neumann exacto \(\rho(t)=e^{-iNt}\rho e^{iNt}\).
        Si \(N\) es diagonal: \(\rho_{jk}(t)=\rho_{jk}(0)e^{-i(\nu_j-\nu_k)t}\).
        """
        nu = _diag_potential(self.dimension, N_diag)
        phase = np.exp(-1j * float(t) * (nu[:, None] - nu[None, :]))
        rho_t = self.matrix * phase
        return 0.5 * (rho_t + rho_t.conj().T)

    def first_return_phase_time(
        self,
        i: int = 0,
        j: int = 1,
        N_diag: Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Tiempo de primer retorno del ángulo \(\phi=\arg\rho_{ij}\) a \(\phi\equiv\phi_0\pmod{2\pi}\).
        Exacto: \(T=2\pi/|\nu_i-\nu_j|\) (periodo kepleriano del par).
        """
        nu = _diag_potential(self.dimension, N_diag)
        n = self.dimension
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return float("inf")
        omega = float(nu[i] - nu[j])
        if abs(omega) < 1e-15:
            return float("inf")
        return float(2.0 * math.pi / abs(omega))

    # ── Función de Wigner vectorizada (FFT) y Husimi ─────────────────────
    def wigner_function(self) -> np.ndarray:
        r"""
        Función de Wigner discreta sobre \(\mathbb{Z}_n\times\mathbb{Z}_n\):
            \(W_\rho(q,p)=\frac1n\sum_x e^{-2\pi i px/n}\rho_{q+x,\,q-x}\).
        Implementación FFT: \(W[q,:]=\mathrm{FFT}_x(\rho_{q+x,q-x})/n\).
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
        r"""
        Función de Husimi \(Q_\rho=\mathcal{G}_\sigma * W_\rho\) (suavizado gaussiano
        del Wigner; \(Q\ge 0\) en el límite continuo). Kernel periódico discreto.
        """
        W = self.wigner_function()
        n = self.dimension
        ax = np.arange(n)
        dx = np.minimum(ax, n - ax)
        g = np.exp(-0.5 * (dx ** 2) / max(sigma * sigma, 1e-12))
        g = g / np.sum(g)
        # Convolución separable 2D vía FFT
        G = np.outer(g, g)
        Wf = np.fft.fft2(W)
        Gf = np.fft.fft2(np.fft.ifftshift(G))
        Q = np.real(np.fft.ifft2(Wf * Gf))
        s = float(np.sum(Q))
        return Q / s if abs(s) > 1e-15 else Q

    def wigner_marginals(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""Marginales \((\langle q|\rho|q\rangle,\;\langle p|\rho|p\rangle_{\mathrm{Wigner}})\)."""
        W = self.wigner_function()
        return np.sum(W, axis=1), np.sum(W, axis=0)

    def wigner_negativity(self) -> float:
        r"""Negatividad de Wigner \(N_W(\rho)=\sum|W|-1\). \(N_W=0\) ⇔ clásico."""
        W = self.wigner_function()
        return float(np.sum(np.abs(W)) - 1.0)

    # ── Capacidad simpléctica (Gromov) ───────────────────────────────────
    def symplectic_capacity_gromov(self) -> float:
        r"""
        Capacidad de Gromov heurística \(c_G(\rho)=4/(\mathrm{Var}_q+\mathrm{Var}_p)\).
        No-aplastamiento: \(c_G(B^{2n}(r))=\pi r^2\le\pi R^2=c_G(Z^{2n}(R))\).
        """
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

    # ── Flujo espectral de Chern, Maslov, recurrencia, Berry–Hannay ──────
    def spectral_flow_chern(self, t_max: float = 1.0, n_steps: int = 32) -> float:
        r"""Flujo espectral de Atiyah–Patodi–Singer de \(H(t)=H_0+tN\), normalizado."""
        H0 = self.matrix
        n = self.dimension
        N = np.diag(_diag_potential(n))
        evals_init = la.eigvalsh(H0)
        H_fin = H0 + t_max * N
        evals_fin = la.eigvalsh(0.5 * (H_fin + H_fin.conj().T))
        flow = 0
        for k in range(n):
            if evals_fin[k] > evals_init[k] + 1e-9:
                flow += 1
            elif evals_fin[k] < evals_init[k] - 1e-9:
                flow -= 1
        return float(flow) / float(max(1, 2 * n_steps))

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
        Tiempo de recurrencia de Poincaré.

        Heurística espectral: \(\tau_{\mathrm{spec}}=1/\min_{i\neq j}|\lambda_i-\lambda_j|\).
        Cota de volumen (Kac): \(\tau_{\mathrm{Kac}}\simeq 1/\mathrm{Vol}_L(\lambda)=\bigl(\prod\lambda_i\bigr)^{-1}\).
        Cota kepleriana: \(T=2\pi/\min_{j\neq k}|\nu_j-\nu_k|\).
        Se devuelve el mínimo finito (recurrencia más rápida observable).
        """
        lam = np.sort(self.spectrum())
        tau_spec = float("inf")
        if lam.size >= 2:
            min_gap = float(np.min(np.abs(np.diff(lam))))
            if min_gap >= 1e-15:
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
        r"""
        Fase de Berry–Pancharatnam a lo largo de \(\gamma=(\rho_0,\dots,\rho_m)\):
            \(\gamma_B=\sum_k\arg\langle\psi_k|\psi_{k+1}\rangle\),
        con \(\psi_k\) el autovector dominante. Si `closed`, se añade el cierre
        \(\arg\langle\psi_m|\psi_0\rangle\) (holonomía \(U(1)\)).
        """
        if len(curve) < 2:
            return 0.0
        phis: List[np.ndarray] = []
        for rho_k in curve:
            rho_h = 0.5 * (rho_k + rho_k.conj().T)
            _, evecs = la.eigh(rho_h)
            phis.append(evecs[:, -1])
        total_arg = 0.0
        nphi = len(phis)
        last = nphi if not closed else nphi
        for k in range(last):
            a = phis[k]
            b = phis[(k + 1) % nphi] if closed else (phis[k + 1] if k + 1 < nphi else phis[k])
            if not closed and k + 1 >= nphi:
                break
            total_arg += float(np.angle(np.vdot(a, b)))
        if not closed:
            total_arg = 0.0
            for k in range(nphi - 1):
                total_arg += float(np.angle(np.vdot(phis[k], phis[k + 1])))
        return float(total_arg)

    def hannay_angle(self, N_diag: Optional[np.ndarray] = None) -> float:
        r"""
        Ángulo de Hannay (análogo clásico de Berry) en carta acción-ángulo:
            \(\theta_H=-\sum_i\partial_{\lambda}\oint J_i\,d\theta_i\simeq -\sum_i J_i\cdot 0 + \sum\Delta\theta_i\),
        reducido aquí al invariante relativo \(\sum J_i\theta_i\) módulo \(2\pi\).
        """
        I = self.poincare_integral_invariant()
        return float((I + math.pi) % (2.0 * math.pi) - math.pi)

    def chirikov_overlap(self, N_diag: Optional[np.ndarray] = None) -> float:
        r"""
        Parámetro de solapamiento de Chirikov
            \(s=(\Delta\omega_1+\Delta\omega_2)/(2|\omega_1-\omega_2|)\),
        con semianchos \(\Delta\omega_i\simeq 2\sqrt{|J_i\varepsilon|}\) y
        \(\varepsilon=\|[ \rho,N]\|_F\). \(s\gtrsim 1\) ⇒ caos por solapamiento.
        """
        n = self.dimension
        if n < 2:
            return 0.0
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
        comm = self.matrix @ N - N @ self.matrix
        eps = float(la.norm(comm, "fro"))
        J = self.action_variables()
        omega = nu * float(np.trace(self.matrix @ N).real)
        # pares adyacentes en frecuencia
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
        r"""
        Suma de Brjuno \(B(\alpha)=\sum_{k\ge 0}(\log q_{k+1})/q_k\) del cociente
        de frecuencias dominante \(\alpha=\omega_1/\omega_0\). \(B<\infty\) es la
        condición de Siegel–Brjuno de linealización analítica (más débil que KAM
        diofantino).
        """
        n = self.dimension
        if n < 2:
            return 0.0
        nu = _diag_potential(n, N_diag)
        if abs(nu[0]) < 1e-15:
            return float("inf")
        alpha = float(nu[1] / nu[0])
        cf = _continued_fraction(alpha, max_terms=max_cf)
        conv = _convergents(cf)
        if len(conv) < 2:
            return 0.0
        acc = 0.0
        for k in range(len(conv) - 1):
            qk = max(abs(conv[k][1]), 1)
            qk1 = max(abs(conv[k + 1][1]), 1)
            acc += math.log(qk1) / qk
        return float(acc)

    def lyapunov_characteristic_exponents(
        self, N_diag: Optional[np.ndarray] = None, dt: float = 0.05, n_steps: int = 64
    ) -> np.ndarray:
        r"""
        Exponentes característicos de Poincaré (Lyapunov) del flujo de von Neumann
        linealizado. El flujo unitario tiene espectro de Lyapunov idénticamente nulo
        (conservativo). Se reporta el espectro numérico de
        \(\frac1T\log\operatorname{spec}|U(T)|\) con \(U(T)=e^{-iNT}\otimes e^{iNT}\),
        que debe concentrarse en \(\{0\}\) — desviaciones miden error de proyección.
        """
        n = self.dimension
        nu = _diag_potential(n, N_diag)
        T = dt * n_steps
        # Autovalores de Ad_{e^{-iNT}}: e^{-i(ν_j-ν_k)T} ⇒ log-módulos = 0.
        lce = np.zeros(n, dtype=float)
        # Diagnóstico: deriva de isospectralidad bajo proyección numérica.
        rho = self.matrix.copy()
        phase_dt = np.exp(-1j * dt * (nu[:, None] - nu[None, :]))
        for _ in range(n_steps):
            rho = 0.5 * ((rho * phase_dt) + (rho * phase_dt).conj().T)
            tr = float(np.trace(rho).real)
            if tr > 0:
                rho = rho / tr
        drift = float(la.norm(rho - self.hamiltonian_evolve(T, np.diag(nu)), "fro"))
        lce[0] = math.log(max(drift, 1e-30)) / max(T, 1e-12)
        return lce


# ── §1.5 Medida espectral con estructura Liouvilliana ─────────────────────
@dataclass(frozen=True, slots=True)
class SpectralMeasure:
    r"""
    Medida espectral \(\lambda\in\Delta^{n-1}\) con observables derivados.

    Volumen de Liouville discreto \(\mathrm{Vol}_L(\lambda)=\prod_i\lambda_i\).
    Métrica de Fisher–Rao \(g_{ij}=\delta_{ij}/\lambda_i\) (en el interior).
    """

    eigenvalues: np.ndarray
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
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
        r"""Norma Fisher–Rao \(\|v\|_g^2=\sum_i v_i^2/\lambda_i\) en \(T_\lambda\Delta\)."""
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
        if min_gap < 1e-15:
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
    Sección de Poincaré *transversal* al flujo hamiltoniano de von Neumann,
    contenida en la superficie de energía \(M_c=\{\operatorname{Tr}(\rho N)=c\}\).

    Construcción (Poincaré 1892, *Méthodes nouvelles* I):
        \(M_c\) es invariante (\(\operatorname{Tr}(\rho N)\) es integral primera).
        El corte \(\Sigma=\{ \rho\in M_c:\arg\rho_{ij}=\phi_*\}\) es transversal
        ssi \(\nu_i\neq\nu_j\). El mapa de primer retorno \(P:\Sigma\to\Sigma\)
        es un twist que preserva el área de Liouville (Poincaré–Cartan).

    `level`   : energía \(c=\operatorname{Tr}(\rho N)\).
    `phase_cut`: ángulo \(\phi_*\) del par \((i_cut,j_cut)\).
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
        z = rho[self.i_cut, self.j_cut]
        return float(np.angle(z))

    def signed_phase(self, rho: np.ndarray) -> float:
        r"""Fase reducida a \((-\pi,\pi]\) relativa a `phase_cut`."""
        d = self.phase(rho) - self.phase_cut
        return float((d + math.pi) % (2.0 * math.pi) - math.pi)

    def crosses(self, rho_before: np.ndarray, rho_after: np.ndarray) -> bool:
        r"""¿El segmento cruza el corte de fase transversalmente?"""
        d0 = self.signed_phase(rho_before)
        d1 = self.signed_phase(rho_after)
        return (d0 * d1) < 0.0

    def is_transversal(self, rho: np.ndarray, rhodot: np.ndarray) -> bool:
        r"""
        Transversalidad: \(\frac{d}{dt}\arg\rho_{ij}\neq 0\).
        Equivale a \(\operatorname{Im}(\overline{\rho_{ij}}(\dot\rho)_{ij})\neq 0\).
        """
        z = rho[self.i_cut, self.j_cut]
        zdot = rhodot[self.i_cut, self.j_cut]
        g = float(np.imag(np.conj(z) * zdot))
        return abs(g) > self.transversality_tol

    def twist_condition(self, omega: np.ndarray) -> bool:
        r"""
        Condición de twist de Poincaré–Birkhoff / Moser:
        \(\partial\theta_1/\partial J_1\neq 0\), i.e. el número de rotación
        depende monótonamente de la acción. Se aproxima por \(\mathrm{std}(\omega)>0\).
        """
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

    Campos:
        vector, order, residue_min, kam_residue, is_kam_safe,
        continued_fraction (convergente \(p/q\) dominante),
        brjuno_sum, chirikov_overlap.
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


# ── §1.7 Protocolo del pasaporte inmutable ────────────────────────────────
@runtime_checkable
class ImmunizationPassport(Protocol):
    r"""Protocolo estructural del pasaporte de inmunización."""

    immunization_hash: str
    heyting_verdict: HeytingOmega3
    gromov_witten_invariant: float

    def is_immune(self) -> bool:
        ...


# ── §1.8 Estado onírico con campos celestes completos ─────────────────────
@dataclass(frozen=True, slots=True)
class OniricFieldState:
    r"""
    Estado onírico tras la auditoría, con métricas de Poincaré–Lefschetz,
    rigidez de Gromov, recurrencia, Maslov, Wigner, KAM, Hannay, Greene,
    Chirikov, Nekhoroshev, Tisserand y estrato \(\Omega_3\) + Crowbar ESP32.
    """

    cycle_id: str
    scenario_id: str
    dream_isolation_flag: bool
    density_matrix: np.ndarray
    dirichlet_energy: float
    dirac_total_variation: float
    gromov_witten_invariant: float
    tqft_amplitude: float
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    cstar_residual: float
    betti_0: int
    betti_1_loops: int
    betti_2: int
    euler_characteristic: int
    heyting_verdict: HeytingOmega3
    immunization_hash: str
    timestamp_utc: float
    holonomy_partial: float
    wilson_phase: complex
    poincare_lefschetz_defect: float = 0.0
    symplectic_capacity_ratio: float = 1.0
    is_boundary_consistent: bool = True
    crowbar_triggered: bool = False
    gpio14_signal: str = "LOW"
    proof_merkle_sha512: str = ""
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_recurrence_time: float = float("inf")
    maslov_index: int = 0
    resonance_order: int = 0
    kam_residue: float = float("inf")
    lyapunov_sign: int = -1
    poincare_stratum: str = "kam-torus-invariant"
    # ── Campos celestes v9.1.0 ──────────────────────────────────────────
    hannay_angle: float = 0.0
    greene_residue: float = 0.0
    chirikov_overlap: float = 0.0
    nekhoroshev_time: float = float("inf")
    brjuno_sum: float = 0.0
    tisserand: float = 0.0
    jacobi_integral: float = 0.0
    floquet_radius: float = 1.0
    mean_motion: float = 0.0
    eccentricity: float = 0.0

    def __post_init__(self) -> None:
        rho = np.asarray(self.density_matrix, dtype=np.complex128)
        object.__setattr__(self, "density_matrix", np.array(rho, copy=True))
        if self.betti_0 < 0 or self.betti_1_loops < 0 or self.betti_2 < 0:
            raise ValueError("Los números de Betti deben ser ≥ 0.")
        if self.gromov_witten_invariant < 0.0:
            raise ValueError("I_GW debe ser ≥ 0.")

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        r"""¿\(\rho\) es hermitiano, PSD y de traza 1?"""
        rho = self.density_matrix
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        if np.any(la.eigvalsh(rho) < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    def is_topologically_consistent(self) -> bool:
        r"""\(\chi=\beta_0-\beta_1+\beta_2\) y coherencia de estrato con la frontera."""
        if self.euler_characteristic != (self.betti_0 - self.betti_1_loops + self.betti_2):
            return False
        if self.heyting_verdict == HeytingOmega3.VETOED:
            if not self.dream_isolation_flag:
                return True
            return self.betti_1_loops > 3 or (not self.is_boundary_consistent)
        return True

    def is_immune(self) -> bool:
        r"""
        Inmunidad: aislamiento \(\wedge\) \(\neg\)VETOED \(\wedge\) física cuántica
        \(\wedge\) topología \(\wedge\) no-crowbar \(\wedge\) Gromov \(\wedge\) KAM
        \(\wedge\) Chirikov \(s<1\) \(\wedge\) Greene \(|R|\le 1\).
        """
        return (
            self.dream_isolation_flag
            and self.heyting_verdict != HeytingOmega3.VETOED
            and self.is_quantum_physical()
            and self.is_topologically_consistent()
            and self.is_boundary_consistent
            and self.symplectic_capacity_ratio <= 1.25
            and not self.crowbar_triggered
            and self.kam_residue >= 1.0
            and self.chirikov_overlap < 1.0
            and abs(self.greene_residue) <= 1.0 + 1e-9
        )

    def passport_prefix(self, n: int = 16) -> str:
        return self.immunization_hash[:n]


# ── §1.9 Semilla abstracta Spec — cierra FASE-1 ───────────────────────────
class SpectralMeasureSeed(ABC):
    r"""
    Semilla del endofuntor \(\mathrm{Spec}:\rho\mapsto(\lambda,\gamma,S,\Delta\lambda,\varepsilon_{C^*})\).

    Esta ABC cierra el andamiaje de FASE-1. FASE-2 **continúa** exactamente
    aquí: `OniricSpectraEngine.extract_spectral_measure` es el primer método
    real de FASE-2 y realiza la semilla.
    """

    @abstractmethod
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        r"""
        Extrae la medida espectral \(\lambda\in\Delta^{n-1}\) del estado \(\rho\in\mathfrak{D}(\mathcal{H}_n)\).
        CONTINÚA EN FASE-2: OniricSpectraEngine.extract_spectral_measure.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — TQFT, GROMOV-WITTEN RELATIVO, VON NEUMANN, KAM, RESONANCIAS
#          Y TRAZA ABIERTA
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método de esta fase ES la realización del último de
# FASE-1 (extract_spectral_measure). El último artefacto de FASE-2
# (UnsealedOniricTrace, producido por evaluate_dream_spectrum) es el germen
# formal de FASE-3 (_seal_and_accumulate).
# ══════════════════════════════════════════════════════════════════════════════
class OniricSpectraEngine(SpectralMeasureSeed):
    r"""
    Motor espectral onírico: dualidad Poincaré–Lefschetz, rigidez de Gromov,
    flujo de von Neumann exacto, Wigner/Husimi, resonancias, KAM, Brjuno,
    Nekhoroshev, Chirikov, Greene, Maslov, Birkhoff–Dulac, Melnikov, Kepler.

    Funtor \(F_{\mathrm{spectral}}:\mathfrak{D}(\mathcal{H}_n)\to(\lambda,W,R,\mu,\mathrm{KAM},\dots)\).
    """

    BETTI_MAX: Final[int] = 3
    GW_MIN: Final[float] = 0.05
    DIRICHLET_MAX: Final[float] = 0.85
    DIRAC_TV_MAX: Final[float] = 1.50
    EIGENVALUE_FLOOR: Final[float] = _EIG_FLOOR
    DIRICHLET_CONSISTENCY_TOL: Final[float] = 1e-3
    DIOPHANTINE_GAMMA: Final[float] = 1e-3
    DIOPHANTINE_TAU: Final[float] = 1.5
    DIOPHANTINE_KMAP: Final[int] = 4
    NEKHOROSHEV_EPS_STAR: Final[float] = 0.15

    def __init__(self, gw_auditor: Optional[GromovWittenOniricAuditor] = None) -> None:
        self.gw_auditor = gw_auditor or GromovWittenOniricAuditor()

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
        CONTINUACIÓN FORMAL de SpectralMeasureSeed.extract_spectral_measure.

        Extrae \(\lambda\in\Delta^{n-1}\), pureza \(\gamma\), entropía \(S_{\mathrm{vN}}\),
        brecha \(\Delta\lambda\) y residuo \(C^*\) del estado \(\rho\).
        """
        rho_h = _hermitize_trace_one(rho)
        eigvals = la.eigvalsh(rho_h)
        lam = self._project_spectrum(eigvals)
        purity = float(np.sum(lam ** 2))
        entropy = -float(np.sum(lam * np.log(lam)))
        gap = float(lam[-1] - lam[-2]) if lam.size >= 2 else 0.0
        op = float(la.norm(rho_h.conj().T @ rho_h, 2))
        nrm = float(la.norm(rho_h, 2))
        cstar = abs(op - nrm * nrm)
        return SpectralMeasure(
            eigenvalues=lam,
            purity=purity,
            von_neumann_entropy=entropy,
            spectral_gap=gap,
            dimension=int(lam.size),
            cstar_residual=cstar,
        )

    # ── §2.2 Energías de Dirichlet y variación total de Dirac ────────────
    @classmethod
    def compute_dirichlet_energy(cls, eigvals: np.ndarray) -> float:
        r"""\(E_D(\lambda)=\frac12\sum_i(\lambda_{i+1}-\lambda_i)^2\)."""
        lam = np.sort(cls._project_spectrum(eigvals))
        if lam.size < 2:
            return 0.0
        return 0.5 * float(np.sum(np.diff(lam) ** 2))

    @classmethod
    def compute_dirac_total_variation(cls, eigvals: np.ndarray) -> float:
        r"""\(\mathrm{TV}(\lambda)=\sum_i|\lambda_{i+1}-\lambda_i|\)."""
        lam = np.sort(cls._project_spectrum(eigvals))
        if lam.size < 2:
            return 0.0
        return float(np.sum(np.abs(np.diff(lam))))

    # ── §2.3 Análisis celeste: Wigner, Gromov, resonancias, KAM, Brjuno ─
    @classmethod
    def compute_wigner_function(cls, rho: np.ndarray) -> np.ndarray:
        r"""Wigner discreto (envoltura de DensityOperator, proyección previa)."""
        return DensityOperator.from_array(rho).wigner_function()

    @classmethod
    def symplectic_capacity_gromov(cls, rho: np.ndarray) -> float:
        r"""Capacidad de Gromov heurística \(c_G(\rho)=4/(\mathrm{Var}_q+\mathrm{Var}_p)\)."""
        return DensityOperator.from_array(rho).symplectic_capacity_gromov()

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
        \(R\ge 1\) ⇔ \(\omega\) es \((\gamma,\tau)\)-diofantino (toro KAM viable).

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
            val = abs(float(np.dot(k, omega))) / max(denom, 1e-30)
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
        r"""Indicador de toro KAM: \((R\ge 1,\; R)\)."""
        R = cls.diophantine_residue(omega)
        return (R >= 1.0), float(R)

    @classmethod
    def nekhoroshev_stability_time(
        cls,
        epsilon: float,
        n_dof: int,
        eps_star: Optional[float] = None,
    ) -> float:
        r"""
        Tiempo de estabilidad de Nekhoroshev
            \(T_N\sim\exp\bigl((\varepsilon^*/\varepsilon)^{1/(2n)}\bigr)\),
        válido para \(\varepsilon<\varepsilon^*\) en sistemas casi integrables
        exponencialmente estables (no meros KAM).
        """
        eps_star = cls.NEKHOROSHEV_EPS_STAR if eps_star is None else float(eps_star)
        eps = max(float(epsilon), 1e-30)
        if eps >= eps_star:
            return 1.0
        expo = (eps_star / eps) ** (1.0 / max(2 * max(n_dof, 1), 1))
        # saturación numérica
        expo = min(expo, 80.0)
        return float(math.exp(expo))

    # ── §2.4 Mapa de retorno de Poincaré (flujo de von Neumann exacto) ───
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

        Si el par de corte tiene \(\omega=\nu_i-\nu_j\neq 0\), el retorno es
        analítico: \(T=2\pi/|\omega|\). El bucle de muestreo se conserva como
        verificación de transversalidad y para cortes no cartanianos.
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
        # Muestreo exacto (no Euler): avanza por dt con el propagador cerrado.
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

    @classmethod
    def monodromy_and_greene(
        cls,
        rho: np.ndarray,
        section: PoincareSectionOniric,
        N_diag: np.ndarray,
        eps: float = 1e-6,
    ) -> FloquetSpectrum:
        r"""
        Monodromía 2D del mapa de Poincaré en el plano \((\mathrm{Re}\rho_{ij},\mathrm{Im}\rho_{ij})\)
        por diferencias finitas del retorno exacto, residuo de Greene y radio de Floquet.

        El flujo lineal integrable produce una rotación: \(\operatorname{Tr} M=2\cos(\omega T)=2\),
        \(R=0\) (elíptico degenerado de periodo exacto). Perturbaciones numéricas
        empujan \(R\) fuera de 0.
        """
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
            # Perturbación hermitiana del par (i,j)
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
        # Simplecticidad discreta: det M ≃ 1
        detM = float(np.linalg.det(M).real) if np.isrealobj(np.linalg.det(M)) else float(np.real(np.linalg.det(M)))
        return FloquetSpectrum(
            multipliers=ev,
            spectral_radius=radius,
            greene_residue=float(R),
            trace_monodromy=tr,
            is_elliptic=bool(abs(R) < 1.0),
            is_symplectic=bool(abs(detM - 1.0) < 0.15),
        )

    # ── §2.5 Índice de Maslov y recurrencia ──────────────────────────────
    @classmethod
    def maslov_index(cls, rho: np.ndarray, N_diag: Optional[np.ndarray] = None) -> int:
        r"""Índice de Maslov de \(\rho\) respecto de \(N\)."""
        return DensityOperator.from_array(rho).maslov_index(N_diag)

    @classmethod
    def poincare_recurrence_time(cls, rho: np.ndarray) -> float:
        r"""Tiempo de recurrencia de Poincaré (spec / Kac / Kepler)."""
        return DensityOperator.from_array(rho).poincare_recurrence_time()

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

        En un punto fijo hamiltoniano el campo es \(\dot\rho=-i[N,\rho]\).
        La linealización es \(\mathrm{ad}_{-iN}\). Los términos resonantes de
        orden \(k\) se obtienen proyectando el polinomio de homología
        \(\mathcal{L}_{H_0}W_k=H_k^{\mathrm{nres}}\) al núcleo (pequeños divisores
        \(\langle m,\omega\rangle=0\)).

        Se devuelve \(X_1=\mathrm{ad}_N^2(\rho)\) filtrado por Schur respecto de
        \(\rho\) (compatibilidad v9) más, si `order>=2`, el residuo cuadrático
        \(P_2=[N,[N,[N,\rho]]]\) proyectado al complemento resonante.
        """
        rho_h = _hermitize_trace_one(rho)
        n = rho_h.shape[0]
        nu = _diag_potential(n, N_diag)
        N = np.diag(nu)
        comm1 = N @ rho_h - rho_h @ N
        X1 = N @ comm1 - comm1 @ N
        tr_num = float(np.real(np.trace(X1 @ rho_h.conj().T)))
        tr_den = float(np.real(np.trace(rho_h @ rho_h.conj().T)))
        if tr_den > 1e-15:
            X1 = X1 - (tr_num / tr_den) * rho_h
        if order >= 2:
            comm2 = N @ X1 - X1 @ N
            X2 = N @ comm2 - comm2 @ N
            tr2 = float(np.real(np.trace(X2 @ rho_h.conj().T)))
            if tr_den > 1e-15:
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
        \(H_0=\operatorname{Tr}(\rho N)\), \(H_1=\operatorname{Tr}(\rho\rho_{\mathrm{base}})\),
        \(\{H_0,H_1\}= -i\operatorname{Tr}([N,\rho]\,\rho_{\mathrm{base}})\).

        \(M\neq 0\) ⇒ persistencia del tubo homoclínico; \(M=0\) ⇒ ruptura
        (caos transitorio, estrato VETOED).
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

    # ── §2.8 Teorema de Poincaré–Birkhoff (último teorema geométrico) ────
    @classmethod
    def poincare_birkhoff_count(
        cls, p: int, q: int, twist_angle: float
    ) -> int:
        r"""
        Último teorema geométrico de Poincaré (Birkhoff 1913):
        un twist que preserva área en el anillo, con rotación \(\theta\in(0,2\pi)\)
        y \(0<p/q<1\) racional, posee al menos \(2q\) puntos periódicos de periodo \(q\).
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
        r"""
        Condición de twist de Moser: \(\det(\partial\omega/\partial J)\neq 0\).
        Se aproxima por independencia lineal discreta de \(\Delta\omega/\Delta J\).
        """
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

    # ── §2.9 Ecuación de Kepler (Halley + arranque de Danby) ─────────────
    @classmethod
    def solve_kepler(
        cls,
        mean_anomaly: float,
        eccentricity: float,
        tol: float = 1e-12,
        max_iter: int = 50,
    ) -> float:
        r"""
        Kepler \(E-e\sin E=M\) por el método de Halley con arranque de Danby:
            \(E_0=M+e\sin M/(1-\sin(M+e)+\sin M)\),
            \(E\leftarrow E-f/(f'-ff''/(2f'))\).
        """
        e = float(eccentricity)
        M = float(mean_anomaly)
        if not (0.0 <= e < 1.0):
            raise ValueError("Excentricidad debe estar en [0, 1).")
        # reducir M a (−π, π]
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
            den = fp - 0.5 * f * fpp / max(fp, 1e-30)
            dE = f / max(den, 1e-30)
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
        r"""
        Ecuaciones planetarias de Gauss (osculadores) a perturbación radial
        isotrópica de magnitud `pert_eps` (diagnóstico de deriva secular):
            \(\dot a=2\sqrt{a^3/\mu}\,R\), \(\dot e\simeq 0\) si \(R\) central.
        Con \(\mu=1\), \(R=\textit{pert_eps}\).
        """
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

    @classmethod
    def laskar_naff_frequency(
        cls,
        rho: np.ndarray,
        N_diag: Optional[np.ndarray] = None,
        n_samples: int = 128,
        dt: float = 0.05,
    ) -> np.ndarray:
        r"""
        Análisis de frecuencias de Laskar (NAFF reducido): FFT del observable
        \(f(t)=\operatorname{Tr}(\rho(t)X)\), \(X=\operatorname{diag}(1,\dots,n)\),
        a lo largo del flujo exacto. Devuelve las 3 frecuencias dominantes \(\ge 0\).
        """
        rho0 = _hermitize_trace_one(rho)
        n = rho0.shape[0]
        nu = _diag_potential(n, N_diag)
        X = np.diag(nu)
        samples = np.empty(n_samples, dtype=float)
        current = rho0
        phase_dt = np.exp(-1j * dt * (nu[:, None] - nu[None, :]))
        for k in range(n_samples):
            samples[k] = float(np.trace(current @ X).real)
            current = current * phase_dt
        spec = np.abs(np.fft.rfft(samples - np.mean(samples)))
        freqs = np.fft.rfftfreq(n_samples, d=dt)
        order = np.argsort(spec)[::-1]
        top = freqs[order[: min(3, order.size)]]
        return np.asarray(np.abs(top), dtype=float)

    # ── §2.10 Núcleo pipeline: evaluate_dream_spectrum ───────────────────
    def evaluate_dream_spectrum(
        self,
        density_matrix: np.ndarray,
        dirichlet_energy: Optional[float],
        betti_1: int,
        dream_isolation: bool,
        betti_0: int = 1,
        betti_2: int = 0,
        rho_base: Optional[np.ndarray] = None,
        boundary_stalk: Optional[np.ndarray] = None,
        scenario_id: str = "DREAM-EVAL",
    ) -> "UnsealedOniricTrace":
        r"""
        Evalúa el espectro onírico completo con análisis de Poincaré.

        Pipeline:
            1. Spec : extract_spectral_measure(ρ)
            2. E_D  : compute_dirichlet_energy
            3. TV   : compute_dirac_total_variation
            4. GW   : invariante relativo Poincaré–Lefschetz
            5. Celeste : Wigner, KAM, Brjuno, Chirikov, Greene, Nekhoroshev,
                         Hannay, Tisserand, Jacobi, Maslov, recurrencia.

        CONTINÚA EN FASE-3: `_seal_and_accumulate` consume UnsealedOniricTrace.
        """
        measure = self.extract_spectral_measure(density_matrix)
        lam = measure.as_simplex()
        ed_internal = self.compute_dirichlet_energy(lam)
        ed_dirac = self.compute_dirac_total_variation(lam)
        ed = ed_internal if dirichlet_energy is None else float(dirichlet_energy)
        ed_residual = abs(ed - ed_internal)
        b0 = max(int(betti_0), 0)
        b1 = max(int(betti_1), 0)
        b2 = max(int(betti_2), 0)
        chi = b0 - b1 + b2
        n = int(np.asarray(density_matrix).shape[0])
        base = rho_base if rho_base is not None else np.eye(n, dtype=np.complex128) / float(n)
        stalk = boundary_stalk if boundary_stalk is not None else np.eye(n, dtype=np.complex128)
        poincare_cert = self.gw_auditor.evaluate_poincare_lefschetz_gw_invariant(
            rho_dream=density_matrix,
            rho_base=base,
            boundary_stalk_matrix=stalk,
            betti_1_cycles=b1,
            dirichlet_energy=ed,
            dream_isolation=dream_isolation,
            scenario_id=scenario_id,
        )
        gw = poincare_cert.gw_relative_invariant
        verdict = poincare_cert.heyting_verdict
        try:
            rho_op = DensityOperator.from_array(density_matrix)
            omega = rho_op.mean_motion_frequencies()
            chirikov = rho_op.chirikov_overlap()
            resonance = self.detect_resonances(omega, chirikov=chirikov)
            wigner_neg = rho_op.wigner_negativity()
            capacity = rho_op.symplectic_capacity_gromov()
            recurrence = rho_op.poincare_recurrence_time()
            maslov = rho_op.maslov_index()
            stratum = verdict.poincare_stratum_name()
            lyap = verdict.lyapunov_exponent_sign()
            elems = rho_op.celestial_elements()
            hannay = rho_op.hannay_angle()
            brjuno = rho_op.siegel_brjuno_sum()
            N4 = np.diag(_diag_potential(n))
            section = self.poincare_section_at(elems.energy, n)
            floq = self.monodromy_and_greene(rho_op.matrix, section, N4)
            eps_comm = float(la.norm(rho_op.matrix @ N4 - N4 @ rho_op.matrix, "fro"))
            t_nek = self.nekhoroshev_stability_time(eps_comm, n)
            greene = floq.greene_residue
            floquet_r = floq.spectral_radius
            tisserand = elems.tisserand
            jacobi = elems.jacobi_integral
            mean_motion = elems.mean_motion
            ecc = elems.eccentricity
        except Exception:
            logger.exception("Fallo en el análisis celeste; se aplican neutros.")
            resonance = ResonanceStructure(
                vector=(),
                order=0,
                residue_min=float("inf"),
                kam_residue=float("inf"),
                is_kam_safe=False,
            )
            wigner_neg = 0.0
            capacity = 1.0
            recurrence = float("inf")
            maslov = 0
            stratum = "undetermined"
            lyap = 0
            hannay = 0.0
            brjuno = 0.0
            greene = 0.0
            floquet_r = 1.0
            t_nek = float("inf")
            tisserand = 0.0
            jacobi = 0.0
            mean_motion = 0.0
            ecc = 0.0
            chirikov = 0.0
        return UnsealedOniricTrace(
            density_matrix=np.array(_hermitize_trace_one(density_matrix), copy=True),
            measure=measure,
            dirichlet_energy=ed,
            dirichlet_internal=ed_internal,
            dirichlet_residual=ed_residual,
            dirac_total_variation=ed_dirac,
            gromov_witten_invariant=gw,
            tqft_amplitude=gw,
            betti_0=b0,
            betti_1=b1,
            betti_2=b2,
            euler_characteristic=chi,
            heyting_verdict=verdict,
            dream_isolation=dream_isolation,
            poincare_cert=poincare_cert,
            wigner_negativity=wigner_neg,
            symplectic_capacity=capacity,
            poincare_recurrence_time=recurrence,
            maslov_index=maslov,
            resonance_order=resonance.order,
            kam_residue=resonance.kam_residue,
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
            eccentricity=ecc,
        )


# ── §2.11 Traza abierta: último artefacto de FASE-2, germen de FASE-3 ─────
@dataclass(frozen=True, slots=True)
class UnsealedOniricTrace:
    r"""
    Traza onírica sin sellar. Portadora del bundle
    \((\mathrm{Spec},E_D,\mathrm{TV},\mathrm{GW},\mathrm{Wigner},\mathrm{KAM},
    \mathrm{Maslov},\mathrm{Hannay},\mathrm{Greene},\mathrm{Chirikov},
    \mathrm{Nekhoroshev},\mathrm{Brjuno})\) generado en FASE-2.

    Cierra FASE-2. FASE-3 **continúa** exactamente aquí:
    `TOONOniricAuditorEngine._seal_and_accumulate` consume esta traza.
    """

    density_matrix: np.ndarray
    measure: SpectralMeasure
    dirichlet_energy: float
    dirichlet_internal: float
    dirichlet_residual: float
    dirac_total_variation: float
    gromov_witten_invariant: float
    tqft_amplitude: float
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    heyting_verdict: HeytingOmega3
    dream_isolation: bool
    poincare_cert: Optional[ImmunizationCertificate] = None
    wigner_negativity: float = 0.0
    symplectic_capacity: float = 1.0
    poincare_recurrence_time: float = float("inf")
    maslov_index: int = 0
    resonance_order: int = 0
    kam_residue: float = float("inf")
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
    eccentricity: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — AUDITOR, SELLO, HOLONOMÍA BERRY–HANNAY, MERKLE, PASAPORTE
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: _seal_and_accumulate consume UnsealedOniricTrace (último artefacto
# de FASE-2). Se completa el pipeline 𝒲 = Seal ∘ Crowbar ∘ GW ∘ Spec.
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    r"""
    Prueba de inclusión Merkle SHA-256.

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


class TOONOniricAuditorEngine:
    r"""
    Motor espectral auditor de sueños (fase REM) con dualidad Poincaré–Lefschetz,
    rigidez de Gromov, flujo de von Neumann, holonomía Berry–Hannay, disyuntor
    ESP32 Crowbar (GPIO14), sello SHA-256 y árbol de Merkle.

    Co-gobierna el Estrato Wisdom (\(V_\mathbb{W}\)). Cada ciclo produce un
    `OniricFieldState` inmutable indexado por `cycle_id`.
    """

    def __init__(
        self,
        engine_id: str = "ONIRIC-ENGINE-SABIO-01",
        gw_auditor: Optional[GromovWittenOniricAuditor] = None,
        spectra_engine: Optional[OniricSpectraEngine] = None,
    ) -> None:
        self.engine_id: str = engine_id
        self.auditor = gw_auditor or GromovWittenOniricAuditor()
        self.spectra: OniricSpectraEngine = (
            spectra_engine
            if spectra_engine is not None
            else OniricSpectraEngine(gw_auditor=self.auditor)
        )
        self.cycle_count: int = 0
        self.history: List[OniricFieldState] = []
        self._holonomy_accum: float = 0.0
        self._hannay_accum: float = 0.0

    # ── §3.1 HAND-OFF FASE-2 → FASE-3: sello y acumulación ───────────────
    def _seal_and_accumulate(
        self,
        trace: UnsealedOniricTrace,
        cycle_id: str,
        scenario_id: str,
    ) -> OniricFieldState:
        r"""
        CONTINUACIÓN FORMAL de `evaluate_dream_spectrum` (FASE-2).

        Consume `UnsealedOniricTrace`, sella con SHA-256, acumula la holonomía
        de Wilson \(e^{i\sum\gamma_{\mathrm{GW}}}\) y la holonomía de Hannay
        \(e^{i\sum\theta_H}\), y construye el `OniricFieldState` inmutable.
        """
        t_seal = time.time()
        imm_hash = self._seal_passport(
            cycle_id=cycle_id,
            scenario_id=scenario_id,
            verdict=trace.heyting_verdict,
            gw_invariant=trace.gromov_witten_invariant,
            t_seal=t_seal,
        )
        self._holonomy_accum += trace.gromov_witten_invariant
        self._hannay_accum += trace.hannay_angle
        wilson = complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))
        p_cert = trace.poincare_cert
        defect = p_cert.poincare_lefschetz_defect if p_cert else 0.0
        cap_ratio = p_cert.symplectic_capacity_ratio if p_cert else 1.0
        is_consistent = p_cert.is_boundary_consistent if p_cert else True
        crowbar = (
            p_cert.crowbar_triggered
            if p_cert
            else (trace.heyting_verdict == HeytingOmega3.VETOED)
        )
        gpio14 = (
            p_cert.gpio14_signal
            if p_cert
            else ("HIGH" if crowbar else "LOW")
        )
        proof_512 = p_cert.proof_merkle_sha512 if p_cert else ""
        return OniricFieldState(
            cycle_id=cycle_id,
            scenario_id=scenario_id,
            dream_isolation_flag=trace.dream_isolation,
            density_matrix=trace.density_matrix,
            dirichlet_energy=trace.dirichlet_energy,
            dirac_total_variation=trace.dirac_total_variation,
            gromov_witten_invariant=trace.gromov_witten_invariant,
            tqft_amplitude=trace.tqft_amplitude,
            purity=trace.measure.purity,
            von_neumann_entropy=trace.measure.von_neumann_entropy,
            spectral_gap=trace.measure.spectral_gap,
            cstar_residual=trace.measure.cstar_residual,
            betti_0=trace.betti_0,
            betti_1_loops=trace.betti_1,
            betti_2=trace.betti_2,
            euler_characteristic=trace.euler_characteristic,
            heyting_verdict=trace.heyting_verdict,
            immunization_hash=imm_hash,
            timestamp_utc=t_seal,
            holonomy_partial=self._holonomy_accum,
            wilson_phase=wilson,
            poincare_lefschetz_defect=defect,
            symplectic_capacity_ratio=cap_ratio,
            is_boundary_consistent=is_consistent,
            crowbar_triggered=crowbar,
            gpio14_signal=gpio14,
            proof_merkle_sha512=proof_512,
            wigner_negativity=trace.wigner_negativity,
            symplectic_capacity=trace.symplectic_capacity,
            poincare_recurrence_time=trace.poincare_recurrence_time,
            maslov_index=trace.maslov_index,
            resonance_order=trace.resonance_order,
            kam_residue=trace.kam_residue,
            lyapunov_sign=trace.lyapunov_sign,
            poincare_stratum=trace.poincare_stratum,
            hannay_angle=trace.hannay_angle,
            greene_residue=trace.greene_residue,
            chirikov_overlap=trace.chirikov_overlap,
            nekhoroshev_time=trace.nekhoroshev_time,
            brjuno_sum=trace.brjuno_sum,
            tisserand=trace.tisserand,
            jacobi_integral=trace.jacobi_integral,
            floquet_radius=trace.floquet_radius,
            mean_motion=trace.mean_motion,
            eccentricity=trace.eccentricity,
        )

    def _seal_passport(
        self,
        cycle_id: str,
        scenario_id: str,
        verdict: HeytingOmega3,
        gw_invariant: float,
        t_seal: float,
    ) -> str:
        r"""Sello SHA-256 inyectivo sobre \((\mathrm{id}\|c\|s\|v\|\gamma^*\|t)\)."""
        hasher = hashlib.sha256()
        payload = (
            f"{self.engine_id}::{cycle_id}::{scenario_id}::"
            f"{verdict.name}::{gw_invariant:.10f}::{t_seal:.6f}"
        )
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    # ── §3.2 Pipeline rápido (interfaz reducida) ─────────────────────────
    def process_oniric_audit_pipeline(
        self,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        boundary_stalk: np.ndarray,
        betti_1_cycles: int = 0,
    ) -> Dict[str, Any]:
        r"""Tubería compacta TQFT + rigidez simpléctica + crowbar si VETOED."""
        cert = self.auditor.evaluate_poincare_lefschetz_gw_invariant(
            rho_dream=rho_dream,
            rho_base=rho_base,
            boundary_stalk_matrix=boundary_stalk,
            betti_1_cycles=betti_1_cycles,
        )
        crowbar_triggered = False
        gpio14_signal = "LOW"
        if cert.heyting_verdict == HeytingOmega3.VETOED:
            crowbar_triggered = True
            gpio14_signal = "HIGH"
        return {
            "gw_relative_invariant": cert.gw_relative_invariant,
            "poincare_lefschetz_defect": cert.poincare_lefschetz_defect,
            "symplectic_capacity_ratio": cert.symplectic_capacity_ratio,
            "heyting_verdict": cert.heyting_verdict.name,
            "heyting_code": cert.heyting_verdict.value,
            "is_boundary_consistent": cert.is_boundary_consistent,
            "crowbar_triggered": crowbar_triggered,
            "gpio14_signal": gpio14_signal,
            "merkle_sha512": cert.proof_merkle_sha512,
            "schema_version": "9.1.0-Poincare-Celeste-KAM-Crowbar",
        }

    # ── §3.3 Ciclo onírico completo ──────────────────────────────────────
    def audit_oniric_cycle(
        self,
        scenario_id: str,
        density_matrix: np.ndarray,
        dirichlet_energy: Optional[float] = None,
        betti_1: int = 0,
        dream_isolation: bool = True,
        betti_0: int = 1,
        betti_2: int = 0,
        rho_base: Optional[np.ndarray] = None,
        boundary_stalk: Optional[np.ndarray] = None,
    ) -> OniricFieldState:
        r"""
        Ciclo completo: \(\mathrm{Spec}\to\mathrm{GW}\to\mathrm{KAM}\to\mathrm{Seal}\to\mathrm{Crowbar}\to\mathrm{Merkle}\).
        """
        self.cycle_count += 1
        cycle_id = f"CYC-ONIRIC-AUDIT-{self.cycle_count:04d}"
        t_start = time.time()
        logger.info(
            "=== Iniciando Auditoría Espectral Onírica %s | Escenario: %s ===",
            cycle_id,
            scenario_id,
        )
        trace = self.spectra.evaluate_dream_spectrum(
            density_matrix=density_matrix,
            dirichlet_energy=dirichlet_energy,
            betti_1=betti_1,
            dream_isolation=dream_isolation,
            betti_0=betti_0,
            betti_2=betti_2,
            rho_base=rho_base,
            boundary_stalk=boundary_stalk,
            scenario_id=scenario_id,
        )
        state = self._seal_and_accumulate(trace, cycle_id, scenario_id)
        self.history.append(state)
        logger.info(
            "Ciclo %s en %.2f ms | %s | I_GW=%.6f | R_KAM=%.3e | "
            "Chirikov=%.3f | Greene=%.3f | Nek=%.3e | Stratum=%s | Crowbar=%s",
            cycle_id,
            (time.time() - t_start) * 1000.0,
            state.heyting_verdict.name,
            state.gromov_witten_invariant,
            state.kam_residue,
            state.chirikov_overlap,
            state.greene_residue,
            state.nekhoroshev_time,
            state.poincare_stratum,
            state.crowbar_triggered,
        )
        return state

    # ── §3.4 Vistas y propiedades agregadas ──────────────────────────────
    @property
    def registry(self) -> Tuple[OniricFieldState, ...]:
        return tuple(self.history)

    @property
    def holonomy_accum(self) -> float:
        return self._holonomy_accum

    @property
    def wilson_loop(self) -> complex:
        r"""Lazo de Wilson \(e^{i\sum\gamma_{\mathrm{GW}}}\in U(1)\)."""
        return complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))

    @property
    def hannay_loop(self) -> complex:
        r"""Holonomía de Hannay \(e^{i\sum\theta_H}\in U(1)\)."""
        return complex(math.cos(self._hannay_accum), math.sin(self._hannay_accum))

    @property
    def global_verdict(self) -> HeytingOmega3:
        r"""Ínfimo (meet) de los veredictos — objeto terminal de \(\Omega_3\)."""
        gv = HeytingOmega3.COHERENT
        for s in self.history:
            gv = gv.meet(s.heyting_verdict)
        return gv

    @property
    def berry_phase_along_history(self) -> float:
        r"""Fase de Berry–Pancharatnam cerrada a lo largo de la historia."""
        if len(self.history) < 2:
            return 0.0
        rho_curve = [s.density_matrix for s in self.history]
        return DensityOperator.from_array(rho_curve[0]).berry_phase_curve(rho_curve, closed=True)

    # ── §3.5 Árbol de Merkle y pruebas de inclusión ──────────────────────
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
        return self._merkle_tree_root([s.immunization_hash for s in self.history])

    def merkle_proofs_ok(self) -> bool:
        leaves = [s.immunization_hash for s in self.history]
        root = self._merkle_tree_root(leaves)
        for i in range(len(leaves)):
            proof = self._merkle_proof(leaves, i)
            if proof.root != root or not proof.verify():
                return False
        return True

    # ── §3.6 Auditoría retrospectiva y pasaporte ────────────────────────
    def audit_registry(self) -> Dict[str, Any]:
        r"""Auditoría retrospectiva del registro onírico, holonomías y Merkle."""
        n = len(self.history)
        empty_dist = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "holonomy_accum": 0.0,
                "wilson_loop": 1.0 + 0.0j,
                "hannay_loop": 1.0 + 0.0j,
                "berry_phase": 0.0,
                "avg_gw_invariant": 0.0,
                "avg_dirichlet_energy": 0.0,
                "avg_dirac_tv": 0.0,
                "avg_purity": 0.0,
                "avg_kam_residue": 0.0,
                "avg_capacity": 0.0,
                "avg_chirikov": 0.0,
                "avg_greene": 0.0,
                "n_immune": 0,
                "all_physically_valid": True,
                "registry_integrity_ok": True,
                "merkle_proofs_ok": True,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        total_gw = total_ed = total_tv = total_p = 0.0
        total_kam = total_cap = total_ch = total_gr = 0.0
        n_imm = 0
        all_valid = True
        hashes: Set[str] = set()
        collide = False
        for s in self.history:
            dist[s.heyting_verdict.name] += 1
            total_gw += s.gromov_witten_invariant
            total_ed += s.dirichlet_energy
            total_tv += s.dirac_total_variation
            total_p += s.purity
            total_kam += s.kam_residue if math.isfinite(s.kam_residue) else 0.0
            total_cap += s.symplectic_capacity
            total_ch += s.chirikov_overlap
            total_gr += s.greene_residue
            if s.is_immune():
                n_imm += 1
            if not s.is_quantum_physical():
                all_valid = False
            if s.immunization_hash in hashes:
                collide = True
            hashes.add(s.immunization_hash)
        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "hannay_loop": self.hannay_loop,
            "berry_phase": self.berry_phase_along_history,
            "avg_gw_invariant": total_gw * inv,
            "avg_dirichlet_energy": total_ed * inv,
            "avg_dirac_tv": total_tv * inv,
            "avg_purity": total_p * inv,
            "avg_kam_residue": total_kam * inv,
            "avg_capacity": total_cap * inv,
            "avg_chirikov": total_ch * inv,
            "avg_greene": total_gr * inv,
            "n_immune": n_imm,
            "all_physically_valid": all_valid,
            "registry_integrity_ok": not collide,
            "merkle_proofs_ok": self.merkle_proofs_ok(),
        }

    def emit_passport(self) -> Dict[str, Any]:
        r"""Pasaporte criptográfico agregado (GodelEngine / Ciudadela de Cristal)."""
        h = hashlib.sha256()
        h.update(
            f"{self.engine_id}::{self.cycle_count}::{self._holonomy_accum:.10f}".encode("utf-8")
        )
        for s in self.history:
            h.update(s.immunization_hash.encode("utf-8"))
        return {
            "engine_id": self.engine_id,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "hannay_loop": self.hannay_loop,
            "berry_phase": self.berry_phase_along_history,
            "global_verdict": self.global_verdict.name,
            "evidence_hash": h.hexdigest(),
            "merkle_root": self.merkle_root(),
            "registry_size": self.cycle_count,
            "n_immune": sum(1 for s in self.history if s.is_immune()),
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
    print("DEMOSTRACIÓN GRANULAR: TOON Oniric Auditor Engine v9.1.0-Poincare-Celeste")
    print("FASES: Ω₃+Spec+Wigner → TQFT/von Neumann/KAM/Maslov → Seal/Merkle/Crowbar")
    print("═" * 80)

    print("\n[§0] VERIFICACIÓN FORMAL DE Ω₃")
    assert HeytingOmega3.verify_residuation_axiom(), "Residuación falla"
    assert HeytingOmega3.DEGRADED.excluded_middle_holds() is False
    assert HeytingOmega3.COHERENT.excluded_middle_holds() is True
    assert HeytingOmega3.VETOED.is_regular() is True
    assert HeytingOmega3.DEGRADED.is_regular() is False
    assert HeytingOmega3.VETOED.poincare_stratum_name() == "hyperbolic-escape-separatrix"
    assert HeytingOmega3.COHERENT.is_kam_stratum() is True
    assert HeytingOmega3.COHERENT.lyapunov_exponent_sign() == -1
    print("  • Residuación, tercio excluso, regularidad y estratos: OK")

    # Red entera: (2,0) debe existir.
    lat = _iterate_integer_lattice(2, 2)
    assert any(np.array_equal(v, np.array([2, 0])) for v in lat)
    assert any(np.array_equal(v, np.array([-2, 0])) for v in lat)
    print("  • Red ℤ² de norma ℓ¹=2 contiene (±2,0): OK")

    rng = np.random.default_rng(20250321)
    engine = TOONOniricAuditorEngine()

    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    rho_base = np.eye(4, dtype=np.complex128) / 4.0
    stalk = np.eye(4, dtype=np.complex128)

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
        f"e={elems.eccentricity:.4f} i={elems.inclination:.4f} Tiss={elems.tisserand:.4f}"
    )

    E_sol = OniricSpectraEngine.solve_kepler(mean_anomaly=1.2, eccentricity=0.3)
    resid = OniricSpectraEngine.kepler_residual(E_sol, 1.2, 0.3)
    assert abs(resid) < 1e-10, f"Kepler resid={resid}"
    print(f"  • Kepler-Halley E={E_sol:.8f}, resid={resid:.3e}")

    assert OniricSpectraEngine.poincare_birkhoff_count(1, 3, math.pi / 2) == 6
    print("  • Poincaré–Birkhoff p/q=1/3 ⇒ 2q=6 puntos fijos")

    omega_irr = np.array([math.sqrt(2), math.sqrt(3), math.sqrt(5), math.sqrt(7)])
    omega_rat = np.array([1.0, 1.0, 1.0, 1.0])
    R_irr = OniricSpectraEngine.diophantine_residue(omega_irr)
    R_rat = OniricSpectraEngine.diophantine_residue(omega_rat)
    print(f"  • R_irr={R_irr:.6e}, R_rat={R_rat:.6e}")

    T_ret = rho_op.first_return_phase_time()
    print(f"  • Retorno kepleriano exacto T=2π/|ν₀-ν₁|={T_ret:.6f}")

    print("\n>>> ESCENARIO PIPELINE TQFT POINCARÉ–LEFSCHETZ...")
    res = engine.process_oniric_audit_pipeline(
        rho_dream=rho, rho_base=rho_base, boundary_stalk=stalk, betti_1_cycles=0
    )
    for k, v in res.items():
        print(f"    - {k:<26}: {v}")

    print("\n>>> ESCENARIO A: Estado físico canónico...")
    s1 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-ONIRIC-001",
        density_matrix=rho,
        dirichlet_energy=0.35,
        betti_1=0,
        dream_isolation=True,
    )
    print(f"    - ID Ciclo             : {s1.cycle_id}")
    print(f"    - Veredicto Heyting    : {s1.heyting_verdict.name}")
    print(f"    - Estrato Poincaré     : {s1.poincare_stratum}")
    print(f"    - I_GW / TQFT          : {s1.gromov_witten_invariant:.6f}")
    print(f"    - Cap. Gromov          : c_G={s1.symplectic_capacity:.4f}, ratio={s1.symplectic_capacity_ratio:.4f}")
    print(f"    - Chirikov s           : {s1.chirikov_overlap:.6f}")
    print(f"    - Greene R             : {s1.greene_residue:.6f}")
    print(f"    - Nekhoroshev T_N      : {s1.nekhoroshev_time:.6e}")
    print(f"    - Hannay θ_H           : {s1.hannay_angle:.6f}")
    print(f"    - Tisserand T          : {s1.tisserand:.6f}")
    print(f"    - Crowbar / GPIO14     : {s1.crowbar_triggered} / {s1.gpio14_signal}")

    print("\n>>> ESCENARIO B: Violación de aislamiento REM (veto duro)...")
    s3 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-ONIRIC-BREACH",
        density_matrix=rho,
        dirichlet_energy=0.15,
        betti_1=5,
        dream_isolation=False,
    )
    print(f"    - Veredicto / Estrato  : {s3.heyting_verdict.name} / {s3.poincare_stratum}")
    print(f"    - Crowbar / GPIO14     : {s3.crowbar_triggered} / {s3.gpio14_signal}")
    assert s3.heyting_verdict == HeytingOmega3.VETOED
    assert s3.crowbar_triggered is True
    assert s3.gpio14_signal == "HIGH"

    print("\n>>> SECCIÓN DE POINCARÉ (corte de fase sobre M_c)...")
    N_diag_4 = np.diag(np.arange(1, 5, dtype=float))
    level_c = float(np.trace(rho @ N_diag_4).real)
    section = OniricSpectraEngine.poincare_section_at(level_c, 4)
    t_ret, rho_ret = OniricSpectraEngine.poincare_return_time(
        rho, section, N_diag_4, dt=0.02, max_time=100.0
    )
    print(f"    - Nivel M_c            : {level_c:.6f}")
    print(f"    - Tiempo de retorno    : {t_ret:.6f}  (analítico {T_ret:.6f})")
    rhodot = -1j * (N_diag_4 @ rho - rho @ N_diag_4)
    print(f"    - Transversalidad      : {section.is_transversal(rho, rhodot)}")

    print("\n>>> HOLONOMÍAS BERRY–HANNAY / WILSON...")
    print(f"    - γ_Berry              : {engine.berry_phase_along_history:.6f} rad")
    print(f"    - Wilson loop          : {engine.wilson_loop}")
    print(f"    - Hannay loop          : {engine.hannay_loop}")

    print("\n>>> AUDITORÍA RETROSPECTIVA...")
    audit = engine.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<26}: {v}")
    assert audit["registry_integrity_ok"]
    assert audit["merkle_proofs_ok"]

    print("\n>>> PASAPORTE AGREGADO...")
    passport = engine.emit_passport()
    for k, v in passport.items():
        print(f"    - {k:<22}: {v}")

    print("\n" + "═" * 80)
    print("✓ Verificación TOON Oniric Auditor Engine v9.1.0-Poincare-Celeste completada.")
    print("═" * 80)
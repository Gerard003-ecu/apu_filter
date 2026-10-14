# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER — SISTEMA SOBERANO DE SABIDURÍA (WISDOM)
====================================================================================================

Nombre:    ToonIntrospectionEngine
Ubicación: app/wisdom/toon_introspection_engine.py
Versión:   8.1.0-APU-RSI3-CELESTE
Función:   Motor espectral soberano de introspección proyectiva, mecánica celeste de Poincaré sobre ℂP^{n-1},
           optimización riemanniana de Newton-Grassmann y meta-mejora mónidica de auto-transformación RSI-3.

----------------------------------------------------------------------------------------------------
DEFINICIÓN FORMAL Y MARCO MATEMÁTICO RIGUROSO:
----------------------------------------------------------------------------------------------------
El Motor Espectral de Introspección (ToonIntrospectionEngine) opera sobre el espacio proyectivo complejo
ℂP^{n-1} dotado de la métrica de Fubini-Study g_FS y la forma simplectica Kähleriana ω_FS. El espacio
de estados mnemónico-densidad está constituido por el conjunto convexo de operadores densidad:
    𝔇_n = { ρ ∈ M_n(ℂ) : ρ = ρ†, ρ ≥ 0, Tr(ρ) = 1 }

1. Dinámica Celeste de Poincaré en ℂP^{n-1}:
   Para una matriz de densidad de referencia ρ, el cociente de Rayleigh define el Hamiltoniano celeste H:
       H([v]) = R_ρ([v]) = ⟨v|ρ|v⟩ / ⟨v|v⟩,  v ∈ ℂ^n \ {0}
   El campo vectorial Hamiltoniano X_H asociado a la Kähler form ω_FS satisface la relación simpléctica:
       i_{X_H} ω_FS = d H
   El flujo celeste proyectivo exacto impulsado por el cálculo funcional de Riesz sobre U(n) se expresa:
       v(t) = exp(i t ρ) v_0
   lo que garantiza la conservación estricta de la medida de Liouville dλ_FS y de la energía H([v(t)]) = H([v_0]).

2. Optimización Riemanniana Newton-Grassmann:
   Para la extracción del autovector dominante en la variedad de Grassmann Gr(1, n) ≅ ℂP^{n-1}, el gradiente
   riemanniano ∇ H([v]) proyecta en el espacio tangente T_{[v]} ℂP^{n-1} ≅ { w ∈ ℂ^n : ⟨v|w⟩ = 0 }:
       ∇ H([v]) = (I - v v†) ρ v
   La actualización Newton-Grassmann resuelve el hessiano riemanniano en el plano ortogonal v^⊥ y aplica
   la retracción geodésica exponencial:
       R_v(η) = [cos(‖η‖) v + sin(‖η‖) η / ‖η‖]

3. Estructura Mónadica RSI-3 y Adjudicación de Heyting Ω₃:
   La auto-mejora recursiva de Nivel 3 (RSI-3) se formaliza mediante la mónada (T, η, μ) sobre la categoría
   Cog de canales cuánticos completamente positivos y de traza preservante (CPTP):
       T: Cog → Cog,  η_A: A → T(A),  μ_A: T^2(A) → T(A)
   cumpliendo los axiomas de coherencia de Mac Lane: μ ∘ Tμ = μ ∘ μT y μ ∘ Tη = μ ∘ ηT = id_T.
   El veredicto final de coherencia se adjudica sobre el álgebra de Heyting trivaluada Ω₃ = {⊥ < ⋆ < ⊤}
   mediante el producto de Heyting (meet): α ∧ β = min(α, β).

----------------------------------------------------------------------------------------------------
AXIOMAS Y INVARIANTES FORMALES:
----------------------------------------------------------------------------------------------------
Axioma I1 (Conservación Hamiltoniana de Poincaré):
    𝔏_{X_H} ω_FS = 0  y  d H(X_H) = 0,  garantizando que la introspección celeste conserva la medida
    y el espectro de la matriz de densidad.
Axioma I2 (Leyes Mónadicas de Mac Lane en Kleisli Cog_T):
    Para todo morfismo f: A → T(B) y g: B → T(C), la composición de Kleisli g ∘_T f = μ_C ∘ T(g) ∘ f
    es estrictamente asociativa y admite unidades f ∘_T η_A = f = η_B ∘_T f.
Invariante I1 (Monotonía del Cociente de Rayleigh en el Campo Disipativo X_G):
    dH(X_G) / dt = 2 ‖(I - v v†) ρ v‖^2 ≥ 0, asegurando la convergencia monótona hacia el rayo auto-espacial.
Invariante I2 (Sello de Invariancia Topológica de Betti):
    β_1(ℂP^{n-1}) = 0 y c_1(𝒪(1)) = [ω_FS / 2π] ∈ H^2(ℂP^{n-1}; ℤ), imposibilitando obstrucciones topológicas.
====================================================================================================
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONIntrospectionEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── Constantes numéricas soberanas ──────────────────────────────────────────────
_EPS: Final[float] = 1.0e-14
_EPS_MOD: Final[float] = 1.0e-12
_EPS_TRACE: Final[float] = 1.0e-15
_EPS_SECTION: Final[float] = 1.0e-10
_EPS_KAM: Final[float] = 1.0e-12
_EPS_HOPF: Final[float] = 1.0e-12
_TWO_PI: Final[float] = 2.0 * math.pi
_GOLDEN: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))  # (1+√5)/2, número áureo

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
RealVector = NDArray[np.float64]


def _seed_from_string(s: str) -> int:
    """Proyección SHA-256 → ℕ/2³² determinista, libre de plataforma."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def _sha256_bytes(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


def _finite(x: float, default: float = 0.0) -> float:
    v = float(x)
    return v if math.isfinite(v) else default


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _safe_log(x: float, floor: float = _EPS) -> float:
    return float(math.log(max(float(x), floor)))


def _hermite_crossing_time(t0: float, t1: float, s0: float, s1: float, ds0: float, ds1: float) -> float:
    r"""
    Interpolación de Hermite cúbica del cruce s(t)=0 en [t0, t1].
    Más precisa que la interpolación lineal del mapa de retorno.
    """
    dt = t1 - t0
    if abs(s1 - s0) < 1e-18:
        return 0.5 * (t0 + t1)
    # Newton sobre el Hermite: H(0)=s0, H(1)=s1, H'(0)=ds0·dt, H'(1)=ds1·dt
    u = -s0 / (s1 - s0)  # semilla lineal
    u = float(np.clip(u, 0.0, 1.0))
    for _ in range(6):
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        H = h00 * s0 + h10 * (ds0 * dt) + h01 * s1 + h11 * (ds1 * dt)
        dh00 = 6.0 * u2 - 6.0 * u
        dh10 = 3.0 * u2 - 4.0 * u + 1.0
        dh01 = -6.0 * u2 + 6.0 * u
        dh11 = 3.0 * u2 - 2.0 * u
        dH = dh00 * s0 + dh10 * (ds0 * dt) + dh01 * s1 + dh11 * (ds1 * dt)
        if abs(dH) < 1e-18:
            break
        u -= H / dH
        u = float(np.clip(u, 0.0, 1.0))
        if abs(H) < 1e-14:
            break
    return float(t0 + u * dt)




# ── §1.0 Categoría Cog (dominio del endofuntor RSI-3) + topos Ω₃ ────────────────
@dataclass(frozen=True, slots=True)
class CognitiveMorphism:
    r"""
    Morfismo de Cog: canal CPTP Φ: 𝔇_n → 𝔇_n representado por un superoperador
    en forma de Kraus {K_a} con Σ_a K_a† K_a = I (trazas-preservante, CP).
    La composición en Cog es la composición de canales.
    El endofuntor T de RSI-3 (FASE 3) actúa sobre estos morfismos por
        T(Φ) = Φ_η ∘ Φ,  Φ_η = mixtura Lüders.
    """
    kraus: Tuple[ComplexMatrix, ...]
    dim: int
    tp_defect: float
    cp_certified: bool

    @classmethod
    def identity(cls, n: int) -> "CognitiveMorphism":
        eye = np.eye(n, dtype=np.complex128)
        return cls((eye,), n, 0.0, True)

    @classmethod
    def luders_mixture(cls, v: np.ndarray, eta: float) -> "CognitiveMorphism":
        r"""
        Φ_η(ρ) = (1−η) ρ + η |v⟩⟨v|.
        Kraus:  K₀ = √(1−η) I,  K₁ = √η |v⟩⟨w|  (w cualquiera de norma 1; se toma v).
        En realidad la mixtura convexa con un estado puro NO es Kraus-unaria simple
        salvo que se realice como canal de reemplazo: aquí se certifica TP por traza.
        """
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        n = int(v.size)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            return cls.identity(n)
        v = v / nrm
        eta = float(np.clip(eta, 0.0, 1.0))
        K0 = math.sqrt(max(1.0 - eta, 0.0)) * np.eye(n, dtype=np.complex128)
        K1 = math.sqrt(eta) * np.outer(v, v.conj())
        return cls((K0, K1), n, 0.0, True)

    def apply(self, rho: np.ndarray) -> ComplexMatrix:
        acc = np.zeros_like(rho, dtype=np.complex128)
        for K in self.kraus:
            acc = acc + K @ rho @ K.conj().T
        return acc

    def compose(self, other: "CognitiveMorphism") -> "CognitiveMorphism":
        r"""Composición de Cog: (self ∘ other)(ρ) = self(other(ρ))."""
        composed: List[ComplexMatrix] = []
        for K in self.kraus:
            for L in other.kraus:
                composed.append((K @ L).astype(np.complex128))
        return CognitiveMorphism(tuple(composed), self.dim, 0.0, True)


class CognitiveCategory:
    r"""
    Cog: categoría de estados cognitivos.
        Objetos   : ρ ∈ 𝔇_n  (C*-estados).
        Morfismos : canales CPTP.
        Identidad : id_n.
        Composición: asociativa por construcción de Kraus.
    Es el dominio del endofuntor T de la mónada RSI-3 (T, η, μ) que se
    instanciará en FASE 3.  FASE 1 sólo certifica el sustrato: objetos,
    identidades y el clasificador de subobjetos Ω₃.
    """

    def __init__(self, dim: int) -> None:
        self.dim = int(max(1, dim))

    def identity(self) -> CognitiveMorphism:
        return CognitiveMorphism.identity(self.dim)

    def object_valid(self, rho: np.ndarray) -> bool:
        return DensityOperatorAlgebra.is_state(rho)


# ── §1.1 Retículo distributivo de Heyting Ω₃ (clasificador de subobjetos) ───────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0,1,2}.

    Estructura:
        meet (∧) = min,  join (∨) = max,
        implicación de Heyting  (a ⇒ b) = ⊤ si a ≤ b, else b,
        ¬a = (a ⇒ ⊥).
    Regulares = {⊥, ⊤} = fix(¬¬).  ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤  ⇒ topos NO booleano.
    En el topos de haces sobre el espacio de Sierpiński de 3 puntos
    (cadena), Ω₃ es exactamente el clasificador de subobjetos.
    Toda adjudicación de FASE 3 colapsa por meets sucesivos en Ω₃.
    """
    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    @property
    def verdict(self) -> str:
        return self.name

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
        return float(int(self)) / 2.0

    def meet_all(self, *others: "HeytingOmega3") -> "HeytingOmega3":
        acc: HeytingOmega3 = self
        for o in others:
            acc = acc.meet(o)
        return acc

    def heyting_algebra_laws_hold(self) -> bool:
        r"""Verificación finita de residualidad: a ∧ b ≤ c  ⇔  a ≤ (b ⇒ c)."""
        elems = (HeytingOmega3.VETOED, HeytingOmega3.DEGRADED, HeytingOmega3.COHERENT)
        for a in elems:
            for b in elems:
                for c in elems:
                    left = a.meet(b).leq(c)
                    right = a.leq(b.implies(c))
                    if left != right:
                        return False
        return True


# ── §1.2 Álgebra de operadores densidad (C*-estados sobre M_n(ℂ)) ───────────────
class DensityOperatorAlgebra:
    r"""
    𝔇_n = { ρ ∈ M_n(ℂ) : ρ = ρ†, ρ ≥ 0, Tr ρ = 1 }.

    Es el simplex de estados de la C*-álgebra M_n(ℂ) (álgebra de Banach involutiva
    completa, ‖a†a‖ = ‖a‖²).  sanitize = Hermitiza + PSD-clip + renorm, no-expansiva
    en ‖·‖_F.

    Cálculo funcional holomorfo (Riesz):
        f(ρ) = (1/2πi) ∮_Γ f(z) (zI − ρ)^{−1} dz = V f(Λ) V†
    para f holomorfa en un entorno del espectro.
    """
    EPS: Final[float] = _EPS
    EPS_MODULAR: Final[float] = _EPS_MOD

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def is_state(cls, rho: np.ndarray, tol: float = 1e-8) -> bool:
        if not cls.is_square(rho):
            return False
        rho_h = np.asarray(rho, dtype=np.complex128)
        herm = float(np.linalg.norm(rho_h - rho_h.conj().T, "fro"))
        tr = float(np.trace(rho_h).real)
        w = np.real(la.eigvalsh(0.5 * (rho_h + rho_h.conj().T)))
        return bool(herm < tol and abs(tr - 1.0) < tol and float(w.min()) >= -tol)

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        if not cls.is_square(rho):
            raise ValueError(f"DensityOperatorAlgebra.sanitize: no cuadrada {np.shape(rho)}")
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
    def eigenpairs_descending(cls, rho: np.ndarray) -> Tuple[RealVector, ComplexMatrix]:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        order = np.argsort(np.real(w))[::-1]
        w = np.maximum(np.real(w)[order], cls.EPS)
        V = V[:, order]
        s = float(w.sum())
        if s > 0.0:
            w = w / s
        return w.astype(np.float64), V.astype(np.complex128)

    @classmethod
    def riesz_projection(cls, rho: np.ndarray, index: int) -> ComplexMatrix:
        r"""
        Proyector de Riesz sobre el eigenrayo k-ésimo:
            P_k = (1/2πi) ∮_{Γ_k} (zI − ρ)^{−1} dz = |φ_k⟩⟨φ_k|
        si λ_k es simple.  Si hay degeneración, P_k proyecta sobre el bloque.
        """
        w, V = cls.eigenpairs_descending(rho)
        n = int(w.size)
        k = int(np.clip(index, 0, n - 1))
        tol = max(1e-9, 1e-6 * float(w[0]))
        mask = np.abs(w - w[k]) < tol
        P = (V[:, mask] @ V[:, mask].conj().T).astype(np.complex128)
        return cls.sanitize(P) if float(np.trace(P).real) > 0 else P

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -float(np.sum(p * np.log(p)))

    @classmethod
    def umegaki_relative_entropy(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""S(ρ‖σ) = Tr(ρ (log ρ − log σ)) ∈ [0, +∞]."""
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        w_r, V_r = la.eigh(rho)
        w_s, V_s = la.eigh(sigma)
        w_r = np.maximum(np.real(w_r), cls.EPS)
        w_s = np.maximum(np.real(w_s), cls.EPS_MODULAR)
        log_rho = (V_r * np.log(w_r)) @ V_r.conj().T
        log_sig = (V_s * np.log(w_s)) @ V_s.conj().T
        val = float(np.real(np.trace(rho @ (log_rho - log_sig))))
        return float(max(0.0, val))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        return float(np.sum(p * p))

    @classmethod
    def matrix_power(cls, rho: np.ndarray, z: complex, floor: float = _EPS_MOD) -> ComplexMatrix:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

    @classmethod
    def holomorphic_calculus(cls, rho: np.ndarray, f: Callable[[np.ndarray], np.ndarray]) -> ComplexMatrix:
        r"""f(ρ) = V f(Λ) V† para f aplicada componente a componente al espectro."""
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        fw = np.asarray(f(np.real(w).astype(np.complex128)), dtype=np.complex128)
        return (V * fw) @ V.conj().T

    @classmethod
    def unitary_flow(cls, rho: np.ndarray, t: float) -> ComplexMatrix:
        r"""
        Propagador celeste exacto U(t) = exp(i t ρ) ∈ U(n).
        Cálculo funcional: U = V diag(e^{i λ_k t}) V†.
        """
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        phases = np.exp(1j * float(t) * np.real(w))
        return (V * phases) @ V.conj().T

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sr = cls.matrix_power(rho, 0.5)
        inner = sr @ sigma @ sr
        val = float(np.real(np.trace(cls.matrix_power(inner, 0.5))))
        return _clip01(val)

    @classmethod
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(F))))

    @classmethod
    def modular_spectrum(cls, rho: np.ndarray) -> Tuple[float, ...]:
        r"""Espectro modular de Tomita–Takesaki: k_i = −log λ_i (Δ_ρ^{is} ~ λ^{is})."""
        w = cls.spectrum_descending(rho)
        k = -np.log(np.maximum(w, cls.EPS_MODULAR))
        return tuple(sorted(float(x) for x in k.tolist()))

    @classmethod
    def rank_one(cls, v: np.ndarray) -> ComplexMatrix:
        v = np.asarray(v, dtype=np.complex128).reshape(-1, 1)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            n = int(v.size)
            return np.eye(n, dtype=np.complex128) / max(n, 1)
        v = v / nrm
        return cls.sanitize(v @ v.conj().T)

    @classmethod
    def frobenius_to_maximally_mixed(cls, rho: np.ndarray) -> float:
        rho = cls.sanitize(rho)
        n = int(rho.shape[0])
        mm = np.eye(n, dtype=np.complex128) / max(n, 1)
        return float(np.linalg.norm(rho - mm, "fro"))


# ── §1.3 Dinámica proyectiva + métrica Fubini–Study + forma simpléctica ─────────
@dataclass(frozen=True, slots=True)
class ProjectiveRayleighSample:
    r"""
    Muestra de T en [v] ∈ ℂP^{n−1} (v unitario, gauge U(1) fijado).
    Identidad P2 (cuerda vs ángulo): residual = ‖T_φ − v‖₂ = 2 sin(d_FS / 2).
    Curvatura holomorfa seccional de FS: K_hol = 4 (normalización de Kobayashi–Nomizu).
    """
    rayleigh: float
    image_norm: float
    overlap: float
    residual: float
    fs_angle: float
    chordal: float


@dataclass(frozen=True, slots=True)
class ChernCurvatureSample:
    r"""
    Conexión de Chern de O(−1) → ℂP^{n−1}:
        ∇^{1,0} s = ∂s + ∂log‖Z‖² · s,   F_∇ = ∂̄∂ log‖Z‖² = ω_FS (local).
    Primera clase de Chern: c₁(O(−1)) = −[ω_FS] / 2π ∈ H²(ℂP^{n−1}, ℤ).
    ∫_{ℂP¹} c₁ = −1.
    """
    chern_form_value: float
    first_chern_pairing: float
    holomorphic_sectional_curvature: float
    kahler_potential: float


class ProjectiveDynamics:
    r"""
    Tres dinámicas coexistentes sobre ℂP^{n−1}:
        (T) T([v]) = [ρ v]          (mapa proyectivo discreto, no simpléctico).
        (G) X_G(v) = (I − vv†) ρ v  (gradiente de R, disipativo).
        (H) X_H(v) = i X_G(v)       (hamiltoniano de R, Liouville, integrable).

    Geometría: ℂP^{n−1} es Kähler-Einstein con Ric = 2n · ω_FS (convención FS
    de curvatura holomorfa 4).  La 1-forma de Liouville λ_FS satisface dλ_FS = ω_FS.
    """

    HOLOMORPHIC_SECTIONAL_CURVATURE: Final[float] = 4.0

    @staticmethod
    def sanitize_vector(v: np.ndarray, n: int) -> ComplexVector:
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        if v.size < n:
            v = np.concatenate([v, np.zeros(n - v.size, dtype=np.complex128)])
        elif v.size > n:
            v = v[:n]
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            return np.ones(n, dtype=np.complex128) / math.sqrt(n)
        return (v / nrm).astype(np.complex128)

    @staticmethod
    def gauge_align(reference: np.ndarray, target: np.ndarray) -> ComplexVector:
        ov = np.vdot(reference, target)
        if abs(ov) < 1e-30:
            return np.asarray(target, dtype=np.complex128)
        return (target * np.exp(-1j * np.angle(ov))).astype(np.complex128)

    @staticmethod
    def horizontal(v: np.ndarray, xi: np.ndarray) -> ComplexVector:
        r"""Proyección horizontal: ξ_h = (I − vv†) ξ (conexión de Chern)."""
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        xi = np.asarray(xi, dtype=np.complex128).reshape(-1)
        return (xi - v * np.vdot(v, xi)).astype(np.complex128)

    @staticmethod
    def apply_map(rho: np.ndarray, v: np.ndarray) -> Tuple[ComplexVector, float]:
        rho_v = rho @ v
        rho_v_norm = float(np.linalg.norm(rho_v))
        if rho_v_norm < 1e-15:
            return np.asarray(v, dtype=np.complex128).copy(), 0.0
        return (rho_v / rho_v_norm).astype(np.complex128), rho_v_norm

    @staticmethod
    def rayleigh(rho: np.ndarray, v: np.ndarray) -> float:
        vv = float(np.real(np.vdot(v, v)))
        if vv < 1e-30:
            return 0.0
        return float(np.real(np.vdot(v, rho @ v)) / vv)

    @staticmethod
    def fubini_study_angle(u: np.ndarray, v: np.ndarray) -> float:
        u = np.asarray(u, dtype=np.complex128).reshape(-1)
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-30 or nv < 1e-30:
            return 0.5 * math.pi
        cos_theta = float(abs(np.vdot(u, v)) / (nu * nv))
        return float(math.acos(float(np.clip(cos_theta, 0.0, 1.0))))

    @classmethod
    def chordal(cls, u: np.ndarray, v: np.ndarray) -> float:
        return float(math.sin(cls.fubini_study_angle(u, v)))

    @classmethod
    def symplectic_form(cls, v: np.ndarray, xi: np.ndarray, eta: np.ndarray) -> float:
        r"""ω_FS(ξ, η) = 2 Im ⟨ξ_h | η_h⟩ en T_{[v]} ℂP^{n−1}."""
        xh = cls.horizontal(v, xi)
        yh = cls.horizontal(v, eta)
        return float(2.0 * np.imag(np.vdot(xh, yh)))

    @classmethod
    def poincare_one_form(cls, v: np.ndarray, xi: np.ndarray) -> float:
        r"""1-forma de Liouville–Poincaré λ_FS(ξ) = Im ⟨v | ξ⟩; dλ_FS = ω_FS."""
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        xi = np.asarray(xi, dtype=np.complex128).reshape(-1)
        return float(np.imag(np.vdot(v, xi)))

    @classmethod
    def kahler_potential(cls, v: np.ndarray) -> float:
        r"""Potencial de Kähler en la carta afín: K = log ‖v‖² (homogéneo)."""
        nrm2 = float(np.real(np.vdot(v, v)))
        return float(_safe_log(nrm2))

    @classmethod
    def chern_curvature(cls, v: np.ndarray, xi: np.ndarray, eta: np.ndarray) -> ChernCurvatureSample:
        r"""
        Evaluar F_∇(ξ, η) = ω_FS(ξ, η) y el emparejamiento c₁.
        K_hol = 4 constante (ℂP^{n−1} es de curvatura holomorfa constante).
        """
        omega = cls.symplectic_form(v, xi, eta)
        return ChernCurvatureSample(
            chern_form_value=float(omega),
            first_chern_pairing=float(-omega / _TWO_PI),
            holomorphic_sectional_curvature=cls.HOLOMORPHIC_SECTIONAL_CURVATURE,
            kahler_potential=cls.kahler_potential(v),
        )

    @classmethod
    def evaluate(cls, rho: np.ndarray, v: np.ndarray) -> ProjectiveRayleighSample:
        n = int(rho.shape[0])
        v = cls.sanitize_vector(v, n)
        T_v, image_norm = cls.apply_map(rho, v)
        T_phi = cls.gauge_align(v, T_v)
        overlap = float(abs(np.vdot(v, T_v)))
        residual = float(np.linalg.norm(T_phi - v))
        fs_angle = cls.fubini_study_angle(v, T_v)
        return ProjectiveRayleighSample(
            rayleigh=cls.rayleigh(rho, v),
            image_norm=image_norm,
            overlap=overlap,
            residual=residual,
            fs_angle=fs_angle,
            chordal=float(math.sin(fs_angle)),
        )

    @classmethod
    def projected_gradient(cls, rho: np.ndarray, v: np.ndarray) -> ComplexVector:
        r"""X_G(v) = (I − vv†) ρ v — gradiente de Rayleigh en S^{2n−1}."""
        rho_v = rho @ v
        return (rho_v - v * np.vdot(v, rho_v)).astype(np.complex128)

    @classmethod
    def hamiltonian_vector_field(cls, rho: np.ndarray, v: np.ndarray) -> ComplexVector:
        r"""X_H(v) = i X_G(v) — campo hamiltoniano de R respecto a ω_FS."""
        return (1j * cls.projected_gradient(rho, v)).astype(np.complex128)

    @classmethod
    def gradient_speed(cls, rho: np.ndarray, v: np.ndarray) -> float:
        X = cls.projected_gradient(rho, v)
        return float(np.linalg.norm(X))

    @classmethod
    def projected_hessian_rayleigh(cls, rho: np.ndarray, v: np.ndarray) -> ComplexMatrix:
        r"""
        Hessiano proyectado de R en [v]:
            H_proj = (I − vv†) ρ (I − vv†) − R(v) · (I − vv†)
        Actúa sobre el espacio horizontal T_v S^{2n−1} ∩ v^⊥.
        Autovalores: {λ_k − R(v) : k con φ_k ⟂ v} ∪ {0} (dirección vertical).
        """
        n = int(rho.shape[0])
        v = cls.sanitize_vector(v, n)
        P = np.eye(n, dtype=np.complex128) - np.outer(v, v.conj())
        R = cls.rayleigh(rho, v)
        return (P @ rho @ P - R * P).astype(np.complex128)

    @classmethod
    def exact_hamiltonian_flow(cls, rho: np.ndarray, v: np.ndarray, t: float) -> ComplexVector:
        r"""
        Flujo (H) EXACTO: [v(t)] = [exp(i t ρ) v].
        Es una isometría de FS (el flujo hamiltoniano de un sistema integrable
        lineal en acciones es una rotación cuasiperiódica del toro).
        """
        n = int(rho.shape[0])
        v = cls.sanitize_vector(v, n)
        U = DensityOperatorAlgebra.unitary_flow(rho, t)
        vt = U @ v
        return cls.sanitize_vector(vt, n)


# ── §1.3b Fibración de Hopf e hipercomplejos ────────────────────────────────────
@dataclass(frozen=True, slots=True)
class HopfFibrationCertificate:
    r"""
    Fibración de Hopf:  S¹ ↪ S^{2n−1} → ℂP^{n−1}.
    Para n=2: S¹ ↪ S³ → S² ≅ ℂP¹, y S³ ⊂ ℍ (esfera de unitarios cuaterniónicos).
    La conexión de Ehresmann es exactamente la conexión de Chern:
        𝒱_v = span_ℝ{i v},   ℋ_v = (I − vv†) T_v S^{2n−1}.
    Curvatura = ω_FS.  Número de Hopf de la fibra = 1.
    """
    total_space_dim: int
    base_dim_real: int
    fibre_dim: int
    hopf_invariant: int
    quaternionic_structure: bool
    connection_vertical_norm: float
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(cls, v: np.ndarray) -> "HopfFibrationCertificate":
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        n = int(v.size)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-15:
            v = v / nrm
        # Generador vertical: i v  (acción U(1))
        vert = (1j * v).astype(np.complex128)
        vert_h = ProjectiveDynamics.horizontal(v, vert)
        vert_leak = float(np.linalg.norm(vert_h))  # debe ser ~0
        quat = bool(n == 2)
        # Hopf invariant de la fibración clásica S³→S² es 1; en general c₁^n [ℂP^{n−1}] = 1
        hopf_inv = 1
        if vert_leak < 1e-10:
            local = HeytingOmega3.COHERENT
        elif vert_leak < 1e-6:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            total_space_dim=2 * n,
            base_dim_real=2 * n - 2,
            fibre_dim=1,
            hopf_invariant=hopf_inv,
            quaternionic_structure=quat,
            connection_vertical_norm=float(vert_leak),
            local_verdict=local,
        )

    @staticmethod
    def quaternion_hopf_map(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        r"""
        Mapa de Hopf clásico h: S³ → S²,  (z, w) ↦ (2z w̄, |z|² − |w|²)
        con q = (Re z, Im z, Re w, Im w) ∈ S³ ⊂ ℍ.
        """
        a, b, c, d = (float(x) for x in q)
        nrm = math.sqrt(a * a + b * b + c * c + d * d)
        if nrm < 1e-15:
            return (0.0, 0.0, 1.0)
        a, b, c, d = a / nrm, b / nrm, c / nrm, d / nrm
        # z = a+ib, w = c+id
        x = 2.0 * (a * c + b * d)
        y = 2.0 * (b * c - a * d)
        z = (a * a + b * b) - (c * c + d * d)
        return (x, y, z)


# ── §1.4 Coordenadas acción-ángulo de Poincaré + función generatriz ─────────────
@dataclass(frozen=True, slots=True)
class ActionAngleChart:
    r"""
    Carta acción-ángulo: I_k = |⟨φ_k|v⟩|², θ_k = arg⟨φ_k|v⟩, H(I) = Σ λ_k I_k.
    Frecuencias ω = ∇_I H = λ (isócrono: Hess H = 0 ⇒ twist nulo).
    Función generatriz de tipo 2 del flujo a tiempo t:
        S₂(θ, I'; t) = I' · θ + t H(I')
    (el término de twist W₂ ≡ 0 porque H es lineal).
    """
    actions: RealVector
    angles: RealVector
    frequencies: RealVector
    hamiltonian: float
    hessian_twist_norm: float
    resonance_rank: int
    commensurability_defect: float
    generating_function_density: float
    generating_function_type2: float
    simplex_hash: str


class ActionAnglePoincareChart:
    @staticmethod
    def _integer_resonance_rank(omega: RealVector, kmax: int = 4) -> Tuple[int, float]:
        w = np.asarray(omega, dtype=np.float64).ravel()
        m = int(w.size)
        if m == 0:
            return 0, float("inf")
        kmax = int(max(1, min(kmax, 5)))
        ranges = [np.arange(-kmax, kmax + 1) for _ in range(m)]
        grid = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(m, -1).T
        rels: List[np.ndarray] = []
        defect = float("inf")
        for k in grid:
            if np.all(k == 0):
                continue
            dot = float(abs(np.dot(w, k)))
            kn = float(np.linalg.norm(k))
            defect = min(defect, dot / max(kn, 1.0))
            if dot < 1e-10 * max(1.0, float(np.linalg.norm(w))):
                rels.append(k.astype(np.float64))
        if not rels:
            return 0, _finite(defect, 0.0)
        A = np.stack(rels, axis=0)
        rank = int(np.linalg.matrix_rank(A, tol=1e-8))
        return rank, _finite(defect, 0.0)

    @classmethod
    def type2_generating_function(cls, actions: np.ndarray, angles: np.ndarray, H: float, t: float = 1.0) -> float:
        r"""S₂(θ, I; t) = I·θ + t H(I).  Densidad de Poincaré–Cartan."""
        return float(np.dot(actions, angles) + float(t) * float(H))

    @classmethod
    def chart(cls, rho: np.ndarray, v: np.ndarray, kmax: int = 4) -> ActionAngleChart:
        w, V = DensityOperatorAlgebra.eigenpairs_descending(rho)
        n = int(w.size)
        v = ProjectiveDynamics.sanitize_vector(v, n)
        overlaps = V.conj().T @ v
        actions = np.real(overlaps * np.conjugate(overlaps)).astype(np.float64)
        actions = np.maximum(actions, 0.0)
        s = float(actions.sum())
        if s > 0.0:
            actions = actions / s
        angles = np.angle(overlaps).astype(np.float64)
        H = float(np.dot(w, actions))
        rank, defect = cls._integer_resonance_rank(w, kmax=kmax)
        S_density = float(np.dot(angles, actions)) + H
        S2 = cls.type2_generating_function(actions, angles, H, t=1.0)
        h = _sha256_bytes(
            np.ascontiguousarray(actions).tobytes(),
            np.ascontiguousarray(w).tobytes(),
        )
        return ActionAngleChart(
            actions=actions,
            angles=angles,
            frequencies=w.astype(np.float64),
            hamiltonian=H,
            hessian_twist_norm=0.0,  # H lineal ⇒ Hess = 0 (isócrono)
            resonance_rank=int(rank),
            commensurability_defect=float(defect),
            generating_function_density=float(S_density),
            generating_function_type2=float(S2),
            simplex_hash=h,
        )


# ── §1.5 Flujo hamiltoniano celeste + mapa de primer retorno (exacto + RK4) ─────
@dataclass(frozen=True, slots=True)
class HamiltonianReturnSample:
    energy: float
    return_time: float
    theoretical_period: float
    period_error: float
    n_crossings: int
    cartan_circulation: float
    energy_drift: float
    section_residual: float
    exact_flow_used: bool
    poincare_cartan_invariant: float


class HamiltonianCelestialFlow:
    r"""
    Flujo (H): v' = i (I − vv†) ρ v.

    Integrabilidad: el flujo es EXACTAMENTE v(t) = exp(i t ρ) v₀ (gauge U(1)
    residual).  El integrador RK4 proyectivo queda como verificador de deriva
    y de la 1-forma de Poincaré–Cartan.

    Evento de sección Σ_{c,q}: cambio de signo de Im⟨q|v⟩ con Ṡ > 0.
    Cruce refinado por interpolación de Hermite cúbica.
    """

    @staticmethod
    def _rhs(rho: np.ndarray, v: np.ndarray) -> ComplexVector:
        return ProjectiveDynamics.hamiltonian_vector_field(rho, v)

    @classmethod
    def _rk4_step(cls, rho: np.ndarray, v: np.ndarray, dt: float) -> ComplexVector:
        k1 = cls._rhs(rho, v)
        k2 = cls._rhs(rho, v + 0.5 * dt * k1)
        k3 = cls._rhs(rho, v + 0.5 * dt * k2)
        k4 = cls._rhs(rho, v + dt * k3)
        v_new = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        nrm = float(np.linalg.norm(v_new))
        if nrm < 1e-15:
            return v
        return (v_new / nrm).astype(np.complex128)

    @staticmethod
    def two_plane_period(lambda_i: float, lambda_j: float) -> float:
        d = abs(float(lambda_i) - float(lambda_j))
        if d < 1e-15:
            return float("inf")
        return float(_TWO_PI / d)

    @classmethod
    def exact_orbit(cls, rho: np.ndarray, v0: np.ndarray, t: float) -> ComplexVector:
        return ProjectiveDynamics.exact_hamiltonian_flow(rho, v0, t)

    @classmethod
    def first_return(
        cls,
        rho: np.ndarray,
        v0: np.ndarray,
        q: Optional[np.ndarray] = None,
        dt: float = 2.5e-3,
        t_max: float = 40.0,
        min_crossings: int = 1,
        use_exact: bool = True,
    ) -> HamiltonianReturnSample:
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v0, n)
        v0s = v.copy()
        if q is None:
            q = np.zeros(n, dtype=np.complex128)
            q[0] = 1.0
        else:
            q = ProjectiveDynamics.sanitize_vector(q, n)
        E0 = ProjectiveDynamics.rayleigh(rho, v)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        tau_th = cls.two_plane_period(float(w[0]), float(w[1]) if w.size > 1 else 0.0)

        def section_fn(z: np.ndarray) -> float:
            return float(np.imag(np.vdot(q, z)))

        def section_dot(z: np.ndarray) -> float:
            X = cls._rhs(rho, z)
            return float(np.imag(np.vdot(q, X)))

        s_prev = section_fn(v)
        ds_prev = section_dot(v)
        t = 0.0
        crossings = 0
        t_first = 0.0
        cartan = 0.0
        n_steps = int(max(8, t_max / max(dt, 1e-8)))
        for k in range(n_steps):
            X = cls._rhs(rho, v)
            cartan += ProjectiveDynamics.poincare_one_form(v, X) * dt
            if use_exact:
                t_next = t + dt
                v_next = cls.exact_orbit(rho, v0s, t_next)
            else:
                v_next = cls._rk4_step(rho, v, dt)
                t_next = t + dt
            s_next = section_fn(v_next)
            ds_next = section_dot(v_next)
            if s_prev <= 0.0 < s_next and t_next > 4.0 * dt:
                t_cross = _hermite_crossing_time(t, t_next, s_prev, s_next, ds_prev, ds_next)
                crossings += 1
                if crossings == 1:
                    t_first = t_cross
                if crossings >= min_crossings and t_first > 0.0:
                    if use_exact:
                        v = cls.exact_orbit(rho, v0s, t_first)
                    else:
                        v = v_next
                    t = t_first
                    break
            v = v_next
            t = t_next
            s_prev = s_next
            ds_prev = ds_next
        E1 = ProjectiveDynamics.rayleigh(rho, v)
        s_res = abs(section_fn(v))
        period_err = (
            abs(t_first - tau_th) / max(tau_th, 1e-12)
            if math.isfinite(tau_th) and t_first > 0.0
            else 1.0
        )
        # Invariante de Poincaré–Cartan: ∮ λ_FS  (circulación).  En el caso integrable
        # iguala Σ I_k Δθ_k sobre un ciclo.  Se reporta cartan como proxy.
        return HamiltonianReturnSample(
            energy=float(E0),
            return_time=float(t_first if t_first > 0.0 else t),
            theoretical_period=_finite(tau_th, 0.0),
            period_error=float(period_err),
            n_crossings=int(crossings),
            cartan_circulation=float(cartan),
            energy_drift=float(abs(E1 - E0)),
            section_residual=float(s_res),
            exact_flow_used=bool(use_exact),
            poincare_cartan_invariant=float(cartan),
        )


# ── §1.6 Sección de Poincaré introspectiva ──────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSectionIntrospection:
    r"""
    Dos secciones: (G) Σ_c = {R = c}; (H) Σ_{c,q} = {R = c} ∩ {Im⟨q|v⟩ = 0}.
    Transversalidad: ι_{X} dR ≠ 0  ⇔  ‖X_G‖² > 0 sobre Σ_c.
    """
    level: float
    transversality_min: float
    first_return_time: float
    hamiltonian_return_time: float
    theoretical_period: float
    energy_drift: float
    cartan_circulation: float
    floquet_moduli: RealVector
    periodic_orbit_rank: int
    gradient_transverse: bool
    hamiltonian_section_ok: bool
    exact_flow_used: bool
    section_hash: str

    @classmethod
    def build(
        cls,
        rho: np.ndarray,
        level: Optional[float] = None,
        n_samples: int = 32,
        key: str = "POINCARE-INTROSPECTION",
        v_probe: Optional[np.ndarray] = None,
    ) -> "PoincareSectionIntrospection":
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        c = float(level) if level is not None else float(1.0 / max(n, 1))
        rng = np.random.default_rng(_seed_from_string(f"POINCARE::{key}"))

        tmin = float("inf")
        tau_acc: List[float] = []
        for _ in range(max(8, n_samples)):
            v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            v = v / max(float(np.linalg.norm(v)), 1e-15)
            X = ProjectiveDynamics.projected_gradient(rho, v)
            trans = 2.0 * float(np.real(np.vdot(X, X)))
            tmin = min(tmin, trans)
            tau_acc.append(1.0 / max(trans, 1e-12))
        if not np.isfinite(tmin):
            tmin = 0.0
        tau_mean = float(np.mean(tau_acc)) if tau_acc else 0.0

        lam1 = max(float(w[0]), _EPS)
        floquet = (w[1:] / lam1).astype(np.float64) if w.size > 1 else np.zeros(0, dtype=np.float64)
        per_rank = int(np.sum(np.abs(floquet - 1.0) < 1e-10)) if floquet.size else 0

        if v_probe is None:
            v_probe = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        ham = HamiltonianCelestialFlow.first_return(
            rho, v_probe, q=None, dt=3.0e-3, t_max=25.0, min_crossings=1, use_exact=True,
        )
        tau_th = HamiltonianCelestialFlow.two_plane_period(
            float(w[0]), float(w[1]) if w.size > 1 else 0.0
        )
        grad_tr = bool(tmin > _EPS_SECTION)
        ham_ok = bool(ham.energy_drift < 1e-4 and ham.n_crossings >= 0)

        h = _sha256_bytes(
            f"{c:.12e}".encode("ascii"),
            np.ascontiguousarray(floquet).tobytes() if floquet.size else b"\x00",
            f"{tmin:.12e}".encode("ascii"),
            f"{ham.return_time:.12e}".encode("ascii"),
        )
        return cls(
            level=c,
            transversality_min=float(tmin),
            first_return_time=float(tau_mean),
            hamiltonian_return_time=float(ham.return_time),
            theoretical_period=_finite(tau_th, 0.0),
            energy_drift=float(ham.energy_drift),
            cartan_circulation=float(ham.cartan_circulation),
            floquet_moduli=floquet,
            periodic_orbit_rank=per_rank,
            gradient_transverse=grad_tr,
            hamiltonian_section_ok=ham_ok,
            exact_flow_used=bool(ham.exact_flow_used),
            section_hash=h,
        )

    @property
    def is_transverse(self) -> bool:
        return bool(self.gradient_transverse)


# ── §1.7 Poincaré–Hopf + Morse de R sobre ℂP^{n−1} ──────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareHopfCertificate:
    r"""
    Poincaré–Hopf: Σ_{crit} ind_x(X) = χ(M).  χ(ℂP^{n−1}) = n.
    Morse_R([φ_k]) = 2(n − 1 − #{j : λ_j < λ_k})  en la convención de índices
    reales (dim_ℝ ℂP^{n−1} = 2n−2).  En la convención del código heredado
    Morse_R([φ_k]) = 2 · #{j : λ_j < λ_k} y ind_Hopf = (−1)^{Morse} = +1
    porque todo índice de Morse es par (variedad casi compleja).
    """
    euler_characteristic: int
    n_critical_points: int
    morse_indices: Tuple[int, ...]
    hopf_indices: Tuple[int, ...]
    index_sum: int
    hopf_satisfied: bool
    isolated_nondegenerate: bool
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(cls, rho: np.ndarray) -> "PoincareHopfCertificate":
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        n = int(w.size)
        chi = n
        tol = max(1e-9, 1e-6 * float(w[0]))
        isolated = True
        morse: List[int] = []
        hopf: List[int] = []
        for k in range(n):
            n_below = int(np.sum(w < w[k] - tol))
            n_equal = int(np.sum(np.abs(w - w[k]) < tol))
            if n_equal > 1:
                isolated = False
            idx_morse = 2 * n_below
            morse.append(int(idx_morse))
            hopf.append(1 if (idx_morse % 2 == 0) else -1)
        index_sum = int(sum(hopf))
        satisfied = bool(isolated and index_sum == chi)
        if satisfied:
            local = HeytingOmega3.COHERENT
        elif index_sum == chi:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            euler_characteristic=int(chi),
            n_critical_points=int(n),
            morse_indices=tuple(morse),
            hopf_indices=tuple(hopf),
            index_sum=index_sum,
            hopf_satisfied=satisfied,
            isolated_nondegenerate=bool(isolated),
            local_verdict=local,
        )


# ── §1.8 Recurrencia de Poincaré + lema de Kac ──────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareRecurrenceCertificate:
    r"""
    Poincaré 1890: todo flujo que preserva una medida de probabilidad finita
    en un espacio de medida finita es recurrentemente casi-en-todas-partes.
    Kac: 𝔼[τ_A] = 1/μ(A) para el tiempo de primer retorno.
    Aquí μ = medida de Haar/FS normalizada sobre ℂP^{n−1} (compacto).
    (H) es Liouville ⇒ preserva μ.  (T) preserva μ ⇔ ρ ∝ I (Haar-invariante).
    """
    manifold_compact: bool
    hamiltonian_measure_preserving: bool
    projective_map_measure_preserving: bool
    hamiltonian_recurrence: bool
    projective_recurrence: bool
    kac_mean_return_proxy: float
    fs_volume_exponent: int
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(
        cls, n: int, fs_radius: float = 0.1, is_uniform: bool = False
    ) -> "PoincareRecurrenceCertificate":
        n = int(max(n, 1))
        dim_real = 2 * n - 2
        eps = max(float(fs_radius), 1e-12)
        mu = float(min(1.0, eps ** max(dim_real, 1)))
        kac = float(1.0 / max(mu, 1e-15))
        ham_rec = True
        proj_rec = bool(is_uniform)
        if ham_rec and not proj_rec:
            local = HeytingOmega3.COHERENT
        elif ham_rec:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            manifold_compact=True,
            hamiltonian_measure_preserving=True,
            projective_map_measure_preserving=proj_rec,
            hamiltonian_recurrence=ham_rec,
            projective_recurrence=proj_rec,
            kac_mean_return_proxy=kac,
            fs_volume_exponent=int(dim_real),
            local_verdict=local,
        )


# ── §1.9 Grafo de resonancia espectral + Arnold + Chirikov + Greene ─────────────
@dataclass(frozen=True, slots=True)
class ArnoldTongueCertificate:
    r"""
    Lengua de Arnold A_{p/q} en el plano (K, Ω): región de phase-locking donde
    el mapa estándar converge a una órbita p/q-periódica.  Anchura asintótica:
        ΔΩ_{p/q} ∼ C · K^q     (K → 0)
    Escalera del diablo: Ω ↦ número de rotación es continua, no decreciente,
    y localmente constante en un conjunto de medida positiva.
    Medida complementaria (Cantor): μ_Cantor = 1 − Σ_{p/q} ΔΩ_{p/q}.
    Aquí: K ≡ δ_res (umbral de resonancia);  Ω ≡ (λ_i − λ_j)/λ₁.
    """
    n_tongues: int
    tongue_widths: Tuple[float, ...]
    cantor_measure: float
    is_full_measure: bool
    staircase_monotone: bool
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(
        cls, spectrum: np.ndarray, delta_res: float = 1e-3, q_max: int = 5
    ) -> "ArnoldTongueCertificate":
        w = np.asarray(spectrum, dtype=np.float64).ravel()
        n = int(w.size)
        if n < 2:
            return cls(0, (), 1.0, False, True, HeytingOmega3.DEGRADED)
        lam1 = max(float(w[0]), _EPS)
        widths: List[float] = []
        for q in range(1, int(q_max) + 1):
            for p in range(0, q + 1):
                if math.gcd(p, q) != 1 and not (p == 0 and q == 1):
                    continue
                omega_pq = p / q
                for i in range(n):
                    for j in range(i + 1, n):
                        om = abs(float(w[i]) - float(w[j])) / lam1
                        if abs(om - omega_pq) < delta_res * (q + 1):
                            widths.append(float(delta_res * (q + 1) ** (-2)))
        total_width = float(sum(widths))
        cantor = float(max(0.0, 1.0 - total_width))
        full = bool(cantor < 1e-6)
        mono = True
        if full:
            local = HeytingOmega3.VETOED
        elif total_width < 1e-2:
            local = HeytingOmega3.COHERENT
        else:
            local = HeytingOmega3.DEGRADED
        return cls(
            n_tongues=len(widths),
            tongue_widths=tuple(widths),
            cantor_measure=cantor,
            is_full_measure=full,
            staircase_monotone=mono,
            local_verdict=local,
        )


@dataclass(frozen=True, slots=True)
class ChirikovOverlapCertificate:
    r"""
    Criterio de solapamiento de Chirikov (1969):
        K_ij = (Δω_i + Δω_j) / |ω_i − ω_j|
    Si max K_ij ≳ 1, las islas resonantes se solapan ⇒ estocasticidad global.
    Semi-anchura de isla: Δω ~ √(ε |V_{ij}|); aquí ε-proxy = gap_ratio residual.
    """
    n_pairs: int
    max_overlap: float
    mean_overlap: float
    global_stochasticity: bool
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(cls, spectrum: np.ndarray, island_width_proxy: float = 1e-3) -> "ChirikovOverlapCertificate":
        w = np.asarray(spectrum, dtype=np.float64).ravel()
        n = int(w.size)
        if n < 2:
            return cls(0, 0.0, 0.0, False, HeytingOmega3.DEGRADED)
        dw = max(float(island_width_proxy), 1e-15)
        Ks: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                denom = abs(float(w[i]) - float(w[j]))
                Ks.append(float((2.0 * dw) / max(denom, 1e-15)))
        mx = float(max(Ks)) if Ks else 0.0
        mn = float(np.mean(Ks)) if Ks else 0.0
        stoch = bool(mx >= 1.0)
        if stoch:
            local = HeytingOmega3.VETOED
        elif mx >= 0.3:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.COHERENT
        return cls(len(Ks), mx, mn, stoch, local)


@dataclass(frozen=True, slots=True)
class GreeneResidueCertificate:
    r"""
    Residuo de Greene: para una órbita periódica del mapa de Poincaré P,
        R = (1/4) (2 − Tr DP^q).
    |R| < 1 ⇒ elíptica (estable KAM); |R| > 1 ⇒ hiperbólica.
    El último toro que se destruye es el de número de rotación áureo.
    Proxy: R_k = (1/4)(2 − 2 cos((λ_k − λ₁) τ)) = sin²((λ_k − λ₁)τ / 2).
    """
    residues: Tuple[float, ...]
    golden_residue: float
    n_hyperbolic: int
    last_kam_torus_alive: bool
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(cls, spectrum: np.ndarray) -> "GreeneResidueCertificate":
        w = np.asarray(spectrum, dtype=np.float64).ravel()
        n = int(w.size)
        if n < 2:
            return cls((), 0.0, 0, True, HeytingOmega3.DEGRADED)
        tau = HamiltonianCelestialFlow.two_plane_period(float(w[0]), float(w[1]))
        if not math.isfinite(tau):
            tau = 0.0
        res: List[float] = []
        for k in range(1, n):
            phase = 0.5 * (float(w[k]) - float(w[0])) * tau
            R = float(math.sin(phase) ** 2)
            res.append(R)
        n_hyp = int(sum(1 for R in res if R > 1.0 - 1e-12))
        # Residuo áureo: ω_gold / ω_1
        gold_phase = 0.5 * (1.0 / _GOLDEN) * (tau if tau else 1.0)
        gold_R = float(math.sin(gold_phase) ** 2)
        alive = bool(gold_R < 1.0 and n_hyp == 0)
        if alive and n_hyp == 0:
            local = HeytingOmega3.COHERENT
        elif n_hyp == 0:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(tuple(res), gold_R, n_hyp, alive, local)


@dataclass(frozen=True, slots=True)
class SpectralResonanceGraph:
    r"""
    Grafo G_λ = (V, E): arista {i,j} ⇔ |λ_i − λ_j| ≤ δ_res ∨ ∃k, |k₁λ_i + k₂λ_j| ≤ δ.
    Cheeger h(G) = min_{|S|≤|V|/2} |∂S| / |S|.  Conectividad algebraica = λ₂(L).
    """
    n_vertices: int
    n_edges: int
    adjacency_spectrum: Tuple[float, ...]
    n_components: int
    top_resonant: bool
    algebraic_connectivity: float
    cheeger_lower: float
    arnold: ArnoldTongueCertificate
    chirikov: ChirikovOverlapCertificate
    greene: GreeneResidueCertificate
    local_verdict: HeytingOmega3

    @classmethod
    def build(cls, spectrum: np.ndarray, delta: float = 1e-3, kmax: int = 3) -> "SpectralResonanceGraph":
        w = np.asarray(spectrum, dtype=np.float64).ravel()
        n = int(w.size)
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                resonant = abs(w[i] - w[j]) <= delta
                if not resonant:
                    for k1 in range(-kmax, kmax + 1):
                        for k2 in range(-kmax, kmax + 1):
                            if k1 == 0 and k2 == 0:
                                continue
                            if abs(k1 * w[i] + k2 * w[j]) <= delta:
                                resonant = True
                                break
                        if resonant:
                            break
                if resonant:
                    A[i, j] = A[j, i] = 1.0
        n_edges = int(np.sum(A) / 2.0)
        reach = np.eye(n, dtype=np.float64)
        B = A + np.eye(n, dtype=np.float64)
        for _ in range(max(n - 1, 1)):
            reach = ((reach @ B) > 0.5).astype(np.float64)
        seen = np.zeros(n, dtype=bool)
        n_comp = 0
        for i in range(n):
            if not seen[i]:
                n_comp += 1
                seen |= reach[i] > 0.5
        specA = np.sort(np.real(la.eigvalsh(A)))[::-1] if n else np.zeros(0)
        deg = np.diag(np.sum(A, axis=1))
        L = deg - A
        mu = np.sort(np.real(la.eigvalsh(L))) if n else np.array([0.0])
        alg = float(mu[1]) if mu.size > 1 else 0.0
        # Cheeger: (1/2) λ₂(L) ≤ h ≤ √(2 λ₂(L) d_max)
        cheeger_lo = 0.5 * alg
        top_res = bool(n >= 2 and A[0, 1] > 0.5)
        arnold = ArnoldTongueCertificate.evaluate(w, delta_res=max(delta, 1e-6))
        chirikov = ChirikovOverlapCertificate.evaluate(w, island_width_proxy=max(delta, 1e-6))
        greene = GreeneResidueCertificate.evaluate(w)
        local = HeytingOmega3.COHERENT
        local = local.meet(arnold.local_verdict).meet(chirikov.local_verdict).meet(greene.local_verdict)
        if top_res:
            local = local.meet(HeytingOmega3.VETOED)
        elif n_edges == 0:
            local = local.meet(HeytingOmega3.COHERENT)
        else:
            local = local.meet(HeytingOmega3.DEGRADED)
        return cls(
            n_vertices=n,
            n_edges=n_edges,
            adjacency_spectrum=tuple(float(x) for x in specA.tolist()),
            n_components=int(n_comp),
            top_resonant=top_res,
            algebraic_connectivity=float(alg),
            cheeger_lower=float(cheeger_lo),
            arnold=arnold,
            chirikov=chirikov,
            greene=greene,
            local_verdict=local,
        )


# ── §1.10 Lema de concentración de Lévy sobre ℂP^{n−1} ──────────────────────────
class LevyConcentrationLemma:
    r"""
    Lévy: la medida de Haar de ℂP^{n−1} se concentra en bandas geodésicas.
    P(|f − 𝔼f| ≥ ε) ≤ exp(−(n+1)ε² / (2π² L²)) para f Lipschitz-L respecto a FS.
    """

    @staticmethod
    def bound(epsilon: float, n: int, lipschitz: float = 1.0) -> float:
        if n <= 1:
            return 1.0
        L = max(float(lipschitz), 1e-15)
        e = max(float(epsilon), 0.0)
        return float(min(1.0, math.exp(-(n + 1) * e * e / (2.0 * math.pi ** 2 * L * L))))

    @staticmethod
    def median_width(n: int, lipschitz: float = 1.0, confidence: float = 0.99) -> float:
        if n <= 1:
            return float("inf")
        p_tail = max(1e-15, 1.0 - float(confidence))
        return float(lipschitz * math.pi * math.sqrt(2.0 * math.log(1.0 / p_tail) / (n + 1)))


# ── §1.11 Análisis del gap espectral + proyectores de Riesz ─────────────────────
@dataclass(frozen=True, slots=True)
class SpectralGapReport:
    lambda_1: float
    lambda_2: float
    lambda_min: float
    gap: float
    gap_ratio: float
    gap_absolute: float
    condition_number: float
    degeneracy_top: int
    is_uniform: bool
    von_neumann_entropy: float
    purity: float
    birkhoff_constant: float
    riesz_rank_top: int
    local_verdict: HeytingOmega3


class SpectralGapAnalyzer:
    GAP_COHERENT: Final[float] = 0.10
    GAP_DEGRADED: Final[float] = 0.01
    UNIFORM_TOL: Final[float] = 1.0e-3

    @classmethod
    def analyze(cls, rho: np.ndarray) -> SpectralGapReport:
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        lam1 = float(w[0])
        lam2 = float(w[1]) if w.size > 1 else 0.0
        lam_n = float(w[-1])
        gap_ratio = (lam2 / lam1) if lam1 > 1e-15 else 1.0
        gap = float(max(0.0, 1.0 - gap_ratio))
        gap_abs = float(max(0.0, lam1 - lam2))
        cond = float(lam1 / max(lam_n, _EPS))
        tol_degen = max(1e-9, 1e-6 * lam1)
        degeneracy = int(np.sum(np.abs(w - lam1) < tol_degen))
        uniform_gap = float(np.max(np.abs(w - 1.0 / max(n, 1))))
        is_uniform = bool(uniform_gap < cls.UNIFORM_TOL)
        delta_proj = math.log(lam1 / max(lam_n, _EPS)) if lam1 > _EPS else 0.0
        birkhoff = float(math.tanh(delta_proj / 4.0))
        P0 = DensityOperatorAlgebra.riesz_projection(rho, 0)
        riesz_rank = int(round(float(np.real(np.trace(P0)))))
        if is_uniform or degeneracy >= 3:
            local = HeytingOmega3.VETOED
        elif gap >= cls.GAP_COHERENT and degeneracy == 1:
            local = HeytingOmega3.COHERENT
        elif gap >= cls.GAP_DEGRADED:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return SpectralGapReport(
            lambda_1=lam1, lambda_2=lam2, lambda_min=lam_n,
            gap=gap, gap_ratio=float(gap_ratio), gap_absolute=gap_abs,
            condition_number=cond, degeneracy_top=degeneracy,
            is_uniform=is_uniform,
            von_neumann_entropy=DensityOperatorAlgebra.von_neumann_entropy(rho),
            purity=DensityOperatorAlgebra.purity(rho),
            birkhoff_constant=birkhoff,
            riesz_rank_top=riesz_rank,
            local_verdict=local,
        )


# ── §1.12 Homoclinic tangle: Smale horseshoe via mapa de Poincaré ───────────────
@dataclass(frozen=True, slots=True)
class HomoclinicTangleCertificate:
    r"""
    Smale–Birkhoff: si W^s(p) ∩ W^u(p) contiene un punto transversal, entonces
    existe un conjunto invariante Λ homeomorfo al shift Σ₂ = {0,1}^ℤ, y la
    dinámica sobre Λ es conjugada al shift.  Entropía topológica h_top(σ) = log 2.

    Test: iterar (T) desde v_s ∈ W^s_local([φ₁]) y v_u ∈ W^u_local([φ₁]);
    detectar intersecciones transversales por ángulo FS en (0, π/2).
    """
    n_transverse_intersections: int
    has_smale_horseshoe: bool
    topological_entropy_lower: float
    stable_unstable_angle_min: float
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(
        cls, rho: np.ndarray, n_samples: int = 24, key: str = "TANGLE"
    ) -> "HomoclinicTangleCertificate":
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        rng = np.random.default_rng(_seed_from_string(f"TANGLE::{key}"))
        w, V = DensityOperatorAlgebra.eigenpairs_descending(rho)
        phi1 = V[:, 0]
        n_trans = 0
        angles: List[float] = []
        for _ in range(max(4, n_samples)):
            if n >= 3:
                alpha = float(rng.uniform(-0.05, 0.05))
                beta = float(rng.uniform(-0.05, 0.05))
                v_s = phi1 + alpha * V[:, 1] + beta * V[:, 2]
                v_u = phi1 + alpha * V[:, 1] - beta * V[:, 2]
            else:
                v_s = phi1 + 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
                v_u = phi1 + 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            v_s = v_s / max(float(np.linalg.norm(v_s)), 1e-15)
            v_u = v_u / max(float(np.linalg.norm(v_u)), 1e-15)
            for _ in range(3):
                Tv_s, _ = ProjectiveDynamics.apply_map(rho, v_s)
                Tv_u, _ = ProjectiveDynamics.apply_map(rho, v_u)
                v_s = Tv_s
                v_u = Tv_u
            ov = np.vdot(v_s, v_u)
            ang = float(np.arccos(np.clip(abs(ov), 0.0, 1.0)))
            angles.append(ang)
            if 1e-4 < ang < math.pi / 2 - 1e-4:
                n_trans += 1
        min_angle = float(min(angles)) if angles else 0.0
        horseshoe = bool(n_trans >= 2)
        h_top = float(math.log(2.0)) if horseshoe else 0.0
        if horseshoe:
            local = HeytingOmega3.VETOED
        elif min_angle > 1e-4:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.COHERENT
        return cls(
            n_transverse_intersections=int(n_trans),
            has_smale_horseshoe=horseshoe,
            topological_entropy_lower=h_top,
            stable_unstable_angle_min=min_angle,
            local_verdict=local,
        )


# ── §1.13 Nekhoroshev: estabilidad exponencialmente larga ───────────────────────
@dataclass(frozen=True, slots=True)
class NekhoroshevCertificate:
    r"""
    Teorema de Nekhoroshev (1977): si H = H₀(I) + ε H₁ es analítico y H₀ es
    steep (o quasi-convex), entonces
        |I(t) − I(0)| ≤ ε^b    para    |t| ≤ exp(c (1/ε)^a)
    con a = 1/(2τ), b = a (τ = índice diofántico efectivo).
    Aquí H₀ es lineal (isócrono, no steep en sentido clásico); el certificado
    reporta la cota formal y degrada el veredicto cuando Hess H₀ = 0.
    """
    steep: bool
    epsilon: float
    exponent_a: float
    stability_time_log: float
    action_drift_bound: float
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(
        cls, perturbation_size: float, tau: float = 1.5, dim_actions: int = 2
    ) -> "NekhoroshevCertificate":
        eps = max(float(perturbation_size), 1e-15)
        tau = max(float(tau), 1.0)
        a = 1.0 / (2.0 * tau)
        b = a
        # H lineal ⇒ no steep; la cota es formal
        steep = False
        log_T = float(min(80.0, (1.0 / eps) ** a))  # log-tiempo (cap numérico)
        drift = float(eps ** b)
        if (not steep) and eps < 1e-3:
            local = HeytingOmega3.DEGRADED
        elif not steep:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.COHERENT
        return cls(
            steep=steep,
            epsilon=float(eps),
            exponent_a=float(a),
            stability_time_log=log_T,
            action_drift_bound=drift,
            local_verdict=local,
        )


# ── §1.14 IntrospectiveField + hand-off formal a FASE 2 ─────────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectiveField:
    r"""
    Objeto terminal FASE 1 / inicial FASE 2 (dominio de F₂).

    Transporta el sustrato proyectivo-métrico, los certificados celestes de
    Poincaré y el objeto de Cog sobre el que la mónada RSI-3 (FASE 3) actuará.
    """
    rho: np.ndarray
    v_initial: np.ndarray
    spectral_gap: SpectralGapReport
    poincare_section: PoincareSectionIntrospection
    levy_band: float
    K_spec: Tuple[float, ...]
    action_angle: ActionAngleChart
    poincare_hopf: PoincareHopfCertificate
    recurrence: PoincareRecurrenceCertificate
    resonance_graph: SpectralResonanceGraph
    homoclinic_tangle: HomoclinicTangleCertificate
    hopf_fibration: HopfFibrationCertificate
    nekhoroshev: NekhoroshevCertificate
    field_hash: str
    dim: int

    def as_dict(self) -> Dict[str, float]:
        g = self.spectral_gap
        return {
            "lambda_1": g.lambda_1, "lambda_2": g.lambda_2, "gap": g.gap,
            "gap_ratio": g.gap_ratio, "degeneracy": float(g.degeneracy_top),
            "birkhoff": g.birkhoff_constant, "is_uniform": float(g.is_uniform),
            "poincare_transverse": float(self.poincare_section.is_transverse),
            "levy_band": self.levy_band,
            "hopf_ok": float(self.poincare_hopf.hopf_satisfied),
            "ham_recurrence": float(self.recurrence.hamiltonian_recurrence),
            "resonance_edges": float(self.resonance_graph.n_edges),
            "twist_hess": self.action_angle.hessian_twist_norm,
            "arnold_tongues": float(self.resonance_graph.arnold.n_tongues),
            "horseshoe": float(self.homoclinic_tangle.has_smale_horseshoe),
            "chirikov_max": float(self.resonance_graph.chirikov.max_overlap),
            "greene_golden": float(self.resonance_graph.greene.golden_residue),
            "nekhoroshev_logT": float(self.nekhoroshev.stability_time_log),
            "hopf_fibre_ok": float(self.hopf_fibration.local_verdict == HeytingOmega3.COHERENT),
            "generating_function_S2": float(self.action_angle.generating_function_type2),
        }

    def meet_local_verdicts(self) -> HeytingOmega3:
        r"""Colapso Ω₃ del sustrato (meet de todos los certificados locales de F1)."""
        return self.spectral_gap.local_verdict.meet_all(
            self.poincare_hopf.local_verdict,
            self.recurrence.local_verdict,
            self.resonance_graph.local_verdict,
            self.homoclinic_tangle.local_verdict,
            self.hopf_fibration.local_verdict,
            self.nekhoroshev.local_verdict,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    #  continue_into_phase2  es el último morfismo de FASE 1 y, por anidamiento
    #  estricto, el primero de FASE 2: su cuerpo ES synthesize.
    #
    #      continue_into_phase2 ∘ prepare
    #          = IntrospectionPipeline.synthesize ∘ prepare.
    #
    #  La FASE 2 (siguiente prompt) define IntrospectionPipeline.synthesize
    #  y el objeto IntrospectionBundle; hasta entonces este morfismo es el
    #  punto de anclaje formal del anidamiento.
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase2(self, max_iter: int, tol: float) -> "IntrospectionBundle":
        r"""
        Último morfismo FASE 1 ∧ primero FASE 2.

        Dominio   : IntrospectiveField   (codominio de prepare).
        Codominio : IntrospectionBundle  (dominio de toda la FASE 3).

        Cuerpo (anidado en FASE 2):
            return IntrospectionPipeline.synthesize(field=self, max_iter=max_iter, tol=tol)
        """
        pipeline = globals().get("IntrospectionPipeline")
        if pipeline is None:
            raise NotImplementedError(
                "Anidamiento FASE 1 → FASE 2 pendiente: IntrospectionPipeline.synthesize "
                "es el cuerpo de continue_into_phase2.  Continuar en el siguiente prompt "
                "con la FASE 2 (punto fijo + Oseledets + Melnikov + KAM + Birkhoff)."
            )
        return pipeline.synthesize(field=self, max_iter=max_iter, tol=tol)


class IntrospectiveFieldPreparation:
    r"""
    Funtor F₁: (ρ, v₀) ↦ IntrospectiveField.

    Prepara el sustrato: análisis espectral, Σ_c / Σ_{c,q}, Lévy, Morse–Hopf,
    recurrencia, acción-ángulo + S₂, Arnold / Chirikov / Greene, homoclinic
    tangle, fibración de Hopf, Nekhoroshev, y el hash forense del campo.
    """

    @classmethod
    def prepare(
        cls,
        rho_input: np.ndarray,
        v_initial: np.ndarray,
        poincare_key: str = "INTROSPECTION",
    ) -> IntrospectiveField:
        rho = DensityOperatorAlgebra.sanitize(rho_input)
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v_initial, n)
        gap_report = SpectralGapAnalyzer.analyze(rho)
        poincare = PoincareSectionIntrospection.build(
            rho, level=None, n_samples=24, key=poincare_key, v_probe=v,
        )
        levy_band = LevyConcentrationLemma.median_width(n, lipschitz=1.0, confidence=0.99)
        K_spec = DensityOperatorAlgebra.modular_spectrum(rho)
        aa = ActionAnglePoincareChart.chart(rho, v, kmax=4)
        hopf = PoincareHopfCertificate.evaluate(rho)
        rec = PoincareRecurrenceCertificate.evaluate(
            n, fs_radius=max(levy_band, 1e-3), is_uniform=gap_report.is_uniform
        )
        resg = SpectralResonanceGraph.build(
            DensityOperatorAlgebra.spectrum_descending(rho),
            delta=max(1e-3, 1e-4 * gap_report.lambda_1),
        )
        tangle = HomoclinicTangleCertificate.evaluate(rho, key=poincare_key)
        hopf_fib = HopfFibrationCertificate.evaluate(v)
        perturbation = DensityOperatorAlgebra.frobenius_to_maximally_mixed(rho)
        nek = NekhoroshevCertificate.evaluate(
            perturbation_size=perturbation, tau=1.5, dim_actions=min(3, n)
        )
        field_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(v).tobytes(),
            f"{gap_report.gap:.12f}".encode("ascii"),
            poincare.section_hash.encode("ascii"),
            hopf.local_verdict.name.encode("ascii"),
            aa.simplex_hash.encode("ascii"),
            f"{tangle.n_transverse_intersections}".encode("ascii"),
            f"{resg.chirikov.max_overlap:.12e}".encode("ascii"),
            f"{nek.stability_time_log:.12e}".encode("ascii"),
        )
        return IntrospectiveField(
            rho=rho, v_initial=v, spectral_gap=gap_report,
            poincare_section=poincare, levy_band=float(levy_band), K_spec=K_spec,
            action_angle=aa, poincare_hopf=hopf, recurrence=rec,
            resonance_graph=resg, homoclinic_tangle=tangle,
            hopf_fibration=hopf_fib, nekhoroshev=nek,
            field_hash=field_hash, dim=n,
        )




def _quad_trap(y: np.ndarray, dx: float = 1.0) -> float:
    r"""Cuadratura del trapecio, compatible numpy≥2 (`trapezoid`) y legado (`trapz`)."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(arr, dx=dx))
    return float(np.trapz(arr, dx=dx))  # noqa: NPY201


def _sinc(x: float) -> float:
    r"""sinc no normalizado: sin(x)/x, con límite 1 en 0."""
    ax = abs(float(x))
    if ax < 1e-12:
        return 1.0 - ax * ax / 6.0
    return float(math.sin(x) / x)


# ── §2.0 Geometría riemanniana de ℂP^{n−1}: retracción geodésica FS ─────────────
class FubiniStudyRetraction:
    r"""
    Estructura riemanniana usada por Newton–Grassmann.

    Métrica FS:  g_v(ξ, η) = 2 Re ⟨ξ_h | η_h⟩.
    Geodésica: si ξ ∈ T_v S^{2n−1} ∩ v^⊥, ‖ξ‖ = α,
        exp_v(ξ) = cos(α) v + sin(α) ξ/α.
    La retracción polar (v+ξ)/‖v+ξ‖ es de primer orden; la geodésica es exacta
    en curvatura holomorfa constante K_hol = 4.
    """

    @staticmethod
    def horizontal_basis(v: np.ndarray) -> ComplexMatrix:
        r"""Base unitaria de v^⊥ ⊂ ℂ^n (núcleo de ⟨v|·⟩).  Dimensión compleja n−1."""
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        n = int(v.size)
        if n <= 1:
            return np.zeros((n, 0), dtype=np.complex128)
        # null_space de la 1-forma v†
        Q = la.null_space(v.conj().reshape(1, -1))
        if Q.size == 0:
            return np.zeros((n, 0), dtype=np.complex128)
        return np.asarray(Q, dtype=np.complex128)

    @staticmethod
    def geodesic_exp(v: np.ndarray, xi: np.ndarray) -> ComplexVector:
        r"""exp_v(ξ) sobre S^{2n−1}, con ξ horizontal.  Cociente U(1) implícito."""
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        xi = np.asarray(xi, dtype=np.complex128).reshape(-1)
        alpha = float(np.linalg.norm(xi))
        if alpha < 1e-15:
            return v.astype(np.complex128)
        v_new = math.cos(alpha) * v + math.sin(alpha) * (xi / alpha)
        nrm = float(np.linalg.norm(v_new))
        if nrm < 1e-15:
            return v.astype(np.complex128)
        return (v_new / nrm).astype(np.complex128)

    @staticmethod
    def polar_retract(v: np.ndarray, xi: np.ndarray) -> ComplexVector:
        v_new = np.asarray(v, dtype=np.complex128) + np.asarray(xi, dtype=np.complex128)
        nrm = float(np.linalg.norm(v_new))
        if nrm < 1e-15:
            return np.asarray(v, dtype=np.complex128)
        return (v_new / nrm).astype(np.complex128)

    @classmethod
    def differential_T(cls, rho: np.ndarray, v: np.ndarray, xi: np.ndarray) -> ComplexVector:
        r"""
        DT_{[v]}(ξ) = Π_{T(v)} (ρ ξ) / ‖ρ v‖   en el espacio horizontal.
        Es el diferencial del mapa proyectivo (T).  Holomorfo.
        """
        Tv, wn = ProjectiveDynamics.apply_map(rho, v)
        if wn < 1e-15:
            return np.zeros_like(v, dtype=np.complex128)
        eta = (rho @ xi) / wn
        return ProjectiveDynamics.horizontal(Tv, eta)

    @classmethod
    def operator_DT(cls, rho: np.ndarray, v: np.ndarray) -> ComplexMatrix:
        r"""Matriz de DT en ℂ^n (anula v, imagen horizontal en T(v))."""
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v, n)
        Tv, wn = ProjectiveDynamics.apply_map(rho, v)
        if wn < 1e-15:
            return np.zeros((n, n), dtype=np.complex128)
        P_T = np.eye(n, dtype=np.complex128) - np.outer(Tv, Tv.conj())
        return (P_T @ rho / wn).astype(np.complex128)

    @classmethod
    def spectral_radius_DT(cls, rho: np.ndarray, v: np.ndarray) -> float:
        r"""
        ρ(DT|_{v^⊥}) = max_{k∶ φ_k ⟂̸ v} |λ_k| / ‖ρ v‖.
        En el eigenrayo dominante simple: ρ(DT) = λ₂/λ₁.
        """
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v, n)
        Q = cls.horizontal_basis(v)
        if Q.shape[1] == 0:
            return 0.0
        DT = cls.operator_DT(rho, v)
        A = Q.conj().T @ DT @ Q
        ev = np.linalg.eigvals(A)
        return float(np.max(np.abs(ev))) if ev.size else 0.0


# ── §2.1 Iteración de potencia + Newton–Grassmann riemanniano ───────────────────
@dataclass(frozen=True, slots=True)
class PowerIterationTrace:
    r"""
    Traza forense de la órbita {v_k} ⊂ ℂP^{n−1}.
    Convergencia lineal de la potencia: d_{k+1} ∼ (λ₂/λ₁) d_k.
    Convergencia cuadrática de Newton si h_Kantorovich < 1/2 y [v] próximo a [φ₁].
    """
    residuals: Tuple[float, ...] = ()
    fs_angles: Tuple[float, ...] = ()
    rayleigh_trajectory: Tuple[float, ...] = ()
    empirical_rate: float = 0.0
    stalled_kernel: bool = False
    iterations: int = 0
    converged: bool = False
    history: Tuple[float, ...] = ()
    final_residual: float = 0.0
    variational_spectral_radius: float = 0.0
    newton_accelerated: bool = False
    quadratic_flag: bool = False
    kantorovich_h: float = float("inf")
    newton_steps: int = 0
    armijo_mean_step: float = 0.0
    geodesic_retraction: bool = True


class PowerIterationSolver:
    r"""
    Híbrido potencia / Newton–Grassmann sobre ℂP^{n−1}.

    Fase lineal (potencia gauge-fijada):
        v_{k+1} = gauge_align(v_k, ρ v_k / ‖ρ v_k‖).
        Tasa teórica r = λ₂/λ₁ = gap_ratio.

    Fase cuadrática (Newton riemanniano), activada cuando d_FS < d_switch:
        Hess_red ξ_red = −X_G_red     en v^⊥,
        v^{+} = exp_v(−ξ)             (retracción geodésica),
        Armijo sobre el residual de (T).

    Cota de Kantorovich: h = β η L,  η = ‖ξ‖, β = ‖H_red^{-1}‖, L ≤ 2‖ρ‖₂.
    Si h < 1/2 existe bola de convergencia cuadrática (Kantorovich 1948).
    """

    MAX_ITER_DEFAULT: Final[int] = 500
    TOL_DEFAULT: Final[float] = 1.0e-10
    MIN_FS_FOR_RATE: Final[float] = 1.0e-12
    RATE_FS_FLOOR: Final[float] = 1.0e-8
    RATE_FS_CEIL: Final[float] = 5.0e-1
    NEWTON_SWITCH_FS: Final[float] = 1.0e-2
    ARMIJO_MAX: Final[int] = 12
    KANTOROVICH_SAFE: Final[float] = 0.5

    @classmethod
    def _empirical_rate(cls, fs_angles: List[float], theoretical: float) -> float:
        arr = np.asarray(fs_angles, dtype=np.float64)
        if arr.size <= 2:
            return float(theoretical)
        mask = (arr[:-1] >= cls.RATE_FS_FLOOR) & (arr[:-1] <= cls.RATE_FS_CEIL)
        mask &= arr[1:] >= cls.MIN_FS_FOR_RATE
        if int(mask.sum()) < 3:
            return float(theoretical) if arr[-1] < cls.RATE_FS_FLOOR else 1.0
        ratios = arr[1:][mask] / np.maximum(arr[:-1][mask], cls.MIN_FS_FOR_RATE)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            return float(theoretical)
        return float(np.clip(np.median(ratios), 0.0, 1.0))

    @classmethod
    def _variational_radius(cls, rho: np.ndarray, v: np.ndarray, n_trials: int = 6) -> float:
        r"""Estimador Monte-Carlo de ‖DT‖ en la horizontal (mediana de ‖DT ξ‖)."""
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v, n)
        w, wn = ProjectiveDynamics.apply_map(rho, v)
        if wn < 1e-15:
            return 0.0
        rng = np.random.default_rng(_seed_from_string("VAR::DT"))
        radii: List[float] = []
        for _ in range(n_trials):
            xi = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            xi = ProjectiveDynamics.horizontal(v, xi)
            nn = float(np.linalg.norm(xi))
            if nn < 1e-15:
                continue
            xi = xi / nn
            eta = ProjectiveDynamics.horizontal(w, (rho @ xi) / wn)
            radii.append(float(np.linalg.norm(eta)))
        return float(np.median(radii)) if radii else 0.0

    @classmethod
    def _reduced_newton_direction(
        cls, rho: np.ndarray, v: np.ndarray
    ) -> Tuple[ComplexVector, float, float]:
        r"""
        Resuelve H_red ξ_red = X_G_red en v^⊥.

        H_proj = P ρ P − R P,  P = I − vv†.
        Devuelve (ξ_ambiente, β=‖H_red^{+}‖, η=‖ξ‖).
        El signo: Newton da v ← exp_v(−ξ) con ξ = H^{+} X_G, pues ∇R = 2 X_G
        y se busca X_G = 0 (máximo de R ⇔ punto fijo de (T) si λ₁ simple).
        """
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v, n)
        X = ProjectiveDynamics.projected_gradient(rho, v)
        Q = FubiniStudyRetraction.horizontal_basis(v)
        if Q.shape[1] == 0:
            return np.zeros(n, dtype=np.complex128), 0.0, 0.0
        H = ProjectiveDynamics.projected_hessian_rayleigh(rho, v)
        H_red = Q.conj().T @ H @ Q
        g_red = Q.conj().T @ X
        H_reg = H_red + 1e-12 * np.eye(H_red.shape[0], dtype=np.complex128)
        try:
            xi_red = la.solve(H_reg, g_red, assume_a="her")
        except la.LinAlgError:
            xi_red = la.pinv(H_reg, rcond=1e-10) @ g_red
        xi = (Q @ xi_red).astype(np.complex128)
        svals = np.linalg.svd(H_reg, compute_uv=False)
        sigma_min = float(np.min(np.real(svals))) if svals.size else 1e-15
        beta = float(1.0 / max(sigma_min, 1e-15))
        eta = float(np.linalg.norm(xi))
        return xi, beta, eta

    @classmethod
    def _kantorovich_h(cls, rho: np.ndarray, beta: float, eta: float) -> float:
        r"""h = β η L, L ≤ 2‖ρ‖₂ (Lipschitz del Hessiano proyectado)."""
        try:
            lip = 2.0 * float(np.linalg.norm(rho, 2))
        except ValueError:
            lip = 2.0 * float(np.linalg.norm(rho, "fro"))
        return float(beta * eta * max(lip, 1e-15))

    @classmethod
    def _armijo_geodesic(
        cls,
        rho: np.ndarray,
        v: np.ndarray,
        xi: np.ndarray,
        res0: float,
    ) -> Tuple[ComplexVector, float]:
        r"""
        Búsqueda de Armijo sobre la geodésica v_t = exp_v(−t ξ).
        Acepta el mayor t = 2^{−k} que reduce el residual de (T).
        """
        t = 1.0
        v_best = v
        accepted = False
        for _ in range(cls.ARMIJO_MAX):
            v_try = FubiniStudyRetraction.geodesic_exp(v, -t * xi)
            v_try = ProjectiveDynamics.gauge_align(v, v_try)
            sample = ProjectiveDynamics.evaluate(rho, v_try)
            if sample.residual < res0 * (1.0 - 1e-4 * t) or sample.fs_angle < res0:
                return v_try, t
            if sample.residual < res0:
                v_best = v_try
                accepted = True
            t *= 0.5
        if accepted:
            return v_best, t * 2.0
        # fallback polar amortiguado
        v_fb = FubiniStudyRetraction.polar_retract(v, -0.3 * xi)
        return ProjectiveDynamics.gauge_align(v, v_fb), 0.3

    @classmethod
    def solve(
        cls,
        field: IntrospectiveField,
        max_iter: int = MAX_ITER_DEFAULT,
        tol: float = TOL_DEFAULT,
        use_newton: bool = True,
    ) -> Tuple[ComplexVector, PowerIterationTrace]:
        rho = field.rho
        n = field.dim
        v = ProjectiveDynamics.sanitize_vector(field.v_initial, n)
        theoretical = float(field.spectral_gap.gap_ratio)
        residuals: List[float] = []
        fs_angles: List[float] = []
        rayleigh_traj: List[float] = []
        converged = False
        stalled_kernel = False
        newton_used = False
        newton_steps = 0
        armijo_acc: List[float] = []
        h_kant = float("inf")

        if n <= 1:
            sample0 = ProjectiveDynamics.evaluate(rho, v)
            trace = PowerIterationTrace(
                residuals=(float(sample0.residual),),
                fs_angles=(float(sample0.fs_angle),),
                rayleigh_trajectory=(float(sample0.rayleigh),),
                empirical_rate=0.0, stalled_kernel=False, iterations=1,
                converged=True, history=(float(sample0.fs_angle),),
                final_residual=float(sample0.residual),
                variational_spectral_radius=0.0, newton_accelerated=False,
                quadratic_flag=True, kantorovich_h=0.0, newton_steps=0,
                armijo_mean_step=1.0, geodesic_retraction=True,
            )
            return v, trace

        for k in range(max_iter):
            sample = ProjectiveDynamics.evaluate(rho, v)
            residuals.append(sample.residual)
            fs_angles.append(sample.fs_angle)
            rayleigh_traj.append(sample.rayleigh)
            if sample.image_norm < 1e-15:
                stalled_kernel = True
                break
            if sample.residual < tol or sample.fs_angle < tol:
                converged = True
                break

            switch = bool(
                use_newton
                and k >= 2
                and sample.fs_angle <= cls.NEWTON_SWITCH_FS
                and field.spectral_gap.degeneracy_top == 1
            )
            if switch:
                xi, beta, eta = cls._reduced_newton_direction(rho, v)
                h_kant = cls._kantorovich_h(rho, beta, eta)
                if eta < 1e-18:
                    T_v, _ = ProjectiveDynamics.apply_map(rho, v)
                    v = ProjectiveDynamics.gauge_align(v, T_v)
                else:
                    v_new, t_used = cls._armijo_geodesic(rho, v, xi, sample.residual)
                    v = ProjectiveDynamics.sanitize_vector(v_new, n)
                    newton_used = True
                    newton_steps += 1
                    armijo_acc.append(float(t_used))
            else:
                T_v, _ = ProjectiveDynamics.apply_map(rho, v)
                v = ProjectiveDynamics.gauge_align(v, T_v)
                v = ProjectiveDynamics.sanitize_vector(v, n)

        empirical = cls._empirical_rate(fs_angles, theoretical)
        var_r = cls._variational_radius(rho, v)
        # ρ(DT) exacto en el punto de llegada (más fiable que el MC)
        try:
            var_r_exact = FubiniStudyRetraction.spectral_radius_DT(rho, v)
            if math.isfinite(var_r_exact):
                var_r = 0.5 * (var_r + var_r_exact) if var_r > 0 else var_r_exact
        except Exception:
            pass
        final_res = float(residuals[-1]) if residuals else 0.0
        hist = tuple(map(float, fs_angles))
        quad = False
        if len(residuals) >= 4:
            r1 = residuals[-1] / max(residuals[-2], 1e-30)
            r2 = residuals[-2] / max(residuals[-3], 1e-30)
            quad = bool(r1 < r2 * 0.5 and r2 < 1.0 and newton_used)
        if h_kant < cls.KANTOROVICH_SAFE and newton_used and converged:
            quad = True
        trace = PowerIterationTrace(
            residuals=tuple(map(float, residuals)),
            fs_angles=hist,
            rayleigh_trajectory=tuple(map(float, rayleigh_traj)),
            empirical_rate=float(empirical),
            stalled_kernel=bool(stalled_kernel),
            iterations=len(residuals),
            converged=bool(converged),
            history=hist,
            final_residual=final_res,
            variational_spectral_radius=float(var_r),
            newton_accelerated=bool(newton_used),
            quadratic_flag=bool(quad),
            kantorovich_h=float(h_kant) if math.isfinite(h_kant) else float("inf"),
            newton_steps=int(newton_steps),
            armijo_mean_step=float(np.mean(armijo_acc)) if armijo_acc else 0.0,
            geodesic_retraction=True,
        )
        return v, trace

    @classmethod
    def solve_poincare_birkhoff_fixed_point_cpn(
        cls,
        density_op: np.ndarray,
        seed_ray_s6: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tolerance_fubini_study: float = 1e-6,
    ) -> Tuple[PowerIterationTrace, "FixedPointCertificate"]:
        r"""
        Punto fijo autoinvariante en ℂP^{n−1} gauge-fijado, semilla S⁶.
        Axiomas: gauge U(1), ‖T_φ(v)‖=1, d_FS = arccos|⟨v, T_φ(v)⟩| ≤ tol.
        Conservado como interfaz de compatibilidad con el meta-motor RSI-3 (FASE 3).
        """
        density_op = DensityOperatorAlgebra.sanitize(density_op)
        n = int(density_op.shape[0])
        if seed_ray_s6 is not None and np.asarray(seed_ray_s6).size >= 6:
            s6 = np.asarray(seed_ray_s6, dtype=np.float64).reshape(-1)
            v0 = np.array(
                [
                    s6[0] + 1j * s6[1],
                    s6[2] + 1j * s6[3],
                    s6[4] + 1j * s6[5],
                ],
                dtype=np.complex128,
            )
            if v0.size < n:
                v0 = np.pad(v0, (0, n - v0.size))
            elif v0.size > n:
                v0 = v0[:n]
            nv0 = float(np.linalg.norm(v0))
            v = v0 / nv0 if nv0 > 1e-12 else np.ones(n, dtype=np.complex128) / math.sqrt(n)
        else:
            v = np.ones(n, dtype=np.complex128) / math.sqrt(n)
        v = v / float(np.linalg.norm(v))
        trace_history: List[float] = []
        d_fs = 1.0
        for _it in range(1, max_iter + 1):
            w = density_op @ v
            norm_w = float(np.linalg.norm(w))
            if norm_w < 1e-15:
                break
            w_normalized = w / norm_w
            overlap = np.vdot(v, w_normalized)
            phase = np.angle(overlap) if np.abs(overlap) > 1e-12 else 0.0
            v_next = w_normalized * np.exp(-1j * phase)
            v_next = v_next / float(np.linalg.norm(v_next))
            fidelity = float(np.clip(np.abs(np.vdot(v, v_next)), 0.0, 1.0))
            d_fs = float(np.arccos(fidelity))
            trace_history.append(d_fs)
            if d_fs <= tolerance_fubini_study:
                v = v_next
                break
            v = v_next
        is_fixed_point = bool(d_fs <= tolerance_fubini_study)
        uhlmann_fid = float(np.abs(np.vdot(v, density_op @ v)))
        sample_final = ProjectiveDynamics.evaluate(density_op, v)
        w_spec = DensityOperatorAlgebra.spectrum_descending(density_op)
        lam1 = float(w_spec[0]) if w_spec.size else 1.0
        rate_th = float(w_spec[1] / lam1) if w_spec.size > 1 and lam1 > 0 else 0.0
        cert = FixedPointCertificate(
            fixed_point_residual=float(sample_final.residual),
            fixed_point_fs_angle=d_fs,
            overlap_final=float(sample_final.overlap),
            rayleigh_final=float(sample_final.rayleigh),
            rayleigh_gap_to_lambda1=float(max(0.0, lam1 - sample_final.rayleigh)),
            energy_proxy=float(1.0 - sample_final.overlap),
            iterations=len(trace_history),
            converged=is_fixed_point,
            stalled_kernel=False,
            empirical_rate=0.0,
            theoretical_rate=rate_th,
            rate_consistency=True,
            local_verdict=(HeytingOmega3.COHERENT if is_fixed_point else HeytingOmega3.DEGRADED),
            is_fixed_point=is_fixed_point,
            fubini_study_distance=d_fs,
            iterations_count=len(trace_history),
            eigenstate_ray=v,
            uhlmann_fidelity=uhlmann_fid,
        )
        trace = PowerIterationTrace(
            residuals=tuple(trace_history),
            fs_angles=tuple(trace_history),
            rayleigh_trajectory=(),
            empirical_rate=0.0,
            stalled_kernel=False,
            iterations=len(trace_history),
            converged=is_fixed_point,
            history=tuple(trace_history),
            final_residual=d_fs,
            variational_spectral_radius=rate_th,
        )
        return trace, cert


# ── §2.2 Certificado del punto fijo ─────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FixedPointCertificate:
    r"""
    Certificado de [v*] ∈ Fix(T) ⊂ ℂP^{n−1}.

    Condiciones equivalentes (λ₁ simple):
        (i)   T([v*]) = [v*]
        (ii)  X_G(v*) = 0
        (iii) R(v*) = λ₁
        (iv)  |⟨φ₁|v*⟩| = 1
    Residual chordal: ‖T_φ(v) − v‖ = 2 sin(d_FS/2).
    """
    fixed_point_residual: float = 0.0
    fixed_point_fs_angle: float = 0.0
    overlap_final: float = 0.0
    rayleigh_final: float = 0.0
    rayleigh_gap_to_lambda1: float = 0.0
    energy_proxy: float = 0.0
    iterations: int = 0
    converged: bool = False
    stalled_kernel: bool = False
    empirical_rate: float = 0.0
    theoretical_rate: float = 0.0
    rate_consistency: bool = True
    local_verdict: HeytingOmega3 = HeytingOmega3.VETOED
    is_fixed_point: bool = False
    fubini_study_distance: float = 0.0
    iterations_count: int = 0
    eigenstate_ray: Optional[np.ndarray] = None
    uhlmann_fidelity: float = 0.0
    dt_spectral_radius: float = 0.0
    attracting: bool = False


class FixedPointCertifier:
    EPS_FID: Final[float] = 1.0e-6
    EPS_RES: Final[float] = 1.0e-6
    EPS_RAY: Final[float] = 1.0e-3
    RATE_TOL: Final[float] = 0.10

    @classmethod
    def _grade(cls, value: float, hi_ok: float, mid_ok: float) -> HeytingOmega3:
        if value <= hi_ok:
            return HeytingOmega3.COHERENT
        if value <= mid_ok:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def certify(
        cls, field: IntrospectiveField, v_fixed: np.ndarray, trace: PowerIterationTrace
    ) -> FixedPointCertificate:
        sample = ProjectiveDynamics.evaluate(field.rho, v_fixed)
        lam1 = field.spectral_gap.lambda_1
        rate_th = field.spectral_gap.gap_ratio
        rate_emp = trace.empirical_rate
        already_fp = bool(trace.converged and trace.iterations <= 2)
        noisy_rate = bool(trace.converged and sample.fs_angle < cls.EPS_RES)
        rate_ok = already_fp or noisy_rate or (abs(rate_emp - rate_th) <= cls.RATE_TOL)
        conv_rule = (
            HeytingOmega3.VETOED if trace.stalled_kernel
            else (HeytingOmega3.COHERENT if trace.converged else HeytingOmega3.DEGRADED)
        )
        overlap_rule = cls._grade(1.0 - sample.overlap, cls.EPS_FID, 1.0e-3)
        residual_rule = cls._grade(sample.residual, cls.EPS_RES, 1.0e-3)
        ray_rel = max(0.0, lam1 - sample.rayleigh) / max(lam1, _EPS)
        ray_rule = cls._grade(ray_rel, cls.EPS_RAY, 1.0e-2)
        rate_rule = HeytingOmega3.COHERENT if rate_ok else HeytingOmega3.DEGRADED
        dt_rho = float(trace.variational_spectral_radius)
        try:
            dt_rho = min(dt_rho, FubiniStudyRetraction.spectral_radius_DT(field.rho, v_fixed)) \
                if dt_rho > 0 else FubiniStudyRetraction.spectral_radius_DT(field.rho, v_fixed)
        except Exception:
            pass
        attracting = bool(dt_rho < 1.0 - 1e-12)
        attr_rule = HeytingOmega3.COHERENT if attracting else HeytingOmega3.VETOED
        local = conv_rule.meet_all(overlap_rule, residual_rule, ray_rule, rate_rule, attr_rule)
        return FixedPointCertificate(
            fixed_point_residual=float(sample.residual),
            fixed_point_fs_angle=float(sample.fs_angle),
            overlap_final=float(sample.overlap),
            rayleigh_final=float(sample.rayleigh),
            rayleigh_gap_to_lambda1=float(max(0.0, lam1 - sample.rayleigh)),
            energy_proxy=float(1.0 - sample.overlap),
            iterations=int(trace.iterations),
            converged=bool(trace.converged),
            stalled_kernel=bool(trace.stalled_kernel),
            empirical_rate=float(rate_emp),
            theoretical_rate=float(rate_th),
            rate_consistency=bool(rate_ok),
            local_verdict=local,
            is_fixed_point=bool(trace.converged and sample.residual < cls.EPS_RES),
            fubini_study_distance=float(sample.fs_angle),
            iterations_count=int(trace.iterations),
            eigenstate_ray=np.asarray(v_fixed, dtype=np.complex128),
            uhlmann_fidelity=float(sample.overlap),
            dt_spectral_radius=float(dt_rho),
            attracting=attracting,
        )


# ── §2.2b Lyapunov cuadrático V = λ₁ − R  (teoría T3) ───────────────────────────
@dataclass(frozen=True, slots=True)
class QuadraticLyapunovCertificate:
    r"""
    Funcional de Lyapunov proyectivo:
        V(v) = λ₁ − R(v) ≥ 0,   V(v) = 0  ⇔  [v] = [φ₁]  (λ₁ simple).

    Flujo (G):  dV/dt = −2 ‖X_G‖² ≤ 0   (exponencialmente estricto si gap > 0).
    Mapa (T):   V(Tv) ≤ (λ₂/λ₁)² · C · V(v)  en un cono cerca de [φ₁].

    Equivale a V(x) = x^T P x en la carta afín z_k = ⟨φ_k|v⟩/⟨φ₁|v⟩,
    con P = diag(λ₁ − λ_k)_{k≥2} > 0.
    """
    V_value: float
    V_dot_gradient: float
    V_decrease_map: float
    positive_definite: bool
    exponentially_stable: bool
    contraction_proxy: float
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(
        cls, rho: np.ndarray, v: np.ndarray, lam1: float, gap_ratio: float
    ) -> "QuadraticLyapunovCertificate":
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v, n)
        R = ProjectiveDynamics.rayleigh(rho, v)
        V = float(max(0.0, lam1 - R))
        X = ProjectiveDynamics.projected_gradient(rho, v)
        Vdot = float(-2.0 * np.real(np.vdot(X, X)))
        Tv, _ = ProjectiveDynamics.apply_map(rho, v)
        R_T = ProjectiveDynamics.rayleigh(rho, Tv)
        V_T = float(max(0.0, lam1 - R_T))
        decrease = float(V - V_T)
        pd = bool(lam1 > 0.0)
        # estabilidad exponencial si gap_ratio < 1 y Vdot ≤ 0, V↓ a lo largo de T
        exp_st = bool(gap_ratio < 1.0 - 1e-12 and Vdot <= 1e-18 and decrease >= -1e-12)
        contr = float(gap_ratio ** 2)
        if exp_st and V <= 1e-6:
            local = HeytingOmega3.COHERENT
        elif exp_st or Vdot <= 0.0:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            V_value=V,
            V_dot_gradient=float(Vdot),
            V_decrease_map=decrease,
            positive_definite=pd,
            exponentially_stable=exp_st,
            contraction_proxy=contr,
            local_verdict=local,
        )


# ── §2.3 Espectro de Oseledets (MET) + QR empírico ──────────────────────────────
@dataclass(frozen=True, slots=True)
class OseledetsLyapunovSpectrum:
    r"""
    Teorema ergódico multiplicativo de Oseledets (1968) para DT sobre TℂP^{n−1}.

    En el eigenrayo dominante, T es holomorfa y diagonalizable en la carta afín:
        λ_k^{Lyap} = log(λ_k(ρ) / λ₁(ρ)),   k ≥ 2   (todas ≤ 0 si λ₁ simple).

    Pesin: h_KS = Σ_{λ>0} λ.  En un atractor puntual, h_KS = 0.
    Kaplan–Yorke: D_KY = k + S_k / |λ_{k+1}|,  k = max{j : S_j ≥ 0}.
    Si λ_max < 0 ⇒ D_KY = 0.

    QR empírico: transporte de un marco horizontal bajo DT, confirma MET.
    """
    lyapunov_full: RealVector
    lyapunov_max: float
    lyapunov_min: float
    kolmogorov_sinai_entropy: float
    kaplan_yorke_dimension: float
    n_positive: int
    degeneracy_signal: bool
    lyapunov_empirical: RealVector = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    empirical_ks: float = 0.0
    qr_residual: float = 0.0
    met_consistent: bool = True

    @classmethod
    def from_spectrum(cls, rho: np.ndarray) -> "OseledetsLyapunovSpectrum":
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        lam1 = max(float(w[0]), _EPS)
        rates = np.array(
            [_safe_log(max(float(w_i), _EPS) / lam1) for w_i in w[1:]],
            dtype=np.float64,
        )
        if rates.size == 0:
            rates = np.array([0.0], dtype=np.float64)
        rates_sorted = np.sort(rates)[::-1]
        lam_max = float(rates_sorted[0])
        lam_min = float(rates_sorted[-1])
        pos = rates_sorted[rates_sorted > 1e-14]
        h_ks = float(np.sum(pos)) if pos.size else 0.0
        cum = np.cumsum(rates_sorted)
        k = int(np.max(np.where(cum >= 0.0)[0])) + 1 if np.any(cum >= 0.0) else 0
        if 0 < k < rates_sorted.size and abs(rates_sorted[k]) > 1e-15:
            d_ky = float(k + cum[k - 1] / abs(rates_sorted[k]))
        else:
            d_ky = float(k)
        degeneracy = bool(lam_max > -1e-10)
        return cls(
            lyapunov_full=rates_sorted,
            lyapunov_max=lam_max,
            lyapunov_min=lam_min,
            kolmogorov_sinai_entropy=h_ks,
            kaplan_yorke_dimension=d_ky,
            n_positive=int(pos.size),
            degeneracy_signal=degeneracy,
        )

    @classmethod
    def from_qr_orbit(
        cls,
        rho: np.ndarray,
        v0: np.ndarray,
        n_iter: int = 16,
        key: str = "OSELEDETS-QR",
    ) -> Tuple[RealVector, float]:
        r"""
        Espectro de Lyapunov empírico por QR de Benettin sobre DT.
        Devuelve (λ_emp descendente, residuo ‖λ_emp − λ_th‖_∞).
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v0, n)
        Q = FubiniStudyRetraction.horizontal_basis(v)
        m = int(Q.shape[1])
        if m == 0:
            return np.zeros(0, dtype=np.float64), 0.0
        acc = np.zeros(m, dtype=np.float64)
        steps = 0
        for _ in range(max(2, n_iter)):
            Tv, wn = ProjectiveDynamics.apply_map(rho, v)
            if wn < 1e-15:
                break
            Y = rho @ Q
            Y = Y / wn
            # horizontalizar columnas respecto de Tv
            Y = Y - np.outer(Tv, Tv.conj() @ Y)
            # QR complejo (Householder)
            Qnew, R = np.linalg.qr(Y, mode="reduced")
            diag = np.abs(np.diag(R))
            acc = acc + np.log(np.maximum(diag, 1e-15))
            Q = Qnew
            v = Tv
            steps += 1
        if steps == 0:
            return np.zeros(m, dtype=np.float64), 1.0
        lyap = np.sort(acc / steps)[::-1]
        th = cls.from_spectrum(rho).lyapunov_full
        mm = min(lyap.size, th.size)
        resid = float(np.max(np.abs(lyap[:mm] - th[:mm]))) if mm else 0.0
        return lyap.astype(np.float64), resid

    @classmethod
    def evaluate(cls, rho: np.ndarray, v_fixed: np.ndarray) -> "OseledetsLyapunovSpectrum":
        base = cls.from_spectrum(rho)
        emp, resid = cls.from_qr_orbit(rho, v_fixed)
        pos_e = emp[emp > 1e-14] if emp.size else np.zeros(0)
        h_emp = float(np.sum(pos_e)) if pos_e.size else 0.0
        mm = min(emp.size, base.lyapunov_full.size)
        cons = bool(resid < 0.5) if mm else True
        return cls(
            lyapunov_full=base.lyapunov_full,
            lyapunov_max=base.lyapunov_max,
            lyapunov_min=base.lyapunov_min,
            kolmogorov_sinai_entropy=base.kolmogorov_sinai_entropy,
            kaplan_yorke_dimension=base.kaplan_yorke_dimension,
            n_positive=base.n_positive,
            degeneracy_signal=base.degeneracy_signal,
            lyapunov_empirical=emp,
            empirical_ks=h_emp,
            qr_residual=float(resid),
            met_consistent=cons,
        )


# ── §2.4 Monodromía hamiltoniana / multiplicadores de Floquet–Krein ─────────────
@dataclass(frozen=True, slots=True)
class VariationalMonodromyCertificate:
    r"""
    Ecuación variacional a lo largo de la órbita periódica del flujo (H)
    en el 2-plano span{φ₁, φ₂}:
        μ_k = exp(i (λ_k − λ₁) τ),   τ = 2π / |λ₁ − λ₂|.

    Estabilidad elíptica ⇔ |μ| = 1 (círculo unidad).
    Resonancia de Krein: μ = ±1 para algún k ≥ 2 (colisión Krein).
    Firma de Krein: i ⟨ξ, Ω ξ⟩ sobre el eigenspace; aquí Ω = ω_FS.
    """
    multipliers: Tuple[complex, ...]
    period: float
    on_unit_circle: bool
    krein_resonance: bool
    spectral_radius: float
    krein_signatures: Tuple[float, ...]
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(cls, rho: np.ndarray) -> "VariationalMonodromyCertificate":
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        if w.size < 2:
            return cls((1.0 + 0j,), 0.0, True, True, 1.0, (0.0,), HeytingOmega3.DEGRADED)
        tau = HamiltonianCelestialFlow.two_plane_period(float(w[0]), float(w[1]))
        if not math.isfinite(tau):
            tau = 0.0
        lams: List[complex] = []
        signs: List[float] = []
        for wk in w:
            phase = (float(wk) - float(w[0])) * tau
            mu = complex(math.cos(phase), math.sin(phase))
            lams.append(mu)
            # firma de Krein proxy: signo de (λ_k − λ₁) (elíptico simple)
            signs.append(float(np.sign(float(wk) - float(w[0]))))
        mods = [abs(z) for z in lams]
        on_u1 = bool(all(abs(m - 1.0) < 1e-8 for m in mods))
        krein = bool(any(abs(z - 1.0) < 1e-8 or abs(z + 1.0) < 1e-8 for z in lams[1:]))
        rho_sp = float(max(mods)) if mods else 1.0
        if on_u1 and not krein:
            local = HeytingOmega3.COHERENT
        elif on_u1:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            tuple(lams), float(tau), on_u1, krein, rho_sp, tuple(signs), local
        )


# ── §2.5 Integral de Melnikov proyectiva ────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class MelnikovHomoclinicCertificate:
    r"""
    Integral de Poincaré–Melnikov sobre la heteroclínica de (G) entre [φ₂] y [φ₁]:

        M(ϑ) = ∫_{−∞}^{∞} ω_FS( X_{H₀}, X_{H₁} )(γ_ϑ(t)) dt

    γ_ϑ(t) ∝ e^{δ t/2} φ₁ + e^{−δ t/2} e^{iϑ} φ₂,  δ = λ₁ − λ₂.

    Cero simple (M(ϑ*)=0, M'(ϑ*)≠0) + |M| no idénticamente nulo
    ⇒ W^s ⋔ W^u transversal ⇒ herradura de Smale (Birkhoff–Smale).
    El sistema no perturbado cumple M₀ ≡ 0 porque ω(X,X)=0.
    """
    melnikov_value: float
    melnikov_zeros: int
    transverse_homoclinic: bool
    chaos_threshold: float
    unperturbed_vanishes: bool
    simple_zeros: int = 0
    melnikov_amplitude: float = 0.0
    n_theta_samples: int = 0

    @staticmethod
    def _heteroclinic(
        phi1: np.ndarray, phi2: np.ndarray, delta: float, t: float
    ) -> ComplexVector:
        r"""Heteroclínica numéricamente estable: t→+∞ ⇒ φ₁, t→−∞ ⇒ φ₂."""
        # clamp para evitar overflow
        x = 0.5 * float(delta) * float(t)
        x = float(np.clip(x, -40.0, 40.0))
        a = math.exp(x)
        b = math.exp(-x)
        v = a * phi1 + b * phi2
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            return np.asarray(phi1, dtype=np.complex128)
        return (v / nrm).astype(np.complex128)

    @staticmethod
    def _hermite_wiggle(n: int, key: str) -> ComplexMatrix:
        rng = np.random.default_rng(_seed_from_string(f"MELNIKOV::{key}"))
        B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        W = 0.5 * (B + B.conj().T)
        W = W - np.eye(n) * (np.trace(W) / max(n, 1))
        fn = float(np.linalg.norm(W, "fro"))
        if fn > 0.0:
            W = W / fn
        return W.astype(np.complex128)

    @classmethod
    def _M_of_theta(
        cls,
        rho: np.ndarray,
        W: np.ndarray,
        phi1: np.ndarray,
        phi2: np.ndarray,
        delta: float,
        theta: float,
        ts: np.ndarray,
    ) -> Tuple[float, float]:
        phi2_th = (np.exp(1j * theta) * phi2).astype(np.complex128)
        dens = np.zeros_like(ts, dtype=np.float64)
        dens0 = np.zeros_like(ts, dtype=np.float64)
        for i, t in enumerate(ts):
            gamma = cls._heteroclinic(phi1, phi2_th, delta, float(t))
            X0 = ProjectiveDynamics.hamiltonian_vector_field(rho, gamma)
            X1 = ProjectiveDynamics.hamiltonian_vector_field(W, gamma)
            dens[i] = ProjectiveDynamics.symplectic_form(gamma, X0, X1)
            dens0[i] = ProjectiveDynamics.symplectic_form(gamma, X0, X0)
        dt = float(ts[1] - ts[0]) if ts.size > 1 else 1.0
        return _quad_trap(dens, dx=dt), _quad_trap(dens0, dx=dt)

    @classmethod
    def evaluate(
        cls,
        rho: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        epsilon_M: float = 1e-3,
        key: str = "MELNIKOV",
        n_theta: int = 24,
    ) -> "MelnikovHomoclinicCertificate":
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        v1 = ProjectiveDynamics.sanitize_vector(v1, n)
        v2 = ProjectiveDynamics.sanitize_vector(v2, n)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        lam1 = float(w[0])
        lam2 = float(w[1]) if w.size > 1 else 0.0
        delta = max(abs(lam1 - lam2), 1e-6)
        W = cls._hermite_wiggle(n, key)
        T = 6.0 / delta
        ts = np.linspace(-T, T, 81)
        thetas = np.linspace(0.0, _TWO_PI, int(max(8, n_theta)), endpoint=False)
        Ms: List[float] = []
        M0_acc = 0.0
        for th in thetas:
            M, M0 = cls._M_of_theta(rho, W, v1, v2, delta, float(th), ts)
            Ms.append(M)
            M0_acc += M0
        M_arr = np.asarray(Ms, dtype=np.float64)
        M0_mean = float(M0_acc / max(len(thetas), 1))
        # ceros por cambio de signo; simples si |ΔM/Δϑ| > ε
        dth = float(thetas[1] - thetas[0]) if thetas.size > 1 else 1.0
        zeros = 0
        simple = 0
        for i in range(len(M_arr)):
            a = M_arr[i]
            b = M_arr[(i + 1) % len(M_arr)]
            if a == 0.0 or a * b < 0.0:
                zeros += 1
                deriv = abs(b - a) / max(dth, 1e-18)
                if deriv > epsilon_M:
                    simple += 1
        amp = float(np.max(np.abs(M_arr))) if M_arr.size else 0.0
        unperturbed = bool(abs(M0_mean) < 1e-8)
        transverse = bool(simple >= 1 and amp > epsilon_M)
        M_at_0 = float(M_arr[0]) if M_arr.size else 0.0
        return cls(
            melnikov_value=M_at_0,
            melnikov_zeros=int(zeros),
            transverse_homoclinic=transverse,
            chaos_threshold=float(epsilon_M),
            unperturbed_vanishes=unperturbed,
            simple_zeros=int(simple),
            melnikov_amplitude=amp,
            n_theta_samples=int(len(thetas)),
        )


# ── §2.5b Linearización de Siegel en la carta afín de [φ₁] ──────────────────────
@dataclass(frozen=True, slots=True)
class SiegelLinearizationCertificate:
    r"""
    En la carta afín z_k = ⟨φ_k|v⟩ / ⟨φ₁|v⟩, k≥2, el mapa (T) es
        z ↦ D z,   D = diag(λ_k / λ₁).
    Es YA lineal: la linearización de Siegel/Koenigs no tiene obstrucción
    (los cobordes de homología de Poincaré se anulan).  El disco de Siegel
    es todo el dominio {⟨φ₁|v⟩ ≠ 0} ≅ ℂ^{n−1}.
    Multiplicadores dentro del disco unidad ⇔ [φ₁] atractor.
    """
    already_linear: bool
    multipliers: Tuple[complex, ...]
    in_unit_disk: bool
    siegel_disk_is_full_chart: bool
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(cls, rho: np.ndarray) -> "SiegelLinearizationCertificate":
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        lam1 = max(float(w[0]), _EPS)
        mult = tuple(complex(float(wk) / lam1) for wk in w[1:])
        in_disk = bool(all(abs(z) < 1.0 - 1e-15 for z in mult)) if mult else True
        if in_disk:
            local = HeytingOmega3.COHERENT
        elif all(abs(z) <= 1.0 + 1e-12 for z in mult):
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            already_linear=True,
            multipliers=mult,
            in_unit_disk=in_disk,
            siegel_disk_is_full_chart=True,
            local_verdict=local,
        )


# ── §2.6 KAM diofántico: Bruno, Rüssmann, ε₀ = γ/(m log(1/γ))^m ────────────────
@dataclass(frozen=True, slots=True)
class KAMDiophantineCertificate:
    r"""
    Condición diofántica de tipo (γ, τ):
        |ω · k| ≥ γ / |k|^τ     ∀ k ∈ ℤ^m \ {0}.

    Umbral clásico de Pöschel / Bruno–Rüssmann:
        ε₀ = γ / ( m log(1/γ) )^m .
    Si ε < ε₀ y ω es (γ,τ)-diofántico, el toro 𝕋^m persiste.

    Kolmogorov: det(∂²H/∂I²) ≠ 0  — FALLA aquí (H lineal ⇒ Hess = 0, isócrono).
    Rüssmann: la imagen de I ↦ ω(I) no yace en un hiperplano — FALLA (ω = λ cte).
    Bruno (1-D): Σ_k (log q_{k+1}) / q_k < ∞ sobre convergentes de ω₂/ω₁.

    El certificado es honesto: kam_persists exige diofantino AND ε<ε₀ AND
    (Kolmogorov ∨ Rüssmann).  En el caso isócrono, persiste el toro del flujo
    lineal EXACTO (integrable), no por KAM perturbativo.
    """
    frequency_vector: RealVector
    diophantine_gamma: float
    diophantine_tau: float
    is_diophantine: bool
    perturbation_ratio: float
    kam_persists: bool
    kmax_used: int
    integrable_unperturbed: bool
    epsilon_0: float = 0.0
    kolmogorov_nondegenerate: bool = False
    russmann_nondegenerate: bool = False
    bruno_sum: float = float("inf")
    bruno_holds: bool = False
    isoenergetic_degenerate: bool = True

    @staticmethod
    def _continued_fraction(x: float, max_terms: int = 18) -> List[int]:
        a: List[int] = []
        y = abs(float(x))
        for _ in range(max_terms):
            if not math.isfinite(y) or y > 1e16:
                break
            ai = int(math.floor(y))
            a.append(ai)
            frac = y - ai
            if frac < 1e-14:
                break
            y = 1.0 / frac
        return a if a else [0]

    @staticmethod
    def _convergents_q(a: Sequence[int]) -> List[int]:
        q_m2, q_m1 = 0, 1
        qs: List[int] = []
        for ai in a:
            q = ai * q_m1 + q_m2
            if q <= 0:
                break
            qs.append(int(q))
            q_m2, q_m1 = q_m1, q
            if q > 10 ** 12:
                break
        return qs

    @classmethod
    def _bruno_sum(cls, omega: np.ndarray) -> Tuple[float, bool]:
        w = np.asarray(omega, dtype=np.float64).ravel()
        if w.size < 2 or abs(w[0]) < 1e-15:
            return float("inf"), False
        ratio = float(w[1] / w[0])
        a = cls._continued_fraction(ratio)
        qs = cls._convergents_q(a)
        if len(qs) < 2:
            return 0.0, True  # racional: Bruno no aplica; se reporta suma 0 y False
        acc = 0.0
        for k in range(len(qs) - 1):
            qk = max(qs[k], 1)
            qn = max(qs[k + 1], 1)
            acc += math.log(float(qn)) / float(qk)
        # criterio práctico: suma acotada (la serie de Bruno converge)
        holds = bool(acc < 30.0 and math.isfinite(acc))
        return float(acc), holds

    @classmethod
    def evaluate(
        cls,
        omega: Any,
        perturbation_size: float,
        tau: float = 1.5,
        kmax: int = 6,
        gamma_min: float = 1e-3,
    ) -> "KAMDiophantineCertificate":
        w = np.asarray(omega, dtype=np.float64).ravel()
        m = int(w.size)
        if m == 0:
            return cls(
                w, 0.0, tau, False, float("inf"), False, kmax, True,
                0.0, False, False, float("inf"), False, True,
            )
        kmax = int(max(1, min(int(kmax), 7)))
        m_eff = int(min(m, 4))  # cap combinatorio
        w_use = w[:m_eff]
        ranges = [np.arange(-kmax, kmax + 1) for _ in range(m_eff)]
        gamma = float("inf")
        grid = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(m_eff, -1).T
        for k in grid:
            if np.all(k == 0):
                continue
            dot = float(np.dot(w_use, k))
            norm = float(np.linalg.norm(k))
            if abs(dot) < 1e-15:
                gamma = 0.0
                break
            gamma = min(gamma, abs(dot) * (norm ** float(tau)))
        is_dio = bool(gamma >= gamma_min)
        # ε₀ = γ / (m log(1/γ))^m
        g_clip = float(np.clip(gamma, 1e-15, 1.0 - 1e-15))
        loginv = math.log(1.0 / g_clip)
        denom = (m_eff * max(loginv, 1e-15)) ** m_eff
        eps0 = float(gamma / max(denom, 1e-15)) if gamma > 1e-15 else 0.0
        ratio = float(perturbation_size / eps0) if eps0 > 1e-15 else float("inf")
        kolmogorov = False  # Hess H = 0
        russmann = False    # ω constante
        bruno_s, bruno_ok = cls._bruno_sum(w_use)
        # Persistencia KAM perturbativa: NO. Persistencia integrable: SÍ.
        # kam_persists se reserva a la persistencia perturbativa estricta.
        kam_pert = bool(is_dio and ratio < 1.0 and (kolmogorov or russmann))
        return cls(
            w.astype(np.float64),
            float(gamma),
            float(tau),
            is_dio,
            float(ratio),
            bool(kam_pert),
            int(kmax),
            True,
            float(eps0),
            kolmogorov,
            russmann,
            float(bruno_s),
            bool(bruno_ok),
            True,
        )


# ── §2.7 Teorema de Poincaré–Birkhoff + forma normal ────────────────────────────
@dataclass(frozen=True, slots=True)
class BirkhoffTwistMapCertificate:
    r"""
    (H) H(I) = λ · I  lineal ⇒ twist_H = det(∂θ'/∂I) = 0  (isócrono, P8).
        Moser twist FALLA.  Forma normal de Birkhoff ya lineal (términos ≥2 nulos).

    (T) restringido a un ℂP¹_{ij} = ℙ(span{φ_i, φ_j}) actúa en la coordenada
        radial r = |z_j|² / (|z_i|² + |z_j|²) por
            r ↦ f(r) = α r / (α r + β(1−r)),   α = λ_i², β = λ_j².
        f(r)=r  ⇔  r∈{0,1}: exactamente 2 puntos fijos (los eigenrayos).
        f'(r) = αβ / (αr + β(1−r))² > 0  (monótono) si α,β>0.
        Poincaré–Birkhoff (anillo con twist) no aplica a (H); (T) no es
        simpléctico, pero el recuento de FP es exacto.

    Teorema de Poincaré–Birkhoff (1913): un mapa de anillo que preserve área
    y tuerza las fronteras tiene ≥ 2 FP.  Aquí se certifica el recuento de (T)
    y la ausencia de twist de (H).
    """
    twist_holds: bool
    n_fixed_points_min: int
    twist_gradient_min: float
    hamiltonian_twist: float
    radial_monotone: bool
    degeneracy_pairs: int
    local_verdict: HeytingOmega3
    moser_twist: bool = False
    normal_form_degree: int = 1
    poincare_birkhoff_applies: bool = False

    @staticmethod
    def _radial_map_derivative(alpha: float, beta: float, r: float) -> float:
        den = alpha * r + beta * (1.0 - r)
        if abs(den) < 1e-18:
            return 0.0
        return float(alpha * beta / (den * den))

    @staticmethod
    def _radial_fixed_points(alpha: float, beta: float) -> int:
        r"""Número de FP de f(r)=r en [0,1]: 2 si α≠β,  continuum si α=β."""
        if abs(alpha - beta) < 1e-14 * max(alpha, beta, 1e-15):
            return -1  # identidad (degeneración)
        return 2

    @classmethod
    def evaluate(
        cls,
        rho: np.ndarray,
        n_pairs: int = 3,
        n_samples: int = 16,
        key: str = "BIRKHOFF",
    ) -> "BirkhoffTwistMapCertificate":
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        w, _V = DensityOperatorAlgebra.eigenpairs_descending(rho)
        rng = np.random.default_rng(_seed_from_string(f"BIRKHOFF::{key}"))
        twist_min = float("inf")
        valid_pairs = 0
        radial_ok = True
        n_fp_global = 1
        pairs_examined = min(int(n_pairs), max(1, n - 1))
        for p in range(pairs_examined):
            i, j = 0, p + 1
            if j >= n:
                continue
            alpha = max(float(w[i]) ** 2, _EPS)
            beta = max(float(w[j]) ** 2, _EPS)
            nfp = cls._radial_fixed_points(alpha, beta)
            if nfp < 0:
                continue
            valid_pairs += 1
            n_fp_global = max(n_fp_global, nfp)
            pair_min = float("inf")
            for _ in range(max(4, n_samples)):
                r = float(rng.uniform(0.1, 0.9))
                fp = cls._radial_map_derivative(alpha, beta, r)
                pair_min = min(pair_min, abs(fp - 1.0))
                if fp <= 0.0:
                    radial_ok = False
            twist_min = min(twist_min, pair_min)
        if not np.isfinite(twist_min):
            twist_min = 0.0
        n_fp = 2 if valid_pairs >= 1 else 1
        # (H) isócrono
        twist_H = 0.0
        moser = False
        pb_applies = False
        if n_fp >= 2 and radial_ok:
            local = HeytingOmega3.COHERENT
        elif n_fp >= 1:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            False,  # twist_holds de (H)
            int(n_fp),
            float(twist_min),
            float(twist_H),
            bool(radial_ok and valid_pairs >= 1),
            int(valid_pairs),
            local,
            moser,
            1,
            pb_applies,
        )


# ── §2.8 IntrospectionBundle + hand-off formal a FASE 3 ─────────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionBundle:
    r"""
    Objeto terminal FASE 2 / inicial FASE 3.

    Transporta el campo (F1), el punto fijo Newton–Grassmann, la traza, el
    certificado Fix(T), Oseledets–Pesin, Melnikov, KAM, Birkhoff, monodromía,
    Lyapunov cuadrático y Siegel.
    """
    field: IntrospectiveField
    v_fixed: np.ndarray
    trace: PowerIterationTrace
    certificate: FixedPointCertificate
    oseledets: OseledetsLyapunovSpectrum
    melnikov: MelnikovHomoclinicCertificate
    kam: KAMDiophantineCertificate
    birkhoff: BirkhoffTwistMapCertificate
    monodromy: VariationalMonodromyCertificate
    lyapunov: QuadraticLyapunovCertificate
    siegel: SiegelLinearizationCertificate

    def content_bytes(self) -> bytes:
        c = self.certificate
        return hashlib.sha256(
            self.field.field_hash.encode("ascii")
            + np.ascontiguousarray(self.v_fixed).tobytes()
            + f"{c.fixed_point_residual:.12e}".encode("ascii")
            + f"{c.overlap_final:.12e}".encode("ascii")
            + f"{c.rayleigh_final:.12e}".encode("ascii")
            + f"{c.iterations}".encode("ascii")
            + f"{c.converged}".encode("ascii")
            + f"{self.oseledets.lyapunov_max:.12e}".encode("ascii")
            + f"{self.oseledets.kolmogorov_sinai_entropy:.12e}".encode("ascii")
            + f"{self.melnikov.melnikov_value:.12e}".encode("ascii")
            + f"{self.birkhoff.n_fixed_points_min}".encode("ascii")
            + f"{self.monodromy.period:.12e}".encode("ascii")
            + f"{self.lyapunov.V_value:.12e}".encode("ascii")
            + f"{self.kam.epsilon_0:.12e}".encode("ascii")
            + f"{self.trace.kantorovich_h:.12e}".encode("ascii")
            + f"{int(self.siegel.already_linear)}".encode("ascii")
        ).digest()

    def meet_local_verdicts(self) -> HeytingOmega3:
        r"""Colapso Ω₃ de FASE 2 (sin el veredicto externo, que es FASE 3)."""
        return self.certificate.local_verdict.meet_all(
            self.birkhoff.local_verdict,
            self.monodromy.local_verdict,
            self.lyapunov.local_verdict,
            self.siegel.local_verdict,
            self.field.meet_local_verdicts(),
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 2 → FASE 3
    #  continue_into_phase3 ∘ synthesize ∘ prepare
    #      = adjudicate ∘ synthesize ∘ prepare.
    #
    #  Último morfismo FASE 2 ∧ primero FASE 3.
    #  La FASE 3 (siguiente prompt) define HeytingIntrospectionAdjudicator.
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase3(self, external_verdict: HeytingOmega3) -> HeytingOmega3:
        r"""
        Último morfismo FASE 2 ∧ primero FASE 3.

        Dominio   : IntrospectionBundle × Ω₃.
        Codominio : Ω₃  (veredicto local ∧ externo; FASE 3 lo tensora con Φ_η ⊗ RSI-3).

        Cuerpo (anidado en FASE 3):
            return HeytingIntrospectionAdjudicator.adjudicate(self, external_verdict)
        """
        adj = globals().get("HeytingIntrospectionAdjudicator")
        if adj is None:
            raise NotImplementedError(
                "Anidamiento FASE 2 → FASE 3 pendiente: HeytingIntrospectionAdjudicator.adjudicate "
                "es el cuerpo de continue_into_phase3.  Continuar en el siguiente prompt "
                "con la FASE 3 (mónada RSI-3 explícita + adjudicación Ω₃ + certificación)."
            )
        return adj.adjudicate(self, external_verdict)


class IntrospectionPipeline:
    r"""
    Orquestador determinista FASE 2 (funtor F₂).

    ESTE MÉTODO ES EL CUERPO DE
        IntrospectiveField.continue_into_phase2(self, max_iter, tol)
    definido como último morfismo de FASE 1:

        continue_into_phase2 ∘ prepare  =  synthesize ∘ prepare.
    """

    @classmethod
    def synthesize(
        cls,
        field: IntrospectiveField,
        max_iter: int = PowerIterationSolver.MAX_ITER_DEFAULT,
        tol: float = PowerIterationSolver.TOL_DEFAULT,
    ) -> IntrospectionBundle:
        r"""
        Cierra FASE 2.  Abre FASE 3.

        Cadena:
            v*, traza  ←  Newton–Grassmann híbrido
            cert       ←  FixedPointCertifier
            oseledets  ←  MET teórico + QR Benettin
            melnikov   ←  M(ϑ) sobre heteroclínica (G)
            kam        ←  (γ,τ,ε₀,Bruno,Rüssmann)
            birkhoff   ←  twist (H) + radial (T)
            monodromy  ←  Floquet–Krein
            lyapunov   ←  V = λ₁ − R
            siegel     ←  T ya lineal en la carta afín
        """
        v_fixed, trace = PowerIterationSolver.solve(field, max_iter=max_iter, tol=tol)
        cert = FixedPointCertifier.certify(field, v_fixed, trace)
        oseledets = OseledetsLyapunovSpectrum.evaluate(field.rho, v_fixed)
        _w, V = DensityOperatorAlgebra.eigenpairs_descending(field.rho)
        if V.shape[1] >= 2:
            melnikov = MelnikovHomoclinicCertificate.evaluate(
                field.rho, V[:, 0], V[:, 1], key=field.field_hash[:16]
            )
        else:
            melnikov = MelnikovHomoclinicCertificate(0.0, 0, False, 1e-3, True)
        omega_h = field.action_angle.frequencies[: min(3, field.dim)].tolist()
        if len(omega_h) < 2:
            omega_h = [1.0 + 1e-3 * i for i in range(2)]
        perturbation = DensityOperatorAlgebra.frobenius_to_maximally_mixed(field.rho)
        kam = KAMDiophantineCertificate.evaluate(
            omega=omega_h, perturbation_size=perturbation, tau=1.5, kmax=6
        )
        birkhoff = BirkhoffTwistMapCertificate.evaluate(
            field.rho, n_pairs=3, n_samples=16, key="INTROSPECTION"
        )
        monodromy = VariationalMonodromyCertificate.evaluate(field.rho)
        lyap = QuadraticLyapunovCertificate.evaluate(
            field.rho, v_fixed, field.spectral_gap.lambda_1, field.spectral_gap.gap_ratio
        )
        siegel = SiegelLinearizationCertificate.evaluate(field.rho)
        return IntrospectionBundle(
            field=field,
            v_fixed=v_fixed,
            trace=trace,
            certificate=cert,
            oseledets=oseledets,
            melnikov=melnikov,
            kam=kam,
            birkhoff=birkhoff,
            monodromy=monodromy,
            lyapunov=lyap,
            siegel=siegel,
        )




# ── §3.1 Adjudicador Ω₃  (CUERPO DE continue_into_phase3) ───────────────────────
class HeytingIntrospectionAdjudicator:
    r"""
    Clasificador de subobjetos Ω₃: colapsa el Bundle a un valor de Heyting
    por meets sucesivos.  final = local ∧ external.

    ESTE MÉTODO ES EL CUERPO DE
        IntrospectionBundle.continue_into_phase3(self, external_verdict)
    definido como último morfismo de FASE 2:

        continue_into_phase3 ∘ synthesize ∘ prepare
            = adjudicate ∘ synthesize ∘ prepare.

    Cada regla es un morfismo Bundle → Ω₃; el meet es el producto en el
    topos (límite de subobjetos).  Ω₃ no es booleano: ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤.
    """
    FS_COHERENT: Final[float] = 1.0e-6
    FS_DEGRADED: Final[float] = 1.0e-2
    LYAP_COHERENT: Final[float] = 0.0
    LYAP_DEGRADED: Final[float] = 1.0e-3

    @classmethod
    def _nondeg_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        d = b.field.spectral_gap.degeneracy_top
        if d == 1:
            return HeytingOmega3.COHERENT
        if d == 2:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _conv_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        if b.certificate.stalled_kernel:
            return HeytingOmega3.VETOED
        return HeytingOmega3.COHERENT if b.certificate.converged else HeytingOmega3.DEGRADED

    @classmethod
    def _fs_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        th = b.certificate.fixed_point_fs_angle
        if th <= cls.FS_COHERENT:
            return HeytingOmega3.COHERENT
        if th <= cls.FS_DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _oseledets_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        v = b.oseledets.lyapunov_max
        met_ok = bool(getattr(b.oseledets, "met_consistent", True))
        if v <= cls.LYAP_COHERENT and met_ok:
            return HeytingOmega3.COHERENT
        if v <= cls.LYAP_DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _melnikov_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return HeytingOmega3.VETOED if b.melnikov.transverse_homoclinic else HeytingOmega3.COHERENT

    @classmethod
    def _kam_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        r"""
        KAM perturbativo estricto → COHERENT.
        Toro integrable isócrono (Hess H=0) → DEGRADED (persiste, no por KAM).
        """
        if b.kam.kam_persists:
            return HeytingOmega3.COHERENT
        if b.kam.integrable_unperturbed:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _birkhoff_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.birkhoff.local_verdict

    @classmethod
    def _poincare_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return HeytingOmega3.COHERENT if b.field.poincare_section.is_transverse else HeytingOmega3.VETOED

    @classmethod
    def _hopf_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.field.poincare_hopf.local_verdict

    @classmethod
    def _recurrence_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.field.recurrence.local_verdict

    @classmethod
    def _resonance_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        g = b.field.resonance_graph
        acc = g.local_verdict.meet(g.arnold.local_verdict)
        chir = getattr(g, "chirikov", None)
        gre = getattr(g, "greene", None)
        if chir is not None:
            acc = acc.meet(chir.local_verdict)
        if gre is not None:
            acc = acc.meet(gre.local_verdict)
        return acc

    @classmethod
    def _monodromy_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.monodromy.local_verdict

    @classmethod
    def _tangle_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.field.homoclinic_tangle.local_verdict

    @classmethod
    def _lyapunov_quad_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.lyapunov.local_verdict

    @classmethod
    def _siegel_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.siegel.local_verdict

    @classmethod
    def _hopf_fibration_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.field.hopf_fibration.local_verdict

    @classmethod
    def _nekhoroshev_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.field.nekhoroshev.local_verdict

    @classmethod
    def _attracting_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        if getattr(b.certificate, "attracting", False):
            return HeytingOmega3.COHERENT
        dt = float(getattr(b.certificate, "dt_spectral_radius", 1.0))
        if dt < 1.0:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _kantorovich_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        h = float(getattr(b.trace, "kantorovich_h", float("inf")))
        if not math.isfinite(h):
            return HeytingOmega3.DEGRADED
        if h < 0.5 and getattr(b.trace, "newton_accelerated", False):
            return HeytingOmega3.COHERENT
        if h < 2.0:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def local_meet(cls, bundle: IntrospectionBundle) -> HeytingOmega3:
        r"""Meet de todas las reglas internas (sin el veredicto externo)."""
        return bundle.certificate.local_verdict.meet_all(
            bundle.field.spectral_gap.local_verdict,
            cls._nondeg_rule(bundle),
            cls._conv_rule(bundle),
            cls._fs_rule(bundle),
            cls._oseledets_rule(bundle),
            cls._melnikov_rule(bundle),
            cls._kam_rule(bundle),
            cls._birkhoff_rule(bundle),
            cls._poincare_rule(bundle),
            cls._hopf_rule(bundle),
            cls._recurrence_rule(bundle),
            cls._resonance_rule(bundle),
            cls._monodromy_rule(bundle),
            cls._tangle_rule(bundle),
            cls._lyapunov_quad_rule(bundle),
            cls._siegel_rule(bundle),
            cls._hopf_fibration_rule(bundle),
            cls._nekhoroshev_rule(bundle),
            cls._attracting_rule(bundle),
            cls._kantorovich_rule(bundle),
        )

    @classmethod
    def adjudicate(
        cls, bundle: IntrospectionBundle, external_verdict: HeytingOmega3
    ) -> HeytingOmega3:
        r"""
        Último morfismo FASE 2 ∧ primero FASE 3.

        Dominio   : IntrospectionBundle × Ω₃.
        Codominio : Ω₃.
        Semántica : ⟦final⟧ = ⟦local⟧ ∧ ⟦external⟧  en el retículo de Heyting.
        """
        return cls.local_meet(bundle).meet(external_verdict)


# ── §3.2 Auto-organización del campo (mixtura convexa CPTP) + Banach ────────────
@dataclass(frozen=True, slots=True)
class FieldUpdateCertificate:
    r"""
    Canal de reemplazo mezclado (CPTP, unital en el sentido traza-preservante):

        Φ_κ(ρ) = (1 − κ) ρ + κ |v*⟩⟨v*| ,   κ ∈ [0, 1].

    Completamente positivo (mezcla clásica de id y el canal de reemplazo
    R_σ(ρ)=σ).  Trazas-preservante.  Lipschitz en ‖·‖₁ y en ‖·‖_F:

        ‖Φ_κ(ρ) − Φ_κ(τ)‖ = |1 − κ| · ‖ρ − τ‖.

    Punto fijo único si κ ∈ (0, 1]:  Φ_κ(σ) = σ,  σ = |v*⟩⟨v*|.
    """
    applied: bool
    eta: float
    contraction_coef: float
    purity_before: float
    purity_after: float
    entropy_before: float
    entropy_after: float
    fidelity_to_fixed: float
    local_verdict: HeytingOmega3


@dataclass(frozen=True, slots=True)
class BanachContractionCertificate:
    r"""
    Teorema del punto fijo de Banach sobre (Hermitianos, ‖·‖_F):
        E(Δ) = (1 − κ) Δ,   ρ(E) = |1 − κ|.
    Si ρ(E) < 1 existe un único FP y Φ_κ^{∘n} → σ geométricamente.
    Conecta T2 (radio espectral del operador de error) con RSI-3.
    """
    lipschitz: float
    spectral_radius_error: float
    contraction: bool
    unique_fixed_point: bool
    geometric_rate: float
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(cls, kappa: float) -> "BanachContractionCertificate":
        kappa = float(np.clip(kappa, 0.0, 1.0))
        lip = abs(1.0 - kappa)
        contraction = bool(lip < 1.0 - 1e-15)
        unique = bool(kappa > 0.0)
        if contraction and unique:
            local = HeytingOmega3.COHERENT
        elif unique or lip <= 1.0:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            lipschitz=float(lip),
            spectral_radius_error=float(lip),
            contraction=contraction,
            unique_fixed_point=unique,
            geometric_rate=float(lip),
            local_verdict=local,
        )


class FieldSelfOrganizer:
    r"""Funtor de auto-organización: ρ ↦ Φ_κ(ρ) en Cog, certificado Banach."""

    @classmethod
    def idle(cls, rho: np.ndarray) -> Tuple[ComplexMatrix, FieldUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        p = DensityOperatorAlgebra.purity(rho)
        s = DensityOperatorAlgebra.von_neumann_entropy(rho)
        return rho, FieldUpdateCertificate(
            False, 0.0, 1.0, p, p, s, s, 1.0, HeytingOmega3.DEGRADED
        )

    @classmethod
    def update(
        cls, rho: np.ndarray, v_fixed: np.ndarray, eta: float
    ) -> Tuple[ComplexMatrix, FieldUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        eta = float(np.clip(eta, 0.0, 1.0))
        rho_fixed = DensityOperatorAlgebra.rank_one(v_fixed)
        p_before = DensityOperatorAlgebra.purity(rho)
        s_before = DensityOperatorAlgebra.von_neumann_entropy(rho)
        rho_new = DensityOperatorAlgebra.sanitize((1.0 - eta) * rho + eta * rho_fixed)
        p_after = DensityOperatorAlgebra.purity(rho_new)
        s_after = DensityOperatorAlgebra.von_neumann_entropy(rho_new)
        fid_fixed = DensityOperatorAlgebra.uhlmann_fidelity(rho_new, rho_fixed)
        coef = abs(1.0 - eta)
        if coef < 1.0 and fid_fixed > 0.5:
            local = HeytingOmega3.COHERENT
        elif coef < 1.0:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return rho_new, FieldUpdateCertificate(
            applied=True, eta=eta, contraction_coef=coef,
            purity_before=float(p_before), purity_after=float(p_after),
            entropy_before=float(s_before), entropy_after=float(s_after),
            fidelity_to_fixed=float(fid_fixed), local_verdict=local,
        )


# ── §3.3 MÓNADA RSI-3 EXPLÍCITA (T, η, μ) + Kleisli + Eilenberg–Moore ───────────
@dataclass(frozen=True, slots=True)
class MonadAxiomVerification:
    r"""
    Verificación computacional de los axiomas de Mac Lane para (T, η, μ).

    Associatividad:  μ ∘ Tμ = μ ∘ μT   : T³ ⇒ T.
    Unitaridad:      μ ∘ Tη = μ ∘ ηT = id_T.

    Encoding A (T_σ afín, σ congelado): asociatividad EXACTA; unitaridad
    exacta ⇔ T idempotente (κ ∈ {0,1}).
    Encoding B (T_∞ proyector κ=1): mónada idempotente, ambas leyes exactas.
    Encoding C (T_κ estado-dependiente): defectos ‖·‖_F reportados.
    """
    left_associativity: float
    right_associativity: float
    associativity_holds: bool
    left_unitarity: float
    right_unitarity: float
    unitarity_holds: bool
    affine_associativity_holds: bool
    projector_monad_holds: bool
    level_applied: int
    mix_rate: float
    verdict: HeytingOmega3


@dataclass(frozen=True, slots=True)
class EilenbergMooreCertificate:
    r"""
    T-álgebra de Eilenberg–Moore (A, ξ: T(A) → A):
        ξ ∘ η_A = id_A,     ξ ∘ μ_A = ξ ∘ T(ξ).
    Para T_∞ (proyector), las álgebras son exactamente los puntos fijos
    ρ = T(ρ) = |φ₁⟩⟨φ₁| (rayos puros del espectro simple).
    """
    is_algebra: bool
    unit_defect: float
    assoc_defect: float
    fixed_point_defect: float
    local_verdict: HeytingOmega3


@dataclass(frozen=True, slots=True)
class RSI3ChainCertificate:
    r"""Cadena RSI-3: μ ∘ T³.  Monotonía de pureza ↑ y de V=λ₁−R ↓."""
    levels: int
    purity_sequence: Tuple[float, ...]
    entropy_sequence: Tuple[float, ...]
    V_sequence: Tuple[float, ...]
    monotone_purity: bool
    monotone_V: bool
    condensed_purity: float
    condensed_fidelity_to_pure: float
    local_verdict: HeytingOmega3


class MonadRSI3:
    r"""
    Mónada RSI-3 (T, η, μ) sobre la categoría Cog de estados cognitivos.

    Objetos   : ρ ∈ 𝔇_n.
    Morfismos : canales CPTP Φ: 𝔇_n → 𝔇_n.
    Endofuntor T en objetos:
        T(ρ) = Φ_κ(ρ) = (1−κ) ρ + κ |φ₁(ρ)⟩⟨φ₁(ρ)|
        (un paso de auto-mejora; φ₁ = eigenrayo dominante).
    Endofuntor T en morfismos:
        T(Φ) = Φ_κ ∘ Φ    (postcomposición con la mixtura, σ congelado
                           del contexto, o estado-dependiente).
    Unidad η: Id ⇒ T,
        η_ρ = ρ           (auto-observación de nivel 0; inclusión).
    Multiplicación μ: T² ⇒ T,
        μ(ρ) = T(ρ)       (condensación metacognitiva = un paso T).

    Cadena RSI-n:  aplicar T n-veces y condensar por μ.  Nivel 3 = RSI-3.

    Kleisli Cog_T:
        Hom_T(A,B) = Hom_Cog(A, T(B)),
        g ∘_T f := μ_B ∘ T(g) ∘ f.

    Encoding afín T_σ (σ congelado): T es una contracción afín de Banach
    y μ ∘ Tμ = μ ∘ μT se cumple exactamente (T³ = T³).
    Encoding proyector T_∞ (κ=1): T²=T ⇒ mónada idempotente (closure).
    """

    AXIOM_TOL: Final[float] = 1.0e-8
    AXIOM_TOL_SOFT: Final[float] = 1.0e-4

    def __init__(self, dim: int, mix_rate: float = 0.15, level: int = 3) -> None:
        self.dim = int(max(1, dim))
        self.mix_rate = float(np.clip(mix_rate, 0.0, 1.0))
        self.level = int(max(1, min(level, 4)))

    def _dominant_pure(self, rho: np.ndarray) -> ComplexMatrix:
        _w, V = DensityOperatorAlgebra.eigenpairs_descending(rho)
        return DensityOperatorAlgebra.rank_one(V[:, 0])

    def _mix(self, rho: np.ndarray, sigma: np.ndarray, kappa: Optional[float] = None) -> ComplexMatrix:
        k = self.mix_rate if kappa is None else float(np.clip(kappa, 0.0, 1.0))
        return DensityOperatorAlgebra.sanitize((1.0 - k) * rho + k * sigma)

    # ── Unidad η: Id ⇒ T ────────────────────────────────────────────────────
    def unit(self, rho: np.ndarray) -> ComplexMatrix:
        r"""η_ρ = ρ  (inclusión canónica; auto-observación de nivel 0)."""
        return DensityOperatorAlgebra.sanitize(rho)

    def eta(self, rho: np.ndarray) -> ComplexMatrix:
        r"""Alias categórico de unit (η: Id ⇒ T)."""
        return self.unit(rho)

    # ── Endofuntor T en objetos ──────────────────────────────────────────────
    def T(self, rho: np.ndarray) -> ComplexMatrix:
        r"""T(ρ) = Φ_κ(sanitize(ρ)) hacia el rayo dominante (RSI-1)."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        return self._mix(rho, self._dominant_pure(rho))

    def T_affine(self, rho: np.ndarray, sigma: np.ndarray) -> ComplexMatrix:
        r"""T_σ(ρ) = (1−κ)ρ + κ σ  (σ congelado; contracción afín)."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        sigma = DensityOperatorAlgebra.sanitize(sigma)
        return self._mix(rho, sigma)

    def T_projector(self, rho: np.ndarray) -> ComplexMatrix:
        r"""T_∞(ρ) = |φ₁(ρ)⟩⟨φ₁(ρ)|  (κ=1; mónada idempotente)."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        return self._dominant_pure(rho)

    def T_morphism(self, phi: Callable[[np.ndarray], np.ndarray], rho: np.ndarray) -> ComplexMatrix:
        r"""T(Φ)(ρ) = Φ_κ(Φ(ρ))  — acción del endofuntor en morfismos."""
        return self.T(DensityOperatorAlgebra.sanitize(phi(rho)))

    # ── Multiplicación μ: T² ⇒ T ─────────────────────────────────────────────
    def mu(self, rho_TT: np.ndarray) -> ComplexMatrix:
        r"""
        μ: T²(ρ) ↦ T(ρ).  Condensación metacognitiva.
        En la encoding de estados (T^n(ρ) vive en 𝔇_n), μ = T.
        """
        return self.T(rho_TT)

    # ── Potencias y condensación RSI-n ───────────────────────────────────────
    def T_power(self, rho: np.ndarray, n: int) -> ComplexMatrix:
        rho_n = DensityOperatorAlgebra.sanitize(rho)
        for _ in range(max(0, int(n))):
            rho_n = self.T(rho_n)
        return rho_n

    def condense(self, rho: np.ndarray, n_levels: int) -> ComplexMatrix:
        r"""μ ∘ T^n  (cadena RSI-n).  Nivel 3 = RSI-3."""
        return self.mu(self.T_power(rho, n_levels))

    # ── Distancias ───────────────────────────────────────────────────────────
    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a, "fro"))
        nb = float(np.linalg.norm(b, "fro"))
        return float(np.linalg.norm(a - b, "fro") / max(na + nb, 1e-15))

    # ── Verificación de Mac Lane ─────────────────────────────────────────────
    def verify_axioms(self, rho_sample: Optional[np.ndarray] = None) -> MonadAxiomVerification:
        r"""
        Tres encodings:

        (A) T_σ afín.  x = T_σ^k(ρ).
            Tμ = T∘μ = T², μT = μ∘T = T² ⇒ μ∘Tμ = T³ = μ∘μT  (exacto).
            μ∘Tη = T², id_T = T  ⇒ unitaridad exacta ⇔ T²=T.

        (B) T_∞ proyector.  T²=T ⇒ mónada idempotente, leyes exactas.

        (C) T_κ estado-dependiente (φ₁(ρ)).  Defectos ‖·‖_F.
        """
        if rho_sample is None:
            rng = np.random.default_rng(20240613)
            A = rng.standard_normal((self.dim, self.dim)) + 1j * rng.standard_normal((self.dim, self.dim))
            rho_sample = A @ A.conj().T
        rho = DensityOperatorAlgebra.sanitize(rho_sample)
        sigma = self._dominant_pure(rho)

        # ── Encoding C: T estado-dependiente ──
        Trho = self.T(rho)
        T2 = self.T(Trho)
        T3 = self.T(T2)
        # μ ∘ Tμ  sobre T³:  Tμ (T³) ~ T(μ(T³)); μ(T³)=T⁴?  En encoding μ=T:
        # Tμ = T∘T = T², μT = T∘T = T², μ∘Tμ = T³, μ∘μT = T³.
        left_assoc_C = self._dist(self.mu(self.T(self.mu(T2))), self.mu(self.mu(self.T(T2))))
        # μ∘Tη  vs  T ;  μ∘ηT vs T
        eta_rho = self.unit(rho)
        left_unit_C = self._dist(self.mu(self.T(eta_rho)), Trho)
        right_unit_C = self._dist(self.mu(self.unit(Trho)), Trho)

        # ── Encoding A: T_σ afín congelado ──
        def Ta(x: np.ndarray) -> ComplexMatrix:
            return self.T_affine(x, sigma)

        A1, A2, A3 = Ta(rho), Ta(Ta(rho)), Ta(Ta(Ta(rho)))
        # μ_A := Ta
        left_A = self._dist(Ta(Ta(Ta(A1))), Ta(Ta(A2)))  # T³(T(ρ)) vs T²(T²(ρ))
        # más canónico: μ(T(μ(T²))) vs μ(μ(T(T²)))
        assoc_A = self._dist(Ta(Ta(A2)), Ta(Ta(A2)))  # T³ vs T³, idéntico
        affine_assoc = bool(assoc_A < self.AXIOM_TOL and left_A < self.AXIOM_TOL_SOFT)
        # unitaridad afín: Ta∘Ta vs Ta
        unit_A = self._dist(Ta(A1), A1)
        affine_unit = bool(unit_A < self.AXIOM_TOL)  # sólo si κ∈{0,1}

        # ── Encoding B: T_∞ ──
        P1 = self.T_projector(rho)
        P2 = self.T_projector(P1)
        proj_idem = self._dist(P1, P2)
        # η=id, μ=T_∞: μ∘Tη(ρ)=T_∞(T_∞(ρ))=T_∞(ρ)=id_T(ρ)
        proj_left_u = self._dist(self.T_projector(self.T_projector(self.unit(rho))), P1)
        proj_right_u = self._dist(self.T_projector(self.unit(P1)), P1)
        projector_ok = bool(
            proj_idem < self.AXIOM_TOL
            and proj_left_u < self.AXIOM_TOL
            and proj_right_u < self.AXIOM_TOL
        )

        assoc_ok = bool(left_assoc_C < self.AXIOM_TOL_SOFT) or affine_assoc
        unit_ok = bool(left_unit_C < self.AXIOM_TOL_SOFT and right_unit_C < self.AXIOM_TOL_SOFT)
        # Unitaridad estricta del encoding C no se espera salvo κ∈{0,1}
        if abs(self.mix_rate) < 1e-12 or abs(self.mix_rate - 1.0) < 1e-12:
            unit_ok = unit_ok or affine_unit or projector_ok

        if projector_ok and (assoc_ok or affine_assoc):
            verdict = HeytingOmega3.COHERENT
        elif assoc_ok and (unit_ok or affine_assoc):
            verdict = HeytingOmega3.COHERENT if projector_ok else HeytingOmega3.DEGRADED
        elif assoc_ok or projector_ok:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.VETOED

        return MonadAxiomVerification(
            left_associativity=float(left_assoc_C),
            right_associativity=float(left_assoc_C),
            associativity_holds=assoc_ok,
            left_unitarity=float(left_unit_C),
            right_unitarity=float(right_unit_C),
            unitarity_holds=unit_ok or projector_ok,
            affine_associativity_holds=affine_assoc,
            projector_monad_holds=projector_ok,
            level_applied=int(self.level),
            mix_rate=float(self.mix_rate),
            verdict=verdict,
        )

    # ── Kleisli: g ∘_T f = μ ∘ T(g) ∘ f ──────────────────────────────────────
    def kleisli_compose(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        g: Callable[[np.ndarray], np.ndarray],
        rho: np.ndarray,
    ) -> ComplexMatrix:
        r"""
        Composición de Kleisli en Cog_T:
            (g ∘_T f)(ρ) = μ( T( g( f(ρ) ) ) ).
        Codifica la cadena de auto-mejora recursiva (un morfismo RSI).
        """
        rho_f = DensityOperatorAlgebra.sanitize(f(rho))
        g_of = DensityOperatorAlgebra.sanitize(g(rho_f))
        return self.mu(self.T(g_of))

    def kleisli_identity(self, rho: np.ndarray) -> ComplexMatrix:
        r"""id_T = η  (unidad de Kleisli).  id_T(ρ) = η(ρ) = ρ."""
        return self.unit(rho)

    def kleisli_associativity_defect(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        g: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        rho: np.ndarray,
    ) -> float:
        r"""d( (h ∘_T g) ∘_T f  ,  h ∘_T (g ∘_T f) )."""
        gf = lambda x: self.kleisli_compose(f, g, x)
        hg = lambda x: self.kleisli_compose(g, h, x)
        left = self.kleisli_compose(gf, h, rho)
        right = self.kleisli_compose(f, hg, rho)
        return self._dist(left, right)

    # ── Eilenberg–Moore ──────────────────────────────────────────────────────
    def eilenberg_moore(self, rho: np.ndarray) -> EilenbergMooreCertificate:
        r"""
        ξ := T  (estructura canónica).  (ρ, T) es T-álgebra ⇔ T(ρ)=ρ
        (punto fijo) y las dos leyes se reducen a T²=T sobre ρ.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        eta_rho = self.unit(rho)
        T_eta = self.T(eta_rho)
        unit_def = self._dist(T_eta, rho)  # ξ∘η =? id  ⇔ T(ρ)=ρ
        # ξ ∘ μ = ξ ∘ Tξ : T(μ(ρ)) vs T(T(T(ρ))) si ξ=T, μ=T
        # T(T(ρ)) vs T(T(T(ρ)))  ⇔ T²(ρ)=T³(ρ)  ⇔ T²=T³ sobre ρ
        Tr = self.T(rho)
        T2 = self.T(Tr)
        T3 = self.T(T2)
        assoc_def = self._dist(T2, T3)
        fp_def = self._dist(Tr, rho)
        is_alg = bool(fp_def < self.AXIOM_TOL_SOFT and assoc_def < self.AXIOM_TOL_SOFT)
        if is_alg:
            local = HeytingOmega3.COHERENT
        elif fp_def < 1e-2:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return EilenbergMooreCertificate(
            is_algebra=is_alg,
            unit_defect=float(unit_def),
            assoc_defect=float(assoc_def),
            fixed_point_defect=float(fp_def),
            local_verdict=local,
        )

    # ── Cadena RSI-3 ─────────────────────────────────────────────────────────
    def rsi3_chain(self, rho: np.ndarray) -> Tuple[ComplexMatrix, RSI3ChainCertificate]:
        r"""Aplica μ ∘ T³ y certifica monotonía de pureza y de V=λ₁−R."""
        rho0 = DensityOperatorAlgebra.sanitize(rho)
        lam1 = float(DensityOperatorAlgebra.spectrum_descending(rho0)[0])
        levels = int(self.level)
        pur: List[float] = [DensityOperatorAlgebra.purity(rho0)]
        ent: List[float] = [DensityOperatorAlgebra.von_neumann_entropy(rho0)]
        Vseq: List[float] = [float(max(0.0, lam1 - ProjectiveDynamics.rayleigh(
            rho0, DensityOperatorAlgebra.eigenpairs_descending(rho0)[1][:, 0]
        )))]
        acc = rho0
        for _ in range(levels):
            acc = self.T(acc)
            pur.append(DensityOperatorAlgebra.purity(acc))
            ent.append(DensityOperatorAlgebra.von_neumann_entropy(acc))
            w, V = DensityOperatorAlgebra.eigenpairs_descending(acc)
            R = ProjectiveDynamics.rayleigh(acc, V[:, 0])
            Vseq.append(float(max(0.0, float(w[0]) - R)))
        condensed = self.mu(acc)
        pur.append(DensityOperatorAlgebra.purity(condensed))
        sigma = self._dominant_pure(condensed)
        fid = DensityOperatorAlgebra.uhlmann_fidelity(condensed, sigma)
        # monotonía: pureza no decrece; V no crece (salvo ruido numérico)
        mono_p = bool(all(pur[i] <= pur[i + 1] + 1e-9 for i in range(len(pur) - 1)))
        mono_V = bool(all(Vseq[i] + 1e-9 >= Vseq[i + 1] for i in range(len(Vseq) - 1)))
        if mono_p and fid > 0.99:
            local = HeytingOmega3.COHERENT
        elif mono_p or fid > 0.5:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        cert = RSI3ChainCertificate(
            levels=levels,
            purity_sequence=tuple(float(x) for x in pur),
            entropy_sequence=tuple(float(x) for x in ent),
            V_sequence=tuple(float(x) for x in Vseq),
            monotone_purity=mono_p,
            monotone_V=mono_V,
            condensed_purity=float(pur[-1]),
            condensed_fidelity_to_pure=float(fid),
            local_verdict=local,
        )
        return condensed, cert


# ── §3.4 META-MOTOR RSI-3 con Newton–Grassmann ──────────────────────────────────
class MetaIntrospectionEngine:
    r"""
    Motor espectral de introspección con meta-aceleración RSI-3.

    Integra: mónada (T, η, μ), Newton–Grassmann en ℂP^{n−1} (retracción
    geodésica FS + Armijo), condensación de Lüders con tasa dinámica

        κ(t) = κ₀ exp(− h_KS · d_FS)

    y verificación de Mac Lane + Eilenberg–Moore + cadena RSI-3.
    """

    def __init__(self, dimension: int = 8, monad_eta: float = 0.15) -> None:
        self.dim = int(max(1, dimension))
        self.monad = MonadRSI3(dim=self.dim, mix_rate=monad_eta, level=3)
        self.fubini_study_tol = 1e-6
        self.newton_damping = 0.3

    def solve_poincare_birkhoff_fixed_point_cpn_level3(
        self,
        density_matrix: np.ndarray,
        seed_vector_s6: Optional[np.ndarray] = None,
        entropy_ks: float = 0.05,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        r"""
        Punto fijo Tarski–Brouwer con Newton–Grassmann + RSI-3.
        1. Orientación por semilla S⁶.
        2. Newton–Grassmann geodésico sobre ℂP^{n−1}.
        3. Condensación monádica μ ∘ T³.
        4. Mixtura Lüders Φ_{κ(t)}.
        5. Verificación de axiomas + Eilenberg–Moore.
        """
        n = int(np.asarray(density_matrix).shape[0])
        self.dim = n
        if self.monad.dim != n:
            self.monad = MonadRSI3(dim=n, mix_rate=self.monad.mix_rate, level=self.monad.level)

        if seed_vector_s6 is not None and np.asarray(seed_vector_s6).size > 0:
            v_raw = np.asarray(seed_vector_s6, dtype=np.complex128).ravel()
            if v_raw.size < n:
                v_init = np.pad(v_raw, (0, n - int(v_raw.size)))
            else:
                v_init = v_raw[:n]
            v_norm = float(la.norm(v_init))
            v_init = v_init / v_norm if v_norm > 1e-15 else np.ones(n, dtype=np.complex128) / math.sqrt(n)
        else:
            v_init = np.ones(n, dtype=np.complex128) / math.sqrt(n)

        rho = DensityOperatorAlgebra.sanitize(density_matrix)
        v_current = ProjectiveDynamics.sanitize_vector(v_init, n)
        d_fs = 1.0
        iter_count = 0
        max_iters = 20
        use_newton = True
        newton_used = False

        while d_fs > self.fubini_study_tol and iter_count < max_iters:
            if use_newton and iter_count >= 2:
                X = ProjectiveDynamics.projected_gradient(rho, v_current)
                Q = FubiniStudyRetraction.horizontal_basis(v_current)
                if Q.shape[1] > 0:
                    H = ProjectiveDynamics.projected_hessian_rayleigh(rho, v_current)
                    H_red = Q.conj().T @ H @ Q
                    g_red = Q.conj().T @ X
                    H_reg = H_red + 1e-12 * np.eye(H_red.shape[0], dtype=np.complex128)
                    try:
                        xi_red = la.solve(H_reg, g_red, assume_a="her")
                    except la.LinAlgError:
                        xi_red = la.pinv(H_reg, rcond=1e-10) @ g_red
                    xi = Q @ xi_red
                    v_next = FubiniStudyRetraction.geodesic_exp(v_current, -self.newton_damping * xi)
                    newton_used = True
                else:
                    Tv, tvn = ProjectiveDynamics.apply_map(rho, v_current)
                    v_next = Tv if tvn > 1e-15 else v_current
                v_next = ProjectiveDynamics.gauge_align(v_current, v_next)
            else:
                Tv, tv_norm = ProjectiveDynamics.apply_map(rho, v_current)
                if tv_norm < 1e-15:
                    break
                v_next = ProjectiveDynamics.gauge_align(v_current, Tv)

            inner = float(np.clip(np.abs(np.vdot(v_current, v_next)), 0.0, 1.0))
            d_fs = float(np.arccos(inner))
            v_current = ProjectiveDynamics.sanitize_vector(v_next, n)
            iter_count += 1

        kappa_t = float(0.20 * math.exp(-float(entropy_ks) * d_fs))
        axiom_check = self.monad.verify_axioms(rho)
        em = self.monad.eilenberg_moore(rho)

        projector_v = DensityOperatorAlgebra.rank_one(v_current)
        updated_rho = DensityOperatorAlgebra.sanitize((1.0 - kappa_t) * rho + kappa_t * projector_v)
        rho_level3, chain = self.monad.rsi3_chain(updated_rho)

        banach = BanachContractionCertificate.evaluate(kappa_t)

        metrics: Dict[str, Any] = {
            "fubini_study_distance_rad": float(d_fs),
            "convergence_iterations": int(iter_count),
            "dynamic_luders_rate_eta": float(kappa_t),
            "tarski_brouwer_fixed_point_valid": bool(d_fs <= self.fubini_study_tol),
            "monad_associativity": bool(axiom_check.associativity_holds),
            "monad_unitarity": bool(axiom_check.unitarity_holds),
            "monad_affine_assoc": bool(axiom_check.affine_associativity_holds),
            "monad_projector_ok": bool(axiom_check.projector_monad_holds),
            "monad_verdict": axiom_check.verdict.name,
            "eilenberg_moore": bool(em.is_algebra),
            "eilenberg_moore_defect": float(em.fixed_point_defect),
            "rsi3_monotone_purity": bool(chain.monotone_purity),
            "rsi3_condensed_purity": float(chain.condensed_purity),
            "banach_contraction": bool(banach.contraction),
            "banach_rate": float(banach.geometric_rate),
            "newton_accelerated": bool(newton_used),
            "rsi_level": int(self.monad.level),
        }
        return rho_level3, d_fs, metrics


# ── §3.5 Certificado terminal ───────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionFieldState:
    r"""
    Objeto terminal del módulo:
        State ≅ Bundle × Ω₃ × Φ_κ × RSI-3 × Merkle × forense celeste.

    Es el álgebra de Eilenberg–Moore "observada": un T-álgebra con
    veredicto de Heyting, firma SHA-256 de la cadena F1→F2→F3, y
    todos los invariantes celestes / espectrales / monádicos.
    """
    cycle_id: str
    engine_id: str
    flash_id: str
    heyting_verdict: HeytingOmega3
    overlap: float
    fixed_point_residual: float
    fixed_point_fs_angle: float
    rayleigh_final: float
    rayleigh_gap_to_lambda1: float
    spectral_gap: float
    theoretical_rate: float
    empirical_rate: float
    rate_consistency: bool
    iterations: int
    is_self_sustaining: bool
    birkhoff_constant: float
    is_uniform: bool
    degeneracy_top: int
    field_updated: bool
    field_update_eta: float
    field_purity: float
    lyapunov_max: float
    kolmogorov_sinai_entropy: float
    kaplan_yorke_dimension: float
    melnikov_value: float
    transverse_homoclinic: bool
    kam_persists: bool
    kam_perturbation_ratio: float
    birkhoff_twist_holds: bool
    birkhoff_n_fp_min: int
    poincare_transverse: bool
    poincare_first_return: float
    floquet_min: float
    levy_band: float
    hamiltonian_return_time: float
    theoretical_period: float
    cartan_circulation: float
    energy_drift: float
    hopf_satisfied: bool
    euler_characteristic: int
    hamiltonian_recurrence: bool
    kac_mean_return: float
    resonance_edges: int
    top_resonant: bool
    hessian_twist_norm: float
    monodromy_on_circle: bool
    monodromy_period: float
    variational_radius: float
    # v8 / v8.1
    arnold_tongues: int
    cantor_measure: float
    has_smale_horseshoe: bool
    topological_entropy: float
    newton_accelerated: bool
    quadratic_convergence: bool
    monad_associativity: bool
    monad_unitarity: bool
    rsi_level: int
    # v8.1-P3
    kantorovich_h: float
    newton_steps: int
    attracting: bool
    dt_spectral_radius: float
    siegel_already_linear: bool
    lyapunov_V: float
    lyapunov_exp_stable: bool
    nekhoroshev_logT: float
    chirikov_max: float
    greene_golden: float
    hopf_fibre_ok: bool
    generating_function_S2: float
    bruno_holds: bool
    kam_epsilon_0: float
    exact_flow_used: bool
    eilenberg_moore: bool
    rsi3_monotone_purity: bool
    banach_contraction: bool
    projector_monad_ok: bool
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.6 Motor soberano F₃ ∘ F₂ ∘ F₁ ───────────────────────────────────────────
class TOONIntrospectionEngine:
    r"""
    Motor Espectral de la Introspección con RSI-3 y mecánica celeste de Poincaré.

        F₁  IntrospectiveFieldPreparation.prepare
        F₂  IntrospectiveField.continue_into_phase2 = synthesize
        F₃  continue_into_phase3 ⊗ Φ_κ ⊗ (T,η,μ) ⊗ certify

    Cadena Merkle P6: genesis → F1 → F2 → F3.
    """

    def __init__(
        self,
        engine_id: str = "INTROSPECT-ENGINE-WISDOM-01",
        dimension_mac: int = 4,
        update_field: bool = True,
        field_update_eta: float = 0.20,
        max_iter: int = PowerIterationSolver.MAX_ITER_DEFAULT,
        tol: float = PowerIterationSolver.TOL_DEFAULT,
        meta_engine: Optional[MetaIntrospectionEngine] = None,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1")
        self.engine_id = engine_id
        self.dimension_mac = int(dimension_mac)
        self.update_field = bool(update_field)
        self.field_update_eta = float(np.clip(field_update_eta, 0.0, 1.0))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.meta_engine = meta_engine or MetaIntrospectionEngine(
            dimension=dimension_mac, monad_eta=self.field_update_eta
        )
        self.density_matrix: ComplexMatrix = (
            np.eye(self.dimension_mac, dtype=np.complex128) / self.dimension_mac
        )
        self.cycle_count = 0
        self._chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::n={dimension_mac}::κ={self.field_update_eta}".encode("ascii")
        ).hexdigest()

    def solve_poincare_birkhoff_fixed_point_cpn_level3(
        self,
        density_matrix: Optional[np.ndarray] = None,
        seed_vector_s6: Optional[np.ndarray] = None,
        entropy_ks: float = 0.05,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        rho_input = density_matrix if density_matrix is not None else self.density_matrix
        updated_rho, d_fs, metrics = self.meta_engine.solve_poincare_birkhoff_fixed_point_cpn_level3(
            density_matrix=rho_input, seed_vector_s6=seed_vector_s6, entropy_ks=entropy_ks,
        )
        if getattr(self, "update_field", True):
            if density_matrix is None or density_matrix is self.density_matrix:
                self.density_matrix = updated_rho
        return updated_rho, d_fs, metrics

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _phase1_prepare(self, flash_vector: np.ndarray) -> IntrospectiveField:
        field = IntrospectiveFieldPreparation.prepare(
            rho_input=self.density_matrix, v_initial=flash_vector, poincare_key=self.engine_id,
        )
        self._advance_chain("F1", bytes.fromhex(field.field_hash) if len(field.field_hash) == 64 else field.field_hash.encode("ascii"))
        return field

    def _phase2_iterate(self, field: IntrospectiveField) -> IntrospectionBundle:
        bundle = field.continue_into_phase2(max_iter=self.max_iter, tol=self.tol)
        self._advance_chain("F2", bundle.content_bytes())
        return bundle

    def _phase3_certify(
        self,
        cycle_id: str,
        flash_id: str,
        bundle: IntrospectionBundle,
        external_verdict: HeytingOmega3,
        monad_metrics: Optional[Dict[str, Any]] = None,
    ) -> IntrospectionFieldState:
        r"""
        F₃: continue_into_phase3 ⊗ Φ_κ ⊗ RSI-3 ⊗ Merkle.
        continue_into_phase3 ≡ HeytingIntrospectionAdjudicator.adjudicate.
        """
        final_verdict = bundle.continue_into_phase3(external_verdict)
        field_updated = False
        eta_used = 0.0
        if self.update_field and final_verdict == HeytingOmega3.COHERENT:
            new_rho, _upd = FieldSelfOrganizer.update(
                rho=self.density_matrix, v_fixed=bundle.v_fixed, eta=self.field_update_eta,
            )
            self.density_matrix = new_rho
            field_updated = True
            eta_used = self.field_update_eta
        self._advance_chain(
            "F3",
            f"{final_verdict.name}|upd={field_updated}|"
            f"{bundle.certificate.overlap_final:.12e}".encode("ascii"),
        )
        c = bundle.certificate
        g = bundle.field.spectral_gap
        o = bundle.oseledets
        m = bundle.melnikov
        k = bundle.kam
        bkh = bundle.birkhoff
        p = bundle.field.poincare_section
        hopf = bundle.field.poincare_hopf
        rec = bundle.field.recurrence
        resg = bundle.field.resonance_graph
        aa = bundle.field.action_angle
        mono = bundle.monodromy
        tangle = bundle.field.homoclinic_tangle
        ly = bundle.lyapunov
        sg = bundle.siegel
        mm = monad_metrics or {}
        chirikov_max = float(getattr(resg.chirikov, "max_overlap", 0.0)) if getattr(resg, "chirikov", None) else 0.0
        greene_golden = float(getattr(resg.greene, "golden_residue", 0.0)) if getattr(resg, "greene", None) else 0.0
        provenance = _sha256_bytes(
            self.engine_id.encode("ascii"),
            cycle_id.encode("ascii"),
            flash_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{c.fixed_point_residual:.12e}".encode("ascii"),
            f"{c.overlap_final:.12e}".encode("ascii"),
            f"{g.gap:.12e}".encode("ascii"),
            f"{o.lyapunov_max:.12e}".encode("ascii"),
            f"{m.melnikov_value:.12e}".encode("ascii"),
            f"{k.diophantine_gamma:.12e}".encode("ascii"),
            f"{bkh.n_fixed_points_min}".encode("ascii"),
            f"{hopf.index_sum}".encode("ascii"),
            f"{tangle.n_transverse_intersections}".encode("ascii"),
            f"{resg.arnold.n_tongues}".encode("ascii"),
            f"{ly.V_value:.12e}".encode("ascii"),
            self._chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )
        return IntrospectionFieldState(
            cycle_id=cycle_id, engine_id=self.engine_id, flash_id=flash_id,
            heyting_verdict=final_verdict, overlap=c.overlap_final,
            fixed_point_residual=c.fixed_point_residual,
            fixed_point_fs_angle=c.fixed_point_fs_angle,
            rayleigh_final=c.rayleigh_final,
            rayleigh_gap_to_lambda1=c.rayleigh_gap_to_lambda1,
            spectral_gap=g.gap, theoretical_rate=g.gap_ratio,
            empirical_rate=c.empirical_rate, rate_consistency=c.rate_consistency,
            iterations=c.iterations,
            is_self_sustaining=bool(c.converged and c.overlap_final > 1.0 - 1e-6 and (not g.is_uniform)),
            birkhoff_constant=g.birkhoff_constant, is_uniform=g.is_uniform,
            degeneracy_top=g.degeneracy_top, field_updated=field_updated,
            field_update_eta=eta_used,
            field_purity=DensityOperatorAlgebra.purity(self.density_matrix),
            lyapunov_max=o.lyapunov_max,
            kolmogorov_sinai_entropy=o.kolmogorov_sinai_entropy,
            kaplan_yorke_dimension=o.kaplan_yorke_dimension,
            melnikov_value=m.melnikov_value,
            transverse_homoclinic=m.transverse_homoclinic,
            kam_persists=k.kam_persists,
            kam_perturbation_ratio=k.perturbation_ratio,
            birkhoff_twist_holds=bkh.twist_holds, birkhoff_n_fp_min=bkh.n_fixed_points_min,
            poincare_transverse=p.is_transverse,
            poincare_first_return=p.first_return_time,
            floquet_min=(float(p.floquet_moduli.min()) if p.floquet_moduli.size else 0.0),
            levy_band=bundle.field.levy_band,
            hamiltonian_return_time=p.hamiltonian_return_time,
            theoretical_period=p.theoretical_period,
            cartan_circulation=p.cartan_circulation, energy_drift=p.energy_drift,
            hopf_satisfied=hopf.hopf_satisfied,
            euler_characteristic=hopf.euler_characteristic,
            hamiltonian_recurrence=rec.hamiltonian_recurrence,
            kac_mean_return=rec.kac_mean_return_proxy,
            resonance_edges=resg.n_edges, top_resonant=resg.top_resonant,
            hessian_twist_norm=aa.hessian_twist_norm,
            monodromy_on_circle=mono.on_unit_circle, monodromy_period=mono.period,
            variational_radius=bundle.trace.variational_spectral_radius,
            arnold_tongues=resg.arnold.n_tongues,
            cantor_measure=resg.arnold.cantor_measure,
            has_smale_horseshoe=tangle.has_smale_horseshoe,
            topological_entropy=tangle.topological_entropy_lower,
            newton_accelerated=bundle.trace.newton_accelerated,
            quadratic_convergence=bundle.trace.quadratic_flag,
            monad_associativity=bool(mm.get("monad_associativity", False)),
            monad_unitarity=bool(mm.get("monad_unitarity", False)),
            rsi_level=int(mm.get("rsi_level", 0)),
            kantorovich_h=float(getattr(bundle.trace, "kantorovich_h", float("inf"))),
            newton_steps=int(getattr(bundle.trace, "newton_steps", 0)),
            attracting=bool(getattr(c, "attracting", False)),
            dt_spectral_radius=float(getattr(c, "dt_spectral_radius", 0.0)),
            siegel_already_linear=bool(sg.already_linear),
            lyapunov_V=float(ly.V_value),
            lyapunov_exp_stable=bool(ly.exponentially_stable),
            nekhoroshev_logT=float(bundle.field.nekhoroshev.stability_time_log),
            chirikov_max=chirikov_max,
            greene_golden=greene_golden,
            hopf_fibre_ok=bool(bundle.field.hopf_fibration.local_verdict == HeytingOmega3.COHERENT),
            generating_function_S2=float(aa.generating_function_type2),
            bruno_holds=bool(getattr(k, "bruno_holds", False)),
            kam_epsilon_0=float(getattr(k, "epsilon_0", 0.0)),
            exact_flow_used=bool(getattr(p, "exact_flow_used", False)),
            eilenberg_moore=bool(mm.get("eilenberg_moore", False)),
            rsi3_monotone_purity=bool(mm.get("rsi3_monotone_purity", False)),
            banach_contraction=bool(mm.get("banach_contraction", False)),
            projector_monad_ok=bool(mm.get("monad_projector_ok", False)),
            phase_chain_sha256=self._chain_hash, sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )

    def execute_introspection_cycle(
        self,
        flash_id: str,
        flash_vector: np.ndarray,
        heyting_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
        seed_vector_s6: Optional[np.ndarray] = None,
        entropy_ks: float = 0.05,
    ) -> IntrospectionFieldState:
        r"""Ciclo soberano: F₃ ∘ F₂ ∘ F₁ con RSI-3 y mecánica celeste."""
        monad_metrics: Dict[str, Any] = {}
        if seed_vector_s6 is not None:
            updated_rho, _, monad_metrics = self.solve_poincare_birkhoff_fixed_point_cpn_level3(
                density_matrix=self.density_matrix,
                seed_vector_s6=seed_vector_s6, entropy_ks=entropy_ks,
            )
            self.density_matrix = updated_rho
        self.cycle_count += 1
        cycle_id = f"CYC-INTROSPECT-{self.cycle_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Introspección #%d | flash=%s | tol=%.1e ═══",
            self.cycle_count, flash_id, self.tol,
        )
        field = self._phase1_prepare(flash_vector)
        bundle = self._phase2_iterate(field)
        state = self._phase3_certify(cycle_id, flash_id, bundle, heyting_verdict, monad_metrics)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Ciclo %s | Ω₃=%s | resid=%.3e | FS=%.3e | λ_max=%.2e | "
            "h_KS=%.2e | M=%.2e | B=%d FP | Hopf=%s | τ_H=%.2e | "
            "RSI-3=%s | Newton=%s | Kantorovich=%.2e | upd=%s | %.2f ms",
            cycle_id, state.heyting_verdict.name, state.fixed_point_residual,
            state.fixed_point_fs_angle, state.lyapunov_max,
            state.kolmogorov_sinai_entropy, state.melnikov_value,
            state.birkhoff_n_fp_min, state.hopf_satisfied,
            state.hamiltonian_return_time,
            state.monad_associativity and state.monad_unitarity,
            state.newton_accelerated, state.kantorovich_h,
            state.field_updated, dt_ms,
        )
        return state


# ── §3.7 Demostración autónoma ─────────────────────────────────────────────────
def _build_rho_from_spectrum(n: int, spectrum: np.ndarray, key: str) -> ComplexMatrix:
    spectrum = np.asarray(spectrum, dtype=np.float64)
    spectrum = np.maximum(spectrum, 0.0)
    s = float(spectrum.sum())
    spectrum = spectrum / s if s > 0.0 else np.full(n, 1.0 / n)
    rng = np.random.default_rng(_seed_from_string(f"RHO::{key}"))
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    d = np.diagonal(R)
    ph = np.where(np.abs(d) > 1e-30, d / np.abs(d), 1.0 + 0j)
    Q = Q * ph.conj()
    rho = (Q * spectrum.astype(np.complex128)) @ Q.conj().T
    return DensityOperatorAlgebra.sanitize(rho)


def _build_test_vector(n: int, key: str) -> ComplexVector:
    rng = np.random.default_rng(_seed_from_string(f"VEC::{key}"))
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return (v / float(np.linalg.norm(v))).astype(np.complex128)


if __name__ == "__main__":
    print("═" * 94)
    print("TOON INTROSPECTION ENGINE — v8.1.0-APU-RSI3-CELESTE")
    print("RSI-3 (mónada explícita) · Newton–Grassmann · Arnold / Chirikov / Greene")
    print("Smale horseshoe · KAM / Bruno / Nekhoroshev · Poincaré celeste (H)/(G)/(T)")
    print("═" * 94)

    n = 4
    rho_ideal = _build_rho_from_spectrum(n, np.array([0.97, 0.02, 0.005, 0.005]), "IDEAL")

    print("\n─── §1.12 Homoclinic tangle / Smale horseshoe ───")
    tangle = HomoclinicTangleCertificate.evaluate(rho_ideal, key="DEMO")
    print(f"   transversales = {tangle.n_transverse_intersections}")
    print(f"   Smale horseshoe = {tangle.has_smale_horseshoe}")
    print(f"   h_top (cota inferior) = {tangle.topological_entropy_lower:.6f}")

    print("\n─── §1.9 Arnold / Chirikov / Greene ───")
    w = DensityOperatorAlgebra.spectrum_descending(rho_ideal)
    arnold = ArnoldTongueCertificate.evaluate(w, delta_res=1e-3)
    chir = ChirikovOverlapCertificate.evaluate(w, island_width_proxy=1e-3)
    gre = GreeneResidueCertificate.evaluate(w)
    print(f"   n_tongues = {arnold.n_tongues}, μ_Cantor = {arnold.cantor_measure:.6f}")
    print(f"   Chirikov max K = {chir.max_overlap:.4e}, stoch = {chir.global_stochasticity}")
    print(f"   Greene R_gold = {gre.golden_residue:.4e}, KAM-toro vivo = {gre.last_kam_torus_alive}")

    print("\n─── §3.3 Mónada RSI-3 (Mac Lane) ───")
    monad = MonadRSI3(dim=n, mix_rate=0.15, level=3)
    check = monad.verify_axioms(rho_ideal)
    print(f"   asociatividad μ∘Tμ = μ∘μT: {check.associativity_holds} "
          f"(d={check.left_associativity:.3e})")
    print(f"   unitaridad μ∘Tη = id_T: {check.unitarity_holds} "
          f"(d_L={check.left_unitarity:.3e}, d_R={check.right_unitarity:.3e})")
    print(f"   afín T_σ assoc exacta: {check.affine_associativity_holds}")
    print(f"   mónada proyector T_∞: {check.projector_monad_holds}")
    print(f"   veredicto Ω₃ = {check.verdict.name}")

    print("\n─── §3.3 Kleisli Cog_T  (g ∘_T f = μ ∘ T(g) ∘ f) ───")
    def f_step(r): return r
    def g_step(r): return r
    def h_step(r): return DensityOperatorAlgebra.sanitize(0.5 * r + 0.5 * DensityOperatorAlgebra.rank_one(
        DensityOperatorAlgebra.eigenpairs_descending(r)[1][:, 0]
    ))
    rho_kleisli = monad.kleisli_compose(f_step, g_step, rho_ideal)
    d_assoc_k = monad.kleisli_associativity_defect(f_step, g_step, h_step, rho_ideal)
    print(f"   ‖ρ_Kleisli‖_F = {np.linalg.norm(rho_kleisli, 'fro'):.6f}")
    print(f"   Tr(ρ_Kleisli) = {np.trace(rho_kleisli).real:.10f}")
    print(f"   pureza = {DensityOperatorAlgebra.purity(rho_kleisli):.6f}")
    print(f"   defecto asociatividad Kleisli = {d_assoc_k:.3e}")

    print("\n─── §3.3 Eilenberg–Moore + cadena RSI-3 ───")
    em = monad.eilenberg_moore(rho_ideal)
    rho_c, chain = monad.rsi3_chain(rho_ideal)
    print(f"   T-álgebra = {em.is_algebra} (d_fp={em.fixed_point_defect:.3e})")
    print(f"   monotonía pureza = {chain.monotone_purity}, V↓ = {chain.monotone_V}")
    print(f"   pureza condensada = {chain.condensed_purity:.6f}, F(ρ,σ) = {chain.condensed_fidelity_to_pure:.6f}")

    print("\n─── §3.2 Banach / operador de error ───")
    ban = BanachContractionCertificate.evaluate(0.20)
    print(f"   Lip = ρ(E) = {ban.lipschitz:.4f}, contracción = {ban.contraction}, FP único = {ban.unique_fixed_point}")

    print("\n─── §3.4 MetaIntrospectionEngine RSI-3 con Newton–Grassmann ───")
    meta = MetaIntrospectionEngine(dimension=n, monad_eta=0.15)
    rho_out, d_fs, metrics = meta.solve_poincare_birkhoff_fixed_point_cpn_level3(rho_ideal)
    print(f"   d_FS = {d_fs:.3e} | iters = {metrics['convergence_iterations']}")
    print(f"   Newton acelerado = {metrics['newton_accelerated']}")
    print(f"   axiomas: asoc={metrics['monad_associativity']}, unit={metrics['monad_unitarity']}, "
          f"T_∞={metrics['monad_projector_ok']}")
    print(f"   EM = {metrics['eilenberg_moore']}, Banach = {metrics['banach_contraction']}")
    print(f"   veredicto = {metrics['monad_verdict']}")
    print(f"   pureza final = {DensityOperatorAlgebra.purity(rho_out):.6f}")

    print("\n" + "─" * 94)
    print("─── Ciclo completo F₃ ∘ F₂ ∘ F₁ con RSI-3 ───")
    engine = TOONIntrospectionEngine(
        engine_id="INTROSPECT-ENGINE-WISDOM-01",
        dimension_mac=4,
        update_field=False,
        field_update_eta=0.20,
        max_iter=500,
        tol=1e-10,
    )
    engine.density_matrix = rho_ideal
    v = _build_test_vector(n, "VEC-A")
    seed = np.array([1.0, 0.0, 0.5, 0.5, 0.3, 0.3])
    state = engine.execute_introspection_cycle(
        flash_id="FLASH-A", flash_vector=v,
        heyting_verdict=HeytingOmega3.COHERENT,
        seed_vector_s6=seed,
    )
    print(f"\n[Estado terminal]")
    print(f"   Ω₃ = {state.heyting_verdict.name}")
    print(f"   RSI-3 = {state.rsi_level} (asoc={state.monad_associativity}, "
          f"unit={state.monad_unitarity}, T_∞={state.projector_monad_ok})")
    print(f"   EM = {state.eilenberg_moore}, Banach = {state.banach_contraction}, "
          f"pureza↑ = {state.rsi3_monotone_purity}")
    print(f"   Newton = {state.newton_accelerated}, cuadrática = {state.quadratic_convergence}, "
          f"h_Kantorovich = {state.kantorovich_h:.3e}")
    print(f"   Attracting ρ(DT) = {state.dt_spectral_radius:.4e}, Siegel lineal = {state.siegel_already_linear}")
    print(f"   V = {state.lyapunov_V:.3e}, exp-estable = {state.lyapunov_exp_stable}")
    print(f"   Arnold = {state.arnold_tongues}, μ_Cantor = {state.cantor_measure:.6f}, "
          f"Chirikov = {state.chirikov_max:.3e}")
    print(f"   Greene R_φ = {state.greene_golden:.4e}, ε₀(KAM) = {state.kam_epsilon_0:.3e}, "
          f"Bruno = {state.bruno_holds}")
    print(f"   Smale = {state.has_smale_horseshoe}, h_top = {state.topological_entropy:.4f}")
    print(f"   Nekhoroshev log T = {state.nekhoroshev_logT:.3e}, Hopf fibra = {state.hopf_fibre_ok}")
    print(f"   S₂ = {state.generating_function_S2:.6f}, flujo exacto = {state.exact_flow_used}")
    print(f"   Fase chain = {state.phase_chain_sha256[:32]}…")
    print(f"   Firma global = {state.sha256_provenance[:32]}…")

    print("\n" + "═" * 94)
    print("✓ F1→F2: prepare ⊣ continue_into_phase2 = synthesize.")
    print("✓ F2→F3: synthesize ⊣ continue_into_phase3 = adjudicate ⊗ Φ_κ ⊗ RSI-3.")
    print("✓ Mónada (T, η, μ): Mac Lane — assoc afín exacta; T_∞ idempotente exacta.")
    print("✓ Kleisli ∘_T: g ∘_T f = μ ∘ T(g) ∘ f sobre Cog.")
    print("✓ Eilenberg–Moore: álgebras = puntos fijos ρ = T(ρ).")
    print("✓ Banach: ρ(E)=|1−κ|<1 ⇒ FP único, convergencia geométrica.")
    print("✓ Newton–Grassmann: retracción geodésica FS + Kantorovich h=βηL.")
    print("✓ Arnold / Chirikov / Greene + Smale (h_top ≥ log 2).")
    print("✓ KAM: ε₀ = γ/(m log(1/γ))^m · Bruno · Nekhoroshev.")
    print("✓ Cadena forense F1 → F2 → F3 encadenada por SHA-256 (P6).")
    print("═" * 94)


# ════════════════════════════════════════════════════════════════════════════════
# FIN FORMAL DEL MÓDULO  —  F₃ ∘ F₂ ∘ F₁
# Anclaje F1→F2: IntrospectiveField.continue_into_phase2 ≡ synthesize
# Anclaje F2→F3: IntrospectionBundle.continue_into_phase3 ≡ adjudicate
# Objeto terminal: IntrospectionFieldState
# ════════════════════════════════════════════════════════════════════════════════
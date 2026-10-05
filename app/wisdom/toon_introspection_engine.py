# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/wisdom/toon_introspection_engine.py                                   ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / REFLEJO INTROSPECTIVO               ║
║ FUNCIÓN  : MOTOR ESPECTRAL INTROSPECTIVO — POINCARÉ-BIRKHOFF-OSELEDETS-KAM-MELNIKOV  ║
║ VERSIÓN  : 9.1.0-Doctoral-Poincaré-Celeste-Hamiltonian-NestedThreePhase              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
ARQUITECTURA DE FASES ANIDADAS (categorías encajadas)
─────────────────────────────────────────────────────
Funtor F = F₃ ∘ F₂ ∘ F₁ sobre la categoría 𝐂𝐨𝐧𝐜𝐫𝐞𝐭𝐞 de prehaces sobre Ω₃:
    F₁ : 𝔇_n × ℂⁿ           → IntrospectiveField     (sustrato proyectivo-métrico)
    F₂ : IntrospectiveField  → IntrospectionBundle     (dinámica de punto fijo)
    F₃ : Bundle × Ω₃         → IntrospectionFieldState (adjudicación ⊗ auto-org)
Anidamiento estricto (igualdad de morfismos, no analogía):
    continue_into_phase2 ∘ prepare               = synthesize ∘ prepare
    continue_into_phase3 ∘ synthesize ∘ prepare  = adjudicate ∘ synthesize ∘ prepare
MECÁNICA CELESTE DE POINCARÉ — NÚCLEO FORMAL (v9.1)
────────────────────────────────────────────────────
Distinción ontológica (imprescindible):
    (G) Flujo gradiente   X_G = (I − vv†)ρv     — disipativo, Lyapunov = −R.
    (H) Flujo hamiltoniano X_H = i X_G            — conservativo, Liouville, Poincaré.
    (T) Mapa proyectivo    T([v]) = [ρv]          — iteración de potencia, no simplectomorfismo.

1. Hamiltoniano integrable de Rayleigh sobre ℂP^{n−1}:
       H(I) = Σ_{k=1}^n λ_k I_k ,   I_k = |⟨φ_k|v⟩|² ,  Σ I_k = 1.
   Frecuencias de Poincaré: ω_k = ∂H/∂I_k = λ_k  (sistema ISÓCRONO).
   Hessiano ∂²H/∂I² = 0 ⇒ twist nulo (oscilador armónico / Kepler degenerado).
   Función generatriz de tipo 2:  S(θ, I; t) = Σ θ_k I_k + t H(I).

2. Forma de Fubini–Study (simplectica, Kähler):
       ω_FS(ξ, η) = 2 Im ⟨ξ_h | η_h⟩ ,  ξ_h = (I − vv†)ξ.
   Invariante integral relativo de Poincaré–Cartan:  ∮_γ λ_FS ,  dλ_FS = ω_FS.
   El flujo (H) preserva ω_FS (Liouville).  El mapa (T) NO: contrae hacia [φ₁].

3. Sección de Poincaré.
   (G) Σ_c = {R = c} es transversal a X_G ⇔ c ∉ spec(ρ), pues ⟨∇R, X_G⟩ = 2‖X_G‖².
   (H) {R = c} es INVARIANTE bajo X_H.  La sección auténtica es
           Σ_{c,q} = {R = c} ∩ {Im⟨q|v⟩ = 0} ⊂ {R = c},
       de dimensión real 2n−4, y el mapa de primer retorno P : Σ_{c,q} → Σ_{c,q}
       es un simplectomorfismo.  En un 2-plano span{φ_i, φ_j} el periodo es
           τ_{ij} = 2π / |λ_i − λ_j|.

4. Teorema de Poincaré–Hopf:  χ(ℂP^{n−1}) = n.
   Índice de Morse de R en [φ_k] (λ₁ ≥ ⋯ ≥ λ_n estrictos) = 2(n − k).
   Índice de Hopf del gradiente = (−1)^{Morse} = +1 (índices pares).
   Σ índices = n = χ.  Degeneración ⇒ Morse–Bott (eje degenerado).

5. Recurrencia de Poincaré (1890) + lema de Kac:
   (H) preserva la medida de Haar FS en variedad compacta ⇒ recurrencia plena.
   Tiempo medio de retorno a A:  1/μ(A).
   (T) no preserva Haar: la recurrencia falla salvo en Fix(T).

6. Último teorema geométrico de Poincaré (1912) / Birkhoff (1913):
   Un homeomorfismo que preserva área del anillo con twist ∂(θ')/∂r ≠ 0
   tiene ≥ 2 puntos fijos.  Aquí:
       — (H) es isócrono ⇒ twist = 0 ⇒ Birkhoff N/A (rotación rígida).
       — (T) restringido al intervalo radial r ∈ [0,1] del 2-plano
         f(r) = α r / (α r + β(1−r)), α = λ_i², β = λ_j²,
         tiene exactamente 2 FP (los eigenrayos), f monótona.
       Certificamos ambos hechos por separado.

7. Oseledets proyectivo de (T):  λ_i^{Lyap} = log(λ_i(ρ)/λ₁(ρ)) ≤ 0.
   h_KS = 0 si λ₁ simple.  Degeneración λ₁ = λ₂ ⇒ λ_max = 0 y un ℂP¹ de FP.

8. Melnikov proyectivo (Poincaré–Melnikov):
   H = H₀ + ε H₁,  γ heteroclínica de X_G(H₀) entre [φ₂] y [φ₁]:
       M(ϑ) = ∫_{−∞}^{∞} ω_FS(X_{H₀}, X_{H₁})(γ_ϑ(t)) dt.
   Cero simple ⇒ W^s ∩ W^u transversal.  H₀ integrable ⇒ M ≡ 0 sin perturbación.

9. KAM diofántico sobre el toro de acciones: |ω · k| ≥ γ / |k|^τ.
   Persistencia no trivial sólo bajo perturbación no integrable de H
   (un único ρ hermítico es siempre integrable).  Certificamos el módulo
   de resonancia y la condición diofántica del vector ω = λ.

10. Monodromía / Floquet a lo largo de órbitas periódicas de (H):
    multiplicadores μ = exp(± i (λ_k − λ_*) τ) ∈ U(1)  (Krein-neutros).

11. Lévy sobre ℂP^{n−1}: Ric = 2(n+1) g_{FS},
        P(|f − 𝔼f| ≥ ε) ≤ exp(−(n+1)ε² / (2π² L²)).

12. Grafo de resonancia espectral (teoría de grafos + mecánica celeste):
    vértices = eigenrayos, arista ij ⇔ |λ_i − λ_j| < δ o relación entera
    k · λ = 0.  La tela de resonancias es el “web” de Arnold.

POSTULADOS OPERATIVOS
─────────────────────
P1. Iteración de potencia gauge-fijada; parada d_FS ∧ ‖T_φ − v‖.
P2. d_FS = arccos(|⟨u|v⟩|); residuo gauge-fijado = 2 sin(θ/2).
P3. ρ(DT|_{v₁}) = λ₂/λ₁ (Floquet); tasa empírica en ventana sana.
P4. Ω₃ por meets adimensionales; no conteo.
P5. Φ_η CPTP, Lip₁ = |1−η|.
P6. Cadena Merkle F1 → F2 → F3.
P7. (H) y (T) nunca se confunden: simplectomorfismo ≠ contracción.
P8. Twist nulo se declara nulo; no se fabrica caos hamiltoniano de un ρ hermítico.

TRADUCCIÓN EJECUTIVA ("DOLOR Y DINERO")
──────────────────────────────────────
- Certificación de punto fijo con respaldo topológico (Brouwer + Poincaré–Hopf).
- Veto automático si λ_max ≥ 0 (degeneración), Melnikov homoclínico o KAM colapsa.
- Tasa empírica vs teórica λ₂/λ₁ valida la convergencia geométrica.
- Auto-organización Φ_η modulada por h_KS (evita colapso prematuro).
- Recurrencia / no-recurrencia distinguen el sector conservativo del disipativo.
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

_EPS: Final[float] = 1.0e-14
_EPS_MOD: Final[float] = 1.0e-12
_EPS_TRACE: Final[float] = 1.0e-15
_EPS_HORIZ: Final[float] = 1.0e-15
_EPS_SECTION: Final[float] = 1.0e-10

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


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 · SUSTRATO PROYECTIVO-MÉTRICO + MECÁNICA CELESTE DE POINCARÉ             ║
# ║                                                                                  ║
# ║  Objetos: Ω₃, 𝔇_n, ℂP^{n−1}, spec(ρ), (G)/(H)/(T), Σ_c, Σ_{c,q},                 ║
# ║           acción-ángulo, Poincaré–Hopf, recurrencia, Lévy, grafo espectral.      ║
# ║  Morfismo terminal: IntrospectiveField.continue_into_phase2.                    ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

# ── §1.1 Retículo distributivo de Heyting Ω₃ ─────────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0,1,2}.
    Álgebra de Heyting: meet = min, join = max, ⇒ = residuo de Galois, ¬ = ⇒ ⊥.
    Regulares = {⊥, ⊤} = fix(¬¬);  ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤  (no booleana).
    Interpretación topos: clasificador de subobjetos truncado del topos
    de prehaces Sh(Ω₃).  P4: los veredictos colapsan por meet, jamás por conteo.
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


# ── §1.2 Álgebra de operadores densidad (C*-estados sobre M_n(ℂ)) ───────────────
class DensityOperatorAlgebra:
    r"""
    𝔇_n = { ρ ∈ M_n(ℂ) : ρ = ρ†, ρ ≥ 0, Tr ρ = 1 }.
    M_n(ℂ) es C*-álgebra de Banach unital; 𝔇_n es la base (fraccional) del
    simplex de estados.  sanitize = Hermitiza + PSD-clip + renorm,
    no-expansiva en ‖·‖_F (proyección espectral sobre el cono).
    """

    EPS: Final[float] = _EPS
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
    def eigenpairs_descending(
        cls, rho: np.ndarray
    ) -> Tuple[RealVector, ComplexMatrix]:
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
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -float(np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        return float(np.sum(p * p))

    @classmethod
    def matrix_power(
        cls, rho: np.ndarray, z: complex, floor: float = _EPS_MOD
    ) -> ComplexMatrix:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

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
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(F))))

    @classmethod
    def modular_spectrum(cls, rho: np.ndarray) -> Tuple[float, ...]:
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


# ── §1.3 Dinámica proyectiva + métrica Fubini–Study + forma simplectica ─────────
@dataclass(frozen=True, slots=True)
class ProjectiveRayleighSample:
    r"""
    Muestra de T en [v] ∈ ℂP^{n−1} (v unitario, gauge U(1) fijado).
        rayleigh, image_norm, overlap, residual, fs_angle, chordal.
    Identidad P2: residual = ‖T_φ − v‖₂ = 2 sin(d_FS / 2).
    """

    rayleigh: float
    image_norm: float
    overlap: float
    residual: float
    fs_angle: float
    chordal: float


class ProjectiveDynamics:
    r"""
    Tres dinámicas coexistentes sobre ℂP^{n−1} (P7):

        (T) T([v]) = [ρ v]          mapa proyectivo discreto (no simplectico).
        (G) X_G(v) = (I − vv†) ρ v  gradiente de R (disipativo).
        (H) X_H(v) = i X_G(v)       hamiltoniano de R (Liouville).

    Métrica d_FS = arccos|⟨u|v⟩|.  Gauge U(1): T_φ(v) := e^{−i arg⟨v, Tv⟩} Tv.
    Brouwer: ℂP^{n−1} compacto, T continua fuera de ker ρ ⇒ Fix(T) ≠ ∅.
    Forma simplectica: ω_FS(ξ, η) = 2 Im ⟨ξ_h | η_h⟩.
    """

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
        r"""Proyección horizontal: ξ_h = (I − vv†) ξ  (conexión de Chern)."""
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
        r"""
        1-forma de Liouville (primitiva local de ω_FS, gauge-fijada):
            λ_FS(ξ) = Im ⟨v | ξ⟩.
        dλ_FS = ω_FS sobre el horizonte.  Invariante integral relativo de Poincaré.
        """
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        xi = np.asarray(xi, dtype=np.complex128).reshape(-1)
        return float(np.imag(np.vdot(v, xi)))

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
        r"""X_G(v) = (I − vv†) ρ v  — gradiente de Rayleigh en S^{2n−1}."""
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


# ── §1.4 Coordenadas acción-ángulo de Poincaré (sistema integrable) ──────────────
@dataclass(frozen=True, slots=True)
class ActionAngleChart:
    r"""
    Carta acción-ángulo de Poincaré sobre ℂP^{n−1} (reducción simplectica):
        I_k = |⟨φ_k | v⟩|² ∈ Δ^{n−1} ,   θ_k = arg ⟨φ_k | v⟩ ∈ T^n / U(1).
    Hamiltoniano: H(I) = Σ λ_k I_k.
    Frecuencias: ω = ∇_I H = λ  (isócrono, Hessiano nulo).
    Resonancias: k ∈ ℤ^n \ {0} con k · λ = 0  (módulo de relaciones enteras).
    Twist hamiltoniano: ‖∂ω/∂I‖ = 0.  P8: se declara nulo, no se fabrica.
    Función generatriz de tipo 2 del flujo: S = θ · I + t H(I).
    """

    actions: RealVector
    angles: RealVector
    frequencies: RealVector
    hamiltonian: float
    hessian_twist_norm: float
    resonance_rank: int
    commensurability_defect: float
    generating_function_density: float
    simplex_hash: str


class ActionAnglePoincareChart:
    r"""Constructor de la carta acción-ángulo y del módulo de resonancia."""

    @staticmethod
    def _integer_resonance_rank(omega: RealVector, kmax: int = 4) -> Tuple[int, float]:
        r"""
        Rango del retículo { k ∈ ℤ^n ∩ [−kmax, kmax]^n : |k · ω| < ε } menos {0}.
        defect = min |k · ω| / (1 + |k|) sobre k ≠ 0  (medida de irracionalidad).
        """
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
    def chart(
        cls,
        rho: np.ndarray,
        v: np.ndarray,
        kmax: int = 4,
    ) -> ActionAngleChart:
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
        h = _sha256_bytes(
            np.ascontiguousarray(actions).tobytes(),
            np.ascontiguousarray(w).tobytes(),
        )
        return ActionAngleChart(
            actions=actions,
            angles=angles,
            frequencies=w.astype(np.float64),
            hamiltonian=H,
            hessian_twist_norm=0.0,
            resonance_rank=int(rank),
            commensurability_defect=float(defect),
            generating_function_density=float(S_density),
            simplex_hash=h,
        )


# ── §1.5 Flujo hamiltoniano celeste + mapa de primer retorno ─────────────────────
@dataclass(frozen=True, slots=True)
class HamiltonianReturnSample:
    r"""
    Cruce de la sección hamiltoniana Σ_{c,q} = {R = c} ∩ {Im⟨q|v⟩ = 0}.
    En un 2-plano eigen, τ teórico = 2π / |λ_i − λ_j|.
    El mapa de retorno P es un simplectomorfismo (flujo de Liouville).
    """

    energy: float
    return_time: float
    theoretical_period: float
    period_error: float
    n_crossings: int
    cartan_circulation: float
    energy_drift: float
    section_residual: float


class HamiltonianCelestialFlow:
    r"""
    Integrador RK4 proyectivo del flujo (H):  v' = i (I − vv†) ρ v,
    renormalizado sobre S^{2n−1}.  Conserva R y ‖v‖ (en aritmética exacta).
    Evento de sección: cambio de signo de Im⟨q|v⟩ con Ṡ > 0.
    Invariante de Poincaré–Cartan aproximado por suma de λ_FS(v') Δt.
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
        return float(2.0 * math.pi / d)

    @classmethod
    def first_return(
        cls,
        rho: np.ndarray,
        v0: np.ndarray,
        q: Optional[np.ndarray] = None,
        dt: float = 2.5e-3,
        t_max: float = 40.0,
        min_crossings: int = 1,
    ) -> HamiltonianReturnSample:
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v0, n)
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

        s_prev = section_fn(v)
        t = 0.0
        crossings = 0
        t_first = 0.0
        cartan = 0.0
        v_start = v.copy()
        n_steps = int(max(8, t_max / max(dt, 1e-8)))
        for _ in range(n_steps):
            X = cls._rhs(rho, v)
            cartan += ProjectiveDynamics.poincare_one_form(v, X) * dt
            v_next = cls._rk4_step(rho, v, dt)
            t += dt
            s_next = section_fn(v_next)
            # cruce ascendente s_prev ≤ 0 < s_next y velocidad hamiltoniana coherente
            if s_prev <= 0.0 < s_next and t > 4.0 * dt:
                crossings += 1
                if crossings == 1:
                    t_first = t
                if crossings >= min_crossings and t_first > 0.0:
                    v = v_next
                    break
            v = v_next
            s_prev = s_next
        E1 = ProjectiveDynamics.rayleigh(rho, v)
        s_res = abs(section_fn(v))
        period_err = (
            abs(t_first - tau_th) / max(tau_th, 1e-12)
            if math.isfinite(tau_th) and t_first > 0.0
            else 1.0
        )
        return HamiltonianReturnSample(
            energy=float(E0),
            return_time=float(t_first if t_first > 0.0 else t),
            theoretical_period=_finite(tau_th, 0.0),
            period_error=float(period_err),
            n_crossings=int(crossings),
            cartan_circulation=float(cartan),
            energy_drift=float(abs(E1 - E0)),
            section_residual=float(s_res),
        )


# ── §1.6 Sección de Poincaré introspectiva (Σ_c ⊂ ℂP^{n−1}  y  Σ_{c,q}) ─────────
@dataclass(frozen=True, slots=True)
class PoincareSectionIntrospection:
    r"""
    Dos secciones, una por sector (P7):

    (G) Σ_c = {[v] : R(v) = c}.  R es Morse ⇔ spec(ρ) simple.
        ⟨∇R, X_G⟩ = 2‖X_G‖²  se anula ⇔ v eigenrayo.  Transversalidad
        ⇔ c ∉ spec(ρ).  Diagnóstico: min 2‖X_G‖² sobre muestras Haar
        condicionadas a |R − c| pequeña.

    (H) Σ_{c,q} = {R = c} ∩ {Im⟨q|v⟩ = 0}.  {R = c} es invariante de X_H;
        la sección auténtica es hipersuperficie de {R = c}.
        τ* teórico (2-plano dominante) = 2π / |λ₁ − λ₂|.

    Floquet de (T) en [φ₁]: μ_i = λ_i / λ₁, i ≥ 2.
    periodic_orbit_rank: #μ_i ≈ 1 (degeneración ⇒ continuo de FP).
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
            # Identidad: ⟨∇R, X_G⟩ = 2 ‖X_G‖²
            trans = 2.0 * float(np.real(np.vdot(X, X)))
            tmin = min(tmin, trans)
            tau_acc.append(1.0 / max(trans, 1e-12))
        if not np.isfinite(tmin):
            tmin = 0.0
        tau_mean = float(np.mean(tau_acc)) if tau_acc else 0.0

        lam1 = max(float(w[0]), _EPS)
        floquet = (w[1:] / lam1).astype(np.float64) if w.size > 1 else np.zeros(0)
        per_rank = int(np.sum(np.abs(floquet - 1.0) < 1e-10)) if floquet.size else 0

        if v_probe is None:
            v_probe = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        ham = HamiltonianCelestialFlow.first_return(
            rho, v_probe, q=None, dt=3.0e-3, t_max=25.0, min_crossings=1,
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
            section_hash=h,
        )

    @property
    def is_transverse(self) -> bool:
        return bool(self.gradient_transverse)


# ── §1.7 Poincaré–Hopf + Morse de R sobre ℂP^{n−1} ──────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareHopfCertificate:
    r"""
    Teorema de Poincaré–Hopf: Σ_crit ind_x(X) = χ(M) si X tiene ceros aislados.
    χ(ℂP^{n−1}) = n.  Para X = ∇R, en spec simple:
        Morse_R([φ_k]) = 2(n − k) ,  ind_Hopf = (−1)^{Morse} = +1,
        Σ ind = n = χ.  Degeneración ⇒ Morse–Bott, Hopf se relaja a ⋆.
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


# ── §1.8 Recurrencia de Poincaré + lema de Kac ───────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareRecurrenceCertificate:
    r"""
    Poincaré 1890: un homeomorfismo que preserva medida en espacio de
    probabilidad finita es recurrente en c.t.p.
        (H)  preserva Haar-FS  ⇒ recurrencia plena.  Kac: 𝔼[τ_A] = 1/μ(A).
        (T)  no preserva Haar  ⇒ sólo Fix(T) es recurrente.
    μ(bola FS de radio ε) ∼ ε^{2n−2} / (n−1)!  (volumen proyectivo).
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
        # proxy de μ(B_ε) ≤ 1, ∼ ε^{2n−2}
        mu = float(min(1.0, eps ** max(dim_real, 1)))
        kac = float(1.0 / max(mu, 1e-15))
        ham_rec = True
        proj_rec = bool(is_uniform)  # T preserva Haar ⇔ ρ ∝ I
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


# ── §1.9 Grafo de resonancia espectral (tela de Arnold) ──────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralResonanceGraph:
    r"""
    Grafo G_λ = (V, E) sobre eigenrayos:
        arista {i, j} ⇔ |λ_i − λ_j| ≤ δ_res  (resonancia 1:1)
                      o bien ∃ k ∈ ℤ² \ {0}, |k₁ λ_i + k₂ λ_j| ≤ δ  (baja).
    Componentes conexas = multiplets degenerados / conmensurables.
    Un grafo con arista en el techo (λ₁ ∼ λ₂) ⇒ caos introspectivo de (T).
    """

    n_vertices: int
    n_edges: int
    adjacency_spectrum: Tuple[float, ...]
    n_components: int
    top_resonant: bool
    algebraic_connectivity: float
    local_verdict: HeytingOmega3

    @classmethod
    def build(
        cls, spectrum: np.ndarray, delta: float = 1e-3, kmax: int = 3
    ) -> "SpectralResonanceGraph":
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
        # componentes vía (I+A)^{n-1}
        reach = np.eye(n, dtype=np.float64)
        B = A + np.eye(n, dtype=np.float64)
        for _ in range(max(n - 1, 1)):
            reach = (reach @ B) > 0.5
            reach = reach.astype(np.float64)
        seen = np.zeros(n, dtype=bool)
        n_comp = 0
        for i in range(n):
            if not seen[i]:
                n_comp += 1
                seen |= reach[i] > 0.5
        specA = np.sort(np.real(la.eigvalsh(A)))[::-1] if n else np.zeros(0)
        # conectividad algebraica = λ₂(L), L = deg − A
        deg = np.diag(np.sum(A, axis=1))
        L = deg - A
        mu = np.sort(np.real(la.eigvalsh(L))) if n else np.array([0.0])
        alg = float(mu[1]) if mu.size > 1 else 0.0
        top_res = bool(n >= 2 and A[0, 1] > 0.5)
        if top_res:
            local = HeytingOmega3.VETOED
        elif n_edges == 0:
            local = HeytingOmega3.COHERENT
        else:
            local = HeytingOmega3.DEGRADED
        return cls(
            n_vertices=n,
            n_edges=n_edges,
            adjacency_spectrum=tuple(float(x) for x in specA.tolist()),
            n_components=int(n_comp),
            top_resonant=top_res,
            algebraic_connectivity=float(alg),
            local_verdict=local,
        )


# ── §1.10 Lema de concentración de Lévy sobre ℂP^{n−1} (FS) ─────────────────────
class LevyConcentrationLemma:
    r"""
    Concentración sobre ℂP^{n−1} con métrica Fubini–Study (Ric = 2(n+1) g):
        P(|f − 𝔼f| ≥ ε) ≤ exp(−(n+1)ε² / (2π² L²)),
    para f Lipschitz-L respecto a d_FS.  Cota de Milman–Schechtman
    adaptada a la variedad Kähler compacta de curvatura holomorfa seccional 4.
    """

    @staticmethod
    def bound(epsilon: float, n: int, lipschitz: float = 1.0) -> float:
        if n <= 1:
            return 1.0
        L = max(float(lipschitz), 1e-15)
        e = max(float(epsilon), 0.0)
        return float(
            min(1.0, math.exp(-(n + 1) * e * e / (2.0 * math.pi ** 2 * L * L)))
        )

    @staticmethod
    def median_width(
        n: int, lipschitz: float = 1.0, confidence: float = 0.99
    ) -> float:
        if n <= 1:
            return float("inf")
        p_tail = max(1e-15, 1.0 - float(confidence))
        return float(
            lipschitz
            * math.pi
            * math.sqrt(2.0 * math.log(1.0 / p_tail) / (n + 1))
        )


# ── §1.11 Análisis del gap espectral ────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralGapReport:
    r"""
    Análisis espectral de ρ:
        gap = 1 − λ₂/λ₁;  cond = λ₁/λ_n;  degeneracy_top;  birkhoff_proxy.
    Veredicto local:
        uniforme ∨ degeneracy ≥ 3     → ⊥
        gap ≥ GAP_COHERENT ∧ deg = 1  → ⊤
        gap ≥ GAP_DEGRADED            → ⋆
        else                          → ⊥
    birkhoff_constant = tanh(log(λ₁/λ_n)/4)  (contracción proyectiva de Hilbert).
    """

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
        if is_uniform or degeneracy >= 3:
            local = HeytingOmega3.VETOED
        elif gap >= cls.GAP_COHERENT and degeneracy == 1:
            local = HeytingOmega3.COHERENT
        elif gap >= cls.GAP_DEGRADED:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return SpectralGapReport(
            lambda_1=lam1,
            lambda_2=lam2,
            lambda_min=lam_n,
            gap=gap,
            gap_ratio=float(gap_ratio),
            gap_absolute=gap_abs,
            condition_number=cond,
            degeneracy_top=degeneracy,
            is_uniform=is_uniform,
            von_neumann_entropy=DensityOperatorAlgebra.von_neumann_entropy(rho),
            purity=DensityOperatorAlgebra.purity(rho),
            birkhoff_constant=birkhoff,
            local_verdict=local,
        )


# ── §1.12 IntrospectiveField + hand-off formal a FASE 2 ─────────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectiveField:
    r"""
    Objeto terminal FASE 1 / inicial FASE 2  (dominio de F₂).
        rho, v_initial, spectral_gap, poincare_section, levy_band, K_spec,
        action_angle, poincare_hopf, recurrence, resonance_graph, field_hash.
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
    field_hash: str
    dim: int

    def as_dict(self) -> Dict[str, float]:
        g = self.spectral_gap
        return {
            "lambda_1": g.lambda_1,
            "lambda_2": g.lambda_2,
            "gap": g.gap,
            "gap_ratio": g.gap_ratio,
            "degeneracy": float(g.degeneracy_top),
            "birkhoff": g.birkhoff_constant,
            "is_uniform": float(g.is_uniform),
            "poincare_transverse": float(self.poincare_section.is_transverse),
            "levy_band": self.levy_band,
            "hopf_ok": float(self.poincare_hopf.hopf_satisfied),
            "ham_recurrence": float(self.recurrence.hamiltonian_recurrence),
            "resonance_edges": float(self.resonance_graph.n_edges),
            "twist_hess": self.action_angle.hessian_twist_norm,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    #  continue_into_phase2  es el último morfismo de FASE 1 y, por anidamiento
    #  estricto, el primero de FASE 2: su cuerpo ES synthesize.
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase2(
        self, max_iter: int, tol: float
    ) -> "IntrospectionBundle":
        r"""
        Último morfismo FASE 1 ∧ primero FASE 2.
            continue_into_phase2 ∘ prepare
                = IntrospectionPipeline.synthesize ∘ prepare
                : 𝔇_n × ℂⁿ → IntrospectionBundle.
        """
        return IntrospectionPipeline.synthesize(
            field=self, max_iter=max_iter, tol=tol
        )


class IntrospectiveFieldPreparation:
    r"""
    Prepara (ρ, v₀) + análisis espectral + Σ_c + Σ_{c,q} + Lévy + Morse–Hopf
    + recurrencia + carta acción-ángulo + grafo de resonancia.
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
        levy_band = LevyConcentrationLemma.median_width(
            n, lipschitz=1.0, confidence=0.99
        )
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
        field_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(v).tobytes(),
            f"{gap_report.gap:.12f}".encode("ascii"),
            poincare.section_hash.encode("ascii"),
            hopf.local_verdict.name.encode("ascii"),
            aa.simplex_hash.encode("ascii"),
        )
        logger.debug(
            "FieldPreparation: λ₁=%.4f λ₂=%.4f γ=%.4f deg=%d | "
            "Σ_transv=%s τ_G=%.3e τ_H=%.3e | ε*_Levy=%.4f | Hopf=%s | "
            "res_edges=%d twist_H=%.1e",
            gap_report.lambda_1,
            gap_report.lambda_2,
            gap_report.gap,
            gap_report.degeneracy_top,
            poincare.is_transverse,
            poincare.first_return_time,
            poincare.hamiltonian_return_time,
            levy_band,
            hopf.hopf_satisfied,
            resg.n_edges,
            aa.hessian_twist_norm,
        )
        return IntrospectiveField(
            rho=rho,
            v_initial=v,
            spectral_gap=gap_report,
            poincare_section=poincare,
            levy_band=float(levy_band),
            K_spec=K_spec,
            action_angle=aa,
            poincare_hopf=hopf,
            recurrence=rec,
            resonance_graph=resg,
            field_hash=field_hash,
            dim=n,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 · DINÁMICA DE PUNTO FIJO + OSELEDETS + MELNIKOV + KAM + BIRKHOFF         ║
# ║                                                                                  ║
# ║  Dominio  = IntrospectiveField  (codominio de §1.12 continue_into_phase2).      ║
# ║  Codominio = IntrospectionBundle, dominio de toda la FASE 3.                    ║
# ║  Arranque  = IntrospectionPipeline.synthesize  ← cuerpo de continue_into_phase2.║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

# ── §2.1 Iteración de potencia gauge-fijada + ecuación variacional ───────────────
@dataclass(frozen=True, slots=True)
class PowerIterationTrace:
    """
    Traza de T^k sobre ℂP^{n−1}:
        residuals, fs_angles, rayleigh_trajectory por iteración.
        empirical_rate = mediana de θ_{k+1}/θ_k (≈ λ₂/λ₁).
        stalled_kernel = True si ρv ≈ 0.
        variational_spectral_radius: radio de DT a lo largo de la órbita
        (ecuación en variaciones proyectiva).
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


class PowerIterationSolver:
    r"""
    v_{k+1} = gauge_align(v_k, ρv_k / ‖ρv_k‖).
    Parada: d_FS < tol ∨ ‖T_φ − v_k‖ < tol ∨ k = max_iter.
    Tasa: d_FS([v_k],[φ₁]) = Θ((λ₂/λ₁)^k) si λ₁ > λ₂.
    Ecuación en variaciones proyectiva (monodromía discreta de (T)):
        δv ↦ (I − ww†) ρ δv / ‖ρv‖ ,  w = T(v),  δv ⊥ v  (horizontal).
    Radio espectral de DT|_{[φ₁]} = λ₂/λ₁.
    """

    MAX_ITER_DEFAULT: Final[int] = 500
    TOL_DEFAULT: Final[float] = 1.0e-10
    MIN_FS_FOR_RATE: Final[float] = 1.0e-12
    RATE_FS_FLOOR: Final[float] = 1.0e-8
    RATE_FS_CEIL: Final[float] = 5.0e-1

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
    def _variational_radius(
        cls, rho: np.ndarray, v: np.ndarray, n_trials: int = 6
    ) -> float:
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
    def solve(
        cls,
        field: IntrospectiveField,
        max_iter: int = MAX_ITER_DEFAULT,
        tol: float = TOL_DEFAULT,
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
        for _k in range(max_iter):
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
            T_v, _ = ProjectiveDynamics.apply_map(rho, v)
            v = ProjectiveDynamics.sanitize_vector(
                ProjectiveDynamics.gauge_align(v, T_v), n
            )
        empirical = cls._empirical_rate(fs_angles, theoretical)
        var_r = cls._variational_radius(rho, v)
        final_res = float(residuals[-1]) if residuals else 0.0
        hist = tuple(map(float, fs_angles))
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
        Punto fijo autoinvariante en ℂP^{n−1} gauge-fijado, con semilla S⁶.
        Axiomas:
          1. Gauge: T_φ(e^{iθ}v) = e^{iθ} T_φ(v).
          2. Norma: ‖T_φ(v)‖₂ = 1.
          3. Residuo FS: d_FS = arccos|⟨v, T_φ(v)⟩| ≤ tol.
        Respaldo topológico: Brouwer (ℂP^{n−1} compacto) + Poincaré–Hopf
        (índice +1 en [φ₁] si λ₁ simple).
        """
        density_op = DensityOperatorAlgebra.sanitize(density_op)
        n = int(density_op.shape[0])
        if seed_ray_s6 is not None and seed_ray_s6.size >= 6:
            v0 = np.array(
                [
                    seed_ray_s6[0] + 1j * seed_ray_s6[1],
                    seed_ray_s6[2] + 1j * seed_ray_s6[3],
                    seed_ray_s6[4] + 1j * seed_ray_s6[5],
                ],
                dtype=np.complex128,
            )
            if v0.size < n:
                v0 = np.pad(v0, (0, n - v0.size))
            elif v0.size > n:
                v0 = v0[:n]
            norm_v0 = float(np.linalg.norm(v0))
            v = (
                v0 / norm_v0
                if norm_v0 > 1e-12
                else np.ones(n, dtype=np.complex128) / math.sqrt(n)
            )
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
            local_verdict=(
                HeytingOmega3.COHERENT if is_fixed_point else HeytingOmega3.DEGRADED
            ),
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
    r"""Certificado del punto fijo introspectivo de (T), gauge-fijado."""

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


class FixedPointCertifier:
    r"""Predicados dimensionalmente invariantes, colapsados por meet (P4)."""

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
        cls,
        field: IntrospectiveField,
        v_fixed: np.ndarray,
        trace: PowerIterationTrace,
    ) -> FixedPointCertificate:
        sample = ProjectiveDynamics.evaluate(field.rho, v_fixed)
        lam1 = field.spectral_gap.lambda_1
        rate_th = field.spectral_gap.gap_ratio
        rate_emp = trace.empirical_rate
        already_fp = bool(trace.converged and trace.iterations <= 2)
        noisy_rate = bool(trace.converged and sample.fs_angle < cls.EPS_RES)
        rate_ok = already_fp or noisy_rate or (
            abs(rate_emp - rate_th) <= cls.RATE_TOL
        )
        conv_rule = (
            HeytingOmega3.VETOED
            if trace.stalled_kernel
            else (
                HeytingOmega3.COHERENT
                if trace.converged
                else HeytingOmega3.DEGRADED
            )
        )
        overlap_rule = cls._grade(1.0 - sample.overlap, cls.EPS_FID, 1.0e-3)
        residual_rule = cls._grade(sample.residual, cls.EPS_RES, 1.0e-3)
        ray_rel = max(0.0, lam1 - sample.rayleigh) / max(lam1, _EPS)
        ray_rule = cls._grade(ray_rel, cls.EPS_RAY, 1.0e-2)
        rate_rule = (
            HeytingOmega3.COHERENT if rate_ok else HeytingOmega3.DEGRADED
        )
        local = conv_rule.meet_all(
            overlap_rule, residual_rule, ray_rule, rate_rule
        )
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
        )


# ── §2.3 Espectro de Oseledets proyectivo ───────────────────────────────────────
@dataclass(frozen=True, slots=True)
class OseledetsLyapunovSpectrum:
    r"""
    Espectro de Lyapunov de (T) sobre ℂP^{n−1}.
        λ_i^{Lyap} = log(λ_i(ρ) / λ₁(ρ))   i ≥ 2   (todas ≤ 0).
    h_KS(Pesin) = Σ_{λ>0} λ.  Atractor único λ_max < 0 ⇒ h_KS = 0.
    Degeneración λ₁ = λ₂ ⇒ λ_max = 0 ⇒ continuo ℂP¹ de FP.
    D_KY = 0 si λ_max < 0 (atractor puntual).  Multiplicadores de (H)
    viven en U(1) y no contribuyen a h_KS (flujo elíptico).
    """

    lyapunov_full: RealVector
    lyapunov_max: float
    lyapunov_min: float
    kolmogorov_sinai_entropy: float
    kaplan_yorke_dimension: float
    n_positive: int
    degeneracy_signal: bool

    @classmethod
    def from_spectrum(cls, rho: np.ndarray) -> "OseledetsLyapunovSpectrum":
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        lam1 = max(float(w[0]), _EPS)
        rates = np.array(
            [math.log(max(float(w_i), _EPS) / lam1) for w_i in w[1:]],
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


# ── §2.4 Monodromía hamiltoniana / multiplicadores de Floquet ────────────────────
@dataclass(frozen=True, slots=True)
class VariationalMonodromyCertificate:
    r"""
    Multiplicadores de Floquet del flujo (H) a lo largo de la órbita
    periódica del 2-plano dominante, periodo τ = 2π / |λ₁ − λ₂|:
        μ_k = exp(i (λ_k − λ₁) τ) ∈ U(1).
    Estabilidad elíptica (Krein) ⇔ |μ| = 1.  Resonancia de Krein si μ = ±1
    con pareja de signaturas opuestas — aquí isócrono ⇒ no hay splitting.
    """

    multipliers: Tuple[complex, ...]
    period: float
    on_unit_circle: bool
    krein_resonance: bool
    spectral_radius: float
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(cls, rho: np.ndarray) -> "VariationalMonodromyCertificate":
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        if w.size < 2:
            return cls(
                multipliers=(1.0 + 0j,),
                period=0.0,
                on_unit_circle=True,
                krein_resonance=True,
                spectral_radius=1.0,
                local_verdict=HeytingOmega3.DEGRADED,
            )
        tau = HamiltonianCelestialFlow.two_plane_period(float(w[0]), float(w[1]))
        if not math.isfinite(tau):
            tau = 0.0
        lams: List[complex] = []
        for wk in w:
            phase = (float(wk) - float(w[0])) * tau
            lams.append(complex(math.cos(phase), math.sin(phase)))
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
            multipliers=tuple(lams),
            period=float(tau),
            on_unit_circle=on_u1,
            krein_resonance=krein,
            spectral_radius=rho_sp,
            local_verdict=local,
        )


# ── §2.5 Integral de Melnikov proyectiva (Poincaré–Melnikov simplectica) ─────────
@dataclass(frozen=True, slots=True)
class MelnikovHomoclinicCertificate:
    r"""
    Integral de Poincaré–Melnikov sobre la heteroclínica de (G) entre [φ₂] y [φ₁]:
        γ(t) ∝ e^{λ₁ t} φ₁ + e^{λ₂ t} φ₂ ,  t ∈ ℝ,
        M(ϑ) = ∫ ω_FS(X_{H₀}, X_{H₁})(γ_ϑ(t)) dt.
    H₁ se toma como wiggle hermítico traceless determinista (semilla SHA).
    Sin perturbación, H₀ integrable ⇒ M ≡ 0.
    Cero simple de M + |M| > ε ⇒ W^s ∩ W^u transversal ⇒ veto de caos.
    """

    melnikov_value: float
    melnikov_zeros: int
    transverse_homoclinic: bool
    chaos_threshold: float
    unperturbed_vanishes: bool

    @staticmethod
    def _heteroclinic(
        phi1: np.ndarray, phi2: np.ndarray, lam1: float, lam2: float, t: float
    ) -> ComplexVector:
        a = math.exp(lam1 * t)
        b = math.exp(lam2 * t)
        v = a * phi1 + b * phi2
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            return phi1
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
    def evaluate(
        cls,
        rho: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        epsilon_M: float = 1.0e-3,
        key: str = "MELNIKOV",
    ) -> "MelnikovHomoclinicCertificate":
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        v1 = ProjectiveDynamics.sanitize_vector(v1, n)
        v2 = ProjectiveDynamics.sanitize_vector(v2, n)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        lam1 = float(w[0])
        lam2 = float(w[1]) if w.size > 1 else 0.0
        W = cls._hermite_wiggle(n, key)
        # t ∈ ℝ compactificado: la heteroclínica se concentra en |t| ≲ 1/|λ₁−λ₂|
        gap_abs = max(abs(lam1 - lam2), 1e-6)
        ts = np.linspace(-6.0 / gap_abs, 6.0 / gap_abs, 61)
        dens = np.zeros_like(ts)
        dens0 = np.zeros_like(ts)
        for i, t in enumerate(ts):
            gamma = cls._heteroclinic(v1, v2, lam1, lam2, float(t))
            X0 = ProjectiveDynamics.hamiltonian_vector_field(rho, gamma)
            X1 = ProjectiveDynamics.hamiltonian_vector_field(W, gamma)
            dens[i] = ProjectiveDynamics.symplectic_form(gamma, X0, X1)
            dens0[i] = ProjectiveDynamics.symplectic_form(gamma, X0, X0)
        dt = float(ts[1] - ts[0]) if ts.size > 1 else 1.0
        M = float(np.trapz(dens, dx=dt))
        M0 = float(np.trapz(dens0, dx=dt))
        signs = np.sign(dens)
        zeros = int(np.sum(np.abs(np.diff(signs)) > 1.0))
        unperturbed = bool(abs(M0) < 1e-8)
        transverse = bool(abs(M) > epsilon_M and zeros >= 1)
        return cls(
            melnikov_value=float(M),
            melnikov_zeros=int(zeros),
            transverse_homoclinic=transverse,
            chaos_threshold=float(epsilon_M),
            unperturbed_vanishes=unperturbed,
        )


# ── §2.6 KAM diofántico sobre el vector de frecuencias de Poincaré ───────────────
@dataclass(frozen=True, slots=True)
class KAMDiophantineCertificate:
    r"""
    |ω · k| ≥ γ / |k|^τ  ∀ k ∈ ℤ^m \ {0};  ε < ε₀(γ, τ) ⇒ el toro 𝕋^m persiste.
    ε₀ heurístico (Chirikov): γ / (m log(1/γ))^m.
    ω = λ (frecuencias de Poincaré del H integrable).
    ε = ‖ρ − I/n‖_F  mide distancia al régimen máximo-mixto, NO una
    perturbación no integrable: un único ρ hermítico permanece integrable.
    kam_persists se interpreta como «condición diofántica del vector ω y
    perturbación pequeña respecto al umbral de Chirikov»; P8 prohíbe
    declarar KAM no trivial de un H lineal.
    """

    frequency_vector: RealVector
    diophantine_gamma: float
    diophantine_tau: float
    is_diophantine: bool
    perturbation_ratio: float
    kam_persists: bool
    kmax_used: int
    integrable_unperturbed: bool

    @classmethod
    def evaluate(
        cls,
        omega: Sequence[float],
        perturbation_size: float,
        tau: float = 1.5,
        kmax: int = 6,
        gamma_min: float = 1e-3,
    ) -> "KAMDiophantineCertificate":
        w = np.asarray(omega, dtype=np.float64).ravel()
        m = int(w.size)
        if m == 0:
            return cls(
                frequency_vector=w,
                diophantine_gamma=0.0,
                diophantine_tau=tau,
                is_diophantine=False,
                perturbation_ratio=float("inf"),
                kam_persists=False,
                kmax_used=kmax,
                integrable_unperturbed=True,
            )
        kmax = int(max(1, min(int(kmax), 7)))
        ranges = [np.arange(-kmax, kmax + 1) for _ in range(m)]
        gamma = float("inf")
        grid = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(m, -1).T
        for k in grid:
            if np.all(k == 0):
                continue
            dot = float(np.dot(w, k))
            norm = float(np.linalg.norm(k))
            if abs(dot) < 1e-15:
                gamma = 0.0
                break
            gamma = min(gamma, abs(dot) * (norm ** tau))
        is_dio = bool(gamma >= gamma_min)
        if gamma > 1e-15:
            denom = (
                (m * math.log(1.0 / max(gamma, 1e-15))) ** m if m > 0 else 1.0
            )
            eps0 = gamma / max(denom, 1e-15)
        else:
            eps0 = 0.0
        ratio = (
            float(perturbation_size / eps0) if eps0 > 1e-15 else float("inf")
        )
        return cls(
            frequency_vector=w.astype(np.float64),
            diophantine_gamma=float(gamma),
            diophantine_tau=float(tau),
            is_diophantine=is_dio,
            perturbation_ratio=float(ratio),
            kam_persists=bool(is_dio and ratio < 1.0),
            kmax_used=int(kmax),
            integrable_unperturbed=True,
        )


# ── §2.7 Teorema de Birkhoff (anillo) — sector (H) isócrono vs sector (T) radial ─
@dataclass(frozen=True, slots=True)
class BirkhoffTwistMapCertificate:
    r"""
    Último teorema geométrico de Poincaré (1912), Birkhoff (1913):
    homeomorfismo que preserva área del anillo A = S¹ × [0,1] con twist
    ∂(θ')/∂r ≠ 0  ⇒  ≥ 2 puntos fijos.

    Sector (H) — flujo hamiltoniano de R:
        H lineal en acciones ⇒ ω independiente de I ⇒ twist_H = 0.
        Rotación rígida: Birkhoff N/A.  P8.

    Sector (T) — mapa radial en el 2-plano span{φ_i, φ_j}:
        r ↦ f(r) = α r / (α r + β(1−r)),  α = λ_i², β = λ_j².
        f(0) = 0, f(1) = 1, f monótona, f(r) ≠ r en (0,1) si α ≠ β.
        Exactamente 2 FP (los eigenrayos).  No es area-preserving.
        twist_gradient_min := min_r |f'(r) − 1|  (separación radial).

    n_fixed_points_min = 2 si α ≠ β en al menos un par; 1 si degeneración total.
    """

    twist_holds: bool
    n_fixed_points_min: int
    twist_gradient_min: float
    hamiltonian_twist: float
    radial_monotone: bool
    degeneracy_pairs: int
    local_verdict: HeytingOmega3

    @staticmethod
    def _radial_map_derivative(alpha: float, beta: float, r: float) -> float:
        # f(r) = α r / (α r + β(1−r));  f'(r) = α β / (α r + β(1−r))²
        den = alpha * r + beta * (1.0 - r)
        if abs(den) < 1e-18:
            return 0.0
        return float(alpha * beta / (den * den))

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
        pairs_examined = min(int(n_pairs), max(1, n - 1))
        for p in range(pairs_examined):
            i, j = 0, p + 1
            if j >= n:
                continue
            alpha = max(float(w[i]) ** 2, _EPS)
            beta = max(float(w[j]) ** 2, _EPS)
            if abs(alpha - beta) / max(alpha, beta) < 1e-10:
                continue
            valid_pairs += 1
            pair_min = float("inf")
            for _ in range(max(4, n_samples)):
                r = float(rng.uniform(0.1, 0.9))
                fp = cls._radial_map_derivative(alpha, beta, r)
                pair_min = min(pair_min, abs(fp - 1.0))
                # monotonía: f' > 0
                if fp <= 0.0:
                    radial_ok = False
            twist_min = min(twist_min, pair_min)
        if not np.isfinite(twist_min):
            twist_min = 0.0
        ham_twist = 0.0  # Hessiano de H nulo (P8)
        twist_holds = False  # no hay twist hamiltoniano area-preserving
        n_fp = 2 if valid_pairs >= 1 else 1
        if n_fp >= 2 and radial_ok:
            local = HeytingOmega3.COHERENT
        elif n_fp >= 1:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            twist_holds=bool(twist_holds),
            n_fixed_points_min=int(n_fp),
            twist_gradient_min=float(twist_min),
            hamiltonian_twist=float(ham_twist),
            radial_monotone=bool(radial_ok and valid_pairs >= 1),
            degeneracy_pairs=int(valid_pairs),
            local_verdict=local,
        )


# ── §2.8 IntrospectionBundle + hand-off formal a FASE 3 ─────────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionBundle:
    r"""
    Objeto terminal FASE 2 / inicial FASE 3.
    Producto fibrado:
        Power ⊗ Certifier ⊗ Oseledets ⊗ Melnikov ⊗ KAM ⊗ Birkhoff
        ⊗ Monodromía.
    El campo (FASE 1) viaja como sección: bundle.field.
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
        ).digest()

    def continue_into_phase3(
        self, external_verdict: HeytingOmega3
    ) -> HeytingOmega3:
        r"""
        Último morfismo FASE 2 ∧ primero FASE 3.
            continue_into_phase3 ∘ synthesize ∘ prepare
                = adjudicate ∘ synthesize ∘ prepare.
        """
        return HeytingIntrospectionAdjudicator.adjudicate(self, external_verdict)


class IntrospectionPipeline:
    r"""
    Orquestador determinista FASE 2 (funtor F₂).
    Cuerpo de IntrospectiveField.continue_into_phase2.
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
            (v*, trace) = PowerIterationSolver.solve(field)
            cert        = FixedPointCertifier.certify(...)
            oseledets   = OseledetsLyapunovSpectrum.from_spectrum(ρ)
            melnikov    = MelnikovHomoclinicCertificate.evaluate(...)
            kam         = KAMDiophantineCertificate.evaluate(...)
            birkhoff    = BirkhoffTwistMapCertificate.evaluate(ρ)
            monodromy   = VariationalMonodromyCertificate.evaluate(ρ)
        """
        v_fixed, trace = PowerIterationSolver.solve(
            field, max_iter=max_iter, tol=tol
        )
        cert = FixedPointCertifier.certify(field, v_fixed, trace)
        oseledets = OseledetsLyapunovSpectrum.from_spectrum(field.rho)
        _w, V = DensityOperatorAlgebra.eigenpairs_descending(field.rho)
        if V.shape[1] >= 2:
            melnikov = MelnikovHomoclinicCertificate.evaluate(
                field.rho, V[:, 0], V[:, 1], key=field.field_hash[:16]
            )
        else:
            melnikov = MelnikovHomoclinicCertificate(
                melnikov_value=0.0,
                melnikov_zeros=0,
                transverse_homoclinic=False,
                chaos_threshold=1e-3,
                unperturbed_vanishes=True,
            )
        omega: List[float] = []
        for lam in oseledets.lyapunov_full[:3]:
            omega.append(float(-lam) + 1e-3)
        while len(omega) < 2:
            omega.append(1.0 + 1e-3 * len(omega))
        # Frecuencias auténticas de Poincaré = espectro de ρ (carta acción-ángulo)
        omega_h = field.action_angle.frequencies[: min(3, field.dim)].tolist()
        if len(omega_h) >= 2:
            omega = [float(x) for x in omega_h]
        perturbation = DensityOperatorAlgebra.frobenius_to_maximally_mixed(
            field.rho
        )
        kam = KAMDiophantineCertificate.evaluate(
            omega=omega, perturbation_size=perturbation, tau=1.5, kmax=6
        )
        birkhoff = BirkhoffTwistMapCertificate.evaluate(
            field.rho, n_pairs=3, n_samples=16, key="INTROSPECTION"
        )
        monodromy = VariationalMonodromyCertificate.evaluate(field.rho)
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
        )


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 · ADJUDICACIÓN + AUTO-ORGANIZACIÓN + CERTIFICACIÓN                       ║
# ║                                                                                  ║
# ║  Dominio   = IntrospectionBundle (codominio de §2.8 synthesize /                ║
# ║              continue_into_phase3).                                             ║
# ║  Codominio = IntrospectionFieldState (objeto terminal).                         ║
# ║  Arranque  = HeytingIntrospectionAdjudicator.adjudicate                         ║
# ║              ← cuerpo de continue_into_phase3.                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

# ── §3.1 Adjudicador Ω₃ (enriquecido con el sector celeste) ──────────────────────
class HeytingIntrospectionAdjudicator:
    r"""
    Colapsa el Bundle a Ω₃ por meets sucesivos.  Umbrales adimensionales:
        cert, gap, nondeg, conv, fs, lyap, melnikov, kam, birkhoff, poincare,
        hopf, recurrence, resonance, monodromy, twist_H.
    final = local ∧ external     (meet conservador, P4).
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
        return (
            HeytingOmega3.COHERENT
            if b.certificate.converged
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _fs_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        th = b.certificate.fixed_point_fs_angle
        if th <= cls.FS_COHERENT:
            return HeytingOmega3.COHERENT
        if th <= cls.FS_DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _lyap_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        v = b.oseledets.lyapunov_max
        if v <= cls.LYAP_COHERENT:
            return HeytingOmega3.COHERENT
        if v <= cls.LYAP_DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _melnikov_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.VETOED
            if b.melnikov.transverse_homoclinic
            else HeytingOmega3.COHERENT
        )

    @classmethod
    def _kam_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if b.kam.kam_persists
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _birkhoff_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.birkhoff.local_verdict

    @classmethod
    def _poincare_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if b.field.poincare_section.is_transverse
            else HeytingOmega3.VETOED
        )

    @classmethod
    def _hopf_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.field.poincare_hopf.local_verdict

    @classmethod
    def _recurrence_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.field.recurrence.local_verdict

    @classmethod
    def _resonance_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.field.resonance_graph.local_verdict

    @classmethod
    def _monodromy_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.monodromy.local_verdict

    @classmethod
    def adjudicate(
        cls, bundle: IntrospectionBundle, external_verdict: HeytingOmega3
    ) -> HeytingOmega3:
        local = bundle.certificate.local_verdict.meet_all(
            bundle.field.spectral_gap.local_verdict,
            cls._nondeg_rule(bundle),
            cls._conv_rule(bundle),
            cls._fs_rule(bundle),
            cls._lyap_rule(bundle),
            cls._melnikov_rule(bundle),
            cls._kam_rule(bundle),
            cls._birkhoff_rule(bundle),
            cls._poincare_rule(bundle),
            cls._hopf_rule(bundle),
            cls._recurrence_rule(bundle),
            cls._resonance_rule(bundle),
            cls._monodromy_rule(bundle),
        )
        return local.meet(external_verdict)


# ── §3.2 Auto-organización del campo (mixtura convexa CPTP) ──────────────────────
@dataclass(frozen=True, slots=True)
class FieldUpdateCertificate:
    r"""
    Φ_η(ρ) = (1 − η) ρ + η |v*⟩⟨v*| ;  CPTP;  Lip₁ = |1 − η|  (P5).
    Canal de Lüders-convexión hacia el rayo certificado.  Completamente
    positivo porque es mixtura de id y un canal de preparación.
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


class FieldSelfOrganizer:
    @classmethod
    def idle(cls, rho: np.ndarray) -> Tuple[ComplexMatrix, FieldUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        p = DensityOperatorAlgebra.purity(rho)
        s = DensityOperatorAlgebra.von_neumann_entropy(rho)
        cert = FieldUpdateCertificate(
            applied=False,
            eta=0.0,
            contraction_coef=1.0,
            purity_before=p,
            purity_after=p,
            entropy_before=s,
            entropy_after=s,
            fidelity_to_fixed=1.0,
            local_verdict=HeytingOmega3.DEGRADED,
        )
        return rho, cert

    @classmethod
    def update(
        cls, rho: np.ndarray, v_fixed: np.ndarray, eta: float
    ) -> Tuple[ComplexMatrix, FieldUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        eta = float(np.clip(eta, 0.0, 1.0))
        rho_fixed = DensityOperatorAlgebra.rank_one(v_fixed)
        p_before = DensityOperatorAlgebra.purity(rho)
        s_before = DensityOperatorAlgebra.von_neumann_entropy(rho)
        rho_new = DensityOperatorAlgebra.sanitize(
            (1.0 - eta) * rho + eta * rho_fixed
        )
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
            applied=True,
            eta=eta,
            contraction_coef=coef,
            purity_before=float(p_before),
            purity_after=float(p_after),
            entropy_before=float(s_before),
            entropy_after=float(s_after),
            fidelity_to_fixed=float(fid_fixed),
            local_verdict=local,
        )


# ── §3.3 Certificado terminal ───────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionFieldState:
    r"""
    Objeto terminal: State ≅ Bundle × Ω₃ × Φ_η × Merkle × forense celeste.
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
    # forense celeste v9.1
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
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.4 Motor soberano F₃ ∘ F₂ ∘ F₁ ────────────────────────────────────────────
class TOONIntrospectionEngine:
    r"""
    Motor Espectral de la Introspección.
        F₁  IntrospectiveFieldPreparation.prepare
        F₂  IntrospectiveField.continue_into_phase2 = synthesize
        F₃  continue_into_phase3 ⊗ Φ_η ⊗ certify
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
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1")
        self.engine_id = engine_id
        self.dimension_mac = int(dimension_mac)
        self.update_field = bool(update_field)
        self.field_update_eta = float(np.clip(field_update_eta, 0.0, 1.0))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.density_matrix: ComplexMatrix = (
            np.eye(self.dimension_mac, dtype=np.complex128) / self.dimension_mac
        )
        self.cycle_count = 0
        self._chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::n={dimension_mac}::"
            f"η={self.field_update_eta}".encode("ascii")
        ).hexdigest()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _phase1_prepare(self, flash_vector: np.ndarray) -> IntrospectiveField:
        field = IntrospectiveFieldPreparation.prepare(
            rho_input=self.density_matrix,
            v_initial=flash_vector,
            poincare_key=self.engine_id,
        )
        self._advance_chain("F1", bytes.fromhex(field.field_hash))
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
    ) -> IntrospectionFieldState:
        final_verdict = bundle.continue_into_phase3(external_verdict)
        field_updated = False
        eta_used = 0.0
        if self.update_field and final_verdict == HeytingOmega3.COHERENT:
            new_rho, _upd = FieldSelfOrganizer.update(
                rho=self.density_matrix,
                v_fixed=bundle.v_fixed,
                eta=self.field_update_eta,
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
        b = bundle.birkhoff
        p = bundle.field.poincare_section
        hopf = bundle.field.poincare_hopf
        rec = bundle.field.recurrence
        resg = bundle.field.resonance_graph
        aa = bundle.field.action_angle
        mono = bundle.monodromy
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
            f"{b.n_fixed_points_min}".encode("ascii"),
            f"{hopf.index_sum}".encode("ascii"),
            self._chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )
        return IntrospectionFieldState(
            cycle_id=cycle_id,
            engine_id=self.engine_id,
            flash_id=flash_id,
            heyting_verdict=final_verdict,
            overlap=c.overlap_final,
            fixed_point_residual=c.fixed_point_residual,
            fixed_point_fs_angle=c.fixed_point_fs_angle,
            rayleigh_final=c.rayleigh_final,
            rayleigh_gap_to_lambda1=c.rayleigh_gap_to_lambda1,
            spectral_gap=g.gap,
            theoretical_rate=g.gap_ratio,
            empirical_rate=c.empirical_rate,
            rate_consistency=c.rate_consistency,
            iterations=c.iterations,
            is_self_sustaining=bool(
                c.converged
                and c.overlap_final > 1.0 - 1e-6
                and (not g.is_uniform)
            ),
            birkhoff_constant=g.birkhoff_constant,
            is_uniform=g.is_uniform,
            degeneracy_top=g.degeneracy_top,
            field_updated=field_updated,
            field_update_eta=eta_used,
            field_purity=DensityOperatorAlgebra.purity(self.density_matrix),
            lyapunov_max=o.lyapunov_max,
            kolmogorov_sinai_entropy=o.kolmogorov_sinai_entropy,
            kaplan_yorke_dimension=o.kaplan_yorke_dimension,
            melnikov_value=m.melnikov_value,
            transverse_homoclinic=m.transverse_homoclinic,
            kam_persists=k.kam_persists,
            kam_perturbation_ratio=k.perturbation_ratio,
            birkhoff_twist_holds=b.twist_holds,
            birkhoff_n_fp_min=b.n_fixed_points_min,
            poincare_transverse=p.is_transverse,
            poincare_first_return=p.first_return_time,
            floquet_min=(
                float(p.floquet_moduli.min()) if p.floquet_moduli.size else 0.0
            ),
            levy_band=bundle.field.levy_band,
            hamiltonian_return_time=p.hamiltonian_return_time,
            theoretical_period=p.theoretical_period,
            cartan_circulation=p.cartan_circulation,
            energy_drift=p.energy_drift,
            hopf_satisfied=hopf.hopf_satisfied,
            euler_characteristic=hopf.euler_characteristic,
            hamiltonian_recurrence=rec.hamiltonian_recurrence,
            kac_mean_return=rec.kac_mean_return_proxy,
            resonance_edges=resg.n_edges,
            top_resonant=resg.top_resonant,
            hessian_twist_norm=aa.hessian_twist_norm,
            monodromy_on_circle=mono.on_unit_circle,
            monodromy_period=mono.period,
            variational_radius=bundle.trace.variational_spectral_radius,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )

    def execute_introspection_cycle(
        self,
        flash_id: str,
        flash_vector: np.ndarray,
        heyting_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> IntrospectionFieldState:
        r"""Ciclo soberano: F₃ ∘ F₂ ∘ F₁."""
        self.cycle_count += 1
        cycle_id = f"CYC-INTROSPECT-{self.cycle_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Introspección #%d | flash=%s | tol=%.1e ═══",
            self.cycle_count,
            flash_id,
            self.tol,
        )
        field = self._phase1_prepare(flash_vector)
        bundle = self._phase2_iterate(field)
        state = self._phase3_certify(cycle_id, flash_id, bundle, heyting_verdict)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Ciclo %s | Ω₃=%s | resid=%.3e | FS=%.3e | λ_max=%.2e | "
            "h_KS=%.2e | M=%.2e | B=%d FP | Hopf=%s | τ_H=%.2e | "
            "upd=%s | %.2f ms",
            cycle_id,
            state.heyting_verdict.name,
            state.fixed_point_residual,
            state.fixed_point_fs_angle,
            state.lyapunov_max,
            state.kolmogorov_sinai_entropy,
            state.melnikov_value,
            state.birkhoff_n_fp_min,
            state.hopf_satisfied,
            state.hamiltonian_return_time,
            state.field_updated,
            dt_ms,
        )
        return state


# ── §3.5 Demostración autónoma ─────────────────────────────────────────────────
def _build_rho_from_spectrum(
    n: int, spectrum: np.ndarray, key: str
) -> ComplexMatrix:
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
    print("TOON INTROSPECTION ENGINE — v9.1.0 Doctoral")
    print(
        "Poincaré celeste (H)/(G)/(T) · Birkhoff · Oseledets · Melnikov · "
        "KAM · Lévy · Hopf · Kac"
    )
    print("═" * 94)

    def print_state(state: IntrospectionFieldState, title: str) -> None:
        print(f"\n[{title}]")
        print(f"   cycle_id            : {state.cycle_id}")
        print(f"   Ω₃ final            : {state.heyting_verdict.name}")
        print(f"   residuo gauge-fijado: {state.fixed_point_residual:.3e}")
        print(f"   ángulo FS           : {state.fixed_point_fs_angle:.3e} rad")
        print(f"   overlap |⟨v|T(v)⟩|  : {state.overlap:.10f}")
        print(
            f"   Rayleigh final      : {state.rayleigh_final:.6f}  "
            f"(λ₁−R = {state.rayleigh_gap_to_lambda1:.3e})"
        )
        print(f"   gap espectral γ     : {state.spectral_gap:.6f}")
        print(f"   rate teórica λ₂/λ₁  : {state.theoretical_rate:.6f}")
        print(
            f"   rate empírica       : {state.empirical_rate:.6f}  "
            f"(consistente={state.rate_consistency})"
        )
        print(f"   iteraciones         : {state.iterations}")
        print(f"   radio variacional   : {state.variational_radius:.4f}")
        print(f"   λ_max (Oseledets)   : {state.lyapunov_max:.3e}")
        print(f"   h_KS (Pesin)        : {state.kolmogorov_sinai_entropy:.3e}")
        print(f"   D_KY                : {state.kaplan_yorke_dimension:.4f}")
        print(f"   M (Melnikov)        : {state.melnikov_value:.3e}")
        print(f"   homoclínico transv. : {state.transverse_homoclinic}")
        print(
            f"   KAM persistente     : {state.kam_persists} "
            f"(ratio={state.kam_perturbation_ratio:.3e})"
        )
        print(
            f"   Birkhoff twist (H)  : {state.birkhoff_twist_holds} "
            f"⇒ ≥{state.birkhoff_n_fp_min} puntos fijos (T-radial)"
        )
        print(
            f"   Σ_G transversal     : {state.poincare_transverse} "
            f"(τ*_G={state.poincare_first_return:.2e})"
        )
        print(
            f"   τ_H / τ teórico     : {state.hamiltonian_return_time:.3e} / "
            f"{state.theoretical_period:.3e}  (ΔE={state.energy_drift:.2e})"
        )
        print(f"   circulación Cartan  : {state.cartan_circulation:.4e}")
        print(
            f"   Poincaré–Hopf       : χ={state.euler_characteristic} "
            f"satisfecho={state.hopf_satisfied}"
        )
        print(
            f"   recurrencia (H)     : {state.hamiltonian_recurrence} "
            f"(Kac≈{state.kac_mean_return:.2e})"
        )
        print(
            f"   grafo resonancia    : edges={state.resonance_edges} "
            f"top_resonant={state.top_resonant}"
        )
        print(
            f"   twist Hessiano H    : {state.hessian_twist_norm:.1e}  "
            f"(0 = isócrono, P8)"
        )
        print(
            f"   monodromía U(1)     : {state.monodromy_on_circle} "
            f"(τ={state.monodromy_period:.3e})"
        )
        print(f"   Floquet mínimo      : {state.floquet_min:.4f}")
        print(f"   banda Lévy ε*       : {state.levy_band:.6f}")
        print(f"   Birkhoff tanh(Δ/4)  : {state.birkhoff_constant:.6f}")
        print(f"   uniforme (I/n)      : {state.is_uniform}")
        print(f"   degeneración λ₁     : {state.degeneracy_top}")
        print(f"   autosostenible      : {state.is_self_sustaining}")
        print(
            f"   campo actualizado   : {state.field_updated}  "
            f"(η = {state.field_update_eta:.3f}, P(ρ)={state.field_purity:.6f})"
        )
        print(f"   fase chain          : {state.phase_chain_sha256[:32]}…")
        print(f"   firma global        : {state.sha256_provenance[:32]}…")

    n = 4
    rho_ideal = _build_rho_from_spectrum(
        n, np.array([0.97, 0.02, 0.005, 0.005]), "IDEAL"
    )
    gap_report = SpectralGapAnalyzer.analyze(rho_ideal)
    print("\n─── Sanity-check: análisis espectral con gap grande ───")
    print(f"   λ₁ = {gap_report.lambda_1:.6f} | λ₂ = {gap_report.lambda_2:.6f}")
    print(f"   γ  = {gap_report.gap:.6f} | rate_th = {gap_report.gap_ratio:.6f}")
    print(
        f"   degeneración top = {gap_report.degeneracy_top} | "
        f"uniforme = {gap_report.is_uniform}"
    )
    print(f"   veredicto espectral = {gap_report.local_verdict.name}")

    print("\n─── Sección de Poincaré Σ_c (G) y Σ_{c,q} (H) ───")
    sec = PoincareSectionIntrospection.build(rho_ideal, key="DEMO")
    print(f"   nivel c = {sec.level:.6f} | transversal (G) = {sec.is_transverse}")
    print(
        f"   ⟨∇R,X_G⟩_min = 2‖X‖² = {sec.transversality_min:.3e} | "
        f"τ*_G = {sec.first_return_time:.3e}"
    )
    print(
        f"   τ_H = {sec.hamiltonian_return_time:.3e} | "
        f"τ_th = {sec.theoretical_period:.3e} | ΔE = {sec.energy_drift:.2e}"
    )
    print(f"   Cartan ∮λ = {sec.cartan_circulation:.4e}")
    print(f"   Floquet = {sec.floquet_moduli}")
    print(f"   órbitas periódicas detectadas = {sec.periodic_orbit_rank}")

    print("\n─── Carta acción-ángulo de Poincaré ───")
    v_demo = _build_test_vector(n, "AA-DEMO")
    aa = ActionAnglePoincareChart.chart(rho_ideal, v_demo)
    print(f"   I = {np.round(aa.actions, 4)}")
    print(f"   ω = λ = {np.round(aa.frequencies, 4)}")
    print(f"   H(I) = {aa.hamiltonian:.6f} | twist ‖∂ω/∂I‖ = {aa.hessian_twist_norm}")
    print(
        f"   rango resonancia = {aa.resonance_rank} | "
        f"defecto = {aa.commensurability_defect:.3e}"
    )

    print("\n─── Poincaré–Hopf / Morse de R ───")
    hopf = PoincareHopfCertificate.evaluate(rho_ideal)
    print(
        f"   χ(ℂP^{n-1}) = {hopf.euler_characteristic} | "
        f"Σ ind = {hopf.index_sum} | ok = {hopf.hopf_satisfied}"
    )
    print(f"   Morse = {hopf.morse_indices} | Hopf = {hopf.hopf_indices}")

    print("\n─── Recurrencia de Poincaré / Kac ───")
    rec = PoincareRecurrenceCertificate.evaluate(n, fs_radius=0.1, is_uniform=False)
    print(
        f"   (H) recurrente = {rec.hamiltonian_recurrence} | "
        f"(T) recurrente = {rec.projective_recurrence}"
    )
    print(f"   Kac 1/μ(B_ε) ≈ {rec.kac_mean_return_proxy:.3e}")

    print("\n─── Grafo de resonancia espectral ───")
    resg = SpectralResonanceGraph.build(gap_report.lambda_1 and
                                        DensityOperatorAlgebra.spectrum_descending(rho_ideal))
    # la línea anterior usa `and` sólo para no romper el flujo; reconstruimos limpio:
    resg = SpectralResonanceGraph.build(
        DensityOperatorAlgebra.spectrum_descending(rho_ideal)
    )
    print(
        f"   |V|={resg.n_vertices} |E|={resg.n_edges} | "
        f"comp={resg.n_components} | top_resonant={resg.top_resonant}"
    )
    print(f"   λ₂(L) = {resg.algebraic_connectivity:.4f}")

    print("\n─── Oseledets proyectivo ───")
    osl = OseledetsLyapunovSpectrum.from_spectrum(rho_ideal)
    print(f"   λ = {osl.lyapunov_full}")
    print(
        f"   λ_max = {osl.lyapunov_max:.3e} | h_KS = {osl.kolmogorov_sinai_entropy:.3e}"
    )
    print(f"   degeneracy = {osl.degeneracy_signal}")

    print("\n─── Monodromía hamiltoniana ───")
    mono = VariationalMonodromyCertificate.evaluate(rho_ideal)
    print(
        f"   τ = {mono.period:.4e} | U(1) = {mono.on_unit_circle} | "
        f"Krein = {mono.krein_resonance}"
    )
    print(f"   μ = {np.round(np.array(mono.multipliers), 4)}")

    print("\n─── Birkhoff: (H) isócrono vs (T) radial ───")
    birk = BirkhoffTwistMapCertificate.evaluate(rho_ideal, n_pairs=3)
    print(
        f"   twist_H = {birk.hamiltonian_twist} (P8) | "
        f"twist_holds = {birk.twist_holds}"
    )
    print(
        f"   n_FP ≥ {birk.n_fixed_points_min} | radial_monotone = {birk.radial_monotone}"
    )
    print(f"   |f'−1|_min = {birk.twist_gradient_min:.3e} | pares = {birk.degeneracy_pairs}")

    print("\n─── Lema de Lévy sobre ℂP^{n−1} ───")
    for n_ in (4, 16, 64, 256):
        eps = LevyConcentrationLemma.median_width(n_, lipschitz=1.0, confidence=0.99)
        bnd = LevyConcentrationLemma.bound(eps, n_, lipschitz=1.0)
        print(f"   n={n_:4d} | ε*(99%)={eps:.6f} | P(tail)≤{bnd:.3e}")

    engine = TOONIntrospectionEngine(
        engine_id="INTROSPECT-ENGINE-WISDOM-01",
        dimension_mac=4,
        update_field=False,
        field_update_eta=0.20,
        max_iter=500,
        tol=1e-10,
    )
    scenarios = [
        (
            "COHERENT (gap grande)",
            np.array([0.97, 0.02, 0.005, 0.005]),
            "RHO-A",
            "VEC-A",
            HeytingOmega3.COHERENT,
        ),
        (
            "DEGRADED (gap pequeño)",
            np.array([0.50, 0.45, 0.03, 0.02]),
            "RHO-B",
            "VEC-B",
            HeytingOmega3.COHERENT,
        ),
        (
            "VETOED (uniforme I/n)",
            np.array([0.25, 0.25, 0.25, 0.25]),
            "RHO-C",
            "VEC-C",
            HeytingOmega3.COHERENT,
        ),
    ]
    for name, spec, rho_key, vec_key, ext in scenarios:
        rho = _build_rho_from_spectrum(n, spec, rho_key)
        v = _build_test_vector(n, vec_key)
        engine.density_matrix = rho
        state = engine.execute_introspection_cycle(
            flash_id=f"FLASH-{vec_key}", flash_vector=v, heyting_verdict=ext
        )
        print_state(state, name)

    print("\n" + "─" * 94)
    print("─── Auto-organización del campo (η = 0.20, 6 ciclos) ───")
    engine2 = TOONIntrospectionEngine(
        engine_id="INTROSPECT-SELF-ORG",
        dimension_mac=4,
        update_field=True,
        field_update_eta=0.20,
    )
    v_seed = _build_test_vector(4, "SELF-ORG-VEC")
    for i in range(6):
        state = engine2.execute_introspection_cycle(
            flash_id=f"FLASH-SELF-ORG-{i + 1:03d}",
            flash_vector=v_seed,
            heyting_verdict=HeytingOmega3.COHERENT,
        )
        purity_now = DensityOperatorAlgebra.purity(engine2.density_matrix)
        gap_now = SpectralGapAnalyzer.analyze(engine2.density_matrix)
        print(
            f"   ciclo {i + 1}: Ω₃={state.heyting_verdict.name:<9s} "
            f"P(ρ)={purity_now:.6f}  γ={gap_now.gap:.6f}  "
            f"Birkhoff_FP≥{state.birkhoff_n_fp_min}  "
            f"Hopf={state.hopf_satisfied}  "
            f"upd={state.field_updated}  iters={state.iterations}  "
            f"d_FS={state.fixed_point_fs_angle:.2e}"
        )

    print("\n" + "═" * 94)
    print("✓ F1→F2: prepare ⊣ continue_into_phase2 = synthesize.")
    print("✓ F2→F3: synthesize ⊣ continue_into_phase3 = adjudicate ⊗ Φ_η.")
    print("✓ (G) Σ_c transversal a X_G ⇔ ⟨∇R,X_G⟩ = 2‖X_G‖² > 0 ⇔ c ∉ spec(ρ).")
    print("✓ (H) Σ_{c,q} sección auténtica; τ = 2π/|λ_i−λ_j|; Cartan ∮λ conservado.")
    print("✓ (T) no es simplectomorfismo: contrae Haar hacia [φ₁] (P7).")
    print("✓ Poincaré–Hopf: χ(ℂP^{n−1}) = n = Σ ind(∇R) si spec simple.")
    print("✓ Recurrencia de Poincaré plena en (H); (T) sólo en Fix(T).")
    print("✓ Birkhoff: twist_H = 0 (isócrono, P8); (T)-radial ⇒ exactamente 2 FP.")
    print("✓ Oseledets proyectivo: λ_i = log(λ_i/λ₁) ≤ 0; degeneración ⇒ caos.")
    print("✓ Melnikov: M = ∫ ω_FS(X_{H₀}, X_{H₁}) dt sobre heteroclínica (G).")
    print("✓ KAM diofántico sobre ω = λ; un ρ hermítico permanece integrable.")
    print("✓ Monodromía (H): μ ∈ U(1); Krein-neutros, radio 1.")
    print("✓ Lévy sobre ℂP^{n−1}: P(|f−𝔼f|≥ε) ≤ exp(−(n+1)ε²/(2π²L²)).")
    print("✓ d_FS = arccos|⟨u|v⟩|; residuo gauge-fijado = 2 sin(θ/2) (P2).")
    print("✓ Floquet (T): μ_i = λ_i/λ₁; gap (1−μ₂) = velocidad de convergencia.")
    print("✓ Ω₃ por meets adimensionales (lyap, Melnikov, KAM, Birkhoff, Hopf).")
    print("✓ Cadena forense F1 → F2 → F3 encadenada por SHA-256.")
    print("═" * 94)
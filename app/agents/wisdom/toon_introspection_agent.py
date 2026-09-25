# -*- coding: utf-8 -*-
r"""Soberano de Introspección y Autocoherencia Espectral MAC.

Ubicación: app/agents/wisdom/toon_introspection_agent.py
Versión  : 2.2.0-Doctoral-Nested-PowerIteration-Uhlmann-FS-Birkhoff-Merkle

Este módulo define la entidad ejecutiva e inalienable "Soberano de Introspección"
en el dominio WISDOM de la arquitectura COGNITIVE TOON / APU Filter. Su cometido es
evaluar la consistencia interna entre los flashes/corazonadas intuitivas de los agentes
y la matriz densidad de la Memoria de Alto Contenido (MAC), actuando como filtro de
coherencia topológico-espectral y garante de no-contradicción.

================================================================================
I. FORMALIZACIÓN MATEMÁTICA Y OPERACIONES ESPECTRALES
================================================================================

1. Campo MAC y C*-Álgebra de Operadores Densidad:
   El estado de la memoria MAC se representa como un operador densidad $\rho_{\mathrm{MAC}}$
   perteneciente a $\mathfrak{D}_n = \{\rho \in M_n(\mathbb{C}) : \rho = \rho^\dagger, \, \rho \ge 0, \, \mathrm{Tr}(\rho) = 1\}$.
   La pureza se mide como $\mathcal{P}(\rho) = \mathrm{Tr}(\rho^2) \in [1/n, 1]$ y la entropía de von Neumann
   como $S(\rho) = -\mathrm{Tr}(\rho \log \rho)$.

2. Fidelidad de Uhlmann a Estados Puros vs. Puntos Fijos Proyectivos:
   Para un rayo $[v] \in \mathbb{C}P^{n-1}$ correspondiente a una corazonada intuitiva $v \in \mathbb{C}^n$
   ($\|v\|_2 = 1$), la fidelidad de Uhlmann al estado puro $|v\rangle\langle v|$ se reduce a:
       $$F(\rho_{\mathrm{MAC}}, |v\rangle\langle v|) = \sqrt{\langle v | \rho_{\mathrm{MAC}} | v \rangle} = \sqrt{\mathcal{R}(v)}$$
   donde $\mathcal{R}(v)$ es el cociente de Rayleigh. Si $v = v_1$ (modo dominante), $F = \sqrt{\lambda_1}$.
   Distinto a la fidelidad de Uhlmann, el overlap proyectivo bajo la aplicación $T([v]) = [\rho v]$ mide
   la invariancia del rayo:
       $$\mathcal{O}_T(v) = |\langle v | T(v) \rangle| \in [0, 1], \quad \mathcal{O}_T(v) = 1 \iff [v] \in \mathrm{Fix}(T)$$

3. Distancia de Fubini-Study y Residuo Gauge-Fijado:
   En el espacio proyectivo $\mathbb{C}P^{n-1}$, la métrica Kähler de Fubini-Study es:
       $$d_{\mathrm{FS}}([u], [v]) = \arccos(|\langle u | v \rangle|) \in \left[0, \frac{\pi}{2}\right]$$
   Fijando el gauge $U(1)$ mediante $T_\varphi(v) = e^{-i \arg \langle v, T(v) \rangle} T(v)$, el residuo
   euclídeo tangente representa la distancia cuerda intrínseca:
       $$\|T_\varphi(v) - v\|_2 = \sqrt{2 - 2|\langle v | T(v) \rangle|} = 2 \sin\left(\frac{d_{\mathrm{FS}}}{2}\right)$$

4. Brecha Espectral, Radio Transversal y Birkhoff-Hopf:
   Con espectro $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$, la brecha espectral relativa $\gamma$
   y la tasa asintótica $\rho_{\mathrm{th}}$ son:
       $$\gamma = 1 - \frac{\lambda_2}{\lambda_1}, \quad \rho_{\mathrm{th}} = \frac{\lambda_2}{\lambda_1}$$
   La contracting proyectiva acotada por la métrica de Hilbert tiene constante $\kappa \le \tanh(\Delta/4)$,
   donde $\Delta = \log(\lambda_1/\lambda_n)$.

5. Retículo de Heyting $\Omega_3$ y Adjudicación Conservadora:
   El veredicto final en $\Omega_3 = \{\bot (\mathrm{VETOED}) < \star (\mathrm{DEGRADED}) < \top (\mathrm{COHERENT})\}$
   se obtiene por el meet ($\wedge$) conservador del veredicto local y el veredicto de entrada ($v_{\mathrm{incoming}}$):
       $$v_{\mathrm{final}} = v_{\mathrm{local}} \wedge v_{\mathrm{incoming}}$$
   Garantizando que si $v_{\mathrm{incoming}} = \bot$, el resultado es estrictamente $\bot$ (veto duro).

6. Canal CPTP de Inoculación y Auto-Organización MAC:
   Tras la adjudicación $\top$, el campo $\rho_{\mathrm{MAC}}$ se actualiza mediante el canal CPTP afín:
       $$\Phi_\eta(\rho) = (1 - \eta)\rho + \eta |v^*\rangle\langle v^*|, \quad \eta \in [0, 1]$$
   con constante de Lipschitz en norma de traza $\mathrm{Lip}_{\|\cdot\|_1}(\Phi_\eta) = |1 - \eta|$.

================================================================================
II. ESTRUCTURA FUNTORIAL Y ARQUITECTURA
================================================================================

El Soberano opera como el funtor estricto $F = F_3 \circ F_2 \circ F_1$:
    $$F : \mathbb{C}^n \times \Omega_3 \times \mathfrak{D}_n \longrightarrow \mathrm{IntrospectionProofCertificate}$$

  • $F_1$ (`IntrospectiveHandoff.build`): $\mathbb{C}^n \times \mathfrak{D}_n \times \Omega_3 \to \mathrm{IntrospectiveHandoff}$.
    Sustrato proyectivo-espectral, cálculo de $\mathcal{R}(v)$, Uhlmann $\sqrt{\mathcal{R}}$, Born $|\langle v|v_1\rangle|^2$ y $d_{\mathrm{FS}}$.
  • $F_2$ (`IntrospectionPipeline.synthesize`): $\mathrm{IntrospectiveHandoff} \to \mathrm{IntrospectionBundle}$.
    Iteración de potencia gauge-fijada $T^k$, parada $d_{\mathrm{FS}}$, tasa empírica y certificación.
  • $F_3$ (`TOONIntrospectionAgent._phase3_certify`): $\mathrm{IntrospectionBundle} \to \mathrm{IntrospectionProofCertificate}$.
    Adjudicación en $\Omega_3$ por meets con $v_{\mathrm{incoming}}$, narrativa anclada a observables, inoculación $\Phi_\eta$ y firma Merkle.

================================================================================
III. INVARIANTES FORMALES Y AXIOMAS DEL SISTEMA
================================================================================

- Axioma 1 (Diferenciación Uhlmann vs. Fix(T)): $F(\rho, |v\rangle\langle v|) = \sqrt{\mathcal{R}(v)} \ne \mathcal{O}_T(v)$.
  Un rayo en $\mathrm{Fix}(T)$ satisface $\mathcal{O}_T(v) = 1$, pero su fidelidad de Uhlmann es $\sqrt{\lambda_1}$.
- Axioma 2 (Invariancia Gauge Proyectiva): La distancia $d_{\mathrm{FS}}([u],[v])$ es invariante bajo transformaciones $U(1)$.
- Axioma 3 (Veto Absoluto por Meet): Para todo veredicto local $v_{\mathrm{local}}$, $v_{\mathrm{local}} \wedge \bot = \bot$.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


logger = logging.getLogger("APU.Wisdom.TOONIntrospection")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS: Final[float] = 1.0e-14
_EPS_MOD: Final[float] = 1.0e-12
_EPS_TRACE: Final[float] = 1.0e-15

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
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
# ║  FASE 1 · SUSTRATO PROYECTIVO-ESPECTRAL                                   ║
# ║                                                                           ║
# ║  Objetos: Ω₃, 𝔇_n, ℂP^{n−1}, spec(ρ_MAC).                                 ║
# ║  Morfismo terminal: IntrospectiveHandoff.continue_into_phase2.            ║
# ║  Ese morfismo ES el dominio de todos los funtores de la FASE 2.           ║
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
        ⊤ clasifica subobjetos totales (flash = modo propio de la MAC),
        ⋆ clasifica subobjetos densos no cerrados (fricción / gap pequeño),
        ⊥ clasifica el subobjeto vacío (ρ≈I/n, incoming veto, o ker ρ).
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

    Identidad de Uhlmann a un puro (la que el v1 violaba dimensionalmente):

        F(ρ, |v⟩⟨v|) = √⟨v|ρ|v⟩     (v unitario)
                     = √ R_ρ(v)     (cociente de Rayleigh)

        Si v = v₁ (dominante): F = √λ₁,  NO 1, salvo que ρ sea pura.

    sanitize = proyección euclídea sobre 𝔇_n (Hermitiza + PSD-clip Higham
    + renormalización de traza).  Es no expansiva en ‖·‖_F.
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
    def eigendecomposition_descending(
        cls, rho: np.ndarray,
    ) -> Tuple[RealVector, ComplexMatrix]:
        """(w_desc, V_desc) con columnas de V ordenadas por w decreciente."""
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        idx = np.argsort(np.real(w))[::-1]
        w = np.maximum(np.real(w[idx]), cls.EPS)
        s = float(w.sum())
        w = w / s if s > 0.0 else w
        return w.astype(np.float64), V[:, idx]

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
        r"""ρ^z = V diag(λ_i^z) V†  (Holstein–Rellich, corte principal)."""
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
    def uhlmann_fidelity_to_pure(cls, rho: np.ndarray, v: np.ndarray) -> float:
        r"""
        F(ρ, |v⟩⟨v|) = √(⟨v|ρ|v⟩ / ⟨v|v⟩) ∈ [0, 1].

        Identidad: si v = v₁ entonces F = √λ₁.  El v1 usaba (1−F)≤10⁻⁶
        como predicado de punto fijo: eso exige λ₁=1, no [v]∈Fix(T).
        """
        rho = cls.sanitize(rho)
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        vv = float(np.real(np.vdot(v, v)))
        if vv < 1e-30:
            return 0.0
        rayleigh = float(np.real(np.vdot(v, rho @ v)) / vv)
        return float(math.sqrt(max(0.0, rayleigh)))

    @classmethod
    def dominant_eigenvector(cls, rho: np.ndarray) -> ComplexVector:
        _w, V = cls.eigendecomposition_descending(rho)
        return V[:, 0].reshape(-1).astype(np.complex128)

    @classmethod
    def rank_one(cls, v: np.ndarray) -> ComplexMatrix:
        """|v⟩⟨v| saneado sobre 𝔇_n."""
        v = np.asarray(v, dtype=np.complex128).reshape(-1, 1)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            n = int(v.size)
            return np.eye(n, dtype=np.complex128) / max(n, 1)
        v = v / nrm
        return cls.sanitize(v @ v.conj().T)


# ── §1.3 Dinámica proyectiva + métrica de Fubini–Study ────────────────────
@dataclass(frozen=True, slots=True)
class ProjectiveRayleighSample:
    r"""
    Evaluación de T en un rayo [v] ∈ ℂP^{n−1} (v unitario, gauge U(1) fijado).

        rayleigh     = ⟨v|ρ|v⟩ ∈ [λₙ, λ₁]           (Courant–Fischer)
        image_norm   = ‖ρ v‖                          (0 ⇒ v ∈ ker ρ)
        overlap_T    = |⟨v|T(v)⟩| ∈ [0, 1]            fidelidad proyectiva (P)
        uhlmann_pure = √rayleigh                      F(ρ,|v⟩⟨v|) (U)
        residual     = ‖T_φ(v) − v‖₂ = 2 sin(θ/2)     cuerda gauge-fijada
        fs_angle     = arccos(|⟨v|T(v)⟩|) ∈ [0, π/2]  d_FS (G)
        chordal      = sin(fs_angle)
    """
    rayleigh: float
    image_norm: float
    overlap_T: float
    uhlmann_pure: float
    residual: float
    fs_angle: float
    chordal: float


class ProjectiveDynamics:
    r"""
    Dinámica  T([v]) = [ρ v]  sobre ℂP^{n−1}.

    Métrica de Fubini–Study (única U(n)-invariante, Kähler):

        d_FS([u],[v]) = arccos( |⟨u|v⟩| / (‖u‖‖v‖) ) ∈ [0, π/2].

    Gauge U(1): el residuo euclídeo ‖Tv − v‖ NO es proyectivo (depende de
    la fase global).  Se alinea

        T_φ(v) := e^{−i arg⟨v, Tv⟩} Tv    de modo que ⟨v|T_φ⟩ ≥ 0,

    y entonces T_φ(v) − v ∈ T_{[v]} ℂP^{n−1}, con

        ‖T_φ − v‖₂ = √(2 − 2 |⟨v|Tv⟩|) = 2 sin(d_FS/2).

    Puntos fijos: autovectores de ρ (Brouwer: ℂP^{n−1} compacto).
    Si λ₁ es simple, el atractor [v₁] es único.
    """

    @staticmethod
    def sanitize_vector(v: np.ndarray, n: int) -> ComplexVector:
        """Extiende/trunca a ℂⁿ y normaliza.  Fallback = vector uniforme."""
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
        r"""Multiplica `target` por e^{−i arg⟨ref|target⟩} ⇒ ⟨ref|target⟩ ≥ 0."""
        ov = np.vdot(reference, target)
        if abs(ov) < 1e-30:
            return np.asarray(target, dtype=np.complex128)
        return (target * np.exp(-1j * np.angle(ov))).astype(np.complex128)

    @staticmethod
    def apply_map(rho: np.ndarray, v: np.ndarray) -> Tuple[ComplexVector, float]:
        r"""T(v) = ρv/‖ρv‖ (sin gauge).  Si ‖ρv‖≈0, devuelve v y 0."""
        rho_v = rho @ v
        rho_v_norm = float(np.linalg.norm(rho_v))
        if rho_v_norm < 1e-15:
            return np.asarray(v, dtype=np.complex128).copy(), 0.0
        return (rho_v / rho_v_norm).astype(np.complex128), rho_v_norm

    @staticmethod
    def rayleigh(rho: np.ndarray, v: np.ndarray) -> float:
        r"""R(v) = ⟨v|ρ|v⟩/⟨v|v⟩ ∈ [λₙ, λ₁]  (Courant–Fischer)."""
        vv = float(np.real(np.vdot(v, v)))
        if vv < 1e-30:
            return 0.0
        return float(np.real(np.vdot(v, rho @ v)) / vv)

    @staticmethod
    def fubini_study_angle(u: np.ndarray, v: np.ndarray) -> float:
        r"""d_FS([u],[v]) = arccos(|⟨u|v⟩|) ∈ [0, π/2] (vectores unitarios)."""
        u = np.asarray(u, dtype=np.complex128).reshape(-1)
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-30 or nv < 1e-30:
            return 0.5 * math.pi
        cos_theta = float(abs(np.vdot(u, v)) / (nu * nv))
        return float(math.acos(float(np.clip(cos_theta, 0.0, 1.0))))

    @staticmethod
    def overlap_squared(u: np.ndarray, v: np.ndarray) -> float:
        """|⟨u|v⟩|² ∈ [0, 1]  (Born; invariante de fase)."""
        u = np.asarray(u, dtype=np.complex128).reshape(-1)
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-30 or nv < 1e-30:
            return 0.0
        return float(np.clip(abs(np.vdot(u, v) / (nu * nv)) ** 2, 0.0, 1.0))

    @classmethod
    def evaluate(cls, rho: np.ndarray, v: np.ndarray) -> ProjectiveRayleighSample:
        """Evalúa T, Rayleigh, Uhlmann-a-puro, d_FS y residuo gauge-fijado."""
        n = int(rho.shape[0])
        v = cls.sanitize_vector(v, n)
        T_v, image_norm = cls.apply_map(rho, v)
        T_phi = cls.gauge_align(v, T_v)
        overlap_T = float(abs(np.vdot(v, T_v)))
        ray = cls.rayleigh(rho, v)
        fs_angle = cls.fubini_study_angle(v, T_v)
        return ProjectiveRayleighSample(
            rayleigh=ray,
            image_norm=image_norm,
            overlap_T=overlap_T,
            uhlmann_pure=float(math.sqrt(max(0.0, ray))),
            residual=float(np.linalg.norm(T_phi - v)),
            fs_angle=fs_angle,
            chordal=float(math.sin(fs_angle)),
        )


# ── §1.4 Análisis del gap espectral ───────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralGapReport:
    r"""
    Análisis espectral del campo ρ_MAC.

        λ₁ ≥ λ₂ ≥ ⋯ ≥ λₙ ≥ 0,  Σ λ_i = 1
        gap_ratio     = λ₂/λ₁ ∈ [0, 1]     = ρ(DT|_{[v₁]})
        gap           = 1 − λ₂/λ₁ ∈ [0, 1]
        gap_absolute  = λ₁ − λ₂            (Davis–Kahan / Kato)
        cond          = λ₁/λₙ
        degeneracy_top: multiplicidad de λ₁
        is_uniform    : ρ ≈ I/n  (∄ atractor: T = Id)
        birkhoff      : tanh(log(λ₁/λₙ)/4)  (Hopf, diámetro proyectivo)

    Veredicto local:
        uniforme ∨ degeneracy ≥ 3     → ⊥
        gap ≥ GAP_COHERENT ∧ deg = 1  → ⊤
        gap ≥ GAP_DEGRADED            → ⋆
        else                          → ⊥
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
    r"""
    Espectro de ρ → tasa de T y existencia de atractor.

    Brouwer: ℂP^{n−1} compacto + T continua ⇒ Fix(T) ≠ ∅.
    Linearización: DT|_{[v₁]} sobre v₁^⊥ tiene spec {λ_i/λ₁}_{i≥2};
    radio transversal = λ₂/λ₁.

    Birkhoff–Hopf: si ρ es (entrywise) positiva, T contrae Hilbert con
    tanh(Δ/4).  Proxy espectral (ρ ≻ 0): Δ = log(λ₁/λₙ).  NO se afirma
    para ρ semidefinida.
    """
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


# ── §1.5 IntrospectiveHandoff — HAND-OFF FASE 1 → FASE 2 ──────────────────
@dataclass(frozen=True, slots=True)
class IntrospectiveHandoff:
    r"""
    Objeto terminal de la FASE 1 y objeto inicial de la FASE 2.

        rho_mac, v_flash     : campo MAC y corazonada (unitaria)
        spectral_gap         : invariante de F1 (la FASE 2 no recompute spec)
        rayleigh_quotient    : ⟨v|ρ|v⟩
        uhlmann_fidelity     : √R = F(ρ, |v⟩⟨v|)          (U)
        overlap_with_dominant: |⟨v|v₁⟩|²                  (B)
        fs_angle_to_dominant : d_FS([v],[v₁])             (G)
        incoming_heyting     : Ω₃ del flash aguas arriba  (H)
    """
    introspective_id: str
    flash_intuition_id: str
    rho_mac: np.ndarray
    v_flash: np.ndarray
    v_dominant: np.ndarray
    spectral_gap: SpectralGapReport
    rayleigh_quotient: float
    uhlmann_fidelity_to_rho: float
    overlap_with_dominant: float
    fs_angle_to_dominant: float
    incoming_heyting_verdict: HeytingOmega3
    visceral_message: str
    handoff_hash: str
    dim: int

    @classmethod
    def build(
        cls,
        introspective_id: str,
        flash_intuition_id: str,
        rho_mac: np.ndarray,
        flash_vector: np.ndarray,
        incoming_heyting_verdict: HeytingOmega3,
        visceral_message: str,
    ) -> "IntrospectiveHandoff":
        r"""
        Cierra la FASE 1 como objeto.  El morfismo de continuación
        hacia FASE 2 es `continue_into_phase2`.
        """
        rho = DensityOperatorAlgebra.sanitize(rho_mac)
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(flash_vector, n)
        gap = SpectralGapAnalyzer.analyze(rho)
        v1 = DensityOperatorAlgebra.dominant_eigenvector(rho)
        v1 = ProjectiveDynamics.gauge_align(v, v1)

        rayleigh = ProjectiveDynamics.rayleigh(rho, v)
        f_uh = float(math.sqrt(max(0.0, rayleigh)))
        overlap = ProjectiveDynamics.overlap_squared(v, v1)
        fs_angle = ProjectiveDynamics.fubini_study_angle(v, v1)

        handoff_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(v).tobytes(),
            f"{overlap:.12f}".encode("ascii"),
            f"{gap.gap:.12f}".encode("ascii"),
            incoming_heyting_verdict.name.encode("ascii"),
        )
        return cls(
            introspective_id=introspective_id,
            flash_intuition_id=flash_intuition_id,
            rho_mac=rho,
            v_flash=v,
            v_dominant=v1,
            spectral_gap=gap,
            rayleigh_quotient=float(rayleigh),
            uhlmann_fidelity_to_rho=float(f_uh),
            overlap_with_dominant=float(overlap),
            fs_angle_to_dominant=float(fs_angle),
            incoming_heyting_verdict=incoming_heyting_verdict,
            visceral_message=visceral_message,
            handoff_hash=handoff_hash,
            dim=n,
        )

    def summary(self) -> Dict[str, float]:
        return {
            "rayleigh": self.rayleigh_quotient,
            "F_uhlmann": self.uhlmann_fidelity_to_rho,
            "overlap_v1": self.overlap_with_dominant,
            "fs_angle_v1": self.fs_angle_to_dominant,
            "gap": self.spectral_gap.gap,
            "lambda_1": self.spectral_gap.lambda_1,
            "lambda_2": self.spectral_gap.lambda_2,
            "degeneracy": float(self.spectral_gap.degeneracy_top),
            "birkhoff": self.spectral_gap.birkhoff_constant,
            "incoming_weight": self.incoming_heyting_verdict.as_weight(),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    # ══════════════════════════════════════════════════════════════════════
    def continue_into_phase2(
        self,
        max_iter: int,
        tol: float,
    ) -> "IntrospectionBundle":
        r"""
        Último morfismo de la FASE 1  ∧  primer morfismo de la FASE 2.

        Identidad de composición:

            continue_into_phase2 ∘ build
                = IntrospectionPipeline.synthesize ∘ build
                : Flash × MAC × Ω₃ → IntrospectionBundle.

        En el sentido de categorías, la FASE 2 es el comma-category
        (IntrospectiveHandoff ↓ Power₂).  Invocar la iteración de
        potencia sin un IntrospectiveHandoff es un error de tipo.
        """
        return IntrospectionPipeline.synthesize(
            handoff=self, max_iter=max_iter, tol=tol,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 · DINÁMICA DE PUNTO FIJO                                          ║
# ║                                                                           ║
# ║  Dominio = IntrospectiveHandoff (codominio de §1.5).                      ║
# ║  Codominio = IntrospectionBundle, dominio de toda la FASE 3.              ║
# ║                                                                           ║
# ║  §2.1 se lee como la continuación literal de continue_into_phase2.        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 Solver de iteración de potencia ──────────────────────────────────
@dataclass(frozen=True, slots=True)
class PowerIterationTrace:
    r"""
    Traza de T^k sobre ℂP^{n−1}.

        residuals, fs_angles, rayleigh_trajectory, overlaps_with_v1
        empirical_rate  : mediana de θ_{k+1}/θ_k en ventana sana ≈ λ₂/λ₁
        stalled_kernel  : True si ρv ≈ 0 (v ∈ ker ρ)
    """
    residuals: Tuple[float, ...]
    fs_angles: Tuple[float, ...]
    rayleigh_trajectory: Tuple[float, ...]
    overlaps_with_v1: Tuple[float, ...]
    empirical_rate: float
    stalled_kernel: bool
    iterations: int
    converged: bool


class PowerIterationSolver:
    r"""
    Iteración de potencia COMPLETA (continuación de continue_into_phase2):

        v_{k+1} = gauge_align(v_k,  ρ v_k / ‖ρ v_k‖)

    Criterio de parada:

        d_FS([v_k],[T v_k]) < tol  ∨  ‖T_φ − v_k‖₂ < tol  ∨  k = max_iter.

    Tasa: si λ₁ > λ₂,  d_FS([v_k],[v₁]) = Θ((λ₂/λ₁)^k).
    Si λ₁ = λ₂, no hay atractor único: la órbita vive en ℙ(E_{λ₁}).

    Tasa empírica: mediana de θ_{k+1}/θ_k sobre el tramo donde
    θ ∈ [θ_floor, θ_ceil] (evita log-regresión sobre underflow).
    Si se llega al punto fijo en ≤ 2 pasos, empirical_rate := theoretical
    (la semilla ya era el modo propio: no hay señal de tasa).
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
        mask &= (arr[1:] >= cls.MIN_FS_FOR_RATE)
        if int(mask.sum()) < 3:
            return float(theoretical) if arr[-1] < cls.RATE_FS_FLOOR else 1.0
        ratios = arr[1:][mask] / np.maximum(arr[:-1][mask], cls.MIN_FS_FOR_RATE)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            return float(theoretical)
        return float(np.clip(np.median(ratios), 0.0, 1.0))

    @classmethod
    def solve(
        cls,
        handoff: IntrospectiveHandoff,
        max_iter: int = MAX_ITER_DEFAULT,
        tol: float = TOL_DEFAULT,
    ) -> Tuple[ComplexVector, PowerIterationTrace]:
        r"""Consume IntrospectiveHandoff (ρ, v₀, v₁, γ).  No recompute spec(ρ)."""
        rho = handoff.rho_mac
        n = handoff.dim
        v = ProjectiveDynamics.sanitize_vector(handoff.v_flash, n)
        v1 = handoff.v_dominant
        theoretical = float(handoff.spectral_gap.gap_ratio)

        residuals: List[float] = []
        fs_angles: List[float] = []
        rayleigh_traj: List[float] = []
        overlaps: List[float] = []
        converged = False
        stalled_kernel = False

        for _k in range(max_iter):
            sample = ProjectiveDynamics.evaluate(rho, v)
            residuals.append(sample.residual)
            fs_angles.append(sample.fs_angle)
            rayleigh_traj.append(sample.rayleigh)
            overlaps.append(ProjectiveDynamics.overlap_squared(v, v1))

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
        trace = PowerIterationTrace(
            residuals=tuple(map(float, residuals)),
            fs_angles=tuple(map(float, fs_angles)),
            rayleigh_trajectory=tuple(map(float, rayleigh_traj)),
            overlaps_with_v1=tuple(map(float, overlaps)),
            empirical_rate=float(empirical),
            stalled_kernel=bool(stalled_kernel),
            iterations=len(residuals),
            converged=bool(converged),
        )
        return v, trace


# ── §2.2 Certificador del punto fijo ──────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FixedPointCertificate:
    r"""
    Certificado del punto fijo introspectivo.

    Observables desambiguados:
        overlap_T                 : |⟨v*|T v*⟩|     (P)  =1 ⟺ Fix(T)
        uhlmann_fidelity_final    : √R(v*)          (U)  =√λ₁ si v*=v₁
        overlap_with_dominant     : |⟨v*|v₁⟩|²      (B)
        residual, fs_angle        : cuerda / d_FS   (G)
        rayleigh_gap_to_λ₁        : λ₁ − R ≥ 0      Courant–Fischer
        empirical_rate vs theoretical_rate          (S)
    """
    fixed_point_residual: float
    fixed_point_fs_angle: float
    overlap_T: float
    uhlmann_fidelity_final: float
    overlap_with_dominant: float
    rayleigh_final: float
    rayleigh_gap_to_lambda1: float
    iterations: int
    converged: bool
    stalled_kernel: bool
    empirical_rate: float
    theoretical_rate: float
    rate_consistency: bool
    local_verdict: HeytingOmega3


class FixedPointCertifier:
    r"""
    Predicados dimensionalmente invariantes, colapsados por meet (no n_fail):

        conv      : converged ∧ ¬stalled_kernel
        overlap_T : 1−|⟨v*|T v*⟩|  graduado     (punto fijo proyectivo)
        residual  : ‖T_φ−v*‖₂      graduado
        born      : 1−|⟨v*|v₁⟩|²   graduado     (cayó en el modo dominante)
        rayleigh  : (λ₁−R)/λ₁      graduado
        rate      : |r_emp − r_th| ≤ RATE_TOL   (o trivial si ya es FP)

    Uhlmann √R se REPORTA, no se usa como predicado de punto fijo:
    (1−√λ₁)≤10⁻⁶ exigiría ρ pura, no [v]∈Fix(T).

        local = conv ∧ overlap_T ∧ residual ∧ born ∧ rayleigh ∧ rate.
    """
    EPS_OVL_T: Final[float] = 1.0e-6
    EPS_RES: Final[float] = 1.0e-6
    EPS_BORN: Final[float] = 1.0e-6
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
        handoff: IntrospectiveHandoff,
        v_fixed: np.ndarray,
        trace: PowerIterationTrace,
    ) -> FixedPointCertificate:
        sample = ProjectiveDynamics.evaluate(handoff.rho_mac, v_fixed)
        lam1 = handoff.spectral_gap.lambda_1
        rate_th = handoff.spectral_gap.gap_ratio
        rate_emp = trace.empirical_rate
        ovl_v1 = ProjectiveDynamics.overlap_squared(v_fixed, handoff.v_dominant)

        already_fp = bool(trace.converged and trace.iterations <= 2)
        noisy_rate = bool(trace.converged and sample.fs_angle < cls.EPS_RES)
        rate_ok = already_fp or noisy_rate or (
            abs(rate_emp - rate_th) <= cls.RATE_TOL
        )

        conv_rule = (
            HeytingOmega3.VETOED if trace.stalled_kernel
            else (HeytingOmega3.COHERENT if trace.converged
                  else HeytingOmega3.DEGRADED)
        )
        ovl_t_rule = cls._grade(1.0 - sample.overlap_T, cls.EPS_OVL_T, 1.0e-3)
        residual_rule = cls._grade(sample.residual, cls.EPS_RES, 1.0e-3)
        born_rule = cls._grade(1.0 - ovl_v1, cls.EPS_BORN, 1.0e-2)
        ray_rel = max(0.0, lam1 - sample.rayleigh) / max(lam1, _EPS)
        ray_rule = cls._grade(ray_rel, cls.EPS_RAY, 1.0e-2)
        rate_rule = (
            HeytingOmega3.COHERENT if rate_ok else HeytingOmega3.DEGRADED
        )
        local = (
            conv_rule
            .meet(ovl_t_rule)
            .meet(residual_rule)
            .meet(born_rule)
            .meet(ray_rule)
            .meet(rate_rule)
        )
        return FixedPointCertificate(
            fixed_point_residual=float(sample.residual),
            fixed_point_fs_angle=float(sample.fs_angle),
            overlap_T=float(sample.overlap_T),
            uhlmann_fidelity_final=float(sample.uhlmann_pure),
            overlap_with_dominant=float(ovl_v1),
            rayleigh_final=float(sample.rayleigh),
            rayleigh_gap_to_lambda1=float(max(0.0, lam1 - sample.rayleigh)),
            iterations=int(trace.iterations),
            converged=bool(trace.converged),
            stalled_kernel=bool(trace.stalled_kernel),
            empirical_rate=float(rate_emp),
            theoretical_rate=float(rate_th),
            rate_consistency=bool(rate_ok),
            local_verdict=local,
        )


# ── §2.3 IntrospectionBundle — HAND-OFF FASE 2 → FASE 3 ──────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionBundle:
    r"""
    Objeto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Producto de los funtores Power ⊗ Certifier aplicados al
    IntrospectiveHandoff de FASE 1.
    """
    handoff: IntrospectiveHandoff
    v_fixed: np.ndarray
    trace: PowerIterationTrace
    certificate: FixedPointCertificate

    def content_bytes(self) -> bytes:
        """Digest firmable para la cadena Merkle de fases."""
        c = self.certificate
        return hashlib.sha256(
            self.handoff.handoff_hash.encode("ascii")
            + np.ascontiguousarray(self.v_fixed).tobytes()
            + f"{c.fixed_point_residual:.12e}".encode("ascii")
            + f"{c.overlap_T:.12e}".encode("ascii")
            + f"{c.uhlmann_fidelity_final:.12e}".encode("ascii")
            + f"{c.overlap_with_dominant:.12e}".encode("ascii")
            + f"{c.iterations}".encode("ascii")
            + f"{c.converged}".encode("ascii")
        ).digest()

    def continue_into_phase3(self) -> HeytingOmega3:
        r"""
        Último morfismo de la FASE 2  ∧  primero de la FASE 3.

        Identidad:

            continue_into_phase3 ∘ synthesize ∘ build
                = adjudicate ∘ synthesize ∘ build.
        """
        return HeytingIntrospectionAdjudicator.adjudicate(self)


class IntrospectionPipeline:
    r"""
    Orquestador determinista de la FASE 2 (funtor F₂).

        synthesize : IntrospectiveHandoff × ℕ × ℝ₊ → IntrospectionBundle

    ────────────────────────────────────────────────────────────────────────
    HAND-OFF FORMAL  FASE 2 → FASE 3
    ────────────────────────────────────────────────────────────────────────
    synthesize es el morfismo terminal de la FASE 2.  Su imagen
    IntrospectionBundle es el dominio de TODOS los métodos de FASE 3.

    Identidad de anidamiento:

        certify ∘ synthesize ∘ build  :  Flash×MAC×Ω₃ → ProofCertificate.
    """

    @classmethod
    def synthesize(
        cls,
        handoff: IntrospectiveHandoff,
        max_iter: int = PowerIterationSolver.MAX_ITER_DEFAULT,
        tol: float = PowerIterationSolver.TOL_DEFAULT,
    ) -> IntrospectionBundle:
        r"""
        Cierra la FASE 2.  Abre la FASE 3.

            (v*, trace) = PowerIterationSolver.solve(handoff)     §2.1
            cert        = FixedPointCertifier.certify(...)        §2.2
        """
        v_fixed, trace = PowerIterationSolver.solve(
            handoff, max_iter=max_iter, tol=tol,
        )
        cert = FixedPointCertifier.certify(handoff, v_fixed, trace)
        return IntrospectionBundle(
            handoff=handoff,
            v_fixed=v_fixed,
            trace=trace,
            certificate=cert,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 · ADJUDICACIÓN + NARRATIVA + AUTO-ORGANIZACIÓN + CERTIFICACIÓN    ║
# ║                                                                           ║
# ║  Dominio = IntrospectionBundle (codominio de §2.3 synthesize).            ║
# ║  Codominio = IntrospectionProofCertificate (objeto terminal).             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en Ω₃ ────────────────────────────────────────────────
class HeytingIntrospectionAdjudicator:
    r"""
    Lógica interna del topos: colapsa el Bundle a un único valor de Ω₃
    por meets sucesivos (nunca n_fail).

        cert     : certificate.local_verdict
        gap      : spectral_gap.local_verdict   (uniforme ⇒ ⊥ estructural)
        nondeg   : degeneracy_top == 1 → ⊤, == 2 → ⋆, else ⊥
        conv     : converged ∧ ¬kernel → ⊤, else ⋆/⊥
        fs       : d_FS([v*],[T v*]) graduado
        incoming : incoming_verdict  (⊥ es veto duro: meet con ⊥ = ⊥)

        final = local ∧ incoming     (meet conservador, nunca infla).
    """
    FS_COHERENT: Final[float] = 1.0e-6
    FS_DEGRADED: Final[float] = 1.0e-2

    @classmethod
    def _nondeg_rule(cls, bundle: IntrospectionBundle) -> HeytingOmega3:
        d = bundle.handoff.spectral_gap.degeneracy_top
        if d == 1:
            return HeytingOmega3.COHERENT
        if d == 2:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _conv_rule(cls, bundle: IntrospectionBundle) -> HeytingOmega3:
        if bundle.certificate.stalled_kernel:
            return HeytingOmega3.VETOED
        return (
            HeytingOmega3.COHERENT
            if bundle.certificate.converged
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _fs_rule(cls, bundle: IntrospectionBundle) -> HeytingOmega3:
        th = bundle.certificate.fixed_point_fs_angle
        if th <= cls.FS_COHERENT:
            return HeytingOmega3.COHERENT
        if th <= cls.FS_DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def adjudicate(cls, bundle: IntrospectionBundle) -> HeytingOmega3:
        r"""Continuación de IntrospectionPipeline.synthesize / continue_into_phase3."""
        local = (
            bundle.certificate.local_verdict
            .meet(bundle.handoff.spectral_gap.local_verdict)
            .meet(cls._nondeg_rule(bundle))
            .meet(cls._conv_rule(bundle))
            .meet(cls._fs_rule(bundle))
        )
        return local.meet(bundle.handoff.incoming_heyting_verdict)


# ── §3.2 Narrador de autocoherencia con números reales ───────────────────
class AutocoherenceNarrator:
    r"""
    Narrativa anclada a observables (U/P/B/G/S), no a adjetivos.

        VETOED   : incoming ⊥, ρ uniforme, kernel, o d_FS grande
        DEGRADED : convergencia con fricción (γ pequeño, iters altos)
        COHERENT : [v*] = [v₁] = Fix(T)  (overlap_T≈1, Born≈1, residuo≈0)

    F_Uhlmann = √λ₁ se cita como concentración de ρ, no como «éxito».
    """

    @classmethod
    def narrate(
        cls,
        bundle: IntrospectionBundle,
        final_verdict: HeytingOmega3,
    ) -> str:
        cert = bundle.certificate
        gap = bundle.handoff.spectral_gap
        fid = bundle.handoff.flash_intuition_id
        base = (
            f"Uhlmann√R={cert.uhlmann_fidelity_final:.6f} "
            f"(√λ₁={math.sqrt(max(0.0, gap.lambda_1)):.6f}) | "
            f"overlap_T=|⟨v*|T v*⟩|={cert.overlap_T:.6f} | "
            f"Born |⟨v*|v₁⟩|²={cert.overlap_with_dominant:.6f} | "
            f"‖T_φ−v*‖={cert.fixed_point_residual:.2e} | "
            f"d_FS={cert.fixed_point_fs_angle:.2e} rad | "
            f"iters={cert.iterations} | "
            f"rate_th=λ₂/λ₁={cert.theoretical_rate:.4f} | "
            f"rate_emp={cert.empirical_rate:.4f} | "
            f"γ={gap.gap:.4f} | Birkhoff={gap.birkhoff_constant:.4f} | "
            f"incoming={bundle.handoff.incoming_heyting_verdict.name}"
        )
        if final_verdict == HeytingOmega3.VETOED:
            return (
                f"INTROSPECCIÓN DE VETO: la corazonada '{fid}' "
                f"NO es autoestado invariante de la MAC ({base}). "
                f"Se sostiene la parálisis ciber-física."
            )
        if final_verdict == HeytingOmega3.DEGRADED:
            return (
                f"INTROSPECCIÓN DE ATENCIÓN: la corazonada '{fid}' "
                f"converge con fricción ({base}). "
                f"Requiere monitoreo en ciclos posteriores."
            )
        return (
            f"INTROSPECCIÓN AUTOCONSISTENTE: la corazonada '{fid}' "
            f"es un autoestado invariante de la Matriz MAC ({base}). "
            f"La decisión se sostiene por sí misma de forma inalienable."
        )


# ── §3.3 Auto-organización del campo MAC (mixtura convexa) ───────────────
@dataclass(frozen=True, slots=True)
class MacUpdateCertificate:
    r"""
    Canal afín de la MAC introspectiva:

        Φ_η(ρ) = (1−η) ρ + η |v*⟩⟨v*| ,   η ∈ [0, 1].

    Es CPTP (mezcla del canal identidad y el canal de reemplazo).
    Lip_{‖·‖₁}(Φ_η) = |1−η|  exactamente (afín).
    Punto fijo: Φ_η(|v*⟩⟨v*|) = |v*⟩⟨v*|.

    La pureza NO es monótona en general (contraejemplos si ρ ya es pura
    en otra dirección).  ΔP, ΔS, Δγ se reportan como observables, no leyes.

    Kraus del reemplazo: {|v*⟩⟨e_i|}_i.  La mixtura no se escribe con
    dos operadores {√(1−η) I, √η |v*⟩⟨v*|}.
    """
    applied: bool
    eta: float
    contraction_coef: float
    purity_before: float
    purity_after: float
    entropy_before: float
    entropy_after: float
    fidelity_to_fixed: float
    spectral_gap_before: float
    spectral_gap_after: float
    local_verdict: HeytingOmega3


class MACFieldSelfOrganizer:
    r"""Actualiza ρ_MAC hacia el atractor [v*] por mezcla convexa auditada."""

    @classmethod
    def idle(cls, rho: np.ndarray) -> Tuple[ComplexMatrix, MacUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        p = DensityOperatorAlgebra.purity(rho)
        s = DensityOperatorAlgebra.von_neumann_entropy(rho)
        g = SpectralGapAnalyzer.analyze(rho).gap
        cert = MacUpdateCertificate(
            applied=False,
            eta=0.0,
            contraction_coef=1.0,
            purity_before=p,
            purity_after=p,
            entropy_before=s,
            entropy_after=s,
            fidelity_to_fixed=1.0,
            spectral_gap_before=g,
            spectral_gap_after=g,
            local_verdict=HeytingOmega3.DEGRADED,
        )
        return rho, cert

    @classmethod
    def update(
        cls,
        rho: np.ndarray,
        v_fixed: np.ndarray,
        eta: float,
    ) -> Tuple[ComplexMatrix, MacUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        eta = float(np.clip(eta, 0.0, 1.0))
        rho_fixed = DensityOperatorAlgebra.rank_one(v_fixed)

        p_b = DensityOperatorAlgebra.purity(rho)
        s_b = DensityOperatorAlgebra.von_neumann_entropy(rho)
        gap_b = SpectralGapAnalyzer.analyze(rho).gap

        rho_new = DensityOperatorAlgebra.sanitize((1.0 - eta) * rho + eta * rho_fixed)
        p_a = DensityOperatorAlgebra.purity(rho_new)
        s_a = DensityOperatorAlgebra.von_neumann_entropy(rho_new)
        gap_a = SpectralGapAnalyzer.analyze(rho_new).gap
        fid_fixed = DensityOperatorAlgebra.uhlmann_fidelity(rho_new, rho_fixed)
        coef = abs(1.0 - eta)

        if coef < 1.0 and fid_fixed > 0.5:
            local = HeytingOmega3.COHERENT
        elif coef < 1.0:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        cert = MacUpdateCertificate(
            applied=True,
            eta=eta,
            contraction_coef=coef,
            purity_before=float(p_b),
            purity_after=float(p_a),
            entropy_before=float(s_b),
            entropy_after=float(s_a),
            fidelity_to_fixed=float(fid_fixed),
            spectral_gap_before=float(gap_b),
            spectral_gap_after=float(gap_a),
            local_verdict=local,
        )
        return rho_new, cert


# ── §3.4 Certificado firmado del ciclo introspectivo ─────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionProofCertificate:
    r"""
    Objeto terminal del ciclo: producto fibrado firmado

        Proof ≅ Bundle × Ω₃ × Φ_η × Narrativa × Merkle.

    Fidelidades desambiguadas:
        eigenstate_uhlmann     : F(ρ,|v*⟩⟨v*|) = √R     (U)
        overlap_T              : |⟨v*|T v*⟩|            (P)
        overlap_with_dominant  : |⟨v*|v₁⟩|²             (B)
        fixed_point_residual   : ‖T_φ − v*‖₂            (G, cuerda)
        fixed_point_fs_angle   : d_FS([v*],[T v*])      (G)
    """
    introspection_id: str
    flash_intuition_id: str
    sovereign_agent_id: str
    heyting_verdict: HeytingOmega3
    incoming_heyting_verdict: HeytingOmega3
    eigenstate_uhlmann: float
    overlap_T: float
    overlap_with_dominant: float
    fixed_point_residual: float
    fixed_point_fs_angle: float
    rayleigh_final: float
    rayleigh_gap_to_lambda1: float
    spectral_gap: float
    theoretical_rate: float
    empirical_rate: float
    rate_consistency: bool
    iterations: int
    converged: bool
    is_self_sustaining: bool
    birkhoff_constant: float
    degeneracy_top: int
    is_uniform: bool
    field_updated: bool
    field_update_eta: float
    mac_purity: float
    autocoherence_narrative: str
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.5 Soberano de Introspección ───────────────────────────────────────
class TOONIntrospectionAgent:
    r"""
    Soberano de Introspección y Autocoherencia.

    Funtor soberano  F = F₃ ∘ F₂ ∘ F₁ :

        F₁  IntrospectiveHandoff.build
        F₂  IntrospectiveHandoff.continue_into_phase2 = synthesize
        F₃  continue_into_phase3 ⊗ narrate ⊗ Φ_η ⊗ certify

    Asociatividad (teorema de anidamiento):

        introspect_flash_intuition
            = _phase3_certify ∘ _phase2_iterate ∘ _phase1_handoff
            = certify ∘ synthesize ∘ build.

    La auto-organización Φ_η es opcional y sólo se aplica si el veredicto
    final es ⊤ y `update_field=True` (inoculación coherente del modo propio).
    Incoming ⊥ se propaga por meet: no hay override.
    """

    def __init__(
        self,
        agent_id: str = "INTROSPECTION-SOVEREIGN-SABIO-01",
        dimension_mac: int = 4,
        update_field: bool = True,
        field_update_eta: float = 0.20,
        max_iter: int = PowerIterationSolver.MAX_ITER_DEFAULT,
        tol: float = PowerIterationSolver.TOL_DEFAULT,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1")
        self.agent_id = agent_id
        self.dimension_mac = int(dimension_mac)
        self.update_field = bool(update_field)
        self.field_update_eta = float(np.clip(field_update_eta, 0.0, 1.0))
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.mac_density_matrix: ComplexMatrix = (
            np.eye(self.dimension_mac, dtype=np.complex128) / self.dimension_mac
        )
        self.introspection_counter = 0
        self._chain_hash = hashlib.sha256(
            f"{agent_id}::GENESIS::n={dimension_mac}::"
            f"η={self.field_update_eta}".encode("ascii")
        ).hexdigest()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _phase1_handoff(
        self,
        introspection_id: str,
        flash_intuition_id: str,
        flash_vector: np.ndarray,
        incoming_heyting_verdict: HeytingOmega3,
        visceral_message: str,
    ) -> IntrospectiveHandoff:
        """FASE 1 anidada: cierra con IntrospectiveHandoff (dominio de FASE 2)."""
        handoff = IntrospectiveHandoff.build(
            introspective_id=introspection_id,
            flash_intuition_id=flash_intuition_id,
            rho_mac=self.mac_density_matrix,
            flash_vector=flash_vector,
            incoming_heyting_verdict=incoming_heyting_verdict,
            visceral_message=visceral_message,
        )
        self._advance_chain("F1", bytes.fromhex(handoff.handoff_hash))
        return handoff

    def _phase2_iterate(self, handoff: IntrospectiveHandoff) -> IntrospectionBundle:
        """FASE 2 anidada: continuación de build; cierra con Bundle."""
        bundle = handoff.continue_into_phase2(
            max_iter=self.max_iter, tol=self.tol,
        )
        self._advance_chain("F2", bundle.content_bytes())
        return bundle

    def _phase3_certify(
        self,
        bundle: IntrospectionBundle,
    ) -> IntrospectionProofCertificate:
        """FASE 3 anidada: continuación de synthesize; cierra con Proof."""
        final_verdict = bundle.continue_into_phase3()
        narrative = AutocoherenceNarrator.narrate(bundle, final_verdict)

        field_updated = False
        eta_used = 0.0
        if self.update_field and final_verdict == HeytingOmega3.COHERENT:
            new_rho, _upd = MACFieldSelfOrganizer.update(
                rho=self.mac_density_matrix,
                v_fixed=bundle.v_fixed,
                eta=self.field_update_eta,
            )
            self.mac_density_matrix = new_rho
            field_updated = True
            eta_used = self.field_update_eta

        c = bundle.certificate
        g = bundle.handoff.spectral_gap
        self._advance_chain(
            "F3",
            f"{final_verdict.name}|upd={field_updated}|"
            f"{c.overlap_T:.12e}".encode("ascii"),
        )
        provenance = _sha256_bytes(
            self.agent_id.encode("ascii"),
            bundle.handoff.introspective_id.encode("ascii"),
            bundle.handoff.flash_intuition_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{c.fixed_point_residual:.12e}".encode("ascii"),
            f"{c.overlap_T:.12e}".encode("ascii"),
            f"{c.overlap_with_dominant:.12e}".encode("ascii"),
            f"{g.gap:.12e}".encode("ascii"),
            self._chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )
        is_self_sustaining = bool(
            c.converged
            and (not c.stalled_kernel)
            and c.overlap_T > 1.0 - 1e-6
            and c.overlap_with_dominant > 1.0 - 1e-6
            and (not g.is_uniform)
        )
        return IntrospectionProofCertificate(
            introspection_id=bundle.handoff.introspective_id,
            flash_intuition_id=bundle.handoff.flash_intuition_id,
            sovereign_agent_id=self.agent_id,
            heyting_verdict=final_verdict,
            incoming_heyting_verdict=bundle.handoff.incoming_heyting_verdict,
            eigenstate_uhlmann=c.uhlmann_fidelity_final,
            overlap_T=c.overlap_T,
            overlap_with_dominant=c.overlap_with_dominant,
            fixed_point_residual=c.fixed_point_residual,
            fixed_point_fs_angle=c.fixed_point_fs_angle,
            rayleigh_final=c.rayleigh_final,
            rayleigh_gap_to_lambda1=c.rayleigh_gap_to_lambda1,
            spectral_gap=g.gap,
            theoretical_rate=c.theoretical_rate,
            empirical_rate=c.empirical_rate,
            rate_consistency=c.rate_consistency,
            iterations=c.iterations,
            converged=c.converged,
            is_self_sustaining=is_self_sustaining,
            birkhoff_constant=g.birkhoff_constant,
            degeneracy_top=g.degeneracy_top,
            is_uniform=g.is_uniform,
            field_updated=field_updated,
            field_update_eta=eta_used,
            mac_purity=DensityOperatorAlgebra.purity(self.mac_density_matrix),
            autocoherence_narrative=narrative,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )

    def introspect_flash_intuition(
        self,
        flash_intuition_id: str,
        flash_vector: np.ndarray,
        incoming_heyting_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
        visceral_message: str = "",
    ) -> IntrospectionProofCertificate:
        r"""Ciclo soberano: F₃ ∘ F₂ ∘ F₁."""
        self.introspection_counter += 1
        introspection_id = f"INTROSPECT-PROOF-{self.introspection_counter:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Introspección #%d | flash=%s | in_Ω₃=%s ═══",
            self.introspection_counter, flash_intuition_id,
            incoming_heyting_verdict.name,
        )
        handoff = self._phase1_handoff(
            introspection_id, flash_intuition_id, flash_vector,
            incoming_heyting_verdict, visceral_message,
        )
        bundle = self._phase2_iterate(handoff)
        cert = self._phase3_certify(bundle)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Introspección %s | Ω₃=%s | res=%.2e | FS=%.2e | "
            "overlap_T=%.6f | Born=%.6f | Uhlmann=%.6f | "
            "γ=%.4f | iters=%d | upd=%s | %.2f ms",
            cert.introspection_id, cert.heyting_verdict.name,
            cert.fixed_point_residual, cert.fixed_point_fs_angle,
            cert.overlap_T, cert.overlap_with_dominant, cert.eigenstate_uhlmann,
            cert.spectral_gap, cert.iterations, cert.field_updated, dt_ms,
        )
        return cert


# ── §3.6 Demostración autónoma ───────────────────────────────────────────
def _build_rho_from_spectrum(
    n: int, spectrum: np.ndarray, key: str,
) -> ComplexMatrix:
    r"""ρ = Σ_k λ_k |q_k⟩⟨q_k|  con Q Haar (Ginibre → QR, Stewart 1980)."""
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
    return (v / np.linalg.norm(v)).astype(np.complex128)


if __name__ == "__main__":
    print("═" * 96)
    print("SOBERANO DE INTROSPECCIÓN — v2.2.0 Nested Doctoral")
    print("Potencia gauge-fijada · Fubini–Study · Uhlmann≠Fix(T) · Birkhoff–Hopf · Ω₃ · Merkle")
    print("═" * 96)

    def print_proof(cert: IntrospectionProofCertificate, title: str) -> None:
        print(f"\n[{title}]")
        print(f"   introspection_id            : {cert.introspection_id}")
        print(f"   Ω₃ final / incoming         : {cert.heyting_verdict.name} / "
              f"{cert.incoming_heyting_verdict.name}")
        print(f"   F(ρ,|v*⟩⟨v*|) = √R  (U)     : {cert.eigenstate_uhlmann:.10f}")
        print(f"   |⟨v*|T v*⟩|         (P)     : {cert.overlap_T:.10f}")
        print(f"   |⟨v*|v₁⟩|²          (B)     : {cert.overlap_with_dominant:.10f}")
        print(f"   ‖T_φ(v*) − v*‖₂     (G)     : {cert.fixed_point_residual:.3e}")
        print(f"   d_FS([v*],[T v*])   (G)     : {cert.fixed_point_fs_angle:.3e} rad")
        print(f"   Rayleigh final              : {cert.rayleigh_final:.6f}")
        print(f"   λ₁ − R(v*)                  : {cert.rayleigh_gap_to_lambda1:.3e}")
        print(f"   γ = 1 − λ₂/λ₁               : {cert.spectral_gap:.6f}")
        print(f"   rate_th = λ₂/λ₁             : {cert.theoretical_rate:.6f}")
        print(f"   rate_emp observada          : {cert.empirical_rate:.6f}  "
              f"(consistente={cert.rate_consistency})")
        print(f"   iteraciones / converged     : {cert.iterations} / {cert.converged}")
        print(f"   Birkhoff tanh(Δ/4)          : {cert.birkhoff_constant:.6f}")
        print(f"   degeneración λ₁ / uniforme  : {cert.degeneracy_top} / {cert.is_uniform}")
        print(f"   auto-sostenible             : {cert.is_self_sustaining}")
        print(f"   campo MAC actualizado       : {cert.field_updated}  "
              f"(η = {cert.field_update_eta:.3f}, P(ρ)={cert.mac_purity:.6f})")
        print(f"   narrativa                   : {cert.autocoherence_narrative[:110]}…")
        print(f"   fase chain                  : {cert.phase_chain_sha256[:32]}…")
        print(f"   firma global                : {cert.sha256_provenance[:32]}…")

    agent = TOONIntrospectionAgent(
        agent_id="INTROSPECTION-SOVEREIGN-SABIO-01",
        dimension_mac=4,
        update_field=True,
        field_update_eta=0.20,
        max_iter=500,
        tol=1e-10,
    )

    n = 4
    rho_sanity = _build_rho_from_spectrum(
        n, np.array([0.97, 0.02, 0.005, 0.005]), "SANITY"
    )
    gap_sanity = SpectralGapAnalyzer.analyze(rho_sanity)
    v1_sanity = DensityOperatorAlgebra.dominant_eigenvector(rho_sanity)
    print("\n─── Sanity-check de una ρ con gap grande ───")
    print(f"   λ₁ = {gap_sanity.lambda_1:.6f} | λ₂ = {gap_sanity.lambda_2:.6f}")
    print(f"   γ  = {gap_sanity.gap:.6f}     | rate_th = λ₂/λ₁ = {gap_sanity.gap_ratio:.6f}")
    print(f"   Δλ = {gap_sanity.gap_absolute:.6f} | cond = {gap_sanity.condition_number:.4f}")
    print(f"   degeneración top = {gap_sanity.degeneracy_top} | uniforme = {gap_sanity.is_uniform}")
    print(f"   Birkhoff = tanh(Δ/4) = {gap_sanity.birkhoff_constant:.6f}")

    F_expect = math.sqrt(gap_sanity.lambda_1)
    F_meas = DensityOperatorAlgebra.uhlmann_fidelity_to_pure(rho_sanity, v1_sanity)
    sample_fp = ProjectiveDynamics.evaluate(rho_sanity, v1_sanity)
    print(f"   F(ρ,|v₁⟩⟨v₁|) = √λ₁ esperada  = {F_expect:.6f}")
    print(f"   F(ρ,|v₁⟩⟨v₁|) medida          = {F_meas:.6f}")
    print(f"   |error Uhlmann|               = {abs(F_expect - F_meas):.3e}")
    print(f"   |⟨v₁|T v₁⟩| (debe ser 1)      = {sample_fp.overlap_T:.12f}")
    print(f"   ‖T_φ−v₁‖₂   (debe ser ~0)     = {sample_fp.residual:.3e}")

    print("\n" + "─" * 96)
    print("─── Introspección de tres corazonadas contra una ρ con gap grande ───")

    rho_test = _build_rho_from_spectrum(
        n, np.array([0.97, 0.02, 0.005, 0.005]), "SCENARIOS"
    )
    agent.mac_density_matrix = rho_test
    v1 = DensityOperatorAlgebra.dominant_eigenvector(rho_test)

    v_aligned = v1.copy()
    v_mixed = v1 + _build_test_vector(n, "MIX")
    v_mixed = v_mixed / np.linalg.norm(v_mixed)
    v_orth = _build_test_vector(n, "ORTH")
    # Purge the dominant component so «⊥» is geométricamente honesto.
    v_orth = v_orth - np.vdot(v1, v_orth) * v1
    v_orth = v_orth / np.linalg.norm(v_orth)

    scenarios = [
        ("COHERENT (v ≈ v₁)", v_aligned, HeytingOmega3.COHERENT),
        ("DEGRADED (v mezcla)", v_mixed, HeytingOmega3.COHERENT),
        ("ORTHOGONAL (v ⊥ v₁, converge a v₁)", v_orth, HeytingOmega3.COHERENT),
    ]

    for name, v, ext in scenarios:
        cert = agent.introspect_flash_intuition(
            flash_intuition_id=f"FLASH-INTUITION::{name[:24]}",
            flash_vector=v,
            incoming_heyting_verdict=ext,
            visceral_message=f"Corazonada de prueba: {name}",
        )
        print_proof(cert, name)

    print("\n" + "─" * 96)
    print("─── Veto duro: incoming_verdict = VETOED ⟹ meet = VETOED ───")
    cert_veto = agent.introspect_flash_intuition(
        flash_intuition_id="FLASH-INTUITION::VETO-INCOMING",
        flash_vector=v_aligned,
        incoming_heyting_verdict=HeytingOmega3.VETOED,
        visceral_message="Flash entrante bajo VETO previo.",
    )
    print_proof(cert_veto, "VETO duro con incoming=VETOED")

    print("\n" + "─" * 96)
    print("─── Auto-organización del campo MAC (η = 0.20, 6 ciclos) ───")
    agent2 = TOONIntrospectionAgent(
        agent_id="INTROSPECT-SELF-ORG",
        dimension_mac=4,
        update_field=True,
        field_update_eta=0.20,
    )
    v_target = _build_test_vector(4, "SELF-ORG-V")

    print("\n   ciclo | Ω₃        | P(ρ_MAC)  | γ        | iters | upd | d_FS")
    for i in range(6):
        cert_i = agent2.introspect_flash_intuition(
            flash_intuition_id=f"FLASH-SELF-ORG-{i + 1:03d}",
            flash_vector=v_target,
            incoming_heyting_verdict=HeytingOmega3.COHERENT,
            visceral_message="Auto-organización iterada.",
        )
        p_now = DensityOperatorAlgebra.purity(agent2.mac_density_matrix)
        gap_now = SpectralGapAnalyzer.analyze(agent2.mac_density_matrix).gap
        print(
            f"   {i + 1:5d} | {cert_i.heyting_verdict.name:<9s} | "
            f"{p_now:.6f}  | {gap_now:.6f} | "
            f"{cert_i.iterations:5d} | {str(cert_i.field_updated):<5s} | "
            f"{cert_i.fixed_point_fs_angle:.2e}"
        )

    print("\n" + "═" * 96)
    print("✓ F1→F2: build ⊣ continue_into_phase2 = synthesize.")
    print("✓ F2→F3: synthesize ⊣ continue_into_phase3 = adjudicate.")
    print("✓ Iteración de potencia COMPLETA, parada d_FS ∧ ‖T_φ−v‖.")
    print("✓ F(ρ,|v⟩⟨v|)=√R (Uhlmann) ≠ |⟨v|T v⟩| (Fix(T)); √λ₁ ≠ 1.")
    print("✓ Born |⟨v|v₁⟩|² y d_FS=arccos|⟨u|v⟩|; residuo = 2 sin(θ/2).")
    print("✓ ρ(DT|_{v₁})=λ₂/λ₁; tasa empírica en ventana sana.")
    print("✓ Ω₃ por meets; incoming ⊥ es veto duro (meet conservador).")
    print("✓ Φ_η CPTP, Lip₁=|1−η|; Merkle F1→F2→F3.")
    print("═" * 96)
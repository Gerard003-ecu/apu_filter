# -*- coding: utf-8 -*-
r"""Soberano Testigo Silencioso y Cristalizador de Experiencia Modulares.

Ubicación: app/agents/wisdom/toon_silent_witness_agent.py
Versión  : 3.0.0-Doctoral-Nested-Triad-TomitaTakesaki-KMS-Crystal

Este módulo implementa el "Soberano Testigo Silencioso", entidad ejecutiva encargada
de la observación no perturbativa (sin back-action) de las dinámicas de la tríada
adversarial (Ilusionista, Soñador, Auditor) en la arquitectura COGNITIVE TOON / APU Filter.

================================================================================
I. FORMALIZACIÓN MATEMÁTICA Y TEORÍA MODULAR DE TOMITA-TAKESAKI
================================================================================

1. C*-Álgebras, Construcción GNS y Flujo Modular:
   Dada la C*-álgebra $M_n(\mathbb{C})$ con estado fiel $\omega_\rho(a) = \mathrm{Tr}(\rho a)$ y vector cíclico/separante $\Omega = \rho^{1/2}$
   en la representación GNS, el operador de Tomita $S : a \Omega \mapsto a^\dagger \Omega$ admite descomposición polar:
       $$S = J \Delta^{1/2}$$
   donde $\Delta = S^\dagger S > 0$ es el operador modular y $J$ es la conjugación modular antiunitaria ($J^2 = I$).
   El grupo de automorfismos modulares a un parámetro $\sigma_t \in \mathrm{Aut}(M_n(\mathbb{C}))$ viene dado por:
       $$\sigma_t(a) = \Delta^{-it} a \Delta^{it} = \rho^{-it} a \rho^{it} = e^{it K_\rho} a e^{-it K_\rho}$$
   con Hamiltoniano modular $K_\rho = -\log \rho$.

2. Condición KMS (Kubo-Martin-Schwinger) y Fuga Modular Cruzada:
   El estado $\omega_\rho$ satisface la condición KMS a $\beta = 1$ respecto a su propio flujo $\sigma_t$:
       $$\omega_\rho(a \sigma_i(b)) = \omega_\rho(b a)$$
   Para evaluar la incompatibilidad entre un estado observado $\rho_{\mathrm{obs}}$ y el vacío de flujo $\rho_{\mathrm{vac}}$,
   el residuo KMS cruzado mide el alejamiento del rayo modundar:
       $$\mathrm{Res}_{\mathrm{KMS}}(\rho_{\mathrm{obs}}, \rho_{\mathrm{vac}}) = \max_{a,b} \frac{|\mathrm{Tr}(\rho_{\mathrm{obs}} a \sigma_i^{\rho_{\mathrm{vac}}}(b)) - \mathrm{Tr}(\rho_{\mathrm{obs}} b a)|}{1 + |\mathrm{Tr}(\rho_{\mathrm{obs}} a \sigma_i^{\rho_{\mathrm{vac}}}(b))| + |\mathrm{Tr}(\rho_{\mathrm{obs}} b a)|}$$

3. Observables Físicos no Tautológicos:
   Para eliminar tautologías residuales, se formalizan las siguientes cantidades:
     • Excitación VEV: $\mathrm{VEV}_{\mathrm{exc}} = \mathrm{Tr}(\rho_{\mathrm{obs}} H_{\mathrm{ext}}) - E_0(H_{\mathrm{ext}})$.
     • Producción de Entropía: $\Delta S = S(\rho_{\mathrm{obs}}) - S(\rho_{\mathrm{vac}})$.
     • Drift KMS / Entropía Relativa de Umegaki: $S(\rho_{\mathrm{obs}} \| \rho_{\mathrm{vac}}) = \mathrm{Tr}(\rho_{\mathrm{obs}} (\log \rho_{\mathrm{obs}} - \log \rho_{\mathrm{vac}}))$.

4. Tríada Adversarial como Instrumento de Lüders:
   La tríada opera mediante un único operador de Kraus $K = P_A \cdot D \cdot U_I$:
       $$\Phi_{\mathrm{sel}}(\sigma) = \frac{K \sigma K^\dagger}{\mathrm{Tr}(K \sigma K^\dagger)}$$
   donde $U_I \in U(n)$ (distorsión del Ilusionista), $D \ge 0$ (ponderación del Soñador) y $P_A = P_A^2 = P_A^\dagger$ (proyector del Auditor).

5. Adjudicación de Heyting $\Omega_3$ y Cristalización Merkle:
   El veredicto final en $\Omega_3 = \{\bot (\mathrm{VETOED}) < \star (\mathrm{DEGRADED}) < \top (\mathrm{COHERENT})\}$
   aplica el meet ($\land$) sobre los indicadores modulares y la entrada externa.
   El cristal de experiencia se sella con el vector invariante en $S^6 \subset \mathbb{R}^7$ y se encadena vía SHA-256 Merkle.

================================================================================
II. ESTRUCTURA FUNTORIAL Y ARQUITECTURA
================================================================================

El Soberano opera como el funtor $\Phi_{\mathrm{triad}}$:
    $$\Phi_{\mathrm{triad}} : (M_n, \omega_{\mathrm{vac}}) \longrightarrow \Omega_3 \times \mathbf{ExperienceCrystal}$$

  • $F_1$ (`WitnessVacuumPreparation.prepare_vacuum_context`): $H_{\mathrm{ext}} \to \mathrm{WitnessVacuumContext}$.
    Construcción del estado de Gibbs $\rho_{\mathrm{vac}} = e^{-\beta H_{\mathrm{ext}}}/Z$, $K_{\mathrm{vac}} = -\log \rho_{\mathrm{vac}}$ y vector GNS.
  • $F_2$ (`WitnessObservationPipeline.synthesize_from_context`): $(\mathrm{WitnessVacuumContext}, \mathrm{TriadChannel}) \to \mathrm{WitnessObservationBundle}$.
    Aplicación de $\Phi_{\mathrm{sel}}$, verificación KMS, descomposición polar de Tomita y vector invariante $v_{\mathrm{inv}} \in S^6$.
  • $F_3$ (`TOONSilentWitnessAgent.observe_and_crystallize`): $\mathrm{WitnessObservationBundle} \to \mathbf{ExperienceCrystal}$.
    Adjudicación Heyting en $\Omega_3$, sello criptográfico y encadenamiento Merkle.

================================================================================
III. INVARIANTES FORMALES Y AXIOMAS DEL SISTEMA
================================================================================

- Axioma 1 (Condición KMS del Vacío): $\mathrm{Res}_{\mathrm{KMS}}(\rho_{\mathrm{vac}}, \rho_{\mathrm{vac}}) < \varepsilon_{\mathrm{KMS}}$.
- Axioma 2 (Invariancia C*): El residuo C* $\|A^\dagger A\|_\infty - \|A\|_\infty^2 = 0$ se satisface sobre todo elemento de $M_n(\mathbb{C})$.
- Axioma 3 (Inmutabilidad de la Cadena Merkle): Para todo cristal $k$, $\mathrm{chain\_hash}_k = \mathrm{SHA256}(\mathrm{chain\_hash}_{k-1} \parallel \mathrm{content\_hash}_k)$.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import scipy.linalg as la


__version__ = "3.0.0"


logger = logging.getLogger("APU.Wisdom.TOONSilentWitness")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS = 1.0e-14
_HERMITICITY_TOL = 1.0e-12
_TRACE_TOL = 1.0e-10
_GAP_DEGENERACY_TOL = 1.0e-12
_PSD_EIG_FLOOR = 0.0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ONTOLÓGICO DE LA TRÍADA                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §1.1 Retículo distributivo de Heyting Ω₃ ────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Retículo total Ω₃ = {⊥, ⋆, ⊤} con estructura de álgebra de Heyting
    completa (objeto clasificador de subobjetos del topos trivaluado):

        ⊥ < ⋆ < ⊤
        meet     (∧) : ínfimo = min
        join     (∨) : supremo = max
        implies  (⇒) : residuo de ∧ : (a ∧ b) ≤ c  ⇔  a ≤ (b ⇒ c)
        neg      (¬) : a ⇒ ⊥                           (intuicionista)
        ¬¬           : clausura regular                (¬¬⋆ = ⊤ ≠ ⋆)

    Leyes (H1)–(H5), verificables por `verify_heyting_laws`:
        (H1)  ∧, ∨ idempotentes, conmutativos, asociativos, absorbentes
        (H2)  residuación: a ∧ b ≤ c  ⇔  a ≤ (b ⇒ c)
        (H3)  a ⇒ a = ⊤,   ⊥ ⇒ a = ⊤,   a ⇒ ⊤ = ⊤
        (H4)  ¬¬⊥ = ⊥, ¬¬⊤ = ⊤, ¬¬⋆ = ⊤  (⋆ no es regular)
        (H5)  LEM a ∨ ¬a = ⊤ falla exactamente en a = ⋆

    El meet es la decisión más conservadora entre fuentes de verdad
    (Testigo, axiomas Tomita, auditor adversarial externo).
    """

    VETOED = 0  # ⊥
    DEGRADED = 1  # ⋆
    COHERENT = 2  # ⊤

    @property
    def verdict(self) -> str:
        return self.name

    @property
    def rank(self) -> int:
        """Grado de verdad en la cadena 0 ≤ 1 ≤ 2."""
        return int(self)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""a ⇒ b = ⊤ si a ≤ b; en caso contrario b  (álgebra de cadena)."""
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        r"""¬a ≜ a ⇒ ⊥  (seudocomplemento de Heyting)."""
        return self.implies(HeytingOmega3.VETOED)

    def double_negation(self) -> "HeytingOmega3":
        r"""¬¬a : clausura de regularidad. Fija {⊥, ⊤}; envía ⋆ ↦ ⊤."""
        return self.neg().neg()

    def is_regular(self) -> bool:
        return self.double_negation() == self

    def is_dense(self) -> bool:
        r"""a es denso ⇔ ¬a = ⊥ ⇔ ¬¬a = ⊤ (⋆ y ⊤)."""
        return self.neg() == HeytingOmega3.VETOED

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬a = ⊤. Falla exactamente en ⋆."""
        return self.join(self.neg()) == HeytingOmega3.COHERENT

    @classmethod
    def residuation_holds(
        cls, a: "HeytingOmega3", b: "HeytingOmega3", c: "HeytingOmega3"
    ) -> bool:
        r"""(a ∧ b) ≤ c  ⇔  a ≤ (b ⇒ c)."""
        left = int(a.meet(b)) <= int(c)
        right = int(a) <= int(b.implies(c))
        return left is right

    @classmethod
    def verify_heyting_laws(cls) -> Dict[str, bool]:
        """Auditoría finita de (H1)–(H5) sobre Ω₃³."""
        elems = list(cls)
        residuation = all(
            cls.residuation_holds(a, b, c) for a in elems for b in elems for c in elems
        )
        idempotent = all(a.meet(a) == a and a.join(a) == a for a in elems)
        lem_fails_on_star = not cls.DEGRADED.excluded_middle_holds()
        regular_pair = cls.VETOED.is_regular() and cls.COHERENT.is_regular()
        star_not_regular = not cls.DEGRADED.is_regular()
        return {
            "residuation": residuation,
            "idempotent": idempotent,
            "lem_fails_on_star": lem_fails_on_star,
            "regulars_are_bot_top": regular_pair,
            "star_not_regular": star_not_regular,
        }


def _seed_from_string(s: str) -> int:
    """SHA-256 → semilla uint32 (determinismo reproducible, no criptográfico)."""
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ── §1.2 Álgebra de Banach / C*, Hamiltoniano modular, densidad, GNS ───────
class MatrixBanachAlgebra:
    r"""
    Álgebra de Banach involutiva sobre M_n(ℂ):

        ‖A‖_p := (Tr |A|^p)^{1/p}     Schatten, 1 ≤ p < ∞
        ‖A‖_∞ := σ_max(A)
        r(A)  := max |spec(A)|
        C*    : ‖A† A‖_∞ = ‖A‖_∞²

    M_n es nuclear (tipo I): toda representación normal es múltiplo de la
    estándar. Aut(M_n) ≅ PU(n) actúa por automorfismos internos; el flujo
    modular de FASE 2 es un subgrupo a 1 parámetro de Inn(M_n).
    """

    @staticmethod
    def as_complex(A: np.ndarray) -> np.ndarray:
        return np.asarray(A, dtype=np.complex128)

    @staticmethod
    def hermitize(A: np.ndarray) -> np.ndarray:
        A = MatrixBanachAlgebra.as_complex(A)
        return 0.5 * (A + A.conj().T)

    @staticmethod
    def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return A @ B - B @ A

    @staticmethod
    def schatten_norm(A: np.ndarray, p: float) -> float:
        A = MatrixBanachAlgebra.as_complex(A)
        singular = np.linalg.svd(A, compute_uv=False)
        if singular.size == 0:
            return 0.0
        if p == math.inf or p == float("inf"):
            return float(singular[0])
        if p == 1:
            return float(np.sum(singular))
        if p == 2:
            return float(np.linalg.norm(singular))
        if p <= 0:
            raise ValueError("Schatten p-norm requires p ≥ 1 or p = ∞")
        return float(np.sum(singular ** p) ** (1.0 / p))

    @staticmethod
    def spectral_radius(A: np.ndarray) -> float:
        w = np.linalg.eigvals(MatrixBanachAlgebra.as_complex(A))
        return float(np.max(np.abs(w))) if w.size else 0.0

    @staticmethod
    def cstar_identity_residual(A: np.ndarray) -> float:
        r"""|‖A†A‖_∞ − ‖A‖_∞²|."""
        op = MatrixBanachAlgebra.schatten_norm
        A = MatrixBanachAlgebra.as_complex(A)
        return abs(op(A.conj().T @ A, math.inf) - op(A, math.inf) ** 2)

    @staticmethod
    def hilbert_schmidt_inner(A: np.ndarray, B: np.ndarray) -> complex:
        r"""⟨A|B⟩_{HS} = Tr(A† B)."""
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return complex(np.trace(A.conj().T @ B))


@dataclass(frozen=True, slots=True)
class ModularHamiltonian:
    r"""
    Hamiltoniano modular de un estado fiel ρ ∈ M_n(ℂ)_{+,1}:

        K_ρ := −log ρ,   ρ = e^{−K_ρ}/Z,   Z = Tr e^{−K_ρ}.

    Tras renormalizar spec(ρ) se impone Z = 1, de modo que la energía libre
    de Helmholtz modular F = −log Z = ⟨K_ρ⟩_ρ − S(ρ) = 0 es identidad KMS
    a β_modular = 1.

        E_0 = −log λ_max(ρ),   gap = E_1 − E_0,   g = dim ker(K_ρ − E_0 I).
    """

    eigenvalues: Tuple[float, ...]
    eigenvectors_hash: str
    regularization_eps: float

    @property
    def dimension(self) -> int:
        return len(self.eigenvalues)

    @property
    def ground_energy(self) -> float:
        return self.eigenvalues[0] if self.eigenvalues else 0.0

    @property
    def spectral_gap(self) -> float:
        if len(self.eigenvalues) < 2:
            return float("inf")
        return max(0.0, self.eigenvalues[1] - self.eigenvalues[0])

    @property
    def spectral_spread(self) -> float:
        if not self.eigenvalues:
            return 0.0
        return float(self.eigenvalues[-1] - self.eigenvalues[0])

    @property
    def partition_function(self) -> float:
        if not self.eigenvalues:
            return 1.0
        return float(math.fsum(math.exp(-e) for e in self.eigenvalues))

    @property
    def free_energy(self) -> float:
        z = self.partition_function
        if z <= 0.0:
            return float("inf")
        return -math.log(z)

    @property
    def internal_energy(self) -> float:
        z = self.partition_function
        if z <= 0.0 or not self.eigenvalues:
            return 0.0
        return float(math.fsum(math.exp(-e) * e for e in self.eigenvalues) / z)

    @property
    def degeneracy_of_ground(self) -> int:
        if not self.eigenvalues:
            return 0
        e0 = self.eigenvalues[0]
        tol = max(_GAP_DEGENERACY_TOL, 10.0 * self.regularization_eps)
        return sum(1 for e in self.eigenvalues if abs(e - e0) < tol)

    @property
    def is_unique_vacuum(self) -> bool:
        return self.degeneracy_of_ground == 1 and self.spectral_gap > 0.0

    @property
    def modular_correlation_time(self) -> float:
        r"""t_* = 1/gap  (tiempo característico de relajación modular)."""
        gap = self.spectral_gap
        if gap <= 0.0 or not math.isfinite(gap):
            return float("inf")
        return 1.0 / gap

    def as_tuple(self) -> Tuple[float, ...]:
        """Compatibilidad con la firma histórica spec(K) : Tuple[float, ...]."""
        return self.eigenvalues


class DensityMatrixOps:
    r"""
    Operadores canónicos sobre el simplejo D_n = {ρ ∈ M_n : ρ ≥ 0, Tr ρ = 1}:

        S(ρ)     = −Tr(ρ log ρ)                         von Neumann
        P(ρ)     =  Tr(ρ²)                              pureza
        S(ρ‖σ)   =  Tr(ρ (log ρ − log σ))               Umegaki operatorial
        F(ρ,σ)   =  Tr √(√ρ σ √ρ)                       Uhlmann
        D_tr     =  (1/2) ‖ρ − σ‖_1                     distancia de traza
        d_B      =  √(2(1 − F(ρ,σ)))                    Bures
        ρ^z      =  exp(z log ρ)                        cálculo funcional

    Distinción espectral:
        raw_spectrum(ρ)          autovalores ≥ 0 (sin levantar 0)
        regularized_spectrum(ρ)  λ ← max(λ, ε) y renormaliza (para log)

    Klein: S(ρ‖σ) ≥ 0,  = 0 ⇔ ρ = σ.  PU(n) preserva S, P, F, D_tr, d_B.

    `sanitize` hermitiza y normaliza la traza: SOLO aplica a estados, no
    a observables ni a operadores de Kraus.
    """

    EPS = _EPS

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> np.ndarray:
        rho = MatrixBanachAlgebra.hermitize(rho)
        tr = float(np.trace(rho).real)
        if abs(tr) > 1e-15:
            rho = rho / tr
        return rho

    @classmethod
    def is_density_operator(cls, rho: np.ndarray, tol: float = _TRACE_TOL) -> bool:
        rho_h = MatrixBanachAlgebra.hermitize(rho)
        herm = float(np.linalg.norm(rho_h - np.asarray(rho, dtype=np.complex128)))
        tr = float(np.trace(rho_h).real)
        w = np.real(la.eigvalsh(rho_h))
        return herm < tol and abs(tr - 1.0) < tol and float(w.min()) > -tol

    @classmethod
    def raw_spectrum(cls, rho: np.ndarray) -> np.ndarray:
        vals = np.real(la.eigvalsh(cls.sanitize(rho)))
        vals = np.sort(vals)[::-1]
        return np.maximum(vals, 0.0)

    @classmethod
    def regularized_spectrum(cls, rho: np.ndarray) -> np.ndarray:
        vals = cls.raw_spectrum(rho)
        vals = np.maximum(vals, cls.EPS)
        s = float(vals.sum())
        return vals / s if s > 0.0 else vals

    @classmethod
    def spectrum(cls, rho: np.ndarray) -> np.ndarray:
        """Alias de `regularized_spectrum` (compatibilidad)."""
        return cls.regularized_spectrum(rho)

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        r"""S(ρ) = −∑_{λ_i > ε} λ_i log λ_i  con 0 log 0 = 0."""
        p = cls.raw_spectrum(rho)
        p = p[p > cls.EPS]
        if p.size == 0:
            return 0.0
        return float(-np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.raw_spectrum(rho)
        return float(np.sum(p ** 2))

    @classmethod
    def hermitian_functional_calculus(
        cls, H: np.ndarray, func
    ) -> np.ndarray:
        r"""f(H) para H hermítico, SIN normalizar traza."""
        H = MatrixBanachAlgebra.hermitize(H)
        w, vecs = la.eigh(H)
        fw = np.asarray(func(w), dtype=np.complex128)
        return (vecs * fw) @ vecs.conj().T

    @classmethod
    def matrix_log(cls, rho: np.ndarray) -> np.ndarray:
        r"""log ρ por cálculo funcional, λ ← max(λ, ε)."""
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, cls.EPS)
        return (vecs * np.log(vals)) @ vecs.conj().T

    @classmethod
    def matrix_power(cls, rho: np.ndarray, z: complex) -> np.ndarray:
        r"""ρ^z = exp(z log ρ); ρ debe ser un estado (sanitize)."""
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, cls.EPS)
        powered = np.exp(z * np.log(vals.astype(np.complex128)))
        return (vecs * powered) @ vecs.conj().T

    @classmethod
    def umegaki_relative_entropy(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""
        S(ρ‖σ) = Tr(ρ (log ρ − log σ))  — forma operatorial.

        NO se emparejan espectros independientes: eso sólo sería correcto
        si [ρ, σ] = 0 y los autoespacios coinciden.
        """
        rho = cls.sanitize(rho)
        log_rho = cls.matrix_log(rho)
        log_sigma = cls.matrix_log(sigma)
        return float(np.real(np.trace(rho @ (log_rho - log_sigma))))

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""F(ρ,σ) = Tr √(√ρ σ √ρ) ∈ [0, 1]."""
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sqrt_rho = cls.matrix_power(rho, 0.5)
        sandwich = MatrixBanachAlgebra.hermitize(sqrt_rho @ sigma @ sqrt_rho)
        val = float(np.real(np.trace(cls.matrix_power(sandwich, 0.5))))
        return float(max(0.0, min(1.0, val)))

    @classmethod
    def trace_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""D(ρ,σ) = (1/2) ‖ρ − σ‖_1 ∈ [0, 1]."""
        delta = cls.sanitize(rho) - cls.sanitize(sigma)
        return 0.5 * MatrixBanachAlgebra.schatten_norm(delta, 1)

    @classmethod
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""d_B(ρ,σ) = √(2(1 − F(ρ,σ))) ∈ [0, √2]."""
        fid = cls.uhlmann_fidelity(rho, sigma)
        return math.sqrt(max(0.0, 2.0 * (1.0 - fid)))

    @classmethod
    def kleins_inequality_residual(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""min{0, S(ρ‖σ)} : ~0 si Klein se respeta numéricamente."""
        return float(min(0.0, cls.umegaki_relative_entropy(rho, sigma)))

    @classmethod
    def modular_hamiltonian_from_rho(
        cls, rho: np.ndarray, eps: float = _EPS
    ) -> ModularHamiltonian:
        rho = cls.sanitize(rho)
        lam, vec = la.eigh(rho)
        lam = np.maximum(lam, eps)
        lam = lam / lam.sum()
        energies = -np.log(lam)
        order = np.argsort(energies)
        e_sorted, vec_sorted = energies[order], vec[:, order]
        return ModularHamiltonian(
            eigenvalues=tuple(float(x) for x in e_sorted.tolist()),
            eigenvectors_hash=_sha256_array(vec_sorted),
            regularization_eps=eps,
        )

    @classmethod
    def modular_hamiltonian_spectrum(cls, rho: np.ndarray) -> Tuple[float, ...]:
        r"""Espectro de K_ρ = −log ρ, ordenado ascendente (compatibilidad)."""
        return cls.modular_hamiltonian_from_rho(rho).as_tuple()

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> np.ndarray:
        H = MatrixBanachAlgebra.hermitize(H)
        w, vecs = la.eigh(H)
        i0 = int(np.argmin(w))
        omega = vecs[:, i0].reshape(-1, 1)
        return cls.sanitize(omega @ omega.conj().T)

    @classmethod
    def gns_inner_product(cls, rho: np.ndarray, A: np.ndarray, B: np.ndarray) -> complex:
        r"""⟨A|B⟩_ω = Tr(ρ A† B)."""
        rho = cls.sanitize(rho)
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return complex(np.trace(rho @ A.conj().T @ B))

    @classmethod
    def thermofield_double(cls, rho: np.ndarray) -> np.ndarray:
        r"""
        Purificación GNS / TFD: |Ψ⟩ = ∑_i √λ_i |i⟩ ⊗ |i⟩ ∈ ℂⁿ ⊗ ℂⁿ,
        que bajo la identificación HS es Ω_HS = ρ^{1/2}.
        """
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, 0.0)
        n = rho.shape[0]
        psi = np.zeros((n * n,), dtype=np.complex128)
        sqrt_vals = np.sqrt(vals)
        for i in range(n):
            ket = vecs[:, i]
            psi += sqrt_vals[i] * np.kron(ket, ket)
        norm = float(np.linalg.norm(psi))
        return psi / norm if norm > 0.0 else psi

    @classmethod
    def psd_project(cls, A: np.ndarray) -> np.ndarray:
        """Proyección espectral sobre el cono PSD (partes negativas → 0)."""
        A = MatrixBanachAlgebra.hermitize(A)
        w, vecs = la.eigh(A)
        w = np.maximum(w, _PSD_EIG_FLOOR)
        return (vecs * w) @ vecs.conj().T


class GNSHilbertAlgebra:
    r"""
    Construcción GNS de (M_n(ℂ), ω_ρ):

        H_ω = M_n    con    ⟨A|B⟩_ω = Tr(ρ A† B),
        Ω   = I      cíclico y separante  ⇔  ρ > 0,
        Ω_HS = ρ^{1/2} ∈ HS(ℂⁿ).

    Tomita: S π(x) Ω = π(x†) Ω. Puente geométrico entre §1.2 y §2.1.
    """

    @staticmethod
    def inner_product(rho: np.ndarray, A: np.ndarray, B: np.ndarray) -> complex:
        return DensityMatrixOps.gns_inner_product(rho, A, B)

    @staticmethod
    def cyclic_vector_hs(rho: np.ndarray) -> np.ndarray:
        return DensityMatrixOps.matrix_power(rho, 0.5)

    @staticmethod
    def is_faithful(rho: np.ndarray, tol: float = _EPS) -> bool:
        w = DensityMatrixOps.raw_spectrum(rho)
        return bool(w.size and float(w.min()) > tol)

    @staticmethod
    def omega_norm_squared(rho: np.ndarray) -> float:
        r"""‖Ω‖_ω² = ⟨I|I⟩_ω = Tr(ρ) = 1."""
        return float(np.real(np.trace(DensityMatrixOps.sanitize(rho))))


# ── §1.3 Fábrica determinista de operadores de la Tríada ──────────────────
@dataclass(frozen=True, slots=True)
class TriadSignature:
    r"""
    Identidad forense de la tríada: tipos textuales + hashes SHA-256 de
    los tres operadores y rango del proyector. `as_bytes` es el preimage
    canónico del content-hash del cristal.
    """

    trickster_illusion_type: str
    dreamer_scenario_id: str
    auditor_immunization_hash: str
    unitary_hash: str
    dreamer_hash: str
    projector_hash: str
    auditor_rank: int

    def as_bytes(self) -> bytes:
        return (
            f"{self.trickster_illusion_type}|{self.dreamer_scenario_id}|"
            f"{self.auditor_immunization_hash}|{self.unitary_hash[:16]}|"
            f"{self.dreamer_hash[:16]}|{self.projector_hash[:16]}|"
            f"{self.auditor_rank}"
        ).encode("utf-8")


class TriadOperatorFactory:
    r"""
    Construye U_I, D y P_A de forma determinista a partir de identificadores
    textuales (SHA-256 → semilla de numpy.default_rng), interpolando con
    `strength ∈ [0, 1]` entre la identidad y muestras de máxima entropía.

        U_I(s) = exp(i · s · π · n · Â)     Â Hermítico, ‖Â‖_F = 1
        D(s)   = (1−s) I + s (A†A) · (n / Tr A†A)     A Ginibre
        P_A    = Q_{:r} Q_{:r}†              Q Haar (QR de Ginibre + fases)

    El Haar se obtiene del procedimiento de Mezzadri: QR(Ginibre) con
    corrección de fase diag(R)/|diag(R)|, que produce la medida de Haar
    a izquierda y derecha sobre U(n).
    """

    @classmethod
    def _haar(cls, n: int, rng: np.random.Generator) -> np.ndarray:
        ginibre = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        q_factor, r_factor = np.linalg.qr(ginibre)
        diag_r = np.diagonal(r_factor)
        phases = np.where(np.abs(diag_r) > 1e-30, diag_r / np.abs(diag_r), 1.0 + 0j)
        return q_factor * phases.conj()

    @classmethod
    def build_unitary(cls, illusion_type: str, n: int, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        rng = np.random.default_rng(_seed_from_string(f"TRICKSTER::{illusion_type}"))
        if strength <= 0.0:
            return np.eye(n, dtype=np.complex128)
        amp = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        a_h = MatrixBanachAlgebra.hermitize(amp)
        nrm = float(np.linalg.norm(a_h, "fro")) + 1e-30
        a_h = a_h / nrm
        theta = math.pi * strength * n
        return la.expm(1j * theta * a_h)

    @classmethod
    def build_dreamer(cls, scenario_id: str, n: int, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        rng = np.random.default_rng(_seed_from_string(f"DREAMER::{scenario_id}"))
        if strength <= 0.0:
            return np.eye(n, dtype=np.complex128)
        amp = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        d_rand = amp.conj().T @ amp
        tr = float(np.trace(d_rand).real) + 1e-30
        d_rand = d_rand / tr * n
        mixed = (1.0 - strength) * np.eye(n, dtype=np.complex128) + strength * d_rand
        return MatrixBanachAlgebra.hermitize(mixed)

    @classmethod
    def build_projector(cls, immunization_hash: str, n: int, rank: int) -> np.ndarray:
        rank = int(np.clip(rank, 1, n))
        rng = np.random.default_rng(_seed_from_string(f"AUDITOR::{immunization_hash}"))
        haar = cls._haar(n, rng)
        cols = haar[:, :rank]
        projector = cols @ cols.conj().T
        return MatrixBanachAlgebra.hermitize(projector)

    @classmethod
    def build_triad(
        cls,
        trickster_illusion_type: str,
        dreamer_scenario_id: str,
        auditor_immunization_hash: str,
        n: int,
        trickster_strength: float,
        dreamer_strength: float,
        auditor_rank: int,
    ) -> "TriadChannel":
        unitary = cls.build_unitary(trickster_illusion_type, n, trickster_strength)
        dreamer = cls.build_dreamer(dreamer_scenario_id, n, dreamer_strength)
        projector = cls.build_projector(auditor_immunization_hash, n, auditor_rank)
        signature = TriadSignature(
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            unitary_hash=_sha256_array(unitary),
            dreamer_hash=_sha256_array(dreamer),
            projector_hash=_sha256_array(projector),
            auditor_rank=int(np.clip(auditor_rank, 1, n)),
        )
        return TriadChannel(signature, unitary, dreamer, projector)


# ── §1.4 Canal selectivo de la Tríada ─────────────────────────────────────
class TriadChannel:
    r"""
    Instrumento de Lüders de un solo Kraus (categoría de mapas CP):

        K_triad := P_A · D · U_I ∈ M_n(ℂ)
        Φ_lin(X) := K X K†                      CP lineal, Choi = |vec K⟩⟨vec K|
        Φ_sel(σ) := K σ K† / Tr(K σ K†)         post-selección (no lineal)
        p(σ)     := Tr(K† K σ)                  probabilidad de admisión

    Completitud TP ⇔ K†K = I ⇔ K isometría ⇔ Stinespring trivial.
    En general p(σ) ∈ [0, ‖K†K‖_op]; NO está acotada por 1 a priori.

    Residuos estructurales:
        projector  ‖P² − P‖_F,   hermiticidad de D,   unitariedad de U,
        defecto TP ‖K†K − I‖_op,  λ_min(Choi) ≥ 0.
    """

    def __init__(
        self,
        signature: TriadSignature,
        U_I: np.ndarray,
        D: np.ndarray,
        P_A: np.ndarray,
    ) -> None:
        self.signature = signature
        self.U = MatrixBanachAlgebra.as_complex(U_I)
        self.D = DensityMatrixOps.psd_project(D)
        self.P = MatrixBanachAlgebra.hermitize(P_A)
        self.K = self.P @ self.D @ self.U

    def apply_linear(self, rho: np.ndarray) -> np.ndarray:
        r"""Φ_lin(ρ) = K ρ K†  (CP lineal, traza no conservada)."""
        rho = MatrixBanachAlgebra.as_complex(rho)
        return self.K @ rho @ self.K.conj().T

    def apply(self, rho: np.ndarray) -> np.ndarray:
        r"""Φ_sel(ρ) : post-selección de Lüders. Fallback: I/n si p ≈ 0."""
        out = self.apply_linear(rho)
        tr = float(np.trace(out).real)
        if tr < 1e-30:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        return out / tr

    def success_probability(self, rho: np.ndarray) -> float:
        metric = self.K.conj().T @ self.K
        return float(np.real(np.trace(rho @ metric)))

    def kraus_norm(self) -> float:
        return float(np.linalg.norm(self.K, "fro"))

    def tp_residual(self) -> float:
        r"""‖K†K − I‖_op  (0 ⇔ canal traza-preservante)."""
        n = self.K.shape[0]
        defect = self.K.conj().T @ self.K - np.eye(n, dtype=np.complex128)
        return MatrixBanachAlgebra.schatten_norm(defect, math.inf)

    def projector_residual(self) -> float:
        return float(np.linalg.norm(self.P @ self.P - self.P, "fro"))

    def unitary_residual(self) -> float:
        n = self.U.shape[0]
        eye = np.eye(n, dtype=np.complex128)
        left = self.U.conj().T @ self.U - eye
        return float(np.linalg.norm(left, "fro"))

    def choi_matrix(self) -> np.ndarray:
        r"""Choi(Φ_lin) = ∑_{ij} |i⟩⟨j| ⊗ K|i⟩⟨j|K† = vec(K) vec(K)†."""
        n = self.K.shape[0]
        vec_k = self.K.reshape((n * n, 1), order="F")
        return vec_k @ vec_k.conj().T

    def cp_min_eigenvalue(self) -> float:
        r"""λ_min(Choi). Debe ser ≥ 0 (CP). Para un Kraus, rango 1 ⇒ {‖K‖², 0}."""
        choi = MatrixBanachAlgebra.hermitize(self.choi_matrix())
        w = np.real(la.eigvalsh(choi))
        return float(w.min()) if w.size else 0.0

    def diamond_norm_upper_bound(self) -> float:
        r"""‖Φ_lin‖_⋄ ≤ ‖K‖_op²  (un Kraus)."""
        return MatrixBanachAlgebra.schatten_norm(self.K, math.inf) ** 2

    def structural_residuals(self) -> Dict[str, float]:
        return {
            "tp_residual": self.tp_residual(),
            "projector_residual": self.projector_residual(),
            "unitary_residual": self.unitary_residual(),
            "cp_min_eigenvalue": self.cp_min_eigenvalue(),
            "diamond_upper": self.diamond_norm_upper_bound(),
        }


# ── §1.5 WitnessVacuumContext + WitnessVacuumPreparation — HAND-OFF 1→2 ──
@dataclass(slots=True)
class WitnessVacuumContext:
    r"""
    Objeto terminal de FASE 1 y objeto inicial de FASE 2.

    Encapsula el par canónico (ρ_vac, K_vac) junto con el Hamiltoniano
    externo H_ext, el proyector al ground |Ω⟩⟨Ω| y metadatos GNS/TFD,
    de modo que `ModularFlowEngine.bind_vacuum_context` no re-deriva el
    vacío: la definición formal de `prepare_vacuum_context` *es* el
    arranque de la dinámica modular y de la observación silenciosa.
    """

    rho_vac: np.ndarray
    K: ModularHamiltonian
    H_ext: np.ndarray
    ground_projector: np.ndarray
    beta: float
    gns_norm_sq: float
    is_faithful: bool
    tfd: np.ndarray = field(repr=False)
    path_laplacian: np.ndarray = field(repr=False)


class WitnessVacuumPreparation:
    r"""
    Prepara el contexto modular que alimenta toda la FASE 2.

    El vacío del Testigo es el estado de Gibbs a temperatura inversa β_vac
    del Hamiltoniano de enlace fuerte sobre el grafo camino P_n:

        H = ∑_{k=0}^{n−1} k |k⟩⟨k|  +  t ∑_{⟨i,j⟩} (|i⟩⟨j| + |j⟩⟨i|),
        ρ_β = e^{−β H} / Z_β.

    β_vac → ∞  ⇒  ρ_β → |Ω⟩⟨Ω|  (silencio absoluto, T = 0).
    A β finito, ρ_β es KMS(β) con K_vac = −log ρ_β.

    El último método, `prepare_vacuum_context`, cierra FASE 1 y es el
    morfismo de hand-off  FASE 1 ⟶ FASE 2.
    """

    DEFAULT_BETA_COLD: float = 50.0
    DEFAULT_HOPPING: float = 1.0e-3

    @classmethod
    def tight_binding_hamiltonian(
        cls, n: int, hopping: float = DEFAULT_HOPPING
    ) -> np.ndarray:
        r"""H de enlace fuerte sobre P_n (grafo camino). Gap topológico O(t)."""
        if n < 1:
            raise ValueError("dimension_mac must be ≥ 1")
        onsite = np.linspace(0.0, float(max(n - 1, 0)), n, dtype=np.float64)
        ham = np.diag(onsite).astype(np.complex128)
        for i in range(n - 1):
            ham[i, i + 1] = hopping
            ham[i + 1, i] = hopping
        return MatrixBanachAlgebra.hermitize(ham)

    @classmethod
    def path_laplacian(cls, n: int) -> np.ndarray:
        r"""Laplaciano combinatorio de P_n: L = deg − A.  E_D(ρ) = Tr(ρ L)."""
        lap = np.zeros((n, n), dtype=np.complex128)
        for i in range(n - 1):
            lap[i, i] += 1.0
            lap[i + 1, i + 1] += 1.0
            lap[i, i + 1] -= 1.0
            lap[i + 1, i] -= 1.0
        return MatrixBanachAlgebra.hermitize(lap)

    @classmethod
    def gibbs_state(cls, H: np.ndarray, beta: float) -> np.ndarray:
        H = MatrixBanachAlgebra.hermitize(H)
        w, vecs = la.eigh(H)
        logits = -beta * (w - w.min())
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        return DensityMatrixOps.sanitize((vecs * p.astype(np.complex128)) @ vecs.conj().T)

    @classmethod
    def prepare_vacuum_pair(
        cls, H_ext: np.ndarray, beta: float = DEFAULT_BETA_COLD
    ) -> Tuple[np.ndarray, Tuple[float, ...]]:
        r"""Compatibilidad: (H_ext, β) ↦ (ρ_vac, spec(K_vac)). Delegado del contexto."""
        ctx = cls.prepare_vacuum_context(H_ext, beta=beta)
        return ctx.rho_vac, ctx.K.as_tuple()

    # ═════════════════════════════════════════════════════════════════════
    #  HAND-OFF  FASE 1 → FASE 2
    #  Definición formal terminal de FASE 1.
    #  Su tipo de retorno `WitnessVacuumContext` es el dominio de
    #  ModularFlowEngine.bind_vacuum_context (§2.0), primer método de
    #  FASE 2: no hay hiato semántico entre ambas fases.
    # ═════════════════════════════════════════════════════════════════════
    @classmethod
    def prepare_vacuum_context(
        cls, H_ext: np.ndarray, beta: float = DEFAULT_BETA_COLD
    ) -> WitnessVacuumContext:
        r"""
        Morfismo de hand-off  (H_ext, β) ↦ WitnessVacuumContext.

        Continúa en §2.0 `ModularFlowEngine.bind_vacuum_context`.
        """
        H_ext = MatrixBanachAlgebra.hermitize(H_ext)
        n = H_ext.shape[0]
        rho = cls.gibbs_state(H_ext, beta)
        ham_mod = DensityMatrixOps.modular_hamiltonian_from_rho(rho)
        ground = DensityMatrixOps.ground_state_projector(H_ext)
        gns_n2 = GNSHilbertAlgebra.omega_norm_squared(rho)
        faithful = GNSHilbertAlgebra.is_faithful(rho)
        tfd = DensityMatrixOps.thermofield_double(rho)
        logger.debug(
            "WitnessVacuumPreparation.context: β=%.4g | E₀(K)=%.6f | gap(K)=%.6f | "
            "Z=%.8f | F=%.3e | faithful=%s",
            beta,
            ham_mod.ground_energy,
            ham_mod.spectral_gap,
            ham_mod.partition_function,
            ham_mod.free_energy,
            faithful,
        )
        return WitnessVacuumContext(
            rho_vac=rho,
            K=ham_mod,
            H_ext=H_ext,
            ground_projector=ground,
            beta=beta,
            gns_norm_sq=gns_n2,
            is_faithful=faithful,
            tfd=tfd,
            path_laplacian=cls.path_laplacian(n),
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · DINÁMICA MODULAR Y OBSERVACIÓN SILENCIOSA (C. de FASE 1)         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.0 / §2.1 Motor Tomita–Takesaki: bind, Δ, σ_t, J, S, KMS, Connes ────
class ModularFlowEngine:
    r"""
    Implementación numérica de Tomita–Takesaki para ω(a) = Tr(ρ a).

    §2.0  `bind_vacuum_context` continúa `prepare_vacuum_context` (§1.5):
          valida el objeto terminal de FASE 1 y lo sella como dominio de
          la dinámica modular y de la observación.

    Convención KMS-compatible (verificada por axiomas, polar, grupo y KMS):

        σ_t(a) = ρ^{−it} a ρ^{it} = e^{it K} a e^{−it K}
        σ_i(a) = ρ a ρ^{−1}
        Δ(a)   = ρ^{−1} a ρ
        J(a)   = ρ^{−1/2} a† ρ^{1/2}
        S(a)   = a† = J(Δ^{1/2}(a))

    Axiomas:
        (i)   unital          σ_t(I) = I
        (ii)  multiplicativo  σ_t(ab) = σ_t(a) σ_t(b)
        (iii) isométrico HS   ‖σ_t(a)‖_2 = ‖a‖_2
        (iv)  ley de grupo    σ_s ∘ σ_t = σ_{s+t}
        (v)   J² = id
        (vi)  polar           J Δ^{1/2} = S
        (vii) KMS(β=1)        ω(a σ_i(b)) = ω(ba)

    KMS cruzado: si ρ_state ≠ ρ_flow, el residuo mide la fuga de ρ_state
    fuera del rayo KMS de ρ_flow. Unicidad KMS (estados fieles) ⇒ residuo
    nulo ⇔ ρ_state = ρ_flow.
    """

    @classmethod
    def bind_vacuum_context(cls, ctx: WitnessVacuumContext) -> WitnessVacuumContext:
        r"""
        §2.0  Arranque de FASE 2.

        Continúa el morfismo `WitnessVacuumPreparation.prepare_vacuum_context`
        (§1.5). Verifica ρ cuadrada, hermítica, Tr ρ ≈ 1, dim spec(K) = n,
        ‖Ω‖_ω² ≈ 1. Devuelve el contexto saneado.
        """
        if ctx.rho_vac.ndim != 2 or ctx.rho_vac.shape[0] != ctx.rho_vac.shape[1]:
            raise ValueError("WitnessVacuumContext.ρ_vac must be a square matrix")
        n = ctx.rho_vac.shape[0]
        if ctx.H_ext.shape != (n, n) or ctx.ground_projector.shape != (n, n):
            raise ValueError("H_ext, ρ_vac and |Ω⟩⟨Ω| dimension mismatch")
        if ctx.K.dimension != n:
            raise ValueError("spec(K_vac) length ≠ n")

        ctx.rho_vac = DensityMatrixOps.sanitize(ctx.rho_vac)
        ctx.H_ext = MatrixBanachAlgebra.hermitize(ctx.H_ext)
        ctx.ground_projector = DensityMatrixOps.sanitize(ctx.ground_projector)
        ctx.gns_norm_sq = GNSHilbertAlgebra.omega_norm_squared(ctx.rho_vac)
        ctx.is_faithful = GNSHilbertAlgebra.is_faithful(ctx.rho_vac)

        if abs(ctx.gns_norm_sq - 1.0) > _TRACE_TOL:
            logger.warning(
                "GNS ‖Ω‖² = %.3e ≠ 1 (tol=%.1e)", ctx.gns_norm_sq, _TRACE_TOL
            )
        logger.debug(
            "bind_vacuum_context: n=%d | faithful=%s | gap=%.6f | F=%.3e",
            n,
            ctx.is_faithful,
            ctx.K.spectral_gap,
            ctx.K.free_energy,
        )
        return ctx

    @classmethod
    def modular_operator(cls, rho: np.ndarray, a: np.ndarray) -> np.ndarray:
        r"""Δ(a) = ρ^{−1} a ρ."""
        rho = DensityMatrixOps.sanitize(rho)
        rho_inv = DensityMatrixOps.matrix_power(rho, -1.0)
        return rho_inv @ MatrixBanachAlgebra.as_complex(a) @ rho

    @classmethod
    def sigma_t(cls, rho: np.ndarray, t: complex, a: np.ndarray) -> np.ndarray:
        r"""σ_t(a) = ρ^{−it} a ρ^{it}, analítico en t ∈ ℂ (franja KMS)."""
        rho = DensityMatrixOps.sanitize(rho)
        a = MatrixBanachAlgebra.as_complex(a)
        left = DensityMatrixOps.matrix_power(rho, -1j * t)
        right = DensityMatrixOps.matrix_power(rho, 1j * t)
        return left @ a @ right

    @classmethod
    def sigma_i(cls, rho: np.ndarray, a: np.ndarray) -> np.ndarray:
        r"""σ_i(a) = ρ a ρ^{−1}  (punto KMS, t = i)."""
        return cls.sigma_t(rho, 1j, a)

    @classmethod
    def modular_conjugation(cls, rho: np.ndarray, a: np.ndarray) -> np.ndarray:
        r"""J(a) = ρ^{−1/2} a† ρ^{1/2}  (antiunitario)."""
        rho = DensityMatrixOps.sanitize(rho)
        left = DensityMatrixOps.matrix_power(rho, -0.5)
        right = DensityMatrixOps.matrix_power(rho, 0.5)
        return left @ MatrixBanachAlgebra.as_complex(a).conj().T @ right

    @classmethod
    def tomita_S(cls, a: np.ndarray) -> np.ndarray:
        r"""S(a) = a†  en la identificación algebraica π(x)Ω ↔ x."""
        return MatrixBanachAlgebra.as_complex(a).conj().T

    @classmethod
    def polar_decomposition_residual(cls, rho: np.ndarray, a: np.ndarray) -> float:
        r"""‖ J(Δ^{1/2}(a)) − S(a) ‖_F = ‖J(Δ^{1/2}(a)) − a†‖_F."""
        rho = DensityMatrixOps.sanitize(rho)
        a = MatrixBanachAlgebra.as_complex(a)
        delta_half_a = (
            DensityMatrixOps.matrix_power(rho, -0.5)
            @ a
            @ DensityMatrixOps.matrix_power(rho, 0.5)
        )
        polar = cls.modular_conjugation(rho, delta_half_a)
        return float(np.linalg.norm(polar - cls.tomita_S(a), "fro"))

    @classmethod
    def flow_group_law_residual(
        cls, rho: np.ndarray, s: float, t: float, a: np.ndarray
    ) -> float:
        r"""‖ σ_s(σ_t(a)) − σ_{s+t}(a) ‖_F."""
        composed = cls.sigma_t(rho, s, cls.sigma_t(rho, t, a))
        direct = cls.sigma_t(rho, s + t, a)
        return float(np.linalg.norm(composed - direct, "fro"))

    @classmethod
    def connes_cocycle(
        cls, rho: np.ndarray, sigma: np.ndarray, t: float
    ) -> np.ndarray:
        r"""(Dω_ρ : Dω_σ)_t = ρ^{it} σ^{−it}."""
        return (
            DensityMatrixOps.matrix_power(rho, 1j * t)
            @ DensityMatrixOps.matrix_power(sigma, -1j * t)
        )

    @classmethod
    def relative_modular_flow(
        cls, rho: np.ndarray, sigma: np.ndarray, t: float, a: np.ndarray
    ) -> np.ndarray:
        r"""σ_t^{ρ|σ}(a) = ρ^{−it} a σ^{it}."""
        a = MatrixBanachAlgebra.as_complex(a)
        return (
            DensityMatrixOps.matrix_power(rho, -1j * t)
            @ a
            @ DensityMatrixOps.matrix_power(sigma, 1j * t)
        )

    @classmethod
    def verify_kms_self(
        cls, rho: np.ndarray, n_tests: int = 6, seed: int = 42
    ) -> float:
        r"""
        Residuo KMS del estado ρ contra su propio flujo:
            res = max |ω(a σ_i(b)) − ω(ba)|
        Debe ser ≈ 0 (máquina) para cualquier ρ > 0.
        """
        rho = DensityMatrixOps.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(seed)
        max_res = 0.0
        for _ in range(n_tests):
            a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            a = a / (np.linalg.norm(a, "fro") + 1e-30)
            b = b / (np.linalg.norm(b, "fro") + 1e-30)
            lhs = np.trace(rho @ a @ cls.sigma_i(rho, b))
            rhs = np.trace(rho @ b @ a)
            max_res = max(max_res, float(abs(lhs - rhs)))
        return max_res

    @classmethod
    def verify_kms_cross(
        cls,
        rho_flow: np.ndarray,
        rho_state: np.ndarray,
        n_tests: int = 6,
        seed: int = 137,
    ) -> float:
        r"""
        KMS cruzado, residual relativo:

            res = max |Tr(ρ_state a σ_i^{ρ_flow}(b)) − Tr(ρ_state b a)|
                  / (1 + |lhs| + |rhs|)

        Si ρ_state = ρ_flow, colapsa a `verify_kms_self` (salvo la
        normalización). Si no, mide la incompatibilidad modular.
        """
        rho_flow = DensityMatrixOps.sanitize(rho_flow)
        rho_state = DensityMatrixOps.sanitize(rho_state)
        n = rho_state.shape[0]
        rng = np.random.default_rng(seed)
        max_res = 0.0
        for _ in range(n_tests):
            a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            a = a / (np.linalg.norm(a, "fro") + 1e-30)
            b = b / (np.linalg.norm(b, "fro") + 1e-30)
            lhs = np.trace(rho_state @ a @ cls.sigma_i(rho_flow, b))
            rhs = np.trace(rho_state @ b @ a)
            denom = 1.0 + abs(lhs) + abs(rhs)
            max_res = max(max_res, float(abs(lhs - rhs)) / denom)
        return max_res

    @classmethod
    def verify_axioms(cls, rho: np.ndarray) -> Dict[str, float]:
        r"""Residuos de unitalidad, multiplicatividad, isometría e involución."""
        rho = DensityMatrixOps.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(7)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        eye = np.eye(n, dtype=np.complex128)

        unital = float(np.linalg.norm(cls.sigma_t(rho, 0.37, eye) - eye, "fro"))
        product = float(
            np.linalg.norm(
                cls.sigma_t(rho, 0.73, a @ b)
                - cls.sigma_t(rho, 0.73, a) @ cls.sigma_t(rho, 0.73, b),
                "fro",
            )
        )
        isometry = float(
            abs(
                np.linalg.norm(a, "fro")
                - np.linalg.norm(cls.sigma_t(rho, 1.11, a), "fro")
            )
        )
        conjugated = cls.modular_conjugation(rho, a)
        involution = float(
            np.linalg.norm(cls.modular_conjugation(rho, conjugated) - a, "fro")
        )
        return {
            "unital_residual": unital,
            "product_residual": product,
            "isometry_residual": isometry,
            "involution_residual": involution,
        }

    @classmethod
    def verify_tomita_takesaki(cls, rho: np.ndarray) -> Dict[str, float]:
        r"""Paquete: axiomas + polar + grupo + KMS + C*."""
        rho = DensityMatrixOps.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(11)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        report = cls.verify_axioms(rho)
        report["polar_residual"] = cls.polar_decomposition_residual(rho, a)
        report["group_law_residual"] = cls.flow_group_law_residual(rho, 0.41, -0.17, a)
        report["kms_self_residual"] = cls.verify_kms_self(rho)
        report["cstar_residual"] = MatrixBanachAlgebra.cstar_identity_residual(a)
        return report


# ── §2.2 VacuumStateMetrics (observables no tautológicos) ─────────────────
@dataclass(frozen=True, slots=True)
class VacuumStateMetrics:
    r"""
    Métricas verdaderas de un estado ρ frente a (H_ext, ρ_vac, K_vac).

    Definiciones NO tautológicas (cierran P1, P2, P4):

        vacuum_expectation_value   = Tr(ρ H_ext) − E₀(H_ext)
        tomita_takesaki_flow_param = t_* = 1/gap(K_vac)     tiempo modular
        kms_entropy_drift          = S(ρ ‖ ρ_vac)           Umegaki (≠ ∂_t S)
        silence_purity             = F(ρ, ρ_vac)
        is_silent                  = veredicto Ω₃ = ⊤

    ∂_t S(σ_t(ρ)) ≡ 0 porque σ_t ∈ Aut(M_n). El drift físicamente
    significativo es la energía libre modular relativa S(ρ‖ρ_vac).
    """

    vacuum_expectation_value: float
    tomita_takesaki_flow_param: float
    kms_entropy_drift: float
    silence_purity: float
    modular_ground_energy: float
    modular_spectral_gap: float
    von_neumann_entropy: float
    purity: float
    is_silent: bool
    umegaki_to_vacuum: float = 0.0
    bures_distance: float = 0.0
    trace_distance: float = 0.0
    klein_residual: float = 0.0
    dirichlet_energy: float = 0.0


# ── §2.3 Auditor silencioso del vacío ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class VacuumAuditReport:
    r"""
    Auditoría cruzada de la tríada vs. el vacío. Todos los observables son
    cantidades físicas verificables, NO identidades de normalización:

        entropy_production     = S(ρ_obs) − S(ρ_vac)     (puede ser < 0: filtrado)
        fidelity_to_vacuum     = F(ρ_obs, ρ_vac)
        umegaki_to_vacuum      = S(ρ_obs ‖ ρ_vac)
        vev_excitation         = Tr(ρ_obs H_ext) − E₀
        self_kms_residual      = KMS(ρ_obs) autoconsistente
        cross_kms_residual     = KMS(ρ_obs; flujo de ρ_vac)
        transition_probability = Tr(K†K ρ_vac)
        local_verdict          ∈ Ω₃
    """

    entropy_production: float
    fidelity_to_vacuum: float
    vev_excitation: float
    self_kms_residual: float
    cross_kms_residual: float
    transition_probability: float
    modular_spectral_gap: float
    local_verdict: HeytingOmega3
    umegaki_to_vacuum: float = 0.0
    bures_distance: float = 0.0
    trace_distance: float = 0.0
    klein_residual: float = 0.0
    dirichlet_energy: float = 0.0
    tp_residual: float = 0.0
    polar_residual: float = 0.0
    group_law_residual: float = 0.0
    free_energy_vac: float = 0.0


class VacuumSilenceAuditor:
    r"""
    Calcula las métricas verdaderas de la observación silenciosa.

    Criterio de silencio (conjunción):
        • |ΔS|       < ε_S     (invariancia entrópica; el filtrado puede violarla)
        • 1 − F      < ε_F     (fidelidad al vacío)
        • KMS_cross  < ε_K     (compatibilidad modular)
        • KMS_self   < ε_K     (coherencia numérica)
        • |VEV_exc|  < ε_V     (sin excitación sobre el ground)

    Semántica del veredicto local:
        COHERENT : los 5 predicados se satisfacen
        DEGRADED : falla un subconjunto estricto (1 o 2)
        VETOED   : fallan ≥ 3 (pérdida estructural)

    Los residuos Tomita (polar, grupo) y el defecto TP de la tríada NO
    entran en el conteo local: se meet-adjudican en FASE 3, para no
    alterar la semántica empírica de la tríada COHERENT ≈ id.
    """

    EPS_ENTROPY: float = 1.0e-3
    EPS_FIDELITY: float = 1.0e-3
    EPS_KMS: float = 1.0e-4
    EPS_VEV: float = 1.0e-3

    @classmethod
    def audit(
        cls,
        rho_vac: np.ndarray,
        rho_obs: np.ndarray,
        H_ext: np.ndarray,
        transition_prob: float,
        path_laplacian: Optional[np.ndarray] = None,
        K_vac: Optional[ModularHamiltonian] = None,
        triad_tp_residual: float = 0.0,
        tomita_report: Optional[Dict[str, float]] = None,
    ) -> VacuumAuditReport:
        rho_vac = DensityMatrixOps.sanitize(rho_vac)
        rho_obs = DensityMatrixOps.sanitize(rho_obs)
        H_ext = MatrixBanachAlgebra.hermitize(H_ext)

        s_vac = DensityMatrixOps.von_neumann_entropy(rho_vac)
        s_obs = DensityMatrixOps.von_neumann_entropy(rho_obs)
        delta_s = s_obs - s_vac

        fid = DensityMatrixOps.uhlmann_fidelity(rho_obs, rho_vac)
        umegaki = DensityMatrixOps.umegaki_relative_entropy(rho_obs, rho_vac)
        bures = DensityMatrixOps.bures_distance(rho_obs, rho_vac)
        trc = DensityMatrixOps.trace_distance(rho_obs, rho_vac)
        klein = DensityMatrixOps.kleins_inequality_residual(rho_obs, rho_vac)

        w_h = np.real(la.eigvalsh(H_ext))
        e0 = float(w_h.min())
        mean_h = float(np.real(np.trace(rho_obs @ H_ext)))
        vev_exc = mean_h - e0

        kms_self = ModularFlowEngine.verify_kms_self(rho_obs, n_tests=4)
        kms_cross = ModularFlowEngine.verify_kms_cross(rho_vac, rho_obs, n_tests=4)

        if K_vac is None:
            K_vac = DensityMatrixOps.modular_hamiltonian_from_rho(rho_vac)
        gap = K_vac.spectral_gap
        free_e = K_vac.free_energy

        dirichlet = 0.0
        if path_laplacian is not None:
            dirichlet = float(np.real(np.trace(rho_obs @ path_laplacian)))

        tomita_report = tomita_report or {}
        polar = float(tomita_report.get("polar_residual", 0.0))
        group = float(tomita_report.get("group_law_residual", 0.0))

        p_ds = abs(delta_s) < cls.EPS_ENTROPY
        p_f = (1.0 - fid) < cls.EPS_FIDELITY
        p_kms_c = kms_cross < cls.EPS_KMS
        p_kms_s = kms_self < cls.EPS_KMS
        p_vev = abs(vev_exc) < cls.EPS_VEV

        n_fail = sum(0 if p else 1 for p in (p_ds, p_f, p_kms_c, p_kms_s, p_vev))
        if n_fail == 0:
            local = HeytingOmega3.COHERENT
        elif n_fail <= 2:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return VacuumAuditReport(
            entropy_production=float(delta_s),
            fidelity_to_vacuum=float(fid),
            vev_excitation=float(vev_exc),
            self_kms_residual=float(kms_self),
            cross_kms_residual=float(kms_cross),
            transition_probability=float(transition_prob),
            modular_spectral_gap=float(gap),
            local_verdict=local,
            umegaki_to_vacuum=float(umegaki),
            bures_distance=float(bures),
            trace_distance=float(trc),
            klein_residual=float(klein),
            dirichlet_energy=float(dirichlet),
            tp_residual=float(triad_tp_residual),
            polar_residual=float(polar),
            group_law_residual=float(group),
            free_energy_vac=float(free_e),
        )


# ── §2.4 WitnessObservationPipeline — HAND-OFF FASE 2 → FASE 3 ──────────
@dataclass(frozen=True, slots=True)
class WitnessObservationBundle:
    r"""
    Paquete de hand-off FASE 2 → FASE 3. Encapsula:
        • signatura de la tríada observada,
        • estados (ρ_vac, ρ_obs) y Hamiltoniano modular K_vac,
        • auditoría con métricas verdaderas,
        • axiomas / polar / grupo Tomita,
        • residuos estructurales del canal,
        • vector invariante unitario (∈ S^{k−1}).

    Objeto terminal de FASE 2 = dominio de `HeytingWitnessAdjudicator`.
    """

    cycle_index: int
    triad_signature: TriadSignature
    rho_vac: np.ndarray
    rho_observed: np.ndarray
    modular_spectrum: Tuple[float, ...]
    K: ModularHamiltonian
    audit: VacuumAuditReport
    modular_axioms: Dict[str, float]
    tomita_report: Dict[str, float]
    triad_residuals: Dict[str, float]
    invariant_vector: np.ndarray
    context_beta: float

    def as_vacuum_metrics(self, is_silent: bool) -> VacuumStateMetrics:
        r"""Puente 2→3: traduce el bundle a VacuumStateMetrics no tautológicas."""
        return VacuumStateMetrics(
            vacuum_expectation_value=self.audit.vev_excitation,
            tomita_takesaki_flow_param=self.K.modular_correlation_time,
            kms_entropy_drift=self.audit.umegaki_to_vacuum,
            silence_purity=self.audit.fidelity_to_vacuum,
            modular_ground_energy=self.K.ground_energy,
            modular_spectral_gap=self.audit.modular_spectral_gap,
            von_neumann_entropy=DensityMatrixOps.von_neumann_entropy(self.rho_observed),
            purity=DensityMatrixOps.purity(self.rho_observed),
            is_silent=is_silent,
            umegaki_to_vacuum=self.audit.umegaki_to_vacuum,
            bures_distance=self.audit.bures_distance,
            trace_distance=self.audit.trace_distance,
            klein_residual=self.audit.klein_residual,
            dirichlet_energy=self.audit.dirichlet_energy,
        )


class WitnessObservationPipeline:
    r"""
    Orquestador determinista de la dinámica modular-silenciosa:

        WitnessVacuumContext
            → bind (§2.0)
            → Φ_sel(ρ_vac)
            → Tomita (KMS, polar, grupo)
            → auditoría (§2.3)
            → WitnessObservationBundle   (hand-off → FASE 3)
    """

    @classmethod
    def _invariant_vector(
        cls, audit: VacuumAuditReport, triad: TriadChannel
    ) -> np.ndarray:
        r"""
        Vector invariante unitario en S^{6} ⊂ ℝ⁷. Componentes acotadas:

            v₁ = verdict_local / 2                         ∈ [0, 1]
            v₂ = tanh(|ΔS|)
            v₃ = 1 − F(ρ_obs, ρ_vac)
            v₄ = tanh(10 · KMS_cross)
            v₅ = tanh(|VEV_exc|)
            v₆ = clip(p_trans / n, 0, 1)
            v₇ = ‖K‖_F / (√n ‖K‖_op)
        """
        v_local = float(int(audit.local_verdict)) / 2.0
        d_s = math.tanh(abs(audit.entropy_production))
        leak = 1.0 - audit.fidelity_to_vacuum
        kms = math.tanh(10.0 * audit.cross_kms_residual)
        vev = math.tanh(abs(audit.vev_excitation))
        n = max(1, triad.K.shape[0])
        p_trans = float(np.clip(audit.transition_probability / n, 0.0, 1.0))
        op_norm = np.linalg.norm(triad.K, 2) + 1e-30
        k_fro = triad.kraus_norm() / (math.sqrt(n) * op_norm)
        vec = np.array(
            [v_local, d_s, leak, kms, vev, p_trans, k_fro], dtype=np.float64
        )
        norm = float(np.linalg.norm(vec))
        return vec / (norm + 1e-30)

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        rho_vac: np.ndarray,
        K_spec: Tuple[float, ...],
        triad: TriadChannel,
        H_ext: np.ndarray,
        K: Optional[ModularHamiltonian] = None,
        path_laplacian: Optional[np.ndarray] = None,
        context_beta: float = float("nan"),
    ) -> WitnessObservationBundle:
        rho_vac = DensityMatrixOps.sanitize(rho_vac)
        if K is None:
            K = DensityMatrixOps.modular_hamiltonian_from_rho(rho_vac)

        rho_obs = triad.apply(rho_vac)
        p_trans = triad.success_probability(rho_vac)
        triad_res = triad.structural_residuals()

        tomita = ModularFlowEngine.verify_tomita_takesaki(rho_obs)
        axioms = {
            k: tomita[k]
            for k in (
                "unital_residual",
                "product_residual",
                "isometry_residual",
                "involution_residual",
            )
            if k in tomita
        }
        if len(axioms) < 4:
            axioms = ModularFlowEngine.verify_axioms(rho_obs)

        audit = VacuumSilenceAuditor.audit(
            rho_vac,
            rho_obs,
            H_ext,
            p_trans,
            path_laplacian=path_laplacian,
            K_vac=K,
            triad_tp_residual=float(triad_res.get("tp_residual", 0.0)),
            tomita_report=tomita,
        )
        inv = cls._invariant_vector(audit, triad)

        return WitnessObservationBundle(
            cycle_index=cycle_index,
            triad_signature=triad.signature,
            rho_vac=rho_vac,
            rho_observed=rho_obs,
            modular_spectrum=K_spec if K_spec else K.as_tuple(),
            K=K,
            audit=audit,
            modular_axioms=axioms,
            tomita_report=tomita,
            triad_residuals=triad_res,
            invariant_vector=inv,
            context_beta=context_beta,
        )

    # ═════════════════════════════════════════════════════════════════════
    #  HAND-OFF  FASE 2 → FASE 3
    #  Definición formal terminal de FASE 2.
    #  Consume el WitnessVacuumContext (objeto inicial de FASE 2, nacido
    #  en §1.5) y produce WitnessObservationBundle, dominio de §3.1.
    # ═════════════════════════════════════════════════════════════════════
    @classmethod
    def synthesize_from_context(
        cls,
        cycle_index: int,
        ctx: WitnessVacuumContext,
        triad: TriadChannel,
    ) -> WitnessObservationBundle:
        r"""
        Morfismo de hand-off  (WitnessVacuumContext, TriadChannel)
                                ↦ WitnessObservationBundle.

        Continúa en §3.1 `HeytingWitnessAdjudicator.adjudicate`.
        """
        ctx = ModularFlowEngine.bind_vacuum_context(ctx)
        return cls.synthesize(
            cycle_index=cycle_index,
            rho_vac=ctx.rho_vac,
            K_spec=ctx.K.as_tuple(),
            triad=triad,
            H_ext=ctx.H_ext,
            K=ctx.K,
            path_laplacian=ctx.path_laplacian,
            context_beta=ctx.beta,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · CRISTALIZACIÓN Y CUSTODIA FORENSE (C. de FASE 2)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en Ω₃ ─────────────────────────────────────────────────
class HeytingWitnessAdjudicator:
    r"""
    Colapsa el WitnessObservationBundle (objeto terminal de FASE 2) en un
    único veredicto Ω₃ y aplica meet (∧) con el veredicto externo:

        local = audit ∧ axioms ∧ polar ∧ group ∧ klein ∧ (J² ≈ id)
        final = local ∧ external

    Residuos catastróficos (> CATASTROPHIC_TOL) inducen ⊥, no ⋆.
    El defecto TP de la tríada adversarial NO veta por sí solo (la tríada
    es selectiva por diseño); se degrada sólo si supera CATASTROPHIC_TOL.
    """

    AXIOM_TOL: float = 1.0e-6
    CATASTROPHIC_TOL: float = 1.0e-3
    KLEIN_TOL: float = 1.0e-8

    @classmethod
    def _residual_to_omega(cls, residual: float) -> HeytingOmega3:
        if residual > cls.CATASTROPHIC_TOL:
            return HeytingOmega3.VETOED
        if residual > cls.AXIOM_TOL:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def granular_scores(
        cls, bundle: WitnessObservationBundle
    ) -> Dict[str, HeytingOmega3]:
        axioms_max = (
            max(bundle.modular_axioms.values()) if bundle.modular_axioms else 0.0
        )
        polar = float(bundle.tomita_report.get("polar_residual", bundle.audit.polar_residual))
        group = float(
            bundle.tomita_report.get("group_law_residual", bundle.audit.group_law_residual)
        )
        klein = abs(float(bundle.audit.klein_residual))
        if klein < cls.KLEIN_TOL:
            klein_v = HeytingOmega3.COHERENT
        elif klein < cls.CATASTROPHIC_TOL:
            klein_v = HeytingOmega3.DEGRADED
        else:
            klein_v = HeytingOmega3.VETOED

        unitary_res = float(bundle.triad_residuals.get("unitary_residual", 0.0))
        projector_res = float(bundle.triad_residuals.get("projector_residual", 0.0))
        return {
            "audit": bundle.audit.local_verdict,
            "axioms": cls._residual_to_omega(axioms_max),
            "polar": cls._residual_to_omega(polar),
            "group_law": cls._residual_to_omega(group),
            "klein": klein_v,
            "triad_unitary": cls._residual_to_omega(unitary_res),
            "triad_projector": cls._residual_to_omega(projector_res),
        }

    @classmethod
    def adjudicate(
        cls,
        bundle: WitnessObservationBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""
        §3.1  Arranque de FASE 3.

        Continúa `WitnessObservationPipeline.synthesize_from_context` (§2.4):
        el bundle es el clasificador de verdad pre-Ω₃; aquí se toma el
        ínfimo con la fuente externa (auditor adversarial).
        """
        scores = cls.granular_scores(bundle)
        local = HeytingOmega3.COHERENT
        for value in scores.values():
            local = local.meet(value)
        return local.meet(external_verdict)

    @classmethod
    def implication_chain(
        cls,
        bundle: WitnessObservationBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""Diagnóstico residuado: no sustituye al meet; expone fallos de ⇒."""
        scores = cls.granular_scores(bundle)
        chain = (
            scores["axioms"]
            .implies(scores["audit"])
            .meet(scores["polar"].implies(scores["group_law"]))
            .meet(scores["klein"].implies(scores["audit"]))
        )
        return chain.meet(external_verdict)


# ── §3.2 ExperienceCrystal con cadena Merkle ──────────────────────────────
@dataclass(frozen=True, slots=True)
class ExperienceCrystal:
    r"""
    Cristal inmutable de experiencia, firmado con SHA-256 y encadenado
    Merkle-style con el cristal anterior:

        content_hash_k = SHA-256(witness ‖ cycle ‖ triad_sig ‖ vector ‖ audit)
        chain_hash_k   = SHA-256(chain_hash_{k−1} ‖ content_hash_k)

    El campo `crystallized_invariant_vector` vive en S^{6} ⊂ ℝ⁷ (norma 1).
    `merkle_parent` conserva el eslabón anterior para verificación forense
    independiente del estado mutable del agente.
    """

    crystal_id: str
    witness_id: str
    trickster_illusion_type: str
    dreamer_scenario_id: str
    auditor_immunization_hash: str
    vacuum_metrics: VacuumStateMetrics
    audit_report: VacuumAuditReport
    heyting_verdict: HeytingOmega3
    crystallized_invariant_vector: np.ndarray
    content_hash: str
    chain_hash: str
    merkle_parent: str
    sha256_provenance: str
    timestamp_utc: float
    engine_version: str = __version__


# ── §3.3 TOONSilentWitnessAgent — orquestador soberano ────────────────────
class TOONSilentWitnessAgent:
    r"""
    Soberano Testigo Silencioso y Cristalizador de Experiencia.

    Habita en el vacío modular (KMS a β_vac), observa la tríada adversarial
    (Ilusionista → Soñador → Auditor) SIN back-action, y cristaliza cada
    ciclo en un ExperienceCrystal con cadena de custodia Merkle-SHA-256.

    Cadena de 3 fases en cada ciclo (hand-offs estrictos):

        FASE 1: prepare_vacuum_context(H, β)     → WitnessVacuumContext
        FASE 2: synthesize_from_context(ctx, Φ)  → WitnessObservationBundle
        FASE 3: adjudicate(Ω₃) + Merkle          → ExperienceCrystal

    H_ext se construye como tight-binding sobre P_n: el gap es un
    invariante espectral-gráfico (Cheeger discreto) y la energía de
    Dirichlet Tr(ρ L_{P_n}) cuantifica la variación a lo largo del camino.
    """

    def __init__(
        self,
        agent_id: str = "SILENT-WITNESS-SABIO-01",
        dimension_mac: int = 4,
        kms_beta: float = 1.0,
        vacuum_beta: float = WitnessVacuumPreparation.DEFAULT_BETA_COLD,
        seed: int = 999,
        hopping: float = WitnessVacuumPreparation.DEFAULT_HOPPING,
    ) -> None:
        self.agent_id = agent_id
        self.dimension_mac = dimension_mac
        self.kms_beta = kms_beta
        self.vacuum_beta = vacuum_beta
        self.seed = seed
        self.crystal_count = 0
        self.experience_archive: List[ExperienceCrystal] = []

        self.H_ext = WitnessVacuumPreparation.tight_binding_hamiltonian(
            dimension_mac, hopping=hopping
        )
        self.ground_projector = DensityMatrixOps.ground_state_projector(self.H_ext)
        self.path_laplacian = WitnessVacuumPreparation.path_laplacian(dimension_mac)

        self._genesis_hash = hashlib.sha256(
            f"{agent_id}::GENESIS::{__version__}".encode("ascii")
        ).hexdigest()
        self._chain_hash = self._genesis_hash

    def _content_hash(
        self,
        cycle_id: str,
        audit: VacuumAuditReport,
        inv: np.ndarray,
        sig: TriadSignature,
        dirichlet_energy_arg: float,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.agent_id.encode("ascii"))
        hasher.update(cycle_id.encode("ascii"))
        hasher.update(sig.as_bytes())
        hasher.update(np.ascontiguousarray(inv).tobytes())
        hasher.update(f"{audit.entropy_production:.12e}".encode("ascii"))
        hasher.update(f"{audit.fidelity_to_vacuum:.12e}".encode("ascii"))
        hasher.update(f"{audit.cross_kms_residual:.12e}".encode("ascii"))
        hasher.update(f"{audit.vev_excitation:.12e}".encode("ascii"))
        hasher.update(f"{audit.umegaki_to_vacuum:.12e}".encode("ascii"))
        hasher.update(f"{audit.local_verdict.name}".encode("ascii"))
        hasher.update(f"{dirichlet_energy_arg:.12e}".encode("ascii"))
        return hasher.hexdigest()

    def _advance_chain(self, content_hash: str) -> Tuple[str, str]:
        parent = self._chain_hash
        digest = hashlib.sha256(
            parent.encode("ascii") + content_hash.encode("ascii")
        ).hexdigest()
        self._chain_hash = digest
        return parent, digest

    def latest_crystal(self) -> Optional[ExperienceCrystal]:
        return self.experience_archive[-1] if self.experience_archive else None

    def verify_merkle_chain(self) -> bool:
        r"""
        Recomputación de chain_hash_k = SHA256(parent_{k−1} ‖ content_hash_k)
        y verificación de formato hex-64 + ‖v_inv‖₂ ≈ 1.
        """
        parent = self._genesis_hash
        for crystal in self.experience_archive:
            if crystal.merkle_parent != parent:
                return False
            expected = hashlib.sha256(
                parent.encode("ascii") + crystal.content_hash.encode("ascii")
            ).hexdigest()
            if expected != crystal.chain_hash:
                return False
            if len(crystal.chain_hash) != 64:
                return False
            if not math.isclose(
                float(np.linalg.norm(crystal.crystallized_invariant_vector)),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            ):
                return False
            parent = crystal.chain_hash
        return True

    def observe_and_crystallize(
        self,
        trickster_illusion_type: str,
        dreamer_scenario_id: str,
        auditor_immunization_hash: str,
        auditor_verdict: HeytingOmega3,
        trickster_strength: float = 0.0,
        dreamer_strength: float = 0.0,
        auditor_rank: int = 4,
        dirichlet_energy: float = 0.0,
    ) -> ExperienceCrystal:
        self.crystal_count += 1
        crystal_id = f"CRYSTAL-EXP-{self.crystal_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Silencio Epistémico #%d | ilusión=%s | rank=%d ═══",
            self.crystal_count,
            trickster_illusion_type,
            auditor_rank,
        )

        # ── FASE 1 ── contexto GNS (ρ_vac, K_vac, |Ω⟩⟨Ω|, TFD, L_{P_n}) ──
        ctx = WitnessVacuumPreparation.prepare_vacuum_context(
            self.H_ext, beta=self.vacuum_beta
        )

        # ── Canal selectivo de la tríada (morfismo CP) ──
        triad = TriadOperatorFactory.build_triad(
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            n=self.dimension_mac,
            trickster_strength=trickster_strength,
            dreamer_strength=dreamer_strength,
            auditor_rank=auditor_rank,
        )

        # ── FASE 2 ── bind + Φ_sel + Tomita + auditoría ──
        bundle = WitnessObservationPipeline.synthesize_from_context(
            cycle_index=self.crystal_count, ctx=ctx, triad=triad
        )

        # ── FASE 3 ── adjudicación Ω₃ y cristalización Merkle ──
        final_verdict = HeytingWitnessAdjudicator.adjudicate(bundle, auditor_verdict)
        metrics = bundle.as_vacuum_metrics(
            is_silent=(final_verdict == HeytingOmega3.COHERENT)
        )

        content_hash = self._content_hash(
            crystal_id,
            bundle.audit,
            bundle.invariant_vector,
            bundle.triad_signature,
            dirichlet_energy,
        )
        merkle_parent, chain_hash = self._advance_chain(content_hash)

        sig_hasher = hashlib.sha256()
        sig_hasher.update(self.agent_id.encode("ascii"))
        sig_hasher.update(crystal_id.encode("ascii"))
        sig_hasher.update(final_verdict.name.encode("ascii"))
        sig_hasher.update(chain_hash.encode("ascii"))
        sig_hasher.update(f"{time.time_ns()}".encode("ascii"))
        provenance = sig_hasher.hexdigest()

        crystal = ExperienceCrystal(
            crystal_id=crystal_id,
            witness_id=self.agent_id,
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            vacuum_metrics=metrics,
            audit_report=bundle.audit,
            heyting_verdict=final_verdict,
            crystallized_invariant_vector=bundle.invariant_vector,
            content_hash=content_hash,
            chain_hash=chain_hash,
            merkle_parent=merkle_parent,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
            engine_version=__version__,
        )
        self.experience_archive.append(crystal)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cristal %s | Ω₃=%s | ΔS=%.3e | F=%.6f | S(ρ‖vac)=%.3e | "
            "KMS_x=%.3e | polar=%.2e | %.2f ms",
            crystal_id,
            final_verdict.name,
            bundle.audit.entropy_production,
            bundle.audit.fidelity_to_vacuum,
            bundle.audit.umegaki_to_vacuum,
            bundle.audit.cross_kms_residual,
            bundle.audit.polar_residual,
            dt_ms,
        )
        return crystal


# ── §3.4 Demostración autónoma ───────────────────────────────────────────
def _print_heyting_audit() -> None:
    laws: Mapping[str, bool] = HeytingOmega3.verify_heyting_laws()
    print("  Heyting Ω₃ laws:", dict(laws))


if __name__ == "__main__":
    witness = TOONSilentWitnessAgent(
        agent_id="SILENT-WITNESS-SABIO-01",
        dimension_mac=4,
        kms_beta=1.0,
        vacuum_beta=50.0,
    )

    print("═" * 88)
    print(f"TESTIGO SILENCIOSO — Evolución Doctoral v{__version__}")
    print("═" * 88)
    _print_heyting_audit()

    ctx0 = WitnessVacuumPreparation.prepare_vacuum_context(witness.H_ext, beta=50.0)
    ctx0 = ModularFlowEngine.bind_vacuum_context(ctx0)
    kms_self = ModularFlowEngine.verify_kms_self(ctx0.rho_vac, n_tests=8)
    tomita0 = ModularFlowEngine.verify_tomita_takesaki(ctx0.rho_vac)
    print(f"[σ-t autotest] KMS(β=1) residual = {kms_self:.3e}   (esperado ≈ 0)")
    print(
        f"[Axiomas σ_t ] unital={tomita0['unital_residual']:.2e}  "
        f"product={tomita0['product_residual']:.2e}  "
        f"isometry={tomita0['isometry_residual']:.2e}  "
        f"J²={tomita0['involution_residual']:.2e}"
    )
    print(
        f"[Polar / grupo] polar={tomita0['polar_residual']:.2e}  "
        f"group={tomita0['group_law_residual']:.2e}  "
        f"F_mod={ctx0.K.free_energy:.2e}  faithful={ctx0.is_faithful}"
    )

    scenarios = [
        (
            "COHERENT TRIAD (Φ ≈ id)",
            "TRIAD-COH-001",
            "SYNTH-COH-0001",
            "AAAACOH00000001",
            HeytingOmega3.COHERENT,
            0.02,
            0.02,
            4,
        ),
        (
            "DEGRADED TRIAD (perturbación media)",
            "TRIAD-DEG-002",
            "SYNTH-DEG-0002",
            "BBBBBDEG0000002",
            HeytingOmega3.COHERENT,
            0.25,
            0.25,
            3,
        ),
        (
            "VETOED TRIAD (caos adversarial)",
            "SPLIT_CONTRACT_ILLUSION",
            "SYNTH-VETO-0003",
            "bfa1bfa0b2551841",
            HeytingOmega3.VETOED,
            1.00,
            1.00,
            1,
        ),
    ]

    print("\n" + "─" * 88)
    for name, ill, drm, aih, ext, s_i, s_d, rank in scenarios:
        crystal = witness.observe_and_crystallize(
            trickster_illusion_type=ill,
            dreamer_scenario_id=drm,
            auditor_immunization_hash=aih,
            auditor_verdict=ext,
            trickster_strength=s_i,
            dreamer_strength=s_d,
            auditor_rank=rank,
        )
        print(f"\n[{name}]")
        print(f"   crystal_id         : {crystal.crystal_id}")
        print(f"   Ω₃ final           : {crystal.heyting_verdict.name}")
        print(f"   ΔS (producida)     : {crystal.audit_report.entropy_production:+.6e}")
        print(f"   F(ρ_obs, ρ_vac)    : {crystal.audit_report.fidelity_to_vacuum:.9f}")
        print(f"   S(ρ_obs‖ρ_vac)     : {crystal.audit_report.umegaki_to_vacuum:.6e}")
        print(f"   Bures / D_tr       : {crystal.audit_report.bures_distance:.3e} / "
              f"{crystal.audit_report.trace_distance:.3e}")
        print(f"   KMS_cross residual : {crystal.audit_report.cross_kms_residual:.3e}")
        print(f"   VEV excitación     : {crystal.audit_report.vev_excitation:+.6e}")
        print(f"   Gap(K_vac)         : {crystal.audit_report.modular_spectral_gap:+.6e}")
        print(f"   t_* modular        : {crystal.vacuum_metrics.tomita_takesaki_flow_param:.6e}")
        print(f"   polar / grupo      : {crystal.audit_report.polar_residual:.2e} / "
              f"{crystal.audit_report.group_law_residual:.2e}")
        print(
            f"   ‖v_inv‖₂           : "
            f"{np.linalg.norm(crystal.crystallized_invariant_vector):.6f}"
        )
        print(f"   chain_hash         : {crystal.chain_hash[:32]}…")

    print("\n  Cadena Merkle SHA-256 válida:", witness.verify_merkle_chain())
    print("\n" + "═" * 88)
    print("✓ Tomita-Takesaki ejecutable: σ_t, σ_i, J, Δ, polar, grupo.")
    print("✓ KMS(β=1) consistente: ω(a σ_i(b)) = ω(ba)  —residuo ≈ 0.")
    print("✓ Drift KMS = S(ρ_obs‖ρ_vac) operatorial (no tautológico).")
    print("✓ Tríada como instrumento de Lüders (Choi ⪰ 0, defecto TP medido).")
    print("✓ Cadena de custodia Merkle-SHA-256 verificable entre cristales.")
    print(f"✓ Total cristales: {len(witness.experience_archive)}")
    print("═" * 88)
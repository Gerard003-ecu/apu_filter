# -*- coding: utf-8 -*-
r"""Soberano Simulador Onírico REM y Metabolizador de Perturbaciones Contrafactuales.

Ubicación: app/agents/wisdom/toon_oniric_dreamer_agent.py
Versión  : 4.0.0-Doctoral-Nested-Ω₄-LT-Clifford-Enclave-GKSL-Merkle

Este módulo implementa el "Soberano Simulador Onírico", agente encargado de ejecutar
simulaciones contrafactuales en fase REM para explorar colapsos presupuestarios,
fugas ciber-físicas y escenarios de riesgo en la arquitectura COGNITIVE TOON / APU Filter.

================================================================================
I. FORMALIZACIÓN MATEMÁTICA Y TEORÍA DE GRUPOS / OPERADORES
================================================================================

1. Retículo de Heyting Ω₄ y Topología de Lawvere-Tierney $j$:
   El espacio de clasificadores de subobjetos es la cadena finita:
       $$\Omega_4 = \{0 (\bot, \mathrm{VETOED}) < 1 (\partial, \mathrm{BOUNDARY\_CRITICAL}) < 2 (\sharp, \mathrm{TOPOLOGICAL\_STABLE}) < 3 (\top, \mathrm{VERUM\_COHERENT})\}$$
   La topología de Lawvere-Tierney $j : \Omega_4 \to \Omega_4$ satisface los axiomas de Grothendieck:
     • (j1) $a \le j(a)$ (extensividad)
     • (j2) $j(j(a)) = j(a)$ (idempotencia)
     • (j3) $j(a \land b) = j(a) \land j(b)$ (preservación de encuentros)
     • $j(\top) = \top$

2. Álgebras de Bicuaterniones $\mathbb{C} \otimes \mathbb{H} \cong \mathrm{Cl}^+_{1,3}(\mathbb{R}) \cong M_2(\mathbb{C})$:
   Para $q \in \mathbb{C} \otimes \mathbb{H}$, la norma reducida compleja $N(q) = \det(\phi(q)) \in \mathbb{C}$ es multiplicativa:
       $$N(q_1 q_2) = N(q_1) N(q_2)$$
   La parte hermítica de $\phi(q)$ modula el Hamiltoniano $H$, mientras que la anti-hermítica alimenta la disipación condicional.

3. Complejo de Cadenas Simplicial $K = (C_0, C_1, C_2)$ y Teoría de Hodge:
   Con operadores de borde $\partial_2 : C_2 \to C_1$ y $\partial_1 : C_1 \to C_0$ tales que $\partial_1 \partial_2 = 0$,
   los Laplacianos de Combinatoria son $L_0 = \partial_1 \partial_1^T$, $L_1 = \partial_1^T \partial_1 + \partial_2 \partial_2^T$,
   y los números de Betti $\beta_k = \dim \ker L_k$ satisfacen la fórmula de Euler-Poincaré:
       $$\chi(K) = |V| - |E| + |F| = \beta_0 - \beta_1 + \beta_2$$

4. Ecuación Maestra GKSL y Enclave de No-Señilización:
   La evolución temporal en el subespacio de superselección del enclave $P_d \mathcal{H} P_d$ sigue la dinámica CPTP:
       $$\frac{d\rho}{dt} = -i [H_{\mathrm{eff}}, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho\} \right)$$
   con tasa de escape condicional $\Gamma(\rho) = \sum_k \gamma_k \mathrm{Tr}(\rho L_k^\dagger L_k) \ge 0$.

5. Inmunización por Cobertura Espectral e Inclusión Merkle SHA-512:
   La vacuna espectral $P_{\mathrm{vac}} = \sum_{i=1}^k |v_i\rangle\langle v_i|$ se construye ordenando los autovalores $\lambda_1 \ge \lambda_2 \ge \dots$
   hasta alcanzar la masa acumulada $\sum_{i=1}^k \lambda_i \ge f_{\mathrm{cov}}$. Las trazas de auditoría forman
   un árbol de Merkle inmutable verificado vía SHA-512.

================================================================================
II. ESTRUCTURA FUNTORIAL Y ARQUITECTURA
================================================================================

El Soberano realiza el funtor estricto $\mathcal{D} = V \circ I \circ \Phi_t \circ \Pi_{\mathrm{enc}} \circ H \circ K$:
    $$\mathcal{D} : \mathbf{Scenario} \longrightarrow \mathbf{OniricScenarioCertificate}$$

  • $F_1$ (`CategoricalCircuitCartridge.synthesize_cartridge`): $\mathbf{Scenario} \to \mathrm{CategoricalCircuitCartridge}$.
    Construcción del 2-complejo $K$, números de Betti $\beta_\bullet$, red de Tellegen y $H = \mathrm{lift\_enclave\_hamiltonian}$.
  • $F_2$ (`MetabolizedPerturbationField.evolve_and_certify`): $\mathrm{CategoricalCircuitCartridge} \to \mathrm{MetabolizedPerturbationField}$.
    Evolución GKSL en $P_d \mathcal{H} P_d$, entropía de Umegaki $S(\rho \| \rho_0)$, distancia de Bures $D_B$ y tasa de escape $\Gamma$.
  • $F_3$ (`TOONOniricDreamerAgent.dream_scenario`): $\mathrm{MetabolizedPerturbationField} \to \mathrm{OniricScenarioCertificate}$.
    Proyección de vacuna $P_{\mathrm{vac}}$, constante de aprendizaje adaptativa $\eta(\Omega_4)$, firma de Merkle y pasaporte REM.

================================================================================
III. INVARIANTES FORMALES Y AXIOMAS DEL SISTEMA
================================================================================

- Axioma 1 (Completitud del Enclave): $P_d + P_p = I$, $P_d P_p = 0$, $P_d^2 = P_d = P_d^\dagger$.
- Axioma 2 (Exactitud Simplicial): $\partial_1 \partial_2 = 0$ y $\chi = \beta_0 - \beta_1 + \beta_2$.
- Axioma 3 (Positividad de Kossakowski): $\gamma_k \ge 0$, garantizando que la evolución Lindblad es un canal CPTP.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONOniricDreamer.v4")

__all__ = [
    "HeytingToposAlgebra",
    "BiquaternionClifford",
    "SimplicialHodgeGraph",
    "NonReciprocalTellegenNetwork",
    "OpenQuantumDynamicsSeed",
    "CategoricalCircuitCartridge",
    "BanachSpectralAlgebra",
    "NonCommutativeNoSignalingEnclave",
    "NonHermitianLindbladMasterEngine",
    "MetabolizedPerturbationField",
    "MerkleInclusionProof",
    "OniricScenarioCertificate",
    "MacroWakeSleepAuditReport",
    "SpectralImmuneVaccineEngine",
    "TOONOniricDreamerAgent",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — TOPOS, LAWVERE-TIERNEY, CLIFFORD, HODGE EXACTO, TELLEGEN Y SEMILLA H
# ══════════════════════════════════════════════════════════════════════════════
# Andamiaje del topos 𝓣_Ω₄ y del 2-complejo de Hodge. El ÚLTIMO método de
# esta fase (CategoricalCircuitCartridge.lift_enclave_hamiltonian) es el
# germen formal de FASE 2: NonHermitianLindbladMasterEngine lo invoca
# sin duplicar la construcción de H.
# ══════════════════════════════════════════════════════════════════════════════


class HeytingToposAlgebra(IntEnum):
    r"""
    Cadena finita Ω₄ = {0 ≺ 1 ≺ 2 ≺ 3} = {⊥ ≺ ∂ ≺ ♯ ≺ ⊤}, álgebra de Heyting
    completa (toda cadena finita con máximo y mínimo lo es) y clasificador
    de subobjetos de un topos de haces sobre un sitio finito de Grothendieck:

        a ∧ b  = min(a, b)
        a ∨ b  = max(a, b)
        a → b  = ⋁{ c ∈ Ω₄ | c ∧ a ≤ b }     (residuo)
        ¬a     = a → ⊥

    El esqueleto booleano es {⊥, ⊤}; 1 y 2 violan el tercio excluso, de modo
    que Ω₄ es estrictamente intuicionista.

    El operador j = lawvere_tierney_closure densifica BOUNDARY_CRITICAL →
    TOPOLOGICAL_STABLE y satisface los tres axiomas de una topología de
    Lawvere-Tierney (SGA4 / Mac Lane–Moerdijk), certificables exhaustivamente.
    """

    VETOED_ABSURDUM = 0      # ⊥
    BOUNDARY_CRITICAL = 1    # ∂
    TOPOLOGICAL_STABLE = 2   # ♯
    VERUM_COHERENT = 3       # ⊤

    @property
    def verdict(self) -> str:
        return self.name

    def meet(self, other: "HeytingToposAlgebra") -> "HeytingToposAlgebra":
        return HeytingToposAlgebra(min(int(self), int(other)))

    def join(self, other: "HeytingToposAlgebra") -> "HeytingToposAlgebra":
        return HeytingToposAlgebra(max(int(self), int(other)))

    def implies(self, other: "HeytingToposAlgebra") -> "HeytingToposAlgebra":
        r"""Residuo a → b = ⋁{ c ∈ Ω₄ | c ∧ a ≤ b }."""
        a, b = int(self), int(other)
        candidates = [c for c in range(4) if min(c, a) <= b]
        return HeytingToposAlgebra(max(candidates))

    def pseudo_complement(self) -> "HeytingToposAlgebra":
        return self.implies(HeytingToposAlgebra.VETOED_ABSURDUM)

    def classical_negation(self) -> "HeytingToposAlgebra":
        return HeytingToposAlgebra(3 - int(self))

    def double_negation(self) -> "HeytingToposAlgebra":
        r"""¬¬a ≥ a (ley débil; igualdad sólo en regulares)."""
        return self.pseudo_complement().pseudo_complement()

    def is_regular(self) -> bool:
        r"""a regular ⟺ ¬¬a = a. En Ω₄: {⊥, ⊤}."""
        return self.double_negation() == self

    def is_dense(self) -> bool:
        return self.pseudo_complement() == HeytingToposAlgebra.VETOED_ABSURDUM

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬a = ⊤  ⇔  a ∈ {⊥, ⊤}."""
        return self.join(self.pseudo_complement()) == HeytingToposAlgebra.VERUM_COHERENT

    def lawvere_tierney_closure(self) -> "HeytingToposAlgebra":
        r"""
        Operador modal j: Ω → Ω candidato a topología de Grothendieck.
        Densifica estados marginales (BOUNDARY_CRITICAL → TOPOLOGICAL_STABLE),
        modelando la clausura de plausibilidad de escenarios contrafactuales.
        """
        val = int(self)
        if val == 0:
            return HeytingToposAlgebra.VETOED_ABSURDUM
        if val == 1:
            return HeytingToposAlgebra.TOPOLOGICAL_STABLE
        return self

    def is_j_closed(self) -> bool:
        r"""a es j-cerrado ⟺ j(a) = a. En esta j: {⊥, ♯, ⊤}."""
        return self.lawvere_tierney_closure() == self

    @classmethod
    def bottom(cls) -> "HeytingToposAlgebra":
        return cls.VETOED_ABSURDUM

    @classmethod
    def top(cls) -> "HeytingToposAlgebra":
        return cls.VERUM_COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingToposAlgebra":
        return cls.VERUM_COHERENT if b else cls.VETOED_ABSURDUM

    def to_bool(self) -> bool:
        if self not in (
            HeytingToposAlgebra.VETOED_ABSURDUM,
            HeytingToposAlgebra.VERUM_COHERENT,
        ):
            raise ValueError(f"{self.name} no admite proyección fiel a 𝔹₂.")
        return self == HeytingToposAlgebra.VERUM_COHERENT

    @classmethod
    def verify_residuation_axiom(cls) -> bool:
        r"""∀ a,b,c ∈ Ω₄:  (c ∧ a ≤ b)  ⟺  (c ≤ (a → b))."""
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

    @classmethod
    def verify_lawvere_tierney_topology_axioms(cls) -> bool:
        r"""
        Tres axiomas de una topología de Lawvere-Tierney j: Ω → Ω:
          (j1) a ≤ j(a)                         extensividad
          (j2) j(j(a)) = j(a)                   idempotencia
          (j3) j(a ∧ b) = j(a) ∧ j(b)           preservación de encuentros
        Además se exige j(⊤) = ⊤ (preservación del terminal).
        """
        elements = list(cls)
        if cls.VERUM_COHERENT.lawvere_tierney_closure() != cls.VERUM_COHERENT:
            return False
        for a in elements:
            ja = a.lawvere_tierney_closure()
            if int(a) > int(ja):
                return False
            if ja.lawvere_tierney_closure() != ja:
                return False
            for b in elements:
                lhs = a.meet(b).lawvere_tierney_closure()
                rhs = ja.meet(b.lawvere_tierney_closure())
                if lhs != rhs:
                    return False
        return True


@dataclass(frozen=True, slots=True)
class BiquaternionClifford:
    r"""
    Elemento del álgebra de bicuaterniones

        𝔹 ≅ ℍ ⊗_ℝ ℂ  ≅  Cl⁺_{1,3}(ℝ)  ≅  M₂(ℂ)

    inmerso fielmente en M₂(ℂ) vía Pauli-Dirac. Distingue dos nociones de
    «tamaño» no intercambiables:

      • reduced_norm_complex  N(q) = det φ(q) ∈ ℂ
        invariante algebraico de la forma cuadrática (signatura mixta:
        puede ser complejo o negativo; NO es una métrica).
        Multiplicatividad: N(q₁ q₂) = N(q₁) N(q₂).

      • frobenius_operator_norm  ‖φ(q)‖_HS ≥ 0
        norma de Hilbert-Schmidt genuina, apta para estabilidad.

    La parte hermítica de φ(q) es el único término admisible en un H de
    GKSL; la anti-hermítica alimenta el generador condicional H_cond.
    """

    w_re: float
    w_im: float
    x_re: float
    x_im: float
    y_re: float
    y_im: float
    z_re: float
    z_im: float

    _NORM_FLOOR: Final[float] = 1e-15

    def _as_complex_tuple(self) -> Tuple[complex, complex, complex, complex]:
        return (
            complex(self.w_re, self.w_im),
            complex(self.x_re, self.x_im),
            complex(self.y_re, self.y_im),
            complex(self.z_re, self.z_im),
        )

    @classmethod
    def from_complex(
        cls, w: complex, x: complex, y: complex, z: complex
    ) -> "BiquaternionClifford":
        return cls(
            w.real, w.imag, x.real, x.imag, y.real, y.imag, z.real, z.imag
        )

    def to_matrix(self) -> np.ndarray:
        r"""Homomorfismo inyectivo de anillos 𝔹 ↪ M₂(ℂ)."""
        w, x, y, z = self._as_complex_tuple()
        return np.array(
            [
                [w + 1j * z, y + 1j * x],
                [-y + 1j * x, w - 1j * z],
            ],
            dtype=np.complex128,
        )

    def __add__(self, other: "BiquaternionClifford") -> "BiquaternionClifford":
        return BiquaternionClifford(
            self.w_re + other.w_re,
            self.w_im + other.w_im,
            self.x_re + other.x_re,
            self.x_im + other.x_im,
            self.y_re + other.y_re,
            self.y_im + other.y_im,
            self.z_re + other.z_re,
            self.z_im + other.z_im,
        )

    def __mul__(self, other: object) -> "BiquaternionClifford":
        if isinstance(other, BiquaternionClifford):
            w1, x1, y1, z1 = self._as_complex_tuple()
            w2, x2, y2, z2 = other._as_complex_tuple()
            return BiquaternionClifford.from_complex(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        if isinstance(other, (int, float, complex)):
            s = complex(other)
            w, x, y, z = self._as_complex_tuple()
            return BiquaternionClifford.from_complex(s * w, s * x, s * y, s * z)
        return NotImplemented

    def conjugate(self) -> "BiquaternionClifford":
        r"""Conjugación cuaterniónica (no compleja): w − x i − y j − z k."""
        return BiquaternionClifford(
            self.w_re,
            self.w_im,
            -self.x_re,
            -self.x_im,
            -self.y_re,
            -self.y_im,
            -self.z_re,
            -self.z_im,
        )

    def hermitian_part(self) -> np.ndarray:
        M = self.to_matrix()
        return 0.5 * (M + M.conj().T)

    def anti_hermitian_part(self) -> np.ndarray:
        M = self.to_matrix()
        return 0.5 * (M - M.conj().T)

    def reduced_norm_complex(self) -> complex:
        r"""N(q) = det φ(q) ∈ ℂ. Invariante algebraico, no métrica."""
        return complex(np.linalg.det(self.to_matrix()))

    def frobenius_operator_norm(self) -> float:
        M = self.to_matrix()
        return float(np.sqrt(np.real(np.trace(M.conj().T @ M))))

    def reduced_norm_multiplicativity_residual(
        self, other: "BiquaternionClifford"
    ) -> float:
        r"""|N(q₁ q₂) − N(q₁) N(q₂)|  (nulo en aritmética exacta)."""
        n_prod = (self * other).reduced_norm_complex()
        n_sep = self.reduced_norm_complex() * other.reduced_norm_complex()
        return abs(n_prod - n_sep)

    def cstar_residual(self) -> float:
        r"""| ‖M† M‖₂ − ‖M‖₂² | sobre φ(q) (identidad C* residual)."""
        M = self.to_matrix()
        op = float(la.norm(M.conj().T @ M, 2))
        nrm = float(la.norm(M, 2))
        return abs(op - nrm * nrm)


class SimplicialHodgeGraph:
    r"""
    Complejo de cadenas simplicial finito K = (C₀, C₁, C₂) que modela el
    grafo de dependencias críticas de costos e insumos.

        ∂₂ : C₂ → C₁,   ∂₁ : C₁ → C₀,   ∂₁ ∂₂ = 0

        L₀ = ∂₁ ∂₁ᵀ,   L₁ = ∂₁ᵀ ∂₁ + ∂₂ ∂₂ᵀ,   L₂ = ∂₂ᵀ ∂₂

    Números de Betti (Eckmann–Hodge):
        βₖ = dim ker Lₖ,    χ(K) = |V|−|E|+|F| = β₀−β₁+β₂.

    Descomposición de Hodge en 1-cadenas:
        C₁ = im ∂₂  ⊕  im ∂₁ᵀ  ⊕  ker L₁.
    """

    def __init__(
        self,
        num_vertices: int,
        edges: List[Tuple[int, int]],
        faces: List[Tuple[int, int, int]],
    ) -> None:
        if num_vertices < 1:
            raise ValueError("num_vertices debe ser ≥ 1.")
        self.num_vertices = int(num_vertices)
        self.edges = [tuple(sorted((u, v))) for u, v in edges]
        self.faces = [tuple(sorted((u, v, w))) for u, v, w in faces]

        self._validate_topology()
        self.boundary_1 = self._build_boundary_1()
        self.boundary_2 = self._build_boundary_2()
        self._validate_chain_complex_exactness()

        self.laplacian_0 = self.boundary_1 @ self.boundary_1.T
        self.laplacian_1 = (self.boundary_1.T @ self.boundary_1) + (
            self.boundary_2 @ self.boundary_2.T
        )
        if self.boundary_2.size > 0:
            self.laplacian_2 = self.boundary_2.T @ self.boundary_2
        else:
            self.laplacian_2 = np.zeros(
                (len(self.faces), len(self.faces)), dtype=np.float64
            )

    def _validate_topology(self) -> None:
        for (u, v) in self.edges:
            if not (0 <= u < self.num_vertices and 0 <= v < self.num_vertices):
                raise ValueError(
                    f"Arista {(u, v)} fuera de rango [0, {self.num_vertices})."
                )
            if u == v:
                raise ValueError(f"Lazo degenerado {(u, v)} no es un 1-símplice.")
        edge_set = set(self.edges)
        for (u, v, w) in self.faces:
            if len({u, v, w}) != 3:
                raise ValueError(f"Cara degenerada {(u, v, w)}.")
            for e in ((u, v), (v, w), (u, w)):
                if tuple(sorted(e)) not in edge_set:
                    raise ValueError(
                        f"Cara {(u, v, w)} referencia arista inexistente {e}."
                    )

    def _validate_chain_complex_exactness(self, tol: float = 1e-9) -> None:
        if self.boundary_2.size == 0:
            return
        residual_norm = float(la.norm(self.boundary_1 @ self.boundary_2))
        if residual_norm > tol:
            raise ValueError(f"Violación de ∂₁∂₂=0: ||∂₁∂₂|| = {residual_norm:.3e}")

    def _build_boundary_1(self) -> np.ndarray:
        B1 = np.zeros((self.num_vertices, len(self.edges)), dtype=np.float64)
        for e_idx, (u, v) in enumerate(self.edges):
            B1[u, e_idx] = -1.0
            B1[v, e_idx] = 1.0
        return B1

    def _build_boundary_2(self) -> np.ndarray:
        r"""∂[u,v,w] = [v,w] − [u,w] + [u,v] sobre la orientación canónica ordenada."""
        if not self.faces:
            return np.zeros((len(self.edges), 0), dtype=np.float64)
        B2 = np.zeros((len(self.edges), len(self.faces)), dtype=np.float64)
        edge_map = {e: idx for idx, e in enumerate(self.edges)}
        for f_idx, (u, v, w) in enumerate(self.faces):
            e_vw = edge_map.get(tuple(sorted((v, w))))
            e_uw = edge_map.get(tuple(sorted((u, w))))
            e_uv = edge_map.get(tuple(sorted((u, v))))
            if e_vw is not None:
                B2[e_vw, f_idx] += 1.0
            if e_uw is not None:
                B2[e_uw, f_idx] -= 1.0
            if e_uv is not None:
                B2[e_uv, f_idx] += 1.0
        return B2

    @property
    def coboundary_0(self) -> np.ndarray:
        r"""δ₀ = ∂₁ᵀ : C⁰ → C¹."""
        return self.boundary_1.T

    @property
    def coboundary_1(self) -> np.ndarray:
        r"""δ₁ = ∂₂ᵀ : C¹ → C²."""
        return self.boundary_2.T

    def connected_components(self) -> List[Set[int]]:
        adjacency: Dict[int, Set[int]] = {v: set() for v in range(self.num_vertices)}
        for (u, v) in self.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in range(self.num_vertices):
            if start in visited:
                continue
            stack, comp = [start], set()
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                stack.extend(adjacency[node] - comp)
            visited |= comp
            components.append(comp)
        return components

    def euler_characteristic(self) -> int:
        return self.num_vertices - len(self.edges) + len(self.faces)

    def compute_betti_numbers(self, tol: float = 1e-9) -> Tuple[int, int, int]:
        eigvals_0 = la.eigvalsh(self.laplacian_0)
        betti_0 = int(np.sum(np.abs(eigvals_0) < tol))
        eigvals_1 = la.eigvalsh(self.laplacian_1)
        betti_1 = int(np.sum(np.abs(eigvals_1) < tol))
        betti_2 = 0
        if self.laplacian_2.size > 0:
            eigvals_2 = la.eigvalsh(self.laplacian_2)
            betti_2 = int(np.sum(np.abs(eigvals_2) < tol))
        n_comp = len(self.connected_components())
        if n_comp != betti_0:
            logger.warning(
                "Discrepancia β₀ espectral (%d) vs. combinatoria (%d).",
                betti_0, n_comp,
            )
        return betti_0, betti_1, betti_2

    def verify_betti_euler_consistency(self, tol: float = 1e-9) -> bool:
        b0, b1, b2 = self.compute_betti_numbers(tol)
        return (b0 - b1 + b2) == self.euler_characteristic()

    def algebraic_connectivity(self, tol: float = 1e-12) -> float:
        r"""Valor de Fiedler: segundo autovalor de L₀."""
        evals = np.sort(la.eigvalsh(self.laplacian_0))
        if evals.size < 2:
            return 0.0
        return float(max(evals[1], 0.0)) if evals[1] > tol else 0.0

    def compute_spectral_gap(self, tol: float = 1e-9) -> float:
        eigvals_1 = np.sort(la.eigvalsh(self.laplacian_1))
        non_zero = eigvals_1[eigvals_1 > tol]
        return float(non_zero[0]) if len(non_zero) > 0 else 0.0

    def harmonic_1_forms(self, tol: float = 1e-9) -> np.ndarray:
        w, v = la.eigh(self.laplacian_1)
        mask = np.abs(w) < tol
        if not np.any(mask):
            return np.zeros((len(self.edges), 0), dtype=np.float64)
        return v[:, mask]

    def harmonic_2_forms(self, tol: float = 1e-9) -> np.ndarray:
        if self.laplacian_2.size == 0:
            return np.zeros((len(self.faces), 0), dtype=np.float64)
        w, v = la.eigh(self.laplacian_2)
        mask = np.abs(w) < tol
        if not np.any(mask):
            return np.zeros((len(self.faces), 0), dtype=np.float64)
        return v[:, mask]

    def hodge_decompose_1_form(
        self, omega: np.ndarray, rcond: float = 1e-10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        ω = ∂₂ α + ∂₁ᵀ β + γ,  γ ∈ ker L₁.
        Retorna (exacta, coexacta, armónica).
        """
        omega = np.asarray(omega, dtype=np.float64).reshape(-1)
        if omega.size != len(self.edges):
            raise ValueError("ω debe vivir en C₁ (dim = |E|).")
        B2 = self.boundary_2
        B1 = self.boundary_1
        if B2.size > 0 and B2.shape[1] > 0:
            exact = B2 @ la.lstsq(B2, omega, cond=rcond)[0]
        else:
            exact = np.zeros_like(omega)
        remainder = omega - exact
        coexact = B1.T @ la.lstsq(B1.T, remainder, cond=rcond)[0]
        harmonic = remainder - coexact
        return exact, coexact, harmonic


class NonReciprocalTellegenNetwork:
    r"""
    Red AC no recíproca sobre el 1-esqueleto, con tierra por componente
    conexa (β₀ ≥ 1). Giroscopios (Y_b − Y_bᵀ ≠ 0) modelan fricción no
    conservativa de la economía de obra.

    Tellegen (identidad topológica):  Σ_e v_e i_e* = V† I_nodal.
    Pasividad: λ_min((Y_b + Y_b†)/2) ≥ −ε.
    """

    def __init__(
        self, graph: SimplicialHodgeGraph, frequency_rad_s: float = 50.0
    ) -> None:
        self.graph = graph
        self.omega = float(frequency_rad_s)
        self.num_edges = len(self.graph.edges)
        self.branch_admittance = self._build_admittance_matrix()

    def _build_admittance_matrix(self) -> np.ndarray:
        dim = self.num_edges
        Yb = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            r_k = 0.8 + 0.05 * (i + 1)
            x_k = self.omega * 0.02 - 1.0 / (self.omega * 0.08 + 1e-6)
            Yb[i, i] = 1.0 / complex(r_k, x_k)
            if i + 1 < dim:
                gyration = 0.15 * (i + 1)
                Yb[i, i + 1] += complex(0.0, gyration)
                Yb[i + 1, i] -= complex(0.0, gyration)
        return Yb

    def reciprocity_defect_norm(self) -> float:
        antisym = 0.5 * (self.branch_admittance - self.branch_admittance.T)
        sv = la.svdvals(antisym)
        return float(np.max(sv)) if len(sv) > 0 else 0.0

    def passivity_margin(self) -> float:
        r"""λ_min((Y_b + Y_b†)/2). Negativo ⇒ violación de pasividad."""
        herm = 0.5 * (self.branch_admittance + self.branch_admittance.conj().T)
        return float(np.min(la.eigvalsh(herm)))

    def compute_bus_admittance_matrix(self) -> np.ndarray:
        B1 = self.graph.boundary_1
        return B1 @ self.branch_admittance @ B1.T

    def solve_network_dissipation(
        self, current_stimulus: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        r"""
        Resuelve Y_bus V = I con grounding por componente.
        Retorna (P = Re Σ v_e i_e*,  branch_v,  V_nodes).
        """
        n = self.graph.num_vertices
        components = self.graph.connected_components()
        ground_nodes = {min(c) for c in components}
        free_nodes = [i for i in range(n) if i not in ground_nodes]
        B1 = self.graph.boundary_1
        Y_bus = B1 @ self.branch_admittance @ B1.T
        V_nodes = np.zeros(n, dtype=np.complex128)
        if free_nodes:
            Y_red = Y_bus[np.ix_(free_nodes, free_nodes)]
            I_red = current_stimulus[free_nodes]
            try:
                V_red = la.solve(Y_red, I_red)
            except la.LinAlgError:
                logger.warning("Y_red casi-singular: lstsq regularizado.")
                V_red, *_ = la.lstsq(Y_red, I_red)
            for local_idx, node in enumerate(free_nodes):
                V_nodes[node] = V_red[local_idx]
        branch_v = B1.T @ V_nodes
        branch_i = self.branch_admittance @ branch_v
        dissipation = float(np.real(np.sum(branch_v * np.conj(branch_i))))
        return dissipation, branch_v, V_nodes

    def kcl_residual(self, v_nodes: np.ndarray, i_inj: np.ndarray) -> float:
        I_pred = self.compute_bus_admittance_matrix() @ v_nodes
        denom = max(float(np.linalg.norm(i_inj)), 1.0)
        return float(np.linalg.norm(I_pred - i_inj) / denom)

    def verify_tellegen_conservation(
        self, branch_v: np.ndarray, v_nodes: np.ndarray
    ) -> float:
        B1 = self.graph.boundary_1
        Y_bus = B1 @ self.branch_admittance @ B1.T
        I_full = Y_bus @ v_nodes
        branch_i = self.branch_admittance @ branch_v
        lhs = np.sum(branch_v * np.conj(branch_i))
        rhs = np.sum(np.conj(v_nodes) * I_full)
        denom = max(abs(rhs), 1e-12)
        return float(abs(lhs - rhs) / denom)


class OpenQuantumDynamicsSeed(ABC):
    r"""
    Germen formal de la flecha H : cartucho ↦ 𝔥𝔢𝔯(ℋ_dream).

    Cierra el andamiaje de FASE-1. FASE-2 *continúa* exactamente en
    lift_enclave_hamiltonian: el motor GKSL delega aquí y sólo aplica
    la compresión de superselección Π_enc.
    """

    @abstractmethod
    def lift_enclave_hamiltonian(self, hilbert_dim: int) -> np.ndarray:
        r"""
        Produce H = H† (antes de Π_enc) a partir del cartucho.

        CONTINÚA EN FASE-2:
        NonHermitianLindbladMasterEngine._build_effective_hamiltonian.
        """
        ...


@dataclass(frozen=True, slots=True)
class CategoricalCircuitCartridge(OpenQuantumDynamicsSeed):
    r"""
    NEXO FORMAL TERMINAL DE LA FASE 1.

    Cartucho TOON elevado a entidad geométrica y física sobre un haz
    celular exacto, valuado en Ω₄ vía j ∘ χ (clausura de Lawvere-Tierney
    del funcional de coherencia). Argumento estricto de entrada a FASE 2.

    Último método: lift_enclave_hamiltonian — germen de GKSL.
    """

    cartridge_id: str
    scenario_type: str
    cost_delta_ratio: float
    token_count: int
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    euler_betti_consistent: bool
    hodge_spectral_gap: float
    algebraic_connectivity: float
    hodge_decomposition_residual: float
    circuit_dissipation_watts: float
    reciprocity_defect: float
    passivity_margin: float
    tellegen_residual: float
    kcl_residual: float
    clifford_gauge_perturbation: BiquaternionClifford
    clifford_cstar_residual: float
    hodge_graph: SimplicialHodgeGraph
    tellegen_network: NonReciprocalTellegenNetwork
    spectral_coherence_index: float
    topos_heyting_sieve: HeytingToposAlgebra
    is_homologically_isolated: bool
    metadata_payload: Dict[str, Any]

    _KAPPA_BETTI: ClassVar[float] = 1.2
    _KAPPA_DISSIPATION: ClassVar[float] = 14.0
    _KAPPA_COST: ClassVar[float] = 0.55

    @staticmethod
    def _spectral_coherence_functional(
        betti_1: int,
        betti_2: int,
        dissipation: float,
        cost_delta_ratio: float,
        is_dream_state: bool,
    ) -> float:
        r"""
        C = 𝟙[aislado] · exp(−(β₁+β₂)/κ_β) · exp(−|D|/κ_D) · exp(−|Δc|/κ_c).
        """
        if not is_dream_state:
            return 0.0
        kb = CategoricalCircuitCartridge._KAPPA_BETTI
        kd = CategoricalCircuitCartridge._KAPPA_DISSIPATION
        kc = CategoricalCircuitCartridge._KAPPA_COST
        return (
            math.exp(-(betti_1 + betti_2) / kb)
            * math.exp(-abs(dissipation) / kd)
            * math.exp(-abs(cost_delta_ratio) / kc)
        )

    @staticmethod
    def _classify_sieve(coherence: float) -> HeytingToposAlgebra:
        if coherence >= 0.70:
            return HeytingToposAlgebra.VERUM_COHERENT
        if coherence >= 0.40:
            return HeytingToposAlgebra.TOPOLOGICAL_STABLE
        if coherence >= 0.12:
            return HeytingToposAlgebra.BOUNDARY_CRITICAL
        return HeytingToposAlgebra.VETOED_ABSURDUM

    @classmethod
    def synthesize_cartridge(
        cls,
        cartridge_id: str,
        scenario_type: str,
        cost_delta_ratio: float,
        synthetic_betti_1: int,
        is_dream_state: bool,
        payload: Dict[str, Any],
        deterministic_extra_edge: Optional[Tuple[int, int]] = None,
    ) -> "CategoricalCircuitCartridge":
        r"""Constructor unificado de FASE 1 (preludio de lift_enclave_hamiltonian)."""
        num_v = 6
        edges = [
            (0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5), (5, 0)
        ]
        faces = [(0, 1, 2), (2, 3, 4)]

        if deterministic_extra_edge is not None:
            u, v = deterministic_extra_edge
            if u != v:
                edges.append(tuple(sorted((u, v))))
        if synthetic_betti_1 >= 1:
            edges.append((1, 4))
        if synthetic_betti_1 >= 2:
            edges.append((3, 0))
        if synthetic_betti_1 >= 3:
            edges.append((1, 5))

        seen: Set[Tuple[int, int]] = set()
        dedup_edges: List[Tuple[int, int]] = []
        for e in edges:
            e_s = tuple(sorted(e))
            if e_s not in seen:
                seen.add(e_s)
                dedup_edges.append(e_s)

        graph = SimplicialHodgeGraph(
            num_vertices=num_v, edges=dedup_edges, faces=faces
        )
        b0, b1, b2 = graph.compute_betti_numbers()
        chi = graph.euler_characteristic()
        euler_ok = graph.verify_betti_euler_consistency()
        if not euler_ok:
            logger.warning(
                "Inconsistencia de Euler-Poincaré en cartucho %s: χ=%d.",
                cartridge_id, chi,
            )
        spectral_gap = graph.compute_spectral_gap()
        fiedler = graph.algebraic_connectivity()

        tellegen = NonReciprocalTellegenNetwork(graph=graph)
        stimulus = np.zeros(num_v, dtype=np.complex128)
        stimulus[0] = complex(1.0 + cost_delta_ratio, 0.2)
        stimulus[-1] = -complex(1.0 + cost_delta_ratio, 0.2)
        dissipation, branch_v, v_nodes = tellegen.solve_network_dissipation(stimulus)
        reciprocity_defect = tellegen.reciprocity_defect_norm()
        passivity = tellegen.passivity_margin()
        tellegen_res = tellegen.verify_tellegen_conservation(branch_v, v_nodes)
        kcl = tellegen.kcl_residual(v_nodes, stimulus)

        omega_probe = np.real(branch_v)
        if omega_probe.size == len(graph.edges) and omega_probe.size > 0:
            ex, co, ha = graph.hodge_decompose_1_form(omega_probe)
            hodge_res = float(np.linalg.norm(omega_probe - (ex + co + ha)))
        else:
            hodge_res = 0.0

        clifford_gauge = BiquaternionClifford(
            w_re=1.0,
            w_im=cost_delta_ratio * 0.1,
            x_re=cost_delta_ratio * 0.5,
            x_im=0.0,
            y_re=0.0,
            y_im=cost_delta_ratio * 0.2,
            z_re=float(b1) * 0.3,
            z_im=0.0,
        )

        coherence = cls._spectral_coherence_functional(
            b1, b2, dissipation, cost_delta_ratio, is_dream_state
        )
        sieve = cls._classify_sieve(coherence)
        sieve_closed = sieve.lawvere_tierney_closure()

        token_count = max(1, 24 + int(abs(cost_delta_ratio) * 40) + 8 * b1)

        return cls(
            cartridge_id=cartridge_id,
            scenario_type=scenario_type,
            cost_delta_ratio=cost_delta_ratio,
            token_count=token_count,
            betti_0=b0,
            betti_1=b1,
            betti_2=b2,
            euler_characteristic=chi,
            euler_betti_consistent=euler_ok,
            hodge_spectral_gap=spectral_gap,
            algebraic_connectivity=fiedler,
            hodge_decomposition_residual=hodge_res,
            circuit_dissipation_watts=dissipation,
            reciprocity_defect=reciprocity_defect,
            passivity_margin=passivity,
            tellegen_residual=tellegen_res,
            kcl_residual=kcl,
            clifford_gauge_perturbation=clifford_gauge,
            clifford_cstar_residual=clifford_gauge.cstar_residual(),
            hodge_graph=graph,
            tellegen_network=tellegen,
            spectral_coherence_index=coherence,
            topos_heyting_sieve=sieve_closed,
            is_homologically_isolated=is_dream_state,
            metadata_payload=payload,
        )

    # ══════════════════════════════════════════════════════════════════════
    # HAND-OFF  FASE 1 → FASE 2
    # Último método de la FASE 1. Su salida H = H† (pre-enclave) es el
    # generador unitario del Liouvilliano GKSL. CONTINÚA EN
    # NonHermitianLindbladMasterEngine._build_effective_hamiltonian.
    # ══════════════════════════════════════════════════════════════════════
    def lift_enclave_hamiltonian(self, hilbert_dim: int) -> np.ndarray:
        r"""
        Flecha H: cartucho ↦ H ∈ 𝔥𝔢𝔯(ℋₙ)  (antes de Π_enc).

        Construcción:
            • niveles diagonales modulados por la brecha de Hodge y la
              disipación de Tellegen;
            • bloque 2×2 = φ(q)_herm  (sólo la parte hermítica del gauge
              de Clifford es admisible en un H de GKSL).

        CONTINÚA EN FASE-2: el motor aplica Π_enc = P_d (·) P_d y añade
        los saltos de Lindblad (γ_k ≥ 0).
        """
        if hilbert_dim < 1:
            raise ValueError("hilbert_dim debe ser ≥ 1.")
        dim = int(hilbert_dim)
        H = np.zeros((dim, dim), dtype=np.complex128)
        gap = self.hodge_spectral_gap
        for i in range(dim):
            H[i, i] = (i + 1) * gap + (self.circuit_dissipation_watts * 0.02)
        herm = self.clifford_gauge_perturbation.hermitian_part()
        block = min(2, dim)
        H[0:block, 0:block] += herm[:block, :block] * 0.25
        return 0.5 * (H + H.conj().T)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — C*/BANACH, ENCLAVE NO-SIGNALING, GKSL ADAPTATIVO Y CAMPO METABOLIZADO
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método operativo (_build_effective_hamiltonian) ES
# la continuación de lift_enclave_hamiltonian. El último (evolve_and_certify)
# produce MetabolizedPerturbationField, germen formal de FASE 3.
# ══════════════════════════════════════════════════════════════════════════════


class BanachSpectralAlgebra:
    r"""
    C*-álgebra B(ℋ) de dimensión finita. Normas de Schatten ‖·‖₁, ‖·‖₂,
    ‖·‖_∞, variedad 𝔇(ℋ), Umegaki y geometría de Bures.

        S(ρ‖σ) = Tr(ρ log ρ) − Tr(ρ log σ)     (Klein: ≥ 0)
        F(ρ,σ) = [Tr √(√ρ σ √ρ)]²               (Uhlmann–Jozsa)
        D_B    = √(2(1 − √F))                   (Bures)
    """

    SPECTRUM_FLOOR: Final[float] = 1e-15

    @staticmethod
    def project_to_state_manifold(rho: np.ndarray) -> np.ndarray:
        rho_h = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals = np.maximum(np.real(eigvals), BanachSpectralAlgebra.SPECTRUM_FLOOR)
        eigvals /= np.sum(eigvals)
        proj = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        return 0.5 * (proj + proj.conj().T)

    @staticmethod
    def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-8) -> bool:
        if float(la.norm(rho - rho.conj().T)) > tol:
            return False
        eigvals = la.eigvalsh(0.5 * (rho + rho.conj().T))
        if np.any(eigvals < -tol):
            return False
        return abs(np.real(np.trace(rho)) - 1.0) < tol

    @staticmethod
    def schatten_1_norm(A: np.ndarray) -> float:
        return float(np.sum(la.svdvals(A)))

    @staticmethod
    def schatten_2_norm(A: np.ndarray) -> float:
        return float(np.sqrt(np.real(np.trace(A.conj().T @ A))))

    @staticmethod
    def schatten_infty_norm(A: np.ndarray) -> float:
        s = la.svdvals(A)
        return float(s[0]) if len(s) > 0 else 0.0

    @staticmethod
    def von_neumann_entropy(rho: np.ndarray) -> float:
        eigvals = la.eigvalsh(rho)
        eigvals = eigvals[eigvals > BanachSpectralAlgebra.SPECTRUM_FLOOR]
        return -float(np.sum(eigvals * np.log2(eigvals)))

    @staticmethod
    def purity(rho: np.ndarray) -> float:
        return float(np.real(np.trace(rho @ rho)))

    @staticmethod
    def _matrix_log_regularized(A: np.ndarray, floor: float = 1e-12) -> np.ndarray:
        eigvals, eigvecs = la.eigh(0.5 * (A + A.conj().T))
        log_eigvals = np.log(np.maximum(eigvals, floor))
        return eigvecs @ np.diag(log_eigvals) @ eigvecs.conj().T

    @staticmethod
    def quantum_relative_entropy(
        rho: np.ndarray, sigma: np.ndarray, floor: float = 1e-12
    ) -> float:
        log_rho = BanachSpectralAlgebra._matrix_log_regularized(rho, floor)
        log_sigma = BanachSpectralAlgebra._matrix_log_regularized(sigma, floor)
        val = np.real(np.trace(rho @ (log_rho - log_sigma))) / math.log(2.0)
        return float(max(val, 0.0))

    @staticmethod
    def quantum_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
        rho_h = 0.5 * (rho + rho.conj().T)
        sigma_h = 0.5 * (sigma + sigma.conj().T)
        sqrt_rho = np.real_if_close(la.sqrtm(rho_h), tol=1e6)
        inner = sqrt_rho @ sigma_h @ sqrt_rho.conj().T
        inner_h = 0.5 * (inner + inner.conj().T)
        eigvals = np.clip(np.real(la.eigvalsh(inner_h)), 0.0, None)
        fidelity_val = float(np.sum(np.sqrt(eigvals)) ** 2)
        return float(np.clip(fidelity_val, 0.0, 1.0))

    @staticmethod
    def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
        f = BanachSpectralAlgebra.quantum_fidelity(rho, sigma)
        return float(math.sqrt(max(2.0 * (1.0 - math.sqrt(max(f, 0.0))), 0.0)))

    @staticmethod
    def cstar_residual(rho: np.ndarray) -> float:
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)


class NonCommutativeNoSignalingEnclave:
    r"""
    Garantía de aislamiento homológico por teorema del conmutador:

        𝒜_dream ⊂ B(ℋ),   𝒜_real ⊂ B(ℋ),   [𝒜_dream, 𝒜_real] = {0}

    con descomposición ℋ = ℋ_dream ⊕ ℋ_physical y resolución de la
    identidad {P_d, P_p}:

        P_d + P_p = I,   P_d P_p = 0,   P_d² = P_d = P_d†.

    Distinción honesta de dos mapas:
      • enforce_strict_enclave  O ↦ P_d O P_d     (compresión, no CPTP)
      • pinching_channel        ρ ↦ P_d ρ P_d + P_p ρ P_p   (CPTP, pinching)
    """

    LEAKAGE_TOL: Final[float] = 1e-12

    def __init__(self, hilbert_dim: int = 4, num_physical_modes: int = 1) -> None:
        if hilbert_dim < 2:
            raise ValueError("hilbert_dim debe ser ≥ 2 para partir dream/physical.")
        self.dim = int(hilbert_dim)
        self.num_physical = max(1, min(num_physical_modes, hilbert_dim - 1))
        cutoff = hilbert_dim - self.num_physical
        self.P_dream = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.P_physical = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(cutoff):
            self.P_dream[i, i] = 1.0
        for i in range(cutoff, hilbert_dim):
            self.P_physical[i, i] = 1.0

    def verify_projector_completeness(self, tol: float = 1e-12) -> bool:
        identity_defect = float(
            la.norm(self.P_dream + self.P_physical - np.eye(self.dim))
        )
        orthogonality_defect = float(la.norm(self.P_dream @ self.P_physical))
        idem_d = float(la.norm(self.P_dream @ self.P_dream - self.P_dream))
        idem_p = float(la.norm(self.P_physical @ self.P_physical - self.P_physical))
        return (
            identity_defect < tol
            and orthogonality_defect < tol
            and idem_d < tol
            and idem_p < tol
        )

    def verify_isolation_commutator(
        self, dream_operator: np.ndarray
    ) -> Tuple[bool, float]:
        r"""δ = ‖[O_dream, P_physical]‖_∞. δ > tol ⇒ fuga hacia hardware real."""
        comm = dream_operator @ self.P_physical - self.P_physical @ dream_operator
        leakage = BanachSpectralAlgebra.schatten_infty_norm(comm)
        return bool(leakage < self.LEAKAGE_TOL), leakage

    def enforce_strict_enclave(self, dream_operator: np.ndarray) -> np.ndarray:
        r"""Compresión de superselección: O ↦ P_d O P_d  (no es un canal CPTP)."""
        return self.P_dream @ dream_operator @ self.P_dream

    def pinching_channel(self, rho: np.ndarray) -> np.ndarray:
        r"""Canal CPTP de pinching: Φ(ρ) = P_d ρ P_d + P_p ρ P_p."""
        return self.P_dream @ rho @ self.P_dream + self.P_physical @ rho @ self.P_physical


class NonHermitianLindbladMasterEngine:
    r"""
    CONTINUACIÓN FORMAL de CategoricalCircuitCartridge.lift_enclave_hamiltonian.

    Semigrupo CPTP de GKSL (Kossakowski: γ_k ≥ 0):

        dρ/dt = −i[H_eff, ρ] + Σ_k γ_k (L_k ρ L_k† − ½ {L_k† L_k, ρ})

    H_eff es estrictamente hermítico (requisito de Lindblad). El epíteto
    «NoHermitian» se reserva al generador *condicional* de trayectorias
    cuánticas (Dalibard–Castin–Mølmer / Carmichael), del cual GKSL es el
    promedio estocástico:

        H_cond = H_eff − (i/2) Σ_k γ_k L_k† L_k  (+ anti-hermítico de Clifford)

    Γ_escape = Σ_k γ_k Tr(ρ L_k† L_k) es la tasa de primer salto
    (diagnóstico de tiempo de ruina presupuestaria).
    """

    GAMMA_FLOOR: Final[float] = 0.0
    TRACE_DEFECT_LOG: Final[float] = 1e-6

    def __init__(
        self, cartridge: CategoricalCircuitCartridge, hilbert_dim: int = 4
    ) -> None:
        r"""Primer consumidor de FASE-2: ancla el cartucho producido por FASE-1."""
        self.cartridge = cartridge
        self.dim = int(hilbert_dim)
        self.enclave = NonCommutativeNoSignalingEnclave(hilbert_dim=self.dim)
        if not self.enclave.verify_projector_completeness():
            raise RuntimeError(
                "El enclave de no-señalización no satisface completitud de proyectores."
            )
        self.H_eff = self._build_effective_hamiltonian()
        self.jump_operators = self._build_jump_operators()

    def _build_effective_hamiltonian(self) -> np.ndarray:
        r"""
        CONTINUACIÓN de lift_enclave_hamiltonian:
        delegación estricta + compresión Π_enc.
        """
        H_seed = self.cartridge.lift_enclave_hamiltonian(self.dim)
        H_herm = 0.5 * (H_seed + H_seed.conj().T)
        return self.enclave.enforce_strict_enclave(H_herm)

    def _build_jump_operators(self) -> List[Tuple[float, np.ndarray]]:
        gamma_betti = max(self.GAMMA_FLOOR, 0.1 * (self.cartridge.betti_1 + 1.0))
        L_dephase = np.diag(
            [math.sqrt(i + 1) for i in range(self.dim)]
        ).astype(np.complex128)
        gamma_tellegen = max(
            self.GAMMA_FLOOR, 0.01 * abs(self.cartridge.circuit_dissipation_watts)
        )
        L_diss = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim - 2):
            L_diss[i, i + 1] = 1.0
        return [
            (gamma_betti, self.enclave.enforce_strict_enclave(L_dephase)),
            (gamma_tellegen, self.enclave.enforce_strict_enclave(L_diss)),
        ]

    def effective_non_hermitian_conditional_hamiltonian(self) -> np.ndarray:
        r"""H_cond de la evolución sin salto (instrumento de primer escape)."""
        H_cond = self.H_eff.astype(np.complex128).copy()
        for gamma, L in self.jump_operators:
            H_cond = H_cond - 0.5j * gamma * (L.conj().T @ L)
        anti_herm = self.cartridge.clifford_gauge_perturbation.anti_hermitian_part()
        padded = np.zeros_like(H_cond)
        block = min(2, self.dim)
        padded[0:block, 0:block] = anti_herm[:block, :block] * 0.25
        H_cond = H_cond + self.enclave.enforce_strict_enclave(padded)
        return H_cond

    def jump_rate(self, rho: np.ndarray) -> float:
        r"""Γ(ρ) = Σ_k γ_k Tr(ρ L_k† L_k)  ≥ 0. Tiempo medio de primer salto 1/Γ."""
        rate = 0.0
        for gamma, L in self.jump_operators:
            rate += gamma * float(np.real(np.trace(rho @ (L.conj().T @ L))))
        return max(rate, 0.0)

    def mean_first_jump_time(self, rho: np.ndarray) -> float:
        gamma = self.jump_rate(rho)
        if gamma <= 1e-18:
            return float("inf")
        return 1.0 / gamma

    def _liouvillian(self, r: np.ndarray) -> np.ndarray:
        comm = -1j * (self.H_eff @ r - r @ self.H_eff)
        diss = np.zeros_like(r, dtype=np.complex128)
        for gamma, L in self.jump_operators:
            L_dag = L.conj().T
            L_dag_L = L_dag @ L
            diss += gamma * (
                L @ r @ L_dag - 0.5 * (L_dag_L @ r + r @ L_dag_L)
            )
        return comm + diss

    def _rk4_step(self, rho: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._liouvillian(rho)
        k2 = self._liouvillian(rho + 0.5 * dt * k1)
        k3 = self._liouvillian(rho + 0.5 * dt * k2)
        k4 = self._liouvillian(rho + dt * k3)
        return rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _adaptive_propagate(
        self,
        rho: np.ndarray,
        dt: float,
        tol: float = 1e-7,
        max_depth: int = 6,
        depth: int = 0,
    ) -> np.ndarray:
        r"""
        RK4 + duplicación + Richardson:
            ρ* = ρ_{h/2,h/2} + (ρ_{h/2,h/2} − ρ_h)/15
        Tras aceptar, se proyecta a 𝔇(ℋ).
        """
        full_step = self._rk4_step(rho, dt)
        half_step = self._rk4_step(rho, dt / 2.0)
        two_half_steps = self._rk4_step(half_step, dt / 2.0)
        local_error = BanachSpectralAlgebra.schatten_2_norm(two_half_steps - full_step)
        if local_error < tol or depth >= max_depth:
            rho_star = two_half_steps + (two_half_steps - full_step) / 15.0
            return BanachSpectralAlgebra.project_to_state_manifold(rho_star)
        left = self._adaptive_propagate(
            rho, dt / 2.0, tol / 2.0, max_depth, depth + 1
        )
        return self._adaptive_propagate(
            left, dt / 2.0, tol / 2.0, max_depth, depth + 1
        )

    def propagate_state(
        self, base_rho: np.ndarray, dt: float = 0.08, tol: float = 1e-7
    ) -> np.ndarray:
        rho0 = BanachSpectralAlgebra.project_to_state_manifold(base_rho)
        trace_defect = abs(np.real(np.trace(self._liouvillian(rho0))))
        if trace_defect > self.TRACE_DEFECT_LOG:
            logger.debug("Defecto de traza del generador GKSL: %.3e", trace_defect)
        rho_evolved_raw = self._adaptive_propagate(rho0, dt, tol)
        return BanachSpectralAlgebra.project_to_state_manifold(rho_evolved_raw)


@dataclass(frozen=True, slots=True)
class MetabolizedPerturbationField:
    r"""
    NEXO FORMAL TERMINAL DE LA FASE 2.

    Estado metabolizado por GKSL, certificado bajo aislamiento no
    conmutativo (resolución de proyectores verificada) y normas de Banach,
    clasificado en Ω₄ por un funcional de coherencia metabólica. Argumento
    estricto de entrada a FASE 3.

    Último método de clase: evolve_and_certify.
    CONTINÚA EN FASE-3:
    SpectralImmuneVaccineEngine.construct_from_metabolized_field.
    """

    source_cartridge: CategoricalCircuitCartridge
    rho_perturbed: np.ndarray
    dirichlet_spectral_energy: float
    von_neumann_entropy: float
    purity: float
    relative_entropy_to_base: float
    bures_distance_to_base: float
    cstar_residual: float
    no_signaling_leakage_norm: float
    projector_completeness_ok: bool
    is_strictly_isolated: bool
    jump_rate: float
    mean_first_jump_time: float
    metabolic_coherence_index: float
    topos_heyting_evaluation: HeytingToposAlgebra

    _KAPPA_ENERGY: ClassVar[float] = 12.0
    _KAPPA_RELATIVE_ENTROPY: ClassVar[float] = 5.0

    @staticmethod
    def _metabolic_coherence_functional(
        purity: float,
        dirichlet_energy: float,
        relative_entropy: float,
        is_isolated: bool,
    ) -> float:
        r"""
        M = 𝟙[aislado] · pur(ρ) · exp(−E_D/κ_E) · exp(−S_rel/κ_S).
        """
        if not is_isolated:
            return 0.0
        kE = MetabolizedPerturbationField._KAPPA_ENERGY
        kS = MetabolizedPerturbationField._KAPPA_RELATIVE_ENTROPY
        return (
            purity
            * math.exp(-abs(dirichlet_energy) / kE)
            * math.exp(-abs(relative_entropy) / kS)
        )

    @staticmethod
    def _classify_metabolic_state(coherence: float) -> HeytingToposAlgebra:
        if coherence >= 0.55:
            return HeytingToposAlgebra.VERUM_COHERENT
        if coherence >= 0.30:
            return HeytingToposAlgebra.TOPOLOGICAL_STABLE
        if coherence >= 0.08:
            return HeytingToposAlgebra.BOUNDARY_CRITICAL
        return HeytingToposAlgebra.VETOED_ABSURDUM

    @classmethod
    def evolve_and_certify(
        cls, cartridge: CategoricalCircuitCartridge, base_rho: np.ndarray
    ) -> "MetabolizedPerturbationField":
        r"""
        ÚLTIMO método de FASE-2: Φ_t ∘ Π_enc ∘ H.

        CONTINÚA EN FASE-3 (inmunización I sobre rho_perturbed).
        """
        engine = NonHermitianLindbladMasterEngine(
            cartridge=cartridge, hilbert_dim=base_rho.shape[0]
        )
        rho_p = engine.propagate_state(base_rho=base_rho)

        completeness_ok = engine.enclave.verify_projector_completeness()
        is_isolated, leakage = engine.enclave.verify_isolation_commutator(rho_p)
        strict_isolated = bool(
            is_isolated and cartridge.is_homologically_isolated and completeness_ok
        )

        purity = BanachSpectralAlgebra.purity(rho_p)
        entropy = BanachSpectralAlgebra.von_neumann_entropy(rho_p)
        rel_entropy = BanachSpectralAlgebra.quantum_relative_entropy(rho_p, base_rho)
        bures = BanachSpectralAlgebra.bures_distance(rho_p, base_rho)
        cstar = BanachSpectralAlgebra.cstar_residual(rho_p)

        comm_h = rho_p @ engine.H_eff - engine.H_eff @ rho_p
        dirichlet_energy = (
            0.5 * float(np.real(np.trace(rho_p @ engine.H_eff)))
            + 0.5 * (BanachSpectralAlgebra.schatten_2_norm(comm_h) ** 2)
            + 0.1 * cartridge.circuit_dissipation_watts
        )
        gamma = engine.jump_rate(rho_p)
        t_jump = engine.mean_first_jump_time(rho_p)

        coherence = cls._metabolic_coherence_functional(
            purity, dirichlet_energy, rel_entropy, strict_isolated
        )
        heyting_verdict = cls._classify_metabolic_state(coherence)
        final_verdict = cartridge.topos_heyting_sieve.meet(heyting_verdict)

        return cls(
            source_cartridge=cartridge,
            rho_perturbed=rho_p,
            dirichlet_spectral_energy=dirichlet_energy,
            von_neumann_entropy=entropy,
            purity=purity,
            relative_entropy_to_base=rel_entropy,
            bures_distance_to_base=bures,
            cstar_residual=cstar,
            no_signaling_leakage_norm=leakage,
            projector_completeness_ok=completeness_ok,
            is_strictly_isolated=strict_isolated,
            jump_rate=gamma,
            mean_first_jump_time=t_jump,
            metabolic_coherence_index=coherence,
            topos_heyting_evaluation=final_verdict,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INMUNIZACIÓN, MERKLE, WAKE-SLEEP, AUDITORÍA Y PASAPORTE SOBERANO
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método (construct_from_metabolized_field) consume
# MetabolizedPerturbationField, valor de retorno del último método de
# FASE-2. Aquí se realiza I (vacuna), V (meet Ω₄), el sello Merkle y la
# orquestación REM del metacórtex.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    r"""Prueba de inclusión Merkle (camino de hermanos, convención Bitcoin/CT)."""

    leaf_hash: str
    siblings: Tuple[str, ...]
    index: int
    root: str

    def verify(self) -> bool:
        node = bytes.fromhex(self.leaf_hash)
        idx = self.index
        for sib_hex in self.siblings:
            sib = bytes.fromhex(sib_hex)
            if idx % 2 == 0:
                node = hashlib.sha512(node + sib).digest()
            else:
                node = hashlib.sha512(sib + node).digest()
            idx //= 2
        return node.hex() == self.root


@dataclass(frozen=True, slots=True)
class OniricScenarioCertificate:
    r"""Certificado forense del sueño contrafactual REM."""

    dream_id: str
    dreamer_agent_id: str
    cartridge_id: str
    scenario_type: str
    heyting_verdict: HeytingToposAlgebra
    dirichlet_energy: float
    purity: float
    entropy: float
    quantum_fidelity: float
    bures_distance: float
    cstar_residual: float
    spectral_coverage_mass: float
    projector_idempotency_residual: float
    isolation_guarantee: bool
    no_signaling_leakage: float
    jump_rate: float
    mean_first_jump_time: float
    immune_vaccine_effective: bool
    learning_rate_applied: float
    merkle_sha512_provenance: str
    timestamp_utc: float

    def is_vetoed(self) -> bool:
        return self.heyting_verdict == HeytingToposAlgebra.VETOED_ABSURDUM

    def is_immune(self) -> bool:
        return (
            not self.is_vetoed()
            and self.isolation_guarantee
            and self.immune_vaccine_effective
        )


@dataclass(frozen=True, slots=True)
class MacroWakeSleepAuditReport:
    r"""Reporte de la transición de fase REM y vacunación masiva."""

    audit_cycle_id: str
    total_scenarios_simulated: int
    vaccines_absorbed_count: int
    wake_entropy: float
    sleep_entropy: float
    entropy_reduction_ratio: float
    overall_topos_verdict: HeytingToposAlgebra
    merkle_leaf_count: int
    merkle_root_hash: str
    merkle_proofs_ok: bool
    execution_duration_ms: float


class SpectralImmuneVaccineEngine:
    r"""
    CONTINUACIÓN FORMAL de MetabolizedPerturbationField.evolve_and_certify.

    Proyector de inmunidad sobre el subespacio de cobertura de masa
    espectral (análogo PCA):

        P_vac = Σ_{i=1}^{k} |v_i⟩⟨v_i|,
        k = min{ m : Σ_{i=1}^{m} λ_i ≥ f_cov },   λ₁ ≥ λ₂ ≥ ⋯

    Invariantes: P² = P = P† (se reporta el residuo de idempotencia).
    """

    COVERAGE_FRACTION: Final[float] = 0.75
    RANK_FRACTION_CAP: Final[float] = 0.75
    MASS_FLOOR: Final[float] = 1e-15

    @classmethod
    def construct_from_metabolized_field(
        cls,
        perturbed_state: MetabolizedPerturbationField,
        coverage_fraction: float = COVERAGE_FRACTION,
    ) -> Tuple[np.ndarray, bool, float, float]:
        r"""
        CONTINUACIÓN de evolve_and_certify: consume rho_perturbed.

        Retorna (P_vac, efectiva, masa_retenida, residuo_idempotencia).
        """
        return cls.construct_spectral_vaccine(perturbed_state, coverage_fraction)

    @classmethod
    def construct_spectral_vaccine(
        cls,
        perturbed_state: MetabolizedPerturbationField,
        coverage_fraction: float = COVERAGE_FRACTION,
    ) -> Tuple[np.ndarray, bool, float, float]:
        rho = perturbed_state.rho_perturbed
        dim = rho.shape[0]
        if perturbed_state.topos_heyting_evaluation == HeytingToposAlgebra.VETOED_ABSURDUM:
            eye = np.eye(dim, dtype=np.complex128) / dim
            return eye, False, 0.0, 0.0

        eigvals, eigvecs = la.eigh(rho)
        order = np.argsort(eigvals)[::-1]
        eigvals_sorted = np.clip(eigvals[order], 0.0, None)
        eigvecs_sorted = eigvecs[:, order]
        total_mass = float(np.sum(eigvals_sorted))
        if total_mass <= cls.MASS_FLOOR:
            eye = np.eye(dim, dtype=np.complex128) / dim
            return eye, False, 0.0, 0.0

        cumulative = np.cumsum(eigvals_sorted) / total_mass
        k = int(np.searchsorted(cumulative, coverage_fraction) + 1)
        k = min(max(k, 1), dim)
        retained_mass = float(cumulative[k - 1])
        P_vac = eigvecs_sorted[:, :k] @ eigvecs_sorted[:, :k].conj().T
        P_vac = 0.5 * (P_vac + P_vac.conj().T)
        idem_res = float(la.norm(P_vac @ P_vac - P_vac, "fro"))
        is_effective = bool(
            retained_mass >= coverage_fraction
            and perturbed_state.is_strictly_isolated
            and k <= math.ceil(dim * cls.RANK_FRACTION_CAP)
        )
        return P_vac, is_effective, retained_mass, idem_res


class TOONOniricDreamerAgent:
    r"""
    SOBERANO AGENTE SIMULADOR ONÍRICO (FASE REM).

    Orquesta 𝒟 = V ∘ I ∘ Φ_t ∘ Π_enc ∘ H ∘ K:

      FASE 1: cartucho TOON → complejo de Hodge + Tellegen
              → lift_enclave_hamiltonian
      FASE 2: Π_enc + GKSL adaptativo + H_cond → MetabolizedPerturbationField
      FASE 3: vacuna por cobertura, η(Ω₄), Merkle, pasaporte

    Topología contrafactual determinista por SHA-256 del escenario.
    `energy_threshold` degrada η si E_D excede el umbral de Dirichlet.
    """

    _HEYTING_LEARNING_RATE_MAP: Final[Dict[int, float]] = {
        int(HeytingToposAlgebra.VERUM_COHERENT): 0.30,
        int(HeytingToposAlgebra.TOPOLOGICAL_STABLE): 0.15,
        int(HeytingToposAlgebra.BOUNDARY_CRITICAL): 0.05,
        int(HeytingToposAlgebra.VETOED_ABSURDUM): 0.0,
    }

    def __init__(
        self,
        agent_id: str = "TOON-DREAMER-SABIO-01",
        dimension_mac: int = 4,
        energy_threshold: float = 18.0,
        seed: int = 777,
    ) -> None:
        if dimension_mac < 2:
            raise ValueError("dimension_mac debe ser ≥ 2 (partición dream/physical).")
        if energy_threshold <= 0.0:
            raise ValueError("energy_threshold debe ser > 0.")
        self.agent_id = agent_id
        self.dimension_mac = int(dimension_mac)
        self.energy_threshold = float(energy_threshold)
        self.seed_counter = int(seed)
        self.dream_count = 0
        self.base_rho = np.eye(dimension_mac, dtype=np.complex128) / dimension_mac
        self.dream_certificates_history: List[OniricScenarioCertificate] = []

    @staticmethod
    def _deterministic_topology_perturbation(
        scenario_type: str, num_vertices: int
    ) -> Optional[Tuple[int, int]]:
        digest = hashlib.sha256(scenario_type.encode("utf-8")).digest()
        if digest[0] % 2 != 0:
            return None
        u, v = digest[1] % num_vertices, digest[2] % num_vertices
        return None if u == v else (u, v)

    def generate_synthetic_cartridge(
        self,
        scenario_type: str,
        cost_delta_ratio: float,
        synthetic_betti_1: int = 0,
        is_dream_state: bool = True,
    ) -> CategoricalCircuitCartridge:
        r"""FASE 1 DE ENLACE: sintetiza el cartucho como haz simplicial no recíproco."""
        self.dream_count += 1
        cartridge_id = f"SYNTH-TOON-DREAM-{self.dream_count:05d}"
        payload = {
            "synthetic_apu_code": "APU-SIM-9999",
            "material_variance_pct": cost_delta_ratio * 100.0,
            "simulated_labor_strike_prob": min(1.0, max(0.0, cost_delta_ratio * 0.6)),
            "simulated_supply_chain_latency_days": int(abs(cost_delta_ratio) * 30),
            "isolation_token": f"DREAM_STATE_HOMOLOGICAL_LOCK_{self.dream_count:04d}",
        }
        extra_edge = self._deterministic_topology_perturbation(scenario_type, 6)
        return CategoricalCircuitCartridge.synthesize_cartridge(
            cartridge_id=cartridge_id,
            scenario_type=scenario_type,
            cost_delta_ratio=cost_delta_ratio,
            synthetic_betti_1=synthetic_betti_1,
            is_dream_state=is_dream_state,
            payload=payload,
            deterministic_extra_edge=extra_edge,
        )

    def _adaptive_learning_rate(
        self, verdict: HeytingToposAlgebra, dirichlet_energy: float
    ) -> float:
        eta = self._HEYTING_LEARNING_RATE_MAP[int(verdict)]
        if dirichlet_energy > self.energy_threshold:
            eta *= 0.25
        return float(eta)

    def dream_scenario(
        self,
        scenario_type: str,
        cost_delta_ratio: float,
        synthetic_betti_1: int = 0,
        force_isolation_breach: bool = False,
    ) -> OniricScenarioCertificate:
        r"""Ciclo REM completo anidando Fases 1, 2 y 3: K → H → Π_enc → Φ_t → I → V."""
        start_time = time.time()
        self.seed_counter += 1
        is_dream_state = not force_isolation_breach

        # ── FASE 1: complejo + Tellegen + semilla H ──
        cartridge = self.generate_synthetic_cartridge(
            scenario_type=scenario_type,
            cost_delta_ratio=cost_delta_ratio,
            synthetic_betti_1=synthetic_betti_1,
            is_dream_state=is_dream_state,
        )

        # ── FASE 2: Π_enc + GKSL (continúa lift_enclave_hamiltonian) ──
        metabolized_field = MetabolizedPerturbationField.evolve_and_certify(
            cartridge=cartridge, base_rho=self.base_rho
        )

        # ── FASE 3: inmunización (continúa evolve_and_certify) ──
        P_vac, vaccine_effective, coverage_mass, idem_res = (
            SpectralImmuneVaccineEngine.construct_from_metabolized_field(
                metabolized_field
            )
        )
        learning_rate = self._adaptive_learning_rate(
            metabolized_field.topos_heyting_evaluation,
            metabolized_field.dirichlet_spectral_energy,
        )

        if learning_rate > 0.0 and vaccine_effective:
            vaccinated_rho = P_vac @ metabolized_field.rho_perturbed @ P_vac.conj().T
            trace_v = float(np.real(np.trace(vaccinated_rho)))
            if trace_v > 1e-12:
                vaccinated_rho = vaccinated_rho / trace_v
            self.base_rho = (
                (1.0 - learning_rate) * self.base_rho + learning_rate * vaccinated_rho
            )
            self.base_rho = BanachSpectralAlgebra.project_to_state_manifold(self.base_rho)

        quantum_fidelity = BanachSpectralAlgebra.quantum_fidelity(
            self.base_rho, metabolized_field.rho_perturbed
        )

        hasher = hashlib.sha512()
        signature_payload = (
            f"{self.agent_id}::{cartridge.cartridge_id}::{scenario_type}::"
            f"{metabolized_field.topos_heyting_evaluation.name}::"
            f"{metabolized_field.dirichlet_spectral_energy:.8f}::"
            f"{metabolized_field.purity:.8f}::{metabolized_field.is_strictly_isolated}::"
            f"{coverage_mass:.6f}::{start_time}"
        ).encode("utf-8")
        hasher.update(signature_payload)
        merkle_provenance = hasher.hexdigest()

        cert = OniricScenarioCertificate(
            dream_id=cartridge.cartridge_id,
            dreamer_agent_id=self.agent_id,
            cartridge_id=cartridge.cartridge_id,
            scenario_type=scenario_type,
            heyting_verdict=metabolized_field.topos_heyting_evaluation,
            dirichlet_energy=metabolized_field.dirichlet_spectral_energy,
            purity=metabolized_field.purity,
            entropy=metabolized_field.von_neumann_entropy,
            quantum_fidelity=quantum_fidelity,
            bures_distance=metabolized_field.bures_distance_to_base,
            cstar_residual=metabolized_field.cstar_residual,
            spectral_coverage_mass=coverage_mass,
            projector_idempotency_residual=idem_res,
            isolation_guarantee=metabolized_field.is_strictly_isolated,
            no_signaling_leakage=metabolized_field.no_signaling_leakage_norm,
            jump_rate=metabolized_field.jump_rate,
            mean_first_jump_time=metabolized_field.mean_first_jump_time,
            immune_vaccine_effective=vaccine_effective,
            learning_rate_applied=learning_rate,
            merkle_sha512_provenance=merkle_provenance,
            timestamp_utc=start_time,
        )
        self.dream_certificates_history.append(cert)
        logger.info(
            "Sueño REM #%04d [%s] ➔ Heyting: %s | E_D: %.4f | Cobertura: %.3f | "
            "η: %.2f | Vacuna OK: %s | Isol: %s | Γ: %.3e | P²−P: %.2e",
            self.dream_count,
            scenario_type,
            cert.heyting_verdict.name,
            cert.dirichlet_energy,
            coverage_mass,
            learning_rate,
            cert.immune_vaccine_effective,
            cert.isolation_guarantee,
            cert.jump_rate,
            idem_res,
        )
        return cert

    @staticmethod
    def _merkle_tree_root(leaf_hashes: Sequence[str]) -> str:
        if not leaf_hashes:
            return hashlib.sha512(b"EMPTY_MERKLE_ROOT").hexdigest()
        level = [bytes.fromhex(h) for h in leaf_hashes]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [
                hashlib.sha512(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
        return level[0].hex()

    @staticmethod
    def _merkle_proof(
        leaf_hashes: Sequence[str], index: int
    ) -> MerkleInclusionProof:
        if not leaf_hashes:
            empty = hashlib.sha512(b"EMPTY_MERKLE_ROOT").hexdigest()
            return MerkleInclusionProof(empty, tuple(), 0, empty)
        level = [bytes.fromhex(h) for h in leaf_hashes]
        siblings: List[str] = []
        idx = index
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            pair = idx ^ 1
            siblings.append(level[pair].hex())
            next_level = [
                hashlib.sha512(level[i] + level[i + 1]).digest()
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

    def execute_macro_wake_sleep_audit(
        self, batch_scenarios: List[Tuple[str, float, int]]
    ) -> MacroWakeSleepAuditReport:
        r"""Fase REM completa: metaboliza el lote, reduce energía libre, certifica Merkle."""
        t0 = time.time()
        initial_entropy = BanachSpectralAlgebra.von_neumann_entropy(self.base_rho)
        vaccines_absorbed = 0
        cumulative_topos = HeytingToposAlgebra.VERUM_COHERENT
        batch_leaf_hashes: List[str] = []

        for sc_type, cost_delta, betti_1 in batch_scenarios:
            cert = self.dream_scenario(
                scenario_type=sc_type,
                cost_delta_ratio=cost_delta,
                synthetic_betti_1=betti_1,
            )
            cumulative_topos = cumulative_topos.meet(cert.heyting_verdict)
            batch_leaf_hashes.append(cert.merkle_sha512_provenance)
            if cert.immune_vaccine_effective:
                vaccines_absorbed += 1

        final_entropy = BanachSpectralAlgebra.von_neumann_entropy(self.base_rho)
        entropy_ratio = float(
            (initial_entropy - final_entropy) / (initial_entropy + 1e-12)
        )
        duration = (time.time() - t0) * 1000.0

        merkle_root = self._merkle_tree_root(batch_leaf_hashes)
        proofs_ok = True
        for i in range(len(batch_leaf_hashes)):
            proof = self._merkle_proof(batch_leaf_hashes, i)
            if proof.root != merkle_root or not proof.verify():
                proofs_ok = False
                break

        context_binder = hashlib.sha512()
        context_binder.update(
            f"AUDIT-CONTEXT::{self.agent_id}::{merkle_root}::{initial_entropy:.8f}::"
            f"{final_entropy:.8f}::{vaccines_absorbed}::{t0}".encode("utf-8")
        )
        root_hash = context_binder.hexdigest()

        return MacroWakeSleepAuditReport(
            audit_cycle_id=f"AUDIT-REM-{self.dream_count:05d}",
            total_scenarios_simulated=len(batch_scenarios),
            vaccines_absorbed_count=vaccines_absorbed,
            wake_entropy=initial_entropy,
            sleep_entropy=final_entropy,
            entropy_reduction_ratio=entropy_ratio,
            overall_topos_verdict=cumulative_topos,
            merkle_leaf_count=len(batch_leaf_hashes),
            merkle_root_hash=root_hash,
            merkle_proofs_ok=proofs_ok,
            execution_duration_ms=duration,
        )

    @property
    def registry_view(self) -> Tuple[OniricScenarioCertificate, ...]:
        return tuple(self.dream_certificates_history)

    @property
    def global_verdict(self) -> HeytingToposAlgebra:
        gv = HeytingToposAlgebra.VERUM_COHERENT
        for c in self.dream_certificates_history:
            gv = gv.meet(c.heyting_verdict)
        return gv

    def audit_registry(self) -> Dict[str, Any]:
        n = len(self.dream_certificates_history)
        empty_dist = {v.name: 0 for v in HeytingToposAlgebra}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingToposAlgebra.VERUM_COHERENT.name,
                "avg_purity": 0.0,
                "avg_coverage_mass": 0.0,
                "avg_dirichlet_energy": 0.0,
                "n_immune": 0,
                "n_vaccines_effective": 0,
                "registry_integrity_ok": True,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingToposAlgebra}
        s_p = s_c = s_d = 0.0
        n_imm = n_vac = 0
        hashes: Set[str] = set()
        collide = False
        for c in self.dream_certificates_history:
            dist[c.heyting_verdict.name] += 1
            s_p += c.purity
            s_c += c.spectral_coverage_mass
            s_d += c.dirichlet_energy
            if c.is_immune():
                n_imm += 1
            if c.immune_vaccine_effective:
                n_vac += 1
            if c.merkle_sha512_provenance in hashes:
                collide = True
            hashes.add(c.merkle_sha512_provenance)
        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "avg_purity": s_p * inv,
            "avg_coverage_mass": s_c * inv,
            "avg_dirichlet_energy": s_d * inv,
            "n_immune": n_imm,
            "n_vaccines_effective": n_vac,
            "registry_integrity_ok": not collide,
        }

    def emit_dreamer_passport(self) -> Dict[str, Any]:
        h = hashlib.sha512()
        h.update(f"{self.agent_id}::{self.dream_count}".encode("utf-8"))
        for c in self.dream_certificates_history:
            h.update(c.merkle_sha512_provenance.encode("utf-8"))
        return {
            "agent_id": self.agent_id,
            "registry_size": self.dream_count,
            "global_verdict": self.global_verdict.name,
            "n_immune": sum(1 for c in self.dream_certificates_history if c.is_immune()),
            "evidence_hash": h.hexdigest(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN RIGUROSA Y AUDITORÍA UNITARIA END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("╔" + "═" * 78 + "╗")
    print("║     DEMOSTRACIÓN DOCTORAL DEL TOON ONIRIC DREAMER AGENT v4.0         ║")
    print("║  FASES ANIDADAS: Ω₄+LT+ℂ⊗ℍ+H → Enclave/GKSL/H_cond → I/Merkle/REM   ║")
    print("╚" + "═" * 78 + "╝")

    print("\n[§0] VERIFICACIÓN FORMAL DE Ω₄, LAWVERE-TIERNEY Y 𝔹")
    residuation_ok = HeytingToposAlgebra.verify_residuation_axiom()
    lt_axioms_ok = HeytingToposAlgebra.verify_lawvere_tierney_topology_axioms()
    print(f"  • Ley de residuación de Heyting        : {residuation_ok}")
    print(f"  • Axiomas Lawvere-Tierney (j1–j3, j⊤)  : {lt_axioms_ok}")
    assert residuation_ok
    assert lt_axioms_ok
    assert HeytingToposAlgebra.BOUNDARY_CRITICAL.excluded_middle_holds() is False
    assert HeytingToposAlgebra.VERUM_COHERENT.excluded_middle_holds() is True
    assert HeytingToposAlgebra.VETOED_ABSURDUM.is_regular() is True
    assert HeytingToposAlgebra.BOUNDARY_CRITICAL.is_j_closed() is False
    assert HeytingToposAlgebra.TOPOLOGICAL_STABLE.is_j_closed() is True

    q1 = BiquaternionClifford(1.0, 0.2, 0.3, -0.1, 0.0, 0.4, 0.5, 0.0)
    q2 = BiquaternionClifford(0.5, 0.0, -0.2, 0.1, 0.3, 0.0, 0.1, -0.05)
    assert q1.reduced_norm_multiplicativity_residual(q2) < 1e-10
    assert q1.cstar_residual() < 1e-10

    dreamer_agent = TOONOniricDreamerAgent(
        agent_id="TOON-DREAMER-SABIO-01",
        dimension_mac=4,
        energy_threshold=18.0,
        seed=10101,
    )

    print("\n[§1] FASE 1: HACES SIMPLICIALES EXACTOS, HODGE Y TELLEGEN")
    cartridge_test = dreamer_agent.generate_synthetic_cartridge(
        scenario_type="TEST_SYNTHETIC_QUARRY_COLLAPSE",
        cost_delta_ratio=0.35,
        synthetic_betti_1=1,
        is_dream_state=True,
    )
    print(f"  • ID Cartucho              : {cartridge_test.cartridge_id}")
    print(f"  • Betti (β0, β1, β2)       : ({cartridge_test.betti_0}, {cartridge_test.betti_1}, {cartridge_test.betti_2})")
    print(f"  • χ (consistente Euler)    : {cartridge_test.euler_characteristic} ({cartridge_test.euler_betti_consistent})")
    print(f"  • Hodge gap / Fiedler      : {cartridge_test.hodge_spectral_gap:.4f} / {cartridge_test.algebraic_connectivity:.4f}")
    print(f"  • Residuo Hodge ω−(ex+co+h): {cartridge_test.hodge_decomposition_residual:.3e}")
    print(f"  • Disipación / Tellegen    : {cartridge_test.circuit_dissipation_watts:.4f} W / {cartridge_test.tellegen_residual:.3e}")
    print(f"  • No-recip. / pasividad    : {cartridge_test.reciprocity_defect:.4f} / {cartridge_test.passivity_margin:.4f}")
    print(f"  • Residual C* de Clifford  : {cartridge_test.clifford_cstar_residual:.3e}")
    print(f"  • Coherencia / sieve j(χ)  : {cartridge_test.spectral_coherence_index:.4f} / {cartridge_test.topos_heyting_sieve.name}")
    H_seed = cartridge_test.lift_enclave_hamiltonian(4)
    assert np.allclose(H_seed, H_seed.conj().T), "H semilla no hermítico."

    print("\n[§2] FASE 2: GKSL ADAPTATIVO, ENCLAVE Y H_cond (continúa H)")
    metabolized_field = MetabolizedPerturbationField.evolve_and_certify(
        cartridge=cartridge_test, base_rho=dreamer_agent.base_rho
    )
    print(f"  • Pureza C*                : {metabolized_field.purity:.4f}")
    print(f"  • Entropía von Neumann     : {metabolized_field.von_neumann_entropy:.4f} bits")
    print(f"  • Umegaki S(ρ‖ρ₀)          : {metabolized_field.relative_entropy_to_base:.4f} bits")
    print(f"  • Bures / residual C*      : {metabolized_field.bures_distance_to_base:.4f} / {metabolized_field.cstar_residual:.3e}")
    print(f"  • Energía Dirichlet        : {metabolized_field.dirichlet_spectral_energy:.4f}")
    print(f"  • Fuga ‖[O,P_p]‖_∞         : {metabolized_field.no_signaling_leakage_norm:.2e}")
    print(f"  • Completitud {P_d,P_p}    : {metabolized_field.projector_completeness_ok}")
    print(f"  • Aislamiento estricto     : {metabolized_field.is_strictly_isolated}")
    print(f"  • Γ salto / ⟨τ⟩            : {metabolized_field.jump_rate:.4e} / {metabolized_field.mean_first_jump_time:.4f}")
    print(f"  • Coherencia M / Ω₄        : {metabolized_field.metabolic_coherence_index:.4f} / {metabolized_field.topos_heyting_evaluation.name}")
    assert BanachSpectralAlgebra.is_valid_density_matrix(metabolized_field.rho_perturbed)
    assert metabolized_field.projector_completeness_ok

    print("\n[§3] FASE 3: ESCENARIOS CONTRAFACTUALES REM")
    print("\n>>> ESCENARIO 1: Cisne Negro acero estructural (+35%)...")
    cert1 = dreamer_agent.dream_scenario(
        scenario_type="BLACK_SWAN_STEEL_SPIKE",
        cost_delta_ratio=0.35,
        synthetic_betti_1=0,
    )
    print(f"    - Veredicto Heyting  : {cert1.heyting_verdict.name}")
    print(f"    - Cobertura / η      : {cert1.spectral_coverage_mass:.3f} / {cert1.learning_rate_applied:.2f}")
    print(f"    - Uhlmann / Bures    : {cert1.quantum_fidelity:.4f} / {cert1.bures_distance:.4f}")
    print(f"    - Vacuna / P²−P      : {cert1.immune_vaccine_effective} / {cert1.projector_idempotency_residual:.2e}")
    print(f"    - SHA-512 Merkle     : {cert1.merkle_sha512_provenance[:32]}...")

    print("\n>>> ESCENARIO 2: Insolvencia circular β₁ = 3...")
    cert2 = dreamer_agent.dream_scenario(
        scenario_type="CIRCULAR_FRAUD_SUB_CONTRACT",
        cost_delta_ratio=0.85,
        synthetic_betti_1=3,
    )
    print(f"    - Veredicto Heyting  : {cert2.heyting_verdict.name}")
    print(f"    - Vacuna / η         : {cert2.immune_vaccine_effective} / {cert2.learning_rate_applied}")

    print("\n>>> ESCENARIO 3: Intrusión y fuga de enclave...")
    cert3 = dreamer_agent.dream_scenario(
        scenario_type="HARDWARE_ISOLATION_BREACH_ATTACK",
        cost_delta_ratio=0.10,
        synthetic_betti_1=0,
        force_isolation_breach=True,
    )
    print(f"    - Veredicto Heyting  : {cert3.heyting_verdict.name}")
    print(f"    - Aislamiento físico : {cert3.isolation_guarantee}")
    assert cert3.heyting_verdict == HeytingToposAlgebra.VETOED_ABSURDUM
    assert cert3.learning_rate_applied == 0.0
    assert cert3.isolation_guarantee is False

    print("\n[§4] FASE MACRO REM: AUDITORÍA, MERKLE Y VACUNACIÓN MASIVA")
    batch = [
        ("CEMENT_SUPPLY_BLOCKADE_40", 0.40, 1),
        ("LABOR_UNION_STRIKE_7DAYS", 0.25, 0),
        ("RIVER_FLOOD_FOUNDATION_APU", 0.30, 0),
        ("CIRCULAR_KICKBACK_SCHEME", 0.90, 2),
    ]
    report = dreamer_agent.execute_macro_wake_sleep_audit(batch)
    print(f"  • ID Auditoría REM       : {report.audit_cycle_id}")
    print(f"  • Vacunas consolidadas   : {report.vaccines_absorbed_count}/{report.total_scenarios_simulated}")
    print(f"  • Ratio reducción caos   : {report.entropy_reduction_ratio * 100:.2f}%")
    print(f"  • Veredicto topos global : {report.overall_topos_verdict.name}")
    print(f"  • Merkle hojas / pruebas : {report.merkle_leaf_count} / {report.merkle_proofs_ok}")
    print(f"  • Merkle root (SHA-512)  : {report.merkle_root_hash[:48]}...")
    print(f"  • Tiempo de cómputo      : {report.execution_duration_ms:.2f} ms")
    assert report.merkle_proofs_ok, "Fallo en pruebas de inclusión Merkle."

    print("\n>>> AUDITORÍA RETROSPECTIVA")
    for k, v in dreamer_agent.audit_registry().items():
        print(f"    - {k:<28}: {v}")
    print("\n>>> PASAPORTE DEL DREAMER")
    for k, v in dreamer_agent.emit_dreamer_passport().items():
        print(f"    - {k:<22}: {v}")

    print("\n" + "═" * 80)
    print("✓ AUDITORÍA Y METABOLIZACIÓN ONÍRICA CONCLUIDA SATISFACTORIAMENTE.")
    print("═" * 80)
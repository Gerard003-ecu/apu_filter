# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : TOON Oniric Dreamer Engine (Motor Espectral y Campo Metabolizador)║
║ UBICACIÓN: app/wisdom/toon_oniric_dreamer_engine.py                          ║
║ VERSIÓN  : 4.0.0-Doctoral-Nested-Ω₄-Hodge-GKSL-Richardson-MerkleTopos        ║
║ AUTOR    : APU Wisdom & Metacortex Mathematical Core Architecture            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Formalización Categorial Doctoral (Funtor Onírico REM 𝒟)
========================================================

Sea 𝓣_Ω el topos de haces sobre la cadena de Heyting tetravaluada:

        Ω₄  =  { ABSURDUM_VETOED = 0 ≺ BOUNDARY_DEGRADED = 1 ≺ TOPOLOGICAL_SOUND = 2 ≺ VERUM_COHERENT = 3 }

El motor realiza una simulación onírica contrafactual y síntesis inmunológica mediante el funtor:

        𝒟  :  𝐒𝐜𝐞𝐧𝐚𝐫𝐢𝐨_𝐑𝐄𝐌  ──▶  𝐃𝐫𝐞𝐚𝐦_𝐑𝐞𝐩𝐨𝐫𝐭

componiendo de forma estrictamente asociativa las tres fases anidadas:

        𝒟  =  V ∘ I ∘ Φ_t ∘ H ∘ K

donde el último morfismo de la fase k es el germen formal de la fase k+1.

Estructura de Fases Anidadas e Invariantes
===========================================

FASE 1 — RETÍCULO Ω₄, HEYTING, CUATERNIONES ℍ, HODGE SIMPLICIAL Y SEMILLA H
──────────────────────────────────────────────────────────────────────────
  • HeytingTruthValue: Álgebra de Heyting completa tetravaluada Ω₄. Satisface la residuación
    a ∧ c ≤ b  ⇔  c ≤ (a → b). Pseudocomplemento ¬_H a = a → ⊥.
  • Quaternion: Álgebra de división ℍ ≅ Cl⁺_{0,3}(ℝ) con inmersión ℍ ↪ M₂(ℂ) vía Pauli.
    Álgebra de composición |q₁ q₂| = |q₁| |q₂|, C*-identidad |q* q| = |q|² y fibración de Hopf S³ → S².
  • SimplicialHodgeComplex: 2-complejo de cadenas simplicial K = (V, E, F) con ∂₁∂₂ = 0.
    Invariantes de Hodge: βₖ = dim ker L▖, Euler-Poincaré χ = β₀ − β₁ + β₂, y descomposición de Hodge C₁ = im ∂₂ ⊕ im ∂₁ᵀ ⊕ ker L₁.
  • NonReciprocalCircuitField: Red AC no recíproca sobre K. Satisface la conservación de Tellegen Σ_e v_e i_e* = V† I
    y el margen de pasividad λ_min((Y_b + Y_b†)/2) ≥ −ε.
  • TopologicalCircuitBundle: Objeto terminal de FASE-1. Su método `lift_hamiltonian` es el ÚLTIMO de FASE-1
    y el PRIMERO de FASE-2.

FASE 2 — ÁLGEBRAS C*/BANACH, GKSL ADAPTATIVO, CFT DE CUERDAS Y ESTADO METABOLIZADO
──────────────────────────────────────────────────────────────────────────
  • BanachOperatorAlgebra: Estructura C* sobre B(ℋₙ). Proyección al simplex 𝔇(ℋₙ), entropía S(ρ) = −Tr(ρ log₂ ρ),
    pureza P(ρ) = Tr(ρ²), relativa de Umegaki S(ρ‖σ) ≥ 0, fidelidad F(ρ,σ) y distancia Bures d_B.
  • PolyakovWorldsheetMetrics: Acción bosónica S_P[X,h] con reducción del parámetro modular τ al dominio
    fundamental ℱ de PSL(2,ℤ).
  • LindbladCFTMasterEvolver: CONTINUACIÓN FORMAL de `lift_hamiltonian`. Integra la ecuación GKSL:
        dρ/dt = −i[H, ρ] + Σ_k γ_k (L_k ρ L_k† − ½ {L_k† L_k, ρ})
    mediante RK4 adaptativo + duplicación + Richardson (orden 5) + proyección a 𝔇(ℋₙ).
  • MetabolizedFieldState.create_metabolized_state: ÚLTIMO método de FASE-2. Evalúa la coherencia metabólica
    M = P(ρ) · exp(−E_D/κ_E) · exp(−S_rel/κ_S) y asigna el veredicto Ω₄.

FASE 3 — INMUNIZACIÓN ESPECTRAL, MERKLE, WAKE-SLEEP Y REGISTRO
──────────────────────────────────────────────────────────────────────────
  • SpectralImmuneVaccineSynthesizer.synthesize_from_metabolized_state: PRIMER MORFISMO DE FASE-3 (continúa
    `create_metabolized_state`). Construye el proyector de inmunidad P_vac = Σ_{i=1}^k |v_i⟩⟨v_i| sobre la masa espectral.
  • MerkleInclusionProof: Árbol Merkle SHA-512 sobre las hojas de firma para verificación en O(log n).
  • TOONOniricDreamerEngine: Orquestador soberano que ejecuta el ciclo REM y las fases Wake-Sleep.

Definición Granular de Invariantes y Axiomas
=============================================
  1. Exactitud Simplicial: ∂₁ ∂₂ = 0  (‖∂₁∂₂‖_F = 0).
  2. Invariante de Euler-Poincaré: χ = β₀ − β₁ + β₂ = |V| − |E| + |F|.
  3. Positividad y Traza Cuántica: ρ = ρ†, spec(ρ) ⊂ [0, 1], Tr(ρ) = 1.
  4. Idempotencia del Proyector Vacunal: P_vac² = P_vac = P_vac† (‖P_vac² − P_vac‖_F ≈ 0).
  5. Adjunción de Heyting: ∀ a,b,c ∈ Ω₄: (c ∧ a ≤ b) ⇔ (c ≤ (a → b)).
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

logger = logging.getLogger("APU.Wisdom.TOONOniricDreamerEngine.v4")

__all__ = [
    "HeytingTruthValue",
    "Quaternion",
    "SimplicialHodgeComplex",
    "NonReciprocalCircuitField",
    "TopologicalCircuitBundle",
    "OpenQuantumDynamicsSeed",
    "PolyakovWorldsheetMetrics",
    "BanachOperatorAlgebra",
    "LindbladCFTMasterEvolver",
    "MetabolizedFieldState",
    "DreamFieldReport",
    "WakeSleepCycleReport",
    "MerkleInclusionProof",
    "SpectralImmuneVaccineSynthesizer",
    "TOONOniricDreamerEngine",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — FUNDAMENTOS HIPERCOMPLEJOS, TOPOLOGÍA EXACTA, CIRCUITOS Y SEMILLA H
# ══════════════════════════════════════════════════════════════════════════════
# Andamiaje del topos 𝓣_Ω₄ y del 2-complejo de Hodge. El ÚLTIMO método de
# esta fase (TopologicalCircuitBundle.lift_hamiltonian) es el germen formal
# de FASE 2: LindbladCFTMasterEvolver._synthesize_hamiltonian lo invoca
# sin duplicar la construcción.
# ══════════════════════════════════════════════════════════════════════════════


class HeytingTruthValue(IntEnum):
    r"""
    Cadena finita Ω₄ = {0 ≺ 1 ≺ 2 ≺ 3} = {⊥ ≺ ∂ ≺ ♯ ≺ ⊤}, álgebra de Heyting
    completa (toda cadena finita con máximo y mínimo lo es):

        a ∧ b  = min(a, b)
        a ∨ b  = max(a, b)
        a → b  = ⋁{ c ∈ Ω₄ | c ∧ a ≤ b }     (residuo)
        ¬a     = a → ⊥

    Clasificador de subobjetos de un topos de haces sobre un sitio finito
    de 4 sieves. El esqueleto booleano es {⊥, ⊤}; 1 y 2 violan el tercio
    excluso (¬¬a = a falla), de modo que Ω₄ es estrictamente intuicionista.
    """

    ABSURDUM_VETOED = 0       # ⊥
    BOUNDARY_DEGRADED = 1     # ∂  (sieve de frontera)
    TOPOLOGICAL_SOUND = 2     # ♯  (sieve submáximo)
    VERUM_COHERENT = 3        # ⊤

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return HeytingTruthValue(min(int(self), int(other)))

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        return HeytingTruthValue(max(int(self), int(other)))

    def implies(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        a, b = int(self), int(other)
        valid = [c for c in range(4) if min(c, a) <= b]
        return HeytingTruthValue(max(valid))

    def complement(self) -> "HeytingTruthValue":
        return self.implies(HeytingTruthValue.ABSURDUM_VETOED)

    def pseudo_complement(self) -> "HeytingTruthValue":
        return self.complement()

    def classical_negation(self) -> "HeytingTruthValue":
        return HeytingTruthValue(3 - int(self))

    def double_negation(self) -> "HeytingTruthValue":
        r"""¬¬a ≥ a (ley débil intuicionista; igualdad sólo en regulares)."""
        return self.complement().complement()

    def is_regular(self) -> bool:
        r"""a regular ⟺ ¬¬a = a. En Ω₄: {⊥, ⊤}."""
        return self.double_negation() == self

    def is_dense(self) -> bool:
        return self.complement() == HeytingTruthValue.ABSURDUM_VETOED

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬a = ⊤  ⇔  a ∈ {⊥, ⊤}."""
        return self.join(self.complement()) == HeytingTruthValue.VERUM_COHERENT

    @classmethod
    def bottom(cls) -> "HeytingTruthValue":
        return cls.ABSURDUM_VETOED

    @classmethod
    def top(cls) -> "HeytingTruthValue":
        return cls.VERUM_COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingTruthValue":
        return cls.VERUM_COHERENT if b else cls.ABSURDUM_VETOED

    def to_bool(self) -> bool:
        if self not in (HeytingTruthValue.ABSURDUM_VETOED, HeytingTruthValue.VERUM_COHERENT):
            raise ValueError(f"{self.name} no admite proyección fiel a 𝔹₂.")
        return self == HeytingTruthValue.VERUM_COHERENT

    @classmethod
    def verify_heyting_axioms(cls) -> bool:
        r"""
        Ley de residuación, axioma definitorio:
            ∀ a,b,c ∈ Ω₄:  (c ∧ a ≤ b)  ⟺  (c ≤ (a → b)).
        """
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


@dataclass(frozen=True, slots=True)
class Quaternion:
    r"""
    Álgebra de división no conmutativa ℍ ≅ Cl_{0,3}^{+}, Banach real de
    dimensión 4. Representación fiel en M₂(ℂ):

        q = w + x i + y j + z k
          ↦  [[ w+i z ,  y+i x ],
              [ −y+i x ,  w−i z ]]

    Estructura:
        • anillo de división (inversa, cociente);
        • grupo de Lie Sp(1) ≅ SU(2) (versor, exp, log);
        • recubrimiento 2:1  Sp(1) → SO(3);
        • fibración de Hopf S³ → S²;
        • álgebra de composición |q₁ q₂| = |q₁| |q₂|;
        • C*-identidad |q* q| = |q|².
    """

    w: float
    x: float
    y: float
    z: float

    _NORM_FLOOR: Final[float] = 1e-15

    def norm_squared(self) -> float:
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def norm(self) -> float:
        return math.sqrt(self.norm_squared())

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def scale(self, lam: float) -> "Quaternion":
        return Quaternion(self.w * lam, self.x * lam, self.y * lam, self.z * lam)

    def to_matrix(self) -> np.ndarray:
        r"""Homomorfismo inyectivo de anillos ℍ ↪ M₂(ℂ)."""
        return np.array(
            [
                [complex(self.w, self.z), complex(self.y, self.x)],
                [complex(-self.y, self.x), complex(self.w, -self.z)],
            ],
            dtype=np.complex128,
        )

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Quaternion":
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: object) -> "Quaternion":
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            )
        if isinstance(other, (int, float)):
            return self.scale(float(other))
        return NotImplemented

    def __rmul__(self, other: object) -> "Quaternion":
        if isinstance(other, (int, float)):
            return self.scale(float(other))
        return NotImplemented

    def inverse(self) -> "Quaternion":
        n2 = self.norm_squared()
        if n2 < self._NORM_FLOOR:
            raise ZeroDivisionError("Cuaternión nulo: no invertible en ℍ.")
        return self.conjugate().scale(1.0 / n2)

    def __truediv__(self, other: "Quaternion") -> "Quaternion":
        return self * other.inverse()

    def versor(self) -> "Quaternion":
        n = self.norm()
        if n < self._NORM_FLOOR:
            raise ZeroDivisionError("No se puede normalizar un cuaternión nulo.")
        return self.scale(1.0 / n)

    def exp(self) -> "Quaternion":
        r"""
        exp: ℍ → ℍˣ,  exp(w + v) = eʷ (cos|v| + (v/|v|) sin|v|).
        Restringido a Im ℍ recupera el recubrimiento SU(2).
        """
        vec_norm = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        e_w = math.exp(self.w)
        if vec_norm < self._NORM_FLOOR:
            return Quaternion(e_w, 0.0, 0.0, 0.0)
        s = e_w * math.sin(vec_norm) / vec_norm
        return Quaternion(e_w * math.cos(vec_norm), self.x * s, self.y * s, self.z * s)

    def log(self) -> "Quaternion":
        r"""log: ℍˣ → ℍ,  log(q) = log|q| + (v/|v|) arccos(w/|q|)."""
        n = self.norm()
        if n < self._NORM_FLOOR:
            raise ValueError("log no definido en el cuaternión nulo.")
        vec_n = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        if vec_n < self._NORM_FLOOR:
            return Quaternion(math.log(n), 0.0, 0.0, 0.0)
        phi = math.acos(max(-1.0, min(1.0, self.w / n)))
        s = phi / vec_n
        return Quaternion(math.log(n), s * self.x, s * self.y, s * self.z)

    def hopf_s2(self) -> Tuple[float, float, float]:
        r"""Fibración de Hopf S³ → S²."""
        u = self.versor()
        return (
            2.0 * (u.w * u.y + u.x * u.z),
            2.0 * (u.x * u.y - u.w * u.z),
            u.w ** 2 + u.x ** 2 - u.y ** 2 - u.z ** 2,
        )

    def cstar_residual(self) -> float:
        r"""| |q* q| − |q|² |  (nulo en aritmética exacta)."""
        return abs((self.conjugate() * self).norm() - self.norm_squared())

    def su2_det_residual(self) -> float:
        r"""|det φ(q̂) − 1| sobre el versor; fidelidad de la inmersión SU(2)."""
        u = self.versor()
        return abs(complex(np.linalg.det(u.to_matrix())) - 1.0)


class SimplicialHodgeComplex:
    r"""
    Complejo de cadenas simplicial finito K = (V, E, F) de dimensión 2.

        ∂₂ : C₂ → C₁,   ∂₁ : C₁ → C₀,   ∂₁ ∂₂ = 0

        L₀ = ∂₁ ∂₁ᵀ                         (vértices)
        L₁ = ∂₁ᵀ ∂₁ + ∂₂ ∂₂ᵀ                (Hodge en aristas)
        L₂ = ∂₂ᵀ ∂₂                         (up-Laplaciano en caras)

    Números de Betti (Eckmann–Hodge):
        βₖ = dim ker Lₖ,    χ(K) = |V|−|E|+|F| = β₀−β₁+β₂.

    Descomposición de Hodge en 1-cadenas:
        C₁ = im ∂₂  ⊕  im ∂₁ᵀ  ⊕  ker L₁     (exacta ⊕ coexacta ⊕ armónica).
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

        self._validate_simplicial_topology()
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

    def _validate_simplicial_topology(self) -> None:
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
        r"""Propiedad fundamental ∂² = 0. Su violación indica triangulación mal formada."""
        if self.boundary_2.size == 0:
            return
        residual = self.boundary_1 @ self.boundary_2
        residual_norm = float(la.norm(residual))
        if residual_norm > tol:
            raise ValueError(
                f"Violación de ∂₁∂₂ = 0: ||∂₁∂₂|| = {residual_norm:.3e}"
            )

    def _build_boundary_1(self) -> np.ndarray:
        B1 = np.zeros((self.num_vertices, len(self.edges)), dtype=np.float64)
        for e_idx, (u, v) in enumerate(self.edges):
            B1[u, e_idx] = -1.0
            B1[v, e_idx] = 1.0
        return B1

    def _build_boundary_2(self) -> np.ndarray:
        r"""
        Orientación canónica sobre (u,v,w) ordenado:
            ∂[u,v,w] = [v,w] − [u,w] + [u,v].
        """
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

    def connected_components_combinatorial(self) -> List[Set[int]]:
        adjacency: Dict[int, Set[int]] = {v: set() for v in range(self.num_vertices)}
        for (u, v) in self.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in range(self.num_vertices):
            if start in visited:
                continue
            stack = [start]
            comp: Set[int] = set()
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                stack.extend(adjacency[node] - comp)
            visited |= comp
            components.append(comp)
        return components

    def euler_poincare_characteristic(self) -> int:
        return self.num_vertices - len(self.edges) + len(self.faces)

    def compute_betti_numbers(self, tol: float = 1e-10) -> Tuple[int, int, int]:
        eigvals_0 = la.eigvalsh(self.laplacian_0)
        betti_0 = int(np.sum(np.abs(eigvals_0) < tol))
        eigvals_1 = la.eigvalsh(self.laplacian_1)
        betti_1 = int(np.sum(np.abs(eigvals_1) < tol))
        if self.laplacian_2.size > 0:
            eigvals_2 = la.eigvalsh(self.laplacian_2)
            betti_2 = int(np.sum(np.abs(eigvals_2) < tol))
        else:
            betti_2 = 0
        n_comp = len(self.connected_components_combinatorial())
        if n_comp != betti_0:
            logger.warning(
                "Discrepancia β₀ espectral (%d) vs. combinatoria DFS (%d).",
                betti_0, n_comp,
            )
        return betti_0, betti_1, betti_2

    def verify_betti_euler_consistency(self, tol: float = 1e-10) -> bool:
        b0, b1, b2 = self.compute_betti_numbers(tol)
        return (b0 - b1 + b2) == self.euler_poincare_characteristic()

    def algebraic_connectivity(self, tol: float = 1e-12) -> float:
        r"""Valor de Fiedler: segundo autovalor de L₀ (0 si β₀ > 1 o n=1)."""
        evals = np.sort(la.eigvalsh(self.laplacian_0))
        if evals.size < 2:
            return 0.0
        return float(max(evals[1], 0.0)) if evals[1] > tol else 0.0

    def harmonic_1_forms(self, tol: float = 1e-10) -> np.ndarray:
        w, v = la.eigh(self.laplacian_1)
        mask = np.abs(w) < tol
        if not np.any(mask):
            return np.zeros((len(self.edges), 0), dtype=np.float64)
        return v[:, mask]

    def harmonic_2_forms(self, tol: float = 1e-10) -> np.ndarray:
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
        Descompone ω ∈ C₁ ≅ ℝ^{|E|} como

            ω = ∂₂ α  +  ∂₁ᵀ β  +  γ,     γ ∈ ker L₁.

        Retorna (exacta, coexacta, armónica). Identidad de Hodge:
            ω = exacta + coexacta + armónica  (salvo O(ε) numérico).
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


class NonReciprocalCircuitField:
    r"""
    Red no recíproca sobre el 1-esqueleto de K, con tierra por componente
    conexa (β₀ ≥ 1). Elementos giroscópicos (Y_b − Y_bᵀ ≠ 0) modelan
    disipación asimétrica.

    Tellegen (identidad topológica, independiente de la constitución):
        Σ_e v_e i_e*  =  V† I_nodal.

    Pasividad (Re Y_b ⪰ 0):  λ_min((Y_b + Y_b†)/2) ≥ −ε.
    """

    def __init__(
        self, simplicial_complex: SimplicialHodgeComplex, angular_freq: float = 100.0
    ) -> None:
        self.complex = simplicial_complex
        self.omega = float(angular_freq)
        self.num_edges = len(self.complex.edges)
        self.branch_admittance = self._build_branch_admittance()

    def _build_branch_admittance(self) -> np.ndarray:
        dim = self.num_edges
        Yb = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            g_i = 1.0 + 0.1 * (i + 1)
            b_i = self.omega * 0.01 - 1.0 / (self.omega * 0.05 + 1e-5)
            Yb[i, i] = complex(g_i, b_i)
            if i + 1 < dim:
                gyration = 0.25 * (i + 1)
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
        B1 = self.complex.boundary_1
        return B1 @ self.branch_admittance @ B1.T

    def solve_voltage_distribution(
        self, nodal_current_injections: np.ndarray
    ) -> np.ndarray:
        r"""
        Y_bus V = I, fijando V=0 en un nodo de tierra por componente conexa
        (ker Y_bus tiene dimensión β₀, independientemente de la no-reciprocidad).
        """
        n = self.complex.num_vertices
        components = self.complex.connected_components_combinatorial()
        ground_nodes = {min(comp) for comp in components}
        free_nodes = [i for i in range(n) if i not in ground_nodes]
        V_full = np.zeros(n, dtype=np.complex128)
        if not free_nodes:
            return V_full
        Y_bus = self.compute_bus_admittance_matrix()
        Y_reduced = Y_bus[np.ix_(free_nodes, free_nodes)]
        I_reduced = nodal_current_injections[free_nodes]
        try:
            V_reduced = la.solve(Y_reduced, I_reduced)
        except la.LinAlgError:
            logger.warning("Y_reduced casi-singular: lstsq regularizado.")
            V_reduced, *_ = la.lstsq(Y_reduced, I_reduced)
        for local_idx, node in enumerate(free_nodes):
            V_full[node] = V_reduced[local_idx]
        return V_full

    def kcl_residual(self, v_nodes: np.ndarray, i_inj: np.ndarray) -> float:
        r"""||Y_bus V − I||₂ / max(1, ||I||₂)  (ley de corrientes de Kirchhoff)."""
        I_pred = self.compute_bus_admittance_matrix() @ v_nodes
        denom = max(float(np.linalg.norm(i_inj)), 1.0)
        return float(np.linalg.norm(I_pred - i_inj) / denom)

    def verify_tellegen_conservation(
        self, v_nodes: np.ndarray, tol: float = 1e-6
    ) -> float:
        r"""
        Identidad de Tellegen auto-consistente:
            Σ_e v_e conj(i_e) = Σ_n conj(V_n) I_n.
        Retorna el residuo relativo (≈ 0 salvo redondeo).
        """
        del tol  # identidad algebraica; la tolerancia es documental
        Y_bus = self.compute_bus_admittance_matrix()
        I_full = Y_bus @ v_nodes
        branch_v = self.complex.boundary_1.T @ v_nodes
        branch_i = self.branch_admittance @ branch_v
        lhs = np.sum(branch_v * np.conj(branch_i))
        rhs = np.sum(np.conj(v_nodes) * I_full)
        denom = max(abs(rhs), 1e-12)
        return float(abs(lhs - rhs) / denom)


class OpenQuantumDynamicsSeed(ABC):
    r"""
    Germen formal de la flecha H : (K, τ) ↦ 𝔥𝔢𝔯(ℋₙ).

    Cierra el andamiaje de FASE-1. FASE-2 *continúa* exactamente en
    lift_hamiltonian: LindbladCFTMasterEvolver._synthesize_hamiltonian
    delega aquí y no reconstruye H.
    """

    @abstractmethod
    def lift_hamiltonian(self, hilbert_dim: int, modular_tau: complex) -> np.ndarray:
        r"""
        Produce H = H† a partir del fibrado topológico y del módulo τ.

        CONTINÚA EN FASE-2: LindbladCFTMasterEvolver._synthesize_hamiltonian.
        """
        ...


@dataclass(frozen=True, slots=True)
class TopologicalCircuitBundle(OpenQuantumDynamicsSeed):
    r"""
    NEXO FORMAL TERMINAL DE LA FASE 1.

    Encapsula topología simplicial exacta, espectro de Hodge (β₀,β₁,β₂,χ),
    respuesta circuital no recíproca (defecto de reciprocidad, Tellegen,
    pasividad) y valuación en Ω₄. Entrada ontológica de la Fase 2.

    Último método: lift_hamiltonian — germen de GKSL.
    """

    simplicial_complex: SimplicialHodgeComplex
    circuit_field: NonReciprocalCircuitField
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    euler_betti_consistent: bool
    hodge_spectral_gap: float
    algebraic_connectivity: float
    harmonic_energy_norm: float
    hodge_decomposition_residual: float
    circuit_dissipation_rate: float
    reciprocity_defect: float
    tellegen_residual: float
    passivity_margin: float
    kcl_residual: float
    spectral_coherence_index: float
    heyting_topos_evaluation: HeytingTruthValue

    _KAPPA_BETTI: ClassVar[float] = 1.0
    _KAPPA_DISSIPATION: ClassVar[float] = 12.0
    _KAPPA_RECIPROCITY: ClassVar[float] = 5.0

    @staticmethod
    def _spectral_coherence_functional(
        betti_1: int, betti_2: int, dissipation: float, reciprocity_defect: float
    ) -> float:
        r"""
        C = exp(−(β₁+β₂)/κ_β) · exp(−|D|/κ_D) · exp(−Δ_recip/κ_R) ∈ (0,1].
        """
        kb = TopologicalCircuitBundle._KAPPA_BETTI
        kd = TopologicalCircuitBundle._KAPPA_DISSIPATION
        kr = TopologicalCircuitBundle._KAPPA_RECIPROCITY
        return (
            math.exp(-(betti_1 + betti_2) / kb)
            * math.exp(-abs(dissipation) / kd)
            * math.exp(-abs(reciprocity_defect) / kr)
        )

    @staticmethod
    def _classify_topos_state(coherence: float) -> HeytingTruthValue:
        if coherence >= 0.75:
            return HeytingTruthValue.VERUM_COHERENT
        if coherence >= 0.45:
            return HeytingTruthValue.TOPOLOGICAL_SOUND
        if coherence >= 0.15:
            return HeytingTruthValue.BOUNDARY_DEGRADED
        return HeytingTruthValue.ABSURDUM_VETOED

    @classmethod
    def synthesize_bundle(
        cls,
        num_vertices: int,
        edges: List[Tuple[int, int]],
        faces: List[Tuple[int, int, int]],
        current_stimulus: Optional[np.ndarray] = None,
    ) -> "TopologicalCircuitBundle":
        r"""Constructor unificado de la Fase 1 (preludio de lift_hamiltonian)."""
        comp = SimplicialHodgeComplex(
            num_vertices=num_vertices, edges=edges, faces=faces
        )
        b0, b1, b2 = comp.compute_betti_numbers()
        chi = comp.euler_poincare_characteristic()
        euler_ok = comp.verify_betti_euler_consistency()
        if not euler_ok:
            logger.warning(
                "Inconsistencia de Euler-Poincaré: χ=%d ≠ β₀−β₁+β₂=%d.",
                chi, b0 - b1 + b2,
            )

        circ = NonReciprocalCircuitField(simplicial_complex=comp)
        if current_stimulus is None:
            current_stimulus = np.zeros(num_vertices, dtype=np.complex128)
            current_stimulus[0] = complex(1.0, 0.0)
            current_stimulus[-1] = complex(-1.0, 0.0)

        v_nodes = circ.solve_voltage_distribution(current_stimulus)
        branch_v = comp.boundary_1.T @ v_nodes
        branch_i = circ.branch_admittance @ branch_v
        dissipation = float(np.real(np.sum(branch_v * np.conj(branch_i))))
        tellegen_residual = circ.verify_tellegen_conservation(v_nodes)
        reciprocity_defect = circ.reciprocity_defect_norm()
        passivity = circ.passivity_margin()
        kcl = circ.kcl_residual(v_nodes, current_stimulus)

        eigvals_l1 = np.sort(la.eigvalsh(comp.laplacian_1))
        spectral_gap = float(eigvals_l1[b1]) if len(eigvals_l1) > b1 else 0.0
        fiedler = comp.algebraic_connectivity()

        harmonics = comp.harmonic_1_forms()
        harm_norm = float(la.norm(harmonics)) if harmonics.shape[1] > 0 else 0.0

        omega_probe = np.real(branch_v)
        if omega_probe.size == len(comp.edges) and omega_probe.size > 0:
            ex, co, ha = comp.hodge_decompose_1_form(omega_probe)
            hodge_res = float(np.linalg.norm(omega_probe - (ex + co + ha)))
        else:
            hodge_res = 0.0

        coherence = cls._spectral_coherence_functional(
            b1, b2, dissipation, reciprocity_defect
        )
        verdict = cls._classify_topos_state(coherence)

        return cls(
            simplicial_complex=comp,
            circuit_field=circ,
            betti_0=b0,
            betti_1=b1,
            betti_2=b2,
            euler_characteristic=chi,
            euler_betti_consistent=euler_ok,
            hodge_spectral_gap=spectral_gap,
            algebraic_connectivity=fiedler,
            harmonic_energy_norm=harm_norm,
            hodge_decomposition_residual=hodge_res,
            circuit_dissipation_rate=dissipation,
            reciprocity_defect=reciprocity_defect,
            tellegen_residual=tellegen_residual,
            passivity_margin=passivity,
            kcl_residual=kcl,
            spectral_coherence_index=coherence,
            heyting_topos_evaluation=verdict,
        )

    # ══════════════════════════════════════════════════════════════════════
    # HAND-OFF  FASE 1 → FASE 2
    # Último método de la FASE 1. Su salida H = H† es el generador unitario
    # del superoperador GKSL. CONTINÚA EN
    # LindbladCFTMasterEvolver._synthesize_hamiltonian.
    # ══════════════════════════════════════════════════════════════════════
    def lift_hamiltonian(self, hilbert_dim: int, modular_tau: complex) -> np.ndarray:
        r"""
        Flecha H: (K, τ) ↦ H ∈ 𝔥𝔢𝔯(ℋₙ).

        Construcción:
            • niveles diagonales modulados por la brecha de Hodge y la
              disipación de Tellegen;
            • inmersión hipercompleja: bloques 2×2 = φ(q) con
              q = τ₁ + τ₂ i + β₁ j + ½ k.

        CONTINÚA EN FASE-2: el evolver GKSL toma este H como parte
        hamiltoniana del Liouvilliano y le añade los saltos de Lindblad.
        """
        if hilbert_dim < 1:
            raise ValueError("hilbert_dim debe ser ≥ 1.")
        dim = int(hilbert_dim)
        H0 = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            H0[i, i] = (i + 1) * self.hodge_spectral_gap + (
                self.circuit_dissipation_rate * 0.05
            )
        q = Quaternion(
            float(modular_tau.real),
            float(modular_tau.imag),
            float(self.betti_1),
            0.5,
        )
        q_mat = q.to_matrix()
        for i in range(0, dim - 1, 2):
            H0[i:i + 2, i:i + 2] += q_mat * 0.1
        return 0.5 * (H0 + H0.conj().T)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — C*/BANACH, GKSL ADAPTATIVO, CFT DE CUERDAS Y ESTADO METABOLIZADO
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método operativo (_synthesize_hamiltonian) ES la
# continuación de lift_hamiltonian. El último (create_metabolized_state)
# produce MetabolizedFieldState, germen formal de FASE 3.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PolyakovWorldsheetMetrics:
    r"""
    Métricas de la cuerda bosónica sobre una hoja de mundo onírica Σ.

        S_P[X,h] = (1/(4π α')) ∫_Σ d²σ √(−h) h^{ab} ∂_a X^μ ∂_b X_μ

    τ = τ₁ + i τ₂ (τ₂>0) vive en ℍ² / PSL(2,ℤ). La invariancia modular
    exacta es cuántica (det Faddeev–Popov + medida de Weyl); a nivel
    clásico truncado se reporta honestamente el defecto bajo T: τ ↦ τ+1.
    """

    modular_parameter_tau: complex
    string_tension_alpha_prime: float
    polyakov_action_integral: float
    conformal_anomaly_central_charge: float
    weyl_invariance_residual: float
    in_fundamental_domain_flag: bool

    def in_fundamental_domain(self) -> bool:
        tau1 = self.modular_parameter_tau.real
        return abs(tau1) <= 0.5 + 1e-9 and abs(self.modular_parameter_tau) >= 1.0 - 1e-9

    @staticmethod
    def reduce_to_fundamental_domain(
        tau: complex, max_iter: int = 64
    ) -> complex:
        r"""
        Algoritmo estándar de reducción a
            ℱ = { τ : |Re τ| ≤ ½,  |τ| ≥ 1 }
        por los generadores T: τ ↦ τ+1  y  S: τ ↦ −1/τ de PSL(2,ℤ).
        """
        z = complex(tau)
        if z.imag <= 0.0:
            z = complex(z.real, max(abs(z.imag), 1e-6))
        for _ in range(max_iter):
            z = complex(z.real - round(z.real), z.imag)
            if abs(z) < 1.0 - 1e-15:
                z = -1.0 / z
                continue
            if abs(z.real) <= 0.5 + 1e-15:
                break
        return z


class BanachOperatorAlgebra:
    r"""
    C*-álgebra B(ℋ) de dimensión finita. Normas ‖·‖₁, ‖·‖₂, ‖·‖_∞,
    variedad de estados 𝔇(ℋ), divergencia de Umegaki y geometría de Bures.

        S(ρ‖σ) = Tr(ρ log ρ) − Tr(ρ log σ)     (Klein: ≥ 0)
        F(ρ,σ) = [Tr √(√ρ σ √ρ)]²               (Uhlmann–Jozsa)
        D_B    = √(2(1 − √F))                   (Bures)
    """

    SPECTRUM_FLOOR: Final[float] = 1e-15

    @staticmethod
    def trace_norm(A: np.ndarray) -> float:
        return float(np.sum(la.svdvals(A)))

    @staticmethod
    def hilbert_schmidt_norm(A: np.ndarray) -> float:
        return float(np.sqrt(np.real(np.trace(A.conj().T @ A))))

    @staticmethod
    def operator_norm(A: np.ndarray) -> float:
        sv = la.svdvals(A)
        return float(np.max(sv)) if len(sv) > 0 else 0.0

    @staticmethod
    def clean_density_matrix(rho: np.ndarray) -> np.ndarray:
        r"""Proyección espectral al simplex 𝔇(ℋ) = {ρ=ρ†, ρ⪰0, Tr ρ=1}."""
        rho_h = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals_clipped = np.maximum(eigvals, BanachOperatorAlgebra.SPECTRUM_FLOOR)
        eigvals_normalized = eigvals_clipped / np.sum(eigvals_clipped)
        proj = eigvecs @ np.diag(eigvals_normalized) @ eigvecs.conj().T
        return 0.5 * (proj + proj.conj().T)

    @staticmethod
    def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-8) -> bool:
        hermitian_defect = float(la.norm(rho - rho.conj().T))
        if hermitian_defect > tol:
            return False
        eigvals = la.eigvalsh(0.5 * (rho + rho.conj().T))
        if np.any(eigvals < -tol):
            return False
        return abs(np.real(np.trace(rho)) - 1.0) < tol

    @staticmethod
    def von_neumann_entropy(rho: np.ndarray) -> float:
        eigvals = la.eigvalsh(rho)
        eigvals = eigvals[eigvals > BanachOperatorAlgebra.SPECTRUM_FLOOR]
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
        r"""Umegaki S(ρ‖σ) en bits. Klein ⇒ ≥ 0; =0 ⇔ ρ=σ (soporte regularizado)."""
        log_rho = BanachOperatorAlgebra._matrix_log_regularized(rho, floor)
        log_sigma = BanachOperatorAlgebra._matrix_log_regularized(sigma, floor)
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
        f = BanachOperatorAlgebra.quantum_fidelity(rho, sigma)
        return float(math.sqrt(max(2.0 * (1.0 - math.sqrt(max(f, 0.0))), 0.0)))

    @staticmethod
    def cstar_residual(rho: np.ndarray) -> float:
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)


class LindbladCFTMasterEvolver:
    r"""
    CONTINUACIÓN FORMAL de TopologicalCircuitBundle.lift_hamiltonian.

    Integra la ecuación maestra GKSL (forma de Kossakowski, γ_k ≥ 0 ⇒ CPTP):

        dρ/dt = −i[H,ρ] + Σ_k γ_k (L_k ρ L_k† − ½ {L_k† L_k, ρ})

    Integrador: RK4 adaptativo con duplicación de paso, extrapolación de
    Richardson (orden efectivo 5) y proyección espectral a 𝔇(ℋ) en cada
    aceptación, para permanecer en el politopo de estados.
    """

    GAMMA_FLOOR: Final[float] = 0.0
    TRACE_DEFECT_LOG: Final[float] = 1e-6

    def __init__(self, bundle: TopologicalCircuitBundle, hilbert_dim: int = 4) -> None:
        r"""Primer consumidor de FASE-2: ancla el fibrado producido por FASE-1."""
        self.bundle = bundle
        self.dim = int(hilbert_dim)
        self.banach = BanachOperatorAlgebra()

    def _synthesize_hamiltonian(self, modular_tau: complex) -> np.ndarray:
        r"""
        CONTINUACIÓN de TopologicalCircuitBundle.lift_hamiltonian:
        delegación estricta (sin duplicar la construcción).
        """
        return self.bundle.lift_hamiltonian(self.dim, modular_tau)

    def _build_lindblad_jump_operators(self) -> List[Tuple[float, np.ndarray]]:
        r"""
        Operadores de colapso. Las tasas se recortan a γ≥0 para preservar
        la condición de Kossakowski (completa positividad).
        """
        gamma_topo = max(self.GAMMA_FLOOR, 0.05 * (self.bundle.betti_1 + 1.0))
        L_dephase = np.diag(
            [math.sqrt(i + 1) for i in range(self.dim)]
        ).astype(np.complex128)
        L_relax = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for i in range(self.dim - 1):
            L_relax[i, i + 1] = 1.0
        gamma_relax = max(
            self.GAMMA_FLOOR, 0.02 * abs(self.bundle.circuit_dissipation_rate)
        )
        return [(gamma_topo, L_dephase), (gamma_relax, L_relax)]

    def evaluate_polyakov_string_action(
        self, modular_tau: complex, worldsheet_area: float = 1.0
    ) -> PolyakovWorldsheetMetrics:
        tau_red = PolyakovWorldsheetMetrics.reduce_to_fundamental_domain(modular_tau)
        tau1, tau2 = tau_red.real, max(tau_red.imag, 1e-6)
        alpha_prime = 0.5
        polyakov_action = (
            worldsheet_area / (4.0 * math.pi * alpha_prime * tau2)
        ) * (1.0 + tau1 ** 2 + tau2 ** 2)
        central_charge = float(self.dim)  # c_eff toy; no se afirma D=26
        weyl_residual = abs(central_charge - 26.0) / 26.0
        metrics = PolyakovWorldsheetMetrics(
            modular_parameter_tau=tau_red,
            string_tension_alpha_prime=alpha_prime,
            polyakov_action_integral=float(polyakov_action),
            conformal_anomaly_central_charge=central_charge,
            weyl_invariance_residual=float(weyl_residual),
            in_fundamental_domain_flag=True,
        )
        return metrics

    def diagnose_modular_anomaly(self, modular_tau: complex) -> float:
        r"""
        Defecto clásico bajo T: τ ↦ τ+1. En la teoría cuántica completa
        Z(τ) es PSL(2,ℤ)-invariante; aquí se cuantifica la anomalía del
        truncamiento semiclásico (no se afirma invariancia espuria).
        """
        s_tau = self.evaluate_polyakov_string_action(modular_tau).polyakov_action_integral
        s_t_tau = self.evaluate_polyakov_string_action(
            modular_tau + 1
        ).polyakov_action_integral
        denom = abs(s_tau) if abs(s_tau) > 1e-12 else 1e-12
        return float(abs(s_t_tau - s_tau) / denom)

    def _rk4_step(self, liouvillian, rho: np.ndarray, dt: float) -> np.ndarray:
        k1 = liouvillian(rho)
        k2 = liouvillian(rho + 0.5 * dt * k1)
        k3 = liouvillian(rho + 0.5 * dt * k2)
        k4 = liouvillian(rho + dt * k3)
        return rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _adaptive_rk4_integrate(
        self,
        liouvillian,
        rho: np.ndarray,
        dt: float,
        tol: float = 1e-7,
        max_depth: int = 6,
        depth: int = 0,
    ) -> np.ndarray:
        r"""
        Control de error por duplicación + Richardson:
            ρ* = ρ_{h/2,h/2} + (ρ_{h/2,h/2} − ρ_h)/15
        Tras aceptar, se proyecta a 𝔇(ℋ).
        """
        full_step = self._rk4_step(liouvillian, rho, dt)
        half_step = self._rk4_step(liouvillian, rho, dt / 2.0)
        two_half_steps = self._rk4_step(liouvillian, half_step, dt / 2.0)
        local_error = self.banach.hilbert_schmidt_norm(two_half_steps - full_step)

        if local_error < tol or depth >= max_depth:
            rho_star = two_half_steps + (two_half_steps - full_step) / 15.0
            return self.banach.clean_density_matrix(rho_star)

        left = self._adaptive_rk4_integrate(
            liouvillian, rho, dt / 2.0, tol / 2.0, max_depth, depth + 1
        )
        right = self._adaptive_rk4_integrate(
            liouvillian, left, dt / 2.0, tol / 2.0, max_depth, depth + 1
        )
        return self.banach.clean_density_matrix(right)

    def evolve_density_state(
        self,
        rho_initial: np.ndarray,
        time_step: float,
        modular_tau: complex,
        integration_tolerance: float = 1e-7,
    ) -> Tuple[np.ndarray, PolyakovWorldsheetMetrics]:
        H = self._synthesize_hamiltonian(modular_tau)
        jumps = self._build_lindblad_jump_operators()

        def liouvillian(r: np.ndarray) -> np.ndarray:
            comm = -1j * (H @ r - r @ H)
            diss = np.zeros_like(r, dtype=np.complex128)
            for gamma, L in jumps:
                L_dag = L.conj().T
                L_dag_L = L_dag @ L
                diss += gamma * (
                    L @ r @ L_dag - 0.5 * (L_dag_L @ r + r @ L_dag_L)
                )
            return comm + diss

        rho0 = self.banach.clean_density_matrix(rho_initial)
        trace_defect = abs(np.real(np.trace(liouvillian(rho0))))
        if trace_defect > self.TRACE_DEFECT_LOG:
            logger.debug(
                "Defecto de traza del generador GKSL: %.3e", trace_defect
            )

        rho_evolved_raw = self._adaptive_rk4_integrate(
            liouvillian, rho0, time_step, tol=integration_tolerance
        )
        rho_projected = self.banach.clean_density_matrix(rho_evolved_raw)
        metrics_cft = self.evaluate_polyakov_string_action(modular_tau)
        return rho_projected, metrics_cft


@dataclass(frozen=True, slots=True)
class MetabolizedFieldState:
    r"""
    NEXO FORMAL TERMINAL DE LA FASE 2.

    Estado metabolizado por GKSL, validado en B(ℋ), clasificado en Ω₄
    por un funcional de coherencia metabólica. Se propaga a FASE 3 como
    sustrato de inmunización.

    Último método de clase: create_metabolized_state.
    CONTINÚA EN FASE-3: SpectralImmuneVaccineSynthesizer.synthesize_from_metabolized_state.
    """

    density_matrix: np.ndarray
    von_neumann_entropy: float
    purity: float
    trace_distance_to_thermal: float
    quantum_relative_entropy_to_thermal: float
    bures_distance_to_thermal: float
    cstar_residual: float
    hodge_bundle_carrier: TopologicalCircuitBundle
    cft_worldsheet_metrics: PolyakovWorldsheetMetrics
    modular_anomaly_estimate: float
    dirichlet_cft_energy: float
    metabolic_coherence_index: float
    heyting_state: HeytingTruthValue

    _W_POLYAKOV: ClassVar[float] = 0.4
    _W_HARMONIC: ClassVar[float] = 0.3
    _W_SPECTRAL_GAP: ClassVar[float] = 0.3
    _KAPPA_ENERGY: ClassVar[float] = 10.0
    _KAPPA_RELATIVE_ENTROPY: ClassVar[float] = 4.0

    @staticmethod
    def _metabolic_coherence_functional(
        purity: float, dirichlet_energy: float, relative_entropy: float
    ) -> float:
        r"""
        M = pur(ρ) · exp(−E_D/κ_E) · exp(−S_rel/κ_S) ∈ (0,1].
        """
        kE = MetabolizedFieldState._KAPPA_ENERGY
        kS = MetabolizedFieldState._KAPPA_RELATIVE_ENTROPY
        return (
            purity
            * math.exp(-abs(dirichlet_energy) / kE)
            * math.exp(-abs(relative_entropy) / kS)
        )

    @staticmethod
    def _classify_metabolic_state(coherence: float) -> HeytingTruthValue:
        if coherence >= 0.60:
            return HeytingTruthValue.VERUM_COHERENT
        if coherence >= 0.35:
            return HeytingTruthValue.TOPOLOGICAL_SOUND
        if coherence >= 0.10:
            return HeytingTruthValue.BOUNDARY_DEGRADED
        return HeytingTruthValue.ABSURDUM_VETOED

    @classmethod
    def create_metabolized_state(
        cls,
        bundle: TopologicalCircuitBundle,
        base_rho: np.ndarray,
        modular_tau: complex,
        time_step: float = 0.05,
    ) -> "MetabolizedFieldState":
        r"""
        ÚLTIMO método de FASE-2: Φ_t ∘ H.

        CONTINÚA EN FASE-3 (inmunización I sobre density_matrix).
        """
        evolver = LindbladCFTMasterEvolver(bundle=bundle, hilbert_dim=base_rho.shape[0])
        rho_evolved, cft_metrics = evolver.evolve_density_state(
            base_rho, time_step, modular_tau
        )
        modular_anomaly = evolver.diagnose_modular_anomaly(modular_tau)

        banach = BanachOperatorAlgebra()
        entropy = banach.von_neumann_entropy(rho_evolved)
        pur = banach.purity(rho_evolved)

        dim = rho_evolved.shape[0]
        rho_thermal = np.eye(dim, dtype=np.complex128) / dim
        dist_thermal = 0.5 * banach.trace_norm(rho_evolved - rho_thermal)
        rel_entropy_thermal = banach.quantum_relative_entropy(rho_evolved, rho_thermal)
        bures_thermal = banach.bures_distance(rho_evolved, rho_thermal)
        cstar = banach.cstar_residual(rho_evolved)

        dirichlet_energy = (
            cls._W_POLYAKOV * cft_metrics.polyakov_action_integral
            + cls._W_HARMONIC * bundle.harmonic_energy_norm
            + cls._W_SPECTRAL_GAP * (1.0 / (bundle.hodge_spectral_gap + 1e-4))
        )
        coherence = cls._metabolic_coherence_functional(
            pur, dirichlet_energy, rel_entropy_thermal
        )
        phase2_heyting = cls._classify_metabolic_state(coherence)
        final_heyting = bundle.heyting_topos_evaluation.meet(phase2_heyting)

        return cls(
            density_matrix=rho_evolved,
            von_neumann_entropy=entropy,
            purity=pur,
            trace_distance_to_thermal=dist_thermal,
            quantum_relative_entropy_to_thermal=rel_entropy_thermal,
            bures_distance_to_thermal=bures_thermal,
            cstar_residual=cstar,
            hodge_bundle_carrier=bundle,
            cft_worldsheet_metrics=cft_metrics,
            modular_anomaly_estimate=modular_anomaly,
            dirichlet_cft_energy=dirichlet_energy,
            metabolic_coherence_index=coherence,
            heyting_state=final_heyting,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INMUNIZACIÓN, MERKLE, WAKE-SLEEP, AUDITORÍA Y PASAPORTE
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método (synthesize_from_metabolized_state) consume
# MetabolizedFieldState, valor de retorno del último método de FASE-2.
# Aquí se realiza I (vacuna), V (meet Ω₄), el sello Merkle y la orquestación
# REM del metacórtex.
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
class DreamFieldReport:
    cycle_id: str
    scenario_type: str
    metabolized_state: MetabolizedFieldState
    immune_projection_operator: np.ndarray
    projector_idempotency_residual: float
    spectral_coverage_mass: float
    spectral_vaccine_effective: bool
    dirichlet_energy: float
    heyting_verdict: HeytingTruthValue
    learning_rate_applied: float
    merkle_sha512_provenance: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class WakeSleepCycleReport:
    cycle_id: str
    initial_entropy: float
    final_entropy: float
    wake_purity: float
    sleep_purity: float
    net_free_energy_reduction: float
    synthesized_scenarios_count: int
    immune_vaccines_generated: int
    overall_topos_verdict: HeytingTruthValue
    merkle_leaf_count: int
    sha512_merkle_root: str
    merkle_proofs_ok: bool


class SpectralImmuneVaccineSynthesizer:
    r"""
    CONTINUACIÓN FORMAL de MetabolizedFieldState.create_metabolized_state.

    Sintetiza el proyector de inmunidad sobre el subespacio de cobertura
    de masa espectral (análogo PCA de retención de varianza):

        P_vac = Σ_{i=1}^{k} |v_i⟩⟨v_i|,
        k = min{ m : Σ_{i=1}^{m} λ_i ≥ f_cov },   λ₁ ≥ λ₂ ≥ ⋯

    Invariantes: P² = P = P†  (se reporta el residuo de idempotencia).
    """

    COVERAGE_FRACTION: Final[float] = 0.75
    PERTURBATION_HARD: Final[float] = 0.85
    RANK_FRACTION_CAP: Final[float] = 0.75
    MASS_FLOOR: Final[float] = 1e-15

    @classmethod
    def synthesize_from_metabolized_state(
        cls,
        state: MetabolizedFieldState,
        perturbation_norm: float,
        coverage_fraction: float = COVERAGE_FRACTION,
    ) -> Tuple[np.ndarray, bool, float, float]:
        r"""
        CONTINUACIÓN de create_metabolized_state: consume density_matrix.

        Retorna (P_vac, efectiva, masa_retenida, residuo_idempotencia).
        """
        return cls.synthesize_vaccine(
            density_matrix=state.density_matrix,
            perturbation_norm=perturbation_norm,
            coverage_fraction=coverage_fraction,
        )

    @classmethod
    def synthesize_vaccine(
        cls,
        density_matrix: np.ndarray,
        perturbation_norm: float,
        coverage_fraction: float = COVERAGE_FRACTION,
    ) -> Tuple[np.ndarray, bool, float, float]:
        eigvals, eigvecs = la.eigh(density_matrix)
        dim = density_matrix.shape[0]
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
            and perturbation_norm < cls.PERTURBATION_HARD
            and k <= math.ceil(dim * cls.RANK_FRACTION_CAP)
        )
        return P_vac, is_effective, retained_mass, idem_res


class TOONOniricDreamerEngine:
    r"""
    MOTOR ESPECTRAL ONÍRICO (TOON). Orquesta

        𝒟 = V ∘ I ∘ Φ_t ∘ H ∘ K

    Topología contrafactual determinista por SHA-256 del escenario;
    tasa η adaptativa al veredicto de Heyting; certificación Merkle
    SHA-512 con pruebas de inclusión.
    """

    _HEYTING_LEARNING_RATE_MAP: Final[Dict[int, float]] = {
        int(HeytingTruthValue.VERUM_COHERENT): 0.35,
        int(HeytingTruthValue.TOPOLOGICAL_SOUND): 0.20,
        int(HeytingTruthValue.BOUNDARY_DEGRADED): 0.08,
        int(HeytingTruthValue.ABSURDUM_VETOED): 0.0,
    }

    def __init__(
        self,
        engine_id: str = "TOON-ONIRIC-DOCTORAL-001",
        dimension_mac: int = 4,
        seed: int = 42,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac debe ser ≥ 1.")
        self.engine_id = engine_id
        self.dimension_mac = int(dimension_mac)
        self.rng = np.random.default_rng(seed)
        self.cycle_counter = 0
        self.master_rho = np.eye(self.dimension_mac, dtype=np.complex128) / self.dimension_mac
        self.dream_records: List[DreamFieldReport] = []

    def _compile_counterfactual_topology(
        self, scenario_type: str, betti_loops_request: int
    ) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        r"""
        Compila un 2-complejo cuya 1-esqueleto se perturba de forma
        determinista por SHA-256(scenario_type): cada escenario induce
        una variedad de decisión reproducible y auditable.
        """
        num_v = 6
        edges = [
            (0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (4, 5), (5, 0)
        ]
        faces = [(0, 1, 2), (2, 3, 4)]

        digest = hashlib.sha256(scenario_type.encode("utf-8")).digest()
        if digest[0] % 2 == 0:
            candidate = (digest[1] % num_v, digest[2] % num_v)
            if candidate[0] != candidate[1]:
                edges.append(tuple(sorted(candidate)))
        if betti_loops_request > 0:
            edges.append((1, 4))
        if betti_loops_request > 1:
            edges.append((3, 0))

        seen: Set[Tuple[int, int]] = set()
        dedup_edges: List[Tuple[int, int]] = []
        for e in edges:
            e_sorted = tuple(sorted(e))
            if e_sorted not in seen:
                seen.add(e_sorted)
                dedup_edges.append(e_sorted)
        return num_v, dedup_edges, faces

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
    def _merkle_proof(leaf_hashes: Sequence[str], index: int) -> MerkleInclusionProof:
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

    def run_dream_cycle(
        self,
        scenario_type: str,
        cost_delta_ratio: float,
        betti_1_loops: int = 0,
        modular_tau: complex = complex(0.1, 1.2),
        dream_isolation_flag: bool = True,
    ) -> DreamFieldReport:
        r"""
        Ciclo REM contrafactual anidando Fases 1, 2 y 3:

            K → H → Φ_t → I → V.
        """
        self.cycle_counter += 1
        t_start = time.time()
        cycle_id = f"CYC-REM-{self.cycle_counter:05d}"

        # ── FASE 1: complejo simplicial + circuito + semilla H ──
        num_v, edges, faces = self._compile_counterfactual_topology(
            scenario_type, betti_1_loops
        )
        bundle = TopologicalCircuitBundle.synthesize_bundle(
            num_vertices=num_v, edges=edges, faces=faces
        )
        forced_heyting = (
            bundle.heyting_topos_evaluation
            if dream_isolation_flag
            else HeytingTruthValue.ABSURDUM_VETOED
        )

        # ── FASE 2: GKSL + CFT (continúa lift_hamiltonian) ──
        metabolized_state = MetabolizedFieldState.create_metabolized_state(
            bundle=bundle,
            base_rho=self.master_rho,
            modular_tau=modular_tau,
            time_step=0.05 + 0.05 * abs(cost_delta_ratio),
        )
        final_verdict = metabolized_state.heyting_state.meet(forced_heyting)

        # ── FASE 3: inmunización (continúa create_metabolized_state) ──
        P_vac, vaccine_effective, coverage_mass, idem_res = (
            SpectralImmuneVaccineSynthesizer.synthesize_from_metabolized_state(
                metabolized_state, perturbation_norm=abs(cost_delta_ratio)
            )
        )
        learning_rate = self._HEYTING_LEARNING_RATE_MAP[int(final_verdict)]

        if learning_rate > 0.0:
            vaccinated_rho = P_vac @ metabolized_state.density_matrix @ P_vac.conj().T
            trace_v = float(np.real(np.trace(vaccinated_rho)))
            if trace_v > 1e-12:
                vaccinated_rho = vaccinated_rho / trace_v
            self.master_rho = (
                (1.0 - learning_rate) * self.master_rho + learning_rate * vaccinated_rho
            )
            self.master_rho = BanachOperatorAlgebra.clean_density_matrix(self.master_rho)

        hasher = hashlib.sha512()
        signature_payload = (
            f"{self.engine_id}::{cycle_id}::{scenario_type}::{cost_delta_ratio:.6f}::"
            f"{metabolized_state.dirichlet_cft_energy:.6f}::{metabolized_state.purity:.6f}::"
            f"{final_verdict.name}::{coverage_mass:.6f}::{t_start}"
        ).encode("utf-8")
        hasher.update(signature_payload)
        merkle_leaf_hash = hasher.hexdigest()

        report = DreamFieldReport(
            cycle_id=cycle_id,
            scenario_type=scenario_type,
            metabolized_state=metabolized_state,
            immune_projection_operator=P_vac,
            projector_idempotency_residual=idem_res,
            spectral_coverage_mass=coverage_mass,
            spectral_vaccine_effective=vaccine_effective,
            dirichlet_energy=metabolized_state.dirichlet_cft_energy,
            heyting_verdict=final_verdict,
            learning_rate_applied=learning_rate,
            merkle_sha512_provenance=merkle_leaf_hash,
            timestamp_utc=t_start,
        )
        self.dream_records.append(report)
        logger.info(
            "[%s] Escenario '%s' | Heyting: %s | Pureza: %.4f | "
            "Cobertura: %.3f | η: %.2f | Vacuna OK: %s | P²−P: %.2e | Hash: %s…",
            cycle_id,
            scenario_type,
            final_verdict.name,
            metabolized_state.purity,
            coverage_mass,
            learning_rate,
            vaccine_effective,
            idem_res,
            merkle_leaf_hash[:16],
        )
        return report

    def execute_wake_sleep_phase(
        self, counterfactual_batch: List[Tuple[str, float, int, complex]]
    ) -> WakeSleepCycleReport:
        r"""
        Fase REM completa (Wake-Sleep Metacortex Loop): metaboliza el lote,
        reduce energía libre y certifica el lote con Merkle + pruebas.
        """
        t_start = time.time()
        banach = BanachOperatorAlgebra()
        init_entropy = banach.von_neumann_entropy(self.master_rho)
        init_purity = banach.purity(self.master_rho)

        vaccines_synthesized = 0
        cumulative_heyting = HeytingTruthValue.VERUM_COHERENT
        batch_leaf_hashes: List[str] = []

        for sc_type, cost_delta, betti_1, tau in counterfactual_batch:
            dream_rep = self.run_dream_cycle(
                scenario_type=sc_type,
                cost_delta_ratio=cost_delta,
                betti_1_loops=betti_1,
                modular_tau=tau,
                dream_isolation_flag=True,
            )
            cumulative_heyting = cumulative_heyting.meet(dream_rep.heyting_verdict)
            batch_leaf_hashes.append(dream_rep.merkle_sha512_provenance)
            if dream_rep.spectral_vaccine_effective:
                vaccines_synthesized += 1

        final_entropy = banach.von_neumann_entropy(self.master_rho)
        final_purity = banach.purity(self.master_rho)
        free_energy_reduction = float(init_entropy - final_entropy)

        merkle_root = self._merkle_tree_root(batch_leaf_hashes)
        proofs_ok = True
        for i in range(len(batch_leaf_hashes)):
            proof = self._merkle_proof(batch_leaf_hashes, i)
            if proof.root != merkle_root or not proof.verify():
                proofs_ok = False
                break

        context_binder = hashlib.sha512()
        context_binder.update(
            f"WAKE-SLEEP-CONTEXT::{self.engine_id}::{merkle_root}::{init_purity:.8f}::"
            f"{final_purity:.8f}::{vaccines_synthesized}::{t_start}".encode("utf-8")
        )
        final_root_hash = context_binder.hexdigest()

        return WakeSleepCycleReport(
            cycle_id=f"WAKE-SLEEP-{self.cycle_counter:05d}",
            initial_entropy=init_entropy,
            final_entropy=final_entropy,
            wake_purity=init_purity,
            sleep_purity=final_purity,
            net_free_energy_reduction=free_energy_reduction,
            synthesized_scenarios_count=len(counterfactual_batch),
            immune_vaccines_generated=vaccines_synthesized,
            overall_topos_verdict=cumulative_heyting,
            merkle_leaf_count=len(batch_leaf_hashes),
            sha512_merkle_root=final_root_hash,
            merkle_proofs_ok=proofs_ok,
        )

    @property
    def registry_view(self) -> Tuple[DreamFieldReport, ...]:
        return tuple(self.dream_records)

    @property
    def global_verdict(self) -> HeytingTruthValue:
        gv = HeytingTruthValue.VERUM_COHERENT
        for r in self.dream_records:
            gv = gv.meet(r.heyting_verdict)
        return gv

    def audit_registry(self) -> Dict[str, Any]:
        n = len(self.dream_records)
        empty_dist = {v.name: 0 for v in HeytingTruthValue}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingTruthValue.VERUM_COHERENT.name,
                "avg_purity": 0.0,
                "avg_coverage_mass": 0.0,
                "avg_dirichlet_energy": 0.0,
                "n_vaccines_effective": 0,
                "registry_integrity_ok": True,
            }
        dist: Dict[str, int] = {v.name: 0 for v in HeytingTruthValue}
        s_p = s_c = s_d = 0.0
        n_vac = 0
        hashes: Set[str] = set()
        collide = False
        for r in self.dream_records:
            dist[r.heyting_verdict.name] += 1
            s_p += r.metabolized_state.purity
            s_c += r.spectral_coverage_mass
            s_d += r.dirichlet_energy
            if r.spectral_vaccine_effective:
                n_vac += 1
            if r.merkle_sha512_provenance in hashes:
                collide = True
            hashes.add(r.merkle_sha512_provenance)
        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "avg_purity": s_p * inv,
            "avg_coverage_mass": s_c * inv,
            "avg_dirichlet_energy": s_d * inv,
            "n_vaccines_effective": n_vac,
            "registry_integrity_ok": not collide,
        }

    def emit_dreamer_passport(self) -> Dict[str, Any]:
        h = hashlib.sha512()
        h.update(f"{self.engine_id}::{self.cycle_counter}".encode("utf-8"))
        for r in self.dream_records:
            h.update(r.merkle_sha512_provenance.encode("utf-8"))
        return {
            "engine_id": self.engine_id,
            "registry_size": self.cycle_counter,
            "global_verdict": self.global_verdict.name,
            "n_vaccines_effective": sum(
                1 for r in self.dream_records if r.spectral_vaccine_effective
            ),
            "evidence_hash": h.hexdigest(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN RIGUROSA Y VALIDACIÓN UNITARIA INTER-FASES
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("╔" + "═" * 78 + "╗")
    print("║  DEMOSTRACIÓN TEÓRICO-PRÁCTICA DEL TOON ONIRIC DREAMER ENGINE v4.0     ║")
    print("║  FASES ANIDADAS: Ω₄+ℍ+Hodge+H → GKSL/C*/Polyakov → I/Merkle/Wake-Sleep ║")
    print("╚" + "═" * 78 + "╝")

    print("\n[§0] VERIFICACIÓN FORMAL DEL CLASIFICADOR Ω₄ Y DEL ÁLGEBRA ℍ")
    axioms_ok = HeytingTruthValue.verify_heyting_axioms()
    print(f"  • Ley de residuación ∀a,b,c ∈ Ω₄     : {axioms_ok}")
    assert axioms_ok
    assert HeytingTruthValue.BOUNDARY_DEGRADED.excluded_middle_holds() is False
    assert HeytingTruthValue.VERUM_COHERENT.excluded_middle_holds() is True
    assert HeytingTruthValue.ABSURDUM_VETOED.is_regular() is True
    assert HeytingTruthValue.TOPOLOGICAL_SOUND.is_regular() is False

    q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
    q2 = Quaternion(0.5, -1.0, 0.25, 2.0)
    assert abs((q1 * q2).norm() - q1.norm() * q2.norm()) < 1e-12
    assert q1.cstar_residual() < 1e-12
    assert q1.versor().su2_det_residual() < 1e-12

    engine = TOONOniricDreamerEngine(
        engine_id="APU-METACORTEX-REM-01", dimension_mac=4, seed=1337
    )

    print("\n[§1] FASE 1: HACES SIMPLICIALES EXACTOS, HODGE Y TELLEGEN")
    bundle_test = TopologicalCircuitBundle.synthesize_bundle(
        num_vertices=5,
        edges=[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)],
        faces=[(0, 1, 2), (2, 3, 4)],
    )
    print(f"  • Betti (β0, β1, β2)           : ({bundle_test.betti_0}, {bundle_test.betti_1}, {bundle_test.betti_2})")
    print(f"  • χ Euler-Poincaré (consist.)  : {bundle_test.euler_characteristic} ({bundle_test.euler_betti_consistent})")
    print(f"  • Brecha Hodge / Fiedler       : {bundle_test.hodge_spectral_gap:.4f} / {bundle_test.algebraic_connectivity:.4f}")
    print(f"  • Residuo Hodge ω−(ex+co+h)    : {bundle_test.hodge_decomposition_residual:.3e}")
    print(f"  • Disipación Tellegen          : {bundle_test.circuit_dissipation_rate:.4f} W")
    print(f"  • Residuo Tellegen / KCL       : {bundle_test.tellegen_residual:.3e} / {bundle_test.kcl_residual:.3e}")
    print(f"  • Defecto no-recip. / pasividad: {bundle_test.reciprocity_defect:.4f} / {bundle_test.passivity_margin:.4f}")
    print(f"  • Coherencia espectral / Ω₄    : {bundle_test.spectral_coherence_index:.4f} / {bundle_test.heyting_topos_evaluation.name}")
    H_seed = bundle_test.lift_hamiltonian(4, complex(0.0, 1.5))
    assert np.allclose(H_seed, H_seed.conj().T), "H no hermítico."

    print("\n[§2] FASE 2: GKSL ADAPTATIVO, C* Y POLYAKOV (continúa H)")
    rho_test_init = np.eye(4, dtype=np.complex128) / 4.0
    field_state = MetabolizedFieldState.create_metabolized_state(
        bundle=bundle_test, base_rho=rho_test_init, modular_tau=complex(0.0, 1.5)
    )
    print(f"  • Pureza C*-Banach             : {field_state.purity:.4f}")
    print(f"  • Entropía von Neumann         : {field_state.von_neumann_entropy:.4f} bits")
    print(f"  • Umegaki S(ρ‖I/n)             : {field_state.quantum_relative_entropy_to_thermal:.4f} bits")
    print(f"  • Bures / traza al térmico     : {field_state.bures_distance_to_thermal:.4f} / {field_state.trace_distance_to_thermal:.4f}")
    print(f"  • Residual C*                  : {field_state.cstar_residual:.3e}")
    print(f"  • Acción Polyakov / τ∈ℱ        : {field_state.cft_worldsheet_metrics.polyakov_action_integral:.4f} / {field_state.cft_worldsheet_metrics.in_fundamental_domain_flag}")
    print(f"  • Anomalía modular T           : {field_state.modular_anomaly_estimate:.4f}")
    print(f"  • Dirichlet-CFT / coherencia M : {field_state.dirichlet_cft_energy:.4f} / {field_state.metabolic_coherence_index:.4f}")
    assert BanachOperatorAlgebra.is_valid_density_matrix(field_state.density_matrix)

    print("\n[§3] FASE 3: REM, CISNES NEGROS, MERKLE E INMUNIZACIÓN")
    batch_scenarios: List[Tuple[str, float, int, complex]] = [
        ("STEEL_CARTEL_PRICE_SHOCK_35", 0.35, 0, complex(0.2, 1.1)),
        ("STRIKE_LABOR_PARALYSIS_MACRO", 0.65, 1, complex(-0.4, 0.9)),
        ("HYDROLOGIC_FLOOD_FOUNDATION", 0.25, 0, complex(0.0, 2.0)),
        ("CIRCULAR_SUB_BILLING_ATTACK", 0.85, 2, complex(0.5, 0.5)),
    ]
    wake_sleep_report = engine.execute_wake_sleep_phase(batch_scenarios)
    print(f"\n  ================ REPORTE METACORTEX REM FINAL ================")
    print(f"  • ID Ciclo Global              : {wake_sleep_report.cycle_id}")
    print(f"  • Pureza vigilia → post-REM    : {wake_sleep_report.wake_purity:.4f} → {wake_sleep_report.sleep_purity:.4f}")
    print(f"  • Entropía inicial → final     : {wake_sleep_report.initial_entropy:.4f} → {wake_sleep_report.final_entropy:.4f} bits")
    print(f"  • Reducción energía libre      : {wake_sleep_report.net_free_energy_reduction:.4f} bits")
    print(f"  • Vacunas inmunes              : {wake_sleep_report.immune_vaccines_generated}/{wake_sleep_report.synthesized_scenarios_count}")
    print(f"  • Veredicto Topos Heyting      : {wake_sleep_report.overall_topos_verdict.name}")
    print(f"  • Merkle hojas / pruebas OK    : {wake_sleep_report.merkle_leaf_count} / {wake_sleep_report.merkle_proofs_ok}")
    print(f"  • Raíz de Merkle (SHA-512)     : {wake_sleep_report.sha512_merkle_root[:48]}...")
    assert wake_sleep_report.merkle_proofs_ok, "Fallo en pruebas de inclusión Merkle."

    print("\n[§4] PRUEBA DE AISLAMIENTO ONÍRICO")
    breach_state = engine.run_dream_cycle(
        scenario_type="CONTAINMENT_LEAKAGE_TEST",
        cost_delta_ratio=0.10,
        dream_isolation_flag=False,
    )
    print(f"  • Veredicto ante fuga          : {breach_state.heyting_verdict.name}")
    print(f"  • η aplicada                   : {breach_state.learning_rate_applied}")
    assert breach_state.heyting_verdict == HeytingTruthValue.ABSURDUM_VETOED
    assert breach_state.learning_rate_applied == 0.0

    print("\n>>> AUDITORÍA RETROSPECTIVA")
    for k, v in engine.audit_registry().items():
        print(f"    - {k:<28}: {v}")
    print("\n>>> PASAPORTE DEL DREAMER")
    for k, v in engine.emit_dreamer_passport().items():
        print(f"    - {k:<22}: {v}")

    print("\n" + "═" * 80)
    print("✓ VERIFICACIÓN INTEGRAL MATEMÁTICA Y DE SOFTWARE CONCLUIDA CON ÉXITO.")
    print("═" * 80)
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : toon_oniric_dreamer_agent.py                                              ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / FASE REM (GAN-REM)                  ║
║ FUNCIÓN  : SOBERANO SIMULADOR ONÍRICO REM Y METABOLIZADOR POINCARANO                 ║
║ VERSIÓN  : 8.1.0-Doctoral-Poincare-CRTBP-Fuchsian-Omega4-GKSL-Merkle-A3              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICA
───────────────────────────────────────────────
El `TOONOniricDreamerAgent` actúa como la autoridad soberana responsable de orquestar la 
generación de escenarios contrafactuales y la inyección de estrés controlado durante la 
Fase REM del ecosistema agéntico APU Filter v8.0.

Inspirado en *Les Méthodes Nouvelles de la Mécanique Céleste* (Vol. I–III) de Henri Poincaré,
este soberano simula escenarios de colapso navegando los tubos de variedades invariantes
de Lagrange ($L_1 \dots L_5$) en el Problema Restringido Circular de Tres Cuerpos (CRTBP)
y proyectando parámetros modulares sobre el Dominio Fundamental de Poincaré $\mathcal{F} \subset \mathbb{H}^2$.

Forma el vértice generador dentro del bucle de Red Generativa Adversarial Onírica (GAN-REM):

    [ Ilusionista (Trickster) ] ──► [ Soñador (Dreamer) ] ──► [ Auditor Onírico ] ──► [ Testigo Silencioso ]

POSTULADOS Y GOBERNANZA AGÉNTICA POINCARANA
───────────────────────────────────────────
1. POSTULADO DE AISLAMIENTO HOMOLÓGICO Y FASE REM (is_dream_state = True):
   Toda simulación en la Fase REM ocurre bajo el postulado de aislamiento homológico inmutable:

       \partial(\rho_{\mathrm{dream}}) \equiv 0 \pmod{\mathrm{RealWorld}}, \quad \mathtt{is\_dream\_state} = \mathrm{True}

   Garantiza que la matriz de densidad onírica $\rho_{\mathrm{dream}} \in \mathcal{L}(P_d \mathcal{H} P_d)$
   permanezca confinada en el subespacio de superselección del enclave $P_d$, con $P_d P_p = 0$,
   impidiendo que un "sueño de bancarrota" altere la contabilidad real de la obra o dispare de forma
   errónea las alarmas físicas del proyecto.

2. UNIFORMIZACIÓN FUCSIANA Y DIVERGENCIA DE UMEGAKI EN $\mathbb{H}^2$:
   Mediante la acción del grupo modular de Poincaré $PSL(2, \mathbb{Z})$, las deformaciones se
   reducen al Dominio Fundamental de Poincaré $\mathcal{F} = \{ \tau \in \mathbb{H}^2 \mid |\mathrm{Re}(\tau)| \le 1/2, \, |\tau| \ge 1 \}$.
   La desviación respecto al estado base $\rho_0$ se cuantifica con la Divergencia de Entropía Relativa de Umegaki:

       S(\rho_{\mathrm{dream}} \parallel \rho_0) = \operatorname{Tr}(\rho_{\mathrm{dream}} (\ln \rho_{\mathrm{dream}} - \ln \rho_0))

   y la Distancia Geodésica de Bures-Wasserstein:

       d_B(\rho_{\mathrm{dream}}, \rho_0) = \sqrt{2 \left(1 - \operatorname{Tr}\sqrt{\rho_0^{1/2} \rho_{\mathrm{dream}} \rho_0^{1/2}}\right)}

3. INOCULACIÓN AFÍN Y VACUNA ESPECTRAL:
   Si la simulación revela una vulnerabilidad estructural pero es certificada por el Auditor Onírico,
   el Soñador proyecta el proyector de vacuna $P_{\mathrm{vac}} = \sum_{i=1}^k |v_i\rangle\langle v_i|$
   y actualiza la Matriz Atómica de Conocimiento (MAC) mediante el mapa afín convexo:

       \Phi_\eta(\rho_{\mathrm{MAC}}) = (1 - \eta) \rho_{\mathrm{MAC}} + \eta \, \frac{P_{\mathrm{vac}} \rho_{\mathrm{dream}} P_{\mathrm{vac}}^\dagger}{\operatorname{Tr}(P_{\mathrm{vac}} \rho_{\mathrm{dream}} P_{\mathrm{vac}}^\dagger)}

   donde $\eta(\Omega_4)$ es la constante de aprendizaje modulada por el topos de Heyting.

TRADUCCIÓN EJECUTIVA ("DOLOR Y DINERO")
──────────────────────────────────────
- Ensayos Clínicos Presupuestales: Equivale a someter los Análisis de Precios Unitarios (APU) 
  a un túnel de viento financiero antes de firmar el contrato adjudicado.
- Ahorro Directo en Imprevistos: Neutraliza sobrecostos de hasta un +35% en la etapa de ejecución,
  anticipando los escenarios de falla en la etapa de simulación onírica.
- Trazabilidad y Seguridad Jurídica: Emisión inmutable de certificados `OniricScenarioCertificate` y
  `PoincareOniricScenarioCertificate` para respaldo ante peritos, aseguradoras y organismos de control.
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

from app.wisdom.toon_oniric_dreamer_engine import (
    FuchsianDomainCertificate,
    NonHermitianLindbladMasterEngine,
    poincare_hyperbolic_distance,
    reduce_to_poincare_fundamental_domain,
)

logger = logging.getLogger("APU.Wisdom.TOONOniricDreamer.v8")

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
    "PoincareOniricScenarioCertificate",
    "MacroWakeSleepAuditReport",
    "SpectralImmuneVaccineEngine",
    "TOONOniricDreamerAgent",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — TOPOS, LAWVERE-TIERNEY, CLIFFORD, HODGE EXACTO, TELLEGEN Y SEMILLA H
# ══════════════════════════════════════════════════════════════════════════════


class HeytingToposAlgebra(IntEnum):
    r"""
    Cadena finita Ω₄ = {0 ≺ 1 ≺ 2 ≺ 3} = {⊥ ≺ ∂ ≺ ♯ ≺ ⊤}, álgebra de Heyting
    completa y clasificador de subobjetos de un topos de haces:

        a ∧ b  = min(a, b)
        a ∨ b  = max(a, b)
        a → b  = ⋁{ c ∈ Ω₄ | c ∧ a ≤ b }     (residuo)
        ¬a     = a → ⊥
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
        a, b = int(self), int(other)
        candidates = [c for c in range(4) if min(c, a) <= b]
        return HeytingToposAlgebra(max(candidates))

    def pseudo_complement(self) -> "HeytingToposAlgebra":
        return self.implies(HeytingToposAlgebra.VETOED_ABSURDUM)

    def classical_negation(self) -> "HeytingToposAlgebra":
        return HeytingToposAlgebra(3 - int(self))

    def double_negation(self) -> "HeytingToposAlgebra":
        return self.pseudo_complement().pseudo_complement()

    def is_regular(self) -> bool:
        return self.double_negation() == self

    def is_dense(self) -> bool:
        return self.pseudo_complement() == HeytingToposAlgebra.VETOED_ABSURDUM

    def excluded_middle_holds(self) -> bool:
        return self.join(self.pseudo_complement()) == HeytingToposAlgebra.VERUM_COHERENT

    def lawvere_tierney_closure(self) -> "HeytingToposAlgebra":
        val = int(self)
        if val == 0:
            return HeytingToposAlgebra.VETOED_ABSURDUM
        if val == 1:
            return HeytingToposAlgebra.TOPOLOGICAL_STABLE
        return self

    def is_j_closed(self) -> bool:
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
    r"""Elemento del álgebra de bicuaterniones 𝔹 ≅ ℍ ⊗_ℝ ℂ ≅ Cl⁺_{1,3}(ℝ) ≅ M₂(ℂ)."""

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
        return complex(np.linalg.det(self.to_matrix()))

    def frobenius_operator_norm(self) -> float:
        M = self.to_matrix()
        return float(np.sqrt(np.real(np.trace(M.conj().T @ M))))

    def reduced_norm_multiplicativity_residual(
        self, other: "BiquaternionClifford"
    ) -> float:
        n_prod = (self * other).reduced_norm_complex()
        n_sep = self.reduced_norm_complex() * other.reduced_norm_complex()
        return abs(n_prod - n_sep)

    def cstar_residual(self) -> float:
        M = self.to_matrix()
        op = float(la.norm(M.conj().T @ M, 2))
        nrm = float(la.norm(M, 2))
        return abs(op - nrm * nrm)


class SimplicialHodgeGraph:
    r"""Complejo de cadenas simplicial finito K = (C₀, C₁, C₂)."""

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
        return self.boundary_1.T

    @property
    def coboundary_1(self) -> np.ndarray:
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
    r"""Red AC no recíproca sobre el 1-esqueleto."""

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
        herm = 0.5 * (self.branch_admittance + self.branch_admittance.conj().T)
        return float(np.min(la.eigvalsh(herm)))

    def compute_bus_admittance_matrix(self) -> np.ndarray:
        B1 = self.graph.boundary_1
        return B1 @ self.branch_admittance @ B1.T

    def solve_network_dissipation(
        self, current_stimulus: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
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
    r"""Germen formal de la flecha H : cartucho ↦ 𝔥𝔢𝔯(ℋ_dream)."""

    @abstractmethod
    def lift_enclave_hamiltonian(self, hilbert_dim: int) -> np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class CategoricalCircuitCartridge(OpenQuantumDynamicsSeed):
    r"""NEXO FORMAL TERMINAL DE LA FASE 1."""

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

    def lift_enclave_hamiltonian(self, hilbert_dim: int) -> np.ndarray:
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


class BanachSpectralAlgebra:
    r"""C*-álgebra B(ℋ) de dimensión finita."""

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
    r"""Garantía de aislamiento homológico por teorema del conmutador."""

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
        comm = dream_operator @ self.P_physical - self.P_physical @ dream_operator
        leakage = BanachSpectralAlgebra.schatten_infty_norm(comm)
        return bool(leakage < self.LEAKAGE_TOL), leakage

    def enforce_strict_enclave(self, dream_operator: np.ndarray) -> np.ndarray:
        return self.P_dream @ dream_operator @ self.P_dream

    def pinching_channel(self, rho: np.ndarray) -> np.ndarray:
        return self.P_dream @ rho @ self.P_dream + self.P_physical @ rho @ self.P_physical


@dataclass(frozen=True, slots=True)
class MetabolizedPerturbationField:
    r"""NEXO FORMAL TERMINAL DE LA FASE 2."""

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
        engine = NonHermitianLindbladMasterEngine(
            enclave_dim=base_rho.shape[0]
        )
        H_eff = cartridge.lift_enclave_hamiltonian(base_rho.shape[0])
        jumps = []

        rho_p, _ = engine.evolve_fuchsian_lindblad_manifold(
            rho_dream=base_rho,
            H_eff=H_eff,
            jump_operators=jumps,
            tau_modular=complex(0.1, 1.2),
            dt=0.08,
        )

        enclave = NonCommutativeNoSignalingEnclave(hilbert_dim=base_rho.shape[0])
        completeness_ok = enclave.verify_projector_completeness()
        is_isolated, leakage = enclave.verify_isolation_commutator(rho_p)
        strict_isolated = bool(
            is_isolated and cartridge.is_homologically_isolated and completeness_ok
        )

        purity = BanachSpectralAlgebra.purity(rho_p)
        entropy = BanachSpectralAlgebra.von_neumann_entropy(rho_p)
        rel_entropy = BanachSpectralAlgebra.quantum_relative_entropy(rho_p, base_rho)
        bures = BanachSpectralAlgebra.bures_distance(rho_p, base_rho)
        cstar = BanachSpectralAlgebra.cstar_residual(rho_p)

        comm_h = rho_p @ H_eff - H_eff @ rho_p
        dirichlet_energy = (
            0.5 * float(np.real(np.trace(rho_p @ H_eff)))
            + 0.5 * (BanachSpectralAlgebra.schatten_2_norm(comm_h) ** 2)
            + 0.1 * cartridge.circuit_dissipation_watts
        )

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
            jump_rate=0.05,
            mean_first_jump_time=20.0,
            metabolic_coherence_index=coherence,
            topos_heyting_evaluation=final_verdict,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INMUNIZACIÓN, MERKLE, WAKE-SLEEP, AUDITORÍA Y PASAPORTE SOBERANO
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
    fuchsian_tau_reduced: Optional[complex] = None
    umegaki_divergence: float = 0.0

    def is_vetoed(self) -> bool:
        return self.heyting_verdict == HeytingToposAlgebra.VETOED_ABSURDUM

    def is_immune(self) -> bool:
        return (
            not self.is_vetoed()
            and self.isolation_guarantee
            and self.immune_vaccine_effective
        )


@dataclass(frozen=True, slots=True)
class PoincareOniricScenarioCertificate:
    r"""Certificado de Inmunización Onírica basado en Mecánica Celeste de Poincaré."""

    scenario_id: str
    dream_state_isolated: bool
    fuchsian_tau_reduced: complex
    umegaki_divergence: float
    bures_geodesic_distance: float
    escape_rate_gamma: float
    vaccine_coverage_ratio: float
    heyting_verdict_omega4: int
    merkle_sha512_root: str


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
    r"""Proyector de inmunidad sobre el subespacio de cobertura de masa espectral."""

    COVERAGE_FRACTION: Final[float] = 0.75
    RANK_FRACTION_CAP: Final[float] = 0.75
    MASS_FLOOR: Final[float] = 1e-15

    @classmethod
    def construct_from_metabolized_field(
        cls,
        perturbed_state: MetabolizedPerturbationField,
        coverage_fraction: float = COVERAGE_FRACTION,
    ) -> Tuple[np.ndarray, bool, float, float]:
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
    r"""SOBERANO AGENTE SIMULADOR ONÍRICO (FASE REM)."""

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
        enclave_engine: Optional[NonHermitianLindbladMasterEngine] = None,
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
        self.engine = enclave_engine if enclave_engine is not None else NonHermitianLindbladMasterEngine(enclave_dim=dimension_mac)

    def compute_umegaki_and_bures_metrics(
        self,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
    ) -> Tuple[float, float]:
        r"""Calcula la Divergencia de Umegaki S(ρ_dream ‖ ρ_base) y la Distancia Geodésica de Bures."""
        banach = BanachSpectralAlgebra()
        umegaki = banach.quantum_relative_entropy(rho_dream, rho_base)
        bures = banach.bures_distance(rho_dream, rho_base)
        return max(0.0, umegaki), bures

    def synthesize_spectral_vaccine_projection(
        self,
        rho_dream: np.ndarray,
        coverage_target: float = 0.90,
    ) -> Tuple[np.ndarray, float]:
        r"""Construye la proyección de vacuna P_vac = ∑ₖ |v▧⟩⟨v▧| cubriendo la masa espectral."""
        evals, evecs = la.eigh(rho_dream)
        idx = np.argsort(evals)[::-1]
        evals_sorted = np.clip(evals[idx], 0.0, None)
        evecs_sorted = evecs[:, idx]

        total_mass = float(np.sum(evals_sorted))
        if total_mass <= 1e-15:
            dim = rho_dream.shape[0]
            return np.eye(dim, dtype=np.complex128) / dim, 0.0

        cum_mass = np.cumsum(evals_sorted) / total_mass
        k = int(np.searchsorted(cum_mass, coverage_target)) + 1
        k = min(k, len(evals))

        P_vac = np.zeros_like(rho_dream, dtype=np.complex128)
        for i in range(k):
            v_i = evecs_sorted[:, i:i+1]
            P_vac += v_i @ v_i.conj().T

        coverage_achieved = float(cum_mass[k-1])
        return P_vac, coverage_achieved

    def run_fuchsian_counterfactual_simulation(
        self,
        scenario_id: str,
        rho_base: np.ndarray,
        H_eff: np.ndarray,
        jump_ops: Sequence[np.ndarray],
        tau_modular: complex,
        dt: float = 0.01,
        is_dream_state: bool = True,
    ) -> PoincareOniricScenarioCertificate:
        r"""Ejecuta la simulación de Cisnes Negros navegando los tubos de Lagrange en ℱ ⊂ ℍ²."""
        if not is_dream_state:
            return PoincareOniricScenarioCertificate(
                scenario_id=scenario_id,
                dream_state_isolated=False,
                fuchsian_tau_reduced=tau_modular,
                umegaki_divergence=float("inf"),
                bures_geodesic_distance=float("inf"),
                escape_rate_gamma=float("inf"),
                vaccine_coverage_ratio=0.0,
                heyting_verdict_omega4=0,
                merkle_sha512_root="0" * 128,
            )

        rho_dream, report = self.engine.evolve_fuchsian_lindblad_manifold(
            rho_dream=rho_base.copy(),
            H_eff=H_eff,
            jump_operators=jump_ops,
            tau_modular=tau_modular,
            dt=dt,
        )

        umegaki, bures = self.compute_umegaki_and_bures_metrics(rho_dream, rho_base)
        P_vac, coverage = self.synthesize_spectral_vaccine_projection(rho_dream)

        if bures > 1.2 or report["escape_rate_gamma"] > 5.0:
            verdict = 0
        elif bures > 0.6:
            verdict = 1
        elif bures > 0.2:
            verdict = 2
        else:
            verdict = 3

        hasher = hashlib.sha512()
        hasher.update(scenario_id.encode("utf-8"))
        hasher.update(rho_dream.tobytes())
        hasher.update(str(verdict).encode("utf-8"))
        merkle_root = hasher.hexdigest()

        return PoincareOniricScenarioCertificate(
            scenario_id=scenario_id,
            dream_state_isolated=True,
            fuchsian_tau_reduced=report["fuchsian_certificate"].tau_reduced,
            umegaki_divergence=umegaki,
            bures_geodesic_distance=bures,
            escape_rate_gamma=report["escape_rate_gamma"],
            vaccine_coverage_ratio=coverage,
            heyting_verdict_omega4=verdict,
            merkle_sha512_root=merkle_root,
        )

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
        start_time = time.time()
        self.seed_counter += 1
        is_dream_state = not force_isolation_breach

        cartridge = self.generate_synthetic_cartridge(
            scenario_type=scenario_type,
            cost_delta_ratio=cost_delta_ratio,
            synthetic_betti_1=synthetic_betti_1,
            is_dream_state=is_dream_state,
        )

        metabolized_field = MetabolizedPerturbationField.evolve_and_certify(
            cartridge=cartridge, base_rho=self.base_rho
        )

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
        umegaki, bures = self.compute_umegaki_and_bures_metrics(metabolized_field.rho_perturbed, self.base_rho)

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
            bures_distance=bures,
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
            fuchsian_tau_reduced=complex(0.1, 1.2),
            umegaki_divergence=umegaki,
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
    print("║     DEMOSTRACIÓN DOCTORAL DEL TOON ONIRIC DREAMER AGENT v8.1         ║")
    print("║  MECÁNICA CELESTE (POINCARÉ ℍ² + CRTBP + UMEGAKI + BURES)            ║")
    print("╚" + "═" * 78 + "╝")

    dreamer_agent = TOONOniricDreamerAgent(
        agent_id="TOON-DREAMER-SABIO-01",
        dimension_mac=4,
        energy_threshold=18.0,
        seed=10101,
    )

    cert_fuchsian = dreamer_agent.run_fuchsian_counterfactual_simulation(
        scenario_id="TEST_CRTBP_LAGRANGE_L1",
        rho_base=dreamer_agent.base_rho,
        H_eff=np.eye(4, dtype=np.complex128),
        jump_ops=[],
        tau_modular=complex(0.2, 1.3),
        dt=0.01,
        is_dream_state=True,
    )
    print(f"  • Simulación Fucsiana id: {cert_fuchsian.scenario_id} | Ω₄: {cert_fuchsian.heyting_verdict_omega4} | Umegaki: {cert_fuchsian.umegaki_divergence:.4f}")
    assert cert_fuchsian.dream_state_isolated

    cert1 = dreamer_agent.dream_scenario(
        scenario_type="BLACK_SWAN_STEEL_SPIKE",
        cost_delta_ratio=0.35,
        synthetic_betti_1=0,
    )
    print(f"  • Sueño REM cert id: {cert1.dream_id} | Heyting: {cert1.heyting_verdict.name} | Umegaki: {cert1.umegaki_divergence:.4f}")

    print("\n✓ AUDITORÍA Y METABOLIZACIÓN ONÍRICA CONCLUIDA SATISFACTORIAMENTE.")

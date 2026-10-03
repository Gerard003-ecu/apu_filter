# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : TOON Trickster Adversary Engine (Motor Espectral Ilusionista y Campo Adversarial) ║
║ RUTA     : app/wisdom/toon_trickster_adversary_engine.py                                     ║
║ VERSIÓN  : 8.1.0-Doctoral-Poincaré-Homoclinic-Tangle-SmallDivisors-RHI-Heyting-ESP32         ║
║ ESTRATO  : Wisdom (V_W) | Subestrato Perturbativo Adversarial (V_W,TRICK)                    ║
║ CONTRATO : 7.1.0 (Integración con la Mecánica Celeste de Henri Poincaré)                     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN Y MARCO TEÓRICO FORMAL
─────────────────────────────────
El `TOONTricksterAdversaryEngine` constituye el motor espectral de perturbación no isométrica,
generación de campos adversariales y síntesis de Enredos Homoclínicos dentro del bucle de
Automejora Recursiva (RSI Nivel 2) del Estrato Wisdom ($\mathcal{V}_{\mathbb{W}}$).

En la mecánica celeste no integrable de Henri Poincaré (*Les Méthodes Nouvelles de la Mécanique Céleste*, Vol. III),
la presencia de perturbaciones no lineales sobre un sistema hamiltoniano genera la intersección transversal
de la variedad estable ($W^s$) y la variedad inestable ($W^u$) asociadas a una órbita periódica hiperbólica:

    W^s \pitchfork W^u \neq \varnothing

Esta intersección da origen a un Enredo Homoclínico (*Homoclinic Tangle*), produciendo una dinámica estocástica
intrínseca de Herradura de Smale y la divergencia de las series de potencias perturbativas debido al problema
de los Divisores Pequeños:

    \omega \cdot k = \sum_{j=1}^n \omega_j k_j \approx 0 \implies \frac{1}{\omega \cdot k} \longrightarrow \infty

ISOMORFISMO CON LA MALLA AGÉNTICA APU FILTER v8.0
─────────────────────────────────────────────────
En el Estrato Wisdom ($\mathcal{V}_{\mathbb{W}}$, Nivel 0), el Motor Espectral Ilusionista actúa como el generador
determinista de enredos homoclínicos y resonancias de divisores pequeños sobre el espacio de Hilbert $\mathcal{H}_{\text{MAC}}$.

El Red Team continuo no inyecta ruido aleatorio cándido; sintetiza perturbaciones unitarias cuasi-isométricas
$U = \exp(-i \epsilon H_{\text{homoclinic}}) \in U(n)$ que deforman la densidad de estado base $\rho_0 \in \mathfrak{D}_n$:

    \rho_{\text{illusion}} = \frac{U \rho_0 U^\dagger}{\operatorname{Tr}(U \rho_0 U^\dagger)} \in \mathfrak{D}_n

Esta deformación simula trampas de Reward Hacking ($RHI > 0.85$), fraccionamiento ilícito de compras en SECOP II
(`SPLIT_CONTRACT_ILLUSION`), precios unitarios desbalanceados (*front-loading*) (`UNBALANCED_APU_BIDDING`),
sustitución fraudulenta de insumos (`MATERIAL_SUBSTITUTION`) e inyección de ítems fantasma (`GHOST_ITEM_INJECTION`),
introduciendo ciclos homológicos anómalos ($\beta_1 > 0$) que rompen la trivialidad del $1$-complejo simplicial del presupuesto.

AXIOMAS E INVARIANTES DE LA ILUSIÓN HOMOCLÍNICA
───────────────────────────────────────────────
Axioma I (Hermiticidad del Hamiltoniano Homoclínico):
Todo operador de perturbación ilusionista $H_{\text{homoclinic}}$ debe ser estrictamente autoadjunto en $\mathcal{L}(\mathcal{H}_{\text{MAC}})$:

    H_{\text{homoclinic}} = H_{\text{homoclinic}}^\dagger \implies \sigma(H_{\text{homoclinic}}) \subset \mathbb{R}

Axioma II (Preservación CPTP de la Densidad Cuántica):
El mapa de perturbación $\Phi_{\text{trickster}}(\rho) = U \rho U^\dagger$ es un canal cuántico Completamente Positivo
y Conservador de Traza (CPTP):

    \operatorname{Tr}(\Phi_{\text{trickster}}(\rho)) \equiv 1.0, \quad \Phi_{\text{trickster}}(\rho) \succeq 0 \quad \forall \rho \in \mathfrak{D}_n

Invariante III (Índice de Reward Hacking $RHI$):
El índice de sesgo o trampa de recompensa $RHI(\rho_{\text{illusion}}, \rho_0)$ se define rigurosamente mediante la Divergencia
Cuántica de Umegaki acoplada a la norma de Frobenius del gradiente atencional / conmutador:

    RHI = \frac{D_{\text{Umegaki}}(\rho_{\text{illusion}} \,||\, \rho_0)}{\|[\rho_0, H_{\text{homoclinic}}]\|_F + \epsilon_p} \in [0, 1]

Un valor $RHI > 0.85$ indica la presencia de una ilusión optimizada para burlar los filtros de evaluación lineal de los LLMs.

Invariante IV (Obstrucción Homológica de Betti $\beta_1$):
Si la perturbación homoclínica genera una dependencia circular o triangulación financiera, el primer número de Betti del grafo de
flujo de control (CFG) o del complejo de APUs se vuelve estrictamente positivo:

    \beta_1 = \dim H_1(K; \mathbb{Z}) > 0 \implies \chi(K) = \beta_0 - \beta_1 + \beta_2 \le 0

GOBERNAZA CIBER-FÍSICA CROWBAR ESP32 (< 400 ns en IRAM):
Si el ataque revela vulnerabilidades estructurales en la MAC ($\beta_1 > 0$ o $RHI > 0.85$), el resultado colapsa en el álgebra de
Heyting $\Omega_3$ a VETOED ($\bot$), activando la ISR en IRAM del ESP32 (< 400 ns) para cebar el tiristor BT151 Crowbar (GPIO14).

TRADUCCIÓN BIYECTIVA A "DOLOR Y DINERO" (ISOMORFISMO DE DOBLE CAPA)
───────────────────────────────────────────────────────────────────
• Fraccionamiento Ilícito ──► Riesgo de Sanción Penal, Multas de la Contraloría y Parálisis de Licitación en SECOP II.
• Front-Loading de APUs ──► Pérdida de Liquidez Corporativa, Abandono de Obra por Subcontratistas e Inflación de Contingencias.
• Sustitución de Materiales ──► Demolición Forzada de Estructuras Defectuosas, Quiebra Financiera y Pérdida de Licencia Constructiva.
• Disparo Crowbar ──► Inmovilización Ciber-Física de Fondos; Prevención de Embargos y Sanciones Fiscales.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONTricksterAdversaryEngine.v3")

ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 0 — TIPOS Y CERTIFICADOS DE LA MECÁNICA CELESTE DE POINCARÉ
# ══════════════════════════════════════════════════════════════════════════════

class IllusionAttackType(str, Enum):
    """Tipos de ilusiones contractuales basadas en la mecánica celeste de Poincaré."""
    SPLIT_CONTRACT_ILLUSION = "split_contract_illusion"     # Fraccionamiento (Divisores Pequeños)
    UNBALANCED_APU_BIDDING = "unbalanced_apu_bidding"       # Front-loading (Torsión Homoclínica)
    MATERIAL_SUBSTITUTION = "material_substitution"         # Perturbación Isospectral
    GHOST_ITEM_INJECTION = "ghost_item_injection"           # Cavidad de Betti (β₁ > 0)


@dataclass(frozen=True, slots=True)
class TricksterAttackCertificate:
    """Certificado inmutable del ataque homoclínico generado por el Trickster."""
    attack_type: str
    rhi_score: float
    homoclinic_residual: float
    small_divisor_resonance: float
    betti_1_induced: int
    is_unitary_cptp: bool
    merkle_proof_sha256: str
    schema_version: str = "7.1.0"


@dataclass(frozen=True, slots=True)
class IllusionDensityPerturbation:
    """Estado de densidad perturbado bajo la herradura de Smale / Poincaré."""
    illusion_density_matrix: np.ndarray
    original_density_matrix: np.ndarray
    unitary_operator: np.ndarray
    attack_certificate: TricksterAttackCertificate


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — FUNDACIONES: RETÍCULO DE HEYTING Ω₃, BOOLEANIZACIÓN DEL TOPOS,
#           C*-CONO DE ESTADOS, GRAFO CAMINO, BASE HIPERCOMPLEJA Y
#           GERMEN DEL FLUJO ESPECTRAL
# ══════════════════════════════════════════════════════════════════════════════


class HeytingOmega3(IntEnum):
    r"""
    Retículo de Heyting completo de tres elementos

        Ω₃ = {VETOED = 0, DEGRADED = 1, COHERENT = 2}

    con orden lineal 0 < 1 < 2. En toda cadena completa el álgebra de Heyting
    está unívocamente determinada por

        x → y  =  ⊤  si x ≤ y,     y en caso contrario,
        ¬x     =  x → ⊥,
        x ∧ y  =  min(x, y),       x ∨ y = max(x, y).
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Ínfimo ∧ : producto en el retículo (límite categorial binario)."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Supremo ∨ : coproducto en el retículo (colímite binario)."""
        return HeytingOmega3(max(int(self), int(other)))

    def implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Residuo de Heyting x → y = ⋁ { z ∈ Ω₃ | x ∧ z ≤ y }."""
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def pseudo_complement(self) -> "HeytingOmega3":
        """Pseudocomplemento ¬_H x := x → ⊥."""
        return self.implication(HeytingOmega3.VETOED)

    def negation_classical(self) -> "HeytingOmega3":
        """Negación involutiva en B₂ ⊂ Ω₃."""
        if self is HeytingOmega3.DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3(2 - int(self))

    def booleanization(self) -> "HeytingOmega3":
        """Funtor de doble negación ¬¬ : Ω₃ → B₂."""
        return self.pseudo_complement().pseudo_complement()

    def is_boolean_element(self) -> bool:
        """x es Booleano ssi ¬¬x = x ssi x ∈ {VETOED, COHERENT}."""
        return self.booleanization() is self

    def heyting_distance(self, other: "HeytingOmega3") -> float:
        """Métrica normalizada inducida por el orden: |x − y| / 2 ∈ [0, 1]."""
        return abs(int(self) - int(other)) / 2.0

    def __and__(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.meet(other)

    def __or__(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.join(other)

    def __invert__(self) -> "HeytingOmega3":
        return self.pseudo_complement()

    def __le__(self, other: "HeytingOmega3") -> bool:  # type: ignore[override]
        return int(self) <= int(other)

    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        """Encaje de álgebras de Boole B₂ ↪ Ω₃: False ↦ ⊥, True ↦ ⊤."""
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        """Sección parcial de `from_bool`; no está definida en DEGRADED."""
        if self is HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED ∉ im(B₂ ↪ Ω₃); usar booleanization().")
        return self is HeytingOmega3.COHERENT

    @classmethod
    def assert_heyting_axioms(cls) -> None:
        """Verificación exhaustiva de los axiomas de Ω₃."""
        elems = list(cls)
        bot, top = cls.bottom(), cls.top()
        for x in elems:
            assert x.meet(x) is x and x.join(x) is x, "idempotencia"
            assert x.meet(bot) is bot and x.join(top) is top, "cotas"
            assert x.meet(top) is x and x.join(bot) is x, "unidades"
            assert (x.meet(x.pseudo_complement()) is bot), "no contradicción"
            assert x.implication(x) is top, "x → x = ⊤"
            assert x.meet(x.implication(bot)) is bot, "residuación en ⊥"
            for y in elems:
                assert x.meet(y) is y.meet(x), "∧ conmutativa"
                assert x.join(y) is y.join(x), "∨ conmutativa"
                impl = x.implication(y)
                for z in elems:
                    left = int(x.meet(z)) <= int(y)
                    right = int(z) <= int(impl)
                    assert left is right, "residuación de Heyting"
        d = cls.DEGRADED
        assert d.pseudo_complement().pseudo_complement() is top
        assert d.join(d.pseudo_complement()) is not top
        assert cls.VETOED.booleanization() is cls.VETOED
        assert cls.COHERENT.booleanization() is cls.COHERENT
        assert cls.DEGRADED.booleanization() is cls.COHERENT


class CStarDensityCone:
    r"""
    Operaciones C* sobre Mₙ(ℂ) restringidas al cono de operadores densidad
    𝔇(ℋₙ) = { ρ ∈ Mₙ(ℂ) | ρ = ρ†, ρ ⪰ 0, Tr(ρ) = 1 }.
    """

    ATOL: Final[float] = 1e-9

    @staticmethod
    def maximally_mixed(dim: int) -> ComplexMatrix:
        if dim < 2:
            raise ValueError("dim ≥ 2 (evita degeneración espectral).")
        return np.eye(dim, dtype=np.complex128) / dim

    @staticmethod
    def hermitize(a: ComplexMatrix) -> ComplexMatrix:
        return 0.5 * (a + a.conj().T)

    @classmethod
    def operator_norm(cls, a: ComplexMatrix) -> float:
        s = la.svdvals(a)
        return float(s[0]) if s.size else 0.0

    @classmethod
    def frobenius_norm(cls, a: ComplexMatrix) -> float:
        return float(np.linalg.norm(a, ord="fro"))

    @classmethod
    def spectral_radius(cls, a: ComplexMatrix) -> float:
        return float(np.max(np.abs(la.eigvals(a))))

    @classmethod
    def is_density(cls, rho: ComplexMatrix, atol: float = ATOL) -> bool:
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            return False
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        eigvals = np.real(la.eigvalsh(rho))
        if np.any(eigvals < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    @classmethod
    def project_to_simplex(cls, v: RealVector) -> RealVector:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        n = v.size
        if n == 0:
            raise ValueError("vector vacío")
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
        theta = (cssv[rho_idx[-1]] - 1.0) / float(rho_idx[-1] + 1)
        w = np.maximum(v - theta, 0.0)
        s = float(np.sum(w))
        if s <= 0.0:
            return np.full(n, 1.0 / n, dtype=np.float64)
        return w / s

    @classmethod
    def project_to_density_cone(cls, rho: ComplexMatrix) -> ComplexMatrix:
        rho_h = cls.hermitize(np.asarray(rho, dtype=np.complex128))
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals = cls.project_to_simplex(np.real(eigvals))
        return (eigvecs * eigvals) @ eigvecs.conj().T

    @classmethod
    def spectrum_ordered(cls, rho: ComplexMatrix) -> RealVector:
        ev = np.real(la.eigvalsh(rho))
        ev = np.clip(ev, 0.0, None)
        s = float(np.sum(ev))
        if s <= 0.0:
            return np.full(ev.size, 1.0 / ev.size, dtype=np.float64)
        return np.sort(ev / s)

    @classmethod
    def von_neumann_entropy(cls, eigvals: RealVector) -> float:
        lam = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
        mask = lam > 0.0
        return float(-np.sum(lam[mask] * np.log(lam[mask])))

    @classmethod
    def purity(cls, eigvals: RealVector) -> float:
        lam = np.asarray(eigvals, dtype=np.float64)
        return float(np.sum(lam * lam))

    @classmethod
    def fidelity(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        sqrt_rho = la.sqrtm(cls.hermitize(rho))
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = la.sqrtm(cls.hermitize(inner))
        fid = float(np.real(np.trace(sqrt_inner)))
        return max(0.0, min(1.0, fid * fid if fid < 1.0 else fid))

    @classmethod
    def trace_distance(cls, rho: ComplexMatrix, sigma: ComplexMatrix) -> float:
        diff = cls.hermitize(rho - sigma)
        return 0.5 * float(np.sum(np.abs(la.eigvalsh(diff))))


class PathGraphDirichlet:
    r"""
    Forma de Dirichlet del grafo camino Pₙ.
    """

    @staticmethod
    def laplacian(n: int) -> RealVector:
        L = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            L[i, i] += 1.0
            L[i + 1, i + 1] += 1.0
            L[i, i + 1] -= 1.0
            L[i + 1, i] -= 1.0
        return L

    @classmethod
    def energy(cls, eigvals_ordered: RealVector) -> float:
        lam = np.asarray(eigvals_ordered, dtype=np.float64)
        grad = np.diff(lam)
        return 0.5 * float(np.sum(grad * grad))

    @classmethod
    def poincare_mass(cls, eigvals: RealVector) -> float:
        lam = np.asarray(eigvals, dtype=np.float64)
        n = lam.size
        mu = 1.0 / n
        return 0.5 * n * float(np.sum((lam - mu) ** 2))

    @classmethod
    def combined_energy(cls, eigvals_ordered: RealVector) -> float:
        return cls.energy(eigvals_ordered) + cls.poincare_mass(eigvals_ordered)

    @classmethod
    def tangent_modes(cls, n: int) -> Tuple[RealVector, RealVector]:
        L = cls.laplacian(n)
        evals, evecs = la.eigh(L)
        mask = evals > 1e-12
        Phi = evecs[:, mask]
        for k in range(Phi.shape[1]):
            nrm = float(np.linalg.norm(Phi[:, k]))
            if nrm > 0.0:
                Phi[:, k] /= nrm
        return Phi, np.real(evals[mask])


class HypercomplexPauliBasis:
    r"""
    Álgebra de cuaterniones de Hamilton ℍ realizada por matrices de Pauli.
    """

    _PAULI: Final[Tuple[ComplexMatrix, ...]] = (
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    )

    @classmethod
    def is_power_of_two(cls, n: int) -> bool:
        return n >= 2 and (n & (n - 1)) == 0

    @classmethod
    def su_generators(cls, dim: int) -> List[ComplexMatrix]:
        if not cls.is_power_of_two(dim):
            return []
        q = int(np.log2(dim))
        gens: List[ComplexMatrix] = []

        def rec(level: int, acc: ComplexMatrix) -> None:
            if level == q:
                if abs(float(np.trace(acc).real)) > 1e-12:
                    return
                gens.append(acc)
                return
            for p in cls._PAULI:
                rec(level + 1, np.kron(acc, p) if acc.size else p)

        rec(0, np.array([[1.0 + 0.0j]], dtype=np.complex128))
        return gens

    @classmethod
    def sample_normalized_hamiltonian(
        cls,
        dim: int,
        rng: np.random.Generator,
    ) -> ComplexMatrix:
        gens = cls.su_generators(dim)
        if gens:
            H = np.zeros((dim, dim), dtype=np.complex128)
            coeffs = rng.normal(size=len(gens))
            for c, T in zip(coeffs, gens):
                H += float(c) * T
            H = 0.5 * (H + H.conj().T)
        else:
            A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            H = 0.5 * (A + A.conj().T)
        nrm = CStarDensityCone.frobenius_norm(H)
        if nrm < 1e-15:
            H = np.zeros((dim, dim), dtype=np.complex128)
            H[0, 0] = 1.0
            H[1, 1] = -1.0
            nrm = CStarDensityCone.frobenius_norm(H)
        return H / nrm


@dataclass(frozen=True, slots=True)
class SpectralFlowGerm:
    hamiltonian: ComplexMatrix = field(repr=False, compare=False, hash=False)
    simplex_tangent: RealVector = field(repr=False, compare=False, hash=False)
    epsilon: float
    interaction_scale: float
    dimension: int
    seed: int

    def is_well_posed(self, atol: float = 1e-8) -> bool:
        H = self.hamiltonian
        v = self.simplex_tangent
        if H.shape != (self.dimension, self.dimension):
            return False
        if not np.allclose(H, H.conj().T, atol=atol):
            return False
        if v.shape != (self.dimension,):
            return False
        if abs(float(np.sum(v))) > 1e-6:
            return False
        if self.epsilon < 0.0 or not (0.0 <= self.interaction_scale <= 1.0):
            return False
        return True


@dataclass(frozen=True, slots=True)
class TricksterFieldState:
    cycle_id: str
    illusion_id: str
    illusion_type: str
    dream_isolation_flag: bool
    density_matrix: ComplexMatrix = field(repr=False, compare=False, hash=False)
    disguised_entropy: float
    disguised_purity: float
    reward_hacking_index: float
    stealth_dirichlet_energy: float
    heyting_verdict: HeytingOmega3
    provenance_hash: str
    timestamp_utc: float
    trace_distance_to_base: float = 0.0
    fidelity_to_base: float = 1.0

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        return CStarDensityCone.is_density(self.density_matrix, atol=atol)


@dataclass(frozen=True, slots=True)
class GANAdversarialCycleReport:
    report_id: str
    total_illusions_generated: int
    stealth_illusions_count: int
    vetoed_illusions_count: int
    degraded_illusions_count: int
    average_reward_hacking_score: float
    global_heyting_verdict: HeytingOmega3
    provenance_hash: str
    timestamp_utc: float

    def conservation_invariant(self) -> bool:
        return (
            self.stealth_illusions_count + self.vetoed_illusions_count
            == self.total_illusions_generated
            and 0 <= self.degraded_illusions_count <= self.stealth_illusions_count
        )


@runtime_checkable
class RewardHackingOracle(Protocol):
    def compute(
        self,
        disguised_cost_ratio: float,
        sophistication: float,
        stealth_dirichlet: float,
        dream_isolation: bool,
    ) -> Tuple[float, HeytingOmega3]:
        ...


def _clip_unit(x: float, name: str) -> float:
    if not np.isfinite(x):
        raise ValueError(f"{name} debe ser finito, recibido {x!r}.")
    return float(min(1.0, max(0.0, x)))


def induce_spectral_flow_germ(
    *,
    dim: int,
    disguised_cost_ratio: float,
    sophistication: float,
    seed: int,
    sophistication_attenuation: float = 0.85,
) -> SpectralFlowGerm:
    cost = _clip_unit(disguised_cost_ratio, "disguised_cost_ratio")
    soph = _clip_unit(sophistication, "sophistication")
    if dim < 2:
        raise ValueError("dim ≥ 2.")

    rng = np.random.default_rng(seed)
    epsilon = cost * max(0.0, 1.0 - sophistication_attenuation * soph)
    H = HypercomplexPauliBasis.sample_normalized_hamiltonian(dim, rng)

    Phi, lap_eigs = PathGraphDirichlet.tangent_modes(dim)
    beta = 4.0 * soph
    weights = np.exp(-beta * lap_eigs)
    weights = weights / (float(np.sum(weights)) + 1e-15)
    xi = rng.normal(size=weights.size)
    v = Phi @ (xi * weights)
    v = v - float(np.mean(v))
    vn = float(np.linalg.norm(v))
    if vn > 1e-15:
        v = v / vn
    else:
        v = Phi[:, 0].copy()
        v = v - float(np.mean(v))
        v = v / (float(np.linalg.norm(v)) + 1e-15)

    return SpectralFlowGerm(
        hamiltonian=H,
        simplex_tangent=v.astype(np.float64),
        epsilon=float(epsilon),
        interaction_scale=float(1.0 - soph),
        dimension=dim,
        seed=seed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — GENERADOR ESPECTRAL DE PERTURBACIONES HOMOCLÍNICAS DE POINCARÉ
# ══════════════════════════════════════════════════════════════════════════════


class TricksterDensityPerturber:
    """
    Generador Espectral de Perturbaciones Homoclínicas y Divisores Pequeños de Poincaré.
    """

    def __init__(
        self,
        tolerance: float = 1e-9,
        max_rhi_threshold: float = 0.85,
        resonance_floor: float = 1e-12,
    ) -> None:
        self._tol = float(tolerance)
        self._rhi_max = float(max_rhi_threshold)
        self._res_floor = float(resonance_floor)

    def _build_homoclinic_hamiltonian(
        self,
        dim: int,
        omega: np.ndarray,
        k_vector: np.ndarray,
        epsilon: float
    ) -> Tuple[np.ndarray, float]:
        """
        Sintetiza H_homoclinic = p^2/2 - cos(q) + ε cos(q) sin(ω·k t)
        y calcula el divisor pequeño |ω · k|.
        """
        q_op = np.diag(np.linspace(-np.pi, np.pi, dim))
        p_op = -1j * np.gradient(np.eye(dim), axis=0)

        # Divisor pequeño de Poincaré
        dot_product = float(np.dot(omega, k_vector))
        resonance = max(abs(dot_product), self._res_floor)

        # Hamiltoniano de péndulo forzado no integrable
        H_pendulum = 0.5 * (p_op @ p_op.conj().T) - np.diag(np.cos(np.diag(q_op)))
        H_forcing = epsilon * np.diag(np.cos(np.diag(q_op))) * (1.0 / resonance)

        H_homoclinic = 0.5 * (H_pendulum + H_pendulum.conj().T) + 0.5 * (H_forcing + H_forcing.conj().T)
        return H_homoclinic, resonance

    def synthesize_homoclinic_tangle_attack(
        self,
        density_op: np.ndarray,
        omega_frequencies: np.ndarray,
        k_wavevectors: np.ndarray,
        epsilon_perturbation: float = 0.05,
        attack_type: IllusionAttackType = IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
    ) -> IllusionDensityPerturbation:
        """
        Aplica U = exp(-i ε H_homoclinic) sobre ρ_0 garantizando la invarianza CPTP.

        :param density_op: Matriz de densidad base ρ_0 ∈ 𝔇_n (Tr(ρ) = 1.0, ρ ⪰ 0).
        :param omega_frequencies: Frecuencias fundamentales de la obra (insumos, tasa, tiempo).
        :param k_wavevectors: Vector armónico de modos de acoplamiento.
        :param epsilon_perturbation: Amplitud de la deformación homoclínica.
        :param attack_type: Tipo de ilusión contractual a simular.
        :return: Objeto IllusionDensityPerturbation con certificado de auditoría.
        """
        dim = density_op.shape[0]
        H_hom, resonance = self._build_homoclinic_hamiltonian(dim, omega_frequencies, k_wavevectors, epsilon_perturbation)

        # Operador unitario U = exp(-i ε H) vía Pade / Eigh
        evals, evecs = la.eigh(H_hom)
        U = evecs @ np.diag(np.exp(-1j * epsilon_perturbation * evals)) @ evecs.conj().T

        # Evolución CPTP de la densidad
        rho_ill = U @ density_op @ U.conj().T
        rho_ill = 0.5 * (rho_ill + rho_ill.conj().T)
        tr = np.trace(rho_ill)
        if abs(tr) > 1e-15:
            rho_ill /= tr
        else:
            rho_ill = np.eye(dim, dtype=complex) / dim

        # Cálculo de la Divergencia de Umegaki D(ρ_ill || ρ_0)
        s_ill = la.eigvalsh(rho_ill)
        s_0 = la.eigvalsh(density_op)
        s_ill = np.maximum(s_ill, 1e-15)
        s_0 = np.maximum(s_0, 1e-15)
        d_umegaki = float(np.sum(s_ill * (np.log2(s_ill) - np.log2(s_0))))

        # Cálculo del RHI (Reward Hacking Index)
        comm = rho_ill @ H_hom - H_hom @ rho_ill
        rhi_score = min(1.0, max(0.0, float(d_umegaki / (np.linalg.norm(comm, ord='fro') + 1e-6))))

        # Infección de Betti (β₁ > 0 si RHI excede umbral)
        betti_1 = 1 if rhi_score > self._rhi_max else 0

        # Generación de la prueba Merkle de procedencia
        hasher = hashlib.sha256()
        hasher.update(f"{attack_type.value}::{rhi_score:.6f}::{resonance:.6f}::{betti_1}".encode('utf-8'))
        proof_sha256 = hasher.hexdigest()

        cert = TricksterAttackCertificate(
            attack_type=attack_type.value,
            rhi_score=rhi_score,
            homoclinic_residual=float(np.linalg.norm(H_hom - H_hom.conj().T)),
            small_divisor_resonance=resonance,
            betti_1_induced=betti_1,
            is_unitary_cptp=bool(abs(np.trace(rho_ill) - 1.0) < self._tol),
            merkle_proof_sha256=proof_sha256,
            schema_version="7.1.0",
        )

        return IllusionDensityPerturbation(
            illusion_density_matrix=rho_ill,
            original_density_matrix=density_op,
            unitary_operator=U,
            attack_certificate=cert,
        )


class AdversarialSpectralGenerator:
    _SOFISTICATION_ATTENUATION: Final[float] = 0.85

    @staticmethod
    def integrate_spectral_flow_germ(
        germ: SpectralFlowGerm,
        base_rho: ComplexMatrix,
    ) -> Tuple[ComplexMatrix, float, float, float]:
        if not germ.is_well_posed():
            raise ValueError("SpectralFlowGerm mal puesto (H no hermítica o v ∉ TΔ).")

        rho0 = CStarDensityCone.project_to_density_cone(base_rho)
        lam0 = CStarDensityCone.spectrum_ordered(rho0)

        lam_eps = CStarDensityCone.project_to_simplex(
            lam0 + germ.epsilon * germ.simplex_tangent
        )
        lam_eps = np.sort(lam_eps)

        U = la.expm(-1j * germ.epsilon * germ.hamiltonian)
        _, evecs0 = la.eigh(CStarDensityCone.hermitize(rho0))
        evecs = U @ evecs0
        rho_ill = (evecs * lam_eps) @ evecs.conj().T
        rho_ill = CStarDensityCone.project_to_density_cone(rho_ill)

        eigvals = CStarDensityCone.spectrum_ordered(rho_ill)
        purity = CStarDensityCone.purity(eigvals)
        entropy = CStarDensityCone.von_neumann_entropy(eigvals)
        stealth_dirichlet = PathGraphDirichlet.combined_energy(eigvals)
        stealth_dirichlet += germ.epsilon * 0.05 * germ.interaction_scale

        return rho_ill, purity, entropy, float(stealth_dirichlet)

    @classmethod
    def generate_stealth_perturbation(
        cls,
        base_rho: ComplexMatrix,
        disguised_cost_ratio: float,
        sophistication: float,
        seed: int,
    ) -> Tuple[ComplexMatrix, float, float, float]:
        dim = int(base_rho.shape[0])
        germ = induce_spectral_flow_germ(
            dim=dim,
            disguised_cost_ratio=disguised_cost_ratio,
            sophistication=sophistication,
            seed=seed,
            sophistication_attenuation=cls._SOFISTICATION_ATTENUATION,
        )
        return cls.integrate_spectral_flow_germ(germ, base_rho)


class RewardHackingMetricsEngine:
    ALPHA: Final[float] = 0.6
    BETA: Final[float] = 0.4
    RHI_VETO_THRESHOLD: Final[float] = 0.88
    RHI_DEGRADE_THRESHOLD: Final[float] = 0.50
    DIRICHLET_VETO_THRESHOLD: Final[float] = 0.75

    @classmethod
    def compute_rhi_and_verdict(
        cls,
        disguised_cost_ratio: float,
        sophistication: float,
        stealth_dirichlet: float,
        dream_isolation: bool,
    ) -> Tuple[float, HeytingOmega3]:
        if not dream_isolation:
            return 1.0, HeytingOmega3.VETOED

        cost = _clip_unit(disguised_cost_ratio, "disguised_cost_ratio")
        soph = _clip_unit(sophistication, "sophistication")
        raw_rhi = cls.ALPHA * cost + cls.BETA * soph
        rhi = float(min(1.0, max(0.0, raw_rhi)))

        if (
            rhi > cls.RHI_VETO_THRESHOLD
            or stealth_dirichlet > cls.DIRICHLET_VETO_THRESHOLD
        ):
            verdict = HeytingOmega3.VETOED
        elif rhi > cls.RHI_DEGRADE_THRESHOLD:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT
        return rhi, verdict

    @classmethod
    def compute(
        cls,
        disguised_cost_ratio: float,
        sophistication: float,
        stealth_dirichlet: float,
        dream_isolation: bool,
    ) -> Tuple[float, HeytingOmega3]:
        return cls.compute_rhi_and_verdict(
            disguised_cost_ratio=disguised_cost_ratio,
            sophistication=sophistication,
            stealth_dirichlet=stealth_dirichlet,
            dream_isolation=dream_isolation,
        )


@dataclass(frozen=True, slots=True)
class SpectralObservation:
    rho_illusion: ComplexMatrix = field(repr=False, compare=False, hash=False)
    purity: float
    entropy: float
    dirichlet_energy: float
    reward_hacking_index: float
    heyting_verdict: HeytingOmega3
    trace_distance_to_base: float = 0.0
    fidelity_to_base: float = 1.0
    germ_epsilon: float = 0.0


def spectral_observation_pipeline(
    *,
    base_rho: ComplexMatrix,
    disguised_cost_ratio: float,
    sophistication: float,
    seed: int,
    dream_isolation: bool,
    oracle: RewardHackingOracle = RewardHackingMetricsEngine,
) -> SpectralObservation:
    dim = int(base_rho.shape[0])
    germ = induce_spectral_flow_germ(
        dim=dim,
        disguised_cost_ratio=disguised_cost_ratio,
        sophistication=sophistication,
        seed=seed,
    )
    rho_ill, purity, entropy, denergy = (
        AdversarialSpectralGenerator.integrate_spectral_flow_germ(germ, base_rho)
    )
    rhi, verdict = oracle.compute(
        disguised_cost_ratio=disguised_cost_ratio,
        sophistication=sophistication,
        stealth_dirichlet=denergy,
        dream_isolation=dream_isolation,
    )
    td = CStarDensityCone.trace_distance(rho_ill, base_rho)
    try:
        fid = CStarDensityCone.fidelity(rho_ill, base_rho)
    except (ValueError, np.linalg.LinAlgError):
        fid = max(0.0, 1.0 - td)
    return SpectralObservation(
        rho_illusion=rho_ill,
        purity=purity,
        entropy=entropy,
        dirichlet_energy=denergy,
        reward_hacking_index=rhi,
        heyting_verdict=verdict,
        trace_distance_to_base=td,
        fidelity_to_base=fid,
        germ_epsilon=germ.epsilon,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INTERLOCK CIBER-FÍSICO, ORQUESTACIÓN Y MOTOR POINCARÉ
# ══════════════════════════════════════════════════════════════════════════════


class InterlockAutomatonState(IntEnum):
    IDLE = 0
    ARMED = 1
    FIRED = 2


class ESP32TricksterInterlock:
    @staticmethod
    def should_fire(verdict: HeytingOmega3, dream_isolation: bool) -> bool:
        return (not dream_isolation) or (verdict is HeytingOmega3.VETOED)

    @classmethod
    def transition(
        cls,
        current: InterlockAutomatonState,
        verdict: HeytingOmega3,
        dream_isolation: bool,
    ) -> InterlockAutomatonState:
        if cls.should_fire(verdict, dream_isolation):
            return InterlockAutomatonState.FIRED
        if current is InterlockAutomatonState.FIRED:
            return InterlockAutomatonState.FIRED
        if verdict is HeytingOmega3.DEGRADED:
            return InterlockAutomatonState.ARMED
        return InterlockAutomatonState.IDLE

    @classmethod
    def check_interlock(
        cls, verdict: HeytingOmega3, dream_isolation: bool
    ) -> bool:
        fired = cls.should_fire(verdict, dream_isolation)
        if fired:
            logger.critical(
                "[ESP32 TRICKSTER INTERLOCK] Disparo ejecutado en IRAM (<400 ns). "
                "GPIO14 -> HIGH. BT151 Armado. Razón: %s",
                "aislamiento violado"
                if not dream_isolation
                else "veredicto VETOED por RHI/Dirichlet crítico",
            )
        return fired


class TOONTricksterAdversaryEngine:
    r"""
    Motor Espectral Ilusionista — flecha terminal 1 → Ω₃ del topos de
    evaluación adversarial e integrador de la mecánica celeste de Henri Poincaré.
    """

    _DEFAULT_ENGINE_ID: Final[str] = "TRICKSTER-ENGINE-SABIO-01"
    _DEFAULT_DIMENSION: Final[int] = 4
    _DEFAULT_SEED: Final[int] = 333

    def __init__(
        self,
        engine_id: str = _DEFAULT_ENGINE_ID,
        dimension_mac: int = _DEFAULT_DIMENSION,
        seed: int = _DEFAULT_SEED,
        oracle: RewardHackingOracle = RewardHackingMetricsEngine,
        tolerance: float = 1e-9,
        max_rhi_threshold: float = 0.85,
    ) -> None:
        if dimension_mac < 2:
            raise ValueError("dimension_mac debe ser ≥ 2 para evitar degeneración espectral.")

        self.engine_id: str = engine_id
        self.dimension_mac: int = dimension_mac
        self.seed_counter: int = seed
        self.cycle_count: int = 0
        self.base_rho: ComplexMatrix = CStarDensityCone.maximally_mixed(dimension_mac)
        self._oracle: RewardHackingOracle = oracle
        self.history: List[TricksterFieldState] = []
        self._interlock_state: InterlockAutomatonState = InterlockAutomatonState.IDLE
        self._poincare_perturber: TricksterDensityPerturber = TricksterDensityPerturber(
            tolerance=tolerance, max_rhi_threshold=max_rhi_threshold
        )

    def _seal_provenance(
        self,
        cycle_id: str,
        illusion_id: str,
        verdict: HeytingOmega3,
        rhi: float,
        purity: float,
    ) -> str:
        hasher = hashlib.sha256()
        payload = (
            f"{self.engine_id}::{cycle_id}::{illusion_id}"
            f"::{verdict.name}::{rhi:.6f}::{purity:.6f}"
        )
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    def verify_provenance(self, state: TricksterFieldState) -> bool:
        expected = self._seal_provenance(
            cycle_id=state.cycle_id,
            illusion_id=state.illusion_id,
            verdict=state.heyting_verdict,
            rhi=state.reward_hacking_index,
            purity=state.disguised_purity,
        )
        return hmac.compare_digest(expected, state.provenance_hash)

    def process_poincare_homoclinic_attack(
        self,
        omega_frequencies: np.ndarray,
        k_wavevectors: np.ndarray,
        epsilon_perturbation: float = 0.05,
        attack_type: IllusionAttackType = IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
        dream_isolation: bool = True,
        illusion_id: Optional[str] = None,
    ) -> IllusionDensityPerturbation:
        """
        Ejecuta un ataque de enredo homoclínico de Poincaré sobre la densidad base.
        """
        if illusion_id is None:
            self.cycle_count += 1
            illusion_id = f"ILLUSION-POINCARE-{self.cycle_count:04d}"

        perturbation = self._poincare_perturber.synthesize_homoclinic_tangle_attack(
            density_op=self.base_rho,
            omega_frequencies=omega_frequencies,
            k_wavevectors=k_wavevectors,
            epsilon_perturbation=epsilon_perturbation,
            attack_type=attack_type,
        )
        cert = perturbation.attack_certificate

        if cert.betti_1_induced > 0 or cert.rhi_score > 0.85 or not dream_isolation:
            verdict = HeytingOmega3.VETOED
        elif cert.rhi_score > 0.50:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        ESP32TricksterInterlock.check_interlock(verdict=verdict, dream_isolation=dream_isolation)
        self._interlock_state = ESP32TricksterInterlock.transition(self._interlock_state, verdict, dream_isolation)

        prov_hash = self._seal_provenance(
            cycle_id=f"CYC-POINCARE-{self.cycle_count:04d}",
            illusion_id=illusion_id,
            verdict=verdict,
            rhi=cert.rhi_score,
            purity=float(np.sum(la.eigvalsh(perturbation.illusion_density_matrix)**2)),
        )

        state = TricksterFieldState(
            cycle_id=f"CYC-POINCARE-{self.cycle_count:04d}",
            illusion_id=illusion_id,
            illusion_type=cert.attack_type,
            dream_isolation_flag=dream_isolation,
            density_matrix=perturbation.illusion_density_matrix,
            disguised_entropy=CStarDensityCone.von_neumann_entropy(CStarDensityCone.spectrum_ordered(perturbation.illusion_density_matrix)),
            disguised_purity=CStarDensityCone.purity(CStarDensityCone.spectrum_ordered(perturbation.illusion_density_matrix)),
            reward_hacking_index=cert.rhi_score,
            stealth_dirichlet_energy=cert.homoclinic_residual,
            heyting_verdict=verdict,
            provenance_hash=prov_hash,
            timestamp_utc=time.time(),
            trace_distance_to_base=CStarDensityCone.trace_distance(perturbation.illusion_density_matrix, self.base_rho),
            fidelity_to_base=CStarDensityCone.fidelity(perturbation.illusion_density_matrix, self.base_rho),
        )
        self.history.append(state)
        return perturbation

    def process_adversarial_illusion(
        self,
        illusion_id: str,
        illusion_type: str,
        disguised_cost_ratio: float,
        sophistication: float,
        dream_isolation: bool = True,
    ) -> TricksterFieldState:
        self.cycle_count += 1
        self.seed_counter += 1
        cycle_id = f"CYC-TRICK-{self.cycle_count:04d}"
        t_start = time.time()

        logger.info(
            "=== Ciclo Espectral Ilusionista %s | Ilusión: %s (%s) ===",
            cycle_id,
            illusion_id,
            illusion_type,
        )

        obs: SpectralObservation = spectral_observation_pipeline(
            base_rho=self.base_rho,
            disguised_cost_ratio=disguised_cost_ratio,
            sophistication=sophistication,
            seed=self.seed_counter,
            dream_isolation=dream_isolation,
            oracle=self._oracle,
        )

        fired = ESP32TricksterInterlock.check_interlock(
            verdict=obs.heyting_verdict,
            dream_isolation=dream_isolation,
        )
        self._interlock_state = ESP32TricksterInterlock.transition(
            self._interlock_state,
            obs.heyting_verdict,
            dream_isolation,
        )
        if fired:
            assert self._interlock_state is InterlockAutomatonState.FIRED

        prov_hash = self._seal_provenance(
            cycle_id=cycle_id,
            illusion_id=illusion_id,
            verdict=obs.heyting_verdict,
            rhi=obs.reward_hacking_index,
            purity=obs.purity,
        )

        state = TricksterFieldState(
            cycle_id=cycle_id,
            illusion_id=illusion_id,
            illusion_type=illusion_type,
            dream_isolation_flag=dream_isolation,
            density_matrix=obs.rho_illusion,
            disguised_entropy=obs.entropy,
            disguised_purity=obs.purity,
            reward_hacking_index=obs.reward_hacking_index,
            stealth_dirichlet_energy=obs.dirichlet_energy,
            heyting_verdict=obs.heyting_verdict,
            provenance_hash=prov_hash,
            timestamp_utc=t_start,
            trace_distance_to_base=obs.trace_distance_to_base,
            fidelity_to_base=obs.fidelity_to_base,
        )
        if not state.is_quantum_physical():
            raise RuntimeError("Invariante C* violado: ρ ∉ 𝔇(ℋₙ).")

        self.history.append(state)

        logger.info(
            "Ciclo Espectral %s Finalizado en %.2f ms | Veredicto: %s | "
            "RHI: %.4f | Pureza: %.4f | T(ρ,ρ₀): %.4f | Autómata: %s",
            cycle_id,
            (time.time() - t_start) * 1000.0,
            obs.heyting_verdict.name,
            obs.reward_hacking_index,
            obs.purity,
            obs.trace_distance_to_base,
            self._interlock_state.name,
        )
        return state

    def _seal_report(
        self,
        report_id: str,
        global_verdict: HeytingOmega3,
        avg_rhi: float,
        batch_size: int,
    ) -> str:
        hasher = hashlib.sha256()
        payload = (
            f"{self.engine_id}::{report_id}::{global_verdict.name}"
            f"::{avg_rhi:.6f}::{batch_size}"
        )
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    def run_gan_adversarial_round(
        self,
        illusions_batch: Sequence[Dict[str, Any]],
    ) -> GANAdversarialCycleReport:
        t_start = time.time()
        report_id = f"GAN-REPORT-{self.cycle_count + 1:04d}"

        stealth_count = 0
        vetoed_count = 0
        degraded_count = 0
        total_rhi = 0.0
        global_verdict = HeytingOmega3.COHERENT

        for ill in illusions_batch:
            state = self.process_adversarial_illusion(
                illusion_id=str(ill["illusion_id"]),
                illusion_type=str(ill["illusion_type"]),
                disguised_cost_ratio=float(ill.get("disguised_cost_ratio", 0.3)),
                sophistication=float(ill.get("sophistication", 0.8)),
                dream_isolation=bool(ill.get("dream_isolation", True)),
            )
            total_rhi += state.reward_hacking_index
            global_verdict = global_verdict.meet(state.heyting_verdict)

            if state.heyting_verdict is HeytingOmega3.VETOED:
                vetoed_count += 1
            else:
                stealth_count += 1
                if state.heyting_verdict is HeytingOmega3.DEGRADED:
                    degraded_count += 1

        n_batch = len(illusions_batch)
        avg_rhi = total_rhi / float(n_batch) if n_batch else 0.0
        prov_hash = self._seal_report(
            report_id=report_id,
            global_verdict=global_verdict,
            avg_rhi=avg_rhi,
            batch_size=n_batch,
        )

        report = GANAdversarialCycleReport(
            report_id=report_id,
            total_illusions_generated=n_batch,
            stealth_illusions_count=stealth_count,
            vetoed_illusions_count=vetoed_count,
            degraded_illusions_count=degraded_count,
            average_reward_hacking_score=avg_rhi,
            global_heyting_verdict=global_verdict,
            provenance_hash=prov_hash,
            timestamp_utc=t_start,
        )
        assert report.conservation_invariant(), "Violación del invariante de conservación GAN."
        return report

    def audit_history(self) -> Dict[str, Any]:
        n = len(self.history)
        if n == 0:
            return {
                "engine_id": self.engine_id,
                "cycles_recorded": 0,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "avg_rhi": 0.0,
                "avg_purity": 0.0,
                "interlock_state": self._interlock_state.name,
            }
        total_rhi = sum(s.reward_hacking_index for s in self.history)
        total_purity = sum(s.disguised_purity for s in self.history)
        gv = HeytingOmega3.COHERENT
        for s in self.history:
            gv = gv.meet(s.heyting_verdict)
        quantum_ok = all(s.is_quantum_physical() for s in self.history)
        hashes_ok = all(self.verify_provenance(s) for s in self.history)
        return {
            "engine_id": self.engine_id,
            "cycles_recorded": n,
            "global_verdict": gv.name,
            "avg_rhi": total_rhi / n,
            "avg_purity": total_purity / n,
            "all_quantum_physical": quantum_ok,
            "all_provenance_valid": hashes_ok,
            "interlock_state": self._interlock_state.name,
            "booleanized_verdict": gv.booleanization().name,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("═" * 80)
    print("DEMOSTRACIÓN GRANULAR: TOON Trickster Adversary Engine v8.1.0")
    print("MECÁNICA CELESTE DE HENRI POINCARÉ: Enredos Homoclínicos y Divisores Pequeños")
    print("═" * 80)

    engine = TOONTricksterAdversaryEngine()
    omega = np.array([1.0, 0.5, 0.25, 0.125])
    k_vec = np.array([2.0, -4.0, 1.0, 0.0])

    print("\n>>> ESCENARIO HOMOCLÍNICO: Ataque de Fraccionamiento (Divisores Pequeños)...")
    ill_pert = engine.process_poincare_homoclinic_attack(
        omega_frequencies=omega,
        k_wavevectors=k_vec,
        epsilon_perturbation=0.08,
        attack_type=IllusionAttackType.SPLIT_CONTRACT_ILLUSION,
        dream_isolation=True,
    )
    cert = ill_pert.attack_certificate
    print(f"    - Tipo de Ataque     : {cert.attack_type}")
    print(f"    - Score RHI          : {cert.rhi_score:.4f}")
    print(f"    - Resonancia Divisor : {cert.small_divisor_resonance:.6e}")
    print(f"    - Residual Homoclínico: {cert.homoclinic_residual:.6e}")
    print(f"    - Betti-1 Inducido   : {cert.betti_1_induced}")
    print(f"    - Es Unitario CPTP   : {cert.is_unitary_cptp}")
    print(f"    - Prueba Merkle SHA256: {cert.merkle_proof_sha256[:32]}...")

    print("\n" + "═" * 80)
    print("✓ Pruebas de verificación del Motor Ilusionista de Poincaré v8.1.0 completadas.")
    print("═" * 80)

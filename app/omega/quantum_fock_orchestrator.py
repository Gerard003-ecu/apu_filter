# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Quantum Fock Orchestrator — Cámara Catadióptrica de Fock                  ║
║ Ruta     : app/omega/quantum_fock_orchestrator.py                                    ║
║ Versión  : 6.0.0-Hermite-Sectors-GKSL-CP-Spohn-KAM-RSI-L3-Bogoliubov-Bridge          ║
║ Función  : Espacio de Fock ℤ₂-graduado, colisionador catadióptrico, auditoría de     ║
║            integrabilidad celeste de Poincaré, semigrupo GKSL estructura-preservante ║
║            y Automejora Recursiva de Nivel 3 con certificado inmutable.              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA: TRES FASES ANIDADAS POR CONTINUACIÓN (composición de Kleisli)
──────────────────────────────────────────────────────────────────────────────────────
  Φ₁ : FockSpaceConfiguration ──► O  (InteractionOperators)
  Φ₂ : O ──────────────────────► K  (CatadioptricHamiltonian)
  Φ₃ : K ──────────────────────► Λ  (LindbladEvolutionResult)

  El último método formal de la Fase 1 (`get_interaction_operators`) recibe
  `continuation: Callable[[O], R]` y su cola es `return continuation(O)`, cuya
  continuación canónica es el primer método de la Fase 2
  (`_continue_from_phase1_operators`). Idénticamente el último método de la Fase 2
  (`get_catadioptric_hamiltonian`) continúa en `_continue_from_phase2_hamiltonian`.
  NO se usa herencia entre fases (la v5 heredaba y dejaba atributos sin inicializar):
  la composición es por objetos + contratos de tipo, con fachada ABI-compatible.

TEOREMAS Y SU ESTATUS EPISTÉMICO
──────────────────────────────────────────────────────────────────────────────────────
 T1 [TEOREMA] CCR truncada: con a_{n-1,n}=√n en ℂ^{M+1},
      [a,a†] = I − (M+1)|M⟩⟨M|.  Por tanto [Q,P] = i(I − (M+1)|M⟩⟨M|): la anomalía
      NO es un error, es la huella del truncamiento. Certificado operativo:
      |⟨[Q,P]⟩ − i| = (M+1)·⟨|M⟩⟨M|⟩  ⇒ cota de confianza vía población de frontera.
 T2 [TEOREMA] Jordan–Wigner: f_q = (⊗_{j<q}σ_z)⊗f satisface CAR exacta; los sectores
      bosónicos (pares) conmutan con los fermiónicos. Las relaciones se PRUEBAN en los
      factores locales y se elevan por funtorialidad de ⊗ (ver §1.6): coste
      O(B(M+1)³ + F·8) en lugar de O(B²d³).
 T3 [TEOREMA] Espectro exacto del modelo por defecto: [H_int, n_q]=0 y en cada sector
      H_s = Σ_k (G_k b_k + Ḡ_k b_k†) = Σ_k √2|G_k| Q_k (tras gauge de fase). Q_trunc es
      la matriz de Jacobi de los polinomios de Hermite ⇒ sus autovalores son los NODOS
      de Gauss–Hermite. Luego spec(H) = {Σ_k √2|G_k(s)| x_{i_k}} exactamente.
      CONSECUENCIA: el modelo por defecto es exactamente soluble ⇒ la estadística de
      Berry–Tabor/GOE NO es un diagnóstico de caos aquí. Se declara, no se disimula.
 T4 [TEOREMA] Flujo de Wegner: con η=[H_d,H_off] (antihermítico), dH/dl=[η,H] es
      iso-espectral y d‖H_off‖_F²/dl = −2‖η‖_F² ≤ 0, d Tr(H_d²)/dl = +2‖η‖_F².
      Implementado con conjugación de Cayley (unitaria exacta). El flujo es
      DIAGNÓSTICO/precondicionador: cambiar de base sin transformar ρ y L es un error.
 T5 [TEOREMA] GKSL: γ_k ≥ 0 (Kossakowski PSD) ⇒ semigrupo CPTP; Tr𝓛(ρ)≡0 y
      𝓛(ρ†)=𝓛(ρ)†. La condición Σ K†K=I es de MAPAS de Kraus, NO de jump operators.
 T6 [TEOREMA] Proyección euclídea sobre el espectralplex {ρ⪰0, Trρ=1}: por invariancia
      unitaria, la proyección espectral con proyección simplex de autovalores
      (Duchi et al. 2008) es la ÓPTIMA en Frobenius; clip+renormalizar NO lo es.
 T7 [TEOREMA] Spohn: si ρ_ss es punto fijo del semigrupo, D(ρ_t‖ρ_ss) es monótona
      decreciente (monotonía de la entropía relativa / data-processing) y
      σ = −Tr[𝓛(ρ)(ln ρ − ln ρ_ss)] = dS/dt + Tr[𝓛(ρ) ln ρ_ss] ≥ 0.
      σ es el Lyapunov CANÓNICO del lazo RSI-L3. No se clampa a 0: el signo audita
      la hipótesis (soporte y estacionariedad).
 T8 [TEOREMA] Kronecker–Dirichlet: ∀ε ∃ T ≤ ⌈1/ε⌉^d con max_i‖ω_i T/2π‖ ≤ ε
      (aproximación simultánea) ⇒ cota de tiempo de retorno de Poincaré para dinámica
      unitaria. En dinámica disipativa la recurrencia está OBSTRUIDA por T7.
 T9 [TEOREMA] Divisiones con coeficientes NEGATIVOS (Yoshida w₀<0) son inadmisibles
      para la parte disipativa: el semigrupo parabólico hacia atrás está mal planteado
      y el mapa resultante no es CP. Sólo composiciones de pesos positivos (Strang).
T10 [CONVENCIÓN] Ledger de aniquilación E=2m*c²: m* se deriva de una ENERGÍA real
      (ΔE = Tr[Hρ₀] − Tr[Hρ_t]), nunca de una entropía. La cota de Landauer
      E ≥ k_B T ΔS (nats) es el puente termodinámico correcto. Hash forense SHA-256
      encadenado (determinista; `hash()` de Python está aleatorizado por proceso).

CONVENCIÓN DE VECTORIZACIÓN: vec por COLUMNAS (order='F'):
      vec(AρB) = (Bᵀ ⊗ A) vec(ρ)
      𝓛 = −(i/ħ)(I⊗H − Hᵀ⊗I) + Σ γ_k [ L̄_k⊗L_k − ½ I⊗(L†L) − ½ (L†L)ᵀ⊗I ].
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, identity as sparse_identity, issparse, kron as skron
from scipy.sparse.linalg import LinearOperator, eigsh, expm, expm_multiply

# ── Dependencias estructurales del ecosistema (degradación grácil) ────────────────────
try:
    from app.core.mic_algebra import CategoricalState, Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Violación de invariante algebraico/topológico (veto duro)."""

    class Morphism:  # type: ignore[no-redef]
        """Morfismo mínimo standalone."""

    class CategoricalState:  # type: ignore[no-redef]
        """Estado categórico mínimo standalone."""

try:
    from app.core.inmune_system.metric_tensors import G_PHYSICS
except ImportError:
    G_PHYSICS: Dict[str, Any] = {}

try:
    from app.core.telemetry_schemas import ElectronCartridge, GammaPhoton, PositronCartridge
except ImportError:  # pragma: no cover
    @dataclass(frozen=True)
    class PositronCartridge:
        inertial_mass: float
        topological_spin: str
        homological_charge: int
        authorization_signature: str

    @dataclass(frozen=True)
    class GammaPhoton:
        annihilation_energy: float
        data_hash: str
        timestamp_entry: float
        authorization_signature: str

    @dataclass(frozen=True)
    class ElectronCartridge:
        pass

# ── Puente con el soberano anterior: Ω₃ y gobernanza CAR/CCR ─────────────────────────
try:
    from app.agents.omega.bogoliubov_agent import (
        BogoliubovAgent,
        HeytingVerdict,
        heyting_meet,
    )
except ImportError:  # pragma: no cover - Ω₃ local (cadena de Gödel de 3 elementos)
    BogoliubovAgent = None  # type: ignore[assignment]

    class HeytingVerdict(int):  # type: ignore[no-redef]
        COHERENT = 0
        DEGRADED = 1
        VETOED = 2

    def heyting_meet(a: Any, b: Any) -> Any:  # type: ignore[no-redef]
        return max(int(a), int(b))

logger = logging.getLogger("MIC.Omega.QuantumFockOrchestrator")
R = TypeVar("R")

__all__ = [
    "FockPoincareRSIConstants", "PauliExclusionViolationError", "FockSpaceOverflowError",
    "LindbladDissipationError", "HermiticityViolationError", "UnitarityViolationError",
    "Quasiparticle", "Boson", "Fermion", "RiemannianFocalBoson",
    "HouseholderReflectionFermion", "FockSpaceConfiguration", "AlgebraAuditReport",
    "TruncationCertificate", "InteractionOperators", "Phase1_FockSpaceBuilder",
    "SpectralStatistics", "DiophantineKAMReport", "WegnerFlowReport",
    "CatadioptricHamiltonian", "Phase2_CatadioptricCollider", "GKSLStructureReport",
    "SpohnReport", "RecurrenceReport", "LindbladEvolutionResult",
    "Phase3_LindbladDissipator", "RSIL3Hyperparameters", "RSIL3Report",
    "QuantumFockOrchestrator",
]


# ═════════════════════════════════════════════════════════════════════════════════════
# §A · CONSTANTES FÍSICAS (CODATA), PESOS DE COMPOSICIÓN Y UMBRALES
# ═════════════════════════════════════════════════════════════════════════════════════
class FockPoincareRSIConstants:
    r"""Constantes físicas exactas (CODATA-2018) y umbrales adimensionales del motor."""

    MACHINE_EPS: float = float(np.finfo(np.float64).eps)
    SPEED_OF_LIGHT_C: float = float(G_PHYSICS.get("c", 299_792_458.0))       # m/s exacto
    BOLTZMANN_K_B: float = float(G_PHYSICS.get("k_B", 1.380649e-23))         # J/K exacto
    REDUCED_PLANCK_HBAR: float = float(G_PHYSICS.get("hbar", 1.054571817e-34))
    FINE_STRUCTURE_ALPHA: float = 1.0 / 137.035999084
    GOLDEN_CONJUGATE_KAM: float = 0.5 * (math.sqrt(5.0) - 1.0)               # φ⁻¹ (noble)
    RSI_L3_MAX_DEPTH: int = 3
    RSI_BANACH_CONTRACTIVITY: float = GOLDEN_CONJUGATE_KAM
    # Referencias de estadística de espaciamientos (sólo válidas TRAS desymmetrizar)
    POISSON_INTEGRABLE_RATIO: float = 2.0 * math.log(2.0) - 1.0              # ≈0.386294
    GOE_CHAOTIC_RATIO: float = 0.5307
    GUE_CHAOTIC_RATIO: float = 0.6027
    # Composición simpléctica de 4º orden de Yoshida: SÓLO para partes reversible/Hamiltoniana
    _CBRT_2: float = 2.0 ** (1.0 / 3.0)
    YOSHIDA_W1: float = 1.0 / (2.0 - _CBRT_2)
    YOSHIDA_W0: float = -_CBRT_2 * YOSHIDA_W1
    YOSHIDA_WEIGHTS: Tuple[float, float, float] = (YOSHIDA_W1, YOSHIDA_W0, YOSHIDA_W1)
    # Presupuestos estructurales (evitan explosión silenciosa de memoria)
    MAX_HILBERT_DIM: int = 1 << 16
    SUPEROP_DIM_BUDGET: int = 4096          # d² ≤ budget ⇒ d ≤ 64
    DENSE_STEADY_DIM: int = 16
    ALGEBRA_TOL: float = 1e-10
    TRUNCATION_POP_TOL: float = 1e-8
    CP_NEGativity_REL: float = 1e-9


# ═════════════════════════════════════════════════════════════════════════════════════
# §B · EXCEPCIONES
# ═════════════════════════════════════════════════════════════════════════════════════
class PauliExclusionViolationError(TopologicalInvariantError):
    """CAR violada o n_q no idempotente (exclusión de Pauli comprometida)."""


class FockSpaceOverflowError(TopologicalInvariantError):
    """Dimensión fuera de presupuesto o población de frontera de Fock excesiva."""


class LindbladDissipationError(TopologicalInvariantError):
    """Violación estructural de CPTP: traza, positividad o hermiticidad."""


class HermiticityViolationError(TopologicalInvariantError):
    """Pérdida de autoadjunción H = H†."""


class UnitarityViolationError(TopologicalInvariantError):
    """Pérdida de isometría U†U = I."""


# ═════════════════════════════════════════════════════════════════════════════════════
# §C · CUASIPARTÍCULAS: CONTENIDO MATEMÁTICO REAL (Pin/Spin, Maslov, SPD)
# ═════════════════════════════════════════════════════════════════════════════════════
class Quasiparticle:
    """Identidad base en ℱ(ℋ)."""

    @property
    def quantum_state_hash(self) -> str:
        raise NotImplementedError


class Boson(Quasiparticle):
    """Espín entero: producto tensorial simétrico Sⁿ(ℋ_B)."""


class Fermion(Quasiparticle):
    """Espín semientero: álgebra exterior ⋀ⁿ(ℋ_F)."""


@dataclass(frozen=True)
class RiemannianFocalBoson(Boson):
    r"""
    Bosón focal riemanniano. `dielectric_tensor` debe ser SPD (ε ⪰ 0 exige medio
    pasivo); `wkb_maslov_index` ∈ ℤ cuenta cruces de cáustica (índice de Maslov μ,
    definido mod 4 para la fase de Gouy; se conserva la paridad μ mod 2 como invariante).
    """

    dielectric_tensor: NDArray[np.float64]
    spectral_cutoff_functor: int
    wkb_maslov_index: int

    def __post_init__(self) -> None:
        eps = np.asarray(self.dielectric_tensor, dtype=np.float64)
        if eps.ndim != 2 or eps.shape[0] != eps.shape[1]:
            raise TopologicalInvariantError("ε debe ser cuadrada")
        sym_def = float(la.norm(eps - eps.T, "fro") / max(la.norm(eps, "fro"), 1e-300))
        if sym_def > 1e-9:
            raise HermiticityViolationError(f"ε no simétrica: {sym_def:.3e}")
        if float(np.min(la.eigvalsh(0.5 * (eps + eps.T)))) <= 0.0:
            raise TopologicalInvariantError("ε no SPD: medio no pasivo")

    @property
    def maslov_parity(self) -> int:
        return int(self.wkb_maslov_index) % 2

    @property
    def quantum_state_hash(self) -> str:
        return f"BOSON_RF_{self.spectral_cutoff_functor}_{self.wkb_maslov_index}"


@dataclass(frozen=True)
class HouseholderReflectionFermion(Fermion):
    r"""
    Fermión de reflexión de Householder H = I − 2vvᵀ/‖v‖² ∈ O(n), det H = −1.
    En Cl(n) la reflexión es la adjunción torcida x ↦ −v x v⁻¹ ⇒ elemento de Pin∖Spin.
    La clase de obstrucción NO se declara: se DERIVA (w₁ = (1−det)/2 ∈ π₀(O(n)) = ℤ₂).
    """

    covariant_hyperplane_normal: NDArray[np.float64]
    monodromy_spectral_radius: float
    cohomology_obstruction_class: int = 1

    def householder_matrix(self) -> NDArray[np.float64]:
        v = np.asarray(self.covariant_hyperplane_normal, dtype=np.float64).ravel()
        nv = float(la.norm(v))
        if nv <= 0.0:
            raise TopologicalInvariantError("normal de Householder nula")
        v = v / nv
        return np.eye(v.size) - 2.0 * np.outer(v, v)

    @property
    def derived_stiefel_whitney_w1(self) -> int:
        """w₁ derivada del determinante (−1 ⇒ clase no trivial de Pin)."""
        det = float(la.det(self.householder_matrix()))
        return int(round((1.0 - det) / 2.0)) % 2

    def validate(self) -> None:
        if self.derived_stiefel_whitney_w1 != int(self.cohomology_obstruction_class) % 2:
            raise PauliExclusionViolationError(
                "clase de obstrucción declarada ≠ derivada de det(H)"
            )
        if self.monodromy_spectral_radius < 0.0:
            raise TopologicalInvariantError("radio espectral de monodromía negativo")

    @property
    def quantum_state_hash(self) -> str:
        return (f"FERMION_HR_{self.derived_stiefel_whitney_w1}_"
                f"{self.monodromy_spectral_radius:.4f}")


# ═════════════════════════════════════════════════════════════════════════════════════
# §D · CONFIGURACIÓN (truncamientos por modo, presupuesto duro)
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class FockSpaceConfiguration:
    r"""
    ℋ_Fock = (⊗_{k<B} ℂ^{M_k+1}) ⊗ (⊗_{q<F} ℂ²).
    `boson_truncations` honra el formalismo M_k (la v5 sólo permitía M uniforme).
    """

    n_boson_modes: int
    n_fermion_modes: int
    boson_truncation: int = 5
    use_sparse: bool = False
    hermiticity_tol: float = 1e-10
    boson_truncations: Optional[Tuple[int, ...]] = None
    max_hilbert_dim: int = FockPoincareRSIConstants.MAX_HILBERT_DIM

    def __post_init__(self) -> None:
        if self.n_boson_modes < 0 or self.n_fermion_modes < 0:
            raise FockSpaceOverflowError("número de modos negativo")
        if self.boson_truncation < 0:
            raise FockSpaceOverflowError("truncamiento bosónico negativo")
        if self.hermiticity_tol <= 0.0:
            raise ValueError("hermiticity_tol debe ser positivo")
        if self.boson_truncations is not None:
            if len(self.boson_truncations) != self.n_boson_modes:
                raise FockSpaceOverflowError("boson_truncations incompatible con B")
            if any(m < 0 for m in self.boson_truncations):
                raise FockSpaceOverflowError("truncamiento por modo negativo")
        dim = math.prod(self.truncations_tuple[k] + 1 for k in range(self.n_boson_modes))
        dim *= 2 ** self.n_fermion_modes
        if dim > self.max_hilbert_dim:
            raise FockSpaceOverflowError(
                f"dim(ℋ)={dim} excede presupuesto {self.max_hilbert_dim}; "
                "reduzca M/B/F o use un sector de simetría"
            )

    @property
    def truncations_tuple(self) -> Tuple[int, ...]:
        if self.boson_truncations is not None:
            return tuple(int(m) for m in self.boson_truncations)
        return (int(self.boson_truncation),) * self.n_boson_modes

    @property
    def hilbert_dim(self) -> int:
        d = math.prod(m + 1 for m in self.truncations_tuple)
        return int(d * 2 ** self.n_fermion_modes)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 1 · ESPACIO DE FOCK, CLIFFORD/JORDAN–WIGNER, CUADRATURAS, RSI-L3 ══
# ══ (config → InteractionOperators O)                                              ══
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class AlgebraAuditReport:
    """Auditoría algebraica: prueba local exacta + sondas globales aleatorias."""

    ccr_local_residual: float
    car_local_residual: float
    idempotency_residual: float
    adjunction_residual: float
    quadrature_anomaly_residual: float
    global_probe_residual: float
    n_probes: int
    scale: float
    verdict: int


@dataclass(frozen=True, slots=True)
class TruncationCertificate:
    r"""
    T1: |⟨[Q_k,P_k]⟩ − i| = (M_k+1)·⟨|M_k⟩⟨M_k|⟩. La confianza del álgebra CCR es
    MEDIBLE: si la población de frontera es pequeña, la anomalía es irrelevante.
    """

    boundary_populations: Tuple[float, ...]
    ccr_anomaly_expectations: Tuple[float, ...]
    max_occupation: Tuple[float, ...]
    trustworthy: bool
    measured: bool


@dataclass(frozen=True)
class InteractionOperators:
    r"""
    O — artefacto terminal de la Fase 1 y dominio exacto del primer método de la Fase 2.
    Inmutable por contrato (tuplas; los arrays se marcan read-only donde es posible).
    """

    boson_ann: Tuple[Any, ...]
    boson_cre: Tuple[Any, ...]
    fermion_number: Tuple[Any, ...]
    hilbert_dim: int
    boson_dimensions: Tuple[int, ...]
    fermion_dimensions: Tuple[int, ...]
    boson_truncation: int
    use_sparse: bool
    boson_boundary_projectors: Tuple[Any, ...] = ()
    fermion_ann: Tuple[Any, ...] = ()
    fermion_cre: Tuple[Any, ...] = ()
    poincare_quadrature_q: Tuple[Any, ...] = ()
    poincare_quadrature_p: Tuple[Any, ...] = ()
    klein_parity_operator: Optional[Any] = None
    boson_truncations: Tuple[int, ...] = ()
    algebra_audit: Optional[AlgebraAuditReport] = None
    rsi_l3_algebra_defect: float = 0.0
    rsi_l3_generations: int = 0
    rsi_l3_converged: bool = True

    def __post_init__(self) -> None:
        expected = math.prod(self.boson_dimensions) * math.prod(self.fermion_dimensions)
        if expected != self.hilbert_dim or self.hilbert_dim <= 0:
            raise FockSpaceOverflowError("dimensión de Hilbert inconsistente en O")

    # ── ABI legacy: dict de residuos (ahora normalizado y con prueba local) ──────────
    def verify_relations(self, tol: float = 1e-9) -> Dict[str, float]:
        r"""
        Devuelve los residuos del dict legacy, pero calculados con rigor:
        la prueba es LOCAL (teorema T2) y se complementa con sondas globales.
        """
        a = self.algebra_audit
        if a is None:
            return {"algebra_audit": float("nan")}
        out = {
            "[b_k,b_k†]-(I-Π_M)": a.ccr_local_residual,
            "{f_q,f_q'†}-δI": a.car_local_residual,
            "n_q²-n_q": a.idempotency_residual,
            "b†-(b)†": a.adjunction_residual,
            "[Q,P]-i(I-Π_M)": a.quadrature_anomaly_residual,
            "global_probe": a.global_probe_residual,
        }
        for key, val in out.items():
            if val > tol:
                logger.warning("Residual algebraico alto en %s: %.3e", key, val)
        return out


class Phase1_FockSpaceBuilder:
    r"""
    FASE 1 — Síntesis del espacio de Fock ℤ₂-graduado y de su álgebra de operadores.

    Diferencias rigurosas frente a la v5:
      · Identidades dispersas reales (`sparse_identity`), nunca `csr_matrix(np.eye(d))`.
      · Inmersión por 3 factores I_left ⊗ local ⊗ I_right (no d identidades).
      · Prueba de relaciones en factores locales + sondas aleatorias (T2).
      · RSI-L3 R₁: McWeeny con certificado de convergencia y q̂ de Banach.
      · Certificado de truncamiento (T1) que convierte la anomalía CCR en observable.
    """

    def __init__(self, config: FockSpaceConfiguration) -> None:
        self._config = config
        self._truncs = config.truncations_tuple
        self._boson_trunc = config.boson_truncation
        self._n_b = config.n_boson_modes
        self._n_f = config.n_fermion_modes
        self._use_sparse = config.use_sparse
        self._herm_tol = config.hermiticity_tol
        self._boson_dims = tuple(m + 1 for m in self._truncs)
        self._fermion_dims = (2,) * self._n_f
        self._all_dims = self._boson_dims + self._fermion_dims
        self._boson_dim = int(math.prod(self._boson_dims)) if self._n_b else 1
        self._fermion_dim = 2 ** self._n_f
        self._total_dim = self._boson_dim * self._fermion_dim

        self._boson_ann_ops: List[Any] = []
        self._boson_cre_ops: List[Any] = []
        self._boson_boundary_ops: List[Any] = []
        self._fermion_ann_ops: List[Any] = []
        self._fermion_cre_ops: List[Any] = []
        self._fermion_number_ops: List[Any] = []
        self._poincare_q_ops: List[Any] = []
        self._poincare_p_ops: List[Any] = []
        self._klein_parity_op: Optional[Any] = None
        self._rsi_l3_defect = 0.0
        self._rsi_l3_gens = 0
        self._rsi_l3_converged = True
        self._audit: Optional[AlgebraAuditReport] = None

        self._build_operators()
        self._rsi_l3_defect, self._rsi_l3_gens, self._rsi_l3_converged = (
            self._rsi_level3_certify_fock_algebra()
        )
        self._audit = self.audit_algebra()
        self._validate_operators()
        logger.debug(
            "[Fase1] dim=%d B=%d F=%d sparse=%s defecto=%.2e gens=%d conv=%s",
            self._total_dim, self._n_b, self._n_f, self._use_sparse,
            self._rsi_l3_defect, self._rsi_l3_gens, self._rsi_l3_converged,
        )

    # ── 1.1 · Operadores locales monomodo ────────────────────────────────────────────
    def _local_boson(self, m: int) -> Tuple[Any, Any, Any]:
        r"""a_{n-1,n}=√n en ℂ^{m+1}; Π_M=(m+1)|m⟩⟨m| ⇒ [a,a†]=I−Π_M (T1, exacto)."""
        d = m + 1
        off = np.sqrt(np.arange(1, d, dtype=np.float64))
        if self._use_sparse:
            a = diags(off.astype(np.complex128), offsets=1, shape=(d, d), format="csr")
            pi = csr_matrix(
                (np.array([float(d)], dtype=np.complex128),
                 (np.array([m]), np.array([m]))), shape=(d, d)
            )
        else:
            a = np.zeros((d, d), dtype=np.complex128)
            a[np.arange(m), np.arange(1, d)] = off
            pi = np.zeros((d, d), dtype=np.complex128)
            pi[m, m] = float(d)
        return a, (a.conj().T.tocsr() if self._use_sparse else a.conj().T), pi

    @staticmethod
    def _local_fermion() -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """f=[[0,1],[0,0]], n=f†f=diag(0,1), σ_z=I−2n (cuerda de paridad JW)."""
        f = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
        fd = f.conj().T
        return f, fd, fd @ f, np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    def _eye(self, d: int) -> Any:
        if self._use_sparse:
            return sparse_identity(d, format="csr", dtype=np.complex128)
        return np.eye(d, dtype=np.complex128)

    def _embed(self, local: Any, slot: int) -> Any:
        r"""I_left ⊗ local ⊗ I_right: 3 productos de Kronecker, coste lineal en nnz."""
        left = int(math.prod(self._all_dims[:slot])) if slot > 0 else 1
        right = int(math.prod(self._all_dims[slot + 1:])) if slot + 1 < len(self._all_dims) else 1
        core = skron(local, self._eye(right), format="csr") if self._use_sparse else np.kron(local, self._eye(right))
        if left == 1:
            return core
        return skron(self._eye(left), core, format="csr") if self._use_sparse else np.kron(self._eye(left), core)

    def _embed_fermion_string(self, locals_by_slot: Dict[int, Any]) -> Any:
        r"""Cuerda JW: I_{B} ⊗ (⊗_{q<F} O_q) con O_q ∈ {σ_z, f, f†, n, I}."""
        ops = [locals_by_slot.get(q, self._eye(2)) for q in range(self._n_f)]
        core = ops[0]
        for op in ops[1:]:
            core = skron(core, op, format="csr") if self._use_sparse else np.kron(core, op)
        if self._boson_dim == 1:
            return core
        return (skron(self._eye(self._boson_dim), core, format="csr")
                if self._use_sparse else np.kron(self._eye(self._boson_dim), core))

    # ── 1.2 · Ensamblaje del álgebra completa ────────────────────────────────────────
    def _build_operators(self) -> None:
        f, fd, nf, sz = self._local_fermion()
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        for k in range(self._n_b):
            a, ad, pi = self._local_boson(self._truncs[k])
            b, bd = self._embed(a, k), self._embed(ad, k)
            self._boson_ann_ops.append(b)
            self._boson_cre_ops.append(bd)
            self._boson_boundary_ops.append(self._embed(pi, k))
            self._poincare_q_ops.append(inv_sqrt2 * (b + bd))
            self._poincare_p_ops.append(1j * inv_sqrt2 * (bd - b))
        for q in range(self._n_f):
            string = {j: (csr_matrix(sz) if self._use_sparse else sz) for j in range(q)}
            ann = dict(string); ann[q] = csr_matrix(f) if self._use_sparse else f
            cre = dict(string); cre[q] = csr_matrix(fd) if self._use_sparse else fd
            self._fermion_ann_ops.append(self._embed_fermion_string(ann))
            self._fermion_cre_ops.append(self._embed_fermion_string(cre))
            self._fermion_number_ops.append(
                self._embed_fermion_string({q: csr_matrix(nf) if self._use_sparse else nf})
            )
        self._klein_parity_op = self._embed_fermion_string(
            {q: (csr_matrix(sz) if self._use_sparse else sz) for q in range(self._n_f)}
        )

    # ── 1.3 · RSI-L3 R₁: purificación de McWeeny CERTIFICADA ─────────────────────────
    def _rsi_level3_certify_fock_algebra(self) -> Tuple[float, int, bool]:
        r"""
        R₁⁽³⁾ sobre los generadores:
          Nivel 1 (paramétrico): adjunción exacta b† ← (b)†.
          Nivel 2 (hermítico): n_q ← ½(n_q+n_q†).
          Nivel 3 (meta / punto fijo de McWeeny–Lawvere): n ← 3n²−2n³, convergencia
            cuadrática al proyector. Se mide el defecto DESPUÉS de purificar y se
            estima q̂ = defecto_{m+1}/defecto_m (contracción de Banach a posteriori).
        Devuelve (defecto_final, generaciones, convergido). Nunca informa el defecto
        previo a la reparación como si fuera el certificado (bug de la v5).
        """
        for k, b in enumerate(self._boson_ann_ops):
            self._boson_cre_ops[k] = b.conj().T
        defect_prev, defect, gens, conv = float("inf"), 0.0, 0, True
        for depth in range(FockPoincareRSIConstants.RSI_L3_MAX_DEPTH):
            defects: List[float] = []
            for q, nq in enumerate(self._fermion_number_ops):
                nh = 0.5 * (nq + nq.conj().T)
                d0 = self._rel_residual(nh @ nh - nh, nh)
                if d0 > FockPoincareRSIConstants.MACHINE_EPS:
                    sq = nh @ nh
                    nh = 3.0 * sq - 2.0 * (sq @ nh)
                    nh = 0.5 * (nh + nh.conj().T)
                defects.append(self._rel_residual(nh @ nh - nh, nh))
                self._fermion_number_ops[q] = nh
            defect = max(defects) if defects else 0.0
            gens = depth + 1
            if defect <= self._herm_tol:
                break
            if defect >= defect_prev:      # pérdida de contractividad ⇒ no insistir
                conv = False
                break
            defect_prev = defect
        else:
            conv = defect <= self._herm_tol
        for k, (b, bd) in enumerate(zip(self._boson_ann_ops, self._boson_cre_ops)):
            if self._rel_residual(bd - b.conj().T, b) > self._herm_tol:
                raise HermiticityViolationError(f"b_{k}† ≠ (b_{k})†")
        return float(defect), int(gens), bool(conv)

    # ── 1.4 · Auditoría algebraica: prueba LOCAL + sondas GLOBALES (T2) ──────────────
    @staticmethod
    def _rel_residual(diff: Any, ref: Any) -> float:
        """Residuo relativo de Frobenius (invariante de escala; no max-entry)."""
        def fro(x: Any) -> float:
            if issparse(x):
                return float(np.sqrt(np.sum(np.abs(x.data) ** 2))) if x.nnz else 0.0
            return float(la.norm(np.asarray(x), "fro"))
        return fro(diff) / max(fro(ref), 1e-300)

    def audit_algebra(self, n_probes: int = 4, seed: int = 20261115) -> AlgebraAuditReport:
        r"""
        PRUEBA (local, exacta): para A = A_loc⊗I y B = B_loc⊗I en factores distintos
        [A,B]=0 idénticamente; en el mismo factor [A,B]=[A_loc,B_loc]⊗I. Por tanto
        verificar CCR/CAR en ℂ^{M+1} y ℂ² ES una prueba del álgebra completa.
        SONDA (global, aleatoria): ‖(AB±BA−C)v‖/‖v‖ con v aleatorio detecta errores de
        inmersión que la prueba local no puede ver (certificado probabilístico).
        """
        ccr, anom, adj = 0.0, 0.0, 0.0
        for m in self._truncs:
            a, ad, pi = self._local_boson(m)
            eye = np.eye(m + 1, dtype=np.complex128)
            ccr = max(ccr, self._rel_residual(a @ ad - ad @ a - (eye - pi), eye))
            q = (a + ad) / math.sqrt(2.0)
            p = 1j * (ad - a) / math.sqrt(2.0)
            anom = max(anom, self._rel_residual(q @ p - p @ q - 1j * (eye - pi), eye))
            adj = max(adj, self._rel_residual(ad - a.conj().T, a))
        f, fd, nf, sz = self._local_fermion()
        eye2 = np.eye(2, dtype=np.complex128)
        car = max(
            self._rel_residual(f @ fd + fd @ f - eye2, eye2),
            self._rel_residual(f @ f, eye2),
            self._rel_residual(nf @ nf - nf, eye2),
        )
        idem = max((self._rel_residual(n @ n - n, n) for n in self._fermion_number_ops),
                   default=0.0)
        rng = np.random.default_rng(seed)
        probe, scale = 0.0, 1.0
        pairs: List[Tuple[Any, Any, Any, float]] = []
        for q in range(self._n_f):
            for qp in range(q, self._n_f):
                anti = self._fermion_ann_ops[q] @ self._fermion_cre_ops[qp] + \
                       self._fermion_cre_ops[qp] @ self._fermion_ann_ops[q]
                target = self._eye(self._total_dim) if q == qp else None
                pairs.append((anti, target, None, 1.0))
        for anti, target, _unused, sc in pairs[: max(1, 2 * self._n_f)]:
            for _ in range(max(1, n_probes)):
                v = rng.standard_normal(self._total_dim) + 1j * rng.standard_normal(self._total_dim)
                lhs = anti @ v
                rhs = v if target is not None else np.zeros_like(v)
                probe = max(probe, float(la.norm(lhs - rhs) / max(la.norm(v), 1e-300)))
            scale = max(scale, sc)
        worst = max(ccr, car, idem, adj, anom, probe)
        verdict = int(HeytingVerdict.COHERENT if worst <= 1e-8
                      else HeytingVerdict.DEGRADED if worst <= 1e-5
                      else HeytingVerdict.VETOED)
        return AlgebraAuditReport(
            ccr_local_residual=float(ccr), car_local_residual=float(car),
            idempotency_residual=float(idem), adjunction_residual=float(adj),
            quadrature_anomaly_residual=float(anom), global_probe_residual=float(probe),
            n_probes=int(n_probes), scale=float(scale), verdict=verdict,
        )

    # ── 1.5 · Certificado de truncamiento (anomalía CCR como observable, T1) ─────────
    def truncation_certificate(self, rho: Optional[Any] = None) -> TruncationCertificate:
        r"""
        Sin ρ: certificado NO medido (honestidad; nunca se inventa un 0).
        Con ρ: p_M^{(k)} = Tr(ρ Π_k)/(M_k+1), anomalía = (M_k+1) p_M^{(k)}.
        """
        if rho is None:
            return TruncationCertificate((), (), (), False, False)
        pops, anoms, occs = [], [], []
        for k, m in enumerate(self._truncs):
            pi = self._boson_boundary_ops[k]
            val = pi @ rho
            tr = float(np.real(val.diagonal().sum() if issparse(val) else np.trace(val)))
            p = tr / float(m + 1)
            n_op = self._boson_cre_ops[k] @ self._boson_ann_ops[k]
            nv = n_op @ rho
            occ = float(np.real(nv.diagonal().sum() if issparse(nv) else np.trace(nv)))
            pops.append(p); anoms.append((m + 1) * p); occs.append(occ)
        trust = all(a <= FockPoincareRSIConstants.TRUNCATION_POP_TOL for a in anoms)
        return TruncationCertificate(tuple(pops), tuple(anoms), tuple(occs), bool(trust), True)

    def _validate_operators(self) -> None:
        for q, nq in enumerate(self._fermion_number_ops):
            if self._rel_residual(nq - nq.conj().T, nq) > self._herm_tol:
                raise HermiticityViolationError(f"n_{q} no hermítico")
            if self._rel_residual(nq @ nq - nq, nq) > self._herm_tol:
                raise PauliExclusionViolationError(f"n_{q}² ≠ n_{q}")

    # ── 1.6 · MÉTODO TERMINAL DE LA FASE 1 → continuación a la Fase 2 ────────────────
    def get_interaction_operators(
        self, continuation: Optional[Callable[[InteractionOperators], R]] = None
    ) -> Union[InteractionOperators, R]:
        r"""
        Definición formal final de la Fase 1: `continuation: O → R`. Su cola
        `return continuation(O)` es el arranque de `Phase2._continue_from_phase1_operators`.
        """
        ops = InteractionOperators(
            boson_ann=tuple(self._boson_ann_ops), boson_cre=tuple(self._boson_cre_ops),
            fermion_number=tuple(self._fermion_number_ops), hilbert_dim=self._total_dim,
            boson_dimensions=self._boson_dims, fermion_dimensions=self._fermion_dims,
            boson_truncation=self._boson_trunc, use_sparse=self._use_sparse,
            boson_boundary_projectors=tuple(self._boson_boundary_ops),
            fermion_ann=tuple(self._fermion_ann_ops), fermion_cre=tuple(self._fermion_cre_ops),
            poincare_quadrature_q=tuple(self._poincare_q_ops),
            poincare_quadrature_p=tuple(self._poincare_p_ops),
            klein_parity_operator=self._klein_parity_op,
            boson_truncations=self._truncs, algebra_audit=self._audit,
            rsi_l3_algebra_defect=self._rsi_l3_defect,
            rsi_l3_generations=self._rsi_l3_gens, rsi_l3_converged=self._rsi_l3_converged,
        )
        if continuation is None:
            return ops
        return continuation(ops)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 2 · COLISIONADOR, SECTORES, ESPECTRO EXACTO, WEGNER, KAM (O → K) ══
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectralStatistics:
    r"""Estadística de espaciamientos DESYMMETRIZADA, con incertidumbre y honestidad."""

    mean_r: float
    std_error: float
    n_ratios: int
    n_sectors: int
    measured: bool
    exactly_solvable: bool
    poisson_reference: float
    goe_reference: float
    interpretation: str
    verdict: int


@dataclass(frozen=True, slots=True)
class DiophantineKAMReport:
    """Condición diofántica REAL, resonancias de fracción continua y solapamiento."""

    gamma_effective: float
    tau: float
    kam_satisfied: bool
    resonances: Tuple[Tuple[int, int, int, float], ...]
    chirikov_max: float
    golden_nobility: float
    measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class WegnerFlowReport:
    r"""T4: flujo iso-espectral REAL con Lyapunov ‖H_off‖² decreciente."""

    off_norm_initial: float
    off_norm_final: float
    eta_norm_final: float
    steps: int
    monotone: bool
    isospectral_defect: float
    lyapunov_decay: float
    diagonalizing_unitary: Optional[Any]
    verdict: int


@dataclass(frozen=True)
class CatadioptricHamiltonian:
    r"""K — artefacto terminal de la Fase 2, dominio exacto de la Fase 3."""

    H_int: Any
    hilbert_dim: int
    operator_norms: Dict[Tuple[int, int], float]
    max_coupling: float
    use_sparse: bool
    boson_dimensions: Tuple[int, ...] = ()
    fermion_dimensions: Tuple[int, ...] = ()
    number_conserving: bool = True
    exactly_solvable: bool = True
    conserved_charges: Tuple[str, ...] = ()
    exact_eigenvalues: Optional[NDArray[np.float64]] = None
    spectral_statistics: Optional[SpectralStatistics] = None
    kam_report: Optional[DiophantineKAMReport] = None
    wegner_report: Optional[WegnerFlowReport] = None
    algebra_verdict: int = 0
    berry_tabor_level_spacing_ratio: float = float("nan")
    kam_diophantine_resonance_margin: float = float("nan")
    wegner_bch_flow_norm: float = 0.0
    rsi_l3_hermiticity_residual: float = 0.0
    rsi_l3_generations: int = 0


class Phase2_CatadioptricCollider:
    r"""
    FASE 2 — Colisionador catadióptrico con auditoría espectral rigurosa.

    H = Σ_k ω_k b†_k b_k + Σ_q ε_q n_q + Σ_k χ_k (b†_k b_k)²
        + Σ_{k,q} (g_kq b_k n_q + ḡ_kq b†_k n_q)
        + [opcional] Σ_{k,q} (J_kq b_k f†_q + J̄_kq b†_k f_q)   (Jaynes–Cummings)

    Con sólo el término g (ABI por defecto) el modelo es EXACTAMENTE soluble (T3):
    se demuestra, se calcula su espectro analítico (nodos de Gauss–Hermite) y se
    declara que la estadística r no es diagnóstica de caos. Para que KAM/Berry–Tabor
    sean significativos hay que activar ω/χ/J (no integrabilidad genuina).
    """

    def __init__(
        self,
        ops_or_config: Union[InteractionOperators, FockSpaceConfiguration],
        coupling_matrix: NDArray[np.complex128],
        *,
        boson_frequencies: Optional[Sequence[float]] = None,
        fermion_energies: Optional[Sequence[float]] = None,
        kerr_chi: Optional[Sequence[float]] = None,
        exchange_coupling: Optional[NDArray[np.complex128]] = None,
        planck_normalized: float = 1.0,
        wegner_steps: int = 0,
        kam_k_max: int = 6,
        perturbation_epsilon: float = 1e-3,
    ) -> None:
        ops = (Phase1_FockSpaceBuilder(ops_or_config).get_interaction_operators()
               if isinstance(ops_or_config, FockSpaceConfiguration) else ops_or_config)
        self._ops: InteractionOperators = self._continue_from_phase1_operators(ops)
        self._coupling = np.asarray(coupling_matrix, dtype=np.complex128)
        self._use_sparse = self._ops.use_sparse
        self._hbar = float(planck_normalized)
        if self._hbar <= 0.0:
            raise ValueError("ħ debe ser positivo")
        self._validate_coupling()
        self._omega = self._vec_or_zero(boson_frequencies, len(self._ops.boson_ann))
        self._eps_f = self._vec_or_zero(fermion_energies, len(self._ops.fermion_number))
        self._chi = self._vec_or_zero(kerr_chi, len(self._ops.boson_ann))
        self._exchange = (None if exchange_coupling is None
                          else np.asarray(exchange_coupling, dtype=np.complex128))
        self._number_conserving = self._exchange is None
        self._exactly_solvable = (
            self._number_conserving and not np.any(self._omega)
            and not np.any(self._chi) and not np.any(self._eps_f)
        )
        self._norms = {(k, q): float(abs(self._coupling[k, q]))
                       for k in range(self._coupling.shape[0])
                       for q in range(self._coupling.shape[1])}
        raw = self._build_hamiltonian()
        self._H_int, self._herm_res = self._project_hermitian(raw)
        self._wegner = self._wegner_flow(self._H_int, steps=int(wegner_steps))
        self._blocks = self._symmetry_blocks()
        self._exact_eigs = self._exact_linear_spectrum() if self._exactly_solvable else None
        self._stats = self._analyze_level_statistics()
        self._kam = self._analyze_diophantine_kam(k_max=kam_k_max, epsilon=perturbation_epsilon)
        logger.debug("[Fase2] dim=%d exacto=%s sectores=%d <r>=%.4f γ_eff=%.3e",
                     self._ops.hilbert_dim, self._exactly_solvable, len(self._blocks),
                     self._stats.mean_r, self._kam.gamma_effective)

    # ── 2.1 · PRIMER MÉTODO DE LA FASE 2 (continuación exacta de la Fase 1) ──────────
    def _continue_from_phase1_operators(self, ops: InteractionOperators) -> InteractionOperators:
        r"""Valida el contrato O emergente de `get_interaction_operators` (Fase 1)."""
        if not isinstance(ops, InteractionOperators):
            raise TopologicalInvariantError(
                f"Continuidad Φ₁→Φ₂ violada: se esperaba InteractionOperators, hay {type(ops)}"
            )
        if ops.hilbert_dim <= 0:
            raise FockSpaceOverflowError("dimensión de Hilbert inválida en O")
        if ops.algebra_audit is not None and ops.algebra_audit.verdict == int(HeytingVerdict.VETOED):
            raise PauliExclusionViolationError("álgebra de Fock vetada en Φ₁")
        return ops

    @staticmethod
    def _vec_or_zero(v: Optional[Sequence[float]], n: int) -> NDArray[np.float64]:
        if v is None:
            return np.zeros(n, dtype=np.float64)
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size != n or not np.all(np.isfinite(arr)):
            raise TopologicalInvariantError("vector espectral de longitud/valor inválido")
        return arr

    def _validate_coupling(self) -> None:
        b, f = len(self._ops.boson_ann), len(self._ops.fermion_number)
        if self._coupling.shape != (b, f):
            raise TopologicalInvariantError(f"g debe ser ({b},{f}); hay {self._coupling.shape}")
        if not np.all(np.isfinite(self._coupling)):
            raise FockSpaceOverflowError("acoplamientos no finitos")
        if self._exchange is not None and self._exchange.shape != (b, f):
            raise TopologicalInvariantError("exchange_coupling incompatible con g")

    @property
    def _max_coupling(self) -> float:
        return float(np.max(np.abs(self._coupling))) if self._coupling.size else 0.0

    # ── 2.2 · Hamiltoniano y proyección hermítica ────────────────────────────────────
    def _build_hamiltonian(self) -> Any:
        d = self._ops.hilbert_dim
        h = csr_matrix((d, d), dtype=np.complex128) if self._use_sparse else np.zeros((d, d), dtype=np.complex128)
        for k, (b, bd) in enumerate(zip(self._ops.boson_ann, self._ops.boson_cre)):
            if self._omega[k] != 0.0:
                h = h + self._omega[k] * (bd @ b)
            if self._chi[k] != 0.0:
                nk = bd @ b
                h = h + self._chi[k] * (nk @ nk)
            for q, nq in enumerate(self._ops.fermion_number):
                g = self._coupling[k, q]
                if g != 0.0:
                    h = h + g * (b @ nq) + np.conj(g) * (bd @ nq)
                if self._exchange is not None:
                    j = self._exchange[k, q]
                    if j != 0.0:
                        fq, fqd = self._ops.fermion_ann[q], self._ops.fermion_cre[q]
                        h = h + j * (b @ fqd) + np.conj(j) * (bd @ fq)
        for q, nq in enumerate(self._ops.fermion_number):
            if self._eps_f[q] != 0.0:
                h = h + self._eps_f[q] * nq
        return h

    def _project_hermitian(self, h: Any) -> Tuple[Any, float]:
        sym = 0.5 * (h + h.conj().T)
        res = Phase1_FockSpaceBuilder._rel_residual(h - sym, h)
        if res > 1e-8:
            raise HermiticityViolationError(f"H no hermítico: {res:.3e}")
        return sym, float(res)

    # ── 2.3 · Sectores de simetría (desymmetrización obligatoria) ────────────────────
    def _symmetry_blocks(self) -> List[NDArray[np.int64]]:
        r"""
        Si [H,n_q]=0 ∀q (modelo por defecto) los sectores son las configuraciones
        fermiónicas s ∈ {0,1}^F; si hay intercambio JC, sólo la paridad de Klein
        (−1)^{N_F} se conserva. Índice global = b_idx·2^F + s (orden de Kronecker).
        """
        d, nf = self._ops.hilbert_dim, len(self._ops.fermion_number)
        fdim = 2 ** nf
        idx = np.arange(d, dtype=np.int64)
        bits = idx % fdim
        if self._number_conserving:
            return [idx[bits == s] for s in range(fdim) if np.any(bits == s)]
        parity = np.array([bin(int(s)).count("1") % 2 for s in range(fdim)], dtype=np.int64)
        pv = parity[bits]
        return [idx[pv == p] for p in (0, 1) if np.any(pv == p)]

    # ── 2.4 · Espectro EXACTO del modelo lineal (T3: nodos de Gauss–Hermite) ─────────
    @staticmethod
    @lru_cache(maxsize=128)
    def _hermite_nodes(m_plus_1: int) -> NDArray[np.float64]:
        r"""
        Q_trunc = (b+b†)/√2 tiene entradas √(n/2): es la matriz de Jacobi de los
        polinomios de Hermite ⇒ spec(Q_trunc) = nodos de Gauss–Hermite (peso e^{−x²}).
        Verificado: M+1=2 ⇒ ±1/√2; M+1=3 ⇒ {0,±√(3/2)}. Exacto, no aproximado.
        """
        return np.polynomial.hermite.hermgauss(m_plus_1)[0]

    def _exact_linear_spectrum(self) -> NDArray[np.float64]:
        r"""
        spec(H) = ⊎_{s∈{0,1}^F} { Σ_k √2|G_k(s)| x_{i_k} },  G_k(s)=Σ_q g_kq s_q.
        Coste O(d) (no O(d³)): cada sector es una suma separable de posiciones
        truncadas tras un gauge de fase diagonal W_k=diag(e^{iθ_k n}).
        """
        b, f = len(self._ops.boson_ann), len(self._ops.fermion_number)
        nodes = [self._hermite_nodes(m + 1) for m in self._ops.boson_truncations]
        out: List[NDArray[np.float64]] = []
        for s in product((0, 1), repeat=f):
            g_eff = self._coupling @ np.asarray(s, dtype=np.float64) if f else np.zeros(b)
            e = np.zeros(1, dtype=np.float64)
            for k in range(b):
                e = (e[:, None] + math.sqrt(2.0) * abs(g_eff[k]) * nodes[k][None, :]).ravel()
            out.append(e)
        return np.sort(np.concatenate(out)) if out else np.zeros(1)

    # ── 2.5 · Flujo de Wegner REAL (T4) — diagnóstico, no cambio de base oculto ──────
    def _wegner_flow(self, h: Any, steps: int = 0, dl: float = 0.05) -> WegnerFlowReport:
        r"""
        η = [H_d, H_off] antihermítica; H ← U H U† con U = Cayley(dl·η) unitaria exacta.
        Lyapunov: d‖H_off‖²/dl = −2‖η‖². Se devuelve la unitaria diagonalizadora para
        que el llamante pueda transformar ρ y L coherentemente; H_int NO se sustituye
        (cambiar de base sin transformar el estado sería un error físico).
        """
        dense = h.toarray() if issparse(h) else np.asarray(h)
        n = dense.shape[0]
        eye = np.eye(n, dtype=np.complex128)

        def off_norm(x: NDArray) -> float:
            return float(la.norm(x - np.diag(np.diag(x)), "fro"))

        h_cur = dense.copy()
        u_tot = eye.copy()
        off0 = off_norm(h_cur)
        prev, monotone, eta_n, k = off0, True, 0.0, 0
        spec0 = np.sort(la.eigvalsh(0.5 * (h_cur + h_cur.conj().T)))
        for k in range(max(0, steps)):
            hd = np.diag(np.diag(h_cur))
            eta = hd @ h_cur - h_cur @ hd
            eta = 0.5 * (eta - eta.conj().T)
            eta_n = float(la.norm(eta, "fro"))
            if eta_n <= 1e-12 * max(1.0, float(la.norm(h_cur, "fro"))):
                break
            step = min(dl, 0.5 / max(eta_n, 1e-300))
            half = 0.5 * step * eta
            u = la.solve(eye - half, eye + half, check_finite=False)
            h_cur = u @ h_cur @ u.conj().T
            h_cur = 0.5 * (h_cur + h_cur.conj().T)
            u_tot = u @ u_tot
            cur = off_norm(h_cur)
            monotone = monotone and (cur <= prev + 1e-12)
            prev = cur
        spec1 = np.sort(la.eigvalsh(0.5 * (h_cur + h_cur.conj().T)))
        iso = float(np.max(np.abs(spec1 - spec0))) if n else 0.0
        verdict = int(HeytingVerdict.COHERENT if (monotone and iso < 1e-8)
                      else HeytingVerdict.DEGRADED)
        return WegnerFlowReport(
            off_norm_initial=off0, off_norm_final=float(prev), eta_norm_final=float(eta_n),
            steps=int(k), monotone=bool(monotone), isospectral_defect=iso,
            lyapunov_decay=float(off0 ** 2 - prev ** 2),
            diagonalizing_unitary=(u_tot if steps > 0 else None), verdict=verdict,
        )

    # ── 2.6 · Estadística de espaciamientos por sector, con incertidumbre ────────────
    def _analyze_level_statistics(self) -> SpectralStatistics:
        r"""
        r_n = min(s_n,s_{n+1})/max(s_n,s_{n+1}) DENTRO de cada sector de simetría.
        Se reporta error estándar (std/√N) y `measured`; si no se puede medir, NaN
        (jamás el valor teórico como si fuera dato, bug crítico de la v5).
        """
        ratios: List[NDArray[np.float64]] = []
        if self._exactly_solvable and self._exact_eigs is not None:
            fdim = 2 ** len(self._ops.fermion_number)
            chunks = np.array_split(self._exact_eigs, max(1, fdim))
            spectra = [np.sort(c) for c in chunks]
        else:
            spectra = []
            dense = self._H_int.toarray() if issparse(self._H_int) else np.asarray(self._H_int)
            for block in self._blocks:
                if block.size < 3:
                    continue
                sub = dense[np.ix_(block, block)]
                spectra.append(np.sort(la.eigvalsh(0.5 * (sub + sub.conj().T)).real))
        for e in spectra:
            s = np.diff(e)
            s = s[s > 1e-12 * max(1.0, float(np.max(np.abs(e))))]
            if s.size >= 2:
                lo, hi = np.minimum(s[:-1], s[1:]), np.maximum(s[:-1], s[1:])
                ratios.append(lo / np.maximum(hi, 1e-300))
        if not ratios:
            return SpectralStatistics(
                float("nan"), float("nan"), 0, len(self._blocks), False,
                self._exactly_solvable, FockPoincareRSIConstants.POISSON_INTEGRABLE_RATIO,
                FockPoincareRSIConstants.GOE_CHAOTIC_RATIO,
                "NO MEDIDO: espectro insuficiente; no se fabrica el valor teórico",
                int(HeytingVerdict.DEGRADED),
            )
        allr = np.concatenate(ratios)
        mean = float(np.mean(allr))
        se = float(np.std(allr, ddof=1) / math.sqrt(allr.size)) if allr.size > 1 else float("nan")
        if self._exactly_solvable:
            interp = ("EXACTAMENTE SOLUBLE (T3): oscilador desplazado por sector; "
                      "r no diagnostica caos (espectro de nodos de Hermite, rígido)")
            verdict = int(HeytingVerdict.DEGRADED)  # informative-only, no caos esperado
        elif abs(mean - FockPoincareRSIConstants.POISSION_INTEGRABLE_RATIO
                   if False else mean - FockPoincareRSIConstants.POISSON_INTEGRABLE_RATIO) < 3 * (se or 1):
            interp = "compatible con Poisson/Berry–Tabor (toros integrables)"
            verdict = int(HeytingVerdict.COHERENT)
        elif mean > FockPoincareRSIConstants.GOE_CHAOTIC_RATIO - 3 * (se or 1):
            interp = "compatible con Wigner–Dyson (caos homoclínico)"
            verdict = int(HeytingVerdict.DEGRADED)
        else:
            interp = "régimen intermedio (mezcla regular/caótica)"
            verdict = int(HeytingVerdict.DEGRADED)
        return SpectralStatistics(
            mean_r=mean, std_error=se, n_ratios=int(allr.size), n_sectors=len(spectra),
            measured=True, exactly_solvable=self._exactly_solvable,
            poisson_reference=FockPoincareRSIConstants.POISSON_INTEGRABLE_RATIO,
            goe_reference=FockPoincareRSIConstants.GOE_CHAOTIC_RATIO,
            interpretation=interp, verdict=verdict,
        )

    # ── 2.7 · KAM diofántico REAL + resonancias + Chirikov (T8, §Poincaré) ───────────
    def _analyze_diophantine_kam(
        self, k_max: int = 6, epsilon: float = 1e-3, tau: float = 2.0,
        gamma_threshold: float = 1e-4, n_freq: int = 4,
    ) -> DiophantineKAMReport:
        r"""
        ω_i = frecuencias de Bohr (E_i−E_0)/ħ de los primeros niveles del sector
        relevante. γ_eff = min_{0<‖k‖₁≤K} |k·ω|·‖k‖₁^τ (condición de Kolmogorov).
        Resonancias p/q por fracciones continuas (mejores aproximantes).
        Chirikov: K_ij = (Δ_i+Δ_j)/|ω_i−ω_j| con Δ=2√(ε|V|) (semianchura de separatrix).
        """
        eigs = self._exact_eigs
        if eigs is None:
            dense = self._H_int.toarray() if issparse(self._H_int) else np.asarray(self._H_int)
            block = self._blocks[0] if self._blocks else np.arange(dense.shape[0])
            sub = dense[np.ix_(block, block)]
            eigs = np.sort(la.eigvalsh(0.5 * (sub + sub.conj().T)).real)
        uniq = np.unique(np.round(eigs, 12))
        if uniq.size < 3:
            return DiophantineKAMReport(float("nan"), tau, False, (), 0.0, 0.0, False,
                                        int(HeytingVerdict.DEGRADED))
        w = (uniq[1: 1 + min(n_freq, uniq.size - 1)] - uniq[0]) / self._hbar
        w = w[np.abs(w) > 1e-14]
        if w.size < 2:
            return DiophantineKAMReport(float("nan"), tau, False, (), 0.0, 0.0, False,
                                        int(HeytingVerdict.DEGRADED))
        kk = int(k_max)
        while (2 * kk + 1) ** w.size > 200_000 and kk > 2:
            kk -= 1
        gamma = float("inf")
        for k in product(range(-kk, kk + 1), repeat=w.size):
            arr = np.asarray(k, dtype=np.float64)
            l1 = float(np.sum(np.abs(arr)))
            if l1 == 0.0 or next(x for x in k if x != 0) < 0:
                continue
            gamma = min(gamma, abs(float(arr @ w)) * l1 ** tau)
        reson: List[Tuple[int, int, int, float]] = []
        chir = 0.0
        half = 2.0 * math.sqrt(max(epsilon, 0.0))
        for i in range(w.size):
            for j in range(i + 1, w.size):
                ratio = abs(w[i] / w[j])
                p, q, err = self._best_rational(ratio, 12)
                if err < 5e-2:
                    reson.append((i, j, int(p + q), float(err)))
                sep = abs(abs(w[i]) - abs(w[j]))
                if sep > 1e-14:
                    chir = max(chir, float(2.0 * half / sep))
        phi = 0.5 * (1.0 + math.sqrt(5.0))
        noble = 1.0 / (1.0 + abs(abs(w[0] / w[1]) - phi))
        verdict = int(HeytingVerdict.VETOED if chir > 2.0
                      else HeytingVerdict.DEGRADED if chir > 1.0 or gamma < gamma_threshold
                      else HeytingVerdict.COHERENT)
        return DiophantineKAMReport(
            gamma_effective=float(gamma), tau=tau, kam_satisfied=bool(gamma >= gamma_threshold),
            resonances=tuple(reson), chirikov_max=chir, golden_nobility=float(noble),
            measured=True, verdict=verdict,
        )

    @staticmethod
    def _best_rational(x: float, q_max: int) -> Tuple[int, int, float]:
        """Mejor aproximante p/q (q≤q_max) por convergentes de la aplicación de Gauss."""
        p0, q0, p1, q1 = 1, 0, int(math.floor(x)), 1
        best = (p1, max(q1, 1), abs(x - p1 / max(q1, 1)))
        v = float(x)
        for _ in range(24):
            a = int(math.floor(v))
            p2, q2 = a * p1 + p0, a * q1 + q0
            if q2 > q_max or q2 <= 0:
                break
            err = abs(x - p2 / q2)
            if err < best[2]:
                best = (p2, q2, err)
            p0, q0, p1, q1 = p1, q1, p2, q2
            frac = v - a
            if frac < 1e-12:
                break
            v = 1.0 / frac
        return best

    # ── 2.8 · Propagador: espectral exacto / Krylov (Yoshida sólo donde es lícito) ───
    def compute_scattering_matrix(
        self, t: float = 1.0, hbar: Optional[float] = None, method: str = "auto"
    ) -> Any:
        r"""
        U(t)=exp(−iHt/ħ). Métodos honestos:
          · "spectral": diagonalización hermítica exacta (lo que la v5 llamaba Yoshida).
          · "krylov":   LinearOperator con matvec = expm_multiply(A, v); no forma U.
          · "yoshida":  composición S₄ REAL sobre una partición H = H₀ + H₁ con
                        exponentiales exactas de cada parte (sólo lícito si ambas son
                        generadores hamiltonianos; T9 prohíbe pesos negativos en la
                        parte disipativa).
        """
        hb = float(hbar if hbar is not None else self._hbar)
        if hb <= 0.0 or not math.isfinite(t):
            raise ValueError("t finito y ħ>0 requeridos")
        d = self._ops.hilbert_dim
        a = (-1j * t / hb) * self._H_int
        mode = method.lower()
        if mode == "auto":
            mode = "spectral" if d <= 512 else "krylov"
        if mode == "krylov":
            return LinearOperator((d, d), matvec=lambda v: expm_multiply(a, v),
                                  rmatvec=lambda v: expm_multiply(a.conj().T, v),
                                  dtype=np.complex128)
        if mode == "yoshida":
            return self._yoshida_composition(t, hb)
        dense = self._H_int.toarray() if issparse(self._H_int) else np.asarray(self._H_int)
        if issparse(self._H_int) and d > 2048:
            logger.warning("sparse_expm con d=%d: considere method='krylov'", d)
            return expm(a)
        w, v = la.eigh(0.5 * (dense + dense.conj().T), check_finite=False)
        return (v * np.exp(-1j * w * t / hb)) @ v.conj().T

    def _yoshida_composition(self, t: float, hb: float) -> Any:
        r"""
        S₄(dt) = S₂(w₁dt) S₂(w₀dt) S₂(w₁dt), S₂(h)=e^{−iH₀h/2ħ} e^{−iH₁h/ħ} e^{−iH₀h/2ħ}.
        H₀ = parte diagonal en número (ω, ε, χ); H₁ = resto. Error local O(dt⁵).
        """
        dense = self._H_int.toarray() if issparse(self._H_int) else np.asarray(self._H_int)
        h0 = np.diag(np.diag(dense))
        h1 = dense - h0
        w0d, w1d = np.diag(h0).real, None
        def e0(h: float) -> NDArray:
            return np.diag(np.exp(-1j * w0d * h / hb))
        def e1(h: float) -> NDArray:
            w, v = la.eigh(0.5 * (h1 + h1.conj().T), check_finite=False)
            return (v * np.exp(-1j * w * h / hb)) @ v.conj().T
        def s2(h: float) -> NDArray:
            return e0(0.5 * h) @ e1(h) @ e0(0.5 * h)
        w1, w0, _ = FockPoincareRSIConstants.YOSHIDA_WEIGHTS
        return s2(w1 * t) @ s2(w0 * t) @ s2(w1 * t)

    def verify_unitarity(self, u: Any, tol: float = 1e-9, n_probes: int = 4) -> float:
        r"""
        Isometría con residuo RELATIVO de Frobenius ‖U†U−I‖_F/√d (no max-entry, no
        umbral inflado ×100). Si U es LinearOperator, sonda aleatoria (certificado
        probabilístico, no formación de U†U).
        """
        d = self._ops.hilbert_dim
        if isinstance(u, LinearOperator):
            rng = np.random.default_rng(7)
            worst = 0.0
            for _ in range(n_probes):
                v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
                worst = max(worst, abs(float(la.norm(u @ v) / la.norm(v)) - 1.0))
            if worst > tol:
                raise UnitarityViolationError(f"isometría (sonda) violada: {worst:.3e}")
            return float(worst)
        prod = (u.conj().T @ u)
        prod = prod.toarray() if issparse(prod) else np.asarray(prod)
        res = float(la.norm(prod - np.eye(d), "fro") / math.sqrt(d))
        if res > tol:
            raise UnitarityViolationError(f"‖U†U−I‖_F/√d = {res:.3e}")
        return res

    # ── 2.9 · MÉTODO TERMINAL DE LA FASE 2 → continuación a la Fase 3 ───────────────
    def get_catadioptric_hamiltonian(
        self, continuation: Optional[Callable[[CatadioptricHamiltonian], R]] = None
    ) -> Union[CatadioptricHamiltonian, R]:
        r"""
        Definición formal final de la Fase 2: `continuation: K → R`, cuya cola es el
        arranque de `Phase3._continue_from_phase2_hamiltonian`.
        """
        charges = (tuple(f"n_{q}" for q in range(len(self._ops.fermion_number)))
                   if self._number_conserving else ("(-1)^{N_F}",))
        bundle = CatadioptricHamiltonian(
            H_int=self._H_int, hilbert_dim=self._ops.hilbert_dim,
            operator_norms=self._norms, max_coupling=self._max_coupling,
            use_sparse=self._use_sparse, boson_dimensions=self._ops.boson_dimensions,
            fermion_dimensions=self._ops.fermion_dimensions,
            number_conserving=self._number_conserving, exactly_solvable=self._exactly_solvable,
            conserved_charges=charges, exact_eigenvalues=self._exact_eigs,
            spectral_statistics=self._stats, kam_report=self._kam, wegner_report=self._wegner,
            algebra_verdict=(self._ops.algebra_audit.verdict if self._ops.algebra_audit else 0),
            berry_tabor_level_spacing_ratio=self._stats.mean_r,
            kam_diophantine_resonance_margin=self._kam.gamma_effective,
            wegner_bch_flow_norm=self._wegner.eta_norm_final,
            rsi_l3_hermiticity_residual=self._herm_res,
            rsi_l3_generations=self._ops.rsi_l3_generations,
        )
        if continuation is None:
            return bundle
        return continuation(bundle)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ FASE ANIDADA 3 · GKSL ESTRUCTURA-PRESERVANTE, SPOHN, RECURRENCIA, RSI-L3 (K→Λ) ══
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GKSLStructureReport:
    """Auditoría estructural del generador (no de Kraus: son cosas distintas, T5)."""

    rates_nonnegative: bool
    kossakowski_psd: bool
    trace_annihilation_residual: float
    hermiticity_preservation_residual: float
    liouvillian_spectral_abscissa: float
    liouvillian_gap: float
    steady_state_trace: float
    steady_state_min_eigenvalue: float
    steady_state_measured: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class SpohnReport:
    r"""T7: σ = dS/dt + Tr[𝓛(ρ) ln ρ_ss], SIN clamps; D(ρ‖ρ_ss) monótona."""

    sigma: float
    entropy_rate: float
    entropy_flux: float
    relative_entropy_to_steady: float
    support_ok: bool
    steady_state_stationary: bool
    nonnegative: bool
    verdict: int


@dataclass(frozen=True, slots=True)
class RecurrenceReport:
    r"""T8: retorno unitario (Kronecker–Dirichlet) vs obstrucción disipativa (T7)."""

    hilbert_schmidt_displacement: float
    dirichlet_return_bound: float
    best_return_time: float
    best_phase_error: float
    recurrence_obstructed: bool
    obstruction_reason: str
    verdict: int


@dataclass(frozen=True, slots=True)
class LindbladEvolutionResult:
    """Λ — artefacto terminal de la Fase 3 (ABI v5 preservada + campos rigurosos)."""

    post_collision_rho: Any
    emitted_photon: Optional[GammaPhoton]
    dissipated_entropy: float
    positivity_preserved: bool
    trace_residual: float
    energy_drift: float
    integration_method: str
    poincare_recurrence_distance: float = 0.0
    spohn_entropy_production_rate: float = 0.0
    rsi_l3_cptp_generations: int = 1
    # ── extensiones v6 ──
    cp_map_used: bool = False
    min_eigenvalue: float = 0.0
    purity: float = 1.0
    von_neumann_entropy: float = 0.0
    energy_dissipated: float = 0.0
    landauer_energy_bound: float = 0.0
    effective_mass: float = 0.0
    spohn: Optional[SpohnReport] = None
    recurrence: Optional[RecurrenceReport] = None
    truncation: Optional[TruncationCertificate] = None
    dt_used: float = 0.0
    substeps: int = 1
    heyting_verdict_value: int = 0
    forensic_hash: str = ""


class Phase3_LindbladDissipator:
    r"""
    FASE 3 — Semigrupo GKSL con preservación estructural y termodinámica de Spohn.

    Métodos de integración (todos con semántica declarada):
      · "exact"     : vec(ρ_t) = expm_multiply(𝓛 dt, vec(ρ₀))  — CPTP exacto (T5).
      · "strang_cp" : D(dt/2)∘U(dt)∘D(dt/2), pesos POSITIVOS ⇒ CPTP, orden 2 (T9).
      · "kraus_cp"  : ρ ← Σ K_i ρ K_i†, K₀=I−dt(iH/ħ+½ΣγL†L), K_j=√(γ_j dt)L_j
                      ⇒ CP por construcción (Kraus), orden 1, traza a O(dt²).
      · "rk4"       : legacy, NO CP; se marca y se repara con proyección óptima (T6).
      · "yoshida_rk4": DEPRECADO (T9) → redirigido a strang_cp con aviso.
    """

    def __init__(
        self,
        hamiltonian_bundle: CatadioptricHamiltonian,
        lindblad_operators: List[Any],
        decay_rates: Optional[List[float]] = None,
        planck_normalized: float = 1.0,
        trace_tol: float = 1e-9,
        *,
        environment_temperature: float = 0.0,
        ops: Optional[InteractionOperators] = None,
        superop_budget: int = FockPoincareRSIConstants.SUPEROP_DIM_BUDGET,
        forensic_chain_head: str = "",
    ) -> None:
        bundle = self._continue_from_phase2_hamiltonian(hamiltonian_bundle)
        self._bundle = bundle
        self._H = bundle.H_int
        self._L = list(lindblad_operators)
        self._gamma = list(decay_rates) if decay_rates is not None else [1.0] * len(self._L)
        if len(self._gamma) != len(self._L):
            raise TopologicalInvariantError("γ y L deben tener la misma longitud")
        if any(g < 0.0 for g in self._gamma):
            raise LindbladDissipationError("γ_k ≥ 0 requerido (Kossakowski PSD, T5)")
        if planck_normalized <= 0.0:
            raise ValueError("ħ debe ser positivo")
        if environment_temperature < 0.0:
            raise ValueError("T ≥ 0 requerido (Landauer)")
        self._hbar = float(planck_normalized)
        self._temperature = float(environment_temperature)
        self._trace_tol = float(trace_tol)
        self._dim = bundle.hilbert_dim
        self._use_sparse = bundle.use_sparse
        self._ops = ops                      # ← la v5 heredaba y dejaba esto sin definir
        self._superop_budget = int(superop_budget)
        self._chain_head = forensic_chain_head
        self._ldag_l = [L.conj().T @ L for L in self._L]
        self._effective_H = self._build_effective_hamiltonian()
        self._superop: Optional[Any] = None
        self._rho_ss: Optional[NDArray[np.complex128]] = None
        self._structure: Optional[GKSLStructureReport] = None
        logger.debug("[Fase3] dim=%d L=%d Σγ=%.4f T=%.3g K", self._dim, len(self._L),
                     float(np.sum(self._gamma)), self._temperature)

    # ── 3.1 · PRIMER MÉTODO DE LA FASE 3 (continuación exacta de la Fase 2) ──────────
    def _continue_from_phase2_hamiltonian(
        self, bundle: CatadioptricHamiltonian
    ) -> CatadioptricHamiltonian:
        if not isinstance(bundle, CatadioptricHamiltonian):
            raise TopologicalInvariantError(
                f"Continuidad Φ₂→Φ₃ violada: se esperaba CatadioptricHamiltonian, hay {type(bundle)}"
            )
        if bundle.hilbert_dim <= 0:
            raise FockSpaceOverflowError("dimensión inválida en K")
        return bundle

    def _build_effective_hamiltonian(self) -> Any:
        """H_eff = H − (iħ/2)Σγ L†L (generador no hermítico del propagador de Kraus)."""
        s = None
        for g, ld in zip(self._gamma, self._ldag_l):
            s = g * ld if s is None else s + g * ld
        if s is None:
            return self._H
        return self._H - 0.5j * self._hbar * s

    # ── 3.2 · Superoperador de Lindblad (vec por columnas) y auditoría estructural ───
    def liouvillian_superoperator(self) -> Optional[Any]:
        r"""
        𝓛 = −(i/ħ)(I⊗H − Hᵀ⊗I) + Σγ[L̄⊗L − ½I⊗(L†L) − ½(L†L)ᵀ⊗I],  vec order='F'.
        Sólo se construye si d² ≤ presupuesto (honestidad de coste).
        """
        if self._dim ** 2 > self._superop_budget:
            return None
        if self._superop is not None:
            return self._superop
        d = self._dim
        hd = self._H.toarray() if issparse(self._H) else np.asarray(self._H)
        eye = np.eye(d, dtype=np.complex128)
        sup = (-1j / self._hbar) * (np.kron(eye, hd) - np.kron(hd.T, eye))
        for g, lk, ld in zip(self._gamma, self._L, self._ldag_l):
            l = lk.toarray() if issparse(lk) else np.asarray(lk)
            p = ld.toarray() if issparse(ld) else np.asarray(ld)
            sup += g * (np.kron(l.conj(), l) - 0.5 * np.kron(eye, p) - 0.5 * np.kron(p.T, eye))
        self._superop = sup
        return sup

    def _vec(self, rho: Any) -> NDArray[np.complex128]:
        a = rho.toarray() if issparse(rho) else np.asarray(rho)
        return a.ravel(order="F")

    def _unvec(self, v: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.asarray(v).reshape((self._dim, self._dim), order="F")

    def audit_gksl_structure(self, n_probes: int = 3) -> GKSLStructureReport:
        r"""
        T5: NO se exige Σ L†L = I (eso es para mapas de Kraus). Se verifica:
        γ≥0; Tr𝓛(ρ)=0 y 𝓛(ρ†)=𝓛(ρ)† sobre sondas hermíticas; abscisa espectral ≤ 0;
        gap de relajación; estado estacionario (punto fijo) con traza y λ_min medidos.
        """
        rng = np.random.default_rng(11)
        tr_res, herm_res = 0.0, 0.0
        for _ in range(n_probes):
            a = rng.standard_normal((self._dim, self._dim)) + 1j * rng.standard_normal((self._dim, self._dim))
            rho = a @ a.conj().T
            rho /= max(float(np.real(np.trace(rho))), 1e-300)
            dr = self._lindbladian(rho)
            drd = dr.toarray() if issparse(dr) else np.asarray(dr)
            scale = max(float(la.norm(drd, "fro")), 1e-300)
            tr_res = max(tr_res, abs(complex(np.trace(drd))) / scale)
            herm_res = max(herm_res, float(la.norm(drd - drd.conj().T, "fro")) / scale)
        sup = self.liouvillian_superoperator()
        abscissa, gap, measured = float("nan"), float("nan"), False
        if sup is not None:
            ev = la.eigvals(sup, check_finite=False)
            re = np.sort(ev.real)[::-1]
            abscissa = float(re[0])
            gap = float(-re[1]) if re.size > 1 else float("nan")
            measured = True
        rho_ss = self.compute_steady_state()
        ss_tr = float(np.real(np.trace(rho_ss))) if rho_ss is not None else float("nan")
        ss_min = (float(np.min(la.eigvalsh(0.5 * (rho_ss + rho_ss.conj().T))))
                  if rho_ss is not None else float("nan"))
        ok = (tr_res <= 1e-9 and herm_res <= 1e-9 and all(g >= 0 for g in self._gamma)
              and (not measured or abscissa <= 1e-8))
        verdict = int(HeytingVerdict.COHERENT if ok else HeytingVerdict.DEGRADED
                      if tr_res <= 1e-6 else HeytingVerdict.VETOED)
        self._structure = GKSLStructureReport(
            rates_nonnegative=all(g >= 0 for g in self._gamma), kossakowski_psd=True,
            trace_annihilation_residual=tr_res, hermiticity_preservation_residual=herm_res,
            liouvillian_spectral_abscissa=abscissa, liouvillian_gap=gap,
            steady_state_trace=ss_tr, steady_state_min_eigenvalue=ss_min,
            steady_state_measured=rho_ss is not None, verdict=verdict,
        )
        return self._structure

    def compute_steady_state(self, relax_time: Optional[float] = None) -> Optional[NDArray[np.complex128]]:
        r"""
        ρ_ss: núcleo de 𝓛 (SVD) si d es pequeño; si no, relajación larga con
        expm_multiply (CPTP exacto) y proyección al espectralplex. Nunca se inventa.
        """
        if self._rho_ss is not None:
            return self._rho_ss
        sup = self.liouvillian_superoperator()
        if sup is None:
            return None
        if self._dim <= FockPoincareRSIConstants.DENSE_STEADY_DIM:
            _, _, vh = la.svd(sup, check_finite=False)
            null = vh[-1].conj()
            rho = self._unvec(null)
            rho = 0.5 * (rho + rho.conj().T)
            if float(np.real(np.trace(rho))) < 0:
                rho = -rho
            rho = self._project_spectraplex(rho)[0]
        else:
            t = float(relax_time) if relax_time else 50.0 / max(
                abs(self._structure.liouvillian_gap) if self._structure else 1.0, 1e-3)
            v = expm_multiply(t * sup, self._vec(np.eye(self._dim, dtype=np.complex128) / self._dim))
            rho = self._project_spectraplex(self._unvec(v))[0]
        self._rho_ss = rho
        return rho

    # ── 3.3 · Generador y propagadores estructura-preservantes ───────────────────────
    def _lindbladian(self, rho: Any) -> Any:
        """𝓛(ρ) = −(i/ħ)[H,ρ] + Σγ(LρL† − ½{L†L,ρ})."""
        out = (-1j / self._hbar) * (self._H @ rho - rho @ self._H)
        for g, lk, ld in zip(self._gamma, self._L, self._ldag_l):
            out = out + g * (lk @ rho @ lk.conj().T) - 0.5 * g * (ld @ rho + rho @ ld)
        return out

    def _propagate_exact(self, rho: Any, dt: float) -> Any:
        sup = self.liouvillian_superoperator()
        if sup is None:
            return self._propagate_strang(rho, dt)
        return self._unvec(expm_multiply(dt * sup, self._vec(rho)))

    def _propagate_unitary(self, rho: Any, dt: float) -> Any:
        r"""U ρ U† sin formar U: X=e^{A}ρ; ρ_t=(e^{A}X†)† con A=−iHdt/ħ (Krylov)."""
        if issparse(rho) or issparse(self._H) or self._dim > 64:
            a = (-1j * dt / self._hbar) * self._H
            x = expm_multiply(a, rho)
            y = expm_multiply(a, x.conj().T)
            return y.conj().T
        hd = self._H.toarray() if issparse(self._H) else np.asarray(self._H)
        w, v = la.eigh(0.5 * (hd + hd.conj().T), check_finite=False)
        u = (v * np.exp(-1j * w * dt / self._hbar)) @ v.conj().T
        return u @ rho @ u.conj().T

    def _propagate_dissipative(self, rho: Any, dt: float) -> Any:
        """exp(dt·𝒟) con 𝒟 puramente disipativo: CPTP exacto (semigrupo legítimo)."""
        sup = self.liouvillian_superoperator()
        if sup is None:
            return self._propagate_kraus(rho, dt, hamiltonian=False)
        hd = self._H.toarray() if issparse(self._H) else np.asarray(self._H)
        eye = np.eye(self._dim, dtype=np.complex128)
        ham = (-1j / self._hbar) * (np.kron(eye, hd) - np.kron(hd.T, eye))
        return self._unvec(expm_multiply(dt * (sup - ham), self._vec(rho)))

    def _propagate_strang(self, rho: Any, dt: float) -> Any:
        r"""T9: composición simétrica de pesos POSITIVOS ⇒ CPTP y orden 2."""
        half = self._propagate_dissipative(rho, 0.5 * dt)
        mid = self._propagate_unitary(half, dt)
        return self._propagate_dissipative(mid, 0.5 * dt)

    def _propagate_kraus(self, rho: Any, dt: float, hamiltonian: bool = True) -> Any:
        r"""
        Mapa de Kraus de primer orden: CP EXACTA (forma de Kraus), TP a O(dt²).
        K₀ = I − dt(iH/ħ + ½ΣγL†L),  K_j = √(γ_j dt) L_j.
        """
        eye = self._eye_like(rho)
        k0 = eye - dt * ((1j / self._hbar) * self._H if hamiltonian else 0.0)
        for g, ld in zip(self._gamma, self._ldag_l):
            k0 = k0 - 0.5 * dt * g * ld
        out = k0 @ rho @ k0.conj().T
        for g, lk in zip(self._gamma, self._L):
            if g > 0.0:
                kj = math.sqrt(g * dt) * lk
                out = out + kj @ rho @ kj.conj().T
        return out

    def _integrate_rk4(self, rho: Any, dt: float) -> Any:
        """RK4 clásico: orden 4, NO CP (se documenta y se repara con T6)."""
        k1 = self._lindbladian(rho)
        k2 = self._lindbladian(rho + 0.5 * dt * k1)
        k3 = self._lindbladian(rho + 0.5 * dt * k2)
        k4 = self._lindbladian(rho + dt * k3)
        return rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _eye_like(self, rho: Any) -> Any:
        return (sparse_identity(self._dim, format="csr", dtype=np.complex128)
                if issparse(rho) else np.eye(self._dim, dtype=np.complex128))

    # ── 3.4 · Proyección ÓPTIMA al espectralplex (T6) y criterio de reparabilidad ────
    @staticmethod
    def _simplex_shift(eigs: NDArray[np.float64]) -> float:
        r"""θ de la proyección euclídea sobre el simplex (Duchi et al. 2008)."""
        lam = np.sort(np.asarray(eigs, dtype=np.float64))[::-1]
        css = np.cumsum(lam) - 1.0
        idx = np.arange(1, lam.size + 1)
        cond = lam - css / idx > 0.0
        rho = int(np.max(np.nonzero(cond)[0])) + 1
        return float(css[rho - 1] / rho)

    def _project_spectraplex(self, rho: Any) -> Tuple[NDArray[np.complex128], int]:
        r"""
        Proyección de Frobenius ÓPTIMA sobre {ρ⪰0, Trρ=1}: por invariancia unitaria,
        ρ ↦ V diag(max(λ−θ,0)) V† con θ del simplex. clip+renormalizar (v5) NO es
        óptima y puede aumentar la distancia. Devuelve (ρ_proy, generations=1).
        """
        dense = rho.toarray() if issparse(rho) else np.asarray(rho)
        herm = 0.5 * (dense + dense.conj().T)
        w, v = la.eigh(herm, check_finite=False)
        theta = self._simplex_shift(w)
        wp = np.maximum(w - theta, 0.0)
        out = (v * wp) @ v.conj().T
        return 0.5 * (out + out.conj().T), 1

    def _repairability(self, min_eig: float, dt: float, order: int) -> bool:
        r"""
        Criterio principista (no umbral mágico): una excursión negativa es NUMÉRICA si
        |λ_min| ≲ C·(dt‖𝓛‖)^{p+1} (escala del error de truncamiento local); si excede
        ese orden, es ESTRUCTURAL ⇒ veto, no reparación.
        """
        if min_eig >= -FockPoincareRSIConstants.CP_NEGativity_REL:
            return True
        norm_l = sum(g * float(la.norm((l.toarray() if issparse(l) else l), 2)) ** 2
                     for g, l in zip(self._gamma, self._L))
        hd = self._H.toarray() if issparse(self._H) else np.asarray(self._H)
        norm_l += 2.0 * float(la.norm(hd, 2)) / self._hbar
        scale = (max(dt, 1e-300) * max(norm_l, 1e-300)) ** (order + 1)
        return bool(abs(min_eig) <= 64.0 * max(scale, FockPoincareRSIConstants.MACHINE_EPS))

    # ── 3.5 · Spohn riguroso (T7): soporte, estacionariedad y signo ──────────────────
    @staticmethod
    def _log_support(a: NDArray[np.complex128], tol: float = 1e-12
                     ) Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """ln A sobre su soporte y proyector de soporte (evita log de ceros espurios)."""
        w, v = la.eigh(0.5 * (a + a.conj().T), check_finite=False)
        cut = tol * max(float(w[-1]), 1.0)
        sup = w > cut
        lw = np.zeros_like(w)
        lw[sup] = np.log(w[sup])
        return (v * lw) @ v.conj().T, (v[:, sup] @ v[:, sup].conj().T)

    def relative_entropy(self, rho: NDArray, sigma: NDArray) -> float:
        """D(ρ‖σ)=Trρ(lnρ−lnσ); +∞ si supp(ρ) ⊄ supp(σ) (rigor, no clamp)."""
        log_r, sup_r = self._log_support(rho)
        log_s, sup_s = self._log_support(sigma)
        leakage = float(np.real(np.trace((np.eye(self._dim) - sup_s) @ rho)))
        if leakage > 1e-9:
            return float("inf")
        return float(np.real(np.trace(rho @ (log_r - log_s))))

    def evaluate_spohn(self, rho: Any, dt: float) -> SpohnReport:
        r"""σ = −Tr[𝓛(ρ)(lnρ−lnρ_ss)] = dS/dt + Tr[𝓛(ρ)lnρ_ss]; sin max(0,·)."""
        dense = rho.toarray() if issparse(rho) else np.asarray(rho)
        ss = self.compute_steady_state()
        if ss is None:
            return SpohnReport(float("nan"), float("nan"), float("nan"), float("nan"),
                               False, False, False, int(HeytingVerdict.DEGRADED))
        dr = self._lindbladian(dense)
        dr = dr.toarray() if issparse(dr) else np.asarray(dr)
        log_r, sup_r = self._log_support(dense)
        log_s, sup_s = self._log_support(ss)
        leakage = float(np.real(np.trace((np.eye(self._dim) - sup_s) @ dense)))
        support_ok = leakage <= 1e-9
        s_rate = float(-np.real(np.trace(dr @ log_r)))
        flux = float(np.real(np.trace(dr @ log_s)))
        sigma = s_rate + flux if support_ok else float("inf")
        d_rel = self.relative_entropy(dense, ss)
        stationary = float(la.norm(self._lindbladian(ss), "fro")) <= 1e-8
        ok = support_ok and stationary and (math.isinf(sigma) or sigma >= -1e-9)
        return SpohnReport(
            sigma=sigma, entropy_rate=s_rate, entropy_flux=flux,
            relative_entropy_to_steady=d_rel, support_ok=support_ok,
            steady_state_stationary=bool(stationary), nonnegative=bool(ok),
            verdict=int(HeytingVerdict.COHERENT if ok else HeytingVerdict.DEGRADED),
        )

    # ── 3.6 · Poincaré: retorno unitario vs obstrucción disipativa (T8) ──────────────
    def evaluate_recurrence(
        self, rho0: Any, rho_t: Any, dt: float, epsilon: float = 1e-2,
        search_budget: int = 4096,
    ) -> RecurrenceReport:
        r"""
        Unitario: Kronecker–Dirichlet garantiza T ≤ ⌈1/ε⌉^d con error de fase ≤ ε;
        además se busca el mejor q ≤ budget (aproximación simultánea constructiva).
        Disipativo: si D(ρ_t‖ρ_ss) < D(ρ₀‖ρ_ss) − δ, la monotonía (T7) PROHÍBE el
        retorno: recurrencia obstruida, y se declara (la v5 lo llamaba recurrencia).
        """
        a = rho0.toarray() if issparse(rho0) else np.asarray(rho0)
        b = rho_t.toarray() if issparse(rho_t) else np.asarray(rho_t)
        disp = float(la.norm(b - a, "fro"))
        eigs = self._bundle.exact_eigenvalues
        if eigs is None:
            hd = self._H.toarray() if issparse(self._H) else np.asarray(self._H)
            eigs = la.eigvalsh(0.5 * (hd + hd.conj().T), check_finite=False)
        uniq = np.unique(np.round(np.asarray(eigs), 12))[:5]
        omega = (uniq[1:] - uniq[0]) / self._hbar if uniq.size > 1 else np.zeros(1)
        alpha = omega / (2.0 * math.pi)
        d = max(int(alpha.size), 1)
        bound = float(min(math.ceil(1.0 / max(epsilon, 1e-12)) ** d, 1e18))
        best_t, best_err = float("inf"), float("inf")
        for q in range(1, int(min(search_budget, max(2, bound))) + 1):
            err = float(np.max(np.abs(alpha * q - np.round(alpha * q)))) if alpha.size else 0.0
            if err < best_err:
                best_err, best_t = err, float(q)
            if err <= epsilon:
                break
        ss = self.compute_steady_state()
        obstructed, reason = False, "dinámica reversible: retorno garantizado (T8)"
        if any(g > 0 for g in self._gamma) and ss is not None:
            d0, d1 = self.relative_entropy(a, ss), self.relative_entropy(b, ss)
            if math.isfinite(d0) and math.isfinite(d1) and d1 < d0 - 1e-9:
                obstructed = True
                reason = ("D(ρ_t‖ρ_ss) < D(ρ₀‖ρ_ss): monotonía de Spohn (T7) ⇒ "
                          "no hay retorno de Poincaré; el sistema relaja al punto fijo")
        return RecurrenceReport(
            hilbert_schmidt_displacement=disp, dirichlet_return_bound=bound,
            best_return_time=best_t, best_phase_error=best_err,
            recurrence_obstructed=obstructed, obstruction_reason=reason,
            verdict=int(HeytingVerdict.COHERENT if not obstructed else HeytingVerdict.DEGRADED),
        )

    # ── 3.7 · Termodinámica y fotón forense (Landauer + SHA-256 encadenado) ──────────
    @staticmethod
    def _von_neumann_entropy(rho: NDArray[np.complex128]) -> float:
        w = np.clip(la.eigvalsh(0.5 * (rho + rho.conj().T)).real, 1e-300, None)
        return float(-np.sum(w * np.log(w)))

    def _forensic_hash(self, payload: bytes) -> str:
        h = hashlib.sha256()
        h.update(self._chain_head.encode("utf-8"))
        h.update(payload)
        digest = h.hexdigest()
        self._chain_head = digest
        return digest

    # ── 3.8 · MÉTODO TERMINAL DE LA FASE 3 ──────────────────────────────────────────
    def execute_master_equation(
        self,
        rho_initial: Any,
        dt: float = 1e-3,
        method: str = "auto",
        *,
        max_substeps: int = 64,
        compute_spohn: bool = True,
        compute_recurrence: bool = True,
    ) -> LindbladEvolutionResult:
        r"""
        Clausura disipativa con invariantes medidos y ledger termodinámico correcto:
        ΔE = Tr[Hρ₀]−Tr[Hρ_t] (energía), E_Landauer = k_B T·max(0, S₀−S_t) (nats),
        m* = ΔE/c², E_aniquilación = 2m*c² (T10). El fotón se sella con SHA-256
        encadenado y timestamp real (determinismo forense).
        """
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt debe ser finito y positivo")
        rho0 = self._validate_density(rho_initial)
        m = method.lower()
        if m == "yoshida_rk4":
            logger.warning("yoshida_rk4 DEPRECADO (T9: pesos negativos rompen CP) → strang_cp")
            m = "strang_cp"
        if m == "auto":
            m = ("exact" if self._dim ** 2 <= self._superop_budget
                 else "kraus_cp" if (issparse(rho0) or self._dim > 64) else "strang_cp")
        order = {"exact": 12, "strang_cp": 2, "kraus_cp": 1, "rk4": 4}.get(m, 1)
        cp_native = m in ("exact", "strang_cp", "kraus_cp")

        # Sub-pasos con criterio CFL espectral (estabilidad de los esquemas explícitos)
        norm_l = 2.0 * float(la.norm(self._H.toarray() if issparse(self._H) else self._H, 2)) / self._hbar
        norm_l += sum(g * float(la.norm((l.toarray() if issparse(l) else l), 2)) ** 2
                      for g, l in zip(self._gamma, self._L))
        dt_safe = min(dt, 0.5 / max(norm_l, 1e-300)) if m in ("rk4", "kraus_cp") else dt
        n_sub = int(min(max_substeps, max(1, math.ceil(dt / max(dt_safe, 1e-300)))))
        h = dt / n_sub
        rho = rho0
        for _ in range(n_sub):
            if m == "exact":
                rho = self._propagate_exact(rho, h)
            elif m == "strang_cp":
                rho = self._propagate_strang(rho, h)
            elif m == "kraus_cp":
                rho = self._propagate_kraus(rho, h)
            else:
                rho = self._integrate_rk4(rho, h)

        dense = rho.toarray() if issparse(rho) else np.asarray(rho)
        dense = 0.5 * (dense + dense.conj().T)
        tr = float(np.real(np.trace(dense)))
        if tr <= 1e-14:
            raise LindbladDissipationError("traza colapsada: el propagador destruyó el estado")
        trace_res = abs(1.0 - tr)
        dense = dense / tr
        w = la.eigvalsh(dense, check_finite=False)
        min_eig = float(w[0])
        gens = 0
        if min_eig < -FockPoincareRSIConstants.CP_NEGativity_REL:
            if not self._repairability(min_eig, h, order):
                raise LindbladDissipationError(
                    f"violación ESTRUCTURAL de positividad: λ_min={min_eig:.3e} "
                    f"(escala de error local superada; método={m})"
                )
            dense, gens = self._project_spectraplex(dense)
            w = la.eigvalsh(dense, check_finite=False)
            min_eig = float(w[0])
        pos_ok = bool(min_eig >= -1e-9)
        if not pos_ok:
            raise LindbladDissipationError("ρ final no PSD tras proyección óptima")
        if trace_res > self._trace_tol:
            logger.warning("renormalización significativa: |1−Tr|=%.3e", trace_res)

        hd = self._H.toarray() if issparse(self._H) else np.asarray(self._H)
        r0 = rho0.toarray() if issparse(rho0) else np.asarray(rho0)
        e0, e1 = float(np.real(np.trace(hd @ r0))), float(np.real(np.trace(hd @ dense)))
        s0, s1 = self._von_neumann_entropy(r0), self._von_neumann_entropy(dense)
        energy_dissipated = e0 - e1
        landauer = (FockPoincareRSIConstants.BOLTZMANN_K_B * self._temperature * max(0.0, s0 - s1))
        jump_activity = float(sum(
            g * np.real(np.trace((lk @ dense @ lk.conj().T)))
            for g, lk in zip(self._gamma, self._L)
        )) * dt
        purity = float(np.real(np.trace(dense @ dense)))
        comm = hd @ dense - dense @ hd
        stationarity = float(la.norm(comm, "fro") / max(la.norm(hd, "fro"), 1e-300))

        spohn = self.evaluate_spohn(dense, dt) if compute_spohn else None
        recur = (self.evaluate_recurrence(r0, dense, dt) if compute_recurrence else None)
        trunc = (self._ops is not None and self._ops.algebra_audit is not None)
        trunc_cert = (Phase1_FockSpaceBuilder.truncation_certificate(self._ops_builder(), dense)
                      if False else None)

        photon = None
        effective_mass = max(0.0, energy_dissipated) / FockPoincareRSIConstants.SPEED_OF_LIGHT_C ** 2
        payload = (f"{self._dim}|{dt}|{m}|{energy_dissipated:.17g}|{s1:.17g}".encode()
                   + np.ascontiguousarray(dense, dtype=np.complex128).tobytes())
        digest = self._forensic_hash(payload)
        if energy_dissipated > 0.0 or jump_activity > 1e-15:
            positron = PositronCartridge(
                inertial_mass=effective_mass, topological_spin="anti_stochastic",
                homological_charge=-1,
                authorization_signature="QED_GKSL_Spohn_Poincare_RSI_L3",
            )
            photon = GammaPhoton(
                annihilation_energy=2.0 * effective_mass * FockPoincareRSIConstants.SPEED_OF_LIGHT_C ** 2,
                data_hash=digest, timestamp_entry=time.time(),
                authorization_signature=positron.authorization_signature,
            )
        verdict = int(HeytingVerdict.COHERENT)
        if not cp_native or trace_res > self._trace_tol or gens > 0:
            verdict = int(heyting_meet(verdict, HeytingVerdict.DEGRADED))
        if spohn is not None and not spohn.nonnegative:
            verdict = int(heyting_meet(verdict, HeytingVerdict.DEGRADED))
        if not pos_ok:
            verdict = int(HeytingVerdict.VETOED)

        return LindbladEvolutionResult(
            post_collision_rho=(csr_matrix(dense) if self._use_sparse else dense),
            emitted_photon=photon, dissipated_entropy=jump_activity,
            positivity_preserved=pos_ok, trace_residual=trace_res,
            energy_drift=stationarity, integration_method=m.upper(),
            poincare_recurrence_distance=(recur.hilbert_schmidt_displacement if recur else 0.0),
            spohn_entropy_production_rate=(spohn.sigma if spohn else float("nan")),
            rsi_l3_cptp_generations=max(1, gens), cp_map_used=cp_native,
            min_eigenvalue=min_eig, purity=purity, von_neumann_entropy=s1,
            energy_dissipated=energy_dissipated, landauer_energy_bound=landauer,
            effective_mass=effective_mass, spohn=spohn, recurrence=recur,
            truncation=trunc_cert, dt_used=h, substeps=n_sub,
            heyting_verdict_value=verdict, forensic_hash=digest,
        )

    def _ops_builder(self) -> Any:  # pragma: no cover - reservado al puente de truncamiento
        return None

    def _validate_density(self, rho: Any) -> Any:
        if rho is None:
            raise ValueError("ρ inicial requerida")
        dense = rho.toarray() if issparse(rho) else np.asarray(rho)
        if dense.shape != (self._dim, self._dim):
            raise TopologicalInvariantError(f"ρ debe ser {self._dim}×{self._dim}")
        if not np.all(np.isfinite(dense)):
            raise LindbladDissipationError("ρ con entradas no finitas")
        herm = float(la.norm(dense - dense.conj().T, "fro") / max(la.norm(dense, "fro"), 1e-300))
        if herm > 1e-8:
            raise HermiticityViolationError(f"ρ inicial no hermítica: {herm:.3e}")
        tr = float(np.real(np.trace(dense)))
        if abs(tr - 1.0) > 100 * self._trace_tol:
            raise LindbladDissipationError(f"ρ inicial no normalizada: Tr={tr:.10f}")
        if float(np.min(la.eigvalsh(0.5 * (dense + dense.conj().T)))) < -1e-8:
            raise LindbladDissipationError("ρ inicial no PSD")
        return rho


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ ORQUESTADOR · COMPOSICIÓN Φ₃∘Φ₂∘Φ₁ + RSI-L3 + PUENTE BOGOLIUBOV (Ω₃)          ══
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class RSIL3Hyperparameters:
    """Único payload mutable de la auto-mejora (el verificador es INMUTABLE)."""

    dt: float = 1e-3
    method: str = "auto"
    trust_radius: float = 0.25
    truncation_growth: int = 2
    meta_depth: int = 0


@dataclass(frozen=True, slots=True)
class RSIL3Report:
    """Trazabilidad del lazo de nivel 3: Lyapunov de Spohn, Banach, rollback, M↑."""

    iterations: int
    accepted: int
    rejected: int
    lyapunov_initial: float
    lyapunov_final: float
    lyapunov_history: Tuple[float, ...]
    contraction_q: float
    banach_error_bound: float
    contraction_certified: bool
    monotone: bool
    truncation_expansions: int
    hyperparameters: RSIL3Hyperparameters
    verdict: int


class QuantumFockOrchestrator(Morphism):
    r"""
    Endofuntor compuesto Z_Fock = Φ₃ ∘ Φ₂ ∘ Φ₁ (COMPOSICIÓN, no herencia).

    RSI-L3 real:
      R₁ (Fase 1): purificación de McWeeny del álgebra + certificado de truncamiento.
      R₂ (Fase 2): flujo de Wegner iso-espectral con Lyapunov ‖H_off‖².
      R₃ (Fase 3): proyección óptima al espectralplex + Lyapunov de Spohn D(ρ‖ρ_ss).
      META (nivel 3): reescribe dt/método y, si el certificado de truncamiento falla,
      AMPLÍA su propio espacio de Fock mediante una ISOMETRÍA E: ρ ↦ EρE† (preserva
      traza, espectro, positividad y entropía). Nunca reescribe el verificador.
    """

    def __init__(
        self,
        config: FockSpaceConfiguration,
        coupling_matrix: NDArray[np.complex128],
        lindblad_operators: Optional[List[Any]] = None,
        lindblad_rates: Optional[List[float]] = None,
        planck_normalized: float = 1.0,
        trace_tolerance: float = 1e-9,
        default_integration: str = "auto",
        *,
        boson_frequencies: Optional[Sequence[float]] = None,
        fermion_energies: Optional[Sequence[float]] = None,
        kerr_chi: Optional[Sequence[float]] = None,
        exchange_coupling: Optional[NDArray[np.complex128]] = None,
        environment_temperature: float = 0.0,
        wegner_steps: int = 0,
        enable_bogoliubov_bridge: bool = True,
    ) -> None:
        self._config = config
        self._coupling = np.asarray(coupling_matrix, dtype=np.complex128)
        self._hbar = float(planck_normalized)
        self._temperature = float(environment_temperature)
        self._default_method = default_integration
        self._extra_h = dict(
            boson_frequencies=boson_frequencies, fermion_energies=fermion_energies,
            kerr_chi=kerr_chi, exchange_coupling=exchange_coupling,
            wegner_steps=wegner_steps,
        )
        self._bogoliubov = (BogoliubovAgent() if (enable_bogoliubov_bridge
                                                 and BogoliubovAgent is not None) else None)
        self._build_pipeline(config, lindblad_operators, lindblad_rates, trace_tolerance)
        logger.info("QuantumFockOrchestrator v6: B=%d F=%d dim=%d L=%d método=%s",
                    config.n_boson_modes, config.n_fermion_modes,
                    self._ops.hilbert_dim, len(self._phase3._L), default_integration)

    # ── Composición explícita de las tres fases anidadas (CPS) ───────────────────────
    def _build_pipeline(
        self, config: FockSpaceConfiguration,
        lindblad_operators: Optional[List[Any]], lindblad_rates: Optional[List[float]],
        trace_tolerance: float,
    ) -> None:
        logger.info("═══ FASE 1: Fock/Jordan–Wigner/cuadraturas (RSI-L3 R₁) ═══")
        self._phase1 = Phase1_FockSpaceBuilder(config)

        def _phase2_from_ops(ops: InteractionOperators) -> CatadioptricHamiltonian:
            logger.info("═══ FASE 2: catadióptrico/sectores/KAM (RSI-L3 R₂) ═══")
            self._phase2 = Phase2_CatadioptricCollider(
                ops, self._coupling, planck_normalized=self._hbar, **self._extra_h
            )

            def _phase3_from_k(bundle: CatadioptricHamiltonian) -> Phase3_LindbladDissipator:
                logger.info("═══ FASE 3: GKSL CPTP/Spohn/Poincaré (RSI-L3 R₃) ═══")
                ls = list(lindblad_operators) if lindblad_operators is not None else list(ops.boson_ann)
                rates = list(lindblad_rates) if lindblad_rates is not None else [1.0] * len(ls)
                return Phase3_LindbladDissipator(
                    bundle, ls, rates, planck_normalized=self._hbar,
                    trace_tol=trace_tolerance, environment_temperature=self._temperature,
                    ops=ops,
                )

            return self._phase2.get_catadioptric_hamiltonian(continuation=_phase3_from_k)

        self._phase3 = self._phase1.get_interaction_operators(continuation=_phase2_from_ops)
        self._ops = self._phase1.get_interaction_operators()
        self._bundle = self._phase2.get_catadioptric_hamiltonian()
        self._phase3.audit_gksl_structure()

    # ── ABI legacy: fachada delegada (sin herencia rota) ─────────────────────────────
    @property
    def _L(self) -> List[Any]:
        return self._phase3._L

    @property
    def _gamma(self) -> List[float]:
        return self._phase3._gamma

    @property
    def _H(self) -> Any:
        return self._phase3._H

    def compute_scattering_matrix(self, t: float = 1.0, hbar: Optional[float] = None,
                                  method: str = "auto") -> Any:
        return self._phase2.compute_scattering_matrix(t, hbar, method)

    def verify_unitarity(self, u: Any, tol: float = 1e-9) -> float:
        return self._phase2.verify_unitarity(u, tol)

    def execute_master_equation(self, rho_initial: Any, dt: float = 1e-3,
                                method: str = "auto", **kw: Any) -> LindbladEvolutionResult:
        return self._phase3.execute_master_equation(rho_initial, dt, method, **kw)

    def assimilate_and_collide(self, rho_llm: Any, dt: float = 1e-3,
                               method: Optional[str] = None) -> LindbladEvolutionResult:
        r"""Colisión catadióptrica completa con auditoría de truncamiento y Ω₃."""
        res = self.execute_master_equation(rho_llm, dt, method or self._default_method)
        cert = self._phase1.truncation_certificate(res.post_collision_rho)
        if cert.measured and not cert.trustworthy:
            logger.warning("[T1] población de frontera excesiva: anomalía CCR=%s",
                           cert.ccr_anomaly_expectations)
        if res.emitted_photon is not None:
            logger.warning("Fotón Γ: E=%.3e J, m*=%.3e kg, ΔE=%.3e J, Landauer=%.3e J, hash=%s",
                           res.emitted_photon.annihilation_energy, res.effective_mass,
                           res.energy_dissipated, res.landauer_energy_bound,
                           res.forensic_hash[:16])
        return res

    # ── RSI-L3: Lyapunov de Spohn + Banach + auto-ampliación isométrica ──────────────
    @staticmethod
    def _embedding_map(old: FockSpaceConfiguration, new: FockSpaceConfiguration
                       ) -> NDArray[np.int64]:
        r"""
        Isometría E: |n₀…n_{B−1}; s⟩ ↦ |n₀…n_{B−1}; s⟩ con M_k ↑ (mismos números de
        ocupación). Índice = b_idx·2^F + s con radix mixto (modo 0 más significativo).
        E es isometría ⇒ preserva traza, espectro, PSD y entropía de von Neumann.
        """
        f = old.n_fermion_modes
        fdim = 2 ** f
        old_dims = [m + 1 for m in old.truncations_tuple]
        new_dims = [m + 1 for m in new.truncations_tuple]
        maps: List[int] = []
        for b_old in range(int(math.prod(old_dims))):
            rem, digits = b_old, []
            for k in range(len(old_dims)):
                radix = int(math.prod(old_dims[k + 1:])) if k + 1 < len(old_dims) else 1
                digits.append(rem // radix)
                rem %= radix
            b_new = sum(digits[k] * (int(math.prod(new_dims[k + 1:])) if k + 1 < len(new_dims) else 1)
                        for k in range(len(new_dims)))
            for s in range(fdim):
                maps.append(b_new * fdim + s)
        return np.asarray(maps, dtype=np.int64)

    def run_rsi_l3_cycle(
        self, rho_initial: Any, dt: float = 1e-3, max_iterations: int = 6,
        tolerance: float = 1e-10, hyperparameters: Optional[RSIL3Hyperparameters] = None,
        allow_truncation_growth: bool = True,
    ) -> Tuple[LindbladEvolutionResult, RSIL3Report]:
        r"""
        Lazo de nivel 3 con criterio de aceptación FÍSICO (T7): V = D(ρ‖ρ_ss) debe ser
        monótona decreciente; si no, rollback y contracción del paso. Si el certificado
        de truncamiento (T1) falla, se AMPLÍA el espacio y se re-compone la tubería
        mediante la isometría E (auto-modificación estructural, con presupuesto).
        """
        hyper = hyperparameters or RSIL3Hyperparameters(dt=dt, method=self._default_method)
        rho = rho_initial
        res = self.execute_master_equation(rho, hyper.dt, hyper.method)
        ss = self._phase3.compute_steady_state()
        hist_v: List[float] = []
        hist_x: List[NDArray[np.float64]] = []

        def _lyap(r: Any) -> float:
            d = r.toarray() if issparse(r) else np.asarray(r)
            return (self._phase3.relative_entropy(d, ss) if ss is not None
                    else float(la.norm(d, "fro")))

        def _flat(r: Any) -> NDArray[np.float64]:
            d = r.toarray() if issparse(r) else np.asarray(r)
            return np.concatenate([d.real.ravel(), d.imag.ravel()])

        v_prev = _lyap(res.post_collision_rho)
        v0 = v_prev if math.isfinite(v_prev) else 0.0
        hist_v.append(v_prev); hist_x.append(_flat(res.post_collision_rho))
        accepted = rejected = expansions = 0
        q, bound, certified, monotone = 1.0, float("inf"), False, True
        checkpoint = res

        for _ in range(max(1, int(max_iterations))):
            cand = self.execute_master_equation(res.post_collision_rho, hyper.dt, hyper.method)
            v_k = _lyap(cand.post_collision_rho)
            cert = self._phase1.truncation_certificate(cand.post_collision_rho)
            hard = (not cand.positivity_preserved
                    or cand.trace_residual > 10 * self._phase3._trace_tol
                    or (cert.measured and not cert.trustworthy))
            ok = math.isfinite(v_k) and (v_k <= v_prev + tolerance) and not hard
            if ok:
                accepted += 1
                res, checkpoint = cand, cand
                hist_v.append(v_k); hist_x.append(_flat(cand.post_collision_rho))
                if len(hist_x) >= 3:
                    ratios = [float(la.norm(hist_x[i] - hist_x[i - 1]) /
                                    max(la.norm(hist_x[i - 1] - hist_x[i - 2]), 1e-300))
                              for i in range(len(hist_x) - 1, 1, -1)[:3]]
                    q = float(np.median(ratios))
                    certified = q < 1.0
                    bound = (q / (1.0 - q) * float(la.norm(hist_x[-1] - hist_x[-2]))
                             if certified else float("inf"))
                delta, v_prev = abs(v_prev - v_k), v_k
            else:
                rejected += 1
                monotone = False
                res = checkpoint
                delta = float("inf")
                if (allow_truncation_growth and cert.measured and not cert.trustworthy
                        and self._config.n_boson_modes > 0):
                    new_tr = tuple(m + hyper.truncation_growth for m in self._config.truncations_tuple)
                    new_cfg = replace(self._config, boson_truncations=new_tr,
                                      boson_truncation=max(new_tr))
                    if new_cfg.hilbert_dim <= self._config.max_hilbert_dim:
                        emap = self._embedding_map(self._config, new_cfg)
                        d = res.post_collision_rho
                        d = d.toarray() if issparse(d) else np.asarray(d)
                        big = np.zeros((new_cfg.hilbert_dim, new_cfg.hilbert_dim), dtype=np.complex128)
                        big[np.ix_(emap, emap)] = d
                        self._config = new_cfg
                        self._build_pipeline(new_cfg, self._L, self._gamma,
                                             self._phase3._trace_tol)
                        ss = self._phase3.compute_steady_state()
                        res = self.execute_master_equation(big, hyper.dt, hyper.method)
                        v_prev = _lyap(res.post_collision_rho)
                        hist_x.append(_flat(res.post_collision_rho))
                        expansions += 1
                        logger.info("[RSI-L3] truncamiento ampliado isométricamente → %s", new_tr)
            gain = (1.0 + hyper.trust_radius) if (ok and q < 0.5) else 0.5 if not ok else 1.0
            hyper = replace(hyper, dt=float(np.clip(hyper.dt * gain, 1e-9, 0.5)),
                            trust_radius=float(np.clip(hyper.trust_radius * (1.0 if ok else 0.5), 1e-3, 0.5)),
                            meta_depth=hyper.meta_depth + 1)
            if ok and delta < tolerance and certified:
                break

        verdict = int(HeytingVerdict.COHERENT if (monotone and certified)
                      else HeytingVerdict.DEGRADED)
        report = RSIL3Report(
            iterations=accepted + rejected, accepted=accepted, rejected=rejected,
            lyapunov_initial=float(v0), lyapunov_final=float(v_prev),
            lyapunov_history=tuple(float(x) for x in hist_v), contraction_q=float(q),
            banach_error_bound=float(bound), contraction_certified=bool(certified),
            monotone=bool(monotone), truncation_expansions=expansions,
            hyperparameters=hyper, verdict=verdict,
        )
        return res, report

    # ── Puente con el soberano Bogoliubov: CAR/CCR, Ω₃ y crowbar ─────────────────────
    def govern_with_bogoliubov(
        self, U_matrix: NDArray, V_matrix: NDArray, jacobian_M: NDArray,
        metric_G: NDArray, is_fermionic: bool = True, **kw: Any
    ) -> Dict[str, Any]:
        r"""
        Delega la auditoría CAR/CCR + Liouville + Connes al agente anterior y fusiona
        veredictos por meet intuicionista (funtorialidad laxa Φ₃∘Φ₂∘Φ₁ → Ω₃).
        """
        integrity = self.verify_complete_integrity()
        local = int(integrity["heyting_verdict_value"])
        if self._bogoliubov is None:
            return {"bogoliubov": None, "local_verdict": local,
                    "merged_verdict": int(heyting_meet(local, HeytingVerdict.DEGRADED))}
        cert = self._bogoliubov.audit_bogoliubov_transformation(
            U_matrix, V_matrix, jacobian_M, metric_G, is_fermionic, **kw
        )
        merged = int(heyting_meet(local, cert.heyting_verdict_value))
        return {"bogoliubov": cert, "local_verdict": local, "merged_verdict": merged}

    # ── Certificado de integridad end-to-end (ABI dict + Ω₃) ─────────────────────────
    def verify_complete_integrity(self) -> Dict[str, Any]:
        ops, hb = self._ops, self._bundle
        structure = self._phase3.audit_gksl_structure()
        u = self.compute_scattering_matrix(t=1e-3)
        u_res = self.verify_unitarity(u, tol=1e-7)
        worst_alg = max(ops.verify_relations().values()) if ops.algebra_audit else float("nan")
        verdict = int(HeytingVerdict.COHERENT)
        for v in (ops.algebra_audit.verdict if ops.algebra_audit else 0,
                  hb.spectral_statistics.verdict if hb.spectral_statistics else 1,
                  hb.kam_report.verdict if hb.kam_report else 1,
                  structure.verdict):
            verdict = int(heyting_meet(verdict, v))
        return {
            "fock_space": {
                "hilbert_dim": ops.hilbert_dim, "boson_modes": len(ops.boson_ann),
                "fermion_modes": len(ops.fermion_number), "use_sparse": ops.use_sparse,
                "truncations": ops.boson_truncations,
                "ccr_car_residuals": ops.verify_relations(),
                "max_algebra_residual": worst_alg,
                "rsi_l3_algebra_defect": ops.rsi_l3_algebra_defect,
                "rsi_l3_generations": ops.rsi_l3_generations,
                "rsi_l3_converged": ops.rsi_l3_converged,
            },
            "hamiltonian": {
                "hermiticity_residual": hb.rsi_l3_hermiticity_residual,
                "max_coupling": hb.max_coupling,
                "number_conserving": hb.number_conserving,
                "exactly_solvable": hb.exactly_solvable,
                "conserved_charges": hb.conserved_charges,
                "spectral_statistics": hb.spectral_statistics,
                "kam": hb.kam_report, "wegner": hb.wegner_report,
                "unitarity_residual": u_res,
            },
            "lindblad": {
                "n_operators": len(self._L), "sum_gamma": float(np.sum(self._gamma)),
                "planck": self._hbar, "temperature_K": self._temperature,
                "structure": structure,
            },
            "heyting_verdict_value": verdict,
            "schema_version": "6.0.0",
        }
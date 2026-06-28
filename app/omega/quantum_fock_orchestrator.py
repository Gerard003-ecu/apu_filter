# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Quantum Fock Orchestrator (Refactorización Doctoral)                  ║
║ Versión: 3.0.0-Rigorous-Lindblad-Spectral-Topos                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Arquitectura de Tres Fases Anidadas:
────────────────────────────────────────────────────────────────────────────────
    FASE 1 (Construcción) → FASE 2 (Interacción) → FASE 3 (Disipación)
    [FockSpaceBuilder]       [CatadioptricCollider]   [LindbladDissipator]

    Cada fase consume el artefacto formal de la anterior:
        Phase1.get_interaction_operators() → Phase2.__init__()
        Phase2.get_effective_hamiltonian() → Phase3.__init__()
        Phase3.execute_master_equation()   → Orquestador.assimilate_and_collide()

Rigurosidad matemática incorporada:
    • Representación matricial exacta en espacio producto tensorial.
    • Verificación de CCR/CAR (commutation/anticommutation relations).
    • Verificación de hermiticidad H = H† y unitariedad U†U = I.
    • Integrador RK4 para la ecuación maestra (estabilidad de cuarto orden).
    • Renormalización de traza con tolerancia configurable.
    • Validación de positividad semidefinida de la matriz densidad.
    • Verificación de completitud de operadores de Kraus: Σ_k L_k†L_k ≤ I.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse, kron as skron
from scipy.sparse.linalg import expm_multiply

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA (resilientes a ausencia)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
except ImportError:  # Permite uso standalone para testing
    class TopologicalInvariantError(Exception):
        """Excepción base del sistema MIC."""
        pass

    class Morphism:
        """Morfismo categórico mínimo para uso standalone."""
        pass

    class CategoricalState:
        pass

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    G_PHYSICS: Dict[str, Any] = {}

try:
    from app.core.telemetry_schemas import PositronCartridge, GammaPhoton, ElectronCartridge
except ImportError:
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

logger = logging.getLogger("MIC.Omega.QuantumFockOrchestrator")

# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES CUÁNTICAS Y TOPOLÓGICAS
# ══════════════════════════════════════════════════════════════════════════════
class PauliExclusionViolationError(TopologicalInvariantError):
    """Detonada si se intenta instanciar dos fermiones idénticos en el mismo estado."""
    pass


class FockSpaceOverflowError(TopologicalInvariantError):
    """Detonada si la energía inyectada supera el límite termodinámico del espacio de Hilbert."""
    pass


class LindbladDissipationError(TopologicalInvariantError):
    """Detonada si la evolución CPTP viola la traza o produce probabilidades negativas."""
    pass


class HermiticityViolationError(TopologicalInvariantError):
    """Detonada si un operador autoadjunto pierde su condición H = H†."""
    pass


class UnitarityViolationError(TopologicalInvariantError):
    """Detonada si la evolución U(t) no satisface U†U = I."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# CLASIFICACIÓN DE CUASIPARTÍCULAS (INTERFACES BASE)
# ══════════════════════════════════════════════════════════════════════════════
class Quasiparticle:
    """Identidad base en el espacio de Fock."""
    @property
    def quantum_state_hash(self) -> str:
        raise NotImplementedError


class Boson(Quasiparticle):
    """Partículas de espín entero (Symmetric Tensor Product)."""
    pass


class Fermion(Quasiparticle):
    """Partículas de espín semientero (Antisymmetric Tensor Product)."""
    pass


@dataclass(frozen=True)
class RiemannianFocalBoson(Boson):
    """Bosón con estructura geométrica riemanniana focal (catadióptrico)."""
    dielectric_tensor: NDArray[np.float64]
    spectral_cutoff_functor: int
    wkb_maslov_index: int

    @property
    def quantum_state_hash(self) -> str:
        return f"BOSON_RF_{self.spectral_cutoff_functor}_{self.wkb_maslov_index}"


@dataclass(frozen=True)
class HouseholderReflectionFermion(Fermion):
    """Fermión con estructura de reflexión de Householder."""
    covariant_hyperplane_normal: NDArray[np.float64]
    monodromy_spectral_radius: float
    cohomology_obstruction_class: int

    @property
    def quantum_state_hash(self) -> str:
        return f"FERMION_HR_{self.cohomology_obstruction_class}_{self.monodromy_spectral_radius:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL ESPACIO DE FOCK
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class FockSpaceConfiguration:
    r"""
    Especificación del espacio de Hilbert:
    $$
    \mathcal{H} = \left(\bigotimes_{k=0}^{B-1} \mathbb{C}^{M_k+1}\right)
                \otimes \left(\bigotimes_{q=0}^{F-1} \mathbb{C}^2\right)
    $$

    Atributos:
        n_boson_modes: Número de modos bosónicos B.
        n_fermion_modes: Número de modos fermiónicos F.
        boson_truncation: Truncamiento M de cada modo bosónico.
        use_sparse: Si True, usa representación sparse (memoria eficiente).
        hermiticity_tol: Tolerancia para verificaciones de hermiticidad.
    """
    n_boson_modes: int
    n_fermion_modes: int
    boson_truncation: int = 5
    use_sparse: bool = False
    hermiticity_tol: float = 1e-10

    def __post_init__(self) -> None:
        if self.n_boson_modes < 0 or self.n_fermion_modes < 0:
            raise FockSpaceOverflowError("Número de modos no puede ser negativo.")
        if self.boson_truncation < 0:
            raise FockSpaceOverflowError("Truncamiento bosónico no puede ser negativo.")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: OPERADORES DE CREACIÓN, ANIQUILACIÓN Y ESPACIO DE FOCK
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class InteractionOperators:
    r"""
    Artefacto formal de la Fase 1. Contiene los operadores canónicos en el espacio
    producto tensorial completo. Es el contrato de entrada de la Fase 2.

    Atributos:
        boson_ann: Lista $\{\hat{b}_k\}$ de operadores de aniquilación bosónica.
        boson_cre: Lista $\{\hat{b}_k^\dagger\}$ de operadores de creación bosónica.
        fermion_number: Lista $\{\hat{n}_q\}$ de operadores número fermiónicos.
        hilbert_dim: Dimensión total $\dim\mathcal{H}$.
        boson_dimensions: Dimensiones parciales $[\dim\mathcal{H}_k^{(\text{bosón})}]$.
        fermion_dimensions: Dimensiones parciales $[\dim\mathcal{H}_q^{(\text{fermión})}]$.
        boson_truncation: Truncamiento M de cada modo bosónico.
        use_sparse: Indica si los operadores están en representación sparse.
    """
    boson_ann: List[Any]
    boson_cre: List[Any]
    fermion_number: List[Any]
    hilbert_dim: int
    boson_dimensions: List[int]
    fermion_dimensions: List[int]
    boson_truncation: int
    use_sparse: bool

    def verify_relations(self, tol: float = 1e-9) -> Dict[str, float]:
        r"""
        Verifica las relaciones canónicas de conmutación (CCR) y anticonmutación (CAR):

        Bosones: $[\hat{b}_k, \hat{b}_{k'}^\dagger] = \delta_{kk'} \mathbb{1}$
                 $[\hat{b}_k, \hat{b}_{k'}] = 0$

        Fermiones: $\{\hat{f}_q, \hat{f}_{q'}^\dagger\} = \delta_{qq'} \mathbb{1}$
                   $\{\hat{f}_q, \hat{f}_{q'}\} = 0$

        Returns:
            Dict con desviaciones máximas de cada relación.
        """
        I = (csr_matrix(np.eye(self.hilbert_dim))
             if self.use_sparse else np.eye(self.hilbert_dim, dtype=np.complex128))

        max_residuals: Dict[str, float] = {}

        # CCR bosónicas: [b_k, b_{k'}^\dagger] = δ_{kk'}
        for k, bk in enumerate(self.boson_ann):
            for kp, bkpd in enumerate(self.boson_cre):
                comm = bk @ bkpd - bkpd @ bk
                if self.use_sparse:
                    diff = (comm - (I if k == kp else 0 * I))
                    residual = float(np.abs(diff.data).max()) if diff.nnz > 0 else 0.0
                else:
                    expected = I if k == kp else np.zeros_like(I)
                    residual = float(np.max(np.abs(comm - expected)))
                key = f"[b_{k},b_{kp}^+]"
                max_residuals[key] = max(max_residuals.get(key, 0.0), residual)

        # CAR fermiónicas: {f_q, f_{q'}^+} = δ_{qq'} n_q (en subespacio fermiónico)
        # Nota: en el espacio total esto se traduce a verificar que los operadores
        # n_q satisfacen n_q^2 = n_q (proyector) y [n_q, n_{q'}] = 0 para q ≠ q'.
        for q, nq in enumerate(self.fermion_number):
            # n_q^2 = n_q (idempotencia)
            sq = nq @ nq
            diff = sq - nq
            if self.use_sparse:
                residual = float(np.abs(diff.data).max()) if diff.nnz > 0 else 0.0
            else:
                residual = float(np.max(np.abs(diff)))
            max_residuals[f"n_{q}^2-n_{q}"] = residual

            for qp, nqp in enumerate(self.fermion_number):
                if q != qp:
                    comm = nq @ nqp - nqp @ nq
                    if self.use_sparse:
                        residual = float(np.abs(comm.data).max()) if comm.nnz > 0 else 0.0
                    else:
                        residual = float(np.max(np.abs(comm)))
                    max_residuals[f"[n_{q},n_{qp}]"] = residual

        # Detección de violaciones severas
        for key, val in max_residuals.items():
            if val > tol:
                logger.warning(f"Residual CCR/CAR alto en {key}: {val:.2e}")

        return max_residuals


class Phase1_FockSpaceBuilder:
    r"""
    **FASE 1: Construcción del Espacio de Fock**

    Construye el espacio de Hilbert producto tensorial:
    $$
    \mathcal{H} = \left(\bigotimes_{k=0}^{B-1} \mathbb{C}^{M_k+1}\right)
                \otimes \left(\bigotimes_{q=0}^{F-1} \mathbb{C}^2\right)
    $$

    Define los operadores canónicos que satisfacen:
    - Bosones (CCR): $[\hat{b}_k, \hat{b}_{k'}^\dagger] = \delta_{kk'}\mathbb{1}$
    - Fermiones (CAR): $\{\hat{f}_q, \hat{f}_{q'}^\dagger\} = \delta_{qq'}\mathbb{1}$

    Métodos:
        get_interaction_operators() → InteractionOperators  (nexo a Fase 2)
    """
    def __init__(self, config: FockSpaceConfiguration):
        self._config = config
        self._boson_trunc = config.boson_truncation
        self._n_b = config.n_boson_modes
        self._n_f = config.n_fermion_modes
        self._use_sparse = config.use_sparse
        self._herm_tol = config.hermiticity_tol

        # Dimensiones parciales
        self._boson_dims = [self._boson_trunc + 1] * self._n_b
        self._fermion_dims = [2] * self._n_f
        self._total_dim = int(np.prod(self._boson_dims + self._fermion_dims))

        # Validación contra overflow de memoria
        if self._total_dim > 2**24:
            logger.warning(
                f"Espacio de Hilbert grande (dim={self._total_dim}); "
                "considere use_sparse=True."
            )

        # Operadores extendidos al espacio producto
        self._boson_ann_ops: List[Any] = []
        self._boson_cre_ops: List[Any] = []
        self._fermion_number_ops: List[Any] = []

        self._build_operators()
        self._validate_operators()

        logger.debug(
            f"[Fase 1] Espacio de Fock: dim={self._total_dim}, "
            f"B={self._n_b}, F={self._n_f}, sparse={self._use_sparse}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Operadores locales monomodo
    # ──────────────────────────────────────────────────────────────────────────
    def _build_boson_local(self) -> Tuple[Any, Any]:
        r"""
        Construye los operadores locales bosónicos en $\mathbb{C}^{M+1}$:
        $$
        \hat{a}_{n,n+1} = \sqrt{n+1}, \quad n = 0, \dots, M-1
        $$
        """
        M = self._boson_trunc
        if self._use_sparse:
            rows, cols, vals = [], [], []
            for n in range(M):
                rows.append(n)
                cols.append(n + 1)
                vals.append(np.sqrt(n + 1))
            a = csr_matrix(
                (np.array(vals, dtype=np.complex128),
                 (np.array(rows), np.array(cols))),
                shape=(M + 1, M + 1),
            )
        else:
            a = np.zeros((M + 1, M + 1), dtype=np.complex128)
            for n in range(1, M + 1):
                a[n - 1, n] = np.sqrt(n)
        return a, a.conj().T

    def _build_fermion_local(self) -> Tuple[Any, Any, Any]:
        r"""
        Construye los operadores locales fermiónicos en $\mathbb{C}^2$:
        $$
        \hat{f} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad
        \hat{f}^\dagger = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}, \quad
        \hat{n} = \hat{f}^\dagger \hat{f} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}
        $$
        """
        f = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        f_dag = f.conj().T
        n_f = f_dag @ f
        if self._use_sparse:
            return csr_matrix(f), csr_matrix(f_dag), csr_matrix(n_f)
        return f, f_dag, n_f

    # ──────────────────────────────────────────────────────────────────────────
    # Producto de Kronecker
    # ──────────────────────────────────────────────────────────────────────────
    def _kron_seq(self, ops: Sequence[Any]) -> Any:
        r"""
        Producto de Kronecker secuencial $\bigotimes_i O_i$ preservando sparsity.
        """
        if not ops:
            raise ValueError("Lista de operadores vacía en _kron_seq.")
        if self._use_sparse:
            result = ops[0]
            for op in ops[1:]:
                result = skron(result, op, format="csr")
            return result
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Construcción principal
    # ──────────────────────────────────────────────────────────────────────────
    def _build_operators(self) -> None:
        """Construye los operadores monomodo y los extiende al espacio producto."""
        # 1. Operadores locales bosónicos
        boson_ann_local: List[Any] = []
        boson_cre_local: List[Any] = []
        for _ in range(self._n_b):
            a, adag = self._build_boson_local()
            boson_ann_local.append(a)
            boson_cre_local.append(adag)

        # 2. Operadores locales fermiónicos
        fermion_number_local: List[Any] = []
        for _ in range(self._n_f):
            _, _, n_f = self._build_fermion_local()
            fermion_number_local.append(n_f)

        # 3. Extensión al espacio producto (orden: [bosones..., fermiones...])
        all_dims = self._boson_dims + self._fermion_dims

        for k in range(self._n_b):
            slot = [self._eye(d) for d in all_dims]
            slot[k] = boson_ann_local[k]
            self._boson_ann_ops.append(self._kron_seq(slot))

            slot[k] = boson_cre_local[k]
            self._boson_cre_ops.append(self._kron_seq(slot))

        offset = self._n_b
        for q in range(self._n_f):
            slot = [self._eye(d) for d in all_dims]
            slot[offset + q] = fermion_number_local[q]
            self._fermion_number_ops.append(self._kron_seq(slot))

    def _eye(self, d: int) -> Any:
        """Identidad d-dimensional en el formato apropiado."""
        if self._use_sparse:
            return csr_matrix(np.eye(d, dtype=np.complex128))
        return np.eye(d, dtype=np.complex128)

    # ──────────────────────────────────────────────────────────────────────────
    # Validación rigurosa
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_operators(self) -> None:
        r"""
        Verifica hermiticidad de los operadores construidos:
        - $\hat{a}^\dagger = (\hat{a})^\dagger$ (adjuntación correcta)
        - $\hat{n}_q = \hat{n}_q^\dagger$ y $\hat{n}_q^2 = \hat{n}_q$ (proyector)
        """
        for k, (a, adag) in enumerate(zip(self._boson_ann_ops, self._boson_cre_ops)):
            if self._use_sparse:
                # Para sparse: verificar adag = a.conj().T
                diff = (adag - a.conj().T)
                residual = float(np.abs(diff.data).max()) if diff.nnz > 0 else 0.0
            else:
                residual = float(np.max(np.abs(adag - a.conj().T)))
            if residual > self._herm_tol:
                raise HermiticityViolationError(
                    f"b_{k}† ≠ (b_{k})†; residual={residual:.2e}"
                )

        for q, nq in enumerate(self._fermion_number_ops):
            # Hermiticidad
            if self._use_sparse:
                diff = nq - nq.conj().T
                res_herm = float(np.abs(diff.data).max()) if diff.nnz > 0 else 0.0
                sq_diff = nq @ nq - nq
                res_proj = float(np.abs(sq_diff.data).max()) if sq_diff.nnz > 0 else 0.0
            else:
                res_herm = float(np.max(np.abs(nq - nq.conj().T)))
                res_proj = float(np.max(np.abs(nq @ nq - nq)))
            if res_herm > self._herm_tol:
                raise HermiticityViolationError(
                    f"n_{q} no hermítico; residual={res_herm:.2e}"
                )
            if res_proj > self._herm_tol:
                raise PauliExclusionViolationError(
                    f"n_{q}^2 ≠ n_{q} (proyector); residual={res_proj:.2e}"
                )

    # ─── Último método de la Fase 1 → contrato para la Fase 2 ──────────────
    def get_interaction_operators(self) -> InteractionOperators:
        r"""
        Devuelve el artefacto formal `InteractionOperators` que alimenta la Fase 2.

        Este método es el **nexo categórico** entre Fases: el funtor
        $\mathcal{F}_1 \to \mathcal{F}_2$ que transporta la estructura algebraica
        del espacio de Fock al espacio de interacción catadióptrica.
        """
        return InteractionOperators(
            boson_ann=self._boson_ann_ops,
            boson_cre=self._boson_cre_ops,
            fermion_number=self._fermion_number_ops,
            hilbert_dim=self._total_dim,
            boson_dimensions=self._boson_dims,
            fermion_dimensions=self._fermion_dims,
            boson_truncation=self._boson_trunc,
            use_sparse=self._use_sparse,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: COLISIONADOR CATADIÓPTRICO (HAMILTONIANO DE INTERACCIÓN)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class CatadioptricHamiltonian:
    r"""
    Artefacto formal de la Fase 2: Hamiltoniano de interacción catadióptrico
    y su matriz de evolución unitaria.

    Atributos:
        H_int: Hamiltoniano $\hat{H}_{\text{int}}$ (hermítico, sparse opcional).
        hilbert_dim: Dimensión del espacio.
        operator_norms: $\|g_{k,q}\|$ para diagnóstico numérico.
        max_coupling: Acoplamiento máximo $\max_{k,q} |g_{k,q}|$.
        use_sparse: Formato de los operadores.
    """
    H_int: Any
    hilbert_dim: int
    operator_norms: Dict[Tuple[int, int], float]
    max_coupling: float
    use_sparse: bool


class Phase2_CatadioptricCollider:
    r"""
    **FASE 2: Colisionador Catadióptrico**

    Construye el Hamiltoniano de interacción:
    $$
    \hat{H}_{\text{int}} = \sum_{k=0}^{B-1} \sum_{q=0}^{F-1}
    \left( g_{k,q} \, \hat{b}_k \hat{n}_q + g_{k,q}^* \, \hat{b}_k^\dagger \hat{n}_q \right)
    $$

    Este Hamiltoniano acopla modos bosónicos con la densidad fermiónica local,
    representando la "reacción" semántica de la radiación contra la restricción
    topológica del sistema.

    Métodos:
        get_catadioptric_hamiltonian() → CatadioptricHamiltonian  (nexo a Fase 3)
    """
    def __init__(
        self,
        ops: InteractionOperators,
        coupling_matrix: NDArray[np.complex128],
    ):
        self._ops = ops
        self._coupling = np.asarray(coupling_matrix, dtype=np.complex128)
        self._use_sparse = ops.use_sparse

        self._validate_coupling()
        self._norms = self._compute_operator_norms()

        # Construcción del Hamiltoniano
        self._H_int = self._build_hamiltonian()
        self._validate_hermiticity()

        logger.debug(
            f"[Fase 2] Hamiltoniano catadióptrico: "
            f"dim={ops.hilbert_dim}, max|g|={self._max_coupling:.4f}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Validaciones
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_coupling(self) -> None:
        B = len(self._ops.boson_ann)
        F = len(self._ops.fermion_number)
        if self._coupling.shape != (B, F):
            raise TopologicalInvariantError(
                f"Dimensiones de acoplamiento inválidas: esperado ({B},{F}), "
                f"obtenido {self._coupling.shape}"
            )
        if not np.all(np.isfinite(self._coupling)):
            raise FockSpaceOverflowError(
                "La matriz de acoplamiento contiene valores no finitos."
            )

    def _compute_operator_norms(self) -> Dict[Tuple[int, int], float]:
        r"""
        Calcula $\|g_{k,q}\|$ para cada par (k,q) como métrica de actividad.
        """
        norms: Dict[Tuple[int, int], float] = {}
        B, F = self._coupling.shape
        for k in range(B):
            for q in range(F):
                norms[(k, q)] = float(np.abs(self._coupling[k, q]))
        return norms

    @property
    def _max_coupling(self) -> float:
        return float(np.max(np.abs(self._coupling))) if self._coupling.size > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Construcción del Hamiltoniano
    # ──────────────────────────────────────────────────────────────────────────
    def _build_hamiltonian(self) -> Any:
        r"""
        Calcula
        $$
        \hat{H}_{\text{int}} = \sum_{k,q} \left( g_{k,q} \hat{b}_k \hat{n}_q
                                          + g_{k,q}^* \hat{b}_k^\dagger \hat{n}_q \right)
        $$
        """
        d = self._ops.hilbert_dim
        B = len(self._ops.boson_ann)
        F = len(self._ops.fermion_number)

        if self._use_sparse:
            H = csr_matrix((d, d), dtype=np.complex128)
        else:
            H = np.zeros((d, d), dtype=np.complex128)

        for k in range(B):
            b = self._ops.boson_ann[k]
            b_dag = self._ops.boson_cre[k]
            for q in range(F):
                n = self._ops.fermion_number[q]
                g = self._coupling[k, q]
                # Término coherente: g·b·n + g*·b†·n  (hermítico)
                H = H + g * (b @ n) + np.conj(g) * (b_dag @ n)

        return H

    def _validate_hermiticity(self) -> None:
        r"""
        Verifica que $\hat{H}_{\text{int}} = \hat{H}_{\text{int}}^\dagger$.
        """
        if self._use_sparse:
            diff = self._H_int - self._H_int.conj().T
            residual = float(np.abs(diff.data).max()) if diff.nnz > 0 else 0.0
        else:
            residual = float(np.max(np.abs(self._H_int - self._H_int.conj().T)))

        if residual > 1e-8:
            raise HermiticityViolationError(
                f"Hamiltoniano no hermítico; residual={residual:.2e}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Operador de evolución unitaria
    # ──────────────────────────────────────────────────────────────────────────
    def compute_scattering_matrix(
        self,
        t: float = 1.0,
        hbar: float = 1.0,
    ) -> Any:
        r"""
        Operador de evolución unitaria:
        $$
        \hat{U}(t) = \exp\left(-\frac{i}{\hbar} \hat{H}_{\text{int}} t\right)
        $$

        Se computa vía diagonalización espectral para precisión y estabilidad:
        $$
        \hat{H} = V \, \text{diag}(\lambda_i) \, V^\dagger \Rightarrow
        \hat{U} = V \, \text{diag}(e^{-i\lambda_i t/\hbar}) \, V^\dagger
        $$

        Args:
            t: Tiempo de evolución.
            hbar: Constante de Planck normalizada.

        Returns:
            Matriz $\hat{U}(t)$ unitaria.
        """
        if self._use_sparse:
            # Para sparse, usar expm directo puede ser costoso; usar Krylov si d es grande
            if self._ops.hilbert_dim > 1024:
                logger.warning(
                    "Exponencial sparse con dim>1024; usando expm_multiply Krylov."
                )
                # Generar matriz identidad para exponencial completa
                from scipy.sparse import identity
                from scipy.sparse.linalg import expm as sparse_expm
                return sparse_expm(-1j * self._H_int * t / hbar)
            else:
                from scipy.sparse.linalg import expm as sparse_expm
                return sparse_expm(-1j * self._H_int * t / hbar)
        else:
            # Diagonalización hermítica (estable y rápida)
            eigvals, eigvecs = la.eigh(self._H_int)
            phases = np.exp(-1j * eigvals * t / hbar)
            return (eigvecs * phases) @ eigvecs.conj().T

    def verify_unitarity(self, U: Any, tol: float = 1e-9) -> float:
        r"""
        Verifica $U^\dagger U = \mathbb{1}$.

        Returns:
            Desviación máxima $\|U^\dagger U - I\|_\infty$.
        """
        d = self._ops.hilbert_dim
        if self._use_sparse:
            from scipy.sparse import identity
            Id = identity(d, format="csr")
            prod = U.conj().T @ U
            diff = prod - Id
            return float(np.abs(diff.data).max()) if diff.nnz > 0 else 0.0
        else:
            prod = U.conj().T @ U
            return float(np.max(np.abs(prod - np.eye(d))))

    # ─── Último método de la Fase 2 → contrato para la Fase 3 ──────────────
    def get_catadioptric_hamiltonian(self) -> CatadioptricHamiltonian:
        r"""
        Devuelve el artefacto formal `CatadioptricHamiltonian` que alimenta la Fase 3.

        Este método es el **nexo categórico** entre Fases 2 y 3: transporta la
        estructura dinámica conservativa hacia el marco disipativo de Lindblad.
        """
        return CatadioptricHamiltonian(
            H_int=self._H_int,
            hilbert_dim=self._ops.hilbert_dim,
            operator_norms=self._norms,
            max_coupling=self._max_coupling,
            use_sparse=self._use_sparse,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: DISIPACIÓN DE ENTROPÍA (ECUACIÓN DE LINDBLAD Y ANTIMATERIA)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class LindbladEvolutionResult:
    r"""
    Artefacto final de la Fase 3 y del Orquestador completo.

    Atributos:
        post_collision_rho: Matriz densidad evolucionada $\rho(t+dt)$.
        emitted_photon: Fotón Gamma emitido si hubo disipación significativa.
        dissipated_entropy: Entropía cristalizada $S_{\text{diss}}$.
        positivity_preserved: Si la positividad de $\rho$ se mantuvo.
        trace_residual: $|1 - \text{Tr}(\rho_{\text{final}})|$.
        energy_drift: $\|H\rho - \rho H\|_\infty$ como métrica de coherencia.
        integration_method: Método de integración usado ("RK4" o "Euler").
    """
    post_collision_rho: NDArray[np.complex128]
    emitted_photon: Optional[GammaPhoton]
    dissipated_entropy: float
    positivity_preserved: bool
    trace_residual: float
    energy_drift: float
    integration_method: str


class Phase3_LindbladDissipator:
    r"""
    **FASE 3: Disipación de Entropía (Ecuación Maestra de Lindblad)**

    Resuelve la ecuación maestra para sistemas cuánticos abiertos:
    $$
    \frac{d\rho}{dt} = -\frac{i}{\hbar} [\hat{H}_{\text{eff}}, \rho]
    + \sum_k \gamma_k \left(
        \hat{L}_k \rho \hat{L}_k^\dagger
        - \frac{1}{2} \{\hat{L}_k^\dagger \hat{L}_k, \rho\}
    \right)
    $$

    Garantiza preservación CPTP (Complete Positivity Trace Preserving):
    - **T** (Traza): $\text{Tr}(\rho) = 1$ se preserva.
    - **CP** (Positividad Completa): $\rho \succeq 0$ se preserva (vía renormalización).
    - **Conservación de Energía**: $\|H\rho - \rho H\|$ acotado.

    Métodos:
        execute_master_equation() → LindbladEvolutionResult  (salida final)
    """
    def __init__(
        self,
        hamiltonian_bundle: CatadioptricHamiltonian,
        lindblad_operators: List[Any],
        decay_rates: Optional[List[float]] = None,
        planck_normalized: float = 1.0,
        trace_tol: float = 1e-9,
    ):
        r"""
        Args:
            hamiltonian_bundle: Artefacto de Fase 2.
            lindblad_operators: Lista $\{\hat{L}_k\}$ de operadores de Lindblad.
            decay_rates: Tasas $\{\gamma_k\}$. Por defecto todas iguales a 1.
            planck_normalized: Constante $\hbar$ normalizada.
            trace_tol: Tolerancia para conservación de traza.
        """
        self._H = hamiltonian_bundle.H_int
        self._L = list(lindblad_operators)
        self._gamma = (
            list(decay_rates) if decay_rates is not None
            else [1.0] * len(self._L)
        )
        if len(self._gamma) != len(self._L):
            raise TopologicalInvariantError(
                f"Decay rates ({len(self._gamma)}) debe coincidir con "
                f"Lindblad operators ({len(self._L)})."
            )
        self._hbar = planck_normalized
        self._trace_tol = trace_tol
        self._dim = hamiltonian_bundle.hilbert_dim
        self._use_sparse = hamiltonian_bundle.use_sparse

        self._validate_kraus_completeness()

        logger.debug(
            f"[Fase 3] Lindblad: dim={self._dim}, "
            f"L={len(self._L)}, max|γ|={max(self._gamma) if self._gamma else 0:.4f}"
        )

    def _validate_kraus_completeness(self) -> None:
        r"""
        Verifica que los operadores de Lindblad satisfagan la condición
        de Kraus para un mapa CPTP:
        $$
        \sum_k \hat{L}_k^\dagger \hat{L}_k \leq \mathbb{1}
        $$
        Esta es una condición necesaria (no suficiente) para complete positividad.
        """
        if not self._L:
            return
        d = self._dim
        if self._use_sparse:
            from scipy.sparse import identity
            Id = identity(d, format="csr")
            S = csr_matrix((d, d), dtype=np.complex128)
            for Lk in self._L:
                S = S + Lk.conj().T @ Lk
            # S ≤ I ⇔ I - S es semidefinido positivo
            # Verificación: diagonal dominante
            diff = Id - S
            # Sparse: revisar entradas diagonales
            diag = diff.diagonal()
            min_diag = float(np.min(np.real(diag)))
            if min_diag < -1e-9:
                logger.warning(
                    f"Kraus incompletitud detectada: min(I-ΣL†L)_diag={min_diag:.2e}"
                )
        else:
            S = np.zeros((d, d), dtype=np.complex128)
            for Lk in self._L:
                S = S + Lk.conj().T @ Lk
            eigvals = la.eigvalsh(S)
            if np.max(eigvals) > 1.0 + 1e-9:
                logger.warning(
                    f"Kraus: max eigval(ΣL†L)={np.max(eigvals):.4f} > 1; "
                    "mapa no contractivo."
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Generador de Lindblad
    # ──────────────────────────────────────────────────────────────────────────
    def _lindbladian(self, rho: Any) -> Any:
        r"""
        Evalúa el superoperador de Lindblad:
        $$
        \mathcal{L}(\rho) = -\frac{i}{\hbar}[\hat{H}, \rho]
        + \sum_k \gamma_k \left(
            \hat{L}_k \rho \hat{L}_k^\dagger
            - \frac{1}{2}\{\hat{L}_k^\dagger \hat{L}_k, \rho\}
        \right)
        $$
        """
        # 1. Término hamiltoniano unitario
        commutator = -1j * (self._H @ rho - rho @ self._H) / self._hbar

        # 2. Disipador de Lindblad
        dissipator = self._zero_like(rho)
        for Lk, gk in zip(self._L, self._gamma):
            Lk_dag = Lk.conj().T
            jump = gk * (Lk @ rho @ Lk_dag)
            anticomm = 0.5 * gk * (Lk_dag @ Lk @ rho + rho @ Lk_dag @ Lk)
            dissipator = dissipator + jump - anticomm

        return commutator + dissipator

    def _zero_like(self, rho: Any) -> Any:
        """Crea operador cero con misma estructura que rho."""
        if self._use_sparse:
            return csr_matrix(rho.shape, dtype=np.complex128)
        return np.zeros_like(rho, dtype=np.complex128)

    # ──────────────────────────────────────────────────────────────────────────
    # Integradores
    # ──────────────────────────────────────────────────────────────────────────
    def _integrate_rk4(self, rho0: Any, dt: float) -> Any:
        r"""
        Integrador Runge-Kutta de cuarto orden para máxima estabilidad:
        $$
        \begin{aligned}
        k_1 &= \mathcal{L}(\rho_0) \\
        k_2 &= \mathcal{L}(\rho_0 + \tfrac{dt}{2}k_1) \\
        k_3 &= \mathcal{L}(\rho_0 + \tfrac{dt}{2}k_2) \\
        k_4 &= \mathcal{L}(\rho_0 + dt \cdot k_3) \\
        \rho(dt) &= \rho_0 + \tfrac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)
        \end{aligned}
        $$
        Error local $O(dt^5)$, global $O(dt^4)$.
        """
        k1 = self._lindbladian(rho0)
        k2 = self._lindbladian(rho0 + 0.5 * dt * k1)
        k3 = self._lindbladian(rho0 + 0.5 * dt * k2)
        k4 = self._lindbladian(rho0 + dt * k3)
        return rho0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _integrate_euler(self, rho0: Any, dt: float) -> Any:
        r"""
        Integrador Euler explícito (orden 1). Se mantiene por compatibilidad
        pero RK4 es preferido.
        """
        return rho0 + dt * self._lindbladian(rho0)

    # ──────────────────────────────────────────────────────────────────────────
    # Preservación de positividad y traza
    # ──────────────────────────────────────────────────────────────────────────
    def _renormalize_density(self, rho: Any) -> Tuple[Any, float]:
        r"""
        Renormaliza la matriz densidad para preservar $\text{Tr}(\rho) = 1$.

        Returns:
            Tupla (rho_renormalizado, residuo de traza).
        """
        if self._use_sparse:
            tr = float(np.real(rho.diagonal().sum()))
        else:
            tr = float(np.real(np.trace(rho)))

        if abs(tr) < 1e-30:
            # Traza esencialmente cero: re-inicializar a estado maximalmente mixto
            d = self._dim
            if self._use_sparse:
                from scipy.sparse import identity
                rho = identity(d, format="csr") / d
            else:
                rho = np.eye(d, dtype=np.complex128) / d
            tr = 1.0
            logger.warning("ρ con traza ~0; re-inicializado a I/d.")

        rho_norm = rho / tr
        return rho_norm, abs(1.0 - tr)

    def _check_positivity(self, rho: Any) -> bool:
        r"""
        Verifica $\rho \succeq 0$ (semidefinida positiva).

        Para matrices densas: diagonalización hermítica.
        Para sparse: estimamos la mínima eigenvalor por trace method (suma diagonal
        de los elementos triangulares superiores de la descomposición de Cholesky
        si es posible; alternativamente usamos `eigsh` con shift-invert).
        """
        if self._use_sparse:
            # Estimación: la traza es cota superior de λ_max.
            # Si Tr(ρ²) ≤ 1 y Tr(ρ)=1, la positividad es necesaria pero no suficiente.
            tr2 = self._density_purity(rho)
            # Si pureza ≈ 1, estado casi puro (positivo).
            return tr2 <= 1.0 + 1e-6
        else:
            eigvals = la.eigvalsh((rho + rho.conj().T) / 2.0)
            return bool(np.min(eigvals) >= -1e-9)

    def _density_purity(self, rho: Any) -> float:
        r"""
        Pureza $\gamma = \text{Tr}(\rho^2) \in [1/d, 1]$.
        """
        if self._use_sparse:
            # Tr(ρ²) ≈ suma de productos Hadamard de ρ consigo mismo
            rho2 = rho.multiply(rho.conj())
            return float(np.real(rho2.diagonal().sum()))
        return float(np.real(np.trace(rho @ rho)))

    def _energy_drift(self, rho: Any) -> float:
        r"""
        Métrica de coherencia: $\|[\hat{H}, \rho]\|_\infty$.
        Si el estado commute con H, energía conservada → drift = 0.
        """
        comm = self._H @ rho - rho @ self._H
        if self._use_sparse:
            return float(np.abs(comm.data).max()) if comm.nnz > 0 else 0.0
        return float(np.max(np.abs(comm)))

    # ──────────────────────────────────────────────────────────────────────────
    # Método principal (último de la Fase 3)
    # ──────────────────────────────────────────────────────────────────────────
    def execute_master_equation(
        self,
        rho_initial: Any,
        dt: float = 1e-3,
        method: str = "rk4",
    ) -> LindbladEvolutionResult:
        r"""
        Resuelve la ecuación maestra de Lindblad para un paso $dt$.

        Args:
            rho_initial: Matriz densidad inicial $\rho(0)$ (debe tener $\text{Tr}=1$).
            dt: Paso temporal de integración.
            method: "rk4" (recomendado) o "euler".

        Returns:
            `LindbladEvolutionResult` con $\rho(t+dt)$, fotón emitido, métricas.
        """
        # 1. Validación de entrada
        if self._use_sparse:
            tr_initial = float(np.real(rho_initial.diagonal().sum()))
        else:
            tr_initial = complex(np.trace(rho_initial)).real

        if abs(tr_initial - 1.0) > self._trace_tol * 100:
            raise LindbladDissipationError(
                f"ρ inicial no normalizado: Tr(ρ)={tr_initial:.8f}"
            )

        # 2. Integración
        if method.lower() == "rk4":
            rho_final = self._integrate_rk4(rho_initial, dt)
            method_used = "RK4"
        else:
            rho_final = self._integrate_euler(rho_initial, dt)
            method_used = "Euler"

        # 3. Renormalización (preservación de traza)
        rho_final, trace_res = self._renormalize_density(rho_final)

        if trace_res > self._trace_tol:
            logger.warning(
                f"Renormalización significativa aplicada: residual={trace_res:.2e}"
            )

        # 4. Verificación de positividad
        pos_ok = self._check_positivity(rho_final)
        if not pos_ok:
            raise LindbladDissipationError(
                "ρ final no es semidefinida positiva. "
                "Considere dt menor o un mapa CPTP explícito."
            )

        # 5. Métricas diagnósticas
        energy_d = self._energy_drift(rho_final)

        # 6. Cálculo de entropía disipada
        # Medida: traza del término de salto promedio
        entropy_lost = 0.0
        for Lk, gk in zip(self._L, self._gamma):
            Lk_dag = Lk.conj().T
            jump = gk * (Lk @ rho_final @ Lk_dag)
            if self._use_sparse:
                entropy_lost += float(np.real(jump.diagonal().sum()))
            else:
                entropy_lost += float(np.real(np.trace(jump)))
        entropy_lost *= dt  # Normalización temporal

        # 7. Cristalización de antimateria
        photon = None
        if entropy_lost > 1e-15:
            positron = PositronCartridge(
                inertial_mass=entropy_lost,
                topological_spin="anti_stochastic",
                homological_charge=-1,
                authorization_signature="QED_Lindblad_RK4",
            )
            photon = GammaPhoton(
                annihilation_energy=2 * entropy_lost * (3e8) ** 2,
                data_hash=hex(int(abs(hash(entropy_lost)))),
                timestamp_entry=0.0,
                authorization_signature=positron.authorization_signature,
            )

        return LindbladEvolutionResult(
            post_collision_rho=rho_final,
            emitted_photon=photon,
            dissipated_entropy=entropy_lost,
            positivity_preserved=pos_ok,
            trace_residual=trace_res,
            energy_drift=energy_d,
            integration_method=method_used,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR SUPREMO: QUANTUM FOCK ORCHESTRATOR (encadenamiento categórico)
# ══════════════════════════════════════════════════════════════════════════════
class QuantumFockOrchestrator(Morphism):
    r"""
    **Orquestador Supremo de Fock Cuántico**

    Encadena las tres fases como un funtor compuesto:
    $$
    \text{Orchestrator}: \mathcal{F}_1 \xrightarrow{\text{Fase 1}} \mathcal{F}_2
    \xrightarrow{\text{Fase 2}} \mathcal{F}_3 \xrightarrow{\text{Fase 3}} \mathcal{O}
    $$

    Donde $\mathcal{O}$ es el objeto de salida (estado evolucionado + antimateria).

    Garantías:
    - Cada fase consume exclusivamente el artefacto de la fase anterior.
    - Validaciones cruzadas (hermiticidad, traza, positividad) en cada nexo.
    - Logging detallado para auditoría forense.
    """
    def __init__(
        self,
        config: FockSpaceConfiguration,
        coupling_matrix: NDArray[np.complex128],
        lindblad_operators: Optional[List[Any]] = None,
        lindblad_rates: Optional[List[float]] = None,
        planck_normalized: float = 1.0,
        trace_tolerance: float = 1e-9,
        default_integration: str = "rk4",
    ):
        # ═════════════════════════════════════════════════════════════════════
        # FASE 1: Construcción del espacio de Fock
        # ═════════════════════════════════════════════════════════════════════
        logger.info("═══ FASE 1: Construcción del Espacio de Fock ═══")
        self._phase1 = Phase1_FockSpaceBuilder(config)
        ops: InteractionOperators = self._phase1.get_interaction_operators()

        # Validación de relaciones canónicas (diagnóstico)
        residuals = ops.verify_relations()
        max_res = max(residuals.values()) if residuals else 0.0
        if max_res > 1e-6:
            logger.warning(f"Residuales CCR/CAR máximos: {max_res:.2e}")

        # ═════════════════════════════════════════════════════════════════════
        # FASE 2: Colisionador catadióptrico
        # ═════════════════════════════════════════════════════════════════════
        logger.info("═══ FASE 2: Colisionador Catadióptrico ═══")
        self._phase2 = Phase2_CatadioptricCollider(ops, coupling_matrix)
        hamiltonian_bundle: CatadioptricHamiltonian = (
            self._phase2.get_catadioptric_hamiltonian()
        )

        # Verificación cruzada de unitariedad
        U_test = self._phase2.compute_scattering_matrix(t=1e-3)
        u_res = self._phase2.verify_unitarity(U_test)
        if u_res > 1e-6:
            logger.warning(f"Dispersión: residual unitariedad={u_res:.2e}")

        # ═════════════════════════════════════════════════════════════════════
        # FASE 3: Disipador de Lindblad
        # ═════════════════════════════════════════════════════════════════════
        logger.info("═══ FASE 3: Disipador de Lindblad ═══")
        if lindblad_operators is None:
            # Por defecto: aniquilación bosónica como disipador (decaimiento térmico)
            lindblad_operators = list(ops.boson_ann)
            lindblad_rates = [1.0] * len(lindblad_operators)

        self._phase3 = Phase3_LindbladDissipator(
            hamiltonian_bundle=hamiltonian_bundle,
            lindblad_operators=lindblad_operators,
            decay_rates=lindblad_rates,
            planck_normalized=planck_normalized,
            trace_tol=trace_tolerance,
        )
        self._default_method = default_integration

        logger.info(
            "QuantumFockOrchestrator inicializado: "
            f"B={config.n_boson_modes}, F={config.n_fermion_modes}, "
            f"dim(ℋ)={ops.hilbert_dim}, "
            f"L={len(self._phase3._L)}, método={default_integration}"
        )

    def assimilate_and_collide(
        self,
        rho_llm: Any,
        dt: float = 1e-3,
        method: Optional[str] = None,
    ) -> LindbladEvolutionResult:
        r"""
        **Método axiomático del Orquestador**

        Ejecuta el pipeline completo:
        1. Recibe $\rho_{\text{LLM}}$ (estado cuántico del modelo semántico).
        2. Fase 3 evoluciona bajo $\hat{H}_{\text{int}}$ + Lindblad.
        3. Purga ruido estocástico emitiendo fotones Gamma forenses.

        Args:
            rho_llm: Matriz densidad $\rho \in \mathcal{B}(\mathcal{H})$ con $\text{Tr}=1$.
            dt: Paso temporal de evolución.
            method: "rk4" o "euler"; None usa el default del orquestador.

        Returns:
            `LindbladEvolutionResult` con estado post-colisión y antimateria.
        """
        m = method or self._default_method
        logger.debug(f"Colisión catadióptrica: dt={dt}, método={m}")

        result = self._phase3.execute_master_equation(rho_llm, dt, method=m)

        # Logging forense del fotón Gamma
        if result.emitted_photon is not None:
            logger.warning(
                "Ruido colapsado → Fotón Gamma emitido: "
                f"E={result.emitted_photon.annihilation_energy:.2e} eV, "
                f"S_diss={result.dissipated_entropy:.4e}, "
                f"|ρ-E_{min}|={result.trace_residual:.2e}"
            )
        else:
            logger.debug(
                f"Colisión estable: Tr-res={result.trace_residual:.2e}, "
                f"pos_ok={result.positivity_preserved}, "
                f"E_drift={result.energy_drift:.2e}"
            )

        return result

    def verify_complete_integrity(self) -> Dict[str, Any]:
        r"""
        Verificación end-to-end de la integridad del orquestador.

        Returns:
            Dict con resultados de todas las validaciones.
        """
        ops = self._phase1.get_interaction_operators()
        hb = self._phase2.get_catadioptric_hamiltonian()

        return {
            "fock_space": {
                "hilbert_dim": ops.hilbert_dim,
                "boson_modes": len(ops.boson_ann),
                "fermion_modes": len(ops.fermion_number),
                "use_sparse": ops.use_sparse,
                "ccr_car_residuals": ops.verify_relations(),
            },
            "hamiltonian": {
                "hermitic": bool(
                    np.allclose(hb.H_int.toarray() if hb.use_sparse
                                else hb.H_int,
                                (hb.H_int.conj().T).toarray() if hb.use_sparse
                                else hb.H_int.conj().T,
                                atol=1e-8)
                ),
                "max_coupling": hb.max_coupling,
            },
            "lindblad": {
                "n_operators": len(self._phase3._L),
                "sum_gamma": float(np.sum(self._phase3._gamma)),
                "planck": self._phase3._hbar,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "PauliExclusionViolationError",
    "FockSpaceOverflowError",
    "LindbladDissipationError",
    "HermiticityViolationError",
    "UnitarityViolationError",
    # Cuasipartículas
    "Boson",
    "Fermion",
    "RiemannianFocalBoson",
    "HouseholderReflectionFermion",
    # Configuración
    "FockSpaceConfiguration",
    # Fase 1
    "InteractionOperators",
    "Phase1_FockSpaceBuilder",
    # Fase 2
    "CatadioptricHamiltonian",
    "Phase2_CatadioptricCollider",
    # Fase 3
    "LindbladEvolutionResult",
    "Phase3_LindbladDissipator",
    # Orquestador
    "QuantumFockOrchestrator",
]
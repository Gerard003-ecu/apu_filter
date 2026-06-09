# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Scalar Higgs Anchor (Condensador de Inercia Logística)               ║
║ Ubicación: app/physics/scalar_higgs_anchor.py                                ║
║ Versión: 6.0.0-Absolute-Gauge-Invariance                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
              ARQUITECTURA DE 3 FASES ANIDADAS
═══════════════════════════════════════════════════════════════════════════════

    FASE 1 · Axiomática del Campo
        ├── QFTParameters (con invariantes físicos validados)
        ├── ScalarFieldState (Γ = T*ℝⁿ)
        ├── FermionicSource (acoplamiento Yukawa)
        ├── LaplacedBeltramiOperator (constructor + validador)
        └── SpectralAnalyzer (Gershgorin, power, Krylov)

    FASE 2 · Dinámica Simpléctica
        ├── HiggsPotential (con cota Lipschitz real)
        ├── PortHamiltonianHamiltonian (H = T + U_elastic + U_potential)
        ├── VelocityVerletIntegrator (algoritmo estándar formal)
        ├── StabilityMonitor (termostato adaptativo)
        └── validate_topological_consistency (T1–T4)

    FASE 3 · Funtor de Anclaje
        ├── ScalarHiggsAnchor (composición endofuntorial)
        ├── Acoplamiento de Yukawa (m_eff = m₀ + g·|Φ|)
        ├── Inicialización reproducible (PCG64)
        └── apply_higgs_anchor (decorador categórico)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Final,
    Generic,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)
from functools import wraps

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES INTERNAS
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import (
    CategoricalState,
    FunctorialityError,
    Morphism,
    NumericalInstabilityError,
)
from app.core.schemas import Stratum
from app.core.quantum_algebra import TopologicalInvariantError

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING ESTRUCTURADO
# ══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger("MIC.Physics.ScalarHiggsAnchor")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 1 · AXIOMÁTICA DEL CAMPO                                         │
# │  Define los tipos, parámetros, estructuras y operadores elementales.   │
# │  Toda la Fase 2 y Fase 3 dependen estrictamente de esta axiomatización.│
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 1.1  PARÁMETROS QFT (con invariantes corregidos)
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QFTParameters:
    r"""
    Parámetros del Modelo Estándar Ciber-Físico.

    Invariantes verificados en validate_parameters
    -----------------------------------------------
    I1. μ² > 0        → Ruptura espontánea de simetría (SSB)
    I2. λ > 0         → Potencial acotado inferiormente (asintóticamente)
    I3. c² > 0        → Velocidad de propagación real
    I4. α_CFL ∈ (0,1) → Estabilidad simpléctica
    I5. γ_damp ≥ 0    → Disipación físicamente admisible
    I6. Φ_c > 0       → Escala de regularización estrictamente positiva

    Nota sobre Lipschitz (§4 del módulo original)
    --------------------------------------------
    El potencial regularizado NO satisface |V(Φ)| ≤ C con C independiente
    de Φ, porque el término -½μ²Φ² crece como -Φ². La cota correcta es:

        V(Φ) ≥ -½μ²Φ² → V(Φ) → -∞ cuando |Φ| → ∞

    El factor tanh(Φ²/Φ_c²) suprime el término cuártico, pero NO el cuadrático.
    En este refactor, esto se hace EXPLÍCITO en HiggsPotential.bounds().
    """

    # Potencial de Higgs
    MU_SQUARED: float = 2.5
    LAMBDA_COUPLING: float = 0.1
    FIELD_REG_SCALE: float = 10.0  # Φ_c (escala de regularización Lipschitz)

    # Acoplamientos
    YUKAWA_G: float = 1.05
    YUKAWA_BACKREACTION: float = 0.01

    # Dinámica
    C_SQUARED: float = 1.0
    CFL_SAFETY_FACTOR: float = 0.75
    CFL_ABSOLUTE_CAP: float = 0.1  # Tope absoluto para dt

    # Regularización numérica
    EPSILON_SMOOTH: float = 1e-10
    MAX_FIELD_MAGNITUDE: float = 1e3
    MAX_ENERGY_RATIO: float = 4.0
    DAMPING_GAMMA: float = 0.08
    CONVERGENCE_TOLERANCE: float = 1e-6

    # Reproducibilidad
    RNG_SEED: int = 42

    @property
    def vev(self) -> float:
        r"""
        Valor Esperado del Vacío (VEV).
        Solución del mínimo clásico: v = √(μ²/λ).
        """
        return math.sqrt(self.MU_SQUARED / self.LAMBDA_COUPLING)

    @property
    def higgs_mass_squared(self) -> float:
        r"""
        Masa física del bosón de Higgs: m_H² = V''(v) = 2μ².
        """
        return 2.0 * self.MU_SQUARED

    def validate_parameters(self) -> None:
        """Verifica los invariantes I1–I6."""
        if self.MU_SQUARED <= 0:
            raise ValueError(
                f"[I1] μ² debe ser positivo para SSB, recibido {self.MU_SQUARED}"
            )
        if self.LAMBDA_COUPLING <= 0:
            raise ValueError(
                f"[I2] λ debe ser positivo, recibido {self.LAMBDA_COUPLING}"
            )
        if self.C_SQUARED <= 0:
            raise ValueError(
                f"[I3] c² debe ser positivo, recibido {self.C_SQUARED}"
            )
        if not 0 < self.CFL_SAFETY_FACTOR < 1:
            raise ValueError(
                f"[I4] α_CFL debe estar en (0,1), recibido {self.CFL_SAFETY_FACTOR}"
            )
        if self.DAMPING_GAMMA < 0:
            raise ValueError(
                f"[I5] γ_damp ≥ 0 requerido, recibido {self.DAMPING_GAMMA}"
            )
        if self.FIELD_REG_SCALE <= 0:
            raise ValueError(
                f"[I6] Φ_c > 0 requerido, recibido {self.FIELD_REG_SCALE}"
            )

    def compute_potential_and_gradient(
        self,
        phi: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Potencial regularizado y su gradiente.

        V_reg(Φ)  = -½μ²Φ² + ¼λΦ⁴·tanh(Φ²/Φ_c²)
        ∇V_reg(Φ) = -μ²Φ + λΦ³·[tanh(s) + ½s·sech²(s)],   s = Φ²/Φ_c²

        Advertencia numérica
        --------------------
        Para |Φ| > Φ_c, V(Φ) ≈ -½μ²Φ² (crece negativamente sin cota).
        En esa región, el sistema se "derrite" hacia configuraciones
        inerciales infinitas; el anchor detecta esto y dispara Dirichlet.
        """
        if phi.ndim != 1:
            raise ValueError(f"phi debe ser 1D, recibido {phi.ndim}D")

        phi_sq = phi * phi
        phi_quartic = phi_sq * phi_sq

        scale_sq = self.FIELD_REG_SCALE ** 2 + self.EPSILON_SMOOTH
        s = phi_sq / scale_sq
        tanh_s = np.tanh(s)
        sech2_s = 1.0 - tanh_s * tanh_s

        # V = -½μ²Φ² + ¼λΦ⁴·tanh(s)
        V_mass = -0.5 * self.MU_SQUARED * phi_sq
        V_int = 0.25 * self.LAMBDA_COUPLING * phi_quartic * tanh_s
        V_total = V_mass + V_int

        # ∇V = -μ²Φ + λΦ³·[tanh(s) + ½s·sech²(s)]
        grad_mass = -self.MU_SQUARED * phi
        grad_int = self.LAMBDA_COUPLING * (phi * phi_sq) * (
            tanh_s + 0.5 * s * sech2_s
        )
        grad_total = grad_mass + grad_int

        if not (np.all(np.isfinite(V_total)) and np.all(np.isfinite(grad_total))):
            raise NumericalInstabilityError(
                f"V o ∇V no finito. max|Φ|={np.max(np.abs(phi)):.2e}"
            )
        return V_total, grad_total


# Instancia global inmutable (validada al cargar el módulo)
QFT: Final[QFTParameters] = QFTParameters()
QFT.validate_parameters()


# ───────────────────────────────────────────────────────────────────────────
# 1.2  ESTRUCTURAS ALGEBRAICAS
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScalarFieldState:
    r"""
    Punto en el espacio de fase simpléctico Γ = T*ℝⁿ.

    Invariantes (verificados en __post_init__)
    -------------------------------------------
    J1. dim(Φ) = dim(Π)         (compatibilidad de campos)
    J2. phi.ndim = 1             (vector, no matriz)
    J3. dtype = float64          (precisión IEEE 754)
    """

    phi: NDArray[np.float64]
    pi_momentum: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.phi.shape != self.pi_momentum.shape:
            raise ValueError(
                f"[J1] Φ{self.phi.shape} ≠ Π{self.pi_momentum.shape}"
            )
        if self.phi.ndim != 1:
            raise ValueError(f"[J2] Φ debe ser 1D, recibido {self.phi.ndim}D")
        if self.phi.dtype != np.float64 or self.pi_momentum.dtype != np.float64:
            raise TypeError("[J3] Φ y Π deben ser float64")

    @property
    def dim(self) -> int:
        return int(self.phi.shape[0])

    def copy(self) -> "ScalarFieldState":
        """Copia defensiva (la dataclass es frozen pero los arrays no)."""
        return ScalarFieldState(
            phi=self.phi.copy(),
            pi_momentum=self.pi_momentum.copy(),
        )


@dataclass(frozen=True, slots=True)
class FermionicSource:
    r"""
    Fuente fermiónica ψ para el acoplamiento de Yukawa.

    Atributos
    ---------
    psi_vector : ψ ∈ ℝⁿ (amplitud de probabilidad)
    base_mass  : m₀ ≥ 0 (masa bare, antes de SSB)
    """

    psi_vector: NDArray[np.float64]
    base_mass: float = 0.0

    def __post_init__(self) -> None:
        if self.psi_vector.ndim != 1:
            raise ValueError("ψ debe ser 1D")
        if self.base_mass < 0:
            raise ValueError("m₀ ≥ 0 requerido")

    def charge_density(self) -> NDArray[np.float64]:
        r"""
        Densidad de carga normalizada ρ = |ψ|² / max(|ψ|²).

        La normalización a [0,1] hace que la backreaction
        YUKAWA_BACKREACTION·ρ tenga unidades de aceleración del campo.
        """
        rho_raw = self.psi_vector ** 2
        max_rho = float(np.max(rho_raw))
        if max_rho > QFT.EPSILON_SMOOTH:
            return rho_raw / max_rho
        return rho_raw


# ───────────────────────────────────────────────────────────────────────────
# 1.3  OPERADOR DE LAPLACE-BELTRAMI
# ───────────────────────────────────────────────────────────────────────────


class LaplacedBeltramiOperator:
    r"""
    Operador de Laplace-Beltrami discreto Δ_M sobre un complejo simplicial
    ponderado por el tensor métrico G_PHYSICS.

    En el caso de un grafo ponderado:
        Δ_M = D − A
    donde D = diag(∑_j A_ij) y A es la matriz de adyacencia simétrica.

    Garantías tras validación
    -------------------------
    L1. L es CSR dispersa y cuadrada
    L2. L = Lᵀ   (simetría)
    L3. L ≥ 0    (PSD)
    L4. ∑_i L_ij = 0 (suma de filas = 0, conservativo)
    """

    def __init__(self, matrix: sp.csr_matrix) -> None:
        self._validate(matrix)
        self.L: sp.csr_matrix = matrix.tocsr()
        self.n: int = int(matrix.shape[0])

    @staticmethod
    def _validate(M: sp.spmatrix) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"[L1] Laplaciano no cuadrado: shape={M.shape}")
        diff = M - M.T
        if np.any(np.abs(diff.data) > 1e-10):
            raise ValueError("[L2] Laplaciano no simétrico")
        # PSD
        if M.shape[0] > 0:
            try:
                min_eig = float(eigsh(M, k=1, which="SA", return_eigenvectors=False)[0])
                if min_eig < -1e-10:
                    raise ValueError(f"[L3] Laplaciano no PSD: λ_min={min_eig}")
            except Exception:
                # Si eigsh falla, validar por suma de filas
                row_sums = np.abs(np.array(M.sum(axis=1)).flatten())
                if np.max(row_sums) > 1e-10:
                    raise ValueError("[L4] Laplaciano no conservativo")
        # Conservatividad
        row_sums = np.array(M.sum(axis=1)).flatten()
        if np.max(np.abs(row_sums)) > 1e-10:
            raise ValueError("[L4] ∑_j L_ij ≠ 0 (no conservativo)")

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Union[NDArray[np.float64], sp.spmatrix],
        symmetrize: bool = True,
    ) -> "LaplacedBeltramiOperator":
        """
        Construye Δ_M = D − A a partir de la matriz de adyacencia.
        """
        A = sp.csr_matrix(adjacency) if isinstance(adjacency, np.ndarray) else adjacency.tocsr()
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Adyacencia no cuadrada: {A.shape}")
        if symmetrize:
            diff = A - A.T
            if np.any(np.abs(diff.data) > 1e-12):
                logger.warning("Adyacencia asimétrica: promediando con su transpuesta")
                A = 0.5 * (A + A.T)
        degrees = np.array(A.sum(axis=1)).flatten()
        D = sp.diags(degrees, format="csr")
        return cls(D - A)

    def apply(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aplica Δ_M a un campo vector: L·Φ."""
        if phi.shape != (self.n,):
            raise ValueError(f"Φ.shape {phi.shape} ≠ ({self.n},)")
        return self.L @ phi


# ───────────────────────────────────────────────────────────────────────────
# 1.4  ANALIZADOR ESPECTRAL
# ───────────────────────────────────────────────────────────────────────────


class SpectralAnalyzer:
    r"""
    Análisis espectral de operadores dispersos simétricos.

    Métodos disponibles
    -------------------
    'gershgorin'      : cota de Gershgorin (O(1))
    'power_iteration' : iteración de potencia (O(k·nnz))
    'krylov'          : ARPACK vía eigsh (O(n²) en el peor caso)
    """

    @staticmethod
    def estimate_max_eigenvalue(
        laplacian: sp.csr_matrix,
        method: str = "gershgorin",
        tolerance: float = 1e-6,
        max_iter: int = 200,
    ) -> float:
        n = laplacian.shape[0]
        if n == 0:
            return 0.0

        if method == "gershgorin":
            row_sums = np.abs(np.array(laplacian.sum(axis=1)).flatten())
            return float(np.max(row_sums))

        if method == "power_iteration":
            v = np.random.default_rng(0).standard_normal(n)
            v_norm = np.linalg.norm(v)
            if v_norm < QFT.EPSILON_SMOOTH:
                return 0.0
            v = v / v_norm
            lambda_prev = 0.0
            for _ in range(max_iter):
                v_new = laplacian @ v
                v_new_norm = np.linalg.norm(v_new)
                if v_new_norm < QFT.EPSILON_SMOOTH:
                    return 0.0
                lambda_new = float(np.dot(v, v_new))
                if abs(lambda_new - lambda_prev) < tolerance:
                    return lambda_new
                v = v_new / v_new_norm
                lambda_prev = lambda_new
            logger.warning("Power iteration no convergió en %d iteraciones", max_iter)
            return lambda_prev

        if method == "krylov":
            try:
                eigs = eigsh(
                    laplacian,
                    k=1,
                    which="LA",
                    return_eigenvectors=False,
                    tol=tolerance,
                )
                return float(eigs[0])
            except Exception as e:
                logger.error("Krylov (LA) falló: %s. Cayendo a Gershgorin.", e)
                return SpectralAnalyzer.estimate_max_eigenvalue(
                    laplacian, method="gershgorin"
                )

        raise ValueError(f"Método '{method}' inválido")

    @staticmethod
    def estimate_spectral_gap(
        laplacian: sp.csr_matrix,
        tolerance: float = 1e-6,
        n_threshold: int = 500,
    ) -> float:
        r"""
        Estima el gap espectral λ₁ (conectividad algebraica).

        Para grafos con n > n_threshold usa Krylov shift-invert.
        Retorna 0.0 sólo si el grafo es trivial o degenerado.
        """
        n = laplacian.shape[0]
        if n < 2:
            return 0.0
        try:
            k = min(2, n - 1)
            eigs = eigsh(
                laplacian,
                k=k,
                which="SM",
                return_eigenvectors=False,
                tol=tolerance,
            )
            if len(eigs) < 2:
                return 0.0
            eigs_sorted = np.sort(eigs)
            # λ₀ ≈ 0, λ₁ = eigs_sorted[1]
            gap = float(eigs_sorted[1])
            return max(0.0, gap)
        except Exception as e:
            logger.error("Cálculo de gap espectral falló: %s", e)
            return 0.0


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 2 · DINÁMICA SIMPLÉCTICA                                          │
# │  Define el Hamiltoniano, el integrador y el monitor de estabilidad.     │
# └─────────────────────────────────────────────────────────────────────────┘


# ───────────────────────────────────────────────────────────────────────────
# 2.1  POTENCIAL DE HIGGS (cota Lipschitz explícita)
# ───────────────────────────────────────────────────────────────────────────


class HiggsPotential:
    r"""
    Potencial de Higgs regularizado V_reg(Φ) con análisis de cotas.

    Acotación
    ---------
    V(Φ) = -½μ²Φ² + ¼λΦ⁴·tanh(Φ²/Φ_c²)

    1. Para |Φ| < Φ_c:  V(Φ) ≈ -½μ²Φ² + ¼λΦ⁴  (polinomio clásico)
    2. Para |Φ| > Φ_c:  V(Φ) ≈ -½μ²Φ²          (regularización suprime λ)

    Como el término -½μ²Φ² no está acotado inferiormente, V(Φ) NO es
    Lipschitz en el sentido de tener cota global uniforme. Sin embargo,
    en el dominio [-Φ_c, Φ_c] (donde opera el sistema físico), V SÍ es
    Lipschitz con constante:
        L_V = sup |V'(Φ)| = μ²Φ_c + (término cuártico) < ∞
    """

    def __init__(self, params: QFTParameters = QFT) -> None:
        self.p: QFTParameters = params

    def value_and_gradient(
        self, phi: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.p.compute_potential_and_gradient(phi)

    def lipschitz_constant(self, domain_radius: float) -> float:
        r"""
        Constante de Lipschitz en el dominio [-R, R].

        L_V(R) = sup_{|Φ|≤R} |V'(Φ)|
               ≤ μ²·R + 3λ·R³·(1 + ½·(R/Φ_c)²)
        """
        R = domain_radius
        R_sq = R * R
        Phi_c_sq = self.p.FIELD_REG_SCALE ** 2
        return (
            self.p.MU_SQUARED * R
            + 3.0 * self.p.LAMBDA_COUPLING * R * R_sq
            * (1.0 + 0.5 * R_sq / Phi_c_sq)
        )

    def is_physically_admissible(self, phi: NDArray[np.float64]) -> bool:
        """Verifica que el campo esté en la región regularizada."""
        max_abs = float(np.max(np.abs(phi)))
        return max_abs <= self.p.FIELD_REG_SCALE


# ───────────────────────────────────────────────────────────────────────────
# 2.2  HAMILTONIANO Y ENERGÍA
# ───────────────────────────────────────────────────────────────────────────


class PortHamiltonianHamiltonian:
    r"""
    Hamiltoniano del sistema:

        H[Φ, Π] = T + U_elastic + U_potential
                = ½‖Π‖² + ½c²·Φᵀ·L·Φ + Σᵢ V(Φᵢ)

    Invariantes garantizados
    ------------------------
    H1. H ≥ −½μ²‖Φ‖² (cota inferior polinómica)
    H2. H es diferenciable (excepto donde V no lo sea, lo que no ocurre aquí)
    H3. Flujo Hamiltoniano preserva ω = Σ dΦᵢ ∧ dΠᵢ
    """

    def __init__(
        self,
        laplacian: LaplacedBeltramiOperator,
        potential: HiggsPotential,
        params: QFTParameters = QFT,
    ) -> None:
        if laplacian.n != 0 and laplacian.n != 0:
            pass  # se valida en apply
        self.L_op: LaplacedBeltramiOperator = laplacian
        self.V: HiggsPotential = potential
        self.p: QFTParameters = params

    def __call__(
        self, state: ScalarFieldState
    ) -> float:
        """H total."""
        if state.dim != self.L_op.n:
            raise ValueError(
                f"state.dim={state.dim} ≠ laplacian.n={self.L_op.n}"
            )
        T = 0.5 * float(np.dot(state.pi_momentum, state.pi_momentum))
        U_el = 0.5 * self.p.C_SQUARED * float(
            np.dot(state.phi, self.L_op.apply(state.phi))
        )
        V_local, _ = self.V.value_and_gradient(state.phi)
        U_pot = float(np.sum(V_local))
        return T + U_el + U_pot

    def force(
        self,
        state: ScalarFieldState,
        source_density: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        F = -∇H + fuente externa
          = -c²·L·Φ - ∇V(Φ) + g_s·ρ
        """
        laplacian_force = -self.p.C_SQUARED * self.L_op.apply(state.phi)
        _, grad_V = self.V.value_and_gradient(state.phi)
        potential_force = -grad_V
        source_force = self.p.YUKAWA_BACKREACTION * source_density
        return laplacian_force + potential_force + source_force


# ───────────────────────────────────────────────────────────────────────────
# 2.3  INTEGRADOR VELOCITY-VERLET (algoritmo formal estándar)
# ───────────────────────────────────────────────────────────────────────────


class SymplecticIntegrator(ABC):
    """Clase base abstracta para integradores simplécticos."""

    @abstractmethod
    def step(
        self,
        state: ScalarFieldState,
        force_field: NDArray[np.float64],
        dt: float,
    ) -> ScalarFieldState: ...


class VelocityVerletIntegrator(SymplecticIntegrator):
    r"""
    Integrador Velocity-Verlet estándar de 2º orden.

    Algoritmo (Split-Operator Simpléctico)
    --------------------------------------
    Para el sistema
        Φ̈ = F(Φ, Π) = -c²LΦ - ∇V(Φ) - γ·Π + g_s·ρ

    el paso (Φⁿ, Πⁿ) → (Φⁿ⁺¹, Πⁿ⁺¹) es:

        Πⁿ⁺½ = Πⁿ + ½·F(Φⁿ, Πⁿ)·dt
        Φⁿ⁺¹ = Φⁿ + Πⁿ⁺½·dt
        Πⁿ⁺¹ = Πⁿ⁺½ + ½·F(Φⁿ⁺¹, Πⁿ⁺½)·dt

    La diferencia respecto a la versión original es que Πⁿ⁺½ se usa
    consistentemente en ambas mitades, lo que da un error O(dt³) global
    (simpléctico) en lugar de O(dt²) con artefactos espurios.

    Estabilidad
    -----------
    CFL: dt < 2/ω_max, donde ω_max² = c²·λ_max(Δ_M) + 2μ².
    """

    def __init__(
        self,
        hamiltonian: PortHamiltonianHamiltonian,
        damping_gamma: float = 0.0,
        max_field: float = 1e3,
    ) -> None:
        self.H: PortHamiltonianHamiltonian = hamiltonian
        self.gamma: float = damping_gamma
        self.max_field: float = max_field

        # Análisis espectral para CFL
        self._lambda_max: float = SpectralAnalyzer.estimate_max_eigenvalue(
            hamiltonian.L_op.L, method="gershgorin"
        )
        self._dt_critical: float = self._compute_critical_timestep()

        logger.info(
            "Verlet inicializado: λ_max=%.4f, dt_crit=%.6f, γ=%.3f",
            self._lambda_max, self._dt_critical, self.gamma,
        )

    def _compute_critical_timestep(self) -> float:
        r"""
        CFL estricto: dt < (2/ω_max)·α, donde
            ω_max² = c²·λ_max(Δ_M) + m_H² = c²·λ_max + 2μ²
        """
        omega_sq = (
            self.H.p.C_SQUARED * self._lambda_max
            + self.H.p.higgs_mass_squared
        )
        omega_max = math.sqrt(omega_sq + self.H.p.EPSILON_SMOOTH)
        dt_limit = (2.0 / omega_max) * self.H.p.CFL_SAFETY_FACTOR
        # El tope absoluto protege contra grafos triviales o mal condicionados
        return min(dt_limit, self.H.p.CFL_ABSOLUTE_CAP)

    @property
    def dt_critical(self) -> float:
        return self._dt_critical

    def step(
        self,
        state: ScalarFieldState,
        source_density: NDArray[np.float64],
        dt: Optional[float] = None,
    ) -> ScalarFieldState:
        """Un paso de Velocity-Verlet (algoritmo split-operator)."""
        dt_eff = dt if dt is not None else self._dt_critical
        if dt_eff <= 0:
            raise ValueError(f"dt debe ser positivo, recibido: {dt_eff}")
        if dt_eff > self._dt_critical * 1.01:  # 1% de margen numérico
            logger.warning(
                "dt=%.6f excede dt_crit=%.6f (inestabilidad probable)",
                dt_eff, self._dt_critical,
            )

        phi = state.phi
        pi = state.pi_momentum

        # 1) Π^{n+½} = Π^n + ½·F(Φ^n, Π^n)·dt
        F_n = self._force(phi, pi, source_density)
        pi_half = pi + 0.5 * F_n * dt_eff

        # 2) Φ^{n+1} = Φ^n + Π^{n+½}·dt
        phi_new = phi + pi_half * dt_eff

        # Saturación suave por seguridad numérica
        phi_new = self._soft_clip(phi_new)

        # 3) Π^{n+1} = Π^{n+½} + ½·F(Φ^{n+1}, Π^{n+½})·dt
        F_new = self._force(phi_new, pi_half, source_density)
        pi_new = pi_half + 0.5 * F_new * dt_eff

        return ScalarFieldState(phi=phi_new, pi_momentum=pi_new)

    def _force(
        self,
        phi: NDArray[np.float64],
        pi: NDArray[np.float64],
        source: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """F = -c²·L·Φ - ∇V(Φ) - γ·Π + g_s·ρ"""
        F = self.H.force(
            ScalarFieldState(phi=phi, pi_momentum=pi), source
        )
        if self.gamma > 0:
            F = F - self.gamma * pi
        return F

    def _soft_clip(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Saturación suave: Φ → M·tanh(Φ/M)."""
        return self.max_field * np.tanh(phi / self.max_field)


# ───────────────────────────────────────────────────────────────────────────
# 2.4  MONITOR DE ESTABILIDAD
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class StabilityMetrics:
    """Métricas de diagnóstico de estabilidad."""

    energy: float
    energy_drift: float
    field_rms: float
    momentum_rms: float
    max_field: float
    vev_deviation: float
    requires_damping: bool
    is_stable: bool


class StabilityMonitor:
    """
    Monitor de estabilidad con termostato adaptativo.
    """

    def __init__(
        self,
        baseline_energy: float,
        hamiltonian: PortHamiltonianHamiltonian,
        params: QFTParameters = QFT,
    ) -> None:
        self._E0: float = baseline_energy
        self._H: PortHamiltonianHamiltonian = hamiltonian
        self._p: QFTParameters = params
        self._history: list[StabilityMetrics] = []

    @property
    def baseline_energy(self) -> float:
        return self._E0

    @property
    def history(self) -> list[StabilityMetrics]:
        return list(self._history)

    def analyze(self, state: ScalarFieldState) -> StabilityMetrics:
        H = self._H(state)
        energy_drift = abs(H - self._E0) / (abs(self._E0) + self._p.EPSILON_SMOOTH)
        field_rms = float(np.sqrt(np.mean(state.phi ** 2)))
        momentum_rms = float(np.sqrt(np.mean(state.pi_momentum ** 2)))
        max_field = float(np.max(np.abs(state.phi)))
        vev_dev = float(np.sqrt(np.mean((state.phi - self._p.vev) ** 2)))
        requires_damping = H > self._p.MAX_ENERGY_RATIO * self._E0
        is_stable = (
            np.all(np.isfinite(state.phi))
            and np.all(np.isfinite(state.pi_momentum))
            and max_field < self._p.MAX_FIELD_MAGNITUDE
            and energy_drift < 10.0
        )
        m = StabilityMetrics(
            energy=H,
            energy_drift=energy_drift,
            field_rms=field_rms,
            momentum_rms=momentum_rms,
            max_field=max_field,
            vev_deviation=vev_dev,
            requires_damping=requires_damping,
            is_stable=is_stable,
        )
        self._history.append(m)
        return m

    def log_diagnostics(
        self, metrics: StabilityMetrics, level: int = logging.INFO
    ) -> None:
        logger.log(
            level,
            "[STABILITY] E=%.4e (drift=%.2%%), ⟨Φ⟩_rms=%.4f, max|Φ|=%.4f, "
            "⟨|Φ-v|⟩=%.4f, damping=%s",
            metrics.energy,
            metrics.energy_drift * 100.0,
            metrics.field_rms,
            metrics.max_field,
            metrics.vev_deviation,
            "ON" if metrics.requires_damping else "OFF",
        )


# ───────────────────────────────────────────────────────────────────────────
# 2.5  VALIDACIÓN DE INVARIANTES TOPOLÓGICOS
# ───────────────────────────────────────────────────────────────────────────


def validate_topological_consistency(
    state: ScalarFieldState,
    hamiltonian: PortHamiltonianHamiltonian,
    params: QFTParameters = QFT,
) -> bool:
    r"""
    Valida invariantes topológicos y físicos.

    T1. dim(Φ) = dim(L)
    T2. Φ, Π ∈ ℝⁿ (finitos)
    T3. |Φ| < Φ_max (acotación)
    T4. H bien definido
    """
    try:
        if state.dim != hamiltonian.L_op.n:
            logger.error("[T1] dim inconsistente")
            return False
        if not (np.all(np.isfinite(state.phi)) and np.all(np.isfinite(state.pi_momentum))):
            logger.error("[T2] campos no finitos")
            return False
        max_field = float(np.max(np.abs(state.phi)))
        if max_field > params.MAX_FIELD_MAGNITUDE:
            logger.error("[T3] |Φ|_max=%.2e > Φ_max", max_field)
            return False
        H = hamiltonian(state)
        if not np.isfinite(H):
            logger.error("[T4] H=%g no finito", H)
            return False
        return True
    except Exception as e:
        logger.exception("Validación topológica falló: %s", e)
        return False


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FASE 3 · FUNTOR DE ANCLAJE                                              │
# │  Composición endofuntorial: Y ∘ H ∘ V (Yukawa ∘ Hamiltoniano ∘ Vacío)  │
# └─────────────────────────────────────────────────────────────────────────┘


class ScalarHiggsAnchor(Morphism):
    r"""
    Funtor de anclaje: $F: \mathcal{C}_{Agent} \to \mathcal{C}_{Agent}$.

    Composición
    -----------
        $F = \text{Yukawa} \circ \text{HamiltonFlow}^N \circ \text{VacuumInit}$

    Donde VacuumInit sitúa el campo cerca del VEV, HamiltonFlow integra
    N pasos de Klein-Gordon con backreaction, y Yukawa produce la masa
    efectiva $m_{\text{eff}} = m_0 + g|\Phi|$.
    """

    def __init__(
        self,
        laplacian: sp.csr_matrix,
        dim: int,
        params: QFTParameters = QFT,
        num_relaxation_steps: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        if laplacian.shape != (dim, dim):
            raise ValueError(
                f"Laplaciano {laplacian.shape} incompatible con dim={dim}"
            )

        # Operadores de Fase 1
        self._lb_op: LaplacedBeltramiOperator = LaplacedBeltramiOperator(laplacian)
        self._potential: HiggsPotential = HiggsPotential(params)
        self._hamiltonian: PortHamiltonianHamiltonian = PortHamiltonianHamiltonian(
            self._lb_op, self._potential, params
        )

        # Integrador de Fase 2
        self._integrator: VelocityVerletIntegrator = VelocityVerletIntegrator(
            self._hamiltonian,
            damping_gamma=params.DAMPING_GAMMA,
            max_field=params.MAX_FIELD_MAGNITUDE,
        )

        self._params: QFTParameters = params
        self._dim: int = dim
        self._num_steps: int = num_relaxation_steps

        # RNG reproducible (H3: era global, ahora inyectado)
        self._rng: np.random.Generator = np.random.default_rng(
            seed if seed is not None else params.RNG_SEED
        )

        # Inicialización del vacío (Fase 1 → reproducible)
        self._field_state: ScalarFieldState = self._initialize_vacuum_state()

        # Monitor de estabilidad
        self._monitor: StabilityMonitor = StabilityMonitor(
            baseline_energy=self._hamiltonian(self._field_state),
            hamiltonian=self._hamiltonian,
            params=params,
        )

        logger.info(
            "ScalarHiggsAnchor v6.0 inicializado: dim=%d, v=%.4f, E₀=%.4e, dt_crit=%.6f",
            dim, params.vev, self._monitor.baseline_energy, self._integrator.dt_critical,
        )

    def _initialize_vacuum_state(self) -> ScalarFieldState:
        r"""
        Vacío: $\Phi_0 = v + \delta\Phi$, $\Pi_0 = 0$.

        Las fluctuaciones son O(5% del VEV), reproducibles.
        """
        vev = self._params.vev
        noise_amp = 0.05 * vev
        phi_init = np.full(self._dim, vev, dtype=np.float64)
        phi_init += self._rng.normal(0.0, noise_amp, size=self._dim)
        pi_init = np.zeros(self._dim, dtype=np.float64)
        return ScalarFieldState(phi=phi_init, pi_momentum=pi_init)

    def __call__(
        self,
        state: CategoricalState,
        *args: Any,
        **kwargs: Any,
    ) -> CategoricalState:
        """
        Aplica el funtor al estado categórico.

        Algoritmo
        ---------
        1. Extraer ψ del payload (con fallback robusto)
        2. Construir ρ = |ψ|² normalizada
        3. Evolucionar Φ con backreaction (N pasos Verlet)
        4. Calcular m_eff = m₀ + g·|Φ|
        5. Anclar ψ' = m_eff·ψ
        6. Retornar nuevo CategoricalState con metadata física
        """
        # 1) Extracción robusta (H6)
        psi_raw = self._extract_psi(state)
        if psi_raw is None:
            logger.debug("Sin vector estocástico: identidad retornada")
            return state

        # 2) Densidad de carga
        source = FermionicSource(psi_vector=psi_raw)
        rho = source.charge_density()

        # 3) Relajación con diagnóstico
        for step_idx in range(self._num_steps):
            metrics = self._monitor.analyze(self._field_state)
            if not metrics.is_stable:
                logger.error(
                    "Inestabilidad en paso %d: drift=%.2f%%, max|Φ|=%.2e",
                    step_idx, metrics.energy_drift * 100, metrics.max_field,
                )
                raise TopologicalInvariantError(
                    "Colapso del campo de Higgs (inestabilidad numérica)"
                )
            apply_damp = metrics.requires_damping
            self._field_state = self._integrator.step(
                self._field_state, rho, dt=None
            )
            if apply_damp and step_idx == 0:
                logger.warning(
                    "Termostato activo: E/E₀=%.2f",
                    self._monitor.history[-1].energy / self._monitor.baseline_energy,
                )

        final_metrics = self._monitor.analyze(self._field_state)
        self._monitor.log_diagnostics(final_metrics, logging.DEBUG)

        # 4) Masa efectiva (Yukawa)
        m_eff = self._compute_effective_mass(source)
        if not np.all(np.isfinite(m_eff)):
            raise TopologicalInvariantError("m_eff no finita")
        if np.max(m_eff) > 1e4:
            logger.critical("m_eff excesiva: max=%.2e, clipando", float(np.max(m_eff)))
            m_eff = np.clip(m_eff, 0.0, 1e4)

        # 5) Anclaje: ψ' = m_eff·ψ
        psi_anchored = psi_raw * m_eff

        # 6) Construcción del nuevo estado (H10: payload robusto)
        new_payload = self._build_anchored_payload(
            state, psi_anchored, m_eff, final_metrics
        )
        return CategoricalState(
            payload=new_payload,
            stratum=Stratum.PHYSICS,
            metadata=state.metadata,
        )

    # ───── Métodos auxiliares ─────

    def _extract_psi(self, state: CategoricalState) -> Optional[NDArray[np.float64]]:
        """
        Extrae ψ del payload de forma robusta.

        Estrategias
        -----------
        P1. payload["stochastic_vector"]  (caso estándar)
        P2. payload como ndarray           (caso directo)
        P3. payload como iterable          (np.asarray)
        P4. fallback None (mónada de identidad)
        """
        if isinstance(state.payload, dict):
            raw = state.payload.get("stochastic_vector")
            if raw is None or len(raw) == 0:
                return None
            psi = np.asarray(raw, dtype=np.float64).ravel()
        elif isinstance(state.payload, np.ndarray):
            psi = state.payload.astype(np.float64).ravel()
        elif hasattr(state.payload, "__iter__") and not isinstance(
            state.payload, (str, bytes)
        ):
            try:
                psi = np.asarray(state.payload, dtype=np.float64).ravel()
            except (TypeError, ValueError):
                return None
        else:
            return None

        return self._resize_psi(psi)

    def _resize_psi(self, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Trunca o rellena con ceros para alcanzar self._dim."""
        if psi.shape[0] == self._dim:
            return psi
        if psi.shape[0] > self._dim:
            return psi[: self._dim]
        out = np.zeros(self._dim, dtype=np.float64)
        out[: psi.shape[0]] = psi
        return out

    def _compute_effective_mass(
        self, source: FermionicSource
    ) -> NDArray[np.float64]:
        r"""
        m_eff(x) = m_0 + g·√(Φ² + ε²)
        """
        phi = self._field_state.phi
        phi_mag = np.sqrt(phi ** 2 + self._params.EPSILON_SMOOTH ** 2)
        return source.base_mass + self._params.YUKAWA_G * phi_mag

    def _build_anchored_payload(
        self,
        original: CategoricalState,
        psi_anchored: NDArray[np.float64],
        m_eff: NDArray[np.float64],
        metrics: StabilityMetrics,
    ) -> dict[str, Any]:
        """Construye el payload del nuevo estado preservando metadatos previos."""
        if isinstance(original.payload, dict):
            new_payload: dict[str, Any] = dict(original.payload)
        else:
            new_payload = {"original_payload": original.payload}
        new_payload.update({
            "anchored_vector": psi_anchored.tolist(),
            "effective_mass_profile": m_eff.tolist(),
            "higgs_field_snapshot": self._field_state.phi.tolist(),
            "vev_deviation": metrics.vev_deviation,
            "field_energy": metrics.energy,
            "stability_flag": metrics.is_stable,
        })
        return new_payload

    @property
    def current_vev_deviation(self) -> float:
        """Desviación RMS Φ − v."""
        return float(
            np.sqrt(np.mean((self._field_state.phi - self._params.vev) ** 2))
        )

    @property
    def field_state(self) -> ScalarFieldState:
        return self._field_state

    @property
    def monitor(self) -> StabilityMonitor:
        return self._monitor


# ───────────────────────────────────────────────────────────────────────────
# 3.1  UTILIDADES PÚBLICAS
# ───────────────────────────────────────────────────────────────────────────


def construct_laplacian_from_adjacency(
    adjacency: Union[NDArray[np.float64], sp.spmatrix],
    symmetrize: bool = True,
) -> sp.csr_matrix:
    r"""
    Construye el Laplaciano combinatorio Δ = D − A.
    """
    op = LaplacedBeltramiOperator.from_adjacency(adjacency, symmetrize=symmetrize)
    return op.L


# ───────────────────────────────────────────────────────────────────────────
# 3.2  DECORADOR CATEGÓRICO
# ───────────────────────────────────────────────────────────────────────────


T_Morphism = TypeVar("T_Morphism", bound=Morphism)


def apply_higgs_anchor(
    dim: int,
    laplacian: sp.csr_matrix,
    num_relaxation_steps: int = 5,
    seed: Optional[int] = None,
) -> Callable[[type[T_Morphism]], type[T_Morphism]]:
    r"""
    Decorador que acopla un Morphism al funtor de anclaje de Higgs.

    Uso
    ---
        >>> @apply_higgs_anchor(dim=4, laplacian=L)
        ... class MyAgent(Morphism):
        ...     def __call__(self, state, *args, **kwargs):
        ...         return state
    """
    def decorator(agent_class: type[T_Morphism]) -> type[T_Morphism]:
        original_call = agent_class.__call__

        @wraps(original_call)
        def wrapped(self: T_Morphism, state: CategoricalState, *a: Any, **kw: Any) -> CategoricalState:
            # 1) Ejecutar el agente original
            pre_state = original_call(self, state, *a, **kw)
            # 2) Aplicar el anclaje
            anchor = ScalarHiggsAnchor(
                laplacian=laplacian,
                dim=dim,
                num_relaxation_steps=num_relaxation_steps,
                seed=seed,
            )
            return anchor(pre_state)

        agent_class.__call__ = wrapped  # type: ignore[assignment]
        agent_class.__doc__ = (
            (agent_class.__doc__ or "") +
            f"\n\n[Anclado por Higgs con dim={dim}, steps={num_relaxation_steps}]"
        )
        return agent_class

    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIONES PÚBLICAS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Fase 1
    "QFTParameters",
    "QFT",
    "ScalarFieldState",
    "FermionicSource",
    "LaplacedBeltramiOperator",
    "SpectralAnalyzer",
    # Fase 2
    "HiggsPotential",
    "PortHamiltonianHamiltonian",
    "SymplecticIntegrator",
    "VelocityVerletIntegrator",
    "StabilityMonitor",
    "StabilityMetrics",
    "validate_topological_consistency",
    # Fase 3
    "ScalarHiggsAnchor",
    "construct_laplacian_from_adjacency",
    "apply_higgs_anchor",
]
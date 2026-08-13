# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Symplectic Verlet Integrator (Conservador de la Forma Simpléctica)  ║
║ Ruta   : app/physics/symplectic_verlet_integrator.py                         ║
║ Versión: 3.1.0-Verlet-Yoshida-PHS-Neumaier-Liouville-Strict                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL (Fases Anidadas Evolucionadas):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la maestría de la integración geométrica sobre el espacio 
de fase en el Estrato PHYSICS. A diferencia de los integradores de Runge-Kutta 
clásicos, que violan la simplecticidad desde órdenes inferiores, este integrador 
conserva de manera asintótica y estructurada la 2-forma simpléctica $\omega$, 
blindando la fidelidad física y termodinámica del foso numérico ante las 
perturbaciones estocásticas de la Inteligencia Artificial.

La versión 3.1.0 evoluciona el motor incorporando la diferenciación por paso complejo
(CSD) y el cálculo analítico exacto de la matriz Jacobiana de transición de fase.
Esto permite la verificación de la preservación de la forma simpléctica a nivel
del épsilon de máquina, eludiendo la inestabilidad de las diferencias finitas.

INVARIANTES MATEMÁTICOS, TOPOLÓGICOS Y LEYES CONSERVATIVAS PRESERVADOS:
────────────────────────────────────────────────────────────────────────────────
  [I1] Preservación de la 2-Forma Simpléctica Canónica:
       Para todo paso de integración temporal en régimen conservativo ($R_d = \mathbf{0}$), 
       el flujo discreto conserva de forma exacta la forma bilineal simpléctica:
       $$\omega = \sum_{i=1}^n dq_i \wedge dp_i$$
       Esto exige que el Jacobiano local del paso $J_{\mathrm{map}}$ satisfaga:
       $$J_{\mathrm{map}}^\top \Omega J_{\mathrm{map}} = \Omega \quad\big[4, 93\big]$$
       Donde el tensor simpléctico elíptico canónico $\Omega \in \mathbb{R}^{2n \times 2n}$ 
       se define en bloques matriciales como:
       $$\Omega = \begin{pmatrix} \mathbf{0} & \mathbf{I}_n \\ -\mathbf{I}_n & \mathbf{0} \end{pmatrix}$$

  [I2] Teorema de Liouville y Conservación del Volumen de Fase:
       Al ser la transición local un simplectomorfismo exacto, se hereda la 
       preservación del volumen en el espacio de fase, garantizando la divergencia 
       nula del flujo de control en la variedad simpléctica:
       $$\det(J_{\mathrm{map}}) = 1 \implies \operatorname{div}(\dot{x}) \equiv 0 \quad\big[4\big]$$

  [I3] Pasividad Termodinámica y Disipación de Rayleigh:
       El sistema Port-Hamiltoniano se somete a la desigualdad de Clausius-Duhem:
       $$\dot{H}_d = -\nabla H_d^\top R_d \nabla H_d \le 0 \pmod{\varepsilon_{\mathrm{machine}}} \quad\big[8, 25\big]$$
       Donde $H_d(q, p) = T(p) + V(q)$ es la energía Hamiltoniana basal amortiguada 
       por la matriz simétrica semidefinida positiva de Rayleigh $R_d \succeq \mathbf{0}$.
       Cualquier inyección parásita de energía ($\dot{H}_d > 0$) colapsa el lazo.

  [I4] Estabilidad Asintótica BIBO (Abscisa Espectral y Lyapunov):
       Para garantizar la convergencia hacia el atractor de geodésica de menor coste, 
       los polos del sistema en el plano de frecuencia de Laplace complejos $s = \sigma + j\omega$ 
       deben residir estrictamente en el semiplano izquierdo (LHP):
       $$\forall p_i \in \sigma(H(s)), \quad \Re(p_i) < 0 \implies \lambda_{\mathrm{Lyapunov}} < 0 \quad\big[10\big]$$

  [I5] Composición de Yoshida-Takahashi de Cuarto Orden:
       Para lograr alta precisión de cuarto orden, se componen tres pasos de 
       Störmer-Verlet con sub-pasos temporales $dt_i = w_i \Delta t$ mediante coeficientes:
       $$w_1 = \frac{1}{2 - 2^{1/3}}, \qquad w_0 = 1 - 2 w_1$$
       Los cuales satisfacen estrictamente los axiomas algebraicos de Lie:
       $$w_0 + 2w_1 = 1, \qquad w_0^3 + 2w_1^3 = 0 \quad\big[4, 23\big]$$

ESTRUCTURA DE TRES FASES ANIDADAS (Composición Funtorial):
────────────────────────────────────────────────────────────────────────────────
La transición de estados se rige por un contrato algebraico rígido donde la salida 
inmutable de cada fase actúa como la precondición formal de la siguiente:

  Fase 1 ──► FASE 1: ESPECTROSCOPÍA DEL MOMENTUM (Phase1_PortHamiltonianSpectrometer)
             Verifica que la matriz de inercia inversa $M^{-1}$ sea real, simétrica 
             definida positiva (SPD) y que su número de condición espectral cumpla 
             con la cota de Wilkinson.
             Entrega: MechanicalBundleData como precondición formal de la Fase 2.

  Fase 2 ──► FASE 2: SÍNTESIS VARIACIONAL Y FLUJO DE FASE (Phase2_VariationalFlowSynthesizer)
             Resuelve el Kick-Drift-Kick del paso de Störmer-Verlet y calcula 
             el Jacobiano de transición $J_{\mathrm{map}}$ mediante diferenciación por 
             Paso Complejo (CSD) para eludir la pérdida de significancia en la FPU:
             $$J_{\mathrm{map}, \, ij} = \frac{\operatorname{Im}(\Phi_{\Delta t}(x + j h \cdot e_i)_j)}{h} + \mathcal{O}(h^2)$$\n             Entrega: PhaseTransitionData como precondición formal de la Fase 2.

  Fase 3 ──► FASE 3: COMPOSICIÓN DE YOSHIDA Y VEREDICTO DE LYAPUNOV (Phase3_YoshidaThermodynamicIntegrator)
             Combina los sub-pasos de Yoshida, calcula la energía total $H = T + V$ 
             empleando la forma de Neumaier y valida el balance de Rayleigh.
             Entrega: SymplecticIntegratorReport como certificado final de viabilidad de fase.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        """Violación base de un invariante topológico-categórico."""

        pass

    class Morphism:
        """Clase base de composición funtorial del ecosistema MIC."""

        pass


logger = logging.getLogger("MIC.Physics.SymplecticVerletIntegrator")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES ESPECTRALES, YOSHIDA Y LÍMITES TERMODINÁMICOS
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_MIN_DT: Final[float] = 1.0e-15
_MAX_DT: Final[float] = 1.0e6
_CONDITION_NUMBER_MAX: Final[float] = 1e12
_WILKINSON_CONSTANT: Final[float] = 8.0
_CSD_STEP_DEFAULT: Final[float] = 1.0e-20
_CSD_STEP_MIN: Final[float] = 1.0e-30
_CSD_STEP_MAX: Final[float] = 1.0e-8
_HIGHAM_EIGEN_FLOOR: Final[float] = 1.0e-12

# Pesos de Yoshida–Forest–Ruth.  w0 := 1-2 w1 impone consistencia exacta.
_YOSHIDA_CBRT2: Final[float] = float(2.0 ** (1.0 / 3.0))
_YOSHIDA_W1: Final[float] = float(1.0 / (2.0 - _YOSHIDA_CBRT2))
_YOSHIDA_W0: Final[float] = float(1.0 - 2.0 * _YOSHIDA_W1)

# Umbrales de veredicto sobre residuos geométricos.
_SYMPLECTIC_SOFT_TOL: Final[float] = 1.0e-8
_SYMPLECTIC_HARD_TOL: Final[float] = 1.0e-4
_LIOUVILLE_SOFT_TOL: Final[float] = 1.0e-8
_LIOUVILLE_HARD_TOL: Final[float] = 1.0e-4
_ENERGY_SOFT_TOL: Final[float] = 1.0e-8
_ENERGY_HARD_TOL: Final[float] = 1.0e-3
_PASSIVITY_SOFT_TOL: Final[float] = 1.0e-10
_PASSIVITY_HARD_TOL: Final[float] = 1.0e-6

# Tolerancias de invariancia por método de Jacobiano (flujo conservativo).
_JACOBIAN_INVARIANCE_TOL: Final[dict[str, float]] = {
    "numerical": 1.0e-5,
    "csd": 1.0e-11,
    "analytical": 1.0e-12,
}

__version__: Final[str] = "3.0.0-Verlet-Yoshida-PHS-Neumaier-Liouville-Strict"


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES SIMPLÉCTICAS
# ══════════════════════════════════════════════════════════════════════════════
class SymplecticIntegrationError(TopologicalInvariantError):
    """Excepción raíz del integrador simpléctico."""

    pass


class SystemConfigurationError(SymplecticIntegrationError):
    """El bundle Port-Hamiltoniano no satisface el contrato espectral."""

    pass


class IntegrationStepError(SymplecticIntegrationError):
    """Fallo durante la ejecución de un mapa de Verlet / Yoshida."""

    pass


class SymplecticInvarianceError(SymplecticIntegrationError):
    """Violación de $$J^\top\Omega J=\Omega$$ por encima de la cota dura."""

    pass


class PassivityViolationError(SymplecticIntegrationError):
    """La tasa de Rayleigh deja de ser no positiva."""

    pass


class MassMatrixSPDError(SystemConfigurationError):
    """$$M^{-1}$$ no es SPD ni siquiera tras la proyección de Higham."""

    pass


class PhaseHandoffError(SymplecticIntegrationError):
    """Un certificado de fase no satisface las precondiciones de la siguiente."""

    pass


class YoshidaCompositionError(IntegrationStepError):
    """Los pesos de Yoshida o el paso medio implícito son inadmisibles."""

    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. CLASIFICADORES DEL TOPOS OPERATIVO
# ══════════════════════════════════════════════════════════════════════════════
class IntegrationVerdict(IntEnum):
    r"""
    Cadena de Heyting $$\Omega_3$$ de calidad de integración:

        $$\bot=\mathtt{COHERENT}
          \le\mathtt{DEGRADED}
          \le\mathtt{VETOED}=\top.$$
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: IntegrationVerdict) -> IntegrationVerdict:
        if not isinstance(other, IntegrationVerdict):
            raise TypeError("join exige IntegrationVerdict.")
        return IntegrationVerdict(max(self.value, other.value))


class IntegrationAction(Enum):
    """Acción de mitigación asociada al veredicto."""

    NONE = "none"
    REDUCE_TIMESTEP = "reduce_timestep"
    HALT_INTEGRATION = "halt_integration"


class JacobianMethod(str, Enum):
    """Síntesis del Jacobiano de transición de fase."""

    NUMERICAL = "numerical"
    CSD = "csd"
    ANALYTICAL = "analytical"


# ══════════════════════════════════════════════════════════════════════════════
# §D. DTOs INMUTABLES DEL ESPACIO DE FASE
# ══════════════════════════════════════════════════════════════════════════════
def _freeze_array(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copia defensiva y sello de solo-lectura."""
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class MechanicalBundleData:
    """
    Artefacto certificado de la Fase 1.

    Es, por contrato, el objeto inicial de la Fase 2
    (véase ``handoff_phase1_to_phase2``).
    """

    coordinates: NDArray[np.float64]
    momenta: NDArray[np.float64]
    mass_matrix_inv: NDArray[np.float64]
    damping_matrix_r: NDArray[np.float64] | None
    external_forcing: NDArray[np.float64] | None
    rayleigh_factor: NDArray[np.float64]
    timestep: float
    mass_condition_number: float
    mass_spectral_minimum: float
    damping_spectral_minimum: float
    is_conservative: bool
    is_implicit_rayleigh: bool


@dataclass(frozen=True, slots=True)
class PhaseTransitionData:
    """
    Artefacto certificado de la Fase 2.

    Es, por contrato, el objeto inicial de la Fase 3
    (véase ``handoff_phase2_to_phase3``).
    """

    coordinates_next: NDArray[np.float64]
    momenta_next: NDArray[np.float64]
    momenta_half: NDArray[np.float64]
    jacobian_matrix: NDArray[np.float64] | None
    jacobian_method: str
    symplectic_residual: float
    relative_symplectic_residual: float
    liouville_residual: float
    is_symplectically_invariant: bool
    invariance_tolerance: float


@dataclass(frozen=True, slots=True)
class SymplecticState:
    r"""Punto inmutable en $$T^*\mathcal{Q}\simeq\mathbb{R}^{n}\times\mathbb{R}^{n}$$. """

    coordinates: NDArray[np.float64]
    momenta: NDArray[np.float64]
    hamiltonian: float


@dataclass(frozen=True, slots=True)
class SymplecticIntegratorReport:
    """Contrato inmutable de un paso de integración (veredicto terminal)."""

    state_next: SymplecticState
    symplectic_residual: float
    is_symplectically_invariant: bool
    dissipation_rate: float
    is_lyapunov_passive: bool
    jacobian_matrix: NDArray[np.float64] | None = None
    liouville_residual: float = 0.0
    hamiltonian_drift: float = 0.0
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    potential_is_surrogate: bool = False
    integration_verdict: IntegrationVerdict = IntegrationVerdict.COHERENT
    recommended_action: IntegrationAction = IntegrationAction.NONE
    symplectic_verdict: IntegrationVerdict = IntegrationVerdict.COHERENT
    passivity_verdict: IntegrationVerdict = IntegrationVerdict.COHERENT
    energy_verdict: IntegrationVerdict = IntegrationVerdict.COHERENT
    jacobian_method: str = JacobianMethod.NUMERICAL.value
    timestep: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → ESPECTROSCOPÍA PORT-HAMILTONIANA (Configure)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_PortHamiltonianSpectrometer:
    r"""
    Certifica el bundle mecánico $$(q,p,M^{-1},R,u,\Delta t)$$ y construye
    la 2-forma canónica $$\Omega$$ junto con el factor de Rayleigh

        $$A=I+\tfrac{\Delta t}{2}RM^{-1}.$$

    El morfismo terminal ``handoff_phase1_to_phase2`` eleva el certificado
    a precondición formal de la Fase 2.
    """

    def __init__(self, dimension: int, tolerance: float = _DEFAULT_TOL) -> None:
        self._n: int = self._validate_dimension(dimension)
        self._tol: float = self._validate_tolerance(tolerance)
        omega = self._build_canonical_omega(self._n)
        omega.setflags(write=False)
        self._omega: NDArray[np.float64] = omega

    # ──────────────────────────────────────────────────────────────────────────
    # 1.1  Validadores elementales
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_dimension(self, dimension: object) -> int:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise SystemConfigurationError(
                "La dimensión debe ser un entero no booleano."
            )
        if dimension <= 0:
            raise SystemConfigurationError("La dimensión debe ser estrictamente positiva.")
        return dimension

    def _validate_tolerance(self, tolerance: object) -> float:
        try:
            value = float(tolerance)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise SystemConfigurationError(
                "La tolerancia debe ser un número real."
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise SystemConfigurationError(
                "La tolerancia debe ser finita y estrictamente positiva."
            )
        return value

    def _as_finite_float(self, value: object, name: str) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise SystemConfigurationError(
                f"{name} no debe ser booleano; se requiere un número real."
            )
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise SystemConfigurationError(
                f"{name} no puede convertirse a un número real."
            ) from exc
        if not math.isfinite(result):
            raise SystemConfigurationError(f"{name} no es finito.")
        return result

    def _assert_finite(self, array: NDArray[np.floating], name: str) -> None:
        if not np.all(np.isfinite(array)):
            raise SystemConfigurationError(f"{name} contiene valores no finitos.")

    def _validate_vector(
        self,
        vector: object,
        name: str,
        expected_size: int,
    ) -> NDArray[np.float64]:
        try:
            array = np.array(vector, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise SystemConfigurationError(
                f"{name} no pudo convertirse a un vector float64."
            ) from exc
        if array.ndim != 1:
            raise SystemConfigurationError(f"{name} debe ser un vector 1-D.")
        if array.size != expected_size:
            raise SystemConfigurationError(
                f"{name} debe tener tamaño {expected_size}, recibido {array.size}."
            )
        self._assert_finite(array, name)
        return array

    def _validate_square_matrix(
        self,
        matrix: object,
        name: str,
        expected_size: int,
    ) -> NDArray[np.float64]:
        try:
            array = np.array(matrix, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise SystemConfigurationError(
                f"{name} no pudo convertirse a una matriz float64."
            ) from exc
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise SystemConfigurationError(f"{name} debe ser una matriz cuadrada.")
        if array.shape != (expected_size, expected_size):
            raise SystemConfigurationError(
                f"{name} debe tener forma ({expected_size}, {expected_size}), "
                f"recibida {array.shape}."
            )
        self._assert_finite(array, name)
        return array

    def _frobenius_norm(self, matrix: NDArray[np.float64], name: str) -> float:
        self._assert_finite(matrix, name)
        value = float(np.linalg.norm(matrix, ord="fro"))
        if not math.isfinite(value):
            raise SystemConfigurationError(
                f"La norma de Frobenius de {name} no es finita."
            )
        return value

    def _symmetrize(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        return 0.5 * (matrix + matrix.T)

    def _symmetric_residual(self, matrix: NDArray[np.float64]) -> float:
        return self._frobenius_norm(matrix - matrix.T, "residuo simétrico")

    def _matrix_residual_tolerance(
        self,
        matrix: NDArray[np.float64],
        base_tol: float | None = None,
    ) -> float:
        scale = self._frobenius_norm(matrix, "matriz para tolerancia")
        floor = self._tol if base_tol is None else base_tol
        return max(floor, floor * max(1.0, scale))

    # ──────────────────────────────────────────────────────────────────────────
    # 1.2  Neumaier y formas cuadráticas
    # ──────────────────────────────────────────────────────────────────────────
    def _neumaier_sum(self, terms: NDArray[np.float64], name: str) -> float:
        """Suma compensada de Neumaier (Kahan–Babuška–Neumaier)."""
        self._assert_finite(terms, name)
        total = 0.0
        compensation = 0.0
        for raw in terms.flat:
            term = float(raw)
            trial = total + term
            if abs(total) >= abs(term):
                compensation += (total - trial) + term
            else:
                compensation += (term - trial) + total
            if not math.isfinite(trial) or not math.isfinite(compensation):
                raise IntegrationStepError(
                    f"La sumación de Neumaier de {name} divergió."
                )
            total = trial
        result = total + compensation
        if not math.isfinite(result):
            raise IntegrationStepError(f"La suma compensada de {name} no es finita.")
        return result

    def _neumaier_quadratic_form(
        self,
        vector: NDArray[np.float64],
        matrix: NDArray[np.float64],
        name: str,
    ) -> float:
        r"""Forma cuadrática compensada $$x^\top A x$$."""
        vector = self._validate_vector(vector, f"{name}.x", vector.size)
        if matrix.shape != (vector.size, vector.size):
            raise IntegrationStepError(
                f"Dimensión incompatible en la forma cuadrática {name}."
            )
        terms = np.empty(vector.size, dtype=np.float64)
        aux = matrix @ vector
        self._assert_finite(aux, f"{name}.Ax")
        for i in range(vector.size):
            terms[i] = float(vector[i]) * float(aux[i])
        return self._neumaier_sum(terms, name)

    # ──────────────────────────────────────────────────────────────────────────
    # 1.3  Espectro SPD / PSD y cota de Wilkinson
    # ──────────────────────────────────────────────────────────────────────────
    def _spectral_bounds(
        self,
        matrix: NDArray[np.float64],
        name: str,
    ) -> tuple[NDArray[np.float64], float, float]:
        symmetric = self._symmetrize(matrix)
        try:
            eigenvalues = np.linalg.eigvalsh(symmetric)
        except np.linalg.LinAlgError as exc:
            raise SystemConfigurationError(
                f"La descomposición espectral de {name} falló."
            ) from exc
        self._assert_finite(eigenvalues, f"espectro de {name}")
        return eigenvalues, float(eigenvalues[0]), float(eigenvalues[-1])

    def _assert_symmetric(
        self,
        matrix: NDArray[np.float64],
        name: str,
    ) -> NDArray[np.float64]:
        residual = self._symmetric_residual(matrix)
        if residual > self._matrix_residual_tolerance(matrix):
            raise SystemConfigurationError(
                f"{name} no es simétrica. Residuo = {residual:.4e}."
            )
        return self._symmetrize(matrix)

    def _assert_spd(
        self,
        matrix: NDArray[np.float64],
        name: str,
    ) -> tuple[NDArray[np.float64], float, float, float]:
        r"""
        Exige $$A=A^\top\succ 0$$ por encima del umbral de Wilkinson

            $$\lambda_{\min}
              > \max\bigl(\tau,\, n\,\varepsilon\,\lambda_{\max}\bigr).$$
        """
        symmetric = self._assert_symmetric(matrix, name)
        _evals, lambda_min, lambda_max = self._spectral_bounds(symmetric, name)
        if lambda_max <= 0.0:
            raise MassMatrixSPDError(
                f"{name} no es definida positiva: λ_max = {lambda_max:.4e}."
            )
        floor = max(self._tol, float(symmetric.shape[0]) * _MACHINE_EPS * lambda_max)
        if lambda_min <= floor:
            raise MassMatrixSPDError(
                f"{name} no es definida positiva. "
                f"λ_min = {lambda_min:.4e}, umbral = {floor:.4e}."
            )
        condition = lambda_max / lambda_min
        if not math.isfinite(condition) or condition > _CONDITION_NUMBER_MAX:
            raise MassMatrixSPDError(
                f"{name} está mal condicionada: κ = {condition:.4e}."
            )
        return symmetric, condition, lambda_min, lambda_max

    def _assert_psd(
        self,
        matrix: NDArray[np.float64],
        name: str,
    ) -> tuple[NDArray[np.float64], float]:
        r"""Exige $$A=A^\top\succeq 0$$ salvo un suelo de redondeo."""
        symmetric = self._assert_symmetric(matrix, name)
        _evals, lambda_min, lambda_max = self._spectral_bounds(symmetric, name)
        floor = -max(self._tol, float(symmetric.shape[0]) * _MACHINE_EPS * max(1.0, abs(lambda_max)))
        if lambda_min < floor:
            raise SystemConfigurationError(
                f"{name} no es semidefinida positiva. "
                f"λ_min = {lambda_min:.4e}."
            )
        return symmetric, lambda_min

    # ──────────────────────────────────────────────────────────────────────────
    # 1.4  Bundle mecánico: masa, Rayleigh, forzamiento, paso
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_mass_matrix_inv(
        self,
        mass_matrix_inv: object,
    ) -> tuple[NDArray[np.float64], float, float]:
        matrix = self._validate_square_matrix(
            mass_matrix_inv, "mass_matrix_inv", self._n
        )
        sanitized, condition, lambda_min, _lambda_max = self._assert_spd(
            matrix, "mass_matrix_inv"
        )
        return sanitized, condition, lambda_min

    def _validate_damping_matrix_r(
        self,
        damping_matrix_r: object | None,
    ) -> tuple[NDArray[np.float64] | None, float]:
        if damping_matrix_r is None:
            return None, 0.0
        matrix = self._validate_square_matrix(
            damping_matrix_r, "damping_matrix_r", self._n
        )
        sanitized, lambda_min = self._assert_psd(matrix, "damping_matrix_r")
        return sanitized, lambda_min

    def _validate_external_forcing(
        self,
        external_forcing: object | None,
    ) -> NDArray[np.float64] | None:
        if external_forcing is None:
            return None
        return self._validate_vector(external_forcing, "external_forcing", self._n)

    def _validate_timestep(
        self,
        timestep: object,
        *,
        allow_negative: bool,
    ) -> float:
        value = self._as_finite_float(timestep, "dt")
        if value == 0.0:
            raise IntegrationStepError("El paso temporal dt no puede ser nulo.")
        if not allow_negative and value < 0.0:
            raise IntegrationStepError(
                f"El paso temporal dt debe ser positivo, recibido {value}."
            )
        if abs(value) < _MIN_DT:
            raise IntegrationStepError(
                f"|dt| = {value:.4e} es menor que el mínimo admisible {_MIN_DT:.4e}."
            )
        if abs(value) > _MAX_DT:
            raise IntegrationStepError(
                f"|dt| = {value:.4e} excede el máximo admisible {_MAX_DT:.4e}."
            )
        return value

    def _build_canonical_omega(self, dimension: int) -> NDArray[np.float64]:
        r"""
        2-forma canónica de Liouville en coordenadas $$z=(q,p)$$:

            $$\Omega=\begin{pmatrix}0&I\\-I&0\end{pmatrix}.$$
        """
        identity = np.eye(dimension, dtype=np.float64)
        zeros = np.zeros((dimension, dimension), dtype=np.float64)
        return np.block([[zeros, identity], [-identity, zeros]])

    def _assemble_rayleigh_factor(
        self,
        mass_matrix_inv: NDArray[np.float64],
        damping_matrix_r: NDArray[np.float64] | None,
        timestep: float,
    ) -> tuple[NDArray[np.float64], bool]:
        r"""
        Factor implícito de Rayleigh

            $$A=I+\tfrac{\Delta t}{2}RM^{-1}.$$

        Si $$R=0$$ se devuelve $$I$$ y el flag de implicitud es falso.
        Se exige que $$A$$ sea invertible (crítico en el paso medio
        de Yoshida, donde $$\Delta t<0$$).
        """
        identity = np.eye(self._n, dtype=np.float64)
        if damping_matrix_r is None:
            return identity, False

        coupling = damping_matrix_r @ mass_matrix_inv
        self._assert_finite(coupling, "R M^{-1}")
        factor = identity + (0.5 * timestep) * coupling
        self._assert_finite(factor, "factor de Rayleigh A")

        try:
            determinant_sign, log_abs_det = np.linalg.slogdet(factor)
        except np.linalg.LinAlgError as exc:
            raise IntegrationStepError(
                "No fue posible auditar la invertibilidad de A."
            ) from exc
        if determinant_sign == 0.0 or not math.isfinite(log_abs_det):
            raise YoshidaCompositionError(
                "El factor de Rayleigh A = I + (dt/2) R M^{-1} es singular. "
                f"dt = {timestep:.4e}."
            )
        return factor, True

    def _apply_rayleigh_factor(
        self,
        factor: NDArray[np.float64],
        rhs: NDArray[np.float64],
        name: str,
    ) -> NDArray[np.float64]:
        """Resuelve $$A x = b$$.  Si $$A=I$$, devuelve $$b$$."""
        if np.array_equal(factor, np.eye(self._n, dtype=np.float64)):
            return rhs
        try:
            solved = np.linalg.solve(factor, rhs)
        except np.linalg.LinAlgError as exc:
            raise IntegrationStepError(
                f"El sistema de Rayleigh {name} es singular."
            ) from exc
        self._assert_finite(solved, name)
        return solved

    # ──────────────────────────────────────────────────────────────────────────
    # 1.5  Núcleo terminal de la Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def execute_phase1(
        self,
        coordinates: NDArray[np.float64],
        momenta: NDArray[np.float64],
        mass_matrix_inv: NDArray[np.float64],
        timestep: float,
        damping_matrix_r: NDArray[np.float64] | None = None,
        external_forcing: NDArray[np.float64] | None = None,
        *,
        allow_negative_dt: bool = False,
    ) -> MechanicalBundleData:
        """
        Método terminal de la Fase 1.

        Su salida constituye el dominio formal de
        ``handoff_phase1_to_phase2``.
        """
        q_coord = self._validate_vector(coordinates, "q", self._n)
        p_mom = self._validate_vector(momenta, "p", self._n)
        mass_inv, condition, lambda_min = self._validate_mass_matrix_inv(
            mass_matrix_inv
        )
        damping, damping_min = self._validate_damping_matrix_r(damping_matrix_r)
        forcing = self._validate_external_forcing(external_forcing)
        dt_step = self._validate_timestep(timestep, allow_negative=allow_negative_dt)
        factor, implicit = self._assemble_rayleigh_factor(mass_inv, damping, dt_step)
        conservative = damping is None or self._frobenius_norm(damping, "R") <= (
            float(self._n) * _MACHINE_EPS
        )

        bundle = MechanicalBundleData(
            coordinates=_freeze_array(q_coord),
            momenta=_freeze_array(p_mom),
            mass_matrix_inv=_freeze_array(mass_inv),
            damping_matrix_r=None if damping is None else _freeze_array(damping),
            external_forcing=None if forcing is None else _freeze_array(forcing),
            rayleigh_factor=_freeze_array(factor),
            timestep=dt_step,
            mass_condition_number=condition,
            mass_spectral_minimum=lambda_min,
            damping_spectral_minimum=damping_min,
            is_conservative=conservative,
            is_implicit_rayleigh=implicit,
        )
        logger.debug(
            "Fase 1 bundle: n=%d, dt=%.4e, κ(M^{-1})=%.4e, conservativo=%s, implícito=%s.",
            self._n,
            bundle.timestep,
            bundle.mass_condition_number,
            bundle.is_conservative,
            bundle.is_implicit_rayleigh,
        )
        return bundle

    def handoff_phase1_to_phase2(
        self,
        bundle: MechanicalBundleData,
    ) -> MechanicalBundleData:
        r"""
        Morfismo de transición

            $$\Phi_{12}:
              \mathrm{MechanicalBundleData}
              \longrightarrow
              \mathrm{MechanicalBundleData}.$$

        Poscondición de la Fase 1  ≡  precondición de la Fase 2:

            $$q,p\in\mathbb{R}^n$$ finitos, $$M^{-1}\succ 0$$,
            $$A$$ invertible, $$\Delta t\neq 0$$.

        Este método es la definición formal final de la Fase 1 y, a la
        vez, el dominio sobre el que la Fase 2 construye el mapa de
        Verlet.  ``Phase2_VariationalFlowSynthesizer`` comienza
        invocándolo.
        """
        if not isinstance(bundle, MechanicalBundleData):
            raise PhaseHandoffError(
                "handoff_phase1_to_phase2 exige MechanicalBundleData."
            )
        if bundle.coordinates.size != self._n or bundle.momenta.size != self._n:
            raise PhaseHandoffError("El bundle no vive en la dimensión del integrador.")
        if bundle.timestep == 0.0 or not math.isfinite(bundle.timestep):
            raise PhaseHandoffError("dt del bundle es nulo o no finito.")
        if bundle.mass_condition_number > _CONDITION_NUMBER_MAX:
            raise PhaseHandoffError("κ(M^{-1}) del bundle excede la cota admisible.")
        if bundle.rayleigh_factor.shape != (self._n, self._n):
            raise PhaseHandoffError("El factor de Rayleigh tiene forma inadmisible.")
        return bundle


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → SÍNTESIS DEL FLUJO VARIACIONAL (Advance)
#          continuación formal de handoff_phase1_to_phase2
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_VariationalFlowSynthesizer(Phase1_PortHamiltonianSpectrometer):
    r"""
    Recibe el bundle certificado por ``handoff_phase1_to_phase2`` y
    construye el mapa de Störmer–Verlet

        $$p_{1/2}=A^{-1}\bigl(p+\tfrac{\Delta t}{2}(F(q)+u)\bigr),$$
        $$q_{+}=q+\Delta t\,M^{-1}p_{1/2},$$
        $$p_{+}=A^{-1}\bigl(p_{1/2}+\tfrac{\Delta t}{2}(F(q_{+})+u)\bigr),$$

    junto con su Jacobiano de transición y los residuos de
    $$\mathrm{Sp}(2n)$$ y de Liouville.

    El morfismo terminal ``handoff_phase2_to_phase3`` eleva el
    certificado a precondición formal de la Fase 3.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 2.1  Continuación inmediata del handoff de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_bundle(
        self,
        bundle: MechanicalBundleData,
    ) -> MechanicalBundleData:
        """
        Primer método de la Fase 2.

        Es la continuación literal de ``handoff_phase1_to_phase2``.
        """
        return self.handoff_phase1_to_phase2(bundle)

    # ──────────────────────────────────────────────────────────────────────────
    # 2.2  Campo de fuerza y mapa de Verlet
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_force_callable(
        self,
        force_field: object,
        name: str,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        if not callable(force_field):
            raise IntegrationStepError(f"{name} debe ser un callable.")
        return force_field  # type: ignore[return-value]

    def _evaluate_force(
        self,
        force_field: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        coordinates: NDArray[np.float64],
        name: str,
    ) -> NDArray[np.float64]:
        try:
            raw = force_field(coordinates)
        except Exception as exc:  # noqa: BLE001 — se reenvuelve tipado
            raise IntegrationStepError(
                f"{name} lanzó una excepción al evaluarse."
            ) from exc
        return self._validate_vector(raw, name, self._n)

    def _zero_force(self, bundle: MechanicalBundleData) -> NDArray[np.float64]:
        if bundle.external_forcing is None:
            return np.zeros(self._n, dtype=np.float64)
        return np.array(bundle.external_forcing, dtype=np.float64, copy=True)

    def _advance_verlet_map(
        self,
        bundle: MechanicalBundleData,
        force_field: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Un paso kick–drift–kick con Rayleigh implícita.

        Retorna ``(q_{+}, p_{+}, p_{1/2})``.
        """
        dt_step = bundle.timestep
        forcing = self._zero_force(bundle)
        mass_inv = np.array(bundle.mass_matrix_inv, dtype=np.float64, copy=True)
        factor = np.array(bundle.rayleigh_factor, dtype=np.float64, copy=True)
        q_coord = np.array(bundle.coordinates, dtype=np.float64, copy=True)
        p_mom = np.array(bundle.momenta, dtype=np.float64, copy=True)

        force_q = self._evaluate_force(force_field, q_coord, "force_gradient_q(q)")
        p_half = self._apply_rayleigh_factor(
            factor,
            p_mom + 0.5 * dt_step * (force_q + forcing),
            "p_1/2",
        )
        q_next = q_coord + dt_step * (mass_inv @ p_half)
        self._assert_finite(q_next, "q_{+}")
        force_next = self._evaluate_force(
            force_field, q_next, "force_gradient_q(q_{+})"
        )
        p_next = self._apply_rayleigh_factor(
            factor,
            p_half + 0.5 * dt_step * (force_next + forcing),
            "p_{+}",
        )
        return q_next, p_next, p_half

    def _advance_from_state(
        self,
        q_coord: NDArray[np.float64],
        p_mom: NDArray[np.float64],
        bundle: MechanicalBundleData,
        force_field: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Avanza un estado auxiliar (para Jacobianos por perturbación)."""
        shadow = MechanicalBundleData(
            coordinates=_freeze_array(q_coord),
            momenta=_freeze_array(p_mom),
            mass_matrix_inv=bundle.mass_matrix_inv,
            damping_matrix_r=bundle.damping_matrix_r,
            external_forcing=bundle.external_forcing,
            rayleigh_factor=bundle.rayleigh_factor,
            timestep=bundle.timestep,
            mass_condition_number=bundle.mass_condition_number,
            mass_spectral_minimum=bundle.mass_spectral_minimum,
            damping_spectral_minimum=bundle.damping_spectral_minimum,
            is_conservative=bundle.is_conservative,
            is_implicit_rayleigh=bundle.is_implicit_rayleigh,
        )
        q_next, p_next, _ = self._advance_verlet_map(shadow, force_field)
        return np.concatenate([q_next, p_next])

    # ──────────────────────────────────────────────────────────────────────────
    # 2.3  Jacobianos de transición
    # ──────────────────────────────────────────────────────────────────────────
    def _resolve_jacobian_method(self, method: object) -> JacobianMethod:
        if isinstance(method, JacobianMethod):
            return method
        if isinstance(method, str):
            try:
                return JacobianMethod(method.lower())
            except ValueError as exc:
                raise IntegrationStepError(
                    "jacobian_method debe ser 'numerical', 'csd' o 'analytical'."
                ) from exc
        raise IntegrationStepError(
            "jacobian_method debe ser una cadena o un JacobianMethod."
        )

    def _finite_difference_step(
        self,
        q_coord: NDArray[np.float64],
        p_mom: NDArray[np.float64],
        override: float | None,
    ) -> float:
        if override is not None:
            step = self._as_finite_float(override, "h")
            if step <= 0.0:
                raise IntegrationStepError("El paso de diferencias h debe ser positivo.")
            return step
        scale = max(
            1.0,
            float(np.max(np.abs(q_coord))) if q_coord.size else 1.0,
            float(np.max(np.abs(p_mom))) if p_mom.size else 1.0,
        )
        return (_MACHINE_EPS ** (1.0 / 3.0)) * scale

    def _compute_numerical_jacobian(
        self,
        bundle: MechanicalBundleData,
        force_field: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        step: float | None = None,
    ) -> NDArray[np.float64]:
        r"""
        Jacobiano por diferencias adelantadas con paso

            $$h=\max(1,\|q\|_\infty,\|p\|_\infty)\,\varepsilon^{1/3}.$$
        """
        q_coord = np.array(bundle.coordinates, dtype=np.float64, copy=True)
        p_mom = np.array(bundle.momenta, dtype=np.float64, copy=True)
        increment = self._finite_difference_step(q_coord, p_mom, step)
        dimension = 2 * self._n
        jacobian = np.zeros((dimension, dimension), dtype=np.float64)
        base = self._advance_from_state(q_coord, p_mom, bundle, force_field)

        for index in range(self._n):
            q_pert = q_coord.copy()
            q_pert[index] += increment
            jacobian[:, index] = (
                self._advance_from_state(q_pert, p_mom, bundle, force_field) - base
            ) / increment

            p_pert = p_mom.copy()
            p_pert[index] += increment
            jacobian[:, self._n + index] = (
                self._advance_from_state(q_coord, p_pert, bundle, force_field) - base
            ) / increment

        self._assert_finite(jacobian, "Jacobiano numérico")
        return jacobian

    def _compute_csd_jacobian(
        self,
        bundle: MechanicalBundleData,
        force_field_complex: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
        step: float = _CSD_STEP_DEFAULT,
    ) -> NDArray[np.float64]:
        r"""
        Diferenciación por paso complejo (Squire–Trapp):

            $$\partial_x f=\Im f(x+ih)/h+O(h^2).$$

        No hay cancelación sustractiva; $$h\sim 10^{-20}$$ es admisible.
        """
        if not callable(force_field_complex):
            raise IntegrationStepError(
                "force_gradient_complex_q debe ser un callable complejo."
            )
        increment = self._as_finite_float(step, "h_csd")
        if increment < _CSD_STEP_MIN or increment > _CSD_STEP_MAX:
            raise IntegrationStepError(
                f"h_csd = {increment:.4e} está fuera de "
                f"[{_CSD_STEP_MIN:.0e}, {_CSD_STEP_MAX:.0e}]."
            )

        mass_inv = bundle.mass_matrix_inv.astype(np.complex128)
        factor = bundle.rayleigh_factor.astype(np.complex128)
        forcing = (
            np.zeros(self._n, dtype=np.complex128)
            if bundle.external_forcing is None
            else bundle.external_forcing.astype(np.complex128)
        )
        dt_step = complex(bundle.timestep)

        def evaluate_complex(position: NDArray[np.complex128], name: str) -> NDArray[np.complex128]:
            try:
                raw = force_field_complex(position)
            except Exception as exc:  # noqa: BLE001
                raise IntegrationStepError(
                    f"{name} lanzó una excepción en aritmética compleja."
                ) from exc
            try:
                force = np.asarray(raw, dtype=np.complex128)
            except (TypeError, ValueError) as exc:
                raise IntegrationStepError(
                    f"{name} no devolvió un vector complejo."
                ) from exc
            if force.shape != (self._n,):
                raise IntegrationStepError(
                    f"{name} debe devolver un vector de tamaño {self._n}."
                )
            if not np.all(np.isfinite(force)):
                raise IntegrationStepError(f"{name} produjo valores no finitos.")
            return force

        def step_complex(
            q_val: NDArray[np.complex128],
            p_val: NDArray[np.complex128],
        ) -> NDArray[np.complex128]:
            force_q = evaluate_complex(q_val, "force_gradient_complex_q(q)")
            rhs_half = p_val + 0.5 * dt_step * (force_q + forcing)
            p_half = np.linalg.solve(factor, rhs_half)
            q_next = q_val + dt_step * (mass_inv @ p_half)
            force_next = evaluate_complex(q_next, "force_gradient_complex_q(q_{+})")
            rhs_next = p_half + 0.5 * dt_step * (force_next + forcing)
            p_next = np.linalg.solve(factor, rhs_next)
            return np.concatenate([q_next, p_next])

        q_c = bundle.coordinates.astype(np.complex128)
        p_c = bundle.momenta.astype(np.complex128)
        dimension = 2 * self._n
        jacobian = np.zeros((dimension, dimension), dtype=np.float64)

        for index in range(self._n):
            q_pert = q_c.copy()
            q_pert[index] += complex(0.0, increment)
            jacobian[:, index] = np.imag(step_complex(q_pert, p_c)) / increment

            p_pert = p_c.copy()
            p_pert[index] += complex(0.0, increment)
            jacobian[:, self._n + index] = np.imag(step_complex(q_c, p_pert)) / increment

        self._assert_finite(jacobian, "Jacobiano CSD")
        return jacobian

    def _compute_analytic_jacobian(
        self,
        bundle: MechanicalBundleData,
        stiffness_matrix_k: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Jacobiano exacto del mapa de Verlet linealizado
        ($$K=\partial F/\partial q=-\nabla^2 V$$ constante).

        Con $$A=I+\tfrac{\Delta t}{2}RM^{-1}$$:

            $$\partial p_{1/2}/\partial q=A^{-1}(\Delta t/2)K,\quad
              \partial p_{1/2}/\partial p=A^{-1}.$$
        """
        stiffness = self._validate_square_matrix(
            stiffness_matrix_k, "stiffness_matrix_k", self._n
        )
        residual = self._symmetric_residual(stiffness)
        if residual > self._matrix_residual_tolerance(stiffness):
            raise IntegrationStepError(
                "stiffness_matrix_k debe ser simétrica (Hessiano de V). "
                f"Residuo = {residual:.4e}."
            )
        stiffness = self._symmetrize(stiffness)

        identity = np.eye(self._n, dtype=np.float64)
        factor = np.array(bundle.rayleigh_factor, dtype=np.float64, copy=True)
        mass_inv = np.array(bundle.mass_matrix_inv, dtype=np.float64, copy=True)
        dt_step = bundle.timestep
        factor_inv = self._apply_rayleigh_factor(factor, identity, "A^{-1}")

        dp_half_dq = factor_inv @ ((0.5 * dt_step) * stiffness)
        dp_half_dp = factor_inv
        dq_next_dq = identity + dt_step * (mass_inv @ dp_half_dq)
        dq_next_dp = dt_step * (mass_inv @ dp_half_dp)
        dp_next_dq = factor_inv @ (
            dp_half_dq + (0.5 * dt_step) * (stiffness @ dq_next_dq)
        )
        dp_next_dp = factor_inv @ (
            dp_half_dp + (0.5 * dt_step) * (stiffness @ dq_next_dp)
        )
        jacobian = np.block([[dq_next_dq, dq_next_dp], [dp_next_dq, dp_next_dp]])
        self._assert_finite(jacobian, "Jacobiano analítico")
        return jacobian

    def _synthesize_jacobian(
        self,
        bundle: MechanicalBundleData,
        force_field: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        method: JacobianMethod,
        stiffness_matrix_k: NDArray[np.float64] | None,
        force_field_complex: Callable[[NDArray[np.complex128]], NDArray[np.complex128]] | None,
    ) -> tuple[NDArray[np.float64], float]:
        if method is JacobianMethod.CSD:
            if force_field_complex is None:
                raise IntegrationStepError(
                    "Para jacobian_method='csd' debe suministrar "
                    "force_gradient_complex_q."
                )
            return (
                self._compute_csd_jacobian(bundle, force_field_complex),
                _JACOBIAN_INVARIANCE_TOL[method.value],
            )
        if method is JacobianMethod.ANALYTICAL:
            if stiffness_matrix_k is None:
                raise IntegrationStepError(
                    "Para jacobian_method='analytical' debe suministrar "
                    "stiffness_matrix_k."
                )
            return (
                self._compute_analytic_jacobian(bundle, stiffness_matrix_k),
                _JACOBIAN_INVARIANCE_TOL[method.value],
            )
        return (
            self._compute_numerical_jacobian(bundle, force_field),
            _JACOBIAN_INVARIANCE_TOL[JacobianMethod.NUMERICAL.value],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 2.4  Residuos de Sp(2n) y de Liouville
    # ──────────────────────────────────────────────────────────────────────────
    def _verify_symplectic_form(
        self,
        jacobian: NDArray[np.float64],
    ) -> tuple[float, float]:
        r"""
        Residuo absoluto y relativo de la identidad simpléctica:

            $$r=\|J^\top\Omega J-\Omega\|_F,\qquad
              r_{\mathrm{rel}}=r/\max(1,\|\Omega\|_F).$$
        """
        pulled = jacobian.T @ self._omega @ jacobian
        self._assert_finite(pulled, "Jᵀ Ω J")
        residual = self._frobenius_norm(pulled - self._omega, "Jᵀ Ω J − Ω")
        omega_scale = self._frobenius_norm(self._omega, "Ω")
        relative = residual / max(1.0, omega_scale)
        return residual, relative

    def _verify_liouville_volume(self, jacobian: NDArray[np.float64]) -> float:
        r"""
        Residuo de volumen $$\lvert\det J-1\rvert$$ vía ``slogdet``:

            $$r=\lvert\mathrm{sign}-1\rvert+\lvert\log\lvert\det J\rvert\rvert.$$
        """
        try:
            sign, log_abs_det = np.linalg.slogdet(jacobian)
        except np.linalg.LinAlgError as exc:
            raise IntegrationStepError(
                "No fue posible calcular det(J) para el test de Liouville."
            ) from exc
        if not math.isfinite(log_abs_det):
            raise IntegrationStepError("log|det J| no es finito.")
        return abs(float(sign) - 1.0) + abs(float(log_abs_det))

    # ──────────────────────────────────────────────────────────────────────────
    # 2.5  Núcleo terminal de la Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def execute_phase2(
        self,
        bundle: MechanicalBundleData,
        force_gradient_q: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        jacobian_method: str | JacobianMethod = JacobianMethod.NUMERICAL,
        stiffness_matrix_k: NDArray[np.float64] | None = None,
        force_gradient_complex_q: Callable[[NDArray[np.complex128]], NDArray[np.complex128]] | None = None,
        *,
        compute_jacobian: bool = True,
    ) -> PhaseTransitionData:
        """
        Método terminal de la Fase 2.

        Recibe la salida formal de ``execute_phase1`` (vía
        ``handoff_phase1_to_phase2``) y produce la entrada canónica de
        ``handoff_phase2_to_phase3``.
        """
        certified = self._receive_certified_bundle(bundle)
        force_field = self._validate_force_callable(
            force_gradient_q, "force_gradient_q"
        )
        q_next, p_next, p_half = self._advance_verlet_map(certified, force_field)

        method = self._resolve_jacobian_method(jacobian_method)
        jacobian: NDArray[np.float64] | None = None
        residual = 0.0
        relative = 0.0
        liouville = 0.0
        tolerance = _JACOBIAN_INVARIANCE_TOL[method.value]
        invariant = True

        if compute_jacobian:
            jacobian, tolerance = self._synthesize_jacobian(
                certified,
                force_field,
                method,
                stiffness_matrix_k,
                force_gradient_complex_q,
            )
            residual, relative = self._verify_symplectic_form(jacobian)
            liouville = self._verify_liouville_volume(jacobian)
            if certified.is_conservative:
                invariant = residual <= tolerance
            else:
                # El flujo disipativo no está obligado a preservar ω.
                invariant = True

        transition = PhaseTransitionData(
            coordinates_next=_freeze_array(q_next),
            momenta_next=_freeze_array(p_next),
            momenta_half=_freeze_array(p_half),
            jacobian_matrix=None if jacobian is None else _freeze_array(jacobian),
            jacobian_method=method.value,
            symplectic_residual=residual,
            relative_symplectic_residual=relative,
            liouville_residual=liouville,
            is_symplectically_invariant=invariant,
            invariance_tolerance=tolerance,
        )
        logger.debug(
            "Fase 2 transición: método=%s, r_ω=%.4e, r_Liouville=%.4e, invariante=%s.",
            transition.jacobian_method,
            transition.symplectic_residual,
            transition.liouville_residual,
            transition.is_symplectically_invariant,
        )
        return transition

    def handoff_phase2_to_phase3(
        self,
        transition: PhaseTransitionData,
    ) -> PhaseTransitionData:
        r"""
        Morfismo de transición

            $$\Phi_{23}:
              \mathrm{PhaseTransitionData}
              \longrightarrow
              \mathrm{PhaseTransitionData}.$$

        Poscondición de la Fase 2  ≡  precondición de la Fase 3:

            $$q_{+},p_{+}\in\mathbb{R}^n$$ finitos; si hay Jacobiano,
            éste es $$2n\times 2n$$ y finito.

        Este método es la definición formal final de la Fase 2 y, a la
        vez, el dominio sobre el que la Fase 3 evalúa $$H$$ y el
        veredicto de Lyapunov.  ``Phase3_YoshidaThermodynamicIntegrator``
        comienza invocándolo.
        """
        if not isinstance(transition, PhaseTransitionData):
            raise PhaseHandoffError(
                "handoff_phase2_to_phase3 exige PhaseTransitionData."
            )
        if (
            transition.coordinates_next.size != self._n
            or transition.momenta_next.size != self._n
        ):
            raise PhaseHandoffError(
                "La transición no vive en la dimensión del integrador."
            )
        if not np.all(np.isfinite(transition.coordinates_next)):
            raise PhaseHandoffError("q_{+} de la transición no es finito.")
        if not np.all(np.isfinite(transition.momenta_next)):
            raise PhaseHandoffError("p_{+} de la transición no es finito.")
        if transition.jacobian_matrix is not None:
            expected = (2 * self._n, 2 * self._n)
            if transition.jacobian_matrix.shape != expected:
                raise PhaseHandoffError(
                    f"El Jacobiano debe ser {expected}, "
                    f"recibido {transition.jacobian_matrix.shape}."
                )
        if not math.isfinite(transition.symplectic_residual):
            raise PhaseHandoffError("El residuo simpléctico no es finito.")
        return transition


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → INTEGRACIÓN DE YOSHIDA Y VEREDICTO DE LYAPUNOV (Decide)
#          continuación formal de handoff_phase2_to_phase3
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_YoshidaThermodynamicIntegrator(Phase2_VariationalFlowSynthesizer):
    r"""
    Recibe la transición certificada por ``handoff_phase2_to_phase3``,
    evalúa el Hamiltoniano con Neumaier, la pasividad de Rayleigh y el
    drift discreto, y —en orden 4— compone tres mapas de Verlet con
    los pesos de Yoshida–Forest–Ruth.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 3.1  Continuación inmediata del handoff de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_transition(
        self,
        transition: PhaseTransitionData,
    ) -> PhaseTransitionData:
        """
        Primer método de la Fase 3.

        Es la continuación literal de ``handoff_phase2_to_phase3``.
        """
        return self.handoff_phase2_to_phase3(transition)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.2  Hamiltoniano, Rayleigh y veredictos
    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_potential(
        self,
        potential_q: Callable[[NDArray[np.float64]], float] | None,
        coordinates: NDArray[np.float64],
        force_next: NDArray[np.float64] | None,
    ) -> tuple[float, bool]:
        if potential_q is None:
            if force_next is None:
                return 0.0, True
            # Surrogado cuadrático: V ≈ −½ ⟨q, F⟩, exacto ssi F = −K q.
            surrogate = -0.5 * self._neumaier_sum(
                coordinates * force_next, "surrogado de V"
            )
            return surrogate, True
        if not callable(potential_q):
            raise IntegrationStepError("potential_q debe ser un callable.")
        try:
            raw = potential_q(coordinates)
        except Exception as exc:  # noqa: BLE001
            raise IntegrationStepError("potential_q lanzó una excepción.") from exc
        value = self._as_finite_float(raw, "potential_q(q)")
        return value, False

    def _evaluate_hamiltonian(
        self,
        coordinates: NDArray[np.float64],
        momenta: NDArray[np.float64],
        mass_matrix_inv: NDArray[np.float64],
        potential_q: Callable[[NDArray[np.float64]], float] | None,
        force_next: NDArray[np.float64] | None,
    ) -> tuple[float, float, float, bool]:
        kinetic = 0.5 * self._neumaier_quadratic_form(
            momenta, mass_matrix_inv, "energía cinética"
        )
        if kinetic < 0.0:
            floor = max(self._tol, 1000.0 * _MACHINE_EPS * max(1.0, abs(kinetic)))
            if kinetic < -floor:
                raise IntegrationStepError(
                    f"Energía cinética negativa significativa: {kinetic:.4e}."
                )
            kinetic = 0.0
        potential, surrogate = self._evaluate_potential(
            potential_q, coordinates, force_next
        )
        hamiltonian = kinetic + potential
        if not math.isfinite(hamiltonian):
            raise IntegrationStepError("El Hamiltoniano evaluado no es finito.")
        return hamiltonian, kinetic, potential, surrogate

    def _evaluate_dissipation(
        self,
        momenta: NDArray[np.float64],
        mass_matrix_inv: NDArray[np.float64],
        damping_matrix_r: NDArray[np.float64] | None,
    ) -> float:
        r"""
        Tasa instantánea de Rayleigh (PHS):

            $$\dot{H}\big|_R=-v^\top R v,\qquad v=M^{-1}p.$$

        Es $$\le 0$$ ssi $$R\succeq 0$$.
        """
        if damping_matrix_r is None:
            return 0.0
        velocity = mass_matrix_inv @ momenta
        self._assert_finite(velocity, "v = M^{-1} p")
        return -self._neumaier_quadratic_form(
            velocity, damping_matrix_r, "vᵀ R v"
        )

    def _threshold_verdict(
        self,
        value: float,
        soft: float,
        hard: float,
    ) -> IntegrationVerdict:
        if not math.isfinite(value) or value > hard:
            return IntegrationVerdict.VETOED
        if value > soft:
            return IntegrationVerdict.DEGRADED
        return IntegrationVerdict.COHERENT

    def _classify_report(
        self,
        bundle: MechanicalBundleData,
        transition: PhaseTransitionData,
        dissipation_rate: float,
        hamiltonian_drift: float,
    ) -> tuple[IntegrationVerdict, IntegrationVerdict, IntegrationVerdict, IntegrationVerdict, IntegrationAction]:
        if bundle.is_conservative:
            symplectic_verdict = self._threshold_verdict(
                transition.symplectic_residual,
                max(transition.invariance_tolerance, _SYMPLECTIC_SOFT_TOL),
                max(10.0 * transition.invariance_tolerance, _SYMPLECTIC_HARD_TOL),
            )
            energy_verdict = self._threshold_verdict(
                abs(hamiltonian_drift),
                _ENERGY_SOFT_TOL,
                _ENERGY_HARD_TOL,
            )
        else:
            symplectic_verdict = IntegrationVerdict.COHERENT
            energy_verdict = IntegrationVerdict.COHERENT

        # diss_rate debe ser ≤ 0; el veredicto mira la parte positiva.
        passivity_verdict = self._threshold_verdict(
            max(0.0, dissipation_rate),
            _PASSIVITY_SOFT_TOL,
            _PASSIVITY_HARD_TOL,
        )
        final = symplectic_verdict.join(passivity_verdict).join(energy_verdict)
        if final is IntegrationVerdict.VETOED:
            action = IntegrationAction.HALT_INTEGRATION
        elif final is IntegrationVerdict.DEGRADED:
            action = IntegrationAction.REDUCE_TIMESTEP
        else:
            action = IntegrationAction.NONE
        return symplectic_verdict, passivity_verdict, energy_verdict, final, action

    def _freeze_state(
        self,
        coordinates: NDArray[np.float64],
        momenta: NDArray[np.float64],
        hamiltonian: float,
    ) -> SymplecticState:
        return SymplecticState(
            coordinates=_freeze_array(coordinates),
            momenta=_freeze_array(momenta),
            hamiltonian=float(hamiltonian),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 3.3  Núcleo terminal de la Fase 3
    # ──────────────────────────────────────────────────────────────────────────
    def execute_phase3(
        self,
        bundle: MechanicalBundleData,
        transition: PhaseTransitionData,
        potential_q: Callable[[NDArray[np.float64]], float] | None,
        force_gradient_q: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        hamiltonian_previous: float | None = None,
    ) -> SymplecticIntegratorReport:
        """
        Método terminal de la Fase 3.

        Recibe la salida formal de ``execute_phase2`` (vía
        ``handoff_phase2_to_phase3``) y devuelve el informe termodinámico.
        """
        certified_bundle = self._receive_certified_bundle(bundle)
        certified = self._receive_certified_transition(transition)

        force_next: NDArray[np.float64] | None = None
        if force_gradient_q is not None:
            force_field = self._validate_force_callable(
                force_gradient_q, "force_gradient_q"
            )
            force_next = self._evaluate_force(
                force_field,
                np.array(certified.coordinates_next, dtype=np.float64, copy=True),
                "force_gradient_q(q_{+})",
            )

        hamiltonian, kinetic, potential, surrogate = self._evaluate_hamiltonian(
            np.array(certified.coordinates_next, dtype=np.float64, copy=True),
            np.array(certified.momenta_next, dtype=np.float64, copy=True),
            np.array(certified_bundle.mass_matrix_inv, dtype=np.float64, copy=True),
            potential_q,
            force_next,
        )
        dissipation = self._evaluate_dissipation(
            np.array(certified.momenta_next, dtype=np.float64, copy=True),
            np.array(certified_bundle.mass_matrix_inv, dtype=np.float64, copy=True),
            None
            if certified_bundle.damping_matrix_r is None
            else np.array(certified_bundle.damping_matrix_r, dtype=np.float64, copy=True),
        )
        is_passive = dissipation <= _PASSIVITY_HARD_TOL

        if hamiltonian_previous is None or not math.isfinite(hamiltonian_previous):
            drift = 0.0
        else:
            drift = hamiltonian - float(hamiltonian_previous)

        (
            symplectic_verdict,
            passivity_verdict,
            energy_verdict,
            final_verdict,
            action,
        ) = self._classify_report(certified_bundle, certified, dissipation, drift)

        report = SymplecticIntegratorReport(
            state_next=self._freeze_state(
                certified.coordinates_next,
                certified.momenta_next,
                hamiltonian,
            ),
            symplectic_residual=certified.symplectic_residual,
            is_symplectically_invariant=certified.is_symplectically_invariant,
            dissipation_rate=dissipation,
            is_lyapunov_passive=is_passive,
            jacobian_matrix=certified.jacobian_matrix,
            liouville_residual=certified.liouville_residual,
            hamiltonian_drift=drift,
            kinetic_energy=kinetic,
            potential_energy=potential,
            potential_is_surrogate=surrogate,
            integration_verdict=final_verdict,
            recommended_action=action,
            symplectic_verdict=symplectic_verdict,
            passivity_verdict=passivity_verdict,
            energy_verdict=energy_verdict,
            jacobian_method=certified.jacobian_method,
            timestep=certified_bundle.timestep,
        )
        logger.debug(
            "Fase 3 veredicto: H=%.6e, ΔH=%.4e, Ḣ_R=%.4e, final=%s, acción=%s.",
            hamiltonian,
            drift,
            dissipation,
            final_verdict.name,
            action.value,
        )
        return report

    # ──────────────────────────────────────────────────────────────────────────
    # 3.4  API pública de integración (orden 2 y orden 4)
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_state(self, state: object) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        if not isinstance(state, SymplecticState):
            raise IntegrationStepError(
                "state_curr debe ser una instancia de SymplecticState."
            )
        q_coord = self._validate_vector(state.coordinates, "state_curr.coordinates", self._n)
        p_mom = self._validate_vector(state.momenta, "state_curr.momenta", self._n)
        hamiltonian = self._as_finite_float(state.hamiltonian, "state_curr.hamiltonian")
        return q_coord, p_mom, hamiltonian

    def _certify_yoshida_weights(self) -> tuple[float, float]:
        r"""
        Certifica las identidades algebraicas de Yoshida:

            $$w_0+2w_1=1,\qquad w_0^3+2w_1^3=0.$$
        """
        consistency = _YOSHIDA_W0 + 2.0 * _YOSHIDA_W1
        cancellation = (_YOSHIDA_W0 ** 3) + 2.0 * (_YOSHIDA_W1 ** 3)
        if abs(consistency - 1.0) > 8.0 * _MACHINE_EPS:
            raise YoshidaCompositionError(
                f"Yoshida pierde consistencia: w0+2w1 = {consistency:.16e}."
            )
        if abs(cancellation) > 1.0e-14:
            raise YoshidaCompositionError(
                f"Yoshida pierde el orden 4: w0³+2w1³ = {cancellation:.16e}."
            )
        return _YOSHIDA_W1, _YOSHIDA_W0

    def integrate_step_2nd_order(
        self,
        state_curr: SymplecticState,
        force_gradient_q: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        mass_matrix_inv: NDArray[np.float64],
        dt: float,
        damping_matrix_r: NDArray[np.float64] | None = None,
        external_forcing: NDArray[np.float64] | None = None,
        potential_q: Callable[[NDArray[np.float64]], float] | None = None,
        jacobian_method: str = JacobianMethod.NUMERICAL.value,
        stiffness_matrix_k: NDArray[np.float64] | None = None,
        force_gradient_complex_q: Callable[[NDArray[np.complex128]], NDArray[np.complex128]] | None = None,
        *,
        allow_negative_dt: bool = False,
        compute_jacobian: bool = True,
    ) -> SymplecticIntegratorReport:
        r"""
        Un paso de Störmer–Verlet de 2.º orden.

        Compone las tres fases:

            $$\Phi_{03}
              =F_3\circ\Phi_{23}\circ F_2\circ\Phi_{12}\circ F_1.$$

        ``allow_negative_dt`` está reservado a la composición de Yoshida
        (el API público debe invocarlo con $$\Delta t>0$$).
        """
        q_coord, p_mom, hamiltonian_prev = self._validate_state(state_curr)
        bundle = self.execute_phase1(
            q_coord,
            p_mom,
            mass_matrix_inv,
            dt,
            damping_matrix_r,
            external_forcing,
            allow_negative_dt=allow_negative_dt,
        )
        transition = self.execute_phase2(
            bundle,
            force_gradient_q,
            jacobian_method=jacobian_method,
            stiffness_matrix_k=stiffness_matrix_k,
            force_gradient_complex_q=force_gradient_complex_q,
            compute_jacobian=compute_jacobian,
        )
        return self.execute_phase3(
            bundle,
            transition,
            potential_q,
            force_gradient_q=force_gradient_q,
            hamiltonian_previous=hamiltonian_prev,
        )

    def integrate_step_4th_order(
        self,
        state_curr: SymplecticState,
        force_gradient_q: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        mass_matrix_inv: NDArray[np.float64],
        dt: float,
        damping_matrix_r: NDArray[np.float64] | None = None,
        external_forcing: NDArray[np.float64] | None = None,
        potential_q: Callable[[NDArray[np.float64]], float] | None = None,
        jacobian_method: str = JacobianMethod.NUMERICAL.value,
        stiffness_matrix_k: NDArray[np.float64] | None = None,
        force_gradient_complex_q: Callable[[NDArray[np.complex128]], NDArray[np.complex128]] | None = None,
    ) -> SymplecticIntegratorReport:
        r"""
        Un paso de Yoshida–Forest–Ruth de 4.º orden:

            $$\Phi_{\Delta t}^{[4]}
              =\Phi_{w_1\Delta t}^{[2]}
               \circ\Phi_{w_0\Delta t}^{[2]}
               \circ\Phi_{w_1\Delta t}^{[2]}.$$

        El Jacobiano global es el producto $$J_3 J_2 J_1$$.  El
        Hamiltoniano y la Rayleigh se evalúan en el estado terminal.
        """
        _q, _p, hamiltonian_prev = self._validate_state(state_curr)
        dt_step = self._validate_timestep(dt, allow_negative=False)
        weight_one, weight_zero = self._certify_yoshida_weights()

        substeps = (weight_one * dt_step, weight_zero * dt_step, weight_one * dt_step)
        cursor = state_curr
        jacobians: list[NDArray[np.float64]] = []
        last_report: SymplecticIntegratorReport | None = None

        for sub_dt in substeps:
            last_report = self.integrate_step_2nd_order(
                cursor,
                force_gradient_q,
                mass_matrix_inv,
                sub_dt,
                damping_matrix_r,
                external_forcing,
                potential_q,
                jacobian_method,
                stiffness_matrix_k,
                force_gradient_complex_q,
                allow_negative_dt=True,
                compute_jacobian=True,
            )
            cursor = last_report.state_next
            if last_report.jacobian_matrix is not None:
                jacobians.append(
                    np.array(last_report.jacobian_matrix, dtype=np.float64, copy=True)
                )

        if last_report is None:
            raise YoshidaCompositionError("La composición de Yoshida no produjo pasos.")

        composed: NDArray[np.float64] | None = None
        residual = last_report.symplectic_residual
        liouville = last_report.liouville_residual
        invariant = last_report.is_symplectically_invariant
        method = self._resolve_jacobian_method(jacobian_method)
        tolerance = _JACOBIAN_INVARIANCE_TOL[method.value]

        if len(jacobians) == 3:
            composed = jacobians[2] @ jacobians[1] @ jacobians[0]
            self._assert_finite(composed, "Jacobiano compuesto de Yoshida")
            residual, _relative = self._verify_symplectic_form(composed)
            liouville = self._verify_liouville_volume(composed)
            damping = (
                None
                if damping_matrix_r is None
                else self._validate_square_matrix(
                    damping_matrix_r, "damping_matrix_r", self._n
                )
            )
            conservative = damping is None or self._frobenius_norm(damping, "R") <= (
                float(self._n) * _MACHINE_EPS
            )
            invariant = residual <= tolerance if conservative else True

        drift = last_report.state_next.hamiltonian - hamiltonian_prev
        shadow_transition = PhaseTransitionData(
            coordinates_next=last_report.state_next.coordinates,
            momenta_next=last_report.state_next.momenta,
            momenta_half=last_report.state_next.momenta,
            jacobian_matrix=None if composed is None else _freeze_array(composed),
            jacobian_method=method.value,
            symplectic_residual=residual,
            relative_symplectic_residual=residual,
            liouville_residual=liouville,
            is_symplectically_invariant=invariant,
            invariance_tolerance=tolerance,
        )
        shadow_bundle = MechanicalBundleData(
            coordinates=state_curr.coordinates,
            momenta=state_curr.momenta,
            mass_matrix_inv=_freeze_array(
                self._validate_square_matrix(mass_matrix_inv, "mass_matrix_inv", self._n)
            ),
            damping_matrix_r=(
                None
                if damping_matrix_r is None
                else _freeze_array(
                    self._validate_square_matrix(
                        damping_matrix_r, "damping_matrix_r", self._n
                    )
                )
            ),
            external_forcing=(
                None
                if external_forcing is None
                else _freeze_array(
                    self._validate_vector(external_forcing, "external_forcing", self._n)
                )
            ),
            rayleigh_factor=_freeze_array(np.eye(self._n, dtype=np.float64)),
            timestep=dt_step,
            mass_condition_number=1.0,
            mass_spectral_minimum=1.0,
            damping_spectral_minimum=0.0,
            is_conservative=damping_matrix_r is None,
            is_implicit_rayleigh=False,
        )
        (
            symplectic_verdict,
            passivity_verdict,
            energy_verdict,
            final_verdict,
            action,
        ) = self._classify_report(
            shadow_bundle,
            shadow_transition,
            last_report.dissipation_rate,
            drift,
        )

        logger.debug(
            "Integración 4º orden: r_ω=%.4e, r_Liouville=%.4e, ΔH=%.4e, final=%s.",
            residual,
            liouville,
            drift,
            final_verdict.name,
        )
        return SymplecticIntegratorReport(
            state_next=last_report.state_next,
            symplectic_residual=residual,
            is_symplectically_invariant=invariant,
            dissipation_rate=last_report.dissipation_rate,
            is_lyapunov_passive=last_report.is_lyapunov_passive,
            jacobian_matrix=None if composed is None else _freeze_array(composed),
            liouville_residual=liouville,
            hamiltonian_drift=drift,
            kinetic_energy=last_report.kinetic_energy,
            potential_energy=last_report.potential_energy,
            potential_is_surrogate=last_report.potential_is_surrogate,
            integration_verdict=final_verdict,
            recommended_action=action,
            symplectic_verdict=symplectic_verdict,
            passivity_verdict=passivity_verdict,
            energy_verdict=energy_verdict,
            jacobian_method=method.value,
            timestep=dt_step,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR: SYMPLECTIC VERLET INTEGRATOR
# ══════════════════════════════════════════════════════════════════════════════
class SymplecticVerletIntegrator(Morphism, Phase3_YoshidaThermodynamicIntegrator):
    r"""
    Integrador geométrico de lazo cerrado.

    Expone el contrato histórico

        ``integrate_step_2nd_order``, ``integrate_step_4th_order``

    como la composición

        $$F_{\mathrm{Integrator}}
          =F_3\circ\Phi_{23}\circ F_2\circ\Phi_{12}\circ F_1.$$
    """

    def __init__(self, dimension: int, tolerance: float = _DEFAULT_TOL) -> None:
        Phase1_PortHamiltonianSpectrometer.__init__(self, dimension, tolerance)

    @property
    def dimension(self) -> int:
        """Dimensión $$n$$ del espacio de configuración."""
        return self._n

    @property
    def omega(self) -> NDArray[np.float64]:
        r"""2-forma canónica $$\Omega\in\mathbb{R}^{2n\times 2n}$$, de solo lectura."""
        return self._omega


# ══════════════════════════════════════════════════════════════════════════════
# §F. REGULARIZACIÓN DE HIGHAM Y JACOBIANO DISPERSO
# ══════════════════════════════════════════════════════════════════════════════
def stable_mass_matrix_higham(
    matrix: NDArray[np.float64],
    floor: float = _HIGHAM_EIGEN_FLOOR,
) -> NDArray[np.float64]:
    r"""
    Proyección de Higham (2002) sobre el cono SPD:

        $$A\mapsto Q\,\mathrm{diag}(\max(\lambda_i,\vartheta))\,Q^\top,$$

    tras simetrizar $$A\leftarrow\tfrac12(A+A^\top)$$.  El resultado se
    re-simetriza y se exige $$\lambda_{\min}\ge\vartheta$$.
    """
    try:
        array = np.array(matrix, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise SystemConfigurationError(
            "A debe ser convertible a una matriz float64."
        ) from exc
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise SystemConfigurationError("A debe ser una matriz cuadrada.")
    if array.shape[0] == 0:
        raise SystemConfigurationError("A no puede ser vacía.")
    if not np.all(np.isfinite(array)):
        raise SystemConfigurationError("A contiene valores no finitos.")
    try:
        eigen_floor = float(floor)
    except (TypeError, ValueError) as exc:
        raise SystemConfigurationError("floor debe ser un número real.") from exc
    if not math.isfinite(eigen_floor) or eigen_floor <= 0.0:
        raise SystemConfigurationError("floor debe ser finito y estrictamente positivo.")

    symmetric = 0.5 * (array + array.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise MassMatrixSPDError(
            "La descomposición espectral de Higham falló."
        ) from exc
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise MassMatrixSPDError("El espectro de Higham no es finito.")

    projected = np.maximum(eigenvalues, eigen_floor)
    rebuilt = (eigenvectors * projected) @ eigenvectors.T
    rebuilt = 0.5 * (rebuilt + rebuilt.T)
    if not np.all(np.isfinite(rebuilt)):
        raise MassMatrixSPDError("La reconstrucción de Higham produjo no-finitos.")
    lambda_min = float(np.min(np.linalg.eigvalsh(rebuilt)))
    if lambda_min < 0.5 * eigen_floor:
        raise MassMatrixSPDError(
            "Higham no logró proyectar al cono SPD. "
            f"λ_min = {lambda_min:.4e}, floor = {eigen_floor:.4e}."
        )
    return rebuilt


def compute_sparse_jacobian(
    q_coord: NDArray[np.float64],
    p_mom: NDArray[np.float64],
    force_gradient_q: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    mass_matrix_inv_sparse: Any,
    dt: float,
    damping_matrix_r_sparse: Any | None = None,
    h: float = 1.0e-8,
) -> Any:
    r"""
    Jacobiano disperso por diferencias adelantadas (compañero de gran escala).

    Usa el campo PHS *explícito* $$F-Rv+u$$ (sin factor $$A$$) para no
    forzar una factorización dispersa por cada columna.  Es un
    aproximante de primer orden, no un certificado de $$\mathrm{Sp}(2n)$$.
    """
    import scipy.sparse as sp

    try:
        q_array = np.array(q_coord, dtype=np.float64, copy=True)
        p_array = np.array(p_mom, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise SystemConfigurationError(
            "q y p deben ser convertibles a float64."
        ) from exc
    if q_array.ndim != 1 or p_array.ndim != 1:
        raise SystemConfigurationError("q y p deben ser vectores 1-D.")
    if q_array.size == 0 or q_array.size != p_array.size:
        raise SystemConfigurationError("q y p deben tener la misma dimensión positiva.")
    if not np.all(np.isfinite(q_array)) or not np.all(np.isfinite(p_array)):
        raise SystemConfigurationError("q o p contienen valores no finitos.")
    if not callable(force_gradient_q):
        raise SystemConfigurationError("force_gradient_q debe ser un callable.")
    try:
        timestep = float(dt)
        increment = float(h)
    except (TypeError, ValueError) as exc:
        raise SystemConfigurationError("dt y h deben ser números reales.") from exc
    if not math.isfinite(timestep) or timestep == 0.0:
        raise SystemConfigurationError("dt debe ser finito y no nulo.")
    if not math.isfinite(increment) or increment <= 0.0:
        raise SystemConfigurationError("h debe ser finito y estrictamente positivo.")

    dimension = q_array.size
    mass_inv = sp.csr_matrix(mass_matrix_inv_sparse)
    damping = (
        None
        if damping_matrix_r_sparse is None
        else sp.csr_matrix(damping_matrix_r_sparse)
    )
    if mass_inv.shape != (dimension, dimension):
        raise SystemConfigurationError(
            "mass_matrix_inv_sparse debe ser de forma (n, n)."
        )
    if damping is not None and damping.shape != (dimension, dimension):
        raise SystemConfigurationError(
            "damping_matrix_r_sparse debe ser de forma (n, n)."
        )

    def step_explicit(
        q_val: NDArray[np.float64],
        p_val: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        velocity = mass_inv.dot(p_val)
        rayleigh = np.zeros(dimension, dtype=np.float64) if damping is None else damping.dot(velocity)
        force_q = np.asarray(force_gradient_q(q_val), dtype=np.float64)
        if force_q.shape != (dimension,) or not np.all(np.isfinite(force_q)):
            raise IntegrationStepError(
                "force_gradient_q devolvió un vector inadmisible."
            )
        p_half = p_val + 0.5 * timestep * (force_q - rayleigh)
        q_next = q_val + timestep * mass_inv.dot(p_half)
        velocity_half = mass_inv.dot(p_half)
        rayleigh_half = (
            np.zeros(dimension, dtype=np.float64)
            if damping is None
            else damping.dot(velocity_half)
        )
        force_next = np.asarray(force_gradient_q(q_next), dtype=np.float64)
        if force_next.shape != (dimension,) or not np.all(np.isfinite(force_next)):
            raise IntegrationStepError(
                "force_gradient_q(q_{+}) devolvió un vector inadmisible."
            )
        p_next = p_half + 0.5 * timestep * (force_next - rayleigh_half)
        return np.concatenate([q_next, p_next])

    base = step_explicit(q_array, p_array)
    jacobian = sp.lil_matrix((2 * dimension, 2 * dimension), dtype=np.float64)

    for index in range(dimension):
        q_pert = q_array.copy()
        q_pert[index] += increment
        column = (step_explicit(q_pert, p_array) - base) / increment
        for row, value in enumerate(column):
            if abs(float(value)) > _MACHINE_EPS:
                jacobian[row, index] = float(value)

        p_pert = p_array.copy()
        p_pert[index] += increment
        column = (step_explicit(q_array, p_pert) - base) / increment
        for row, value in enumerate(column):
            if abs(float(value)) > _MACHINE_EPS:
                jacobian[row, dimension + index] = float(value)

    return jacobian.tocsr()


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "SymplecticIntegrationError",
    "SystemConfigurationError",
    "IntegrationStepError",
    "SymplecticInvarianceError",
    "PassivityViolationError",
    "MassMatrixSPDError",
    "PhaseHandoffError",
    "YoshidaCompositionError",
    "IntegrationVerdict",
    "IntegrationAction",
    "JacobianMethod",
    "MechanicalBundleData",
    "PhaseTransitionData",
    "SymplecticState",
    "SymplecticIntegratorReport",
    "Phase1_PortHamiltonianSpectrometer",
    "Phase2_VariationalFlowSynthesizer",
    "Phase3_YoshidaThermodynamicIntegrator",
    "SymplecticVerletIntegrator",
    "stable_mass_matrix_higham",
    "compute_sparse_jacobian",
]
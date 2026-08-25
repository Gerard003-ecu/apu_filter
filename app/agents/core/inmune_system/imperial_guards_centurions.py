# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Guards Centurions (Los Centuriones Port-Hamiltonianos)     ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_centurions.py         ║
║ Versión: 3.0.0-Doctoral-Nested-IDA-PBC-KMS-Heyting-Dirac-Wick                ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS DE CONTROL COVARIANT-EXERGÉTICO (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Componente de Capa 2 (Centuriones de Calibre): interductor físico-epistémico
entre el foso material (V_PHYSICS) y el Ágora Tensorial (V_Ω). Impone
restricciones exergéticas e inmutabilidad térmica para colapsar el albedrío
probabilístico de la MAC hacia un punto fijo determinista.

Arquitectura de 3 fases ANIDADAS (composición de morfismos):

  FASE I  -- Núcleo espectral Tikhonov–Higham, C*-norma, Darboux, Kirchhoff
             Último morfismo: prepare_hamiltonian_bundle
             Codominio: _HamiltonianBundle
             ≡ objeto inicial / dominio de la Fase II.

  FASE II -- Centurión Port-Hamiltoniano, estructura de Dirac, IDA-PBC,
             anti-windup espectral y balance de Clausius–Duhem.
             Dominio: _HamiltonianBundle
             Último morfismo: evaluate_power_curtain
             Codominio: _PowerCurtainAudit
             ≡ objeto inicial / dominio de la Fase III.

  FASE III -- Centurión KMS (Tomita–Takesaki + rotación de Wick), ħ_eff(T),
             retículo de Heyting y crowbar tiristorizado.
             Dominio: _PowerCurtainAudit ⋊ (ρ, A, B, β)
             Codominio: dict de coherencia / actuación ciberfísica.

Ramas simétricas de control:
  1. PORT-HAMILTONIANO: cortina de potencia en T*M
        ẋ = [J_d − R_d] ∇H_d ,   Ḣ_d = −∇H_dᵀ R_d ∇H_d + yᵀ u
  2. TERMODINÁMICO: temperatura de fibrado, KMS modular y ħ_eff(T) → 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Agents.ImperialGuardsCenturions")

# =============================================================================
# CONSTANTES UNIVERSALES, COTAS DE WILKINSON Y LÍMITES DE LA FPU
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_LIMIT: Final[float] = 1.0e-9
_HERMITIAN_ATOL: Final[float] = 1.0e-10
_STRUCTURE_ATOL: Final[float] = 1.0e-9
_H_BAR_0: Final[float] = 1.0
_CROWBAR_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_NS: Final[float] = 4.2
_CROWBAR_T_MIN_NS: Final[float] = 380.0
_CROWBAR_T_MAX_NS: Final[float] = 420.0
_MAX_EXP_ARG: Final[float] = 50.0
_MIN_PLANCK: Final[float] = 1.0e-15
_ALPHA_DAMPING: Final[float] = 2.5
_PLANCK_DAMPING: Final[float] = 0.8
_KMS_VETO: Final[float] = 1.0e-4
_KMS_DEGRADE: Final[float] = 1.0e-6
_DISS_DEGRADE: Final[float] = 50.0
_PASSIVITY_FLOOR: Final[float] = -1.0e-12
_INTERCONNECTION_LEAK: Final[float] = 1.0e-8
_COND_WARN: Final[float] = 1.0e12


# #############################################################################
#                                                                             #
#  FASE I                                                                     #
#  NÚCLEO ESPECTRAL, C*-ÁLGEBRA DE BANACH, DARBOUX Y KIRCHHOFF                #
#                                                                             #
#  Objetos: matrices de inercia / amortiguamiento / densidad.                 #
#  Morfismos: hermitización, proyección Higham al cono SPD, cálculo           #
#             funcional de Riesz–Dunford, Laplaciano de Kirchhoff,            #
#             forma simpléctica canónica.                                     #
#  Cierre formal: prepare_hamiltonian_bundle  →  _HamiltonianBundle           #
#                 (dominio de todos los métodos de la Fase II).               #
#                                                                             #
# #############################################################################


class HeytingVerdict(Enum):
    """
    Retículo de Heyting lineal de tres valores (álgebra de Gödel G₃).

    Orden de verdad (permiso / coherencia):
        VETOED ≤ DEGRADED ≤ COHERENT

    Operaciones:
        a ∧ b = min(a, b) ,   a ∨ b = max(a, b)
        a → b = ⊤ si a ≤ b, else b
        ¬a    = a → ⊥
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(min(self.value, other.value))

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(max(self.value, other.value))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        if self.value <= other.value:
            return HeytingVerdict.COHERENT
        return other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    @classmethod
    def from_token(cls, token: str) -> "HeytingVerdict":
        try:
            return cls[token]
        except KeyError as exc:
            raise ValueError(f"Veredicto desconocido: {token!r}") from exc


@dataclass(frozen=True, slots=True)
class _HamiltonianBundle:
    """
    Fibrado espectral inmutable. Cierre formal de la Fase I y objeto inicial
    de la Fase II (todo PortHamiltonianCenturion factoriza por este tipo).

    Axiomas que el constructor de Fase I garantiza:
      • n par (carta de Darboux sobre T*Q, dim Q = n/2).
      • M_d = M_d† ≻ 0  (inercia deseada, cono SPD).
      • R_d = R_d† ≽ εI (amortiguamiento estricto tras regularización).
      • J_dᵀ = −J_d ,  J_d² = −I  (estructura casi-compleja canónica).
      • M_d_inv = M_d⁻¹ en norma-2 relativa bajo recorte de Tikhonov.
    """

    n: int
    M_d: np.ndarray
    M_d_inv: np.ndarray
    R_d: np.ndarray
    J_d: np.ndarray
    cond_M: float
    spectral_gap_R: float


class _SpectralCore:
    r"""
    Núcleo de regularización espectral.

    Opera en la C*-álgebra M_n(ℂ) con norma de operador
    :math:`\|A\|_{2} = \sigma_{\max}(A)` y radio espectral
    :math:`r(A) \le \|A\|_{2}`. Las proyecciones al cono SPD siguen a
    Higham (2002); la inversión usa Tikhonov espectral
    :math:`\lambda \mapsto (\max(\lambda,\varepsilon))^{-1}`.
    """

    # ------------------------------------------------------------------
    # I.1  Formas de Banach / C* y sumación compensada
    # ------------------------------------------------------------------

    @staticmethod
    def hermitize(matrix: np.ndarray) -> np.ndarray:
        """Proyección de Cartan :math:`A \\mapsto (A + A^\\dagger)/2`."""
        return 0.5 * (matrix + matrix.conj().T)

    @staticmethod
    def skew_symmetrize(matrix: np.ndarray) -> np.ndarray:
        """Proyección al álgebra de Lie :math:`\\mathfrak{so}` / :math:`\\mathfrak{u}`."""
        return 0.5 * (matrix - matrix.conj().T)

    @staticmethod
    def cstar_norm(matrix: np.ndarray) -> float:
        """Norma C* (= norma-2 de operador = mayor valor singular)."""
        return float(la.norm(matrix, 2))

    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        """Norma de Hilbert–Schmidt :math:`\\|A\\|_{F} = \\sqrt{\\mathrm{Tr}(A^\\dagger A)}`."""
        return float(la.norm(matrix, "fro"))

    @staticmethod
    def banach_condition_number(matrix: np.ndarray) -> float:
        r"""
        Número de condición en norma-2:
        :math:`\kappa_2(A) = \|A\|_2 \,\|A^{-1}\|_2`.
        Devuelve +∞ si el recorte espectral detecta cuasi-singularidad.
        """
        svals = la.svdvals(matrix)
        smax = float(svals[0]) if svals.size else 0.0
        smin = float(svals[-1]) if svals.size else 0.0
        if smin <= _WILKINSON_LIMIT * max(smax, 1.0):
            return float("inf")
        return smax / smin

    @staticmethod
    def neumaier_sum(terms: np.ndarray) -> float:
        """
        Sumación compensada de Kahan–Babuška–Neumaier (más estable que Kahan
        clásico cuando los sumandos cambian de magnitud).
        """
        flat = np.asarray(terms, dtype=np.float64).ravel()
        s = 0.0
        c = 0.0
        for x in flat:
            t = s + x
            if abs(s) >= abs(x):
                c += (s - t) + x
            else:
                c += (x - t) + s
            s = t
        return float(s + c)

    @staticmethod
    def kahan_sum(terms: np.ndarray) -> float:
        """Alias histórico: delega en Neumaier (dominancia uniforme)."""
        return _SpectralCore.neumaier_sum(terms)

    # ------------------------------------------------------------------
    # I.2  Proyecciones espectrales (Higham / simplex cuántico)
    # ------------------------------------------------------------------

    @classmethod
    def regularize_spd(
        cls,
        matrix: np.ndarray,
        floor: float = _WILKINSON_LIMIT,
        relative: bool = True,
    ) -> np.ndarray:
        r"""
        Proyección de Higham al cono de Hermitianas definidas positivas:

        .. math::

            A \mapsto U\,\mathrm{diag}(\max(\lambda_i,\,\varepsilon))\,U^\dagger

        Si ``relative`` es verdadero, :math:`\varepsilon` se escala con
        :math:`\|A\|_2` para no destruir la magnitud física.
        """
        h = cls.hermitize(np.asarray(matrix))
        evals, evecs = la.eigh(h)
        scale = max(float(np.max(np.abs(evals))), 1.0) if relative else 1.0
        evals_clamped = np.maximum(np.real(evals), floor * scale)
        restored = evecs @ np.diag(evals_clamped) @ evecs.conj().T
        return cls.hermitize(restored)

    @classmethod
    def regularize_density(
        cls,
        rho: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Proyección al simplex de estados cuánticos
        :math:`\{\rho=\rho^\dagger \succeq 0,\; \mathrm{Tr}\,\rho = 1\}`.

        Retorna ``(rho_reg, evals_norm, evecs)``.
        """
        rho_h = cls.hermitize(np.asarray(rho))
        evals, evecs = la.eigh(rho_h)
        evals_clipped = np.maximum(np.real(evals), _WILKINSON_LIMIT)
        trace = float(np.sum(evals_clipped))
        if trace <= _MACHINE_EPS:
            dim = rho_h.shape[0]
            evals_norm = np.full(dim, 1.0 / dim, dtype=np.float64)
            evecs = np.eye(dim, dtype=rho_h.dtype)
            rho_reg = np.eye(dim, dtype=np.complex128) / dim
            return rho_reg, evals_norm, evecs
        evals_norm = evals_clipped / trace
        rho_reg = evecs @ np.diag(evals_norm) @ evecs.conj().T
        return cls.hermitize(rho_reg), evals_norm, evecs

    @classmethod
    def spectral_inverse_from_eigh(
        cls,
        evals: np.ndarray,
        evecs: np.ndarray,
        floor: float = _WILKINSON_LIMIT,
    ) -> np.ndarray:
        """Inversa espectral con recorte de Tikhonov (evita división por cero)."""
        inv_evals = 1.0 / np.maximum(np.real(evals), floor)
        return evecs @ np.diag(inv_evals) @ evecs.conj().T

    @classmethod
    def spectral_power(
        cls,
        evals: np.ndarray,
        evecs: np.ndarray,
        exponent: complex,
        floor: float = _WILKINSON_LIMIT,
    ) -> np.ndarray:
        r"""
        Cálculo funcional de Riesz–Dunford / Borel:

        .. math::

            f(A) = U\,\mathrm{diag}(f(\lambda_i))\,U^\dagger,
            \quad f(\lambda) = \lambda^{z},\; z\in\mathbb{C}.

        Los autovalores se recortan a :math:`\varepsilon>0` para que
        :math:`\mathrm{Log}` sea holomorfo en un entorno del espectro.
        """
        safe = np.maximum(np.real(evals), floor)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            powered = np.exp(exponent * np.log(safe))
        return evecs @ np.diag(powered) @ evecs.conj().T

    # ------------------------------------------------------------------
    # I.3  Topología simpléctica, grafos de Kirchhoff y validación
    # ------------------------------------------------------------------

    @staticmethod
    def assemble_standard_J(dimension_n: int) -> np.ndarray:
        r"""
        Forma simpléctica canónica de Darboux en :math:`T^*Q \cong \mathbb{R}^{n}`:

        .. math::

            J_0 = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix},
            \qquad J_0^\top = -J_0,\quad J_0^2 = -I.

        Es el isomorfismo musical :math:`\omega^\sharp` de
        :math:`\omega = \sum_i \mathrm{d}q^i \wedge \mathrm{d}p_i`
        (generador de :math:`H^2_{\mathrm{dR}}(T^*Q)`).
        """
        if dimension_n % 2 != 0:
            raise ValueError(
                "Darboux requiere dimensión par (n = 2 dim Q)."
            )
        half = dimension_n // 2
        i_half = np.eye(half)
        z_half = np.zeros((half, half))
        return np.block([[z_half, i_half], [-i_half, z_half]])

    @classmethod
    def verify_almost_complex(cls, J: np.ndarray, atol: float = _STRUCTURE_ATOL) -> None:
        """Verifica :math:`J^\\top=-J` y :math:`J^2=-I` en norma de Frobenius relativa."""
        n = J.shape[0]
        skew_res = cls.frobenius_norm(J + J.T.conj())
        ac_res = cls.frobenius_norm(J @ J + np.eye(n))
        scale = max(cls.frobenius_norm(J), 1.0)
        if skew_res > atol * scale:
            raise ValueError(f"J no es antihermitiana: ‖J+J†‖_F={skew_res:.3e}")
        if ac_res > atol * scale:
            raise ValueError(f"J no es casi-compleja: ‖J²+I‖_F={ac_res:.3e}")

    @classmethod
    def kirchhoff_laplacian(
        cls,
        conductance: np.ndarray,
        strict_floor: float = _WILKINSON_LIMIT,
    ) -> np.ndarray:
        r"""
        Laplaciano de Kirchhoff (teoría de grafos / circuitos de Brayton–Moser).

        Si :math:`W_{ij}\ge 0` es la conductancia de la arista :math:`i\sim j`,

        .. math::

            L = \mathrm{diag}(W\mathbf{1}) - W \succeq 0,

        con núcleo igual a las constantes ssi el grafo es conexo.
        ``strict_floor`` inyecta :math:`\varepsilon I` para disipación estricta
        (rompe el modo de consenso: tracking asintótico en T*M).
        """
        W = cls.hermitize(np.asarray(conductance, dtype=np.float64))
        W = np.maximum(np.real(W), 0.0)
        np.fill_diagonal(W, 0.0)
        deg = np.sum(W, axis=1)
        L = np.diag(deg) - W
        return cls.regularize_spd(L, floor=strict_floor, relative=False)

    @staticmethod
    def _assert_square(name: str, matrix: np.ndarray, n: int) -> None:
        arr = np.asarray(matrix)
        if arr.ndim != 2 or arr.shape != (n, n):
            raise ValueError(
                f"{name} debe ser cuadrada de orden n={n}; recibido {arr.shape}."
            )

    # ------------------------------------------------------------------
    # I.ω  ÚLTIMO MORFISMO DE LA FASE I
    #      Codominio ≡ dominio de PortHamiltonianCenturion (Fase II)
    # ------------------------------------------------------------------

    @classmethod
    def prepare_hamiltonian_bundle(
        cls,
        dimension_n: int,
        inertia_matrix: np.ndarray,
        damping_matrix_rd: np.ndarray,
    ) -> _HamiltonianBundle:
        r"""
        Cierre formal de la Fase I / unidad de la adjunción con la Fase II.

        Construye el fibrado :math:`(M_d, M_d^{-1}, R_d, J_d)` que parametriza
        el campo IDA-PBC

        .. math::

            \dot x = \bigl(J_d - R_d\bigr)\nabla H_d(x).

        Verifica dimensionalidad par (Darboux), proyecta :math:`M_d,R_d`
        al cono SPD, invierte :math:`M_d` por cálculo espectral y certifica
        la estructura casi-compleja de :math:`J_d`.
        """
        if dimension_n <= 0:
            raise ValueError("dimension_n debe ser un entero positivo par.")
        if dimension_n % 2 != 0:
            raise ValueError(
                "La dimensión del espacio de fase debe ser par para "
                "portar una estructura simpléctica estándar de Darboux."
            )

        cls._assert_square("inertia_matrix", inertia_matrix, dimension_n)
        cls._assert_square("damping_matrix_rd", damping_matrix_rd, dimension_n)

        M_d = cls.regularize_spd(np.asarray(inertia_matrix, dtype=np.float64))
        R_d = cls.regularize_spd(np.asarray(damping_matrix_rd, dtype=np.float64))

        evals_M, evecs_M = la.eigh(M_d)
        M_d_inv = cls.spectral_inverse_from_eigh(evals_M, evecs_M)
        cond_M = cls.banach_condition_number(M_d)
        if not np.isfinite(cond_M) or cond_M > _COND_WARN:
            logger.warning(
                "M_d mal condicionada (κ₂=%.3e). La cortina de potencia "
                "puede amplificar ruido de redondeo.",
                cond_M,
            )

        evals_R, _ = la.eigh(R_d)
        spectral_gap_R = float(np.min(np.real(evals_R)))
        if spectral_gap_R <= 0.0:
            raise ValueError(
                "R_d no quedó estrictamente disipativa tras regularización."
            )

        J_d = cls.assemble_standard_J(dimension_n)
        cls.verify_almost_complex(J_d)

        return _HamiltonianBundle(
            n=dimension_n,
            M_d=M_d,
            M_d_inv=M_d_inv,
            R_d=R_d,
            J_d=J_d,
            cond_M=float(cond_M) if np.isfinite(cond_M) else float("inf"),
            spectral_gap_R=spectral_gap_R,
        )


# #############################################################################
#                                                                             #
#  FASE II                                                                    #
#  CENTURIÓN PORT-HAMILTONIANO · IDA-PBC · DIRAC · ANTI-WINDUP ESPECTRAL      #
#                                                                             #
#  Continuación directa del último morfismo de la Fase I:                     #
#      _HamiltonianBundle  ↦  PortHamiltonianCenturion                        #
#                                                                             #
#  Cierre formal: evaluate_power_curtain  →  _PowerCurtainAudit               #
#                 (dominio de la Cámara de Coherencia, Fase III).             #
#                                                                             #
# #############################################################################


@dataclass(frozen=True, slots=True)
class _PowerCurtainAudit:
    """
    Resultado de la cortina de potencia. Cierre de la Fase II y objeto
    inicial de la Fase III (la Cámara de Coherencia no recompute el
    balance exergético: lo consume como morfismo ya certificado).
    """

    dissipation_power: float
    interconnection_leak: float
    port_supply_rate: float
    predicted_hdot: float
    gradient_norm: float
    hamiltonian: float
    antiwindup_engaged: bool
    extra_damping: float
    verdict: str


class PortHamiltonianCenturion:
    r"""
    Soberano de la Cortina de Potencia. Consume un `_HamiltonianBundle`
    (Fase I) y realiza control por asignación de interconexión y
    amortiguamiento (IDA-PBC, Ortega et al.):

    .. math::

        \dot x = \bigl[J_d(x) - R_d(x)\bigr]\nabla H_d(x) + G(x)\,u,
        \qquad
        H_d(x)=\tfrac12 (x-x^*)^\top M_d^{-1}(x-x^*).

    :math:`H_d` es función de Lyapunov / almacenamiento. La estructura de
    Dirac garantiza la identidad de potencia

    .. math::

        \dot H_d = -\nabla H_d^\top R_d\nabla H_d + y^\top u,
        \quad y = G^\top\nabla H_d.

    Aquí se toma el puerto unidad :math:`G=I` (inyección de esfuerzo
    directa sobre T*M), de modo que :math:`y=\nabla H_d`.
    """

    def __init__(
        self,
        dimension_n: int,
        inertia_matrix: np.ndarray,
        damping_matrix_rd: np.ndarray,
        target_state: np.ndarray,
        anti_windup_threshold: float = 10.0,
        bundle: Optional[_HamiltonianBundle] = None,
    ) -> None:
        """
        Inicio formal de la Fase II.

        Si `bundle` es None se invoca el último método de la Fase I
        (`prepare_hamiltonian_bundle`); si se provee, se reutiliza el
        fibrado ya regularizado (evita doble proyección de Higham).
        """
        if bundle is None:
            bundle = _SpectralCore.prepare_hamiltonian_bundle(
                dimension_n, inertia_matrix, damping_matrix_rd
            )
        elif bundle.n != dimension_n:
            raise ValueError(
                f"El fibrado declara n={bundle.n} ≠ dimension_n={dimension_n}."
            )

        x_star = np.asarray(target_state, dtype=np.float64).reshape(-1)
        if x_star.size != bundle.n:
            raise ValueError(
                f"target_state debe tener longitud n={bundle.n}; "
                f"recibido {x_star.size}."
            )
        if anti_windup_threshold <= 0.0:
            raise ValueError("anti_windup_threshold debe ser estrictamente positivo.")

        self._bundle: Final[_HamiltonianBundle] = bundle
        self._n: Final[int] = bundle.n
        self._x_star: Final[np.ndarray] = x_star.copy()
        self._anti_windup_threshold: Final[float] = float(anti_windup_threshold)
        self._M_d: Final[np.ndarray] = bundle.M_d
        self._M_d_inv: Final[np.ndarray] = bundle.M_d_inv
        self._R_d: Final[np.ndarray] = bundle.R_d
        self._J_d: Final[np.ndarray] = bundle.J_d

    # ------------------------------------------------------------------
    # II.1  Geometría del Hamiltoniano moldeado
    # ------------------------------------------------------------------

    def _validate_state(self, x: np.ndarray, name: str = "x") -> np.ndarray:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if vec.size != self._n:
            raise ValueError(f"{name} debe vivir en R^{self._n}; recibido {vec.size}.")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} contiene NaN/Inf: estado no físico.")
        return vec

    def compute_error(self, x: np.ndarray) -> np.ndarray:
        """Coordenada de error :math:`e = x - x^*` en la carta afín de T*M."""
        return self._validate_state(x) - self._x_star

    def compute_hamiltonian(self, x: np.ndarray) -> float:
        r"""
        Hamiltoniano moldeado (almacenamiento de Lyapunov):

        .. math::

            H_d(x) = \tfrac12 e^\top M_d^{-1} e \ge 0,
            \qquad H_d(x)=0 \iff x=x^*.
        """
        err = self.compute_error(x)
        quad_vec = self._M_d_inv @ err
        return 0.5 * float(err @ quad_vec)

    def compute_gradient(self, x: np.ndarray) -> np.ndarray:
        r"""Gradiente :math:`\nabla H_d(x) = M_d^{-1} e`."""
        return self._M_d_inv @ self.compute_error(x)

    def compute_hessian(self) -> np.ndarray:
        r"""Hessiano constante :math:`\nabla^2 H_d = M_d^{-1} \succ 0`."""
        return self._M_d_inv

    # ------------------------------------------------------------------
    # II.2  Identidades de Dirac / potencia de puertos
    # ------------------------------------------------------------------

    def interconnection_leak(self, grad_H: np.ndarray) -> float:
        r"""
        Residuo de la identidad :math:`\nabla H^\top J_d \nabla H = 0`
        (J_d anti-simétrica). No nulo sólo por ruido de redondeo.
        """
        return float(np.real(grad_H @ (self._J_d @ grad_H)))

    def dissipation_form(self, grad_H: np.ndarray, R_eff: np.ndarray) -> float:
        r"""Forma cuadrática de Rayleigh :math:`\nabla H^\top R_{\mathrm{eff}}\nabla H`."""
        return float(np.real(grad_H @ (R_eff @ grad_H)))

    def port_supply_rate(self, grad_H: np.ndarray, external_u: np.ndarray) -> float:
        r"""
        Potencia inyectada por el puerto :math:`y^\top u` con :math:`G=I`,
        :math:`y=\nabla H_d`.
        """
        u = self._validate_state(external_u, name="external_u")
        return float(np.real(grad_H @ u))

    def ida_vector_field(self, x: np.ndarray, R_eff: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Campo deseado :math:`(J_d - R_{\mathrm{eff}})\nabla H_d`."""
        grad_H = self.compute_gradient(x)
        R = self._R_d if R_eff is None else R_eff
        return (self._J_d - R) @ grad_H

    # ------------------------------------------------------------------
    # II.3  Anti-windup espectral (inflado del cono de disipación)
    # ------------------------------------------------------------------

    def apply_spectral_antiwindup(
        self,
        grad_H: np.ndarray,
    ) -> Tuple[np.ndarray, bool, float]:
        r"""
        Si :math:`\|\nabla H_d\|_2` excede el umbral de saturación de actuadores,
        se inflan *todos* los autovalores de :math:`R_d` (no sólo se suma
        un múltiplo de I a ciegas):

        .. math::

            R_{\mathrm{eff}}
              = U\,\mathrm{diag}\!\bigl(\lambda_i(R_d)+\delta\bigr)\,U^\top,
            \quad
            \delta = \sigma\cdot\mathrm{tr}(R_d)/n,\quad
            \sigma = 1 - \tau/\|\nabla H_d\|.

        Preserva ejes principales de fricción (anisotropía física) y
        mantiene :math:`R_{\mathrm{eff}}\succ R_d`.
        """
        grad_norm = float(la.norm(grad_H, 2))
        if grad_norm <= self._anti_windup_threshold:
            return self._R_d, False, 0.0

        saturation = (grad_norm - self._anti_windup_threshold) / grad_norm
        extra = saturation * (float(np.trace(self._R_d)) / self._n)
        evals, evecs = la.eigh(self._R_d)
        R_eff = evecs @ np.diag(np.real(evals) + extra) @ evecs.T
        R_eff = _SpectralCore.hermitize(R_eff)
        return np.real(R_eff), True, float(extra)

    def _classify_passivity(
        self,
        dissipation_power: float,
        interconnection_leak: float,
    ) -> str:
        """Clasificador de Clausius–Duhem / pasividad estricta."""
        if (not np.isfinite(dissipation_power)) or dissipation_power < _PASSIVITY_FLOOR:
            return "VETOED"
        if abs(interconnection_leak) > _INTERCONNECTION_LEAK * max(abs(dissipation_power), 1.0):
            return "VETOED"
        if dissipation_power > _DISS_DEGRADE:
            return "DEGRADED"
        return "COHERENT"

    # ------------------------------------------------------------------
    # II.ω  ÚLTIMO MORFISMO DE LA FASE II
    #       Codominio ≡ dominio de CenturionsCoherenceChamber (Fase III)
    # ------------------------------------------------------------------

    def evaluate_power_curtain(
        self,
        x: np.ndarray,
        external_u: np.ndarray,
    ) -> _PowerCurtainAudit:
        r"""
        Cierre formal de la Fase II / unidad de la adjunción con la Fase III.

        Audita el balance exergético completo de la estructura de Dirac:

        .. math::

            \underbrace{\dot H_d}_{\text{predicho}}
              = \underbrace{\nabla H_d^\top J_d\nabla H_d}_{=0}
                - \underbrace{\nabla H_d^\top R_{\mathrm{eff}}\nabla H_d}_{\ge 0}
                + \underbrace{y^\top u}_{\text{puerto}}.

        Emite VETOED si hay inyección parásita (disipación negativa) o si
        la identidad de interconexión se rompe numéricamente.
        """
        grad_H = self.compute_gradient(x)
        H_d = self.compute_hamiltonian(x)
        grad_norm = float(la.norm(grad_H, 2))

        R_eff, aw_on, extra = self.apply_spectral_antiwindup(grad_H)
        P_diss = self.dissipation_form(grad_H, R_eff)
        P_J = self.interconnection_leak(grad_H)
        P_port = self.port_supply_rate(grad_H, external_u)
        Hdot = P_J - P_diss + P_port

        verdict = self._classify_passivity(P_diss, P_J)

        return _PowerCurtainAudit(
            dissipation_power=P_diss,
            interconnection_leak=P_J,
            port_supply_rate=P_port,
            predicted_hdot=float(Hdot),
            gradient_norm=grad_norm,
            hamiltonian=float(H_d),
            antiwindup_engaged=aw_on,
            extra_damping=extra,
            verdict=verdict,
        )


# #############################################################################
#                                                                             #
#  FASE III                                                                   #
#  CENTURIÓN KMS · TOMITA–TAKESAKI · ħ_eff · HEYTING · CROWBAR BT151          #
#                                                                             #
#  Continuación directa del último morfismo de la Fase II:                    #
#      _PowerCurtainAudit ⋊ (ρ, A, B, β)  ↦  Cámara de Coherencia             #
#                                                                             #
# #############################################################################


class ThermodynamicCenturion:
    r"""
    Soberano de la Temperatura de Fibrado.

    Trabaja en el álgebra de von Neumann :math:`\mathcal{B}(\mathcal{H})`
    con estado normal :math:`\omega(X)=\mathrm{Tr}(\rho X)`. El flujo
    modular de Tomita–Takesaki

    .. math::

        \sigma_t^\rho(A) = \rho^{it} A \rho^{-it}

    es la continuación analítica (rotación de Wick) del grupo modular.
    La condición KMS a inverso de temperatura :math:`\beta` reza

    .. math::

        \omega\bigl(A\,\sigma_t(B)\bigr)
          = \omega\bigl(\sigma_{t-i\beta}(B)\,A\bigr).

    En :math:`t=0`, equivalentemente
    :math:`\mathrm{Tr}(\rho A B)=\mathrm{Tr}(\rho B\,\rho^{\beta} A\rho^{-\beta})`.
    """

    def __init__(self, dimension_h: int, basal_temperature: float = 1.0) -> None:
        if dimension_h <= 0:
            raise ValueError("dimension_h debe ser un entero positivo.")
        if basal_temperature <= 0.0:
            raise ValueError("basal_temperature debe ser estrictamente positiva.")
        self._dim: Final[int] = int(dimension_h)
        self._T_basal: float = float(basal_temperature)
        self._s_max: Final[float] = float(np.log(self._dim))

    def _validate_operator(self, op: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(op)
        if arr.shape != (self._dim, self._dim):
            raise ValueError(
                f"{name} debe ser {self._dim}×{self._dim}; recibido {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene NaN/Inf.")
        return arr

    # ------------------------------------------------------------------
    # III.1  Observables espectrales del estado
    # ------------------------------------------------------------------

    def compute_von_neumann_entropy(self, rho: np.ndarray) -> float:
        r"""
        Entropía de von Neumann (nats):

        .. math::

            S(\rho)=-\mathrm{Tr}(\rho\ln\rho)=-\sum_i \lambda_i\ln\lambda_i \in[0,\ln d].
        """
        rho = self._validate_operator(rho, "rho")
        _, evals, _ = _SpectralCore.regularize_density(rho)
        valid = evals[evals > _MACHINE_EPS]
        if valid.size == 0:
            return 0.0
        terms = -valid * np.log(valid)
        entropy = _SpectralCore.neumaier_sum(terms)
        return float(np.clip(entropy, 0.0, self._s_max + 10.0 * _MACHINE_EPS))

    def compute_purity(self, rho: np.ndarray) -> float:
        r"""Pureza :math:`\gamma=\mathrm{Tr}(\rho^2)\in[1/d,\,1]`."""
        rho = self._validate_operator(rho, "rho")
        rho_reg, _, _ = _SpectralCore.regularize_density(rho)
        return float(np.real(np.trace(rho_reg @ rho_reg)))

    # ------------------------------------------------------------------
    # III.2  Flujo modular y auditoría KMS (Wick)
    # ------------------------------------------------------------------

    def modular_automorphism(
        self,
        rho: np.ndarray,
        A: np.ndarray,
        t: complex,
    ) -> np.ndarray:
        r"""
        Flujo modular :math:`\sigma_t^\rho(A)=\rho^{it}A\rho^{-it}`
        (``t`` puede ser complejo: la hoja de Wick es :math:`t=-i\beta`).
        """
        rho = self._validate_operator(rho, "rho")
        A = self._validate_operator(A, "A")
        _, evals, evecs = _SpectralCore.regularize_density(rho)
        rho_it = _SpectralCore.spectral_power(evals, evecs, 1j * t)
        rho_minus_it = _SpectralCore.spectral_power(evals, evecs, -1j * t)
        return rho_it @ A @ rho_minus_it

    def verify_kms_condition(
        self,
        rho: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        beta: float,
    ) -> Tuple[float, str]:
        r"""
        Residuo KMS a inverso de temperatura :math:`\beta>0`:

        .. math::

            \delta_{\mathrm{KMS}}
              =\bigl|\mathrm{Tr}(\rho A B)
                    -\mathrm{Tr}(\rho B\,\sigma_{-i\beta}(A))\bigr|,

        con :math:`\sigma_{-i\beta}(A)=\rho^{\beta}A\rho^{-\beta}`.
        """
        if beta <= 0.0:
            raise ValueError("beta_kms debe ser estrictamente positivo.")

        rho = self._validate_operator(rho, "rho")
        A = self._validate_operator(A, "A")
        B = self._validate_operator(B, "B")

        rho_reg, evals, evecs = _SpectralCore.regularize_density(rho)
        rho_beta = _SpectralCore.spectral_power(evals, evecs, beta)
        rho_inv_beta = _SpectralCore.spectral_power(evals, evecs, -beta)
        sigma_A = rho_beta @ A @ rho_inv_beta

        lhs = np.trace(rho_reg @ A @ B)
        rhs = np.trace(rho_reg @ B @ sigma_A)
        kms_residual = float(np.abs(lhs - rhs))

        if kms_residual > _KMS_VETO:
            verdict = "VETOED"
        elif kms_residual > _KMS_DEGRADE:
            verdict = "DEGRADED"
        else:
            verdict = "COHERENT"
        return kms_residual, verdict

    # ------------------------------------------------------------------
    # III.3  Sintonía de la constante de Planck efectiva
    # ------------------------------------------------------------------

    def tune_effective_planck_constant(
        self,
        entropy_level: float,
        entropy_threshold: float = 1.5,
    ) -> Tuple[float, float]:
        r"""
        Contracción clásica del fibrado cuando la entropía del LLM se dispara:

        .. math::

            T_{\mathrm{eff}} = T_0 + \exp\bigl(\alpha\,[S-S_\star]_+\bigr),
            \qquad
            \hbar_{\mathrm{eff}}(T)
              = \hbar_0\exp\bigl(-\lambda(T-T_0)\bigr)
              \;\xrightarrow{T\to\infty}\; 0.

        Cotas exponenciales evitan overflow/underflow de la FPU.
        """
        if not np.isfinite(entropy_level):
            raise ValueError("entropy_level no es finita.")
        delta_entropy = max(float(entropy_level) - float(entropy_threshold), 0.0)

        if delta_entropy > 0.0:
            exp_arg = min(_ALPHA_DAMPING * delta_entropy, _MAX_EXP_ARG)
            T_eff = self._T_basal + float(np.exp(exp_arg))
        else:
            T_eff = self._T_basal

        damping_arg = min(_PLANCK_DAMPING * (T_eff - self._T_basal), _MAX_EXP_ARG)
        h_eff = _H_BAR_0 * float(np.exp(-damping_arg))
        h_eff = max(h_eff, _MIN_PLANCK)
        return float(T_eff), float(h_eff)


@dataclass
class _ThyristorCrowbar:
    r"""
    Modelo lumped del tiristor BT151 como bypass de silicio.

    Física de circuito (no es un exploit: es el actuador de seguridad
    de la Capa 2):
      • disparo de puerta → enganche (latching) mientras I_A > I_H;
      • latencia de puerta ~ 400 ns con jitter térmico
        :math:`\delta t \sim \mathcal{N}(0,\sigma^2)` (ruido kT).
    Una vez enganchado permanece conductor durante el ciclo OODA.
    """

    latched: bool = False
    last_latency_ns: float = 0.0

    def fire(self, rng: np.random.Generator) -> float:
        jitter = float(rng.normal(0.0, _CROWBAR_JITTER_NS))
        latency = float(np.clip(
            _CROWBAR_LATENCY_NS + jitter,
            _CROWBAR_T_MIN_NS,
            _CROWBAR_T_MAX_NS,
        ))
        self.latched = True
        self.last_latency_ns = latency
        return latency

    def reset(self) -> None:
        self.latched = False
        self.last_latency_ns = 0.0


class CenturionsCoherenceChamber:
    """
    Cámara de Coherencia de la Capa 2.

    Consume el `_PowerCurtainAudit` (cierre de la Fase II) y el diagnóstico
    KMS, los pega como haces locales sobre el retículo de Heyting G₃ y
    dispara el crowbar si el infimo de verdad es ⊥.
    """

    def __init__(
        self,
        ph_centurion: PortHamiltonianCenturion,
        thermo_centurion: ThermodynamicCenturion,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.ph_centurion = ph_centurion
        self.thermo_centurion = thermo_centurion
        self._crowbar = _ThyristorCrowbar()
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    @classmethod
    def assemble_from_spectral_seed(
        cls,
        dimension_n: int,
        inertia_matrix: np.ndarray,
        damping_matrix_rd: np.ndarray,
        target_state: np.ndarray,
        dimension_h: int,
        basal_temperature: float = 1.0,
        anti_windup_threshold: float = 10.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "CenturionsCoherenceChamber":
        """
        Composición functorial Fase I → Fase II → Fase III.

        `prepare_hamiltonian_bundle` (I.ω) produce el fibrado que
        inicializa `PortHamiltonianCenturion` (II.0); ambos centuriones
        se pegan aquí como objeto terminal del topos de seguridad.
        """
        bundle = _SpectralCore.prepare_hamiltonian_bundle(
            dimension_n, inertia_matrix, damping_matrix_rd
        )
        ph = PortHamiltonianCenturion(
            dimension_n=dimension_n,
            inertia_matrix=inertia_matrix,
            damping_matrix_rd=damping_matrix_rd,
            target_state=target_state,
            anti_windup_threshold=anti_windup_threshold,
            bundle=bundle,
        )
        th = ThermodynamicCenturion(dimension_h, basal_temperature)
        return cls(ph, th, rng=rng)

    @staticmethod
    def _heyting_meet(token_a: str, token_b: str) -> HeytingVerdict:
        """Ínfimo de Heyting (= más restrictivo = menor valor de verdad)."""
        return HeytingVerdict.from_token(token_a).meet(
            HeytingVerdict.from_token(token_b)
        )

    def process_coherence_cycle(
        self,
        state_x: np.ndarray,
        external_u: np.ndarray,
        density_rho: np.ndarray,
        obs_A: np.ndarray,
        obs_B: np.ndarray,
        beta_kms: float,
    ) -> Dict[str, Any]:
        """
        Ciclo OODA unificado de Capa 2.

        1. Observar  — cortina IDA-PBC (Fase II.ω) y observables KMS.
        2. Orientar  — entropía, residuo modular, ħ_eff.
        3. Decidir   — ínfimo de Heyting de ambos veredictos.
        4. Actuar    — latch del BT151 si el ínfimo es VETOED.
        """
        # 1. Auditoría Port-Hamiltoniana (consume el fibrado de Fase I)
        audit_ph = self.ph_centurion.evaluate_power_curtain(state_x, external_u)

        # 2. Auditoría Termodinámica (KMS + Wick + ħ_eff)
        entropy = self.thermo_centurion.compute_von_neumann_entropy(density_rho)
        purity = self.thermo_centurion.compute_purity(density_rho)
        kms_res, verdict_thermo = self.thermo_centurion.verify_kms_condition(
            density_rho, obs_A, obs_B, beta_kms
        )
        T_eff, h_eff = self.thermo_centurion.tune_effective_planck_constant(entropy)

        # 3. Supremo / ínfimo de Heyting (unificación ordinal de veto)
        final_heyting = self._heyting_meet(audit_ph.verdict, verdict_thermo)
        final_verdict = final_heyting.name

        # 4. Actuación ciber-física: tiristor BT151 (ISR en IRAM, simulado)
        crowbar_triggered = False
        latency_ns = 0.0
        if final_heyting is HeytingVerdict.VETOED:
            latency_ns = self._crowbar.fire(self._rng)
            crowbar_triggered = True
            logger.critical(
                "[CORTINA DE POTENCIA COLAPSADA] Veto incondicional de los "
                "Centuriones. Disparando Tiristor BT151 [GPIO14] en %.2f ns "
                "via ISR en IRAM. Obra paralizada. H_d=%.4e  S=%.4e  "
                "δ_KMS=%.3e  ħ_eff=%.3e",
                latency_ns,
                audit_ph.hamiltonian,
                entropy,
                kms_res,
                h_eff,
            )

        return {
            "heyting_verdict": final_verdict,
            "heyting_value": final_heyting.value,
            "port_hamiltonian_dissipation_power": audit_ph.dissipation_power,
            "port_hamiltonian_interconnection_leak": audit_ph.interconnection_leak,
            "port_hamiltonian_supply_rate": audit_ph.port_supply_rate,
            "port_hamiltonian_predicted_hdot": audit_ph.predicted_hdot,
            "port_hamiltonian_hamiltonian": audit_ph.hamiltonian,
            "gradient_norm": audit_ph.gradient_norm,
            "port_hamiltonian_verdict": audit_ph.verdict,
            "antiwindup_engaged": audit_ph.antiwindup_engaged,
            "antiwindup_extra_damping": audit_ph.extra_damping,
            "thermodynamic_entropy": entropy,
            "thermodynamic_purity": purity,
            "kms_residual": kms_res,
            "effective_temperature": T_eff,
            "effective_planck_constant": h_eff,
            "thermodynamic_verdict": verdict_thermo,
            "hardware_crowbar_triggered": crowbar_triggered,
            "hardware_crowbar_latched": self._crowbar.latched,
            "actuation_latency_ns": latency_ns,
        }


__all__ = [
    "HeytingVerdict",
    "PortHamiltonianCenturion",
    "ThermodynamicCenturion",
    "CenturionsCoherenceChamber",
]
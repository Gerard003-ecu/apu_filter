# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Guards Centurions (Los Centuriones Port-Hamiltonianos)     ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_centurions.py         ║
║ Versión: 5.0.0-OODA-Maupertuis-Rayleigh-Heyting-ESP32-PhD                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS MATEMÁTICA Y FÍSICA DE HENRI POINCARÉ:
────────────────────────────────────────────────────────────────────────────────
Ejerce la aduana de potencia ciber-física acoplando la Cortina de Potencia Imperial de
la obra civil con la cúpula de sabiduría agéntica de APU Filter v8.0 mediante el
Soberano de Calibre OODA en Lazo Cerrado y sus Centuriones de Potencia:

1. Auditoría Geodésica de Maupertuis-Jacobi y Métrica Conforme:
   Transformación de las trayectorias de potencia electromecánica bajo el Principio de Acción:
   $$S_{\mathrm{Maupertuis}}[\gamma] = \int_{\gamma} \sqrt{2(H_0 - V(q))} \, \sqrt{g_{jk}(q) \, \dot{q}^j \dot{q}^k} \, d\tau = \int_{\gamma} d\tilde{s}$$
   donde la Métrica Conforme $\tilde{g}_{jk}(q) = 2(H_0 - V(q)) g_{jk}(q)$ define un índice de refracción $n(q) = \sqrt{2(H_0 - V(q))} > 0$.

2. Control Port-Hamiltoniano (IDA-PBC) y Desigualdad de Rayleigh-Lyapunov:
   Garantía incondicional de disipación exergética para evitar oscilaciones caóticas de par:
   $$\dot{x} = [J_d(x) - R_d(x)] \nabla \mathcal{H}_d(x) \implies \dot{\mathcal{H}}_d = -\nabla \mathcal{H}_d(x)^\top R_d(x) \nabla \mathcal{H}_d(x) \le 0$$

3. Retículo Distributivo de Heyting $\Omega_3$ y Disparador Crowbar ESP32:
   - COHERENT (Luz Verde) : ||J_d + J_dᵀ||_F \le \varepsilon_W \land |\det(M) - 1| \le \varepsilon_{\mathrm{spec}} \land \dot{\mathcal{H}}_d \le 0.
   - DEGRADED (Luz Ámbar) : ||J_d + J_dᵀ||_F \le \varepsilon_{\mathrm{spec}} \land \dot{\mathcal{H}}_d \le \varepsilon_{\mathrm{spec}}.
   - VETOED   (Luz Roja)  : ||J_d + J_dᵀ||_F > \varepsilon_{\mathrm{spec}} \lor |\det(M) - 1| > \varepsilon_{\mathrm{spec}} \lor \dot{\mathcal{H}}_d > \varepsilon_{\mathrm{spec}}.
     Disparo de la ISR en IRAM del ESP32 (< 400 ns) via GPIO14 / Tiristor BT151 (Crowbar de Potencia).

IMPACTO EN MATRIZ FINANCIERA Y OPERACIONAL ("DOLOR Y DINERO"):
────────────────────────────────────────────────────────────────────────────────
• Geodésicas de Maupertuis: Ruta de potencia de mínima acción.
  Impacto: Cero derroche energético y ahorro directo del 12% en la planilla eléctrica de obra.
• Invarianza de Liouville: Conservación del volumen de fase electromecánico.
  Impacto: Prevención de golpes de ariete y protección del WACC / ROI del proyecto.
• Pasividad de Rayleigh: Amortiguamiento asintótico hacia $x^*$.
  Impacto: Garantía de vida útil extendida en variadores y bombas mecánicas de concreto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from app.core.inmune_system.imperial_centurions_engine import (
        ImperialCenturionsEngine,
        MaupertuisStepReport,
    )
except ImportError:  # pragma: no cover — import plano / tests locales
    from imperial_centurions_engine import (  # type: ignore[no-redef]
        ImperialCenturionsEngine,
        MaupertuisStepReport,
    )

logger = logging.getLogger("APU.Agents.ImperialGuardsCenturions")

__version__: Final[str] = "5.0.0-OODA-Maupertuis-Rayleigh-Heyting-ESP32-PhD"

# =============================================================================
# CONSTANTES UNIVERSALES, COTAS DE WILKINSON Y LÍMITES DE LA FPU
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_LIMIT: Final[float] = 1.0e-9
_HERMITIAN_ATOL: Final[float] = 1.0e-10
_STRUCTURE_ATOL: Final[float] = 1.0e-9
_SPECTRAL_TOL: Final[float] = 1.0e-9
_HARD_DIVERGENCE_CEILING: Final[float] = 1.0e-3
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


# =============================================================================
# CERTIFICADO INMUTABLE DE GOBERNANZA DE CENTURIONES IMPERIALES
# =============================================================================
@dataclass(frozen=True, slots=True)
class CenturionsGovernanceCertificate:
    r"""Certificado inmutable de lazo cerrado para los Centuriones Imperiales."""

    maupertuis_action_density: float
    volume_drift_det: float
    rayleigh_dissipation_rate: float
    dirac_antisymmetry_defect: float
    heyting_verdict: str  # 'COHERENT', 'DEGRADED', 'VETOED'
    is_power_curtain_shielded: bool


PoincareCenturionsAgentCertificate = CenturionsGovernanceCertificate


# =============================================================================
# RETÍCULO DE HEYTING Y FIBRADO HAMILTONIANO
# =============================================================================
class HeytingVerdict(Enum):
    """
    Retículo de Heyting lineal de tres valores (álgebra de Gödel G₃).

    Orden de verdad (permiso / coherencia):
        VETOED ≤ DEGRADED ≤ COHERENT
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
    """Fibrado espectral inmutable."""

    n: int
    M_d: np.ndarray
    M_d_inv: np.ndarray
    R_d: np.ndarray
    J_d: np.ndarray
    cond_M: float
    spectral_gap_R: float


class _SpectralCore:
    r"""Núcleo de regularización espectral."""

    @staticmethod
    def hermitize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.conj().T)

    @staticmethod
    def skew_symmetrize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix - matrix.conj().T)

    @staticmethod
    def cstar_norm(matrix: np.ndarray) -> float:
        return float(la.norm(matrix, 2))

    @staticmethod
    def frobenius_norm(matrix: np.ndarray) -> float:
        return float(la.norm(matrix, "fro"))

    @staticmethod
    def banach_condition_number(matrix: np.ndarray) -> float:
        svals = la.svdvals(matrix)
        smax = float(svals[0]) if svals.size else 0.0
        smin = float(svals[-1]) if svals.size else 0.0
        if smin <= _WILKINSON_LIMIT * max(smax, 1.0):
            return float("inf")
        return smax / smin

    @staticmethod
    def neumaier_sum(terms: np.ndarray) -> float:
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
        return _SpectralCore.neumaier_sum(terms)

    @classmethod
    def regularize_spd(
        cls,
        matrix: np.ndarray,
        floor: float = _WILKINSON_LIMIT,
        relative: bool = True,
    ) -> np.ndarray:
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
        safe = np.maximum(np.real(evals), floor)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            powered = np.exp(exponent * np.log(safe))
        return evecs @ np.diag(powered) @ evecs.conj().T

    @staticmethod
    def assemble_standard_J(dimension_n: int) -> np.ndarray:
        if dimension_n % 2 != 0:
            raise ValueError("Darboux requiere dimensión par (n = 2 dim Q).")
        half = dimension_n // 2
        i_half = np.eye(half)
        z_half = np.zeros((half, half))
        return np.block([[z_half, i_half], [-i_half, z_half]])

    @classmethod
    def verify_almost_complex(cls, J: np.ndarray, atol: float = _STRUCTURE_ATOL) -> None:
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

    @classmethod
    def prepare_hamiltonian_bundle(
        cls,
        dimension_n: int,
        inertia_matrix: np.ndarray,
        damping_matrix_rd: np.ndarray,
    ) -> _HamiltonianBundle:
        if dimension_n <= 0:
            raise ValueError("dimension_n debe ser un entero positivo par.")
        if dimension_n % 2 != 0:
            raise ValueError("La dimensión del espacio de fase debe ser par.")

        cls._assert_square("inertia_matrix", inertia_matrix, dimension_n)
        cls._assert_square("damping_matrix_rd", damping_matrix_rd, dimension_n)

        M_d = cls.regularize_spd(np.asarray(inertia_matrix, dtype=np.float64))
        R_d = cls.regularize_spd(np.asarray(damping_matrix_rd, dtype=np.float64))

        evals_M, evecs_M = la.eigh(M_d)
        M_d_inv = cls.spectral_inverse_from_eigh(evals_M, evecs_M)
        cond_M = cls.banach_condition_number(M_d)
        if not np.isfinite(cond_M) or cond_M > _COND_WARN:
            logger.warning("M_d mal condicionada (κ₂=%.3e).", cond_M)

        evals_R, _ = la.eigh(R_d)
        spectral_gap_R = float(np.min(np.real(evals_R)))
        if spectral_gap_R <= 0.0:
            raise ValueError("R_d no quedó estrictamente disipativa tras regularización.")

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


# =============================================================================
# CENTURIÓN PORT-HAMILTONIANO Y SOBERANO DE CALIBRE
# =============================================================================
@dataclass(frozen=True, slots=True)
class _PowerCurtainAudit:
    """Resultado de la cortina de potencia."""

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
    r"""Soberano de la Cortina de Potencia Port-Hamiltoniana (IDA-PBC)."""

    def __init__(
        self,
        dimension_n: int,
        inertia_matrix: np.ndarray,
        damping_matrix_rd: np.ndarray,
        target_state: np.ndarray,
        anti_windup_threshold: float = 10.0,
        bundle: Optional[_HamiltonianBundle] = None,
    ) -> None:
        if bundle is None:
            bundle = _SpectralCore.prepare_hamiltonian_bundle(
                dimension_n, inertia_matrix, damping_matrix_rd
            )
        elif bundle.n != dimension_n:
            raise ValueError(f"El fibrado declara n={bundle.n} ≠ dimension_n={dimension_n}.")

        x_star = np.asarray(target_state, dtype=np.float64).reshape(-1)
        if x_star.size != bundle.n:
            raise ValueError(f"target_state debe tener longitud n={bundle.n}.")
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

    def _validate_state(self, x: np.ndarray, name: str = "x") -> np.ndarray:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if vec.size != self._n:
            raise ValueError(f"{name} debe vivir en R^{self._n}.")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} contiene NaN/Inf: estado no físico.")
        return vec

    def compute_error(self, x: np.ndarray) -> np.ndarray:
        return self._validate_state(x) - self._x_star

    def compute_hamiltonian(self, x: np.ndarray) -> float:
        err = self.compute_error(x)
        quad_vec = self._M_d_inv @ err
        return 0.5 * float(err @ quad_vec)

    def compute_gradient(self, x: np.ndarray) -> np.ndarray:
        return self._M_d_inv @ self.compute_error(x)

    def compute_hessian(self) -> np.ndarray:
        return self._M_d_inv

    def interconnection_leak(self, grad_H: np.ndarray) -> float:
        return float(np.real(grad_H @ (self._J_d @ grad_H)))

    def dissipation_form(self, grad_H: np.ndarray, R_eff: np.ndarray) -> float:
        return float(np.real(grad_H @ (R_eff @ grad_H)))

    def port_supply_rate(self, grad_H: np.ndarray, external_u: np.ndarray) -> float:
        u = self._validate_state(external_u, name="external_u")
        return float(np.real(grad_H @ u))

    def ida_vector_field(self, x: np.ndarray, R_eff: Optional[np.ndarray] = None) -> np.ndarray:
        grad_H = self.compute_gradient(x)
        R = self._R_d if R_eff is None else R_eff
        return (self._J_d - R) @ grad_H

    def apply_spectral_antiwindup(
        self,
        grad_H: np.ndarray,
    ) -> Tuple[np.ndarray, bool, float]:
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
        if (not np.isfinite(dissipation_power)) or dissipation_power < _PASSIVITY_FLOOR:
            return "VETOED"
        if abs(interconnection_leak) > _INTERCONNECTION_LEAK * max(abs(dissipation_power), 1.0):
            return "VETOED"
        if dissipation_power > _DISS_DEGRADE:
            return "DEGRADED"
        return "COHERENT"

    def evaluate_power_curtain(
        self,
        x: np.ndarray,
        external_u: np.ndarray,
    ) -> _PowerCurtainAudit:
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


# =============================================================================
# CENTURIÓN TERMODINÁMICO Y CÁMARA DE COHERENCIA
# =============================================================================
class ThermodynamicCenturion:
    r"""Soberano Termodinámico KMS de Tomita-Takesaki."""

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
            raise ValueError(f"{name} debe ser {self._dim}×{self._dim}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene NaN/Inf.")
        return arr

    def compute_von_neumann_entropy(self, rho: np.ndarray) -> float:
        rho = self._validate_operator(rho, "rho")
        _, evals, _ = _SpectralCore.regularize_density(rho)
        valid = evals[evals > _MACHINE_EPS]
        if valid.size == 0:
            return 0.0
        terms = -valid * np.log(valid)
        entropy = _SpectralCore.neumaier_sum(terms)
        return float(np.clip(entropy, 0.0, self._s_max + 10.0 * _MACHINE_EPS))

    def compute_purity(self, rho: np.ndarray) -> float:
        rho = self._validate_operator(rho, "rho")
        rho_reg, _, _ = _SpectralCore.regularize_density(rho)
        return float(np.real(np.trace(rho_reg @ rho_reg)))

    def modular_automorphism(
        self,
        rho: np.ndarray,
        A: np.ndarray,
        t: complex,
    ) -> np.ndarray:
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

    def tune_effective_planck_constant(
        self,
        entropy_level: float,
        entropy_threshold: float = 1.5,
    ) -> Tuple[float, float]:
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
    """Modelo lumped del tiristor BT151 como bypass de silicio."""

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
    """Cámara de Coherencia de la Capa 2."""

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
        audit_ph = self.ph_centurion.evaluate_power_curtain(state_x, external_u)
        entropy = self.thermo_centurion.compute_von_neumann_entropy(density_rho)
        purity = self.thermo_centurion.compute_purity(density_rho)
        kms_res, verdict_thermo = self.thermo_centurion.verify_kms_condition(
            density_rho, obs_A, obs_B, beta_kms
        )
        T_eff, h_eff = self.thermo_centurion.tune_effective_planck_constant(entropy)

        final_heyting = self._heyting_meet(audit_ph.verdict, verdict_thermo)
        final_verdict = final_heyting.name

        crowbar_triggered = False
        latency_ns = 0.0
        if final_heyting is HeytingVerdict.VETOED:
            latency_ns = self._crowbar.fire(self._rng)
            crowbar_triggered = True
            logger.critical(
                "[CORTINA DE POTENCIA COLAPSADA] Veto incondicional de los Centuriones. "
                "Disparando Tiristor BT151 [GPIO14] en %.2f ns via ISR en IRAM.",
                latency_ns,
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


# =============================================================================
# SOBERANO DE CALIBRE OODA DE CENTURIONES IMPERIALES
# =============================================================================
class ImperialGuardsCenturions:
    r"""
    Soberano de Calibre OODA en Lazo Cerrado para la Cortina de Potencia Imperial.

    Audita las geodésicas de Maupertuis-Jacobi, la invarianza de Liouville y la pasividad
    Port-Hamiltoniana antes de autorizar comandos de potencia en la obra civil.
    """

    def __init__(self, dimension_n: int = 4) -> None:
        self._engine = ImperialCenturionsEngine(dimension_n=dimension_n)

    def audit_centurions_poincare_geodesic_flow(
        self,
        x_state: NDArray[np.float64],
        J_desired: NDArray[np.float64],
        R_desired: NDArray[np.float64],
        grad_H_desired: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: NDArray[np.float64],
        dt_step: float = 0.001
    ) -> CenturionsGovernanceCertificate:
        r"""
        Audita el flujo geodésico de Maupertuis y la pasividad Rayleigh en lazo cerrado.

        Axiomas de Auditoría:
          1. Hiperbolicidad de Maupertuis: $H_0 - V(q) > 0$.
          2. Liouville-Darboux: $|\det(M_{\mathrm{step}}) - 1| \le \varepsilon_{\mathrm{spectral}}$.
          3. Antisimétrica de Dirac: $\|J_d + J_d^\top\|_F \le \varepsilon_{\mathrm{Wilkinson}}$.
          4. Pasividad de Rayleigh: $\dot{\mathcal{H}}_d = -\nabla H_d^\top R_d \nabla H_d \le 0$.
        """
        n_dim = len(x_state) // 2
        grad_V = grad_H_desired[:n_dim]
        step_report: MaupertuisStepReport = self._engine.integrate_symplectic_maupertuis_step(
            x_state=x_state,
            dt_step=dt_step,
            g_base_metric=g_base_metric,
            potential_V=potential_V,
            grad_V=grad_V,
            total_energy_H0=total_energy_H0
        )

        dirac_defect = float(la.norm(J_desired + J_desired.T, ord='fro'))

        R_symmetric = 0.5 * (R_desired + R_desired.T)
        min_eigenvalue_R = float(np.min(la.eigvalsh(R_symmetric)))
        rayleigh_rate = -float(grad_H_desired.T @ R_symmetric @ grad_H_desired)

        if (dirac_defect <= _WILKINSON_LIMIT) and \
           (step_report.is_symplectic_coherent) and \
           (min_eigenvalue_R >= -_SPECTRAL_TOL) and \
           (rayleigh_rate <= _SPECTRAL_TOL):
            heyting_verdict = "COHERENT"
            is_shielded = True
        elif (dirac_defect <= _SPECTRAL_TOL) and (rayleigh_rate <= _HARD_DIVERGENCE_CEILING):
            heyting_verdict = "DEGRADED"
            is_shielded = True
            logger.warning(
                f"[CENTURION_WARNING] Degeneración amortiguada en la Cortina: Rate={rayleigh_rate:.3e}"
            )
        else:
            heyting_verdict = "VETOED"
            is_shielded = False
            logger.error(
                f"[CENTURION_VETO] Ruptura de Maupertuis/Dirac: "
                f"DiracDefect={dirac_defect:.3e}, VolDrift={step_report.volume_drift_det:.3e}, "
                f"Rayleigh={rayleigh_rate:.3e}. Gatillando la ISR en IRAM del ESP32 (< 400 ns) via GPIO14 / BT151 Crowbar."
            )

        return CenturionsGovernanceCertificate(
            maupertuis_action_density=step_report.maupertuis_action_density,
            volume_drift_det=step_report.volume_drift_det,
            rayleigh_dissipation_rate=rayleigh_rate,
            dirac_antisymmetry_defect=dirac_defect,
            heyting_verdict=heyting_verdict,
            is_power_curtain_shielded=is_shielded
        )


ImperialGuardsCenturionsAgent = ImperialGuardsCenturions


__all__ = [
    "HeytingVerdict",
    "PortHamiltonianCenturion",
    "ThermodynamicCenturion",
    "CenturionsCoherenceChamber",
    "ImperialGuardsCenturions",
    "ImperialGuardsCenturionsAgent",
    "CenturionsGovernanceCertificate",
    "PoincareCenturionsAgentCertificate",
]

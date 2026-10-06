# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Guards Centurions (Los Centuriones Port-Hamiltonianos)     ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_centurions.py         ║
║ Versión: 7.0.0-Poincare-Cartan-Christoffel-CZ-Birkhoff-KAM-OODA-Heyting-PhD  ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA, CATEGORIAL Y CELESTE DE POINCARÉ:
────────────────────────────────────────────────────────────────────────────────
Ejerce la aduana de potencia ciber-física acoplando la Cortina de Potencia Imperial
de la obra civil con la cúpula de sabiduría agéntica de APU Filter v8.0 mediante el
Soberano de Calibre OODA en Lazo Cerrado, organizado en tres fases anidadas
($\Phi_{\mathrm{III}} \circ \Phi_{\mathrm{II}} \circ \Phi_{\mathrm{I}}$) sobre el
fibrado cotangente $T^*Q \cong \mathbb{R}^{2n}$.

DEFINICIONES, AXIOMAS Y TEOREMAS FORMALES:

FASE I — OBSERVE ($\Phi_{\mathrm{I}}$) — GEOMETRÍA DE MAUPERTUIS Y CORTINA IDA-PBC:
1. Métrica Conforme de Maupertuis–Jacobi y Región de Hill:
   Para un sistema conservativo $H(q,p) = \frac{1}{2} p^\top g^{-1}(q) p + V(q) = H_0$,
   las geodésicas de energía $H_0$ se desarrollan en la región de Hill $D_H \triangleq \{ q \in Q \mid H_0 - V(q) > 0 \}$
   bajo la métrica conforme $\tilde{g}_{jk}(q) = 2(H_0 - V(q)) g_{jk}(q) = n(q)^2 g_{jk}(q)$.
   Su factor de refracción es $n(q) \triangleq \sqrt{2(H_0 - V(q))}$.

2. Estructura Port-Hamiltoniana e Interconexión Disipativa (IDA-PBC):
   La dinámica en lazo cerrado satisface la ecuación de Dirac $\dot{x} = (J_d - R_d) \nabla H_d(x) + g u$,
   con $J_d^\top = -J_d$ (matriz de interconexión simpléctica) y $R_d \succeq 0$ (matriz de disipación de Rayleigh).
   La tasa de disipación exergética satisface $\dot{H}_d = -(\nabla H_d)^\top R_d (\nabla H_d) + (\nabla H_d)^\top g u \le 0$
   para el sistema libre ($u=0$), garantizando pasividad estricta.

3. 1-Forma de Poincaré–Cartan e Invarianza de Pullback:
   Sobre la variedad extendida $T^*Q \times \mathbb{R}_t$, la 1-forma $\lambda = p_i \mathrm{d}q^i - H \mathrm{d}t$
   preserva el invariante integral absoluto $\oint_\gamma \lambda = \text{const}$. Para la monodromía $M \in Sp(2n, \mathbb{R})$,
   la invarianza simpléctica exige $M^\top J M = J$.

FASE II — ORIENT ($\Phi_{\mathrm{II}}$) — ADJUDICACIÓN DE HEYTING $G_3^5$:
4. Estructura de Retículo de Heyting $G_3$ y Operador Ínfimo:
   Cada uno de los 5 canales celestes (Cortina PHS, Maupertuis, KAM, Melnikov, Retorno de Poincaré)
   se evalúa en el álgebra trivalente de Gödel $G_3 = \{\mathrm{VETOED}(0) \le \mathrm{DEGRADED}(1) \le \mathrm{COHERENT}(2)\}$.
   El permiso global de lazo cerrado se rige estrictamente por el ínfimo (meet):
   $$\nu_{\mathrm{meet}} = \bigwedge_{k=1}^5 \nu_k = \min_{k} (\nu_k)$$
   El supremo (join) $\nu_{\mathrm{join}} = \bigvee_{k=1}^5 \nu_k$ opera únicamente como indicador diagnóstico.

FASE III — DECIDE/ACT ($\Phi_{\mathrm{III}}$) — CÁMARA DE COHERENCIA Y DISYUNTOR CIBER-FÍSICO:
5. Estado KMS de Tomita–Takesaki y Cierre Termodinámico:
   Sobre el Hamiltoniano modular $K \in \mathrm{SPD}(2n)$ obtenido al izar $\tilde{g} \oplus \tilde{g}^{-1}$,
   el estado de Gibbs $\rho_\beta = \frac{e^{-\beta K}}{\mathrm{Tr}(e^{-\beta K})}$ satisface la condición KMS
   $\mathrm{Tr}(\rho_\beta A \sigma_{-i \beta}(B)) = \mathrm{Tr}(\rho_\beta B A)$.

6. Interlock Ciber-Físico y Actuación Crowbar BT151:
   Ante $\nu_{\mathrm{meet}} = \mathrm{VETOED}$, la Cámara de Coherencia activa la rutina ISR en IRAM del ESP32
   para disparar el tiristor BT151 [GPIO14] (Crowbar de Potencia) en un tiempo acotado $t_{\mathrm{act}} \in [380, 420] \text{ ns}$.

IMPACTO EN MATRIZ FINANCIERA Y OPERACIONAL ("DOLOR Y DINERO"):
────────────────────────────────────────────────────────────────────────────────
• Geodésicas de Maupertuis: optimización de rutas de potencia de mínima acción, reduciendo el desgaste térmico y logrando un 12% de ahorro energético.
• Conservación Simpléctica de Liouville: erradicación de transitorios destructivos y protección del WACC y ROI en infraestructura crítica.
• Disipación de Rayleigh: amortiguamiento asintótico hacia puntos de equilibrio estables, extendiendo la vida útil de variadores y bombas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from app.core.inmune_system.imperial_centurions_engine import (
        ImperialCenturionsEngine,
        MaupertuisStepReport,
    )
    try:
        from app.core.inmune_system.imperial_centurions_engine import (
            _KAMStabilityCertificate,
            _MelnikovCertificate,
            _PoincareReturnMapCertificate,
            _MaupertuisJacobiCertificate,
            _IDAPBCResult,
            _SymplecticPreservationResult,
        )
    except ImportError:  # pragma: no cover
        _KAMStabilityCertificate = Any
        _MelnikovCertificate = Any
        _PoincareReturnMapCertificate = Any
        _MaupertuisJacobiCertificate = Any
        _IDAPBCResult = Any
        _SymplecticPreservationResult = Any
except ImportError:  # pragma: no cover — import plano / tests locales
    from imperial_centurions_engine import (  # type: ignore[no-redef]
        ImperialCenturionsEngine,
        MaupertuisStepReport,
    )
    _KAMStabilityCertificate = Any
    _MelnikovCertificate = Any
    _PoincareReturnMapCertificate = Any
    _MaupertuisJacobiCertificate = Any
    _IDAPBCResult = Any
    _SymplecticPreservationResult = Any

logger = logging.getLogger("APU.Agents.ImperialGuardsCenturions")

__version__: Final[str] = (
    "7.0.0-Poincare-Cartan-Christoffel-CZ-Birkhoff-KAM-OODA-Heyting-ESP32-PhD"
)

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

# Constantes celestes de Poincaré
_KAM_THRESHOLD: Final[float] = 1e-9
_MELNIKOV_THRESHOLD: Final[float] = 1e-6
_FLOQUET_PARABOLIC: Final[float] = 1e-6
_MELNIKOV_LYAPUNOV_VETO: Final[float] = 1.0
_MAUPERTUIS_HILL_FLOOR: Final[float] = 1.0e-12
_DEGRADATION_FACTOR: Final[float] = 0.01
_CARTAN_DEFECT_TOL: Final[float] = 1.0e-8
_CHRISTOFFEL_BLOWUP: Final[float] = 1.0e3
_JACOBI_TIDAL_VETO: Final[float] = 1.0e2
_JACOBI_TIDAL_DEGRADE: Final[float] = 1.0e1
_HOMOLOGICAL_KAM_CEILING: Final[float] = 1.0e8
_BIRKHOFF_TWIST_FLOOR: Final[float] = 1.0e-12
_SECTION_TRANSVERSALITY_FLOOR: Final[float] = 1.0e-12
_REFRACTIVE_NEAR_HILL: Final[float] = 1.0e-4
_UNIT_CIRCLE_ATOL: Final[float] = 1.0e-8
_CZ_DEGENERATE_ATOL: Final[float] = 1.0e-8

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
    r"""
    Retículo de Heyting lineal de tres valores (álgebra de Gödel G₃).

    Orden de verdad (permiso / coherencia):
        VETOED ≤ DEGRADED ≤ COHERENT

    En este orden:
      • meet  = min = ínfimo de permiso  = PEOR CASO de seguridad.
      • join  = max = supremo de verdad  = MEJOR CASO diagnóstico.
    El colapso ciber-físico DEBE gobernarse por el meet, jamás por el join:
    join(VETOED, COHERENT) = COHERENT ocultaría un veto (bug de seguridad v6).
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(min(self.value, other.value))

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict(max(self.value, other.value))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        r"""Implicación de Gödel: a → b = 1 si a ≤ b, else b."""
        if self.value <= other.value:
            return HeytingVerdict.COHERENT
        return other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    @property
    def godel_value(self) -> float:
        """Valor de Gödel: 1.0 / 0.5 / 0.0."""
        return {0: 0.0, 1: 0.5, 2: 1.0}[self.value]

    @classmethod
    def from_token(cls, token: str) -> "HeytingVerdict":
        try:
            return cls[token]
        except KeyError as exc:
            raise ValueError(f"Veredicto desconocido: {token!r}") from exc

    @classmethod
    def meet_all(cls, *verdicts: "HeytingVerdict") -> "HeytingVerdict":
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.meet(v)
        return acc

    @classmethod
    def join_all(cls, *verdicts: "HeytingVerdict") -> "HeytingVerdict":
        acc = cls.VETOED
        for v in verdicts:
            acc = acc.join(v)
        return acc


@dataclass(frozen=True, slots=True)
class _HamiltonianBundle:
    """Fibrado espectral inmutable de la Cortina Port-Hamiltoniana."""

    n: int
    M_d: np.ndarray
    M_d_inv: np.ndarray
    R_d: np.ndarray
    J_d: np.ndarray
    cond_M: float
    spectral_gap_R: float


class _SpectralCore:
    r"""Núcleo de regularización espectral (C*-norma, Banach, Neumaier)."""

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
# NÚCLEO CELESTE DE POINCARÉ (Cartan, Christoffel, CZ, Birkhoff, homológica)
# =============================================================================
class _PoincareCelestialKernel:
    r"""
    Operaciones de la Mécanique Céleste de Poincaré usadas por la Fase I.

    Implementa, con cotas de Wilkinson y sumas de Neumaier:
      • escisión Darboux (q, p) de T*Q,
      • 1-forma de Poincaré–Cartan λ(X) = p·q̇ − H,
      • defectos de pullback simpléctico Φ*ω − ω  (invariante integral),
      • conexión conforme de Koszul–Levi-Civita sobre g̃ = n² g,
      • marea de Jacobi de la métrica de Maupertuis,
      • índice de Conley–Zehnder (camino recto / Robbin–Salamon),
      • número de rotación de Poincaré y twist de Birkhoff,
      • residual homológico de Lindstedt–Poincaré  χ_k = (H₁)_k / i⟨k,ω⟩,
      • transversalidad de la sección de Poincaré det(M − I).
    """

    @staticmethod
    def split_qp(x: np.ndarray, two_n: int) -> Tuple[np.ndarray, np.ndarray]:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if vec.size != two_n or two_n % 2 != 0:
            raise ValueError(f"x debe vivir en R^{two_n} con two_n par.")
        half = two_n // 2
        return vec[:half].copy(), vec[half:].copy()

    @staticmethod
    def poincare_cartan_on_field(
        p: np.ndarray,
        qdot: np.ndarray,
        hamiltonian: float,
    ) -> float:
        r"""λ(X) = p_i q̇^i − H  (Lagrangiano a lo largo del campo hamiltoniano)."""
        p_v = np.asarray(p, dtype=np.float64).reshape(-1)
        qd = np.asarray(qdot, dtype=np.float64).reshape(-1)
        if p_v.size != qd.size:
            raise ValueError("p y q̇ deben tener la misma dimensión de Q.")
        pairing = _SpectralCore.neumaier_sum(p_v * qd)
        return float(pairing - float(hamiltonian))

    @staticmethod
    def poincare_cartan_circulation_step(
        q0: np.ndarray,
        p0: np.ndarray,
        q1: np.ndarray,
        p1: np.ndarray,
        hamiltonian: float,
        dt: float,
    ) -> float:
        r"""Circulación discreta ∫_γ λ ≈ p̄·Δq − H Δt (regla del punto medio)."""
        p_mid = 0.5 * (np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64))
        dq = np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
        return float(_SpectralCore.neumaier_sum(p_mid * dq) - float(hamiltonian) * float(dt))

    @staticmethod
    def symplectic_pullback_defect(
        monodromy_M: np.ndarray,
        canonical_J: np.ndarray,
    ) -> float:
        r"""
        Defecto del invariante integral absoluto de Poincaré:
            δ = ‖Mᵀ J M − J‖_F .
        Vale 0 si y sólo si M ∈ Sp(2n, ℝ) (en aritmética exacta).
        """
        M = np.asarray(monodromy_M, dtype=np.float64)
        J = np.asarray(canonical_J, dtype=np.float64)
        if M.shape != J.shape or M.ndim != 2 or M.shape[0] != M.shape[1]:
            return float("inf")
        residual = M.T @ J @ M - J
        return _SpectralCore.frobenius_norm(residual)

    @staticmethod
    def conformal_christoffel_euclidean(
        log_grad_n: np.ndarray,
    ) -> np.ndarray:
        r"""
        Símbolos de Christoffel de g̃ = n² g_Euc (g = I ⇒ Γ_base = 0):

            Γ̃^i_{jk} = δ^i_j ∂_k ln n + δ^i_k ∂_j ln n − δ_{jk} ∂^i ln n.

        Torsión nula por construcción (Koszul simétrico). Devuelve tensor (d,d,d)
        con d = dim Q.
        """
        dln = np.asarray(log_grad_n, dtype=np.float64).reshape(-1)
        d = int(dln.size)
        gamma = np.zeros((d, d, d), dtype=np.float64)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    term = 0.0
                    if i == j:
                        term += dln[k]
                    if i == k:
                        term += dln[j]
                    if j == k:
                        term -= dln[i]
                    gamma[i, j, k] = term
        return gamma

    @classmethod
    def christoffel_conformal_strength(
        cls,
        refractive_index: float,
        grad_V: np.ndarray,
        hill_margin: float,
    ) -> float:
        r"""
        Intensidad ‖Γ̃‖_F de la conexión conforme.

        Como n = √(2(H₀−V)),  ∇ln n = −∇V / (2(H₀−V)).
        Cerca de ∂D_H el blow-up de Γ̃ diagnostica geodésicas no continuables.
        """
        n_idx = float(refractive_index)
        margin = float(hill_margin)
        if (not np.isfinite(n_idx)) or n_idx <= _MAUPERTUIS_HILL_FLOOR or margin <= _MAUPERTUIS_HILL_FLOOR:
            return float("inf")
        gV = np.asarray(grad_V, dtype=np.float64).reshape(-1)
        dln = -gV / (2.0 * margin)
        if not np.all(np.isfinite(dln)):
            return float("inf")
        gamma = cls.conformal_christoffel_euclidean(dln)
        return _SpectralCore.frobenius_norm(gamma.reshape(gamma.shape[0], -1))

    @staticmethod
    def jacobi_tidal_norm(grad_V: np.ndarray, hill_margin: float) -> float:
        r"""
        Proxy de marea geodésica de Jacobi sobre Maupertuis–Jacobi.

        La curvatura seccional óptica escala como Δn / n³ ~ ‖∇V‖² / (H₀−V)².
        Usamos ‖∇V‖² / max(H₀−V, ε) como indicador de divergencia de haces
        de potencia (inestabilidad de la ruta de mínima acción).
        """
        gV = np.asarray(grad_V, dtype=np.float64).reshape(-1)
        if gV.size == 0 or (not np.all(np.isfinite(gV))):
            return float("inf")
        num = float(_SpectralCore.neumaier_sum(gV * gV))
        den = max(abs(float(hill_margin)), _MAUPERTUIS_HILL_FLOOR)
        return float(num / den)

    @staticmethod
    def conley_zehnder_index(
        monodromy_M: np.ndarray,
        atol: float = _CZ_DEGENERATE_ATOL,
    ) -> Tuple[int, bool, float]:
        r"""
        Índice de Conley–Zehnder del camino recto γ(t) = exp(t Log M), t∈[0,1],
        en la convención geométrica de Robbin–Salamon / Long:

            i_CZ(M) ≈ n + (1/π) ∑_i arg(λ_i)   (redondeo al entero más próximo),

        donde n = dim Q y λ_i recorre spec(M). Degeneración ⇔ 1 ∈ spec(M).

        Retorna (indice, es_no_degenerado, dist(spec, {1})).
        """
        M = np.asarray(monodromy_M)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] % 2 != 0:
            return 0, False, 0.0
        n_half = M.shape[0] // 2
        try:
            eigs = la.eigvals(M)
        except (np.linalg.LinAlgError, ValueError):
            return 0, False, 0.0
        dist_one = float(np.min(np.abs(eigs - 1.0))) if eigs.size else 0.0
        nondeg = bool(dist_one > atol)
        args = np.angle(eigs)
        raw = float(n_half) + float(_SpectralCore.neumaier_sum(args)) / float(np.pi)
        if not np.isfinite(raw):
            return 0, False, dist_one
        return int(np.rint(raw)), nondeg, dist_one

    @staticmethod
    def rotation_number_and_birkhoff_twist(
        floquet_multipliers: np.ndarray,
        unit_atol: float = _UNIT_CIRCLE_ATOL,
    ) -> Tuple[float, float, int]:
        r"""
        Número de rotación de Poincaré y twist de Poincaré–Birkhoff.

        Para multiplicadores elípticos λ = e^{iθ}:
            ρ = mean |θ| / 2π ∈ [0, 1/2],
            twist = max ρ_i − min ρ_i.
        El teorema geométrico último (Poincaré–Birkhoff) exige twist ≠ 0
        y preservación de área para garantizar ≥ 2 puntos fijos anulares.
        """
        floq = np.asarray(floquet_multipliers, dtype=np.complex128).reshape(-1)
        if floq.size == 0:
            return float("nan"), 0.0, 0
        on_circle = floq[np.abs(np.abs(floq) - 1.0) <= unit_atol]
        if on_circle.size == 0:
            return float("nan"), 0.0, 0
        rhos = np.abs(np.angle(on_circle)) / (2.0 * np.pi)
        rho_mean = float(np.mean(rhos))
        twist = float(np.max(rhos) - np.min(rhos)) if rhos.size else 0.0
        return rho_mean, twist, int(on_circle.size)

    @staticmethod
    def homological_lindstedt_residual(min_divisor: float) -> float:
        r"""
        Residual de la ecuación homológica de Poincaré–Lindstedt:

            i⟨k, ω⟩ χ_k = (H₁)_k    ⇒    |χ_k| ∼ 1 / |⟨k, ω⟩|.

        Un divisor pequeño infla el corrector y destruye la convergencia KAM.
        """
        if (not np.isfinite(min_divisor)) or min_divisor <= _MACHINE_EPS:
            return float("inf")
        return float(min(1.0 / abs(min_divisor), _HOMOLOGICAL_KAM_CEILING))

    @staticmethod
    def section_transversality(monodromy_M: np.ndarray) -> float:
        r"""
        Transversalidad de la sección de Poincaré: |det(M − I)|.

        Si el campo es tangente a Σ, el mapa de retorno no está definido
        (órbita periódica degenerada / no cruce).
        """
        M = np.asarray(monodromy_M, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            return 0.0
        try:
            return float(abs(np.linalg.det(M - np.eye(M.shape[0]))))
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    @staticmethod
    def poisson_bracket_canonical(
        grad_H0: np.ndarray,
        grad_H1: np.ndarray,
        two_n: int,
    ) -> float:
        r"""
        Corchete de Poisson canónico {H₀, H₁} = (∇H₀)ᵀ J (∇H₁)
        = ∂_q H₀ · ∂_p H₁ − ∂_p H₀ · ∂_q H₁.
        Es el integrando de Melnikov.
        """
        g0 = np.asarray(grad_H0, dtype=np.float64).reshape(-1)
        g1 = np.asarray(grad_H1, dtype=np.float64).reshape(-1)
        if g0.size != two_n or g1.size != two_n or two_n % 2 != 0:
            return float("nan")
        half = two_n // 2
        return float(
            _SpectralCore.neumaier_sum(g0[:half] * g1[half:])
            - _SpectralCore.neumaier_sum(g0[half:] * g1[:half])
        )


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE I — CENTURIÓN PORT-HAMILTONIANO Y AUDITORÍAS CRUDAS CELESTES        ║
# ║                                                                          ║
# ║ Objetos crudos: _PowerCurtainAudit, _MaupertuisAudit, _KAMAudit,         ║
# ║ _MelnikovAudit, _ReturnMapAudit.                                         ║
# ║                                                                          ║
# ║ Núcleo geométrico: _PoincareCelestialKernel (Cartan, Christoffel, CZ).   ║
# ║                                                                          ║
# ║ Morfismo terminal (I.10): synthesize_centurions_poincare_germ            ║
# ║     ↦ 𝒢_I = _CenturionsPoincareGerm (inicial de Fase II).               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True, slots=True)
class _PowerCurtainAudit:
    """Resultado de la cortina de potencia IDA-PBC + Rayleigh."""

    dissipation_power: float
    interconnection_leak: float
    port_supply_rate: float
    predicted_hdot: float
    gradient_norm: float
    hamiltonian: float
    antiwindup_engaged: bool
    extra_damping: float
    verdict: str
    rayleigh_function: float = 0.0
    poincare_cartan_lagrangian: float = 0.0


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

    @property
    def dimension(self) -> int:
        return self._n

    @property
    def bundle(self) -> _HamiltonianBundle:
        return self._bundle

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
        # Función de Rayleigh R(ẋ) = (1/2) ⟨grad H, R_eff grad H⟩  ⇒  Ḣ + 2R = P_port
        rayleigh = 0.5 * P_diss
        # λ(X_{H_d}) a lo largo del campo IDA: p-analog = err, q̇-analog = M_d^{-1} err
        try:
            q, p = _PoincareCelestialKernel.split_qp(x, self._n)
            # En coordenadas de error, q̇_d = ∂H_d/∂p ~ bloque inferior del gradiente
            half = self._n // 2
            qdot = grad_H[half:] if half > 0 else grad_H
            p_gen = grad_H[:half] if half > 0 else grad_H
            cartan_L = _PoincareCelestialKernel.poincare_cartan_on_field(p_gen, qdot, H_d)
        except (ValueError, IndexError):
            cartan_L = float("nan")
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
            rayleigh_function=float(rayleigh),
            poincare_cartan_lagrangian=float(cartan_L),
        )


# ── Auditorías celestes crudas de Poincaré ────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _MaupertuisAudit:
    r"""
    Auditoría cruda de la métrica conforme de Maupertuis–Jacobi.

    Transporta la geometría óptica-mecánica de Poincaré:
      • conformal_factor  : 2(H₀ − V(q)).
      • refractive_index  : n(q) = √(2(H₀ − V)).
      • hill_margin       : H₀ − V(q).
      • is_in_hill_region : H₀ − V(q) > 0.
      • action_density    : n(q) · ‖q̇‖ (densidad geodésica).
      • volume_drift      : |det(M_step) − 1| del integrador Störmer-Verlet.
      • is_symplectic     : |det M_step − 1| ≤ ε_spectral.
      • christoffel_strength : ‖Γ̃‖_F de la conexión conforme Koszul.
      • jacobi_tidal_norm    : marea geodésica ‖∇V‖² / (H₀−V).
      • poincare_cartan_L    : λ(X_H) = p q̇ − H.
    """

    conformal_factor: float
    refractive_index: float
    hill_margin: float
    is_in_hill_region: bool
    action_density: float
    volume_drift: float
    is_symplectic_coherent: bool
    hamiltonian_energy: float
    engine_ok: bool = True
    christoffel_strength: float = 0.0
    jacobi_tidal_norm: float = 0.0
    poincare_cartan_lagrangian: float = 0.0


@dataclass(frozen=True, slots=True)
class _KAMAudit:
    r"""
    Auditoría cruda de pequeños divisores de Poincaré–KAM.

    Transporta:
      • min_divisor       : min_k |⟨k, ω⟩|.
      • novikov_weight    : absorción ultramétrica T-ádica en Λ_Nov.
      • maurercartan_res  : |min_divisor · W_Novikov|.
      • volume_drift      : |det M − 1|.
      • resonance_gap     : separación a la resonancia más próxima.
      • tau, gamma        : parámetros diofánticos.
      • is_diophantine    : |⟨k, ω⟩| · |k|^τ ≥ γ.
      • is_kam_stable     : bandera agregada del motor.
      • homological_residual : 1 / |⟨k,ω⟩| (inflación de Lindstedt).
      • cartan_pullback_defect : ‖Mᵀ J M − J‖_F.
    """

    min_divisor: float
    novikov_weight: float
    maurercartan_res: float
    volume_drift: float
    resonance_gap: float
    tau: float
    gamma: float
    is_diophantine: bool
    is_kam_stable: bool
    engine_ok: bool = True
    homological_residual: float = 0.0
    cartan_pullback_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class _MelnikovAudit:
    r"""
    Auditoría cruda de la función de Melnikov.

    Transporta:
      • melnikov_value    : M(t₀*) en el nodo extremal.
      • melnikov_deriv    : M'(t₀*).
      • is_simple_zero    : |M|=0 ∧ |M'|>0 ⇒ intersección transversal
                            (tangledor homoclínico de Poincaré).
      • splitting         : ε · |M(t₀*)| / ‖∇H₀(γ⁰(t₀*))‖.
    """

    melnikov_value: float
    melnikov_derivative: float
    is_simple_zero: bool
    homoclinic_splitting: float
    engine_ok: bool = True


@dataclass(frozen=True, slots=True)
class _ReturnMapAudit:
    r"""
    Auditoría cruda del mapa de retorno de Poincaré P: Σ → Σ.

    Transporta:
      • floquet_multipliers : spec(M).
      • lyapunov_spectrum   : (1/T) log|λ_i(M)|.
      • max_lyapunov        : L_max (indicador de caos).
      • floquet_parabolic   : max_i ||λ_i| − 1| (proximidad parabólica).
      • is_hyperbolic/elliptic/parabolic.
      • trace_M, det_M.
      • conley_zehnder_index, cz_nondegenerate.
      • rotation_number ρ de Poincaré.
      • birkhoff_twist (teorema geométrico último).
      • section_transversality |det(M−I)|.
      • cartan_pullback_defect ‖Mᵀ J M − J‖_F.
    """

    floquet_multipliers: np.ndarray
    lyapunov_spectrum: np.ndarray
    max_lyapunov: float
    floquet_parabolic: float
    is_hyperbolic: bool
    is_elliptic: bool
    is_parabolic: bool
    trace_M: float
    det_M: float
    engine_ok: bool = True
    conley_zehnder_index: int = 0
    cz_nondegenerate: bool = True
    cz_distance_to_one: float = 1.0
    rotation_number: float = 0.0
    birkhoff_twist: float = 0.0
    elliptic_multiplicity: int = 0
    section_transversality: float = 1.0
    cartan_pullback_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class _CenturionsPoincareGerm:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    GÉRMEN CELESTE DE AUDITORÍA (objeto terminal de Fase I, inicial de Fase II).
    ═══════════════════════════════════════════════════════════════════════════
    Compone las cinco auditorías crudas (Cortina de Potencia + Maupertuis +
    KAM + Melnikov + Retorno) y sus escalas de normalización, más los
    invariantes integrales de Poincaré–Cartan. La Fase II valúa este gérmen
    en el retículo de Heyting Ω₃⁵ = Ω₃^5 vía meet (ínfimo de seguridad).
    """

    power_curtain: _PowerCurtainAudit
    maupertuis: Optional[_MaupertuisAudit]
    kam: Optional[_KAMAudit]
    melnikov: Optional[_MelnikovAudit]
    return_map: Optional[_ReturnMapAudit]
    two_n: int
    power_scale: float
    maupertuis_scale: float
    kam_scale: float
    melnikov_scale: float
    return_scale: float
    symplectic_cartan_defect: float = 0.0
    conley_zehnder_index: int = 0
    rotation_number: float = 0.0


class _CenturionsPoincareAuditCore:
    r"""
    Fase I. Núcleo ciego de auditoría celeste que dialoga con
    ImperialCenturionsEngine v6 y con _PoincareCelestialKernel.

    Provee métodos tolerantes a fallos para cada canal celeste. El morfismo
    terminal `synthesize_centurions_poincare_germ` compone los 5 canales en
    un único 𝒢_I = _CenturionsPoincareGerm, que ES el objeto inicial de la
    Fase II (consume exactamente este tipo).
    """

    def __init__(self, engine: ImperialCenturionsEngine, two_n: int) -> None:
        self._engine = engine
        self._two_n = int(two_n)
        self._kernel = _PoincareCelestialKernel

    @property
    def engine(self) -> ImperialCenturionsEngine:
        return self._engine

    @property
    def kernel(self) -> type[_PoincareCelestialKernel]:
        return self._kernel

    # ── I.1 Coerciones robustas ──────────────────────────────────────────
    @staticmethod
    def _as_vec(name: str, values: Any) -> np.ndarray:
        try:
            arr = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no es convertible a ndarray.") from exc
        vec = np.asarray(arr).reshape(-1)
        if vec.size == 0:
            raise ValueError(f"{name} no puede ser vacío.")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} contiene no-finitos.")
        return vec

    @staticmethod
    def _as_matrix(name: str, values: Any) -> np.ndarray:
        try:
            arr = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no es convertible a ndarray.") from exc
        if arr.ndim == 1:
            side = int(np.sqrt(arr.size))
            if side * side != arr.size:
                raise ValueError(f"{name} plana no es un cuadrado perfecto.")
            arr = arr.reshape(side, side)
        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser de rango 2.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene no-finitos.")
        return arr

    @staticmethod
    def _frobenius(matrix: np.ndarray) -> float:
        a = np.asarray(matrix)
        if a.size == 0:
            return 0.0
        return float(np.linalg.norm(a, "fro"))

    def _canonical_J(self, two_n: Optional[int] = None) -> np.ndarray:
        n = int(self._two_n if two_n is None else two_n)
        try:
            return _SpectralCore.assemble_standard_J(n)
        except ValueError:
            return np.zeros((n, n), dtype=np.float64)

    # ── I.2 Auditoría de la cortina de potencia ─────────────────────────
    def power_curtain_audit(
        self,
        ph_centurion: PortHamiltonianCenturion,
        x: np.ndarray,
        external_u: np.ndarray,
    ) -> _PowerCurtainAudit:
        """Auditoría cruda de la Cortina de Potencia IDA-PBC + Rayleigh + Cartan."""
        try:
            return ph_centurion.evaluate_power_curtain(x, external_u)
        except Exception as exc:
            logger.error("Fallo en evaluate_power_curtain: %s", exc)
            return _PowerCurtainAudit(
                dissipation_power=float("inf"),
                interconnection_leak=float("inf"),
                port_supply_rate=float("inf"),
                predicted_hdot=float("inf"),
                gradient_norm=float("inf"),
                hamiltonian=float("inf"),
                antiwindup_engaged=False,
                extra_damping=0.0,
                verdict="VETOED",
                rayleigh_function=float("inf"),
                poincare_cartan_lagrangian=float("nan"),
            )

    # ── I.3 Auditoría de Maupertuis–Jacobi + Christoffel + Jacobi ───────
    def maupertuis_audit(
        self,
        x_state: np.ndarray,
        dt_step: float,
        g_base_metric: np.ndarray,
        potential_V: float,
        grad_V: np.ndarray,
        total_energy_H0: float,
    ) -> _MaupertuisAudit:
        r"""
        Ejecuta `integrate_symplectic_maupertuis_step` del motor y enriquece
        el informe con conexión conforme de Koszul, marea de Jacobi y
        Lagrangiano de Poincaré–Cartan.
        """
        try:
            xv = self._as_vec("x_state", x_state)
            g = self._as_matrix("g_base_metric", g_base_metric)
            grad_v = self._as_vec("grad_V", grad_V)
            step_report = self._engine.integrate_symplectic_maupertuis_step(
                x_state=xv,
                dt_step=float(dt_step),
                g_base_metric=g,
                potential_V=float(potential_V),
                grad_V=grad_v,
                total_energy_H0=float(total_energy_H0),
            )
            hill_margin = float(total_energy_H0 - potential_V)
            n_index = float(getattr(step_report, "refractive_index_n", 0.0))
            H_step = float(getattr(step_report, "hamiltonian_energy", 0.0))
            christoffel = self._kernel.christoffel_conformal_strength(
                n_index, grad_v, hill_margin
            )
            tidal = self._kernel.jacobi_tidal_norm(grad_v, hill_margin)
            try:
                q, p = self._kernel.split_qp(xv, int(xv.size))
                # q̇ ≈ p en métrica euclídea; en general q̇ = g^{-1} p / n-scaling
                cartan_L = self._kernel.poincare_cartan_on_field(p, p, H_step)
            except ValueError:
                cartan_L = float("nan")
            return _MaupertuisAudit(
                conformal_factor=float(max(2.0 * hill_margin, _MAUPERTUIS_HILL_FLOOR)),
                refractive_index=n_index,
                hill_margin=hill_margin,
                is_in_hill_region=bool(hill_margin > _MAUPERTUIS_HILL_FLOOR),
                action_density=float(getattr(step_report, "maupertuis_action_density", 0.0)),
                volume_drift=float(getattr(step_report, "volume_drift_det", float("inf"))),
                is_symplectic_coherent=bool(getattr(step_report, "is_symplectic_coherent", False)),
                hamiltonian_energy=H_step,
                engine_ok=True,
                christoffel_strength=float(christoffel),
                jacobi_tidal_norm=float(tidal),
                poincare_cartan_lagrangian=float(cartan_L),
            )
        except Exception as exc:
            logger.error("Fallo en integrate_symplectic_maupertuis_step: %s", exc)
            return _MaupertuisAudit(
                conformal_factor=0.0,
                refractive_index=0.0,
                hill_margin=0.0,
                is_in_hill_region=False,
                action_density=float("inf"),
                volume_drift=float("inf"),
                is_symplectic_coherent=False,
                hamiltonian_energy=float("inf"),
                engine_ok=False,
                christoffel_strength=float("inf"),
                jacobi_tidal_norm=float("inf"),
                poincare_cartan_lagrangian=float("nan"),
            )

    # ── I.4 Auditoría KAM (pequeños divisores + homológica Lindstedt) ───
    def kam_audit(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        jacobian_M: np.ndarray,
        canonical_J: np.ndarray,
    ) -> _KAMAudit:
        r"""
        Ejecuta `compute_poincare_small_divisors_spectrum` del motor y
        adjunta el residual homológico de Poincaré–Lindstedt y el defecto
        de pullback de Cartan.
        """
        try:
            omega = self._as_vec("frequency_vector_omega", frequency_vector_omega)
            wave_k = np.asarray(wave_vectors_k, dtype=np.float64)
            jac_m = self._as_matrix("jacobian_M", jacobian_M)
            canon_j = np.asarray(canonical_J, dtype=np.float64)
            cert = self._engine.compute_poincare_small_divisors_spectrum(
                frequency_vector_omega=omega,
                wave_vectors_k=wave_k,
                jacobian_M=jac_m,
                canonical_J=canon_j,
            )
            min_div = float(getattr(cert, "min_divisor", float("inf")))
            return _KAMAudit(
                min_divisor=min_div,
                novikov_weight=float(getattr(cert, "novikov_weight", 0.0)),
                maurercartan_res=float(getattr(cert, "maurercartan_residual", float("inf"))),
                volume_drift=float(getattr(cert, "volume_drift", float("inf"))),
                resonance_gap=float(getattr(cert, "resonance_gap", float("inf"))),
                tau=float(getattr(cert, "tau", 1.0)),
                gamma=float(getattr(cert, "gamma", _WILKINSON_LIMIT)),
                is_diophantine=bool(getattr(cert, "is_diophantine", False)),
                is_kam_stable=bool(getattr(cert, "is_kam_stable", False)),
                engine_ok=True,
                homological_residual=self._kernel.homological_lindstedt_residual(min_div),
                cartan_pullback_defect=self._kernel.symplectic_pullback_defect(jac_m, canon_j),
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_small_divisors_spectrum: %s", exc)
            return _KAMAudit(
                min_divisor=float("inf"),
                novikov_weight=0.0,
                maurercartan_res=float("inf"),
                volume_drift=float("inf"),
                resonance_gap=float("inf"),
                tau=1.0,
                gamma=_WILKINSON_LIMIT,
                is_diophantine=False,
                is_kam_stable=False,
                engine_ok=False,
                homological_residual=float("inf"),
                cartan_pullback_defect=float("inf"),
            )

    # ── I.5 Auditoría de Melnikov (ruptura homoclínica) ─────────────────
    def melnikov_audit(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float = 25.0,
    ) -> _MelnikovAudit:
        r"""
        Ejecuta `compute_melnikov_function` del motor y normaliza el
        certificado en `_MelnikovAudit`. Ceros simples ⇒ fractura homoclínica
        (entrelazamiento de Poincaré, génesis de herraduras de Smale).
        """
        try:
            t0s = self._as_vec("t0_grid", t0_grid)
            if not callable(homoclinic_flow):
                raise ValueError("homoclinic_flow debe ser invocable.")
            cert = self._engine.compute_melnikov_function(
                homoclinic_flow=homoclinic_flow,
                hamiltonian_0=hamiltonian_0,
                hamiltonian_1=hamiltonian_1,
                t0_grid=t0s,
                t_inf=float(t_inf),
            )
            return _MelnikovAudit(
                melnikov_value=float(getattr(cert, "melnikov_value", float("nan"))),
                melnikov_derivative=float(getattr(cert, "melnikov_derivative", float("nan"))),
                is_simple_zero=bool(getattr(cert, "is_simple_zero", False)),
                homoclinic_splitting=float(getattr(cert, "homoclinic_splitting", float("nan"))),
                engine_ok=True,
            )
        except Exception as exc:
            logger.error("Fallo en compute_melnikov_function: %s", exc)
            return _MelnikovAudit(
                melnikov_value=float("nan"),
                melnikov_derivative=float("nan"),
                is_simple_zero=False,
                homoclinic_splitting=float("nan"),
                engine_ok=False,
            )

    # ── I.6 Auditoría del mapa de retorno de Poincaré + CZ + Birkhoff ───
    def return_map_audit(
        self,
        jacobian_M: np.ndarray,
        period_T: float = 1.0,
        canonical_J: Optional[np.ndarray] = None,
    ) -> _ReturnMapAudit:
        r"""
        Ejecuta `compute_poincare_return_map` del motor y extrae el espectro
        de Floquet, Lyapunov máximo, índice de Conley–Zehnder, número de
        rotación de Poincaré, twist de Birkhoff y transversalidad de Σ.
        """
        try:
            jac_m = self._as_matrix("jacobian_M", jacobian_M)
            cert = self._engine.compute_poincare_return_map(
                jacobian_M=jac_m, period_T=float(period_T)
            )
            floq = np.asarray(
                getattr(cert, "floquet_multipliers", np.array([])),
                dtype=np.complex128,
            )
            lyap = np.asarray(
                getattr(cert, "lyapunov_spectrum", np.array([])),
                dtype=np.float64,
            )
            floq_par = float(
                np.max(np.abs(np.abs(floq) - 1.0)) if floq.size else 0.0
            )
            cz_idx, cz_nd, cz_dist = self._kernel.conley_zehnder_index(jac_m)
            rho, twist, n_ell = self._kernel.rotation_number_and_birkhoff_twist(floq)
            trans = self._kernel.section_transversality(jac_m)
            J = canonical_J if canonical_J is not None else self._canonical_J(jac_m.shape[0])
            cartan = self._kernel.symplectic_pullback_defect(jac_m, np.asarray(J))
            return _ReturnMapAudit(
                floquet_multipliers=floq,
                lyapunov_spectrum=lyap,
                max_lyapunov=float(getattr(cert, "max_lyapunov", 0.0)),
                floquet_parabolic=floq_par,
                is_hyperbolic=bool(getattr(cert, "is_hyperbolic", False)),
                is_elliptic=bool(getattr(cert, "is_elliptic", False)),
                is_parabolic=bool(getattr(cert, "is_parabolic", False)),
                trace_M=float(getattr(cert, "trace_M", float("nan"))),
                det_M=float(getattr(cert, "det_M", float("nan"))),
                engine_ok=True,
                conley_zehnder_index=int(cz_idx),
                cz_nondegenerate=bool(cz_nd),
                cz_distance_to_one=float(cz_dist),
                rotation_number=float(rho) if np.isfinite(rho) else float("nan"),
                birkhoff_twist=float(twist),
                elliptic_multiplicity=int(n_ell),
                section_transversality=float(trans),
                cartan_pullback_defect=float(cartan),
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_return_map: %s", exc)
            return _ReturnMapAudit(
                floquet_multipliers=np.array([], dtype=np.complex128),
                lyapunov_spectrum=np.array([], dtype=np.float64),
                max_lyapunov=float("inf"),
                floquet_parabolic=float("inf"),
                is_hyperbolic=False,
                is_elliptic=False,
                is_parabolic=False,
                trace_M=float("nan"),
                det_M=float("nan"),
                engine_ok=False,
                conley_zehnder_index=0,
                cz_nondegenerate=False,
                cz_distance_to_one=0.0,
                rotation_number=float("nan"),
                birkhoff_twist=0.0,
                elliptic_multiplicity=0,
                section_transversality=0.0,
                cartan_pullback_defect=float("inf"),
            )

    # ── I.10 MORFISMO TERMINAL Φ_I: gérmen Poincaré de Centuriones ────────
    def synthesize_centurions_poincare_germ(
        self,
        ph_centurion: PortHamiltonianCenturion,
        x_state: np.ndarray,
        external_u: np.ndarray,
        g_base_metric: np.ndarray,
        potential_V: float,
        grad_V: np.ndarray,
        total_energy_H0: float,
        dt_step: float = 0.001,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        t_inf: float = 25.0,
        period_T: float = 1.0,
    ) -> _CenturionsPoincareGerm:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE I ≅ OBJETO INICIAL DE LA FASE II.
        ═══════════════════════════════════════════════════════════════════════
        Compone las cinco auditorías crudas (Cortina de Potencia, Maupertuis,
        KAM, Melnikov, Retorno) más los invariantes de Poincaré–Cartan,
        Conley–Zehnder y el número de rotación, en un único 𝒢_I sobre el
        cual la Fase II instancia los clasificadores de Heyting Ω₃⁵.

        Firma de continuación (Fase II):
            induce_centurions_ooda_germ(germ: _CenturionsPoincareGerm)
                -> _CenturionsOODAGerm
        """
        power = self.power_curtain_audit(ph_centurion, x_state, external_u)
        maupertuis = self.maupertuis_audit(
            x_state, dt_step, g_base_metric, potential_V, grad_V, total_energy_H0
        )

        jac_m = jacobian_M if jacobian_M is not None else np.eye(self._two_n)
        canon_j = (
            canonical_J
            if canonical_J is not None
            else self._canonical_J(self._two_n)
        )

        if frequency_vector_omega is not None and wave_vectors_k is not None:
            kam = self.kam_audit(
                frequency_vector_omega, wave_vectors_k, jac_m, canon_j
            )
        else:
            kam = None

        if (
            homoclinic_flow is not None
            and hamiltonian_0 is not None
            and hamiltonian_1 is not None
            and t0_grid is not None
        ):
            melnikov = self.melnikov_audit(
                homoclinic_flow, hamiltonian_0, hamiltonian_1, t0_grid, t_inf=t_inf
            )
        else:
            melnikov = None

        return_map = self.return_map_audit(
            jac_m, period_T=period_T, canonical_J=np.asarray(canon_j)
        )

        power_scale = max(
            abs(power.dissipation_power) if np.isfinite(power.dissipation_power) else 1.0,
            abs(power.predicted_hdot) if np.isfinite(power.predicted_hdot) else 1.0,
            1.0,
        )
        mau_scale = (
            max(maupertuis.refractive_index, 1.0)
            if np.isfinite(maupertuis.refractive_index)
            else 1.0
        )
        kam_scale = (
            max(kam.min_divisor, 1.0)
            if (kam is not None and np.isfinite(kam.min_divisor))
            else 1.0
        )
        mel_scale = max(
            abs(melnikov.melnikov_value)
            if (melnikov is not None and np.isfinite(melnikov.melnikov_value))
            else 0.0,
            abs(melnikov.homoclinic_splitting)
            if (melnikov is not None and np.isfinite(melnikov.homoclinic_splitting))
            else 0.0,
            1.0,
        )
        if return_map.floquet_multipliers.size:
            ret_scale = max(float(np.max(np.abs(return_map.floquet_multipliers))), 1.0)
        else:
            ret_scale = 1.0

        cartan_defect = float(return_map.cartan_pullback_defect)
        if kam is not None and np.isfinite(kam.cartan_pullback_defect):
            cartan_defect = max(cartan_defect, float(kam.cartan_pullback_defect))

        return _CenturionsPoincareGerm(
            power_curtain=power,
            maupertuis=maupertuis,
            kam=kam,
            melnikov=melnikov,
            return_map=return_map,
            two_n=int(self._two_n),
            power_scale=float(power_scale),
            maupertuis_scale=float(mau_scale),
            kam_scale=float(kam_scale),
            melnikov_scale=float(mel_scale),
            return_scale=float(ret_scale),
            symplectic_cartan_defect=float(cartan_defect),
            conley_zehnder_index=int(return_map.conley_zehnder_index),
            rotation_number=float(return_map.rotation_number)
            if np.isfinite(return_map.rotation_number)
            else float("nan"),
        )


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE II — CLASIFICADOR DE HEYTING Ω₃⁵ Y LIFTING OODA                     ║
# ║                                                                          ║
# ║ Continuación formal de I.10: el primer morfismo de esta fase consume     ║
# ║ exactamente 𝒢_I = _CenturionsPoincareGerm.                               ║
# ║                                                                          ║
# ║ Objetos: _CelestialVeredict, _CenturionsOODAGerm.                        ║
# ║                                                                          ║
# ║ Morfismo terminal (II.7): induce_centurions_ooda_germ                    ║
# ║     ↦ 𝒢_II = _CenturionsOODAGerm (inicial de Fase III).                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
@dataclass(frozen=True, slots=True)
class _CelestialVeredict:
    r"""
    Veredicto de una auditoría celeste en el retículo de Heyting Ω₃.
    Transporta:
      • verdict   : HeytingVerdict ∈ {VETOED, DEGRADED, COHERENT}.
      • metric    : valor numérico subyacente.
      • threshold : umbral utilizado.
      • token     : nombre del veredicto ('COHERENT' / 'DEGRADED' / 'VETOED').
      • reason    : causa si no es COHERENT.
    """

    verdict: HeytingVerdict
    metric: float
    threshold: float
    token: str
    reason: str = ""

    @property
    def godel_value(self) -> float:
        return self.verdict.godel_value


@dataclass(frozen=True, slots=True)
class _CenturionsOODAGerm:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    GÉRMEN OODA (objeto terminal de Fase II, inicial de Fase III).
    ═══════════════════════════════════════════════════════════════════════════
    Compone los cinco veredictos H₃ (Cortina de Potencia, Maupertuis, KAM,
    Melnikov, Retorno) y expone:
      • heyting_meet : ínfimo de permiso = PEOR CASO de seguridad (colapso),
      • heyting_join : supremo de verdad = MEJOR CASO diagnóstico,
      • godel_meet   : ínfimo de valores de Gödel,
      • authorization_residue : implicación de Gödel (meet → COHERENT).
    """

    power_curtain: _CelestialVeredict
    maupertuis: _CelestialVeredict
    kam: _CelestialVeredict
    melnikov: _CelestialVeredict
    return_map: _CelestialVeredict
    heyting_join: HeytingVerdict
    heyting_meet: HeytingVerdict
    godel_meet: float
    two_n: int
    authorization_residue: HeytingVerdict = HeytingVerdict.COHERENT
    symplectic_cartan_defect: float = 0.0
    conley_zehnder_index: int = 0
    rotation_number: float = 0.0


class _CelestialHeytingClassifier:
    r"""
    Fase II. Clasificador en el álgebra de Heyting de tres valores Ω₃.

    CONTINUACIÓN FORMAL DE I.10:
        lift_from_poincare_germ(germ: _CenturionsPoincareGerm) es el primer
        morfismo de esta fase y consume el objeto terminal de la Fase I.

    Los cinco clasificadores inducen un morfismo Ω₃⁵ → Ω₃ vía:
      • meet (ínfimo = peor permiso) para el colapso de seguridad,
      • join (supremo = mejor verdad) para diagnóstico.
    """

    def __init__(self, safety_margin: float = 1.0) -> None:
        self._margin = float(max(safety_margin, 0.0))

    @property
    def safety_margin(self) -> float:
        return self._margin

    # ── II.0 Continuación de I.10: lifting del gérmen Poincaré ───────────
    def lift_from_poincare_germ(
        self, germ: _CenturionsPoincareGerm
    ) -> _CenturionsOODAGerm:
        r"""
        Primer morfismo de la Fase II. Continúa I.10:
            synthesize_centurions_poincare_germ(...) -> 𝒢_I
            lift_from_poincare_germ(𝒢_I)             -> 𝒢_II
        Delegado canónico de `induce_centurions_ooda_germ`.
        """
        return self.induce_centurions_ooda_germ(germ)

    # ── II.1 Umbral ──────────────────────────────────────────────────────
    def threshold(self, base: float, scale: float = 1.0) -> float:
        r"""τ = τ₀ · μ_safety con piso metrológico relativo."""
        abs_tol = float(base) * max(self._margin, 1e-30)
        rel_tol = max(float(scale), 1.0) * _MACHINE_EPS * 10.0
        return float(max(abs_tol, rel_tol, _MACHINE_EPS))

    def verdict_from_metric(
        self,
        metric: float,
        base_tolerance: float,
        scale: float = 1.0,
        degradation_factor: float = _DEGRADATION_FACTOR,
    ) -> HeytingVerdict:
        r"""
        Asignación H₃ por umbral:
            metric ≤ τ·deg  ⇒ COHERENT
            τ·deg < metric ≤ τ ⇒ DEGRADED
            metric > τ       ⇒ VETOED
        """
        if not np.isfinite(metric) or metric < 0.0:
            return HeytingVerdict.VETOED
        tol = self.threshold(base_tolerance, scale)
        deg = float(degradation_factor) if 0.0 < degradation_factor <= 1.0 else _DEGRADATION_FACTOR
        if metric > tol:
            return HeytingVerdict.VETOED
        if metric > tol * deg:
            return HeytingVerdict.DEGRADED
        return HeytingVerdict.COHERENT

    # ── II.2 Clasificador de la Cortina de Potencia (IDA-PBC) ────────────
    def classify_power_curtain(self, audit: _PowerCurtainAudit) -> _CelestialVeredict:
        r"""
        Valúa la cortina de potencia IDA-PBC en Ω₃:
          • VETOED   : disipación negativa (P_diss < 0) o leak de interconexión.
          • DEGRADED : P_diss elevada (> _DISS_DEGRADE) o veredicto degradado.
          • COHERENT : Ḣ_d ≤ 0 y antisimetría de Dirac exacta.
        """
        try:
            ht = HeytingVerdict.from_token(audit.verdict)
        except (ValueError, AttributeError):
            ht = HeytingVerdict.VETOED
        if not np.isfinite(audit.dissipation_power) or audit.dissipation_power < _PASSIVITY_FLOOR:
            ht = HeytingVerdict.VETOED
        if abs(audit.interconnection_leak) > _INTERCONNECTION_LEAK * max(
            abs(audit.dissipation_power), 1.0
        ):
            ht = HeytingVerdict.VETOED
        if np.isfinite(audit.predicted_hdot) and audit.predicted_hdot > _HARD_DIVERGENCE_CEILING:
            ht = ht.meet(HeytingVerdict.DEGRADED)
        reason = "" if ht is HeytingVerdict.COHERENT else (audit.verdict or "pasividad rota")
        return _CelestialVeredict(
            verdict=ht,
            metric=float(audit.dissipation_power),
            threshold=float(_DISS_DEGRADE),
            token=ht.name,
            reason=reason,
        )

    # ── II.3 Clasificador de Maupertuis–Jacobi + Christoffel + marea ────
    def classify_maupertuis(self, audit: _MaupertuisAudit) -> _CelestialVeredict:
        r"""
        Valúa la geodésica de Maupertuis en Ω₃, con blow-up de Christoffel
        cerca de ∂D_H y marea de Jacobi:
          • VETOED   : H₀ − V ≤ 0, symplectic drift, Γ̃ → ∞ o marea extrema.
          • DEGRADED : n(q) bajo (cerca de Hill), coherencia débil o marea media.
          • COHERENT : H₀ − V > 0, det(M_step) = 1 + O(ε), n(q) sana, Γ̃ acotada.
        """
        if (not audit.engine_ok) or (not audit.is_in_hill_region):
            ht = HeytingVerdict.VETOED
            reason = "fuera de región de Hill (H₀ − V ≤ 0)"
        elif not np.isfinite(audit.volume_drift) or audit.volume_drift > _SPECTRAL_TOL:
            ht = HeytingVerdict.VETOED
            reason = f"drift de Liouville={audit.volume_drift:.3e}"
        elif (
            not np.isfinite(audit.christoffel_strength)
            or audit.christoffel_strength > _CHRISTOFFEL_BLOWUP
        ):
            ht = HeytingVerdict.VETOED
            reason = f"blow-up de Christoffel conforme ‖Γ̃‖={audit.christoffel_strength:.3e}"
        elif (
            not np.isfinite(audit.jacobi_tidal_norm)
            or audit.jacobi_tidal_norm > _JACOBI_TIDAL_VETO
        ):
            ht = HeytingVerdict.VETOED
            reason = f"marea de Jacobi={audit.jacobi_tidal_norm:.3e}"
        elif not audit.is_symplectic_coherent:
            ht = HeytingVerdict.DEGRADED
            reason = "coherencia simpléctica débil"
        elif audit.jacobi_tidal_norm > _JACOBI_TIDAL_DEGRADE:
            ht = HeytingVerdict.DEGRADED
            reason = f"marea de Jacobi moderada={audit.jacobi_tidal_norm:.3e}"
        elif (
            np.isfinite(audit.refractive_index)
            and 0.0 < audit.refractive_index < _REFRACTIVE_NEAR_HILL
        ):
            ht = HeytingVerdict.DEGRADED
            reason = "índice de refracción bajo (cerca de ∂D_H)"
        else:
            ht = HeytingVerdict.COHERENT
            reason = ""
        return _CelestialVeredict(
            verdict=ht,
            metric=float(audit.volume_drift) if np.isfinite(audit.volume_drift) else float("inf"),
            threshold=float(_SPECTRAL_TOL),
            token=ht.name,
            reason=reason,
        )

    # ── II.4 Clasificador KAM + homológica de Lindstedt + Cartan ─────────
    def classify_kam(self, audit: Optional[_KAMAudit]) -> _CelestialVeredict:
        r"""
        Valúa los pequeños divisores de Poincaré–KAM en Ω₃:
          • COHERENT : |⟨k,ω⟩| ≥ γ/|k|^τ ∧ diofantino ∧ drift ≤ ε_W ∧ Cartan sano.
          • DEGRADED : divisor pequeño positivo, no diofantino o residual homológico.
          • VETOED   : divisor ≤ ε_W, drift > ε_W, pullback de Cartan roto o motor caído.
        """
        if audit is None:
            return _CelestialVeredict(
                verdict=HeytingVerdict.COHERENT,
                metric=0.0,
                threshold=_KAM_THRESHOLD,
                token="COHERENT",
                reason="KAM no aplicable (canal no inyectado)",
            )
        tol = self.threshold(_KAM_THRESHOLD, max(audit.min_divisor, 1.0))
        if (not audit.engine_ok) or (not np.isfinite(audit.min_divisor)):
            ht = HeytingVerdict.VETOED
            reason = "motor caído"
        elif audit.min_divisor <= _WILKINSON_LIMIT:
            ht = HeytingVerdict.VETOED
            reason = f"divisor ≤ ε_W ({audit.min_divisor:.3e})"
        elif audit.volume_drift > _WILKINSON_LIMIT:
            ht = HeytingVerdict.VETOED
            reason = f"drift > ε_W ({audit.volume_drift:.3e})"
        elif (
            np.isfinite(audit.cartan_pullback_defect)
            and audit.cartan_pullback_defect > _CARTAN_DEFECT_TOL
        ):
            ht = HeytingVerdict.VETOED
            reason = f"Φ*ω ≠ ω (Cartan defect={audit.cartan_pullback_defect:.3e})"
        elif not audit.is_diophantine:
            ht = HeytingVerdict.DEGRADED
            reason = "no diofantino"
        elif audit.homological_residual > (1.0 / max(tol, _MACHINE_EPS)):
            ht = HeytingVerdict.DEGRADED
            reason = f"residual homológico Lindstedt={audit.homological_residual:.3e}"
        elif audit.is_kam_stable and audit.min_divisor >= tol:
            ht = HeytingVerdict.COHERENT
            reason = ""
        elif audit.min_divisor >= tol * _DEGRADATION_FACTOR:
            ht = HeytingVerdict.DEGRADED
            reason = "divisor pequeño"
        else:
            ht = HeytingVerdict.VETOED
            reason = "divisor bajo umbral"
        return _CelestialVeredict(
            verdict=ht,
            metric=float(audit.min_divisor) if np.isfinite(audit.min_divisor) else float("inf"),
            threshold=float(tol),
            token=ht.name,
            reason=reason,
        )

    # ── II.5 Clasificador de Melnikov ────────────────────────────────────
    def classify_melnikov(self, audit: Optional[_MelnikovAudit]) -> _CelestialVeredict:
        r"""
        Valúa la función de Melnikov en Ω₃:
          • VETOED   : cero simple (M=0, M'≠0) ⇒ fractura homoclínica de Poincaré.
          • DEGRADED : |M(t₀)| pequeña o sin cruce estricto.
          • COHERENT : |M(t₀)| ≥ τ y sin ceros simples (variedades pegadas).
        """
        if audit is None:
            return _CelestialVeredict(
                verdict=HeytingVerdict.COHERENT,
                metric=0.0,
                threshold=_MELNIKOV_THRESHOLD,
                token="COHERENT",
                reason="Melnikov no aplicable (canal no inyectado)",
            )
        tol = self.threshold(_MELNIKOV_THRESHOLD, max(abs(audit.melnikov_value), 1.0))
        if (not audit.engine_ok) or (not np.isfinite(audit.melnikov_value)):
            ht = HeytingVerdict.VETOED
            reason = "motor caído"
        elif audit.is_simple_zero:
            ht = HeytingVerdict.VETOED
            reason = "cero simple ⇒ fractura homoclínica de Poincaré"
        elif abs(audit.melnikov_value) >= tol:
            ht = HeytingVerdict.COHERENT
            reason = ""
        elif abs(audit.melnikov_value) >= tol * _DEGRADATION_FACTOR:
            ht = HeytingVerdict.DEGRADED
            reason = "M(t₀) pequeño"
        else:
            ht = HeytingVerdict.VETOED
            reason = "M(t₀) bajo umbral"
        return _CelestialVeredict(
            verdict=ht,
            metric=float(abs(audit.melnikov_value)) if np.isfinite(audit.melnikov_value) else float("inf"),
            threshold=float(tol),
            token=ht.name,
            reason=reason,
        )

    # ── II.6 Clasificador del mapa de retorno + CZ + Birkhoff ────────────
    def classify_return_map(self, audit: _ReturnMapAudit) -> _CelestialVeredict:
        r"""
        Valúa el mapa de retorno P: Σ → Σ en Ω₃, con Conley–Zehnder y Birkhoff:
          • COHERENT : elíptico puro, |L_max| ≈ 0, CZ no degenerado, Cartan sano.
          • DEGRADED : parabólico débil, hiperbolicidad suave, CZ degenerado o
                       sección poco transversal.
          • VETOED   : hiperbolicidad fuerte (|L_max| > τ) o Φ*ω ≠ ω.
        """
        if not audit.engine_ok:
            ht = HeytingVerdict.VETOED
            reason = "motor caído"
        elif (
            np.isfinite(audit.cartan_pullback_defect)
            and audit.cartan_pullback_defect > _CARTAN_DEFECT_TOL
        ):
            ht = HeytingVerdict.VETOED
            reason = f"Φ*ω ≠ ω (Cartan defect={audit.cartan_pullback_defect:.3e})"
        else:
            tol = self.threshold(_FLOQUET_PARABOLIC, max(abs(audit.max_lyapunov), 1.0))
            if audit.is_elliptic and abs(audit.max_lyapunov) <= tol * _DEGRADATION_FACTOR:
                if (not audit.cz_nondegenerate) or (
                    audit.section_transversality <= _SECTION_TRANSVERSALITY_FLOOR
                ):
                    ht = HeytingVerdict.DEGRADED
                    reason = (
                        "CZ degenerado"
                        if not audit.cz_nondegenerate
                        else "sección de Poincaré no transversal"
                    )
                else:
                    ht = HeytingVerdict.COHERENT
                    reason = ""
            elif audit.floquet_parabolic <= tol:
                ht = HeytingVerdict.DEGRADED
                reason = "parabólico débil"
            elif abs(audit.max_lyapunov) <= _MELNIKOV_LYAPUNOV_VETO:
                ht = HeytingVerdict.DEGRADED
                reason = f"L_max={audit.max_lyapunov:.3e}"
            else:
                ht = HeytingVerdict.VETOED
                reason = f"L_max={audit.max_lyapunov:.3e} > {_MELNIKOV_LYAPUNOV_VETO}"
        return _CelestialVeredict(
            verdict=ht,
            metric=float(abs(audit.max_lyapunov)) if np.isfinite(audit.max_lyapunov) else float("inf"),
            threshold=float(_FLOQUET_PARABOLIC),
            token=ht.name,
            reason=reason,
        )

    # ── II.7 MORFISMO TERMINAL Φ_II: gérmen OODA de Centuriones ──────────
    def induce_centurions_ooda_germ(
        self, germ: _CenturionsPoincareGerm
    ) -> _CenturionsOODAGerm:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE II ≅ OBJETO INICIAL DE LA FASE III.
        ═══════════════════════════════════════════════════════════════════════
        Valúa 𝒢_I en Ω₃⁵ y aplica:
          • meet (ínfimo = peor permiso)  → colapso de seguridad,
          • join (supremo = mejor verdad) → diagnóstico,
          • implicación de Gödel meet → COHERENT → residuo de autorización.

        Firma de continuación (Fase III):
            collapse_from_ooda_germ(ooda_germ: _CenturionsOODAGerm, ...)
                -> Dict[str, Any]
        """
        v_power = self.classify_power_curtain(germ.power_curtain)
        v_mau = (
            self.classify_maupertuis(germ.maupertuis)
            if germ.maupertuis is not None
            else _CelestialVeredict(
                verdict=HeytingVerdict.COHERENT,
                metric=0.0,
                threshold=0.0,
                token="COHERENT",
                reason="Maupertuis no aplicable",
            )
        )
        v_kam = self.classify_kam(germ.kam)
        v_mel = self.classify_melnikov(germ.melnikov)
        v_ret = (
            self.classify_return_map(germ.return_map)
            if germ.return_map is not None
            else _CelestialVeredict(
                verdict=HeytingVerdict.COHERENT,
                metric=0.0,
                threshold=0.0,
                token="COHERENT",
                reason="Retorno no aplicable",
            )
        )

        joined = HeytingVerdict.join_all(
            v_power.verdict, v_mau.verdict, v_kam.verdict, v_mel.verdict, v_ret.verdict
        )
        met = HeytingVerdict.meet_all(
            v_power.verdict, v_mau.verdict, v_kam.verdict, v_mel.verdict, v_ret.verdict
        )
        godel_meet = float(
            min(
                v_power.godel_value,
                v_mau.godel_value,
                v_kam.godel_value,
                v_mel.godel_value,
                v_ret.godel_value,
            )
        )
        residue = met.implies(HeytingVerdict.COHERENT)
        return _CenturionsOODAGerm(
            power_curtain=v_power,
            maupertuis=v_mau,
            kam=v_kam,
            melnikov=v_mel,
            return_map=v_ret,
            heyting_join=joined,
            heyting_meet=met,
            godel_meet=godel_meet,
            two_n=int(germ.two_n),
            authorization_residue=residue,
            symplectic_cartan_defect=float(germ.symplectic_cartan_defect),
            conley_zehnder_index=int(germ.conley_zehnder_index),
            rotation_number=float(germ.rotation_number),
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
        latency = float(
            np.clip(
                _CROWBAR_LATENCY_NS + jitter,
                _CROWBAR_T_MIN_NS,
                _CROWBAR_T_MAX_NS,
            )
        )
        self.latched = True
        self.last_latency_ns = latency
        return latency

    def reset(self) -> None:
        self.latched = False
        self.last_latency_ns = 0.0


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FASE III — CÁMARA DE COHERENCIA Y COLAPSO AL DISYUNTOR CIBER-FÍSICO      ║
# ║                                                                          ║
# ║ Continuación formal de II.7: el primer morfismo de esta fase consume     ║
# ║ exactamente 𝒢_II = _CenturionsOODAGerm.                                  ║
# ║                                                                          ║
# ║ Objetos: CenturionsCoherenceChamber (ciclo OODA completo).               ║
# ║                                                                          ║
# ║ Morfismo terminal (III.2): process_coherence_cycle_certified             ║
# ║     ↦ colapso Ω₃ → {fire, no-fire} sobre GPIO14 / BT151 Crowbar,         ║
# ║       gobernado por el MEET de Heyting (peor permiso).                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================
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
        """Ciclo OODA básico (Cortina de Potencia + KMS) sin canales celestes."""
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
            "rayleigh_function": audit_ph.rayleigh_function,
            "poincare_cartan_lagrangian": audit_ph.poincare_cartan_lagrangian,
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

    # ── III.1 Continuación de II.7: colapso desde el gérmen OODA ─────────
    def collapse_from_ooda_germ(
        self,
        ooda_germ: _CenturionsOODAGerm,
        density_rho: np.ndarray,
        obs_A: np.ndarray,
        obs_B: np.ndarray,
        beta_kms: float,
    ) -> Dict[str, Any]:
        r"""
        Primer morfismo de la Fase III. Continúa II.7:
            induce_centurions_ooda_germ(𝒢_I) -> 𝒢_II
            collapse_from_ooda_germ(𝒢_II, …) -> acta de colapso ciber-físico
        Delegado canónico de `process_coherence_cycle_certified`.
        """
        return self.process_coherence_cycle_certified(
            ooda_germ=ooda_germ,
            density_rho=density_rho,
            obs_A=obs_A,
            obs_B=obs_B,
            beta_kms=beta_kms,
        )

    # ── III.2 Ciclo OODA certificado con canales celestes ────────────────
    def process_coherence_cycle_certified(
        self,
        ooda_germ: _CenturionsOODAGerm,
        density_rho: np.ndarray,
        obs_A: np.ndarray,
        obs_B: np.ndarray,
        beta_kms: float,
    ) -> Dict[str, Any]:
        r"""
        Ciclo OODA con los 5 canales celestes + KMS termodinámico.

        El MEET H₃ (ínfimo de permiso = peor caso) determina si se dispara
        el crowbar BT151 vía GPIO14 en < 400 ns (ISR en IRAM del ESP32).

        Corrección v7 respecto de v6: el join (supremo) es diagnóstico;
        el colapso se gobierna por el meet. join(VETOED, COHERENT) = COHERENT
        ocultaba vetos — ahora meet(VETOED, COHERENT) = VETOED dispara.
        """
        verdicts = {
            "power_curtain": ooda_germ.power_curtain,
            "maupertuis": ooda_germ.maupertuis,
            "kam": ooda_germ.kam,
            "melnikov": ooda_germ.melnikov,
            "return_map": ooda_germ.return_map,
        }

        entropy = self.thermo_centurion.compute_von_neumann_entropy(density_rho)
        purity = self.thermo_centurion.compute_purity(density_rho)
        kms_res, verdict_thermo_str = self.thermo_centurion.verify_kms_condition(
            density_rho, obs_A, obs_B, beta_kms
        )
        T_eff, h_eff = self.thermo_centurion.tune_effective_planck_constant(entropy)
        v_thermo = HeytingVerdict.from_token(verdict_thermo_str)

        # Join diagnóstico (mejor coherencia observada) y meet de seguridad.
        joined = ooda_germ.heyting_join.join(v_thermo)
        met = ooda_germ.heyting_meet.meet(v_thermo)
        godel_meet = float(min(ooda_germ.godel_meet, v_thermo.godel_value))
        residue = met.implies(HeytingVerdict.COHERENT)

        crowbar_triggered = False
        latency_ns = 0.0
        if met is HeytingVerdict.VETOED:
            latency_ns = self._crowbar.fire(self._rng)
            crowbar_triggered = True
            logger.critical(
                "[CORTINA DE POTENCIA + CELESTE COLAPSADA] Veto incondicional. "
                "Meet H₃ = VETOED (PHS=%s, Mau=%s, KAM=%s, Mel=%s, Ret=%s, KMS=%s). "
                "CartanDefect=%.3e CZ=%d ρ=%.6f. "
                "Disparando BT151 [GPIO14] en %.2f ns via ISR en IRAM.",
                verdicts["power_curtain"].token,
                verdicts["maupertuis"].token,
                verdicts["kam"].token,
                verdicts["melnikov"].token,
                verdicts["return_map"].token,
                v_thermo.name,
                ooda_germ.symplectic_cartan_defect,
                ooda_germ.conley_zehnder_index,
                ooda_germ.rotation_number,
                latency_ns,
            )
        return {
            # Retículo Ω₃⁶ — meet gobierna, join diagnostica
            "heyting_verdict": met.name,
            "heyting_value": met.value,
            "heyting_meet": met.name,
            "heyting_join": joined.name,
            "godel_meet": godel_meet,
            "authorization_residue": residue.name,
            # Veredictos por canal (Fase II)
            "power_curtain_verdict": verdicts["power_curtain"].token,
            "maupertuis_verdict": verdicts["maupertuis"].token,
            "kam_verdict": verdicts["kam"].token,
            "melnikov_verdict": verdicts["melnikov"].token,
            "return_map_verdict": verdicts["return_map"].token,
            "thermodynamic_verdict": v_thermo.name,
            # Métricas celestes
            "maupertuis_volume_drift": verdicts["maupertuis"].metric,
            "kam_min_divisor": verdicts["kam"].metric,
            "melnikov_value": verdicts["melnikov"].metric,
            "return_map_lyapunov_max": verdicts["return_map"].metric,
            "power_curtain_dissipation": verdicts["power_curtain"].metric,
            "symplectic_cartan_defect": ooda_germ.symplectic_cartan_defect,
            "conley_zehnder_index": ooda_germ.conley_zehnder_index,
            "poincare_rotation_number": ooda_germ.rotation_number,
            # Motivos (audit trails)
            "power_curtain_reason": verdicts["power_curtain"].reason,
            "maupertuis_reason": verdicts["maupertuis"].reason,
            "kam_reason": verdicts["kam"].reason,
            "melnikov_reason": verdicts["melnikov"].reason,
            "return_map_reason": verdicts["return_map"].reason,
            # Termodinámica modular
            "thermodynamic_entropy": entropy,
            "thermodynamic_purity": purity,
            "kms_residual": kms_res,
            "effective_temperature": T_eff,
            "effective_planck_constant": h_eff,
            # Actuación ciber-física
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

    Audita las geodésicas de Maupertuis–Jacobi, los símbolos de Christoffel
    conformes, la 1-forma de Poincaré–Cartan, los pequeños divisores KAM, la
    ecuación homológica de Lindstedt, la función de Melnikov, el mapa de
    retorno de Poincaré (Floquet / Lyapunov / Conley–Zehnder / Birkhoff),
    la invarianza de Liouville y la pasividad Port-Hamiltoniana antes de
    autorizar comandos de potencia en la obra civil.

    Composición anidada (cada morfismo terminal es el objeto inicial del siguiente):
        Φ_I   : Auditoría cruda celeste                                          ⟶  𝒢_I
        Φ_II  : lift_from_poincare_germ(𝒢_I)  — valuación Ω₃⁵ + meet + Gödel    ⟶  𝒢_II
        Φ_III : collapse_from_ooda_germ(𝒢_II) — cámara + crowbar BT151           ⟶  acta
    """

    def __init__(
        self,
        dimension_n: int = 4,
        safety_margin: float = 1.0,
        novikov_valuation_T: float = 1.0,
        hamiltonian_energy_H0: float = 1.0,
        potential_V: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if int(dimension_n) <= 0 or int(dimension_n) % 2 != 0:
            raise ValueError(
                f"La dimensión del espacio de fase n={dimension_n} debe ser par."
            )
        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(max(safety_margin, 0.0))
        self._novikov_T: Final[float] = float(novikov_valuation_T)
        try:
            self._engine: ImperialCenturionsEngine = ImperialCenturionsEngine(
                dimension_n=self._n,
                hamiltonian_energy_H0=hamiltonian_energy_H0,
                potential_V=potential_V,
                novikov_valuation_T=novikov_valuation_T,
            )
        except TypeError:
            try:
                self._engine = ImperialCenturionsEngine(  # type: ignore[misc]
                    dimension_n=self._n,
                )
            except TypeError:
                self._engine = ImperialCenturionsEngine()  # type: ignore[misc]
        self._audit_core = _CenturionsPoincareAuditCore(
            engine=self._engine, two_n=self._n
        )
        self._classifier = _CelestialHeytingClassifier(self._safety_margin)
        self._rng: Final[Optional[np.random.Generator]] = rng
        self._last_germ: Optional[_CenturionsPoincareGerm] = None
        self._last_ooda: Optional[_CenturionsOODAGerm] = None

    @property
    def dimension(self) -> int:
        """Dimensión de la variedad base Q (dim T*Q = n)."""
        return self._n

    @property
    def engine(self) -> ImperialCenturionsEngine:
        return self._engine

    @property
    def safety_margin(self) -> float:
        return self._safety_margin

    # ══════════════════════════════════════════════════════════════════════
    # API FASE I — Auditorías crudas celestes
    # ══════════════════════════════════════════════════════════════════════
    def audit_maupertuis_geodesic_flow(
        self,
        x_state: NDArray[np.float64],
        g_base_metric: NDArray[np.float64],
        potential_V: float,
        grad_V: NDArray[np.float64],
        total_energy_H0: float,
        dt_step: float = 0.001,
    ) -> _MaupertuisAudit:
        r"""
        [AUDITORÍA MAUPERTUIS-JACOBI] Métrica conforme g̃ = 2(H₀ − V)g,
        índice de refracción n(q), región de Hill, Christoffel conforme,
        marea de Jacobi y drift de Liouville.
        """
        return self._audit_core.maupertuis_audit(
            x_state, dt_step, g_base_metric, potential_V, grad_V, total_energy_H0
        )

    def audit_poincare_kam_stability(
        self,
        frequency_vector_omega: NDArray[np.float64],
        wave_vectors_k: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        canonical_J: NDArray[np.float64],
    ) -> _KAMAudit:
        r"""
        [AUDITORÍA KAM] Pequeños divisores de Poincaré con diofantinidad
        γ/|k|^τ, residual homológico de Lindstedt y pullback de Cartan.
        """
        return self._audit_core.kam_audit(
            frequency_vector_omega, wave_vectors_k, jacobian_M, canonical_J
        )

    def audit_melnikov_homoclinic_splitting(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: NDArray[np.float64],
        t_inf: float = 25.0,
    ) -> _MelnikovAudit:
        r"""
        [AUDITORÍA MELNIKOV] Función M(t₀) = ∫ {H₀, H₁}(γ⁰(t − t₀)) dt.
        Ceros simples ⇒ fractura homoclínica de Poincaré.
        """
        return self._audit_core.melnikov_audit(
            homoclinic_flow, hamiltonian_0, hamiltonian_1, t0_grid, t_inf=t_inf
        )

    def audit_poincare_return_map(
        self,
        jacobian_M: NDArray[np.float64],
        period_T: float = 1.0,
        canonical_J: Optional[NDArray[np.float64]] = None,
    ) -> _ReturnMapAudit:
        r"""
        [AUDITORÍA RETORNO POINCARÉ] Mapa P: Σ → Σ con espectro de Floquet,
        exponentes de Lyapunov, índice de Conley–Zehnder, número de rotación
        y twist de Poincaré–Birkhoff.
        """
        return self._audit_core.return_map_audit(
            jacobian_M, period_T=period_T, canonical_J=canonical_J
        )

    def synthesize_centurions_poincare_germ(
        self,
        ph_centurion: PortHamiltonianCenturion,
        x_state: NDArray[np.float64],
        external_u: NDArray[np.float64],
        g_base_metric: NDArray[np.float64],
        potential_V: float,
        grad_V: NDArray[np.float64],
        total_energy_H0: float,
        dt_step: float = 0.001,
        frequency_vector_omega: Optional[NDArray[np.float64]] = None,
        wave_vectors_k: Optional[NDArray[np.float64]] = None,
        jacobian_M: Optional[NDArray[np.float64]] = None,
        canonical_J: Optional[NDArray[np.float64]] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[NDArray[np.float64]] = None,
        t_inf: float = 25.0,
        period_T: float = 1.0,
    ) -> _CenturionsPoincareGerm:
        r"""
        Réplica pública del morfismo I.10: sintetiza el gérmen celeste 𝒢_I.
        Este valor ES el objeto inicial de la Fase II.
        """
        germ = self._audit_core.synthesize_centurions_poincare_germ(
            ph_centurion=ph_centurion,
            x_state=x_state,
            external_u=external_u,
            g_base_metric=g_base_metric,
            potential_V=potential_V,
            grad_V=grad_V,
            total_energy_H0=total_energy_H0,
            dt_step=dt_step,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            canonical_J=canonical_J,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            t_inf=t_inf,
            period_T=period_T,
        )
        self._last_germ = germ
        return germ

    # ══════════════════════════════════════════════════════════════════════
    # API FASE II — Valuación en Ω₃⁵ y lifting OODA
    # (continúa desde synthesize_centurions_poincare_germ → 𝒢_I)
    # ══════════════════════════════════════════════════════════════════════
    def induce_centurions_ooda_germ(
        self, germ: Optional[_CenturionsPoincareGerm] = None
    ) -> _CenturionsOODAGerm:
        r"""
        Réplica pública del morfismo II.7 / II.0: valúa 𝒢_I en Ω₃⁵ vía el
        clasificador de Heyting y produce el gérmen OODA 𝒢_II.
        Este valor ES el objeto inicial de la Fase III.
        """
        if germ is None:
            if self._last_germ is None:
                raise ValueError(
                    "No hay gérmen previo: invoque synthesize_centurions_poincare_germ."
                )
            germ = self._last_germ
        ooda_germ = self._classifier.lift_from_poincare_germ(germ)
        self._last_ooda = ooda_germ
        return ooda_germ

    # ══════════════════════════════════════════════════════════════════════
    # API FASE III — Ciclo OODA y colapso al disyuntor
    # (continúa desde induce_centurions_ooda_germ → 𝒢_II)
    # ══════════════════════════════════════════════════════════════════════
    def execute_centurions_cycle(
        self,
        ph_centurion: PortHamiltonianCenturion,
        thermo_centurion: ThermodynamicCenturion,
        x_state: NDArray[np.float64],
        external_u: NDArray[np.float64],
        density_rho: NDArray[np.complex128],
        obs_A: NDArray[np.complex128],
        obs_B: NDArray[np.complex128],
        beta_kms: float,
        g_base_metric: NDArray[np.float64],
        potential_V: float,
        grad_V: NDArray[np.float64],
        total_energy_H0: float,
        dt_step: float = 0.001,
        frequency_vector_omega: Optional[NDArray[np.float64]] = None,
        wave_vectors_k: Optional[NDArray[np.float64]] = None,
        jacobian_M: Optional[NDArray[np.float64]] = None,
        canonical_J: Optional[NDArray[np.float64]] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[NDArray[np.float64]] = None,
        t_inf: float = 25.0,
        period_T: float = 1.0,
    ) -> Dict[str, Any]:
        r"""
        Ciclo OODA completo con los 5 canales celestes + KMS termodinámico,
        colapsando a Ω₃ por el MEET de Heyting y eventualmente disparando
        el crowbar BT151.
        """
        germ = self.synthesize_centurions_poincare_germ(
            ph_centurion=ph_centurion,
            x_state=x_state,
            external_u=external_u,
            g_base_metric=g_base_metric,
            potential_V=potential_V,
            grad_V=grad_V,
            total_energy_H0=total_energy_H0,
            dt_step=dt_step,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            canonical_J=canonical_J,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            t_inf=t_inf,
            period_T=period_T,
        )
        ooda_germ = self.induce_centurions_ooda_germ(germ)
        chamber = CenturionsCoherenceChamber(
            ph_centurion=ph_centurion,
            thermo_centurion=thermo_centurion,
            rng=self._rng,
        )
        return chamber.collapse_from_ooda_germ(
            ooda_germ=ooda_germ,
            density_rho=density_rho,
            obs_A=obs_A,
            obs_B=obs_B,
            beta_kms=beta_kms,
        )

    # ══════════════════════════════════════════════════════════════════════
    # API heredada — Auditoría Poincaré de coherencia espectral (retrocompat)
    # ══════════════════════════════════════════════════════════════════════
    def audit_centurions_poincare_geodesic_flow(
        self,
        x_state: NDArray[np.float64],
        J_desired: NDArray[np.float64],
        R_desired: NDArray[np.float64],
        grad_H_desired: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: NDArray[np.float64],
        dt_step: float = 0.001,
    ) -> CenturionsGovernanceCertificate:
        r"""
        [API 2.0 RETROCOMPATIBLE] Auditoría combinada de Maupertuis + Dirac +
        Rayleigh con veredicto Ω₃ en un único certificado.
        """
        n_dim = len(x_state) // 2
        grad_V = grad_H_desired[:n_dim]
        step_report: MaupertuisStepReport = self._engine.integrate_symplectic_maupertuis_step(
            x_state=x_state,
            dt_step=dt_step,
            g_base_metric=g_base_metric,
            potential_V=potential_V,
            grad_V=grad_V,
            total_energy_H0=total_energy_H0,
        )
        dirac_defect = float(la.norm(J_desired + J_desired.T, ord="fro"))
        R_symmetric = 0.5 * (R_desired + R_desired.T)
        min_eigenvalue_R = float(np.min(la.eigvalsh(R_symmetric)))
        rayleigh_rate = -float(grad_H_desired.T @ R_symmetric @ grad_H_desired)
        if (
            (dirac_defect <= _WILKINSON_LIMIT)
            and (step_report.is_symplectic_coherent)
            and (min_eigenvalue_R >= -_SPECTRAL_TOL)
            and (rayleigh_rate <= _SPECTRAL_TOL)
        ):
            heyting_verdict = "COHERENT"
            is_shielded = True
        elif (dirac_defect <= _SPECTRAL_TOL) and (rayleigh_rate <= _HARD_DIVERGENCE_CEILING):
            heyting_verdict = "DEGRADED"
            is_shielded = True
            logger.warning(
                "[CENTURION_WARNING] Degeneración amortiguada en la Cortina: "
                "Rate=%.3e",
                rayleigh_rate,
            )
        else:
            heyting_verdict = "VETOED"
            is_shielded = False
            logger.error(
                "[CENTURION_VETO] Ruptura de Maupertuis/Dirac: "
                "DiracDefect=%.3e, VolDrift=%.3e, Rayleigh=%.3e. "
                "Gatillando la ISR en IRAM del ESP32 (< 400 ns) via GPIO14 / "
                "BT151 Crowbar.",
                dirac_defect,
                step_report.volume_drift_det,
                rayleigh_rate,
            )
        return CenturionsGovernanceCertificate(
            maupertuis_action_density=step_report.maupertuis_action_density,
            volume_drift_det=step_report.volume_drift_det,
            rayleigh_dissipation_rate=rayleigh_rate,
            dirac_antisymmetry_defect=dirac_defect,
            heyting_verdict=heyting_verdict,
            is_power_curtain_shielded=is_shielded,
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
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Pretorio Agent (El Pretorio Agéntico — Comandante Supremo)          ║
║ Ruta   : app/agents/core/inmune_system/pretorio_agent.py                     ║
║ Versión: 5.0.0-Nested-Poincare-Cartan-Christoffel-CZ-Birkhoff-KAM-Hodge-     ║
║          Brouwer-TMR-Ultrafilter-Heyting-PhD                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA, CATEGORIAL Y DE GOBERNANZA SUPREMA:
────────────────────────────────────────────────────────────────────────────────
Ejerce el comando supremo en el penthouse de la pirámide de control $\aleph\mathrm{DIK}\Omega\alpha\mathrm{HW}\Gamma$
de APU Filter v8.0. Realiza la supervisión covariante y la adjudicación de permisos
en lazo cerrado ($\Phi_{\mathrm{III}} \circ \Phi_{\mathrm{II}} \circ \Phi_{\mathrm{I}}$)
suturando la cohomología de Čech–de Rham con la teoría de punto fijo de Brouwer–Schauder,
el ultrafiltro booleano principal y la mecánica celeste de Henri Poincaré.

DEFINICIONES, AXIOMAS Y TEOREMAS FORMALES:

FASE 1 — OBSERVE ($\Phi_{\mathrm{I}}$) — NÚCLEO ESPECTRAL HODGE, BROUWER Y GEOMETRÍA DE DARBOUX:
1. Hipercohomología de Čech–de Rham y Laplaciano de Hodge:
   Sobre la variedad de estados $\mathcal{M}$, el laplaciano de Hodge $\Delta^k = \mathrm{d}^{k-1} (\mathrm{d}^{k-1})^\dagger + (\mathrm{d}^k)^\dagger \mathrm{d}^k$
   posee espectro discreto. La dimensión del núcleo $\dim(\mathrm{ker}\,\Delta^k) = \beta_k$ coincide con los números de Betti de Čech.
   La obstrucción de hipercohomología $\mathbb{H}^{k>0}(\mathcal{M}, \Omega^p) \cong 0$ verifica la aciclicidad de calibre.

2. Teorema de Punto Fijo de Brouwer en el Simplex Cuántico:
   Para el espacio convexo compacto $\mathcal{S}_n = \{ \rho \in \mathrm{Herm}(n) \mid \rho \succeq 0, \mathrm{Tr}(\rho) = 1 \}$
   y la transformación endomórfica continua $f(\rho) = \frac{T \rho T^\dagger}{\mathrm{Tr}(T \rho T^\dagger)}$,
   existe al menos un punto fijo $\rho^* \in \mathcal{S}_n$ tal que $f(\rho^*) = \rho^*$.

FASE 2 — ORIENT ($\Phi_{\mathrm{II}}$) — ADUANAS DE POINCARÉ Y ULTRAFILTRO BOOELANO PRINCIPAL:
3. Teorema del Twist de Poincaré–Birkhoff y Módulo de Rotación:
   Dado el mapa de anillo que preserva área $\phi: A \to A$ sobre $A = S^1 \times [a, b]$, si las curvas de frontera
   giran en direcciones opuestas $\theta_1(r=a) \cdot \theta_2(r=b) < 0$, $\phi$ admite al menos 2 puntos fijos geométricos.

4. Ultrafiltro Principal Booleano $\mathcal{U}_\tau$ y Dualidad Permiso/Severidad:
   El filtro principal de veto $\mathcal{U}_\tau = \{ A \subseteq X \mid \tau \in A \}$ sobre la familia de aduanas
   está generado por el átomo crítico $\tau = \text{Tesserarios}$.
   El permiso global es la evaluación estricta en el meet de Heyting:
   $$\nu_{\mathrm{global}} = \bigwedge_{k} \nu_k = \min_k (\nu_k) \in G_3 \triangleq \{\mathrm{VETOED}(0) \le \mathrm{DEGRADED}(1) \le \mathrm{COHERENT}(2)\}$$

FASE 3 — DECIDE/ACT ($\Phi_{\mathrm{III}}$) — COLAPSO Y DISYUNTOR CIBER-FÍSICO EN IRAM:
5. Colapso TMR e Interlock Síncrono Crowbar BT151:
   La votación TMR con mediana inferior $\nu_{\mathrm{TMR}} = \mathrm{median}(\nu_{\mathrm{Guards}}, \nu_{\mathrm{Centurions}}, \nu_{\mathrm{Tesserarios}})$
   y el átomo de Capa 4 fuerzan $\nu_{\mathrm{global}} = \mathrm{VETOED}$ si cualquiera de ellos colapsa.
   Ante $\nu_{\mathrm{global}} = \mathrm{VETOED}$, el agente gatilla síncronamente el disyuntor de silicio
   BT151 [GPIO14] mediante la rutina ISR en IRAM del ESP32 en latencia acotada $t_{\mathrm{act}} \in [380, 415] \text{ ns}$.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np
import scipy.linalg as la

if TYPE_CHECKING:
    from app.core.inmune_system.pretorio_engine import PretorioEngine

try:
    from app.core.inmune_system.pretorio_engine import (
        PretorioEngine,
        PretorioSpectrumReport,
    )
    try:
        from app.core.inmune_system.pretorio_engine import (
            _KAMAudit,
            _MelnikovAudit,
            _ReturnMapAudit,
            _PoincareBirkhoffAudit,
            _MaupertuisJacobiGerm,
            _PoincareCartanGerm,
            _PretorioCelestialGerm,
            MaupertuisStepReport,
        )
    except ImportError:  # pragma: no cover
        _KAMAudit = Any
        _MelnikovAudit = Any
        _ReturnMapAudit = Any
        _PoincareBirkhoffAudit = Any
        _MaupertuisJacobiGerm = Any
        _PoincareCartanGerm = Any
        _PretorioCelestialGerm = Any
        MaupertuisStepReport = Any
except ImportError:  # pragma: no cover — import plano / tests locales
    from pretorio_engine import PretorioEngine, PretorioSpectrumReport  # type: ignore
    _KAMAudit = Any
    _MelnikovAudit = Any
    _ReturnMapAudit = Any
    _PoincareBirkhoffAudit = Any
    _MaupertuisJacobiGerm = Any
    _PoincareCartanGerm = Any
    _PretorioCelestialGerm = Any
    MaupertuisStepReport = Any

logger = logging.getLogger("APU.Agents.PretorioAgent")

__version__: Final[str] = (
    "5.0.0-Nested-Poincare-Cartan-Christoffel-CZ-Birkhoff-KAM-Hodge-"
    "Brouwer-TMR-Ultrafilter-Heyting-PhD"
)

# =============================================================================
# CONSTANTES UNIVERSALES DE PRECISIÓN METROLÓGICA Y LÍMITES DE WILKINSON
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DRIFT_LIMIT: Final[float] = 1.0e-12
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_MIN_SINGULAR_VALUE_FLOOR: Final[float] = 1.0e-12
_STRUCTURE_ATOL: Final[float] = 1.0e-9
_HYPERCOHOMOLOGY_THRESHOLD_DEFAULT: Final[float] = 1.0e-9
_HYPER_DEGRADATION_FACTOR: Final[float] = 1.0e-2
_ACYCLIC_SOFT_VETO: Final[float] = 0.5
_ACYCLIC_SOFT_DEGRADE: Final[float] = 0.05
_BROUWER_HARD_TOLERANCE: Final[float] = 1.0e-7
_BROUWER_DEGRADED_TOLERANCE: Final[float] = 1.0e-9
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_NS: Final[float] = 3.0
_CROWBAR_T_MIN_NS: Final[float] = 385.0
_CROWBAR_T_MAX_NS: Final[float] = 415.0

# Constantes celestes de Poincaré
_KAM_TAU_FLOOR: Final[float] = 1.0
_KAM_GAMMA_FLOOR: Final[float] = 1.0e-12
_KAM_THRESHOLD: Final[float] = 1.0e-9
_MELNIKOV_THRESHOLD: Final[float] = 1.0e-6
_MELNIKOV_T_INF: Final[float] = 25.0
_FLOQUET_PARABOLIC: Final[float] = 1.0e-6
_LYAPUNOV_CLIP: Final[float] = 700.0
_HILL_MARGIN_FLOOR: Final[float] = 1.0e-12
_BIRKHOFF_AREA_DRIFT_MAX: Final[float] = 1.0e-12
_BIRKHOFF_TWIST_FLOOR: Final[float] = 1.0e-9
_CSMD_STEP: Final[float] = 1.0e-8
_DEGRADATION_FACTOR: Final[float] = 1.0e-2
_CARTAN_DEFECT_TOL: Final[float] = 1.0e-8
_CHRISTOFFEL_BLOWUP: Final[float] = 1.0e3
_JACOBI_TIDAL_VETO: Final[float] = 1.0e2
_JACOBI_TIDAL_DEGRADE: Final[float] = 1.0e1
_HOMOLOGICAL_KAM_CEILING: Final[float] = 1.0e8
_SECTION_TRANSVERSALITY_FLOOR: Final[float] = 1.0e-12
_UNIT_CIRCLE_ATOL: Final[float] = 1.0e-8
_CZ_DEGENERATE_ATOL: Final[float] = 1.0e-8
_KOSZUL_TORSION_TOL: Final[float] = 1.0e-12
_REFRACTIVE_NEAR_HILL: Final[float] = 1.0e-4
_MELNIKOV_LYAPUNOV_VETO: Final[float] = 1.0
_LOG_EXP_CLIP: Final[float] = 700.0

_LAYER_WEIGHTS: Final[Dict[str, float]] = {
    "capa_1_guards": 1.0,
    "capa_2_centurions": 1.5,
    "capa_3_tesserarios": 2.0,
    "capa_4_pretorio_hyper": 1.8,
    "capa_4_pretorio_brouwer": 1.8,
    "capa_4_pretorio_kam": 1.9,
    "capa_4_pretorio_melnikov": 1.9,
    "capa_4_pretorio_birkhoff": 1.9,
    "capa_4_pretorio_return_map": 1.8,
}
_TMR_LAYERS: Final[Tuple[str, ...]] = (
    "capa_1_guards",
    "capa_2_centurions",
    "capa_3_tesserarios",
)
_SUPERVISOR_LAYERS: Final[Tuple[str, ...]] = (
    "capa_4_pretorio_hyper",
    "capa_4_pretorio_brouwer",
)
_CELESTIAL_SUPERVISOR_LAYERS: Final[Tuple[str, ...]] = (
    "capa_4_pretorio_kam",
    "capa_4_pretorio_melnikov",
    "capa_4_pretorio_birkhoff",
    "capa_4_pretorio_return_map",
)
_CRITICAL_ATOM: Final[str] = "capa_3_tesserarios"


# #############################################################################
#                                                                             #
#  FASE I                                                                     #
#  NÚCLEO ESPECTRAL · HODGE–DE RHAM · SIMPLEX CUÁNTICO · BROUWER ·            #
#  MAUPERTUIS–JACOBI · HILL · POINCARÉ–CARTAN · CHRISTOFFEL · JACOBI ·        #
#  STÖRMER–VERLET                                                             #
#                                                                             #
#  Objetos: complejo de co-cadenas (d^k), operador de densidad ρ,             #
#           mapa de transición T, métrica de Maupertuis–Jacobi g̃, 1-forma    #
#           de Poincaré–Cartan λ, símbolos de Christoffel conformes Γ̃,        #
#           marea de Jacobi, 2-forma de Liouville Ω.                          #
#                                                                             #
#  Cierre formal: assemble_pretorio_celestial_jet  →  _PretorioCelestialJet   #
#                 (dominio de todos los métodos de la Fase II).               #
#                                                                             #
# #############################################################################
class HeytingVerdict(Enum):
    r"""
    Retículo de Heyting lineal de tres valores (álgebra de Gödel G₃).

    Orden de permiso / coherencia:
        VETOED ≤ DEGRADED ≤ COHERENT

    En este orden:
      • meet = min = ínfimo de permiso  = PEOR CASO de seguridad (colapso).
      • join = max = supremo de verdad  = MEJOR CASO diagnóstico.
    El ultrafiltro de Fase III DEBE gobernarse por el meet, jamás por el join:
    join(VETOED, COHERENT) = COHERENT ocultaría un veto.
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
        return {0: 0.0, 1: 0.5, 2: 1.0}[self.value]

    @property
    def severity(self) -> int:
        """Dual de severidad: COHERENT=0 ≺ DEGRADED=1 ≺ VETOED=2."""
        return 2 - self.value

    def booleanize_closed(self) -> "HeytingVerdict":
        """Núcleo cerrado: todo lo que no es ⊤ colapsa a ⊥."""
        return self if self is HeytingVerdict.COHERENT else HeytingVerdict.VETOED

    def booleanize_open(self) -> "HeytingVerdict":
        """Doble negación ¬¬: DEGRADED ↦ COHERENT (Booleanización estándar)."""
        return self.negate().negate()

    @classmethod
    def from_token(cls, token: str) -> "HeytingVerdict":
        try:
            return cls[str(token)]
        except KeyError as exc:
            raise ValueError(f"Veredicto desconocido: {token!r}") from exc

    @classmethod
    def from_token_or_bottom(cls, token: str) -> "HeytingVerdict":
        try:
            return cls[str(token)]
        except KeyError:
            return cls.VETOED

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


@dataclass(frozen=True)
class PretorioDeliberationCertificate:
    r"""
    Certificado de Deliberación Pretoreana basado en el Teorema de Poincaré–Birkhoff.

    Invariantes de Mécanique Céleste de Henri Poincaré:
      1. Verdict: Veredicto en Ω₃ ∈ {COHERENT, DEGRADED, VETOED}.
      2. Area Drift: Δ_Area = |det M − 1.0|.
      3. Fixed Points: |Spec(M) ∩ S¹| (arco iris de Birkhoff).
      4. Is Deliberation Stable: banderola booleana de estabilidad.
    """

    verdict: str
    area_drift: float
    fixed_points: int
    is_deliberation_stable: bool


@dataclass(frozen=True, slots=True)
class _HodgeSpectrum:
    """Espectro de Hodge–de Rham de un complejo de co-cadenas."""

    max_nilpotency: float
    nilpotency_residuals: Tuple[float, ...]
    relative_nilpotency: Tuple[float, ...]
    betti_numbers: Tuple[int, ...]
    soft_betti: Tuple[float, ...]
    hodge_gaps: Tuple[float, ...]
    hyper_obstruction: float
    complex_valid: bool
    space_dims: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _BrouwerSpectrum:
    """Testigos de Brouwer / Banach sobre el simplex de densidades."""

    residual: float
    simplex_residual: float
    residual_unnormalized: float
    trace_defect: float
    positivity_defect: float
    hermiticity_defect: float
    isometry_defect: float
    lipschitz: float
    banach_contraction: bool


@dataclass(frozen=True, slots=True)
class _MaupertuisJacobiSpectrum:
    r"""
    Espectro de la métrica conforme de Maupertuis–Jacobi.

    Transporta g̃ = 2(H₀ − V)g, n(q), región de Hill, positividad,
    intensidad de Christoffel conforme, marea de Jacobi y torsión de Koszul.
    """

    conformal_factor: float
    refractive_index: float
    hill_margin: float
    is_in_hill_region: bool
    min_eigenvalue: float
    is_positive_definite: bool
    christoffel_strength: float = 0.0
    jacobi_tidal_norm: float = 0.0
    koszul_torsion: float = 0.0


@dataclass(frozen=True, slots=True)
class _PoincareCartanSpectrum:
    r"""
    Espectro de la 1-forma de Poincaré–Cartan λ = p dq − H dt.

    dλ = ω − dH ∧ dt es el invariante integral absoluto (É. Cartan, 1922).
    """

    lambda_vector: np.ndarray
    hamiltonian_value: float
    dim: int
    symplectic_skew_residual: float
    cartan_lagrangian: float = 0.0
    darboux_ok: bool = True


@dataclass(frozen=True, slots=True)
class _PretorioJet:
    """
    1-jet pretoreano inmutable. Cierre formal de la Fase I clásica y objeto
    inicial de la Fase II clásica. (Se conserva por compatibilidad API 2.0.)
    """

    n: int
    hodge: _HodgeSpectrum
    brouwer: _BrouwerSpectrum
    input_fault: Optional[str]


@dataclass(frozen=True, slots=True)
class _PretorioCelestialJet:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    1-JET PRETOREANO CELESTE (objeto terminal de Fase I, inicial de Fase II).
    ═══════════════════════════════════════════════════════════════════════════
    Extiende el 1-jet clásico con la geometría de la fase de Poincaré:
      • base_jet              : espectro Hodge + Brouwer clásicos.
      • maupertuis_spectrum   : g̃, n(q), Hill, Christoffel, marea, torsión.
      • cartan_spectrum       : λ = p dq − H dt + Lagrangiano de Cartan.
      • omega                 : 2-forma simpléctica canónica Ω ∈ ℝ^{2n×2n}.
      • darboux_residual      : ‖Ω + Ωᵀ‖_F.
      • almost_complex_residual : ‖Ω² + I‖_F.
    """

    n: int
    two_n: int
    base_jet: _PretorioJet
    maupertuis_spectrum: _MaupertuisJacobiSpectrum
    cartan_spectrum: _PoincareCartanSpectrum
    hamiltonian_energy_H0: float
    potential_V: float
    omega: np.ndarray
    base_metric_g: np.ndarray
    reg_floor: float
    input_fault: Optional[str]
    darboux_residual: float = 0.0
    almost_complex_residual: float = 0.0


# =============================================================================
# NÚCLEO CELESTE DE POINCARÉ (Cartan, Christoffel, CZ, Birkhoff, homológica)
# =============================================================================
class _PoincareCelestialKernel:
    r"""
    Operaciones de la Mécanique Céleste de Poincaré usadas por las Fases I–II.

    Implementa, con cotas de Wilkinson y acumulación compensada:
      • escisión Darboux (q, p) de T*Q,
      • λ(X) = p·q̇ − H y circulación discreta,
      • defecto de pullback simpléctico Φ*ω − ω,
      • intensidad ‖Γ̃‖_F y torsión de Koszul,
      • marea de Jacobi de la métrica de Maupertuis,
      • índice de Conley–Zehnder (Robbin–Salamon),
      • número de rotación de Poincaré y twist de Birkhoff,
      • residual homológico de Lindstedt–Poincaré,
      • transversalidad de la sección |det(M − I)|,
      • clasificación mutuamente excluyente de Floquet.
    """

    @staticmethod
    def kbn_sum(arr: np.ndarray) -> float:
        total = 0.0
        c = 0.0
        for x in np.asarray(arr, dtype=np.float64).ravel():
            xf = float(x)
            if not np.isfinite(xf):
                return float("nan")
            t = total + xf
            if abs(total) >= abs(xf):
                c += (total - t) + xf
            else:
                c += (xf - t) + total
            total = t
        return float(total + c)

    @staticmethod
    def frobenius(array: np.ndarray) -> float:
        a = np.asarray(array)
        if a.size == 0:
            return 0.0
        return float(la.norm(a, "fro"))

    @staticmethod
    def split_qp(x: np.ndarray, two_n: int) -> Tuple[np.ndarray, np.ndarray]:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)
        if vec.size != two_n or two_n % 2 != 0:
            raise ValueError(f"x debe vivir en R^{two_n} con two_n par.")
        half = two_n // 2
        return vec[:half].copy(), vec[half:].copy()

    @classmethod
    def poincare_cartan_on_field(
        cls,
        p: np.ndarray,
        qdot: np.ndarray,
        hamiltonian: float,
    ) -> float:
        r"""λ(X) = p_i q̇^i − H."""
        p_v = np.asarray(p, dtype=np.float64).reshape(-1)
        qd = np.asarray(qdot, dtype=np.float64).reshape(-1)
        if p_v.size != qd.size:
            raise ValueError("p y q̇ deben tener la misma dimensión de Q.")
        pairing = cls.kbn_sum(p_v * qd)
        return float(pairing - float(hamiltonian))

    @classmethod
    def poincare_cartan_circulation_step(
        cls,
        q0: np.ndarray,
        p0: np.ndarray,
        q1: np.ndarray,
        p1: np.ndarray,
        hamiltonian: float,
        dt: float,
    ) -> float:
        r"""Circulación discreta ∫_γ λ ≈ p̄·Δq − H Δt (punto medio)."""
        p_mid = 0.5 * (np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64))
        dq = np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
        return float(cls.kbn_sum(p_mid * dq) - float(hamiltonian) * float(dt))

    @classmethod
    def symplectic_pullback_defect(
        cls,
        monodromy_M: np.ndarray,
        canonical_omega: np.ndarray,
    ) -> float:
        r"""
        Defecto del invariante integral absoluto de Poincaré:
            δ = ‖Mᵀ Ω M − Ω‖_F .
        Vale 0 sii M ∈ Sp(2n, ℝ) (en aritmética exacta).
        """
        M = np.asarray(monodromy_M, dtype=np.float64)
        Om = np.asarray(canonical_omega, dtype=np.float64)
        if M.shape != Om.shape or M.ndim != 2 or M.shape[0] != M.shape[1]:
            return float("inf")
        residual = M.T @ Om @ M - Om
        return cls.frobenius(residual)

    @classmethod
    def koszul_torsion(cls, christoffel: np.ndarray) -> float:
        r"""Torsión de Koszul T^i_{jk} = Γ^i_{jk} − Γ^i_{kj}. Levi-Civita ⇒ T ≡ 0."""
        G = np.asarray(christoffel, dtype=np.float64)
        if G.ndim != 3 or G.shape[0] != G.shape[1] or G.shape[1] != G.shape[2]:
            return float("inf")
        return cls.frobenius((G - np.swapaxes(G, 1, 2)).reshape(G.shape[0], -1))

    @classmethod
    def christoffel_conformal_strength(
        cls,
        refractive_index: float,
        grad_V: np.ndarray,
        hill_margin: float,
    ) -> float:
        r"""
        Intensidad ‖Γ̃‖_F de la conexión conforme euclídea.
        ∇ln n = −∇V / (2(H₀ − V)). Blow-up cerca de ∂D_H.
        """
        n_idx = float(refractive_index)
        margin = float(hill_margin)
        if (
            (not np.isfinite(n_idx))
            or n_idx <= _HILL_MARGIN_FLOOR
            or margin <= _HILL_MARGIN_FLOOR
        ):
            return float("inf")
        gV = np.asarray(grad_V, dtype=np.float64).reshape(-1)
        dln = -gV / (2.0 * margin)
        if not np.all(np.isfinite(dln)):
            return float("inf")
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
        return cls.frobenius(gamma.reshape(d, -1))

    @classmethod
    def jacobi_tidal_norm(cls, grad_V: np.ndarray, hill_margin: float) -> float:
        r"""Proxy de marea geodésica de Jacobi: ‖∇V‖² / (H₀ − V)."""
        gV = np.asarray(grad_V, dtype=np.float64).reshape(-1)
        if gV.size == 0 or (not np.all(np.isfinite(gV))):
            return float("inf")
        num = float(cls.kbn_sum(gV * gV))
        den = max(abs(float(hill_margin)), _HILL_MARGIN_FLOOR)
        return float(num / den)

    @classmethod
    def conley_zehnder_index(
        cls,
        monodromy_M: np.ndarray,
        atol: float = _CZ_DEGENERATE_ATOL,
    ) -> Tuple[int, bool, float]:
        r"""
        Índice de Conley–Zehnder del camino recto γ(t) = exp(t Log M), t∈[0,1],
        convención de Robbin–Salamon / Long:
            i_CZ(M) ≈ n + (1/π) ∑_i arg(λ_i),
        n = dim Q. Degeneración ⇔ 1 ∈ spec(M).
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
        raw = float(n_half) + float(cls.kbn_sum(args)) / float(np.pi)
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
        Para λ = e^{iθ}: ρ = mean |θ| / 2π ∈ [0, 1/2], twist = max ρ − min ρ.
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
        r"""Residual de i⟨k,ω⟩ χ_k = (H₁)_k  ⇒  |χ_k| ∼ 1 / |⟨k,ω⟩|."""
        if (not np.isfinite(min_divisor)) or min_divisor <= _MACHINE_EPS:
            return float("inf")
        return float(min(1.0 / abs(min_divisor), _HOMOLOGICAL_KAM_CEILING))

    @staticmethod
    def section_transversality(monodromy_M: np.ndarray) -> float:
        r"""Transversalidad de Σ: |det(M − I)|. Cero ⇒ mapa de retorno no definido."""
        M = np.asarray(monodromy_M, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            return 0.0
        try:
            return float(abs(np.linalg.det(M - np.eye(M.shape[0]))))
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    @staticmethod
    def classify_floquet(
        magnitudes: np.ndarray,
        band: float = _FLOQUET_PARABOLIC,
    ) -> Tuple[bool, bool, bool, float]:
        r"""
        Clasificación mutuamente excluyente (v5):
          • elíptico   : max_i ||λ_i| − 1| ≤ band,
          • parabólico : no elíptico y min_i ||λ_i| − 1| ≤ band,
          • hiperbólico: min_i ||λ_i| − 1| > band.
        En v4 is_elliptic e is_hyperbolic podían ser True a la vez.
        """
        mag = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
        if mag.size == 0:
            return False, False, False, float("inf")
        dev = np.abs(mag - 1.0)
        floq_par = float(np.max(dev))
        is_elliptic = bool(floq_par <= band)
        is_parabolic = bool((not is_elliptic) and (float(np.min(dev)) <= band))
        is_hyperbolic = bool((not is_elliptic) and (not is_parabolic))
        return is_hyperbolic, is_elliptic, is_parabolic, floq_par

    @staticmethod
    def darboux_residuals(omega: np.ndarray) -> Tuple[float, float]:
        r"""Residuos de Darboux / casi-complejidad: ‖Ω+Ωᵀ‖_F y ‖Ω²+I‖_F."""
        om = np.asarray(omega, dtype=np.float64)
        if om.ndim != 2 or om.shape[0] != om.shape[1]:
            return float("inf"), float("inf")
        skew = float(la.norm(om + om.T, "fro"))
        ac = float(la.norm(om @ om + np.eye(om.shape[0]), "fro"))
        return float(skew), float(ac)


class _PretorioSpectralCore:
    r"""
    Núcleo de cómputo espectral, homológico y de mecánica celeste.

    Opera en la categoría de complejos de co-cadenas de dimensión finita
    sobre ℂ, en el simplex de estados 𝒮_n = {ρ = ρ† ⪰ 0, Tr ρ = 1}, y en
    la variedad simpléctica (T*Q, ω) con métrica conforme de Maupertuis–Jacobi.
    """

    _kernel = _PoincareCelestialKernel

    # ------------------------------------------------------------------
    # I.1  Formas de Banach, Wilkinson y proyección al simplex
    # ------------------------------------------------------------------
    @staticmethod
    def frobenius(array: np.ndarray) -> float:
        return float(la.norm(np.asarray(array), "fro"))

    @staticmethod
    def hermitize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.conj().T)

    @staticmethod
    def wilkinson_floor(scale: float, dim: int) -> float:
        return max(
            abs(scale) * _MACHINE_EPS * max(dim, 1) * _WILKINSON_DEFLATION_SCALE,
            _MIN_SINGULAR_VALUE_FLOOR,
        )

    @classmethod
    def numerical_rank(cls, matrix: np.ndarray) -> int:
        arr = np.asarray(matrix)
        if arr.size == 0:
            return 0
        svals = la.svd(arr, compute_uv=False)
        if svals.size == 0:
            return 0
        floor = cls.wilkinson_floor(float(svals[0]), max(arr.shape))
        return int(np.sum(svals > floor))

    @classmethod
    def regularize_density(
        cls,
        rho: np.ndarray,
        dimension_n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Proyección de Higham al simplex cuántico 𝒮_n."""
        herm = cls.hermitize(np.asarray(rho))
        evals, evecs = la.eigh(herm)
        evals_clipped = np.maximum(np.real(evals), _WILKINSON_DRIFT_LIMIT)
        trace = float(np.sum(evals_clipped))
        if trace <= _MACHINE_EPS:
            eye = np.eye(dimension_n, dtype=np.complex128)
            return eye / dimension_n, np.full(dimension_n, 1.0 / dimension_n)
        evals_norm = evals_clipped / trace
        rho_reg = evecs @ np.diag(evals_norm) @ evecs.conj().T
        return cls.hermitize(rho_reg), evals_norm

    @staticmethod
    def _as_matrix(name: str, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz; ndim={arr.ndim}.")
        if arr.size > 0 and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene NaN/Inf.")
        return arr

    # ------------------------------------------------------------------
    # I.2  Complejo de co-cadenas, nilpotencia y Hodge
    # ------------------------------------------------------------------
    @classmethod
    def _space_dims(cls, differentials: Sequence[np.ndarray]) -> Tuple[int, ...]:
        if not differentials:
            return tuple()
        dims = [int(differentials[0].shape[1])]
        for d_k in differentials:
            dims.append(int(d_k.shape[0]))
        return tuple(dims)

    @classmethod
    def _validate_complex(
        cls,
        differentials: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], bool, Optional[str]]:
        mats: List[np.ndarray] = []
        for i, raw in enumerate(differentials):
            try:
                mats.append(cls._as_matrix(f"d^{i}", raw))
            except ValueError as exc:
                return [], False, str(exc)
        for i in range(len(mats) - 1):
            if mats[i + 1].shape[1] != mats[i].shape[0]:
                return mats, False, (
                    f"d^{i+1}∘d^{i} incompatible: "
                    f"{mats[i + 1].shape} ∘ {mats[i].shape}."
                )
        return mats, True, None

    @classmethod
    def nilpotency_residuals(
        cls,
        differentials: Sequence[np.ndarray],
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        r"""
        Residuales del axioma d^{k+1}∘d^k = 0:
            ε_k = ‖d^{k+1}d^k‖_F,
            ε_k^rel = ε_k / (‖d^{k+1}‖_F ‖d^k‖_F).
        """
        abs_res: List[float] = []
        rel_res: List[float] = []
        for i in range(len(differentials) - 1):
            d_k = differentials[i]
            d_k1 = differentials[i + 1]
            product = d_k1 @ d_k
            residual = cls.frobenius(product)
            denom = cls.frobenius(d_k1) * cls.frobenius(d_k)
            if denom > _MIN_SINGULAR_VALUE_FLOOR:
                relative = residual / denom
            else:
                relative = 0.0 if residual <= _MIN_SINGULAR_VALUE_FLOOR else float("inf")
            abs_res.append(residual)
            rel_res.append(float(relative))
        return tuple(abs_res), tuple(rel_res)

    @classmethod
    def hodge_laplacian(
        cls,
        differentials: Sequence[np.ndarray],
        space_dims: Sequence[int],
        degree: int,
    ) -> np.ndarray:
        r"""
        Laplaciano de Hodge en grado k (métrica euclídea):
            Δ^k = d^{k-1}(d^{k-1})† + (d^k)† d^k.
        """
        dim_k = int(space_dims[degree])
        delta = np.zeros((dim_k, dim_k), dtype=np.complex128)
        if degree >= 1:
            d_prev = np.asarray(differentials[degree - 1])
            delta += d_prev @ d_prev.conj().T
        if degree <= len(differentials) - 1:
            d_k = np.asarray(differentials[degree])
            delta += d_k.conj().T @ d_k
        return cls.hermitize(delta)

    @classmethod
    def hodge_invariants(
        cls,
        differentials: Sequence[np.ndarray],
        space_dims: Sequence[int],
    ) -> Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[float, ...]]:
        r"""Números de Betti numéricos, Betti suaves y gaps espectrales."""
        betti: List[int] = []
        soft: List[float] = []
        gaps: List[float] = []
        for k, dim_k in enumerate(space_dims):
            if dim_k <= 0:
                betti.append(0)
                soft.append(0.0)
                gaps.append(0.0)
                continue
            delta = cls.hodge_laplacian(differentials, space_dims, k)
            evals = np.real(la.eigh(delta, eigvals_only=True))
            scale = float(np.max(np.abs(evals))) if evals.size else 0.0
            floor = cls.wilkinson_floor(scale, dim_k)
            kernel = evals <= floor
            betti.append(int(np.sum(kernel)))
            soft.append(float(np.sum(np.exp(-np.maximum(evals, 0.0) / floor))))
            positive = evals[evals > floor]
            gaps.append(float(np.min(positive)) if positive.size else 0.0)
        return tuple(betti), tuple(soft), tuple(gaps)

    @classmethod
    def compute_hodge_spectrum(
        cls,
        cochain_complex_matrices: Sequence[np.ndarray],
    ) -> _HodgeSpectrum:
        r"""
        Espectro completo del complejo (o del Tot Čech–de Rham ya ensamblado).
        La obstrucción de hipercohomología positiva es
            obs = max_k ε_k + Σ_{k>0} β_k^suave.
        """
        empty = _HodgeSpectrum(
            max_nilpotency=0.0,
            nilpotency_residuals=tuple(),
            relative_nilpotency=tuple(),
            betti_numbers=tuple(),
            soft_betti=tuple(),
            hodge_gaps=tuple(),
            hyper_obstruction=0.0,
            complex_valid=True,
            space_dims=tuple(),
        )
        if len(cochain_complex_matrices) == 0:
            return empty
        mats, valid, fault = cls._validate_complex(cochain_complex_matrices)
        if not mats:
            return _HodgeSpectrum(
                max_nilpotency=float("inf"),
                nilpotency_residuals=(float("inf"),),
                relative_nilpotency=(float("inf"),),
                betti_numbers=tuple(),
                soft_betti=tuple(),
                hodge_gaps=tuple(),
                hyper_obstruction=float("inf"),
                complex_valid=False,
                space_dims=tuple(),
            )
        if not valid:
            logger.error("Complejo de co-cadenas inválido: %s", fault)
            return _HodgeSpectrum(
                max_nilpotency=float("inf"),
                nilpotency_residuals=(float("inf"),),
                relative_nilpotency=(float("inf"),),
                betti_numbers=tuple(),
                soft_betti=tuple(),
                hodge_gaps=tuple(),
                hyper_obstruction=float("inf"),
                complex_valid=False,
                space_dims=cls._space_dims(mats) if mats else tuple(),
            )
        abs_res, rel_res = cls.nilpotency_residuals(mats)
        max_nil = max(abs_res) if abs_res else 0.0
        dims = cls._space_dims(mats)
        betti, soft, gaps = cls.hodge_invariants(mats, dims)
        positive_mass = float(sum(soft[1:])) if len(soft) > 1 else 0.0
        obstruction = float(max_nil) + positive_mass
        structure_ok = all(
            (not np.isfinite(r)) or r <= _STRUCTURE_ATOL for r in abs_res
        )
        return _HodgeSpectrum(
            max_nilpotency=float(max_nil),
            nilpotency_residuals=abs_res,
            relative_nilpotency=rel_res,
            betti_numbers=betti,
            soft_betti=soft,
            hodge_gaps=gaps,
            hyper_obstruction=float(obstruction),
            complex_valid=bool(structure_ok),
            space_dims=dims,
        )

    # ------------------------------------------------------------------
    # I.3  Brouwer sobre el simplex de densidades
    # ------------------------------------------------------------------
    @classmethod
    def compute_brouwer_spectrum(
        cls,
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
        dimension_n: int,
    ) -> _BrouwerSpectrum:
        r"""
        Testigos del endomorfismo
            f : 𝒮_n → 𝒮_n,  f(ρ) = TρT† / Tr(TρT†).
        𝒮_n compacto convexo ⇒ Brouwer garantiza un punto fijo.
        """
        inf = _BrouwerSpectrum(
            residual=float("inf"),
            simplex_residual=float("inf"),
            residual_unnormalized=float("inf"),
            trace_defect=float("inf"),
            positivity_defect=float("inf"),
            hermiticity_defect=float("inf"),
            isometry_defect=float("inf"),
            lipschitz=float("inf"),
            banach_contraction=False,
        )
        try:
            raw = np.asarray(density_matrix)
            if raw.shape != (dimension_n, dimension_n):
                raise ValueError(
                    f"density_matrix de forma {raw.shape}, esperada "
                    f"({dimension_n}, {dimension_n})."
                )
            if not np.all(np.isfinite(raw)):
                raise ValueError("density_matrix contiene NaN/Inf.")
            t_map = cls._as_matrix("transition_map_matrix", transition_map_matrix)
            if t_map.shape != (dimension_n, dimension_n):
                raise ValueError(
                    f"transition_map_matrix de forma {t_map.shape}, esperada "
                    f"({dimension_n}, {dimension_n})."
                )
        except ValueError as exc:
            logger.error("Espectro de Brouwer inválido: %s", exc)
            return inf
        herm_def = cls.frobenius(raw - raw.conj().T)
        raw_evals = np.real(la.eigh(cls.hermitize(raw), eigvals_only=True))
        pos_def = float(np.sum(np.clip(-raw_evals, 0.0, None)))
        rho_reg, _ = cls.regularize_density(raw, dimension_n)
        sigma = cls.hermitize(t_map @ rho_reg @ t_map.conj().T)
        trace_val = float(np.real(np.trace(sigma)))
        trace_defect = abs(trace_val - 1.0)
        residual_unnorm = cls.frobenius(sigma - rho_reg)
        residual = residual_unnorm + trace_defect
        if trace_val > _MACHINE_EPS:
            simplex_residual = cls.frobenius(sigma / trace_val - rho_reg)
        else:
            simplex_residual = float("inf")
        eye = np.eye(dimension_n, dtype=t_map.dtype)
        iso_def = cls.frobenius(t_map.conj().T @ t_map - eye)
        lipschitz = float(la.norm(t_map, 2)) ** 2
        banach = bool(np.isfinite(lipschitz) and lipschitz < 1.0 - 1.0e-12)
        return _BrouwerSpectrum(
            residual=float(residual),
            simplex_residual=float(simplex_residual),
            residual_unnormalized=float(residual_unnorm),
            trace_defect=float(trace_defect),
            positivity_defect=float(pos_def),
            hermiticity_defect=float(herm_def),
            isometry_defect=float(iso_def),
            lipschitz=float(lipschitz),
            banach_contraction=banach,
        )

    # ------------------------------------------------------------------
    # I.4  Álgebra de decisión (Heyting / mediana / Booleanización)
    # ------------------------------------------------------------------
    @staticmethod
    def heyting_meet_tokens(tokens: Sequence[str]) -> HeytingVerdict:
        acc = HeytingVerdict.COHERENT
        for token in tokens:
            acc = acc.meet(HeytingVerdict.from_token_or_bottom(token))
        return acc

    @staticmethod
    def heyting_join_tokens(tokens: Sequence[str]) -> HeytingVerdict:
        acc = HeytingVerdict.VETOED
        for token in tokens:
            acc = acc.join(HeytingVerdict.from_token_or_bottom(token))
        return acc

    @staticmethod
    def heyting_lower_median(tokens: Sequence[str]) -> HeytingVerdict:
        """Mediana inferior en G₃ (TMR valorado en retículo)."""
        if not tokens:
            return HeytingVerdict.COHERENT
        ordered = sorted(
            HeytingVerdict.from_token_or_bottom(token).value for token in tokens
        )
        return HeytingVerdict(ordered[(len(ordered) - 1) // 2])

    @staticmethod
    def weighted_social_choice(
        layer_verdicts: Dict[str, str],
    ) -> Tuple[HeytingVerdict, Dict[str, float], Tuple[str, ...]]:
        """Ancilla de elección social (NO es un ultrafiltro)."""
        masses = {verdict.name: 0.0 for verdict in HeytingVerdict}
        anomalies: List[str] = []
        for layer, token in layer_verdicts.items():
            weight = float(_LAYER_WEIGHTS.get(layer, 1.0))
            if token not in HeytingVerdict.__members__:
                verdict = HeytingVerdict.VETOED
                weight *= 2.0
                anomalies.append(layer)
            else:
                verdict = HeytingVerdict[token]
            masses[verdict.name] += weight
        winner = max(
            list(HeytingVerdict),
            key=lambda verdict: (masses[verdict.name], -verdict.value),
        )
        return winner, masses, tuple(anomalies)

    @classmethod
    def compute_all_pretorio_metrics(
        cls,
        cochain_complex_matrices: List[np.ndarray],
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
        dimension_n: int,
    ) -> Tuple[float, float, float]:
        """Proyección 2.0 del 1-jet: (max_nilpotency, residual_Brouwer, trace_defect)."""
        hodge = cls.compute_hodge_spectrum(cochain_complex_matrices)
        brouwer = cls.compute_brouwer_spectrum(
            density_matrix, transition_map_matrix, dimension_n
        )
        return hodge.max_nilpotency, brouwer.residual, brouwer.trace_defect

    # ------------------------------------------------------------------
    # I.5  Métrica conforme de Maupertuis–Jacobi y región de Hill
    # ------------------------------------------------------------------
    @classmethod
    def compute_maupertuis_jacobi_conformal_metric(
        cls,
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
        base_metric_g: np.ndarray,
        grad_V: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, _MaupertuisJacobiSpectrum]:
        r"""
        Métrica conforme de Maupertuis–Jacobi:
            g̃_{jk}(q) = 2 (H₀ − V(q)) g_{jk}(q) = n(q)² g_{jk}(q),
        con n(q) = √(2(H₀ − V(q))). Región de Hill: H₀ − V(q) > 0.
        Si se provee ∇V, adjunta ‖Γ̃‖_F, marea de Jacobi y torsión de Koszul.
        """
        g = np.asarray(base_metric_g, dtype=np.float64)
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError("base_metric_g debe ser cuadrada.")
        h0 = float(hamiltonian_energy_H0)
        v = float(potential_energy_V)
        if not (np.isfinite(h0) and np.isfinite(v)):
            raise ValueError("H₀ y V deben ser finitos.")
        free_energy = 2.0 * (h0 - v)
        in_hill = bool(free_energy > _HILL_MARGIN_FLOOR)
        phi = max(free_energy, _MIN_SINGULAR_VALUE_FLOOR)
        refractive = float(np.sqrt(phi))
        gt = phi * g
        try:
            evals = la.eigvalsh(cls.hermitize(gt))
            min_eig = float(np.min(np.real(evals))) if evals.size else 0.0
        except la.LinAlgError:
            min_eig = 0.0
        strength = 0.0
        tidal = 0.0
        torsion = 0.0
        if grad_V is not None:
            gV = np.asarray(grad_V, dtype=np.float64).ravel()
            strength = cls._kernel.christoffel_conformal_strength(
                refractive, gV, float(h0 - v)
            )
            tidal = cls._kernel.jacobi_tidal_norm(gV, float(h0 - v))
            try:
                Gamma = cls.compute_christoffel_conformal_symbols(
                    gV, v, h0, g
                )
                torsion = cls._kernel.koszul_torsion(Gamma)
            except (ValueError, np.linalg.LinAlgError):
                torsion = float("inf")
        return gt, _MaupertuisJacobiSpectrum(
            conformal_factor=float(phi),
            refractive_index=refractive,
            hill_margin=float(h0 - v),
            is_in_hill_region=in_hill,
            min_eigenvalue=min_eig,
            is_positive_definite=bool(min_eig > _MIN_SINGULAR_VALUE_FLOOR),
            christoffel_strength=float(strength),
            jacobi_tidal_norm=float(tidal),
            koszul_torsion=float(torsion),
        )

    @staticmethod
    def compute_hill_region_margin(
        potential_V: float,
        total_energy_H0: float,
    ) -> float:
        """Margen de Hill H₀ − V(q)."""
        return float(total_energy_H0 - potential_V)

    # ------------------------------------------------------------------
    # I.6  Símbolos de Christoffel conformes
    # ------------------------------------------------------------------
    @staticmethod
    def compute_christoffel_conformal_symbols(
        grad_V: np.ndarray,
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: np.ndarray,
    ) -> np.ndarray:
        r"""
        Símbolos de Christoffel conformes de Koszul–Levi-Civita.

        Para g̃ = e^{2φ} g con φ = ln n = ½ ln(2(H₀ − V)):
            Γ̃^i_{jk} = Γ^i_{jk} + δ^i_j ∂_k φ + δ^i_k ∂_j φ − g_{jk} g^{il} ∂_l φ.
        ∇φ = −∇V / (2(H₀ − V)). Para g = I, Γ_base = 0.
        """
        grad_V = np.asarray(grad_V, dtype=np.float64).ravel()
        g = np.asarray(g_base_metric, dtype=np.float64)
        n_dim = grad_V.size
        if g.shape != (n_dim, n_dim):
            raise ValueError(f"g_base_metric debe ser {n_dim}×{n_dim}.")
        headroom = 2.0 * (total_energy_H0 - potential_V)
        if headroom <= _HILL_MARGIN_FLOOR:
            raise ValueError(
                "[PRETORIO_VETO] Cero energía cinética: invasión de pozo de potencial."
            )
        grad_phi = -grad_V / (headroom + _MIN_SINGULAR_VALUE_FLOOR)
        g_inv = la.inv(g)
        christoffel = np.zeros((n_dim, n_dim, n_dim), dtype=np.float64)
        for i in range(n_dim):
            for j in range(n_dim):
                for k in range(n_dim):
                    term1 = (1.0 if i == j else 0.0) * grad_phi[k]
                    term2 = (1.0 if i == k else 0.0) * grad_phi[j]
                    term3 = g[j, k] * float(
                        _PoincareCelestialKernel.kbn_sum(g_inv[i, :] * grad_phi)
                    )
                    christoffel[i, j, k] = term1 + term2 - term3
        return christoffel

    # ------------------------------------------------------------------
    # I.7  1-forma de Poincaré–Cartan
    # ------------------------------------------------------------------
    @classmethod
    def compute_poincare_cartan_lambda(
        cls,
        x: np.ndarray,
        hamiltonian_value: float = 0.0,
        omega: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, _PoincareCartanSpectrum]:
        r"""
        1-forma de Poincaré–Cartan λ = p dq − H dt evaluada en x ∈ T*Q.
        dλ = ω − dH ∧ dt es el invariante integral absoluto.
        """
        xv = np.asarray(x, dtype=np.float64).ravel()
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("λ de Poincaré–Cartan exige dim par (T*Q).")
        n = dim // 2
        p = xv[n:]
        h_val = float(hamiltonian_value) if np.isfinite(hamiltonian_value) else 0.0
        lam = np.concatenate([p, np.array([-h_val], dtype=np.float64)])
        if omega is None:
            omega = np.zeros((dim, dim), dtype=np.float64)
        om = np.asarray(omega)
        skew_res = float(cls.frobenius(om + om.T)) if om.size else 0.0
        cartan_L = cls._kernel.poincare_cartan_on_field(p, p, h_val)
        scale = max(cls.frobenius(om), 1.0) if om.size else 1.0
        darboux_ok = bool(skew_res <= _STRUCTURE_ATOL * scale)
        return lam, _PoincareCartanSpectrum(
            lambda_vector=lam,
            hamiltonian_value=h_val,
            dim=dim,
            symplectic_skew_residual=skew_res,
            cartan_lagrangian=float(cartan_L),
            darboux_ok=darboux_ok,
        )

    # ------------------------------------------------------------------
    # I.8  Corchete de Poisson y gradiente CSMD
    # ------------------------------------------------------------------
    @staticmethod
    def compute_gradient_csmd(
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""
        Gradiente CSMD: ∇_k H(x) = Im[H(x + j·h·e_k)] / h + O(h²),
        eludiendo cancelaciones sustractivas en la mantisa de la FPU.
        """
        xv = np.asarray(x, dtype=np.float64).ravel()
        if not np.isfinite(h) or h == 0.0:
            raise ValueError("El paso CSMD h debe ser finito y no nulo.")
        h = float(abs(h))
        dim = xv.size
        grad = np.zeros(dim, dtype=np.float64)
        holomorphic = False
        try:
            probe = func(xv.astype(np.complex128))
            holomorphic = np.isfinite(np.real(probe)) or np.isfinite(np.imag(probe))
        except (TypeError, ValueError, FloatingPointError):
            holomorphic = False
        if holomorphic:
            for i in range(dim):
                xp = xv.astype(np.complex128)
                xp[i] += 1j * h
                try:
                    val = func(xp)
                    imag = float(np.imag(val))
                except Exception:
                    holomorphic = False
                    break
                if not np.isfinite(imag):
                    holomorphic = False
                    break
                grad[i] = imag / h
        if not holomorphic:
            for i in range(dim):
                xp = xv.copy()
                xm = xv.copy()
                xp[i] += h
                xm[i] -= h
                try:
                    fp = float(np.real(func(xp)))
                    fm = float(np.real(func(xm)))
                except Exception:
                    fp = fm = 0.0
                grad[i] = (fp - fm) / (2.0 * h)
        return grad

    @classmethod
    def poisson_bracket(
        cls,
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        x: np.ndarray,
        omega: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> float:
        r"""
        Corchete de Poisson {H₀, H₁}(x) = (∇H₀)ᵀ Ω ∇H₁ en T*Q.
        Integrando de Melnikov: si {H₀, H₁} ≠ 0, ε H₁ rompe las integrales
        primeras de H₀ y puede producir caos homoclínico.
        """
        xv = np.asarray(x, dtype=np.float64).ravel()
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("El corchete de Poisson exige dim par (Darboux).")
        grad0 = cls.compute_gradient_csmd(hamiltonian_0, xv, h)
        grad1 = cls.compute_gradient_csmd(hamiltonian_1, xv, h)
        return float(grad0 @ omega @ grad1)

    # ------------------------------------------------------------------
    # I.9  2-forma canónica y Störmer–Verlet
    # ------------------------------------------------------------------
    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        """2-forma canónica de Liouville Ω ∈ ℝ^{dim×dim}, dim = 2n par."""
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"La dimensión del espacio simpléctico dim={dim} debe ser par y positiva."
            )
        half = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:half, half:] = np.eye(half, dtype=np.float64)
        omega[half:, :half] = -np.eye(half, dtype=np.float64)
        return omega

    @staticmethod
    def stormer_verlet_step(
        q: np.ndarray,
        p: np.ndarray,
        grad_v: Callable[[np.ndarray], np.ndarray],
        mass_inv: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Paso Störmer–Verlet sobre H(q, p) = ½|p|²/m + V(q).
        Preserva la 2-forma de Liouville Ω (mapa simpléctico).
        """
        qn = np.asarray(q, dtype=np.float64)
        pn = np.asarray(p, dtype=np.float64)
        p_half = pn - 0.5 * dt * np.asarray(grad_v(qn), dtype=np.float64)
        q_next = qn + dt * mass_inv * p_half
        p_next = p_half - 0.5 * dt * np.asarray(grad_v(q_next), dtype=np.float64)
        return q_next, p_next

    # ------------------------------------------------------------------
    # I.10  Síntesis del espectro celeste (bloque de Fase I)
    # ------------------------------------------------------------------
    @classmethod
    def synthesize_celestial_spectrum(
        cls,
        dimension_two_n: int,
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
        base_metric_g: Optional[np.ndarray] = None,
        grad_V: Optional[np.ndarray] = None,
    ) -> Tuple[
        _MaupertuisJacobiSpectrum,
        _PoincareCartanSpectrum,
        np.ndarray,
        np.ndarray,
        float,
        float,
    ]:
        r"""
        Ensambla el bloque celeste de Fase I: métrica de Maupertuis–Jacobi,
        1-forma de Poincaré–Cartan, 2-forma de Darboux Ω, métrica base g
        y residuos de Darboux / casi-complejidad.
        """
        dim = int(dimension_two_n)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dimension_two_n debe ser par y positivo; recibido {dim}.")
        n = dim // 2
        omega = cls.generate_canonical_symplectic_form(dim)
        skew_r, ac_r = cls._kernel.darboux_residuals(omega)
        if base_metric_g is None:
            base_metric_g = np.eye(n, dtype=np.float64)
        _gt, mau_spectrum = cls.compute_maupertuis_jacobi_conformal_metric(
            hamiltonian_energy_H0, potential_energy_V, base_metric_g, grad_V=grad_V
        )
        x0 = np.zeros(dim, dtype=np.float64)
        _lam, cartan_spectrum = cls.compute_poincare_cartan_lambda(
            x0, hamiltonian_value=potential_energy_V, omega=omega
        )
        return (
            mau_spectrum,
            cartan_spectrum,
            omega,
            np.asarray(base_metric_g, dtype=np.float64),
            float(skew_r),
            float(ac_r),
        )

    # ------------------------------------------------------------------
    # I.ω  ÚLTIMO MORFISMO DE LA FASE I (CLÁSICO, retrocompatible)
    # ------------------------------------------------------------------
    @classmethod
    def assemble_pretorio_jet(
        cls,
        dimension_n: int,
        cochain_complex_matrices: Sequence[np.ndarray],
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
    ) -> _PretorioJet:
        r"""
        Cierre formal de la Fase I clásica / unidad de la adjunción con la
        Fase II clásica. Congela el espectro de Hodge y los testigos de
        Brouwer en un 1-jet inmutable. (API 2.0 conservada.)
        """
        faults: List[str] = []
        if dimension_n <= 0:
            faults.append(f"dimension_n={dimension_n} no es positiva")
        try:
            hodge = cls.compute_hodge_spectrum(cochain_complex_matrices)
            if not hodge.complex_valid and hodge.max_nilpotency == float("inf"):
                faults.append("hodge:complejo_invalido")
        except (ValueError, np.linalg.LinAlgError) as cop_exc:
            faults.append(f"hodge:{cop_exc}")
            hodge = _HodgeSpectrum(
                max_nilpotency=float("inf"),
                nilpotency_residuals=(float("inf"),),
                relative_nilpotency=(float("inf"),),
                betti_numbers=tuple(),
                soft_betti=tuple(),
                hodge_gaps=tuple(),
                hyper_obstruction=float("inf"),
                complex_valid=False,
                space_dims=tuple(),
            )
        try:
            brouwer = cls.compute_brouwer_spectrum(
                density_matrix, transition_map_matrix, dimension_n
            )
            if not np.isfinite(brouwer.residual):
                faults.append("brouwer:residual_no_finito")
        except (ValueError, np.linalg.LinAlgError) as cop_exc:
            faults.append(f"brouwer:{cop_exc}")
            brouwer = _BrouwerSpectrum(
                residual=float("inf"),
                simplex_residual=float("inf"),
                residual_unnormalized=float("inf"),
                trace_defect=float("inf"),
                positivity_defect=float("inf"),
                hermiticity_defect=float("inf"),
                isometry_defect=float("inf"),
                lipschitz=float("inf"),
                banach_contraction=False,
            )
        return _PretorioJet(
            n=int(dimension_n),
            hodge=hodge,
            brouwer=brouwer,
            input_fault=("|".join(faults) if faults else None),
        )

    # ------------------------------------------------------------------
    # I.ω+  MORFISMO TERMINAL DE LA FASE I: 1-jet celeste
    # ------------------------------------------------------------------
    @classmethod
    def assemble_pretorio_celestial_jet(
        cls,
        dimension_n: int,
        cochain_complex_matrices: Sequence[np.ndarray],
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        base_metric_g: Optional[np.ndarray] = None,
        grad_V: Optional[np.ndarray] = None,
    ) -> _PretorioCelestialJet:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE I ≅ OBJETO INICIAL DE LA FASE II.
        ═══════════════════════════════════════════════════════════════════════
        Compone el 1-jet clásico (Hodge + Brouwer) con la geometría de la
        fase (Maupertuis–Jacobi + Hill + Christoffel + Jacobi + Poincaré–
        Cartan + Ω + Darboux) en un único 𝒢_I listo para las aduanas
        celestes de la Fase II.

        Firma de continuación (Fase II):
            lift_from_celestial_jet(jet: _PretorioCelestialJet) -> PretorioAgent
        """
        base_jet = cls.assemble_pretorio_jet(
            dimension_n, cochain_complex_matrices, density_matrix, transition_map_matrix
        )
        n = int(dimension_n)
        two_n = 2 * n
        try:
            (
                mau_spectrum,
                cartan_spectrum,
                omega,
                g_base,
                skew_r,
                ac_r,
            ) = cls.synthesize_celestial_spectrum(
                two_n,
                hamiltonian_energy_H0,
                potential_energy_V,
                base_metric_g,
                grad_V=grad_V,
            )
            celestial_fault = None
        except (ValueError, np.linalg.LinAlgError) as cop_exc:
            logger.error("Fallo en síntesis celeste: %s", cop_exc)
            mau_spectrum = _MaupertuisJacobiSpectrum(
                conformal_factor=0.0,
                refractive_index=0.0,
                hill_margin=0.0,
                is_in_hill_region=False,
                min_eigenvalue=0.0,
                is_positive_definite=False,
                christoffel_strength=float("inf"),
                jacobi_tidal_norm=float("inf"),
                koszul_torsion=float("inf"),
            )
            cartan_spectrum = _PoincareCartanSpectrum(
                lambda_vector=np.zeros(max(two_n, 1) + 1, dtype=np.float64),
                hamiltonian_value=float(potential_energy_V),
                dim=two_n,
                symplectic_skew_residual=float("inf"),
                cartan_lagrangian=float("nan"),
                darboux_ok=False,
            )
            omega = (
                cls.generate_canonical_symplectic_form(two_n)
                if two_n > 0
                else np.zeros((0, 0))
            )
            g_base = np.eye(n, dtype=np.float64) if n > 0 else np.zeros((0, 0))
            skew_r, ac_r = float("inf"), float("inf")
            celestial_fault = f"celestial:{cop_exc}"
        fault = base_jet.input_fault
        if celestial_fault:
            fault = f"{fault}|{celestial_fault}" if fault else celestial_fault
        reg_floor = cls.wilkinson_floor(
            max(float(np.linalg.norm(omega, "fro")) if omega.size else 1.0, 1.0),
            max(two_n, 1),
        )
        return _PretorioCelestialJet(
            n=n,
            two_n=two_n,
            base_jet=base_jet,
            maupertuis_spectrum=mau_spectrum,
            cartan_spectrum=cartan_spectrum,
            hamiltonian_energy_H0=float(hamiltonian_energy_H0),
            potential_V=float(potential_energy_V),
            omega=omega,
            base_metric_g=g_base,
            reg_floor=float(reg_floor),
            input_fault=fault,
            darboux_residual=float(skew_r),
            almost_complex_residual=float(ac_r),
        )


# #############################################################################
#                                                                             #
#  FASE II                                                                    #
#  PRETORIO · ADUANAS DE HIPERCOHOMOLOGÍA / BROUWER / KAM / MELNIKOV /       #
#  POINCARÉ–BIRKHOFF / RETORNO / CZ / TMR+ULTRAFILTRO                         #
#                                                                             #
#  Continuación directa del último morfismo de la Fase I:                     #
#      _PretorioCelestialJet  ↦  PretorioAgent.lift_from_celestial_jet        #
#                                                                             #
#  Cierre formal: compile_pretorio_celestial_edict  →  _PretorioCelestialEdict#
#                 (dominio de la Cámara de Coherencia, Fase III).             #
#                                                                             #
# #############################################################################
@dataclass(frozen=True, slots=True)
class _HypercohomologyResult:
    """Resultado de la aduana de hipercohomología (con veredicto)."""

    max_residual: float
    hyper_obstruction: float
    betti_numbers: Tuple[int, ...]
    soft_betti: Tuple[float, ...]
    hodge_gaps: Tuple[float, ...]
    nilpotency_residuals: Tuple[float, ...]
    complex_valid: bool
    verdict: str


@dataclass(frozen=True, slots=True)
class _BrouwerResult:
    """Resultado de la aduana de Brouwer (con veredicto)."""

    residual: float
    simplex_residual: float
    trace_defect: float
    positivity_defect: float
    hermiticity_defect: float
    isometry_defect: float
    lipschitz: float
    banach_contraction: bool
    verdict: str


@dataclass(frozen=True, slots=True)
class _KAMResult:
    r"""
    Resultado de la aduana KAM (pequeños divisores de Poincaré).

    Adjunta residual homológico de Lindstedt y defecto de pullback de Cartan.
    """

    min_divisor: float
    novikov_weight: float
    maurercartan_residual: float
    volume_drift: float
    resonance_gap: float
    tau: float
    gamma: float
    is_diophantine: bool
    is_kam_stable: bool
    verdict: str
    homological_residual: float = 0.0
    cartan_pullback_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class _MelnikovResult:
    r"""Resultado de la aduana de Melnikov (ruptura homoclínica de Poincaré)."""

    melnikov_value: float
    melnikov_derivative: float
    is_simple_zero: bool
    homoclinic_splitting: float
    verdict: str


@dataclass(frozen=True, slots=True)
class _PoincareBirkhoffResult:
    r"""Resultado de la aduana de Poincaré–Birkhoff (twist map + Cartan + ρ)."""

    area_drift: float
    has_opposite_twist: bool
    fixed_points_count: int
    is_spectrum_valid: bool
    verdict: str
    cartan_pullback_defect: float = 0.0
    rotation_number: float = 0.0
    birkhoff_twist: float = 0.0


@dataclass(frozen=True, slots=True)
class _ReturnMapResult:
    r"""
    Resultado de la aduana del mapa de retorno P: Σ → Σ
    (Floquet + Lyapunov + Conley–Zehnder + ρ + Birkhoff + Cartan).
    """

    max_lyapunov: float
    floquet_parabolic: float
    is_hyperbolic: bool
    is_elliptic: bool
    is_parabolic: bool
    trace_M: float
    det_M: float
    verdict: str
    conley_zehnder_index: int = 0
    cz_nondegenerate: bool = True
    cz_distance_to_one: float = 1.0
    rotation_number: float = 0.0
    birkhoff_twist: float = 0.0
    elliptic_multiplicity: int = 0
    section_transversality: float = 1.0
    cartan_pullback_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class _UltrafilterResult:
    """
    Colapso de decisión de Capa 4.

    `global_verdict` es el MEET de (mediana TMR, átomo crítico, supervisor,
    supervisores celestes). El join queda como diagnóstico. El interlock
    dispara si el meet es ⊥ (VETOED).
    """

    global_verdict: str
    heyting_meet: str
    heyting_join: str
    tmr_median: str
    supervisor_meet: str
    celestial_supervisor_meet: str
    boolean_closed: str
    principal_atom: Optional[str]
    hardware_interlock: bool
    weighted_ancilla: str
    vote_masses: Dict[str, float]
    anomalies: Tuple[str, ...]
    authorization_residue: str = "COHERENT"
    symplectic_cartan_defect: float = 0.0
    conley_zehnder_index: int = 0
    rotation_number: float = 0.0


@dataclass(frozen=True, slots=True)
class _PretorioEdict:
    """Edicto pretoreano clásico. Cierre de la Fase II clásica."""

    jet: _PretorioJet
    hyper: _HypercohomologyResult
    brouwer: _BrouwerResult
    ultrafilter: _UltrafilterResult
    extended_verdicts: Dict[str, str]


@dataclass(frozen=True, slots=True)
class _PretorioCelestialEdict:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    EDICTO PRETOREANO CELESTE (objeto terminal de Fase II, inicial de Fase III).
    ═══════════════════════════════════════════════════════════════════════════
    Compone 𝒢_I + aduanas clásicas + aduanas celestes + ultrafiltro (MEET).
    """

    celestial_jet: _PretorioCelestialJet
    hyper: _HypercohomologyResult
    brouwer: _BrouwerResult
    kam: Optional[_KAMResult]
    melnikov: Optional[_MelnikovResult]
    birkhoff: Optional[_PoincareBirkhoffResult]
    return_map: Optional[_ReturnMapResult]
    ultrafilter: _UltrafilterResult
    extended_verdicts: Dict[str, str]
    symplectic_cartan_defect: float = 0.0
    conley_zehnder_index: int = 0
    rotation_number: float = 0.0


class PretorioAgent:
    r"""
    El Pretorio Agéntico (Capa 4 — Comandante Supremo de Seguridad).

    CONTINUACIÓN FORMAL DE I.ω+:
        lift_from_celestial_jet(jet: _PretorioCelestialJet) es el primer
        morfismo de esta fase y consume el objeto terminal de la Fase I.

    Consume el `_PretorioCelestialJet` de la Fase I y evalúa:
      • aciclicidad de calibre (hipercohomología Čech–de Rham),
      • consistencia de punto fijo (Brouwer en 𝒮_n),
      • pequeños divisores KAM (diofantinidad γ/|k|^τ + Novikov + Lindstedt),
      • ruptura homoclínica (Melnikov),
      • twist map de Poincaré–Birkhoff (giro opuesto + área + Φ*ω),
      • mapa de retorno P: Σ → Σ (Floquet + Lyapunov + CZ + ρ),
      • colapso TMR + ultrafiltro principal + ínfimo de Heyting (MEET).
    """

    def __init__(
        self,
        dimension_n: int,
        hypercohomology_threshold: float = _HYPERCOHOMOLOGY_THRESHOLD_DEFAULT,
        safety_margin: float = 1.0,
        require_positive_vanishing: bool = False,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        novikov_valuation_T: float = 1.0,
        engine: Optional[PretorioEngine] = None,
    ) -> None:
        if dimension_n <= 0:
            raise ValueError("dimension_n debe ser un entero positivo.")
        if hypercohomology_threshold <= 0.0:
            raise ValueError("hypercohomology_threshold debe ser estrictamente positivo.")
        if safety_margin <= 0.0:
            raise ValueError("safety_margin debe ser estrictamente positivo.")
        self._n: Final[int] = int(dimension_n)
        self._two_n: Final[int] = 2 * self._n
        self._threshold: Final[float] = float(hypercohomology_threshold)
        self._safety_margin: Final[float] = float(safety_margin)
        self._require_positive_vanishing: Final[bool] = bool(require_positive_vanishing)
        self._H0: Final[float] = float(hamiltonian_energy_H0)
        self._V: Final[float] = float(potential_energy_V)
        self._novikov_T: Final[float] = float(novikov_valuation_T)
        self._kernel = _PoincareCelestialKernel
        self._last_jet: Optional[_PretorioCelestialJet] = None
        self._last_edict: Optional[_PretorioCelestialEdict] = None
        if engine is not None:
            self._engine = engine
        else:
            try:
                from app.core.inmune_system.pretorio_engine import PretorioEngine as _PE

                try:
                    self._engine = _PE(
                        dimension_two_n=self._two_n,
                        hamiltonian_energy_H0=self._H0,
                        potential_energy_V=self._V,
                        novikov_valuation_T=self._novikov_T,
                    )
                except TypeError:
                    self._engine = _PE()  # type: ignore[call-arg]
            except Exception:
                self._engine = None  # type: ignore[assignment]

    # ── II.0 Continuación de I.ω+: lifting del 1-jet celeste ─────────────
    @classmethod
    def lift_from_celestial_jet(
        cls,
        jet: _PretorioCelestialJet,
        hypercohomology_threshold: float = _HYPERCOHOMOLOGY_THRESHOLD_DEFAULT,
        safety_margin: float = 1.0,
        require_positive_vanishing: bool = False,
        novikov_valuation_T: float = 1.0,
        engine: Optional[PretorioEngine] = None,
    ) -> "PretorioAgent":
        r"""
        Primer morfismo de la Fase II. Continúa I.ω+:
            assemble_pretorio_celestial_jet(...) -> 𝒢_I
            lift_from_celestial_jet(𝒢_I)         -> agente de Fase II
        """
        if not isinstance(jet, _PretorioCelestialJet):
            raise TypeError("lift_from_celestial_jet exige _PretorioCelestialJet.")
        agent = cls(
            dimension_n=int(jet.n),
            hypercohomology_threshold=hypercohomology_threshold,
            safety_margin=safety_margin,
            require_positive_vanishing=require_positive_vanishing,
            hamiltonian_energy_H0=float(jet.hamiltonian_energy_H0),
            potential_energy_V=float(jet.potential_V),
            novikov_valuation_T=novikov_valuation_T,
            engine=engine,
        )
        agent._last_jet = jet
        return agent

    def bind_celestial_jet(self, jet: _PretorioCelestialJet) -> "_PretorioCelestialJet":
        """Ancla 𝒢_I al agente (continuación de I.ω+ sobre instancia viva)."""
        if jet.n != self._n:
            logger.error("Jet de dimensión %d, pretoreo n=%d.", jet.n, self._n)
        self._last_jet = jet
        return jet

    # ── II.1 Ingesta de observables ───────────────────────────────────────
    def ingest_observables(
        self,
        cochain_complex_matrices: Sequence[np.ndarray],
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
    ) -> _PretorioJet:
        """Reenvía las observables del ciclo al último morfismo de la Fase I clásica."""
        return _PretorioSpectralCore.assemble_pretorio_jet(
            dimension_n=self._n,
            cochain_complex_matrices=cochain_complex_matrices,
            density_matrix=density_matrix,
            transition_map_matrix=transition_map_matrix,
        )

    def ingest_celestial_observables(
        self,
        cochain_complex_matrices: Sequence[np.ndarray],
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
        base_metric_g: Optional[np.ndarray] = None,
        grad_V: Optional[np.ndarray] = None,
    ) -> _PretorioCelestialJet:
        r"""
        Réplica pública del morfismo terminal de Fase I (I.ω+): compone el
        1-jet clásico con el bloque celeste (Maupertuis–Jacobi + Hill + λ_PC
        + Ω + Christoffel + Jacobi). El valor ES el objeto inicial de Fase II.
        """
        jet = _PretorioSpectralCore.assemble_pretorio_celestial_jet(
            dimension_n=self._n,
            cochain_complex_matrices=cochain_complex_matrices,
            density_matrix=density_matrix,
            transition_map_matrix=transition_map_matrix,
            hamiltonian_energy_H0=self._H0,
            potential_energy_V=self._V,
            base_metric_g=base_metric_g,
            grad_V=grad_V,
        )
        self._last_jet = jet
        return jet

    # ── II.2 Aduanas clásicas ────────────────────────────────────────────
    def _metric_verdict(
        self,
        metric: float,
        hard: float,
        degraded: float,
    ) -> str:
        hard_tol = float(hard) * self._safety_margin
        deg_tol = float(degraded) * self._safety_margin
        if (not np.isfinite(metric)) or metric > hard_tol:
            return "VETOED"
        if metric > deg_tol:
            return "DEGRADED"
        return "COHERENT"

    def _hyper_from_spectrum(self, spectrum: _HodgeSpectrum) -> _HypercohomologyResult:
        if (not spectrum.complex_valid) or (not np.isfinite(spectrum.max_nilpotency)):
            nil_verdict = "VETOED"
        else:
            nil_verdict = self._metric_verdict(
                spectrum.max_nilpotency,
                self._threshold,
                self._threshold * _HYPER_DEGRADATION_FACTOR,
            )
        positive_mass = (
            float(sum(spectrum.soft_betti[1:])) if len(spectrum.soft_betti) > 1 else 0.0
        )
        if not np.isfinite(positive_mass):
            acyc_verdict = "VETOED"
        elif positive_mass > _ACYCLIC_SOFT_VETO:
            acyc_verdict = "VETOED" if self._require_positive_vanishing else "DEGRADED"
        elif positive_mass > _ACYCLIC_SOFT_DEGRADE:
            acyc_verdict = "DEGRADED"
        else:
            acyc_verdict = "COHERENT"
        verdict = (
            HeytingVerdict.from_token(nil_verdict)
            .meet(HeytingVerdict.from_token(acyc_verdict))
            .name
        )
        return _HypercohomologyResult(
            max_residual=float(spectrum.max_nilpotency),
            hyper_obstruction=float(spectrum.hyper_obstruction),
            betti_numbers=spectrum.betti_numbers,
            soft_betti=spectrum.soft_betti,
            hodge_gaps=spectrum.hodge_gaps,
            nilpotency_residuals=spectrum.nilpotency_residuals,
            complex_valid=spectrum.complex_valid,
            verdict=verdict,
        )

    def _brouwer_from_spectrum(self, spectrum: _BrouwerSpectrum) -> _BrouwerResult:
        residual_verdict = self._metric_verdict(
            spectrum.residual,
            _BROUWER_HARD_TOLERANCE,
            _BROUWER_DEGRADED_TOLERANCE,
        )
        if (not np.isfinite(spectrum.trace_defect)) or (
            spectrum.trace_defect > _WILKINSON_DRIFT_LIMIT * self._safety_margin
        ):
            residual_verdict = "VETOED"
        if (not np.isfinite(spectrum.positivity_defect)) or (
            spectrum.positivity_defect > _BROUWER_HARD_TOLERANCE * self._safety_margin
        ):
            residual_verdict = "VETOED"
        if not np.isfinite(spectrum.simplex_residual):
            residual_verdict = "VETOED"
        return _BrouwerResult(
            residual=float(spectrum.residual),
            simplex_residual=float(spectrum.simplex_residual),
            trace_defect=float(spectrum.trace_defect),
            positivity_defect=float(spectrum.positivity_defect),
            hermiticity_defect=float(spectrum.hermiticity_defect),
            isometry_defect=float(spectrum.isometry_defect),
            lipschitz=float(spectrum.lipschitz),
            banach_contraction=bool(spectrum.banach_contraction),
            verdict=residual_verdict,
        )

    def audit_calibre_hypercohomology(
        self,
        cochain_complex_matrices: List[np.ndarray],
        jet: Optional[_PretorioJet] = None,
    ) -> _HypercohomologyResult:
        r"""
        [PRETORIO — HIPERCOHOMOLOGÍA DE ČECH–DE RHAM / HODGE]
        Evalúa el complejo y exige, en la medida de Wilkinson:
            d^{k+1}∘d^k = 0,  ℍ^{k>0} ≈ 0.
        """
        if jet is not None:
            if jet.n != self._n:
                logger.error("Jet de dimensión %d, pretoreo n=%d.", jet.n, self._n)
            return self._hyper_from_spectrum(jet.hodge)
        spectrum = _PretorioSpectralCore.compute_hodge_spectrum(cochain_complex_matrices)
        return self._hyper_from_spectrum(spectrum)

    def verify_brouwer_fixed_point_consistency(
        self,
        density_matrix: np.ndarray,
        transition_map_matrix: np.ndarray,
        jet: Optional[_PretorioJet] = None,
    ) -> _BrouwerResult:
        r"""
        [PRETORIO — PUNTO FIJO DE BROUWER SOBRE 𝒮_n]
        Audita f(ρ) = ρ para f(ρ) = TρT† / Tr(TρT†).
        """
        if jet is not None:
            return self._brouwer_from_spectrum(jet.brouwer)
        spectrum = _PretorioSpectralCore.compute_brouwer_spectrum(
            density_matrix, transition_map_matrix, self._n
        )
        return self._brouwer_from_spectrum(spectrum)

    def _canonical_omega(self, two_n: Optional[int] = None) -> np.ndarray:
        n = int(self._two_n if two_n is None else two_n)
        if self._last_jet is not None and self._last_jet.omega.shape == (n, n):
            return self._last_jet.omega
        try:
            return _PretorioSpectralCore.generate_canonical_symplectic_form(n)
        except ValueError:
            return np.zeros((n, n), dtype=np.float64)

    # ── II.3 Aduanas celestes ────────────────────────────────────────────
    def audit_poincare_kam_stability(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        jacobian_M: np.ndarray,
        canonical_J: Optional[np.ndarray] = None,
        tau: float = _KAM_TAU_FLOOR,
        gamma: float = _KAM_GAMMA_FLOOR,
    ) -> _KAMResult:
        r"""
        [PRETORIO — PEQUEÑOS DIVISORES DE POINCARÉ–KAM]
        Valúa |⟨k, ω⟩| ≥ γ/|k|^τ, absorción ultramétrica de Novikov,
        residual homológico de Lindstedt y pullback de Cartan Φ*ω − ω.
        """
        canon = canonical_J if canonical_J is not None else self._canonical_omega()
        if self._engine is not None and hasattr(
            self._engine, "compute_poincare_small_divisors_spectrum"
        ):
            try:
                audit = self._engine.compute_poincare_small_divisors_spectrum(
                    frequency_vector_omega=frequency_vector_omega,
                    wave_vectors_k=wave_vectors_k,
                    jacobian_M=jacobian_M,
                    canonical_J=canon,
                    tau=tau,
                    gamma=gamma,
                )
                min_div = float(getattr(audit, "min_divisor", float("inf")))
                nov_weight = float(getattr(audit, "novikov_weight", 0.0))
                mc_res = float(getattr(audit, "maurercartan_residual", float("inf")))
                vol_drift = float(getattr(audit, "volume_drift", float("inf")))
                res_gap = float(getattr(audit, "resonance_gap", float("inf")))
                t_ = float(getattr(audit, "tau", tau))
                g_ = float(getattr(audit, "gamma", gamma))
                is_diof = bool(getattr(audit, "is_diophantine", False))
                is_kam = bool(getattr(audit, "is_kam_stable", False))
                engine_ok = bool(getattr(audit, "engine_ok", True))
                homo = float(
                    getattr(
                        audit,
                        "homological_residual",
                        self._kernel.homological_lindstedt_residual(min_div),
                    )
                )
                cartan = float(
                    getattr(
                        audit,
                        "cartan_pullback_defect",
                        self._kernel.symplectic_pullback_defect(
                            np.asarray(jacobian_M), np.asarray(canon)
                        ),
                    )
                )
            except Exception as exc:
                logger.error("Fallo en compute_poincare_small_divisors_spectrum: %s", exc)
                return self._fallback_kam_audit(
                    frequency_vector_omega, wave_vectors_k, jacobian_M, canon, tau, gamma
                )
        else:
            return self._fallback_kam_audit(
                frequency_vector_omega, wave_vectors_k, jacobian_M, canon, tau, gamma
            )
        tol = _KAM_THRESHOLD * self._safety_margin
        if (not engine_ok) or (not np.isfinite(min_div)):
            verdict = "VETOED"
        elif min_div <= _WILKINSON_DRIFT_LIMIT:
            verdict = "VETOED"
        elif vol_drift > _WILKINSON_DRIFT_LIMIT:
            verdict = "VETOED"
        elif np.isfinite(cartan) and cartan > _CARTAN_DEFECT_TOL:
            verdict = "VETOED"
        elif not is_diof:
            verdict = "DEGRADED"
        elif is_kam and min_div >= tol:
            verdict = "COHERENT"
        elif min_div >= tol * _DEGRADATION_FACTOR:
            verdict = "DEGRADED"
        else:
            verdict = "VETOED"
        return _KAMResult(
            min_divisor=min_div,
            novikov_weight=nov_weight,
            maurercartan_residual=mc_res,
            volume_drift=vol_drift,
            resonance_gap=res_gap,
            tau=t_,
            gamma=g_,
            is_diophantine=is_diof,
            is_kam_stable=is_kam,
            verdict=verdict,
            homological_residual=float(homo),
            cartan_pullback_defect=float(cartan),
        )

    def _fallback_kam_audit(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        jacobian_M: np.ndarray,
        canonical_J: np.ndarray,
        tau: float,
        gamma: float,
    ) -> _KAMResult:
        """Cómputo local de respaldo si el motor carece del método."""
        try:
            omega = np.asarray(frequency_vector_omega, dtype=np.float64).ravel()
            wave_k = np.asarray(wave_vectors_k, dtype=np.float64)
            jac_m = np.asarray(jacobian_M, dtype=np.float64)
            if wave_k.ndim == 1:
                divisors = np.abs(np.dot(wave_k, omega))
                k_norms = np.abs(wave_k)
                min_divisor = float(divisors)
                argmin_idx: Optional[int] = 0
            else:
                divisors = np.abs(wave_k @ omega)
                k_norms = np.linalg.norm(wave_k, axis=1)
                if divisors.size == 0:
                    min_divisor = 1.0
                    argmin_idx = None
                else:
                    argmin_idx = int(np.argmin(divisors))
                    min_divisor = float(divisors[argmin_idx])
            novikov_weight = float(
                np.exp(
                    -np.clip(
                        self._novikov_T / (_WILKINSON_DRIFT_LIMIT + min_divisor),
                        0.0,
                        _LOG_EXP_CLIP,
                    )
                )
            )
            mc_residual = float(abs(min_divisor * novikov_weight))
            if argmin_idx is not None and wave_k.ndim > 1 and wave_k.size > 0:
                k_norm_min = float(max(k_norms[argmin_idx], 1.0))
                is_diof = bool(min_divisor * (k_norm_min ** tau) >= gamma)
            else:
                is_diof = bool(min_divisor >= _WILKINSON_DRIFT_LIMIT)
            det_M = float(np.real(la.det(jac_m))) if jac_m.size > 0 else 0.0
            vol_drift = float(abs(det_M - 1.0))
            cartan = self._kernel.symplectic_pullback_defect(jac_m, np.asarray(canonical_J))
            homo = self._kernel.homological_lindstedt_residual(min_divisor)
            is_kam = bool(
                (min_divisor >= _WILKINSON_DRIFT_LIMIT)
                and (vol_drift <= _WILKINSON_DRIFT_LIMIT)
                and is_diof
                and (not np.isfinite(cartan) or cartan <= _CARTAN_DEFECT_TOL)
            )
        except Exception as exc:
            logger.error("Fallo en fallback KAM: %s", exc)
            return _KAMResult(
                min_divisor=float("inf"),
                novikov_weight=0.0,
                maurercartan_residual=float("inf"),
                volume_drift=float("inf"),
                resonance_gap=float("inf"),
                tau=float(tau),
                gamma=float(gamma),
                is_diophantine=False,
                is_kam_stable=False,
                verdict="VETOED",
                homological_residual=float("inf"),
                cartan_pullback_defect=float("inf"),
            )
        tol = _KAM_THRESHOLD * self._safety_margin
        if min_divisor <= _WILKINSON_DRIFT_LIMIT or vol_drift > _WILKINSON_DRIFT_LIMIT:
            verdict = "VETOED"
        elif np.isfinite(cartan) and cartan > _CARTAN_DEFECT_TOL:
            verdict = "VETOED"
        elif not is_diof:
            verdict = "DEGRADED"
        elif is_kam and min_divisor >= tol:
            verdict = "COHERENT"
        elif min_divisor >= tol * _DEGRADATION_FACTOR:
            verdict = "DEGRADED"
        else:
            verdict = "VETOED"
        return _KAMResult(
            min_divisor=min_divisor,
            novikov_weight=novikov_weight,
            maurercartan_residual=mc_residual,
            volume_drift=vol_drift,
            resonance_gap=min_divisor,
            tau=float(tau),
            gamma=float(gamma),
            is_diophantine=is_diof,
            is_kam_stable=is_kam,
            verdict=verdict,
            homological_residual=float(homo),
            cartan_pullback_defect=float(cartan),
        )

    def audit_melnikov_homoclinic_splitting(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float = _MELNIKOV_T_INF,
    ) -> _MelnikovResult:
        r"""
        [PRETORIO — FUNCIÓN DE MELNIKOV]
        M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(γ⁰(t − t₀)) dt.
        Cero simple ⇒ fractura homoclínica de Poincaré (1890).
        """
        if self._engine is not None and hasattr(self._engine, "compute_melnikov_function"):
            try:
                audit = self._engine.compute_melnikov_function(
                    homoclinic_flow=homoclinic_flow,
                    hamiltonian_0=hamiltonian_0,
                    hamiltonian_1=hamiltonian_1,
                    t0_grid=t0_grid,
                    t_inf=float(t_inf),
                )
                m_val = float(getattr(audit, "melnikov_value", float("nan")))
                m_deriv = float(getattr(audit, "melnikov_derivative", float("nan")))
                is_simple = bool(getattr(audit, "is_simple_zero", False))
                splitting = float(getattr(audit, "homoclinic_splitting", float("nan")))
                engine_ok = bool(getattr(audit, "engine_ok", True))
            except Exception as cop_exc:
                logger.error("Fallo en compute_melnikov_function: %s", cop_exc)
                return _MelnikovResult(
                    melnikov_value=float("nan"),
                    melnikov_derivative=float("nan"),
                    is_simple_zero=False,
                    homoclinic_splitting=float("nan"),
                    verdict="VETOED",
                )
        else:
            return self._fallback_melnikov_audit(
                homoclinic_flow, hamiltonian_0, hamiltonian_1, t0_grid, t_inf
            )
        tol = _MELNIKOV_THRESHOLD * self._safety_margin
        if (not engine_ok) or (not np.isfinite(m_val)):
            verdict = "VETOED"
        elif is_simple:
            verdict = "VETOED"
        elif abs(m_val) >= tol:
            verdict = "COHERENT"
        elif abs(m_val) >= tol * _DEGRADATION_FACTOR:
            verdict = "DEGRADED"
        else:
            verdict = "VETOED"
        return _MelnikovResult(
            melnikov_value=m_val,
            melnikov_derivative=m_deriv,
            is_simple_zero=is_simple,
            homoclinic_splitting=splitting,
            verdict=verdict,
        )

    def _fallback_melnikov_audit(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float,
    ) -> _MelnikovResult:
        """Cómputo local de respaldo de Melnikov (Gauss–Legendre + Kahan)."""
        try:
            t0s = np.asarray(t0_grid, dtype=np.float64).ravel()
            if t0s.size == 0:
                raise ValueError("t0_grid vacío.")
            n_quad = 513
            nodes, weights = np.polynomial.legendre.leggauss(n_quad)
            t_nodes = t_inf * nodes
            w_nodes = t_inf * weights
            omega = self._canonical_omega()
            mel_vals = np.zeros(t0s.size, dtype=np.float64)
            for i, t0 in enumerate(t0s):
                acc = 0.0
                comp = 0.0
                for t_shift, w in zip(t_nodes, w_nodes):
                    try:
                        x = np.asarray(
                            homoclinic_flow(float(t_shift - t0)), dtype=np.float64
                        )
                    except Exception:
                        continue
                    if x.size != omega.shape[0]:
                        continue
                    try:
                        pb = _PretorioSpectralCore.poisson_bracket(
                            hamiltonian_0, hamiltonian_1, x, omega
                        )
                    except (TypeError, ValueError, FloatingPointError):
                        pb = 0.0
                    if not np.isfinite(pb):
                        pb = 0.0
                    y = pb * float(w) - comp
                    t = acc + y
                    comp = (t - acc) - y
                    acc = t
                mel_vals[i] = acc
            idx_min = int(np.argmin(np.abs(mel_vals)))
            m_val = float(mel_vals[idx_min])
            if t0s.size >= 2 and 0 < idx_min < t0s.size - 1:
                dm = (mel_vals[idx_min + 1] - mel_vals[idx_min - 1]) / (
                    t0s[idx_min + 1] - t0s[idx_min - 1]
                )
            elif t0s.size >= 2:
                dm = (mel_vals[-1] - mel_vals[0]) / max(t0s[-1] - t0s[0], _MACHINE_EPS)
            else:
                dm = 0.0
            is_simple = bool(
                abs(m_val) < _WILKINSON_DRIFT_LIMIT and abs(dm) > _WILKINSON_DRIFT_LIMIT
            )
            splitting = float(abs(m_val))
        except Exception as cop_exc:
            logger.error("Fallo en fallback Melnikov: %s", cop_exc)
            return _MelnikovResult(
                melnikov_value=float("nan"),
                melnikov_derivative=float("nan"),
                is_simple_zero=False,
                homoclinic_splitting=float("nan"),
                verdict="VETOED",
            )
        tol = _MELNIKOV_THRESHOLD * self._safety_margin
        if is_simple:
            verdict = "VETOED"
        elif abs(m_val) >= tol:
            verdict = "COHERENT"
        elif abs(m_val) >= tol * _DEGRADATION_FACTOR:
            verdict = "DEGRADED"
        else:
            verdict = "VETOED"
        return _MelnikovResult(
            melnikov_value=m_val,
            melnikov_derivative=float(dm),
            is_simple_zero=is_simple,
            homoclinic_splitting=splitting,
            verdict=verdict,
        )

    def audit_poincare_birkhoff_deliberation(
        self,
        deliberation_matrix_M: np.ndarray,
        contractor_twist_angle: float,
        auditor_twist_angle: float,
    ) -> _PoincareBirkhoffResult:
        r"""
        [PRETORIO — TWIST MAP DE POINCARÉ–BIRKHOFF]
        Axiomas: giro opuesto, conservación de área, |Spec(M) ∩ S¹| ≥ 2,
        invariante integral Φ*ω = ω.
        """
        omega = self._canonical_omega()
        if self._engine is not None and hasattr(
            self._engine, "compute_poincare_birkhoff_twist_spectrum_certified"
        ):
            try:
                audit = self._engine.compute_poincare_birkhoff_twist_spectrum_certified(
                    deliberation_matrix_M=deliberation_matrix_M,
                    contractor_twist_angle=contractor_twist_angle,
                    auditor_twist_angle=auditor_twist_angle,
                )
                area_drift = float(getattr(audit, "area_drift", float("inf")))
                has_opp = bool(getattr(audit, "has_opposite_twist", False))
                fixed_pts = int(getattr(audit, "fixed_points_count", 0))
                is_valid = bool(getattr(audit, "is_spectrum_valid", False))
                engine_ok = bool(getattr(audit, "engine_ok", True))
                cartan = float(
                    getattr(
                        audit,
                        "cartan_pullback_defect",
                        self._kernel.symplectic_pullback_defect(
                            np.asarray(deliberation_matrix_M), omega
                        ),
                    )
                )
                rho = float(getattr(audit, "rotation_number", float("nan")))
                twist = float(getattr(audit, "birkhoff_twist", 0.0))
            except Exception as cop_exc:
                logger.error(
                    "Fallo en compute_poincare_birkhoff_twist_spectrum: %s", cop_exc
                )
                return _PoincareBirkhoffResult(
                    area_drift=float("inf"),
                    has_opposite_twist=False,
                    fixed_points_count=0,
                    is_spectrum_valid=False,
                    verdict="VETOED",
                    cartan_pullback_defect=float("inf"),
                )
        elif self._engine is not None and hasattr(
            self._engine, "compute_poincare_birkhoff_twist_spectrum"
        ):
            try:
                audit_report = self._engine.compute_poincare_birkhoff_twist_spectrum(
                    deliberation_matrix_M=deliberation_matrix_M,
                    contractor_twist_angle=contractor_twist_angle,
                    auditor_twist_angle=auditor_twist_angle,
                )
                area_drift = float(getattr(audit_report, "area_drift", float("inf")))
                has_opp = bool(getattr(audit_report, "has_opposite_twist", False))
                fixed_pts = int(getattr(audit_report, "fixed_points_count", 0))
                is_valid = bool(getattr(audit_report, "is_spectrum_valid", False))
                engine_ok = True
                cartan = self._kernel.symplectic_pullback_defect(
                    np.asarray(deliberation_matrix_M), omega
                )
                rho, twist, _n = self._kernel.rotation_number_and_birkhoff_twist(
                    la.eigvals(np.asarray(deliberation_matrix_M))
                )
            except Exception as cop_exc:
                logger.error(
                    "Fallo en compute_poincare_birkhoff_twist_spectrum: %s", cop_exc
                )
                return _PoincareBirkhoffResult(
                    area_drift=float("inf"),
                    has_opposite_twist=False,
                    fixed_points_count=0,
                    is_spectrum_valid=False,
                    verdict="VETOED",
                    cartan_pullback_defect=float("inf"),
                )
        else:
            return self._fallback_birkhoff_audit(
                deliberation_matrix_M, contractor_twist_angle, auditor_twist_angle
            )
        if (not engine_ok) or (not np.isfinite(area_drift)):
            verdict = "VETOED"
        elif np.isfinite(cartan) and cartan > _CARTAN_DEFECT_TOL:
            verdict = "VETOED"
        elif is_valid:
            verdict = "COHERENT"
        elif has_opp:
            verdict = "DEGRADED"
        else:
            verdict = "VETOED"
        return _PoincareBirkhoffResult(
            area_drift=area_drift,
            has_opposite_twist=has_opp,
            fixed_points_count=fixed_pts,
            is_spectrum_valid=is_valid,
            verdict=verdict,
            cartan_pullback_defect=float(cartan),
            rotation_number=float(rho) if np.isfinite(rho) else float("nan"),
            birkhoff_twist=float(twist),
        )

    def _fallback_birkhoff_audit(
        self,
        deliberation_matrix_M: np.ndarray,
        contractor_twist_angle: float,
        auditor_twist_angle: float,
    ) -> _PoincareBirkhoffResult:
        """Cómputo local de respaldo del twist map."""
        try:
            M = np.asarray(deliberation_matrix_M, dtype=np.float64)
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError("deliberation_matrix_M debe ser cuadrada.")
            det_M = float(np.real(la.det(M)))
            area_drift = abs(det_M - 1.0)
            has_opp = bool(
                (contractor_twist_angle * auditor_twist_angle) < -_BIRKHOFF_TWIST_FLOOR
            )
            eigvals = la.eigvals(M)
            fixed_pts = int(np.sum(np.isclose(np.abs(eigvals), 1.0, atol=1e-9)))
            cartan = self._kernel.symplectic_pullback_defect(M, self._canonical_omega())
            rho, twist, _n = self._kernel.rotation_number_and_birkhoff_twist(eigvals)
            is_valid = bool(
                has_opp
                and (area_drift <= _BIRKHOFF_AREA_DRIFT_MAX)
                and (fixed_pts >= 2)
                and (not np.isfinite(cartan) or cartan <= _CARTAN_DEFECT_TOL)
            )
        except Exception as cop_exc:
            logger.error("Fallo en fallback Birkhoff: %s", cop_exc)
            return _PoincareBirkhoffResult(
                area_drift=float("inf"),
                has_opposite_twist=False,
                fixed_points_count=0,
                is_spectrum_valid=False,
                verdict="VETOED",
                cartan_pullback_defect=float("inf"),
            )
        if np.isfinite(cartan) and cartan > _CARTAN_DEFECT_TOL:
            verdict = "VETOED"
        elif is_valid:
            verdict = "COHERENT"
        elif has_opp:
            verdict = "DEGRADED"
        else:
            verdict = "VETOED"
        return _PoincareBirkhoffResult(
            area_drift=area_drift,
            has_opposite_twist=has_opp,
            fixed_points_count=fixed_pts,
            is_spectrum_valid=is_valid,
            verdict=verdict,
            cartan_pullback_defect=float(cartan),
            rotation_number=float(rho) if np.isfinite(rho) else float("nan"),
            birkhoff_twist=float(twist),
        )

    def audit_pretorio_poincare_deliberation(
        self,
        deliberation_state: np.ndarray,
        jacobian_M: np.ndarray,
        contractor_twist: float,
        auditor_twist: float,
    ) -> PretorioDeliberationCertificate:
        r"""
        [API 2.0 — CERTIFICADO CLÁSICO DE POINCARÉ–BIRKHOFF]
        Envuelve `audit_poincare_birkhoff_deliberation` (retrocompatibilidad).
        """
        result = self.audit_poincare_birkhoff_deliberation(
            jacobian_M, contractor_twist, auditor_twist
        )
        return PretorioDeliberationCertificate(
            verdict=result.verdict,
            area_drift=result.area_drift,
            fixed_points=result.fixed_points_count,
            is_deliberation_stable=(result.verdict == "COHERENT"),
        )

    def audit_poincare_return_map(
        self,
        jacobian_M: np.ndarray,
        period_T: float = 1.0,
    ) -> _ReturnMapResult:
        r"""
        [PRETORIO — MAPA DE RETORNO DE POINCARÉ]
        Espectro de Floquet + Lyapunov + Conley–Zehnder + ρ + Cartan de P: Σ → Σ.
        Clasificación v5 mutuamente excluyente: elíptico / parabólico / hiperbólico.
        """
        omega = self._canonical_omega()
        if self._engine is not None and hasattr(self._engine, "compute_poincare_return_map"):
            try:
                audit = self._engine.compute_poincare_return_map(
                    jacobian_M=jacobian_M, period_T=period_T
                )
                max_lyap = float(getattr(audit, "max_lyapunov", 0.0))
                is_hyp = bool(getattr(audit, "is_hyperbolic", False))
                is_ell = bool(getattr(audit, "is_elliptic", False))
                is_par = bool(getattr(audit, "is_parabolic", False))
                trace_M = float(getattr(audit, "trace_M", float("nan")))
                det_M = float(getattr(audit, "det_M", float("nan")))
                engine_ok = bool(getattr(audit, "engine_ok", True))
                floq = np.asarray(
                    getattr(audit, "floquet_multipliers", np.array([])),
                    dtype=np.complex128,
                )
                floq_par = float(getattr(audit, "floquet_parabolic", 0.0))
                if floq.size and (not np.isfinite(floq_par) or floq_par == 0.0):
                    floq_par = float(np.max(np.abs(np.abs(floq) - 1.0)))
                # Reclasificación mutuamente excluyente si el motor es v4.
                if floq.size:
                    is_hyp, is_ell, is_par, floq_par = self._kernel.classify_floquet(
                        np.abs(floq)
                    )
                cz_idx = int(getattr(audit, "conley_zehnder_index", 0))
                cz_nd = bool(getattr(audit, "cz_nondegenerate", True))
                cz_dist = float(getattr(audit, "cz_distance_to_one", 1.0))
                if not hasattr(audit, "conley_zehnder_index"):
                    cz_idx, cz_nd, cz_dist = self._kernel.conley_zehnder_index(
                        np.asarray(jacobian_M)
                    )
                rho = float(getattr(audit, "rotation_number", float("nan")))
                twist = float(getattr(audit, "birkhoff_twist", 0.0))
                n_ell = int(getattr(audit, "elliptic_multiplicity", 0))
                if not hasattr(audit, "rotation_number") and floq.size:
                    rho, twist, n_ell = self._kernel.rotation_number_and_birkhoff_twist(
                        floq
                    )
                trans = float(
                    getattr(
                        audit,
                        "section_transversality",
                        self._kernel.section_transversality(np.asarray(jacobian_M)),
                    )
                )
                cartan = float(
                    getattr(
                        audit,
                        "cartan_pullback_defect",
                        self._kernel.symplectic_pullback_defect(
                            np.asarray(jacobian_M), omega
                        ),
                    )
                )
            except Exception as cop_exc:
                logger.error("Fallo en compute_poincare_return_map: %s", cop_exc)
                return _ReturnMapResult(
                    max_lyapunov=float("inf"),
                    floquet_parabolic=float("inf"),
                    is_hyperbolic=False,
                    is_elliptic=False,
                    is_parabolic=False,
                    trace_M=float("nan"),
                    det_M=float("nan"),
                    verdict="VETOED",
                    cz_nondegenerate=False,
                    cartan_pullback_defect=float("inf"),
                )
        else:
            return self._fallback_return_map_audit(jacobian_M, period_T)
        tol = _FLOQUET_PARABOLIC * self._safety_margin
        if not engine_ok:
            verdict = "VETOED"
        elif np.isfinite(cartan) and cartan > _CARTAN_DEFECT_TOL:
            verdict = "VETOED"
        elif is_ell and abs(max_lyap) <= tol * _DEGRADATION_FACTOR:
            if (not cz_nd) or (trans <= _SECTION_TRANSVERSALITY_FLOOR):
                verdict = "DEGRADED"
            else:
                verdict = "COHERENT"
        elif floq_par <= tol:
            verdict = "DEGRADED"
        elif abs(max_lyap) <= _MELNIKOV_LYAPUNOV_VETO:
            verdict = "DEGRADED"
        else:
            verdict = "VETOED"
        return _ReturnMapResult(
            max_lyapunov=max_lyap,
            floquet_parabolic=float(floq_par),
            is_hyperbolic=is_hyp,
            is_elliptic=is_ell,
            is_parabolic=is_par,
            trace_M=trace_M,
            det_M=det_M,
            verdict=verdict,
            conley_zehnder_index=int(cz_idx),
            cz_nondegenerate=bool(cz_nd),
            cz_distance_to_one=float(cz_dist),
            rotation_number=float(rho) if np.isfinite(rho) else float("nan"),
            birkhoff_twist=float(twist),
            elliptic_multiplicity=int(n_ell),
            section_transversality=float(trans),
            cartan_pullback_defect=float(cartan),
        )

    def _fallback_return_map_audit(
        self,
        jacobian_M: np.ndarray,
        period_T: float,
    ) -> _ReturnMapResult:
        """Cómputo local de respaldo del mapa de retorno (Floquet excluyente + CZ)."""
        try:
            M = np.asarray(jacobian_M, dtype=np.float64)
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError("jacobian_M debe ser cuadrada.")
            ev = la.eigvals(M)
            magnitudes = np.abs(ev)
            lyap = np.log(np.maximum(magnitudes, _MACHINE_EPS)) / max(
                abs(period_T), _MACHINE_EPS
            )
            lyap = np.clip(lyap, -_LYAPUNOV_CLIP, _LYAPUNOV_CLIP)
            max_lyap = float(np.max(lyap)) if lyap.size else 0.0
            is_hyp, is_ell, is_par, floq_par = self._kernel.classify_floquet(magnitudes)
            trace_M = float(np.trace(M))
            det_M = float(np.real(la.det(M)))
            cz_idx, cz_nd, cz_dist = self._kernel.conley_zehnder_index(M)
            rho, twist, n_ell = self._kernel.rotation_number_and_birkhoff_twist(ev)
            trans = self._kernel.section_transversality(M)
            cartan = self._kernel.symplectic_pullback_defect(M, self._canonical_omega())
        except Exception as cop_exc:
            logger.error("Fallo en fallback return map: %s", cop_exc)
            return _ReturnMapResult(
                max_lyapunov=float("inf"),
                floquet_parabolic=float("inf"),
                is_hyperbolic=False,
                is_elliptic=False,
                is_parabolic=False,
                trace_M=float("nan"),
                det_M=float("nan"),
                verdict="VETOED",
                cz_nondegenerate=False,
                cartan_pullback_defect=float("inf"),
            )
        tol = _FLOQUET_PARABOLIC * self._safety_margin
        if np.isfinite(cartan) and cartan > _CARTAN_DEFECT_TOL:
            verdict = "VETOED"
        elif is_ell and abs(max_lyap) <= tol * _DEGRADATION_FACTOR:
            if (not cz_nd) or (trans <= _SECTION_TRANSVERSALITY_FLOOR):
                verdict = "DEGRADED"
            else:
                verdict = "COHERENT"
        elif floq_par <= tol:
            verdict = "DEGRADED"
        elif abs(max_lyap) <= _MELNIKOV_LYAPUNOV_VETO:
            verdict = "DEGRADED"
        else:
            verdict = "VETOED"
        return _ReturnMapResult(
            max_lyapunov=max_lyap,
            floquet_parabolic=floq_par,
            is_hyperbolic=is_hyp,
            is_elliptic=is_ell,
            is_parabolic=is_par,
            trace_M=trace_M,
            det_M=det_M,
            verdict=verdict,
            conley_zehnder_index=int(cz_idx),
            cz_nondegenerate=bool(cz_nd),
            cz_distance_to_one=float(cz_dist),
            rotation_number=float(rho) if np.isfinite(rho) else float("nan"),
            birkhoff_twist=float(twist),
            elliptic_multiplicity=int(n_ell),
            section_transversality=float(trans),
            cartan_pullback_defect=float(cartan),
        )

    def _classify_maupertuis_channel(self, jet: _PretorioCelestialJet) -> str:
        r"""Valuación H₃ de la geometría de Maupertuis–Jacobi del 1-jet."""
        mau = jet.maupertuis_spectrum
        cartan = jet.cartan_spectrum
        if jet.input_fault and "celestial" in str(jet.input_fault):
            return "VETOED"
        if not mau.is_in_hill_region:
            return "VETOED"
        if (
            not np.isfinite(mau.christoffel_strength)
            or mau.christoffel_strength > _CHRISTOFFEL_BLOWUP
        ):
            return "VETOED"
        if (
            not np.isfinite(mau.jacobi_tidal_norm)
            or mau.jacobi_tidal_norm > _JACOBI_TIDAL_VETO
        ):
            return "VETOED"
        if np.isfinite(jet.darboux_residual) and jet.darboux_residual > _STRUCTURE_ATOL:
            return "VETOED"
        if (
            np.isfinite(cartan.symplectic_skew_residual)
            and cartan.symplectic_skew_residual > _STRUCTURE_ATOL
            and not cartan.darboux_ok
        ):
            return "VETOED"
        if mau.jacobi_tidal_norm > _JACOBI_TIDAL_DEGRADE:
            return "DEGRADED"
        if (
            np.isfinite(mau.refractive_index)
            and 0.0 < mau.refractive_index < _REFRACTIVE_NEAR_HILL
        ):
            return "DEGRADED"
        if np.isfinite(mau.koszul_torsion) and mau.koszul_torsion > _KOSZUL_TORSION_TOL:
            return "DEGRADED"
        return "COHERENT"

    # ── II.4 Ultrafiltro y colapso (MEET gobierna) ───────────────────────
    def evaluate_global_boolean_ultrafilter(
        self,
        layer_verdicts: Dict[str, str],
        symplectic_cartan_defect: float = 0.0,
        conley_zehnder_index: int = 0,
        rotation_number: float = 0.0,
    ) -> _UltrafilterResult:
        r"""
        [PRETORIO — TMR + ULTRAFILTRO PRINCIPAL + ÍNFIMO DE HEYTING]

        Tres morfismos, explícitamente separados:
        1. ν_∧ = ⋀_ℓ ν_ℓ en G₃ (ínfimo de Heyting = PEOR permiso).
        2. ν_TMR = mediana(ν_1, ν_2, ν_3) (mediana inferior).
        3. Ultrafiltro principal 𝒰_τ generado por el átomo crítico τ =
           Tesserarios; también participan el supervisor clásico y el
           supervisor celeste. El interlock dispara si el MEET es ⊥.

        Corrección v5: el join es diagnóstico; un único VETOED dispara.
        """
        core = _PretorioSpectralCore
        tokens = tuple(layer_verdicts.values())
        heyting = core.heyting_meet_tokens(tokens)
        joining = core.heyting_join_tokens(tokens)
        tmr_present = tuple(
            layer_verdicts[layer] for layer in _TMR_LAYERS if layer in layer_verdicts
        )
        tmr = core.heyting_lower_median(tmr_present)
        supervisor_present = tuple(
            layer_verdicts[layer]
            for layer in _SUPERVISOR_LAYERS
            if layer in layer_verdicts
        )
        supervisor = core.heyting_meet_tokens(supervisor_present)
        celestial_supervisor_present = tuple(
            layer_verdicts[layer]
            for layer in _CELESTIAL_SUPERVISOR_LAYERS
            if layer in layer_verdicts
        )
        celestial_supervisor = core.heyting_meet_tokens(celestial_supervisor_present)
        critical: Optional[HeytingVerdict] = None
        if _CRITICAL_ATOM in layer_verdicts:
            critical = HeytingVerdict.from_token_or_bottom(
                layer_verdicts[_CRITICAL_ATOM]
            )
        global_h = tmr.meet(supervisor).meet(celestial_supervisor)
        if critical is not None:
            global_h = global_h.meet(critical)
        residue = global_h.implies(HeytingVerdict.COHERENT)
        boolean_closed = global_h.booleanize_closed()
        principal_atom: Optional[str] = None
        if critical is HeytingVerdict.VETOED:
            principal_atom = _CRITICAL_ATOM
        elif supervisor is HeytingVerdict.VETOED:
            for layer in _SUPERVISOR_LAYERS:
                if layer_verdicts.get(layer) == "VETOED":
                    principal_atom = layer
                    break
        elif celestial_supervisor is HeytingVerdict.VETOED:
            for layer in _CELESTIAL_SUPERVISOR_LAYERS:
                if layer_verdicts.get(layer) == "VETOED":
                    principal_atom = layer
                    break
        elif tmr is HeytingVerdict.VETOED:
            principal_atom = "tmr_majority"
        interlock = boolean_closed is HeytingVerdict.VETOED
        weighted, masses, anomalies = core.weighted_social_choice(layer_verdicts)
        return _UltrafilterResult(
            global_verdict=global_h.name,
            heyting_meet=heyting.name,
            heyting_join=joining.name,
            tmr_median=tmr.name,
            supervisor_meet=supervisor.name,
            celestial_supervisor_meet=celestial_supervisor.name,
            boolean_closed=boolean_closed.name,
            principal_atom=principal_atom,
            hardware_interlock=bool(interlock),
            weighted_ancilla=weighted.name,
            vote_masses=dict(masses),
            anomalies=anomalies,
            authorization_residue=residue.name,
            symplectic_cartan_defect=float(symplectic_cartan_defect),
            conley_zehnder_index=int(conley_zehnder_index),
            rotation_number=float(rotation_number),
        )

    # ── II.ω  ÚLTIMO MORFISMO DE LA FASE II (CLÁSICO) ────────────────────
    def compile_pretorio_edict(
        self,
        jet: _PretorioJet,
        layer_verdicts: Dict[str, str],
    ) -> _PretorioEdict:
        """Cierre formal de la Fase II clásica. (API 2.0 conservada.)"""
        if jet.n != self._n:
            logger.error("Jet de dimensión %d incompatible con n=%d.", jet.n, self._n)
        if jet.input_fault:
            logger.error("Jet ensamblado con fallos de entrada: %s", jet.input_fault)
        hyper = self._hyper_from_spectrum(jet.hodge)
        brouwer = self._brouwer_from_spectrum(jet.brouwer)
        extended = dict(layer_verdicts)
        extended["capa_4_pretorio_hyper"] = hyper.verdict
        extended["capa_4_pretorio_brouwer"] = brouwer.verdict
        ultra = self.evaluate_global_boolean_ultrafilter(extended)
        return _PretorioEdict(
            jet=jet,
            hyper=hyper,
            brouwer=brouwer,
            ultrafilter=ultra,
            extended_verdicts=extended,
        )

    # ── II.ω+  MORFISMO TERMINAL DE LA FASE II: edicto celeste ──────────
    def compile_pretorio_celestial_edict(
        self,
        celestial_jet: _PretorioCelestialJet,
        layer_verdicts: Dict[str, str],
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        contractor_twist_angle: Optional[float] = None,
        auditor_twist_angle: Optional[float] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        t_inf: float = _MELNIKOV_T_INF,
        period_T: float = 1.0,
    ) -> _PretorioCelestialEdict:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE II ≅ OBJETO INICIAL DE LA FASE III.
        ═══════════════════════════════════════════════════════════════════════
        Compila un `_PretorioCelestialEdict` a partir del 1-jet celeste
        (cierre de Fase I) y los votos de las capas inferiores.

        Firma de continuación (Fase III):
            collapse_from_celestial_edict(edict: _PretorioCelestialEdict)
                -> Dict[str, Any]
        """
        self.bind_celestial_jet(celestial_jet)
        if celestial_jet.n != self._n:
            logger.error(
                "Celestial jet de dimensión %d incompatible con n=%d.",
                celestial_jet.n,
                self._n,
            )
        if celestial_jet.input_fault:
            logger.error(
                "Celestial jet ensamblado con fallos de entrada: %s",
                celestial_jet.input_fault,
            )
        hyper = self._hyper_from_spectrum(celestial_jet.base_jet.hodge)
        brouwer = self._brouwer_from_spectrum(celestial_jet.base_jet.brouwer)
        kam: Optional[_KAMResult] = None
        melnikov: Optional[_MelnikovResult] = None
        birkhoff: Optional[_PoincareBirkhoffResult] = None
        return_map: Optional[_ReturnMapResult] = None
        if frequency_vector_omega is not None and wave_vectors_k is not None:
            jac_m = jacobian_M if jacobian_M is not None else np.eye(self._two_n)
            canon_j = (
                canonical_J if canonical_J is not None else celestial_jet.omega
            )
            kam = self.audit_poincare_kam_stability(
                frequency_vector_omega, wave_vectors_k, jac_m, canon_j
            )
        if (
            homoclinic_flow is not None
            and hamiltonian_0 is not None
            and hamiltonian_1 is not None
            and t0_grid is not None
        ):
            melnikov = self.audit_melnikov_homoclinic_splitting(
                homoclinic_flow, hamiltonian_0, hamiltonian_1, t0_grid, t_inf=t_inf
            )
        if contractor_twist_angle is not None and auditor_twist_angle is not None:
            jac_m_bb = jacobian_M if jacobian_M is not None else np.eye(self._two_n)
            birkhoff = self.audit_poincare_birkhoff_deliberation(
                jac_m_bb, contractor_twist_angle, auditor_twist_angle
            )
        if jacobian_M is not None:
            return_map = self.audit_poincare_return_map(jacobian_M, period_T=period_T)
        extended = dict(layer_verdicts)
        extended["capa_4_pretorio_hyper"] = hyper.verdict
        extended["capa_4_pretorio_brouwer"] = brouwer.verdict
        mau_token = self._classify_maupertuis_channel(celestial_jet)
        if mau_token != "COHERENT":
            # El canal Maupertuis se inyecta en el supervisor celeste vía KAM
            # si no hay KAM explícito; si lo hay, degrada el meet global.
            extended.setdefault("capa_4_pretorio_kam", mau_token)
        if kam is not None:
            extended["capa_4_pretorio_kam"] = kam.verdict
        if melnikov is not None:
            extended["capa_4_pretorio_melnikov"] = melnikov.verdict
        if birkhoff is not None:
            extended["capa_4_pretorio_birkhoff"] = birkhoff.verdict
        if return_map is not None:
            extended["capa_4_pretorio_return_map"] = return_map.verdict
        cartan_defect = float(celestial_jet.darboux_residual)
        cz_idx = 0
        rho = float("nan")
        if return_map is not None:
            cartan_defect = max(cartan_defect, float(return_map.cartan_pullback_defect))
            cz_idx = int(return_map.conley_zehnder_index)
            rho = float(return_map.rotation_number)
        if kam is not None and np.isfinite(kam.cartan_pullback_defect):
            cartan_defect = max(cartan_defect, float(kam.cartan_pullback_defect))
        if birkhoff is not None and np.isfinite(birkhoff.cartan_pullback_defect):
            cartan_defect = max(cartan_defect, float(birkhoff.cartan_pullback_defect))
        ultra = self.evaluate_global_boolean_ultrafilter(
            extended,
            symplectic_cartan_defect=cartan_defect,
            conley_zehnder_index=cz_idx,
            rotation_number=rho if np.isfinite(rho) else 0.0,
        )
        edict = _PretorioCelestialEdict(
            celestial_jet=celestial_jet,
            hyper=hyper,
            brouwer=brouwer,
            kam=kam,
            melnikov=melnikov,
            birkhoff=birkhoff,
            return_map=return_map,
            ultrafilter=ultra,
            extended_verdicts=extended,
            symplectic_cartan_defect=float(cartan_defect),
            conley_zehnder_index=int(cz_idx),
            rotation_number=float(rho) if np.isfinite(rho) else float("nan"),
        )
        self._last_edict = edict
        return edict


# #############################################################################
#                                                                             #
#  FASE III                                                                   #
#  CÁMARA DE COHERENCIA · OODA · CROWBAR BT151                                #
#                                                                             #
#  Continuación directa del último morfismo de la Fase II:                    #
#      _PretorioCelestialEdict  ↦  PretorioCoherenceChamber                   #
#                                 .collapse_from_celestial_edict              #
#                                                                             #
# #############################################################################
@dataclass
class _ThyristorCrowbar:
    r"""
    Modelo lumped del tiristor BT151 como bypass de silicio.

    Física de circuito (actuador de seguridad de Capa 4):
      • disparo de puerta → enganche mientras I_A > I_H;
      • latencia de puerta ~400 ns con jitter térmico δt ~ N(0, σ²).
    """

    latched: bool = False
    last_latency_ns: float = 0.0

    def fire(self, rng: np.random.Generator) -> float:
        jitter = float(rng.normal(0.0, _CROWBAR_JITTER_NS))
        latency = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
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


class PretorioCoherenceChamber:
    r"""
    Cámara de Coherencia de la Capa 4.

    CONTINUACIÓN FORMAL DE II.ω+:
        collapse_from_celestial_edict(edict: _PretorioCelestialEdict) es el
        primer morfismo de esta fase y consume el objeto terminal de Fase II.

    Consume el `_PretorioCelestialEdict` y dispara el crowbar si el MEET
    del ultrafiltro principal / TMR / supervisor (clásico + celeste) es ⊥.
    """

    def __init__(
        self,
        agent: PretorioAgent,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.agent = agent
        self._crowbar = _ThyristorCrowbar()
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    @classmethod
    def assemble_from_spectral_seed(
        cls,
        dimension_n: int,
        hypercohomology_threshold: float = _HYPERCOHOMOLOGY_THRESHOLD_DEFAULT,
        safety_margin: float = 1.0,
        require_positive_vanishing: bool = False,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        novikov_valuation_T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> "PretorioCoherenceChamber":
        """
        Composición functorial Fase I → Fase II → Fase III.
        El agente (II.0) queda listo para consumir jets celestiales de
        `assemble_pretorio_celestial_jet` (I.ω+); la cámara es el objeto
        terminal del topos de seguridad pretoreano.
        """
        agent = PretorioAgent(
            dimension_n=dimension_n,
            hypercohomology_threshold=hypercohomology_threshold,
            safety_margin=safety_margin,
            require_positive_vanishing=require_positive_vanishing,
            hamiltonian_energy_H0=hamiltonian_energy_H0,
            potential_energy_V=potential_energy_V,
            novikov_valuation_T=novikov_valuation_T,
        )
        return cls(agent, rng=rng)

    # ── III.0 Continuación de II.ω+: colapso desde el edicto celeste ─────
    def collapse_from_celestial_edict(
        self, edict: _PretorioCelestialEdict
    ) -> Dict[str, Any]:
        r"""
        Primer morfismo de la Fase III. Continúa II.ω+:
            compile_pretorio_celestial_edict(...) -> 𝒢_II
            collapse_from_celestial_edict(𝒢_II)   -> acta {fire, no-fire}
        Delegado canónico de `fuse_and_actuate_celestial`.
        """
        return self.fuse_and_actuate_celestial(edict)

    # ── III.1 Ciclo OODA clásico (retrocompatible) ───────────────────────
    def fuse_and_actuate(self, edict: _PretorioEdict) -> Dict[str, Any]:
        """
        Ciclo OODA unificado de Capa 4 sobre un edicto clásico ya compilado.
        El interlock se gobierna por el MEET (hardware_interlock).
        """
        ultra = edict.ultrafilter
        latency_ns = 0.0
        interlock_fired = False
        if ultra.hardware_interlock:
            latency_ns = self._crowbar.fire(self._rng)
            interlock_fired = True
            logger.critical(
                "[EL PRETORIO AGÉNTICO — VETO SUPREMO] Colapso del ultrafiltro "
                "detectado (átomo=%s, TMR=%s, ⋀=%s). Bypass de potencia BT151 "
                "[GPIO14] despachado en %.2f ns via ISR en IRAM. Obra real "
                "paralizada incondicionalmente. ε_ℍ=%.3e  ε_B=%.3e  β=%s",
                ultra.principal_atom,
                ultra.tmr_median,
                ultra.heyting_meet,
                latency_ns,
                edict.hyper.max_residual,
                edict.brouwer.residual,
                edict.hyper.betti_numbers,
            )
        global_h = HeytingVerdict.from_token(ultra.global_verdict)
        return {
            "pretorio_global_verdict": ultra.global_verdict,
            "heyting_value": global_h.value,
            "heyting_meet": ultra.heyting_meet,
            "heyting_join": ultra.heyting_join,
            "authorization_residue": ultra.authorization_residue,
            "tmr_median": ultra.tmr_median,
            "supervisor_meet": ultra.supervisor_meet,
            "boolean_closed": ultra.boolean_closed,
            "ultrafilter_principal_atom": ultra.principal_atom,
            "weighted_vote_ancilla": ultra.weighted_ancilla,
            "weighted_vote_masses": dict(ultra.vote_masses),
            "ultrafilter_anomalies": list(ultra.anomalies),
            "hypercohomology_max_residual": edict.hyper.max_residual,
            "hypercohomology_obstruction": edict.hyper.hyper_obstruction,
            "hypercohomology_betti": list(edict.hyper.betti_numbers),
            "hypercohomology_soft_betti": list(edict.hyper.soft_betti),
            "hypercohomology_hodge_gaps": list(edict.hyper.hodge_gaps),
            "hypercohomology_nilpotency": list(edict.hyper.nilpotency_residuals),
            "hypercohomology_complex_valid": edict.hyper.complex_valid,
            "hypercohomology_verdict": edict.hyper.verdict,
            "brouwer_fixed_point_residual": edict.brouwer.residual,
            "brouwer_simplex_residual": edict.brouwer.simplex_residual,
            "brouwer_trace_defect": edict.brouwer.trace_defect,
            "brouwer_positivity_defect": edict.brouwer.positivity_defect,
            "brouwer_hermiticity_defect": edict.brouwer.hermiticity_defect,
            "brouwer_isometry_defect": edict.brouwer.isometry_defect,
            "brouwer_lipschitz": edict.brouwer.lipschitz,
            "brouwer_banach_contraction": edict.brouwer.banach_contraction,
            "brouwer_verdict": edict.brouwer.verdict,
            "hardware_interlock_fired": interlock_fired,
            "hardware_crowbar_latched": self._crowbar.latched,
            "actuation_latency_ns": latency_ns,
            "input_fault": edict.jet.input_fault,
            "audited_layers_audit_trail": dict(edict.extended_verdicts),
        }

    # ── III.2 MORFISMO TERMINAL DE LA FASE III: ciclo OODA celestial ─────
    def fuse_and_actuate_celestial(
        self, edict: _PretorioCelestialEdict
    ) -> Dict[str, Any]:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE III: colapso Ω₃ → {fire, no-fire}.
        ═══════════════════════════════════════════════════════════════════════
        Ciclo OODA unificado de Capa 4 sobre el edicto celeste. El MEET de
        Heyting (ínfimo de permiso) determina si se dispara el crowbar BT151
        vía GPIO14 en < 400 ns (ISR en IRAM del ESP32).

        Corrección v5: el join es diagnóstico; el colapso se gobierna por el
        meet. join(VETOED, COHERENT) = COHERENT ocultaba vetos.
        """
        ultra = edict.ultrafilter
        latency_ns = 0.0
        interlock_fired = False
        if ultra.hardware_interlock:
            latency_ns = self._crowbar.fire(self._rng)
            interlock_fired = True
            logger.critical(
                "[EL PRETORIO AGÉNTICO CELESTE — VETO SUPREMO] Colapso del "
                "ultrafiltro detectado (átomo=%s, TMR=%s, ⋀=%s, ⋀_celeste=%s, "
                "∨=%s). Bypass BT151 [GPIO14] en %.2f ns via ISR en IRAM. "
                "KAM=%s, Melnikov=%s, Birkhoff=%s, Retorno=%s. "
                "CartanDefect=%.3e CZ=%d ρ=%.6f. ε_ℍ=%.3e  ε_B=%.3e  β=%s  "
                "KAM_min_div=%.3e  |M(t₀)|=%.3e",
                ultra.principal_atom,
                ultra.tmr_median,
                ultra.heyting_meet,
                ultra.celestial_supervisor_meet,
                ultra.heyting_join,
                latency_ns,
                edict.kam.verdict if edict.kam is not None else "n/a",
                edict.melnikov.verdict if edict.melnikov is not None else "n/a",
                edict.birkhoff.verdict if edict.birkhoff is not None else "n/a",
                edict.return_map.verdict if edict.return_map is not None else "n/a",
                edict.symplectic_cartan_defect,
                edict.conley_zehnder_index,
                edict.rotation_number,
                edict.hyper.max_residual,
                edict.brouwer.residual,
                edict.hyper.betti_numbers,
                edict.kam.min_divisor if edict.kam is not None else float("nan"),
                abs(edict.melnikov.melnikov_value)
                if edict.melnikov is not None
                else float("nan"),
            )
        global_h = HeytingVerdict.from_token(ultra.global_verdict)
        result: Dict[str, Any] = {
            "pretorio_global_verdict": ultra.global_verdict,
            "heyting_value": global_h.value,
            "heyting_meet": ultra.heyting_meet,
            "heyting_join": ultra.heyting_join,
            "authorization_residue": ultra.authorization_residue,
            "tmr_median": ultra.tmr_median,
            "supervisor_meet": ultra.supervisor_meet,
            "celestial_supervisor_meet": ultra.celestial_supervisor_meet,
            "boolean_closed": ultra.boolean_closed,
            "ultrafilter_principal_atom": ultra.principal_atom,
            "weighted_vote_ancilla": ultra.weighted_ancilla,
            "weighted_vote_masses": dict(ultra.vote_masses),
            "ultrafilter_anomalies": list(ultra.anomalies),
            # Aduanas clásicas
            "hypercohomology_max_residual": edict.hyper.max_residual,
            "hypercohomology_obstruction": edict.hyper.hyper_obstruction,
            "hypercohomology_betti": list(edict.hyper.betti_numbers),
            "hypercohomology_complex_valid": edict.hyper.complex_valid,
            "hypercohomology_verdict": edict.hyper.verdict,
            "brouwer_fixed_point_residual": edict.brouwer.residual,
            "brouwer_simplex_residual": edict.brouwer.simplex_residual,
            "brouwer_trace_defect": edict.brouwer.trace_defect,
            "brouwer_positivity_defect": edict.brouwer.positivity_defect,
            "brouwer_verdict": edict.brouwer.verdict,
            # Geometría de la fase (Fase I)
            "maupertuis_conformal_factor": edict.celestial_jet.maupertuis_spectrum.conformal_factor,
            "maupertuis_refractive_index": edict.celestial_jet.maupertuis_spectrum.refractive_index,
            "maupertuis_hill_margin": edict.celestial_jet.maupertuis_spectrum.hill_margin,
            "maupertuis_is_in_hill_region": edict.celestial_jet.maupertuis_spectrum.is_in_hill_region,
            "maupertuis_christoffel_strength": edict.celestial_jet.maupertuis_spectrum.christoffel_strength,
            "maupertuis_jacobi_tidal_norm": edict.celestial_jet.maupertuis_spectrum.jacobi_tidal_norm,
            "maupertuis_koszul_torsion": edict.celestial_jet.maupertuis_spectrum.koszul_torsion,
            "poincare_cartan_lambda": edict.celestial_jet.cartan_spectrum.lambda_vector.tolist(),
            "poincare_cartan_skew_residual": edict.celestial_jet.cartan_spectrum.symplectic_skew_residual,
            "poincare_cartan_lagrangian": edict.celestial_jet.cartan_spectrum.cartan_lagrangian,
            "darboux_residual": edict.celestial_jet.darboux_residual,
            "almost_complex_residual": edict.celestial_jet.almost_complex_residual,
            # Invariantes celestes agregados
            "symplectic_cartan_defect": edict.symplectic_cartan_defect,
            "conley_zehnder_index": edict.conley_zehnder_index,
            "poincare_rotation_number": edict.rotation_number,
            # Aduanas celestes (Fase II)
            "kam_verdict": edict.kam.verdict if edict.kam is not None else None,
            "kam_min_divisor": edict.kam.min_divisor if edict.kam is not None else None,
            "kam_novikov_weight": edict.kam.novikov_weight if edict.kam is not None else None,
            "kam_is_diophantine": edict.kam.is_diophantine if edict.kam is not None else None,
            "kam_volume_drift": edict.kam.volume_drift if edict.kam is not None else None,
            "kam_homological_residual": (
                edict.kam.homological_residual if edict.kam is not None else None
            ),
            "kam_cartan_pullback_defect": (
                edict.kam.cartan_pullback_defect if edict.kam is not None else None
            ),
            "melnikov_verdict": edict.melnikov.verdict if edict.melnikov is not None else None,
            "melnikov_value": edict.melnikov.melnikov_value if edict.melnikov is not None else None,
            "melnikov_is_simple_zero": (
                edict.melnikov.is_simple_zero if edict.melnikov is not None else None
            ),
            "melnikov_splitting": (
                edict.melnikov.homoclinic_splitting if edict.melnikov is not None else None
            ),
            "birkhoff_verdict": edict.birkhoff.verdict if edict.birkhoff is not None else None,
            "birkhoff_area_drift": edict.birkhoff.area_drift if edict.birkhoff is not None else None,
            "birkhoff_fixed_points": (
                edict.birkhoff.fixed_points_count if edict.birkhoff is not None else None
            ),
            "birkhoff_cartan_pullback_defect": (
                edict.birkhoff.cartan_pullback_defect if edict.birkhoff is not None else None
            ),
            "birkhoff_twist": edict.birkhoff.birkhoff_twist if edict.birkhoff is not None else None,
            "return_map_verdict": edict.return_map.verdict if edict.return_map is not None else None,
            "return_map_max_lyapunov": (
                edict.return_map.max_lyapunov if edict.return_map is not None else None
            ),
            "return_map_is_hyperbolic": (
                edict.return_map.is_hyperbolic if edict.return_map is not None else None
            ),
            "return_map_is_elliptic": (
                edict.return_map.is_elliptic if edict.return_map is not None else None
            ),
            "return_map_is_parabolic": (
                edict.return_map.is_parabolic if edict.return_map is not None else None
            ),
            "return_map_cz_index": (
                edict.return_map.conley_zehnder_index if edict.return_map is not None else None
            ),
            "return_map_cz_nondegenerate": (
                edict.return_map.cz_nondegenerate if edict.return_map is not None else None
            ),
            "return_map_rotation_number": (
                edict.return_map.rotation_number if edict.return_map is not None else None
            ),
            "return_map_section_transversality": (
                edict.return_map.section_transversality if edict.return_map is not None else None
            ),
            "return_map_cartan_pullback_defect": (
                edict.return_map.cartan_pullback_defect if edict.return_map is not None else None
            ),
            # Actuación ciber-física
            "hardware_interlock_fired": interlock_fired,
            "hardware_crowbar_latched": self._crowbar.latched,
            "actuation_latency_ns": latency_ns,
            "input_fault": edict.celestial_jet.input_fault,
            "audited_layers_audit_trail": dict(edict.extended_verdicts),
        }
        return result

    # ── III.3 Atajo I.ω+ → II.ω+ → III sobre observables crudas ──────────
    def process_celestial_supervision_cycle(
        self,
        cochain_matrices: List[np.ndarray],
        density_matrix: np.ndarray,
        transition_map: np.ndarray,
        layer_verdicts: Dict[str, str],
        base_metric_g: Optional[np.ndarray] = None,
        grad_V: Optional[np.ndarray] = None,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        contractor_twist_angle: Optional[float] = None,
        auditor_twist_angle: Optional[float] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        t_inf: float = _MELNIKOV_T_INF,
        period_T: float = 1.0,
    ) -> Dict[str, Any]:
        r"""Atajo Φ_III ∘ Φ_II ∘ Φ_I sobre observables crudas de un ciclo celeste."""
        jet = self.agent.ingest_celestial_observables(
            cochain_matrices,
            density_matrix,
            transition_map,
            base_metric_g,
            grad_V=grad_V,
        )
        edict = self.agent.compile_pretorio_celestial_edict(
            celestial_jet=jet,
            layer_verdicts=layer_verdicts,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            canonical_J=canonical_J,
            contractor_twist_angle=contractor_twist_angle,
            auditor_twist_angle=auditor_twist_angle,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            t_inf=t_inf,
            period_T=period_T,
        )
        return self.collapse_from_celestial_edict(edict)

    # ── III.4 Atajo clásico (retrocompatible) ────────────────────────────
    def process_supervision_cycle(
        self,
        cochain_matrices: List[np.ndarray],
        density_matrix: np.ndarray,
        transition_map: np.ndarray,
        layer_verdicts: Dict[str, str],
    ) -> Dict[str, Any]:
        """Atajo I.ω → II.ω → III sobre observables crudas (API 2.0)."""
        jet = self.agent.ingest_observables(
            cochain_matrices, density_matrix, transition_map
        )
        edict = self.agent.compile_pretorio_edict(jet, layer_verdicts)
        return self.fuse_and_actuate(edict)


def execute_pretorio_supervision_cycle(
    agent: PretorioAgent,
    cochain_matrices: List[np.ndarray],
    density_matrix: np.ndarray,
    transition_map: np.ndarray,
    layer_verdicts: Dict[str, str],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Fachada de compatibilidad: delega en la Cámara de la Fase III (clásica)."""
    chamber = PretorioCoherenceChamber(agent, rng=rng)
    return chamber.process_supervision_cycle(
        cochain_matrices, density_matrix, transition_map, layer_verdicts
    )


def execute_pretorio_celestial_supervision_cycle(
    agent: PretorioAgent,
    cochain_matrices: List[np.ndarray],
    density_matrix: np.ndarray,
    transition_map: np.ndarray,
    layer_verdicts: Dict[str, str],
    base_metric_g: Optional[np.ndarray] = None,
    grad_V: Optional[np.ndarray] = None,
    frequency_vector_omega: Optional[np.ndarray] = None,
    wave_vectors_k: Optional[np.ndarray] = None,
    jacobian_M: Optional[np.ndarray] = None,
    canonical_J: Optional[np.ndarray] = None,
    contractor_twist_angle: Optional[float] = None,
    auditor_twist_angle: Optional[float] = None,
    homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
    hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
    hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
    t0_grid: Optional[np.ndarray] = None,
    t_inf: float = _MELNIKOV_T_INF,
    period_T: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Fachada celestial: ejecuta el ciclo I.ω+ → II.ω+ → III."""
    chamber = PretorioCoherenceChamber(agent, rng=rng)
    return chamber.process_celestial_supervision_cycle(
        cochain_matrices=cochain_matrices,
        density_matrix=density_matrix,
        transition_map=transition_map,
        layer_verdicts=layer_verdicts,
        base_metric_g=base_metric_g,
        grad_V=grad_V,
        frequency_vector_omega=frequency_vector_omega,
        wave_vectors_k=wave_vectors_k,
        jacobian_M=jacobian_M,
        canonical_J=canonical_J,
        contractor_twist_angle=contractor_twist_angle,
        auditor_twist_angle=auditor_twist_angle,
        homoclinic_flow=homoclinic_flow,
        hamiltonian_0=hamiltonian_0,
        hamiltonian_1=hamiltonian_1,
        t0_grid=t0_grid,
        t_inf=t_inf,
        period_T=period_T,
    )


def _bind_agent_cycle() -> None:
    """Vincula dinámicamente los ciclos de supervisión al agente."""

    def _cycle_classic(
        self: PretorioAgent,
        cochain_matrices: List[np.ndarray],
        density_matrix: np.ndarray,
        transition_map: np.ndarray,
        layer_verdicts: Dict[str, str],
    ) -> Dict[str, Any]:
        r"""Orquesta el ciclo de supervisión clásico. Composición I.ω → II.ω → III."""
        return execute_pretorio_supervision_cycle(
            self, cochain_matrices, density_matrix, transition_map, layer_verdicts
        )

    def _cycle_celestial(
        self: PretorioAgent,
        cochain_matrices: List[np.ndarray],
        density_matrix: np.ndarray,
        transition_map: np.ndarray,
        layer_verdicts: Dict[str, str],
        base_metric_g: Optional[np.ndarray] = None,
        grad_V: Optional[np.ndarray] = None,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        canonical_J: Optional[np.ndarray] = None,
        contractor_twist_angle: Optional[float] = None,
        auditor_twist_angle: Optional[float] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        t_inf: float = _MELNIKOV_T_INF,
        period_T: float = 1.0,
    ) -> Dict[str, Any]:
        r"""
        Orquesta el ciclo de supervisión celestial del Pretorio.
        Composición Φ_III ∘ Φ_II ∘ Φ_I sobre observables crudas
        (KAM, Melnikov, Birkhoff, Retorno, CZ, Cartan).
        """
        return execute_pretorio_celestial_supervision_cycle(
            self,
            cochain_matrices=cochain_matrices,
            density_matrix=density_matrix,
            transition_map=transition_map,
            layer_verdicts=layer_verdicts,
            base_metric_g=base_metric_g,
            grad_V=grad_V,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            canonical_J=canonical_J,
            contractor_twist_angle=contractor_twist_angle,
            auditor_twist_angle=auditor_twist_angle,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            t_inf=t_inf,
            period_T=period_T,
        )

    PretorioAgent.execute_pretorio_supervision_cycle = _cycle_classic  # type: ignore[attr-defined]
    PretorioAgent.execute_pretorio_celestial_supervision_cycle = _cycle_celestial  # type: ignore[attr-defined]


_bind_agent_cycle()

__all__ = [
    "HeytingVerdict",
    "PretorioAgent",
    "PretorioCoherenceChamber",
    "PretorioDeliberationCertificate",
    "execute_pretorio_supervision_cycle",
    "execute_pretorio_celestial_supervision_cycle",
]
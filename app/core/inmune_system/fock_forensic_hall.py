# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Fock Forensic Hall (Salón de Eventos del Espacio de Fock)           ║
║ Ruta   : app/core/inmune_system/fock_forensic_hall.py                        ║
║ Versión: 3.0.0-Doctoral-CAR-GKSL-Weyl-KyFan-Heyting-Secure                   ║
║                                                                              ║
║ ARQUITECTURA DE FASES ANIDADAS (morfismos, no meras secciones):              ║
║                                                                              ║
║   Fase 1  --η-->  Fase 2  --χ-->  Fase 3                                     ║
║   Observe+Orient    Decide (Ω₃)    Act + sello de telemetría                 ║
║                                                                              ║
║   η  = _phase1_terminal_morphism  = objeto inicial de la Fase 2              ║
║   χ  = clasificador de subobjetos = objeto inicial de la Fase 3              ║
║                                                                              ║
║ FÍSICA (sin metáfora suelta):                                                ║
║   • ℋ = ℱ_-(ℂ^{n}) ≅ ℂ^{2ⁿ},  CAR vía Jordan–Wigner.                        ║
║   • H = Σ_j N_j,  L = a₀a₁ (n≥2)  ó  L = a₀ (n=1).                           ║
║     Canal de co-aniquilación de par (no unital, no unital-dual).             ║
║   • Semigrupo: ρ(t) = e^{tℒ}(ρ₀), ℒ en forma GKSL (CPTP, traza-preservante).║
║   • T^{μν} = p^μ p^ν  (polvo / fotón puntual). Sin ∂g no hay ∇_ν T^{μν}      ║
║     puntual; se certifican identidades algebraicas y la traza de Weyl.       ║
║   • Exergía = (⟨N⟩₀ − ⟨N⟩_t)/⟨N⟩₀  ∈ [0,1].                                  ║
║   • Ω₃ = {COHERENT < DEGRADED < VETOED} cadena de Heyting.                   ║
║   • Fase 3: ISR *simulada*. No hay GPIO, ESP32 ni BT151 reales.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import scipy.linalg as la
from scipy.special import xlogy

try:
    from scipy.sparse.linalg import LinearOperator, expm_multiply
except ImportError:  # pragma: no cover
    LinearOperator = None  # type: ignore[misc, assignment]
    expm_multiply = None  # type: ignore[misc, assignment]


logger = logging.getLogger("APU.Physics.FockForensicHall")

# =============================================================================
# Constantes espectrales (IEEE-754 float64, Wilkinson–Higham–Kahan)
# =============================================================================
_EPS64: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 50.0 * _EPS64
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0

_HERMITIAN_RTOL: Final[float] = 1e-12
_PSD_EIGEN_FLOOR: Final[float] = -1e-12
_TRACE_ATOL: Final[float] = 1e-10
_METRIC_SYMMETRY_RTOL: Final[float] = 1e-12
_MOMENTUM_NORM_FLOOR: Final[float] = 1e-12
_CAR_TOLERANCE: Final[float] = 1e-10
_SPECTRAL_RESIDUAL_TOL: Final[float] = 1e-10
_TP_RESIDUAL_TOL: Final[float] = 1e-10
_ENERGY_COND_TOL: Final[float] = 1e-10
_PARTICLE_MONOTONE_TOL: Final[float] = 1e-10
_VACUUM_MONOTONE_TOL: Final[float] = 1e-10
_EXPM_DENSE_DIM_MAX: Final[int] = 32

_DEFAULT_ENTROPY_THRESHOLD: Final[float] = 0.5
_DEFAULT_DIVERGENCE_THRESHOLD: Final[float] = 1e-4
_DEFAULT_LINDBLAD_DT: Final[float] = 1.0e-2
_DEFAULT_RNG_SEED: Final[int] = 42
_DIGEST_ROUND_DECIMALS: Final[int] = 12

_PAULI_I: Final[np.ndarray] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
_PAULI_Z: Final[np.ndarray] = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
# c|1⟩ = |0⟩  ⇒  c = |0⟩⟨1|
_FERMION_C: Final[np.ndarray] = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)


# =============================================================================
# Tipos ordinales
# =============================================================================
class HeytingVerdict(str, Enum):
    """Cadena de Heyting Ω₃: COHERENT ⊥ < DEGRADED < VETOED ⊤."""

    COHERENT = "COHERENT"
    DEGRADED = "DEGRADED"
    VETOED = "VETOED"

    @property
    def rank(self) -> int:
        return {self.COHERENT: 0, self.DEGRADED: 1, self.VETOED: 2}[self]


class HeytingOmega3:
    r"""
    Ω₃ = {0 = COHERENT, ½ = DEGRADED, 1 = VETOED} con el orden de cadena.

        a ∧ b = min(a, b)
        a ∨ b = max(a, b)
        a → b = ⊤  si a ≤ b,  else b
        ¬a    = a → ⊥
    """

    @classmethod
    def meet(cls, a: HeytingVerdict, b: HeytingVerdict) -> HeytingVerdict:
        return a if a.rank <= b.rank else b

    @classmethod
    def join(cls, a: HeytingVerdict, b: HeytingVerdict) -> HeytingVerdict:
        return a if a.rank >= b.rank else b

    @classmethod
    def implies(cls, a: HeytingVerdict, b: HeytingVerdict) -> HeytingVerdict:
        return HeytingVerdict.VETOED if a.rank <= b.rank else b

    @classmethod
    def neg(cls, a: HeytingVerdict) -> HeytingVerdict:
        return cls.implies(a, HeytingVerdict.COHERENT)

    @classmethod
    def join_all(cls, elems: Sequence[HeytingVerdict]) -> HeytingVerdict:
        acc = HeytingVerdict.COHERENT
        for e in elems:
            acc = cls.join(acc, e)
        return acc

    @classmethod
    def classify(
        cls,
        critical: Sequence[bool],
        marginal: Sequence[bool],
    ) -> HeytingVerdict:
        atoms: List[HeytingVerdict] = []
        for flag in critical:
            atoms.append(HeytingVerdict.VETOED if flag else HeytingVerdict.COHERENT)
        for flag in marginal:
            atoms.append(HeytingVerdict.DEGRADED if flag else HeytingVerdict.COHERENT)
        return cls.join_all(atoms)


# =============================================================================
# Utilidades C* / Banach / IEEE-754
# =============================================================================
def _as_c128(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.size == 0:
        raise ValueError(f"{name} está vacío")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene NaN o Inf")
    return np.array(arr, dtype=np.complex128, copy=True)


def _as_f64(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.size == 0:
        raise ValueError(f"{name} está vacío")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene NaN o Inf")
    return np.array(arr, dtype=np.float64, copy=True)


def _freeze(a: np.ndarray) -> np.ndarray:
    out = np.array(a, copy=True)
    out.setflags(write=False)
    return out


def _hermite(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def _scaled_tol(norm: float, rtol: float, atol: float) -> float:
    return float(max(atol, rtol * max(norm, 1.0)))


def _safe_xlogx(x: np.ndarray) -> np.ndarray:
    """x log x con convención 0 log 0 = 0."""
    xr = np.real(np.asarray(x, dtype=np.float64))
    return xlogy(xr, xr)


def _vec(m: np.ndarray) -> np.ndarray:
    return m.reshape(-1, order="F")


def _unvec(v: np.ndarray, dim: int) -> np.ndarray:
    return v.reshape((dim, dim), order="F")


# =============================================================================
# Paquetes inmutables (objetos de las fases)
# =============================================================================
@dataclass(frozen=True, slots=True)
class PhaseOneEnginePacket:
    """
    Objeto terminal de la Fase 1 / objeto inicial de la Fase 2.

    Contiene el espectro certificado, el semigrupo GKSL, los invariantes
    de canal (N, vacío, CAR, TP) y el tensor T^{μν} auditado.
    """

    rho_initial: np.ndarray
    rho_final: np.ndarray
    eigenvalues_initial: np.ndarray
    eigenvalues_final: np.ndarray
    fock_entropy: float
    renyi_entropy_2: float
    min_entropy: float
    exergy_efficiency: float
    purity_initial: float
    purity_final: float
    particle_number_initial: float
    particle_number_final: float
    particle_number_monotone: bool
    vacuum_fidelity_initial: float
    vacuum_fidelity_final: float
    vacuum_fidelity_monotone: bool
    energy_momentum_tensor: np.ndarray
    momentum_divergence_residual: float
    tensor_symmetry_residual: float
    weyl_invariant_trace: float
    trace_energy_momentum: float
    energy_condition_ok: bool
    car_residual: float
    spectral_residual_initial: float
    spectral_residual_final: float
    liouvillian_tp_residual: float
    metric_signature: Tuple[int, int, int]
    gamma_annihilation: float
    time_step: float

    def __post_init__(self) -> None:
        if self.rho_initial.ndim != 2 or self.rho_final.ndim != 2:
            raise ValueError("rho_initial y rho_final deben ser matrices")
        if self.eigenvalues_initial.shape != self.eigenvalues_final.shape:
            raise ValueError("espectros de dimensión distinta")


@dataclass(frozen=True, slots=True)
class PhaseTwoDecisionPacket:
    """
    Objeto terminal de la Fase 2 / objeto inicial de la Fase 3.

    Transporta el veredicto Ω₃, los predicados atómicos y χ (True ⇔ ⊤).
    """

    heyting_verdict: str
    entropy_violation: bool
    divergence_violation: bool
    particle_violation: bool
    vacuum_violation: bool
    car_violation: bool
    energy_condition_violation: bool
    tp_violation: bool
    subobject_classifier: bool
    implication_coherent_to_verdict: str
    join_of_atoms: str


@dataclass(frozen=True, slots=True)
class TelemetryStamp:
    """
    Sello inmutable de telemetría (Fase 3).
    `to_dict()` exporta exactamente las claves de la v2 (compatibilidad ABI).
    """

    heyting_verdict: str
    fock_entropy: float
    exergy_efficiency: float
    energy_momentum_trace: float
    momentum_divergence_residual: float
    quantum_state_purity: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    weyl_invariant_trace: float = 0.0
    particle_number_final: float = 0.0
    car_residual: float = 0.0
    forensic_digest: str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Interfaz original: ocho claves, ni una más."""
        return {
            "heyting_verdict": self.heyting_verdict,
            "fock_entropy": self.fock_entropy,
            "exergy_efficiency": self.exergy_efficiency,
            "energy_momentum_trace": self.energy_momentum_trace,
            "momentum_divergence_residual": self.momentum_divergence_residual,
            "quantum_state_purity": self.quantum_state_purity,
            "hardware_interlock_fired": self.hardware_interlock_fired,
            "actuation_latency_ns": self.actuation_latency_ns,
        }

    def to_dict_extended(self) -> Dict[str, Any]:
        base = self.to_dict()
        base.update(
            {
                "weyl_invariant_trace": self.weyl_invariant_trace,
                "particle_number_final": self.particle_number_final,
                "car_residual": self.car_residual,
                "forensic_digest": self.forensic_digest,
            }
        )
        return base


# =============================================================================
# Motor — tres fases anidadas
# =============================================================================
class FockForensicHall:
    r"""
    Pista de baile cuántica sobre ℱ_-(ℂ^{n}).

    Métodos públicos (contrato con FockForensicHallAgent):

        solve_lindblad_annihilation(ρ, γ, dt) → (ρ_t, S, η_ex)
        compute_energy_momentum_tensor(p, g)  → (T^{μν}, r_div)
        execute_forensic_cycle(...)           → dict  (ABI v2)

    Fases anidadas:

        _phase1_observe_orient  →  η(_phase1_terminal_morphism)
                └──────────────────────────────────────────┐
        _phase2_heyting_decision  ←  η(packet)             ┘
                └─ χ
        _phase3_act_and_telemetry ← χ
    """

    def __init__(
        self,
        num_modes: int = 2,
        reg_param: float = 1e-15,
        entropy_threshold: float = _DEFAULT_ENTROPY_THRESHOLD,
        divergence_threshold: float = _DEFAULT_DIVERGENCE_THRESHOLD,
        rng_seed: Optional[int] = _DEFAULT_RNG_SEED,
        lindblad_time_step: float = _DEFAULT_LINDBLAD_DT,
    ) -> None:
        if num_modes < 1:
            raise ValueError("num_modes debe ser ≥ 1")
        if lindblad_time_step < 0.0:
            raise ValueError("lindblad_time_step debe ser ≥ 0")

        self._modes: Final[int] = int(num_modes)
        self._reg: Final[float] = max(float(reg_param), _WILKINSON_FLOOR)
        self._dim_fock: Final[int] = 1 << self._modes
        self._entropy_threshold: Final[float] = max(float(entropy_threshold), _WILKINSON_FLOOR)
        self._divergence_threshold: Final[float] = max(
            float(divergence_threshold), _WILKINSON_FLOOR
        )
        self._dt_default: Final[float] = float(lindblad_time_step)
        self._rng = np.random.default_rng(rng_seed)

        self._a_ops, self._adag_ops, car_res = self._build_canonical_fermi_operators()
        self._car_residual: Final[float] = float(car_res)
        if self._car_residual > _CAR_TOLERANCE:
            logger.warning(
                "Residuo CAR = %.3e > %.3e (Jordan–Wigner numéricamente degradado).",
                self._car_residual,
                _CAR_TOLERANCE,
            )

        self._number_op: Final[np.ndarray] = self._build_number_operator()
        self._hamiltonian: Final[np.ndarray] = _hermite(self._number_op)
        self._jump_ops, self._jump_ldl = self._build_jump_operators()
        self._super_cache: Dict[float, np.ndarray] = {}

        logger.debug(
            "FockForensicHall n=%d dim=%d CAR=%.3e n_L=%d",
            self._modes,
            self._dim_fock,
            self._car_residual,
            len(self._jump_ops),
        )

    # -------------------------------------------------------------------------
    # Accesores de calibre (útiles al agente / ensayos)
    # -------------------------------------------------------------------------
    @property
    def num_modes(self) -> int:
        return self._modes

    @property
    def dim_fock(self) -> int:
        return self._dim_fock

    @property
    def number_operator(self) -> np.ndarray:
        return self._number_op

    @property
    def annihilators(self) -> Tuple[np.ndarray, ...]:
        return tuple(self._a_ops)

    @property
    def car_residual(self) -> float:
        return self._car_residual

    # =========================================================================
    # FASE 1 — OBSERVE + ORIENT
    #   1.1  Jordan–Wigner + CAR
    #   1.2  validación C* del estado y de (p, g)
    #   1.3  espectro certificado (Weyl–Wilkinson)
    #   1.4  semigrupo GKSL exacto (expm / Krylov)
    #   1.5  exergía de ocupación y entropías
    #   1.6  T^{μν} polvo + identidades algebraicas + Weyl
    #   1.7  morfismo terminal η  →  inicio formal de la Fase 2
    # =========================================================================
    def _kron_n(self, ops: Sequence[np.ndarray]) -> np.ndarray:
        acc = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        for op in ops:
            acc = np.kron(acc, op)
        return acc

    def _build_canonical_fermi_operators(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        r"""
        Operadores de Jordan–Wigner en ℱ_-(ℂ^{n}):

            a_j = Z^{⊗j} ⊗ σ⁻ ⊗ I^{⊗(n−j−1)}

        que realizan las CAR

            {a_i, a_j} = 0,   {a_i†, a_j†} = 0,   {a_i, a_j†} = δ_{ij} I.

        La verificación se hace sobre las listas locales *antes* de
        publicarlas en `self` (el v2 consultaba `self._a_ops` aún no ligado).
        """
        a_ops: List[np.ndarray] = []
        adag_ops: List[np.ndarray] = []
        for i in range(self._modes):
            factors: List[np.ndarray] = []
            for j in range(self._modes):
                if j < i:
                    factors.append(_PAULI_Z)
                elif j == i:
                    factors.append(_FERMION_C)
                else:
                    factors.append(_PAULI_I)
            a_i = self._kron_n(factors)
            a_ops.append(a_i)
            adag_ops.append(a_i.conj().T.copy())
        residual = self._car_max_residual(a_ops, adag_ops)
        return a_ops, adag_ops, residual

    def _car_max_residual(
        self,
        a_ops: Sequence[np.ndarray],
        adag_ops: Sequence[np.ndarray],
    ) -> float:
        ident = np.eye(self._dim_fock, dtype=np.complex128)
        max_res = 0.0
        n = len(a_ops)
        for i in range(n):
            for j in range(n):
                aa = a_ops[i] @ a_ops[j] + a_ops[j] @ a_ops[i]
                max_res = max(max_res, float(np.linalg.norm(aa, ord="fro")))
                adad = adag_ops[i] @ adag_ops[j] + adag_ops[j] @ adag_ops[i]
                max_res = max(max_res, float(np.linalg.norm(adad, ord="fro")))
                aad = a_ops[i] @ adag_ops[j] + adag_ops[j] @ a_ops[i]
                target = ident if i == j else 0.0
                max_res = max(max_res, float(np.linalg.norm(aad - target, ord="fro")))
        return max_res

    def _build_number_operator(self) -> np.ndarray:
        n_op = np.zeros((self._dim_fock, self._dim_fock), dtype=np.complex128)
        for a, ad in zip(self._a_ops, self._adag_ops):
            n_op += ad @ a
        return _hermite(n_op)

    def _build_jump_operators(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        r"""
        Canal de la gala:
            n = 1  →  L = a₀          (decaimiento simple al vacío)
            n ≥ 2  →  L = a₀ a₁       (co-aniquilación de par e⁻/e⁺)

        Un único canal mantiene el semigrupo uniparamétrico y el
        significado exergético de *un* choque.
        """
        if self._modes >= 2:
            jumps = [self._a_ops[0] @ self._a_ops[1]]
        else:
            jumps = [self._a_ops[0]]
        ldls = [_hermite(L.conj().T @ L) for L in jumps]
        return jumps, ldls

    def _validate_density_matrix(self, rho: np.ndarray, name: str = "rho") -> np.ndarray:
        """Estado cuántico en M_d(ℂ): cuadrada, finita, hermítica, PSD, Tr=1, d=2ⁿ."""
        rho_c = _as_c128(rho, name)
        if rho_c.ndim != 2 or rho_c.shape[0] != rho_c.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada; se recibió {rho_c.shape}")
        if rho_c.shape[0] != self._dim_fock:
            raise ValueError(
                f"dim({name})={rho_c.shape[0]} ≠ 2^{self._modes}={self._dim_fock}"
            )
        herm_res = float(np.linalg.norm(rho_c - rho_c.conj().T, ord="fro"))
        herm_tol = _scaled_tol(
            float(np.linalg.norm(rho_c, ord="fro")), _HERMITIAN_RTOL, _WILKINSON_FLOOR
        )
        if herm_res > herm_tol:
            raise ValueError(
                f"{name} no es hermítica: ‖ρ−ρ†‖_F={herm_res:.3e} > {herm_tol:.3e}"
            )
        rho_h = _hermite(rho_c)
        eigvals = la.eigvalsh(rho_h)
        if np.any(eigvals < _PSD_EIGEN_FLOOR):
            raise ValueError(
                f"{name} no es PSD: λ_min={float(np.min(eigvals)):.3e}"
            )
        tr = float(np.real(np.trace(rho_h)))
        if abs(tr - 1.0) > _TRACE_ATOL:
            raise ValueError(f"{name} debe tener traza 1; se obtuvo {tr:.6e}")
        return rho_h

    def _validate_momentum_vector(self, p: np.ndarray) -> np.ndarray:
        vec = _as_f64(np.asarray(p, dtype=np.float64), "momentum_vector")
        if vec.ndim != 1:
            raise ValueError(f"momentum_vector debe ser (N,); se recibió {vec.shape}")
        if float(np.linalg.norm(vec)) < _MOMENTUM_NORM_FLOOR:
            logger.warning("momentum_vector de norma casi nula (p ≈ 0).")
        return vec

    def _validate_metric_tensor(self, g: np.ndarray, dim_p: int) -> np.ndarray:
        """
        Métrica real simétrica no degenerada.  NO se exige definida positiva:
        una g lorentziana es el caso físicamente relevante para T^{μν}.
        """
        met = _as_f64(np.asarray(g, dtype=np.float64), "metric_tensor")
        if met.ndim != 2 or met.shape[0] != met.shape[1]:
            raise ValueError(f"metric_tensor debe ser cuadrada; se recibió {met.shape}")
        if met.shape[0] != dim_p:
            raise ValueError(
                f"dim(g)={met.shape[0]} incompatible con dim(p)={dim_p}"
            )
        g_h = 0.5 * (met + met.T)
        sym_res = float(np.linalg.norm(met - met.T, ord="fro"))
        sym_tol = _scaled_tol(
            float(np.linalg.norm(met, ord="fro")),
            _METRIC_SYMMETRY_RTOL,
            _WILKINSON_FLOOR,
        )
        if sym_res > sym_tol:
            raise ValueError(
                f"metric_tensor no es simétrica: ‖g−gᵀ‖_F={sym_res:.3e}"
            )
        ev = np.real(la.eigvalsh(g_h))
        if np.any(np.abs(ev) < self._reg):
            raise ValueError(
                f"metric_tensor degenerada: min|λ|={float(np.min(np.abs(ev))):.3e}"
            )
        return g_h

    def _certified_eigh(
        self, herm: np.ndarray, name: str
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """A = UΛU† con residuo de Wilkinson r = ‖AU−UΛ‖_F / max(1,‖A‖_F)."""
        eigvals, eigvecs = la.eigh(herm)
        recon = herm @ eigvecs - eigvecs @ np.diag(eigvals)
        denom = max(float(np.linalg.norm(herm, ord="fro")), 1.0)
        residual = float(np.linalg.norm(recon, ord="fro") / denom)
        if residual > _SPECTRAL_RESIDUAL_TOL:
            logger.warning("Residuo espectral elevado en %s: r=%.3e.", name, residual)
        return np.real(eigvals), eigvecs, residual

    def _compute_von_neumann_entropy(self, eigenvalues: np.ndarray) -> float:
        lam = np.real(np.asarray(eigenvalues, dtype=np.float64))
        lam = lam[lam > _WILKINSON_FLOOR]
        if lam.size == 0:
            return 0.0
        return float(-np.sum(_safe_xlogx(lam)))

    def _compute_renyi_entropy(self, eigenvalues: np.ndarray, alpha: float) -> float:
        if alpha <= 0.0 or abs(alpha - 1.0) < _WILKINSON_FLOOR:
            raise ValueError("Rényi exige α>0, α≠1")
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), 0.0, None)
        power = float(np.sum(np.power(lam, alpha)))
        if power <= _WILKINSON_FLOOR:
            return 0.0
        return float(np.log(power) / (1.0 - alpha))

    def _compute_min_entropy(self, eigenvalues: np.ndarray) -> float:
        lam_max = float(np.max(np.real(eigenvalues))) if eigenvalues.size else 0.0
        if lam_max <= _WILKINSON_FLOOR:
            return 0.0
        return float(-np.log(lam_max))

    def _compute_purity_from_eigenvalues(self, eigenvalues: np.ndarray) -> float:
        return float(np.real(np.sum(np.square(eigenvalues))))

    def _project_state(self, rho: np.ndarray) -> np.ndarray:
        """
        Proyección numérica al símplice de estados: hermitización, recorte
        de autovalores *negativos* a 0 (nunca un suelo ε>0) y renormalización.
        """
        rho_h = _hermite(rho)
        eigvals, eigvecs = la.eigh(rho_h)
        eigvals = np.clip(np.real(eigvals), 0.0, None)
        total = float(np.sum(eigvals))
        if total <= _WILKINSON_FLOOR:
            raise ValueError("traza numéricamente nula tras proyección PSD")
        eigvals = eigvals / total
        return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    def _apply_gksl_generator(self, rho: np.ndarray, gamma: float) -> np.ndarray:
        r"""
        ℒ(ρ) = −i[H,ρ] + γ Σ_k (L_k ρ L_k† − ½ {L_k† L_k, ρ}).
        Completamente positivo y traza-preservante por construcción GKSL.
        """
        h_rho = self._hamiltonian @ rho
        comm = -1.0j * (h_rho - h_rho.conj().T + _hermite(rho @ self._hamiltonian) * 0.0)
        # −i[H,ρ] = −i(Hρ − ρH); la línea anterior evita un segundo producto si H=H†:
        comm = -1.0j * (h_rho - rho @ self._hamiltonian)
        diss = np.zeros_like(rho)
        for jump, ldl in zip(self._jump_ops, self._jump_ldl):
            diss += jump @ rho @ jump.conj().T - 0.5 * (ldl @ rho + rho @ ldl)
        return comm + float(gamma) * diss

    def _gksl_superoperator(self, gamma: float) -> np.ndarray:
        """ℒ en convención columna (F-order): vec(AXB) = (Bᵀ ⊗ A) vec(X)."""
        key = float(gamma)
        cached = self._super_cache.get(key)
        if cached is not None:
            return cached
        d = self._dim_fock
        ident = np.eye(d, dtype=np.complex128)
        ham = self._hamiltonian
        ell = -1.0j * (np.kron(ident, ham) - np.kron(ham.T, ident))
        scale = float(gamma)
        for jump, ldl in zip(self._jump_ops, self._jump_ldl):
            ell = ell + scale * (
                np.kron(np.conj(jump), jump)
                - 0.5 * (np.kron(ident, ldl) + np.kron(ldl.T, ident))
            )
        self._super_cache[key] = ell
        return ell

    def _apply_semigroup(
        self, rho: np.ndarray, gamma: float, time_step: float
    ) -> Tuple[np.ndarray, float]:
        """
        ρ(t) = e^{tℒ}(ρ).  Denso si d≤32; Krylov (`expm_multiply`) si no.
        Devuelve (ρ_t, residuo_TP del generador sobre ρ).
        """
        d = self._dim_fock
        rhs0 = self._apply_gksl_generator(rho, gamma)
        tp_res = float(abs(np.real(np.trace(rhs0))))
        if time_step == 0.0 or gamma == 0.0 and np.allclose(self._hamiltonian, 0.0):
            if time_step == 0.0:
                return rho, tp_res

        if d <= _EXPM_DENSE_DIM_MAX:
            prop = la.expm(self._gksl_superoperator(gamma) * float(time_step))
            rho_t = _unvec(prop @ _vec(rho), d)
        elif LinearOperator is not None and expm_multiply is not None:
            def matvec(v: np.ndarray) -> np.ndarray:
                rho_v = _unvec(np.asarray(v, dtype=np.complex128), d)
                return _vec(self._apply_gksl_generator(rho_v, gamma))

            oper = LinearOperator(
                shape=(d * d, d * d),
                matvec=matvec,
                dtype=np.complex128,
            )
            rho_t = _unvec(
                expm_multiply(oper, _vec(rho), start=0.0, stop=float(time_step), endpoint=True),
                d,
            )
        else:  # pragma: no cover - fallback clásico de orden 4
            logger.warning("expm_multiply ausente: se integra GKSL con RK4.")
            n_sub = max(8, int(np.ceil(abs(time_step) / 1.0e-3)))
            h = float(time_step) / n_sub
            rho_t = rho
            for _ in range(n_sub):
                k1 = self._apply_gksl_generator(rho_t, gamma)
                k2 = self._apply_gksl_generator(rho_t + 0.5 * h * k1, gamma)
                k3 = self._apply_gksl_generator(rho_t + 0.5 * h * k2, gamma)
                k4 = self._apply_gksl_generator(rho_t + h * k3, gamma)
                rho_t = rho_t + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return self._project_state(rho_t), tp_res

    def _particle_number(self, rho: np.ndarray) -> float:
        return float(np.real(np.trace(self._number_op @ rho)))

    def _vacuum_fidelity(self, rho: np.ndarray) -> float:
        """F(ρ,|0⟩⟨0|) = ⟨0|ρ|0⟩ en la base JW (índice 0 = vacío)."""
        return float(np.real(rho[0, 0]))

    def _occupation_exergy(self, n0: float, n1: float) -> float:
        if n0 <= _WILKINSON_FLOOR:
            return 0.0
        return float(np.clip((n0 - n1) / n0, 0.0, 1.0))

    def solve_lindblad_annihilation(
        self,
        rho_initial: np.ndarray,
        gamma_annihilation: float,
        time_step: float = 0.01,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        Un paso del semigrupo de co-aniquilación

            dρ/dt = −i[H,ρ] + γ (LρL† − ½{L†L,ρ}),

        con H = N y L = a₀a₁ (ó a₀ si n=1).

        Returns:
            (ρ_t, S(ρ_t), η_ex)  con η_ex = (⟨N⟩₀−⟨N⟩_t)/⟨N⟩₀.
        """
        if gamma_annihilation < 0.0:
            raise ValueError("gamma_annihilation debe ser ≥ 0 (disipador GKSL)")
        if time_step < 0.0:
            raise ValueError("time_step debe ser ≥ 0")

        rho0 = self._validate_density_matrix(rho_initial, "rho_initial")
        rho_t, _tp = self._apply_semigroup(rho0, float(gamma_annihilation), float(time_step))
        eig_t = np.real(la.eigvalsh(_hermite(rho_t)))
        entropy = self._compute_von_neumann_entropy(eig_t)
        exergy = self._occupation_exergy(self._particle_number(rho0), self._particle_number(rho_t))
        return rho_t, entropy, exergy

    def _regularized_metric_inverse(self, metric: np.ndarray) -> np.ndarray:
        r"""
        Inversa espectral con suelo *con signo*:

            λ ↦ copysign(max(|λ|, ε), λ)     (ε = reg_param)

        Conserva la signatura de Sylvester; el `clip(λ, ε, None)` del v2
        convertía toda g lorentziana en riemanniana.
        """
        eigvals, eigvecs = la.eigh(metric)
        ev = np.real(eigvals)
        signs = np.where(ev >= 0.0, 1.0, -1.0)
        ev_safe = signs * np.maximum(np.abs(ev), self._reg)
        inv = (eigvecs * (1.0 / ev_safe)) @ eigvecs.T
        return np.real(inv)

    def _metric_signature(self, metric: np.ndarray) -> Tuple[int, int, int]:
        ev = np.real(la.eigvalsh(metric))
        n_plus = int(np.sum(ev > _ENERGY_COND_TOL))
        n_minus = int(np.sum(ev < -_ENERGY_COND_TOL))
        n_zero = int(ev.size - n_plus - n_minus)
        return n_plus, n_minus, n_zero

    def _energy_conditions(
        self,
        tensor_up: np.ndarray,
        metric: np.ndarray,
        signature: Tuple[int, int, int],
    ) -> Tuple[bool, float]:
        r"""
        Polvo T^{μν}=p^μ p^ν ⇒ T_{μν}=p_μ p_ν ⇒ T(t,t)=(p·t)² ≥ 0 (WEC trivial).
        Se audita además T^μ_ν = g_{να} T^{μα} por si el motor externo
        inyectara un T distinto.
        """
        t_up = 0.5 * (np.real(tensor_up) + np.real(tensor_up).T)
        g_h = 0.5 * (np.real(metric) + np.real(metric).T)
        t_down = g_h @ t_up @ g_h
        n_plus, n_minus, n_zero = signature
        if n_zero > 0:
            return False, float("inf")
        ev = np.real(la.eigvalsh(0.5 * (t_down + t_down.T)))
        if n_minus == 0:
            viol = float(max(0.0, -np.min(ev)))
            return viol <= _ENERGY_COND_TOL, viol
        viol = float(max(0.0, -np.min(ev)))
        return viol <= _ENERGY_COND_TOL * max(1.0, float(np.max(np.abs(ev)))), viol

    def compute_energy_momentum_tensor(
        self,
        momentum_vector: np.ndarray,
        metric_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Tensor de polvo / fotón puntual

            T^{μν} = p^μ p^ν,    p^μ = g^{μν} p_ν.

        Residuo algebraico (no es ∇_ν T^{μν}, que exige ∂g y Γ[g]):

            r = ‖T−Tᵀ‖_F + ‖p_μ − g_{μν}p^ν‖₂ + ‖T^{μν}p_ν − (p·p) p^μ‖₂.

        El v2 usaba T = pp + ½ g^{-1}(p·p) y medía ‖Tp − p(p·p)‖, identidad
        que ese T nunca satisface.
        """
        p_down = self._validate_momentum_vector(momentum_vector)
        g_h = self._validate_metric_tensor(metric_tensor, p_down.shape[0])
        g_inv = self._regularized_metric_inverse(g_h)
        p_up = g_inv @ p_down
        tensor_t = np.outer(p_up, p_up)
        p_sq = float(np.dot(p_up, p_down))
        sym_res = float(np.linalg.norm(tensor_t - tensor_t.T, ord="fro"))
        lower_res = float(np.linalg.norm(p_down - g_h @ p_up))
        cons_res = float(np.linalg.norm(tensor_t @ p_down - p_up * p_sq))
        return tensor_t, float(sym_res + lower_res + cons_res)

    def _weyl_invariant_trace(self, tensor_up: np.ndarray, metric: np.ndarray) -> float:
        """g_{μν} T^{μν}  (traza invariante; np.trace(T^{μν}) no lo es)."""
        return float(np.real(np.einsum("ij,ij->", metric, tensor_up)))

    def _phase1_observe_orient(
        self,
        rho_initial: np.ndarray,
        momentum_vector: np.ndarray,
        metric_tensor: np.ndarray,
        gamma_annihilation: float,
        time_step: Optional[float] = None,
    ) -> PhaseOneEnginePacket:
        """
        [FASE 1 · Observe + Orient]

        Valida (ρ₀, p, g), integra e^{tℒ}, certifica espectros y T^{μν},
        y emite el paquete que **es** el objeto inicial de la Fase 2 vía η.
        """
        logger.info("Fase 1 (Observe+Orient): captura de ρ₀ y orientación de calibre.")
        if gamma_annihilation < 0.0:
            raise ValueError("gamma_annihilation debe ser ≥ 0")
        dt = self._dt_default if time_step is None else float(time_step)
        if dt < 0.0:
            raise ValueError("time_step debe ser ≥ 0")

        rho0 = self._validate_density_matrix(rho_initial, "rho_initial")
        p_vec = self._validate_momentum_vector(momentum_vector)
        g_met = self._validate_metric_tensor(metric_tensor, p_vec.shape[0])

        rho1, tp_res = self._apply_semigroup(rho0, float(gamma_annihilation), dt)
        rho1 = self._validate_density_matrix(rho1, "rho_final")

        eig0, _, spec0 = self._certified_eigh(rho0, "rho_initial")
        eig1, _, spec1 = self._certified_eigh(rho1, "rho_final")

        entropy = self._compute_von_neumann_entropy(eig1)
        n0 = self._particle_number(rho0)
        n1 = self._particle_number(rho1)
        exergy = self._occupation_exergy(n0, n1)

        tensor_t, div_res = self.compute_energy_momentum_tensor(p_vec, g_met)
        signature = self._metric_signature(g_met)
        econd_ok, _ = self._energy_conditions(tensor_t, g_met, signature)
        weyl = self._weyl_invariant_trace(tensor_t, g_met)
        v0 = self._vacuum_fidelity(rho0)
        v1 = self._vacuum_fidelity(rho1)

        packet = PhaseOneEnginePacket(
            rho_initial=_freeze(rho0),
            rho_final=_freeze(rho1),
            eigenvalues_initial=_freeze(eig0),
            eigenvalues_final=_freeze(eig1),
            fock_entropy=float(entropy),
            renyi_entropy_2=float(self._compute_renyi_entropy(eig1, 2.0)),
            min_entropy=float(self._compute_min_entropy(eig1)),
            exergy_efficiency=float(exergy),
            purity_initial=self._compute_purity_from_eigenvalues(eig0),
            purity_final=self._compute_purity_from_eigenvalues(eig1),
            particle_number_initial=float(n0),
            particle_number_final=float(n1),
            particle_number_monotone=bool((n1 - n0) <= _PARTICLE_MONOTONE_TOL),
            vacuum_fidelity_initial=float(v0),
            vacuum_fidelity_final=float(v1),
            vacuum_fidelity_monotone=bool((v0 - v1) <= _VACUUM_MONOTONE_TOL),
            energy_momentum_tensor=_freeze(tensor_t),
            momentum_divergence_residual=float(div_res),
            tensor_symmetry_residual=float(
                np.linalg.norm(tensor_t - tensor_t.T, ord="fro")
            ),
            weyl_invariant_trace=float(weyl),
            trace_energy_momentum=float(np.real(np.trace(tensor_t))),
            energy_condition_ok=bool(econd_ok),
            car_residual=float(self._car_residual),
            spectral_residual_initial=float(spec0),
            spectral_residual_final=float(spec1),
            liouvillian_tp_residual=float(tp_res),
            metric_signature=tuple(int(x) for x in signature),  # type: ignore[arg-type]
            gamma_annihilation=float(gamma_annihilation),
            time_step=float(dt),
        )
        logger.debug(
            "Fase 1: S=%.6f  η=%.4f  ‖T‖_alg=%.3e  N: %.4f→%.4f  CAR=%.3e",
            packet.fock_entropy,
            packet.exergy_efficiency,
            packet.momentum_divergence_residual,
            packet.particle_number_initial,
            packet.particle_number_final,
            packet.car_residual,
        )
        # ---- morfismo terminal η: cierra Fase 1 y abre formalmente Fase 2 ----
        return self._phase1_terminal_morphism(packet)

    def _phase1_terminal_morphism(
        self,
        packet: PhaseOneEnginePacket,
    ) -> PhaseOneEnginePacket:
        r"""
        [FASE 1 · morfismo terminal η]  ≡  [FASE 2 · objeto inicial]

        Unidad idempotente Spec → Spec.  Todo método de la Fase 2
        **comienza** reaplicando η: recertifica invariantes y no muta.
        """
        if packet.rho_initial.shape != (self._dim_fock, self._dim_fock):
            raise ValueError("η: dim(ρ₀) corrupta")
        if packet.rho_final.shape != (self._dim_fock, self._dim_fock):
            raise ValueError("η: dim(ρ₁) corrupta")
        if packet.eigenvalues_initial.size != self._dim_fock:
            raise ValueError("η: espectro inicial de longitud distinta de 2ⁿ")
        if packet.fock_entropy < -_TRACE_ATOL:
            raise ValueError("η: entropía negativa (inconsistencia numérica)")
        if packet.purity_final < -_TRACE_ATOL or packet.purity_final > 1.0 + _TRACE_ATOL:
            raise ValueError("η: pureza fuera de [0, 1]")
        if not np.isfinite(
            [
                packet.fock_entropy,
                packet.exergy_efficiency,
                packet.momentum_divergence_residual,
                packet.car_residual,
                packet.liouvillian_tp_residual,
            ]
        ).all():
            raise ValueError("η: observables no finitos")
        return packet

    # =========================================================================
    # FASE 2 — DECIDE  (continúa η; produce χ ∈ Ω₃)
    #   2.0  η(packet)                         ← último morfismo de Fase 1
    #   2.1  predicados atómicos
    #   2.2  operaciones de Heyting (∧, ∨, →, ¬)
    #   2.3  clasificador de subobjetos χ
    # =========================================================================
    def _classify_heyting_atoms(
        self,
        packet: PhaseOneEnginePacket,
    ) -> Dict[str, bool]:
        """
        Críticos → join VETOED.  Marginales → join DEGRADED.
        Ky Fan / majorización NO es crítica: el canal no es unital.
        """
        return {
            "entropy_violation": packet.fock_entropy > self._entropy_threshold,
            "divergence_violation": (
                packet.momentum_divergence_residual > self._divergence_threshold
            ),
            "particle_violation": not packet.particle_number_monotone,
            "vacuum_violation": not packet.vacuum_fidelity_monotone,
            "car_violation": packet.car_residual > _CAR_TOLERANCE,
            "energy_condition_violation": not packet.energy_condition_ok,
            "tp_violation": packet.liouvillian_tp_residual > _TP_RESIDUAL_TOL,
            "spectral_violation": (
                packet.spectral_residual_final > _SPECTRAL_RESIDUAL_TOL
                or packet.spectral_residual_initial > _SPECTRAL_RESIDUAL_TOL
            ),
            "entropy_marginal": packet.fock_entropy > 0.1 * self._entropy_threshold,
            "divergence_marginal": (
                packet.momentum_divergence_residual
                > 0.01 * self._divergence_threshold
            ),
            "weyl_marginal": abs(packet.weyl_invariant_trace)
            > _ENERGY_COND_TOL
            * max(1.0, float(np.linalg.norm(packet.energy_momentum_tensor))),
        }

    def _phase2_heyting_decision(
        self,
        packet: PhaseOneEnginePacket,
    ) -> PhaseTwoDecisionPacket:
        """
        [FASE 2 · Decide]

        Continuación formal de `_phase1_terminal_morphism`: re-certifica η
        y clasifica en Ω₃ = {COHERENT < DEGRADED < VETOED}.
        χ = ⊤  ⇔  veredicto = VETOED.
        """
        certified = self._phase1_terminal_morphism(packet)  # ← nido Fase 1 → 2
        atoms = self._classify_heyting_atoms(certified)

        critical = (
            atoms["entropy_violation"],
            atoms["divergence_violation"],
            atoms["particle_violation"],
            atoms["vacuum_violation"],
            atoms["car_violation"],
            atoms["energy_condition_violation"],
            atoms["tp_violation"],
            atoms["spectral_violation"],
        )
        marginal = (
            atoms["entropy_marginal"],
            atoms["divergence_marginal"],
            atoms["weyl_marginal"],
        )
        verdict = HeytingOmega3.classify(critical, marginal)
        chi_top = verdict is HeytingVerdict.VETOED
        impl = HeytingOmega3.implies(HeytingVerdict.COHERENT, verdict)

        decision = PhaseTwoDecisionPacket(
            heyting_verdict=verdict.value,
            entropy_violation=atoms["entropy_violation"],
            divergence_violation=atoms["divergence_violation"],
            particle_violation=atoms["particle_violation"],
            vacuum_violation=atoms["vacuum_violation"],
            car_violation=atoms["car_violation"],
            energy_condition_violation=atoms["energy_condition_violation"],
            tp_violation=atoms["tp_violation"],
            subobject_classifier=bool(chi_top),
            implication_coherent_to_verdict=impl.value,
            join_of_atoms=verdict.value,
        )
        logger.debug(
            "Fase 2: Ω₃=%s  χ=%s  (S↑=%s  ∇T=%s  N=%s  CAR=%s)",
            decision.heyting_verdict,
            "⊤" if chi_top else "¬⊤",
            decision.entropy_violation,
            decision.divergence_violation,
            decision.particle_violation,
            decision.car_violation,
        )
        return decision

    # =========================================================================
    # FASE 3 — ACT  (continúa χ; ISR simulada + sello)
    #   3.0  consume PhaseTwoDecisionPacket
    #   3.1  interlock crowbar *simulado*
    #   3.2  digest SHA-256 y sello ABI-compatible
    # =========================================================================
    def _act_hardware_interlock_simulation(self, verdict: str) -> Tuple[bool, float]:
        """
        [FASE 3 · ISR simulada]

        Modelo N(400, 5) ns recortado a [380, 420].
        No toca GPIO, ESP32 ni el tiristor BT151 reales.
        """
        if verdict == HeytingVerdict.VETOED.value:
            jitter = float(self._rng.normal(0.0, 5.0))
            latency = float(np.clip(_CROWBAR_IRAM_LATENCY_NS + jitter, 380.0, 420.0))
            logger.critical(
                "VETO Ω₃=⊤: interlock *simulado* (GPIO14/BT151 modelo) en %.2f ns.",
                latency,
            )
            return True, latency
        return False, 0.0

    def _round_obs(self, x: float) -> float:
        if not np.isfinite(x):
            raise ValueError("observable no finito en el digest")
        return float(round(float(x), _DIGEST_ROUND_DECIMALS))

    def _forensic_digest(self, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, allow_nan=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _phase3_act_and_telemetry(
        self,
        decision: PhaseTwoDecisionPacket,
        packet1: PhaseOneEnginePacket,
    ) -> TelemetryStamp:
        """
        [FASE 3 · Act]

        Recibe χ (Fase 2) y η(packet) (Fase 1), actúa el interlock
        simulado si χ=⊤ y emite el sello con digest SHA-256.
        """
        certified = self._phase1_terminal_morphism(packet1)
        interlock_fired, latency = self._act_hardware_interlock_simulation(
            decision.heyting_verdict
        )

        if decision.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "Fase 3: VETO. S=%.6f  r_T=%.3e  N_mono=%s  vac_mono=%s  CAR=%.3e",
                certified.fock_entropy,
                certified.momentum_divergence_residual,
                certified.particle_number_monotone,
                certified.vacuum_fidelity_monotone,
                certified.car_residual,
            )

        digest_payload = {
            "verdict": decision.heyting_verdict,
            "S": self._round_obs(certified.fock_entropy),
            "exergy": self._round_obs(certified.exergy_efficiency),
            "TrT_coord": self._round_obs(certified.trace_energy_momentum),
            "weyl": self._round_obs(certified.weyl_invariant_trace),
            "div": self._round_obs(certified.momentum_divergence_residual),
            "purity": self._round_obs(certified.purity_final),
            "N": self._round_obs(certified.particle_number_final),
            "car": self._round_obs(certified.car_residual),
            "chi": bool(decision.subobject_classifier),
            "interlock": bool(interlock_fired),
        }

        stamp = TelemetryStamp(
            heyting_verdict=decision.heyting_verdict,
            fock_entropy=certified.fock_entropy,
            exergy_efficiency=certified.exergy_efficiency,
            energy_momentum_trace=certified.trace_energy_momentum,
            momentum_divergence_residual=certified.momentum_divergence_residual,
            quantum_state_purity=certified.purity_final,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            weyl_invariant_trace=certified.weyl_invariant_trace,
            particle_number_final=certified.particle_number_final,
            car_residual=certified.car_residual,
            forensic_digest=self._forensic_digest(digest_payload),
        )
        logger.debug(
            "Fase 3: interlock=%s  latencia=%.2f ns  digest=%s",
            "ON" if interlock_fired else "OFF",
            latency,
            stamp.forensic_digest[:16],
        )
        return stamp

    # =========================================================================
    # ORQUESTACIÓN OODA  (composición η ; χ ; Act)
    # =========================================================================
    def execute_forensic_cycle(
        self,
        rho_initial: np.ndarray,
        momentum_vector: np.ndarray,
        metric_tensor: np.ndarray,
        gamma_annihilation: float,
    ) -> Dict[str, Any]:
        """
        Compone las tres fases anidadas:

            sello = Act( χ( η( Observe(ρ₀, p, g, γ) ) ) )

        Cualquier excepción no contemplada colapsa a VETOED preventivo
        (fail-closed) y devuelve el dict ABI v2.
        """
        try:
            packet1 = self._phase1_observe_orient(
                rho_initial, momentum_vector, metric_tensor, gamma_annihilation
            )
            decision = self._phase2_heyting_decision(packet1)
            stamp = self._phase3_act_and_telemetry(decision, packet1)
            return stamp.to_dict()
        except Exception:
            logger.exception("Excepción en el ciclo OODA: VETO preventivo (fail-closed).")
            return TelemetryStamp(
                heyting_verdict=HeytingVerdict.VETOED.value,
                fock_entropy=float("nan"),
                exergy_efficiency=float("nan"),
                energy_momentum_trace=float("nan"),
                momentum_divergence_residual=float("nan"),
                quantum_state_purity=float("nan"),
                hardware_interlock_fired=True,
                actuation_latency_ns=_CROWBAR_IRAM_LATENCY_NS,
                forensic_digest="emergency_veto",
            ).to_dict()


# -----------------------------------------------------------------------------
# Exportación de firmas de calibre
# -----------------------------------------------------------------------------
__all__ = [
    "FockForensicHall",
    "PhaseOneEnginePacket",
    "PhaseTwoDecisionPacket",
    "TelemetryStamp",
    "HeytingVerdict",
    "HeytingOmega3",
]
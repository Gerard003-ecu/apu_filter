# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Fock Forensic Hall Agent (Soberano de Calibre de la Gala Forense)   ║
║ Ruta   : app/agents/core/inmune_system/fock_forensic_hall_agent.py           ║
║ Versión: 3.0.0-Doctoral-Heyting-OODA-GKSL-Weyl-KyFan-Banach-Secure           ║
║                                                                              ║
║ ARQUITECTURA DE FASES ANIDADAS (morfismos, no meras secciones):              ║
║                                                                              ║
║   Fase 1  --η-->  Fase 2  --χ-->  Fase 3                                     ║
║   Observe+Orient    Decide (Ω₃)    Act + certificado                         ║
║                                                                              ║
║   η  = morfismo terminal de Fase 1 = objeto inicial de Fase 2                ║
║   χ  = clasificador de subobjetos de Heyting = objeto inicial de Fase 3      ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA (rigor, sin metáfora suelta):                            ║
║   • Estados: operadores densidad en B(ℋ), ℋ = ℱ_-(ℂ^{n}) ≅ ℂ^{2ⁿ}.         ║
║   • Dinámica: semigrupo GKSL (Lindblad) de aniquilación fermiónica.          ║
║   • Espectro: teorema de Weyl / Wilkinson; residuos ‖Av−λv‖₂.                ║
║   • Entropías: von Neumann, Rényi-2, min-entropía.                           ║
║   • Majorización HLP/Ky Fan: diagnóstica (el canal NO es unital).            ║
║   • Invariantes físicos del canal de aniquilación:                           ║
║         ⟨N⟩(t) no creciente,  F(ρ(t),|0⟩⟨0|) no decreciente.                 ║
║   • Tensor T^{μν}: simetría, condiciones de energía, anomalía de Weyl Tr(T). ║
║     La divergencia covariante puntual exige Γ[g] y ∂g; sin ellos se reporta  ║
║     el residuo algebraico y, si el motor lo provee, su div_residual.         ║
║   • Ω₃ = {COHERENT < DEGRADED < VETOED} cadena de Heyting.                   ║
║   • Fase 3: ISR *simulada* (sin GPIO real). Crowbar BT151 es un modelo.      ║
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
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from scipy.special import xlogy

# =============================================================================
# Motor externo (ciego) — se tipifica; si falta, se usa el motor interno GKSL.
# =============================================================================
try:
    from fock_forensic_hall import FockForensicHall as _ExternalFockForensicHall
except ImportError:  # pragma: no cover - entorno sin el motor de FPU
    _ExternalFockForensicHall = None  # type: ignore[misc, assignment]


logger = logging.getLogger("APU.Agents.FockForensicHallAgent")

# =============================================================================
# Constantes espectrales (IEEE-754 float64, Wilkinson–Higham–Kahan)
# =============================================================================
_EPS64: Final[float] = float(np.finfo(np.float64).eps)          # ≈ 2.22e-16
_WILKINSON_LIMIT: Final[float] = 50.0 * _EPS64                  # ≈ 1.11e-14
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_HARD_ENTROPY_CEILING: Final[float] = 0.5
_HARD_DIVERGENCE_CEILING: Final[float] = 1e-4

_HERMITIAN_RTOL: Final[float] = 1e-12
_PSD_EIGEN_FLOOR: Final[float] = -1e-12
_TRACE_ATOL: Final[float] = 1e-10
_KY_FAN_TOLERANCE: Final[float] = 1e-12
_SPECTRAL_RESIDUAL_TOL: Final[float] = 1e-10
_ENERGY_COND_TOL: Final[float] = 1e-10
_PARTICLE_MONOTONE_TOL: Final[float] = 1e-10
_VACUUM_MONOTONE_TOL: Final[float] = 1e-10
_KMS_RESIDUAL_TOL: Final[float] = 1e-8
_TOEPLITZ_TOL: Final[float] = 1e-10

_DEFAULT_LINDBLAD_DT: Final[float] = 1.0e-2
_RNG_SEED: Final[int] = 42
_DIGEST_ROUND_DECIMALS: Final[int] = 12

_PAULI_I: Final[np.ndarray] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
_PAULI_Z: Final[np.ndarray] = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
# c |1⟩ = |0⟩  ⇒  c = |0⟩⟨1|
_FERMION_C: Final[np.ndarray] = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)


# =============================================================================
# Tipos ordinales del clasificador
# =============================================================================
class HeytingVerdict(str, Enum):
    """Cadena de Heyting Ω₃: COHERENT ⊥ < DEGRADED < VETOED ⊤."""

    COHERENT = "COHERENT"
    DEGRADED = "DEGRADED"
    VETOED = "VETOED"

    @property
    def rank(self) -> int:
        return {self.COHERENT: 0, self.DEGRADED: 1, self.VETOED: 2}[self]


# =============================================================================
# Protocolo del motor de Fock (contrato mínimo)
# =============================================================================
@runtime_checkable
class FockEngineProtocol(Protocol):
    """Contrato del motor ciego: semigrupo de aniquilación + tensor T."""

    def solve_lindblad_annihilation(
        self,
        rho_initial: np.ndarray,
        gamma_annihilation: float,
        time_step: float,
    ) -> Tuple[np.ndarray, float, float]:
        ...

    def compute_energy_momentum_tensor(
        self,
        momentum_vector: np.ndarray,
        metric_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        ...


# =============================================================================
# Utilidades numéricas de Banach / C* / IEEE-754
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


def _scaled_tol(norm: float, rtol: float, atol: float) -> float:
    """Tolerancia de Higham: max(atol, rtol · ‖A‖)."""
    return float(max(atol, rtol * max(norm, 1.0)))


def _hermite(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def _operator_norm_hermitian(eigvals: np.ndarray) -> float:
    """‖A‖_{B(ℋ)} = ρ(A) = max |λ| para A = A†."""
    return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def _frobenius_norm_from_eig(eigvals: np.ndarray) -> float:
    """‖A‖₂ (Frobenius) = √(Σ λᵢ²) para A = A†."""
    return float(np.sqrt(np.real(np.sum(np.square(eigvals)))))


def _safe_xlogx(x: np.ndarray) -> np.ndarray:
    """x log x con convención 0 log 0 = 0 (scipy.special.xlogy)."""
    xr = np.real(np.asarray(x, dtype=np.float64))
    return xlogy(xr, xr)


# =============================================================================
# Álgebra de Heyting finita Ω₃ (cualquier cadena es un álgebra de Heyting)
# =============================================================================
class HeytingOmega3:
    r"""
    Ω₃ = {0 = COHERENT, ½ = DEGRADED, 1 = VETOED} con el orden de cadena.

    Operaciones (Heyting):
        a ∧ b = min(a, b)
        a ∨ b = max(a, b)
        a → b = ⊤  si a ≤ b,  else b
        ¬a    = a → ⊥

    El clasificador de subobjetos χ_S : 1 → Ω₃ envía el estado al
    supremo de los predicados de violación que se activan.
    """

    _ORDER: Final[Tuple[HeytingVerdict, ...]] = (
        HeytingVerdict.COHERENT,
        HeytingVerdict.DEGRADED,
        HeytingVerdict.VETOED,
    )

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
        """
        χ: si algún predicado crítico es ⊤ ⇒ VETOED;
           si no, si algún marginal es ⊤ ⇒ DEGRADED;
           si no ⇒ COHERENT.
        Equivale al join de los valores de verdad de cada predicado.
        """
        atoms: List[HeytingVerdict] = []
        for flag in critical:
            atoms.append(HeytingVerdict.VETOED if flag else HeytingVerdict.COHERENT)
        for flag in marginal:
            atoms.append(HeytingVerdict.DEGRADED if flag else HeytingVerdict.COHERENT)
        return cls.join_all(atoms)


# =============================================================================
# Paquetes inmutables (objetos de las fases)
# =============================================================================
@dataclass(frozen=True, slots=True)
class PhaseOneSpectralPacket:
    """
    Objeto terminal de la Fase 1 / objeto inicial de la Fase 2.

    Contiene el espectro certificado, las normas de Banach, los invariantes
    de canal (N, vacío), la majorización diagnóstica y el tensor T^{μν}.
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
    banach_op_norm: float
    frobenius_norm: float
    trace_distance: float
    fidelity_uhlmann: float
    majorization_monotonic: bool
    ky_fan_residual: float
    hlp_equality_residual: float
    particle_number_initial: float
    particle_number_final: float
    particle_number_monotone: bool
    vacuum_fidelity_initial: float
    vacuum_fidelity_final: float
    vacuum_fidelity_monotone: bool
    energy_momentum_tensor: np.ndarray
    momentum_divergence_residual: float
    tensor_symmetry_residual: float
    weyl_trace_anomaly: float
    energy_condition_ok: bool
    wilkinson_violation: bool
    spectral_residual_initial: float
    spectral_residual_final: float
    kms_gibbs_residual: float
    fock_betti_0: int
    toeplitz_residual: float
    metric_signature: Tuple[int, int, int]

    def __post_init__(self) -> None:
        # Invariantes algebraicos mínimos del paquete (falla rápida).
        if self.rho_initial.ndim != 2 or self.rho_final.ndim != 2:
            raise ValueError("rho_initial y rho_final deben ser matrices")
        if self.eigenvalues_initial.shape != self.eigenvalues_final.shape:
            raise ValueError("los espectros inicial y final tienen dimensión distinta")


@dataclass(frozen=True, slots=True)
class PhaseTwoHeytingPacket:
    """
    Objeto terminal de la Fase 2 / objeto inicial de la Fase 3.

    Transporta el veredicto Ω₃, los predicados atómicos y el valor del
    clasificador de subobjetos χ (True ⇔ χ = ⊤ = VETOED).
    """

    heyting_verdict: str
    entropy_violation: bool
    divergence_violation: bool
    majorization_violation: bool
    particle_violation: bool
    vacuum_violation: bool
    energy_condition_violation: bool
    kms_violation: bool
    spectral_certificate_violation: bool
    subobject_classifier: bool
    implication_coherent_to_verdict: str
    join_of_atoms: str


@dataclass(frozen=True, slots=True)
class FockForensicCertificate:
    """Certificado inmutable de la Gala Forense (Fase 3)."""

    phase: str
    heyting_verdict: str
    fock_entropy: float
    renyi_entropy_2: float
    min_entropy: float
    exergy_efficiency: float
    energy_momentum_trace: float
    weyl_trace_anomaly: float
    momentum_divergence_residual: float
    quantum_state_purity: float
    fidelity_uhlmann: float
    particle_number_final: float
    majorization_monotonic: bool
    ky_fan_residual: float
    kms_gibbs_residual: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    forensic_digest: str = field(default="")


# =============================================================================
# Motor interno GKSL + tensor de polvo (fallback y verificación independiente)
# =============================================================================
class _InternalFockEngine:
    r"""
    Motor autónomo de dimensión 2ⁿ.

    • Aniquilación: Lⱼ = √γ cⱼ, cⱼ = Jordan–Wigner(c).
    • Semigrupo: vec(ρ(t)) = exp(t ℒ) vec(ρ₀), ℒ en forma GKSL (H = 0).
    • T^{μν} = p^μ p^ν  (tensor de polvo, escala 1), p^μ = g^{μν} p_ν.
      Residuo de divergencia algebraico = ‖T − Tᵀ‖_F + indicador de
      no-finitud; sin ∂g no hay ∇_ν T^{μν} puntual honesto.
    """

    def __init__(self, num_modes: int) -> None:
        if num_modes < 1:
            raise ValueError("num_modes debe ser ≥ 1")
        self._n = int(num_modes)
        self._dim = 1 << self._n
        self._annihilators = self._jordan_wigner_annihilators()
        self._number_op = self._build_number_operator()

    def _kron_n(self, ops: Sequence[np.ndarray]) -> np.ndarray:
        acc = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        for op in ops:
            acc = np.kron(acc, op)
        return acc

    def _jordan_wigner_annihilators(self) -> List[np.ndarray]:
        ops: List[np.ndarray] = []
        for j in range(self._n):
            factors: List[np.ndarray] = []
            for k in range(self._n):
                if k < j:
                    factors.append(_PAULI_Z)
                elif k == j:
                    factors.append(_FERMION_C)
                else:
                    factors.append(_PAULI_I)
            ops.append(self._kron_n(factors))
        return ops

    def _build_number_operator(self) -> np.ndarray:
        n_op = np.zeros((self._dim, self._dim), dtype=np.complex128)
        for c in self._annihilators:
            n_op += c.conj().T @ c
        return _hermite(n_op)

    @property
    def number_operator(self) -> np.ndarray:
        return self._number_op

    def _gksl_superoperator(self, gamma: float) -> np.ndarray:
        d = self._dim
        ident = np.eye(d, dtype=np.complex128)
        ell = np.zeros((d * d, d * d), dtype=np.complex128)
        scale = np.sqrt(max(float(gamma), 0.0))
        for c in self._annihilators:
            lj = scale * c
            ld = lj.conj().T
            # vec(L ρ L†) = (conj(L) ⊗ L) vec(ρ) = ((L†)ᵀ ⊗ L) vec(ρ)
            ell += np.kron(np.conj(lj), lj)
            ldl = ld @ lj
            # −½ vec({L†L, ρ}) = −½ (I ⊗ A + Aᵀ ⊗ I) vec(ρ)
            ell -= 0.5 * (np.kron(ident, ldl) + np.kron(ldl.T, ident))
        return ell

    def solve_lindblad_annihilation(
        self,
        rho_initial: np.ndarray,
        gamma_annihilation: float,
        time_step: float,
    ) -> Tuple[np.ndarray, float, float]:
        if gamma_annihilation < 0.0:
            raise ValueError("gamma_annihilation debe ser ≥ 0 (disipador GKSL)")
        if time_step < 0.0:
            raise ValueError("time_step debe ser ≥ 0")
        rho = _as_c128(rho_initial, "rho_initial")
        if rho.shape != (self._dim, self._dim):
            raise ValueError(
                f"rho_initial tiene forma {rho.shape}, se esperaba "
                f"({self._dim}, {self._dim})"
            )
        super_op = self._gksl_superoperator(gamma_annihilation)
        prop = la.expm(super_op * float(time_step))
        vec_final = prop @ rho.reshape(-1, order="F")
        rho_final = vec_final.reshape((self._dim, self._dim), order="F")
        rho_final = _hermite(rho_final)
        # Proyección de traza (el semigrupo GKSL es TP; esto cancela redondeo).
        tr = np.trace(rho_final)
        if abs(tr) < _WILKINSON_LIMIT:
            raise ValueError("traza de rho_final numéricamente nula")
        rho_final = rho_final / tr

        eig_f = np.clip(np.real(la.eigvalsh(rho_final)), 0.0, None)
        eig_f = eig_f / max(float(np.sum(eig_f)), _WILKINSON_LIMIT)
        entropy = float(-np.sum(_safe_xlogx(eig_f[eig_f > _WILKINSON_LIMIT])))

        n_i = float(np.real(np.trace(self._number_op @ rho)))
        n_f = float(np.real(np.trace(self._number_op @ rho_final)))
        # Exergía: fracción de número de ocupación aniquilada (en [0, 1]).
        exergy = 0.0 if n_i <= _WILKINSON_LIMIT else float(
            np.clip((n_i - n_f) / n_i, 0.0, 1.0)
        )
        return rho_final, entropy, exergy

    def compute_energy_momentum_tensor(
        self,
        momentum_vector: np.ndarray,
        metric_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        p_down = np.real(_as_f64(np.asarray(momentum_vector, dtype=np.float64), "p"))
        g = np.real(_as_f64(np.asarray(metric_tensor, dtype=np.float64), "g"))
        if p_down.ndim != 1:
            raise ValueError("momentum_vector debe ser 1-forma (N,)")
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError("metric_tensor debe ser cuadrado")
        if g.shape[0] != p_down.shape[0]:
            raise ValueError("dimensión de p_μ y G_{μν} incompatible")
        g_h = 0.5 * (g + g.T)
        try:
            g_inv = la.inv(g_h)
        except la.LinAlgError as exc:
            raise ValueError("metric_tensor es singular") from exc
        p_up = g_inv @ p_down
        tensor_t = np.outer(p_up, p_up)
        # Residuo algebraico: antisimetría + inconsistencia p_μ − g_{μν} p^ν.
        sym_res = float(np.linalg.norm(tensor_t - tensor_t.T, ord="fro"))
        lower_res = float(np.linalg.norm(p_down - g_h @ p_up))
        div_residual = sym_res + lower_res
        return tensor_t, div_residual


# =============================================================================
# Agente soberano — tres fases anidadas
# =============================================================================
class FockForensicHallAgent:
    r"""
    Supervisor de lazo cerrado sobre el motor Fock.

    Fases anidadas (composición estricta de morfismos):

        phase1_spectral_orientation  →  PhaseOneSpectralPacket
                └─ _phase1_terminal_morphism  = η   ┐
                                                    │  (identidad de adjunción)
        phase2_heyting_decision      ←  η(packet)   ┘
                └─ PhaseTwoHeytingPacket = χ
        phase3_actuation_and_certificate ← χ
    """

    def __init__(
        self,
        num_modes: int = 2,
        entropy_threshold: float = _HARD_ENTROPY_CEILING,
        divergence_threshold: float = _HARD_DIVERGENCE_CEILING,
        rng_seed: Optional[int] = _RNG_SEED,
        lindblad_time_step: float = _DEFAULT_LINDBLAD_DT,
        engine: Optional[FockEngineProtocol] = None,
    ) -> None:
        if num_modes < 1:
            raise ValueError("num_modes debe ser ≥ 1")
        if lindblad_time_step < 0.0:
            raise ValueError("lindblad_time_step debe ser ≥ 0")

        self._modes: Final[int] = int(num_modes)
        self._dim: Final[int] = 1 << self._modes
        self._entropy_thresh: Final[float] = max(float(entropy_threshold), _WILKINSON_LIMIT)
        self._div_thresh: Final[float] = max(float(divergence_threshold), _WILKINSON_LIMIT)
        self._dt: Final[float] = float(lindblad_time_step)

        self._internal: Final[_InternalFockEngine] = _InternalFockEngine(self._modes)
        if engine is not None:
            self._engine: FockEngineProtocol = engine
        elif _ExternalFockForensicHall is not None:
            self._engine = _ExternalFockForensicHall(num_modes=self._modes)  # type: ignore[call-arg]
        else:
            self._engine = self._internal
            logger.info(
                "Motor externo ausente: se usa _InternalFockEngine GKSL (n=%d, dim=%d).",
                self._modes,
                self._dim,
            )

        self._rng = np.random.default_rng(rng_seed)

    # =========================================================================
    # FASE 1 — OBSERVE + ORIENT
    #   1.1 validación C* del estado
    #   1.2 espectro certificado (Weyl–Wilkinson)
    #   1.3 entropías y normas de Banach
    #   1.4 majorización HLP / normas de Ky Fan (diagnóstica)
    #   1.5 invariantes de canal (N, vacío) y grafo de Fock
    #   1.6 semigrupo GKSL
    #   1.7 tensor T^{μν}, Weyl, condiciones de energía
    #   1.8 morfismo terminal η  →  inicio formal de la Fase 2
    # =========================================================================
    def _validate_density_matrix(self, rho: np.ndarray, name: str = "rho") -> np.ndarray:
        """
        Condiciones de estado cuántico en M_d(ℂ):
          cuadrada, finita, hermítica, PSD, traza 1, dim = 2ⁿ.
        Devuelve una copia C128 (no congela: el motor puede necesitar escritura).
        """
        rho_c = _as_c128(rho, name)
        if rho_c.ndim != 2 or rho_c.shape[0] != rho_c.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada; se recibió {rho_c.shape}")
        n = rho_c.shape[0]
        if n != self._dim:
            raise ValueError(
                f"dim({name})={n} ≠ 2^{self._modes}={self._dim} (base de Fock fermiónica)"
            )
        herm_res = float(np.linalg.norm(rho_c - rho_c.conj().T, ord="fro"))
        herm_tol = _scaled_tol(float(np.linalg.norm(rho_c, ord="fro")), _HERMITIAN_RTOL, _WILKINSON_LIMIT)
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

    def _certified_eigh(self, herm: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Descomposición espectral A = U Λ U† con residuo de Wilkinson
            r = ‖A U − U Λ‖_F / max(1, ‖A‖_F).
        Weyl: |λᵢ(A+E) − λᵢ(A)| ≤ ‖E‖₂.
        """
        eigvals, eigvecs = la.eigh(herm)
        recon = herm @ eigvecs - eigvecs @ np.diag(eigvals)
        denom = max(float(np.linalg.norm(herm, ord="fro")), 1.0)
        residual = float(np.linalg.norm(recon, ord="fro") / denom)
        if residual > _SPECTRAL_RESIDUAL_TOL:
            logger.warning(
                "Residuo espectral elevado en %s: r=%.3e (Weyl/Wilkinson).",
                name,
                residual,
            )
        return np.real(eigvals), eigvecs, residual

    def _compute_von_neumann_entropy(self, eigenvalues: np.ndarray) -> float:
        """S(ρ) = −Tr(ρ log ρ) = −Σ λ log λ,  0 log 0 := 0.  Unidades: nats."""
        lam = np.real(np.asarray(eigenvalues, dtype=np.float64))
        lam = lam[lam > _WILKINSON_LIMIT]
        if lam.size == 0:
            return 0.0
        return float(-np.sum(_safe_xlogx(lam)))

    def _compute_renyi_entropy(self, eigenvalues: np.ndarray, alpha: float) -> float:
        r"""
        S_α(ρ) = (1−α)^{−1} log Tr(ρ^α), α>0, α≠1.
        α=2 ⇒ −log Tr(ρ²) = −log(pureza).
        """
        if alpha <= 0.0 or abs(alpha - 1.0) < _WILKINSON_LIMIT:
            raise ValueError("Rényi exige α>0, α≠1")
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), 0.0, None)
        power = float(np.sum(np.power(lam, alpha)))
        if power <= _WILKINSON_LIMIT:
            return 0.0
        return float(np.log(power) / (1.0 - alpha))

    def _compute_min_entropy(self, eigenvalues: np.ndarray) -> float:
        """S_∞(ρ) = −log ‖ρ‖_∞ = −log λ_max."""
        lam_max = float(np.max(np.real(eigenvalues))) if eigenvalues.size else 0.0
        if lam_max <= _WILKINSON_LIMIT:
            return 0.0
        return float(-np.log(lam_max))

    def _compute_purity_from_eigenvalues(self, eigenvalues: np.ndarray) -> float:
        return float(np.real(np.sum(np.square(eigenvalues))))

    def _trace_distance(self, eig_diff_abs_sum: float) -> float:
        """
        T(ρ,σ) = (1/2)‖ρ−σ‖₁.  Aquí se pasa ‖ρ−σ‖₁ ya calculado
        (suma de valores singulares = suma |autovalores| si ρ−σ es hermítico).
        """
        return 0.5 * float(eig_diff_abs_sum)

    def _uhlmann_fidelity(self, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""
        F(ρ,σ) = (Tr √(√ρ σ √ρ))²  (Uhlmann).
        Para ρ,σ PSD hermíticos se usa el algoritmo de Nielsen–Chuang
        vía productor de Schatten: ‖√ρ √σ‖₁².
        """
        w_r, u_r = la.eigh(_hermite(rho))
        w_r = np.clip(np.real(w_r), 0.0, None)
        sqrt_rho = (u_r * np.sqrt(w_r)) @ u_r.conj().T
        inner = sqrt_rho @ _hermite(sigma) @ sqrt_rho
        w_i = np.clip(np.real(la.eigvalsh(_hermite(inner))), 0.0, None)
        return float(np.square(np.sum(np.sqrt(w_i))))

    def _verify_ky_fan_majorization(
        self,
        eigenvalues_initial: np.ndarray,
        eigenvalues_final: np.ndarray,
    ) -> Tuple[bool, float, float]:
        r"""
        Submajorización de Ky Fan / HLP (diagnóstica, no ley del canal):

            Σ_{i=1}^k λ↓_final(i)  ≤  Σ_{i=1}^k λ↓_initial(i) + ε,  k=1..n
            y  |Σ λ_final − Σ λ_initial| ≤ ε   (igualdad de traza).

        Devuelve (monótona, residuo_KyFan, residuo_traza).
        Para canales no unitales (aniquilación) la monotonía PUEDE fallar
        sin que la dinámica sea antifísica.
        """
        eig_i = np.sort(np.real(eigenvalues_initial))[::-1]
        eig_f = np.sort(np.real(eigenvalues_final))[::-1]
        if eig_i.shape != eig_f.shape:
            raise ValueError("espectros de longitudes distintas en Ky Fan")
        diff = np.cumsum(eig_f) - np.cumsum(eig_i)
        max_violation = float(np.max(diff)) if diff.size else 0.0
        residual = max(0.0, max_violation)
        hlp_eq = abs(float(np.sum(eig_f) - np.sum(eig_i)))
        monotonic = (max_violation <= _KY_FAN_TOLERANCE) and (hlp_eq <= _KY_FAN_TOLERANCE)
        return monotonic, residual, hlp_eq

    def _fock_number_expectation(self, rho: np.ndarray) -> float:
        return float(np.real(np.trace(self._internal.number_operator @ rho)))

    def _vacuum_fidelity(self, rho: np.ndarray) -> float:
        """F(ρ, |0⟩⟨0|) = ⟨0|ρ|0⟩ en la base de ocupación JW (bit 0 = vacío)."""
        return float(np.real(rho[0, 0]))

    def _fock_support_betti0(self, rho: np.ndarray) -> int:
        """
        b₀ del subgrafo inducido en el hipercubo {0,1}ⁿ por el soporte
        de la diagonal de ρ (masa poblacional > Wilkinson).
        b₀ = número de componentes conexas (adyacencia = Hamming 1).
        """
        diag = np.real(np.diag(rho))
        support = np.flatnonzero(diag > _WILKINSON_LIMIT)
        if support.size == 0:
            return 0
        parent = {int(i): int(i) for i in support}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        supp_set = set(parent)
        for i in support:
            ii = int(i)
            for bit in range(self._modes):
                j = ii ^ (1 << bit)
                if j in supp_set:
                    union(ii, j)
        return len({find(int(i)) for i in support})

    def _toeplitz_residual(self, rho: np.ndarray) -> float:
        """
        Defecto de estructura de Toeplitz: max |ρ_{i,j} − ρ_{i+1,j+1}|.
        Cero ⇒ estado estacionario en esa base (símbolo de Szegő constante).
        """
        n = rho.shape[0]
        if n < 2:
            return 0.0
        acc = 0.0
        for k in range(1, n):
            diag = np.diag(rho, k=k)
            acc = max(acc, float(np.max(np.abs(np.diff(diag))))) if diag.size > 1 else acc
            diag_m = np.diag(rho, k=-k)
            acc = max(acc, float(np.max(np.abs(np.diff(diag_m))))) if diag_m.size > 1 else acc
        main = np.diag(rho)
        if main.size > 1:
            acc = max(acc, float(np.max(np.abs(np.diff(main)))))
        return acc

    def _metric_signature(self, metric: np.ndarray) -> Tuple[int, int, int]:
        """Signatura (n₊, n₋, n₀) de G_{μν} vía autovalores (Ley de inercia de Sylvester)."""
        ev = np.real(la.eigvalsh(_hermite(metric)))
        n_plus = int(np.sum(ev > _ENERGY_COND_TOL))
        n_minus = int(np.sum(ev < -_ENERGY_COND_TOL))
        n_zero = int(ev.size - n_plus - n_minus)
        return n_plus, n_minus, n_zero

    def _validate_metric_and_momentum(
        self,
        momentum_vector: np.ndarray,
        metric_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        p = _as_f64(np.asarray(momentum_vector, dtype=np.float64), "momentum_vector")
        g = _as_f64(np.asarray(metric_tensor, dtype=np.float64), "metric_tensor")
        if p.ndim != 1:
            raise ValueError(f"momentum_vector debe ser (N,); se recibió {p.shape}")
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError(f"metric_tensor debe ser cuadrado; se recibió {g.shape}")
        if g.shape[0] != p.shape[0]:
            raise ValueError("p_μ y G_{μν} tienen dimensión incompatible")
        g_h = 0.5 * (g + g.T)
        if abs(float(np.linalg.det(g_h))) < _WILKINSON_LIMIT:
            raise ValueError("metric_tensor es singular (det ≈ 0)")
        return p, g_h

    def _energy_conditions(
        self,
        tensor_t: np.ndarray,
        metric: np.ndarray,
        signature: Tuple[int, int, int],
    ) -> Tuple[bool, float]:
        r"""
        • Riemanniana (n₋=0, n₀=0): T ⪰ 0  (estrés cinético PSD).
        • Lorentziana (min(n₊,n₋)=1, n₀=0): condición débil
              T_{μν} t^μ t^ν ≥ 0  para t timelike de prueba
          aproximada por el criterio espectral de T^μ_ν = g^{μα} T_{αν}
          sobre el cono numérico (autovalores de la mezcla g^{-1}T).
        """
        t_h = 0.5 * (np.real(tensor_t) + np.real(tensor_t).T)
        g_h = 0.5 * (np.real(metric) + np.real(metric).T)
        n_plus, n_minus, n_zero = signature
        if n_zero > 0:
            return False, float("inf")
        if n_minus == 0:
            ev = np.real(la.eigvalsh(t_h))
            viol = float(max(0.0, -np.min(ev)))
            return viol <= _ENERGY_COND_TOL, viol
        # Lorentziana: forma mixta T^μ_ν.
        try:
            mixed = la.solve(g_h, t_h, assume_a="sym")
        except la.LinAlgError:
            return False, float("inf")
        ev = np.real(la.eigvals(mixed))
        # Condición débil relajada: energía no negativa en la dirección
        # de mayor |autovalor| mixto (proxy numérico, no un teorema global).
        viol = float(max(0.0, -np.min(ev)))
        return viol <= _ENERGY_COND_TOL * max(1.0, float(np.max(np.abs(ev)))), viol

    def _kms_gibbs_residual(self, eigenvalues: np.ndarray, energy_levels: np.ndarray) -> float:
        r"""
        Residuo de Gibbs/KMS a un parámetro: si ρ = e^{−βH}/Z entonces
            log λᵢ + β Eᵢ  es constante en el soporte.
        Se estima β por mínimos cuadrados sobre {i : λᵢ > ε} y se reporta
        la desviación estándar residual.  0 ⇒ estado térmico exacto respecto de H.
        H se toma como el operador de número (única escala del motor interno).
        """
        lam = np.real(np.asarray(eigenvalues, dtype=np.float64))
        en = np.real(np.asarray(energy_levels, dtype=np.float64))
        mask = lam > _WILKINSON_LIMIT
        if int(np.count_nonzero(mask)) < 2:
            return 0.0
        y = -np.log(lam[mask])
        x = en[mask]
        x_c = x - np.mean(x)
        y_c = y - np.mean(y)
        denom = float(np.dot(x_c, x_c))
        if denom <= _WILKINSON_LIMIT:
            return float(np.std(y_c))
        beta = float(np.dot(x_c, y_c) / denom)
        resid = y_c - beta * x_c
        return float(np.sqrt(np.mean(np.square(resid))))

    def _number_eigenvalues_on_fock_basis(self) -> np.ndarray:
        """E_i = Hamming(i) = autovalores de N en la base computacional JW."""
        return np.array(
            [bin(i).count("1") for i in range(self._dim)],
            dtype=np.float64,
        )

    def _independent_tensor_audit(
        self,
        momentum_vector: np.ndarray,
        metric_tensor: np.ndarray,
        engine_tensor: np.ndarray,
        engine_div: float,
    ) -> Tuple[float, float, bool, Tuple[int, int, int]]:
        """Verificación independiente del T del motor + identidades algebraicas."""
        t_loc, div_loc = self._internal.compute_energy_momentum_tensor(
            momentum_vector, metric_tensor
        )
        t_eng = np.real(np.asarray(engine_tensor, dtype=np.float64))
        if t_eng.shape != t_loc.shape:
            logger.warning(
                "Forma de T del motor %s ≠ local %s; se audita solo el motor.",
                t_eng.shape,
                t_loc.shape,
            )
            t_use = 0.5 * (t_eng + t_eng.T) if t_eng.ndim == 2 else t_loc
        else:
            t_use = t_eng
        sym_res = float(np.linalg.norm(t_use - t_use.T, ord="fro"))
        signature = self._metric_signature(metric_tensor)
        econd_ok, _ = self._energy_conditions(t_use, metric_tensor, signature)
        weyl = float(np.real(np.trace(t_use)))
        # Conservamos el residuo del motor (puede incorporar ∇ numérico);
        # lo acotamos por debajo con el residuo algebraico local.
        div_residual = float(max(float(engine_div), div_loc, sym_res))
        return div_residual, weyl, econd_ok, signature

    def phase1_spectral_orientation(
        self,
        rho_initial: np.ndarray,
        momentum_vector: np.ndarray,
        metric_tensor: np.ndarray,
        gamma_annihilation: float,
    ) -> PhaseOneSpectralPacket:
        """
        [FASE 1 · Observe + Orient]

        Valida ρ₀, integra el semigrupo GKSL, certifica espectros, construye
        observables de Banach/entropía/canal/T^{μν} y emite el paquete que
        **es** el objeto inicial de la Fase 2 vía `_phase1_terminal_morphism`.
        """
        logger.info("Fase 1 (Observe+Orient): captura de ρ₀ y orientación de calibre.")

        if gamma_annihilation < 0.0:
            raise ValueError("gamma_annihilation debe ser ≥ 0")

        rho0 = self._validate_density_matrix(rho_initial, "rho_initial")
        p_vec, g_met = self._validate_metric_and_momentum(momentum_vector, metric_tensor)

        rho_final_raw, entropy_engine, exergy = self._engine.solve_lindblad_annihilation(
            rho_initial=rho0,
            gamma_annihilation=float(gamma_annihilation),
            time_step=self._dt,
        )
        rho1 = self._validate_density_matrix(rho_final_raw, "rho_final")

        eig0, _, spec_res_0 = self._certified_eigh(rho0, "rho_initial")
        eig1, _, spec_res_1 = self._certified_eigh(rho1, "rho_final")

        entropy_local = self._compute_von_neumann_entropy(eig1)
        if abs(entropy_local - float(entropy_engine)) > 100.0 * _WILKINSON_LIMIT:
            logger.warning(
                "Discrepancia de entropía motor=%.8f vs local=%.8f; se adopta local.",
                float(entropy_engine),
                entropy_local,
            )
            entropy = entropy_local
        else:
            entropy = float(entropy_engine)

        purity0 = self._compute_purity_from_eigenvalues(eig0)
        purity1 = self._compute_purity_from_eigenvalues(eig1)
        renyi2 = self._compute_renyi_entropy(eig1, alpha=2.0)
        smin = self._compute_min_entropy(eig1)
        banach = _operator_norm_hermitian(eig1)
        frob = _frobenius_norm_from_eig(eig1)

        delta = _hermite(rho0 - rho1)
        td = self._trace_distance(float(np.sum(np.abs(la.eigvalsh(delta)))))
        fid = self._uhlmann_fidelity(rho0, rho1)

        maj_ok, ky_res, hlp_eq = self._verify_ky_fan_majorization(eig0, eig1)

        n0 = self._fock_number_expectation(rho0)
        n1 = self._fock_number_expectation(rho1)
        n_mono = (n1 - n0) <= _PARTICLE_MONOTONE_TOL

        v0 = self._vacuum_fidelity(rho0)
        v1 = self._vacuum_fidelity(rho1)
        v_mono = (v0 - v1) <= _VACUUM_MONOTONE_TOL

        tensor_t, div_engine = self._engine.compute_energy_momentum_tensor(
            momentum_vector=p_vec,
            metric_tensor=g_met,
        )
        tensor_t = np.real(np.asarray(tensor_t, dtype=np.float64))
        div_res, weyl, econd_ok, signature = self._independent_tensor_audit(
            p_vec, g_met, tensor_t, float(div_engine)
        )
        sym_res = float(np.linalg.norm(tensor_t - tensor_t.T, ord="fro"))
        wilkinson_violation = div_res > self._div_thresh

        e_levels = self._number_eigenvalues_on_fock_basis()
        # Residuo KMS respecto de N en la base de ρ_final (diagonal de ocupación
        # como proxy de población; el espectro de ρ se empareja por orden).
        kms_res = self._kms_gibbs_residual(eig1, np.sort(e_levels))

        packet = PhaseOneSpectralPacket(
            rho_initial=_freeze(rho0),
            rho_final=_freeze(rho1),
            eigenvalues_initial=_freeze(eig0),
            eigenvalues_final=_freeze(eig1),
            fock_entropy=float(entropy),
            renyi_entropy_2=float(renyi2),
            min_entropy=float(smin),
            exergy_efficiency=float(exergy),
            purity_initial=float(purity0),
            purity_final=float(purity1),
            banach_op_norm=float(banach),
            frobenius_norm=float(frob),
            trace_distance=float(td),
            fidelity_uhlmann=float(fid),
            majorization_monotonic=bool(maj_ok),
            ky_fan_residual=float(ky_res),
            hlp_equality_residual=float(hlp_eq),
            particle_number_initial=float(n0),
            particle_number_final=float(n1),
            particle_number_monotone=bool(n_mono),
            vacuum_fidelity_initial=float(v0),
            vacuum_fidelity_final=float(v1),
            vacuum_fidelity_monotone=bool(v_mono),
            energy_momentum_tensor=_freeze(tensor_t),
            momentum_divergence_residual=float(div_res),
            tensor_symmetry_residual=float(sym_res),
            weyl_trace_anomaly=float(weyl),
            energy_condition_ok=bool(econd_ok),
            wilkinson_violation=bool(wilkinson_violation),
            spectral_residual_initial=float(spec_res_0),
            spectral_residual_final=float(spec_res_1),
            kms_gibbs_residual=float(kms_res),
            fock_betti_0=int(self._fock_support_betti0(rho1)),
            toeplitz_residual=float(self._toeplitz_residual(rho1)),
            metric_signature=tuple(int(x) for x in signature),  # type: ignore[arg-type]
        )
        logger.debug(
            "Fase 1: S=%.6f  S2=%.6f  ‖∇·T‖=%.3e  N: %.4f→%.4f  maj=%s  b0=%d",
            packet.fock_entropy,
            packet.renyi_entropy_2,
            packet.momentum_divergence_residual,
            packet.particle_number_initial,
            packet.particle_number_final,
            "OK" if packet.majorization_monotonic else "DIAG",
            packet.fock_betti_0,
        )
        # ---- morfismo terminal η: cierra Fase 1 y abre formalmente Fase 2 ----
        return self._phase1_terminal_morphism(packet)

    def _phase1_terminal_morphism(
        self,
        packet: PhaseOneSpectralPacket,
    ) -> PhaseOneSpectralPacket:
        r"""
        [FASE 1 · morfismo terminal η]  ≡  [FASE 2 · objeto inicial]

        Certifica invariantes internos del paquete espectral (idempotente).
        En el topos: η : Spec → Spec es la unidad que la Fase 2 consume.
        Cualquier método de la Fase 2 **comienza** reaplicando η.
        """
        if packet.rho_initial.shape != (self._dim, self._dim):
            raise ValueError("η: dim(ρ₀) corrupta")
        if packet.rho_final.shape != (self._dim, self._dim):
            raise ValueError("η: dim(ρ₁) corrupta")
        if packet.eigenvalues_initial.size != self._dim:
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
                packet.ky_fan_residual,
            ]
        ).all():
            raise ValueError("η: observables no finitos")
        return packet

    # =========================================================================
    # FASE 2 — DECIDE  (continúa η; produce χ ∈ Ω₃)
    #   2.0  η(packet)                         ← último morfismo de Fase 1
    #   2.1  predicados atómicos de violación
    #   2.2  operaciones de Heyting (∧, ∨, →, ¬)
    #   2.3  clasificador de subobjetos χ
    #   2.4  paquete de decisión → objeto inicial de Fase 3
    # =========================================================================
    def _classify_heyting_atoms(
        self,
        packet: PhaseOneSpectralPacket,
    ) -> Dict[str, bool]:
        """
        Predicados atómicos.  Críticos (join → VETOED) vs marginales
        (join → DEGRADED).  La majorización Ky Fan NO es crítica: el
        canal de aniquilación no es unital.
        """
        return {
            "entropy_violation": packet.fock_entropy > self._entropy_thresh,
            "divergence_violation": packet.momentum_divergence_residual > self._div_thresh,
            "particle_violation": not packet.particle_number_monotone,
            "vacuum_violation": not packet.vacuum_fidelity_monotone,
            "energy_condition_violation": not packet.energy_condition_ok,
            "spectral_certificate_violation": (
                packet.spectral_residual_final > _SPECTRAL_RESIDUAL_TOL
                or packet.spectral_residual_initial > _SPECTRAL_RESIDUAL_TOL
            ),
            "majorization_violation": not packet.majorization_monotonic,
            "kms_violation": packet.kms_gibbs_residual > _KMS_RESIDUAL_TOL
            and packet.fock_entropy > 0.1 * self._entropy_thresh,
            "entropy_marginal": packet.fock_entropy > 0.1 * self._entropy_thresh,
            "divergence_marginal": (
                packet.momentum_divergence_residual > 0.01 * self._div_thresh
            ),
            "weyl_marginal": abs(packet.weyl_trace_anomaly)
            > _ENERGY_COND_TOL * max(1.0, float(np.linalg.norm(packet.energy_momentum_tensor))),
            "toeplitz_marginal": packet.toeplitz_residual > _TOEPLITZ_TOL,
        }

    def phase2_heyting_decision(
        self,
        packet: PhaseOneSpectralPacket,
    ) -> PhaseTwoHeytingPacket:
        """
        [FASE 2 · Decide]

        Continuación formal de `_phase1_terminal_morphism`: re-certifica η
        y clasifica en el álgebra de Heyting

            Ω₃ = { COHERENT < DEGRADED < VETOED }.

        χ = ⊤  ⇔  veredicto = VETOED  ⇔  `subobject_classifier is True`.
        """
        certified = self._phase1_terminal_morphism(packet)  # ← nido Fase 1 → 2
        atoms = self._classify_heyting_atoms(certified)

        critical = (
            atoms["entropy_violation"],
            atoms["divergence_violation"],
            atoms["particle_violation"],
            atoms["vacuum_violation"],
            atoms["energy_condition_violation"],
            atoms["spectral_certificate_violation"],
        )
        marginal = (
            atoms["majorization_violation"],
            atoms["kms_violation"],
            atoms["entropy_marginal"],
            atoms["divergence_marginal"],
            atoms["weyl_marginal"],
            atoms["toeplitz_marginal"],
        )
        verdict = HeytingOmega3.classify(critical, marginal)
        chi_top = verdict is HeytingVerdict.VETOED
        impl = HeytingOmega3.implies(HeytingVerdict.COHERENT, verdict)

        packet2 = PhaseTwoHeytingPacket(
            heyting_verdict=verdict.value,
            entropy_violation=atoms["entropy_violation"],
            divergence_violation=atoms["divergence_violation"],
            majorization_violation=atoms["majorization_violation"],
            particle_violation=atoms["particle_violation"],
            vacuum_violation=atoms["vacuum_violation"],
            energy_condition_violation=atoms["energy_condition_violation"],
            kms_violation=atoms["kms_violation"],
            spectral_certificate_violation=atoms["spectral_certificate_violation"],
            subobject_classifier=bool(chi_top),
            implication_coherent_to_verdict=impl.value,
            join_of_atoms=verdict.value,
        )
        logger.debug(
            "Fase 2: Ω₃=%s  χ=%s  (S↑=%s  ∇T=%s  N=%s  vac=%s)",
            packet2.heyting_verdict,
            "⊤" if chi_top else "¬⊤",
            packet2.entropy_violation,
            packet2.divergence_violation,
            packet2.particle_violation,
            packet2.vacuum_violation,
        )
        return packet2

    # =========================================================================
    # FASE 3 — ACT  (continúa χ; ISR simulada + certificado)
    #   3.0  consume PhaseTwoHeytingPacket
    #   3.1  interlock crowbar *simulado* (sin GPIO, sin hardware real)
    #   3.2  digest forense SHA-256 y certificado inmutable
    # =========================================================================
    def _act_hardware_interlock_simulation(self, verdict: str) -> Tuple[bool, float]:
        """
        [FASE 3 · ISR simulada]

        Modelo estadístico de latencia IRAM (N(400, 5) ns, recortada a
        [380, 420]).  No toca GPIO, ESP32, ni el tiristor BT151 reales.
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

    def _forensic_digest(self, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, allow_nan=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _round_obs(self, x: float) -> float:
        if not np.isfinite(x):
            raise ValueError("observable no finito en el digest")
        return float(round(float(x), _DIGEST_ROUND_DECIMALS))

    def phase3_actuation_and_certificate(
        self,
        decision_packet: PhaseTwoHeytingPacket,
        spectral_packet: PhaseOneSpectralPacket,
    ) -> FockForensicCertificate:
        """
        [FASE 3 · Act]

        Recibe χ (Fase 2) y el paquete η (Fase 1), actúa el interlock
        simulado si χ=⊤ y emite el certificado con digest SHA-256.
        """
        certified = self._phase1_terminal_morphism(spectral_packet)
        interlock_fired, latency = self._act_hardware_interlock_simulation(
            decision_packet.heyting_verdict
        )

        if decision_packet.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "Fase 3: VETO. S=%.6f  ‖∇·T‖=%.3e  N_mono=%s  vac_mono=%s  maj=%s",
                certified.fock_entropy,
                certified.momentum_divergence_residual,
                certified.particle_number_monotone,
                certified.vacuum_fidelity_monotone,
                certified.majorization_monotonic,
            )

        trace_t = float(np.real(np.trace(np.asarray(certified.energy_momentum_tensor))))
        digest_payload = {
            "verdict": decision_packet.heyting_verdict,
            "S": self._round_obs(certified.fock_entropy),
            "S2": self._round_obs(certified.renyi_entropy_2),
            "Sinf": self._round_obs(certified.min_entropy),
            "exergy": self._round_obs(certified.exergy_efficiency),
            "TrT": self._round_obs(trace_t),
            "div": self._round_obs(certified.momentum_divergence_residual),
            "purity": self._round_obs(certified.purity_final),
            "F": self._round_obs(certified.fidelity_uhlmann),
            "N": self._round_obs(certified.particle_number_final),
            "ky": self._round_obs(certified.ky_fan_residual),
            "kms": self._round_obs(certified.kms_gibbs_residual),
            "chi": bool(decision_packet.subobject_classifier),
            "interlock": bool(interlock_fired),
        }
        digest = self._forensic_digest(digest_payload)

        certificate = FockForensicCertificate(
            phase="G_FOCK_FORENSIC_HALL_SUTURATED",
            heyting_verdict=decision_packet.heyting_verdict,
            fock_entropy=certified.fock_entropy,
            renyi_entropy_2=certified.renyi_entropy_2,
            min_entropy=certified.min_entropy,
            exergy_efficiency=certified.exergy_efficiency,
            energy_momentum_trace=trace_t,
            weyl_trace_anomaly=certified.weyl_trace_anomaly,
            momentum_divergence_residual=certified.momentum_divergence_residual,
            quantum_state_purity=certified.purity_final,
            fidelity_uhlmann=certified.fidelity_uhlmann,
            particle_number_final=certified.particle_number_final,
            majorization_monotonic=certified.majorization_monotonic,
            ky_fan_residual=certified.ky_fan_residual,
            kms_gibbs_residual=certified.kms_gibbs_residual,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            forensic_digest=digest,
        )
        logger.debug(
            "Fase 3: interlock=%s  latencia=%.2f ns  digest=%s",
            "ON" if interlock_fired else "OFF",
            latency,
            digest[:16],
        )
        return certificate

    # =========================================================================
    # ORQUESTACIÓN OODA  (composición η ; χ ; Act)
    # =========================================================================
    def execute_forensic_agent_cycle(
        self,
        rho_initial: np.ndarray,
        momentum_vector: np.ndarray,
        metric_tensor: np.ndarray,
        gamma_annihilation: float,
    ) -> FockForensicCertificate:
        """
        Compone las tres fases anidadas:

            cert = Act( χ( η( Observe(ρ₀, p, g, γ) ) ) )

        Cualquier excepción no contemplada colapsa a VETOED preventivo
        (fallo-cerrado) y emite certificado de emergencia.
        """
        try:
            spectral_packet = self.phase1_spectral_orientation(
                rho_initial=rho_initial,
                momentum_vector=momentum_vector,
                metric_tensor=metric_tensor,
                gamma_annihilation=gamma_annihilation,
            )
            decision_packet = self.phase2_heyting_decision(spectral_packet)
            return self.phase3_actuation_and_certificate(decision_packet, spectral_packet)
        except Exception:
            logger.exception("Excepción en el ciclo OODA: VETO preventivo (fail-closed).")
            return FockForensicCertificate(
                phase="G_FOCK_FORENSIC_HALL_EMERGENCY_VETO",
                heyting_verdict=HeytingVerdict.VETOED.value,
                fock_entropy=float("nan"),
                renyi_entropy_2=float("nan"),
                min_entropy=float("nan"),
                exergy_efficiency=float("nan"),
                energy_momentum_trace=float("nan"),
                weyl_trace_anomaly=float("nan"),
                momentum_divergence_residual=float("nan"),
                quantum_state_purity=float("nan"),
                fidelity_uhlmann=float("nan"),
                particle_number_final=float("nan"),
                majorization_monotonic=False,
                ky_fan_residual=float("nan"),
                kms_gibbs_residual=float("nan"),
                hardware_interlock_fired=True,
                actuation_latency_ns=_CROWBAR_IRAM_LATENCY_NS,
                forensic_digest="emergency_veto",
            )


# -----------------------------------------------------------------------------
# Exportación de firmas de calibre
# -----------------------------------------------------------------------------
__all__ = [
    "FockForensicCertificate",
    "PhaseOneSpectralPacket",
    "PhaseTwoHeytingPacket",
    "FockForensicHallAgent",
    "FockEngineProtocol",
    "HeytingVerdict",
    "HeytingOmega3",
]
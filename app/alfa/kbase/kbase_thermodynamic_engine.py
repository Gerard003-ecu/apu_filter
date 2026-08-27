# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : KBase Thermodynamic Engine (Motor Termodinámico de Base — Estrato α)║
║ Ruta   : app/alfa/kbase/kbase_thermodynamic_engine.py                        ║
║ Versión: 3.0.0-Doctoral-Riemann-Congruence-Gibbs-Clausius-Duhem-KBN-CSMD     ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA (rigor doctoral, no ornamental):                         ║
║ Sea (M, g) una variedad riemanniana de dimensión n = dim(Canvas) y sea       ║
║ (ℋ, ⟨·,·⟩) un espacio de Hilbert de dimensión n con álgebra de von Neumann  ║
║ ℬ(ℋ). El motor calcula, en la FPU y con inmunidad de Wilkinson, cuatro      ║
║ objetos termogeométricos acoplados:                                          ║
║                                                                              ║
║   (Congruencia)   T^♭ = g T g ,   T^♯ = g⁻¹ T g⁻¹                            ║
║                   (esto NO es un pullback por un difeomorfismo φ;            ║
║                    φ*T = J† T J se expone aparte si se inyecta J = dφ)       ║
║   (Potenciales)   H = U + P V ,   F = U − T S ,   G = H − T S = F + P V      ║
║                   (∂(G/T)/∂T)_P = −H/T²   (Gibbs–Helmholtz; sonda CSMD)      ║
║   (Clausius–Duhem) Φ = σ_int − ⟨q , dT⟩ / T²  ≥  0                           ║
║                   ⟨q , dT⟩ := q^μ ∂_μ T   (emparejamiento de dualidad,       ║
║                    invariante y SIN métrica; q contravariante, dT covariante)║
║   (Estacionariedad) ℐ = ‖[H, ρ]‖_HS²                                         ║
║                   ℐ = 0  ⇔  ρ conmuta con H (punto fijo unitario).           ║
║                   La irreversibilidad termodinámica real se reporta como     ║
║                   exceso de energía libre relativa al estado de Gibbs        ║
║                   ρ_β = e^{−βH}/Z :  D(ρ‖ρ_β) = β(ℱ − ℱ_eq) ≥ 0  (Spohn).   ║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (OODA):                       ║
║   Fase 1  Observe+Orient  :  In → PhaseOneKBasePacket                        ║
║   Fase 2  Decide          :  PhaseOneKBasePacket → PhaseTwoKBasePacket       ║
║   Fase 3  Act             :  (PhaseOne × PhaseTwo) → KBaseTelemetry          ║
║                                                                              ║
║ El último morfismo de la Fase k es, por construcción de tipos, el objeto     ║
║ inicial de la Fase k+1 (continuidad formal del retículo de métodos).         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Physics.KBaseThermodynamicEngine")

# ---------------------------------------------------------------------------
# Constantes metrológicas (análisis de Wilkinson / IEEE-754 binary64)
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CSMD_STEP: Final[float] = 1e-20
_HERMITIAN_TOLERANCE: Final[float] = 1e-12
_DENSITY_TRACE_TOLERANCE: Final[float] = 1e-10
_PSD_TOLERANCE: Final[float] = 1e-10


# =============================================================================
# Retículo de Heyting (cadena de 4 puntos = lógica interna de un topos
# sobre un espacio de fase discreto totalmente ordenado).
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena  ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.

    En un orden total el álgebra de Heyting es única:
        a ∧ b = min(a, b),   a ∨ b = max(a, b),
        a → b = ⊤  si a ≼ b,  y  a → b = b  en caso contrario,
        ¬a = a → ⊥.
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"
    CERTIFIED = "CERTIFIED"

    @property
    def rank(self) -> int:
        return {
            HeytingVerdict.VETOED: 0,
            HeytingVerdict.DEGRADED: 1,
            HeytingVerdict.COHERENT: 2,
            HeytingVerdict.CERTIFIED: 3,
        }[self]

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.CERTIFIED if self.rank <= other.rank else other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)


# =============================================================================
# Cartas espectrales / de densidad y paquetes inmutables
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Copia C-contigua de solo lectura: inmuniza el paquete frente a aliasing."""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _logsumexp(values: np.ndarray) -> float:
    """log ∑ exp(x_i) estable (evita overflow de la función de partición)."""
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        return -np.inf
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -np.inf
    shift = float(np.max(finite))
    return shift + float(np.log(np.sum(np.exp(finite - shift))))


@dataclass(frozen=True, slots=True)
class SpectralChart:
    r"""
    Factorización espectral g = U Λ U† y objetos derivados.

    Inversión de Tikhonov:  g_α⁻¹ = U (Λ + α I)⁻¹ U†  ≡  (g + α I)⁻¹.
    κ₂(g) = λ_max / λ_min (tras el desplazamiento), vol_g = √|det g|.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    inv_metric: np.ndarray
    condition_number: float
    volume_density: float
    operator_norm: float
    frobenius_norm: float
    spectral_gap: float
    regularized: bool
    tikhonov_alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "eigenvalues", _freeze_array(self.eigenvalues))
        object.__setattr__(self, "eigenvectors", _freeze_array(self.eigenvectors))
        object.__setattr__(self, "inv_metric", _freeze_array(self.inv_metric))


@dataclass(frozen=True, slots=True)
class DensityOperatorChart:
    r"""
    Carta espectral de un estado cuántico ρ ∈ 𝒟(ℋ).

    Axiomas: ρ = ρ†,  ρ ≥ 0,  Tr ρ = 1.
    S_vN = −∑ η_i log η_i  (convención 0 log 0 = 0),
    γ    = Tr(ρ²) ∈ [1/n, 1],
    E    = Tr(ρ H),
    ℱ    = E − T S_vN ,
    D(ρ‖ρ_β) = −S_vN + β E + log Z  ≥ 0.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    trace: float
    min_eigenvalue: float
    is_density: bool
    von_neumann_entropy: float
    purity: float
    energy_expectation: float
    quantum_free_energy: float
    log_partition: float
    relative_entropy_to_gibbs: float
    stationarity_defect: float
    banach_submultiplicative_defect: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "eigenvalues", _freeze_array(self.eigenvalues))
        object.__setattr__(self, "eigenvectors", _freeze_array(self.eigenvectors))


@dataclass(frozen=True, slots=True)
class PhaseOneKBasePacket:
    r"""
    Objeto terminal de la Fase 1 ≡ objeto inicial de la Fase 2.

    Contiene el 1-jet termogeométrico: congruencia métrica, potenciales de
    Gibbs–Helmholtz, sonda CSMD, desigualdad de Clausius–Duhem, carta de ρ
    y residuos de las identidades algebraicas de la termodinámica.
    """

    pullback_tensor: np.ndarray
    pullback_hermiticity_residual: float
    pullback_spectral_entropy: float
    gibbs_energy: float
    helmholtz_energy: float
    enthalpy: float
    internal_energy: float
    thermo_identity_residual: float
    csmd_dG_dT: float
    csmd_entropy_error: float
    dissipation_potential: float
    thermal_entropy_production: float
    clausius_coherent: bool
    fourier_conductivity: float
    fourier_residual: float
    thermal_dirichlet_energy: float
    spectral_chart: SpectralChart
    density_chart: DensityOperatorChart
    stationarity_defect: float
    temperature: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pullback_tensor", _freeze_array(self.pullback_tensor))


@dataclass(frozen=True, slots=True)
class SpectralThermoReport:
    r"""Subobjeto de la Fase 2: integridad espectral de g y de ρ."""

    metric_condition: float
    metric_gap: float
    density_is_valid: bool
    density_purity: float
    relative_entropy: float
    stationarity_defect: float
    banach_defect: float
    integrity_score: float


@dataclass(frozen=True, slots=True)
class PhaseTwoKBasePacket:
    r"""
    Objeto terminal de la Fase 2 ≡ primer factor del dominio de la Fase 3.
    """

    irreversibility: float
    heyting_verdict: str
    heyting_rank: int
    clausius_threshold_violated: bool
    stationarity_threshold_violated: bool
    density_axiom_violated: bool
    condition_threshold_violated: bool
    spectral_integrity: SpectralThermoReport
    heyting_score: float
    exergy_destruction: float


@dataclass(frozen=True, slots=True)
class KBaseTelemetry:
    r"""Sello inmutable de telemetría (objeto terminal del functor OODA)."""

    pullback_tensor: np.ndarray
    gibbs_energy: float
    enthalpy: float
    helmholtz_energy: float
    dissipation_potential: float
    clausius_coherent: bool
    irreversibility: float
    heyting_verdict: str
    thermo_identity_residual: float
    csmd_entropy_error: float
    relative_entropy_to_gibbs: float
    von_neumann_entropy: float
    purity: float
    condition_number: float
    conservation_residual: float
    exergy_destruction: float
    spectral_integrity_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pullback_tensor", _freeze_array(self.pullback_tensor))

    def to_dict(self) -> Dict[str, Any]:
        """Serialización compatible con el pasaporte de la Malla (claves v2 + v3)."""
        return {
            "pullback_tensor": np.array(self.pullback_tensor, copy=True),
            "gibbs_energy": self.gibbs_energy,
            "enthalpy": self.enthalpy,
            "helmholtz_energy": self.helmholtz_energy,
            "dissipation_potential": self.dissipation_potential,
            "clausius_coherent": self.clausius_coherent,
            "irreversibility": self.irreversibility,
            "heyting_verdict": self.heyting_verdict,
            "thermo_identity_residual": self.thermo_identity_residual,
            "csmd_entropy_error": self.csmd_entropy_error,
            "relative_entropy_to_gibbs": self.relative_entropy_to_gibbs,
            "von_neumann_entropy": self.von_neumann_entropy,
            "purity": self.purity,
            "condition_number": self.condition_number,
            "conservation_residual": self.conservation_residual,
            "exergy_destruction": self.exergy_destruction,
            "spectral_integrity_score": self.spectral_integrity_score,
        }


# =============================================================================
# Motor
# =============================================================================
class KBaseThermodynamicEngine:
    r"""
    Motor Termodinámico de la Base Táctica (Estrato α — KBASE).

    El endofunctor OODA  T = Act ∘ Decide ∘ Observe  actúa sobre el espacio
    de estados materiales. Cada fase es un morfismo explícito; la composición
    `execute_thermodynamic_cycle` es T mismo.
    """

    def __init__(self, dimension_n: int, reg_param: float = 1e-15) -> None:
        if dimension_n <= 0:
            raise ValueError("La dimensión del espacio de fase debe ser estrictamente positiva.")
        self._n: Final[int] = int(dimension_n)
        self._reg: Final[float] = float(max(reg_param, _WILKINSON_FLOOR))

    # =========================================================================
    # FASE 1 — OBSERVE + ORIENT
    # Aparato de medición (KBN, CSMD, espectro de g y de ρ) y 1-jets físicos.
    # El último método, `_phase1_observe_orient`, tiene por codominio
    # `PhaseOneKBasePacket`, que ES el dominio de `_phase2_spectral_integrity`.
    # =========================================================================
    def kahan_sum(self, arr: np.ndarray) -> float:
        r"""
        Suma compensada de Kahan–Babuška–Neumaier sobre un 1-tensor.

        Neumaier (1974) corrige el caso |x_{i+1}| > |S_i| que Kahan clásico
        pierde. El error hacia delante satisface

            |fl(∑ x_i) − ∑ x_i|  ≤  (2u + O(u²)) ∑ |x_i|

        independiente de n. Se proyecta a la parte real: las observables
        termodinámicas de este motor son reales.
        """
        if np.asarray(arr).ndim != 1:
            raise ValueError(
                f"kahan_sum espera un vector 1-D, se recibió {np.asarray(arr).shape}"
            )
        return float(np.real(self._neumaier_sum(arr)))

    def _neumaier_sum(self, arr: np.ndarray) -> complex:
        """KBN sobre ℝ o ℂ; no fuerza float64 a priori para preservar holomorfía CSMD."""
        data = np.asarray(arr).ravel()
        if data.size == 0:
            return 0.0
        if not np.all(np.isfinite(data)):
            raise ValueError("kahan/neumaier: el sumando contiene valores no finitos")
        use_complex = np.iscomplexobj(data)
        total: complex = 0.0j if use_complex else 0.0
        compensator: complex = 0.0j if use_complex else 0.0
        for x in data:
            t = total + x
            if abs(total) >= abs(x):
                compensator += (total - t) + x
            else:
                compensator += (x - t) + total
            total = t
        return total + compensator

    def _validate_square_matrix(self, M: np.ndarray, name: str) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(
                f"{name} debe ser una matriz cuadrada (n × n), se recibió {M.shape}"
            )
        if M.shape[0] != self._n:
            raise ValueError(f"{name} debe ser {self._n} × {self._n}, se recibió {M.shape}")
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_vector(self, v: np.ndarray, name: str) -> None:
        if v.ndim != 1 or v.shape[0] != self._n:
            raise ValueError(
                f"{name} debe ser un vector de dimensión {self._n}, se recibió {v.shape}"
            )
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_scalar(self, value: Any, name: str) -> float:
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} debe ser un escalar finito")
        return float(value)

    def _validate_hermitian(
        self,
        M: np.ndarray,
        name: str,
        tolerance: float = _HERMITIAN_TOLERANCE,
    ) -> None:
        skew = M - M.conj().T
        if la.norm(skew, ord="fro") > tolerance * max(1.0, la.norm(M, ord="fro")):
            raise ValueError(
                f"{name} no es hermítica: ||M − M†||_F / ||M||_F excede {tolerance}"
            )

    def _hermitize(self, M: np.ndarray) -> np.ndarray:
        """Proyección de Hilbert–Schmidt sobre el subespacio de hermitianas."""
        return 0.5 * (M + M.conj().T)

    def _factorize_metric(self, metric_tensor: np.ndarray) -> SpectralChart:
        r"""
        Una sola descomposición de Hilbert (eigh) — jamás dos.

        Regularización de Tikhonov espectral: λ ↦ λ + α, α = max(reg, 0).
        Equivale a invertir g + α I, proximal de Moreau de la energía de
        Dirichlet en el álgebra de Banach (M_n, ||·||₂).
        """
        self._validate_square_matrix(metric_tensor, "metric_tensor")
        herm = self._hermitize(metric_tensor)
        self._validate_hermitian(herm, "metric_tensor")

        eigenvalues, eigenvectors = la.eigh(herm)
        eigenvalues = np.real(eigenvalues)

        regularized = bool(np.any(eigenvalues <= self._reg))
        if regularized:
            logger.warning(
                "Métrica no definida positiva o mal condicionada (λ_min=%.3e); "
                "Tikhonov α=%.3e.",
                float(np.min(eigenvalues)),
                self._reg,
            )
        shifted = np.maximum(eigenvalues + self._reg, _WILKINSON_FLOOR)
        inv_metric = (eigenvectors * (1.0 / shifted)) @ eigenvectors.conj().T

        lam_abs = np.abs(shifted)
        cond = float(np.max(lam_abs) / np.min(lam_abs))
        log_vol = 0.5 * float(
            self.kahan_sum(np.log(np.maximum(lam_abs, _WILKINSON_FLOOR)))
        )
        volume = float(np.exp(log_vol))
        op_norm = float(np.max(lam_abs))
        frob = float(
            np.sqrt(
                np.real(self._neumaier_sum(shifted.astype(np.complex128) ** 2))
            )
        )
        if shifted.size >= 2:
            ordered = np.sort(lam_abs)
            gap = float(ordered[1] - ordered[0])
        else:
            gap = float(lam_abs[0])

        return SpectralChart(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            inv_metric=np.real_if_close(inv_metric, tol=1e6),
            condition_number=cond,
            volume_density=volume,
            operator_norm=op_norm,
            frobenius_norm=frob,
            spectral_gap=gap,
            regularized=regularized,
            tikhonov_alpha=self._reg,
        )

    def compute_riemannian_pullback(
        self,
        G: np.ndarray,
        tensor_M: np.ndarray,
        tensor_type: str = "covariant",
        jacobian: Optional[np.ndarray] = None,
        spectral_chart: Optional[SpectralChart] = None,
    ) -> np.ndarray:
        r"""
        [KBASE 1 — CONGRUENCIA MÉTRICA / PULLBACK POR dφ]

        Dos operaciones geométricas distintas conviven bajo este nombre
        histórico; se despachan de forma explícita.

        1. Congruencia (tipo tensorial, carta fija):
           (0,2) covariante     \(\tilde T = g\, T\, g\)
           (2,0) contravariante \(\tilde T = g^{-1} T g^{-1}\)
           con \(g^{-1} = (g + \alpha I)^{-1}\) (Tikhonov, una sola eigh).

        2. Pullback verdadero por un difeomorfismo φ, si se inyecta
           \(J = d\phi\):
           \((\phi^* T)_p(X, Y) = T_{\phi(p)}(JX, JY)\)
           ⇒  \(\phi^* T = J^\dagger T J\)  para un (0,2)-tensor.

        La simetrización de Weyl \(g \leftarrow (g + g^\dagger)/2\) se aplica
        antes de cualquier producto, para proyectar ruido antihermítico.

        Args:
            G: Tensor métrico Riemanniano g_μν (n × n, hermítico).
            tensor_M: 2-tensor basal a transportar (n × n).
            tensor_type: ``"covariant"`` o ``"contravariant"``.
            jacobian: Jacobiano dφ opcional; si se da, se devuelve J† M J.
            spectral_chart: Carta reutilizable (evita refactorizar g).

        Returns:
            2-tensor resultante (n × n).
        """
        self._validate_square_matrix(G, "G")
        self._validate_square_matrix(tensor_M, "tensor_M")
        self._validate_hermitian(G, "G")

        if jacobian is not None:
            self._validate_square_matrix(jacobian, "jacobian")
            return jacobian.conj().T @ tensor_M @ jacobian

        chart = spectral_chart if spectral_chart is not None else self._factorize_metric(G)
        g_sym = self._hermitize(np.asarray(G))
        M = np.asarray(tensor_M)

        if tensor_type == "covariant":
            return g_sym @ M @ g_sym
        if tensor_type == "contravariant":
            g_inv = np.asarray(chart.inv_metric)
            return g_inv @ M @ g_inv
        raise ValueError(
            "Tipo de tensor no soportado. Debe ser 'covariant' o 'contravariant'."
        )

    def _pullback_spectral_entropy(self, tensor: np.ndarray) -> float:
        r"""
        Entropía espectral de von Neumann del 2-tensor transportado:

            η_i = |λ_i| / ∑ |λ_j| ,   S = −∑ η_i log η_i .

        Es un invariante de congruencia unitaria; NO es una entropía de
        Hodge–de Rham (ésta exigiría un laplaciano sobre un complejo de
        cocadenas, que este motor no finge poseer).
        """
        herm = self._hermitize(np.asarray(tensor, dtype=np.complex128))
        evals = np.real(la.eigvalsh(herm))
        weights = np.abs(evals)
        total = float(np.real(self._neumaier_sum(weights.astype(np.float64))))
        if total <= _WILKINSON_FLOOR:
            return 0.0
        probs = weights / total
        acc: List[float] = []
        for p in probs:
            if p > _WILKINSON_FLOOR:
                acc.append(float(-p * np.log(p)))
        return float(np.real(self._neumaier_sum(np.asarray(acc, dtype=np.float64))))

    def compute_gibbs_free_energy(
        self,
        internal_energy_U: float,
        pressure_P: float,
        volume_V: float,
        temperature_T: float,
        entropy_S: float,
    ) -> Tuple[float, float]:
        r"""
        [KBASE 2 — POTENCIALES DE GIBBS–HELMHOLTZ]

        Identidades algebraicas (sistema cerrado, k_B = 1):

            H = U + P V ,     F = U − T S ,     G = H − T S = F + P V .

        La ecuación de Gibbs–Helmholtz
            (∂(G/T)/∂T)_P = −H / T²
        se reduce, a H y S fijos, a  ∂G/∂T = −S, que se audita por CSMD
        en la Fase 1. Esta firma pública se conserva bit a bit y devuelve
        únicamente (G, H).

        Returns:
            (energía libre de Gibbs, entalpía).
        """
        U = self._validate_scalar(internal_energy_U, "internal_energy_U")
        P = self._validate_scalar(pressure_P, "pressure_P")
        V = self._validate_scalar(volume_V, "volume_V")
        T = self._validate_scalar(temperature_T, "temperature_T")
        S = self._validate_scalar(entropy_S, "entropy_S")
        if T <= _WILKINSON_FLOOR:
            raise ValueError(
                "La temperatura absoluta debe ser estrictamente positiva "
                f"(se recibió {T})."
            )
        if S < -_WILKINSON_FLOOR:
            logger.warning("Entropía negativa (S=%.6e): admisible en Shannon diferencial, anómala en Clausius.", S)
        if V <= 0.0:
            logger.warning("Volumen no positivo (V=%.6e).", V)

        enthalpy = self.kahan_sum(np.array([U, P * V], dtype=np.float64))
        gibbs_energy = self.kahan_sum(np.array([enthalpy, -T * S], dtype=np.float64))
        return float(gibbs_energy), float(enthalpy)

    def _complex_step_dG_dT(
        self,
        enthalpy: float,
        entropy_S: float,
        temperature_T: float,
        step: float = _CSMD_STEP,
    ) -> float:
        r"""
        CSMD de la extensión holomorfa  G(z) = H − z S  (H, S constantes).

        Im(G(T + i h)) / h = −S  exactamente, sin cancelación sustractiva.
        El conjugado está prohibido: rompería la holomorfía.
        """
        value = complex(enthalpy) - (complex(temperature_T) + 1.0j * step) * complex(entropy_S)
        return float(np.imag(value) / step)

    def evaluate_clausius_duhem(
        self,
        entropy_production_rate: float,
        heat_flux_q: np.ndarray,
        temp_gradient_gradT: np.ndarray,
        temperature_T: float,
    ) -> Tuple[float, bool]:
        r"""
        [KBASE 3 — DESIGUALDAD DE CLAUSIUS–DUHEM]

        Emparejamiento de dualidad (invariante, sin métrica):

            ⟨q , dT⟩ := q^μ ∂_μ T

        siempre que `heat_flux_q` se interprete como campo contravariante
        y `temp_gradient_gradT` como la 1-forma dT. Entonces

            σ_th = − ⟨q , dT⟩ / T² ,     Φ = σ_int + σ_th = σ_int − ⟨q,dT⟩/T² .

        Fourier q = −k ∇T implica ⟨q,dT⟩ ≤ 0 y por tanto σ_th ≥ 0.
        Φ ≥ −ε_u  certifica la segunda ley a tolerancia de Wilkinson.

        Returns:
            (potencial de disipación neto Φ, coherencia booleana).
        """
        sigma_int = self._validate_scalar(entropy_production_rate, "entropy_production_rate")
        T = self._validate_scalar(temperature_T, "temperature_T")
        if T <= _WILKINSON_FLOOR:
            raise ValueError(
                "La temperatura absoluta del fibrado debe ser estrictamente "
                f"mayor que cero (se recibió {T})."
            )
        self._validate_vector(np.asarray(heat_flux_q), "heat_flux_q")
        self._validate_vector(np.asarray(temp_gradient_gradT), "temp_gradient_gradT")

        pairing = float(np.real(np.dot(np.asarray(heat_flux_q), np.asarray(temp_gradient_gradT))))
        conduction_term = pairing / (T ** 2)
        dissipation_potential = self.kahan_sum(
            np.array([sigma_int, -conduction_term], dtype=np.float64)
        )
        is_coherent = dissipation_potential >= -_WILKINSON_FLOOR
        return float(dissipation_potential), bool(is_coherent)

    def _fourier_audit(
        self,
        heat_flux_q: np.ndarray,
        temp_gradient_gradT: np.ndarray,
        spectral_chart: SpectralChart,
    ) -> Tuple[float, float, float]:
        r"""
        Auditoría de Fourier isotrópica y energía de Dirichlet de q.

        k̂ = − ⟨q,dT⟩ / ||dT||²_2   (mínimos cuadrados),
        r  = ||q + k̂ ∇T||_2 / (||q||_2 + ε) ,
        E(q) = q^μ g_{μν} q^ν      (energía de la 1-forma musicalmente bajada).
        """
        q = np.asarray(heat_flux_q, dtype=np.float64)
        gT = np.asarray(temp_gradient_gradT, dtype=np.float64)
        gTgT = float(np.real(np.dot(gT, gT)))
        if gTgT <= _WILKINSON_FLOOR:
            conductivity = 0.0
            residual = float(la.norm(q))
        else:
            conductivity = float(-np.real(np.dot(q, gT)) / gTgT)
            residual = float(la.norm(q + conductivity * gT) / max(la.norm(q), _WILKINSON_FLOOR))
        g = self._hermitize(
            np.asarray(spectral_chart.eigenvectors)
            @ np.diag(np.asarray(spectral_chart.eigenvalues, dtype=np.float64))
            @ np.asarray(spectral_chart.eigenvectors).conj().T
        )
        dirichlet = float(np.real(q @ g @ q))
        return conductivity, residual, dirichlet

    def calculate_spectral_irreversibility(
        self,
        rho: np.ndarray,
        hamiltonian_H: np.ndarray,
    ) -> float:
        r"""
        [KBASE 4 — DEFECTO DE ESTACIONARIEDAD / NORMA HS DEL CONMUTADOR]

        ℐ = ‖[H, ρ]‖_HS² = Tr( [H,ρ]† [H,ρ] ).

        Advertencia semántica (corrección respecto a v2): el flujo de
        von Neumann  ρ̇ = −i[H,ρ]  es unitario y conserva S_vN. ℐ mide
        la *partida de la estacionariedad hamiltoniana*, no la producción
        de entropía. La irreversibilidad termodinámica se reporta en la
        carta de densidad como D(ρ‖ρ_β).

        Returns:
            ℐ ≥ 0  (Hilbert–Schmidt al cuadrado, KBN sobre la diagonal).
        """
        self._validate_square_matrix(rho, "rho")
        self._validate_square_matrix(hamiltonian_H, "hamiltonian_H")
        self._validate_hermitian(rho, "rho")
        self._validate_hermitian(hamiltonian_H, "hamiltonian_H")

        commutator = hamiltonian_H @ rho - rho @ hamiltonian_H
        gram = commutator.conj().T @ commutator
        diag_elements = np.real(np.diagonal(gram))
        irreversibility = self.kahan_sum(np.asarray(diag_elements, dtype=np.float64))
        return max(0.0, float(irreversibility))

    def _factorize_density(
        self,
        rho: np.ndarray,
        hamiltonian_H: np.ndarray,
        temperature_T: float,
    ) -> DensityOperatorChart:
        r"""
        Carta espectral de ρ y observables de equilibrio (Gibbs / KMS).

        Z = Tr e^{−βH}  se evalúa por log-sum-exp sobre el espectro de H.
        D(ρ‖ρ_β) = Tr(ρ log ρ) − Tr(ρ log ρ_β)
                 = −S_vN + β Tr(ρ H) + log Z .
        """
        self._validate_square_matrix(rho, "rho")
        self._validate_square_matrix(hamiltonian_H, "hamiltonian_H")
        self._validate_hermitian(rho, "rho")
        self._validate_hermitian(hamiltonian_H, "hamiltonian_H")

        rho_h = self._hermitize(np.asarray(rho, dtype=np.complex128))
        H_h = self._hermitize(np.asarray(hamiltonian_H, dtype=np.complex128))

        evals, evecs = la.eigh(rho_h)
        evals = np.real(evals)
        trace = float(np.real(self._neumaier_sum(evals.astype(np.float64))))
        min_eig = float(np.min(evals))
        is_density = (
            min_eig >= -_PSD_TOLERANCE
            and abs(trace - 1.0) <= _DENSITY_TRACE_TOLERANCE
        )
        if not is_density:
            logger.warning(
                "ρ no satisface los axiomas de operador densidad "
                "(tr=%.6e, λ_min=%.6e).",
                trace, min_eig,
            )

        acc_s: List[float] = []
        acc_g: List[float] = []
        for eta in evals:
            clipped = max(float(eta), 0.0)
            acc_g.append(clipped * clipped)
            if clipped > _WILKINSON_FLOOR:
                acc_s.append(float(-clipped * np.log(clipped)))
        von_neumann = float(np.real(self._neumaier_sum(np.asarray(acc_s, dtype=np.float64))))
        purity = float(np.real(self._neumaier_sum(np.asarray(acc_g, dtype=np.float64))))

        energy = float(np.real(np.trace(rho_h @ H_h)))
        T = float(temperature_T)
        free = energy - T * von_neumann

        h_eigs = np.real(la.eigvalsh(H_h))
        if T <= _WILKINSON_FLOOR:
            log_z = np.inf
            relative = np.inf
        else:
            beta = 1.0 / T
            log_z = _logsumexp(-beta * h_eigs)
            relative = float(-von_neumann + beta * energy + log_z)
            relative = max(0.0, relative)

        stationarity = self.calculate_spectral_irreversibility(rho_h, H_h)

        op_H = float(la.norm(H_h, ord=2))
        op_rho = float(la.norm(rho_h, ord=2))
        lhs = float(la.norm(H_h @ rho_h, ord=2))
        rhs = op_H * op_rho
        banach = max(0.0, lhs / rhs - 1.0) if rhs > _WILKINSON_FLOOR else 0.0

        return DensityOperatorChart(
            eigenvalues=evals,
            eigenvectors=evecs,
            trace=trace,
            min_eigenvalue=min_eig,
            is_density=is_density,
            von_neumann_entropy=von_neumann,
            purity=purity,
            energy_expectation=energy,
            quantum_free_energy=free,
            log_partition=float(log_z),
            relative_entropy_to_gibbs=float(relative),
            stationarity_defect=stationarity,
            banach_submultiplicative_defect=float(banach),
        )

    def _phase1_observe_orient(
        self,
        G: np.ndarray,
        tensor_M: np.ndarray,
        tensor_type: str,
        internal_energy_U: float,
        pressure_P: float,
        volume_V: float,
        temperature_T: float,
        entropy_S: float,
        entropy_production_rate: float,
        heat_flux_q: np.ndarray,
        temp_gradient_gradT: np.ndarray,
        rho: np.ndarray,
        hamiltonian_H: np.ndarray,
        jacobian: Optional[np.ndarray] = None,
    ) -> PhaseOneKBasePacket:
        r"""
        [FASE 1 — ÚLTIMO MORFISMO: OBSERVE + ORIENT]

        Codominio
        ---------
        `PhaseOneKBasePacket`

        Continuidad formal
        ------------------
        Este paquete **es** el dominio del primer morfismo de la Fase 2,
        `_phase2_spectral_integrity(self, packet: PhaseOneKBasePacket)`.
        No existe ningún objeto intersticial: la composición

            _phase2_spectral_integrity ∘ _phase1_observe_orient

        está tipada estrictamente (retículo de métodos anidados).
        """
        logger.info(
            "Fase 1: carta espectral de g, congruencia, Gibbs–Helmholtz+CSMD, "
            "Clausius–Duhem, carta de ρ."
        )

        chart = self._factorize_metric(G)
        pullback = self.compute_riemannian_pullback(
            G, tensor_M, tensor_type, jacobian=jacobian, spectral_chart=chart
        )
        pull_skew = pullback - pullback.conj().T
        pull_herm_res = float(la.norm(pull_skew, ord="fro"))
        pull_entropy = self._pullback_spectral_entropy(pullback)

        U = self._validate_scalar(internal_energy_U, "internal_energy_U")
        P = self._validate_scalar(pressure_P, "pressure_P")
        V = self._validate_scalar(volume_V, "volume_V")
        T = self._validate_scalar(temperature_T, "temperature_T")
        S = self._validate_scalar(entropy_S, "entropy_S")

        gibbs, enthalpy = self.compute_gibbs_free_energy(U, P, V, T, S)
        helmholtz = self.kahan_sum(np.array([U, -T * S], dtype=np.float64))
        identity_terms = np.array(
            [
                abs(enthalpy - (U + P * V)),
                abs(gibbs - (enthalpy - T * S)),
                abs(helmholtz - (U - T * S)),
                abs(gibbs - (helmholtz + P * V)),
            ],
            dtype=np.float64,
        )
        thermo_residual = float(np.real(self._neumaier_sum(identity_terms)))

        csmd = self._complex_step_dG_dT(enthalpy, S, T)
        csmd_err = float(abs(csmd + S))

        dissipation, coherent = self.evaluate_clausius_duhem(
            entropy_production_rate, heat_flux_q, temp_gradient_gradT, T
        )
        pairing = float(
            np.real(np.dot(np.asarray(heat_flux_q), np.asarray(temp_gradient_gradT)))
        )
        sigma_th = float(-pairing / (T ** 2))
        conductivity, fourier_res, dirichlet = self._fourier_audit(
            heat_flux_q, temp_gradient_gradT, chart
        )

        density = self._factorize_density(rho, hamiltonian_H, T)

        packet = PhaseOneKBasePacket(
            pullback_tensor=pullback,
            pullback_hermiticity_residual=pull_herm_res,
            pullback_spectral_entropy=pull_entropy,
            gibbs_energy=gibbs,
            helmholtz_energy=float(helmholtz),
            enthalpy=enthalpy,
            internal_energy=U,
            thermo_identity_residual=thermo_residual,
            csmd_dG_dT=csmd,
            csmd_entropy_error=csmd_err,
            dissipation_potential=dissipation,
            thermal_entropy_production=sigma_th,
            clausius_coherent=coherent,
            fourier_conductivity=conductivity,
            fourier_residual=fourier_res,
            thermal_dirichlet_energy=dirichlet,
            spectral_chart=chart,
            density_chart=density,
            stationarity_defect=density.stationarity_defect,
            temperature=T,
        )
        logger.debug(
            "Fase 1 completa: G=%.6f F=%.6f H=%.6f Φ=%.6e CD=%s ℐ=%.6e "
            "D(ρ‖ρβ)=%.6e CSMD=%.3e κ=%.3e",
            gibbs, helmholtz, enthalpy, dissipation, coherent,
            density.stationarity_defect, density.relative_entropy_to_gibbs,
            csmd_err, chart.condition_number,
        )
        return packet

    # =========================================================================
    # FASE 2 — DECIDE
    # Dominio = PhaseOneKBasePacket  (codominio del último método de Fase 1).
    # Se evalúa la integridad espectral de g y de ρ y se clasifica el estado
    # en el álgebra de Heyting. El último método, `_phase2_decide`, tiene por
    # codominio PhaseTwoKBasePacket, segundo factor de `_phase3_conservation_audit`.
    # =========================================================================
    def _phase2_spectral_integrity(
        self,
        packet: PhaseOneKBasePacket,
    ) -> SpectralThermoReport:
        r"""
        [FASE 2 — PRIMER MORFISMO: INTEGRIDAD ESPECTRAL]

        Dominio
        -------
        `PhaseOneKBasePacket`  ← continuación formal de `_phase1_observe_orient`.

        Diagnósticos (teoría espectral + álgebra de Banach (M_n, ||·||₂)):
          • κ₂(g), gap de g;
          • axiomas de ρ, pureza, D(ρ‖ρ_β), ℐ = ‖[H,ρ]‖_HS²;
          • defecto de submultiplicatividad ‖Hρ‖₂ ≤ ‖H‖₂‖ρ‖₂.
        """
        chart = packet.spectral_chart
        dens = packet.density_chart

        cond_term = 1.0 / (1.0 + np.log10(max(chart.condition_number, 1.0)))
        gap_term = float(np.tanh(chart.spectral_gap / max(chart.operator_norm, _WILKINSON_FLOOR)))
        dens_term = 1.0 if dens.is_density else 0.0
        rel_term = float(np.exp(-dens.relative_entropy_to_gibbs))
        stat_term = float(np.exp(-dens.stationarity_defect))
        banach_term = float(np.exp(-dens.banach_submultiplicative_defect / _MACHINE_EPS))
        cd_term = 1.0 if packet.clausius_coherent else 0.0
        csmd_term = float(np.exp(-packet.csmd_entropy_error / 1e-12))
        score = float(
            cond_term
            * max(gap_term, 0.0)
            * dens_term
            * rel_term
            * stat_term
            * min(banach_term, 1.0)
            * cd_term
            * csmd_term
        )
        return SpectralThermoReport(
            metric_condition=float(chart.condition_number),
            metric_gap=float(chart.spectral_gap),
            density_is_valid=dens.is_density,
            density_purity=dens.purity,
            relative_entropy=dens.relative_entropy_to_gibbs,
            stationarity_defect=dens.stationarity_defect,
            banach_defect=dens.banach_submultiplicative_defect,
            integrity_score=score,
        )

    def _phase2_heyting_classify(
        self,
        packet: PhaseOneKBasePacket,
        integrity: SpectralThermoReport,
        irreversibility_threshold: float,
        condition_threshold: float,
    ) -> Tuple[HeytingVerdict, float, Dict[str, bool]]:
        r"""
        Clasificación en el álgebra de Heyting.

        Scores en [0, 1] (semántica [0,1]-valuada del topos) y meet.
        Un veto atómico (Clausius–Duhem, ρ ilegal, T ≤ 0, CSMD rota)
        colapsa a VETOED por modus ponens interno.
        """
        cd_viol = not packet.clausius_coherent
        stat_viol = packet.stationarity_defect > irreversibility_threshold
        dens_viol = not packet.density_chart.is_density
        cond_viol = integrity.metric_condition > condition_threshold
        ident_viol = packet.thermo_identity_residual > 1e-10
        csmd_viol = packet.csmd_entropy_error > 1e-8
        temp_viol = packet.temperature <= _WILKINSON_FLOOR

        cd_score = 1.0 if packet.clausius_coherent else 0.0
        # Φ grande y positivo no es un defecto: es disipación admisible.
        # Se penaliza únicamente Φ negativo (ya cubierto por cd_score).
        stat_score = float(
            np.exp(
                -packet.stationarity_defect
                / max(irreversibility_threshold, _WILKINSON_FLOOR)
            )
        )
        rel_score = float(np.exp(-packet.density_chart.relative_entropy_to_gibbs))
        ident_score = float(np.exp(-packet.thermo_identity_residual / 1e-10))
        csmd_score = float(np.exp(-packet.csmd_entropy_error / 1e-8))
        meet = min(
            cd_score,
            stat_score,
            rel_score,
            ident_score,
            csmd_score,
            integrity.integrity_score,
            0.0 if dens_viol else 1.0,
            0.0 if temp_viol else 1.0,
        )

        hard_veto = cd_viol or dens_viol or temp_viol or csmd_viol or ident_viol
        if hard_veto:
            verdict = HeytingVerdict.VETOED
        elif meet >= 0.99:
            verdict = HeytingVerdict.CERTIFIED
        elif meet >= 0.90:
            verdict = HeytingVerdict.COHERENT
        elif meet >= 0.50:
            verdict = HeytingVerdict.DEGRADED
        else:
            verdict = HeytingVerdict.VETOED

        flags = {
            "clausius": cd_viol,
            "stationarity": stat_viol,
            "density": dens_viol,
            "condition": cond_viol,
            "identity": ident_viol,
            "csmd": csmd_viol,
            "temperature": temp_viol,
        }
        return verdict, float(meet), flags

    def _phase2_decide(
        self,
        packet: PhaseOneKBasePacket,
        irreversibility_threshold: float = 0.1,
        condition_threshold: float = 1e12,
        ambient_temperature: Optional[float] = None,
    ) -> PhaseTwoKBasePacket:
        r"""
        [FASE 2 — ÚLTIMO MORFISMO: DECIDE]

        Codominio
        ---------
        `PhaseTwoKBasePacket`

        Continuidad formal
        ------------------
        Junto con el `PhaseOneKBasePacket` residual, este objeto constituye
        el dominio del primer morfismo de la Fase 3,
        `_phase3_conservation_audit(self, packet1, packet2)`.

        La destrucción de exergía (Gouy–Stodola) es  T₀ Φ, con T₀ la
        temperatura ambiente si se inyecta, o T del sistema en su defecto.
        """
        integrity = self._phase2_spectral_integrity(packet)
        verdict, score, flags = self._phase2_heyting_classify(
            packet,
            integrity,
            irreversibility_threshold=irreversibility_threshold,
            condition_threshold=condition_threshold,
        )
        t0 = (
            float(ambient_temperature)
            if ambient_temperature is not None
            else packet.temperature
        )
        exergy = float(t0 * packet.dissipation_potential)
        decision = PhaseTwoKBasePacket(
            irreversibility=packet.stationarity_defect,
            heyting_verdict=verdict.value,
            heyting_rank=verdict.rank,
            clausius_threshold_violated=flags["clausius"],
            stationarity_threshold_violated=flags["stationarity"],
            density_axiom_violated=flags["density"],
            condition_threshold_violated=flags["condition"],
            spectral_integrity=integrity,
            heyting_score=score,
            exergy_destruction=exergy,
        )
        logger.debug(
            "Fase 2 completa: veredicto=%s score=%.4f ℐ=%.6e Ḃ=%.6e κ=%.3e",
            verdict.value, score, packet.stationarity_defect, exergy,
            integrity.metric_condition,
        )
        return decision

    # =========================================================================
    # FASE 3 — ACT
    # Dominio = PhaseOneKBasePacket × PhaseTwoKBasePacket.
    # Audita leyes de conservación y sella telemetría inmutable.
    # =========================================================================
    def _phase3_conservation_audit(
        self,
        packet1: PhaseOneKBasePacket,
        packet2: PhaseTwoKBasePacket,
    ) -> float:
        r"""
        [FASE 3 — PRIMER MORFISMO: AUDITORÍA DE CONSERVACIÓN]

        Dominio
        -------
        `PhaseOneKBasePacket × PhaseTwoKBasePacket`
        ← continuación formal de `_phase1_observe_orient` y `_phase2_decide`.

        Residuos adimensionales acumulados (KBN) de identidades que deben
        anularse en aritmética exacta:
          1. G = H − T S = F + P V  (ya reducido a `thermo_identity_residual`);
          2. sonda CSMD  |∂G/∂T + S| ;
          3. |Tr ρ − 1|_+  y  (−λ_min(ρ))_+ ;
          4. parte antihermítica residual del pullback;
          5. inconsistencia del flag CD con el signo de Φ;
          6. parte negativa de D(ρ‖ρ_β) (violación numérica de Klein).
        """
        dens = packet1.density_chart
        cd_inconsistent = (
            1.0
            if (packet1.clausius_coherent and packet1.dissipation_potential < -_WILKINSON_FLOOR)
            or ((not packet1.clausius_coherent) and packet1.dissipation_potential >= 0.0)
            else 0.0
        )
        terms = np.asarray(
            [
                packet1.thermo_identity_residual,
                packet1.csmd_entropy_error,
                max(0.0, abs(dens.trace - 1.0) - _DENSITY_TRACE_TOLERANCE),
                max(0.0, -dens.min_eigenvalue - _PSD_TOLERANCE),
                packet1.pullback_hermiticity_residual,
                cd_inconsistent,
                max(0.0, -dens.relative_entropy_to_gibbs),
            ],
            dtype=np.float64,
        )
        residual = float(np.real(self._neumaier_sum(terms)))
        logger.debug(
            "Fase 3 auditoría: residuo de conservación=%.6e (veredicto previo=%s)",
            residual, packet2.heyting_verdict,
        )
        return residual

    def _phase3_act(
        self,
        packet1: PhaseOneKBasePacket,
        packet2: PhaseTwoKBasePacket,
        conservation_residual: float,
    ) -> KBaseTelemetry:
        """
        [FASE 3 — ÚLTIMO MORFISMO: ACT]

        Sella el objeto terminal `KBaseTelemetry`. No hay Fase 4: este sello
        es el valor de T(estado) inyectable en el pasaporte de la Malla.
        """
        return KBaseTelemetry(
            pullback_tensor=packet1.pullback_tensor,
            gibbs_energy=packet1.gibbs_energy,
            enthalpy=packet1.enthalpy,
            helmholtz_energy=packet1.helmholtz_energy,
            dissipation_potential=packet1.dissipation_potential,
            clausius_coherent=packet1.clausius_coherent,
            irreversibility=packet2.irreversibility,
            heyting_verdict=packet2.heyting_verdict,
            thermo_identity_residual=packet1.thermo_identity_residual,
            csmd_entropy_error=packet1.csmd_entropy_error,
            relative_entropy_to_gibbs=packet1.density_chart.relative_entropy_to_gibbs,
            von_neumann_entropy=packet1.density_chart.von_neumann_entropy,
            purity=packet1.density_chart.purity,
            condition_number=packet2.spectral_integrity.metric_condition,
            conservation_residual=conservation_residual,
            exergy_destruction=packet2.exergy_destruction,
            spectral_integrity_score=packet2.spectral_integrity.integrity_score,
        )

    def execute_thermodynamic_cycle(
        self,
        G: np.ndarray,
        tensor_M: np.ndarray,
        tensor_type: str,
        internal_energy_U: float,
        pressure_P: float,
        volume_V: float,
        temperature_T: float,
        entropy_S: float,
        entropy_production_rate: float,
        heat_flux_q: np.ndarray,
        temp_gradient_gradT: np.ndarray,
        rho: np.ndarray,
        hamiltonian_H: np.ndarray,
        irreversibility_threshold: float = 0.1,
        jacobian: Optional[np.ndarray] = None,
        ambient_temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        r"""
        Functor OODA  T = Act ∘ Decide ∘ Observe.

        Encadena las tres fases anidadas sin objetos huérfanos:

            In  --F1→  PhaseOne  --F2→  PhaseTwo  --F3→  KBaseTelemetry  →  dict

        La firma pública (más `jacobian` y `ambient_temperature` opcionales)
        permanece compatible con el pasaporte de telemetría de la Malla.
        """
        packet1 = self._phase1_observe_orient(
            G,
            tensor_M,
            tensor_type,
            internal_energy_U,
            pressure_P,
            volume_V,
            temperature_T,
            entropy_S,
            entropy_production_rate,
            heat_flux_q,
            temp_gradient_gradT,
            rho,
            hamiltonian_H,
            jacobian=jacobian,
        )
        packet2 = self._phase2_decide(
            packet1,
            irreversibility_threshold=irreversibility_threshold,
            ambient_temperature=ambient_temperature,
        )
        conservation = self._phase3_conservation_audit(packet1, packet2)
        telemetry = self._phase3_act(packet1, packet2, conservation)
        return telemetry.to_dict()


# -----------------------------------------------------------------------------
# Bloque de autocomprobación
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando autocomprobación de KBase Thermodynamic Engine v3...")
    engine = KBaseThermodynamicEngine(dimension_n=3)

    G = np.eye(3, dtype=np.float64)
    Tm = np.diag([1.0, 2.0, 3.0])

    T_pull = engine.compute_riemannian_pullback(G, Tm, tensor_type="covariant")
    print(f"¿Congruencia con g=I es idéntica?: {np.allclose(Tm, T_pull)}")

    J = np.diag([2.0, 1.0, 0.5])
    T_true = engine.compute_riemannian_pullback(G, Tm, jacobian=J)
    print(f"¿Pullback J†TJ coincide?: {np.allclose(T_true, J.T @ Tm @ J)}")

    gibbs, h = engine.compute_gibbs_free_energy(100.0, 10.0, 2.0, 300.0, 0.1)
    print(f"Energía de Gibbs: {gibbs:.4f}, Entalpía: {h:.4f}  (esperado G=90, H=120)")

    diss, ok = engine.evaluate_clausius_duhem(
        entropy_production_rate=0.05,
        heat_flux_q=np.array([1.0, 0.5, 0.0]),
        temp_gradient_gradT=np.array([0.1, 0.0, 0.2]),
        temperature_T=300.0,
    )
    print(f"Potencial de disipación: {diss:.6f}, ¿cumple Clausius-Duhem?: {ok}")

    rho = np.diag([0.7, 0.2, 0.1])
    H = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    irrev = engine.calculate_spectral_irreversibility(rho, H)
    print(f"Defecto de estacionariedad ‖[H,ρ]‖_HS²: {irrev:.6f}")

    result = engine.execute_thermodynamic_cycle(
        G=G,
        tensor_M=Tm,
        tensor_type="covariant",
        internal_energy_U=100.0,
        pressure_P=10.0,
        volume_V=2.0,
        temperature_T=300.0,
        entropy_S=0.1,
        entropy_production_rate=0.05,
        heat_flux_q=np.array([1.0, 0.5, 0.0]),
        temp_gradient_gradT=np.array([0.1, 0.0, 0.2]),
        rho=rho,
        hamiltonian_H=H,
        irreversibility_threshold=0.1,
    )
    print("Veredicto:", result["heyting_verdict"])
    print("Helmholtz:", result["helmholtz_energy"])
    print("Residuo identidades:", result["thermo_identity_residual"])
    print("Error CSMD |∂G/∂T + S|:", result["csmd_entropy_error"])
    print("D(ρ‖ρβ):", result["relative_entropy_to_gibbs"])
    print("S_vN:", result["von_neumann_entropy"])
    print("Residuo de conservación:", result["conservation_residual"])

    # Veto duro: producción intrínseca insuficiente frente a conducción antípoda.
    veto = engine.execute_thermodynamic_cycle(
        G=G,
        tensor_M=Tm,
        tensor_type="covariant",
        internal_energy_U=100.0,
        pressure_P=10.0,
        volume_V=2.0,
        temperature_T=300.0,
        entropy_S=0.1,
        entropy_production_rate=-1.0,
        heat_flux_q=np.array([1.0, 0.0, 0.0]),
        temp_gradient_gradT=np.array([1.0, 0.0, 0.0]),
        rho=rho,
        hamiltonian_H=H,
    )
    print("Veto Clausius-Duhem esperado:", veto["heyting_verdict"], "CD=", veto["clausius_coherent"])


__all__ = [
    "KBaseThermodynamicEngine",
    "PhaseOneKBasePacket",
    "PhaseTwoKBasePacket",
    "KBaseTelemetry",
    "SpectralChart",
    "DensityOperatorChart",
    "SpectralThermoReport",
    "HeytingVerdict",
]
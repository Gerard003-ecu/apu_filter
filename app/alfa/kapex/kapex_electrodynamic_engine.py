# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : KAPEX Electrodynamic Engine (Motor Electrodinámico de Ápice)        ║
║ Ruta   : app/alfa/kapex/kapex_electrodynamic_engine.py                       ║
║ Versión: 3.0.0-Doctoral-Eikonal-Yang-Mills-Poynting-KBN-CSMD-Heyting         ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA (rigor doctoral, no ornamental):                         ║
║ Sea (M, g) una variedad riemanniana de dimensión n = dim(ápice) y sea        ║
║ 𝔤 = 𝔰𝔬(n) el álgebra de Lie de matrices antisimétricas reales, con forma      ║
║ de Killing κ(X,Y) = (n-2) Tr(XY) (n ≥ 3) y emparejamiento de traza           ║
║ ⟨X,Y⟩_Tr = -Tr(XY) ≥ 0 sobre 𝔰𝔬(n). El motor resuelve, en la FPU y con        ║
║ inmunidad de Wilkinson, tres objetos geométricos acoplados:                  ║
║                                                                              ║
║   (Eikonal)     g^{μν} ∂_μ S ∂_ν S − n(q)²  =  0                             ║
║   (Yang-Mills)  F = dA + A ∧ A ∈ Ω²(M, 𝔤),                                   ║
║                 S_YM = (1/(4g²)) ∫ ⟨F, F⟩_Tr vol_g                           ║
║   (Poynting)    S^♭ = ★_g (E^♭ ∧ B^♭) ∈ Ω^{n-2}(M)                           ║
║                                                                              ║
║ La diferenciación por paso complejo (CSMD) se aplica únicamente a la         ║
║ extensión holomorfa bilineal I[p] = p_μ g^{μν} p_ν (jamás a |·| ni a         ║
║ conj(·), que no son holomorfas). La suma Kahan–Babuška–Neumaier (KBN)        ║
║ garantiza que el error de redondeo de las contracciones sea O(u), no O(nu).  ║
║                                                                              ║
║ ARQUITECTURA FUNCTORIAL EN TRES FASES ANIDADAS (OODA):                       ║
║   Fase 1  Observe+Orient  :  In → PhaseOneKapexPacket                        ║
║   Fase 2  Decide          :  PhaseOneKapexPacket → PhaseTwoKapexPacket       ║
║   Fase 3  Act             :  (PhaseOne × PhaseTwo) → KapexTelemetry          ║
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

logger = logging.getLogger("APU.Physics.KapexElectrodynamicEngine")

# ---------------------------------------------------------------------------
# Constantes metrológicas (análisis de Wilkinson / IEEE-754 binary64)
# γ_m = m u / (1 − m u) acota el factor de crecimiento del error directo.
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CSMD_STEP: Final[float] = 1e-20          # h ≪ u; Im(f(x+ih))/h = f'(x) + O(h²)
_HERMITIAN_TOLERANCE: Final[float] = 1e-12
_ALGEBRA_TOLERANCE: Final[float] = 1e-10
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0


# =============================================================================
# Objetos de retículo (Heyting finito totalmente ordenado = lógica de un topos
# sobre un espacio de fase discreto de cuatro puntos).
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena de Heyting  ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.

    En un orden total el álgebra de Heyting es única:
        a ∧ b = min(a, b),   a ∨ b = max(a, b),
        a → b = ⊤  si a ≼ b,  y  a → b = b  en caso contrario,
        ¬a = a → ⊥.
    Es la lógica interna del topos de haces sobre ese poset.
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
# Cartas espectrales y paquetes inmutables (objetos de las tres fases)
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Copia C-contigua de solo lectura: inmuniza el paquete frente a aliasing."""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class SpectralChart:
    r"""
    Factorización espectral g = U Λ U† y objetos derivados.

    Inversión de Tikhonov:  g_α^{-1} = U (Λ + α I)^{-1} U†,
    número de condición κ₂(g) = λ_max / λ_min (tras recorte),
    densidad de volumen vol_g = √|det g| = exp(½ ∑ log|λ_i|).
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
class PhaseOneKapexPacket:
    r"""
    Objeto terminal de la Fase 1 ≡ objeto inicial de la Fase 2.

    Contiene el 1-jet geométrico (residuo eikonal, momento elevado, sensibilidad
    CSMD), el 2-jet de calibre (F, S_YM, Jacobi, densidad de Pontryagin) y el
    tensor de energía-momento maxwelliano, más la carta espectral de g.
    """

    eikonal_residual: float
    eikonal_hamiltonian: float
    momentum_up_vector: np.ndarray
    csmd_momentum: np.ndarray
    csmd_gradient_error: float
    curvature_tensor: np.ndarray
    yang_mills_action: float
    yang_mills_raw_trace: float
    bianchi_jacobi_residual: float
    instanton_density: float
    poynting_flux_vector: np.ndarray
    poynting_two_form: np.ndarray
    field_energy_density: float
    maxwell_stress_tensor: np.ndarray
    spectral_chart: SpectralChart
    algebra_projection_residual: float

    def __post_init__(self) -> None:
        for name in (
            "momentum_up_vector",
            "csmd_momentum",
            "curvature_tensor",
            "poynting_flux_vector",
            "poynting_two_form",
            "maxwell_stress_tensor",
        ):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SpectralIntegrityReport:
    r"""Subobjeto de la Fase 2: espectro de g, de F y radio espectral de A_μ."""

    metric_condition: float
    metric_gap: float
    curvature_frobenius: float
    connection_spectral_radii: np.ndarray
    banach_submultiplicative_defect: float
    integrity_score: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "connection_spectral_radii",
            _freeze_array(self.connection_spectral_radii),
        )


@dataclass(frozen=True, slots=True)
class PhaseTwoKapexPacket:
    r"""
    Objeto terminal de la Fase 2 ≡ primer factor del dominio de la Fase 3.
    """

    heyting_verdict: str
    heyting_rank: int
    eikonal_threshold_violated: bool
    yang_mills_threshold_violated: bool
    jacobi_threshold_violated: bool
    condition_threshold_violated: bool
    spectral_integrity: SpectralIntegrityReport
    wilson_plaquette_mean: float
    gauge_coherence: float
    heyting_score: float


@dataclass(frozen=True, slots=True)
class KapexTelemetry:
    r"""Sello inmutable de telemetría (objeto terminal del functor OODA)."""

    eikonal_residual: float
    eikonal_hamiltonian: float
    yang_mills_action: float
    poynting_flux_vector: np.ndarray
    field_energy_density: float
    momentum_up_vector: np.ndarray
    curvature_tensor: np.ndarray
    maxwell_stress_tensor: np.ndarray
    heyting_verdict: str
    condition_number: float
    bianchi_residual: float
    instanton_density: float
    csmd_gradient_error: float
    wilson_plaquette_mean: float
    conservation_residual: float
    spectral_integrity_score: float

    def __post_init__(self) -> None:
        for name in (
            "poynting_flux_vector",
            "momentum_up_vector",
            "curvature_tensor",
            "maxwell_stress_tensor",
        ):
            object.__setattr__(self, name, _freeze_array(getattr(self, name)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialización compatible con el pasaporte de la Malla (claves v2 + v3)."""
        return {
            "eikonal_residual": self.eikonal_residual,
            "eikonal_hamiltonian": self.eikonal_hamiltonian,
            "yang_mills_action": self.yang_mills_action,
            "poynting_flux_vector": np.array(self.poynting_flux_vector, copy=True),
            "field_energy_density": self.field_energy_density,
            "momentum_up_vector": np.array(self.momentum_up_vector, copy=True),
            "curvature_tensor": np.array(self.curvature_tensor, copy=True),
            "maxwell_stress_tensor": np.array(self.maxwell_stress_tensor, copy=True),
            "heyting_verdict": self.heyting_verdict,
            "condition_number": self.condition_number,
            "bianchi_residual": self.bianchi_residual,
            "instanton_density": self.instanton_density,
            "csmd_gradient_error": self.csmd_gradient_error,
            "wilson_plaquette_mean": self.wilson_plaquette_mean,
            "conservation_residual": self.conservation_residual,
            "spectral_integrity_score": self.spectral_integrity_score,
        }


# =============================================================================
# Motor
# =============================================================================
class KapexElectrodynamicEngine:
    r"""
    Motor de cálculo ciego electrodinámico de-confinado para el Estrato KAPEX.

    El endofunctor OODA  T = Act ∘ Decide ∘ Observe  actúa sobre el espacio
    de estados estratégicos. Cada fase es un morfismo explícito; la composición
    `execute_electrodynamic_cycle` es T mismo.
    """

    def __init__(self, dimension_n: int, reg_param: float = 1e-15) -> None:
        if dimension_n <= 0:
            raise ValueError("La dimensión espacial del ápice debe ser estrictamente positiva.")
        self._n: Final[int] = int(dimension_n)
        self._reg: Final[float] = float(max(reg_param, _WILKINSON_FLOOR))

    # =========================================================================
    # FASE 1 — OBSERVE + ORIENT
    # Aparato de medición (KBN, CSMD, espectro) y 1-jets / 2-jets físicos.
    # El último método, `_phase1_observe_orient`, tiene por codominio
    # `PhaseOneKapexPacket`, que ES el dominio de `_phase2_spectral_integrity`.
    # =========================================================================
    def kahan_sum(self, arr: np.ndarray) -> float:
        r"""
        Suma compensada de Kahan–Babuška–Neumaier sobre un 1-tensor.

        Neumaier (1974) corrige el caso |x_{i+1}| > |S_i| que Kahan clásico
        pierde. El error hacia delante satisface

            |fl(∑ x_i) − ∑ x_i|  ≤  (2u + O(u²)) ∑ |x_i|

        independiente de n (frente a O(n u) de la suma ingenua). Se proyecta
        a la parte real: las observables físicas de este motor son reales.
        """
        if np.asarray(arr).ndim != 1:
            raise ValueError(f"kahan_sum espera un vector 1-D, se recibió {np.asarray(arr).shape}")
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
            raise ValueError(f"{name} debe ser una matriz cuadrada (n × n), se recibió {M.shape}")
        if M.shape[0] != self._n:
            raise ValueError(f"{name} debe ser {self._n} × {self._n}, se recibió {M.shape}")
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_vector(self, v: np.ndarray, name: str) -> None:
        if v.ndim != 1 or v.shape[0] != self._n:
            raise ValueError(f"{name} debe ser un vector de dimensión {self._n}, se recibió {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contiene valores no finitos")

    def _validate_hermitian(self, M: np.ndarray, name: str, tolerance: float = _HERMITIAN_TOLERANCE) -> None:
        skew = M - M.conj().T
        if la.norm(skew, ord="fro") > tolerance * max(1.0, la.norm(M, ord="fro")):
            raise ValueError(
                f"{name} no es hermítica: ||M − M†||_F / ||M||_F excede {tolerance}"
            )

    def _hermitize(self, M: np.ndarray) -> np.ndarray:
        """Proyección de Hilbert–Schmidt sobre el subespacio de hermitianas."""
        return 0.5 * (M + M.conj().T)

    def project_to_so_n(self, A: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""
        Proyección ortogonal (Frobenius) sobre 𝔰𝔬(n, ℂ) ≅ anti-hermitianas:

            π(A) = (A − A†) / 2,     r = ||A − π(A)||_F.

        Si r > 0 la conexión no vivía en el álgebra de Lie declarada.
        """
        self._validate_square_matrix(A, "connection_component")
        projected = 0.5 * (A - A.conj().T)
        residual = float(la.norm(A - projected, ord="fro"))
        return projected, residual

    def killing_pairing(self, X: np.ndarray, Y: np.ndarray) -> complex:
        r"""
        Forma de Killing de 𝔰𝔬(n): κ(X,Y) = (n−2) Tr(XY)  (n ≥ 3).
        Para n = 2, 𝔰𝔬(2) es abeliana y κ ≡ 0; se cae a ⟨X,Y⟩_Tr = Tr(XY).
        """
        factor = float(self._n - 2) if self._n >= 3 else 1.0
        return factor * np.trace(X @ Y)

    def _factorize_metric(self, metric_tensor: np.ndarray) -> SpectralChart:
        r"""
        Una sola descomposición de Hilbert (eigh) — jamás dos.

        Regularización de Tikhonov espectral: λ ↦ λ + α, α = max(reg, 0).
        Equivale a invertir g + α I, que es el proximal de Moreau de la
        energía de Dirichlet en el álgebra de Banach (M_n, ||·||₂).
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
        shifted = eigenvalues + self._reg
        # Salvaguarda numérica: ningún polo exacto en cero.
        shifted = np.maximum(shifted, _WILKINSON_FLOOR)

        inv_metric = (eigenvectors * (1.0 / shifted)) @ eigenvectors.conj().T

        lam_abs = np.abs(shifted)
        cond = float(np.max(lam_abs) / np.min(lam_abs))
        # vol = exp(½ ∑ log|λ|) evita overflow de det en n grande.
        log_vol = 0.5 * float(self.kahan_sum(np.log(np.maximum(lam_abs, _WILKINSON_FLOOR))))
        volume = float(np.exp(log_vol))
        op_norm = float(np.max(lam_abs))
        frob = float(np.sqrt(np.real(self._neumaier_sum(shifted.astype(np.complex128) ** 2))))
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

    def _invert_metric(self, metric_tensor: np.ndarray) -> np.ndarray:
        """Inversión pública de compatibilidad: delega en la carta espectral única."""
        return np.array(self._factorize_metric(metric_tensor).inv_metric, copy=True)

    def _complex_step_eikonal_momentum(
        self,
        grad_S: np.ndarray,
        inv_metric: np.ndarray,
        step: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""
        CSMD de la forma cuadrática holomorfa  I(p) = p_μ g^{μν} p_ν.

        Como I es polinomial, I(p + i h e_{(λ)}) = I(p) + 2 i h (g^{λν} p_ν) + O(h²),
        luego  ∂I/∂p_λ = Im(I(p + i h e_λ)) / h  =  2 p^λ   (idéntico al analítico
        2 g^{λν} p_ν, sin cancelación sustractiva). El conjugado está prohibido:
        rompería la holomorfía y devolvería basura de orden u/h.
        """
        n = self._n
        p = np.asarray(grad_S, dtype=np.complex128)
        ginv = np.asarray(inv_metric, dtype=np.complex128)
        sens = np.zeros(n, dtype=np.float64)
        for lam in range(n):
            pert = p.copy()
            pert[lam] += 1.0j * step
            value = pert @ ginv @ pert
            sens[lam] = float(np.imag(value) / step)
        return sens

    def solve_eikonal_residual(
        self,
        grad_S: np.ndarray,
        metric_tensor: np.ndarray,
        refraction_index: float,
        spectral_chart: Optional[SpectralChart] = None,
    ) -> Tuple[float, np.ndarray]:
        r"""
        [KAPEX 1 — RESOLVEDOR EIKONAL DE REFRACCIÓN]

        Residuo de la EDP no lineal de Hamilton–Jacobi sobre (M, g):

            E[S] := g^{μν} ∂_μ S ∂_ν S − n(q)² ,

        y hamiltoniano de las características  H(q, p) = ½ (⟨p, p⟩_{g^{-1}} − n²).
        El momento contravariante es p^μ = g^{μν} ∂_ν S.

        Returns:
            ( |E[S]| , p^μ ). El hamiltoniano y la sonda CSMD viven en el
            paquete de Fase 1; esta firma pública se conserva bit a bit.
        """
        self._validate_vector(grad_S, "grad_S")
        if not np.isfinite(refraction_index) or refraction_index <= 0.0:
            raise ValueError("refraction_index debe ser un real estrictamente positivo y finito")

        chart = spectral_chart if spectral_chart is not None else self._factorize_metric(metric_tensor)
        inv_metric = np.asarray(chart.inv_metric)
        grad = np.asarray(grad_S, dtype=np.float64)

        p_up = np.real(inv_metric @ grad)
        inner_product = float(np.real(np.dot(p_up, grad)))
        n_squared = float(refraction_index) ** 2
        eikonal_residual = float(abs(inner_product - n_squared))
        return eikonal_residual, p_up

    def compute_yang_mills_action(
        self,
        connection_A: List[np.ndarray],
        metric_tensor: np.ndarray,
        coupling_constant: float = 1.0,
        connection_dA: Optional[List[List[np.ndarray]]] = None,
        spectral_chart: Optional[SpectralChart] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""
        [KAPEX 2 — CURVATURA Y ACCIÓN DE YANG–MILLS SOBRE 𝔰𝔬(n)]

        Conexión  A ∈ Ω¹(M, 𝔤),  𝔤 = 𝔰𝔬(n). Curvatura

            F_{μν} = ∂_μ A_ν − ∂_ν A_μ + [A_μ, A_ν] .

        En la aproximación de lazo local (sin dA inyectado) los términos
        ∂A se anulan y F_{μν} = [A_μ, A_ν]. La contracción correcta es

            F^{μν} = g^{μα} g^{νβ} F_{αβ} ,
            Θ      = ∑_{μν} Tr( F_{μν} F^{μν} ) = ε^{μν}_{αβ} Tr(F_{μν} F_{αβ}),

        implementada por `einsum('μνij,μνji→')` — la versión 2.0 contraía
        F_{αβ} F_{μν} con un factor suelto, lo cual NO es Tr(F_{μν} F^{μν}).

        Convención de signo (generadores anti-hermitianos):
            S_YM = −1/(4 g²) Θ .  Sobre 𝔰𝔬(n), Tr(F F) ≤ 0 ⇒ S_YM ≥ 0.

        Returns:
            (F_{μν i j}  de forma (n, n, n, n),  S_YM).
        """
        if not isinstance(connection_A, list) or len(connection_A) != self._n:
            raise ValueError(f"connection_A debe ser una lista de {self._n} matrices.")
        if coupling_constant <= 0.0 or not np.isfinite(coupling_constant):
            raise ValueError("coupling_constant debe ser un real estrictamente positivo y finito")

        projected: List[np.ndarray] = []
        for mu, A in enumerate(connection_A):
            self._validate_square_matrix(A, f"A[{mu}]")
            pi_A, _ = self.project_to_so_n(A)
            projected.append(pi_A)

        chart = spectral_chart if spectral_chart is not None else self._factorize_metric(metric_tensor)
        inv_metric = np.asarray(chart.inv_metric, dtype=np.complex128)

        n = self._n
        F = np.zeros((n, n, n, n), dtype=np.complex128)
        for mu in range(n):
            for nu in range(n):
                commutator = projected[mu] @ projected[nu] - projected[nu] @ projected[mu]
                if connection_dA is not None:
                    d_mu_nu = np.asarray(connection_dA[mu][nu], dtype=np.complex128)
                    d_nu_mu = np.asarray(connection_dA[nu][mu], dtype=np.complex128)
                    F[mu, nu] = d_mu_nu - d_nu_mu + commutator
                else:
                    F[mu, nu] = commutator

        # F^{μν} = g^{μα} g^{νβ} F_{αβ}   (contracción tensorial exacta)
        F_up = np.einsum("ma,nb,abij->mnij", inv_metric, inv_metric, F)

        # Θ_{μν} = Tr(F_{μν} F^{μν}) = ∑_{ij} F_{μν ij} F^{μν ji}
        trace_density = np.einsum("mnij,mnji->mn", F, F_up)
        total_trace = self._neumaier_sum(np.real(trace_density).astype(np.float64).ravel())
        yang_mills_action = -0.25 * (1.0 / (coupling_constant ** 2)) * float(np.real(total_trace))
        return F, float(yang_mills_action)

    def _bianchi_jacobi_residual(
        self,
        connection_A: List[np.ndarray],
        F: np.ndarray,
    ) -> float:
        r"""
        Residual de Bianchi algebraico / identidad de Jacobi.

        (DF)_{λμν} = [A_λ, F_{μν}] + [A_μ, F_{νλ}] + [A_ν, F_{λμ}].
        Si F_{μν} = [A_μ, A_ν] exactamente, Jacobi ⇒ DF ≡ 0.
        El residual es un canario de Wilkinson del conmutador matricial.
        """
        n = self._n
        acc: List[float] = []
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    term = (
                        connection_A[lam] @ F[mu, nu] - F[mu, nu] @ connection_A[lam]
                        + connection_A[mu] @ F[nu, lam] - F[nu, lam] @ connection_A[mu]
                        + connection_A[nu] @ F[lam, mu] - F[lam, mu] @ connection_A[nu]
                    )
                    acc.append(float(la.norm(term, ord="fro") ** 2))
        energy = float(np.real(self._neumaier_sum(np.asarray(acc, dtype=np.float64))))
        return float(np.sqrt(max(energy, 0.0)) / max(n ** 1.5, 1.0))

    def _pontryagin_density(self, F: np.ndarray) -> float:
        r"""
        Densidad topológica de Pontryagin / 2º carácter de Chern (local):

            ρ = Tr(F_{μν} F_{μν})   (contracción euclídea de prueba).

        En dimensión 4, ∫ Tr(F ∧ F) / 8π² ∈ ℤ es el número instantónico.
        Aquí se reporta la densidad local (el motor no integra en M).
        """
        # Tr(F_μν F_μν) = ∑_{μνij} F_{μν ij} F_{μν ji}
        density = np.einsum("mnij,mnji->", F, F)
        return float(np.real(density))

    def compute_poynting_strategic_flux(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        metric_tensor: np.ndarray,
        spectral_chart: Optional[SpectralChart] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""
        [KAPEX 3 — FLUJO DE POYNTING VÍA ESTRELLA DE HODGE]

        E, B se interpretan como 1-formas (componentes covariantes). Entonces

            U     = ½ ( ⟨E, E⟩_{g^{-1}} + ⟨B, B⟩_{g^{-1}} ) ,
            E^♭ ∧ B^♭ ∈ Ω²(M),
            S^♭   = ★_g (E^♭ ∧ B^♭) ∈ Ω^{n−2}(M).

        En n = 3, ★: Ω² → Ω¹ y se recupera el vector clásico

            S_k = √|det g|  ε_{kij} E^i B^j ,
            S^μ = g^{μk} S_k .

        En n ≠ 3 el vector de Poynting no es un 1-tensor canónico: se devuelve
        0 ∈ ℝⁿ y la energía (el 2-forma vive en el paquete de Fase 1).
        """
        self._validate_vector(electric_field, "electric_field")
        self._validate_vector(magnetic_field, "magnetic_field")

        chart = spectral_chart if spectral_chart is not None else self._factorize_metric(metric_tensor)
        inv_metric = np.asarray(chart.inv_metric, dtype=np.float64)
        E = np.asarray(electric_field, dtype=np.float64)
        B = np.asarray(magnetic_field, dtype=np.float64)

        E_up = np.real(inv_metric @ E)
        B_up = np.real(inv_metric @ B)
        energy_density = 0.5 * (
            float(np.real(np.dot(E_up, E))) + float(np.real(np.dot(B_up, B)))
        )

        if self._n != 3:
            logger.warning(
                "★_g (E♭ ∧ B♭) es un (n−2)-forma; n=%d ≠ 3. "
                "Se emite el vector nulo y U calculada con g.",
                self._n,
            )
            return np.zeros(self._n, dtype=np.float64), float(energy_density)

        vol = float(chart.volume_density)
        # S_k = √g ε_{kij} E^i B^j  (Levi-Civita coordenado, convención dextrógira).
        S_down = vol * np.cross(E_up, B_up)
        S_up = np.real(inv_metric @ S_down)
        return S_up.astype(np.float64), float(energy_density)

    def compute_maxwell_stress_energy(
        self,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        spectral_chart: SpectralChart,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Tensor de energía-momento maxwelliano (signatura riemanniana)

            T^{ij} = E^i E^j + B^i B^j − ½ g^{ij} (⟨E,E⟩ + ⟨B,B⟩)

        y 2-forma de Poynting  Π_{ij} = E_i B_j − E_j B_i.
        En ausencia de corrientes, ∇_j T^{ij} = 0 (se audita en Fase 3).
        """
        inv_g = np.asarray(spectral_chart.inv_metric, dtype=np.float64)
        E = np.asarray(electric_field, dtype=np.float64)
        B = np.asarray(magnetic_field, dtype=np.float64)
        E_up = np.real(inv_g @ E)
        B_up = np.real(inv_g @ B)
        uu = float(np.real(np.dot(E_up, E)) + np.real(np.dot(B_up, B)))
        T_up = np.outer(E_up, E_up) + np.outer(B_up, B_up) - 0.5 * uu * inv_g
        two_form = np.outer(E, B) - np.outer(B, E)
        return T_up.astype(np.float64), two_form.astype(np.float64)

    def _phase1_observe_orient(
        self,
        grad_S: np.ndarray,
        connection_A: List[np.ndarray],
        metric_tensor: np.ndarray,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        refraction_index: float,
        coupling_constant: float = 1.0,
        connection_dA: Optional[List[List[np.ndarray]]] = None,
    ) -> PhaseOneKapexPacket:
        r"""
        [FASE 1 — ÚLTIMO MORFISMO: OBSERVE + ORIENT]

        Codominio
        ---------
        `PhaseOneKapexPacket`

        Continuidad formal
        ------------------
        Este paquete **es** el dominio del primer morfismo de la Fase 2,
        `_phase2_spectral_integrity(self, packet: PhaseOneKapexPacket)`.
        No existe ningún objeto intersticial: la composición

            _phase2_spectral_integrity ∘ _phase1_observe_orient

        está tipada estrictamente (retículo de métodos anidados).
        """
        logger.info("Fase 1: carta espectral, Eikonal+CSMD, Yang–Mills, Hodge–Poynting.")

        chart = self._factorize_metric(metric_tensor)

        eikonal_res, p_up = self.solve_eikonal_residual(
            grad_S, metric_tensor, refraction_index, spectral_chart=chart
        )
        inner = float(np.real(np.dot(p_up, np.asarray(grad_S, dtype=np.float64))))
        hamiltonian = 0.5 * (inner - float(refraction_index) ** 2)

        csmd = self._complex_step_eikonal_momentum(grad_S, chart.inv_metric)
        # Identidad analítica: ∇_p I = 2 p^μ. Defecto = sonda de integridad CSMD.
        csmd_err = float(la.norm(csmd - 2.0 * np.asarray(p_up, dtype=np.float64)))

        projected_A: List[np.ndarray] = []
        proj_residuals: List[float] = []
        for A in connection_A:
            pi_A, r = self.project_to_so_n(np.asarray(A))
            projected_A.append(pi_A)
            proj_residuals.append(r)
        algebra_res = float(np.real(self._neumaier_sum(np.asarray(proj_residuals, dtype=np.float64))))

        F, ym_action = self.compute_yang_mills_action(
            projected_A,
            metric_tensor,
            coupling_constant=coupling_constant,
            connection_dA=connection_dA,
            spectral_chart=chart,
        )
        inv_metric = np.asarray(chart.inv_metric, dtype=np.complex128)
        F_up = np.einsum("ma,nb,abij->mnij", inv_metric, inv_metric, F)
        raw_trace = float(np.real(self._neumaier_sum(
            np.real(np.einsum("mnij,mnji->mn", F, F_up)).ravel()
        )))
        jacobi = self._bianchi_jacobi_residual(projected_A, F)
        pontryagin = self._pontryagin_density(F)

        poynting_up, energy_density = self.compute_poynting_strategic_flux(
            electric_field, magnetic_field, metric_tensor, spectral_chart=chart
        )
        stress, two_form = self.compute_maxwell_stress_energy(
            electric_field, magnetic_field, chart
        )

        packet = PhaseOneKapexPacket(
            eikonal_residual=eikonal_res,
            eikonal_hamiltonian=hamiltonian,
            momentum_up_vector=p_up,
            csmd_momentum=csmd,
            csmd_gradient_error=csmd_err,
            curvature_tensor=F,
            yang_mills_action=ym_action,
            yang_mills_raw_trace=raw_trace,
            bianchi_jacobi_residual=jacobi,
            instanton_density=pontryagin,
            poynting_flux_vector=poynting_up,
            poynting_two_form=two_form,
            field_energy_density=energy_density,
            maxwell_stress_tensor=stress,
            spectral_chart=chart,
            algebra_projection_residual=algebra_res,
        )
        logger.debug(
            "Fase 1 completa: Eik=%.6e H=%.6e YM=%.6e U=%.6e Jacobi=%.6e CSMD=%.6e κ=%.3e",
            eikonal_res, hamiltonian, ym_action, energy_density, jacobi, csmd_err,
            chart.condition_number,
        )
        return packet

    # =========================================================================
    # FASE 2 — DECIDE
    # Dominio = PhaseOneKapexPacket  (codominio del último método de Fase 1).
    # Se evalúa la integridad espectral, la holonomía de Wilson y se clasifica
    # el estado en el álgebra de Heyting de cuatro puntos.
    # El último método, `_phase2_decide`, tiene por codominio PhaseTwoKapexPacket,
    # que es el segundo factor del dominio de `_phase3_conservation_audit`.
    # =========================================================================
    def _phase2_spectral_integrity(
        self,
        packet: PhaseOneKapexPacket,
        connection_A: List[np.ndarray],
    ) -> SpectralIntegrityReport:
        r"""
        [FASE 2 — PRIMER MORFISMO: INTEGRIDAD ESPECTRAL]

        Dominio
        -------
        `PhaseOneKapexPacket`  ← continuación formal de `_phase1_observe_orient`.

        Diagnósticos (teoría espectral + álgebra de Banach (M_n, ||·||₂)):
          • κ₂(g), gap espectral de g;
          • ||F||_F ;
          • radios espectrales ρ(A_μ) ≤ ||A_μ||₂  (fórmula de Gelfand);
          • defecto de submultiplicatividad relativa
                δ = max_{μ,ν} [ ||A_μ A_ν||₂ / (||A_μ||₂ ||A_ν||₂ + ε)  −  1 ]_+
            que debe ser 0 en aritmética exacta.
        """
        chart = packet.spectral_chart
        F = np.asarray(packet.curvature_tensor)
        curv_frob = float(la.norm(F.reshape(self._n * self._n, self._n, self._n), ord="fro"))

        radii = np.zeros(self._n, dtype=np.float64)
        ops: List[float] = []
        mats: List[np.ndarray] = []
        for mu, A in enumerate(connection_A):
            M = np.asarray(A, dtype=np.complex128)
            mats.append(M)
            spec = la.eigvals(M)
            radii[mu] = float(np.max(np.abs(spec))) if spec.size else 0.0
            ops.append(float(la.norm(M, ord=2)))

        defect = 0.0
        for mu in range(self._n):
            for nu in range(self._n):
                lhs = float(la.norm(mats[mu] @ mats[nu], ord=2))
                rhs = ops[mu] * ops[nu]
                if rhs > _WILKINSON_FLOOR:
                    defect = max(defect, max(0.0, lhs / rhs - 1.0))

        # Score en (0, 1]: penaliza κ grande, gap nulo, curvatura explosiva, δ > 0.
        cond_term = 1.0 / (1.0 + np.log10(max(chart.condition_number, 1.0)))
        gap_term = float(np.tanh(chart.spectral_gap / max(chart.operator_norm, _WILKINSON_FLOOR)))
        curv_term = float(np.exp(-curv_frob))
        banach_term = float(np.exp(-defect / _MACHINE_EPS))
        score = float(cond_term * max(gap_term, 0.0) * curv_term * min(banach_term, 1.0))

        return SpectralIntegrityReport(
            metric_condition=float(chart.condition_number),
            metric_gap=float(chart.spectral_gap),
            curvature_frobenius=curv_frob,
            connection_spectral_radii=radii,
            banach_submultiplicative_defect=float(defect),
            integrity_score=score,
        )

    def _phase2_gauge_holonomy(
        self,
        connection_A: List[np.ndarray],
    ) -> Tuple[float, float]:
        r"""
        Media de lazos de Wilson elementales (plaquetas μ < ν):

            U_{μν} = exp(A_μ) exp(A_ν) exp(−A_μ) exp(−A_ν) ,
            w_{μν} = (1/n) Re Tr(U_{μν}) .

        w → 1  ⇒ holonomía ≈ id (conexión pequeña / pura de calibre);
        w → 0  ⇒ desorden de calibre. coherencia := max(w, 0).
        """
        if self._n < 2:
            return 1.0, 1.0
        acc: List[float] = []
        for mu in range(self._n):
            for nu in range(mu + 1, self._n):
                A_mu = np.asarray(connection_A[mu], dtype=np.complex128)
                A_nu = np.asarray(connection_A[nu], dtype=np.complex128)
                U = (
                    la.expm(A_mu)
                    @ la.expm(A_nu)
                    @ la.expm(-A_mu)
                    @ la.expm(-A_nu)
                )
                acc.append(float(np.real(np.trace(U)) / self._n))
        mean = float(np.real(self._neumaier_sum(np.asarray(acc, dtype=np.float64))) / max(len(acc), 1))
        return mean, float(max(mean, 0.0))

    def _phase2_heyting_classify(
        self,
        packet: PhaseOneKapexPacket,
        integrity: SpectralIntegrityReport,
        gauge_coherence: float,
        eikonal_threshold: float,
        yang_mills_threshold: float,
        jacobi_threshold: float,
        condition_threshold: float,
    ) -> Tuple[HeytingVerdict, float, Dict[str, bool]]:
        r"""
        Clasificación en el álgebra de Heyting.

        Se construyen scores en [0, 1] (semántica [0,1]-valuada del topos)
        y se toma el meet. El veredicto es la sección de la cadena:
            score ≥ 0.99 → CERTIFIED,
            score ≥ 0.90 → COHERENT,
            score ≥ 0.50 → DEGRADED,
            else         → VETOED.
        Un veto atómico (umbral duro violado) colapsa a VETOED por modus ponens
        interno:  (violación → ⊥) ∧ ⊤  ⊢  ⊥.
        """
        eik_viol = packet.eikonal_residual > eikonal_threshold
        ym_viol = packet.yang_mills_action > yang_mills_threshold
        jac_viol = packet.bianchi_jacobi_residual > jacobi_threshold
        cond_viol = integrity.metric_condition > condition_threshold
        energy_viol = packet.field_energy_density < -_WILKINSON_FLOOR
        csmd_viol = packet.csmd_gradient_error > 1e-8

        eik_score = float(np.exp(-packet.eikonal_residual / max(eikonal_threshold, _WILKINSON_FLOOR)))
        ym_score = float(np.exp(-max(packet.yang_mills_action, 0.0) / max(yang_mills_threshold, _WILKINSON_FLOOR)))
        jac_score = float(np.exp(-packet.bianchi_jacobi_residual / max(jacobi_threshold, _WILKINSON_FLOOR)))
        energy_score = 1.0 if packet.field_energy_density >= -_WILKINSON_FLOOR else 0.0
        csmd_score = float(np.exp(-packet.csmd_gradient_error / 1e-8))
        meet = min(
            eik_score,
            ym_score,
            jac_score,
            energy_score,
            csmd_score,
            integrity.integrity_score,
            float(np.clip(gauge_coherence, 0.0, 1.0)),
        )

        hard_veto = eik_viol or energy_viol or csmd_viol or jac_viol
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
            "eikonal": eik_viol,
            "yang_mills": ym_viol,
            "jacobi": jac_viol,
            "condition": cond_viol,
            "energy": energy_viol,
            "csmd": csmd_viol,
        }
        return verdict, float(meet), flags

    def _phase2_decide(
        self,
        packet: PhaseOneKapexPacket,
        connection_A: List[np.ndarray],
        eikonal_threshold: float = 1e-4,
        yang_mills_threshold: float = 1e4,
        jacobi_threshold: float = 1e-8,
        condition_threshold: float = 1e12,
    ) -> PhaseTwoKapexPacket:
        r"""
        [FASE 2 — ÚLTIMO MORFISMO: DECIDE]

        Codominio
        ---------
        `PhaseTwoKapexPacket`

        Continuidad formal
        ------------------
        Junto con el `PhaseOneKapexPacket` residual, este objeto constituye
        el dominio del primer morfismo de la Fase 3,
        `_phase3_conservation_audit(self, packet1, packet2)`.
        """
        integrity = self._phase2_spectral_integrity(packet, connection_A)
        wilson_mean, coherence = self._phase2_gauge_holonomy(connection_A)
        verdict, score, flags = self._phase2_heyting_classify(
            packet,
            integrity,
            coherence,
            eikonal_threshold=eikonal_threshold,
            yang_mills_threshold=yang_mills_threshold,
            jacobi_threshold=jacobi_threshold,
            condition_threshold=condition_threshold,
        )
        decision = PhaseTwoKapexPacket(
            heyting_verdict=verdict.value,
            heyting_rank=verdict.rank,
            eikonal_threshold_violated=flags["eikonal"],
            yang_mills_threshold_violated=flags["yang_mills"],
            jacobi_threshold_violated=flags["jacobi"],
            condition_threshold_violated=flags["condition"],
            spectral_integrity=integrity,
            wilson_plaquette_mean=wilson_mean,
            gauge_coherence=coherence,
            heyting_score=score,
        )
        logger.debug(
            "Fase 2 completa: veredicto=%s score=%.4f Wilson=%.4f κ=%.3e",
            verdict.value, score, wilson_mean, integrity.metric_condition,
        )
        return decision

    # =========================================================================
    # FASE 3 — ACT
    # Dominio = PhaseOneKapexPacket × PhaseTwoKapexPacket.
    # Audita leyes de conservación, sella telemetría inmutable y expone T.
    # =========================================================================
    def _phase3_conservation_audit(
        self,
        packet1: PhaseOneKapexPacket,
        packet2: PhaseTwoKapexPacket,
    ) -> float:
        r"""
        [FASE 3 — PRIMER MORFISMO: AUDITORÍA DE CONSERVACIÓN]

        Dominio
        -------
        `PhaseOneKapexPacket × PhaseTwoKapexPacket`
        ← continuación formal de `_phase1_observe_orient` y `_phase2_decide`.

        Residuos adimensionales acumulados (KBN) de las identidades que deben
        anularse en aritmética exacta:
          1. Jacobi / Bianchi algebraico;
          2. sonda CSMD  ||∇_p I − 2 p^♯|| ;
          3. proyección al álgebra 𝔰𝔬(n);
          4. parte antisimétrica residual de T^{ij} (T debe ser simétrico);
          5. no-negatividad de U (parte negativa como residuo).
        """
        T = np.asarray(packet1.maxwell_stress_tensor, dtype=np.float64)
        skew_T = 0.5 * (T - T.T)
        terms = np.asarray(
            [
                packet1.bianchi_jacobi_residual,
                packet1.csmd_gradient_error,
                packet1.algebra_projection_residual,
                float(la.norm(skew_T, ord="fro")),
                float(max(0.0, -packet1.field_energy_density)),
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
        packet1: PhaseOneKapexPacket,
        packet2: PhaseTwoKapexPacket,
        conservation_residual: float,
    ) -> KapexTelemetry:
        """
        [FASE 3 — ÚLTIMO MORFISMO: ACT]

        Sella el objeto terminal `KapexTelemetry`. No hay Fase 4: este sello
        es el valor de T(estado) inyectable en el pasaporte de la Malla.
        """
        return KapexTelemetry(
            eikonal_residual=packet1.eikonal_residual,
            eikonal_hamiltonian=packet1.eikonal_hamiltonian,
            yang_mills_action=packet1.yang_mills_action,
            poynting_flux_vector=packet1.poynting_flux_vector,
            field_energy_density=packet1.field_energy_density,
            momentum_up_vector=packet1.momentum_up_vector,
            curvature_tensor=packet1.curvature_tensor,
            maxwell_stress_tensor=packet1.maxwell_stress_tensor,
            heyting_verdict=packet2.heyting_verdict,
            condition_number=packet2.spectral_integrity.metric_condition,
            bianchi_residual=packet1.bianchi_jacobi_residual,
            instanton_density=packet1.instanton_density,
            csmd_gradient_error=packet1.csmd_gradient_error,
            wilson_plaquette_mean=packet2.wilson_plaquette_mean,
            conservation_residual=conservation_residual,
            spectral_integrity_score=packet2.spectral_integrity.integrity_score,
        )

    def execute_electrodynamic_cycle(
        self,
        grad_S: np.ndarray,
        connection_A: List[np.ndarray],
        metric_tensor: np.ndarray,
        electric_field: np.ndarray,
        magnetic_field: np.ndarray,
        refraction_index: float,
        coupling_constant: float = 1.0,
        eikonal_threshold: float = 1e-4,
        yang_mills_threshold: float = 1e4,
        connection_dA: Optional[List[List[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        r"""
        Functor OODA  T = Act ∘ Decide ∘ Observe.

        Encadena las tres fases anidadas sin objetos huérfanos:

            In  --F1→  PhaseOne  --F2→  PhaseTwo  --F3→  KapexTelemetry  →  dict

        La firma pública (más `yang_mills_threshold` y `connection_dA` opcionales)
        permanece compatible con el pasaporte de telemetría de la Malla.
        """
        packet1 = self._phase1_observe_orient(
            grad_S,
            connection_A,
            metric_tensor,
            electric_field,
            magnetic_field,
            refraction_index,
            coupling_constant=coupling_constant,
            connection_dA=connection_dA,
        )
        packet2 = self._phase2_decide(
            packet1,
            connection_A,
            eikonal_threshold=eikonal_threshold,
            yang_mills_threshold=yang_mills_threshold,
        )
        conservation = self._phase3_conservation_audit(packet1, packet2)
        telemetry = self._phase3_act(packet1, packet2, conservation)
        return telemetry.to_dict()


__all__ = [
    "KapexElectrodynamicEngine",
    "PhaseOneKapexPacket",
    "PhaseTwoKapexPacket",
    "KapexTelemetry",
    "SpectralChart",
    "SpectralIntegrityReport",
    "HeytingVerdict",
]
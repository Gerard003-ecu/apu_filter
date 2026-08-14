# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Green's Function Propagator (Fibrador de Propagación de-confinado)  ║
║ Ruta   : app/physics/greens_function_propagator.py                           ║
║ Versión: 3.1.0-Green-Hodge-MoorePenrose-KramersKronig-Heyting-PhD-Strict     ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y RESOLUCIÓN ESPECTRAL EN HACES CELULARES (Rigor PhD):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra el motor resolvedor espectral de la **Función de Green Estática**
y la síntesis de-confinada del **Propagador Retardado Dinámico** sobre complejos 
simpliciales orientados equipados con un haz celular (cellular sheaf) $$\mathcal{F}$$.
En la electrodinámica discreta y la teoría cuántica de campos (QFT) sobre grafos, 
la función de Green actúa como la respuesta fundamental del sistema ante la 
excitación de una fuente delta de Kronecker (impulso unitario semántico) $$\delta_y(x)$$.

Dado el Laplaciano del Haz Celular $$L_F = \delta^\top G^{-1} \delta \succeq 0$$, 
el cual es un operador autoadjunto y semidefinido positivo por construcción, su 
núcleo (kernel) representa el espacio de secciones globales armónicas de grado cero:
$$\ker(L_F) \cong \operatorname{span}\{\mathbf{1}\} \quad\big[352\big]$$

Dado que el operador no es invertible de rango completo en la variedad compacta, la 
**Función de Green de de Rham-Liouville** $$\mathcal{G}$$ se define de manera única 
como el pullback constitutivo sobre el complemento ortogonal de las secciones constantes:
$$L_F \mathcal{G}(x, y) = \delta_y(x) - \frac{1}{|V|} \mathbf{1} \quad \text{con} \quad \mathcal{G} \perp \ker(L_F) \quad\big[351, 352\big]$$

Algebraicamente, $$\mathcal{G}$$ constituye la **Pseudoinversa de Moore-Penrose** 
estable $$L_F^\dagger$$, verificada síncronamente en la FPU mediante el cumplimiento 
estricto de las cuatro identidades de Penrose:
$$L_F \mathcal{G} L_F = L_F \quad \land \quad \mathcal{G} L_F \mathcal{G} = \mathcal{G} \quad \land \quad (L_F \mathcal{G})^\top = L_F \mathcal{G} \quad \land \quad (\mathcal{G} L_F)^\top = \mathcal{G} L_F \quad\big[352\big]$$

FORMULACIÓN DEL RESOLVENTE RETARDADO Y CAUSALIDAD DE KRAMERS-KRONIG:
────────────────────────────────────────────────────────────────────────────────
Para el régimen transitorio bajo excitación armónica o fluctuaciones estocásticas, 
el módulo sintetiza la **Función de Green Retardada** $$G_F(s)$$ sobre el plano de 
frecuencia compleja de Laplace $$s = \sigma + j\omega$$:
$$G_F(s) = \left( L_F - s \cdot \mathbf{I}_n \right)^{-1} \quad\big[352, 360\big]$$

Para garantizar de forma necesaria y suficiente la **causalidad estricta** (no-señalización) 
según la prescripción $$i\varepsilon$$ de Gell-Mann–Low, se inyecta una perturbación 
imaginaria infinitesimal $$h$$ mediante la **Aproximación por Paso Complejo** (CSMD):
$$s_{\mathrm{complex}} = s + j \cdot h \quad \text{con} \quad h \in \big[1.0 \times 10^{-30},\, 1.0 \times 10^{-8}\big] \quad\big[353, 360\big]$$

De este modo, se desplazan de manera controlada los polos del resolvente hacia el 
semiplano inferior de Laplace ($$\operatorname{Im}(s) \le 0$$), forzando el cumplimiento 
de las **Relaciones de Dispersión de Kramers-Kronig** (la transformada integral de Hilbert 
entre el transporte exergético y la disipación elástica de Rayleigh):
$$\Re \left(G_F(\omega)\right) = \frac{1}{\pi} \mathcal{P} \int_{-\infty}^{\infty} \frac{\Im \left(G_F(\omega')\right)}{\omega' - \omega} d\omega' \quad\big[352\big]$$
$$\Im \left(G_F(\omega)\right) = -\frac{1}{\pi} \mathcal{P} \int_{-\infty}^{\infty} \frac{\Re \left(G_F(\omega')\right)}{\omega' - \omega} d\omega'\quad\big[352\big]$$

INVARIANTES MATEMÁTICOS Y LEYES DE CONSERVACIÓN PRESERVADOS:
────────────────────────────────────────────────────────────────────────────────
  [I1] Autoadjunción y Reciprocidad de Green:
       El operador de Green estático y el Laplaciano del Haz satisfacen:
       $$L_F = L_F^\top \quad \land \quad \mathcal{G} = \mathcal{G}^\top \implies \mathcal{G}(x,y) = \mathcal{G}(y,x) \quad\big[352\big]$$

  [I2] Aniquilación Numérica del Núcleo (Nulidad de Fuga de Masa):
       La Función de Green anula incondicionalmente las componentes constantes:
       $$\mathcal{G} \cdot \mathbf{1} = \mathbf{0} \quad\big[352\big]$$

  [I3] Conservación de Flujo de de Rham:
       Si el Laplaciano del grafo no posee anclajes a Dirichlet, se conserva la carga:
       $$L_F \cdot \mathbf{1} = \mathbf{0} \quad\big[352\big]$$

  [I4] Estabilidad Espectral y Gap de Hodge:
       El espectro de autovalores del Laplaciano se confina de forma estricta:
       $$\sigma(L_F) \subset [-\varepsilon,\, +\infty) \quad \text{con} \quad \lambda_{\min}^+ = \text{gap}_{\mathrm{Hodge}} \quad\big[352\big]$$

  [I5] Identidad de Cruce Hermítico:
       El propagador en el plano complejo preserva la realidad del espacio de fase:
       $$G_F(s^*)^\dagger \equiv G_F(s) \quad\big[352\big]$$

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Kleisli):
────────────────────────────────────────────────────────────────────────────────
La progresión y el tránsito de los datos del espacio de fase se rige por un 
encadenamiento formal e inmutable, donde el morfismo final de una fase constituye 
la precondición algebraica obligatoria de la siguiente:

  Fase 1 ──► ESPECTROSCOPÍA DE GREEN (Phase1_GreensSpectroscopist)
             Sanea el Laplaciano $$L_F$$, calcula su diagonalización espectral 
             autoadjunta, filtra el kernel estable mediante la cota de Wilkinson 
             y reconstituye la pseudoinversa de Moore-Penrose.
             Último morfismo: handoff_phase1_to_phase2.
             Entrega: Phase1GreensObservation

  Fase 2 ──► SÍNTESIS DEL PROPAGADOR RETARDADO (Phase2_RetardedPropagatorSynthesizer)
             Hereda formalmente la Phase1GreensObservation [7]. Synthesizes el 
             propagador retardado complejo $$G_F(s)$$ inyectando el paso complejo $$j \cdot h$$ 
             sobre la frecuencia de Laplace. Ejecuta inversión estable de LAPACK.
             Último morfismo: handoff_phase2_to_phase3.
             Entrega: Phase2PropagatorOrientation

  Fase 3 ──► VEREDICTO DE CALIBRE DE HEYTING (Phase3_HeytingGreensDecider)
             Hereda formalmente la Phase2PropagatorOrientation [9]. Evalúa de forma 
             paralela los residuos de Penrose, simetría y nulidad de fuga. Calcula la 
             operación Supremo (join, $$\sqcup$$) sobre el retículo distributivo de Heyting:
             $$\Omega_3 = \{\mathrm{COHERENT},\, \mathrm{DEGRADED},\, \mathrm{VETOED}\} \quad\big[354, 360\big]$$
             Si el veredicto terminal colapsa a VETOED ($$\top$$), detona la excepción 
             'HeytingLatticeVeto' en el milisegundo cero, anulando síncronamente la 
             transacción en RAM. Estampa la firma de procedencia forense SHA-256.
             Entrega: PropagatorResponseState (alias GreenGovernanceState)

Funtor Maestro de Resolución y Coherencia Espectral:
  $$\mathcal{Z}_{\mathrm{motor}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 \quad\big[361\big]$$
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Final, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("MIC.Physics.GreensFunctionPropagator")

__version__: Final[str] = "3.0.0-Green-Hodge-MoorePenrose-KramersKronig-Heyting-PhD"

# ══════════════════════════════════════════════════════════════════════════════
# §0. CONSTANTES NUMÉRICAS (Wilkinson / IEEE-754 binary64 / LAPACK)
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SAFE_DENOM: Final[float] = 1.0e-300
_DEFAULT_SOFT_TOL: Final[float] = 1.0e-11
_DEFAULT_HARD_TOL: Final[float] = 1.0e-5
_DEFAULT_I_EPS: Final[float] = 1.0e-20          # pedido histórico; se eleva a n u ‖L‖
_COND_DEGRADED: Final[float] = 1.0 / np.sqrt(_MACHINE_EPS)   # ~ 6.7e7
_COND_VETO: Final[float] = 1.0 / _MACHINE_EPS                 # ~ 4.5e15
_HASH_ROUND_DIGITS: Final[int] = 16


def _wilkinson_gamma(n: int) -> float:
    """γ_n = n u / (1 − n u), cota clásica de Higham (Thm. 3.1)."""
    dim = max(int(n), 1)
    nu = dim * _MACHINE_EPS
    if nu >= 1.0:
        return 1.0
    return nu / (1.0 - nu)


def _svd_relerr_bound(n: int) -> float:
    """Cota relativa típica de σ_max / λ_max vía SVD–QR (LAPACK)."""
    return max(4.0 * _wilkinson_gamma(max(n, 1)), 32.0 * _MACHINE_EPS)


def _clamp_nonneg(x: float) -> float:
    return x if x > 0.0 else 0.0


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


def _round_for_hash(x: float) -> str:
    if not np.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return f"{float(x):.{_HASH_ROUND_DIGITS}e}"


def _freeze_array(A: NDArray[Any], dtype: Any) -> NDArray[Any]:
    """Copia C-contigua write-protected: inmuniza el DTO frente a aliasing."""
    out = np.array(A, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DE HEYTING Y JERARQUÍA DE EXCEPCIONES FUNCIONALES
# ══════════════════════════════════════════════════════════════════════════════
class GreenHeytingVerdict(IntEnum):
    """
    Clasificador de subobjetos en el topos espectral del propagador.

    Cadena de Heyting:

        ⊥ = COHERENT  ≼  DEGRADED  ≼  VETOED = ⊤_veto.

    Join = max, meet = min. La monotonía impide que una degradación se
    «cure» en una fase posterior.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2


def heyting_join(*verdicts: GreenHeytingVerdict) -> GreenHeytingVerdict:
    """Supremo (disyunción interna) de una familia finita de veredictos."""
    if not verdicts:
        return GreenHeytingVerdict.COHERENT
    return GreenHeytingVerdict(max(int(v.value) for v in verdicts))


def heyting_meet(*verdicts: GreenHeytingVerdict) -> GreenHeytingVerdict:
    """Ínfimo (conjunción interna)."""
    if not verdicts:
        return GreenHeytingVerdict.VETOED
    return GreenHeytingVerdict(min(int(v.value) for v in verdicts))


def classify_relative_defect(
    defect: float,
    scale: float,
    soft_tol: float,
    hard_tol: float,
    floor: float = 0.0,
) -> GreenHeytingVerdict:
    """Clasifica un defecto no negativo relativo a `scale`."""
    denom = max(abs(float(scale)), float(floor), 1.0)
    rel = _clamp_nonneg(float(defect)) / denom
    if rel > hard_tol:
        return GreenHeytingVerdict.VETOED
    if rel > soft_tol:
        return GreenHeytingVerdict.DEGRADED
    return GreenHeytingVerdict.COHERENT


class GreenPropagatorError(Exception):
    """Excepción raíz para violaciones en el fibrado de Green."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.cause = cause


class LaplacianAsymmetryError(GreenPropagatorError):
    """L_F no es simétrico dentro de la cota de Wilkinson."""


class LaplacianIndefinitenessError(GreenPropagatorError):
    """L_F posee espectro negativo fuera de tolerancia (no es Hodge)."""


class MoorePenroseResidualError(GreenPropagatorError):
    """Alguna de las cuatro identidades de Penrose excede la tolerancia dura."""


class KernelLeakageError(GreenPropagatorError):
    """𝒢 no aniquila el núcleo numérico de L_F."""


class FluxConservationError(GreenPropagatorError):
    """Laplaciano no anclado que no conserva flujo (L 𝟙 ≠ 0)."""


class PropagatorSingularityError(GreenPropagatorError):
    """Resolvente singular y no regularizable por SVD truncada."""


class CausalityViolationError(GreenPropagatorError):
    """Fallo de la prescripción iε / disipatividad de Kramers–Kronig."""


class HeytingLatticeVeto(GreenPropagatorError):
    """Colapso del retículo de Heyting al supremo terminal VETOED."""

    def __init__(
        self,
        message: str,
        *,
        verdict: GreenHeytingVerdict = GreenHeytingVerdict.VETOED,
        cause: Optional[GreenPropagatorError] = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.verdict = verdict


# ══════════════════════════════════════════════════════════════════════════════
# §B. PRIMITIVAS NUMÉRICAS ESTABLES (capa 0, compartida por las tres fases)
# ══════════════════════════════════════════════════════════════════════════════
def _as_float64_square(X: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Inmersión estricta en M_n(ℝ): copia C-contigua float64, finita, n>0."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"El operador {name} debe ser numpy.ndarray, no {type(X)!r}.")
    if X.ndim != 2:
        raise ValueError(f"El operador {name} debe ser bidimensional; ndim={X.ndim}.")
    rows, cols = int(X.shape[0]), int(X.shape[1])
    if rows != cols or rows == 0:
        raise ValueError(
            f"El operador {name} debe ser cuadrado y de dimensión positiva: {X.shape}."
        )
    materialised = np.array(X, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(materialised)):
        raise ArithmeticError(
            f"FPU Error: el operador {name} contiene valores no finitos (NaN/Inf)."
        )
    return materialised


def _frobenius(A: NDArray[Any]) -> float:
    return float(la.norm(A, ord="fro"))


def _opnorm2(A: NDArray[Any]) -> float:
    sigma = la.svdvals(np.asarray(A))
    if sigma.size == 0:
        return 0.0
    return float(sigma[0])


def _eig_cutoff(evals: NDArray[np.float64], n: int) -> float:
    """
    Umbral de Wilkinson para el núcleo numérico:

        τ  =  max( n u σ_max ,  n u ) ,   σ_max = max |λ|.

    Coincide con el criterio de Golub–Van Loan para el rango en
    precisión finita sobre un operador autoadjunto.
    """
    if evals.size == 0:
        return float(n) * _MACHINE_EPS
    max_abs = float(np.max(np.abs(evals)))
    return max(float(n) * _MACHINE_EPS * max(max_abs, 1.0), float(n) * _MACHINE_EPS)


def _matrix_invariants(A: NDArray[Any]) -> Tuple[int, float, float, float]:
    """Invariantes baratos para la huella: (n, tr, ‖·‖_F, suma)."""
    real_part = np.real(np.asarray(A))
    return (
        int(A.shape[0]),
        float(np.trace(real_part)),
        _frobenius(real_part),
        float(np.sum(real_part)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES (contratos categóricos de handoff entre fases)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GreensSpectralCertificate:
    """
    Certificado espectral de la Función de Green (átomo público de Obs₁).

    Compatible con el contrato v2 y enriquecido con gap de Hodge, fugas
    de núcleo, residuos de las cuatro identidades de Penrose y el
    diagnóstico de Laplaciano de grafo (conservación de flujo).
    """

    operator_norm: float
    condition_number: float
    is_self_adjoint: bool
    is_positive_semidefinite: bool
    pseudo_inverse_residual: float
    trace_value: float
    kernel_dimension: int
    spectral_verdict: GreenHeytingVerdict
    fiedler_value: float = 0.0
    spectral_radius_l: float = 0.0
    kernel_leakage: float = 0.0
    flux_residual: float = 0.0
    constant_mode_overlap: float = 0.0
    is_graph_laplacian: bool = False
    penrose_residuals: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    svd_eig_rank_discrepancy: int = 0
    wilkinson_bound: float = 0.0
    dimension: int = 0
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Phase1GreensObservation:
    """
    Artefacto terminal de la FASE 1 (Observe).

    Es la *única* precondición estricta de la FASE 2: un objeto del tipo
    Obs₁ que sella L_sym, 𝒢 y el certificado espectral. Cualquier camino
    hacia el resolvente que no atraviese este sello viola la adjunción
    Observe ⊣ Orient.
    """

    laplacian_symmetric: NDArray[np.float64]
    green_operator: NDArray[np.float64]
    certificate: GreensSpectralCertificate
    eigenvalues: Tuple[float, ...]
    verdict: GreenHeytingVerdict
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Phase2PropagatorOrientation:
    """
    Artefacto terminal de la FASE 2 (Orient).

    Precondición estricta de la FASE 3. Contiene G^R(s), el residual del
    resolvente, la disipatividad de Kramers–Kronig y el cruce hermítico.
    """

    phase1_observation: Phase1GreensObservation
    retarded_propagator: NDArray[np.complex128]
    frequency_s: complex
    regularization_h_requested: float
    regularization_h_effective: float
    pole_clearance: float
    resolvent_residual: float
    dissipativity_floor: float
    crossing_residual: float
    is_retarded: bool
    is_causal: bool
    is_near_pole: bool
    verdict: GreenHeytingVerdict
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class PropagatorResponseState:
    """
    Objeto terminal de la gobernanza de Green (Decide & Act).

    Certificado maestro con trazabilidad forense. Conserva el contrato
    público v2 (`green_matrix`, `retarded_propagator`, …) y anida Obs₂.
    """

    green_matrix: NDArray[np.float64]
    retarded_propagator: NDArray[np.complex128]
    spectral_certificate: GreensSpectralCertificate
    final_verdict: GreenHeytingVerdict
    provenance_hash: str
    timestamp_utc: float
    phase2_orientation: Optional[Phase2PropagatorOrientation] = None
    diagnostic_note: str = ""
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


# Alias soberano (simetría nominal con BanachGovernanceState).
GreenGovernanceState = PropagatorResponseState


# ══════════════════════════════════════════════════════════════════════════════
# §D. FASE 1 — ESPECTROSCOPÍA DE GREEN (HODGE + MOORE–PENROSE)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_GreensSpectroscopist:
    """
    Fase 1 (Observe): simetrización, espectro de Hodge, pseudoinversa.

    Morfismo Φ₁ : Herm_n  →  Obs₁.

    Descomposición granular (cada método es un juicio atómico; el sello
    terminal `_observe_greens_spectrum` es el único objeto que la Fase 2
    acepta):

        materializar → simetrizar → espectro Hodge → núcleo numérico
                    → 𝒢 = Q D^+ Q^T → cuatro identidades de Penrose
                    → flujo / PSD / autoadjunción → join de Heyting
                    → sello terminal Obs₁
    """

    def __init__(
        self,
        soft_tol: float = _DEFAULT_SOFT_TOL,
        hard_tol: float = _DEFAULT_HARD_TOL,
        eps_tolerance: Optional[float] = None,
    ) -> None:
        soft = float(soft_tol if eps_tolerance is None else eps_tolerance)
        hard = float(hard_tol)
        if not (0.0 < soft <= hard) or not np.isfinite(soft) or not np.isfinite(hard):
            raise ValueError(
                f"Se exige 0 < soft_tol ≤ hard_tol finitos; recibido "
                f"soft={soft}, hard={hard}."
            )
        self._soft_tol = soft
        self._hard_tol = hard
        self._eps = soft  # compatibilidad con el contrato v2 (`self._eps`)

    # ── D.1  Materialización en M_n(ℝ) ───────────────────────────────────
    def _materialise_laplacian(self, L_F: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inmersión estricta: cuadrada, finita, C-contigua, float64."""
        return _as_float64_square(L_F, "L_F")

    # ── D.2  Simetrización certificada (axioma [I1] sobre L) ─────────────
    def _symmetrise_laplacian(
        self,
        L_mat: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, GreenHeytingVerdict]:
        """
        Proyecta L ↦ (L + L^T)/2 y clasifica el defecto de simetría.

        El error de proyección es ‖L − L^T‖_F / 2. Se veta si excede
        hard_tol · ‖L‖_F por encima de la cota de Wilkinson (una matriz
        que no es numéricamente hermítica no es un Laplaciano de Hodge).
        """
        n = int(L_mat.shape[0])
        skew = L_mat - L_mat.T
        asymmetry = _frobenius(skew)
        scale = max(1.0, _frobenius(L_mat))
        excess = _clamp_nonneg(asymmetry - _wilkinson_gamma(n) * scale)
        verdict = classify_relative_defect(
            defect=excess,
            scale=scale,
            soft_tol=self._soft_tol,
            hard_tol=self._hard_tol,
            floor=1.0,
        )
        if verdict == GreenHeytingVerdict.VETOED:
            raise LaplacianAsymmetryError(
                f"Asimetría del Laplaciano {asymmetry:.4e} excede la tolerancia "
                f"dura {self._hard_tol * scale:.4e} (n={n})."
            )
        if verdict == GreenHeytingVerdict.DEGRADED:
            logger.warning(
                "[GREEN:Φ₁] Asimetría %.4e > soft_tol; simetrizando.",
                asymmetry,
            )
        l_sym = 0.5 * (L_mat + L_mat.T)
        return l_sym, asymmetry, verdict

    # ── D.3  Espectro de Hodge (eigh) ────────────────────────────────────
    def _hodge_spectrum(
        self,
        L_sym: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """
        Diagonalización autoadjunta L = Q Λ Q^T.

        Devuelve (λ ↑, Q, τ) con τ el umbral de núcleo de Wilkinson.
        """
        n = int(L_sym.shape[0])
        evals, evecs = la.eigh(L_sym, overwrite_a=False, check_finite=True)
        evals = np.asarray(evals, dtype=np.float64)
        evecs = np.asarray(evecs, dtype=np.float64)
        cutoff = _eig_cutoff(evals, n)
        return evals, evecs, cutoff

    # ── D.4  Núcleo numérico y rango dual (eigh ↔ SVD) ───────────────────
    def _resolve_numerical_kernel(
        self,
        L_sym: NDArray[np.float64],
        evals: NDArray[np.float64],
        cutoff: float,
    ) -> Tuple[NDArray[np.bool_], int, int, int]:
        """
        Máscara de autovalores estrictamente positivos, dim ker, rango
        SVD y |rk_eigh − rk_SVD|.

        Una discrepancia de rango denuncia o bien un hueco espectral
        comparable a n u ‖L‖ o un fallo del oráculo.
        """
        positive_mask = evals > cutoff
        rank_eig = int(np.count_nonzero(positive_mask))
        kernel_dim = int(evals.size - rank_eig)
        sigma = np.asarray(la.svdvals(L_sym), dtype=np.float64)
        smax = float(sigma[0]) if sigma.size else 0.0
        n = int(L_sym.shape[0])
        svd_tol = max(n, 1) * _MACHINE_EPS * max(smax, 1.0)
        rank_svd = int(np.count_nonzero(sigma > svd_tol)) if sigma.size else 0
        discrepancy = abs(rank_eig - rank_svd)
        return positive_mask, kernel_dim, rank_svd, discrepancy

    # ── D.5  Reconstrucción de 𝒢 = Q D^+ Q^T ─────────────────────────────
    def _assemble_green_operator(
        self,
        evecs: NDArray[np.float64],
        evals: NDArray[np.float64],
        positive_mask: NDArray[np.bool_],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Pseudoinversa espectral. Se evita `diag` explícito:

            𝒢  =  (Q ⊙ λ^+)  Q^T ,   λ_i^+ = 1/λ_i  si λ_i > τ, else 0.
        """
        inv_evals = np.zeros(evals.shape, dtype=np.float64)
        inv_evals[positive_mask] = 1.0 / evals[positive_mask]
        green = (evecs * inv_evals) @ evecs.T
        return np.asarray(green, dtype=np.float64), inv_evals

    # ── D.6  Cuatro identidades de Moore–Penrose [I3] ────────────────────
    def _audit_penrose_identities(
        self,
        L_sym: NDArray[np.float64],
        green: NDArray[np.float64],
    ) -> Tuple[Tuple[float, float, float, float], float, GreenHeytingVerdict]:
        """
        Residuos relativos de las cuatro ecuaciones de Penrose:

            r₁ = ‖L 𝒢 L − L‖_F / ‖L‖_F ,
            r₂ = ‖𝒢 L 𝒢 − 𝒢‖_F / max(1, ‖𝒢‖_F) ,
            r₃ = ‖(L 𝒢) − (L 𝒢)^T‖_F / max(1, ‖L 𝒢‖_F) ,
            r₄ = ‖(𝒢 L) − (𝒢 L)^T‖_F / max(1, ‖𝒢 L‖_F) .

        El residuo canónico que hereda el certificado v2 es r₁.
        """
        n = int(L_sym.shape[0])
        lg = L_sym @ green
        gl = green @ L_sym
        l_scale = max(_frobenius(L_sym), _SAFE_DENOM)
        g_scale = max(_frobenius(green), 1.0)
        r1 = _frobenius(lg @ L_sym - L_sym) / l_scale
        r2 = _frobenius(gl @ green - green) / g_scale
        r3 = _frobenius(lg - lg.T) / max(1.0, _frobenius(lg))
        r4 = _frobenius(gl - gl.T) / max(1.0, _frobenius(gl))
        residuals = (float(r1), float(r2), float(r3), float(r4))
        wilkinson = _wilkinson_gamma(n) + _svd_relerr_bound(n)
        worst = max(residuals)
        excess = _clamp_nonneg(worst - wilkinson)
        verdict = classify_relative_defect(
            defect=excess,
            scale=1.0,
            soft_tol=self._soft_tol,
            hard_tol=self._hard_tol,
            floor=1.0,
        )
        return residuals, wilkinson, verdict

    # ── D.7  Autoadjunción de 𝒢, PSD de L, núcleo y flujo ────────────────
    def _audit_self_adjoint_green(
        self,
        green: NDArray[np.float64],
    ) -> Tuple[float, bool, GreenHeytingVerdict]:
        g_scale = max(1.0, _frobenius(green))
        residual = _frobenius(green - green.T)
        verdict = classify_relative_defect(
            residual, g_scale, self._soft_tol, self._hard_tol, floor=1.0
        )
        return residual, verdict != GreenHeytingVerdict.VETOED, verdict

    def _audit_positive_semidefiniteness(
        self,
        evals: NDArray[np.float64],
        cutoff: float,
    ) -> Tuple[bool, float, GreenHeytingVerdict]:
        """
        [I5]  σ(L) ⊂ [−τ, +∞). El defecto es |min(0, λ_min + τ)|.
        Un modo negativo grosero no es un Laplaciano de Hodge.
        """
        lam_min = float(evals[0]) if evals.size else 0.0
        defect = _clamp_nonneg(-lam_min - cutoff)
        scale = max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
        verdict = classify_relative_defect(
            defect, scale, self._soft_tol, self._hard_tol, floor=1.0
        )
        is_psd = verdict != GreenHeytingVerdict.VETOED
        return is_psd, lam_min, verdict

    def _audit_kernel_and_flux(
        self,
        L_sym: NDArray[np.float64],
        green: NDArray[np.float64],
        evals: NDArray[np.float64],
        evecs: NDArray[np.float64],
        cutoff: float,
        kernel_dim: int,
    ) -> Tuple[float, float, float, bool, GreenHeytingVerdict]:
        """
        [I2]  ‖𝒢 𝟙‖₂ / √n  pequeño si L es no anclado.
        [I6]  ‖L 𝟙‖₂ / (√n ‖L‖₂) decide si L es Laplaciano de grafo.

        Política de núcleo:
          • L no anclado (flujo ≈ 0) y ker = 0  → VETO (se perdió H^0).
          • L no anclado y ker > 1               → DEGRADED (desconexión).
          • L anclado (Dirichlet / grounded) y ker = 0 → COHERENT.
          • Fuga ‖𝒢 Π_ker‖ grande                 → VETO.
        """
        n = int(L_sym.shape[0])
        ones = np.ones((n,), dtype=np.float64)
        ones_unit = ones / np.sqrt(n)
        l_norm = max(_opnorm2(L_sym), _SAFE_DENOM)
        flux = float(np.linalg.norm(L_sym @ ones_unit)) / l_norm
        is_graph = flux <= max(self._hard_tol, _svd_relerr_bound(n))

        leakage = float(np.linalg.norm(green @ ones_unit))
        if kernel_dim > 0:
            ker_mask = evals <= cutoff
            ker_basis = evecs[:, ker_mask]
            leakage = max(leakage, _opnorm2(green @ ker_basis))
            overlap = float(np.linalg.norm(ker_basis.T @ ones_unit))
            # overlap ∈ [0, 1]; =1 si ker contiene exactamente las constantes.
        else:
            overlap = 0.0

        atoms: list[GreenHeytingVerdict] = []
        atoms.append(
            classify_relative_defect(
                leakage, max(1.0, _opnorm2(green)), self._soft_tol, self._hard_tol, 1.0
            )
        )
        if is_graph:
            if kernel_dim == 0:
                atoms.append(GreenHeytingVerdict.VETOED)
            elif kernel_dim > 1:
                atoms.append(GreenHeytingVerdict.DEGRADED)
            else:
                atoms.append(GreenHeytingVerdict.COHERENT)
            # Las constantes deben vivir en el núcleo.
            if kernel_dim >= 1 and overlap < 1.0 - self._hard_tol:
                atoms.append(GreenHeytingVerdict.DEGRADED)
        else:
            # Anclado: un flujo residual enorme es coherente; ker>0 se degrada.
            if kernel_dim > 0:
                atoms.append(GreenHeytingVerdict.DEGRADED)
            else:
                atoms.append(GreenHeytingVerdict.COHERENT)

        return leakage, flux, overlap, is_graph, heyting_join(*atoms)

    def _classify_condition(self, cond: float) -> GreenHeytingVerdict:
        if not np.isfinite(cond) or cond >= _COND_VETO:
            return GreenHeytingVerdict.VETOED
        if cond >= _COND_DEGRADED:
            return GreenHeytingVerdict.DEGRADED
        return GreenHeytingVerdict.COHERENT

    # ── D.8  SELLO TERMINAL DE LA FASE 1 ─────────────────────────────────
    def _observe_greens_spectrum(
        self,
        L_F: NDArray[np.float64],
    ) -> Phase1GreensObservation:
        """
        Φ₁ — morfismo terminal de la Fase 1.

        Audita sobre L_F:
          • [I1]  simetrización certificada;
          • [I5]  espectro de Hodge, PSD, gap de Fiedler;
          • [I3]  las cuatro identidades de Moore–Penrose;
          • [I2]  aniquilación del núcleo numérico;
          • [I6]  conservación de flujo / tipo de Laplaciano.

        Definición formal del artefacto emitido
        ───────────────────────────────────────
        Sea Obs₁ := Phase1GreensObservation. El juicio

            Φ₁(L_F) ∈ Obs₁

        es un objeto inmutable del topos de Green y constituye, por
        contrato categórico, la *unidad de arranque* de la Fase 2:

            Φ₂  se aplica únicamente a  (Φ₁(L_F), s, h) ,
            y su primer método `_ingest_phase1_precondition` es la
            continuación lógica y tipada de este sello.

        Cualquier consumidor que no pase por este método viola la
        adjunción Observe ⊣ Orient.
        """
        l_raw = self._materialise_laplacian(L_F)
        l_sym, asymmetry, sym_verdict = self._symmetrise_laplacian(l_raw)
        n = int(l_sym.shape[0])

        evals, evecs, cutoff = self._hodge_spectrum(l_sym)
        positive_mask, kernel_dim, _rank_svd, rk_disc = self._resolve_numerical_kernel(
            l_sym, evals, cutoff
        )
        green, inv_evals = self._assemble_green_operator(evecs, evals, positive_mask)

        penrose, wilkinson, penrose_verdict = self._audit_penrose_identities(l_sym, green)
        g_sym_res, is_sa, sa_verdict = self._audit_self_adjoint_green(green)
        is_psd, lam_min, psd_verdict = self._audit_positive_semidefiniteness(evals, cutoff)
        leakage, flux, overlap, is_graph, ker_verdict = self._audit_kernel_and_flux(
            l_sym, green, evals, evecs, cutoff, kernel_dim
        )

        pos = evals[positive_mask]
        if pos.size >= 1:
            fiedler = float(np.min(pos))
            lam_max = float(np.max(pos))
            cond = lam_max / fiedler if fiedler > 0.0 else float("inf")
        else:
            fiedler = 0.0
            lam_max = 0.0
            cond = float("inf")
        cond_verdict = self._classify_condition(cond)
        rank_verdict = (
            GreenHeytingVerdict.DEGRADED if rk_disc > 0 else GreenHeytingVerdict.COHERENT
        )

        # ‖𝒢‖₂ = 1 / λ_min^+ en aritmética exacta (más estable que un SVD extra).
        op_norm = (1.0 / fiedler) if fiedler > 0.0 else 0.0
        trace_val = float(np.sum(inv_evals))
        spectral_radius_l = float(np.max(np.abs(evals))) if evals.size else 0.0

        combined = heyting_join(
            sym_verdict,
            penrose_verdict,
            sa_verdict,
            psd_verdict,
            ker_verdict,
            cond_verdict,
            rank_verdict,
        )

        atoms: Tuple[str, ...] = (
            f"n={n}",
            f"b0={kernel_dim}",
            f"λ_min={lam_min:.6e}",
            f"λ_Fiedler={fiedler:.6e}",
            f"κ_⊥={cond:.6e}",
            f"r₁={penrose[0]:.6e}",
            f"‖𝒢−𝒢ᵀ‖={g_sym_res:.6e}",
            f"leak={leakage:.6e}",
            f"flux={flux:.6e}",
            f"⟨ker,𝟙⟩={overlap:.6e}",
            f"graph={is_graph}",
            f"asym={asymmetry:.6e}",
            f"rkΔ={rk_disc}",
            f"join={combined.name}",
        )

        certificate = GreensSpectralCertificate(
            operator_norm=float(op_norm),
            condition_number=float(cond),
            is_self_adjoint=bool(is_sa),
            is_positive_semidefinite=bool(is_psd),
            pseudo_inverse_residual=float(penrose[0]),
            trace_value=float(trace_val),
            kernel_dimension=int(kernel_dim),
            spectral_verdict=combined,
            fiedler_value=float(fiedler),
            spectral_radius_l=float(spectral_radius_l),
            kernel_leakage=float(leakage),
            flux_residual=float(flux),
            constant_mode_overlap=float(overlap),
            is_graph_laplacian=bool(is_graph),
            penrose_residuals=penrose,
            svd_eig_rank_discrepancy=int(rk_disc),
            wilkinson_bound=float(wilkinson),
            dimension=n,
            diagnostic_atoms=atoms,
        )

        if combined == GreenHeytingVerdict.VETOED:
            logger.error("[GREEN:Φ₁] Espectroscopía VETOED. %s", " | ".join(atoms))
        elif combined == GreenHeytingVerdict.DEGRADED:
            logger.warning("[GREEN:Φ₁] Espectroscopía DEGRADED. %s", " | ".join(atoms))
        else:
            logger.info("[GREEN:Φ₁] Espectroscopía COHERENT. n=%d b0=%d", n, kernel_dim)

        # Sello terminal: este return ES el morfismo de arranque de la Fase 2.
        return Phase1GreensObservation(
            laplacian_symmetric=_freeze_array(l_sym, np.float64),
            green_operator=_freeze_array(green, np.float64),
            certificate=certificate,
            eigenvalues=tuple(float(x) for x in evals.tolist()),
            verdict=combined,
            diagnostic_atoms=atoms,
        )

    # ── Fachada pública v2 (extrae el par (𝒢, certificado) de Obs₁) ──────
    def _prepare_laplacian(self, L_F: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compatibilidad v2: valida y simetriza.

        Preferir `_observe_greens_spectrum` en código nuevo: este método
        no sella el certificado y no es un puerto de la Fase 2.
        """
        l_raw = self._materialise_laplacian(L_F)
        l_sym, _asym, _verdict = self._symmetrise_laplacian(l_raw)
        return l_sym

    def _compute_greens_operator_from_symmetric(
        self,
        L_sym: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], GreensSpectralCertificate]:
        """Compatibilidad v2: espectroscopía sobre un L ya simétrico."""
        obs = self._observe_greens_spectrum(L_sym)
        return np.array(obs.green_operator, dtype=np.float64, copy=True), obs.certificate

    def compute_greens_operator(
        self,
        L_F: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], GreensSpectralCertificate]:
        """API pública de la Fase 1 (contrato v2)."""
        obs = self._observe_greens_spectrum(L_F)
        return np.array(obs.green_operator, dtype=np.float64, copy=True), obs.certificate


# ══════════════════════════════════════════════════════════════════════════════
# §E. FASE 2 — SÍNTESIS DEL PROPAGADOR RETARDADO (KRAMERS–KRONIG)
#     Continuación formal del sello terminal de la Fase 1.
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_RetardedPropagatorSynthesizer(Phase1_GreensSpectroscopist):
    """
    Fase 2 (Orient): resolvente retardado G^R(s) = (L − (s + i h) I)^{-1}.

    Morfismo Φ₂ : Obs₁ × ℂ × ℝ₊ → Obs₂.

    El primer método de esta fase, `_ingest_phase1_precondition`, es la
    *continuación tipada* de `_observe_greens_spectrum`. No existe camino
    legal hacia el resolvente que no atraviese ese ingest.
    """

    def __init__(
        self,
        soft_tol: float = _DEFAULT_SOFT_TOL,
        hard_tol: float = _DEFAULT_HARD_TOL,
        eps_tolerance: Optional[float] = None,
        default_i_eps: float = _DEFAULT_I_EPS,
    ) -> None:
        super().__init__(
            soft_tol=soft_tol,
            hard_tol=hard_tol,
            eps_tolerance=eps_tolerance,
        )
        h0 = float(default_i_eps)
        if not np.isfinite(h0) or h0 < 0.0:
            raise ValueError(f"default_i_eps debe ser real ≥ 0; recibido {h0}.")
        self._default_i_eps = h0

    # ── E.1  INICIO DE FASE 2 = continuación del sello Φ₁ ────────────────
    def _ingest_phase1_precondition(
        self,
        phase1_obs: Phase1GreensObservation,
    ) -> Phase1GreensObservation:
        """
        Continuación formal de `Phase1_GreensSpectroscopist._observe_greens_spectrum`.

        Teorema de handoff (Obs₁ ↪ Fase 2)
        ──────────────────────────────────
        Hipótesis: `phase1_obs` es el valor de retorno de
        `_observe_greens_spectrum` (objeto congelado de tipo
        Phase1GreensObservation).

        Tesis: el objeto es una precondición *habitable* de Φ₂. Se verifica:
          (i)   tipado e inmutabilidad;
          (ii)  L_sym, 𝒢 cuadrados, finitos, de la misma dimensión;
          (iii) el certificado porta un veredicto del retículo;
          (iv)  el espectro no es vacío.

        No se repara un VETOED de Fase 1: el join monótono lo transporta.
        Este método es el único puerto de entrada de la Fase 2.
        """
        if not isinstance(phase1_obs, Phase1GreensObservation):
            raise TypeError(
                "Fase 2 exige el sello terminal de Fase 1 "
                f"(Phase1GreensObservation); recibido {type(phase1_obs)!r}."
            )
        l_sym = phase1_obs.laplacian_symmetric
        green = phase1_obs.green_operator
        if not isinstance(l_sym, np.ndarray) or not isinstance(green, np.ndarray):
            raise TypeError("Obs₁ no contiene operadores ndarray.")
        if l_sym.shape != green.shape or l_sym.ndim != 2 or l_sym.shape[0] != l_sym.shape[1]:
            raise ValueError(
                f"Obs₁ tiene formas incompatibles: L={l_sym.shape}, 𝒢={green.shape}."
            )
        if not np.all(np.isfinite(l_sym)) or not np.all(np.isfinite(green)):
            raise ArithmeticError("Obs₁ contiene valores no finitos en L o 𝒢.")
        if not isinstance(phase1_obs.certificate, GreensSpectralCertificate):
            raise TypeError("Obs₁.certificate no es un GreensSpectralCertificate.")
        if not isinstance(phase1_obs.verdict, GreenHeytingVerdict):
            raise TypeError("Obs₁.verdict no pertenece al retículo de Heyting.")
        if not phase1_obs.eigenvalues:
            raise ValueError("Obs₁.eigenvalues no puede ser vacío.")
        if int(phase1_obs.certificate.dimension) != int(l_sym.shape[0]):
            raise ValueError(
                f"Obs₁.dimension={phase1_obs.certificate.dimension} "
                f"≠ n={l_sym.shape[0]}."
            )
        logger.debug(
            "[GREEN:Φ₂←Φ₁] Ingestión de Obs₁ (verdict=%s, n=%d, b0=%d).",
            phase1_obs.verdict.name,
            l_sym.shape[0],
            phase1_obs.certificate.kernel_dimension,
        )
        return phase1_obs

    # ── E.2  Validación de (s, h) y prescripción iε ──────────────────────
    def _validate_complex_frequency(
        self,
        frequency_s: complex,
        regularization_h: float,
        l_norm: float,
        n: int,
    ) -> Tuple[complex, float, float, bool]:
        """
        Normaliza s ∈ ℂ y h ≥ 0, y eleva h al umbral de máquina si el
        desplazamiento imaginario total es menor que n u ‖L‖.

        Devuelve (s, h_pedida, h_efectiva, es_retardado).

        Convención: G^R(s) = (L − (s + i h) I)^{-1}. Los polos caen en
        s = λ − i h. Con h_eff ≥ 0, Im(polo) ≤ 0 (retardado). Si
        Im(s) + h < 0 el resolvente es *avanzado* y se degrada [I4].
        """
        s_val = complex(frequency_s)
        if not np.isfinite(s_val.real) or not np.isfinite(s_val.imag):
            raise ValueError(f"frequency_s debe ser un complejo finito; recibido {frequency_s!r}.")
        h_req = float(regularization_h)
        if not np.isfinite(h_req) or h_req < 0.0:
            raise ValueError(f"regularization_h debe ser real ≥ 0; recibido {h_req}.")

        machine_shift = max(float(n) * _MACHINE_EPS * max(l_norm, 1.0), _MACHINE_EPS)
        imag_total = float(s_val.imag) + h_req
        is_retarded = imag_total >= -machine_shift
        # Elevación causal: garantiza un iε numéricamente visible.
        if imag_total < machine_shift:
            h_eff = h_req + (machine_shift - min(imag_total, machine_shift))
            h_eff = max(h_eff, machine_shift)
        else:
            h_eff = h_req
        return s_val, h_req, float(h_eff), bool(is_retarded)

    # ── E.3  Holgura a polos  dist(s+ih, σ(L)) ───────────────────────────
    def _pole_clearance(
        self,
        eigenvalues: Sequence[float],
        z_shift: complex,
    ) -> float:
        """δ = min_i |λ_i − z| / max(1, |λ_i|, |z|). δ ≪ n u ⇒ cerca de un polo."""
        if not eigenvalues:
            return float("inf")
        z = complex(z_shift)
        best = float("inf")
        for lam in eigenvalues:
            scale = max(1.0, abs(float(lam)), abs(z))
            best = min(best, abs(complex(lam) - z) / scale)
        return float(best)

    # ── E.4  Inversión estable del resolvente ────────────────────────────
    def _invert_resolvent(
        self,
        L_sym: NDArray[np.float64],
        z_shift: complex,
    ) -> Tuple[NDArray[np.complex128], float]:
        """
        Calcula (L − z I)^{-1} por el oráculo más estable disponible:

          (a) z ∈ ℝ y L − z I bien condicionado → `solve` hermítico;
          (b) z ∈ ℂ                             → LU con pivoteo parcial;
          (c) singularidad                      → SVD truncada (pseudoinversa).

        Devuelve (G^R, residual relativo ‖(L−zI)G − I‖_F / √n).
        """
        n = int(L_sym.shape[0])
        z = complex(z_shift)
        eye_c = np.eye(n, dtype=np.complex128)
        resolvent = L_sym.astype(np.complex128) - z * eye_c

        propagator: Optional[NDArray[np.complex128]] = None
        used_pinv = False

        z_is_real = abs(z.imag) <= _MACHINE_EPS * max(1.0, abs(z.real))
        try:
            if z_is_real:
                shifted = L_sym - float(z.real) * np.eye(n, dtype=np.float64)
                propagator = la.solve(
                    shifted, np.eye(n, dtype=np.float64), assume_a="sym", check_finite=True
                ).astype(np.complex128)
            else:
                lu, piv = la.lu_factor(resolvent, overwrite_a=False, check_finite=True)
                propagator = la.lu_solve((lu, piv), eye_c, check_finite=True)
        except la.LinAlgError:
            used_pinv = True
            logger.warning(
                "[GREEN:Φ₂] Singularidad en z=%+.6e%+.6ej. SVD truncada.",
                z.real,
                z.imag,
            )

        if propagator is None or used_pinv:
            u, sigma, vh = la.svd(resolvent, full_matrices=False, check_finite=True)
            sigma = np.asarray(sigma, dtype=np.float64)
            smax = float(sigma[0]) if sigma.size else 0.0
            tol_svd = max(self._eps, n * _MACHINE_EPS * max(smax, 1.0))
            inv_s = np.zeros_like(sigma)
            keep = sigma > tol_svd
            if not np.any(keep):
                raise PropagatorSingularityError(
                    f"Resolvente nulo en z={z!r}: todos los valores singulares "
                    f"caen bajo τ={tol_svd:.4e}."
                )
            inv_s[keep] = 1.0 / sigma[keep]
            propagator = (vh.conj().T * inv_s) @ u.conj().T

        if not np.all(np.isfinite(propagator)):
            raise PropagatorSingularityError(
                "El propagador resultante contiene valores no finitos."
            )

        residual_mat = resolvent @ propagator - eye_c
        residual = _frobenius(residual_mat) / max(np.sqrt(n), 1.0)
        return np.asarray(propagator, dtype=np.complex128), float(residual)

    # ── E.5  Disipatividad de Kramers–Kronig y cruce hermítico [I4][I8] ─
    def _audit_dissipativity(
        self,
        propagator: NDArray[np.complex128],
        is_retarded: bool,
    ) -> Tuple[float, GreenHeytingVerdict]:
        """
        Para G = (L − (ω + i h)I)^{-1} con L = L^T y h>0,

            Im G  =  (G − G^†) / (2 i)   ≽   0.

        `dissipativity_floor` es el menor autovalor de Im G. Un piso
        groseramente negativo viola la prescripción iε.
        """
        anti = (propagator - propagator.conj().T) / (2.0j)
        herm_im = 0.5 * (anti + anti.conj().T)  # proyección hermítica numérica
        evals_im = la.eigvalsh(np.asarray(herm_im, dtype=np.complex128))
        floor = float(np.min(np.real(evals_im))) if evals_im.size else 0.0
        # Si el resolvente es avanzado, el signo se invierte: se mide |piso|.
        defect = _clamp_nonneg(-floor if is_retarded else floor)
        scale = max(1.0, _opnorm2(propagator))
        verdict = classify_relative_defect(
            defect, scale, self._soft_tol, self._hard_tol, floor=1.0
        )
        return floor, verdict

    def _audit_hermitian_crossing(
        self,
        L_sym: NDArray[np.float64],
        propagator: NDArray[np.complex128],
        z_shift: complex,
    ) -> Tuple[float, GreenHeytingVerdict]:
        """
        Identidad de cruce para L hermítico: G(z*)^† = G(z).

        Equivale a ‖G^† (L − z* I) − I‖ pequeño, o, más barato,
        ‖G^† L − L G^T‖ y la desviación de G respecto de la identidad
        funcional G^† = (L − z* I)^{-1}. Se mide

            ‖ G^† (L − z* I) − I ‖_F / √n
        que debe ser comparable al residual directo del resolvente.
        """
        n = int(L_sym.shape[0])
        zc = complex(z_shift).conjugate()
        left = propagator.conj().T @ (L_sym.astype(np.complex128) - zc * np.eye(n, dtype=np.complex128))
        residual = _frobenius(left - np.eye(n, dtype=np.complex128)) / max(np.sqrt(n), 1.0)
        verdict = classify_relative_defect(
            residual, 1.0, self._soft_tol, self._hard_tol, floor=1.0
        )
        return float(residual), verdict

    def _classify_resolvent_residual(
        self,
        residual: float,
        n: int,
    ) -> GreenHeytingVerdict:
        amp = 1.0 + _svd_relerr_bound(n)
        return classify_relative_defect(
            residual, 1.0, self._soft_tol * amp, self._hard_tol * amp, floor=1.0
        )

    # ── E.6  SELLO TERMINAL DE LA FASE 2 ─────────────────────────────────
    def _orient_retarded_propagator(
        self,
        phase1_obs: Phase1GreensObservation,
        frequency_s: complex,
        regularization_h: Optional[float] = None,
    ) -> Phase2PropagatorOrientation:
        """
        Φ₂ — morfismo terminal de la Fase 2.

        Construye G^R(s) = (L_sym − (s + i h) I)^{-1} reutilizando el
        L_sym *sellado* por Φ₁ (nunca se re-simetriza). Audita:
          • residual del resolvente;
          • prescripción iε / carácter retardado;
          • disipatividad (Im G ≽ 0);
          • cruce hermítico G(z*)^† = G(z);
          • holgura a polos.

        Definición formal del artefacto emitido
        ───────────────────────────────────────
        Sea Obs₂ := Phase2PropagatorOrientation. El juicio

            Φ₂(Φ₁(L_F), s, h) ∈ Obs₂

        es la *unidad de arranque* de la Fase 3: su primer método
        `_ingest_phase2_precondition` es la continuación lógica de
        este sello.
        """
        obs1 = self._ingest_phase1_precondition(phase1_obs)
        l_sym = np.array(obs1.laplacian_symmetric, dtype=np.float64, copy=True)
        n = int(l_sym.shape[0])
        l_norm = max(obs1.certificate.spectral_radius_l, _opnorm2(l_sym), 1.0)
        h_in = self._default_i_eps if regularization_h is None else float(regularization_h)

        s_val, h_req, h_eff, is_retarded = self._validate_complex_frequency(
            frequency_s, h_in, l_norm, n
        )
        z_shift = complex(s_val.real, s_val.imag + h_eff)
        clearance = self._pole_clearance(obs1.eigenvalues, z_shift)
        near_pole = clearance <= max(self._hard_tol, _svd_relerr_bound(n))

        propagator, res_residual = self._invert_resolvent(l_sym, z_shift)
        res_verdict = self._classify_resolvent_residual(res_residual, n)
        floor, diss_verdict = self._audit_dissipativity(propagator, is_retarded)
        cross_res, cross_verdict = self._audit_hermitian_crossing(l_sym, propagator, z_shift)

        retarded_verdict = (
            GreenHeytingVerdict.COHERENT if is_retarded else GreenHeytingVerdict.DEGRADED
        )
        pole_verdict = (
            GreenHeytingVerdict.DEGRADED if near_pole else GreenHeytingVerdict.COHERENT
        )
        combined = heyting_join(
            obs1.verdict,
            res_verdict,
            diss_verdict,
            cross_verdict,
            retarded_verdict,
            pole_verdict,
        )
        is_causal = (
            is_retarded
            and diss_verdict != GreenHeytingVerdict.VETOED
            and cross_verdict != GreenHeytingVerdict.VETOED
        )

        atoms: Tuple[str, ...] = (
            f"s={s_val.real:+.6e}{s_val.imag:+.6e}j",
            f"h_req={h_req:.3e}",
            f"h_eff={h_eff:.3e}",
            f"δ_polo={clearance:.6e}",
            f"r_res={res_residual:.6e}",
            f"ImG_min={floor:.6e}",
            f"r_×={cross_res:.6e}",
            f"retarded={is_retarded}",
            f"causal={is_causal}",
            f"join={combined.name}",
        )

        if combined == GreenHeytingVerdict.VETOED:
            logger.error("[GREEN:Φ₂] Orientación VETOED. %s", " | ".join(atoms))
        elif combined == GreenHeytingVerdict.DEGRADED:
            logger.warning("[GREEN:Φ₂] Orientación DEGRADED. %s", " | ".join(atoms))
        else:
            logger.info("[GREEN:Φ₂] Orientación COHERENT. δ=%.3e", clearance)

        # Sello terminal: este return ES el morfismo de arranque de la Fase 3.
        return Phase2PropagatorOrientation(
            phase1_observation=obs1,
            retarded_propagator=_freeze_array(propagator, np.complex128),
            frequency_s=s_val,
            regularization_h_requested=float(h_req),
            regularization_h_effective=float(h_eff),
            pole_clearance=float(clearance),
            resolvent_residual=float(res_residual),
            dissipativity_floor=float(floor),
            crossing_residual=float(cross_res),
            is_retarded=bool(is_retarded),
            is_causal=bool(is_causal),
            is_near_pole=bool(near_pole),
            verdict=combined,
            diagnostic_atoms=atoms,
        )

    # ── Fachadas públicas v2 ─────────────────────────────────────────────
    def _compute_retarded_propagator_from_symmetric(
        self,
        L_sym: NDArray[np.float64],
        frequency_s: complex,
        regularization_h: float,
    ) -> NDArray[np.complex128]:
        """Compatibilidad v2: observa L_sym y orienta el resolvente."""
        obs1 = self._observe_greens_spectrum(L_sym)
        obs2 = self._orient_retarded_propagator(obs1, frequency_s, regularization_h)
        return np.array(obs2.retarded_propagator, dtype=np.complex128, copy=True)

    def compute_retarded_propagator(
        self,
        L_F: NDArray[np.float64],
        frequency_s: complex,
        regularization_h: float = _DEFAULT_I_EPS,
    ) -> NDArray[np.complex128]:
        """API pública de la Fase 2 (contrato v2)."""
        obs1 = self._observe_greens_spectrum(L_F)
        obs2 = self._orient_retarded_propagator(obs1, frequency_s, regularization_h)
        return np.array(obs2.retarded_propagator, dtype=np.complex128, copy=True)


# ══════════════════════════════════════════════════════════════════════════════
# §F. FASE 3 — VEREDICTO DE CALIBRE DE HEYTING
#     Continuación formal del sello terminal de la Fase 2.
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingGreensDecider(Phase2_RetardedPropagatorSynthesizer):
    """
    Fase 3 (Decide & Act): join terminal, causa raíz y sello forense.

    Morfismo Φ₃ : Obs₂ → Gov_Green.
    """

    def __init__(
        self,
        soft_tol: float = _DEFAULT_SOFT_TOL,
        hard_tol: float = _DEFAULT_HARD_TOL,
        eps_tolerance: Optional[float] = None,
        default_i_eps: float = _DEFAULT_I_EPS,
    ) -> None:
        super().__init__(
            soft_tol=soft_tol,
            hard_tol=hard_tol,
            eps_tolerance=eps_tolerance,
            default_i_eps=default_i_eps,
        )

    # ── F.1  INICIO DE FASE 3 = continuación del sello Φ₂ ────────────────
    def _ingest_phase2_precondition(
        self,
        phase2_orient: Phase2PropagatorOrientation,
    ) -> Phase2PropagatorOrientation:
        """
        Continuación formal de
        `Phase2_RetardedPropagatorSynthesizer._orient_retarded_propagator`.

        Teorema de handoff (Obs₂ ↪ Fase 3)
        ──────────────────────────────────
        Hipótesis: `phase2_orient` es el valor de retorno de
        `_orient_retarded_propagator`.

        Tesis: Obs₂ es habitable. Se exige:
          (i)   tipado Phase2PropagatorOrientation;
          (ii)  re-ingesta de Obs₁ (y, transitivamente, de L y 𝒢);
          (iii) G^R finito, cuadrado, conforme con L;
          (iv)  s, h, residuales numéricamente legales.

        El veredicto de Obs₂ se transporta monótonamente; no se repara.
        """
        if not isinstance(phase2_orient, Phase2PropagatorOrientation):
            raise TypeError(
                "Fase 3 exige el sello terminal de Fase 2 "
                f"(Phase2PropagatorOrientation); recibido {type(phase2_orient)!r}."
            )
        obs1 = self._ingest_phase1_precondition(phase2_orient.phase1_observation)
        prop = phase2_orient.retarded_propagator
        n = int(obs1.laplacian_symmetric.shape[0])
        if not isinstance(prop, np.ndarray) or prop.shape != (n, n):
            raise ValueError(
                f"Obs₂.retarded_propagator tiene forma {getattr(prop, 'shape', None)}, "
                f"se esperaba {(n, n)}."
            )
        if not np.all(np.isfinite(prop)):
            raise PropagatorSingularityError(
                "Obs₂.retarded_propagator contiene valores no finitos."
            )
        if not np.isfinite(phase2_orient.regularization_h_effective):
            raise ValueError("Obs₂.regularization_h_effective no es finito.")
        if not isinstance(phase2_orient.verdict, GreenHeytingVerdict):
            raise TypeError("Obs₂.verdict no pertenece al retículo de Heyting.")
        logger.debug(
            "[GREEN:Φ₃←Φ₂] Ingestión de Obs₂ (verdict=%s, δ=%.3e).",
            phase2_orient.verdict.name,
            phase2_orient.pole_clearance,
        )
        return phase2_orient

    # ── F.2  Join terminal y causa raíz ──────────────────────────────────
    def _compose_terminal_verdict(
        self,
        phase2_orient: Phase2PropagatorOrientation,
    ) -> GreenHeytingVerdict:
        """
        v_Green = v_{Φ₁} ∨ v_{Φ₂} ∨ v_finitud(G^R).

        La finitud ya se exigió en el ingest; se re-verifica como
        cinturón de seguridad ante corrupción del DTO.
        """
        finitude = (
            GreenHeytingVerdict.COHERENT
            if np.all(np.isfinite(phase2_orient.retarded_propagator))
            else GreenHeytingVerdict.VETOED
        )
        return heyting_join(
            phase2_orient.phase1_observation.verdict,
            phase2_orient.verdict,
            finitude,
        )

    def _root_cause_exception(
        self,
        final_verdict: GreenHeytingVerdict,
        phase2_orient: Phase2PropagatorOrientation,
    ) -> Optional[GreenPropagatorError]:
        if final_verdict != GreenHeytingVerdict.VETOED:
            return None
        cert = phase2_orient.phase1_observation.certificate
        if not cert.is_positive_semidefinite:
            return LaplacianIndefinitenessError(
                f"L_F no es semidefinido positivo (λ_min fuera de tolerancia). "
                f"Veredicto={final_verdict.name}."
            )
        if max(cert.penrose_residuals) > self._hard_tol:
            return MoorePenroseResidualError(
                f"Identidades de Penrose rotas: r={cert.penrose_residuals}."
            )
        if cert.kernel_leakage > self._hard_tol * max(1.0, cert.operator_norm):
            return KernelLeakageError(
                f"𝒢 no aniquila ker L (leak={cert.kernel_leakage:.6e})."
            )
        if (
            cert.is_graph_laplacian
            and cert.kernel_dimension == 0
        ):
            return FluxConservationError(
                "Laplaciano no anclado con b_0=0: se perdió el modo constante."
            )
        if not phase2_orient.is_causal:
            return CausalityViolationError(
                f"Fallo de causalidad iε (retarded={phase2_orient.is_retarded}, "
                f"ImG_min={phase2_orient.dissipativity_floor:.6e}, "
                f"cruce={phase2_orient.crossing_residual:.6e})."
            )
        if phase2_orient.resolvent_residual > self._hard_tol:
            return PropagatorSingularityError(
                f"Residual del resolvente {phase2_orient.resolvent_residual:.6e} "
                f"excede hard_tol={self._hard_tol:.3e}."
            )
        return GreenPropagatorError(
            f"Colapso de Green: invariante espectral violado. "
            f"Veredicto supremo={final_verdict.name}."
        )

    # ── F.3  Huella forense ──────────────────────────────────────────────
    def _forensic_provenance(
        self,
        phase2_orient: Phase2PropagatorOrientation,
        final_verdict: GreenHeytingVerdict,
    ) -> str:
        cert = phase2_orient.phase1_observation.certificate
        s_val = phase2_orient.frequency_s
        l_inv = _matrix_invariants(phase2_orient.phase1_observation.laplacian_symmetric)
        g_inv = _matrix_invariants(phase2_orient.phase1_observation.green_operator)
        payload = "|".join(
            (
                f"n={cert.dimension}",
                _round_for_hash(cert.pseudo_inverse_residual),
                _round_for_hash(cert.trace_value),
                _round_for_hash(cert.condition_number),
                _round_for_hash(cert.kernel_leakage),
                _round_for_hash(cert.flux_residual),
                _round_for_hash(cert.fiedler_value),
                _round_for_hash(phase2_orient.resolvent_residual),
                _round_for_hash(phase2_orient.dissipativity_floor),
                _round_for_hash(phase2_orient.crossing_residual),
                _round_for_hash(phase2_orient.pole_clearance),
                f"s={s_val.real:.16e}{s_val.imag:+.16e}j",
                f"h={phase2_orient.regularization_h_effective:.16e}",
                f"L={l_inv}",
                f"G={g_inv}",
                f"b0={cert.kernel_dimension}",
                f"V={int(final_verdict.value)}",
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── F.4  SELLO TERMINAL DE LA FASE 3 ─────────────────────────────────
    def _evaluate_green_governance(
        self,
        phase2_orient: Phase2PropagatorOrientation,
        raise_on_veto: bool = True,
    ) -> PropagatorResponseState:
        """
        Φ₃ — morfismo terminal de la Fase 3 y del funtor Z_Green.

        Join en Ω:

            v_Green  =  v_{Φ₁} ∨ v_{Φ₂} ∨ v_finitud.

        Si v_Green = VETOED y `raise_on_veto`, se eleva la excepción de
        causa raíz (Penrose / núcleo / flujo / causalidad / singularidad)
        encapsulada en HeytingLatticeVeto.
        """
        obs2 = self._ingest_phase2_precondition(phase2_orient)
        final_verdict = self._compose_terminal_verdict(obs2)
        cert = obs2.phase1_observation.certificate

        if final_verdict == GreenHeytingVerdict.VETOED:
            logger.critical(
                "[GREEN:Φ₃] VETO TERMINAL. r₁=%.4e leak=%.4e r_res=%.4e ImG=%.4e.",
                cert.pseudo_inverse_residual,
                cert.kernel_leakage,
                obs2.resolvent_residual,
                obs2.dissipativity_floor,
            )
            cause = self._root_cause_exception(final_verdict, obs2)
            if raise_on_veto:
                message = (
                    f"Colapso de Green: invariante espectral violado. "
                    f"Veredicto Supremo = {final_verdict.name}."
                )
                if cause is not None:
                    raise HeytingLatticeVeto(
                        f"{message} Causa: {cause}",
                        verdict=final_verdict,
                        cause=cause,
                    ) from cause
                raise HeytingLatticeVeto(message, verdict=final_verdict)
        elif final_verdict == GreenHeytingVerdict.DEGRADED:
            logger.warning(
                "[GREEN:Φ₃] Estado DEGRADED. κ_⊥=%.4e δ_polo=%.4e.",
                cert.condition_number,
                obs2.pole_clearance,
            )

        stamp = time.time()
        provenance = self._forensic_provenance(obs2, final_verdict)
        atoms: Tuple[str, ...] = (
            f"v={final_verdict.name}",
            f"v1={obs2.phase1_observation.verdict.name}",
            f"v2={obs2.verdict.name}",
            f"b0={cert.kernel_dimension}",
            f"κ={cert.condition_number:.6e}",
            f"r₁={cert.pseudo_inverse_residual:.6e}",
            f"r_res={obs2.resolvent_residual:.6e}",
            f"causal={obs2.is_causal}",
            f"hash={provenance[:16]}",
        )
        diagnostic_note = (
            f"Veredicto final: {final_verdict.name}. "
            f"Autoadjunto: {cert.is_self_adjoint}. "
            f"PSD: {cert.is_positive_semidefinite}. "
            f"b_0={cert.kernel_dimension}. "
            f"Fiedler={cert.fiedler_value:.4e}. "
            f"Penrose r₁={cert.pseudo_inverse_residual:.4e}. "
            f"Causal: {obs2.is_causal} "
            f"(δ_polo={obs2.pole_clearance:.4e}, "
            f"ImG_min={obs2.dissipativity_floor:.4e}). "
            f"Huella: {provenance[:16]}…"
        )

        return PropagatorResponseState(
            green_matrix=obs2.phase1_observation.green_operator,
            retarded_propagator=obs2.retarded_propagator,
            spectral_certificate=cert,
            final_verdict=final_verdict,
            provenance_hash=provenance,
            timestamp_utc=stamp,
            phase2_orientation=obs2,
            diagnostic_note=diagnostic_note,
            diagnostic_atoms=atoms,
        )

    def _decide_heyting_verdict(
        self,
        certificate: GreensSpectralCertificate,
        propagator: NDArray[np.complex128],
    ) -> GreenHeytingVerdict:
        """
        Compatibilidad v2: join reducido certificado ⊕ finitud de G^R.

        El camino canónico es `_evaluate_green_governance`, que consume
        el sello completo de Fase 2 (disipatividad, cruce, holgura).
        """
        finitude = (
            GreenHeytingVerdict.COHERENT
            if isinstance(propagator, np.ndarray) and np.all(np.isfinite(propagator))
            else GreenHeytingVerdict.VETOED
        )
        psd = (
            GreenHeytingVerdict.COHERENT
            if certificate.is_positive_semidefinite
            else GreenHeytingVerdict.VETOED
        )
        mp = classify_relative_defect(
            certificate.pseudo_inverse_residual,
            1.0,
            self._soft_tol,
            self._hard_tol,
            floor=1.0,
        )
        return heyting_join(certificate.spectral_verdict, finitude, psd, mp)

    def execute_propagator_governance(
        self,
        L_F: NDArray[np.float64],
        frequency_s: complex,
        regularization_h: float = _DEFAULT_I_EPS,
        raise_on_veto: bool = True,
    ) -> PropagatorResponseState:
        """
        Orquesta Φ₃ ∘ Φ₂ ∘ Φ₁ sobre (L_F, s, h).

        El Laplaciano se materializa *una sola vez* en Φ₁; Φ₂ reutiliza
        el L_sym sellado; Φ₃ no vuelve a factorizar.
        """
        phase1_obs = self._observe_greens_spectrum(L_F)
        phase2_orient = self._orient_retarded_propagator(
            phase1_obs, frequency_s, regularization_h
        )
        return self._evaluate_green_governance(
            phase2_orient, raise_on_veto=raise_on_veto
        )


# ══════════════════════════════════════════════════════════════════════════════
# §G. SOBERANO PRINCIPAL — COMPOSICIÓN FUNTORIAL Φ₃ ∘ Φ₂ ∘ Φ₁
# ══════════════════════════════════════════════════════════════════════════════
class SheafGreensPropagatorSolver(Phase3_HeytingGreensDecider):
    """
    Soberano del Propagador de Green de-confinado sobre un haz celular.

    Garantiza invertibilidad Hodge, causalidad iε y coherencia de
    Moore–Penrose en el C*-álgebra concreta B(H_n).

        Z_Green(L_F, s)  =  Φ₃( Φ₂( Φ₁(L_F), s, h ) ).

    En circuitos eléctricos, L_F es el Laplaciano nodal (admitancia
    estática) y G^R(s) el operador de impedancia de transferencia a la
    frecuencia compleja s: el veto de Heyting aborta una red cuyo
    resolvente deja de ser causal o deja de conservar carga.
    """

    def __init__(
        self,
        eps_tolerance: float = _DEFAULT_SOFT_TOL,
        hard_tol: float = _DEFAULT_HARD_TOL,
        default_i_eps: float = _DEFAULT_I_EPS,
    ) -> None:
        if eps_tolerance <= 0.0 or not np.isfinite(eps_tolerance):
            raise ValueError("eps_tolerance debe ser un número positivo finito.")
        super().__init__(
            soft_tol=float(eps_tolerance),
            hard_tol=float(hard_tol),
            eps_tolerance=float(eps_tolerance),
            default_i_eps=float(default_i_eps),
        )

    def solve(
        self,
        L_F: NDArray[np.float64],
        frequency_s: complex = 0.0j,
        regularization_h: float = _DEFAULT_I_EPS,
        raise_on_veto: bool = True,
    ) -> PropagatorResponseState:
        """Alias idiomático de `execute_propagator_governance`."""
        return self.execute_propagator_governance(
            L_F=L_F,
            frequency_s=frequency_s,
            regularization_h=regularization_h,
            raise_on_veto=raise_on_veto,
        )

    def static_green(
        self,
        L_F: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], GreensSpectralCertificate]:
        """Sólo Φ₁: función de Green estática y certificado de Hodge."""
        return self.compute_greens_operator(L_F)

    def retarded(
        self,
        L_F: NDArray[np.float64],
        frequency_s: complex,
        regularization_h: float = _DEFAULT_I_EPS,
    ) -> NDArray[np.complex128]:
        """Sólo Φ₁ ∘ Φ₂: resolvente retardado, sin veto de Fase 3."""
        return self.compute_retarded_propagator(L_F, frequency_s, regularization_h)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "__version__",
    "GreenHeytingVerdict",
    "heyting_join",
    "heyting_meet",
    "classify_relative_defect",
    "GreenPropagatorError",
    "LaplacianAsymmetryError",
    "LaplacianIndefinitenessError",
    "MoorePenroseResidualError",
    "KernelLeakageError",
    "FluxConservationError",
    "PropagatorSingularityError",
    "CausalityViolationError",
    "HeytingLatticeVeto",
    "GreensSpectralCertificate",
    "Phase1GreensObservation",
    "Phase2PropagatorOrientation",
    "PropagatorResponseState",
    "GreenGovernanceState",
    "Phase1_GreensSpectroscopist",
    "Phase2_RetardedPropagatorSynthesizer",
    "Phase3_HeytingGreensDecider",
    "SheafGreensPropagatorSolver",
]
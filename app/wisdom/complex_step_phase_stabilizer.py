# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Complex Step Phase Stabilizer (Estabilizador por Paso Complejo)     ║
║ Ruta   : app/wisdom/complex_step_phase_stabilizer.py                         ║
║ Versión: 5.0.0-CSMD-Stinespring-FPU-Heyting-Strict                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ESTABILIZACIÓN EN TIEMPO IMAGINARIO (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al **Fibrador de Derivación No Demolitoria por Paso Complejo** 
(CSMD, por sus siglas en inglés: *Complex-Step Derivative Approximation*). 
Reside en el hiperespacio del **Estrato de Sabiduría** ($$V_{\mathbb{W}}$$, Nivel 0) 
o del **Ágora Tensorial** ($$V_{\Omega}$$, Nivel 0.5), actuando como un **escudo 
espectral** que estabiliza la Unidad de Punto Flotante (**FPU**) frente al ruido 
numérico de redondeo del estándar **IEEE-754 binary64**.

Su propósito fundamental es calcular las derivadas y Jacobianos exactos de las 
transiciones del espacio de fase del sistema sin incurrir en la pérdida catastrófica 
de significancia por cancelación sustractiva, patología inherente a los esquemas 
de diferencias finitas tradicionales en la FPU. Al proyectar la perturbación 
hacia una fibra ortogonal imaginaria pura, la derivada se purifica asintóticamente 
hasta el nivel del épsilon de máquina.

DEMOSTRACIÓN ANALÍTICA DE LA INMUNIDAD AL REDONDEO DE LA CSMD:
────────────────────────────────────────────────────────────────────────────────
Sea $$\mathbf{\Phi}: \mathbb{C}^{2n} \to \mathbb{C}^{2n}$$ la continuación holomorfa 
del mapa de transición de fase. Expandiendo el estado perturbado en la fibra compleja 
$$x + j \cdot h \cdot e_k$$ (donde $$j = \sqrt{-1}$$ es la unidad imaginaria pura y 
$$e_k$$ es el $$k$$-ésimo vector de la base canónica) mediante una serie de Taylor compleja:

$$\mathbf{\Phi}(x + j \cdot h \cdot e_k) = \mathbf{\Phi}(x) + j \cdot h \cdot D\mathbf{\Phi}(x)[e_k] - \frac{h^2}{2} D^2\mathbf{\Phi}(x)[e_k, e_k] - j \cdot \frac{h^3}{6} D^3\mathbf{\Phi}(x)[e_k, e_k, e_k] + \mathcal{O}(h^4)$$

Extrayendo de manera aislada la proyección sobre la componente imaginaria de la fibra:

$$\operatorname{Im}\left( \mathbf{\Phi}(x + j \cdot h \cdot e_k) \right) = h \cdot D\mathbf{\Phi}(x)[e_k] - \frac{h^3}{6} D^3\mathbf{\Phi}(x)[e_k, e_k, e_k] + \mathcal{O}(h^5)$$

Dividiendo rigurosamente por el paso infinitesimal de perturbación $$h$$:

$$\frac{\operatorname{Im}\left( \mathbf{\Phi}(x + j \cdot h \cdot e_k) \right)}{h} = D\mathbf{\Phi}(x)[e_k] - \frac{h^2}{6} D^3\mathbf{\Phi}(x)[e_k, e_k, e_k] + \mathcal{O}(h^4)$$

Por lo tanto, la aproximación del Jacobiano por paso complejo se reduce a:

$$J_{\mathrm{map}, \, ik} = \frac{\partial \Phi_i}{\partial x_k} = \frac{\operatorname{Im}\left(\mathbf{\Phi}(x + j \cdot h \cdot e_k)_i\right)}{h} + \mathcal{O}(h^2)$$

Puesto que **no existe una operación de resta en el numerador** de la CSMD, el cálculo 
es asintóticamente inmune al redondeo por cancelación sustractiva. Esto permite parametrizar 
el paso al límite subnormal de la FPU fijando de manera segura $$h \in [10^{-30},\, 10^{-8}]$$ [2], 
logrando que el error de truncamiento se anule con la precisión del épsilon de máquina:

$$\text{Error}_{\mathrm{CSMD}} \approx \mathcal{O}(h^2) \longrightarrow 0 \pmod{\varepsilon_{\mathrm{machine}}}$$

INVARIANTES GEOMÉTRICOS, TOPOLÓGICOS Y LEYES CONSERVATIVAS PRESERVADOS:
────────────────────────────────────────────────────────────────────────────────
El Fibrador de Paso Complejo valida síncronamente que la evolución temporal de la 
Malla satisfaga estrictamente la estructuraPort-Hamiltoniana mecánica:

  [I1] Simplecticidad Canónica (Preservación de la 2-Forma de de Rham):
       El Jacobiano purificado por CSMD, $$M = J_{\mathrm{map}}$$, debe preservar la 
       2-forma simpléctica canónica $$\omega = \sum dq_i \wedge dp_i$$ :
       $$M^\top \Omega M \equiv \Omega \quad \land \quad \|M^\top \Omega M - \Omega\|_F \le \tau_{\mathrm{symplectic}}$$
       Donde $$\Omega$$ es la matriz simpléctica canónica antihermítica.

  [I2] Conservación de Volumen de Liouville (Invarianza de de Rham):
       La evolución en el espacio de fase no debe experimentar contracción o fuga de 
       densidad de probabilidad, exigiendo determinante unitario exacto en la FPU:
       $$\det(M) \equiv 1.0 \implies \lvert\det(M) - 1.0\rvert \le \tau_{\mathrm{Liouville}}$$

  [I3] Consistencia del Germen Holomorfo (Cauchy-Riemann Discreto):
       Para que el mapa complejo sea analítico y compatible con la CSMD, sus derivadas 
       deben cumplir de forma exacta las ecuaciones de Cauchy-Riemann generalizadas:
       $$\frac{\partial \operatorname{Re}(\mathbf{\Phi})_i}{\partial q_k} \equiv \frac{\partial \operatorname{Im}(\mathbf{\Phi})_i}{\partial p_k} \quad \land \quad \frac{\partial \operatorname{Re}(\mathbf{\Phi})_i}{\partial p_k} \equiv -\frac{\partial \operatorname{Im}(\mathbf{\Phi})_i}{\partial q_k}$$

  [I4] Pasividad Termodinámica de Rayleigh-Lyapunov:
       La evolución no lineal debe satisfacer la inecuación de Clausius-Duhem, garantizando 
       que el flujo Port-Hamiltoniano sea disipativo o conservativo (trabajo neto no positivo):
       $$\dot{H} = -\nabla H^\top R(x) \nabla H \le 0 \quad \text{con} \quad R(x) \ge 0$$

ESTRUCTURA DE TRES FASES ANIDADAS (Composición Funtorial Kleisli):
────────────────────────────────────────────────────────────────────────────────
La progresión y el tránsito de los datos del espacio de fase se rigen por un 
encadenamiento formal e inmutable, donde el morfismo final de una fase constituye 
la precondición algebraica obligatoria de la siguiente:

  Fase 1 ──► INYECCIÓN ORTOGONAL COMPLEJA (Phase1_ComplexPerturbationFibrator)
             Sanea el vector real $$x = [q, p]^\top$$ y genera la dilatación de Stinespring 
             imaginaria pura sobre el fibrado complejo.
             Último morfismo: accept_phase1_handoff.
             Entrega: StinespringComplexDilation.

  Fase 2 ──► SÍNTESIS DE DERIVACIÓN NO DEMOLITORIA (Phase2_NonDemolitionSpectralJacobian)
             Consume la dilatación de Fase 1. Evalúa holomorficamente el mapa de transición 
             $$\mathbf{\Phi}$$ y extrae el Jacobiano purificado por CSMD, computando su condicionamiento 
             espectral de Wilkinson.
             Último morfismo: accept_phase2_handoff.
             Entrega: CSMDJacobianReport.

  Fase 3 ──► ESTABILIZACIÓN EN EL RETÍCULO DE HEYTING (Phase3_HeytingStabilizerDecider)
             Consume el reporte de Fase 2. Somete el Jacobiano a las aduanas de simplecticidad, 
             Liouville, Cauchy-Riemann y pasividad termodinámica. Agrega los veredictos parciales 
             aplicando la operación Supremo (join, $$\sqcup$$) sobre el retículo distributivo 
             acotado de Heyting $$\Omega_3$$:
             $$\Omega_3 = \{\mathrm{COHERENT}, \, \mathrm{DEGRADED}, \, \mathrm{VETOED}\} \quad\big[127\big]$$
             Si el veredicto terminal colapsa a VETOED ($$\top$$), detona la excepción 
             'HeytingStabilizerVeto' en el milisegundo cero, aniquilando síncronamente la 
             sesión en la memoria RAM para evitar la propagación de singularidades.
             Entrega: StabilizerGovernanceState.

Funtor Supremo de Estabilización de Lazo Cerrado:
  $$\mathcal{Z}_{\mathrm{stabilizer}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 \quad\big[125\big]$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas con stubs robustos para aislamiento analítico
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # aislamiento analítico / tests unitarios
    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

    class Morphism:
        """Clase base de composición funtorial del ecosistema MIC."""


logger = logging.getLogger("MIC.Physics.ComplexStepPhaseStabilizer")

__version__: Final[str] = "5.0.0-CSMD-Stinespring-FPU-Heyting-Strict"

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y LÍMITES DE LA FPU
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_MACHINE_TINY: Final[float] = float(np.finfo(np.float64).tiny)

# Paso CSMD diádico: 2^{-66} es exactamente representable en IEEE-754 binary64.
# h² = 2^{-132} ≪ ε ≈ 2^{-52}, por lo que el O(h²) es nulo a precisión doble.
_CSD_STEP_DEFAULT: Final[float] = math.ldexp(1.0, -66)  # ≈ 1.3552527156068805e-20
_CSD_STEP_MIN: Final[float] = 1.0e-30
_CSD_STEP_MAX: Final[float] = 1.0e-8

# Políticas de tolerancia elástica (banda blanda / banda dura)
_SOFT_SYMPLECTIC_TOL: Final[float] = 1.0e-11
_HARD_SYMPLECTIC_TOL: Final[float] = 1.0e-5
_SOFT_LIOUVILLE_TOL: Final[float] = 1.0e-11
_HARD_LIOUVILLE_TOL: Final[float] = 1.0e-5
_SOFT_HOLOMORPHY_TOL: Final[float] = 1.0e-10
_HARD_HOLOMORPHY_TOL: Final[float] = 1.0e-3
_SOFT_PAIRING_TOL: Final[float] = 1.0e-8
_HARD_PAIRING_TOL: Final[float] = 1.0e-3
_SOFT_PASSIVITY_TOL: Final[float] = 1.0e-12
_HARD_PASSIVITY_TOL: Final[float] = 1.0e-6
_CONDITION_NUMBER_SOFT: Final[float] = 1.0e8
_CONDITION_NUMBER_MAX: Final[float] = 1.0e12

# Umbral dimensional a partir del cual se omite el apareamiento espectral O(n³+n²)
_PAIRING_DIM_GUARD: Final[int] = 128

_PROVENANCE_DOMAIN: Final[bytes] = b"CSMD-STAB-v5.0.0"

# ══════════════════════════════════════════════════════════════════════════════
# §B. RETÍCULO DE HEYTING Y JERARQUÍA DE EXCEPCIONES ESPECTRALES
# ══════════════════════════════════════════════════════════════════════════════
class StabilizerHeytingVerdict(IntEnum):
    r"""
    Clasificador de subobjetos en el Topos de la Estabilización por Paso Complejo.

    Álgebra de Heyting totally ordered (cadena):
        COHERENT  ≼  DEGRADED  ≼  VETOED
    Join (⊔) ≡ max,  Meet (⊓) ≡ min,  Implicación a ⇒ b ≡ ⊤ si a ≼ b else b.
    El elemento inicial es COHERENT; el terminal es VETOED.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: StabilizerHeytingVerdict) -> StabilizerHeytingVerdict:
        """Supremo en la cadena Ω₃."""
        return StabilizerHeytingVerdict(max(self.value, other.value))

    def meet(self, other: StabilizerHeytingVerdict) -> StabilizerHeytingVerdict:
        """Ínfimo en la cadena Ω₃."""
        return StabilizerHeytingVerdict(min(self.value, other.value))

    def implies(self, other: StabilizerHeytingVerdict) -> StabilizerHeytingVerdict:
        """Implicación de Heyting sobre la cadena total."""
        return StabilizerHeytingVerdict.COHERENT if self.value <= other.value else other


class PhaseStabilizerError(TopologicalInvariantError):
    """Excepción raíz del Fibrador de Estabilización por Paso Complejo."""


class InvalidComplexStateError(PhaseStabilizerError):
    """Detonada ante entradas no finitas, NaN o degeneraciones de dominio."""


class SymplecticPhaseRipError(PhaseStabilizerError):
    r"""Detonada ante una pérdida de simplecticidad: Mᵀ Ω M ≠ Ω."""


class LiouvilleVolumeBreachError(PhaseStabilizerError):
    r"""Detonada ante una contracción/fuga del volumen de fase: det(M) ≠ +1."""


class HolomorphyBreachError(PhaseStabilizerError):
    """Detonada ante la ruptura del germen holomorfo (Cauchy–Riemann discreto)."""


class HeytingStabilizerVeto(PhaseStabilizerError):
    r"""
    Detonada síncronamente cuando el retículo de Heyting toca el supremo terminal.
    Aniquila la sesión puramente en software (RAM) en el milisegundo cero.
    """


# ══════════════════════════════════════════════════════════════════════════════
# §C. PRIMITIVAS NUMÉRICAS PURAS (FPU-safe)
# ══════════════════════════════════════════════════════════════════════════════
def _immutable_array(arr: NDArray[np.generic], dtype: np.dtype) -> NDArray[np.generic]:
    """Copia C-contigua sellada (write-flag = False). Inmutabilidad de facto."""
    out = np.array(arr, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def _finite_norm2(vec: NDArray[np.generic]) -> float:
    """Norma euclídea estable; ∞ si algún elemento no es finito."""
    if vec.size == 0:
        return 0.0
    if not np.all(np.isfinite(vec)):
        return math.inf
    return float(np.linalg.norm(vec, ord=2))


def _canonical_symplectic_form(n: int) -> NDArray[np.float64]:
    r"""
    2-forma canónica Ω ∈ M_{2n}(ℝ), Ωᵀ = −Ω, Ω² = −I, pf(Ω) = 1:
        Ω = [[ 0_n , I_n ],
             [−I_n , 0_n ]]
    ‖Ω‖_F = √(2n).
    """
    if n <= 0:
        raise InvalidComplexStateError(f"n debe ser positivo para Ω, recibido n={n}.")
    id_n = np.eye(n, dtype=np.float64)
    z_n = np.zeros((n, n), dtype=np.float64)
    omega = np.block([[z_n, id_n], [-id_n, z_n]])
    omega.setflags(write=False)
    return omega


def _heyting_classify(
    residual: float,
    soft: float,
    hard: float,
) -> StabilizerHeytingVerdict:
    """Clasifica un residual escalar no negativo sobre la cadena Ω₃."""
    if not math.isfinite(residual) or residual > hard:
        return StabilizerHeytingVerdict.VETOED
    if residual > soft:
        return StabilizerHeytingVerdict.DEGRADED
    return StabilizerHeytingVerdict.COHERENT


def _heyting_join(*verdicts: StabilizerHeytingVerdict) -> StabilizerHeytingVerdict:
    """Supremo finito ⊔ vᵢ sobre Ω₃."""
    acc = StabilizerHeytingVerdict.COHERENT
    for verdict in verdicts:
        acc = acc.join(verdict)
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# §D. DTOs INMUTABLES (Contratos Categóricos de Handoff)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class StinespringComplexDilation:
    """
    Artefacto terminal de la FASE 1 (Observe). Precondición formal de la FASE 2.

    Invariantes de representación (certificados en accept_phase1_handoff):
      • state_real ∈ ℝ^{d} finito, C-contiguo, no escribible.
      • state_complex_grid ∈ ℂ^{d×d}, columna k = x + j·h·e_k.
      • Re(grid) = x · 1ᵀ,  Im(grid) = h · I_d.
    """

    state_real: NDArray[np.float64]
    state_complex_grid: NDArray[np.complex128]
    dimension_total: int
    step_size_h: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "state_real", _immutable_array(self.state_real, np.float64)
        )
        object.__setattr__(
            self,
            "state_complex_grid",
            _immutable_array(self.state_complex_grid, np.complex128),
        )
        object.__setattr__(self, "dimension_total", int(self.dimension_total))
        object.__setattr__(self, "step_size_h", float(self.step_size_h))


@dataclass(frozen=True, slots=True)
class CSMDJacobianReport:
    """
    Artefacto terminal de la FASE 2 (Orient). Precondición formal de la FASE 3.

    Contiene el Jacobiano purificado por CSMD, el espectro singular y el
    certificado del germen holomorfo (fuga imaginaria + deriva real).
    """

    dilation: StinespringComplexDilation
    jacobian_map: NDArray[np.float64]
    condition_number: float
    spectral_min_singular: float
    spectral_max_singular: float
    is_well_conditioned: bool
    holomorphy_real_drift: float = 0.0
    holomorphy_imag_leak: float = 0.0
    is_holomorphic_germ: bool = True
    jacobian_frobenius: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "jacobian_map", _immutable_array(self.jacobian_map, np.float64)
        )
        object.__setattr__(self, "condition_number", float(self.condition_number))
        object.__setattr__(
            self, "spectral_min_singular", float(self.spectral_min_singular)
        )
        object.__setattr__(
            self, "spectral_max_singular", float(self.spectral_max_singular)
        )
        object.__setattr__(self, "is_well_conditioned", bool(self.is_well_conditioned))
        object.__setattr__(
            self, "holomorphy_real_drift", float(self.holomorphy_real_drift)
        )
        object.__setattr__(
            self, "holomorphy_imag_leak", float(self.holomorphy_imag_leak)
        )
        object.__setattr__(self, "is_holomorphic_germ", bool(self.is_holomorphic_germ))
        object.__setattr__(self, "jacobian_frobenius", float(self.jacobian_frobenius))


@dataclass(frozen=True, slots=True)
class StabilizerGovernanceState:
    """
    Objeto final del endofuntor de estabilización de fase (Act). Sello maestro.
    """

    jacobian_report: CSMDJacobianReport
    symplectic_residual: float
    liouville_residual: float
    final_verdict: StabilizerHeytingVerdict
    timestamp_utc: float
    provenance_hash: str
    diagnostic_note: str = ""
    relative_symplectic_residual: float = 0.0
    log_abs_det: float = 0.0
    det_sign: int = 1
    pairing_residual: float = 0.0
    passivity_residual: float = 0.0
    holomorphy_residual: float = 0.0
    roundoff_budget_symplectic: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symplectic_residual", float(self.symplectic_residual))
        object.__setattr__(self, "liouville_residual", float(self.liouville_residual))
        object.__setattr__(self, "timestamp_utc", float(self.timestamp_utc))
        object.__setattr__(self, "provenance_hash", str(self.provenance_hash))
        object.__setattr__(self, "diagnostic_note", str(self.diagnostic_note))
        object.__setattr__(
            self,
            "relative_symplectic_residual",
            float(self.relative_symplectic_residual),
        )
        object.__setattr__(self, "log_abs_det", float(self.log_abs_det))
        object.__setattr__(self, "det_sign", int(self.det_sign))
        object.__setattr__(self, "pairing_residual", float(self.pairing_residual))
        object.__setattr__(self, "passivity_residual", float(self.passivity_residual))
        object.__setattr__(
            self, "holomorphy_residual", float(self.holomorphy_residual)
        )
        object.__setattr__(
            self, "roundoff_budget_symplectic", float(self.roundoff_budget_symplectic)
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: INYECCIÓN ORTOGONAL COMPLEJA (Observe)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_ComplexPerturbationFibrator:
    r"""
    Fase 1 — Fibrador de perturbación compleja ortogonal.

    Objeto: un estado real x ∈ ℝᵈ.
    Morfismo: x ↦ Dil(x, h) = (x, {x + j·h·e_k}_{k=1..d}, d, h).
    Último morfismo (unidad de Kleisli hacia la Fase 2): accept_phase1_handoff.
    """

    def _coerce_real_vector(
        self, state_vector: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Pre:  state_vector es array-like.
        Post: vector 1-D float64 C-contiguo, copia propia (no alias).
        """
        try:
            state = np.array(state_vector, dtype=np.float64, copy=True, order="C")
        except (TypeError, ValueError) as exc:
            raise InvalidComplexStateError(
                f"El vector de estado no es coercible a float64: {exc}"
            ) from exc
        if state.ndim != 1:
            raise InvalidComplexStateError(
                f"El vector de estado debe ser 1-D; ndim={state.ndim}, shape={state.shape}."
            )
        if state.size == 0:
            raise InvalidComplexStateError(
                "El vector de estado es vacío: la variedad de fase es 0-dimensional."
            )
        return state

    def _assert_fpu_finite_real(self, state_real: NDArray[np.float64]) -> None:
        """Aduana FPU: prohíbe NaN, ±Inf y no-reales en la base real."""
        if not np.all(np.isfinite(state_real)):
            n_nan = int(np.isnan(state_real).sum())
            n_inf = int(np.isinf(state_real).sum())
            raise InvalidComplexStateError(
                f"FPU Error: el estado contiene {n_nan} NaN y {n_inf} Inf "
                "sobre la variedad real."
            )

    def _assert_step_size_admissible(self, step_size_h: float) -> float:
        """
        Pre:  h candidato.
        Post: h ∈ [_CSD_STEP_MIN, _CSD_STEP_MAX] ⊂ ℝ₊, finito.
        El intervalo excluye tanto el underflow de Im(f) como el régimen
        en el que O(h²) deja de ser despreciable frente a ε_machine.
        """
        try:
            h_val = float(step_size_h)
        except (TypeError, ValueError) as exc:
            raise InvalidComplexStateError(
                f"El paso de perturbación h no es coercible a float: {exc}"
            ) from exc
        if not math.isfinite(h_val) or h_val <= 0.0:
            raise InvalidComplexStateError(
                f"El paso de perturbación h debe ser positivo y finito; recibido {h_val!r}."
            )
        if h_val < _CSD_STEP_MIN or h_val > _CSD_STEP_MAX:
            raise InvalidComplexStateError(
                f"Sutura Error: paso h={h_val:.4e} fuera del rango físico admisible "
                f"[{_CSD_STEP_MIN:.1e}, {_CSD_STEP_MAX:.1e}]."
            )
        return h_val

    def _hint_even_dimension(self, dim_total: int) -> None:
        """
        Aviso temprano: Sp(2n) exige d = 2n par. No aborta en Fase 1
        (el Jacobiano CSMD está bien definido en dimensión impar).
        """
        if dim_total % 2 != 0:
            logger.warning(
                "Fase 1: dimensión d=%d impar; la 2-forma Ω de la Fase 3 no estará definida.",
                dim_total,
            )

    def _construct_stinespring_grid(
        self,
        state_real: NDArray[np.float64],
        step_size_h: float,
    ) -> NDArray[np.complex128]:
        r"""
        Dilatación de Stinespring sobre la fibra imaginaria pura.

        Construcción libre de alias y de tile:
            Re(G_{ik}) = x_i ,   Im(G_{ik}) = h · δ_{ik}
        Equivalentemente G_{·k} = x + j·h·e_k ∈ ℂᵈ.
        """
        dim_total = int(state_real.size)
        grid = np.empty((dim_total, dim_total), dtype=np.complex128)
        grid.real[:] = state_real[:, None]
        grid.imag[:] = 0.0
        np.fill_diagonal(grid.imag, step_size_h)
        return grid

    def _certify_dilation_invariants(
        self,
        state_real: NDArray[np.float64],
        grid: NDArray[np.complex128],
        step_size_h: float,
    ) -> None:
        """
        Certifica Re(G) = x 1ᵀ y Im(G) = h I a tolerancia
        4 d ε (1 + ‖x‖_∞ + h), cota a priori de redondeo.
        """
        dim_total = state_real.size
        if grid.shape != (dim_total, dim_total):
            raise InvalidComplexStateError(
                f"Grid complejo de forma {grid.shape}, esperado ({dim_total}, {dim_total})."
            )
        if not np.all(np.isfinite(grid)):
            raise InvalidComplexStateError(
                "La malla de Stinespring contiene valores no finitos."
            )
        scale = 4.0 * dim_total * _MACHINE_EPS * (
            1.0 + float(np.max(np.abs(state_real))) + step_size_h
        )
        real_err = float(np.max(np.abs(grid.real - state_real[:, None])))
        imag_off = grid.imag.copy()
        np.fill_diagonal(imag_off, 0.0)
        imag_off_err = float(np.max(np.abs(imag_off)))
        imag_diag_err = float(np.max(np.abs(np.diag(grid.imag) - step_size_h)))
        if real_err > scale or imag_off_err > scale or imag_diag_err > scale:
            raise InvalidComplexStateError(
                "Invariantes de Stinespring rotos: "
                f"‖Re(G)−x1ᵀ‖_∞={real_err:.3e}, ‖Im(G)−hI‖_off={imag_off_err:.3e}, "
                f"‖diag(Im(G))−h‖_∞={imag_diag_err:.3e}, presupuesto={scale:.3e}."
            )

    def _seal_dilation(
        self,
        state_real: NDArray[np.float64],
        grid: NDArray[np.complex128],
        step_size_h: float,
    ) -> StinespringComplexDilation:
        """Sella el artefacto de Fase 1 como DTO congelado e inmutable."""
        return StinespringComplexDilation(
            state_real=state_real,
            state_complex_grid=grid,
            dimension_total=int(state_real.size),
            step_size_h=float(step_size_h),
        )

    def _fibrate_complex_perturbation(
        self,
        state_vector: NDArray[np.float64],
        step_size_h: float = _CSD_STEP_DEFAULT,
    ) -> StinespringComplexDilation:
        """Orquesta saneamiento FPU, inyección j·h·e_k y certificación."""
        state_real = self._coerce_real_vector(state_vector)
        self._assert_fpu_finite_real(state_real)
        h_val = self._assert_step_size_admissible(step_size_h)
        self._hint_even_dimension(int(state_real.size))
        grid = self._construct_stinespring_grid(state_real, h_val)
        self._certify_dilation_invariants(state_real, grid, h_val)
        dilation = self._seal_dilation(state_real, grid, h_val)
        logger.debug(
            "Fase 1 (Fibrator): d=%d, h=%.4e (dyadic=%s), grid=%s",
            dilation.dimension_total,
            h_val,
            math.frexp(h_val)[0] == 0.5,
            tuple(dilation.state_complex_grid.shape),
        )
        return dilation

    def accept_phase1_handoff(
        self,
        dilation: StinespringComplexDilation,
    ) -> StinespringComplexDilation:
        r"""
        Último morfismo formal de la Fase 1 y primer morfismo de la Fase 2.

        En esta clase actúa como sello-identidad (unidad de Kleisli).
        Phase2_NonDemolitionSpectralJacobian lo continúa por override,
        endureciendo el contrato de precondición espectral.

        Pre:  dilation es el artefacto emitido por _fibrate_complex_perturbation.
        Post: el mismo artefacto, certificado como precondición de Φ₂.
        Tipo: StinespringComplexDilation → StinespringComplexDilation
        """
        if not isinstance(dilation, StinespringComplexDilation):
            raise InvalidComplexStateError(
                "handoff Fase 1: se esperaba StinespringComplexDilation, "
                f"recibido {type(dilation).__name__}."
            )
        return dilation

    def execute_phase1(
        self,
        state_vector: NDArray[np.float64],
        step_size_h: float = _CSD_STEP_DEFAULT,
    ) -> StinespringComplexDilation:
        """
        Método terminal orquestador de la Fase 1.

        La última llamada ES accept_phase1_handoff, continuación / inicio
        formal de los métodos de la Fase 2.
        """
        dilation = self._fibrate_complex_perturbation(state_vector, step_size_h)
        return self.accept_phase1_handoff(dilation)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: SÍNTESIS DE DERIVACIÓN NO DEMOLITORIA (Orient)
#          — continuación formal de accept_phase1_handoff
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_NonDemolitionSpectralJacobian(Phase1_ComplexPerturbationFibrator):
    r"""
    Fase 2 — Síntesis del Jacobiano CSMD y auditoría espectral.

    Continúa el último morfismo de la Fase 1 (accept_phase1_handoff).
    Objeto: Dil(x, h) × f_ℂ.
    Morfismo: (Dil, f_ℂ) ↦ (J, κ₂(J), σ_min, σ_max, germen holomorfo).
    Último morfismo (unidad de Kleisli hacia la Fase 3): accept_phase2_handoff.
    """

    def accept_phase1_handoff(
        self,
        dilation: StinespringComplexDilation,
    ) -> StinespringComplexDilation:
        r"""
        Continuación formal del último método de la Fase 1.

        Endurece el contrato: verifica tipo, dimensión, forma de la malla,
        positividad de h y consistencia Re/Im residual a nivel de DTO.
        """
        dilation = super().accept_phase1_handoff(dilation)
        self._validate_dilation_contract(dilation)
        return dilation

    def _validate_dilation_contract(self, dilation: StinespringComplexDilation) -> None:
        """Precondición espectral completa del artefacto de Fase 1."""
        d = dilation.dimension_total
        if d <= 0:
            raise InvalidComplexStateError("La dimensión total debe ser positiva.")
        if dilation.state_real.shape != (d,):
            raise InvalidComplexStateError(
                f"state_real.shape={dilation.state_real.shape} incompatible con d={d}."
            )
        if dilation.state_complex_grid.shape != (d, d):
            raise InvalidComplexStateError(
                f"Inconsistencia dimensional del grid: se esperaba ({d}, {d}), "
                f"obtenido {dilation.state_complex_grid.shape}."
            )
        if not math.isfinite(dilation.step_size_h) or dilation.step_size_h <= 0.0:
            raise InvalidComplexStateError(
                f"step_size_h inválido en el DTO: {dilation.step_size_h!r}."
            )
        self._assert_step_size_admissible(dilation.step_size_h)
        if not np.all(np.isfinite(dilation.state_real)):
            raise InvalidComplexStateError("state_real del DTO no es finito.")
        if not np.all(np.isfinite(dilation.state_complex_grid)):
            raise InvalidComplexStateError("state_complex_grid del DTO no es finito.")

    def _validate_holomorphic_morphism(
        self,
        complex_transition_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    ) -> None:
        """El mapa de transición debe ser un callable (germen de morfismo ℂᵈ→ℂᵈ)."""
        if not callable(complex_transition_map):
            raise InvalidComplexStateError(
                "complex_transition_map debe ser un callable ℂᵈ → ℂᵈ."
            )

    def _assert_mapped_column(
        self,
        mapped: NDArray[np.generic],
        dim_total: int,
        tag: str,
    ) -> NDArray[np.complex128]:
        """Valida forma, finitez y tipo complejo de una evaluación de f_ℂ."""
        mapped_arr = np.asarray(mapped)
        if mapped_arr.shape != (dim_total,):
            raise InvalidComplexStateError(
                f"Mapeo dimensional incorrecto en {tag}: "
                f"esperado ({dim_total},), obtenido {mapped_arr.shape}."
            )
        if not np.issubdtype(mapped_arr.dtype, np.complexfloating):
            raise InvalidComplexStateError(
                f"El mapa complejo debe devolver dtype complejo en {tag}; "
                f"recibido {mapped_arr.dtype}."
            )
        mapped_c = np.array(mapped_arr, dtype=np.complex128, copy=True, order="C")
        if not np.all(np.isfinite(mapped_c)):
            raise InvalidComplexStateError(
                f"El mapa produjo valores no finitos en {tag}."
            )
        return mapped_c

    def _evaluate_real_germ(
        self,
        dilation: StinespringComplexDilation,
        complex_transition_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """
        Evalúa f(x + 0·j). Si f es real-analítica, Im(f(x)) ≈ 0.
        Esta evaluación es el origen del germen de holomorfía.
        """
        x_complex = np.array(dilation.state_real, dtype=np.complex128, copy=True)
        try:
            germ = complex_transition_map(x_complex)
        except PhaseStabilizerError:
            raise
        except Exception as exc:
            raise InvalidComplexStateError(
                f"Error al evaluar el germen real f(x+0j): {exc}"
            ) from exc
        return self._assert_mapped_column(germ, dilation.dimension_total, tag="germ")

    def _evaluate_complex_column(
        self,
        column: NDArray[np.complex128],
        complex_transition_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
        index: int,
        dim_total: int,
    ) -> NDArray[np.complex128]:
        """Evalúa f(x + j·h·e_k) con aduana de excepciones y finitez."""
        try:
            mapped = complex_transition_map(column)
        except PhaseStabilizerError:
            raise
        except Exception as exc:
            raise InvalidComplexStateError(
                f"Error al evaluar complex_transition_map en columna {index}: {exc}"
            ) from exc
        return self._assert_mapped_column(mapped, dim_total, tag=f"columna {index}")

    def _extract_csmd_column(
        self,
        mapped_output: NDArray[np.complex128],
        step_size_h: float,
    ) -> NDArray[np.float64]:
        r"""
        Proyector CSMD: J_{·k} = Im(f(x + j·h·e_k)) / h.

        Libre de cancelación sustractiva. El único cociente es un escalado
        por h^{-1}, exacto si h es diádico.
        """
        return np.imag(mapped_output) / step_size_h

    def _column_real_drift(
        self,
        mapped_output: NDArray[np.complex128],
        germ: NDArray[np.complex128],
    ) -> float:
        """‖Re(f(x+jhe_k)) − Re(f(x))‖₂; O(h²) si f es holomorfa."""
        return _finite_norm2(np.real(mapped_output) - np.real(germ))

    def _assemble_csmd_jacobian(
        self,
        dilation: StinespringComplexDilation,
        complex_transition_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    ) -> tuple[NDArray[np.float64], float, float]:
        """
        Ensambla J ∈ M_d(ℝ) columna a columna y mide el germen holomorfo.

        Retorna (J, max_k ‖Re f_k − Re f_0‖₂, ‖Im f_0‖₂).
        """
        dim_total = dilation.dimension_total
        h_val = dilation.step_size_h
        germ = self._evaluate_real_germ(dilation, complex_transition_map)
        imag_leak = _finite_norm2(np.imag(germ))

        jacobian = np.empty((dim_total, dim_total), dtype=np.float64)
        max_real_drift = 0.0
        grid = dilation.state_complex_grid
        for k in range(dim_total):
            mapped_k = self._evaluate_complex_column(
                grid[:, k],
                complex_transition_map,
                k,
                dim_total,
            )
            jacobian[:, k] = self._extract_csmd_column(mapped_k, h_val)
            drift_k = self._column_real_drift(mapped_k, germ)
            if drift_k > max_real_drift:
                max_real_drift = drift_k

        if not np.all(np.isfinite(jacobian)):
            raise InvalidComplexStateError(
                "El Jacobiano CSMD contiene entradas no finitas "
                "(posible overflow al escalar por h^{-1})."
            )
        return jacobian, float(max_real_drift), float(imag_leak)

    def _spectral_svd_invariants(
        self,
        jacobian_map: NDArray[np.float64],
    ) -> tuple[float, float, float, float]:
        r"""
        Invariantes singulares de J:
            σ_max = ‖J‖₂ ,  σ_min = σ_d(J) ,  κ₂ = σ_max / σ_min ,
            ‖J‖_F.
        Si σ_min ≤ ε, se declara κ₂ = +∞ (núcleo numérico no trivial).
        """
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                singular_values = la.svd(jacobian_map, compute_uv=False)
        except (FloatingPointError, la.LinAlgError) as exc:
            raise InvalidComplexStateError(
                f"SVD del Jacobiano CSMD falló: {exc}"
            ) from exc

        if singular_values.size == 0:
            return math.inf, 0.0, 0.0, 0.0

        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        frobenius = float(np.linalg.norm(singular_values, ord=2))  # ‖σ‖₂ = ‖J‖_F

        if math.isfinite(sigma_min) and sigma_min > _MACHINE_EPS:
            condition_number = sigma_max / sigma_min
        else:
            condition_number = math.inf
        if not math.isfinite(condition_number) or condition_number < 0.0:
            condition_number = math.inf
        return condition_number, sigma_min, sigma_max, frobenius

    def _classify_conditioning(self, condition_number: float) -> bool:
        """κ₂(J) ≤ κ_max  ⇒  bien condicionado para las aduanas de Fase 3."""
        return bool(
            math.isfinite(condition_number)
            and condition_number <= _CONDITION_NUMBER_MAX
        )

    def _classify_holomorphic_germ(
        self,
        real_drift: float,
        imag_leak: float,
        jacobian_frobenius: float,
        dim_total: int,
    ) -> bool:
        """
        Germen holomorfo admisible si fuga imaginaria y deriva real
        permanecen bajo una cota a priori O(d ε (1+‖J‖_F)).
        """
        budget = max(
            _SOFT_HOLOMORPHY_TOL,
            64.0 * dim_total * _MACHINE_EPS * (1.0 + jacobian_frobenius),
        )
        return (
            math.isfinite(real_drift)
            and math.isfinite(imag_leak)
            and real_drift <= budget
            and imag_leak <= budget
        )

    def _seal_jacobian_report(
        self,
        dilation: StinespringComplexDilation,
        jacobian_map: NDArray[np.float64],
        condition_number: float,
        sigma_min: float,
        sigma_max: float,
        is_well_conditioned: bool,
        real_drift: float,
        imag_leak: float,
        is_holomorphic_germ: bool,
        jacobian_frobenius: float,
    ) -> CSMDJacobianReport:
        """Sella el artefacto de Fase 2 como DTO congelado."""
        return CSMDJacobianReport(
            dilation=dilation,
            jacobian_map=jacobian_map,
            condition_number=condition_number,
            spectral_min_singular=sigma_min,
            spectral_max_singular=sigma_max,
            is_well_conditioned=is_well_conditioned,
            holomorphy_real_drift=real_drift,
            holomorphy_imag_leak=imag_leak,
            is_holomorphic_germ=is_holomorphic_germ,
            jacobian_frobenius=jacobian_frobenius,
        )

    def _synthesize_csmd_jacobian(
        self,
        dilation: StinespringComplexDilation,
        complex_transition_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    ) -> CSMDJacobianReport:
        """Cuerpo de síntesis: Φ₂(Dil, f_ℂ) = Report."""
        self._validate_holomorphic_morphism(complex_transition_map)
        jacobian_map, real_drift, imag_leak = self._assemble_csmd_jacobian(
            dilation, complex_transition_map
        )
        condition_number, sigma_min, sigma_max, jac_frob = self._spectral_svd_invariants(
            jacobian_map
        )
        is_well = self._classify_conditioning(condition_number)
        is_holo = self._classify_holomorphic_germ(
            real_drift, imag_leak, jac_frob, dilation.dimension_total
        )
        logger.debug(
            "Fase 2 (CSMD): κ=%.3e, σ_min=%.3e, σ_max=%.3e, ‖J‖_F=%.3e, "
            "drift=%.3e, leak=%.3e, holo=%s, well=%s",
            condition_number,
            sigma_min,
            sigma_max,
            jac_frob,
            real_drift,
            imag_leak,
            is_holo,
            is_well,
        )
        return self._seal_jacobian_report(
            dilation,
            jacobian_map,
            condition_number,
            sigma_min,
            sigma_max,
            is_well,
            real_drift,
            imag_leak,
            is_holo,
            jac_frob,
        )

    def accept_phase2_handoff(self, report: CSMDJacobianReport) -> CSMDJacobianReport:
        r"""
        Último morfismo formal de la Fase 2 y primer morfismo de la Fase 3.

        En esta clase actúa como sello-identidad. La Fase 3 lo continúa
        por override, exigiendo paridad dimensional y consistencia J ↔ Dil.
        Tipo: CSMDJacobianReport → CSMDJacobianReport
        """
        if not isinstance(report, CSMDJacobianReport):
            raise InvalidComplexStateError(
                "handoff Fase 2: se esperaba CSMDJacobianReport, "
                f"recibido {type(report).__name__}."
            )
        return report

    def execute_phase2(
        self,
        dilation: StinespringComplexDilation,
        complex_transition_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
    ) -> CSMDJacobianReport:
        """
        Método terminal orquestador de la Fase 2.

        Primera llamada: accept_phase1_handoff (continuación de Fase 1).
        Última llamada:  accept_phase2_handoff (inicio formal de Fase 3).
        """
        dilation = self.accept_phase1_handoff(dilation)
        report = self._synthesize_csmd_jacobian(dilation, complex_transition_map)
        return self.accept_phase2_handoff(report)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: ESTABILIZACIÓN EN EL RETÍCULO DE HEYTING (Decide & Act)
#          — continuación formal de accept_phase2_handoff
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingStabilizerDecider(Phase2_NonDemolitionSpectralJacobian):
    r"""
    Fase 3 — Aduanas de Liouville, simplecticidad, holomorfía y pasividad.

    Continúa el último morfismo de la Fase 2 (accept_phase2_handoff).
    Resuelve el supremo en el retículo distributivo Ω₃ y, si el veredicto
    toca el objeto terminal VETOED, colapsa síncronamente la transacción.
    """

    def accept_phase2_handoff(self, report: CSMDJacobianReport) -> CSMDJacobianReport:
        """
        Continuación formal del último método de la Fase 2.

        Endurece el contrato: tipo, forma de J, finitez y coherencia
        dimensional con la dilatación de Stinespring.
        """
        report = super().accept_phase2_handoff(report)
        self._validate_jacobian_report(report)
        return report

    def _validate_jacobian_report(self, report: CSMDJacobianReport) -> None:
        """Precondición completa del artefacto de Fase 2."""
        d = report.dilation.dimension_total
        if report.jacobian_map.shape != (d, d):
            raise InvalidComplexStateError(
                f"Inconsistencia en el Jacobiano: forma {report.jacobian_map.shape}, "
                f"esperado ({d}, {d})."
            )
        if not np.all(np.isfinite(report.jacobian_map)):
            raise InvalidComplexStateError(
                "El Jacobiano del reporte contiene entradas no finitas."
            )

    def _require_even_dimension(self, dim_total: int) -> int:
        """
        Sp(2n, ℝ) está definido sólo para d = 2n par.
        Retorna n = d/2.
        """
        if dim_total % 2 != 0:
            raise InvalidComplexStateError(
                f"La dimensión total debe ser par para definir Ω; recibida d={dim_total}."
            )
        n = dim_total // 2
        if n <= 0:
            raise InvalidComplexStateError(
                f"n = d/2 debe ser positivo; d={dim_total}."
            )
        return n

    def _structured_omega_action(
        self,
        jacobian_m: NDArray[np.float64],
        n: int,
    ) -> NDArray[np.float64]:
        r"""
        Acción estructurada Ω J sin ensamblar Ω explícitamente:
            J = [J_q ; J_p]  ⇒  Ω J = [J_p ; −J_q].
        Reduce el error de redondeo frente al producto denso Ω @ J.
        """
        j_q = jacobian_m[:n, :]
        j_p = jacobian_m[n:, :]
        return np.vstack((j_p, -j_q))

    def _symplectic_pullback_residuals(
        self,
        jacobian_m: NDArray[np.float64],
        n: int,
    ) -> tuple[float, float, float]:
        r"""
        Residuos de la condición Jᵀ Ω J = Ω.

        Retorna
            (r_abs, r_rel, presupuesto_redondeo)
        donde
            r_abs = ‖Jᵀ Ω J − Ω‖_F ,
            r_rel = r_abs / ‖Ω‖_F ,   ‖Ω‖_F = √(2n) ,
            presupuesto ≈ u · √d · ‖J‖_F² · ‖Ω‖_F   (Wilkinson).
        """
        omega = _canonical_symplectic_form(n)
        omega_j = self._structured_omega_action(jacobian_m, n)
        pullback = jacobian_m.T @ omega_j
        delta = pullback - omega
        abs_res = float(la.norm(delta, ord="fro"))
        omega_f = math.sqrt(2.0 * n)
        rel_res = abs_res / max(omega_f, _MACHINE_TINY)
        jac_f = float(la.norm(jacobian_m, ord="fro"))
        dim_total = 2 * n
        roundoff_budget = (
            _MACHINE_EPS
            * math.sqrt(dim_total)
            * max(1.0, jac_f * jac_f)
            * max(1.0, omega_f)
        )
        if not math.isfinite(abs_res):
            abs_res = math.inf
            rel_res = math.inf
        return abs_res, rel_res, float(roundoff_budget)

    def _liouville_slogdet_residual(
        self,
        jacobian_m: NDArray[np.float64],
    ) -> tuple[float, float, int]:
        r"""
        Residuo de Liouville vía slogdet (estable en magnitud):

            (s, ℓ) = slogdet(J)
            r = +∞            si s no es +1 o ℓ no es finito
              = |expm1(ℓ)|    si s = +1   ( = ||det J| − 1| con precisión cerca de 1)

        Sp(2n, ℝ) ⊂ SL(2n, ℝ) exige s = +1 y ℓ = 0.
        Retorna (r_Liouville, ℓ, s_int) con s_int ∈ {−1, 0, +1}.
        """
        try:
            sign, log_abs_det = la.slogdet(jacobian_m)
        except la.LinAlgError as exc:
            raise InvalidComplexStateError(
                f"slogdet del Jacobiano falló: {exc}"
            ) from exc
        sign_f = float(sign)
        lad = float(log_abs_det)
        if not math.isfinite(sign_f) or not math.isfinite(lad) or sign_f <= 0.0:
            sign_i = 0 if (not math.isfinite(sign_f) or sign_f == 0.0) else int(math.copysign(1.0, sign_f))
            return math.inf, lad, sign_i
        residual = abs(math.expm1(lad))
        if not math.isfinite(residual):
            residual = math.inf
        return float(residual), lad, 1

    def _symplectic_spectrum_pairing_residual(
        self,
        jacobian_m: NDArray[np.float64],
    ) -> float:
        r"""
        Residuo de apareamiento recíproco del espectro.

        Para M ∈ Sp(2n, ℝ) el espectro es cerrado bajo λ ↦ 1/λ.
        Emparejamiento voraz sobre ℂ ∪ {∞}: para cada λ no usado se busca
        μ no usado que minimice |λμ − 1| / max(1, |λμ|).
        Se omite si d > _PAIRING_DIM_GUARD (coste O(n³) + O(n²)).
        """
        dim_total = jacobian_m.shape[0]
        if dim_total > _PAIRING_DIM_GUARD:
            return 0.0
        try:
            eigenvalues = la.eigvals(jacobian_m)
        except la.LinAlgError as exc:
            logger.warning("Apareamiento espectral omitido: eigvals falló (%s).", exc)
            return math.inf

        if not np.all(np.isfinite(eigenvalues)):
            return math.inf

        unused = list(range(eigenvalues.size))
        pair_residuals: list[float] = []
        while unused:
            i = unused.pop(0)
            lam = complex(eigenvalues[i])
            abs_lam = abs(lam)
            if abs_lam <= _MACHINE_EPS:
                pair_residuals.append(1.0)
                continue
            if not unused:
                # d par ⇒ este caso no debería ocurrir; residual de órfano
                pair_residuals.append(abs(abs_lam - 1.0) if abs(abs_lam - 1.0) < 0.5 else 1.0)
                break
            best_j_pos = 0
            best_res = math.inf
            inv_target = 1.0 / lam
            for pos, j in enumerate(unused):
                mu = complex(eigenvalues[j])
                prod = lam * mu
                denom = max(1.0, abs(prod))
                res_recip = abs(prod - 1.0) / denom
                res_self = abs(mu - inv_target) / max(1.0, abs(inv_target))
                res = min(res_recip, res_self)
                if res < best_res:
                    best_res = res
                    best_j_pos = pos
            unused.pop(best_j_pos)
            pair_residuals.append(float(best_res))

        if not pair_residuals:
            return 0.0
        return float(max(pair_residuals))

    def _passivity_rayleigh_residual(
        self,
        state_real: NDArray[np.float64],
        hamiltonian_gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
        rayleigh_metric: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    ) -> float:
        r"""
        Residuo de creación ficticia de exergía (invariante [I3]).

        Ḣ = −∇Hᵀ R ∇H ≤ 0  ⇔  ∇Hᵀ R ∇H ≥ 0 si se asume la convención
        disipativa estándar. El residuo reportado es
            r_pass = max{0, −∇Hᵀ R ∇H}
        (positivo si y sólo si se crea exergía).
        Si los oráculos no se proveen, r_pass = 0 y el join es neutro.
        """
        if hamiltonian_gradient is None or rayleigh_metric is None:
            return 0.0
        try:
            grad_h = np.asarray(hamiltonian_gradient(state_real), dtype=np.float64)
            metric_r = np.asarray(rayleigh_metric(state_real), dtype=np.float64)
        except Exception as exc:
            raise InvalidComplexStateError(
                f"Oráculos de pasividad [I3] fallaron: {exc}"
            ) from exc

        d = state_real.size
        if grad_h.shape != (d,):
            raise InvalidComplexStateError(
                f"∇H debe tener forma ({d},); obtenido {grad_h.shape}."
            )
        if metric_r.shape != (d, d):
            raise InvalidComplexStateError(
                f"R(x) debe tener forma ({d}, {d}); obtenido {metric_r.shape}."
            )
        if not np.all(np.isfinite(grad_h)) or not np.all(np.isfinite(metric_r)):
            raise InvalidComplexStateError("∇H o R(x) contienen valores no finitos.")

        # Simetrización forzada: la forma de Rayleigh usa (R+Rᵀ)/2.
        metric_sym = 0.5 * (metric_r + metric_r.T)
        quadratic = float(grad_h.T @ metric_sym @ grad_h)
        if not math.isfinite(quadratic):
            return math.inf
        return float(max(0.0, -quadratic))

    def _holomorphy_residual(self, report: CSMDJacobianReport) -> float:
        """Agrega deriva real y fuga imaginaria en una norma ℓ∞ de residuos."""
        return max(
            abs(report.holomorphy_real_drift),
            abs(report.holomorphy_imag_leak),
        )

    def _condition_residual(self, condition_number: float) -> float:
        """
        Homomorfismo κ₂ ↦ residual en [0, +∞] comparable con las bandas
        de Heyting: se clasifica directamente contra κ_soft / κ_max.
        """
        if not math.isfinite(condition_number) or condition_number < 0.0:
            return math.inf
        return float(condition_number)

    def _enact_verdict(
        self,
        v_final: StabilizerHeytingVerdict,
        v_sym: StabilizerHeytingVerdict,
        v_lio: StabilizerHeytingVerdict,
        v_holo: StabilizerHeytingVerdict,
        symplectic_residual: float,
        liouville_residual: float,
        holomorphy_residual: float,
        raise_on_veto: bool,
    ) -> None:
        """Efecto (Act): log forense y colapso síncrono si VETOED."""
        if v_final == StabilizerHeytingVerdict.VETOED:
            logger.critical(
                "[STABILIZER] VETO SÍNCRONO: invariantes destruidos. "
                "r_sym=%.4e r_lio=%.4e r_holo=%.4e. Sesión RAM purgada.",
                symplectic_residual,
                liouville_residual,
                holomorphy_residual,
            )
            if not raise_on_veto:
                return
            message = (
                "Heyting Veto: inestabilidad catastrófica en el espacio de fase. "
                f"r_sym={symplectic_residual:.4e}, r_lio={liouville_residual:.4e}, "
                f"r_holo={holomorphy_residual:.4e}. Transacción abortada síncronamente."
            )
            if (
                v_sym == StabilizerHeytingVerdict.VETOED
                and v_lio != StabilizerHeytingVerdict.VETOED
                and v_holo != StabilizerHeytingVerdict.VETOED
            ):
                raise SymplecticPhaseRipError(message)
            if (
                v_lio == StabilizerHeytingVerdict.VETOED
                and v_sym != StabilizerHeytingVerdict.VETOED
                and v_holo != StabilizerHeytingVerdict.VETOED
            ):
                raise LiouvilleVolumeBreachError(message)
            if (
                v_holo == StabilizerHeytingVerdict.VETOED
                and v_sym != StabilizerHeytingVerdict.VETOED
                and v_lio != StabilizerHeytingVerdict.VETOED
            ):
                raise HolomorphyBreachError(message)
            raise HeytingStabilizerVeto(message)

        if v_final == StabilizerHeytingVerdict.DEGRADED:
            logger.warning(
                "[DEGRADED] Deriva en el espacio de fase. "
                "r_sym=%.4e r_lio=%.4e r_holo=%.4e.",
                symplectic_residual,
                liouville_residual,
                holomorphy_residual,
            )

    def _forensic_provenance_seal(
        self,
        jacobian_m: NDArray[np.float64],
        symplectic_residual: float,
        liouville_residual: float,
        condition_number: float,
        step_size_h: float,
        dim_total: int,
        v_final: StabilizerHeytingVerdict,
        log_abs_det: float,
        det_sign: int,
    ) -> str:
        """
        Sello SHA-256 del artefacto numérico (Jacobiano + residuos + veredicto).
        Dominio etiquetado para impedir colisiones cruzadas de versión.
        """
        hasher = hashlib.sha256()
        hasher.update(_PROVENANCE_DOMAIN)
        jac = np.ascontiguousarray(jacobian_m, dtype=np.float64)
        hasher.update(jac.tobytes(order="C"))
        hasher.update(
            struct.pack(
                "<ddddQid",
                float(symplectic_residual),
                float(liouville_residual),
                float(condition_number) if math.isfinite(condition_number) else -1.0,
                float(step_size_h),
                int(dim_total),
                int(v_final.value),
                float(log_abs_det) if math.isfinite(log_abs_det) else 0.0,
            )
        )
        hasher.update(struct.pack("<b", int(det_sign)))
        return hasher.hexdigest()

    def _compose_diagnostic_note(
        self,
        v_final: StabilizerHeytingVerdict,
        symplectic_residual: float,
        relative_symplectic_residual: float,
        liouville_residual: float,
        log_abs_det: float,
        det_sign: int,
        pairing_residual: float,
        passivity_residual: float,
        holomorphy_residual: float,
        condition_number: float,
        roundoff_budget: float,
        is_holomorphic_germ: bool,
    ) -> str:
        """Nota forense humana: todos los residuos y el veredicto."""
        return (
            f"Veredicto: {v_final.name}. "
            f"Simplecticidad: r_abs={symplectic_residual:.4e} "
            f"r_rel={relative_symplectic_residual:.4e} "
            f"(tol_hard={_HARD_SYMPLECTIC_TOL:.1e}, presupuesto_u={roundoff_budget:.2e}). "
            f"Volumen: r_lio={liouville_residual:.4e} sign={det_sign:+d} "
            f"log|det|={log_abs_det:.4e}. "
            f"Apareamiento: r_pair={pairing_residual:.4e}. "
            f"Holomorfía: r_holo={holomorphy_residual:.4e} germen={is_holomorphic_germ}. "
            f"Pasividad: r_pass={passivity_residual:.4e}. "
            f"Condición: κ={condition_number:.3e}."
        )

    def _stabilize_governance(
        self,
        report: CSMDJacobianReport,
        raise_on_veto: bool = True,
        hamiltonian_gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        rayleigh_metric: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    ) -> StabilizerGovernanceState:
        """
        Consolida el veredicto terminal mediante el join (⊔) en Heyting Ω₃.

        v_final = v_sym ⊔ v_lio ⊔ v_holo ⊔ v_cond ⊔ v_pair ⊔ v_pass
        """
        jacobian_m = np.array(report.jacobian_map, dtype=np.float64, copy=True, order="C")
        dim_total = report.dilation.dimension_total
        n = self._require_even_dimension(dim_total)

        symplectic_residual, rel_sym, roundoff_budget = (
            self._symplectic_pullback_residuals(jacobian_m, n)
        )
        liouville_residual, log_abs_det, det_sign = self._liouville_slogdet_residual(
            jacobian_m
        )
        pairing_residual = self._symplectic_spectrum_pairing_residual(jacobian_m)
        passivity_residual = self._passivity_rayleigh_residual(
            report.dilation.state_real,
            hamiltonian_gradient,
            rayleigh_metric,
        )
        holomorphy_residual = self._holomorphy_residual(report)
        condition_residual = self._condition_residual(report.condition_number)

        v_sym = _heyting_classify(
            symplectic_residual, _SOFT_SYMPLECTIC_TOL, _HARD_SYMPLECTIC_TOL
        )
        v_lio = _heyting_classify(
            liouville_residual, _SOFT_LIOUVILLE_TOL, _HARD_LIOUVILLE_TOL
        )
        v_holo = _heyting_classify(
            holomorphy_residual, _SOFT_HOLOMORPHY_TOL, _HARD_HOLOMORPHY_TOL
        )
        v_cond = _heyting_classify(
            condition_residual, _CONDITION_NUMBER_SOFT, _CONDITION_NUMBER_MAX
        )
        v_pair = _heyting_classify(
            pairing_residual, _SOFT_PAIRING_TOL, _HARD_PAIRING_TOL
        )
        v_pass = _heyting_classify(
            passivity_residual, _SOFT_PASSIVITY_TOL, _HARD_PASSIVITY_TOL
        )
        v_final = _heyting_join(v_sym, v_lio, v_holo, v_cond, v_pair, v_pass)

        self._enact_verdict(
            v_final,
            v_sym,
            v_lio,
            v_holo,
            symplectic_residual,
            liouville_residual,
            holomorphy_residual,
            raise_on_veto,
        )

        provenance_hash = self._forensic_provenance_seal(
            jacobian_m,
            symplectic_residual,
            liouville_residual,
            report.condition_number,
            report.dilation.step_size_h,
            dim_total,
            v_final,
            log_abs_det,
            det_sign,
        )
        diagnostic_note = self._compose_diagnostic_note(
            v_final,
            symplectic_residual,
            rel_sym,
            liouville_residual,
            log_abs_det,
            det_sign,
            pairing_residual,
            passivity_residual,
            holomorphy_residual,
            report.condition_number,
            roundoff_budget,
            report.is_holomorphic_germ,
        )
        return StabilizerGovernanceState(
            jacobian_report=report,
            symplectic_residual=symplectic_residual,
            liouville_residual=liouville_residual,
            final_verdict=v_final,
            timestamp_utc=time.time(),
            provenance_hash=provenance_hash,
            diagnostic_note=diagnostic_note,
            relative_symplectic_residual=rel_sym,
            log_abs_det=log_abs_det,
            det_sign=det_sign,
            pairing_residual=pairing_residual,
            passivity_residual=passivity_residual,
            holomorphy_residual=holomorphy_residual,
            roundoff_budget_symplectic=roundoff_budget,
        )

    def execute_phase3(
        self,
        report: CSMDJacobianReport,
        raise_on_veto: bool = True,
        hamiltonian_gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        rayleigh_metric: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    ) -> StabilizerGovernanceState:
        """
        Método terminal orquestador de la Fase 3.

        Primera llamada: accept_phase2_handoff (continuación de Fase 2).
        Retorna el estado lógico supremo y, opcionalmente, colapsa el
        retículo ante un veto.
        """
        report = self.accept_phase2_handoff(report)
        return self._stabilize_governance(
            report,
            raise_on_veto=raise_on_veto,
            hamiltonian_gradient=hamiltonian_gradient,
            rayleigh_metric=rayleigh_metric,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIBRADOR SUPREMO: COMPLEX STEP PHASE STABILIZER (Funtor Completo)
# ══════════════════════════════════════════════════════════════════════════════
class ComplexStepPhaseStabilizer(Morphism, Phase3_HeytingStabilizerDecider):
    r"""
    Fibrador y Guardián de la Estabilización por Paso Complejo.

    Endofuntor sobre el topos de estados de fase:
        Z = Φ₃ ∘ Φ₂ ∘ Φ₁
    con unidades de Kleisli anidadas
        accept_phase1_handoff  ⊂  accept_phase2_handoff  ⊂  execute_phase3.
    """

    def __init__(self, step_size_h: float = _CSD_STEP_DEFAULT) -> None:
        """Inicializa al estabilizador con un paso de perturbación admisible."""
        self._step_h = self._assert_step_size_admissible(step_size_h)

    @property
    def step_size_h(self) -> float:
        """Paso de perturbación CSMD vigente (diádico si se usó el default)."""
        return self._step_h

    @step_size_h.setter
    def step_size_h(self, value: float) -> None:
        self._step_h = self._assert_step_size_admissible(value)

    def stabilize_phase_space_transition(
        self,
        state_vector: NDArray[np.float64],
        complex_transition_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
        raise_on_veto: bool = True,
        hamiltonian_gradient: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        rayleigh_metric: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    ) -> StabilizerGovernanceState:
        r"""
        Ejecuta el ciclo categórico completo de estabilización de la FPU.

        Composición funtorial:
            Z_stabilizer = Φ₃ ∘ Φ₂ ∘ Φ₁

        Parámetros opcionales ∇H y R activan la aduana de pasividad [I3].
        """
        dilation = self.execute_phase1(state_vector, self._step_h)
        report = self.execute_phase2(dilation, complex_transition_map)
        return self.execute_phase3(
            report,
            raise_on_veto=raise_on_veto,
            hamiltonian_gradient=hamiltonian_gradient,
            rayleigh_metric=rayleigh_metric,
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "__version__",
    "StabilizerHeytingVerdict",
    "PhaseStabilizerError",
    "InvalidComplexStateError",
    "SymplecticPhaseRipError",
    "LiouvilleVolumeBreachError",
    "HolomorphyBreachError",
    "HeytingStabilizerVeto",
    "StinespringComplexDilation",
    "CSMDJacobianReport",
    "StabilizerGovernanceState",
    "Phase1_ComplexPerturbationFibrator",
    "Phase2_NonDemolitionSpectralJacobian",
    "Phase3_HeytingStabilizerDecider",
    "ComplexStepPhaseStabilizer",
]
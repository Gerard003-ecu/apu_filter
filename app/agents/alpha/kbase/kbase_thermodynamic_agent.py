# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo : KBase Thermodynamic Agent (Asesor de Cimientos Financieros)         |
| Ruta   : app/agents/alpha/kbase/kbase_thermodynamic_agent.py                 |
| Versión: 5.0.0-Rigorous-Sheaf-Williamson-Boolean-Spectral-Passivity          |
+==============================================================================+

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA DIFERENCIAL
===============================================
Este módulo consagra el Foso Termodinámico del ecosistema (K_BASE). Actúa como
un Endofuntor Port-Hamiltoniano que gobierna la inercia, la capacitancia y la
fricción entrópica del modelo de negocio.

HAMILTONIANO BASAL (TRAS PULLBACK RIEMANNIANO)
=============================================
    H_BASE(q, p) = ½ qᵀ C̃_soc⁻¹ q + ½ pᵀ M̃_rec⁻¹ p

con
    C̃_soc = G_q C_soc G_qᵀ ,   M̃_rec = G_p M_rec G_pᵀ

ECUACIÓN DE ESTADO PORT-HAMILTONIANA
====================================
    ẋ = (J_BASE − R_cost) ∇H(x)
    Ḣ = −∇Hᵀ R_cost ∇H ≤ 0          (pasividad / 2ª Ley)

DISIPACIÓN DE RAYLEIGH
======================
    P_diss = ∇Hᵀ R_cost ∇H ≥ 0
    τ_diss ≈ 2H / P_diss             (constante de tiempo entrópica local)

FORMA NORMAL DE WILLIAMSON (MODOS CONSERVATIVOS)
================================================
    H^{1/2} = block_diag(C̃_soc^{-1/2}, M̃_rec^{-1/2})
    A_sym   = H^{1/2} J_BASE H^{1/2}   (real antisimétrica)
    ω_k     = valores singulares emparejados de A_sym
    E_0     = (ħ/2) Σ_k ω_k

COFRONTERA DE HAZ (COCADENA APILADA)
====================================
    δ_metric = block_diag(C̃_soc^{-1/2}, M̃_rec^{-1/2}) ∈ ℝ^{n×n}
    δ_diss   = R_cost^{+1/2}                          ∈ ℝ^{n×n}
    δ_BASE   = [δ_metric ; δ_diss]                    ∈ ℝ^{2n×n}
    Δ_BASE   = δ_BASEᵀ δ_BASE = ∇²H + R_cost          (SPD)

ESTRUCTURA DE FASES ANIDADAS (CONTINUIDAD FORMAL)
=================================================
    Phase1_MatrixTopology.build_topological_context()
        →  TopologicalContext
    Phase2_HamiltonianDynamics.__init__(context) / .synthesize_basal_state()
        →  BasalStateTensor
    Phase3_SheafProjection.__init__(context) / .export_stalk(state_x=[q;p])
        →  SheafStalk

Cada frontera de fase es un DTO inmutable (frozen dataclass).
"""

from __future__ import annotations

# =============================================================================
# Biblioteca estándar
# =============================================================================
import enum
import logging
from typing import Final, Optional, Tuple

# =============================================================================
# Álgebra numérica de alta precisión
# =============================================================================
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# =============================================================================
# Estructuras de datos inmutables
# =============================================================================
from dataclasses import dataclass

# =============================================================================
# Dependencias arquitectónicas del ecosistema APU Filter
# =============================================================================
try:
    from app.core.mic_algebra import CategoricalState, Morphism
except ImportError:
    class CategoricalState:  # type: ignore[no-redef]
        """Stub: estado categórico del ecosistema MIC."""

    class Morphism:  # type: ignore[no-redef]
        """Stub: morfismo funtorial del ecosistema MIC."""


# =============================================================================
# Logger y constantes globales
# =============================================================================
logger = logging.getLogger("MIC.Alpha.KBaseThermodynamicAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)


# =============================================================================
# SECCIÓN 0 — EXCEPCIONES TERMODINÁMICAS ESTRICTAS
# =============================================================================


class ThermodynamicBaseError(Exception):
    """
    Excepción categórica raíz para violaciones en el Estrato K_BASE.

    Toda excepción de este módulo hereda de esta clase, garantizando que
    los manejadores de nivel superior puedan capturar cualquier fallo
    termodinámico con un solo ``except ThermodynamicBaseError``.
    """


class DimensionMismatchError(ThermodynamicBaseError):
    """
    Lanzada cuando las dimensiones de las matrices constitutivas son
    inconsistentes con el espacio de fases (q, p) declarado, o cuando los
    tensores métricos G_q, G_p no coinciden con dim_q, dim_p.
    """


class CapacitanceDegeneracyError(ThermodynamicBaseError):
    """
    Lanzada cuando C̃_soc o M̃_rec (tras el pullback métrico) no son SPD
    incluso después de aplicar regularización de Tikhonov adaptativa.

    Diagnóstico incluye κ(A), λ_min y el jitter τ finalmente ensayado.
    """


class InertialFlybackError(ThermodynamicBaseError):
    """
    Lanzada cuando la inercia de recuperación genera un voltaje transitorio
    de Flyback que excede el límite de ruptura dieléctrica.

    Condición de disparo: ‖M̃_rec · (∂f/∂t)‖_∞ > V_breakdown
    """


class RayleighDissipationViolation(ThermodynamicBaseError):
    """
    Lanzada cuando el modelo disipativo indica entropía negativa (ganancia
    fantasma), violando la Segunda Ley de la Termodinámica.

    Condición de disparo: ∇Hᵀ R_cost ∇H < −tol  (P_diss < −tol < 0)
    """


class IllConditionedMatrixError(ThermodynamicBaseError):
    """
    Lanzada cuando el número de condición espectral κ(A) = λ_max/λ_min
    supera el umbral configurable, indicando cuasi-singularidad numérica.
    """


class MetricTensorSingularityError(ThermodynamicBaseError):
    """
    Lanzada cuando el tensor métrico Riemanniano G_q o G_p es singular o
    está mal condicionado, invalidando el pullback congruente
    Ã = G A Gᵀ requerido para absorber el estrés anisotrópico.
    """


class SheafCoboundaryError(ThermodynamicBaseError):
    """
    Lanzada cuando la cofrontera discreta δ_BASE no satisface la
    identidad de Hodge local δᵀδ = ∇²H + R_cost dentro de tolerancia
    de máquina escalada por ‖Δ_BASE‖_F.
    """


class StructuralConsistencyError(ThermodynamicBaseError):
    """
    Lanzada cuando la identidad algebraica exacta del sistema
    Port-Hamiltoniano, ∇Hᵀẋ ≡ −P_diss (porque ∇Hᵀ J ∇H ≡ 0 para J
    antisimétrica), se viola más allá del error de redondeo esperado.

    A diferencia de ``RayleighDissipationViolation`` (violación física de
    la 2ª Ley), esta excepción indica un **error de implementación o de
    cableado** entre J_base, R_cost y el gradiente ∇H.
    """


class PassivityCertificateError(ThermodynamicBaseError):
    """
    Lanzada cuando el certificado de pasividad Ḣ + P_diss ≈ 0 falla, o
    cuando la tasa de decaimiento entrópica es inconsistente con el
    Hamiltoniano y el campo vectorial.
    """


class WilliamsonNormalFormError(ThermodynamicBaseError):
    """
    Lanzada cuando el diagnóstico de Williamson falla: A_sym no es
    numéricamente antisimétrica, o el emparejamiento de valores
    singulares produce frecuencias no físicas (NaN / negativas).
    """


# =============================================================================
# SECCIÓN 1 — ÁLGEBRA DE BOOLE DE ESTABILIDAD
# =============================================================================


class StabilityFlags(enum.Flag):
    r"""
    Retícula Booleana de predicados de estabilidad termodinámica.

    ``enum.Flag`` provee de forma nativa una **álgebra de Boole** completa
    sobre el conjunto de predicados: conjunción (`&`), disyunción (`|`),
    complemento (`~`), elemento ínfimo ``NONE`` (0̄) y elemento supremo
    ``ALL`` (1̄), satisfaciendo las leyes de De Morgan, distributividad,
    absorción e idempotencia por construcción del tipo.

    Miembros
    --------
    ENERGY_NONNEGATIVE
        V(q) ≥ 0 ∧ K(p) ≥ 0 (garantizado por SPD, pero verificado).
    DISSIPATION_VALID
        P_diss ≥ 0 (Segunda Ley, verificado independientemente).
    FLYBACK_SAFE
        ‖V_fb‖_∞ ≤ margin · V_breakdown (margen de seguridad blando).
    STRUCTURAL_CONSISTENCY
        |∇Hᵀẋ + P_diss| ≤ tol (identidad algebraica exacta del PH-system).
    SPECTRAL_CONDITIONING_SOUND
        κ(C̃_soc), κ(M̃_rec) por debajo de la mitad de κ_max.
    PASSIVITY_CERTIFICATE
        |Ḣ_num + P_diss| ≤ tol (certificado de pasividad en caliente).
    EULER_HOMOGENEITY
        |q·∇_q H + p·∇_p H − 2H| ≤ tol (H exactamente cuadrática).
    """

    NONE = 0
    ENERGY_NONNEGATIVE = enum.auto()
    DISSIPATION_VALID = enum.auto()
    FLYBACK_SAFE = enum.auto()
    STRUCTURAL_CONSISTENCY = enum.auto()
    SPECTRAL_CONDITIONING_SOUND = enum.auto()
    PASSIVITY_CERTIFICATE = enum.auto()
    EULER_HOMOGENEITY = enum.auto()
    ALL = (
        ENERGY_NONNEGATIVE
        | DISSIPATION_VALID
        | FLYBACK_SAFE
        | STRUCTURAL_CONSISTENCY
        | SPECTRAL_CONDITIONING_SOUND
        | PASSIVITY_CERTIFICATE
        | EULER_HOMOGENEITY
    )


def describe_stability_flags(flags: StabilityFlags) -> str:
    """
    Serializa la retícula ``StabilityFlags`` a una cadena legible,
    listando explícitamente los predicados satisfechos y los violados
    (complemento relativo a ``StabilityFlags.ALL``).
    """
    atomic = [
        f
        for f in StabilityFlags
        if f not in (StabilityFlags.NONE, StabilityFlags.ALL)
    ]
    satisfied = [f.name for f in atomic if f in flags]
    violated = [f.name for f in atomic if f not in flags]
    return (
        f"SATISFECHOS={satisfied or ['ninguno']} | "
        f"VIOLADOS={violated or ['ninguno']} | "
        f"ESTABLE_TOTAL={flags == StabilityFlags.ALL}"
    )


# =============================================================================
# SECCIÓN 2 — ESTRUCTURAS INMUTABLES (DTOs TENSORIALES)
# =============================================================================


@dataclass(frozen=True, slots=True)
class TopologicalContext:
    r"""
    Contexto inmutable producido por la **Fase 1** (Topología Matricial,
    Métrica Riemanniana y Teoría Espectral).

    Continuidad formal
    ------------------
    Este DTO es el **único argumento** del constructor de
    ``Phase2_HamiltonianDynamics`` y de ``Phase3_SheafProjection``.
    Su emisión por ``Phase1_MatrixTopology.build_topological_context()``
    constituye la frontera Fase 1 → Fases 2/3.

    Atributos
    ----------
    L_C, L_M : factores de Cholesky de C̃_soc, M̃_rec.
    C_inv_sqrt, M_inv_sqrt : raíces de las inversas (precalculadas).
    C_tilde, M_tilde : matrices pulled-back (trazabilidad / Flyback exacto).
    R_cost, R_sqrt, J_base : disipación, raíz espectral e interconexión.
    G_q, G_p : tensores métricos Riemannianos aplicados.
    kappa_C, kappa_M, kappa_G_q, kappa_G_p : números de condición.
    epsilon_C, epsilon_M : jitter de Tikhonov aplicado (0 si no fue necesario).
    dim_q, dim_p : dimensiones de coordenadas y momentos.
    rank_R, spectral_gap_R, betti_0_R : diagnósticos de R_cost.
    spectral_gap_C, spectral_gap_M : gaps de C̃_soc, M̃_rec.
    pullback_amp_C, pullback_amp_M : amplificación de κ por el pullback:
        κ(Ã) / κ(A) (≈ 1 si G ≈ I; grande si G es anisotrópico agresivo).
    spectral_entropy_R : entropía de von Neumann del espectro de R_cost.
    """

    L_C: NDArray[np.float64]
    L_M: NDArray[np.float64]
    C_inv_sqrt: NDArray[np.float64]
    M_inv_sqrt: NDArray[np.float64]
    C_tilde: NDArray[np.float64]
    M_tilde: NDArray[np.float64]
    R_cost: NDArray[np.float64]
    R_sqrt: NDArray[np.float64]
    J_base: NDArray[np.float64]
    G_q: NDArray[np.float64]
    G_p: NDArray[np.float64]
    kappa_C: float
    kappa_M: float
    kappa_G_q: float
    kappa_G_p: float
    epsilon_C: float
    epsilon_M: float
    dim_q: int
    dim_p: int
    rank_R: int
    spectral_gap_R: float
    betti_0_R: int
    spectral_gap_C: float
    spectral_gap_M: float
    pullback_amp_C: float
    pullback_amp_M: float
    spectral_entropy_R: float


@dataclass(frozen=True, slots=True)
class BasalStateTensor:
    r"""
    Tensor inmutable del estado termodinámico completo del foso.

    Producido por la **Fase 2** (Dinámica Port-Hamiltoniana).

    Continuidad formal
    ------------------
    El vector de estado x = [q; p] (reconstruible desde la llamada) y el
    campo ``vector_field`` alimentan la proyección de la Fase 3. La
    frontera formal Fase 2 → Fase 3 es ``state_x = concatenate([q, p])``
    pasado a ``Phase3_SheafProjection.export_stalk``.

    Atributos
    ----------
    potential_energy, kinetic_energy, total_hamiltonian : V, K, H ≥ 0.
    dissipated_power : P_diss ≥ 0.
    flyback_voltage_norm : ‖M̃_rec · ∂f/∂t‖_∞.
    grad_H_norm : ‖∇H‖₂.
    vector_field : ẋ = (J − R) ∇H.
    euler_homogeneity_residual : |q·∇_q H + p·∇_p H − 2H|.
    structural_consistency_residual : |∇Hᵀẋ + P_diss|.
    passivity_residual : |Ḣ_num + P_diss| (certificado de pasividad).
    dissipation_time_constant : τ_diss = 2H / P_diss (∞ si P_diss ≈ 0).
    stability_flags : retícula Booleana de predicados.
    is_thermodynamically_stable : flags == ALL.
    normal_mode_frequencies, zero_point_energy : Williamson (opcional).
    state_vector : x = [q; p] empaquetado para la Fase 3.
    """

    potential_energy: float
    kinetic_energy: float
    total_hamiltonian: float
    dissipated_power: float
    flyback_voltage_norm: float
    grad_H_norm: float
    vector_field: NDArray[np.float64]
    euler_homogeneity_residual: float
    structural_consistency_residual: float
    passivity_residual: float
    dissipation_time_constant: float
    stability_flags: StabilityFlags
    is_thermodynamically_stable: bool
    normal_mode_frequencies: Optional[NDArray[np.float64]]
    zero_point_energy: Optional[float]
    state_vector: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SheafStalk:
    r"""
    Fibrado celular exportado para el cálculo global del Laplaciano de Haz.

    Producido por la **Fase 3** (Proyección Cohomológica en Haces).

    Continuidad formal
    ------------------
    Este DTO es la **salida terminal** de la cadena de tres fases.

    Ensamblaje
    ----------
        δ_metric ∈ ℝ^{n×n},  δ_diss ∈ ℝ^{n×n},  δ_BASE ∈ ℝ^{2n×n}
        Δ_BASE = δ_BASEᵀ δ_BASE = ∇²H + R_cost  (SPD)

    Atributos adicionales v5
    ------------------------
    spectral_entropy_hodge : entropía de von Neumann de Spec(Δ_BASE).
    cheeger_proxy : λ₂(Δ_BASE) / λ_max(Δ_BASE) (proxy de expansión / Cheeger).
    """

    delta_base: NDArray[np.float64]
    delta_metric: NDArray[np.float64]
    delta_dissipative: NDArray[np.float64]
    hodge_laplacian: NDArray[np.float64]
    hodge_identity_residual: float
    hodge_spectral_gap: float
    hodge_condition_number: float
    harmonic_dimension: int
    lossless_subspace_dimension: int
    state_vector: NDArray[np.float64]
    projected_state_metric: NDArray[np.float64]
    projected_state_dissipative: NDArray[np.float64]
    rank_delta: int
    spectral_entropy_hodge: float
    cheeger_proxy: float


# =============================================================================
# SECCIÓN 3 — ORQUESTADOR: KBaseThermodynamicAgent
#             Tres fases anidadas de rigor creciente
# =============================================================================


class KBaseThermodynamicAgent(Morphism):
    r"""
    Orquestador Funtorial del Foso Termodinámico K_BASE.

    Integra el modelo Port-Hamiltoniano del estrato K_BASE mediante tres
    clases anidadas que operan en cascada estricta:

        Phase1_MatrixTopology
            ↓  TopologicalContext
        Phase2_HamiltonianDynamics
            ↓  BasalStateTensor
        Phase3_SheafProjection
            ↓  SheafStalk

    Parámetros de Construcción
    --------------------------
    C_soc, M_rec, R_cost, J_base : matrices constitutivas.
    breakdown_voltage : umbral de ruptura dieléctrica del Flyback.
    kappa_max : umbral máximo de número de condición.
    G_q, G_p : tensores métricos Riemannianos (None ⇒ I).
    hbar : constante de analogía cuántica para E_0.
    flyback_safety_margin : fracción de V_bd para bandera blanda.
    """

    FRIENDLY_NAME: str = "Asesor de Cimientos Financieros"
    VERSION: str = "5.0.0-Rigorous-Sheaf-Williamson-Boolean-Spectral-Passivity"

    def __init__(
        self,
        C_soc: NDArray[np.float64],
        M_rec: NDArray[np.float64],
        R_cost: NDArray[np.float64],
        J_base: NDArray[np.float64],
        breakdown_voltage: float = 1.0e5,
        kappa_max: float = 1.0e10,
        G_q: Optional[NDArray[np.float64]] = None,
        G_p: Optional[NDArray[np.float64]] = None,
        hbar: float = 1.0,
        flyback_safety_margin: float = 0.9,
    ) -> None:
        r"""
        Inicializa las matrices constitutivas y ejecuta la Fase 1 de inmediato.

        Lanza
        -----
        ValueError
            Si breakdown_voltage ≤ 0, kappa_max ≤ 1, hbar < 0 o
            flyback_safety_margin ∉ (0, 1].
        ThermodynamicBaseError (subclases)
            Propagadas desde la Fase 1.
        """
        if breakdown_voltage <= 0.0:
            raise ValueError(
                f"breakdown_voltage debe ser > 0; se obtuvo {breakdown_voltage}."
            )
        if kappa_max <= 1.0:
            raise ValueError(
                f"kappa_max debe ser > 1; se obtuvo {kappa_max}."
            )
        if hbar < 0.0:
            raise ValueError(f"hbar debe ser ≥ 0; se obtuvo {hbar}.")
        if not (0.0 < flyback_safety_margin <= 1.0):
            raise ValueError(
                f"flyback_safety_margin debe estar en (0, 1]; "
                f"se obtuvo {flyback_safety_margin}."
            )

        self.breakdown_voltage: float = breakdown_voltage
        self.kappa_max: float = kappa_max
        self.hbar: float = hbar
        self.flyback_safety_margin: float = flyback_safety_margin

        # ------------------------------------------------------------------
        # Fase 1: Topología Matricial, Riemann y Espectro (inmediata)
        # ------------------------------------------------------------------
        self.phase1: KBaseThermodynamicAgent.Phase1_MatrixTopology = (
            KBaseThermodynamicAgent.Phase1_MatrixTopology(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
                kappa_max=kappa_max,
                G_q=G_q,
                G_p=G_p,
            )
        )
        self.context: TopologicalContext = (
            self.phase1.build_topological_context()
        )

        # ------------------------------------------------------------------
        # Fase 2: Dinámica Port-Hamiltoniana (continuación formal de Fase 1)
        # ------------------------------------------------------------------
        self.phase2: KBaseThermodynamicAgent.Phase2_HamiltonianDynamics = (
            KBaseThermodynamicAgent.Phase2_HamiltonianDynamics(
                context=self.context,
                breakdown_voltage=self.breakdown_voltage,
                kappa_max=self.kappa_max,
                hbar=self.hbar,
                flyback_safety_margin=self.flyback_safety_margin,
            )
        )

        # Fase 3: Proyección en Haces (instanciación perezosa)
        self.phase3: Optional[
            KBaseThermodynamicAgent.Phase3_SheafProjection
        ] = None

        logger.info(
            "[KBaseThermodynamicAgent v%s] Inicialización completa. "
            "dim_q=%d, dim_p=%d, κ(C̃)=%.3e, κ(M̃)=%.3e, rank(R)=%d, "
            "betti_0(R)=%d, gap(R)=%.3e, S_vN(R)=%.4f, "
            "amp_pullback(C)=%.3e, amp_pullback(M)=%.3e.",
            self.VERSION,
            self.context.dim_q,
            self.context.dim_p,
            self.context.kappa_C,
            self.context.kappa_M,
            self.context.rank_R,
            self.context.betti_0_R,
            self.context.spectral_gap_R,
            self.context.spectral_entropy_R,
            self.context.pullback_amp_C,
            self.context.pullback_amp_M,
        )

    # =========================================================================
    # FASE 1 — TOPOLOGÍA MATRICIAL, PULLBACK RIEMANNIANO Y VALIDACIÓN ESPECTRAL
    # =========================================================================

    class Phase1_MatrixTopology:
        r"""
        **Fase 1 – Topología Matricial, Métrica Riemanniana y Teoría Espectral.**

        Responsabilidades (orden estricto):

          a) Coherencia dimensional de C_soc, M_rec, R_cost, J_base.
          b) Antisimetría de J_base; simetría de C_soc, M_rec, R_cost.
          c) Invertibilidad / κ de G_q, G_p (identidad por defecto).
          d) Pullback congruente: C̃ = G_q C G_qᵀ, M̃ = G_p M G_pᵀ.
          e) Verificación post-pullback de SPD (Ley de Sylvester en caliente).
          f) κ, gap espectral y amplificación de pullback.
          g) Cholesky regularizado (Tikhonov adaptativo).
          h) Precálculo de C̃^{-1/2}, M̃^{-1/2}.
          i) PSD + raíz espectral + gap + Betti-0 + entropía de von Neumann
             de R_cost.
          j) Empaquetado en ``TopologicalContext`` inmutable.

        Tolerancias
        -----------
            tol_sym = ε_mach · max(‖A‖_F, 1)
            tol_pd  = ε_mach · max(|λ_max|, 1)
            tol_psd = ε_mach · max(‖A‖_F, 1)
        """

        _EPS: Final[float] = _MACHINE_EPS

        def __init__(
            self,
            C_soc: NDArray[np.float64],
            M_rec: NDArray[np.float64],
            R_cost: NDArray[np.float64],
            J_base: NDArray[np.float64],
            kappa_max: float = 1.0e10,
            G_q: Optional[NDArray[np.float64]] = None,
            G_p: Optional[NDArray[np.float64]] = None,
        ) -> None:
            """Almacena referencias; copias y defaults en el método terminal."""
            self._C_soc: NDArray[np.float64] = C_soc
            self._M_rec: NDArray[np.float64] = M_rec
            self._R_cost: NDArray[np.float64] = R_cost
            self._J_base: NDArray[np.float64] = J_base
            self._kappa_max: float = kappa_max
            self._G_q_raw: Optional[NDArray[np.float64]] = G_q
            self._G_p_raw: Optional[NDArray[np.float64]] = G_p

        # ---------------------------------------------------------------------
        # Métodos privados de validación (orden lógico de ejecución)
        # ---------------------------------------------------------------------

        def _check_dimensions(self) -> Tuple[int, int]:
            r"""
            Verifica dimensiones del espacio de fases (q, p).

            Condiciones formales:
              • C_soc ∈ ℝ^{dim_q × dim_q}, M_rec ∈ ℝ^{dim_p × dim_p}
              • R_cost, J_base ∈ ℝ^{n × n}, n = dim_q + dim_p
              • Todos los arrays son 2D

            Retorna
            -------
            Tuple[int, int]
                (dim_q, dim_p)
            """
            for mat, name in [
                (self._C_soc, "C_soc"),
                (self._M_rec, "M_rec"),
                (self._R_cost, "R_cost"),
                (self._J_base, "J_base"),
            ]:
                if mat.ndim != 2:
                    raise DimensionMismatchError(
                        f"La matriz '{name}' debe ser 2D; "
                        f"se obtuvo ndim={mat.ndim}, shape={mat.shape}."
                    )

            if self._C_soc.shape[0] != self._C_soc.shape[1]:
                raise DimensionMismatchError(
                    f"C_soc debe ser cuadrada; se obtuvo shape={self._C_soc.shape}."
                )
            dim_q: int = int(self._C_soc.shape[0])
            if dim_q < 1:
                raise DimensionMismatchError(
                    f"dim_q debe ser ≥ 1; se obtuvo {dim_q}."
                )

            if self._M_rec.shape[0] != self._M_rec.shape[1]:
                raise DimensionMismatchError(
                    f"M_rec debe ser cuadrada; se obtuvo shape={self._M_rec.shape}."
                )
            dim_p: int = int(self._M_rec.shape[0])
            if dim_p < 1:
                raise DimensionMismatchError(
                    f"dim_p debe ser ≥ 1; se obtuvo {dim_p}."
                )

            n: int = dim_q + dim_p

            if self._R_cost.shape != (n, n):
                raise DimensionMismatchError(
                    f"R_cost debe ser ({n},{n}) = (dim_q+dim_p)², "
                    f"pero se obtuvo {self._R_cost.shape}. "
                    f"dim_q={dim_q}, dim_p={dim_p}."
                )
            if self._J_base.shape != (n, n):
                raise DimensionMismatchError(
                    f"J_base debe ser ({n},{n}); se obtuvo {self._J_base.shape}."
                )

            logger.debug(
                "[Fase1] Dimensiones verificadas: dim_q=%d, dim_p=%d, n=%d.",
                dim_q,
                dim_p,
                n,
            )
            return dim_q, dim_p

        def _validate_symmetry(
            self, A: NDArray[np.float64], name: str
        ) -> None:
            r"""Verifica A = Aᵀ con tol = ε_mach · max(‖A‖_F, 1)."""
            norm_A: float = float(la.norm(A, "fro"))
            tol: float = self._EPS * max(norm_A, 1.0)
            residual: float = float(la.norm(A - A.T, "fro"))
            if residual > tol:
                raise ThermodynamicBaseError(
                    f"La matriz '{name}' no es simétrica. "
                    f"‖A−Aᵀ‖_F = {residual:.6e}, tol = {tol:.6e}, "
                    f"asimetría relativa = "
                    f"{residual / max(norm_A, 1e-300):.6e}."
                )
            logger.debug(
                "[Fase1] Simetría de '%s': residual=%.3e, tol=%.3e.",
                name,
                residual,
                tol,
            )

        def _validate_antisymmetry(
            self, J: NDArray[np.float64], name: str
        ) -> None:
            r"""
            Verifica J = −Jᵀ con tol relativa.

            La antisimetría garantiza que la parte conservativa no produce
            ni consume energía: ∇Hᵀ J ∇H ≡ 0.
            """
            norm_J: float = float(la.norm(J, "fro"))
            tol: float = self._EPS * max(norm_J, 1.0)
            residual: float = float(la.norm(J + J.T, "fro"))
            if residual > tol:
                raise ThermodynamicBaseError(
                    f"La matriz '{name}' no es antisimétrica (J ≠ −Jᵀ). "
                    f"‖J+Jᵀ‖_F = {residual:.6e}, tol = {tol:.6e}."
                )
            logger.debug(
                "[Fase1] Antisimetría de '%s': residual=%.3e, tol=%.3e.",
                name,
                residual,
                tol,
            )

        def _validate_metric_tensor(
            self,
            G: NDArray[np.float64],
            name: str,
            expected_dim: int,
        ) -> float:
            r"""
            Valida G ∈ ℝ^{d×d} invertible y bien condicionada vía SVD:

                κ(G) = σ_max / σ_min

            Condición necesaria para que el pullback Ã = G A Gᵀ preserve
            la signatura de A (Ley de Inercia de Sylvester) sin amplificar
            patológicamente el número de condición.
            """
            if G.ndim != 2 or G.shape != (expected_dim, expected_dim):
                raise DimensionMismatchError(
                    f"El tensor métrico '{name}' debe ser "
                    f"({expected_dim},{expected_dim}); se obtuvo {G.shape}."
                )

            singular_values: NDArray[np.float64] = la.svdvals(G)
            sigma_max: float = float(singular_values[0])
            sigma_min: float = float(singular_values[-1])

            if sigma_max <= 0.0 or sigma_min <= self._EPS * max(sigma_max, 1.0):
                raise MetricTensorSingularityError(
                    f"El tensor métrico '{name}' es singular o casi singular. "
                    f"σ_min={sigma_min:.6e}, σ_max={sigma_max:.6e}."
                )

            kappa_G: float = sigma_max / sigma_min
            if kappa_G > self._kappa_max:
                raise MetricTensorSingularityError(
                    f"El tensor métrico '{name}' está mal condicionado: "
                    f"κ(G)={kappa_G:.6e} > κ_max={self._kappa_max:.6e}. "
                    f"El pullback amplificaría patológicamente el estrés "
                    f"anisotrópico."
                )

            logger.debug(
                "[Fase1] κ(%s)=%.6e (σ_min=%.6e, σ_max=%.6e).",
                name,
                kappa_G,
                sigma_min,
                sigma_max,
            )
            return kappa_G

        def _congruence_pullback(
            self,
            A: NDArray[np.float64],
            G: NDArray[np.float64],
            name: str,
        ) -> NDArray[np.float64]:
            r"""
            Pullback geométrico congruente:

                Ã = G · A · Gᵀ

            Re-simetriza defensivamente para eliminar ruido O(ε·‖A‖·‖G‖²).
            """
            A_tilde: NDArray[np.float64] = G @ A @ G.T
            A_tilde = 0.5 * (A_tilde + A_tilde.T)
            rel_change: float = float(la.norm(A_tilde - A, "fro")) / max(
                float(la.norm(A, "fro")), 1e-300
            )
            logger.debug(
                "[Fase1] Pullback Riemanniano de '%s': ‖Ã−A‖_F/‖A‖_F=%.3e.",
                name,
                rel_change,
            )
            return A_tilde

        def _spectral_extremes_and_gap(
            self,
            A: NDArray[np.float64],
            name: str,
            *,
            require_spd: bool = True,
        ) -> Tuple[float, float, float, float]:
            r"""
            Extrae (κ, λ_min, λ_max, gap) de una matriz simétrica.

            Para n ≥ 2 usa ``eigh(subset_by_index=...)`` en los extremos y
            en el segundo más pequeño (gap = λ₂ − λ₁). Para n = 1, gap = 0.

            Si require_spd=True, lanza CapacitanceDegeneracyError si
            λ_min ≤ tol_pd, e IllConditionedMatrixError si κ > κ_max.
            """
            n: int = int(A.shape[0])
            A_sym: NDArray[np.float64] = 0.5 * (A + A.T)

            if n == 1:
                lambda_min = lambda_max = float(A_sym[0, 0])
                lambda_second = lambda_min
            else:
                # Extremos
                lambda_min = float(
                    la.eigh(
                        A_sym, subset_by_index=[0, 0], eigvals_only=True
                    )[0]
                )
                lambda_max = float(
                    la.eigh(
                        A_sym,
                        subset_by_index=[n - 1, n - 1],
                        eigvals_only=True,
                    )[0]
                )
                # Segundo más pequeño para el gap (si n ≥ 2)
                lambda_second = float(
                    la.eigh(
                        A_sym, subset_by_index=[1, 1], eigvals_only=True
                    )[0]
                )

            tol_pd: float = self._EPS * max(abs(lambda_max), 1.0)

            if require_spd and lambda_min <= tol_pd:
                raise CapacitanceDegeneracyError(
                    f"La matriz '{name}' no es Definida Positiva (SPD). "
                    f"λ_min = {lambda_min:.6e} ≤ tol_pd = {tol_pd:.6e}. "
                    f"λ_max = {lambda_max:.6e}."
                )

            if lambda_min > tol_pd:
                kappa: float = lambda_max / lambda_min
            else:
                kappa = float("inf")

            if require_spd and kappa > self._kappa_max:
                raise IllConditionedMatrixError(
                    f"La matriz '{name}' está numéricamente mal condicionada. "
                    f"κ = {kappa:.6e} > κ_max = {self._kappa_max:.6e}. "
                    f"Considere regularización de Tikhonov o re-escalado."
                )

            spectral_gap: float = max(lambda_second - lambda_min, 0.0)

            logger.debug(
                "[Fase1] Espectro '%s': λ_min=%.3e, λ₂=%.3e, λ_max=%.3e, "
                "κ=%.3e, gap=%.3e.",
                name,
                lambda_min,
                lambda_second,
                lambda_max,
                kappa,
                spectral_gap,
            )
            return kappa, lambda_min, lambda_max, spectral_gap

        def _cholesky_spd_regularized(
            self,
            A: NDArray[np.float64],
            name: str,
            max_attempts: int = 6,
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Cholesky A = L Lᵀ con Tikhonov adaptativo:

                A_τ = A + τ I,
                τ₀ = 0,
                τ₁ = ε_mach · tr(A)/n,
                τ_{k+1} = 10 · τ_k

            Retorna (L, tau_final).
            """
            A_sym: NDArray[np.float64] = 0.5 * (A + A.T)
            n: int = int(A_sym.shape[0])
            trace_scale: float = float(np.trace(A_sym)) / max(n, 1)
            tau: float = 0.0
            identity_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)

            for attempt in range(max_attempts + 1):
                try:
                    L: NDArray[np.float64] = la.cholesky(
                        A_sym + tau * identity_n, lower=True
                    )
                    if attempt > 0:
                        logger.warning(
                            "[Fase1] Tikhonov en '%s': τ=%.3e tras %d intento(s).",
                            name,
                            tau,
                            attempt,
                        )
                    else:
                        logger.debug(
                            "[Fase1] Cholesky de '%s' sin regularización. "
                            "L[0,0]=%.6e, L[-1,-1]=%.6e.",
                            name,
                            float(L[0, 0]),
                            float(L[-1, -1]),
                        )
                    return L, tau
                except la.LinAlgError:
                    if tau == 0.0:
                        tau = self._EPS * max(trace_scale, 1.0)
                    else:
                        tau *= 10.0

            raise CapacitanceDegeneracyError(
                f"Fallo persistente de Cholesky (LAPACK dpotrf) en '{name}' "
                f"tras {max_attempts} intentos de Tikhonov "
                f"(τ_final={tau:.3e}). Degeneración estructural."
            )

        @staticmethod
        def _von_neumann_entropy(
            eigvals: NDArray[np.float64],
        ) -> float:
            r"""
            Entropía de von Neumann del espectro no negativo:

                S = −Σ p_i ln p_i ,  p_i = λ_i / Σ λ_j

            S = 0 ⇔ un solo modo; S → ln(rango) ⇔ espectro plano.
            """
            total: float = float(np.sum(eigvals))
            if total <= 0.0:
                return 0.0
            p: NDArray[np.float64] = eigvals / total
            p_pos = p[p > 0.0]
            return float(-np.sum(p_pos * np.log(p_pos)))

        def _validate_psd_and_spectral_diagnostics(
            self,
            R: NDArray[np.float64],
            name: str,
        ) -> Tuple[NDArray[np.float64], int, float, float]:
            r"""
            Verifica R ⪰ 0 y calcula:

              1. Raíz espectral R^{1/2}
              2. rank_R
              3. Brecha espectral λ₂ − λ₁
              4. Entropía de von Neumann del espectro

            Retorna
            -------
            (R_sqrt, rank_R, spectral_gap, spectral_entropy)
            """
            R_sym: NDArray[np.float64] = 0.5 * (R + R.T)
            norm_R: float = float(la.norm(R_sym, "fro"))
            tol_psd: float = self._EPS * max(norm_R, 1.0)

            eigvals: NDArray[np.float64]
            eigvecs: NDArray[np.float64]
            eigvals, eigvecs = la.eigh(R_sym)

            lambda_min: float = float(eigvals[0])
            if lambda_min < -tol_psd:
                raise RayleighDissipationViolation(
                    f"La matriz '{name}' no es Semidefinida Positiva (PSD). "
                    f"λ_min = {lambda_min:.6e} < −tol = {-tol_psd:.6e}. "
                    f"Entropía negativa (ganancia fantasma): violación de "
                    f"la Segunda Ley de la Termodinámica."
                )

            eigvals_clamped: NDArray[np.float64] = np.maximum(eigvals, 0.0)

            R_sqrt: NDArray[np.float64] = (
                eigvecs * np.sqrt(eigvals_clamped)[np.newaxis, :]
            ) @ eigvecs.T
            R_sqrt = 0.5 * (R_sqrt + R_sqrt.T)

            rank_R: int = int(np.sum(eigvals_clamped > tol_psd))
            spectral_gap: float = (
                float(eigvals_clamped[1] - eigvals_clamped[0])
                if len(eigvals_clamped) > 1
                else 0.0
            )
            spectral_entropy: float = self._von_neumann_entropy(
                eigvals_clamped
            )

            logger.debug(
                "[Fase1] %s PSD: rank=%d/%d, λ_min=%.3e, λ_max=%.3e, "
                "gap=%.3e, S_vN=%.4f.",
                name,
                rank_R,
                len(eigvals),
                lambda_min,
                float(eigvals[-1]),
                spectral_gap,
                spectral_entropy,
            )
            return R_sqrt, rank_R, spectral_gap, spectral_entropy

        def _kappa_of_original(
            self, A: NDArray[np.float64], name: str
        ) -> float:
            """κ de la matriz original (pre-pullback) para medir amplificación."""
            try:
                kappa, _, _, _ = self._spectral_extremes_and_gap(
                    A, name, require_spd=True
                )
                return kappa
            except (CapacitanceDegeneracyError, IllConditionedMatrixError):
                # Si la original ya es borderline, reportamos inf
                return float("inf")

        # ---------------------------------------------------------------------
        # Método terminal de la Fase 1 — entrada directa de la Fase 2
        # ---------------------------------------------------------------------

        def build_topological_context(self) -> "TopologicalContext":
            r"""
            **Método terminal de la Fase 1.**

            Ejecuta en secuencia estricta validación, pullback, espectro y
            factorización; empaqueta el resultado en ``TopologicalContext``.

            El contexto es el **único argumento** del constructor de
            ``Phase2_HamiltonianDynamics`` (y de ``Phase3_SheafProjection``).

            Flujo interno
            -------------
            1. Dimensiones.
            2. Antisimetría de J_base; simetría de C, M, R.
            3. Resolución y validación de G_q, G_p.
            4. κ pre-pullback de C_soc, M_rec.
            5. Pullback Riemanniano → C̃, M̃.
            6. κ, gap post-pullback; amplificación κ̃/κ.
            7. Cholesky regularizado → L_C, L_M.
            8. Precálculo C̃^{-1/2}, M̃^{-1/2}.
            9. PSD + raíz + gap + Betti-0 + S_vN de R_cost.
            10. Empaquetado.

            Retorna
            -------
            TopologicalContext
            """
            # ---- Paso 1: dimensiones ----
            dim_q, dim_p = self._check_dimensions()

            # ---- Paso 2: simetrías estructurales ----
            self._validate_antisymmetry(self._J_base, "J_base")
            self._validate_symmetry(self._C_soc, "C_soc")
            self._validate_symmetry(self._M_rec, "M_rec")
            self._validate_symmetry(self._R_cost, "R_cost")

            # ---- Paso 3: tensores métricos ----
            G_q: NDArray[np.float64] = (
                self._G_q_raw
                if self._G_q_raw is not None
                else np.eye(dim_q, dtype=np.float64)
            )
            G_p: NDArray[np.float64] = (
                self._G_p_raw
                if self._G_p_raw is not None
                else np.eye(dim_p, dtype=np.float64)
            )
            kappa_G_q: float = self._validate_metric_tensor(
                G_q, "G_q", dim_q
            )
            kappa_G_p: float = self._validate_metric_tensor(
                G_p, "G_p", dim_p
            )

            # ---- Paso 4: κ pre-pullback ----
            kappa_C_pre: float = self._kappa_of_original(
                self._C_soc, "C_soc(pre)"
            )
            kappa_M_pre: float = self._kappa_of_original(
                self._M_rec, "M_rec(pre)"
            )

            # ---- Paso 5: pullback ----
            C_tilde: NDArray[np.float64] = self._congruence_pullback(
                self._C_soc, G_q, "C_soc"
            )
            M_tilde: NDArray[np.float64] = self._congruence_pullback(
                self._M_rec, G_p, "M_rec"
            )

            # ---- Paso 6: espectro post-pullback ----
            kappa_C, _, _, gap_C = self._spectral_extremes_and_gap(
                C_tilde, "C̃_soc", require_spd=True
            )
            kappa_M, _, _, gap_M = self._spectral_extremes_and_gap(
                M_tilde, "M̃_rec", require_spd=True
            )

            pullback_amp_C: float = (
                kappa_C / kappa_C_pre
                if math_isfinite(kappa_C_pre) and kappa_C_pre > 0.0
                else float("inf")
            )
            pullback_amp_M: float = (
                kappa_M / kappa_M_pre
                if math_isfinite(kappa_M_pre) and kappa_M_pre > 0.0
                else float("inf")
            )

            # ---- Paso 7: Cholesky regularizado ----
            L_C, epsilon_C = self._cholesky_spd_regularized(
                C_tilde, "C̃_soc"
            )
            L_M, epsilon_M = self._cholesky_spd_regularized(
                M_tilde, "M̃_rec"
            )

            # ---- Paso 8: raíces de las inversas ----
            # C̃^{-1/2} = L_C^{-T}  (porque C̃^{-1} = L_C^{-T} L_C^{-1}
            # y C̃^{-1/2} simétrica = L_C^{-T} en la factorización polar
            # de la inversa cuando se usa como bloque métrico).
            # Construcción estable: resolver L_C Y = I ⇒ Y = L_C^{-1};
            # C_inv_sqrt := Y.T = L_C^{-T}.
            C_inv_sqrt: NDArray[np.float64] = la.solve_triangular(
                L_C,
                np.eye(dim_q, dtype=np.float64),
                lower=True,
                check_finite=False,
            ).T
            M_inv_sqrt: NDArray[np.float64] = la.solve_triangular(
                L_M,
                np.eye(dim_p, dtype=np.float64),
                lower=True,
                check_finite=False,
            ).T

            # ---- Paso 9: R_cost ----
            R_sqrt, rank_R, spectral_gap_R, spectral_entropy_R = (
                self._validate_psd_and_spectral_diagnostics(
                    self._R_cost, "R_cost"
                )
            )
            n_total: int = dim_q + dim_p
            betti_0_R: int = n_total - rank_R

            # ---- Paso 10: empaquetado ----
            context = TopologicalContext(
                L_C=L_C,
                L_M=L_M,
                C_inv_sqrt=C_inv_sqrt,
                M_inv_sqrt=M_inv_sqrt,
                C_tilde=C_tilde.copy(),
                M_tilde=M_tilde.copy(),
                R_cost=self._R_cost.copy(),
                R_sqrt=R_sqrt,
                J_base=self._J_base.copy(),
                G_q=G_q.copy(),
                G_p=G_p.copy(),
                kappa_C=kappa_C,
                kappa_M=kappa_M,
                kappa_G_q=kappa_G_q,
                kappa_G_p=kappa_G_p,
                epsilon_C=epsilon_C,
                epsilon_M=epsilon_M,
                dim_q=dim_q,
                dim_p=dim_p,
                rank_R=rank_R,
                spectral_gap_R=spectral_gap_R,
                betti_0_R=betti_0_R,
                spectral_gap_C=gap_C,
                spectral_gap_M=gap_M,
                pullback_amp_C=pullback_amp_C,
                pullback_amp_M=pullback_amp_M,
                spectral_entropy_R=spectral_entropy_R,
            )

            logger.info(
                "[Fase1] TopologicalContext ensamblado: dim_q=%d, dim_p=%d, "
                "κ(C̃)=%.3e, κ(M̃)=%.3e, rank(R)=%d, betti_0(R)=%d, "
                "gap(R)=%.3e, S_vN(R)=%.4f, amp_C=%.3e, amp_M=%.3e.",
                dim_q,
                dim_p,
                kappa_C,
                kappa_M,
                rank_R,
                betti_0_R,
                spectral_gap_R,
                spectral_entropy_R,
                pullback_amp_C,
                pullback_amp_M,
            )

            # ================================================================
            # CONTRATO DE INTERFAZ FASE 1 → FASE 2
            # `context` es el argumento directo del constructor de
            # Phase2_HamiltonianDynamics. Esta devolución es la frontera
            # formal entre ambas fases anidadas.
            # ================================================================
            return context

    # =========================================================================
    # FASE 2 — DINÁMICA PORT-HAMILTONIANA, RAYLEIGH, WILLIAMSON Y PASIVIDAD
    # =========================================================================

    class Phase2_HamiltonianDynamics:
        r"""
        **Fase 2 – Dinámica Port-Hamiltoniana y Disipación de Rayleigh.**

        Recibe el ``TopologicalContext`` de la Fase 1 y calcula:

          • Energías, gradiente y campo vectorial:
                ẋ = (J_BASE − R_cost) ∇H
          • Identidad estructural ∇Hᵀ J ∇H ≡ 0
          • Homogeneidad de Euler (H cuadrática)
          • Certificado de pasividad Ḣ + P_diss ≈ 0
          • Constante de tiempo entrópica τ_diss = 2H / P_diss
          • Modos normales de Williamson (opt-in)
          • Retícula Booleana de estabilidad ampliada

        Identidad de energía sin inversión explícita
        ---------------------------------------------
            qᵀ Ã⁻¹ q = ‖ L⁻¹ q ‖²     (Ã = L Lᵀ)
        """

        _EPS: Final[float] = _MACHINE_EPS

        def __init__(
            self,
            context: "TopologicalContext",
            breakdown_voltage: float,
            kappa_max: float,
            hbar: float = 1.0,
            flyback_safety_margin: float = 0.9,
        ) -> None:
            r"""
            **Constructor de la Fase 2: continuación directa de la Fase 1.**

            No re-valida matrices: la corrección (incl. pullback) está
            garantizada por la Fase 1.
            """
            self._ctx: "TopologicalContext" = context
            self._breakdown_voltage: float = breakdown_voltage
            self._kappa_max: float = kappa_max
            self._hbar: float = hbar
            self._flyback_safety_margin: float = flyback_safety_margin

            logger.debug(
                "[Fase2] Inicializada: V_bd=%.3e, κ_max=%.3e, ħ=%.3e, "
                "margen_fb=%.2f, dim_q=%d, dim_p=%d.",
                breakdown_voltage,
                kappa_max,
                hbar,
                flyback_safety_margin,
                context.dim_q,
                context.dim_p,
            )

        # ---------------------------------------------------------------------
        # Energías y gradientes
        # ---------------------------------------------------------------------

        def _evaluate_potential_energy(
            self, q: NDArray[np.float64]
        ) -> Tuple[float, NDArray[np.float64]]:
            r"""
            V(q) = ½ ‖L_C⁻¹ q‖² = ½ qᵀ C̃_soc⁻¹ q
            ∇_q V = C̃_soc⁻¹ q
            """
            if q.shape != (self._ctx.dim_q,):
                raise DimensionMismatchError(
                    f"Vector q debe tener shape ({self._ctx.dim_q},); "
                    f"se obtuvo {q.shape}."
                )
            y: NDArray[np.float64] = la.solve_triangular(
                self._ctx.L_C, q, lower=True, check_finite=False
            )
            V_q: float = 0.5 * float(np.dot(y, y))
            grad_V_q: NDArray[np.float64] = la.solve_triangular(
                self._ctx.L_C,
                y,
                lower=True,
                trans="T",
                check_finite=False,
            )
            # Clampeado numérico: V no puede ser legítimamente negativa
            if V_q < 0.0:
                V_q = 0.0
            logger.debug(
                "[Fase2] V(q)=%.6e, ‖∇V‖=%.6e.",
                V_q,
                float(np.linalg.norm(grad_V_q)),
            )
            return V_q, grad_V_q

        def _compute_kinetic_energy(
            self, p: NDArray[np.float64]
        ) -> Tuple[float, NDArray[np.float64]]:
            r"""
            K(p) = ½ ‖L_M⁻¹ p‖² = ½ pᵀ M̃_rec⁻¹ p
            ∇_p K = M̃_rec⁻¹ p
            """
            if p.shape != (self._ctx.dim_p,):
                raise DimensionMismatchError(
                    f"Vector p debe tener shape ({self._ctx.dim_p},); "
                    f"se obtuvo {p.shape}."
                )
            y: NDArray[np.float64] = la.solve_triangular(
                self._ctx.L_M, p, lower=True, check_finite=False
            )
            K_p: float = 0.5 * float(np.dot(y, y))
            grad_K_p: NDArray[np.float64] = la.solve_triangular(
                self._ctx.L_M,
                y,
                lower=True,
                trans="T",
                check_finite=False,
            )
            if K_p < 0.0:
                K_p = 0.0
            logger.debug(
                "[Fase2] K(p)=%.6e, ‖∇K‖=%.6e.",
                K_p,
                float(np.linalg.norm(grad_K_p)),
            )
            return K_p, grad_K_p

        def _verify_euler_homogeneity(
            self,
            q: NDArray[np.float64],
            p: NDArray[np.float64],
            grad_V: NDArray[np.float64],
            grad_K: NDArray[np.float64],
            H_total: float,
        ) -> float:
            r"""
            Teorema de Euler para H homogénea de grado 2:

                q · ∇_q H + p · ∇_p H = 2 H

            Retorna el residuo absoluto.
            """
            euler_lhs: float = float(np.dot(q, grad_V)) + float(
                np.dot(p, grad_K)
            )
            residual: float = abs(euler_lhs - 2.0 * H_total)
            scale: float = max(abs(euler_lhs), abs(2.0 * H_total), 1.0)
            tol: float = 1.0e3 * self._EPS * scale
            if residual > tol:
                logger.warning(
                    "[Fase2] Residuo de Euler elevado: %.6e > tol=%.6e.",
                    residual,
                    tol,
                )
            return residual

        def _enforce_rayleigh_dissipation(
            self, grad_H: NDArray[np.float64]
        ) -> float:
            r"""
            Ḣ_diss = −∇Hᵀ R_cost ∇H ≤ 0
            P_diss = |Ḣ_diss| ≥ 0
            """
            n: int = self._ctx.dim_q + self._ctx.dim_p
            if grad_H.shape != (n,):
                raise DimensionMismatchError(
                    f"grad_H debe tener shape ({n},); se obtuvo {grad_H.shape}."
                )

            R_grad: NDArray[np.float64] = self._ctx.R_cost @ grad_H
            quad_form: float = float(np.dot(grad_H, R_grad))

            norm_R: float = float(la.norm(self._ctx.R_cost, "fro"))
            norm_gH2: float = float(np.dot(grad_H, grad_H))
            tol_diss: float = self._EPS * norm_R * max(norm_gH2, 1.0)

            if quad_form < -tol_diss:
                logger.error(
                    "[Fase2] Violación de Rayleigh: ∇HᵀR∇H=%.6e < −tol=%.6e.",
                    quad_form,
                    -tol_diss,
                )
                raise RayleighDissipationViolation(
                    f"Violación de la Segunda Ley de la Termodinámica. "
                    f"∇Hᵀ R_cost ∇H = {quad_form:.6e} < −tol = {-tol_diss:.6e}. "
                    f"Generación de exergía espontánea."
                )

            P_diss: float = abs(quad_form)
            logger.debug("[Fase2] P_diss=%.6e (Rayleigh OK).", P_diss)
            return P_diss

        def _measure_flyback_voltage(
            self, df_dt: NDArray[np.float64]
        ) -> float:
            r"""
            ‖M̃_rec · ∂f/∂t‖_∞ reconstruido vía L_M sin materializar M̃:

                M̃ v = L_M (L_Mᵀ v)
            """
            if df_dt.shape != (self._ctx.dim_p,):
                raise DimensionMismatchError(
                    f"df_dt debe tener shape ({self._ctx.dim_p},); "
                    f"se obtuvo {df_dt.shape}."
                )
            L_M_T_v: NDArray[np.float64] = self._ctx.L_M.T @ df_dt
            V_fb_vec: NDArray[np.float64] = self._ctx.L_M @ L_M_T_v
            v_fb_norm: float = float(la.norm(V_fb_vec, np.inf))

            if v_fb_norm > self._breakdown_voltage:
                logger.critical(
                    "[Fase2] Golpe de Ariete Logístico: ‖V_fb‖=%.6e > V_bd=%.6e.",
                    v_fb_norm,
                    self._breakdown_voltage,
                )
                raise InertialFlybackError(
                    f"Voltaje de Flyback ‖M̃_rec·∂f/∂t‖_∞ = {v_fb_norm:.6e} "
                    f"excede V_bd = {self._breakdown_voltage:.6e}. "
                    f"Detención de emergencia exigida."
                )

            logger.debug(
                "[Fase2] ‖V_fb‖_∞=%.6e (límite=%.6e, uso=%.1f%%).",
                v_fb_norm,
                self._breakdown_voltage,
                100.0 * v_fb_norm / max(self._breakdown_voltage, 1e-300),
            )
            return v_fb_norm

        # ---------------------------------------------------------------------
        # Campo vectorial y verificaciones estructurales / pasividad
        # ---------------------------------------------------------------------

        def _compute_vector_field(
            self, grad_H: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""ẋ = (J_BASE − R_cost) ∇H"""
            return (self._ctx.J_base - self._ctx.R_cost) @ grad_H

        def _verify_structural_consistency(
            self,
            grad_H: NDArray[np.float64],
            x_dot: NDArray[np.float64],
            P_diss: float,
        ) -> float:
            r"""
            Identidad exacta del PH-system:

                ∇Hᵀ ẋ = ∇Hᵀ J ∇H − ∇Hᵀ R ∇H = −P_diss

            porque ∇Hᵀ J ∇H ≡ 0 (J antisimétrica).
            """
            H_dot: float = float(np.dot(grad_H, x_dot))
            residual: float = abs(H_dot + P_diss)
            scale: float = max(abs(H_dot), P_diss, 1.0)
            tol: float = float(np.sqrt(self._EPS)) * scale

            if residual > tol:
                raise StructuralConsistencyError(
                    f"Inconsistencia estructural Port-Hamiltoniana: "
                    f"∇Hᵀẋ = {H_dot:.6e}, −P_diss = {-P_diss:.6e}, "
                    f"|residuo| = {residual:.6e} > tol = {tol:.6e}. "
                    f"Verifique el cableado de J_base y R_cost."
                )
            logger.debug(
                "[Fase2] Consistencia estructural: residuo=%.3e, tol=%.3e.",
                residual,
                tol,
            )
            return residual

        def _verify_passivity_certificate(
            self,
            grad_H: NDArray[np.float64],
            x_dot: NDArray[np.float64],
            P_diss: float,
        ) -> float:
            r"""
            Certificado de pasividad en caliente:

                Ḣ_num ≜ ∇Hᵀ ẋ
                |Ḣ_num + P_diss| ≤ √ε · scale

            Equivalente algebraicamente a la consistencia estructural, pero
            se reporta como predicado independiente en la retícula Booleana
            (PASSIVITY_CERTIFICATE) para diagnóstico granular.

            Adicionalmente verifica Ḣ_num ≤ tol (no hay creación de energía).
            """
            H_dot: float = float(np.dot(grad_H, x_dot))
            residual: float = abs(H_dot + P_diss)
            scale: float = max(abs(H_dot), P_diss, 1.0)
            tol: float = float(np.sqrt(self._EPS)) * scale

            # Ḣ debe ser ≤ 0 (pasividad)
            if H_dot > tol:
                raise PassivityCertificateError(
                    f"Certificado de pasividad violado: Ḣ_num = {H_dot:.6e} > 0 "
                    f"(tol={tol:.6e}). El sistema generaría energía espontánea."
                )
            if residual > tol:
                raise PassivityCertificateError(
                    f"Certificado de pasividad inconsistente: "
                    f"|Ḣ_num + P_diss| = {residual:.6e} > tol = {tol:.6e}."
                )
            return residual

        def _dissipation_time_constant(
            self, H_total: float, P_diss: float
        ) -> float:
            r"""
            Constante de tiempo entrópica local:

                τ_diss = 2 H / P_diss

            (escala de decaimiento si Ḣ ≈ −P_diss y H es cuadrática).
            Retorna +∞ si P_diss ≈ 0 (subsistema conservativo puro).
            """
            if P_diss <= self._EPS * max(H_total, 1.0):
                return float("inf")
            return 2.0 * H_total / P_diss

        # ---------------------------------------------------------------------
        # Williamson (diagnóstico opt-in)
        # ---------------------------------------------------------------------

        def compute_normal_modes(
            self,
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Frecuencias propias del subsistema conservativo linealizado
            vía Teorema de Williamson (aritmética real):

                H^{1/2} = block_diag(C̃^{-1/2}, M̃^{-1/2})
                A_sym   = H^{1/2} J H^{1/2}   (antisimétrica)
                ω_k     = SV emparejados de A_sym
                E_0     = (ħ/2) Σ ω_k

            Lanza
            -----
            WilliamsonNormalFormError
                Si A_sym no es numéricamente antisimétrica o hay NaN.
            """
            dim_q, dim_p = self._ctx.dim_q, self._ctx.dim_p
            n: int = dim_q + dim_p

            H_half: NDArray[np.float64] = np.zeros((n, n), dtype=np.float64)
            H_half[:dim_q, :dim_q] = self._ctx.C_inv_sqrt
            H_half[dim_q:, dim_q:] = self._ctx.M_inv_sqrt

            A_sym: NDArray[np.float64] = H_half @ self._ctx.J_base @ H_half
            # Re-antisimetrización defensiva
            A_skew: NDArray[np.float64] = 0.5 * (A_sym - A_sym.T)

            # Verificar residual de antisimetría
            skew_res: float = float(la.norm(A_skew + A_skew.T, "fro"))
            tol_skew: float = 100.0 * self._EPS * max(
                float(la.norm(A_skew, "fro")), 1.0
            )
            if skew_res > tol_skew:
                raise WilliamsonNormalFormError(
                    f"A_sym = H^{{1/2}} J H^{{1/2}} no es antisimétrica: "
                    f"‖A+Aᵀ‖_F = {skew_res:.6e} > tol = {tol_skew:.6e}."
                )

            singular_values: NDArray[np.float64] = la.svdvals(A_skew)
            if not np.all(np.isfinite(singular_values)):
                raise WilliamsonNormalFormError(
                    "Valores singulares de A_sym contienen NaN/Inf."
                )

            n_pairs: int = n // 2
            omegas: NDArray[np.float64] = np.empty(
                n_pairs, dtype=np.float64
            )
            for k in range(n_pairs):
                # Cada frecuencia física aparece ~2 veces (par ±iω)
                if 2 * k + 1 < len(singular_values):
                    omegas[k] = 0.5 * (
                        singular_values[2 * k]
                        + singular_values[2 * k + 1]
                    )
                else:
                    omegas[k] = singular_values[2 * k]

            # Modo residual si n impar: el SV más pequeño ≈ 0
            omegas = np.sort(np.maximum(omegas, 0.0))[::-1]
            zero_point_energy: float = (
                0.5 * self._hbar * float(np.sum(omegas))
            )

            logger.debug(
                "[Fase2] Williamson: %d pares, ω_max=%.6e, E_0=%.6e (ħ=%.3e).",
                n_pairs,
                float(omegas[0]) if n_pairs > 0 else 0.0,
                zero_point_energy,
                self._hbar,
            )
            return omegas, zero_point_energy

        # ---------------------------------------------------------------------
        # Retícula Booleana de estabilidad
        # ---------------------------------------------------------------------

        def _evaluate_stability_flags(
            self,
            V_q: float,
            K_p: float,
            P_diss: float,
            v_fb: float,
            structural_residual: float,
            passivity_residual: float,
            euler_residual: float,
            H_total: float,
        ) -> StabilityFlags:
            """Evalúa la retícula StabilityFlags (7 predicados atómicos)."""
            flags: StabilityFlags = StabilityFlags.NONE
            energy_tol: float = 1.0e3 * self._EPS
            struct_tol: float = float(np.sqrt(self._EPS)) * max(
                P_diss, 1.0
            )
            euler_tol: float = 1.0e3 * self._EPS * max(
                abs(2.0 * H_total), 1.0
            )

            if V_q >= -energy_tol and K_p >= -energy_tol:
                flags |= StabilityFlags.ENERGY_NONNEGATIVE
            if P_diss >= -energy_tol:
                flags |= StabilityFlags.DISSIPATION_VALID
            if v_fb <= self._flyback_safety_margin * self._breakdown_voltage:
                flags |= StabilityFlags.FLYBACK_SAFE
            if structural_residual <= struct_tol:
                flags |= StabilityFlags.STRUCTURAL_CONSISTENCY
            if (
                self._ctx.kappa_C <= 0.5 * self._kappa_max
                and self._ctx.kappa_M <= 0.5 * self._kappa_max
            ):
                flags |= StabilityFlags.SPECTRAL_CONDITIONING_SOUND
            if passivity_residual <= struct_tol:
                flags |= StabilityFlags.PASSIVITY_CERTIFICATE
            if euler_residual <= euler_tol:
                flags |= StabilityFlags.EULER_HOMOGENEITY

            logger.debug("[Fase2] %s", describe_stability_flags(flags))
            return flags

        # ---------------------------------------------------------------------
        # Método terminal de la Fase 2 — entrada directa de la Fase 3
        # ---------------------------------------------------------------------

        def synthesize_basal_state(
            self,
            q: NDArray[np.float64],
            p: NDArray[np.float64],
            df_dt: NDArray[np.float64],
            compute_normal_modes: bool = False,
        ) -> "BasalStateTensor":
            r"""
            **Método terminal de la Fase 2.**

            Computa el estado termodinámico completo del foso K_BASE.
            El campo ``state_vector = [q; p]`` es el **dato primario**
            consumido por ``Phase3_SheafProjection.export_stalk``.

            Parámetros
            ----------
            q, p, df_dt : estado y perturbación de flujo.
            compute_normal_modes : activa Williamson (O(n³) SVD, opt-in).

            Retorna
            -------
            BasalStateTensor
            """
            # Energías y gradientes
            V_q, grad_V_q = self._evaluate_potential_energy(q)
            K_p, grad_K_p = self._compute_kinetic_energy(p)
            H_total: float = V_q + K_p

            grad_H: NDArray[np.float64] = np.concatenate(
                [grad_V_q, grad_K_p]
            )
            grad_H_norm: float = float(la.norm(grad_H, 2))

            # Euler
            euler_residual: float = self._verify_euler_homogeneity(
                q, p, grad_V_q, grad_K_p, H_total
            )

            # Rayleigh
            P_diss: float = self._enforce_rayleigh_dissipation(grad_H)

            # Flyback
            v_fb: float = self._measure_flyback_voltage(df_dt)

            # Campo vectorial
            x_dot: NDArray[np.float64] = self._compute_vector_field(grad_H)

            # Consistencia estructural
            structural_residual: float = (
                self._verify_structural_consistency(
                    grad_H, x_dot, P_diss
                )
            )

            # Certificado de pasividad
            passivity_residual: float = self._verify_passivity_certificate(
                grad_H, x_dot, P_diss
            )

            # Constante de tiempo entrópica
            tau_diss: float = self._dissipation_time_constant(
                H_total, P_diss
            )

            # Retícula Booleana
            flags: StabilityFlags = self._evaluate_stability_flags(
                V_q,
                K_p,
                P_diss,
                v_fb,
                structural_residual,
                passivity_residual,
                euler_residual,
                H_total,
            )

            # Williamson (opt-in)
            omegas: Optional[NDArray[np.float64]] = None
            E0: Optional[float] = None
            if compute_normal_modes:
                omegas, E0 = self.compute_normal_modes()

            state_vector: NDArray[np.float64] = np.concatenate([q, p])

            logger.info(
                "[Fase2] Estado basal: H=%.6e, V=%.6e, K=%.6e, P_diss=%.6e, "
                "‖V_fb‖=%.6e, ‖∇H‖=%.6e, τ_diss=%.3e, %s.",
                H_total,
                V_q,
                K_p,
                P_diss,
                v_fb,
                grad_H_norm,
                tau_diss,
                describe_stability_flags(flags),
            )

            # ================================================================
            # CONTRATO DE INTERFAZ FASE 2 → FASE 3
            # `state_vector = [q; p]` es el argumento directo de
            # Phase3_SheafProjection.export_stalk().
            # ================================================================
            return BasalStateTensor(
                potential_energy=V_q,
                kinetic_energy=K_p,
                total_hamiltonian=H_total,
                dissipated_power=P_diss,
                flyback_voltage_norm=v_fb,
                grad_H_norm=grad_H_norm,
                vector_field=x_dot,
                euler_homogeneity_residual=euler_residual,
                structural_consistency_residual=structural_residual,
                passivity_residual=passivity_residual,
                dissipation_time_constant=tau_diss,
                stability_flags=flags,
                is_thermodynamically_stable=(flags == StabilityFlags.ALL),
                normal_mode_frequencies=omegas,
                zero_point_energy=E0,
                state_vector=state_vector,
            )

    # =========================================================================
    # FASE 3 — PROYECCIÓN COHOMOLÓGICA EN HACES: COCADENA APILADA Y HODGE LOCAL
    # =========================================================================

    class Phase3_SheafProjection:
        r"""
        **Fase 3 – Proyección en Haces y Cofrontera Discreta δ_BASE.**

        Recibe el ``TopologicalContext`` de la Fase 1 y el ``state_vector``
        del ``BasalStateTensor`` de la Fase 2.

        Construcción de la cocadena apilada
        ------------------------------------
            δ_metric = block_diag(C̃^{-1/2}, M̃^{-1/2}) ∈ ℝ^{n×n}
            δ_diss   = R_cost^{+1/2}                   ∈ ℝ^{n×n}
            δ_BASE   = [δ_metric ; δ_diss]             ∈ ℝ^{2n×n}
            Δ_BASE   = δ_BASEᵀ δ_BASE = ∇²H + R_cost   (SPD)

        Diagnósticos espectrales exportados
        ------------------------------------
          • gap de Hodge, κ(Δ_BASE)
          • dim armónica = dim ker(δ_metric) ≡ 0
          • dim lossless = betti_0(R)
          • entropía de von Neumann de Spec(Δ_BASE)
          • proxy de Cheeger λ₂/λ_max
        """

        _EPS: Final[float] = _MACHINE_EPS

        def __init__(self, context: "TopologicalContext") -> None:
            r"""
            **Constructor de la Fase 3: continuación directa de la Fase 2.**

            Ensambla δ_metric, δ_diss, δ_BASE y Δ_BASE **una única vez**
            (contexto inmutable); ``export_stalk`` reutiliza los precálculos.
            """
            self._ctx: "TopologicalContext" = context

            dim_q: int = context.dim_q
            dim_p: int = context.dim_p
            n: int = dim_q + dim_p

            delta_metric: NDArray[np.float64] = np.zeros(
                (n, n), dtype=np.float64
            )
            delta_metric[:dim_q, :dim_q] = context.C_inv_sqrt
            delta_metric[dim_q:, dim_q:] = context.M_inv_sqrt
            self._delta_metric: NDArray[np.float64] = delta_metric

            self._delta_diss: NDArray[np.float64] = context.R_sqrt.copy()

            self._delta_base: NDArray[np.float64] = np.vstack(
                [self._delta_metric, self._delta_diss]
            )  # (2n, n)

            # Δ_BASE = ∇²H + R_cost
            hessian_block: NDArray[np.float64] = (
                self._delta_metric.T @ self._delta_metric
            )
            self._hodge_laplacian: NDArray[np.float64] = (
                hessian_block + context.R_cost
            )
            self._hodge_laplacian = 0.5 * (
                self._hodge_laplacian + self._hodge_laplacian.T
            )

            # Verificar SPD de Δ_BASE con Cholesky (invariante estructural)
            try:
                la.cholesky(self._hodge_laplacian, lower=True)
            except la.LinAlgError as exc:
                raise SheafCoboundaryError(
                    f"Δ_BASE = ∇²H + R_cost no es SPD. "
                    f"Invariante estructural violado. Error: {exc}"
                ) from exc

            # rank(δ_BASE) = n (columna completa: δ_metric invertible)
            self._rank_delta: int = n

            logger.debug(
                "[Fase3] Precalculado: δ_metric=%s, δ_diss=%s, δ_BASE=%s, "
                "rank_delta=%d.",
                self._delta_metric.shape,
                self._delta_diss.shape,
                self._delta_base.shape,
                self._rank_delta,
            )

        def _verify_hodge_identity(self) -> float:
            r"""
            Verifica δ_BASEᵀ δ_BASE = Δ_BASE.

            Lanza SheafCoboundaryError si el error relativo > 100·ε_mach.
            """
            delta_T_delta: NDArray[np.float64] = (
                self._delta_base.T @ self._delta_base
            )
            residual_F: float = float(
                la.norm(delta_T_delta - self._hodge_laplacian, "fro")
            )
            norm_hodge: float = float(
                la.norm(self._hodge_laplacian, "fro")
            )
            rel_error: float = residual_F / max(norm_hodge, 1.0)
            tol_metric: float = 100.0 * self._EPS

            if rel_error > tol_metric:
                raise SheafCoboundaryError(
                    f"δ_BASE no satisface la identidad de Hodge local. "
                    f"‖δᵀδ − Δ_BASE‖_F / ‖Δ_BASE‖_F = {rel_error:.6e} > "
                    f"tol = {tol_metric:.6e}."
                )
            logger.debug(
                "[Fase3] Identidad de Hodge: rel_error=%.3e, tol=%.3e.",
                rel_error,
                tol_metric,
            )
            return rel_error

        def _compute_hodge_spectrum(
            self,
        ) -> Tuple[float, float, int, float, float]:
            r"""
            Diagonaliza Δ_BASE y retorna:

                (spectral_gap, condition_number, harmonic_dimension,
                 spectral_entropy, cheeger_proxy)

            cheeger_proxy ≜ λ₂ / λ_max ∈ [0, 1]
            """
            eigvals: NDArray[np.float64] = la.eigvalsh(
                self._hodge_laplacian
            )
            eigvals = np.maximum(eigvals, 0.0)  # clamp ruido

            lambda_min: float = float(eigvals[0])
            lambda_second: float = (
                float(eigvals[1]) if len(eigvals) > 1 else lambda_min
            )
            lambda_max: float = float(eigvals[-1])

            spectral_gap: float = max(lambda_second - lambda_min, 0.0)
            condition_number: float = (
                lambda_max / lambda_min
                if lambda_min > self._EPS * max(lambda_max, 1.0)
                else float("inf")
            )

            # dim ker(δ_metric): debe ser 0
            tol_kernel: float = self._EPS * max(
                float(la.norm(self._delta_metric, "fro")), 1.0
            )
            harmonic_dimension: int = int(
                np.sum(la.svdvals(self._delta_metric) <= tol_kernel)
            )

            # Entropía de von Neumann
            total: float = float(np.sum(eigvals))
            if total > 0.0:
                p = eigvals / total
                p_pos = p[p > 0.0]
                spectral_entropy: float = float(
                    -np.sum(p_pos * np.log(p_pos))
                )
            else:
                spectral_entropy = 0.0

            # Proxy de Cheeger
            cheeger_proxy: float = (
                lambda_second / lambda_max if lambda_max > 0.0 else 0.0
            )

            logger.debug(
                "[Fase3] Espectro Hodge: gap=%.6e, κ=%.6e, dim_arm=%d, "
                "S_vN=%.4f, Cheeger≈%.4f.",
                spectral_gap,
                condition_number,
                harmonic_dimension,
                spectral_entropy,
                cheeger_proxy,
            )
            return (
                spectral_gap,
                condition_number,
                harmonic_dimension,
                spectral_entropy,
                cheeger_proxy,
            )

        # ---------------------------------------------------------------------
        # Método terminal de la Fase 3 (salida pública del ecosistema)
        # ---------------------------------------------------------------------

        def export_stalk(
            self, state_x: NDArray[np.float64]
        ) -> "SheafStalk":
            r"""
            **Método terminal de la Fase 3 y del agente completo.**

            Construye el ``SheafStalk`` con δ_BASE verificado, Δ_BASE,
            diagnósticos espectrales y proyecciones sobre cada fibra.

            Parámetros
            ----------
            state_x : NDArray[np.float64], shape (dim_q + dim_p,)
                x = [q; p], típicamente ``BasalStateTensor.state_vector``.

            Retorna
            -------
            SheafStalk
            """
            n: int = self._ctx.dim_q + self._ctx.dim_p
            if state_x.shape != (n,):
                raise DimensionMismatchError(
                    f"state_x debe tener shape ({n},) = (dim_q+dim_p,); "
                    f"se obtuvo {state_x.shape}."
                )

            hodge_residual: float = self._verify_hodge_identity()
            (
                spectral_gap,
                condition_number,
                harmonic_dimension,
                spectral_entropy,
                cheeger_proxy,
            ) = self._compute_hodge_spectrum()

            projected_metric: NDArray[np.float64] = (
                self._delta_metric @ state_x
            )
            projected_diss: NDArray[np.float64] = (
                self._delta_diss @ state_x
            )

            logger.info(
                "[Fase3] SheafStalk exportado: rank_delta=%d, gap=%.3e, "
                "κ(Δ)=%.3e, dim_arm=%d, dim_lossless=%d, S_vN=%.4f, "
                "Cheeger≈%.4f, ‖δ_m x‖=%.6e, ‖δ_d x‖=%.6e.",
                self._rank_delta,
                spectral_gap,
                condition_number,
                harmonic_dimension,
                self._ctx.betti_0_R,
                spectral_entropy,
                cheeger_proxy,
                float(la.norm(projected_metric, 2)),
                float(la.norm(projected_diss, 2)),
            )

            # ================================================================
            # CONTRATO DE SALIDA DEL AGENTE COMPLETO
            # ================================================================
            return SheafStalk(
                delta_base=self._delta_base.copy(),
                delta_metric=self._delta_metric.copy(),
                delta_dissipative=self._delta_diss.copy(),
                hodge_laplacian=self._hodge_laplacian.copy(),
                hodge_identity_residual=hodge_residual,
                hodge_spectral_gap=spectral_gap,
                hodge_condition_number=condition_number,
                harmonic_dimension=harmonic_dimension,
                lossless_subspace_dimension=self._ctx.betti_0_R,
                state_vector=state_x.copy(),
                projected_state_metric=projected_metric,
                projected_state_dissipative=projected_diss,
                rank_delta=self._rank_delta,
                spectral_entropy_hodge=spectral_entropy,
                cheeger_proxy=cheeger_proxy,
            )

    # =========================================================================
    # INTERFAZ PÚBLICA DEL AGENTE
    # =========================================================================

    def synthesize_basal_hamiltonian(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        df_dt: NDArray[np.float64],
        compute_normal_modes: bool = False,
    ) -> BasalStateTensor:
        r"""
        Punto de entrada público para la evaluación termodinámica completa.

        Delega en ``Phase2_HamiltonianDynamics.synthesize_basal_state``.

        Parámetros
        ----------
        q, p, df_dt : estado y perturbación de flujo.
        compute_normal_modes : activa Williamson (opt-in).

        Retorna
        -------
        BasalStateTensor
        """
        return self.phase2.synthesize_basal_state(
            q=q,
            p=p,
            df_dt=df_dt,
            compute_normal_modes=compute_normal_modes,
        )

    def export_sheaf_stalk(
        self, state_x: NDArray[np.float64]
    ) -> SheafStalk:
        r"""
        Exporta el Stalk del haz celular y la cofrontera δ_BASE.

        Instancia la Fase 3 perezosamente en la primera llamada.

        Parámetros
        ----------
        state_x : NDArray[np.float64], shape (dim_q + dim_p,)
            Preferiblemente ``BasalStateTensor.state_vector``.

        Retorna
        -------
        SheafStalk
        """
        if self.phase3 is None:
            self.phase3 = KBaseThermodynamicAgent.Phase3_SheafProjection(
                context=self.context
            )
            logger.info(
                "[KBaseThermodynamicAgent] Phase3_SheafProjection "
                "instanciada (lazy init). rank_delta=%d.",
                self.phase3._rank_delta,
            )
        return self.phase3.export_stalk(state_x=state_x)

    def evaluate_full_pipeline(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        df_dt: NDArray[np.float64],
        compute_normal_modes: bool = False,
    ) -> Tuple[BasalStateTensor, SheafStalk]:
        r"""
        Atajo de conveniencia: Fase 2 + Fase 3 en una sola llamada.

        Retorna
        -------
        Tuple[BasalStateTensor, SheafStalk]
        """
        basal = self.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
            compute_normal_modes=compute_normal_modes,
        )
        stalk = self.export_sheaf_stalk(basal.state_vector)
        return basal, stalk


# =============================================================================
# Utilidad local (evita importar math solo por isfinite en un punto)
# =============================================================================


def math_isfinite(x: float) -> bool:
    """True si x es finito (no NaN, no ±Inf)."""
    return bool(np.isfinite(x))
# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo : KApex Electrodynamic Agent (Director de Retorno y Expansión)        |
| Ruta   : app/agents/alpha/kapex/kapex_electrodynamic_agent.py                |
| Versión: 6.0.0-Rigorous-Gauge-Curvature-Sheaf                                |
+==============================================================================+

NATURALEZA CIBER-FÍSICA Y ÓPTICA GEOMÉTRICA
---------------------------------------------
Este módulo consagra el Ápice Estratégico como un Endofuntor de Campo de Calibre
que inyecta Fuerza Electromotriz (FEM), resuelve la refracción y audita el
retorno mediante una curvatura de Yang-Mills genuinamente antisimétrica.

CAMBIOS ESTRUCTURALES RESPECTO A v5.0.0 (evolución granular)
--------------------------------------------------------------
1. CURVATURA DE CALIBRE CORREGIDA: la formulación anterior sumaba un término
   antisimétrico (A−Aᵀ) con un conmutador [A,Aᵀ] que es **siempre simétrico**
   (identidad algebraica exacta), produciendo un tensor F que NO era una
   2-forma de curvatura válida (F debe satisfacer F=−Fᵀ, pues el grupo de
   holonomía compatible con G_μν es O(n), cuya álgebra 𝔰𝔬(n) es antisimétrica).
   Se introduce una **plaqueta de dos direcciones** (A₁, A₂) con:
   \[ F = \big(A_2^{a}-A_1^{a}\big) + [A_1^{a}, A_2^{a}], \qquad A_i^a := \tfrac12(A_i-A_i^\top) \]
   que es antisimétrica por construcción (el conmutador de dos matrices
   antisimétricas es antisimétrico — demostración incluida en el docstring
   de ``_compute_curvature``).

2. DIAGNÓSTICO OPCIONAL DE COVARIANZA DE CALIBRE (Fase 2):
   ``verify_gauge_covariance`` comprueba que S_YM es invariante bajo la
   transformación de calibre A_i ↦ Q A_i Qᵀ para toda isometría Q de G_μν
   (Q^⊤GQ=G) — propiedad definitoria de toda teoría de Yang-Mills.

3. FIBRA DISIPATIVA ACTIVADA EN LA COFRONTERA (Fase 3): R_sqrt, calculado
   en Fase 1 pero nunca usado en v5.0.0, ahora construye una fibra
   disipativa genuina δ_diss = R_sqrt·δ_metric, con Laplaciano de Hodge
   local Δ_APEX = I + δ_dissᵀδ_diss, en paridad estructural con el
   ``SheafStalk`` de ``KBaseThermodynamicAgent``.

4. ÁLGEBRA DE BOOLE DE VIABILIDAD: ``ApexViabilityFlags(enum.Flag)``
   sustituye al booleano monolítico ``is_electrodynamically_viable``.

5. CORRECCIONES NUMÉRICAS: número de condición κ(G_μν) reutilizado
   correctamente (antes se re-derivaba de forma inválida desde los
   elementos diagonales de Cholesky); prueba de supresión de calibre
   ahora es invariante de escala; P_diss se calcula como ‖R_sqrt·∇H‖²
   (no-negativo por construcción); Cholesky de G_μν regularizado con
   jitter de Tikhonov adaptativo.

ECUACIÓN EIKONAL DE ABSORCIÓN:
\[ G^{\mu\nu} \partial_\mu S \partial_\nu S = n^2(\sigma^*) \]

FLUJO EXERGÉTICO DE POYNTING:
\[ P_{exergia} = E\cdot H - \|R_{cost}^{1/2}\nabla H\|^2 \ge 0 \]

CURVATURA DE YANG-MILLS (CORREGIDA):
\[ S_{YM} = \frac{1}{2}\,\mathrm{Tr}\big(F^\top G_{\mu\nu} F\, G^{\mu\nu}\big),
   \qquad F \in \mathfrak{so}(n) \]
"""
from __future__ import annotations

# Biblioteca estándar
import enum
import logging
from typing import Optional, Tuple

# Álgebra numérica de alta precisión
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Estructuras de datos inmutables
from dataclasses import dataclass

# Dependencias arquitectónicas del ecosistema APU Filter
try:
    from app.core.mic_algebra import CategoricalState, Morphism
except ImportError:
    class CategoricalState:  # type: ignore[no-redef]
        """Stub: estado categórico del ecosistema MIC."""

    class Morphism:  # type: ignore[no-redef]
        """Stub: morfismo funtorial del ecosistema MIC."""


# Logger del módulo
logger = logging.getLogger("MIC.Alpha.KApexElectrodynamicAgent")

# Precisión de máquina IEEE-754 double, reutilizada en todo el módulo
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)


#
# SECCIÓN 0 — EXCEPCIONES ELECTRODINÁMICAS ESTRICTAS
#


class ElectrodynamicApexError(Exception):
    """
    Excepción categórica raíz para violaciones en el Estrato K_APEX.

    Toda excepción de este módulo hereda de esta clase, permitiendo que
    los manejadores de nivel superior capturen cualquier fallo
    electrodinámico con un único ``except ElectrodynamicApexError``.
    """


class ApexDimensionError(ElectrodynamicApexError):
    """
    Lanzada cuando las dimensiones de las matrices constitutivas son
    inconsistentes entre sí o con los vectores de campo recibidos.
    """


class ApexParameterError(ElectrodynamicApexError):
    """
    Lanzada cuando un parámetro escalar de control (κ_max, eikonal_slack,
    holonomy_tol_rel) está fuera de su rango matemáticamente admisible,
    detectado en tiempo de construcción antes de procesar ninguna matriz.
    """


class ApexSymmetryError(ElectrodynamicApexError):
    """
    Lanzada cuando G_μν o R_cost violan su propiedad de simetría,
    con diagnóstico cuantitativo normalizado al Frobenius.
    """


class ApexConditionError(ElectrodynamicApexError):
    """
    Lanzada cuando el número de condición espectral κ(G_μν) o κ(G_inv)
    supera el umbral admisible, comprometiendo la coherencia de la variedad.
    """


class MetricInverseError(ElectrodynamicApexError):
    """
    Lanzada cuando G_inv no es la inversa de G_μν dentro de la tolerancia
    de máquina relativa, escalada por el número de condición real κ(G_μν)
    (análisis de error hacia atrás de Wilkinson).
    """


class GaugePotentialError(ElectrodynamicApexError):
    """
    Lanzada cuando el estrés estructural Tr(G_μν) colapsa el factor de
    supresión de calibre por debajo de la resolución de máquina,
    independientemente de la magnitud del diferencial dΦ inyectado.

    Condición de disparo (invariante de escala):
        exp(−½·Tr(G_μν)) < ε_mach
    """


class EikonalRefractionError(ElectrodynamicApexError):
    """
    Lanzada cuando el mercado objetivo es topológicamente inalcanzable:
    la norma riemanniana del gradiente de fase no satisface la ecuación
    Eikonal con el índice de refracción calculado.
    """


class FinancialBlackHoleError(ElectrodynamicApexError):
    """
    Lanzada cuando el flujo de Poynting (ingresos) no compensa la
    disipación termodinámica (costes): P_exergia < −tol_poynting.
    """


class HolonomyVetoError(ElectrodynamicApexError):
    """
    Lanzada cuando la acción de Yang-Mills discreta S_YM supera el umbral
    relativo a ‖A_gauge‖²_F, revelando curvatura de calibre no nula
    (ciclos parásitos en la logística estratégica).
    """


class GaugeCovarianceError(ElectrodynamicApexError):
    """
    Lanzada por el diagnóstico opcional ``verify_gauge_covariance`` cuando
    S_YM no es invariante bajo una transformación de calibre genuina
    (Q isometría de G_μν), indicando un error de implementación en la
    construcción de la curvatura (violación de un invariante algebraico
    exacto, no un fenómeno físico).
    """


class SheafMetricError(ElectrodynamicApexError):
    """
    Lanzada cuando la cofrontera δ_{APEX} no satisface la identidad
    de la métrica de Hodge local: δ_metric^⊤ G_μν δ_metric ≈ I.
    """


#
# SECCIÓN 1 — ÁLGEBRA DE BOOLE DE VIABILIDAD ELECTRODINÁMICA
#


class ApexViabilityFlags(enum.Flag):
    r"""
    Retícula de Boole de predicados de viabilidad electrodinámica.

    Análoga a ``StabilityFlags`` de ``KBaseThermodynamicAgent``: cada
    bandera es un predicado independiente, y su conjunción determina la
    viabilidad global sin perder el diagnóstico granular de qué condición
    falló específicamente.

    Miembros
    --------
    GAUGE_INJECTION_NONTRIVIAL
        exp(−½Tr(G_μν)) ≥ margen de seguridad (no colapso estructural).
    EIKONAL_MARGIN_SOUND
        ‖∂S‖²_{G_inv} ≥ n²(σ*)·(1 − eikonal_slack/2)  (margen holgado,
        más estricto que el límite duro que dispara la excepción).
    EXERGY_NONNEGATIVE
        P_exergia ≥ 0 (Segunda Ley, verificado independientemente).
    HOLONOMY_TRIVIAL
        S_YM ≤ tol_ym (curvatura de calibre despreciable).
    CURVATURE_ANTISYMMETRIC
        ‖F+Fᵀ‖_F/‖F‖_F ≤ tol (invariante algebraico exacto de F∈𝔰𝔬(n)).
    METRIC_WELL_CONDITIONED
        κ(G_μν) ≤ ½·κ_max (alerta temprana de degeneración métrica).
    """

    NONE = 0
    GAUGE_INJECTION_NONTRIVIAL = enum.auto()
    EIKONAL_MARGIN_SOUND = enum.auto()
    EXERGY_NONNEGATIVE = enum.auto()
    HOLONOMY_TRIVIAL = enum.auto()
    CURVATURE_ANTISYMMETRIC = enum.auto()
    METRIC_WELL_CONDITIONED = enum.auto()
    ALL = (
        GAUGE_INJECTION_NONTRIVIAL
        | EIKONAL_MARGIN_SOUND
        | EXERGY_NONNEGATIVE
        | HOLONOMY_TRIVIAL
        | CURVATURE_ANTISYMMETRIC
        | METRIC_WELL_CONDITIONED
    )


def describe_viability_flags(flags: ApexViabilityFlags) -> str:
    """Serializa ``ApexViabilityFlags`` a cadena legible, listando predicados satisfechos/violados."""
    satisfied = [f.name for f in ApexViabilityFlags if f not in (ApexViabilityFlags.NONE, ApexViabilityFlags.ALL) and f in flags]
    violated = [f.name for f in ApexViabilityFlags if f not in (ApexViabilityFlags.NONE, ApexViabilityFlags.ALL) and f not in flags]
    return (
        f"SATISFECHOS={satisfied or 'ninguno'} | "
        f"VIOLADOS={violated or 'ninguno'} | "
        f"VIABLE_TOTAL={flags == ApexViabilityFlags.ALL}"
    )


#
# SECCIÓN 2 — ESTRUCTURAS INMUTABLES (DTOs TENSORIALES)
#


@dataclass(frozen=True, slots=True)
class ApexPreparationContext:
    r"""
    Contexto inmutable producido por la **Fase 1** (Validación Métrica).

    Atributos
    ----------
    G_mu_nu, G_inv, R_cost : NDArray[np.float64], shape (n, n)
        Matrices constitutivas validadas, copias inmutables.
    L_G : NDArray[np.float64], shape (n, n)
        Factor de Cholesky (regularizado) de G_μν: G_μν ≈ L_G·L_Gᵀ.
    R_sqrt : NDArray[np.float64], shape (n, n)
        Raíz cuadrada espectral de R_cost. Consumida en Fase 3 (fibra
        disipativa) — a diferencia de v5.0.0, ya no es código muerto.
    kappa_G, kappa_G_inv : float
        Números de condición espectral de G_μν y G_inv respectivamente.
        Por la identidad exacta κ(A⁻¹)=κ(A), ambos deben coincidir hasta
        precisión de máquina si G_inv es genuinamente la inversa de G_μν;
        su divergencia es un diagnóstico temprano de inconsistencia.
    epsilon_G : float
        Jitter de Tikhonov aplicado en la factorización de Cholesky de
        G_μν (0.0 si no fue necesario).
    rank_R : int
        Rango numérico de R_cost.
    spectral_gap_R : float
        Brecha espectral λ₂−λ₁ de R_cost (teoría espectral de grafos).
    betti_0_R : int
        Dimensión del núcleo de R_cost = n − rank_R (subespacio de
        disipación nula, heredado en la Fase 3 como ``lossless_subspace_dimension``).
    inverse_residual : float
        ‖G·G_inv − I‖_F / n: residuo normalizado de consistencia métrica.
    dim : int
        Dimensión del espacio de calibre n.
    """

    G_mu_nu: NDArray[np.float64]
    G_inv: NDArray[np.float64]
    R_cost: NDArray[np.float64]
    L_G: NDArray[np.float64]
    R_sqrt: NDArray[np.float64]
    kappa_G: float
    kappa_G_inv: float
    epsilon_G: float
    rank_R: int
    spectral_gap_R: float
    betti_0_R: int
    inverse_residual: float
    dim: int


@dataclass(frozen=True, slots=True)
class ApexStateTensor:
    r"""
    Tensor inmutable que encapsula el estado electrodinámico de la cúspide.

    Producido por la **Fase 2** (Síntesis Electrodinámica).

    Atributos
    ----------
    gauge_injection_vector : NDArray[np.float64], shape (n,)
        s_val = dΦ · exp(−½ Tr(G_μν)).
    suppression_factor : float
        exp(−½ Tr(G_μν)) ∈ (0, 1].
    fermat_refractive_index : float
        n(σ*) = 1 + tanh(α·σ*).
    eikonal_norm_sq : float
        G^μν ∂_μS ∂_νS.
    poynting_income, poynting_dissipation, poynting_exergy_flux : float
        P_in, P_diss = ‖R_sqrt∇H‖² (no-negativo por construcción), P_exergia.
    yang_mills_action : float
        S_YM = ½ Tr(Fᵀ G F G⁻¹), con F ∈ 𝔰𝔬(n) genuino (ver ``_compute_curvature``).
    curvature_antisymmetry_residual : float
        ‖F+Fᵀ‖_F/max(‖F‖_F,1): verificación del invariante algebraico
        exacto F=−Fᵀ. Debe ser O(ε_mach).
    viability_flags : ApexViabilityFlags
        Retícula de Boole de predicados de viabilidad.
    is_electrodynamically_viable : bool
        Equivalente a ``viability_flags == ApexViabilityFlags.ALL``.
    gauge_covariance_residual : Optional[float]
        |S_YM(A) − S_YM(QAQᵀ)| / max(S_YM(A),1), calculado sólo si se
        solicitó el diagnóstico opcional de covarianza de calibre.
    """

    gauge_injection_vector: NDArray[np.float64]
    suppression_factor: float
    fermat_refractive_index: float
    eikonal_norm_sq: float
    poynting_income: float
    poynting_dissipation: float
    poynting_exergy_flux: float
    yang_mills_action: float
    curvature_antisymmetry_residual: float
    viability_flags: ApexViabilityFlags
    is_electrodynamically_viable: bool
    gauge_covariance_residual: Optional[float]


@dataclass(frozen=True, slots=True)
class SheafStalkApex:
    r"""
    Fibrado celular exportado para el cálculo global del Laplaciano de Haz.

    Producido por la **Fase 3** (Proyección en Haces).

    Construcción (activando la fibra disipativa, antes código muerto)
    ---------------------------------------------------------------------
        δ_metric = G_μν^{-1/2} = L_G^{-⊤}                    ∈ ℝ^{n×n}
        δ_diss   = R_sqrt · δ_metric                          ∈ ℝ^{n×n}
        δ_APEX   = [ δ_metric ; δ_diss ]                      ∈ ℝ^{2n×n}
        Δ_APEX   = δ_APEXᵀ δ_APEX = I_n + δ_dissᵀ δ_diss       (SPD)

    Δ_APEX combina la "planitud" del marco normalizado (I_n, tras
    deshacer la métrica G_μν vía δ_metric) con la disipación pulled-back
    al mismo marco ortonormal, en paridad estructural directa con el
    Laplaciano de Hodge local ∇²H+R_cost de ``KBaseThermodynamicAgent``.

    Atributos
    ----------
    delta_apex : NDArray[np.float64], shape (2n, n)
        Cocadena apilada completa.
    delta_metric, delta_dissipative : NDArray[np.float64], shape (n, n)
        Componentes hacia cada fibra incidente.
    hodge_laplacian : NDArray[np.float64], shape (n, n)
        Δ_APEX = I_n + δ_dissᵀδ_diss.
    hodge_metric_residual : float
        ‖δ_metricᵀ G_μν δ_metric − I‖_F/n. Debe ser O(ε_mach).
    hodge_spectral_gap : float
        λ₂(Δ_APEX) − λ₁(Δ_APEX).
    hodge_condition_number : float
        κ(Δ_APEX).
    lossless_subspace_dimension : int
        n − rank(R_cost), heredado de ``betti_0_R``.
    source_injection : NDArray[np.float64], shape (n,)
        s_val del estado electrodinámico actual.
    projected_source_metric, projected_source_dissipative : NDArray[np.float64]
        δ_metric·s_val y δ_diss·s_val respectivamente.
    rank_delta : int
        Rango de δ_APEX = n (columna completa, δ_metric invertible).
    """

    delta_apex: NDArray[np.float64]
    delta_metric: NDArray[np.float64]
    delta_dissipative: NDArray[np.float64]
    hodge_laplacian: NDArray[np.float64]
    hodge_metric_residual: float
    hodge_spectral_gap: float
    hodge_condition_number: float
    lossless_subspace_dimension: int
    source_injection: NDArray[np.float64]
    projected_source_metric: NDArray[np.float64]
    projected_source_dissipative: NDArray[np.float64]
    rank_delta: int


#
# SECCIÓN 3 — ORQUESTADOR: KApexElectrodynamicAgent
#             Tres fases anidadas de rigor creciente
#


class KApexElectrodynamicAgent(Morphism):
    r"""
    Orquestador Funtorial del Ápice Estratégico K_{APEX}.

    Garantiza la invarianza de Gauge y audita el bucle de holonomía global
    mediante tres clases anidadas que operan en cascada estricta:

        Phase1_MetricValidation          (métrica + Cholesky regularizado + espectro)
            ↓  ApexPreparationContext
        Phase2_ElectrodynamicSynthesis   (curvatura 𝔰𝔬(n) + covarianza + Boole)
            ↓  ApexStateTensor
        Phase3_SheafProjection           (fibra métrica + fibra disipativa)
            ↓  SheafStalkApex

    Parámetros de Construcción
    --------------------------
    G_mu_nu, G_inv, R_cost : NDArray[np.float64], shape (n, n)
        Matrices constitutivas (ver Fase 1 para condiciones formales).
    kappa_max : float, default 1e10
        Umbral de número de condición espectral κ(G_μν). Debe ser > 1.
    eikonal_slack : float, default 0.1
        Tolerancia relativa de la ecuación Eikonal, en [0, 1).
    holonomy_tol_rel : float, default 1e-6
        Tolerancia relativa para la acción de Yang-Mills. Debe ser > 0.
    """

    FRIENDLY_NAME: str = "Director de Retorno y Expansión de Mercado"

    def __init__(
        self,
        G_mu_nu: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        R_cost: NDArray[np.float64],
        kappa_max: float = 1.0e10,
        eikonal_slack: float = 0.1,
        holonomy_tol_rel: float = 1.0e-6,
    ) -> None:
        r"""
        Inicializa las matrices constitutivas y ejecuta la Fase 1 de inmediato.

        Lanza
        -----
        ApexParameterError
            Si kappa_max ≤ 1, eikonal_slack ∉ [0,1) u holonomy_tol_rel ≤ 0.
        ApexDimensionError, ApexSymmetryError, ApexConditionError, MetricInverseError
            Propagadas desde la Fase 1 si alguna propiedad es violada.
        """
        if kappa_max <= 1.0:
            raise ApexParameterError(f"kappa_max debe ser > 1; se obtuvo {kappa_max}.")
        if not (0.0 <= eikonal_slack < 1.0):
            raise ApexParameterError(f"eikonal_slack debe estar en [0,1); se obtuvo {eikonal_slack}.")
        if holonomy_tol_rel <= 0.0:
            raise ApexParameterError(f"holonomy_tol_rel debe ser > 0; se obtuvo {holonomy_tol_rel}.")

        self.kappa_max: float = kappa_max
        self.eikonal_slack: float = eikonal_slack
        self.holonomy_tol_rel: float = holonomy_tol_rel

        # Fase 1: Validación Métrica y de Calibre (inmediata)
        self.phase1: KApexElectrodynamicAgent.Phase1_MetricValidation = (
            KApexElectrodynamicAgent.Phase1_MetricValidation(
                G_mu_nu=G_mu_nu, G_inv=G_inv, R_cost=R_cost, kappa_max=kappa_max,
            )
        )
        self.context: ApexPreparationContext = self.phase1.build_context()

        # Fase 2: Síntesis Electrodinámica (instanciación inmediata)
        self.phase2: KApexElectrodynamicAgent.Phase2_ElectrodynamicSynthesis = (
            KApexElectrodynamicAgent.Phase2_ElectrodynamicSynthesis(
                context=self.context,
                eikonal_slack=self.eikonal_slack,
                holonomy_tol_rel=self.holonomy_tol_rel,
                kappa_max=self.kappa_max,
            )
        )

        # Fase 3: instanciación perezosa
        self.phase3: Optional[KApexElectrodynamicAgent.Phase3_SheafProjection] = None

        logger.info(
            "[KApexElectrodynamicAgent] Inicializado: dim=%d, κ(G)=%.3e, "
            "rank(R)=%d, betti_0(R)=%d, inv_residual=%.3e.",
            self.context.dim, self.context.kappa_G, self.context.rank_R,
            self.context.betti_0_R, self.context.inverse_residual,
        )

    #
    # ==========================================================================
    # FASE 1 — VALIDACIÓN MÉTRICA, CHOLESKY REGULARIZADO Y DIAGNÓSTICO ESPECTRAL
    # ==========================================================================
    #

    class Phase1_MetricValidation:
        r"""
        **Fase 1 – Validación Métrica y de Calibre.**

        Responsabilidades, en orden estricto de ejecución:
          a) Verificar dimensiones y cuadratura de G_μν, G_inv, R_cost.
          b) Verificar simetría de G_μν, G_inv y R_cost.
          c) Verificar SPD de G_μν y calcular κ(G_μν) vía extremos espectrales
             eficientes (``eigh(subset_by_index=...)``).
          d) Factorizar G_μν por Cholesky **regularizado** (Tikhonov adaptativo).
          e) Verificar SPD de G_inv y calcular κ(G_inv) (debe coincidir con
             κ(G_μν) por la identidad exacta κ(A⁻¹)=κ(A)).
          f) Verificar consistencia métrica G·G_inv ≈ I, con tolerancia
             correctamente escalada por el κ(G_μν) **real** ya computado.
          g) Verificar PSD de R_cost, raíz espectral, brecha espectral y β₀.
          h) Empaquetar en ``ApexPreparationContext``.
        """

        _EPS: float = _MACHINE_EPS

        def __init__(
            self,
            G_mu_nu: NDArray[np.float64],
            G_inv: NDArray[np.float64],
            R_cost: NDArray[np.float64],
            kappa_max: float = 1.0e10,
        ) -> None:
            r"""Almacena referencias sin copiar; las copias ocurren en ``build_context``."""
            self._G: NDArray[np.float64] = G_mu_nu
            self._G_inv: NDArray[np.float64] = G_inv
            self._R: NDArray[np.float64] = R_cost
            self._kappa_max: float = kappa_max

        #
        # Métodos privados de validación (orden lógico de ejecución)
        #

        def _check_dimensions(self) -> int:
            r"""
            Verifica que G_μν, G_inv y R_cost son cuadradas y de la misma
            dimensión n.

            Retorna
            -------
            int
                Dimensión n del espacio de calibre.

            Lanza
            -----
            ApexDimensionError
            """
            for mat, name in [(self._G, "G_mu_nu"), (self._G_inv, "G_inv"), (self._R, "R_cost")]:
                if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                    raise ApexDimensionError(
                        f"'{name}' debe ser cuadrada 2D; se obtuvo shape={mat.shape}."
                    )
            n: int = self._G.shape[0]
            if self._G_inv.shape[0] != n:
                raise ApexDimensionError(
                    f"G_inv debe tener shape ({n},{n}) coherente con G_μν; se obtuvo {self._G_inv.shape}."
                )
            if self._R.shape[0] != n:
                raise ApexDimensionError(
                    f"R_cost debe tener shape ({n},{n}) coherente con G_μν; se obtuvo {self._R.shape}."
                )
            logger.debug("[Fase1] Dimensiones verificadas: n=%d.", n)
            return n

        def _validate_symmetry(self, A: NDArray[np.float64], name: str) -> None:
            r"""Verifica A=Aᵀ con tolerancia tol=ε_mach·max(‖A‖_F,1)."""
            norm_A: float = float(la.norm(A, "fro"))
            tol: float = self._EPS * max(norm_A, 1.0)
            residual: float = float(la.norm(A - A.T, "fro"))
            if residual > tol:
                raise ApexSymmetryError(
                    f"La matriz '{name}' no es simétrica. ‖A-Aᵀ‖_F={residual:.6e}, tol={tol:.6e}, "
                    f"asimetría relativa={residual/max(norm_A,1e-300):.6e}."
                )
            logger.debug("[Fase1] Simetría de '%s': residual=%.3e, tol=%.3e.", name, residual, tol)

        def _validate_spd(self, A: NDArray[np.float64], name: str) -> Tuple[float, float, float]:
            r"""
            Verifica SPD de A y calcula κ(A)=λ_max/λ_min explotando la
            simetría vía extremos espectrales eficientes
            (``eigh(subset_by_index=...)``), evitando diagonalización
            completa O(n³) cuando sólo se requieren los extremos.

            Retorna
            -------
            Tuple[float, float, float]
                (kappa, lambda_min, lambda_max).

            Lanza
            -----
            ElectrodynamicApexError
                Si A no es SPD (λ_min ≤ tol_pd).
            ApexConditionError
                Si κ(A) > kappa_max.
            """
            A_sym: NDArray[np.float64] = 0.5 * (A + A.T)
            n: int = A_sym.shape[0]
            if n == 1:
                lambda_min = lambda_max = float(A_sym[0, 0])
            else:
                lambda_min = float(la.eigh(A_sym, subset_by_index=[0, 0], eigvals_only=True)[0])
                lambda_max = float(la.eigh(A_sym, subset_by_index=[n - 1, n - 1], eigvals_only=True)[0])

            tol_pd: float = self._EPS * max(abs(lambda_max), 1.0)
            if lambda_min <= tol_pd:
                raise ElectrodynamicApexError(
                    f"'{name}' no es Definida Positiva (SPD). λ_min={lambda_min:.6e} ≤ tol_pd={tol_pd:.6e}. "
                    f"λ_max={lambda_max:.6e}."
                )

            kappa: float = lambda_max / lambda_min
            if kappa > self._kappa_max:
                raise ApexConditionError(
                    f"'{name}' está numéricamente mal condicionada: κ={kappa:.6e} > κ_max={self._kappa_max:.6e}. "
                    f"λ_min={lambda_min:.6e}, λ_max={lambda_max:.6e}."
                )
            logger.debug("[Fase1] SPD '%s': κ=%.3e, λ_min=%.3e, λ_max=%.3e.", name, kappa, lambda_min, lambda_max)
            return kappa, lambda_min, lambda_max

        def _cholesky_regularized(
            self, A: NDArray[np.float64], name: str, max_attempts: int = 6
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Cholesky A=L·Lᵀ con regularización de Tikhonov adaptativa:

                A_τ = A + τ·I,   τ_0 = ε_mach·tr(A)/n,   τ_{k+1}=10·τ_k

            Retorna
            -------
            Tuple[NDArray[np.float64], float]
                (L, tau_final).

            Lanza
            -----
            ElectrodynamicApexError
                Si tras ``max_attempts`` reintentos la factorización sigue fallando.
            """
            A_sym: NDArray[np.float64] = 0.5 * (A + A.T)
            n: int = A_sym.shape[0]
            trace_scale: float = float(np.trace(A_sym)) / max(n, 1)
            tau: float = 0.0
            I_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)

            for attempt in range(max_attempts + 1):
                try:
                    L: NDArray[np.float64] = la.cholesky(A_sym + tau * I_n, lower=True)
                    if attempt > 0:
                        logger.warning(
                            "[Fase1] Regularización de Tikhonov en '%s': τ=%.3e tras %d intento(s).",
                            name, tau, attempt,
                        )
                    return L, tau
                except la.LinAlgError:
                    tau = self._EPS * max(trace_scale, 1.0) if tau == 0.0 else tau * 10.0

            raise ElectrodynamicApexError(
                f"Fallo persistente de Cholesky en '{name}' tras {max_attempts} reintentos "
                f"(τ_final={tau:.3e}). Indica degeneración estructural."
            )

        def _validate_inverse_consistency(
            self, kappa_G: float, n: int,
        ) -> float:
            r"""
            Verifica G_inv = G_μν⁻¹ con tolerancia basada en el número de
            condición **real** ya calculado espectralmente (a diferencia de
            v5.0.0, que lo re-derivaba de forma matemáticamente inválida a
            partir de la razón de elementos diagonales del factor Cholesky).

            Tolerancia (análisis de error hacia atrás de Wilkinson):
                tol_inv = κ(G_μν) · ε_mach

            Retorna
            -------
            float
                Residuo normalizado ‖G·G_inv − I‖_F / n.

            Lanza
            -----
            MetricInverseError
            """
            prod: NDArray[np.float64] = self._G @ self._G_inv
            I_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)
            residual: float = float(la.norm(prod - I_n, "fro")) / n
            tol_inv: float = kappa_G * self._EPS * n

            if residual > tol_inv:
                raise MetricInverseError(
                    f"G_inv no es la inversa de G_μν. ‖G·G_inv-I‖_F/n={residual:.6e} > "
                    f"tol_inv={tol_inv:.6e} (κ(G)={kappa_G:.3e}). Inconsistencia métrica."
                )
            logger.debug("[Fase1] Consistencia métrica: residual/n=%.3e, tol=%.3e.", residual, tol_inv)
            return residual

        def _validate_psd_and_spectral_diagnostics(
            self, R: NDArray[np.float64], name: str,
        ) -> Tuple[NDArray[np.float64], int, float]:
            r"""
            Verifica R⪰0, calcula raíz espectral R_sqrt, rango numérico y
            brecha espectral λ₂−λ₁ (teoría espectral de grafos).

            Retorna
            -------
            Tuple[NDArray[np.float64], int, float]
                (R_sqrt, rank_R, spectral_gap).

            Lanza
            -----
            ApexSymmetryError
                Si λ_min < −tol_psd (entropía negativa genuina).
            """
            norm_R: float = float(la.norm(R, "fro"))
            tol_psd: float = self._EPS * max(norm_R, 1.0)
            eigvals, eigvecs = la.eigh(0.5 * (R + R.T))
            lambda_min: float = float(eigvals[0])

            if lambda_min < -tol_psd:
                raise ApexSymmetryError(
                    f"'{name}' no es PSD. λ_min={lambda_min:.6e} < -tol={-tol_psd:.6e}. "
                    f"Entropía negativa detectada."
                )

            eigvals_clamped: NDArray[np.float64] = np.maximum(eigvals, 0.0)
            R_sqrt: NDArray[np.float64] = (eigvecs * np.sqrt(eigvals_clamped)[np.newaxis, :]) @ eigvecs.T
            R_sqrt = 0.5 * (R_sqrt + R_sqrt.T)
            rank_R: int = int(np.sum(eigvals_clamped > tol_psd))
            spectral_gap: float = float(eigvals_clamped[1] - eigvals_clamped[0]) if len(eigvals_clamped) > 1 else 0.0

            logger.debug(
                "[Fase1] PSD '%s': rank=%d/%d, λ_min=%.3e, λ_max=%.3e, gap=%.3e.",
                name, rank_R, len(eigvals), lambda_min, float(eigvals[-1]), spectral_gap,
            )
            return R_sqrt, rank_R, spectral_gap

        #
        # Método terminal de la Fase 1 — entrada directa de la Fase 2
        #

        def build_context(self) -> "ApexPreparationContext":
            r"""
            **Método terminal de la Fase 1.**

            Flujo interno
            -------------
            1. Verificación dimensional.
            2. Simetría de G_μν, G_inv y R_cost.
            3. SPD y κ(G_μν) → Cholesky regularizado L_G.
            4. SPD y κ(G_inv) (cross-check: debe coincidir con κ(G_μν)).
            5. Consistencia métrica G·G_inv≈I, con tolerancia correcta.
            6. PSD de R_cost + raíz espectral + diagnóstico espectral.
            7. Empaquetado en ApexPreparationContext.

            Retorna
            -------
            ApexPreparationContext
            """
            # Paso 1
            n: int = self._check_dimensions()

            # Paso 2
            self._validate_symmetry(self._G, "G_mu_nu")
            self._validate_symmetry(self._G_inv, "G_inv")
            self._validate_symmetry(self._R, "R_cost")

            # Paso 3: SPD + κ(G_μν) + Cholesky regularizado
            kappa_G, _, _ = self._validate_spd(self._G, "G_mu_nu")
            L_G, epsilon_G = self._cholesky_regularized(self._G, "G_mu_nu")

            # Paso 4: SPD + κ(G_inv) (validación independiente + cross-check exacto)
            kappa_G_inv, _, _ = self._validate_spd(self._G_inv, "G_inv")
            kappa_mismatch_rel: float = abs(kappa_G_inv - kappa_G) / max(kappa_G, 1.0)
            if kappa_mismatch_rel > 1.0e-3:
                logger.warning(
                    "[Fase1] κ(G_inv)=%.6e difiere de κ(G_μν)=%.6e en %.2f%% "
                    "(identidad exacta κ(A⁻¹)=κ(A) sugiere posible inconsistencia).",
                    kappa_G_inv, kappa_G, 100.0 * kappa_mismatch_rel,
                )

            # Paso 5: Consistencia métrica con tolerancia basada en κ(G) real
            inv_residual: float = self._validate_inverse_consistency(kappa_G, n)

            # Paso 6: PSD de R_cost + diagnóstico espectral
            R_sqrt, rank_R, spectral_gap_R = self._validate_psd_and_spectral_diagnostics(self._R, "R_cost")
            betti_0_R: int = n - rank_R

            # Paso 7: Empaquetado
            context = ApexPreparationContext(
                G_mu_nu=self._G.copy(),
                G_inv=self._G_inv.copy(),
                R_cost=self._R.copy(),
                L_G=L_G,
                R_sqrt=R_sqrt,
                kappa_G=kappa_G,
                kappa_G_inv=kappa_G_inv,
                epsilon_G=epsilon_G,
                rank_R=rank_R,
                spectral_gap_R=spectral_gap_R,
                betti_0_R=betti_0_R,
                inverse_residual=inv_residual,
                dim=n,
            )

            logger.info(
                "[Fase1] ApexPreparationContext ensamblado: dim=%d, κ(G)=%.3e, "
                "rank(R)=%d, betti_0(R)=%d, inv_res=%.3e.",
                n, kappa_G, rank_R, betti_0_R, inv_residual,
            )
            return context

    #
    # ==========================================================================
    # FASE 2 — SÍNTESIS ELECTRODINÁMICA: CURVATURA 𝔰𝔬(n), COVARIANZA Y BOOLE
    # ==========================================================================
    #

    class Phase2_ElectrodynamicSynthesis:
        r"""
        **Fase 2 – Síntesis Electrodinámica.**

        Recibe el ``ApexPreparationContext`` de la Fase 1 y ejecuta:

          1. **Inyección de Potencial de Gauge** (invariante de escala).
          2. **Refracción Eikonal de Mercado**.
          3. **Exergía de Poynting** (P_diss=‖R_sqrt∇H‖², no-negativo exacto).
          4. **Curvatura de Yang-Mills genuina** F∈𝔰𝔬(n) sobre una plaqueta
             de dos direcciones (A₁,A₂), con verificación algebraica de
             antisimetría y diagnóstico opcional de covarianza de calibre.
        """

        _EPS: float = _MACHINE_EPS

        def __init__(
            self,
            context: "ApexPreparationContext",
            eikonal_slack: float,
            holonomy_tol_rel: float,
            kappa_max: float,
        ) -> None:
            r"""**Constructor de la Fase 2: continuación directa de la Fase 1.**"""
            self._ctx: "ApexPreparationContext" = context
            self._eikonal_slack: float = eikonal_slack
            self._holonomy_tol_rel: float = holonomy_tol_rel
            self._kappa_max: float = kappa_max

            self._trace_G: float = float(np.trace(context.G_mu_nu))
            self._suppression_factor: float = float(np.exp(-0.5 * self._trace_G))

            logger.debug(
                "[Fase2] Inicializada: dim=%d, Tr(G)=%.6e, suppression=%.6e, "
                "eikonal_slack=%.3f, holo_tol=%.3e.",
                context.dim, self._trace_G, self._suppression_factor,
                eikonal_slack, holonomy_tol_rel,
            )

        #
        # Subproceso 1: Inyección de Potencial de Gauge (corregido: invariante de escala)
        #

        def inject_gauge_potential(
            self, d_Phi: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            s_val = dΦ · exp(−½ Tr(G_μν)).

            **Corrección respecto a v5.0.0**: la verificación de colapso ya
            no depende de ‖dΦ‖ (lo cual producía falsos positivos para
            diferenciales pequeños con supresión nula). Se verifica
            directamente el factor de supresión, invariante de escala:

                exp(−½ Tr(G_μν)) < ε_mach  ⟹  colapso estructural genuino

            Retorna
            -------
            Tuple[NDArray[np.float64], float]
                (s_val, suppression_factor).

            Lanza
            -----
            ApexDimensionError
            GaugePotentialError
            """
            n: int = self._ctx.dim
            if d_Phi.shape != (n,):
                raise ApexDimensionError(f"d_Phi debe tener shape ({n},); se obtuvo {d_Phi.shape}.")

            if self._suppression_factor < self._EPS:
                raise GaugePotentialError(
                    f"Estrés estructural extremo: Tr(G_μν)={self._trace_G:.6e}. "
                    f"Factor de supresión={self._suppression_factor:.6e} < ε_mach. "
                    f"Inyección de propuesta de valor completamente colapsada, "
                    f"independientemente de la magnitud de dΦ."
                )

            s_val: NDArray[np.float64] = d_Phi * self._suppression_factor
            logger.debug(
                "[Fase2] Inyección de calibre: suppression=%.6e, ‖s_val‖_∞=%.6e.",
                self._suppression_factor, float(la.norm(s_val, np.inf)),
            )
            return s_val, self._suppression_factor

        #
        # Subproceso 2: Refracción Eikonal de Mercado
        #

        def compute_eikonal_absorption(
            self, phase_gradient: NDArray[np.float64], sigma_stress: float, alpha_fermat: float = 0.5,
        ) -> Tuple[float, float]:
            r"""
            n(σ*)=1+tanh(α·σ*); verifica G^μν∂S∂S ≥ n²(σ*)·(1−eikonal_slack).

            Retorna
            -------
            Tuple[float, float]
                (n_refract, eikonal_norm_sq).

            Lanza
            -----
            ApexDimensionError
            EikonalRefractionError
            """
            n: int = self._ctx.dim
            if phase_gradient.shape != (n,):
                raise ApexDimensionError(f"phase_gradient debe tener shape ({n},); se obtuvo {phase_gradient.shape}.")

            n_refract: float = 1.0 + float(np.tanh(alpha_fermat * sigma_stress))
            G_inv_grad: NDArray[np.float64] = self._ctx.G_inv @ phase_gradient
            eikonal_norm_sq: float = float(np.dot(phase_gradient, G_inv_grad))

            n_sq: float = n_refract ** 2
            eikonal_threshold: float = n_sq * (1.0 - self._eikonal_slack)

            if eikonal_norm_sq < eikonal_threshold:
                raise EikonalRefractionError(
                    f"Fallo Eikonal: ‖∂S‖²_{{G_inv}}={eikonal_norm_sq:.6e} < "
                    f"n²(1-slack)={eikonal_threshold:.6e}. n(σ*)={n_refract:.6f}, σ*={sigma_stress:.6f}. "
                    f"La campaña se dispersó antes de alcanzar el colector."
                )

            logger.debug(
                "[Fase2] Eikonal OK: ‖∂S‖²=%.6e, n²=%.6e, n_refract=%.6f.",
                eikonal_norm_sq, n_sq, n_refract,
            )
            return n_refract, eikonal_norm_sq

        #
        # Subproceso 3: Exergía de Poynting (corregido: no-negatividad por construcción)
        #

        def evaluate_poynting_exergy(
            self, E_field: NDArray[np.float64], H_field: NDArray[np.float64], grad_H: NDArray[np.float64],
        ) -> Tuple[float, float, float]:
            r"""
            P_in = E·H;  P_diss = ‖R_sqrt·∇H‖²  (≥0 por construcción exacta,
            a diferencia de la forma cuadrática directa ∇Hᵀ R ∇H de v5.0.0,
            que podía producir residuos negativos espurios por redondeo);
            P_exergia = P_in − P_diss.

            Retorna
            -------
            Tuple[float, float, float]
                (P_in, P_diss, P_exergia).

            Lanza
            -----
            ApexDimensionError
            FinancialBlackHoleError
            """
            n: int = self._ctx.dim
            for vec, name in [(E_field, "E_field"), (H_field, "H_field"), (grad_H, "grad_H")]:
                if vec.shape != (n,):
                    raise ApexDimensionError(f"'{name}' debe tener shape ({n},); se obtuvo {vec.shape}.")

            P_in: float = float(np.dot(E_field, H_field))

            # P_diss = ‖R_sqrt·∇H‖² = ∇Hᵀ·R_sqrt·R_sqrt·∇H = ∇Hᵀ·R_cost·∇H (exacto),
            # pero garantizado ≥0 al ser una norma euclidiana al cuadrado.
            R_sqrt_grad: NDArray[np.float64] = self._ctx.R_sqrt @ grad_H
            P_diss: float = float(np.dot(R_sqrt_grad, R_sqrt_grad))

            P_exergia: float = P_in - P_diss
            tol_exergy: float = self._EPS * max(abs(P_in), abs(P_diss), 1.0)

            if P_exergia < -tol_exergy:
                raise FinancialBlackHoleError(
                    f"La entropía operativa devora la energía inyectada. P_in={P_in:.6e}, "
                    f"P_diss={P_diss:.6e}, P_exergia={P_exergia:.6e} < -tol={-tol_exergy:.6e}. "
                    f"Veto termodinámico absoluto emitido."
                )

            logger.debug("[Fase2] Poynting: P_in=%.6e, P_diss=%.6e, P_exergia=%.6e.", P_in, P_diss, P_exergia)
            return P_in, P_diss, P_exergia

        #
        # Subproceso 4: Curvatura de Yang-Mills genuina y auditoría de holonomía
        #

        def _compute_curvature(
            self, A_1: NDArray[np.float64], A_2: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            Ensambla la curvatura discreta de una plaqueta con dos
            direcciones de conexión A₁, A₂:

                A_i^{a} := ½(A_i − A_iᵀ) ∈ 𝔰𝔬(n)      (proyección al álgebra de Lie)
                F := (A_2^{a} − A_1^{a}) + [A_1^{a}, A_2^{a}]

            **Corrección fundamental respecto a v5.0.0**: la fórmula
            anterior F=(A−Aᵀ)+[A,Aᵀ] es algebraicamente inválida, porque
            para cualquier matriz real A:

                [A,Aᵀ]ᵀ = (AAᵀ−AᵀA)ᵀ = (AAᵀ)ᵀ−(AᵀA)ᵀ = AAᵀ−AᵀA = [A,Aᵀ]

            es decir, **[A,Aᵀ] es siempre simétrica** (pues AAᵀ y AᵀA son
            ambas simétricas), mientras que (A−Aᵀ) es antisimétrica. Su
            suma no es una 2-forma de curvatura válida, pues el grupo de
            holonomía compatible con una métrica riemanniana G_μν es O(n),
            cuya álgebra de Lie 𝔰𝔬(n) consiste exclusivamente en matrices
            antisimétricas.

            La presente formulación usa **dos generadores genuinos** del
            álgebra 𝔰𝔬(n) (las proyecciones antisimétricas de A₁, A₂), y
            se apoya en la identidad, demostrable directamente: si
            Bᵀ=−B y Cᵀ=−C, entonces

                [B,C]ᵀ = (BC−CB)ᵀ = CᵀBᵀ−BᵀCᵀ = CB−BC = −[B,C]

            es decir, **𝔰𝔬(n) es cerrado bajo el corchete de Lie**. Por lo
            tanto F, suma de matrices antisimétricas y un corchete de
            matrices antisimétricas, es antisimétrica por construcción.

            Retorna
            -------
            NDArray[np.float64], shape (n, n)
                F, antisimétrica hasta precisión de máquina.
            """
            A1_antisym: NDArray[np.float64] = 0.5 * (A_1 - A_1.T)
            A2_antisym: NDArray[np.float64] = 0.5 * (A_2 - A_2.T)
            commutator: NDArray[np.float64] = (
                A1_antisym @ A2_antisym - A2_antisym @ A1_antisym
            )
            F: NDArray[np.float64] = (A2_antisym - A1_antisym) + commutator
            return F

        def audit_yang_mills_holonomy(
            self, A_gauge_1: NDArray[np.float64], A_gauge_2: NDArray[np.float64],
        ) -> Tuple[float, float]:
            r"""
            Audita la curvatura de calibre de una plaqueta de dos
            direcciones (A_gauge_1, A_gauge_2) mediante la acción de
            Yang-Mills ponderada por G_μν:

                S_YM = ½ Tr(Fᵀ · G_μν · F · G^μν) ≥ 0

            No-negatividad exacta: para cualquier F real y G_μν SPD,
            FᵀG_μνF es PSD (∀x: xᵀFᵀG_μνFx=(Fx)ᵀG_μν(Fx)≥0), y el producto
            de traza de dos matrices PSD simétricas es no-negativo.

            Se verifica adicionalmente el invariante algebraico exacto
            F=−Fᵀ (garantizado por construcción en ``_compute_curvature``;
            un residuo significativo indicaría un error de implementación,
            no un fenómeno físico).

            Parámetros
            ----------
            A_gauge_1, A_gauge_2 : NDArray[np.float64], shape (n, n)
                Potenciales de calibre en las dos direcciones de la plaqueta.

            Retorna
            -------
            Tuple[float, float]
                (S_ym, antisymmetry_residual_relative).

            Lanza
            -----
            ApexDimensionError
            HolonomyVetoError
            ElectrodynamicApexError
                Si la antisimetría de F se viola más allá de 1e-8 relativo
                (canario de error de implementación).
            """
            n: int = self._ctx.dim
            for mat, name in [(A_gauge_1, "A_gauge_1"), (A_gauge_2, "A_gauge_2")]:
                if mat.shape != (n, n):
                    raise ApexDimensionError(f"'{name}' debe tener shape ({n},{n}); se obtuvo {mat.shape}.")

            F: NDArray[np.float64] = self._compute_curvature(A_gauge_1, A_gauge_2)

            norm_F: float = float(la.norm(F, "fro"))
            antisym_residual: float = float(la.norm(F + F.T, "fro")) / max(norm_F, 1.0)
            if antisym_residual > 1.0e-8:
                raise ElectrodynamicApexError(
                    f"Curvatura F no es antisimétrica (violación de invariante algebraico exacto): "
                    f"‖F+Fᵀ‖_F/‖F‖_F={antisym_residual:.6e} > 1e-8. Revise _compute_curvature."
                )

            G_F: NDArray[np.float64] = self._ctx.G_mu_nu @ F
            F_G_inv: NDArray[np.float64] = F @ self._ctx.G_inv
            S_ym: float = max(0.5 * float(np.trace(F.T @ G_F @ F_G_inv)), 0.0)

            norm_A_sq: float = float(la.norm(A_gauge_1, "fro") ** 2 + la.norm(A_gauge_2, "fro") ** 2)
            tol_ym: float = self._holonomy_tol_rel * max(norm_A_sq, 1.0)

            if S_ym > tol_ym:
                raise HolonomyVetoError(
                    f"El bucle de Wilson revela curvatura de calibre no nula. "
                    f"S_YM={S_ym:.6e} > tol_rel={tol_ym:.6e}. Fugas logísticas ocultas detectadas."
                )

            logger.debug(
                "[Fase2] Holonomía: S_YM=%.6e, tol=%.6e, antisym_residual=%.3e.",
                S_ym, tol_ym, antisym_residual,
            )
            return S_ym, antisym_residual

        def verify_gauge_covariance(
            self,
            A_gauge_1: NDArray[np.float64],
            A_gauge_2: NDArray[np.float64],
            Q_isometry: NDArray[np.float64],
            isometry_tol: float = 1.0e-8,
        ) -> float:
            r"""
            **Diagnóstico opcional** (no forma parte del camino caliente):
            verifica la propiedad definitoria de toda teoría de calibre —
            invarianza de la acción de Yang-Mills bajo transformaciones de
            calibre A_i ↦ Q·A_i·Qᵀ, para toda isometría Q de G_μν
            (QᵀG_μνQ = G_μν, es decir Q pertenece al grupo de holonomía
            O_G(n) compatible con la métrica).

            Se verifica primero que Q es una isometría genuina; de lo
            contrario, el teorema de invarianza de calibre no aplica y se
            lanza ``GaugeCovarianceError`` con diagnóstico explícito.

            Parámetros
            ----------
            A_gauge_1, A_gauge_2 : NDArray[np.float64], shape (n, n)
                Potenciales de calibre originales.
            Q_isometry : NDArray[np.float64], shape (n, n)
                Transformación de calibre candidata.
            isometry_tol : float, default 1e-8
                Tolerancia relativa para validar QᵀG_μνQ ≈ G_μν.

            Retorna
            -------
            float
                Residuo relativo |S_YM(A) − S_YM(QAQᵀ)| / max(S_YM(A), 1).

            Lanza
            -----
            ApexDimensionError
            GaugeCovarianceError
                Si Q no es isometría de G_μν, o si la invarianza de S_YM
                se viola (bug de implementación de la curvatura).
            """
            n: int = self._ctx.dim
            if Q_isometry.shape != (n, n):
                raise ApexDimensionError(f"Q_isometry debe tener shape ({n},{n}); se obtuvo {Q_isometry.shape}.")

            isometry_residual: float = float(
                la.norm(Q_isometry.T @ self._ctx.G_mu_nu @ Q_isometry - self._ctx.G_mu_nu, "fro")
            ) / max(float(la.norm(self._ctx.G_mu_nu, "fro")), 1.0)

            if isometry_residual > isometry_tol:
                raise GaugeCovarianceError(
                    f"Q_isometry no preserva G_μν: ‖QᵀGQ-G‖_F/‖G‖_F={isometry_residual:.6e} "
                    f"> tol={isometry_tol:.6e}. La invarianza de calibre no es aplicable."
                )

            S_ym_original, _ = self.audit_yang_mills_holonomy(A_gauge_1, A_gauge_2)
            A_1_transformed: NDArray[np.float64] = Q_isometry @ A_gauge_1 @ Q_isometry.T
            A_2_transformed: NDArray[np.float64] = Q_isometry @ A_gauge_2 @ Q_isometry.T
            S_ym_transformed, _ = self.audit_yang_mills_holonomy(A_1_transformed, A_2_transformed)

            covariance_residual: float = abs(S_ym_original - S_ym_transformed) / max(S_ym_original, 1.0)

            if covariance_residual > 1.0e-6:
                raise GaugeCovarianceError(
                    f"S_YM no es invariante bajo la transformación de calibre: "
                    f"S_YM(A)={S_ym_original:.6e}, S_YM(QAQᵀ)={S_ym_transformed:.6e}, "
                    f"residuo relativo={covariance_residual:.6e} > 1e-6. "
                    f"Error de implementación en la curvatura de calibre."
                )

            logger.debug("[Fase2] Covarianza de calibre verificada: residuo=%.3e.", covariance_residual)
            return covariance_residual

        #
        # Retícula Booleana de viabilidad
        #

        def _evaluate_viability_flags(
            self,
            suppression_factor: float,
            eikonal_norm_sq: float,
            n_sq: float,
            P_exergia: float,
            S_ym: float,
            tol_ym: float,
            antisym_residual: float,
        ) -> ApexViabilityFlags:
            r"""
            Combina predicados independientes mediante disyunción bit a
            bit, formando el elemento de la retícula Booleana correspondiente.
            """
            flags: ApexViabilityFlags = ApexViabilityFlags.NONE
            soft_slack: float = self._eikonal_slack / 2.0

            if suppression_factor >= np.sqrt(self._EPS):
                flags |= ApexViabilityFlags.GAUGE_INJECTION_NONTRIVIAL
            if eikonal_norm_sq >= n_sq * (1.0 - soft_slack):
                flags |= ApexViabilityFlags.EIKONAL_MARGIN_SOUND
            if P_exergia >= -1.0e3 * self._EPS:
                flags |= ApexViabilityFlags.EXERGY_NONNEGATIVE
            if S_ym <= tol_ym:
                flags |= ApexViabilityFlags.HOLONOMY_TRIVIAL
            if antisym_residual <= 1.0e-8:
                flags |= ApexViabilityFlags.CURVATURE_ANTISYMMETRIC
            if self._ctx.kappa_G <= 0.5 * self._kappa_max:
                flags |= ApexViabilityFlags.METRIC_WELL_CONDITIONED

            logger.debug("[Fase2] %s", describe_viability_flags(flags))
            return flags

        #
        # Método terminal de la Fase 2 — entrada directa de la Fase 3
        #

        def synthesize(
            self,
            d_Phi: NDArray[np.float64],
            phase_gradient: NDArray[np.float64],
            sigma_stress: float,
            E_field: NDArray[np.float64],
            H_field: NDArray[np.float64],
            grad_H: NDArray[np.float64],
            A_gauge_1: NDArray[np.float64],
            A_gauge_2: NDArray[np.float64],
            alpha_fermat: float = 0.5,
            Q_isometry_diagnostic: Optional[NDArray[np.float64]] = None,
        ) -> "ApexStateTensor":
            r"""
            **Método terminal de la Fase 2.**

            Integra los cuatro subprocesos electrodinámicos y retorna el
            ``ApexStateTensor`` completo. El campo ``gauge_injection_vector``
            es el dato primario consumido por la Fase 3.

            Parámetros
            ----------
            A_gauge_1, A_gauge_2 : NDArray[np.float64], shape (n, n)
                Potenciales de calibre en las dos direcciones de la plaqueta
                (reemplaza el ``A_gauge`` único de v5.0.0, matemáticamente
                insuficiente para producir curvatura no-abeliana genuina).
            Q_isometry_diagnostic : Optional[NDArray[np.float64]], default None
                Si se provee, ejecuta ``verify_gauge_covariance`` como
                diagnóstico adicional (coste extra de una segunda auditoría
                de holonomía; opt-in para no penalizar el camino caliente).

            Retorna
            -------
            ApexStateTensor
            """
            s_val, suppression = self.inject_gauge_potential(d_Phi)
            n_refract, eikonal_norm_sq = self.compute_eikonal_absorption(phase_gradient, sigma_stress, alpha_fermat)
            P_in, P_diss, P_exergia = self.evaluate_poynting_exergy(E_field, H_field, grad_H)
            S_ym, antisym_residual = self.audit_yang_mills_holonomy(A_gauge_1, A_gauge_2)

            norm_A_sq: float = float(la.norm(A_gauge_1, "fro") ** 2 + la.norm(A_gauge_2, "fro") ** 2)
            tol_ym: float = self._holonomy_tol_rel * max(norm_A_sq, 1.0)

            flags: ApexViabilityFlags = self._evaluate_viability_flags(
                suppression_factor=suppression,
                eikonal_norm_sq=eikonal_norm_sq,
                n_sq=n_refract ** 2,
                P_exergia=P_exergia,
                S_ym=S_ym,
                tol_ym=tol_ym,
                antisym_residual=antisym_residual,
            )

            gauge_covariance_residual: Optional[float] = None
            if Q_isometry_diagnostic is not None:
                gauge_covariance_residual = self.verify_gauge_covariance(
                    A_gauge_1, A_gauge_2, Q_isometry_diagnostic
                )

            logger.info(
                "[Fase2] Síntesis completada: suppression=%.3e, n_refract=%.4f, "
                "P_exergia=%.6e, S_YM=%.6e, %s.",
                suppression, n_refract, P_exergia, S_ym, describe_viability_flags(flags),
            )

            # Contrato de interfaz Fase 2 → Fase 3: gauge_injection_vector=s_val
            return ApexStateTensor(
                gauge_injection_vector=s_val,
                suppression_factor=suppression,
                fermat_refractive_index=n_refract,
                eikonal_norm_sq=eikonal_norm_sq,
                poynting_income=P_in,
                poynting_dissipation=P_diss,
                poynting_exergy_flux=P_exergia,
                yang_mills_action=S_ym,
                curvature_antisymmetry_residual=antisym_residual,
                viability_flags=flags,
                is_electrodynamically_viable=(flags == ApexViabilityFlags.ALL),
                gauge_covariance_residual=gauge_covariance_residual,
            )

    #
    # ==========================================================================
    # FASE 3 — PROYECCIÓN EN HACES: FIBRA MÉTRICA + FIBRA DISIPATIVA
    # ==========================================================================
    #

    class Phase3_SheafProjection:
        r"""
        **Fase 3 – Proyección en Haces y Cofrontera Discreta δ_{APEX}.**

        Recibe el ``ApexPreparationContext`` de la Fase 1 y el vector de
        inyección s_val producido por la Fase 2.

        Corrección respecto a v5.0.0
        ------------------------------
        R_sqrt se calculaba en Fase 1 pero **nunca se usaba** en ninguna
        fase posterior (código muerto). Se activa aquí una fibra
        disipativa genuina:

            δ_metric = G_μν^{-1/2} = L_G^{-⊤}         ∈ ℝ^{n×n}  (invertible)
            δ_diss   = R_sqrt · δ_metric                ∈ ℝ^{n×n}
            δ_APEX   = [ δ_metric ; δ_diss ]            ∈ ℝ^{2n×n}
            Δ_APEX   = δ_APEXᵀ δ_APEX = I_n + δ_dissᵀδ_diss   (SPD)

        en paridad estructural directa con el Laplaciano de Hodge local
        ∇²H+R_cost de ``KBaseThermodynamicAgent.Phase3_SheafProjection``.
        """

        _EPS: float = _MACHINE_EPS

        def __init__(self, context: "ApexPreparationContext") -> None:
            r"""**Constructor de la Fase 3: continuación directa de la Fase 2.**"""
            self._ctx: "ApexPreparationContext" = context
            n: int = context.dim

            I_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)
            L_G_inv: NDArray[np.float64] = la.solve_triangular(
                context.L_G, I_n, lower=True, check_finite=False
            )
            self._delta_metric: NDArray[np.float64] = L_G_inv.T

            # Fibra disipativa activada (antes código muerto vía R_sqrt sin usar)
            self._delta_dissipative: NDArray[np.float64] = context.R_sqrt @ self._delta_metric

            self._delta_apex: NDArray[np.float64] = np.vstack(
                [self._delta_metric, self._delta_dissipative]
            )  # shape (2n, n)

            self._hodge_laplacian: NDArray[np.float64] = (
                I_n + self._delta_dissipative.T @ self._delta_dissipative
            )
            self._hodge_laplacian = 0.5 * (self._hodge_laplacian + self._hodge_laplacian.T)

            self._rank_delta: int = n

            self._hodge_metric_residual: float = self._verify_hodge_identity()

            logger.debug(
                "[Fase3] Precalculado: δ_metric shape=%s, δ_diss shape=%s, "
                "δ_APEX shape=%s, rank_delta=%d, Hodge_res=%.3e.",
                self._delta_metric.shape, self._delta_dissipative.shape,
                self._delta_apex.shape, self._rank_delta, self._hodge_metric_residual,
            )

        def _verify_hodge_identity(self) -> float:
            r"""
            Verifica δ_metricᵀ G_μν δ_metric ≈ I (identidad de Hodge local
            de la fibra métrica pura).

            Retorna
            -------
            float
                Residuo relativo ‖δ_metricᵀG_μνδ_metric − I‖_F / n.

            Lanza
            -----
            SheafMetricError
            """
            n: int = self._ctx.dim
            delta_T_G_delta: NDArray[np.float64] = (
                self._delta_metric.T @ self._ctx.G_mu_nu @ self._delta_metric
            )
            I_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)
            residual_F: float = float(la.norm(delta_T_G_delta - I_n, "fro"))
            rel_error: float = residual_F / n
            tol_hodge: float = 100.0 * self._EPS

            if rel_error > tol_hodge:
                raise SheafMetricError(
                    f"Identidad de Hodge violada: ‖δᵀGδ-I‖_F/n={rel_error:.6e} > tol={tol_hodge:.6e}."
                )
            return rel_error

        def _compute_hodge_spectrum(self) -> Tuple[float, float]:
            r"""
            Diagonaliza Δ_APEX (SPD, n×n) para extraer brecha espectral y
            número de condición.

            Retorna
            -------
            Tuple[float, float]
                (spectral_gap, condition_number).
            """
            eigvals: NDArray[np.float64] = la.eigvalsh(self._hodge_laplacian)
            lambda_min: float = float(eigvals[0])
            lambda_second: float = float(eigvals[1]) if len(eigvals) > 1 else lambda_min
            lambda_max: float = float(eigvals[-1])
            spectral_gap: float = lambda_second - lambda_min
            condition_number: float = lambda_max / lambda_min if lambda_min > 0 else float("inf")
            logger.debug("[Fase3] Espectro de Hodge: gap=%.6e, κ=%.6e.", spectral_gap, condition_number)
            return spectral_gap, condition_number

        #
        # Método terminal de la Fase 3 (salida pública del ecosistema)
        #

        def export_stalk(self, s_val: NDArray[np.float64]) -> "SheafStalkApex":
            r"""
            **Método terminal de la Fase 3 y del agente completo.**

            Proyecta s_val sobre ambas fibras (métrica y disipativa) y
            retorna el ``SheafStalkApex`` completo con diagnósticos
            espectrales del Laplaciano de Hodge local.

            Parámetros
            ----------
            s_val : NDArray[np.float64], shape (n,)
                Típicamente ``ApexStateTensor.gauge_injection_vector``.

            Retorna
            -------
            SheafStalkApex

            Lanza
            -----
            ApexDimensionError
            """
            n: int = self._ctx.dim
            if s_val.shape != (n,):
                raise ApexDimensionError(f"s_val debe tener shape ({n},); se obtuvo {s_val.shape}.")

            spectral_gap, condition_number = self._compute_hodge_spectrum()

            projected_metric: NDArray[np.float64] = self._delta_metric @ s_val
            projected_diss: NDArray[np.float64] = self._delta_dissipative @ s_val

            logger.info(
                "[Fase3] SheafStalkApex exportado: dim=%d, rank=%d, Hodge_res=%.3e, "
                "gap=%.3e, κ(Δ)=%.3e, ‖proj_metric‖=%.6e, ‖proj_diss‖=%.6e.",
                n, self._rank_delta, self._hodge_metric_residual, spectral_gap, condition_number,
                float(la.norm(projected_metric, 2)), float(la.norm(projected_diss, 2)),
            )

            return SheafStalkApex(
                delta_apex=self._delta_apex.copy(),
                delta_metric=self._delta_metric.copy(),
                delta_dissipative=self._delta_dissipative.copy(),
                hodge_laplacian=self._hodge_laplacian.copy(),
                hodge_metric_residual=self._hodge_metric_residual,
                hodge_spectral_gap=spectral_gap,
                hodge_condition_number=condition_number,
                lossless_subspace_dimension=self._ctx.betti_0_R,
                source_injection=s_val.copy(),
                projected_source_metric=projected_metric,
                projected_source_dissipative=projected_diss,
                rank_delta=self._rank_delta,
            )

    #
    # ==========================================================================
    # INTERFAZ PÚBLICA DEL AGENTE (punto de entrada externo)
    # ==========================================================================
    #

    def synthesize_apex_field(
        self,
        d_Phi: NDArray[np.float64],
        phase_gradient: NDArray[np.float64],
        sigma_stress: float,
        E_field: NDArray[np.float64],
        H_field: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        A_gauge_1: NDArray[np.float64],
        A_gauge_2: NDArray[np.float64],
        alpha_fermat: float = 0.5,
        Q_isometry_diagnostic: Optional[NDArray[np.float64]] = None,
    ) -> ApexStateTensor:
        r"""
        Punto de entrada público para la síntesis electrodinámica completa.

        **Cambio de firma respecto a v5.0.0**: ``A_gauge`` (único) se
        reemplaza por ``A_gauge_1, A_gauge_2`` (plaqueta de dos
        direcciones), requerido para que la curvatura de Yang-Mills sea
        matemáticamente válida (ver ``_compute_curvature``).

        Retorna
        -------
        ApexStateTensor
        """
        return self.phase2.synthesize(
            d_Phi=d_Phi,
            phase_gradient=phase_gradient,
            sigma_stress=sigma_stress,
            E_field=E_field,
            H_field=H_field,
            grad_H=grad_H,
            A_gauge_1=A_gauge_1,
            A_gauge_2=A_gauge_2,
            alpha_fermat=alpha_fermat,
            Q_isometry_diagnostic=Q_isometry_diagnostic,
        )

    def export_sheaf_stalk(self, s_val: NDArray[np.float64]) -> SheafStalkApex:
        r"""
        Exporta el Stalk del haz electrodinámico (fibra métrica + disipativa).

        Instancia la Fase 3 perezosamente en la primera llamada.

        Retorna
        -------
        SheafStalkApex
        """
        if self.phase3 is None:
            self.phase3 = KApexElectrodynamicAgent.Phase3_SheafProjection(context=self.context)
            logger.info(
                "[KApexElectrodynamicAgent] Phase3_SheafProjection instanciada (lazy init). "
                "rank_delta=%d, Hodge_residual=%.3e.",
                self.phase3._rank_delta, self.phase3._hodge_metric_residual,
            )
        return self.phase3.export_stalk(s_val=s_val)
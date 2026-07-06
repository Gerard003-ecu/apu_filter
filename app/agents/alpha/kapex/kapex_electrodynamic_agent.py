# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo : KApex Electrodynamic Agent (Director de Retorno y Expansión)        |
| Ruta   : app/agents/alpha/kapex/kapex_electrodynamic_agent.py                |
| Versión: 5.0.0-Rigorous-Gauge-Holonomy-Eikonal                               |
+==============================================================================+

NATURALEZA CIBER-FÍSICA Y ÓPTICA GEOMÉTRICA:
Este módulo consagra el Ápice Estratégico como un Endofuntor de Campo de Calibre que
inyecta Fuerza Electromotriz (FEM), resuelve la refracción y audita el retorno.

ECUACIÓN EIKONAL DE ABSORCIÓN:
\[ G^{\mu\nu} \partial_\mu S \partial_\nu S = N^{\mu\nu} \sigma_{\mu\nu}^* \]

FLUJO EXERGÉTICO DE POYNTING:
\[ P_{exergia} = \langle E \smile \star H, [\partial K] \rangle - \int_K \nabla H^\top R_{cost} \nabla H \ge 0 \]

CURVATURA DE YANG-MILLS:
\[ S_{YM} = \frac{1}{2} \int_M Tr(F \wedge \star F) \quad \text{donde} \quad F = dA + A \wedge A \]
"""
from __future__ import annotations

#    Biblioteca est ndar
import logging
from typing import Optional, Tuple

#     lgebra num rica de alta precisi n
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

#    Estructuras de datos inmutables
from dataclasses import dataclass

#    Dependencias arquitect nicas del ecosistema APU Filter
try:
    from app.core.mic_algebra import CategoricalState, Morphism
except ImportError:
    class CategoricalState:  # type: ignore[no-redef]
        """Stub: estado categ rico del ecosistema MIC."""

    class Morphism:  # type: ignore[no-redef]
        """Stub: morfismo funtorial del ecosistema MIC."""


#    Logger del m dulo
logger = logging.getLogger("MIC.Alpha.KApexElectrodynamicAgent")


#
#    SECCI N 0   EXCEPCIONES ELECTRODIN MICAS ESTRICTAS
#


class ElectrodynamicApexError(Exception):
    """
    Excepci n categ rica ra z para violaciones en el Estrato K_APEX.

    Toda excepci n de este m dulo hereda de esta clase, permitiendo que
    los manejadores de nivel superior capturen cualquier fallo electrodin mico
    con un  nico ``except ElectrodynamicApexError``.
    """


class ApexDimensionError(ElectrodynamicApexError):
    """
    Lanzada cuando las dimensiones de las matrices constitutivas son
    inconsistentes entre s  o con los vectores de campo recibidos.

    Diagn stico incluye las formas detectadas y las esperadas.
    """


class ApexSymmetryError(ElectrodynamicApexError):
    """
    Lanzada cuando G_   o R_cost violan su propiedad de simetr a,
    con diagn stico cuantitativo normalizado al Frobenius.
    """


class ApexConditionError(ElectrodynamicApexError):
    """
    Lanzada cuando el n mero de condici n espectral  (G_  ) supera
    el umbral admisible, comprometiendo la coherencia de la variedad.
    """


class MetricInverseError(ElectrodynamicApexError):
    """
    Lanzada cuando G_inv no es la inversa de G_   dentro de la tolerancia
    de m quina relativa:  G G_inv - I _F / n > tol.

    Incluye el residuo cuantitativo para diagn stico.
    """


class GaugePotentialError(ElectrodynamicApexError):
    """
    Lanzada cuando el estr s estructural Tr(G_  ) colapsa la inyecci n
    de valor suprimi ndola por debajo de la resoluci n num rica.

    Condici n de disparo:
         s_val _  /  d  _  <  _mach  (supresi n total relativa)
    """


class EikonalRefractionError(ElectrodynamicApexError):
    """
    Lanzada cuando el mercado objetivo es topol gicamente inalcanzable:
    la norma riemanniana del gradiente de fase no satisface la ecuaci n
    Eikonal con el  ndice de refracci n calculado.

    Condici n de disparo:
        G^    _ S  _ S < n ( *)   (1 -  _eikonal)
    """


class FinancialBlackHoleError(ElectrodynamicApexError):
    """
    Lanzada cuando el flujo de Poynting (ingresos) no compensa la
    disipaci n termodin mica (costes):

        P_exergia = E H -  H^  R_cost  H < -tol_poynting

    La tolerancia es relativa a max(|P_in|, |P_diss|, 1) para evitar
    disparos espurios por errores de redondeo.
    """


class HolonomyVetoError(ElectrodynamicApexError):
    """
    Lanzada cuando la acci n de Yang-Mills discreta:

        S_YM =   Tr(F^  G_   F G^  )

    supera el umbral relativo a  A_gauge _F , revelando curvatura de
    calibre no nula (ciclos par sitos en la log stica estrat gica).
    """


class SheafMetricError(ElectrodynamicApexError):
    """
    Lanzada cuando la cofrontera  _{APEX} no satisface la identidad
    de la m trica de Hodge local:

         _{APEX}^  G_    _{APEX} ~= I

    dentro de la tolerancia de m quina.
    """


#
#    SECCI N 1   ESTRUCTURAS INMUTABLES (DTOs TENSORIALES)
#


@dataclass(frozen=True, slots=True)
class ApexPreparationContext:
    r"""
    Contexto inmutable producido por la **Fase 1** (Validación Métrica).

    Contiene las matrices constitutivas validadas y sus factorizaciones
    precomputadas para uso eficiente en Fases 2 y 3.

    Atributos
    ----------
    G_mu_nu : NDArray[np.float64], shape (n, n)
        Tensor métrico riemanniano validado (SPD, simétrico).

    G_inv : NDArray[np.float64], shape (n, n)
        Tensor métrico inverso G^μν, verificado como G·G_inv ≈ I.

    R_cost : NDArray[np.float64], shape (n, n)
        Matriz de disipación termodinámica (PSD, simétrica).

    L_G : NDArray[np.float64], shape (n, n)
        Factor Cholesky inferior de G_μν: G_μν = L_G · L_G^⊤.
        Precalculado en Fase 1 para uso eficiente en Fases 2 y 3.

    R_sqrt : NDArray[np.float64], shape (n, n)
        Raíz cuadrada espectral de R_cost: R_sqrt^⊤·R_sqrt = R_cost.
        Precalculada en Fase 1 para la cofrontera de la Fase 3.

    kappa_G : float
        Número de condición espectral de G_μν: κ = λ_max/λ_min.

    rank_R : int
        Rango numérico de R_cost (puede ser < n si PSD rango-deficiente).

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
    rank_R: int
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
        Vector de inyección de FEM: s_val = dΦ · exp(−½ Tr(G_μν)).

    suppression_factor : float
        Factor de supresión de Gauge: exp(−½ Tr(G_μν)) ∈ (0, 1].
        Documentado para diagnóstico de estrés estructural.

    fermat_refractive_index : float
        Índice de refracción efectivo n(σ*) = 1 + tanh(α · σ*).

    eikonal_norm_sq : float
        G^μν ∂_μS ∂_νS: norma riemanniana cuadrada del gradiente de fase.
        Debe ser ≥ n²(σ*) · (1 − ε_eikonal).

    poynting_income : float
        P_in = E · H: potencia de entrada (ingresos brutos).

    poynting_dissipation : float
        P_diss = ∇H^⊤ R_cost ∇H: potencia disipada (costes operativos).

    poynting_exergy_flux : float
        P_exergia = P_in − P_diss ≥ 0: flujo exergético neto.

    yang_mills_action : float
        S_YM = ½ Tr(F^⊤ G_μν F G^μν): acción de Yang-Mills ponderada.

    is_electrodynamically_viable : bool
        True si y sólo si todos los subprocesos pasan sin excepción.
    """

    gauge_injection_vector: NDArray[np.float64]
    suppression_factor: float
    fermat_refractive_index: float
    eikonal_norm_sq: float
    poynting_income: float
    poynting_dissipation: float
    poynting_exergy_flux: float
    yang_mills_action: float
    is_electrodynamically_viable: bool


@dataclass(frozen=True, slots=True)
class SheafStalkApex:
    r"""
    Fibrado celular exportado para el cálculo global del Laplaciano de Haz.

    Producido por la **Fase 3** (Proyección en Haces).

    Atributos
    ----------
    delta_apex : NDArray[np.float64], shape (n, n)
        Cofrontera discreta δ_{APEX} = G_μν^{-1/2} = L_G^{-⊤}.
        Satisface la identidad de la métrica de Hodge local:
        δ_{APEX}^⊤ · G_μν · δ_{APEX} = I.

    hodge_metric_residual : float
        ‖δ_{APEX}^⊤ G_μν δ_{APEX} − I‖_F / n:
        Error relativo de la identidad de Hodge. Debe ser O(ε_mach).

    source_injection : NDArray[np.float64], shape (n,)
        Vector de inyección s_val del estado electrodinámico actual.

    projected_source : NDArray[np.float64], shape (n,)
        Proyección δ_{APEX} · s_val sobre la fibra local.

    rank_delta : int
        Rango numérico de δ_{APEX} = n (pleno rango, pues G_μν ≻ 0).
    """

    delta_apex: NDArray[np.float64]
    hodge_metric_residual: float
    source_injection: NDArray[np.float64]
    projected_source: NDArray[np.float64]
    rank_delta: int


#
#    SECCI N 2   ORQUESTADOR: KApexElectrodynamicAgent
#                Tres fases anidadas de rigor creciente
#


class KApexElectrodynamicAgent(Morphism):
    r"""
    Orquestador Funtorial del Ápice Estratégico K_{APEX}.

    Garantiza la invarianza de Gauge y audita el bucle de holonomía global
    mediante tres clases anidadas que operan en cascada estricta:

        Phase1_MetricValidation
            ↓  ApexPreparationContext
        Phase2_ElectrodynamicSynthesis
            ↓  ApexStateTensor
        Phase3_SheafProjection
            ↓  SheafStalkApex

    El constructor instancia y ejecuta la Fase 1 de forma inmediata,
    garantizando que cualquier inconsistencia métrica o de calibre sea
    detectada antes de que el agente opere en el ecosistema APU Filter.

    Parámetros de Construcción
    --------------------------
    G_mu_nu : NDArray[np.float64], shape (n, n)
        Tensor métrico riemanniano (SPD). Semántica: estructura interna
        del espacio de propuestas de valor.

    G_inv : NDArray[np.float64], shape (n, n)
        Tensor métrico inverso G^μν (SPD). Debe satisfacer G·G_inv ≈ I.

    R_cost : NDArray[np.float64], shape (n, n)
        Matriz de disipación termodinámica (PSD). Semántica: tasa de
        consumo de exergía por costes operativos estratégicos.

    kappa_max : float, default 1e10
        Umbral de número de condición espectral κ(G_μν).

    eikonal_slack : float, default 0.1
        Tolerancia relativa de la ecuación Eikonal:
        se acepta G^μν ∂S ∂S ≥ n²(σ*) · (1 − eikonal_slack).

    holonomy_tol_rel : float, default 1e-6
        Tolerancia relativa para la acción de Yang-Mills:
        S_YM / max(‖A_gauge‖_F², 1) < holonomy_tol_rel.
    """

    FRIENDLY_NAME: str = "Director de Retorno y Expansi n de Mercado"

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
        ApexDimensionError, ApexSymmetryError, ApexConditionError,
        MetricInverseError
            Propagadas desde la Fase 1 si alguna propiedad es violada.
        """
        self.kappa_max: float = kappa_max
        self.eikonal_slack: float = eikonal_slack
        self.holonomy_tol_rel: float = holonomy_tol_rel

        #    Fase 1: Validaci n M trica y de Calibre (inmediata)
        self.phase1: KApexElectrodynamicAgent.Phase1_MetricValidation = (
            KApexElectrodynamicAgent.Phase1_MetricValidation(
                G_mu_nu=G_mu_nu,
                G_inv=G_inv,
                R_cost=R_cost,
                kappa_max=kappa_max,
            )
        )
        self.context: ApexPreparationContext = self.phase1.build_context()

        #    Fase 2: S ntesis Electrodin mica (instanciaci n inmediata)
        self.phase2: KApexElectrodynamicAgent.Phase2_ElectrodynamicSynthesis = (
            KApexElectrodynamicAgent.Phase2_ElectrodynamicSynthesis(
                context=self.context,
                eikonal_slack=self.eikonal_slack,
                holonomy_tol_rel=self.holonomy_tol_rel,
            )
        )

        #    Fase 3: instanciaci n perezosa
        self.phase3: Optional[
            KApexElectrodynamicAgent.Phase3_SheafProjection
        ] = None

        logger.info(
            "[KApexElectrodynamicAgent] Inicializado: dim=%d, "
            " (G)=%.3e, rank(R)=%d, inv_residual=%.3e.",
            self.context.dim,
            self.context.kappa_G,
            self.context.rank_R,
            self.context.inverse_residual,
        )

    #
    # FASE 1   VALIDACI N M TRICA Y DE CALIBRE
    #

    class Phase1_MetricValidation:
        r"""
        **Fase 1 – Validación Métrica y de Calibre.**

        Responsabilidades exclusivas de esta fase:
          a) Verificar dimensiones y cuadratura de G_μν, G_inv, R_cost.
          b) Verificar simetría de G_μν y R_cost con tolerancia relativa.
          c) Verificar SPD de G_μν y G_inv con factorización de Cholesky.
          d) Calcular κ(G_μν) y rechazar si κ > kappa_max.
          e) Verificar consistencia métrica G·G_inv ≈ I con tolerancia relativa.
          f) Verificar PSD de R_cost con tolerancia normalizada.
          g) Precalcular L_G (Cholesky de G_μν) y R_sqrt (raíz espectral de R_cost).
          h) Retornar ``ApexPreparationContext`` inmutable.

        Todas las tolerancias son relativas a la norma de Frobenius de la
        matriz evaluada multiplicada por ε_mach, eliminando falsos positivos.
        """

        _EPS: float = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            G_mu_nu: NDArray[np.float64],
            G_inv: NDArray[np.float64],
            R_cost: NDArray[np.float64],
            kappa_max: float = 1.0e10,
        ) -> None:
            r"""
            Almacena referencias sin copiar. Las copias ocurren sólo en
            ``build_context`` al empaquetar el ``ApexPreparationContext``.
            """
            self._G: NDArray[np.float64] = G_mu_nu
            self._G_inv: NDArray[np.float64] = G_inv
            self._R: NDArray[np.float64] = R_cost
            self._kappa_max: float = kappa_max

        #
        # M todos privados de validaci n (orden l gico de ejecuci n)
        #

        def _check_dimensions(self) -> int:
            r"""
            Verifica que G_μν, G_inv y R_cost son cuadradas y de la misma
            dimensión n, y que todos los arrays son 2D.

            Retorna
            -------
            int
                Dimensión n del espacio de calibre.

            Lanza
            -----
            ApexDimensionError
                Con diagnóstico explícito de la inconsistencia detectada.
            """
            for mat, name in [
                (self._G, "G_mu_nu"),
                (self._G_inv, "G_inv"),
                (self._R, "R_cost"),
            ]:
                if mat.ndim != 2:
                    raise ApexDimensionError(
                        f"'{name}' debe ser un array 2D; "
                        f"se obtuvo ndim={mat.ndim}, shape={mat.shape}."
                    )
                if mat.shape[0] != mat.shape[1]:
                    raise ApexDimensionError(
                        f"'{name}' debe ser cuadrada; "
                        f"se obtuvo shape={mat.shape}."
                    )

            n: int = self._G.shape[0]

            if self._G_inv.shape[0] != n:
                raise ApexDimensionError(
                    f"G_inv debe tener shape ({n},{n}) coherente con G_  ; "
                    f"se obtuvo {self._G_inv.shape}."
                )

            if self._R.shape[0] != n:
                raise ApexDimensionError(
                    f"R_cost debe tener shape ({n},{n}) coherente con G_  ; "
                    f"se obtuvo {self._R.shape}."
                )

            logger.debug("[Fase1] Dimensiones verificadas: n=%d.", n)
            return n

        def _validate_symmetry(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> None:
            r"""
            Verifica A = A^⊤ con tolerancia relativa al Frobenius:

                tol = ε_mach · ‖A‖_F

            Lanza
            -----
            ApexSymmetryError
                Con residuo absoluto y relativo ‖A − A^⊤‖_F / ‖A‖_F.
            """
            norm_A: float = float(la.norm(A, "fro"))
            tol: float = self._EPS * max(norm_A, 1.0)
            residual: float = float(la.norm(A - A.T, "fro"))

            if residual > tol:
                raise ApexSymmetryError(
                    f"La matriz '{name}' no es sim trica. "
                    f" A - A^  _F = {residual:.6e}, tol = {tol:.6e}, "
                    f"asimetr a relativa = {residual / max(norm_A, 1e-300):.6e}."
                )
            logger.debug(
                "[Fase1] Simetr a de '%s': residual=%.3e, tol=%.3e.",
                name, residual, tol,
            )

        def _cholesky_spd(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> Tuple[NDArray[np.float64], float, float, float]:
            r"""
            Verifica SPD de A mediante factorización de Cholesky y calcula
            κ(A) = λ_max / λ_min usando eigvalsh.

            Re-simetriza defensivamente antes de la factorización:
                A_sym = (A + A^⊤) / 2

            Retorna
            -------
            Tuple[NDArray, float, float, float]
                (L, kappa, lambda_min, lambda_max)

            Lanza
            -----
            ApexConditionError
                Si κ(A) > kappa_max.
            ElectrodynamicApexError
                Si A no es SPD (fallo de Cholesky LAPACK).
            """
            A_sym: NDArray[np.float64] = 0.5 * (A + A.T)
            eigvals: NDArray[np.float64] = la.eigvalsh(A_sym)
            lambda_min: float = float(eigvals[0])
            lambda_max: float = float(eigvals[-1])

            # Tolerancia de definici n positiva:  _mach    _max
            tol_pd: float = self._EPS * max(lambda_max, 1.0)
            if lambda_min <= tol_pd:
                raise ElectrodynamicApexError(
                    f"'{name}' no es Definida Positiva (SPD). "
                    f" _min = {lambda_min:.6e} <= tol_pd = {tol_pd:.6e}. "
                    f" _max = {lambda_max:.6e}."
                )

            kappa: float = lambda_max / lambda_min
            if kappa > self._kappa_max:
                raise ApexConditionError(
                    f"'{name}' est  num ricamente mal condicionada: "
                    f"  = {kappa:.6e} >  _max = {self._kappa_max:.6e}. "
                    f" _min = {lambda_min:.6e},  _max = {lambda_max:.6e}. "
                    f"Considere regularizaci n de Tikhonov."
                )

            try:
                L: NDArray[np.float64] = la.cholesky(A_sym, lower=True)
            except la.LinAlgError as exc:
                raise ElectrodynamicApexError(
                    f"Fallo de Cholesky (LAPACK dpotrf) en '{name}' "
                    f"tras validaci n espectral positiva. Error: {exc}"
                ) from exc

            logger.debug(
                "[Fase1] Cholesky '%s':  =%.3e,  _min=%.3e,  _max=%.3e.",
                name, kappa, lambda_min, lambda_max,
            )
            return L, kappa, lambda_min, lambda_max

        def _validate_inverse_consistency(
            self,
            L_G: NDArray[np.float64],
            n: int,
        ) -> float:
            r"""
            Verifica que G_inv = G_μν^{-1} con tolerancia relativa.

            La consistencia se verifica como:

                residual = ‖G_μν · G_inv − I‖_F / n

            La tolerancia es:

                tol_inv = κ(G_μν) · ε_mach

            pues el error de inversión de una matriz con número de condición
            κ es del orden O(κ · ε_mach) según el análisis hacia atrás de Wilkinson.

            Se usa L_G (factor Cholesky) en lugar de re-factorizar G_μν,
            amortizando el coste de la factorización ya realizada en ``_cholesky_spd``.

            Retorna
            -------
            float
                Residuo normalizado ‖G·G_inv − I‖_F / n.

            Lanza
            -----
            MetricInverseError
                Si el residuo supera la tolerancia, con diagnóstico cuantitativo.
            """
            # G_     G_inv usando L_G: G x = L_G (L_G^  x)   solve system
            # M s eficiente: directo por multiplicaci n (ya tenemos G_  )
            prod: NDArray[np.float64] = self._G @ self._G_inv
            I_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)
            residual_mat: NDArray[np.float64] = prod - I_n
            residual: float = float(la.norm(residual_mat, "fro")) / n

            # Tolerancia basada en an lisis de error hacia atr s de Wilkinson
            #  _G se extrae del factor Cholesky:   ~= (L[-1,-1]/L[0,0])
            # Usamos la estimaci n directa por eigvalsh (ya calculada)
            kappa_G_est: float = (
                float(L_G[-1, -1] / L_G[0, 0]) ** 2
            )
            tol_inv: float = kappa_G_est * self._EPS * n

            if residual > tol_inv:
                raise MetricInverseError(
                    f"G_inv no es la inversa de G_  . "
                    f" G G_inv - I _F / n = {residual:.6e} > "
                    f"tol_inv = {tol_inv:.6e} ( _est   n). "
                    f"Inconsistencia m trica que compromete la variedad."
                )

            logger.debug(
                "[Fase1] Consistencia m trica: residual/n=%.3e, tol=%.3e.",
                residual, tol_inv,
            )
            return residual

        def _validate_psd_and_sqrt(
            self,
            R: NDArray[np.float64],
            name: str,
            n: int,
        ) -> Tuple[NDArray[np.float64], int]:
            r"""
            Verifica que R ⪰ 0 (PSD) con tolerancia normalizada y calcula
            su raíz cuadrada espectral:

                R_sqrt = V · diag(√max(λ, 0)) · V^⊤

            donde R = V · diag(λ) · V^⊤ es la descomposición espectral.

            La tolerancia distingue autovalores nulos por diseño de
            autovalores genuinamente negativos (violación PSD):

                tol_psd = ε_mach · ‖R‖_F

            Retorna
            -------
            Tuple[NDArray[np.float64], int]
                (R_sqrt, rank_R) donde rank_R = #{λ_i > tol_psd}.

            Lanza
            -----
            ApexSymmetryError
                Si λ_min < −tol_psd (entropía negativa genuina).
            """
            norm_R: float = float(la.norm(R, "fro"))
            tol_psd: float = self._EPS * max(norm_R, 1.0)

            eigvals: NDArray[np.float64]
            eigvecs: NDArray[np.float64]
            eigvals, eigvecs = la.eigh(0.5 * (R + R.T))

            lambda_min: float = float(eigvals[0])
            if lambda_min < -tol_psd:
                raise ApexSymmetryError(
                    f"'{name}' no es Semidefinida Positiva (PSD). "
                    f" _min = {lambda_min:.6e} < -tol = {-tol_psd:.6e}. "
                    f"Entrop a negativa detectada (ganancia fantasma)."
                )

            eigvals_clamped: NDArray[np.float64] = np.maximum(eigvals, 0.0)
            R_sqrt: NDArray[np.float64] = (
                eigvecs * np.sqrt(eigvals_clamped)[np.newaxis, :]
            ) @ eigvecs.T
            R_sqrt = 0.5 * (R_sqrt + R_sqrt.T)

            rank_R: int = int(np.sum(eigvals_clamped > tol_psd))

            logger.debug(
                "[Fase1] PSD '%s': rank=%d/%d,  _min=%.3e,  _max=%.3e.",
                name, rank_R, n, lambda_min, float(eigvals[-1]),
            )
            return R_sqrt, rank_R

        #
        # M todo terminal de la Fase 1   entrada directa de la Fase 2
        #

        def build_context(self) -> "ApexPreparationContext":
            r"""
            **Método terminal de la Fase 1.**

            Ejecuta en secuencia estricta y ordenada todos los métodos de
            validación y precálculo, y empaqueta sus resultados en un
            ``ApexPreparationContext`` inmutable.

            El ``ApexPreparationContext`` resultante es el **único argumento**
            que necesita el constructor de ``Phase2_ElectrodynamicSynthesis``,
            garantizando la continuidad formal entre fases.

            Flujo interno
            -------------
            1. Verificación dimensional (G_μν, G_inv, R_cost).
            2. Simetría de G_μν y R_cost.
            3. SPD y κ de G_μν → factor Cholesky L_G.
            4. SPD y κ de G_inv.
            5. Consistencia métrica G · G_inv ≈ I.
            6. PSD de R_cost → raíz espectral R_sqrt y rank_R.
            7. Empaquetado en ApexPreparationContext.

            Retorna
            -------
            ApexPreparationContext
                Contexto métrico completo, inmutable y listo para Fase 2.
            """
            #    Paso 1: Dimensiones
            n: int = self._check_dimensions()

            #    Paso 2: Simetr a
            self._validate_symmetry(self._G, "G_mu_nu")
            self._validate_symmetry(self._G_inv, "G_inv")
            self._validate_symmetry(self._R, "R_cost")

            #    Paso 3: SPD de G_   + factor Cholesky
            L_G, kappa_G, _, _ = self._cholesky_spd(self._G, "G_mu_nu")

            #    Paso 4: SPD de G_inv
            _, kappa_G_inv, _, _ = self._cholesky_spd(self._G_inv, "G_inv")
            logger.debug("[Fase1]  (G_inv)=%.3e (debe ser ~=  (G)=%.3e).", kappa_G_inv, kappa_G)

            #    Paso 5: Consistencia m trica
            inv_residual: float = self._validate_inverse_consistency(L_G, n)

            #    Paso 6: PSD de R_cost + ra z espectral
            R_sqrt, rank_R = self._validate_psd_and_sqrt(self._R, "R_cost", n)

            #    Paso 7: Empaquetado
            context = ApexPreparationContext(
                G_mu_nu=self._G.copy(),
                G_inv=self._G_inv.copy(),
                R_cost=self._R.copy(),
                L_G=L_G,
                R_sqrt=R_sqrt,
                kappa_G=kappa_G,
                rank_R=rank_R,
                inverse_residual=inv_residual,
                dim=n,
            )

            logger.info(
                "[Fase1] ApexPreparationContext ensamblado: "
                "dim=%d,  (G)=%.3e, rank(R)=%d, inv_res=%.3e.",
                n, kappa_G, rank_R, inv_residual,
            )

            #    Contrato de interfaz Fase 1   Fase 2
            return context

    #
    # FASE 2   S NTESIS ELECTRODIN MICA
    #

    class Phase2_ElectrodynamicSynthesis:
        r"""
        **Fase 2 – Síntesis Electrodinámica.**

        Recibe el ``ApexPreparationContext`` de la Fase 1 y lo usa para
        ejecutar los cuatro subprocesos electrodinámicos fundamentales:

          1. **Inyección de Potencial de Gauge** (``inject_gauge_potential``):
             Calcula s_val = dΦ · exp(−½ Tr(G_μν)) con diagnóstico previo
             del factor de supresión y verificación relativa de la norma.

          2. **Refracción Eikonal de Mercado** (``compute_eikonal_absorption``):
             Valida la dimensión de phase_gradient, calcula la norma
             riemanniana exacta G^μν ∂S ∂S sin bifurcación por ValueError,
             y verifica la ecuación Eikonal con tolerancia relativa configurable.

          3. **Exergía de Poynting** (``evaluate_poynting_exergy``):
             Valida dimensiones de E_field, H_field, grad_H, calcula
             P_diss como forma cuadrática rigurosa y verifica la 2ª Ley
             con tolerancia relativa a max(|P_in|, |P_diss|, 1).

          4. **Auditoría de Holonomía Yang-Mills** (``audit_yang_mills_holonomy``):
             Calcula la curvatura correcta F_μν = A − A^⊤ + [A, A^⊤]
             (discr. de dA + ½[A∧A]) y la acción Yang-Mills ponderada por
             G_μν: S_YM = ½ Tr(F^⊤ G F G_inv). Umbral relativo a ‖A‖_F².

        El constructor es la **continuación directa** de ``build_context``.
        """

        _EPS: float = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            context: "ApexPreparationContext",
            eikonal_slack: float,
            holonomy_tol_rel: float,
        ) -> None:
            r"""
            **Constructor de la Fase 2: continuación directa de la Fase 1.**

            Recibe el ``ApexPreparationContext`` devuelto por
            ``Phase1_MetricValidation.build_context()`` y extrae las
            matrices y factorizaciones precomputadas para uso eficiente.

            No realiza ninguna validación adicional: la corrección de todas
            las matrices está garantizada por la Fase 1.
            """
            self._ctx: "ApexPreparationContext" = context
            self._eikonal_slack: float = eikonal_slack
            self._holonomy_tol_rel: float = holonomy_tol_rel

            # Precomputa Tr(G_  ) una sola vez (invariante durante la vida del agente)
            self._trace_G: float = float(np.trace(context.G_mu_nu))

            # Factor de supresi n de Gauge: exp(-  Tr(G))
            # Evaluado con protecci n de desbordamiento aritm tico
            self._suppression_factor: float = float(
                np.exp(-0.5 * self._trace_G)
            )

            logger.debug(
                "[Fase2] Inicializada: dim=%d, Tr(G)=%.6e, "
                "suppression=%.6e, eikonal_slack=%.3f, holo_tol=%.3e.",
                context.dim,
                self._trace_G,
                self._suppression_factor,
                eikonal_slack,
                holonomy_tol_rel,
            )

        #
        # Subproceso 1: Inyecci n de Potencial de Gauge
        #

        def inject_gauge_potential(
            self,
            d_Phi: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Calcula el vector de inyección de FEM (Fuerza Electromotriz):

                s_val = dΦ · exp(−½ Tr(G_μν))

            El factor de supresión exp(−½ Tr(G_μν)) modela el estrés
            estructural interno: un tensor métrico de traza grande corresponde
            a un espacio de propuestas altamente contraído, suprimiendo la
            capacidad de inyección de valor.

            El factor de supresión se precalcula en el constructor (invariante
            durante la vida del agente) para evitar re-evaluaciones costosas.

            Verificación de supresión total:
                ‖s_val‖_∞ / max(‖dΦ‖_∞, 1) < ε_mach

            Si la supresión es total (relativa a la norma de dΦ), se lanza
            ``GaugePotentialError`` con diagnóstico de Tr(G_μν).

            Parámetros
            ----------
            d_Phi : NDArray[np.float64], shape (n,)
                Diferencial exterior del potencial de Gauge dΦ.

            Retorna
            -------
            Tuple[NDArray[np.float64], float]
                (s_val, suppression_factor):
                  s_val             ∈ ℝⁿ: vector de inyección.
                  suppression_factor ∈ (0,1]: factor de supresión exp(−½Tr(G)).

            Lanza
            -----
            ApexDimensionError
                Si d_Phi no tiene shape (n,).
            GaugePotentialError
                Si la supresión relativa colapsa s_val a la resolución numérica.
            """
            n: int = self._ctx.dim

            if d_Phi.shape != (n,):
                raise ApexDimensionError(
                    f"d_Phi debe tener shape ({n},); se obtuvo {d_Phi.shape}."
                )

            s_val: NDArray[np.float64] = d_Phi * self._suppression_factor

            # Verificaci n de supresi n relativa (no absoluta)
            norm_s: float = float(la.norm(s_val, np.inf))
            norm_dPhi: float = float(la.norm(d_Phi, np.inf))
            suppression_rel: float = norm_s / max(norm_dPhi, 1.0)

            if suppression_rel < self._EPS:
                logger.warning(
                    "[Fase2] Supresi n de Gauge total: "
                    " s_val _ / d  _  = %.3e <  _mach. "
                    "Tr(G_  ) = %.6e (estr s extremo).",
                    suppression_rel, self._trace_G,
                )
                raise GaugePotentialError(
                    f"Estr s estructural extremo: Tr(G_  ) = {self._trace_G:.6e}. "
                    f"Factor de supresi n = {self._suppression_factor:.6e}. "
                    f"Supresi n relativa  s_val / d   = {suppression_rel:.6e} <  _mach. "
                    f"Inyecci n de propuesta de valor completamente colapsada."
                )

            logger.debug(
                "[Fase2] Inyecci n de calibre: suppression=%.6e, "
                " s_val _ =%.6e,  d  _ =%.6e.",
                self._suppression_factor, norm_s, norm_dPhi,
            )
            return s_val, self._suppression_factor

        #
        # Subproceso 2: Refracci n Eikonal de Mercado
        #

        def compute_eikonal_absorption(
            self,
            phase_gradient: NDArray[np.float64],
            sigma_stress: float,
            alpha_fermat: float = 0.5,
        ) -> Tuple[float, float]:
            r"""
            Calcula el índice de refracción efectivo de Fermat y verifica
            la ecuación Eikonal de absorción de mercado:

                G^μν ∂_μS ∂_νS = n²(σ*)
                n(σ*) = 1 + tanh(α_fermat · σ*)

            La norma riemanniana se calcula directamente como forma cuadrática:

                ‖∂S‖²_{G_inv} = phase_gradient^⊤ · G_inv · phase_gradient

            usando la matriz G_inv precalculada en el contexto (sin ninguna
            inversión adicional).

            La verificación Eikonal acepta una tolerancia relativa configurable:

                ‖∂S‖²_{G_inv} ≥ n²(σ*) · (1 − eikonal_slack)

            Parámetros
            ----------
            phase_gradient : NDArray[np.float64], shape (n,)
                Gradiente de la fase de mercado ∂S en el espacio de calibre.
            sigma_stress : float
                Estrés de mercado normalizado σ* ∈ ℝ.
            alpha_fermat : float, default 0.5
                Coeficiente de sensibilidad de la ley de Fermat.

            Retorna
            -------
            Tuple[float, float]
                (n_refract, eikonal_norm_sq):
                  n_refract      : índice de refracción n(σ*).
                  eikonal_norm_sq: ‖∂S‖²_{G_inv} (norma riemanniana cuadrada).

            Lanza
            -----
            ApexDimensionError
                Si phase_gradient no tiene shape (n,).
            EikonalRefractionError
                Si la ecuación Eikonal no se satisface dentro de eikonal_slack.
            """
            n: int = self._ctx.dim

            if phase_gradient.shape != (n,):
                raise ApexDimensionError(
                    f"phase_gradient debe tener shape ({n},); "
                    f"se obtuvo {phase_gradient.shape}."
                )

            #  ndice de refracci n de Fermat: n( *) = 1 + tanh(     *)
            n_refract: float = 1.0 + float(np.tanh(alpha_fermat * sigma_stress))

            # Norma riemanniana cuadrada: phase_gradient^  G_inv phase_gradient
            # G_inv ya est  validado como SPD; esta forma cuadr tica es >= 0
            G_inv_grad: NDArray[np.float64] = self._ctx.G_inv @ phase_gradient
            eikonal_norm_sq: float = float(np.dot(phase_gradient, G_inv_grad))

            # Umbral Eikonal con tolerancia relativa configurable
            n_sq: float = n_refract ** 2
            eikonal_threshold: float = n_sq * (1.0 - self._eikonal_slack)

            if eikonal_norm_sq < eikonal_threshold:
                logger.error(
                    "[Fase2] Fallo Eikonal:   S  _{G_inv}=%.6e < "
                    "n  (1-slack)=%.6e (n=%.4f, slack=%.3f).",
                    eikonal_norm_sq, eikonal_threshold,
                    n_refract, self._eikonal_slack,
                )
                raise EikonalRefractionError(
                    f"El mercado objetivo exhibe fricci n extrema (fallo Eikonal). "
                    f"  S  _{{G_inv}} = {eikonal_norm_sq:.6e} < "
                    f"n  (1-slack) = {eikonal_threshold:.6e}. "
                    f"n( *) = {n_refract:.6f},  * = {sigma_stress:.6f}. "
                    f"La campa a se dispers  antes de alcanzar el colector."
                )

            logger.debug(
                "[Fase2] Eikonal OK:   S  =%.6e, n =%.6e, "
                "n_refract=%.6f, margen=%.1f%%.",
                eikonal_norm_sq, n_sq,
                n_refract,
                100.0 * (eikonal_norm_sq / n_sq - 1.0 + self._eikonal_slack),
            )
            return n_refract, eikonal_norm_sq

        #
        # Subproceso 3: Exerg a de Poynting
        #

        def evaluate_poynting_exergy(
            self,
            E_field: NDArray[np.float64],
            H_field: NDArray[np.float64],
            grad_H: NDArray[np.float64],
        ) -> Tuple[float, float, float]:
            r"""
            Evalúa el flujo de Poynting y la exergía operativa neta:

                P_in     = E · H                   (ingresos brutos)
                P_diss   = ∇H^⊤ · R_cost · ∇H     (disipación ≥ 0)
                P_exergia = P_in − P_diss           (debe ser ≥ −tol)

            La tolerancia para detectar un Agujero Negro Financiero genuino
            es relativa a la magnitud de las potencias involucradas:

                tol_exergy = ε_mach · max(|P_in|, |P_diss|, 1)

            Esto evita disparos espurios por errores de redondeo cuando
            P_in ≈ P_diss (equilibrio termodinámico liminal).

            Parámetros
            ----------
            E_field : NDArray[np.float64], shape (n,)
                Campo eléctrico (vector de ingresos por canal).
            H_field : NDArray[np.float64], shape (n,)
                Campo magnético (vector de actividad por canal).
            grad_H : NDArray[np.float64], shape (n,)
                Gradiente del Hamiltoniano operativo ∇H(x).

            Retorna
            -------
            Tuple[float, float, float]
                (P_in, P_diss, P_exergia):
                  P_in      : potencia de entrada (ingresos brutos).
                  P_diss    : potencia disipada (costes).
                  P_exergia : exergía neta ≥ 0.

            Lanza
            -----
            ApexDimensionError
                Si E_field, H_field o grad_H no tienen shape (n,).
            FinancialBlackHoleError
                Si P_exergia < −tol_exergy (disipación supera ingresos).
            """
            n: int = self._ctx.dim

            for vec, name in [
                (E_field, "E_field"),
                (H_field, "H_field"),
                (grad_H, "grad_H"),
            ]:
                if vec.shape != (n,):
                    raise ApexDimensionError(
                        f"'{name}' debe tener shape ({n},); se obtuvo {vec.shape}."
                    )

            # Potencia de entrada: producto escalar euclidiano
            P_in: float = float(np.dot(E_field, H_field))

            # Potencia disipada: forma cuadr tica R_cost (>= 0 pues R_cost   0)
            R_grad: NDArray[np.float64] = self._ctx.R_cost @ grad_H
            P_diss: float = float(np.dot(grad_H, R_grad))

            # Exerg a neta
            P_exergia: float = P_in - P_diss

            # Tolerancia relativa para detectar agujero negro genuino
            tol_exergy: float = self._EPS * max(abs(P_in), abs(P_diss), 1.0)

            if P_exergia < -tol_exergy:
                logger.critical(
                    "[Fase2] Agujero Negro Financiero: "
                    "P_in=%.6e, P_diss=%.6e, P_exergia=%.6e < -tol=%.6e.",
                    P_in, P_diss, P_exergia, -tol_exergy,
                )
                raise FinancialBlackHoleError(
                    f"La entrop a operativa devora la energ a inyectada. "
                    f"P_in = {P_in:.6e}, P_diss = {P_diss:.6e}, "
                    f"P_exergia = {P_exergia:.6e} < -tol = {-tol_exergy:.6e}. "
                    f"Veto termodin mico absoluto emitido."
                )

            logger.debug(
                "[Fase2] Poynting: P_in=%.6e, P_diss=%.6e, "
                "P_exergia=%.6e, margen_relativo=%.1f%%.",
                P_in, P_diss, P_exergia,
                100.0 * P_exergia / max(abs(P_in), 1.0),
            )
            return P_in, P_diss, P_exergia

        #
        # Subproceso 4: Auditor a de Holonom a Yang-Mills
        #

        def audit_yang_mills_holonomy(
            self,
            A_gauge: NDArray[np.float64],
        ) -> float:
            r"""
            Audita la curvatura de calibre mediante la acción de Yang-Mills
            ponderada por el tensor métrico G_μν.

            Curvatura discreta (2-forma de Yang-Mills discretizada):
            En la discretización matricial, A_gauge ∈ ℝ^{n×n} representa
            la 1-forma de conexión en un único plaqueta. La curvatura es:

                F = A − A^⊤ + [A, A^⊤]

            donde:
              • A − A^⊤: parte antisimétrica de A (análogo de dA en el
                continuo, diferencia de 1-formas en bordes opuestos)
              • [A, A^⊤] = A·A^⊤ − A^⊤·A: conmutador de Lie de A con su
                adjunta (término no abeliano ½[A∧A] discretizado)

            Acción de Yang-Mills ponderada por G_μν:

                S_YM = ½ Tr(F^⊤ · G_μν · F · G^μν)

            Esta forma pondera la curvatura con la métrica del espacio de
            calibre, siendo invariante bajo transformaciones de base.

            Umbral relativo (en lugar del absoluto original 1e-6):

                tol_ym = holonomy_tol_rel · max(‖A_gauge‖_F², 1)

            Parámetros
            ----------
            A_gauge : NDArray[np.float64], shape (n, n)
                Potencial de calibre (1-forma de conexión discretizada).

            Retorna
            -------
            float
                S_YM ≥ 0: acción de Yang-Mills ponderada.

            Lanza
            -----
            ApexDimensionError
                Si A_gauge no tiene shape (n, n).
            HolonomyVetoError
                Si S_YM > tol_ym (curvatura de calibre no nula detectada).
            """
            n: int = self._ctx.dim

            if A_gauge.shape != (n, n):
                raise ApexDimensionError(
                    f"A_gauge debe tener shape ({n},{n}); "
                    f"se obtuvo {A_gauge.shape}."
                )

            #    Curvatura discreta F = (A - A^ ) + [A, A^ ]
            # Parte antisim trica: discretizaci n de dA en un plaqueta
            A_antisym: NDArray[np.float64] = A_gauge - A_gauge.T

            # Conmutador de Lie: [A, A^ ] = A A^  - A^  A
            # Nota: en la teor a de Yang-Mills abeliana, [A, A] = 0;
            # en la no-abeliana (grupo GL(n, )), el conmutador es no nulo.
            A_AT: NDArray[np.float64] = A_gauge @ A_gauge.T
            AT_A: NDArray[np.float64] = A_gauge.T @ A_gauge
            commutator_lie: NDArray[np.float64] = A_AT - AT_A

            # 2-forma de curvatura discreta
            F_mu_nu: NDArray[np.float64] = A_antisym + commutator_lie

            #    Acci n de Yang-Mills ponderada por G_
            # S_YM =   Tr(F^    G_     F   G^  )
            # Equivalente a:    F  _{G,G_inv} =   Tr((G^{1/2} F G^{-1/2})^  (G^{1/2} F G^{-1/2}))
            G_F: NDArray[np.float64] = self._ctx.G_mu_nu @ F_mu_nu
            F_G_inv: NDArray[np.float64] = F_mu_nu @ self._ctx.G_inv
            S_ym: float = 0.5 * float(np.trace(F_mu_nu.T @ G_F @ F_G_inv))

            # S_YM debe ser no negativo por construcci n (norma matricial ponderada)
            # Si es levemente negativo por errores de redondeo, lo clampeamos
            S_ym = max(S_ym, 0.0)

            #    Umbral relativo
            norm_A_sq: float = float(la.norm(A_gauge, "fro") ** 2)
            tol_ym: float = self._holonomy_tol_rel * max(norm_A_sq, 1.0)

            if S_ym > tol_ym:
                logger.error(
                    "[Fase2] Holonom a no nula: S_YM=%.6e > tol=%.6e "
                    "( A _F =%.6e).",
                    S_ym, tol_ym, norm_A_sq,
                )
                raise HolonomyVetoError(
                    f"El bucle de Wilson revela curvatura de calibre no nula. "
                    f"S_YM = {S_ym:.6e} > tol_rel = {tol_ym:.6e}. "
                    f" A_gauge _F  = {norm_A_sq:.6e}. "
                    f"Fugas log sticas ocultas (ciclos par sitos) detectadas."
                )

            logger.debug(
                "[Fase2] Holonom a: S_YM=%.6e, tol=%.6e, "
                "margen=%.1f%%.",
                S_ym, tol_ym,
                100.0 * (1.0 - S_ym / max(tol_ym, 1e-300)),
            )
            return S_ym

        #
        # M todo terminal de la Fase 2   entrada directa de la Fase 3
        #

        def synthesize(
            self,
            d_Phi: NDArray[np.float64],
            phase_gradient: NDArray[np.float64],
            sigma_stress: float,
            E_field: NDArray[np.float64],
            H_field: NDArray[np.float64],
            grad_H: NDArray[np.float64],
            A_gauge: NDArray[np.float64],
            alpha_fermat: float = 0.5,
        ) -> "ApexStateTensor":
            r"""
            **Método terminal de la Fase 2.**

            Integra los cuatro subprocesos electrodinámicos en secuencia
            determinista y retorna el ``ApexStateTensor`` completo.

            El ``ApexStateTensor`` resultante contiene ``gauge_injection_vector``
            (s_val) que es el **dato primario** consumido por la Fase 3 para
            construir la proyección en la fibra. La frontera formal entre Fase 2
            y Fase 3 es el campo ``gauge_injection_vector`` de este tensor.

            Parámetros
            ----------
            d_Phi : NDArray[np.float64], shape (n,)
                Diferencial del potencial de Gauge.
            phase_gradient : NDArray[np.float64], shape (n,)
                Gradiente de la función de fase de mercado.
            sigma_stress : float
                Estrés de mercado normalizado.
            E_field : NDArray[np.float64], shape (n,)
                Campo eléctrico (ingresos por canal).
            H_field : NDArray[np.float64], shape (n,)
                Campo magnético (actividad por canal).
            grad_H : NDArray[np.float64], shape (n,)
                Gradiente del Hamiltoniano operativo.
            A_gauge : NDArray[np.float64], shape (n, n)
                Potencial de calibre (1-forma de conexión).
            alpha_fermat : float, default 0.5
                Coeficiente de sensibilidad Eikonal.

            Retorna
            -------
            ApexStateTensor
                Estado electrodinámico completo, inmutable.
            """
            #    Subproceso 1: Inyecci n de Gauge
            s_val, suppression = self.inject_gauge_potential(d_Phi)

            #    Subproceso 2: Refracci n Eikonal
            n_refract, eikonal_norm_sq = self.compute_eikonal_absorption(
                phase_gradient, sigma_stress, alpha_fermat
            )

            #    Subproceso 3: Exerg a de Poynting
            P_in, P_diss, P_exergia = self.evaluate_poynting_exergy(
                E_field, H_field, grad_H
            )

            #    Subproceso 4: Holonom a Yang-Mills
            S_ym = self.audit_yang_mills_holonomy(A_gauge)

            logger.info(
                "[Fase2] S ntesis electrodin mica completada: "
                "suppression=%.3e, n_refract=%.4f, "
                "P_exergia=%.6e, S_YM=%.6e.",
                suppression, n_refract, P_exergia, S_ym,
            )

            #    Contrato de interfaz Fase 2   Fase 3
            # `gauge_injection_vector = s_val` es el argumento directo de
            # Phase3_SheafProjection.export_stalk(). Esta devoluci n es la
            # frontera formal entre ambas fases anidadas.
            return ApexStateTensor(
                gauge_injection_vector=s_val,
                suppression_factor=suppression,
                fermat_refractive_index=n_refract,
                eikonal_norm_sq=eikonal_norm_sq,
                poynting_income=P_in,
                poynting_dissipation=P_diss,
                poynting_exergy_flux=P_exergia,
                yang_mills_action=S_ym,
                is_electrodynamically_viable=True,
            )

    #
    # FASE 3   PROYECCI N EN HACES Y COFRONTERA DISCRETA  _{APEX}
    #

    class Phase3_SheafProjection:
        r"""
        **Fase 3 – Proyección en Haces y Cofrontera Discreta δ_{APEX}.**

        Recibe el ``ApexPreparationContext`` de la Fase 1 (vía el agente
        orquestador) y el vector de inyección s_val producido por la Fase 2,
        construyendo el ``SheafStalkApex`` que alimenta el Laplaciano de Haz.

        El constructor es la **continuación directa** del método ``synthesize``
        de la Fase 2: el campo ``gauge_injection_vector`` del ``ApexStateTensor``
        devuelto allá es lo que aquí se proyecta.

        Fundamento Matemático de la Cofrontera δ_{APEX}
        -------------------------------------------------
        En el fibrado principal π: P → M sobre el espacio de calibre,
        la restricción del stalk en K_{APEX} es el operador:

            δ_{APEX}: F(K_{APEX}) → F(e_{APEX})

        Para que el Laplaciano de Haz global sea consistente con la métrica
        riemanniana G_μν del espacio de calibre, δ_{APEX} debe satisfacer
        la identidad de la métrica de Hodge local:

            δ_{APEX}^⊤ · G_μν · δ_{APEX} = I_n

        La elección canónica que satisface esta identidad es:

            δ_{APEX} = G_μν^{-1/2} = (L_G^⊤)^{-1} = L_G^{-⊤}

        donde L_G es el factor Cholesky inferior de G_μν = L_G · L_G^⊤.

        Verificación:
            δ_{APEX}^⊤ G_μν δ_{APEX}
              = L_G^{-1} · (L_G · L_G^⊤) · L_G^{-⊤}
              = L_G^{-1} · L_G · (L_G^⊤ · L_G^{-⊤})
              = I · I = I  ✓

        Esta elección es la inversa de la norma riemanniana local:
        δ_{APEX} "deshace" la métrica para producir coordenadas normales.
        """

        _EPS: float = float(np.finfo(np.float64).eps)

        def __init__(self, context: "ApexPreparationContext") -> None:
            r"""
            **Constructor de la Fase 3: continuación directa de la Fase 2.**

            Precalcula δ_{APEX} = L_G^{-⊤} usando el factor Cholesky L_G
            almacenado en el contexto (sin re-factorizar G_μν).

            La precalculación en el constructor amortiza el coste sobre
            múltiples llamadas a ``export_stalk``.

            Lanza
            -----
            SheafMetricError
                Si la identidad de Hodge δ^⊤ G δ = I no se satisface
                dentro de 100·ε_mach relativo.
            """
            self._ctx: "ApexPreparationContext" = context
            n: int = context.dim

            #     _{APEX} = L_G^{- } (triangular superior)
            # L_G^{-1}: soluci n de L_G   X = I (sustituci n hacia adelante)
            I_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)
            L_G_inv: NDArray[np.float64] = la.solve_triangular(
                context.L_G, I_n, lower=True, check_finite=False
            )
            # L_G^{- } = (L_G^{-1})^ : triangular superior
            self._delta_apex: NDArray[np.float64] = L_G_inv.T

            # Rango:  _{APEX} es cuadrada y no singular (G_     0 implica L_G no singular)
            self._rank_delta: int = n

            #    Verificaci n de la identidad de Hodge local
            self._hodge_residual: float = self._verify_hodge_identity()

            logger.debug(
                "[Fase3]  _{APEX} precalculada: dim=%d, rank=%d, "
                "Hodge_residual=%.3e.",
                n, self._rank_delta, self._hodge_residual,
            )

        def _verify_hodge_identity(self) -> float:
            r"""
            Verifica la identidad de la métrica de Hodge local:

                ‖ δ_{APEX}^⊤ · G_μν · δ_{APEX} − I ‖_F / n < 100 · ε_mach

            Esta identidad garantiza que la cofrontera δ_{APEX} es la
            isometría correcta entre la fibra local y el espacio euclídeo,
            asegurando que el Laplaciano de Haz global sea el operador de
            Hodge-Kirchhoff correcto.

            Retorna
            -------
            float
                Error relativo ‖δ^⊤ G δ − I‖_F / n.

            Lanza
            -----
            SheafMetricError
                Si el error relativo supera 100·ε_mach.
            """
            n: int = self._ctx.dim

            #  ^    G     = L_G^{-1}   (L_G   L_G^ )   L_G^{- } = I (exacto)
            delta_T_G_delta: NDArray[np.float64] = (
                self._delta_apex.T
                @ self._ctx.G_mu_nu
                @ self._delta_apex
            )
            I_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)
            residual_F: float = float(la.norm(delta_T_G_delta - I_n, "fro"))
            rel_error: float = residual_F / n

            tol_hodge: float = 100.0 * self._EPS

            if rel_error > tol_hodge:
                raise SheafMetricError(
                    f"Identidad de Hodge violada: "
                    f"  ^  G   - I _F / n = {rel_error:.6e} > tol = {tol_hodge:.6e}. "
                    f"Error de ensamble en la cofrontera  _{{APEX}}."
                )

            return rel_error

        #
        # M todo terminal de la Fase 3 (salida p blica del ecosistema)
        #

        def export_stalk(
            self,
            s_val: NDArray[np.float64],
        ) -> "SheafStalkApex":
            r"""
            **Método terminal de la Fase 3 y del agente completo.**

            Proyecta el vector de inyección s_val sobre la fibra local
            mediante δ_{APEX} = G_μν^{-1/2} (precalculada en el constructor)
            y retorna el ``SheafStalkApex`` completo.

            La proyección δ_{APEX} · s_val mapea el vector de inyección de
            FEM del espacio riemanniano al espacio euclídeo de la fibra,
            eliminando la curvatura métrica local para el cálculo global
            del Laplaciano de Haz.

            Parámetros
            ----------
            s_val : NDArray[np.float64], shape (n,)
                Vector de inyección de Gauge producido por la Fase 2.
                Típicamente: ``ApexStateTensor.gauge_injection_vector``.

            Retorna
            -------
            SheafStalkApex
                Fibrado celular completo, inmutable.

            Lanza
            -----
            ApexDimensionError
                Si s_val no tiene shape (n,).
            """
            n: int = self._ctx.dim

            if s_val.shape != (n,):
                raise ApexDimensionError(
                    f"s_val debe tener shape ({n},); se obtuvo {s_val.shape}."
                )

            # Proyecci n sobre la fibra:  _{APEX}   s_val
            projected: NDArray[np.float64] = self._delta_apex @ s_val

            logger.info(
                "[Fase3] SheafStalkApex exportado: dim=%d, rank=%d, "
                "Hodge_res=%.3e,    s =%.6e.",
                n,
                self._rank_delta,
                self._hodge_residual,
                float(la.norm(projected, 2)),
            )

            #    Contrato de salida del agente completo
            return SheafStalkApex(
                delta_apex=self._delta_apex,
                hodge_metric_residual=self._hodge_residual,
                source_injection=s_val.copy(),
                projected_source=projected,
                rank_delta=self._rank_delta,
            )

    #
    # INTERFAZ P BLICA DEL AGENTE (punto de entrada externo)
    #

    def synthesize_apex_field(
        self,
        d_Phi: NDArray[np.float64],
        phase_gradient: NDArray[np.float64],
        sigma_stress: float,
        E_field: NDArray[np.float64],
        H_field: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        A_gauge: NDArray[np.float64],
        alpha_fermat: float = 0.5,
    ) -> ApexStateTensor:
        r"""
        Punto de entrada público para la síntesis electrodinámica completa.

        Delega íntegramente en ``Phase2_ElectrodynamicSynthesis.synthesize``,
        que ya posee el contexto métrico validado por la Fase 1.

        Parámetros
        ----------
        d_Phi : NDArray[np.float64], shape (n,)
            Diferencial del potencial de Gauge dΦ.
        phase_gradient : NDArray[np.float64], shape (n,)
            Gradiente de la función de fase de mercado ∂S.
        sigma_stress : float
            Estrés de mercado normalizado σ*.
        E_field : NDArray[np.float64], shape (n,)
            Campo eléctrico (vector de ingresos por canal).
        H_field : NDArray[np.float64], shape (n,)
            Campo magnético (vector de actividad por canal).
        grad_H : NDArray[np.float64], shape (n,)
            Gradiente del Hamiltoniano operativo ∇H(x).
        A_gauge : NDArray[np.float64], shape (n, n)
            Potencial de calibre (1-forma de conexión discretizada).
        alpha_fermat : float, default 0.5
            Coeficiente de sensibilidad de la ley de Fermat.

        Retorna
        -------
        ApexStateTensor
            Estado electrodinámico completo e inmutable.
        """
        return self.phase2.synthesize(
            d_Phi=d_Phi,
            phase_gradient=phase_gradient,
            sigma_stress=sigma_stress,
            E_field=E_field,
            H_field=H_field,
            grad_H=grad_H,
            A_gauge=A_gauge,
            alpha_fermat=alpha_fermat,
        )

    def export_sheaf_stalk(
        self,
        s_val: NDArray[np.float64],
    ) -> SheafStalkApex:
        r"""
        Exporta el Stalk del haz electrodinámico y la cofrontera δ_{APEX}.

        Instancia la Fase 3 perezosamente en la primera llamada: el coste de
        calcular L_G^{-⊤} se paga una sola vez y se reutiliza para múltiples
        proyecciones (δ_{APEX} es invariante mientras G_μν no cambie).

        Parámetros
        ----------
        s_val : NDArray[np.float64], shape (n,)
            Vector de inyección de Gauge. Típicamente:
            ``ApexStateTensor.gauge_injection_vector`` del estado actual.

        Retorna
        -------
        SheafStalkApex
            Fibrado celular completo e inmutable.
        """
        if self.phase3 is None:
            self.phase3 = KApexElectrodynamicAgent.Phase3_SheafProjection(
                context=self.context
            )
            logger.info(
                "[KApexElectrodynamicAgent] Phase3_SheafProjection "
                "instanciada (lazy init): rank_delta=%d, "
                "Hodge_residual=%.3e.",
                self.phase3._rank_delta,
                self.phase3._hodge_residual,
            )

        return self.phase3.export_stalk(s_val=s_val)
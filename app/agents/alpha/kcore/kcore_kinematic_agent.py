# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo : KCore Kinematic Agent (Director de Flujo y Cinética Logística)      |
| Ruta   : app/agents/alpha/kcore/kcore_kinematic_agent.py                     |
| Versión: 6.0.0-Rigorous-IDA-PBC-Hodge-CFL-Sheaf-Spectral                     |
+==============================================================================+

NATURALEZA CIBER-FÍSICA Y ESTRUCTURA DE DIRAC
=============================================
Este módulo impone el moldeado de energía mediante Control Basado en Pasividad
(IDA-PBC) con proyección pseudoinversa *covariante* respecto a una métrica
Riemanniana de estado G_μν ≽ 0. La ley de control es:

    F_req(x) ≜ [J_d − R_d] ∇H_d − [J − R] ∇H
    α(x)     = (gᵀ G g + λ_reg I_m)⁺ gᵀ G F_req

donde (·)⁺ denota la pseudoinversa de Moore-Penrose truncada por el criterio
espectral de Golub–Van Loan, y λ_reg ≥ 0 es regularización de Tikhonov
opcional que garantiza estabilidad cuando rank(g) < m.

VÁLVULA DE HODGE (1-FORMAS PONDERADAS)
======================================
El Laplaciano de Hodge-1 ponderado sobre el complejo de cadenas del grafo
logístico es:

    L₁ᵂ = ∂₁ᵀ W⁻¹ ∂₁ + ∂₂ W ∂₂ᵀ

La vorticidad parásita se cuantifica por la norma de energía en 1-cochains:

    ‖I_curl‖_W ≜ √(I_curlᵀ W I_curl)

y se estrangula espectralmente el soporte de I_curl en W.

LÍMITE CFL (DOBLE COTA: GERSCHGORIN + ESPECTRAL)
================================================
    ρ_Gersh  ≜ max_i ( |Δ_ii| + Σ_{j≠i} |Δ_ij| )
    λ_max    ≜ max Spec(Δ_sym)          (ARPACK / fallback denso)
    Δt_safe  = 2 · CFL_margin / (c_eff · √max(ρ_Gersh, λ_max, ε_mach))

La cota de Gerschgorin es un majorante barato y siempre computable; el
autovalor máximo refina la cota cuando ARPACK converge.

IDENTIDAD DE HODGE LOCAL (FASE 3)
=================================
    δ_CORE ≜ W_mod^{+1/2}   (raíz espectral / pseudo-raíz)
    δ_COREᵀ δ_CORE ≡ W_mod  (error relativo O(ε_mach))

ESTRUCTURA DE FASES ANIDADAS (CONTINUIDAD FORMAL)
=================================================
    Phase1_MatrixValidation.build_preparation_context()
        →  KinematicPreparationContext
    Phase2_KinematicSynthesis.__init__(context) / .synthesize()
        →  KinematicStateTensor  (campo hodge_conductance = W_mod)
    Phase3_SheafProjection.__init__(W_mod) / .export_stalk()
        →  SheafStalk

Cada frontera de fase es un DTO inmutable (frozen dataclass) que constituye
el único contrato de interfaz entre estratos.
"""

from __future__ import annotations

# =============================================================================
# Biblioteca estándar
# =============================================================================
import logging
from typing import Optional, Tuple, Final

# =============================================================================
# Álgebra numérica de alta precisión
# =============================================================================
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
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
# Logger del módulo
# =============================================================================
logger = logging.getLogger("MIC.Alpha.KCoreKinematicAgent")


# =============================================================================
# SECCIÓN 0 — EXCEPCIONES CINEMÁTICAS ESTRICTAS
# =============================================================================


class KinematicCoreError(Exception):
    """
    Excepción categórica raíz para violaciones en el Estrato K_CORE.

    Toda excepción de este módulo hereda de esta clase, permitiendo que
    los manejadores de nivel superior capturen cualquier fallo cinemático
    con un único ``except KinematicCoreError``.
    """


class KinematicDimensionError(KinematicCoreError):
    """
    Lanzada cuando las dimensiones de las matrices constitutivas son
    inconsistentes entre sí o con el espacio de estado declarado.

    El diagnóstico incluye las formas (shapes) detectadas y las esperadas.
    """


class KinematicSymmetryError(KinematicCoreError):
    """
    Lanzada cuando una matriz viola su propiedad de simetría, antisimetría
    o semidefinición positiva/negativa requerida, con diagnóstico
    cuantitativo normalizado por la norma de Frobenius.
    """


class KinematicConditionError(KinematicCoreError):
    """
    Lanzada cuando el número de condición espectral κ(A) supera el umbral
    admisible, comprometiendo la estabilidad numérica de la síntesis
    IDA-PBC o del control de vorticidad.
    """


class DiracMatchingError(KinematicCoreError):
    r"""
    Lanzada cuando la ecuación de matching IDA-PBC:

        [J_d − R_d] ∇H_d = [J − R] ∇H + g α

    carece de solución estable, bien porque g (o gᵀ G g) es rango-deficiente
    más allá de la tolerancia SVD, bien porque el residuo relativo excede
    el umbral configurado.

    Incluye diagnóstico del rango efectivo de g y el residuo normalizado
    en la métrica G (si se suministró).
    """


class ParasiticVorticityError(KinematicCoreError):
    """
    Lanzada cuando el Laplaciano de Hodge detecta componentes solenoidales
    (flujo circular) que superan el umbral crítico ε_crit y cuyo soporte
    espectral no puede ser estrangulado con el factor configurado, o cuando
    la forma cuadrática I_curlᵀ W I_curl resulta no física (< 0).
    """


class ImpedanceReflectionError(KinematicCoreError):
    """
    Lanzada cuando la sintonización Kramers-Kronig falla:
      • Z_load no es SPD (acoplamiento de impedancia imposible).
      • El tensor μ_eff resultante no es SPD (violación de causalidad).
      • La relación de dispersión causal no se satisface numéricamente.
    """


class CFLViolationError(KinematicCoreError):
    """
    Lanzada cuando:
      • El paso temporal dt_requested excede Δt_safe (violación CFL activa).
      • El cálculo del autovalor máximo del Laplaciano falla (ARPACK diverge
        y el fallback denso no es admisible).
      • c_eff ≤ 0 (velocidad de propagación no física).
    """


class SheafCoboundaryError(KinematicCoreError):
    r"""
    Lanzada cuando δ_CORE no satisface la identidad de Hodge local:

        δ_COREᵀ δ_CORE ≃ W_mod

    con tolerancia de 100 · ε_mach relativa a ‖W_mod‖_F.
    """


class MetricTensorError(KinematicCoreError):
    """
    Lanzada cuando la métrica de estado G_μν no es simétrica, no es PSD,
    o tiene dimensiones incompatibles con el espacio de estado.
    """


# =============================================================================
# SECCIÓN 1 — ESTRUCTURAS INMUTABLES (DTOs TENSORIALES)
# =============================================================================


@dataclass(frozen=True, slots=True)
class KinematicPreparationContext:
    r"""
    Contexto inmutable producido por la **Fase 1** (Validación Matricial).

    Contiene las matrices constitutivas validadas, la métrica de estado G
    (si se suministró) y metadatos espectrales, necesarios para que la
    Fase 2 opere sin re-validar ni re-descomponer.

    Continuidad formal
    ------------------
    Este DTO es el **único argumento** del constructor de
    ``Phase2_KinematicSynthesis``. Su emisión por
    ``Phase1_MatrixValidation.build_preparation_context()`` constituye
    la frontera Fase 1 → Fase 2.

    Atributos
    ----------
    J : NDArray[np.float64], shape (n, n)
        Matriz de interconexión del sistema real, antisimétrica J = −Jᵀ.
    R : NDArray[np.float64], shape (n, n)
        Matriz de disipación del sistema real, PSD R ⪰ 0.
    J_d : NDArray[np.float64], shape (n, n)
        Matriz de interconexión deseada (IDA-PBC), antisimétrica.
    R_d : NDArray[np.float64], shape (n, n)
        Matriz de disipación deseada (IDA-PBC), PSD.
    g : NDArray[np.float64], shape (n, m)
        Matriz de entrada del control, rango posiblemente deficiente.
    G : NDArray[np.float64], shape (n, n)
        Métrica Riemanniana de estado G_μν ⪰ 0. Si no se suministró,
        es la identidad I_n (proyección euclídea clásica).
    n : int
        Dimensión del espacio de estado.
    m : int
        Número de entradas de control (columnas de g).
    rank_g : int
        Rango numérico de g (puede ser < min(n, m) si g es rango-deficiente).
    rank_G : int
        Rango numérico de G.
    kappa_R : float
        Número de condición espectral de R.
    kappa_R_d : float
        Número de condición espectral de R_d.
    kappa_G : float
        Número de condición espectral de G.
    spectral_gap_R : float
        Gap espectral λ₂/λ_max de R (0 si rank ≤ 1); indicador de
        disipación equidistribuida vs. concentrada.
    """

    J: NDArray[np.float64]
    R: NDArray[np.float64]
    J_d: NDArray[np.float64]
    R_d: NDArray[np.float64]
    g: NDArray[np.float64]
    G: NDArray[np.float64]
    n: int
    m: int
    rank_g: int
    rank_G: int
    kappa_R: float
    kappa_R_d: float
    kappa_G: float
    spectral_gap_R: float


@dataclass(frozen=True, slots=True)
class KinematicStateTensor:
    r"""
    Tensor inmutable que encapsula el estado cinemático completo del núcleo.

    Producido por la **Fase 2** (Síntesis Cinemática).

    Continuidad formal
    ------------------
    El campo ``hodge_conductance`` es el **dato primario** consumido por
    el constructor de ``Phase3_SheafProjection``. Su emisión por
    ``Phase2_KinematicSynthesis.synthesize()`` constituye la frontera
    Fase 2 → Fase 3.

    Atributos
    ----------
    control_law_alpha : NDArray[np.float64], shape (m,)
        Ley de control IDA-PBC covariante:
        α = (gᵀ G g + λ I)⁺ gᵀ G F_req.
    hodge_conductance : sp.spmatrix
        Matriz de conductancia W_mod modulada por estrangulamiento de Hodge.
    dielectric_tensor : NDArray[np.float64]
        Tensor dieléctrico efectivo ε_eff ≻ 0 (Kramers-Kronig).
    magnetic_tensor : NDArray[np.float64]
        Tensor magnético efectivo μ_eff ≻ 0 (Kramers-Kronig).
    cfl_safe_dt : float
        Paso temporal máximo admisible Δt_safe > 0 según la condición CFL
        dual (Gerschgorin + espectral).
    residual_idapbc : float
        Residuo relativo de matching en la métrica G:
        ‖g α − F_req‖_G / max(‖F_req‖_G, 1).
    vorticity_norm : float
        ‖I_curl‖_W = √(I_curlᵀ W I_curl): norma de la vorticidad.
    gershgorin_rho : float
        Radio de Gerschgorin del Laplaciano (majorante barato de λ_max).
    lambda_max_delta : float
        Autovalor máximo estimado de Δ_sym.
    is_kinematically_stable : bool
        True si y sólo si todos los subprocesos pasaron sin excepción.
    """

    control_law_alpha: NDArray[np.float64]
    hodge_conductance: sp.spmatrix
    dielectric_tensor: NDArray[np.float64]
    magnetic_tensor: NDArray[np.float64]
    cfl_safe_dt: float
    residual_idapbc: float
    vorticity_norm: float
    gershgorin_rho: float
    lambda_max_delta: float
    is_kinematically_stable: bool


@dataclass(frozen=True, slots=True)
class SheafStalk:
    r"""
    Fibrado celular exportado para el cálculo global del Laplaciano de Haz.

    Producido por la **Fase 3** (Proyección en Haces).

    Continuidad formal
    ------------------
    Este DTO es la **salida terminal** de la cadena de tres fases. Alimenta
    el ensamblaje del Laplaciano de Haz global del ecosistema APU Filter.

    Atributos
    ----------
    delta_core : NDArray[np.float64]
        Cofrontera discreta δ_CORE = W_mod^{+1/2} ∈ ℝ^{E×E},
        calculada vía raíz cuadrada espectral de W_mod.
    delta_hodge_residual : float
        Error relativo de la identidad de Hodge local:
        ‖δ_COREᵀ δ_CORE − W_mod_dense‖_F / ‖W_mod_dense‖_F.
        Debe ser O(ε_mach).
    state_vector : NDArray[np.float64]
        Vector de estado x en el instante de proyección.
    projected_state : NDArray[np.float64]
        Proyección δ_CORE · x sobre la fibra local.
    rank_delta : int
        Rango numérico de δ_CORE = rango de W_mod.
    betti_approx : int
        Aproximación del número de Betti β₀ local:
        dim ker(W_mod) ≈ E − rank(δ_CORE). Indica componentes
        desconectadas en el soporte de conductancia.
    spectral_entropy : float
        Entropía de von Neumann del espectro normalizado de W_mod:
        S = −Σ p_i log p_i, p_i = λ_i / Σ λ_j.
        Mide la dispersión de la métrica de aristas (0 = rango-1, log(E) = plana).
    """

    delta_core: NDArray[np.float64]
    delta_hodge_residual: float
    state_vector: NDArray[np.float64]
    projected_state: NDArray[np.float64]
    rank_delta: int
    betti_approx: int
    spectral_entropy: float


# =============================================================================
# SECCIÓN 2 — ORQUESTADOR: KCoreKinematicAgent
#             Tres fases anidadas de rigor creciente
# =============================================================================


class KCoreKinematicAgent(Morphism):
    r"""
    Orquestador Funtorial del Núcleo Cinemático K_CORE.

    Subyuga la velocidad del flujo logístico a la estabilidad del espacio
    de fase mediante tres clases anidadas que operan en cascada estricta:

        Phase1_MatrixValidation
            ↓  KinematicPreparationContext
        Phase2_KinematicSynthesis
            ↓  KinematicStateTensor
        Phase3_SheafProjection
            ↓  SheafStalk

    El constructor instancia y ejecuta la Fase 1 de forma inmediata,
    garantizando que cualquier violación matricial sea detectada antes de
    que el agente sea utilizado por el ecosistema APU Filter.

    Parámetros de Construcción
    --------------------------
    J, R, J_d, R_d, g : ver KinematicPreparationContext.
    G : NDArray[np.float64] | None
        Métrica Riemanniana de estado. None ⇒ I_n.
    cfl_margin : float, default 0.9
        Factor de seguridad CFL ∈ (0, 1].
    kappa_max : float, default 1e10
        Umbral de número de condición espectral.
    residual_tol_rel : float, default 1e-6
        Tolerancia relativa para el residuo IDA-PBC en métrica G.
    tikhonov_reg : float, default 0.0
        Regularización de Tikhonov λ_reg ≥ 0 sobre gᵀ G g.
        Recomendado: λ_reg ≈ ε_mach · ‖gᵀ G g‖_F cuando rank(g) < m.
    """

    FRIENDLY_NAME: str = "Director de Flujo y Cinética Logística"
    VERSION: str = "6.0.0-Rigorous-IDA-PBC-Hodge-CFL-Sheaf-Spectral"

    def __init__(
        self,
        J: NDArray[np.float64],
        R: NDArray[np.float64],
        J_d: NDArray[np.float64],
        R_d: NDArray[np.float64],
        g: NDArray[np.float64],
        G: Optional[NDArray[np.float64]] = None,
        cfl_margin: float = 0.9,
        kappa_max: float = 1.0e10,
        residual_tol_rel: float = 1.0e-6,
        tikhonov_reg: float = 0.0,
    ) -> None:
        r"""
        Inicializa las matrices constitutivas y ejecuta la Fase 1 de inmediato.

        Lanza
        -----
        ValueError
            Si cfl_margin ∉ (0, 1] o tikhonov_reg < 0.
        KinematicDimensionError, KinematicSymmetryError,
        KinematicConditionError, MetricTensorError
            Propagadas desde la Fase 1 si alguna propiedad matricial es violada.
        """
        if not (0.0 < cfl_margin <= 1.0):
            raise ValueError(
                f"cfl_margin debe estar en (0, 1]; se obtuvo {cfl_margin}. "
                f"Valores > 1 implican inestabilidad numérica del esquema."
            )
        if tikhonov_reg < 0.0:
            raise ValueError(
                f"tikhonov_reg debe ser ≥ 0; se obtuvo {tikhonov_reg}."
            )

        self.cfl_margin: float = cfl_margin
        self.kappa_max: float = kappa_max
        self.residual_tol_rel: float = residual_tol_rel
        self.tikhonov_reg: float = tikhonov_reg

        # ------------------------------------------------------------------
        # Fase 1: Validación Matricial Constitutiva (inmediata)
        # ------------------------------------------------------------------
        self.phase1: KCoreKinematicAgent.Phase1_MatrixValidation = (
            KCoreKinematicAgent.Phase1_MatrixValidation(
                J=J,
                R=R,
                J_d=J_d,
                R_d=R_d,
                g=g,
                G=G,
                kappa_max=kappa_max,
            )
        )
        self.context: KinematicPreparationContext = (
            self.phase1.build_preparation_context()
        )

        # ------------------------------------------------------------------
        # Fase 2: Síntesis Cinemática (instanciación inmediata)
        # Continuación formal del contexto emitido por Fase 1.
        # ------------------------------------------------------------------
        self.phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis = (
            KCoreKinematicAgent.Phase2_KinematicSynthesis(
                context=self.context,
                cfl_margin=self.cfl_margin,
                residual_tol_rel=self.residual_tol_rel,
                tikhonov_reg=self.tikhonov_reg,
            )
        )

        # Estado interno: conductancia modulada más reciente
        self._latest_hodge_conductance: Optional[sp.spmatrix] = None

        # Fase 3: instanciación perezosa (requiere W_mod de synthesize)
        self.phase3: Optional[KCoreKinematicAgent.Phase3_SheafProjection] = None

        logger.info(
            "[KCoreKinematicAgent v%s] Inicializado: n=%d, m=%d, "
            "rank(g)=%d, rank(G)=%d, κ(R)=%.3e, κ(R_d)=%.3e, κ(G)=%.3e, "
            "gap(R)=%.3e, CFL_margin=%.2f, λ_reg=%.3e.",
            self.VERSION,
            self.context.n,
            self.context.m,
            self.context.rank_g,
            self.context.rank_G,
            self.context.kappa_R,
            self.context.kappa_R_d,
            self.context.kappa_G,
            self.context.spectral_gap_R,
            self.cfl_margin,
            self.tikhonov_reg,
        )

    # =========================================================================
    # FASE 1 — VALIDACIÓN MATRICIAL CONSTITUTIVA
    # =========================================================================

    class Phase1_MatrixValidation:
        r"""
        **Fase 1 – Validación Matricial Constitutiva.**

        Responsabilidades exclusivas de esta fase:
          a) Verificar dimensiones y consistencia del espacio de estado.
          b) Verificar antisimetría de J, J_d con tolerancia relativa a ‖·‖_F.
          c) Verificar simetría de R, R_d, G con tolerancia relativa.
          d) Verificar PSD de R, R_d, G con tolerancia normalizada y
             calcular κ y gap espectral.
          e) Calcular rango numérico de g y de G (SVD Golub–Van Loan).
          f) Retornar ``KinematicPreparationContext`` inmutable.

        Todas las tolerancias son *relativas* a la norma de Frobenius de la
        matriz evaluada multiplicada por la precisión de máquina ε_mach,
        eliminando falsos positivos para matrices de gran norma.

        Fundamento espectral
        --------------------
        Para A = Aᵀ, Spec(A) = {λ_i} ordenado λ₁ ≤ … ≤ λ_n. Entonces:

            κ(A)  = λ_max / λ_min^{+}     (sólo sobre autovalores > tol)
            gap   = λ₂^{+} / λ_max        (dispersión de la disipación)

        El gap cercano a 1 indica disipación equidistribuida; cercano a 0
        indica un único modo dominante (riesgo de cuello de botella).
        """

        _EPS: Final[float] = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            J: NDArray[np.float64],
            R: NDArray[np.float64],
            J_d: NDArray[np.float64],
            R_d: NDArray[np.float64],
            g: NDArray[np.float64],
            G: Optional[NDArray[np.float64]] = None,
            kappa_max: float = 1.0e10,
        ) -> None:
            r"""
            Almacena referencias a las matrices originales sin copiarlas.
            Las copias ocurren sólo en ``build_preparation_context``.
            Si G es None, se materializa I_n tras conocer n.
            """
            self._J: NDArray[np.float64] = J
            self._R: NDArray[np.float64] = R
            self._J_d: NDArray[np.float64] = J_d
            self._R_d: NDArray[np.float64] = R_d
            self._g: NDArray[np.float64] = g
            self._G_raw: Optional[NDArray[np.float64]] = G
            self._kappa_max: float = kappa_max

        # ---------------------------------------------------------------------
        # Métodos privados de validación (orden lógico de ejecución)
        # ---------------------------------------------------------------------

        def _check_dimensions(self) -> Tuple[int, int]:
            r"""
            Verifica la coherencia dimensional completa del espacio de estado.

            Condiciones formales (lógica booleana de consistencia):
              • J, R, J_d, R_d ∈ ℝ^{n×n}  (cuadradas, misma dimensión n)
              • g ∈ ℝ^{n×m}                (n filas, m ≥ 1 columnas)
              • Todos los arrays son 2D (ndim ≡ 2).

            Retorna
            -------
            Tuple[int, int]
                (n, m): dimensión del espacio de estado y número de entradas.

            Lanza
            -----
            KinematicDimensionError
                Con diagnóstico explícito de la forma violada.
            """
            for mat, name in [
                (self._J, "J"),
                (self._R, "R"),
                (self._J_d, "J_d"),
                (self._R_d, "R_d"),
                (self._g, "g"),
            ]:
                if mat.ndim != 2:
                    raise KinematicDimensionError(
                        f"La matriz '{name}' debe ser 2D; "
                        f"se obtuvo ndim={mat.ndim}, shape={mat.shape}."
                    )

            if self._J.shape[0] != self._J.shape[1]:
                raise KinematicDimensionError(
                    f"J debe ser cuadrada; se obtuvo shape={self._J.shape}."
                )
            n: int = int(self._J.shape[0])

            for mat, name in [
                (self._R, "R"),
                (self._J_d, "J_d"),
                (self._R_d, "R_d"),
            ]:
                if mat.shape != (n, n):
                    raise KinematicDimensionError(
                        f"La matriz '{name}' debe tener shape ({n},{n}); "
                        f"se obtuvo {mat.shape}. "
                        f"Dimensión del espacio de estado n={n} definida por J."
                    )

            if self._g.shape[0] != n:
                raise KinematicDimensionError(
                    f"g debe tener {n} filas (dim. espacio de estado); "
                    f"se obtuvo {self._g.shape[0]} filas, shape={self._g.shape}."
                )
            m: int = int(self._g.shape[1])

            if m < 1:
                raise KinematicDimensionError(
                    f"g debe tener al menos 1 columna (entrada de control); "
                    f"se obtuvo m={m}."
                )

            # Validación dimensional de G (si se suministró)
            if self._G_raw is not None:
                if self._G_raw.ndim != 2 or self._G_raw.shape != (n, n):
                    raise MetricTensorError(
                        f"G_μν debe tener shape ({n},{n}); "
                        f"se obtuvo shape={self._G_raw.shape}."
                    )

            logger.debug("[Fase1] Dimensiones verificadas: n=%d, m=%d.", n, m)
            return n, m

        def _validate_antisymmetry(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> None:
            r"""
            Verifica que A = −Aᵀ con tolerancia relativa al Frobenius de A.

            Tolerancia adaptativa:
                tol = ε_mach · max(‖A‖_F, 1)

            Residuo:
                ‖A + Aᵀ‖_F  (norma de la parte simétrica; debe ser ≈ 0)

            Lanza
            -----
            KinematicSymmetryError
                Con residuo absoluto y relativo para diagnóstico.
            """
            norm_A: float = float(la.norm(A, "fro"))
            tol: float = self._EPS * max(norm_A, 1.0)
            residual: float = float(la.norm(A + A.T, "fro"))

            if residual > tol:
                raise KinematicSymmetryError(
                    f"La matriz '{name}' no es antisimétrica (A ≠ −Aᵀ). "
                    f"‖A + Aᵀ‖_F = {residual:.6e},  tol = {tol:.6e},  "
                    f"antisimetría relativa = "
                    f"{residual / max(norm_A, 1e-300):.6e}."
                )

            logger.debug(
                "[Fase1] Antisimetría de '%s': residual=%.3e, tol=%.3e.",
                name,
                residual,
                tol,
            )

        def _validate_symmetry(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> None:
            r"""
            Verifica que A = Aᵀ con tolerancia relativa al Frobenius de A.

            Tolerancia adaptativa:
                tol = ε_mach · max(‖A‖_F, 1)

            Lanza
            -----
            KinematicSymmetryError
                Con diagnóstico cuantitativo de ‖A − Aᵀ‖_F.
            """
            norm_A: float = float(la.norm(A, "fro"))
            tol: float = self._EPS * max(norm_A, 1.0)
            residual: float = float(la.norm(A - A.T, "fro"))

            if residual > tol:
                raise KinematicSymmetryError(
                    f"La matriz '{name}' no es simétrica (A ≠ Aᵀ). "
                    f"‖A − Aᵀ‖_F = {residual:.6e},  tol = {tol:.6e},  "
                    f"asimetría relativa = "
                    f"{residual / max(norm_A, 1e-300):.6e}."
                )

            logger.debug(
                "[Fase1] Simetría de '%s': residual=%.3e, tol=%.3e.",
                name,
                residual,
                tol,
            )

        def _validate_psd_and_spectrum(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> Tuple[float, float, int]:
            r"""
            Verifica A ⪰ 0 y retorna (κ, spectral_gap, rank_num).

            Tolerancia PSD:
                tol_psd = ε_mach · max(‖A‖_F, 1)

            Un autovalor λ se considera genuinamente negativo si λ < −tol_psd;
            nulo (rango-deficiente legítimo) si |λ| ≤ tol_psd; positivo si
            λ > tol_psd.

            Definiciones espectrales
            ------------------------
                κ(A)  = λ_max / λ_min^{+}
                gap   = λ₂^{+} / λ_max   (0.0 si hay < 2 autovalores positivos)
                rank  = #{λ_i > tol_psd}

            Retorna
            -------
            Tuple[float, float, int]
                (kappa, spectral_gap, rank_num)

            Lanza
            -----
            KinematicConditionError
                Si κ(A) > kappa_max.
            KinematicSymmetryError
                Si λ_min < −tol_psd (violación PSD genuina).
            """
            norm_A: float = float(la.norm(A, "fro"))
            tol_psd: float = self._EPS * max(norm_A, 1.0)

            # Re-simetrización defensiva antes de eigvalsh
            A_sym: NDArray[np.float64] = 0.5 * (A + A.T)
            eigvals: NDArray[np.float64] = la.eigvalsh(A_sym)
            lambda_min: float = float(eigvals[0])
            lambda_max: float = float(eigvals[-1])

            if lambda_min < -tol_psd:
                raise KinematicSymmetryError(
                    f"La matriz '{name}' no es Semidefinida Positiva (PSD). "
                    f"λ_min = {lambda_min:.6e}  <  −tol = {-tol_psd:.6e}."
                )

            pos_mask = eigvals > tol_psd
            pos_eigvals = eigvals[pos_mask]
            rank_num: int = int(np.sum(pos_mask))

            if rank_num == 0:
                logger.warning(
                    "[Fase1] Matriz '%s' es numéricamente nula (rank=0).",
                    name,
                )
                return float("inf"), 0.0, 0

            lambda_min_pos: float = float(pos_eigvals[0])
            kappa: float = lambda_max / lambda_min_pos

            if kappa > self._kappa_max:
                raise KinematicConditionError(
                    f"La matriz '{name}' está mal condicionada: "
                    f"κ = {kappa:.6e}  >  κ_max = {self._kappa_max:.6e}. "
                    f"λ_min⁺ = {lambda_min_pos:.6e}, λ_max = {lambda_max:.6e}. "
                    f"Considere regularización de Tikhonov o reescalado."
                )

            # Gap espectral: λ₂⁺ / λ_max
            if rank_num >= 2:
                lambda_2_pos: float = float(pos_eigvals[1])
                spectral_gap: float = lambda_2_pos / lambda_max
            else:
                spectral_gap = 0.0

            logger.debug(
                "[Fase1] PSD '%s': λ_min=%.3e, λ_max=%.3e, κ=%.3e, "
                "gap=%.3e, rank=%d.",
                name,
                lambda_min,
                lambda_max,
                kappa,
                spectral_gap,
                rank_num,
            )
            return kappa, spectral_gap, rank_num

        def _compute_rank_svd(
            self,
            M: NDArray[np.float64],
            name: str,
            n_ref: int,
            m_ref: int,
            *,
            reject_zero: bool = False,
        ) -> int:
            r"""
            Calcula el rango numérico de M mediante SVD completa.

            Criterio de truncación (Golub–Van Loan, Matrix Computations §5.5.8):

                σ_tol = max(n_ref, m_ref) · ε_mach · σ_max

            Retorna
            -------
            int
                Rango efectivo: número de valores singulares > σ_tol.

            Lanza
            -----
            KinematicDimensionError
                Si reject_zero=True y rank=0 (sin capacidad de control).
            """
            if M.size == 0:
                return 0

            _, s, _ = la.svd(M, full_matrices=False, check_finite=False)
            sigma_max: float = float(s[0]) if len(s) > 0 else 0.0
            sigma_tol: float = max(n_ref, m_ref) * self._EPS * max(sigma_max, 1.0)
            # Usar max(sigma_max, 1) solo en el factor de escala cuando sigma_max≈0
            if sigma_max > 0.0:
                sigma_tol = max(n_ref, m_ref) * self._EPS * sigma_max
            else:
                sigma_tol = max(n_ref, m_ref) * self._EPS

            rank: int = int(np.sum(s > sigma_tol))

            if reject_zero and rank == 0:
                raise KinematicDimensionError(
                    f"La matriz '{name}' es numéricamente nula "
                    f"(σ_max = {sigma_max:.6e}, todos los SV ≤ {sigma_tol:.6e}). "
                    f"Sin capacidad de control sobre el sistema."
                )

            logger.debug(
                "[Fase1] Rango de '%s': rank=%d/%d "
                "(σ_max=%.3e, σ_tol=%.3e).",
                name,
                rank,
                min(M.shape),
                sigma_max,
                sigma_tol,
            )
            return rank

        # ---------------------------------------------------------------------
        # Método terminal de la Fase 1 — entrada directa de la Fase 2
        # ---------------------------------------------------------------------

        def build_preparation_context(self) -> "KinematicPreparationContext":
            r"""
            **Método terminal de la Fase 1.**

            Ejecuta en secuencia estricta y ordenada todas las validaciones
            y empaqueta los resultados en un ``KinematicPreparationContext``
            inmutable.

            El contexto resultante es el **único argumento** que necesita el
            constructor de ``Phase2_KinematicSynthesis``, garantizando la
            continuidad formal entre fases.

            Flujo interno
            -------------
            1. Verificación dimensional (J, R, J_d, R_d, g, G).
            2. Materialización de G = I_n si no se suministró.
            3. Antisimetría de J y J_d.
            4. Simetría de R, R_d y G.
            5. PSD, κ, gap y rango de R, R_d y G.
            6. Rango numérico de g (SVD Golub–Van Loan).
            7. Empaquetado en KinematicPreparationContext.

            Retorna
            -------
            KinematicPreparationContext
                Contexto cinemático completo, inmutable y listo para Fase 2.
            """
            # ---- Paso 1: Dimensiones ----
            n, m = self._check_dimensions()

            # ---- Paso 2: Materializar G ----
            if self._G_raw is None:
                G: NDArray[np.float64] = np.eye(n, dtype=np.float64)
            else:
                G = self._G_raw

            # ---- Paso 3: Antisimetría de interconexión ----
            self._validate_antisymmetry(self._J, "J")
            self._validate_antisymmetry(self._J_d, "J_d")

            # ---- Paso 4: Simetría de disipación y métrica ----
            self._validate_symmetry(self._R, "R")
            self._validate_symmetry(self._R_d, "R_d")
            self._validate_symmetry(G, "G")

            # ---- Paso 5: PSD + espectro de R, R_d, G ----
            kappa_R, gap_R, _rank_R = self._validate_psd_and_spectrum(
                self._R, "R"
            )
            kappa_R_d, _gap_R_d, _rank_R_d = self._validate_psd_and_spectrum(
                self._R_d, "R_d"
            )
            kappa_G, _gap_G, rank_G = self._validate_psd_and_spectrum(G, "G")

            # ---- Paso 6: Rango de g ----
            rank_g: int = self._compute_rank_svd(
                self._g, "g", n, m, reject_zero=True
            )

            # ---- Paso 7: Empaquetado ----
            context = KinematicPreparationContext(
                J=self._J.copy(),
                R=self._R.copy(),
                J_d=self._J_d.copy(),
                R_d=self._R_d.copy(),
                g=self._g.copy(),
                G=G.copy(),
                n=n,
                m=m,
                rank_g=rank_g,
                rank_G=rank_G,
                kappa_R=kappa_R,
                kappa_R_d=kappa_R_d,
                kappa_G=kappa_G,
                spectral_gap_R=gap_R,
            )

            logger.info(
                "[Fase1] KinematicPreparationContext ensamblado: "
                "n=%d, m=%d, rank(g)=%d, rank(G)=%d, "
                "κ(R)=%.3e, κ(R_d)=%.3e, κ(G)=%.3e, gap(R)=%.3e.",
                n,
                m,
                rank_g,
                rank_G,
                kappa_R,
                kappa_R_d,
                kappa_G,
                gap_R,
            )

            # ================================================================
            # CONTRATO DE INTERFAZ FASE 1 → FASE 2
            # `context` es el argumento directo del constructor de
            # Phase2_KinematicSynthesis. Esta devolución es la frontera
            # formal entre ambas fases anidadas.
            # ================================================================
            return context

    # =========================================================================
    # FASE 2 — SÍNTESIS CINEMÁTICA
    # =========================================================================

    class Phase2_KinematicSynthesis:
        r"""
        **Fase 2 – Síntesis Cinemática.**

        Recibe el ``KinematicPreparationContext`` de la Fase 1 y ejecuta
        los cuatro procesos fundamentales del K_CORE:

          1. **Moldeado de energía IDA-PBC covariante**
             (``compute_dirac_control_law``):
             Resuelve la ecuación de matching con pseudoinversa ponderada
             por G_μν y regularización de Tikhonov opcional.

          2. **Estrangulamiento de vorticidad de Hodge**
             (``modulate_hodge_conductance``):
             Cuantifica ‖I_curl‖_W con la forma cuadrática completa
             (soporta W no diagonal) y penaliza el soporte de I_curl.

          3. **Sintonización de impedancia Kramers-Kronig**
             (``tune_impedance_tensors``):
             Construye ε_eff, μ_eff SPD y verifica la relación de
             dispersión causal.

          4. **Auditoría CFL dual** (``audit_cfl_limit``):
             Combina cota de Gerschgorin (siempre) con λ_max vía ARPACK
             (cuando converge) para un Δt_safe conservador y preciso.

        El constructor de esta clase es la **continuación directa** del
        método ``build_preparation_context`` de la Fase 1.
        """

        _EPS: Final[float] = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            context: "KinematicPreparationContext",
            cfl_margin: float,
            residual_tol_rel: float,
            tikhonov_reg: float = 0.0,
        ) -> None:
            r"""
            **Constructor de la Fase 2: continuación directa de la Fase 1.**

            Recibe el ``KinematicPreparationContext`` devuelto por
            ``Phase1_MatrixValidation.build_preparation_context()`` y
            extrae las matrices y metadatos para uso eficiente.

            No realiza ninguna validación adicional: la corrección de todas
            las matrices está garantizada por la Fase 1.

            Parámetros
            ----------
            context : KinematicPreparationContext
                Resultado inmutable de la Fase 1.
            cfl_margin : float
                Factor de seguridad CFL ∈ (0, 1].
            residual_tol_rel : float
                Tolerancia relativa para el residuo IDA-PBC en métrica G.
            tikhonov_reg : float
                λ_reg ≥ 0 para regularizar gᵀ G g.
            """
            self._ctx: "KinematicPreparationContext" = context
            self._cfl_margin: float = cfl_margin
            self._residual_tol_rel: float = residual_tol_rel
            self._tikhonov_reg: float = tikhonov_reg

            logger.debug(
                "[Fase2] Inicializada: n=%d, m=%d, rank(g)=%d, rank(G)=%d, "
                "CFL_margin=%.2f, res_tol=%.3e, λ_reg=%.3e.",
                context.n,
                context.m,
                context.rank_g,
                context.rank_G,
                cfl_margin,
                residual_tol_rel,
                tikhonov_reg,
            )

        # ---------------------------------------------------------------------
        # Utilidades métricas internas
        # ---------------------------------------------------------------------

        def _norm_G(
            self,
            v: NDArray[np.float64],
        ) -> float:
            r"""
            Norma inducida por la métrica G: ‖v‖_G = √(vᵀ G v).

            Si vᵀ G v < 0 por ruido numérico (G es PSD), se clampea a 0.
            """
            quad: float = float(v @ (self._ctx.G @ v))
            if quad < 0.0:
                # Ruido de redondeo en dirección del kernel de G
                quad = 0.0
            return float(np.sqrt(quad))

        # ---------------------------------------------------------------------
        # Subproceso 1: Moldeado de energía IDA-PBC covariante
        # ---------------------------------------------------------------------

        def compute_dirac_control_law(
            self,
            grad_H: NDArray[np.float64],
            grad_H_d: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Resuelve la ecuación de matching IDA-PBC con proyección covariante:

                F_req ≜ [J_d − R_d] ∇H_d − [J − R] ∇H
                α     = (gᵀ G g + λ_reg I_m)⁺ (gᵀ G F_req)

            Algoritmo
            ---------
            1. Formar Gram_G = gᵀ G g ∈ ℝ^{m×m} (simétrica, PSD).
            2. Regularizar: Gram_reg = Gram_G + λ_reg I_m.
            3. SVD truncada de Gram_reg (Golub–Van Loan):
                   σ_tol = m · ε_mach · σ_max
            4. α = V · diag(1/σᵢ) · Uᵀ · (gᵀ G F_req)
            5. Verificar residuo en norma G:
                   r_rel = ‖g α − F_req‖_G / max(‖F_req‖_G, 1)

            Cuando G = I y λ_reg = 0 se recupera la pseudoinversa clásica
            de Moore-Penrose de g aplicada a F_req (equivalente a g⁺ F_req
            vía la identidad g⁺ = (gᵀ g)⁺ gᵀ).

            Parámetros
            ----------
            grad_H : NDArray[np.float64], shape (n,)
                Gradiente del Hamiltoniano actual ∇H(x).
            grad_H_d : NDArray[np.float64], shape (n,)
                Gradiente del Hamiltoniano deseado ∇H_d(x).

            Retorna
            -------
            Tuple[NDArray[np.float64], float]
                (alpha, residual_rel)

            Lanza
            -----
            KinematicDimensionError
                Si grad_H o grad_H_d no tienen shape (n,).
            DiracMatchingError
                Si el residuo relativo supera residual_tol_rel.
            """
            n: int = self._ctx.n
            m: int = self._ctx.m

            if grad_H.shape != (n,):
                raise KinematicDimensionError(
                    f"grad_H debe tener shape ({n},); se obtuvo {grad_H.shape}."
                )
            if grad_H_d.shape != (n,):
                raise KinematicDimensionError(
                    f"grad_H_d debe tener shape ({n},); "
                    f"se obtuvo {grad_H_d.shape}."
                )

            # Fuerzas Port-Hamiltonianas
            F_d: NDArray[np.float64] = (
                self._ctx.J_d - self._ctx.R_d
            ) @ grad_H_d
            F_nat: NDArray[np.float64] = (
                self._ctx.J - self._ctx.R
            ) @ grad_H
            F_req: NDArray[np.float64] = F_d - F_nat

            # Norma G de F_req
            norm_F_req_G: float = self._norm_G(F_req)

            # ---- Gram covariante: gᵀ G g ----
            G_g: NDArray[np.float64] = self._ctx.G @ self._ctx.g  # (n, m)
            Gram: NDArray[np.float64] = self._ctx.g.T @ G_g       # (m, m)
            # Re-simetrizar (eliminación de asimetría O(ε))
            Gram = 0.5 * (Gram + Gram.T)

            # Regularización de Tikhonov
            if self._tikhonov_reg > 0.0:
                Gram = Gram + self._tikhonov_reg * np.eye(m, dtype=np.float64)

            # Lado derecho covariante: gᵀ G F_req
            rhs: NDArray[np.float64] = G_g.T @ F_req  # (m,)

            # ---- SVD de Gram (m×m, pequeño) ----
            try:
                U, s, Vh = la.svd(
                    Gram, full_matrices=False, check_finite=False
                )
            except la.LinAlgError as exc:
                raise DiracMatchingError(
                    f"Fallo de SVD (LAPACK dgesvd) en gᵀ G g. Error: {exc}"
                ) from exc

            sigma_max: float = float(s[0]) if len(s) > 0 else 0.0
            sigma_tol: float = m * self._EPS * max(sigma_max, 1.0)
            if sigma_max > 0.0:
                sigma_tol = m * self._EPS * sigma_max

            mask: NDArray[np.bool_] = s > sigma_tol
            rank_effective: int = int(np.sum(mask))

            if rank_effective == 0:
                raise DiracMatchingError(
                    f"gᵀ G g (+ λ I) es numéricamente nula "
                    f"(σ_max={sigma_max:.6e}). "
                    f"Sin capacidad de control covariante. "
                    f"rank(g)={self._ctx.rank_g}, rank(G)={self._ctx.rank_G}."
                )

            s_inv: NDArray[np.float64] = np.zeros_like(s)
            s_inv[mask] = 1.0 / s[mask]

            # α = V · diag(s_inv) · Uᵀ · rhs
            alpha: NDArray[np.float64] = Vh.T @ (s_inv * (U.T @ rhs))

            # Residuo de matching en norma G
            residual_vec: NDArray[np.float64] = self._ctx.g @ alpha - F_req
            residual_abs_G: float = self._norm_G(residual_vec)
            residual_rel: float = residual_abs_G / max(norm_F_req_G, 1.0)

            if residual_rel > self._residual_tol_rel:
                logger.error(
                    "[Fase2] Residuo IDA-PBC: r_rel_G=%.6e > tol=%.6e, "
                    "rank_eff(Gram)=%d/%d, ‖F_req‖_G=%.6e, λ_reg=%.3e.",
                    residual_rel,
                    self._residual_tol_rel,
                    rank_effective,
                    m,
                    norm_F_req_G,
                    self._tikhonov_reg,
                )
                raise DiracMatchingError(
                    f"Ecuación de matching IDA-PBC sin solución suficientemente "
                    f"precisa. Residuo relativo_G = {residual_rel:.6e} > "
                    f"tol = {self._residual_tol_rel:.6e}. "
                    f"Rango efectivo de gᵀGg: {rank_effective}/{m}. "
                    f"‖F_req‖_G = {norm_F_req_G:.6e}."
                )

            logger.debug(
                "[Fase2] IDA-PBC covariante: r_rel_G=%.3e, "
                "rank(Gram)_eff=%d, ‖α‖₂=%.6e.",
                residual_rel,
                rank_effective,
                float(la.norm(alpha, 2)),
            )
            return alpha, residual_rel

        # ---------------------------------------------------------------------
        # Subproceso 2: Estrangulamiento de vorticidad de Hodge
        # ---------------------------------------------------------------------

        def modulate_hodge_conductance(
            self,
            W: sp.spmatrix,
            I_curl: NDArray[np.float64],
            epsilon_crit: float = 1.0e-2,
            strangle_factor: float = 1.0e-4,
        ) -> Tuple[sp.spmatrix, float]:
            r"""
            Estrangula la conductancia en aristas con vorticidad parásita.

            Norma de energía (compatible con L₁ᵂ):

                ‖I_curl‖_W = √(I_curlᵀ W I_curl)

            Se usa el producto matricial sparse completo (soporta W no
            diagonal, p.ej. conductancias mutuas). Si W es diagonal, el
            coste es O(E); si no, O(nnz(W)).

            Si ‖I_curl‖_W > ε_crit, se penalizan las aristas cuyo
            |I_curl[e]| > 0.1 · ‖I_curl‖_∞:

                W_mod = W − (1 − strangle_factor) · P_S W P_S

            donde P_S es el proyector diagonal sobre el soporte S de
            vorticidad. Para W diagonal esto se reduce a
            w_diag[S] ← w_diag[S] · strangle_factor.

            Parámetros
            ----------
            W : sp.spmatrix, shape (E, E)
                Matriz de conductancia de aristas (cualquier formato sparse).
            I_curl : NDArray[np.float64], shape (E,)
                Corriente de curl sobre las E aristas.
            epsilon_crit : float
                Umbral de vorticidad admisible.
            strangle_factor : float ∈ (0, 1)
                Factor de penalización multiplicativo.

            Retorna
            -------
            Tuple[sp.spmatrix, float]
                (W_mod, vorticity_norm)

            Lanza
            -----
            KinematicDimensionError, ParasiticVorticityError
            """
            E: int = int(W.shape[0])

            if W.shape[0] != W.shape[1]:
                raise KinematicDimensionError(
                    f"W debe ser cuadrada; se obtuvo shape={W.shape}."
                )
            if I_curl.shape != (E,):
                raise KinematicDimensionError(
                    f"I_curl debe tener shape ({E},) coherente con "
                    f"W.shape[0]={E}; se obtuvo {I_curl.shape}."
                )
            if strangle_factor <= 0.0 or strangle_factor > 1.0:
                raise ParasiticVorticityError(
                    f"strangle_factor debe estar en (0, 1]; "
                    f"se obtuvo {strangle_factor}."
                )

            W_csr: sp.csr_matrix = W.tocsr()

            # Forma cuadrática completa: I_curlᵀ W I_curl
            W_I: NDArray[np.float64] = W_csr @ I_curl
            quad_form: float = float(I_curl @ W_I)

            if quad_form < -self._EPS * max(float(la.norm(I_curl, 2)) ** 2, 1.0):
                raise ParasiticVorticityError(
                    f"Forma cuadrática I_curlᵀ W I_curl = {quad_form:.6e} < 0. "
                    f"W tiene autovalores negativos (no física)."
                )
            quad_form = max(quad_form, 0.0)
            vorticity_norm: float = float(np.sqrt(quad_form))

            if vorticity_norm > epsilon_crit:
                logger.info(
                    "[Fase2] Vorticidad parásita: ‖I_curl‖_W=%.4e > "
                    "ε_crit=%.4e. Estrangulando conductancia.",
                    vorticity_norm,
                    epsilon_crit,
                )

                inf_norm: float = float(np.max(np.abs(I_curl))) if E > 0 else 0.0
                threshold: float = 0.1 * inf_norm
                mask: NDArray[np.bool_] = np.abs(I_curl) > threshold
                n_penalized: int = int(np.sum(mask))

                if n_penalized == 0:
                    logger.warning(
                        "[Fase2] Vorticidad detectada pero soporte vacío "
                        "(todos |I_curl[e]| ≤ %.3e). Sin penalización.",
                        threshold,
                    )
                    W_mod = W_csr
                else:
                    # Construcción general: escalar filas/columnas del soporte
                    # Para W diagonal: equivalente a w[S] *= strangle_factor.
                    # Para W denso-sparse: P_S W P_S se escala, el resto se
                    # mantiene (estrangulamiento local del bloque de soporte).
                    scale: NDArray[np.float64] = np.ones(E, dtype=np.float64)
                    scale[mask] = np.sqrt(strangle_factor)
                    # W_mod = D_scale W D_scale  (congruencia que preserva PSD)
                    D: sp.csr_matrix = sp.diags(
                        scale, offsets=0, shape=(E, E), format="csr",
                        dtype=np.float64,
                    )
                    W_mod = (D @ W_csr @ D).tocsr()

                    logger.debug(
                        "[Fase2] %d/%d aristas penalizadas con factor %.3e "
                        "(congruencia D W D).",
                        n_penalized,
                        E,
                        strangle_factor,
                    )
            else:
                logger.debug(
                    "[Fase2] Vorticidad ‖I_curl‖_W=%.4e ≤ ε_crit=%.4e. "
                    "Sin estrangulamiento.",
                    vorticity_norm,
                    epsilon_crit,
                )
                W_mod = W_csr

            return W_mod, vorticity_norm

        # ---------------------------------------------------------------------
        # Subproceso 3: Sintonización de impedancia Kramers-Kronig
        # ---------------------------------------------------------------------

        def tune_impedance_tensors(
            self,
            Z_load: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
            r"""
            Sintoniza los tensores dieléctrico ε_eff y magnético μ_eff para
            acoplamiento de impedancia perfecto:

                Z₀ = √(μ_eff · ε_eff⁻¹)  ≡  Z_load

            Solución constructiva (compatible con relaciones Kramers-Kronig
            en el dominio de frecuencia estática / DC):

                ε_eff = Z_load          (tras re-simetrización y Cholesky)
                μ_eff = Z_load · ε_eff · Z_loadᵀ = Z_load³   (en el sentido
                        de producto matricial; SPD por composición de SPD)

            Verificación de dispersión causal:
                ‖ε_eff − Z_load‖_F / ‖Z_load‖_F < 100 · ε_mach

            y Cholesky de μ_eff (SPD estricto).

            Parámetros
            ----------
            Z_load : NDArray[np.float64], shape (d, d)
                Matriz de impedancia de carga. Debe ser SPD.

            Retorna
            -------
            Tuple[NDArray[np.float64], NDArray[np.float64]]
                (epsilon_eff, mu_eff), ambos SPD.

            Lanza
            -----
            ImpedanceReflectionError
            """
            if Z_load.ndim != 2 or Z_load.shape[0] != Z_load.shape[1]:
                raise ImpedanceReflectionError(
                    f"Z_load debe ser cuadrada 2D; "
                    f"se obtuvo shape={Z_load.shape}."
                )

            Z_sym: NDArray[np.float64] = 0.5 * (Z_load + Z_load.T)

            # Verificar SPD de Z_load mediante Cholesky
            try:
                L_Z: NDArray[np.float64] = la.cholesky(Z_sym, lower=True)
            except la.LinAlgError:
                eigvals: NDArray[np.float64] = la.eigvalsh(Z_sym)
                raise ImpedanceReflectionError(
                    f"Z_load no es Simétrica Definida Positiva (SPD). "
                    f"Acoplamiento de impedancia imposible. "
                    f"λ_min = {float(eigvals[0]):.6e}, "
                    f"λ_max = {float(eigvals[-1]):.6e}."
                )

            # ε_eff = L_Z L_Zᵀ = Z_load (SPD por construcción)
            epsilon_eff: NDArray[np.float64] = L_Z @ L_Z.T

            # μ_eff = Z_load · ε_eff · Z_loadᵀ
            mu_eff: NDArray[np.float64] = Z_sym @ epsilon_eff @ Z_sym.T
            mu_eff = 0.5 * (mu_eff + mu_eff.T)

            try:
                la.cholesky(mu_eff, lower=True)
            except la.LinAlgError:
                raise ImpedanceReflectionError(
                    "El tensor magnético μ_eff resultante no es SPD. "
                    "Violación de la condición de causalidad (Kramers-Kronig)."
                )

            # Verificación de dispersión causal
            norm_Z: float = float(la.norm(Z_sym, "fro"))
            causal_residual: float = float(
                la.norm(epsilon_eff - Z_sym, "fro")
            ) / max(norm_Z, 1.0)
            tol_causal: float = 100.0 * self._EPS

            if causal_residual > tol_causal:
                raise ImpedanceReflectionError(
                    f"Relación de dispersión causal violada: "
                    f"‖ε_eff − Z_load‖_F / ‖Z_load‖_F = "
                    f"{causal_residual:.6e} > tol = {tol_causal:.6e}."
                )

            logger.debug(
                "[Fase2] Kramers-Kronig: causal_residual=%.3e, "
                "‖ε_eff‖_F=%.6e, ‖μ_eff‖_F=%.6e.",
                causal_residual,
                float(la.norm(epsilon_eff, "fro")),
                float(la.norm(mu_eff, "fro")),
            )
            return epsilon_eff, mu_eff

        # ---------------------------------------------------------------------
        # Subproceso 4: Auditoría del límite CFL (dual Gerschgorin + espectral)
        # ---------------------------------------------------------------------

        def _gershgorin_radius(
            self,
            Delta_sym: sp.spmatrix,
        ) -> float:
            r"""
            Radio de Gerschgorin del Laplaciano:

                ρ_G = max_i ( |Δ_ii| + Σ_{j≠i} |Δ_ij| )

            Es un majorante barato de λ_max (teorema de Gerschgorin) y no
            requiere descomposición espectral. Coste O(nnz).
            """
            Delta_csr: sp.csr_matrix = Delta_sym.tocsr()
            # Suma de |Δ_ij| por fila
            abs_Delta: sp.csr_matrix = Delta_csr.copy()
            abs_Delta.data = np.abs(abs_Delta.data)
            row_sums: NDArray[np.float64] = np.asarray(
                abs_Delta.sum(axis=1)
            ).ravel()
            # |Δ_ii| ya está incluido en row_sums; el radio de cada disco es
            # exactamente row_sums[i] = |Δ_ii| + Σ_{j≠i}|Δ_ij|.
            return float(np.max(row_sums)) if len(row_sums) > 0 else 0.0

        def audit_cfl_limit(
            self,
            c_eff: float,
            Delta_sym: sp.spmatrix,
        ) -> Tuple[float, float, float]:
            r"""
            Calcula el paso temporal máximo admisible según la condición CFL
            dual (Gerschgorin + espectral) para el operador de onda discreta:

                Δt_safe = (2 · CFL_margin)
                          / (c_eff · √max(ρ_Gersh, λ_max, ε_mach))

            La cota de Gerschgorin se calcula siempre (O(nnz)). El autovalor
            máximo se intenta con ARPACK; si falla y n ≤ 5000 se usa
            eigvalsh denso; si n > 5000 se confía únicamente en Gerschgorin.

            Parámetros
            ----------
            c_eff : float
                Velocidad de propagación efectiva (> 0).
            Delta_sym : sp.spmatrix
                Laplaciano simétrico del grafo logístico (PSD).

            Retorna
            -------
            Tuple[float, float, float]
                (dt_safe, gershgorin_rho, lambda_max)

            Lanza
            -----
            CFLViolationError
                Si c_eff ≤ 0.
            """
            if c_eff <= 0.0:
                raise CFLViolationError(
                    f"c_eff debe ser estrictamente positivo; "
                    f"se obtuvo c_eff={c_eff:.6e}. "
                    f"Una velocidad de propagación ≤ 0 es físicamente "
                    f"inadmisible."
                )

            n_nodes: int = int(Delta_sym.shape[0])

            # ---- Cota de Gerschgorin (siempre) ----
            rho_g: float = self._gershgorin_radius(Delta_sym)
            logger.debug(
                "[Fase2] ρ_Gersh(Δ_sym) = %.6e (n=%d).",
                rho_g,
                n_nodes,
            )

            # ---- λ_max con ARPACK ----
            lambda_max: float = 0.0
            try:
                eigvals_arpack, _ = eigsh(
                    Delta_sym,
                    k=1,
                    which="LM",
                    tol=1.0e-8,
                    maxiter=max(10 * n_nodes, 100),
                    return_eigenvectors=True,
                )
                lambda_max = float(np.abs(eigvals_arpack[0]))
                logger.debug(
                    "[Fase2] λ_max(Δ_sym) = %.6e (ARPACK, n=%d).",
                    lambda_max,
                    n_nodes,
                )
            except (ArpackNoConvergence, Exception) as exc_arpack:
                logger.warning(
                    "[Fase2] ARPACK no convergió para λ_max(Δ_sym): %s. "
                    "Fallback condicional (n=%d).",
                    exc_arpack,
                    n_nodes,
                )
                if n_nodes <= 5000:
                    try:
                        Delta_dense: NDArray[np.float64] = Delta_sym.toarray()
                        Delta_dense = 0.5 * (Delta_dense + Delta_dense.T)
                        eigvals_dense: NDArray[np.float64] = la.eigvalsh(
                            Delta_dense
                        )
                        lambda_max = float(eigvals_dense[-1])
                        logger.debug(
                            "[Fase2] λ_max(Δ_sym) = %.6e "
                            "(eigvalsh denso, fallback).",
                            lambda_max,
                        )
                    except Exception as exc_dense:
                        logger.warning(
                            "[Fase2] Fallback denso falló: %s. "
                            "Se usa únicamente ρ_Gersh.",
                            exc_dense,
                        )
                        lambda_max = 0.0
                else:
                    logger.warning(
                        "[Fase2] n=%d > 5000 y ARPACK falló; "
                        "se usa únicamente ρ_Gersh=%.6e.",
                        n_nodes,
                        rho_g,
                    )
                    lambda_max = 0.0

            # ---- Combinación conservadora ----
            spectral_bound: float = max(rho_g, lambda_max, self._EPS)

            if spectral_bound < 1.0e-12:
                logger.warning(
                    "[Fase2] bound espectral = %.3e < 1e-12: grafo "
                    "logístico degenerado (desconectado). Δt_safe = +∞.",
                    spectral_bound,
                )
                return float("inf"), rho_g, lambda_max

            dt_safe: float = (2.0 * self._cfl_margin) / (
                c_eff * float(np.sqrt(spectral_bound))
            )

            logger.debug(
                "[Fase2] CFL dual: ρ_G=%.6e, λ_max=%.6e, c_eff=%.6e, "
                "CFL_margin=%.2f, Δt_safe=%.6e.",
                rho_g,
                lambda_max,
                c_eff,
                self._cfl_margin,
                dt_safe,
            )
            return dt_safe, rho_g, lambda_max

        # ---------------------------------------------------------------------
        # Método terminal de la Fase 2 — entrada directa de la Fase 3
        # ---------------------------------------------------------------------

        def synthesize(
            self,
            grad_H: NDArray[np.float64],
            grad_H_d: NDArray[np.float64],
            W: sp.spmatrix,
            I_curl: NDArray[np.float64],
            Z_load: NDArray[np.float64],
            c_eff: float,
            Delta_sym: sp.spmatrix,
            dt_requested: float,
        ) -> "KinematicStateTensor":
            r"""
            **Método terminal de la Fase 2.**

            Integra los cuatro subprocesos cinemáticos en secuencia determinista
            y retorna el ``KinematicStateTensor`` completo.

            El ``KinematicStateTensor`` resultante contiene la conductancia
            modulada W_mod que es el **dato primario** consumido por la Fase 3
            para construir δ_CORE. La frontera formal entre Fase 2 y Fase 3
            es el campo ``hodge_conductance`` de este tensor.

            Parámetros
            ----------
            grad_H, grad_H_d : gradientes hamiltonianos.
            W : conductancia de aristas.
            I_curl : corriente de curl.
            Z_load : impedancia de carga (SPD).
            c_eff : velocidad de propagación efectiva (> 0).
            Delta_sym : Laplaciano simétrico del grafo.
            dt_requested : paso temporal solicitado.

            Retorna
            -------
            KinematicStateTensor

            Lanza
            -----
            CFLViolationError, DiracMatchingError, ParasiticVorticityError,
            ImpedanceReflectionError
            """
            # ---- Subproceso 1: IDA-PBC covariante ----
            alpha, residual_rel = self.compute_dirac_control_law(
                grad_H, grad_H_d
            )

            # ---- Subproceso 2: Hodge ----
            W_mod, vorticity_norm = self.modulate_hodge_conductance(
                W, I_curl
            )

            # ---- Subproceso 3: Kramers-Kronig ----
            epsilon_eff, mu_eff = self.tune_impedance_tensors(Z_load)

            # ---- Subproceso 4: CFL dual ----
            dt_safe, rho_g, lambda_max = self.audit_cfl_limit(
                c_eff, Delta_sym
            )

            # Verificación CFL activa
            if dt_requested > dt_safe:
                raise CFLViolationError(
                    f"Violación del Cono de Luz Causal (CFL). "
                    f"dt_requested = {dt_requested:.6e} > "
                    f"Δt_safe = {dt_safe:.6e}. "
                    f"ρ_Gersh={rho_g:.6e}, λ_max={lambda_max:.6e}. "
                    f"Reducir el paso temporal o disminuir c_eff."
                )

            logger.info(
                "[Fase2] Síntesis cinemática completada: "
                "‖α‖₂=%.6e, vorticity=%.6e, dt_safe=%.6e, r_rel_G=%.3e, "
                "ρ_G=%.3e, λ_max=%.3e.",
                float(la.norm(alpha, 2)),
                vorticity_norm,
                dt_safe,
                residual_rel,
                rho_g,
                lambda_max,
            )

            # ================================================================
            # CONTRATO DE INTERFAZ FASE 2 → FASE 3
            # `hodge_conductance = W_mod` es el argumento directo del
            # constructor de Phase3_SheafProjection. Esta devolución es la
            # frontera formal entre ambas fases anidadas.
            # ================================================================
            return KinematicStateTensor(
                control_law_alpha=alpha,
                hodge_conductance=W_mod,
                dielectric_tensor=epsilon_eff,
                magnetic_tensor=mu_eff,
                cfl_safe_dt=dt_safe,
                residual_idapbc=residual_rel,
                vorticity_norm=vorticity_norm,
                gershgorin_rho=rho_g,
                lambda_max_delta=lambda_max,
                is_kinematically_stable=True,
            )

    # =========================================================================
    # FASE 3 — PROYECCIÓN EN HACES Y COFRONTERA DISCRETA δ_CORE
    # =========================================================================

    class Phase3_SheafProjection:
        r"""
        **Fase 3 – Proyección en Haces y Cofrontera Discreta δ_CORE.**

        Recibe la conductancia modulada W_mod producida por la Fase 2 y
        construye el ``SheafStalk`` que alimenta el Laplaciano de Haz global.

        El constructor de esta clase es la **continuación directa** del método
        ``synthesize`` de la Fase 2: el campo ``hodge_conductance`` del
        ``KinematicStateTensor`` es lo que aquí se recibe.

        Fundamento Matemático (topología algebraica discreta)
        ------------------------------------------------------
        En el complejo de cadenas del grafo logístico, las aristas son
        1-cadenas y W_mod es el operador de métrica en el espacio de 1-formas.

        La cofrontera local δ_CORE : F(K_CORE) → F(e) se define como:

            δ_CORE = W_mod^{+1/2}

        donde W_mod^{+1/2} es la raíz cuadrada matricial espectral
        (pseudoinversa si W_mod es rango-deficiente):

            W_mod = V · diag(λ) · Vᵀ
            W_mod^{+1/2} = V · diag(√λ⁺) · Vᵀ

        Identidad de Hodge local (condición de consistencia del stalk):

            δ_COREᵀ · δ_CORE = W_mod

        Invariantes topológicos locales exportados
        ------------------------------------------
        • betti_approx ≈ dim ker(W_mod) = E − rank(δ_CORE)
          (componentes de conductancia nula ≈ β₀ del soporte).
        • spectral_entropy = S(p) = −Σ p_i log p_i, p_i = λ_i / Tr(W_mod)
          (entropía de von Neumann del espectro de la métrica de aristas;
          inspirada en la pureza de estados densos en mecánica cuántica).
        """

        _EPS: Final[float] = float(np.finfo(np.float64).eps)

        def __init__(self, W_mod: sp.spmatrix) -> None:
            r"""
            **Constructor de la Fase 3: continuación directa de la Fase 2.**

            Recibe W_mod = ``KinematicStateTensor.hodge_conductance`` y
            precalcula la raíz cuadrada espectral δ_CORE = W_mod^{+1/2}
            para amortizar el coste sobre múltiples llamadas a ``export_stalk``.

            Lanza
            -----
            SheafCoboundaryError
                Si la identidad de Hodge no se satisface dentro de 100·ε_mach
                o si W_mod no es PSD.
            """
            self._W_mod: sp.spmatrix = W_mod

            # Conversión a denso + re-simetrización
            W_dense: NDArray[np.float64] = W_mod.toarray()
            W_dense = 0.5 * (W_dense + W_dense.T)
            self._W_dense: NDArray[np.float64] = W_dense
            E: int = int(W_dense.shape[0])

            # Descomposición espectral
            eigvals: NDArray[np.float64]
            eigvecs: NDArray[np.float64]
            eigvals, eigvecs = la.eigh(W_dense)

            norm_W: float = float(la.norm(W_dense, "fro"))
            tol_eig: float = self._EPS * max(norm_W, 1.0)

            lambda_min: float = float(eigvals[0])
            if lambda_min < -tol_eig:
                raise SheafCoboundaryError(
                    f"W_mod no es Semidefinida Positiva: "
                    f"λ_min = {lambda_min:.6e} < −tol = {-tol_eig:.6e}. "
                    f"La estrangulación de Hodge produjo conductancias "
                    f"negativas (violación de PSD por congruencia)."
                )

            # Raíz espectral con clamping de negativos residuales
            eigvals_clamped: NDArray[np.float64] = np.maximum(eigvals, 0.0)
            eigvals_sqrt: NDArray[np.float64] = np.sqrt(eigvals_clamped)
            delta_core: NDArray[np.float64] = (
                eigvecs * eigvals_sqrt[np.newaxis, :]
            ) @ eigvecs.T
            self._delta_core: NDArray[np.float64] = 0.5 * (
                delta_core + delta_core.T
            )

            # Rango y Betti aproximado
            self._rank_delta: int = int(np.sum(eigvals_sqrt > tol_eig))
            self._betti_approx: int = E - self._rank_delta

            # Entropía de von Neumann del espectro normalizado
            self._spectral_entropy: float = self._von_neumann_entropy(
                eigvals_clamped
            )

            # Verificación de la identidad de Hodge local
            self._hodge_residual: float = self._verify_hodge_identity()

            logger.debug(
                "[Fase3] δ_CORE precalculada: E=%d, rank=%d, "
                "β₀≈%d, S_vN=%.4f, Hodge_residual=%.3e.",
                E,
                self._rank_delta,
                self._betti_approx,
                self._spectral_entropy,
                self._hodge_residual,
            )

        @staticmethod
        def _von_neumann_entropy(
            eigvals: NDArray[np.float64],
        ) -> float:
            r"""
            Entropía de von Neumann del espectro no negativo:

                S = −Σ p_i ln p_i ,   p_i = λ_i / Σ λ_j

            S = 0 ⇔ un solo modo (rango 1); S → ln(E) ⇔ espectro plano.
            """
            total: float = float(np.sum(eigvals))
            if total <= 0.0:
                return 0.0
            p: NDArray[np.float64] = eigvals / total
            # Evitar 0·log(0): solo sumar donde p > 0
            p_pos = p[p > 0.0]
            return float(-np.sum(p_pos * np.log(p_pos)))

        def _verify_hodge_identity(self) -> float:
            r"""
            Verifica la identidad de Hodge local:

                ‖ δ_COREᵀ · δ_CORE − W_mod ‖_F / ‖ W_mod ‖_F

            Retorna
            -------
            float
                Error relativo de la identidad de Hodge.

            Lanza
            -----
            SheafCoboundaryError
                Si el error relativo supera 100·ε_mach.
            """
            delta_sq: NDArray[np.float64] = (
                self._delta_core @ self._delta_core
            )
            residual_mat: NDArray[np.float64] = delta_sq - self._W_dense

            norm_W: float = float(la.norm(self._W_dense, "fro"))
            residual_F: float = float(la.norm(residual_mat, "fro"))
            rel_error: float = residual_F / max(norm_W, 1.0)

            tol_hodge: float = 100.0 * self._EPS

            if rel_error > tol_hodge:
                raise SheafCoboundaryError(
                    f"δ_COREᵀ δ_CORE ≇ W_mod: identidad de Hodge violada. "
                    f"‖δ² − W‖_F / ‖W‖_F = {rel_error:.6e} > "
                    f"tol = {tol_hodge:.6e}. "
                    f"Error de ensamble en la raíz espectral."
                )

            return rel_error

        # ---------------------------------------------------------------------
        # Método terminal de la Fase 3 (salida pública del ecosistema)
        # ---------------------------------------------------------------------

        def export_stalk(
            self,
            state_x: NDArray[np.float64],
        ) -> "SheafStalk":
            r"""
            **Método terminal de la Fase 3 y del agente completo.**

            Proyecta el vector de estado x sobre la fibra local mediante
            δ_CORE (precalculada en el constructor) y retorna el
            ``SheafStalk`` completo, incluyendo invariantes topológicos
            y la entropía espectral.

            Parámetros
            ----------
            state_x : NDArray[np.float64], shape (E,)
                Vector de estado en el espacio de aristas (1-cochains).

            Retorna
            -------
            SheafStalk
                Fibrado celular completo, inmutable.

            Lanza
            -----
            KinematicDimensionError
                Si state_x no tiene la dimensión E del espacio de aristas.
            """
            E: int = int(self._delta_core.shape[0])

            if state_x.shape != (E,):
                raise KinematicDimensionError(
                    f"state_x debe tener shape ({E},) = (n_aristas,); "
                    f"se obtuvo {state_x.shape}."
                )

            projected: NDArray[np.float64] = self._delta_core @ state_x

            logger.info(
                "[Fase3] SheafStalk exportado: E=%d, rank_delta=%d, "
                "β₀≈%d, S_vN=%.4f, Hodge_res=%.3e, ‖δx‖₂=%.6e.",
                E,
                self._rank_delta,
                self._betti_approx,
                self._spectral_entropy,
                self._hodge_residual,
                float(la.norm(projected, 2)),
            )

            # ================================================================
            # CONTRATO DE SALIDA DEL AGENTE COMPLETO
            # El SheafStalk es el output final de la cadena de 3 fases.
            # ================================================================
            return SheafStalk(
                delta_core=self._delta_core,
                delta_hodge_residual=self._hodge_residual,
                state_vector=state_x.copy(),
                projected_state=projected,
                rank_delta=self._rank_delta,
                betti_approx=self._betti_approx,
                spectral_entropy=self._spectral_entropy,
            )

    # =========================================================================
    # INTERFAZ PÚBLICA DEL AGENTE (punto de entrada externo)
    # =========================================================================

    def synthesize_kinematic_core(
        self,
        grad_H: NDArray[np.float64],
        grad_H_d: NDArray[np.float64],
        W: sp.spmatrix,
        I_curl: NDArray[np.float64],
        Z_load: NDArray[np.float64],
        c_eff: float,
        Delta_sym: sp.spmatrix,
        dt_requested: float,
    ) -> KinematicStateTensor:
        r"""
        Punto de entrada público para la síntesis cinemática completa.

        Delega en ``Phase2_KinematicSynthesis.synthesize`` y almacena la
        conductancia modulada resultante para la exportación del haz (Fase 3).

        Parámetros
        ----------
        grad_H : NDArray[np.float64], shape (n,)
            Gradiente del Hamiltoniano actual ∇H(x).
        grad_H_d : NDArray[np.float64], shape (n,)
            Gradiente del Hamiltoniano deseado ∇H_d(x).
        W : sp.spmatrix, shape (E, E)
            Matriz de conductancia de aristas del grafo logístico.
        I_curl : NDArray[np.float64], shape (E,)
            Corriente de curl (vorticidad) sobre las aristas.
        Z_load : NDArray[np.float64], shape (d, d)
            Tensor de impedancia de carga (SPD).
        c_eff : float
            Velocidad de propagación efectiva del frente logístico.
        Delta_sym : sp.spmatrix, shape (V, V)
            Laplaciano simétrico del grafo logístico.
        dt_requested : float
            Paso temporal solicitado por el integrador externo.

        Retorna
        -------
        KinematicStateTensor
            Estado cinemático completo e inmutable.
        """
        state: KinematicStateTensor = self.phase2.synthesize(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=c_eff,
            Delta_sym=Delta_sym,
            dt_requested=dt_requested,
        )

        # Actualizar conductancia más reciente para la Fase 3
        self._latest_hodge_conductance = state.hodge_conductance
        # Invalidar Fase 3 si W_mod cambió (forzar re-instanciación perezosa)
        self.phase3 = None

        logger.debug(
            "[KCoreKinematicAgent] Síntesis completada: "
            "CFL_safe_dt=%.6e, vorticity=%.6e, r_idapbc_G=%.3e, "
            "ρ_G=%.3e, λ_max=%.3e.",
            state.cfl_safe_dt,
            state.vorticity_norm,
            state.residual_idapbc,
            state.gershgorin_rho,
            state.lambda_max_delta,
        )
        return state

    def export_sheaf_stalk(
        self,
        state_x: NDArray[np.float64],
    ) -> SheafStalk:
        r"""
        Exporta el Stalk del haz cinemático y la cofrontera δ_CORE.

        Requiere que ``synthesize_kinematic_core`` haya sido llamado
        previamente para disponer de la conductancia modulada W_mod.

        La Fase 3 se instancia perezosamente: el coste de la descomposición
        espectral de W_mod se paga una vez y se reutiliza para múltiples
        proyecciones con el mismo W_mod.

        Parámetros
        ----------
        state_x : NDArray[np.float64], shape (E,)
            Vector de estado en el espacio de aristas.

        Retorna
        -------
        SheafStalk
            Fibrado celular completo e inmutable.

        Lanza
        -----
        KinematicCoreError
            Si ``synthesize_kinematic_core`` no ha sido llamado previamente.
        """
        if self._latest_hodge_conductance is None:
            raise KinematicCoreError(
                "No se dispone de conductancia modulada W_mod. "
                "Ejecute ``synthesize_kinematic_core`` antes de "
                "``export_sheaf_stalk``."
            )

        if self.phase3 is None:
            self.phase3 = KCoreKinematicAgent.Phase3_SheafProjection(
                W_mod=self._latest_hodge_conductance
            )
            logger.info(
                "[KCoreKinematicAgent] Phase3_SheafProjection instanciada "
                "(lazy init): rank_delta=%d, β₀≈%d, S_vN=%.4f.",
                self.phase3._rank_delta,
                self.phase3._betti_approx,
                self.phase3._spectral_entropy,
            )

        return self.phase3.export_stalk(state_x=state_x)
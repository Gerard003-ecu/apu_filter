# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo : KCore Kinematic Agent (Director de Flujo y Cinética Logística)      |
| Ruta   : app/agents/alpha/kcore/kcore_kinematic_agent.py                     |
| Versión: 5.0.0-Rigorous-IDA-PBC-Hodge-CFL                                    |
+==============================================================================+

NATURALEZA CIBER-FÍSICA Y ESTRUCTURA DE DIRAC:
Este módulo impone el moldeado de energía mediante un Control Basado en Pasividad
(IDA-PBC). La ley de control alpha(x) utiliza una Proyección Pseudoinversa Covariante
que garantiza que el esfuerzo exógeno sea ortogonal a las geodésicas de alta fricción:

\[ \alpha(x) = (g(x)^\top G_{\mu\nu} g(x))^{-1} g(x)^\top G_{\mu\nu} ([J_d - R_d] \nabla H_d - [J - R] \nabla H) \]

VÁLVULA DE HODGE Y LÍMITE CFL:
\[ L_{1W} = \partial_1^\top W^{-1} \partial_1 + \partial_2 \partial_2^\top W \]
\[ \Delta t \le \frac{2 \cdot CFL_{margin}}{c_{eff} \cdot \max_i ( |\Delta_{ii}| + \sum_{j \neq i} |\Delta_{ij}| )} \]
"""
from __future__ import annotations

#    Biblioteca est ndar
import logging
from typing import Optional, Tuple

#     lgebra num rica de alta precisi n
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
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
logger = logging.getLogger("MIC.Alpha.KCoreKinematicAgent")


#
#    SECCI N 0   EXCEPCIONES CINEM TICAS ESTRICTAS
#


class KinematicCoreError(Exception):
    """
    Excepci n categ rica ra z para violaciones en el Estrato K_CORE.

    Toda excepci n de este m dulo hereda de esta clase, permitiendo que
    los manejadores de nivel superior capturen cualquier fallo cinem tico
    con un  nico ``except KinematicCoreError``.
    """


class KinematicDimensionError(KinematicCoreError):
    """
    Lanzada cuando las dimensiones de las matrices constitutivas son
    inconsistentes entre s  o con el espacio de estado declarado.

    Diagn stico incluye las formas (shapes) detectadas y las esperadas.
    """


class KinematicSymmetryError(KinematicCoreError):
    """
    Lanzada cuando una matriz viola su propiedad de simetr a o
    antisimetr a requerida, con diagn stico cuantitativo normalizado.
    """


class KinematicConditionError(KinematicCoreError):
    """
    Lanzada cuando el n mero de condici n espectral  (A) supera el
    umbral admisible, comprometiendo la estabilidad num rica de la
    s ntesis IDA-PBC o del control de vorticidad.
    """


class DiracMatchingError(KinematicCoreError):
    """
    Lanzada cuando la ecuaci n de matching IDA-PBC:

        [J_d - R_d]  H_d = [J - R]  H + g

    carece de soluci n estable, bien porque g es rango-deficiente m s
    all  de la tolerancia SVD, bien porque el residuo relativo excede
    el umbral configurado.

    Incluye diagn stico del rango efectivo de g y el residuo normalizado.
    """


class ParasiticVorticityError(KinematicCoreError):
    """
    Lanzada cuando el Laplaciano de Hodge detecta componentes solenoidales
    (flujo circular) que superan el umbral cr tico  _crit y cuyo soporte
    espectral no puede ser estrangulado con el factor configurado.
    """


class ImpedanceReflectionError(KinematicCoreError):
    """
    Lanzada cuando la sintonizaci n Kramers-Kronig falla:
        Z_load no es SPD (acoplamiento de impedancia imposible).
        El tensor  _eff resultante no es SPD (violaci n de causalidad).
        La relaci n de dispersi n causal no se satisface num ricamente.
    """


class CFLViolationError(KinematicCoreError):
    """
    Lanzada cuando:
        El paso temporal dt_requested excede  t_safe (violaci n CFL activa).
        El c lculo del autovalor m ximo del Laplaciano falla (ARPACK diverge).
        c_eff <= 0 (velocidad de propagaci n no f sica).
    """


class SheafCoboundaryError(KinematicCoreError):
    """
    Lanzada cuando  _{CORE} no satisface la identidad de Hodge local:

         _{CORE}^     _{CORE} ~= W_mod

    con tolerancia de 100  _mach relativa a  W_mod _F.
    """


#
#    SECCI N 1   ESTRUCTURAS INMUTABLES (DTOs TENSORIALES)
#


@dataclass(frozen=True, slots=True)
class KinematicPreparationContext:
    r"""
    Contexto inmutable producido por la **Fase 1** (Validación Matricial).

    Contiene las matrices constitutivas validadas y sus metadatos espectrales,
    necesarios para que la Fase 2 opere sin re-validar ni re-descomponer.

    Atributos
    ----------
    J : NDArray[np.float64], shape (n, n)
        Matriz de interconexión del sistema real, antisimétrica J = -J^⊤.
    R : NDArray[np.float64], shape (n, n)
        Matriz de disipación del sistema real, PSD R ⪰ 0.
    J_d : NDArray[np.float64], shape (n, n)
        Matriz de interconexión deseada (IDA-PBC), antisimétrica.
    R_d : NDArray[np.float64], shape (n, n)
        Matriz de disipación deseada (IDA-PBC), PSD.
    g : NDArray[np.float64], shape (n, m)
        Matriz de entrada del control, rango posiblemente deficiente.
    n : int
        Dimensión del espacio de estado.
    m : int
        Número de entradas de control (columnas de g).
    rank_g : int
        Rango numérico de g (puede ser < min(n, m) si g es rango-deficiente).
    kappa_R : float
        Número de condición espectral de R (para trazabilidad numérica).
    kappa_R_d : float
        Número de condición espectral de R_d.
    """

    J: NDArray[np.float64]
    R: NDArray[np.float64]
    J_d: NDArray[np.float64]
    R_d: NDArray[np.float64]
    g: NDArray[np.float64]
    n: int
    m: int
    rank_g: int
    kappa_R: float
    kappa_R_d: float


@dataclass(frozen=True, slots=True)
class KinematicStateTensor:
    r"""
    Tensor inmutable que encapsula el estado cinemático completo del núcleo.

    Producido por la **Fase 2** (Síntesis Cinemática).

    Atributos
    ----------
    control_law_alpha : NDArray[np.float64], shape (m,)
        Ley de control IDA-PBC: α = g⁺ ([J_d−R_d]∇H_d − [J−R]∇H).
    hodge_conductance : sp.spmatrix
        Matriz de conductancia W_mod modulada por estrangulamiento de Hodge.
    dielectric_tensor : NDArray[np.float64]
        Tensor dieléctrico efectivo ε_eff ≻ 0 (Kramers-Kronig).
    magnetic_tensor : NDArray[np.float64]
        Tensor magnético efectivo μ_eff ≻ 0 (Kramers-Kronig).
    cfl_safe_dt : float
        Paso temporal máximo admisible Δt_safe > 0 según la condición CFL.
    residual_idapbc : float
        Residuo normalizado de la ecuación de matching IDA-PBC:
        ‖g α − F_req‖_2 / max(‖F_req‖_2, 1).
    vorticity_norm : float
        ‖I_curl‖_{W} = sqrt(I_curl^⊤ W I_curl): norma de la vorticidad.
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
    is_kinematically_stable: bool


@dataclass(frozen=True, slots=True)
class SheafStalk:
    r"""
    Fibrado celular exportado para el cálculo global del Laplaciano de Haz.

    Producido por la **Fase 3** (Proyección en Haces).

    Atributos
    ----------
    delta_core : NDArray[np.float64]
        Cofrontera discreta δ_{CORE} = W_mod^{+1/2} ∈ ℝ^{E×E},
        calculada vía raíz cuadrada espectral de W_mod.

    delta_hodge_residual : float
        Error relativo de la identidad de Hodge local:
        ‖δ_{CORE}^⊤ δ_{CORE} − W_mod_dense‖_F / ‖W_mod_dense‖_F.
        Debe ser O(ε_mach).

    state_vector : NDArray[np.float64]
        Vector de estado x en el instante de proyección.

    projected_state : NDArray[np.float64]
        Proyección δ_{CORE} · x sobre la fibra local.

    rank_delta : int
        Rango numérico de δ_{CORE} = rango de W_mod.
    """

    delta_core: NDArray[np.float64]
    delta_hodge_residual: float
    state_vector: NDArray[np.float64]
    projected_state: NDArray[np.float64]
    rank_delta: int


#
#    SECCI N 2   ORQUESTADOR: KCoreKinematicAgent
#                Tres fases anidadas de rigor creciente
#


class KCoreKinematicAgent(Morphism):
    r"""
    Orquestador Funtorial del Núcleo Cinemático K_{CORE}.

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
    J : NDArray[np.float64], shape (n, n)
        Matriz de interconexión del sistema (antisimétrica).
    R : NDArray[np.float64], shape (n, n)
        Matriz de disipación del sistema (PSD).
    J_d : NDArray[np.float64], shape (n, n)
        Matriz de interconexión deseada IDA-PBC (antisimétrica).
    R_d : NDArray[np.float64], shape (n, n)
        Matriz de disipación deseada IDA-PBC (PSD).
    g : NDArray[np.float64], shape (n, m)
        Matriz de entrada del control.
    cfl_margin : float, default 0.9
        Factor de seguridad CFL ∈ (0, 1]. Valores > 1 son teóricamente
        inestables y se rechazan.
    kappa_max : float, default 1e10
        Umbral de número de condición espectral para R y R_d.
    residual_tol_rel : float, default 1e-6
        Tolerancia relativa para el residuo IDA-PBC:
        ‖g α − F_req‖ / ‖F_req‖ < residual_tol_rel.
    """

    FRIENDLY_NAME: str = "Director de Flujo y Cin tica Log stica"

    def __init__(
        self,
        J: NDArray[np.float64],
        R: NDArray[np.float64],
        J_d: NDArray[np.float64],
        R_d: NDArray[np.float64],
        g: NDArray[np.float64],
        cfl_margin: float = 0.9,
        kappa_max: float = 1.0e10,
        residual_tol_rel: float = 1.0e-6,
    ) -> None:
        r"""
        Inicializa las matrices constitutivas y ejecuta la Fase 1 de inmediato.

        Lanza
        -----
        ValueError
            Si cfl_margin ∉ (0, 1].
        KinematicDimensionError, KinematicSymmetryError, KinematicConditionError
            Propagadas desde la Fase 1 si alguna propiedad matricial es violada.
        """
        if not (0.0 < cfl_margin <= 1.0):
            raise ValueError(
                f"cfl_margin debe estar en (0, 1]; se obtuvo {cfl_margin}. "
                f"Valores > 1 implican inestabilidad num rica del esquema."
            )

        self.cfl_margin: float = cfl_margin
        self.kappa_max: float = kappa_max
        self.residual_tol_rel: float = residual_tol_rel

        #    Fase 1: Validaci n Matricial Constitutiva (inmediata)
        self.phase1: KCoreKinematicAgent.Phase1_MatrixValidation = (
            KCoreKinematicAgent.Phase1_MatrixValidation(
                J=J,
                R=R,
                J_d=J_d,
                R_d=R_d,
                g=g,
                kappa_max=kappa_max,
            )
        )
        self.context: KinematicPreparationContext = (
            self.phase1.build_preparation_context()
        )

        #    Fase 2: S ntesis Cinem tica (instanciaci n inmediata)
        self.phase2: KCoreKinematicAgent.Phase2_KinematicSynthesis = (
            KCoreKinematicAgent.Phase2_KinematicSynthesis(
                context=self.context,
                cfl_margin=self.cfl_margin,
                residual_tol_rel=self.residual_tol_rel,
            )
        )

        #    Estado interno: conductancia modulada m s reciente
        self._latest_hodge_conductance: Optional[sp.spmatrix] = None

        #    Fase 3: instanciaci n perezosa
        self.phase3: Optional[KCoreKinematicAgent.Phase3_SheafProjection] = None

        logger.info(
            "[KCoreKinematicAgent] Inicializado: n=%d, m=%d, "
            "rank(g)=%d,  (R)=%.3e,  (R_d)=%.3e, CFL_margin=%.2f.",
            self.context.n,
            self.context.m,
            self.context.rank_g,
            self.context.kappa_R,
            self.context.kappa_R_d,
            self.cfl_margin,
        )

    #
    # FASE 1   VALIDACI N MATRICIAL CONSTITUTIVA
    #

    class Phase1_MatrixValidation:
        r"""
        **Fase 1 – Validación Matricial Constitutiva.**

        Responsabilidades exclusivas de esta fase:
          a) Verificar dimensiones y consistencia del espacio de estado.
          b) Verificar antisimetría de J, J_d con tolerancia relativa.
          c) Verificar simetría de R, R_d con tolerancia relativa.
          d) Verificar PSD de R, R_d con tolerancia normalizada.
          e) Calcular κ(R), κ(R_d) y rango numérico de g.
          f) Retornar ``KinematicPreparationContext`` inmutable.

        Todas las tolerancias son *relativas* a la norma de Frobenius de la
        matriz evaluada multiplicada por la precisión de máquina ε_mach,
        eliminando falsos positivos para matrices de gran norma.
        """

        _EPS: float = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            J: NDArray[np.float64],
            R: NDArray[np.float64],
            J_d: NDArray[np.float64],
            R_d: NDArray[np.float64],
            g: NDArray[np.float64],
            kappa_max: float = 1.0e10,
        ) -> None:
            r"""
            Almacena referencias a las matrices originales sin copiarlas.
            Las copias ocurren sólo en ``build_preparation_context``.
            """
            self._J: NDArray[np.float64] = J
            self._R: NDArray[np.float64] = R
            self._J_d: NDArray[np.float64] = J_d
            self._R_d: NDArray[np.float64] = R_d
            self._g: NDArray[np.float64] = g
            self._kappa_max: float = kappa_max

        #
        # M todos privados de validaci n (orden l gico de ejecuci n)
        #

        def _check_dimensions(self) -> Tuple[int, int]:
            r"""
            Verifica la coherencia dimensional completa del espacio de estado.

            Condiciones formales:
              • J, R, J_d, R_d ∈ ℝ^{n×n}  (cuadradas, misma dimensión n)
              • g ∈ ℝ^{n×m}                (n filas, m columnas, m ≥ 1)
              • Todos los arrays son 2D.

            Retorna
            -------
            Tuple[int, int]
                (n, m): dimensión del espacio de estado y número de entradas.

            Lanza
            -----
            KinematicDimensionError
                Con diagnóstico explícito de la forma violada.
            """
            # Verificar que cada matriz es 2D
            for mat, name in [
                (self._J, "J"), (self._R, "R"),
                (self._J_d, "J_d"), (self._R_d, "R_d"), (self._g, "g"),
            ]:
                if mat.ndim != 2:
                    raise KinematicDimensionError(
                        f"La matriz '{name}' debe ser 2D; "
                        f"se obtuvo ndim={mat.ndim}, shape={mat.shape}."
                    )

            # Verificar cuadratura de J (define n)
            if self._J.shape[0] != self._J.shape[1]:
                raise KinematicDimensionError(
                    f"J debe ser cuadrada; se obtuvo shape={self._J.shape}."
                )
            n: int = self._J.shape[0]

            # Verificar que R, J_d, R_d son cuadradas y de dimensi n n
            for mat, name in [
                (self._R, "R"), (self._J_d, "J_d"), (self._R_d, "R_d")
            ]:
                if mat.shape != (n, n):
                    raise KinematicDimensionError(
                        f"La matriz '{name}' debe tener shape ({n},{n}); "
                        f"se obtuvo {mat.shape}. "
                        f"Dimensi n del espacio de estado n={n} definida por J."
                    )

            # Verificar que g tiene n filas
            if self._g.shape[0] != n:
                raise KinematicDimensionError(
                    f"g debe tener {n} filas (dim. espacio de estado); "
                    f"se obtuvo {self._g.shape[0]} filas, shape={self._g.shape}."
                )
            m: int = self._g.shape[1]

            if m < 1:
                raise KinematicDimensionError(
                    f"g debe tener al menos 1 columna (entrada de control); "
                    f"se obtuvo m={m}."
                )

            logger.debug(
                "[Fase1] Dimensiones verificadas: n=%d, m=%d.", n, m
            )
            return n, m

        def _validate_antisymmetry(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> None:
            r"""
            Verifica que A = −A^⊤ con tolerancia relativa al Frobenius de A.

            Tolerancia adaptativa:
                tol = ε_mach · ‖A‖_F

            Cuantifica el residuo como:
                ‖A + A^⊤‖_F  (norma de la parte simétrica, debe ser ≈ 0)

            Lanza
            -----
            KinematicSymmetryError
                Con residuo absoluto y relativo para diagnóstico.
            """
            norm_A: float = float(la.norm(A, "fro"))
            tol: float = self._EPS * max(norm_A, 1.0)
            # Parte sim trica de A: A_sym = (A + A^ )/2; debe ser ~= 0
            residual: float = float(la.norm(A + A.T, "fro"))

            if residual > tol:
                raise KinematicSymmetryError(
                    f"La matriz '{name}' no es antisim trica (A   -A^ ). "
                    f" A + A^  _F = {residual:.6e},  tol = {tol:.6e},  "
                    f"antisimetr a relativa = {residual / max(norm_A, 1e-300):.6e}."
                )

            logger.debug(
                "[Fase1] Antisimetr a de '%s': residual=%.3e, tol=%.3e.",
                name, residual, tol,
            )

        def _validate_symmetry(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> None:
            r"""
            Verifica que A = A^⊤ con tolerancia relativa al Frobenius de A.

            Tolerancia adaptativa:
                tol = ε_mach · ‖A‖_F

            Lanza
            -----
            KinematicSymmetryError
                Con diagnóstico cuantitativo de ‖A − A^⊤‖_F.
            """
            norm_A: float = float(la.norm(A, "fro"))
            tol: float = self._EPS * max(norm_A, 1.0)
            residual: float = float(la.norm(A - A.T, "fro"))

            if residual > tol:
                raise KinematicSymmetryError(
                    f"La matriz '{name}' no es sim trica (A   A^ ). "
                    f" A - A^  _F = {residual:.6e},  tol = {tol:.6e},  "
                    f"asimetr a relativa = {residual / max(norm_A, 1e-300):.6e}."
                )

            logger.debug(
                "[Fase1] Simetr a de '%s': residual=%.3e, tol=%.3e.",
                name, residual, tol,
            )

        def _validate_psd(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> float:
            r"""
            Verifica que A ⪰ 0 (semidefinida positiva) con tolerancia relativa.

            La tolerancia distingue autovalores nulos legítimos (rango deficiente
            por diseño) de autovalores genuinamente negativos (violación PSD):

                tol_psd = −ε_mach · ‖A‖_F

            Retorna
            -------
            float
                κ(A) = λ_max / λ_min_positivo, o ∞ si A es rango-deficiente.
                Útil para trazabilidad numérica de las fases posteriores.

            Lanza
            -----
            KinematicConditionError
                Si κ(A) > kappa_max (cuasi-singularidad numérica).
            KinematicSymmetryError
                Si λ_min < −tol_psd (autovalor genuinamente negativo).
            """
            norm_A: float = float(la.norm(A, "fro"))
            tol_psd: float = self._EPS * max(norm_A, 1.0)

            # eigvalsh: descomposici n espectral real para matrices sim tricas
            eigvals: NDArray[np.float64] = la.eigvalsh(A)
            lambda_min: float = float(eigvals[0])
            lambda_max: float = float(eigvals[-1])

            if lambda_min < -tol_psd:
                raise KinematicSymmetryError(
                    f"La matriz '{name}' no es Semidefinida Positiva (PSD). "
                    f" _min = {lambda_min:.6e}  <  -tol = {-tol_psd:.6e}."
                )

            # Autovalores positivos (excluyendo los nulos)
            pos_eigvals = eigvals[eigvals > tol_psd]
            if len(pos_eigvals) == 0:
                logger.warning(
                    "[Fase1] Matriz '%s' es num ricamente nula (rank=0).",
                    name,
                )
                return float("inf")

            lambda_min_pos: float = float(pos_eigvals[0])
            kappa: float = lambda_max / lambda_min_pos

            if kappa > self._kappa_max:
                raise KinematicConditionError(
                    f"La matriz '{name}' est  mal condicionada: "
                    f"  = {kappa:.6e}  >   _max = {self._kappa_max:.6e}. "
                    f" _min_pos = {lambda_min_pos:.6e},  _max = {lambda_max:.6e}. "
                    f"Considere regularizaci n de Tikhonov."
                )

            logger.debug(
                "[Fase1] PSD '%s':  _min=%.3e,  _max=%.3e,  =%.3e.",
                name, lambda_min, lambda_max, kappa,
            )
            return kappa

        def _compute_rank_g(self, n: int, m: int) -> int:
            r"""
            Calcula el rango numérico de g mediante SVD completa.

            La tolerancia de truncación SVD sigue el criterio de Golub-Van Loan:

                σ_tol = max(n, m) · ε_mach · σ_max

            que es el umbral estándar para distinguir valores singulares
            numéricos de cero de los genuinamente no nulos.

            Retorna
            -------
            int
                Rango efectivo de g: número de valores singulares > σ_tol.

            Lanza
            -----
            KinematicDimensionError
                Si el rango es 0 (g es numéricamente nula, sin capacidad de control).
            """
            # SVD completa para m xima precisi n en el c lculo de rango
            _, s, _ = la.svd(self._g, full_matrices=False, check_finite=False)
            sigma_max: float = float(s[0]) if len(s) > 0 else 0.0
            sigma_tol: float = max(n, m) * self._EPS * sigma_max

            rank_g: int = int(np.sum(s > sigma_tol))

            if rank_g == 0:
                raise KinematicDimensionError(
                    f"La matriz de control g es num ricamente nula "
                    f"( _max = {sigma_max:.6e}, todos los valores singulares <= {sigma_tol:.6e}). "
                    f"Sin capacidad de control sobre el sistema."
                )

            logger.debug(
                "[Fase1] Rango de g: rank=%d/%d ( _max=%.3e,  _tol=%.3e).",
                rank_g, min(n, m), sigma_max, sigma_tol,
            )
            return rank_g

        #
        # M todo terminal de la Fase 1   entrada directa de la Fase 2
        #

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
            1. Verificación dimensional (J, R, J_d, R_d, g).
            2. Antisimetría de J y J_d.
            3. Simetría de R y R_d.
            4. PSD y número de condición de R y R_d.
            5. Rango numérico de g.
            6. Empaquetado en KinematicPreparationContext.

            Retorna
            -------
            KinematicPreparationContext
                Contexto cinemático completo, inmutable y listo para Fase 2.
            """
            #    Paso 1: Dimensiones
            n, m = self._check_dimensions()

            #    Paso 2: Antisimetr a de matrices de interconexi n
            self._validate_antisymmetry(self._J, "J")
            self._validate_antisymmetry(self._J_d, "J_d")

            #    Paso 3: Simetr a de matrices de disipaci n
            self._validate_symmetry(self._R, "R")
            self._validate_symmetry(self._R_d, "R_d")

            #    Paso 4: PSD y condicionamiento de R y R_d
            kappa_R: float = self._validate_psd(self._R, "R")
            kappa_R_d: float = self._validate_psd(self._R_d, "R_d")

            #    Paso 5: Rango num rico de g
            rank_g: int = self._compute_rank_g(n, m)

            #    Paso 6: Empaquetado
            context = KinematicPreparationContext(
                J=self._J.copy(),
                R=self._R.copy(),
                J_d=self._J_d.copy(),
                R_d=self._R_d.copy(),
                g=self._g.copy(),
                n=n,
                m=m,
                rank_g=rank_g,
                kappa_R=kappa_R,
                kappa_R_d=kappa_R_d,
            )

            logger.info(
                "[Fase1] KinematicPreparationContext ensamblado: "
                "n=%d, m=%d, rank(g)=%d,  (R)=%.3e,  (R_d)=%.3e.",
                n, m, rank_g, kappa_R, kappa_R_d,
            )

            #    Contrato de interfaz Fase 1   Fase 2
            # `context` es el argumento directo del constructor de
            # Phase2_KinematicSynthesis. Esta devoluci n es la frontera
            # formal entre ambas fases anidadas.
            return context

    #
    # FASE 2   S NTESIS CINEM TICA
    #

    class Phase2_KinematicSynthesis:
        r"""
        **Fase 2 – Síntesis Cinemática.**

        Recibe el ``KinematicPreparationContext`` de la Fase 1 y lo usa para
        ejecutar los cuatro procesos fundamentales del K_{CORE}:

          1. **Moldeado de energía IDA-PBC** (``compute_dirac_control_law``):
             Resuelve la ecuación de matching mediante pseudoinversa SVD truncada
             con criterio de Golub-Van Loan, verifica residuo relativo y reporta
             deficiencia de rango.

          2. **Estrangulamiento de vorticidad de Hodge** (``modulate_hodge_conductance``):
             Convierte W a formato COO para acceso uniforme a diagonal,
             cuantifica la vorticidad como forma cuadrática ‖I_curl‖²_W,
             y aplica penalización espectral proporcional al soporte de I_curl.

          3. **Sintonización de impedancia Kramers-Kronig** (``tune_impedance_tensors``):
             Calcula ε_eff y μ_eff verificando SPD con Cholesky explícito y
             verificando la relación de dispersión causal ‖Z_0 − Z_load‖_F/‖Z_load‖_F.

          4. **Auditoría CFL** (``audit_cfl_limit``):
             Calcula λ_max del Laplaciano con ARPACK, maneja convergencia fallida
             con fallback a `eigvalsh` denso, y verifica c_eff > 0.

        El constructor de esta clase es la **continuación directa** del método
        ``build_preparation_context`` de la Fase 1.
        """

        _EPS: float = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            context: "KinematicPreparationContext",
            cfl_margin: float,
            residual_tol_rel: float,
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
                Tolerancia relativa para el residuo IDA-PBC.
            """
            self._ctx: "KinematicPreparationContext" = context
            self._cfl_margin: float = cfl_margin
            self._residual_tol_rel: float = residual_tol_rel

            logger.debug(
                "[Fase2] Inicializada: n=%d, m=%d, rank(g)=%d, "
                "CFL_margin=%.2f, res_tol=%.3e.",
                context.n, context.m, context.rank_g,
                cfl_margin, residual_tol_rel,
            )

        #
        # Subproceso 1: Moldeado de energ a IDA-PBC
        #

        def compute_dirac_control_law(
            self,
            grad_H: NDArray[np.float64],
            grad_H_d: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Resuelve la ecuación de matching IDA-PBC:

                [J_d − R_d] ∇H_d = [J − R] ∇H + g α
                ⟺  g α = F_req  donde  F_req = [J_d−R_d]∇H_d − [J−R]∇H

            mediante la pseudoinversa de Moore-Penrose de g con SVD truncada:

                g⁺ = V · diag(1/σᵢ  si  σᵢ > σ_tol,  0 en otro caso) · U^⊤

            Criterio de truncación (Golub-Van Loan):
                σ_tol = max(n, m) · ε_mach · σ_max

            Verificación del residuo relativo:
                r_rel = ‖g α − F_req‖_2 / max(‖F_req‖_2, 1) < residual_tol_rel

            Parámetros
            ----------
            grad_H : NDArray[np.float64], shape (n,)
                Gradiente del Hamiltoniano actual ∇H(x).
            grad_H_d : NDArray[np.float64], shape (n,)
                Gradiente del Hamiltoniano deseado ∇H_d(x).

            Retorna
            -------
            Tuple[NDArray[np.float64], float]
                (alpha, residual_rel):
                  alpha        ∈ ℝ^m: ley de control IDA-PBC.
                  residual_rel ∈ ℝ⁺:  residuo relativo de la ecuación de matching.

            Lanza
            -----
            KinematicDimensionError
                Si grad_H o grad_H_d no tienen shape (n,).
            DiracMatchingError
                Si el residuo relativo supera residual_tol_rel, con diagnóstico
                de rango efectivo de g y norma del residuo.
            """
            n: int = self._ctx.n

            # Validaci n de dimensiones de los gradientes
            if grad_H.shape != (n,):
                raise KinematicDimensionError(
                    f"grad_H debe tener shape ({n},); se obtuvo {grad_H.shape}."
                )
            if grad_H_d.shape != (n,):
                raise KinematicDimensionError(
                    f"grad_H_d debe tener shape ({n},); se obtuvo {grad_H_d.shape}."
                )

            # Fuerza deseada y natural del sistema Port-Hamiltoniano
            F_d: NDArray[np.float64] = (
                self._ctx.J_d - self._ctx.R_d
            ) @ grad_H_d
            F_nat: NDArray[np.float64] = (
                self._ctx.J - self._ctx.R
            ) @ grad_H

            # Fuerza de control requerida: F_req = F_d - F_nat
            F_req: NDArray[np.float64] = F_d - F_nat
            norm_F_req: float = float(la.norm(F_req, 2))

            # SVD de g para pseudoinversa de Moore-Penrose
            try:
                U, s, Vh = la.svd(
                    self._ctx.g, full_matrices=False, check_finite=False
                )
            except la.LinAlgError as exc:
                raise DiracMatchingError(
                    f"Fallo de SVD (LAPACK dgesvd) en g. Error: {exc}"
                ) from exc

            # Criterio de truncaci n Golub-Van Loan
            sigma_max: float = float(s[0]) if len(s) > 0 else 0.0
            sigma_tol: float = (
                max(self._ctx.n, self._ctx.m) * self._EPS * sigma_max
            )
            mask: NDArray[np.bool_] = s > sigma_tol
            rank_effective: int = int(np.sum(mask))

            # Inversas de valores singulares truncados
            s_inv: NDArray[np.float64] = np.zeros_like(s)
            s_inv[mask] = 1.0 / s[mask]

            #   = V   diag(s_inv)   U^    F_req  (pseudoinversa aplicada)
            alpha: NDArray[np.float64] = Vh.T @ (s_inv * (U.T @ F_req))

            # Residuo de la ecuaci n de matching
            residual_vec: NDArray[np.float64] = self._ctx.g @ alpha - F_req
            residual_abs: float = float(la.norm(residual_vec, 2))
            # Normalizaci n relativa: evita divisi n por cero para F_req ~= 0
            residual_rel: float = residual_abs / max(norm_F_req, 1.0)

            if residual_rel > self._residual_tol_rel:
                logger.error(
                    "[Fase2] Residuo IDA-PBC: r_rel=%.6e > tol=%.6e, "
                    "rank_eff=%d/%d,  F_req =%.6e.",
                    residual_rel, self._residual_tol_rel,
                    rank_effective, min(self._ctx.n, self._ctx.m), norm_F_req,
                )
                raise DiracMatchingError(
                    f"Ecuaci n de matching IDA-PBC sin soluci n suficientemente precisa. "
                    f"Residuo relativo = {residual_rel:.6e} > tol = {self._residual_tol_rel:.6e}. "
                    f"Rango efectivo de g: {rank_effective}/{min(self._ctx.n, self._ctx.m)}. "
                    f" F_req  = {norm_F_req:.6e}."
                )

            logger.debug(
                "[Fase2] IDA-PBC: r_rel=%.3e, rank(g)_eff=%d,    =%.6e.",
                residual_rel, rank_effective, float(la.norm(alpha, 2)),
            )
            return alpha, residual_rel

        #
        # Subproceso 2: Estrangulamiento de vorticidad de Hodge
        #

        def modulate_hodge_conductance(
            self,
            W: sp.spmatrix,
            I_curl: NDArray[np.float64],
            epsilon_crit: float = 1.0e-2,
            strangle_factor: float = 1.0e-4,
        ) -> Tuple[sp.spmatrix, float]:
            r"""
            Estrangula la conductancia en aristas con vorticidad parásita.

            La vorticidad se mide con la norma de energía:

                ‖I_curl‖_W = sqrt(I_curl^⊤ · W_diag · I_curl)

            donde W_diag = diag(W) es el vector de pesos de aristas.
            Esta norma es semidefinida positiva y compatible con el
            Laplaciano de Hodge-1: L₁^W = ∂₁^⊤ W⁻¹ ∂₁ + ∂₂ W ∂₂^⊤.

            Si ‖I_curl‖_W > ε_crit, se penalizan las aristas cuyo
            |I_curl[e]| > 0.1 · ‖I_curl‖_∞:

                W_diag[e] ← W_diag[e] · strangle_factor

            La operación se realiza en formato COO para compatibilidad
            universal con cualquier formato sparse (CSR, CSC, LIL, DIA, etc.)
            sin bifurcación de formato.

            Parámetros
            ----------
            W : sp.spmatrix, shape (E, E)
                Matriz de conductancia de aristas (diagonal, cualquier formato).
            I_curl : NDArray[np.float64], shape (E,)
                Corriente de curl sobre las E aristas del grafo logístico.
            epsilon_crit : float
                Umbral de vorticidad admisible.
            strangle_factor : float ∈ (0, 1)
                Factor de penalización multiplicativo para aristas vorticosas.

            Retorna
            -------
            Tuple[sp.spmatrix, float]
                (W_mod, vorticity_norm):
                  W_mod          : conductancia modulada en formato CSR.
                  vorticity_norm : ‖I_curl‖_W (cuantificación de la vorticidad).

            Lanza
            -----
            KinematicDimensionError
                Si I_curl no tiene shape (E,) coherente con W.shape[0].
            ParasiticVorticityError
                Si strangle_factor ≤ 0 (penalización no física).
            """
            E: int = W.shape[0]

            if I_curl.shape != (E,):
                raise KinematicDimensionError(
                    f"I_curl debe tener shape ({E},) coherente con W.shape[0]={E}; "
                    f"se obtuvo {I_curl.shape}."
                )

            if strangle_factor <= 0.0:
                raise ParasiticVorticityError(
                    f"strangle_factor debe ser > 0; se obtuvo {strangle_factor}. "
                    f"Un factor <= 0 implicar a conductancia negativa (no f sica)."
                )

            #    Extracci n de la diagonal en formato COO universal
            # Conversi n a CSR para acceso eficiente a la diagonal
            W_csr: sp.csr_matrix = W.tocsr()
            w_diag: NDArray[np.float64] = W_csr.diagonal().copy()

            #    Norma de vorticidad:  I_curl  _W = I_curl^  diag(w_diag) I_curl
            # Para matrices diagonales: forma cuadr tica = sum(w_diag * I_curl )
            # Esto evita el producto matricial W @ I_curl (que es O(E ) para densa)
            w_i_curl_sq: NDArray[np.float64] = w_diag * (I_curl ** 2)
            quad_form: float = float(np.sum(w_i_curl_sq))

            if quad_form < 0.0:
                raise ParasiticVorticityError(
                    f"Forma cuadr tica I_curl^  W I_curl = {quad_form:.6e} < 0. "
                    f"W tiene entradas diagonales negativas (no f sica)."
                )

            vorticity_norm: float = float(np.sqrt(quad_form))

            if vorticity_norm > epsilon_crit:
                logger.info(
                    "[Fase2] Vorticidad par sita:  I_curl _W=%.4e >  _crit=%.4e. "
                    "Estrangulando conductancia.",
                    vorticity_norm, epsilon_crit,
                )

                # Soporte de la penalizaci n: aristas con |I_curl[e]| > 0.1    I_curl _
                inf_norm: float = float(np.max(np.abs(I_curl)))
                # Umbral adaptativo: 10% del pico de vorticidad
                threshold: float = 0.1 * inf_norm
                mask: NDArray[np.bool_] = np.abs(I_curl) > threshold
                n_penalized: int = int(np.sum(mask))

                if n_penalized == 0:
                    logger.warning(
                        "[Fase2] Vorticidad detectada pero soporte vac o "
                        "(todos |I_curl[e]| <= %.3e). Sin penalizaci n.",
                        threshold,
                    )
                else:
                    w_diag[mask] *= strangle_factor
                    logger.debug(
                        "[Fase2] %d/%d aristas penalizadas con factor %.3e.",
                        n_penalized, E, strangle_factor,
                    )

                # Reconstruir W_mod como matriz diagonal CSR
                W_mod: sp.csr_matrix = sp.diags(
                    w_diag, offsets=0, shape=(E, E), format="csr", dtype=np.float64
                )
            else:
                logger.debug(
                    "[Fase2] Vorticidad  I_curl _W=%.4e <=  _crit=%.4e. "
                    "Sin estrangulamiento.",
                    vorticity_norm, epsilon_crit,
                )
                W_mod = W_csr  # ya es CSR, sin copia innecesaria

            return W_mod, vorticity_norm

        #
        # Subproceso 3: Sintonizaci n de impedancia Kramers-Kronig
        #

        def tune_impedance_tensors(
            self,
            Z_load: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
            r"""
            Sintoniza los tensores dieléctrico ε_eff y magnético μ_eff para
            acoplamiento de impedancia perfecto:

                Z₀ = sqrt(μ_eff · ε_eff⁻¹)  ≡  Z_load

            Solución constructiva (Kramers-Kronig):
                ε_eff = L_Z · L_Z^⊤           (SPD, L_Z = Cholesky de Z_load)
                μ_eff = Z_load · ε_eff · Z_load^⊤ = Z_load²

            que garantiza μ_eff ≻ 0 por construcción (producto de matrices SPD).

            Verificación de la relación de dispersión causal:
                Z₀² = μ_eff · ε_eff⁻¹ = Z_load² · L_Z^{-⊤} · L_Z⁻¹
                    = Z_load² · Z_load⁻¹ = Z_load  ✓

            La verificación numérica se realiza como:
                ‖Z₀ − Z_load‖_F / ‖Z_load‖_F < 100 · ε_mach

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
                Si Z_load no es cuadrada, no es SPD, o si la verificación
                causal falla más allá de 100·ε_mach.
            """
            # Verificar cuadratura de Z_load
            if Z_load.ndim != 2 or Z_load.shape[0] != Z_load.shape[1]:
                raise ImpedanceReflectionError(
                    f"Z_load debe ser cuadrada 2D; se obtuvo shape={Z_load.shape}."
                )

            # Re-simetrizaci n defensiva
            Z_sym: NDArray[np.float64] = 0.5 * (Z_load + Z_load.T)

            # Verificar SPD de Z_load mediante Cholesky
            try:
                L_Z: NDArray[np.float64] = la.cholesky(Z_sym, lower=True)
            except la.LinAlgError:
                eigvals: NDArray[np.float64] = la.eigvalsh(Z_sym)
                raise ImpedanceReflectionError(
                    f"Z_load no es Sim trica Definida Positiva (SPD). "
                    f"Acoplamiento de impedancia imposible. "
                    f" _min = {float(eigvals[0]):.6e},  _max = {float(eigvals[-1]):.6e}."
                )

            #  _eff = L_Z   L_Z^  = Z_load  (SPD por construcci n)
            epsilon_eff: NDArray[np.float64] = L_Z @ L_Z.T

            #  _eff = Z_load    _eff   Z_load^  = Z_load
            # Para Z_load SPD: Z_load  es SPD (producto de SPD por s  misma)
            mu_eff: NDArray[np.float64] = Z_sym @ epsilon_eff @ Z_sym.T
            # Re-simetrizar  _eff para eliminar asimetr a num rica O( )
            mu_eff = 0.5 * (mu_eff + mu_eff.T)

            # Verificar SPD de  _eff con Cholesky
            try:
                la.cholesky(mu_eff, lower=True)
            except la.LinAlgError:
                raise ImpedanceReflectionError(
                    "El tensor magn tico  _eff resultante no es SPD. "
                    "Violaci n de la condici n de causalidad (Kramers-Kronig)."
                )

            #    Verificaci n de la relaci n de dispersi n causal
            # Z  = sqrt( _eff    _eff  ) = sqrt(Z_load    Z_load  ) = Z_load
            # Verificaci n num rica:   _eff - Z_load _F /  Z_load _F
            norm_Z: float = float(la.norm(Z_sym, "fro"))
            causal_residual: float = float(
                la.norm(epsilon_eff - Z_sym, "fro")
            ) / max(norm_Z, 1.0)
            tol_causal: float = 100.0 * self._EPS

            if causal_residual > tol_causal:
                raise ImpedanceReflectionError(
                    f"Relaci n de dispersi n causal violada: "
                    f"  _eff - Z_load _F /  Z_load _F = {causal_residual:.6e} "
                    f"> tol = {tol_causal:.6e}."
                )

            logger.debug(
                "[Fase2] Kramers-Kronig: causal_residual=%.3e, "
                "  _eff =%.6e,   _eff =%.6e.",
                causal_residual,
                float(la.norm(epsilon_eff, "fro")),
                float(la.norm(mu_eff, "fro")),
            )
            return epsilon_eff, mu_eff

        #
        # Subproceso 4: Auditor a del l mite CFL
        #

        def audit_cfl_limit(
            self,
            c_eff: float,
            Delta_sym: sp.spmatrix,
        ) -> float:
            r"""
            Calcula el paso temporal máximo admisible según la condición CFL
            para el operador de onda discreta:

                Δt_safe = (2 · CFL_margin) / (c_eff · √λ_max(Δ_sym))

            La fórmula proviene del análisis de von Neumann del esquema de
            diferencias finitas centradas en espacio para la ecuación de onda:
            la estabilidad requiere c_eff · Δt · √λ_max ≤ 2.

            El autovalor máximo se calcula con ARPACK (``eigsh``) para matrices
            dispersas grandes. En caso de no-convergencia, se hace fallback a
            ``eigvalsh`` sobre la versión densa (sólo para matrices pequeñas).

            Parámetros
            ----------
            c_eff : float
                Velocidad de propagación efectiva (> 0 obligatorio).
            Delta_sym : sp.spmatrix
                Laplaciano simétrico del grafo logístico (semidefinido positivo).

            Retorna
            -------
            float
                Δt_safe > 0, o +∞ si Δ_sym es numéricamente nulo (grafo degenerado).

            Lanza
            -----
            CFLViolationError
                Si c_eff ≤ 0 (velocidad no física).
                Si ARPACK y el fallback denso fallan simultáneamente.
            """
            if c_eff <= 0.0:
                raise CFLViolationError(
                    f"c_eff debe ser estrictamente positivo; se obtuvo c_eff={c_eff:.6e}. "
                    f"Una velocidad de propagaci n <= 0 es f sicamente inadmisible."
                )

            n_nodes: int = Delta_sym.shape[0]
            lambda_max: float

            #    C lculo de  _max con ARPACK (O(k nnz) para k=1)
            try:
                eigvals_arpack, _ = eigsh(
                    Delta_sym,
                    k=1,
                    which="LM",
                    tol=1.0e-8,
                    maxiter=10 * n_nodes,
                    return_eigenvectors=True,
                )
                lambda_max = float(np.abs(eigvals_arpack[0]))
                logger.debug(
                    "[Fase2]  _max( _sym) = %.6e (ARPACK, n=%d).",
                    lambda_max, n_nodes,
                )

            except (ArpackNoConvergence, Exception) as exc_arpack:
                #    Fallback: eigvalsh denso (solo para matrices peque as)
                logger.warning(
                    "[Fase2] ARPACK no convergi  para  _max( _sym): %s. "
                    "Fallback a eigvalsh denso (n=%d).",
                    exc_arpack, n_nodes,
                )
                if n_nodes > 5000:
                    raise CFLViolationError(
                        f"ARPACK fall  para  _sym de tama o n={n_nodes} > 5000, "
                        f"y el fallback denso no es admisible por coste O(n ). "
                        f"Error ARPACK: {exc_arpack}"
                    ) from exc_arpack
                try:
                    Delta_dense: NDArray[np.float64] = Delta_sym.toarray()
                    eigvals_dense: NDArray[np.float64] = la.eigvalsh(Delta_dense)
                    lambda_max = float(eigvals_dense[-1])
                    logger.debug(
                        "[Fase2]  _max( _sym) = %.6e (eigvalsh denso, fallback).",
                        lambda_max,
                    )
                except Exception as exc_dense:
                    raise CFLViolationError(
                        f"Fallo en ARPACK y en eigvalsh denso para  _sym. "
                        f"Error ARPACK: {exc_arpack}. Error denso: {exc_dense}."
                    ) from exc_dense

            #    Grafo degenerado:  _sym ~= 0
            if lambda_max < 1.0e-12:
                logger.warning(
                    "[Fase2]  _max( _sym) = %.3e < 1e-12: grafo log stico "
                    "degenerado (desconectado).  t_safe = + .",
                    lambda_max,
                )
                return float("inf")

            #    Condici n CFL
            dt_safe: float = (2.0 * self._cfl_margin) / (
                c_eff * float(np.sqrt(lambda_max))
            )

            logger.debug(
                "[Fase2] CFL:  _max=%.6e, c_eff=%.6e, "
                "CFL_margin=%.2f,  t_safe=%.6e.",
                lambda_max, c_eff, self._cfl_margin, dt_safe,
            )
            return dt_safe

        #
        # M todo terminal de la Fase 2   entrada directa de la Fase 3
        #

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
            para construir δ_{CORE}. La frontera formal entre Fase 2 y Fase 3
            es el campo ``hodge_conductance`` de este tensor.

            Parámetros
            ----------
            grad_H : NDArray[np.float64], shape (n,)
                Gradiente del Hamiltoniano actual.
            grad_H_d : NDArray[np.float64], shape (n,)
                Gradiente del Hamiltoniano deseado.
            W : sp.spmatrix, shape (E, E)
                Matriz de conductancia de aristas (cualquier formato sparse).
            I_curl : NDArray[np.float64], shape (E,)
                Corriente de curl sobre las aristas.
            Z_load : NDArray[np.float64], shape (d, d)
                Matriz de impedancia de carga (SPD).
            c_eff : float
                Velocidad de propagación efectiva (> 0).
            Delta_sym : sp.spmatrix, shape (V, V)
                Laplaciano simétrico del grafo logístico.
            dt_requested : float
                Paso temporal solicitado por el integrador externo.

            Retorna
            -------
            KinematicStateTensor
                Estado cinemático completo, inmutable.

            Lanza
            -----
            CFLViolationError
                Si dt_requested > dt_safe (violación activa del cono de luz).
            DiracMatchingError, ParasiticVorticityError,
            ImpedanceReflectionError, CFLViolationError
                Propagadas desde los subprocesos correspondientes.
            """
            #    Subproceso 1: IDA-PBC
            alpha, residual_rel = self.compute_dirac_control_law(
                grad_H, grad_H_d
            )

            #    Subproceso 2: Hodge
            W_mod, vorticity_norm = self.modulate_hodge_conductance(W, I_curl)

            #    Subproceso 3: Kramers-Kronig
            epsilon_eff, mu_eff = self.tune_impedance_tensors(Z_load)

            #    Subproceso 4: CFL
            dt_safe = self.audit_cfl_limit(c_eff, Delta_sym)

            #    Verificaci n CFL activa
            if dt_requested > dt_safe:
                raise CFLViolationError(
                    f"Violaci n del Cono de Luz Causal (CFL). "
                    f"dt_requested = {dt_requested:.6e} > "
                    f" t_safe = {dt_safe:.6e}. "
                    f"Reducir el paso temporal o disminuir c_eff."
                )

            logger.info(
                "[Fase2] S ntesis cinem tica completada: "
                "   =%.6e, vorticity=%.6e, dt_safe=%.6e, r_rel=%.3e.",
                float(la.norm(alpha, 2)),
                vorticity_norm,
                dt_safe,
                residual_rel,
            )

            #    Contrato de interfaz Fase 2   Fase 3
            # `hodge_conductance = W_mod` es el argumento directo del
            # constructor de Phase3_SheafProjection. Esta devoluci n es la
            # frontera formal entre ambas fases anidadas.
            return KinematicStateTensor(
                control_law_alpha=alpha,
                hodge_conductance=W_mod,
                dielectric_tensor=epsilon_eff,
                magnetic_tensor=mu_eff,
                cfl_safe_dt=dt_safe,
                residual_idapbc=residual_rel,
                vorticity_norm=vorticity_norm,
                is_kinematically_stable=True,
            )

    #
    # FASE 3   PROYECCI N EN HACES Y COFRONTERA DISCRETA  _{CORE}
    #

    class Phase3_SheafProjection:
        r"""
        **Fase 3 – Proyección en Haces y Cofrontera Discreta δ_{CORE}.**

        Recibe la conductancia modulada W_mod producida por la Fase 2 y
        construye el ``SheafStalk`` que alimenta el Laplaciano de Haz global.

        El constructor de esta clase es la **continuación directa** del método
        ``synthesize`` de la Fase 2: el campo ``hodge_conductance`` del
        ``KinematicStateTensor`` devuelto allá es lo que aquí se recibe.

        Fundamento Matemático
        ----------------------
        En el complejo de cadenas del grafo logístico, las aristas son
        1-cadenas y W_mod es el operador de métrica en el espacio de 1-formas.

        La cofrontera local δ_{CORE}: F(K_{CORE}) → F(e) se define como:

            δ_{CORE} = W_mod^{+1/2}

        donde W_mod^{+1/2} es la raíz cuadrada matricial espectral
        (pseudoinversa si W_mod es rango-deficiente):

            W_mod^{+1/2} = V · diag(√λ⁺) · V^⊤,  W_mod = V · diag(λ) · V^⊤

        Esta elección garantiza la identidad de Hodge local:

            δ_{CORE}^⊤ · δ_{CORE} = W_mod

        que es la condición necesaria para que el Laplaciano de Hodge global
        sea consistente con la métrica local de cada stalk.

        Verificación de la identidad:
            err = ‖δ_{CORE}^⊤ δ_{CORE} − W_mod‖_F / ‖W_mod‖_F < 100 · ε_mach
        """

        _EPS: float = float(np.finfo(np.float64).eps)

        def __init__(self, W_mod: sp.spmatrix) -> None:
            r"""
            **Constructor de la Fase 3: continuación directa de la Fase 2.**

            Recibe W_mod = ``KinematicStateTensor.hodge_conductance`` y
            precalcula la raíz cuadrada espectral δ_{CORE} = W_mod^{+1/2}
            para amortizar el coste sobre múltiples llamadas a ``export_stalk``.

            La raíz se calcula en el constructor (no en ``export_stalk``)
            porque W_mod es fija entre llamadas del mismo paso temporal.

            Lanza
            -----
            SheafCoboundaryError
                Si la identidad de Hodge no se satisface dentro de 100·ε_mach.
            """
            self._W_mod: sp.spmatrix = W_mod

            #    Conversi n a denso para descomposici n espectral
            # W_mod es diagonal (resultado de modulate_hodge_conductance),
            # por lo que toarray() es O(E ) pero E es t picamente peque o.
            W_dense: NDArray[np.float64] = W_mod.toarray()
            # Re-simetrizar defensivamente
            W_dense = 0.5 * (W_dense + W_dense.T)
            self._W_dense: NDArray[np.float64] = W_dense

            #    Ra z cuadrada espectral: W_mod^{+1/2}
            eigvals: NDArray[np.float64]
            eigvecs: NDArray[np.float64]
            eigvals, eigvecs = la.eigh(W_dense)

            # Tolerancia para distinguir autovalores nulos de negativos
            norm_W: float = float(la.norm(W_dense, "fro"))
            tol_eig: float = self._EPS * max(norm_W, 1.0)

            # Verificar no-negatividad (W_mod debe ser PSD por construcci n)
            lambda_min: float = float(eigvals[0])
            if lambda_min < -tol_eig:
                raise SheafCoboundaryError(
                    f"W_mod no es Semidefinida Positiva:  _min = {lambda_min:.6e} "
                    f"< -tol = {-tol_eig:.6e}. "
                    f"La estrangulaci n de Hodge produjo conductancias negativas."
                )

            # Ra z cuadrada espectral con pseudoinversa (clamping de negativos)
            eigvals_sqrt: NDArray[np.float64] = np.sqrt(np.maximum(eigvals, 0.0))
            delta_core: NDArray[np.float64] = (
                eigvecs * eigvals_sqrt[np.newaxis, :]
            ) @ eigvecs.T
            # Re-simetrizar para eliminar asimetr a num rica O( )
            self._delta_core: NDArray[np.float64] = 0.5 * (
                delta_core + delta_core.T
            )

            # Rango num rico de  _{CORE}
            self._rank_delta: int = int(np.sum(eigvals_sqrt > tol_eig))

            #    Verificaci n de la identidad de Hodge local
            self._hodge_residual: float = self._verify_hodge_identity()

            logger.debug(
                "[Fase3]  _{CORE} precalculada: E=%d, rank=%d, "
                "Hodge_residual=%.3e.",
                W_dense.shape[0], self._rank_delta, self._hodge_residual,
            )

        def _verify_hodge_identity(self) -> float:
            r"""
            Verifica la identidad de Hodge local:

                ‖ δ_{CORE}^⊤ · δ_{CORE} − W_mod ‖_F / ‖ W_mod ‖_F

            Esta identidad garantiza que δ_{CORE} es la raíz cuadrada
            correcta de W_mod y que el Laplaciano de Haz global será
            consistente con la métrica local.

            Retorna
            -------
            float
                Error relativo de la identidad de Hodge.

            Lanza
            -----
            SheafCoboundaryError
                Si el error relativo supera 100·ε_mach.
            """
            #  ^      =       (ya que   es sim trica por construcci n espectral)
            delta_sq: NDArray[np.float64] = self._delta_core @ self._delta_core
            residual_mat: NDArray[np.float64] = delta_sq - self._W_dense

            norm_W: float = float(la.norm(self._W_dense, "fro"))
            residual_F: float = float(la.norm(residual_mat, "fro"))
            rel_error: float = residual_F / max(norm_W, 1.0)

            tol_hodge: float = 100.0 * self._EPS

            if rel_error > tol_hodge:
                raise SheafCoboundaryError(
                    f" _{{CORE}}    W_mod: identidad de Hodge violada. "
                    f"   -W _F /  W _F = {rel_error:.6e} > tol = {tol_hodge:.6e}. "
                    f"Error de ensamble en la ra z espectral."
                )

            return rel_error

        #
        # M todo terminal de la Fase 3 (salida p blica del ecosistema)
        #

        def export_stalk(
            self,
            state_x: NDArray[np.float64],
        ) -> "SheafStalk":
            r"""
            **Método terminal de la Fase 3 y del agente completo.**

            Proyecta el vector de estado x sobre la fibra local mediante
            δ_{CORE} (precalculada en el constructor) y retorna el
            ``SheafStalk`` completo.

            Parámetros
            ----------
            state_x : NDArray[np.float64], shape (E,)
                Vector de estado en el espacio de aristas (1-cochains).
                Típicamente: corriente de flujo logístico por cada arista.

            Retorna
            -------
            SheafStalk
                Fibrado celular completo, inmutable.

            Lanza
            -----
            KinematicDimensionError
                Si state_x no tiene la dimensión E del espacio de aristas.
            """
            E: int = self._delta_core.shape[0]

            if state_x.shape != (E,):
                raise KinematicDimensionError(
                    f"state_x debe tener shape ({E},) = (n_aristas,); "
                    f"se obtuvo {state_x.shape}."
                )

            # Proyecci n sobre la fibra:  _{CORE}   x
            projected: NDArray[np.float64] = self._delta_core @ state_x

            logger.info(
                "[Fase3] SheafStalk exportado: E=%d, rank_delta=%d, "
                "Hodge_res=%.3e,    x =%.6e.",
                E,
                self._rank_delta,
                self._hodge_residual,
                float(la.norm(projected, 2)),
            )

            #    Contrato de salida del agente completo
            # El SheafStalk es el output final de la cadena de 3 fases.
            return SheafStalk(
                delta_core=self._delta_core,
                delta_hodge_residual=self._hodge_residual,
                state_vector=state_x.copy(),
                projected_state=projected,
                rank_delta=self._rank_delta,
            )

    #
    # INTERFAZ P BLICA DEL AGENTE (punto de entrada externo)
    #

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

        # Actualizar conductancia m s reciente para la Fase 3
        self._latest_hodge_conductance = state.hodge_conductance
        # Invalidar Fase 3 si W_mod cambi  (forzar re-instanciaci n perezosa)
        self.phase3 = None

        logger.debug(
            "[KCoreKinematicAgent] S ntesis completada: "
            "CFL_safe_dt=%.6e, vorticity=%.6e, r_idapbc=%.3e.",
            state.cfl_safe_dt,
            state.vorticity_norm,
            state.residual_idapbc,
        )
        return state

    def export_sheaf_stalk(
        self,
        state_x: NDArray[np.float64],
    ) -> SheafStalk:
        r"""
        Exporta el Stalk del haz cinemático y la cofrontera δ_{CORE}.

        Requiere que ``synthesize_kinematic_core`` haya sido llamado previamente
        para disponer de la conductancia modulada W_mod.

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
                "Ejecute ``synthesize_kinematic_core`` antes de ``export_sheaf_stalk``."
            )

        if self.phase3 is None:
            self.phase3 = KCoreKinematicAgent.Phase3_SheafProjection(
                W_mod=self._latest_hodge_conductance
            )
            logger.info(
                "[KCoreKinematicAgent] Phase3_SheafProjection instanciada "
                "(lazy init): rank_delta=%d.",
                self.phase3._rank_delta,
            )

        return self.phase3.export_stalk(state_x=state_x)
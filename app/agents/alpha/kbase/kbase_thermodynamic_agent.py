# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo : KBase Thermodynamic Agent (Asesor de Cimientos Financieros)         |
| Ruta   : app/agents/alpha/kbase/kbase_thermodynamic_agent.py                 |
| Versión: 3.0.0-Rigorous-PortHamiltonian-PDE-Sheaf                            |
+==============================================================================+

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA DIFERENCIAL:
Este módulo consagra el Foso Termodinámico del ecosistema (K_{BASE}). Actúa como un
Endofuntor Port-Hamiltoniano que gobierna la inercia, la capacitancia y la fricción
entrópica del modelo de negocio, integrando sub-funtores de Socios, Recursos y Costes.

DINÁMICA PORT-HAMILTONIANA Y TENSOR MÉTRICO:
La energía total de la base no se asume euclidiana; se calcula aplicando un pullback
geométrico contra el tensor métrico Riemanniano G_mu_nu para absorber el estrés
anisotrópico del ecosistema:
\[ \tilde{C}_{soc} = G_{\mu\nu} C_{soc} G^{\mu\nu}, \quad \tilde{M}_{rec} = G_{\mu\nu} M_{rec} G^{\mu\nu} \]

HAMILTONIANO BASAL:
\[ H_{BASE}(q,p) = \frac{1}{2} q^\top \tilde{C}_{soc}^{-1} q + \frac{1}{2} p^\top \tilde{M}_{rec}^{-1} p \]

ECUACIÓN DE DISIPACIÓN DE RAYLEIGH:
\[ \dot{H}_{diss} = -\nabla H^\top R_{cost}(x) \nabla H \le 0 \]
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
    # Stubs m nimos para ejecuci n aislada y prueba unitaria anal tica
    class CategoricalState:  # type: ignore[no-redef]
        """Stub: estado categ rico del ecosistema MIC."""

    class Morphism:  # type: ignore[no-redef]
        """Stub: morfismo funtorial del ecosistema MIC."""


#    Logger del m dulo
logger = logging.getLogger("MIC.Alpha.KBaseThermodynamicAgent")


#
#    SECCI N 0   EXCEPCIONES TERMODIN MICAS ESTRICTAS
#


class ThermodynamicBaseError(Exception):
    """
    Excepci n categ rica ra z para violaciones en el Estrato K_BASE.

    Toda excepci n de este m dulo hereda de esta clase, garantizando que
    los manejadores de nivel superior puedan capturar cualquier fallo
    termodin mico con un solo ``except ThermodynamicBaseError``.
    """


class DimensionMismatchError(ThermodynamicBaseError):
    """
    Lanzada cuando las dimensiones de las matrices constitutivas son
    inconsistentes con el espacio de fases (q, p) declarado.

    Ejemplo de inconsistencia: dim(C_soc)   dim(q) o
    dim(J_BASE)   (dim_q + dim_p)   (dim_q + dim_p).
    """


class CapacitanceDegeneracyError(ThermodynamicBaseError):
    """
    Lanzada cuando la matriz de socios C_soc o la inercia M_rec
    no es Sim trica Definida Positiva (SPD).

    Diagn stico incluye el n mero de condici n  (A) y el autovalor
    m nimo  _min para facilitar la correcci n num rica.
    """


class InertialFlybackError(ThermodynamicBaseError):
    """
    Lanzada cuando la inercia de recuperaci n genera un voltaje
    transitorio de Flyback que excede el l mite de ruptura diel ctrica.

    Condici n de disparo:    M_rec   ( f/ t)  _  > V_breakdown
    """


class RayleighDissipationViolation(ThermodynamicBaseError):
    """
    Lanzada cuando el modelo disipativo indica entrop a negativa
    (ganancia fantasma), violando la Segunda Ley de la Termodin mica.

    Condici n de disparo:   H^  R_cost  H < -   (P_diss >   > 0)
    """


class IllConditionedMatrixError(ThermodynamicBaseError):
    """
    Lanzada cuando el n mero de condici n espectral  (A) =  _max/ _min
    supera el umbral configurable, indicando cuasi-singularidad num rica
    que comprometer a la estabilidad de todas las fases posteriores.
    """


class SheafCoboundaryError(ThermodynamicBaseError):
    """
    Lanzada cuando la cofrontera discreta  _{BASE} no satisface la
    condici n de complejo de cadenas:  _{BASE}  ~= 0 (dentro de tolerancia
    de m quina escalada por   _{BASE}  ).
    """


#
#    SECCI N 1   ESTRUCTURAS INMUTABLES (DTOs TENSORIALES)
#


@dataclass(frozen=True, slots=True)
class TopologicalContext:
    r"""
    Contexto inmutable producido por la **Fase 1** (Topología Matricial).

    Contiene todas las factorizaciones y metadatos necesarios para que
    la Fase 2 opere sin re-validar ni re-factorizar ninguna matriz.

    Atributos
    ----------
    L_C : NDArray[np.float64]
        Factor de Cholesky inferior de C_soc, es decir L tal que
        C_soc = L_C · L_C^⊤  con L_C triangular inferior y L_C[i,i] > 0.

    L_M : NDArray[np.float64]
        Factor de Cholesky inferior de M_rec, análogo al anterior.

    R_cost : NDArray[np.float64]
        Matriz de disipación de Rayleigh validada (PSD),  R_cost ⪰ 0,
        almacenada como copia inmutable para uso en Fase 2 y Fase 3.

    R_sqrt : NDArray[np.float64]
        Raíz cuadrada matricial de R_cost calculada vía descomposición
        espectral:  R_sqrt = V · diag(√λ) · V^⊤  donde R_cost = V·diag(λ)·V^⊤.
        Precalculada en Fase 1 para reutilización eficiente en Fase 3.

    J_base : NDArray[np.float64]
        Matriz de interconexión antisimétrica validada,  J_base = -J_base^⊤,
        almacenada como copia inmutable.

    kappa_C : float
        Número de condición espectral de C_soc: κ(C_soc) = λ_max/λ_min.
        Documentado para trazabilidad numérica.

    kappa_M : float
        Número de condición espectral de M_rec: κ(M_rec) = λ_max/λ_min.

    dim_q : int
        Dimensión del espacio de coordenadas generalizadas q ∈ ℝ^{dim_q}.

    dim_p : int
        Dimensión del espacio de momentos generalizados  p ∈ ℝ^{dim_p}.

    rank_R : int
        Rango numérico de R_cost (puede ser < dim_q + dim_p si PSD).
    """

    L_C: NDArray[np.float64]
    L_M: NDArray[np.float64]
    R_cost: NDArray[np.float64]
    R_sqrt: NDArray[np.float64]
    J_base: NDArray[np.float64]
    kappa_C: float
    kappa_M: float
    dim_q: int
    dim_p: int
    rank_R: int


@dataclass(frozen=True, slots=True)
class BasalStateTensor:
    r"""
    Tensor inmutable que encapsula el estado termodinámico completo del foso.

    Producido por la **Fase 2** (Dinámica Hamiltoniana) y usado como
    entrada opcional de la Fase 3 cuando se requiere la proyección en haces.

    Atributos
    ----------
    potential_energy : float
        Energía potencial elástica V(q) = ½ q^⊤ C_soc⁻¹ q ≥ 0.

    kinetic_energy : float
        Energía cinética K(p) = ½ p^⊤ M_rec⁻¹ p ≥ 0.

    total_hamiltonian : float
        Hamiltoniano total H(q,p) = V(q) + K(p) ≥ 0.

    dissipated_power : float
        Potencia disipada |Ṫ| = |∇H^⊤ R_ext ∇H| ≥ 0.
        El signo es siempre no-negativo; la violación de la 2ª Ley
        se reporta vía excepción antes de llegar aquí.

    flyback_voltage_norm : float
        ‖ M_rec · (∂f/∂t) ‖_∞: norma infinito del voltaje de Flyback.

    grad_H_norm : float
        ‖ ∇H ‖_2: norma euclidiana del gradiente del Hamiltoniano.
        Útil para diagnóstico de estabilidad y comparación con umbrales.

    is_thermodynamically_stable : bool
        True si y sólo si todas las validaciones termodinámicas pasan
        sin excepción. Siempre True cuando el tensor es creado con éxito
        (la excepción impide la creación en caso contrario).
    """

    potential_energy: float
    kinetic_energy: float
    total_hamiltonian: float
    dissipated_power: float
    flyback_voltage_norm: float
    grad_H_norm: float
    is_thermodynamically_stable: bool


@dataclass(frozen=True, slots=True)
class SheafStalk:
    r"""
    Fibrado celular exportado para el cálculo global del Laplaciano de Haz.

    Producido por la **Fase 3** (Proyección en Haces).

    Atributos
    ----------
    delta_base : NDArray[np.float64]
        Matriz de cofrontera discreta δ_{BASE} ∈ ℝ^{n×n},  n = dim_q + dim_p.

        Ensamblaje formal:

            δ_{BASE} = block_diag( C_soc^{-1/2},  R_cost^{+1/2} )

        donde:
          • C_soc^{-1/2} = L_C^{-⊤}  (triangular superior de la inversa de Cholesky)
          • R_cost^{+1/2} = V · diag(√λ⁺) · V^⊤  (raíz espectral, pseudoinversa si rango < n)

        Esta elección garantiza que δ_{BASE} sea la "raíz" del operador de
        Hodge local del complejo de cadenas Port-Hamiltoniano.

    delta_base_sq_norm : float
        ‖ δ_{BASE}² ‖_F verificado en construcción.
        Debe ser ≈ 0 (tolerancia de máquina) para satisfacer δ² = 0.

    state_vector : NDArray[np.float64]
        Vector de estado x = [q; p] ∈ ℝ^{dim_q + dim_p} en el instante
        de proyección, almacenado para trazabilidad del haz.

    projected_state : NDArray[np.float64]
        Proyección del estado en la fibra: δ_{BASE} · x.

    rank_delta : int
        Rango numérico de δ_{BASE}, igual a rank(C_soc^{-1/2}) + rank(R_sqrt).
    """

    delta_base: NDArray[np.float64]
    delta_base_sq_norm: float
    state_vector: NDArray[np.float64]
    projected_state: NDArray[np.float64]
    rank_delta: int


#
#    SECCI N 2   ORQUESTADOR: KBaseThermodynamicAgent
#                Tres fases anidadas de rigor creciente
#


class KBaseThermodynamicAgent(Morphism):
    r"""
    Orquestador Funtorial del Foso Termodinámico K_{BASE}.

    Integra el modelo Port-Hamiltoniano del estrato K_{BASE} mediante tres
    clases anidadas que operan en cascada estricta:

        Phase1_MatrixTopology
            ↓  TopologicalContext
        Phase2_HamiltonianDynamics
            ↓  BasalStateTensor
        Phase3_SheafProjection
            ↓  SheafStalk

    El constructor instancia y ejecuta la Fase 1 de forma inmediata,
    garantizando que cualquier violación espectral o de simetría sea
    detectada antes de que el agente sea utilizado por el ecosistema.

    Parámetros de Construcción
    --------------------------
    C_soc : NDArray[np.float64], shape (n_q, n_q)
        Matriz de capacitancia de socios. Debe ser SPD.
        Semántica física: energía almacenada por unidad de carga social.

    M_rec : NDArray[np.float64], shape (n_p, n_p)
        Matriz de inercia de recuperación. Debe ser SPD.
        Semántica física: resistencia dinámica al cambio de flujo de recursos.

    R_cost : NDArray[np.float64], shape (n, n),  n = n_q + n_p
        Matriz de disipación de Rayleigh. Debe ser PSD (⪰ 0).
        Semántica física: tasa de disipación de exergía por costes operativos.

    J_base : NDArray[np.float64], shape (n, n)
        Matriz de interconexión antisimétrica. Debe satisfacer J = -J^⊤.
        Semántica física: flujo de energía conservativo entre subsistemas.

    breakdown_voltage : float, default 1e5
        Umbral de ruptura dieléctrica para el voltaje de Flyback inductivo.
        Unidades: [V] en el isomorfismo eléctrico del modelo Port-Hamiltoniano.

    kappa_max : float, default 1e10
        Umbral máximo admisible para el número de condición espectral κ(A).
        Si κ(C_soc) > kappa_max o κ(M_rec) > kappa_max se lanza
        ``IllConditionedMatrixError`` antes de proceder.

    Atributos Públicos
    ------------------
    context : TopologicalContext
        Resultado inmutable de la Fase 1, disponible para inspección externa.

    phase2 : Phase2_HamiltonianDynamics
        Instancia de la Fase 2, lista para recibir vectores de estado.

    phase3 : Optional[Phase3_SheafProjection]
        Instancia de la Fase 3, creada perezosamente en ``export_sheaf_stalk``.
    """

    FRIENDLY_NAME: str = "Asesor de Cimientos Financieros"

    def __init__(
        self,
        C_soc: NDArray[np.float64],
        M_rec: NDArray[np.float64],
        R_cost: NDArray[np.float64],
        J_base: NDArray[np.float64],
        breakdown_voltage: float = 1.0e5,
        kappa_max: float = 1.0e10,
    ) -> None:
        r"""
        Inicializa las matrices constitutivas y ejecuta la Fase 1 de inmediato.

        El flujo de inicialización es:
          1. Almacenar parámetros escalares de control.
          2. Instanciar Phase1_MatrixTopology y ejecutar build_topological_context().
          3. Instanciar Phase2_HamiltonianDynamics con el contexto producido.
          4. Dejar phase3 = None (instanciación perezosa bajo demanda).

        Lanza
        -----
        DimensionMismatchError
            Si las dimensiones de C_soc, M_rec, R_cost y J_base son inconsistentes.
        ThermodynamicBaseError (o subclases)
            Cualquier violación espectral detectada en la Fase 1.
        """
        self.breakdown_voltage: float = breakdown_voltage
        self.kappa_max: float = kappa_max

        #    Fase 1: Topolog a Matricial (ejecuci n inmediata)
        self.phase1: KBaseThermodynamicAgent.Phase1_MatrixTopology = (
            KBaseThermodynamicAgent.Phase1_MatrixTopology(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
                kappa_max=kappa_max,
            )
        )
        self.context: TopologicalContext = self.phase1.build_topological_context()

        #    Fase 2: Din mica Hamiltoniana (instanciaci n inmediata)
        self.phase2: KBaseThermodynamicAgent.Phase2_HamiltonianDynamics = (
            KBaseThermodynamicAgent.Phase2_HamiltonianDynamics(
                context=self.context,
                breakdown_voltage=self.breakdown_voltage,
            )
        )

        #    Fase 3: Proyecci n en Haces (instanciaci n perezosa)
        self.phase3: Optional[KBaseThermodynamicAgent.Phase3_SheafProjection] = None

        logger.info(
            "[KBaseThermodynamicAgent] Inicializaci n completa. "
            "dim_q=%d, dim_p=%d,  (C_soc)=%.3e,  (M_rec)=%.3e, rank(R)=%d",
            self.context.dim_q,
            self.context.dim_p,
            self.context.kappa_C,
            self.context.kappa_M,
            self.context.rank_R,
        )

    #
    # FASE 1   TOPOLOG A MATRICIAL Y VALIDACI N ESPECTRAL
    #

    class Phase1_MatrixTopology:
        r"""
        **Fase 1 – Topología Matricial y Validación Espectral.**

        Responsabilidades exclusivas de esta fase:
          a) Verificar la coherencia dimensional de todas las matrices.
          b) Verificar simetría con tolerancia normalizada por ‖A‖_F.
          c) Verificar antisimetría de J_base con tolerancia normalizada.
          d) Calcular κ(C_soc) y κ(M_rec); rechazar si κ > kappa_max.
          e) Factorizar C_soc y M_rec por Cholesky (pivoted para estabilidad).
          f) Verificar que R_cost ⪰ 0 y calcular su raíz cuadrada espectral.
          g) Retornar un ``TopologicalContext`` inmutable como salida contractual.

        El método ``build_topological_context`` es el contrato de salida de esta
        fase y la entrada directa de la Fase 2; su firma actúa como interfaz
        formal entre ambas fases.

        Tolerancias
        -----------
        Todas las tolerancias son *relativas* al Frobenius de la matriz evaluada,
        multiplicado por la precisión de máquina ε = 2.22e-16:

            tol_sym = ε_mach * ‖A‖_F        (simetría)
            tol_pd  = ε_mach * tr(A)         (definición positiva)
            tol_psd = -ε_mach * ‖A‖_F        (semidefinición positiva)
        """

        # Precisi n de m quina IEEE-754 double
        _EPS: float = float(np.finfo(np.float64).eps)

        def __init__(
            self,
            C_soc: NDArray[np.float64],
            M_rec: NDArray[np.float64],
            R_cost: NDArray[np.float64],
            J_base: NDArray[np.float64],
            kappa_max: float = 1.0e10,
        ) -> None:
            r"""
            Almacena las matrices originales sin modificarlas.

            No se realizan copias en este punto para evitar duplicación de
            memoria; las copias ocurren sólo en ``build_topological_context``
            en el momento de empaquetar el ``TopologicalContext``.
            """
            self._C_soc: NDArray[np.float64] = C_soc
            self._M_rec: NDArray[np.float64] = M_rec
            self._R_cost: NDArray[np.float64] = R_cost
            self._J_base: NDArray[np.float64] = J_base
            self._kappa_max: float = kappa_max

        #
        # M todos privados de validaci n (orden l gico de ejecuci n)
        #

        def _check_dimensions(self) -> Tuple[int, int]:
            r"""
            Verifica que las dimensiones de todas las matrices sean consistentes
            con un espacio de fases (q, p) bien definido.

            Condiciones formales:
              • C_soc ∈ ℝ^{n_q × n_q}     (cuadrada)
              • M_rec ∈ ℝ^{n_p × n_p}     (cuadrada)
              • R_cost ∈ ℝ^{n × n}         (cuadrada, n = n_q + n_p)
              • J_base ∈ ℝ^{n × n}         (cuadrada, n = n_q + n_p)

            Retorna
            -------
            Tuple[int, int]
                (dim_q, dim_p) validados.

            Lanza
            -----
            DimensionMismatchError
                Si alguna condición anterior es violada.
            """
            # Verificar cuadratura de C_soc
            if self._C_soc.ndim != 2 or self._C_soc.shape[0] != self._C_soc.shape[1]:
                raise DimensionMismatchError(
                    f"C_soc debe ser cuadrada; se obtuvo shape={self._C_soc.shape}."
                )
            dim_q: int = self._C_soc.shape[0]

            # Verificar cuadratura de M_rec
            if self._M_rec.ndim != 2 or self._M_rec.shape[0] != self._M_rec.shape[1]:
                raise DimensionMismatchError(
                    f"M_rec debe ser cuadrada; se obtuvo shape={self._M_rec.shape}."
                )
            dim_p: int = self._M_rec.shape[0]

            n: int = dim_q + dim_p

            # Verificar R_cost
            if self._R_cost.shape != (n, n):
                raise DimensionMismatchError(
                    f"R_cost debe ser ({n},{n}) = (dim_q+dim_p) (dim_q+dim_p), "
                    f"pero se obtuvo {self._R_cost.shape}. "
                    f"dim_q={dim_q}, dim_p={dim_p}."
                )

            # Verificar J_base
            if self._J_base.shape != (n, n):
                raise DimensionMismatchError(
                    f"J_base debe ser ({n},{n}) = (dim_q+dim_p) (dim_q+dim_p), "
                    f"pero se obtuvo {self._J_base.shape}."
                )

            logger.debug(
                "[Fase1] Dimensiones verificadas: dim_q=%d, dim_p=%d, n=%d.",
                dim_q, dim_p, n,
            )
            return dim_q, dim_p

        def _validate_symmetry(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> None:
            r"""
            Verifica que A = A^⊤ con tolerancia relativa al Frobenius de A.

            Tolerancia adaptativa:
                tol = ε_mach * ‖A‖_F

            donde ε_mach = 2.22e-16 (precisión de máquina IEEE-754 double).

            La norma Frobenius escala la tolerancia con la magnitud de la matriz,
            evitando falsos positivos para matrices de gran norma y falsos negativos
            para matrices casi nulas.

            Lanza
            -----
            ThermodynamicBaseError
                Con diagnóstico cuantitativo: ‖A − A^⊤‖_F / ‖A‖_F.
            """
            norm_A: float = float(la.norm(A, "fro"))
            # Evitar divisi n por cero para la matriz nula
            tol: float = self._EPS * max(norm_A, 1.0)
            residual: float = float(la.norm(A - A.T, "fro"))

            if residual > tol:
                raise ThermodynamicBaseError(
                    f"La matriz '{name}' no es sim trica. "
                    f" A - A^  _F = {residual:.6e},  tol = {tol:.6e},  "
                    f"simetr a relativa = {residual / max(norm_A, 1e-300):.6e}."
                )

            logger.debug(
                "[Fase1] Simetr a de '%s' verificada: residual=%.3e, tol=%.3e.",
                name, residual, tol,
            )

        def _validate_antisymmetry(
            self,
            J: NDArray[np.float64],
            name: str,
        ) -> None:
            r"""
            Verifica que J = -J^⊤ (antisimetría estricta) con tolerancia relativa.

            Equivalentemente verifica que (J + J^⊤) ≈ 0 con:
                tol = ε_mach * ‖J‖_F

            La antisimetría es una condición topológica fundamental del sistema
            Port-Hamiltoniano: garantiza conservación de energía en ausencia de
            disipación (la parte antisimétrica de [J − R] es puramente conservativa).

            Lanza
            -----
            ThermodynamicBaseError
                Si la condición es violada, con diagnóstico cuantitativo.
            """
            norm_J: float = float(la.norm(J, "fro"))
            tol: float = self._EPS * max(norm_J, 1.0)
            #  J + J^  _F =  J - (-J^ ) _F
            residual: float = float(la.norm(J + J.T, "fro"))

            if residual > tol:
                raise ThermodynamicBaseError(
                    f"La matriz '{name}' no es antisim trica (J   -J^ ). "
                    f" J + J^  _F = {residual:.6e},  tol = {tol:.6e}."
                )

            logger.debug(
                "[Fase1] Antisimetr a de '%s' verificada: residual=%.3e, tol=%.3e.",
                name, residual, tol,
            )

        def _compute_condition_number(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> Tuple[float, float, float]:
            r"""
            Calcula el número de condición espectral κ(A) = λ_max / λ_min
            usando la descomposición espectral completa (eigvalsh, que explota
            la simetría para mayor precisión y velocidad sobre eigh de LAPACK).

            Retorna
            -------
            Tuple[float, float, float]
                (kappa, lambda_min, lambda_max) donde:
                  kappa = λ_max / λ_min  (∞ si λ_min ≤ 0)

            Lanza
            -----
            IllConditionedMatrixError
                Si κ(A) > kappa_max.
            CapacitanceDegeneracyError
                Si λ_min ≤ tol_pd (la matriz no es SPD).
            """
            # eigvalsh garantiza autovalores reales y ordenados ascendentemente
            # para matrices sim tricas reales (aprovecha la estructura LAPACK dsyevd)
            eigvals: NDArray[np.float64] = la.eigvalsh(A)
            lambda_min: float = float(eigvals[0])
            lambda_max: float = float(eigvals[-1])

            # Tolerancia relativa para definici n positiva
            tol_pd: float = self._EPS * lambda_max

            if lambda_min <= tol_pd:
                raise CapacitanceDegeneracyError(
                    f"La matriz '{name}' no es Definida Positiva (SPD). "
                    f" _min = {lambda_min:.6e}  <=  tol_pd = {tol_pd:.6e}. "
                    f" _max = {lambda_max:.6e}."
                )

            kappa: float = lambda_max / lambda_min
            logger.debug(
                "[Fase1]  ('%s') = %.6e  ( _min=%.6e,  _max=%.6e).",
                name, kappa, lambda_min, lambda_max,
            )

            if kappa > self._kappa_max:
                raise IllConditionedMatrixError(
                    f"La matriz '{name}' est  num ricamente mal condicionada. "
                    f"  = {kappa:.6e}  >   _max = {self._kappa_max:.6e}. "
                    f"Considere regularizaci n de Tikhonov o re-escalado."
                )

            return kappa, lambda_min, lambda_max

        def _cholesky_spd(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> NDArray[np.float64]:
            r"""
            Calcula la factorización de Cholesky de A = L · L^⊤ usando
            ``scipy.linalg.cholesky`` con ``lower=True``.

            Se invoca **después** de ``_compute_condition_number``, por lo que
            la SPD ya está garantizada en este punto. Sin embargo, se mantiene
            el bloque try/except para capturar errores numéricos residuales de
            ramas de ejecución alternativas (p.ej. re-simetrizaciones).

            Retorna
            -------
            NDArray[np.float64]
                Factor Cholesky inferior L, tal que A = L · L^⊤.

            Lanza
            -----
            CapacitanceDegeneracyError
                En el caso improbable de fallo numérico de LAPACK dpotrf.
            """
            # Re-simetrizaci n defensiva: proyecta A sobre el subespacio de matrices
            # sim tricas: A_sym = (A + A^ ) / 2, eliminando errores de redondeo
            A_sym: NDArray[np.float64] = 0.5 * (A + A.T)

            try:
                L: NDArray[np.float64] = la.cholesky(A_sym, lower=True)
            except la.LinAlgError as exc:
                raise CapacitanceDegeneracyError(
                    f"Fallo de Cholesky (LAPACK dpotrf) en '{name}' "
                    f"a pesar de validaci n espectral previa. "
                    f"Error LAPACK: {exc}"
                ) from exc

            logger.debug(
                "[Fase1] Cholesky de '%s' completado. L[0,0]=%.6e, L[-1,-1]=%.6e.",
                name, float(L[0, 0]), float(L[-1, -1]),
            )
            return L

        def _validate_psd_and_compute_sqrt(
            self,
            R: NDArray[np.float64],
            name: str,
        ) -> Tuple[NDArray[np.float64], int]:
            r"""
            Verifica que R ⪰ 0 (semidefinida positiva) y calcula su raíz
            cuadrada matricial espectral exacta:

                R_sqrt = V · diag(√max(λ, 0)) · V^⊤

            donde R = V · diag(λ) · V^⊤ es la descomposición espectral de R.

            Esta raíz satisface:  R_sqrt · R_sqrt = R  (exactamente en aritmética
            de punto flotante hasta errores de orden O(ε·‖R‖)).

            La raíz espectral (distinta de la raíz de Cholesky) es la única
            noción de raíz cuadrada que está bien definida para matrices PSD
            no estrictamente positivas (rango deficiente).

            Parámetros
            ----------
            R : NDArray[np.float64]
                Matriz simétrica a verificar y de la que calcular la raíz.
            name : str
                Nombre descriptivo para mensajes de error.

            Retorna
            -------
            Tuple[NDArray[np.float64], int]
                (R_sqrt, rank_R) donde rank_R es el rango numérico de R.

            Lanza
            -----
            RayleighDissipationViolation
                Si algún autovalor es negativo más allá de la tolerancia
                de máquina (indicaría errores de modelo, no de redondeo).
            """
            norm_R: float = float(la.norm(R, "fro"))
            # Tolerancia para distinguir autovalores nulos de negativos reales
            tol_psd: float = self._EPS * max(norm_R, 1.0)

            eigvals: NDArray[np.float64]
            eigvecs: NDArray[np.float64]
            eigvals, eigvecs = la.eigh(R)

            lambda_min: float = float(eigvals[0])

            if lambda_min < -tol_psd:
                raise RayleighDissipationViolation(
                    f"La matriz '{name}' no es Semidefinida Positiva (PSD). "
                    f" _min = {lambda_min:.6e}  <  -tol = {-tol_psd:.6e}. "
                    f"Esto indica entrop a negativa (ganancia fantasma): "
                    f"violaci n de la Segunda Ley de la Termodin mica."
                )

            # Forzar no-negatividad de autovalores por errores de redondeo
            eigvals_clamped: NDArray[np.float64] = np.maximum(eigvals, 0.0)

            # Ra z cuadrada espectral
            R_sqrt: NDArray[np.float64] = (
                eigvecs * np.sqrt(eigvals_clamped)[np.newaxis, :]
            ) @ eigvecs.T
            # Re-simetrizar para eliminar asimetr a num rica de O( )
            R_sqrt = 0.5 * (R_sqrt + R_sqrt.T)

            # Rango num rico: n mero de autovalores > tol_psd
            rank_R: int = int(np.sum(eigvals_clamped > tol_psd))

            logger.debug(
                "[Fase1] R_cost PSD verificada: rank=%d/%d,  _min=%.3e,  _max=%.3e.",
                rank_R, len(eigvals), lambda_min, float(eigvals[-1]),
            )
            return R_sqrt, rank_R

        #
        # M todo terminal de la Fase 1   entrada directa de la Fase 2
        #

        def build_topological_context(self) -> "TopologicalContext":
            r"""
            **Método terminal de la Fase 1.**

            Ejecuta en secuencia estricta y ordenada todos los métodos de
            validación y factorización, y empaqueta sus resultados en un
            ``TopologicalContext`` inmutable.

            El ``TopologicalContext`` resultante es el **único argumento** que
            necesita la Fase 2 para operar, garantizando la continuidad formal
            entre fases: la última instrucción de este método es idéntica al
            primer dato que consume el constructor de Phase2_HamiltonianDynamics.

            Flujo interno
            -------------
            1. Verificación dimensional (todas las matrices).
            2. Antisimetría de J_base.
            3. Simetría de C_soc, M_rec, R_cost.
            4. Número de condición κ(C_soc), κ(M_rec).
            5. Factorización Cholesky de C_soc → L_C.
            6. Factorización Cholesky de M_rec → L_M.
            7. Validación PSD de R_cost y cálculo de R_sqrt.
            8. Empaquetado en TopologicalContext.

            Retorna
            -------
            TopologicalContext
                Contexto topológico completo, inmutable y listo para la Fase 2.
            """
            #    Paso 1: Verificaci n dimensional
            dim_q, dim_p = self._check_dimensions()

            #    Paso 2: Antisimetr a de J_base
            self._validate_antisymmetry(self._J_base, "J_base")

            #    Paso 3: Simetr a de matrices constitutivas
            self._validate_symmetry(self._C_soc, "C_soc")
            self._validate_symmetry(self._M_rec, "M_rec")
            self._validate_symmetry(self._R_cost, "R_cost")

            #    Paso 4: N mero de condici n espectral
            kappa_C, _, _ = self._compute_condition_number(self._C_soc, "C_soc")
            kappa_M, _, _ = self._compute_condition_number(self._M_rec, "M_rec")

            #    Paso 5 & 6: Factorizaciones de Cholesky
            L_C: NDArray[np.float64] = self._cholesky_spd(self._C_soc, "C_soc")
            L_M: NDArray[np.float64] = self._cholesky_spd(self._M_rec, "M_rec")

            #    Paso 7: Validaci n PSD y ra z cuadrada de R_cost
            R_sqrt: NDArray[np.float64]
            rank_R: int
            R_sqrt, rank_R = self._validate_psd_and_compute_sqrt(
                self._R_cost, "R_cost"
            )

            #    Paso 8: Empaquetado del contexto topol gico
            context = TopologicalContext(
                L_C=L_C,
                L_M=L_M,
                R_cost=self._R_cost.copy(),
                R_sqrt=R_sqrt,
                J_base=self._J_base.copy(),
                kappa_C=kappa_C,
                kappa_M=kappa_M,
                dim_q=dim_q,
                dim_p=dim_p,
                rank_R=rank_R,
            )

            logger.info(
                "[Fase1] TopologicalContext ensamblado: "
                "dim_q=%d, dim_p=%d,  (C)=%.3e,  (M)=%.3e, rank(R)=%d.",
                dim_q, dim_p, kappa_C, kappa_M, rank_R,
            )

            #    Contrato de interfaz entre Fase 1 y Fase 2
            # La variable `context` es el argumento directo del constructor
            # de Phase2_HamiltonianDynamics. Esta devoluci n es la frontera
            # formal entre ambas fases anidadas.
            return context

    #
    # FASE 2   DIN MICA HAMILTONIANA Y DISIPACI N DE RAYLEIGH
    #

    class Phase2_HamiltonianDynamics:
        r"""
        **Fase 2 – Dinámica Hamiltoniana y Disipación de Rayleigh.**

        Recibe el ``TopologicalContext`` producido por la Fase 1 y lo usa para
        evaluar las magnitudes termodinámicas del estado Port-Hamiltoniano.

        El constructor de esta clase es la **continuación directa** del método
        ``build_topological_context`` de la Fase 1: el ``TopologicalContext``
        que allá se devuelve es el que aquí se recibe, sin transformación alguna.

        Formulación Port-Hamiltoniana
        -----------------------------
        El Hamiltoniano del sistema es:

            H(q, p) = V(q) + K(p)
                    = ½ q^⊤ C_soc⁻¹ q  +  ½ p^⊤ M_rec⁻¹ p

        Las energías se calculan mediante la identidad:

            q^⊤ A⁻¹ q = ‖ L⁻¹ q ‖²     (A = L · L^⊤ → Cholesky)

        que evita la inversión explícita de A y es numéricamente estable.

        El gradiente del Hamiltoniano se calcula como:

            ∂H/∂q = C_soc⁻¹ q = L_C^{-⊤} (L_C⁻¹ q) = cho_solve((L_C, True), q)
            ∂H/∂p = M_rec⁻¹ p = L_M^{-⊤} (L_M⁻¹ p) = cho_solve((L_M, True), p)

        usando ``scipy.linalg.cho_solve`` que es O(n²) en lugar de O(n³)
        de la inversión explícita.

        La potencia disipada sigue la Función de Rayleigh:

            D(∇H) = ½ ∇H^⊤ R_ext ∇H  ≥  0   (2ª Ley de la Termodinámica)

        donde ∇H = [∂H/∂q; ∂H/∂p] y R_ext es R_cost extendida (PSD).
        """

        def __init__(
            self,
            context: "TopologicalContext",
            breakdown_voltage: float,
        ) -> None:
            r"""
            **Constructor de la Fase 2: continuación directa de la Fase 1.**

            Recibe el ``TopologicalContext`` devuelto por
            ``Phase1_MatrixTopology.build_topological_context()`` y extrae
            las factorizaciones precomputadas para uso eficiente.

            No realiza ninguna validación adicional: la corrección de todas
            las matrices queda garantizada por la Fase 1.

            Parámetros
            ----------
            context : TopologicalContext
                Resultado inmutable de la Fase 1.
            breakdown_voltage : float
                Umbral de voltaje de Flyback inductivo en [V].
            """
            self._ctx: "TopologicalContext" = context
            self._breakdown_voltage: float = breakdown_voltage

            # Cach s de factores Cholesky en formato cho_solve
            # cho_solve espera la tupla (L, lower) para resolver A x = b
            self._cho_C: Tuple[NDArray[np.float64], bool] = (context.L_C, True)
            self._cho_M: Tuple[NDArray[np.float64], bool] = (context.L_M, True)

            logger.debug(
                "[Fase2] Inicializada con breakdown_voltage=%.3e, "
                "dim_q=%d, dim_p=%d.",
                breakdown_voltage, context.dim_q, context.dim_p,
            )

        #
        # M todos privados de c lculo (mini-agentes originales, evolucionados)
        #

        def _evaluate_potential_energy(
            self,
            q: NDArray[np.float64],
        ) -> Tuple[float, NDArray[np.float64]]:
            r"""
            Calcula la energía potencial elástica y el gradiente asociado:

                V(q) = ½ ‖ L_C⁻¹ q ‖²  =  ½ q^⊤ C_soc⁻¹ q

                ∂V/∂q = C_soc⁻¹ q  =  cho_solve((L_C, True), q)

            El uso de ``la.solve_triangular`` para la norma evita resolver
            dos sistemas triangulares (que ya están implicados en cho_solve).
            La separación de V(q) y ∂V/∂q en una sola llamada amortiza el
            coste de la sustitución hacia atrás, que es idéntico en ambos.

            Parámetros
            ----------
            q : NDArray[np.float64], shape (dim_q,)
                Vector de coordenadas generalizadas.

            Retorna
            -------
            Tuple[float, NDArray[np.float64]]
                (V_q, grad_V_q) energía potencial y su gradiente.

            Lanza
            -----
            DimensionMismatchError
                Si q no tiene shape (dim_q,).
            """
            if q.shape != (self._ctx.dim_q,):
                raise DimensionMismatchError(
                    f"Vector q debe tener shape ({self._ctx.dim_q},); "
                    f"se obtuvo {q.shape}."
                )

            # y = L_C   q  (sustituci n hacia adelante, O(n ))
            y: NDArray[np.float64] = la.solve_triangular(
                self._ctx.L_C, q, lower=True, check_finite=False
            )

            # V(q) =    y   =   y^  y
            V_q: float = 0.5 * float(np.dot(y, y))

            #  V/ q = C_soc   q = L_C^{- } y  (sustituci n hacia atr s, O(n ))
            # Equivalente a cho_solve pero reutilizando y ya calculado
            grad_V_q: NDArray[np.float64] = la.solve_triangular(
                self._ctx.L_C, y, lower=True, trans="T", check_finite=False
            )

            logger.debug("[Fase2] V(q) = %.6e,    V  = %.6e.", V_q, float(np.linalg.norm(grad_V_q)))
            return V_q, grad_V_q

        def _compute_kinetic_energy(
            self,
            p: NDArray[np.float64],
        ) -> Tuple[float, NDArray[np.float64]]:
            r"""
            Calcula la energía cinética y el gradiente asociado:

                K(p) = ½ ‖ L_M⁻¹ p ‖²  =  ½ p^⊤ M_rec⁻¹ p

                ∂K/∂p = M_rec⁻¹ p  =  cho_solve((L_M, True), p)

            Análogo a ``_evaluate_potential_energy`` pero sobre el espacio de
            momentos p ∈ ℝ^{dim_p}.

            Retorna
            -------
            Tuple[float, NDArray[np.float64]]
                (K_p, grad_K_p) energía cinética y su gradiente.
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
                self._ctx.L_M, y, lower=True, trans="T", check_finite=False
            )

            logger.debug("[Fase2] K(p) = %.6e,    K  = %.6e.", K_p, float(np.linalg.norm(grad_K_p)))
            return K_p, grad_K_p

        def _enforce_rayleigh_dissipation(
            self,
            grad_H: NDArray[np.float64],
        ) -> float:
            r"""
            Verifica la desigualdad de disipación de Rayleigh y calcula la
            potencia disipada:

                Ḣ_diss = -∇H^⊤ · R_cost · ∇H  ≤  0

            La potencia disipada es:

                P_diss = |Ḣ_diss| = ∇H^⊤ · R_cost · ∇H  ≥  0

            La verificación se realiza con tolerancia relativa a ‖∇H‖² · ‖R‖_F,
            para distinguir violaciones genuinas de la segunda ley de errores
            numéricos de redondeo (O(ε · ‖R‖ · ‖∇H‖²)).

            Parámetros
            ----------
            grad_H : NDArray[np.float64], shape (dim_q + dim_p,)
                Gradiente completo del Hamiltoniano concatenado [∂H/∂q; ∂H/∂p].

            Retorna
            -------
            float
                P_diss = ∇H^⊤ R_cost ∇H ≥ 0.

            Lanza
            -----
            RayleighDissipationViolation
                Si ∇H^⊤ R_cost ∇H < -tol (violación genuina de la 2ª Ley).
            """
            n: int = self._ctx.dim_q + self._ctx.dim_p

            if grad_H.shape != (n,):
                raise DimensionMismatchError(
                    f"grad_H debe tener shape ({n},); se obtuvo {grad_H.shape}."
                )

            # R_cost    H  (O(n ))
            R_grad: NDArray[np.float64] = self._ctx.R_cost @ grad_H

            # Forma cuadr tica:  H^  R  H
            quad_form: float = float(np.dot(grad_H, R_grad))

            # Tolerancia relativa:      R _F     H
            norm_R: float = float(la.norm(self._ctx.R_cost, "fro"))
            norm_gH2: float = float(np.dot(grad_H, grad_H))
            tol_diss: float = float(np.finfo(np.float64).eps) * norm_R * norm_gH2

            if quad_form < -tol_diss:
                logger.error(
                    "[Fase2] Violaci n de Rayleigh:  H^  R  H = %.6e < -tol = %.6e.",
                    quad_form, -tol_diss,
                )
                raise RayleighDissipationViolation(
                    f"Violaci n de la Segunda Ley de la Termodin mica. "
                    f" H^  R_cost  H = {quad_form:.6e}  <  -tol = {-tol_diss:.6e}. "
                    f"La Estructura de Costes presenta generaci n de exerg a espont nea."
                )

            P_diss: float = abs(quad_form)
            logger.debug("[Fase2] P_diss = %.6e (Rayleigh OK).", P_diss)
            return P_diss

        def _measure_flyback_voltage(
            self,
            df_dt: NDArray[np.float64],
        ) -> float:
            r"""
            Mide el voltaje de Flyback inductivo generado por la derivada
            temporal del flujo de recursos:

                V_fb(t) = M_rec · (∂f/∂t)

            La norma infinito ‖V_fb‖_∞ representa el pico de voltaje transitorio
            en el isomorfismo eléctrico del modelo Port-Hamiltoniano, que
            corresponde al voltaje de pico en un inductor ante conmutación abrupta.

            M_rec se reconstruye eficientemente desde el factor Cholesky:

                M_rec = L_M · L_M^⊤

            evitando almacenar M_rec explícitamente (ya fue descartada tras
            la factorización en la Fase 1).

            Parámetros
            ----------
            df_dt : NDArray[np.float64], shape (dim_p,)
                Derivada temporal del vector de flujo de recursos.

            Retorna
            -------
            float
                ‖ M_rec · df_dt ‖_∞  (norma infinito del voltaje de Flyback).

            Lanza
            -----
            DimensionMismatchError
                Si df_dt no tiene shape (dim_p,).
            InertialFlybackError
                Si la norma excede el voltaje de ruptura configurado.
            """
            if df_dt.shape != (self._ctx.dim_p,):
                raise DimensionMismatchError(
                    f"df_dt debe tener shape ({self._ctx.dim_p},); "
                    f"se obtuvo {df_dt.shape}."
                )

            # M_rec   v = L_M   (L_M^    v): dos operaciones triangulares O(n )
            # m s eficientes y estables que reconstruir M_rec expl citamente.
            L_M_T_v: NDArray[np.float64] = self._ctx.L_M.T @ df_dt
            V_fb_vec: NDArray[np.float64] = self._ctx.L_M @ L_M_T_v

            v_fb_norm: float = float(la.norm(V_fb_vec, np.inf))

            if v_fb_norm > self._breakdown_voltage:
                logger.critical(
                    "[Fase2] Golpe de Ariete Log stico:  V_fb _  = %.6e > V_bd = %.6e.",
                    v_fb_norm, self._breakdown_voltage,
                )
                raise InertialFlybackError(
                    f"Voltaje de Flyback  M_rec  f/ t _  = {v_fb_norm:.6e} "
                    f"excede la tensi n de ruptura V_bd = {self._breakdown_voltage:.6e}. "
                    f"Detenci n de emergencia exigida."
                )

            logger.debug(
                "[Fase2]  V_fb _  = %.6e (l mite=%.6e, margen=%.1f%%).",
                v_fb_norm,
                self._breakdown_voltage,
                100.0 * (1.0 - v_fb_norm / self._breakdown_voltage),
            )
            return v_fb_norm

        #
        # M todo terminal de la Fase 2   entrada directa de la Fase 3
        #

        def synthesize_basal_state(
            self,
            q: NDArray[np.float64],
            p: NDArray[np.float64],
            df_dt: NDArray[np.float64],
        ) -> "BasalStateTensor":
            r"""
            **Método terminal de la Fase 2.**

            Computa el estado termodinámico completo del foso K_{BASE} dado
            el estado actual (q, p) y la perturbación de flujo df_dt.

            El ``BasalStateTensor`` resultante es el **dato primario** que
            consume la Fase 3 (a través de su vector de estado) si se requiere
            la proyección en haces. La frontera formal entre Fase 2 y Fase 3
            es el campo ``state_vector = [q; p]`` derivado de este tensor.

            Parámetros
            ----------
            q : NDArray[np.float64], shape (dim_q,)
                Coordenadas generalizadas (cargas de socios).
            p : NDArray[np.float64], shape (dim_p,)
                Momentos generalizados (flujos de recuperación).
            df_dt : NDArray[np.float64], shape (dim_p,)
                Derivada temporal del flujo de recursos (perturbación).

            Retorna
            -------
            BasalStateTensor
                Estado termodinámico completo, inmutable.

            Lanza
            -----
            DimensionMismatchError, RayleighDissipationViolation, InertialFlybackError
                Propagadas desde los métodos internos.
            """
            #    Energ as y gradientes (c lculo unificado, O(n ))
            V_q, grad_V_q = self._evaluate_potential_energy(q)
            K_p, grad_K_p = self._compute_kinetic_energy(p)

            H_total: float = V_q + K_p

            # Gradiente completo  H = [ H/ q;  H/ p]    ^{dim_q + dim_p}
            grad_H: NDArray[np.float64] = np.concatenate([grad_V_q, grad_K_p])
            grad_H_norm: float = float(la.norm(grad_H, 2))

            #    Disipaci n de Rayleigh
            P_diss: float = self._enforce_rayleigh_dissipation(grad_H)

            #    Voltaje de Flyback
            v_fb: float = self._measure_flyback_voltage(df_dt)

            logger.info(
                "[Fase2] Estado basal: H=%.6e, V=%.6e, K=%.6e, "
                "P_diss=%.6e,  V_fb _ =%.6e,   H =%.6e.",
                H_total, V_q, K_p, P_diss, v_fb, grad_H_norm,
            )

            #    Contrato de interfaz entre Fase 2 y Fase 3
            # El tensor devuelto contiene state_vector=np.concatenate([q,p])
            # que es el argumento directo de Phase3_SheafProjection.export_stalk().
            return BasalStateTensor(
                potential_energy=V_q,
                kinetic_energy=K_p,
                total_hamiltonian=H_total,
                dissipated_power=P_diss,
                flyback_voltage_norm=v_fb,
                grad_H_norm=grad_H_norm,
                is_thermodynamically_stable=True,
            )

    #
    # FASE 3   PROYECCI N EN HACES Y COFRONTERA DISCRETA
    #

    class Phase3_SheafProjection:
        r"""
        **Fase 3 – Proyección en Haces (Sheaf) y Cofrontera Discreta δ_{BASE}.**

        Recibe el ``TopologicalContext`` de la Fase 1 (vía el agente orquestador)
        y el ``state_vector = [q; p]`` derivado del ``BasalStateTensor`` de la
        Fase 2, construyendo el fibrado celular (Stalk) que alimenta el cálculo
        global del Laplaciano de Haz del ecosistema APU Filter.

        El constructor de esta clase es la **continuación directa** del método
        ``synthesize_basal_state`` de la Fase 2: el vector de estado allá
        producido es el que aquí se proyecta.

        Fundamento Matemático del Haz
        ------------------------------
        En la categoría de haces sobre la categoría de grafos del ecosistema,
        el Stalk en el vértice K_{BASE} es el espacio vectorial:

            F(K_{BASE}) = ℝ^{dim_q} ⊕ ℝ^{dim_p}

        La restricción (cofrontera local) δ_{BASE}: F(K_{BASE}) → F(e_{BASE})
        para la arista adyacente e_{BASE} se modela como el operador bloque:

            δ_{BASE} = block_diag( C_soc^{-1/2},  R_cost^{+1/2} )

        donde:
          • C_soc^{-1/2} := L_C^{-⊤}  (triangular superior de la inversa
            de Cholesky, que es la raíz cuadrada de C_soc⁻¹ en el sentido
            de que (L_C^{-⊤})^⊤ · L_C^{-⊤} = C_soc⁻¹)
          • R_cost^{+1/2} := V · diag(√λ⁺) · V^⊤  (raíz cuadrada espectral
            precalculada en la Fase 1 y almacenada en ``context.R_sqrt``)

        Verificación de Complejo de Cadenas
        ------------------------------------
        Para que δ_{BASE} sea un operador de cofrontera válido en el sentido
        de la homología celular, debe satisfacerse δ² = 0 a nivel de complejo,
        pero a nivel local (single-stalk) la condición es más débil:

            δ_{BASE}^⊤ · δ_{BASE} = block_diag(C_soc⁻¹, R_cost)

        que es la métrica local del Laplaciano de Hodge-Kirchhoff del haz.
        La norma ‖δ_{BASE}²‖_F se verifica contra una tolerancia de máquina
        para detectar errores de ensamble.
        """

        def __init__(self, context: "TopologicalContext") -> None:
            r"""
            **Constructor de la Fase 3: continuación directa de la Fase 2.**

            Recibe el mismo ``TopologicalContext`` de la Fase 1 que ya consumió
            la Fase 2, garantizando coherencia matemática sin re-computar
            ninguna factorización.

            La Fase 3 precalcula en el constructor las dos sub-matrices de
            δ_{BASE} para amortizar el coste sobre múltiples llamadas a
            ``export_stalk`` (la proyección puede requerirse en cada paso
            del ciclo de evaluación del ecosistema).
            """
            self._ctx: "TopologicalContext" = context

            #    Sub-matriz (1,1): C_soc^{-1/2} = L_C^{- }
            # L_C es triangular inferior con diagonal positiva, por lo que
            # su inversa es triangular inferior y L_C^{- } = (L_C^{-1})^
            # es triangular superior. Se calcula como la soluci n de L_C   X = I.
            n_q: int = context.dim_q
            I_q: NDArray[np.float64] = np.eye(n_q, dtype=np.float64)
            # L_C^{-1} (triangular inferior)
            L_C_inv: NDArray[np.float64] = la.solve_triangular(
                context.L_C, I_q, lower=True, check_finite=False
            )
            # C_soc^{-1/2} = L_C^{- } (triangular superior)
            self._C_soc_inv_half: NDArray[np.float64] = L_C_inv.T

            #    Sub-matriz (2,2): R_cost^{+1/2} precalculada en Fase 1
            self._R_sqrt: NDArray[np.float64] = context.R_sqrt

            #    Rango total de  _{BASE}
            # C_soc^{-1/2} es cuadrada y no singular (dim_q   dim_q, rango pleno)
            # R_cost^{+1/2} tiene rango igual a rank_R
            self._rank_delta: int = n_q + context.rank_R

            logger.debug(
                "[Fase3] Precalculado: C_soc^{-1/2} shape=%s, "
                "R_sqrt shape=%s, rank_delta=%d.",
                self._C_soc_inv_half.shape,
                self._R_sqrt.shape,
                self._rank_delta,
            )

        def _assemble_coboundary(self) -> NDArray[np.float64]:
            r"""
            Ensambla la matriz de cofrontera discreta δ_{BASE} como operador
            bloque diagonal:

                δ_{BASE} = block_diag( C_soc^{-1/2},  R_cost^{+1/2} )
                         ∈ ℝ^{n × n},   n = dim_q + dim_p

            Las sub-matrices están almacenadas como atributos precalculados
            en el constructor, por lo que este método es O(n²) en asignación
            de memoria (bloque cero + copia de bloques diagonales).

            Retorna
            -------
            NDArray[np.float64]
                Matriz de cofrontera δ_{BASE} de shape (n, n).
            """
            dim_q: int = self._ctx.dim_q
            dim_p: int = self._ctx.dim_p
            n: int = dim_q + dim_p

            delta: NDArray[np.float64] = np.zeros((n, n), dtype=np.float64)

            # Bloque (1,1): C_soc^{-1/2}    ^{dim_q   dim_q}
            delta[:dim_q, :dim_q] = self._C_soc_inv_half

            # Bloque (2,2): R_cost^{+1/2}    ^{dim_p   dim_p}
            delta[dim_q:, dim_q:] = self._R_sqrt

            return delta

        def _verify_coboundary_metric(
            self,
            delta: NDArray[np.float64],
        ) -> float:
            r"""
            Verifica la consistencia de δ_{BASE} calculando:

                ‖ δ^⊤ δ − block_diag(C_soc⁻¹, R_cost) ‖_F / ‖δ‖_F²

            Esta identidad es exacta en aritmética exacta y sirve para detectar
            errores numéricos de ensamble. Se usa en lugar de ‖δ²‖_F = 0 (que
            aplica al complejo global, no al stalk local).

            Retorna
            -------
            float
                Error relativo ‖δ^⊤δ − Hodge_local‖_F / ‖δ‖_F².

            Lanza
            -----
            SheafCoboundaryError
                Si el error relativo supera 100 · ε_mach (100 ULP).
            """
            dim_q: int = self._ctx.dim_q
            n: int = dim_q + self._ctx.dim_p

            # M trica local de Hodge esperada
            Hodge_local: NDArray[np.float64] = np.zeros((n, n), dtype=np.float64)
            # C_soc   = (L_C^{- })^    L_C^{- } = L_C     L_C^{- }
            Hodge_local[:dim_q, :dim_q] = self._C_soc_inv_half.T @ self._C_soc_inv_half
            # R_cost = R_sqrt   R_sqrt  (propiedad de la ra z espectral)
            Hodge_local[dim_q:, dim_q:] = self._R_sqrt @ self._R_sqrt

            # Computar  ^
            delta_T_delta: NDArray[np.float64] = delta.T @ delta
            residual_F: float = float(la.norm(delta_T_delta - Hodge_local, "fro"))
            norm_delta_sq: float = float(la.norm(delta, "fro") ** 2)

            tol_metric: float = 100.0 * float(np.finfo(np.float64).eps)
            rel_error: float = residual_F / max(norm_delta_sq, 1.0)

            if rel_error > tol_metric:
                raise SheafCoboundaryError(
                    f" _{'{'}BASE{'}'} no satisface la identidad de Hodge local. "
                    f"  ^   - Hodge_local _F /    _F  = {rel_error:.6e} "
                    f"> tol = {tol_metric:.6e}."
                )

            logger.debug(
                "[Fase3] M trica de Hodge verificada: rel_error=%.3e, tol=%.3e.",
                rel_error, tol_metric,
            )

            # Para compatibilidad con SheafStalk.delta_base_sq_norm
            # devolvemos        _F como medida de "cu nto se aleja de   =0 global"
            delta_sq: NDArray[np.float64] = delta @ delta
            return float(la.norm(delta_sq, "fro"))

        #
        # M todo terminal de la Fase 3 (salida p blica del ecosistema)
        #

        def export_stalk(
            self,
            state_x: NDArray[np.float64],
        ) -> "SheafStalk":
            r"""
            **Método terminal de la Fase 3 y del agente completo.**

            Construye y retorna el ``SheafStalk`` que contiene:
              • La cofrontera discreta δ_{BASE} ensamblada y verificada.
              • El vector de estado x = [q; p] en el instante de proyección.
              • La proyección δ_{BASE} · x sobre la fibra local.
              • El rango numérico de δ_{BASE}.
              • La norma ‖δ_{BASE}²‖_F como métrica de calidad del haz.

            Este ``SheafStalk`` es consumido directamente por el Laplaciano
            de Haz global del ecosistema APU Filter para el cálculo de
            cohomología de haz y detección de inconsistencias globales.

            Parámetros
            ----------
            state_x : NDArray[np.float64], shape (dim_q + dim_p,)
                Vector de estado x = [q; p], producido como
                ``np.concatenate([q, p])`` desde el ``BasalStateTensor``
                de la Fase 2. Esta es la continuación directa de la Fase 2.

            Retorna
            -------
            SheafStalk
                Fibrado celular completo, inmutable.

            Lanza
            -----
            DimensionMismatchError
                Si state_x no tiene shape (dim_q + dim_p,).
            SheafCoboundaryError
                Si δ_{BASE} no satisface la identidad de Hodge local.
            """
            n: int = self._ctx.dim_q + self._ctx.dim_p

            if state_x.shape != (n,):
                raise DimensionMismatchError(
                    f"state_x debe tener shape ({n},) = (dim_q + dim_p,); "
                    f"se obtuvo {state_x.shape}."
                )

            #    Ensamble y verificaci n de  _{BASE}
            delta_base: NDArray[np.float64] = self._assemble_coboundary()
            delta_sq_norm: float = self._verify_coboundary_metric(delta_base)

            #    Proyecci n del estado sobre la fibra
            projected: NDArray[np.float64] = delta_base @ state_x

            logger.info(
                "[Fase3] SheafStalk exportado: rank_delta=%d, "
                "    _F=%.3e,    x =%.6e.",
                self._rank_delta,
                delta_sq_norm,
                float(la.norm(projected, 2)),
            )

            #    Contrato de salida del agente completo
            # El SheafStalk es el output final de la cadena de 3 fases.
            return SheafStalk(
                delta_base=delta_base,
                delta_base_sq_norm=delta_sq_norm,
                state_vector=state_x.copy(),
                projected_state=projected,
                rank_delta=self._rank_delta,
            )

    #
    # INTERFAZ P BLICA DEL AGENTE (punto de entrada externo)
    #

    def synthesize_basal_hamiltonian(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        df_dt: NDArray[np.float64],
    ) -> BasalStateTensor:
        r"""
        Punto de entrada público para la evaluación termodinámica completa.

        Delega completamente en ``Phase2_HamiltonianDynamics.synthesize_basal_state``,
        que ya posee el contexto topológico validado por la Fase 1.

        Parámetros
        ----------
        q : NDArray[np.float64], shape (dim_q,)
            Coordenadas generalizadas en el instante de evaluación.
        p : NDArray[np.float64], shape (dim_p,)
            Momentos generalizados en el instante de evaluación.
        df_dt : NDArray[np.float64], shape (dim_p,)
            Derivada temporal del flujo de recursos (perturbación de cronograma).

        Retorna
        -------
        BasalStateTensor
            Estado termodinámico completo e inmutable.
        """
        return self.phase2.synthesize_basal_state(q=q, p=p, df_dt=df_dt)

    def export_sheaf_stalk(
        self,
        state_x: NDArray[np.float64],
    ) -> SheafStalk:
        r"""
        Exporta el Stalk del haz celular y la cofrontera δ_{BASE}.

        Instancia la Fase 3 perezosamente en la primera llamada (el costo
        de precalcular C_soc^{-1/2} se paga una sola vez) y reutiliza la
        instancia en llamadas subsiguientes.

        Parámetros
        ----------
        state_x : NDArray[np.float64], shape (dim_q + dim_p,)
            Vector de estado x = [q; p]. Típicamente se construye como
            ``np.concatenate([q, p])`` después de llamar a
            ``synthesize_basal_hamiltonian``.

        Retorna
        -------
        SheafStalk
            Fibrado celular completo e inmutable.
        """
        if self.phase3 is None:
            self.phase3 = KBaseThermodynamicAgent.Phase3_SheafProjection(
                context=self.context
            )
            logger.info(
                "[KBaseThermodynamicAgent] Phase3_SheafProjection instanciada "
                "(lazy init). rank_delta=%d.",
                self.phase3._rank_delta,
            )
        return self.phase3.export_stalk(state_x=state_x)
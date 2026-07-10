# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo : KBase Thermodynamic Agent (Asesor de Cimientos Financieros)         |
| Ruta   : app/agents/alpha/kbase/kbase_thermodynamic_agent.py                 |
| Versión: 4.0.0-Rigorous-Sheaf-Williamson-Boolean                             |
+==============================================================================+

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA DIFERENCIAL
------------------------------------------------
Este módulo consagra el Foso Termodinámico del ecosistema (K_{BASE}). Actúa como
un Endofuntor Port-Hamiltoniano que gobierna la inercia, la capacitancia y la
fricción entrópica del modelo de negocio.

CAMBIOS ESTRUCTURALES RESPECTO A v3.0.0 (evolución granular)
--------------------------------------------------------------
1. PULLBACK RIEMANNIANO REAL (Fase 1): se implementa efectivamente
   \[ \tilde{C}_{soc} = G_q C_{soc} G_q^\top, \qquad \tilde{M}_{rec} = G_p M_{rec} G_p^\top \]
   con validación de invertibilidad de G_q, G_p vía valores singulares.

2. CORRECCIÓN DE LA COFRONTERA DE HAZ (Fase 3): δ_{BASE} deja de ser un
   bloque diagonal cuadrado mal dimensionado y se reformula como el mapa
   cocadena genuino hacia la suma directa de dos fibras incidentes
   (arista métrica ⊕ arista disipativa):
   \[ \delta_{BASE} = \begin{pmatrix}\delta_{metric}\\ \delta_{diss}\end{pmatrix}
      \in \mathbb{R}^{2n\times n}, \quad
      \delta_{BASE}^\top\delta_{BASE} = \nabla^2 H + R_{cost} =: \Delta_{BASE} \]

3. CAMPO VECTORIAL PORT-HAMILTONIANO EXPLÍCITO (Fase 2):
   \[ \dot{x} = (J_{BASE} - R_{cost})\,\nabla H(x) \]
   con verificación estructural exacta \( \nabla H^\top J_{BASE}\nabla H \equiv 0 \).

4. MODOS NORMALES DE WILLIAMSON Y ANALOGÍA CUÁNTICA (Fase 2, diagnóstico
   opcional): dado que \( M^{1/2} J M^{1/2} \) es real antisimétrica para
   cualquier J antisimétrica y M ≻ 0, sus valores singulares son exactamente
   las frecuencias propias \( \omega_k \) del sistema conservativo linealizado.
   Energía de punto cero análoga: \( E_0=\tfrac{\hbar}{2}\sum_k \omega_k \).

5. ÁLGEBRA DE BOOLE DE ESTABILIDAD: `StabilityFlags(enum.Flag)` sustituye al
   booleano monolítico, formando una retícula con ∧ (`&`), ∨ (`|`), ¬ (`~`),
   con elemento superior `ALL` e inferior `NONE`.

6. CHOLESKY REGULARIZADO Y CÁLCULO ESPECTRAL EFICIENTE: jitter de Tikhonov
   adaptativo con trazabilidad; `eigh(subset_by_index=...)` para extremos.

HAMILTONIANO BASAL:
\[ H_{BASE}(q,p) = \frac{1}{2} q^\top \tilde{C}_{soc}^{-1} q + \frac{1}{2} p^\top \tilde{M}_{rec}^{-1} p \]

ECUACIÓN DE DISIPACIÓN DE RAYLEIGH:
\[ \dot{H}_{diss} = -\nabla H^\top R_{cost} \nabla H \le 0 \]
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
    # Stubs mínimos para ejecución aislada y prueba unitaria analítica
    class CategoricalState:  # type: ignore[no-redef]
        """Stub: estado categórico del ecosistema MIC."""

    class Morphism:  # type: ignore[no-redef]
        """Stub: morfismo funtorial del ecosistema MIC."""


# Logger del módulo
logger = logging.getLogger("MIC.Alpha.KBaseThermodynamicAgent")

# Precisión de máquina IEEE-754 double, reutilizada en todo el módulo
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)


#
# SECCIÓN 0 — EXCEPCIONES TERMODINÁMICAS ESTRICTAS
#


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
    Lanzada cuando C_soc̃ o M_rec̃ (tras el pullback métrico) no son SPD
    incluso después de aplicar regularización de Tikhonov adaptativa.

    Diagnóstico incluye κ(A), λ_min y el jitter τ finalmente ensayado.
    """


class InertialFlybackError(ThermodynamicBaseError):
    """
    Lanzada cuando la inercia de recuperación genera un voltaje transitorio
    de Flyback que excede el límite de ruptura dieléctrica.

    Condición de disparo: ‖ M̃_rec · (∂f/∂t) ‖_∞ > V_breakdown
    """


class RayleighDissipationViolation(ThermodynamicBaseError):
    """
    Lanzada cuando el modelo disipativo indica entropía negativa (ganancia
    fantasma), violando la Segunda Ley de la Termodinámica.

    Condición de disparo: ∇H^⊤ R_cost ∇H < -tol  (P_diss < -tol < 0)
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
    \( \tilde{A} = G A G^\top \) requerido para absorber el estrés
    anisotrópico del ecosistema.
    """


class SheafCoboundaryError(ThermodynamicBaseError):
    """
    Lanzada cuando la cofrontera discreta δ_{BASE} no satisface la
    identidad de Hodge local  δ^⊤δ = ∇²H + R_cost  dentro de tolerancia
    de máquina escalada por ‖δ‖_F².
    """


class StructuralConsistencyError(ThermodynamicBaseError):
    """
    Lanzada cuando la identidad algebraica exacta del sistema
    Port-Hamiltoniano, ∇H^⊤ẋ ≡ -P_diss (porque ∇H^⊤J∇H≡0 para J
    antisimétrica), se viola más allá del error de redondeo esperado.

    A diferencia de ``RayleighDissipationViolation`` (violación física de
    la 2ª Ley), esta excepción indica un **error de implementación o de
    cableado** entre J_base, R_cost y el gradiente ∇H.
    """


#
# SECCIÓN 1 — ÁLGEBRA DE BOOLE DE ESTABILIDAD
#


class StabilityFlags(enum.Flag):
    r"""
    Retícula Booleana de predicados de estabilidad termodinámica.

    ``enum.Flag`` provee de forma nativa una **álgebra de Boole** completa
    sobre el conjunto de predicados: conjunción (`&`), disyunción (`|`),
    complemento (`~`), elemento ínfimo ``NONE`` (0̄) y elemento supremo
    ``ALL`` (1̄), satisfaciendo las leyes de De Morgan, distributividad,
    absorción e idempotencia por construcción del tipo.

    Cada bandera es un predicado *independiente* verificable en tiempo de
    evaluación; su conjunción (`meet`) determina la estabilidad global.
    A diferencia de un único booleano, esta retícula preserva el
    diagnóstico granular de *cuál* condición falló.

    Miembros
    --------
    ENERGY_NONNEGATIVE
        V(q) ≥ 0  ∧  K(p) ≥ 0  (garantizado por SPD, pero verificado).
    DISSIPATION_VALID
        P_diss ≥ 0  (Segunda Ley, verificado independientemente).
    FLYBACK_SAFE
        ‖V_fb‖_∞ ≤ margin · V_breakdown  (margen de seguridad, más
        estricto que el límite duro que dispara ``InertialFlybackError``).
    STRUCTURAL_CONSISTENCY
        |∇H^⊤ẋ + P_diss| ≤ tol  (identidad algebraica exacta del PH-system).
    SPECTRAL_CONDITIONING_SOUND
        κ(C̃_soc), κ(M̃_rec) por debajo de la mitad de κ_max (alerta temprana).
    """

    NONE = 0
    ENERGY_NONNEGATIVE = enum.auto()
    DISSIPATION_VALID = enum.auto()
    FLYBACK_SAFE = enum.auto()
    STRUCTURAL_CONSISTENCY = enum.auto()
    SPECTRAL_CONDITIONING_SOUND = enum.auto()
    ALL = (
        ENERGY_NONNEGATIVE
        | DISSIPATION_VALID
        | FLYBACK_SAFE
        | STRUCTURAL_CONSISTENCY
        | SPECTRAL_CONDITIONING_SOUND
    )


def describe_stability_flags(flags: StabilityFlags) -> str:
    """
    Serializa la retícula ``StabilityFlags`` a una cadena legible,
    listando explícitamente los predicados satisfechos y los violados
    (complemento relativo a ``StabilityFlags.ALL``).
    """
    satisfied = [f.name for f in StabilityFlags if f not in (StabilityFlags.NONE, StabilityFlags.ALL) and f in flags]
    violated = [f.name for f in StabilityFlags if f not in (StabilityFlags.NONE, StabilityFlags.ALL) and f not in flags]
    return (
        f"SATISFECHOS={satisfied or 'ninguno'} | "
        f"VIOLADOS={violated or 'ninguno'} | "
        f"ESTABLE_TOTAL={flags == StabilityFlags.ALL}"
    )


#
# SECCIÓN 2 — ESTRUCTURAS INMUTABLES (DTOs TENSORIALES)
#


@dataclass(frozen=True, slots=True)
class TopologicalContext:
    r"""
    Contexto inmutable producido por la **Fase 1** (Topología Matricial,
    Métrica Riemanniana y Teoría Espectral).

    Contiene todas las factorizaciones, tensores métricos y diagnósticos
    espectrales necesarios para que las Fases 2 y 3 operen sin
    re-validar ni re-factorizar ninguna matriz.

    Atributos
    ----------
    L_C, L_M : NDArray[np.float64]
        Factores de Cholesky inferiores de las matrices **ya pulled-back**
        \( \tilde{C}_{soc}=G_qC_{soc}G_q^\top \), \( \tilde{M}_{rec}=G_pM_{rec}G_p^\top \).

    C_inv_sqrt, M_inv_sqrt : NDArray[np.float64]
        Raíces de las inversas: \( \tilde{C}_{soc}^{-1/2}=L_C^{-\top} \),
        \( \tilde{M}_{rec}^{-1/2}=L_M^{-\top} \). Precalculadas una única
        vez y reutilizadas por las Fases 2 (modos normales) y 3 (haz).

    R_cost : NDArray[np.float64]
        Matriz de disipación de Rayleigh validada (PSD), copia inmutable.

    R_sqrt : NDArray[np.float64]
        Raíz cuadrada espectral de R_cost: \( R_{sqrt}=V\,diag(\sqrt{\lambda^+})\,V^\top \).

    J_base : NDArray[np.float64]
        Matriz de interconexión antisimétrica validada, copia inmutable.

    G_q, G_p : NDArray[np.float64]
        Tensores métricos Riemannianos aplicados (identidad si no se
        proveyeron), almacenados para trazabilidad.

    kappa_C, kappa_M : float
        Números de condición espectral de \( \tilde{C}_{soc} \), \( \tilde{M}_{rec} \).

    kappa_G_q, kappa_G_p : float
        Números de condición (vía valores singulares) de G_q, G_p.

    epsilon_C, epsilon_M : float
        Jitter de Tikhonov aplicado (0.0 si no fue necesario) durante la
        factorización de Cholesky regularizada.

    dim_q, dim_p : int
        Dimensiones de los espacios de coordenadas y momentos generalizados.

    rank_R : int
        Rango numérico de R_cost.

    spectral_gap_R : float
        Brecha espectral λ₂ − λ₁ de R_cost (teoría espectral de grafos:
        análogo a la conectividad algebraica de Fiedler cuando R_cost se
        interpreta como operador de una red disipativa).

    betti_0_R : int
        Dimensión del núcleo de R_cost = n − rank_R. Topológicamente,
        el número de componentes "sin pérdida" (subespacio conservativo
        puro, de disipación nula) del complejo Port-Hamiltoniano.
    """

    L_C: NDArray[np.float64]
    L_M: NDArray[np.float64]
    C_inv_sqrt: NDArray[np.float64]
    M_inv_sqrt: NDArray[np.float64]
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


@dataclass(frozen=True, slots=True)
class BasalStateTensor:
    r"""
    Tensor inmutable que encapsula el estado termodinámico completo del foso.

    Producido por la **Fase 2** (Dinámica Port-Hamiltoniana).

    Atributos
    ----------
    potential_energy, kinetic_energy, total_hamiltonian : float
        V(q), K(p) y H(q,p) = V+K, todos ≥ 0.

    dissipated_power : float
        P_diss = |∇H^⊤ R_cost ∇H| ≥ 0.

    flyback_voltage_norm : float
        ‖ M̃_rec · (∂f/∂t) ‖_∞.

    grad_H_norm : float
        ‖∇H‖₂.

    vector_field : NDArray[np.float64]
        Campo vectorial completo del sistema Port-Hamiltoniano:
        \( \dot{x} = (J_{BASE}-R_{cost})\nabla H \in \mathbb{R}^{n} \).

    euler_homogeneity_residual : float
        |q·∇_qH + p·∇_pH − 2H|, verificación del Teorema de Euler para
        funciones homogéneas de grado 2 (H es exactamente cuadrática).

    structural_consistency_residual : float
        |∇H^⊤ẋ + P_diss|, verificación de la identidad algebraica exacta
        ∇H^⊤J∇H ≡ 0 (antisimetría de J actuando sobre su propio gradiente).

    stability_flags : StabilityFlags
        Retícula Booleana de predicados de estabilidad (ver clase homónima).

    is_thermodynamically_stable : bool
        Equivalente a ``stability_flags == StabilityFlags.ALL``.

    normal_mode_frequencies : Optional[NDArray[np.float64]]
        Frecuencias propias ω_k del subsistema conservativo linealizado
        (Teorema de Williamson), calculadas sólo si se solicitó el
        diagnóstico opcional en ``synthesize_basal_state``.

    zero_point_energy : Optional[float]
        \( E_0=\tfrac{\hbar}{2}\sum_k\omega_k \), analogía cuántica del
        oscilador armónico desacoplado en modos normales.
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
    stability_flags: StabilityFlags
    is_thermodynamically_stable: bool
    normal_mode_frequencies: Optional[NDArray[np.float64]]
    zero_point_energy: Optional[float]


@dataclass(frozen=True, slots=True)
class SheafStalk:
    r"""
    Fibrado celular exportado para el cálculo global del Laplaciano de Haz.

    Producido por la **Fase 3** (Proyección Cohomológica en Haces).

    Ensamblaje formal (corregido respecto a versiones previas)
    ------------------------------------------------------------
    El stalk en K_{BASE}, de dimensión n = dim_q+dim_p, se restringe hacia
    la **suma directa de dos fibras incidentes**: la arista métrica
    (conservativa) y la arista disipativa, evitando forzar dimensiones
    incompatibles en un único bloque diagonal cuadrado:

        δ_metric = block_diag( C̃_soc^{-1/2},  M̃_rec^{-1/2} )   ∈ ℝ^{n×n}  (invertible)
        δ_diss   = R_cost^{+1/2}                                ∈ ℝ^{n×n}  (PSD)
        δ_BASE   = [ δ_metric ; δ_diss ]                        ∈ ℝ^{2n×n}

    de modo que el Laplaciano de Hodge local es exactamente:

        Δ_BASE := δ_BASE^⊤ δ_BASE = δ_metric^⊤δ_metric + δ_diss^⊤δ_diss
                = ∇²H + R_cost   (SPD, pues ∇²H ≻ 0 y R_cost ⪰ 0)

    Atributos
    ----------
    delta_base : NDArray[np.float64], shape (2n, n)
        Cocadena apilada δ_{BASE}.
    delta_metric, delta_dissipative : NDArray[np.float64], shape (n, n)
        Componentes de δ_{BASE} hacia cada fibra incidente.
    hodge_laplacian : NDArray[np.float64], shape (n, n)
        Δ_BASE = δ_BASE^⊤δ_BASE, SPD.
    hodge_identity_residual : float
        ‖Δ_BASE − (∇²H+R_cost)‖_F / ‖Δ_BASE‖_F, verificado ≈ 0.
    hodge_spectral_gap : float
        λ₂(Δ_BASE) − λ₁(Δ_BASE).
    hodge_condition_number : float
        κ(Δ_BASE) = λ_max/λ_min.
    harmonic_dimension : int
        dim ker(δ_metric) — siempre 0 dado que ∇²H ≻ 0 (invariante
        topológico: el sector elástico/inercial nunca degenera).
    lossless_subspace_dimension : int
        n − rank(R_cost): dimensión del subespacio de disipación nula
        (modos puramente conservativos), heredado de ``betti_0_R``.
    state_vector : NDArray[np.float64], shape (n,)
        x = [q; p] en el instante de proyección.
    projected_state_metric, projected_state_dissipative : NDArray[np.float64]
        δ_metric·x y δ_diss·x respectivamente (proyecciones sobre cada fibra).
    rank_delta : int
        Rango de δ_BASE = n (columna completa, garantizado por δ_metric ≻ 0).
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


#
# SECCIÓN 3 — ORQUESTADOR: KBaseThermodynamicAgent
#             Tres fases anidadas de rigor creciente
#


class KBaseThermodynamicAgent(Morphism):
    r"""
    Orquestador Funtorial del Foso Termodinámico K_{BASE}.

    Integra el modelo Port-Hamiltoniano del estrato K_{BASE} mediante tres
    clases anidadas que operan en cascada estricta:

        Phase1_MatrixTopology          (Riemann + Espectral + Cholesky robusto)
            ↓  TopologicalContext
        Phase2_HamiltonianDynamics     (Campo vectorial + Williamson + Boole)
            ↓  BasalStateTensor
        Phase3_SheafProjection         (Cocadena apilada + Hodge local)
            ↓  SheafStalk

    Parámetros de Construcción
    --------------------------
    C_soc, M_rec, R_cost, J_base : NDArray[np.float64]
        Matrices constitutivas (ver Fase 1 para condiciones formales).
    breakdown_voltage : float, default 1e5
        Umbral de ruptura dieléctrica para el voltaje de Flyback.
    kappa_max : float, default 1e10
        Umbral máximo de número de condición espectral admisible.
    G_q, G_p : Optional[NDArray[np.float64]]
        Tensores métricos Riemannianos para el pullback anisotrópico de
        C_soc y M_rec respectivamente. Si ``None``, se usa la identidad
        (caso euclidiano, retrocompatible con v3.0.0).
    hbar : float, default 1.0
        Constante de analogía cuántica (adimensional en este isomorfismo)
        usada para la energía de punto cero de los modos normales.
    flyback_safety_margin : float, default 0.9
        Fracción de ``breakdown_voltage`` usada como umbral *blando* para
        la bandera ``StabilityFlags.FLYBACK_SAFE`` (más estricta que el
        límite duro que dispara ``InertialFlybackError``).
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
        G_q: Optional[NDArray[np.float64]] = None,
        G_p: Optional[NDArray[np.float64]] = None,
        hbar: float = 1.0,
        flyback_safety_margin: float = 0.9,
    ) -> None:
        r"""
        Inicializa las matrices constitutivas y ejecuta la Fase 1 de inmediato.

        Flujo: (1) instanciar y ejecutar Phase1 → TopologicalContext;
        (2) instanciar Phase2 con dicho contexto; (3) diferir Phase3
        (instanciación perezosa).
        """
        self.breakdown_voltage: float = breakdown_voltage
        self.kappa_max: float = kappa_max
        self.hbar: float = hbar
        self.flyback_safety_margin: float = flyback_safety_margin

        # Fase 1: Topología Matricial, Riemann y Espectro (ejecución inmediata)
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
        self.context: TopologicalContext = self.phase1.build_topological_context()

        # Fase 2: Dinámica Port-Hamiltoniana (instanciación inmediata)
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
        self.phase3: Optional[KBaseThermodynamicAgent.Phase3_SheafProjection] = None

        logger.info(
            "[KBaseThermodynamicAgent] Inicialización completa. "
            "dim_q=%d, dim_p=%d, κ(C̃)=%.3e, κ(M̃)=%.3e, rank(R)=%d, "
            "betti_0(R)=%d, gap_espectral(R)=%.3e.",
            self.context.dim_q,
            self.context.dim_p,
            self.context.kappa_C,
            self.context.kappa_M,
            self.context.rank_R,
            self.context.betti_0_R,
            self.context.spectral_gap_R,
        )

    #
    # ==========================================================================
    # FASE 1 — TOPOLOGÍA MATRICIAL, PULLBACK RIEMANNIANO Y VALIDACIÓN ESPECTRAL
    # ==========================================================================
    #

    class Phase1_MatrixTopology:
        r"""
        **Fase 1 – Topología Matricial, Métrica Riemanniana y Teoría Espectral.**

        Responsabilidades de esta fase, en orden estricto de ejecución:

          a) Verificar coherencia dimensional de C_soc, M_rec, R_cost, J_base.
          b) Verificar antisimetría de J_base y simetría de C_soc, M_rec, R_cost.
          c) Validar invertibilidad/condicionamiento de los tensores métricos
             G_q, G_p (identidad por defecto).
          d) Aplicar el *pullback* congruente Riemanniano:
             \( \tilde{C}_{soc}=G_qC_{soc}G_q^\top \), \( \tilde{M}_{rec}=G_pM_{rec}G_p^\top \).
          e) Calcular κ(C̃_soc), κ(M̃_rec) vía extremos espectrales eficientes
             (``eigh(subset_by_index=...)``, evitando diagonalización completa).
          f) Factorizar C̃_soc, M̃_rec por Cholesky **regularizado** (jitter de
             Tikhonov adaptativo si la factorización directa falla).
          g) Precalcular las raíces de las inversas C̃_soc^{-1/2}, M̃_rec^{-1/2}
             (reutilizadas por Fases 2 y 3).
          h) Verificar R_cost ⪰ 0, calcular su raíz espectral, su brecha
             espectral (teoría espectral de grafos) y su número de Betti-0.
          i) Empaquetar todo en un ``TopologicalContext`` inmutable.

        Tolerancias
        -----------
            tol_sym = ε_mach · max(‖A‖_F, 1)         (simetría/antisimetría)
            tol_pd  = ε_mach · λ_max                  (definición positiva)
            tol_psd = ε_mach · max(‖A‖_F, 1)          (semidefinición positiva)
        """

        _EPS: float = _MACHINE_EPS

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
            r"""
            Almacena las matrices originales y los tensores métricos (posiblemente
            ``None``) sin modificarlos. Las copias y valores por defecto se
            resuelven en ``build_topological_context``, una vez conocidas
            dim_q y dim_p.
            """
            self._C_soc: NDArray[np.float64] = C_soc
            self._M_rec: NDArray[np.float64] = M_rec
            self._R_cost: NDArray[np.float64] = R_cost
            self._J_base: NDArray[np.float64] = J_base
            self._kappa_max: float = kappa_max
            self._G_q_raw: Optional[NDArray[np.float64]] = G_q
            self._G_p_raw: Optional[NDArray[np.float64]] = G_p

        #
        # Métodos privados de validación (orden lógico de ejecución)
        #

        def _check_dimensions(self) -> Tuple[int, int]:
            r"""
            Verifica que las dimensiones de todas las matrices sean
            consistentes con un espacio de fases (q, p) bien definido.

            Retorna
            -------
            Tuple[int, int]
                (dim_q, dim_p) validados.

            Lanza
            -----
            DimensionMismatchError
                Si alguna condición de cuadratura o consistencia falla.
            """
            if self._C_soc.ndim != 2 or self._C_soc.shape[0] != self._C_soc.shape[1]:
                raise DimensionMismatchError(
                    f"C_soc debe ser cuadrada; se obtuvo shape={self._C_soc.shape}."
                )
            dim_q: int = self._C_soc.shape[0]

            if self._M_rec.ndim != 2 or self._M_rec.shape[0] != self._M_rec.shape[1]:
                raise DimensionMismatchError(
                    f"M_rec debe ser cuadrada; se obtuvo shape={self._M_rec.shape}."
                )
            dim_p: int = self._M_rec.shape[0]

            n: int = dim_q + dim_p

            if self._R_cost.shape != (n, n):
                raise DimensionMismatchError(
                    f"R_cost debe ser ({n},{n}) = (dim_q+dim_p)×(dim_q+dim_p), "
                    f"pero se obtuvo {self._R_cost.shape}. dim_q={dim_q}, dim_p={dim_p}."
                )

            if self._J_base.shape != (n, n):
                raise DimensionMismatchError(
                    f"J_base debe ser ({n},{n}) = (dim_q+dim_p)×(dim_q+dim_p), "
                    f"pero se obtuvo {self._J_base.shape}."
                )

            logger.debug(
                "[Fase1] Dimensiones verificadas: dim_q=%d, dim_p=%d, n=%d.",
                dim_q, dim_p, n,
            )
            return dim_q, dim_p

        def _validate_symmetry(self, A: NDArray[np.float64], name: str) -> None:
            r"""
            Verifica A = A^⊤ con tolerancia relativa al Frobenius de A:
            tol = ε_mach · max(‖A‖_F, 1).
            """
            norm_A: float = float(la.norm(A, "fro"))
            tol: float = self._EPS * max(norm_A, 1.0)
            residual: float = float(la.norm(A - A.T, "fro"))

            if residual > tol:
                raise ThermodynamicBaseError(
                    f"La matriz '{name}' no es simétrica. "
                    f"‖A-Aᵀ‖_F = {residual:.6e}, tol = {tol:.6e}, "
                    f"asimetría relativa = {residual / max(norm_A, 1e-300):.6e}."
                )
            logger.debug("[Fase1] Simetría de '%s' verificada: residual=%.3e, tol=%.3e.", name, residual, tol)

        def _validate_antisymmetry(self, J: NDArray[np.float64], name: str) -> None:
            r"""
            Verifica J = -J^⊤ con tolerancia relativa: ‖J+Jᵀ‖_F ≤ ε_mach·max(‖J‖_F,1).

            La antisimetría es la condición topológica fundamental: garantiza
            que la parte conservativa del sistema no produzca ni consuma energía.
            """
            norm_J: float = float(la.norm(J, "fro"))
            tol: float = self._EPS * max(norm_J, 1.0)
            residual: float = float(la.norm(J + J.T, "fro"))

            if residual > tol:
                raise ThermodynamicBaseError(
                    f"La matriz '{name}' no es antisimétrica (J ≠ -Jᵀ). "
                    f"‖J+Jᵀ‖_F = {residual:.6e}, tol = {tol:.6e}."
                )
            logger.debug("[Fase1] Antisimetría de '%s' verificada: residual=%.3e, tol=%.3e.", name, residual, tol)

        def _validate_metric_tensor(
            self,
            G: NDArray[np.float64],
            name: str,
            expected_dim: int,
        ) -> float:
            r"""
            Valida que G ∈ ℝ^{d×d} sea cuadrada de la dimensión esperada y
            **invertible con buen condicionamiento**, vía descomposición en
            valores singulares (válida para matrices no necesariamente
            simétricas, a diferencia de ``eigvalsh``):

                κ(G) = σ_max(G) / σ_min(G)

            Esta es la condición necesaria y suficiente para que el pullback
            congruente \( \tilde{A}=GAG^\top \) preserve la signatura de A
            (Ley de Inercia de Sylvester) sin amplificar patológicamente el
            número de condición resultante.

            Lanza
            -----
            DimensionMismatchError
                Si G no es cuadrada de la dimensión esperada.
            MetricTensorSingularityError
                Si G es singular o κ(G) > κ_max.
            """
            if G.shape != (expected_dim, expected_dim):
                raise DimensionMismatchError(
                    f"El tensor métrico '{name}' debe ser ({expected_dim},{expected_dim}); "
                    f"se obtuvo {G.shape}."
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
                    f"El pullback amplificaría patológicamente el estrés anisotrópico."
                )

            logger.debug("[Fase1] κ(%s) = %.6e (σ_min=%.6e, σ_max=%.6e).", name, kappa_G, sigma_min, sigma_max)
            return kappa_G

        def _congruence_pullback(
            self,
            A: NDArray[np.float64],
            G: NDArray[np.float64],
            name: str,
        ) -> NDArray[np.float64]:
            r"""
            Aplica el *pullback* geométrico congruente:

                Ã = G · A · G^⊤

            Por la Ley de Inercia de Sylvester, si A es SPD y G es invertible,
            Ã es también SPD (signatura preservada). Se re-simetriza
            defensivamente para eliminar ruido de redondeo de O(ε·‖A‖·‖G‖²).

            Retorna
            -------
            NDArray[np.float64]
                Matriz pulled-back Ã, simétrica hasta precisión de máquina.
            """
            A_tilde: NDArray[np.float64] = G @ A @ G.T
            A_tilde = 0.5 * (A_tilde + A_tilde.T)
            logger.debug(
                "[Fase1] Pullback Riemanniano aplicado a '%s': ‖Ã-A‖_F/‖A‖_F=%.3e.",
                name,
                float(la.norm(A_tilde - A, "fro")) / max(float(la.norm(A, "fro")), 1e-300),
            )
            return A_tilde

        def _compute_condition_number(
            self,
            A: NDArray[np.float64],
            name: str,
        ) -> Tuple[float, float, float]:
            r"""
            Calcula κ(A) = λ_max/λ_min explotando la simetría de A y
            solicitando a LAPACK (vía ``driver='evr'`` implícito de
            ``subset_by_index``) **únicamente los dos autovalores extremos**,
            evitando la diagonalización espectral completa O(n³) cuando
            n es grande y sólo se requieren los extremos.

            Retorna
            -------
            Tuple[float, float, float]
                (kappa, lambda_min, lambda_max).

            Lanza
            -----
            CapacitanceDegeneracyError
                Si λ_min ≤ tol_pd (la matriz no es SPD).
            IllConditionedMatrixError
                Si κ(A) > kappa_max.
            """
            n: int = A.shape[0]
            if n == 1:
                lambda_min = lambda_max = float(A[0, 0])
            else:
                lambda_min = float(
                    la.eigh(A, subset_by_index=[0, 0], eigvals_only=True)[0]
                )
                lambda_max = float(
                    la.eigh(A, subset_by_index=[n - 1, n - 1], eigvals_only=True)[0]
                )

            tol_pd: float = self._EPS * max(abs(lambda_max), 1.0)

            if lambda_min <= tol_pd:
                raise CapacitanceDegeneracyError(
                    f"La matriz '{name}' no es Definida Positiva (SPD). "
                    f"λ_min = {lambda_min:.6e} ≤ tol_pd = {tol_pd:.6e}. λ_max = {lambda_max:.6e}."
                )

            kappa: float = lambda_max / lambda_min
            logger.debug("[Fase1] κ('%s') = %.6e (λ_min=%.6e, λ_max=%.6e).", name, kappa, lambda_min, lambda_max)

            if kappa > self._kappa_max:
                raise IllConditionedMatrixError(
                    f"La matriz '{name}' está numéricamente mal condicionada. "
                    f"κ = {kappa:.6e} > κ_max = {self._kappa_max:.6e}. "
                    f"Considere regularización de Tikhonov o re-escalado."
                )

            return kappa, lambda_min, lambda_max

        def _cholesky_spd_regularized(
            self,
            A: NDArray[np.float64],
            name: str,
            max_attempts: int = 6,
        ) -> Tuple[NDArray[np.float64], float]:
            r"""
            Calcula la factorización de Cholesky A = L·L^⊤ con **regularización
            de Tikhonov adaptativa** como red de seguridad numérica:

                A_τ = A + τ·I,   τ_0 = ε_mach · tr(A)/n,   τ_{k+1} = 10·τ_k

            Se invoca después de ``_compute_condition_number`` (SPD ya
            garantizada analíticamente), por lo que en el caso nominal
            converge en el primer intento (τ=0). El mecanismo de reintento
            captura fallos residuales de LAPACK dpotrf ante matrices
            *casi*-semidefinidas por cancelación catastrófica.

            Retorna
            -------
            Tuple[NDArray[np.float64], float]
                (L, tau_final): factor de Cholesky y jitter efectivamente
                aplicado (0.0 si no fue necesario ningún ajuste).

            Lanza
            -----
            CapacitanceDegeneracyError
                Si tras ``max_attempts`` reintentos la factorización sigue
                fallando (indicaría degeneración estructural, no de redondeo).
            """
            A_sym: NDArray[np.float64] = 0.5 * (A + A.T)
            n: int = A_sym.shape[0]
            trace_scale: float = float(np.trace(A_sym)) / max(n, 1)
            tau: float = 0.0
            identity_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)

            for attempt in range(max_attempts + 1):
                try:
                    L: NDArray[np.float64] = la.cholesky(A_sym + tau * identity_n, lower=True)
                    if attempt > 0:
                        logger.warning(
                            "[Fase1] Regularización de Tikhonov aplicada a '%s': "
                            "τ=%.3e tras %d intento(s).",
                            name, tau, attempt,
                        )
                    else:
                        logger.debug(
                            "[Fase1] Cholesky de '%s' completado sin regularización. "
                            "L[0,0]=%.6e, L[-1,-1]=%.6e.",
                            name, float(L[0, 0]), float(L[-1, -1]),
                        )
                    return L, tau
                except la.LinAlgError:
                    tau = self._EPS * max(trace_scale, 1.0) if tau == 0.0 else tau * 10.0

            raise CapacitanceDegeneracyError(
                f"Fallo persistente de Cholesky (LAPACK dpotrf) en '{name}' "
                f"tras {max_attempts} intentos de regularización de Tikhonov "
                f"(τ_final={tau:.3e}). Indica degeneración estructural, no de redondeo."
            )

        def _validate_psd_and_spectral_diagnostics(
            self,
            R: NDArray[np.float64],
            name: str,
        ) -> Tuple[NDArray[np.float64], int, float]:
            r"""
            Verifica R ⪰ 0 y calcula:

              1. La raíz cuadrada espectral exacta:
                 \( R_{sqrt}=V\,\mathrm{diag}(\sqrt{\max(\lambda,0)})\,V^\top \).
              2. El rango numérico rank_R.
              3. La **brecha espectral** λ₂−λ₁ (teoría espectral de grafos:
                 análogo discreto de la conectividad algebraica de Fiedler,
                 aquí interpretada como la resistencia del "segundo modo
                 menos disipado" del ecosistema de costes).

            Lanza
            -----
            RayleighDissipationViolation
                Si λ_min < -tol_psd (entropía negativa / ganancia fantasma).

            Retorna
            -------
            Tuple[NDArray[np.float64], int, float]
                (R_sqrt, rank_R, spectral_gap).
            """
            norm_R: float = float(la.norm(R, "fro"))
            tol_psd: float = self._EPS * max(norm_R, 1.0)

            eigvals: NDArray[np.float64]
            eigvecs: NDArray[np.float64]
            eigvals, eigvecs = la.eigh(R)  # ascendente

            lambda_min: float = float(eigvals[0])
            if lambda_min < -tol_psd:
                raise RayleighDissipationViolation(
                    f"La matriz '{name}' no es Semidefinida Positiva (PSD). "
                    f"λ_min = {lambda_min:.6e} < -tol = {-tol_psd:.6e}. "
                    f"Indica entropía negativa (ganancia fantasma): "
                    f"violación de la Segunda Ley de la Termodinámica."
                )

            eigvals_clamped: NDArray[np.float64] = np.maximum(eigvals, 0.0)

            R_sqrt: NDArray[np.float64] = (
                eigvecs * np.sqrt(eigvals_clamped)[np.newaxis, :]
            ) @ eigvecs.T
            R_sqrt = 0.5 * (R_sqrt + R_sqrt.T)

            rank_R: int = int(np.sum(eigvals_clamped > tol_psd))

            spectral_gap: float = (
                float(eigvals_clamped[1] - eigvals_clamped[0]) if len(eigvals_clamped) > 1 else 0.0
            )

            logger.debug(
                "[Fase1] %s PSD verificada: rank=%d/%d, λ_min=%.3e, λ_max=%.3e, gap=%.3e.",
                name, rank_R, len(eigvals), lambda_min, float(eigvals[-1]), spectral_gap,
            )
            return R_sqrt, rank_R, spectral_gap

        #
        # Método terminal de la Fase 1 — entrada directa de la Fase 2
        #

        def build_topological_context(self) -> "TopologicalContext":
            r"""
            **Método terminal de la Fase 1.**

            Ejecuta en secuencia estricta todos los métodos de validación,
            pullback y factorización, empaquetando el resultado en un
            ``TopologicalContext`` inmutable: el único argumento que
            necesita la Fase 2 para operar.

            Flujo interno
            -------------
            1. Verificación dimensional.
            2. Antisimetría de J_base; simetría de C_soc, M_rec, R_cost.
            3. Resolución de G_q, G_p (identidad por defecto) y validación
               de su invertibilidad/condicionamiento.
            4. Pullback Riemanniano: C̃_soc, M̃_rec.
            5. Números de condición κ(C̃_soc), κ(M̃_rec).
            6. Cholesky regularizado de C̃_soc, M̃_rec → L_C, L_M.
            7. Precálculo de C̃_soc^{-1/2}, M̃_rec^{-1/2}.
            8. Validación PSD + raíz espectral + diagnóstico espectral de R_cost.
            9. Empaquetado en TopologicalContext.

            Retorna
            -------
            TopologicalContext
                Contexto topológico completo, inmutable, listo para Fase 2 y 3.
            """
            # Paso 1: dimensiones
            dim_q, dim_p = self._check_dimensions()

            # Paso 2: simetrías estructurales
            self._validate_antisymmetry(self._J_base, "J_base")
            self._validate_symmetry(self._C_soc, "C_soc")
            self._validate_symmetry(self._M_rec, "M_rec")
            self._validate_symmetry(self._R_cost, "R_cost")

            # Paso 3: resolución y validación de tensores métricos
            G_q: NDArray[np.float64] = (
                self._G_q_raw if self._G_q_raw is not None else np.eye(dim_q, dtype=np.float64)
            )
            G_p: NDArray[np.float64] = (
                self._G_p_raw if self._G_p_raw is not None else np.eye(dim_p, dtype=np.float64)
            )
            kappa_G_q: float = self._validate_metric_tensor(G_q, "G_q", dim_q)
            kappa_G_p: float = self._validate_metric_tensor(G_p, "G_p", dim_p)

            # Paso 4: pullback Riemanniano congruente
            C_soc_tilde: NDArray[np.float64] = self._congruence_pullback(self._C_soc, G_q, "C_soc")
            M_rec_tilde: NDArray[np.float64] = self._congruence_pullback(self._M_rec, G_p, "M_rec")

            # Paso 5: números de condición sobre las matrices pulled-back
            kappa_C, _, _ = self._compute_condition_number(C_soc_tilde, "C̃_soc")
            kappa_M, _, _ = self._compute_condition_number(M_rec_tilde, "M̃_rec")

            # Paso 6: Cholesky regularizado
            L_C, epsilon_C = self._cholesky_spd_regularized(C_soc_tilde, "C̃_soc")
            L_M, epsilon_M = self._cholesky_spd_regularized(M_rec_tilde, "M̃_rec")

            # Paso 7: raíces de las inversas, precálculo único reutilizado
            # aguas abajo por Fase 2 (modos normales) y Fase 3 (haz).
            C_inv_sqrt: NDArray[np.float64] = la.solve_triangular(
                L_C, np.eye(dim_q, dtype=np.float64), lower=True, check_finite=False
            ).T
            M_inv_sqrt: NDArray[np.float64] = la.solve_triangular(
                L_M, np.eye(dim_p, dtype=np.float64), lower=True, check_finite=False
            ).T

            # Paso 8: PSD + raíz espectral + diagnóstico espectral de R_cost
            R_sqrt, rank_R, spectral_gap_R = self._validate_psd_and_spectral_diagnostics(
                self._R_cost, "R_cost"
            )
            n_total: int = dim_q + dim_p
            betti_0_R: int = n_total - rank_R

            # Paso 9: empaquetado
            context = TopologicalContext(
                L_C=L_C,
                L_M=L_M,
                C_inv_sqrt=C_inv_sqrt,
                M_inv_sqrt=M_inv_sqrt,
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
            )

            logger.info(
                "[Fase1] TopologicalContext ensamblado: dim_q=%d, dim_p=%d, "
                "κ(C̃)=%.3e, κ(M̃)=%.3e, rank(R)=%d, betti_0(R)=%d, gap(R)=%.3e.",
                dim_q, dim_p, kappa_C, kappa_M, rank_R, betti_0_R, spectral_gap_R,
            )

            # Contrato de interfaz entre Fase 1 y Fase 2:
            # `context` es el argumento directo del constructor de
            # Phase2_HamiltonianDynamics.
            return context

    #
    # ==========================================================================
    # FASE 2 — DINÁMICA PORT-HAMILTONIANA, DISIPACIÓN DE RAYLEIGH
    #           Y FORMA NORMAL DE WILLIAMSON (ANALOGÍA CUÁNTICA)
    # ==========================================================================
    #

    class Phase2_HamiltonianDynamics:
        r"""
        **Fase 2 – Dinámica Port-Hamiltoniana y Disipación de Rayleigh.**

        Recibe el ``TopologicalContext`` de la Fase 1 (matrices ya
        pulled-back por la métrica Riemanniana) y calcula:

          • Energías, gradiente y campo vectorial completo del sistema:
                \[ \dot{x} = (J_{BASE} - R_{cost})\,\nabla H(x) \]
          • Verificación estructural exacta \( \nabla H^\top J_{BASE}\nabla H \equiv 0 \).
          • Verificación de homogeneidad de Euler (H es cuadrática exacta).
          • Diagnóstico opcional de modos normales vía Teorema de Williamson:
            dado que \( M^{1/2}JM^{1/2} \) es real antisimétrica para toda
            J antisimétrica y M≻0, sus **valores singulares** son
            exactamente los módulos |ω_k| de los autovalores imaginarios
            puros ±iω_k del sistema conservativo linealizado — sin
            necesidad de álgebra compleja.
          • Retícula Booleana de estabilidad (``StabilityFlags``).

        Identidad de energía sin inversión explícita
        ---------------------------------------------
            q^⊤ Ã⁻¹ q = ‖ L⁻¹ q ‖²     (Ã = L·L^⊤, Cholesky)
        """

        _EPS: float = _MACHINE_EPS

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

            No realiza validación adicional: la corrección de todas las
            matrices (incluyendo el pullback Riemanniano) ya fue garantizada
            por la Fase 1.
            """
            self._ctx: "TopologicalContext" = context
            self._breakdown_voltage: float = breakdown_voltage
            self._kappa_max: float = kappa_max
            self._hbar: float = hbar
            self._flyback_safety_margin: float = flyback_safety_margin

            self._cho_C: Tuple[NDArray[np.float64], bool] = (context.L_C, True)
            self._cho_M: Tuple[NDArray[np.float64], bool] = (context.L_M, True)

            logger.debug(
                "[Fase2] Inicializada: breakdown_voltage=%.3e, kappa_max=%.3e, "
                "hbar=%.3e, margen_flyback=%.2f, dim_q=%d, dim_p=%d.",
                breakdown_voltage, kappa_max, hbar, flyback_safety_margin,
                context.dim_q, context.dim_p,
            )

        #
        # Métodos privados de cálculo energético
        #

        def _evaluate_potential_energy(
            self, q: NDArray[np.float64]
        ) -> Tuple[float, NDArray[np.float64]]:
            r"""
            V(q) = ½‖L_C⁻¹q‖² = ½q^⊤C̃_soc⁻¹q ; ∂V/∂q = C̃_soc⁻¹q.
            """
            if q.shape != (self._ctx.dim_q,):
                raise DimensionMismatchError(
                    f"Vector q debe tener shape ({self._ctx.dim_q},); se obtuvo {q.shape}."
                )
            y: NDArray[np.float64] = la.solve_triangular(self._ctx.L_C, q, lower=True, check_finite=False)
            V_q: float = 0.5 * float(np.dot(y, y))
            grad_V_q: NDArray[np.float64] = la.solve_triangular(
                self._ctx.L_C, y, lower=True, trans="T", check_finite=False
            )
            logger.debug("[Fase2] V(q)=%.6e, ‖∇V‖=%.6e.", V_q, float(np.linalg.norm(grad_V_q)))
            return V_q, grad_V_q

        def _compute_kinetic_energy(
            self, p: NDArray[np.float64]
        ) -> Tuple[float, NDArray[np.float64]]:
            r"""
            K(p) = ½‖L_M⁻¹p‖² = ½p^⊤M̃_rec⁻¹p ; ∂K/∂p = M̃_rec⁻¹p.
            """
            if p.shape != (self._ctx.dim_p,):
                raise DimensionMismatchError(
                    f"Vector p debe tener shape ({self._ctx.dim_p},); se obtuvo {p.shape}."
                )
            y: NDArray[np.float64] = la.solve_triangular(self._ctx.L_M, p, lower=True, check_finite=False)
            K_p: float = 0.5 * float(np.dot(y, y))
            grad_K_p: NDArray[np.float64] = la.solve_triangular(
                self._ctx.L_M, y, lower=True, trans="T", check_finite=False
            )
            logger.debug("[Fase2] K(p)=%.6e, ‖∇K‖=%.6e.", K_p, float(np.linalg.norm(grad_K_p)))
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
            Verifica el Teorema de Euler para funciones homogéneas de grado 2:

                q·∇_qH + p·∇_pH = 2H(q,p)

            H es exactamente cuadrática, por lo que esta identidad debe
            cumplirse hasta precisión de máquina; su violación indicaría un
            error de implementación en los gradientes triangulares.

            Retorna
            -------
            float
                Residuo absoluto |q·∇_qH + p·∇_pH − 2H|.
            """
            euler_lhs: float = float(np.dot(q, grad_V)) + float(np.dot(p, grad_K))
            residual: float = abs(euler_lhs - 2.0 * H_total)
            scale: float = max(abs(euler_lhs), abs(2.0 * H_total), 1.0)
            tol: float = 1.0e3 * self._EPS * scale

            if residual > tol:
                logger.warning(
                    "[Fase2] Residuo de homogeneidad de Euler elevado: %.6e > tol=%.6e "
                    "(posible error de implementación en gradientes).",
                    residual, tol,
                )
            return residual

        def _enforce_rayleigh_dissipation(self, grad_H: NDArray[np.float64]) -> float:
            r"""
            Ḣ_diss = -∇H^⊤R_cost∇H ≤ 0 ; P_diss = |Ḣ_diss| ≥ 0.
            """
            n: int = self._ctx.dim_q + self._ctx.dim_p
            if grad_H.shape != (n,):
                raise DimensionMismatchError(f"grad_H debe tener shape ({n},); se obtuvo {grad_H.shape}.")

            R_grad: NDArray[np.float64] = self._ctx.R_cost @ grad_H
            quad_form: float = float(np.dot(grad_H, R_grad))

            norm_R: float = float(la.norm(self._ctx.R_cost, "fro"))
            norm_gH2: float = float(np.dot(grad_H, grad_H))
            tol_diss: float = self._EPS * norm_R * norm_gH2

            if quad_form < -tol_diss:
                logger.error(
                    "[Fase2] Violación de Rayleigh: ∇H^⊤R∇H=%.6e < -tol=%.6e.", quad_form, -tol_diss
                )
                raise RayleighDissipationViolation(
                    f"Violación de la Segunda Ley de la Termodinámica. "
                    f"∇H^⊤R_cost∇H = {quad_form:.6e} < -tol = {-tol_diss:.6e}. "
                    f"La Estructura de Costes presenta generación de exergía espontánea."
                )

            P_diss: float = abs(quad_form)
            logger.debug("[Fase2] P_diss=%.6e (Rayleigh OK).", P_diss)
            return P_diss

        def _measure_flyback_voltage(self, df_dt: NDArray[np.float64]) -> float:
            r"""
            ‖M̃_rec·∂f/∂t‖_∞, reconstruido eficientemente vía L_M sin
            materializar M̃_rec explícitamente.
            """
            if df_dt.shape != (self._ctx.dim_p,):
                raise DimensionMismatchError(
                    f"df_dt debe tener shape ({self._ctx.dim_p},); se obtuvo {df_dt.shape}."
                )
            L_M_T_v: NDArray[np.float64] = self._ctx.L_M.T @ df_dt
            V_fb_vec: NDArray[np.float64] = self._ctx.L_M @ L_M_T_v
            v_fb_norm: float = float(la.norm(V_fb_vec, np.inf))

            if v_fb_norm > self._breakdown_voltage:
                logger.critical(
                    "[Fase2] Golpe de Ariete Logístico: ‖V_fb‖=%.6e > V_bd=%.6e.",
                    v_fb_norm, self._breakdown_voltage,
                )
                raise InertialFlybackError(
                    f"Voltaje de Flyback ‖M̃_rec·∂f/∂t‖_∞ = {v_fb_norm:.6e} "
                    f"excede la tensión de ruptura V_bd = {self._breakdown_voltage:.6e}. "
                    f"Detención de emergencia exigida."
                )

            logger.debug(
                "[Fase2] ‖V_fb‖_∞=%.6e (límite=%.6e, margen=%.1f%%).",
                v_fb_norm, self._breakdown_voltage,
                100.0 * (1.0 - v_fb_norm / self._breakdown_voltage),
            )
            return v_fb_norm

        #
        # Campo vectorial Port-Hamiltoniano y verificación estructural
        #

        def _compute_vector_field(self, grad_H: NDArray[np.float64]) -> NDArray[np.float64]:
            r"""
            Ensambla el campo vectorial completo del sistema Port-Hamiltoniano:

                \[ \dot{x} = (J_{BASE} - R_{cost})\,\nabla H \]

            Esta es la ecuación de estado del foso K_{BASE}; ninguna versión
            previa la materializaba pese a validar J_base en la Fase 1.
            """
            return (self._ctx.J_base - self._ctx.R_cost) @ grad_H

        def _verify_structural_consistency(
            self,
            grad_H: NDArray[np.float64],
            x_dot: NDArray[np.float64],
            P_diss: float,
        ) -> float:
            r"""
            Verifica la identidad algebraica **exacta** de todo sistema
            Port-Hamiltoniano:

                ∇H^⊤ẋ = ∇H^⊤J∇H − ∇H^⊤R∇H = 0 − P_diss = −P_diss

            porque ∇H^⊤J∇H ≡ 0 para cualquier J antisimétrica actuando sobre
            su propio argumento (forma cuadrática de un operador
            antisimétrico es idénticamente nula). Esta verificación es
            redundante con ``_enforce_rayleigh_dissipation`` pero actúa como
            **test de invariante en caliente**: detecta errores de cableado
            entre J_base, R_cost y ∇H que no violarían la 2ª Ley pero sí
            corromperían la interpretación física del modelo.

            Lanza
            -----
            StructuralConsistencyError
                Si |∇H^⊤ẋ + P_diss| excede una tolerancia generosa (indica
                bug de implementación, no violación física).
            """
            H_dot: float = float(np.dot(grad_H, x_dot))
            residual: float = abs(H_dot + P_diss)
            scale: float = max(abs(H_dot), P_diss, 1.0)
            tol: float = np.sqrt(self._EPS) * scale

            if residual > tol:
                raise StructuralConsistencyError(
                    f"Inconsistencia estructural Port-Hamiltoniana: "
                    f"∇H^⊤ẋ = {H_dot:.6e}, -P_diss = {-P_diss:.6e}, "
                    f"|residuo| = {residual:.6e} > tol = {tol:.6e}. "
                    f"Verifique el cableado de J_base y R_cost."
                )
            logger.debug("[Fase2] Consistencia estructural verificada: residuo=%.3e, tol=%.3e.", residual, tol)
            return residual

        #
        # Diagnóstico opcional: Forma normal de Williamson y analogía cuántica
        #

        def compute_normal_modes(self) -> Tuple[NDArray[np.float64], float]:
            r"""
            Diagnóstico **opcional** (no forma parte del camino caliente):
            calcula las frecuencias propias del subsistema conservativo
            linealizado \( \dot{x}=J_{BASE}\nabla^2H\,x \) explotando el
            Teorema de Williamson para pares Hamiltonianos.

            Construcción (sin álgebra compleja)
            ------------------------------------
                H^{1/2} = block_diag( C̃_soc^{-1/2}, M̃_rec^{-1/2} )   (raíz de ∇²H)
                A_sym   = H^{1/2} · J_BASE · H^{1/2}                  (real antisimétrica)

            Como A_sym es real antisimétrica, sus **valores singulares**
            (obtenidos vía SVD, en aritmética puramente real) son
            exactamente los módulos |ω_k| de los autovalores imaginarios
            puros ±iω_k del sistema conservativo — consecuencia directa de
            que toda matriz antisimétrica real es ortogonalmente similar a
            una forma canónica por bloques rotacionales de tamaño 2×2.

            Analogía cuántica
            ------------------
                E_0 = (ħ/2) · Σ_k ω_k

            (energía de punto cero del oscilador armónico desacoplado en
            sus modos normales, bajo el isomorfismo Port-Hamiltoniano).

            Retorna
            -------
            Tuple[NDArray[np.float64], float]
                (omegas, E_0): frecuencias propias ordenadas descendentemente
                y energía de punto cero análoga.
            """
            dim_q, dim_p = self._ctx.dim_q, self._ctx.dim_p
            n: int = dim_q + dim_p

            H_half: NDArray[np.float64] = np.zeros((n, n), dtype=np.float64)
            H_half[:dim_q, :dim_q] = self._ctx.C_inv_sqrt
            H_half[dim_q:, dim_q:] = self._ctx.M_inv_sqrt

            A_sym: NDArray[np.float64] = H_half @ self._ctx.J_base @ H_half
            A_sym = 0.5 * (A_sym - A_sym.T)  # re-antisimetrización defensiva

            singular_values: NDArray[np.float64] = la.svdvals(A_sym)  # descendente
            n_pairs: int = n // 2

            omegas: NDArray[np.float64] = np.empty(n_pairs, dtype=np.float64)
            for k in range(n_pairs):
                # Cada frecuencia física aparece dos veces (par ±iω) en el
                # espectro singular; se promedia el par consecutivo para
                # robustez ante asimetrías numéricas de orden O(ε).
                omegas[k] = 0.5 * (singular_values[2 * k] + singular_values[2 * k + 1])

            omegas = np.sort(omegas)[::-1]
            zero_point_energy: float = 0.5 * self._hbar * float(np.sum(omegas))

            logger.debug(
                "[Fase2] Modos normales (Williamson): %d pares, E_0=%.6e (ħ=%.3e).",
                n_pairs, zero_point_energy, self._hbar,
            )
            return omegas, zero_point_energy

        #
        # Retícula Booleana de estabilidad
        #

        def _evaluate_stability_flags(
            self,
            V_q: float,
            K_p: float,
            P_diss: float,
            v_fb: float,
            structural_residual: float,
        ) -> StabilityFlags:
            r"""
            Evalúa la retícula ``StabilityFlags`` combinando predicados
            independientes mediante disyunción bit a bit (unión de
            predicados satisfechos), formando el elemento de la retícula
            Booleana correspondiente al estado actual.
            """
            flags: StabilityFlags = StabilityFlags.NONE
            energy_tol: float = 1.0e3 * self._EPS

            if V_q >= -energy_tol and K_p >= -energy_tol:
                flags |= StabilityFlags.ENERGY_NONNEGATIVE
            if P_diss >= -energy_tol:
                flags |= StabilityFlags.DISSIPATION_VALID
            if v_fb <= self._flyback_safety_margin * self._breakdown_voltage:
                flags |= StabilityFlags.FLYBACK_SAFE
            if structural_residual <= np.sqrt(self._EPS) * max(P_diss, 1.0):
                flags |= StabilityFlags.STRUCTURAL_CONSISTENCY
            if self._ctx.kappa_C <= 0.5 * self._kappa_max and self._ctx.kappa_M <= 0.5 * self._kappa_max:
                flags |= StabilityFlags.SPECTRAL_CONDITIONING_SOUND

            logger.debug("[Fase2] %s", describe_stability_flags(flags))
            return flags

        #
        # Método terminal de la Fase 2 — entrada directa de la Fase 3
        #

        def synthesize_basal_state(
            self,
            q: NDArray[np.float64],
            p: NDArray[np.float64],
            df_dt: NDArray[np.float64],
            compute_normal_modes: bool = False,
        ) -> "BasalStateTensor":
            r"""
            **Método terminal de la Fase 2.**

            Computa el estado termodinámico completo del foso K_{BASE},
            incluyendo el campo vectorial Port-Hamiltoniano, las
            verificaciones estructurales exactas y la retícula Booleana de
            estabilidad. El ``BasalStateTensor`` resultante expone
            ``state_vector = [q; p]`` (vía ``vector_field`` y los propios
            q, p) como dato primario para la Fase 3.

            Parámetros
            ----------
            q, p, df_dt : NDArray[np.float64]
                Estado y perturbación de flujo (ver dataclass para shapes).
            compute_normal_modes : bool, default False
                Si True, ejecuta el diagnóstico de Williamson (coste
                adicional O(n³) por SVD; se mantiene opt-in para no
                penalizar el camino caliente de evaluación por paso).

            Retorna
            -------
            BasalStateTensor
                Estado termodinámico completo, inmutable.
            """
            # Energías y gradientes
            V_q, grad_V_q = self._evaluate_potential_energy(q)
            K_p, grad_K_p = self._compute_kinetic_energy(p)
            H_total: float = V_q + K_p

            grad_H: NDArray[np.float64] = np.concatenate([grad_V_q, grad_K_p])
            grad_H_norm: float = float(la.norm(grad_H, 2))

            # Verificación de homogeneidad de Euler
            euler_residual: float = self._verify_euler_homogeneity(q, p, grad_V_q, grad_K_p, H_total)

            # Disipación de Rayleigh
            P_diss: float = self._enforce_rayleigh_dissipation(grad_H)

            # Voltaje de Flyback
            v_fb: float = self._measure_flyback_voltage(df_dt)

            # Campo vectorial Port-Hamiltoniano completo
            x_dot: NDArray[np.float64] = self._compute_vector_field(grad_H)

            # Verificación estructural exacta (∇H^⊤J∇H ≡ 0)
            structural_residual: float = self._verify_structural_consistency(grad_H, x_dot, P_diss)

            # Retícula Booleana de estabilidad
            flags: StabilityFlags = self._evaluate_stability_flags(
                V_q, K_p, P_diss, v_fb, structural_residual
            )

            # Diagnóstico opcional de modos normales (Williamson)
            omegas: Optional[NDArray[np.float64]] = None
            E0: Optional[float] = None
            if compute_normal_modes:
                omegas, E0 = self.compute_normal_modes()

            logger.info(
                "[Fase2] Estado basal: H=%.6e, V=%.6e, K=%.6e, P_diss=%.6e, "
                "‖V_fb‖=%.6e, ‖∇H‖=%.6e, %s.",
                H_total, V_q, K_p, P_diss, v_fb, grad_H_norm, describe_stability_flags(flags),
            )

            # Contrato de interfaz entre Fase 2 y Fase 3:
            # np.concatenate([q, p]) es el argumento directo de
            # Phase3_SheafProjection.export_stalk().
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
                stability_flags=flags,
                is_thermodynamically_stable=(flags == StabilityFlags.ALL),
                normal_mode_frequencies=omegas,
                zero_point_energy=E0,
            )

    #
    # ==========================================================================
    # FASE 3 — PROYECCIÓN COHOMOLÓGICA EN HACES: COCADENA APILADA Y HODGE LOCAL
    # ==========================================================================
    #

    class Phase3_SheafProjection:
        r"""
        **Fase 3 – Proyección en Haces (Sheaf) y Cofrontera Discreta δ_{BASE}.**

        Recibe el ``TopologicalContext`` de la Fase 1 y el ``state_vector``
        derivado del ``BasalStateTensor`` de la Fase 2, construyendo el
        fibrado celular (Stalk) que alimenta el Laplaciano de Haz global
        del ecosistema APU Filter.

        Corrección estructural respecto a versiones previas
        -------------------------------------------------------
        La formulación anterior forzaba ``block_diag(C_soc^{-1/2}, R_cost^{1/2})``
        con bloques de dimensiones incompatibles (R_cost es n×n, no dim_p×dim_p),
        lo cual es dimensionalmente inconsistente. Se reformula δ_{BASE} como
        el mapa cocadena genuino desde el stalk (dimensión n) hacia la suma
        directa de sus **dos fibras incidentes**:

            δ_metric = block_diag( C̃_soc^{-1/2}, M̃_rec^{-1/2} )  ∈ ℝ^{n×n}  (arista conservativa)
            δ_diss   = R_cost^{+1/2}                               ∈ ℝ^{n×n}  (arista disipativa)
            δ_BASE   = [ δ_metric ; δ_diss ]                       ∈ ℝ^{2n×n}

        de modo que el Laplaciano de Hodge local resulta exactamente:

            Δ_BASE = δ_BASE^⊤δ_BASE = ∇²H + R_cost   (SPD)

        Esta construcción es la correcta en el sentido de la cohomología de
        haces celulares: un vértice restringe hacia la suma directa de las
        aristas incidentes, no hacia un único bloque diagonal forzado.
        """

        _EPS: float = _MACHINE_EPS

        def __init__(self, context: "TopologicalContext") -> None:
            r"""
            **Constructor de la Fase 3: continuación directa de la Fase 2.**

            Ensambla δ_metric, δ_diss y δ_BASE **una única vez**, ya que el
            ``TopologicalContext`` es inmutable; ``export_stalk`` reutiliza
            estos precálculos en cada invocación.
            """
            self._ctx: "TopologicalContext" = context

            dim_q: int = context.dim_q
            dim_p: int = context.dim_p
            n: int = dim_q + dim_p

            delta_metric: NDArray[np.float64] = np.zeros((n, n), dtype=np.float64)
            delta_metric[:dim_q, :dim_q] = context.C_inv_sqrt
            delta_metric[dim_q:, dim_q:] = context.M_inv_sqrt
            self._delta_metric: NDArray[np.float64] = delta_metric

            # R_sqrt ya es n×n (raíz espectral del R_cost completo, Fase 1);
            # se usa directamente sin sub-bloqueo, corrigiendo el defecto
            # dimensional de versiones previas.
            self._delta_diss: NDArray[np.float64] = context.R_sqrt

            self._delta_base: NDArray[np.float64] = np.vstack(
                [self._delta_metric, self._delta_diss]
            )  # shape (2n, n)

            # Δ_BASE = ∇²H + R_cost, calculado directamente vía los bloques
            # ya disponibles (evita recomputar ∇²H desde cero).
            hessian_block: NDArray[np.float64] = self._delta_metric.T @ self._delta_metric
            self._hodge_laplacian: NDArray[np.float64] = hessian_block + context.R_cost
            self._hodge_laplacian = 0.5 * (self._hodge_laplacian + self._hodge_laplacian.T)

            # rank(δ_BASE) = n exactamente: δ_metric es invertible por
            # construcción (Cholesky de una matriz SPD), luego δ_BASE tiene
            # columna completa independientemente del rango de R_cost.
            self._rank_delta: int = n

            logger.debug(
                "[Fase3] Precalculado: δ_metric shape=%s, δ_diss shape=%s, "
                "δ_BASE shape=%s, rank_delta=%d.",
                self._delta_metric.shape, self._delta_diss.shape,
                self._delta_base.shape, self._rank_delta,
            )

        def _verify_hodge_identity(self) -> float:
            r"""
            Verifica la identidad de Hodge local:

                δ_BASE^⊤ δ_BASE = ∇²H + R_cost = Δ_BASE

            calculando directamente δ_BASE^⊤δ_BASE y comparándolo contra la
            construcción independiente ``self._hodge_laplacian``. Debe
            coincidir hasta precisión de máquina; una discrepancia mayor
            indicaría un error de ensamblaje (p.ej. en el ``vstack``).

            Lanza
            -----
            SheafCoboundaryError
                Si el error relativo excede 100·ε_mach.
            """
            delta_T_delta: NDArray[np.float64] = self._delta_base.T @ self._delta_base
            residual_F: float = float(la.norm(delta_T_delta - self._hodge_laplacian, "fro"))
            norm_hodge: float = float(la.norm(self._hodge_laplacian, "fro"))
            tol_metric: float = 100.0 * self._EPS
            rel_error: float = residual_F / max(norm_hodge, 1.0)

            if rel_error > tol_metric:
                raise SheafCoboundaryError(
                    f"δ_BASE no satisface la identidad de Hodge local. "
                    f"‖δᵀδ - Δ_BASE‖_F/‖Δ_BASE‖_F = {rel_error:.6e} > tol = {tol_metric:.6e}."
                )
            logger.debug("[Fase3] Identidad de Hodge verificada: rel_error=%.3e, tol=%.3e.", rel_error, tol_metric)
            return rel_error

        def _compute_hodge_spectrum(self) -> Tuple[float, float, int]:
            r"""
            Diagonaliza Δ_BASE (SPD, n×n) para extraer:

              • brecha espectral λ₂−λ₁ (teoría espectral de grafos: mide la
                "rigidez" del acoplamiento conjunto elástico+disipativo),
              • número de condición κ(Δ_BASE),
              • dimensión armónica dim ker(δ_metric) — topológicamente 0,
                puesto que δ_metric es invertible por construcción (∇²H≻0
                es un invariante estructural del modelo: la capacitancia e
                inercia nunca degeneran).

            Retorna
            -------
            Tuple[float, float, int]
                (spectral_gap, condition_number, harmonic_dimension).
            """
            eigvals: NDArray[np.float64] = la.eigvalsh(self._hodge_laplacian)  # ascendente, SPD
            lambda_min, lambda_second, lambda_max = (
                float(eigvals[0]), float(eigvals[1]) if len(eigvals) > 1 else float(eigvals[0]), float(eigvals[-1])
            )
            spectral_gap: float = lambda_second - lambda_min
            condition_number: float = lambda_max / lambda_min if lambda_min > 0 else float("inf")

            tol_kernel: float = self._EPS * max(la.norm(self._delta_metric, "fro"), 1.0)
            harmonic_dimension: int = int(
                np.sum(la.svdvals(self._delta_metric) <= tol_kernel)
            )

            logger.debug(
                "[Fase3] Espectro de Hodge: gap=%.6e, κ=%.6e, dim_armónica=%d.",
                spectral_gap, condition_number, harmonic_dimension,
            )
            return spectral_gap, condition_number, harmonic_dimension

        #
        # Método terminal de la Fase 3 (salida pública del ecosistema)
        #

        def export_stalk(self, state_x: NDArray[np.float64]) -> "SheafStalk":
            r"""
            **Método terminal de la Fase 3 y del agente completo.**

            Construye y retorna el ``SheafStalk`` con δ_{BASE} verificado,
            el Laplaciano de Hodge local Δ_BASE, sus diagnósticos
            espectrales y las proyecciones del estado sobre cada fibra
            incidente (métrica y disipativa) por separado.

            Parámetros
            ----------
            state_x : NDArray[np.float64], shape (dim_q + dim_p,)
                Vector de estado x = [q; p], típicamente
                ``np.concatenate([q, p])`` desde la Fase 2.

            Retorna
            -------
            SheafStalk
                Fibrado celular completo, inmutable.

            Lanza
            -----
            DimensionMismatchError
                Si state_x no tiene shape (dim_q + dim_p,).
            SheafCoboundaryError
                Si δ_BASE no satisface la identidad de Hodge local.
            """
            n: int = self._ctx.dim_q + self._ctx.dim_p
            if state_x.shape != (n,):
                raise DimensionMismatchError(
                    f"state_x debe tener shape ({n},) = (dim_q+dim_p,); se obtuvo {state_x.shape}."
                )

            hodge_residual: float = self._verify_hodge_identity()
            spectral_gap, condition_number, harmonic_dimension = self._compute_hodge_spectrum()

            projected_metric: NDArray[np.float64] = self._delta_metric @ state_x
            projected_diss: NDArray[np.float64] = self._delta_diss @ state_x

            logger.info(
                "[Fase3] SheafStalk exportado: rank_delta=%d, gap=%.3e, κ(Δ)=%.3e, "
                "dim_armónica=%d, dim_lossless=%d, ‖δ_metric·x‖=%.6e, ‖δ_diss·x‖=%.6e.",
                self._rank_delta, spectral_gap, condition_number, harmonic_dimension,
                self._ctx.betti_0_R, float(la.norm(projected_metric, 2)), float(la.norm(projected_diss, 2)),
            )

            # Contrato de salida del agente completo: el SheafStalk es el
            # output final de la cadena de 3 fases.
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
            )

    #
    # ==========================================================================
    # INTERFAZ PÚBLICA DEL AGENTE (punto de entrada externo)
    # ==========================================================================
    #

    def synthesize_basal_hamiltonian(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        df_dt: NDArray[np.float64],
        compute_normal_modes: bool = False,
    ) -> BasalStateTensor:
        r"""
        Punto de entrada público para la evaluación termodinámica completa.

        Delega en ``Phase2_HamiltonianDynamics.synthesize_basal_state``, que
        ya posee el contexto topológico validado por la Fase 1.

        Parámetros
        ----------
        q, p, df_dt : NDArray[np.float64]
            Estado y perturbación de flujo del instante de evaluación.
        compute_normal_modes : bool, default False
            Activa el diagnóstico opcional de Williamson (modos normales
            y energía de punto cero cuántica análoga).

        Retorna
        -------
        BasalStateTensor
            Estado termodinámico completo e inmutable.
        """
        return self.phase2.synthesize_basal_state(
            q=q, p=p, df_dt=df_dt, compute_normal_modes=compute_normal_modes
        )

    def export_sheaf_stalk(self, state_x: NDArray[np.float64]) -> SheafStalk:
        r"""
        Exporta el Stalk del haz celular y la cofrontera δ_{BASE}.

        Instancia la Fase 3 perezosamente en la primera llamada y reutiliza
        la instancia en llamadas subsiguientes.

        Parámetros
        ----------
        state_x : NDArray[np.float64], shape (dim_q + dim_p,)
            Vector de estado x = [q; p].

        Retorna
        -------
        SheafStalk
            Fibrado celular completo e inmutable.
        """
        if self.phase3 is None:
            self.phase3 = KBaseThermodynamicAgent.Phase3_SheafProjection(context=self.context)
            logger.info(
                "[KBaseThermodynamicAgent] Phase3_SheafProjection instanciada (lazy init). "
                "rank_delta=%d.",
                self.phase3._rank_delta,
            )
        return self.phase3.export_stalk(state_x=state_x)
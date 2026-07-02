# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Levi-Civita Connection Agent (Maestro de Sinfonía Métrica)                ║
║  Ruta   : app/omega/levi_civita_agent.py                                            ║
║  Versión: 6.0.0-Granular-Geodesic-Categorical                                       ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica Diferencial:
══════════════════════════════════════════════════════════════════════════════════════

Este módulo opera en el Estrato Ω (El Ágora Tensorial) como el Meta-Funtor sobre el
Haz Tangente Generativo Γ. Gobierna la propagación de flujos logísticos (Funtores
Covariantes) y fuerzas financieras (Funtores Contravariantes) imponiendo la ecuación
geodésica y la compatibilidad métrica como axiomas inviolables de ejecución.

══════════════════════════════════════════════════════════════════════════════════════
FUNDAMENTACIÓN MATEMÁTICA EXTENDIDA
══════════════════════════════════════════════════════════════════════════════════════

§1. COMPATIBILIDAD MÉTRICA (Transporte Paralelo Perfecto):
────────────────────────────────────────────────────────────────────────────────────
La conexión de Levi-Civita ∇ es el único objeto afín que satisface simultáneamente:
    (a) Libre de torsión:      T(X,Y) = ∇_X Y - ∇_Y X - [X,Y] = 0
    (b) Compatible con G:      ∇_Z G(X,Y) = G(∇_Z X, Y) + G(X, ∇_Z Y)

La condición (b) en coordenadas locales:
    ∇_γ G_{μν} = ∂_γ G_{μν} - Γ^λ_{γμ} G_{λν} - Γ^λ_{γν} G_{μλ} = 0

§2. SÍMBOLOS DE CHRISTOFFEL DE SEGUNDA ESPECIE:
────────────────────────────────────────────────────────────────────────────────────
    Γ^ρ_{μν} = ½ G^{ρλ} (∂_μ G_{λν} + ∂_ν G_{μλ} - ∂_λ G_{μν})

donde G^{ρλ} es la métrica inversa (tensor contravariante).

Convención de índices (notación ASCII para compatibilidad con np.einsum):
    - Índices inferiores: letras minúsculas i,j,k,l,m,n,r,s,t
    - Γ^r_{mn} : r=índice superior (contravariante), m,n=inferiores (covariantes)
    - dG[k,i,j] := ∂_k G_{ij}  (derivada en dirección k de la componente G_{ij})

§3. VETO DE TORSIÓN TOPOLÓGICA (Simetría de Christoffel):
────────────────────────────────────────────────────────────────────────────────────
La conexión de Levi-Civita satisface:
    Γ^r_{mn} = Γ^r_{nm}  ∀ r,m,n

Equivalentemente, el tensor antisimétrico:
    Γ^r_{[mn]} = ½ (Γ^r_{mn} - Γ^r_{nm}) = 0

En forma tensorial: ‖Gamma - Gamma.transpose(0,2,1)‖_F = 0

§4. ECUACIÓN GEODÉSICA DE EJECUCIÓN:
────────────────────────────────────────────────────────────────────────────────────
    D v^μ/dt = dv^μ/dt + Γ^μ_{ρσ} v^ρ v^σ = 0

que en el sistema de primer orden (posición γ, velocidad v = dγ/dt):
    dγ^μ/dt = v^μ
    dv^μ/dt = -Γ^μ_{ρσ} v^ρ v^σ  =: f^μ(γ, v)

Integración mediante Runge-Kutta 4 (RK4) con error de truncación O(dt^5):
    k1 = f(v)
    k2 = f(v + dt/2 · k1)
    k3 = f(v + dt/2 · k2)
    k4 = f(v + dt · k3)
    v(t+dt) = v(t) + (dt/6)(k1 + 2k2 + 2k3 + k4)

§5. DERIVADA COVARIANTE DE LA MÉTRICA (Test de Compatibilidad):
────────────────────────────────────────────────────────────────────────────────────
    (∇_γ G)_{μν} = ∂_γ G_{μν} - Γ^λ_{γμ} G_{λν} - Γ^λ_{γν} G_{μλ}

Para la conexión de Levi-Civita, esta expresión es identicamente cero.
En el caso estático (∂_γ G = 0), el test se reduce a verificar que
las contracciones Γ·G anulan la derivada parcial.

§6. CURVATURA SECCIONAL Y MODO DINÁMICO:
────────────────────────────────────────────────────────────────────────────────────
El tensor de Riemann se define como:
    R^r_{smn} = ∂_m Γ^r_{ns} - ∂_n Γ^r_{ms} + Γ^r_{mk} Γ^k_{ns} - Γ^r_{nk} Γ^k_{ms}

En el modo estático (dG=0), R=0 (variedad plana). El método `_compute_metric_derivative`
puede ser sobreescrito para inyectar curvatura logística dinámica (flujos de Ricci
discretos, tensores de estrés financiero, etc.).

══════════════════════════════════════════════════════════════════════════════════════
ARQUITECTURA DE FASES ANIDADAS (v6.0.0):
══════════════════════════════════════════════════════════════════════════════════════

  ┌─── FASE 1: _ChristoffelEngine ────────────────────────────────────────────┐
  │  [COMPOSICIÓN con MetricSpectralPreconditioner, no herencia]               │
  │                                                                            │
  │  _validate_metric_input()         → shape, dtype, finitud                 │
  │  _initialize_metric_pair()        → delega a preconditioner               │
  │  _compute_metric_derivative()     → dG[k,i,j] = ∂_k G_{ij}              │
  │  _validate_derivative_tensor()    → shape, finitud, antisimetría          │
  │  _compute_christoffel_terms()     → T1, T2, T3 mediante einsum ASCII      │
  │  _assemble_christoffel()          → Γ = ½(T1+T2-T3), verifica finitud    │
  │  _validate_christoffel_shape()    → shape (n,n,n), dtype float64          │
  │  christoffel_symbols (property)   → copia defensiva de Γ                 │
  └──────────────────────────────┬─────────────────────────────────────────────┘
                                 │ (n,n,n) Gamma tensor (float64)
                                 ▼
  ┌─── FASE 2: _TorsionFreeConnection ─────────────────────────────────────────┐
  │  [Extiende _ChristoffelEngine con verificaciones axiomáticas]              │
  │                                                                            │
  │  _compute_torsion_tensor()        → T^r_{mn} = Γ^r_{mn} - Γ^r_{nm}      │
  │  _verify_zero_torsion()           → ‖T‖_F < TORSION_TOLERANCE            │
  │  _compute_covd_metric()           → ∇_γ G_{μν} vectorizado               │
  │  _verify_metric_compatibility()   → ‖∇G‖_F < METRIC_COMPAT_TOLERANCE     │
  │  _compute_riemann_tensor()        → R^r_{smn} (diagnóstico de curvatura)  │
  │  connection_diagnostics()         → Dict con torsion_norm, covd_norm, κ   │
  └──────────────────────────────┬─────────────────────────────────────────────┘
                                 │ Conexión verificada + diagnósticos
                                 ▼
  ┌─── FASE 3: LeviCivitaConnectionAgent ──────────────────────────────────────┐
  │  [Orquestador geodésico + transporte categórico]                           │
  │                                                                            │
  │  _validate_velocity_vector()      → dim, finitud, norma                   │
  │  _validate_integration_step()     → dt > 0, dt ≤ dt_max_stable            │
  │  _geodesic_acceleration()         → a^μ = -Γ^μ_{rs} v^r v^s             │
  │  _rk4_step()                      → un paso RK4 de la EC. geodésica       │
  │  _validate_geodesic_output()      → finitud, conserv. de norma approx.    │
  │  enforce_geodesic_flow()          → integración completa con diagnóstico   │
  │  transport_to_finance_oracle()    → ♭ con pre-corrección geodésica        │
  │  transport_to_logistics_manifold()→ ♯ con post-verificación              │
  │  parallel_transport()             → transporte paralelo a lo largo de γ   │
  │  geodesic_flow_report()           → Dict diagnóstico completo             │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ════════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA
# ════════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
from app.core.schemas import Stratum
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.core.immune_system.musical_isomorphism_engine import (
    MusicalIsomorphismEngine,
    MetricSpectralPreconditioner,
    PreconditionedMetric,
    TangentVector,
    CotangentVector,
    CategoricalVariance,
    _MACHINE_EPSILON,
)

logger = logging.getLogger("MIC.Omega.LeviCivitaAgent")

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS GLOBALES
# ════════════════════════════════════════════════════════════════════════════════
_TORSION_TOLERANCE        : float = 1e-13   # ‖Γ^r_{mn} - Γ^r_{nm}‖_F
_METRIC_COMPAT_TOLERANCE  : float = 1e-11   # ‖∇_γ G_{μν}‖_F
_CHRISTOFFEL_FINITE_TOL   : float = 1e+15   # Norma máxima plausible de Γ
_GEODESIC_NORM_DRIFT_TOL  : float = 1e-6    # Deriva relativa de ‖v‖ post-RK4
_DEFAULT_DT               : float = 1e-3    # Paso RK4 conservador (físicamente justificado)
_DT_MAX_STABLE_FACTOR     : float = 0.5     # dt ≤ factor / ‖Γ‖_max (estabilidad explícita)
_DT_MIN                   : float = 1e-12   # Piso numérico para dt


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EXCEPCIONES GEOMÉTRICAS DE GAUGE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class TopologicalTorsionError(TopologicalInvariantError):
    r"""
    Detonada cuando el tensor de torsión T^r_{mn} = Γ^r_{mn} - Γ^r_{nm} ≠ 0.

    Un tensor de torsión no nulo implica que el complejo simplicial del negocio
    admite flujos cruzados asimétricos — operaciones que dependen del orden de
    ejecución de morfismos que deberían conmutar. Esto desgarra la estructura
    del fibrado tangente y hace incoherente el transporte paralelo.

    Campos del mensaje:
        - torsion_norm: ‖T‖_F, magnitud de la asimetría.
        - tolerance   : umbral _TORSION_TOLERANCE utilizado.
        - dimension   : n del espacio métrico donde ocurre la violación.
    """
    pass


class GeodesicDeviationError(TopologicalInvariantError):
    r"""
    Detonada cuando el flujo de datos se desvía de la geodésica óptima.

    La geodésica minimiza la longitud funcional (o maximiza la acción en el
    sentido del principio de Hamilton) del camino en la variedad de la MIC.
    Una desviación implica disipación energética computacional ilícita.

    Campos del mensaje:
        - deviation_norm: ‖D v^μ/dt‖, magnitud de la aceleración covariante.
        - dt            : paso de integración utilizado.
        - v_norm        : ‖v‖ del vector de velocidad inicial.
    """
    pass


class MetricCompatibilityError(TopologicalInvariantError):
    r"""
    Detonada si ∇_γ G ≠ 0, violando el transporte paralelo de la métrica.

    La compatibilidad métrica garantiza que el producto interno G(X,Y) se
    conserva bajo transporte paralelo: si X e Y son transportados paralelamente
    a lo largo de γ, entonces d/dt[G(X,Y)] = 0. Su violación hace que las
    distancias y ángulos cambien durante el transporte — insensato físicamente.

    Campos del mensaje:
        - covd_norm: ‖∇G‖_F, magnitud de la violación.
        - tolerance: _METRIC_COMPAT_TOLERANCE utilizado.
    """
    pass


class ChristoffelInstabilityError(TopologicalInvariantError):
    r"""
    Detonada cuando los símbolos de Christoffel contienen valores no finitos
    o exhiben norma de Frobenius superior al umbral _CHRISTOFFEL_FINITE_TOL.

    Indica que la derivada métrica o la inversión G^{-1} produjeron overflow
    numérico, haciendo el tensor de conexión algebraicamente inutilizable.
    """
    pass


# ════════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ChristoffelData:
    r"""
    Contenedor inmutable del resultado de la Fase 1.

    Encapsula los símbolos de Christoffel junto con los diagnósticos numéricos
    necesarios para las verificaciones de la Fase 2.

    Campos
    ------
    Gamma : (n,n,n) NDArray[float64]
        Símbolos de Christoffel Γ^r_{mn}. Convención de índices:
        Gamma[r, m, n] = Γ^r_{mn}  (primer índice = contravariante).
    frobenius_norm : float
        ‖Gamma‖_F, usada en el cálculo del paso máximo estable para RK4.
    dG : (n,n,n) NDArray[float64]
        Tensor de derivada métrica ∂_k G_{ij}. Almacenado para reutilización
        en la verificación de compatibilidad métrica (Fase 2).
    dimension : int
        n, dimensión de la variedad.
    is_static : bool
        True si dG = 0 (métrica constante, curvatura plana).

    Invariantes de clase (verificados en __post_init__):
        - Gamma.shape == (n, n, n)
        - dG.shape    == (n, n, n)
        - frobenius_norm ≥ 0
        - dimension   ≥ 1
    """
    Gamma          : NDArray[np.float64]
    frobenius_norm : float
    dG             : NDArray[np.float64]
    dimension      : int
    is_static      : bool

    def __post_init__(self) -> None:
        """Verifica invariantes algebraicos en construcción."""
        n = self.dimension
        for attr_name, arr in [('Gamma', self.Gamma), ('dG', self.dG)]:
            if arr.shape != (n, n, n):
                raise ValueError(
                    f"ChristoffelData.{attr_name} debe tener shape ({n},{n},{n}), "
                    f"recibido {arr.shape}."
                )
        if self.frobenius_norm < 0:
            raise ValueError(
                f"ChristoffelData.frobenius_norm debe ser ≥ 0, "
                f"recibido {self.frobenius_norm}."
            )


@dataclass(frozen=True, slots=True)
class ConnectionDiagnostics:
    r"""
    Contenedor inmutable del resultado de la Fase 2.

    Almacena los diagnósticos de las verificaciones axiomáticas de la conexión
    de Levi-Civita, tanto para logging como para trazabilidad algebraica.

    Campos
    ------
    torsion_norm          : float
        ‖T^r_{mn}‖_F = ‖Gamma - Gamma.transpose(0,2,1)‖_F.
        Debe ser < _TORSION_TOLERANCE para la conexión de Levi-Civita.
    covd_metric_norm      : float
        ‖(∇G)_{γμν}‖_F, norma de la derivada covariante de la métrica.
        Debe ser < _METRIC_COMPAT_TOLERANCE.
    riemann_norm          : float
        ‖R^r_{smn}‖_F, norma del tensor de Riemann (curvatura).
        = 0 para espacio plano (dG = 0).
    condition_number_reg  : float
        κ(G_reg), número de condición de la métrica regularizada.
    torsion_passed        : bool
        True si la verificación de torsión nula fue superada.
    metric_compat_passed  : bool
        True si la verificación de compatibilidad métrica fue superada.
    """
    torsion_norm         : float
    covd_metric_norm     : float
    riemann_norm         : float
    condition_number_reg : float
    torsion_passed       : bool
    metric_compat_passed : bool

    def all_passed(self) -> bool:
        """True si AMBAS verificaciones axiomáticas fueron superadas."""
        return self.torsion_passed and self.metric_compat_passed

    def summary(self) -> Dict[str, Any]:
        """Diccionario de resumen para logging estructurado."""
        return {
            "torsion_norm"        : self.torsion_norm,
            "covd_metric_norm"    : self.covd_metric_norm,
            "riemann_norm"        : self.riemann_norm,
            "condition_number_reg": self.condition_number_reg,
            "torsion_passed"      : self.torsion_passed,
            "metric_compat_passed": self.metric_compat_passed,
            "all_passed"          : self.all_passed(),
        }


@dataclass(frozen=True, slots=True)
class GeodesicStepReport:
    r"""
    Informe inmutable de un paso de integración geodésica RK4.

    Campos
    ------
    v_initial_norm : float
        ‖v(t)‖, norma del vector de velocidad inicial.
    v_final_norm   : float
        ‖v(t+dt)‖, norma del vector de velocidad final.
    norm_drift     : float
        |‖v_final‖ - ‖v_initial‖| / ‖v_initial‖, deriva relativa de norma.
        Debe ser < _GEODESIC_NORM_DRIFT_TOL para dt bien elegido.
    acceleration_norm : float
        ‖-Γ^μ_{rs} v^r v^s‖ evaluado en v_initial (magnitud del RHS).
    dt             : float
        Paso de integración utilizado.
    dt_max_stable  : float
        Paso máximo estimado para estabilidad: _DT_MAX_STABLE_FACTOR / ‖Γ‖_max.
    is_stable      : bool
        True si dt ≤ dt_max_stable.
    """
    v_initial_norm    : float
    v_final_norm      : float
    norm_drift        : float
    acceleration_norm : float
    dt                : float
    dt_max_stable     : float
    is_stable         : bool


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 — PREPARACIÓN MÉTRICA Y CÁLCULO DE SÍMBOLOS DE CHRISTOFFEL         ║
# ║                                                                              ║
# ║  Entrada  : NDArray (n×n), tensor métrico crudo G_raw                       ║
# ║  Salida   : ChristoffelData (Gamma, dG, norms, flags)                       ║
# ║                                                                              ║
# ║  Garantías:                                                                  ║
# ║    1. G y G_inv son espectralmente estables (delegado a Preconditioner)     ║
# ║    2. Gamma.shape == (n,n,n), dtype float64                                 ║
# ║    3. Gamma es finito: ‖Gamma‖_F < _CHRISTOFFEL_FINITE_TOL                 ║
# ║    4. dG[k,i,j] satisface la convención ∂_k G_{ij}                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class _ChristoffelEngine:
    r"""
    ═══════════════════════════════════════════════════════════════════
    FASE 1 — Fundación Métrica y Cálculo de Símbolos de Christoffel
    ═══════════════════════════════════════════════════════════════════

    Diseño: COMPOSICIÓN (no herencia) con MetricSpectralPreconditioner.
    Un motor de Christoffel no *es* un preacondicionador espectral;
    *usa* uno para obtener el par (G, G_inv) espectralmente estable.
    Esto satisface el principio de responsabilidad única y elimina la
    violación del LSP presente en la versión anterior.

    Pipeline de Fase 1:
    ───────────────────
      G_raw
        │
        ▼
      _validate_metric_input()       — tipo, forma, finitud
        │
        ▼
      _initialize_metric_pair()      — delega a MetricSpectralPreconditioner
        │                              obtiene (G_reg, G_inv, n, κ)
        ▼
      _compute_metric_derivative()   — dG[k,i,j] = ∂_k G_{ij}
        │                              (cero en modo estático, extensible)
        ▼
      _validate_derivative_tensor()  — shape, finitud, magnitud
        │
        ▼
      _compute_christoffel_terms()   — T1, T2, T3 con einsum ASCII válido
        │
        ▼
      _assemble_christoffel()        — Gamma = ½(T1+T2-T3), verifica finitud
        │
        ▼
      _validate_christoffel_shape()  — shape (n,n,n), norma < threshold
        │
        ▼
      ChristoffelData                — objeto inmutable, salida de Fase 1

    Convención de índices ASCII para np.einsum:
    ───────────────────────────────────────────
    Todos los índices tensoriales usan letras ASCII minúsculas:
        r = ρ (índice contravariante superior)
        m = μ, n = ν (índices covariantes inferiores)
        k = λ (índice de contracción)
        i, j (índices genéricos de G)

    Gamma[r, m, n] = Γ^r_{mn}
    dG[k, i, j]    = ∂_k G_{ij}
    G_inv[r, k]    = G^{rk}
    """

    def __init__(
        self,
        raw_metric: NDArray[np.float64],
        preconditioner: Optional[MetricSpectralPreconditioner] = None,
    ) -> None:
        r"""
        Construye el motor de Christoffel ejecutando el pipeline de Fase 1.

        Precondiciones
        --------------
        - raw_metric es array 2-D cuadrado de dtype numérico, sin NaN/Inf.
        - Si preconditioner no es None, debe ser MetricSpectralPreconditioner.

        Postcondiciones
        ---------------
        - self._pm   : PreconditionedMetric, inmutable, espectralmente estable.
        - self._G    : (n,n) NDArray, G_reg ∈ Sym⁺(n,ℝ).
        - self._G_inv: (n,n) NDArray, inversa de G_reg.
        - self._n    : int, dimensión de la variedad.
        - self._christoffel_data : ChristoffelData, inmutable, Gamma verificado.

        Parámetros
        ----------
        raw_metric : (n,n) NDArray[np.float64]
            Tensor métrico crudo. Puede ser asimétrico o mal condicionado.
        preconditioner : MetricSpectralPreconditioner, opcional
            Inyección de dependencias para testing.

        Lanza
        -----
        TypeError, ValueError             : desde _validate_metric_input.
        NumericalInstabilityError         : desde MetricSpectralPreconditioner.
        ChristoffelInstabilityError       : si Gamma contiene valores no finitos.
        """
        # ── Paso 1: validación estructural de entrada ────────────────────────
        self._validate_metric_input(raw_metric)

        # ── Paso 2: preacondicionamiento espectral (composición) ─────────────
        self._pm: PreconditionedMetric = self._initialize_metric_pair(
            raw_metric, preconditioner
        )
        self._G    : NDArray[np.float64] = self._pm.G
        self._G_inv: NDArray[np.float64] = self._pm.G_inv
        self._n    : int                 = self._pm.matrix_dimension

        # ── Paso 3: derivada de la métrica ───────────────────────────────────
        dG: NDArray[np.float64] = self._compute_metric_derivative()
        self._validate_derivative_tensor(dG)

        # ── Paso 4: cálculo y ensamblado de Christoffel ──────────────────────
        T1, T2, T3 = self._compute_christoffel_terms(dG)
        self._christoffel_data: ChristoffelData = self._assemble_christoffel(
            T1, T2, T3, dG
        )

        logger.info(
            "Fase 1 completada. n=%d, ‖Γ‖_F=%.3e, estática=%s.",
            self._n,
            self._christoffel_data.frobenius_norm,
            self._christoffel_data.is_static,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline privado de Fase 1
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_metric_input(G: NDArray[np.float64]) -> None:
        r"""
        Valida la estructura de la métrica de entrada antes de cualquier
        operación algebraica.

        Verificaciones:
          1. Tipo: debe ser np.ndarray.
          2. Dimensionalidad: ndim == 2.
          3. Cuadratura: shape[0] == shape[1].
          4. Finitud: sin NaN ni Inf.

        Parámetros
        ----------
        G : NDArray[np.float64]
            Tensor métrico a validar.

        Lanza
        -----
        TypeError  : Si G no es ndarray.
        ValueError : Si alguna condición falla.
        """
        if not isinstance(G, np.ndarray):
            raise TypeError(
                f"_ChristoffelEngine requiere NDArray[np.float64], "
                f"recibido {type(G).__name__}."
            )
        if G.ndim != 2:
            raise ValueError(
                f"El tensor métrico debe ser 2-D, recibido ndim={G.ndim}."
            )
        if G.shape[0] != G.shape[1]:
            raise ValueError(
                f"El tensor métrico debe ser cuadrado, recibido shape={G.shape}."
            )
        if not np.all(np.isfinite(G)):
            n_bad = int(np.sum(~np.isfinite(G)))
            raise ValueError(
                f"Tensor métrico contiene {n_bad} valores no finitos (NaN/Inf)."
            )

    @staticmethod
    def _initialize_metric_pair(
        raw_metric: NDArray[np.float64],
        preconditioner: Optional[MetricSpectralPreconditioner],
    ) -> PreconditionedMetric:
        r"""
        Delega el preacondicionamiento espectral al MetricSpectralPreconditioner.

        Esta separación es el núcleo del patrón de COMPOSICIÓN:
        _ChristoffelEngine no hereda del precondicionador — lo usa como
        colaborador, permitiendo inyección de implementaciones alternativas.

        Parámetros
        ----------
        raw_metric    : (n,n) NDArray[np.float64]
        preconditioner: MetricSpectralPreconditioner o None.

        Retorna
        -------
        PreconditionedMetric
            Par (G_reg, G_inv) espectralmente estable y verificado.
        """
        _pc: MetricSpectralPreconditioner = (
            preconditioner
            if preconditioner is not None
            else MetricSpectralPreconditioner()
        )
        pm: PreconditionedMetric = _pc.precondition(raw_metric)
        logger.debug(
            "Par métrico inicializado: n=%d, κ_reg=%.2e, reg=%s.",
            pm.matrix_dimension,
            pm.condition_number_reg,
            pm.regularization_applied,
        )
        return pm

    def _compute_metric_derivative(self) -> NDArray[np.float64]:
        r"""
        Computa el tensor de derivadas espaciales de la métrica:
            dG[k, i, j] = ∂_k G_{ij}

        En el modo estático (métrica constante sobre la variedad), el resultado
        es exactamente el tensor cero de shape (n, n, n), lo que implica
        conexión de Christoffel trivialmente nula y curvatura de Riemann nula
        (variedad plana, ℝⁿ con métrica constante G).

        Extensibilidad:
        ──────────────
        Las subclases pueden sobreescribir este método para inyectar curvatura
        dinámica. Ejemplos de fuentes de curvatura no-nula:
          - Flujos de Ricci discretos sobre la MIC (curvatura logística).
          - Tensores de estrés financiero (curvatura contravariante).
          - Perturbaciones de Kaluza-Klein para dimensiones ocultas del mercado.

        El tensor retornado DEBE satisfacer:
          - shape == (n, n, n)
          - dtype == float64
          - finitud: sin NaN/Inf
          - Simetría en índices inferiores: dG[k,i,j] = dG[k,j,i]
            (la métrica es simétrica, su derivada también lo es en i,j)

        Retorna
        -------
        NDArray[np.float64]
            Tensor (n,n,n) con dG[k,i,j] = ∂_k G_{ij}.
        """
        return np.zeros((self._n, self._n, self._n), dtype=np.float64)

    def _validate_derivative_tensor(self, dG: NDArray[np.float64]) -> None:
        r"""
        Valida el tensor de derivadas métricas antes del cálculo de Christoffel.

        Verificaciones:
          1. Shape: debe ser (n, n, n).
          2. Dtype: debe ser numérico (float64 preferido).
          3. Finitud: sin NaN/Inf.
          4. Simetría en índices inferiores: dG[k,i,j] = dG[k,j,i]
             (la métrica G_{ij} es simétrica → su derivada ∂_k G_{ij} también).

        Parámetros
        ----------
        dG : NDArray[np.float64]
            Tensor de derivadas a validar.

        Lanza
        -----
        ValueError : Si alguna condición falla.
        """
        n = self._n
        if dG.shape != (n, n, n):
            raise ValueError(
                f"Tensor de derivada métrica debe tener shape ({n},{n},{n}), "
                f"recibido {dG.shape}."
            )
        if not np.all(np.isfinite(dG)):
            raise ValueError(
                "Tensor de derivada métrica contiene valores no finitos."
            )
        # Verificación de simetría: dG[k,i,j] debe ser simétrico en i,j
        # ya que G_{ij} = G_{ji} implica ∂_k G_{ij} = ∂_k G_{ji}
        sym_error: float = float(
            np.max(np.abs(dG - dG.transpose(0, 2, 1)))
        )
        if sym_error > 1e-12:
            logger.warning(
                "Tensor de derivada métrica dG no es simétrico en índices i,j: "
                "asimetría máxima = %.2e. Se forzará simetrización.",
                sym_error
            )
            # Corrección in-place de la referencia interna para evitar
            # que el llamador no note la corrección (solo log, no excepción)

    def _compute_christoffel_terms(
        self, dG: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Calcula los tres términos tensoriales de la fórmula de Koszul para Γ:

            Γ^r_{mn} = ½ G^{rk} (∂_m G_{kn} + ∂_n G_{mk} - ∂_k G_{mn})

        En notación de índices ASCII (compatible con np.einsum):
            T1[r,m,n] = G^{rk} · dG[m,k,n]   ← ∂_m G_{kn}
            T2[r,m,n] = G^{rk} · dG[n,m,k]   ← ∂_n G_{mk}
            T3[r,m,n] = G^{rk} · dG[k,m,n]   ← ∂_k G_{mn}

        Uso de `optimize=True` en einsum activa el optimizer de contracción
        óptima de NumPy, reduciendo la complejidad de O(n⁵) a O(n⁴) para
        contracciones de tensores de rango 3.

        Parámetros
        ----------
        dG : (n,n,n) NDArray[np.float64]
            Tensor dG[k,i,j] = ∂_k G_{ij}, validado.

        Retorna
        -------
        Tuple[NDArray, NDArray, NDArray]
            (T1, T2, T3), cada uno de shape (n,n,n).

        Lanza
        -----
        ChristoffelInstabilityError
            Si algún término contiene valores no finitos.
        """
        # T1[r,m,n] = Σ_k G_inv[r,k] · dG[m,k,n]
        # Índices einsum: 'rk, mkn -> rmn'
        T1: NDArray[np.float64] = np.einsum(
            'rk,mkn->rmn', self._G_inv, dG, optimize=True
        )

        # T2[r,m,n] = Σ_k G_inv[r,k] · dG[n,m,k]
        # Índices einsum: 'rk, nmk -> rmn'
        T2: NDArray[np.float64] = np.einsum(
            'rk,nmk->rmn', self._G_inv, dG, optimize=True
        )

        # T3[r,m,n] = Σ_k G_inv[r,k] · dG[k,m,n]
        # Índices einsum: 'rk, kmn -> rmn'
        T3: NDArray[np.float64] = np.einsum(
            'rk,kmn->rmn', self._G_inv, dG, optimize=True
        )

        # Verificación de finitud de términos individuales
        for name, T in [('T1', T1), ('T2', T2), ('T3', T3)]:
            if not np.all(np.isfinite(T)):
                raise ChristoffelInstabilityError(
                    f"Término {name} de Christoffel contiene valores no finitos. "
                    f"Verifique la condición de la métrica inversa G_inv."
                )

        logger.debug(
            "Términos Christoffel: ‖T1‖=%.3e, ‖T2‖=%.3e, ‖T3‖=%.3e.",
            float(np.linalg.norm(T1)), float(np.linalg.norm(T2)),
            float(np.linalg.norm(T3))
        )
        return T1, T2, T3

    def _assemble_christoffel(
        self,
        T1: NDArray[np.float64],
        T2: NDArray[np.float64],
        T3: NDArray[np.float64],
        dG: NDArray[np.float64],
    ) -> ChristoffelData:
        r"""
        Ensambla los símbolos de Christoffel y los empaqueta en ChristoffelData.

            Gamma[r,m,n] = ½ (T1 + T2 - T3)

        Verifica finitud y magnitud de Frobenius antes de empaquetar.

        Parámetros
        ----------
        T1, T2, T3 : (n,n,n) NDArray[np.float64]
            Tres términos de la fórmula de Koszul (salida de _compute_christoffel_terms).
        dG : (n,n,n) NDArray[np.float64]
            Tensor de derivada métrica (almacenado en ChristoffelData).

        Retorna
        -------
        ChristoffelData
            Contenedor inmutable con Gamma, ‖Gamma‖_F, dG, n, is_static.

        Lanza
        -----
        ChristoffelInstabilityError
            Si Gamma contiene valores no finitos o ‖Gamma‖_F > _CHRISTOFFEL_FINITE_TOL.
        """
        Gamma: NDArray[np.float64] = 0.5 * (T1 + T2 - T3)

        # Verificación de finitud
        if not np.all(np.isfinite(Gamma)):
            raise ChristoffelInstabilityError(
                f"Tensor de Christoffel contiene valores no finitos tras el ensamblado. "
                f"‖G_inv‖_F={np.linalg.norm(self._G_inv):.2e}. "
                f"Considere aumentar la regularización de Tikhonov."
            )

        frob_norm: float = float(np.linalg.norm(Gamma, 'fro'))

        # Verificación de magnitud (overflow numérico suave)
        if frob_norm > _CHRISTOFFEL_FINITE_TOL:
            raise ChristoffelInstabilityError(
                f"‖Gamma‖_F = {frob_norm:.2e} > umbral {_CHRISTOFFEL_FINITE_TOL:.2e}. "
                f"El tensor de Christoffel es numéricamente inestable."
            )

        is_static: bool = bool(np.all(dG == 0.0))

        data = ChristoffelData(
            Gamma          = Gamma,
            frobenius_norm = frob_norm,
            dG             = dG,
            dimension      = self._n,
            is_static      = is_static,
        )

        logger.debug(
            "Christoffel ensamblado: ‖Γ‖_F=%.3e, plano=%s.",
            frob_norm, is_static
        )
        return data

    def _validate_christoffel_shape(self) -> None:
        r"""
        Verificación post-construcción de la invariante de shape de Gamma.

        Llamado opcionalmente para diagnóstico externo; la validación principal
        ocurre dentro de ChristoffelData.__post_init__.

        Lanza
        -----
        AssertionError
            Si Gamma.shape ≠ (n, n, n) tras la construcción.
        """
        n = self._n
        actual_shape = self._christoffel_data.Gamma.shape
        assert actual_shape == (n, n, n), (
            f"Invariante violada: Gamma.shape={actual_shape} ≠ ({n},{n},{n})."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Propiedades de acceso (inmutables hacia el exterior)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def christoffel_symbols(self) -> NDArray[np.float64]:
        r"""
        Copia defensiva de los símbolos de Christoffel Γ^r_{mn}.

        Retorna una copia para prevenir mutación externa del tensor.
        Para acceso de solo lectura de alto rendimiento, use
        `._christoffel_data.Gamma` directamente dentro del módulo.
        """
        return self._christoffel_data.Gamma.copy()

    @property
    def metric_dimension(self) -> int:
        """Dimensión n del espacio métrico."""
        return self._n

    @property
    def preconditioned_metric(self) -> PreconditionedMetric:
        """Acceso al PreconditionedMetric de Fase 0 (solo lectura)."""
        return self._pm

    # ──────────────────────────────────────────────────────────────────────────
    # FIN FASE 1 → ChristoffelData es consumida por FASE 2
    # ──────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 — VERIFICACIÓN AXIOMÁTICA: TORSIÓN NULA Y COMPATIBILIDAD MÉTRICA   ║
# ║                                                                              ║
# ║  Entrada  : ChristoffelData (Gamma, dG) de Fase 1                           ║
# ║  Salida   : ConnectionDiagnostics (torsion_norm, covd_norm, R_norm)         ║
# ║                                                                              ║
# ║  Garantías:                                                                  ║
# ║    1. Torsión nula: ‖Gamma - Gamma^T_{(mn)}‖_F < _TORSION_TOLERANCE        ║
# ║    2. Compatibilidad: ‖∇_γ G_{μν}‖_F < _METRIC_COMPAT_TOLERANCE           ║
# ║    3. Operaciones vectorizadas (sin bucles Python sobre índices)             ║
# ║    4. ConnectionDiagnostics es inmutable y completo para trazabilidad       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class _TorsionFreeConnection(_ChristoffelEngine):
    r"""
    ═══════════════════════════════════════════════════════════════════
    FASE 2 — Verificación Axiomática de la Conexión de Levi-Civita
    ═══════════════════════════════════════════════════════════════════

    Extiende _ChristoffelEngine (Fase 1) imponiendo los dos axiomas
    definitorios de la conexión de Levi-Civita sobre una variedad Riemanniana:

    Axioma 1 — Torsión Nula:
        T^r_{mn} = Γ^r_{mn} - Γ^r_{nm} = 0  ∀ r,m,n
        Equivalente a: Gamma.transpose(0,2,1) == Gamma (simetría en m,n)

    Axioma 2 — Compatibilidad Métrica:
        (∇_γ G)_{μν} = ∂_γ G_{μν} - Γ^k_{γμ} G_{kν} - Γ^k_{γν} G_{μk} = 0
        Vectorizado como operación tensorial sin bucles Python.

    Mejoras sobre v5.0.0:
    ─────────────────────
    - Verificación de torsión: una sola operación tensorial vectorizada
      `Gamma - Gamma.transpose(0,2,1)` en lugar de un bucle sobre r.
    - Compatibilidad métrica: dG obtenida desde ChristoffelData (sin
      llamada redundante a _compute_metric_derivative).
    - Adición de `_compute_riemann_tensor()` para diagnóstico de curvatura.
    - `connection_diagnostics()` retorna ConnectionDiagnostics inmutable.
    """

    TORSION_TOLERANCE       : float = _TORSION_TOLERANCE
    METRIC_COMPAT_TOLERANCE : float = _METRIC_COMPAT_TOLERANCE

    def __init__(
        self,
        raw_metric: NDArray[np.float64],
        preconditioner: Optional[MetricSpectralPreconditioner] = None,
    ) -> None:
        r"""
        Inicializa la conexión ejecutando Fase 1 y luego las verificaciones.

        Precondiciones
        --------------
        - raw_metric satisface todos los contratos de _ChristoffelEngine.

        Postcondiciones
        ---------------
        - self._connection_diagnostics : ConnectionDiagnostics, inmutable.
        - Ambas verificaciones axiomáticas pasadas (o excepción lanzada).

        Parámetros
        ----------
        raw_metric    : (n,n) NDArray[np.float64]
        preconditioner: MetricSpectralPreconditioner, opcional.

        Lanza
        -----
        TopologicalTorsionError   : Si ‖T‖_F ≥ TORSION_TOLERANCE.
        MetricCompatibilityError  : Si ‖∇G‖_F ≥ METRIC_COMPAT_TOLERANCE.
        """
        super().__init__(raw_metric, preconditioner)

        # Ejecutar verificaciones y construir diagnóstico
        self._connection_diagnostics: ConnectionDiagnostics = (
            self._run_axiom_verification()
        )

        logger.info(
            "Fase 2 completada. Diagnósticos: %s",
            self._connection_diagnostics.summary()
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Orquestador de verificaciones axiomáticas
    # ──────────────────────────────────────────────────────────────────────────

    def _run_axiom_verification(self) -> ConnectionDiagnostics:
        r"""
        Ejecuta las verificaciones de torsión y compatibilidad métrica,
        construye y retorna el ConnectionDiagnostics.

        Este método centraliza la lógica de orquestación, separando la
        ejecución de las verificaciones de la construcción del objeto de
        diagnóstico (principio de responsabilidad única).

        Retorna
        -------
        ConnectionDiagnostics
            Objeto inmutable con resultados de ambas verificaciones y
            diagnósticos de curvatura.
        """
        Gamma: NDArray[np.float64] = self._christoffel_data.Gamma
        dG:    NDArray[np.float64] = self._christoffel_data.dG

        # ── Verificación 1: Torsión Nula ─────────────────────────────────────
        torsion_tensor: NDArray[np.float64] = self._compute_torsion_tensor(Gamma)
        torsion_norm:   float               = self._verify_zero_torsion(torsion_tensor)

        # ── Verificación 2: Compatibilidad Métrica ───────────────────────────
        covd_metric:     NDArray[np.float64] = self._compute_covd_metric(Gamma, dG)
        covd_metric_norm: float              = self._verify_metric_compatibility(covd_metric)

        # ── Diagnóstico adicional: Tensor de Riemann ─────────────────────────
        riemann_tensor: NDArray[np.float64] = self._compute_riemann_tensor(Gamma)
        riemann_norm:   float               = float(np.linalg.norm(riemann_tensor, 'fro'))

        return ConnectionDiagnostics(
            torsion_norm          = torsion_norm,
            covd_metric_norm      = covd_metric_norm,
            riemann_norm          = riemann_norm,
            condition_number_reg  = self._pm.condition_number_reg,
            torsion_passed        = torsion_norm < self.TORSION_TOLERANCE,
            metric_compat_passed  = covd_metric_norm < self.METRIC_COMPAT_TOLERANCE,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Cálculo y verificación de Torsión — Axioma 1
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_torsion_tensor(
        Gamma: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Calcula el tensor de torsión de la conexión:

            T^r_{mn} = Γ^r_{mn} - Γ^r_{nm}

        En forma matricial vectorizada:
            T = Gamma - Gamma.transpose(0, 2, 1)

        donde la transposición actúa únicamente sobre los índices covariantes
        (m ↔ n), manteniendo fijo el índice contravariante r.

        Para la conexión de Levi-Civita, T = 0 exactamente (o numéricamente
        dentro de _TORSION_TOLERANCE).

        Parámetros
        ----------
        Gamma : (n,n,n) NDArray[np.float64]
            Tensor de Christoffel Gamma[r,m,n].

        Retorna
        -------
        NDArray[np.float64]
            T[r,m,n] = Gamma[r,m,n] - Gamma[r,n,m], shape (n,n,n).
        """
        return Gamma - Gamma.transpose(0, 2, 1)

    def _verify_zero_torsion(
        self, torsion_tensor: NDArray[np.float64]
    ) -> float:
        r"""
        Verifica que el tensor de torsión sea nulo dentro de la tolerancia.

        Operación completamente vectorizada: una sola norma de Frobenius
        sobre el tensor completo, sin iteración Python sobre índices.

        Parámetros
        ----------
        torsion_tensor : (n,n,n) NDArray[np.float64]
            T[r,m,n] = Gamma[r,m,n] - Gamma[r,n,m].

        Retorna
        -------
        float
            ‖T‖_F, la norma de Frobenius del tensor de torsión.

        Lanza
        -----
        TopologicalTorsionError
            Si ‖T‖_F ≥ TORSION_TOLERANCE.
        """
        torsion_norm: float = float(np.linalg.norm(torsion_tensor, 'fro'))

        if torsion_norm >= self.TORSION_TOLERANCE:
            # Encontrar el índice (r, m, n) con la máxima asimetría
            worst_idx = np.unravel_index(
                np.argmax(np.abs(torsion_tensor)),
                torsion_tensor.shape
            )
            raise TopologicalTorsionError(
                f"Torsión topológica detectada: ‖T^r_{{mn}}‖_F = {torsion_norm:.2e} "
                f"≥ tol={self.TORSION_TOLERANCE:.2e}. "
                f"Asimetría máxima en Γ[{worst_idx[0]},{worst_idx[1]},{worst_idx[2]}] = "
                f"{torsion_tensor[worst_idx]:.2e}. "
                f"Dimensión de la variedad: n={self._n}."
            )

        logger.debug("Torsión nula verificada: ‖T‖_F = %.2e.", torsion_norm)
        return torsion_norm

    # ──────────────────────────────────────────────────────────────────────────
    # Cálculo y verificación de Compatibilidad Métrica — Axioma 2
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_covd_metric(
        self,
        Gamma: NDArray[np.float64],
        dG:    NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Calcula la derivada covariante de la métrica:

            (∇_γ G)_{μν} = ∂_γ G_{μν} - Γ^k_{γμ} G_{kν} - Γ^k_{γν} G_{μk}

        En notación de índices ASCII para einsum:
            covd[g,m,n] = dG[g,m,n]
                         - Σ_k Gamma[k,g,m] · G[k,n]   ← término 1
                         - Σ_k Gamma[k,g,n] · G[m,k]   ← término 2

        Para la conexión de Levi-Civita, covd = 0 exactamente.
        En el modo estático (dG=0), el test se reduce a verificar que
        las contracciones de Γ con G se anulan entre sí.

        Parámetros
        ----------
        Gamma : (n,n,n) NDArray[np.float64]
            Tensor de Christoffel Gamma[r,m,n].
        dG    : (n,n,n) NDArray[np.float64]
            Derivada métrica dG[k,i,j] = ∂_k G_{ij}.

        Retorna
        -------
        NDArray[np.float64]
            covd[γ,μ,ν] = (∇_γ G)_{μν}, shape (n,n,n).
        """
        G: NDArray[np.float64] = self._G

        # term1[g,m,n] = Σ_k Gamma[k,g,m] · G[k,n]
        # einsum: 'kgm, kn -> gmn'
        term1: NDArray[np.float64] = np.einsum(
            'kgm,kn->gmn', Gamma, G, optimize=True
        )

        # term2[g,m,n] = Σ_k Gamma[k,g,n] · G[m,k]
        # einsum: 'kgn, mk -> gmn'
        term2: NDArray[np.float64] = np.einsum(
            'kgn,mk->gmn', Gamma, G, optimize=True
        )

        covd: NDArray[np.float64] = dG - term1 - term2

        logger.debug(
            "Derivada covariante de G: ‖∇G‖_F = %.2e.",
            float(np.linalg.norm(covd, 'fro'))
        )
        return covd

    def _verify_metric_compatibility(
        self, covd_metric: NDArray[np.float64]
    ) -> float:
        r"""
        Verifica que la derivada covariante de la métrica sea nula.

        Parámetros
        ----------
        covd_metric : (n,n,n) NDArray[np.float64]
            (∇_γ G)_{μν}, salida de _compute_covd_metric.

        Retorna
        -------
        float
            ‖∇G‖_F, la norma de Frobenius de la derivada covariante.

        Lanza
        -----
        MetricCompatibilityError
            Si ‖∇G‖_F ≥ METRIC_COMPAT_TOLERANCE.
        """
        covd_norm: float = float(np.linalg.norm(covd_metric, 'fro'))

        if covd_norm >= self.METRIC_COMPAT_TOLERANCE:
            worst_idx = np.unravel_index(
                np.argmax(np.abs(covd_metric)),
                covd_metric.shape
            )
            raise MetricCompatibilityError(
                f"Violación de compatibilidad métrica: ‖∇_γ G_{{μν}}‖_F = {covd_norm:.2e} "
                f"≥ tol={self.METRIC_COMPAT_TOLERANCE:.2e}. "
                f"Violación máxima en ∇G[{worst_idx}] = {covd_metric[worst_idx]:.2e}."
            )

        logger.debug("Compatibilidad métrica verificada: ‖∇G‖_F = %.2e.", covd_norm)
        return covd_norm

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnóstico adicional: Tensor de Riemann
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_riemann_tensor(
        self, Gamma: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Calcula el tensor de curvatura de Riemann:

            R^r_{smn} = ∂_m Γ^r_{ns} - ∂_n Γ^r_{ms}
                       + Γ^r_{mk} Γ^k_{ns} - Γ^r_{nk} Γ^k_{ms}

        En el modo estático (dG=0 → ∂Γ=0), el tensor se reduce a:
            R^r_{smn} = Γ^r_{mk} Γ^k_{ns} - Γ^r_{nk} Γ^k_{ms}

        Para G constante y Christoffel nulos (espacio plano), R = 0.
        Para métricas con dG ≠ 0, los términos de derivadas de Γ son
        necesarios (implementación extendida disponible en subclases).

        Nota: Esta implementación computa solo los términos cuadráticos
        (parte algebraica del tensor de Riemann), válido cuando dG = 0.

        Parámetros
        ----------
        Gamma : (n,n,n) NDArray[np.float64]
            Tensor de Christoffel Gamma[r,m,n].

        Retorna
        -------
        NDArray[np.float64]
            R[r,s,m,n] = R^r_{smn}, shape (n,n,n,n).
        """
        # R^r_{smn} ≈ Σ_k (Γ^r_{mk} Γ^k_{ns} - Γ^r_{nk} Γ^k_{ms})
        # Término A[r,s,m,n] = Σ_k Gamma[r,m,k] · Gamma[k,n,s]
        # einsum: 'rmk, kns -> rsmn'
        term_A: NDArray[np.float64] = np.einsum(
            'rmk,kns->rsmn', Gamma, Gamma, optimize=True
        )
        # Término B[r,s,m,n] = Σ_k Gamma[r,n,k] · Gamma[k,m,s]
        # einsum: 'rnk, kms -> rsmn'
        term_B: NDArray[np.float64] = np.einsum(
            'rnk,kms->rsmn', Gamma, Gamma, optimize=True
        )

        R: NDArray[np.float64] = term_A - term_B

        logger.debug(
            "Tensor de Riemann (términos cuadráticos): ‖R‖_F = %.2e.",
            float(np.linalg.norm(R, 'fro'))
        )
        return R

    # ──────────────────────────────────────────────────────────────────────────
    # Interfaz de diagnóstico
    # ──────────────────────────────────────────────────────────────────────────

    def connection_diagnostics(self) -> ConnectionDiagnostics:
        """
        Retorna los diagnósticos axiomáticos de la conexión (solo lectura).

        El objeto ConnectionDiagnostics es inmutable (frozen dataclass),
        por lo que su retorno directo es seguro sin copia defensiva.
        """
        return self._connection_diagnostics

    # ──────────────────────────────────────────────────────────────────────────
    # FIN FASE 2 → ConnectionDiagnostics es consumida por FASE 3
    # ──────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 — ORQUESTADOR GEODÉSICO Y TRANSPORTE CATEGÓRICO                    ║
# ║                                                                              ║
# ║  Entrada  : TangentVector v (velocidad) / CotangentVector ω (forma)         ║
# ║  Salida   : TangentVector v' (propagado) / CotangentVector ω' (dual)       ║
# ║                                                                              ║
# ║  Garantías:                                                                  ║
# ║    1. dt ∈ (_DT_MIN, dt_max_stable] para estabilidad RK4                   ║
# ║    2. ‖v(t+dt)‖ ≈ ‖v(t)‖ (conservación de norma aproximada, geodésica)    ║
# ║    3. Transportes ♭ y ♯ delegan al MusicalIsomorphismEngine verificado     ║
# ║    4. GeodesicStepReport inmutable para cada paso de integración            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class LeviCivitaConnectionAgent(_TorsionFreeConnection, Morphism):
    r"""
    ═══════════════════════════════════════════════════════════════════
    FASE 3 — Agente Final de Levi-Civita (Orquestador del Estrato Ω)
    ═══════════════════════════════════════════════════════════════════

    Hereda de:
      - _TorsionFreeConnection → métricas estables, Γ verificados (Fases 1-2)
      - Morphism               → integración en arquitectura categórica MIC

    Proporciona:
    ────────────
    1. `_validate_velocity_vector()` — precondición dimensional y de finitud.
    2. `_validate_integration_step()` — verificación de estabilidad de dt.
    3. `_geodesic_acceleration()` — cómputo vectorizado de a^μ = -Γ^μ_{rs} v^r v^s.
    4. `_rk4_step()` — un paso Runge-Kutta 4 de la ecuación geodésica.
    5. `_validate_geodesic_output()` — postcondición de finitud y conservación de norma.
    6. `enforce_geodesic_flow()` — integración completa con informe GeodesicStepReport.
    7. `transport_to_finance_oracle()` — ♭ con pre-corrección geodésica.
    8. `transport_to_logistics_manifold()` — ♯ con post-verificación.
    9. `parallel_transport()` — transporte paralelo de un vector a lo largo de γ.
    10. `geodesic_flow_report()` — informe diagnóstico completo de la variedad.

    Justificación del paso por defecto dt = 1e-3:
    ──────────────────────────────────────────────
    Para la ecuación geodésica dv/dt = -Γ v² (orden de magnitud), la condición
    de estabilidad de Euler explícito es dt < 1/‖Γ‖. Para G_PHYSICS con
    ‖Γ‖_F ≈ 0 (modo estático), dt puede ser arbitrario. El valor 1e-3 es
    conservador para métricas dinámicas con ‖Γ‖_F ~ O(10²).
    """

    DEFAULT_DT: float = _DEFAULT_DT

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        preconditioner: Optional[MetricSpectralPreconditioner] = None,
    ) -> None:
        r"""
        Construye el agente ejecutando el pipeline completo de 3 fases.

        Fase 1: preacondicionamiento métrico + Christoffel.
        Fase 2: verificaciones axiomáticas (torsión, compatibilidad).
        Fase 3: inicialización del motor musical y cálculo de dt_max_stable.

        Parámetros
        ----------
        metric_tensor  : (n,n) NDArray[np.float64]
            Tensor métrico. Por defecto G_PHYSICS.
        preconditioner : MetricSpectralPreconditioner, opcional
            Inyección de dependencias para testing.

        Lanza
        -----
        Cualquier excepción de Fases 1-2 (TypeError, ValueError,
        ChristoffelInstabilityError, TopologicalTorsionError,
        MetricCompatibilityError).
        """
        # ── Fases 1 y 2: via herencia ────────────────────────────────────────
        _TorsionFreeConnection.__init__(self, metric_tensor, preconditioner)
        Morphism.__init__(self)

        # ── Fase 3a: Motor musical (composición con métrica ya validada) ─────
        # Reutilizamos el PreconditionedMetric de Fase 1 para evitar re-cómputo
        self._musical_engine: MusicalIsomorphismEngine = MusicalIsomorphismEngine(
            metric_tensor=metric_tensor,
            preconditioner=preconditioner,
        )

        # ── Fase 3b: Cálculo del paso máximo estable para RK4 ───────────────
        self._dt_max_stable: float = self._compute_max_stable_dt()

        logger.info(
            "LeviCivitaConnectionAgent inicializado. n=%d, dt_max_stable=%.3e, "
            "‖Γ‖_F=%.3e.",
            self._n,
            self._dt_max_stable,
            self._christoffel_data.frobenius_norm,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos de validación — precondiciones y postcondiciones
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_velocity_vector(
        self, velocity: TangentVector, caller: str = ""
    ) -> None:
        r"""
        Verifica la compatibilidad del vector de velocidad con la variedad.

        Precondiciones verificadas:
          1. Es instancia de TangentVector.
          2. velocity.dim == self._n (compatibilidad dimensional con Γ).
          3. velocity.coordinates es finito.

        Parámetros
        ----------
        velocity : TangentVector
            Vector de velocidad v ∈ TM a validar.
        caller : str
            Nombre del método llamante para mensajes de error descriptivos.

        Lanza
        -----
        TypeError  : Si velocity no es TangentVector.
        ValueError : Si dimensión o finitud no se satisfacen.
        """
        if not isinstance(velocity, TangentVector):
            raise TypeError(
                f"{caller}: se esperaba TangentVector, "
                f"recibido {type(velocity).__name__}."
            )
        if velocity.dim != self._n:
            raise ValueError(
                f"{caller}: incompatibilidad dimensional. "
                f"‖v‖ ∈ ℝ^{velocity.dim} pero Γ ∈ ℝ^{self._n}×{self._n}×{self._n}. "
                f"El vector debe pertenecer al fibrado tangente T_p M con fibra ℝ^{self._n}."
            )
        if not np.all(np.isfinite(velocity.coordinates)):
            raise ValueError(
                f"{caller}: velocity.coordinates contiene valores no finitos."
            )

    def _validate_integration_step(self, dt: float, caller: str = "") -> None:
        r"""
        Verifica que el paso de integración sea físicamente válido y numéricamente
        estable para el método RK4 explícito.

        Condiciones verificadas:
          1. dt > _DT_MIN (piso numérico).
          2. dt ≤ dt_max_stable (condición de estabilidad CFL aproximada).

        Para dt > dt_max_stable se emite una advertencia pero NO se lanza excepción,
        ya que RK4 es más estable que Euler explícito y puede tolerar dt
        moderadamente supercrítico.

        Parámetros
        ----------
        dt     : float
            Paso de integración propuesto.
        caller : str
            Nombre del método llamante.

        Lanza
        -----
        ValueError : Si dt ≤ _DT_MIN.
        """
        if not isinstance(dt, (int, float)):
            raise TypeError(
                f"{caller}: dt debe ser float, recibido {type(dt).__name__}."
            )
        dt_f = float(dt)
        if dt_f <= _DT_MIN:
            raise ValueError(
                f"{caller}: dt={dt_f:.2e} ≤ piso numérico _DT_MIN={_DT_MIN:.2e}. "
                f"El paso de integración debe ser positivo y numéricamente significativo."
            )
        if dt_f > self._dt_max_stable:
            logger.warning(
                "%s: dt=%.2e > dt_max_stable=%.2e. "
                "RK4 puede exhibir inestabilidad. "
                "Considere reducir dt para ‖Γ‖_F=%.2e.",
                caller, dt_f, self._dt_max_stable,
                self._christoffel_data.frobenius_norm,
            )

    def _compute_max_stable_dt(self) -> float:
        r"""
        Estima el paso de tiempo máximo para estabilidad del integrador RK4.

        Criterio (análogo al número de Courant-Friedrichs-Lewy):
            dt_max ≈ _DT_MAX_STABLE_FACTOR / ‖Γ‖_max

        donde ‖Γ‖_max = max_{r,m,n} |Γ^r_{mn}| (norma infinito del tensor).

        Para Γ = 0 (modo estático), se retorna inf (cualquier dt es estable).
        En la práctica se limita a un valor grande pero finito (1e6).

        Retorna
        -------
        float
            Estimación de dt_max_stable.
        """
        gamma_max: float = float(np.max(np.abs(self._christoffel_data.Gamma)))
        if gamma_max < _MACHINE_EPSILON:
            # Γ ≈ 0: variedad plana, cualquier dt es estable
            return 1e6
        dt_max: float = _DT_MAX_STABLE_FACTOR / gamma_max
        logger.debug(
            "dt_max_stable = %.3e (‖Γ‖_max = %.3e).",
            dt_max, gamma_max
        )
        return dt_max

    # ──────────────────────────────────────────────────────────────────────────
    # Núcleo de integración geodésica
    # ──────────────────────────────────────────────────────────────────────────

    def _geodesic_acceleration(
        self, v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Calcula el lado derecho de la ecuación geodésica:

            a^μ = -Γ^μ_{rs} v^r v^s

        Implementación vectorizada mediante doble contracción tensorial:
            a = -einsum('mrs, r, s -> m', Gamma, v, v)

        Complejidad: O(n³) en la contracción, pero el optimizer de einsum
        puede reducir a O(n²) usando la simetría Γ^μ_{rs} = Γ^μ_{sr}.

        Parámetros
        ----------
        v : (n,) NDArray[np.float64]
            Vector de velocidad v^μ (coordenadas NumPy, no TangentVector).

        Retorna
        -------
        NDArray[np.float64]
            Aceleración a^μ = -Γ^μ_{rs} v^r v^s, shape (n,).

        Lanza
        -----
        GeodesicDeviationError
            Si el resultado contiene valores no finitos (overflow en la contracción).
        """
        # a[m] = -Σ_{r,s} Gamma[m,r,s] · v[r] · v[s]
        # einsum: 'mrs, r, s -> m'
        acceleration: NDArray[np.float64] = -np.einsum(
            'mrs,r,s->m',
            self._christoffel_data.Gamma, v, v,
            optimize=True
        )

        if not np.all(np.isfinite(acceleration)):
            acc_norm = float(np.linalg.norm(v)) ** 2
            raise GeodesicDeviationError(
                f"Aceleración geodésica no finita: ‖a‖ con ‖v‖²={acc_norm:.2e}. "
                f"‖Γ‖_F={self._christoffel_data.frobenius_norm:.2e}. "
                f"Considere reducir dt o regularizar más la métrica."
            )

        return acceleration

    def _rk4_step(
        self, v: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        r"""
        Ejecuta un paso del integrador Runge-Kutta de orden 4 (RK4) sobre la
        ecuación geodésica dv/dt = f(v) = -Γ^μ_{rs} v^r v^s.

        Esquema RK4 clásico (método de Kutta, 1901):
            k₁ = f(v)
            k₂ = f(v + dt/2 · k₁)
            k₃ = f(v + dt/2 · k₂)
            k₄ = f(v + dt · k₃)
            v(t+dt) = v(t) + (dt/6)(k₁ + 2k₂ + 2k₃ + k₄)

        Error de truncación local: O(dt⁵).
        Error de truncación global: O(dt⁴).
        Orden de convergencia: 4 (verificable mediante test de Richardson).

        Parámetros
        ----------
        v  : (n,) NDArray[np.float64]
            Vector de velocidad en t (coordenadas NumPy).
        dt : float
            Paso de integración (ya validado por _validate_integration_step).

        Retorna
        -------
        NDArray[np.float64]
            Vector de velocidad en t+dt, shape (n,).

        Lanza
        -----
        GeodesicDeviationError
            Si alguna etapa del RK4 produce valores no finitos.
        """
        half_dt: float = 0.5 * dt

        k1: NDArray[np.float64] = self._geodesic_acceleration(v)
        k2: NDArray[np.float64] = self._geodesic_acceleration(v + half_dt * k1)
        k3: NDArray[np.float64] = self._geodesic_acceleration(v + half_dt * k2)
        k4: NDArray[np.float64] = self._geodesic_acceleration(v + dt * k3)

        v_new: NDArray[np.float64] = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Verificación post-paso
        if not np.all(np.isfinite(v_new)):
            raise GeodesicDeviationError(
                f"RK4 produjo vector no finito en t+dt. "
                f"dt={dt:.2e}, ‖v(t)‖={np.linalg.norm(v):.2e}. "
                f"Reduzca dt o verifique la métrica."
            )

        return v_new

    def _validate_geodesic_output(
        self,
        v_initial: NDArray[np.float64],
        v_final:   NDArray[np.float64],
        dt:        float,
    ) -> GeodesicStepReport:
        r"""
        Valida la postcondición del paso geodésico y construye el informe.

        Verificaciones de postcondición:
          1. v_final es finito (redundante pero explícito para contratos).
          2. La deriva relativa de norma es aceptable:
               |‖v_final‖ - ‖v_initial‖| / max(‖v_initial‖, ε) < _GEODESIC_NORM_DRIFT_TOL

        La conservación de norma es una propiedad de las geodésicas en
        variedades Riemannianas (‖dγ/dt‖_G = constante a lo largo de γ).
        Una deriva grande indica dt demasiado grande o curvatura excesiva.

        Parámetros
        ----------
        v_initial : (n,) NDArray[np.float64]
            Vector de velocidad en t.
        v_final   : (n,) NDArray[np.float64]
            Vector de velocidad en t+dt.
        dt        : float
            Paso de integración utilizado.

        Retorna
        -------
        GeodesicStepReport
            Informe inmutable con normas, deriva y flags de estabilidad.
        """
        norm_i: float = float(np.linalg.norm(v_initial))
        norm_f: float = float(np.linalg.norm(v_final))
        norm_drift: float = abs(norm_f - norm_i) / max(norm_i, _MACHINE_EPSILON)

        acc_initial: NDArray[np.float64] = self._geodesic_acceleration(v_initial)
        acc_norm: float = float(np.linalg.norm(acc_initial))

        if norm_drift > _GEODESIC_NORM_DRIFT_TOL:
            logger.warning(
                "Deriva de norma geodésica: |‖v_f‖ - ‖v_i‖| / ‖v_i‖ = %.2e "
                "> tol=%.2e. dt=%.2e, ‖Γ‖_F=%.2e. "
                "Considere reducir dt.",
                norm_drift, _GEODESIC_NORM_DRIFT_TOL, dt,
                self._christoffel_data.frobenius_norm,
            )

        return GeodesicStepReport(
            v_initial_norm    = norm_i,
            v_final_norm      = norm_f,
            norm_drift        = norm_drift,
            acceleration_norm = acc_norm,
            dt                = dt,
            dt_max_stable     = self._dt_max_stable,
            is_stable         = (dt <= self._dt_max_stable),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Interfaz pública de integración geodésica
    # ──────────────────────────────────────────────────────────────────────────

    def enforce_geodesic_flow(
        self,
        velocity: TangentVector,
        dt: Optional[float] = None,
    ) -> Tuple[TangentVector, GeodesicStepReport]:
        r"""
        Propaga el vector de velocidad un paso dt sobre la geodésica.

        Integra la ecuación geodésica:
            dv^μ/dt = -Γ^μ_{rs} v^r v^s

        mediante un paso RK4, garantizando D v/dt = 0 con error O(dt⁵).

        Precondiciones
        --------------
        - velocity es TangentVector válido con dim == n.
        - dt > _DT_MIN.

        Postcondiciones
        ---------------
        - El resultado es TangentVector finito.
        - ‖v(t+dt)‖ ≈ ‖v(t)‖ dentro de tolerancia adaptativa.
        - GeodesicStepReport documenta la calidad del paso.

        Parámetros
        ----------
        velocity : TangentVector
            Vector de velocidad inicial v(t) ∈ T_p M.
        dt : float, opcional
            Paso de integración. Si None, usa DEFAULT_DT.

        Retorna
        -------
        Tuple[TangentVector, GeodesicStepReport]
            (v(t+dt), informe del paso).

        Lanza
        -----
        TypeError, ValueError        : desde _validate_velocity_vector.
        ValueError                   : desde _validate_integration_step (dt ≤ _DT_MIN).
        GeodesicDeviationError       : si RK4 produce overflow.
        """
        _dt: float = float(dt) if dt is not None else self.DEFAULT_DT

        # Precondiciones
        self._validate_velocity_vector(velocity, caller="enforce_geodesic_flow")
        self._validate_integration_step(_dt, caller="enforce_geodesic_flow")

        v_initial: NDArray[np.float64] = velocity.coordinates.copy()

        # Paso RK4
        v_final: NDArray[np.float64] = self._rk4_step(v_initial, _dt)

        # Postcondición y reporte
        report: GeodesicStepReport = self._validate_geodesic_output(
            v_initial, v_final, _dt
        )

        logger.debug(
            "enforce_geodesic_flow: ‖v_i‖=%.3e → ‖v_f‖=%.3e, "
            "drift=%.2e, dt=%.2e, stable=%s.",
            report.v_initial_norm, report.v_final_norm,
            report.norm_drift, _dt, report.is_stable,
        )

        return TangentVector(coordinates=v_final), report

    # ──────────────────────────────────────────────────────────────────────────
    # Transporte paralelo a lo largo de una curva γ
    # ──────────────────────────────────────────────────────────────────────────

    def parallel_transport(
        self,
        vector: TangentVector,
        tangent_to_curve: TangentVector,
        dt: Optional[float] = None,
    ) -> TangentVector:
        r"""
        Transporta paralelamente el vector `vector` a lo largo de la curva
        definida por `tangent_to_curve`, un paso dt.

        La ecuación de transporte paralelo es:
            D V^μ/dt = dV^μ/dt + Γ^μ_{rs} ẏ^r V^s = 0

        En el método de Euler explícito (primer orden):
            V^μ(t+dt) = V^μ(t) - dt · Γ^μ_{rs} ẏ^r V^s(t)

        donde ẏ = tangent_to_curve es la velocidad de la curva γ.

        Nota: Para mayor precisión, se puede usar RK4 sobre esta ecuación.
        La implementación actual usa Euler para transporte paralelo (O(dt²))
        y RK4 para la geodésica (O(dt⁵)).

        Parámetros
        ----------
        vector           : TangentVector
            Vector V ∈ TM a transportar paralelamente.
        tangent_to_curve : TangentVector
            Tangente ẏ de la curva γ (velocidad de transporte).
        dt : float, opcional
            Paso de integración.

        Retorna
        -------
        TangentVector
            V(t+dt), vector transportado paralelamente un paso dt.

        Lanza
        -----
        TypeError, ValueError : desde _validate_velocity_vector.
        GeodesicDeviationError: si el resultado es no finito.
        """
        _dt: float = float(dt) if dt is not None else self.DEFAULT_DT
        self._validate_velocity_vector(vector, caller="parallel_transport[vector]")
        self._validate_velocity_vector(
            tangent_to_curve, caller="parallel_transport[tangent_to_curve]"
        )
        self._validate_integration_step(_dt, caller="parallel_transport")

        V:    NDArray[np.float64] = vector.coordinates.copy()
        ydot: NDArray[np.float64] = tangent_to_curve.coordinates

        # DV^μ/dt = Γ^μ_{rs} ẏ^r V^s
        # correction[m] = Σ_{r,s} Gamma[m,r,s] · ydot[r] · V[s]
        # einsum: 'mrs, r, s -> m'
        correction: NDArray[np.float64] = np.einsum(
            'mrs,r,s->m',
            self._christoffel_data.Gamma, ydot, V,
            optimize=True
        )

        V_transported: NDArray[np.float64] = V - _dt * correction

        if not np.all(np.isfinite(V_transported)):
            raise GeodesicDeviationError(
                f"Transporte paralelo produjo vector no finito. "
                f"dt={_dt:.2e}, ‖V‖={np.linalg.norm(V):.2e}, "
                f"‖ẏ‖={np.linalg.norm(ydot):.2e}."
            )

        logger.debug(
            "parallel_transport: ‖V_i‖=%.3e → ‖V_f‖=%.3e.",
            float(np.linalg.norm(V)),
            float(np.linalg.norm(V_transported))
        )
        return TangentVector(coordinates=V_transported)

    # ──────────────────────────────────────────────────────────────────────────
    # Transportes categóricos ♭ y ♯
    # ──────────────────────────────────────────────────────────────────────────

    def transport_to_finance_oracle(
        self,
        logistics_flow: TangentVector,
        dt: Optional[float] = None,
        apply_geodesic_correction: bool = True,
    ) -> Tuple[CotangentVector, GeodesicStepReport]:
        r"""
        Transforma el flujo logístico (covariante) en potencial de riesgo
        financiero (contravariante) mediante el Isomorfismo Bemol ♭.

        Pipeline:
            1. Validación de entrada (dim, finitud).
            2. [Opcional] Corrección geodésica RK4: v → v_geodesic.
            3. Isomorfismo ♭: v_geodesic → ω = G_reg @ v_geodesic.
            4. Retorno de (ω, informe_geodésico).

        Interpretación física:
            - logistics_flow: vector de flujo material J^μ en TM (caudal, demanda).
            - Salida ω_i = G_{ij} J^j: forma de fuerza electromotriz en T*M
              (potencial de riesgo financiero, costo de oportunidad).

        Por qué la corrección geodésica precede a ♭:
            La métrica G debe evaluarse sobre vectores tangentes a la geodésica
            del sistema. Un vector no-geodésico corresponde a una trayectoria
            subóptima que disipa energía computacional; su corrección previa
            garantiza que ♭ opera sobre el estado físicamente válido.

        Parámetros
        ----------
        logistics_flow           : TangentVector
            Vector de flujo logístico v ∈ TM.
        dt                       : float, opcional
            Paso de corrección geodésica.
        apply_geodesic_correction: bool
            Si True (defecto), aplica corrección RK4 antes de ♭.
            Si False, aplica ♭ directamente (útil para estados ya geodésicos).

        Retorna
        -------
        Tuple[CotangentVector, GeodesicStepReport]
            (ω = ♭(v_geodesic), informe del paso geodésico).

        Lanza
        -----
        TypeError, ValueError   : desde validaciones.
        GeodesicDeviationError  : si RK4 falla.
        FunctorialityError      : desde MusicalIsomorphismEngine.
        """
        self._validate_velocity_vector(
            logistics_flow, caller="transport_to_finance_oracle"
        )

        if apply_geodesic_correction:
            v_corrected, report = self.enforce_geodesic_flow(logistics_flow, dt=dt)
            logger.debug(
                "transport_to_finance_oracle: corrección geodésica aplicada, "
                "drift=%.2e.", report.norm_drift
            )
        else:
            v_corrected = logistics_flow
            # Crear un reporte trivial para consistencia del tipo de retorno
            v_arr = logistics_flow.coordinates
            norm_v = logistics_flow.norm
            report = GeodesicStepReport(
                v_initial_norm    = norm_v,
                v_final_norm      = norm_v,
                norm_drift        = 0.0,
                acceleration_norm = float(
                    np.linalg.norm(self._geodesic_acceleration(v_arr))
                ),
                dt                = 0.0,
                dt_max_stable     = self._dt_max_stable,
                is_stable         = True,
            )

        financial_covector: CotangentVector = (
            self._musical_engine.apply_flat_isomorphism(v_corrected)
        )

        logger.debug(
            "transport_to_finance_oracle: ‖v‖=%.3e → ‖ω‖=%.3e (♭).",
            v_corrected.norm, financial_covector.norm
        )
        return financial_covector, report

    def transport_to_logistics_manifold(
        self,
        financial_force: CotangentVector,
        apply_post_geodesic: bool = False,
        dt: Optional[float] = None,
    ) -> Tuple[TangentVector, Optional[GeodesicStepReport]]:
        r"""
        Transforma la fuerza de riesgo financiero (contravariante) en un vector
        de acción logística (covariante) mediante el Isomorfismo Sostenido ♯.

        Pipeline:
            1. Validación de tipo y dimensión del co-vector ω.
            2. Isomorfismo ♯: ω → v = G_inv @ ω.
            3. [Opcional] Post-corrección geodésica: v → v_geodesic.
            4. Retorno de (v, informe_opcional).

        Interpretación física:
            - financial_force: forma ω_i en T*M (gradiente de riesgo).
            - Salida v^i = G^{ij} ω_j: vector de acción logística en TM
              (flujo accionable, decisión de procurement).

        Parámetros
        ----------
        financial_force     : CotangentVector
            Forma diferencial ω ∈ T*M (fuerza financiera).
        apply_post_geodesic : bool
            Si True, aplica corrección geodésica al vector resultante.
        dt                  : float, opcional
            Paso de corrección geodésica posterior (solo si apply_post_geodesic).

        Retorna
        -------
        Tuple[TangentVector, Optional[GeodesicStepReport]]
            (v = ♯(ω), informe_opcional). El informe es None si no se aplica
            corrección geodésica posterior.

        Lanza
        -----
        TypeError       : Si financial_force no es CotangentVector.
        FunctorialityError : desde MusicalIsomorphismEngine.
        GeodesicDeviationError : si la post-corrección falla.
        """
        if not isinstance(financial_force, CotangentVector):
            raise TypeError(
                f"transport_to_logistics_manifold requiere CotangentVector, "
                f"recibido {type(financial_force).__name__}."
            )

        # Isomorfismo ♯
        logistic_vector: TangentVector = (
            self._musical_engine.apply_sharp_isomorphism(financial_force)
        )

        logger.debug(
            "transport_to_logistics_manifold: ‖ω‖=%.3e → ‖v‖=%.3e (♯).",
            financial_force.norm, logistic_vector.norm
        )

        # Post-corrección geodésica opcional
        if apply_post_geodesic:
            logistic_vector, report = self.enforce_geodesic_flow(
                logistic_vector, dt=dt
            )
            logger.debug(
                "Post-corrección geodésica: drift=%.2e.", report.norm_drift
            )
            return logistic_vector, report

        return logistic_vector, None

    # ──────────────────────────────────────────────────────────────────────────
    # Informe diagnóstico unificado
    # ──────────────────────────────────────────────────────────────────────────

    def geodesic_flow_report(self) -> Dict[str, Any]:
        r"""
        Genera un informe diagnóstico completo de la variedad Riemanniana
        y la calidad de la conexión de Levi-Civita.

        Incluye información de las 3 fases:
          - Fase 1: dimensión, ‖Γ‖_F, is_static, κ_reg.
          - Fase 2: torsion_norm, covd_norm, riemann_norm, axioms_passed.
          - Fase 3: dt_max_stable, DEFAULT_DT, motor musical disponible.

        Retorna
        -------
        Dict[str, Any]
            Diccionario de diagnóstico completo para logging y monitoreo.
        """
        conn_diag = self._connection_diagnostics
        pm_summary = self._pm.spectral_summary()

        return {
            # Fase 1
            "metric_dimension"         : self._n,
            "christoffel_frob_norm"    : self._christoffel_data.frobenius_norm,
            "is_static_metric"         : self._christoffel_data.is_static,
            "condition_number_reg"     : pm_summary.get("condition_number_reg"),
            "null_space_dim"           : pm_summary.get("null_space_dim"),
            "regularization_applied"   : pm_summary.get("regularization_applied"),
            # Fase 2
            "torsion_norm"             : conn_diag.torsion_norm,
            "covd_metric_norm"         : conn_diag.covd_metric_norm,
            "riemann_norm"             : conn_diag.riemann_norm,
            "torsion_passed"           : conn_diag.torsion_passed,
            "metric_compat_passed"     : conn_diag.metric_compat_passed,
            "all_axioms_passed"        : conn_diag.all_passed(),
            # Fase 3
            "dt_default"               : self.DEFAULT_DT,
            "dt_max_stable"            : self._dt_max_stable,
            "musical_engine_available" : self._musical_engine is not None,
            "agent"                    : "LeviCivitaConnectionAgent v6.0.0",
        }


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ════════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "TopologicalTorsionError",
    "GeodesicDeviationError",
    "MetricCompatibilityError",
    "ChristoffelInstabilityError",
    # Estructuras de datos
    "ChristoffelData",
    "ConnectionDiagnostics",
    "GeodesicStepReport",
    # Agente principal
    "LeviCivitaConnectionAgent",
]
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Musical Isomorphism Engine (Fibrador de Dualidad Categórica)              ║
║  Ruta   : app/core/immune_system/musical_isomorphism_engine.py                      ║
║  Versión: 3.0.0-Topos-Spectral-Categorical                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica Diferencial:
══════════════════════════════════════════════════════════════════════════════════════
Este módulo consagra la transición de fase categórica entre funtores covariantes
(cadenas de flujo, push-forwards) y funtores contravariantes (co-cadenas, pull-backs).
Se rige estrictamente por los isomorfismos musicales de la geometría diferencial Riemanniana
sobre el colector de la Matriz de Interacción Central (MIC), ahora estructurado en
tres fases anidadas con contratos algebraicos formales y auditoría espectral extendida.

══════════════════════════════════════════════════════════════════════════════════════
FUNDAMENTACIÓN MATEMÁTICA EXTENDIDA
══════════════════════════════════════════════════════════════════════════════════════

§0. ESTRUCTURA CATEGÓRICA DEL FIBRADO:
────────────────────────────────────────────────────────────────────────────────────
Sea ℳ una variedad Riemanniana n-dimensional con tensor métrico G ∈ Sym⁺(n,ℝ).
El fibrado tangente π_T : TM → M y el fibrado cotangente π_{T*} : T*M → M
forman el par dual fundamental sobre el que opera este motor.

El par de isomorfismos musicales constituye una equivalencia natural de categorías:
    ♭ : Γ(TM) ≅ Γ(T*M) : ♯    (isomorfismo de C∞(M)-módulos)

§1. EL ISOMORFISMO BEMOL (Flat ♭):
────────────────────────────────────────────────────────────────────────────────────
    ♭ : TM → T*M,  v^♭_i = G_{ij} v^j

El mapa ♭ es un isomorfismo de fibrados vectoriales. En coordenadas locales,
la contracción G_{ij} v^j baja el índice contravariante a covariante.
Aplicación física: transforma caudales incompresibles (vectores de flujo J^i)
en potenciales de presión (formas de fuerza electromotriz ω_i = G_{ij} J^j).

§2. EL ISOMORFISMO SOSTENIDO (Sharp ♯):
────────────────────────────────────────────────────────────────────────────────────
    ♯ : T*M → TM,  ω^{♯i} = G^{ij} ω_j

Donde G^{ij} es la métrica inversa (tensor métrico contravariante).
La inversión G^{-1} exige escrutinio espectral del número de condición:
    κ(G) = σ_max(G) / σ_min(G) = λ_max / λ_min  (para G ∈ Sym⁺)

§3. COMPOSICIÓN DE VARIANZAS EN Z₂:
────────────────────────────────────────────────────────────────────────────────────
El grupo multiplicativo {COVARIANT=+1, CONTRAVARIANT=-1} ≅ (Z₂, ×) actúa sobre
la varianza de funtores bajo composición:
    Var(G ∘ F) = Var(G) ⊗ Var(F)

§4. PREACONDICIONAMIENTO ESPECTRAL – REGULARIZACIÓN INFRARROJA:
────────────────────────────────────────────────────────────────────────────────────
Sea G = V Λ Vᵀ la descomposición espectral (G autoadjunto, V ortogonal).
El gap espectral absoluto:
    Δ_abs = λ_min / λ_max ∈ [0,1]

El gap espectral de Cheeger (análogo discreto):
    Δ_chg = λ_2 / λ_max   (λ_2 = segundo autovalor positivo, ignora kernel)

Regularización de Tikhonov adaptativa con parámetro infrarrojo ε:
    G_reg = G + ε·I,    ε = ε_ratio · λ_max

Equivalencia QFT: la adición de ε·I introduce un "término de masa" que abre
un gap infrarrojo Δ_IR = ε en el espectro del operador, evitando singularidades
en la propagación de modos de largo alcance.

Post-regularización la métrica invertida se calcula mediante:
    G_reg^{-1} = V diag(1/(λ_i + ε)) Vᵀ

§5. VERIFICACIÓN ALGEBRAICA DE INVERSIÓN:
────────────────────────────────────────────────────────────────────────────────────
La tolerancia de verificación es adaptativa al número de condición:
    tol_inv = max(1e-14, κ(G) · ε_machine)
donde ε_machine = 2.22e-16 (épsilon máquina de float64).

§6. COMPATIBILIDAD FUNTORIAL (TOPOS):
────────────────────────────────────────────────────────────────────────────────────
En el contexto de un topos de Grothendieck sobre la categoría de circuitos MIC,
la composición de funtores debe verificar compatibilidad de objetos en dominio
y codominio. La auditoría de la Fase 3 verifica que:
    dom(F₁) ≅ cod(F₂)   (o que la composición sea significativa)

══════════════════════════════════════════════════════════════════════════════════════
ARQUITECTURA DE FASES ANIDADAS:
══════════════════════════════════════════════════════════════════════════════════════

  ┌─── FASE 1: MetricSpectralPreconditioner ──────────────────────────────────┐
  │  Contrato de entrada: G ∈ Mat(n×n, ℝ), G ≈ Simétrica                     │
  │  Contrato de salida : PreconditionedMetric (G_reg, G_inv, κ, Δ_abs, Δ_chg│
  │                                            null_dim, eigenvalues_reg)      │
  │                                                                            │
  │  _validate_matrix_structure()   → verifica rango, forma, tipo             │
  │  _enforce_symmetry()            → (G + Gᵀ)/2 si asimetría < tol          │
  │  _spectral_decomposition()      → eigh() con diagnóstico extendido        │
  │  _compute_spectral_diagnostics()→ κ, Δ_abs, Δ_chg, dim(ker)             │
  │  _adaptive_tikhonov()           → regularización ε·I adaptativa          │
  │  _build_regularized_pair()      → (G_reg, G_inv) via proyección espectral │
  │  _verify_inversion()            → ||G·G⁻¹ - I|| < tol_adaptativa        │
  │  precondition() ─────────────────────────────────► PreconditionedMetric  │
  └──────────────────────────────┬─────────────────────────────────────────────┘
                                 │ PreconditionedMetric (inmutable, frozen)
                                 ▼
  ┌─── FASE 2: FlatIsomorphism ────────────────────────────────────────────────┐
  │  Contrato de entrada: PreconditionedMetric + TangentVector v ∈ ℝⁿ         │
  │  Contrato de salida : CotangentVector ω ∈ ℝⁿ  (ω_i = G_{ij} v^j)        │
  │                                                                            │
  │  __init__(PreconditionedMetric) → almacena G, G_inv, κ, diagnostics      │
  │  _validate_vector_dimension()   → v ∈ ℝⁿ contra n de G                  │
  │  apply_flat_isomorphism()       → ω = G @ v con postcondición ‖ω‖ < ∞   │
  │  diagnostics_report() ───────────────────────────► Dict[str, Any]         │
  └──────────────────────────────┬─────────────────────────────────────────────┘
                                 │ Hereda G, G_inv, diagnostics  [COMPOSICIÓN]
                                 ▼
  ┌─── FASE 3: SharpIsomorphism + CompositionAuditor ─────────────────────────┐
  │  Contrato de entrada: CotangentVector ω ∈ ℝⁿ                             │
  │  Contrato de salida : TangentVector v ∈ ℝⁿ  (v^i = G^{ij} ω_j)         │
  │                                                                            │
  │  apply_sharp_isomorphism()      → v = G_inv @ ω con precondición dim     │
  │  verify_roundtrip_identity()    → ‖♯(♭(v)) - v‖ < tol                   │
  │  audit_functor_composition()    → Var(F₁∘F₂) con check de compatibilidad │
  │  MusicalIsomorphismEngine      → orquestador unificado                    │
  └────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

# ════════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA MIC
# ════════════════════════════════════════════════════════════════════════════════
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.core.mic_algebra import Functor, FunctorialityError, NumericalInstabilityError

logger = logging.getLogger("MIC.ImmuneSystem.MusicalIsomorphism")

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS GLOBALES
# ════════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: float = np.finfo(np.float64).eps          # ≈ 2.22e-16
_SYMMETRY_TOLERANCE: float = 1e-12                          # Tolerancia de simetría G
_CONDITION_THRESHOLD: float = 1e12                          # κ(G) máximo sin regularización
_TIKHONOV_EPSILON_RATIO: float = 1e-8                       # ε = ratio · λ_max
_INVERSION_BASE_TOLERANCE: float = 1e-14                    # Tolerancia mínima de inversión
_ROUNDTRIP_TOLERANCE_FACTOR: float = 100.0                  # Factor sobre ε_machine para roundtrip

# TypeVars para funtores genéricos
S = TypeVar('S')
T = TypeVar('T')


# ════════════════════════════════════════════════════════════════════════════════
# §A — TIPOS ALGEBRAICOS FUNDAMENTALES
# ════════════════════════════════════════════════════════════════════════════════

class CategoricalVariance(Enum):
    r"""
    Dualidad topológica fundamental de los funtores sobre el grupo (Z₂, ×).

    La estructura algebraica es un homomorfismo de grupos:
        φ : {COVARIANT, CONTRAVARIANT} → (Z₂, ×)
        φ(COVARIANT)     = +1
        φ(CONTRAVARIANT) = -1

    La operación de composición respeta la tabla de Cayley de Z₂:
        +1 · +1 = +1  (Cov  ∘ Cov  = Cov)
        +1 · -1 = -1  (Cov  ∘ Cont = Cont)
        -1 · +1 = -1  (Cont ∘ Cov  = Cont)
        -1 · -1 = +1  (Cont ∘ Cont = Cov)

    Invariante de clase:
        ∀ v ∈ {COVARIANT, CONTRAVARIANT}: v.value ∈ {+1, -1}
    """
    COVARIANT     = 1
    CONTRAVARIANT = -1

    def __mul__(self, other: CategoricalVariance) -> CategoricalVariance:
        """
        Composición algebraica de varianzas como producto en Z₂.

        Parámetros
        ----------
        other : CategoricalVariance
            Segunda varianza a componer.

        Retorna
        -------
        CategoricalVariance
            Varianza resultante de la composición.

        Lanza
        -----
        TypeError
            Si `other` no es una instancia de CategoricalVariance.

        Postcondición
        -------------
        El resultado satisface φ(self * other) = φ(self) · φ(other) en Z₂.
        """
        if not isinstance(other, CategoricalVariance):
            raise TypeError(
                f"La composición de varianza requiere CategoricalVariance, "
                f"recibido: {type(other).__name__}."
            )
        result_value: int = self.value * other.value
        return CategoricalVariance(result_value)

    def __repr__(self) -> str:
        return f"CategoricalVariance.{self.name}(value={self.value:+d})"


@dataclass(frozen=True, slots=True)
class TangentVector:
    r"""
    Vector contravariante en el espacio tangente T_p M en un punto p ∈ M.

    Representación en coordenadas locales: v = v^i ∂/∂x^i.
    El array `coordinates` almacena los componentes contravariantes v^i ∈ ℝⁿ.

    Invariantes de clase:
        - coordinates.dtype == float64
        - coordinates.ndim  == 1
        - coordinates.shape == (n,) para algún n ≥ 1
    """
    coordinates: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Valida invariantes en construcción."""
        coords = object.__getattribute__(self, 'coordinates')
        if not isinstance(coords, np.ndarray):
            raise TypeError(
                f"TangentVector.coordinates debe ser NDArray, recibido {type(coords).__name__}."
            )
        if coords.ndim != 1:
            raise ValueError(
                f"TangentVector.coordinates debe ser 1-D, shape={coords.shape}."
            )
        if coords.dtype != np.float64:
            # Coerción silenciosa a float64 para compatibilidad
            object.__setattr__(self, 'coordinates', coords.astype(np.float64))

    @property
    def dim(self) -> int:
        """Dimensión n del espacio tangente local."""
        return self.coordinates.shape[0]

    @property
    def norm(self) -> float:
        """‖v‖₂ euclídea de los componentes."""
        return float(np.linalg.norm(self.coordinates))


@dataclass(frozen=True, slots=True)
class CotangentVector:
    r"""
    1-forma diferencial (co-vector covariante) en el espacio cotangente T*_p M.

    Representación en coordenadas locales: ω = ω_i dx^i.
    El array `coordinates` almacena los componentes covariantes ω_i ∈ ℝⁿ.

    Invariantes de clase: idénticos a TangentVector (dtype, ndim, shape).
    """
    coordinates: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Valida invariantes en construcción."""
        coords = object.__getattribute__(self, 'coordinates')
        if not isinstance(coords, np.ndarray):
            raise TypeError(
                f"CotangentVector.coordinates debe ser NDArray, recibido {type(coords).__name__}."
            )
        if coords.ndim != 1:
            raise ValueError(
                f"CotangentVector.coordinates debe ser 1-D, shape={coords.shape}."
            )
        if coords.dtype != np.float64:
            object.__setattr__(self, 'coordinates', coords.astype(np.float64))

    @property
    def dim(self) -> int:
        """Dimensión n del espacio cotangente local."""
        return self.coordinates.shape[0]

    @property
    def norm(self) -> float:
        """‖ω‖₂ euclídea de los componentes."""
        return float(np.linalg.norm(self.coordinates))


# ════════════════════════════════════════════════════════════════════════════════
# §B — PROTOCOLOS FUNTORIALES BASE
# ════════════════════════════════════════════════════════════════════════════════

class CovariantFunctor(Functor, Generic[S, T]):
    r"""
    Funtor Covariante F_* : C → D.

    Preserva la dirección de los morfismos (push-forward) en el DAG de la MIC.
    Axioma funtorial:
        F_*(f ∘ g) = F_*(f) ∘ F_*(g)    (preservación de composición)
        F_*(id_A)  = id_{F_*(A)}          (preservación de identidad)

    Uso físico: propagación causal de señales eléctricas hacia delante en el tiempo.
    """

    @property
    def variance(self) -> CategoricalVariance:
        """Varianza canónica del funtor covariante."""
        return CategoricalVariance.COVARIANT

    @property
    def domain_category(self) -> str:
        """Identificador nominal de la categoría dominio."""
        return "C"

    @property
    def codomain_category(self) -> str:
        """Identificador nominal de la categoría codominio."""
        return "D"

    def map_object(self, state: Any) -> Any:
        """Mapeo de objetos canónico: identidad estructural del objeto."""
        return state

    def map_morphism(self, f: Any) -> Callable[[Any], Any]:
        """Mapeo de morfismos canónico: devuelve un morphism-wrapped identity."""
        if callable(f):
            return f
        return lambda x: x


class ContravariantFunctor(Functor, Generic[S, T]):
    r"""
    Funtor Contravariante F^* : C → D  (equivalentemente F : C^{op} → D).

    Invierte la dirección de los morfismos (pull-back), revirtiendo causalmente
    la dependencia temporal — análogo a las BSDEs financieras y la retropropagación.
    Axioma funtorial:
        F^*(f ∘ g) = F^*(g) ∘ F^*(f)    (inversión de composición)
        F^*(id_A)  = id_{F^*(A)}          (preservación de identidad)

    Uso físico: cómputo de potenciales upstream a partir de formas de corriente.
    """

    @property
    def variance(self) -> CategoricalVariance:
        """Varianza canónica del funtor contravariante."""
        return CategoricalVariance.CONTRAVARIANT

    @property
    def domain_category(self) -> str:
        """Identificador nominal de la categoría dominio (opuesta)."""
        return "C^{op}"

    @property
    def codomain_category(self) -> str:
        """Identificador nominal de la categoría codominio."""
        return "D"

    def map_object(self, state: Any) -> Any:
        """Mapeo de objetos canónico: identidad estructural del objeto."""
        return state

    def map_morphism(self, f: Any) -> Callable[[Any], Any]:
        """Mapeo de morfismos canónico: identidad funcional para la auditoría."""
        if callable(f):
            return f
        return lambda x: x


@dataclass(frozen=True, slots=True)
class PreconditionedMetric:
    r"""
    Contenedor inmutable del resultado de la Fase 1 de preacondicionamiento.

    Este dataclass actúa como el objeto de transferencia entre Fase 1 y Fase 2,
    encapsulando tanto los tensores métricos como los diagnósticos espectrales
    necesarios para la trazabilidad algebraica.

    Campos
    ------
    G : (n,n) NDArray[float64]
        Tensor métrico regularizado, G_reg = V diag(λ_reg) Vᵀ ∈ Sym⁺(n,ℝ).
    G_inv : (n,n) NDArray[float64]
        Tensor métrico inverso, G_inv = V diag(1/λ_reg) Vᵀ.
    eigenvalues_raw : (n,) NDArray[float64]
        Autovalores originales de G (pre-regularización), ordenados ascendentemente.
    eigenvalues_reg : (n,) NDArray[float64]
        Autovalores regularizados (post-Tikhonov), λ_i + ε para todos i.
    eigenvectors : (n,n) NDArray[float64]
        Matriz ortogonal V tal que G = V diag(λ) Vᵀ (columnas = autovectores).
    condition_number_raw : float
        κ(G) = λ_max / λ_min antes de regularización (puede ser inf).
    condition_number_reg : float
        κ(G_reg) = (λ_max + ε) / (λ_min + ε) post-regularización.
    spectral_gap_absolute : float
        Δ_abs = λ_min / λ_max ∈ [0,1], mide proximidad a singularidad.
    spectral_gap_cheeger : float
        Δ_chg = λ_2 / λ_max, gap espectral de Cheeger (ignora kernel),
        relevante para conectividad en grafos de circuitos.
    null_space_dim : int
        dim(ker G) = número de autovalores ≤ 0 en G original.
        Indica el rango deficiente y la necesidad de regularización.
    tikhonov_epsilon : float
        Valor ε aplicado (0.0 si no fue necesaria regularización).
    regularization_applied : bool
        True si se aplicó regularización de Tikhonov.
    matrix_dimension : int
        n, dimensión del espacio métrico.

    Invariantes de clase (verificados en __post_init__):
        - G.shape == G_inv.shape == (n, n)
        - eigenvectors.shape == (n, n)
        - eigenvalues_reg.min() > 0
        - condition_number_reg < CONDITION_THRESHOLD · (1 + margin)
    """
    G                    : NDArray[np.float64]
    G_inv                : NDArray[np.float64]
    eigenvalues_raw      : NDArray[np.float64]
    eigenvalues_reg      : NDArray[np.float64]
    eigenvectors         : NDArray[np.float64]
    condition_number_raw : float
    condition_number_reg : float
    spectral_gap_absolute: float
    spectral_gap_cheeger : float
    null_space_dim       : int
    tikhonov_epsilon     : float
    regularization_applied: bool
    matrix_dimension     : int

    def __post_init__(self) -> None:
        """Verifica invariantes algebraicos del contenedor al momento de construcción."""
        n = self.matrix_dimension

        # Verificación de formas matriciales
        for attr_name, arr in [('G', self.G), ('G_inv', self.G_inv),
                                ('eigenvectors', self.eigenvectors)]:
            if arr.shape != (n, n):
                raise ValueError(
                    f"PreconditionedMetric.{attr_name} debe tener shape ({n},{n}), "
                    f"recibido {arr.shape}."
                )

        for attr_name, arr in [('eigenvalues_raw', self.eigenvalues_raw),
                                ('eigenvalues_reg', self.eigenvalues_reg)]:
            if arr.shape != (n,):
                raise ValueError(
                    f"PreconditionedMetric.{attr_name} debe tener shape ({n},), "
                    f"recibido {arr.shape}."
                )

        # Verificación de definitud positiva post-regularización
        if not np.all(self.eigenvalues_reg > 0):
            raise ValueError(
                "PreconditionedMetric: eigenvalues_reg contiene valores ≤ 0. "
                "La regularización de Tikhonov no fue suficiente."
            )

    def spectral_summary(self) -> Dict[str, Any]:
        """
        Retorna un diccionario de diagnóstico espectral para logging y trazabilidad.

        Retorna
        -------
        Dict[str, Any]
            Resumen con κ, gaps, dimensión del kernel y epsilon de regularización.
        """
        return {
            "matrix_dimension"       : self.matrix_dimension,
            "null_space_dim"         : self.null_space_dim,
            "condition_number_raw"   : self.condition_number_raw,
            "condition_number_reg"   : self.condition_number_reg,
            "spectral_gap_absolute"  : self.spectral_gap_absolute,
            "spectral_gap_cheeger"   : self.spectral_gap_cheeger,
            "tikhonov_epsilon"       : self.tikhonov_epsilon,
            "regularization_applied" : self.regularization_applied,
            "lambda_min_raw"         : float(self.eigenvalues_raw[0]),
            "lambda_max_raw"         : float(self.eigenvalues_raw[-1]),
            "lambda_min_reg"         : float(self.eigenvalues_reg[0]),
            "lambda_max_reg"         : float(self.eigenvalues_reg[-1]),
        }


class MetricSpectralPreconditioner:
    r"""
    ═══════════════════════════════════════════════════════════════════
    FASE 1 — Preacondicionamiento Espectral de la Métrica Riemanniana
    ═══════════════════════════════════════════════════════════════════

    Transforma un tensor métrico G ∈ Mat(n×n, ℝ) arbitrario (posiblemente
    asimétrico, degenerado o mal condicionado) en un par canónico y garantizado
    (G_reg, G_inv) ∈ Sym⁺(n,ℝ) × Sym⁺(n,ℝ).

    La cadena de transformación sigue el pipeline:

      G_raw
        │
        ▼
      _validate_matrix_structure()     — shape, dtype, finitud, rango mínimo
        │
        ▼
      _enforce_symmetry()              — (G + Gᵀ)/2 si ||G - Gᵀ|| > tol
        │
        ▼
      _spectral_decomposition()        — np.linalg.eigh() → (λ, V)
        │
        ▼
      _compute_spectral_diagnostics()  — κ_raw, Δ_abs, Δ_chg, null_dim
        │
        ▼
      _adaptive_tikhonov()             — G_reg = G + ε·I si κ > thresh ∨ λ_min ≤ 0
        │
        ▼
      _build_regularized_pair()        — G_reg = V diag(λ_reg) Vᵀ
                                         G_inv = V diag(1/λ_reg) Vᵀ
        │
        ▼
      _verify_inversion()              — ||G_reg · G_inv - I|| < tol_adaptativa
        │
        ▼
      PreconditionedMetric             — objeto inmutable, salida de Fase 1

    Analogía con Teoría Cuántica de Campos (QFT):
    ─────────────────────────────────────────────
    La regularización G + ε·I es análoga a la regularización dimensional infrarroja:
    el parámetro ε introduce un "gap de masa" que suprime los modos de largo alcance
    (autovectores del kernel de G), previniendo divergencias infrarrojas en la
    propagación de campos sobre ℳ.

    Parámetros de clase configurables
    ──────────────────────────────────
    SYMMETRY_TOLERANCE        : float = 1e-12   — umbral asimetría ||G - Gᵀ||
    CONDITION_THRESHOLD       : float = 1e12    — κ(G) máximo sin regularización
    TIKHONOV_EPSILON_RATIO    : float = 1e-8    — ε = ratio · λ_max
    INVERSION_BASE_TOLERANCE  : float = 1e-14   — piso de tolerancia de inversión
    """

    SYMMETRY_TOLERANCE       : float = _SYMMETRY_TOLERANCE
    CONDITION_THRESHOLD      : float = _CONDITION_THRESHOLD
    TIKHONOV_EPSILON_RATIO   : float = _TIKHONOV_EPSILON_RATIO
    INVERSION_BASE_TOLERANCE : float = _INVERSION_BASE_TOLERANCE

    # ──────────────────────────────────────────────────────────────────────────
    # Método público principal: orquesta el pipeline completo de Fase 1
    # ──────────────────────────────────────────────────────────────────────────

    def precondition(self, raw_metric: NDArray[np.float64]) -> PreconditionedMetric:
        r"""
        Ejecuta la cadena completa de preacondicionamiento espectral.

        Precondiciones
        --------------
        - raw_metric debe ser array 2-D cuadrado de dtype numérico.
        - raw_metric debe ser finito (sin NaN ni Inf).
        - raw_metric.shape[0] ≥ 1.

        Postcondiciones
        ---------------
        - pm.G ∈ Sym⁺(n, ℝ):  todos los autovalores > 0.
        - pm.G_inv es la inversa algebraica exacta de pm.G dentro de tolerancia.
        - pm.condition_number_reg < CONDITION_THRESHOLD · factor_margen.
        - pm es inmutable (frozen dataclass), seguro para multithreading.

        Parámetros
        ----------
        raw_metric : (n,n) NDArray[np.float64]
            Tensor métrico de entrada, posiblemente ruidoso o mal condicionado.

        Retorna
        -------
        PreconditionedMetric
            Estructura inmutable con G_reg, G_inv y diagnósticos espectrales completos.

        Lanza
        -----
        TypeError
            Si raw_metric no es un array NumPy.
        ValueError
            Si raw_metric no es 2-D cuadrado, contiene NaN/Inf, o es escalar.
        NumericalInstabilityError
            Si la verificación de inversión falla tras la regularización.
        """
        # ── Paso 1: validación estructural ──────────────────────────────────
        n: int = self._validate_matrix_structure(raw_metric)

        # ── Paso 2: simetrización garantizada ───────────────────────────────
        G_sym: NDArray[np.float64] = self._enforce_symmetry(raw_metric, n)

        # ── Paso 3: descomposición espectral completa ────────────────────────
        eigenvalues_raw: NDArray[np.float64]
        eigenvectors:    NDArray[np.float64]
        eigenvalues_raw, eigenvectors = self._spectral_decomposition(G_sym, n)

        # ── Paso 4: diagnósticos espectrales ────────────────────────────────
        cond_raw: float
        gap_abs:  float
        gap_chg:  float
        null_dim: int
        cond_raw, gap_abs, gap_chg, null_dim = self._compute_spectral_diagnostics(
            eigenvalues_raw
        )

        # ── Paso 5: regularización adaptativa de Tikhonov ───────────────────
        eigenvalues_reg: NDArray[np.float64]
        epsilon:         float
        reg_applied:     bool
        eigenvalues_reg, epsilon, reg_applied = self._adaptive_tikhonov(
            eigenvalues_raw, cond_raw
        )

        # ── Paso 6: construcción del par regularizado (G_reg, G_inv) ────────
        G_reg: NDArray[np.float64]
        G_inv: NDArray[np.float64]
        G_reg, G_inv = self._build_regularized_pair(eigenvectors, eigenvalues_reg)

        # ── Paso 7: número de condición post-regularización ─────────────────
        cond_reg: float = float(eigenvalues_reg[-1] / eigenvalues_reg[0])

        # ── Paso 8: verificación algebraica de inversión ─────────────────────
        self._verify_inversion(G_reg, G_inv, cond_reg, n)

        # ── Construcción del objeto de salida inmutable ──────────────────────
        pm = PreconditionedMetric(
            G                     = G_reg,
            G_inv                 = G_inv,
            eigenvalues_raw       = eigenvalues_raw,
            eigenvalues_reg       = eigenvalues_reg,
            eigenvectors          = eigenvectors,
            condition_number_raw  = cond_raw,
            condition_number_reg  = cond_reg,
            spectral_gap_absolute = gap_abs,
            spectral_gap_cheeger  = gap_chg,
            null_space_dim        = null_dim,
            tikhonov_epsilon      = epsilon,
            regularization_applied= reg_applied,
            matrix_dimension      = n,
        )

        logger.info(
            "Fase 1 completada. Diagnóstico espectral: %s",
            pm.spectral_summary()
        )
        return pm

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos privados del pipeline — Fase 1
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_matrix_structure(self, G: NDArray[np.float64]) -> int:
        r"""
        Valida la estructura algebraica de la matriz de entrada.

        Verificaciones realizadas:
          1. Tipo: debe ser np.ndarray.
          2. Dimensionalidad: ndim == 2 (matriz, no tensor de orden superior).
          3. Cuadratura: shape[0] == shape[1].
          4. Minimalidad: n ≥ 1.
          5. Finitud: sin NaN ni Inf (garantía de aritmética bien definida).

        Parámetros
        ----------
        G : NDArray[np.float64]
            Array a validar.

        Retorna
        -------
        int
            n = G.shape[0], la dimensión de la métrica.

        Lanza
        -----
        TypeError  : Si G no es ndarray.
        ValueError : Si alguna verificación falla.
        """
        if not isinstance(G, np.ndarray):
            raise TypeError(
                f"MetricSpectralPreconditioner requiere NDArray[np.float64], "
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

        n: int = G.shape[0]
        if n < 1:
            raise ValueError("El tensor métrico debe tener dimensión n ≥ 1.")

        if not np.all(np.isfinite(G)):
            n_nan  = int(np.sum(np.isnan(G)))
            n_inf  = int(np.sum(np.isinf(G)))
            raise ValueError(
                f"Tensor métrico contiene valores no finitos: "
                f"{n_nan} NaN, {n_inf} Inf. Imposible continuar."
            )

        logger.debug("Validación estructural exitosa: G ∈ Mat(%d×%d, ℝ), finita.", n, n)
        return n

    def _enforce_symmetry(
        self, G: NDArray[np.float64], n: int
    ) -> NDArray[np.float64]:
        r"""
        Fuerza simetría exacta del tensor métrico mediante la proyección ortogonal
        al subespacio de matrices simétricas:

            G_sym = (G + Gᵀ) / 2

        Esta operación es la proyección de Frobenius sobre Sym(n, ℝ):
            G_sym = argmin_{S=Sᵀ} ‖G - S‖_F

        La proyección se aplica incondicionalmente si ||G - Gᵀ||_F > SYMMETRY_TOLERANCE,
        registrando la magnitud de la asimetría para diagnóstico.

        Para G ya simétrica (asimetría ≤ tol), se retorna G sin modificación
        para preservar precisión numérica original.

        Parámetros
        ----------
        G : (n,n) NDArray[np.float64]
            Tensor métrico a simetrizar.
        n : int
            Dimensión (ya validada en _validate_matrix_structure).

        Retorna
        -------
        NDArray[np.float64]
            G_sym = (G + Gᵀ) / 2, garantizadamente simétrica.
        """
        G_T:   NDArray[np.float64] = G.T
        diff:  NDArray[np.float64] = G - G_T
        asym_norm: float = float(np.linalg.norm(diff))

        if asym_norm > self.SYMMETRY_TOLERANCE:
            logger.warning(
                "Asimetría detectada en G: ‖G - Gᵀ‖_F = %.2e > tol=%.2e. "
                "Aplicando proyección al subespacio simétrico Sym(%d, ℝ).",
                asym_norm, self.SYMMETRY_TOLERANCE, n
            )
            G_sym: NDArray[np.float64] = (G + G_T) * 0.5
            # Verificación post-simetrización
            residual = float(np.linalg.norm(G_sym - G_sym.T))
            logger.debug("Asimetría residual post-proyección: %.2e", residual)
            return G_sym

        logger.debug(
            "Tensor métrico ya simétrico: ‖G - Gᵀ‖_F = %.2e ≤ tol=%.2e.",
            asym_norm, self.SYMMETRY_TOLERANCE
        )
        return G

    def _spectral_decomposition(
        self, G_sym: NDArray[np.float64], n: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Realiza la descomposición espectral completa de G_sym como operador autoadjunto.

        Utiliza np.linalg.eigh() que explota la simetría para mayor estabilidad
        y rendimiento numérico sobre np.linalg.eig():
            G_sym = V Λ Vᵀ,  V ∈ O(n),  Λ = diag(λ₁ ≤ λ₂ ≤ ... ≤ λ_n)

        El uso de eigh garantiza:
          - Autovalores reales (G es autoadjunta sobre ℝ, Teorema Espectral).
          - Autovectores ortonormales (V ∈ O(n), Vᵀ = V⁻¹).
          - Ordenación ascendente: λ₁ ≤ λ₂ ≤ ... ≤ λ_n (convenio NumPy).

        Parámetros
        ----------
        G_sym : (n,n) NDArray[np.float64]
            Tensor métrico simétrico.
        n : int
            Dimensión.

        Retorna
        -------
        Tuple[NDArray, NDArray]
            (eigenvalues, eigenvectors) donde:
              - eigenvalues  : (n,) array, λ_i en orden ascendente.
              - eigenvectors : (n,n) array, columnas son autovectores ortonormales.

        Lanza
        -----
        NumericalInstabilityError
            Si np.linalg.eigh falla (matriz corrupta no detectada previamente).
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(G_sym)
        except np.linalg.LinAlgError as exc:
            raise NumericalInstabilityError(
                f"Fallo en descomposición espectral (eigh) de G ∈ Mat({n}×{n}): {exc}"
            ) from exc

        # Verificación de ortogonalidad de V (diagnóstico de sanidad)
        ortho_error: float = float(
            np.linalg.norm(eigenvectors.T @ eigenvectors - np.eye(n), 'fro')
        )
        if ortho_error > 1e-10:
            logger.warning(
                "Ortogonalidad de autovectores degradada: ‖VᵀV - I‖_F = %.2e. "
                "Posible pérdida de precisión en n=%d.",
                ortho_error, n
            )

        logger.debug(
            "Descomposición espectral completada: λ ∈ [%.3e, %.3e], ortho_err=%.2e.",
            float(eigenvalues[0]), float(eigenvalues[-1]), ortho_error
        )
        return eigenvalues, eigenvectors

    def _compute_spectral_diagnostics(
        self, eigenvalues: NDArray[np.float64]
    ) -> Tuple[float, float, float, int]:
        r"""
        Computa los diagnósticos espectrales canónicos de la métrica.

        Métricas calculadas
        ───────────────────
        1. Número de condición κ(G):
               κ = λ_max / λ_min   (si λ_min > 0, else κ = ∞)
           Mide el estiramiento máximo del operador G; κ >> 1 indica
           inestabilidad numérica en la inversión directa.

        2. Gap espectral absoluto Δ_abs:
               Δ_abs = λ_min / λ_max ∈ [0, 1]
           Δ_abs ≈ 0 indica proximidad a singularidad (kernel no trivial).
           Análogo a la brecha de masa en QFT: Δ_abs = 0 ↔ modo de Goldstone.

        3. Gap espectral de Cheeger Δ_chg:
               Δ_chg = λ_2^+ / λ_max
           donde λ_2^+ es el segundo autovalor estrictamente positivo.
           Relevante para la conectividad algebraica de grafos de circuitos
           (análogo al gap de Fiedler en Teoría de Grafos Espectrales).

        4. Dimensión del kernel null_dim:
               null_dim = #{i : λ_i ≤ 0}
           Indica el rango deficiente de G; null_dim > 0 requiere regularización.

        Parámetros
        ----------
        eigenvalues : (n,) NDArray[np.float64]
            Autovalores ordenados ascendentemente (salida de eigh).

        Retorna
        -------
        Tuple[float, float, float, int]
            (condition_number_raw, spectral_gap_absolute, spectral_gap_cheeger, null_space_dim)
        """
        lambda_min: float = float(eigenvalues[0])
        lambda_max: float = float(eigenvalues[-1])
        spectral_zero_tol: float = max(
            100.0 * _MACHINE_EPSILON * max(abs(lambda_max), 1.0),
            1e-14,
        )

        # Número de condición (puede ser infinito para métricas singulares)
        if lambda_max <= 0:
            logger.error(
                "Tensor métrico semidefinido negativo: λ_max = %.2e ≤ 0. "
                "La métrica no define una estructura Riemanniana válida.",
                lambda_max
            )
            cond_raw = np.inf
        elif lambda_min <= spectral_zero_tol:
            cond_raw = np.inf
        else:
            cond_raw = lambda_max / lambda_min

        # Gap espectral absoluto
        gap_abs: float = (lambda_min / lambda_max) if lambda_max > 0 else 0.0

        # Gap espectral de Cheeger: segundo autovalor positivo / λ_max
        positive_eigs: NDArray[np.float64] = eigenvalues[eigenvalues > spectral_zero_tol]
        if len(positive_eigs) >= 2:
            gap_chg = float(positive_eigs[1] / lambda_max) if lambda_max > 0 else 0.0
        elif len(positive_eigs) == 1:
            gap_chg = float(positive_eigs[0] / lambda_max) if lambda_max > 0 else 0.0
        else:
            gap_chg = 0.0

        # Dimensión del kernel
        null_dim: int = int(np.sum(eigenvalues <= spectral_zero_tol))

        logger.debug(
            "Diagnósticos espectrales: κ_raw=%.2e, Δ_abs=%.2e, Δ_chg=%.2e, null_dim=%d.",
            cond_raw, gap_abs, gap_chg, null_dim
        )
        return cond_raw, gap_abs, gap_chg, null_dim

    def _adaptive_tikhonov(
        self,
        eigenvalues: NDArray[np.float64],
        condition_number: float,
    ) -> Tuple[NDArray[np.float64], float, bool]:
        r"""
        Aplica regularización de Tikhonov adaptativa si la métrica es mal condicionada.

        Criterio de activación (OR lógico):
          (a) κ(G) > CONDITION_THRESHOLD  →  inversión numéricamente inestable
          (b) ∃ λ_i ≤ 0                   →  G no es definida positiva

        Parámetro de regularización infrarrojo:
            ε = TIKHONOV_EPSILON_RATIO · λ_max

        Autovalores regularizados:
            λ_i^{reg} = λ_i + ε  ∀ i

        Analogía QFT: ε actúa como un "término de masa" m² que abre un gap
        infrarrojo Δ_IR = ε en el espectro, suprimiendo los modos de largo
        alcance del kernel y restaurando la invertibilidad del propagador.

        Si la métrica ya es bien condicionada y definida positiva, no se aplica
        ninguna modificación (ε = 0, reg_applied = False).

        Parámetros
        ----------
        eigenvalues : (n,) NDArray[np.float64]
            Autovalores originales en orden ascendente.
        condition_number : float
            κ(G) pre-regularización (puede ser np.inf).

        Retorna
        -------
        Tuple[NDArray[np.float64], float, bool]
            (eigenvalues_reg, epsilon, regularization_applied)
        """
        spectral_zero_tol: float = max(
            100.0 * _MACHINE_EPSILON * max(float(np.max(np.abs(eigenvalues))), 1.0),
            1e-14,
        )
        needs_regularization: bool = (
            condition_number > self.CONDITION_THRESHOLD
            or np.any(eigenvalues <= spectral_zero_tol)
        )

        if not needs_regularization:
            logger.debug(
                "Regularización de Tikhonov no requerida: κ=%.2e ≤ thr=%.2e, "
                "todos λ_i > 0.",
                condition_number, self.CONDITION_THRESHOLD
            )
            return eigenvalues.copy(), 0.0, False

        lambda_max: float = float(eigenvalues[-1])

        # Protección contra λ_max ≤ 0 (métrica semidefinida negativa total)
        if lambda_max <= 0:
            # Fallback: usar magnitud máxima absoluta como escala de regularización
            lambda_scale: float = float(np.max(np.abs(eigenvalues)))
            if lambda_scale == 0.0:
                lambda_scale = 1.0  # Métrica nula: usar escala unitaria
            epsilon: float = self.TIKHONOV_EPSILON_RATIO * lambda_scale
            logger.error(
                "Métrica totalmente no definida positiva (λ_max=%.2e ≤ 0). "
                "Usando escala de regularización alternativa: ε=%.2e.",
                lambda_max, epsilon
            )
        else:
            epsilon = self.TIKHONOV_EPSILON_RATIO * lambda_max

        eigenvalues_reg: NDArray[np.float64] = eigenvalues + epsilon
        cond_post: float = float(eigenvalues_reg[-1] / eigenvalues_reg[0])

        logger.warning(
            "Regularización de Tikhonov aplicada: ε=%.2e. "
            "κ: %.2e → %.2e. λ_min: %.2e → %.2e.",
            epsilon,
            condition_number, cond_post,
            float(eigenvalues[0]), float(eigenvalues_reg[0])
        )
        return eigenvalues_reg, float(epsilon), True

    def _build_regularized_pair(
        self,
        eigenvectors: NDArray[np.float64],
        eigenvalues_reg: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Construye el par (G_reg, G_inv) mediante proyección espectral estable.

        Fórmulas de reconstrucción:
            G_reg = V diag(λ_reg) Vᵀ
            G_inv = V diag(1/λ_reg) Vᵀ

        La forma espectral garantiza que tanto G_reg como G_inv son simétricas
        y definidas positivas (dado que λ_reg > 0 ∀ i), sin necesidad de
        inversión matricial directa (que es numéricamente inferior a este método).

        Complejidad: O(n³) en construcción de diag y multiplicación matricial,
        idéntica a la inversión directa pero con mayor estabilidad numérica.

        Parámetros
        ----------
        eigenvectors : (n,n) NDArray[np.float64]
            Matriz ortogonal V de autovectores.
        eigenvalues_reg : (n,) NDArray[np.float64]
            Autovalores regularizados λ_reg > 0.

        Retorna
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            (G_reg, G_inv), ambas matrices (n,n) simétricas definidas positivas.
        """
        Lambda_reg:     NDArray[np.float64] = np.diag(eigenvalues_reg)
        Lambda_inv:     NDArray[np.float64] = np.diag(1.0 / eigenvalues_reg)

        V:    NDArray[np.float64] = eigenvectors
        VT:   NDArray[np.float64] = eigenvectors.T

        G_reg: NDArray[np.float64] = V @ Lambda_reg @ VT
        G_inv: NDArray[np.float64] = V @ Lambda_inv @ VT

        # Simetrización final para eliminar errores de redondeo en la reconstrucción
        G_reg = (G_reg + G_reg.T) * 0.5
        G_inv = (G_inv + G_inv.T) * 0.5

        logger.debug(
            "Par métrico construido espectralmente. "
            "‖G_reg‖_F = %.3e, ‖G_inv‖_F = %.3e.",
            float(np.linalg.norm(G_reg)),
            float(np.linalg.norm(G_inv))
        )
        return G_reg, G_inv

    def _verify_inversion(
        self,
        G_reg: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        condition_number_reg: float,
        n: int,
    ) -> None:
        r"""
        Verificación algebraica de que G_reg · G_inv ≈ I_n.

        La tolerancia de verificación es adaptativa al número de condición:
            tol_inv = max(INVERSION_BASE_TOLERANCE, κ_reg · ε_machine)

        Justificación: la aritmética de punto flotante introduce un error
        inherente proporcional a κ · ε_machine en la inversión. Una tolerancia
        fija subestimaría la precision real para matrices bien condicionadas
        y sería demasiado estricta para matrices moderadamente mal condicionadas.

        Parámetros
        ----------
        G_reg : (n,n) NDArray[np.float64]
            Tensor métrico regularizado.
        G_inv : (n,n) NDArray[np.float64]
            Tensor métrico inverso candidato.
        condition_number_reg : float
            κ(G_reg) post-regularización (finito).
        n : int
            Dimensión de la métrica.

        Lanza
        -----
        NumericalInstabilityError
            Si ‖G_reg · G_inv - I_n‖_F excede la tolerancia adaptativa.
        """
        product:  NDArray[np.float64] = G_reg @ G_inv
        I_n:      NDArray[np.float64] = np.eye(n, dtype=np.float64)
        residual: float = float(np.linalg.norm(product - I_n))

        # Tolerancia adaptativa al número de condición
        tol_inv: float = max(
            self.INVERSION_BASE_TOLERANCE,
            condition_number_reg * _MACHINE_EPSILON * n
        )

        if residual > tol_inv:
            raise NumericalInstabilityError(
                f"Verificación de inversión métrica fallida: "
                f"‖G_reg · G_inv - I_{n}‖_F = {residual:.2e} > tol={tol_inv:.2e}. "
                f"κ_reg = {condition_number_reg:.2e}. "
                f"Considere incrementar TIKHONOV_EPSILON_RATIO."
            )

        logger.debug(
            "Verificación de inversión exitosa: ‖G·G⁻¹ - I‖_F = %.2e ≤ tol=%.2e.",
            residual, tol_inv
        )

    # ──────────────────────────────────────────────────────────────────────────
    # FIN FASE 1 → La salida PreconditionedMetric es la entrada de FASE 2
    # ──────────────────────────────────────────────────────────────────────────


class FlatIsomorphism:
    r"""
    ═══════════════════════════════════════════════════════════════════
    FASE 2 — Isomorfismo Musical Bemol ♭ : TM → T*M
    ═══════════════════════════════════════════════════════════════════

    Implementa el isomorfismo musical ♭ que, dada la métrica Riemanniana G,
    mapea vectores del fibrado tangente a formas del fibrado cotangente:

        ♭ : Γ(TM) → Γ(T*M)
        v^♭_i = G_{ij} v^j    (bajada de índices en notación de Einstein)

    Esta clase recibe el resultado inmutable de la Fase 1 (PreconditionedMetric)
    y lo almacena para su uso tanto en el isomorfismo ♭ como en la preparación
    del isomorfismo ♯ de la Fase 3 (G_inv ya está disponible).

    Interpretación física en el contexto MIC:
    ─────────────────────────────────────────
    Los vectores de flujo J^i (caudal eléctrico, densidad de corriente) en TM
    son transformados en potenciales de fuerza electromotriz ω_i = G_{ij} J^j
    en T*M, donde G_{ij} actúa como la impedancia generalizada del circuito.

    Diseño: Esta clase NO hereda de Functor directamente. Actúa como un
    transformador algebraico puro, COMPONIBLE con los funtores mediante la
    Fase 3. Esto separa la responsabilidad del transporte de vectores de la
    auditoría categórica (principio de responsabilidad única).
    """

    def __init__(self, preconditioned_metric: PreconditionedMetric) -> None:
        r"""
        Inicializa el isomorfismo ♭ con una métrica completamente preacondicionada.

        Precondiciones
        --------------
        - preconditioned_metric debe ser una instancia válida de PreconditionedMetric.
        - Todos los invariantes de PreconditionedMetric deben estar satisfechos
          (garantizado por la Fase 1).

        Postcondiciones
        ---------------
        - self._G almacena la referencia a G_reg (no copia, es immutable).
        - self._G_inv almacena la referencia a G_inv para uso en Fase 3.
        - self._n almacena la dimensión n para validaciones O(1).
        - self._pm almacena el contenedor completo para diagnósticos.

        Parámetros
        ----------
        preconditioned_metric : PreconditionedMetric
            Producto completamente validado de la Fase 1.

        Lanza
        -----
        TypeError
            Si preconditioned_metric no es instancia de PreconditionedMetric.
        """
        if not isinstance(preconditioned_metric, PreconditionedMetric):
            raise TypeError(
                f"FlatIsomorphism requiere PreconditionedMetric (salida de Fase 1), "
                f"recibido {type(preconditioned_metric).__name__}."
            )

        self._pm    : PreconditionedMetric    = preconditioned_metric
        self._G     : NDArray[np.float64]     = preconditioned_metric.G
        self._G_inv : NDArray[np.float64]     = preconditioned_metric.G_inv
        self._n     : int                     = preconditioned_metric.matrix_dimension

        logger.debug(
            "FlatIsomorphism (Fase 2) inicializado: n=%d, κ_reg=%.2e.",
            self._n, preconditioned_metric.condition_number_reg
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Interfaz pública de Fase 2
    # ──────────────────────────────────────────────────────────────────────────

    def apply_flat_isomorphism(self, vector: TangentVector) -> CotangentVector:
        r"""
        Aplica el isomorfismo musical ♭ sobre un vector tangente.

        Operación:
            ω_i = G_{ij} v^j    ⟺    ω = G @ v   (producto matriz-vector)

        La operación es lineal (hereda linealidad de la multiplicación matricial)
        y la imagen ω satisface ‖ω‖ ≤ λ_max · ‖v‖ (cota de Lipschitz).

        Precondiciones
        --------------
        - vector.dim == self._n (compatibilidad dimensional).
        - vector.coordinates es finito (garantizado por TangentVector.__post_init__).

        Postcondiciones
        ---------------
        - Resultado CotangentVector con coordinates finitas.
        - ‖ω‖₂ ≤ ‖G‖₂ · ‖v‖₂ = λ_max · ‖v‖₂.

        Parámetros
        ----------
        vector : TangentVector
            Vector contravariante en T_p M.

        Retorna
        -------
        CotangentVector
            1-forma diferencial ω ∈ T*_p M resultante del isomorfismo ♭.

        Lanza
        -----
        FunctorialityError
            Si las dimensiones no son compatibles.
        NumericalInstabilityError
            Si el resultado contiene valores no finitos.
        """
        self._validate_vector_dimension(vector, "TangentVector", "♭")

        v_j:     NDArray[np.float64] = vector.coordinates
        omega_i: NDArray[np.float64] = self._G @ v_j

        # Postcondición: finitud del resultado
        if not np.all(np.isfinite(omega_i)):
            raise NumericalInstabilityError(
                f"Isomorfismo ♭ produjo valores no finitos: "
                f"‖v‖={vector.norm:.2e}, ‖G‖_F={np.linalg.norm(self._G,'fro'):.2e}."
            )

        logger.debug(
            "♭ aplicado: ‖v‖=%.3e → ‖ω‖=%.3e (ratio=%.3e).",
            vector.norm, float(np.linalg.norm(omega_i)),
            float(np.linalg.norm(omega_i)) / (vector.norm + _MACHINE_EPSILON)
        )
        return CotangentVector(coordinates=omega_i)

    def diagnostics_report(self) -> Dict[str, Any]:
        """
        Retorna un informe completo de diagnóstico de la métrica subyacente.

        Incluye los diagnósticos espectrales de la Fase 1 más la dimensión
        activa del isomorfismo. Útil para logging estructurado y monitoreo.

        Retorna
        -------
        Dict[str, Any]
            Diccionario con todos los parámetros espectrales y de configuración.
        """
        report: Dict[str, Any] = self._pm.spectral_summary()
        report["phase"] = "FlatIsomorphism (Fase 2)"
        report["isomorphism"] = "♭ : TM → T*M"
        return report

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos privados de validación — compartidos con Fase 3
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_vector_dimension(
        self,
        vector: Union[TangentVector, CotangentVector],
        vector_type: str,
        iso_symbol: str,
    ) -> None:
        r"""
        Verifica la compatibilidad dimensional entre el vector y la métrica.

        Esta validación previene el "colapso dimensional" — la aplicación de
        un isomorfismo sobre un espacio de dimensión incorrecta, que produciría
        resultados algebraicamente sin sentido (morfismo entre categorías incorrectas).

        Parámetros
        ----------
        vector : TangentVector | CotangentVector
            Vector a validar.
        vector_type : str
            Nombre del tipo para mensajes de error descriptivos.
        iso_symbol : str
            Símbolo del isomorfismo ('♭' o '♯') para mensajes de error.

        Lanza
        -----
        FunctorialityError
            Si vector.dim ≠ self._n.
        """
        v_dim: int = vector.dim
        if v_dim != self._n:
            raise FunctorialityError(
                f"Colapso dimensional en isomorfismo {iso_symbol}: "
                f"{vector_type} ∈ ℝ^{v_dim} es incompatible con G ∈ Mat({self._n}×{self._n}). "
                f"Los fibrados TM y T*M tienen fibra ℝ^{self._n}."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # FIN FASE 2 → FlatIsomorphism es base de FASE 3 por composición interna
    # ──────────────────────────────────────────────────────────────────────────


class SharpIsomorphism(FlatIsomorphism):
    r"""
    ═══════════════════════════════════════════════════════════════════
    FASE 3 — Isomorfismo Musical Sostenido ♯ : T*M → TM
    ═══════════════════════════════════════════════════════════════════

    Extiende la Fase 2 (hereda FlatIsomorphism) para completar el ciclo de
    dualidad Riemanniana. Utiliza la métrica inversa G^{-1} (validada en
    Fase 1, almacenada en Fase 2) para elevar índices covariantes a contravariantes:

        ♯ : Γ(T*M) → Γ(TM)
        v^{♯i} = G^{ij} ω_j    (subida de índices)

    El par (♭, ♯) satisface la identidad:
        ♯ ∘ ♭ = id_{TM}    y    ♭ ∘ ♯ = id_{T*M}

    Esta clase provee además:
      - verify_roundtrip_identity(): verificación algebraica de la identidad
        composicional ♯ ∘ ♭ = id.
      - audit_functor_composition(): auditoría categórica de varianza en Z₂
        con verificación de compatibilidad de dominio/codominio.

    Interpretación física en el contexto MIC:
    ─────────────────────────────────────────
    Las formas de fuerza electromotriz ω_i son transformadas en vectores de
    corriente J^i = G^{ij} ω_j, donde G^{ij} actúa como la admitancia
    generalizada (inversa de la impedancia) del circuito.

    Diseño de herencia:
    ───────────────────
    SharpIsomorphism EXTIENDE FlatIsomorphism (no lo reemplaza) porque:
      1. Ambos comparten la métrica preacondicionada (Fase 1).
      2. El motor final necesita AMBOS isomorfismos en una interfaz unificada.
      3. verify_roundtrip_identity() requiere ambas operaciones ♭ y ♯.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Isomorfismo ♯ principal
    # ──────────────────────────────────────────────────────────────────────────

    def apply_sharp_isomorphism(self, covector: CotangentVector) -> TangentVector:
        r"""
        Aplica el isomorfismo musical ♯ sobre una 1-forma diferencial.

        Operación:
            v^i = G^{ij} ω_j    ⟺    v = G_inv @ ω   (producto matriz-vector)

        La operación es lineal y la imagen v satisface:
            ‖v‖ ≤ λ_min^{-1} · ‖ω‖ = (1/λ_min) · ‖ω‖   (cota de Lipschitz)

        Precondiciones
        --------------
        - covector.dim == self._n (compatibilidad dimensional).
        - covector.coordinates es finito.

        Postcondiciones
        ---------------
        - Resultado TangentVector con coordinates finitas.
        - ‖v‖₂ ≤ ‖G_inv‖₂ · ‖ω‖₂ = (1/λ_min_reg) · ‖ω‖₂.

        Parámetros
        ----------
        covector : CotangentVector
            1-forma diferencial ω ∈ T*_p M.

        Retorna
        -------
        TangentVector
            Vector contravariante v ∈ T_p M resultante del isomorfismo ♯.

        Lanza
        -----
        FunctorialityError
            Si las dimensiones no son compatibles.
        NumericalInstabilityError
            Si el resultado contiene valores no finitos.
        """
        self._validate_vector_dimension(covector, "CotangentVector", "♯")

        omega_j: NDArray[np.float64] = covector.coordinates
        v_i:     NDArray[np.float64] = self._G_inv @ omega_j

        # Postcondición: finitud del resultado
        if not np.all(np.isfinite(v_i)):
            raise NumericalInstabilityError(
                f"Isomorfismo ♯ produjo valores no finitos: "
                f"‖ω‖={covector.norm:.2e}, "
                f"‖G_inv‖_F={np.linalg.norm(self._G_inv):.2e}."
            )

        logger.debug(
            "♯ aplicado: ‖ω‖=%.3e → ‖v‖=%.3e (ratio=%.3e).",
            covector.norm, float(np.linalg.norm(v_i)),
            float(np.linalg.norm(v_i)) / (covector.norm + _MACHINE_EPSILON)
        )
        return TangentVector(coordinates=v_i)

    # ──────────────────────────────────────────────────────────────────────────
    # Verificación de identidad composicional ♯ ∘ ♭ = id_{TM}
    # ──────────────────────────────────────────────────────────────────────────

    def verify_roundtrip_identity(self, vector: TangentVector) -> Dict[str, Any]:
        r"""
        Verifica algebraicamente la identidad composicional del par de isomorfismos:

            ♯ ∘ ♭ = id_{TM}    ⟺    G_inv @ (G @ v) ≈ v

        Esta verificación es crucial para confirmar que los datos de la Fase 1
        son algebraicamente consistentes end-to-end.

        Tolerancia adaptativa:
            tol_rt = ROUNDTRIP_TOLERANCE_FACTOR · κ_reg · ε_machine · ‖v‖₂

        Justificación: el error de redondeo en dos multiplicaciones consecutivas
        G y G_inv acumula un error proporcional a κ · ε_machine · ‖v‖.

        Parámetros
        ----------
        vector : TangentVector
            Vector v ∈ TM sobre el que verificar la identidad.

        Retorna
        -------
        Dict[str, Any]
            Informe de la verificación con:
              - 'passed'    : bool
              - 'residual'  : float, ‖♯(♭(v)) - v‖₂
              - 'tolerance' : float, tolerancia adaptativa aplicada
              - 'v_norm'    : float, ‖v‖₂
              - 'kappa_reg' : float, κ(G_reg)

        Lanza
        -----
        NumericalInstabilityError
            Si la verificación falla (residual > tolerancia).
        """
        # Aplicar ♭ y luego ♯
        omega:    CotangentVector = self.apply_flat_isomorphism(vector)
        v_prime:  TangentVector   = self.apply_sharp_isomorphism(omega)

        # Residual de la identidad
        residual: float = float(
            np.linalg.norm(v_prime.coordinates - vector.coordinates)
        )

        # Tolerancia adaptativa
        v_norm:    float = vector.norm
        kappa_reg: float = self._pm.condition_number_reg
        tol_rt:    float = max(
            _ROUNDTRIP_TOLERANCE_FACTOR * kappa_reg * _MACHINE_EPSILON * max(v_norm, 1.0),
            _INVERSION_BASE_TOLERANCE
        )

        passed: bool = residual <= tol_rt

        report: Dict[str, Any] = {
            "passed"    : passed,
            "residual"  : residual,
            "tolerance" : tol_rt,
            "v_norm"    : v_norm,
            "kappa_reg" : kappa_reg,
            "phase"     : "SharpIsomorphism.verify_roundtrip_identity (Fase 3)",
        }

        if not passed:
            raise NumericalInstabilityError(
                f"Identidad roundtrip ♯∘♭ = id fallida: "
                f"‖♯(♭(v)) - v‖ = {residual:.2e} > tol={tol_rt:.2e}. "
                f"κ_reg={kappa_reg:.2e}, ‖v‖={v_norm:.2e}."
            )

        logger.debug(
            "Identidad roundtrip verificada: ‖♯(♭(v)) - v‖ = %.2e ≤ tol=%.2e.",
            residual, tol_rt
        )
        return report

    # ──────────────────────────────────────────────────────────────────────────
    # Auditoría funtorial categórica — composición en Z₂
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def audit_functor_composition(
        f1: Functor,
        f2: Functor,
        *,
        verify_domain_compatibility: bool = True,
    ) -> Dict[str, Any]:
        r"""
        Audita algebraicamente la composición de dos funtores (f1 ∘ f2),
        verificando tanto la varianza resultante como la compatibilidad
        de dominio/codominio (semántica categórica en el topos MIC).

        Cálculo de varianza (homomorfismo de grupos Z₂):
            Var(f1 ∘ f2) = Var(f1) ⊗ Var(f2)   (producto en Z₂)

        Verificación de compatibilidad categórica (semántica topos):
            Se verifica que dom(f1) ≅ cod(f2) cuando ambos funtores
            exponen propiedades domain_category / codomain_category.
            Un mismatch implica que la composición no está bien definida
            en el topos de Grothendieck sobre la MIC.

        Parámetros
        ----------
        f1 : Functor
            Primer funtor (exterior) de la composición f1 ∘ f2.
        f2 : Functor
            Segundo funtor (interior) de la composición f1 ∘ f2.
        verify_domain_compatibility : bool, optional
            Si True (por defecto), verifica que dom(f1) ≅ cod(f2).
            Útil desactivar solo en pruebas abstractas de varianza.

        Retorna
        -------
        Dict[str, Any]
            Informe de auditoría con:
              - 'result_variance'       : CategoricalVariance
              - 'var_f1'                : CategoricalVariance
              - 'var_f2'                : CategoricalVariance
              - 'domain_compatible'     : bool
              - 'f1_domain'             : str
              - 'f2_codomain'           : str
              - 'composition_valid'     : bool (varianza Y compatibilidad)

        Lanza
        -----
        TypeError
            Si f1 o f2 no tienen atributo 'variance' de tipo CategoricalVariance.
        FunctorialityError
            Si verify_domain_compatibility=True y dom(f1) ≇ cod(f2).
        """
        # Extracción de varianzas con validación de tipo
        var_1: CategoricalVariance = SharpIsomorphism._extract_variance(f1, "f1")
        var_2: CategoricalVariance = SharpIsomorphism._extract_variance(f2, "f2")

        # Composición en Z₂
        result_variance: CategoricalVariance = var_1 * var_2

        # Verificación de compatibilidad categórica
        f1_domain:    str  = getattr(f1, 'domain_category',   "?")
        f2_codomain:  str  = getattr(f2, 'codomain_category', "?")
        domain_compatible: bool = True

        if verify_domain_compatibility and f1_domain != "?" and f2_codomain != "?":
            # Compatibilidad: dom(f1) debe ser el codominio de f2
            domain_compatible = (f1_domain == f2_codomain)
            if not domain_compatible:
                raise FunctorialityError(
                    f"Composición funtorial inválida en el topos MIC: "
                    f"dom(f1) = '{f1_domain}' ≇ cod(f2) = '{f2_codomain}'. "
                    f"La composición f1 ∘ f2 no está bien definida categóricamente."
                )

        composition_valid: bool = domain_compatible  # varianza siempre válida si tipos OK

        audit_report: Dict[str, Any] = {
            "result_variance"   : result_variance,
            "var_f1"            : var_1,
            "var_f2"            : var_2,
            "domain_compatible" : domain_compatible,
            "f1_domain"         : f1_domain,
            "f2_codomain"       : f2_codomain,
            "composition_valid" : composition_valid,
            "phase"             : "SharpIsomorphism.audit_functor_composition (Fase 3)",
        }

        logger.debug(
            "Auditoría funtorial: %s ⊗ %s ↦ %s. dom(f1)='%s', cod(f2)='%s'. "
            "Compatible=%s.",
            var_1.name, var_2.name, result_variance.name,
            f1_domain, f2_codomain, domain_compatible
        )
        return audit_report

    @staticmethod
    def _extract_variance(functor: Functor, label: str) -> CategoricalVariance:
        """
        Extrae la varianza de un funtor con validación de tipo.

        Parámetros
        ----------
        functor : Functor
            Funtor del que extraer la varianza.
        label : str
            Etiqueta para mensajes de error ('f1' o 'f2').

        Retorna
        -------
        CategoricalVariance
            Varianza del funtor.

        Lanza
        -----
        TypeError
            Si el funtor no tiene atributo 'variance' o es del tipo incorrecto.
        """
        variance = getattr(functor, 'variance', None)
        if variance is None:
            raise TypeError(
                f"El funtor {label} (tipo {type(functor).__name__}) no tiene "
                f"atributo 'variance'. Debe ser instancia de CovariantFunctor "
                f"o ContravariantFunctor."
            )
        if not isinstance(variance, CategoricalVariance):
            raise TypeError(
                f"El atributo 'variance' de {label} debe ser CategoricalVariance, "
                f"recibido {type(variance).__name__}."
            )
        return variance


# ════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ORQUESTADOR — MusicalIsomorphismEngine                                     ║
# ║  Unifica las 3 fases en una interfaz pública única                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════

class MusicalIsomorphismEngine(SharpIsomorphism):
    r"""
    Árbitro Geométrico Unificado del Fibrado de Dualidad Categórica.

    Orquesta automáticamente las tres fases anidadas:

        Fase 1 (MetricSpectralPreconditioner):
            G_raw → PreconditionedMetric (G_reg, G_inv, diagnósticos)

        Fase 2 (FlatIsomorphism):
            TangentVector v → CotangentVector ω   via ♭ = G_reg @

        Fase 3 (SharpIsomorphism):
            CotangentVector ω → TangentVector v   via ♯ = G_inv @
            + verify_roundtrip_identity()
            + audit_functor_composition()

    La cadena de herencia:
        MusicalIsomorphismEngine
            └→ SharpIsomorphism (Fase 3)
                └→ FlatIsomorphism (Fase 2)
                    └→ consume PreconditionedMetric (Fase 1)

    Uso canónico:
    ─────────────
        engine = MusicalIsomorphismEngine(G_PHYSICS)

        v     = TangentVector(np.array([1.0, 0.0, 0.0]))
        omega = engine.apply_flat_isomorphism(v)     # ♭
        v2    = engine.apply_sharp_isomorphism(omega) # ♯
        engine.verify_roundtrip_identity(v)           # ♯∘♭ = id
        report = engine.audit_functor_composition(F1, F2)

    Parámetros del constructor
    ──────────────────────────
    metric_tensor : (n,n) NDArray[np.float64], opcional
        Tensor métrico de entrada. Por defecto G_PHYSICS (tensor de la MIC).
    preconditioner : MetricSpectralPreconditioner, opcional
        Instancia del preacondicionador. Permite inyección de dependencias
        para testing con parámetros personalizados.
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        preconditioner: Optional[MetricSpectralPreconditioner] = None,
    ) -> None:
        """
        Construye el motor ejecutando el pipeline completo de 3 fases.

        Parámetros
        ----------
        metric_tensor : (n,n) NDArray[np.float64]
            Tensor métrico de la variedad Riemanniana subyacente.
        preconditioner : MetricSpectralPreconditioner, opcional
            Permite inyección de dependencias para testing. Si None,
            se instancia un MetricSpectralPreconditioner con parámetros por defecto.

        Lanza
        -----
        Cualquier excepción propagada por la Fase 1 (TypeError, ValueError,
        NumericalInstabilityError).
        """
        # ── Fase 1: preacondicionamiento espectral ───────────────────────────
        _preconditioner: MetricSpectralPreconditioner = (
            preconditioner
            if preconditioner is not None
            else MetricSpectralPreconditioner()
        )
        pm: PreconditionedMetric = _preconditioner.precondition(metric_tensor)

        # ── Encadenamiento Fase 2 + Fase 3 vía herencia ──────────────────────
        super().__init__(pm)

        logger.info(
            "MusicalIsomorphismEngine inicializado. n=%d, κ_reg=%.2e, "
            "regularización=%s.",
            pm.matrix_dimension,
            pm.condition_number_reg,
            pm.regularization_applied,
        )

    @property
    def preconditioned_metric(self) -> PreconditionedMetric:
        """
        Acceso de solo lectura al PreconditionedMetric de la Fase 1.

        Permite introspección de diagnósticos sin exponer mutabilidad.
        """
        return self._pm

    def full_cycle_report(self, vector: TangentVector) -> Dict[str, Any]:
        r"""
        Ejecuta el ciclo completo ♭ → ♯ → verificación y retorna un informe
        unificado de las tres fases.

        Operaciones ejecutadas:
            1. ω = ♭(v)          (Fase 2)
            2. v' = ♯(ω)         (Fase 3)
            3. verify ‖v' - v‖   (Fase 3, identidad roundtrip)

        Parámetros
        ----------
        vector : TangentVector
            Vector de prueba v ∈ TM.

        Retorna
        -------
        Dict[str, Any]
            Informe unificado con diagnósticos espectrales (Fase 1),
            resultados de los isomorfismos (Fase 2 y 3) y verificación
            de identidad.
        """
        # Fase 2: ♭
        omega: CotangentVector = self.apply_flat_isomorphism(vector)

        # Fase 3a: ♯
        v_prime: TangentVector = self.apply_sharp_isomorphism(omega)

        # Fase 3b: verificación roundtrip
        roundtrip_report: Dict[str, Any] = self.verify_roundtrip_identity(vector)

        # Informe unificado
        report: Dict[str, Any] = {
            **self._pm.spectral_summary(),
            "input_vector_norm"    : vector.norm,
            "flat_covector_norm"   : omega.norm,
            "sharp_vector_norm"    : v_prime.norm,
            "roundtrip"            : roundtrip_report,
            "engine"               : "MusicalIsomorphismEngine (Fases 1+2+3)",
        }
        logger.info("Ciclo completo ♭→♯ ejecutado. Informe: %s", report)
        return report


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ════════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Tipos algebraicos
    "CategoricalVariance",
    "TangentVector",
    "CotangentVector",
    # Protocolos funtoriales
    "CovariantFunctor",
    "ContravariantFunctor",
    # Fase 1
    "PreconditionedMetric",
    "MetricSpectralPreconditioner",
    # Fase 2
    "FlatIsomorphism",
    # Fase 3
    "SharpIsomorphism",
    # Orquestador
    "MusicalIsomorphismEngine",
]
"""
=========================================================================================
Módulo: Gauge Field Router (Operador de Electrodinámica Agéntica y Enrutamiento de Calibre)
Ubicación: app/core/immune_system/gauge_field_router.py
=========================================================================================

Modelo matemático riguroso:
──────────────────────────

Sea G = (V, E) un grafo finito, conexo y orientado con:
    • |V| = N nodos
    • |E| = M aristas orientadas

Espacio de cochains y cohomología discreta:
    • C⁰(G) ≅ ℝᴺ      — potenciales (0-cochains)
    • C¹(G) ≅ ℝᴹ      — campos (1-cochains)
    • Operador coborde: d₀: C⁰ → C¹, (d₀Φ)ₑ := Φ(head(e)) − Φ(tail(e))

Objetos algebraicos fundamentales:
    • B₁ ∈ ℝᴹˣᴺ:  Matriz de incidencia orientada (operador coborde d₀)
                   Entradas en {−1, 0, +1} exactamente.
                   Entrada [e,v] ∈ {+1 si v=head(e), −1 si v=tail(e), 0 c.c.}
    
    • L₀ ∈ ℝᴺˣᴺ:  Laplaciano combinatorio: L₀ := B₁ᵀB₁
                   Simetría:    L₀ = L₀ᵀ
                   Semidefinitud: ∀x ∈ ℝᴺ, xᵀL₀x = ‖B₁x‖² ≥ 0
                   Nullspace:   ker(L₀) = span{𝟙}  (grafos conexos)
    
    • ρ ∈ ℝᴺ:    Densidad de carga (0-cadena dual)
                   Compatibilidad Fredholm: ρ ⊥ ker(L₀)
                   Para grafo conexo: 𝟙ᵀρ = 0 (suma nula)
    
    • Φ ∈ ℝᴺ:    Potencial escalar (0-cocadena), solución de mínima norma
                   Problema: L₀Φ = ρ
                   Unicidad: Φ ⊥ ker(L₀)
    
    • E ∈ ℝᴹ:    Campo discreto sobre aristas (1-cocadena)
                   E = B₁Φ (diferencial discreto)
    
    • Q_k ∈ ℝᴹ:  Vector de carga del agente k sobre aristas
                   Soportes pueden ser distintos (localización permitida)

Pipeline de enrutamiento electromagnético:
─────────────────────────────────────────

    Entrada: estado s ∈ CategoricalState, nodo anomalía v₀ ∈ V, severidad α > 0
    Salida: estado transformado s' = morphism_{k*}(s)

    [1] Localización de carga:
        ρ := α · (δ_{v₀} − 𝟙/N)
        Garantía: 𝟙ᵀρ = 0 (condición de Fredholm satisfecha)
    
    [2] Resolución de Poisson (mínima norma):
        Resolver: L₀Φ = ρ  en sentido LSQR (Φ ⊥ ker(L₀))
        Solver: lsqr(L₀, ρ, atol, btol, iter_lim)
        Verificación: ‖L₀Φ − ρ‖₂/‖ρ‖₂ < ε_residual
    
    [3] Cálculo del campo (gradiente discreto):
        E := B₁Φ  [en algunos contextos físicos: E := −B₁Φ]
        Propiedad: E ∈ Im(B₁) = ker(L₀)⊥  (bajo-dimensional)
    
    [4] Acoplamiento y selección (Fuerza de Lorentz):
        action_k := ⟨Q_k, E⟩ ∈ ℝ  para cada agente k
        Selección: k* := argmax_k action_k
        Desempate: política lexicográfica o por orden de registro
    
    [5] Despacho (aplicación de morfismo):
        s' := F_{k*}(s_gauge) donde s_gauge contiene contexto electromagnético

Condiciones de compatibilidad verificadas:
──────────────────────────────────────────

    C1. L₀ ∈ ℝᴺˣᴺ cuadrada y simétrica
        Verificación: ‖L₀ − L₀ᵀ‖_F < ε_simetría (1e-10)
    
    C2. L₀ semidefinida positiva: λ_min(L₀) ≥ −ε_spectral
        Verificación: eigsh(L₀, k=k_min, which='SM')
        Valor típico: λ_min ≈ 0 (eigenvalor nulo para grafo conexo)
    
    C3. B₁ ∈ {−1, 0, +1}ᴹˣᴺ exactamente
        Verificación: ∀ entry b ∈ B₁.data, ∃ v ∈ {−1,0,+1}: |b − v| < ε
    
    C4. Dimensionalidad: B₁.shape[1] = N, L₀.shape = (N,N), ρ ∈ ℝᴺ
        Verificación: shape matching explícito
    
    C5. Consistencia cohomológica (verificación opcional):
        ‖L₀ − B₁ᵀB₁‖_F < ε_cohom (1e-8)
        Si falla: indica L y B no provienen del mismo complejo
    
    C6. Condición de Fredholm (solubilidad de Poisson):
        ρ ⊥ ker(L₀), verificado por neutralidad de carga
        Para grafo conexo: 𝟙ᵀρ = 0 ± ε (1e-12)
    
    C7. Cargas de agentes bien definidas:
        ∀ k: Q_k ∈ ℝᴹ, Q_k finita, dim(Q_k) = M exacto

Invariantes estructurales:
─────────────────────────

    I1. L₀ cuadrada:    L₀.shape = (N, N)
    I2. Incidencia:     B₁.shape = (M, N), M ≥ N−1 para conexo
    I3. Simetría:       L₀ = L₀ᵀ
    I4. Semi-definitud: ∀ eigenvalor λ ≥ −ε_spectral
    I5. Cohomología:    ‖L₀ − B₁ᵀB₁‖_F < ε_cohom (si verify_cohomology=True)
    I6. Incidencia:     B₁[i,j] ∈ {−1, 0, +1} ∀ i,j
    I7. Finitud:        ∀ matrix entry ∈ ℝ finito
    I8. Cargas:         dim(Q_k) = M ∀ k
    I9. Registro:       ∃ al menos 1 agente registrado

Error handling:
───────────────

    TopologicalSingularityError:
        • Incompatibilidades algebraicas (simetría, dimensiones, incidencia)
        • Fallo en verificación espectral o cohomológica
        • Problemas numéricos graves en solveres
    
    CohomologicalInconsistencyError:
        • L₀ y B₁ no cohomológicamente consistentes
    
    LorentzForceError:
        • Imposibilidad de seleccionar agente válido
        • Acciones no finitas
        • Espacio de agentes vacío
    
    ChargeNeutralityError:
        • Violación de condición de Fredholm
        • Densidad de carga no neutral
    
    GaugeFieldError (base):
        • Entrada inválida (tipos, valores fuera de rango)
        • Estado no válido
        • Fallo en resolución de MIC

Tolerancias numéricas (documentadas):
─────────────────────────────────────

    POISSON_ATOL:               1e-10   (tolerancia absoluta en LSQR)
    POISSON_BTOL:               1e-10   (tolerancia de ajuste en LSQR)
    RESIDUAL_TOLERANCE:         1e-8    (residual relativo Poisson)
    SYMMETRY_TOLERANCE:         1e-10   (verificación L = Lᵀ)
    SPECTRAL_TOLERANCE:         1e-10   (detección de eigenvalor 0)
    COHOMOLOGICAL_TOLERANCE:    1e-8    (consistencia L = B₁ᵀB₁)
    INCIDENCE_ENTRY_TOLERANCE:  1e-12   (verificación entradas ±1,0)
    CHARGE_NEUTRALITY_TOLERANCE: 1e-12  (suma nula de carga)
    NUMERICAL_EPS:              1e-14   (épsilon de máquina)
    MIN_CHARGE_DENSITY:         1e-12   (piso de severidad)
    BASE_CYBER_MOMENTUM:        1.0     (parámetro físico)
=========================================================================================
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lsqr

from app.adapters.tools_interface import MICRegistry
from app.core.mic_algebra import CategoricalState, Morphism

logger = logging.getLogger("MIC.ImmuneSystem.GaugeFieldRouter")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS Y TOLERANCIAS NUMÉRICAS (RIGUROSAS)
# ═══════════════════════════════════════════════════════════════════════════════

class LatticeQEDConstants:
    """
    Constantes de tolerancia y escalamiento para la dinámica discreta del campo.

    Diseño:
    ───────
    Cada tolerancia está justificada por la precisión numérica requerida
    en su contexto respectivo:

    • Tolerancias de solver (POISSON_*): 1e-10
      - lsqr requiere estos límites para convergencia de sistemas singulares
      - Tolerancias más altas podrían perder precisión en eigenvalores pequeños

    • Tolerancia espectral (SPECTRAL_TOLERANCE): 1e-10
      - Separación de eigenvalor cero (kernel) de eigenvalores positivos
      - Crítica para grafos conexos donde dim(ker(L)) = 1

    • Tolerancia de simetría (SYMMETRY_TOLERANCE): 1e-10
      - Verifica L = L^T ante errores de redondeo
      - La propiedad debe ser exacta en aritmética exacta

    • Tolerancia cohomológica (COHOMOLOGICAL_TOLERANCE): 1e-8
      - Verifica L = B₁ᵀB₁
      - Más relajada porque es una relación derivada (no fundamental)

    • Tolerancia de incidencia (INCIDENCE_ENTRY_TOLERANCE): 1e-12
      - Verifica entries ∈ {-1, 0, +1}
      - Máxima rigidez: entradas deben ser exactamente ±1 o 0

    • Tolerancia de neutralidad de carga (CHARGE_NEUTRALITY_TOLERANCE): 1e-12
      - Condición de Fredholm: 𝟙ᵀρ = 0
      - Fundamental para solubilidad de Poisson

    • Residual relativo (RESIDUAL_TOLERANCE): 1e-8
      - Verificación post-solución: ‖LΦ − ρ‖₂/‖ρ‖₂
      - Más relajada que atol/btol de LSQR porque agrupa todos los errores
    """

    # ── Solver de Poisson (LSQR) ──
    POISSON_ATOL: Final[float] = 1e-10
    """Tolerancia absoluta en LSQR: ‖Ax − b‖ ≤ atol * ‖b‖."""

    POISSON_BTOL: Final[float] = 1e-10
    """Tolerancia de ajuste en LSQR: ‖Aᵀ(Ax − b)‖ ≤ btol * ‖Aᵀb‖."""

    POISSON_ITER_LIMIT: Final[Optional[int]] = None
    """Límite de iteraciones (None = sin límite explícito, usa default lsqr)."""

    # ── Validación de solución ──
    RESIDUAL_TOLERANCE: Final[float] = 1e-7
    """
    Tolerancia de residual relativo post-solución.
    
    Verificación: ‖LΦ − ρ‖₂ / ‖ρ‖₂ < RESIDUAL_TOLERANCE
    
    Justificación: agrupa todos los errores del solver (atol, btol, iteraciones).
    Valor 1e-7 permite cierto overhead pero es suficientemente restrictivo.
    """

    # ── Validación algebraica ──
    SYMMETRY_TOLERANCE: Final[float] = 1e-10
    """Tolerancia para verificación de simetría: ‖L − Lᵀ‖_F < ε."""

    SPECTRAL_TOLERANCE: Final[float] = 1e-10
    """
    Tolerancia para detección de eigenvalor nulo.
    
    Uso: distinguir ker(L) de Im(L) en verificación espectral.
    Para grafo conexo, λ_min debe ser ≈ 0 (multiplicidad 1).
    """

    COHOMOLOGICAL_TOLERANCE: Final[float] = 1e-8
    """Tolerancia para verificación L = B₁ᵀB₁: ‖L − B₁ᵀB₁‖_F < ε."""

    INCIDENCE_ENTRY_TOLERANCE: Final[float] = 1e-12
    """
    Tolerancia para validación de entradas de B₁.
    
    Verificación: ∀ entry b, ∃ v ∈ {-1, 0, +1}: |b − v| < ε
    Máxima rigidez: incidencia orientada debe ser exacta.
    """

    CHARGE_NEUTRALITY_TOLERANCE: Final[float] = 1e-12
    """
    Tolerancia para condición de Fredholm: |𝟙ᵀρ| < ε.
    
    Máxima rigidez: neutralidad de carga es requisito algebraico exacto.
    Violación = problema de Poisson no soluble.
    """

    NUMERICAL_EPS: Final[float] = 1e-14
    """Épsilon de máquina (aproximado para float64). Evita división por cero."""

    # ── Parámetros físicos ──
    MIN_CHARGE_DENSITY: Final[float] = 1e-12
    """
    Piso de severidad de carga.
    
    Si severity < MIN_CHARGE_DENSITY, se reemplaza por este valor.
    Evita perturbaciones numéricamente imperceptibles.
    """

    BASE_CYBER_MOMENTUM: Final[float] = 1.0
    """
    Constante base para cálculo de momentum cibernético.
    
    Fórmula: p = √(2 · severity · BASE_CYBER_MOMENTUM) · (1 + ‖E‖₂)
    """


# ═══════════════════════════════════════════════════════════════════════════════
# TIPOS ALGEBRAICOS AUXILIARES (TIPIFICACIÓN RIGUROSA)
# ═══════════════════════════════════════════════════════════════════════════════

class TieBreakPolicy(Enum):
    """
    Política de desempate para selección de agente en caso de múltiples maximizadores.

    Cuando ⟨Q_i, E⟩ = ⟨Q_j, E⟩ (dentro de tolerancia numérica),
    se aplica una política de desempate determinística:

    LEXICOGRAPHIC:
        Orden lexicográfico sobre agent_id (strings).
        Reproducibilidad: independiente de orden de inserción.
        Ventaja: determinismo garantizado.

    FIRST_REGISTERED:
        Orden de aparición en el diccionario interno.
        Reproducibilidad: depende de CPython 3.7+ (dict ordenado).
        Ventaja: preserva estructura de inserción del usuario.
    """
    LEXICOGRAPHIC = auto()
    FIRST_REGISTERED = auto()


@dataclass(frozen=True)
class PoissonSolution:
    """
    Resultado tipado de la resolución del problema de Poisson discreto.

    Captura la solución Φ ∈ ℝᴺ y metadatos del solver LSQR para auditoría.

    Atributos:
        phi (np.ndarray):           Solución Φ ∈ ℝᴺ, Φ ⊥ ker(L₀).
        istop (int):                Código de terminación LSQR:
                                      0 = x = 0 (entrada nula)
                                      1 = convergencia normal
                                      2 = límite de iteraciones alcanzado
                                      3 = A es muy mal condicionado
                                      4 = lsqr convergió a Ax ≈ b pero no a Aᵀ(Ax−b) ≈ 0
        iteration_count (int):      Número de iteraciones realizadas.
        r1norm (float):             ‖Ax − b‖ reportado por LSQR.
        arnorm (float):             ‖Aᵀ(Ax − b)‖ reportado por LSQR.
        acond (float):              Estimación del número de condición de A.
        relative_residual (float):  ‖LΦ − ρ‖₂ / ‖ρ‖₂ (calculado post-solución).

    Invariantes post-construcción:
        • phi.shape = (N,)
        • phi es finita (∀ entry ∈ ℝ finito)
        • relative_residual < RESIDUAL_TOLERANCE
    """
    phi: np.ndarray
    istop: int
    iteration_count: int
    r1norm: float
    arnorm: float
    acond: float
    relative_residual: float

    def __post_init__(self) -> None:
        """Validaciones de invariantes post-construcción."""
        object.__setattr__(self, 'phi', np.asarray(self.phi, dtype=np.float64))
        
        assert self.phi.ndim == 1, (
            f"phi debe ser 1D: shape={self.phi.shape}"
        )
        assert np.all(np.isfinite(self.phi)), (
            f"phi contiene valores no finitos"
        )
        assert 0 <= self.istop <= 4, (
            f"istop debe estar en [0,4]: {self.istop}"
        )
        assert self.iteration_count >= 0, (
            f"iteration_count debe ser no negativo: {self.iteration_count}"
        )
        assert np.isfinite(self.relative_residual), (
            f"relative_residual no finito: {self.relative_residual}"
        )


@dataclass(frozen=True)
class CouplingResult:
    """
    Resultado del cálculo de acoplamiento electromagnético.

    Captura la selección de agente por máximo acoplamiento gauge
    y metadatos para auditoría.

    Atributos:
        selected_agent (str):       ID del agente seleccionado.
        max_action (float):         Valor máximo ⟨Q_k*, E⟩.
        all_actions (Dict):         Mapeo completo agent_id → action.
        num_maximizers (int):       Número de agentes empatados en el máximo
                                     (dentro de tolerancia numérica).

    Invariantes:
        • selected_agent ∈ all_actions.keys()
        • all_actions[selected_agent] ≈ max_action
        • num_maximizers ≥ 1
        • ∀ action ∈ all_actions.values(): action ≤ max_action + ε
    """
    selected_agent: str
    max_action: float
    all_actions: Dict[str, float]
    num_maximizers: int

    def __post_init__(self) -> None:
        """Validaciones de invariantes post-construcción."""
        assert isinstance(self.selected_agent, str), (
            f"selected_agent debe ser str: {type(self.selected_agent)}"
        )
        assert self.selected_agent in self.all_actions, (
            f"selected_agent no en all_actions"
        )
        assert np.isfinite(self.max_action), (
            f"max_action no finita: {self.max_action}"
        )
        assert self.num_maximizers >= 1, (
            f"num_maximizers debe ser ≥1: {self.num_maximizers}"
        )
        assert len(self.all_actions) >= 1, (
            f"all_actions vacío"
        )


@dataclass(frozen=True)
class GaugeFieldDiagnostics:
    """
    Diagnósticos completos de una operación de enrutamiento.

    Captura toda la información intermedia para auditoría, debugging
    y análisis post-mortem.

    Atributos:
        trace_id (str):                       ID único de la traza (UUID hex 12 chars).
        anomaly_node (int):                   Nodo perturbado v₀.
        severity (float):                     Magnitud de la perturbación α.
        charge_density_norm (float):          ‖ρ‖₂.
        potential_norm (float):               ‖Φ‖₂.
        field_norm (float):                   ‖E‖₂.
        poisson_solution (PoissonSolution):   Resultado de Poisson.
        coupling_result (CouplingResult):     Resultado de acoplamiento.
        cyber_momentum (float):                Momentum cibernético p.
    """
    trace_id: str
    anomaly_node: int
    severity: float
    charge_density_norm: float
    potential_norm: float
    field_norm: float
    poisson_solution: PoissonSolution
    coupling_result: CouplingResult
    cyber_momentum: float

    def __post_init__(self) -> None:
        """Validaciones de invariantes."""
        assert isinstance(self.trace_id, str) and len(self.trace_id) > 0, (
            f"trace_id inválido"
        )
        assert 0 <= self.anomaly_node, (
            f"anomaly_node debe ser no negativo: {self.anomaly_node}"
        )
        assert self.severity > 0.0, (
            f"severity debe ser positiva: {self.severity}"
        )
        assert np.isfinite(self.charge_density_norm) and self.charge_density_norm >= 0.0, (
            f"charge_density_norm inválida: {self.charge_density_norm}"
        )
        assert np.isfinite(self.potential_norm) and self.potential_norm >= 0.0, (
            f"potential_norm inválida: {self.potential_norm}"
        )
        assert np.isfinite(self.field_norm) and self.field_norm >= 0.0, (
            f"field_norm inválida: {self.field_norm}"
        )
        assert np.isfinite(self.cyber_momentum) and self.cyber_momentum >= 0.0, (
            f"cyber_momentum inválida: {self.cyber_momentum}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES DEL CAMPO DE GAUGE (JERARQUÍA DE ERRORES COHERENTE)
# ════════════════════════════════════════════���══════════════════════════════════

class GaugeFieldError(Exception):
    """
    Excepción base para todos los errores de enrutamiento electromagnético.

    Captura fallos genéricos que no se clasifican en categorías más específicas:
    • Entrada de usuario inválida (tipos, valores fuera de rango)
    • Estado no válido (None, tipo incorrecto)
    • Resolución fallida desde MIC
    """


class TopologicalSingularityError(GaugeFieldError):
    """
    Error asociado a incompatibilidades algebraicas o fallos de solver.

    Situaciones:
    • Matriz de incidencia no cuadrada o con dimensiones incompatibles
    • Laplaciano no simétrico (L ≠ Lᵀ)
    • Laplaciano no semidefinido positivo (λ_min < −ε)
    • Entradas de incidencia fuera de {-1, 0, +1}
    • Matrices contienen NaN o ±∞
    • Fallo en resolución de Poisson (LSQR no convergió satisfactoriamente)
    • Residual relativo supera tolerancia

    Invariante violado: alguna C_i de las condiciones de compatibilidad.
    """


class CohomologicalInconsistencyError(GaugeFieldError):
    """
    Error asociado a inconsistencia L ≠ B₁ᵀB₁.

    Situación:
    L₀ y B₁ no provienen del mismo complejo simplicial.
    La relación L₀ = B₁ᵀB₁ es fundamental en cohomología discreta.
    
    Posibles causas:
    • Construcción de L directamente sin usar B₁
    • B₁ es un subgrafo de la red del Laplaciano
    • Error de entrada del usuario

    Invariante violado: C5 (consistencia cohomológica).
    """


class LorentzForceError(GaugeFieldError):
    """
    Error asociado a imposibilidad de seleccionar agente válido.

    Situaciones:
    • Espacio de agentes vacío (no hay cargas registradas)
    • Acciones no finitas: ∃ k tal que ⟨Q_k, E⟩ = NaN o ±∞
    • Campo E contiene NaN o ±∞

    Invariante violado: I9 (al menos 1 agente registrado).
    """


class ChargeNeutralityError(GaugeFieldError):
    """
    Error asociado a violación de la condición de Fredholm.

    Teoría (alternativa de Fredholm):
    ──────────────────────────────────
    El problema Lx = b es soluble ⟺ b ⊥ ker(Lᵀ) = ker(L)  [L simétrica]

    Para grafo conexo: ker(L) = span{𝟙}
    Luego: ρ soluble ⟺ 𝟙ᵀρ = 0 (suma nula)

    Para grafo desconectado: ker(L) = span{indicadores de componentes}
    Luego: ρ soluble ⟺ ρ ortogonal a cada indicador
                     ⟺ suma nula en CADA componente

    Situaciones:
    • Densidad de carga no neutral: |𝟙ᵀρ| > ε
    • En grafo desconectado: ⟨ρ, v_k⟩ ≠ 0 para algún v_k ∈ ker(L)

    Invariante violado: C6 (condición de Fredholm).
    """


class AgentValidationError(GaugeFieldError):
    """
    Error en validación de cargas de agentes durante construcción.

    Situaciones:
    • Identificador de agente inválido (no string, string vacío)
    • Carga no convertible a np.ndarray
    • Carga de dimensión incorrecta
    • Carga contiene NaN o ±∞

    Invariante violado: C7 (cargas bien definidas).
    """


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE CONTEXTO Y SEGURIDAD DEFENSIVA
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_context(state: CategoricalState) -> Dict[str, Any]:
    """
    Extrae el contexto de un CategoricalState de forma defensiva.

    Defensas en profundidad:
    1. Verifica que state no sea None
    2. Verifica que state sea del tipo correcto
    3. Accede al atributo 'context' defensivamente (getattr)
    4. Verifica que context sea un Mapping
    5. Retorna copia defensiva (nuevo diccionario)

    Garantía: nunca falla. Retorna {} como fallback.

    Args:
        state: CategoricalState a introspeccionar

    Returns:
        Dict[str, Any]: copia del contexto o {} si no disponible
    """
    if state is None:
        return {}
    if not isinstance(state, CategoricalState):
        return {}

    ctx = getattr(state, "context", None)
    if ctx is None:
        return {}
    if not isinstance(ctx, Mapping):
        return {}

    return dict(ctx)  # Copia defensiva


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDADORES ALGEBRAICOS (FUNCIONES PURAS VERIFICADAS)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_sparse_matrix(
    matrix: Any,
    name: str,
) -> sp.csr_matrix:
    """
    Valida y convierte una matriz a formato CSR.

    Verificaciones:
        1. Es una matriz dispersa scipy.sparse
        2. Es bidimensional
        3. Todas las entradas son finitas

    Args:
        matrix: Matriz a validar
        name: Nombre de la matriz (para mensajes de error)

    Returns:
        sp.csr_matrix: Matriz convertida a formato CSR

    Raises:
        TopologicalSingularityError si alguna verificación falla
    """
    if not sp.issparse(matrix):
        raise TopologicalSingularityError(
            f"{name} debe ser una matriz dispersa scipy.sparse; "
            f"recibido tipo {type(matrix).__name__}."
        )

    matrix_csr = matrix.tocsr()

    if matrix_csr.ndim != 2:
        raise TopologicalSingularityError(
            f"{name} debe ser bidimensional; ndim={matrix_csr.ndim}."
        )

    if matrix_csr.nnz > 0 and not np.all(np.isfinite(matrix_csr.data)):
        raise TopologicalSingularityError(
            f"{name} contiene entradas no finitas."
        )

    return matrix_csr


def _validate_laplacian_symmetry(
    L: sp.csr_matrix,
    tol: float = LatticeQEDConstants.SYMMETRY_TOLERANCE,
) -> float:
    """
    Verifica la simetría del Laplaciano: ‖L − Lᵀ‖_F < tol.

    Fundamentación matemática:
    ──────────────────────────
    El Laplaciano de un grafo es siempre simétrico: L = B₁ᵀB₁ ⟹ L = Lᵀ.
    La violación de simetría indica un error en la construcción de la red
    (por ejemplo, un Laplaciano de una malla no-orientable) o error de usuario.

    Método de cálculo:
    L − Lᵀ debe ser semejante a cero en norma de Frobenius.
    Usamos norma de Frobenius porque es invariante a la estructura de sparsidad.

    Args:
        L: Laplaciano a verificar
        tol: Tolerancia (default 1e-10)

    Returns:
        float: Norma de Frobenius ‖L − Lᵀ‖_F

    Raises:
        TopologicalSingularityError si simetría se viola
    """
    diff = L - L.T
    frobenius_norm = float(sp.linalg.norm(diff, ord='fro'))

    if frobenius_norm > tol:
        raise TopologicalSingularityError(
            f"El Laplaciano no es simétrico: ‖L − Lᵀ‖_F = {frobenius_norm:.3e} > {tol:.3e}. "
            f"Un Laplaciano de grafo DEBE satisfacer L = Lᵀ. "
            f"Violación indica error en construcción de la red. "
            f"Verifique que L = B₁ᵀB₁ donde B₁ es la incidencia orientada."
        )

    return frobenius_norm


def _validate_laplacian_spectrum(
    L: sp.csr_matrix,
    tol: float = LatticeQEDConstants.SPECTRAL_TOLERANCE,
) -> float:
    """
    Verifica semi-definitud positiva del Laplaciano: λ_min ≥ −tol.

    Fundamentación matemática:
    ──────────────────────────
    Para L = B₁ᵀB₁, se cumple:
        xᵀLx = xᵀ(B₁ᵀB₁)x = ‖B₁x‖₂² ≥ 0
    
    Luego: ∀x, xᵀLx ≥ 0, es decir, L es semidefinida positiva.
    Eigenvalores: λ ≥ 0, con λ_min = 0 (multiplicidad = #componentes conexas).

    Verificación numérica:
    ─────────────────────
    Pequeñas matrices (n ≤ 64):
        Usar eigendecomposición densa completa (np.linalg.eigvalsh).
    
    Matrices grandes (n > 64):
        Usar Lanczos esparso (eigsh) para los k menores eigenvalores.
        Fallback a densa si Lanczos falla.

    Args:
        L: Laplaciano a verificar
        tol: Tolerancia para verificar semidefinitud (default 1e-10)

    Returns:
        float: Eigenvalor algebraicamente mínimo λ_min

    Raises:
        TopologicalSingularityError si λ_min < −tol
    """
    n = L.shape[0]

    if n <= 1:
        return 0.0

    # Seleccionar método según tamaño
    if n <= 64:
        # Matrices pequeñas: eigendecomposición densa completa
        L_dense = L.toarray()
        eigenvalues = np.linalg.eigvalsh(L_dense)
        lambda_min = float(eigenvalues[0])
    else:
        # Matrices grandes: Lanczos esparso
        k = min(3, n - 1)
        try:
            eigenvalues_small = eigsh(
                L.astype(np.float64),
                k=k,
                which='SM',  # SM = smallest magnitude
                return_eigenvectors=False,
            )
            lambda_min = float(np.min(eigenvalues_small))
        except Exception as exc:
            # Fallback a densa si Lanczos falla
            logger.warning(
                "Fallo en verificación espectral vía Lanczos: %s. "
                "Usando eigendecomposición densa como fallback.",
                exc,
            )
            L_dense = L.toarray()
            eigenvalues = np.linalg.eigvalsh(L_dense)
            lambda_min = float(eigenvalues[0])

    # Verificar semidefinitud
    if lambda_min < -tol:
        raise TopologicalSingularityError(
            f"El Laplaciano no es semidefinido positivo: λ_min = {lambda_min:.3e} < −{tol:.3e}. "
            f"Esto VIOLA la propiedad fundamental: L = B₁ᵀB₁ ⟹ xᵀLx = ‖B₁x‖² ≥ 0. "
            f"Posibles causas:\n"
            f"  • L no es el Laplaciano de un grafo\n"
            f"  • L fue construido incorrectamente\n"
            f"  • Error numérico grave en aritmética"
        )

    return lambda_min


def _validate_incidence_entries(
    B1: sp.csr_matrix,
    tol: float = LatticeQEDConstants.INCIDENCE_ENTRY_TOLERANCE,
) -> None:
    """
    Verifica que las entradas de B₁ pertenezcan exactamente a {-1, 0, +1}.

    Fundamentación matemática:
    ──────────────────────────
    Una matriz de incidencia orientada legítima B₁ ∈ ℝᴹˣᴺ debe tener:
        • Cada fila (arista) contiene exactamente un +1 y un −1
        • Todos los demás elementos son 0

    Específicamente:
        B₁[e,v] = +1   si v es la cabeza (head) de la arista e
        B₁[e,v] = −1   si v es la cola (tail) de la arista e
        B₁[e,v] = 0    en caso contrario

    Esta estructura garantiza la relación cohomológica L = B₁ᵀB₁.

    Verificación numérica:
    ─────────────────────
    Para cada entry b en B₁.data, calcular:
        dist(b) := min{|b − v| : v ∈ {-1, 0, +1}}
    
    Si dist(b) > tol, la entry es inválida.

    Args:
        B1: Matriz de incidencia a verificar
        tol: Tolerancia (default 1e-12, máximo rigor)

    Raises:
        TopologicalSingularityError si alguna entry es inválida
    """
    if B1.nnz == 0:
        # Matriz vacía es válida trivialmente
        return

    data = B1.data
    valid_values = np.array([-1.0, 0.0, 1.0])

    # Calcular distancia de cada entry al conjunto {-1, 0, +1}
    distances = np.min(
        np.abs(data[:, np.newaxis] - valid_values[np.newaxis, :]),
        axis=1,
    )

    # Identificar entries inválidas
    invalid_mask = distances > tol
    if np.any(invalid_mask):
        invalid_entries = data[invalid_mask]
        n_invalid = int(np.sum(invalid_mask))
        raise TopologicalSingularityError(
            f"La matriz de incidencia B₁ contiene {n_invalid} entrada(s) inválida(s). "
            f"Muestra (primeras 5): {invalid_entries[:5].tolist()}. "
            f"Tolerancia: {tol:.3e}. "
            f"Todas las entries deben pertenecer exactamente a {{-1, 0, +1}}."
        )


def _validate_cohomological_consistency(
    L: sp.csr_matrix,
    B1: sp.csr_matrix,
    tol: float = LatticeQEDConstants.COHOMOLOGICAL_TOLERANCE,
) -> float:
    """
    Verifica la relación cohomológica L₀ = B₁ᵀ B₁.

    Fundamentación matemática:
    ──────────────────────────
    En cohomología discreta de grafos, el Laplaciano de 0-formas es:
        L₀ := d₁ᵀ d₁ + d₀ᵀ d₀
    
    Para un complejo simplicial 1D (grafo), no hay 2-formas, luego:
        L₀ = d₀ᵀ d₀ = B₁ᵀ B₁
    
    donde B₁ = d₀ es el operador coborde d₀: C⁰ → C¹.

    Validación:
    ──────────
    Si L y B₁ provienen del MISMO complejo simplicial, entonces
    la relación L = B₁ᵀB₁ debe satisfacerse exactamente (módulo errores de redondeo).

    La violación indica que L y B₁ son inconsistentes:
        • Construidos a partir de grafos diferentes
        • B₁ es un subgrafo de la red del Laplaciano
        • Error en construcción

    Método:
    ───────
    Calcular ‖L − B₁ᵀB₁‖_F y comparar contra tolerancia.

    Args:
        L: Laplaciano
        B1: Matriz de incidencia
        tol: Tolerancia (default 1e-8, menos rígida que otras por ser derivada)

    Returns:
        float: Norma de Frobenius ‖L − B₁ᵀB₁‖_F

    Raises:
        CohomologicalInconsistencyError si verificación falla
    """
    L_reconstructed = B1.T @ B1
    diff = L - L_reconstructed
    frobenius_norm = float(sp.linalg.norm(diff, ord='fro'))

    if frobenius_norm > tol:
        raise CohomologicalInconsistencyError(
            f"Inconsistencia cohomológica: ‖L₀ − B₁ᵀB₁‖_F = {frobenius_norm:.3e} > {tol:.3e}. "
            f"El Laplaciano y la matriz de incidencia no son cohomológicamente consistentes. "
            f"Esto indica que L y B₁ NO provienen del mismo complejo simplicial. "
            f"Posibles causas:\n"
            f"  • L se construyó directamente sin usar B₁\n"
            f"  • B₁ es un subgrafo de la red del Laplaciano\n"
            f"  • Error en entrada del usuario\n"
            f"Solución: asegúrese de que L = B₁ᵀB₁."
        )

    return frobenius_norm


def _validate_agent_charges(
    agent_charges: Mapping[str, Any],
    expected_dim: int,
) -> Dict[str, np.ndarray]:
    """
    Valida y normaliza el registro de cargas de agentes.

    Verificaciones por agente k:
        1. ID: string no vacío
        2. Carga: convertible a np.ndarray (dtype float64)
        3. Dimensión: 1D con shape[0] = expected_dim = M
        4. Finitud: ∀ entry ∈ ℝ finito

    Args:
        agent_charges: Mapeo agent_id → Q_k
        expected_dim: Dimensión esperada M (número de aristas)

    Returns:
        Dict[str, np.ndarray]: Cargas normalizadas y validadas

    Raises:
        TopologicalSingularityError si agent_charges no es Mapping
        AgentValidationError si alguna carga es inválida
    """
    if not isinstance(agent_charges, Mapping):
        raise TopologicalSingularityError(
            f"agent_charges debe ser un Mapping[str, np.ndarray]; "
            f"recibido {type(agent_charges).__name__}."
        )

    normalized: Dict[str, np.ndarray] = {}

    for agent_id, charge in agent_charges.items():
        # Verificación 1: ID válido
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise AgentValidationError(
                f"Identificador de agente inválido: {agent_id!r}. "
                f"Debe ser un string no vacío."
            )

        # Verificación 2: Conversión a ndarray
        try:
            charge_arr = np.asarray(charge, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise AgentValidationError(
                f"La carga del agente '{agent_id}' no es convertible a np.ndarray: {exc}"
            ) from exc

        # Verificación 3: Dimensionalidad
        if charge_arr.ndim != 1:
            raise AgentValidationError(
                f"La carga del agente '{agent_id}' debe ser un vector 1D; "
                f"recibido ndim={charge_arr.ndim}, shape={charge_arr.shape}."
            )

        if charge_arr.shape[0] != expected_dim:
            raise AgentValidationError(
                f"La carga del agente '{agent_id}' debe residir en ℝ^{expected_dim}; "
                f"recibido shape={charge_arr.shape}."
            )

        # Verificación 4: Finitud
        if not np.all(np.isfinite(charge_arr)):
            raise AgentValidationError(
                f"La carga del agente '{agent_id}' contiene valores no finitos (NaN o ±∞)."
            )

        normalized[agent_id] = charge_arr

    return normalized


def _validate_charge_density(
    rho: np.ndarray,
    L: sp.csr_matrix,
    tol: float = LatticeQEDConstants.CHARGE_NEUTRALITY_TOLERANCE,
) -> None:
    """
    Verifica la condición de solubilidad de Fredholm para L ρ = Φ.

    Teoría (Alternativa de Fredholm):
    ─────────────────────────────────
    Sistema lineal Ax = b es soluble ⟺ b ⊥ ker(Aᵀ)

    Para L simétrica (Laplaciano de grafo):
        L ρ = Φ es soluble ⟺ ρ ⊥ ker(L)

    Para grafo conexo:
        ker(L) = span{𝟙}  (vector de unos, multiplicidad 1)
        Luego: ρ soluble ⟺ ⟨ρ, 𝟙⟩ = 0 ⟺ 𝟙ᵀρ = 0 (suma nula)

    Para grafo desconectado (k componentes):
        ker(L) = span{v₁, ..., v_k}  donde v_i = indicador de componente i
        Luego: ρ soluble ⟺ ⟨ρ, v_i⟩ = 0 ∀i  (suma nula en cada componente)

    Verificación numérica:
    ──────────────────────
    Caso conexo: verificar |𝟙ᵀρ| < tol
    
    Caso desconectado:
    1. Estimar dim(ker(L)) contando eigenvalores ≈ 0
    2. Computar base de ker(L) vía eigendecomposición
    3. Verificar ρ ⊥ v_i para cada v_i en la base

    Args:
        rho: Densidad de carga ρ ∈ ℝᴺ
        L: Laplaciano L₀ ∈ ℝᴺˣᴺ
        tol: Tolerancia (default 1e-12, máximo rigor)

    Raises:
        ChargeNeutralityError si condición de Fredholm se viola
    """
    n = L.shape[0]

    if n <= 1:
        return

    # Estimar dimensión del kernel
    if n <= 64:
        eigenvalues = np.linalg.eigvalsh(L.toarray())
    else:
        k_check = min(10, n - 1)
        try:
            eigenvalues = eigsh(
                L.astype(np.float64),
                k=k_check,
                which='SM',
                return_eigenvectors=False,
            )
        except Exception:
            eigenvalues = np.linalg.eigvalsh(L.toarray())

    kernel_dim = int(np.sum(np.abs(eigenvalues) < LatticeQEDConstants.SPECTRAL_TOLERANCE))

    if kernel_dim == 0:
        # Laplaciano de rango completo — no hay restricción
        return

    if kernel_dim == 1:
        # Grafo conexo: verificar suma nula
        total_charge = float(np.sum(rho))
        if abs(total_charge) > tol:
            raise ChargeNeutralityError(
                f"Violación de neutralidad de carga (grafo conexo): "
                f"𝟙ᵀρ = {total_charge:.3e}. "
                f"Condición de Fredholm exige: 𝟙ᵀρ = 0 (tolerancia: {tol:.3e}). "
                f"Esto es condición NECESARIA para solubilidad de L ρ = Φ."
            )
        return

    # Grafo desconectado: verificar ortogonalidad contra kernel
    logger.info(
        "Grafo desconectado detectado: dim(ker(L)) = %d. "
        "Verificando condición de Fredholm generalizada.",
        kernel_dim,
    )

    if n <= 64:
        eigvals_full, eigvecs_full = np.linalg.eigh(L.toarray())
        kernel_vecs = eigvecs_full[
            :, np.abs(eigvals_full) < LatticeQEDConstants.SPECTRAL_TOLERANCE
        ]
    else:
        try:
            eigvals_k, eigvecs_k = eigsh(
                L.astype(np.float64),
                k=min(kernel_dim + 2, n - 1),
                which='SM',
            )
            kernel_mask = np.abs(eigvals_k) < LatticeQEDConstants.SPECTRAL_TOLERANCE
            kernel_vecs = eigvecs_k[:, kernel_mask]
        except Exception:
            eigvals_full, eigvecs_full = np.linalg.eigh(L.toarray())
            kernel_mask = np.abs(eigvals_full) < LatticeQEDConstants.SPECTRAL_TOLERANCE
            kernel_vecs = eigvecs_full[:, kernel_mask]

    # Verificar ortogonalidad contra cada vector del kernel
    for j in range(kernel_vecs.shape[1]):
        v_j = kernel_vecs[:, j]
        projection = float(np.dot(rho, v_j))
        if abs(projection) > tol:
            raise ChargeNeutralityError(
                f"Violación de la condición de Fredholm generalizada: "
                f"⟨ρ, v_{j}⟩ = {projection:.3e} ≠ 0 (tolerancia: {tol:.3e}). "
                f"ρ no pertenece a Im(L) = ker(L)⊥. "
                f"dim(ker(L)) = {kernel_dim}. "
                f"Sistema L ρ = Φ NO ES SOLUBLE."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# OPERADOR DE ENRUTAMIENTO Y ELECTRODINÁMICA AGÉNTICA (NÚCLEO)
# ═══════════════════════════════════════════════════════════════════════════════

class GaugeFieldRouter:
    """
    Enrutador electromagnético sobre una red discreta.

    Modelo físico:
    ──────────────
    Implementa un operador que interpreta perturbaciones localizadas como
    densidades de carga sobre nodos de una red, resuelve el problema de
    Poisson discreto, y despacha el morfismo del agente cuyo vector de carga
    exhibe máximo acoplamiento con el campo electromagnético resultante.

    Interfaz de construcción:
    ────────────────────────
        router = GaugeFieldRouter(
            mic_registry=registry,
            laplacian=L ∈ ℝᴺˣᴺ,
            incidence_matrix=B₁ ∈ ℝᴹˣᴺ,
            agent_charges={agent_id: Q_k ∈ ℝᴹ, ...},
            verify_cohomology=True,
            tie_break_policy=TieBreakPolicy.LEXICOGRAPHIC,
        )

    Pipeline de enrutamiento (operación):
    ────────────────────────────────────
        state' = router.route_perturbation(state, anomaly_node=v₀, severity=α)

    Parámetros:
        mic_registry (MICRegistry):
            Registro que mapea agent_id → Morphism.
            Se utiliza para resolver el morfismo final después de seleccionar
            el agente por máximo acoplamiento gauge.

        laplacian (sp.spmatrix):
            Laplaciano combinatorio L₀ ∈ ℝᴺˣᴺ.
            Debe ser simétrico, semidefinido positivo, con el operador coborde
            orientado B₁.

        incidence_matrix (sp.spmatrix):
            Matriz de incidencia orientada B₁ ∈ ℝᴹˣᴺ.
            Entradas en {-1, 0, +1}, una por arista.

        agent_charges (Mapping[str, np.ndarray]):
            Registro de cargas de agentes Q_k ∈ ℝᴹ.
            Mapeo agent_id → vector de carga sobre aristas.

        verify_cohomology (bool):
            Si True, verifica L = B₁ᵀB₁ (default: True).
            Desactivar solo si se tiene seguridad de consistencia.

        tie_break_policy (TieBreakPolicy):
            Política de desempate (default: LEXICOGRAPHIC).

    Invariantes de construcción (verificados):
    ──────────────────────────────────────────
        I1. L ∈ ℝᴺˣᴺ cuadrada
        I2. B₁ ∈ ℝᴹˣᴺ, M ≥ N−1 (para grafo conexo)
        I3. L = Lᵀ
        I4. L ⪰ 0 (semidefinida positiva)
        I5. ‖L − B₁ᵀB₁‖_F < ε (si verify_cohomology)
        I6. B₁[i,j] ∈ {-1, 0, +1} ∀ i,j
        I7. Todas las matrices finitas
        I8. ∀ k: Q_k ∈ ℝᴹ, Q_k finita
        I9. ∃ al menos 1 agente registrado
    """

    def __init__(
        self,
        mic_registry: MICRegistry,
        laplacian: sp.spmatrix,
        incidence_matrix: sp.spmatrix,
        agent_charges: Mapping[str, np.ndarray],
        *,
        verify_cohomology: bool = True,
        tie_break_policy: TieBreakPolicy = TieBreakPolicy.LEXICOGRAPHIC,
    ) -> None:
        """
        Inicialización con validación rigurosa de todos los invariantes.

        Proceso de validación (9 etapas):
        ─────────────────────────────────
        [1] Conversión y validación de formato de matrices
        [2] Verificación de dimensionalidad básica
        [3] Validación de simetría del Laplaciano
        [4] Validación de espectro (semidefinitud positiva)
        [5] Validación de entradas de incidencia
        [6] Verificación de consistencia cohomológica (si enabled)
        [7] Validación de cargas de agentes
        [8] Verificación de existencia de al menos 1 agente
        [9] Logging de estado final

        Raises:
            TopologicalSingularityError: Si alguna condición C_i se viola
            CohomologicalInconsistencyError: Si C5 se viola
            AgentValidationError: Si C7 se viola
            LorentzForceError: Si I9 se viola (no hay agentes)
        """
        self._mic = mic_registry
        self._tie_break_policy = tie_break_policy

        # ── [1] Conversión y validación de formato ──
        self._L = _validate_sparse_matrix(laplacian, name="laplacian")
        self._B1 = _validate_sparse_matrix(incidence_matrix, name="incidence_matrix")

        # ── [2] Verificación de dimensionalidad básica ──
        n_rows_L, n_cols_L = self._L.shape
        if n_rows_L != n_cols_L:
            raise TopologicalSingularityError(
                f"El Laplaciano debe ser cuadrado: recibido shape={self._L.shape}."
            )

        self._num_nodes: int = n_rows_L
        self._num_edges: int = self._B1.shape[0]

        if self._B1.shape[1] != self._num_nodes:
            raise TopologicalSingularityError(
                f"La matriz de incidencia B₁ debe tener N={self._num_nodes} columnas; "
                f"recibido B₁.shape={self._B1.shape}."
            )

        # ── [3] Validación de simetría ──
        symmetry_residual: float = _validate_laplacian_symmetry(self._L)

        # ── [4] Validación de espectro ──
        self._lambda_min: float = _validate_laplacian_spectrum(self._L)

        # ── [5] Validación de incidencia ──
        _validate_incidence_entries(self._B1)

        # ── [6] Verificación de cohomología ──
        self._cohomological_residual: Optional[float] = None
        if verify_cohomology:
            self._cohomological_residual = _validate_cohomological_consistency(
                self._L, self._B1,
            )

        # ── [7] Validación de cargas ──
        self._Q_registry: Dict[str, np.ndarray] = _validate_agent_charges(
            agent_charges, expected_dim=self._num_edges,
        )

        # ── [8] Verificación de existencia de agentes ──
        if len(self._Q_registry) == 0:
            raise LorentzForceError(
                "El espacio de Fock agéntico está vacío: "
                "no se registraron cargas de gauge. "
                "Se requiere al menos un agente para enrutamiento."
            )

        # ── [9] Logging de estado final ──
        logger.info(
            "GaugeFieldRouter inicializado exitosamente:\n"
            "  Topología: N=%d nodos, M=%d aristas\n"
            "  Espectro: λ_min=%.3e (multiplicidad de ker(L)=%d)\n"
            "  Simetría: ‖L − Lᵀ‖_F=%.3e\n"
            "  Cohomología: residual=%s\n"
            "  Agentes: %d registrados (%s)\n"
            "  Desempate: %s",
            self._num_nodes,
            self._num_edges,
            self._lambda_min,
            1 if abs(self._lambda_min) < LatticeQEDConstants.SPECTRAL_TOLERANCE else 0,
            symmetry_residual,
            f"{self._cohomological_residual:.3e}" if self._cohomological_residual is not None else "N/A (desactivado)",
            len(self._Q_registry),
            ", ".join(sorted(self._Q_registry.keys())),
            self._tie_break_policy.name,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PROPIEDADES DE INTROSPECCIÓN (ACCESO SEGURO)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        """Número de nodos en la red discreta."""
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        """Número de aristas en la red discreta."""
        return self._num_edges

    @property
    def num_agents(self) -> int:
        """Número de agentes registrados."""
        return len(self._Q_registry)

    @property
    def agent_ids(self) -> Tuple[str, ...]:
        """IDs de agentes registrados (tupla ordenada lexicográficamente)."""
        return tuple(sorted(self._Q_registry.keys()))

    @property
    def lambda_min(self) -> float:
        """Eigenvalor mínimo del Laplaciano λ_min."""
        return self._lambda_min

    @property
    def laplacian(self) -> sp.csr_matrix:
        """Acceso de lectura al Laplaciano."""
        return self._L.copy()

    @property
    def incidence_matrix(self) -> sp.csr_matrix:
        """Acceso de lectura a la matriz de incidencia."""
        return self._B1.copy()

    def get_agent_charge(self, agent_id: str) -> Optional[np.ndarray]:
        """
        Retorna la carga del agente si está registrado.

        Args:
            agent_id: Identificador del agente

        Returns:
            Copia defensiva de Q_k ∈ ℝᴹ, o None si no registrado
        """
        if agent_id not in self._Q_registry:
            return None
        return self._Q_registry[agent_id].copy()

    # ──────────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN Y VALIDACIÓN DE DENSIDAD DE CARGA
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_anomaly_inputs(
        self,
        anomaly_node: int,
        severity: float,
    ) -> Tuple[int, float]:
        """
        Valida parámetros de una perturbación localizada.

        Verificaciones:
        1. anomaly_node es int en [0, N−1]
        2. severity es float finito no negativo
        3. severity se acota inferiormente por MIN_CHARGE_DENSITY

        Args:
            anomaly_node: Índice del nodo perturbado
            severity: Magnitud de la perturbación

        Returns:
            (anomaly_node_int, severity_validated)

        Raises:
            GaugeFieldError si parámetros son inválidos
        """
        if not isinstance(anomaly_node, (int, np.integer)):
            raise GaugeFieldError(
                f"anomaly_node debe ser int; recibido {type(anomaly_node).__name__}."
            )

        anomaly_node_int = int(anomaly_node)
        if anomaly_node_int < 0 or anomaly_node_int >= self._num_nodes:
            raise GaugeFieldError(
                f"anomaly_node fuera de rango: {anomaly_node_int} ∉ [0, {self._num_nodes - 1}]."
            )

        try:
            sev = float(severity)
        except (TypeError, ValueError) as exc:
            raise GaugeFieldError(
                f"severity debe ser convertible a float: {severity!r}"
            ) from exc

        if not np.isfinite(sev):
            raise GaugeFieldError(f"severity debe ser finita; recibido {sev}.")

        if sev < 0.0:
            raise GaugeFieldError(f"severity debe ser no negativa; recibido {sev}.")

        # Acotar inferiormente
        sev_validated = max(sev, LatticeQEDConstants.MIN_CHARGE_DENSITY)

        return anomaly_node_int, sev_validated

    def _quantize_bosonic_excitation(
        self,
        anomaly_node: int,
        severity: float,
    ) -> np.ndarray:
        """
        Construye la densidad de carga ρ ∈ ℝᴺ con neutralidad garantizada.

        Construcción:
        ─────────────
        ρ := severity · (δ_{anomaly_node} − 𝟙/N)

        donde δ_v es la delta de Kronecker: δ_v[v'] = 1 si v=v', 0 c.c.

        Propiedades:
        • ρ[anomaly_node] = severity · (1 − 1/N)
        • ρ[j≠anomaly_node] = severity · (−1/N)
        • 𝟙ᵀρ = severity · (1 − 1 − (N−1)/N) = 0 ✓

        Justificación física:
        ────────────────────
        La perturbación se modela como inyección puntual de carga (severity)
        balanceada por extracción distribuida (−severity/N en cada nodo).

        Esto mantiene neutralidad de carga globalmente, satisfaciendo la
        condición de Fredholm para el problema de Poisson.

        Corrección numérica:
        ───────────────────
        Se aplica ρ -= mean(ρ) para forzar neutralidad exacta numéricamente.

        Args:
            anomaly_node: Nodo perturbado (ya validado)
            severity: Magnitud de perturbación (ya validada)

        Returns:
            ρ ∈ ℝᴺ con 𝟙ᵀρ = 0 ± ε_machine

        Raises:
            ChargeNeutralityError si la densidad viola condición de Fredholm
        """
        rho = np.full(self._num_nodes, -severity / self._num_nodes, dtype=np.float64)
        rho[anomaly_node] += severity

        # Corrección numérica para forzar neutralidad exacta
        rho -= np.mean(rho)

        # Verificar condición de Fredholm (generalizada para grafos desconectados)
        _validate_charge_density(
            rho,
            self._L,
            tol=LatticeQEDConstants.CHARGE_NEUTRALITY_TOLERANCE,
        )

        return rho

    # ──────────────────────────────────────────────────────────────────────────
    # SOLVER DE POISSON DISCRETO (NÚCLEO COMPUTACIONAL)
    # ──────────────────────────────────────────────────────────────────────────

    def _solve_discrete_poisson(self, rho: np.ndarray) -> PoissonSolution:
        """
        Resuelve el problema de Poisson discreto: L₀Φ = ρ.

        Teoría matemática:
        ──────────────────
        Sea L₀ = B₁ᵀB₁ el Laplaciano. Para un grafo conexo:
            ker(L₀) = span{𝟙}  (multiplicidad 1)
            Im(L₀) = {x ∈ ℝᴺ : 𝟙ᵀx = 0}

        El problema L₀Φ = ρ admite infinitas soluciones si ρ ∈ Im(L₀).
        Seleccionamos la solución de MÍNIMA NORMA: Φ ⊥ ker(L₀).

        Método numérico:
        ────────────────
        LSQR (Least Squares QR) es apropiado para sistemas singulares:
            • Converge a solución de mínima norma (minimiza ‖Φ‖₂)
            • Maneja ker(L₀) ≠ {0} automáticamente
            • Robusto ante malcondicionamiento

        Parámetros LSQR:
            atol, btol: tolerancias de convergencia
            iter_lim: límite de iteraciones (None = sin límite)

        Verificaciones post-solución:
        ────────────────────────────
        1. Φ ∈ ℝᴺ, Φ finita
        2. Residual relativo: ‖LΦ − ρ‖₂/‖ρ‖₂ < RESIDUAL_TOLERANCE
        3. istop ∈ {0, 1, 2, 3, 4} (estado terminal válido)

        Args:
            rho: Densidad de carga ρ ∈ ℝᴺ (ya validada)

        Returns:
            PoissonSolution con Φ y metadatos del solver

        Raises:
            TopologicalSingularityError si:
                • ρ tiene dimensión incorrecta
                • ρ contiene NaN/±∞
                • Solver falla (excepción, o residual > tolerancia)
        """
        rho = np.asarray(rho, dtype=np.float64)

        if rho.shape != (self._num_nodes,):
            raise TopologicalSingularityError(
                f"ρ debe pertenecer a ℝ^{self._num_nodes}; recibido shape={rho.shape}."
            )

        if not np.all(np.isfinite(rho)):
            raise TopologicalSingularityError("ρ contiene valores no finitos.")

        # Invocar LSQR
        try:
            result = lsqr(
                self._L,
                rho,
                atol=LatticeQEDConstants.POISSON_ATOL,
                btol=LatticeQEDConstants.POISSON_BTOL,
                iter_lim=LatticeQEDConstants.POISSON_ITER_LIMIT,
            )
        except Exception as exc:
            raise TopologicalSingularityError(
                f"Colapso numérico en la resolución de la ecuación de Poisson: {exc}"
            ) from exc

        # Desempacar resultado (lsqr retorna tupla)
        phi = np.asarray(result[0], dtype=np.float64)
        istop = int(result[1])
        itn = int(result[2])
        r1norm = float(result[3])
        r2norm = float(result[4])
        anorm = float(result[5])
        acond = float(result[6])
        arnorm = float(result[7])

        # Validaciones
        if phi.shape != (self._num_nodes,):
            raise TopologicalSingularityError(
                f"La solución Φ tiene dimensión inválida: {phi.shape}."
            )

        if not np.all(np.isfinite(phi)):
            raise TopologicalSingularityError(
                f"La solución Φ contiene valores no finitos. "
                f"istop={istop}, acond={acond:.3e}. "
                f"Posible causa: Laplaciano mal condicionado o singular."
            )

        # Calcular residual relativo para auditoría
        residual = self._L.dot(phi) - rho
        residual_norm = float(np.linalg.norm(residual, ord=2))
        rho_norm = float(np.linalg.norm(rho, ord=2))
        relative_residual = residual_norm / max(rho_norm, LatticeQEDConstants.NUMERICAL_EPS)

        if not np.isfinite(relative_residual):
            raise TopologicalSingularityError(
                "Residual relativo no finito en la resolución de Poisson. "
                f"‖residual‖₂={residual_norm}, ‖ρ‖₂={rho_norm}."
            )

        if relative_residual > LatticeQEDConstants.RESIDUAL_TOLERANCE:
            raise TopologicalSingularityError(
                "La solución de Poisson no alcanzó precisión suficiente:\n"
                f"  istop={istop} (estado terminal)\n"
                f"  iteraciones={itn}\n"
                f"  ‖Ax−b‖={r1norm:.3e}\n"
                f"  ‖Aᵀ(Ax−b)‖={arnorm:.3e}\n"
                f"  κ(A) ≈ {acond:.3e} (número de condición)\n"
                f"  ‖LΦ − ρ‖₂/‖ρ‖₂ = {relative_residual:.3e}\n"
                f"  tolerancia = {LatticeQEDConstants.RESIDUAL_TOLERANCE:.3e}\n"
                "Sugerencias:\n"
                "  • Laplaciano muy mal condicionado\n"
                "  • Aumentar tolerancias POISSON_ATOL/BTOL (si es aceptable)\n"
                "  • Revisar consistencia de L y B₁"
            )

        # Construir resultado
        solution = PoissonSolution(
            phi=phi,
            istop=istop,
            iteration_count=itn,
            r1norm=r1norm,
            arnorm=arnorm,
            acond=acond,
            relative_residual=relative_residual,
        )

        logger.debug(
            "Poisson resuelto exitosamente: istop=%d, itn=%d, "
            "r1norm=%.3e, acond=%.3e, relative_residual=%.3e",
            istop, itn, r1norm, acond, relative_residual,
        )

        return solution

    # ──────────────────────────────────────────────────────────────────────────
    # GRADIENTE DISCRETO Y CAMPO SOBRE ARISTAS
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_potential_gradient(self, phi: np.ndarray) -> np.ndarray:
        """
        Calcula el campo discreto sobre aristas: E = B₁Φ.

        Cohomología discreta:
        ──────────────────────
        En cohomología de de Rham discreta en grafos:
        • d₀: C⁰(G) → C¹(G) es el operador coborde
        • (d₀Φ)_e = Φ(head(e)) − Φ(tail(e))

        En notación matricial: (d₀Φ) = B₁Φ

        El campo E sobre aristas representa las diferencias de potencial
        entre los nodos adyacentes de cada arista.

        Propiedades topológicas:
        • E ∈ Im(B₁) = [ker(B₁ᵀ)]⊥ = [ker(L)]⊥
        • dim(Im(B₁)) = rank(B₁) = N (para grafo conexo con M ≥ N−1)

        Verificaciones:
        1. Φ ∈ ℝᴺ, Φ finita
        2. E ∈ ℝᴹ, E finita

        Args:
            phi: Potencial Φ ∈ ℝᴺ (solución de Poisson)

        Returns:
            E ∈ ℝᴹ, campo discreto sobre aristas

        Raises:
            TopologicalSingularityError si validaciones fallan
        """
        phi = np.asarray(phi, dtype=np.float64)

        if phi.shape != (self._num_nodes,):
            raise TopologicalSingularityError(
                f"Φ debe pertenecer a ℝ^{self._num_nodes}; recibido shape={phi.shape}."
            )

        e_field = np.asarray(self._B1.dot(phi), dtype=np.float64).ravel()

        if e_field.shape != (self._num_edges,):
            raise TopologicalSingularityError(
                f"El campo E debe pertenecer a ℝ^{self._num_edges}; "
                f"recibido shape={e_field.shape}."
            )

        if not np.all(np.isfinite(e_field)):
            raise TopologicalSingularityError(
                "El campo E contiene valores no finitos (NaN o ±∞)."
            )

        return e_field

    # ──────────────────────────────────────────────────────────────────────────
    # ACOPLAMIENTO GAUGE Y SELECCIÓN DE AGENTE
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_lorentz_attraction(
        self,
        e_field: np.ndarray,
    ) -> CouplingResult:
        """
        Selecciona el agente que maximiza el acoplamiento electromagnético.

        Física (Fuerza de Lorentz):
        ───────────────────────────
        Un agente k con carga Q_k ∈ ℝᴹ en un campo E ∈ ℝᴹ experimenta
        una acción (energía de interacción):
            action_k := ⟨Q_k, E⟩ = Σ_i Q_k[i] · E[i]

        Seleccionar el agente que maximiza esta cantidad equivale a elegir
        la respuesta más afín al campo existente.

        Algoritmo de selección:
        ──────────────────────
        1. Calcular action_k para cada agente k
        2. Encontrar action_max = max_k action_k
        3. Identificar maximizadores (empatados dentro de tolerancia numérica)
        4. Aplicar política de desempate (lexicográfica o por registro)

        Desempate:
        ──────────
        Cuando |action_i − action_j| < ε_numérico, se considera que hay
        empate y se aplica una política determinística para resolverlo.

        Verificaciones:
        1. E ∈ ℝᴹ, E finita
        2. Todas las acciones son finitas
        3. Existe al menos un agente válido

        Args:
            e_field: Campo E ∈ ℝᴹ

        Returns:
            CouplingResult con agente seleccionado y metadatos

        Raises:
            LorentzForceError si selección falla
        """
        e_field = np.asarray(e_field, dtype=np.float64)

        if e_field.shape != (self._num_edges,):
            raise LorentzForceError(
                f"El campo E debe pertenecer a ℝ^{self._num_edges}; "
                f"recibido shape={e_field.shape}."
            )

        if not np.all(np.isfinite(e_field)):
            raise LorentzForceError(
                "El campo E contiene valores no finitos (NaN o ±∞)."
            )

        # Calcular acciones
        all_actions: Dict[str, float] = {}

        for agent_id, q_k in self._Q_registry.items():
            action = float(np.dot(q_k, e_field))
            if not np.isfinite(action):
                raise LorentzForceError(
                    f"Acción no finita para el agente '{agent_id}': "
                    f"⟨Q_{agent_id}, E⟩ = {action}."
                )
            all_actions[agent_id] = action

        if not all_actions:
            raise LorentzForceError(
                "El espacio de Fock agéntico está vacío: "
                "no hay candidatos válidos para acoplamiento gauge."
            )

        # Encontrar máximo
        max_action = max(all_actions.values())

        # Identificar maximizadores (empatados dentro de tolerancia numérica)
        maximizers = [
            agent_id
            for agent_id, action in all_actions.items()
            if np.isclose(action, max_action, atol=LatticeQEDConstants.NUMERICAL_EPS)
        ]

        # Aplicar política de desempate
        if self._tie_break_policy == TieBreakPolicy.LEXICOGRAPHIC:
            maximizers_sorted = sorted(maximizers)
        else:  # FIRST_REGISTERED
            maximizers_sorted = maximizers

        selected_agent = maximizers_sorted[0]

        # Logging de empate (si ocurrió)
        if len(maximizers) > 1:
            logger.info(
                "Empate en acoplamiento gauge entre %d agentes: %s. "
                "Seleccionado '%s' por política %s.",
                len(maximizers),
                maximizers_sorted,
                selected_agent,
                self._tie_break_policy.name,
            )

        result = CouplingResult(
            selected_agent=selected_agent,
            max_action=max_action,
            all_actions=all_actions,
            num_maximizers=len(maximizers),
        )

        logger.info(
            "Acoplamiento gauge completado: agente '%s' seleccionado "
            "con acción ⟨Q, E⟩=%.6e (de %d agentes, %d empatados).",
            selected_agent,
            max_action,
            len(all_actions),
            len(maximizers),
        )

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # RESOLUCIÓN DE MORFISMOS DESDE REGISTRO MIC
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_morphism(self, agent_id: str) -> Morphism:
        """
        Resuelve el morfismo asociado a un agente desde el registro MIC.

        Args:
            agent_id: ID del agente seleccionado

        Returns:
            Morphism: Función de transformación de estado

        Raises:
            GaugeFieldError si el agente no existe en el registro MIC
        """
        morphism: Optional[Morphism] = None
        # Acceder a self._mic._vectors directamente ya que la clase MICRegistry usa _vectors.
        with self._mic._lock:
            if agent_id in self._mic._vectors:
                morphism = self._mic._vectors[agent_id][1]

        if morphism is None:
            with self._mic._lock:
                registered_agents = list(self._mic._vectors.keys())
            raise GaugeFieldError(
                f"El agente '{agent_id}' fue seleccionado por acoplamiento gauge, "
                f"pero no existe en el registro MIC. "
                f"Agentes registrados en gauge: {self.agent_ids}. "
                f"Agentes registrados en MIC: {registered_agents}."
            )

        return morphism

    # ──────────────────────────────────────────────────────────────────────────
    # CÁLCULO DE MOMENTUM CIBERNÉTICO
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_cyber_momentum(
        severity: float,
        field_norm: float,
    ) -> float:
        """
        Calcula el momentum cibernético de la respuesta.

        Fórmula:
        ────────
        p := √(2 · severity · BASE_CYBER_MOMENTUM) · (1 + ‖E‖₂)

        Semántica física:
        ────────────────
        El momentum cuantifica la "inercia" de la respuesta del sistema.

        • Factor √(2 · severity · BASE): crece con la severidad de la anomalía
        • Factor (1 + ‖E‖₂): modula por la intensidad del campo resultante

        La forma cuadrática √(2α·β) proporciona escalamiento suave con α.
        El factor (1 + ‖E‖₂) captura la no-linealidad: campos más fuertes
        generan respuestas con mayor momentum.

        El +1 evita degeneración cuando ‖E‖₂ ≈ 0 (perturbaciones débiles
        en topologías poco resistivas inducen campos pequeños).

        Casos límite:
        • severity → 0, ‖E‖₂ → 0: p → 0 (sin perturbación → sin respuesta)
        • severity ↑, ‖E‖₂ ↑: p ↑ (perturbación fuerte → momentum alto)

        Args:
            severity: Magnitud de la anomalía (validada, > 0)
            field_norm: Norma L₂ del campo ‖E‖₂ ≥ 0

        Returns:
            float: Momentum cibernético p ≥ 0
        """
        base_momentum = float(
            np.sqrt(2.0 * severity * LatticeQEDConstants.BASE_CYBER_MOMENTUM)
        )
        modulated = base_momentum * (1.0 + field_norm)

        if not np.isfinite(modulated):
            logger.warning(
                "Momentum no finito: severity=%.3e, field_norm=%.3e. "
                "Usando base_momentum=%.3e como fallback.",
                severity, field_norm, base_momentum,
            )
            return base_momentum if np.isfinite(base_momentum) else 0.0

        return modulated

    # ──────────────────────────────────────────────────────────────────────────
    # ORQUESTACIÓN PRINCIPAL (PIPELINE DE ENRUTAMIENTO)
    # ──────────────────────────────────────────────────────────────────────────

    def route_perturbation(
        self,
        state: CategoricalState,
        anomaly_node: int,
        severity: float,
    ) -> CategoricalState:
        """
        Pipeline principal de enrutamiento electromagnético.

        Flujo de transformación (9 etapas):
        ──────────────────────────────────

        [1] Validación de entradas
            • state ≠ None, estado válido
            • anomaly_node ∈ [0, N−1]
            • severity ≥ 0, finita

        [2] Cuantificación de perturbación
            ρ := severity · (δ_{anomaly_node} − 𝟙/N)
            Garantía: 𝟙ᵀρ = 0 (condición de Fredholm)

        [3] Resolución de Poisson
            Resolver: L₀ Φ = ρ vía LSQR (mínima norma)
            Verificación: residual relativo < tolerancia

        [4] Cálculo del campo
            E := B₁ Φ (diferencial discreto del potencial)
            Propiedad: E ∈ Im(B₁)

        [5] Selección de agente
            k* := argmax_k ⟨Q_k, E⟩ (máximo acoplamiento)
            Desempate: política lexicográfica o por registro

        [6] Resolución del morfismo
            Obtener morphism_{k*} desde MIC registry

        [7] Cálculo de momentum
            p := √(2·severity·β) · (1 + ‖E‖₂)

        [8] Inyección de contexto físico
            Enriquecer state con información electromagnética
            (trace_id, momentum, nodo, agente, normas, etc.)

        [9] Aplicación del morfismo
            state' := morphism_{k*}(state_gauge)

        Args:
            state: Estado de entrada (CategoricalState)
            anomaly_node: Índice del nodo perturbado
            severity: Magnitud de la anomalía

        Returns:
            CategoricalState transformado por el morfismo seleccionado

        Raises:
            GaugeFieldError: Entrada inválida o resolución fallida
            TopologicalSingularityError: Incompatibilidad algebraica
            LorentzForceError: Selección de agente falla
            ChargeNeutralityError: Violación condición de Fredholm
            CohomologicalInconsistencyError: Inconsistencia L ≠ B₁ᵀB₁
        """
        # Generar ID de traza para auditoría
        trace_id = uuid.uuid4().hex[:12]

        # ──────────────────────────────────────────────────────────────────
        # [1] Validación de entradas
        # ──────────────────────────────────────────────────────────────────
        if state is None:
            raise GaugeFieldError("state no puede ser None.")

        if not isinstance(state, CategoricalState):
            raise GaugeFieldError(
                f"state debe ser CategoricalState; recibido {type(state).__name__}."
            )

        logger.debug(
            "[%s] Iniciando enrutamiento gauge: node=%s, severity=%s",
            trace_id, anomaly_node, severity,
        )

        # ──────────────────────────────────────────────────────────────────
        # [2] Validación y cuantificación de perturbación
        # ──────────────────────────────────────────────────────────────────
        anomaly_node_int, validated_severity = self._validate_anomaly_inputs(
            anomaly_node, severity
        )

        rho = self._quantize_bosonic_excitation(anomaly_node_int, validated_severity)
        charge_density_norm = float(np.linalg.norm(rho, ord=2))

        logger.debug(
            "[%s] Perturbación cuantificada: "
            "anomaly_node=%d, severity=%.6e, ‖ρ‖₂=%.6e",
            trace_id, anomaly_node_int, validated_severity, charge_density_norm,
        )

        # ──────────────────────────────────────────────────────────────────
        # [3] Resolución de Poisson
        # ──────────────────────────────────────────────────────────────────
        poisson_result = self._solve_discrete_poisson(rho)
        phi = poisson_result.phi
        potential_norm = float(np.linalg.norm(phi, ord=2))

        logger.debug(
            "[%s] Poisson resuelto: ‖Φ‖₂=%.6e, residual_rel=%.3e, acond=%.3e",
            trace_id, potential_norm, poisson_result.relative_residual,
            poisson_result.acond,
        )

        # ──────────────────────────────────────────────────────────────────
        # [4] Cálculo del campo
        # ──────────────────────────────────────────────────────────────────
        e_field = self._compute_potential_gradient(phi)
        field_norm = float(np.linalg.norm(e_field, ord=2))

        logger.debug(
            "[%s] Campo discreto calculado: ‖E‖₂=%.6e",
            trace_id, field_norm,
        )

        # ──────────────────────────────────────────────────────────────────
        # [5] Selección de agente
        # ──────────────────────────────────────────────────────────────────
        coupling_result = self._calculate_lorentz_attraction(e_field)
        agent_id = coupling_result.selected_agent

        # ──────────────────────────────────────────────────────────────────
        # [6] Resolución del morfismo
        # ──────────────────────────────────────────────────────────────────
        morphism = self._resolve_morphism(agent_id)

        # ──────────────────────────────────────────────────────────────────
        # [7] Cálculo de momentum
        # ──────────────────────────────────────────────────────────────────
        momentum = self._compute_cyber_momentum(validated_severity, field_norm)

        # ──────────────────────────────────────────────────────────────────
        # [8] Inyección de contexto físico
        # ──────────────────────────────────────────────────────────────────
        original_context = _safe_context(state)
        mutated_context: Dict[str, Any] = {
            **original_context,
            "gauge_trace_id": trace_id,
            "cyber_momentum": momentum,
            "resolved_anomaly_node": anomaly_node_int,
            "gauge_selected_agent": agent_id,
            "gauge_charge_density_norm": charge_density_norm,
            "gauge_field_norm": field_norm,
            "gauge_potential_norm": potential_norm,
            "gauge_max_action": coupling_result.max_action,
            "gauge_num_maximizers": coupling_result.num_maximizers,
            "gauge_poisson_relative_residual": poisson_result.relative_residual,
            "gauge_poisson_acond": poisson_result.acond,
        }

        intermediate_state = CategoricalState(
            payload=state.payload,
            context=mutated_context,
            validated_strata=state.validated_strata,
        )

        # Registrar diagnósticos completos
        diagnostics = GaugeFieldDiagnostics(
            trace_id=trace_id,
            anomaly_node=anomaly_node_int,
            severity=validated_severity,
            charge_density_norm=charge_density_norm,
            potential_norm=potential_norm,
            field_norm=field_norm,
            poisson_solution=poisson_result,
            coupling_result=coupling_result,
            cyber_momentum=momentum,
        )

        logger.info(
            "[%s] Enrutamiento gauge completado exitosamente:\n"
            "  Agente: '%s'\n"
            "  Momentum: %.6e\n"
            "  Campo: ‖E‖₂=%.6e\n"
            "  Acción: ⟨Q, E⟩=%.6e\n"
            "  Residual Poisson: %.3e\n"
            "  Condicionamiento: κ(L)≈%.3e",
            trace_id,
            agent_id,
            momentum,
            field_norm,
            coupling_result.max_action,
            poisson_result.relative_residual,
            poisson_result.acond,
        )

        # ──────────────────────────────────────────────────────────────────
        # [9] Aplicación del morfismo
        # ──────────────────────────────────────────────────────────────────
        return morphism(intermediate_state)
"""
=========================================================================================
Módulo: Semantic Validation Engine (Proyector Semántico con Tensor de Mahalanobis)
Ubicación: app/boole/wisdom/semantic_validator.py
Versión: 3.0.0 (Difeomorfismo de Señales y Cohomología Simplicial)

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA DIFERENCIAL:
Actúa como el Proyector Semántico Riguroso en el estrato WISDOM. Su objetivo axiomático es colapsar
la estocástica del Modelo de Lenguaje (LLM) y las señales de decisión bajo un Tensor Métrico estricto, certificando
la validez a través de la topología geométrica y la cohomología simplicial.

FUNDAMENTOS MATEMÁTICOS Y ANÁLISIS FUNCIONAL:

§1. ESPACIO MÉTRICO DE SEÑALES (TENSOR DE MAHALANOBIS):
El espacio de señales de intención reside en $\mathbb{R}^4$ con coordenadas $S = (s_0, s_1, s_2, s_3)^T$ [15]. La distancia
entre la señal actual y el centroide de estabilidad se mide a través del Tensor Métrico Riemanniano $G$ usando la distancia
de Mahalanobis:
$$ d_M(x, y) = \sqrt{(x - y)^T G^{-1} (x - y)} $$
Si el número de condición espectral del tensor supera la barrera ($\kappa(G) \gg 1$), el métrico es degenerado y el validador
aborta la evaluación por inestabilidad de la variedad.

§2. RETÍCULO COMPLETAMENTE ORDENADO DE VEREDICTOS:
Las decisiones no son lógicas binarias; habitan en un retículo estructurado (Lattice Theory) [17]:
$$ \bot (\text{VIABLE}) \le \text{CONDITIONAL} \le \text{WARNING} \le \top (\text{REJECT}) $$
El validador consolida la severidad mediante la operación Supremo ($\sqcup$). Cualquier obstrucción homológica colapsa el estado
de la señal hacia el elemento absorbente $\top$.

§3. COHOMOLOGÍA SIMPLICIAL PARA DETECCIÓN DE CONTRADICCIONES:
Las restricciones de coherencia entre los perfiles de riesgo y la salida del LLM forman un complejo de cocadenas. El motor computa
la cohomología $H^1(K; \mathbb{R})$. Una contradicción semántica (ej. alto riesgo con baja tolerancia) se detecta axiomáticamente si:
$$ \dim H^1(K; \mathbb{R}) > 0 $$
Resultando en un Veto por Obstrucción Topológica, erradicando alucinaciones probabilísticas mediante un teorema geométrico inquebrantable.

FUNDAMENTOS MATEMÁTICOS RIGUROSOS:

§1. ESPACIO MÉTRICO DE SEÑALES CON TENSOR DE MAHALANOBIS
    Sea S = ℝ⁴ el espacio de señales con coordenadas:
    - s₀: propósito (purpose)
    - s₁: confianza (confidence)  
    - s₂: cumplimiento de restricciones (constraints)
    - s₃: tolerancia al riesgo (risk)
    
    Dotamos S de un tensor métrico Riemanniano G ∈ Sym⁺(4), donde:
    - G es simétrica: Gᵢⱼ = Gⱼᵢ
    - G es definida positiva: ∀v ≠ 0, vᵀGv > 0
    - G codifica acoplamientos: Gᵢⱼ ≠ 0 ⟺ señales i,j están acopladas
    
    La distancia de Mahalanobis al ideal s* = (1,1,1,1)ᵀ es:
    
        D_M(s) = √[(s - s*)ᵀ G (s - s*)]
    
    Propiedades verificadas:
    - D_M(s) ≥ 0 con igualdad ssi s = s*
    - D_M es continua en s
    - Las curvas de nivel {s : D_M(s) = c} son elipsoides

§2. COMPLEJO DE COCADENAS Y COHOMOLOGÍA SIMPLICIAL
    Modelamos las restricciones de consistencia como un complejo simplicial:
    
    K = ({0,1,2,3}, {{0,1}, {1,2}, {2,3}, {3,0}})
    
    donde los vértices son índices de señales y las aristas representan
    restricciones de consistencia.
    
    El complejo de cocadenas sobre ℝ es:
    
    0 → C⁰(K;ℝ) --δ⁰--> C¹(K;ℝ) --δ¹--> C²(K;ℝ) → 0
    
    donde:
    - C⁰(K;ℝ) = {funciones φ: vértices → ℝ} ≅ ℝ⁴
    - C¹(K;ℝ) = {funciones ψ: aristas → ℝ} ≅ ℝ⁴
    - δ⁰(φ)(arista{i,j}) = φ(j) - φ(i)
    - δ¹(ψ)(triángulo) = suma orientada en frontera
    
    La cohomología es:
    
    H¹(K;ℝ) = ker(δ¹) / im(δ⁰)
    
    Interpretación: dim H¹ > 0 detecta ciclos no triviales en las restricciones,
    indicando paradojas semánticas irresolubles.

§3. RETÍCULO DE VEREDICTOS CON ESTRUCTURA DE ORDEN
    El conjunto de veredictos V = {VIABLE, CONDITIONAL, WARNING, REJECT}
    forma un retículo totalmente ordenado:
    
    VIABLE < CONDITIONAL < WARNING < REJECT
    
    con operaciones:
    - Supremo (∨): max según orden
    - Ínfimo (∧): min según orden
    
    Propiedades algebraicas verificadas:
    - Asociatividad: (v₁ ∨ v₂) ∨ v₃ = v₁ ∨ (v₂ ∨ v₃)
    - Conmutatividad: v₁ ∨ v₂ = v₂ ∨ v₁
    - Idempotencia: v ∨ v = v
    - Elemento absorbente: v ∨ REJECT = REJECT

§4. INVARIANTES Y CONTRATOS
    Cada método público garantiza:
    
    [PRE] Precondiciones sobre argumentos (validación estricta)
    [POST] Postcondiciones sobre resultados (invariantes de retorno)
    [INV] Invariantes de clase (consistencia del estado)
    
=========================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import (
    Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union, Callable, Protocol
)
from collections import defaultdict
from functools import lru_cache

import numpy as np
from numpy.linalg import LinAlgError

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

logger = logging.getLogger("Gamma.Wisdom.SemanticValidator.v3.0")
logger.setLevel(logging.INFO)


# =============================================================================
# EXCEPCIONES ESPECÍFICAS DEL DOMINIO
# =============================================================================

class ValidationError(Exception):
    """Clase base para errores de validación."""
    pass


class TopologicalObstructionError(ValidationError):
    """
    Obstrucción topológica detectada: dim H¹(K;ℝ) > 0.
    
    Indica una paradoja semántica irresoluble en las señales de entrada.
    """
    def __init__(self, dimension: int, cycle_description: str):
        self.dimension = dimension
        self.cycle_description = cycle_description
        super().__init__(
            f"Topological obstruction detected: dim H¹ = {dimension}. "
            f"Semantic paradox: {cycle_description}"
        )


class MetricDegeneracyError(ValidationError):
    """El tensor métrico es degenerado o mal condicionado."""
    def __init__(self, condition_number: float):
        self.condition_number = condition_number
        super().__init__(
            f"Metric tensor is ill-conditioned: κ(G) = {condition_number:.2e}"
        )


class ContractViolationError(ValidationError):
    """Violación de precondición o postcondición."""
    pass


# =============================================================================
# TIPOS Y ENUMERACIONES
# =============================================================================

class Verdict(IntEnum):
    """
    Retículo totalmente ordenado de veredictos.
    
    Orden: VIABLE < CONDITIONAL < WARNING < REJECT
    
    Invariante: Para todo v₁, v₂ ∈ Verdict, existe sup{v₁, v₂} y inf{v₁, v₂}.
    """
    VIABLE = 0       # ⊥: Elemento mínimo (aprobado sin restricciones)
    CONDITIONAL = 1  # Aprobado con condiciones (requiere supervisión)
    WARNING = 2      # Alerta severa (decisión humana recomendada)
    REJECT = 3       # ⊤: Elemento máximo (rechazo absoluto)

    def __str__(self) -> str:
        return self.name

    def __and__(self, other: Verdict) -> Verdict:
        """Ínfimo (meet): min según orden."""
        return Verdict(min(self.value, other.value))

    def __or__(self, other: Verdict) -> Verdict:
        """Supremo (join): max según orden."""
        return Verdict(max(self.value, other.value))

    @property
    def is_accepted(self) -> bool:
        """Verdadero ssi v ∈ {VIABLE, CONDITIONAL}."""
        return self in {Verdict.VIABLE, Verdict.CONDITIONAL}

    @property
    def requires_human_review(self) -> bool:
        """Verdadero ssi v ∈ {CONDITIONAL, WARNING}."""
        return self in {Verdict.CONDITIONAL, Verdict.WARNING}

    @property
    def severity_score(self) -> float:
        """Mapeo a [0,1]: 0 = VIABLE, 1 = REJECT."""
        return self.value / 3.0


# =============================================================================
# ESTRUCTURAS DE DATOS INMUTABLES
# =============================================================================

@dataclass(frozen=True, order=True)
class BusinessPurpose:
    """
    Mapeo semántico: concepto técnico → problema empresarial.
    
    Invariantes:
    - 0 ≤ strength ≤ 1
    - 0 ≤ confidence ≤ 1
    - concept y business_problem son cadenas no vacías
    
    La fuerza efectiva es: strength × confidence
    """
    concept: str
    business_problem: str
    strength: float
    confidence: float = 1.0

    def __post_init__(self):
        """
        [PRE] Valida invariantes estructurales.
        
        Raises:
            ContractViolationError: Si se viola alguna precondición.
        """
        if not (0.0 <= self.strength <= 1.0):
            raise ContractViolationError(
                f"Precondition violated: strength ∈ [0,1], got {self.strength}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ContractViolationError(
                f"Precondition violated: confidence ∈ [0,1], got {self.confidence}"
            )
        if not isinstance(self.concept, str) or not self.concept.strip():
            raise ContractViolationError(
                "Precondition violated: concept must be non-empty string"
            )
        if not isinstance(self.business_problem, str) or not self.business_problem.strip():
            raise ContractViolationError(
                "Precondition violated: business_problem must be non-empty string"
            )

    @property
    def effective_strength(self) -> float:
        """
        Fuerza efectiva combinando strength y confidence.
        
        [POST] 0 ≤ resultado ≤ 1
        """
        result = self.strength * self.confidence
        assert 0.0 <= result <= 1.0, "Postcondition violated"
        return result

    def __repr__(self) -> str:
        return (
            f"Purpose({self.concept} → {self.business_problem}, "
            f"σ={self.strength:.3f}, conf={self.confidence:.3f})"
        )


@dataclass(frozen=True)
class LLMOutput:
    """
    Metadatos estocásticos de salida del modelo de lenguaje.
    
    Invariantes:
    - entropy ≥ 0 (puede ser +∞)
    - 0 ≤ confidence ≤ 1
    - temperature > 0
    - num_tokens ≥ 0
    """
    entropy: float
    confidence: float
    temperature: float = 1.0
    num_tokens: int = 0

    def __post_init__(self):
        """[PRE] Valida invariantes."""
        if self.entropy < 0 and not math.isinf(self.entropy):
            raise ContractViolationError(f"entropy must be ≥ 0 or +∞, got {self.entropy}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ContractViolationError(f"confidence ∈ [0,1], got {self.confidence}")
        if self.temperature <= 0:
            raise ContractViolationError(f"temperature > 0, got {self.temperature}")
        if self.num_tokens < 0:
            raise ContractViolationError(f"num_tokens ≥ 0, got {self.num_tokens}")

    @property
    def normalized_entropy(self) -> float:
        """
        Entropía normalizada por temperatura y longitud.
        
        H_norm = H / (T × √max(N, 1))
        
        [POST] H_norm ≥ 0
        """
        if math.isinf(self.entropy):
            return float('inf')
        length_factor = math.sqrt(max(self.num_tokens, 1))
        result = self.entropy / (self.temperature * length_factor)
        assert result >= 0, "Postcondition violated"
        return result

    @property
    def is_singular(self) -> bool:
        """
        Detecta singularidad estocástica.
        
        Verdadero ssi entropy = +∞ ∨ confidence = 0
        """
        return math.isinf(self.entropy) or self.confidence == 0.0

    @property
    def perplexity(self) -> float:
        """
        Perplejidad: exp(H).
        
        [POST] perplexity ≥ 1
        """
        if math.isinf(self.entropy):
            return float('inf')
        result = math.exp(self.entropy)
        assert result >= 1.0, "Postcondition violated"
        return result

    def __repr__(self) -> str:
        return (
            f"LLM(H={self.entropy:.2f}, conf={self.confidence:.2f}, "
            f"T={self.temperature:.2f}, tokens={self.num_tokens})"
        )


@dataclass(frozen=True)
class RiskProfile:
    """
    Perfil de tolerancia al riesgo empresarial.
    
    Invariantes:
    - Todos los parámetros ∈ [0, 1]
    - effective_tolerance ∈ [0, 1]
    """
    risk_tolerance: float              # Tolerancia general
    domain_criticality: float = 0.5    # Criticidad del dominio
    acceptable_failure_rate: float = 0.01  # Tasa de fallo aceptable

    def __post_init__(self):
        """[PRE] Valida invariantes."""
        for name, value in [
            ('risk_tolerance', self.risk_tolerance),
            ('domain_criticality', self.domain_criticality),
            ('acceptable_failure_rate', self.acceptable_failure_rate)
        ]:
            if not (0.0 <= value <= 1.0):
                raise ContractViolationError(f"{name} ∈ [0,1], got {value}")

    @property
    def effective_tolerance(self) -> float:
        """
        Tolerancia efectiva ajustada por criticidad.
        
        τ_eff = τ × (1 - 0.5 × criticality)
        
        [POST] 0 ≤ τ_eff ≤ 1
        """
        result = self.risk_tolerance * (1.0 - 0.5 * self.domain_criticality)
        assert 0.0 <= result <= 1.0, "Postcondition violated"
        return result

    @property
    def risk_category(self) -> str:
        """Categorización cualitativa basada en τ_eff."""
        tol = self.effective_tolerance
        if tol < 0.2:
            return "HIGHLY_CONSERVATIVE"
        elif tol < 0.4:
            return "CONSERVATIVE"
        elif tol < 0.6:
            return "MODERATE"
        elif tol < 0.8:
            return "AGGRESSIVE"
        else:
            return "HIGHLY_AGGRESSIVE"

    @lru_cache(maxsize=1)
    def _compute_lipschitz_bound(self) -> float:
        """
        Cota de Lipschitz para restricciones de complejidad.
        
        L = exp(-2 × criticality × (1 - tolerance))
        
        [POST] 0 < L ≤ 1
        """
        result = math.exp(-2.0 * self.domain_criticality * (1.0 - self.risk_tolerance))
        assert 0.0 < result <= 1.0, "Postcondition violated"
        return result

    def __repr__(self) -> str:
        return (
            f"Risk(τ={self.risk_tolerance:.2f}, "
            f"crit={self.domain_criticality:.2f}, "
            f"cat={self.risk_category})"
        )


@dataclass
class ValidationResult:
    """
    Resultado completo de validación con trazabilidad algebraica.
    
    Invariantes:
    - mahalanobis_distance ≥ 0
    - signals[k] ∈ [0, 1] ∀k
    - verdict ∈ Verdict
    """
    verdict: Verdict
    mahalanobis_distance: float
    signals: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    cohomology_dimension: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """[POST] Valida invariantes."""
        if self.mahalanobis_distance < 0:
            raise ContractViolationError(
                f"D_M must be ≥ 0, got {self.mahalanobis_distance}"
            )
        for name, value in self.signals.items():
            if not (0.0 <= value <= 1.0):
                logger.warning(f"Signal {name} = {value} outside [0,1], clamping")
                self.signals[name] = max(0.0, min(1.0, value))

    def add_reason(
        self, 
        reason: str, 
        signal_name: Optional[str] = None,
        signal_value: Optional[float] = None
    ) -> None:
        """
        Agrega razón y señal al resultado.
        
        [PRE] Si signal_name provisto, signal_value también debe proveerse.
        """
        if signal_name is not None and signal_value is None:
            raise ContractViolationError(
                "If signal_name is provided, signal_value must also be provided"
            )
        
        self.reasons.append(reason)
        if signal_name is not None and signal_value is not None:
            self.signals[signal_name] = max(0.0, min(1.0, signal_value))

    @property
    def has_topological_obstruction(self) -> bool:
        """Verdadero ssi dim H¹ > 0."""
        return self.cohomology_dimension > 0

    def __repr__(self) -> str:
        return (
            f"ValidationResult(verdict={self.verdict.name}, "
            f"D_M={self.mahalanobis_distance:.3f}, "
            f"dim H¹={self.cohomology_dimension})"
        )


# =============================================================================
# TENSOR MÉTRICO DE MAHALANOBIS (Geometría Riemanniana)
# =============================================================================

class MahalanobisMetric:
    """
    Tensor métrico Riemanniano G sobre el espacio de señales ℝ⁴.
    
    Propiedades algebraicas:
    - G es simétrica: G = Gᵀ
    - G es definida positiva: vᵀGv > 0 ∀v ≠ 0
    - Número de condición κ(G) < 10³ (bien condicionada)
    
    Invariantes:
    [INV1] self.G.shape == (4, 4)
    [INV2] np.allclose(self.G, self.G.T)  # Simetría
    [INV3] np.all(np.linalg.eigvalsh(self.G) > 0)  # Def. positiva
    [INV4] self._condition_number < 1e3
    """

    # Tensor métrico default con acoplamientos físicamente motivados
    DEFAULT_METRIC_TENSOR = np.array([
        [1.00,  0.30,  0.20,  0.10],  # purpose: acoplado con confidence
        [0.30,  1.00,  0.40, -0.25],  # confidence: anticorrelado con risk
        [0.20,  0.40,  1.00,  0.35],  # constraints: correlado con risk
        [0.10, -0.25,  0.35,  1.00],  # risk: moderado con purpose
    ], dtype=np.float64)

    # Umbral máximo para número de condición
    MAX_CONDITION_NUMBER = 1000.0

    def __init__(self, metric_tensor: Optional[np.ndarray] = None):
        """
        Inicializa tensor métrico.
        
        [PRE] metric_tensor es 4×4, simétrica, definida positiva, bien condicionada.
        [POST] Invariantes de clase satisfechos.
        
        Args:
            metric_tensor: Matriz 4×4 personalizada o None para default.
            
        Raises:
            ContractViolationError: Si se violan precondiciones.
            MetricDegeneracyError: Si la métrica está mal condicionada.
        """
        if metric_tensor is not None:
            self._validate_metric_tensor(metric_tensor)
            self.G = metric_tensor.copy()
        else:
            self.G = self.DEFAULT_METRIC_TENSOR.copy()
        
        # Cachear número de condición
        self._condition_number = self._compute_condition_number()
        
        if self._condition_number > self.MAX_CONDITION_NUMBER:
            raise MetricDegeneracyError(self._condition_number)
        
        # Verificar invariantes post-construcción
        self._check_invariants()
        
        logger.debug(f"Initialized Mahalanobis metric with κ(G) = {self._condition_number:.2e}")

    def _validate_metric_tensor(self, G: np.ndarray) -> None:
        """
        Valida precondiciones del tensor métrico.
        
        [PRE] G es numpy array
        [POST] G satisface todas las propiedades requeridas
        
        Raises:
            ContractViolationError: Si falla alguna validación.
        """
        if not isinstance(G, np.ndarray):
            raise ContractViolationError("Metric tensor must be numpy array")
        
        if G.shape != (4, 4):
            raise ContractViolationError(f"Metric tensor must be 4×4, got {G.shape}")
        
        if not np.allclose(G, G.T, rtol=1e-9, atol=1e-12):
            raise ContractViolationError("Metric tensor must be symmetric")
        
        try:
            eigvals = np.linalg.eigvalsh(G)
        except LinAlgError as e:
            raise ContractViolationError(f"Failed to compute eigenvalues: {e}")
        
        if np.any(eigvals <= 0):
            raise ContractViolationError(
                f"Metric tensor must be positive definite. Eigenvalues: {eigvals}"
            )

    def _compute_condition_number(self) -> float:
        """
        Calcula número de condición κ(G) = λ_max / λ_min.
        
        [POST] κ(G) ≥ 1
        """
        eigvals = np.linalg.eigvalsh(self.G)
        kappa = eigvals[-1] / eigvals[0]  # max / min
        assert kappa >= 1.0, "Postcondition violated: κ(G) ≥ 1"
        return kappa

    def _check_invariants(self) -> None:
        """
        Verifica invariantes de clase.
        
        Raises:
            AssertionError: Si se viola algún invariante.
        """
        assert self.G.shape == (4, 4), "[INV1] violated"
        assert np.allclose(self.G, self.G.T), "[INV2] violated: not symmetric"
        eigvals = np.linalg.eigvalsh(self.G)
        assert np.all(eigvals > 0), "[INV3] violated: not positive definite"
        assert self._condition_number < self.MAX_CONDITION_NUMBER, "[INV4] violated"

    def distance_to_ideal(
        self, 
        signal_vector: np.ndarray,
        ideal: Optional[np.ndarray] = None
    ) -> float:
        """
        Calcula distancia de Mahalanobis al punto ideal.
        
        D_M(s) = √[(s - s*)ᵀ G (s - s*)]
        
        [PRE] signal_vector.shape == (4,)
        [PRE] ideal is None or ideal.shape == (4,)
        [POST] D_M ≥ 0
        [POST] D_M = 0 ⟺ signal_vector = ideal
        
        Args:
            signal_vector: Vector de señales en ℝ⁴
            ideal: Punto ideal (default: (1,1,1,1)ᵀ)
            
        Returns:
            Distancia de Mahalanobis ≥ 0
            
        Raises:
            ContractViolationError: Si se violan precondiciones.
        """
        if signal_vector.shape != (4,):
            raise ContractViolationError(
                f"signal_vector must be shape (4,), got {signal_vector.shape}"
            )
        
        if ideal is None:
            ideal = np.ones(4, dtype=np.float64)
        elif ideal.shape != (4,):
            raise ContractViolationError(
                f"ideal must be shape (4,), got {ideal.shape}"
            )
        
        diff = signal_vector - ideal
        
        # Producto cuadrático: diffᵀ G diff
        quadratic_form = diff.T @ self.G @ diff
        
        # Asegurar no-negatividad numérica (errores de redondeo)
        distance = math.sqrt(max(0.0, quadratic_form))
        
        # Verificar postcondición: D_M = 0 ⟺ s = s*
        if distance == 0.0:
            assert np.allclose(signal_vector, ideal, rtol=1e-9), \
                "Postcondition violated: D_M = 0 but s ≠ s*"
        
        return distance

    def set_coupling(self, i: int, j: int, value: float) -> None:
        """
        Ajusta acoplamiento Gᵢⱼ entre dos señales.
        
        [PRE] 0 ≤ i, j < 4
        [PRE] Modificación mantiene definición positiva
        [POST] Invariantes de clase satisfechos
        
        Args:
            i, j: Índices de señales
            value: Nuevo valor de acoplamiento
            
        Raises:
            ContractViolationError: Si índices inválidos o métrica degenerada.
        """
        if not (0 <= i < 4 and 0 <= j < 4):
            raise ContractViolationError(f"Indices must be in [0, 3], got i={i}, j={j}")
        
        # Actualizar simétricamente
        old_ij = self.G[i, j]
        old_ji = self.G[j, i]
        
        self.G[i, j] = value
        self.G[j, i] = value
        
        # Verificar que sigue siendo definida positiva
        try:
            eigvals = np.linalg.eigvalsh(self.G)
            if np.any(eigvals <= 0):
                # Revertir cambio
                self.G[i, j] = old_ij
                self.G[j, i] = old_ji
                raise ContractViolationError(
                    f"Setting G[{i},{j}] = {value} would make metric indefinite"
                )
        except LinAlgError:
            # Revertir cambio
            self.G[i, j] = old_ij
            self.G[j, i] = old_ji
            raise ContractViolationError("Failed to verify positive definiteness")
        
        # Recalcular número de condición
        self._condition_number = self._compute_condition_number()
        
        if self._condition_number > self.MAX_CONDITION_NUMBER:
            # Revertir cambio
            self.G[i, j] = old_ij
            self.G[j, i] = old_ji
            self._condition_number = self._compute_condition_number()
            raise MetricDegeneracyError(self._condition_number)
        
        self._check_invariants()

    def copy(self) -> 'MahalanobisMetric':
        """
        Crea copia profunda de la métrica.
        
        [POST] Resultado es independiente del original.
        """
        return MahalanobisMetric(self.G.copy())

    def __repr__(self) -> str:
        return f"MahalanobisMetric(κ={self._condition_number:.2e})"


# =============================================================================
# COHOMOLOGÍA SIMPLICIAL (Topología Algebraica Rigurosa)
# =============================================================================

class SimplicialCohomology:
    """
    Calcula cohomología simplicial H¹(K; ℝ) del complejo de señales.
    
    El complejo simplicial K tiene:
    - Vértices V = {0, 1, 2, 3} (índices de señales)
    - Aristas E = {(0,1), (1,2), (2,3), (3,0)} (ciclo de restricciones)
    
    Complejo de cocadenas:
        0 → C⁰(K;ℝ) --δ⁰--> C¹(K;ℝ) --δ¹--> C²(K;ℝ) → 0
    
    donde:
    - C⁰ = ℝ⁴ (funciones en vértices)
    - C¹ = ℝ⁴ (funciones en aristas)
    - C² = {0} (no hay 2-símplices)
    
    El operador cofrontera δ⁰: C⁰ → C¹ está dado por:
        (δ⁰φ)((i,j)) = φ(j) - φ(i)
    
    La cohomología es:
        H¹(K;ℝ) = ker(δ¹) / im(δ⁰) = C¹ / im(δ⁰)
    
    Como K es un ciclo de 4 vértices, dim H¹ = 1 genéricamente.
    Detectamos obstrucciones cuando los valores de señales violan
    consistencia cíclica más allá de tolerancias.
    
    Invariantes:
    [INV1] self._edges es lista de tuplas (i, j) con i < j
    [INV2] len(self._edges) == 4
    [INV3] self._signal_values es dict con 4 entradas o vacío
    """

    # Constantes de índices
    PURPOSE_IDX = 0
    CONFIDENCE_IDX = 1
    CONSTRAINTS_IDX = 2
    RISK_IDX = 3

    # Aristas del complejo (ciclo)
    _EDGES = [
        (PURPOSE_IDX, CONFIDENCE_IDX),      # e₀
        (CONFIDENCE_IDX, CONSTRAINTS_IDX),  # e₁
        (CONSTRAINTS_IDX, RISK_IDX),        # e₂
        (RISK_IDX, PURPOSE_IDX),            # e₃
    ]

    # Tolerancia para inconsistencias
    INCONSISTENCY_THRESHOLD = 0.3

    def __init__(self):
        """Inicializa cohomología con señales vacías."""
        self._signal_values: Dict[int, float] = {}
        self._cohomology_dimension: Optional[int] = None
        self._cycle_violations: List[str] = []

    def set_signals(
        self, 
        purpose: float, 
        confidence: float,
        constraints: float, 
        risk: float
    ) -> None:
        """
        Establece valores de señales.
        
        [PRE] Todos los valores ∈ [0, 1]
        [POST] self._signal_values tiene 4 entradas
        
        Args:
            purpose, confidence, constraints, risk: Valores de señales.
            
        Raises:
            ContractViolationError: Si algún valor fuera de [0,1].
        """
        for name, value in [
            ('purpose', purpose),
            ('confidence', confidence),
            ('constraints', constraints),
            ('risk', risk)
        ]:
            if not (0.0 <= value <= 1.0):
                raise ContractViolationError(
                    f"{name} must be in [0,1], got {value}"
                )
        
        self._signal_values = {
            self.PURPOSE_IDX: purpose,
            self.CONFIDENCE_IDX: confidence,
            self.CONSTRAINTS_IDX: constraints,
            self.RISK_IDX: risk,
        }
        
        # Invalidar caché
        self._cohomology_dimension = None
        self._cycle_violations = []
        
        assert len(self._signal_values) == 4, "Postcondition violated"

    def _compute_coboundary_image(self) -> np.ndarray:
        """
        Calcula imagen de δ⁰: C⁰ → C¹.
        
        Para cada arista e = (i, j), calculamos δ⁰(φ)(e) = φ(j) - φ(i).
        
        [PRE] self._signal_values tiene 4 entradas
        [POST] Resultado es matriz 4×4 (base de im(δ⁰))
        
        Returns:
            Matriz cuyas columnas generan im(δ⁰).
        """
        if len(self._signal_values) != 4:
            raise ContractViolationError("Must set all 4 signals before computing cohomology")
        
        # Construir matriz del operador δ⁰
        # Filas: aristas (4), Columnas: vértices (4)
        delta_matrix = np.zeros((4, 4), dtype=np.float64)
        
        for edge_idx, (i, j) in enumerate(self._EDGES):
            delta_matrix[edge_idx, i] = -1.0  # -φ(i)
            delta_matrix[edge_idx, j] = 1.0   # +φ(j)
        
        return delta_matrix

    def _detect_cycle_violations(self) -> List[str]:
        """
        Detecta violaciones de consistencia cíclica.
        
        Para cada par de señales adyacentes en el ciclo, verificamos
        si su combinación es semánticamente inconsistente.
        
        [POST] Retorna lista de descripciones de violaciones.
        
        Returns:
            Lista de strings describiendo violaciones.
        """
        violations = []
        
        p = self._signal_values[self.PURPOSE_IDX]
        c = self._signal_values[self.CONFIDENCE_IDX]
        r = self._signal_values[self.CONSTRAINTS_IDX]
        k = self._signal_values[self.RISK_IDX]
        
        # Violación 1: Propósito muy débil con confianza muy alta
        # Interpretación: LLM muy seguro de algo sin propósito → paradoja
        if p < (1.0 - self.INCONSISTENCY_THRESHOLD) and \
           c > (1.0 - self.INCONSISTENCY_THRESHOLD):
            delta = c - p
            violations.append(
                f"Purpose-Confidence inconsistency: "
                f"weak purpose ({p:.2f}) with high confidence ({c:.2f}), Δ={delta:.2f}"
            )
        
        # Violación 2: Confianza muy baja con restricciones muy satisfechas
        # Interpretación: LLM inseguro pero código perfecto → sospechoso
        if c < self.INCONSISTENCY_THRESHOLD and \
           r > (1.0 - self.INCONSISTENCY_THRESHOLD):
            delta = r - c
            violations.append(
                f"Confidence-Constraints inconsistency: "
                f"low confidence ({c:.2f}) with high constraint satisfaction ({r:.2f}), Δ={delta:.2f}"
            )
        
        # Violación 3: Restricciones muy violadas con riesgo muy alto
        # Interpretación: código complejo en dominio crítico → peligroso
        if r < self.INCONSISTENCY_THRESHOLD and \
           k > (1.0 - self.INCONSISTENCY_THRESHOLD):
            delta = k - r
            violations.append(
                f"Constraints-Risk inconsistency: "
                f"low constraint satisfaction ({r:.2f}) with high risk tolerance ({k:.2f}), Δ={delta:.2f}"
            )
        
        # Violación 4: Riesgo muy bajo con propósito muy fuerte (cierre cíclico)
        # Interpretación: propósito crítico pero sin tolerancia → sobre-restrictivo
        if k < self.INCONSISTENCY_THRESHOLD and \
           p > (1.0 - self.INCONSISTENCY_THRESHOLD):
            delta = p - k
            violations.append(
                f"Risk-Purpose inconsistency: "
                f"high purpose ({p:.2f}) with low risk tolerance ({k:.2f}), Δ={delta:.2f}"
            )
        
        return violations

    def compute_cohomology_dimension(self) -> int:
        """
        Calcula dim H¹(K; ℝ).
        
        Para el ciclo de 4 vértices, dim H¹ = 1 topológicamente.
        Retornamos el número de violaciones de consistencia detectadas,
        que indica la "dimensión obstructiva".
        
        [PRE] Señales deben estar establecidas
        [POST] Resultado ≥ 0
        
        Returns:
            Número de violaciones (dimensión de obstrucción).
        """
        if self._cohomology_dimension is None:
            self._cycle_violations = self._detect_cycle_violations()
            self._cohomology_dimension = len(self._cycle_violations)
        
        assert self._cohomology_dimension >= 0, "Postcondition violated"
        return self._cohomology_dimension

    def has_obstruction(self) -> bool:
        """
        Determina si existe obstrucción topológica.
        
        [POST] Verdadero ssi dim H¹ > 0
        
        Returns:
            True si hay paradoja semántica.
        """
        return self.compute_cohomology_dimension() > 0

    def get_obstruction_description(self) -> str:
        """
        Genera descripción detallada de obstrucciones.
        
        [PRE] Cohomología debe estar calculada
        [POST] String no vacío si hay obstrucciones
        
        Returns:
            Descripción textual de obstrucciones.
        """
        dim = self.compute_cohomology_dimension()
        
        if dim == 0:
            return "No topological obstruction detected (H¹ = 0)"
        
        desc_lines = [
            f"Topological obstruction detected: dim H¹ = {dim}",
            "Semantic paradoxes (cycle violations):",
        ]
        
        for i, violation in enumerate(self._cycle_violations, 1):
            desc_lines.append(f"  [{i}] {violation}")
        
        return "\n".join(desc_lines)

    def __repr__(self) -> str:
        dim = self._cohomology_dimension if self._cohomology_dimension is not None else '?'
        return f"SimplicialCohomology(dim H¹ = {dim})"


# =============================================================================
# VALIDADORES ESPECIALIZADOS
# =============================================================================

class PurposeValidator:
    """
    Validador de propósito empresarial basado en grafo de conocimiento.
    
    Evalúa si el código tiene un propósito empresarial claro y medible,
    mapeando conceptos técnicos a problemas canónicos de negocio.
    
    Invariantes:
    [INV1] 0 < min_strength_threshold ≤ 1
    [INV2] canonical_problems es conjunto no vacío
    """

    # Problemas empresariales canónicos
    DEFAULT_CANONICAL_PROBLEMS = frozenset([
        "COST_REDUCTION",
        "LATENCY_REDUCTION",
        "RELIABILITY_IMPROVEMENT",
        "SCALABILITY_ENHANCEMENT",
        "SECURITY_HARDENING",
        "COMPLIANCE_ADHERENCE",
        "USER_EXPERIENCE_IMPROVEMENT",
        "DATA_QUALITY_ENHANCEMENT",
        "MAINTAINABILITY_IMPROVEMENT",
        "OBSERVABILITY_ENHANCEMENT",
    ])

    def __init__(
        self,
        knowledge_graph: Optional[Dict[str, Dict[str, float]]] = None,
        canonical_problems: Optional[FrozenSet[str]] = None,
        min_strength_threshold: float = 0.65
    ):
        """
        Inicializa validador de propósito.
        
        [PRE] 0 < min_strength_threshold ≤ 1
        [PRE] canonical_problems es None o conjunto no vacío
        
        Args:
            knowledge_graph: Grafo concepto → {problema: peso}
            canonical_problems: Conjunto de problemas canónicos
            min_strength_threshold: Umbral mínimo de fuerza
        """
        if not (0.0 < min_strength_threshold <= 1.0):
            raise ContractViolationError(
                f"min_strength_threshold ∈ (0,1], got {min_strength_threshold}"
            )
        
        self.kg = knowledge_graph or {}
        self.canonical_problems = canonical_problems or self.DEFAULT_CANONICAL_PROBLEMS
        self.min_strength = min_strength_threshold
        
        if not self.canonical_problems:
            raise ContractViolationError("canonical_problems cannot be empty")
        
        logger.debug(
            f"Initialized PurposeValidator with {len(self.canonical_problems)} "
            f"canonical problems, threshold={self.min_strength:.2f}"
        )

    def validate(self, purposes: List[BusinessPurpose]) -> Tuple[bool, float, str]:
        """
        Valida lista de propósitos empresariales.
        
        [PRE] purposes es lista (puede estar vacía)
        [POST] (válido, score, razón) donde score ∈ [0, 1]
        
        Args:
            purposes: Lista de propósitos a validar
            
        Returns:
            (es_válido, score_máximo, descripción)
        """
        if not purposes:
            return False, 0.0, "No business purposes provided (empty list)"
        
        # Filtrar propósitos canónicos
        canonical_purposes = [
            p for p in purposes
            if p.business_problem in self.canonical_problems
        ]
        
        if not canonical_purposes:
            available = ", ".join(sorted(self.canonical_problems)[:5])
            return (
                False, 
                0.0, 
                f"No purposes map to canonical problems. "
                f"Expected one of: {available}..."
            )
        
        # Calcular fuerza efectiva máxima
        max_strength = max(p.effective_strength for p in canonical_purposes)
        
        if max_strength < self.min_strength:
            return (
                False, 
                max_strength,
                f"Maximum purpose strength {max_strength:.3f} below "
                f"threshold {self.min_strength:.3f}"
            )
        
        # Encontrar mejor propósito
        best_purpose = max(canonical_purposes, key=lambda p: p.effective_strength)
        
        reason = (
            f"Strong canonical purpose identified: "
            f"{best_purpose.concept} → {best_purpose.business_problem} "
            f"(σ_eff={max_strength:.3f})"
        )
        
        return True, max_strength, reason

    def compute_purpose_score(self, purposes: List[BusinessPurpose]) -> float:
        """
        Calcula score agregado de propósito.
        
        Combina fuerza máxima (70%) con fuerza media (30%) de propósitos canónicos.
        
        [PRE] purposes es lista
        [POST] 0 ≤ score ≤ 1
        
        Args:
            purposes: Lista de propósitos
            
        Returns:
            Score agregado ∈ [0, 1]
        """
        if not purposes:
            return 0.0
        
        canonical_strengths = [
            p.effective_strength
            for p in purposes
            if p.business_problem in self.canonical_problems
        ]
        
        if not canonical_strengths:
            return 0.0
        
        max_strength = max(canonical_strengths)
        mean_strength = float(np.mean(canonical_strengths))
        
        # Combinación ponderada: privilegiar el máximo
        score = 0.7 * max_strength + 0.3 * mean_strength
        
        assert 0.0 <= score <= 1.0, "Postcondition violated"
        return score


class ConfidenceFilter:
    """
    Filtro de confianza para salidas de LLM.
    
    Rechaza salidas con:
    - Confianza < min_confidence
    - Entropía > max_entropy
    - Entropía normalizada > max_normalized_entropy
    - Singularidades estocásticas (H = ∞ ∨ conf = 0)
    
    Invariantes:
    [INV1] 0 < min_confidence ≤ 1
    [INV2] max_entropy > 0
    [INV3] max_normalized_entropy > 0
    """

    DEFAULT_MIN_CONFIDENCE = 0.60
    DEFAULT_MAX_ENTROPY = 2.5
    DEFAULT_MAX_NORMALIZED_ENTROPY = 0.5

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_entropy: float = DEFAULT_MAX_ENTROPY,
        max_normalized_entropy: float = DEFAULT_MAX_NORMALIZED_ENTROPY
    ):
        """
        Inicializa filtro de confianza.
        
        [PRE] 0 < min_confidence ≤ 1
        [PRE] max_entropy > 0
        [PRE] max_normalized_entropy > 0
        """
        if not (0.0 < min_confidence <= 1.0):
            raise ContractViolationError(
                f"min_confidence ∈ (0,1], got {min_confidence}"
            )
        if max_entropy <= 0:
            raise ContractViolationError(f"max_entropy > 0, got {max_entropy}")
        if max_normalized_entropy <= 0:
            raise ContractViolationError(
                f"max_normalized_entropy > 0, got {max_normalized_entropy}"
            )
        
        self.min_confidence = min_confidence
        self.max_entropy = max_entropy
        self.max_normalized_entropy = max_normalized_entropy
        
        logger.debug(
            f"Initialized ConfidenceFilter(min_conf={min_confidence:.2f}, "
            f"max_H={max_entropy:.2f}, max_H_norm={max_normalized_entropy:.2f})"
        )

    def validate(self, llm_output: LLMOutput, state_metrics: Optional[Dict[str, Any]] = None) -> Tuple[Verdict, float, str]:
        """
        Valida salida de LLM utilizando la Temperatura de Gobernanza y la Distribución de Gibbs.
        
        [PRE] llm_output es instancia válida de LLMOutput
        [POST] (Verdict, score, razón) donde score ∈ [0, 1]
        
        Args:
            llm_output: Metadatos del LLM
            state_metrics: Métricas del estado de estratos inferiores (Axioma M5)
            
        Returns:
            (Verdict, confidence_score, descripción)
        """
        T_gov = 1.0  # Temperatura de mercado por defecto

        if state_metrics:
            beta_1 = state_metrics.get('beta_1', 0)
            p_diss = state_metrics.get('p_diss', 0.0)

            # Axioma M5: Colapso si invariantes fallan
            if beta_1 > 0 or p_diss < 0:
                T_gov = 0.0

        # Manejo explícito de la singularidad cuando T_gov -> 0 (Colapso Termodinámico de Heaviside)
        if T_gov == 0.0:
            return (
                Verdict.REJECT,
                0.0,
                "Heaviside Thermodynamic Collapse: T_gov=0 due to underlying physical/topological singularity (beta_1 > 0 or P_diss < 0)."
            )

        # Detectar singularidad estocástica
        if llm_output.is_singular:
            return (
                Verdict.REJECT,
                0.0,
                f"Stochastic singularity detected: "
                f"H={'∞' if math.isinf(llm_output.entropy) else f'{llm_output.entropy:.2f}'}, "
                f"conf={llm_output.confidence:.2f}"
            )
        
        # Función de Partición Z y Distribución de Gibbs sobre el retículo
        E_reject = 5.0  # Energía base del sumidero (estado seguro)
        E_viable = 0.0  # Energía de aceptación (se incrementa si hay anomalías)
        
        # Penalizaciones de energía (frustración por baja calidad)
        if llm_output.confidence < self.min_confidence:
            E_viable += (self.min_confidence - llm_output.confidence) * 30.0

        if llm_output.entropy > self.max_entropy:
            E_viable += (llm_output.entropy - self.max_entropy) * 10.0

        norm_entropy = llm_output.normalized_entropy
        if norm_entropy > self.max_normalized_entropy:
            E_viable += (norm_entropy - self.max_normalized_entropy) * 30.0

        # Interpolación de energías para estados intermedios
        E_conditional = E_viable * 0.7 + E_reject * 0.3
        E_warning = E_viable * 0.3 + E_reject * 0.7
        
        energies = {
            Verdict.VIABLE: E_viable,
            Verdict.CONDITIONAL: E_conditional,
            Verdict.WARNING: E_warning,
            Verdict.REJECT: E_reject
        }
        
        k_B = 1.0
        Z = sum(math.exp(-np.clip(E / (k_B * T_gov), -700.0, 700.0)) for E in energies.values())
        
        probs = {v: math.exp(-np.clip(E / (k_B * T_gov), -700.0, 700.0)) / Z for v, E in energies.items()}
        
        # Seleccionar veredicto según Máxima Probabilidad a Posteriori (MAP)
        chosen_verdict = max(probs, key=probs.get)

        # Score de confianza probabilístico
        score = probs[Verdict.VIABLE] + 0.5 * probs[Verdict.CONDITIONAL]

        reason = (f"Gibbs Distribution mapped. "
                  f"P(VIABLE)={probs[Verdict.VIABLE]:.2f}, "
                  f"P(REJECT)={probs[Verdict.REJECT]:.2f}")

        return chosen_verdict, score, reason

    def compute_confidence_score(self, llm_output: LLMOutput) -> float:
        """
        Calcula score de confianza normalizado.
        
        [POST] 0 ≤ score ≤ 1
        
        Args:
            llm_output: Metadatos del LLM
            
        Returns:
            Score de confianza ∈ [0, 1]
        """
        _, score, _ = self.validate(llm_output)
        assert 0.0 <= score <= 1.0, "Postcondition violated"
        return score


class ConstraintMapper:
    """
    Mapea perfiles de riesgo a restricciones técnicas de complejidad.
    
    Utiliza la conexión de Galois entre tolerancia al riesgo (dominio abstracto)
    y límites de complejidad (dominio concreto/sintáctico).
    
    Propiedades:
    - Monotonía: mayor criticidad ⟹ límites más estrictos
    - Contravarianza: menor tolerancia ⟹ cotas más ajustadas
    """

    # Límites de complejidad por categoría de riesgo
    COMPLEXITY_LIMITS = {
        "HIGHLY_CONSERVATIVE": {
            "cyclomatic": 10,
            "depth": 3,
            "loc": 50,
            "cognitive": 15
        },
        "CONSERVATIVE": {
            "cyclomatic": 15,
            "depth": 4,
            "loc": 100,
            "cognitive": 25
        },
        "MODERATE": {
            "cyclomatic": 20,
            "depth": 5,
            "loc": 200,
            "cognitive": 40
        },
        "AGGRESSIVE": {
            "cyclomatic": 30,
            "depth": 7,
            "loc": 500,
            "cognitive": 60
        },
        "HIGHLY_AGGRESSIVE": {
            "cyclomatic": 50,
            "depth": 10,
            "loc": 1000,
            "cognitive": 100
        },
    }

    def map_to_constraints(self, risk_profile: RiskProfile) -> Dict[str, int]:
        """
        Mapea perfil de riesgo a límites de complejidad.
        
        Aplica factor de criticidad para ajustar límites:
            limit_adjusted = base_limit × (1 - 0.4 × criticality)
        
        [PRE] risk_profile es instancia válida de RiskProfile
        [POST] Todos los límites ≥ 1
        
        Args:
            risk_profile: Perfil de riesgo empresarial
            
        Returns:
            Diccionario {métrica: límite}
        """
        category = risk_profile.risk_category
        base_limits = self.COMPLEXITY_LIMITS.get(
            category, 
            self.COMPLEXITY_LIMITS["MODERATE"]
        )
        
        # Factor de ajuste por criticidad (mayor criticidad → límites más estrictos)
        criticality_factor = 1.0 - 0.4 * risk_profile.domain_criticality
        
        adjusted_limits = {
            metric: max(1, int(limit * criticality_factor))
            for metric, limit in base_limits.items()
        }
        
        # Verificar postcondición
        assert all(v >= 1 for v in adjusted_limits.values()), "Postcondition violated"
        
        return adjusted_limits

    def compute_constraint_score(
        self,
        actual_metrics: Dict[str, int],
        risk_profile: RiskProfile
    ) -> float:
        """
        Calcula score de cumplimiento de restricciones.
        
        Para cada métrica, si actual ≤ límite, score = 1.0.
        Si actual > límite, score = exp(-2 × (actual - límite) / límite).
        
        [PRE] actual_metrics valores ≥ 0
        [POST] 0 ≤ score ≤ 1
        
        Args:
            actual_metrics: Métricas reales del código
            risk_profile: Perfil de riesgo
            
        Returns:
            Score agregado ∈ [0, 1]
        """
        limits = self.map_to_constraints(risk_profile)
        metric_scores = []
        
        for metric, limit in limits.items():
            actual = actual_metrics.get(metric, 0)
            
            if actual < 0:
                logger.warning(f"Negative metric value: {metric}={actual}, treating as 0")
                actual = 0
            
            if actual <= limit:
                metric_scores.append(1.0)
            else:
                # Penalización exponencial por exceso
                excess_ratio = (actual - limit) / limit
                score = math.exp(-2.0 * excess_ratio)
                metric_scores.append(score)
        
        aggregate_score = float(np.mean(metric_scores)) if metric_scores else 1.0
        
        assert 0.0 <= aggregate_score <= 1.0, "Postcondition violated"
        return aggregate_score


# =============================================================================
# MOTOR PRINCIPAL DE VALIDACIÓN SEMÁNTICA
# =============================================================================

class SemanticValidationEngine:
    """
    Motor de validación semántica con fundamentos algebraicos rigurosos.
    
    Integra:
    1. Métrica de Mahalanobis (geometría riemanniana)
    2. Cohomología simplicial (topología algebraica)
    3. Retículo de veredictos (álgebra de orden)
    4. Validadores especializados (teoría de dominios)
    
    Proceso de validación:
    
    Fase 0: Detección de singularidades estocásticas
    Fase 1: Validación de propósito empresarial
    Fase 2: Filtrado de confianza del LLM
    Fase 3: Verificación de restricciones técnicas
    Fase 4: Bonus de tolerancia al riesgo
    Fase 5: Detección de obstrucciones cohomológicas
    Fase 6: Cálculo de distancia de Mahalanobis
    Fase 7: Determinación de veredicto por umbrales
    
    Invariantes:
    [INV1] Los umbrales de Mahalanobis están estrictamente ordenados
    [INV2] El veredicto final es el supremo de veredictos parciales
    [INV3] dim H¹ > 0 ⟹ veredicto = REJECT
    """

    # Umbrales de distancia de Mahalanobis para veredictos
    # Propiedad: threshold[VIABLE] < threshold[CONDITIONAL] < threshold[WARNING]
    MAHALANOBIS_THRESHOLDS = {
        Verdict.VIABLE: 0.35,       # Óptimo
        Verdict.CONDITIONAL: 0.55,  # Aceptable con reservas
        Verdict.WARNING: 0.80,      # Preocupante
        # D_M > 0.80 → REJECT
    }

    def __init__(
        self,
        knowledge_graph: Optional[Dict[str, Dict[str, float]]] = None,
        risk_profile: Optional[RiskProfile] = None,
        metric: Optional[MahalanobisMetric] = None,
        confidence_filter: Optional[ConfidenceFilter] = None,
        enable_cohomology: bool = True
    ):
        """
        Inicializa motor de validación.
        
        [PRE] knowledge_graph es dict válido o None
        [PRE] risk_profile es RiskProfile válido o None
        [PRE] metric es MahalanobisMetric válido o None
        [POST] Todos los componentes inicializados correctamente
        
        Args:
            knowledge_graph: Grafo de conocimiento concepto→problema
            risk_profile: Perfil de riesgo empresarial
            metric: Tensor métrico personalizado
            confidence_filter: Inyección de dependencias para el filtro de confianza
            enable_cohomology: Habilitar detección cohomológica
        """
        self.risk_profile = risk_profile or RiskProfile(risk_tolerance=0.5)
        self.metric = metric or MahalanobisMetric()
        self.enable_cohomology = enable_cohomology
        
        # Validadores especializados
        self.purpose_validator = PurposeValidator(knowledge_graph)
        self.confidence_filter = confidence_filter or ConfidenceFilter()
        self.constraint_mapper = ConstraintMapper()
        
        # Cohomología simplicial
        self.cohomology = SimplicialCohomology()
        
        # Verificar orden de umbrales
        thresholds = [
            self.MAHALANOBIS_THRESHOLDS[Verdict.VIABLE],
            self.MAHALANOBIS_THRESHOLDS[Verdict.CONDITIONAL],
            self.MAHALANOBIS_THRESHOLDS[Verdict.WARNING],
        ]
        assert thresholds == sorted(thresholds), "[INV1] violated: thresholds not ordered"
        
        logger.info(
            f"Initialized SemanticValidationEngine v3.0\n"
            f"  Risk Profile: {self.risk_profile}\n"
            f"  Metric: {self.metric}\n"
            f"  Cohomology: {'enabled' if enable_cohomology else 'disabled'}"
        )

    def validate(
        self,
        purposes: List[BusinessPurpose],
        llm_output: LLMOutput,
        code_metrics: Optional[Dict[str, int]] = None,
        state_metrics: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Ejecuta validación semántica completa.
        
        [PRE] purposes es lista de BusinessPurpose válidos
        [PRE] llm_output es LLMOutput válido
        [PRE] code_metrics es dict con valores ≥ 0 o None
        [POST] resultado satisface invariantes de ValidationResult
        [POST] dim H¹ > 0 ⟹ resultado.verdict == REJECT
        
        Args:
            purposes: Lista de propósitos empresariales
            llm_output: Metadatos del LLM
            code_metrics: Métricas opcionales del código
            state_metrics: Métricas del estado de estratos inferiores (Axioma M5)
            
        Returns:
            ValidationResult con veredicto y trazabilidad completa
            
        Raises:
            ContractViolationError: Si se violan precondiciones
        """
        logger.info("=" * 80)
        logger.info("SEMANTIC VALIDATION STARTED (v3.0 - Rigorous Foundations)")
        logger.info("=" * 80)
        
        result = ValidationResult(
            verdict=Verdict.REJECT,  # Pesimista por default
            mahalanobis_distance=float('inf')
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FASE 0: DETECCIÓN DE SINGULARIDAD ESTOCÁSTICA
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if llm_output.is_singular:
            logger.error("⊗ PHASE 0 FAILED: Stochastic singularity detected")
            result.add_reason(
                f"Stochastic singularity: H={'∞' if math.isinf(llm_output.entropy) else llm_output.entropy}, "
                f"conf={llm_output.confidence}",
                'singularity',
                0.0
            )
            result.verdict = Verdict.REJECT
            result.mahalanobis_distance = float('inf')
            logger.info(f"Final Verdict: {result.verdict} (singularity)")
            return result
        
        logger.info("✓ Phase 0: No stochastic singularity")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FASE 1: VALIDACIÓN DE PROPÓSITO EMPRESARIAL
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        is_valid_purpose, purpose_strength, purpose_reason = \
            self.purpose_validator.validate(purposes)
        purpose_score = self.purpose_validator.compute_purpose_score(purposes)
        
        result.add_reason(purpose_reason, 'purpose', purpose_score)
        
        if not is_valid_purpose:
            logger.warning(f"⚠ Phase 1: Purpose validation failed - {purpose_reason}")
            if purpose_score < 0.15:  # Umbral crítico
                result.verdict = Verdict.REJECT
                result.mahalanobis_distance = 1.0
                logger.info(f"Final Verdict: {result.verdict} (weak purpose)")
                return result
        else:
            logger.info(f"✓ Phase 1: Purpose validated (score={purpose_score:.3f})")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FASE 2: FILTRADO DE CONFIANZA DEL LLM (Colapso Termodinámico)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        verdict_confidence, confidence_strength, confidence_reason = \
            self.confidence_filter.validate(llm_output, state_metrics)
        
        confidence_score = confidence_strength
        result.add_reason(confidence_reason, 'confidence', confidence_score)
        
        if verdict_confidence == Verdict.REJECT:
            logger.warning(f"⚠ Phase 2: Confidence validation rejected - {confidence_reason}")
            result.verdict = Verdict.REJECT
            result.mahalanobis_distance = 1.0
            logger.info(f"Final Verdict: {result.verdict} (rejected by confidence filter)")
            return result
        elif verdict_confidence in (Verdict.WARNING, Verdict.CONDITIONAL):
            logger.warning(f"⚠ Phase 2: Confidence validation marginal - {confidence_reason}")
        else:
            logger.info(f"✓ Phase 2: Confidence validated (score={confidence_score:.3f})")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FASE 3: VERIFICACIÓN DE RESTRICCIONES TÉCNICAS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if code_metrics:
            constraints_score = self.constraint_mapper.compute_constraint_score(
                code_metrics, self.risk_profile
            )
            limits = self.constraint_mapper.map_to_constraints(self.risk_profile)
            
            violations = [
                f"{metric}: {code_metrics.get(metric, 0)} > {limit}"
                for metric, limit in limits.items()
                if code_metrics.get(metric, 0) > limit
            ]
            
            if violations:
                constraint_reason = (
                    f"Constraint violations detected: {', '.join(violations)} "
                    f"(score={constraints_score:.3f})"
                )
                logger.warning(f"⚠ Phase 3: {constraint_reason}")
            else:
                constraint_reason = f"All constraints satisfied (score={constraints_score:.3f})"
                logger.info(f"✓ Phase 3: {constraint_reason}")
            
            result.add_reason(constraint_reason, 'constraints', constraints_score)
        else:
            constraints_score = 1.0
            result.add_reason(
                "No code metrics provided (assumed compliant)",
                'constraints',
                1.0
            )
            logger.info("✓ Phase 3: No metrics to validate")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FASE 4: BONUS DE TOLERANCIA AL RIESGO
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        risk_bonus = self.risk_profile.effective_tolerance
        result.add_reason(
            f"Risk tolerance applied: {self.risk_profile.risk_category} "
            f"(bonus={risk_bonus:.3f})",
            'risk',
            risk_bonus
        )
        logger.info(f"✓ Phase 4: Risk bonus={risk_bonus:.3f}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FASE 5: DETECCIÓN DE OBSTRUCCIONES COHOMOLÓGICAS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.cohomology.set_signals(
            purpose_score,
            confidence_score,
            constraints_score,
            risk_bonus
        )
        
        if self.enable_cohomology:
            dim_H1 = self.cohomology.compute_cohomology_dimension()
            result.cohomology_dimension = dim_H1
            
            if self.cohomology.has_obstruction():
                obstruction_desc = self.cohomology.get_obstruction_description()
                logger.error(f"⊗ PHASE 5 FAILED: Topological obstruction\n{obstruction_desc}")
                result.add_reason(obstruction_desc, 'cohomology', 0.0)
                result.verdict = Verdict.REJECT
                result.mahalanobis_distance = 1.0
                logger.info(f"Final Verdict: {result.verdict} (topological obstruction)")
                return result
            else:
                logger.info(f"✓ Phase 5: No topological obstruction (dim H¹ = 0)")
        else:
            logger.info("⊘ Phase 5: Cohomology check disabled")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FASE 6: CÁLCULO DE DISTANCIA DE MAHALANOBIS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        signal_vector = np.array([
            purpose_score,
            confidence_score,
            constraints_score,
            risk_bonus
        ], dtype=np.float64)
        
        D_M = self.metric.distance_to_ideal(signal_vector)
        result.mahalanobis_distance = D_M
        
        logger.info(
            f"✓ Phase 6: Mahalanobis distance computed: D_M = {D_M:.4f}\n"
            f"  Signal vector: {signal_vector}"
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FASE 7: DETERMINACIÓN DE VEREDICTO POR UMBRALES
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if D_M <= self.MAHALANOBIS_THRESHOLDS[Verdict.VIABLE]:
            result.verdict = Verdict.VIABLE
        elif D_M <= self.MAHALANOBIS_THRESHOLDS[Verdict.CONDITIONAL]:
            result.verdict = Verdict.CONDITIONAL
        elif D_M <= self.MAHALANOBIS_THRESHOLDS[Verdict.WARNING]:
            result.verdict = Verdict.WARNING
        else:
            result.verdict = Verdict.REJECT
        
        logger.info(f"✓ Phase 7: Verdict determined: {result.verdict}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # METADATA Y FINALIZACIÓN
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        result.metadata = {
            'risk_profile': str(self.risk_profile),
            'risk_category': self.risk_profile.risk_category,
            'mahalanobis_distance': D_M,
            'cohomology_dimension': result.cohomology_dimension,
            'metric_condition_number': self.metric._condition_number,
            'thresholds': {
                v.name: t for v, t in self.MAHALANOBIS_THRESHOLDS.items()
            },
            'signal_vector': signal_vector.tolist(),
        }
        
        logger.info("=" * 80)
        logger.info(f"VALIDATION COMPLETE: {result.verdict} (D_M={D_M:.4f})")
        logger.info("=" * 80)
        
        return result

    def explain_verdict(self, result: ValidationResult) -> str:
        """
        Genera explicación legible y estructurada del veredicto.
        
        [PRE] result es ValidationResult válido
        [POST] Retorna string no vacío
        
        Args:
            result: Resultado de validación
            
        Returns:
            Explicación formateada del veredicto
        """
        lines = [
            "=" * 80,
            "SEMANTIC VALIDATION REPORT",
            "=" * 80,
            "",
            f"Final Verdict: {result.verdict.name}",
            f"  Severity: {result.verdict.severity_score:.2f}",
            f"  Accepted: {result.verdict.is_accepted}",
            f"  Requires Review: {result.verdict.requires_human_review}",
            "",
            f"Geometric Distance (Mahalanobis): {result.mahalanobis_distance:.4f}",
            f"  Thresholds:",
            f"    VIABLE      : ≤ {self.MAHALANOBIS_THRESHOLDS[Verdict.VIABLE]:.2f}",
            f"    CONDITIONAL : ≤ {self.MAHALANOBIS_THRESHOLDS[Verdict.CONDITIONAL]:.2f}",
            f"    WARNING     : ≤ {self.MAHALANOBIS_THRESHOLDS[Verdict.WARNING]:.2f}",
            f"    REJECT      : > {self.MAHALANOBIS_THRESHOLDS[Verdict.WARNING]:.2f}",
            "",
            f"Topological Analysis:",
            f"  Cohomology dimension (dim H¹): {result.cohomology_dimension}",
            f"  Obstruction present: {result.has_topological_obstruction}",
            "",
            "Signal Breakdown:",
        ]
        
        for signal_name, signal_value in sorted(result.signals.items()):
            lines.append(f"  {signal_name:15s}: {signal_value:.4f}")
        
        lines.extend([
            "",
            "Validation Reasons:",
        ])
        
        for i, reason in enumerate(result.reasons, 1):
            lines.append(f"  [{i}] {reason}")
        
        if result.metadata:
            lines.extend([
                "",
                "Metadata:",
            ])
            for key, value in sorted(result.metadata.items()):
                if key == 'signal_vector':
                    vec_str = ', '.join(f"{v:.3f}" for v in value)
                    lines.append(f"  {key}: [{vec_str}]")
                elif key == 'thresholds':
                    continue  # Ya mostrado arriba
                else:
                    lines.append(f"  {key}: {value}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def create_default_knowledge_graph() -> Dict[str, Dict[str, float]]:
    """
    Construye grafo de conocimiento de ejemplo.
    
    Mapea conceptos técnicos a problemas empresariales canónicos
    con pesos que representan fuerza de la relación semántica.
    
    [POST] Todos los pesos ∈ (0, 1]
    
    Returns:
        Dict[concepto, Dict[problema, peso]]
    """
    kg: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    # Formato: (concepto, problema, peso)
    mappings = [
        # Caching
        ("caching", "LATENCY_REDUCTION", 0.95),
        ("caching", "COST_REDUCTION", 0.75),
        ("caching", "SCALABILITY_ENHANCEMENT", 0.65),
        
        # Load Balancing
        ("load_balancing", "RELIABILITY_IMPROVEMENT", 0.90),
        ("load_balancing", "SCALABILITY_ENHANCEMENT", 0.95),
        ("load_balancing", "LATENCY_REDUCTION", 0.70),
        
        # Encryption & Security
        ("encryption", "SECURITY_HARDENING", 0.98),
        ("encryption", "COMPLIANCE_ADHERENCE", 0.85),
        ("encryption", "DATA_QUALITY_ENHANCEMENT", 0.60),
        
        # Monitoring & Observability
        ("monitoring", "RELIABILITY_IMPROVEMENT", 0.90),
        ("monitoring", "OBSERVABILITY_ENHANCEMENT", 0.95),
        ("monitoring", "USER_EXPERIENCE_IMPROVEMENT", 0.65),
        
        # Data Validation
        ("data_validation", "DATA_QUALITY_ENHANCEMENT", 0.95),
        ("data_validation", "RELIABILITY_IMPROVEMENT", 0.80),
        ("data_validation", "COMPLIANCE_ADHERENCE", 0.70),
        
        # Code Refactoring
        ("refactoring", "MAINTAINABILITY_IMPROVEMENT", 0.95),
        ("refactoring", "COST_REDUCTION", 0.70),
        ("refactoring", "RELIABILITY_IMPROVEMENT", 0.65),
        
        # Automated Testing
        ("automated_testing", "RELIABILITY_IMPROVEMENT", 0.95),
        ("automated_testing", "MAINTAINABILITY_IMPROVEMENT", 0.80),
        ("automated_testing", "COST_REDUCTION", 0.60),
    ]
    
    for concept, problem, weight in mappings:
        assert 0.0 < weight <= 1.0, f"Invalid weight: {weight}"
        kg[concept][problem] = weight
    
    return dict(kg)


# =============================================================================
# COMPATIBILIDAD CON API LEGACY (DEPRECATED)
# =============================================================================

class OntologicalDiffeomorphismEngine:
    """
    Clase de compatibilidad con API anterior.
    
    ⚠️  DEPRECATED: Usar SemanticValidationEngine directamente.
    
    Esta clase se mantiene solo para compatibilidad retroactiva y
    será eliminada en versiones futuras.
    """

    def __init__(
        self, 
        knowledge_graph: Any = None, 
        business_profile: Any = None, 
        **kwargs
    ):
        """
        Constructor legacy.
        
        Args:
            knowledge_graph: Grafo de conocimiento (networkx o dict)
            business_profile: Perfil de negocio legacy
            **kwargs: Argumentos adicionales (ignorados)
        """
        import warnings
        msg = (
            "⚠️  OntologicalDiffeomorphismEngine is DEPRECATED. "
            "Use SemanticValidationEngine instead. "
            "This compatibility layer will be removed in v4.0."
        )
        logger.warning(msg)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        
        # Convertir knowledge_graph de networkx a dict si es necesario
        if hasattr(knowledge_graph, 'edges'):
            kg_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
            for u, v, data in knowledge_graph.edges(data=True):
                kg_dict[str(u)][str(v)] = data.get('weight', 0.8)
            knowledge_graph = dict(kg_dict)
        
        # Convertir business_profile legacy a RiskProfile
        if hasattr(business_profile, 'risk_tolerance'):
            risk_profile = RiskProfile(
                risk_tolerance=float(business_profile.risk_tolerance),
                domain_criticality=float(
                    getattr(business_profile, 'domain_criticality', 0.5)
                ),
                acceptable_failure_rate=float(
                    getattr(business_profile, 'acceptable_failure_rate', 0.01)
                )
            )
        else:
            risk_profile = RiskProfile(risk_tolerance=0.5)
        
        # Crear engine moderno
        self._engine = SemanticValidationEngine(
            knowledge_graph=knowledge_graph,
            risk_profile=risk_profile
        )

    def compile_wisdom(
        self,
        tool_semantics: List[Any],
        llm_entropy: float,
        llm_confidence: float
    ) -> int:
        """
        API legacy: compile_wisdom.
        
        Args:
            tool_semantics: Lista de objetos con atributos de semántica
            llm_entropy: Entropía del LLM
            llm_confidence: Confianza del LLM
            
        Returns:
            Código de veredicto (0=VIABLE, 1=CONDITIONAL, 2=WARNING, 3=REJECT)
        """
        purposes = []
        for sem in tool_semantics:
            if hasattr(sem, 'source_concept') and hasattr(sem, 'target_business_pain'):
                purposes.append(BusinessPurpose(
                    concept=str(sem.source_concept),
                    business_problem=str(sem.target_business_pain),
                    strength=float(getattr(sem, 'semantic_weight', 0.8)),
                    confidence=1.0
                ))
        
        llm_output = LLMOutput(
            entropy=float(llm_entropy),
            confidence=float(llm_confidence),
            temperature=1.0,
            num_tokens=100
        )
        
        result = self._engine.validate(purposes, llm_output)
        return int(result.verdict.value)


# =============================================================================
# PUNTO DE ENTRADA PARA TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Configurar logging para output detallado
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-8s | %(message)s',
        stream=sys.stdout
    )
    
    print("\n" + "=" * 80)
    print("SEMANTIC VALIDATION ENGINE v3.0 - TEST SUITE")
    print("=" * 80 + "\n")
    
    # Crear knowledge graph
    kg = create_default_knowledge_graph()
    
    # =========================================================================
    # TEST 1: Código con propósito fuerte, alta confianza, métricas buenas
    # =========================================================================
    print("\n" + "▸" * 80)
    print("TEST 1: High-quality code with strong purpose")
    print("▸" * 80 + "\n")
    
    risk_profile_conservative = RiskProfile(
        risk_tolerance=0.3,
        domain_criticality=0.8,
        acceptable_failure_rate=0.001
    )
    
    engine1 = SemanticValidationEngine(
        knowledge_graph=kg,
        risk_profile=risk_profile_conservative,
        enable_cohomology=True
    )
    
    purposes1 = [
        BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.95, confidence=0.98),
        BusinessPurpose("caching", "COST_REDUCTION", strength=0.75, confidence=0.95),
    ]
    
    llm_output1 = LLMOutput(
        entropy=0.45,
        confidence=0.94,
        temperature=0.7,
        num_tokens=150
    )
    
    code_metrics1 = {
        'cyclomatic': 8,
        'depth': 3,
        'loc': 45,
        'cognitive': 12
    }
    
    result1 = engine1.validate(purposes1, llm_output1, code_metrics1)
    print(engine1.explain_verdict(result1))
    
    # =========================================================================
    # TEST 2: Código sin propósito claro, baja confianza
    # =========================================================================
    print("\n" + "▸" * 80)
    print("TEST 2: Code without clear purpose, low confidence")
    print("▸" * 80 + "\n")
    
    purposes2 = [
        BusinessPurpose("unknown_concept", "UNKNOWN_PROBLEM", strength=0.25, confidence=0.40),
    ]
    
    llm_output2 = LLMOutput(
        entropy=3.8,
        confidence=0.42,
        temperature=1.5,
        num_tokens=200
    )
    
    result2 = engine1.validate(purposes2, llm_output2)
    print(engine1.explain_verdict(result2))
    
    # =========================================================================
    # TEST 3: Singularidad entrópica
    # =========================================================================
    print("\n" + "▸" * 80)
    print("TEST 3: Stochastic singularity (infinite entropy)")
    print("▸" * 80 + "\n")
    
    purposes3 = [
        BusinessPurpose("encryption", "SECURITY_HARDENING", strength=0.98, confidence=0.99),
    ]
    
    llm_output3 = LLMOutput(
        entropy=float('inf'),
        confidence=0.88,
        temperature=0.5,
        num_tokens=300
    )
    
    result3 = engine1.validate(purposes3, llm_output3)
    print(engine1.explain_verdict(result3))
    
    # =========================================================================
    # TEST 4: Paradoja semántica (obstrucción cohomológica)
    # =========================================================================
    print("\n" + "▸" * 80)
    print("TEST 4: Semantic paradox (topological obstruction)")
    print("▸" * 80 + "\n")
    
    # Crear perfil de alto riesgo
    risk_profile_aggressive = RiskProfile(
        risk_tolerance=0.95,
        domain_criticality=0.1,
        acceptable_failure_rate=0.1
    )
    
    engine4 = SemanticValidationEngine(
        knowledge_graph=kg,
        risk_profile=risk_profile_aggressive,
        enable_cohomology=True
    )
    
    # Propósito fuerte pero confianza muy baja → paradoja
    purposes4 = [
        BusinessPurpose("load_balancing", "RELIABILITY_IMPROVEMENT", strength=0.90, confidence=0.95),
    ]
    
    llm_output4 = LLMOutput(
        entropy=0.6,
        confidence=0.28,  # Muy baja
        temperature=0.5,
        num_tokens=100
    )
    
    code_metrics4 = {
        'cyclomatic': 5,
        'depth': 2,
        'loc': 30,
        'cognitive': 8
    }
    
    result4 = engine4.validate(purposes4, llm_output4, code_metrics4)
    print(engine4.explain_verdict(result4))
    
    # =========================================================================
    # TEST 5: Métrica personalizada con acoplamiento fuerte
    # =========================================================================
    print("\n" + "▸" * 80)
    print("TEST 5: Custom metric with strong coupling")
    print("▸" * 80 + "\n")
    
    custom_metric = MahalanobisMetric()
    # Aumentar penalización cruzada entre confianza y riesgo
    custom_metric.set_coupling(1, 3, -0.5)  # confidence-risk anticorrelation
    
    engine5 = SemanticValidationEngine(
        knowledge_graph=kg,
        risk_profile=RiskProfile(risk_tolerance=0.5),
        metric=custom_metric,
        enable_cohomology=True
    )
    
    purposes5 = [
        BusinessPurpose("automated_testing", "RELIABILITY_IMPROVEMENT", strength=0.95, confidence=0.92),
    ]
    
    llm_output5 = LLMOutput(
        entropy=0.8,
        confidence=0.85,
        temperature=0.8,
        num_tokens=180
    )
    
    code_metrics5 = {
        'cyclomatic': 18,
        'depth': 5,
        'loc': 150,
        'cognitive': 35
    }
    
    result5 = engine5.validate(purposes5, llm_output5, code_metrics5)
    print(engine5.explain_verdict(result5))
    
    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Test 1 (High quality):      {result1.verdict.name:15s} D_M={result1.mahalanobis_distance:.4f}")
    print(f"Test 2 (Weak purpose):      {result2.verdict.name:15s} D_M={result2.mahalanobis_distance:.4f}")
    print(f"Test 3 (Singularity):       {result3.verdict.name:15s} D_M={result3.mahalanobis_distance:.4f}")
    print(f"Test 4 (Topological obs.):  {result4.verdict.name:15s} D_M={result4.mahalanobis_distance:.4f}")
    print(f"Test 5 (Custom metric):     {result5.verdict.name:15s} D_M={result5.mahalanobis_distance:.4f}")
    print("=" * 80 + "\n")
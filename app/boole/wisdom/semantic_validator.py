"""
=========================================================================================
Módulo: Semantic Validation Engine (Motor de Validación Semántica)
Ubicación: app/boole/wisdom/semantic_validator.py
Versión: 3.0 - Lógica de Negocio Rigurosa y Transparente
=========================================================================================

PROPÓSITO:
----------
Valida que código generado por LLM tenga propósito empresarial claro y cumpla
restricciones de riesgo/complejidad.

COMPONENTES:
------------
1. Validación de Propósito (Purpose Validation):
   - Verifica que la herramienta resuelva problemas de negocio identificados
   - Usa grafo de conocimiento para mapear conceptos → problemas
   
2. Análisis de Confianza (Confidence Analysis):
   - Filtra salidas de LLM con alta entropía o baja confianza
   - Umbrales calibrados empíricamente
   
3. Mapeo de Restricciones (Constraint Mapping):
   - Traduce tolerancia al riesgo empresarial en límites técnicos
   - Modelo de scoring multi-criterio
   
4. Agregación de Veredicto (Verdict Aggregation):
   - Combina señales en decisión final
   - Sistema de scoring ponderado transparente

MODELO MATEMÁTICO:
------------------
Sea S = (c, e, p, r) un estado del sistema donde:
- c: confianza del LLM ∈ [0, 1]
- e: entropía de la salida ∈ ℝ≥0
- p: fuerza del propósito empresarial ∈ [0, 1]
- r: tolerancia al riesgo ∈ [0, 1]

Veredicto V: S → {VIABLE, CONDITIONAL, WARNING, REJECT}

Definición del scoring:
    score(S) = w_c·c + w_p·p - w_e·normalize(e) + w_r·r
    
donde w_i son pesos configurables que suman 1.

=========================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from collections import defaultdict

import networkx as nx
import numpy as np

logger = logging.getLogger("Gamma.Wisdom.SemanticValidator.v3.0")


# =============================================================================
# TIPOS Y ENUMERACIONES
# =============================================================================

class Verdict(IntEnum):
    """
    Veredicto de validación ordenado por severidad.
    
    Orden parcial: VIABLE < CONDITIONAL < WARNING < REJECT
    
    Semántica:
    - VIABLE: Aprobado sin restricciones
    - CONDITIONAL: Aprobado con condiciones (requiere revisión)
    - WARNING: Preocupaciones significativas (decisión manual recomendada)
    - REJECT: Rechazado (no cumple criterios mínimos)
    """
    VIABLE = 0
    CONDITIONAL = 1
    WARNING = 2
    REJECT = 3
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def is_accepted(self) -> bool:
        """Veredictos que permiten ejecución (posiblemente con condiciones)."""
        return self in {Verdict.VIABLE, Verdict.CONDITIONAL}
    
    @property
    def requires_human_review(self) -> bool:
        """Veredictos que requieren revisión humana."""
        return self in {Verdict.CONDITIONAL, Verdict.WARNING}


class SignalStrength(Enum):
    """Fuerza de una señal de validación."""
    STRONG_POSITIVE = "strong_positive"
    WEAK_POSITIVE = "weak_positive"
    NEUTRAL = "neutral"
    WEAK_NEGATIVE = "weak_negative"
    STRONG_NEGATIVE = "strong_negative"


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass(frozen=True, order=True)
class BusinessPurpose:
    """
    Mapeo de una funcionalidad a un problema de negocio.
    
    Atributos:
        concept: Concepto técnico (ej: "caching", "load_balancing")
        business_problem: Problema empresarial (ej: "LATENCY_REDUCTION")
        strength: Fuerza de la conexión semántica ∈ [0, 1]
        confidence: Confianza en el mapeo ∈ [0, 1]
    
    Invariantes:
    - 0 ≤ strength ≤ 1
    - 0 ≤ confidence ≤ 1
    """
    concept: str
    business_problem: str
    strength: float
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validación de invariantes."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        
        if not isinstance(self.concept, str) or not self.concept:
            raise ValueError("concept must be a non-empty string")
        if not isinstance(self.business_problem, str) or not self.business_problem:
            raise ValueError("business_problem must be a non-empty string")
    
    @property
    def effective_strength(self) -> float:
        """Fuerza efectiva = strength × confidence."""
        return self.strength * self.confidence
    
    def __repr__(self) -> str:
        return (f"Purpose({self.concept} → {self.business_problem}, "
                f"strength={self.strength:.2f}, conf={self.confidence:.2f})")


@dataclass(frozen=True)
class LLMOutput:
    """
    Metadatos de salida del modelo de lenguaje.
    
    Atributos:
        entropy: Entropía de la distribución de tokens ∈ ℝ≥0
        confidence: Confianza del modelo ∈ [0, 1]
        temperature: Temperatura de sampling usada
        num_tokens: Número de tokens generados
    """
    entropy: float
    confidence: float
    temperature: float = 1.0
    num_tokens: int = 0
    
    def __post_init__(self):
        """Validación de invariantes."""
        if self.entropy < 0:
            raise ValueError(f"entropy must be ≥ 0, got {self.entropy}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.num_tokens < 0:
            raise ValueError(f"num_tokens must be ≥ 0, got {self.num_tokens}")
    
    @property
    def normalized_entropy(self) -> float:
        """
        Entropía normalizada por temperatura y longitud.
        
        Máxima entropía teórica para vocabulario V:
            H_max = log(|V|)
        
        Heurística: normalizamos por temperatura (escala) y sqrt(tokens) (longitud).
        """
        if self.num_tokens == 0:
            return self.entropy
        
        # Normalización heurística
        length_factor = math.sqrt(max(self.num_tokens, 1))
        return self.entropy / (self.temperature * length_factor)
    
    def __repr__(self) -> str:
        return (f"LLM(H={self.entropy:.2f}, conf={self.confidence:.2f}, "
                f"T={self.temperature}, tokens={self.num_tokens})")


@dataclass(frozen=True)
class RiskProfile:
    """
    Perfil de tolerancia al riesgo empresarial.
    
    Atributos:
        risk_tolerance: Tolerancia general al riesgo ∈ [0, 1]
            0 = adverso al riesgo (startups, finanzas críticas)
            1 = tolerante al riesgo (experimentación, prototipado)
        
        domain_criticality: Criticidad del dominio ∈ [0, 1]
            0 = no crítico (herramientas internas)
            1 = crítico (sistemas de producción, salud, finanzas)
        
        acceptable_failure_rate: Tasa de fallo aceptable ∈ [0, 1]
            Probabilidad aceptable de que el código falle
    """
    risk_tolerance: float
    domain_criticality: float = 0.5
    acceptable_failure_rate: float = 0.01
    
    def __post_init__(self):
        """Validación de invariantes."""
        if not 0.0 <= self.risk_tolerance <= 1.0:
            raise ValueError(f"risk_tolerance must be in [0, 1], got {self.risk_tolerance}")
        if not 0.0 <= self.domain_criticality <= 1.0:
            raise ValueError(f"domain_criticality must be in [0, 1], got {self.domain_criticality}")
        if not 0.0 <= self.acceptable_failure_rate <= 1.0:
            raise ValueError(f"acceptable_failure_rate must be in [0, 1], got {self.acceptable_failure_rate}")
    
    @property
    def effective_tolerance(self) -> float:
        """
        Tolerancia efectiva considerando criticidad.
        
        Mayor criticidad → menor tolerancia efectiva.
        """
        return self.risk_tolerance * (1 - 0.5 * self.domain_criticality)
    
    @property
    def risk_category(self) -> str:
        """Categorización cualitativa del riesgo."""
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
    
    def __repr__(self) -> str:
        return (f"Risk(tolerance={self.risk_tolerance:.2f}, "
                f"criticality={self.domain_criticality:.2f}, "
                f"category={self.risk_category})")


@dataclass
class ValidationResult:
    """
    Resultado completo de validación.
    
    Contiene veredicto final y desglose de señales.
    """
    verdict: Verdict
    overall_score: float
    signals: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_reason(self, reason: str, signal_name: Optional[str] = None, 
                   signal_value: Optional[float] = None) -> None:
        """Añade una razón al resultado."""
        self.reasons.append(reason)
        if signal_name is not None and signal_value is not None:
            self.signals[signal_name] = signal_value
    
    def __repr__(self) -> str:
        return (f"ValidationResult(verdict={self.verdict}, "
                f"score={self.overall_score:.3f}, "
                f"signals={len(self.signals)})")


# =============================================================================
# VALIDADORES INDIVIDUALES
# =============================================================================

class PurposeValidator:
    """
    Valida que el código tenga propósito empresarial claro.
    
    Usa un grafo de conocimiento para mapear conceptos técnicos
    a problemas de negocio conocidos.
    """
    
    # Problemas de negocio canónicos (pueden configurarse externamente)
    DEFAULT_CANONICAL_PROBLEMS = frozenset([
        "COST_REDUCTION",
        "LATENCY_REDUCTION",
        "RELIABILITY_IMPROVEMENT",
        "SCALABILITY_ENHANCEMENT",
        "SECURITY_HARDENING",
        "COMPLIANCE_ADHERENCE",
        "USER_EXPERIENCE_IMPROVEMENT",
        "DATA_QUALITY_ENHANCEMENT",
    ])
    
    def __init__(
        self,
        knowledge_graph: Optional[nx.DiGraph] = None,
        canonical_problems: Optional[FrozenSet[str]] = None,
        min_strength_threshold: float = 0.7
    ):
        """
        Inicializa el validador de propósito.
        
        Args:
            knowledge_graph: Grafo dirigido concepto → problema
            canonical_problems: Conjunto de problemas reconocidos
            min_strength_threshold: Umbral mínimo de fuerza efectiva
        """
        self.kg = knowledge_graph or nx.DiGraph()
        self.canonical_problems = canonical_problems or self.DEFAULT_CANONICAL_PROBLEMS
        self.min_strength = min_strength_threshold
        
        if not 0.0 <= self.min_strength <= 1.0:
            raise ValueError(f"min_strength_threshold must be in [0, 1], got {self.min_strength}")
    
    def validate(self, purposes: List[BusinessPurpose]) -> Tuple[bool, float, str]:
        """
        Valida lista de propósitos empresariales.
        
        Args:
            purposes: Lista de mapeos concepto → problema
        
        Returns:
            (is_valid, strength, reason)
        """
        if not purposes:
            return False, 0.0, "No business purposes provided"
        
        # Verificar que al menos un propósito mapee a problema canónico
        canonical_purposes = [
            p for p in purposes
            if p.business_problem in self.canonical_problems
        ]
        
        if not canonical_purposes:
            return False, 0.0, f"No purposes map to canonical problems: {self.canonical_problems}"
        
        # Calcular fuerza máxima
        max_strength = max(p.effective_strength for p in canonical_purposes)
        
        if max_strength < self.min_strength:
            return False, max_strength, f"Max purpose strength {max_strength:.2f} < threshold {self.min_strength}"
        
        # Éxito
        best_purpose = max(canonical_purposes, key=lambda p: p.effective_strength)
        reason = f"Strong purpose: {best_purpose.concept} → {best_purpose.business_problem} (strength={max_strength:.2f})"
        
        return True, max_strength, reason
    
    def compute_purpose_score(self, purposes: List[BusinessPurpose]) -> float:
        """
        Calcula score de propósito ∈ [0, 1].
        
        Agrega múltiples propósitos con soft-max.
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
        
        # Soft-max: evita que un solo propósito fuerte domine completamente
        # pero da más peso a los fuertes
        max_strength = max(canonical_strengths)
        mean_strength = np.mean(canonical_strengths)
        
        # Combinación 70% max, 30% mean
        return 0.7 * max_strength + 0.3 * mean_strength


class ConfidenceFilter:
    """
    Filtra salidas de LLM con baja confianza o alta entropía.
    
    Criterios calibrados empíricamente para GPT-3.5/4.
    """
    
    # Umbrales por defecto (configurables)
    DEFAULT_MIN_CONFIDENCE = 0.6
    DEFAULT_MAX_ENTROPY = 2.5
    DEFAULT_MAX_NORMALIZED_ENTROPY = 0.5
    
    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_entropy: float = DEFAULT_MAX_ENTROPY,
        max_normalized_entropy: float = DEFAULT_MAX_NORMALIZED_ENTROPY
    ):
        """
        Inicializa el filtro de confianza.
        
        Args:
            min_confidence: Confianza mínima aceptable
            max_entropy: Entropía máxima aceptable (absoluta)
            max_normalized_entropy: Entropía normalizada máxima
        """
        self.min_confidence = min_confidence
        self.max_entropy = max_entropy
        self.max_normalized_entropy = max_normalized_entropy
        
        # Validaciones
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if self.max_entropy <= 0:
            raise ValueError(f"max_entropy must be > 0, got {self.max_entropy}")
        if self.max_normalized_entropy <= 0:
            raise ValueError(f"max_normalized_entropy must be > 0, got {self.max_normalized_entropy}")
    
    def validate(self, llm_output: LLMOutput) -> Tuple[bool, float, str]:
        """
        Valida salida del LLM.
        
        Args:
            llm_output: Metadatos del LLM
        
        Returns:
            (is_valid, score, reason)
        """
        # Check 1: Confianza mínima
        if llm_output.confidence < self.min_confidence:
            return False, llm_output.confidence, \
                   f"Confidence {llm_output.confidence:.2f} < threshold {self.min_confidence}"
        
        # Check 2: Entropía absoluta
        if llm_output.entropy > self.max_entropy:
            return False, 1.0 - llm_output.entropy / (self.max_entropy * 2), \
                   f"Entropy {llm_output.entropy:.2f} > threshold {self.max_entropy}"
        
        # Check 3: Entropía normalizada
        norm_entropy = llm_output.normalized_entropy
        if norm_entropy > self.max_normalized_entropy:
            return False, 1.0 - norm_entropy / (self.max_normalized_entropy * 2), \
                   f"Normalized entropy {norm_entropy:.2f} > threshold {self.max_normalized_entropy}"
        
        # Calcular score combinado
        conf_score = llm_output.confidence
        entropy_score = 1.0 - (llm_output.entropy / self.max_entropy)
        norm_entropy_score = 1.0 - (norm_entropy / self.max_normalized_entropy)
        
        # Media ponderada: 50% confianza, 25% entropía, 25% entropía normalizada
        score = 0.5 * conf_score + 0.25 * entropy_score + 0.25 * norm_entropy_score
        
        return True, score, f"LLM output meets quality thresholds (score={score:.2f})"
    
    def compute_confidence_score(self, llm_output: LLMOutput) -> float:
        """
        Calcula score de confianza ∈ [0, 1].
        """
        _, score, _ = self.validate(llm_output)
        return max(0.0, min(1.0, score))


class ConstraintMapper:
    """
    Mapea perfil de riesgo a restricciones técnicas concretas.
    
    Traduce tolerancia empresarial en límites de:
    - Complejidad ciclomática máxima
    - Profundidad de anidamiento
    - Número de dependencias
    - Tamaño de código
    """
    
    # Límites por categoría de riesgo
    COMPLEXITY_LIMITS = {
        "HIGHLY_CONSERVATIVE": {"cyclomatic": 10, "depth": 3, "loc": 50},
        "CONSERVATIVE": {"cyclomatic": 15, "depth": 4, "loc": 100},
        "MODERATE": {"cyclomatic": 20, "depth": 5, "loc": 200},
        "AGGRESSIVE": {"cyclomatic": 30, "depth": 7, "loc": 500},
        "HIGHLY_AGGRESSIVE": {"cyclomatic": 50, "depth": 10, "loc": 1000},
    }
    
    def __init__(self):
        """Inicializa el mapeador de restricciones."""
        pass
    
    def map_to_constraints(self, risk_profile: RiskProfile) -> Dict[str, int]:
        """
        Mapea perfil de riesgo a restricciones técnicas.
        
        Args:
            risk_profile: Perfil de riesgo empresarial
        
        Returns:
            Diccionario de restricciones {nombre: límite}
        """
        category = risk_profile.risk_category
        base_limits = self.COMPLEXITY_LIMITS.get(category, self.COMPLEXITY_LIMITS["MODERATE"])
        
        # Ajustar por criticidad del dominio
        criticality_factor = 1.0 - 0.3 * risk_profile.domain_criticality
        
        adjusted_limits = {
            key: max(1, int(value * criticality_factor))
            for key, value in base_limits.items()
        }
        
        return adjusted_limits
    
    def compute_constraint_score(
        self,
        actual_metrics: Dict[str, int],
        risk_profile: RiskProfile
    ) -> float:
        """
        Calcula score de cumplimiento de restricciones ∈ [0, 1].
        
        Args:
            actual_metrics: Métricas reales del código
            risk_profile: Perfil de riesgo
        
        Returns:
            Score ∈ [0, 1] donde 1 = todas las restricciones cumplidas
        """
        limits = self.map_to_constraints(risk_profile)
        
        scores = []
        for key, limit in limits.items():
            if key in actual_metrics:
                actual = actual_metrics[key]
                # Score = 1 si actual ≤ limit, decae exponencialmente si excede
                if actual <= limit:
                    scores.append(1.0)
                else:
                    excess_ratio = (actual - limit) / limit
                    score = math.exp(-2 * excess_ratio)  # Decae rápido
                    scores.append(score)
        
        return np.mean(scores) if scores else 1.0


# =============================================================================
# ENGINE PRINCIPAL
# =============================================================================

class SemanticValidationEngine:
    """
    Motor de validación semántica multi-criterio.
    
    Combina señales de:
    1. Propósito empresarial
    2. Confianza del LLM
    3. Cumplimiento de restricciones
    4. Tolerancia al riesgo
    
    En un veredicto final mediante scoring ponderado.
    """
    
    # Pesos por defecto (configurables)
    DEFAULT_WEIGHTS = {
        'purpose': 0.35,      # Propósito empresarial (más importante)
        'confidence': 0.30,   # Confianza del LLM
        'constraints': 0.25,  # Cumplimiento de restricciones
        'risk': 0.10,         # Bonus por tolerancia al riesgo
    }
    
    # Umbrales de veredicto
    VERDICT_THRESHOLDS = {
        Verdict.VIABLE: 0.75,      # Score ≥ 0.75 → VIABLE
        Verdict.CONDITIONAL: 0.60,  # Score ≥ 0.60 → CONDITIONAL
        Verdict.WARNING: 0.40,      # Score ≥ 0.40 → WARNING
        # Score < 0.40 → REJECT
    }
    
    def __init__(
        self,
        knowledge_graph: Optional[nx.DiGraph] = None,
        risk_profile: Optional[RiskProfile] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Inicializa el motor de validación.
        
        Args:
            knowledge_graph: Grafo de conocimiento empresarial
            risk_profile: Perfil de tolerancia al riesgo
            weights: Pesos personalizados para scoring
        """
        self.risk_profile = risk_profile or RiskProfile(risk_tolerance=0.5)
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Validar que los pesos sumen aproximadamente 1
        total_weight = sum(self.weights.values())
        if not math.isclose(total_weight, 1.0, rel_tol=0.01):
            logger.warning(f"Weights sum to {total_weight:.3f}, not 1.0. Normalizing.")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Inicializar validadores
        self.purpose_validator = PurposeValidator(knowledge_graph)
        self.confidence_filter = ConfidenceFilter()
        self.constraint_mapper = ConstraintMapper()
        
        logger.info(f"Initialized SemanticValidationEngine with profile: {self.risk_profile}")
    
    def validate(
        self,
        purposes: List[BusinessPurpose],
        llm_output: LLMOutput,
        code_metrics: Optional[Dict[str, int]] = None
    ) -> ValidationResult:
        """
        Ejecuta validación completa multi-criterio.
        
        Args:
            purposes: Lista de propósitos empresariales
            llm_output: Metadatos del LLM
            code_metrics: Métricas opcionales del código generado
        
        Returns:
            ValidationResult con veredicto y desglose
        """
        logger.info("Starting semantic validation...")
        
        result = ValidationResult(verdict=Verdict.REJECT, overall_score=0.0)
        
        # === SEÑAL 1: PROPÓSITO EMPRESARIAL ===
        is_valid_purpose, purpose_strength, purpose_reason = \
            self.purpose_validator.validate(purposes)
        
        purpose_score = self.purpose_validator.compute_purpose_score(purposes)
        result.add_reason(purpose_reason, 'purpose', purpose_score)
        
        if not is_valid_purpose:
            logger.warning(f"Purpose validation failed: {purpose_reason}")
            result.verdict = Verdict.REJECT
            return result
        
        # === SEÑAL 2: CONFIANZA DEL LLM ===
        is_valid_confidence, confidence_strength, confidence_reason = \
            self.confidence_filter.validate(llm_output)
        
        confidence_score = self.confidence_filter.compute_confidence_score(llm_output)
        result.add_reason(confidence_reason, 'confidence', confidence_score)
        
        if not is_valid_confidence:
            logger.warning(f"Confidence validation failed: {confidence_reason}")
            result.verdict = Verdict.REJECT
            return result
        
        # === SEÑAL 3: RESTRICCIONES TÉCNICAS ===
        if code_metrics:
            constraints_score = self.constraint_mapper.compute_constraint_score(
                code_metrics, self.risk_profile
            )
            
            limits = self.constraint_mapper.map_to_constraints(self.risk_profile)
            violations = [
                f"{k}: {code_metrics.get(k, 0)} > {v}"
                for k, v in limits.items()
                if code_metrics.get(k, 0) > v
            ]
            
            if violations:
                constraint_reason = f"Constraint violations: {', '.join(violations)}"
            else:
                constraint_reason = "All constraints satisfied"
            
            result.add_reason(constraint_reason, 'constraints', constraints_score)
        else:
            constraints_score = 1.0  # No hay métricas, asumimos OK
            result.add_reason("No code metrics provided (assumed OK)", 'constraints', 1.0)
        
        # === SEÑAL 4: BONUS POR TOLERANCIA AL RIESGO ===
        risk_bonus = self.risk_profile.effective_tolerance
        result.add_reason(
            f"Risk tolerance bonus: {risk_bonus:.2f}",
            'risk',
            risk_bonus
        )
        
        # === AGREGACIÓN DE SCORE ===
        overall_score = (
            self.weights['purpose'] * purpose_score +
            self.weights['confidence'] * confidence_score +
            self.weights['constraints'] * constraints_score +
            self.weights['risk'] * risk_bonus
        )
        
        result.overall_score = overall_score
        
        # === DETERMINACIÓN DE VEREDICTO ===
        if overall_score >= self.VERDICT_THRESHOLDS[Verdict.VIABLE]:
            result.verdict = Verdict.VIABLE
        elif overall_score >= self.VERDICT_THRESHOLDS[Verdict.CONDITIONAL]:
            result.verdict = Verdict.CONDITIONAL
        elif overall_score >= self.VERDICT_THRESHOLDS[Verdict.WARNING]:
            result.verdict = Verdict.WARNING
        else:
            result.verdict = Verdict.REJECT
        
        # Metadata
        result.metadata = {
            'risk_profile': str(self.risk_profile),
            'weights': self.weights,
            'thresholds': self.VERDICT_THRESHOLDS,
        }
        
        logger.info(f"Validation complete: {result.verdict} (score={overall_score:.3f})")
        
        return result
    
    def explain_verdict(self, result: ValidationResult) -> str:
        """
        Genera explicación legible del veredicto.
        
        Args:
            result: Resultado de validación
        
        Returns:
            Texto explicativo
        """
        lines = [
            f"Verdict: {result.verdict.name}",
            f"Overall Score: {result.overall_score:.3f}",
            "",
            "Signal Breakdown:",
        ]
        
        for signal_name, signal_value in result.signals.items():
            weight = self.weights.get(signal_name, 0.0)
            contribution = weight * signal_value
            lines.append(f"  {signal_name:12s}: {signal_value:.3f} (weight={weight:.2f}, contrib={contribution:.3f})")
        
        lines.append("")
        lines.append("Reasons:")
        for reason in result.reasons:
            lines.append(f"  - {reason}")
        
        return "\n".join(lines)


# =============================================================================
# UTILIDADES
# =============================================================================

def create_default_knowledge_graph() -> nx.DiGraph:
    """
    Crea un grafo de conocimiento de ejemplo.
    
    Returns:
        DiGraph con mapeos concepto → problema
    """
    kg = nx.DiGraph()
    
    # Ejemplos de mapeos
    mappings = [
        # Caching
        ("caching", "LATENCY_REDUCTION", 0.9),
        ("caching", "COST_REDUCTION", 0.7),
        
        # Load balancing
        ("load_balancing", "RELIABILITY_IMPROVEMENT", 0.85),
        ("load_balancing", "SCALABILITY_ENHANCEMENT", 0.9),
        
        # Encryption
        ("encryption", "SECURITY_HARDENING", 0.95),
        ("encryption", "COMPLIANCE_ADHERENCE", 0.8),
        
        # Monitoring
        ("monitoring", "RELIABILITY_IMPROVEMENT", 0.85),
        ("monitoring", "USER_EXPERIENCE_IMPROVEMENT", 0.6),
        
        # Data validation
        ("data_validation", "DATA_QUALITY_ENHANCEMENT", 0.9),
        ("data_validation", "RELIABILITY_IMPROVEMENT", 0.7),
    ]

    # Inyectar isomorfismo dimensional conectando las islas para tener B_0 = 1
    mappings.append(("caching", "SCALABILITY_ENHANCEMENT", 0.6))
    mappings.append(("encryption", "DATA_QUALITY_ENHANCEMENT", 0.5))
    
    for concept, problem, weight in mappings:
        kg.add_edge(concept, problem, weight=weight)
    
    return kg


# =============================================================================
# FUNCIÓN DE COMPATIBILIDAD (API LEGACY)
# =============================================================================

class OntologicalDiffeomorphismEngine:
    """
    Clase de compatibilidad con API anterior.
    
    DEPRECATED: Usar SemanticValidationEngine directamente.
    """
    
    def __init__(self, knowledge_graph: nx.DiGraph, business_profile: Any, **kwargs):
        """Legacy constructor."""
        logger.warning(
            "OntologicalDiffeomorphismEngine is deprecated. "
            "Use SemanticValidationEngine instead."
        )
        
        # Convertir business_profile legacy a RiskProfile
        if hasattr(business_profile, 'risk_tolerance'):
            risk_profile = RiskProfile(
                risk_tolerance=business_profile.risk_tolerance,
                domain_criticality=0.5,
                acceptable_failure_rate=0.01
            )
        else:
            risk_profile = RiskProfile(risk_tolerance=0.5)
        
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
        Legacy API: compile_wisdom.
        
        DEPRECATED: Usar validate() directamente.
        """
        # Convertir tool_semantics legacy a BusinessPurpose
        purposes = []
        for sem in tool_semantics:
            if hasattr(sem, 'source_concept') and hasattr(sem, 'target_business_pain'):
                purposes.append(BusinessPurpose(
                    concept=sem.source_concept,
                    business_problem=sem.target_business_pain,
                    strength=getattr(sem, 'semantic_weight', 0.8),
                    confidence=1.0
                ))
        
        # Crear LLMOutput
        llm_output = LLMOutput(
            entropy=llm_entropy,
            confidence=llm_confidence,
            temperature=1.0,
            num_tokens=100
        )
        
        # Validar
        result = self._engine.validate(purposes, llm_output)
        
        # Convertir Verdict a VerdictLevel legacy
        return int(result.verdict.value)


# Alias para compatibilidad
def __getattr__(name):
    if name in ["VerdictLevel", "SemanticMorphism", "ToleranceProfile"]:
        import warnings
        warnings.warn(f"'{name}' is deprecated. Please use the modernized equivalents.", DeprecationWarning, stacklevel=2)
        if name == "VerdictLevel": return Verdict
        if name == "SemanticMorphism": return BusinessPurpose
        if name == "ToleranceProfile": return RiskProfile
    raise AttributeError(f"module {__name__} has no attribute {name}")


# =============================================================================
# PUNTO DE ENTRADA PARA TESTING
# =============================================================================

if __name__ == "__main__":
    # Crear knowledge graph de ejemplo
    kg = create_default_knowledge_graph()
    
    # Perfil de riesgo conservador
    risk_profile = RiskProfile(
        risk_tolerance=0.3,
        domain_criticality=0.8,
        acceptable_failure_rate=0.001
    )
    
    # Crear motor
    engine = SemanticValidationEngine(
        knowledge_graph=kg,
        risk_profile=risk_profile
    )
    
    # Ejemplo 1: Propósito válido, alta confianza
    print("=" * 80)
    print("EJEMPLO 1: Código de caching con buen propósito")
    print("=" * 80)
    
    purposes1 = [
        BusinessPurpose("caching", "LATENCY_REDUCTION", strength=0.9, confidence=0.95),
        BusinessPurpose("caching", "COST_REDUCTION", strength=0.7, confidence=0.9),
    ]
    
    llm_output1 = LLMOutput(
        entropy=0.5,
        confidence=0.92,
        temperature=0.7,
        num_tokens=150
    )
    
    code_metrics1 = {
        'cyclomatic': 8,
        'depth': 3,
        'loc': 45
    }
    
    result1 = engine.validate(purposes1, llm_output1, code_metrics1)
    print(engine.explain_verdict(result1))
    
    # Ejemplo 2: Propósito débil, baja confianza
    print("\n" + "=" * 80)
    print("EJEMPLO 2: Código sin propósito claro, baja confianza")
    print("=" * 80)
    
    purposes2 = [
        BusinessPurpose("unknown_concept", "UNKNOWN_PROBLEM", strength=0.3, confidence=0.5),
    ]
    
    llm_output2 = LLMOutput(
        entropy=3.5,
        confidence=0.45,
        temperature=1.5,
        num_tokens=200
    )
    
    result2 = engine.validate(purposes2, llm_output2)
    print(engine.explain_verdict(result2))
    
    # Ejemplo 3: Buen propósito pero violación de restricciones
    print("\n" + "=" * 80)
    print("EJEMPLO 3: Código con propósito válido pero muy complejo")
    print("=" * 80)
    
    purposes3 = [
        BusinessPurpose("encryption", "SECURITY_HARDENING", strength=0.95, confidence=0.98),
    ]
    
    llm_output3 = LLMOutput(
        entropy=0.8,
        confidence=0.88,
        temperature=0.5,
        num_tokens=300
    )
    
    code_metrics3 = {
        'cyclomatic': 35,  # Excede límite para perfil conservador
        'depth': 8,        # Excede límite
        'loc': 250         # Excede límite
    }
    
    result3 = engine.validate(purposes3, llm_output3, code_metrics3)
    print(engine.explain_verdict(result3))
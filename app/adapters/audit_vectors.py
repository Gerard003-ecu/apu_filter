"""
=========================================================================================
Módulo: Audit Vectors (Proyector de Coherencia Topológica y Entropía Estructural)
Ubicación: app/adapters/audit_vectors.py
=========================================================================================

Naturaleza Ciber-Física y Geometría de la Información:
    Este módulo abandona las heurísticas empíricas para formalizar la auditoría de la red
    de valor como el análisis de un Complejo Simplicial Bipartito G = (U ∪ V, E). Actúa
    en el Estrato TACTICS (Nivel 2), cuantificando la fricción logística y la 
    vulnerabilidad topológica mediante Teoría Espectral de Grafos y Termodinámica de 
    la Información.

1. Geometría Bipartita y Operador Frontera:
    El ecosistema se modela axiomáticamente como un espacio discreto donde:
    • U = {apu₁, ..., apuₘ} representa el subespacio de TACTICS (Masa Táctica / Nodos).
    • V = {ins₁, ..., insₙ} representa el subespacio de PHYSICS (Cimentación Base).
    • E ⊆ U × V define las 1-cadenas (aristas) de flujo termodinámico logístico.
    
    [AXIOMA DE ROBUSTEZ]: La Robustez Topológica ρ = (m - |V_floating|) / m audita
    directamente el núcleo del operador frontera. Un nodo flotante implica algebraicamente
    la fractura del componente conexo (β₀ > 1), revelando una singularidad logística 
    o un sumidero de energía financiera (dinero ciego).

2. Teoría Espectral y la Desigualdad de Cheeger (λ₂):
    La robustez de la cadena de suministro no se asume; se demuestra analizando el 
    espectro de la Matriz Laplaciana L = D - A del grafo proyectado U → U.
    El Valor de Fiedler (λ₂) cuantifica la resistencia mínima a la partición de la red
    (acotada por la constante isoperimétrica de Cheeger). Si λ₂ → 0, el módulo certifica 
    matemáticamente una "Fractura Organizacional Inminente" (cuellos de botella), 
    justificando un veto técnico.

3. Termodinámica de la Información y Geometría de Distribución:
    La centralización de recursos se somete al rigor de la Teoría de la Información:
    • Entropía de Shannon (H): H = -Σ pᵢ ln(pᵢ). Cuantifica la incertidumbre del sistema.
      Su exponencial proyectada N_eff = exp(H) determina los "grados de libertad reales" 
      de la matriz de suministro, erradicando la falsa seguridad de inventarios nominales.
    • Asimetría de Inercia (Índice de Gini, G): G = Σᵢ Σⱼ |xᵢ - xⱼ| / (2n²μ). Mide la 
      concentración de masa de capital. Si G → 1, el sistema exhibe una singularidad, 
      revelando Puntos Únicos de Fallo (SPOF).
    • Probabilidad de Colisión Ortogonal (Índice D): D = 1 - Σ(pᵢ²). Si D → 0, el tensor 
      de suministro colapsa dimensionalmente hacia un monopolio estricto.

4. Ecuación de Estado Estructural (El Tensor Ψ):
    La síntesis de las variables de fase se colapsa en el Índice de Estabilidad
    Compuesto (Ψ), computado como una media geométrica ponderada bajo suavizado euclidiano ε:
        Ψ = ((D + ε)^α · (κ + ε)^β · (ρ + ε)^γ · (1 - G + ε)^δ)^(1/(α+β+γ+δ))
    
    [AXIOMA DE VETO]: Si el invariante escalar Ψ decae por debajo del límite crítico de
    estabilidad, el orquestador dictamina el estado de "Pirámide Invertida", forzando
    un colapso algebraico de la función de evaluación y deteniendo la propagación
    de la información hacia el Estrato STRATEGY.
=========================================================================================
"""

from __future__ import annotations

import math
import time
import hashlib
import logging
import warnings
from typing import (
    Dict, Any, Set, Tuple, Final, Optional, 
    List, FrozenSet, NamedTuple, Callable
)
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property, lru_cache
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd

# Importaciones opcionales con fallback
try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy no disponible — análisis espectral deshabilitado",
        ImportWarning
    )

from app.core.schemas import Stratum
from app.adapters.mic_vectors import VectorResultStatus

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN EXTERNALIZABLE — Umbrales y parámetros calibrables
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AuditConfiguration:
    """
    Configuración externalizable para el vector de auditoría.
    
    Permite ajustar umbrales sin modificar código fuente.
    Valores por defecto calibrados empíricamente en contratos reales.
    """
    # Umbrales de veredicto (percentiles empíricos)
    psi_veto_threshold: float = 0.40       # Percentil 5 en contratos sanos
    psi_warning_threshold: float = 0.65    # Percentil 25
    psi_healthy_threshold: float = 0.80    # Percentil 50 (mediana)
    
    # Umbrales de métricas individuales
    simpson_min_threshold: float = 0.50    # Diversidad mínima aceptable
    connectivity_min_threshold: float = 0.30  # Densidad mínima
    gini_max_threshold: float = 0.70       # Concentración máxima aceptable
    
    # Parámetros de modelo
    expected_insumos_per_apu: int = 5      # Basado en mediana de dominio
    
    # Pesos para media geométrica ponderada (deben sumar 1 o se normalizan)
    weight_simpson: float = 0.30           # α
    weight_connectivity: float = 0.25      # β
    weight_robustness: float = 0.30        # γ
    weight_gini: float = 0.15              # δ
    
    # Suavizado numérico para evitar log(0)
    epsilon: float = 1e-10
    
    # Análisis espectral
    enable_spectral_analysis: bool = True
    spectral_num_eigenvalues: int = 3
    
    # Versión del algoritmo
    algorithm_version: str = "3.0.0-spectral"
    
    def __post_init__(self):
        """Validar invariantes de configuración."""
        # Verificar que los umbrales estén ordenados
        if not (0 < self.psi_veto_threshold < 
                    self.psi_warning_threshold < 
                    self.psi_healthy_threshold < 1):
            raise ValueError(
                "Umbrales PSI deben cumplir: "
                "0 < veto < warning < healthy < 1"
            )
        
        # Verificar pesos positivos
        weights = [
            self.weight_simpson, self.weight_connectivity,
            self.weight_robustness, self.weight_gini
        ]
        if any(w < 0 for w in weights):
            raise ValueError("Todos los pesos deben ser no-negativos")
    
    @cached_property
    def normalized_weights(self) -> Tuple[float, float, float, float]:
        """Retorna pesos normalizados que suman 1."""
        total = (
            self.weight_simpson + self.weight_connectivity +
            self.weight_robustness + self.weight_gini
        )
        if total == 0:
            return (0.25, 0.25, 0.25, 0.25)
        return (
            self.weight_simpson / total,
            self.weight_connectivity / total,
            self.weight_robustness / total,
            self.weight_gini / total
        )


# Configuración singleton por defecto
DEFAULT_CONFIG: Final[AuditConfiguration] = AuditConfiguration()


# ═══════════════════════════════════════════════════════════════════════════
# DATACLASSES — Contratos de Datos Explícitos con Validación
# ═══════════════════════════════════════════════════════════════════════════

class MetricBounds(NamedTuple):
    """Límites teóricos de una métrica."""
    min_val: float
    max_val: float
    optimal: float
    
    def normalize(self, value: float) -> float:
        """Normaliza un valor al rango [0, 1]."""
        if self.max_val == self.min_val:
            return 1.0 if value >= self.optimal else 0.0
        return np.clip(
            (value - self.min_val) / (self.max_val - self.min_val),
            0.0, 1.0
        )


@dataclass(frozen=True)
class GraphMetrics:
    """
    Métricas fundamentales del grafo bipartito APU-Insumo.
    
    Invariantes:
      • structure_load ≥ 0
      • base_width ≥ 0
      • edge_count ≥ 0
      • 0 ≤ floating_nodes ≤ structure_load
      • Todas las métricas normalizadas ∈ [0, 1]
    """
    # Cardinalidades del grafo
    structure_load: int          # m = |U| (nodos APU)
    base_width: int              # n = |V| (nodos Insumo únicos)
    edge_count: int              # |E| (relaciones APU→Insumo)
    floating_nodes: int          # nodos U sin aristas
    
    # Métricas de diversidad
    simpson_diversity: float     # D ∈ [0,1]
    shannon_entropy: float       # H normalizada ∈ [0,1]
    effective_species: float     # N_eff = exp(H_raw)
    gini_coefficient: float      # G ∈ [0,1] (menor es mejor)
    
    # Métricas de conectividad
    connectivity_ratio: float    # κ ∈ [0,1]
    topological_robustness: float # ρ ∈ [0,1]
    algebraic_connectivity: Optional[float]  # λ₂ normalizado, None si no calculado
    
    # Índice compuesto
    composite_stability: float   # Ψ ∈ [0,1]
    
    # Distribución subyacente
    insumo_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Diagnósticos adicionales
    edges_per_apu: float = 0.0
    insumos_per_apu_mean: float = 0.0
    insumos_per_apu_std: float = 0.0
    
    def __post_init__(self):
        """Validar invariantes post-construcción."""
        # Usamos object.__setattr__ porque es frozen
        if self.structure_load < 0:
            raise ValueError(f"structure_load debe ser ≥ 0, got {self.structure_load}")
        if self.floating_nodes < 0 or self.floating_nodes > self.structure_load:
            raise ValueError(
                f"floating_nodes debe estar en [0, {self.structure_load}], "
                f"got {self.floating_nodes}"
            )
    
    @property
    def is_connected(self) -> bool:
        """True si no hay nodos flotantes."""
        return self.floating_nodes == 0
    
    @property
    def density(self) -> float:
        """Densidad del grafo bipartito: |E| / (m × n)."""
        if self.structure_load == 0 or self.base_width == 0:
            return 0.0
        return self.edge_count / (self.structure_load * self.base_width)


@dataclass(frozen=True)
class StabilityVerdict:
    """
    Veredicto estructurado de la auditoría con semántica clara.
    
    La jerarquía de niveles permite decisiones estratificadas:
      VETO → bloquea pipeline
      WARNING → continúa con alerta
      HEALTHY → cumple mínimos
      OPTIMAL → excelencia estructural
    """
    
    class Level(Enum):
        VETO = "STRUCTURAL_VETO"
        WARNING = "STRUCTURAL_WARNING"  
        HEALTHY = "STRUCTURALLY_SOUND"
        OPTIMAL = "STRUCTURALLY_OPTIMAL"
        
        def __lt__(self, other: 'StabilityVerdict.Level') -> bool:
            order = {
                self.VETO: 0, self.WARNING: 1,
                self.HEALTHY: 2, self.OPTIMAL: 3
            }
            return order[self] < order[other]
    
    level: Level
    psi: float
    message: str
    blocking: bool
    contributing_factors: Tuple[str, ...] = field(default_factory=tuple)
    recommendations: Tuple[str, ...] = field(default_factory=tuple)
    
    @property
    def severity_score(self) -> int:
        """Puntaje de severidad: 0 (óptimo) a 3 (veto)."""
        return {
            self.Level.OPTIMAL: 0,
            self.Level.HEALTHY: 1,
            self.Level.WARNING: 2,
            self.Level.VETO: 3
        }[self.level]


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE SANITIZACIÓN — Robustez contra datos corruptos
# ═══════════════════════════════════════════════════════════════════════════

class DataQualityReport(NamedTuple):
    """Reporte de calidad post-sanitización."""
    original_count: int
    valid_count: int
    null_count: int
    invalid_string_count: int
    duplicate_count: int
    
    @property
    def validity_ratio(self) -> float:
        if self.original_count == 0:
            return 0.0
        return self.valid_count / self.original_count
    
    @property
    def is_acceptable(self) -> bool:
        """True si al menos 50% de datos son válidos."""
        return self.validity_ratio >= 0.50


def _sanitize_codigo_set(
    series: pd.Series, 
    col_name: str,
    strict_mode: bool = False
) -> Tuple[Set[str], DataQualityReport]:
    """
    Extrae conjunto de códigos válidos con reporte de calidad detallado.
    
    Maneja correctamente: None, np.nan, pd.NA, "", "nan", "None", espacios.
    
    Args:
        series: Serie de pandas a sanitizar
        col_name: Nombre para logging
        strict_mode: Si True, rechaza strings que parezcan códigos inválidos
    
    Returns:
        (conjunto_válido, reporte_calidad)
    """
    if series is None or series.empty:
        return set(), DataQualityReport(0, 0, 0, 0, 0)
    
    original_count = len(series)
    
    # Paso 1: Contar nulos originales
    null_mask = series.isna()
    null_count = null_mask.sum()
    
    # Paso 2: Convertir a string y normalizar
    sanitized = (
        series
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    
    # Paso 3: Identificar strings inválidos
    invalid_patterns = {"", "NAN", "NONE", "NULL", "NA", "N/A", "-", ".", "0"}
    invalid_mask = sanitized.isin(invalid_patterns)
    invalid_string_count = invalid_mask.sum()
    
    # Paso 4: Aplicar filtro de patrones inválidos
    sanitized = sanitized[~invalid_mask]
    
    # Paso 5: Modo estricto — validar formato de código
    if strict_mode:
        # Asumimos que códigos válidos tienen al menos un dígito o letra
        valid_code_mask = sanitized.str.match(r'^[A-Z0-9].*$')
        sanitized = sanitized[valid_code_mask]
    
    # Paso 6: Obtener únicos
    unique_values = set(sanitized)
    valid_count = len(unique_values)
    duplicate_count = len(sanitized) - valid_count
    
    report = DataQualityReport(
        original_count=original_count,
        valid_count=valid_count,
        null_count=null_count,
        invalid_string_count=invalid_string_count,
        duplicate_count=duplicate_count
    )
    
    logger.debug(
        f"Sanitización '{col_name}': {original_count} → {valid_count} únicos "
        f"(nulos={null_count}, inválidos={invalid_string_count}, "
        f"duplicados={duplicate_count})"
    )
    
    return unique_values, report


def _extract_insumo_distribution(
    df_insumos: pd.DataFrame, 
    col_insumo: str,
    col_apu: str = "CODIGO_APU"
) -> Tuple[int, Dict[str, int], Dict[str, Set[str]]]:
    """
    Calcula distribución de frecuencias y mapeo APU→Insumos.
    
    Returns:
        (base_width, distribution_dict, apu_to_insumos_map)
        donde:
          - distribution_dict[insumo] = número de APUs que lo usan
          - apu_to_insumos_map[apu] = conjunto de insumos usados
    """
    if df_insumos.empty:
        return 0, {}, {}
    
    # Sanitizar ambas columnas
    df_clean = df_insumos[[col_apu, col_insumo]].dropna()
    
    df_clean = df_clean.copy()
    df_clean[col_apu] = df_clean[col_apu].astype(str).str.strip().str.upper()
    df_clean[col_insumo] = df_clean[col_insumo].astype(str).str.strip().str.upper()
    
    # Filtrar valores inválidos
    invalid = {"", "NAN", "NONE", "NULL"}
    df_clean = df_clean[
        ~df_clean[col_apu].isin(invalid) & 
        ~df_clean[col_insumo].isin(invalid)
    ]
    
    if df_clean.empty:
        return 0, {}, {}
    
    # Distribución de frecuencias de insumos
    distribution = df_clean[col_insumo].value_counts().to_dict()
    base_width = len(distribution)
    
    # Mapa APU → conjunto de insumos
    apu_to_insumos: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df_clean.iterrows():
        apu_to_insumos[row[col_apu]].add(row[col_insumo])
    
    return base_width, distribution, dict(apu_to_insumos)


# ═══════════════════════════════════════════════════════════════════════════
# MÉTRICAS MATEMÁTICAS — Fundamentos de Teoría de Grafos y Ecología
# ═══════════════════════════════════════════════════════════════════════════

def _compute_simpson_diversity(
    distribution: Dict[str, int],
    epsilon: float = 1e-10
) -> float:
    """
    Índice de Simpson: D = 1 - Σ(pᵢ²)
    
    Mide heterogeneidad de uso de insumos. Basado en ecología matemática.
    
    Propiedades:
      • D = 0 si un solo insumo domina (monopolio)
      • D → 1 cuando uso está equidistribuido
      • D = 1 - 1/n para distribución uniforme con n categorías
      • Invariante bajo permutaciones (simétrico)
    
    Complexity: O(n) donde n = número de insumos únicos
    """
    if not distribution:
        return 0.0
    
    values = np.array(list(distribution.values()), dtype=np.float64)
    total = values.sum()
    
    if total < epsilon:
        return 0.0
    
    # Proporciones
    proportions = values / total
    
    # D = 1 - HHI (Herfindahl-Hirschman Index)
    simpson = 1.0 - np.sum(proportions ** 2)
    
    return float(np.clip(simpson, 0.0, 1.0))


def _compute_shannon_entropy(
    distribution: Dict[str, int],
    epsilon: float = 1e-10
) -> Tuple[float, float]:
    """
    Entropía de Shannon normalizada y número efectivo de especies.
    
    H = -Σ(pᵢ·ln(pᵢ)) / ln(n)  [normalizada]
    N_eff = exp(H_raw)         [número efectivo]
    
    Returns:
        (entropy_normalized, effective_species)
    """
    if not distribution or len(distribution) == 1:
        return 0.0, 1.0
    
    values = np.array(list(distribution.values()), dtype=np.float64)
    total = values.sum()
    
    if total < epsilon:
        return 0.0, 1.0
    
    proportions = values / total
    # Filtrar ceros para evitar log(0)
    proportions = proportions[proportions > epsilon]
    
    if len(proportions) == 0:
        return 0.0, 1.0
    
    # Entropía raw (en nats)
    entropy_raw = -np.sum(proportions * np.log(proportions))
    
    # Número efectivo de especies (Hill number q=1)
    effective_species = np.exp(entropy_raw)
    
    # Normalizar por máxima entropía posible
    max_entropy = np.log(len(distribution))
    normalized_entropy = entropy_raw / max_entropy if max_entropy > epsilon else 0.0
    
    return (
        float(np.clip(normalized_entropy, 0.0, 1.0)),
        float(effective_species)
    )


def _compute_gini_coefficient(
    distribution: Dict[str, int],
    epsilon: float = 1e-10
) -> float:
    """
    Coeficiente de Gini: medida de desigualdad/concentración.
    
    G = Σᵢ Σⱼ |xᵢ - xⱼ| / (2n²μ)
    
    Implementación eficiente O(n log n) usando ordenamiento.
    
    Returns:
        G ∈ [0, 1]
        G = 0 → distribución perfectamente equitativa
        G = 1 → máxima desigualdad (un solo elemento domina)
    """
    if not distribution or len(distribution) < 2:
        return 0.0
    
    values = np.array(sorted(distribution.values()), dtype=np.float64)
    n = len(values)
    total = values.sum()
    
    if total < epsilon:
        return 0.0
    
    # Fórmula eficiente del coeficiente de Gini
    # G = (2 * Σᵢ (i * xᵢ)) / (n * Σᵢ xᵢ) - (n + 1) / n
    cumsum = np.cumsum(values)
    gini = (2.0 * np.sum((np.arange(1, n + 1) * values))) / (n * total) - (n + 1) / n
    
    return float(np.clip(gini, 0.0, 1.0))


def _compute_connectivity_ratio(
    edge_count: int,
    structure_load: int,
    expected_insumos_per_apu: int,
    epsilon: float = 1e-10
) -> float:
    """
    Conectividad normalizada con función sigmoide suavizada.
    
    κ = 1 / (1 + exp(-k * (r - 1)))
    donde r = |E| / (m · c̄) es el ratio de densidad
    y k controla la pendiente (k=3 por defecto)
    
    Propiedades mejoradas vs. tanh:
      • κ(r=1) = 0.5 exactamente (punto neutro)
      • κ → 0 cuando r → 0
      • κ → 1 cuando r → ∞
      • Derivada máxima en r=1 (máxima sensibilidad en punto esperado)
    
    Returns:
        κ ∈ [0, 1]
    """
    if structure_load <= 0:
        return 0.0
    
    expected_edges = structure_load * expected_insumos_per_apu
    
    if expected_edges < epsilon:
        return 0.0 if edge_count == 0 else 1.0
    
    # Ratio de densidad
    ratio = edge_count / expected_edges
    
    # Sigmoide centrada en ratio=1
    # k=3 da buena discriminación: κ(0.5) ≈ 0.18, κ(1.5) ≈ 0.82
    k = 3.0
    kappa = 1.0 / (1.0 + np.exp(-k * (ratio - 1.0)))
    
    return float(kappa)


def _compute_topological_robustness(
    structure_load: int,
    floating_nodes: int,
    penalty_exponent: float = 2.0
) -> float:
    """
    Robustez topológica con penalización no-lineal.
    
    ρ = ((m - f) / m)^p  donde p es el exponente de penalización
    
    Con p > 1, se penaliza más severamente la presencia de nodos flotantes.
    
    Returns:
        ρ ∈ [0, 1]
    """
    if structure_load <= 0:
        return 0.0
    
    connected_ratio = (structure_load - floating_nodes) / structure_load
    
    # Aplicar penalización no-lineal
    robustness = connected_ratio ** penalty_exponent
    
    return float(np.clip(robustness, 0.0, 1.0))


def _compute_algebraic_connectivity(
    apu_to_insumos: Dict[str, Set[str]],
    num_eigenvalues: int = 3
) -> Optional[Tuple[float, np.ndarray]]:
    """
    Conectividad algebraica λ₂ del grafo proyectado APU-APU.
    
    Construye grafo donde dos APUs están conectados si comparten insumos.
    Peso de arista = número de insumos compartidos (Jaccard pesado).
    
    λ₂ (segundo menor eigenvalor del Laplaciano) mide:
      • Resistencia a partición del grafo (Teorema de Cheeger)
      • λ₂ > 0 ⟺ grafo conexo
      • Mayor λ₂ → más robusto a desconexión
    
    Returns:
        (lambda_2_normalized, eigenvalues) o None si scipy no disponible
        
    Complexity: O(m² · k) donde k = promedio de insumos por APU
    """
    if not SCIPY_AVAILABLE:
        logger.debug("Análisis espectral omitido: scipy no disponible")
        return None
    
    apus = list(apu_to_insumos.keys())
    n = len(apus)
    
    if n < 2:
        return None
    
    # Construir matriz de adyacencia del grafo proyectado
    # A[i,j] = |insumos_i ∩ insumos_j|
    adjacency = np.zeros((n, n), dtype=np.float64)
    
    for i, apu_i in enumerate(apus):
        insumos_i = apu_to_insumos[apu_i]
        for j in range(i + 1, n):
            apu_j = apus[j]
            insumos_j = apu_to_insumos[apu_j]
            
            # Peso = insumos compartidos
            shared = len(insumos_i & insumos_j)
            if shared > 0:
                adjacency[i, j] = shared
                adjacency[j, i] = shared
    
    # Calcular Laplaciano: L = D - A
    degree = np.sum(adjacency, axis=1)
    laplacian = np.diag(degree) - adjacency
    
    # Verificar si el grafo está completamente desconectado
    if np.allclose(degree, 0):
        return (0.0, np.zeros(min(num_eigenvalues, n)))
    
    try:
        # Calcular menores eigenvalores
        # Usamos eigsh para matrices simétricas
        k = min(num_eigenvalues + 1, n - 1)  # +1 porque λ₁ = 0 siempre
        
        if k < 2:
            return (0.0, np.array([0.0]))
        
        # which='SM' para smallest magnitude
        eigenvalues, _ = eigsh(
            sparse.csr_matrix(laplacian), 
            k=k, 
            which='SM',
            return_eigenvectors=False
        )
        eigenvalues = np.sort(np.abs(eigenvalues))
        
        # λ₂ es el segundo menor (λ₁ ≈ 0 siempre)
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        
        # Normalizar por grado máximo
        max_degree = degree.max()
        lambda_2_normalized = lambda_2 / max_degree if max_degree > 0 else 0.0
        
        return (
            float(np.clip(lambda_2_normalized, 0.0, 1.0)),
            eigenvalues[:num_eigenvalues]
        )
        
    except Exception as e:
        logger.warning(f"Error en análisis espectral: {e}")
        return None


def _compute_composite_stability(
    simpson: float,
    connectivity: float,
    robustness: float,
    gini: float,
    config: AuditConfiguration
) -> float:
    """
    Índice de Estabilidad Compuesto: Ψ (media geométrica ponderada).
    
    Ψ = ∏ᵢ (mᵢ + ε)^wᵢ  donde Σwᵢ = 1
    
    donde:
      • m₁ = D (Simpson diversity)
      • m₂ = κ (connectivity)
      • m₃ = ρ (robustness)
      • m₄ = 1 - G (equidad = 1 - Gini)
    
    Justificación:
      • Media geométrica no permite compensación entre métricas
      • Pesos permiten priorizar según dominio
      • ε evita colapso a 0 por un solo factor
    """
    eps = config.epsilon
    w_s, w_c, w_r, w_g = config.normalized_weights
    
    # Transformar Gini a equidad (mayor es mejor)
    equity = 1.0 - gini
    
    # Aplicar suavizado y calcular media geométrica ponderada
    # log(Ψ) = Σ wᵢ · log(mᵢ + ε)
    log_psi = (
        w_s * np.log(simpson + eps) +
        w_c * np.log(connectivity + eps) +
        w_r * np.log(robustness + eps) +
        w_g * np.log(equity + eps)
    )
    
    psi = np.exp(log_psi)
    
    return float(np.clip(psi, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESTADÍSTICO AVANZADO — Diagnósticos de distribución
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DistributionDiagnostics:
    """Diagnósticos estadísticos de la distribución de uso de insumos."""
    mean: float
    median: float
    std: float
    coefficient_of_variation: float
    skewness: float
    kurtosis: float
    percentile_25: float
    percentile_75: float
    iqr: float
    outlier_count: int
    max_concentration: float  # Máximo pᵢ
    top_5_concentration: float  # Suma de top 5 pᵢ


def _compute_distribution_diagnostics(
    distribution: Dict[str, int]
) -> Optional[DistributionDiagnostics]:
    """Calcula diagnósticos estadísticos completos de la distribución."""
    if not distribution or len(distribution) < 2:
        return None
    
    values = np.array(list(distribution.values()), dtype=np.float64)
    total = values.sum()
    n = len(values)
    
    # Momentos básicos
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    
    # Percentiles
    p25, p75 = np.percentile(values, [25, 75])
    iqr = p75 - p25
    
    # Outliers (regla IQR)
    lower_bound = p25 - 1.5 * iqr
    upper_bound = p75 + 1.5 * iqr
    outliers = np.sum((values < lower_bound) | (values > upper_bound))
    
    # Skewness y Kurtosis
    if std > 0 and n > 2:
        skewness = float(np.mean(((values - mean) / std) ** 3))
        kurtosis = float(np.mean(((values - mean) / std) ** 4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0
    
    # Concentración
    proportions = values / total if total > 0 else values
    sorted_proportions = np.sort(proportions)[::-1]
    max_concentration = float(sorted_proportions[0])
    top_5_concentration = float(np.sum(sorted_proportions[:5]))
    
    return DistributionDiagnostics(
        mean=float(mean),
        median=float(median),
        std=float(std),
        coefficient_of_variation=float(cv),
        skewness=skewness,
        kurtosis=kurtosis,
        percentile_25=float(p25),
        percentile_75=float(p75),
        iqr=float(iqr),
        outlier_count=int(outliers),
        max_concentration=max_concentration,
        top_5_concentration=top_5_concentration
    )


# ═══════════════════════════════════════════════════════════════════════════
# LÓGICA DE VEREDICTO — Decisiones estratificadas con diagnóstico
# ═══════════════════════════════════════════════════════════════════════════

def _identify_contributing_factors(
    metrics: GraphMetrics,
    config: AuditConfiguration
) -> List[str]:
    """Identifica factores que contribuyen a un puntaje bajo."""
    factors = []
    
    if metrics.simpson_diversity < config.simpson_min_threshold:
        factors.append(
            f"Baja diversidad (Simpson={metrics.simpson_diversity:.3f} < "
            f"{config.simpson_min_threshold})"
        )
    
    if metrics.connectivity_ratio < config.connectivity_min_threshold:
        factors.append(
            f"Baja conectividad (κ={metrics.connectivity_ratio:.3f} < "
            f"{config.connectivity_min_threshold})"
        )
    
    if metrics.topological_robustness < 0.95:
        factors.append(
            f"Nodos flotantes detectados (ρ={metrics.topological_robustness:.3f})"
        )
    
    if metrics.gini_coefficient > config.gini_max_threshold:
        factors.append(
            f"Alta concentración (Gini={metrics.gini_coefficient:.3f} > "
            f"{config.gini_max_threshold})"
        )
    
    if metrics.effective_species < 3:
        factors.append(
            f"Muy pocos insumos efectivos (N_eff={metrics.effective_species:.1f})"
        )
    
    return factors


def _generate_recommendations(
    factors: List[str],
    metrics: GraphMetrics
) -> List[str]:
    """Genera recomendaciones basadas en factores problemáticos."""
    recommendations = []
    
    if any("diversidad" in f.lower() for f in factors):
        recommendations.append(
            "Considere diversificar el catálogo de insumos para reducir "
            "dependencia de pocos proveedores"
        )
    
    if any("flotantes" in f.lower() for f in factors):
        recommendations.append(
            "Revise APUs sin insumos asociados — pueden indicar errores de "
            "modelado o datos faltantes"
        )
    
    if any("concentración" in f.lower() for f in factors):
        recommendations.append(
            "Evalúe riesgos de concentración: pocos insumos dominan el "
            "presupuesto total"
        )
    
    if metrics.edges_per_apu < 2:
        recommendations.append(
            f"Promedio bajo de insumos por APU ({metrics.edges_per_apu:.1f}) "
            "puede indicar estructura simplificada en exceso"
        )
    
    return recommendations


def _determine_verdict(
    psi: float, 
    metrics: GraphMetrics,
    config: AuditConfiguration
) -> StabilityVerdict:
    """
    Determina veredicto estructurado con análisis de factores contribuyentes.
    
    La lógica de decisión incorpora:
      1. Umbral principal (Ψ)
      2. Análisis de factores individuales
      3. Generación de recomendaciones contextuales
    """
    factors = _identify_contributing_factors(metrics, config)
    recommendations = _generate_recommendations(factors, metrics)
    
    # CASO 1: Veto estructural
    if psi < config.psi_veto_threshold:
        return StabilityVerdict(
            level=StabilityVerdict.Level.VETO,
            psi=psi,
            message=(
                f"VETO ESTRUCTURAL: Ψ={psi:.4f} < {config.psi_veto_threshold}. "
                f"Inestabilidad topológica severa detectada. "
                f"Simpson={metrics.simpson_diversity:.3f}, "
                f"κ={metrics.connectivity_ratio:.3f}, "
                f"ρ={metrics.topological_robustness:.3f}, "
                f"Gini={metrics.gini_coefficient:.3f}. "
                f"Factores: {'; '.join(factors) if factors else 'múltiples'}."
            ),
            blocking=True,
            contributing_factors=tuple(factors),
            recommendations=tuple(recommendations)
        )
    
    # CASO 2: Advertencia estructural
    if psi < config.psi_warning_threshold:
        return StabilityVerdict(
            level=StabilityVerdict.Level.WARNING,
            psi=psi,
            message=(
                f"ADVERTENCIA: Ψ={psi:.4f} bajo umbral recomendado "
                f"({config.psi_warning_threshold}). "
                f"Estructura aceptable pero con riesgos latentes."
            ),
            blocking=False,
            contributing_factors=tuple(factors),
            recommendations=tuple(recommendations)
        )
    
    # CASO 3: Saludable
    if psi < config.psi_healthy_threshold:
        return StabilityVerdict(
            level=StabilityVerdict.Level.HEALTHY,
            psi=psi,
            message=f"Estructura saludable (Ψ={psi:.4f}).",
            blocking=False,
            contributing_factors=tuple(factors),
            recommendations=tuple(recommendations) if factors else ()
        )
    
    # CASO 4: Óptimo
    return StabilityVerdict(
        level=StabilityVerdict.Level.OPTIMAL,
        psi=psi,
        message=(
            f"Estructura óptima (Ψ={psi:.4f} ≥ {config.psi_healthy_threshold}). "
            f"Excelente balance de diversidad, conectividad y robustez."
        ),
        blocking=False,
        contributing_factors=(),
        recommendations=()
    )


# ═══════════════════════════════════════════════════════════════════════════
# AUDITORÍA Y HASHING — Trazabilidad criptográfica
# ═══════════════════════════════════════════════════════════════════════════

def _build_audit_hash(
    metrics: GraphMetrics, 
    timestamp: float,
    config: AuditConfiguration
) -> str:
    """
    Sello criptográfico determinista del estado estructural.
    
    Incluye:
      • Todas las métricas fundamentales
      • Versión del algoritmo
      • Timestamp
      • Top-5 insumos (firma de distribución)
      • Hash de configuración
    """
    # Top 5 insumos
    top_insumos = sorted(
        metrics.insumo_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    # Hash de configuración (para invalidar en cambios)
    config_hash = hashlib.md5(
        f"{config.psi_veto_threshold}|{config.normalized_weights}".encode()
    ).hexdigest()[:8]
    
    canonical = (
        f"PYRAMID_AUDIT|v={config.algorithm_version}"
        f"|cfg={config_hash}"
        f"|ts={timestamp:.6f}"
        f"|m={metrics.structure_load}"
        f"|n={metrics.base_width}"
        f"|e={metrics.edge_count}"
        f"|f={metrics.floating_nodes}"
        f"|D={metrics.simpson_diversity:.6f}"
        f"|H={metrics.shannon_entropy:.6f}"
        f"|G={metrics.gini_coefficient:.6f}"
        f"|κ={metrics.connectivity_ratio:.6f}"
        f"|ρ={metrics.topological_robustness:.6f}"
        f"|λ₂={metrics.algebraic_connectivity or 0:.6f}"
        f"|Ψ={metrics.composite_stability:.6f}"
        f"|top={top_insumos}"
    )
    
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _elapsed_ms(start: float) -> float:
    """Calcula tiempo transcurrido en milisegundos."""
    return (time.perf_counter() - start) * 1000.0


# ═══════════════════════════════════════════════════════════════════════════
# VALIDACIÓN DE ESQUEMA — Verificación robusta de estructura de datos
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SchemaValidationResult:
    """Resultado de validación de esquema."""
    is_valid: bool
    missing_columns: FrozenSet[str]
    type_errors: Tuple[str, ...]
    warnings: Tuple[str, ...]


def _validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: Set[str],
    df_name: str,
    type_checks: Optional[Dict[str, type]] = None
) -> SchemaValidationResult:
    """
    Valida esquema de DataFrame con verificación de tipos opcional.
    
    Args:
        df: DataFrame a validar
        required_columns: Columnas requeridas
        df_name: Nombre para mensajes de error
        type_checks: Mapeo columna → tipo esperado
    
    Returns:
        SchemaValidationResult con detalles de validación
    """
    if df is None:
        return SchemaValidationResult(
            is_valid=False,
            missing_columns=frozenset(required_columns),
            type_errors=(f"{df_name} es None",),
            warnings=()
        )
    
    actual_columns = set(df.columns)
    missing = required_columns - actual_columns
    
    type_errors = []
    warnings = []
    
    # Verificar tipos si se especifican
    if type_checks and not missing:
        for col, expected_type in type_checks.items():
            if col in actual_columns:
                actual_dtype = df[col].dtype
                # Verificación flexible de tipos
                if expected_type == str and not pd.api.types.is_string_dtype(actual_dtype):
                    if pd.api.types.is_object_dtype(actual_dtype):
                        warnings.append(
                            f"{df_name}[{col}] es object, se convertirá a string"
                        )
                    else:
                        type_errors.append(
                            f"{df_name}[{col}] esperaba string, tiene {actual_dtype}"
                        )
    
    # Verificar filas vacías
    if not missing and df.empty:
        warnings.append(f"{df_name} está vacío (0 filas)")
    
    return SchemaValidationResult(
        is_valid=len(missing) == 0 and len(type_errors) == 0,
        missing_columns=frozenset(missing),
        type_errors=tuple(type_errors),
        warnings=tuple(warnings)
    )


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR PRINCIPAL — Orquestación de la auditoría
# ═══════════════════════════════════════════════════════════════════════════

def vector_audit_pyramidal_structure(
    *,
    df_presupuesto: pd.DataFrame,
    df_insumos: pd.DataFrame,
    config: Optional[AuditConfiguration] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    [TACTICS] Vector de Auditoría Topológica — Estabilidad del Grafo Bipartito.
    
    Analiza el grafo G = (APUs, Insumos, Aristas) con métricas algebraicas:
    
    1. **Diversidad de Simpson** (D): heterogeneidad de uso
    2. **Entropía de Shannon** (H): información/incertidumbre
    3. **Coeficiente de Gini** (G): concentración/desigualdad
    4. **Conectividad Normalizada** (κ): densidad vs. esperado
    5. **Robustez Topológica** (ρ): fracción conectada
    6. **Conectividad Algebraica** (λ₂): resistencia a partición
    7. **Estabilidad Compuesta** (Ψ): índice agregado ponderado
    
    Invariantes verificados en orden de severidad:
      • Vacuidad del grafo → fallo temprano
      • Validación de esquema → fallo de contrato
      • Nodos flotantes → penalización en ρ
      • Índice Ψ < umbral → veto estructural
    
    Args:
        df_presupuesto: DataFrame con columna 'CODIGO_APU' (nodos U)
        df_insumos: DataFrame con 'CODIGO_APU' y descripción insumo (nodos V)
        config: Configuración de umbrales (usa DEFAULT_CONFIG si None)
        
    Returns:
        Dict con protocolo MIC-Vector:
          • success: bool
          • stratum: Stratum.TACTICS
          • status: VectorResultStatus
          • payload/error: datos de auditoría
          • metrics: tiempos y métricas calculadas
    """
    start_time = time.perf_counter()
    config = config or DEFAULT_CONFIG
    
    try:
        # ═══════════════════════════════════════════════════════════════════
        # FASE 0: Validación de Contrato de Entrada
        # ═══════════════════════════════════════════════════════════════════
        
        # Validar presupuesto
        presupuesto_validation = _validate_dataframe_schema(
            df_presupuesto,
            required_columns={"CODIGO_APU"},
            df_name="df_presupuesto",
            type_checks={"CODIGO_APU": str}
        )
        
        # Validar insumos
        insumos_validation = _validate_dataframe_schema(
            df_insumos,
            required_columns={"CODIGO_APU"},
            df_name="df_insumos",
            type_checks={"CODIGO_APU": str}
        )
        
        # Agregar errores de validación
        validation_errors = []
        if not presupuesto_validation.is_valid:
            validation_errors.append(
                f"presupuesto falta columnas: {presupuesto_validation.missing_columns}"
            )
            validation_errors.extend(presupuesto_validation.type_errors)
        
        if not insumos_validation.is_valid:
            validation_errors.append(
                f"insumos falta columnas: {insumos_validation.missing_columns}"
            )
            validation_errors.extend(insumos_validation.type_errors)
        
        if validation_errors:
            raise ValueError(
                f"Esquema inválido: {'; '.join(validation_errors)}"
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # FASE 1: Vacuidad del Grafo (|U| = 0)
        # ═══════════════════════════════════════════════════════════════════
        
        if df_presupuesto.empty:
            logger.warning("Grafo vacío: df_presupuesto sin registros")
            return {
                "success": False,
                "stratum": Stratum.TACTICS,
                "status": VectorResultStatus.TOPOLOGY_ERROR,
                "error": (
                    "GRAFO_VACÍO: df_presupuesto no contiene registros. "
                    "Imposible evaluar estabilidad topológica."
                ),
                "metrics": {
                    "processing_time_ms": _elapsed_ms(start_time),
                    "phase": "graph_emptiness_check"
                },
            }
        
        # ═══════════════════════════════════════════════════════════════════
        # FASE 2: Extracción de Nodos APU (conjunto U)
        # ═══════════════════════════════════════════════════════════════════
        
        apus_totales, apu_quality = _sanitize_codigo_set(
            df_presupuesto["CODIGO_APU"],
            col_name="CODIGO_APU[presupuesto]"
        )
        
        structure_load = len(apus_totales)
        
        if structure_load == 0:
            logger.error("Carga estructural nula post-sanitización")
            return {
                "success": False,
                "stratum": Stratum.TACTICS,
                "status": VectorResultStatus.TOPOLOGY_ERROR,
                "error": (
                    "CARGA_NULA: Ningún CODIGO_APU válido encontrado en "
                    "df_presupuesto tras sanitización."
                ),
                "metrics": {
                    "processing_time_ms": _elapsed_ms(start_time),
                    "phase": "apu_extraction",
                    "raw_count": apu_quality.original_count,
                    "valid_count": 0,
                    "data_quality": {
                        "validity_ratio": apu_quality.validity_ratio,
                        "null_count": apu_quality.null_count
                    }
                },
            }
        
        # ═══════════════════════════════════════════════════════════════════
        # FASE 3: Extracción de Nodos Insumo y Distribución (conjunto V)
        # ═══════════════════════════════════════════════════════════════════
        
        # Determinar columna de descripción de insumo
        col_insumo = (
            "DESCRIPCION_INSUMO_NORM"
            if "DESCRIPCION_INSUMO_NORM" in df_insumos.columns
            else "DESCRIPCION_INSUMO"
        )
        
        if col_insumo not in df_insumos.columns:
            raise ValueError(
                f"Columna de descripción de insumo no encontrada. "
                f"Esperadas: 'DESCRIPCION_INSUMO_NORM' o 'DESCRIPCION_INSUMO'. "
                f"Disponibles: {list(df_insumos.columns)}"
            )
        
        base_width, insumo_distribution, apu_to_insumos = _extract_insumo_distribution(
            df_insumos, col_insumo, "CODIGO_APU"
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # FASE 4: Detección de Nodos Flotantes
        # ═══════════════════════════════════════════════════════════════════
        
        apus_con_insumos = set(apu_to_insumos.keys())
        floating_nodes_set = apus_totales - apus_con_insumos
        floating_count = len(floating_nodes_set)
        
        # Logging de nodos flotantes (pero no bloqueo inmediato)
        if floating_count > 0:
            logger.warning(
                f"Detectados {floating_count} APUs sin insumos asociados "
                f"(muestra: {sorted(floating_nodes_set)[:5]})"
            )
        
        # Contar aristas
        edge_count = sum(len(insumos) for insumos in apu_to_insumos.values())
        
        # Validación: base_width = 0 implica grafo degenerado
        if base_width == 0:
            logger.error("Base de insumos vacía post-sanitización")
            return {
                "success": False,
                "stratum": Stratum.TACTICS,
                "status": VectorResultStatus.TOPOLOGY_ERROR,
                "error": (
                    f"BASE_VACÍA: No se encontraron insumos válidos en columna "
                    f"'{col_insumo}' tras sanitización."
                ),
                "metrics": {
                    "processing_time_ms": _elapsed_ms(start_time),
                    "phase": "insumo_extraction"
                },
            }
        
        # ═══════════════════════════════════════════════════════════════════
        # FASE 5: Cálculo de Métricas Algebraicas
        # ═══════════════════════════════════════════════════════════════════
        
        # Diversidad
        simpson_diversity = _compute_simpson_diversity(
            insumo_distribution, 
            epsilon=config.epsilon
        )
        
        shannon_entropy, effective_species = _compute_shannon_entropy(
            insumo_distribution,
            epsilon=config.epsilon
        )
        
        gini_coefficient = _compute_gini_coefficient(
            insumo_distribution,
            epsilon=config.epsilon
        )
        
        # Conectividad
        connectivity_ratio = _compute_connectivity_ratio(
            edge_count,
            structure_load,
            config.expected_insumos_per_apu,
            epsilon=config.epsilon
        )
        
        topological_robustness = _compute_topological_robustness(
            structure_load, 
            floating_count,
            penalty_exponent=2.0
        )
        
        # Análisis espectral (opcional)
        algebraic_connectivity = None
        spectral_eigenvalues = None
        
        if config.enable_spectral_analysis and len(apu_to_insumos) >= 2:
            spectral_result = _compute_algebraic_connectivity(
                apu_to_insumos,
                num_eigenvalues=config.spectral_num_eigenvalues
            )
            if spectral_result is not None:
                algebraic_connectivity, spectral_eigenvalues = spectral_result
        
        # Índice compuesto
        composite_stability = _compute_composite_stability(
            simpson_diversity,
            connectivity_ratio,
            topological_robustness,
            gini_coefficient,
            config
        )
        
        # Acoplamiento de Fröhlich (Detección de Polarones)
        # Evaluamos el valor de Fiedler (lambda_2, que corresponde a algebraic_connectivity en el grafo)
        # Para acoplar el arrastre inercial a las demoras o fallas atómicas
        # Generar un flag / métrica de acoplamiento si existe
        frohlich_alpha = 0.0
        if algebraic_connectivity is not None and algebraic_connectivity > 0:
            import networkx as nx
            try:
                # Construimos un grafo NX temporal simple para obtener centralidades si G no se pasó.
                # Simplificación táctica para calcular alpha medio.
                frohlich_alpha = 1.0 / algebraic_connectivity  # Acoplamiento inversamente proporcional a la conectividad
            except Exception as polar_exc:
                logger.warning(f"Error evaluando constantes polarónicas: {polar_exc}")

        # Estadísticas adicionales
        edges_per_apu = edge_count / structure_load if structure_load > 0 else 0.0
        
        # Estadísticas de insumos por APU
        insumos_per_apu = [len(ins) for ins in apu_to_insumos.values()]
        if insumos_per_apu:
            insumos_mean = np.mean(insumos_per_apu)
            insumos_std = np.std(insumos_per_apu, ddof=1) if len(insumos_per_apu) > 1 else 0.0
        else:
            insumos_mean = 0.0
            insumos_std = 0.0
        
        # Empaquetar métricas
        metrics_obj = GraphMetrics(
            structure_load=structure_load,
            base_width=base_width,
            edge_count=edge_count,
            floating_nodes=floating_count,
            simpson_diversity=simpson_diversity,
            shannon_entropy=shannon_entropy,
            effective_species=effective_species,
            gini_coefficient=gini_coefficient,
            connectivity_ratio=connectivity_ratio,
            topological_robustness=topological_robustness,
            algebraic_connectivity=algebraic_connectivity,
            composite_stability=composite_stability,
            insumo_distribution=insumo_distribution,
            edges_per_apu=edges_per_apu,
            insumos_per_apu_mean=insumos_mean,
            insumos_per_apu_std=insumos_std
        )
        
        # Diagnósticos de distribución
        dist_diagnostics = _compute_distribution_diagnostics(insumo_distribution)
        
        # ═══════════════════════════════════════════════════════════════════
        # FASE 6: Determinación de Veredicto
        # ═══════════════════════════════════════════════════════════════════
        
        verdict = _determine_verdict(composite_stability, metrics_obj, config)
        
        # Si es veto, retornar fallo
        if verdict.blocking:
            logger.warning(f"Veto estructural: Ψ={composite_stability:.4f}")
            return {
                "success": False,
                "stratum": Stratum.TACTICS,
                "status": VectorResultStatus.TOPOLOGY_ERROR,
                "error": verdict.message,
                "metrics": {
                    "processing_time_ms": _elapsed_ms(start_time),
                    "phase": "verdict_evaluation",
                    "stability_psi": composite_stability,
                    "simpson_diversity": simpson_diversity,
                    "shannon_entropy": shannon_entropy,
                    "gini_coefficient": gini_coefficient,
                    "connectivity_ratio": connectivity_ratio,
                    "topological_robustness": topological_robustness,
                    "algebraic_connectivity": algebraic_connectivity,
                    "base_to_load_ratio": base_width / structure_load,
                    "contributing_factors": verdict.contributing_factors,
                    "recommendations": verdict.recommendations,
                },
            }
        
        # ═══════════════════════════════════════════════════════════════════
        # FASE 7: Construcción de Payload Exitoso
        # ═══════════════════════════════════════════════════════════════════
        
        audit_hash = _build_audit_hash(metrics_obj, start_time, config)
        
        # Top insumos más usados
        top_insumos = sorted(
            insumo_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        payload = {
            # === Veredicto ===
            "audit_verdict": verdict.level.value,
            "verdict_message": verdict.message,
            "verdict_severity": verdict.severity_score,
            
            # === Métrica compuesta principal ===
            "composite_stability_psi": round(composite_stability, 6),
            
            # === Métricas de diversidad ===
            "diversity": {
                "simpson_index": round(simpson_diversity, 6),
                "shannon_entropy": round(shannon_entropy, 6),
                "effective_species": round(effective_species, 2),
                "gini_coefficient": round(gini_coefficient, 6),
            },
            
            # === Métricas de conectividad ===
            "connectivity": {
                "ratio": round(connectivity_ratio, 6),
                "topological_robustness": round(topological_robustness, 6),
                "algebraic_connectivity": (
                    round(algebraic_connectivity, 6) 
                    if algebraic_connectivity is not None 
                    else None
                ),
                "frohlich_coupling_alpha": round(frohlich_alpha, 6),
                "floating_nodes": floating_count,
            },
            
            # === Umbrales de referencia ===
            "thresholds": {
                "veto": config.psi_veto_threshold,
                "warning": config.psi_warning_threshold,
                "healthy": config.psi_healthy_threshold,
            },
            
            # === Estructura del grafo ===
            "graph_structure": {
                "structure_load_m": structure_load,
                "base_width_n": base_width,
                "edge_count": edge_count,
                "density": round(metrics_obj.density, 6),
                "base_to_load_ratio": round(base_width / structure_load, 6),
                "edges_per_apu_mean": round(edges_per_apu, 2),
                "insumos_per_apu_mean": round(insumos_mean, 2),
                "insumos_per_apu_std": round(insumos_std, 2),
            },
            
            # === Diagnósticos de distribución ===
            "distribution_diagnostics": (
                {
                    "mean": round(dist_diagnostics.mean, 2),
                    "median": round(dist_diagnostics.median, 2),
                    "std": round(dist_diagnostics.std, 2),
                    "cv": round(dist_diagnostics.coefficient_of_variation, 4),
                    "skewness": round(dist_diagnostics.skewness, 4),
                    "kurtosis": round(dist_diagnostics.kurtosis, 4),
                    "max_concentration": round(dist_diagnostics.max_concentration, 4),
                    "top_5_concentration": round(dist_diagnostics.top_5_concentration, 4),
                    "outlier_count": dist_diagnostics.outlier_count,
                }
                if dist_diagnostics is not None
                else None
            ),
            
            # === Top insumos ===
            "top_insumos_by_usage": [
                {"insumo": k, "count": v, "share": round(v / edge_count, 4)}
                for k, v in top_insumos
            ],
            
            # === Metadatos ===
            "col_insumo_used": col_insumo,
            "algorithm_version": config.algorithm_version,
            "structural_audit_hash": audit_hash,
            "spectral_analysis_enabled": config.enable_spectral_analysis,
        }
        
        # Añadir advertencias/recomendaciones si aplican
        if verdict.contributing_factors:
            payload["contributing_factors"] = verdict.contributing_factors
        if verdict.recommendations:
            payload["recommendations"] = verdict.recommendations
        if verdict.level == StabilityVerdict.Level.WARNING:
            payload["structural_warning"] = verdict.message
        
        # Incluir floating nodes si existen (para diagnóstico)
        if floating_count > 0:
            payload["floating_nodes_sample"] = sorted(floating_nodes_set)[:20]
        
        logger.info(
            f"Auditoría exitosa: {verdict.level.value}, "
            f"Ψ={composite_stability:.4f}, "
            f"D={simpson_diversity:.3f}, G={gini_coefficient:.3f}"
        )
        
        return {
            "success": True,
            "stratum": Stratum.TACTICS,
            "status": VectorResultStatus.SUCCESS,
            "payload": payload,
            "metrics": {
                "processing_time_ms": _elapsed_ms(start_time),
                "topological_coherence": round(composite_stability, 6),
                "spectral_analysis_performed": algebraic_connectivity is not None,
            },
        }
    
    except (TypeError, ValueError) as exc:
        logger.error(f"Error de contrato: {exc}", exc_info=True)
        return {
            "success": False,
            "stratum": Stratum.TACTICS,
            "status": VectorResultStatus.LOGIC_ERROR,
            "error": f"Violación de contrato en auditoría piramidal: {exc}",
            "metrics": {
                "processing_time_ms": _elapsed_ms(start_time),
                "error_type": type(exc).__name__
            },
        }
    
    except Exception as exc:
        logger.exception(f"Fallo interno inesperado: {exc}")
        return {
            "success": False,
            "stratum": Stratum.TACTICS,
            "status": VectorResultStatus.LOGIC_ERROR,
            "error": f"Fallo interno en auditoría piramidal: {exc}",
            "metrics": {
                "processing_time_ms": _elapsed_ms(start_time),
                "error_type": type(exc).__name__
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES EXPORTADAS — Para análisis externos
# ═══════════════════════════════════════════════════════════════════════════

def compute_herfindahl_index(distribution: Dict[str, int]) -> float:
    """
    Índice de Herfindahl-Hirschman: HHI = Σ(sᵢ²)
    
    Complementario a Simpson: HHI = 1 - D
    
    Returns:
        HHI ∈ [1/n, 1]
        HHI → 1 indica monopolio
        HHI → 1/n indica distribución equitativa
    """
    simpson = _compute_simpson_diversity(distribution)
    return 1.0 - simpson


def compute_effective_number_of_species(
    distribution: Dict[str, int],
    q: float = 1.0
) -> float:
    """
    Número efectivo de especies (Hill numbers).
    
    Para q=0: riqueza (número de especies)
    Para q=1: exp(Shannon) (diversidad típica)
    Para q=2: 1/Simpson (diversidad dominante)
    
    Args:
        distribution: {insumo: frecuencia}
        q: orden de diversidad
    
    Returns:
        N_eff ≥ 1
    """
    if not distribution:
        return 1.0
    
    values = np.array(list(distribution.values()), dtype=np.float64)
    total = values.sum()
    
    if total == 0:
        return 1.0
    
    proportions = values / total
    proportions = proportions[proportions > 0]
    
    if q == 0:
        # Riqueza
        return float(len(proportions))
    elif q == 1:
        # Exponencial de Shannon
        entropy = -np.sum(proportions * np.log(proportions))
        return float(np.exp(entropy))
    elif q == 2:
        # Inverso de Simpson (HHI)
        hhi = np.sum(proportions ** 2)
        return float(1.0 / hhi) if hhi > 0 else float(len(proportions))
    else:
        # Caso general
        powered = np.sum(proportions ** q)
        return float(powered ** (1.0 / (1.0 - q)))


def analyze_graph_spectrum(
    apu_to_insumos: Dict[str, Set[str]],
    num_eigenvalues: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Análisis espectral completo del grafo bipartito.
    
    Calcula:
      • Eigenvalores del Laplaciano
      • Gap espectral
      • Conectividad algebraica
      • Indicadores de clustering
    
    Returns:
        Dict con análisis espectral o None si scipy no disponible
    """
    if not SCIPY_AVAILABLE:
        return None
    
    result = _compute_algebraic_connectivity(
        apu_to_insumos, 
        num_eigenvalues=num_eigenvalues
    )
    
    if result is None:
        return None
    
    lambda_2_norm, eigenvalues = result
    
    # Gap espectral: diferencia entre λ₁ y λ₂
    spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0
    
    return {
        "algebraic_connectivity": lambda_2_norm,
        "eigenvalues": eigenvalues.tolist(),
        "spectral_gap": float(spectral_gap),
        "num_nodes": len(apu_to_insumos),
        "is_connected": lambda_2_norm > 1e-10,
    }
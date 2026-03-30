"""
=========================================================================================
Módulo: Semantic Dictionary (El Guardián de la Ontología y Fibrado Semántico)
Ubicación: app/wisdom/semantic_dictionary.py
=========================================================================================

Naturaleza Ciber-Física y Topológica:
    Actúa como la "Piedra Rosetta" de la Malla Agéntica, estableciendo el Espacio Base
    para el Difeomorfismo Semántico. Su responsabilidad axiomática es alojar las 
    proyecciones lingüísticas puras (fibras) que mapean invariantes topológicos, 
    espectrales y termodinámicos hacia el dominio del impacto de negocio corporativo.

1. Fibración Semántica y Preservación de Homotopía:
    Sea M el espacio de métricas invariantes (InvariantSpace) y N el espacio narrativo 
    (ImpactSpace). El diccionario define un funtor de proyección F: M → N que preserva 
    la homotopía: dos estados termodinámicamente equivalentes pero topológicamente 
    distintos se mapean de manera estricta a narrativas disjuntas. La IA generativa 
    opera únicamente sobre las fibras pre-aprobadas de este espacio, erradicando 
    la alucinación estocástica.

2. Mapeo Biyectivo de Invariantes (El Funtor de Traducción):
    El diccionario establece el contrato algebraico inmutable para traducir las 
    patologías estructurales detectadas en los estratos inferiores:
    
        • Topología Simplicial (Homología):
          β₀ > 1  (Fragmentación)    → "Recursos Huérfanos / Islas de Datos"
          β₁ > 0  (Ciclos)           → "Socavón Lógico / Bucle de Dependencia"
          Ψ < 1.0 (Pirámide Inversa) → "Riesgo de Colapso por Base Estrecha"
          
        • Cohomología de Haces (Cellular Sheaves):
          H¹ ≠ 0  (Obstrucción)      → "Veto Estructural / Paradoja Contractual"
          E(x)> ε (Energía Dirichlet)→ "Fricción Operativa / Desgaste de Consenso"
          
        • Teoría Espectral y Termodinámica:
          λ₂ ≈ 0  (Brecha espectral) → "Fractura Organizacional Inminente"
          ΔS > 0  (Alta volatilidad) → "Fiebre Inflacionaria"

3. Invarianza Funcional y Ortogonalidad:
    Este módulo está estrictamente desacoplado del álgebra de decisiones (operación 
    Supremo ⊔ del retículo). Solo provee las "Coordenadas Base" del lenguaje. Si la 
    matriz empresarial requiere alterar el tono ejecutivo (Empatía Táctica), únicamente 
    mutan las fibras en este diccionario mediante el sistema de TTLCache, dejando 
    intactos los motores de evaluación topológica y reticular del Estrato Ω.
=========================================================================================
"""
import logging
import random
import re
import string
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from typing import (
    Any, 
    Callable, 
    Dict, 
    Final, 
    FrozenSet, 
    List, 
    Literal, 
    Optional, 
    Set, 
    Tuple, 
    TypeVar, 
    Union,
)

# =============================================================================
# IMPORT SEGURO CON FALLBACK ROBUSTO
# =============================================================================

try:
    from app.core.schemas import Stratum
except ImportError:
    class Stratum(IntEnum):
        """
        Representación jerárquica del modelo DIKW.
        
        Corresponde a una estratificación topológica donde cada estrato
        define un nivel de abstracción en la pirámide de conocimiento.
        
        Propiedades Algebraicas:
            - Forma un conjunto totalmente ordenado (cadena)
            - El orden induce una filtración: WISDOM ⊂ OMEGA ⊂ STRATEGY ⊂ TACTICS ⊂ PHYSICS
        """
        WISDOM = 0      # Síntesis estratégica (Ápice)
        OMEGA = 1       # Ágora Tensorial
        STRATEGY = 2    # Planificación financiera
        TACTICS = 3     # Estructura operativa
        PHYSICS = 4     # Datos físicos/cimentación (Base)


logger = logging.getLogger("SemanticDictionary")

# =============================================================================
# CONSTANTES Y TIPOS
# =============================================================================

NodeType = Literal["ROOT", "CAPITULO", "APU", "INSUMO"]
VALID_NODE_TYPES: Final[FrozenSet[str]] = frozenset({"ROOT", "CAPITULO", "APU", "INSUMO"})

T = TypeVar('T')


# =============================================================================
# UTILIDADES DE CACHÉ THREAD-SAFE CON TTL
# =============================================================================

class TTLCache:
    """
    Caché LRU con Time-To-Live y límite de tamaño.
    
    Implementa evicción perezosa (lazy eviction) por TTL y evicción
    activa por capacidad máxima siguiendo política LRU.
    
    Thread-safe mediante RLock para operaciones anidadas.
    """
    
    __slots__ = ('_cache', '_timestamps', '_ttl', '_maxsize', '_lock', '_hits', '_misses')
    
    def __init__(self, ttl_seconds: float = 300.0, maxsize: int = 1000):
        if ttl_seconds <= 0:
            raise ValueError("TTL debe ser positivo")
        if maxsize <= 0:
            raise ValueError("maxsize debe ser positivo")
            
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._ttl: float = ttl_seconds
        self._maxsize: int = maxsize
        self._lock = threading.RLock()
        self._hits: int = 0
        self._misses: int = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor si existe y no ha expirado."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            timestamp = self._timestamps.get(key, 0)
            if time.time() - timestamp >= self._ttl:
                # Expirado: eliminar
                self._evict_key(key)
                self._misses += 1
                return None
            
            # Mover al final (más reciente) para LRU
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Almacena valor con timestamp actual."""
        with self._lock:
            # Si ya existe, actualizar y mover al final
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                # Verificar capacidad antes de insertar
                while len(self._cache) >= self._maxsize:
                    self._evict_oldest()
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def _evict_key(self, key: str) -> None:
        """Elimina una clave específica."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def _evict_oldest(self) -> None:
        """Elimina la entrada más antigua (LRU)."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._evict_key(oldest_key)
    
    def clear(self) -> None:
        """Limpia todo el caché."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def cleanup_expired(self) -> int:
        """
        Limpieza activa de entradas expiradas.
        
        Returns:
            Número de entradas eliminadas.
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, ts in self._timestamps.items()
                if now - ts >= self._ttl
            ]
            for key in expired_keys:
                self._evict_key(key)
            return len(expired_keys)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Estadísticas del caché."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


# =============================================================================
# CLASES DE DATOS TOPOLÓGICAS
# =============================================================================

@dataclass(frozen=True)
class PyramidalSemanticVector:
    """
    Vector de estado semántico para un nodo dentro del Grafo del Presupuesto.
    
    Codifica su posición en la estructura piramidal (DIKW) y su carga topológica.
    Actúa como un tensor de información para la generación de GraphRAG.
    
    Definiciones Topológicas:
        - in_degree: |{e ∈ E : target(e) = v}| (aristas entrantes)
        - out_degree: |{e ∈ E : source(e) = v}| (aristas salientes)
        - is_critical_bridge: v es un cut-vertex (su eliminación desconecta el grafo)
        
    Nota: Un cut-vertex NO se determina solo por grados, sino por análisis
    de conectividad (DFS/Tarjan). El flag debe ser calculado externamente.
    
    Invariantes:
        - in_degree >= 0, out_degree >= 0
        - node_type ∈ VALID_NODE_TYPES
    """
    node_id: str
    node_type: NodeType
    stratum: Stratum
    in_degree: int
    out_degree: int
    is_critical_bridge: bool = False
    
    def __post_init__(self):
        """Validaciones post-construcción para integridad topológica."""
        # Validar grados no negativos
        if self.in_degree < 0:
            raise ValueError(
                f"in_degree no puede ser negativo: {self.in_degree}"
            )
        if self.out_degree < 0:
            raise ValueError(
                f"out_degree no puede ser negativo: {self.out_degree}"
            )
        
        # Validar node_type
        if self.node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"node_type inválido: '{self.node_type}'. "
                f"Valores permitidos: {VALID_NODE_TYPES}"
            )
        
        # Validar node_id no vacío
        if not self.node_id or not self.node_id.strip():
            raise ValueError("node_id no puede estar vacío")
    
    @property
    def total_degree(self) -> int:
        """Grado total del nodo (suma de entrantes y salientes)."""
        return self.in_degree + self.out_degree
    
    @property
    def is_leaf(self) -> bool:
        """Verdadero si el nodo es una hoja (sin salientes)."""
        return self.out_degree == 0
    
    @property
    def is_root(self) -> bool:
        """Verdadero si el nodo es raíz (sin entrantes)."""
        return self.in_degree == 0
    
    @property
    def is_isolated(self) -> bool:
        """Verdadero si el nodo está aislado."""
        return self.total_degree == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "stratum": self.stratum.name,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "is_critical_bridge": self.is_critical_bridge,
            "total_degree": self.total_degree,
        }


# =============================================================================
# PROYECTOR SEMÁNTICO DE GRAFOS
# =============================================================================

class GraphSemanticProjector:
    """
    Proyecta la topología algebraica del presupuesto hacia el espacio narrativo.
    
    Este Functor es el motor central de GraphRAG: navega el grafo y traduce
    los invariantes matemáticos en "La Voz del Consejo".
    
    Propiedades Algebraicas:
        - Actúa como un morfismo F: Top → Narr entre categorías
        - Preserva composición: F(g ∘ f) = F(g) ∘ F(f)
        - Preserva identidades: F(id_X) = id_{F(X)}
    """
    
    def __init__(
        self, 
        dictionary_service: 'SemanticDictionaryService',
        cache_ttl: float = 300.0,
        cache_maxsize: int = 500
    ):
        self._dictionary = dictionary_service
        self._cache = TTLCache(ttl_seconds=cache_ttl, maxsize=cache_maxsize)
    
    def _build_cache_key(self, prefix: str, *args) -> str:
        """Construye clave de caché determinística."""
        components = [prefix] + [str(a) for a in args]
        return ":".join(components)
    
    def project_pyramidal_stress(
        self, 
        vector: PyramidalSemanticVector
    ) -> Dict[str, Any]:
        """
        Proyecta un punto de estrés topológico a narrativa.
        
        Consumo PASIVO: Asume que el Arquitecto ya validó que es un punto de estrés.
        Solo inyecta los datos en la plantilla narrativa.
        
        Args:
            vector: Nodo crítico identificado por análisis topológico
        
        Returns:
            Diccionario con narrativa estructurada sobre el punto de estrés
        """
        cache_key = self._build_cache_key(
            "stress",
            vector.node_id,
            vector.in_degree,
            vector.out_degree
        )
        
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Retornar copia para evitar mutaciones
            return dict(cached)
        
        result = self._dictionary.fetch_narrative(
            domain="MISC",
            classification="STRESS_POINT",
            params={
                "node": vector.node_id,
                "degree": vector.total_degree,
                "in_degree": vector.in_degree,
                "out_degree": vector.out_degree,
            }
        )
        
        # Enriquecer con metadatos del vector
        result["vector_metadata"] = vector.to_dict()
        
        self._cache.set(cache_key, result)
        return result
    
    def project_cycle_path(self, path_nodes: List[str]) -> Dict[str, Any]:
        """
        Proyecta la ruta de un ciclo detectado a narrativa.
        
        Args:
            path_nodes: Secuencia ordenada de nodos formando el ciclo detectado.
                       El ciclo se cierra implícitamente (último → primero).
        
        Returns:
            Narrativa explicando la circularidad y sus implicaciones.
        """
        # Validación de entrada
        if not path_nodes:
            logger.warning("project_cycle_path llamado con lista vacía")
            return {
                "success": False, 
                "error": "Ruta vacía proporcionada.",
                "suggestion": "Verifique el algoritmo de detección de ciclos.",
            }
        
        # Sanitizar y convertir a strings
        sanitized_nodes = [str(n).strip() for n in path_nodes if n]
        
        if not sanitized_nodes:
            return {
                "success": False,
                "error": "Todos los nodos de la ruta son inválidos.",
            }
        
        if len(sanitized_nodes) == 1:
            logger.warning(f"Ciclo trivial (self-loop) detectado: {sanitized_nodes[0]}")
        
        # Construir representación de la ruta
        path_str = " → ".join(sanitized_nodes)
        cycle_length = len(sanitized_nodes)
        
        # Cache key basado en contenido ordenado
        cache_key = self._build_cache_key("cycle", hash(tuple(sanitized_nodes)))
        
        cached = self._cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        
        result = self._dictionary.fetch_narrative(
            domain="MISC",
            classification="CYCLE_PATH",
            params={
                "path": path_str,
                "first_node": sanitized_nodes[0],
                "cycle_length": cycle_length,
            }
        )
        
        # Metadatos adicionales
        result["cycle_metadata"] = {
            "nodes": sanitized_nodes,
            "length": cycle_length,
            "is_self_loop": cycle_length == 1,
        }
        
        self._cache.set(cache_key, result)
        return result
    
    def project_fragmentation(
        self, 
        beta_0: int, 
        component_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Proyecta la fragmentación topológica (número de Betti β₀) a narrativa.
        
        Args:
            beta_0: Número de componentes conexas
            component_sizes: Tamaños de cada componente (opcional)
        
        Returns:
            Narrativa sobre conectividad del grafo
        """
        cache_key = self._build_cache_key("fragmentation", beta_0)
        
        cached = self._cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        
        if beta_0 == 0:
            classification = "empty"
        elif beta_0 == 1:
            classification = "unified"
        elif beta_0 <= 5:
            classification = "fragmented"
        else:
            classification = "severely_fragmented"
        
        result = self._dictionary.fetch_narrative(
            domain="TOPOLOGY_CONNECTIVITY",
            classification=classification,
            params={"beta_0": beta_0}
        )
        
        if component_sizes:
            result["component_analysis"] = {
                "sizes": component_sizes,
                "largest": max(component_sizes),
                "smallest": min(component_sizes),
                "gini_coefficient": self._calculate_gini(component_sizes),
            }
        
        self._cache.set(cache_key, result)
        return result
    
    @staticmethod
    def _calculate_gini(values: List[int]) -> float:
        """
        Calcula el coeficiente de Gini para medir desigualdad.
        
        Gini = 0: Perfecta igualdad
        Gini = 1: Máxima desigualdad
        """
        if not values or len(values) == 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = sum(sorted_values)
        
        if cumsum == 0:
            return 0.0
        
        numerator = sum(
            (2 * (i + 1) - n - 1) * v 
            for i, v in enumerate(sorted_values)
        )
        return numerator / (n * cumsum)
    
    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Estadísticas del caché del proyector."""
        return self._cache.stats


# =============================================================================
# UTILIDADES PARA VALIDACIÓN DE PLANTILLAS
# =============================================================================

class TemplateValidator:
    """
    Validador de plantillas con extracción robusta de placeholders.
    
    Utiliza string.Formatter para parseo correcto de format strings.
    """
    
    # Patrón para detectar placeholders con formato
    _PLACEHOLDER_PATTERN = re.compile(r'\{(\w+)(?::[^}]*)?\}')
    
    @classmethod
    def extract_placeholders(cls, template: str) -> Set[str]:
        """
        Extrae todos los placeholders de una plantilla.
        
        Maneja correctamente formatos como {value:.2f}, {name!r}, etc.
        
        Args:
            template: String de plantilla con placeholders
            
        Returns:
            Conjunto de nombres de placeholders
        """
        formatter = string.Formatter()
        placeholders = set()
        
        try:
            for _, field_name, _, _ in formatter.parse(template):
                if field_name is not None:
                    # Extraer solo el nombre base (sin índices ni atributos)
                    base_name = field_name.split('.')[0].split('[')[0]
                    if base_name:
                        placeholders.add(base_name)
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parseando plantilla: {e}")
            # Fallback a regex
            placeholders = set(cls._PLACEHOLDER_PATTERN.findall(template))
        
        return placeholders
    
    @classmethod
    def validate_template(
        cls, 
        template: str, 
        required_params: Optional[Set[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Valida que una plantilla sea sintácticamente correcta.
        
        Args:
            template: Plantilla a validar
            required_params: Parámetros que deben estar presentes (opcional)
            
        Returns:
            Tupla (es_válida, mensaje_error)
        """
        try:
            placeholders = cls.extract_placeholders(template)
            
            # Crear diccionario de prueba con valores dummy
            # Se usa 0.0 para compatibilidad con formatadores 'f' en las validaciones de inicialización.
            test_params = {p: 0.0 for p in placeholders}
            
            # Intentar formatear
            template.format(**test_params)
            
            # Verificar parámetros requeridos
            if required_params:
                missing = required_params - placeholders
                if missing:
                    return False, f"Faltan placeholders requeridos: {missing}"
            
            return True, None
            
        except (KeyError, ValueError, IndexError) as e:
            return False, str(e)
    
    @classmethod
    def validate_all_templates(
        cls, 
        templates: Dict[str, Any],
        path: str = ""
    ) -> List[Dict[str, str]]:
        """
        Valida recursivamente todas las plantillas en un diccionario.
        
        Returns:
            Lista de errores encontrados
        """
        errors = []
        
        for key, value in templates.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                errors.extend(cls.validate_all_templates(value, current_path))
            elif isinstance(value, str) and '{' in value:
                is_valid, error_msg = cls.validate_template(value)
                if not is_valid:
                    errors.append({
                        "path": current_path,
                        "error": error_msg,
                        "template_preview": value[:100] + "..." if len(value) > 100 else value,
                    })
        
        return errors


# =============================================================================
# SERVICIO PRINCIPAL DEL DICCIONARIO SEMÁNTICO
# =============================================================================

class SemanticDictionaryService:
    """
    Servicio Central de Traducción Semántica.
    
    Transforma métricas técnicas crudas en narrativa estratégica comprensible.
    Implementa el patrón Template Method para extensibilidad controlada.
    
    Thread Safety:
        - Todas las operaciones de lectura son thread-safe por diseño inmutable
        - Las operaciones de escritura al caché están protegidas por locks
    """
    
    # =========================================================================
    # UMBRALES DE CLASIFICACIÓN
    # Definidos como constantes de clase para permitir override en subclases
    # =========================================================================
    
    # Umbrales de estabilidad (Ψ): ratio insumos/APUs
    STABILITY_THRESHOLDS: Final[Dict[str, float]] = {
        "robust": 0.85,     # Redundancia significativa
        "stable": 0.70,     # Equilibrio aceptable
        "warning": 0.50,    # Margen mínimo de seguridad
        "critical": 0.30,   # Base insuficiente
    }
    
    # Umbrales de entropía (S): desorden del sistema
    # Nota: Para entropía, MAYOR valor = PEOR estado
    ENTROPY_THRESHOLDS: Final[Dict[str, float]] = {
        "high": 0.70,    # Caos administrativo
        "low": 0.30,     # Procesos estructurados
    }
    
    # Umbrales de cohesión espectral (Fiedler eigenvalue)
    COHESION_THRESHOLDS: Final[Dict[str, float]] = {
        "high": 0.70,      # Fiedler ≥ 0.7
        "standard": 0.40,  # 0.4 ≤ Fiedler < 0.7
        "low": 0.0,        # Fiedler < 0.4
    }
    
    # Umbrales de temperatura (volatilidad de precios)
    TEMPERATURE_THRESHOLDS: Final[Dict[str, float]] = {
        "critical": 100.0,
        "hot": 75.0,
        "warm": 50.0,
        "stable": 25.0,
        "cold": 0.0,
    }
    
    def __init__(self):
        """Inicializa el repositorio de plantillas con validación integral."""
        self._lock = threading.RLock()
        self._templates = self._load_templates()
        self._market_contexts = self._load_market_contexts()
        self._projector: Optional[GraphSemanticProjector] = None
        
        # Validar plantillas en inicialización
        validation_errors = TemplateValidator.validate_all_templates(self._templates)
        if validation_errors:
            logger.warning(
                f"Se encontraron {len(validation_errors)} plantillas con "
                f"posibles problemas de formato"
            )
            for err in validation_errors[:5]:  # Mostrar solo primeros 5
                logger.warning(f"  - {err['path']}: {err['error']}")
        
        logger.info("✅ SemanticDictionaryService inicializado exitosamente")
    
    def _load_market_contexts(self) -> Tuple[str, ...]:
        """Carga los contextos de mercado como tupla inmutable."""
        return (
            "Suelo Estable: Precios de cemento sin variación significativa.",
            "Terreno Inflacionario: Acero al alza (+2.5%). Reforzar estimaciones.",
            "Vientos de Cambio: Volatilidad cambiaria favorable para importaciones.",
            "Falla Geológica Laboral: Escasez de mano de obra calificada.",
            "Mercado Saturado: Alta competencia presiona márgenes.",
        )
    
    def _load_templates(self) -> Dict[str, Any]:
        """
        Carga las plantillas narrativas.
        
        Las plantillas están organizadas por dominio semántico y clasificación.
        Cada plantilla puede contener placeholders {nombre} o {nombre:formato}.
        
        Returns:
            Diccionario anidado de plantillas
        """
        templates = {
            # ========== ANÁLISIS TOPOLÓGICO ==========
            "TOPOLOGY_CYCLES": {
                "clean": (
                    "✅ Integridad Estructural (Género 0): No se detectan socavones lógicos "
                    "(β₁ = 0). La Trazabilidad de Carga de Costos fluye verticalmente desde "
                    "la Cimentación hasta el Ápice sin recirculaciones."
                ),
                "minor": (
                    "🔶 Falla Estructural Local (Género {beta_1}): Se detectaron {beta_1} "
                    "socavones lógicos en la estructura de costos. Estos 'agujeros' impiden "
                    "la correcta Trazabilidad de Carga y deben ser corregidos para evitar "
                    "asentamientos diferenciales en el presupuesto."
                ),
                "moderate": (
                    "🚨 Estructura Geológicamente Inestable (Género {beta_1}): "
                    "Se detectó un Género Estructural de {beta_1}, indicando una estructura "
                    "tipo 'esponja'. Existen múltiples bucles de retroalimentación de costos "
                    "que impiden la Trazabilidad de Carga y hacen colapsar cualquier "
                    "valoración estática."
                ),
                "critical": (
                    "💀 COLAPSO TOPOLÓGICO (Género {beta_1}): "
                    "La estructura está completamente perforada con {beta_1} ciclos "
                    "independientes. Es matemáticamente imposible calcular costos "
                    "determinísticos. Se requiere rediseño fundamental."
                ),
            },
            
            "TOPOLOGY_CONNECTIVITY": {
                "empty": "⚠️ Terreno Vacío: No hay estructura proyectada (β₀ = 0).",
                "unified": (
                    "🔗 Unidad de Obra Monolítica: El proyecto funciona como un solo "
                    "edificio interconectado (β₀ = 1). Todas las cargas tácticas (APUs) "
                    "se transfieren correctamente hacia un único Ápice Estratégico."
                ),
                "fragmented": (
                    "⚠️ Edificios Desconectados (Fragmentación): El proyecto no es una "
                    "estructura única, sino un archipiélago de {beta_0} sub-estructuras "
                    "aisladas. No existe un Ápice unificado que centralice la carga financiera."
                ),
                "severely_fragmented": (
                    "🚨 Fragmentación Severa: El proyecto está fragmentado en {beta_0} islas "
                    "completamente desconectadas. Esto indica múltiples proyectos empaquetados "
                    "como uno solo, o datos severamente incompletos."
                ),
            },
            
            "STABILITY": {
                "critical": (
                    "📉 COLAPSO POR BASE ESTRECHA (Pirámide Invertida): Ψ = {stability:.2f}. "
                    "La Cimentación Logística (Insumos) es demasiado angosta para soportar el "
                    "Peso Táctico (APUs) que tiene encima. El centro de gravedad está muy alto; "
                    "riesgo inminente de vuelco financiero."
                ),
                "warning": (
                    "⚖️ Equilibrio Precario (Isostático): Ψ = {stability:.2f}. "
                    "El proyecto tiene la mínima base necesaria, sin redundancia. Cualquier "
                    "perturbación en el suministro puede desestabilizar toda la estructura."
                ),
                "stable": (
                    "⚖️ Estructura Isostática (Estable): Ψ = {stability:.2f}. "
                    "El equilibrio entre la carga de actividades y el soporte de insumos es "
                    "adecuado, aunque no posee redundancia sísmica."
                ),
                "robust": (
                    "🛡️ ESTRUCTURA ANTISÍSMICA (Resiliente): Ψ = {stability:.2f}. "
                    "La Cimentación de Recursos es amplia y redundante. El proyecto tiene un "
                    "bajo centro de gravedad, capaz de absorber vibraciones del mercado "
                    "(volatilidad) sin sufrir daños estructurales."
                ),
            },
            
            "SPECTRAL_COHESION": {
                "high": (
                    "🔗 Alta Cohesión del Equipo (Eigenvalor Fiedler={fiedler:.2f}): "
                    "La estructura de costos está fuertemente sincronizada."
                ),
                "standard": (
                    "⚖️ Cohesión Estándar (Eigenvalor Fiedler={fiedler:.2f}): "
                    "El proyecto presenta un acoplamiento típico entre sus componentes."
                ),
                "low": (
                    "💔 Fractura Organizacional (Eigenvalor Fiedler={fiedler:.3f}): "
                    "Baja cohesión espectral. Los subsistemas operan aislados, "
                    "riesgo de desalineación en ejecución."
                ),
            },
            
            "SPECTRAL_RESONANCE": {
                "risk": (
                    "🔊 RIESGO DE RESONANCIA FINANCIERA (λ={wavelength:.2f}): "
                    "El espectro de vibración está peligrosamente concentrado. "
                    "Un impacto externo (inflación/escasez) podría amplificarse en toda "
                    "la estructura simultáneamente."
                ),
                "safe": (
                    "🌊 Disipación Ondulatoria (λ={wavelength:.2f}): "
                    "La estructura tiene capacidad para amortiguar impactos locales sin entrar "
                    "en resonancia sistémica."
                ),
            },
            
            "THERMAL_TEMPERATURE": {
                "cold": (
                    "❄️ Temperatura Estable ({temperature:.1f}°C): "
                    "El proyecto está termodinámicamente equilibrado (Precios fríos/fijos)."
                ),
                "stable": (
                    "🌡️ Temperatura Normal ({temperature:.1f}°C): "
                    "Condiciones térmicas estándar del mercado."
                ),
                "warm": (
                    "🌡️ Calentamiento Operativo ({temperature:.1f}°C): "
                    "Existe una exposición moderada a la volatilidad de precios."
                ),
                "hot": (
                    "🔥 EL PROYECTO TIENE FIEBRE ({temperature:.1f}°C): "
                    "El Índice de Inflación Interna es crítico. Los costos de insumos volátiles "
                    "están sobrecalentando la estructura de precios."
                ),
                "critical": (
                    "☢️ FUSIÓN TÉRMICA ({temperature:.1f}°C): "
                    "Temperatura crítica alcanzada. Los costos están en espiral inflacionaria. "
                    "Riesgo de colapso financiero por sobrecalentamiento incontrolado."
                ),
            },
            
            "THERMAL_ENTROPY": {
                "low": (
                    "📋 Orden Administrativo (S={entropy:.2f}): "
                    "Baja entropía indica procesos bien estructurados y datos limpios."
                ),
                "high": (
                    "🌪️ Alta Entropía (S={entropy:.2f}): Caos administrativo detectado. "
                    "La energía del dinero se disipa en fricción operativa "
                    "(datos sucios o desorganizados)."
                ),
            },
            
            "GYROSCOPIC_STABILITY": {
                "stable": "✅ Giroscopio Estable: Flujo con momento angular constante.",
                "precession": "⚠️ Precesión Detectada: Oscilación lateral en el flujo de datos.",
                "nutation": (
                    "🚨 NUTACIÓN CRÍTICA: Inestabilidad rotacional. El proceso corre riesgo "
                    "de colapso inercial."
                ),
            },
            
            "LAPLACE_CONTROL": {
                "robust": "🛡️ Control Robusto: Margen de fase sólido (>45°).",
                "marginal": "⚠️ Estabilidad Marginal: Respuesta oscilatoria ante transitorios.",
                "unstable": "⛔ DIVERGENCIA MATEMÁTICA: Polos en el semiplano derecho (RHP).",
            },
            
            "PUMP_DYNAMICS": {
                "efficiency_high": (
                    "Eficiencia de Inyección: ALTA.\n"
                    "El costo administrativo de procesar esta información es "
                    "{joules_per_record:.2e} Joules por registro."
                ),
                "efficiency_low": (
                    "Eficiencia de Inyección: BAJA.\n"
                    "El costo administrativo de procesar esta información es "
                    "{joules_per_record:.2e} Joules por registro."
                ),
                "water_hammer": (
                    "💥 Inestabilidad de Tubería: Se detectaron golpes de ariete "
                    "(Presión={pressure:.2f}). "
                    "El flujo se detiene bruscamente, causando ondas de choque."
                ),
                "accumulator_pressure": (
                    "🔋 Presión del Acumulador: {pressure:.1f}%. "
                    "Capacidad de amortiguamiento disponible."
                ),
            },
            
            "FINANCIAL_VERDICT": {
                "accept": "🚀 Veredicto: VIABLE (IR={pi:.2f}). Estructura financiable.",
                "conditional": "🔵 Veredicto: CONDICIONAL (IR={pi:.2f}). Viable con ajustes.",
                "review": "🔍 Veredicto: REVISIÓN REQUERIDA.",
                "reject": "🛑 Veredicto: RIESGO CRÍTICO (IR={pi:.2f}). No procedente.",
            },
            
            "FINAL_VERDICTS": {
                "synergy_risk": (
                    "🛑 PARADA DE EMERGENCIA (Efecto Dominó): Se detectaron ciclos interconectados "
                    "que comparten recursos críticos. El riesgo no es aditivo, es multiplicativo. "
                    "Cualquier fallo en el suministro provocará un colapso sistémico en múltiples "
                    "frentes. Desacoplar los ciclos antes de continuar."
                ),
                "inverted_pyramid_viable": (
                    "⚠️ PRECAUCIÓN LOGÍSTICA (Estructura Inestable): Aunque los números "
                    "financieros cuadran, el proyecto es una Pirámide Invertida (Ψ={stability:.2f}). "
                    "Se sostiene sobre una base de recursos demasiado estrecha. "
                    "RECOMENDACIÓN: Ampliar la base de proveedores antes de construir."
                ),
                "inverted_pyramid_reject": (
                    "❌ PROYECTO INVIABLE (Riesgo de Colapso): Combinación letal de inestabilidad "
                    "estructural (Pirámide Invertida) e inviabilidad financiera. "
                    "No proceder bajo ninguna circunstancia sin rediseño total."
                ),
                "has_holes": (
                    "🛑 DETENER PARA REPARACIONES: Se detectaron {beta_1} socavones lógicos "
                    "(ciclos). No se puede verter dinero en una estructura con agujeros. "
                    "Sanear la topología antes de aprobar presupuesto."
                ),
                "certified": (
                    "✅ CERTIFICADO DE SOLIDEZ: Estructura piramidal estable, sin socavones "
                    "lógicos y financieramente viable. Proceder a fase de ejecución."
                ),
                "review_required": (
                    "🔍 REVISIÓN TÉCNICA REQUERIDA: La estructura es sólida pero los números "
                    "no convencen."
                ),
                "analysis_failed": (
                    "⚠️ ANÁLISIS ESTRUCTURAL INTERRUMPIDO: Se detectaron inconsistencias "
                    "matemáticas o falta de datos críticos que impiden certificar la solidez "
                    "del proyecto. Revise los errores en las secciones técnicas."
                ),
            },
            
            "MISC": {
                "MAYER_VIETORIS": (
                    "🧩 Incoherencia de Integración: La fusión de los presupuestos ha generado "
                    "{delta_beta_1} ciclos lógicos fantasmas (Anomalía de Mayer-Vietoris). "
                    "Los datos individuales son válidos, pero su unión crea una contradicción "
                    "topológica."
                ),
                "THERMAL_DEATH": (
                    "☢️ MUERTE TÉRMICA DEL SISTEMA: La entropía ha alcanzado el equilibrio "
                    "máximo. No hay energía libre para procesar información útil."
                ),
                "SYNERGY": (
                    "🔥 Riesgo de Contagio (Efecto Dominó): Se detectó una 'Sinergia de Riesgo' "
                    "en {count} puntos de intersección crítica. Los errores no son aislados; "
                    "si uno falla, provocará una reacción en cadena a través de los frentes de "
                    "obra compartidos."
                ),
                "EULER_EFFICIENCY": (
                    "🕸️ Sobrecarga de Gestión (Entropía): La eficiencia de Euler es baja "
                    "({efficiency:.2f}). Existe una complejidad innecesaria de enlaces que "
                    "dificulta la supervisión y aumenta los costos indirectos de administración."
                ),
                "CYCLE_PATH": (
                    "🔄 Ruta del Ciclo Detectada: La circularidad sigue el camino: [{path}]. "
                    "Esto significa que el costo de '{first_node}' depende indirectamente de "
                    "sí mismo, creando una indeterminación matemática en la valoración."
                ),
                "STRESS_POINT": (
                    "⚡ Punto de Estrés Estructural: El elemento '{node}' actúa como una "
                    "'Piedra Angular' crítica, soportando {degree} conexiones directas. "
                    "Una variación en su precio o disponibilidad impactará desproporcionadamente "
                    "a toda la estructura del proyecto (Punto Único de Falla)."
                ),
                "WACC": "💰 Costo de Oportunidad: WACC = {wacc:.2%}.",
                "CONTINGENCY": "📊 Blindaje Financiero: Contingencia sugerida de ${contingency:,.2f}.",
            },
            
            # ========== TELEMETRY SUCCESS/WARNING/FAILURE ==========
            "TELEMETRY_SUCCESS": {
                "PHYSICS": (
                    "✅ **Cimentación Estable**:\n"
                    "Flujo laminar de datos confirmado. Sin turbulencia (Flyback).\n"
                    "La base física del proyecto es sólida."
                ),
                "TACTICS": (
                    "✅ **Estructura Coherente**:\n"
                    "Topología conexa (β₀=1) y acíclica (β₁=0).\n"
                    "El grafo de dependencias es válido."
                ),
                "STRATEGY": (
                    "✅ **Viabilidad Confirmada**:\n"
                    "El modelo financiero es robusto ante la volatilidad.\n"
                    "Los indicadores de riesgo están dentro de umbrales aceptables."
                ),
                "WISDOM": (
                    "✅ **Síntesis Completa**:\n"
                    "Respuesta generada exitosamente.\n"
                    "Todas las capas del análisis convergen."
                ),
            },
            
            "TELEMETRY_WARNINGS": {
                "PHYSICS": (
                    "⚠️ **Señales de Turbulencia**:\n"
                    "Se detectaron fluctuaciones en el flujo de datos.\n"
                    "Monitorear la situación."
                ),
                "TACTICS": (
                    "⚠️ **Estructura Subóptima**:\n"
                    "El grafo presenta redundancias o complejidad excesiva.\n"
                    "Considerar simplificación."
                ),
                "STRATEGY": (
                    "⚠️ **Sensibilidad Alta**:\n"
                    "El modelo financiero es sensible a variaciones.\n"
                    "Realizar análisis de escenarios."
                ),
                "WISDOM": (
                    "⚠️ **Síntesis Parcial**:\n"
                    "La respuesta se generó con algunas limitaciones.\n"
                    "Revisar calidad de inputs."
                ),
            },
            
            "TELEMETRY_FAILURES_PHYSICS": {
                "default": (
                    "🔥 **Falla en Cimentación**:\n"
                    "Se detectó inestabilidad física (Saturación/Flyback).\n"
                    "Los datos no son confiables."
                ),
                "saturation": (
                    "⚡ **Sobrecarga Detectada**:\n"
                    "El sistema alcanzó saturación crítica.\n"
                    "Reducir carga o escalar recursos."
                ),
                "corruption": (
                    "💥 **Datos Corruptos**:\n"
                    "La integridad de los datos de entrada está comprometida.\n"
                    "Verificar fuentes."
                ),
                "nutation": (
                    "🚨 **NUTACIÓN CRÍTICA**:\n"
                    "Inestabilidad rotacional detectada. El proceso corre riesgo de "
                    "colapso inercial por oscilaciones no amortiguadas."
                ),
                "thermal_death": (
                    "☢️ **MUERTE TÉRMICA DEL SISTEMA**:\n"
                    "La entropía ha alcanzado el equilibrio máximo.\n"
                    "No hay energía libre para procesar información útil."
                ),
                "laplace_unstable": (
                    "⛔ **DIVERGENCIA MATEMÁTICA**:\n"
                    "Polos en el semiplano derecho (RHP). El sistema es intrínsecamente "
                    "explosivo ante variaciones de entrada."
                ),
                "water_hammer": (
                    "🌊 **GOLPE DE ARIETE DETECTADO**:\n"
                    "Ondas de choque en la tubería de datos (Presión > 0.7).\n"
                    "Riesgo de ruptura en la persistencia."
                ),
                "high_injection_work": (
                    "💪 **Fase de Ingesta (Sobrecarga)**:\n"
                    "Alto esfuerzo de inyección detectado. La fricción de los datos "
                    "está consumiendo energía crítica."
                ),
            },
            
            "TELEMETRY_FAILURES_TACTICS": {
                "default": (
                    "🏗️ **Fragmentación Estructural**:\n"
                    "El grafo del proyecto está desconectado.\n"
                    "Existen islas de datos sin conexión."
                ),
                "cycles": (
                    "🔄 **Socavón Lógico Detectado**:\n"
                    "La estructura contiene bucles infinitos (β₁ > 0).\n"
                    "El costo es incalculable."
                ),
                "disconnected": (
                    "🧩 **Componentes Aislados**:\n"
                    "β₀ > 1 indica múltiples componentes desconectados.\n"
                    "Revisar enlaces entre módulos."
                ),
                "mayer_vietoris": (
                    "🧩 **ANOMALÍA DE INTEGRACIÓN (Mayer-Vietoris)**:\n"
                    "La fusión de datasets ha generado ciclos lógicos que no existían "
                    "en las fuentes originales. Inconsistencia topológica."
                ),
            },
            
            "TELEMETRY_FAILURES_STRATEGY": {
                "default": (
                    "📉 **Riesgo Sistémico**:\n"
                    "Aunque la estructura es válida,\n"
                    "la simulación financiera proyecta pérdidas."
                ),
                "high_var": (
                    "🎲 **Alta Volatilidad**:\n"
                    "El VaR excede umbrales aceptables.\n"
                    "Considerar coberturas o reducir exposición."
                ),
                "negative_npv": (
                    "💸 **Destrucción de Valor**:\n"
                    "El NPV proyectado es negativo.\n"
                    "El proyecto no genera valor económico."
                ),
            },
            
            "TELEMETRY_FAILURES_WISDOM": {
                "default": (
                    "⚠️ **Síntesis Comprometida**:\n"
                    "Hubo problemas generando la respuesta final.\n"
                    "Revisar pasos anteriores."
                ),
            },
            
            "TELEMETRY_VERDICTS": {
                "APPROVED": (
                    "🏛️ **CERTIFICADO DE SOLIDEZ INTEGRAL**\n"
                    "El Consejo valida el proyecto en todas sus dimensiones:\n"
                    "Físicamente estable, Topológicamente conexo y Financieramente viable."
                ),
                "REJECTED_PHYSICS": (
                    "⛔ **PROCESO ABORTADO POR INESTABILIDAD FÍSICA**\n"
                    "El Guardián detectó que el flujo de datos es turbulento o corrupto.\n"
                    "No tiene sentido analizar la estrategia financiera de datos que "
                    "no existen físicamente."
                ),
                "REJECTED_TACTICS": (
                    "🚧 **VETO ESTRUCTURAL DEL ARQUITECTO**\n"
                    "Los datos son legibles, pero forman una estructura imposible.\n"
                    "Cualquier cálculo financiero sobre esta base sería una alucinación."
                ),
                "REJECTED_STRATEGY": (
                    "📉 **ALERTA FINANCIERA DEL ORÁCULO**\n"
                    "La estructura es sólida, pero el mercado es hostil o el proyecto "
                    "no es rentable."
                ),
                "REJECTED_WISDOM": (
                    "⚠️ **FALLO EN SÍNTESIS FINAL**\n"
                    "Todas las capas base son válidas, pero hubo un error generando "
                    "la respuesta."
                ),
            },
        }
        
        return templates
    
    @property
    def projector(self) -> GraphSemanticProjector:
        """Obtiene o crea el proyector semántico (lazy initialization)."""
        if self._projector is None:
            with self._lock:
                # Double-check locking
                if self._projector is None:
                    self._projector = GraphSemanticProjector(self)
        return self._projector
    
    def fetch_narrative(
        self,
        domain: str,
        classification: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Construye la narrativa basada en el dominio y la clasificación.
        
        Args:
            domain: Grupo temático de la plantilla (TOPOLOGY, STABILITY, etc.)
            classification: Estado específico dentro del dominio
            params: Variables de sustitución para formateo
            **kwargs: Argumentos adicionales (merged con params)
            
        Returns:
            Diccionario con resultado de operación y narrativa generada
        """
        # Merge params y kwargs
        effective_params = {**(params or {}), **kwargs}
        
        # Manejo especial para MARKET_CONTEXT
        if domain == "MARKET_CONTEXT":
            return self._handle_market_context(effective_params)
        
        # Obtener grupo de plantillas
        template_group = self._templates.get(domain)
        if template_group is None:
            logger.warning(f"Domain '{domain}' no encontrado en plantillas")
            return {
                "success": False,
                "error": f"Domain '{domain}' not found",
                "available_domains": sorted(self._templates.keys()),
            }
        
        try:
            narrative = self._resolve_template(
                template_group, 
                classification, 
                effective_params
            )
            
            return {
                "success": True,
                "narrative": narrative,
                "stratum": Stratum.WISDOM.name,
                "domain": domain,
                "classification": classification,
            }
            
        except KeyError as e:
            logger.error(
                f"Placeholder faltante en {domain}.{classification}: {e}"
            )
            return {
                "success": False,
                "error": f"Missing required parameter: {e}",
                "domain": domain,
                "classification": classification,
                "provided_params": list(effective_params.keys()),
            }
        except Exception as e:
            logger.exception(
                f"Error generando narrativa para {domain}.{classification}"
            )
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
    
    def _handle_market_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja solicitudes de contexto de mercado."""
        deterministic = params.get("deterministic", False)
        
        if deterministic:
            index = params.get("index", 0)
            narrative = self._market_contexts[index % len(self._market_contexts)]
        else:
            narrative = random.choice(self._market_contexts)
        
        return {
            "success": True,
            "narrative": narrative,
            "stratum": Stratum.WISDOM.name,
            "domain": "MARKET_CONTEXT",
            "total_contexts": len(self._market_contexts),
        }
    
    def _resolve_template(
        self,
        template_group: Union[str, Dict[str, str]],
        classification: Optional[str],
        params: Dict[str, Any]
    ) -> str:
        """
        Resuelve una plantilla a su texto final.
        
        Args:
            template_group: Puede ser string directo o dict de clasificaciones
            classification: Clave de clasificación si template_group es dict
            params: Parámetros de sustitución
            
        Returns:
            Texto narrativo formateado
        """
        def _robust_float_cast(value: Any) -> Any:
            """Saneamiento estricto del tensor de tipos para formatos 'f'."""
            if isinstance(value, float):
                return value
            if isinstance(value, int):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return value
            return value

        safe_params = {
            k: _robust_float_cast(v)
            for k, v in params.items()
        }

        if isinstance(template_group, str):
            return template_group.format(**safe_params)
        
        if isinstance(template_group, dict):
            default_msg = "⚠️ Estado desconocido. Clasificación no encontrada."
            template = template_group.get(classification, default_msg)
            if classification is not None and template == default_msg:
                # Normalización en mayúsculas estricta
                template = template_group.get(str(classification).upper(), default_msg)
            return template.format(**safe_params)
        
        # Fallback para otros tipos
        return str(template_group)
    
    def get_classification_by_threshold(
        self,
        metric_name: str,
        value: float,
        inverse: bool = False
    ) -> str:
        """
        Determina clasificación basada en umbrales predefinidos.
        
        Args:
            metric_name: Nombre del umbral (STABILITY, ENTROPY, COHESION, TEMPERATURE)
            value: Valor medido
            inverse: Si True, invierte la lógica (menor valor = mejor clasificación)
            
        Returns:
            Clasificación correspondiente
        """
        threshold_map = {
            "STABILITY": (self.STABILITY_THRESHOLDS, False),
            "ENTROPY": (self.ENTROPY_THRESHOLDS, True),  # Mayor entropía = peor
            "COHESION": (self.COHESION_THRESHOLDS, False),
            "TEMPERATURE": (self.TEMPERATURE_THRESHOLDS, True),  # Mayor temp = peor
        }
        
        config = threshold_map.get(metric_name.upper())
        if config is None:
            raise ValueError(
                f"Métrica '{metric_name}' no tiene umbrales definidos. "
                f"Disponibles: {list(threshold_map.keys())}"
            )
        
        thresholds, default_inverse = config
        should_inverse = inverse or default_inverse
        
        # Ordenar umbrales
        sorted_items = sorted(
            thresholds.items(),
            key=lambda x: x[1],
            reverse=not should_inverse
        )
        
        for classification, threshold in sorted_items:
            if should_inverse:
                if value >= threshold:
                    return classification
            else:
                if value >= threshold:
                    return classification
        
        # Retornar la clasificación con menor umbral
        return min(thresholds.keys(), key=lambda k: thresholds[k])
    
    def convert_stratum_value(self, value: Union[int, str, Stratum]) -> Stratum:
        """
        Convierte cualquier representación válida de Stratum al Enum.
        
        Args:
            value: Puede ser int, string, o ya un Stratum
            
        Returns:
            Instancia Stratum válida
            
        Raises:
            ValueError: Si el valor no corresponde a ningún Stratum válido
            TypeError: Si el tipo no es soportado
        """
        if isinstance(value, Stratum):
            return value
        
        if isinstance(value, int):
            try:
                return Stratum(value)
            except ValueError as e:
                valid_values = [s.value for s in Stratum]
                raise ValueError(
                    f"Valor entero {value} no es un Stratum válido. "
                    f"Valores válidos: {valid_values}"
                ) from e
        
        if isinstance(value, str):
            normalized = value.upper().strip()
            try:
                return Stratum[normalized]
            except KeyError as e:
                valid_names = [s.name for s in Stratum]
                raise ValueError(
                    f"'{value}' no es un nombre de Stratum válido. "
                    f"Nombres válidos: {valid_names}"
                ) from e
        
        raise TypeError(
            f"Tipo no soportado para conversión de Stratum: {type(value).__name__}. "
            f"Se esperaba int, str o Stratum."
        )
    
    def project_graph_narrative(
        self, 
        payload: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Proyecta una anomalía del grafo (ciclos o estrés) a una narrativa.
        
        Esta función actúa como adapter entre el motor topológico y el
        lenguaje natural.
        
        Args:
            payload: Datos crudos de la anomalía detectada
            context: Información adicional del entorno de ejecución
            
        Returns:
            Narrativa estructurada sobre la anomalía
        """
        context = context or {}
        anomaly_type = payload.get("anomaly_type", "").upper()
        
        handlers = {
            "CYCLE": self._project_cycle_anomaly,
            "STRESS": self._project_stress_anomaly,
            "FRAGMENTATION": self._project_fragmentation_anomaly,
        }
        
        handler = handlers.get(anomaly_type)
        if handler is None:
            return {
                "success": False,
                "error": f"Tipo de anomalía '{anomaly_type}' no soportada.",
                "supported_types": list(handlers.keys()),
            }
        
        try:
            return handler(payload, context)
        except Exception as e:
            logger.exception(f"Error proyectando anomalía {anomaly_type}")
            return {
                "success": False,
                "error": str(e),
                "anomaly_type": anomaly_type,
                "traceback": traceback.format_exc(),
            }
    
    def _project_cycle_anomaly(
        self, 
        payload: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Proyecta anomalía de ciclo."""
        path_nodes = payload.get("path_nodes", [])
        
        if not isinstance(path_nodes, (list, tuple)):
            path_nodes = [path_nodes] if path_nodes else []
        
        return self.projector.project_cycle_path(list(path_nodes))
    
    def _project_stress_anomaly(
        self, 
        payload: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Proyecta anomalía de estrés estructural."""
        vector_data = payload.get("vector", {})
        
        if not isinstance(vector_data, dict):
            return {
                "success": False,
                "error": "Campo 'vector' debe ser un diccionario.",
                "received_type": type(vector_data).__name__,
            }
        
        # Conversión robusta de stratum
        if "stratum" in vector_data:
            try:
                vector_data["stratum"] = self.convert_stratum_value(
                    vector_data["stratum"]
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Conversión de Stratum fallida: {e}")
                vector_data["stratum"] = Stratum.TACTICS
        else:
            vector_data["stratum"] = Stratum.TACTICS
        
        # Validar campos requeridos
        required_fields = {"node_id", "node_type", "stratum", "in_degree", "out_degree"}
        provided_fields = set(vector_data.keys())
        missing = required_fields - provided_fields
        
        if missing:
            return {
                "success": False,
                "error": f"Faltan campos requeridos: {sorted(missing)}",
                "received_fields": sorted(provided_fields),
                "required_fields": sorted(required_fields),
            }
        
        # Validar node_type
        if vector_data["node_type"] not in VALID_NODE_TYPES:
            return {
                "success": False,
                "error": f"node_type inválido: '{vector_data['node_type']}'",
                "valid_types": sorted(VALID_NODE_TYPES),
            }
        
        try:
            vector = PyramidalSemanticVector(**vector_data)
            return self.projector.project_pyramidal_stress(vector)
        except (ValueError, TypeError) as e:
            return {
                "success": False,
                "error": f"Error construyendo vector: {str(e)}",
                "vector_data": vector_data,
            }
    
    def _project_fragmentation_anomaly(
        self, 
        payload: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Proyecta anomalía de fragmentación."""
        beta_0 = payload.get("beta_0", 0)
        component_sizes = payload.get("component_sizes")
        
        if not isinstance(beta_0, int) or beta_0 < 0:
            return {
                "success": False,
                "error": "beta_0 debe ser un entero no negativo.",
                "received": beta_0,
            }
        
        return self.projector.project_fragmentation(beta_0, component_sizes)
    
    def register_in_mic(self, mic_registry: Any) -> bool:
        """
        Registra el diccionario en la MIC (Microservice Interface Controller).
        
        Expone vectores de servicio para consultas de plantillas y
        proyección de anomalías topológicas.
        
        Args:
            mic_registry: Instancia del registro de servicios
            
        Returns:
            True si el registro fue exitoso, False en caso contrario
        """
        try:
            from app.adapters.tools_interface import MICRegistry
        except ImportError:
            logger.warning(
                "MICRegistry no disponible. "
                "El servicio funcionará en modo standalone."
            )
            return False
        
        if not isinstance(mic_registry, MICRegistry):
            logger.error(
                f"Tipo de registro inválido: {type(mic_registry).__name__}. "
                f"Se esperaba MICRegistry."
            )
            return False
        
        try:
            mic_registry.register_vector(
                service_name="fetch_narrative",
                stratum=Stratum.WISDOM,
                handler=self.fetch_narrative,
            )
            
            mic_registry.register_vector(
                service_name="project_graph_narrative",
                stratum=Stratum.WISDOM,
                handler=self.project_graph_narrative,
            )
            
            logger.info("✅ Vectores Semánticos registrados en la MIC")
            return True
            
        except Exception as e:
            logger.exception(f"Error registrando vectores en MIC: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Endpoint de salud para monitoreo.
        
        Returns:
            Diccionario con estado del servicio y métricas básicas
        """
        projector_stats = {}
        if self._projector is not None:
            projector_stats = self._projector.cache_stats
        
        return {
            "status": "healthy",
            "service": "SemanticDictionaryService",
            "stratum": Stratum.WISDOM.name,
            "template_domains": len(self._templates),
            "market_contexts_count": len(self._market_contexts),
            "strata_available": [s.name for s in Stratum],
            "thresholds": {
                "stability": self.STABILITY_THRESHOLDS,
                "entropy": self.ENTROPY_THRESHOLDS,
                "cohesion": self.COHESION_THRESHOLDS,
            },
            "projector_cache": projector_stats,
            "timestamp": time.time(),
        }
    
    def get_available_domains(self) -> List[str]:
        """Retorna lista de dominios de plantillas disponibles."""
        return sorted(self._templates.keys())
    
    def get_domain_classifications(self, domain: str) -> Optional[List[str]]:
        """
        Retorna las clasificaciones disponibles para un dominio.
        
        Args:
            domain: Nombre del dominio
            
        Returns:
            Lista de clasificaciones o None si el dominio no existe
        """
        template_group = self._templates.get(domain)
        
        if template_group is None:
            return None
        
        if isinstance(template_group, dict):
            return sorted(template_group.keys())
        
        return []


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_semantic_dictionary_service() -> SemanticDictionaryService:
    """
    Factory function para crear instancias del servicio.
    
    Permite inyección de dependencias y configuración centralizada.
    
    Returns:
        Instancia configurada de SemanticDictionaryService
    """
    service = SemanticDictionaryService()
    logger.info(f"Servicio creado con {len(service.get_available_domains())} dominios")
    return service